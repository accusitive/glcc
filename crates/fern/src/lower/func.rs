//! Function bodies: slots, emission helpers, and pattern binding.
//!
//! Every HIR local gets a slot in the entry block, and scalar bindings
//! move in and out through `Load`/`Store`; block-join values are plain
//! block parameters. Pointer-typed locals also get a GC root slot and are
//! pinned there on every write (see [`super::gc`] for the frame
//! discipline).

use std::collections::HashMap;

use glcc_mir as mir;

use crate::hir::*;
use crate::span::{Diag, KResult, Span};
use crate::ty::Ty;

use super::Lowerer;

pub(super) struct FnCtx<'l, 'd> {
    pub(super) lw: &'l mut Lowerer<'d>,
    pub(super) func: &'l mut mir::Function,
    pub(super) cur: mir::Block,
    /// Set once the current block has a terminator; later code in the
    /// same HIR position is unreachable and skipped.
    pub(super) terminated: bool,
    /// This instance's symbol, for naming the closures it declares.
    pub(super) name: String,
    /// Concrete types for the body's generics.
    pub(super) sub: HashMap<String, Ty>,
    pub(super) slots: Vec<mir::Value>,
    pub(super) local_tys: Vec<mir::Ty>,
    /// This body's GC frame cookie, once the prologue has run.
    pub(super) frame: Option<mir::Value>,
    /// The root slot per local, for locals that hold pointers.
    pub(super) local_slots: Vec<Option<u32>>,
    /// Root slots reserved for locals; temporaries sit above them.
    pub(super) local_roots: u32,
    /// Temporaries currently held in root slots.
    pub(super) temps: u32,
}

/// Entry-block parameters for a body: one MIR parameter per HIR
/// parameter pattern at the pattern's type, preceded by the closure
/// environment when `env` is set.
fn entry_params<'d>(lw: &mut Lowerer<'d>, func: &mut mir::Function, params: &[HirPat], subst: &HashMap<String, Ty>, env: bool) -> KResult<Vec<(mir::Ty, mir::Value)>> {
    let entry = func.entry;
    let base = if env {
        let value = func.values.push(mir::ValueData {
            ty: mir::Ty::ptr(mir::Ty::Void),
            def: mir::Def::Param(entry, 0),
        });
        func.blocks[entry].params.push(value);
        1
    } else {
        0
    };
    let mut out = Vec::with_capacity(params.len());
    for (i, pat) in params.iter().enumerate() {
        let ty = lw.types.get(&pat.ty.substitute(subst), pat.span)?;
        let value = func.values.push(mir::ValueData {
            ty: ty.clone(),
            def: mir::Def::Param(entry, (i + base) as u32),
        });
        func.blocks[entry].params.push(value);
        out.push((ty, value));
    }
    Ok(out)
}

pub(super) fn lower_fn<'d>(lw: &mut Lowerer<'d>, func: &mut mir::Function, hir: &'d HirFn, subst: &HashMap<String, Ty>, span: Span, name: String) -> KResult<()> {
    let params = entry_params(lw, func, &hir.params, subst, false)?;
    let _ = lw.types.get(&hir.ret.substitute(subst), span)?;
    let mut cx = FnCtx::new(lw, func, name, subst);
    cx.open_frame()?;
    for local in &hir.locals {
        let ty = cx.ty_of(&local.ty, local.span)?;
        cx.declare_local(ty)?;
    }
    for (pat, (_, value)) in hir.params.iter().zip(params) {
        let dead = cx.dead_block();
        cx.bind_pattern(pat, value, dead)?;
    }
    let value = cx.expr(&hir.body)?;
    if !cx.terminated {
        cx.close_frame()?;
        cx.terminate(mir::Terminator::Return { val: Some(value) });
    }
    Ok(())
}

pub(super) fn lower_closure<'d>(lw: &mut Lowerer<'d>, func: &mut mir::Function, closure: &'d HirClosure, subst: &HashMap<String, Ty>, span: Span, name: String) -> KResult<()> {
    let params = entry_params(lw, func, &closure.params, subst, true)?;
    let _ = lw.types.get(&closure.ret.substitute(subst), span)?;
    let env = func.blocks[func.entry].params[0];
    let mut cx = FnCtx::new(lw, func, name, subst);
    cx.open_frame()?;
    for local in &closure.locals {
        let ty = cx.ty_of(&local.ty, local.span)?;
        cx.declare_local(ty)?;
    }
    let cap_tys: Vec<Ty> = closure.captures.iter().map(|c| c.ty.substitute(&cx.sub)).collect();
    let env_layout = cx.lw.types.env_layout(&cap_tys, span)?;
    for (i, cap) in closure.captures.iter().enumerate() {
        let ty = cx.lw.types.get(&cap_tys[i], span)?;
        let addr = cx.offset_addr(env, env_layout.offsets[i])?;
        let value = cx.load(addr, &ty)?;
        cx.write_local(cap.inner, value)?;
    }
    for (pat, (_, value)) in closure.params.iter().zip(params) {
        let dead = cx.dead_block();
        cx.bind_pattern(pat, value, dead)?;
    }
    let value = cx.expr(&closure.body)?;
    if !cx.terminated {
        cx.close_frame()?;
        cx.terminate(mir::Terminator::Return { val: Some(value) });
    }
    Ok(())
}

impl<'l, 'd> FnCtx<'l, 'd> {
    fn new(lw: &'l mut Lowerer<'d>, func: &'l mut mir::Function, name: String, subst: &HashMap<String, Ty>) -> FnCtx<'l, 'd> {
        let cur = func.entry;
        FnCtx {
            lw,
            func,
            cur,
            terminated: false,
            name,
            sub: subst.clone(),
            slots: Vec::new(),
            local_tys: Vec::new(),
            frame: None,
            local_slots: Vec::new(),
            local_roots: 0,
            temps: 0,
        }
    }

    /// Declare a local: a stack slot to live in, plus a root slot when
    /// its value is a pointer.
    pub(super) fn declare_local(&mut self, ty: mir::Ty) -> KResult<()> {
        let slot = self.alloca(&ty)?;
        let root = self.reserve_local_root(&ty);
        self.slots.push(slot);
        self.local_tys.push(ty);
        self.local_slots.push(root);
        Ok(())
    }

    /// Lower a fern type in instance context (generic arguments filled in).
    pub(super) fn ty_of(&mut self, ty: &Ty, span: Span) -> KResult<mir::Ty> {
        let ty = ty.substitute(&self.sub);
        self.lw.types.get(&ty, span)
    }

    pub(super) fn emit(&mut self, kind: mir::InstKind, ty: mir::Ty) -> KResult<mir::Value> {
        if self.terminated {
            return Err(Diag::new("code generated after a diverging expression", Span::new(0, 0)));
        }
        let block = self.cur;
        let result = self.func.values.push(mir::ValueData {
            ty,
            def: mir::Def::Inst(mir::Inst::from_usize(self.func.insts.len())),
        });
        let inst = self.func.insts.push(mir::InstData {
            block,
            kind,
            result: Some(result),
            loc: None,
        });
        self.func.blocks[block].insts.push(inst);
        Ok(result)
    }

    pub(super) fn emit_void(&mut self, kind: mir::InstKind) -> KResult<()> {
        if self.terminated {
            return Err(Diag::new("code generated after a diverging expression", Span::new(0, 0)));
        }
        let block = self.cur;
        let inst = self.func.insts.push(mir::InstData {
            block,
            kind,
            result: None,
            loc: None,
        });
        self.func.blocks[block].insts.push(inst);
        Ok(())
    }

    pub(super) fn iconst(&mut self, bits: u32, value: i128) -> mir::Value {
        self.func.push_const(mir::Const::Int { bits, value })
    }

    pub(super) fn fconst(&mut self, value: f64) -> mir::Value {
        self.func.push_const(mir::Const::Float {
            kind: mir::FloatKind::F64,
            bits: value.to_bits() as u128,
        })
    }

    pub(super) fn terminate(&mut self, term: mir::Terminator) {
        self.func.blocks[self.cur].term = Some(term);
        self.terminated = true;
    }

    pub(super) fn new_block(&mut self, params: Vec<mir::Ty>) -> mir::Block {
        self.func.push_block(params)
    }

    pub(super) fn goto(&mut self, block: mir::Block) {
        self.cur = block;
        self.terminated = false;
    }

    pub(super) fn branch(&mut self, dest: mir::Block, args: Vec<mir::Value>) {
        self.terminate(mir::Terminator::Br { dest, args: args.into() });
    }

    /// Branch to a fresh continuation when `cond` holds, else to `fail`.
    pub(super) fn branch_on(&mut self, cond: mir::Value, fail: mir::Block) -> KResult<()> {
        let ok = self.new_block(Vec::new());
        self.terminate(mir::Terminator::CondBr {
            cond,
            then_block: ok,
            else_block: fail,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        self.goto(ok);
        Ok(())
    }

    /// An empty block terminated with `Unreachable`, for pattern
    /// mismatches that the type system does not rule out.
    pub(super) fn dead_block(&mut self) -> mir::Block {
        let dead = self.new_block(Vec::new());
        let (cur, terminated) = (self.cur, self.terminated);
        self.goto(dead);
        self.terminate(mir::Terminator::Unreachable);
        self.cur = cur;
        self.terminated = terminated;
        dead
    }

    pub(super) fn alloca(&mut self, pointee: &mir::Ty) -> KResult<mir::Value> {
        let value_ty = mir::Ty::ptr(pointee.clone());
        self.emit(
            mir::InstKind::Alloca {
                pointee: pointee.clone(),
                count: 1,
            },
            value_ty,
        )
    }

    pub(super) fn load(&mut self, addr: mir::Value, ty: &mir::Ty) -> KResult<mir::Value> {
        self.emit(
            mir::InstKind::Load {
                addr,
                flags: mir::MemFlags::plain(),
            },
            ty.clone(),
        )
    }

    pub(super) fn store(&mut self, addr: mir::Value, val: mir::Value) -> KResult<()> {
        self.emit_void(mir::InstKind::Store {
            addr,
            val,
            flags: mir::MemFlags::plain(),
        })
    }

    pub(super) fn offset_addr(&mut self, base: mir::Value, off: u64) -> KResult<mir::Value> {
        if off == 0 {
            return Ok(base);
        }
        let c = self.iconst(64, off as i128);
        self.emit(mir::InstKind::Add(base, c), mir::Ty::ptr(mir::Ty::Void))
    }

    pub(super) fn read_local(&mut self, id: usize) -> KResult<mir::Value> {
        let ty = self.local_tys[id].clone();
        self.load(self.slots[id], &ty)
    }

    /// Write a local's slot and, when it holds a pointer, mirror the
    /// value into the local's root slot.
    pub(super) fn write_local(&mut self, id: usize, val: mir::Value) -> KResult<()> {
        self.store(self.slots[id], val.clone())?;
        if let Some(slot) = self.local_slots[id] {
            self.pin(slot, val)?;
        }
        Ok(())
    }

    /// Bind `pat` against a value, branching to `fail` on any mismatch.
    /// The value is in local representation: a scalar, or an address for
    /// tuple storage, a data cell, or a closure cell.
    pub(super) fn bind_pattern(&mut self, pat: &HirPat, value: mir::Value, fail: mir::Block) -> KResult<()> {
        if self.terminated {
            return Ok(());
        }
        match &pat.kind {
            HirPatKind::Wild => Ok(()),
            HirPatKind::Bind(id) => self.write_local(*id, value),
            HirPatKind::Int(v) => {
                let want = self.iconst(64, *v as i128);
                let cond = self.emit(
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Eq,
                        lhs: value,
                        rhs: want,
                    },
                    mir::Ty::Bool,
                )?;
                self.branch_on(cond, fail)
            }
            HirPatKind::Bool(b) => {
                let want = self.iconst(8, i128::from(*b));
                let cond = self.emit(
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Eq,
                        lhs: value,
                        rhs: want,
                    },
                    mir::Ty::Bool,
                )?;
                self.branch_on(cond, fail)
            }
            HirPatKind::Tuple(fields) => {
                let layout = self.lw.types.tuple_layout(&pat.ty.substitute(&self.sub), pat.span)?;
                for (i, f) in fields.iter().enumerate() {
                    let addr = self.offset_addr(value.clone(), layout.offsets[i])?;
                    let v = self.load(addr, &layout.field_tys[i])?;
                    self.bind_pattern(f, v, fail)?;
                }
                Ok(())
            }
            HirPatKind::Ctor { ctor, fields, .. } => {
                let layout = self.lw.types.data_layout(&pat.ty.substitute(&self.sub), pat.span)?;
                let tag = self.load(value.clone(), &mir::Ty::int(32))?;
                let want = self.iconst(32, *ctor as i128);
                let cond = self.emit(
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Eq,
                        lhs: tag,
                        rhs: want,
                    },
                    mir::Ty::Bool,
                )?;
                self.branch_on(cond, fail)?;
                let base = self.offset_addr(value, layout.payload_offsets[*ctor])?;
                for (i, f) in fields.iter().enumerate() {
                    let fty = layout.field_tys[*ctor][i].clone();
                    let addr = self.offset_addr(base.clone(), layout.field_offsets[*ctor][i])?;
                    let v = self.load(addr, &fty)?;
                    self.bind_pattern(f, v, fail)?;
                }
                Ok(())
            }
        }
    }
}
