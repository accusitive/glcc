//! Expression lowering.

use glcc_mir as mir;

use crate::ast::{BinOp, UnOp};
use crate::defs::ExternId;
use crate::hir::*;
use crate::span::{Diag, KResult, Span};
use crate::ty::Ty;

use super::func::FnCtx;

impl<'l, 'd> FnCtx<'l, 'd> {
    pub(super) fn expr(&mut self, e: &HirExpr) -> KResult<mir::Value> {
        match &e.kind {
            HirExprKind::Const(c) => self.const_value(c),
            HirExprKind::Local(id) => self.read_local(*id),
            HirExprKind::Ctor { ctor, args, .. } => self.ctor(e, *ctor, args),
            HirExprKind::Fn(target) => self.fn_value(target, e),
            HirExprKind::Closure { id, captures } => self.closure_value(*id, captures, e.span),
            HirExprKind::CallDirect { target, args } => self.call_direct(target, args, e),
            HirExprKind::Call { callee, args } => self.call_indirect(callee, args, e),
            HirExprKind::Tuple(args) => self.tuple(e, args),
            HirExprKind::Let { binds, body } => self.let_expr(binds, body),
            HirExprKind::If { cond, then_body, else_body } => self.if_expr(cond, then_body, else_body, e),
            HirExprKind::Match { scrutinee, arms } => self.match_expr(e, scrutinee, arms),
            HirExprKind::Bin { op, lhs, rhs } => self.bin(*op, lhs, rhs, e),
            HirExprKind::Un { op, operand } => self.un(*op, operand, e),
        }
    }

    fn const_value(&mut self, c: &ConstValue) -> KResult<mir::Value> {
        Ok(match c {
            ConstValue::Int(v) => self.iconst(64, *v as i128),
            ConstValue::Float(v) => self.fconst(*v),
            ConstValue::Bool(b) => self.iconst(8, *b as i128),
            ConstValue::Str(s) => self.string_const(s)?,
        })
    }

    /// A string literal is a NUL-terminated byte array in a private
    /// global. The bytes are never mutated, so the address is a valid
    /// `String` for the program's lifetime.
    fn string_const(&mut self, s: &str) -> KResult<mir::Value> {
        let mut image = s.as_bytes().to_vec();
        image.push(0);
        let n = self.lw.str_count;
        self.lw.str_count += 1;
        let ty = mir::Ty::Array {
            elem: Box::new(mir::Ty::int(8)),
            len: image.len() as u64,
        };
        let global = self
            .lw
            .module
            .push_global(&format!("__fern_str.{n}"), ty, Some(mir::Const::Bytes(image)));
        let data = &mut self.lw.module.globals[global];
        data.local = true;
        data.align = Some(1);
        self.emit(
            mir::InstKind::GlobalAddr { global },
            mir::Ty::ptr(mir::Ty::int(8)),
        )
    }

    /// Build a data cell: the tag, then the constructor's payload. The
    /// cell is pinned while arguments are evaluated, since building them
    /// can allocate.
    fn ctor(&mut self, e: &HirExpr, ctor: usize, args: &[HirExpr]) -> KResult<mir::Value> {
        let src = e.ty.substitute(&self.sub);
        let layout = self.lw.types.data_layout(&src, e.span)?;
        let bytes = self.iconst(64, layout.cell_size as i128);
        let cell = self.call_alloc(bytes)?;
        let tag = self.iconst(32, ctor as i128);
        self.store(cell.clone(), tag)?;
        let mark = self.hold(cell.clone())?;
        let base = self.offset_addr(cell.clone(), layout.payload_offsets[ctor])?;
        for (i, a) in args.iter().enumerate() {
            let value = self.expr(a)?;
            let dst = self.offset_addr(base.clone(), layout.field_offsets[ctor][i])?;
            self.store(dst, value)?;
        }
        self.release(mark)?;
        Ok(cell)
    }

    /// A named function as a value: allocate a cell holding a thunk's
    /// code address; the environment slot stays empty. A top-level
    /// `let` body already evaluates to its closure cell, so it is
    /// simply called.
    fn fn_value(&mut self, target: &FnRef, e: &HirExpr) -> KResult<mir::Value> {
        let sub = self.sub.clone();
        if matches!(target, FnRef::Value { .. }) {
            let (callee, _) = self.lw.resolve_callee(target, &sub, e.span)?;
            let ret = self.ty_of(&e.ty, e.span)?;
            return self.emit(mir::InstKind::Call { callee, args: Vec::new().into() }, ret);
        }
        if let FnRef::Extern { id } = target {
            let (name, params, ret) = {
                let def = &self.lw.defs.externs[id.index()];
                (def.name.clone(), def.params.clone(), def.ret.clone())
            };
            if params.is_empty() {
                return Err(Diag::new(
                    format!("`{name}` has no parameters and cannot be used as a value"),
                    e.span,
                ));
            }
            let mut param_tys = Vec::with_capacity(params.len());
            for param in &params {
                param_tys.push(self.lw.types.get(param, e.span)?);
            }
            let ret_ty = self.lw.types.get(&ret, e.span)?;
            let imported = self.lw.externs[id];
            let thunk = self.lw.thunk_extern(imported, &name, &param_tys, &ret_ty)?;
            let cell = self.alloc_cell(8)?;
            let code = self.emit(mir::InstKind::FuncAddr { func: thunk }, mir::Ty::ptr(mir::Ty::Void))?;
            self.store(cell, code)?;
            return Ok(cell);
        }
        let (callee, name) = self.lw.resolve_callee(target, &sub, e.span)?;
        let src = e.ty.substitute(&sub);
        let (params, ret) = split_arrow(&src);
        let mut param_tys = Vec::with_capacity(params.len());
        for p in &params {
            param_tys.push(self.lw.types.get(p, e.span)?);
        }
        let ret_ty = self.lw.types.get(&ret, e.span)?;
        let thunk = self.lw.thunk(callee, &name, &param_tys, &ret_ty, e.span)?;
        let cell = self.alloc_cell(8)?;
        let code = self.emit(mir::InstKind::FuncAddr { func: thunk }, mir::Ty::ptr(mir::Ty::Void))?;
        self.store(cell, code)?;
        Ok(cell)
    }

    /// A closure value: allocate its environment, store the code
    /// address, then copy each capture from its enclosing local.
    fn closure_value(&mut self, id: u32, captures: &[usize], span: Span) -> KResult<mir::Value> {
        let enclosing = self.name.clone();
        let sub = self.sub.clone();
        let captures_tys: Vec<Ty> = self.lw.program.closures[id as usize].captures.iter().map(|c| c.ty.substitute(&sub)).collect();
        let env = self.lw.types.env_layout(&captures_tys, span)?;
        let func = self.lw.declare_closure(&enclosing, id, &sub, span)?;
        let cell = self.alloc_cell(env.size)?;
        let code = self.emit(mir::InstKind::FuncAddr { func }, mir::Ty::ptr(mir::Ty::Void))?;
        self.store(cell, code)?;
        for (i, &outer) in captures.iter().enumerate() {
            let src = self.read_local(outer)?;
            let dst = self.offset_addr(cell.clone(), env.offsets[i])?;
            self.store(dst, src)?;
        }
        Ok(cell)
    }

    fn alloc_cell(&mut self, size: u64) -> KResult<mir::Value> {
        let bytes = self.iconst(64, size as i128);
        self.call_alloc(bytes)
    }

    fn call_direct(&mut self, target: &FnRef, args: &[HirExpr], e: &HirExpr) -> KResult<mir::Value> {
        if let FnRef::Extern { id } = target {
            return self.call_extern(*id, args, e);
        }
        let sub = self.sub.clone();
        let (callee, _) = self.lw.resolve_callee(target, &sub, e.span)?;
        let mark = self.mark();
        let mut values = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            let value = self.expr(a)?;
            if i + 1 < args.len() {
                self.hold(value.clone())?;
            }
            values.push(value);
        }
        let ret = self.ty_of(&e.ty, e.span)?;
        let call = self.emit(mir::InstKind::Call { callee, args: values.into() }, ret);
        self.release(mark)?;
        call
    }

    /// A foreign call. Every pointer argument stays held across the call:
    /// the C callee cannot pin its own parameters, and helpers such as
    /// `fern_str_concat` allocate while reading them.
    fn call_extern(&mut self, id: ExternId, args: &[HirExpr], e: &HirExpr) -> KResult<mir::Value> {
        let target = self.lw.externs[&id];
        let mark = self.mark();
        let mut values = Vec::with_capacity(args.len());
        for a in args {
            let value = self.expr(a)?;
            self.hold(value.clone())?;
            values.push(value);
        }
        let ret = self.ty_of(&e.ty, e.span)?;
        let call = self.emit(
            mir::InstKind::CallExtern {
                target,
                args: values.into(),
            },
            ret,
        );
        self.release(mark)?;
        call
    }

    /// A call through a closure value: load the code address from the
    /// first cell slot and pass the cell itself as the environment.
    fn call_indirect(&mut self, callee: &HirExpr, args: &[HirExpr], e: &HirExpr) -> KResult<mir::Value> {
        let cell = self.expr(callee)?;
        let ptr = mir::Ty::ptr(mir::Ty::Void);
        let code = self.load(cell.clone(), &ptr)?;
        let mark = self.mark();
        self.hold(cell.clone())?;
        let mut values: Vec<mir::Value> = vec![cell];
        let mut param_tys: Vec<mir::Ty> = vec![ptr];
        for (i, a) in args.iter().enumerate() {
            let value = self.expr(a)?;
            if i + 1 < args.len() {
                self.hold(value.clone())?;
            }
            values.push(value);
            param_tys.push(self.ty_of(&a.ty, a.span)?);
        }
        let ret = self.ty_of(&e.ty, e.span)?;
        let sig = mir::Signature::new(param_tys, ret.clone());
        let call = self.emit(
            mir::InstKind::CallIndirect {
                callee: code,
                args: values.into(),
                sig,
            },
            ret,
        );
        self.release(mark)?;
        call
    }

    /// Build a tuple cell: the fields, packed from offset zero. The cell
    /// is pinned while the fields are evaluated.
    fn tuple(&mut self, e: &HirExpr, args: &[HirExpr]) -> KResult<mir::Value> {
        if args.is_empty() {
            return Ok(self.iconst(8, 0));
        }
        let layout = self.lw.types.tuple_layout(&e.ty.substitute(&self.sub), e.span)?;
        let cell = self.alloc_cell(layout.cell_size)?;
        let mark = self.hold(cell.clone())?;
        for (i, a) in args.iter().enumerate() {
            let value = self.expr(a)?;
            let dst = self.offset_addr(cell.clone(), layout.offsets[i])?;
            self.store(dst, value)?;
        }
        self.release(mark)?;
        Ok(cell)
    }

    fn let_expr(&mut self, binds: &[(HirPat, HirExpr)], body: &HirExpr) -> KResult<mir::Value> {
        for (pat, value) in binds {
            let v = self.expr(value)?;
            let dead = self.dead_block();
            self.bind_pattern(pat, v, dead)?;
        }
        self.expr(body)
    }

    fn if_expr(&mut self, cond: &HirExpr, then_body: &HirExpr, else_body: &HirExpr, e: &HirExpr) -> KResult<mir::Value> {
        let c = self.expr(cond)?;
        let ret = self.ty_of(&e.ty, e.span)?;
        let then_block = self.new_block(Vec::new());
        let else_block = self.new_block(Vec::new());
        let join = self.new_block(vec![ret]);
        self.terminate(mir::Terminator::CondBr {
            cond: c,
            then_block,
            else_block,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        self.goto(then_block);
        let t = self.expr(then_body)?;
        self.branch(join, vec![t]);
        self.goto(else_block);
        let f = self.expr(else_body)?;
        self.branch(join, vec![f]);
        self.goto(join);
        Ok(self.func.blocks[join].params[0])
    }

    /// Arms test in order; a pattern mismatch falls through to the next
    /// arm, a failed guard does the same, and running out of arms is
    /// unreachable (the checker does not prove exhaustiveness).
    fn match_expr(&mut self, e: &HirExpr, scrutinee: &HirExpr, arms: &[HirArm]) -> KResult<mir::Value> {
        let scrut = self.expr(scrutinee)?;
        let mark = self.hold(scrut.clone())?;
        let ret = self.ty_of(&e.ty, e.span)?;
        let join = self.new_block(vec![ret]);
        let tests: Vec<mir::Block> = arms.iter().map(|_| self.new_block(Vec::new())).collect();
        let dead = self.dead_block();
        self.branch(tests[0], Vec::new());
        for (i, arm) in arms.iter().enumerate() {
            self.goto(tests[i]);
            let fail = if i + 1 < arms.len() { tests[i + 1] } else { dead };
            self.bind_pattern(&arm.pattern, scrut.clone(), fail)?;
            if let Some(guard) = &arm.guard {
                let g = self.expr(guard)?;
                let body_block = self.new_block(Vec::new());
                self.terminate(mir::Terminator::CondBr {
                    cond: g,
                    then_block: body_block,
                    else_block: fail,
                    then_args: Default::default(),
                    else_args: Default::default(),
                });
                self.goto(body_block);
            }
            let value = self.expr(&arm.body)?;
            self.branch(join, vec![value]);
        }
        self.goto(join);
        self.release(mark)?;
        Ok(self.func.blocks[join].params[0])
    }

    fn bin(&mut self, op: BinOp, lhs: &HirExpr, rhs: &HirExpr, e: &HirExpr) -> KResult<mir::Value> {
        if matches!(op, BinOp::And | BinOp::Or) {
            return self.short_circuit(op, lhs, rhs);
        }
        let l = self.expr(lhs)?;
        let mark = self.hold(l.clone())?;
        let r = self.expr(rhs)?;
        self.release(mark)?;
        let operand = self.ty_of(&lhs.ty, lhs.span)?;
        // Strings compare by contents, not by address. The runtime
        // helper walks both NUL-terminated byte arrays.
        if matches!(op, BinOp::Eq | BinOp::Ne) && matches!(operand, mir::Ty::Ptr { .. }) {
            let call = self.emit(
                mir::InstKind::CallExtern {
                    target: self.lw.rt.str_eq,
                    args: vec![l, r].into(),
                },
                mir::Ty::Bool,
            )?;
            if matches!(op, BinOp::Ne) {
                let one = self.iconst(8, 1);
                return self.emit(mir::InstKind::Xor(call, one), mir::Ty::Bool);
            }
            return Ok(call);
        }
        let float = matches!(operand, mir::Ty::Float(_));
        let (kind, result) = match op {
            BinOp::Add => (arith_op(float, mir::InstKind::FAdd(l, r), mir::InstKind::Add(l, r)), operand),
            BinOp::Sub => (arith_op(float, mir::InstKind::FSub(l, r), mir::InstKind::Sub(l, r)), operand),
            BinOp::Mul => (arith_op(float, mir::InstKind::FMul(l, r), mir::InstKind::Mul(l, r)), operand),
            BinOp::Div => (arith_op(float, mir::InstKind::FDiv(l, r), mir::InstKind::Sdiv(l, r)), operand),
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne => {
                let kind = if float {
                    mir::InstKind::Fcmp {
                        pred: fcmp_pred(op),
                        lhs: l,
                        rhs: r,
                    }
                } else {
                    mir::InstKind::Icmp {
                        pred: icmp_pred(op),
                        lhs: l,
                        rhs: r,
                    }
                };
                (kind, mir::Ty::Bool)
            }
            BinOp::And | BinOp::Or | BinOp::Cons => {
                return Err(Diag::new("unexpected operator reached lowering", e.span));
            }
        };
        self.emit(kind, result)
    }

    /// `&&` and `||` skip the right operand, so a diverging branch
    /// stays unevaluated.
    fn short_circuit(&mut self, op: BinOp, lhs: &HirExpr, rhs: &HirExpr) -> KResult<mir::Value> {
        let l = self.expr(lhs)?;
        let join = self.new_block(vec![mir::Ty::Bool]);
        let rhs_block = self.new_block(Vec::new());
        let const_block = self.new_block(Vec::new());
        let (then_block, else_block) = match op {
            BinOp::And => (rhs_block, const_block),
            BinOp::Or => (const_block, rhs_block),
            _ => unreachable!(),
        };
        self.terminate(mir::Terminator::CondBr {
            cond: l,
            then_block,
            else_block,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        self.goto(const_block);
        let c = self.iconst(8, i128::from(op == BinOp::Or));
        self.branch(join, vec![c]);
        self.goto(rhs_block);
        let r = self.expr(rhs)?;
        self.branch(join, vec![r]);
        self.goto(join);
        Ok(self.func.blocks[join].params[0])
    }

    fn un(&mut self, op: UnOp, operand: &HirExpr, e: &HirExpr) -> KResult<mir::Value> {
        let v = self.expr(operand)?;
        let ty = self.ty_of(&e.ty, e.span)?;
        match op {
            UnOp::Neg => {
                let kind = if matches!(ty, mir::Ty::Float(_)) {
                    mir::InstKind::FNeg(v)
                } else {
                    mir::InstKind::Neg(v)
                };
                self.emit(kind, ty)
            }
            UnOp::Not => {
                let one = self.iconst(8, 1);
                self.emit(mir::InstKind::Xor(v, one), ty)
            }
        }
    }
}

fn arith_op(float: bool, f: mir::InstKind, i: mir::InstKind) -> mir::InstKind {
    if float { f } else { i }
}

fn icmp_pred(op: BinOp) -> mir::IcmpPred {
    match op {
        BinOp::Lt => mir::IcmpPred::Slt,
        BinOp::Le => mir::IcmpPred::Sle,
        BinOp::Gt => mir::IcmpPred::Sgt,
        BinOp::Ge => mir::IcmpPred::Sge,
        BinOp::Eq => mir::IcmpPred::Eq,
        BinOp::Ne => mir::IcmpPred::Ne,
        _ => unreachable!("not a comparison"),
    }
}

fn fcmp_pred(op: BinOp) -> mir::FcmpPred {
    match op {
        BinOp::Lt => mir::FcmpPred::Lt,
        BinOp::Le => mir::FcmpPred::Le,
        BinOp::Gt => mir::FcmpPred::Gt,
        BinOp::Ge => mir::FcmpPred::Ge,
        BinOp::Eq => mir::FcmpPred::Eq,
        BinOp::Ne => mir::FcmpPred::Ne,
        _ => unreachable!("not a comparison"),
    }
}

fn split_arrow(ty: &Ty) -> (Vec<Ty>, Ty) {
    let mut params = Vec::new();
    let mut cur = ty.clone();
    while let Ty::Arrow(s, d) = cur {
        params.push(*s);
        cur = *d;
    }
    (params, cur)
}
