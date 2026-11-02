use std::collections::{BTreeMap, HashMap};

use cranelift_codegen::ir::immediates::Offset32;
use cranelift_codegen::ir::types::{self};
use cranelift_codegen::ir::{
    FuncRef, InstBuilder, StackSlot,
};
use cranelift_codegen::ir::{Block as ClifBlock, Value as ClifValue};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{FuncId, Module};

use glcc_mir as mir;

use crate::abi::{self, SigLayout, Target, VaCtx};
use crate::memflags;
use crate::ty::clif_ty;
use super::*;

pub(crate) struct Body<'m> {
    pub(crate) func: &'m mir::Function,
    pub(crate) target: Target,
    pub(crate) layout: SigLayout,
    /// Every non-constant MIR value, resolved: scalars to Clif values,
    /// aggregates to the address of their stack slot.
    pub(crate) values: HashMap<mir::Value, ClifValue>,
    /// Constants already materialized in each Clif block. Constants
    /// have no defining block, so a single Clif definition would not
    /// dominate uses in sibling blocks; rebuilding per block keeps
    /// every definition and its uses in the same block.
    pub(crate) consts: HashMap<(ClifBlock, mir::Value), ClifValue>,
    pub(crate) blocks: HashMap<mir::Block, ClifBlock>,
    pub(crate) sret: Option<ClifValue>,
    /// The libc `memcpy` import, present when this function contains a
    /// copy large enough to call it.
    pub(crate) memcpy: Option<FuncRef>,
    /// Variadic prelude state, present exactly in variadic definitions.
    pub(crate) va: Option<VaCtx>,
    /// Where the Clif builder is positioned, to avoid redundant
    /// switches (switching away from an unfilled block panics).
    pub(crate) current: Option<ClifBlock>,
    /// Stack slot per `Alloca` result value, for resolving variable
    /// homes once the frame layout is known.
    pub(crate) slots: HashMap<mir::Value, StackSlot>,
    /// Debug sink while lowering with `-g`: interns instruction source
    /// locations and receives the function record after `define_function`.
    pub(crate) debug: Option<&'m mut glcc_dwarf::LoweredDebug>,
}

pub(crate) fn lower_function<M: Module>(
    id: mir::Func,
    mir_func: &mir::Function,
    target: Target,
    layouts: &HashMap<mir::Func, SigLayout>,
    extern_layouts: &HashMap<mir::Extern, SigLayout>,
    func_ids: &HashMap<mir::Func, FuncId>,
    extern_ids: &HashMap<mir::Extern, FuncId>,
    data_ids: &HashMap<mir::Global, cranelift_module::DataId>,
    tls_data: &std::collections::HashSet<mir::Global>,
    rt: &Rt,
    memcpy_id: Option<FuncId>,
    clif: &mut M,
    wrappers: &mut Wrappers,
    debug: Option<&mut glcc_dwarf::LoweredDebug>,
    unwind: Option<&mut Vec<WinUnwindEntry>>,
) {
    let layout = layouts[&id].clone();
    let mut ctx = clif.make_context();
    ctx.func.signature = abi::clif_signature(&layout, target.call_conv());
    let memcpy = memcpy_id.map(|id| clif.declare_func_in_func(id, &mut ctx.func));
    let mut func_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

    let mut body = Body {
        func: mir_func,
        target,
        layout,
        values: HashMap::new(),
        consts: HashMap::new(),
        blocks: HashMap::new(),
        sret: None,
        memcpy,
        va: None,
        current: None,
        slots: HashMap::new(),
        debug,
    };
    for (block_id, _) in mir_func.blocks.iter_enumerated() {
        body.blocks.insert(block_id, builder.create_block());
    }
    let entry = body.blocks[&mir_func.entry];
    builder.append_block_params_for_function_params(entry);
    body.switch_to(&mut builder, entry);
    body.map_entry_params(&mut builder, mir_func, entry);
    // Non-entry block params become Clif block params up front so
    // forward branches transcribe without fixups.
    for (block_id, block) in mir_func.blocks.iter_enumerated() {
        if block_id == mir_func.entry {
            continue;
        }
        let clif_block = body.blocks[&block_id];
        for param in &block.params {
            let mir_ty = &mir_func.values[*param].ty;
            let param_ty = clif_ty(mir_ty).unwrap_or(types::I64);
            let clif_param = builder.append_block_param(clif_block, param_ty);
            body.values.insert(*param, clif_param);
        }
    }
    if body.layout.variadic {
        let va = body.emit_vararg_prelude(&mut builder);
        body.va = Some(va);
    }
    for block_id in block_fill_order(mir_func) {
        let target_block = body.blocks[&block_id];
        body.switch_to(&mut builder, target_block);
        body.lower_block(&mut builder,
            clif,
            func_ids,
            extern_ids,
            data_ids,
            tls_data,
            rt,
            layouts,
            extern_layouts,
            block_id,
            wrappers,
        );
    }
    builder.seal_all_blocks();
    let config = clif.target_config();
    builder.finalize(config);
    let clif_id = func_ids[&id];
    if let Err(e) = clif.define_function(clif_id, &mut ctx) {
        eprintln!(
            "FAILED {}:
{}",
            mir_func.name,
            ctx.func.display()
        );
        panic!("cannot define function: {e:?}");
    }
    if let Some(debug) = body.debug.take()
        && let Some(code) = ctx.compiled_code() {
            crate::debug::collect_function_debug(debug, clif_id, mir_func, &body.slots, code, data_ids, tls_data);
        }
    if let Some(unwind) = unwind
        && let Some(code) = ctx.compiled_code() {
            let info = code
                .create_unwind_info(clif.isa())
                .unwrap_or_else(|e| panic!("cannot create unwind info for {}: {e:?}", mir_func.name));
            if let Some(cranelift_codegen::isa::unwind::UnwindInfo::WindowsX64(info)) = info {
                let mut xdata = vec![0u8; info.emit_size()];
                info.emit(&mut xdata);
                unwind.push(WinUnwindEntry {
                    func: clif_id,
                    size: code.buffer.data().len() as u32,
                    xdata,
                });
            }
        }
}

/// Order for filling blocks: reverse postorder over the reachable CFG,
/// then unreachable blocks in arena order. A definition dominates its
/// uses, and dominators precede in reverse postorder, so every operand
/// exists as a Clif value by the time its use is lowered — no matter
/// how passes ordered the arena.
pub(crate) fn block_fill_order(func: &mir::Function) -> Vec<mir::Block> {
    let reachable = func.reachable_blocks();
    let mut post = Vec::new();
    let mut seen = vec![false; func.blocks.len()];
    let mut work = vec![(func.entry, false)];
    seen[func.entry.index()] = true;
    while let Some((block, leaving)) = work.pop() {
        if leaving {
            post.push(block);
            continue;
        }
        work.push((block, true));
        if let Some(term) = &func.blocks[block].term {
            for succ in term.successors().iter().rev() {
                if reachable[succ.index()] && !seen[succ.index()] {
                    seen[succ.index()] = true;
                    work.push((*succ, false));
                }
            }
        }
    }
    post.reverse();
    post.extend((0..func.blocks.len()).map(mir::Block::from_usize).filter(|block| !reachable[block.index()]));
    post
}

/// MIR type of the `owner`-th signature parameter.
pub(crate) fn param_mir_ty(mir_func: &mir::Function, owner: usize) -> mir::Ty {
    mir_func.sig.params[owner].clone()
}


impl<'m> Body<'m> {
/// Map entry block params to MIR values: scalars directly, aggregates
/// reassembled into fresh slots from their pieces, chunk groups stored
/// whole, and `StructArgument` addresses copied to locals.
pub(crate) fn map_entry_params(&mut self, builder: &mut FunctionBuilder, mir_func: &mir::Function, entry: ClifBlock) {
    let params = builder.block_params(entry).to_vec();
    // Per-owner piece collection: (MIR type, [(offset, ClifValue)]).
    // `BTreeMap` keeps owner iteration in param order so stack-slot
    // creation and store emission are deterministic across runs
    // (`HashMap` iteration would permute them via its random seed).
    let mut pieces: BTreeMap<usize, (mir::Ty, Vec<(u64, ClifValue)>)> = BTreeMap::new();
    let mut addrs: BTreeMap<usize, ClifValue> = BTreeMap::new();
    let mut slot_idx = 0;
    for slot in self.layout.slots.clone() {
        match slot.kind {
            abi::SlotKind::Sret => {
                self.sret = Some(params[slot_idx]);
                slot_idx += 1;
            }
            abi::SlotKind::Pad => {
                slot_idx += 1;
            }
            abi::SlotKind::Value { owner, offset } => {
                let v = params[slot_idx];
                slot_idx += 1;
                let mir_ty = param_mir_ty(mir_func, owner);
                if !mir_ty.is_aggregate() {
                    // Scalars may still span slots (128-bit pairs);
                    // collect them for joining below.
                    pieces.entry(owner).or_insert_with(|| (mir_ty, Vec::new())).1.push((offset, v));
                    continue;
                }
                pieces.entry(owner).or_insert_with(|| (mir_ty, Vec::new())).1.push((offset, v));
            }
            abi::SlotKind::Chunk { owner, index } => {
                let v = params[slot_idx];
                slot_idx += 1;
                let mir_ty = param_mir_ty(mir_func, owner);
                pieces.entry(owner).or_insert_with(|| (mir_ty, Vec::new())).1.push((index * 8, v));
            }
            abi::SlotKind::StackAddr { owner } | abi::SlotKind::RefAddr { owner } => {
                let v = params[slot_idx];
                slot_idx += 1;
                addrs.insert(owner, v);
            }
        }
    }
    for (owner, (mir_ty, parts)) in &pieces {
        let Some(mir_param) = mir_func.blocks[mir_func.entry].params.get(*owner).copied() else {
            continue;
        };
        if !mir_ty.is_aggregate() {
            if parts.len() == 1 {
                // Narrow scalars arrive in a wider register whose upper
                // bits the caller leaves unspecified on every target
                // (the cover type is register-width, e.g. a `char` in
                // w0 on AAPCS64). Besides dropping caller residue, the
                // reduce keeps the -O0 entry spill (a declaration-sized
                // slot) from receiving an over-wide store.
                let v = narrow_to_declared(builder, parts[0].1, mir_ty);
                self.values.insert(mir_param, v);
            } else {
                // Scalar spanning slots (128-bit pairs) rejoins here.
                let mut sorted = parts.clone();
                sorted.sort_by_key(|(off, _)| *off);
                let joined = join_values(builder, &sorted, mir_ty);
                self.values.insert(mir_param, joined);
            }
            continue;
        }
        let slot = stack_slot(builder, mir_ty);
        let addr = builder.ins().stack_addr(types::I64, slot, 0);
        for (off, v) in parts {
            // Chunk stores cover whole 8-byte granules; piece stores
            // land at exact field offsets. The address is always an
            // integer; only the stored value carries its own width.
            builder.ins().stack_store(types::I64, *v, slot, *off as i32);
        }
        self.values.insert(mir_param, addr);
    }
    for (owner, addr) in &addrs {
        let Some(mir_param) = mir_func.blocks[mir_func.entry].params.get(*owner).copied() else {
            continue;
        };
        let mir_ty = param_mir_ty(mir_func, *owner);
        if mir_ty.is_aggregate() {
            let slot = stack_slot(builder, &mir_ty);
            let dst = builder.ins().stack_addr(types::I64, slot, 0);
            self.emit_copy(builder, dst, *addr, mir_ty.size_bytes());
            self.values.insert(mir_param, dst);
        } else {
            // Scalar behind an address (x86-64 `long double`): load
            // the bits; values stay uniformly bitwise downstream.
            let ty = clif_ty(&mir_ty).expect("scalar address needs a Clif type");
            let v = builder.ins().load(ty, memflags::notrap(), *addr, Offset32::new(0));
            self.values.insert(mir_param, v);
        }
    }
    // Zero-sized aggregate parameters cross as nothing and so have no
    // slot: bind them to a dummy address of their own, since MIR uses
    // (entry spills, address math) expect them defined.
    for (i, param) in mir_func.blocks[mir_func.entry].params.iter().enumerate() {
        if self.values.contains_key(param) {
            continue;
        }
        let mir_ty = param_mir_ty(mir_func, i);
        if mir_ty.is_aggregate() && mir_ty.size_bytes() == 0 {
            let slot = stack_slot(builder, &mir_ty);
            let addr = builder.ins().stack_addr(types::I64, slot, 0);
            self.values.insert(*param, addr);
        }
    }
}
/// Emit the variadic prelude for a definition: spill register-passed
/// parameters and compute the overflow base into [`VaCtx`].
pub(crate) fn emit_vararg_prelude(&mut self, builder: &mut FunctionBuilder) -> VaCtx {
    let params = builder.block_params(self.blocks[&self.func.entry]).to_vec();
    let paired: Vec<(ClifValue, abi::Slot)> = self.layout.slots.clone().into_iter().zip(params).map(|(slot, v)| (v, slot)).collect();
    abi::emit_prelude(self.target, builder, &paired, &self.layout)
}
pub(crate) fn lower_block<M: Module>(
    &mut self,
    builder: &mut FunctionBuilder,
    clif: &mut M,
    func_ids: &HashMap<mir::Func, FuncId>,
    extern_ids: &HashMap<mir::Extern, FuncId>,
    data_ids: &HashMap<mir::Global, cranelift_module::DataId>,
    tls_data: &std::collections::HashSet<mir::Global>,
    rt: &Rt,
    layouts: &HashMap<mir::Func, SigLayout>,
    extern_layouts: &HashMap<mir::Extern, SigLayout>,
    id: mir::Block,
    wrappers: &mut Wrappers,
) {
    let block = &self.func.blocks[id];
    let insts = block.insts.clone();
    for inst in insts {
        self.emit_inst(builder, clif, func_ids, extern_ids, data_ids, tls_data, rt, layouts, extern_layouts, inst, wrappers);
    }
    let Some(term) = block.term.clone() else {
        panic!("unterminated MIR block");
    };
    self.emit_term(builder, term);
}
}
