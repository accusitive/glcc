pub mod constmerge;
pub mod cse;
pub mod instcombine;
pub mod licm;
pub mod sccp;

pub use constmerge::ConstMerge;
pub use cse::Cse;
pub use instcombine::InstCombine;
pub use licm::Licm;
pub use sccp::Sccp;

use crate::ir::{Block, Const, Def, FcmpPred, Function, IcmpPred, Inst, InstKind, Module, Terminator, Value};
use crate::pass::{FunctionPass, ModulePass};
use crate::ty::{FloatKind, Ty};

fn int_const(func: &Function, v: Value) -> Option<(u32, i128)> {
    match func.const_value(v) {
        Some(Const::Int { bits, value }) => Some((*bits, *value)),
        _ => None,
    }
}

/// Integer/bool width of a MIR type, for conversion folds.
fn int_ty_bits(ty: &Ty) -> Option<u32> {
    match ty {
        Ty::Int { bits } => Some(*bits),
        Ty::Bool => Some(1),
        _ => None,
    }
}

fn mask(bits: u32) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else if bits == 0 {
        0
    } else {
        (1u128 << bits) - 1
    }
}

fn wrap(bits: u32, v: i128) -> i128 {
    let m = mask(bits);
    let truncated = (v as u128) & m;
    if bits == 0 {
        return 0;
    }
    if truncated >> (bits - 1) & 1 == 1 {
        (truncated | !m) as i128
    } else {
        truncated as i128
    }
}

fn eval_icmp(pred: IcmpPred, bits: u32, lhs: i128, rhs: i128) -> bool {
    let um = mask(bits);
    let ul = (lhs as u128) & um;
    let ur = (rhs as u128) & um;
    match pred {
        IcmpPred::Eq => ul == ur,
        IcmpPred::Ne => ul != ur,
        IcmpPred::Ult => ul < ur,
        IcmpPred::Ule => ul <= ur,
        IcmpPred::Ugt => ul > ur,
        IcmpPred::Uge => ul >= ur,
        IcmpPred::Slt => lhs < rhs,
        IcmpPred::Sle => lhs <= rhs,
        IcmpPred::Sgt => lhs > rhs,
        IcmpPred::Sge => lhs >= rhs,
    }
}

/// Bits-level fold helpers shared by ConstFold (which resolves `Value`
/// operands first) and SCCP (whose operands are lattice contents). Each
/// refuses to produce a constant where that would erase a runtime trap
/// or read a bit pattern the backend defines: division by a wrapped
/// zero and oversized shifts never fold.
fn fold_int2_bits(bits: u32, x: i128, y: i128, f: impl FnOnce(i128, i128) -> i128) -> Option<Const> {
    Some(Const::Int { bits, value: wrap(bits, f(x, y)) })
}

/// Refuse oversized shifts (C UB, and the backend defines the hardware
/// behavior): those stay for codegen.
fn fold_shift_bits(bits: u32, x: u128, s: i128, f: impl FnOnce(u128, u32) -> u128) -> Option<Const> {
    if bits == 0 || bits > 128 || s < 0 || s as u128 >= bits as u128 {
        return None;
    }
    let m = mask(bits);
    Some(Const::Int {
        bits,
        value: wrap(bits, f(x & m, s as u32) as i128),
    })
}

/// Refuse zero divisors so the runtime trap survives.
fn fold_div_bits(bits: u32, x: i128, y: i128, signed: bool, is_rem: bool) -> Option<Const> {
    if y == 0 || bits == 0 || bits > 128 {
        return None;
    }
    let value = if signed {
        wrap(bits, if is_rem { x.wrapping_rem(y) } else { x.wrapping_div(y) })
    } else {
        let m = mask(bits);
        let xu = (x as u128) & m;
        let yu = (y as u128) & m;
        // `y != 0` as a wrapped value implies `yu != 0`.
        let q = if is_rem { xu.wrapping_rem(yu) } else { xu.wrapping_div(yu) };
        wrap(bits, q as i128)
    };
    Some(Const::Int { bits, value })
}

fn fold_neg_bits(bits: u32, x: i128) -> Option<Const> {
    Some(Const::Int {
        bits,
        value: wrap(bits, x.wrapping_neg()),
    })
}

fn fold_not_bits(bits: u32, x: i128) -> Option<Const> {
    Some(Const::Int { bits, value: wrap(bits, !x) })
}

/// An arithmetic shift right (sign-preserving Rust `>>` on the wrapped
/// value), refusing oversized shifts like the other shifts.
fn fold_ashr_bits(bits: u32, x: i128, s: i128) -> Option<Const> {
    if bits == 0 || bits > 128 || s < 0 || s as u128 >= bits as u128 {
        return None;
    }
    Some(Const::Int {
        bits,
        value: wrap(bits, x >> s as u32),
    })
}

/// IEEE NaN semantics for one `Fcmp` over resolved float values.
fn fold_fcmp_bits(pred: FcmpPred, l: f64, r: f64) -> bool {
    let nan = l.is_nan() || r.is_nan();
    match pred {
        FcmpPred::Eq => !nan && l == r,
        FcmpPred::Ne => nan || l != r,
        FcmpPred::Lt => !nan && l < r,
        FcmpPred::Le => !nan && l <= r,
        FcmpPred::Gt => !nan && l > r,
        FcmpPred::Ge => !nan && l >= r,
        FcmpPred::Uno => nan,
        FcmpPred::Ord => !nan,
    }
}

/// Fold constant arithmetic, comparisons, selects, and branches on
/// constant conditions.
///
/// Division/modulo by zero and oversized shifts never fold: both trap
/// (or are backend-defined) at runtime, and folding them away would
/// delete the trap. Float arithmetic stays unfolded — Cranelift folds
/// it during codegen — except exact bitwise negation.
#[derive(Default)]
pub struct ConstFold;

impl FunctionPass for ConstFold {
    fn name(&self) -> &'static str {
        "const-fold"
    }

    fn run(&mut self, func: &mut Function) -> bool {
        let mut changed = false;
        let insts: Vec<_> = func.blocks.iter().flat_map(|b| b.insts.clone()).collect();
        for inst in insts {
            let kind = func.insts[inst].kind.clone();
            let result_ty = func.insts[inst].result.map(|v| func.values[v].ty.clone());
            let folded: Option<Const> = match &kind {
                InstKind::Add(a, b) => fold_int2(func, *a, *b, |x, y| x.wrapping_add(y)),
                InstKind::Sub(a, b) => fold_int2(func, *a, *b, |x, y| x.wrapping_sub(y)),
                InstKind::Mul(a, b) => fold_int2(func, *a, *b, |x, y| x.wrapping_mul(y)),
                InstKind::And(a, b) => fold_int2(func, *a, *b, |x, y| x & y),
                InstKind::Or(a, b) => fold_int2(func, *a, *b, |x, y| x | y),
                InstKind::Xor(a, b) => fold_int2(func, *a, *b, |x, y| x ^ y),
                InstKind::Shl(a, b) => fold_shift(func, *a, *b, |x, s| x.wrapping_shl(s)),
                InstKind::LShr(a, b) => fold_shift(func, *a, *b, |x, s| x.wrapping_shr(s)),
                InstKind::AShr(a, b) => fold_ashr(func, *a, *b),
                InstKind::Udiv(a, b) => fold_div(func, *a, *b, false, false),
                InstKind::Sdiv(a, b) => fold_div(func, *a, *b, true, false),
                InstKind::Urem(a, b) => fold_div(func, *a, *b, false, true),
                InstKind::Srem(a, b) => fold_div(func, *a, *b, true, true),
                InstKind::Neg(a) => fold_neg(func, *a),
                InstKind::Not(a) => fold_not(func, *a),
                InstKind::Trunc(a) => fold_conv(func, *a, result_ty.as_ref(), |x, _, tb| wrap(tb, x)),
                InstKind::Zext(a) => fold_conv(func, *a, result_ty.as_ref(), |x, ob, tb| wrap(tb, ((x as u128) & mask(ob)) as i128)),
                InstKind::Sext(a) => fold_conv(func, *a, result_ty.as_ref(), |x, _, tb| wrap(tb, x)),
                InstKind::Bitcast(a) => fold_bitcast(func, *a, result_ty.as_ref()),
                InstKind::FNeg(a) => match func.const_value(*a) {
                    Some(Const::Float { kind, bits }) => {
                        let sign = match kind {
                            FloatKind::F32 => 1u128 << 31,
                            FloatKind::F64 => 1u128 << 63,
                            // F80 rides a 16-byte carrier with the
                            // x87 sign at bit 79, not the top.
                            FloatKind::F80 => 1u128 << 79,
                            FloatKind::F128 => 1u128 << 127,
                        };
                        Some(Const::Float { kind: *kind, bits: bits ^ sign })
                    }
                    _ => None,
                },
                InstKind::Fcmp { pred, lhs, rhs } => fold_fcmp(func, *pred, *lhs, *rhs),
                InstKind::Icmp { pred, lhs, rhs } => {
                    if let (Some((bits, l)), Some((_, r))) = (int_const(func, *lhs), int_const(func, *rhs)) {
                        let b = eval_icmp(*pred, bits, l, r);
                        Some(Const::Int { bits: 1, value: i128::from(b) })
                    } else {
                        None
                    }
                }
                InstKind::Select { cond, then_val, else_val } => match int_const(func, *cond) {
                    Some((_, 0)) => func.const_value(*else_val).cloned(),
                    Some(_) => func.const_value(*then_val).cloned(),
                    None => None,
                },
                _ => None,
            };
            if let Some(c) = folded {
                if let Some(result) = func.insts[inst].result {
                    // The replacement stands in for `result`, so it must
                    // carry `result`'s type: `c.ty()` loses it (`Undef` is
                    // `Void`, `Null` is `ptr(Void)`, aggregates have none).
                    let ty = func.values[result].ty.clone();
                    let replacement = func.values.push(crate::ir::ValueData { ty, def: Def::Const(c) });
                    func.replace_uses(result, replacement);
                    changed = true;
                }
            }
        }
        for idx in 0..func.blocks.len() {
            let block_id = crate::ir::Block::from_usize(idx);
            let term = func.blocks[block_id].term.clone();
            let folded = match term {
                Some(Terminator::CondBr {
                    cond,
                    then_block,
                    else_block,
                    then_args,
                    else_args,
                }) => {
                    if then_block == else_block && then_args == else_args {
                        Some(Terminator::Br {
                            dest: then_block,
                            args: then_args,
                        })
                    } else {
                        match func.const_value(cond) {
                            Some(Const::Int { value: 0, .. }) => Some(Terminator::Br {
                                dest: else_block,
                                args: else_args,
                            }),
                            Some(Const::Int { .. }) => Some(Terminator::Br {
                                dest: then_block,
                                args: then_args,
                            }),
                            _ => None,
                        }
                    }
                }
                Some(Terminator::Switch { scrutinee, cases, default }) => match func.const_value(scrutinee) {
                    Some(Const::Int { bits, value }) => {
                        // Cases compare in the scrutinee's own width
                        // (lowering promotes the scrutinee first, so
                        // widths already agree): match low bits.
                        let m = mask(*bits);
                        let v = (*value as u128) & m;
                        let dest = cases.iter().find(|(k, _)| (*k as u128) & m == v).map(|(_, b)| *b).unwrap_or(default);
                        Some(Terminator::Br { dest, args: Default::default() })
                    }
                    _ => None,
                },
                _ => None,
            };
            if let Some(term) = folded {
                func.blocks[block_id].term = Some(term);
                changed = true;
            }
        }
        changed
    }
}

fn fold_int2(func: &Function, a: Value, b: Value, f: impl Fn(i128, i128) -> i128) -> Option<Const> {
    let (bits, x) = int_const(func, a)?;
    let (_, y) = int_const(func, b)?;
    fold_int2_bits(bits, x, y, f)
}

/// Fold a bit-shifting op, refusing oversized shifts (C UB, and the
/// backend defines the hardware behavior): those stay for codegen.
fn fold_shift(func: &Function, a: Value, b: Value, f: impl Fn(u128, u32) -> u128) -> Option<Const> {
    let (bits, x) = int_const(func, a)?;
    let (_, y) = int_const(func, b)?;
    fold_shift_bits(bits, x as u128, y, f)
}

/// Fold division/remainder, refusing zero divisors so the runtime
/// trap survives.
fn fold_div(func: &Function, a: Value, b: Value, signed: bool, is_rem: bool) -> Option<Const> {
    let (bits, x) = int_const(func, a)?;
    let (_, y) = int_const(func, b)?;
    fold_div_bits(bits, x, y, signed, is_rem)
}

/// Fold two's-complement negation with wraparound.
fn fold_neg(func: &Function, a: Value) -> Option<Const> {
    let (bits, x) = int_const(func, a)?;
    fold_neg_bits(bits, x)
}

/// Fold bitwise complement.
fn fold_not(func: &Function, a: Value) -> Option<Const> {
    let (bits, x) = int_const(func, a)?;
    fold_not_bits(bits, x)
}

/// Fold an arithmetic shift right (sign-preserving Rust `>>` on the
/// wrapped value), refusing oversized shifts like the other shifts.
fn fold_ashr(func: &Function, a: Value, b: Value) -> Option<Const> {
    let (bits, x) = int_const(func, a)?;
    let (_, y) = int_const(func, b)?;
    fold_ashr_bits(bits, x, y)
}

/// Fold an int-to-int bitcast, only when the widths agree (a
/// width-changing bitcast has no MIR meaning to preserve).
fn fold_bitcast(func: &Function, a: Value, result_ty: Option<&Ty>) -> Option<Const> {
    let (obits, x) = int_const(func, a)?;
    let tbits = result_ty.and_then(int_ty_bits)?;
    if obits != tbits {
        return None;
    }
    let c = Const::Int {
        bits: tbits,
        value: wrap(tbits, x),
    };
    if result_ty != Some(&c.ty()) {
        return None;
    }
    Some(c)
}
fn fold_conv(func: &Function, a: Value, result_ty: Option<&Ty>, f: impl Fn(i128, u32, u32) -> i128) -> Option<Const> {
    let (obits, x) = int_const(func, a)?;
    let tbits = result_ty.and_then(int_ty_bits)?;
    let c = Const::Int {
        bits: tbits,
        value: f(x, obits, tbits),
    };
    if result_ty != Some(&c.ty()) {
        return None;
    }
    Some(c)
}

/// Fold an `Fcmp` over `f32`/`f64` constants with IEEE NaN semantics.
/// Wider carriers stay unfolded (no host-side `f80`/`f128` compare).
fn fold_fcmp(func: &Function, pred: FcmpPred, lhs: Value, rhs: Value) -> Option<Const> {
    let (l, r) = match (func.const_value(lhs), func.const_value(rhs)) {
        (Some(Const::Float { kind: FloatKind::F32, bits: lb }), Some(Const::Float { kind: FloatKind::F32, bits: rb })) => {
            (f32::from_bits(*lb as u32) as f64, f32::from_bits(*rb as u32) as f64)
        }
        (Some(Const::Float { kind: FloatKind::F64, bits: lb }), Some(Const::Float { kind: FloatKind::F64, bits: rb })) => (f64::from_bits(*lb as u64), f64::from_bits(*rb as u64)),
        _ => return None,
    };
    Some(Const::Int {
        bits: 1,
        value: i128::from(fold_fcmp_bits(pred, l, r)),
    })
}

/// Remove pure instructions whose result is never used.
#[derive(Default)]
pub struct DeadCodeElim;

impl FunctionPass for DeadCodeElim {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn run(&mut self, func: &mut Function) -> bool {
        use std::collections::HashSet;
        let resident: HashSet<crate::ir::Inst> = func.blocks.iter().flat_map(|b| b.insts.iter().copied()).collect();
        let counts = func.count_resident_uses();
        let mut dead = Vec::new();
        for (id, inst) in func.insts.iter_enumerated() {
            if !resident.contains(&id) {
                continue;
            }
            if !inst.kind.is_pure() {
                continue;
            }
            if let Some(result) = inst.result {
                if counts.get(&result).copied().unwrap_or(0) == 0 {
                    dead.push(id);
                }
            }
        }
        if dead.is_empty() {
            return false;
        }
        let gone: HashSet<_> = dead.iter().copied().collect();
        for block in func.blocks.iter_mut() {
            block.insts.retain(|i| !gone.contains(i));
        }
        true
    }
}

/// Addresses that escape into untracked uses: every value observed
/// as an operand anywhere except as a direct `Load`/`Store` address.
/// A slot whose address never escapes is only ever read and written
/// through known `Load`/`Store` sites, which is what makes it
/// promotable (see `Mem2Reg`) and what future store-sinking passes
/// reuse. Arena-orphaned instructions (removed from block lists by
/// earlier passes) still count: conservatively sound, and consistent
/// with `Function::use_count`.
pub fn escaping_addresses(func: &Function) -> std::collections::HashSet<Value> {
    use std::collections::HashSet;
    let mut escaped = HashSet::new();
    for inst in func.insts.iter() {
        match &inst.kind {
            InstKind::Load { .. } => {}
            InstKind::Store { val, .. } => {
                escaped.insert(*val);
            }
            other => {
                other.for_each_operand(&mut |v| {
                    escaped.insert(v);
                });
            }
        }
    }
    for block in func.blocks.iter() {
        if let Some(term) = &block.term {
            term.for_each_operand(&mut |v| {
                escaped.insert(v);
            });
        }
    }
    escaped
}

/// Promote scalar stack slots to SSA values across the whole CFG.
///
/// Lowering spills every parameter and local into an entry-block
/// `Alloca`, so even trivial functions are memory-bound. A slot is
/// promotable when it is a single scalar whose address never escapes
/// (`escaping_addresses`), is never touched by a volatile access, and
/// is only accessed from reachable blocks. Promotion is full SSA:
///
/// - dominance frontiers of the store blocks place phis (block
///   parameters) at every merge that sees several definitions;
/// - a dominator-tree walk renames loads to the current definition,
///   feeds every store into the dataflow, passes each edge's value to
///   the successor phis, and removes the slot's loads, stores, and
///   alloca once nothing references the address.
///
/// A load reached before any store reads `undef` (C UB; the backend
/// materializes it as zero), matching LLVM's mem2reg.
///
/// The entry block and `Switch` targets stay phi-free: entry params
/// map to the function signature, and `Switch` carries no branch
/// arguments. A slot that would need a phi at one of those sites stays
/// in memory (a later `SimplifyCfg` usually removes the reason).
#[derive(Default)]
pub struct Mem2Reg;

impl FunctionPass for Mem2Reg {
    fn name(&self) -> &'static str {
        "mem2reg"
    }

    fn run(&mut self, func: &mut Function) -> bool {
        promote_slots(func)
    }
}

/// A store through a promotable slot.
struct SlotStore {
    block: Block,
    inst: Inst,
}

/// A promotable slot and its access sites in instruction order.
struct SlotSites {
    alloca: Inst,
    addr: Value,
    pointee: Ty,
    loads: Vec<Inst>,
    stores: Vec<SlotStore>,
}

fn promote_slots(func: &mut Function) -> bool {
    use std::collections::{HashMap, HashSet};

    // A `setjmp` call site can be re-entered by `longjmp` with the
    // registers of the first return: any slot written between the two
    // would read stale. Leave every slot in memory.
    if func.returns_twice {
        return false;
    }

    let resident: HashSet<Inst> = func.blocks.iter().flat_map(|block| block.insts.iter().copied()).collect();
    let escaped = escaping_addresses(func);
    let reachable = func.reachable_blocks();

    // Index load/store sites by address once: the per-slot scan used
    // to be O(slots * insts).
    let mut sites: HashMap<Value, Vec<Inst>> = HashMap::new();
    for (id, inst) in func.insts.iter_enumerated() {
        if !resident.contains(&id) {
            continue;
        }
        if let InstKind::Load { addr, .. } | InstKind::Store { addr, .. } = &inst.kind {
            sites.entry(*addr).or_default().push(id);
        }
    }

    let mut promotable: Vec<SlotSites> = Vec::new();
    let mut unused: Vec<Inst> = Vec::new();
    for (id, inst) in func.insts.iter_enumerated() {
        if !resident.contains(&id) {
            continue;
        }
        let InstKind::Alloca { pointee, count } = &inst.kind else {
            continue;
        };
        if *count != 1 || !pointee.is_scalar() {
            continue;
        }
        let Some(addr) = inst.result else { continue };
        if escaped.contains(&addr) {
            continue;
        }
        let mut slot = SlotSites {
            alloca: id,
            addr,
            pointee: pointee.clone(),
            loads: Vec::new(),
            stores: Vec::new(),
        };
        let mut ok = true;
        let mut slot_sites: &[Inst] = &[];
        if let Some(ids) = sites.get(&addr) {
            slot_sites = ids;
        }
        for &site in slot_sites {
            let data = &func.insts[site];
            if !reachable[data.block.index()] {
                ok = false;
                break;
            }
            match &data.kind {
                InstKind::Load { flags, .. } if !flags.volatile => match data.result {
                    Some(result) if func.values[result].ty == slot.pointee => {
                        slot.loads.push(site);
                    }
                    _ => {
                        ok = false;
                        break;
                    }
                },
                InstKind::Store { flags, .. } if !flags.volatile => {
                    slot.stores.push(SlotStore { block: data.block, inst: site });
                }
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        if slot.loads.is_empty() && slot.stores.is_empty() {
            unused.push(id);
        } else {
            promotable.push(slot);
        }
    }

    if promotable.is_empty() && unused.is_empty() {
        return false;
    }

    let mut replace: std::collections::HashMap<Value, Value> = std::collections::HashMap::new();
    let mut dead: std::collections::HashSet<Inst> = std::collections::HashSet::new();
    let mut changed = false;
    if !promotable.is_empty() {
        let dom = DomInfo::compute(func, &reachable);
        let switch_targets = switch_targets(func);
        let mut undef_cache: Vec<(Ty, Value)> = Vec::new();
        for slot in &promotable {
            changed |= promote_slot(func, &dom, &switch_targets, slot, &mut replace, &mut dead, &mut undef_cache);
        }
    }

    if !replace.is_empty() {
        collapse_chains(&mut replace);
        func.remap_operands(&replace);
    }
    if !unused.is_empty() {
        dead.extend(unused.iter().copied());
        changed = true;
    }
    if !dead.is_empty() {
        for block in func.blocks.iter_mut() {
            block.insts.retain(|inst| !dead.contains(inst));
        }
    }
    changed
}

/// Promote one slot: place its phis, then rename its accesses.
fn promote_slot(
    func: &mut Function,
    dom: &DomInfo,
    switch_targets: &std::collections::HashSet<Block>,
    slot: &SlotSites,
    replace: &mut std::collections::HashMap<Value, Value>,
    dead: &mut std::collections::HashSet<Inst>,
    undef_cache: &mut Vec<(Ty, Value)>,
) -> bool {
    let mut store_blocks: Vec<Block> = slot.stores.iter().map(|store| store.block).collect();
    store_blocks.sort_by_key(|block| block.index());
    store_blocks.dedup();
    let Some(phi_blocks) = place_phis(func, dom, &store_blocks, switch_targets) else {
        return false;
    };

    let undef = undef_value(func, &slot.pointee, undef_cache);
    let mut phi_index: std::collections::HashMap<Block, usize> = std::collections::HashMap::new();
    for &block in &phi_blocks {
        let index = func.blocks[block].params.len();
        let param = func.values.push(crate::ir::ValueData {
            ty: slot.pointee.clone(),
            def: Def::Param(block, index as u32),
        });
        func.blocks[block].params.push(param);
        phi_index.insert(block, index);
        extend_incoming_edges(func, block, undef);
    }

    rename_slot(func, dom, slot, &phi_index, undef, replace);
    for &load in &slot.loads {
        dead.insert(load);
    }
    for store in &slot.stores {
        dead.insert(store.inst);
    }
    dead.insert(slot.alloca);
    true
}

/// Cytron's iterated dominance frontier: every block that merges two
/// definitions of the slot gets one phi. Returns `None` when a phi
/// would land somewhere unsupported (the entry block or a `Switch`
/// target), leaving the caller to keep the slot in memory.
fn place_phis(func: &Function, dom: &DomInfo, store_blocks: &[Block], switch_targets: &std::collections::HashSet<Block>) -> Option<Vec<Block>> {
    let mut placed: std::collections::HashSet<Block> = std::collections::HashSet::new();
    let mut work: Vec<Block> = store_blocks.to_vec();
    let mut phis = Vec::new();
    while let Some(block) = work.pop() {
        for &frontier in &dom.df[block.index()] {
            if !placed.insert(frontier) {
                continue;
            }
            if frontier == func.entry || switch_targets.contains(&frontier) {
                return None;
            }
            phis.push(frontier);
            work.push(frontier);
        }
    }
    Some(phis)
}

/// Rename loads to the definition current on the dominator-tree walk,
/// and record each store's value for the edges below it. The stack
/// starts at `undef` so every load has a value even if no store
/// reaches it.
fn rename_slot(
    func: &mut Function,
    dom: &DomInfo,
    slot: &SlotSites,
    phi_index: &std::collections::HashMap<Block, usize>,
    undef: Value,
    replace: &mut std::collections::HashMap<Value, Value>,
) {
    enum Frame {
        Enter(Block),
        Exit(usize),
    }
    let mut work = vec![Frame::Enter(func.entry)];
    let mut stack: Vec<Value> = vec![undef];
    while let Some(frame) = work.pop() {
        match frame {
            Frame::Enter(block) => {
                let mut pushed = 0;
                if let Some(&index) = phi_index.get(&block) {
                    stack.push(func.blocks[block].params[index]);
                    pushed += 1;
                }
                for i in 0..func.blocks[block].insts.len() {
                    let inst = func.blocks[block].insts[i];
                    match &func.insts[inst].kind {
                        InstKind::Load { addr, .. } if *addr == slot.addr => {
                            let result = func.insts[inst].result.expect("candidate load has a result");
                            replace.insert(result, *stack.last().unwrap());
                        }
                        InstKind::Store { addr, val, .. } if *addr == slot.addr => {
                            stack.push(*val);
                            pushed += 1;
                        }
                        _ => {}
                    }
                }
                let current = *stack.last().unwrap();
                set_successor_args(func, block, phi_index, current);
                work.push(Frame::Exit(pushed));
                for &child in dom.children[block.index()].iter().rev() {
                    work.push(Frame::Enter(child));
                }
            }
            Frame::Exit(pushed) => {
                for _ in 0..pushed {
                    stack.pop();
                }
            }
        }
    }
}

/// Overwrite the argument each edge out of `block` passes to a
/// successor phi. `extend_incoming_edges` put placeholders at the same
/// indices on every edge, so edges the walk cannot see (unreachable
/// predecessors) stay arity-correct.
fn set_successor_args(func: &mut Function, block: Block, phi_index: &std::collections::HashMap<Block, usize>, value: Value) {
    let Some(term) = func.blocks[block].term.as_mut() else {
        return;
    };
    match term {
        Terminator::Br { dest, args } => {
            if let Some(&index) = phi_index.get(dest) {
                if let Some(arg) = args.get_mut(index) {
                    *arg = value;
                }
            }
        }
        Terminator::CondBr {
            then_block,
            then_args,
            else_block,
            else_args,
            ..
        } => {
            if let Some(&index) = phi_index.get(then_block) {
                if let Some(arg) = then_args.get_mut(index) {
                    *arg = value;
                }
            }
            if let Some(&index) = phi_index.get(else_block) {
                if let Some(arg) = else_args.get_mut(index) {
                    *arg = value;
                }
            }
        }
        Terminator::Switch { .. } | Terminator::Return { .. } | Terminator::Unreachable => {}
    }
}

/// Append `placeholder` to every terminator argument list feeding
/// `target`, so adding one param keeps br/condbr arity. `place_phis`
/// never picks a `Switch` target, whose arms carry no args.
fn extend_incoming_edges(func: &mut Function, target: Block, placeholder: Value) {
    for idx in 0..func.blocks.len() {
        let block = Block::from_usize(idx);
        let Some(term) = func.blocks[block].term.as_mut() else {
            continue;
        };
        match term {
            Terminator::Br { dest, args } => {
                if *dest == target {
                    args.push(placeholder);
                }
            }
            Terminator::CondBr {
                then_block,
                then_args,
                else_block,
                else_args,
                ..
            } => {
                if *then_block == target {
                    then_args.push(placeholder);
                }
                if *else_block == target {
                    else_args.push(placeholder);
                }
            }
            Terminator::Switch { .. } | Terminator::Return { .. } | Terminator::Unreachable => {}
        }
    }
}

/// One `undef` value per pointee type, since replacements must carry
/// the promoted slot's type.
fn undef_value(func: &mut Function, ty: &Ty, cache: &mut Vec<(Ty, Value)>) -> Value {
    if let Some((_, value)) = cache.iter().find(|(cached, _)| cached == ty) {
        return *value;
    }
    let value = func.values.push(crate::ir::ValueData {
        ty: ty.clone(),
        def: Def::Const(Const::Undef),
    });
    cache.push((ty.clone(), value));
    value
}

/// Resolve replacement chains in place (`v2 -> v1`, `v1 -> x` becomes
/// `v2 -> x`) so one `remap_operands` sweep reaches final targets.
fn collapse_chains(map: &mut std::collections::HashMap<Value, Value>) {
    let keys: Vec<Value> = map.keys().copied().collect();
    for key in keys {
        let mut target = map[&key];
        let mut steps = 0;
        while target != key {
            match map.get(&target).copied() {
                Some(next) if next != target => {
                    target = next;
                    steps += 1;
                    if steps > map.len() {
                        break;
                    }
                }
                _ => break,
            }
        }
        map.insert(key, target);
    }
}

/// Blocks targeted by a `Switch` terminator: params there cannot be
/// fed (switch arms carry no arguments) and the backend rejects them.
fn switch_targets(func: &Function) -> std::collections::HashSet<Block> {
    let mut targets = std::collections::HashSet::new();
    for block in func.blocks.iter() {
        if let Some(Terminator::Switch { cases, default, .. }) = &block.term {
            targets.insert(*default);
            for (_, dest) in cases {
                targets.insert(*dest);
            }
        }
    }
    targets
}

/// Immediate dominators, dominator-tree children, dominance
/// frontiers, and predecessors over the blocks reachable from the
/// entry. `preds`/`idom` serve loop analysis (LICM back edges, SCCP
/// is flow-based and needs none).
struct DomInfo {
    /// Reachable predecessors of each block, one per branch arm.
    preds: Vec<Vec<Block>>,
    /// Immediate dominator of every reachable block except the entry.
    idom: Vec<Option<Block>>,
    children: Vec<Vec<Block>>,
    df: Vec<Vec<Block>>,
}

impl DomInfo {
    fn compute(func: &Function, reachable: &[bool]) -> DomInfo {
        let count = func.blocks.len();
        let rpo = reverse_postorder(func, reachable);
        let mut rpo_index = vec![usize::MAX; count];
        for (i, &block) in rpo.iter().enumerate() {
            rpo_index[block.index()] = i;
        }

        let mut preds: Vec<Vec<Block>> = vec![Vec::new(); count];
        for &block in &rpo {
            if let Some(term) = &func.blocks[block].term {
                for succ in term.successors() {
                    if reachable[succ.index()] && !preds[succ.index()].contains(&block) {
                        preds[succ.index()].push(block);
                    }
                }
            }
        }

        // Cooper/Harvey/Kennedy: intersect processed predecessors in
        // reverse postorder until the immediate dominators settle.
        let mut idom: Vec<Option<Block>> = vec![None; count];
        let mut changed = true;
        while changed {
            changed = false;
            for &block in rpo.iter().skip(1) {
                let mut new: Option<Block> = None;
                for &pred in &preds[block.index()] {
                    if pred != func.entry && idom[pred.index()].is_none() {
                        continue;
                    }
                    new = Some(match new {
                        None => pred,
                        Some(current) => intersect(pred, current, &idom, &rpo_index, func.entry),
                    });
                }
                if let Some(new) = new {
                    if idom[block.index()] != Some(new) {
                        idom[block.index()] = Some(new);
                        changed = true;
                    }
                }
            }
        }

        let mut children = vec![Vec::new(); count];
        for &block in rpo.iter().skip(1) {
            if let Some(parent) = idom[block.index()] {
                children[parent.index()].push(block);
            }
        }

        // DF(x): a join is in DF(x) when x dominates one of the join's
        // predecessors but not the join itself; the dominator walk
        // visits exactly the x's between the two.
        let mut df = vec![Vec::new(); count];
        for &join in &rpo {
            if preds[join.index()].len() < 2 {
                continue;
            }
            for &pred in &preds[join.index()] {
                let mut runner = pred;
                while Some(runner) != idom[join.index()] {
                    df[runner.index()].push(join);
                    if runner == func.entry {
                        break;
                    }
                    runner = idom[runner.index()].unwrap_or(func.entry);
                }
            }
        }

        DomInfo { preds, idom, children, df }
    }
}

/// Climb two dominators until they meet (RPO indices shrink upward).
fn intersect(mut a: Block, mut b: Block, idom: &[Option<Block>], rpo_index: &[usize], entry: Block) -> Block {
    while a != b {
        while rpo_index[a.index()] > rpo_index[b.index()] {
            a = idom[a.index()].unwrap_or(entry);
        }
        while rpo_index[b.index()] > rpo_index[a.index()] {
            b = idom[b.index()].unwrap_or(entry);
        }
    }
    a
}

/// DFS postorder from the entry, reversed; unreachable blocks stay out.
fn reverse_postorder(func: &Function, reachable: &[bool]) -> Vec<Block> {
    enum Step {
        Enter(Block),
        Leave(Block),
    }
    let mut post = Vec::new();
    let mut seen = vec![false; func.blocks.len()];
    let mut work = vec![Step::Enter(func.entry)];
    seen[func.entry.index()] = true;
    while let Some(step) = work.pop() {
        match step {
            Step::Enter(block) => {
                work.push(Step::Leave(block));
                if let Some(term) = &func.blocks[block].term {
                    for succ in term.successors().iter().rev() {
                        if reachable[succ.index()] && !seen[succ.index()] {
                            seen[succ.index()] = true;
                            work.push(Step::Enter(*succ));
                        }
                    }
                }
            }
            Step::Leave(block) => post.push(block),
        }
    }
    post.reverse();
    post
}

/// Drop blocks unreachable from the entry and collapse branch chains.
#[derive(Default)]
pub struct SimplifyCfg;

impl FunctionPass for SimplifyCfg {
    fn name(&self) -> &'static str {
        "simplify-cfg"
    }

    fn run(&mut self, func: &mut Function) -> bool {
        let mut changed = false;
        let reachable = func.reachable_blocks();
        for (idx, live) in reachable.iter().enumerate() {
            if *live {
                continue;
            }
            let block = crate::ir::Block::from_usize(idx);
            let cleared = func.blocks[block].insts.is_empty() && matches!(func.blocks[block].term, Some(Terminator::Unreachable));
            if cleared {
                continue;
            }
            func.blocks[block].insts.clear();
            func.blocks[block].term = Some(Terminator::Unreachable);
            changed = true;
        }
        changed |= merge_single_pred_blocks(func);
        changed |= sink_pure_instructions(func);
        changed
    }
}

/// Incoming-edge counts per block, one per branch arm.
fn predecessor_counts(func: &Function) -> Vec<usize> {
    let mut preds = vec![0usize; func.blocks.len()];
    for block in func.blocks.iter() {
        if let Some(term) = &block.term {
            for succ in term.successors() {
                preds[succ.index()] += 1;
            }
        }
    }
    preds
}

/// Sink a block into its only predecessor across an unconditional
/// branch: the predecessor takes over the target's body, params and
/// terminator, with the target's params replaced by the branch args.
/// Iterating this removes branches the folders have made redundant and
/// the trampoline blocks mem2reg leaves around dead edges.
fn merge_single_pred_blocks(func: &mut Function) -> bool {
    let mut changed = false;
    loop {
        let preds = predecessor_counts(func);
        let mut merged = false;
        for idx in 0..func.blocks.len() {
            let pred = Block::from_usize(idx);
            let Some(Terminator::Br { dest, args }) = func.blocks[pred].term.clone() else {
                continue;
            };
            // The entry must stay the entry, and a shared or
            // self-referencing target cannot be sunk.
            if dest == pred || dest == func.entry || preds[dest.index()] != 1 {
                continue;
            }
            if func.blocks[dest].term.is_none() || func.blocks[dest].params.len() != args.len() {
                continue;
            }
            let params = func.blocks[dest].params.clone();
            let map: std::collections::HashMap<Value, Value> = params.iter().copied().zip(args.iter().copied()).collect();
            func.remap_operands(&map);
            func.blocks[dest].params.clear();
            let insts = std::mem::take(&mut func.blocks[dest].insts);
            for inst in &insts {
                func.insts[*inst].block = pred;
            }
            func.blocks[pred].insts.extend(insts);
            func.blocks[pred].term = func.blocks[dest].term.take();
            func.blocks[dest].term = Some(Terminator::Unreachable);
            changed = true;
            merged = true;
        }
        if !merged {
            return changed;
        }
    }
}

/// For every value with uses, the one block all of them live in —
/// `None` when they are spread over several blocks. Terminator
/// operands count as uses in the branching block, where they are
/// evaluated.
fn sole_use_blocks(func: &Function) -> std::collections::HashMap<Value, Option<Block>> {
    fn note(uses: &mut std::collections::HashMap<Value, Option<Block>>, v: Value, block: Block) {
        use std::collections::hash_map::Entry;
        match uses.entry(v) {
            Entry::Vacant(e) => {
                e.insert(Some(block));
            }
            Entry::Occupied(mut e) => {
                if *e.get() != Some(block) {
                    e.insert(None);
                }
            }
        }
    }
    let mut uses = std::collections::HashMap::new();
    for (idx, block) in func.blocks.iter().enumerate() {
        let here = Block::from_usize(idx);
        for inst in &block.insts {
            func.insts[*inst].kind.for_each_operand(&mut |v| note(&mut uses, v, here));
        }
        if let Some(term) = &block.term {
            term.for_each_operand(&mut |v| note(&mut uses, v, here));
        }
    }
    uses
}

/// Move pure instructions into the successor that holds every use: a
/// computation only one arm of a branch needs should not run on the
/// other. Only a target whose sole predecessor is the current block
/// qualifies, so the instruction cannot start running more often than
/// before (a loop body reached from several blocks is off limits).
fn sink_pure_instructions(func: &mut Function) -> bool {
    let mut changed = false;
    loop {
        let preds = predecessor_counts(func);
        let uses = sole_use_blocks(func);
        let mut moved = false;
        for idx in 0..func.blocks.len() {
            let block = Block::from_usize(idx);
            let Some(term) = func.blocks[block].term.clone() else {
                continue;
            };
            let sinks: Vec<Block> = term.successors().into_iter().filter(|d| *d != block && *d != func.entry && preds[d.index()] == 1).collect();
            if sinks.is_empty() {
                continue;
            }
            let insts = std::mem::take(&mut func.blocks[block].insts);
            let mut kept: Vec<Inst> = Vec::with_capacity(insts.len());
            // Back to front, so a value whose only user just sank is
            // considered again on the next round.
            for inst in insts.into_iter().rev() {
                let dest = func.insts[inst]
                    .result
                    .and_then(|result| uses.get(&result).copied().flatten())
                    .filter(|dest| sinks.contains(dest));
                match dest {
                    Some(dest) if func.insts[inst].kind.is_pure() => {
                        func.insts[inst].block = dest;
                        // Front insertion plus reverse iteration keeps
                        // the original order within `dest`.
                        func.blocks[dest].insts.insert(0, inst);
                        moved = true;
                    }
                    _ => kept.push(inst),
                }
            }
            kept.reverse();
            func.blocks[block].insts = kept;
        }
        if !moved {
            return changed;
        }
        changed = true;
    }
}

/// Inline direct calls.
///
/// Two tiers share one mechanism:
///
/// - Heuristic: small single-block, single-return callees up to
///   `max_insts` inline whether or not they are marked `inline`.
/// - Keyword: callees with `Function::inline` (the C `inline` hint
///   from the frontend) inline up to `max_inline_insts`, including
///   multi-block bodies via return-block rewiring through block
///   params.
///
/// Refuses variadic callees, (mutually) recursive calls, frame-bound
/// `Va*` or `GetFramePointer` bodies, and arity/void mismatches.
#[derive(Debug, Clone)]
pub struct Inline {
    pub max_insts: usize,
    pub max_inline_insts: usize,
}

impl Default for Inline {
    fn default() -> Self {
        Inline::small()
    }
}

impl Inline {
    pub fn small() -> Self {
        Inline {
            max_insts: 24,
            max_inline_insts: 256,
        }
    }

    /// Keyword-only inlining: no heuristic budget, so plain calls
    /// never inline while `inline`-marked callees still do. The
    /// driver uses this at `-O0`.
    pub fn inline_only() -> Self {
        Inline {
            max_insts: 0,
            max_inline_insts: 256,
        }
    }
}

impl ModulePass for Inline {
    fn name(&self) -> &'static str {
        "inline"
    }

    fn run(&mut self, module: &mut Module) -> bool {
        let mut changed = false;
        let func_ids: Vec<_> = (0..module.funcs.len()).map(crate::ir::Func::from_usize).collect();
        for caller_id in func_ids {
            // Collect call sites up front, then re-derive each site's
            // block/args/result at processing time: inlining one site
            // (especially multi-block, which splits the block) moves
            // later sites, so stored blocks go stale.
            let sites: Vec<crate::ir::Inst> = module.funcs[caller_id]
                .blocks
                .iter()
                .flat_map(|b| b.insts.clone())
                .filter(|inst| matches!(module.funcs[caller_id].insts[*inst].kind, InstKind::Call { .. }))
                .collect();
            for site in sites {
                let (block, callee, args, result) = {
                    let caller = &module.funcs[caller_id];
                    if site.index() >= caller.insts.len() {
                        continue;
                    }
                    let data = &caller.insts[site];
                    // The site may have been removed by an earlier
                    // inlining in this same sweep.
                    if !caller.blocks[data.block].insts.contains(&site) {
                        continue;
                    }
                    match &data.kind {
                        InstKind::Call { callee, args } => (data.block, *callee, args.iter().copied().collect::<Vec<_>>(), data.result),
                        _ => continue,
                    }
                };
                if callee.index() >= module.funcs.len() || callee == caller_id {
                    continue;
                }
                // Hard opt-outs: `noinline` is honored as written, and
                // inlining a `setjmp` call site into this caller would
                // move the re-entry point with it without marking the
                // caller, whose promoted slots are then stale after
                // `longjmp`.
                if module.funcs[callee].noinline || module.funcs[callee].returns_twice {
                    continue;
                }
                let marked = module.funcs[callee].inline;
                let total: usize = module.funcs[callee].blocks.iter().map(|b| b.insts.len()).sum();
                let limit = if marked { self.max_inline_insts } else { self.max_insts };
                if total > limit {
                    continue;
                }
                let single_block = module.funcs[callee].blocks.len() == 1;
                if !single_block && !marked {
                    continue;
                }
                if !callee_eligible(module, caller_id, callee, &args, result) {
                    continue;
                }
                if single_block {
                    if inline_single_block(module, caller_id, site, block, callee, &args, result) {
                        changed = true;
                    }
                } else if inline_multi_block(module, caller_id, site, callee, &args, result) {
                    changed = true;
                }
            }
        }
        changed
    }
}

/// Whether a callee may inline into a caller at all (before the
/// single/multi-block split): no variadics, no frame-bound `Va*`
/// bodies, no direct or mutual recursion, matching arity, matching
/// voidness, every block terminated, and at least one `Return`.
fn callee_eligible(module: &Module, caller: crate::ir::Func, callee: crate::ir::Func, args: &[Value], result: Option<Value>) -> bool {
    let f = &module.funcs[callee];
    if f.sig.variadic {
        return false;
    }
    if f.blocks[f.entry].params.len() != args.len() {
        return false;
    }
    if f.blocks.is_empty() {
        return false;
    }
    let ret_void = f.sig.ret.is_void();
    if ret_void && result.is_some() {
        return false;
    }
    let mut saw_return = false;
    for block in f.blocks.iter() {
        match &block.term {
            Some(Terminator::Return { val }) => {
                saw_return = true;
                if val.is_some() == ret_void {
                    return false;
                }
            }
            Some(Terminator::Br { .. }) | Some(Terminator::CondBr { .. }) | Some(Terminator::Switch { .. }) | Some(Terminator::Unreachable) => {}
            None => return false,
        }
    }
    if !saw_return {
        return false;
    }
    if f.insts.iter().any(|inst| {
        matches!(
            inst.kind,
            InstKind::VaStart { .. }
                | InstKind::VaArg { .. }
                | InstKind::VaEnd { .. }
                | InstKind::VaCopy { .. }
                // The frame pointer names the callee's own frame, so
                // inlining would re-parent it onto the caller.
                | InstKind::GetFramePointer
        )
    }) {
        return false;
    }
    if calls_into(f, callee) || calls_into(f, caller) {
        return false;
    }
    true
}

/// Splice a single-block, single-return callee into the call site.
fn inline_single_block(
    module: &mut Module,
    caller_id: crate::ir::Func,
    site: crate::ir::Inst,
    block: crate::ir::Block,
    callee: crate::ir::Func,
    args: &[Value],
    result: Option<Value>,
) -> bool {
    let (ret_val, callee_insts) = {
        let f = &module.funcs[callee];
        let Terminator::Return { val } = f.blocks[f.entry].term.clone().unwrap_or(Terminator::Unreachable) else {
            return false;
        };
        (val, f.blocks[f.entry].insts.clone())
    };
    if ret_val.is_none() && result.is_some() {
        return false;
    }
    let mut map = std::collections::HashMap::new();
    {
        let callee_func = &module.funcs[callee];
        for (i, param) in callee_func.blocks[callee_func.entry].params.clone().into_iter().enumerate() {
            if let Some(arg) = args.get(i) {
                map.insert(param, *arg);
            } else {
                return false;
            }
        }
    }
    let mut cloned: Vec<(crate::ir::Inst, crate::ir::Inst)> = Vec::new();
    let callee_snapshot = module.funcs[callee].clone();
    for src in callee_insts {
        let data = callee_snapshot.insts[src].clone();
        let caller = &mut module.funcs[caller_id];
        let new_result = data.result.map(|old| {
            let ty = callee_snapshot.values[old].ty.clone();
            let next_inst = crate::ir::Inst::from_usize(caller.insts.len());
            let fresh = caller.values.push(crate::ir::ValueData { ty, def: Def::Inst(next_inst) });
            map.insert(old, fresh);
            fresh
        });
        let new_inst = caller.insts.push(crate::ir::InstData {
            block,
            kind: data.kind,
            result: new_result,
            loc: data.loc,
        });
        if let Some(v) = new_result {
            caller.values[v].def = Def::Inst(new_inst);
        }
        cloned.push((src, new_inst));
    }
    // The clone order is not a def-before-use order: a use can precede the
    // definition of its operand. Remap operands once every callee value has
    // a mapping.
    for (src, new_inst) in &cloned {
        let caller = &mut module.funcs[caller_id];
        let kind = remap_operands_materialized(caller, &callee_snapshot, &mut map, &callee_snapshot.insts[*src].kind);
        caller.insts[*new_inst].kind = kind;
    }
    let returned = ret_val.map(|v| {
        let caller = &mut module.funcs[caller_id];
        materialize_operand(caller, &callee_snapshot, &mut map, v)
    });
    if let (Some(old_result), Some(inlined)) = (result, returned) {
        module.funcs[caller_id].replace_uses(old_result, inlined);
    }
    let call_pos = module.funcs[caller_id].blocks[block].insts.iter().position(|i| *i == site).unwrap_or(0);
    // The site may have moved under us; confirm it is still there.
    if !module.funcs[caller_id].blocks[block].insts.contains(&site) {
        return false;
    }
    module.funcs[caller_id].blocks[block].insts.retain(|i| *i != site);
    let block_insts = &mut module.funcs[caller_id].blocks[block].insts;
    let insert_at = call_pos.min(block_insts.len());
    for (offset, (_, fresh)) in cloned.iter().enumerate() {
        block_insts.insert(insert_at + offset, *fresh);
    }
    true
}

/// Inline a multi-block callee: split the call block into a pre-call
/// block and a post-call join, clone every callee block, and rewire
/// each cloned `Return` to the join (carrying the return value as a
/// block param when the call site keeps one).
fn inline_multi_block(module: &mut Module, caller_id: crate::ir::Func, site: crate::ir::Inst, callee: crate::ir::Func, args: &[Value], result: Option<Value>) -> bool {
    let snapshot = module.funcs[callee].clone();
    let entry_params = snapshot.blocks[snapshot.entry].params.clone();
    if entry_params.len() != args.len() {
        return false;
    }
    let ret_void = snapshot.sig.ret.is_void();
    if ret_void && result.is_some() {
        return false;
    }
    let need_post_param = result.is_some() && !ret_void;
    let ret_ty = result.map(|v| module.funcs[caller_id].values[v].ty.clone());
    // Resolve the live call block: earlier inlinings may have moved
    // this site.
    let call_block = module.funcs[caller_id].insts[site].block;
    let call_pos = match module.funcs[caller_id].blocks[call_block].insts.iter().position(|i| *i == site) {
        Some(pos) => pos,
        None => return false,
    };
    let suffix: Vec<crate::ir::Inst> = module.funcs[caller_id].blocks[call_block].insts[call_pos + 1..].to_vec();
    let orig_term = module.funcs[caller_id].blocks[call_block].term.clone();
    let caller = &mut module.funcs[caller_id];
    let post = if need_post_param {
        caller.push_block(vec![ret_ty.clone().expect("return type for valued call")])
    } else {
        caller.push_block(vec![])
    };
    let ret_param = caller.blocks[post].params.first().copied();
    for s in &suffix {
        caller.insts[*s].block = post;
    }
    caller.blocks[post].insts = suffix;
    caller.blocks[post].term = orig_term;
    caller.blocks[call_block].insts.truncate(call_pos);
    if let (Some(old), Some(rp)) = (result, ret_param) {
        caller.replace_uses(old, rp);
    }
    let mut map = std::collections::HashMap::new();
    for (param, arg) in entry_params.iter().zip(args.iter()) {
        map.insert(*param, *arg);
    }
    let mut block_map = std::collections::HashMap::new();
    for idx in 0..snapshot.blocks.len() {
        let old_b = crate::ir::Block::from_usize(idx);
        let new_b = if old_b == snapshot.entry {
            caller.push_block(vec![])
        } else {
            let tys: Vec<_> = snapshot.blocks[old_b].params.iter().map(|v| snapshot.values[*v].ty.clone()).collect();
            let nb = caller.push_block(tys);
            for (old_p, new_p) in snapshot.blocks[old_b].params.iter().zip(caller.blocks[nb].params.clone()) {
                map.insert(*old_p, new_p);
            }
            nb
        };
        block_map.insert(old_b, new_b);
    }
    let cloned_entry = block_map[&snapshot.entry];
    caller.blocks[call_block].term = Some(Terminator::Br {
        dest: cloned_entry,
        args: Default::default(),
    });
    let mut cloned: Vec<(crate::ir::Inst, crate::ir::Inst)> = Vec::new();
    for idx in 0..snapshot.blocks.len() {
        let old_b = crate::ir::Block::from_usize(idx);
        let new_b = block_map[&old_b];
        for src in snapshot.blocks[old_b].insts.clone() {
            let data = snapshot.insts[src].clone();
            let new_result = data.result.map(|old| {
                let ty = snapshot.values[old].ty.clone();
                let next = crate::ir::Inst::from_usize(caller.insts.len());
                let fresh = caller.values.push(crate::ir::ValueData { ty, def: Def::Inst(next) });
                map.insert(old, fresh);
                fresh
            });
            let fresh = caller.insts.push(crate::ir::InstData {
                block: new_b,
                kind: data.kind,
                result: new_result,
                loc: data.loc,
            });
            if let Some(v) = new_result {
                caller.values[v].def = Def::Inst(fresh);
            }
            caller.blocks[new_b].insts.push(fresh);
            cloned.push((src, fresh));
        }
    }
    // Blocks are cloned in arena order, which is not a def-before-use order:
    // earlier inlinings move a block's tail into a fresh successor block, so a
    // use can precede the definition of its operand here. Remap operands once
    // every callee value has a mapping.
    // Blocks are cloned in arena order, which is not a def-before-use order:
    // earlier inlinings move a block's tail into a fresh successor block, so a
    // use can precede the definition of its operand here. Remap operands once
    // every callee value has a mapping.
    for (src, fresh) in cloned {
        let kind = remap_operands_materialized(caller, &snapshot, &mut map, &snapshot.insts[src].kind);
        caller.insts[fresh].kind = kind;
    }
    for idx in 0..snapshot.blocks.len() {
        let old_b = crate::ir::Block::from_usize(idx);
        let new_b = block_map[&old_b];
        let term = snapshot.blocks[old_b].term.clone();
        let next = match term {
            None => None,
            Some(Terminator::Return { val }) => {
                let mut out: smallvec::SmallVec<[Value; 4]> = smallvec::SmallVec::new();
                if need_post_param {
                    let v = val.expect("valued return for valued call");
                    out.push(materialize_operand(caller, &snapshot, &mut map, v));
                }
                Some(Terminator::Br { dest: post, args: out })
            }
            Some(Terminator::Br { dest, args }) => {
                let nd = block_map[&dest];
                let na = args.iter().map(|a| materialize_operand(caller, &snapshot, &mut map, *a)).collect();
                Some(Terminator::Br { dest: nd, args: na })
            }
            Some(Terminator::CondBr {
                cond,
                then_block,
                else_block,
                then_args,
                else_args,
            }) => {
                let nc = materialize_operand(caller, &snapshot, &mut map, cond);
                let nt = block_map[&then_block];
                let ne = block_map[&else_block];
                let nta = then_args.iter().map(|a| materialize_operand(caller, &snapshot, &mut map, *a)).collect();
                let nea = else_args.iter().map(|a| materialize_operand(caller, &snapshot, &mut map, *a)).collect();
                Some(Terminator::CondBr {
                    cond: nc,
                    then_block: nt,
                    else_block: ne,
                    then_args: nta,
                    else_args: nea,
                })
            }
            Some(Terminator::Switch { scrutinee, cases, default }) => {
                let ns = materialize_operand(caller, &snapshot, &mut map, scrutinee);
                let ncases = cases.iter().map(|(k, b)| (*k, block_map[b])).collect();
                Some(Terminator::Switch {
                    scrutinee: ns,
                    cases: ncases,
                    default: block_map[&default],
                })
            }
            Some(Terminator::Unreachable) => Some(Terminator::Unreachable),
        };
        caller.blocks[new_b].term = next;
    }
    true
}

fn calls_into(func: &crate::ir::Function, target: crate::ir::Func) -> bool {
    func.insts.iter().any(|inst| match &inst.kind {
        InstKind::Call { callee, .. } => *callee == target,
        _ => false,
    })
}

/// Map a callee operand into the caller, materializing constants fresh.
/// Value arenas are per-function, so reusing a callee index directly would
/// alias an unrelated caller value.
fn materialize_operand(caller: &mut crate::ir::Function, snapshot: &crate::ir::Function, map: &mut std::collections::HashMap<Value, Value>, v: Value) -> Value {
    if let Some(mapped) = map.get(&v) {
        return *mapped;
    }
    if v.index() >= snapshot.values.len() {
        return v;
    }
    match &snapshot.values[v].def {
        Def::Const(c) => {
            // Re-type the clone from the MIR value, not the constant:
            // `Const::Undef.ty()` is `Void` and would erase, e.g., the
            // scalar type of a Mem2Reg phi undef.
            let ty = snapshot.values[v].ty.clone();
            let fresh = caller.values.push(crate::ir::ValueData { ty, def: Def::Const(c.clone()) });
            map.insert(v, fresh);
            fresh
        }
        _ => panic!("callee value {} has no mapping during inlining", v.index()),
    }
}

fn remap_operands_materialized(caller: &mut crate::ir::Function, snapshot: &crate::ir::Function, map: &mut std::collections::HashMap<Value, Value>, kind: &InstKind) -> InstKind {
    match kind {
        InstKind::Add(a, b) => {
            let (aa, bb) = (*a, *b);
            let x = materialize_operand(caller, snapshot, map, aa);
            let y = materialize_operand(caller, snapshot, map, bb);
            InstKind::Add(x, y)
        }
        InstKind::Sub(a, b) => {
            let (aa, bb) = (*a, *b);
            let x = materialize_operand(caller, snapshot, map, aa);
            let y = materialize_operand(caller, snapshot, map, bb);
            InstKind::Sub(x, y)
        }
        InstKind::Mul(a, b) => {
            let (aa, bb) = (*a, *b);
            let x = materialize_operand(caller, snapshot, map, aa);
            let y = materialize_operand(caller, snapshot, map, bb);
            InstKind::Mul(x, y)
        }
        InstKind::Udiv(a, b) => {
            let (aa, bb) = (*a, *b);
            let x = materialize_operand(caller, snapshot, map, aa);
            let y = materialize_operand(caller, snapshot, map, bb);
            InstKind::Udiv(x, y)
        }
        InstKind::Sdiv(a, b) => {
            let (aa, bb) = (*a, *b);
            let x = materialize_operand(caller, snapshot, map, aa);
            let y = materialize_operand(caller, snapshot, map, bb);
            InstKind::Sdiv(x, y)
        }
        InstKind::Urem(a, b) => {
            let (aa, bb) = (*a, *b);
            let x = materialize_operand(caller, snapshot, map, aa);
            let y = materialize_operand(caller, snapshot, map, bb);
            InstKind::Urem(x, y)
        }
        InstKind::Srem(a, b) => {
            let (aa, bb) = (*a, *b);
            let x = materialize_operand(caller, snapshot, map, aa);
            let y = materialize_operand(caller, snapshot, map, bb);
            InstKind::Srem(x, y)
        }
        InstKind::And(a, b) => {
            let (aa, bb) = (*a, *b);
            let x = materialize_operand(caller, snapshot, map, aa);
            let y = materialize_operand(caller, snapshot, map, bb);
            InstKind::And(x, y)
        }
        InstKind::Or(a, b) => {
            let (aa, bb) = (*a, *b);
            let x = materialize_operand(caller, snapshot, map, aa);
            let y = materialize_operand(caller, snapshot, map, bb);
            InstKind::Or(x, y)
        }
        InstKind::Xor(a, b) => {
            let (aa, bb) = (*a, *b);
            let x = materialize_operand(caller, snapshot, map, aa);
            let y = materialize_operand(caller, snapshot, map, bb);
            InstKind::Xor(x, y)
        }
        InstKind::Shl(a, b) => {
            let (aa, bb) = (*a, *b);
            let x = materialize_operand(caller, snapshot, map, aa);
            let y = materialize_operand(caller, snapshot, map, bb);
            InstKind::Shl(x, y)
        }
        InstKind::LShr(a, b) => {
            let (aa, bb) = (*a, *b);
            let x = materialize_operand(caller, snapshot, map, aa);
            let y = materialize_operand(caller, snapshot, map, bb);
            InstKind::LShr(x, y)
        }
        InstKind::AShr(a, b) => {
            let (aa, bb) = (*a, *b);
            let x = materialize_operand(caller, snapshot, map, aa);
            let y = materialize_operand(caller, snapshot, map, bb);
            InstKind::AShr(x, y)
        }
        InstKind::Neg(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::Neg(x)
        }
        InstKind::Not(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::Not(x)
        }
        InstKind::FAdd(a, b) => {
            let (aa, bb) = (*a, *b);
            let x = materialize_operand(caller, snapshot, map, aa);
            let y = materialize_operand(caller, snapshot, map, bb);
            InstKind::FAdd(x, y)
        }
        InstKind::FSub(a, b) => {
            let (aa, bb) = (*a, *b);
            let x = materialize_operand(caller, snapshot, map, aa);
            let y = materialize_operand(caller, snapshot, map, bb);
            InstKind::FSub(x, y)
        }
        InstKind::FMul(a, b) => {
            let (aa, bb) = (*a, *b);
            let x = materialize_operand(caller, snapshot, map, aa);
            let y = materialize_operand(caller, snapshot, map, bb);
            InstKind::FMul(x, y)
        }
        InstKind::FDiv(a, b) => {
            let (aa, bb) = (*a, *b);
            let x = materialize_operand(caller, snapshot, map, aa);
            let y = materialize_operand(caller, snapshot, map, bb);
            InstKind::FDiv(x, y)
        }
        InstKind::FNeg(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::FNeg(x)
        }
        InstKind::Icmp { pred, lhs, rhs } => {
            let (l, r, p) = (*lhs, *rhs, *pred);
            let x = materialize_operand(caller, snapshot, map, l);
            let y = materialize_operand(caller, snapshot, map, r);
            InstKind::Icmp { pred: p, lhs: x, rhs: y }
        }
        InstKind::Fcmp { pred, lhs, rhs } => {
            let (l, r, p) = (*lhs, *rhs, *pred);
            let x = materialize_operand(caller, snapshot, map, l);
            let y = materialize_operand(caller, snapshot, map, r);
            InstKind::Fcmp { pred: p, lhs: x, rhs: y }
        }
        other => remap_simple_materialized(caller, snapshot, map, other),
    }
}

fn remap_simple_materialized(caller: &mut crate::ir::Function, snapshot: &crate::ir::Function, map: &mut std::collections::HashMap<Value, Value>, kind: &InstKind) -> InstKind {
    match kind {
        InstKind::Trunc(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::Trunc(x)
        }
        InstKind::Zext(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::Zext(x)
        }
        InstKind::Sext(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::Sext(x)
        }
        InstKind::Bitcast(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::Bitcast(x)
        }
        InstKind::SItoF(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::SItoF(x)
        }
        InstKind::UItoF(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::UItoF(x)
        }
        InstKind::SFtoI(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::SFtoI(x)
        }
        InstKind::UFtoI(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::UFtoI(x)
        }
        InstKind::Fptrunc(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::Fptrunc(x)
        }
        InstKind::Fpext(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::Fpext(x)
        }
        InstKind::PtrToInt(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::PtrToInt(x)
        }
        InstKind::IntToPtr(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::IntToPtr(x)
        }
        InstKind::Select { cond, then_val, else_val } => {
            let (c, t, e) = (*cond, *then_val, *else_val);
            let cc = materialize_operand(caller, snapshot, map, c);
            let tt = materialize_operand(caller, snapshot, map, t);
            let ee = materialize_operand(caller, snapshot, map, e);
            InstKind::Select {
                cond: cc,
                then_val: tt,
                else_val: ee,
            }
        }
        InstKind::Alloca { pointee, count } => InstKind::Alloca {
            pointee: pointee.clone(),
            count: *count,
        },
        InstKind::Load { addr, flags } => {
            let x = materialize_operand(caller, snapshot, map, *addr);
            InstKind::Load { addr: x, flags: *flags }
        }
        InstKind::Store { addr, val, flags } => {
            let (aa, vv) = (*addr, *val);
            let x = materialize_operand(caller, snapshot, map, aa);
            let y = materialize_operand(caller, snapshot, map, vv);
            InstKind::Store { addr: x, val: y, flags: *flags }
        }
        InstKind::GetElementPtr { base, indices } => {
            let bb = *base;
            let mut next: smallvec::SmallVec<[crate::ty::GepIndex; 4]> = smallvec::SmallVec::new();
            for idx in indices.iter() {
                match idx {
                    crate::ty::GepIndex::Index(v) => {
                        let m = materialize_operand(caller, snapshot, map, *v);
                        next.push(crate::ty::GepIndex::Index(m));
                    }
                    other => next.push(other.clone()),
                }
            }
            let b = materialize_operand(caller, snapshot, map, bb);
            InstKind::GetElementPtr { base: b, indices: next }
        }
        InstKind::ExtractValue { agg, index } => {
            let x = materialize_operand(caller, snapshot, map, *agg);
            InstKind::ExtractValue { agg: x, index: *index }
        }
        InstKind::InsertValue { agg, elem, index } => {
            let (aa, ee, ii) = (*agg, *elem, *index);
            let x = materialize_operand(caller, snapshot, map, aa);
            let y = materialize_operand(caller, snapshot, map, ee);
            InstKind::InsertValue { agg: x, elem: y, index: ii }
        }
        InstKind::Memcpy { dst, src, len } => {
            let (dd, ss, ll) = (*dst, *src, *len);
            let x = materialize_operand(caller, snapshot, map, dd);
            let y = materialize_operand(caller, snapshot, map, ss);
            InstKind::Memcpy { dst: x, src: y, len: ll }
        }
        InstKind::Memset { dst, byte, len } => {
            let (dd, bb, ll) = (*dst, *byte, *len);
            let x = materialize_operand(caller, snapshot, map, dd);
            InstKind::Memset { dst: x, byte: bb, len: ll }
        }
        InstKind::FuncAddr { func } => InstKind::FuncAddr { func: *func },
        InstKind::VaStart { ap } => {
            let x = materialize_operand(caller, snapshot, map, *ap);
            InstKind::VaStart { ap: x }
        }
        InstKind::VaArg { ap } => {
            let x = materialize_operand(caller, snapshot, map, *ap);
            InstKind::VaArg { ap: x }
        }
        InstKind::VaEnd { ap } => {
            let x = materialize_operand(caller, snapshot, map, *ap);
            InstKind::VaEnd { ap: x }
        }
        InstKind::VaCopy { dst, src } => {
            let (dd, ss) = (*dst, *src);
            let x = materialize_operand(caller, snapshot, map, dd);
            let y = materialize_operand(caller, snapshot, map, ss);
            InstKind::VaCopy { dst: x, src: y }
        }
        InstKind::GlobalAddr { global } => InstKind::GlobalAddr { global: *global },
        InstKind::Call { callee, args } => {
            let arg_vals: Vec<Value> = args.iter().copied().collect();
            let mut out: smallvec::SmallVec<[Value; 4]> = smallvec::SmallVec::new();
            for a in arg_vals {
                out.push(materialize_operand(caller, snapshot, map, a));
            }
            InstKind::Call { callee: *callee, args: out }
        }
        InstKind::CallExtern { target, args } => {
            let arg_vals: Vec<Value> = args.iter().copied().collect();
            let mut out: smallvec::SmallVec<[Value; 4]> = smallvec::SmallVec::new();
            for a in arg_vals {
                out.push(materialize_operand(caller, snapshot, map, a));
            }
            InstKind::CallExtern { target: *target, args: out }
        }
        InstKind::ExternAddr { target } => InstKind::ExternAddr { target: *target },
        InstKind::Ceil(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::Ceil(x)
        }
        InstKind::Floor(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::Floor(x)
        }
        InstKind::Sqrt(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::Sqrt(x)
        }
        InstKind::Clz(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::Clz(x)
        }
        InstKind::Ctz(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::Ctz(x)
        }
        InstKind::Popcnt(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::Popcnt(x)
        }
        InstKind::Bswap(a) => {
            let x = materialize_operand(caller, snapshot, map, *a);
            InstKind::Bswap(x)
        }
        InstKind::CallIndirect { callee, args, sig } => {
            let arg_vals: Vec<Value> = args.iter().copied().collect();
            let mut out: smallvec::SmallVec<[Value; 4]> = smallvec::SmallVec::new();
            for a in arg_vals {
                out.push(materialize_operand(caller, snapshot, map, a));
            }
            let addr = materialize_operand(caller, snapshot, map, *callee);
            InstKind::CallIndirect {
                callee: addr,
                args: out,
                sig: sig.clone(),
            }
        }
        InstKind::AtomicLoad { addr } => {
            let x = materialize_operand(caller, snapshot, map, *addr);
            InstKind::AtomicLoad { addr: x }
        }
        InstKind::AtomicStore { addr, val } => {
            let a = materialize_operand(caller, snapshot, map, *addr);
            let v = materialize_operand(caller, snapshot, map, *val);
            InstKind::AtomicStore { addr: a, val: v }
        }
        InstKind::AtomicRmw { op, addr, val } => {
            let a = materialize_operand(caller, snapshot, map, *addr);
            let v = materialize_operand(caller, snapshot, map, *val);
            InstKind::AtomicRmw { op: *op, addr: a, val: v }
        }
        InstKind::AtomicCas { addr, expected, replacement } => {
            let a = materialize_operand(caller, snapshot, map, *addr);
            let e = materialize_operand(caller, snapshot, map, *expected);
            let r = materialize_operand(caller, snapshot, map, *replacement);
            InstKind::AtomicCas {
                addr: a,
                expected: e,
                replacement: r,
            }
        }
        InstKind::AtomicFence => InstKind::AtomicFence,
        InstKind::InlineAsm { block } => {
            let mut block = block.clone();
            for operand in &mut block.operands {
                operand.value = materialize_operand(caller, snapshot, map, operand.value);
            }
            InstKind::InlineAsm { block }
        }
        InstKind::GetFramePointer => InstKind::GetFramePointer,
        other => panic!("inline of {other:?} needs operand mapping"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::FunctionBuilder;
    use crate::ir::{AsmBlock, AsmOperand, Module};
    use crate::ty::{Signature, Ty};

    fn const_func() -> Function {
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("consts", sig);
        let mut b = FunctionBuilder::new(&mut func, entry);
        let two = b.iconst(32, 2);
        let three = b.iconst(32, 3);
        let four = b.iconst(32, 4);
        let mul = b.mul(Ty::int(32), three, four);
        let sum = b.add(Ty::int(32), two, mul);
        b.ret(Some(sum));
        func
    }

    #[test]
    fn folds_constant_arithmetic() {
        let mut func = const_func();
        let mut fold = ConstFold;
        fold.run(&mut func);
        let term = func.blocks[func.entry].term.clone();
        assert!(matches!(term, Some(Terminator::Return { .. })));
        let Terminator::Return { val } = term.unwrap() else { unreachable!() };
        let c = func.const_value(val.unwrap()).unwrap();
        assert_eq!(c, &Const::Int { bits: 32, value: 14 });
    }

    #[test]
    fn folds_constant_branch() {
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("branch", sig);
        let then_b = func.push_block(vec![]);
        let else_b = func.push_block(vec![]);
        let one = func.push_const(Const::Int { bits: 1, value: 1 });
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond: one,
            then_block: then_b,
            else_block: else_b,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        let mut fold = ConstFold;
        assert!(fold.run(&mut func));
        assert!(matches!(func.blocks[entry].term, Some(Terminator::Br { .. })));
    }

    #[test]
    fn fold_replacement_keeps_result_type() {
        let ptr_ty = Ty::ptr(Ty::int(8));
        let sig = Signature::new(vec![], ptr_ty.clone());
        let (mut func, entry) = Function::new("sel", sig);
        let one = func.push_const(Const::Int { bits: 1, value: 1 });
        let undef = func.values.push(crate::ir::ValueData {
            ty: ptr_ty.clone(),
            def: Def::Const(Const::Undef),
        });
        let null = func.values.push(crate::ir::ValueData {
            ty: ptr_ty.clone(),
            def: Def::Const(Const::Null),
        });
        let result = func.values.push(crate::ir::ValueData {
            ty: ptr_ty.clone(),
            def: Def::Inst(crate::ir::Inst::from_usize(0)),
        });
        let inst = func.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Select {
                cond: one,
                then_val: undef,
                else_val: null,
            },
            result: Some(result),
            loc: None,
        });
        func.values[result].def = Def::Inst(inst);
        func.blocks[entry].insts.push(inst);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(result) });
        let mut fold = ConstFold;
        assert!(fold.run(&mut func));
        let Some(Terminator::Return { val: Some(val) }) = func.blocks[entry].term else {
            panic!("entry still returns");
        };
        assert_eq!(func.const_value(val), Some(&Const::Undef));
        assert_eq!(func.values[val].ty, ptr_ty);
    }

    #[test]
    fn removes_dead_pure_insts() {
        let mut func = const_func();
        let dead_ty = Ty::int(32);
        let x = func.push_const(Const::Int { bits: 32, value: 99 });
        let y = func.push_const(Const::Int { bits: 32, value: 1 });
        let dead_val = func.values.push(crate::ir::ValueData {
            ty: dead_ty.clone(),
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
        });
        let dead_inst = func.insts.push(crate::ir::InstData {
            block: func.entry,
            kind: InstKind::Add(x, y),
            result: Some(dead_val),
            loc: None,
        });
        func.values[dead_val].def = Def::Inst(dead_inst);
        func.blocks[func.entry].insts.push(dead_inst);
        let mut dce = DeadCodeElim;
        assert!(dce.run(&mut func));
        assert!(!dce.run(&mut func));
    }

    #[test]
    fn dce_keeps_unused_va_arg() {
        // An unused `VaArg` still consumes one argument, so removing it
        // would shift every later read of the list.
        let ap_ty = Ty::BuiltinVaList { size: 24, align: 8 };
        let sig = Signature::new(vec![Ty::ptr(ap_ty.clone())], Ty::Void);
        let (mut func, entry) = Function::new("va", sig);
        let ap = func.values.push(crate::ir::ValueData {
            ty: Ty::ptr(ap_ty),
            def: Def::Param(entry, 0),
        });
        func.blocks[entry].params.push(ap);
        let consumed = func.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::VaArg { ap },
            result: Some(consumed),
            loc: None,
        });
        func.values[consumed].def = Def::Inst(inst);
        func.blocks[entry].insts.push(inst);
        func.blocks[entry].term = Some(Terminator::Return { val: None });
        let mut dce = DeadCodeElim;
        assert!(!dce.run(&mut func), "an unused va_arg must survive DCE");
        assert!(crate::verify::verify_func(&func).is_empty());
    }

    #[test]
    fn drops_unreachable_blocks() {
        let sig = Signature::new(vec![], Ty::Void);
        let (mut func, entry) = Function::new("cfg", sig);
        let live = func.push_block(vec![]);
        let dead = func.push_block(vec![]);
        func.blocks[entry].term = Some(Terminator::Br {
            dest: live,
            args: Default::default(),
        });
        func.blocks[live].term = Some(Terminator::Return { val: None });
        func.blocks[dead].term = Some(Terminator::Return { val: None });
        let mut simp = SimplifyCfg;
        assert!(simp.run(&mut func));
        assert!(matches!(func.blocks[dead].term, Some(Terminator::Unreachable)));
    }

    #[test]
    fn merges_single_predecessor_chain() {
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("chain", sig);
        let mid = func.push_block(vec![]);
        let end = func.push_block(vec![Ty::int(32)]);
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let two = func.push_const(Const::Int { bits: 32, value: 2 });
        let sum = emit_bin(&mut func, mid, InstKind::Add(one, two), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Br {
            dest: mid,
            args: Default::default(),
        });
        func.blocks[mid].term = Some(Terminator::Br {
            dest: end,
            args: [sum].into_iter().collect(),
        });
        let q = func.blocks[end].params[0];
        func.blocks[end].term = Some(Terminator::Return { val: Some(q) });

        let mut simp = SimplifyCfg;
        assert!(simp.run(&mut func));
        assert!(crate::verify::verify_func(&func).is_empty());
        assert_eq!(func.blocks[entry].insts.len(), 1);
        let Terminator::Return { val: Some(v) } = func.blocks[entry].term.clone().unwrap() else {
            panic!("chain did not sink into the entry block");
        };
        assert_eq!(v, sum);
        assert!(matches!(func.blocks[mid].term, Some(Terminator::Unreachable)));
        assert!(matches!(func.blocks[end].term, Some(Terminator::Unreachable)));
        assert!(!simp.run(&mut func), "second run is a no-op");
    }

    #[test]
    fn keeps_shared_branch_targets() {
        let sig = Signature::new(vec![], Ty::Void);
        let (mut func, entry) = Function::new("diamond", sig);
        let then_block = func.push_block(vec![]);
        let else_block = func.push_block(vec![]);
        let merge = func.push_block(vec![]);
        let cond = func.push_const(Const::Int { bits: 1, value: 1 });
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block,
            else_block,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        func.blocks[then_block].term = Some(Terminator::Br {
            dest: merge,
            args: Default::default(),
        });
        func.blocks[else_block].term = Some(Terminator::Br {
            dest: merge,
            args: Default::default(),
        });
        func.blocks[merge].term = Some(Terminator::Return { val: None });

        let mut simp = SimplifyCfg;
        assert!(!simp.run(&mut func));
    }

    #[test]
    fn keeps_loop_headers_and_entries() {
        // The header has two preds, so the one-pred latch cannot sink
        // into it.
        let sig = Signature::new(vec![], Ty::Void);
        let (mut func, entry) = Function::new("loop", sig);
        let header = func.push_block(vec![]);
        let body = func.push_block(vec![]);
        let exit = func.push_block(vec![]);
        let cond = func.push_const(Const::Int { bits: 1, value: 1 });
        func.blocks[entry].term = Some(Terminator::Br {
            dest: header,
            args: Default::default(),
        });
        func.blocks[header].term = Some(Terminator::CondBr {
            cond,
            then_block: body,
            else_block: exit,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        func.blocks[body].term = Some(Terminator::Br {
            dest: header,
            args: Default::default(),
        });
        func.blocks[exit].term = Some(Terminator::Return { val: None });
        let mut simp = SimplifyCfg;
        assert!(!simp.run(&mut func));
    }

    #[test]
    fn keeps_entry_block() {
        // The only pred of the entry is a latch: sinking the entry into
        // it would change where execution starts.
        let sig = Signature::new(vec![], Ty::Void);
        let (mut func, entry) = Function::new("entry-loop", sig);
        let body = func.push_block(vec![]);
        let exit = func.push_block(vec![]);
        let cond = func.push_const(Const::Int { bits: 1, value: 1 });
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block: body,
            else_block: exit,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        func.blocks[body].term = Some(Terminator::Br {
            dest: entry,
            args: Default::default(),
        });
        func.blocks[exit].term = Some(Terminator::Return { val: None });
        let mut simp = SimplifyCfg;
        assert!(!simp.run(&mut func));
    }

    #[test]
    fn sinks_params_through_successors() {
        let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("sink", sig);
        let x = func.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Param(entry, 0),
        });
        func.blocks[entry].params.push(x);
        let mid = func.push_block(vec![Ty::int(32)]);
        let end = func.push_block(vec![Ty::int(32)]);
        func.blocks[entry].term = Some(Terminator::Br {
            dest: mid,
            args: [x].into_iter().collect(),
        });
        let p = func.blocks[mid].params[0];
        func.blocks[mid].term = Some(Terminator::Br {
            dest: end,
            args: [p].into_iter().collect(),
        });
        let q = func.blocks[end].params[0];
        func.blocks[end].term = Some(Terminator::Return { val: Some(q) });

        let mut simp = SimplifyCfg;
        assert!(simp.run(&mut func));
        assert!(crate::verify::verify_func(&func).is_empty());
        let Terminator::Return { val: Some(v) } = func.blocks[entry].term.clone().unwrap() else {
            panic!("sink did not collapse the chain");
        };
        assert_eq!(v, x, "param must resolve back to the entry argument");
    }

    #[test]
    fn sinks_branch_only_work_into_the_using_arm() {
        // Both computations feed only the `then` return; the `else`
        // path must not run them.
        let sig = Signature::new(vec![Ty::int(32), Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("sink", sig);
        let x = param(&mut func, entry, Ty::int(32));
        let take = param(&mut func, entry, Ty::Bool);
        let add = emit_bin(&mut func, entry, InstKind::Add(x, x), Ty::int(32));
        let mul = emit_bin(&mut func, entry, InstKind::Mul(add, add), Ty::int(32));
        let then_block = func.push_block(vec![]);
        let else_block = func.push_block(vec![]);
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond: take,
            then_block,
            else_block,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        func.blocks[then_block].term = Some(Terminator::Return { val: Some(mul) });
        let fallback = func.push_const(Const::Int { bits: 32, value: 123 });
        func.blocks[else_block].term = Some(Terminator::Return { val: Some(fallback) });

        let mut simp = SimplifyCfg;
        assert!(simp.run(&mut func));
        assert!(crate::verify::verify_func(&func).is_empty());
        assert!(func.blocks[entry].insts.is_empty(), "entry keeps only the branch");
        let kinds: Vec<InstKind> = func.blocks[then_block].insts.iter().map(|i| func.insts[*i].kind.clone()).collect();
        assert!(matches!(kinds[0], InstKind::Add(..)), "add sinks first: {kinds:?}");
        assert!(matches!(kinds[1], InstKind::Mul(..)), "mul follows its operand: {kinds:?}");
        assert!(!simp.run(&mut func), "second run is a no-op");
    }

    #[test]
    fn keeps_work_used_by_both_arms() {
        let sig = Signature::new(vec![Ty::int(32), Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(32));
        let take = param(&mut func, entry, Ty::Bool);
        let neg = emit_bin(&mut func, entry, InstKind::Neg(x), Ty::int(32));
        let then_block = func.push_block(vec![]);
        let else_block = func.push_block(vec![]);
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond: take,
            then_block,
            else_block,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        func.blocks[then_block].term = Some(Terminator::Return { val: Some(neg) });
        func.blocks[else_block].term = Some(Terminator::Return { val: Some(neg) });
        let mut simp = SimplifyCfg;
        assert!(!simp.run(&mut func));
        assert_eq!(func.blocks[entry].insts.len(), 1);
    }

    #[test]
    fn keeps_work_out_of_blocks_with_several_predecessors() {
        // `v` is needed inside the loop, but moving it to the header
        // would re-run it every iteration; it must stay in the
        // preheader.
        let sig = Signature::new(vec![Ty::int(32), Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(32));
        let cond = param(&mut func, entry, Ty::Bool);
        let v = emit_bin(&mut func, entry, InstKind::Add(x, x), Ty::int(32));
        let header = func.push_block(vec![]);
        let body = func.push_block(vec![]);
        let exit = func.push_block(vec![]);
        func.blocks[entry].term = Some(Terminator::Br {
            dest: header,
            args: Default::default(),
        });
        func.blocks[header].term = Some(Terminator::CondBr {
            cond,
            then_block: body,
            else_block: exit,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        emit_bin(&mut func, body, InstKind::Add(v, x), Ty::int(32));
        func.blocks[body].term = Some(Terminator::Br {
            dest: header,
            args: Default::default(),
        });
        let zero = func.push_const(Const::Int { bits: 32, value: 0 });
        func.blocks[exit].term = Some(Terminator::Return { val: Some(zero) });
        let mut simp = SimplifyCfg;
        assert!(!simp.run(&mut func));
        assert_eq!(func.blocks[entry].insts.len(), 1, "the add stays in the preheader");
    }

    #[test]
    fn inlines_small_callee() {
        let mut module = Module::new();
        let callee_sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
        let (mut callee, centry) = Function::new("add_one", callee_sig);
        let arg = callee.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Param(centry, 0),
        });
        callee.blocks[centry].params.push(arg);
        let one = callee.push_const(Const::Int { bits: 32, value: 1 });
        let sum = callee.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Inst(crate::ir::Inst::from_usize(0)),
        });
        let inst = callee.insts.push(crate::ir::InstData {
            block: centry,
            kind: InstKind::Add(arg, one),
            result: Some(sum),
            loc: None,
        });
        callee.values[sum].def = Def::Inst(inst);
        callee.blocks[centry].insts.push(inst);
        callee.blocks[centry].term = Some(Terminator::Return { val: Some(sum) });
        let callee_id = module.push_func(callee);

        let caller_sig = Signature::new(vec![], Ty::int(32));
        let (mut caller, entry) = Function::new("caller", caller_sig);
        let five = caller.push_const(Const::Int { bits: 32, value: 41 });
        let call_ty = Ty::int(32);
        let call_val = caller.values.push(crate::ir::ValueData {
            ty: call_ty.clone(),
            def: Def::Inst(crate::ir::Inst::from_usize(0)),
        });
        let call = caller.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Call {
                callee: callee_id,
                args: [five].into_iter().collect(),
            },
            result: Some(call_val),
            loc: None,
        });
        caller.values[call_val].def = Def::Inst(call);
        caller.blocks[entry].insts.push(call);
        caller.blocks[entry].term = Some(Terminator::Return { val: Some(call_val) });
        module.push_func(caller);

        let mut inline = Inline::small();
        assert!(inline.run(&mut module));
        let caller = &module.funcs[crate::ir::Func::from_usize(1)];
        let still_calls = caller
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .any(|i| matches!(caller.insts[*i].kind, InstKind::Call { .. }));
        assert!(!still_calls);
        // Inlined `41 + 1` must fold to 42: the callee constant has to be
        // materialized fresh in the caller arena, not reused by index.
        let caller_id = crate::ir::Func::from_usize(1);
        let mut fold = ConstFold;
        fold.run(&mut module.funcs[caller_id]);
        let ret = match &module.funcs[caller_id].blocks[entry].term {
            Some(Terminator::Return { val }) => val.unwrap(),
            _ => panic!("caller returns a value"),
        };
        assert_eq!(module.funcs[caller_id].const_value(ret), Some(&Const::Int { bits: 32, value: 42 }));
    }

    #[test]
    fn inlining_keeps_undef_type() {
        let mut module = Module::new();
        let callee_sig = Signature::new(vec![], Ty::int(32));
        let (mut callee, centry) = Function::new("undef_callee", callee_sig);
        let merge = callee.push_block(vec![Ty::int(32)]);
        let undef = callee.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Const(Const::Undef),
        });
        callee.blocks[centry].term = Some(Terminator::Br {
            dest: merge,
            args: [undef].into_iter().collect(),
        });
        let param = callee.blocks[merge].params[0];
        callee.blocks[merge].term = Some(Terminator::Return { val: Some(param) });
        callee.inline = true;
        let callee_id = module.push_func(callee);

        let caller_sig = Signature::new(vec![], Ty::int(32));
        let (mut caller, entry) = Function::new("caller", caller_sig);
        let call_val = caller.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Inst(crate::ir::Inst::from_usize(0)),
        });
        let call = caller.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Call {
                callee: callee_id,
                args: Default::default(),
            },
            result: Some(call_val),
            loc: None,
        });
        caller.values[call_val].def = Def::Inst(call);
        caller.blocks[entry].insts.push(call);
        caller.blocks[entry].term = Some(Terminator::Return { val: Some(call_val) });
        module.push_func(caller);

        let mut inline = Inline::small();
        assert!(inline.run(&mut module));
        let caller = &module.funcs[crate::ir::Func::from_usize(1)];
        let mut found = false;
        for block in caller.blocks.iter() {
            if let Some(Terminator::Br { args, .. }) = &block.term {
                for &arg in args {
                    if caller.const_value(arg) == Some(&Const::Undef) {
                        found = true;
                        assert_eq!(caller.values[arg].ty, Ty::int(32));
                    }
                }
            }
        }
        assert!(found, "inlined branch keeps its undef argument");
    }

    #[test]
    fn inlining_maps_forward_referenced_values() {
        let mut module = Module::new();
        // Blocks clone in arena order, which is not a def-before-use order:
        // earlier inlinings move a block's tail into a fresh successor, so a
        // later-arena block can define a value used by an earlier-arena one.
        let callee_sig = Signature::new(vec![], Ty::Void);
        let (mut callee, entry) = Function::new("late_def", callee_sig);
        let use_block = callee.push_block(vec![]);
        let def_block = callee.push_block(vec![]);
        let exit = callee.push_block(vec![]);
        let one = callee.push_const(Const::Int { bits: 64, value: 1 });
        let null = callee.values.push(crate::ir::ValueData {
            ty: Ty::ptr(Ty::int(64)),
            def: Def::Const(Const::Null),
        });
        let sum = callee.values.push(crate::ir::ValueData {
            ty: Ty::int(64),
            def: Def::Inst(crate::ir::Inst::from_usize(0)),
        });
        let store = callee.insts.push(crate::ir::InstData {
            block: use_block,
            kind: InstKind::Store {
                addr: null,
                val: sum,
                flags: crate::ir::MemFlags::plain(),
            },
            result: None,
            loc: None,
        });
        callee.blocks[use_block].insts.push(store);
        let add = callee.insts.push(crate::ir::InstData {
            block: def_block,
            kind: InstKind::Add(one, one),
            result: Some(sum),
            loc: None,
        });
        callee.values[sum].def = Def::Inst(add);
        callee.blocks[def_block].insts.push(add);
        callee.blocks[entry].term = Some(Terminator::Br {
            dest: def_block,
            args: Default::default(),
        });
        callee.blocks[def_block].term = Some(Terminator::Br {
            dest: use_block,
            args: Default::default(),
        });
        callee.blocks[use_block].term = Some(Terminator::Br {
            dest: exit,
            args: Default::default(),
        });
        callee.blocks[exit].term = Some(Terminator::Return { val: None });
        callee.inline = true;
        let callee_id = module.push_func(callee);

        // Caller value 2 differs in type from the callee's, so an aliased
        // operand shows up as a type mismatch rather than as silent
        // corruption.
        let caller_sig = Signature::new(vec![], Ty::Void);
        let (mut caller, centry) = Function::new("caller", caller_sig);
        caller.push_const(Const::Int { bits: 64, value: 1 });
        caller.push_const(Const::Int { bits: 64, value: 2 });
        caller.push_const(Const::Int { bits: 32, value: 3 });
        let call = caller.insts.push(crate::ir::InstData {
            block: centry,
            kind: InstKind::Call {
                callee: callee_id,
                args: Default::default(),
            },
            result: None,
            loc: None,
        });
        caller.blocks[centry].insts.push(call);
        caller.blocks[centry].term = Some(Terminator::Return { val: None });
        module.push_func(caller);

        let mut inline = Inline::small();
        assert!(inline.run(&mut module));
        let caller = &module.funcs[crate::ir::Func::from_usize(1)];
        let mut stores = 0;
        for block in caller.blocks.iter() {
            for inst in &block.insts {
                if let InstKind::Store { val, .. } = &caller.insts[*inst].kind {
                    stores += 1;
                    assert_eq!(caller.values[*val].ty, Ty::int(64));
                    assert!(matches!(caller.values[*val].def, Def::Inst(_)));
                }
            }
        }
        assert_eq!(stores, 1, "inlined body still stores");
    }

    #[test]
    fn inlines_callee_with_inline_asm() {
        let mut module = Module::new();
        let callee_sig = Signature::new(vec![], Ty::int(32));
        let (mut callee, centry) = Function::new("with_asm", callee_sig);
        let five = callee.push_const(Const::Int { bits: 32, value: 5 });
        let block = AsmBlock {
            template: "movl $%c0, %0".to_string(),
            operands: vec![AsmOperand {
                value: five,
                constraint: "=r".to_string(),
                ty: Ty::int(32),
                tie: None,
                name: None,
            }],
            outputs: 1,
            clobbers: Vec::new(),
            labels: Vec::new(),
            is_volatile: true,
            is_basic: false,
        };
        let inst = callee.insts.push(crate::ir::InstData {
            block: centry,
            kind: InstKind::InlineAsm { block: Box::new(block) },
            result: None,
            loc: None,
        });
        callee.blocks[centry].insts.push(inst);
        callee.blocks[centry].term = Some(Terminator::Return { val: Some(five) });
        let callee_id = module.push_func(callee);

        let caller_sig = Signature::new(vec![], Ty::int(32));
        let (mut caller, entry) = Function::new("caller", caller_sig);
        let call_val = caller.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Inst(crate::ir::Inst::from_usize(0)),
        });
        let call = caller.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Call {
                callee: callee_id,
                args: [].into_iter().collect(),
            },
            result: Some(call_val),
            loc: None,
        });
        caller.values[call_val].def = Def::Inst(call);
        caller.blocks[entry].insts.push(call);
        caller.blocks[entry].term = Some(Terminator::Return { val: Some(call_val) });
        module.push_func(caller);

        let mut inline = Inline::small();
        assert!(inline.run(&mut module));
        let caller = &module.funcs[crate::ir::Func::from_usize(1)];
        let live: Vec<Inst> = caller.blocks.iter().flat_map(|b| b.insts.iter().copied()).collect();
        assert!(!live.iter().any(|inst| matches!(caller.insts[*inst].kind, InstKind::Call { .. })));
        // The operand must point at a caller value (a fresh constant),
        // never at the callee's arena.
        let operand = live
            .iter()
            .find_map(|inst| match &caller.insts[*inst].kind {
                InstKind::InlineAsm { block } => Some(block.operands[0].value),
                _ => None,
            })
            .expect("inline asm survives inlining");
        assert_eq!(caller.const_value(operand), Some(&Const::Int { bits: 32, value: 5 }));
    }

    fn branchy_callee(marked_inline: bool) -> (Module, crate::ir::Func) {
        // `pick(x) = x > 0 ? 1 : 0`: entry splits to two returns.
        let mut module = Module::new();
        let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
        let (mut callee, entry) = Function::new("pick", sig);
        callee.inline = marked_inline;
        let arg = callee.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Param(entry, 0),
        });
        callee.blocks[entry].params.push(arg);
        let zero = callee.push_const(Const::Int { bits: 32, value: 0 });
        let cond = callee.values.push(crate::ir::ValueData {
            ty: Ty::Bool,
            def: Def::Inst(crate::ir::Inst::from_usize(callee.insts.len())),
        });
        let cmp = callee.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Icmp {
                pred: IcmpPred::Sgt,
                lhs: arg,
                rhs: zero,
            },
            result: Some(cond),
            loc: None,
        });
        callee.values[cond].def = Def::Inst(cmp);
        callee.blocks[entry].insts.push(cmp);
        let then_b = callee.push_block(vec![]);
        let else_b = callee.push_block(vec![]);
        callee.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block: then_b,
            else_block: else_b,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        let one = callee.push_const(Const::Int { bits: 32, value: 1 });
        callee.blocks[then_b].term = Some(Terminator::Return { val: Some(one) });
        let zero_ret = callee.push_const(Const::Int { bits: 32, value: 0 });
        callee.blocks[else_b].term = Some(Terminator::Return { val: Some(zero_ret) });
        let callee_id = module.push_func(callee);

        let caller_sig = Signature::new(vec![], Ty::int(32));
        let (mut caller, centry) = Function::new("caller", caller_sig);
        let five = caller.push_const(Const::Int { bits: 32, value: 5 });
        let call_val = caller.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Inst(crate::ir::Inst::from_usize(caller.insts.len())),
        });
        let call = caller.insts.push(crate::ir::InstData {
            block: centry,
            kind: InstKind::Call {
                callee: callee_id,
                args: [five].into_iter().collect(),
            },
            result: Some(call_val),
            loc: None,
        });
        caller.values[call_val].def = Def::Inst(call);
        caller.blocks[centry].insts.push(call);
        caller.blocks[centry].term = Some(Terminator::Return { val: Some(call_val) });
        module.push_func(caller);
        (module, callee_id)
    }

    fn caller_still_calls(module: &Module, caller: crate::ir::Func) -> bool {
        let func = &module.funcs[caller];
        func.blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .any(|i| matches!(func.insts[*i].kind, InstKind::Call { .. }))
    }

    #[test]
    fn inlines_marked_multi_block_callee() {
        let (mut module, _) = branchy_callee(true);
        let caller = crate::ir::Func::from_usize(1);
        let mut inline = Inline::small();
        assert!(inline.run(&mut module));
        assert!(!caller_still_calls(&module, caller));
        assert!(crate::verify::verify_module(&module).is_empty());
    }

    #[test]
    fn refuses_unmarked_multi_block_callee() {
        let (mut module, _) = branchy_callee(false);
        let caller = crate::ir::Func::from_usize(1);
        let mut inline = Inline::small();
        assert!(!inline.run(&mut module));
        assert!(caller_still_calls(&module, caller));
    }

    #[test]
    fn inline_only_config_inlines_marked_but_not_plain() {
        let (mut marked, _) = branchy_callee(true);
        let mut only = Inline::inline_only();
        assert!(only.run(&mut marked));
        assert!(!caller_still_calls(&marked, crate::ir::Func::from_usize(1)));

        // A small unmarked single-block callee inlines under `small`
        // but not under `inline_only` (one real inst, over the zero
        // heuristic budget but under the keyword budget).
        for (make_marked, expect_inline_only) in [(false, false), (true, true)] {
            let mut module = Module::new();
            let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
            let (mut callee, centry) = Function::new("add_one", sig);
            callee.inline = make_marked;
            let arg = callee.values.push(crate::ir::ValueData {
                ty: Ty::int(32),
                def: Def::Param(centry, 0),
            });
            callee.blocks[centry].params.push(arg);
            let one = callee.push_const(Const::Int { bits: 32, value: 1 });
            let sum = callee.values.push(crate::ir::ValueData {
                ty: Ty::int(32),
                def: Def::Inst(crate::ir::Inst::from_usize(callee.insts.len())),
            });
            let add = callee.insts.push(crate::ir::InstData {
                block: centry,
                kind: InstKind::Add(arg, one),
                result: Some(sum),
                loc: None,
            });
            callee.values[sum].def = Def::Inst(add);
            callee.blocks[centry].insts.push(add);
            callee.blocks[centry].term = Some(Terminator::Return { val: Some(sum) });
            let callee_id = module.push_func(callee);
            let csig = Signature::new(vec![], Ty::int(32));
            let (mut caller, entry) = Function::new("caller", csig);
            let five = caller.push_const(Const::Int { bits: 32, value: 5 });
            let rv = caller.values.push(crate::ir::ValueData {
                ty: Ty::int(32),
                def: Def::Inst(crate::ir::Inst::from_usize(caller.insts.len())),
            });
            let call = caller.insts.push(crate::ir::InstData {
                block: entry,
                kind: InstKind::Call {
                    callee: callee_id,
                    args: [five].into_iter().collect(),
                },
                result: Some(rv),
                loc: None,
            });
            caller.values[rv].def = Def::Inst(call);
            caller.blocks[entry].insts.push(call);
            caller.blocks[entry].term = Some(Terminator::Return { val: Some(rv) });
            module.push_func(caller);
            let mut only = Inline::inline_only();
            assert_eq!(only.run(&mut module), expect_inline_only, "marked={make_marked}");
        }
    }

    #[test]
    fn marked_callee_gets_larger_budget() {
        // 30 chained adds: over the 24-inst heuristic budget but under
        // the 256-inst keyword budget.
        for marked in [false, true] {
            let mut module = Module::new();
            let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
            let (mut callee, centry) = Function::new("long", sig);
            callee.inline = marked;
            let arg = callee.values.push(crate::ir::ValueData {
                ty: Ty::int(32),
                def: Def::Param(centry, 0),
            });
            callee.blocks[centry].params.push(arg);
            let one = callee.push_const(Const::Int { bits: 32, value: 1 });
            let mut acc = arg;
            for _ in 0..30 {
                let next = callee.values.push(crate::ir::ValueData {
                    ty: Ty::int(32),
                    def: Def::Inst(crate::ir::Inst::from_usize(callee.insts.len())),
                });
                let inst = callee.insts.push(crate::ir::InstData {
                    block: centry,
                    kind: InstKind::Add(acc, one),
                    result: Some(next),
                    loc: None,
                });
                callee.values[next].def = Def::Inst(inst);
                callee.blocks[centry].insts.push(inst);
                acc = next;
            }
            callee.blocks[centry].term = Some(Terminator::Return { val: Some(acc) });
            let callee_id = module.push_func(callee);
            let csig = Signature::new(vec![], Ty::int(32));
            let (mut caller, entry) = Function::new("caller", csig);
            let five = caller.push_const(Const::Int { bits: 32, value: 0 });
            let rv = caller.values.push(crate::ir::ValueData {
                ty: Ty::int(32),
                def: Def::Inst(crate::ir::Inst::from_usize(caller.insts.len())),
            });
            let call = caller.insts.push(crate::ir::InstData {
                block: entry,
                kind: InstKind::Call {
                    callee: callee_id,
                    args: [five].into_iter().collect(),
                },
                result: Some(rv),
                loc: None,
            });
            caller.values[rv].def = Def::Inst(call);
            caller.blocks[entry].insts.push(call);
            caller.blocks[entry].term = Some(Terminator::Return { val: Some(rv) });
            module.push_func(caller);
            let mut inline = Inline::small();
            assert_eq!(inline.run(&mut module), marked, "marked={marked} should control the larger budget");
            assert_eq!(caller_still_calls(&module, crate::ir::Func::from_usize(1)), !marked);
        }
    }

    #[test]
    fn refuses_recursive_callee() {
        let mut module = Module::new();
        let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
        let (mut callee, centry) = Function::new("rec", sig);
        callee.inline = true;
        let arg = callee.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Param(centry, 0),
        });
        callee.blocks[centry].params.push(arg);
        let callee_id = module.push_func(callee);
        // Back-patch a self-call: `rec(arg)`.
        {
            let func = &mut module.funcs[callee_id];
            let rv = func.values.push(crate::ir::ValueData {
                ty: Ty::int(32),
                def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
            });
            let call = func.insts.push(crate::ir::InstData {
                block: centry,
                kind: InstKind::Call {
                    callee: callee_id,
                    args: [arg].into_iter().collect(),
                },
                result: Some(rv),
                loc: None,
            });
            func.values[rv].def = Def::Inst(call);
            func.blocks[centry].insts.push(call);
            func.blocks[centry].term = Some(Terminator::Return { val: Some(rv) });
        }
        let csig = Signature::new(vec![], Ty::int(32));
        let (mut caller, entry) = Function::new("caller", csig);
        let five = caller.push_const(Const::Int { bits: 32, value: 1 });
        let rv = caller.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Inst(crate::ir::Inst::from_usize(caller.insts.len())),
        });
        let call = caller.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Call {
                callee: callee_id,
                args: [five].into_iter().collect(),
            },
            result: Some(rv),
            loc: None,
        });
        caller.values[rv].def = Def::Inst(call);
        caller.blocks[entry].insts.push(call);
        caller.blocks[entry].term = Some(Terminator::Return { val: Some(rv) });
        module.push_func(caller);
        let mut inline = Inline::small();
        assert!(!inline.run(&mut module));
        assert!(caller_still_calls(&module, crate::ir::Func::from_usize(1)));
    }

    #[test]
    fn refuses_mutual_recursion() {
        // A calls B and B calls A: inlining either direction would copy
        // a call back into its own caller, so both must be refused.
        let mut module = Module::new();
        let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
        let (mut a, a_entry) = Function::new("a", sig.clone());
        a.inline = true;
        let a_arg = a.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Param(a_entry, 0),
        });
        a.blocks[a_entry].params.push(a_arg);
        let a_id = module.push_func(a);
        let (mut b, b_entry) = Function::new("b", sig);
        b.inline = true;
        let b_arg = b.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Param(b_entry, 0),
        });
        b.blocks[b_entry].params.push(b_arg);
        let b_id = module.push_func(b);
        // Back-patch the cross-calls: `a(x) = b(x)`, `b(x) = a(x)`.
        for (self_id, entry, arg, other) in [(a_id, a_entry, a_arg, b_id), (b_id, b_entry, b_arg, a_id)] {
            let func = &mut module.funcs[self_id];
            let rv = func.values.push(crate::ir::ValueData {
                ty: Ty::int(32),
                def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
            });
            let call = func.insts.push(crate::ir::InstData {
                block: entry,
                kind: InstKind::Call {
                    callee: other,
                    args: [arg].into_iter().collect(),
                },
                result: Some(rv),
                loc: None,
            });
            func.values[rv].def = Def::Inst(call);
            func.blocks[entry].insts.push(call);
            func.blocks[entry].term = Some(Terminator::Return { val: Some(rv) });
        }
        let mut inline = Inline::small();
        assert!(!inline.run(&mut module));
        assert!(caller_still_calls(&module, a_id));
        assert!(caller_still_calls(&module, b_id));
    }

    fn emit_alloca(func: &mut Function, block: crate::ir::Block, pointee: Ty) -> Value {
        let slot = func.values.push(crate::ir::ValueData {
            ty: Ty::ptr(pointee.clone()),
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(crate::ir::InstData {
            block,
            kind: InstKind::Alloca { pointee, count: 1 },
            result: Some(slot),
            loc: None,
        });
        func.values[slot].def = Def::Inst(inst);
        func.blocks[block].insts.push(inst);
        slot
    }

    fn emit_store(func: &mut Function, block: crate::ir::Block, addr: Value, val: Value, vol: bool) -> crate::ir::Inst {
        let flags = if vol { crate::ir::MemFlags::vol() } else { crate::ir::MemFlags::plain() };
        let inst = func.insts.push(crate::ir::InstData {
            block,
            kind: InstKind::Store { addr, val, flags },
            result: None,
            loc: None,
        });
        func.blocks[block].insts.push(inst);
        inst
    }

    fn emit_load(func: &mut Function, block: crate::ir::Block, addr: Value, ty: Ty, vol: bool) -> Value {
        let flags = if vol { crate::ir::MemFlags::vol() } else { crate::ir::MemFlags::plain() };
        let out = func.values.push(crate::ir::ValueData {
            ty,
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(crate::ir::InstData {
            block,
            kind: InstKind::Load { addr, flags },
            result: Some(out),
            loc: None,
        });
        func.values[out].def = Def::Inst(inst);
        func.blocks[block].insts.push(inst);
        out
    }

    fn resident_kinds(func: &Function) -> Vec<InstKind> {
        func.blocks.iter().flat_map(|b| b.insts.iter()).map(|i| func.insts[*i].kind.clone()).collect()
    }

    fn param(func: &mut Function, block: crate::ir::Block, ty: Ty) -> Value {
        let index = func.blocks[block].params.len() as u32;
        let value = func.values.push(crate::ir::ValueData {
            ty,
            def: Def::Param(block, index),
        });
        func.blocks[block].params.push(value);
        value
    }

    fn has_memory_ops(func: &Function) -> bool {
        resident_kinds(func)
            .iter()
            .any(|kind| matches!(kind, InstKind::Alloca { .. } | InstKind::Load { .. } | InstKind::Store { .. }))
    }

    fn br_args(func: &Function, block: crate::ir::Block) -> Vec<Value> {
        match func.blocks[block].term.clone() {
            Some(Terminator::Br { args, .. }) => args.to_vec(),
            other => panic!("expected br, got {other:?}"),
        }
    }

    fn emit_bin(func: &mut Function, block: crate::ir::Block, kind: InstKind, ty: Ty) -> Value {
        let v = func.values.push(crate::ir::ValueData {
            ty,
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(crate::ir::InstData {
            block,
            kind,
            result: Some(v),
            loc: None,
        });
        func.values[v].def = Def::Inst(inst);
        func.blocks[block].insts.push(inst);
        v
    }

    #[test]
    fn ashr_sign_extends_wide_literals() {
        // The frontend pushes `0x8000_0000_0000_0000` as a positive
        // `i128`; as a 64-bit pattern it is the minimum i64, so the
        // shift must treat it as negative.
        let sig = Signature::new(vec![], Ty::int(64));
        let (mut func, entry) = Function::new("ashr", sig);
        let min = func.push_const(Const::Int { bits: 64, value: 1i128 << 63 });
        let shift = func.push_const(Const::Int { bits: 64, value: 63 });
        let ashr = emit_bin(&mut func, entry, InstKind::AShr(min, shift), Ty::int(64));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(ashr) });
        let mut fold = ConstFold;
        fold.run(&mut func);
        let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
            unreachable!()
        };
        assert_eq!(func.const_value(val.unwrap()), Some(&Const::Int { bits: 64, value: -1 }));
    }

    #[test]
    fn folds_shifts_neg_not() {
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("shifts", sig);
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let three = func.push_const(Const::Int { bits: 32, value: 3 });
        let two = func.push_const(Const::Int { bits: 32, value: 2 });
        let sixteen = func.push_const(Const::Int { bits: 32, value: 16 });
        let neg8 = func.push_const(Const::Int { bits: 32, value: -8 });
        let shl = emit_bin(&mut func, entry, InstKind::Shl(one, three), Ty::int(32));
        let lshr = emit_bin(&mut func, entry, InstKind::LShr(sixteen, two), Ty::int(32));
        let ashr = emit_bin(&mut func, entry, InstKind::AShr(neg8, two), Ty::int(32));
        let sum1 = emit_bin(&mut func, entry, InstKind::Add(shl, lshr), Ty::int(32));
        let sum = emit_bin(&mut func, entry, InstKind::Add(sum1, ashr), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        let mut fold = ConstFold;
        fold.run(&mut func);
        // 8 + 4 + (-2) = 10.
        let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
            unreachable!()
        };
        assert_eq!(func.const_value(val.unwrap()), Some(&Const::Int { bits: 32, value: 10 }));

        // Negation and complement fold with wraparound.
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("unary", sig);
        let five = func.push_const(Const::Int { bits: 32, value: 5 });
        let zero = func.push_const(Const::Int { bits: 32, value: 0 });
        let neg = emit_bin(&mut func, entry, InstKind::Neg(five), Ty::int(32));
        let not = emit_bin(&mut func, entry, InstKind::Not(zero), Ty::int(32));
        let sum = emit_bin(&mut func, entry, InstKind::Add(neg, not), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        fold.run(&mut func);
        // -5 + (-1) = -6.
        let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
            unreachable!()
        };
        assert_eq!(func.const_value(val.unwrap()), Some(&Const::Int { bits: 32, value: -6 }));
    }

    #[test]
    fn refuses_trapping_folds() {
        // Division by zero and oversized shifts keep their
        // instructions: folding them would delete the trap.
        let sig = Signature::new(vec![], Ty::Void);
        let (mut func, entry) = Function::new("traps", sig);
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let zero = func.push_const(Const::Int { bits: 32, value: 0 });
        let wide = func.push_const(Const::Int { bits: 32, value: 32 });
        for kind in [
            InstKind::Sdiv(one, zero),
            InstKind::Udiv(one, zero),
            InstKind::Srem(one, zero),
            InstKind::Urem(one, zero),
            InstKind::Shl(one, wide),
            InstKind::LShr(one, wide),
            InstKind::AShr(one, wide),
        ] {
            let v = func.values.push(crate::ir::ValueData {
                ty: Ty::int(32),
                def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
            });
            let inst = func.insts.push(crate::ir::InstData {
                block: entry,
                kind,
                result: Some(v),
                loc: None,
            });
            func.values[v].def = Def::Inst(inst);
            func.blocks[entry].insts.push(inst);
        }
        func.blocks[entry].term = Some(Terminator::Return { val: None });
        let mut fold = ConstFold;
        assert!(!fold.run(&mut func));
        assert_eq!(func.blocks[entry].insts.len(), 7);
    }

    #[test]
    fn folds_div_rem() {
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("div", sig);
        let n7 = func.push_const(Const::Int { bits: 32, value: -7 });
        let two = func.push_const(Const::Int { bits: 32, value: 2 });
        let div = emit_bin(&mut func, entry, InstKind::Sdiv(n7, two), Ty::int(32));
        let rem = emit_bin(&mut func, entry, InstKind::Srem(n7, two), Ty::int(32));
        let sum = emit_bin(&mut func, entry, InstKind::Add(div, rem), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        let mut fold = ConstFold;
        fold.run(&mut func);
        // -7 / 2 = -3 (truncated), -7 % 2 = -1.
        let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
            unreachable!()
        };
        assert_eq!(func.const_value(val.unwrap()), Some(&Const::Int { bits: 32, value: -4 }));
    }

    #[test]
    fn folds_int_conversions() {
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("convs", sig);
        // 0x1FF truncates to -1 in 8 bits; -1 zero-extends to 255.
        let wide = func.push_const(Const::Int { bits: 32, value: 0x1FF });
        let neg1_8 = func.push_const(Const::Int { bits: 8, value: -1 });
        let trunc = {
            let v = func.values.push(crate::ir::ValueData {
                ty: Ty::int(8),
                def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
            });
            let inst = func.insts.push(crate::ir::InstData {
                block: entry,
                kind: InstKind::Trunc(wide),
                result: Some(v),
                loc: None,
            });
            func.values[v].def = Def::Inst(inst);
            func.blocks[entry].insts.push(inst);
            v
        };
        let zext = {
            let v = func.values.push(crate::ir::ValueData {
                ty: Ty::int(32),
                def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
            });
            let inst = func.insts.push(crate::ir::InstData {
                block: entry,
                kind: InstKind::Zext(neg1_8),
                result: Some(v),
                loc: None,
            });
            func.values[v].def = Def::Inst(inst);
            func.blocks[entry].insts.push(inst);
            v
        };
        let sext = {
            let v = func.values.push(crate::ir::ValueData {
                ty: Ty::int(32),
                def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
            });
            let inst = func.insts.push(crate::ir::InstData {
                block: entry,
                kind: InstKind::Sext(neg1_8),
                result: Some(v),
                loc: None,
            });
            func.values[v].def = Def::Inst(inst);
            func.blocks[entry].insts.push(inst);
            v
        };
        // (trunc as i32) + zext + sext = -1 + 255 + -1 = 253.
        let t32 = {
            let v = func.values.push(crate::ir::ValueData {
                ty: Ty::int(32),
                def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
            });
            let inst = func.insts.push(crate::ir::InstData {
                block: entry,
                kind: InstKind::Sext(trunc),
                result: Some(v),
                loc: None,
            });
            func.values[v].def = Def::Inst(inst);
            func.blocks[entry].insts.push(inst);
            v
        };
        let mut b = FunctionBuilder::new(&mut func, entry);
        let s1 = b.add(Ty::int(32), t32, zext);
        let sum = b.add(Ty::int(32), s1, sext);
        b.ret(Some(sum));
        let mut fold = ConstFold;
        fold.run(&mut func);
        let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
            unreachable!()
        };
        assert_eq!(func.const_value(val.unwrap()), Some(&Const::Int { bits: 32, value: 253 }));
    }

    #[test]
    fn folds_switch_on_constant() {
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("switch", sig);
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let b0 = func.push_block(vec![]);
        let b1 = func.push_block(vec![]);
        let bdef = func.push_block(vec![]);
        func.blocks[entry].term = Some(Terminator::Switch {
            scrutinee: one,
            cases: vec![(0, b0), (1, b1)],
            default: bdef,
        });
        for (block, val) in [(b0, 10), (b1, 20), (bdef, 30)] {
            let c = func.push_const(Const::Int { bits: 32, value: val });
            func.blocks[block].term = Some(Terminator::Return { val: Some(c) });
        }
        let mut fold = ConstFold;
        assert!(fold.run(&mut func));
        assert!(matches!(
            func.blocks[entry].term,
            Some(Terminator::Br { dest, .. }) if dest == b1
        ));
    }

    #[test]
    fn folds_fneg_and_fcmp() {
        use crate::ty::FloatKind;
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("floats", sig);
        // -(-1.5f) == 1.5f: exact bitwise negation, then an
        // equality fold over the two constants.
        let neg_in = func.push_const(Const::Float {
            kind: FloatKind::F32,
            bits: 0xBFC0_0000,
        });
        let pos = func.push_const(Const::Float {
            kind: FloatKind::F32,
            bits: 0x3FC0_0000,
        });
        let neg = {
            let v = func.values.push(crate::ir::ValueData {
                ty: Ty::Float(FloatKind::F32),
                def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
            });
            let inst = func.insts.push(crate::ir::InstData {
                block: entry,
                kind: InstKind::FNeg(neg_in),
                result: Some(v),
                loc: None,
            });
            func.values[v].def = Def::Inst(inst);
            func.blocks[entry].insts.push(inst);
            v
        };
        // NaN != 1.5 and NaN is unordered: both fold to 1.
        let nan = func.push_const(Const::Float {
            kind: FloatKind::F64,
            bits: 0x7FF8_0000_0000_0000,
        });
        let one5 = func.push_const(Const::Float {
            kind: FloatKind::F64,
            bits: 0x3FF8_0000_0000_0000,
        });
        let mut flags = Vec::new();
        for (pred, l, r) in [
            (crate::ir::FcmpPred::Eq, neg, pos),
            (crate::ir::FcmpPred::Ne, nan, one5),
            (crate::ir::FcmpPred::Uno, nan, one5),
        ] {
            let v = func.values.push(crate::ir::ValueData {
                ty: Ty::Bool,
                def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
            });
            let inst = func.insts.push(crate::ir::InstData {
                block: entry,
                kind: InstKind::Fcmp { pred, lhs: l, rhs: r },
                result: Some(v),
                loc: None,
            });
            func.values[v].def = Def::Inst(inst);
            func.blocks[entry].insts.push(inst);
            let w = func.values.push(crate::ir::ValueData {
                ty: Ty::int(32),
                def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
            });
            let wi = func.insts.push(crate::ir::InstData {
                block: entry,
                kind: InstKind::Zext(v),
                result: Some(w),
                loc: None,
            });
            func.values[w].def = Def::Inst(wi);
            func.blocks[entry].insts.push(wi);
            flags.push(w);
        }
        let mut b = FunctionBuilder::new(&mut func, entry);
        let s1 = b.add(Ty::int(32), flags[0], flags[1]);
        let sum = b.add(Ty::int(32), s1, flags[2]);
        b.ret(Some(sum));
        let mut fold = ConstFold;
        fold.run(&mut func);
        // 1 + 1 + 1.
        let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
            unreachable!()
        };
        assert_eq!(func.const_value(val.unwrap()), Some(&Const::Int { bits: 32, value: 3 }));
    }

    #[test]
    fn mem2reg_promotes_entry_spill() {
        // `f(x) = x`: entry spills the param, the body loads it.
        let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("id", sig);
        let arg = func.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Param(entry, 0),
        });
        func.blocks[entry].params.push(arg);
        let slot = emit_alloca(&mut func, entry, Ty::int(32));
        emit_store(&mut func, entry, slot, arg, false);
        let body = func.push_block(vec![]);
        func.blocks[entry].term = Some(Terminator::Br {
            dest: body,
            args: Default::default(),
        });
        let loaded = emit_load(&mut func, body, slot, Ty::int(32), false);
        func.blocks[body].term = Some(Terminator::Return { val: Some(loaded) });
        let mut pass = Mem2Reg;
        assert!(pass.run(&mut func));
        assert!(crate::verify::verify_func(&func).is_empty());
        // No memory traffic left: the return reads the parameter.
        for kind in resident_kinds(&func) {
            assert!(
                !matches!(kind, InstKind::Alloca { .. } | InstKind::Load { .. } | InstKind::Store { .. }),
                "memory op survived: {kind:?}"
            );
        }
        let Terminator::Return { val } = func.blocks[body].term.clone().unwrap() else {
            unreachable!()
        };
        assert_eq!(val.unwrap(), arg);
    }

    #[test]
    fn mem2reg_keeps_slots_in_setjmp_functions() {
        // A `setjmp` call site re-enters with the registers of the
        // first return, so a slot written after it must stay memory.
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("jmp", sig);
        let slot = emit_alloca(&mut func, entry, Ty::int(32));
        let seven = func.push_const(Const::Int { bits: 32, value: 7 });
        emit_store(&mut func, entry, slot, seven, false);
        let loaded = emit_load(&mut func, entry, slot, Ty::int(32), false);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });
        func.returns_twice = true;
        let mut pass = Mem2Reg;
        assert!(!pass.run(&mut func), "slots promoted in a setjmp function");
        assert!(crate::verify::verify_func(&func).is_empty());
    }

    #[test]
    fn mem2reg_single_block_chain() {
        // store 1; store 2; load: forwards 2 and kills the first
        // store, then the slot and the second store as dead.
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("chain", sig);
        let slot = emit_alloca(&mut func, entry, Ty::int(32));
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let two = func.push_const(Const::Int { bits: 32, value: 2 });
        emit_store(&mut func, entry, slot, one, false);
        emit_store(&mut func, entry, slot, two, false);
        let loaded = emit_load(&mut func, entry, slot, Ty::int(32), false);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });
        let mut pass = Mem2Reg;
        assert!(pass.run(&mut func));
        assert!(crate::verify::verify_func(&func).is_empty());
        for kind in resident_kinds(&func) {
            assert!(
                !matches!(kind, InstKind::Alloca { .. } | InstKind::Load { .. } | InstKind::Store { .. }),
                "memory op survived: {kind:?}"
            );
        }
        let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
            unreachable!()
        };
        assert_eq!(func.const_value(val.unwrap()), Some(&Const::Int { bits: 32, value: 2 }));
    }

    #[test]
    fn mem2reg_refuses_escaped_slot() {
        // The slot address feeds a GEP: untracked uses stay in memory.
        use crate::ty::GepIndex;
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("esc", sig);
        let slot = emit_alloca(&mut func, entry, Ty::int(32));
        let seven = func.push_const(Const::Int { bits: 32, value: 7 });
        emit_store(&mut func, entry, slot, seven, false);
        let gep = func.values.push(crate::ir::ValueData {
            ty: Ty::ptr(Ty::int(32)),
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
        });
        let gi = func.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::GetElementPtr {
                base: slot,
                indices: [GepIndex::ConstIndex(0)].into_iter().collect(),
            },
            result: Some(gep),
            loc: None,
        });
        func.values[gep].def = Def::Inst(gi);
        func.blocks[entry].insts.push(gi);
        let loaded = emit_load(&mut func, entry, gep, Ty::int(32), false);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });
        let mut pass = Mem2Reg;
        assert!(!pass.run(&mut func));
        assert!(
            resident_kinds(&func)
                .iter()
                .any(|k| matches!(k, InstKind::Load { .. } | InstKind::Store { .. } | InstKind::Alloca { .. }))
        );
    }

    #[test]
    fn mem2reg_refuses_volatile_slot() {
        // Volatile accesses are observable: the slot stays in memory.
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("vol", sig);
        let slot = emit_alloca(&mut func, entry, Ty::int(32));
        let seven = func.push_const(Const::Int { bits: 32, value: 7 });
        emit_store(&mut func, entry, slot, seven, true);
        let loaded = emit_load(&mut func, entry, slot, Ty::int(32), true);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });
        let mut pass = Mem2Reg;
        assert!(!pass.run(&mut func));
        assert!(
            resident_kinds(&func)
                .iter()
                .any(|k| matches!(k, InstKind::Load { .. } | InstKind::Store { .. } | InstKind::Alloca { .. }))
        );
    }

    #[test]
    fn escaping_addresses_marks_non_memory_uses() {
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("esc2", sig);
        let slot = emit_alloca(&mut func, entry, Ty::int(32));
        let seven = func.push_const(Const::Int { bits: 32, value: 7 });
        emit_store(&mut func, entry, slot, seven, false);
        let loaded = emit_load(&mut func, entry, slot, Ty::int(32), false);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });
        let escaped = escaping_addresses(&func);
        assert!(!escaped.contains(&slot));
        assert!(escaped.contains(&loaded));
    }

    #[test]
    fn aggregates_express_alloca_gep_store_load() {
        use crate::ty::GepIndex;
        use crate::verify::verify_func;
        let sig = Signature::new(vec![], Ty::Void);
        let (mut func, entry) = Function::new("agg", sig);
        let point = Ty::Struct {
            fields: vec![Ty::int(32), Ty::int(32)],
            offsets: vec![0, 4],
            size: 8,
        };
        let slot = func.values.push(crate::ir::ValueData {
            ty: Ty::ptr(point.clone()),
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
        });
        let alloca = func.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Alloca { pointee: point, count: 1 },
            result: Some(slot),
            loc: None,
        });
        func.values[slot].def = Def::Inst(alloca);
        func.blocks[entry].insts.push(alloca);
        let field_ptr = func.values.push(crate::ir::ValueData {
            ty: Ty::ptr(Ty::int(32)),
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
        });
        let gep = func.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::GetElementPtr {
                base: slot,
                indices: [GepIndex::Field(1)].into_iter().collect(),
            },
            result: Some(field_ptr),
            loc: None,
        });
        func.values[field_ptr].def = Def::Inst(gep);
        func.blocks[entry].insts.push(gep);
        let seven = func.push_const(Const::Int { bits: 32, value: 7 });
        let store = func.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Store {
                addr: field_ptr,
                val: seven,
                flags: crate::ir::MemFlags::plain(),
            },
            result: None,
            loc: None,
        });
        func.blocks[entry].insts.push(store);
        let loaded = func.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
        });
        let load = func.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Load {
                addr: field_ptr,
                flags: crate::ir::MemFlags::plain(),
            },
            result: Some(loaded),
            loc: None,
        });
        func.values[loaded].def = Def::Inst(load);
        func.blocks[entry].insts.push(load);
        func.blocks[entry].term = Some(Terminator::Return { val: None });
        assert!(verify_func(&func).is_empty());
    }

    #[test]
    fn mem2reg_places_phi_at_branch_merge() {
        // if (c) x = 1; else x = 2; return x;
        let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("diamond", sig);
        let cond = param(&mut func, entry, Ty::Bool);
        let slot = emit_alloca(&mut func, entry, Ty::int(32));
        let then_block = func.push_block(vec![]);
        let else_block = func.push_block(vec![]);
        let merge = func.push_block(vec![]);
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block,
            else_block,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let two = func.push_const(Const::Int { bits: 32, value: 2 });
        emit_store(&mut func, then_block, slot, one, false);
        func.blocks[then_block].term = Some(Terminator::Br {
            dest: merge,
            args: Default::default(),
        });
        emit_store(&mut func, else_block, slot, two, false);
        func.blocks[else_block].term = Some(Terminator::Br {
            dest: merge,
            args: Default::default(),
        });
        let loaded = emit_load(&mut func, merge, slot, Ty::int(32), false);
        func.blocks[merge].term = Some(Terminator::Return { val: Some(loaded) });
        assert!(crate::verify::verify_func(&func).is_empty());

        let mut pass = Mem2Reg;
        assert!(pass.run(&mut func));
        assert!(!has_memory_ops(&func));
        assert!(crate::verify::verify_func(&func).is_empty());
        // Exactly one phi at the merge, fed by both edges.
        let params = func.blocks[merge].params.clone();
        assert_eq!(params.len(), 1);
        assert_eq!(br_args(&func, then_block), vec![one]);
        assert_eq!(br_args(&func, else_block), vec![two]);
        let Terminator::Return { val } = func.blocks[merge].term.clone().unwrap() else {
            unreachable!()
        };
        assert_eq!(val.unwrap(), params[0]);
        // Already promoted: nothing left to do.
        assert!(!pass.run(&mut func));
    }

    #[test]
    fn mem2reg_carries_value_around_a_loop() {
        // x = 0; while (n > 0) { x = x + 1; n = n - 1; } return x;
        let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("loop", sig);
        let n = param(&mut func, entry, Ty::int(32));
        let x_slot = emit_alloca(&mut func, entry, Ty::int(32));
        let n_slot = emit_alloca(&mut func, entry, Ty::int(32));
        let header = func.push_block(vec![]);
        let body = func.push_block(vec![]);
        let exit = func.push_block(vec![]);
        let zero = func.push_const(Const::Int { bits: 32, value: 0 });
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        emit_store(&mut func, entry, x_slot, zero, false);
        emit_store(&mut func, entry, n_slot, n, false);
        func.blocks[entry].term = Some(Terminator::Br {
            dest: header,
            args: Default::default(),
        });
        let nv = emit_load(&mut func, header, n_slot, Ty::int(32), false);
        let pos = emit_bin(
            &mut func,
            header,
            InstKind::Icmp {
                pred: IcmpPred::Sgt,
                lhs: nv,
                rhs: zero,
            },
            Ty::Bool,
        );
        func.blocks[header].term = Some(Terminator::CondBr {
            cond: pos,
            then_block: body,
            else_block: exit,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        let xv = emit_load(&mut func, body, x_slot, Ty::int(32), false);
        let x1 = emit_bin(&mut func, body, InstKind::Add(xv, one), Ty::int(32));
        emit_store(&mut func, body, x_slot, x1, false);
        let nv2 = emit_load(&mut func, body, n_slot, Ty::int(32), false);
        let n1 = emit_bin(&mut func, body, InstKind::Sub(nv2, one), Ty::int(32));
        emit_store(&mut func, body, n_slot, n1, false);
        func.blocks[body].term = Some(Terminator::Br {
            dest: header,
            args: Default::default(),
        });
        let result = emit_load(&mut func, exit, x_slot, Ty::int(32), false);
        func.blocks[exit].term = Some(Terminator::Return { val: Some(result) });
        assert!(crate::verify::verify_func(&func).is_empty());

        let mut pass = Mem2Reg;
        assert!(pass.run(&mut func));
        assert!(!has_memory_ops(&func));
        assert!(crate::verify::verify_func(&func).is_empty());
        // Header phis for both slots, in allocation order: x, then n.
        let params = func.blocks[header].params.clone();
        assert_eq!(params.len(), 2);
        assert_eq!(br_args(&func, entry), vec![zero, n]);
        assert_eq!(br_args(&func, body), vec![x1, n1]);
        let Terminator::Return { val } = func.blocks[exit].term.clone().unwrap() else {
            unreachable!()
        };
        assert_eq!(val.unwrap(), params[0]);
    }

    #[test]
    fn mem2reg_chains_phis_across_nested_merges() {
        // x = 0; if (a) x = 1; if (b) x = 2; return x;
        let sig = Signature::new(vec![Ty::Bool, Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("nested", sig);
        let a = param(&mut func, entry, Ty::Bool);
        let b_cond = param(&mut func, entry, Ty::Bool);
        let slot = emit_alloca(&mut func, entry, Ty::int(32));
        let then_a = func.push_block(vec![]);
        let merge1 = func.push_block(vec![]);
        let then_b = func.push_block(vec![]);
        let merge2 = func.push_block(vec![]);
        let zero = func.push_const(Const::Int { bits: 32, value: 0 });
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let two = func.push_const(Const::Int { bits: 32, value: 2 });
        emit_store(&mut func, entry, slot, zero, false);
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond: a,
            then_block: then_a,
            else_block: merge1,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        emit_store(&mut func, then_a, slot, one, false);
        func.blocks[then_a].term = Some(Terminator::Br {
            dest: merge1,
            args: Default::default(),
        });
        func.blocks[merge1].term = Some(Terminator::CondBr {
            cond: b_cond,
            then_block: then_b,
            else_block: merge2,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        emit_store(&mut func, then_b, slot, two, false);
        func.blocks[then_b].term = Some(Terminator::Br {
            dest: merge2,
            args: Default::default(),
        });
        let loaded = emit_load(&mut func, merge2, slot, Ty::int(32), false);
        func.blocks[merge2].term = Some(Terminator::Return { val: Some(loaded) });

        let mut pass = Mem2Reg;
        assert!(pass.run(&mut func));
        assert!(!has_memory_ops(&func));
        assert!(crate::verify::verify_func(&func).is_empty());
        let params1 = func.blocks[merge1].params.clone();
        let params2 = func.blocks[merge2].params.clone();
        assert_eq!(params1.len(), 1);
        assert_eq!(params2.len(), 1);
        // merge2's phi takes merge1's phi on the fallthrough edge and
        // the fresh constant on the other.
        let merge1_else = match func.blocks[merge1].term.clone().unwrap() {
            Terminator::CondBr { else_args, .. } => else_args.to_vec(),
            other => panic!("expected condbr, got {other:?}"),
        };
        assert_eq!(merge1_else, vec![params1[0]]);
        assert_eq!(br_args(&func, then_b), vec![two]);
        let Terminator::Return { val } = func.blocks[merge2].term.clone().unwrap() else {
            unreachable!()
        };
        assert_eq!(val.unwrap(), params2[0]);
    }

    #[test]
    fn mem2reg_reads_undef_before_the_first_store() {
        // if (c) x = 1; return x; — the else edge has no definition.
        let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("maybe", sig);
        let cond = param(&mut func, entry, Ty::Bool);
        let slot = emit_alloca(&mut func, entry, Ty::int(32));
        let then_block = func.push_block(vec![]);
        let merge = func.push_block(vec![]);
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block,
            else_block: merge,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        emit_store(&mut func, then_block, slot, one, false);
        func.blocks[then_block].term = Some(Terminator::Br {
            dest: merge,
            args: Default::default(),
        });
        let loaded = emit_load(&mut func, merge, slot, Ty::int(32), false);
        func.blocks[merge].term = Some(Terminator::Return { val: Some(loaded) });

        let mut pass = Mem2Reg;
        assert!(pass.run(&mut func));
        assert!(!has_memory_ops(&func));
        assert!(crate::verify::verify_func(&func).is_empty());
        assert_eq!(func.blocks[merge].params.len(), 1);
        assert_eq!(br_args(&func, then_block), vec![one]);
        let else_args = match func.blocks[entry].term.clone().unwrap() {
            Terminator::CondBr { else_args, .. } => else_args.to_vec(),
            other => panic!("expected condbr, got {other:?}"),
        };
        assert_eq!(else_args.len(), 1);
        assert_eq!(func.const_value(else_args[0]), Some(&Const::Undef));
    }

    #[test]
    fn mem2reg_collapses_replacement_chains() {
        // store 1; v1 = load; store v1; v2 = load; return v2
        // v2 is replaced by v1, which itself resolves to 1.
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("chain", sig);
        let slot = emit_alloca(&mut func, entry, Ty::int(32));
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        emit_store(&mut func, entry, slot, one, false);
        let v1 = emit_load(&mut func, entry, slot, Ty::int(32), false);
        emit_store(&mut func, entry, slot, v1, false);
        let v2 = emit_load(&mut func, entry, slot, Ty::int(32), false);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(v2) });

        let mut pass = Mem2Reg;
        assert!(pass.run(&mut func));
        assert!(!has_memory_ops(&func));
        let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
            unreachable!()
        };
        assert_eq!(func.const_value(val.unwrap()), Some(&Const::Int { bits: 32, value: 1 }));
    }

    #[test]
    fn mem2reg_forwards_across_arena_order() {
        // `mid` stores after the block that loads it in arena order;
        // only dominance matters (the Clif lowering fills dominators
        // first), so the load must be replaced with `mid`'s value.
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("forward", sig);
        let slot = emit_alloca(&mut func, entry, Ty::int(32));
        let late = func.push_block(vec![]);
        let mid = func.push_block(vec![]);
        func.blocks[entry].term = Some(Terminator::Br {
            dest: mid,
            args: Default::default(),
        });
        let three = func.push_const(Const::Int { bits: 32, value: 3 });
        let four = func.push_const(Const::Int { bits: 32, value: 4 });
        let seven = emit_bin(&mut func, mid, InstKind::Add(three, four), Ty::int(32));
        emit_store(&mut func, mid, slot, seven, false);
        func.blocks[mid].term = Some(Terminator::Br {
            dest: late,
            args: Default::default(),
        });
        let loaded = emit_load(&mut func, late, slot, Ty::int(32), false);
        func.blocks[late].term = Some(Terminator::Return { val: Some(loaded) });

        let mut pass = Mem2Reg;
        assert!(pass.run(&mut func));
        assert!(!has_memory_ops(&func));
        assert!(crate::verify::verify_func(&func).is_empty());
        let Terminator::Return { val } = func.blocks[late].term.clone().unwrap() else {
            unreachable!()
        };
        assert_eq!(val.unwrap(), seven);
    }

    #[test]
    fn mem2reg_feeds_unreachable_edges_placeholders() {
        // A dead block still branches to the phi block: its edge keeps
        // arity with an undef placeholder.
        let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("dead-edge", sig);
        let cond = param(&mut func, entry, Ty::Bool);
        let slot = emit_alloca(&mut func, entry, Ty::int(32));
        let then_block = func.push_block(vec![]);
        let merge = func.push_block(vec![]);
        let dead = func.push_block(vec![]);
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block,
            else_block: merge,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        emit_store(&mut func, then_block, slot, one, false);
        func.blocks[then_block].term = Some(Terminator::Br {
            dest: merge,
            args: Default::default(),
        });
        let loaded = emit_load(&mut func, merge, slot, Ty::int(32), false);
        func.blocks[merge].term = Some(Terminator::Return { val: Some(loaded) });
        func.blocks[dead].term = Some(Terminator::Br {
            dest: merge,
            args: Default::default(),
        });

        let mut pass = Mem2Reg;
        assert!(pass.run(&mut func));
        assert!(!has_memory_ops(&func));
        assert!(crate::verify::verify_func(&func).is_empty());
        let dead_args = br_args(&func, dead);
        assert_eq!(dead_args.len(), 1);
        assert_eq!(func.const_value(dead_args[0]), Some(&Const::Undef));
    }

    #[test]
    fn mem2reg_leaves_switch_target_phi_slots_in_memory() {
        // The refused slot's phi would land on the switch's default
        // target, which cannot carry arguments; the other slot still
        // promotes in the same run.
        let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("sw", sig);
        let scrut = param(&mut func, entry, Ty::int(32));
        let refused = emit_alloca(&mut func, entry, Ty::int(32));
        let simple = emit_alloca(&mut func, entry, Ty::int(32));
        let case1 = func.push_block(vec![]);
        let default = func.push_block(vec![]);
        func.blocks[entry].term = Some(Terminator::Switch {
            scrutinee: scrut,
            cases: vec![(1, case1)],
            default,
        });
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let seven = func.push_const(Const::Int { bits: 32, value: 7 });
        emit_store(&mut func, case1, refused, one, false);
        func.blocks[case1].term = Some(Terminator::Br {
            dest: default,
            args: Default::default(),
        });
        emit_store(&mut func, entry, simple, seven, false);
        let a = emit_load(&mut func, default, refused, Ty::int(32), false);
        let b = emit_load(&mut func, default, simple, Ty::int(32), false);
        let sum = emit_bin(&mut func, default, InstKind::Add(a, b), Ty::int(32));
        func.blocks[default].term = Some(Terminator::Return { val: Some(sum) });

        let mut pass = Mem2Reg;
        assert!(pass.run(&mut func));
        assert!(crate::verify::verify_func(&func).is_empty());
        // What remains is exactly the refused slot's traffic.
        let memory_ops = resident_kinds(&func)
            .iter()
            .filter(|kind| matches!(kind, InstKind::Alloca { .. } | InstKind::Load { .. } | InstKind::Store { .. }))
            .count();
        assert_eq!(memory_ops, 3);
    }

    #[test]
    fn mem2reg_waits_for_simplify_cfg_on_dead_blocks() {
        // A load in an unreachable block blocks promotion until
        // SimplifyCfg clears it; the pipeline then promotes cleanly.
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("dead", sig);
        let slot = emit_alloca(&mut func, entry, Ty::int(32));
        let seven = func.push_const(Const::Int { bits: 32, value: 7 });
        emit_store(&mut func, entry, slot, seven, false);
        let live = emit_load(&mut func, entry, slot, Ty::int(32), false);
        let dead = func.push_block(vec![]);
        let dead_load = emit_load(&mut func, dead, slot, Ty::int(32), false);
        func.blocks[dead].term = Some(Terminator::Return { val: Some(dead_load) });
        func.blocks[entry].term = Some(Terminator::Return { val: Some(live) });

        let mut pass = Mem2Reg;
        assert!(!pass.run(&mut func));
        assert!(has_memory_ops(&func));

        let mut simplify = SimplifyCfg;
        assert!(simplify.run(&mut func));
        assert!(pass.run(&mut func));
        assert!(!has_memory_ops(&func));
        assert!(crate::verify::verify_func(&func).is_empty());
        let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
            unreachable!()
        };
        assert_eq!(func.const_value(val.unwrap()), Some(&Const::Int { bits: 32, value: 7 }));
    }

    #[test]
    fn mem2reg_drops_untouched_allocas() {
        let sig = Signature::new(vec![], Ty::Void);
        let (mut func, entry) = Function::new("unused", sig);
        emit_alloca(&mut func, entry, Ty::int(32));
        emit_alloca(&mut func, entry, Ty::ptr(Ty::int(32)));
        func.blocks[entry].term = Some(Terminator::Return { val: None });

        let mut pass = Mem2Reg;
        assert!(pass.run(&mut func));
        assert!(!has_memory_ops(&func));
        assert!(!pass.run(&mut func));
    }

    #[test]
    fn mem2reg_leaves_aggregate_slots_in_memory() {
        // Aggregate slots stay addressed memory: promotion is scalar.
        let point = Ty::Struct {
            fields: vec![Ty::int(32), Ty::int(32)],
            offsets: vec![0, 4],
            size: 8,
        };
        let sig = Signature::new(vec![], point.clone());
        let (mut func, entry) = Function::new("agg", sig);
        let slot = emit_alloca(&mut func, entry, point.clone());
        let value = func.values.push(crate::ir::ValueData {
            ty: point.clone(),
            def: Def::Const(Const::Undef),
        });
        emit_store(&mut func, entry, slot, value, false);
        let loaded = emit_load(&mut func, entry, slot, point, false);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        let mut pass = Mem2Reg;
        assert!(!pass.run(&mut func));
        assert!(has_memory_ops(&func));
    }
}
