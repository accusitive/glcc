//! Peephole simplification of pure instructions: algebraic identities,
//! redundant conversions, and comparison patterns.
//!
//! [`ConstFold`](super::ConstFold) needs every operand known at compile
//! time; this pass removes operands that cannot change the result
//! (`x + 0`, `select c, x, x`, a value compared with itself) and
//! collapses the conversion chains C integer promotions produce
//! (`zext` feeding `icmp != 0`, `trunc` of a masked value).
//!
//! A rewrite either reuses an operand, materializes a constant, or
//! rebuilds the instruction with one fewer operand at the rewritten
//! site, so no result ever becomes a non-dominating use. Instructions
//! rewritten this way leave the block immediately; the pipeline's
//! `DeadCodeElim` only has to sweep what the rewrite orphaned.

use crate::ir::{Block, Const, Def, FcmpPred, Function, IcmpPred, Inst, InstData, InstKind, Value, ValueData};
use crate::pass::FunctionPass;
use crate::ty::{GepIndex, Ty};

use super::{eval_icmp, mask, wrap};

/// Algebraic simplification of pure instructions.
#[derive(Default)]
pub struct InstCombine;

impl FunctionPass for InstCombine {
    fn name(&self) -> &'static str {
        "inst-combine"
    }

    fn run(&mut self, func: &mut Function) -> bool {
        // Dead results cannot be simplified usefully, and rewriting them
        // would keep firing until the fixpoint bound; DCE owns them.
        let use_counts = func.count_resident_uses();
        let mut changed = false;
        // Collect rewrites and apply them in one sweep; rewriting per
        // fold is O(function size).
        let mut redirect: std::collections::HashMap<Value, Value> = std::collections::HashMap::new();
        for idx in 0..func.blocks.len() {
            let block = Block::from_usize(idx);
            let old = std::mem::take(&mut func.blocks[block].insts);
            let mut out = Vec::with_capacity(old.len());
            for inst in old {
                let Some(result) = func.insts[inst].result else {
                    out.push(inst);
                    continue;
                };
                if use_counts.get(result.index()).copied().unwrap_or(0) == 0 {
                    out.push(inst);
                    continue;
                }
                let mut kind = func.insts[inst].kind.clone();
                if !redirect.is_empty() {
                    let mut hit = false;
                    kind.for_each_operand(&mut |v| hit |= redirect.contains_key(&v));
                    if hit {
                        kind = crate::ir::remap_inst(&kind, |v| redirect.get(&v).copied().unwrap_or(v));
                    }
                }
                let Some(plan) = plan(func, inst, &kind, &use_counts) else {
                    out.push(inst);
                    continue;
                };
                changed = true;
                let replacement = apply(func, block, inst, result, plan, &mut out);
                redirect.insert(result, replacement);
            }
            func.blocks[block].insts = out;
        }
        if !redirect.is_empty() {
            super::collapse_chains(&mut redirect);
            func.remap_operands(&redirect);
        }
        changed
    }
}

/// A rewrite chosen for one instruction.
enum Plan {
    /// Use an existing value in place of the result.
    Value(Value),
    /// Use a constant in place of the result.
    Const(Const),
    /// Rebuild the instruction with fewer operands (same result type).
    Inst(InstKind),
    /// `v << amount` with a freshly materialized integer constant.
    ShlBy(Value, u32),
    /// `v >> amount` (logical) with a freshly materialized integer constant.
    LshrBy(Value, u32),
    /// `v >> amount` (arithmetic) with a freshly materialized integer constant.
    AshrBy(Value, u32),
    /// `v & mask` with a freshly materialized integer constant.
    AndConst(Value, i128),
    /// `v | mask` with a freshly materialized integer constant.
    OrConst(Value, i128),
    /// `v ^ mask` with a freshly materialized integer constant.
    XorConst(Value, i128),
    /// `v + constant` with a freshly materialized integer constant.
    AddConst(Value, i128),
    /// `v - constant` with a freshly materialized integer constant.
    SubConst(Value, i128),
    /// `v * constant` with a freshly materialized integer constant.
    MulConst(Value, i128),
    /// `!v` for a `Bool`-typed `v`, as `icmp.Eq v, 0`.
    BoolNot(Value),
    /// Compare `v` against zero (`Eq` when `equal`, otherwise `Ne`).
    CompareZero { value: Value, equal: bool },
    /// Signed division (or remainder when `remainder`) by a positive
    /// power of two, rounded toward zero.
    SignedPow2 { value: Value, shift: u32, remainder: bool },
    /// Compare a widened value with a constant after narrowing both to
    /// the original width.
    NarrowCompare { value: Value, pred: IcmpPred, ty: Ty, constant: i128 },
    /// Extend a boolean (optionally negated) to an integer result.
    BoolCast { value: Value, ty: Ty, signed: bool, invert: bool },
    /// Bitwise operation on two booleans, zero-extended to the result.
    ZextBoolOp { op: Bitwise, lhs: Value, rhs: Value, ty: Ty },
    /// Start from `source` and overwrite `fields`, keeping the rest.
    RebuildAggregate { source: Value, fields: Vec<(u32, Value)> },
}

/// Choose a rewrite for `inst`, whose result must have uses.
fn plan(func: &Function, inst: Inst, kind: &InstKind, use_counts: &[u32]) -> Option<Plan> {
    let result = func.insts[inst].result?;
    let result_ty = &func.values[result].ty;
    match kind {
        InstKind::Add(a, b) => {
            if is_int_zero(func, *b) {
                Some(Plan::Value(*a))
            } else if is_int_zero(func, *a) {
                Some(Plan::Value(*b))
            } else if a == b && int_ty_bits(result_ty).is_some() {
                Some(Plan::ShlBy(*a, 1))
            } else {
                reassociate_add(func, result_ty, *a, *b)
            }
        }
        InstKind::Sub(a, b) => {
            if is_int_zero(func, *b) {
                Some(Plan::Value(*a))
            } else if a == b {
                Some(Plan::Const(zero_const(result_ty)?))
            } else if is_int_zero(func, *a) && int_ty_bits(result_ty).is_some() {
                Some(Plan::Inst(InstKind::Neg(*b)))
            } else if is_all_ones(func, *b) {
                Some(Plan::AddConst(*a, 1))
            } else if is_all_ones(func, *a) {
                Some(Plan::Inst(InstKind::Not(*b)))
            } else {
                reassociate_sub(func, result_ty, *a, *b)
            }
        }
        InstKind::Mul(a, b) => {
            if is_int_zero(func, *a) || is_int_zero(func, *b) {
                Some(Plan::Const(zero_const(result_ty)?))
            } else if is_int_one(func, *a) {
                Some(Plan::Value(*b))
            } else if is_int_one(func, *b) {
                Some(Plan::Value(*a))
            } else if is_all_ones(func, *a) {
                Some(Plan::Inst(InstKind::Neg(*b)))
            } else if is_all_ones(func, *b) {
                Some(Plan::Inst(InstKind::Neg(*a)))
            } else if let Some(plan) = reassociate_mul(func, result_ty, *a, *b) {
                Some(plan)
            } else if let Some(shift) = int_pow2(func, *b) {
                Some(Plan::ShlBy(*a, shift))
            } else { int_pow2(func, *a).map(|shift| Plan::ShlBy(*b, shift)) }
        }
        // A divisor of one cannot trap, so the quotient is the dividend
        // and the remainder is zero for every input (including the
        // minimum signed value, whose `-1` cases are the dangerous
        // ones).
        InstKind::Udiv(a, b) => {
            if is_int_one(func, *b) {
                Some(Plan::Value(*a))
            } else {
                int_pow2(func, *b).map(|shift| Plan::LshrBy(*a, shift))
            }
        }
        InstKind::Sdiv(a, b) if is_int_one(func, *b) => Some(Plan::Value(*a)),
        InstKind::Sdiv(a, b) => signed_pow2(func, result_ty, *a, *b, false),
        InstKind::Urem(a, b) => {
            if is_int_one(func, *b) {
                Some(Plan::Const(zero_const(result_ty)?))
            } else {
                int_pow2(func, *b).map(|shift| Plan::AndConst(*a, mask(shift) as i128))
            }
        }
        InstKind::Srem(_, b) if is_int_one(func, *b) => Some(Plan::Const(zero_const(result_ty)?)),
        InstKind::Srem(a, b) => signed_pow2(func, result_ty, *a, *b, true),
        InstKind::And(a, b) => {
            if is_int_zero(func, *a) || is_int_zero(func, *b) {
                Some(Plan::Const(zero_const(result_ty)?))
            } else if a == b {
                Some(Plan::Value(*a))
            } else if is_all_ones(func, *a) {
                Some(Plan::Value(*b))
            } else if is_all_ones(func, *b) {
                Some(Plan::Value(*a))
            } else {
                combine_bitwise(func, result_ty, *a, *b, Bitwise::And).or_else(|| zext_bitwise(func, result_ty, *a, *b, Bitwise::And))
            }
        }
        InstKind::Or(a, b) => {
            if is_all_ones(func, *a) || is_all_ones(func, *b) {
                Some(Plan::Const(all_ones_const(result_ty)?))
            } else if is_int_zero(func, *b) {
                Some(Plan::Value(*a))
            } else if is_int_zero(func, *a) {
                Some(Plan::Value(*b))
            } else if a == b {
                Some(Plan::Value(*a))
            } else {
                combine_bitwise(func, result_ty, *a, *b, Bitwise::Or).or_else(|| zext_bitwise(func, result_ty, *a, *b, Bitwise::Or))
            }
        }
        InstKind::Xor(a, b) => {
            if a == b {
                Some(Plan::Const(zero_const(result_ty)?))
            } else if is_int_zero(func, *b) {
                Some(Plan::Value(*a))
            } else if is_int_zero(func, *a) {
                Some(Plan::Value(*b))
            } else if is_all_ones(func, *a) {
                Some(Plan::Inst(InstKind::Not(*b)))
            } else if is_all_ones(func, *b) {
                Some(Plan::Inst(InstKind::Not(*a)))
            } else {
                combine_bitwise(func, result_ty, *a, *b, Bitwise::Xor).or_else(|| zext_bitwise(func, result_ty, *a, *b, Bitwise::Xor))
            }
        }
        InstKind::Shl(a, b) => plan_shift(func, result_ty, *a, *b, ShiftMode::Left, use_counts),
        InstKind::LShr(a, b) => plan_shift(func, result_ty, *a, *b, ShiftMode::LogicalRight, use_counts),
        InstKind::AShr(a, b) => plan_shift(func, result_ty, *a, *b, ShiftMode::ArithmeticRight, use_counts),
        InstKind::Neg(a) => nested_operand(func, *a, DoubleOp::Neg).map(Plan::Value),
        InstKind::Not(a) => nested_operand(func, *a, DoubleOp::Not).map(Plan::Value),
        InstKind::FNeg(a) => nested_operand(func, *a, DoubleOp::FNeg).map(Plan::Value),
        InstKind::Trunc(a) => {
            let inner = def_inst(func, *a)?;
            match &func.insts[inner].kind {
                // Truncation is idempotent at the narrower width, whether
                // the inner op widened or narrowed.
                InstKind::Trunc(x) => Some(Plan::Inst(InstKind::Trunc(*x))),
                InstKind::Zext(x) | InstKind::Sext(x) if func.values[*x].ty == *result_ty => Some(Plan::Value(*x)),
                // `trunc` keeps only the low bits, so an AND whose mask
                // covers all of them is redundant.
                InstKind::And(x, c) if trunc_mask_covers(func, *c, result_ty) => Some(Plan::Inst(InstKind::Trunc(*x))),
                _ => None,
            }
        }
        InstKind::ExtractValue { agg, index } => {
            let inner = def_inst(func, *agg)?;
            match &func.insts[inner].kind {
                InstKind::InsertValue { agg: base, elem, index: inserted } => {
                    if inserted == index {
                        Some(Plan::Value(*elem))
                    } else {
                        Some(Plan::Inst(InstKind::ExtractValue { agg: *base, index: *index }))
                    }
                }
                _ => None,
            }
        }
        InstKind::InsertValue { .. } => {
            if let Some(source) = reconstructed_aggregate(func, result) {
                return Some(Plan::Value(source));
            }
            if let Some((source, fields)) = aggregate_field_updates(func, result) {
                return Some(Plan::RebuildAggregate { source, fields });
            }
            let InstKind::InsertValue { agg, elem, index } = kind else {
                unreachable!("matched an insert");
            };
            let inner = def_inst(func, *elem)?;
            match &func.insts[inner].kind {
                InstKind::ExtractValue { agg: source, index: source_index } if source == agg && source_index == index => Some(Plan::Value(*agg)),
                _ => None,
            }
        }
        InstKind::Zext(a) => {
            if let Some(plan) = zext_boolean_value(func, *a, result_ty) {
                return Some(plan);
            }
            let inner = def_inst(func, *a)?;
            match &func.insts[inner].kind {
                InstKind::Zext(x) => Some(Plan::Inst(InstKind::Zext(*x))),
                _ => None,
            }
        }
        InstKind::Sext(a) => {
            let inner = def_inst(func, *a)?;
            match &func.insts[inner].kind {
                InstKind::Sext(x) => Some(Plan::Inst(InstKind::Sext(*x))),
                _ => None,
            }
        }
        InstKind::Bitcast(a) => {
            let inner = def_inst(func, *a)?;
            match &func.insts[inner].kind {
                InstKind::Bitcast(x) => Some(Plan::Inst(InstKind::Bitcast(*x))),
                _ => None,
            }
        }
        InstKind::PtrToInt(a) => {
            let inner = def_inst(func, *a)?;
            match &func.insts[inner].kind {
                InstKind::IntToPtr(x) if func.values[*x].ty == *result_ty => Some(Plan::Value(*x)),
                _ => None,
            }
        }
        InstKind::IntToPtr(a) => {
            let inner = def_inst(func, *a)?;
            let InstKind::PtrToInt(source) = func.insts[inner].kind else {
                return None;
            };
            if func.values[source].ty == *result_ty {
                return Some(Plan::Value(source));
            }
            match (&func.values[source].ty, result_ty) {
                (Ty::Ptr { bits: from, .. }, Ty::Ptr { bits: to, .. }) if from == to => Some(Plan::Inst(InstKind::Bitcast(source))),
                _ => None,
            }
        }
        InstKind::Icmp { pred, lhs, rhs } => {
            if lhs == rhs && !is_float(func, *lhs) {
                let truth = matches!(pred, IcmpPred::Eq | IcmpPred::Sle | IcmpPred::Ule | IcmpPred::Sge | IcmpPred::Uge);
                let bits = int_ty_bits(result_ty)?;
                return Some(Plan::Const(Const::Int { bits, value: i128::from(truth) }));
            }
            if let Some(plan) = bool_zero_compare(func, *pred, *lhs, *rhs) {
                return Some(plan);
            }
            if let Some(plan) = unsigned_zero_bound(func, *pred, *lhs, *rhs) {
                return Some(plan);
            }
            if let Some(plan) = ascii_after_sub_one(func, *pred, *lhs, *rhs) {
                return Some(plan);
            }
            if let Some(plan) = narrow_widened_compare(func, *pred, *lhs, *rhs) {
                return Some(plan);
            }
            if let Some(plan) = narrow_widened_const_compare(func, result_ty, *pred, *lhs, *rhs) {
                return Some(plan);
            }
            let (pred, zext_side, zero_side) = (*pred, *lhs, *rhs);
            for (maybe_zext, maybe_zero) in [(zext_side, zero_side), (zero_side, zext_side)] {
                if !is_int_zero(func, maybe_zero) {
                    continue;
                }
                let Some(b) = zext_bool(func, maybe_zext) else {
                    continue;
                };
                return match pred {
                    IcmpPred::Ne => Some(Plan::Value(b)),
                    IcmpPred::Eq => Some(Plan::BoolNot(b)),
                    _ => None,
                };
            }
            None
        }
        InstKind::Select { cond, then_val, else_val } => {
            if then_val == else_val {
                return Some(Plan::Value(*then_val));
            }
            match int_const(func, *cond) {
                Some((_, 0)) => Some(Plan::Value(*else_val)),
                Some(_) => Some(Plan::Value(*then_val)),
                None => select_bool_constant(func, result_ty, *cond, *then_val, *else_val).or_else(|| select_boolean_arm(func, *cond, *then_val, *else_val)),
            }
        }
        InstKind::GetElementPtr { base, indices } => {
            let zero = indices.iter().all(|index| match index {
                GepIndex::Field(field) => *field == 0,
                GepIndex::ConstIndex(index) => *index == 0,
                GepIndex::Index(value) => int_const(func, *value).is_some_and(|(_, value)| value == 0),
            });
            if !zero {
                return None;
            }
            let base_ty = &func.values[*base].ty;
            if base_ty == result_ty {
                Some(Plan::Value(*base))
            } else {
                match (base_ty, result_ty) {
                    (Ty::Ptr { bits: from, .. }, Ty::Ptr { bits: to, .. }) if from == to => Some(Plan::Inst(InstKind::Bitcast(*base))),
                    _ => None,
                }
            }
        }
        _ => None,
    }
}

/// Signed division rounds toward zero; adding `2^shift - 1` only for
/// negative dividends before the arithmetic shift gets that rounding.
fn signed_pow2(func: &Function, result_ty: &Ty, value: Value, divisor: Value, remainder: bool) -> Option<Plan> {
    let bits = arith_bits(result_ty)?;
    if bits < 2 {
        return None;
    }
    let shift = int_pow2(func, divisor)?;
    if shift == 0 || shift >= bits {
        return None;
    }
    Some(Plan::SignedPow2 { value, shift, remainder })
}

fn push_temp(func: &mut Function, block: Block, kind: InstKind, ty: Ty, loc: Option<crate::debug::Location>, out: &mut Vec<Inst>) -> Value {
    let next = Inst::from_usize(func.insts.len());
    let value = func.values.push(ValueData { ty, def: Def::Inst(next) });
    let inst = func.insts.push(InstData {
        block,
        kind,
        result: Some(value),
        loc,
    });
    func.values[value].def = Def::Inst(inst);
    out.push(inst);
    value
}

/// Materialize a plan: create the replacement value, and the rebuilt
/// instruction when the plan needs one.
fn apply(func: &mut Function, block: Block, inst: Inst, result: Value, plan: Plan, out: &mut Vec<Inst>) -> Value {
    let plan = match plan {
        Plan::BoolNot(v) => {
            let ty = func.values[v].ty.clone();
            let zero = func.values.push(ValueData {
                ty,
                def: Def::Const(Const::Int { bits: 1, value: 0 }),
            });
            Plan::Inst(InstKind::Icmp {
                pred: IcmpPred::Eq,
                lhs: v,
                rhs: zero,
            })
        }
        Plan::ShlBy(v, amount) => {
            let ty = func.values[result].ty.clone();
            Plan::Inst(InstKind::Shl(v, push_int_const(func, &ty, amount as i128)))
        }
        Plan::LshrBy(v, amount) => {
            let ty = func.values[result].ty.clone();
            Plan::Inst(InstKind::LShr(v, push_int_const(func, &ty, amount as i128)))
        }
        Plan::AshrBy(v, amount) => {
            let ty = func.values[result].ty.clone();
            Plan::Inst(InstKind::AShr(v, push_int_const(func, &ty, amount as i128)))
        }
        Plan::AndConst(v, value) => {
            let ty = func.values[result].ty.clone();
            Plan::Inst(InstKind::And(v, push_int_const(func, &ty, value)))
        }
        Plan::OrConst(v, value) => {
            let ty = func.values[result].ty.clone();
            Plan::Inst(InstKind::Or(v, push_int_const(func, &ty, value)))
        }
        Plan::XorConst(v, value) => {
            let ty = func.values[result].ty.clone();
            Plan::Inst(InstKind::Xor(v, push_int_const(func, &ty, value)))
        }
        Plan::AddConst(v, value) => {
            let ty = func.values[result].ty.clone();
            Plan::Inst(InstKind::Add(v, push_int_const(func, &ty, value)))
        }
        Plan::SubConst(v, value) => {
            let ty = func.values[result].ty.clone();
            Plan::Inst(InstKind::Sub(v, push_int_const(func, &ty, value)))
        }
        Plan::MulConst(v, value) => {
            let ty = func.values[result].ty.clone();
            Plan::Inst(InstKind::Mul(v, push_int_const(func, &ty, value)))
        }
        Plan::CompareZero { value, equal } => {
            let ty = func.values[value].ty.clone();
            let zero = push_int_const(func, &ty, 0);
            Plan::Inst(InstKind::Icmp {
                pred: if equal { IcmpPred::Eq } else { IcmpPred::Ne },
                lhs: value,
                rhs: zero,
            })
        }
        Plan::NarrowCompare { value, pred, ty, constant } => {
            let bits = int_ty_bits(&ty).expect("narrow comparison has an integer width");
            let constant = func.values.push(ValueData {
                ty,
                def: Def::Const(Const::Int { bits, value: constant }.canonical()),
            });
            Plan::Inst(InstKind::Icmp { pred, lhs: value, rhs: constant })
        }
        other => other,
    };
    match plan {
        Plan::Value(v) => v,
        Plan::Const(c) => {
            let ty = func.values[result].ty.clone();
            let origin = crate::ir::ValueOrigin {
                pass: "instcombine",
                text: crate::display::inst_str(func, &func.insts[inst].kind),
                loc: func.insts[inst].loc,
            };
            let fresh = func.values.push(ValueData { ty, def: Def::Const(c) });
            func.origins.insert(fresh, origin);
            fresh
        }
        Plan::Inst(kind) => {
            let loc = func.insts[inst].loc;
            let next = Inst::from_usize(func.insts.len());
            let replacement = func.values.push(ValueData {
                ty: func.values[result].ty.clone(),
                def: Def::Inst(next),
            });
            let new = func.insts.push(InstData {
                block,
                kind,
                result: Some(replacement),
                loc,
            });
            func.values[replacement].def = Def::Inst(new);
            out.push(new);
            replacement
        }
        Plan::RebuildAggregate { source, fields } => {
            let ty = func.values[result].ty.clone();
            let loc = func.insts[inst].loc;
            let mut current = source;
            for (index, elem) in fields {
                let (next, new) = super::make_value(func, block, InstKind::InsertValue { agg: current, elem, index }, ty.clone());
                func.insts[new].loc = loc;
                out.push(new);
                current = next;
            }
            current
        }
        Plan::BoolCast { value, ty, signed, invert } => {
            let loc = func.insts[inst].loc;
            let cond = if invert {
                let zero = push_int_const(func, &Ty::Bool, 0);
                push_temp(
                    func,
                    block,
                    InstKind::Icmp {
                        pred: IcmpPred::Eq,
                        lhs: value,
                        rhs: zero,
                    },
                    Ty::Bool,
                    loc,
                    out,
                )
            } else {
                value
            };
            let kind = if signed { InstKind::Sext(cond) } else { InstKind::Zext(cond) };
            push_temp(func, block, kind, ty, loc, out)
        }
        Plan::ZextBoolOp { op, lhs, rhs, ty } => {
            let loc = func.insts[inst].loc;
            let kind = match op {
                Bitwise::And => InstKind::And(lhs, rhs),
                Bitwise::Or => InstKind::Or(lhs, rhs),
                Bitwise::Xor => InstKind::Xor(lhs, rhs),
            };
            let inner = push_temp(func, block, kind, Ty::Bool, loc, out);
            push_temp(func, block, InstKind::Zext(inner), ty, loc, out)
        }
        Plan::SignedPow2 { value, shift, remainder } => {
            let ty = func.values[result].ty.clone();
            let bits = int_ty_bits(&ty).expect("signed integer division");
            let loc = func.insts[inst].loc;
            let sign_amount = push_int_const(func, &ty, i128::from(bits - 1));
            let sign = push_temp(func, block, InstKind::AShr(value, sign_amount), ty.clone(), loc, out);
            let bias_amount = push_int_const(func, &ty, i128::from(bits - shift));
            let bias = push_temp(func, block, InstKind::LShr(sign, bias_amount), ty.clone(), loc, out);
            let adjusted = push_temp(func, block, InstKind::Add(value, bias), ty.clone(), loc, out);
            let shift_amount = push_int_const(func, &ty, i128::from(shift));
            let quotient = push_temp(func, block, InstKind::AShr(adjusted, shift_amount), ty.clone(), loc, out);
            if remainder {
                let scaled = push_temp(func, block, InstKind::Shl(quotient, shift_amount), ty.clone(), loc, out);
                push_temp(func, block, InstKind::Sub(value, scaled), ty, loc, out)
            } else {
                quotient
            }
        }
        Plan::BoolNot(_)
        | Plan::ShlBy(..)
        | Plan::LshrBy(..)
        | Plan::AshrBy(..)
        | Plan::AndConst(..)
        | Plan::OrConst(..)
        | Plan::XorConst(..)
        | Plan::AddConst(..)
        | Plan::SubConst(..)
        | Plan::MulConst(..)
        | Plan::CompareZero { .. }
        | Plan::NarrowCompare { .. } => {
            unreachable!("normalized above")
        }
    }
}

/// Doubled unary operators that cancel out.
enum DoubleOp {
    Neg,
    Not,
    FNeg,
}

/// The operand of `v`'s defining instruction when it is `op` again.
fn nested_operand(func: &Function, v: Value, op: DoubleOp) -> Option<Value> {
    let inst = def_inst(func, v)?;
    match (&func.insts[inst].kind, op) {
        (InstKind::Neg(x), DoubleOp::Neg) | (InstKind::Not(x), DoubleOp::Not) | (InstKind::FNeg(x), DoubleOp::FNeg) => Some(*x),
        _ => None,
    }
}

fn def_inst(func: &Function, v: Value) -> Option<Inst> {
    match &func.values[v].def {
        Def::Inst(i) if i.index() < func.insts.len() => Some(*i),
        _ => None,
    }
}

/// The `Bool` value a `Zext` promotes, if `v` is one.
fn zext_bool(func: &Function, v: Value) -> Option<Value> {
    let inst = def_inst(func, v)?;
    match &func.insts[inst].kind {
        InstKind::Zext(x) if func.values[*x].ty == Ty::Bool => Some(*x),
        _ => None,
    }
}

/// `select c, 1, 0` is a canonical boolean extension; the inverse and
/// all-ones forms negate first.
fn select_bool_constant(func: &Function, result_ty: &Ty, cond: Value, then_val: Value, else_val: Value) -> Option<Plan> {
    if func.values[cond].ty != Ty::Bool {
        return None;
    }
    let (_, then_value) = int_const(func, then_val)?;
    let (_, else_value) = int_const(func, else_val)?;
    let (signed, invert) = match (then_value, else_value) {
        (1, 0) => (false, false),
        (0, 1) => (false, true),
        (-1, 0) => (true, false),
        (0, -1) => (true, true),
        _ => return None,
    };
    if *result_ty == Ty::Bool {
        return Some(if invert { Plan::BoolNot(cond) } else { Plan::Value(cond) });
    }
    if int_ty_bits(result_ty)? < 2 {
        return None;
    }
    Some(Plan::BoolCast {
        value: cond,
        ty: result_ty.clone(),
        signed,
        invert,
    })
}

/// A value that is `cond` itself or its zero extension.
fn cond_value(func: &Function, value: Value, cond: Value) -> bool {
    value == cond || zext_bool(func, value) == Some(cond)
}

/// Values guaranteed to be 0 or 1 at the result width.
fn bool_valued(func: &Function, value: Value) -> bool {
    bool_valued_depth(func, value, 0)
}

fn bool_valued_depth(func: &Function, value: Value, depth: u32) -> bool {
    if depth > 16 {
        return false;
    }
    if func.values[value].ty == Ty::Bool || matches!(int_const(func, value), Some((_, 0 | 1))) || zext_bool(func, value).is_some() {
        return true;
    }
    let Some(inst) = def_inst(func, value) else {
        return false;
    };
    match &func.insts[inst].kind {
        InstKind::And(a, b) | InstKind::Or(a, b) | InstKind::Xor(a, b) => bool_valued_depth(func, *a, depth + 1) && bool_valued_depth(func, *b, depth + 1),
        InstKind::Select { then_val, else_val, .. } => bool_valued_depth(func, *then_val, depth + 1) && bool_valued_depth(func, *else_val, depth + 1),
        _ => false,
    }
}

/// A full rebuild from `undef` can start from the source aggregate and
/// overwrite only the fields whose values differ, keeping the rest.
fn aggregate_field_updates(func: &Function, value: Value) -> Option<(Value, Vec<(u32, Value)>)> {
    let ty = &func.values[value].ty;
    let total = match ty {
        Ty::Struct { fields, .. } => fields.len(),
        Ty::Array { len, .. } => usize::try_from(*len).ok()?,
        _ => return None,
    };
    if total == 0 || total > 16 {
        return None;
    }
    // Walking backwards keeps the last value inserted at each index.
    let mut fields = std::collections::HashMap::new();
    let mut current = value;
    for _ in 0..=16 {
        if func.values[current].ty != *ty {
            return None;
        }
        let Some(inst) = def_inst(func, current) else {
            break;
        };
        let InstKind::InsertValue { agg, elem, index } = &func.insts[inst].kind else {
            return None;
        };
        if *index as usize >= total {
            return None;
        }
        fields.entry(*index).or_insert(*elem);
        current = *agg;
    }
    if fields.len() != total || !matches!(func.values[current].def, Def::Const(Const::Undef)) {
        return None;
    }
    let mut source = None;
    let mut changed = Vec::new();
    for index in 0..total as u32 {
        let elem = *fields.get(&index)?;
        let unchanged = def_inst(func, elem).and_then(|inst| match &func.insts[inst].kind {
            InstKind::ExtractValue { agg, index: field } if *field == index && func.values[*agg].ty == *ty => Some(*agg),
            _ => None,
        });
        match unchanged {
            Some(agg) => match source {
                Some(existing) if existing != agg => return None,
                Some(_) => {}
                None => source = Some(agg),
            },
            None => changed.push((index, elem)),
        }
    }
    if changed.is_empty() {
        return None;
    }
    Some((source?, changed))
}

/// A chain that fills every field or element of an aggregate from the
/// matching fields of one source value is that source.
fn reconstructed_aggregate(func: &Function, value: Value) -> Option<Value> {
    let ty = &func.values[value].ty;
    let total = match ty {
        Ty::Struct { fields, .. } => fields.len(),
        Ty::Array { len, .. } => usize::try_from(*len).ok()?,
        _ => return None,
    };
    if total == 0 || total > 16 {
        return None;
    }
    let mut seen = std::collections::HashSet::new();
    let mut source = None;
    walk_reconstruction(func, value, ty, total, &mut seen, &mut source, 0)?;
    if seen.len() != total {
        return None;
    }
    source
}

fn walk_reconstruction(func: &Function, value: Value, ty: &Ty, total: usize, seen: &mut std::collections::HashSet<u32>, source: &mut Option<Value>, depth: u32) -> Option<()> {
    if depth > 16 {
        return None;
    }
    let Some(inst) = def_inst(func, value) else {
        return Some(());
    };
    let InstKind::InsertValue { agg, elem, index } = &func.insts[inst].kind else {
        return Some(());
    };
    let elem_inst = def_inst(func, *elem)?;
    let InstKind::ExtractValue {
        agg: field_source,
        index: field_index,
    } = &func.insts[elem_inst].kind
    else {
        return None;
    };
    if field_index != index || *index as usize >= total || func.values[*field_source].ty != *ty {
        return None;
    }
    if !seen.insert(*index) {
        return None;
    }
    match source {
        Some(existing) if existing != field_source => return None,
        Some(_) => {}
        None => *source = Some(*field_source),
    }
    walk_reconstruction(func, *agg, ty, total, seen, source, depth + 1)
}

/// `zext (x != 0)` is `x` when `x` already holds only 0 or 1.
fn zext_boolean_value(func: &Function, operand: Value, result_ty: &Ty) -> Option<Plan> {
    if func.values[operand].ty != Ty::Bool {
        return None;
    }
    let inst = def_inst(func, operand)?;
    let InstKind::Icmp { pred, lhs, rhs } = &func.insts[inst].kind else {
        return None;
    };
    let value = if is_int_zero(func, *rhs) {
        *lhs
    } else if is_int_zero(func, *lhs) {
        *rhs
    } else {
        return None;
    };
    if !bool_valued(func, value) {
        return None;
    }
    let result_bits = int_ty_bits(result_ty)?;
    let value_bits = int_ty_bits(&func.values[value].ty)?;
    if result_bits < 2 || value_bits == 0 {
        return None;
    }
    match pred {
        IcmpPred::Ne => Some(match value_bits.cmp(&result_bits) {
            std::cmp::Ordering::Equal => Plan::Value(value),
            std::cmp::Ordering::Less => Plan::Inst(InstKind::Zext(value)),
            std::cmp::Ordering::Greater => Plan::Inst(InstKind::Trunc(value)),
        }),
        IcmpPred::Eq if value_bits == result_bits => Some(Plan::XorConst(value, 1)),
        _ => None,
    }
}

/// `select c, x, zext(c)` is `x & zext(c)` when `x` is 0/1, and
/// `select c, zext(c), x` is `x | zext(c)`.
fn select_boolean_arm(func: &Function, cond: Value, then_val: Value, else_val: Value) -> Option<Plan> {
    if func.values[cond].ty != Ty::Bool {
        return None;
    }
    if cond_value(func, else_val, cond) && bool_valued(func, then_val) {
        Some(Plan::Inst(InstKind::And(else_val, then_val)))
    } else if cond_value(func, then_val, cond) && bool_valued(func, else_val) {
        Some(Plan::Inst(InstKind::Or(then_val, else_val)))
    } else {
        None
    }
}

/// `(cmp == 0)` and `(cmp != 0)` for a comparison result `cmp`.
fn bool_zero_compare(func: &Function, outer: IcmpPred, lhs: Value, rhs: Value) -> Option<Plan> {
    if !matches!(outer, IcmpPred::Eq | IcmpPred::Ne) {
        return None;
    }
    let cmp = if is_int_zero(func, rhs) {
        lhs
    } else if is_int_zero(func, lhs) {
        rhs
    } else {
        return None;
    };
    if func.values[cmp].ty != Ty::Bool {
        return None;
    }
    let inst = def_inst(func, cmp)?;
    match &func.insts[inst].kind {
        InstKind::Icmp { pred, lhs, rhs } => match outer {
            IcmpPred::Ne => Some(Plan::Value(cmp)),
            IcmpPred::Eq => Some(Plan::Inst(InstKind::Icmp {
                pred: invert_icmp(*pred),
                lhs: *lhs,
                rhs: *rhs,
            })),
            _ => None,
        },
        InstKind::Fcmp { pred, lhs, rhs } => match outer {
            IcmpPred::Ne => Some(Plan::Value(cmp)),
            IcmpPred::Eq => Some(Plan::Inst(InstKind::Fcmp {
                pred: invert_fcmp(*pred),
                lhs: *lhs,
                rhs: *rhs,
            })),
            _ => None,
        },
        _ => None,
    }
}

/// Unsigned comparisons against zero have a fixed truth value or reduce
/// to an equality test (`x <= 0` is `x == 0`, `0 < x` is `x != 0`).
fn unsigned_zero_bound(func: &Function, pred: IcmpPred, lhs: Value, rhs: Value) -> Option<Plan> {
    let rhs_zero = is_int_zero(func, rhs);
    let lhs_zero = is_int_zero(func, lhs);
    if !rhs_zero && !lhs_zero {
        return None;
    }
    let compare_zero = |value, equal| Some(Plan::CompareZero { value, equal });
    match (pred, rhs_zero) {
        (IcmpPred::Ult, true) | (IcmpPred::Ugt, false) => Some(Plan::Const(Const::Int { bits: 1, value: 0 })),
        (IcmpPred::Uge, true) | (IcmpPred::Ule, false) => Some(Plan::Const(Const::Int { bits: 1, value: 1 })),
        (IcmpPred::Ule, true) => compare_zero(lhs, true),
        (IcmpPred::Uge, false) => compare_zero(rhs, true),
        (IcmpPred::Ugt, true) => compare_zero(lhs, false),
        (IcmpPred::Ult, false) => compare_zero(rhs, false),
        _ => None,
    }
}

/// `(u8)(x - 1) < 0x7f` is exactly `(i8)x > 0`. SQLite's UTF-8
/// scanners spell it this way because it steers gcc to a single memory
/// compare; without this fold the subtraction and two extensions stay in
/// the loop.
fn ascii_after_sub_one(func: &Function, pred: IcmpPred, lhs: Value, rhs: Value) -> Option<Plan> {
    if pred != IcmpPred::Ult {
        return None;
    }
    let (_, limit) = int_const(func, rhs)?;
    if limit != 127 {
        return None;
    }
    let trunc = def_inst(func, lhs)?;
    let InstKind::Trunc(sub) = func.insts[trunc].kind else {
        return None;
    };
    let sub_inst = def_inst(func, sub)?;
    let InstKind::Sub(wide, one) = func.insts[sub_inst].kind else {
        return None;
    };
    let (_, one) = int_const(func, one)?;
    if one != 1 {
        return None;
    }
    let (narrow, Widening::Zero) = widening_source(func, wide)? else {
        return None;
    };
    let ty = func.values[narrow].ty.clone();
    if !matches!(ty, Ty::Int { bits: 8 }) || func.values[lhs].ty != ty {
        return None;
    }
    Some(Plan::NarrowCompare {
        value: narrow,
        pred: IcmpPred::Sgt,
        ty,
        constant: 0,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Widening {
    Zero,
    Sign,
}

/// The narrow value a `Zext` or `Sext` promotes, keeping track of which
/// extension was used so order predicates can be mapped soundly.
fn widening_source(func: &Function, value: Value) -> Option<(Value, Widening)> {
    let inst = def_inst(func, value)?;
    match &func.insts[inst].kind {
        InstKind::Zext(source) => Some((*source, Widening::Zero)),
        InstKind::Sext(source) => Some((*source, Widening::Sign)),
        _ => None,
    }
}

/// Compare promoted integers at their original width. Zero extension makes
/// signed predicates behave like unsigned ones in the wide type; sign
/// extension preserves both orders.
fn narrow_widened_compare(func: &Function, pred: IcmpPred, lhs: Value, rhs: Value) -> Option<Plan> {
    let (narrow_lhs, left) = widening_source(func, lhs)?;
    let (narrow_rhs, right) = widening_source(func, rhs)?;
    if func.values[narrow_lhs].ty != func.values[narrow_rhs].ty || !matches!(func.values[narrow_lhs].ty, Ty::Int { .. }) {
        return None;
    }
    let pred = match (left, right) {
        (Widening::Zero, Widening::Zero) => zext_predicate(pred),
        (Widening::Sign, Widening::Sign) => pred,
        _ => return None,
    };
    Some(Plan::Inst(InstKind::Icmp {
        pred,
        lhs: narrow_lhs,
        rhs: narrow_rhs,
    }))
}

fn zext_predicate(pred: IcmpPred) -> IcmpPred {
    match pred {
        IcmpPred::Slt => IcmpPred::Ult,
        IcmpPred::Sle => IcmpPred::Ule,
        IcmpPred::Sgt => IcmpPred::Ugt,
        IcmpPred::Sge => IcmpPred::Uge,
        other => other,
    }
}

fn swap_icmp(pred: IcmpPred) -> IcmpPred {
    match pred {
        IcmpPred::Ult => IcmpPred::Ugt,
        IcmpPred::Ule => IcmpPred::Uge,
        IcmpPred::Ugt => IcmpPred::Ult,
        IcmpPred::Uge => IcmpPred::Ule,
        IcmpPred::Slt => IcmpPred::Sgt,
        IcmpPred::Sle => IcmpPred::Sge,
        IcmpPred::Sgt => IcmpPred::Slt,
        IcmpPred::Sge => IcmpPred::Sle,
        other => other,
    }
}

/// Narrow a widened value compared against a constant. Values outside the
/// source type's range make the comparison constant; values inside it
/// compare exactly at the source width.
fn narrow_widened_const_compare(func: &Function, result_ty: &Ty, pred: IcmpPred, lhs: Value, rhs: Value) -> Option<Plan> {
    let (pred, wide, constant) = if let Some((_, constant)) = int_const(func, rhs) {
        (pred, lhs, constant)
    } else if let Some((_, constant)) = int_const(func, lhs) {
        (swap_icmp(pred), rhs, constant)
    } else {
        return None;
    };
    let (narrow, widening) = widening_source(func, wide)?;
    let narrow_ty = func.values[narrow].ty.clone();
    let (source_bits, bool_source) = match narrow_ty {
        Ty::Int { bits } => (bits, false),
        Ty::Bool => (1, true),
        _ => return None,
    };
    let wide_bits = int_ty_bits(&func.values[wide].ty)?;
    if source_bits == 0 || source_bits >= wide_bits {
        return None;
    }
    let constant = wrap(wide_bits, constant);
    let result_bits = int_ty_bits(result_ty)?;
    let truth = |value: bool| {
        Some(Plan::Const(Const::Int {
            bits: result_bits,
            value: i128::from(value),
        }))
    };
    match widening {
        Widening::Zero => {
            let maximum = mask(source_bits);
            if (constant as u128) & mask(wide_bits) > maximum {
                return truth(eval_icmp(pred, wide_bits, 0, constant));
            }
            if bool_source {
                if !matches!(pred, IcmpPred::Eq | IcmpPred::Ne) {
                    return None;
                }
                let is_one = (constant as u128) & mask(source_bits) != 0;
                return Some(match (pred, is_one) {
                    (IcmpPred::Eq, true) | (IcmpPred::Ne, false) => Plan::Value(narrow),
                    (IcmpPred::Eq, false) | (IcmpPred::Ne, true) => Plan::BoolNot(narrow),
                    _ => unreachable!("boolean comparison is Eq or Ne"),
                });
            }
            let pred = zext_predicate(pred);
            Some(Plan::NarrowCompare {
                value: narrow,
                pred,
                ty: Ty::int(source_bits),
                constant: wrap(source_bits, constant),
            })
        }
        Widening::Sign => {
            if bool_source || !matches!(pred, IcmpPred::Eq | IcmpPred::Ne | IcmpPred::Slt | IcmpPred::Sle | IcmpPred::Sgt | IcmpPred::Sge) {
                return None;
            }
            let (minimum, maximum) = if source_bits >= 128 {
                (i128::MIN, i128::MAX)
            } else {
                let half = 1i128 << (source_bits - 1);
                (-half, half - 1)
            };
            if constant < minimum || constant > maximum {
                return truth(eval_icmp(pred, wide_bits, minimum, constant));
            }
            Some(Plan::NarrowCompare {
                value: narrow,
                pred,
                ty: Ty::int(source_bits),
                constant: wrap(source_bits, constant),
            })
        }
    }
}

fn invert_icmp(pred: IcmpPred) -> IcmpPred {
    match pred {
        IcmpPred::Eq => IcmpPred::Ne,
        IcmpPred::Ne => IcmpPred::Eq,
        IcmpPred::Ult => IcmpPred::Uge,
        IcmpPred::Ule => IcmpPred::Ugt,
        IcmpPred::Ugt => IcmpPred::Ule,
        IcmpPred::Uge => IcmpPred::Ult,
        IcmpPred::Slt => IcmpPred::Sge,
        IcmpPred::Sle => IcmpPred::Sgt,
        IcmpPred::Sgt => IcmpPred::Sle,
        IcmpPred::Sge => IcmpPred::Slt,
    }
}

fn invert_fcmp(pred: FcmpPred) -> FcmpPred {
    match pred {
        FcmpPred::Eq => FcmpPred::Ne,
        FcmpPred::Ne => FcmpPred::Eq,
        FcmpPred::Lt => FcmpPred::Ge,
        FcmpPred::Le => FcmpPred::Gt,
        FcmpPred::Gt => FcmpPred::Le,
        FcmpPred::Ge => FcmpPred::Lt,
        FcmpPred::Uno => FcmpPred::Ord,
        FcmpPred::Ord => FcmpPred::Uno,
    }
}

/// Integer bits for arithmetic reassociation. `Bool` arithmetic is left
/// alone because C only reaches it through explicit bit operations.
fn arith_bits(ty: &Ty) -> Option<u32> {
    match ty {
        Ty::Int { bits } => Some(*bits),
        _ => None,
    }
}

/// Fold `(x op c1) op c2` into `x op (c1 op c2)` so each chain of
/// constant adjustments costs one instruction instead of one per step.
fn reassociate_add(func: &Function, result_ty: &Ty, a: Value, b: Value) -> Option<Plan> {
    let bits = arith_bits(result_ty)?;
    let (base, constant) = constant_and_base(func, bits, a, b)?;
    let inst = def_inst(func, base)?;
    match &func.insts[inst].kind {
        InstKind::Add(x, y) => {
            if let Some((_, cy)) = int_const(func, *y) {
                return Some(add_constant(*x, wrap(bits, constant.wrapping_add(cy))));
            }
            if let Some((_, cx)) = int_const(func, *x) {
                return Some(add_constant(*y, wrap(bits, constant.wrapping_add(cx))));
            }
            None
        }
        InstKind::Sub(x, y) => {
            let (_, cy) = int_const(func, *y)?;
            Some(add_constant(*x, wrap(bits, constant.wrapping_sub(cy))))
        }
        _ => None,
    }
}

fn reassociate_sub(func: &Function, result_ty: &Ty, a: Value, b: Value) -> Option<Plan> {
    let bits = arith_bits(result_ty)?;
    let (_, subtrahend) = int_const(func, b)?;
    let subtrahend = wrap(bits, subtrahend);
    let inst = def_inst(func, a)?;
    match &func.insts[inst].kind {
        InstKind::Add(x, y) => {
            if let Some((_, cy)) = int_const(func, *y) {
                return Some(add_constant(*x, wrap(bits, cy.wrapping_sub(subtrahend))));
            }
            if let Some((_, cx)) = int_const(func, *x) {
                return Some(add_constant(*y, wrap(bits, cx.wrapping_sub(subtrahend))));
            }
            None
        }
        InstKind::Sub(x, y) => {
            let (_, cy) = int_const(func, *y)?;
            Some(sub_constant(*x, wrap(bits, cy.wrapping_add(subtrahend))))
        }
        _ => None,
    }
}

fn reassociate_mul(func: &Function, result_ty: &Ty, a: Value, b: Value) -> Option<Plan> {
    let bits = arith_bits(result_ty)?;
    let (base, constant) = constant_and_base(func, bits, a, b)?;
    let inst = def_inst(func, base)?;
    let InstKind::Mul(x, y) = &func.insts[inst].kind else {
        return None;
    };
    let (other, inner) = match (int_const(func, *x), int_const(func, *y)) {
        (Some((_, value)), _) => (*y, value),
        (_, Some((_, value))) => (*x, value),
        _ => return None,
    };
    let product = wrap(bits, constant.wrapping_mul(inner));
    if product == 0 {
        return Some(Plan::Const(zero_const(result_ty)?));
    }
    if product == 1 {
        return Some(Plan::Value(other));
    }
    if product == -1 {
        return Some(Plan::Inst(InstKind::Neg(other)));
    }
    Some(Plan::MulConst(other, product))
}

fn constant_and_base(func: &Function, bits: u32, a: Value, b: Value) -> Option<(Value, i128)> {
    if let Some((_, value)) = int_const(func, b) {
        Some((a, wrap(bits, value)))
    } else {
        int_const(func, a).map(|(_, value)| (b, wrap(bits, value)))
    }
}

fn add_constant(value: Value, constant: i128) -> Plan {
    if constant == 0 { Plan::Value(value) } else { Plan::AddConst(value, constant) }
}

fn sub_constant(value: Value, constant: i128) -> Plan {
    if constant == 0 { Plan::Value(value) } else { Plan::SubConst(value, constant) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShiftMode {
    Left,
    LogicalRight,
    ArithmeticRight,
}

fn plan_shift(func: &Function, result_ty: &Ty, value: Value, amount: Value, mode: ShiftMode, use_counts: &[u32]) -> Option<Plan> {
    if is_int_zero(func, amount) {
        return Some(Plan::Value(value));
    }
    mask_shift(func, result_ty, value, amount, mode).or_else(|| combine_shifts(func, result_ty, value, amount, mode, use_counts))
}

/// A shift pair that moves bits out and then back leaves a plain mask.
///
/// `(x << a) >> a` keeps the low `width - a` bits, while `(x >> a) << a`
/// clears the low `a` bits. Arithmetic right shift only has the second
/// form: shifting its sign fill back left still clears the same bits.
fn mask_shift(func: &Function, result_ty: &Ty, value: Value, amount: Value, mode: ShiftMode) -> Option<Plan> {
    let bits = arith_bits(result_ty)?;
    let (_, outer) = int_const(func, amount)?;
    if outer < 0 || outer as u128 >= u128::from(bits) {
        return None;
    }
    let inst = def_inst(func, value)?;
    let (value, inner_amount) = match (&func.insts[inst].kind, mode) {
        (InstKind::Shl(x, by), ShiftMode::LogicalRight) => (*x, *by),
        (InstKind::LShr(x, by) | InstKind::AShr(x, by), ShiftMode::Left) => (*x, *by),
        _ => return None,
    };
    let (_, inner) = int_const(func, inner_amount)?;
    if inner != outer {
        return None;
    }
    if inner == 0 {
        return Some(Plan::Value(value));
    }
    let shift = outer as u32;
    let mask = match mode {
        ShiftMode::LogicalRight => mask(bits - shift) as i128,
        ShiftMode::Left => wrap(bits, (!mask(shift)) as i128),
        ShiftMode::ArithmeticRight => return None,
    };
    Some(Plan::AndConst(value, mask))
}

/// Merge nested shifts in the same direction. Amounts that reach the type
/// width stay untouched because oversized shifts are undefined.
fn combine_shifts(func: &Function, result_ty: &Ty, value: Value, amount: Value, mode: ShiftMode, use_counts: &[u32]) -> Option<Plan> {
    if use_counts.get(value.index()).copied().unwrap_or(0) != 1 {
        return None;
    }
    let bits = arith_bits(result_ty)?;
    let (_, outer) = int_const(func, amount)?;
    if outer < 0 || outer as u128 >= u128::from(bits) {
        return None;
    }
    let inst = def_inst(func, value)?;
    let (inner_value, inner_amount, inner_mode) = match &func.insts[inst].kind {
        InstKind::Shl(x, by) => (*x, *by, ShiftMode::Left),
        InstKind::LShr(x, by) => (*x, *by, ShiftMode::LogicalRight),
        InstKind::AShr(x, by) => (*x, *by, ShiftMode::ArithmeticRight),
        _ => return None,
    };
    if inner_mode != mode {
        return None;
    }
    let (_, inner) = int_const(func, inner_amount)?;
    if inner < 0 || inner as u128 >= u128::from(bits) {
        return None;
    }
    let total = inner + outer;
    if total == 0 {
        return Some(Plan::Value(inner_value));
    }
    if total as u128 >= u128::from(bits) {
        return None;
    }
    let total = total as u32;
    Some(match mode {
        ShiftMode::Left => Plan::ShlBy(inner_value, total),
        ShiftMode::LogicalRight => Plan::LshrBy(inner_value, total),
        ShiftMode::ArithmeticRight => Plan::AshrBy(inner_value, total),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bitwise {
    And,
    Or,
    Xor,
}

/// Doing the bitwise operation on the booleans saves one extension.
fn zext_bitwise(func: &Function, result_ty: &Ty, a: Value, b: Value, op: Bitwise) -> Option<Plan> {
    if !matches!(result_ty, Ty::Int { bits } if *bits >= 2) {
        return None;
    }
    Some(Plan::ZextBoolOp {
        op,
        lhs: zext_bool(func, a)?,
        rhs: zext_bool(func, b)?,
        ty: result_ty.clone(),
    })
}

/// Merge `(x op c1) op c2` into one bitwise operation with a combined
/// mask. The inner operation may stay alive for other users; nothing is
/// added to the block in that case.
fn combine_bitwise(func: &Function, result_ty: &Ty, a: Value, b: Value, mode: Bitwise) -> Option<Plan> {
    let bits = arith_bits(result_ty)?;
    let (base, outer) = constant_and_base(func, bits, a, b)?;
    let inst = def_inst(func, base)?;
    let (value, inner) = match (&func.insts[inst].kind, mode) {
        (InstKind::And(x, y), Bitwise::And) | (InstKind::Or(x, y), Bitwise::Or) | (InstKind::Xor(x, y), Bitwise::Xor) => constant_operand(func, *x, *y)?,
        _ => return None,
    };
    let combined = wrap(
        bits,
        match mode {
            Bitwise::And => outer & inner,
            Bitwise::Or => outer | inner,
            Bitwise::Xor => outer ^ inner,
        },
    );
    if combined == 0 {
        return match mode {
            Bitwise::And => Some(Plan::Const(zero_const(result_ty)?)),
            Bitwise::Or | Bitwise::Xor => Some(Plan::Value(value)),
        };
    }
    if combined == -1 {
        return match mode {
            Bitwise::And => Some(Plan::Value(value)),
            Bitwise::Or => Some(Plan::Const(all_ones_const(result_ty)?)),
            Bitwise::Xor => Some(Plan::Inst(InstKind::Not(value))),
        };
    }
    Some(match mode {
        Bitwise::And => Plan::AndConst(value, combined),
        Bitwise::Or => Plan::OrConst(value, combined),
        Bitwise::Xor => Plan::XorConst(value, combined),
    })
}

fn constant_operand(func: &Function, lhs: Value, rhs: Value) -> Option<(Value, i128)> {
    if let Some((_, value)) = int_const(func, rhs) {
        Some((lhs, value))
    } else {
        int_const(func, lhs).map(|(_, value)| (rhs, value))
    }
}

/// The shift amount for a positive power-of-two constant.
fn int_pow2(func: &Function, v: Value) -> Option<u32> {
    let (bits, value) = int_const(func, v)?;
    if bits == 0 || bits > 128 {
        return None;
    }
    let value = wrap(bits, value);
    if value <= 0 {
        return None;
    }
    let value = value as u128;
    value.is_power_of_two().then(|| value.trailing_zeros())
}

fn int_const(func: &Function, v: Value) -> Option<(u32, i128)> {
    match func.const_value(v) {
        Some(Const::Int { bits, value }) => Some((*bits, *value)),
        _ => None,
    }
}

fn is_int_zero(func: &Function, v: Value) -> bool {
    matches!(int_const(func, v), Some((_, 0)))
}

fn is_int_one(func: &Function, v: Value) -> bool {
    matches!(int_const(func, v), Some((_, 1)))
}

/// Whether every bit of the constant's own width is set.
fn is_all_ones(func: &Function, v: Value) -> bool {
    int_const(func, v).is_some_and(|(bits, value)| wrap(bits, value) == -1)
}

fn is_float(func: &Function, v: Value) -> bool {
    matches!(func.values[v].ty, Ty::Float(_))
}

fn int_ty_bits(ty: &Ty) -> Option<u32> {
    match ty {
        Ty::Int { bits } => Some(*bits),
        Ty::Bool => Some(1),
        _ => None,
    }
}

fn zero_const(ty: &Ty) -> Option<Const> {
    Some(Const::Int { bits: int_ty_bits(ty)?, value: 0 })
}

fn all_ones_const(ty: &Ty) -> Option<Const> {
    Some(Const::Int {
        bits: int_ty_bits(ty)?,
        value: -1,
    })
}

fn push_int_const(func: &mut Function, ty: &Ty, value: i128) -> Value {
    let bits = int_ty_bits(ty).expect("integer plan");
    func.values.push(ValueData {
        ty: ty.clone(),
        def: Def::Const(Const::Int { bits, value }),
    })
}

/// Whether `value`'s low `result_ty` bits are all ones, making an
/// `And` redundant under a narrower `Trunc`.
fn trunc_mask_covers(func: &Function, value: Value, result_ty: &Ty) -> bool {
    let Some((_, mask_value)) = int_const(func, value) else {
        return false;
    };
    let Some(bits) = int_ty_bits(result_ty) else {
        return false;
    };
    if bits == 0 || bits > 128 {
        return false;
    }
    let low = (mask_value as u128) & mask(bits);
    low == mask(bits)
}

#[cfg(test)]
mod tests;

