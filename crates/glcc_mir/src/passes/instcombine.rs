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
use crate::ty::Ty;

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
    SignedPow2 {
        value: Value,
        shift: u32,
        remainder: bool,
    },
    /// Compare a widened value with a constant after narrowing both to
    /// the original width.
    NarrowCompare {
        value: Value,
        pred: IcmpPred,
        ty: Ty,
        constant: i128,
    },
    /// Extend a boolean (optionally negated) to an integer result.
    BoolCast {
        value: Value,
        ty: Ty,
        signed: bool,
        invert: bool,
    },
    /// Bitwise operation on two booleans, zero-extended to the result.
    ZextBoolOp {
        op: Bitwise,
        lhs: Value,
        rhs: Value,
        ty: Ty,
    },
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
            } else if let Some(shift) = int_pow2(func, *a) {
                Some(Plan::ShlBy(*b, shift))
            } else {
                None
            }
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
                combine_bitwise(func, result_ty, *a, *b, Bitwise::And)
                    .or_else(|| zext_bitwise(func, result_ty, *a, *b, Bitwise::And))
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
                combine_bitwise(func, result_ty, *a, *b, Bitwise::Or)
                    .or_else(|| zext_bitwise(func, result_ty, *a, *b, Bitwise::Or))
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
                combine_bitwise(func, result_ty, *a, *b, Bitwise::Xor)
                    .or_else(|| zext_bitwise(func, result_ty, *a, *b, Bitwise::Xor))
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
            match &func.insts[inner].kind {
                InstKind::PtrToInt(x) if func.values[*x].ty == *result_ty => Some(Plan::Value(*x)),
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
                None => select_bool_constant(func, result_ty, *cond, *then_val, *else_val)
                    .or_else(|| select_boolean_arm(func, *cond, *then_val, *else_val)),
            }
        }
        InstKind::GetElementPtr { base, indices } if indices.is_empty() && func.values[*base].ty == *result_ty => Some(Plan::Value(*base)),
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
    Some(Plan::SignedPow2 {
        value,
        shift,
        remainder,
    })
}

fn push_temp(
    func: &mut Function,
    block: Block,
    kind: InstKind,
    ty: Ty,
    loc: Option<crate::debug::Location>,
    out: &mut Vec<Inst>,
) -> Value {
    let next = Inst::from_usize(func.insts.len());
    let value = func.values.push(ValueData {
        ty,
        def: Def::Inst(next),
    });
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
            Plan::Inst(InstKind::Icmp {
                pred,
                lhs: value,
                rhs: constant,
            })
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
            let loc = func.insts[inst].loc.clone();
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
        Plan::BoolCast { value, ty, signed, invert } => {
            let loc = func.insts[inst].loc.clone();
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
                    loc.clone(),
                    out,
                )
            } else {
                value
            };
            let kind = if signed { InstKind::Sext(cond) } else { InstKind::Zext(cond) };
            push_temp(func, block, kind, ty, loc, out)
        }
        Plan::ZextBoolOp { op, lhs, rhs, ty } => {
            let loc = func.insts[inst].loc.clone();
            let kind = match op {
                Bitwise::And => InstKind::And(lhs, rhs),
                Bitwise::Or => InstKind::Or(lhs, rhs),
                Bitwise::Xor => InstKind::Xor(lhs, rhs),
            };
            let inner = push_temp(func, block, kind, Ty::Bool, loc.clone(), out);
            push_temp(func, block, InstKind::Zext(inner), ty, loc, out)
        }
        Plan::SignedPow2 { value, shift, remainder } => {
            let ty = func.values[result].ty.clone();
            let bits = int_ty_bits(&ty).expect("signed integer division");
            let loc = func.insts[inst].loc.clone();
            let sign_amount = push_int_const(func, &ty, i128::from(bits - 1));
            let sign = push_temp(func, block, InstKind::AShr(value, sign_amount), ty.clone(), loc.clone(), out);
            let bias_amount = push_int_const(func, &ty, i128::from(bits - shift));
            let bias = push_temp(func, block, InstKind::LShr(sign, bias_amount), ty.clone(), loc.clone(), out);
            let adjusted = push_temp(func, block, InstKind::Add(value, bias), ty.clone(), loc.clone(), out);
            let shift_amount = push_int_const(func, &ty, i128::from(shift));
            let quotient =
                push_temp(func, block, InstKind::AShr(adjusted, shift_amount), ty.clone(), loc.clone(), out);
            if remainder {
                let scaled = push_temp(func, block, InstKind::Shl(quotient, shift_amount), ty.clone(), loc.clone(), out);
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
fn select_bool_constant(
    func: &Function,
    result_ty: &Ty,
    cond: Value,
    then_val: Value,
    else_val: Value,
) -> Option<Plan> {
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
    if func.values[value].ty == Ty::Bool
        || matches!(int_const(func, value), Some((_, 0 | 1)))
        || zext_bool(func, value).is_some()
    {
        return true;
    }
    let Some(inst) = def_inst(func, value) else {
        return false;
    };
    match &func.insts[inst].kind {
        InstKind::And(a, b) | InstKind::Or(a, b) | InstKind::Xor(a, b) => {
            bool_valued_depth(func, *a, depth + 1) && bool_valued_depth(func, *b, depth + 1)
        }
        InstKind::Select { then_val, else_val, .. } => {
            bool_valued_depth(func, *then_val, depth + 1)
                && bool_valued_depth(func, *else_val, depth + 1)
        }
        _ => false,
    }
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
    let compare_zero = |value, equal| {
        Some(Plan::CompareZero {
            value,
            equal,
        })
    };
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
fn narrow_widened_const_compare(
    func: &Function,
    result_ty: &Ty,
    pred: IcmpPred,
    lhs: Value,
    rhs: Value,
) -> Option<Plan> {
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
    let truth = |value: bool| Some(Plan::Const(Const::Int { bits: result_bits, value: i128::from(value) }));
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
            if bool_source || !matches!(
                pred,
                IcmpPred::Eq | IcmpPred::Ne | IcmpPred::Slt | IcmpPred::Sle | IcmpPred::Sgt | IcmpPred::Sge
            ) {
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
    if constant == 0 {
        Plan::Value(value)
    } else {
        Plan::AddConst(value, constant)
    }
}

fn sub_constant(value: Value, constant: i128) -> Plan {
    if constant == 0 {
        Plan::Value(value)
    } else {
        Plan::SubConst(value, constant)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShiftMode {
    Left,
    LogicalRight,
    ArithmeticRight,
}

fn plan_shift(
    func: &Function,
    result_ty: &Ty,
    value: Value,
    amount: Value,
    mode: ShiftMode,
    use_counts: &[u32],
) -> Option<Plan> {
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
fn combine_shifts(
    func: &Function,
    result_ty: &Ty,
    value: Value,
    amount: Value,
    mode: ShiftMode,
    use_counts: &[u32],
) -> Option<Plan> {
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
mod tests {
    use super::*;
    use crate::ir::Terminator;
    use crate::ty::Signature;
    use crate::verify::verify_func;

    fn param(func: &mut Function, block: Block, ty: Ty, index: u32) -> Value {
        let v = func.values.push(ValueData {
            ty,
            def: Def::Param(block, index),
        });
        func.blocks[block].params.push(v);
        v
    }

    fn emit(func: &mut Function, block: Block, kind: InstKind, ty: Ty) -> Value {
        let result = func.values.push(ValueData {
            ty,
            def: Def::Inst(Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(InstData {
            block,
            kind,
            result: Some(result),
            loc: None,
        });
        func.values[result].def = Def::Inst(inst);
        func.blocks[block].insts.push(inst);
        result
    }

    fn iconst(func: &mut Function, bits: u32, value: i128) -> Value {
        func.push_const(Const::Int { bits, value })
    }

    fn returned(func: &Function) -> Value {
        match &func.blocks[func.entry].term {
            Some(Terminator::Return { val: Some(v) }) => *v,
            other => panic!("expected a value return, got {other:?}"),
        }
    }

    fn run(func: &mut Function) -> bool {
        let changed = InstCombine.run(func);
        let errors = verify_func(func);
        assert!(errors.is_empty(), "verifier: {errors:?}");
        changed
    }

    fn int_func() -> (Function, Block, Value) {
        let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(32), 0);
        (func, entry, x)
    }

    #[test]
    fn add_of_zero_uses_the_other_operand() {
        let (mut func, entry, x) = int_func();
        let zero = iconst(&mut func, 32, 0);
        let sum = emit(&mut func, entry, InstKind::Add(x, zero), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        assert!(run(&mut func));
        assert_eq!(returned(&func), x);
        assert!(!run(&mut func), "second run is a no-op");
    }

    #[test]
    fn mul_by_zero_becomes_a_typed_zero() {
        let (mut func, entry, x) = int_func();
        let zero = iconst(&mut func, 32, 0);
        let product = emit(&mut func, entry, InstKind::Mul(x, zero), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(product) });
        assert!(run(&mut func));
        let out = returned(&func);
        assert_eq!(func.values[out].ty, Ty::int(32));
        assert_eq!(func.const_value(out), Some(&Const::Int { bits: 32, value: 0 }));
    }

    #[test]
    fn self_subtraction_and_xor_fold_to_zero() {
        let (mut func, entry, x) = int_func();
        let out = emit(&mut func, entry, InstKind::Sub(x, x), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(out) });
        assert!(run(&mut func));
        assert_eq!(func.const_value(returned(&func)), Some(&Const::Int { bits: 32, value: 0 }));

        let (mut func, entry, x) = int_func();
        let out = emit(&mut func, entry, InstKind::Xor(x, x), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(out) });
        assert!(run(&mut func));
        assert_eq!(func.const_value(returned(&func)), Some(&Const::Int { bits: 32, value: 0 }));
    }

    #[test]
    fn division_by_one_keeps_the_dividend() {
        for build in [InstKind::Udiv, InstKind::Sdiv] {
            let (mut func, entry, x) = int_func();
            let one = iconst(&mut func, 32, 1);
            let quotient = emit(&mut func, entry, build(x, one), Ty::int(32));
            func.blocks[entry].term = Some(Terminator::Return { val: Some(quotient) });
            assert!(run(&mut func));
            assert_eq!(returned(&func), x);
        }
    }

    #[test]
    fn remainder_by_one_folds_to_zero() {
        for build in [InstKind::Urem, InstKind::Srem] {
            let (mut func, entry, x) = int_func();
            let one = iconst(&mut func, 32, 1);
            let remainder = emit(&mut func, entry, build(x, one), Ty::int(32));
            func.blocks[entry].term = Some(Terminator::Return { val: Some(remainder) });
            assert!(run(&mut func));
            let out = returned(&func);
            assert_eq!(func.values[out].ty, Ty::int(32));
            assert_eq!(func.const_value(out), Some(&Const::Int { bits: 32, value: 0 }));
        }
    }

    #[test]
    fn select_with_equal_arms_keeps_one() {
        let sig = Signature::new(vec![Ty::Bool, Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let cond = param(&mut func, entry, Ty::Bool, 0);
        let x = param(&mut func, entry, Ty::int(32), 1);
        let sel = emit(&mut func, entry, InstKind::Select { cond, then_val: x, else_val: x }, Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
        assert!(run(&mut func));
        assert_eq!(returned(&func), x);
    }

    #[test]
    fn select_with_a_condition_arm_becomes_bitwise() {
        let sig = Signature::new(vec![Ty::Bool, Ty::Bool], Ty::Bool);
        let (mut func, entry) = Function::new("f", sig);
        let cond = param(&mut func, entry, Ty::Bool, 0);
        let x = param(&mut func, entry, Ty::Bool, 1);
        let sel = emit(
            &mut func,
            entry,
            InstKind::Select { cond, then_val: x, else_val: cond },
            Ty::Bool,
        );
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt boolean op");
        assert!(matches!(func.insts[inst].kind, InstKind::And(a, b) if a == cond && b == x));

        let sig = Signature::new(vec![Ty::Bool, Ty::Bool], Ty::Bool);
        let (mut func, entry) = Function::new("f", sig);
        let cond = param(&mut func, entry, Ty::Bool, 0);
        let x = param(&mut func, entry, Ty::Bool, 1);
        let sel = emit(
            &mut func,
            entry,
            InstKind::Select { cond, then_val: cond, else_val: x },
            Ty::Bool,
        );
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt boolean op");
        assert!(matches!(func.insts[inst].kind, InstKind::Or(a, b) if a == cond && b == x));
    }

    #[test]
    fn select_of_zero_extended_arms_becomes_bitwise() {
        let sig = Signature::new(vec![Ty::Bool, Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let cond = param(&mut func, entry, Ty::Bool, 0);
        let x = param(&mut func, entry, Ty::Bool, 1);
        let cond_wide = emit(&mut func, entry, InstKind::Zext(cond), Ty::int(32));
        let x_wide = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
        let sel = emit(
            &mut func,
            entry,
            InstKind::Select { cond, then_val: x_wide, else_val: cond_wide },
            Ty::int(32),
        );
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt bitwise op");
        assert!(matches!(func.insts[inst].kind, InstKind::And(a, b) if a == cond_wide && b == x_wide));

        let sig = Signature::new(vec![Ty::Bool, Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let cond = param(&mut func, entry, Ty::Bool, 0);
        let x = param(&mut func, entry, Ty::Bool, 1);
        let cond_wide = emit(&mut func, entry, InstKind::Zext(cond), Ty::int(32));
        let x_wide = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
        let sel = emit(
            &mut func,
            entry,
            InstKind::Select { cond, then_val: cond_wide, else_val: x_wide },
            Ty::int(32),
        );
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt bitwise op");
        assert!(matches!(func.insts[inst].kind, InstKind::Or(a, b) if a == cond_wide && b == x_wide));
    }

    #[test]
    fn zext_of_a_boolean_comparison_uses_the_boolean_value() {
        let sig = Signature::new(vec![Ty::Bool, Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let a = param(&mut func, entry, Ty::Bool, 0);
        let b = param(&mut func, entry, Ty::Bool, 1);
        let a_wide = emit(&mut func, entry, InstKind::Zext(a), Ty::int(32));
        let b_wide = emit(&mut func, entry, InstKind::Zext(b), Ty::int(32));
        let both = emit(&mut func, entry, InstKind::And(a_wide, b_wide), Ty::int(32));
        let zero = iconst(&mut func, 32, 0);
        let not_zero = emit(
            &mut func,
            entry,
            InstKind::Icmp { pred: IcmpPred::Ne, lhs: both, rhs: zero },
            Ty::Bool,
        );
        match zext_boolean_value(&func, not_zero, &Ty::int(32)) {
            Some(Plan::Value(value)) => assert_eq!(value, both),
            Some(_) => panic!("expected the boolean value"),
            None => panic!("nonzero boolean comparison was not recognized"),
        }

        let is_zero = emit(
            &mut func,
            entry,
            InstKind::Icmp { pred: IcmpPred::Eq, lhs: both, rhs: zero },
            Ty::Bool,
        );
        match zext_boolean_value(&func, is_zero, &Ty::int(32)) {
            Some(Plan::XorConst(value, 1)) => assert_eq!(value, both),
            _ => panic!("expected an inverted boolean"),
        }
    }

    #[test]
    fn select_with_constant_condition_picks_an_arm() {
        let (mut func, entry, x) = int_func();
        let cond = iconst(&mut func, 1, 1);
        let other = iconst(&mut func, 32, 99);
        let sel = emit(
            &mut func,
            entry,
            InstKind::Select {
                cond,
                then_val: x,
                else_val: other,
            },
            Ty::int(32),
        );
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
        assert!(run(&mut func));
        assert_eq!(returned(&func), x);
    }

    /// `if (x < y)` lowers to `icmp` -> `zext` -> `icmp.Ne 0`; the
    /// round-trip through `i32` must collapse back to the condition.
    #[test]
    fn zext_compared_against_zero_collapses() {
        for (pred, truthy) in [(IcmpPred::Ne, true), (IcmpPred::Eq, false)] {
            let sig = Signature::new(vec![Ty::int(32), Ty::int(32)], Ty::Bool);
            let (mut func, entry) = Function::new("f", sig);
            let a = param(&mut func, entry, Ty::int(32), 0);
            let b = param(&mut func, entry, Ty::int(32), 1);
            let cmp = emit(
                &mut func,
                entry,
                InstKind::Icmp {
                    pred: IcmpPred::Slt,
                    lhs: a,
                    rhs: b,
                },
                Ty::Bool,
            );
            let wide = emit(&mut func, entry, InstKind::Zext(cmp), Ty::int(32));
            let zero = iconst(&mut func, 32, 0);
            let test = emit(&mut func, entry, InstKind::Icmp { pred, lhs: wide, rhs: zero }, Ty::Bool);
            func.blocks[entry].term = Some(Terminator::Return { val: Some(test) });
            assert!(run(&mut func), "{pred:?}");
            let out = returned(&func);
            if truthy {
                assert_eq!(out, cmp, "{pred:?} keeps the condition");
            } else {
                assert_ne!(out, cmp, "{pred:?} negates the condition");
                let inst = def_inst(&func, out).expect("rebuilt comparison");
                match &func.insts[inst].kind {
                    InstKind::Icmp { pred, lhs, rhs } => {
                        assert_eq!(*pred, IcmpPred::Eq);
                        assert_eq!(*lhs, cmp);
                        assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 1, value: 0 }));
                    }
                    other => panic!("expected icmp, got {other:?}"),
                }
            }
        }
    }

    #[test]
    fn zext_compared_against_zero_works_with_either_side() {
        let sig = Signature::new(vec![Ty::int(32), Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let a = param(&mut func, entry, Ty::int(32), 0);
        let b = param(&mut func, entry, Ty::int(32), 1);
        let cmp = emit(
            &mut func,
            entry,
            InstKind::Icmp {
                pred: IcmpPred::Sgt,
                lhs: a,
                rhs: b,
            },
            Ty::Bool,
        );
        let wide = emit(&mut func, entry, InstKind::Zext(cmp), Ty::int(64));
        let zero = iconst(&mut func, 64, 0);
        let test = emit(
            &mut func,
            entry,
            InstKind::Icmp {
                pred: IcmpPred::Ne,
                lhs: zero,
                rhs: wide,
            },
            Ty::Bool,
        );
        func.blocks[entry].term = Some(Terminator::Return { val: Some(test) });
        assert!(run(&mut func));
        assert_eq!(returned(&func), cmp);
    }

    #[test]
    fn truncation_of_a_widened_value_round_trips() {
        let sig = Signature::new(vec![Ty::int(8)], Ty::int(8));
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(8), 0);
        let wide = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
        let narrow = emit(&mut func, entry, InstKind::Trunc(wide), Ty::int(8));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(narrow) });
        assert!(run(&mut func));
        assert_eq!(returned(&func), x);
    }

    /// `(unsigned char)(x & 0xff)`: the mask covers the truncation width.
    #[test]
    fn truncation_absorbs_a_covering_mask() {
        let sig = Signature::new(vec![Ty::int(32)], Ty::int(8));
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(32), 0);
        let mask = iconst(&mut func, 32, 255);
        let masked = emit(&mut func, entry, InstKind::And(x, mask), Ty::int(32));
        let narrow = emit(&mut func, entry, InstKind::Trunc(masked), Ty::int(8));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(narrow) });
        assert!(run(&mut func));
        let out = returned(&func);
        let inst = def_inst(&func, out).expect("rebuilt truncation");
        match &func.insts[inst].kind {
            InstKind::Trunc(operand) => assert_eq!(*operand, x),
            other => panic!("expected truncation, got {other:?}"),
        }
    }

    #[test]
    fn nested_widening_collapses_to_one_step() {
        let sig = Signature::new(vec![Ty::int(8)], Ty::int(64));
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(8), 0);
        let wide = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
        let wider = emit(&mut func, entry, InstKind::Zext(wide), Ty::int(64));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(wider) });
        assert!(run(&mut func));
        let out = returned(&func);
        assert_eq!(func.values[out].ty, Ty::int(64));
        let inst = def_inst(&func, out).expect("rebuilt widening");
        match &func.insts[inst].kind {
            InstKind::Zext(operand) => assert_eq!(*operand, x),
            other => panic!("expected widening, got {other:?}"),
        }
    }

    #[test]
    fn self_comparisons_fold_to_constants() {
        for (pred, truth) in [
            (IcmpPred::Eq, true),
            (IcmpPred::Ne, false),
            (IcmpPred::Ult, false),
            (IcmpPred::Ule, true),
            (IcmpPred::Sgt, false),
            (IcmpPred::Sge, true),
            (IcmpPred::Slt, false),
            (IcmpPred::Sle, true),
            (IcmpPred::Ugt, false),
            (IcmpPred::Uge, true),
        ] {
            let (mut func, entry, x) = int_func();
            let cmp = emit(&mut func, entry, InstKind::Icmp { pred, lhs: x, rhs: x }, Ty::Bool);
            func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
            assert!(run(&mut func), "{pred:?}");
            let out = returned(&func);
            assert_eq!(func.values[out].ty, Ty::Bool, "{pred:?}");
            assert_eq!(
                func.const_value(out),
                Some(&Const::Int {
                    bits: 1,
                    value: i128::from(truth)
                }),
                "{pred:?}"
            );
        }
    }

    #[test]
    fn doubled_unary_operators_cancel() {
        let (mut func, entry, x) = int_func();
        let a = emit(&mut func, entry, InstKind::Neg(x), Ty::int(32));
        let b = emit(&mut func, entry, InstKind::Neg(a), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(b) });
        assert!(run(&mut func));
        assert_eq!(returned(&func), x);

        let (mut func, entry, x) = int_func();
        let a = emit(&mut func, entry, InstKind::Not(x), Ty::int(32));
        let b = emit(&mut func, entry, InstKind::Not(a), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(b) });
        assert!(run(&mut func));
        assert_eq!(returned(&func), x);
    }

    #[test]
    fn dead_results_wait_for_dce() {
        // Nothing uses the sum, so there is nothing to rewrite; DCE in
        // the pipeline removes the instruction, and rewriting it here
        // would only spin the fixpoint.
        let sig = Signature::new(vec![], Ty::Void);
        let (mut func, entry) = Function::new("f", sig);
        let x = iconst(&mut func, 32, 1);
        let zero = iconst(&mut func, 32, 0);
        let _ = emit(&mut func, entry, InstKind::Add(x, zero), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: None });
        assert!(!run(&mut func));
    }

    #[test]
    fn doubled_add_becomes_a_shift() {
        let (mut func, entry, x) = int_func();
        let sum = emit(&mut func, entry, InstKind::Add(x, x), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt shift");
        match &func.insts[inst].kind {
            InstKind::Shl(lhs, amount) => {
                assert_eq!(*lhs, x);
                assert_eq!(func.const_value(*amount), Some(&Const::Int { bits: 32, value: 1 }));
            }
            other => panic!("expected shift, got {other:?}"),
        }
    }

    #[test]
    fn signed_power_of_two_division_avoids_division() {
        for remainder in [false, true] {
            let (mut func, entry, x) = int_func();
            let four = iconst(&mut func, 32, 4);
            let kind = if remainder {
                InstKind::Srem(x, four)
            } else {
                InstKind::Sdiv(x, four)
            };
            let out = emit(&mut func, entry, kind, Ty::int(32));
            func.blocks[entry].term = Some(Terminator::Return { val: Some(out) });
            assert!(run(&mut func));
            assert!(!func.blocks[entry].insts.iter().any(|inst| matches!(
                func.insts[*inst].kind,
                InstKind::Sdiv(..) | InstKind::Srem(..)
            )));
            assert!(crate::verify::verify_func(&func).is_empty());
        }
    }

    #[test]
    fn reassociates_constant_additions() {
        let (mut func, entry, x) = int_func();
        let seven = iconst(&mut func, 32, 7);
        let five = iconst(&mut func, 32, 5);
        let inner = emit(&mut func, entry, InstKind::Add(x, seven), Ty::int(32));
        let outer = emit(&mut func, entry, InstKind::Add(inner, five), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(outer) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt addition");
        match &func.insts[inst].kind {
            InstKind::Add(lhs, rhs) => {
                assert_eq!(*lhs, x);
                assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 32, value: 12 }));
            }
            other => panic!("expected addition, got {other:?}"),
        }
    }

    #[test]
    fn reassociation_wraps_like_the_original_operations() {
        let (mut func, entry, x) = int_func();
        let max = iconst(&mut func, 32, i128::from(i32::MAX));
        let one = iconst(&mut func, 32, 1);
        let inner = emit(&mut func, entry, InstKind::Add(x, max), Ty::int(32));
        let outer = emit(&mut func, entry, InstKind::Add(inner, one), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(outer) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt addition");
        match &func.insts[inst].kind {
            InstKind::Add(lhs, rhs) => {
                assert_eq!(*lhs, x);
                assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 32, value: i128::from(i32::MIN) }));
            }
            other => panic!("expected addition, got {other:?}"),
        }
    }

    #[test]
    fn reassociates_constant_multiplications() {
        let (mut func, entry, x) = int_func();
        let six = iconst(&mut func, 32, 6);
        let four = iconst(&mut func, 32, 4);
        let inner = emit(&mut func, entry, InstKind::Mul(x, six), Ty::int(32));
        let outer = emit(&mut func, entry, InstKind::Mul(inner, four), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(outer) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt product");
        match &func.insts[inst].kind {
            InstKind::Mul(lhs, rhs) => {
                assert_eq!(*lhs, x);
                assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 32, value: 24 }));
            }
            other => panic!("expected product, got {other:?}"),
        }
    }

    #[test]
    fn nested_shifts_combine() {
        for (mode, build) in [
            (ShiftMode::Left, InstKind::Shl as fn(Value, Value) -> InstKind),
            (ShiftMode::LogicalRight, InstKind::LShr as fn(Value, Value) -> InstKind),
            (ShiftMode::ArithmeticRight, InstKind::AShr as fn(Value, Value) -> InstKind),
        ] {
            let (mut func, entry, x) = int_func();
            let two = iconst(&mut func, 32, 2);
            let three = iconst(&mut func, 32, 3);
            let inner = emit(&mut func, entry, build(x, two), Ty::int(32));
            let outer = emit(&mut func, entry, build(inner, three), Ty::int(32));
            func.blocks[entry].term = Some(Terminator::Return { val: Some(outer) });
            assert!(run(&mut func), "{mode:?}");
            let inst = def_inst(&func, returned(&func)).expect("rebuilt shift");
            match &func.insts[inst].kind {
                InstKind::Shl(lhs, by) => {
                    assert_eq!(mode, ShiftMode::Left);
                    assert_eq!(*lhs, x);
                    assert_eq!(func.const_value(*by), Some(&Const::Int { bits: 32, value: 5 }));
                }
                InstKind::LShr(lhs, by) => {
                    assert_eq!(mode, ShiftMode::LogicalRight);
                    assert_eq!(*lhs, x);
                    assert_eq!(func.const_value(*by), Some(&Const::Int { bits: 32, value: 5 }));
                }
                InstKind::AShr(lhs, by) => {
                    assert_eq!(mode, ShiftMode::ArithmeticRight);
                    assert_eq!(*lhs, x);
                    assert_eq!(func.const_value(*by), Some(&Const::Int { bits: 32, value: 5 }));
                }
                other => panic!("expected shift, got {other:?}"),
            }
        }
    }

    #[test]
    fn shared_inner_shifts_are_not_duplicated() {
        let (mut func, entry, x) = int_func();
        let two = iconst(&mut func, 32, 2);
        let three = iconst(&mut func, 32, 3);
        let inner = emit(&mut func, entry, InstKind::Shl(x, two), Ty::int(32));
        let outer = emit(&mut func, entry, InstKind::Shl(inner, three), Ty::int(32));
        let sum = emit(&mut func, entry, InstKind::Add(outer, inner), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        assert!(!run(&mut func));
        let inst = def_inst(&func, outer).expect("outer shift");
        match &func.insts[inst].kind {
            InstKind::Shl(lhs, _) => assert_eq!(*lhs, inner),
            other => panic!("expected shift, got {other:?}"),
        }
    }

    #[test]
    fn oversized_shift_chains_are_left_alone() {
        let (mut func, entry, x) = int_func();
        let wide = iconst(&mut func, 32, 30);
        let five = iconst(&mut func, 32, 5);
        let inner = emit(&mut func, entry, InstKind::Shl(x, wide), Ty::int(32));
        let outer = emit(&mut func, entry, InstKind::Shl(inner, five), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(outer) });
        assert!(!run(&mut func));
        let inst = def_inst(&func, outer).expect("outer shift");
        match &func.insts[inst].kind {
            InstKind::Shl(lhs, _) => assert_eq!(*lhs, inner),
            other => panic!("expected shift, got {other:?}"),
        }
    }

    #[test]
    fn bitwise_constant_chains_combine() {
        for (build, left, right, combined, result) in [
            (InstKind::And as fn(Value, Value) -> InstKind, 0xF0, 0x3C, 0x30, "and"),
            (InstKind::Or as fn(Value, Value) -> InstKind, 0xFFFF0000, 0xFFFF, -1, "or"),
            (InstKind::Xor as fn(Value, Value) -> InstKind, 0xF0, 0x3C, 0xCC, "xor"),
        ] {
            let (mut func, entry, x) = int_func();
            let first = iconst(&mut func, 32, left);
            let second = iconst(&mut func, 32, right);
            let inner = emit(&mut func, entry, build(x, first), Ty::int(32));
            let outer = emit(&mut func, entry, build(inner, second), Ty::int(32));
            func.blocks[entry].term = Some(Terminator::Return { val: Some(outer) });
            assert!(run(&mut func), "{result}");
            let out = returned(&func);
            if combined == -1 {
                assert_eq!(func.const_value(out), Some(&Const::Int { bits: 32, value: -1 }), "{result}");
                continue;
            }
            let inst = def_inst(&func, out).expect("rebuilt bitwise op");
            match (&func.insts[inst].kind, result) {
                (InstKind::And(lhs, rhs), "and") | (InstKind::Or(lhs, rhs), "or") | (InstKind::Xor(lhs, rhs), "xor") => {
                    assert_eq!(*lhs, x);
                    assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 32, value: combined }), "{result}");
                }
                (other, _) => panic!("expected {result}, got {other:?}"),
            }
        }
    }

    #[test]
    fn and_chain_with_disjoint_masks_folds_to_zero() {
        let (mut func, entry, x) = int_func();
        let high = iconst(&mut func, 32, 0xF0);
        let low = iconst(&mut func, 32, 0x0F);
        let inner = emit(&mut func, entry, InstKind::And(x, high), Ty::int(32));
        let outer = emit(&mut func, entry, InstKind::And(inner, low), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(outer) });
        assert!(run(&mut func));
        assert_eq!(func.const_value(returned(&func)), Some(&Const::Int { bits: 32, value: 0 }));
    }

    #[test]
    fn xor_chain_with_all_ones_becomes_not() {
        let (mut func, entry, x) = int_func();
        let high = iconst(&mut func, 32, 0xFFFFFF00);
        let rest = iconst(&mut func, 32, 0xFF);
        let inner = emit(&mut func, entry, InstKind::Xor(x, high), Ty::int(32));
        let outer = emit(&mut func, entry, InstKind::Xor(inner, rest), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(outer) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt negation");
        match &func.insts[inst].kind {
            InstKind::Not(operand) => assert_eq!(*operand, x),
            other => panic!("expected not, got {other:?}"),
        }
    }

    #[test]
    fn shift_pairs_fold_to_masks() {
        let cases = [
            (InstKind::Shl as fn(Value, Value) -> InstKind, InstKind::LShr as fn(Value, Value) -> InstKind, 2, 0x3FFFFFFF),
            (InstKind::LShr as fn(Value, Value) -> InstKind, InstKind::Shl as fn(Value, Value) -> InstKind, 3, -8),
            (InstKind::AShr as fn(Value, Value) -> InstKind, InstKind::Shl as fn(Value, Value) -> InstKind, 3, -8),
        ];
        for (inner, outer, amount, expected) in cases {
            let (mut func, entry, x) = int_func();
            let by = iconst(&mut func, 32, amount);
            let shifted = emit(&mut func, entry, inner(x, by), Ty::int(32));
            let masked = emit(&mut func, entry, outer(shifted, by), Ty::int(32));
            func.blocks[entry].term = Some(Terminator::Return { val: Some(masked) });
            assert!(run(&mut func));
            let inst = def_inst(&func, returned(&func)).expect("rebuilt mask");
            match &func.insts[inst].kind {
                InstKind::And(lhs, rhs) => {
                    assert_eq!(*lhs, x);
                    assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 32, value: expected }));
                }
                other => panic!("expected and, got {other:?}"),
            }
        }
    }

    #[test]
    fn arithmetic_shift_after_left_shift_is_not_a_mask() {
        let (mut func, entry, x) = int_func();
        let two = iconst(&mut func, 32, 2);
        let shifted = emit(&mut func, entry, InstKind::Shl(x, two), Ty::int(32));
        let out = emit(&mut func, entry, InstKind::AShr(shifted, two), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(out) });
        assert!(!run(&mut func));
    }

    #[test]
    fn unsigned_comparisons_against_zero_fold() {
        for (pred, truth) in [(IcmpPred::Ult, false), (IcmpPred::Uge, true)] {
            let (mut func, entry, x) = int_func();
            let zero = iconst(&mut func, 32, 0);
            let cmp = emit(&mut func, entry, InstKind::Icmp { pred, lhs: x, rhs: zero }, Ty::Bool);
            func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
            assert!(run(&mut func), "{pred:?}");
            assert_eq!(func.const_value(returned(&func)), Some(&Const::Int { bits: 1, value: i128::from(truth) }));
        }

        for (pred, equal) in [(IcmpPred::Ule, true), (IcmpPred::Ugt, false)] {
            let (mut func, entry, x) = int_func();
            let zero = iconst(&mut func, 32, 0);
            let cmp = emit(&mut func, entry, InstKind::Icmp { pred, lhs: x, rhs: zero }, Ty::Bool);
            func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
            assert!(run(&mut func), "{pred:?}");
            let inst = def_inst(&func, returned(&func)).expect("rebuilt comparison");
            match &func.insts[inst].kind {
                InstKind::Icmp { pred: narrowed, lhs, rhs } => {
                    assert_eq!(*narrowed, if equal { IcmpPred::Eq } else { IcmpPred::Ne });
                    assert_eq!(*lhs, x);
                    assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 32, value: 0 }));
                }
                other => panic!("expected comparison, got {other:?}"),
            }
        }
    }

    #[test]
    fn zero_extension_comparisons_narrow() {
        let sig = Signature::new(vec![Ty::int(8), Ty::int(8)], Ty::Bool);
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(8), 0);
        let y = param(&mut func, entry, Ty::int(8), 1);
        let left = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
        let right = emit(&mut func, entry, InstKind::Zext(y), Ty::int(32));
        let cmp = emit(
            &mut func,
            entry,
            InstKind::Icmp {
                pred: IcmpPred::Slt,
                lhs: left,
                rhs: right,
            },
            Ty::Bool,
        );
        func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt comparison");
        match &func.insts[inst].kind {
            InstKind::Icmp { pred, lhs, rhs } => {
                assert_eq!(*pred, IcmpPred::Ult);
                assert_eq!((*lhs, *rhs), (x, y));
            }
            other => panic!("expected comparison, got {other:?}"),
        }
    }

    #[test]
    fn sign_extension_comparisons_keep_their_order() {
        let sig = Signature::new(vec![Ty::int(8), Ty::int(8)], Ty::Bool);
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(8), 0);
        let y = param(&mut func, entry, Ty::int(8), 1);
        let left = emit(&mut func, entry, InstKind::Sext(x), Ty::int(32));
        let right = emit(&mut func, entry, InstKind::Sext(y), Ty::int(32));
        let cmp = emit(
            &mut func,
            entry,
            InstKind::Icmp {
                pred: IcmpPred::Ugt,
                lhs: left,
                rhs: right,
            },
            Ty::Bool,
        );
        func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt comparison");
        match &func.insts[inst].kind {
            InstKind::Icmp { pred, lhs, rhs } => {
                assert_eq!(*pred, IcmpPred::Ugt);
                assert_eq!((*lhs, *rhs), (x, y));
            }
            other => panic!("expected comparison, got {other:?}"),
        }
    }

    #[test]
    fn zero_extension_comparisons_with_constants_narrow() {
        let sig = Signature::new(vec![Ty::int(8)], Ty::Bool);
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(8), 0);
        let wide = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
        let constant = iconst(&mut func, 32, 200);
        let cmp = emit(
            &mut func,
            entry,
            InstKind::Icmp { pred: IcmpPred::Slt, lhs: wide, rhs: constant },
            Ty::Bool,
        );
        func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt comparison");
        match &func.insts[inst].kind {
            InstKind::Icmp { pred, lhs, rhs } => {
                assert_eq!(*pred, IcmpPred::Ult);
                assert_eq!(*lhs, x);
                assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 8, value: -56 }));
            }
            other => panic!("expected comparison, got {other:?}"),
        }

        for (pred, truth) in [(IcmpPred::Slt, true), (IcmpPred::Eq, false)] {
            let sig = Signature::new(vec![Ty::int(8)], Ty::Bool);
            let (mut func, entry) = Function::new("f", sig);
            let x = param(&mut func, entry, Ty::int(8), 0);
            let wide = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
            let constant = iconst(&mut func, 32, 300);
            let cmp = emit(
                &mut func,
                entry,
                InstKind::Icmp { pred, lhs: wide, rhs: constant },
                Ty::Bool,
            );
            func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
            assert!(run(&mut func));
            assert_eq!(
                func.const_value(returned(&func)),
                Some(&Const::Int { bits: 1, value: i128::from(truth) })
            );
        }
    }

    #[test]
    fn sign_extension_comparisons_with_constants_narrow() {
        let sig = Signature::new(vec![Ty::int(8)], Ty::Bool);
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(8), 0);
        let wide = emit(&mut func, entry, InstKind::Sext(x), Ty::int(32));
        let constant = iconst(&mut func, 32, 100);
        let cmp = emit(
            &mut func,
            entry,
            InstKind::Icmp { pred: IcmpPred::Slt, lhs: wide, rhs: constant },
            Ty::Bool,
        );
        func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt comparison");
        match &func.insts[inst].kind {
            InstKind::Icmp { pred, lhs, rhs } => {
                assert_eq!(*pred, IcmpPred::Slt);
                assert_eq!(*lhs, x);
                assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 8, value: 100 }));
            }
            other => panic!("expected comparison, got {other:?}"),
        }

        let sig = Signature::new(vec![Ty::int(8)], Ty::Bool);
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(8), 0);
        let wide = emit(&mut func, entry, InstKind::Sext(x), Ty::int(32));
        let constant = iconst(&mut func, 32, -200);
        let cmp = emit(
            &mut func,
            entry,
            InstKind::Icmp { pred: IcmpPred::Slt, lhs: wide, rhs: constant },
            Ty::Bool,
        );
        func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
        assert!(run(&mut func));
        assert_eq!(func.const_value(returned(&func)), Some(&Const::Int { bits: 1, value: 0 }));
    }

    #[test]
    fn constant_on_the_left_narrows_the_comparison() {
        let sig = Signature::new(vec![Ty::int(8)], Ty::Bool);
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(8), 0);
        let wide = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
        let constant = iconst(&mut func, 32, 200);
        let cmp = emit(
            &mut func,
            entry,
            InstKind::Icmp { pred: IcmpPred::Sgt, lhs: constant, rhs: wide },
            Ty::Bool,
        );
        func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt comparison");
        match &func.insts[inst].kind {
            InstKind::Icmp { pred, lhs, rhs } => {
                assert_eq!(*pred, IcmpPred::Ult);
                assert_eq!(*lhs, x);
                assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 8, value: -56 }));
            }
            other => panic!("expected comparison, got {other:?}"),
        }
    }

    #[test]
    fn zero_extended_boolean_comparisons_fold() {
        for (pred, constant, expect_value) in [
            (IcmpPred::Eq, 1, true),
            (IcmpPred::Ne, 1, false),
            (IcmpPred::Eq, 0, false),
            (IcmpPred::Ne, 0, true),
        ] {
            let sig = Signature::new(vec![Ty::Bool], Ty::Bool);
            let (mut func, entry) = Function::new("f", sig);
            let x = param(&mut func, entry, Ty::Bool, 0);
            let wide = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
            let constant = iconst(&mut func, 32, constant);
            let cmp = emit(
                &mut func,
                entry,
                InstKind::Icmp { pred, lhs: wide, rhs: constant },
                Ty::Bool,
            );
            func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
            assert!(run(&mut func), "{pred:?}");
            if expect_value {
                assert_eq!(returned(&func), x, "{pred:?}");
            } else {
                let inst = def_inst(&func, returned(&func)).expect("negated boolean");
                match &func.insts[inst].kind {
                    InstKind::Icmp { pred: IcmpPred::Eq, lhs, rhs } => {
                        assert_eq!(*lhs, x);
                        assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 1, value: 0 }));
                    }
                    other => panic!("expected negated boolean, got {other:?}"),
                }
            }
        }

        for (pred, truth) in [(IcmpPred::Eq, false), (IcmpPred::Ne, true)] {
            let sig = Signature::new(vec![Ty::Bool], Ty::Bool);
            let (mut func, entry) = Function::new("f", sig);
            let x = param(&mut func, entry, Ty::Bool, 0);
            let wide = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
            let constant = iconst(&mut func, 32, 2);
            let cmp = emit(
                &mut func,
                entry,
                InstKind::Icmp { pred, lhs: wide, rhs: constant },
                Ty::Bool,
            );
            func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
            assert!(run(&mut func));
            assert_eq!(
                func.const_value(returned(&func)),
                Some(&Const::Int { bits: 1, value: i128::from(truth) })
            );
        }
    }

    #[test]
    fn bitwise_operations_on_extended_booleans_use_boolean_ops() {
        for op in [Bitwise::And, Bitwise::Or, Bitwise::Xor] {
            let sig = Signature::new(vec![Ty::Bool, Ty::Bool], Ty::int(32));
            let (mut func, entry) = Function::new("f", sig);
            let a = param(&mut func, entry, Ty::Bool, 0);
            let b = param(&mut func, entry, Ty::Bool, 1);
            let a_wide = emit(&mut func, entry, InstKind::Zext(a), Ty::int(32));
            let b_wide = emit(&mut func, entry, InstKind::Zext(b), Ty::int(32));
            let kind = match op {
                Bitwise::And => InstKind::And(a_wide, b_wide),
                Bitwise::Or => InstKind::Or(a_wide, b_wide),
                Bitwise::Xor => InstKind::Xor(a_wide, b_wide),
            };
            let out = emit(&mut func, entry, kind, Ty::int(32));
            func.blocks[entry].term = Some(Terminator::Return { val: Some(out) });
            assert!(run(&mut func));
            let outer = def_inst(&func, returned(&func)).expect("extension");
            let inner = match &func.insts[outer].kind {
                InstKind::Zext(inner) => *inner,
                other => panic!("expected boolean extension, got {other:?}"),
            };
            let inner = def_inst(&func, inner).expect("boolean operation");
            match (&func.insts[inner].kind, op) {
                (InstKind::And(x, y), Bitwise::And)
                | (InstKind::Or(x, y), Bitwise::Or)
                | (InstKind::Xor(x, y), Bitwise::Xor) => {
                    assert_eq!(*x, a);
                    assert_eq!(*y, b);
                }
                (other, _) => panic!("expected boolean operation, got {other:?}"),
            }
        }
    }

    #[test]
    fn select_of_boolean_constants_becomes_an_extension() {
        for (then_value, else_value, signed, invert) in [
            (1, 0, false, false),
            (0, 1, false, true),
            (-1, 0, true, false),
            (0, -1, true, true),
        ] {
            let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
            let (mut func, entry) = Function::new("f", sig);
            let cond = param(&mut func, entry, Ty::Bool, 0);
            let then_val = iconst(&mut func, 32, then_value);
            let else_val = iconst(&mut func, 32, else_value);
            let sel = emit(
                &mut func,
                entry,
                InstKind::Select { cond, then_val, else_val },
                Ty::int(32),
            );
            func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
            assert!(run(&mut func), "select {then_value}/{else_value}");
            let outer = def_inst(&func, returned(&func)).expect("extension");
            let inner = match &func.insts[outer].kind {
                InstKind::Zext(inner) | InstKind::Sext(inner) => *inner,
                other => panic!("expected extension, got {other:?}"),
            };
            assert_eq!(matches!(func.insts[outer].kind, InstKind::Sext(_)), signed);
            if invert {
                let inner = def_inst(&func, inner).expect("negated condition");
                match &func.insts[inner].kind {
                    InstKind::Icmp { pred: IcmpPred::Eq, lhs, rhs } => {
                        assert_eq!(*lhs, cond);
                        assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 1, value: 0 }));
                    }
                    other => panic!("expected negated condition, got {other:?}"),
                }
            } else {
                assert_eq!(inner, cond);
            }
        }
    }

    #[test]
    fn select_of_boolean_constants_stays_boolean() {
        let sig = Signature::new(vec![Ty::Bool], Ty::Bool);
        let (mut func, entry) = Function::new("f", sig);
        let cond = param(&mut func, entry, Ty::Bool, 0);
        let then_val = func.push_const(Const::Int { bits: 1, value: 1 });
        let else_val = func.push_const(Const::Int { bits: 1, value: 0 });
        let sel = emit(
            &mut func,
            entry,
            InstKind::Select { cond, then_val, else_val },
            Ty::Bool,
        );
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
        assert!(run(&mut func));
        assert_eq!(returned(&func), cond);

        let sig = Signature::new(vec![Ty::Bool], Ty::Bool);
        let (mut func, entry) = Function::new("f", sig);
        let cond = param(&mut func, entry, Ty::Bool, 0);
        let then_val = func.push_const(Const::Int { bits: 1, value: 0 });
        let else_val = func.push_const(Const::Int { bits: 1, value: 1 });
        let sel = emit(
            &mut func,
            entry,
            InstKind::Select { cond, then_val, else_val },
            Ty::Bool,
        );
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("negated boolean");
        match &func.insts[inst].kind {
            InstKind::Icmp { pred: IcmpPred::Eq, lhs, rhs } => {
                assert_eq!(*lhs, cond);
                assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 1, value: 0 }));
            }
            other => panic!("expected negated boolean, got {other:?}"),
        }
    }

    #[test]
    fn xor_with_all_ones_becomes_not() {
        let (mut func, entry, x) = int_func();
        let ones = iconst(&mut func, 32, -1);
        let out = emit(&mut func, entry, InstKind::Xor(x, ones), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(out) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt negation");
        match &func.insts[inst].kind {
            InstKind::Not(operand) => assert_eq!(*operand, x),
            other => panic!("expected not, got {other:?}"),
        }
    }

    #[test]
    fn multiply_by_a_power_of_two_shifts() {
        for constant_on_the_left in [false, true] {
            let (mut func, entry, x) = int_func();
            let eight = iconst(&mut func, 32, 8);
            let product = if constant_on_the_left {
                emit(&mut func, entry, InstKind::Mul(eight, x), Ty::int(32))
            } else {
                emit(&mut func, entry, InstKind::Mul(x, eight), Ty::int(32))
            };
            func.blocks[entry].term = Some(Terminator::Return { val: Some(product) });
            assert!(run(&mut func), "constant on left: {constant_on_the_left}");
            let inst = def_inst(&func, returned(&func)).expect("rebuilt shift");
            match &func.insts[inst].kind {
                InstKind::Shl(lhs, amount) => {
                    assert_eq!(*lhs, x);
                    assert_eq!(func.const_value(*amount), Some(&Const::Int { bits: 32, value: 3 }));
                }
                other => panic!("expected shift, got {other:?}"),
            }
        }
    }

    #[test]
    fn multiply_by_minus_one_negates() {
        let (mut func, entry, x) = int_func();
        let minus_one = iconst(&mut func, 32, -1);
        let product = emit(&mut func, entry, InstKind::Mul(x, minus_one), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(product) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt negation");
        match &func.insts[inst].kind {
            InstKind::Neg(operand) => assert_eq!(*operand, x),
            other => panic!("expected negation, got {other:?}"),
        }
    }

    #[test]
    fn zero_minus_x_negates() {
        let (mut func, entry, x) = int_func();
        let zero = iconst(&mut func, 32, 0);
        let difference = emit(&mut func, entry, InstKind::Sub(zero, x), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(difference) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt negation");
        match &func.insts[inst].kind {
            InstKind::Neg(operand) => assert_eq!(*operand, x),
            other => panic!("expected negation, got {other:?}"),
        }
    }

    #[test]
    fn unsigned_division_by_a_power_of_two_shifts() {
        let (mut func, entry, x) = int_func();
        let eight = iconst(&mut func, 32, 8);
        let quotient = emit(&mut func, entry, InstKind::Udiv(x, eight), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(quotient) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt shift");
        match &func.insts[inst].kind {
            InstKind::LShr(lhs, amount) => {
                assert_eq!(*lhs, x);
                assert_eq!(func.const_value(*amount), Some(&Const::Int { bits: 32, value: 3 }));
            }
            other => panic!("expected logical shift, got {other:?}"),
        }
    }

    #[test]
    fn unsigned_remainder_by_a_power_of_two_masks() {
        let (mut func, entry, x) = int_func();
        let eight = iconst(&mut func, 32, 8);
        let remainder = emit(&mut func, entry, InstKind::Urem(x, eight), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(remainder) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt mask");
        match &func.insts[inst].kind {
            InstKind::And(lhs, amount) => {
                assert_eq!(*lhs, x);
                assert_eq!(func.const_value(*amount), Some(&Const::Int { bits: 32, value: 7 }));
            }
            other => panic!("expected mask, got {other:?}"),
        }
    }

    #[test]
    fn pointer_integer_round_trips_cancel() {
        let ptr_ty = Ty::ptr(Ty::int(8));
        let sig = Signature::new(vec![ptr_ty.clone()], ptr_ty.clone());
        let (mut func, entry) = Function::new("f", sig);
        let p = param(&mut func, entry, ptr_ty.clone(), 0);
        let int = emit(&mut func, entry, InstKind::PtrToInt(p), Ty::int(64));
        let back = emit(&mut func, entry, InstKind::IntToPtr(int), ptr_ty);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(back) });
        assert!(run(&mut func));
        assert_eq!(returned(&func), p);

        let sig = Signature::new(vec![Ty::int(64)], Ty::int(64));
        let (mut func, entry) = Function::new("g", sig);
        let int = param(&mut func, entry, Ty::int(64), 0);
        let ptr_ty = Ty::ptr(Ty::int(8));
        let p = emit(&mut func, entry, InstKind::IntToPtr(int), ptr_ty);
        let back = emit(&mut func, entry, InstKind::PtrToInt(p), Ty::int(64));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(back) });
        assert!(run(&mut func));
        assert_eq!(returned(&func), int);
    }

    #[test]
    fn negated_comparison_inverts_the_predicate() {
        let sig = Signature::new(vec![Ty::int(32), Ty::int(32)], Ty::Bool);
        let (mut func, entry) = Function::new("f", sig.clone());
        let a = param(&mut func, entry, Ty::int(32), 0);
        let b = param(&mut func, entry, Ty::int(32), 1);
        let less = emit(&mut func, entry, InstKind::Icmp { pred: IcmpPred::Slt, lhs: a, rhs: b }, Ty::Bool);
        let zero = iconst(&mut func, 1, 0);
        let not_less = emit(
            &mut func,
            entry,
            InstKind::Icmp {
                pred: IcmpPred::Eq,
                lhs: less,
                rhs: zero,
            },
            Ty::Bool,
        );
        func.blocks[entry].term = Some(Terminator::Return { val: Some(not_less) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt comparison");
        match &func.insts[inst].kind {
            InstKind::Icmp { pred, lhs, rhs } => {
                assert_eq!(*pred, IcmpPred::Sge);
                assert_eq!((*lhs, *rhs), (a, b));
            }
            other => panic!("expected comparison, got {other:?}"),
        }

        let (mut func, entry) = Function::new("g", sig);
        let a = param(&mut func, entry, Ty::int(32), 0);
        let b = param(&mut func, entry, Ty::int(32), 1);
        let less = emit(&mut func, entry, InstKind::Icmp { pred: IcmpPred::Slt, lhs: a, rhs: b }, Ty::Bool);
        let zero = iconst(&mut func, 1, 0);
        let truthy = emit(
            &mut func,
            entry,
            InstKind::Icmp {
                pred: IcmpPred::Ne,
                lhs: less,
                rhs: zero,
            },
            Ty::Bool,
        );
        func.blocks[entry].term = Some(Terminator::Return { val: Some(truthy) });
        assert!(run(&mut func));
        assert_eq!(returned(&func), less);
    }

    #[test]
    fn leaves_unknown_operands_alone() {
        let sig = Signature::new(vec![Ty::int(32), Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let a = param(&mut func, entry, Ty::int(32), 0);
        let b = param(&mut func, entry, Ty::int(32), 1);
        let sum = emit(&mut func, entry, InstKind::Add(a, b), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        assert!(!run(&mut func));
    }
}
