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

use crate::ir::{Block, Const, Def, Function, IcmpPred, Inst, InstData, InstKind, Value, ValueData};
use crate::pass::FunctionPass;
use crate::ty::Ty;

use super::{mask, wrap};

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
        for idx in 0..func.blocks.len() {
            let block = Block::from_usize(idx);
            let old = std::mem::take(&mut func.blocks[block].insts);
            let mut out = Vec::with_capacity(old.len());
            for inst in old {
                let Some(result) = func.insts[inst].result else {
                    out.push(inst);
                    continue;
                };
                if use_counts.get(&result).copied().unwrap_or(0) == 0 {
                    out.push(inst);
                    continue;
                }
                let Some(plan) = plan(func, inst) else {
                    out.push(inst);
                    continue;
                };
                changed = true;
                let replacement = apply(func, block, inst, result, plan, &mut out);
                func.replace_uses(result, replacement);
            }
            func.blocks[block].insts = out;
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
    /// `!v` for a `Bool`-typed `v`, as `icmp.Eq v, 0`.
    BoolNot(Value),
}

/// Choose a rewrite for `inst`, whose result must have uses.
fn plan(func: &Function, inst: Inst) -> Option<Plan> {
    let result = func.insts[inst].result?;
    let result_ty = &func.values[result].ty;
    match &func.insts[inst].kind {
        InstKind::Add(a, b) => {
            if is_int_zero(func, *b) {
                Some(Plan::Value(*a))
            } else if is_int_zero(func, *a) {
                Some(Plan::Value(*b))
            } else {
                None
            }
        }
        InstKind::Sub(a, b) => {
            if is_int_zero(func, *b) {
                Some(Plan::Value(*a))
            } else if a == b {
                Some(Plan::Const(zero_const(result_ty)?))
            } else {
                None
            }
        }
        InstKind::Mul(a, b) => {
            if is_int_zero(func, *a) || is_int_zero(func, *b) {
                Some(Plan::Const(zero_const(result_ty)?))
            } else if is_int_one(func, *a) {
                Some(Plan::Value(*b))
            } else if is_int_one(func, *b) {
                Some(Plan::Value(*a))
            } else {
                None
            }
        }
        // A divisor of one cannot trap, so the quotient is the dividend
        // and the remainder is zero for every input (including the
        // minimum signed value, whose `-1` cases are the dangerous
        // ones).
        InstKind::Udiv(a, b) | InstKind::Sdiv(a, b) if is_int_one(func, *b) => Some(Plan::Value(*a)),
        InstKind::Urem(_, b) | InstKind::Srem(_, b) if is_int_one(func, *b) => {
            Some(Plan::Const(zero_const(result_ty)?))
        }
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
                None
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
                None
            }
        }
        InstKind::Xor(a, b) => {
            if a == b {
                Some(Plan::Const(zero_const(result_ty)?))
            } else if is_int_zero(func, *b) {
                Some(Plan::Value(*a))
            } else if is_int_zero(func, *a) {
                Some(Plan::Value(*b))
            } else {
                None
            }
        }
        InstKind::Shl(a, b) | InstKind::LShr(a, b) | InstKind::AShr(a, b) => is_int_zero(func, *b).then_some(Plan::Value(*a)),
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
        InstKind::Icmp { pred, lhs, rhs } => {
            if lhs == rhs && !is_float(func, *lhs) {
                let truth = matches!(pred, IcmpPred::Eq | IcmpPred::Sle | IcmpPred::Ule | IcmpPred::Sge | IcmpPred::Uge);
                let bits = int_ty_bits(result_ty)?;
                return Some(Plan::Const(Const::Int { bits, value: i128::from(truth) }));
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
                None => None,
            }
        }
        InstKind::GetElementPtr { base, indices } if indices.is_empty() && func.values[*base].ty == *result_ty => Some(Plan::Value(*base)),
        _ => None,
    }
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
        other => other,
    };
    match plan {
        Plan::Value(v) => v,
        Plan::Const(c) => {
            let ty = func.values[result].ty.clone();
            func.values.push(ValueData { ty, def: Def::Const(c) })
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
        Plan::BoolNot(_) => unreachable!("normalized above"),
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
