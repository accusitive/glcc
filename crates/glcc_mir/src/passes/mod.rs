pub mod constmerge;
pub mod cse;
pub mod global;
pub mod instcombine;
pub mod licm;
pub mod loadstore;
pub mod mem;
pub mod sccp;
pub mod sroa;
pub mod unroll;

pub use constmerge::ConstMerge;
pub use cse::Cse;
pub use global::{DeadModuleItems, GlobalConstProp, MergeReadOnlyGlobals};
pub use instcombine::InstCombine;
pub use licm::Licm;
pub use loadstore::LoadStoreForward;
pub use mem::LowerLibcMemory;
pub use sccp::Sccp;
pub use sroa::Sroa;
pub use unroll::LoopUnroll;

use crate::ir::{Block, Const, Def, FcmpPred, Function, IcmpPred, Inst, InstData, InstKind, MemFlags, Module, Terminator, Value, ValueData, remap_inst};
use crate::pass::{FunctionPass, ModulePass};
use crate::ty::{FloatKind, GepIndex, Ty};

fn int_const(func: &Function, v: Value) -> Option<(u32, i128)> {
    int_const_depth(func, v, 0)
}

/// Resolve the integer value of constants and constant-only integer
/// expressions. Address lowering emits `sext`/`mul`/`add` chains even
/// for literal indices, so analyses that ask for a byte offset need to
/// see through them.
fn int_const_depth(func: &Function, v: Value, depth: u32) -> Option<(u32, i128)> {
    if depth > 64 {
        return None;
    }
    if let Some(Const::Int { bits, value }) = func.const_value(v) {
        return Some((*bits, *value));
    }
    let Def::Inst(inst) = func.values[v].def else {
        return None;
    };
    let bits = int_ty_bits(&func.values[v].ty)?;
    let recurse = |value| int_const_depth(func, value, depth + 1);
    let (bits, value) = match &func.insts[inst].kind {
        InstKind::Zext(x) => {
            let (source_bits, value) = recurse(*x)?;
            (bits, wrap(bits, ((value as u128) & mask(source_bits)) as i128))
        }
        InstKind::Sext(x) => {
            let (_, value) = recurse(*x)?;
            (bits, wrap(bits, value))
        }
        InstKind::Trunc(x) => {
            let (_, value) = recurse(*x)?;
            (bits, wrap(bits, value))
        }
        InstKind::Bitcast(x) => {
            let (source_bits, value) = recurse(*x)?;
            if source_bits != bits {
                return None;
            }
            (bits, wrap(bits, value))
        }
        InstKind::Add(a, b) => {
            let (_, x) = recurse(*a)?;
            let (_, y) = recurse(*b)?;
            (bits, wrap(bits, x.wrapping_add(y)))
        }
        InstKind::Sub(a, b) => {
            let (_, x) = recurse(*a)?;
            let (_, y) = recurse(*b)?;
            (bits, wrap(bits, x.wrapping_sub(y)))
        }
        InstKind::Mul(a, b) => {
            let (_, x) = recurse(*a)?;
            let (_, y) = recurse(*b)?;
            (bits, wrap(bits, x.wrapping_mul(y)))
        }
        InstKind::And(a, b) => {
            let (_, x) = recurse(*a)?;
            let (_, y) = recurse(*b)?;
            (bits, wrap(bits, x & y))
        }
        InstKind::Or(a, b) => {
            let (_, x) = recurse(*a)?;
            let (_, y) = recurse(*b)?;
            (bits, wrap(bits, x | y))
        }
        InstKind::Xor(a, b) => {
            let (_, x) = recurse(*a)?;
            let (_, y) = recurse(*b)?;
            (bits, wrap(bits, x ^ y))
        }
        InstKind::Neg(x) => {
            let (_, value) = recurse(*x)?;
            (bits, wrap(bits, value.wrapping_neg()))
        }
        InstKind::Not(x) => {
            let (_, value) = recurse(*x)?;
            (bits, wrap(bits, !value))
        }
        InstKind::Shl(a, b) => {
            let (_, x) = recurse(*a)?;
            let (_, shift) = recurse(*b)?;
            let Const::Int { bits, value } = fold_shift_bits(bits, x as u128, shift, |x, s| x << s)? else {
                return None;
            };
            return Some((bits, value));
        }
        InstKind::LShr(a, b) => {
            let (_, x) = recurse(*a)?;
            let (_, shift) = recurse(*b)?;
            let Const::Int { bits, value } = fold_shift_bits(bits, x as u128, shift, |x, s| x >> s)? else {
                return None;
            };
            return Some((bits, value));
        }
        InstKind::AShr(a, b) => {
            let (_, x) = recurse(*a)?;
            let (_, shift) = recurse(*b)?;
            let Const::Int { bits, value } = fold_ashr_bits(bits, x, shift)? else {
                return None;
            };
            return Some((bits, value));
        }
        _ => return None,
    };
    Some((bits, value))
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
/// delete the trap. `f32`/`f64` arithmetic folds when both operands are
/// constants; the wider carriers stay for codegen.
#[derive(Default)]
pub struct ConstFold;

impl FunctionPass for ConstFold {
    fn name(&self) -> &'static str {
        "const-fold"
    }

    fn run(&mut self, func: &mut Function) -> bool {
        let mut changed = false;
        // Folded instructions leave the block even when they are impure
        // (`udiv` and friends): the fold proved they cannot trap, and
        // leaving them behind makes the pass fire forever, because DCE
        // does not remove impure instructions.
        let mut folded_away: std::collections::HashSet<crate::ir::Inst> = std::collections::HashSet::new();
        // Collect replacements and apply them in one sweep; rewriting
        // per fold is O(function size).
        let mut redirect: std::collections::HashMap<Value, Value> = std::collections::HashMap::new();
        let insts: Vec<_> = func.blocks.iter().flat_map(|b| b.insts.clone()).collect();
        for inst in insts {
            let mut kind = func.insts[inst].kind.clone();
            if !redirect.is_empty() {
                let mut hit = false;
                kind.for_each_operand(&mut |v| hit |= redirect.contains_key(&v));
                if hit {
                    kind = crate::ir::remap_inst(&kind, |v| redirect.get(&v).copied().unwrap_or(v));
                }
            }
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
                InstKind::FAdd(a, b) => fold_float2(func, *a, *b, FloatOp::Add),
                InstKind::FSub(a, b) => fold_float2(func, *a, *b, FloatOp::Sub),
                InstKind::FMul(a, b) => fold_float2(func, *a, *b, FloatOp::Mul),
                InstKind::FDiv(a, b) => fold_float2(func, *a, *b, FloatOp::Div),
                InstKind::Fcmp { pred, lhs, rhs } => fold_fcmp(func, *pred, *lhs, *rhs),
                InstKind::SItoF(a) => fold_int_to_float(func, *a, result_ty.as_ref(), false),
                InstKind::UItoF(a) => fold_int_to_float(func, *a, result_ty.as_ref(), true),
                InstKind::SFtoI(a) => fold_float_to_int(func, *a, result_ty.as_ref(), false),
                InstKind::UFtoI(a) => fold_float_to_int(func, *a, result_ty.as_ref(), true),
                InstKind::Fptrunc(a) => fold_float_width(func, *a, result_ty.as_ref(), false),
                InstKind::Fpext(a) => fold_float_width(func, *a, result_ty.as_ref(), true),
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
            if let Some(c) = folded
                && let Some(result) = func.insts[inst].result {
                    // The replacement stands in for `result`, so it must
                    // carry `result`'s type: `c.ty()` loses it (`Undef` is
                    // `Void`, `Null` is `ptr(Void)`, aggregates have none).
                    let ty = func.values[result].ty.clone();
                    let origin = crate::ir::ValueOrigin {
                        pass: "const-fold",
                        text: crate::display::inst_str(func, &kind),
                        loc: func.insts[inst].loc,
                    };
                    let replacement = func.values.push(crate::ir::ValueData { ty, def: Def::Const(c) });
                    func.origins.insert(replacement, origin);
                    redirect.insert(result, replacement);
                    folded_away.insert(inst);
                    changed = true;
                }
        }
        if !folded_away.is_empty() {
            for block in func.blocks.iter_mut() {
                block.insts.retain(|i| !folded_away.contains(i));
            }
        }
        for idx in 0..func.blocks.len() {
            let block_id = crate::ir::Block::from_usize(idx);
            let term = func.blocks[block_id].term.clone();
            let term = term.map(|t| {
                let mut hit = false;
                t.for_each_operand(&mut |v| hit |= redirect.contains_key(&v));
                if hit {
                    crate::ir::remap_term(&t, |v| redirect.get(&v).copied().unwrap_or(v))
                } else {
                    t
                }
            });
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
        if !redirect.is_empty() {
            func.remap_operands(&redirect);
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

#[derive(Clone, Copy)]
enum FloatOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Fold `f32`/`f64` arithmetic over two constants. Host arithmetic is
/// IEEE-754 round-to-nearest for both widths, matching the targets.
fn fold_float2(func: &Function, a: Value, b: Value, op: FloatOp) -> Option<Const> {
    let (kind, l, r) = match (func.const_value(a)?, func.const_value(b)?) {
        (
            Const::Float {
                kind: kind @ FloatKind::F32,
                bits: l,
            },
            Const::Float { kind: FloatKind::F32, bits: r },
        ) => (*kind, f32::from_bits(*l as u32) as f64, f32::from_bits(*r as u32) as f64),
        (
            Const::Float {
                kind: kind @ FloatKind::F64,
                bits: l,
            },
            Const::Float { kind: FloatKind::F64, bits: r },
        ) => (*kind, f64::from_bits(*l as u64), f64::from_bits(*r as u64)),
        _ => return None,
    };
    let bits = match kind {
        FloatKind::F32 => {
            let (l, r) = (l as f32, r as f32);
            let value = match op {
                FloatOp::Add => l + r,
                FloatOp::Sub => l - r,
                FloatOp::Mul => l * r,
                FloatOp::Div => l / r,
            };
            value.to_bits() as u128
        }
        FloatKind::F64 => {
            let value = match op {
                FloatOp::Add => l + r,
                FloatOp::Sub => l - r,
                FloatOp::Mul => l * r,
                FloatOp::Div => l / r,
            };
            value.to_bits() as u128
        }
        _ => return None,
    };
    Some(Const::Float { kind, bits })
}

/// Convert an integer constant to `f32`/`f64`. Wider float carriers stay
/// unfolded because they need target-specific helpers.
fn fold_int_to_float(func: &Function, a: Value, result_ty: Option<&Ty>, unsigned: bool) -> Option<Const> {
    let (bits, value) = int_const(func, a)?;
    let kind = match result_ty? {
        Ty::Float(kind @ (FloatKind::F32 | FloatKind::F64)) => *kind,
        _ => return None,
    };
    let magnitude = (value as u128) & mask(bits);
    let bits = match (kind, unsigned) {
        (FloatKind::F32, false) => (value as f32).to_bits() as u128,
        (FloatKind::F64, false) => (value as f64).to_bits() as u128,
        (FloatKind::F32, true) => (magnitude as f32).to_bits() as u128,
        (FloatKind::F64, true) => (magnitude as f64).to_bits() as u128,
        _ => return None,
    };
    Some(Const::Float { kind, bits })
}

/// Convert an `f32`/`f64` constant to an integer. Values outside the
/// target range and non-finite inputs stay unfolded: C leaves those
/// conversions undefined, so the backend keeps its current behavior.
fn fold_float_to_int(func: &Function, a: Value, result_ty: Option<&Ty>, unsigned: bool) -> Option<Const> {
    let value = match func.const_value(a)? {
        Const::Float { kind: FloatKind::F32, bits } => f32::from_bits(*bits as u32) as f64,
        Const::Float { kind: FloatKind::F64, bits } => f64::from_bits(*bits as u64),
        _ => return None,
    };
    let bits = match result_ty? {
        Ty::Int { bits } => *bits,
        _ => return None,
    };
    if bits == 0 || bits > 128 || !value.is_finite() {
        return None;
    }
    let truncated = value.trunc();
    let value = if unsigned {
        if truncated < 0.0 {
            return None;
        }
        if bits == 128 {
            if truncated >= 2f64.powi(128) {
                return None;
            }
        } else if truncated >= 2f64.powi(bits as i32) {
            return None;
        }
        (truncated as u128) as i128
    } else {
        let limit = if bits == 128 { 2f64.powi(127) } else { 2f64.powi(bits as i32 - 1) };
        if truncated < -limit || truncated >= limit {
            return None;
        }
        truncated as i128
    };
    Some(Const::Int { bits, value: wrap(bits, value) })
}

/// Widen or narrow a floating-point constant. The opposite direction is
/// a separate instruction, so only one conversion applies here.
fn fold_float_width(func: &Function, a: Value, result_ty: Option<&Ty>, extend: bool) -> Option<Const> {
    let value = match (func.const_value(a)?, result_ty?) {
        (Const::Float { kind: FloatKind::F32, bits }, Ty::Float(FloatKind::F64)) if extend => (f32::from_bits(*bits as u32) as f64).to_bits(),
        (Const::Float { kind: FloatKind::F64, bits }, Ty::Float(FloatKind::F32)) if !extend => (f64::from_bits(*bits as u64) as f32).to_bits() as u64,
        _ => return None,
    };
    Some(Const::Float {
        kind: if extend { FloatKind::F64 } else { FloatKind::F32 },
        bits: value as u128,
    })
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

/// Collapse `store dst, load src` into one `Memcpy`.
///
/// Aggregates have no Clif value: lowering `load` copies into a fresh
/// stack slot, and the following `store` copies it out again. When the
/// load's only use is the immediately following store, copying straight
/// from the source to the destination is equivalent for the plain,
/// non-volatile accesses lowered this way, and avoids the temporary.
#[derive(Default)]
pub struct ForwardAggregateCopy;

impl FunctionPass for ForwardAggregateCopy {
    fn name(&self) -> &'static str {
        "forward-aggregate-copy"
    }

    fn run(&mut self, func: &mut Function) -> bool {
        let counts = func.count_resident_uses();
        let mut changed = false;
        for block_idx in 0..func.blocks.len() {
            let block = Block::from_usize(block_idx);
            let insts = func.blocks[block].insts.clone();
            let mut removed = Vec::new();
            for pair in insts.windows(2) {
                let (load_inst, store_inst) = (pair[0], pair[1]);
                let InstKind::Load { addr: src, flags: load_flags } = &func.insts[load_inst].kind else {
                    continue;
                };
                let InstKind::Store {
                    addr: dst,
                    val,
                    flags: store_flags,
                } = &func.insts[store_inst].kind
                else {
                    continue;
                };
                let Some(result) = func.insts[load_inst].result else {
                    continue;
                };
                if *val != result || counts.get(result.index()).copied().unwrap_or(0) != 1 {
                    continue;
                }
                if load_flags.volatile || store_flags.volatile {
                    continue;
                }
                let ty = &func.values[result].ty;
                if !ty.is_aggregate() {
                    continue;
                }
                let len = ty.size_bytes();
                func.insts[load_inst].kind = InstKind::Memcpy { dst: *dst, src: *src, len };
                func.insts[load_inst].result = None;
                removed.push(store_inst);
                changed = true;
            }
            if !removed.is_empty() {
                func.blocks[block].insts.retain(|inst| !removed.contains(inst));
            }
        }
        let non_escaping = non_escaping_stack_slots(func);
        changed |= expand_local_zero_memsets(func, &non_escaping);
        changed |= expand_local_aggregate_copies(func, &non_escaping);
        changed |= expand_local_scalar_copies(func, &non_escaping);
        changed |= expand_local_aggregate_stores(func, &non_escaping);
        changed |= expand_returned_aggregate_loads(func, &non_escaping);
        changed |= expand_aggregate_load_extracts(func, &non_escaping);
        changed
    }
}

/// Replace a zero-fill of a non-escaping local aggregate with scalar zero
/// stores, so `Sroa` can split the slot instead of refusing the `Memset`.
fn expand_local_zero_memsets(func: &mut Function, non_escaping: &std::collections::HashSet<Value>) -> bool {
    let mut changed = false;
    for block_idx in 0..func.blocks.len() {
        let block = Block::from_usize(block_idx);
        let insts = func.blocks[block].insts.clone();
        let mut expanded = false;
        let mut rebuilt = Vec::with_capacity(insts.len());
        for inst in insts {
            let memset = match &func.insts[inst].kind {
                InstKind::Memset { dst, byte: 0, len } => Some((*dst, *len)),
                _ => None,
            };
            let Some((dst, len)) = memset else {
                rebuilt.push(inst);
                continue;
            };
            let Ty::Ptr { pointee, bits } = func.values[dst].ty.clone() else {
                rebuilt.push(inst);
                continue;
            };
            if !pointee.is_aggregate() || pointee.size_bytes() != len {
                rebuilt.push(inst);
                continue;
            }
            if !matches!(
                address_info(func, dst).map(|info| info.object),
                Some(MemoryObject::Stack(root)) if non_escaping.contains(&root)
            ) {
                rebuilt.push(inst);
                continue;
            }
            let Some(leaves) = aggregate_leaves(&pointee) else {
                rebuilt.push(inst);
                continue;
            };
            for (path, ty) in leaves {
                let (addr, addr_inst) = make_value(
                    func,
                    block,
                    InstKind::GetElementPtr {
                        base: dst,
                        indices: path.into_iter().collect(),
                    },
                    Ty::ptr_bits(ty.clone(), bits),
                );
                rebuilt.push(addr_inst);
                let zero = func.values.push(ValueData {
                    ty: ty.clone(),
                    def: Def::Const(zero_const_for(&ty)),
                });
                rebuilt.push(make_unit(
                    func,
                    block,
                    InstKind::Store {
                        addr,
                        val: zero,
                        flags: MemFlags::plain(),
                    },
                ));
            }
            expanded = true;
            changed = true;
        }
        if expanded {
            func.blocks[block].insts = rebuilt;
        }
    }
    changed
}

fn zero_const_for(ty: &Ty) -> Const {
    match ty {
        Ty::Bool => Const::Int { bits: 1, value: 0 },
        Ty::Int { bits } => Const::Int { bits: *bits, value: 0 },
        Ty::Float(kind) => Const::Float { kind: *kind, bits: 0 },
        Ty::Ptr { .. } => Const::Null,
        _ => unreachable!("aggregate leaves are scalar"),
    }
}

/// Follow pointer-to-pointer bitcasts to the address they alias.
fn strip_pointer_bitcasts(func: &Function, mut value: Value) -> Value {
    while let Def::Inst(inst) = func.values[value].def {
        let InstKind::Bitcast(inner) = func.insts[inst].kind else {
            break;
        };
        if !matches!(func.values[value].ty, Ty::Ptr { .. }) || !matches!(func.values[inner].ty, Ty::Ptr { .. }) {
            break;
        }
        value = inner;
    }
    value
}

/// Turn `memcpy(&x, p, sizeof x)` for a scalar local into a load and
/// store. The copy is what keeps `x` address-taken: expanding it lets
/// `Sroa`/`Mem2Reg` promote the slot, and the load can then fold into
/// its single use.
fn expand_local_scalar_copies(func: &mut Function, non_escaping: &std::collections::HashSet<Value>) -> bool {
    let mut changed = false;
    for block_idx in 0..func.blocks.len() {
        let block = Block::from_usize(block_idx);
        let insts = func.blocks[block].insts.clone();
        let mut expanded = false;
        let mut rebuilt = Vec::with_capacity(insts.len());
        for inst in insts {
            let copy = match &func.insts[inst].kind {
                InstKind::Memcpy { dst, src, len } => Some((*dst, *src, *len)),
                _ => None,
            };
            let Some((dst, src, len)) = copy else {
                rebuilt.push(inst);
                continue;
            };
            let dst = strip_pointer_bitcasts(func, dst);
            let Ty::Ptr { pointee, .. } = &func.values[dst].ty else {
                rebuilt.push(inst);
                continue;
            };
            if !pointee.is_scalar() || pointee.size_bytes() != len {
                rebuilt.push(inst);
                continue;
            }
            if !matches!(
                address_info(func, dst).map(|info| info.object),
                Some(MemoryObject::Stack(root)) if non_escaping.contains(&root)
            ) {
                rebuilt.push(inst);
                continue;
            }
            let ty = (**pointee).clone();
            let (val, load) = make_value(
                func,
                block,
                InstKind::Load {
                    addr: src,
                    flags: MemFlags::plain(),
                },
                ty,
            );
            rebuilt.push(load);
            rebuilt.push(make_unit(
                func,
                block,
                InstKind::Store {
                    addr: dst,
                    val,
                    flags: MemFlags::plain(),
                },
            ));
            expanded = true;
            changed = true;
        }
        if expanded {
            func.blocks[block].insts = rebuilt;
        }
    }
    changed
}

/// Rewrite a whole-aggregate copy between two disjoint local slots into
/// scalar field copies, exposing the copy to `Sroa` on the next pipeline
/// turn.
fn expand_local_aggregate_copies(func: &mut Function, non_escaping: &std::collections::HashSet<Value>) -> bool {
    let mut changed = false;
    for block_idx in 0..func.blocks.len() {
        let block = Block::from_usize(block_idx);
        let insts = func.blocks[block].insts.clone();
        let mut expanded = false;
        let mut rebuilt = Vec::with_capacity(insts.len());
        for inst in insts {
            let copy = match &func.insts[inst].kind {
                InstKind::Memcpy { dst, src, len } => Some((*dst, *src, *len)),
                _ => None,
            };
            let Some((dst, src, len)) = copy else {
                rebuilt.push(inst);
                continue;
            };
            let Some(leaves) = local_aggregate_copy_leaves(func, dst, src, len, non_escaping) else {
                rebuilt.push(inst);
                continue;
            };
            let Ty::Ptr { bits, .. } = func.values[dst].ty else {
                unreachable!("local copy destinations are pointers");
            };
            for (path, ty) in leaves {
                let (dst_ptr, dst_inst) = make_value(
                    func,
                    block,
                    InstKind::GetElementPtr {
                        base: dst,
                        indices: path.clone().into_iter().collect(),
                    },
                    Ty::ptr_bits(ty.clone(), bits),
                );
                rebuilt.push(dst_inst);
                let (src_ptr, src_inst) = make_value(
                    func,
                    block,
                    InstKind::GetElementPtr {
                        base: src,
                        indices: path.into_iter().collect(),
                    },
                    Ty::ptr_bits(ty.clone(), bits),
                );
                rebuilt.push(src_inst);
                let (value, load_inst) = make_value(
                    func,
                    block,
                    InstKind::Load {
                        addr: src_ptr,
                        flags: MemFlags::plain(),
                    },
                    ty.clone(),
                );
                rebuilt.push(load_inst);
                rebuilt.push(make_unit(
                    func,
                    block,
                    InstKind::Store {
                        addr: dst_ptr,
                        val: value,
                        flags: MemFlags::plain(),
                    },
                ));
            }
            expanded = true;
            changed = true;
        }
        if expanded {
            func.blocks[block].insts = rebuilt;
        }
    }
    changed
}

fn local_aggregate_copy_leaves(func: &Function, dst: Value, src: Value, len: u64, non_escaping: &std::collections::HashSet<Value>) -> Option<Vec<(Vec<GepIndex>, Ty)>> {
    let Ty::Ptr { pointee, .. } = &func.values[dst].ty else {
        return None;
    };
    if func.values[src].ty != func.values[dst].ty || pointee.size_bytes() != len {
        return None;
    }
    let (MemoryObject::Stack(dst_root), MemoryObject::Stack(src_root)) = (address_info(func, dst)?.object, address_info(func, src)?.object) else {
        return None;
    };
    if dst_root == src_root || !non_escaping.contains(&dst_root) || !non_escaping.contains(&src_root) {
        return None;
    }
    let mut leaves = Vec::new();
    collect_copy_leaves(pointee, &mut Vec::new(), &mut leaves, 16).then_some(leaves)
}

fn collect_copy_leaves(ty: &Ty, path: &mut Vec<GepIndex>, out: &mut Vec<(Vec<GepIndex>, Ty)>, budget: usize) -> bool {
    match ty {
        Ty::Bool | Ty::Int { .. } | Ty::Float(_) | Ty::Ptr { .. } if budget > 0 => {
            out.push((path.clone(), ty.clone()));
            true
        }
        Ty::Struct { fields, offsets, .. } => {
            if fields.len() != offsets.len() {
                return false;
            }
            let mut end = 0;
            for (index, (field, offset)) in fields.iter().zip(offsets).enumerate() {
                if *offset < end {
                    return false;
                }
                path.push(GepIndex::Field(index as u32));
                if !collect_copy_leaves(field, path, out, budget.saturating_sub(out.len())) {
                    return false;
                }
                path.pop();
                end = match offset.checked_add(field.size_bytes()) {
                    Some(end) => end,
                    None => return false,
                };
            }
            true
        }
        Ty::Array { elem, len } => {
            for index in 0..*len {
                path.push(GepIndex::ConstIndex(index));
                if !collect_copy_leaves(elem, path, out, budget.saturating_sub(out.len())) {
                    return false;
                }
                path.pop();
            }
            true
        }
        _ => false,
    }
}

fn make_value(func: &mut Function, block: Block, kind: InstKind, ty: Ty) -> (Value, Inst) {
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
    (result, inst)
}

fn make_unit(func: &mut Function, block: Block, kind: InstKind) -> Inst {
    func.insts.push(InstData {
        block,
        kind,
        result: None,
        loc: None,
    })
}

/// Split an aggregate store into its scalar fields. The source aggregate
/// value is read with `ExtractValue`, so a following `Sroa` sees only the
/// individual field stores.
fn expand_local_aggregate_stores(func: &mut Function, non_escaping: &std::collections::HashSet<Value>) -> bool {
    let mut changed = false;
    for block_idx in 0..func.blocks.len() {
        let block = Block::from_usize(block_idx);
        let insts = func.blocks[block].insts.clone();
        let mut expanded = false;
        let mut rebuilt = Vec::with_capacity(insts.len());
        for inst in insts {
            let store = match &func.insts[inst].kind {
                InstKind::Store { addr, val, flags } if !flags.volatile => Some((*addr, *val, *flags)),
                _ => None,
            };
            let Some((addr, val, flags)) = store else {
                rebuilt.push(inst);
                continue;
            };
            let Ty::Ptr { pointee, bits } = func.values[addr].ty.clone() else {
                rebuilt.push(inst);
                continue;
            };
            if func.values[val].ty != *pointee || !pointee.is_aggregate() {
                rebuilt.push(inst);
                continue;
            }
            if !matches!(
                address_info(func, addr).map(|info| info.object),
                Some(MemoryObject::Stack(root)) if non_escaping.contains(&root)
            ) {
                rebuilt.push(inst);
                continue;
            }
            let Some(leaves) = aggregate_leaves(&pointee) else {
                rebuilt.push(inst);
                continue;
            };
            for (path, ty) in leaves {
                let Some(extracted) = extract_value_path(func, block, val, &path, &mut rebuilt) else {
                    unreachable!("validated aggregate path");
                };
                let (dst, dst_inst) = make_value(
                    func,
                    block,
                    InstKind::GetElementPtr {
                        base: addr,
                        indices: path.into_iter().collect(),
                    },
                    Ty::ptr_bits(ty, bits),
                );
                rebuilt.push(dst_inst);
                rebuilt.push(make_unit(func, block, InstKind::Store { addr: dst, val: extracted, flags }));
            }
            expanded = true;
            changed = true;
        }
        if expanded {
            func.blocks[block].insts = rebuilt;
        }
    }
    changed
}

fn extract_value_path(func: &mut Function, block: Block, aggregate: Value, path: &[GepIndex], rebuilt: &mut Vec<Inst>) -> Option<Value> {
    let mut value = aggregate;
    for step in path {
        let (index, ty) = match (&func.values[value].ty, step) {
            (Ty::Struct { fields, .. }, GepIndex::Field(index)) => (*index, fields.get(*index as usize)?.clone()),
            (Ty::Array { elem, len }, GepIndex::ConstIndex(index)) if *index < *len => (*index as u32, (**elem).clone()),
            _ => return None,
        };
        let (next, inst) = make_value(func, block, InstKind::ExtractValue { agg: value, index }, ty);
        rebuilt.push(inst);
        value = next;
    }
    Some(value)
}

/// Replace an aggregate load whose only use is the function's return with a
/// field-wise load and `InsertValue` chain. This lets `Sroa` scalarize the
/// source slot while still returning a value of the aggregate type.
fn expand_returned_aggregate_loads(func: &mut Function, non_escaping: &std::collections::HashSet<Value>) -> bool {
    let counts = func.count_resident_uses();
    let mut changed = false;
    for block_idx in 0..func.blocks.len() {
        let block = Block::from_usize(block_idx);
        let Some(Terminator::Return { val: Some(returned) }) = func.blocks[block].term else {
            continue;
        };
        let insts = func.blocks[block].insts.clone();
        let mut replacement = None;
        let mut rebuilt = Vec::with_capacity(insts.len());
        for inst in insts {
            let load = match &func.insts[inst].kind {
                InstKind::Load { addr, flags } if !flags.volatile => Some((*addr, *flags)),
                _ => None,
            };
            let Some((addr, flags)) = load else {
                rebuilt.push(inst);
                continue;
            };
            let Some(result) = func.insts[inst].result else {
                rebuilt.push(inst);
                continue;
            };
            let ty = func.values[result].ty.clone();
            if result != returned || counts.get(result.index()).copied().unwrap_or(0) != 1 || !ty.is_aggregate() {
                rebuilt.push(inst);
                continue;
            }
            if !matches!(
                address_info(func, addr).map(|info| info.object),
                Some(MemoryObject::Stack(root)) if non_escaping.contains(&root)
            ) {
                rebuilt.push(inst);
                continue;
            }
            let Ty::Ptr { bits, .. } = func.values[addr].ty else {
                rebuilt.push(inst);
                continue;
            };
            let Some(leaves) = aggregate_leaves(&ty) else {
                rebuilt.push(inst);
                continue;
            };
            let mut undefs = Vec::new();
            let mut budget = leaves.len();
            let Some(aggregate) = build_aggregate_value(func, block, addr, &mut Vec::new(), &ty, bits, flags, &mut undefs, &mut budget, &mut rebuilt) else {
                rebuilt.push(inst);
                continue;
            };
            replacement = Some(aggregate);
            changed = true;
        }
        if replacement.is_some() {
            func.blocks[block].insts = rebuilt;
            func.blocks[block].term = Some(Terminator::Return { val: replacement });
        }
    }
    changed
}

#[allow(clippy::too_many_arguments)]
fn build_aggregate_value(
    func: &mut Function,
    block: Block,
    addr: Value,
    path: &mut Vec<GepIndex>,
    ty: &Ty,
    bits: u32,
    flags: MemFlags,
    undefs: &mut Vec<(Ty, Value)>,
    budget: &mut usize,
    rebuilt: &mut Vec<Inst>,
) -> Option<Value> {
    match ty {
        Ty::Bool | Ty::Int { .. } | Ty::Float(_) | Ty::Ptr { .. } => {
            if *budget == 0 {
                return None;
            }
            *budget -= 1;
            let (field, field_inst) = make_value(
                func,
                block,
                InstKind::GetElementPtr {
                    base: addr,
                    indices: path.clone().into_iter().collect(),
                },
                Ty::ptr_bits(ty.clone(), bits),
            );
            rebuilt.push(field_inst);
            let (loaded, load_inst) = make_value(func, block, InstKind::Load { addr: field, flags }, ty.clone());
            rebuilt.push(load_inst);
            Some(loaded)
        }
        Ty::Struct { fields, offsets, .. } => {
            if fields.len() != offsets.len() {
                return None;
            }
            let mut end = 0;
            let mut aggregate = undef_value(func, ty, undefs);
            for (index, (field, offset)) in fields.iter().zip(offsets).enumerate() {
                if *offset < end {
                    return None;
                }
                path.push(GepIndex::Field(index as u32));
                let value = build_aggregate_value(func, block, addr, path, field, bits, flags, undefs, budget, rebuilt)?;
                path.pop();
                let (next, insert) = make_value(
                    func,
                    block,
                    InstKind::InsertValue {
                        agg: aggregate,
                        elem: value,
                        index: index as u32,
                    },
                    ty.clone(),
                );
                rebuilt.push(insert);
                aggregate = next;
                end = offset.checked_add(field.size_bytes())?;
            }
            Some(aggregate)
        }
        Ty::Array { elem, len } => {
            let mut aggregate = undef_value(func, ty, undefs);
            for index in 0..*len {
                path.push(GepIndex::ConstIndex(index));
                let value = build_aggregate_value(func, block, addr, path, elem, bits, flags, undefs, budget, rebuilt)?;
                path.pop();
                let (next, insert) = make_value(
                    func,
                    block,
                    InstKind::InsertValue {
                        agg: aggregate,
                        elem: value,
                        index: index as u32,
                    },
                    ty.clone(),
                );
                rebuilt.push(insert);
                aggregate = next;
            }
            Some(aggregate)
        }
        _ => None,
    }
}

fn aggregate_leaves(ty: &Ty) -> Option<Vec<(Vec<GepIndex>, Ty)>> {
    let mut leaves = Vec::new();
    collect_copy_leaves(ty, &mut Vec::new(), &mut leaves, 16).then_some(leaves)
}

fn extract_step(ty: &Ty, index: u32) -> Option<(GepIndex, Ty)> {
    match ty {
        Ty::Struct { fields, .. } => fields.get(index as usize).map(|field| (GepIndex::Field(index), field.clone())),
        Ty::Array { elem, len } if u64::from(index) < *len => Some((GepIndex::ConstIndex(u64::from(index)), (**elem).clone())),
        _ => None,
    }
}

/// Replace an aggregate load consumed only by scalar `ExtractValue`s with
/// scalar loads at the same point, removing both the aggregate load and the
/// extraction. The source slot is then visible to `Sroa`.
fn expand_aggregate_load_extracts(func: &mut Function, non_escaping: &std::collections::HashSet<Value>) -> bool {
    use std::collections::{HashMap, HashSet};

    let resident = func.resident_mask();
    let counts = func.count_resident_uses();
    let mut term_values = HashSet::new();
    for block in func.blocks.iter() {
        if let Some(term) = &block.term {
            term.for_each_operand(&mut |value| {
                term_values.insert(value);
            });
        }
    }
    let mut by_aggregate: HashMap<Value, Vec<Inst>> = HashMap::new();
    for (id, inst) in func.insts.iter_enumerated() {
        if !resident[id.index()] {
            continue;
        }
        if let InstKind::ExtractValue { agg, .. } = &inst.kind {
            by_aggregate.entry(*agg).or_default().push(id);
        }
    }

    struct Extraction {
        inst: Inst,
        result: Value,
        step: GepIndex,
        ty: Ty,
    }
    let mut candidates: HashMap<Inst, (Value, u32, MemFlags, Vec<Extraction>)> = HashMap::new();
    let mut removed = HashSet::new();
    for (id, inst) in func.insts.iter_enumerated() {
        if !resident[id.index()] {
            continue;
        }
        let InstKind::Load { addr, flags } = &inst.kind else {
            continue;
        };
        if flags.volatile {
            continue;
        }
        let Some(MemoryObject::Stack(root)) = address_info(func, *addr).map(|info| info.object) else {
            continue;
        };
        if !non_escaping.contains(&root) {
            continue;
        }
        let Some(result) = inst.result else { continue };
        if counts.get(result.index()).copied().unwrap_or(0) == 0 || term_values.contains(&result) {
            continue;
        }
        let Some(uses) = by_aggregate.get(&result) else { continue };
        if uses.len() != counts[result.index()] as usize {
            continue;
        }
        let Ty::Ptr { pointee, bits } = &func.values[*addr].ty else {
            continue;
        };
        if **pointee != func.values[result].ty {
            continue;
        }
        let mut extractions = Vec::with_capacity(uses.len());
        let mut valid = true;
        for &use_inst in uses {
            let InstKind::ExtractValue { index, .. } = &func.insts[use_inst].kind else {
                valid = false;
                break;
            };
            let Some(extracted) = func.insts[use_inst].result else {
                valid = false;
                break;
            };
            let Some((step, ty)) = extract_step(&func.values[result].ty, *index) else {
                valid = false;
                break;
            };
            if !ty.is_scalar() {
                valid = false;
                break;
            }
            extractions.push(Extraction {
                inst: use_inst,
                result: extracted,
                step,
                ty,
            });
        }
        if valid {
            removed.extend(extractions.iter().map(|extraction| extraction.inst));
            candidates.insert(id, (*addr, *bits, *flags, extractions));
        }
    }
    if candidates.is_empty() {
        return false;
    }

    let mut replacements = Vec::new();
    for block_idx in 0..func.blocks.len() {
        let block = Block::from_usize(block_idx);
        let insts = func.blocks[block].insts.clone();
        let mut rebuilt = Vec::with_capacity(insts.len());
        for inst in insts {
            if removed.contains(&inst) {
                continue;
            }
            let Some((addr, bits, flags, extractions)) = candidates.get(&inst) else {
                rebuilt.push(inst);
                continue;
            };
            for extraction in extractions {
                let (field, field_inst) = make_value(
                    func,
                    block,
                    InstKind::GetElementPtr {
                        base: *addr,
                        indices: [extraction.step.clone()].into_iter().collect(),
                    },
                    Ty::ptr_bits(extraction.ty.clone(), *bits),
                );
                rebuilt.push(field_inst);
                let (loaded, load_inst) = make_value(func, block, InstKind::Load { addr: field, flags: *flags }, extraction.ty.clone());
                rebuilt.push(load_inst);
                replacements.push((extraction.result, loaded));
            }
        }
        func.blocks[block].insts = rebuilt;
    }
    for (old, new) in replacements {
        func.replace_uses(old, new);
    }
    true
}

/// A later write that may cover an earlier one.
struct MemoryWrite {
    addr: Value,
    len: u64,
}

/// A later read that may observe an earlier write.
struct MemoryRead {
    addr: Value,
    len: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MemoryObject {
    Stack(Value),
    Global(Value),
    /// A payload returned by an allocator extern. Distinct allocation
    /// sites name disjoint objects; offsets into one allocation still
    /// compare by byte position.
    Heap(Value),
}

#[derive(Clone, Copy)]
pub(crate) struct AddressInfo {
    pub(crate) object: MemoryObject,
    pub(crate) offset: Option<u64>,
}

fn add_offset(offset: Option<u64>, delta: Option<(u32, i128)>) -> Option<u64> {
    let offset = offset?;
    let delta = delta?.1;
    let result = offset as i128 + delta;
    (0..=u64::MAX as i128).contains(&result).then_some(result as u64)
}

fn sub_offset(offset: Option<u64>, delta: Option<(u32, i128)>) -> Option<u64> {
    let offset = offset?;
    let delta = delta?.1;
    let result = offset as i128 - delta;
    (0..=u64::MAX as i128).contains(&result).then_some(result as u64)
}

/// Track the stack, global, or allocation object a pointer or address
/// integer derives from, plus a constant byte offset when the arithmetic
/// is known. `PtrToInt`/`IntToPtr` are address algebra here; escape
/// analysis below is what decides whether a stack address actually
/// leaves the function.
pub(crate) fn address_info(func: &Function, value: Value) -> Option<AddressInfo> {
    let Def::Inst(inst) = func.values[value].def else {
        return None;
    };
    match &func.insts[inst].kind {
        InstKind::Alloca { .. } => Some(AddressInfo {
            object: MemoryObject::Stack(value),
            offset: Some(0),
        }),
        InstKind::CallExtern { target, .. } if func.extern_effect(*target).allocates && matches!(func.values[value].ty, Ty::Ptr { .. }) => Some(AddressInfo {
            object: MemoryObject::Heap(value),
            offset: Some(0),
        }),
        InstKind::Bitcast(inner) if matches!(func.values[value].ty, Ty::Ptr { .. }) && matches!(func.values[*inner].ty, Ty::Ptr { .. }) => address_info(func, *inner),
        InstKind::GlobalAddr { .. } => Some(AddressInfo {
            object: MemoryObject::Global(value),
            offset: Some(0),
        }),
        InstKind::GetElementPtr { base, indices } => {
            let mut info = address_info(func, *base)?;
            let mut ty = match &func.values[*base].ty {
                Ty::Ptr { pointee, .. } => (**pointee).clone(),
                _ => return Some(info),
            };
            for index in indices.iter() {
                match index {
                    crate::ty::GepIndex::Field(field) => {
                        let Ty::Struct { fields, offsets, .. } = &ty else {
                            info.offset = None;
                            continue;
                        };
                        let (Some(next), Some(field_offset)) = (fields.get(*field as usize), offsets.get(*field as usize)) else {
                            info.offset = None;
                            continue;
                        };
                        info.offset = info.offset.and_then(|off| off.checked_add(*field_offset));
                        ty = next.clone();
                    }
                    crate::ty::GepIndex::ConstIndex(index) => {
                        let (step, next) = match &ty {
                            Ty::Array { elem, .. } => (elem.size_bytes(), (**elem).clone()),
                            other => (other.size_bytes(), other.clone()),
                        };
                        info.offset = info.offset.and_then(|off| index.checked_mul(step).and_then(|delta| off.checked_add(delta)));
                        ty = next;
                    }
                    crate::ty::GepIndex::Index(index) => {
                        let (step, next) = match &ty {
                            Ty::Array { elem, .. } => (elem.size_bytes(), (**elem).clone()),
                            other => (other.size_bytes(), other.clone()),
                        };
                        info.offset = match (info.offset, int_const(func, *index)) {
                            (Some(off), Some((_, index))) => u64::try_from(index).ok().and_then(|index| index.checked_mul(step)).and_then(|delta| off.checked_add(delta)),
                            _ => None,
                        };
                        ty = next;
                    }
                }
            }
            Some(info)
        }
        InstKind::Bitcast(inner) => address_info(func, *inner),
        InstKind::PtrToInt(inner) => {
            if matches!(func.values[value].ty, Ty::Int { bits: 64 }) {
                address_info(func, *inner)
            } else {
                None
            }
        }
        InstKind::IntToPtr(inner) => {
            if matches!(func.values[*inner].ty, Ty::Int { bits: 64 }) {
                address_info(func, *inner)
            } else {
                None
            }
        }
        InstKind::Add(a, b) => match (address_info(func, *a), address_info(func, *b)) {
            (Some(info), None) => Some(AddressInfo {
                offset: add_offset(info.offset, int_const(func, *b)),
                ..info
            }),
            (None, Some(info)) => Some(AddressInfo {
                offset: add_offset(info.offset, int_const(func, *a)),
                ..info
            }),
            _ => None,
        },
        InstKind::Sub(a, b) => match (address_info(func, *a), address_info(func, *b)) {
            (Some(info), None) => Some(AddressInfo {
                offset: sub_offset(info.offset, int_const(func, *b)),
                ..info
            }),
            _ => None,
        },
        _ => None,
    }
}

fn address_object(func: &Function, value: Value) -> Option<MemoryObject> {
    address_info(func, value).map(|info| info.object)
}

fn escape_root(func: &Function, value: Value, escaped: &mut std::collections::HashSet<Value>) {
    if let Some(MemoryObject::Stack(root)) = address_object(func, value) {
        escaped.insert(root);
    }
}

fn escape_value(func: &Function, value: Value, escaped: &mut std::collections::HashSet<Value>) {
    if !func.values[value].ty.is_aggregate() {
        escape_root(func, value, escaped);
    }
}

/// Stack slots whose address never leaves this function. Unknown
/// pointers cannot point at these slots, so a read through a parameter
/// need not be treated as aliasing them.
fn non_escaping_stack_slots(func: &Function) -> std::collections::HashSet<Value> {
    use std::collections::HashSet;

    let resident = func.resident_mask();
    let mut escaped = HashSet::new();
    for (id, inst) in func.insts.iter_enumerated() {
        if !resident[id.index()] {
            continue;
        }
        let result_ty = inst.result.map(|value| &func.values[value].ty);
        match &inst.kind {
            InstKind::Load { .. }
            | InstKind::Memcpy { .. }
            | InstKind::Memset { .. }
            | InstKind::AtomicLoad { .. }
            | InstKind::VaStart { .. }
            | InstKind::VaArg { .. }
            | InstKind::VaEnd { .. }
            | InstKind::VaCopy { .. }
            | InstKind::Alloca { .. }
            | InstKind::GlobalAddr { .. }
            | InstKind::FuncAddr { .. }
            | InstKind::ExternAddr { .. }
            | InstKind::GetFramePointer
            | InstKind::AtomicFence => {}
            InstKind::Store { val, .. } => escape_value(func, *val, &mut escaped),
            InstKind::GetElementPtr { .. } => {}
            InstKind::Bitcast(inner) => {
                let pointer_bitcast = result_ty.is_some_and(|ty| matches!(ty, Ty::Ptr { .. })) && matches!(func.values[*inner].ty, Ty::Ptr { .. });
                if !pointer_bitcast {
                    escape_root(func, *inner, &mut escaped);
                }
            }
            InstKind::PtrToInt(_) | InstKind::IntToPtr(_) => {}
            InstKind::Add(a, b) | InstKind::Sub(a, b) => {
                if address_info(func, *a).is_some() && address_info(func, *b).is_some() {
                    escape_root(func, *a, &mut escaped);
                    escape_root(func, *b, &mut escaped);
                }
            }
            InstKind::Icmp { .. } | InstKind::Fcmp { .. } => {}
            InstKind::Select { then_val, else_val, .. } => {
                escape_root(func, *then_val, &mut escaped);
                escape_root(func, *else_val, &mut escaped);
            }
            InstKind::ExtractValue { .. } => {}
            InstKind::InsertValue { elem, .. } => escape_value(func, *elem, &mut escaped),
            InstKind::AtomicStore { val, .. } | InstKind::AtomicRmw { val, .. } => {
                escape_value(func, *val, &mut escaped);
            }
            InstKind::AtomicCas { expected, replacement, .. } => {
                escape_value(func, *expected, &mut escaped);
                escape_value(func, *replacement, &mut escaped);
            }
            InstKind::Call { args, .. } | InstKind::CallExtern { args, .. } => {
                for &arg in args.iter() {
                    escape_root(func, arg, &mut escaped);
                }
            }
            InstKind::CallIndirect { callee, args, .. } => {
                escape_root(func, *callee, &mut escaped);
                for &arg in args.iter() {
                    escape_root(func, arg, &mut escaped);
                }
            }
            InstKind::Await { fut } => escape_root(func, *fut, &mut escaped),
            InstKind::InlineAsm { block } => {
                for operand in &block.operands {
                    escape_root(func, operand.value, &mut escaped);
                }
            }
            kind if kind.is_pure() => {
                kind.for_each_operand(&mut |value| escape_root(func, value, &mut escaped));
            }
            other => {
                other.for_each_operand(&mut |value| escape_root(func, value, &mut escaped));
            }
        }
    }
    for block in func.blocks.iter() {
        let Some(term) = &block.term else {
            continue;
        };
        match term {
            Terminator::Return { val: Some(val) } => escape_value(func, *val, &mut escaped),
            Terminator::Br { args, .. } => {
                for &arg in args.iter() {
                    escape_root(func, arg, &mut escaped);
                }
            }
            Terminator::CondBr { then_args, else_args, .. } => {
                for &arg in then_args.iter().chain(else_args.iter()) {
                    escape_root(func, arg, &mut escaped);
                }
            }
            Terminator::Return { val: None } | Terminator::Switch { .. } | Terminator::Unreachable => {}
        }
    }

    func.insts
        .iter_enumerated()
        .filter_map(|(id, inst)| match (&inst.kind, inst.result) {
            (InstKind::Alloca { .. }, Some(slot)) if resident[id.index()] && !escaped.contains(&slot) => Some(slot),
            _ => None,
        })
        .collect()
}

fn range_contains(start: u64, len: u64, inner_start: u64, inner_len: u64) -> bool {
    let (Some(end), Some(inner_end)) = (start.checked_add(len), inner_start.checked_add(inner_len)) else {
        return false;
    };
    start <= inner_start && inner_end <= end
}

fn ranges_overlap(start: u64, len: u64, other_start: u64, other_len: u64) -> bool {
    if len == 0 || other_len == 0 {
        return false;
    }
    let (Some(end), Some(other_end)) = (start.checked_add(len), other_start.checked_add(other_len)) else {
        return true;
    };
    start < other_end && other_start < end
}

fn ranges_may_overlap(func: &Function, non_escaping: &std::collections::HashSet<Value>, a_addr: Value, a_len: u64, b_addr: Value, b_len: u64) -> bool {
    if a_len == 0 || b_len == 0 {
        return false;
    }
    if a_addr == b_addr {
        return true;
    }
    match (address_info(func, a_addr), address_info(func, b_addr)) {
        (Some(a), Some(b)) => {
            let may_alias = match (a.object, b.object) {
                (MemoryObject::Stack(a), MemoryObject::Stack(b)) => a == b,
                (MemoryObject::Heap(a), MemoryObject::Heap(b)) => a == b,
                (MemoryObject::Global(_), MemoryObject::Global(_)) => true,
                _ => false,
            };
            if !may_alias {
                return false;
            }
            match (a.offset, b.offset) {
                (Some(a), Some(b)) => ranges_overlap(a, a_len, b, b_len),
                _ => true,
            }
        }
        (
            Some(AddressInfo {
                object: MemoryObject::Stack(stack),
                ..
            }),
            None,
        )
        | (
            None,
            Some(AddressInfo {
                object: MemoryObject::Stack(stack),
                ..
            }),
        ) => !non_escaping.contains(&stack),
        _ => true,
    }
}

fn write_covers(func: &Function, write_addr: Value, write_len: u64, addr: Value, len: u64) -> bool {
    if write_addr == addr {
        return write_len >= len;
    }
    let Some(write) = address_info(func, write_addr) else {
        return false;
    };
    let Some(target) = address_info(func, addr) else {
        return false;
    };
    if write.object != target.object {
        return false;
    }
    match (write.offset, target.offset) {
        (Some(write_offset), Some(target_offset)) => range_contains(write_offset, write_len, target_offset, len),
        _ => false,
    }
}

fn covered_by_later_write(writes: &[MemoryWrite], func: &Function, addr: Value, len: u64) -> bool {
    writes.iter().any(|write| write_covers(func, write.addr, write.len, addr, len))
}

fn read_after_write(reads: &[MemoryRead], func: &Function, non_escaping: &std::collections::HashSet<Value>, addr: Value, len: u64) -> bool {
    reads
        .iter()
        .any(|read| read.len > 0 && ranges_may_overlap(func, non_escaping, read.addr, read.len, addr, len))
}

/// A write makes later reads of the same bytes read the new value, so
/// those reads no longer keep earlier writes alive.
fn forget_covered_reads(reads: &mut Vec<MemoryRead>, func: &Function, addr: Value, len: u64) {
    reads.retain(|read| !write_covers(func, addr, len, read.addr, read.len));
}

/// Remove a memory write whose bytes are fully overwritten by a later
/// write in the same block, when no intervening read may observe it.
///
/// Writes are tracked backwards through the block. A read from a
/// different known object cannot alias; a read from an unknown pointer
/// may, so it keeps the write live. Calls and volatile/atomic accesses
/// are barriers: they may read anything, so earlier writes stay. An
/// extern explicitly marked as touching no program memory is the one
/// exception.
#[derive(Default)]
pub struct DeadStoreElim;

impl FunctionPass for DeadStoreElim {
    fn name(&self) -> &'static str {
        "dse"
    }

    fn run(&mut self, func: &mut Function) -> bool {
        let non_escaping = non_escaping_stack_slots(func);
        let mut dead = std::collections::HashSet::new();
        for index in 0..func.blocks.len() {
            let block = Block::from_usize(index);
            eliminate_dead_stores_in_block(func, block, &non_escaping, &mut dead);
        }
        if dead.is_empty() {
            return false;
        }
        for block in func.blocks.iter_mut() {
            block.insts.retain(|inst| !dead.contains(inst));
        }
        true
    }
}

fn eliminate_dead_stores_in_block(func: &Function, block: Block, non_escaping: &std::collections::HashSet<Value>, dead: &mut std::collections::HashSet<Inst>) {
    let insts = func.blocks[block].insts.clone();
    let mut writes: Vec<MemoryWrite> = Vec::new();
    let mut reads: Vec<MemoryRead> = Vec::new();
    for &inst in insts.iter().rev() {
        let data = &func.insts[inst];
        match &data.kind {
            InstKind::Load { addr, flags } => {
                if flags.volatile {
                    return;
                }
                let len = data.result.map(|value| func.values[value].ty.size_bytes()).unwrap_or(0);
                if len > 0 {
                    reads.push(MemoryRead { addr: *addr, len });
                }
            }
            InstKind::Store { addr, val, flags } => {
                if flags.volatile {
                    return;
                }
                let len = func.values[*val].ty.size_bytes();
                if len == 0 {
                    continue;
                }
                if covered_by_later_write(&writes, func, *addr, len) && !read_after_write(&reads, func, non_escaping, *addr, len) {
                    dead.insert(inst);
                }
                forget_covered_reads(&mut reads, func, *addr, len);
                writes.push(MemoryWrite { addr: *addr, len });
            }
            InstKind::Memcpy { dst, src, len } => {
                if *len == 0 {
                    continue;
                }
                let is_dead = covered_by_later_write(&writes, func, *dst, *len) && !read_after_write(&reads, func, non_escaping, *dst, *len);
                if is_dead {
                    dead.insert(inst);
                }
                forget_covered_reads(&mut reads, func, *dst, *len);
                if !is_dead {
                    reads.push(MemoryRead { addr: *src, len: *len });
                }
                writes.push(MemoryWrite { addr: *dst, len: *len });
            }
            InstKind::Memset { dst, len, .. } => {
                if *len == 0 {
                    continue;
                }
                if covered_by_later_write(&writes, func, *dst, *len) && !read_after_write(&reads, func, non_escaping, *dst, *len) {
                    dead.insert(inst);
                }
                forget_covered_reads(&mut reads, func, *dst, *len);
                writes.push(MemoryWrite { addr: *dst, len: *len });
            }
            InstKind::ExtractValue { agg, .. } | InstKind::InsertValue { agg, .. } => {
                let len = func.values[*agg].ty.size_bytes();
                if len > 0 {
                    reads.push(MemoryRead { addr: *agg, len });
                }
            }
            kind if kind.is_pure() => {}
            InstKind::Udiv(..) | InstKind::Sdiv(..) | InstKind::Urem(..) | InstKind::Srem(..) => {}
            InstKind::CallExtern { target, .. } => {
                let effect = func.extern_effect(*target);
                if effect.reads || effect.writes {
                    return;
                }
            }
            _ => return,
        }
    }
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
        let resident = func.resident_mask();
        let counts = func.count_resident_uses();
        let mut dead = Vec::new();
        for (id, inst) in func.insts.iter_enumerated() {
            if !resident[id.index()] {
                continue;
            }
            if !inst.kind.is_pure() {
                continue;
            }
            if let Some(result) = inst.result
                && counts.get(result.index()).copied().unwrap_or(0) == 0 {
                    dead.push(id);
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
/// reuse. Arena-orphaned instructions cannot execute, so they do not
/// count.
pub fn escaping_addresses(func: &Function) -> Vec<bool> {
    let resident = func.resident_mask();
    let mut escaped = vec![false; func.values.len()];
    for (id, inst) in func.insts.iter_enumerated() {
        if !resident[id.index()] {
            continue;
        }
        match &inst.kind {
            InstKind::Load { .. } => {}
            InstKind::Store { val, .. } => escaped[val.index()] = true,
            other => other.for_each_operand(&mut |v| escaped[v.index()] = true),
        }
    }
    for block in func.blocks.iter() {
        if let Some(term) = &block.term {
            term.for_each_operand(&mut |v| escaped[v.index()] = true);
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
    // A `setjmp` call site can be re-entered by `longjmp` with the
    // registers of the first return: any slot written between the two
    // would read stale. Leave every slot in memory.
    if func.returns_twice {
        return false;
    }
    let resident = func.resident_mask();
    // Every alloca from lowering is promoted on the first run; later
    // pipeline turns only hold orphaned ones in the arena. Check before
    // the remaining scans so those turns stay cheap.
    if !func
        .insts
        .iter_enumerated()
        .any(|(id, inst)| resident[id.index()] && matches!(&inst.kind, InstKind::Alloca { pointee, count } if *count == 1 && pointee.is_scalar()))
    {
        return false;
    }
    let escaped = escaping_addresses(func);
    let reachable = func.reachable_blocks();

    // Index load/store sites by address once: the per-slot scan used
    // to be O(slots * insts).
    let mut sites: Vec<Vec<Inst>> = vec![Vec::new(); func.values.len()];
    for (id, inst) in func.insts.iter_enumerated() {
        if !resident[id.index()] {
            continue;
        }
        if let InstKind::Load { addr, .. } | InstKind::Store { addr, .. } = &inst.kind {
            sites[addr.index()].push(id);
        }
    }

    let mut promotable: Vec<SlotSites> = Vec::new();
    let mut unused: Vec<Inst> = Vec::new();
    for (id, inst) in func.insts.iter_enumerated() {
        if !resident[id.index()] {
            continue;
        }
        let InstKind::Alloca { pointee, count } = &inst.kind else {
            continue;
        };
        if *count != 1 || !pointee.is_scalar() {
            continue;
        }
        let Some(addr) = inst.result else { continue };
        if escaped[addr.index()] {
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
        for &site in &sites[addr.index()] {
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
        // Promote busy slots first: the order is otherwise arbitrary,
        // and hot slots first measured slightly better.
        promotable.sort_by_key(|slot| (std::cmp::Reverse(slot.loads.len() + slot.stores.len()), slot.alloca.index()));
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
    let Some(phi_blocks) = place_phis(func, dom, &store_blocks, switch_targets, slot) else {
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

/// Cytron's iterated dominance frontier: a block that merges definitions
/// of the slot gets a phi only when the slot is live-in there. Returns
/// `None` when a phi would land somewhere unsupported (the entry block
/// or a `Switch` target), leaving the caller to keep the slot in memory.
fn place_phis(func: &Function, dom: &DomInfo, store_blocks: &[Block], switch_targets: &std::collections::HashSet<Block>, slot: &SlotSites) -> Option<Vec<Block>> {
    let live_in = slot_live_in(func, slot);
    let mut placed: std::collections::HashSet<Block> = std::collections::HashSet::new();
    let mut work: Vec<Block> = store_blocks.to_vec();
    let mut phis = Vec::new();
    while let Some(block) = work.pop() {
        for &frontier in &dom.df[block.index()] {
            if !placed.insert(frontier) || !live_in[frontier.index()] {
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

/// Liveness of one slot's value: true when a block can load the slot
/// before any store on a path from its entry. Pruned SSA only needs a
/// phi where the value is live-in, which keeps variables whose
/// lifetimes stay inside one loop iteration out of the loop header.
fn slot_live_in(func: &Function, slot: &SlotSites) -> Vec<bool> {
    let reachable = func.reachable_blocks();
    let mut live = vec![false; func.blocks.len()];
    let mut defines = vec![false; func.blocks.len()];
    for (index, block) in func.blocks.iter_enumerated() {
        if !reachable[index.index()] {
            continue;
        }
        let mut defined = false;
        for &inst in &block.insts {
            match &func.insts[inst].kind {
                InstKind::Load { addr, .. } if *addr == slot.addr => {
                    if !defined {
                        live[index.index()] = true;
                    }
                }
                InstKind::Store { addr, .. } if *addr == slot.addr => defined = true,
                _ => {}
            }
        }
        defines[index.index()] = defined;
    }
    let mut changed = true;
    while changed {
        changed = false;
        for (index, block) in func.blocks.iter_enumerated() {
            if !reachable[index.index()] || live[index.index()] || defines[index.index()] {
                continue;
            }
            let flows_out = block.term.as_ref().is_some_and(|term| term.successors().into_iter().any(|succ| live[succ.index()]));
            if flows_out {
                live[index.index()] = true;
                changed = true;
            }
        }
    }
    live
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
            if let Some(&index) = phi_index.get(dest)
                && let Some(arg) = args.get_mut(index) {
                    *arg = value;
                }
        }
        Terminator::CondBr {
            then_block,
            then_args,
            else_block,
            else_args,
            ..
        } => {
            if let Some(&index) = phi_index.get(then_block)
                && let Some(arg) = then_args.get_mut(index) {
                    *arg = value;
                }
            if let Some(&index) = phi_index.get(else_block)
                && let Some(arg) = else_args.get_mut(index) {
                    *arg = value;
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
                if let Some(new) = new
                    && idom[block.index()] != Some(new) {
                        idom[block.index()] = Some(new);
                        changed = true;
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

    /// Whether `node` lies below `ancestor` in the dominator tree.
    fn dominates(&self, ancestor: Block, node: Block, entry: Block) -> bool {
        let mut current = node;
        loop {
            if current == ancestor {
                return true;
            }
            if current == entry {
                return false;
            }
            match self.idom[current.index()] {
                Some(next) => current = next,
                None => return false,
            }
        }
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
        changed |= fold_redundant_successors(func);
        changed |= thread_trivial_blocks(func);
        changed |= merge_identical_blocks(func);
        // Threading and merging can make two arms identical; fold them in
        // the same run instead of waiting for another pipeline turn.
        changed |= fold_redundant_successors(func);
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
        changed |= if_convert_diamonds(func);
        // The converted arm now shares its destination with the other
        // arm; fold the fork into selects in the same run.
        changed |= fold_redundant_successors(func);
        changed |= sink_pure_instructions(func);
        changed
    }
}

/// A value's defining block, for dominance checks on threaded edges.
fn value_block(func: &Function, value: Value) -> Option<Block> {
    match func.values[value].def {
        Def::Param(block, _) => Some(block),
        Def::Inst(inst) => Some(func.insts[inst].block),
        Def::Const(_) => None,
    }
}

/// Collapse a branch whose arms end at the same block: equal arguments
/// become one unconditional branch, differing ones become selects at the
/// fork.
fn fold_redundant_successors(func: &mut Function) -> bool {
    let mut changed = false;
    for idx in 0..func.blocks.len() {
        let block = Block::from_usize(idx);
        let Some(term) = func.blocks[block].term.clone() else {
            continue;
        };
        let replacement = match term {
            Terminator::CondBr {
                cond,
                then_block,
                else_block,
                then_args,
                else_args,
            } => {
                if let Some((_, value)) = int_const(func, cond) {
                    let (dest, args) = if value != 0 { (then_block, then_args) } else { (else_block, else_args) };
                    Some(Terminator::Br { dest, args })
                } else if then_block == else_block {
                    fold_fork(func, block, cond, then_block, then_args, else_args)
                } else {
                    None
                }
            }
            Terminator::Switch { cases, default, .. } => {
                if cases.iter().all(|(_, dest)| *dest == default) {
                    Some(Terminator::Br {
                        dest: default,
                        args: Default::default(),
                    })
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(term) = replacement {
            func.blocks[block].term = Some(term);
            changed = true;
        }
    }
    changed
}

/// Turn `brif c, dest(args), dest(args')` into `dest(select c, a, a')`
/// for every differing argument.
fn fold_fork(
    func: &mut Function,
    block: Block,
    cond: Value,
    dest: Block,
    then_args: smallvec::SmallVec<[Value; 4]>,
    else_args: smallvec::SmallVec<[Value; 4]>,
) -> Option<Terminator> {
    if then_args.len() != else_args.len() {
        return None;
    }
    if !matches!(func.values[cond].ty, Ty::Bool | Ty::Int { .. }) {
        return None;
    }
    let mut args = smallvec::SmallVec::new();
    for (then_val, else_val) in then_args.into_iter().zip(else_args) {
        if then_val == else_val {
            args.push(then_val);
            continue;
        }
        let ty = func.values[then_val].ty.clone();
        if !ty.is_scalar() || func.values[else_val].ty != ty {
            return None;
        }
        let result = func.values.push(ValueData {
            ty: ty.clone(),
            def: Def::Inst(Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(InstData {
            block,
            kind: InstKind::Select { cond, then_val, else_val },
            result: Some(result),
            loc: None,
        });
        func.values[result].def = Def::Inst(inst);
        func.blocks[block].insts.push(inst);
        args.push(result);
    }
    Some(Terminator::Br { dest, args })
}

/// Thread edges through empty trampoline blocks. A branch that only
/// forwards to another block (and whose arguments are available at the
/// predecessor) can be bypassed, retiring a block and exposing more
/// identical-arm folds.
fn thread_trivial_blocks(func: &mut Function) -> bool {
    let reachable = func.reachable_blocks();
    let dom = DomInfo::compute(func, &reachable);
    let entry = func.entry;
    let mut changed = false;
    for idx in 0..func.blocks.len() {
        let pred = Block::from_usize(idx);
        if !reachable[idx] {
            continue;
        }
        let Some(term) = func.blocks[pred].term.clone() else {
            continue;
        };
        let replacement = match term {
            Terminator::Br { dest, .. } => thread_edge(func, &dom, entry, pred, dest).map(|(dest, args)| Terminator::Br { dest, args }),
            Terminator::CondBr {
                cond,
                then_block,
                else_block,
                then_args,
                else_args,
            } => {
                let then = thread_edge(func, &dom, entry, pred, then_block);
                let otherwise = thread_edge(func, &dom, entry, pred, else_block);
                if then.is_none() && otherwise.is_none() {
                    None
                } else {
                    let (then_block, then_args) = match then {
                        Some((dest, args)) => (dest, args),
                        None => (then_block, then_args),
                    };
                    let (else_block, else_args) = match otherwise {
                        Some((dest, args)) => (dest, args),
                        None => (else_block, else_args),
                    };
                    Some(Terminator::CondBr {
                        cond,
                        then_block,
                        then_args,
                        else_block,
                        else_args,
                    })
                }
            }
            _ => None,
        };
        if let Some(term) = replacement {
            func.blocks[pred].term = Some(term);
            changed = true;
        }
    }
    changed
}

fn thread_edge(func: &Function, dom: &DomInfo, entry: Block, pred: Block, target: Block) -> Option<(Block, smallvec::SmallVec<[Value; 4]>)> {
    if target == pred || target == entry {
        return None;
    }
    let block = &func.blocks[target];
    if !block.insts.is_empty() || !block.params.is_empty() {
        return None;
    }
    let Some(Terminator::Br { dest, args }) = &block.term else {
        return None;
    };
    if *dest == target || *dest == pred {
        return None;
    }
    for &arg in args.iter() {
        if let Some(def_block) = value_block(func, arg)
            && !dom.dominates(def_block, pred, entry) {
                return None;
            }
    }
    Some((*dest, args.clone()))
}

/// Merge structurally identical empty terminators (`ret 1`, `br b3`,
/// `unreachable`, ...) so branch fans that only differ in which copy of
/// the same tail they enter share one block.
fn merge_identical_blocks(func: &mut Function) -> bool {
    let reachable = func.reachable_blocks();
    let mut keeper: Vec<Option<Block>> = vec![None; func.blocks.len()];
    for index in 0..func.blocks.len() {
        let block = Block::from_usize(index);
        if block == func.entry || !reachable[index] {
            continue;
        }
        let data = &func.blocks[block];
        if !data.params.is_empty() || !data.insts.is_empty() {
            continue;
        }
        let Some(term) = &data.term else {
            continue;
        };
        for other in 0..index {
            let candidate = Block::from_usize(other);
            if candidate == func.entry || keeper[other].is_some() || !reachable[other] {
                continue;
            }
            let data = &func.blocks[candidate];
            if !data.params.is_empty() || !data.insts.is_empty() {
                continue;
            }
            if data.term.as_ref().is_some_and(|other| same_terminator(term, other)) {
                keeper[index] = Some(candidate);
                break;
            }
        }
    }
    if keeper.iter().all(Option::is_none) {
        return false;
    }
    for index in 0..func.blocks.len() {
        let block = Block::from_usize(index);
        let Some(term) = func.blocks[block].term.take() else {
            continue;
        };
        func.blocks[block].term = Some(remap_successors(&term, |succ| keeper[succ.index()].unwrap_or(succ)));
    }
    true
}

fn same_terminator(a: &Terminator, b: &Terminator) -> bool {
    match (a, b) {
        (Terminator::Br { dest: a, args: aa }, Terminator::Br { dest: b, args: ba }) => a == b && aa == ba,
        (
            Terminator::CondBr {
                cond: ac,
                then_block: at,
                else_block: ae,
                then_args: ata,
                else_args: aea,
            },
            Terminator::CondBr {
                cond: bc,
                then_block: bt,
                else_block: be,
                then_args: bta,
                else_args: bea,
            },
        ) => ac == bc && at == bt && ae == be && ata == bta && aea == bea,
        (
            Terminator::Switch {
                scrutinee: as_,
                cases: ac,
                default: ad,
            },
            Terminator::Switch {
                scrutinee: bs,
                cases: bc,
                default: bd,
            },
        ) => as_ == bs && ac == bc && ad == bd,
        (Terminator::Return { val: a }, Terminator::Return { val: b }) => a == b,
        (Terminator::Unreachable, Terminator::Unreachable) => true,
        _ => false,
    }
}

fn remap_successors(term: &Terminator, remap: impl Fn(Block) -> Block) -> Terminator {
    match term {
        Terminator::Br { dest, args } => Terminator::Br {
            dest: remap(*dest),
            args: args.clone(),
        },
        Terminator::CondBr {
            cond,
            then_block,
            else_block,
            then_args,
            else_args,
        } => Terminator::CondBr {
            cond: *cond,
            then_block: remap(*then_block),
            else_block: remap(*else_block),
            then_args: then_args.clone(),
            else_args: else_args.clone(),
        },
        Terminator::Switch { scrutinee, cases, default } => Terminator::Switch {
            scrutinee: *scrutinee,
            cases: cases.iter().map(|(value, dest)| (*value, remap(*dest))).collect(),
            default: remap(*default),
        },
        Terminator::Return { val } => Terminator::Return { val: *val },
        Terminator::Unreachable => Terminator::Unreachable,
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
    // Each merge rewrites uses of the destination's params. Collect the
    // substitutions and sweep once: remapping per merge is O(function
    // size) and quadratic when many blocks collapse.
    let mut redirect: std::collections::HashMap<Value, Value> = std::collections::HashMap::new();
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
            redirect.extend(params.iter().copied().zip(args.iter().copied()));
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
            break;
        }
    }
    if !redirect.is_empty() {
        collapse_chains(&mut redirect);
        func.remap_operands(&redirect);
    }
    changed
}

/// If-convert one diamond per fork per round: a conditional edge into a
/// block that only computes pure values and then joins the other edge
/// becomes unconditional computation at the fork plus selects at the
/// join. Conversions in other forks do not invalidate the use map: they
/// only move or duplicate uses into their own fork, which can turn a
/// confined value into an escaping one but never the reverse.
fn if_convert_diamonds(func: &mut Function) -> bool {
    const MAX_SPECULATED_INSNS: usize = 8;
    let mut changed = false;
    let mut uses = SoleUse::compute(func);
    loop {
        let mut reachable = func.reachable_blocks();
        let mut preds = predecessor_counts(func);
        uses.recompute(func);
        let mut converted = false;
        for idx in 0..func.blocks.len() {
            let fork = Block::from_usize(idx);
            if !reachable[idx] {
                continue;
            }
            let Some(Terminator::CondBr {
                cond,
                then_block,
                else_block,
                then_args,
                else_args,
            }) = func.blocks[fork].term.clone()
            else {
                continue;
            };
            for (arm, incoming, other, other_args) in [
                (then_block, then_args.clone(), else_block, else_args.clone()),
                (else_block, else_args.clone(), then_block, then_args.clone()),
            ] {
                if arm == fork || arm == other || arm == func.entry || other == func.entry {
                    continue;
                }
                if !reachable[arm.index()] || preds[arm.index()] != 1 {
                    continue;
                }
                if func.blocks[arm].insts.len() > MAX_SPECULATED_INSNS {
                    continue;
                }
                let Some(Terminator::Br { dest, args: outgoing }) = func.blocks[arm].term.clone() else {
                    continue;
                };
                if dest == arm || dest == fork || dest == func.entry || func.blocks[dest].term.is_none() {
                    continue;
                }
                let params = func.blocks[arm].params.clone();
                if params.len() != incoming.len() {
                    continue;
                }
                if func.blocks[dest].params.len() != outgoing.len() {
                    continue;
                }
                if dest == other && func.blocks[dest].params.len() != other_args.len() {
                    continue;
                }
                if func.blocks[arm].insts.iter().any(|inst| !func.insts[*inst].kind.is_pure()) {
                    continue;
                }
                let local_values: Vec<Value> = params
                    .iter()
                    .copied()
                    .chain(func.blocks[arm].insts.iter().filter_map(|inst| func.insts[*inst].result))
                    .collect();
                let escapes_arm = local_values.iter().any(|value| !uses.confined_to(*value, arm));
                if escapes_arm {
                    continue;
                }
                if dest == other {
                    let differing = outgoing.iter().zip(&other_args).find(|(a, b)| a != b);
                    if let Some((arm_arg, other_arg)) = differing {
                        let ty = &func.values[*arm_arg].ty;
                        if !ty.is_scalar() || func.values[*other_arg].ty != *ty {
                            continue;
                        }
                    }
                }

                let mut map = std::collections::HashMap::new();
                for (param, arg) in params.iter().copied().zip(incoming.iter().copied()) {
                    map.insert(param, arg);
                }
                let insts = func.blocks[arm].insts.clone();
                let mut cloned = Vec::with_capacity(insts.len());
                for inst in insts {
                    let data = func.insts[inst].clone();
                    let result = data.result.map(|old| {
                        let fresh = func.values.push(ValueData {
                            ty: func.values[old].ty.clone(),
                            def: Def::Inst(Inst::from_usize(func.insts.len())),
                        });
                        map.insert(old, fresh);
                        fresh
                    });
                    let kind = remap_inst(&data.kind, |value| map.get(&value).copied().unwrap_or(value));
                    let fresh = func.insts.push(InstData {
                        block: fork,
                        kind,
                        result,
                        loc: data.loc,
                    });
                    if let Some(value) = result {
                        func.values[value].def = Def::Inst(fresh);
                    }
                    cloned.push(fresh);
                }
                func.blocks[fork].insts.extend(cloned);
                let outgoing: smallvec::SmallVec<[Value; 4]> = outgoing.iter().map(|value| map.get(value).copied().unwrap_or(*value)).collect();
                func.blocks[fork].term = Some(Terminator::CondBr {
                    cond,
                    then_block: if arm == then_block { dest } else { then_block },
                    else_block: if arm == else_block { dest } else { else_block },
                    then_args: if arm == then_block { outgoing.clone() } else { then_args.clone() },
                    else_args: if arm == else_block { outgoing } else { else_args.clone() },
                });
                func.blocks[arm].insts.clear();
                func.blocks[arm].params.clear();
                func.blocks[arm].term = Some(Terminator::Unreachable);
                reachable[arm.index()] = false;
                preds[arm.index()] -= 1;
                preds[dest.index()] += 1;
                changed = true;
                converted = true;
                break;
            }
        }
        if !converted {
            break;
        }
    }
    changed
}

/// For every value, the one block all of its uses live in — `None` when
/// they are spread over several blocks or it has no uses. Terminator
/// operands count as uses in the branching block, where they are
/// evaluated.
///
/// Dense tables keyed by `Value` index: hashing dominated this scan
/// when it was a `HashMap`, and `SimplifyCfg` runs it once per fixpoint
/// round.
struct SoleUse {
    seen: Vec<bool>,
    block: Vec<Option<Block>>,
    total: Vec<u32>,
}

impl SoleUse {
    fn compute(func: &Function) -> Self {
        let mut uses = SoleUse {
            seen: Vec::new(),
            block: Vec::new(),
            total: Vec::new(),
        };
        uses.recompute(func);
        uses
    }

    fn recompute(&mut self, func: &Function) {
        self.seen.clear();
        self.seen.resize(func.values.len(), false);
        self.block.clear();
        self.block.resize(func.values.len(), None);
        self.total.clear();
        self.total.resize(func.values.len(), 0);
        for (idx, data) in func.blocks.iter().enumerate() {
            let here = Block::from_usize(idx);
            for inst in &data.insts {
                func.insts[*inst].kind.for_each_operand(&mut |v| self.note(v, here));
            }
            if let Some(term) = &data.term {
                term.for_each_operand(&mut |v| self.note(v, here));
            }
        }
    }

    fn note(&mut self, value: Value, block: Block) {
        let index = value.index();
        self.total[index] += 1;
        if !self.seen[index] {
            self.seen[index] = true;
            self.block[index] = Some(block);
        } else if self.block[index] != Some(block) {
            self.block[index] = None;
        }
    }

    /// Move every use made by `kind` from `from` to `to`. Operands are
    /// counted first so a duplicated operand (e.g. `add x, x`) moves
    /// atomically; relocating one occurrence at a time would look like a
    /// spread and lose the value's single user block.
    fn relocate_operands(&mut self, kind: &InstKind, from: Block, to: Block) {
        let mut pending: smallvec::SmallVec<[(Value, u32); 4]> = smallvec::SmallVec::new();
        kind.for_each_operand(&mut |value| {
            if let Some(entry) = pending.iter_mut().find(|(v, _)| *v == value) {
                entry.1 += 1;
            } else {
                pending.push((value, 1));
            }
        });
        for (value, count) in pending {
            self.relocate(value, from, to, count);
        }
    }

    /// Move one use of `value` from `from` to `to`.
    fn relocate(&mut self, value: Value, from: Block, to: Block, count: u32) {
        let index = value.index();
        debug_assert!(self.seen[index], "relocating a use that was never noted");
        self.total[index] -= count;
        if self.block[index] == Some(from) {
            // All uses were here, so the rest are now in `to` only when
            // these were the last ones.
            self.block[index] = if self.total[index] == 0 { Some(to) } else { None };
        }
    }

    /// The one block all uses live in; `None` when unused or spread.
    fn sole_block(&self, value: Value) -> Option<Block> {
        let index = value.index();
        if self.seen[index] { self.block[index] } else { None }
    }

    /// Whether every use is in `block` (also true when there are none).
    fn confined_to(&self, value: Value, block: Block) -> bool {
        let index = value.index();
        !self.seen[index] || self.block[index] == Some(block)
    }
}

/// Move pure instructions into the successor that holds every use: a
/// computation only one arm of a branch needs should not run on the
/// other. Only a target whose sole predecessor is the current block
/// qualifies, so the instruction cannot start running more often than
/// before (a loop body reached from several blocks is off limits).
fn sink_pure_instructions(func: &mut Function) -> bool {
    let mut changed = false;
    // Sinking moves instructions but never rewires edges, so the
    // predecessor counts stay valid across rounds. Uses move with their
    // instructions, so `relocate` keeps the map current instead of
    // rehashing every operand each round.
    let preds = predecessor_counts(func);
    let mut uses = SoleUse::compute(func);
    loop {
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
                let dest = func.insts[inst].result.and_then(|result| uses.sole_block(result)).filter(|dest| sinks.contains(dest));
                match dest {
                    Some(dest) if func.insts[inst].kind.is_pure() => {
                        func.insts[inst].block = dest;
                        uses.relocate_operands(&func.insts[inst].kind, block, dest);
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
/// - Heuristic: small callees up to `max_insts` inline whether or not
///   they are marked `inline`. Calls inside a natural loop get the
///   larger `max_hot_insts` budget, and a callee with a single call
///   site gets `max_single_insts`: neither duplicates code, so the
///   usual size limit would under-inline them.
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
    pub max_hot_insts: usize,
    pub max_single_insts: usize,
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
            max_insts: 64,
            max_hot_insts: 96,
            max_single_insts: 192,
            max_inline_insts: 256,
        }
    }

    /// Keyword-only inlining: no heuristic budget, so plain calls
    /// never inline while `inline`-marked callees still do. The
    /// driver uses this at `-O0`.
    pub fn inline_only() -> Self {
        Inline {
            max_insts: 0,
            max_hot_insts: 0,
            max_single_insts: 0,
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
        let mut call_counts = vec![0usize; module.funcs.len()];
        for func in module.funcs.iter() {
            for block in func.blocks.iter() {
                for inst in &block.insts {
                    if let InstKind::Call { callee, .. } = &func.insts[*inst].kind {
                        call_counts[callee.index()] += 1;
                    }
                }
            }
        }
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
            if sites.is_empty() {
                continue;
            }
            let reachable = module.funcs[caller_id].reachable_blocks();
            let dom = DomInfo::compute(&module.funcs[caller_id], &reachable);
            let hot_blocks: std::collections::HashSet<Block> = licm::natural_loops(&module.funcs[caller_id], &dom).into_iter().flat_map(|(_, blocks)| blocks).collect();
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
                let limit = if marked {
                    self.max_inline_insts
                } else if call_counts[callee.index()] == 1 {
                    self.max_single_insts
                } else if hot_blocks.contains(&block) {
                    self.max_hot_insts
                } else {
                    self.max_insts
                };
                if total > limit {
                    continue;
                }
                let single_block = module.funcs[callee].blocks.len() == 1;
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
    fn int_const_folds_address_arithmetic() {
        let (mut func, entry) = Function::new("address", Signature::new(vec![], Ty::int(64)));
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let widened = func.values.push(ValueData {
            ty: Ty::int(64),
            def: Def::Inst(Inst::from_usize(0)),
        });
        let sext = func.insts.push(InstData {
            block: entry,
            kind: InstKind::Sext(one),
            result: Some(widened),
            loc: None,
        });
        func.values[widened].def = Def::Inst(sext);
        let four = func.push_const(Const::Int { bits: 64, value: 4 });
        let scaled = func.values.push(ValueData {
            ty: Ty::int(64),
            def: Def::Inst(Inst::from_usize(0)),
        });
        let mul = func.insts.push(InstData {
            block: entry,
            kind: InstKind::Mul(widened, four),
            result: Some(scaled),
            loc: None,
        });
        func.values[scaled].def = Def::Inst(mul);
        assert_eq!(int_const(&func, scaled), Some((64, 4)));
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

    fn aggregate_copy_func(volatile: bool) -> (Function, crate::ir::Inst, crate::ir::Inst) {
        let ty = Ty::Struct {
            fields: vec![Ty::int(64), Ty::int(64)],
            offsets: vec![0, 8],
            size: 16,
        };
        let sig = Signature::new(vec![Ty::ptr(ty.clone()), Ty::ptr(ty.clone())], Ty::Void);
        let (mut func, entry) = Function::new("copy", sig);
        let dst = func.values.push(crate::ir::ValueData {
            ty: Ty::ptr(ty.clone()),
            def: Def::Param(entry, 0),
        });
        let src = func.values.push(crate::ir::ValueData {
            ty: Ty::ptr(ty.clone()),
            def: Def::Param(entry, 1),
        });
        func.blocks[entry].params.extend([dst, src]);
        let flags = if volatile { crate::ir::MemFlags::vol() } else { crate::ir::MemFlags::plain() };
        let result = func.values.push(crate::ir::ValueData {
            ty: ty.clone(),
            def: Def::Inst(Inst::from_usize(0)),
        });
        let load = func.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Load { addr: src, flags },
            result: Some(result),
            loc: None,
        });
        func.values[result].def = Def::Inst(load);
        func.blocks[entry].insts.push(load);
        let store = func.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Store { addr: dst, val: result, flags },
            result: None,
            loc: None,
        });
        func.blocks[entry].insts.push(store);
        func.blocks[entry].term = Some(Terminator::Return { val: None });
        (func, load, store)
    }

    #[test]
    fn forwards_adjacent_aggregate_load_store() {
        let (mut func, load, store) = aggregate_copy_func(false);
        let mut pass = ForwardAggregateCopy;
        assert!(pass.run(&mut func));
        assert!(matches!(func.insts[load].kind, InstKind::Memcpy { len: 16, .. }));
        assert_eq!(func.insts[load].result, None);
        assert!(!func.blocks[func.entry].insts.contains(&store));
        assert!(crate::verify::verify_func(&func).is_empty());
        assert!(!pass.run(&mut func), "second run is a no-op");
    }

    #[test]
    fn expands_local_scalar_memcpy_for_sroa() {
        let sig = Signature::new(vec![Ty::ptr(Ty::int(8))], Ty::int(64));
        let (mut func, entry) = Function::new("scalar-copy", sig);
        let src = entry_param(&mut func, entry, Ty::ptr(Ty::int(8)), 0);
        let slot = stack_slot(&mut func, entry, Ty::int(64));
        let void_ptr = push_value(&mut func, entry, InstKind::Bitcast(slot), Ty::ptr(Ty::Void));
        let copy = push_unit(&mut func, entry, InstKind::Memcpy { dst: void_ptr, src, len: 8 });
        let loaded = push_scalar_load(&mut func, entry, slot, Ty::int(64));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        let mut forward = ForwardAggregateCopy;
        assert!(forward.run(&mut func));
        assert!(!func.blocks[entry].insts.contains(&copy));
        assert!(!resident_kinds(&func).iter().any(|kind| matches!(kind, InstKind::Memcpy { .. })));

        let mut dce = DeadCodeElim;
        dce.run(&mut func);
        let mut sroa = Sroa;
        let mut mem2reg = Mem2Reg;
        sroa.run(&mut func);
        mem2reg.run(&mut func);
        assert!(!resident_kinds(&func).iter().any(|kind| matches!(kind, InstKind::Alloca { .. } | InstKind::Store { .. })));
        assert!(crate::verify::verify_func(&func).is_empty());
    }

    #[test]
    fn expands_local_aggregate_copies_for_sroa() {
        let ty = Ty::Struct {
            fields: vec![Ty::int(32), Ty::int(32)],
            offsets: vec![0, 4],
            size: 8,
        };
        let (mut func, entry) = Function::new("copy", Signature::new(vec![], Ty::int(32)));
        let dst = stack_slot(&mut func, entry, ty.clone());
        let src = stack_slot(&mut func, entry, ty.clone());
        let copy = push_unit(&mut func, entry, InstKind::Memcpy { dst, src, len: 8 });
        let field = push_value(
            &mut func,
            entry,
            InstKind::GetElementPtr {
                base: dst,
                indices: [crate::ty::GepIndex::Field(0)].into_iter().collect(),
            },
            Ty::ptr(Ty::int(32)),
        );
        let loaded = push_scalar_load(&mut func, entry, field, Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        let mut forward = ForwardAggregateCopy;
        assert!(forward.run(&mut func));
        assert!(!func.blocks[entry].insts.contains(&copy));
        assert!(!func.blocks[entry].insts.iter().any(|inst| matches!(func.insts[*inst].kind, InstKind::Memcpy { .. })));

        let mut sroa = Sroa;
        let mut mem2reg = Mem2Reg;
        assert!(sroa.run(&mut func));
        assert!(mem2reg.run(&mut func));
        assert!(!has_memory_ops(&func));
        assert!(crate::verify::verify_func(&func).is_empty());
    }

    #[test]
    fn expands_padded_aggregate_copies_for_sroa() {
        let ty = Ty::Struct {
            fields: vec![Ty::int(32), Ty::int(32)],
            offsets: vec![0, 8],
            size: 16,
        };
        let (mut func, entry) = Function::new("padded", Signature::new(vec![], Ty::int(32)));
        let dst = stack_slot(&mut func, entry, ty.clone());
        let src = stack_slot(&mut func, entry, ty);
        let copy = push_unit(&mut func, entry, InstKind::Memcpy { dst, src, len: 16 });
        let field = push_value(
            &mut func,
            entry,
            InstKind::GetElementPtr {
                base: dst,
                indices: [crate::ty::GepIndex::Field(1)].into_iter().collect(),
            },
            Ty::ptr(Ty::int(32)),
        );
        let loaded = push_scalar_load(&mut func, entry, field, Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        let mut forward = ForwardAggregateCopy;
        assert!(forward.run(&mut func));
        assert!(!func.blocks[entry].insts.contains(&copy));

        let mut sroa = Sroa;
        let mut mem2reg = Mem2Reg;
        assert!(sroa.run(&mut func));
        assert!(mem2reg.run(&mut func));
        assert!(!has_memory_ops(&func));
        assert!(crate::verify::verify_func(&func).is_empty());
    }

    #[test]
    fn expands_aggregate_parameter_copy_and_return() {
        let ty = Ty::Struct {
            fields: vec![Ty::int(32), Ty::int(32)],
            offsets: vec![0, 4],
            size: 8,
        };
        let (mut func, entry) = Function::new("copy_param", Signature::new(vec![ty.clone()], ty.clone()));
        let input = func.values.push(ValueData {
            ty: ty.clone(),
            def: Def::Param(entry, 0),
        });
        func.blocks[entry].params.push(input);
        let slot = stack_slot(&mut func, entry, ty.clone());
        push_unit(
            &mut func,
            entry,
            InstKind::Store {
                addr: slot,
                val: input,
                flags: MemFlags::plain(),
            },
        );
        let loaded = push_value(
            &mut func,
            entry,
            InstKind::Load {
                addr: slot,
                flags: MemFlags::plain(),
            },
            ty,
        );
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        let mut forward = ForwardAggregateCopy;
        assert!(forward.run(&mut func));
        assert!(!resident_kinds(&func).iter().any(|kind| matches!(kind, InstKind::Memcpy { .. })));

        let mut sroa = Sroa;
        let mut mem2reg = Mem2Reg;
        assert!(sroa.run(&mut func));
        assert!(mem2reg.run(&mut func));
        assert!(!has_memory_ops(&func));
        assert!(crate::verify::verify_func(&func).is_empty());
    }

    #[test]
    fn expands_aggregate_load_consumed_by_field_extracts() {
        let ty = Ty::Struct {
            fields: vec![Ty::int(32), Ty::int(32)],
            offsets: vec![0, 4],
            size: 8,
        };
        let (mut func, entry) = Function::new("load_fields", Signature::new(vec![], Ty::int(32)));
        let slot = stack_slot(&mut func, entry, ty.clone());
        let aggregate = push_value(
            &mut func,
            entry,
            InstKind::Load {
                addr: slot,
                flags: MemFlags::plain(),
            },
            ty,
        );
        let field = push_value(&mut func, entry, InstKind::ExtractValue { agg: aggregate, index: 1 }, Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(field) });

        let mut forward = ForwardAggregateCopy;
        assert!(forward.run(&mut func));
        assert!(!resident_kinds(&func).iter().any(|kind| matches!(kind, InstKind::ExtractValue { .. })));

        let mut sroa = Sroa;
        let mut mem2reg = Mem2Reg;
        assert!(sroa.run(&mut func));
        assert!(mem2reg.run(&mut func));
        assert!(!has_memory_ops(&func));
        assert!(crate::verify::verify_func(&func).is_empty());
    }

    #[test]
    fn expands_local_zero_memsets_for_sroa() {
        let ty = Ty::Struct {
            fields: vec![Ty::int(32), Ty::int(32)],
            offsets: vec![0, 4],
            size: 8,
        };
        let (mut func, entry) = Function::new("zero", Signature::new(vec![], Ty::int(32)));
        let slot = stack_slot(&mut func, entry, ty.clone());
        let memset = push_unit(&mut func, entry, InstKind::Memset { dst: slot, byte: 0, len: 8 });
        let field = push_value(
            &mut func,
            entry,
            InstKind::GetElementPtr {
                base: slot,
                indices: [crate::ty::GepIndex::Field(1)].into_iter().collect(),
            },
            Ty::ptr(Ty::int(32)),
        );
        let loaded = push_scalar_load(&mut func, entry, field, Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        let mut forward = ForwardAggregateCopy;
        assert!(forward.run(&mut func));
        assert!(!func.blocks[entry].insts.contains(&memset));
        assert!(!resident_kinds(&func).iter().any(|kind| matches!(kind, InstKind::Memset { .. })));

        let mut sroa = Sroa;
        let mut mem2reg = Mem2Reg;
        assert!(sroa.run(&mut func));
        assert!(mem2reg.run(&mut func));
        assert!(!has_memory_ops(&func));
        let Some(Terminator::Return { val: Some(value) }) = func.blocks[entry].term else {
            panic!("entry still returns the field");
        };
        assert_eq!(func.const_value(value), Some(&Const::Int { bits: 32, value: 0 }));
        assert!(crate::verify::verify_func(&func).is_empty());
    }

    #[test]
    fn keeps_nonzero_and_escaping_memsets() {
        let ty = Ty::Struct {
            fields: vec![Ty::int(32), Ty::int(32)],
            offsets: vec![0, 4],
            size: 8,
        };
        let (mut func, entry) = Function::new("nonzero", Signature::new(vec![], Ty::Void));
        let slot = stack_slot(&mut func, entry, ty.clone());
        let memset = push_unit(&mut func, entry, InstKind::Memset { dst: slot, byte: 0xFF, len: 8 });
        func.blocks[entry].term = Some(Terminator::Return { val: None });
        let mut forward = ForwardAggregateCopy;
        assert!(!forward.run(&mut func));
        assert!(func.blocks[entry].insts.contains(&memset));

        let (mut func, entry) = Function::new("escape", Signature::new(vec![], Ty::ptr(ty.clone())));
        let slot = stack_slot(&mut func, entry, ty);
        let memset = push_unit(&mut func, entry, InstKind::Memset { dst: slot, byte: 0, len: 8 });
        func.blocks[entry].term = Some(Terminator::Return { val: Some(slot) });
        assert!(!forward.run(&mut func));
        assert!(func.blocks[entry].insts.contains(&memset));
    }

    #[test]
    fn keeps_aggregate_copies_when_a_slot_escapes() {
        let ty = Ty::Struct {
            fields: vec![Ty::int(32), Ty::int(32)],
            offsets: vec![0, 4],
            size: 8,
        };
        let (mut func, entry) = Function::new("escape", Signature::new(vec![], Ty::ptr(ty.clone())));
        let dst = stack_slot(&mut func, entry, ty.clone());
        let src = stack_slot(&mut func, entry, ty);
        let copy = push_unit(&mut func, entry, InstKind::Memcpy { dst, src, len: 8 });
        func.blocks[entry].term = Some(Terminator::Return { val: Some(dst) });

        let mut forward = ForwardAggregateCopy;
        assert!(!forward.run(&mut func));
        assert!(func.blocks[entry].insts.contains(&copy));
        assert!(crate::verify::verify_func(&func).is_empty());
    }

    #[test]
    fn keeps_volatile_aggregate_copies() {
        let (mut func, load, _) = aggregate_copy_func(true);
        let mut pass = ForwardAggregateCopy;
        assert!(!pass.run(&mut func));
        assert!(matches!(func.insts[load].kind, InstKind::Load { .. }));
    }

    fn stack_slot(func: &mut Function, block: Block, ty: Ty) -> Value {
        let mut builder = FunctionBuilder::new(func, block);
        builder.alloca(ty, 1)
    }

    fn push_unit(func: &mut Function, block: Block, kind: InstKind) -> Inst {
        let inst = func.insts.push(crate::ir::InstData {
            block,
            kind,
            result: None,
            loc: None,
        });
        func.blocks[block].insts.push(inst);
        inst
    }

    fn push_scalar_load(func: &mut Function, block: Block, addr: Value, ty: Ty) -> Value {
        let mut builder = FunctionBuilder::new(func, block);
        builder.load(addr, ty)
    }

    fn entry_param(func: &mut Function, block: Block, ty: Ty, index: u32) -> Value {
        let value = func.values.push(crate::ir::ValueData {
            ty,
            def: Def::Param(block, index),
        });
        func.blocks[block].params.push(value);
        value
    }

    fn push_value(func: &mut Function, block: Block, kind: InstKind, ty: Ty) -> Value {
        let result = func.values.push(crate::ir::ValueData {
            ty,
            def: Def::Inst(Inst::from_usize(0)),
        });
        let inst = func.insts.push(crate::ir::InstData {
            block,
            kind,
            result: Some(result),
            loc: None,
        });
        func.values[result].def = Def::Inst(inst);
        func.blocks[block].insts.push(inst);
        result
    }

    #[test]
    fn removes_store_overwritten_in_the_same_block() {
        let (mut func, entry) = Function::new("dse", Signature::new(vec![], Ty::Void));
        let slot = stack_slot(&mut func, entry, Ty::int(32));
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let two = func.push_const(Const::Int { bits: 32, value: 2 });
        let first = push_unit(
            &mut func,
            entry,
            InstKind::Store {
                addr: slot,
                val: one,
                flags: crate::ir::MemFlags::plain(),
            },
        );
        let second = push_unit(
            &mut func,
            entry,
            InstKind::Store {
                addr: slot,
                val: two,
                flags: crate::ir::MemFlags::plain(),
            },
        );
        func.blocks[entry].term = Some(Terminator::Return { val: None });

        let mut pass = DeadStoreElim;
        assert!(pass.run(&mut func));
        assert!(!func.blocks[entry].insts.contains(&first));
        assert!(func.blocks[entry].insts.contains(&second));
        assert!(crate::verify::verify_func(&func).is_empty());
    }

    #[test]
    fn removes_store_overwritten_across_a_neutral_extern() {
        let (mut func, entry) = Function::new("dse", Signature::new(vec![], Ty::Void));
        func.set_extern_effects(&[crate::ir::ExternEffect::neutral()]);
        let slot = stack_slot(&mut func, entry, Ty::int(32));
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let two = func.push_const(Const::Int { bits: 32, value: 2 });
        let first = push_unit(
            &mut func,
            entry,
            InstKind::Store {
                addr: slot,
                val: one,
                flags: crate::ir::MemFlags::plain(),
            },
        );
        push_unit(
            &mut func,
            entry,
            InstKind::CallExtern {
                target: crate::ir::Extern::from_usize(0),
                args: Default::default(),
            },
        );
        let second = push_unit(
            &mut func,
            entry,
            InstKind::Store {
                addr: slot,
                val: two,
                flags: crate::ir::MemFlags::plain(),
            },
        );
        func.blocks[entry].term = Some(Terminator::Return { val: None });

        let mut pass = DeadStoreElim;
        assert!(pass.run(&mut func));
        assert!(!func.blocks[entry].insts.contains(&first));
        assert!(func.blocks[entry].insts.contains(&second));
    }

    #[test]
    fn removes_store_overwritten_before_a_later_load() {
        let (mut func, entry) = Function::new("dse", Signature::new(vec![], Ty::Void));
        let slot = stack_slot(&mut func, entry, Ty::int(32));
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let two = func.push_const(Const::Int { bits: 32, value: 2 });
        let first = push_unit(
            &mut func,
            entry,
            InstKind::Store {
                addr: slot,
                val: one,
                flags: crate::ir::MemFlags::plain(),
            },
        );
        let second = push_unit(
            &mut func,
            entry,
            InstKind::Store {
                addr: slot,
                val: two,
                flags: crate::ir::MemFlags::plain(),
            },
        );
        let loaded = push_scalar_load(&mut func, entry, slot, Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        let mut pass = DeadStoreElim;
        assert!(pass.run(&mut func));
        assert!(!func.blocks[entry].insts.contains(&first));
        assert!(func.blocks[entry].insts.contains(&second));
    }

    #[test]
    fn keeps_store_read_before_the_overwrite() {
        let (mut func, entry) = Function::new("dse", Signature::new(vec![], Ty::Void));
        let slot = stack_slot(&mut func, entry, Ty::int(32));
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let two = func.push_const(Const::Int { bits: 32, value: 2 });
        let first = push_unit(
            &mut func,
            entry,
            InstKind::Store {
                addr: slot,
                val: one,
                flags: crate::ir::MemFlags::plain(),
            },
        );
        let loaded = push_scalar_load(&mut func, entry, slot, Ty::int(32));
        let second = push_unit(
            &mut func,
            entry,
            InstKind::Store {
                addr: slot,
                val: two,
                flags: crate::ir::MemFlags::plain(),
            },
        );
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        let mut pass = DeadStoreElim;
        assert!(!pass.run(&mut func));
        assert!(func.blocks[entry].insts.contains(&first));
        assert!(func.blocks[entry].insts.contains(&second));
    }

    #[test]
    fn removes_overwritten_copy_from_a_different_object() {
        let ty = Ty::Struct {
            fields: vec![Ty::int(64), Ty::int(64)],
            offsets: vec![0, 8],
            size: 16,
        };
        let (mut func, entry) = Function::new("dse", Signature::new(vec![], Ty::Void));
        let dst = stack_slot(&mut func, entry, ty.clone());
        let first_src = stack_slot(&mut func, entry, ty.clone());
        let second_src = stack_slot(&mut func, entry, ty.clone());
        let first = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: first_src, len: 16 });
        let second = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: second_src, len: 16 });
        func.blocks[entry].term = Some(Terminator::Return { val: None });

        let mut pass = DeadStoreElim;
        assert!(pass.run(&mut func));
        assert!(!func.blocks[entry].insts.contains(&first));
        assert!(func.blocks[entry].insts.contains(&second));
        assert!(crate::verify::verify_func(&func).is_empty());
    }

    #[test]
    fn removes_copy_from_unknown_source_for_local_destination() {
        let ty = Ty::Struct {
            fields: vec![Ty::int(64), Ty::int(64)],
            offsets: vec![0, 8],
            size: 16,
        };
        let sig = Signature::new(vec![Ty::ptr(ty.clone()), Ty::ptr(ty.clone())], Ty::Void);
        let (mut func, entry) = Function::new("dse", sig);
        let first_src = entry_param(&mut func, entry, Ty::ptr(ty.clone()), 0);
        let second_src = entry_param(&mut func, entry, Ty::ptr(ty.clone()), 1);
        let dst = stack_slot(&mut func, entry, ty);
        let first = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: first_src, len: 16 });
        let second = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: second_src, len: 16 });
        func.blocks[entry].term = Some(Terminator::Return { val: None });

        let mut pass = DeadStoreElim;
        assert!(pass.run(&mut func));
        assert!(!func.blocks[entry].insts.contains(&first));
        assert!(func.blocks[entry].insts.contains(&second));
    }

    #[test]
    fn removes_copy_overwritten_before_field_load() {
        let ty = Ty::Struct {
            fields: vec![Ty::Array {
                elem: Box::new(Ty::int(64)),
                len: 4,
            }],
            offsets: vec![0],
            size: 32,
        };
        let sig = Signature::new(vec![Ty::ptr(ty.clone()), Ty::ptr(ty.clone())], Ty::int(64));
        let (mut func, entry) = Function::new("dse", sig);
        let first_src = entry_param(&mut func, entry, Ty::ptr(ty.clone()), 0);
        let second_src = entry_param(&mut func, entry, Ty::ptr(ty.clone()), 1);
        let dst = stack_slot(&mut func, entry, ty);
        let first = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: first_src, len: 32 });
        let second = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: second_src, len: 32 });
        let field = push_value(
            &mut func,
            entry,
            InstKind::GetElementPtr {
                base: dst,
                indices: vec![crate::ty::GepIndex::Field(0), crate::ty::GepIndex::ConstIndex(0)].into_iter().collect(),
            },
            Ty::ptr(Ty::int(64)),
        );
        let as_int = push_value(&mut func, entry, InstKind::PtrToInt(field), Ty::int(64));
        let as_ptr = push_value(&mut func, entry, InstKind::IntToPtr(as_int), Ty::ptr(Ty::int(64)));
        let loaded = push_scalar_load(&mut func, entry, as_ptr, Ty::int(64));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        let mut pass = DeadStoreElim;
        assert!(pass.run(&mut func));
        assert!(!func.blocks[entry].insts.contains(&first));
        assert!(func.blocks[entry].insts.contains(&second));
    }

    #[test]
    fn keeps_copy_when_address_is_stored_as_integer() {
        let ty = Ty::Struct {
            fields: vec![Ty::int(64), Ty::int(64)],
            offsets: vec![0, 8],
            size: 16,
        };
        let sig = Signature::new(vec![Ty::ptr(Ty::int(64)), Ty::ptr(ty.clone()), Ty::ptr(ty.clone())], Ty::Void);
        let (mut func, entry) = Function::new("dse", sig);
        let sink = entry_param(&mut func, entry, Ty::ptr(Ty::int(64)), 0);
        let first_src = entry_param(&mut func, entry, Ty::ptr(ty.clone()), 1);
        let second_src = entry_param(&mut func, entry, Ty::ptr(ty.clone()), 2);
        let dst = stack_slot(&mut func, entry, ty);
        let address = push_value(&mut func, entry, InstKind::PtrToInt(dst), Ty::int(64));
        push_unit(
            &mut func,
            entry,
            InstKind::Store {
                addr: sink,
                val: address,
                flags: crate::ir::MemFlags::plain(),
            },
        );
        let first = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: first_src, len: 16 });
        let second = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: second_src, len: 16 });
        func.blocks[entry].term = Some(Terminator::Return { val: None });

        let mut pass = DeadStoreElim;
        assert!(!pass.run(&mut func));
        assert!(func.blocks[entry].insts.contains(&first));
        assert!(func.blocks[entry].insts.contains(&second));
    }

    #[test]
    fn keeps_copy_when_local_address_escapes() {
        let ty = Ty::Struct {
            fields: vec![Ty::int(64), Ty::int(64)],
            offsets: vec![0, 8],
            size: 16,
        };
        let sig = Signature::new(vec![Ty::ptr(Ty::ptr(ty.clone())), Ty::ptr(ty.clone()), Ty::ptr(ty.clone())], Ty::Void);
        let (mut func, entry) = Function::new("dse", sig);
        let sink = entry_param(&mut func, entry, Ty::ptr(Ty::ptr(ty.clone())), 0);
        let first_src = entry_param(&mut func, entry, Ty::ptr(ty.clone()), 1);
        let second_src = entry_param(&mut func, entry, Ty::ptr(ty.clone()), 2);
        let dst = stack_slot(&mut func, entry, ty);
        push_unit(
            &mut func,
            entry,
            InstKind::Store {
                addr: sink,
                val: dst,
                flags: crate::ir::MemFlags::plain(),
            },
        );
        let first = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: first_src, len: 16 });
        let second = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: second_src, len: 16 });
        func.blocks[entry].term = Some(Terminator::Return { val: None });

        let mut pass = DeadStoreElim;
        assert!(!pass.run(&mut func));
        assert!(func.blocks[entry].insts.contains(&first));
        assert!(func.blocks[entry].insts.contains(&second));
    }

    #[test]
    fn keeps_copy_read_before_the_overwrite() {
        let ty = Ty::Struct {
            fields: vec![Ty::int(64), Ty::int(64)],
            offsets: vec![0, 8],
            size: 16,
        };
        let (mut func, entry) = Function::new("dse", Signature::new(vec![], Ty::Void));
        let dst = stack_slot(&mut func, entry, ty.clone());
        let first_src = stack_slot(&mut func, entry, ty.clone());
        let second_src = stack_slot(&mut func, entry, ty.clone());
        let first = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: first_src, len: 16 });
        let loaded = push_scalar_load(&mut func, entry, dst, Ty::int(64));
        let second = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: second_src, len: 16 });
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        let mut pass = DeadStoreElim;
        assert!(!pass.run(&mut func));
        assert!(func.blocks[entry].insts.contains(&first));
        assert!(func.blocks[entry].insts.contains(&second));
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
    fn folds_identical_branch_arms() {
        let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("same-arm", sig);
        let cond = param(&mut func, entry, Ty::Bool);
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let shared = func.push_block(vec![Ty::int(32)]);
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block: shared,
            else_block: shared,
            then_args: [one].into_iter().collect(),
            else_args: [one].into_iter().collect(),
        });
        let p = func.blocks[shared].params[0];
        func.blocks[shared].term = Some(Terminator::Return { val: Some(p) });

        let mut simp = SimplifyCfg;
        assert!(simp.run(&mut func));
        assert!(crate::verify::verify_func(&func).is_empty());
        let Terminator::Return { val: Some(v) } = func.blocks[entry].term.clone().unwrap() else {
            panic!("branch was not folded");
        };
        assert_eq!(func.const_value(v), Some(&Const::Int { bits: 32, value: 1 }));
    }

    #[test]
    fn folds_fork_to_select() {
        let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("fork", sig);
        let cond = param(&mut func, entry, Ty::Bool);
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let two = func.push_const(Const::Int { bits: 32, value: 2 });
        let shared = func.push_block(vec![Ty::int(32)]);
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block: shared,
            else_block: shared,
            then_args: [one].into_iter().collect(),
            else_args: [two].into_iter().collect(),
        });
        let p = func.blocks[shared].params[0];
        func.blocks[shared].term = Some(Terminator::Return { val: Some(p) });

        let mut simp = SimplifyCfg;
        assert!(simp.run(&mut func));
        assert!(crate::verify::verify_func(&func).is_empty());
        let selects: Vec<InstKind> = func.blocks[entry]
            .insts
            .iter()
            .map(|i| func.insts[*i].kind.clone())
            .filter(|k| matches!(k, InstKind::Select { .. }))
            .collect();
        assert_eq!(selects.len(), 1, "fork becomes one select: {selects:?}");
        assert!(matches!(func.blocks[entry].term, Some(Terminator::Return { val: Some(_) })));
    }

    #[test]
    fn threads_empty_trampoline() {
        let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("thread", sig);
        let cond = param(&mut func, entry, Ty::Bool);
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let two = func.push_const(Const::Int { bits: 32, value: 2 });
        let left = func.push_block(vec![]);
        let right = func.push_block(vec![]);
        let join = func.push_block(vec![Ty::int(32)]);
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block: left,
            else_block: right,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        func.blocks[left].term = Some(Terminator::Br {
            dest: join,
            args: [one].into_iter().collect(),
        });
        func.blocks[right].term = Some(Terminator::Br {
            dest: join,
            args: [two].into_iter().collect(),
        });
        let p = func.blocks[join].params[0];
        func.blocks[join].term = Some(Terminator::Return { val: Some(p) });

        let mut simp = SimplifyCfg;
        assert!(simp.run(&mut func));
        assert!(crate::verify::verify_func(&func).is_empty());
        assert!(func.blocks[left].insts.is_empty());
        assert!(func.blocks[left].params.is_empty());
        assert!(matches!(func.blocks[left].term, Some(Terminator::Unreachable)));
        assert!(matches!(func.blocks[right].term, Some(Terminator::Unreachable)));
        assert!(func.blocks[entry].insts.iter().any(|i| matches!(func.insts[*i].kind, InstKind::Select { .. })));
    }

    #[test]
    fn merges_identical_blocks() {
        let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("identical", sig);
        let cond = param(&mut func, entry, Ty::Bool);
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let left = func.push_block(vec![]);
        let right = func.push_block(vec![]);
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block: left,
            else_block: right,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        func.blocks[left].term = Some(Terminator::Return { val: Some(one) });
        func.blocks[right].term = Some(Terminator::Return { val: Some(one) });

        let mut simp = SimplifyCfg;
        assert!(simp.run(&mut func));
        assert!(crate::verify::verify_func(&func).is_empty());
        assert!(matches!(func.blocks[right].term, Some(Terminator::Unreachable)));
        let Terminator::Return { val: Some(v) } = func.blocks[entry].term.clone().unwrap() else {
            panic!("identical arms did not fold");
        };
        assert_eq!(v, one);
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
    fn collapses_empty_diamond() {
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
        assert!(simp.run(&mut func));
        assert!(crate::verify::verify_func(&func).is_empty());
        assert!(matches!(func.blocks[entry].term, Some(Terminator::Return { val: None })));
        assert!(matches!(func.blocks[then_block].term, Some(Terminator::Unreachable)));
        assert!(matches!(func.blocks[else_block].term, Some(Terminator::Unreachable)));
        assert!(matches!(func.blocks[merge].term, Some(Terminator::Unreachable)));
    }

    #[test]
    fn if_converts_pure_diamond() {
        let sig = Signature::new(vec![Ty::Bool, Ty::int(32), Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("diamond", sig);
        let cond = param(&mut func, entry, Ty::Bool);
        let x = param(&mut func, entry, Ty::int(32));
        let y = param(&mut func, entry, Ty::int(32));
        let arm = func.push_block(vec![]);
        let merge = func.push_block(vec![Ty::int(32)]);
        let result = func.blocks[merge].params[0];
        let sum = emit_bin(&mut func, arm, InstKind::Add(x, y), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block: arm,
            else_block: merge,
            then_args: Default::default(),
            else_args: [x].into_iter().collect(),
        });
        func.blocks[arm].term = Some(Terminator::Br {
            dest: merge,
            args: [sum].into_iter().collect(),
        });
        func.blocks[merge].term = Some(Terminator::Return { val: Some(result) });

        let mut simp = SimplifyCfg;
        assert!(simp.run(&mut func));
        assert!(crate::verify::verify_func(&func).is_empty());
        assert!(matches!(func.blocks[arm].term, Some(Terminator::Unreachable)));
        let select = func.blocks[entry]
            .insts
            .iter()
            .find_map(|inst| match &func.insts[*inst].kind {
                InstKind::Select { cond: c, then_val, else_val } if *c == cond && *else_val == x => Some((*inst, *then_val)),
                _ => None,
            })
            .expect("diamond arm became a select");
        let Def::Inst(cloned_sum) = func.values[select.1].def else {
            panic!("select arm was not a computed value");
        };
        assert!(matches!(
            func.insts[cloned_sum].kind,
            InstKind::Add(a, b) if a == x && b == y
        ));

        assert!(simp.run(&mut func));
        assert!(crate::verify::verify_func(&func).is_empty());
        let Terminator::Return { val: Some(value) } = func.blocks[entry].term.clone().unwrap() else {
            panic!("join did not merge into entry");
        };
        assert!(matches!(
            func.values[value].def,
            Def::Inst(inst) if matches!(func.insts[inst].kind, InstKind::Select { .. })
        ));
    }

    #[test]
    fn if_converts_a_diamond_with_two_pure_arms() {
        let sig = Signature::new(vec![Ty::Bool, Ty::int(32), Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("two-arms", sig);
        let cond = param(&mut func, entry, Ty::Bool);
        let x = param(&mut func, entry, Ty::int(32));
        let y = param(&mut func, entry, Ty::int(32));
        let then_block = func.push_block(vec![]);
        let else_block = func.push_block(vec![]);
        let merge = func.push_block(vec![Ty::int(32)]);
        let result = func.blocks[merge].params[0];
        let sum = emit_bin(&mut func, then_block, InstKind::Add(x, y), Ty::int(32));
        let difference = emit_bin(&mut func, else_block, InstKind::Sub(x, y), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block,
            else_block,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        func.blocks[then_block].term = Some(Terminator::Br {
            dest: merge,
            args: [sum].into_iter().collect(),
        });
        func.blocks[else_block].term = Some(Terminator::Br {
            dest: merge,
            args: [difference].into_iter().collect(),
        });
        func.blocks[merge].term = Some(Terminator::Return { val: Some(result) });

        let mut simp = SimplifyCfg;
        assert!(simp.run(&mut func));
        assert!(crate::verify::verify_func(&func).is_empty());
        assert!(matches!(func.blocks[then_block].term, Some(Terminator::Unreachable)));
        assert!(matches!(func.blocks[else_block].term, Some(Terminator::Unreachable)));
        let select = func.blocks[entry]
            .insts
            .iter()
            .find_map(|inst| match &func.insts[*inst].kind {
                InstKind::Select { cond: c, then_val, else_val } if *c == cond => Some((*then_val, *else_val)),
                _ => None,
            })
            .expect("both diamond arms became one select");
        let Def::Inst(then_inst) = func.values[select.0].def else {
            panic!("then arm was not cloned");
        };
        let Def::Inst(else_inst) = func.values[select.1].def else {
            panic!("else arm was not cloned");
        };
        assert!(matches!(func.insts[then_inst].kind, InstKind::Add(a, b) if a == x && b == y));
        assert!(matches!(func.insts[else_inst].kind, InstKind::Sub(a, b) if a == x && b == y));
    }

    #[test]
    fn if_conversion_substitutes_arm_parameters() {
        let sig = Signature::new(vec![Ty::Bool, Ty::int(32), Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("param-diamond", sig);
        let cond = param(&mut func, entry, Ty::Bool);
        let x = param(&mut func, entry, Ty::int(32));
        let y = param(&mut func, entry, Ty::int(32));
        let arm = func.push_block(vec![Ty::int(32)]);
        let param_in_arm = func.blocks[arm].params[0];
        let merge = func.push_block(vec![Ty::int(32)]);
        let result = func.blocks[merge].params[0];
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let sum = emit_bin(&mut func, arm, InstKind::Add(param_in_arm, one), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block: arm,
            else_block: merge,
            then_args: [x].into_iter().collect(),
            else_args: [y].into_iter().collect(),
        });
        func.blocks[arm].term = Some(Terminator::Br {
            dest: merge,
            args: [sum].into_iter().collect(),
        });
        func.blocks[merge].term = Some(Terminator::Return { val: Some(result) });

        let mut simp = SimplifyCfg;
        assert!(simp.run(&mut func));
        assert!(crate::verify::verify_func(&func).is_empty());
        assert!(matches!(func.blocks[arm].term, Some(Terminator::Unreachable)));
        let select = func.blocks[entry]
            .insts
            .iter()
            .find_map(|inst| match &func.insts[*inst].kind {
                InstKind::Select { cond: c, then_val, else_val } if *c == cond && *else_val == y => Some(*then_val),
                _ => None,
            })
            .expect("parameterized diamond arm became a select");
        let Def::Inst(select_inst) = func.values[select].def else {
            panic!("select arm was not computed");
        };
        let InstKind::Add(lhs, rhs) = func.insts[select_inst].kind else {
            panic!("arm computation was not cloned");
        };
        assert_eq!(lhs, x);
        assert_eq!(func.const_value(rhs), Some(&Const::Int { bits: 32, value: 1 }));
    }

    #[test]
    fn does_not_if_convert_trapping_arms() {
        let sig = Signature::new(vec![Ty::Bool, Ty::int(32), Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("trap", sig);
        let cond = param(&mut func, entry, Ty::Bool);
        let x = param(&mut func, entry, Ty::int(32));
        let y = param(&mut func, entry, Ty::int(32));
        let arm = func.push_block(vec![]);
        let merge = func.push_block(vec![Ty::int(32)]);
        let result = func.blocks[merge].params[0];
        let quotient = emit_bin(&mut func, arm, InstKind::Sdiv(x, y), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block: arm,
            else_block: merge,
            then_args: Default::default(),
            else_args: [x].into_iter().collect(),
        });
        func.blocks[arm].term = Some(Terminator::Br {
            dest: merge,
            args: [quotient].into_iter().collect(),
        });
        func.blocks[merge].term = Some(Terminator::Return { val: Some(result) });

        let mut simp = SimplifyCfg;
        assert!(!simp.run(&mut func));
        let Def::Inst(quotient_inst) = func.values[quotient].def else {
            panic!("expected a division instruction");
        };
        assert!(func.blocks[arm].insts.contains(&quotient_inst));
        assert!(crate::verify::verify_func(&func).is_empty());
    }

    #[test]
    fn keeps_loop_headers_and_entries() {
        // The header has two preds, so the one-pred latch cannot sink
        // into it.
        let sig = Signature::new(vec![Ty::Bool], Ty::Void);
        let (mut func, entry) = Function::new("loop", sig);
        let header = func.push_block(vec![]);
        let body = func.push_block(vec![]);
        let exit = func.push_block(vec![]);
        let cond = param(&mut func, entry, Ty::Bool);
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
        let sig = Signature::new(vec![Ty::Bool], Ty::Void);
        let (mut func, entry) = Function::new("entry-loop", sig);
        let body = func.push_block(vec![]);
        let exit = func.push_block(vec![]);
        let cond = param(&mut func, entry, Ty::Bool);
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
        assert!(simp.run(&mut func));
        assert_eq!(func.blocks[entry].insts.len(), 1, "shared work stays in the entry");
        let Terminator::Return { val: Some(v) } = func.blocks[entry].term.clone().unwrap() else {
            panic!("identical arms should collapse");
        };
        assert_eq!(v, neg);
        assert!(!simp.run(&mut func), "second run is a no-op");
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
    fn inlines_small_unmarked_multi_block_callee() {
        let (mut module, _) = branchy_callee(false);
        let caller = crate::ir::Func::from_usize(1);
        let mut inline = Inline::small();
        assert!(inline.run(&mut module));
        assert!(!caller_still_calls(&module, caller));
        assert!(crate::verify::verify_module(&module).is_empty());
    }

    #[test]
    fn refuses_oversized_unmarked_multi_block_callee() {
        let (mut module, _) = branchy_callee(false);
        let caller = crate::ir::Func::from_usize(1);
        let mut inline = Inline::small();
        inline.max_insts = 0;
        inline.max_hot_insts = 0;
        inline.max_single_insts = 0;
        assert!(!inline.run(&mut module));
        assert!(caller_still_calls(&module, caller));
    }

    #[test]
    fn inlines_hot_multi_block_callee() {
        let (mut module, _) = branchy_callee(false);
        let caller = crate::ir::Func::from_usize(1);
        let entry = module.funcs[caller].entry;
        let call = module.funcs[caller].blocks[entry]
            .insts
            .iter()
            .copied()
            .find(|inst| matches!(module.funcs[caller].insts[*inst].kind, InstKind::Call { .. }))
            .expect("call site");
        let result = module.funcs[caller].insts[call].result.expect("call result");
        let (callee, args) = match &module.funcs[caller].insts[call].kind {
            InstKind::Call { callee, args } => (*callee, args.clone()),
            _ => unreachable!("found a call"),
        };
        // A second call site keeps `call_counts` above one, so the
        // single-call budget cannot mask the hot-call budget.
        let body = module.funcs[caller].push_block(vec![]);
        let exit = module.funcs[caller].push_block(vec![]);
        let cond = module.funcs[caller].push_const(Const::Int { bits: 1, value: 1 });
        module.funcs[caller].blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block: body,
            else_block: exit,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        let next_inst = crate::ir::Inst::from_usize(module.funcs[caller].insts.len());
        let second = module.funcs[caller].values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Inst(next_inst),
        });
        let extra = module.funcs[caller].insts.push(crate::ir::InstData {
            block: body,
            kind: InstKind::Call { callee, args },
            result: Some(second),
            loc: None,
        });
        module.funcs[caller].values[second].def = Def::Inst(extra);
        module.funcs[caller].blocks[body].insts.push(extra);
        module.funcs[caller].blocks[body].term = Some(Terminator::Br {
            dest: entry,
            args: Default::default(),
        });
        module.funcs[caller].blocks[exit].term = Some(Terminator::Return { val: Some(result) });

        let mut inline = Inline::small();
        inline.max_insts = 0;
        inline.max_single_insts = 0;
        assert!(inline.run(&mut module));
        assert!(!caller_still_calls(&module, caller));
        assert!(crate::verify::verify_module(&module).is_empty());
    }

    #[test]
    fn inlines_single_call_callee_on_the_larger_budget() {
        let (mut module, _) = branchy_callee(false);
        let caller = crate::ir::Func::from_usize(1);
        let mut inline = Inline::small();
        inline.max_insts = 0;
        inline.max_hot_insts = 0;
        assert!(inline.run(&mut module));
        assert!(!caller_still_calls(&module, caller));
        assert!(crate::verify::verify_module(&module).is_empty());
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
        // 30 chained adds: over the lowered heuristic budgets but under
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
            inline.max_insts = 24;
            inline.max_hot_insts = 24;
            inline.max_single_insts = 0;
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
    fn folds_float_conversions() {
        use crate::ty::FloatKind;
        #[derive(Clone, Copy)]
        enum Conv {
            SItoF,
            UItoF,
            SFtoI,
            UFtoI,
            Fptrunc,
            Fpext,
        }
        let cases = [
            (
                Const::Int { bits: 32, value: -3 },
                Conv::SItoF,
                Ty::Float(FloatKind::F64),
                Const::Float {
                    kind: FloatKind::F64,
                    bits: (-3.0f64).to_bits() as u128,
                },
            ),
            (
                Const::Int { bits: 32, value: -1 },
                Conv::UItoF,
                Ty::Float(FloatKind::F64),
                Const::Float {
                    kind: FloatKind::F64,
                    bits: (u32::MAX as f64).to_bits() as u128,
                },
            ),
            (
                Const::Float {
                    kind: FloatKind::F64,
                    bits: 2.9f64.to_bits() as u128,
                },
                Conv::SFtoI,
                Ty::int(32),
                Const::Int { bits: 32, value: 2 },
            ),
            (
                Const::Float {
                    kind: FloatKind::F64,
                    bits: (-2.9f64).to_bits() as u128,
                },
                Conv::SFtoI,
                Ty::int(32),
                Const::Int { bits: 32, value: -2 },
            ),
            (
                Const::Float {
                    kind: FloatKind::F64,
                    bits: 3.9f64.to_bits() as u128,
                },
                Conv::UFtoI,
                Ty::int(32),
                Const::Int { bits: 32, value: 3 },
            ),
            (
                Const::Float {
                    kind: FloatKind::F64,
                    bits: 1.5f64.to_bits() as u128,
                },
                Conv::Fptrunc,
                Ty::Float(FloatKind::F32),
                Const::Float {
                    kind: FloatKind::F32,
                    bits: 1.5f32.to_bits() as u128,
                },
            ),
            (
                Const::Float {
                    kind: FloatKind::F32,
                    bits: (-1.5f32).to_bits() as u128,
                },
                Conv::Fpext,
                Ty::Float(FloatKind::F64),
                Const::Float {
                    kind: FloatKind::F64,
                    bits: (-1.5f64).to_bits() as u128,
                },
            ),
        ];
        for (input, conv, result_ty, expected) in cases {
            let (mut func, entry) = Function::new("conv", Signature::new(vec![], result_ty.clone()));
            let input = func.push_const(input);
            let kind = match conv {
                Conv::SItoF => InstKind::SItoF(input),
                Conv::UItoF => InstKind::UItoF(input),
                Conv::SFtoI => InstKind::SFtoI(input),
                Conv::UFtoI => InstKind::UFtoI(input),
                Conv::Fptrunc => InstKind::Fptrunc(input),
                Conv::Fpext => InstKind::Fpext(input),
            };
            let result = emit_bin(&mut func, entry, kind, result_ty);
            func.blocks[entry].term = Some(Terminator::Return { val: Some(result) });
            let mut fold = ConstFold;
            assert!(fold.run(&mut func), "conversion did not fold");
            let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
                unreachable!()
            };
            assert_eq!(func.const_value(val.unwrap()), Some(&expected));
        }
    }

    #[test]
    fn folds_float_arithmetic() {
        use crate::ty::FloatKind;
        #[derive(Clone, Copy)]
        enum Arith {
            Add,
            Sub,
            Mul,
            Div,
        }
        let cases = [
            (FloatKind::F64, 1.5f64.to_bits() as u128, 2.25f64.to_bits() as u128, Arith::Add, 3.75f64.to_bits() as u128),
            (FloatKind::F64, 5.0f64.to_bits() as u128, 1.25f64.to_bits() as u128, Arith::Sub, 3.75f64.to_bits() as u128),
            (FloatKind::F64, 1.5f64.to_bits() as u128, 4.0f64.to_bits() as u128, Arith::Mul, 6.0f64.to_bits() as u128),
            (FloatKind::F64, 1.5f64.to_bits() as u128, 2.0f64.to_bits() as u128, Arith::Div, 0.75f64.to_bits() as u128),
            (FloatKind::F32, 1.5f32.to_bits() as u128, 2.25f32.to_bits() as u128, Arith::Add, 3.75f32.to_bits() as u128),
            (
                FloatKind::F32,
                1.0f32.to_bits() as u128,
                0.0f32.to_bits() as u128,
                Arith::Div,
                f32::INFINITY.to_bits() as u128,
            ),
        ];
        for (kind, left, right, op, expected) in cases {
            let ty = Ty::Float(kind);
            let (mut func, entry) = Function::new("float", Signature::new(vec![], ty.clone()));
            let l = func.push_const(Const::Float { kind, bits: left });
            let r = func.push_const(Const::Float { kind, bits: right });
            let arith = match op {
                Arith::Add => InstKind::FAdd(l, r),
                Arith::Sub => InstKind::FSub(l, r),
                Arith::Mul => InstKind::FMul(l, r),
                Arith::Div => InstKind::FDiv(l, r),
            };
            let result = emit_bin(&mut func, entry, arith, ty);
            func.blocks[entry].term = Some(Terminator::Return { val: Some(result) });
            let mut fold = ConstFold;
            assert!(fold.run(&mut func), "float arithmetic did not fold");
            let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
                unreachable!()
            };
            let expected = Const::Float { kind, bits: expected };
            assert_eq!(func.const_value(val.unwrap()), Some(&expected));
        }
    }

    #[test]
    fn keeps_undefined_float_to_int_conversions() {
        use crate::ty::FloatKind;
        let cases = [
            (f64::NAN, false, Ty::int(32)),
            (f64::INFINITY, false, Ty::int(32)),
            (1e300, false, Ty::int(32)),
            (-1.0, true, Ty::int(32)),
            (4294967296.0, true, Ty::int(32)),
        ];
        for (value, unsigned, result_ty) in cases {
            let (mut func, entry) = Function::new("conv", Signature::new(vec![], result_ty.clone()));
            let input = func.push_const(Const::Float {
                kind: FloatKind::F64,
                bits: value.to_bits() as u128,
            });
            let kind = if unsigned { InstKind::UFtoI(input) } else { InstKind::SFtoI(input) };
            let result = emit_bin(&mut func, entry, kind, result_ty);
            func.blocks[entry].term = Some(Terminator::Return { val: Some(result) });
            let mut fold = ConstFold;
            assert!(!fold.run(&mut func), "undefined conversion {value} was folded");
            assert_eq!(func.const_value(result), None);
        }
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
        assert!(!escaped[slot.index()]);
        assert!(escaped[loaded.index()]);
    }

    #[test]
    fn orphaned_address_use_does_not_block_mem2reg() {
        let (mut func, entry) = Function::new("m2r", Signature::new(vec![], Ty::int(32)));
        let slot = emit_alloca(&mut func, entry, Ty::int(32));
        let call = push_unit(
            &mut func,
            entry,
            InstKind::CallExtern {
                target: crate::ir::Extern::from_usize(0),
                args: [slot].into_iter().collect(),
            },
        );
        func.blocks[entry].insts.retain(|inst| *inst != call);

        let seven = func.push_const(Const::Int { bits: 32, value: 7 });
        emit_store(&mut func, entry, slot, seven, false);
        let loaded = emit_load(&mut func, entry, slot, Ty::int(32), false);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        let mut pass = Mem2Reg;
        assert!(pass.run(&mut func), "the dead call no longer makes the slot escape");
        assert!(!has_memory_ops(&func));
        assert!(crate::verify::verify_func(&func).is_empty());
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
