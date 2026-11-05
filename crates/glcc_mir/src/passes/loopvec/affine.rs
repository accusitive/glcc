//! Affine address recognition for the loop vectorizer.
//!
//! A vectorizable memory address is `root + base_off + iv_coeff * iv`:
//! `root` is a loop-invariant pointer (or none, for pure constants), the
//! offsets are constants, and `iv` is the induction variable. The C
//! frontend builds addresses both as GEPs and as `ptrtoint`/`inttoptr`
//! integer arithmetic, so the scanner walks both forms. Invariant
//! integer values inside an address (say `base + n*4` with param `n`)
//! have no affine form and fail recognition — conservative, and rare in
//! the loops worth vectorizing.

use crate::ir::{Def, Function, InstKind, Value};
use crate::ty::{GepIndex, Ty};

/// One memory address as `root + base_off + iv_coeff * iv` bytes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Affine {
    /// Loop-invariant base pointer, or `None` for a pure constant.
    pub(crate) root: Option<Value>,
    /// Constant byte offset from `root`.
    pub(crate) base_off: i64,
    /// Byte stride per induction-variable step.
    pub(crate) iv_coeff: i64,
}

impl Affine {
    fn add(self, other: Affine) -> Option<Affine> {
        let root = match (self.root, other.root) {
            (None, b) => b,
            (a, None) => a,
            (Some(a), Some(b)) if a == b => Some(a),
            _ => return None,
        };
        Some(Affine {
            root,
            base_off: self.base_off.checked_add(other.base_off)?,
            iv_coeff: self.iv_coeff.checked_add(other.iv_coeff)?,
        })
    }
}

/// Resolve an address to its affine form over `iv`, or `None`. `in_loop`
/// reports whether a value is defined inside the loop.
pub(crate) fn affine_of(func: &Function, value: Value, iv: Value, in_loop: impl Fn(Value) -> bool) -> Option<Affine> {
    scalar_affine(func, value, iv, &in_loop, 0)
}

fn scalar_affine(func: &Function, value: Value, iv: Value, in_loop: &impl Fn(Value) -> bool, depth: u32) -> Option<Affine> {
    if depth > 16 {
        return None;
    }
    // The induction variable: one unit per iteration.
    if value == iv {
        return Some(Affine { root: None, base_off: 0, iv_coeff: 1 });
    }
    // Loop-invariant values contribute constants or pointer roots.
    if !in_loop(value) {
        return invariant_affine(func, value);
    }
    let Def::Inst(inst) = func.values[value].def else {
        return None;
    };
    let recurse = |v: Value| scalar_affine(func, v, iv, in_loop, depth + 1);
    match &func.insts[inst].kind {
        InstKind::Bitcast(inner) | InstKind::PtrToInt(inner) | InstKind::IntToPtr(inner) => recurse(*inner),
        InstKind::Sext(inner) | InstKind::Zext(inner) | InstKind::Trunc(inner) => recurse(*inner),
        InstKind::GetElementPtr { base, indices } => {
            let mut out = recurse(*base)?;
            let mut ty = match &func.values[*base].ty {
                Ty::Ptr { pointee, .. } => (**pointee).clone(),
                _ => return None,
            };
            for index in indices.iter() {
                match index {
                    GepIndex::ConstIndex(index) => {
                        let (step, next) = step_of(&ty)?;
                        out.base_off = out.base_off.checked_add((*index as i64).checked_mul(step as i64)?)?;
                        ty = next;
                    }
                    GepIndex::Index(index) => {
                        let (step, next) = step_of(&ty)?;
                        let term = recurse(*index)?;
                        if term.iv_coeff > 1 || term.iv_coeff < 0 {
                            return None;
                        }
                        out.base_off = out.base_off.checked_add(term.base_off.checked_mul(step as i64)?)?;
                        out.iv_coeff = out.iv_coeff.checked_add(term.iv_coeff.checked_mul(step as i64)?)?;
                        ty = next;
                    }
                    GepIndex::Field(field) => {
                        let Ty::Struct { fields, offsets, .. } = &ty else { return None };
                        out.base_off = out.base_off.checked_add(*offsets.get(*field as usize)? as i64)?;
                        ty = fields.get(*field as usize)?.clone();
                    }
                }
            }
            Some(out)
        }
        InstKind::Add(a, b) => recurse(*a)?.add(recurse(*b)?),
        InstKind::Sub(a, b) => {
            let a = recurse(*a)?;
            let b = recurse(*b)?;
            // Subtracting a different invariant pointer is a pointer
            // difference, not an address.
            if b.iv_coeff != 0 || (b.root.is_some() && b.root != a.root) {
                return None;
            }
            Some(Affine {
                root: a.root,
                base_off: a.base_off.checked_sub(b.base_off)?,
                iv_coeff: a.iv_coeff,
            })
        }
        InstKind::Mul(a, b) => scale(recurse(*a)?, recurse(*b)?).or_else(|| scale(recurse(*b)?, recurse(*a)?)),
        InstKind::Shl(a, b) => {
            let a = recurse(*a)?;
            let b = recurse(*b)?;
            // The shift amount must be a pure constant; `x << c` scales
            // by 2^c exactly like a multiply.
            if b.iv_coeff != 0 || b.root.is_some() || !(0..64).contains(&b.base_off) {
                return None;
            }
            scale(a, Affine { root: None, base_off: 1i64 << b.base_off, iv_coeff: 0 })
        }
        _ => None,
    }
}

/// `mul * by` when `by` is a pure non-negative constant.
fn scale(mul: Affine, by: Affine) -> Option<Affine> {
    if by.iv_coeff != 0 || by.root.is_some() || by.base_off < 0 {
        return None;
    }
    Some(Affine {
        root: mul.root,
        base_off: mul.base_off.checked_mul(by.base_off)?,
        iv_coeff: mul.iv_coeff.checked_mul(by.base_off)?,
    })
}

/// GEP step size and element type for one Const/Index step over `ty`.
fn step_of(ty: &Ty) -> Option<(u64, Ty)> {
    match ty {
        Ty::Array { elem, .. } => Some((elem.size_bytes(), (**elem).clone())),
        other => Some((other.size_bytes(), other.clone())),
    }
}

/// The affine form of a loop-invariant value: pointer-typed values root
/// an address (through cast chains — the frontend `ptrtoint`s pointers
/// before doing integer address math), integer constants offset it,
/// everything else fails.
fn invariant_affine(func: &Function, value: Value) -> Option<Affine> {
    if matches!(func.values[value].ty, Ty::Ptr { .. }) {
        return Some(Affine { root: Some(value), base_off: 0, iv_coeff: 0 });
    }
    match &func.values[value].def {
        Def::Inst(inst) => match &func.insts[*inst].kind {
            InstKind::Bitcast(inner) | InstKind::PtrToInt(inner) | InstKind::IntToPtr(inner) => invariant_affine(func, *inner),
            _ => None,
        },
        Def::Const(crate::ir::Const::Int { value: int, .. }) => Some(Affine {
            root: None,
            base_off: i64::try_from(*int).ok()?,
            iv_coeff: 0,
        }),
        _ => None,
    }
}
