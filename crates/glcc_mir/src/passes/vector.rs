//! Bits shared by the two vectorizers ([`Slp`](super::Slp) and
//! [`LoopVec`](super::LoopVec)): the elementwise op set both pack with,
//! and the 128-bit lane shapes the backend lowers.

use crate::ir::{InstKind, Value};
use crate::ty::{FloatKind, Ty};

/// Elementwise ops the vectorizers recognize. All are pure
/// (`InstKind::is_pure`) and lower 1:1 to vector instructions on every
/// target with SIMD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ElemOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    FAdd,
    FSub,
    FMul,
}

pub(crate) fn elem_op(kind: &InstKind) -> Option<ElemOp> {
    Some(match kind {
        InstKind::Add(..) => ElemOp::Add,
        InstKind::Sub(..) => ElemOp::Sub,
        InstKind::Mul(..) => ElemOp::Mul,
        InstKind::And(..) => ElemOp::And,
        InstKind::Or(..) => ElemOp::Or,
        InstKind::Xor(..) => ElemOp::Xor,
        InstKind::Shl(..) => ElemOp::Shl,
        InstKind::FAdd(..) => ElemOp::FAdd,
        InstKind::FSub(..) => ElemOp::FSub,
        InstKind::FMul(..) => ElemOp::FMul,
        _ => return None,
    })
}

impl ElemOp {
    pub(crate) fn kind(self, a: Value, b: Value) -> InstKind {
        match self {
            ElemOp::Add => InstKind::Add(a, b),
            ElemOp::Sub => InstKind::Sub(a, b),
            ElemOp::Mul => InstKind::Mul(a, b),
            ElemOp::And => InstKind::And(a, b),
            ElemOp::Or => InstKind::Or(a, b),
            ElemOp::Xor => InstKind::Xor(a, b),
            ElemOp::Shl => InstKind::Shl(a, b),
            ElemOp::FAdd => InstKind::FAdd(a, b),
            ElemOp::FSub => InstKind::FSub(a, b),
            ElemOp::FMul => InstKind::FMul(a, b),
        }
    }

    /// Operand slots of a lane's instruction, or `None` when its shape
    /// differs from the packed op's (unary vs binary, different opcode).
    /// SLP lane matching is the only user.
    pub(crate) fn operands(self, kind: &InstKind) -> Option<[Value; 2]> {
        if elem_op(kind) != Some(self) {
            return None;
        }
        Some(match kind {
            InstKind::Add(a, b)
            | InstKind::Sub(a, b)
            | InstKind::Mul(a, b)
            | InstKind::And(a, b)
            | InstKind::Or(a, b)
            | InstKind::Xor(a, b)
            | InstKind::Shl(a, b)
            | InstKind::FAdd(a, b)
            | InstKind::FSub(a, b)
            | InstKind::FMul(a, b) => [*a, *b],
            _ => unreachable!("elem_op matched a binary op"),
        })
    }
}

/// Lane count of a 128-bit vector of `elem`, or `None` when the element
/// type has no vector shape.
pub(crate) fn vector_lanes(elem: &Ty) -> Option<u32> {
    match elem {
        Ty::Int { bits } => match bits {
            8 => Some(16),
            16 => Some(8),
            32 => Some(4),
            64 => Some(2),
            _ => None,
        },
        Ty::Float(FloatKind::F32) => Some(4),
        Ty::Float(FloatKind::F64) => Some(2),
        _ => None,
    }
}
