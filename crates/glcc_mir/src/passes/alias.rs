//! Memory-identity resolution shared by the vectorizers.
//!
//! [`address_info`] names stack, global, and allocator objects but
//! returns `None` for addresses rooted at a pointer parameter — yet
//! `dst[i] = src[i]` through parameters is exactly the shape the
//! vectorizers must handle. [`resolve`] extends it with an
//! [`Object::Opaque`] identity: offsets along one opaque root remain
//! computable, but distinct roots may hold the same address, so opaque
//! identities may alias everything and only structural proofs (equal
//! roots, disjoint known objects) establish disjointness.

use crate::ir::{Def, Function, InstKind, Value};
use crate::ty::{GepIndex, Ty};

use super::{MemoryObject, address_info, int_const};

/// Memory identity of an address. Known objects come from
/// [`address_info`]; addresses rooted at a pointer parameter have no
/// object the pass can name — distinct parameters may hold the same
/// address — so every opaque root may alias everything and only lanes
/// sharing one root pack together.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Object {
    Known(MemoryObject),
    Opaque(Value),
}

/// Resolve an address to its [`Object`] and constant byte offset.
pub(crate) fn resolve(func: &Function, value: Value) -> Option<(Object, Option<u64>)> {
    if let Some(info) = address_info(func, value) {
        return Some((Object::Known(info.object), info.offset));
    }
    let (root, offset) = opaque_offset(func, value)?;
    Some((Object::Opaque(root), Some(offset)))
}

/// Byte offset of `value` from an opaque base: a pointer parameter or a
/// value derived from one by constant address arithmetic. Returns the
/// base parameter so callers can tell equal roots from distinct ones.
fn opaque_offset(func: &Function, value: Value) -> Option<(Value, u64)> {
    match &func.values[value].def {
        Def::Param(..) if matches!(func.values[value].ty, Ty::Ptr { .. }) => Some((value, 0)),
        Def::Inst(inst) => match &func.insts[*inst].kind {
            InstKind::Bitcast(inner) | InstKind::PtrToInt(inner) | InstKind::IntToPtr(inner) => opaque_offset(func, *inner),
            InstKind::GetElementPtr { base, indices } => {
                let (root, mut offset) = opaque_offset(func, *base)?;
                let mut ty = match &func.values[*base].ty {
                    Ty::Ptr { pointee, .. } => (**pointee).clone(),
                    _ => return None,
                };
                for index in indices.iter() {
                    // Dynamic and field steps over an opaque base have no
                    // constant offset.
                    let GepIndex::ConstIndex(index) = index else { return None };
                    let (step, next) = match &ty {
                        Ty::Array { elem, .. } => (elem.size_bytes(), (**elem).clone()),
                        other => (other.size_bytes(), other.clone()),
                    };
                    offset = offset.checked_add(index.checked_mul(step)?)?;
                    ty = next;
                }
                Some((root, offset))
            }
            InstKind::Add(a, b) => {
                let (root, offset) = opaque_offset(func, *a)?;
                let (_, delta) = int_const(func, *b)?;
                Some((root, offset.checked_add(u64::try_from(delta).ok()?)?))
            }
            _ => None,
        },
        _ => None,
    }
}
