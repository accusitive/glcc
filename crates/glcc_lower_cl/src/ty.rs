use cranelift_codegen::ir::types::{self, Type};
use cranelift_codegen::ir::{AbiParam, Signature};

use glcc_mir as mir;

/// Scalar MIR type to Clif type. Aggregates have no Clif type: they cross
/// as addressed memory (see [`LoweredSig`]), so this returns
/// `None` for them and callers handle the memory path.
pub fn clif_ty(ty: &mir::Ty) -> Option<Type> {
    match ty {
        mir::Ty::Void => None,
        mir::Ty::Bool => Some(types::I8),
        mir::Ty::Int { bits } => Some(match bits {
            0..=8 => types::I8,
            9..=16 => types::I16,
            17..=32 => types::I32,
            33..=64 => types::I64,
            _ => types::I128,
        }),
        mir::Ty::Float(kind) => Some(match kind {
            mir::FloatKind::F32 => types::F32,
            mir::FloatKind::F64 => types::F64,
            mir::FloatKind::F80 | mir::FloatKind::F128 => types::F128,
        }),
        // Addresses are 64 bits unless MSVC's `__ptr32` narrowed them.
        mir::Ty::Ptr { bits, .. } => Some(if *bits <= 32 { types::I32 } else { types::I64 }),
        // MSVC's `va_list` is a plain 8-byte pointer and loads and
        // stores like one. SysV's array-of-tag `va_list` never crosses
        // as a value (only its address does).
        mir::Ty::BuiltinVaList { size, .. } if *size == 8 => Some(types::I64),
        mir::Ty::BuiltinVaList { .. } => panic!("va_list has no value type"),
        mir::Ty::Array { .. } | mir::Ty::Struct { .. } => None,
    }
}

/// How aggregates cross the Clif boundary. There is deliberately no ABI
/// cleverness here: aggregates always travel by address. Scalar params and
/// returns map directly; an aggregate return becomes a hidden caller-
/// provided out-pointer prepended to the params.
pub struct LoweredSig {
    pub sig: Signature,
    /// Clif param index of each MIR param (`None` never occurs: every MIR
    /// param, scalar or aggregate, takes exactly one Clif slot).
    pub param_slots: Vec<usize>,
    /// Some when the MIR return is an aggregate: index of the hidden
    /// out-pointer param holding it.
    pub sret_slot: Option<usize>,
    /// Clif type of a scalar MIR return, if any.
    pub scalar_ret: Option<Type>,
}

pub fn lower_sig(sig: &mir::Signature) -> LoweredSig {
    if sig.variadic {
        panic!("variadic MIR functions have no Clif lowering");
    }
    let mut out = Signature::new(cranelift_codegen::isa::CallConv::SystemV);
    let mut param_slots = Vec::with_capacity(sig.params.len());
    let mut sret_slot = None;
    if sig.ret.is_aggregate() {
        sret_slot = Some(out.params.len());
        out.params.push(AbiParam::new(types::I64));
    }
    for param in &sig.params {
        param_slots.push(out.params.len());
        match clif_ty(param) {
            Some(ty) => out.params.push(AbiParam::new(ty)),
            // Aggregate params arrive as a pointer to caller memory.
            None => out.params.push(AbiParam::new(types::I64)),
        }
    }
    let scalar_ret = if sig.ret.is_void() || sig.ret.is_aggregate() {
        None
    } else {
        Some(clif_ty(&sig.ret).expect("scalar MIR return has a Clif type"))
    };
    if let Some(ty) = scalar_ret {
        out.returns.push(AbiParam::new(ty));
    }
    LoweredSig {
        sig: out,
        param_slots,
        sret_slot,
        scalar_ret,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_map_one_to_one() {
        assert_eq!(clif_ty(&mir::Ty::int(1)), Some(types::I8));
        assert_eq!(clif_ty(&mir::Ty::int(32)), Some(types::I32));
        assert_eq!(clif_ty(&mir::Ty::int(64)), Some(types::I64));
        assert_eq!(clif_ty(&mir::Ty::int(128)), Some(types::I128));
        assert_eq!(clif_ty(&mir::Ty::Bool), Some(types::I8));
        assert_eq!(clif_ty(&mir::Ty::Float(mir::FloatKind::F64)), Some(types::F64));
        assert_eq!(clif_ty(&mir::Ty::ptr(mir::Ty::Void)), Some(types::I64));
        assert!(
            clif_ty(&mir::Ty::Struct {
                fields: vec![],
                offsets: vec![],
                size: 0
            })
            .is_none()
        );
    }

    #[test]
    fn aggregate_return_gets_sret() {
        let sig = mir::Signature::new(
            vec![mir::Ty::int(32)],
            mir::Ty::Struct {
                fields: vec![mir::Ty::int(32)],
                offsets: vec![0],
                size: 4,
            },
        );
        let lowered = lower_sig(&sig);
        assert_eq!(lowered.sret_slot, Some(0));
        assert_eq!(lowered.param_slots, vec![1]);
        assert!(lowered.scalar_ret.is_none());
        assert!(lowered.sig.returns.is_empty());
    }
}
