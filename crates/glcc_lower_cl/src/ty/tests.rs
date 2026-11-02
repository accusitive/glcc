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
