use super::*;

#[test]
fn aggregates_size_and_offsets() {
    let point = Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(32)],
        offsets: vec![0, 4],
        size: 8,
    };
    assert!(point.is_aggregate());
    assert_eq!(point.size_bytes(), 8);
    assert_eq!(struct_field_offsets(&point), vec![0, 4]);
    assert_eq!(dense_field_offsets(&[Ty::int(8), Ty::int(32)]), vec![0, 4]);
}

#[test]
fn gep_steps_through_aggregates() {
    let row = Ty::Array {
        elem: Box::new(Ty::int(32)),
        len: 4,
    };
    let elem = Signature::gep_step(&row, &GepIndex::ConstIndex(2)).unwrap();
    assert_eq!(elem, Ty::int(32));
    let s = Ty::Struct {
        fields: vec![Ty::int(32), row],
        offsets: vec![0, 4],
        size: 20,
    };
    let arr = Signature::gep_step(&s, &GepIndex::Field(1)).unwrap();
    assert!(arr.is_aggregate());
    assert!(Signature::gep_step(&s, &GepIndex::Field(9)).is_none());
}
