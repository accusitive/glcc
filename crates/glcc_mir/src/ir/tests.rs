use super::*;

#[test]
fn float_to_int_conversions_are_not_pure() {
    let value = Value::from_usize(0);
    // Cranelift's f32/f64-to-int conversions trap on NaN and
    // out-of-range inputs, so they cannot be removed or speculated.
    assert!(!InstKind::SFtoI(value).is_pure());
    assert!(!InstKind::UFtoI(value).is_pure());
    assert!(InstKind::SItoF(value).is_pure());
    assert!(InstKind::UItoF(value).is_pure());
}
