use super::*;

#[test]
fn finds_typos_within_the_threshold() {
    let names = ["count", "counter", "print", "value"];
    assert_eq!(did_you_mean("cout", names), Some("count"));
    assert_eq!(did_you_mean("prnt", names), Some("print"));
    assert_eq!(did_you_mean("valuee", names), Some("value"));
}

#[test]
fn rejects_distant_and_short_names() {
    let names = ["alpha", "beta", "gamma"];
    assert_eq!(did_you_mean("delta", names), None);
    assert_eq!(did_you_mean("a", names), None);
    assert_eq!(did_you_mean("x", ["y"]), None);
}

#[test]
fn codes_are_stable_for_shared_messages() {
    assert_eq!(DiagCode::for_message("conflicting redeclaration"), Some(DiagCode::ConflictingRedeclaration));
    assert_eq!(DiagCode::for_message("invalid operands to binary operator"), Some(DiagCode::InvalidBinaryOperands));
    assert_eq!(DiagCode::for_message("some formatted message"), None);
    assert_eq!(DiagCode::InvalidBinaryOperands.as_str(), "E0302");
}
