use super::*;

#[test]
fn only_trusted_memory_claims_alignment() {
    assert!(trusted().notrap());
    assert!(trusted().aligned());
    assert!(notrap().notrap());
    assert!(!notrap().aligned());
}
