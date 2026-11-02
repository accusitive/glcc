use super::*;
use crate::ty::{Signature, Ty};

#[test]
fn unterminated_block_fails() {
    let sig = Signature::new(vec![], Ty::Void);
    let (func, _) = Function::new("empty", sig);
    assert!(!verify_func(&func).is_empty());
}

#[test]
fn terminated_block_passes() {
    let sig = Signature::new(vec![], Ty::Void);
    let (mut func, entry) = Function::new("ret", sig);
    func.blocks[entry].term = Some(Terminator::Return { val: None });
    assert!(verify_func(&func).is_empty());
}
