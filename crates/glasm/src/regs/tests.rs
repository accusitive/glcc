use super::*;

#[test]
fn lookup_and_flags() {
    let rax = lookup("rax").unwrap();
    assert_eq!(val(rax), 0);
    assert!(flags(rax) & crate::insn::op::BITS64 != 0);
    assert!(flags(rax) & crate::insn::op::REG_GPR == crate::insn::op::REG_GPR);
    let al = lookup("al").unwrap();
    assert_eq!(val(al), 0);
    assert!(flags(al) & crate::insn::op::BITS8 != 0);
    let ah = lookup("ah").unwrap();
    assert_eq!(val(ah), 4);
    assert!(flags(ah) & crate::insn::op::REG_HIGH == crate::insn::op::REG_HIGH);
    assert!(lookup("eaxx").is_none());
}
