//! Register name lookup over the generated nasm register table.

use crate::insns_gen::{RegInfo, REGS};

/// Index into `REGS`.
pub type RegId = usize;

pub const NONE: i32 = -1;

/// Exact-name lookup (nasm register names are lowercase).
pub fn lookup(name: &str) -> Option<RegId> {
    REGS.binary_search_by(|r: &RegInfo| r.name.cmp(name)).ok()
}

pub fn info(id: RegId) -> &'static RegInfo {
    &REGS[id]
}

pub fn flags(id: RegId) -> u64 {
    REGS[id].flags
}

pub fn val(id: RegId) -> i32 {
    REGS[id].val
}

pub const RIP: &str = "rip";

#[cfg(test)]
mod tests {
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
}
