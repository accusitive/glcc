//! Register name lookup over the generated nasm register table.

use crate::insns_gen::{REGS, RegInfo};

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
mod tests;
