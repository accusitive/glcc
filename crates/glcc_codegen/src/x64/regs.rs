//! x86-64 register numbers (hardware encoding order) and ABI sets.

use crate::machine::PReg;

pub const RAX: u8 = 0;
pub const RCX: u8 = 1;
pub const RDX: u8 = 2;
pub const RBX: u8 = 3;
pub const RSP: u8 = 4;
pub const RBP: u8 = 5;
pub const RSI: u8 = 6;
pub const RDI: u8 = 7;
pub const R8: u8 = 8;
pub const R9: u8 = 9;
pub const R10: u8 = 10;
pub const R11: u8 = 11;
pub const R12: u8 = 12;
pub const R13: u8 = 13;
pub const R14: u8 = 14;
pub const R15: u8 = 15;

pub const CL: u8 = RCX;

/// SysV integer argument registers, in order.
pub const ARG_REGS: &[PReg] = &[PReg::gpr(RDI), PReg::gpr(RSI), PReg::gpr(RDX), PReg::gpr(RCX), PReg::gpr(R8), PReg::gpr(R9)];

/// Caller-saved integer registers (SysV): everything a call destroys.
pub const CALLER_SAVED: &[PReg] = &[
    PReg::gpr(RAX),
    PReg::gpr(RCX),
    PReg::gpr(RDX),
    PReg::gpr(RSI),
    PReg::gpr(RDI),
    PReg::gpr(R8),
    PReg::gpr(R9),
    PReg::gpr(R10),
    PReg::gpr(R11),
];

/// SysV floating-point argument registers, in order.
pub const VEC_ARG_REGS: &[PReg] =
    &[PReg::vec(0), PReg::vec(1), PReg::vec(2), PReg::vec(3), PReg::vec(4), PReg::vec(5), PReg::vec(6), PReg::vec(7)];

/// All XMM registers: caller-saved on SysV (no FP callee-saved set).
pub const VEC_CALLER_SAVED: &[PReg] = &[
    PReg::vec(0),
    PReg::vec(1),
    PReg::vec(2),
    PReg::vec(3),
    PReg::vec(4),
    PReg::vec(5),
    PReg::vec(6),
    PReg::vec(7),
    PReg::vec(8),
    PReg::vec(9),
    PReg::vec(10),
    PReg::vec(11),
    PReg::vec(12),
    PReg::vec(13),
    PReg::vec(14),
    PReg::vec(15),
];

/// Every caller-saved register across classes: the call clobber set.
pub const ALL_CALLER_SAVED: &[PReg] = &[
    PReg::gpr(RAX),
    PReg::gpr(RCX),
    PReg::gpr(RDX),
    PReg::gpr(RSI),
    PReg::gpr(RDI),
    PReg::gpr(R8),
    PReg::gpr(R9),
    PReg::gpr(R10),
    PReg::gpr(R11),
    PReg::vec(0),
    PReg::vec(1),
    PReg::vec(2),
    PReg::vec(3),
    PReg::vec(4),
    PReg::vec(5),
    PReg::vec(6),
    PReg::vec(7),
    PReg::vec(8),
    PReg::vec(9),
    PReg::vec(10),
    PReg::vec(11),
    PReg::vec(12),
    PReg::vec(13),
    PReg::vec(14),
    PReg::vec(15),
];
