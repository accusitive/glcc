//! aarch64 register numbers (hardware encoding order = the x/v register
//! numbers) and AAPCS64 ABI sets.

use crate::machine::PReg;

pub const X0: u8 = 0;
pub const X8: u8 = 8;
pub const X16: u8 = 16;
pub const X17: u8 = 17;
pub const X29: u8 = 29;
pub const X30: u8 = 30;
/// SP shares encoding 31 with XZR: add/sub-immediate and load/store
/// base fields read it as SP, everything else as the zero register.
pub const SP: u8 = 31;

/// The encoder's scratch register (IP0): non-allocatable, so post-RA
/// expansion (big displacements, remainder lowering) can clobber it.
pub const IP0: u8 = X16;
/// The allocator's move-cycle scratch (IP1).
pub const IP1: u8 = X17;

/// AAPCS64 integer argument registers, in order.
pub const ARG_REGS: &[PReg] = &[
    PReg::gpr(0),
    PReg::gpr(1),
    PReg::gpr(2),
    PReg::gpr(3),
    PReg::gpr(4),
    PReg::gpr(5),
    PReg::gpr(6),
    PReg::gpr(7),
];

/// Caller-clobbered integer registers: x0..x17 (x18 is platform-
/// reserved and never allocated; x29/x30 are frame/link).
pub const CALLER_SAVED: &[PReg] = &[
    PReg::gpr(0),
    PReg::gpr(1),
    PReg::gpr(2),
    PReg::gpr(3),
    PReg::gpr(4),
    PReg::gpr(5),
    PReg::gpr(6),
    PReg::gpr(7),
    PReg::gpr(9),
    PReg::gpr(10),
    PReg::gpr(11),
    PReg::gpr(12),
    PReg::gpr(13),
    PReg::gpr(14),
    PReg::gpr(15),
    PReg::gpr(16),
    PReg::gpr(17),
];

/// AAPCS64 floating-point argument registers, in order.
pub const VEC_ARG_REGS: &[PReg] = &[
    PReg::vec(0),
    PReg::vec(1),
    PReg::vec(2),
    PReg::vec(3),
    PReg::vec(4),
    PReg::vec(5),
    PReg::vec(6),
    PReg::vec(7),
];

/// Caller-clobbered vector registers: v0..v7 and v16..v31 (v8..v15
/// hold callee-saved low halves).
pub const VEC_CALLER_SAVED: &[PReg] = &[
    PReg::vec(0),
    PReg::vec(1),
    PReg::vec(2),
    PReg::vec(3),
    PReg::vec(4),
    PReg::vec(5),
    PReg::vec(6),
    PReg::vec(7),
    PReg::vec(16),
    PReg::vec(17),
    PReg::vec(18),
    PReg::vec(19),
    PReg::vec(20),
    PReg::vec(21),
    PReg::vec(22),
    PReg::vec(23),
    PReg::vec(24),
    PReg::vec(25),
    PReg::vec(26),
    PReg::vec(27),
    PReg::vec(28),
    PReg::vec(29),
    PReg::vec(30),
    PReg::vec(31),
];

/// Every caller-clobbered register across classes: the call clobber set.
pub const ALL_CALLER_SAVED: &[PReg] = &[
    PReg::gpr(0),
    PReg::gpr(1),
    PReg::gpr(2),
    PReg::gpr(3),
    PReg::gpr(4),
    PReg::gpr(5),
    PReg::gpr(6),
    PReg::gpr(7),
    PReg::gpr(9),
    PReg::gpr(10),
    PReg::gpr(11),
    PReg::gpr(12),
    PReg::gpr(13),
    PReg::gpr(14),
    PReg::gpr(15),
    PReg::vec(0),
    PReg::vec(1),
    PReg::vec(2),
    PReg::vec(3),
    PReg::vec(4),
    PReg::vec(5),
    PReg::vec(6),
    PReg::vec(7),
    PReg::vec(16),
    PReg::vec(17),
    PReg::vec(18),
    PReg::vec(19),
    PReg::vec(20),
    PReg::vec(21),
    PReg::vec(22),
    PReg::vec(23),
    PReg::vec(24),
    PReg::vec(25),
    PReg::vec(26),
    PReg::vec(27),
    PReg::vec(28),
    PReg::vec(29),
    PReg::vec(30),
    PReg::vec(31),
];

/// Callee-saved registers in save order: x19..x28 then d8..d15.
pub const CALLEE_SAVED: &[PReg] = &[
    PReg::gpr(19),
    PReg::gpr(20),
    PReg::gpr(21),
    PReg::gpr(22),
    PReg::gpr(23),
    PReg::gpr(24),
    PReg::gpr(25),
    PReg::gpr(26),
    PReg::gpr(27),
    PReg::gpr(28),
    PReg::vec(8),
    PReg::vec(9),
    PReg::vec(10),
    PReg::vec(11),
    PReg::vec(12),
    PReg::vec(13),
    PReg::vec(14),
    PReg::vec(15),
];
