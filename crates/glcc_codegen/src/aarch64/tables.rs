//! aarch64 target tables: registers, opcodes, selection patterns.
//! Codegen behavior lives here as data wherever possible — adding an
//! instruction is a table row (opcode spec, patterns), not new control
//! flow. Encodings are built directly in `enc.rs` from the operand
//! list; AArch64's fixed 32-bit instruction words need no recipe layer.

use std::sync::LazyLock;

use crate::isa::{Constraint, OpFlags, OpSpec, OpcodeTable, PRegInfo, RegInfo, SlotKind, SlotSpec};
use crate::machine::{OpId, PReg, RegClass};
use crate::sel::PatternTable;

use super::regs;

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Op {
    // -- integer ----------------------------------------------------------
    /// `mov dst, src` (orr dst, xzr, src).
    MovRR,
    /// Materialize an immediate: movz (+movk per nonzero halfword).
    /// Slots: [def gpr, imm].
    Const64,
    /// Address computation: `add dst, base, ...` (possibly several
    /// instructions), or adrp+add for symbol operands. Slots: [def, mem].
    Lea,
    /// Address of a possibly-interposed symbol through the GOT:
    /// `adrp dst, :got:sym; ldr dst, [dst, :got_lo12:sym]`.
    LeaGot,
    /// `ldr` family (zero-extends narrow loads; sign extension is an
    /// explicit MIR Sext). Slots: [def, mem].
    Load,
    /// `str` family. Slots: [use, mem].
    Store,
    /// Three-operand register ALU: `add dst, a, b` etc.
    AddRR,
    SubRR,
    AndRR,
    OrRR,
    EorRR,
    MulRR,
    /// Immediate ALU: `add dst, a, #imm12` (selector only emits when
    /// the immediate fits).
    AddI,
    SubI,
    /// Multiply by a constant: the constant rides the encoder's scratch.
    MulI,
    /// `sdiv/udiv dst, a, b`.
    SdivRR,
    UdivRR,
    /// Remainder: `sdiv x16, a, b; msub dst, x16, b, a`.
    Srem,
    Urem,
    /// `neg dst, src` (sub dst, xzr, src).
    Neg,
    /// `mvn dst, src` (orn dst, xzr, src).
    Not,
    /// Variable shifts: `lslv/lsrv/asrv dst, a, b`.
    ShlRR,
    LshrRR,
    AsrRR,
    /// Shift by immediate (ubfm/sbfm forms).
    ShlI,
    LshrI,
    AsrI,
    /// `cmp a, b` (subs xzr, a, b).
    CmpRR,
    /// `cmp a, #imm12`.
    CmpI,
    /// `ands xzr, a, b` (bool test; the selector canonicalizes narrow
    /// inputs first).
    TstRR,
    /// `cset dst, cond` — cond from the imm slot.
    Cset,
    /// `csel dst, then, else, cond` (3-operand, no reuse constraint).
    Csel,
    /// Extend ops: `sxtb/sxth/sxtw/uxtb/uxth dst, src`.
    Sxtb,
    Sxth,
    Sxtw,
    Uxtb,
    Uxth,
    /// `b.cond skip; b targets[0]; b targets[1]` — cond from the imm
    /// slot; the branch info carries both targets.
    BCond,
    /// `bl #symbol` (relocated Call26).
    Bl,
    /// `blr reg` through the pinned callee.
    Blr,
    /// `dmb ish`.
    Dmb,
    // -- float ------------------------------------------------------------
    /// `fmov s/d dst, src`.
    FMovRR,
    /// `ldr s/d` / `str s/d`.
    FLoad,
    FStore,
    /// Scalar float ALU: width picks the s/d forms.
    FAddRR,
    FSubRR,
    FMulRR,
    FDivRR,
    /// `fneg s/d dst, src`.
    FNeg,
    /// `fsqrt s/d dst, src`.
    FSqrt,
    /// `fcmp s/d a, b` (sets NZCV).
    FCmp,
    /// `fcvt d, s` (f32 -> f64) / `fcvt s, d`.
    FCvtSD,
    FCvtDS,
    /// Integer -> float: `scvtf/ucvtf s/d, w/x`. Width = source width.
    Scvtf,
    Ucvtf,
    /// Float -> integer: `fcvtzs/fcvtzu w/x, s/d`. Width = dest width.
    Fcvtzs,
    Fcvtzu,
    /// Bit moves between classes: `fmov d, x` (W64) / `fmov s, w` (W32),
    /// and the reverse.
    GprToVec,
    VecToGpr,
    /// `fcsel s/d dst, then, else, cond`.
    FCsel,
    // -- frame (prologue/epilogue, physical registers) ----------------------
    /// `stp a, b, [base, #disp]` (signed-offset form).
    StpI,
    /// `ldp a, b, [base, #disp]`.
    LdpI,
    /// Post-index `ldp a, b, [sp], #16` (epilogue unwind).
    LdpPost,
}

pub const OP_COUNT: u32 = Op::LdpPost as u32 + 1;

/// Translate a framework OpId into this table's opcode enum.
pub fn op_of(id: OpId) -> Op {
    debug_assert!((16..16 + OP_COUNT).contains(&id.0), "not an aarch64 opcode: {:?}", id);
    unsafe { std::mem::transmute::<u32, Op>(id.0 - 16) }
}

pub fn op_id(op: Op) -> OpId {
    OpId(op as u32 + 16)
}

const G: RegClass = RegClass::Gpr;
const V: RegClass = RegClass::Vec;

/// Register slots the encoder encodes by number must stay in
/// registers: a def with `Any` could legally land in a spill slot,
/// which the fixed-word encoder cannot represent.
const fn reg(def: bool, constraint: Constraint) -> SlotSpec {
    SlotSpec { kind: SlotKind::Reg(G), def, constraint }
}
const fn vreg(def: bool, constraint: Constraint) -> SlotSpec {
    SlotSpec { kind: SlotKind::Reg(V), def, constraint }
}
const fn imm(constraint: Constraint) -> SlotSpec {
    SlotSpec { kind: SlotKind::Imm, def: false, constraint }
}
const fn mem_slot() -> SlotSpec {
    SlotSpec { kind: SlotKind::Mem, def: false, constraint: Constraint::Any }
}
const fn sym() -> SlotSpec {
    SlotSpec { kind: SlotKind::Sym, def: false, constraint: Constraint::Any }
}

/// Three-operand register ALU: genuinely destructive-free on AArch64 —
/// the result may share a register with an input but need not.
fn alu3_g() -> Vec<SlotSpec> {
    vec![reg(true, Constraint::Reg), reg(false, Constraint::Reg), reg(false, Constraint::Reg)]
}
fn alu3_v() -> Vec<SlotSpec> {
    vec![vreg(true, Constraint::Reg), vreg(false, Constraint::Reg), vreg(false, Constraint::Reg)]
}
/// Two-operand: `op dst, src`.
fn un_g() -> Vec<SlotSpec> {
    vec![reg(true, Constraint::Reg), reg(false, Constraint::Reg)]
}
fn un_v() -> Vec<SlotSpec> {
    vec![vreg(true, Constraint::Reg), vreg(false, Constraint::Reg)]
}
/// ALU immediate: `op dst, a, #imm`.
fn alu_i() -> Vec<SlotSpec> {
    vec![reg(true, Constraint::Reg), reg(false, Constraint::Reg), imm(Constraint::ImmRange(0, 4095))]
}

fn spec(name: &'static str, slots: Vec<SlotSpec>, flags: OpFlags) -> OpSpec {
    OpSpec { name, slots, flags }
}

fn plain(name: &'static str, slots: Vec<SlotSpec>) -> OpSpec {
    spec(name, slots, OpFlags::default())
}

fn branchy(name: &'static str, slots: Vec<SlotSpec>) -> OpSpec {
    spec(name, slots, OpFlags { branch: true, ..Default::default() })
}

fn call(name: &'static str, slots: Vec<SlotSpec>) -> OpSpec {
    spec(name, slots, OpFlags { call: true, clobbers: regs::ALL_CALLER_SAVED, ..Default::default() })
}

pub static OPCODES: LazyLock<OpcodeTable> = LazyLock::new(|| {
    let mut ops: Vec<OpSpec> = (0..16)
        .map(|_| OpSpec { name: "pseudo", slots: Vec::new(), flags: OpFlags::default() })
        .collect();
    ops.extend(vec![
        plain("mov", un_g()),                                                       // MovRR
        plain("const", vec![reg(true, Constraint::Reg), imm(Constraint::Any)]),      // Const64
        plain("lea", vec![reg(true, Constraint::Reg), mem_slot()]),                  // Lea
        plain("lea.got", vec![reg(true, Constraint::Reg), mem_slot()]),              // LeaGot
        plain("load", vec![reg(true, Constraint::Reg), mem_slot()]),                 // Load
        plain("store", vec![reg(false, Constraint::Reg), mem_slot()]),               // Store
        plain("add", alu3_g()),                                                      // AddRR
        plain("sub", alu3_g()),                                                      // SubRR
        plain("and", alu3_g()),                                                      // AndRR
        plain("orr", alu3_g()),                                                      // OrRR
        plain("eor", alu3_g()),                                                      // EorRR
        plain("mul", alu3_g()),                                                      // MulRR
        plain("add.i", alu_i()),                                                     // AddI
        plain("sub.i", vec![reg(true, Constraint::Reg), reg(false, Constraint::Reg), imm(Constraint::ImmRange(0, 4095))]), // SubI
        plain("mul.i", vec![reg(true, Constraint::Reg), reg(false, Constraint::Reg), imm(Constraint::Any)]), // MulI
        plain("sdiv", alu3_g()),                                                     // SdivRR
        plain("udiv", alu3_g()),                                                     // UdivRR
        plain("srem", alu3_g()),                                                     // Srem
        plain("urem", alu3_g()),                                                     // Urem
        plain("neg", un_g()),                                                        // Neg
        plain("mvn", un_g()),                                                        // Not
        plain("lslv", alu3_g()),                                                     // ShlRR
        plain("lsrv", alu3_g()),                                                     // LshrRR
        plain("asrv", alu3_g()),                                                     // AsrRR
        plain("lsl.i", vec![reg(true, Constraint::Reg), reg(false, Constraint::Reg), imm(Constraint::ImmRange(0, 63))]), // ShlI
        plain("lsr.i", vec![reg(true, Constraint::Reg), reg(false, Constraint::Reg), imm(Constraint::ImmRange(0, 63))]), // LshrI
        plain("asr.i", vec![reg(true, Constraint::Reg), reg(false, Constraint::Reg), imm(Constraint::ImmRange(0, 63))]), // AsrI
        plain("cmp", vec![reg(false, Constraint::Reg), reg(false, Constraint::Reg)]), // CmpRR
        plain("cmp.i", vec![reg(false, Constraint::Reg), imm(Constraint::ImmRange(0, 4095))]), // CmpI
        plain("tst", vec![reg(false, Constraint::Reg), reg(false, Constraint::Reg)]), // TstRR
        plain("cset", vec![reg(true, Constraint::Reg), imm(Constraint::Any)]),        // Cset
        plain("csel", vec![reg(true, Constraint::Reg), reg(false, Constraint::Reg), reg(false, Constraint::Reg), imm(Constraint::Any)]), // Csel
        plain("sxtb", un_g()),                                                       // Sxtb
        plain("sxth", un_g()),                                                       // Sxth
        plain("sxtw", un_g()),                                                       // Sxtw
        plain("uxtb", un_g()),                                                       // Uxtb
        plain("uxth", un_g()),                                                       // Uxth
        branchy("bcond", vec![imm(Constraint::Any)]),                                // BCond
        call("bl", vec![sym()]),                                                     // Bl
        call("blr", vec![reg(false, Constraint::Reg)]),                              // Blr
        no_slots("dmb"),                                                             // Dmb
        plain("fmov", un_v()),                                                       // FMovRR
        plain("fld", vec![vreg(true, Constraint::Reg), mem_slot()]),                 // FLoad
        plain("fst", vec![vreg(false, Constraint::Reg), mem_slot()]),                // FStore
        plain("fadd", alu3_v()),                                                     // FAddRR
        plain("fsub", alu3_v()),                                                     // FSubRR
        plain("fmul", alu3_v()),                                                     // FMulRR
        plain("fdiv", alu3_v()),                                                     // FDivRR
        plain("fneg", un_v()),                                                       // FNeg
        plain("fsqrt", un_v()),                                                      // FSqrt
        plain("fcmp", vec![vreg(false, Constraint::Reg), vreg(false, Constraint::Reg)]), // FCmp
        plain("fcvt.sd", vec![vreg(true, Constraint::Reg), vreg(false, Constraint::Reg)]), // FCvtSD
        plain("fcvt.ds", vec![vreg(true, Constraint::Reg), vreg(false, Constraint::Reg)]), // FCvtDS
        // The imm slot names the other precision (0 = single, 1 =
        // double): scvtf/ucvtf's type field is the DESTINATION's,
        // fcvtzs/fcvtzu's the SOURCE's, and the instruction width can
        // only carry one of the two.
        plain("scvtf", vec![vreg(true, Constraint::Reg), reg(false, Constraint::Reg), imm(Constraint::Any)]), // Scvtf
        plain("ucvtf", vec![vreg(true, Constraint::Reg), reg(false, Constraint::Reg), imm(Constraint::Any)]), // Ucvtf
        plain("fcvtzs", vec![reg(true, Constraint::Reg), vreg(false, Constraint::Reg), imm(Constraint::Any)]), // Fcvtzs
        plain("fcvtzu", vec![reg(true, Constraint::Reg), vreg(false, Constraint::Reg), imm(Constraint::Any)]), // Fcvtzu
        plain("fmov.g2v", vec![vreg(true, Constraint::Reg), reg(false, Constraint::Reg)]), // GprToVec
        plain("fmov.v2g", vec![reg(true, Constraint::Reg), vreg(false, Constraint::Reg)]), // VecToGpr
        plain("fcsel", vec![vreg(true, Constraint::Reg), vreg(false, Constraint::Reg), vreg(false, Constraint::Reg), imm(Constraint::Any)]), // FCsel
        plain("stp", vec![reg(false, Constraint::Reg), reg(false, Constraint::Reg), mem_slot()]), // StpI
        plain("ldp", vec![reg(true, Constraint::Reg), reg(true, Constraint::Reg), mem_slot()]), // LdpI
        plain("ldp.post", vec![reg(true, Constraint::Reg), reg(true, Constraint::Reg), mem_slot()]), // LdpPost
    ]);
    assert_eq!(ops.len() as u32, 16 + OP_COUNT, "opcode table out of sync with the Op enum");
    OpcodeTable { ops }
});

fn no_slots(name: &'static str) -> OpSpec {
    plain(name, Vec::new())
}

pub static REGINFO: LazyLock<RegInfo> = LazyLock::new(|| {
    let mut pregs: Vec<PRegInfo> = (0..31u8)
        .map(|hw| PRegInfo {
            preg: PReg::gpr(hw),
            name: GPR_NAMES[hw as usize],
            // x29 (frame), x30 (link), x16/x17 (encoder + allocator
            // scratch), x8 (indirect-result/sret) are reserved. x18 is
            // the platform register — never allocated.
            allocatable: !(8..=18).contains(&hw) && hw != 29 && hw != 30,
            callee_saved: (19..=28).contains(&hw),
        })
        .collect();
    pregs.push(PRegInfo { preg: PReg::gpr(regs::SP), name: "sp", allocatable: false, callee_saved: false });
    pregs.extend((0..32u8).map(|hw| PRegInfo {
        preg: PReg::vec(hw),
        name: Box::leak(format!("v{hw}").into_boxed_str()),
        // v17 is the vector-class allocator scratch.
        allocatable: hw != 17,
        callee_saved: (8..=15).contains(&hw),
    }));
    let allocatable: Vec<PReg> = pregs.iter().filter(|p| p.allocatable && p.preg.class == RegClass::Gpr).map(|p| p.preg).collect();
    let vec_allocatable: Vec<PReg> = (0..32u8).filter(|&hw| hw != 17).map(PReg::vec).collect();
    RegInfo {
        pregs,
        preferred: [allocatable, vec_allocatable],
        non_preferred: [Vec::new(), Vec::new()],
        scratch: [Some(PReg::gpr(regs::IP1)), Some(PReg::vec(17)), None],
        stack_ptr: PReg::gpr(regs::SP),
        frame_ptr: PReg::gpr(regs::X29),
    }
});

const GPR_NAMES: [&str; 31] = [
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17",
    "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
];

/// The selection pattern table: AArch64 lowering is fully hook-driven
/// (`select_custom`), so the table is empty — `try_pattern` never
/// matches and unknown shapes panic with the hook's own message.
pub static PATTERNS: LazyLock<PatternTable> = LazyLock::new(|| PatternTable { patterns: &[] });
