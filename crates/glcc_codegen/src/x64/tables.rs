//! x86-64 target tables: registers, opcodes, selection patterns, and
//! encoding recipes. Codegen behavior lives here as data wherever
//! possible — adding an instruction is three table rows (opcode spec,
//! encoding, patterns), not new control flow.

use std::sync::LazyLock;

use glcc_mir::{InstKind, Value};

use crate::isa::{Constraint, OpFlags, OpSpec, OpcodeTable, PRegInfo, RegInfo, SlotKind, SlotSpec};
use crate::machine::{OpId, PReg, RegClass};
use crate::sel::{Emit, EmitWidth, Pattern, PatternTable, Shape, SlotSel};

use super::enc::{Dir, Enc};
use super::regs;

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum Op {
    MovRR,
    MovRI,
    MovRI64,
    Lea,
    Load,
    Store,
    AddRR,
    SubRR,
    AndRR,
    OrRR,
    XorRR,
    AddI,
    SubI,
    AndI,
    OrI,
    XorI,
    CmpRR,
    CmpI,
    TestRR,
    Jcc,
    Call,
    CallRm,
    Setcc,
    Cmov,
    Movzx8,
    Movzx16,
    Movsx8,
    Movsx16,
    Movsxd,
    ImulRR,
    ImulI,
    Neg,
    Not,
    ShlI,
    ShrI,
    SarI,
    ShlCl,
    ShrCl,
    SarCl,
    Push,
    Pop,
    Ud2,
    Mfence,
    Bswap,
    DivQU,
    DivQS,
    RemQU,
    RemQS,
    Cqo,
    XorRdx,
    MovssL,
    MovssS,
    MovsdL,
    MovsdS,
    AddSS,
    AddSD,
    SubSS,
    SubSD,
    MulSS,
    MulSD,
    DivSS,
    DivSD,
    UcomisS,
    UcomisD,
    SqrtSS,
    SqrtSD,
    CvtSi2ss,
    CvtSi2sd,
    CvtSS2si,
    CvtSD2si,
    CvtSS2sd,
    CvtSD2ss,
    MovqG2X,
    MovqX2G,
}

pub const OP_COUNT: u32 = Op::MovqX2G as u32 + 1;

/// Translate a framework OpId into this table's opcode enum.
pub fn op_of(id: OpId) -> Op {
    debug_assert!((16..16 + OP_COUNT).contains(&id.0), "not an x64 opcode: {:?}", id);
    unsafe { std::mem::transmute::<u32, Op>(id.0 - 16) }
}

pub fn op_id(op: Op) -> OpId {
    OpId(op as u32 + 16)
}

const G: RegClass = RegClass::Gpr;

const fn reg(def: bool, constraint: Constraint) -> SlotSpec {
    SlotSpec { kind: SlotKind::Reg(G), def, constraint }
}
const fn imm(constraint: Constraint) -> SlotSpec {
    SlotSpec { kind: SlotKind::Imm, def: false, constraint }
}
const fn mem_slot() -> SlotSpec {
    SlotSpec { kind: SlotKind::Mem, def: false, constraint: Constraint::Any }
}

const V: RegClass = RegClass::Vec;

const fn vreg(def: bool, constraint: Constraint) -> SlotSpec {
    SlotSpec { kind: SlotKind::Reg(V), def, constraint }
}
const fn vslot() -> SlotSpec {
    SlotSpec { kind: SlotKind::Reg(V), def: false, constraint: Constraint::Reg }
}

/// SSE two-operand arithmetic: `op dst, src` with `src` forced into
/// `dst`, same discipline as the integer ALU family.
fn f2addr() -> Vec<SlotSpec> {
    vec![vreg(true, Constraint::Reuse(1)), vreg(false, Constraint::Reg), vreg(false, Constraint::Reg)]
}
const fn sym() -> SlotSpec {
    SlotSpec { kind: SlotKind::Sym, def: false, constraint: Constraint::Any }
}

/// ALU two-operand register form: `op dst, b` with `a` forced into the
/// same register as `dst` (x86's destructive encoding).
fn alu_rr() -> Vec<SlotSpec> {
    vec![reg(true, Constraint::Reuse(1)), reg(false, Constraint::Reg), reg(false, Constraint::Reg)]
}

/// Shift by %cl: the count operand must sit in RCX, which only the
/// allocator can arrange — the encoder hardcodes the CL register.
fn alu_cl() -> Vec<SlotSpec> {
    vec![reg(true, Constraint::Reuse(1)), reg(false, Constraint::Reg), reg(false, Constraint::Fixed(PReg::gpr(super::regs::RCX)))]
}

/// ALU immediate form: `op dst, imm`, same reuse discipline.
fn alu_i() -> Vec<SlotSpec> {
    vec![reg(true, Constraint::Reuse(1)), reg(false, Constraint::Reg), imm(Constraint::ImmRange(i32::MIN as i64, i32::MAX as i64))]
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

fn no_slots(name: &'static str) -> OpSpec {
    plain(name, Vec::new())
}

pub static OPCODES: LazyLock<OpcodeTable> = LazyLock::new(|| {
    const I32: Constraint = Constraint::ImmRange(i32::MIN as i64, i32::MAX as i64);
    // The framework's pseudo-op range occupies OpIds 0..16; target
    // opcodes start at 16, so the table is padded to match.
    let mut ops: Vec<OpSpec> = (0..16)
        .map(|_| OpSpec { name: "pseudo", slots: Vec::new(), flags: OpFlags::default() })
        .collect();
    ops.extend(vec![
        plain("mov", vec![reg(true, Constraint::Reuse(1)), reg(false, Constraint::Reg)]),          // MovRR
        plain("mov.i", vec![reg(true, Constraint::Reg), imm(I32)]),                                // MovRI
        plain("mov.i64", vec![reg(true, Constraint::Reg), imm(Constraint::Any)]),                  // MovRI64
        plain("lea", vec![reg(true, Constraint::Reg), mem_slot()]),                                // Lea
        plain("load", vec![reg(true, Constraint::Reg), mem_slot()]),                               // Load
        plain("store", vec![reg(false, Constraint::Reg), mem_slot()]),                             // Store
        plain("add", alu_rr()),                                                                 // AddRR
        plain("sub", alu_rr()),                                                                 // SubRR
        plain("and", alu_rr()),                                                                 // AndRR
        plain("or", alu_rr()),                                                                  // OrRR
        plain("xor", alu_rr()),                                                                 // XorRR
        plain("add.i", alu_i()),                                                                // AddI
        plain("sub.i", alu_i()),                                                                // SubI
        plain("and.i", alu_i()),                                                                // AndI
        plain("or.i", alu_i()),                                                                 // OrI
        plain("xor.i", alu_i()),                                                                // XorI
        plain("cmp", vec![reg(false, Constraint::Reg), reg(false, Constraint::Reg)]),              // CmpRR
        plain("cmp.i", vec![reg(false, Constraint::Reg), imm(I32)]),                               // CmpI
        plain("test", vec![reg(false, Constraint::Reg), reg(false, Constraint::Reg)]),             // TestRR
        branchy("jcc", vec![imm(Constraint::Any)]),                                                // Jcc
        call("call", vec![sym()]),                                                                 // Call
        call("call.rm", vec![reg(false, Constraint::Reg)]),                                        // CallRm
        plain("setcc", vec![reg(true, Constraint::Reg), imm(Constraint::Any)]),                    // Setcc
        // Conditional move: `cmovcc dst, then` executes in the false
        // value's register, so the definition is tied (Reuse) to the
        // false-path operand: dst = cond ? then : false_path. One
        // definition, no pre-MOVE — a MOVE would define the same vreg
        // twice, which regalloc2 rejects.
        plain("cmov", vec![reg(true, Constraint::Reuse(1)), reg(false, Constraint::Reg), reg(false, Constraint::Reg), imm(Constraint::Any)]), // Cmov
        plain("movzx8", vec![reg(true, Constraint::Reg), reg(false, Constraint::Reg)]),            // Movzx8
        plain("movzx16", vec![reg(true, Constraint::Reg), reg(false, Constraint::Reg)]),           // Movzx16
        plain("movsx8", vec![reg(true, Constraint::Reg), reg(false, Constraint::Reg)]),            // Movsx8
        plain("movsx16", vec![reg(true, Constraint::Reg), reg(false, Constraint::Reg)]),           // Movsx16
        plain("movsxd", vec![reg(true, Constraint::Reg), reg(false, Constraint::Reg)]),            // Movsxd
        plain("imul", alu_rr()),                                                                // ImulRR
        plain("imul.i", vec![reg(true, Constraint::Reg), reg(false, Constraint::Reg), imm(I32)]),  // ImulI
        plain("neg", vec![reg(true, Constraint::Reuse(1)), reg(false, Constraint::Reg)]),          // Neg
        plain("not", vec![reg(true, Constraint::Reuse(1)), reg(false, Constraint::Reg)]),          // Not
        plain("shl.i", alu_i()),                                                                // ShlI
        plain("shr.i", alu_i()),                                                                // ShrI
        plain("sar.i", alu_i()),                                                                // SarI
        plain("shl.cl", alu_cl()),                                                              // ShlCl
        plain("shr.cl", alu_cl()),                                                              // ShrCl
        plain("sar.cl", alu_cl()),                                                              // SarCl
        plain("push", vec![reg(false, Constraint::Reg)]),                                          // Push
        plain("pop", vec![reg(true, Constraint::Reg)]),                                            // Pop
        no_slots("ud2"),                                                                        // Ud2
        no_slots("mfence"),                                                                     // Mfence
        plain("bswap", vec![reg(true, Constraint::Reuse(1)), reg(false, Constraint::Reg)]),        // Bswap
        // Divide: the setup instruction (Cqo/XorRdx) defines the
        // high-half scratch in RDX and pins the dividend in RAX; the
        // divide holds both with an early fixed use while writing the
        // result (RAX quotient / RDX remainder) at the late point.
        spec("div.q", vec![reg(true, Constraint::Fixed(PReg::gpr(regs::RAX))), reg(false, Constraint::Fixed(PReg::gpr(regs::RDX))), reg(false, Constraint::Fixed(PReg::gpr(regs::RAX))), reg(false, Constraint::Reg)], OpFlags::default()),   // DivQU
        spec("div.s", vec![reg(true, Constraint::Fixed(PReg::gpr(regs::RAX))), reg(false, Constraint::Fixed(PReg::gpr(regs::RDX))), reg(false, Constraint::Fixed(PReg::gpr(regs::RAX))), reg(false, Constraint::Reg)], OpFlags::default()),   // DivQS
        spec("rem.q", vec![reg(true, Constraint::Fixed(PReg::gpr(regs::RDX))), reg(false, Constraint::Fixed(PReg::gpr(regs::RDX))), reg(false, Constraint::Fixed(PReg::gpr(regs::RAX))), reg(false, Constraint::Reg)], OpFlags::default()),   // RemQU
        spec("rem.s", vec![reg(true, Constraint::Fixed(PReg::gpr(regs::RDX))), reg(false, Constraint::Fixed(PReg::gpr(regs::RDX))), reg(false, Constraint::Fixed(PReg::gpr(regs::RAX))), reg(false, Constraint::Reg)], OpFlags::default()),   // RemQS
        spec("cqo", vec![reg(true, Constraint::Fixed(PReg::gpr(regs::RDX))), reg(false, Constraint::Fixed(PReg::gpr(regs::RAX)))], OpFlags::default()), // Cqo: scratch <- sext(rax)
        spec("xor.rdx", vec![reg(true, Constraint::Fixed(PReg::gpr(regs::RDX)))], OpFlags::default()),  // XorRdx: scratch <- 0
        // SSE scalar float family. Load forms (F3/F2 0F 10) also cover
        // register-to-register moves; store forms write memory.
        plain("movss.l", vec![vreg(true, Constraint::Reg), vslot()]),   // MovssL
        plain("movss.s", vec![vreg(false, Constraint::Reg), vslot()]),  // MovssS
        plain("movsd.l", vec![vreg(true, Constraint::Reg), vslot()]),   // MovsdL
        plain("movsd.s", vec![vreg(false, Constraint::Reg), vslot()]),  // MovsdS
        plain("add.ss", f2addr()),                                      // AddSS
        plain("add.sd", f2addr()),                                      // AddSD
        plain("sub.ss", f2addr()),                                      // SubSS
        plain("sub.sd", f2addr()),                                      // SubSD
        plain("mul.ss", f2addr()),                                      // MulSS
        plain("mul.sd", f2addr()),                                      // MulSD
        plain("div.ss", f2addr()),                                      // DivSS
        plain("div.sd", f2addr()),                                      // DivSD
        plain("ucomis.s", vec![vreg(false, Constraint::Reg), vslot()]), // UcomisS
        plain("ucomis.d", vec![vreg(false, Constraint::Reg), vslot()]), // UcomisD
        plain("sqrt.ss", vec![vreg(true, Constraint::Reg), vslot()]),   // SqrtSS
        plain("sqrt.sd", vec![vreg(true, Constraint::Reg), vslot()]),   // SqrtSD
        plain("cvt.si2ss", vec![vreg(true, Constraint::Reg), reg(false, Constraint::Reg)]), // CvtSi2ss
        plain("cvt.si2sd", vec![vreg(true, Constraint::Reg), reg(false, Constraint::Reg)]), // CvtSi2sd
        plain("cvt.ss2si", vec![reg(true, Constraint::Reg), vslot()]),  // CvtSS2si
        plain("cvt.sd2si", vec![reg(true, Constraint::Reg), vslot()]),  // CvtSD2si
        plain("cvt.ss2sd", vec![vreg(true, Constraint::Reg), vslot()]), // CvtSS2sd
        plain("cvt.sd2ss", vec![vreg(true, Constraint::Reg), vslot()]), // CvtSD2ss
        plain("movq.g2x", vec![vreg(true, Constraint::Reg), reg(false, Constraint::Reg)]),  // MovqG2X
        plain("movq.x2g", vec![reg(true, Constraint::Reg), vslot()]),   // MovqX2G
    ]);
    assert_eq!(ops.len() as u32, 16 + OP_COUNT, "opcode table out of sync with the Op enum");
    OpcodeTable { ops }
});

pub static REGINFO: LazyLock<RegInfo> = LazyLock::new(|| {
    let names = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"];
    let xmm_names = ["xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"];
    let mut pregs: Vec<PRegInfo> = names
        .iter()
        .enumerate()
        .map(|(hw, name)| PRegInfo {
            preg: PReg::gpr(hw as u8),
            name,
            // RSP/RBP are reserved; R11 is the allocator's scratch.
            // Callee-saved registers are allocatable: the prologue
            // saves exactly the ones a function uses.
            // r11 is the allocator's move-cycle scratch; r10 pins the
            // indirect-call callee so argument moves cannot clobber it.
            allocatable: !matches!(hw, 4 | 5 | 10 | 11),
            callee_saved: matches!(hw, 3 | 12..=15),
        })
        .collect();
    pregs.extend((0..16).map(|hw| PRegInfo {
        preg: PReg::vec(hw),
        name: xmm_names[hw as usize],
        // xmm15 is reserved as the vector-class allocator scratch;
        // SysV has no callee-saved XMM set.
        allocatable: hw != 15,
        callee_saved: false,
    }));
    let allocatable: Vec<PReg> = pregs.iter().filter(|p| p.allocatable && p.preg.class == RegClass::Gpr).map(|p| p.preg).collect();
    let vec_allocatable: Vec<PReg> = (0..15u8).map(PReg::vec).collect();
    RegInfo {
        pregs,
        preferred: [allocatable.clone(), vec_allocatable],
        non_preferred: [Vec::new(), Vec::new()],
        scratch: [Some(PReg::gpr(regs::R11)), Some(PReg::vec(15)), None],
        stack_ptr: PReg::gpr(regs::RSP),
        frame_ptr: PReg::gpr(regs::RBP),
    }
});

// -- Selection patterns ---------------------------------------------------

/// Two-operand ALU register op: `op dst, b` with a pinned into dst.
fn alu(op: Op) -> Vec<Emit> {
    vec![Emit { op: op_id(op), width: EmitWidth::Result, slots: vec![SlotSel::Result, SlotSel::Opnd(0), SlotSel::Opnd(1)] }]
}

/// Two-operand ALU immediate op with operand 1's constant as imm.
fn alu_imm(op: Op) -> Vec<Emit> {
    vec![Emit { op: op_id(op), width: EmitWidth::Result, slots: vec![SlotSel::Result, SlotSel::Opnd(0), SlotSel::OpndImm(1)] }]
}

fn dummy() -> Value {
    Value::new(0)
}

const I32: Shape = Shape::ConstIn(i32::MIN as i64, i32::MAX as i64);

pub static PATTERNS: LazyLock<PatternTable> = LazyLock::new(|| {
    let mut p: Vec<Pattern> = Vec::new();
    // The ALU family: immediate forms first, then the register form.
    let alu_kinds = [
        (InstKind::Add(dummy(), dummy()), Op::AddRR, Op::AddI),
        (InstKind::Sub(dummy(), dummy()), Op::SubRR, Op::SubI),
        (InstKind::And(dummy(), dummy()), Op::AndRR, Op::AndI),
        (InstKind::Or(dummy(), dummy()), Op::OrRR, Op::OrI),
        (InstKind::Xor(dummy(), dummy()), Op::XorRR, Op::XorI),
        (InstKind::Shl(dummy(), dummy()), Op::ShlCl, Op::ShlI),
        (InstKind::LShr(dummy(), dummy()), Op::ShrCl, Op::ShrI),
        (InstKind::AShr(dummy(), dummy()), Op::SarCl, Op::SarI),
    ];
    for (kind, rr, ri) in alu_kinds {
        p.push(Pattern { kind: kind.clone(), res_max: Some(64), lhs: Shape::Any, rhs: Shape::ConstIn(0, 255), emit: alu_imm(ri) });
        p.push(Pattern { kind: kind.clone(), res_max: Some(64), lhs: Shape::Any, rhs: I32, emit: alu_imm(ri) });
        p.push(Pattern { kind, res_max: Some(64), lhs: Shape::Any, rhs: Shape::Any, emit: alu(rr) });
    }
    // Multiplication by a constant uses the three-operand imul form;
    // otherwise the destructive two-operand one.
    p.push(Pattern {
        kind: InstKind::Mul(dummy(), dummy()),
        res_max: Some(64),
        lhs: Shape::Any,
        rhs: I32,
        emit: vec![Emit { op: op_id(Op::ImulI), width: EmitWidth::Result, slots: vec![SlotSel::Result, SlotSel::Opnd(0), SlotSel::OpndImm(1)] }],
    });
    p.push(Pattern {
        kind: InstKind::Mul(dummy(), dummy()),
        res_max: Some(64),
        lhs: Shape::Any,
        rhs: Shape::Any,
        emit: vec![Emit { op: op_id(Op::ImulRR), width: EmitWidth::Result, slots: vec![SlotSel::Result, SlotSel::Opnd(0), SlotSel::Opnd(1)] }],
    });
    p.push(Pattern {
        kind: InstKind::Load { addr: dummy(), flags: Default::default() },
        res_max: Some(64),
        lhs: Shape::Any,
        rhs: Shape::Any,
        emit: vec![Emit { op: op_id(Op::Load), width: EmitWidth::Result, slots: vec![SlotSel::Result, SlotSel::MemOf(0)] }],
    });
    p.push(Pattern {
        kind: InstKind::Store { addr: dummy(), val: dummy(), flags: Default::default() },
        res_max: None,
        lhs: Shape::Any,
        rhs: Shape::Any,
        emit: vec![Emit { op: op_id(Op::Store), width: EmitWidth::OfOpnd(1), slots: vec![SlotSel::Opnd(1), SlotSel::MemOf(0)] }],
    });
    p.push(Pattern {
        kind: InstKind::Neg(dummy()),
        res_max: Some(64),
        lhs: Shape::Any,
        rhs: Shape::Any,
        emit: vec![Emit { op: op_id(Op::Neg), width: EmitWidth::Result, slots: vec![SlotSel::Result, SlotSel::Opnd(0)] }],
    });
    p.push(Pattern {
        kind: InstKind::Bswap(dummy()),
        res_max: Some(64),
        lhs: Shape::Any,
        rhs: Shape::Any,
        emit: vec![Emit { op: op_id(Op::Bswap), width: EmitWidth::Result, slots: vec![SlotSel::Result, SlotSel::Opnd(0)] }],
    });
    p.push(Pattern {
        kind: InstKind::Not(dummy()),
        res_max: Some(64),
        lhs: Shape::Any,
        rhs: Shape::Any,
        emit: vec![Emit { op: op_id(Op::Not), width: EmitWidth::Result, slots: vec![SlotSel::Result, SlotSel::Opnd(0)] }],
    });
    PatternTable { patterns: p.leak() }
});

// -- Encoding recipes, parallel to OPCODES --------------------------------

pub static ENCODINGS: LazyLock<Vec<Enc>> = LazyLock::new(|| {
    vec![
        Enc::Rm { opc: 0x89, opc8: 0x88, dir: Dir::ToRm, reg_slot: 1, rm_slot: 0 },          // MovRR
        Enc::GrpI { opc: 0xC7, opc8: 0xC6, digit: 0, rm_slot: 0, imm_slot: 1 },              // MovRI
        Enc::PlusRI { opc: 0xB8, reg_slot: 0, imm_slot: 1 },                                 // MovRI64
        Enc::Rm { opc: 0x8D, opc8: 0x8D, dir: Dir::ToReg, reg_slot: 0, rm_slot: 1 },         // Lea
        Enc::Rm { opc: 0x8B, opc8: 0x8A, dir: Dir::ToReg, reg_slot: 0, rm_slot: 1 },         // Load
        Enc::Rm { opc: 0x89, opc8: 0x88, dir: Dir::ToRm, reg_slot: 0, rm_slot: 1 },          // Store
        Enc::Rm { opc: 0x01, opc8: 0x00, dir: Dir::ToRm, reg_slot: 2, rm_slot: 0 },          // AddRR
        Enc::Rm { opc: 0x29, opc8: 0x28, dir: Dir::ToRm, reg_slot: 2, rm_slot: 0 },          // SubRR
        Enc::Rm { opc: 0x21, opc8: 0x20, dir: Dir::ToRm, reg_slot: 2, rm_slot: 0 },          // AndRR
        Enc::Rm { opc: 0x09, opc8: 0x08, dir: Dir::ToRm, reg_slot: 2, rm_slot: 0 },          // OrRR
        Enc::Rm { opc: 0x31, opc8: 0x30, dir: Dir::ToRm, reg_slot: 2, rm_slot: 0 },          // XorRR
        Enc::GrpI { opc: 0x81, opc8: 0x80, digit: 0, rm_slot: 0, imm_slot: 2 },              // AddI
        Enc::GrpI { opc: 0x81, opc8: 0x80, digit: 5, rm_slot: 0, imm_slot: 2 },              // SubI
        Enc::GrpI { opc: 0x81, opc8: 0x80, digit: 4, rm_slot: 0, imm_slot: 2 },              // AndI
        Enc::GrpI { opc: 0x81, opc8: 0x80, digit: 1, rm_slot: 0, imm_slot: 2 },              // OrI
        Enc::GrpI { opc: 0x81, opc8: 0x80, digit: 6, rm_slot: 0, imm_slot: 2 },              // XorI
        Enc::Rm { opc: 0x39, opc8: 0x38, dir: Dir::ToRm, reg_slot: 1, rm_slot: 0 },          // CmpRR
        Enc::GrpI { opc: 0x81, opc8: 0x80, digit: 7, rm_slot: 0, imm_slot: 1 },              // CmpI
        Enc::Rm { opc: 0x85, opc8: 0x84, dir: Dir::ToRm, reg_slot: 1, rm_slot: 0 },          // TestRR
        Enc::Jcc,                                                                            // Jcc
        Enc::Rel32 { call: true },                                                           // Call
        Enc::CallRm,                                                                         // CallRm
        Enc::Setcc,                                                                          // Setcc
        Enc::Cmov { rm_slot: 2, cc_slot: 3 },                                                // Cmov
        Enc::Rm2 { opc: 0xB6, dir: Dir::ToReg, reg_slot: 0, rm_slot: 1 },                    // Movzx8
        Enc::Rm2 { opc: 0xB7, dir: Dir::ToReg, reg_slot: 0, rm_slot: 1 },                    // Movzx16
        Enc::Rm2 { opc: 0xBE, dir: Dir::ToReg, reg_slot: 0, rm_slot: 1 },                    // Movsx8
        Enc::Rm2 { opc: 0xBF, dir: Dir::ToReg, reg_slot: 0, rm_slot: 1 },                    // Movsx16
        Enc::Rm { opc: 0x63, opc8: 0x63, dir: Dir::ToReg, reg_slot: 0, rm_slot: 1 },         // Movsxd
        Enc::Rm2 { opc: 0xAF, dir: Dir::ToReg, reg_slot: 0, rm_slot: 2 },                    // ImulRR (rm = b)
        Enc::RmI { opc: 0x69, opc8: 0x6B, dir: Dir::ToReg, reg_slot: 0, rm_slot: 1, imm_slot: 2 }, // ImulI
        Enc::Grp0 { opc: 0xF7, opc8: 0xF6, digit: 3, rm_slot: 0 },                           // Neg
        Enc::Grp0 { opc: 0xF7, opc8: 0xF6, digit: 2, rm_slot: 0 },                           // Not
        Enc::ShiftI { digit: 4, rm_slot: 0, imm_slot: 2 },                                   // ShlI
        Enc::ShiftI { digit: 5, rm_slot: 0, imm_slot: 2 },                                   // ShrI
        Enc::ShiftI { digit: 7, rm_slot: 0, imm_slot: 2 },                                   // SarI
        Enc::Grp0 { opc: 0xD3, opc8: 0xD2, digit: 4, rm_slot: 0 },                           // ShlCl
        Enc::Grp0 { opc: 0xD3, opc8: 0xD2, digit: 5, rm_slot: 0 },                           // ShrCl
        Enc::Grp0 { opc: 0xD3, opc8: 0xD2, digit: 7, rm_slot: 0 },                           // SarCl
        Enc::PlusR { opc: 0x50, reg_slot: 0 },                                               // Push
        Enc::PlusR { opc: 0x58, reg_slot: 0 },                                               // Pop
        Enc::Fixed(&[0x0F, 0x0B]),                                                           // Ud2
        Enc::Fixed(&[0x0F, 0xAE, 0xF0]),                                                     // Mfence
        Enc::PlusR2 { opc: 0xC8, reg_slot: 0 },                                              // Bswap
        Enc::Grp0 { opc: 0xF7, opc8: 0xF6, digit: 6, rm_slot: 3 },                           // DivQU
        Enc::Grp0 { opc: 0xF7, opc8: 0xF6, digit: 7, rm_slot: 3 },                           // DivQS
        Enc::Grp0 { opc: 0xF7, opc8: 0xF6, digit: 6, rm_slot: 3 },                           // RemQU
        Enc::Grp0 { opc: 0xF7, opc8: 0xF6, digit: 7, rm_slot: 3 },                           // RemQS
        Enc::Fixed(&[0x48, 0x99]),                                                           // Cqo (cqo)
        Enc::Rm { opc: 0x31, opc8: 0x30, dir: Dir::ToRm, reg_slot: 0, rm_slot: 0 },          // XorRdx (rm = the rdx-resident scratch)
        Enc::Sse { prefix: 0xF3, opc: 0x10, dir: Dir::ToRm, reg_slot: 0, rm_slot: 1 },       // MovssL (reg-to-reg rides the load form)
        Enc::Sse { prefix: 0xF3, opc: 0x11, dir: Dir::ToRm, reg_slot: 0, rm_slot: 1 },       // MovssS
        Enc::Sse { prefix: 0xF2, opc: 0x10, dir: Dir::ToRm, reg_slot: 0, rm_slot: 1 },       // MovsdL
        Enc::Sse { prefix: 0xF2, opc: 0x11, dir: Dir::ToRm, reg_slot: 0, rm_slot: 1 },       // MovsdS
        Enc::Sse { prefix: 0xF3, opc: 0x58, dir: Dir::ToRm, reg_slot: 0, rm_slot: 2 },       // AddSS
        Enc::Sse { prefix: 0xF2, opc: 0x58, dir: Dir::ToRm, reg_slot: 0, rm_slot: 2 },       // AddSD
        Enc::Sse { prefix: 0xF3, opc: 0x5C, dir: Dir::ToRm, reg_slot: 0, rm_slot: 2 },       // SubSS
        Enc::Sse { prefix: 0xF2, opc: 0x5C, dir: Dir::ToRm, reg_slot: 0, rm_slot: 2 },       // SubSD
        Enc::Sse { prefix: 0xF3, opc: 0x59, dir: Dir::ToRm, reg_slot: 0, rm_slot: 2 },       // MulSS
        Enc::Sse { prefix: 0xF2, opc: 0x59, dir: Dir::ToRm, reg_slot: 0, rm_slot: 2 },       // MulSD
        Enc::Sse { prefix: 0xF3, opc: 0x5E, dir: Dir::ToRm, reg_slot: 0, rm_slot: 2 },       // DivSS
        Enc::Sse { prefix: 0xF2, opc: 0x5E, dir: Dir::ToRm, reg_slot: 0, rm_slot: 2 },       // DivSD
        Enc::Sse { prefix: 0x00, opc: 0x2E, dir: Dir::ToRm, reg_slot: 0, rm_slot: 1 },       // UcomisS
        Enc::Sse { prefix: 0x66, opc: 0x2E, dir: Dir::ToRm, reg_slot: 0, rm_slot: 1 },       // UcomisD
        Enc::Sse { prefix: 0xF3, opc: 0x51, dir: Dir::ToRm, reg_slot: 0, rm_slot: 1 },       // SqrtSS
        Enc::Sse { prefix: 0xF2, opc: 0x51, dir: Dir::ToRm, reg_slot: 0, rm_slot: 1 },       // SqrtSD
        Enc::Sse { prefix: 0xF3, opc: 0x2A, dir: Dir::ToRm, reg_slot: 0, rm_slot: 1 },       // CvtSi2ss
        Enc::Sse { prefix: 0xF2, opc: 0x2A, dir: Dir::ToRm, reg_slot: 0, rm_slot: 1 },       // CvtSi2sd
        Enc::Sse { prefix: 0xF3, opc: 0x2C, dir: Dir::ToRm, reg_slot: 0, rm_slot: 1 },       // CvtSS2si
        Enc::Sse { prefix: 0xF2, opc: 0x2C, dir: Dir::ToRm, reg_slot: 0, rm_slot: 1 },       // CvtSD2si
        Enc::Sse { prefix: 0xF3, opc: 0x5A, dir: Dir::ToRm, reg_slot: 0, rm_slot: 1 },       // CvtSS2sd
        Enc::Sse { prefix: 0xF2, opc: 0x5A, dir: Dir::ToRm, reg_slot: 0, rm_slot: 1 },       // CvtSD2ss
        Enc::Sse { prefix: 0x66, opc: 0x6E, dir: Dir::ToRm, reg_slot: 0, rm_slot: 1 },       // MovqG2X (REX.W via width)
        // 66 0F 7E /r: REG field is the XMM source, RM the GPR
        // destination — the reverse of the 0x6E form above.
        Enc::Sse { prefix: 0x66, opc: 0x7E, dir: Dir::ToRm, reg_slot: 1, rm_slot: 0 },       // MovqX2G
    ]
});
