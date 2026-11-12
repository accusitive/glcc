//! aarch64 instruction encoding: fixed 32-bit little-endian words.
//! Unlike the x86-64 encoder there is no recipe layer — each opcode's
//! word (or word sequence) is built directly from its operand list.
//! Word constants were verified against GNU as output (see the encoding
//! tests).

use crate::isa::{EncOut, RelocReq};
use crate::machine::{MFunction, MInst, Operand, PReg, Reg, RelocKind, Width};

use super::regs;
use super::tables::{Op, op_of};

/// Condition codes (the `cond` field) and the C-predicate mapping.
pub mod cc {
    pub const EQ: u8 = 0;
    pub const NE: u8 = 1;
    pub const CS: u8 = 2;
    pub const CC: u8 = 3;
    pub const MI: u8 = 4;
    pub const PL: u8 = 5;
    pub const VS: u8 = 6;
    pub const VC: u8 = 7;
    pub const HI: u8 = 8;
    pub const LS: u8 = 9;
    pub const GE: u8 = 10;
    pub const LT: u8 = 11;
    pub const GT: u8 = 12;
    pub const LE: u8 = 13;

    /// Inverse of a condition code (used to skip a branch).
    pub fn invert(cc: u8) -> u8 {
        cc ^ 1
    }
}

/// Encode one post-RA instruction.
pub fn encode(f: &MFunction, inst: &MInst, out: &mut EncOut) {
    if inst.op.is_pseudo() {
        return match inst.op {
            crate::machine::OpId::MOVE => move_encoded(f, inst, out),
            crate::machine::OpId::RET => word(out, 0xD65F0000 | (regs::X30 as u32) << 5), // ret x30
            crate::machine::OpId::UNREACHABLE => word(out, 0xD4200000),                   // brk #0
            crate::machine::OpId::JMP => {
                word(out, 0x14000000); // b (imm26 patched via block_ref)
                out.block_refs.push((0, inst.branch.as_ref().expect("JMP branch").targets[0]));
            }
            other => panic!("glcc native backend: cannot encode pseudo op {other:?}"),
        };
    }

    match op_of(inst.op) {
        Op::MovRR => {
            let (d, s) = (ops_reg(inst, 0), ops_reg(inst, 1));
            // orr dst, xzr, src
            word(out, log_shift(0x2A000000, inst.width) | (s as u32) << 16 | 31 << 5 | d as u32);
        }
        Op::Const64 => {
            let d = ops_reg(inst, 0);
            let v = ops_imm(inst, 1) as u64;
            match inst.width {
                Width::W32 => {
                    let lo = v as u16;
                    let hi = (v >> 16) as u16;
                    word(out, 0x52800000 | (lo as u32) << 5 | d as u32); // movz w
                    if hi != 0 {
                        word(out, 0x72800000 | 1 << 21 | (hi as u32) << 5 | d as u32); // movk w
                    }
                }
                _ => const_words(v, d, out),
            }
        }
        Op::Lea => {
            let d = ops_reg(inst, 0);
            let Operand::Mem(m) = &inst.ops[1] else { panic!("glcc native backend: lea without a memory operand") };
            if let Some(sym) = &m.sym {
                // adrp dst, :pg:sym+A ; add dst, dst, :lo12:sym+A —
                // both relocations carry the full addend, so their sum
                // reconstructs S+A exactly.
                word(out, 0x90000000 | d as u32);
                word(out, 0x91000000 | (d as u32) << 5 | d as u32);
                let mut page = sym.clone();
                page.reloc = RelocKind::PagePc;
                out.relocs.push(RelocReq { at: 0, size: 4, pcrel: true, addend: m.disp, sym: page });
                let mut lo12 = sym.clone();
                lo12.reloc = RelocKind::Lo12;
                out.relocs.push(RelocReq { at: 4, size: 4, pcrel: false, addend: m.disp, sym: lo12 });
                return;
            }
            let base = match m.base {
                Some(Reg::P(p)) => p.hw,
                Some(other) => panic!("glcc native backend: unresolved lea base {other:?}"),
                None => panic!("glcc native backend: lea without base or symbol"),
            };
            let index = match m.index {
                Some(Reg::P(p)) => Some(p.hw),
                Some(other) => panic!("glcc native backend: unresolved lea index {other:?}"),
                None => None,
            };
            match index {
                None => lea_disp(inst, out, d, base, m.disp),
                Some(ix) => {
                    // add dst, base, ix, lsl #scale; then the disp.
                    word(out, log_shift(0x0B000000, inst.width) | (ix as u32) << 16 | (m.scale.min(3) as u32) << 10 | (base as u32) << 5 | d as u32);
                    if m.disp != 0 {
                        lea_disp(inst, out, d, d, m.disp);
                    }
                }
            }
        }
        Op::LeaGot => {
            let d = ops_reg(inst, 0);
            let Operand::Mem(m) = &inst.ops[1] else { panic!("glcc native backend: lea.got without a memory operand") };
            let Some(sym) = &m.sym else { panic!("glcc native backend: lea.got without a symbol") };
            // adrp dst, :got:sym+A ; ldr dst, [dst, :got_lo12:sym+A].
            // The load's zero displacement is patched by the linker; the
            // addend rides both relocations.
            word(out, 0x90000000 | d as u32);
            let ldr = (3 << 30) | 0b111 << 27 | 0b01 << 24 | 0b01 << 22 | ((d as u32) << 5) | d as u32;
            word(out, ldr);
            let mut page = sym.clone();
            page.reloc = RelocKind::GotPage;
            out.relocs.push(RelocReq { at: 0, size: 4, pcrel: true, addend: m.disp, sym: page });
            let mut lo12 = sym.clone();
            lo12.reloc = RelocKind::GotLo12;
            out.relocs.push(RelocReq { at: 4, size: 4, pcrel: false, addend: m.disp, sym: lo12 });
        }
        Op::Load | Op::FLoad => {
            let rt = ops_reg(inst, 0);
            let Operand::Mem(m) = &inst.ops[1] else { panic!("glcc native backend: load without a memory operand") };
            load_store(inst, out, rt, m, false);
        }
        Op::Store | Op::FStore => {
            let rt = ops_reg(inst, 0);
            let Operand::Mem(m) = &inst.ops[1] else { panic!("glcc native backend: store without a memory operand") };
            load_store(inst, out, rt, m, true);
        }
        Op::AddRR => alu3(inst, out, 0x0B000000),
        Op::SubRR => alu3(inst, out, 0x4B000000),
        Op::AndRR => alu3(inst, out, 0x0A000000),
        Op::OrRR => alu3(inst, out, 0x2A000000),
        Op::EorRR => alu3(inst, out, 0x4A000000),
        Op::MulRR => alu3(inst, out, 0x1B000000 | 31 << 10), // mul = madd with Ra=xzr
        Op::AddI => alu_i(inst, out, 0x11000000),
        Op::SubI => alu_i(inst, out, 0x51000000),
        Op::MulI => {
            // No multiply-by-immediate: the constant rides the encoder
            // scratch (non-allocatable, dead across the instruction).
            let (d, a, imm) = (ops_reg(inst, 0), ops_reg(inst, 1), ops_imm(inst, 2));
            const_into(inst.width, regs::IP0, imm, out);
            word(out, log_shift(0x1B000000 | 31 << 10, inst.width) | (regs::IP0 as u32) << 16 | (a as u32) << 5 | d as u32);
        }
        Op::SdivRR => alu3(inst, out, 0x1AC00C00),
        Op::UdivRR => alu3(inst, out, 0x1AC00800),
        Op::Srem | Op::Urem => {
            // sdiv x16, a, b; msub dst, x16, b, a (dst = a - q*b, with
            // a in the Ra field).
            let signed = op_of(inst.op) == Op::Srem;
            let (d, a, b) = (ops_reg(inst, 0), ops_reg(inst, 1), ops_reg(inst, 2));
            let div_op = if signed { 0x1AC00C00 } else { 0x1AC00800 };
            word(out, log_shift(div_op, inst.width) | (b as u32) << 16 | (a as u32) << 5 | regs::IP0 as u32);
            word(out, log_shift(0x1B008000, inst.width) | (regs::IP0 as u32) << 16 | (a as u32) << 10 | (b as u32) << 5 | d as u32);
        }
        Op::Neg => {
            let (d, s) = (ops_reg(inst, 0), ops_reg(inst, 1));
            word(out, log_shift(0x4B000000, inst.width) | (s as u32) << 16 | 31 << 5 | d as u32);
        }
        Op::Not => {
            let (d, s) = (ops_reg(inst, 0), ops_reg(inst, 1));
            word(out, log_shift(0x2A200000, inst.width) | (s as u32) << 16 | 31 << 5 | d as u32);
        }
        Op::ShlRR => alu3(inst, out, 0x1AC02000),
        Op::LshrRR => alu3(inst, out, 0x1AC02400),
        Op::AsrRR => alu3(inst, out, 0x1AC02800),
        Op::ShlI => shift_i(inst, out, ShiftImm::Lsl),
        Op::LshrI => shift_i(inst, out, ShiftImm::Lsr),
        Op::AsrI => shift_i(inst, out, ShiftImm::Asr),
        Op::CmpRR => {
            let (a, b) = (ops_reg(inst, 0), ops_reg(inst, 1));
            word(out, log_shift(0x6B000000, inst.width) | (b as u32) << 16 | (a as u32) << 5 | 31);
        }
        Op::CmpI => {
            let (a, imm) = (ops_reg(inst, 0), ops_imm(inst, 1));
            word(out, log_shift(0x71000000, inst.width) | (imm as u32 & 0xFFF) << 10 | (a as u32) << 5 | 31);
        }
        Op::TstRR => {
            let (a, b) = (ops_reg(inst, 0), ops_reg(inst, 1));
            word(out, log_shift(0x6A000000, inst.width) | (b as u32) << 16 | (a as u32) << 5 | 31);
        }
        Op::Cset => {
            let (d, cond) = (ops_reg(inst, 0), ops_imm(inst, 1) as u8);
            // cset = csinc dst, xzr, xzr, !cond
            word(out, log_shift(0x1A800400, inst.width) | 31 << 16 | (cc::invert(cond) as u32) << 12 | 31 << 5 | d as u32);
        }
        Op::Csel => {
            let (d, t, e, cond) = (ops_reg(inst, 0), ops_reg(inst, 1), ops_reg(inst, 2), ops_imm(inst, 3) as u8);
            word(out, log_shift(0x1A800000, inst.width) | (e as u32) << 16 | (cond as u32) << 12 | (t as u32) << 5 | d as u32);
        }
        Op::Sxtb | Op::Sxth | Op::Sxtw | Op::Uxtb | Op::Uxth => {
            let (d, s) = (ops_reg(inst, 0), ops_reg(inst, 1));
            // The destination width picks the W/X form (sxtw is
            // 64-bit by definition).
            let word_base = match (op_of(inst.op), inst.width) {
                (Op::Sxtb, Width::W64) => 0x93401C00,
                (Op::Sxtb, _) => 0x13001C00,
                (Op::Sxth, Width::W64) => 0x93403C00,
                (Op::Sxth, _) => 0x13003C00,
                (Op::Sxtw, _) => 0x93407C00,
                (Op::Uxtb, Width::W64) => 0xD3401C00,
                (Op::Uxtb, _) => 0x53001C00,
                (Op::Uxth, Width::W64) => 0xD3403C00,
                (Op::Uxth, _) => 0x53003C00,
                _ => unreachable!(),
            };
            word(out, word_base | (s as u32) << 5 | d as u32);
        }
        Op::BCond => {
            // b.!cond skip; b targets[0]; b targets[1] — one imm26
            // block_ref per edge, no imm19 patching.
            let cond = ops_imm(inst, 0) as u8;
            let targets = &inst.branch.as_ref().expect("bcond branch").targets;
            word(out, 0x54000000 | 2 << 5 | (cc::invert(cond) as u32)); // imm19 = +2 words
            word(out, 0x14000000);
            out.block_refs.push((4, targets[0]));
            if targets.len() == 2 {
                word(out, 0x14000000);
                out.block_refs.push((8, targets[1]));
            }
        }
        Op::Bl => {
            // The target symbol sits among the pinned operands (result,
            // arguments) — it is not at a fixed slot.
            let sym = match inst.ops.iter().find_map(|o| match o {
                Operand::Sym(s) => Some(s),
                _ => None,
            }) {
                Some(s) => s.clone(),
                other => panic!("glcc native backend: bl without a symbol {other:?}"),
            };
            word(out, 0x94000000);
            let mut s = sym;
            s.reloc = RelocKind::Call26;
            out.relocs.push(RelocReq { at: 0, size: 4, pcrel: true, addend: 0, sym: s });
        }
        Op::Blr => {
            // Call operands are dynamic (result/argument pins around
            // the target): the callee is the one plain register
            // operand, pinned to the non-allocatable IP0 by selection.
            let n = inst
                .ops
                .iter()
                .find_map(|o| match o {
                    Operand::Reg(Reg::P(p)) => Some(p.hw),
                    _ => None,
                })
                .expect("blr without a register callee");
            word(out, 0xD63F0000 | (n as u32) << 5);
        }
        Op::Dmb => word(out, 0xD5033BBF),
        Op::FMovRR => {
            let (d, s) = (ops_reg(inst, 0), ops_reg(inst, 1));
            let base = f_base(inst.width, 0x1E204000, 0x1E604000);
            word(out, base | (s as u32) << 5 | d as u32);
        }
        Op::FAddRR => alu3f(inst, out, 0x1E202800, 0x1E602800),
        Op::FSubRR => alu3f(inst, out, 0x1E203800, 0x1E603800),
        Op::FMulRR => alu3f(inst, out, 0x1E200800, 0x1E600800),
        Op::FDivRR => alu3f(inst, out, 0x1E201800, 0x1E601800),
        Op::FNeg => {
            let (d, s) = (ops_reg(inst, 0), ops_reg(inst, 1));
            let base = f_base(inst.width, 0x1E214000, 0x1E614000);
            word(out, base | (s as u32) << 5 | d as u32);
        }
        Op::FSqrt => {
            let (d, s) = (ops_reg(inst, 0), ops_reg(inst, 1));
            let base = f_base(inst.width, 0x1E21C000, 0x1E61C000);
            word(out, base | (s as u32) << 5 | d as u32);
        }
        Op::FCmp => {
            let (a, b) = (ops_reg(inst, 0), ops_reg(inst, 1));
            let base = f_base(inst.width, 0x1E202000, 0x1E602000);
            word(out, base | (b as u32) << 16 | (a as u32) << 5);
        }
        Op::FCvtSD => cvt(inst, out, 0x1E22C000), // fcvt d, s
        Op::FCvtDS => cvt(inst, out, 0x1E624000), // fcvt s, d
        Op::Scvtf | Op::Ucvtf => {
            let (d, s) = (ops_reg(inst, 0), ops_reg(inst, 1));
            // type = dest precision (imm slot: 0 = s, 1 = d); rmode=00
            // (nearest); opcode 010 (scvtf) / 011 (ucvtf); sf = source
            // width.
            let type_bits = if ops_imm(inst, 2) != 0 { 0b01 << 22 } else { 0 };
            let opcode = if op_of(inst.op) == Op::Ucvtf { 0b011 << 16 } else { 0b010 << 16 };
            word(out, log_shift(0x1E200000 | type_bits | opcode, inst.width) | (s as u32) << 5 | d as u32);
        }
        Op::Fcvtzs | Op::Fcvtzu => {
            let (d, s) = (ops_reg(inst, 0), ops_reg(inst, 1));
            // type = source precision (imm slot); rmode=11 (toward
            // zero); opcode 000 (fcvtzs) / 001 (fcvtzu); sf = dest
            // width.
            let type_bits = if ops_imm(inst, 2) != 0 { 0b01 << 22 } else { 0 };
            let opcode = if op_of(inst.op) == Op::Fcvtzu { 0b001 << 16 } else { 0 };
            word(out, log_shift(0x1E200000 | type_bits | 0b11 << 19 | opcode, inst.width) | (s as u32) << 5 | d as u32);
        }
        Op::GprToVec => {
            let (d, s) = (ops_reg(inst, 0), ops_reg(inst, 1));
            // fmov d, x / fmov s, w
            let base = f_base(inst.width, 0x1E270000, 0x9E670000);
            word(out, base | (s as u32) << 5 | d as u32);
        }
        Op::VecToGpr => {
            let (d, s) = (ops_reg(inst, 0), ops_reg(inst, 1));
            // fmov x, d / fmov w, s
            let base = f_base(inst.width, 0x1E260000, 0x9E660000);
            word(out, base | (s as u32) << 5 | d as u32);
        }
        Op::FCsel => {
            let (d, t, e, cond) = (ops_reg(inst, 0), ops_reg(inst, 1), ops_reg(inst, 2), ops_imm(inst, 3) as u8);
            let type_bits = f_type(inst.width);
            word(out, 0x1E200C00 | type_bits | (e as u32) << 16 | (cond as u32) << 12 | (t as u32) << 5 | d as u32);
        }
        Op::StpI => {
            let (a, b) = (ops_reg(inst, 0), ops_reg(inst, 1));
            let Operand::Mem(m) = &inst.ops[2] else { panic!("glcc native backend: stp without a memory operand") };
            let base = mem_base(m);
            assert!(m.disp >= 0 && m.disp % 8 == 0 && m.disp / 8 < 64, "glcc native backend: stp displacement {}", m.disp);
            let imm7 = (m.disp / 8) as u32;
            word(out, 0xA9000000 | imm7 << 15 | (b as u32) << 10 | (base as u32) << 5 | a as u32);
        }
        Op::LdpI | Op::LdpPost => {
            let (a, b) = (ops_reg(inst, 0), ops_reg(inst, 1));
            let Operand::Mem(m) = &inst.ops[2] else { panic!("glcc native backend: ldp without a memory operand") };
            let base = mem_base(m);
            if op_of(inst.op) == Op::LdpI {
                assert!(m.disp >= 0 && m.disp % 8 == 0 && m.disp / 8 < 64, "glcc native backend: ldp displacement {}", m.disp);
                let imm7 = (m.disp / 8) as u32;
                word(out, 0xA9400000 | imm7 << 15 | (b as u32) << 10 | (base as u32) << 5 | a as u32);
            } else {
                // Post-index: ldp a, b, [sp], #disp.
                assert!(m.disp >= 0 && m.disp % 8 == 0 && m.disp / 8 < 64, "glcc native backend: ldp.post displacement {}", m.disp);
                let imm7 = (m.disp / 8) as u32;
                word(out, 0xA8C00000 | imm7 << 15 | (b as u32) << 10 | (base as u32) << 5 | a as u32);
            }
        }
    }
}

fn f_type(width: Width) -> u32 {
    if width == Width::W32 {
        0
    } else {
        0b01 << 22
    }
}

fn f_base(width: Width, single: u32, double: u32) -> u32 {
    if width == Width::W32 {
        single
    } else {
        double
    }
}

fn mem_base(m: &crate::machine::MemOperand) -> u8 {
    match m.base {
        Some(Reg::P(p)) => p.hw,
        Some(other) => panic!("glcc native backend: unresolved memory base {other:?}"),
        None => panic!("glcc native backend: memory operand without base"),
    }
}

#[derive(Clone, Copy)]
enum ShiftImm {
    Lsl,
    Lsr,
    Asr,
}

/// Shift-by-immediate through the bitfield move instructions:
/// `lsl #k` = ubfm(d, s, (bits-k)&(bits-1), bits-1-k), `lsr #k` =
/// ubfm(d, s, k, bits-1), `asr #k` = sbfm(d, s, k, bits-1).
fn shift_i(inst: &MInst, out: &mut EncOut, kind: ShiftImm) {
    let (d, s) = (ops_reg(inst, 0), ops_reg(inst, 1));
    let imm = ops_imm(inst, 2) as u32;
    let bits: u32 = if inst.width == Width::W64 { 64 } else { 32 };
    let sf = inst.width == Width::W64;
    let (base, r, s_field) = match kind {
        ShiftImm::Lsl => (ubfm_base(sf), (bits - imm) & (bits - 1), bits - 1 - imm),
        ShiftImm::Lsr => (ubfm_base(sf), imm, bits - 1),
        ShiftImm::Asr => (sbfm_base(sf), imm, bits - 1),
    };
    word(out, base | r << 16 | s_field << 10 | (s as u32) << 5 | d as u32);
}

fn ubfm_base(sf: bool) -> u32 {
    if sf {
        0xD3400000
    } else {
        0x53000000
    }
}

fn sbfm_base(sf: bool) -> u32 {
    if sf {
        0x93400000
    } else {
        0x13000000
    }
}

/// Three-operand float op: `op dst, a, b`. No sf bit — the s/d forms
/// differ in the type field, which the base constants already carry.
fn alu3f(inst: &MInst, out: &mut EncOut, single: u32, double: u32) {
    let (d, a, b) = (ops_reg(inst, 0), ops_reg(inst, 1), ops_reg(inst, 2));
    let base = f_base(inst.width, single, double);
    word(out, base | (b as u32) << 16 | (a as u32) << 5 | d as u32);
}

/// Three-operand register op: `op dst, a, b` (b in Rm).
fn alu3(inst: &MInst, out: &mut EncOut, base: u32) {
    let (d, a, b) = (ops_reg(inst, 0), ops_reg(inst, 1), ops_reg(inst, 2));
    word(out, log_shift(base, inst.width) | (b as u32) << 16 | (a as u32) << 5 | d as u32);
}

/// Add/sub-immediate: `op dst, a, #imm12` (Rd/Rn=31 reads/writes SP).
/// Multiples of 4096 use the shift field (imm12 << 12), which stack
/// adjustments on deep frames rely on.
fn alu_i(inst: &MInst, out: &mut EncOut, base: u32) {
    let (d, a, imm) = (ops_reg(inst, 0), ops_reg(inst, 1), ops_imm(inst, 2));
    let (imm12, sh) = if (0..=4095).contains(&imm) {
        (imm as u32, 0)
    } else if imm % 4096 == 0 && imm / 4096 <= 4095 {
        ((imm / 4096) as u32, 1)
    } else {
        panic!("glcc native backend: immediate {imm} outside imm12");
    };
    word(out, log_shift(base, inst.width) | imm12 << 10 | sh << 22 | (a as u32) << 5 | d as u32);
}

fn cvt(inst: &MInst, out: &mut EncOut, base: u32) {
    let (d, s) = (ops_reg(inst, 0), ops_reg(inst, 1));
    word(out, base | (s as u32) << 5 | d as u32);
}

/// The sf bit: 64-bit operations set bit 31.
fn log_shift(base: u32, width: Width) -> u32 {
    if width == Width::W64 {
        base | 1 << 31
    } else {
        base
    }
}

fn word(out: &mut EncOut, w: u32) {
    out.bytes.extend_from_slice(&w.to_le_bytes());
}

fn ops_reg(inst: &MInst, slot: usize) -> u8 {
    match &inst.ops[slot] {
        Operand::Reg(Reg::P(p)) => p.hw,
        Operand::FixedReg(Reg::P(p), _) => p.hw,
        other => panic!("glcc native backend: register slot has {other:?} at encode time"),
    }
}

fn ops_imm(inst: &MInst, slot: usize) -> i64 {
    match &inst.ops[slot] {
        Operand::Imm(i) => *i,
        other => panic!("glcc native backend: immediate slot has {other:?} at encode time"),
    }
}

fn const_words(v: u64, d: u8, out: &mut EncOut) {
    let hw = [(v & 0xffff) as u16, (v >> 16) as u16, (v >> 32) as u16, (v >> 48) as u16];
    word(out, 0xD2800000 | (hw[0] as u32) << 5 | d as u32); // movz x
    for (k, &h) in hw.iter().enumerate().skip(1) {
        if h != 0 {
            word(out, 0xF2800000 | (k as u32) << 21 | (h as u32) << 5 | d as u32); // movk x
        }
    }
}

/// Materialize an immediate into a register (movz/movk chain), used by
/// encodings that need a constant in the encoder scratch.
fn const_into(width: Width, hw: u8, value: i64, out: &mut EncOut) {
    let v = value as u64;
    match width {
        Width::W32 => {
            let lo = v as u16;
            let hi = (v >> 16) as u16;
            word(out, 0x52800000 | (lo as u32) << 5 | hw as u32);
            if hi != 0 {
                word(out, 0x72800000 | 1 << 21 | (hi as u32) << 5 | hw as u32);
            }
        }
        _ => const_words(v, hw, out),
    }
}

/// `lea` displacement step: `add dst, base, #disp` when it fits imm12
/// (negative displacements subtract), otherwise materialize the
/// displacement into the scratch and add.
fn lea_disp(inst: &MInst, out: &mut EncOut, d: u8, base: u8, disp: i64) {
    if disp == 0 {
        if d != base {
            word(out, log_shift(0x2A000000, inst.width) | (base as u32) << 16 | 31 << 5 | d as u32);
        }
        return;
    }
    if (0..=4095).contains(&disp) {
        word(out, log_shift(0x11000000, inst.width) | (disp as u32) << 10 | (base as u32) << 5 | d as u32);
        return;
    }
    if (-4096..=-1).contains(&disp) {
        word(out, log_shift(0x51000000, inst.width) | ((-disp) as u32) << 10 | (base as u32) << 5 | d as u32);
        return;
    }
    const_into(Width::W64, regs::IP0, disp, out);
    word(out, 0x8B000000 | (regs::IP0 as u32) << 16 | (base as u32) << 5 | d as u32); // add 64-bit reg form
}

/// One load/store of a register against base+disp (no index). Size and
/// V bit follow the instruction width and the integer/float class;
/// load/store differ by opc. Displacement forms: unsigned offset
/// (scaled imm12), unscaled (ldur/stur, -256..255), or scratch
/// materialization for deep frames.
fn load_store(inst: &MInst, out: &mut EncOut, rt: u8, m: &crate::machine::MemOperand, is_store: bool) {
    assert!(m.index.is_none(), "glcc native backend: indexed load/store not supported");
    assert!(m.sym.is_none(), "glcc native backend: symbolic load/store not supported");
    let base = mem_base(m);
    let is_float = matches!(op_of(inst.op), Op::FLoad | Op::FStore);
    let (size_log2, v) = match (inst.width, is_float) {
        (Width::W8, false) => (0, 0),
        (Width::W16, false) => (1, 0),
        (Width::W32, false) => (2, 0),
        (Width::W64, false) => (3, 0),
        (Width::W32, true) => (2, 1),
        (Width::W64, true) => (3, 1),
        (w, f) => panic!("glcc native backend: load/store at width {w:?} float {f}"),
    };
    let opc = if is_store { 0b00 } else { 0b01 };
    let size = 1u32 << size_log2;
    let disp = m.disp;
    let unsigned_word = |imm12: u32, base: u8| {
        (size_log2 << 30) | 0b111 << 27 | (v << 26) | 0b01 << 24 | opc << 22 | (imm12 & 0xFFF) << 10 | (base as u32) << 5 | rt as u32
    };
    let unscaled_word = |imm9: i64, base: u8| {
        assert!((-256..=255).contains(&imm9));
        (size_log2 << 30) | 0b111 << 27 | (v << 26) | 0b00 << 24 | opc << 22 | ((imm9 & 0x1FF) as u32) << 12 | (base as u32) << 5 | rt as u32
    };
    if disp >= 0 && (disp as u32) % size == 0 && (disp as u32) / size < 4096 {
        word(out, unsigned_word((disp as u32) / size, base));
    } else if (-256..=255).contains(&disp) {
        word(out, unscaled_word(disp, base));
    } else {
        // Deep frame: materialize the displacement into the scratch,
        // add, then a zero-offset access through it.
        const_into(Width::W64, regs::IP0, disp, out);
        word(out, 0x8B000000 | (regs::IP0 as u32) << 16 | (base as u32) << 5 | regs::IP0 as u32);
        word(out, unsigned_word(0, regs::IP0));
    }
}

/// MOVE after allocation: register-register copies, register/spill-slot
/// traffic, and no-ops when source and destination agree (ABI pinning).
fn move_encoded(f: &MFunction, inst: &MInst, out: &mut EncOut) {
    let (dst, src) = match (&inst.ops[0], &inst.ops[1]) {
        (Operand::Reg(d), Operand::Reg(s)) => (*d, *s),
        other => panic!("glcc native backend: MOVE operands {other:?}"),
    };
    match (dst, src) {
        (Reg::P(d), Reg::P(s)) if d == s => {}
        // Vector-class moves: fmov s/d, register or spill-slot forms.
        (Reg::P(d), Reg::P(s)) if d.class == crate::machine::RegClass::Vec => {
            let base = f_base(inst.width, 0x1E204000, 0x1E604000);
            word(out, base | (s.hw as u32) << 5 | d.hw as u32);
        }
        (Reg::P(d), Reg::Spill(s)) if d.class == crate::machine::RegClass::Vec => {
            spill_move(f, inst, out, s, d.hw, false);
        }
        (Reg::Spill(s), Reg::P(src)) if src.class == crate::machine::RegClass::Vec => {
            spill_move(f, inst, out, s, src.hw, true);
        }
        (Reg::P(d), Reg::P(s)) => {
            word(out, log_shift(0x2A000000, inst.width) | (s.hw as u32) << 16 | 31 << 5 | d.hw as u32);
        }
        (Reg::P(d), Reg::Spill(s)) => {
            let m = fp_spill(f, s);
            load_store(inst, out, d.hw, &m, false);
        }
        (Reg::Spill(s), Reg::P(src)) => {
            let m = fp_spill(f, s);
            load_store(inst, out, src.hw, &m, true);
        }
        _ => panic!("glcc native backend: unresolved MOVE operand ({dst:?}, {src:?})"),
    }
}

fn spill_move(f: &MFunction, inst: &MInst, out: &mut EncOut, slot: u32, hw: u8, is_store: bool) {
    let m = fp_spill(f, slot);
    load_store(inst, out, hw, &m, is_store);
}

fn fp_spill(f: &MFunction, slot: u32) -> crate::machine::MemOperand {
    let off = spill_disp(f, slot);
    crate::machine::MemOperand { base: Some(Reg::P(PReg::gpr(regs::X29))), index: None, scale: 0, disp: off, sym: None }
}

fn spill_disp(f: &MFunction, slot: u32) -> i64 {
    let index = f.frame.spill_start as usize + slot as usize;
    -(f.frame.slots[index].offset as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aarch64::A64;
    use crate::machine::OpId;
    use crate::aarch64::tables::op_id as opid;
    use crate::aarch64::enc::cc;
    use crate::isa::TargetIsa;
    use crate::machine::{BranchInfo, MFunction, PReg, Reg, SymOperand, SymTarget};

    fn f() -> MFunction {
        MFunction {
            name: "t".into(),
            local: true,
            weak: false,
            blocks: vec![Default::default()],
            frame: Default::default(),
            num_vregs: 0,
            vreg_class: Vec::new(),
        }
    }

    fn g(hw: u8) -> Operand {
        Operand::Reg(Reg::P(PReg::gpr(hw)))
    }

    fn v(hw: u8) -> Operand {
        Operand::Reg(Reg::P(PReg::vec(hw)))
    }

    fn imm(i: i64) -> Operand {
        Operand::Imm(i)
    }

    fn mem(base: u8, disp: i64) -> Operand {
        Operand::Mem(crate::machine::MemOperand { base: Some(Reg::P(PReg::gpr(base))), index: None, scale: 0, disp, sym: None })
    }

    fn enc1(inst: MInst) -> Vec<u32> {
        let f = f();
        let mut out = EncOut::new();
        encode(&f, &inst, &mut out);
        out.bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn expect(words: &[u32], expected: &[u32]) {
        assert_eq!(words, expected, "encoding mismatch");
    }

    fn m(op: Op, width: Width, ops: Vec<Operand>) -> MInst {
        MInst::new(opid(op), width, ops)
    }

    #[test]
    fn moves_and_immediates() {
        expect(&enc1(m(Op::MovRR, Width::W64, vec![g(0), g(1)])), &[0xAA01_03E0]);
        expect(&enc1(m(Op::MovRR, Width::W32, vec![g(0), g(1)])), &[0x2A01_03E0]);
        expect(&enc1(m(Op::Const64, Width::W64, vec![g(0), imm(0x1234)])), &[0xD282_4680]);
        expect(&enc1(m(Op::Const64, Width::W32, vec![g(0), imm(0x1234)])), &[0x5282_4680]);
        // 0x0000567812340000: movz hw0=0, movk hw2=0x5678
        expect(&enc1(m(Op::Const64, Width::W64, vec![g(0), imm(0x5678_1234_0000)])), &[0xD280_0000, 0xF2A2_4680, 0xF2CA_CF00]);
        expect(&enc1(m(Op::Const64, Width::W64, vec![g(0), imm(-1)])), &[0xD282_0000 | 0xFFFF << 5, 0xF282_0000 | 1 << 21 | 0xFFFF << 5, 0xF282_0000 | 2 << 21 | 0xFFFF << 5, 0xF282_0000 | 3 << 21 | 0xFFFF << 5]);
    }

    #[test]
    fn alu() {
        expect(&enc1(m(Op::AddRR, Width::W64, vec![g(0), g(1), g(2)])), &[0x8B02_0020]);
        expect(&enc1(m(Op::AddRR, Width::W32, vec![g(0), g(1), g(2)])), &[0x0B02_0020]);
        expect(&enc1(m(Op::AddI, Width::W64, vec![g(0), g(1), imm(42)])), &[0x9100_A820]);
        expect(&enc1(m(Op::SubI, Width::W64, vec![g(0), g(1), imm(42)])), &[0xD100_A820]);
        expect(&enc1(m(Op::SubI, Width::W64, vec![g(31), g(31), imm(32)])), &[0xD100_83FF]);
        expect(&enc1(m(Op::MulRR, Width::W64, vec![g(0), g(1), g(2)])), &[0x9B02_7C20]);
        expect(&enc1(m(Op::MulRR, Width::W32, vec![g(0), g(1), g(2)])), &[0x1B02_7C20]);
        expect(&enc1(m(Op::SdivRR, Width::W64, vec![g(0), g(1), g(2)])), &[0x9AC2_0C20]);
        expect(&enc1(m(Op::UdivRR, Width::W64, vec![g(0), g(1), g(2)])), &[0x9AC2_0820]);
        expect(&enc1(m(Op::Neg, Width::W64, vec![g(0), g(1)])), &[0xCB01_03E0]);
        expect(&enc1(m(Op::Not, Width::W64, vec![g(0), g(1)])), &[0xAA21_03E0]);
    }

    #[test]
    fn shifts() {
        expect(&enc1(m(Op::ShlRR, Width::W64, vec![g(0), g(1), g(2)])), &[0x9AC2_2020]);
        expect(&enc1(m(Op::LshrRR, Width::W64, vec![g(0), g(1), g(2)])), &[0x9AC2_2420]);
        expect(&enc1(m(Op::AsrRR, Width::W64, vec![g(0), g(1), g(2)])), &[0x9AC2_2820]);
        expect(&enc1(m(Op::ShlI, Width::W64, vec![g(0), g(1), imm(3)])), &[0xD37D_F020]);
        expect(&enc1(m(Op::LshrI, Width::W64, vec![g(0), g(1), imm(3)])), &[0xD343_FC20]);
        expect(&enc1(m(Op::AsrI, Width::W64, vec![g(0), g(1), imm(3)])), &[0x9343_FC20]);
        expect(&enc1(m(Op::ShlI, Width::W32, vec![g(0), g(1), imm(3)])), &[0x531D_7020]);
        expect(&enc1(m(Op::LshrI, Width::W32, vec![g(0), g(1), imm(3)])), &[0x5303_7C20]);
        expect(&enc1(m(Op::AsrI, Width::W32, vec![g(0), g(1), imm(3)])), &[0x1303_7C20]);
    }

    #[test]
    fn compares_and_sets() {
        expect(&enc1(m(Op::CmpRR, Width::W64, vec![g(0), g(1)])), &[0xEB01_001F]);
        expect(&enc1(m(Op::CmpRR, Width::W32, vec![g(0), g(1)])), &[0x6B01_001F]);
        expect(&enc1(m(Op::CmpI, Width::W64, vec![g(0), imm(42)])), &[0xF100_A81F]);
        expect(&enc1(m(Op::TstRR, Width::W64, vec![g(0), g(1)])), &[0xEA01_001F]);
        expect(&enc1(m(Op::TstRR, Width::W32, vec![g(0), g(1)])), &[0x6A01_001F]);
        expect(&enc1(m(Op::Cset, Width::W64, vec![g(0), imm(cc::NE as i64)])), &[0x9A9F_07E0]);
        expect(&enc1(m(Op::Cset, Width::W32, vec![g(0), imm(cc::EQ as i64)])), &[0x1A9F_17E0]);
        expect(&enc1(m(Op::Csel, Width::W64, vec![g(0), g(1), g(2), imm(cc::LT as i64)])), &[0x9A82_B020]);
        expect(&enc1(m(Op::Csel, Width::W32, vec![g(0), g(1), g(2), imm(cc::GT as i64)])), &[0x1A82_C020]);
    }

    #[test]
    fn extends() {
        expect(&enc1(m(Op::Sxtb, Width::W64, vec![g(0), g(1)])), &[0x9340_1C20]);
        expect(&enc1(m(Op::Sxth, Width::W64, vec![g(0), g(1)])), &[0x9340_3C20]);
        expect(&enc1(m(Op::Sxtw, Width::W64, vec![g(0), g(1)])), &[0x9340_7C20]);
        expect(&enc1(m(Op::Sxtb, Width::W32, vec![g(0), g(1)])), &[0x1300_1C20]);
        expect(&enc1(m(Op::Uxtb, Width::W32, vec![g(0), g(1)])), &[0x5300_1C20]);
        expect(&enc1(m(Op::Uxth, Width::W32, vec![g(0), g(1)])), &[0x5300_3C20]);
    }

    #[test]
    fn loads_stores() {
        expect(&enc1(m(Op::Load, Width::W64, vec![g(0), mem(1, 0)])), &[0xF940_0020]);
        expect(&enc1(m(Op::Store, Width::W64, vec![g(0), mem(1, 0)])), &[0xF900_0020]);
        expect(&enc1(m(Op::Load, Width::W32, vec![g(0), mem(1, 0)])), &[0xB940_0020]);
        expect(&enc1(m(Op::Store, Width::W32, vec![g(0), mem(1, 0)])), &[0xB900_0020]);
        expect(&enc1(m(Op::Load, Width::W8, vec![g(0), mem(1, 0)])), &[0x3940_0020]);
        expect(&enc1(m(Op::Store, Width::W8, vec![g(0), mem(1, 0)])), &[0x3900_0020]);
        expect(&enc1(m(Op::Load, Width::W16, vec![g(0), mem(1, 0)])), &[0x7940_0020]);
        expect(&enc1(m(Op::Store, Width::W16, vec![g(0), mem(1, 0)])), &[0x7900_0020]);
        expect(&enc1(m(Op::Load, Width::W64, vec![g(0), mem(1, 8)])), &[0xF940_0420]);
        expect(&enc1(m(Op::Store, Width::W32, vec![g(0), mem(1, 4)])), &[0xB900_0420]);
        expect(&enc1(m(Op::Load, Width::W64, vec![g(0), mem(1, -8)])), &[0xF85F_8020]);
        expect(&enc1(m(Op::Store, Width::W64, vec![g(0), mem(1, -8)])), &[0xF81F_8020]);
        expect(&enc1(m(Op::Load, Width::W32, vec![g(0), mem(1, -4)])), &[0xB85F_C020]);
        expect(&enc1(m(Op::FLoad, Width::W64, vec![v(0), mem(1, 0)])), &[0xFD40_0020]);
        expect(&enc1(m(Op::FStore, Width::W64, vec![v(0), mem(1, 0)])), &[0xFD00_0020]);
        expect(&enc1(m(Op::FLoad, Width::W32, vec![v(0), mem(1, 0)])), &[0xBD40_0020]);
        expect(&enc1(m(Op::FStore, Width::W32, vec![v(0), mem(1, 0)])), &[0xBD00_0020]);
    }

    #[test]
    fn float_ops() {
        expect(&enc1(m(Op::FMovRR, Width::W64, vec![v(0), v(1)])), &[0x1E60_4020]);
        expect(&enc1(m(Op::FMovRR, Width::W32, vec![v(0), v(1)])), &[0x1E20_4020]);
        expect(&enc1(m(Op::FAddRR, Width::W64, vec![v(0), v(1), v(2)])), &[0x1E62_2820]);
        expect(&enc1(m(Op::FAddRR, Width::W32, vec![v(0), v(1), v(2)])), &[0x1E22_2820]);
        expect(&enc1(m(Op::FSubRR, Width::W64, vec![v(0), v(1), v(2)])), &[0x1E62_3820]);
        expect(&enc1(m(Op::FMulRR, Width::W64, vec![v(0), v(1), v(2)])), &[0x1E62_0820]);
        expect(&enc1(m(Op::FDivRR, Width::W64, vec![v(0), v(1), v(2)])), &[0x1E62_1820]);
        expect(&enc1(m(Op::FNeg, Width::W64, vec![v(0), v(1)])), &[0x1E61_4020]);
        expect(&enc1(m(Op::FNeg, Width::W32, vec![v(0), v(1)])), &[0x1E21_4020]);
        expect(&enc1(m(Op::FSqrt, Width::W64, vec![v(0), v(1)])), &[0x1E61_C020]);
        expect(&enc1(m(Op::FCmp, Width::W64, vec![v(0), v(1)])), &[0x1E61_2000]);
        expect(&enc1(m(Op::FCmp, Width::W32, vec![v(0), v(1)])), &[0x1E21_2000]);
        expect(&enc1(m(Op::FCvtSD, Width::W64, vec![v(0), v(1)])), &[0x1E22_C020]);
        expect(&enc1(m(Op::FCvtDS, Width::W64, vec![v(0), v(1)])), &[0x1E62_4020]);
        // scvtf/ucvtf: width = source, imm slot = dest precision.
        expect(&enc1(m(Op::Scvtf, Width::W32, vec![v(0), g(1), imm(0)])), &[0x1E22_0020]);
        expect(&enc1(m(Op::Scvtf, Width::W64, vec![v(0), g(1), imm(1)])), &[0x9E62_0020]);
        expect(&enc1(m(Op::Scvtf, Width::W64, vec![v(0), g(1), imm(0)])), &[0x9E22_0020]);
        expect(&enc1(m(Op::Ucvtf, Width::W32, vec![v(0), g(1), imm(0)])), &[0x1E23_0020]);
        expect(&enc1(m(Op::Ucvtf, Width::W64, vec![v(0), g(1), imm(1)])), &[0x9E63_0020]);
        // fcvtzs/fcvtzu: width = dest, imm slot = source precision.
        expect(&enc1(m(Op::Fcvtzs, Width::W64, vec![g(0), v(1), imm(1)])), &[0x9E78_0020]);
        expect(&enc1(m(Op::Fcvtzs, Width::W32, vec![g(0), v(1), imm(0)])), &[0x1E38_0020]);
        expect(&enc1(m(Op::Fcvtzu, Width::W64, vec![g(0), v(1), imm(1)])), &[0x9E79_0020]);
        expect(&enc1(m(Op::Fcvtzu, Width::W32, vec![g(0), v(1), imm(0)])), &[0x1E39_0020]);
        expect(&enc1(m(Op::GprToVec, Width::W64, vec![v(0), g(1)])), &[0x9E67_0020]);
        expect(&enc1(m(Op::GprToVec, Width::W32, vec![v(0), g(1)])), &[0x1E27_0020]);
        expect(&enc1(m(Op::VecToGpr, Width::W64, vec![g(0), v(1)])), &[0x9E66_0020]);
        expect(&enc1(m(Op::VecToGpr, Width::W32, vec![g(0), v(1)])), &[0x1E26_0020]);
        expect(&enc1(m(Op::FCsel, Width::W64, vec![v(0), v(1), v(2), imm(cc::MI as i64)])), &[0x1E62_4C20]);
        expect(&enc1(m(Op::FCsel, Width::W32, vec![v(0), v(1), v(2), imm(cc::GE as i64)])), &[0x1E22_AC20]);
    }

    #[test]
    fn control_flow_and_frame() {
        expect(&enc1(m(Op::Blr, Width::W64, vec![g(0)])), &[0xD63F_0000]);
        expect(&enc1(MInst::new(OpId::RET, Width::W64, vec![])), &[0xD65F_03C0]);
        expect(&enc1(MInst::new(OpId::UNREACHABLE, Width::W64, vec![])), &[0xD420_0000]);
        expect(&enc1(m(Op::Dmb, Width::W64, vec![])), &[0xD503_3BBF]);
        expect(&enc1(m(Op::StpI, Width::W64, vec![g(29), g(30), mem(31, 0)])), &[0xA900_7BFD]);
        expect(&enc1(m(Op::LdpI, Width::W64, vec![g(29), g(30), mem(31, 0)])), &[0xA940_7BFD]);
        expect(&enc1(m(Op::LdpPost, Width::W64, vec![g(29), g(30), mem(31, 16)])), &[0xA8C1_7BFD]);
        // bcond: b.!cond skip(+2 words); b targets[0]; b targets[1].
        let mut b = m(Op::BCond, Width::W64, vec![imm(cc::NE as i64)]);
        b.branch = Some(BranchInfo { targets: smallvec::smallvec![1, 2], args: vec![smallvec::smallvec![], smallvec::smallvec![]] });
        expect(&enc1(b), &[0x5400_0040, 0x1400_0000, 0x1400_0000]);
    }

    #[test]
    fn lea_and_calls() {
        // adrp+add against a symbol: two words, two reloc requests with
        // the full addend on both.
        let mut inst = MInst::new(
            opid(Op::Lea),
            Width::W64,
            vec![
                g(0),
                Operand::Mem(crate::machine::MemOperand {
                    base: None,
                    index: None,
                    scale: 0,
                    disp: 42,
                    sym: Some(SymOperand { target: SymTarget::Named("x"), reloc: crate::machine::RelocKind::PcRel32, addend: 0 }),
                }),
            ],
        );
        let f = f();
        let mut out = EncOut::new();
        encode(&f, &inst, &mut out);
        let words: Vec<u32> = out.bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        expect(&words, &[0x9000_0000, 0x9100_0000]);
        assert_eq!(out.relocs.len(), 2);
        assert_eq!(out.relocs[0].addend, 42);
        assert!(out.relocs[0].pcrel);
        assert!(!out.relocs[1].pcrel);
        // bl with a symbol: one Call26 reloc.
        inst = MInst::new(opid(Op::Bl), Width::W64, vec![Operand::Sym(SymOperand { target: SymTarget::Named("memcpy"), reloc: crate::machine::RelocKind::Plt32, addend: 0 })]);
        let mut out = EncOut::new();
        encode(&f, &inst, &mut out);
        let words: Vec<u32> = out.bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        expect(&words, &[0x9400_0000]);
        assert_eq!(out.relocs.len(), 1);
    }

    #[test]
    fn imm26_patch() {
        // bl placeholder + patch to target 0x100 bytes ahead of the
        // instruction (delta = 0x100 - 0 - 4).
        let mut text = vec![0u8; 4];
        text[..4].copy_from_slice(&0x9400_0000u32.to_le_bytes());
        A64.patch_branch_field(&mut text, 0, 0x100 - 4);
        let w = u32::from_le_bytes(text[..4].try_into().unwrap());
        assert_eq!(w, 0x9400_0040); // imm26 = 0x100/4 = 0x40
        // Backward branch: target 8 bytes before the instruction start.
        let mut text = vec![0u8; 4];
        text[..4].copy_from_slice(&0x1400_0000u32.to_le_bytes());
        A64.patch_branch_field(&mut text, 0, -8 - 4);
        let w = u32::from_le_bytes(text[..4].try_into().unwrap());
        assert_eq!(w, 0x17FF_FFFE); // imm26 = -2
    }

    #[test]
    fn remainder() {
        // srem x0, x1, x2 = sdiv x16, x1, x2; msub x0, x16, x1, x2
        expect(&enc1(m(Op::Srem, Width::W64, vec![g(0), g(1), g(2)])), &[0x9AC2_0C30, 0x9B10_8440]);
    }

    #[test]
    fn lea_forms() {
        // lea x0, [x1+8]: add x0, x1, #8
        expect(&enc1(m(Op::Lea, Width::W64, vec![g(0), mem(1, 8)])), &[0x9100_2020]);
        // lea x0, [x1-8]: sub x0, x1, #8
        expect(&enc1(m(Op::Lea, Width::W64, vec![g(0), mem(1, -8)])), &[0xD100_2020]);
        // lea x0, [x1+x2 lsl #3] with no disp
        expect(
            &enc1(m(
                Op::Lea,
                Width::W64,
                vec![g(0), Operand::Mem(crate::machine::MemOperand { base: Some(Reg::P(PReg::gpr(1))), index: Some(Reg::P(PReg::gpr(2))), scale: 3, disp: 0, sym: None })],
            )),
            &[0x8B02_0C20],
        );
    }
}
