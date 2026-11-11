//! x86-64 instruction encoding: REX prefixes, ModRM/SIB, immediates
//! and relocations. The encoder interprets the recipes in
//! [`super::tables::ENCODINGS`]; each recipe describes a family of
//! encodings that differ only in opcode bytes and operand slots.

use crate::isa::{EncOut, RelocReq};
use crate::machine::{MFunction, MInst, Operand, Reg, Width};

use super::regs;
use super::tables::{Op, op_of};

/// ModRM `reg` field role. `ToRm`/`ToReg` encode identically (the
/// distinction is documentation: which slot is the destination);
/// `Digit` writes a constant into the reg field (opcode groups).
#[derive(Clone, Copy)]
pub enum Dir {
    ToRm,
    ToReg,
    Digit(u8),
}

/// Encoding recipe families. Slots index into the instruction's
/// operand list; `opc8` is the 8-bit-width opcode variant where one
/// exists.
#[derive(Clone, Copy)]
pub enum Enc {
    /// `opcode /r`: register in `reg_slot`, register-or-memory in
    /// `rm_slot`.
    Rm { opc: u8, opc8: u8, dir: Dir, reg_slot: u8, rm_slot: u8 },
    /// Two-byte opcode (0F xx) form of [`Enc::Rm`].
    Rm2 { opc: u8, dir: Dir, reg_slot: u8, rm_slot: u8 },
    /// `opcode /r` plus a 32-bit immediate (three-operand imul).
    RmI { opc: u8, opc8: u8, dir: Dir, reg_slot: u8, rm_slot: u8, imm_slot: u8 },
    /// Opcode group with register-or-memory in `rm_slot`, a constant
    /// `digit` in the reg field, and an immediate (0x80/0x81/0x83,
    /// C6/C7 families — see `encode_grp`).
    GrpI { opc: u8, opc8: u8, digit: u8, rm_slot: u8, imm_slot: u8 },
    /// Opcode group with no immediate (F7 groups, D3 shift-by-CL).
    Grp0 { opc: u8, opc8: u8, digit: u8, rm_slot: u8 },
    /// Shift by immediate (C0/C1 groups).
    ShiftI { digit: u8, rm_slot: u8, imm_slot: u8 },
    /// `opcode +r`: register number in the low opcode bits (push/pop).
    PlusR { opc: u8, reg_slot: u8 },
    /// Two-byte `0F opcode +r` (bswap).
    PlusR2 { opc: u8, reg_slot: u8 },
    /// `opcode +r` followed by an immediate (mov reg, imm64).
    PlusRI { opc: u8, reg_slot: u8, imm_slot: u8 },
    /// Conditional branch: 0F 80+cc rel32, cc from the imm slot,
    /// target from the branch info.
    Jcc,
    /// Set byte on condition: 0F 90+cc /0, cc from imm slot 1.
    Setcc,
    /// Conditional move: 0F 40+cc /r, cc from `cc_slot`.
    Cmov { rm_slot: u8, cc_slot: u8 },
    /// E8 (call) / E9 (jmp) rel32 against a symbol.
    Rel32 { call: bool },
    /// SSE scalar form: legacy `prefix` (0 = none), then REX, then
    /// `0F opc /r`. The REX.W bit follows the instruction width (set
    /// only by the gpr/xmm bit moves and 64-bit int conversions).
    Sse { prefix: u8, opc: u8, dir: Dir, reg_slot: u8, rm_slot: u8 },
    /// FF /2 through the register callee. Call operands are dynamic
    /// (result/argument pins around the target), so the callee cannot
    /// live at a fixed slot: it is the one plain `Reg` operand.
    CallRm,
    /// Fixed byte sequence, no operands.
    Fixed(&'static [u8]),
}

/// Encode one post-RA instruction.
pub fn encode(f: &MFunction, inst: &MInst, out: &mut EncOut) {
    if inst.op.is_pseudo() {
        return match inst.op {
            crate::machine::OpId::MOVE => move_encoded(f, inst, out),
            crate::machine::OpId::RET => out.bytes.push(0xC3),
            crate::machine::OpId::UNREACHABLE => out.bytes.extend_from_slice(&[0x0F, 0x0B]),
            crate::machine::OpId::JMP => {
                out.bytes.push(0xE9);
                let at = out.bytes.len() as u8;
                out.bytes.extend_from_slice(&[0, 0, 0, 0]);
                out.block_refs.push((at, inst.branch.as_ref().expect("JMP branch").targets[0]));
            }
            other => panic!("glcc native backend: cannot encode pseudo op {other:?}"),
        };
    }

    let enc = &super::tables::ENCODINGS[op_of(inst.op) as usize];
    match *enc {
        Enc::Rm { opc, opc8, dir, reg_slot, rm_slot } => {
            let reg_field = match dir {
                Dir::Digit(d) => d,
                _ => ops_reg(inst, reg_slot),
            };
            emit_rm(f, inst, out, &[pick(inst.width, opc, opc8)], reg_field, rm_slot);
        }
        Enc::Rm2 { opc, dir, reg_slot, rm_slot } => {
            let reg_field = match dir {
                Dir::Digit(d) => d,
                _ => ops_reg(inst, reg_slot),
            };
            emit_rm(f, inst, out, &[0x0F, opc], reg_field, rm_slot);
        }
        Enc::RmI { opc, dir, reg_slot, rm_slot, imm_slot, .. } => {
            // imul r, r/m, imm32 (0x69); the imm8 form is a size win we
            // can pick later.
            let reg_field = match dir {
                Dir::Digit(d) => d,
                _ => ops_reg(inst, reg_slot),
            };
            emit_rm(f, inst, out, &[opc], reg_field, rm_slot);
            // imul's immediate matches the operand size (imm16 under
            // the 0x66 prefix, which emit_rm emits for W16).
            let imm_size = if inst.width == Width::W16 { 2 } else { 4 };
            emit_imm(out, ops_imm(inst, imm_slot), imm_size);
        }
        Enc::GrpI { opc, opc8, digit, rm_slot, imm_slot } => {
            encode_grp(f, inst, out, opc, opc8, digit, rm_slot, imm_slot);
        }
        Enc::Grp0 { opc, opc8, digit, rm_slot } => {
            emit_rm(f, inst, out, &[pick(inst.width, opc, opc8)], digit, rm_slot);
        }
        Enc::ShiftI { digit, rm_slot, imm_slot } => {
            let imm = ops_imm(inst, imm_slot);
            if !(0..=255).contains(&imm) {
                panic!("glcc native backend: shift amount {imm} out of imm8 range");
            }
            emit_rm(f, inst, out, &[pick(inst.width, 0xC1, 0xC0)], digit, rm_slot);
            emit_imm(out, imm, 1);
        }
        Enc::PlusR { opc, reg_slot } => {
            let hw = ops_reg(inst, reg_slot);
            let force8 = inst.width == Width::W8 && hw >= 4;
            // `opcode + r` forms take the register from the low opcode
            // bits extended by REX.B, not the reg field.
            if inst.width == Width::W16 {
                out.bytes.push(0x66);
            }
            emit_rex_f(out, inst.width == Width::W64, false, false, rex_bit(hw), force8);
            out.bytes.push(opc + (hw & 7));
        }
        Enc::PlusR2 { opc, reg_slot } => {
            let hw = ops_reg(inst, reg_slot);
            if inst.width == Width::W16 {
                out.bytes.push(0x66);
            }
            emit_rex_f(out, inst.width == Width::W64, false, false, rex_bit(hw), false);
            out.bytes.push(0x0F);
            out.bytes.push(opc + (hw & 7));
        }
        Enc::PlusRI { opc, reg_slot, imm_slot } => {
            let hw = ops_reg(inst, reg_slot);
            if inst.width == Width::W16 {
                out.bytes.push(0x66);
            }
            emit_rex_f(out, inst.width == Width::W64, false, false, rex_bit(hw), false);
            out.bytes.push(opc + (hw & 7));
            let size = match inst.width {
                Width::W64 => 8,
                Width::W32 => 4,
                Width::W16 => 2,
                Width::W8 => 1,
            };
            emit_imm(out, ops_imm(inst, imm_slot), size);
        }
        Enc::Jcc => {
            // Jcc has no register operands, so no REX can precede the
            // two-byte opcode.
            let cc = ops_imm(inst, 0) as u8;
            let targets = &inst.branch.as_ref().expect("Jcc branch").targets;
            out.bytes.push(0x0F);
            out.bytes.push(0x80 + cc);
            let at = out.bytes.len() as u8;
            out.bytes.extend_from_slice(&[0, 0, 0, 0]);
            out.block_refs.push((at, targets[0]));
            // The not-taken path falls through to whatever the layout
            // places next, which is not necessarily the else block:
            // jump to it explicitly. A branch-layout pass removes
            // this when the else block already follows.
            if targets.len() == 2 {
                out.bytes.push(0xE9);
                let at = out.bytes.len() as u8;
                out.bytes.extend_from_slice(&[0, 0, 0, 0]);
                out.block_refs.push((at, targets[1]));
            }
        }
        Enc::Setcc => {
            let cc = ops_imm(inst, 1) as u8;
            emit_rm(f, inst, out, &[0x0F, 0x90 + cc], 0, 0);
        }
        Enc::Cmov { rm_slot, cc_slot } => {
            let cc = ops_imm(inst, cc_slot) as u8;
            emit_rm(f, inst, out, &[0x0F, 0x40 + cc], ops_reg(inst, 0), rm_slot);
        }
        Enc::Sse { prefix, opc, dir, reg_slot, rm_slot } => {
            if prefix != 0 {
                out.bytes.push(prefix);
            }
            let reg_field = match dir {
                Dir::Digit(d) => d,
                _ => ops_reg(inst, reg_slot),
            };
            emit_rm(f, inst, out, &[0x0F, opc], reg_field, rm_slot);
        }
        Enc::Rel32 { call } => {
            out.bytes.push(if call { 0xE8 } else { 0xE9 });
            let at = out.bytes.len() as u8;
            out.bytes.extend_from_slice(&[0, 0, 0, 0]);
            // The target symbol sits among the pinned operands.
            let sym = match inst.ops.iter().find(|o| matches!(o, Operand::Sym(_))) {
                Some(Operand::Sym(s)) => s.clone(),
                other => panic!("glcc native backend: call without symbol {other:?}"),
            };
            out.relocs.push(RelocReq { at, size: 4, pcrel: true, addend: -4, sym });
        }
        Enc::CallRm => {
            let callee = inst
                .ops
                .iter()
                .find_map(|o| match o {
                    Operand::Reg(Reg::P(p)) => Some(*p),
                    _ => None,
                })
                .expect("call.rm without a register callee");
            emit_rex_f(out, true, false, false, rex_bit(callee.hw), false);
            out.bytes.push(0xFF);
            out.bytes.push(modrm(2, 0b11, callee.hw & 7));
        }
        Enc::Fixed(bytes) => out.bytes.extend_from_slice(bytes),
    }
}

/// Group opcodes with rm + immediate (0x80/0x81/0x83 series, C6/C7
/// moves). The imm8 form of the 0x80 series sign-extends and is used
/// whenever it fits; C7 has no such form. 8-bit operations always take
/// their one-byte immediate forms.
fn encode_grp(f: &MFunction, inst: &MInst, out: &mut EncOut, opc: u8, opc8: u8, digit: u8, rm_slot: u8, imm_slot: u8) {
    let imm = ops_imm(inst, imm_slot);
    let (opcode, imm_size) = match inst.width {
        Width::W8 => (opc8, 1),
        Width::W16 => (opc, 2),
        _ if opc == 0x81 && (-128..=127).contains(&imm) => (0x83, 1),
        _ => (opc, 4),
    };
    emit_rm(f, inst, out, &[opcode], digit, rm_slot);
    emit_imm(out, imm, imm_size);
}

fn pick(width: Width, opc: u8, opc8: u8) -> u8 {
    if width == Width::W8 {
        opc8
    } else {
        opc
    }
}

fn ops_reg(inst: &MInst, slot: u8) -> u8 {
    if slot as usize >= inst.ops.len() {
        panic!("glcc native backend: opcode {:?} expects operand {slot}, has {}", inst.op, inst.ops.len());
    }
    match &inst.ops[slot as usize] {
        Operand::Reg(Reg::P(p)) => p.hw,
        Operand::FixedReg(Reg::P(p), _) => p.hw,
        other => panic!("glcc native backend: register slot has {other:?} at encode time"),
    }
}

fn ops_imm(inst: &MInst, slot: u8) -> i64 {
    match &inst.ops[slot as usize] {
        Operand::Imm(i) => *i,
        other => panic!("glcc native backend: immediate slot has {other:?} at encode time"),
    }
}

fn emit_imm(out: &mut EncOut, value: i64, size: u8) {
    let bytes = value.to_le_bytes();
    out.bytes.extend_from_slice(&bytes[..size as usize]);
}

/// REX bit for a register number: the high half always sets a bit.
/// (8-bit accesses to hw 4..8 also need a REX prefix to reach
/// SIL/DIL/BPL/SPL, but without setting a bit — that is `force`.)
fn rex_bit(hw: u8) -> bool {
    hw >= 8
}

fn emit_rex(out: &mut EncOut, width: Width, r: bool, x: bool, b: bool) {
    emit_rex_f(out, width == Width::W64, r, x, b, false)
}

fn emit_rex_f(out: &mut EncOut, w: bool, r: bool, x: bool, b: bool, force: bool) {
    if !(w || r || x || b || force) {
        return;
    }
    let mut rex = 0x40;
    if w {
        rex |= 0x08;
    }
    if r {
        rex |= 0x04;
    }
    if x {
        rex |= 0x02;
    }
    if b {
        rex |= 0x01;
    }
    out.bytes.push(rex);
}

/// ModRM/SIB/displacement for one operand: a register (rm slot holding
/// a register) or a memory reference. RIP-relative addressing applies
/// when the memory operand has no base. `opcode` is the full opcode
/// sequence (one or two bytes), emitted after the REX prefix.
fn emit_rm(f: &MFunction, inst: &MInst, out: &mut EncOut, opcode: &[u8], reg_field: u8, rm_slot: u8) {
    // Operand-size prefix for 16-bit instructions (legacy prefixes
    // precede REX; the default operand size in long mode is 32-bit).
    if inst.width == Width::W16 {
        out.bytes.push(0x66);
    }
    match &inst.ops[rm_slot as usize] {
        Operand::Mem(m) => emit_mem(f, inst, out, opcode, reg_field, m),
        Operand::Reg(Reg::P(p)) | Operand::FixedReg(Reg::P(p), _) => {
            let hw = p.hw;
            // Byte-register operands need REX even without a bit set
            // (hw 4..8 select SIL/DIL/BPL/SPL only under REX). This
            // includes the sources of 8-bit-extending moves, whose
            // instruction width is the destination's.
            let byte_operand = inst.width == Width::W8 || matches!(op_of(inst.op), Op::Movzx8 | Op::Movsx8);
            let force8 = byte_operand && (hw >= 4 || reg_field >= 4);
            emit_rex_f(out, inst.width == Width::W64, rex_bit(reg_field), false, rex_bit(hw), force8);
            out.bytes.extend_from_slice(opcode);
            out.bytes.push(modrm(reg_field, 0b11, hw & 7));
        }
        other => panic!("glcc native backend: rm slot has {other:?} at encode time"),
    }
}

fn modrm(reg: u8, mode: u8, rm: u8) -> u8 {
    (mode << 6) | ((reg & 7) << 3) | (rm & 7)
}

fn emit_mem(f: &MFunction, inst: &MInst, out: &mut EncOut, opcode: &[u8], reg: u8, m: &crate::machine::MemOperand) {
    let _ = f;

    // RIP-relative (no base, no index): mod=00 rm=101, disp32.
    if m.base.is_none() && m.index.is_none() {
        emit_rex(out, inst.width, rex_bit(reg), false, false);
        out.bytes.extend_from_slice(opcode);
        out.bytes.push(modrm(reg, 0b00, 0b101));
        emit_sym_disp(out, m);
        return;
    }

    let base_hw = match m.base {
        Some(Reg::P(p)) => Some(p.hw),
        Some(other) => panic!("glcc native backend: unresolved memory base {other:?}"),
        None => None,
    };
    let index_hw = match m.index {
        Some(Reg::P(p)) => Some(p.hw),
        Some(other) => panic!("glcc native backend: unresolved memory index {other:?}"),
        None => None,
    };

    let byte_operand = inst.width == Width::W8 || matches!(op_of(inst.op), Op::Movzx8 | Op::Movsx8);
    let force8 = byte_operand && (reg >= 4 || base_hw.is_some_and(|h| h >= 4) || index_hw.is_some_and(|h| h >= 4));
    let b = base_hw.is_some_and(|h| rex_bit(h));
    let x = index_hw.is_some_and(|h| rex_bit(h));
    let r = rex_bit(reg);
    emit_rex_f(out, inst.width == Width::W64, r, x, b, force8);

    match (base_hw, index_hw) {
        (Some(base), index) => {
            let disp = m.disp;
            let (mode, disp_bytes) = if (base & 7) == 5 {
                // RBP/R13 require an explicit displacement.
                if (-128..=127).contains(&disp) { (0b01, 1) } else { (0b10, 4) }
            } else if disp == 0 {
                (0b00, 0)
            } else if (-128..=127).contains(&disp) {
                (0b01, 1)
            } else {
                (0b10, 4)
            };
            out.bytes.extend_from_slice(opcode);
            match index {
                Some(index) => {
                    out.bytes.push(modrm(reg, mode, 0b100));
                    out.bytes.push((m.scale.min(3) << 6) | ((index & 7) << 3) | (base & 7));
                }
                None if (base & 7) == 4 => {
                    // RSP/R12 as base require a SIB byte (rm=100 means
                    // "SIB follows"); the SIB names the real base with
                    // no index.
                    out.bytes.push(modrm(reg, mode, 0b100));
                    out.bytes.push((base & 7) | (0b100 << 3));
                }
                None => {
                    out.bytes.push(modrm(reg, mode, base));
                }
            }
            match disp_bytes {
                0 => {}
                1 => out.bytes.push(disp as i8 as u8),
                _ => out.bytes.extend_from_slice(&disp.to_le_bytes()[..4]),
            }
        }
        (None, Some(index)) => {
            // Index without base: SIB base=101 forces a disp32.
            out.bytes.extend_from_slice(opcode);
            out.bytes.push(modrm(reg, 0b00, 0b100));
            out.bytes.push((m.scale.min(3) << 6) | ((index & 7) << 3) | 0b101);
            out.bytes.extend_from_slice(&m.disp.to_le_bytes()[..4]);
        }
        (None, None) => unreachable!("handled above"),
    }
}

/// RIP-relative displacement field with its relocation (or zero).
fn emit_sym_disp(out: &mut EncOut, m: &crate::machine::MemOperand) {
    let at = out.bytes.len() as u8;
    out.bytes.extend_from_slice(&[0, 0, 0, 0]);
    if let Some(sym) = &m.sym {
        let pcrel = match sym.reloc {
            crate::machine::RelocKind::PcRel32 | crate::machine::RelocKind::Plt32 => true,
            crate::machine::RelocKind::Abs64 => panic!("glcc native backend: 64-bit absolute in a disp32 field"),
        };
        out.relocs.push(RelocReq { at, size: 4, pcrel, addend: -4, sym: sym.clone() });
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
        // Vector-class moves (floats): movss/movsd, register or
        // spill-slot forms. A spill of a float moves the full 8-byte
        // spill unit; consumers only ever read the active low lane.
        (Reg::P(d), Reg::P(s)) if d.class == crate::machine::RegClass::Vec => {
            let prefix = if inst.width == Width::W32 { 0xF3 } else { 0xF2 };
            out.bytes.push(prefix);
            emit_rex_f(out, false, rex_bit(d.hw), false, rex_bit(s.hw), false);
            out.bytes.extend_from_slice(&[0x0F, 0x10]);
            // Unlike the integer move (0x89: reg = source), SSE moves
            // take the destination in the reg field.
            out.bytes.push(modrm(d.hw & 7, 0b11, s.hw & 7));
        }
        (Reg::P(d), Reg::Spill(s)) if d.class == crate::machine::RegClass::Vec => {
            let off = spill_disp(f, s);
            out.bytes.push(if inst.width == Width::W32 { 0xF3 } else { 0xF2 });
            emit_rex_f(out, false, false, false, false, false);
            out.bytes.extend_from_slice(&[0x0F, 0x10]);
            out.bytes.push(modrm(d.hw & 7, 0b10, regs::RBP & 7));
            out.bytes.extend_from_slice(&off.to_le_bytes()[..4]);
        }
        (Reg::Spill(s), Reg::P(src)) if src.class == crate::machine::RegClass::Vec => {
            let off = spill_disp(f, s);
            out.bytes.push(if inst.width == Width::W32 { 0xF3 } else { 0xF2 });
            emit_rex_f(out, false, false, false, rex_bit(src.hw), false);
            out.bytes.extend_from_slice(&[0x0F, 0x11]);
            out.bytes.push(modrm(src.hw & 7, 0b10, regs::RBP & 7));
            out.bytes.extend_from_slice(&off.to_le_bytes()[..4]);
        }
        (Reg::P(d), Reg::P(s)) => {
            let force8 = inst.width == Width::W8 && (s.hw >= 4 || d.hw >= 4);
            if inst.width == Width::W16 {
                out.bytes.push(0x66);
            }
            emit_rex_f(out, inst.width == Width::W64, rex_bit(s.hw), false, rex_bit(d.hw), force8);
            out.bytes.push(pick(inst.width, 0x89, 0x88));
            out.bytes.push(modrm(s.hw, 0b11, d.hw & 7));
        }
        (Reg::P(d), Reg::Spill(s)) => {
            // Reload: mov d, [rbp - off]
            let off = spill_disp(f, s);
            if inst.width == Width::W16 {
                out.bytes.push(0x66);
            }
            emit_rex(out, inst.width, rex_bit(d.hw), false, false);
            out.bytes.push(pick(inst.width, 0x8B, 0x8A));
            out.bytes.push(modrm(d.hw, 0b10, regs::RBP & 7));
            out.bytes.extend_from_slice(&off.to_le_bytes()[..4]);
        }
        (Reg::Spill(s), Reg::P(src)) => {
            // Spill: mov [rbp - off], src
            let off = spill_disp(f, s);
            if inst.width == Width::W16 {
                out.bytes.push(0x66);
            }
            emit_rex(out, inst.width, rex_bit(src.hw), false, false);
            out.bytes.push(pick(inst.width, 0x89, 0x88));
            out.bytes.push(modrm(src.hw, 0b10, regs::RBP & 7));
            out.bytes.extend_from_slice(&off.to_le_bytes()[..4]);
        }
        _ => panic!("glcc native backend: unresolved MOVE operand ({dst:?}, {src:?})"),
    }
}

fn spill_disp(f: &MFunction, slot: u32) -> i64 {
    let index = f.frame.spill_start as usize + slot as usize;
    -(f.frame.slots[index].offset as i64)
}
