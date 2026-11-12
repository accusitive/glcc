//! AAPCS64 calling convention: scalars in x0-x7 / v0-v7, homogeneous
//! float aggregates in V registers at member width, other small
//! composites as 8-byte chunks in X registers, larger composites by
//! reference (the caller copies; the pointer rides an X register), and
//! memory-class returns through the indirect-result register x8 (which
//! is not part of the x0-x7 argument pool).

use glcc_mir::Ty;

use crate::isa::{AbiImpl, ParamLoc, RetLoc};
use crate::machine::{MFunction, MInst, Operand, PReg, Reg};

use super::regs;
use super::tables::{Op, op_id};

pub struct Aapcs64;

impl AbiImpl for Aapcs64 {
    fn ret_loc(&self, ty: &Ty) -> RetLoc {
        match ty {
            Ty::Void => RetLoc::Void,
            Ty::Int { bits } | Ty::Ptr { bits, .. } if *bits <= 64 => RetLoc::Reg(PReg::gpr(0)),
            Ty::Bool => RetLoc::Reg(PReg::gpr(0)),
            // F128 returns in v0 per AAPCS64; its arithmetic panics
            // elsewhere until the quad libcall pass.
            Ty::Float(_) => RetLoc::Reg(PReg::vec(0)),
            Ty::Struct { .. } if ty.size_bytes() <= 16 => {
                if let Some(pieces) = hfa_pieces(ty) {
                    let mut pregs = [PReg::vec(0); 2];
                    for (i, _) in pieces.iter().enumerate() {
                        pregs[i] = PReg::vec(i as u8);
                    }
                    return RetLoc::Regs(pregs, pieces.len() as u8);
                }
                let n = ty.size_bytes().div_ceil(8) as usize;
                let mut pregs = [PReg::gpr(0); 2];
                for i in 0..n {
                    pregs[i] = PReg::gpr(i as u8);
                }
                RetLoc::Regs(pregs, n as u8)
            }
            Ty::Struct { .. } => RetLoc::Memory,
            _ => panic!("glcc native backend: return type {ty:?} not yet supported"),
        }
    }

    fn param_loc(&self, gp_used: usize, fp_used: usize, ty: &Ty) -> ParamLoc {
        match ty {
            Ty::Int { bits } | Ty::Ptr { bits, .. } if *bits <= 64 => {
                if gp_used < regs::ARG_REGS.len() {
                    ParamLoc::Reg(regs::ARG_REGS[gp_used])
                } else {
                    ParamLoc::Stack
                }
            }
            Ty::Bool => self.param_loc(gp_used, fp_used, &Ty::Int { bits: 8 }),
            Ty::Float(_) => {
                if fp_used < regs::VEC_ARG_REGS.len() {
                    ParamLoc::Reg(regs::VEC_ARG_REGS[fp_used])
                } else {
                    ParamLoc::Stack
                }
            }
            // va_list decays to a pointer to the caller's tag; array
            // parameters decay to a pointer to the first element.
            Ty::BuiltinVaList { .. } | Ty::Array { .. } => {
                if gp_used < regs::ARG_REGS.len() {
                    ParamLoc::Reg(regs::ARG_REGS[gp_used])
                } else {
                    ParamLoc::Stack
                }
            }
            Ty::Struct { .. } if ty.size_bytes() <= 16 => {
                if let Some(pieces) = hfa_pieces(ty) {
                    if fp_used + pieces.len() > regs::VEC_ARG_REGS.len() {
                        return ParamLoc::Stack;
                    }
                    let mut pregs = [regs::VEC_ARG_REGS[0]; 2];
                    for i in 0..pieces.len() {
                        pregs[i] = regs::VEC_ARG_REGS[fp_used + i];
                    }
                    return ParamLoc::Regs(pregs, pieces.len() as u8);
                }
                let chunks = ty.size_bytes().div_ceil(8) as usize;
                if gp_used + chunks > regs::ARG_REGS.len() {
                    return ParamLoc::Stack;
                }
                let mut pregs = [regs::ARG_REGS[0]; 2];
                for i in 0..chunks {
                    pregs[i] = regs::ARG_REGS[gp_used + i];
                }
                ParamLoc::Regs(pregs, chunks as u8)
            }
            // Larger composites cross by reference: one GP slot holds
            // the pointer; the caller makes the copy (the call hook).
            Ty::Struct { .. } => {
                if gp_used < regs::ARG_REGS.len() {
                    ParamLoc::Reg(regs::ARG_REGS[gp_used])
                } else {
                    ParamLoc::Stack
                }
            }
            _ => panic!("glcc native backend: parameter type {ty:?} not yet supported"),
        }
    }

    fn arg_regs(&self) -> &'static [PReg] {
        regs::ARG_REGS
    }

    fn vec_arg_regs(&self) -> &'static [PReg] {
        regs::VEC_ARG_REGS
    }

    fn ret_reg(&self) -> PReg {
        PReg::gpr(0)
    }

    fn vec_ret_reg(&self) -> PReg {
        PReg::vec(0)
    }

    fn caller_saved(&self) -> &'static [PReg] {
        regs::ALL_CALLER_SAVED
    }

    fn emit_prologue(&self, f: &mut MFunction) {
        let saved = used_callee_saved(f);
        let entry = &mut f.blocks[0];
        let mut pre: Vec<MInst> = Vec::new();
        let sp = PReg::gpr(regs::SP);
        let fp = PReg::gpr(regs::X29);
        let lr = PReg::gpr(regs::X30);
        // Reserve the frame-pointer/link pair, store it, and aim x29
        // at it. Stack arguments live at x29+16 from here on.
        pre.push(sub_i(sp, sp, 16));
        pre.push(MInst::new(
            op_id(Op::StpI),
            crate::machine::Width::W64,
            vec![Operand::Reg(Reg::P(fp)), Operand::Reg(Reg::P(lr)), Operand::Mem(sp_mem(0))],
        ));
        pre.push(add_i(fp, sp, 0));
        // Body frame below the pair: frame_size counts from x29 down.
        let rest = f.frame.frame_size as i64 - 16;
        if rest > 0 {
            pre.push(sub_i(sp, sp, rest));
        }
        // Callee-saved registers sit at [x29 - 8*(i+1)], inside the
        // space frame.rs reserved (slots start below x29-8-8n).
        for (i, preg) in saved.iter().enumerate() {
            let mem = crate::machine::MemOperand {
                base: Some(Reg::P(fp)),
                index: None,
                scale: 0,
                disp: -8 * (i as i64 + 1),
                sym: None,
            };
            let op = if preg.class == crate::machine::RegClass::Vec { Op::FStore } else { Op::Store };
            pre.push(MInst::new(
                op_id(op),
                crate::machine::Width::W64,
                vec![Operand::Reg(Reg::P(*preg)), Operand::Mem(mem)],
            ));
        }
        entry.insts.splice(0..0, pre);
    }

    fn insert_epilogues(&self, f: &mut MFunction) {
        let saved = used_callee_saved(f);
        let sp = PReg::gpr(regs::SP);
        let fp = PReg::gpr(regs::X29);
        for block in &mut f.blocks {
            for i in 0..block.insts.len() {
                if block.insts[i].op != crate::machine::OpId::RET {
                    continue;
                }
                let mut restore: Vec<MInst> = Vec::new();
                for (k, preg) in saved.iter().enumerate().rev() {
                    let mem = crate::machine::MemOperand {
                        base: Some(Reg::P(fp)),
                        index: None,
                        scale: 0,
                        disp: -8 * (k as i64 + 1),
                        sym: None,
                    };
                    let op = if preg.class == crate::machine::RegClass::Vec { Op::FLoad } else { Op::Load };
                    restore.push(MInst::new(
                        op_id(op),
                        crate::machine::Width::W64,
                        vec![Operand::Reg(Reg::P(*preg)), Operand::Mem(mem)],
                    ));
                }
                // mov sp, x29; ldp x29, x30, [sp], #16 — the pair is
                // read from x29's slot while sp still points at it.
                restore.push(add_i(sp, fp, 0));
                restore.push(MInst::new(
                    op_id(Op::LdpPost),
                    crate::machine::Width::W64,
                    vec![Operand::Reg(Reg::P(fp)), Operand::Reg(Reg::P(PReg::gpr(regs::X30))), Operand::Mem(sp_mem(16))],
                ));
                block.insts.splice(i..i, restore);
                break;
            }
        }
    }
}

fn sub_i(d: PReg, a: PReg, imm: i64) -> MInst {
    MInst::new(
        op_id(Op::SubI),
        crate::machine::Width::W64,
        vec![Operand::Reg(Reg::P(d)), Operand::Reg(Reg::P(a)), Operand::Imm(imm)],
    )
}

fn add_i(d: PReg, a: PReg, imm: i64) -> MInst {
    MInst::new(
        op_id(Op::AddI),
        crate::machine::Width::W64,
        vec![Operand::Reg(Reg::P(d)), Operand::Reg(Reg::P(a)), Operand::Imm(imm)],
    )
}

fn sp_mem(disp: i64) -> crate::machine::MemOperand {
    crate::machine::MemOperand { base: Some(Reg::P(PReg::gpr(regs::SP))), index: None, scale: 0, disp, sym: None }
}

/// The callee-saved registers this (post-RA) function actually uses, in
/// save order. Mirrors the generic scan in `frame.rs` but returns the
/// registers themselves so the prologue can store them by name.
fn used_callee_saved(f: &MFunction) -> Vec<PReg> {
    let mut used = std::collections::HashSet::<(u8, u8)>::new();
    for b in &f.blocks {
        for inst in &b.insts {
            for op in &inst.ops {
                match op {
                    Operand::Reg(Reg::P(p)) | Operand::FixedReg(Reg::P(p), _) => {
                        if super::tables::REGINFO.info(*p).callee_saved {
                            used.insert((p.hw, p.class as u8));
                        }
                    }
                    Operand::Mem(m) => {
                        for r in [m.base, m.index] {
                            if let Some(Reg::P(p)) = r {
                                if super::tables::REGINFO.info(p).callee_saved {
                                    used.insert((p.hw, p.class as u8));
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    regs::CALLEE_SAVED.iter().copied().filter(|p| used.contains(&(p.hw, p.class as u8))).collect()
}

/// HFA member pieces (up to four same-width float members, 16-byte
/// alignment defeats it), or `None`. Returns the member byte widths.
pub(crate) fn hfa_pieces(ty: &Ty) -> Option<Vec<u64>> {
    if glcc_mir::ty::ty_align(ty) == 16 {
        return None;
    }
    let mut leaves: Vec<(u64, u64, bool)> = Vec::new(); // (offset, size, float)
    leaves_of(ty, 0, &mut leaves);
    if leaves.is_empty() || leaves.len() > 4 {
        return None;
    }
    let width = leaves[0].1;
    if width != 4 && width != 8 {
        return None;
    }
    let mut off = 0u64;
    let mut pieces = Vec::new();
    for (leaf_off, len, float) in &leaves {
        if !float || *len != width || *leaf_off != off {
            return None;
        }
        pieces.push(*len);
        off += len;
    }
    Some(pieces)
}

/// Flatten a (possibly nested) aggregate into its leaf fields.
fn leaves_of(ty: &Ty, base: u64, out: &mut Vec<(u64, u64, bool)>) {
    match ty {
        Ty::Struct { fields, offsets, .. } => {
            for (f, off) in fields.iter().zip(offsets.iter()) {
                leaves_of(f, base + off, out);
            }
        }
        Ty::Array { elem, len } => {
            let align = glcc_mir::ty::ty_align(elem).max(1) as u64;
            let stride = elem.size_bytes().next_multiple_of(align).max(align);
            for i in 0..*len {
                leaves_of(elem, base + i * stride, out);
            }
        }
        t => out.push((base, t.size_bytes(), matches!(t, Ty::Float(glcc_mir::FloatKind::F32 | glcc_mir::FloatKind::F64)))),
    }
}

/// Whether `ty` is an aggregate crossing by reference (>16 bytes).
pub(crate) fn by_reference(ty: &Ty) -> bool {
    matches!(ty, Ty::Struct { .. }) && ty.size_bytes() > 16
}
