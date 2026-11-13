//! SysV x86-64 calling convention: scalars in integer/XMM registers,
//! by-value aggregates as one or two eightbyte pieces, stack for the
//! rest, and the register-save-area `va_list` for variadic definitions.

use glcc_mir::Ty;

use crate::isa::{AbiImpl, ParamLoc, RetLoc};
use crate::machine::{MFunction, MInst, Operand, PReg, Reg, RegClass};

use super::regs;
use super::tables::{Op, op_id};

pub struct SysV;

impl AbiImpl for SysV {
    fn ret_loc(&self, ty: &Ty) -> RetLoc {
        match ty {
            Ty::Void => RetLoc::Void,
            Ty::Int { bits } | Ty::Ptr { bits, .. } if *bits <= 64 => RetLoc::Reg(PReg::gpr(regs::RAX)),
            Ty::Bool => RetLoc::Reg(PReg::gpr(regs::RAX)),
            Ty::Float(_) => RetLoc::Reg(PReg::vec(0)),
            Ty::Struct { .. } if ty.size_bytes() <= 16 => {
                let Some(pieces) = struct_pieces(ty) else {
                    return RetLoc::Memory;
                };
                let all_int = pieces.iter().all(|(c, _)| *c == RegClass::Gpr);
                let all_vec = pieces.iter().all(|(c, _)| *c == RegClass::Vec);
                let mut pregs = [PReg::gpr(regs::RAX); 4];
                if all_vec {
                    pregs[0] = PReg::vec(0);
                    pregs[1] = PReg::vec(1);
                } else if !all_int {
                    // Mixed rax/xmm0 returns: rare; defer until needed.
                    return RetLoc::Memory;
                } else if pieces.len() > 1 {
                    // SysV integer pieces: rax then rdx.
                    pregs[1] = PReg::gpr(regs::RDX);
                }
                RetLoc::Regs(pregs, pieces.len() as u8)
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
            // parameters decay to a pointer to the first element. Both
            // ride the integer registers.
            Ty::BuiltinVaList { .. } | Ty::Array { .. } => {
                if gp_used < regs::ARG_REGS.len() {
                    ParamLoc::Reg(regs::ARG_REGS[gp_used])
                } else {
                    ParamLoc::Stack
                }
            }
            // By-value aggregates: one or two eightbyte pieces with
            // per-eightbyte INTEGER/SSE classification (an eightbyte
            // containing any integer field is INTEGER).
            Ty::Struct { .. } if ty.size_bytes() <= 16 => {
                let Some(pieces) = struct_pieces(ty) else {
                    return ParamLoc::Stack;
                };
                let gp_need = pieces.iter().filter(|(c, _)| *c == RegClass::Gpr).count();
                let fp_need = pieces.len() - gp_need;
                if gp_used + gp_need > regs::ARG_REGS.len() || fp_used + fp_need > regs::VEC_ARG_REGS.len() {
                    return ParamLoc::Stack;
                }
                let mut pregs = [regs::ARG_REGS[0]; 4];
                let (mut g, mut f) = (gp_used, fp_used);
                for (i, (class, _)) in pieces.iter().enumerate() {
                    pregs[i] = if *class == RegClass::Gpr {
                        let p = regs::ARG_REGS[g];
                        g += 1;
                        p
                    } else {
                        let p = regs::VEC_ARG_REGS[f];
                        f += 1;
                        p
                    };
                }
                ParamLoc::Regs(pregs, pieces.len() as u8)
            }
            // Larger aggregates travel on the stack, never split.
            Ty::Struct { .. } => ParamLoc::Stack,
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
        PReg::gpr(regs::RAX)
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
        let mut pre = Vec::new();
        pre.push(MInst::new(op_id(Op::Push), crate::machine::Width::W64, vec![Operand::Reg(Reg::P(PReg::gpr(regs::RBP)))]));
        pre.push(MInst::new(
            op_id(Op::MovRR),
            crate::machine::Width::W64,
            vec![Operand::Reg(Reg::P(PReg::gpr(regs::RBP))), Operand::Reg(Reg::P(PReg::gpr(regs::RSP)))],
        ));
        for p in &saved {
            pre.push(MInst::new(op_id(Op::Push), crate::machine::Width::W64, vec![Operand::Reg(Reg::P(*p))]));
        }
        // Keep rsp 16-aligned at call sites: an odd number of pushes
        // (frame pointer + callee-saved) leaves it off by 8, so pad the
        // frame subtraction to compensate.
        let pad = if (8 * (1 + saved.len() as u64)) % 16 == 0 { 8 } else { 0 };
        let sub = f.frame.frame_size as i64 + pad;
        if sub > 0 {
            pre.push(MInst::new(
                op_id(Op::SubI),
                crate::machine::Width::W64,
                vec![
                    Operand::Reg(Reg::P(PReg::gpr(regs::RSP))),
                    Operand::Reg(Reg::P(PReg::gpr(regs::RSP))),
                    Operand::Imm(sub),
                ],
            ));
        }
        entry.insts.splice(0..0, pre);
    }

    fn insert_epilogues(&self, f: &mut MFunction) {
        let saved = used_callee_saved(f);
        for block in &mut f.blocks {
            for i in 0..block.insts.len() {
                if block.insts[i].op == crate::machine::OpId::RET {
                    // Restore rsp to the callee-saved pushes (they sit
                    // directly below the frame pointer), pop them in
                    // reverse push order, then leave.
                    let mut restore: Vec<MInst> = Vec::new();
                    if !saved.is_empty() {
                        restore.push(MInst::new(
                            op_id(Op::Lea),
                            crate::machine::Width::W64,
                            vec![
                                Operand::Reg(Reg::P(PReg::gpr(regs::RSP))),
                                Operand::Mem(crate::machine::MemOperand {
                                    base: Some(Reg::P(PReg::gpr(regs::RBP))),
                                    index: None,
                                    scale: 0,
                                    disp: -(8 * saved.len() as i64),
                                    sym: None,
                                }),
                            ],
                        ));
                    }
                    restore.extend(saved.iter().rev().map(|p| MInst::new(op_id(Op::Pop), crate::machine::Width::W64, vec![Operand::Reg(Reg::P(*p))])));
                    restore.push(MInst::new(
                        op_id(Op::MovRR),
                        crate::machine::Width::W64,
                        vec![Operand::Reg(Reg::P(PReg::gpr(regs::RSP))), Operand::Reg(Reg::P(PReg::gpr(regs::RBP)))],
                    ));
                    restore.push(MInst::new(op_id(Op::Pop), crate::machine::Width::W64, vec![Operand::Reg(Reg::P(PReg::gpr(regs::RBP)))]));
                    block.insts.splice(i..i, restore);
                    break;
                }
            }
        }
    }
}

/// The callee-saved registers this function actually uses (post-RA:
/// every live reference is a physical register). The frame layout
/// reserves space under the frame pointer for exactly these.
fn used_callee_saved(f: &MFunction) -> Vec<PReg> {
    let mut used = [false; 16];
    let mut scan = |r: &Reg| {
        if let Reg::P(p) = r {
            let info = super::tables::REGINFO.info(*p);
            if info.callee_saved {
                used[p.hw as usize] = true;
            }
        }
    };
    for b in &f.blocks {
        for inst in &b.insts {
            if let Some(blob) = &inst.asm {
                // Registers the template clobbers or mentions need the
                // caller's value saved across the function.
                for &p in &blob.clobbers {
                    scan(&Reg::P(p));
                }
            }
            for op in &inst.ops {
                match op {
                    Operand::Reg(r) | Operand::FixedReg(r, _) => scan(r),
                    Operand::Mem(m) => {
                        if let Some(r) = m.base {
                            scan(&r);
                        }
                        if let Some(r) = m.index {
                            scan(&r);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    [regs::RBX, regs::R12, regs::R13, regs::R14, regs::R15]
        .into_iter()
        .filter(|&hw| used[hw as usize])
        .map(|hw| PReg::gpr(hw))
        .collect()
}

/// SysV eightbyte classification of a small by-value struct: for each
/// 8-byte chunk, SSE if any floating field overlaps it, else INTEGER
/// (integer wins ties). `None` when the layout is not representable.
pub(crate) fn struct_pieces(ty: &Ty) -> Option<Vec<(crate::machine::RegClass, u64)>> {
    let Ty::Struct { fields, offsets, .. } = ty else { return None };
    let size = ty.size_bytes();
    let mut pieces = Vec::new();
    let mut base = 0u64;
    while base < size {
        let end = (base + 8).min(size);
        let mut sse = true;
        for (f, off) in fields.iter().zip(offsets.iter()) {
            let f_end = off + f.size_bytes();
            if *off < end && f_end > base && !matches!(f, Ty::Float(_)) {
                sse = false;
            }
        }
        let class = if sse { crate::machine::RegClass::Vec } else { crate::machine::RegClass::Gpr };
        pieces.push((class, end - base));
        base = end;
    }
    Some(pieces)
}
