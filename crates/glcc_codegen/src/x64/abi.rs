//! SysV x86-64 calling convention. Scalar integer/pointer parameters
//! and returns only so far; aggregate classification (eightbyte
//! voting), floats in XMM, and the register-save-area `va_list` arrive
//! with the float/aggregate milestone — ported from
//! `glcc_lower_cl/src/abi` then.

use glcc_mir::Ty;

use crate::isa::{AbiImpl, ParamLoc, RetLoc};
use crate::machine::{MFunction, MInst, Operand, PReg, Reg};

use super::regs;
use super::tables::{Op, op_id};

pub struct SysV;

impl AbiImpl for SysV {
    fn ret_loc(&self, ty: &Ty) -> RetLoc {
        match ty {
            Ty::Void => RetLoc::Void,
            Ty::Int { bits } | Ty::Ptr { bits, .. } if *bits <= 64 => RetLoc::Reg(PReg::gpr(regs::RAX)),
            Ty::Bool => RetLoc::Reg(PReg::gpr(regs::RAX)),
            _ => panic!("glcc native backend: return type {ty:?} not yet supported"),
        }
    }

    fn param_loc(&self, idx: usize, ty: &Ty) -> ParamLoc {
        match ty {
            Ty::Int { bits } | Ty::Ptr { bits, .. } if *bits <= 64 => {
                if idx < regs::ARG_REGS.len() {
                    ParamLoc::Reg(regs::ARG_REGS[idx])
                } else {
                    ParamLoc::Stack
                }
            }
            Ty::Bool => self.param_loc(idx, &Ty::Int { bits: 8 }),
            _ => panic!("glcc native backend: parameter type {ty:?} not yet supported"),
        }
    }

    fn arg_regs(&self) -> &'static [PReg] {
        regs::ARG_REGS
    }

    fn ret_reg(&self) -> PReg {
        PReg::gpr(regs::RAX)
    }

    fn caller_saved(&self) -> &'static [PReg] {
        regs::CALLER_SAVED
    }

    fn emit_prologue(&self, f: &mut MFunction) {
        let entry = &mut f.blocks[0];
        let mut pre = Vec::new();
        pre.push(MInst::new(op_id(Op::Push), crate::machine::Width::W64, vec![Operand::Reg(Reg::P(PReg::gpr(regs::RBP)))]));
        pre.push(MInst::new(
            op_id(Op::MovRR),
            crate::machine::Width::W64,
            vec![Operand::Reg(Reg::P(PReg::gpr(regs::RBP))), Operand::Reg(Reg::P(PReg::gpr(regs::RSP)))],
        ));
        if f.frame.frame_size > 0 {
            pre.push(MInst::new(
                op_id(Op::SubI),
                crate::machine::Width::W64,
                vec![
                    Operand::Reg(Reg::P(PReg::gpr(regs::RSP))),
                    Operand::Reg(Reg::P(PReg::gpr(regs::RSP))),
                    Operand::Imm(f.frame.frame_size as i64),
                ],
            ));
        }
        entry.insts.splice(0..0, pre);
    }

    fn insert_epilogues(&self, f: &mut MFunction) {
        for block in &mut f.blocks {
            for i in 0..block.insts.len() {
                if block.insts[i].op == crate::machine::OpId::RET {
                    // leave: mov rsp, rbp; pop rbp
                    let restore = vec![
                        MInst::new(
                            op_id(Op::MovRR),
                            crate::machine::Width::W64,
                            vec![Operand::Reg(Reg::P(PReg::gpr(regs::RSP))), Operand::Reg(Reg::P(PReg::gpr(regs::RBP)))],
                        ),
                        MInst::new(op_id(Op::Pop), crate::machine::Width::W64, vec![Operand::Reg(Reg::P(PReg::gpr(regs::RBP)))]),
                    ];
                    block.insts.splice(i..i, restore);
                    break;
                }
            }
        }
    }
}
