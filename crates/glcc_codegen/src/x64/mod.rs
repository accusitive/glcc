//! x86-64 target: SysV ABI (Linux ELF), table-driven selection with
//! hooks for the shapes patterns cannot express (calls, control flow,
//! address computation, fusion).

pub mod abi;
pub mod enc;
pub mod regs;
pub mod tables;

use glcc_mir::{Block, Const, Def, Function, GepIndex, Inst, InstKind, Terminator, Ty, Value};

use crate::isa::{EncOut, ParamLoc, RetLoc, SelItem, SlotSpec, TargetIsa};
use crate::machine::{BlockId, MFunction, MInst, OpId, Operand, PReg, Reg, SymOperand, SymTarget, Width};
use crate::sel::{EmitWidth, FuncSel, value_width};

use tables::{Op, op_id};

pub struct X64;

pub static X64_ISA: X64 = X64;

/// Condition-code field value (0F 8x / 0F 9x / 0F 4x families).
fn icc(pred: glcc_mir::IcmpPred) -> u8 {
    use glcc_mir::IcmpPred::*;
    match pred {
        Eq => 0x4,
        Ne => 0x5,
        Ult => 0x2,
        Ule => 0x6,
        Ugt => 0x7,
        Uge => 0x3,
        Slt => 0xC,
        Sle => 0xE,
        Sgt => 0xF,
        Sge => 0xD,
    }
}

fn swap(pred: glcc_mir::IcmpPred) -> glcc_mir::IcmpPred {
    use glcc_mir::IcmpPred::*;
    match pred {
        Eq => Eq,
        Ne => Ne,
        Ult => Ugt,
        Ule => Uge,
        Ugt => Ult,
        Uge => Ule,
        Slt => Sgt,
        Sle => Sge,
        Sgt => Slt,
        Sge => Sle,
    }
}

impl TargetIsa for X64 {
    fn name(&self) -> &'static str {
        "x86_64"
    }

    fn reginfo(&self) -> &'static crate::isa::RegInfo {
        &tables::REGINFO
    }

    fn opcodes(&self) -> &'static crate::isa::OpcodeTable {
        &tables::OPCODES
    }

    fn patterns(&self) -> &'static crate::sel::PatternTable {
        &tables::PATTERNS
    }

    fn abi(&self) -> &'static dyn crate::isa::AbiImpl {
        &abi::SysV
    }

    fn const_move_op(&self, value: i64) -> OpId {
        if (-2147483648i64..=2147483647).contains(&value) {
            op_id(Op::MovRI)
        } else {
            op_id(Op::MovRI64)
        }
    }

    fn gpr_to_vec_op(&self) -> OpId {
        op_id(Op::MovqG2X)
    }

    fn lea_op(&self) -> OpId {
        op_id(Op::Lea)
    }

    fn select_custom(&self, sel: &mut FuncSel, mir: &Function, item: SelItem) -> bool {
        match item {
            SelItem::Entry => entry(sel, mir),
            SelItem::Inst(inst) => inst_hook(sel, mir, inst),
            SelItem::Term(block) => term_hook(sel, mir, block),
        }
    }

    fn encode_inst(&self, f: &MFunction, inst: &crate::machine::MInst, out: &mut EncOut) {
        enc::encode(f, inst, out);
    }
}

fn entry(sel: &mut FuncSel, mir: &Function) -> bool {
    // SysV variadic definitions spill every register-passed named
    // argument into a 176-byte save area (GP at +0, stride 8; FP at
    // +48, stride 16) and point the overflow cursor past the stack-
    // passed named arguments. va_list is {i32 gp_off, i32 fp_off, ptr
    // overflow, ptr reg_save_area}.
    let mut save_slot = 0u32;
    let mut gp_off = 0i64;
    let mut fp_off = 48i64;
    let mut gp_save: Vec<(i64, crate::machine::VReg, bool)> = Vec::new();
    let mut fp_save: Vec<(i64, crate::machine::VReg)> = Vec::new();
    if mir.sig.variadic {
        save_slot = sel.slot(176, 16);
        sel.va_save_slot = Some(save_slot);
        sel.va_overflow_disp = 16;
    }
    let entry_params: Vec<Value> = mir.blocks[mir.entry].params.clone();
    let mut gp = 0;
    let mut fp = 0;
    let mut stack_off = 0i64;
    // Memory-class aggregate returns take a hidden out-pointer in the
    // first integer argument register.
    if matches!(sel.isa.abi().ret_loc(&mir.sig.ret), RetLoc::Memory) {
        let vr = sel.alloc_vreg();
        sel.emit(MInst::new(OpId::MOVE, Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(vr)), Operand::Reg(Reg::P(PReg::gpr(regs::RDI)))]));
        sel.sret_loc = Some(vr);
        gp = 1;
    }
    // Stack-argument homing copies bytes through scratch registers; a
    // register-classed parameter still living in its ABI register must
    // be pinned first, or the allocator may hand the scratch one of
    // those registers. So: pin every register param, then home the
    // stack ones (offsets computed in signature order either way).
    let mut stack_params: Vec<(Value, glcc_mir::Ty, i64)> = Vec::new();
    let (gp_init, fp_init) = (gp, fp);
    for &p in entry_params.iter() {
        let ty = mir.values[p].ty.clone();
        match sel.isa.abi().param_loc(gp, fp, &ty) {
            ParamLoc::Stack => {
                let align = glcc_mir::ty::ty_align(&ty).max(1).max(8) as i64;
                stack_off = (stack_off + align - 1) / align * align;
                if mir.sig.variadic {
                    sel.va_overflow_disp = 16 + stack_off + ty.size_bytes().next_multiple_of(8) as i64;
                }
                let rounded = ty.size_bytes().next_multiple_of(8) as i64;
                stack_params.push((p, ty, stack_off));
                stack_off += rounded;
            }
            ParamLoc::Reg(preg) => match preg.class {
                crate::machine::RegClass::Gpr => gp += 1,
                crate::machine::RegClass::Vec => fp += 1,
            },
            ParamLoc::Regs(pregs, n) => {
                for preg in pregs.iter().take(n as usize) {
                    match preg.class {
                        crate::machine::RegClass::Gpr => gp += 1,
                        crate::machine::RegClass::Vec => fp += 1,
                    }
                }
            }
        }
    }
    // Restart the register counters for the pin pass: the pre-pass
    // above left them at their final values.
    gp = gp_init;
    fp = fp_init;
    for &p in entry_params.iter() {
        let ty = mir.values[p].ty.clone();
        match sel.isa.abi().param_loc(gp, fp, &ty) {
            ParamLoc::Reg(preg) => {
                let vr = sel.value_vreg(mir, p);
                sel.emit(MInst::new(
                    OpId::MOVE,
                    Width::W64,
                    smallvec::smallvec![Operand::Reg(Reg::V(vr)), Operand::Reg(Reg::P(preg))],
                ));
                match preg.class {
                    crate::machine::RegClass::Gpr => {
                        if mir.sig.variadic {
                            gp_save.push((gp as i64, vr, false));
                        }
                        gp_off += 8;
                        gp += 1;
                    }
                    crate::machine::RegClass::Vec => {
                        if mir.sig.variadic {
                            fp_save.push((fp_off, vr));
                        }
                        fp_off += 16;
                        fp += 1;
                    }
                }
            }
            ParamLoc::Regs(pregs, n) => {
                // Store each piece into a fresh home slot; its address
                // is the parameter value.
                let slot = sel.slot(ty.size_bytes().max(1) as u32, glcc_mir::ty::ty_align(&ty).max(1) as u32);
                sel.set_slot(p, slot);
                for (i, preg) in pregs.iter().take(n as usize).enumerate() {
                    let piece_bytes = (ty.size_bytes() - (i as u64) * 8).min(8);
                    let dst = crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: (i as i64) * 8, sym: None };
                    match preg.class {
                        crate::machine::RegClass::Vec => {
                            let op = if piece_bytes >= 8 { Op::MovsdS } else { Op::MovssS };
                            sel.emit(MInst::new(op_id(op), Width::W64, smallvec::smallvec![Operand::Reg(Reg::P(*preg)), Operand::Mem(dst)]));
                        }
                        _ => {
                            let width = if piece_bytes >= 8 { Width::W64 } else { Width::W32 };
                            sel.emit(MInst::new(op_id(Op::Store), width, smallvec::smallvec![Operand::Reg(Reg::P(*preg)), Operand::Mem(dst)]));
                        }
                    }
                    match preg.class {
                        crate::machine::RegClass::Gpr => gp += 1,
                        crate::machine::RegClass::Vec => fp += 1,
                    }
                }
            }
            ParamLoc::Stack => {
                // Deferred to the homing pass below: its scratch must
                // not run before the register pins are in place.
            }
        }
    }
    for (p, ty, off) in stack_params {
        // The caller stored this argument at rbp+16+off (after its
        // return address and saved frame pointer). Copy it into the
        // parameter's home here.
        if ty.is_aggregate() {
            let slot = sel.slot(ty.size_bytes().max(1) as u32, glcc_mir::ty::ty_align(&ty).max(1) as u32);
            sel.set_slot(p, slot);
            let src = crate::machine::MemOperand {
                base: Some(Reg::P(PReg::gpr(regs::RBP))),
                index: None,
                scale: 0,
                disp: 16 + off,
                sym: None,
            };
            let dst = crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: 0, sym: None };
            copy_bytes(sel, &dst, &src, ty.size_bytes(), glcc_mir::ty::ty_align(&ty).max(1).max(8) as u64);
        } else {
            let vr = sel.value_vreg(mir, p);
            let is_float = matches!(ty, glcc_mir::Ty::Float(_));
            let (op, width) = if is_float {
                (if is_f64(mir, p) { Op::MovsdL } else { Op::MovssL }, Width::W64)
            } else {
                (Op::Load, value_width(&ty))
            };
            sel.emit(MInst::new(
                op_id(op),
                width,
                smallvec::smallvec![
                    Operand::Reg(Reg::V(vr)),
                    Operand::Mem(crate::machine::MemOperand {
                        base: Some(Reg::P(PReg::gpr(regs::RBP))),
                        index: None,
                        scale: 0,
                        disp: 16 + off,
                        sym: None,
                    }),
                ],
            ));
        }
    }
    if mir.sig.variadic {
        // The variadic tail may arrive in any of the argument
        // registers: pin the whole register file (named arguments plus
        // anonymous pads) so the entry moves resolve as one parallel
        // set, then spill every register's value into the save area.
        // Slots for registers the caller did not use hold garbage;
        // va_arg never reads past the offsets it tracks.
        let mut tail_gp: Vec<(usize, crate::machine::VReg)> = Vec::new();
        for k in (gp_off / 8) as usize..regs::ARG_REGS.len() {
            let vr = sel.alloc_vreg();
            sel.emit(MInst::new(
                OpId::MOVE,
                Width::W64,
                smallvec::smallvec![Operand::Reg(Reg::V(vr)), Operand::Reg(Reg::P(regs::ARG_REGS[k]))],
            ));
            tail_gp.push((k, vr));
        }
        let mut tail_fp: Vec<(usize, crate::machine::VReg)> = Vec::new();
        for k in ((fp_off - 48) / 16) as usize..regs::VEC_ARG_REGS.len() {
            let vr = sel.alloc_vreg_class(crate::machine::RegClass::Vec);
            sel.emit(MInst::new(
                OpId::MOVE,
                Width::W64,
                smallvec::smallvec![Operand::Reg(Reg::V(vr)), Operand::Reg(Reg::P(regs::VEC_ARG_REGS[k]))],
            ));
            tail_fp.push((k, vr));
        }
        for (k, vr, float) in gp_save.iter() {
            spill_save_area(sel, save_slot, k * 8, *vr, *float);
        }
        for (k, vr) in tail_gp.iter() {
            spill_save_area(sel, save_slot, (*k * 8) as i64, *vr, false);
        }
        for (disp, vr) in fp_save.iter() {
            spill_save_area(sel, save_slot, *disp, *vr, true);
        }
        for (k, vr) in tail_fp.iter() {
            spill_save_area(sel, save_slot, 48 + (*k as i64) * 16, *vr, true);
        }
        sel.va_gp_off = gp_off.min(48);
        sel.va_fp_off = fp_off.min(176);
    }
    true
}

/// Spill one entry parameter into the va register save area.
fn spill_save_area(sel: &mut FuncSel, save_slot: u32, disp: i64, vr: crate::machine::VReg, float: bool) {
    let (op, width) = if float { (Op::MovsdS, Width::W64) } else { (Op::Store, Width::W64) };
    sel.emit(MInst::new(
        op_id(op),
        width,
        smallvec::smallvec![
            Operand::Reg(Reg::V(vr)),
            Operand::Mem(crate::machine::MemOperand { base: Some(Reg::FI(save_slot)), index: None, scale: 0, disp, sym: None }),
        ],
    ));
}

fn inst_hook(sel: &mut FuncSel, mir: &Function, inst: Inst) -> bool {
    let kind = &mir.insts[inst].kind;
    match kind {
        InstKind::Alloca { pointee, .. } => {
            let res = mir.insts[inst].result.expect("alloca has a result");
            let size = pointee.size_bytes().max(1) as u32;
            let align = glcc_mir::ty::ty_align(pointee).max(1) as u32;
            let slot = sel.slot(size, align);
            // The address materializes here, at the alloca: the
            // frontend places allocas in the entry block, so the
            // definition dominates every use. (Lazy materialization at
            // first use would not, in general.)
            let addr = sel.alloc_vreg();
            sel.emit(MInst::new(
                sel.isa.lea_op(),
                Width::W64,
                smallvec::smallvec![
                    Operand::Reg(Reg::V(addr)),
                    Operand::Mem(crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: 0, sym: None })
                ],
            ));
            sel.set_reg(res, addr);
            true
        }
        InstKind::GetElementPtr { .. } => {
            gep(sel, mir, inst);
            true
        }
        InstKind::FuncAddr { func } => {
            addr(sel, mir, inst, SymTarget::Func(*func));
            true
        }
        InstKind::ExternAddr { target } => {
            addr(sel, mir, inst, SymTarget::Extern(*target));
            true
        }
        InstKind::GlobalAddr { global } => {
            addr(sel, mir, inst, SymTarget::Global(*global));
            true
        }
        InstKind::Call { callee, args } => {
            call(sel, mir, inst, glcc_mir::InvokeCallee::Func(*callee), args, None);
            true
        }
        InstKind::CallExtern { target, args } => {
            call(sel, mir, inst, glcc_mir::InvokeCallee::Extern(*target), args, None);
            true
        }
        InstKind::CallIndirect { callee, args, sig } => {
            call(sel, mir, inst, glcc_mir::InvokeCallee::Indirect(*callee), args, Some(sig.clone()));
            true
        }
        InstKind::Icmp { pred, .. } => {
            if sel.is_fused(mir.insts[inst].result.expect("icmp has a result")) {
                return true;
            }
            let (pred, a, b) = cmp_operands(sel, mir, inst, *pred);
            sel.emit(MInst::new(op_id(Op::CmpRR), icmp_width(mir, inst), smallvec::smallvec![Operand::Reg(Reg::V(a)), Operand::Reg(Reg::V(b))]));
            let dst = sel.value_vreg(mir, mir.insts[inst].result.expect("icmp has a result"));
            sel.emit(MInst::new(op_id(Op::Setcc), Width::W8, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Imm(icc(pred) as i64)]));
            true
        }
        InstKind::Select { .. } => {
            select(sel, mir, inst);
            true
        }
        InstKind::Udiv(..) | InstKind::Sdiv(..) | InstKind::Urem(..) | InstKind::Srem(..) => {
            divide(sel, mir, inst);
            true
        }
        InstKind::Trunc(_) => {
            let dst = result_vreg(sel, mir, inst);
            let src = single_operand(sel, mir, inst);
            // Sub-register copy: the encoder's width picks the move.
            sel.emit(MInst::new(op_id(Op::MovRR), result_width(sel, mir, inst), smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
            true
        }
        InstKind::Zext(_) => {
            let dst = result_vreg(sel, mir, inst);
            let src = single_operand(sel, mir, inst);
            let src_bits = scalar_bits_of(mir, single_value(mir, inst));
            let op = match src_bits {
                b if b <= 8 => Op::Movzx8,
                b if b <= 16 => Op::Movzx16,
                _ => Op::MovRR,
            };
            let width = if src_bits <= 16 { Width::W32 } else { result_width(sel, mir, inst) };
            sel.emit(MInst::new(op_id(op), width, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
            true
        }
        InstKind::Sext(_) => {
            let dst = result_vreg(sel, mir, inst);
            let src = single_operand(sel, mir, inst);
            let src_bits = scalar_bits_of(mir, single_value(mir, inst));
            let op = match src_bits {
                b if b <= 8 => Op::Movsx8,
                b if b <= 16 => Op::Movsx16,
                _ => Op::Movsxd,
            };
            let width = if src_bits <= 16 { Width::W32 } else { Width::W64 };
            sel.emit(MInst::new(op_id(op), width, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
            true
        }
        InstKind::PtrToInt(_) | InstKind::IntToPtr(_) | InstKind::Bitcast(_) => {
            let dst = result_vreg(sel, mir, inst);
            let src_v = single_value(mir, inst);
            let src = single_operand(sel, mir, inst);
            // Float<->integer bitcasts ride the GPR/XMM bit moves.
            let src_float = matches!(mir.values[src_v].ty, glcc_mir::Ty::Float(_));
            let dst_float = matches!(mir.values[mir.insts[inst].result.expect("cast has a result")].ty, glcc_mir::Ty::Float(_));
            let op = match (src_float, dst_float) {
                (false, false) => Op::MovRR,
                (true, false) => Op::MovqX2G,
                (false, true) => Op::MovqG2X,
                (true, true) => {
                    let op = if is_f64(mir, src_v) { Op::MovsdL } else { Op::MovssL };
                    sel.emit(MInst::new(op_id(op), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
                    return true;
                }
            };
            sel.emit(MInst::new(op_id(op), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
            true
        }
        InstKind::Memcpy { dst, src, len } => {
            mem_intrinsic(sel, mir, inst, "memcpy", &[*dst, *src], Some((*len as i64, 0)));
            true
        }
        InstKind::Memset { dst, byte, len } => {
            // The byte is a constant, not a MIR value; it rides in
            // `extra` and mem_intrinsic materializes it into slot 1.
            mem_intrinsic(sel, mir, inst, "memset", &[*dst], Some((*len as i64, *byte as i64)));
            true
        }
        // Relaxed/seqcst scalar load and store are plain moves on
        // x86-64; a full fence is mfence.
        InstKind::AtomicLoad { addr } => {
            let res = mir.insts[inst].result.expect("atomic load has a result");
            if matches!(mir.values[res].ty, glcc_mir::Ty::Float(_)) {
                panic!("glcc native backend: atomic float load is not yet supported");
            }
            let dst = result_vreg(sel, mir, inst);
            let mem = sel.mem_of(mir, *addr);
            sel.emit(MInst::new(op_id(Op::Load), result_width(sel, mir, inst), smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Mem(mem)]));
            true
        }
        InstKind::AtomicStore { addr, val } => {
            let ty = mir.values[*val].ty.clone();
            if matches!(ty, glcc_mir::Ty::Float(_)) {
                panic!("glcc native backend: atomic float store is not yet supported");
            }
            let src = sel.value_vreg(mir, *val);
            let mem = sel.mem_of(mir, *addr);
            sel.emit(MInst::new(op_id(Op::Store), value_width(&ty), smallvec::smallvec![Operand::Reg(Reg::V(src)), Operand::Mem(mem)]));
            true
        }
        InstKind::AtomicFence => {
            sel.emit(MInst::new(op_id(Op::Mfence), Width::W64, smallvec::smallvec![]));
            true
        }
        InstKind::AtomicRmw { .. } | InstKind::AtomicCas { .. } => {
            panic!("glcc native backend: atomic read-modify-write is not yet supported")
        }
        InstKind::InlineAsm { .. } => panic!("glcc native backend: inline asm is not yet supported"),
        InstKind::VaStart { ap } => {
            va_start(sel, mir, inst, *ap);
            true
        }
        InstKind::VaArg { .. } => {
            va_arg(sel, mir, inst);
            true
        }
        InstKind::VaEnd { .. } => true,
        InstKind::VaCopy { dst, src } => {
            // The 24-byte tag: three 8-byte words.
            for k in 0..3i64 {
                let tmp = sel.alloc_vreg();
                let mut sm = sel.mem_of(mir, *src);
                sm.disp += k * 8;
                sel.emit(MInst::new(
                    op_id(Op::Load),
                    Width::W64,
                    smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Mem(sm)],
                ));
                let mut dm = sel.mem_of(mir, *dst);
                dm.disp += k * 8;
                sel.emit(MInst::new(
                    op_id(Op::Store),
                    Width::W64,
                    smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Mem(dm)],
                ));
            }
            true
        }
        InstKind::GetFramePointer => panic!("glcc native backend: GetFramePointer is not yet supported"),
        InstKind::EhTypeId { .. } => panic!("glcc native backend: EH is not yet supported"),
        InstKind::FAdd(..) | InstKind::FSub(..) | InstKind::FMul(..) | InstKind::FDiv(..) | InstKind::FNeg(_) => {
            let (op, a, b) = match kind {
                InstKind::FAdd(a, b) => (Op::AddSS, *a, *b),
                InstKind::FSub(a, b) => (Op::SubSS, *a, *b),
                InstKind::FMul(a, b) => (Op::MulSS, *a, *b),
                InstKind::FDiv(a, b) => (Op::DivSS, *a, *b),
                InstKind::FNeg(_) => return fneg(sel, mir, inst),
                _ => unreachable!(),
            };
            let op = widen_sse(op, is_f64(mir, a));
            let dst = result_vreg(sel, mir, inst);
            let a = sel.value_vreg(mir, a);
            let b = sel.value_vreg(mir, b);
            sel.emit(MInst::new(op_id(op), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(a)), Operand::Reg(Reg::V(b))]));
            true
        }
        InstKind::Sqrt(_) => {
            let src_v = single_value(mir, inst);
            let src = single_operand(sel, mir, inst);
            let op = if is_f64(mir, src_v) { Op::SqrtSD } else { Op::SqrtSS };
            let dst = result_vreg(sel, mir, inst);
            sel.emit(MInst::new(op_id(op), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
            true
        }
        InstKind::Fcmp { .. } => {
            fcmp(sel, mir, inst);
            true
        }
        InstKind::SItoF(_) | InstKind::UItoF(_) => {
            let (dst, src) = (result_vreg(sel, mir, inst), single_operand(sel, mir, inst));
            let to_f64 = matches!(mir.values[mir.insts[inst].result.expect("cast has a result")].ty, glcc_mir::Ty::Float(glcc_mir::FloatKind::F64));
            let bits = scalar_bits_of(mir, single_value(mir, inst));
            match kind {
                InstKind::SItoF(_) => {
                    // cvtsi2ss/sd sign-interpret the source; width
                    // carries the source size (32- or 64-bit).
                    let op = if to_f64 { Op::CvtSi2sd } else { Op::CvtSi2ss };
                    let w = if bits <= 32 { Width::W32 } else { Width::W64 };
                    sel.emit(MInst::new(op_id(op), w, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
                    true
                }
                InstKind::UItoF(_) => {
                    if bits <= 32 {
                        // Zero-extend into a 64-bit register (a 32-bit
                        // move clears the upper half), then treat as
                        // signed 64-bit.
                        let zx = sel.alloc_vreg();
                        sel.emit(MInst::new(op_id(Op::MovRR), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(zx)), Operand::Reg(Reg::V(src))]));
                        let op = if to_f64 { Op::CvtSi2sd } else { Op::CvtSi2ss };
                        sel.emit(MInst::new(op_id(op), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(zx))]));
                        true
                    } else {
                        panic!("glcc native backend: unsigned 64-bit to float conversion is not yet supported")
                    }
                }
                _ => unreachable!(),
            }
        }
        InstKind::SFtoI(_) | InstKind::UFtoI(_) => {
            let (dst, src) = (result_vreg(sel, mir, inst), single_operand(sel, mir, inst));
            let op = if is_f64(mir, single_value(mir, inst)) { Op::CvtSD2si } else { Op::CvtSS2si };
            let dst_bits = scalar_bits_of(mir, mir.insts[inst].result.expect("cast has a result"));
            let w = if dst_bits <= 32 { Width::W32 } else { Width::W64 };
            sel.emit(MInst::new(op_id(op), w, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
            true
        }
        InstKind::Fptrunc(_) | InstKind::Fpext(_) => {
            let (dst, src) = (result_vreg(sel, mir, inst), single_operand(sel, mir, inst));
            let op = match kind {
                InstKind::Fptrunc(_) => Op::CvtSD2ss, // f64 -> f32
                _ => Op::CvtSS2sd,                    // f32 -> f64
            };
            sel.emit(MInst::new(op_id(op), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
            true
        }
        InstKind::Load { addr, .. } => {
            let res = mir.insts[inst].result.expect("load has a result");
            if matches!(mir.values[res].ty, glcc_mir::Ty::Float(_)) {
                load_float(sel, mir, inst);
                return true;
            }
            let ty = mir.values[res].ty.clone();
            if ty.is_aggregate() {
                // Aggregate values are addresses: load into a fresh
                // scratch slot; its address is the value.
                let slot = sel.slot(ty.size_bytes().max(1) as u32, glcc_mir::ty::ty_align(&ty).max(1) as u32);
                sel.set_slot(res, slot);
                let src = sel.mem_of(mir, *addr);
                let dstm = crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: 0, sym: None };
                copy_bytes(sel, &dstm, &src, ty.size_bytes(), glcc_mir::ty::ty_align(&ty).max(1) as u64);
                return true;
            }
            false
        }
        InstKind::Store { addr, val, .. } => {
            let ty = mir.values[*val].ty.clone();
            if matches!(ty, glcc_mir::Ty::Float(_)) {
                store_float(sel, mir, inst, *addr, *val);
                return true;
            }
            if ty.is_aggregate() {
                let src = sel.mem_of(mir, *val);
                let dst = sel.mem_of(mir, *addr);
                copy_bytes(sel, &dst, &src, ty.size_bytes(), glcc_mir::ty::ty_align(&ty).max(1) as u64);
                return true;
            }
            false
        }
        InstKind::ExtractValue { agg, index } => {
            let res = mir.insts[inst].result.expect("extractvalue has a result");
            let agg_ty = mir.values[*agg].ty.clone();
            let Ty::Struct { fields, offsets, .. } = &agg_ty else {
                panic!("glcc native backend: extractvalue through {:?}", agg_ty);
            };
            let off = *offsets.get(*index as usize).expect("extractvalue index in range");
            let field_ty = fields[*index as usize].clone();
            if field_ty.is_aggregate() {
                let slot = sel.slot(field_ty.size_bytes().max(1) as u32, glcc_mir::ty::ty_align(&field_ty).max(1) as u32);
                sel.set_slot(res, slot);
                let src = sel.mem_of(mir, *agg);
                let dstm = crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: 0, sym: None };
                copy_bytes(sel, &dstm, &src, field_ty.size_bytes(), glcc_mir::ty::ty_align(&field_ty).max(1) as u64);
            } else {
                // Scalar field: load from agg_addr + offset.
                let dst = result_vreg(sel, mir, inst);
                let base = sel.mem_of(mir, *agg);
                let (op, width) = if matches!(field_ty, glcc_mir::Ty::Float(_)) {
                    (if matches!(field_ty, glcc_mir::Ty::Float(glcc_mir::FloatKind::F64)) { Op::MovsdL } else { Op::MovssL }, Width::W64)
                } else {
                    (Op::Load, value_width(&field_ty))
                };
                sel.emit(MInst::new(
                    op_id(op),
                    width,
                    smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Mem(crate::machine::MemOperand { disp: base.disp + off as i64, ..base })],
                ));
            }
            true
        }
        InstKind::InsertValue { agg, elem, index } => {
            let res = mir.insts[inst].result.expect("insertvalue has a result");
            let agg_ty = mir.values[*agg].ty.clone();
            let Ty::Struct { fields, offsets, .. } = &agg_ty else {
                panic!("glcc native backend: insertvalue through {:?}", agg_ty);
            };
            let off = *offsets.get(*index as usize).expect("insertvalue index in range");
            let field_ty = fields[*index as usize].clone();
            let slot = sel.slot(agg_ty.size_bytes().max(1) as u32, glcc_mir::ty::ty_align(&agg_ty).max(1) as u32);
            sel.set_slot(res, slot);
            let src = sel.mem_of(mir, *agg);
            let dstm = crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: 0, sym: None };
            if !matches!(mir.values[*agg].def, Def::Const(Const::Undef) | Def::Const(Const::Zero)) {
                copy_bytes(sel, &dstm, &src, agg_ty.size_bytes(), glcc_mir::ty::ty_align(&agg_ty).max(1) as u64);
            }
            let field_src = sel.mem_of(mir, *elem);
            let field_dst = crate::machine::MemOperand { disp: off as i64, ..dstm };
            if field_ty.is_aggregate() {
                copy_bytes(sel, &field_dst, &field_src, field_ty.size_bytes(), glcc_mir::ty::ty_align(&field_ty).max(1) as u64);
            } else {
                let v = sel.value_vreg(mir, *elem);
                let (op, width) = if matches!(field_ty, glcc_mir::Ty::Float(_)) {
                    (if matches!(field_ty, glcc_mir::Ty::Float(glcc_mir::FloatKind::F64)) { Op::MovsdS } else { Op::MovssS }, Width::W64)
                } else {
                    (Op::Store, value_width(&field_ty))
                };
                sel.emit(MInst::new(op_id(op), width, smallvec::smallvec![Operand::Reg(Reg::V(v)), Operand::Mem(field_dst)]));
            }
            true
        }
        // Everything else (ALU, load, store, shifts, ...) goes through
        // the pattern table.
        _ => false,
    }
}

fn term_hook(sel: &mut FuncSel, mir: &Function, block: Block) -> bool {
    let term = mir.blocks[block].term.clone().expect("verified terminator");
    match term {
        Terminator::Return { val } => {
            match val {
                Some(v) => {
                    let ty = mir.values[v].ty.clone();
                    if ty.is_aggregate() {
                        // Pieces of a small aggregate return in the
                        // piece registers (rax/rdx or xmm0/xmm1).
                        match sel.isa.abi().ret_loc(&ty) {
                            RetLoc::Regs(pregs, n) => {
                                let addr = sel.mem_of(mir, v);
                                let mut ops = smallvec::SmallVec::new();
                                for (i, preg) in pregs.iter().take(n as usize).enumerate() {
                                    let bytes = (ty.size_bytes() - (i as u64) * 8).min(8);
                                    let tmp = sel.alloc_vreg_class(if preg.class == crate::machine::RegClass::Vec { crate::machine::RegClass::Vec } else { crate::machine::RegClass::Gpr });
                                    match preg.class {
                                        crate::machine::RegClass::Vec => {
                                            let op = if bytes >= 8 { Op::MovsdL } else { Op::MovssL };
                                            sel.emit(MInst::new(op_id(op), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Mem(crate::machine::MemOperand { disp: addr.disp + (i as i64) * 8, ..addr.clone() })]));
                                        }
                                        _ => {
                                            let width = if bytes >= 8 { Width::W64 } else { Width::W32 };
                                            sel.emit(MInst::new(op_id(Op::Load), width, smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Mem(crate::machine::MemOperand { disp: addr.disp + (i as i64) * 8, ..addr.clone() })]));
                                        }
                                    }
                                    ops.push(Operand::FixedReg(Reg::V(tmp), *preg));
                                }
                                sel.emit(MInst::new(OpId::RET, Width::W64, ops));
                            }
                            RetLoc::Memory => {
                                let sret = sel.sret_loc.expect("sret return without the entry prelude");
                                let src = sel.mem_of(mir, v);
                                let dst = crate::machine::MemOperand { base: Some(Reg::V(sret)), index: None, scale: 0, disp: 0, sym: None };
                                copy_bytes(sel, &dst, &src, ty.size_bytes(), glcc_mir::ty::ty_align(&ty).max(1).max(8) as u64);
                                sel.emit(MInst::new(OpId::RET, Width::W64, smallvec::smallvec![]));
                            }
                            other => panic!("glcc native backend: aggregate return {other:?}"),
                        }
                    } else {
                        let vr = sel.value_vreg(mir, v);
                        sel.emit(MInst::new(OpId::RET, Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(vr))]));
                    }
                }
                None => sel.emit(MInst::new(OpId::RET, Width::W64, smallvec::smallvec![])),
            }
            true
        }
        Terminator::Br { dest, args } => {
            let args: Vec<_> = args.iter().map(|a| sel.value_vreg(mir, *a)).collect();
            sel.emit_branch(MInst::new(OpId::JMP, Width::W64, smallvec::smallvec![]), &[dest.index() as BlockId], &[&args]);
            true
        }
        Terminator::CondBr { cond, then_block, else_block, then_args, else_args } => {
            let then_args: Vec<_> = then_args.iter().map(|a| sel.value_vreg(mir, *a)).collect();
            let else_args: Vec<_> = else_args.iter().map(|a| sel.value_vreg(mir, *a)).collect();
            let then_id = then_block.index() as BlockId;
            let else_id = else_block.index() as BlockId;
            let fused = cond_icc(sel, mir, cond);
            let cc = match fused {
                Some((pred, a, b, width)) => {
                    sel.emit(MInst::new(op_id(Op::CmpRR), width, smallvec::smallvec![Operand::Reg(Reg::V(a)), Operand::Reg(Reg::V(b))]));
                    icc(pred) as i64
                }
                None => {
                    let vr = sel.value_vreg(mir, cond);
                    sel.emit(MInst::new(
                        op_id(Op::TestRR),
                        value_width(&mir.values[cond].ty),
                        smallvec::smallvec![Operand::Reg(Reg::V(vr)), Operand::Reg(Reg::V(vr))],
                    ));
                    0xF // ne: bool is 0/1
                }
            };
            sel.emit_branch(
                MInst::new(op_id(Op::Jcc), Width::W64, smallvec::smallvec![Operand::Imm(cc)]),
                &[then_id, else_id],
                &[&then_args, &else_args],
            );
            true
        }
        Terminator::Switch { scrutinee, cases, default } => {
            // Comparison chain; a later milestone adds jump tables. The
            // default target is every chain step's fallthrough, so no
            // explicit jump into it is emitted.
            if cases.is_empty() {
                sel.emit_branch(MInst::new(OpId::JMP, Width::W64, smallvec::smallvec![]), &[default.index() as BlockId], &[&[]]);
                return true;
            }
            let scrut = sel.value_vreg(mir, scrutinee);
            let scrut_width = value_width(&mir.values[scrutinee].ty);
            let mut next = sel.cur_block();
            for (i, &(value, target)) in cases.iter().enumerate() {
                sel.set_cur(next);
                sel.emit(MInst::new(
                    op_id(Op::CmpI),
                    scrut_width,
                    smallvec::smallvec![Operand::Reg(Reg::V(scrut)), Operand::Imm(value as i64)],
                ));
                let last = i + 1 == cases.len();
                let fallthrough = if last { default.index() as BlockId } else { sel.new_block() };
                sel.emit_branch(
                    MInst::new(op_id(Op::Jcc), Width::W64, smallvec::smallvec![Operand::Imm(icc(glcc_mir::IcmpPred::Eq) as i64)]),
                    &[target.index() as BlockId, fallthrough],
                    &[&[], &[]],
                );
                if !last {
                    next = fallthrough;
                }
            }
            true
        }
        Terminator::Unreachable => {
            sel.emit(MInst::new(OpId::UNREACHABLE, Width::W64, smallvec::smallvec![]));
            true
        }
        Terminator::Invoke { .. } | Terminator::Resume { .. } => {
            panic!("glcc native backend: exception handling is not yet supported")
        }
    }
}

// -- helpers ---------------------------------------------------------------

fn result_vreg(sel: &mut FuncSel, mir: &Function, inst: Inst) -> crate::machine::VReg {
    sel.value_vreg(mir, mir.insts[inst].result.expect("instruction has a result"))
}

fn result_width(sel: &FuncSel, mir: &Function, inst: Inst) -> Width {
    value_width(&mir.values[mir.insts[inst].result.expect("instruction has a result")].ty)
}

fn single_value(mir: &Function, inst: Inst) -> Value {
    let mut ops: Vec<Value> = Vec::new();
    mir.insts[inst].kind.for_each_operand(&mut |v| ops.push(v));
    ops[0]
}

fn single_operand(sel: &mut FuncSel, mir: &Function, inst: Inst) -> crate::machine::VReg {
    sel.value_vreg(mir, single_value(mir, inst))
}

fn scalar_bits_of(mir: &Function, v: Value) -> u32 {
    crate::sel::scalar_bits(&mir.values[v].ty)
}

/// Lower a compare feeding a terminator (or standalone): returns
/// `Some((pred, a, b))` and marks the compare fused when the condition
/// is a single-use compare; `None` means test-a-boolean.
fn cond_icc(sel: &mut FuncSel, mir: &Function, cond: Value) -> Option<(glcc_mir::IcmpPred, crate::machine::VReg, crate::machine::VReg, Width)> {
    let Def::Inst(cmp_inst) = mir.values[cond].def else { return None };
    let InstKind::Icmp { pred, lhs, rhs } = mir.insts[cmp_inst].kind else { return None };
    if mir.use_count(cond) != 1 {
        return None;
    }
    sel.mark_fused(cond);
    let (pred, a, b) = cmp_operands(sel, mir, cmp_inst, pred);
    Some((pred, a, b, icmp_width(mir, cmp_inst)))
}

/// The compare width of an `Icmp`: its operands' scalar width. A W32
/// operand in a register may hold garbage in the upper half (32-bit
/// loads zero-extend, but spilled values and arguments do not
/// necessarily), so comparing at W64 would read those bits.
fn icmp_width(mir: &Function, inst: Inst) -> Width {
    let InstKind::Icmp { lhs, .. } = mir.insts[inst].kind else { unreachable!() };
    value_width(&mir.values[lhs].ty)
}

/// Condition code for a select-style branch: fuse a compare feeding
/// the condition, else test the boolean itself.
fn select_cc(sel: &mut FuncSel, mir: &Function, cond: Value) -> u8 {
    match cond_icc(sel, mir, cond) {
        Some((pred, a, b, width)) => {
            sel.emit(MInst::new(op_id(Op::CmpRR), width, smallvec::smallvec![Operand::Reg(Reg::V(a)), Operand::Reg(Reg::V(b))]));
            icc(pred)
        }
        None => {
            let vr = sel.value_vreg(mir, cond);
            sel.emit(MInst::new(
                op_id(Op::TestRR),
                value_width(&mir.values[cond].ty),
                smallvec::smallvec![Operand::Reg(Reg::V(vr)), Operand::Reg(Reg::V(vr))],
            ));
            0xF
        }
    }
}

/// Compare operands for `cmp a, b` (flags describe a - b). A constant
/// on the left swaps sides and mirrors the predicate.
fn cmp_operands(sel: &mut FuncSel, mir: &Function, inst: Inst, pred: glcc_mir::IcmpPred) -> (glcc_mir::IcmpPred, crate::machine::VReg, crate::machine::VReg) {
    let InstKind::Icmp { lhs, rhs, .. } = mir.insts[inst].kind else { unreachable!() };
    let lhs_const = is_const(mir, lhs);
    let rhs_const = is_const(mir, rhs);
    if lhs_const && !rhs_const {
        let a = sel.value_vreg(mir, rhs);
        let b = sel.value_vreg(mir, lhs);
        (swap(pred), a, b)
    } else {
        let a = sel.value_vreg(mir, lhs);
        let b = sel.value_vreg(mir, rhs);
        (pred, a, b)
    }
}

fn is_const(mir: &Function, v: Value) -> bool {
    matches!(mir.values[v].def, Def::Const(_))
}

fn addr(sel: &mut FuncSel, mir: &Function, inst: Inst, target: SymTarget) {
    let dst = result_vreg(sel, mir, inst);
    let mem = crate::machine::MemOperand {
        base: None,
        index: None,
        scale: 0,
        disp: 0,
        sym: Some(SymOperand { target, reloc: crate::machine::RelocKind::PcRel32, addend: 0 }),
    };
    sel.emit(MInst::new(op_id(Op::Lea), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Mem(mem)]));
}

enum CallTarget {
    Direct(SymTarget),
    Indirect(Value),
}

/// Emit the call instruction with arguments/results as fixed-register
/// operands: the allocator inserts the argument moves itself and
/// resolves overlaps between the fixed targets (sequential moves into
/// argument registers can clobber an argument not yet moved).
fn emit_call_inst(sel: &mut FuncSel, mir: &Function, result: Option<Value>, target: CallTarget, args: &[Value], sig: &glcc_mir::Signature) {
    let mut ops: smallvec::SmallVec<[Operand; 4]> = smallvec::SmallVec::new();
    // Aggregate returns: piece temps defined by the call, unpacked
    // into a fresh home slot afterwards.
    let mut ret_pieces: Option<(smallvec::SmallVec<[crate::machine::VReg; 2]>, u64)> = None;
    // Every argument rides a fresh single-use pin vreg: a MIR value
    // reused across calls would otherwise carry fixed constraints to
    // several different registers, which the allocator resolves
    // poorly. Results are the same: the call defines a fresh vreg that
    // becomes the result value's register.
    if let Some(res) = result {
        match sel.isa.abi().ret_loc(&sig.ret) {
            RetLoc::Void => {}
            RetLoc::Reg(preg) => {
                // The call defines a fresh vreg which BECOMES the result
                // value's register (single fixed def per vreg — no
                // unpacking copy needed).
                let pin = sel.alloc_vreg_class(match preg.class {
                    crate::machine::RegClass::Vec => crate::machine::RegClass::Vec,
                    _ => crate::machine::RegClass::Gpr,
                });
                sel.set_reg(res, pin);
                ops.push(Operand::FixedReg(Reg::V(pin), preg));
            }
            RetLoc::Regs(pregs, n) => {
                let mut temps = smallvec::SmallVec::<[crate::machine::VReg; 2]>::new();
                for preg in pregs.iter().take(n as usize) {
                    let tmp = sel.alloc_vreg_class(if preg.class == crate::machine::RegClass::Vec { crate::machine::RegClass::Vec } else { crate::machine::RegClass::Gpr });
                    ops.push(Operand::FixedReg(Reg::V(tmp), *preg));
                    temps.push(tmp);
                }
                ret_pieces = Some((temps, n as u64));
            }
            // Memory-class returns were handled by the sret slot above.
            RetLoc::Memory => {}
        }
    }
    match target {
        CallTarget::Direct(sym) => {
            ops.push(Operand::Sym(SymOperand { target: sym, reloc: crate::machine::RelocKind::Plt32, addend: 0 }));
        }
        CallTarget::Indirect(v) => {
            // The callee pointer moves through the dedicated pin
            // register (excluded from allocation): an ordinary call
            // operand would let the allocator place it in an argument
            // register, which this call's own argument moves clobber.
            let vr = sel.value_vreg(mir, v);
            sel.emit(MInst::new(
                OpId::MOVE,
                Width::W64,
                smallvec::smallvec![Operand::Reg(Reg::P(PReg::gpr(regs::R10))), Operand::Reg(Reg::V(vr))],
            ));
            ops.push(Operand::Reg(Reg::P(PReg::gpr(regs::R10))));
        }
    }
    let mut gp = 0;
    let mut fp = 0;
    // Parameter index for the signature lookup: distinct from the
    // register counters, because a stack-passed argument consumes a
    // parameter position but no register (SysV).
    let mut idx = 0usize;
    let mut reg_args: smallvec::SmallVec<[PReg; 8]> = smallvec::SmallVec::new();
    // (value, byte offset in the outgoing area, type)
    let mut stack: Vec<(Value, i64, glcc_mir::Ty)> = Vec::new();
    let mut stack_off = 0i64;
    // Memory-class returns: the caller's result slot address rides as
    // the hidden first argument (pushed after the target so the
    // allocator sees a use, not a def).
    let mut sret_slot: Option<u32> = None;
    if let Some(res) = result {
        if matches!(sel.isa.abi().ret_loc(&sig.ret), RetLoc::Memory) {
            let ty = mir.values[res].ty.clone();
            let slot = sel.slot(ty.size_bytes().max(1) as u32, glcc_mir::ty::ty_align(&ty).max(1) as u32);
            let addr = sel.alloc_vreg();
            sel.emit(MInst::new(
                op_id(Op::Lea),
                Width::W64,
                smallvec::smallvec![Operand::Reg(Reg::V(addr)), Operand::Mem(crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: 0, sym: None })],
            ));
            ops.push(Operand::FixedReg(Reg::V(addr), PReg::gpr(regs::RDI)));
            sel.set_slot(res, slot);
            sret_slot = Some(slot);
            gp = 1;
        }
    }
    for arg in args.iter() {
        // The hidden sret pointer occupies parameter position 0.
        let ty = sig.params.get(idx).unwrap_or(&mir.values[*arg].ty).clone();
        idx += 1;
        match sel.isa.abi().param_loc(gp, fp, &ty) {
            ParamLoc::Reg(preg) => {
                match preg.class {
                    crate::machine::RegClass::Gpr => gp += 1,
                    crate::machine::RegClass::Vec => fp += 1,
                }
                reg_args.push(preg);
                ops.push(Operand::Reg(Reg::V(sel.value_vreg(mir, *arg))));
            }
            ParamLoc::Regs(pregs, n) => {
                // Load each eightbyte piece from the aggregate's
                // address into a temp, pinned to its piece register.
                let addr = sel.mem_of(mir, *arg);
                for (i, preg) in pregs.iter().take(n as usize).enumerate() {
                    let piece_bytes = (ty.size_bytes() - (i as u64) * 8).min(8);
                    let tmp = sel.alloc_vreg_class(if preg.class == crate::machine::RegClass::Vec {
                        crate::machine::RegClass::Vec
                    } else {
                        crate::machine::RegClass::Gpr
                    });
                    match preg.class {
                        crate::machine::RegClass::Vec => {
                            let op = if piece_bytes >= 8 { Op::MovsdL } else { Op::MovssL };
                            sel.emit(MInst::new(op_id(op), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Mem(crate::machine::MemOperand { disp: addr.disp + (i as i64) * 8, ..addr.clone() })]));
                        }
                        _ => {
                            let width = if piece_bytes >= 8 { Width::W64 } else { Width::W32 };
                            sel.emit(MInst::new(op_id(Op::Load), width, smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Mem(crate::machine::MemOperand { disp: addr.disp + (i as i64) * 8, ..addr.clone() })]));
                        }
                    }
                    match preg.class {
                        crate::machine::RegClass::Gpr => gp += 1,
                        crate::machine::RegClass::Vec => fp += 1,
                    }
                    ops.push(Operand::FixedReg(Reg::V(tmp), *preg));
                }
            }
            ParamLoc::Stack => {
                // Stack arguments consume no register slot; the
                // parameter index advanced above.
                let align = glcc_mir::ty::ty_align(&ty).max(1).max(8) as i64;
                stack_off = (stack_off + align - 1) / align * align;
                let slot = stack_off;
                stack_off += ty.size_bytes().next_multiple_of(8) as i64;
                stack.push((*arg, slot, ty.clone()));
            }
        }
    }
    // Stack arguments: reserve (16-aligned) scratch below rsp, store
    // each argument at [rsp + off] (first stack argument lowest), and
    // pop it all after the call. The callee reads them at rbp+16+.
    let stack_bytes = if stack.is_empty() {
        0
    } else {
        let total = (stack_off as u64).next_multiple_of(16) as i64;
        sel.emit(MInst::new(
            op_id(Op::SubI),
            Width::W64,
            smallvec::smallvec![Operand::Reg(Reg::P(PReg::gpr(regs::RSP))), Operand::Reg(Reg::P(PReg::gpr(regs::RSP))), Operand::Imm(total)],
        ));
        for (arg, off, ty) in stack.iter() {
            if ty.is_aggregate() {
                // By-value aggregate: copy the bytes into the slot.
                let src = sel.mem_of(mir, *arg);
                let dst = crate::machine::MemOperand {
                    base: Some(Reg::P(PReg::gpr(regs::RSP))),
                    index: None,
                    scale: 0,
                    disp: *off,
                    sym: None,
                };
                copy_bytes(sel, &dst, &src, ty.size_bytes(), glcc_mir::ty::ty_align(ty).max(1).max(8) as u64);
                continue;
            }
            let v = sel.value_vreg(mir, *arg);
            let (store_op, width) = if matches!(ty, glcc_mir::Ty::Float(_)) {
                (if matches!(ty, glcc_mir::Ty::Float(glcc_mir::FloatKind::F64)) { Op::MovsdS } else { Op::MovssS }, Width::W64)
            } else {
                (Op::Store, value_width(ty))
            };
            sel.emit(MInst::new(
                op_id(store_op),
                width,
                smallvec::smallvec![
                    Operand::Reg(Reg::V(v)),
                    Operand::Mem(crate::machine::MemOperand {
                        base: Some(Reg::P(PReg::gpr(regs::RSP))),
                        index: None,
                        scale: 0,
                        disp: *off,
                        sym: None,
                    }),
                ],
            ));
        }
        total
    };
    if sig.variadic {
        // SysV: %al holds the number of vector registers used. The
        // count is pinned in %rax for the call; a fixed early use
        // under a fixed late def of the same register is the
        // combination regalloc2 supports.
        let zero = sel.alloc_vreg();
        sel.emit(MInst::new(
            sel.isa.const_move_op(fp as i64),
            Width::W32,
            smallvec::smallvec![Operand::Reg(Reg::V(zero)), Operand::Imm(fp as i64)],
        ));
        ops.push(Operand::FixedReg(Reg::V(zero), PReg::gpr(regs::RAX)));
    }
    let op = match target {
        CallTarget::Direct(_) => Op::Call,
        CallTarget::Indirect(_) => Op::CallRm,
    };
    let mut call_inst = MInst::new(op_id(op), Width::W64, ops);
    call_inst.call_args = Some(reg_args);
    sel.emit(call_inst);
    if let (Some(res), Some((temps, n))) = (result, ret_pieces) {
        // Unpack the returned pieces into the result's home slot.
        let ty = mir.values[res].ty.clone();
        let slot = sel.slot(ty.size_bytes().max(1) as u32, glcc_mir::ty::ty_align(&ty).max(1) as u32);
        sel.set_slot(res, slot);
        let pregs = match sel.isa.abi().ret_loc(&sig.ret) {
            RetLoc::Regs(p, _) => p,
            _ => unreachable!(),
        };
        for (i, tmp) in temps.iter().enumerate().take(n as usize) {
            let bytes = (ty.size_bytes() - (i as u64) * 8).min(8);
            let dst = crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: (i as i64) * 8, sym: None };
            match pregs[i].class {
                crate::machine::RegClass::Vec => {
                    let op = if bytes >= 8 { Op::MovsdS } else { Op::MovssS };
                    sel.emit(MInst::new(op_id(op), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(*tmp)), Operand::Mem(dst)]));
                }
                _ => {
                    let width = if bytes >= 8 { Width::W64 } else { Width::W32 };
                    sel.emit(MInst::new(op_id(Op::Store), width, smallvec::smallvec![Operand::Reg(Reg::V(*tmp)), Operand::Mem(dst)]));
                }
            }
        }
    }
    if stack_bytes > 0 {
        sel.emit(MInst::new(
            op_id(Op::AddI),
            Width::W64,
            smallvec::smallvec![Operand::Reg(Reg::P(PReg::gpr(regs::RSP))), Operand::Reg(Reg::P(PReg::gpr(regs::RSP))), Operand::Imm(stack_bytes)],
        ));
    }
}

fn call(sel: &mut FuncSel, mir: &Function, inst: Inst, callee: glcc_mir::InvokeCallee, args: &[Value], sig: Option<glcc_mir::Signature>) {
    let sig = match (&callee, sig) {
        (glcc_mir::InvokeCallee::Func(f), _) => sel.module.funcs[*f].sig.clone(),
        (glcc_mir::InvokeCallee::Extern(e), _) => sel.module.externs[*e].sig.clone(),
        (glcc_mir::InvokeCallee::Indirect(_), Some(s)) => s,
        (glcc_mir::InvokeCallee::Indirect(_), None) => panic!("glcc native backend: indirect call without signature"),
    };
    let target = match callee {
        glcc_mir::InvokeCallee::Func(f) => CallTarget::Direct(SymTarget::Func(f)),
        glcc_mir::InvokeCallee::Extern(e) => CallTarget::Direct(SymTarget::Extern(e)),
        glcc_mir::InvokeCallee::Indirect(v) => CallTarget::Indirect(v),
    };
    emit_call_inst(sel, mir, mir.insts[inst].result, target, args, &sig);
}

fn select(sel: &mut FuncSel, mir: &Function, inst: Inst) {
    let InstKind::Select { cond, then_val, else_val } = mir.insts[inst].kind else { unreachable!() };
    let dst = result_vreg(sel, mir, inst);
    let then_vr = sel.value_vreg(mir, then_val);
    let else_vr = sel.value_vreg(mir, else_val);
    if matches!(mir.values[mir.insts[inst].result.expect("select has a result")].ty, glcc_mir::Ty::Float(_)) {
        // CMOV is integer-only: move both values through GPRs, cmov,
        // and move the winner back. The upper bits are don't-care for
        // the losing lane.
        let g_else = sel.alloc_vreg();
        let g_then = sel.alloc_vreg();
        sel.emit(MInst::new(op_id(Op::MovqX2G), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(g_else)), Operand::Reg(Reg::V(else_vr))]));
        sel.emit(MInst::new(op_id(Op::MovqX2G), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(g_then)), Operand::Reg(Reg::V(then_vr))]));
        let cc = select_cc(sel, mir, cond);
        let g_res = sel.alloc_vreg();
        sel.emit(MInst::new(op_id(Op::Cmov), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(g_res)), Operand::Reg(Reg::V(g_else)), Operand::Reg(Reg::V(g_then)), Operand::Imm(cc as i64)]));
        sel.emit(MInst::new(op_id(Op::MovqG2X), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(g_res))]));
        return;
    }
    let cc = select_cc(sel, mir, cond);
    // dst is defined once, tied to the false path's register: the
    // cmov overwrites it only when the condition holds.
    sel.emit(MInst::new(
        op_id(Op::Cmov),
        result_width(sel, mir, inst),
        smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(else_vr)), Operand::Reg(Reg::V(then_vr)), Operand::Imm(cc as i64)],
    ));
}

/// Byte stride of one GEP step through `ty`: array indexing steps by
/// the element size; a scalar step is pointer arithmetic over a single
/// element, sized by the pointee itself.
fn step_size(ty: &Ty) -> u64 {
    match ty {
        Ty::Array { elem, .. } => elem.size_bytes(),
        t => t.size_bytes(),
    }
}

/// Address computation: constant steps fold into a displacement,
/// dynamic steps become an index register (scaled when the element
/// size is a power of two, multiplied into a temp otherwise).
fn gep(sel: &mut FuncSel, mir: &Function, inst: Inst) {
    let InstKind::GetElementPtr { base, indices } = mir.insts[inst].kind.clone() else { unreachable!() };
    let mut ty = match &mir.values[base].ty {
        Ty::Ptr { pointee, .. } => (**pointee).clone(),
        t => panic!("glcc native backend: GEP through {t:?}"),
    };
    let mut disp: i64 = 0;
    let mut dyn_terms: Vec<(Value, u64)> = Vec::new();
    for idx in &indices {
        let step = glcc_mir::Signature::gep_step(&ty, idx).unwrap_or_else(|| panic!("glcc native backend: bad GEP step"));
        match idx {
            GepIndex::Field(i) => {
                let Ty::Struct { offsets, .. } = &ty else { panic!("glcc native backend: Field step through {ty:?}") };
                disp += offsets[*i as usize] as i64;
            }
            GepIndex::ConstIndex(c) => disp += *c as i64 * step_size(&ty) as i64,
            GepIndex::Index(v) => dyn_terms.push((*v, step_size(&ty))),
        }
        ty = step;
    }

    let dst = result_vreg(sel, mir, inst);
    let base_reg = match sel.value_loc(mir, base) {
        crate::sel::ValLoc::Slot(s) => Reg::FI(s),
        crate::sel::ValLoc::Reg(vr) => Reg::V(vr),
    };

    // Reduce dynamic terms to one index register. A leading
    // power-of-two-sized term rides the ModRM scale; anything else is
    // multiplied into a temp and summed.
    let mut index: Option<(crate::machine::VReg, u8)> = None;
    for (v, size) in dyn_terms {
        let vr = sel.value_vreg(mir, v);
        if index.is_none() && (size == 1 || (size.is_power_of_two() && size <= 8)) {
            index = Some((vr, size.ilog2() as u8));
            continue;
        }
        let scaled = if size == 1 {
            vr
        } else {
            let tmp = sel.alloc_vreg_note("gep tmp");
            sel.emit(MInst::new(
                op_id(Op::ImulI),
                Width::W64,
                smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Reg(Reg::V(vr)), Operand::Imm(size as i64)],
            ));
            tmp
        };
        index = Some(match index {
            None => (scaled, 0),
            Some((prev, _)) => {
                let sum = sel.alloc_vreg_note("gep sum");
                sel.emit(MInst::new(
                    op_id(Op::AddRR),
                    Width::W64,
                    smallvec::smallvec![Operand::Reg(Reg::V(sum)), Operand::Reg(Reg::V(prev)), Operand::Reg(Reg::V(scaled))],
                ));
                (sum, 0)
            }
        });
    }
    let (index, scale) = index.map(|(v, s)| (Some(v), s)).unwrap_or((None, 0));

    let mem = crate::machine::MemOperand { base: Some(base_reg), index: index.map(Reg::V), scale, disp, sym: None };
    sel.emit(MInst::new(op_id(Op::Lea), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Mem(mem)]));
}

/// Integer divide/remainder. The setup instruction defines the
/// high-half scratch in RDX (sign-extend or zero) and pins the
/// dividend into RAX; the divide's early fixed uses hold RDX/RAX
/// across it while its late fixed def lands the result.
fn divide(sel: &mut FuncSel, mir: &Function, inst: Inst) {
    let (signed, want_rem) = match mir.insts[inst].kind {
        InstKind::Udiv(..) => (false, false),
        InstKind::Sdiv(..) => (true, false),
        InstKind::Urem(..) => (false, true),
        InstKind::Srem(..) => (true, true),
        _ => unreachable!(),
    };
    let mut ops: Vec<Value> = Vec::new();
    mir.insts[inst].kind.for_each_operand(&mut |v| ops.push(v));
    let (a, b) = (ops[0], ops[1]);
    let a_vr = sel.value_vreg(mir, a);
    let b_vr = sel.value_vreg(mir, b);
    let result = sel.value_vreg(mir, mir.insts[inst].result.expect("div has a result"));
    let scratch = sel.alloc_vreg();

    if signed {
        sel.emit(MInst::new(
            op_id(Op::Cqo),
            Width::W64,
            smallvec::smallvec![Operand::Reg(Reg::V(scratch)), Operand::Reg(Reg::V(a_vr))],
        ));
    } else {
        sel.emit(MInst::new(op_id(Op::XorRdx), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(scratch))]));
    }

    let div_op = match (signed, want_rem) {
        (false, false) => Op::DivQU,
        (true, false) => Op::DivQS,
        (false, true) => Op::RemQU,
        (true, true) => Op::RemQS,
    };
    sel.emit(MInst::new(
        op_id(div_op),
        Width::W64,
        smallvec::smallvec![
            Operand::Reg(Reg::V(result)),
            Operand::Reg(Reg::V(scratch)),
            Operand::Reg(Reg::V(a_vr)),
            Operand::Reg(Reg::V(b_vr)),
        ],
    ));
}

/// memcpy/memset lowered as libcalls against the platform libc. The
/// length (and memset byte) are constants here, materialized into
/// fresh vregs. Arguments ride the call's `call_args` pins: the
/// post-RA move run resolves them as one parallel set (the constant
/// vregs may land in the very argument registers they feed).
fn mem_intrinsic(sel: &mut FuncSel, mir: &Function, inst: Inst, name: &'static str, args: &[Value], extra: Option<(i64, i64)>) {
    let arg_regs = sel.isa.abi().arg_regs();
    let mut ops: smallvec::SmallVec<[Operand; 4]> = smallvec::SmallVec::new();
    if let Some(res) = mir.insts[inst].result {
        ops.push(Operand::FixedReg(Reg::V(sel.value_vreg(mir, res)), sel.isa.abi().ret_reg()));
    }
    ops.push(Operand::Sym(SymOperand { target: SymTarget::Named(name), reloc: crate::machine::RelocKind::Plt32, addend: 0 }));

    let mut pregs: smallvec::SmallVec<[PReg; 8]> = smallvec::SmallVec::new();
    for (i, arg) in args.iter().enumerate() {
        ops.push(Operand::Reg(Reg::V(sel.value_vreg(mir, *arg))));
        pregs.push(arg_regs[i]);
    }
    if let Some((len, byte)) = extra {
        let mut imm_arg = |sel: &mut FuncSel, ops: &mut smallvec::SmallVec<[Operand; 4]>, pregs: &mut smallvec::SmallVec<[PReg; 8]>, slot: usize, v: i64, width: Width| {
            let vr = sel.alloc_vreg();
            sel.emit(MInst::new(sel.isa.const_move_op(v), width, smallvec::smallvec![Operand::Reg(Reg::V(vr)), Operand::Imm(v)]));
            ops.push(Operand::Reg(Reg::V(vr)));
            pregs.push(arg_regs[slot]);
        };
        if name == "memset" {
            imm_arg(sel, &mut ops, &mut pregs, 1, byte, Width::W32);
        }
        imm_arg(sel, &mut ops, &mut pregs, 2, len, Width::W64);
    }
    let mut call = MInst::new(op_id(Op::Call), Width::W64, ops);
    call.call_args = Some(pregs);
    sel.emit(call);
}

/// Copy `len` bytes between two memory operands, chunked to the
/// narrower of the two sides' natural alignment (packed aggregates
/// force byte chunks).
fn copy_bytes(sel: &mut FuncSel, dst: &crate::machine::MemOperand, src: &crate::machine::MemOperand, len: u64, align: u64) {
    let unit = align.min(8).max(1);
    let mut off = 0i64;
    while (off as u64) < len {
        let (op, width): (Op, Width) = match unit {
            8 => (Op::Load, Width::W64),
            4 => (Op::Load, Width::W32),
            2 => (Op::Load, Width::W16),
            _ => (Op::Load, Width::W8),
        };
        let tmp = sel.alloc_vreg();
        sel.emit(MInst::new(
            op_id(op),
            width,
            smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Mem(crate::machine::MemOperand { disp: src.disp + off, ..src.clone() })],
        ));
        let (sop, swidth): (Op, Width) = match unit {
            8 => (Op::Store, Width::W64),
            4 => (Op::Store, Width::W32),
            2 => (Op::Store, Width::W16),
            _ => (Op::Store, Width::W8),
        };
        sel.emit(MInst::new(
            op_id(sop),
            swidth,
            smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Mem(crate::machine::MemOperand { disp: dst.disp + off, ..dst.clone() })],
        ));
        off += unit as i64;
    }
}

// -- float helpers ---------------------------------------------------------

fn is_f64(mir: &Function, v: Value) -> bool {
    matches!(mir.values[v].ty, glcc_mir::Ty::Float(glcc_mir::FloatKind::F64))
}

fn widen_sse(op: Op, double: bool) -> Op {
    match (op, double) {
        (Op::AddSS, true) => Op::AddSD,
        (Op::SubSS, true) => Op::SubSD,
        (Op::MulSS, true) => Op::MulSD,
        (Op::DivSS, true) => Op::DivSD,
        (o, _) => o,
    }
}

/// Negation: subtract the operand from -0.0 (no xorps constant pool
/// yet, so the sign bit comes from a materialized constant).
fn fneg(sel: &mut FuncSel, mir: &Function, inst: Inst) -> bool {
    let src_v = single_value(mir, inst);
    let dst = result_vreg(sel, mir, inst);
    let src = single_operand(sel, mir, inst);
    let double = is_f64(mir, src_v);
    let bits: i64 = if double { 0x8000_0000_0000_0000u64 as i64 } else { 0x8000_0000u64 as i64 };
    let gpr = sel.alloc_vreg();
    sel.emit(MInst::new(sel.isa.const_move_op(bits), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(gpr)), Operand::Imm(bits)]));
    let neg0 = sel.alloc_vreg_class(crate::machine::RegClass::Vec);
    sel.emit(MInst::new(op_id(Op::MovqG2X), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(neg0)), Operand::Reg(Reg::V(gpr))]));
    let op = if double { Op::SubSD } else { Op::SubSS };
    sel.emit(MInst::new(op_id(op), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(neg0)), Operand::Reg(Reg::V(src))]));
    true
}

/// Floating compare: ucomis sets ZF/PF/CF (all three on unordered), so
/// the ordered predicates AND with setnp and Ne ORs with setp.
fn fcmp(sel: &mut FuncSel, mir: &Function, inst: Inst) {
    let InstKind::Fcmp { pred, .. } = mir.insts[inst].kind else { unreachable!() };
    let mut ops: Vec<Value> = Vec::new();
    mir.insts[inst].kind.for_each_operand(&mut |v| ops.push(v));
    let (a, b) = (ops[0], ops[1]);
    let op = if is_f64(mir, a) { Op::UcomisD } else { Op::UcomisS };
    let a = sel.value_vreg(mir, a);
    let b = sel.value_vreg(mir, b);
    sel.emit(MInst::new(op_id(op), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(a)), Operand::Reg(Reg::V(b))]));
    let dst = result_vreg(sel, mir, inst);
    let (cc1, cc2): (u8, u8) = match pred {
        glcc_mir::FcmpPred::Eq => (0x4, 0xB),  // e & np
        glcc_mir::FcmpPred::Ne => (0x5, 0xA),  // ne | p
        glcc_mir::FcmpPred::Lt => (0x2, 0xB),  // b & np
        glcc_mir::FcmpPred::Le => (0x6, 0xB),  // be & np
        glcc_mir::FcmpPred::Gt => (0x7, 0xB),  // a & np
        glcc_mir::FcmpPred::Ge => (0x3, 0xB),  // ae & np
        glcc_mir::FcmpPred::Uno => (0xA, u8::MAX), // p
        glcc_mir::FcmpPred::Ord => (0xB, u8::MAX), // np
    };
    if cc2 == u8::MAX {
        sel.emit(MInst::new(op_id(Op::Setcc), Width::W8, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Imm(cc1 as i64)]));
        return;
    }
    let t1 = sel.alloc_vreg();
    sel.emit(MInst::new(op_id(Op::Setcc), Width::W8, smallvec::smallvec![Operand::Reg(Reg::V(t1)), Operand::Imm(cc1 as i64)]));
    let t2 = sel.alloc_vreg();
    sel.emit(MInst::new(op_id(Op::Setcc), Width::W8, smallvec::smallvec![Operand::Reg(Reg::V(t2)), Operand::Imm(cc2 as i64)]));
    let logic = if cc1 == 0x5 { Op::OrRR } else { Op::AndRR };
    sel.emit(MInst::new(op_id(logic), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(t1)), Operand::Reg(Reg::V(t2))]));
}

fn load_float(sel: &mut FuncSel, mir: &Function, inst: Inst) {
    let InstKind::Load { addr, .. } = mir.insts[inst].kind else { unreachable!() };
    let dst = result_vreg(sel, mir, inst);
    let op = if is_f64(mir, mir.insts[inst].result.expect("load has a result")) { Op::MovsdL } else { Op::MovssL };
    let mem = sel.mem_of(mir, addr);
    sel.emit(MInst::new(op_id(op), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Mem(mem)]));
}

fn store_float(sel: &mut FuncSel, mir: &Function, inst: Inst, addr: Value, val: Value) {
    let _ = inst;
    let op = if is_f64(mir, val) { Op::MovsdS } else { Op::MovssS };
    let v = sel.value_vreg(mir, val);
    let mem = sel.mem_of(mir, addr);
    sel.emit(MInst::new(op_id(op), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(v)), Operand::Mem(mem)]));
}

/// va_arg of a small aggregate: INTEGER-class pieces are fetched
/// whole from the register save area when the cursor plus the argument
/// size still fits (48-byte GP area), else from the overflow area;
/// memory-class aggregates always come from the overflow area. The
/// result value is the fresh home slot's address.
fn va_arg_aggregate(sel: &mut FuncSel, mir: &Function, inst: Inst, ap: Value) -> bool {
    let InstKind::VaArg { ap: ap_val } = mir.insts[inst].kind else { unreachable!() };
    let res = mir.insts[inst].result.expect("va_arg has a result");
    let ty = mir.values[res].ty.clone();
    let size = ty.size_bytes();
    if size > 16 {
        panic!("glcc native backend: va_arg of {ty:?} is not yet supported");
    }
    let Some(pieces) = abi::struct_pieces(&ty) else {
        panic!("glcc native backend: va_arg of {ty:?} is not yet supported");
    };
    if pieces.iter().any(|(c, _)| *c != crate::machine::RegClass::Gpr) {
        panic!("glcc native backend: va_arg of {ty:?} with float pieces is not yet supported");
    }
    let ap_addr = sel.value_vreg(mir, ap_val);
    let _ = ap;
    let mut mem = |sel: &mut FuncSel, base: crate::machine::Reg, disp: i64, dst: crate::machine::VReg| {
        sel.emit(MInst::new(
            op_id(Op::Load),
            Width::W64,
            smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Mem(crate::machine::MemOperand { base: Some(base), index: None, scale: 0, disp, sym: None })],
        ));
    };
    let off = sel.alloc_vreg();
    mem(sel, Reg::V(ap_addr), 0, off);
    let save = sel.alloc_vreg();
    mem(sel, Reg::V(ap_addr), 16, save);
    let off64 = sel.alloc_vreg();
    sel.emit(MInst::new(op_id(Op::MovRR), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(off64)), Operand::Reg(Reg::V(off))]));
    let reg_addr = sel.alloc_vreg();
    sel.emit(MInst::new(op_id(Op::AddRR), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(reg_addr)), Operand::Reg(Reg::V(save)), Operand::Reg(Reg::V(off64))]));
    let ovf = sel.alloc_vreg();
    mem(sel, Reg::V(ap_addr), 8, ovf);
    // Fits when gp_offset + size <= 48.
    sel.emit(MInst::new(op_id(Op::CmpI), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(off)), Operand::Imm(48 - size as i64)]));
    let addr = sel.alloc_vreg();
    sel.emit(MInst::new(OpId::MOVE, Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(addr)), Operand::Reg(Reg::V(ovf))]));
    sel.emit(MInst::new(op_id(Op::Cmov), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(addr)), Operand::Reg(Reg::V(ovf)), Operand::Reg(Reg::V(reg_addr)), Operand::Imm(0x2)]));
    // Home the bytes.
    let slot = sel.slot(size.max(1) as u32, glcc_mir::ty::ty_align(&ty).max(1) as u32);
    sel.set_slot(res, slot);
    let dst = crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: 0, sym: None };
    let src = crate::machine::MemOperand { base: Some(Reg::V(addr)), index: None, scale: 0, disp: 0, sym: None };
    copy_bytes(sel, &dst, &src, size, glcc_mir::ty::ty_align(&ty).max(1) as u64);
    // Advance: the GP cursor moves a full slot (8) when the register
    // path was taken; the overflow area always absorbs the whole
    // aligned argument otherwise.
    let step = size.next_multiple_of(8) as i64;
    let new_off = sel.alloc_vreg();
    sel.emit(MInst::new(op_id(Op::AddI), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(new_off)), Operand::Reg(Reg::V(off)), Operand::Imm(step)]));
    sel.emit(MInst::new(op_id(Op::Cmov), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(new_off)), Operand::Reg(Reg::V(ovf)), Operand::Reg(Reg::V(off)), Operand::Imm(0x2)]));
    let mut off_m = sel.mem_of(mir, ap_val);
    sel.emit(MInst::new(
        op_id(Op::Store),
        Width::W32,
        smallvec::smallvec![Operand::Reg(Reg::V(new_off)), Operand::Mem(off_m)],
    ));
    // Overflow cursor only moves when the register path was not taken.
    let plus = sel.alloc_vreg();
    sel.emit(MInst::new(op_id(Op::AddI), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(plus)), Operand::Reg(Reg::V(ovf)), Operand::Imm(step)]));
    let bumped = sel.alloc_vreg();
    sel.emit(MInst::new(op_id(Op::Cmov), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(bumped)), Operand::Reg(Reg::V(plus)), Operand::Reg(Reg::V(ovf)), Operand::Imm(0x2)]));
    let mut ovf_m = sel.mem_of(mir, ap_val);
    ovf_m.disp += 8;
    sel.emit(MInst::new(
        op_id(Op::Store),
        Width::W64,
        smallvec::smallvec![Operand::Reg(Reg::V(bumped)), Operand::Mem(ovf_m)],
    ));
    true
}

/// Seed the va_list tag: {gp_offset, fp_offset, overflow_arg_area,
/// reg_save_area}. The offsets come from the Entry prelude's counts.
fn va_start(sel: &mut FuncSel, mir: &Function, inst: Inst, ap: Value) {
    let save_slot = sel.va_save_slot.expect("va_start without a variadic prelude");
    let _ = inst;
    let mut field = |sel: &mut FuncSel, val: Operand, disp: i64, width: Width| {
        let v = match &val {
            Operand::Reg(Reg::V(v)) => *v,
            other => panic!("glcc native backend: va_start field {other:?}"),
        };
        let mut m = sel.mem_of(mir, ap);
        m.disp += disp;
        sel.emit(MInst::new(
            op_id(Op::Store),
            width,
            smallvec::smallvec![Operand::Reg(Reg::V(v)), Operand::Mem(m)],
        ));
    };
    let gp = sel.alloc_vreg();
    sel.emit(MInst::new(sel.isa.const_move_op(sel.va_gp_off), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(gp)), Operand::Imm(sel.va_gp_off)]));
    field(sel, Operand::Reg(Reg::V(gp)), 0, Width::W32);
    let fp = sel.alloc_vreg();
    sel.emit(MInst::new(sel.isa.const_move_op(sel.va_fp_off), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(fp)), Operand::Imm(sel.va_fp_off)]));
    field(sel, Operand::Reg(Reg::V(fp)), 4, Width::W32);
    let ovf = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::Lea),
        Width::W64,
        smallvec::smallvec![
            Operand::Reg(Reg::V(ovf)),
            Operand::Mem(crate::machine::MemOperand { base: Some(Reg::P(PReg::gpr(regs::RBP))), index: None, scale: 0, disp: sel.va_overflow_disp, sym: None }),
        ],
    ));
    field(sel, Operand::Reg(Reg::V(ovf)), 8, Width::W64);
    let save = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::Lea),
        Width::W64,
        smallvec::smallvec![
            Operand::Reg(Reg::V(save)),
            Operand::Mem(crate::machine::MemOperand { base: Some(Reg::FI(save_slot)), index: None, scale: 0, disp: 0, sym: None }),
        ],
    ));
    field(sel, Operand::Reg(Reg::V(save)), 16, Width::W64);
}

/// Read the next variadic argument of the result type. Scalars only:
/// pick the register-save-area cursor or the overflow area, load the
/// value, and advance the cursor (16 bytes per FP slot, 8 per GP).
fn va_arg(sel: &mut FuncSel, mir: &Function, inst: Inst) -> bool {
    let InstKind::VaArg { ap } = mir.insts[inst].kind else { unreachable!() };
    let res = mir.insts[inst].result.expect("va_arg has a result");
    let res_ty = mir.values[res].ty.clone();
    let is_float = matches!(res_ty, glcc_mir::Ty::Float(_));
    if res_ty.is_aggregate() {
        return va_arg_aggregate(sel, mir, inst, ap);
    }
    if matches!(res_ty, glcc_mir::Ty::WideInt { .. }) {
        panic!("glcc native backend: va_arg of {res_ty:?} is not yet supported");
    }
    let is_f64 = is_f64(mir, res);

    // cursor field: +0 for GP, +4 for FP; limits 48 / 176.
    let (field_disp, limit, step): (i64, i64, i64) = if is_float { (4, 176, 16) } else { (0, 48, 8) };
    let ap_addr = sel.value_vreg(mir, ap);

    let mut mem = |sel: &mut FuncSel, base: crate::machine::Reg, disp: i64, op: Op, width: Width, dst: crate::machine::VReg| {
        sel.emit(MInst::new(
            op_id(op),
            width,
            smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Mem(crate::machine::MemOperand { base: Some(base), index: None, scale: 0, disp, sym: None })],
        ));
    };

    let off = sel.alloc_vreg();
    mem(sel, Reg::V(ap_addr), field_disp, Op::Load, Width::W32, off);
    // reg_save_area + offset
    let save = sel.alloc_vreg();
    mem(sel, Reg::V(ap_addr), 16, Op::Load, Width::W64, save);
    let off64 = sel.alloc_vreg();
    sel.emit(MInst::new(op_id(Op::MovRR), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(off64)), Operand::Reg(Reg::V(off))]));
    let reg_addr = sel.alloc_vreg();
    sel.emit(MInst::new(op_id(Op::AddRR), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(reg_addr)), Operand::Reg(Reg::V(save)), Operand::Reg(Reg::V(off64))]));
    let ovf = sel.alloc_vreg();
    mem(sel, Reg::V(ap_addr), 8, Op::Load, Width::W64, ovf);
    // addr = cursor < limit ? reg_path : overflow
    sel.emit(MInst::new(op_id(Op::CmpI), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(off)), Operand::Imm(limit)]));
    // addr = cursor < limit ? reg_path : overflow: defined once, tied
    // to the overflow (false) path.
    let addr = sel.alloc_vreg();
    sel.emit(MInst::new(op_id(Op::Cmov), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(addr)), Operand::Reg(Reg::V(ovf)), Operand::Reg(Reg::V(reg_addr)), Operand::Imm(0x2)]));

    // the value itself
    let dst = result_vreg(sel, mir, inst);
    let (load_op, width) = if is_float {
        (if is_f64 { Op::MovsdL } else { Op::MovssL }, Width::W64)
    } else {
        (Op::Load, value_width(&res_ty))
    };
    mem(sel, Reg::V(addr), 0, load_op, width, dst);

    // advance: cursor += step; bump the overflow area only when the
    // register path was not taken.
    let new_off = sel.alloc_vreg();
    sel.emit(MInst::new(op_id(Op::AddI), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(new_off)), Operand::Reg(Reg::V(off)), Operand::Imm(step)]));
    let mut off_m = sel.mem_of(mir, ap);
    off_m.disp += field_disp;
    sel.emit(MInst::new(
        op_id(Op::Store),
        Width::W32,
        smallvec::smallvec![Operand::Reg(Reg::V(new_off)), Operand::Mem(off_m)],
    ));
    // overflow cursor bumps only when the register path was taken:
    // bumped = cursor < limit ? ovf : ovf + 8.
    let bumped = sel.alloc_vreg();
    let plus8 = sel.alloc_vreg();
    sel.emit(MInst::new(op_id(Op::AddI), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(plus8)), Operand::Reg(Reg::V(ovf)), Operand::Imm(8)]));
    sel.emit(MInst::new(op_id(Op::Cmov), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(bumped)), Operand::Reg(Reg::V(plus8)), Operand::Reg(Reg::V(ovf)), Operand::Imm(0x2)]));
    let mut ovf_m = sel.mem_of(mir, ap);
    ovf_m.disp += 8;
    sel.emit(MInst::new(
        op_id(Op::Store),
        Width::W64,
        smallvec::smallvec![Operand::Reg(Reg::V(bumped)), Operand::Mem(ovf_m)],
    ));
    true
}
