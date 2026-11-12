//! aarch64 target: AAPCS64 (Linux ELF), fully hook-driven selection.
//! Design notes relative to the x86-64 backend:
//!
//! * Small integers live in the low bits of 32-bit registers with
//!   garbage upper halves (matching x86-64's narrow-op invariant). But
//!   AArch64 has no sub-32-bit compare/divide, so every consumer that
//!   interprets bits beyond the narrow width canonicalizes its inputs
//!   first: signed ops (sdiv/srem/asr, signed compares) sign-extend,
//!   unsigned and equality ops zero-extend.
//! * Three-operand ALU means no `Reuse` constraints: csel/cset define
//!   fresh registers instead of x86's cmov-over-the-false-path dance.
//! * Conditional branches encode as `b.!cond skip; b target; skip:`,
//!   keeping every branch field a patchable imm26.
//! * Composites over 16 bytes cross calls by reference: the call hook
//!   copies the bytes into a stack temporary and passes its address.
//! * `long double` is F128 here; its arithmetic panics until the quad
//!   libcall pass (census-tallied, like F80 on x86-64).

pub mod abi;
pub mod enc;
pub mod regs;
pub mod tables;

use glcc_mir::{Block, Const, Def, FloatKind, Function, GepIndex, Inst, InstKind, Terminator, Ty, Value};

use crate::isa::{EncOut, ParamLoc, RetLoc, SelItem, TargetIsa};
use crate::machine::{BlockId, MFunction, MInst, OpId, Operand, PReg, Reg, SymOperand, SymTarget, Width};
use crate::sel::{FuncSel, value_width};

use tables::{Op, op_id};

pub struct A64;

pub static A64_ISA: A64 = A64;

/// Condition code for an `IcmpPred` (flags describe `a - b`).
fn icc(pred: glcc_mir::IcmpPred) -> u8 {
    use glcc_mir::IcmpPred::*;
    match pred {
        Eq => enc::cc::EQ,
        Ne => enc::cc::NE,
        Ult => enc::cc::CC,
        Ule => enc::cc::LS,
        Ugt => enc::cc::HI,
        Uge => enc::cc::CS,
        Slt => enc::cc::LT,
        Sle => enc::cc::LE,
        Sgt => enc::cc::GT,
        Sge => enc::cc::GE,
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

fn is_signed_pred(pred: glcc_mir::IcmpPred) -> bool {
    matches!(pred, glcc_mir::IcmpPred::Slt | glcc_mir::IcmpPred::Sle | glcc_mir::IcmpPred::Sgt | glcc_mir::IcmpPred::Sge)
}

impl TargetIsa for A64 {
    fn name(&self) -> &'static str {
        "aarch64"
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
        &abi::Aapcs64
    }

    fn const_move_op(&self, _value: i64) -> OpId {
        op_id(Op::Const64)
    }

    fn gpr_to_vec_op(&self) -> OpId {
        op_id(Op::GprToVec)
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

    /// Branch fields hold imm26 at bits [25:0] in instruction units;
    /// the framework's `delta` measures to the field end (+4), so the
    /// instruction distance is (delta + 4) / 4.
    fn patch_branch_field(&self, text: &mut [u8], at: u32, delta: i64) {
        let at = at as usize;
        let mut w = u32::from_le_bytes(text[at..at + 4].try_into().unwrap());
        let imm26 = (((delta + 4) >> 2) as u32) & 0x3FF_FFFF;
        w = (w & 0xFC00_0000) | imm26;
        text[at..at + 4].copy_from_slice(&w.to_le_bytes());
    }

    fn elf_reloc(&self, kind: crate::machine::RelocKind) -> (u32, bool, u8) {
        use crate::machine::RelocKind as K;
        match kind {
            K::PagePc => (object::elf::R_AARCH64_ADR_PREL_PG_HI21.0, true, 4),
            K::Lo12 => (object::elf::R_AARCH64_ADD_ABS_LO12_NC.0, false, 4),
            K::Call26 => (object::elf::R_AARCH64_CALL26.0, true, 4),
            K::GotPage => (object::elf::R_AARCH64_ADR_GOT_PAGE.0, true, 4),
            K::GotLo12 => (object::elf::R_AARCH64_LD64_GOT_LO12_NC.0, false, 4),
            K::PcRel32 => (object::elf::R_AARCH64_PREL32.0, true, 4),
            K::Plt32 => (object::elf::R_AARCH64_JUMP26.0, true, 4),
            K::Abs64 => (object::elf::R_AARCH64_ABS64.0, false, 8),
        }
    }

    fn elf_data_relocs(&self) -> (u32, u32) {
        (object::elf::R_AARCH64_ABS64.0, object::elf::R_AARCH64_ABS32.0)
    }

    fn asm_arch(&self) -> glasm::Arch {
        glasm::Arch::Aarch64
    }
}

// -- entry: parameter pins, stack homing, variadic save area --------------

fn entry(sel: &mut FuncSel, mir: &Function) -> bool {
    // AAPCS64 variadic definitions spill every register-passed argument
    // into a 192-byte save area (GP at +0, stride 8; VR at +64, stride
    // 16) and seed a 32-byte va_list tag:
    // {__stack, __gr_top, __vr_top, __gr_offs, __vr_offs} with
    // descending signed offsets — negative means registers remain.
    let mut save_slot = 0u32;
    let mut gp_off = 0i64;
    let mut fp_off = 0i64;
    let mut named_stack_bytes = 0i64;
    if mir.sig.variadic {
        save_slot = sel.slot(192, 16);
        sel.va_save_slot = Some(save_slot);
        sel.va_overflow_disp = 16;
    }
    let entry_params: Vec<Value> = mir.blocks[mir.entry].params.clone();
    let mut gp = 0usize;
    let mut fp = 0usize;
    let mut stack_off = 0i64;
    // Memory-class returns take a hidden out-pointer in x8, which does
    // NOT consume one of the x0-x7 argument slots.
    if matches!(sel.isa.abi().ret_loc(&mir.sig.ret), RetLoc::Memory) {
        let vr = sel.alloc_vreg();
        sel.emit(MInst::new(OpId::MOVE, Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(vr)), Operand::Reg(Reg::P(PReg::gpr(regs::X8)))]));
        sel.sret_loc = Some(vr);
    }
    // Pre-pass: stack offsets in signature order.
    let mut stack_params: Vec<(Value, glcc_mir::Ty, i64)> = Vec::new();
    let (gp_init, fp_init) = (gp, fp);
    for &p in entry_params.iter() {
        let ty = mir.values[p].ty.clone();
        match sel.isa.abi().param_loc(gp, fp, &ty) {
            ParamLoc::Stack => {
                let align = glcc_mir::ty::ty_align(&ty).max(1).max(8) as i64;
                stack_off = (stack_off + align - 1) / align * align;
                named_stack_bytes = stack_off + ty.size_bytes().next_multiple_of(8) as i64;
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
    gp = gp_init;
    fp = fp_init;
    if mir.sig.variadic {
        sel.va_overflow_disp = 16 + named_stack_bytes;
    }
    // Pin pass: every register-classed parameter moves out of its ABI
    // register as one contiguous run (the allocator resolves it as a
    // parallel set; physical operands are invisible to it).
    let mut piece_stores: Vec<(crate::machine::VReg, crate::machine::RegClass, u64, u32, u64, bool)> = Vec::new();
    let mut gp_save: Vec<(i64, crate::machine::VReg)> = Vec::new();
    let mut fp_save: Vec<(i64, crate::machine::VReg)> = Vec::new();
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
                            gp_save.push((gp as i64, vr));
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
                // Small composite in registers: home slot + per-piece
                // pins, stored after the whole pin run. HFA pieces sit
                // at member stride/width; integer chunks at 8 bytes.
                let slot = sel.slot(ty.size_bytes().next_multiple_of(8).max(1) as u32, glcc_mir::ty::ty_align(&ty).max(1).max(8) as u32);
                sel.set_slot(p, slot);
                let stride = piece_stride(&ty);
                for (i, preg) in pregs.iter().take(n as usize).enumerate() {
                    let pv = sel.alloc_vreg_class(preg.class);
                    sel.emit(MInst::new(
                        OpId::MOVE,
                        Width::W64,
                        smallvec::smallvec![Operand::Reg(Reg::V(pv)), Operand::Reg(Reg::P(*preg))],
                    ));
                    piece_stores.push((pv, preg.class, stride, slot, i as u64 * stride, preg.class == crate::machine::RegClass::Vec));
                    match preg.class {
                        crate::machine::RegClass::Gpr => gp += 1,
                        crate::machine::RegClass::Vec => fp += 1,
                    }
                }
            }
            ParamLoc::Stack => {}
        }
    }
    for (pv, class, stride, slot, off, _float) in piece_stores {
        let dst = crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: off as i64, sym: None };
        // HFA members store at member width; integer chunks whole.
        let (op, width) = match class {
            crate::machine::RegClass::Vec if stride == 4 => (Op::FStore, Width::W32),
            crate::machine::RegClass::Vec => (Op::FStore, Width::W64),
            _ => (Op::Store, Width::W64),
        };
        sel.emit(MInst::new(op_id(op), width, smallvec::smallvec![Operand::Reg(Reg::V(pv)), Operand::Mem(dst)]));
    }
    for (p, ty, off) in stack_params {
        // The caller stored this argument at x29+16+off (after the
        // frame-pointer/link pair). Copy it into the parameter's home.
        if ty.is_aggregate() {
            let slot = sel.slot(ty.size_bytes().max(1) as u32, glcc_mir::ty::ty_align(&ty).max(1) as u32);
            sel.set_slot(p, slot);
            let src = crate::machine::MemOperand { base: Some(Reg::P(PReg::gpr(regs::X29))), index: None, scale: 0, disp: 16 + off, sym: None };
            let dst = crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: 0, sym: None };
            copy_bytes(sel, &dst, &src, ty.size_bytes(), glcc_mir::ty::ty_align(&ty).max(1).max(8) as u64);
        } else {
            let vr = sel.value_vreg(mir, p);
            let is_float = matches!(ty, Ty::Float(_));
            let (op, width) = if is_float {
                (Op::FLoad, float_width(mir, p))
            } else {
                (Op::Load, value_width(&ty))
            };
            sel.emit(MInst::new(
                op_id(op),
                width,
                smallvec::smallvec![
                    Operand::Reg(Reg::V(vr)),
                    Operand::Mem(crate::machine::MemOperand { base: Some(Reg::P(PReg::gpr(regs::X29))), index: None, scale: 0, disp: 16 + off, sym: None }),
                ],
            ));
        }
    }
    if mir.sig.variadic {
        // The variadic tail may arrive in any of the argument
        // registers: pin the whole file (named plus anonymous pads) so
        // the entry moves resolve as one parallel set, then spill every
        // register's value into the save area. Slots for registers the
        // caller did not use hold garbage; va_arg never reads past the
        // offsets it tracks.
        let named_gp = (gp_off / 8) as usize;
        let named_fp = (fp_off / 16) as usize;
        let mut tail_gp: Vec<(usize, crate::machine::VReg)> = Vec::new();
        for k in named_gp..regs::ARG_REGS.len() {
            let vr = sel.alloc_vreg();
            sel.emit(MInst::new(
                OpId::MOVE,
                Width::W64,
                smallvec::smallvec![Operand::Reg(Reg::V(vr)), Operand::Reg(Reg::P(regs::ARG_REGS[k]))],
            ));
            tail_gp.push((k, vr));
        }
        let mut tail_fp: Vec<(usize, crate::machine::VReg)> = Vec::new();
        for k in named_fp..regs::VEC_ARG_REGS.len() {
            let vr = sel.alloc_vreg_class(crate::machine::RegClass::Vec);
            sel.emit(MInst::new(
                OpId::MOVE,
                Width::W64,
                smallvec::smallvec![Operand::Reg(Reg::V(vr)), Operand::Reg(Reg::P(regs::VEC_ARG_REGS[k]))],
            ));
            tail_fp.push((k, vr));
        }
        for (k, vr) in gp_save.iter() {
            spill_save_area(sel, save_slot, k * 8, *vr, false);
        }
        for (k, vr) in tail_gp.iter() {
            spill_save_area(sel, save_slot, (*k * 8) as i64, *vr, false);
        }
        for (at, vr) in fp_save.iter() {
            spill_save_area(sel, save_slot, 64 + *at, *vr, true);
        }
        for (k, vr) in tail_fp.iter() {
            spill_save_area(sel, save_slot, 64 + (*k as i64) * 16, *vr, true);
        }
        // Descending signed cursors: negative means registers remain.
        sel.va_gp_off = (named_gp as i64 - 8) * 8;
        sel.va_fp_off = (named_fp as i64 - 8) * 16;
    }
    true
}

/// Byte stride of one register-classed aggregate piece: HFAs step at
/// member width, integer chunks at 8.
fn piece_stride(ty: &glcc_mir::Ty) -> u64 {
    match abi::hfa_pieces(ty) {
        Some(pieces) => pieces.first().copied().unwrap_or(8),
        None => 8,
    }
}

/// Spill one entry parameter into the va register save area.
fn spill_save_area(sel: &mut FuncSel, save_slot: u32, disp: i64, vr: crate::machine::VReg, float: bool) {
    let (op, width) = if float { (Op::FStore, Width::W64) } else { (Op::Store, Width::W64) };
    sel.emit(MInst::new(
        op_id(op),
        width,
        smallvec::smallvec![
            Operand::Reg(Reg::V(vr)),
            Operand::Mem(crate::machine::MemOperand { base: Some(Reg::FI(save_slot)), index: None, scale: 0, disp, sym: None }),
        ],
    ));
}

// -- instructions ----------------------------------------------------------

fn inst_hook(sel: &mut FuncSel, mir: &Function, inst: Inst) -> bool {
    let kind = &mir.insts[inst].kind;
    match kind {
        InstKind::Alloca { pointee, .. } => {
            let res = mir.insts[inst].result.expect("alloca has a result");
            let size = pointee.size_bytes().max(1) as u32;
            let align = glcc_mir::ty::ty_align(pointee).max(1) as u32;
            let slot = sel.slot(size, align);
            // The address materializes at the alloca (the frontend
            // places them in the entry block, so the definition
            // dominates every use).
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
            let dst = result_vreg(sel, mir, inst);
            let cond = emit_icmp(sel, mir, inst, *pred);
            sel.emit(MInst::new(op_id(Op::Cset), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Imm(cond as i64)]));
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
            sel.emit(MInst::new(op_id(Op::MovRR), result_width(sel, mir, inst), smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
            true
        }
        InstKind::Zext(_) => {
            let dst = result_vreg(sel, mir, inst);
            let src = single_operand(sel, mir, inst);
            let src_bits = scalar_bits_of(mir, single_value(mir, inst));
            // A 32-bit write zero-extends into the upper half, which
            // 64-bit consumers (shifts, div) rely on.
            let (op, width) = match src_bits {
                b if b <= 8 => (Op::Uxtb, Width::W32),
                b if b <= 16 => (Op::Uxth, Width::W32),
                b if b <= 32 => (Op::MovRR, Width::W32),
                _ => (Op::MovRR, Width::W64),
            };
            sel.emit(MInst::new(op_id(op), width, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
            true
        }
        InstKind::Sext(_) => {
            let dst = result_vreg(sel, mir, inst);
            let src = single_operand(sel, mir, inst);
            let src_bits = scalar_bits_of(mir, single_value(mir, inst));
            let (op, width) = match src_bits {
                b if b <= 8 => (Op::Sxtb, Width::W64),
                b if b <= 16 => (Op::Sxth, Width::W64),
                b if b <= 32 => (Op::Sxtw, Width::W64),
                _ => (Op::MovRR, Width::W64),
            };
            sel.emit(MInst::new(op_id(op), width, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
            true
        }
        InstKind::PtrToInt(_) | InstKind::IntToPtr(_) | InstKind::Bitcast(_) => {
            let dst = result_vreg(sel, mir, inst);
            let src_v = single_value(mir, inst);
            let src = single_operand(sel, mir, inst);
            let src_float = matches!(mir.values[src_v].ty, Ty::Float(_));
            let dst_float = matches!(mir.values[mir.insts[inst].result.expect("cast has a result")].ty, Ty::Float(_));
            let op = match (src_float, dst_float) {
                (false, false) => Op::MovRR,
                (true, false) => Op::VecToGpr,
                (false, true) => Op::GprToVec,
                (true, true) => Op::FMovRR,
            };
            sel.emit(MInst::new(op_id(op), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
            true
        }
        InstKind::Memcpy { dst, src, len } => {
            mem_intrinsic(sel, mir, inst, "memcpy", &[*dst, *src], Some((*len as i64, 0)));
            true
        }
        InstKind::Memset { dst, byte, len } => {
            mem_intrinsic(sel, mir, inst, "memset", &[*dst], Some((*len as i64, *byte as i64)));
            true
        }
        // seqcst scalar load/store are plain moves; a full fence is dmb
        // ish. Read-modify-write and CAS panic (like x86-64 for RMW).
        InstKind::AtomicLoad { addr } => {
            let dst = result_vreg(sel, mir, inst);
            if matches!(mir.values[mir.insts[inst].result.expect("load has a result")].ty, Ty::Float(_)) {
                panic!("glcc native backend: atomic float load is not yet supported");
            }
            let mem = sel.mem_of(mir, *addr);
            sel.emit(MInst::new(op_id(Op::Load), result_width(sel, mir, inst), smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Mem(mem)]));
            true
        }
        InstKind::AtomicStore { addr, val } => {
            let ty = mir.values[*val].ty.clone();
            if matches!(ty, Ty::Float(_)) {
                panic!("glcc native backend: atomic float store is not yet supported");
            }
            let src = sel.value_vreg(mir, *val);
            let mem = sel.mem_of(mir, *addr);
            sel.emit(MInst::new(op_id(Op::Store), value_width(&ty), smallvec::smallvec![Operand::Reg(Reg::V(src)), Operand::Mem(mem)]));
            true
        }
        InstKind::AtomicFence => {
            sel.emit(MInst::new(op_id(Op::Dmb), Width::W64, smallvec::smallvec![]));
            true
        }
        InstKind::AtomicRmw { .. } | InstKind::AtomicCas { .. } => {
            panic!("glcc native backend: atomic read-modify-write is not yet supported")
        }
        InstKind::InlineAsm { .. } => panic!("glcc native backend: inline asm is not yet supported"),
        InstKind::VaStart { ap } => {
            va_start(sel, mir, *ap);
            true
        }
        InstKind::VaArg { .. } => {
            va_arg(sel, mir, inst);
            true
        }
        InstKind::VaEnd { .. } => true,
        InstKind::VaCopy { dst, src } => {
            // The 32-byte tag: four 8-byte words (the last covers both
            // 32-bit offset fields).
            for k in 0..4i64 {
                let tmp = sel.alloc_vreg();
                let mut sm = sel.mem_of(mir, *src);
                sm.disp += k * 8;
                sel.emit(MInst::new(op_id(Op::Load), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Mem(sm)]));
                let mut dm = sel.mem_of(mir, *dst);
                dm.disp += k * 8;
                sel.emit(MInst::new(op_id(Op::Store), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Mem(dm)]));
            }
            true
        }
        InstKind::GetFramePointer => panic!("glcc native backend: GetFramePointer is not yet supported"),
        InstKind::EhTypeId { .. } => panic!("glcc native backend: EH is not yet supported"),
        InstKind::Ceil(_) | InstKind::Floor(_) => {
            // Baseline ARMv8-A has no rounding instruction, so these go
            // through libm.
            let src = single_value(mir, inst);
            let dst = result_vreg(sel, mir, inst);
            let arg = sel.value_vreg(mir, src);
            let name = match (kind, is_f64(mir, src)) {
                (InstKind::Ceil(_), true) => "ceil",
                (InstKind::Ceil(_), false) => "ceilf",
                (InstKind::Floor(_), true) => "floor",
                (InstKind::Floor(_), false) => "floorf",
                _ => unreachable!(),
            };
            let mut ops = smallvec::SmallVec::new();
            ops.push(Operand::FixedReg(Reg::V(dst), sel.isa.abi().vec_ret_reg()));
            ops.push(Operand::Sym(SymOperand { target: SymTarget::Named(name), reloc: crate::machine::RelocKind::Plt32, addend: 0 }));
            ops.push(Operand::Reg(Reg::V(arg)));
            let mut call = MInst::new(op_id(Op::Bl), Width::W64, ops);
            call.call_args = Some(smallvec::smallvec![sel.isa.abi().vec_arg_regs()[0]]);
            sel.emit(call);
            true
        }
        InstKind::FAdd(..) | InstKind::FSub(..) | InstKind::FMul(..) | InstKind::FDiv(..) | InstKind::FNeg(_) => {
            let (op, a, b) = match kind {
                InstKind::FAdd(a, b) => (Op::FAddRR, *a, *b),
                InstKind::FSub(a, b) => (Op::FSubRR, *a, *b),
                InstKind::FMul(a, b) => (Op::FMulRR, *a, *b),
                InstKind::FDiv(a, b) => (Op::FDivRR, *a, *b),
                InstKind::FNeg(_) => {
                    let dst = result_vreg(sel, mir, inst);
                    let src = single_operand(sel, mir, inst);
                    let width = float_width(mir, single_value(mir, inst));
                    sel.emit(MInst::new(op_id(Op::FNeg), width, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
                    return true;
                }
                _ => unreachable!(),
            };
            let width = float_width(mir, a);
            let dst = result_vreg(sel, mir, inst);
            let a = sel.value_vreg(mir, a);
            let b = sel.value_vreg(mir, b);
            sel.emit(MInst::new(op_id(op), width, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(a)), Operand::Reg(Reg::V(b))]));
            true
        }
        InstKind::Sqrt(_) => {
            let src_v = single_value(mir, inst);
            let src = single_operand(sel, mir, inst);
            let dst = result_vreg(sel, mir, inst);
            let width = float_width(mir, src_v);
            sel.emit(MInst::new(op_id(Op::FSqrt), width, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
            true
        }
        InstKind::Fcmp { .. } => {
            fcmp(sel, mir, inst);
            true
        }
        InstKind::SItoF(_) | InstKind::UItoF(_) => {
            let (dst, src) = (result_vreg(sel, mir, inst), single_operand(sel, mir, inst));
            let to_f64 = matches!(mir.values[mir.insts[inst].result.expect("cast has a result")].ty, Ty::Float(FloatKind::F64));
            let bits = scalar_bits_of(mir, single_value(mir, inst));
            let op = if matches!(kind, InstKind::SItoF(_)) { Op::Scvtf } else { Op::Ucvtf };
            // Width = source size; the imm slot names the destination
            // precision.
            let w = if bits <= 32 { Width::W32 } else { Width::W64 };
            sel.emit(MInst::new(
                op_id(op),
                w,
                smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src)), Operand::Imm(to_f64 as i64)],
            ));
            true
        }
        InstKind::SFtoI(_) | InstKind::UFtoI(_) => {
            let (dst, src) = (result_vreg(sel, mir, inst), single_operand(sel, mir, inst));
            let op = if matches!(kind, InstKind::SFtoI(_)) { Op::Fcvtzs } else { Op::Fcvtzu };
            let dst_bits = scalar_bits_of(mir, mir.insts[inst].result.expect("cast has a result"));
            // Width = destination size; the imm slot names the source
            // precision.
            let w = if dst_bits <= 32 { Width::W32 } else { Width::W64 };
            let src_d = is_f64(mir, single_value(mir, inst));
            sel.emit(MInst::new(
                op_id(op),
                w,
                smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src)), Operand::Imm(src_d as i64)],
            ));
            true
        }
        InstKind::Fptrunc(_) | InstKind::Fpext(_) => {
            let (dst, src) = (result_vreg(sel, mir, inst), single_operand(sel, mir, inst));
            let op = match kind {
                InstKind::Fptrunc(_) => Op::FCvtDS, // f64 -> f32
                _ => Op::FCvtSD,                    // f32 -> f64
            };
            sel.emit(MInst::new(op_id(op), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
            true
        }
        InstKind::Load { addr, .. } => {
            let res = mir.insts[inst].result.expect("load has a result");
            let ty = mir.values[res].ty.clone();
            check_no_quad(&ty);
            if matches!(ty, Ty::Float(_)) {
                let dst = result_vreg(sel, mir, inst);
                let mem = sel.mem_of(mir, *addr);
                let width = float_width(mir, res);
                sel.emit(MInst::new(op_id(Op::FLoad), width, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Mem(mem)]));
                return true;
            }
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
            let dst = result_vreg(sel, mir, inst);
            let mem = sel.mem_of(mir, *addr);
            sel.emit(MInst::new(op_id(Op::Load), result_width(sel, mir, inst), smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Mem(mem)]));
            true
        }
        InstKind::Store { addr, val, .. } => {
            let ty = mir.values[*val].ty.clone();
            check_no_quad(&ty);
            if matches!(ty, Ty::Float(_)) {
                let v = sel.value_vreg(mir, *val);
                let mem = sel.mem_of(mir, *addr);
                let width = float_width(mir, *val);
                sel.emit(MInst::new(op_id(Op::FStore), width, smallvec::smallvec![Operand::Reg(Reg::V(v)), Operand::Mem(mem)]));
                return true;
            }
            if ty.is_aggregate() {
                let src = sel.mem_of(mir, *val);
                let dst = sel.mem_of(mir, *addr);
                copy_bytes(sel, &dst, &src, ty.size_bytes(), glcc_mir::ty::ty_align(&ty).max(1) as u64);
                return true;
            }
            let src = sel.value_vreg(mir, *val);
            let mem = sel.mem_of(mir, *addr);
            sel.emit(MInst::new(op_id(Op::Store), value_width(&ty), smallvec::smallvec![Operand::Reg(Reg::V(src)), Operand::Mem(mem)]));
            true
        }
        InstKind::ExtractValue { agg, index } => {
            let res = mir.insts[inst].result.expect("extractvalue has a result");
            let agg_ty = mir.values[*agg].ty.clone();
            let Ty::Struct { fields, offsets, .. } = &agg_ty else {
                panic!("glcc native backend: extractvalue through {:?}", agg_ty);
            };
            let off = *offsets.get(*index as usize).expect("extractvalue index in range");
            let field_ty = fields[*index as usize].clone();
            check_no_quad(&field_ty);
            if field_ty.is_aggregate() {
                let slot = sel.slot(field_ty.size_bytes().max(1) as u32, glcc_mir::ty::ty_align(&field_ty).max(1) as u32);
                sel.set_slot(res, slot);
                let src = sel.mem_of(mir, *agg);
                let dstm = crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: 0, sym: None };
                copy_bytes(sel, &dstm, &src, field_ty.size_bytes(), glcc_mir::ty::ty_align(&field_ty).max(1) as u64);
            } else {
                let dst = result_vreg(sel, mir, inst);
                let base = sel.mem_of(mir, *agg);
                if matches!(field_ty, Ty::Float(_)) {
                    let width = match field_ty {
                        Ty::Float(FloatKind::F64) => Width::W64,
                        _ => Width::W32,
                    };
                    sel.emit(MInst::new(
                        op_id(Op::FLoad),
                        width,
                        smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Mem(crate::machine::MemOperand { disp: base.disp + off as i64, ..base })],
                    ));
                } else {
                    sel.emit(MInst::new(
                        op_id(Op::Load),
                        value_width(&field_ty),
                        smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Mem(crate::machine::MemOperand { disp: base.disp + off as i64, ..base })],
                    ));
                }
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
            check_no_quad(&field_ty);
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
                if matches!(field_ty, Ty::Float(_)) {
                    let width = match field_ty {
                        Ty::Float(FloatKind::F64) => Width::W64,
                        _ => Width::W32,
                    };
                    sel.emit(MInst::new(op_id(Op::FStore), width, smallvec::smallvec![Operand::Reg(Reg::V(v)), Operand::Mem(field_dst)]));
                } else {
                    sel.emit(MInst::new(op_id(Op::Store), value_width(&field_ty), smallvec::smallvec![Operand::Reg(Reg::V(v)), Operand::Mem(field_dst)]));
                }
            }
            true
        }
        // Integer ALU, loads, stores: the narrow-width invariant keeps
        // most of these canonicalization-free.
        InstKind::Add(..) | InstKind::Sub(..) | InstKind::And(..) | InstKind::Or(..) | InstKind::Xor(..) | InstKind::Mul(..) => {
            alu(sel, mir, inst);
            true
        }
        InstKind::Shl(..) | InstKind::LShr(..) | InstKind::AShr(..) => {
            shift(sel, mir, inst);
            true
        }
        InstKind::Neg(_) => {
            let dst = result_vreg(sel, mir, inst);
            let src = single_operand(sel, mir, inst);
            sel.emit(MInst::new(op_id(Op::Neg), result_width(sel, mir, inst), smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
            true
        }
        InstKind::Not(_) => {
            let dst = result_vreg(sel, mir, inst);
            let src = single_operand(sel, mir, inst);
            sel.emit(MInst::new(op_id(Op::Not), result_width(sel, mir, inst), smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
            true
        }
        InstKind::Bswap(_) => panic!("glcc native backend: bswap is not yet supported on aarch64"),
        _ => panic!("glcc native backend: cannot select {:?}", kind),
    }
}

// -- terminators -----------------------------------------------------------

fn term_hook(sel: &mut FuncSel, mir: &Function, block: Block) -> bool {
    let term = mir.blocks[block].term.clone().expect("verified terminator");
    match term {
        Terminator::Return { val } => {
            match val {
                Some(v) => {
                    let ty = mir.values[v].ty.clone();
                    check_no_quad(&ty);
                    if ty.is_aggregate() {
                        match sel.isa.abi().ret_loc(&ty) {
                            RetLoc::Regs(pregs, n) => {
                                // Pieces of a small aggregate return in
                                // x0(+x1) or v0(+v1); HFA pieces sit at
                                // member stride.
                                let addr = sel.mem_of(mir, v);
                                let stride = piece_stride(&ty);
                                let mut ops = smallvec::SmallVec::new();
                                for (i, preg) in pregs.iter().take(n as usize).enumerate() {
                                    let off = addr.disp + (i as i64) * stride as i64;
                                    let tmp = sel.alloc_vreg_class(preg.class);
                                    if preg.class == crate::machine::RegClass::Vec {
                                        let width = if stride >= 8 { Width::W64 } else { Width::W32 };
                                        sel.emit(MInst::new(op_id(Op::FLoad), width, smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Mem(crate::machine::MemOperand { disp: off, ..addr.clone() })]));
                                    } else {
                                        sel.emit(MInst::new(op_id(Op::Load), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Mem(crate::machine::MemOperand { disp: off, ..addr.clone() })]));
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
                                // AAPCS64: the callee returns the
                                // indirect-result pointer in x8.
                                sel.emit(MInst::new(
                                    OpId::MOVE,
                                    Width::W64,
                                    smallvec::smallvec![Operand::Reg(Reg::P(PReg::gpr(regs::X8))), Operand::Reg(Reg::V(sret))],
                                ));
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
            let cc = cond_cc(sel, mir, cond);
            sel.emit_branch(
                MInst::new(op_id(Op::BCond), Width::W64, smallvec::smallvec![Operand::Imm(cc as i64)]),
                &[then_id, else_id],
                &[&then_args, &else_args],
            );
            true
        }
        Terminator::Switch { scrutinee, cases, default } => {
            // Comparison chain; a later milestone adds jump tables. The
            // default target is every chain step's fallthrough, so no
            // explicit jump into it is emitted. Narrow scrutinees are
            // canonicalized once (cmp reads full 32-bit registers).
            if cases.is_empty() {
                sel.emit_branch(MInst::new(OpId::JMP, Width::W64, smallvec::smallvec![]), &[default.index() as BlockId], &[&[]]);
                return true;
            }
            let scrut = canonicalize(sel, mir, scrutinee, false);
            let mut next = sel.cur_block();
            for (i, &(value, target)) in cases.iter().enumerate() {
                sel.set_cur(next);
                // imm12 covers 0..4095 only; larger case values
                // materialize into a register first.
                if (0..=4095).contains(&(value as i64)) {
                    sel.emit(MInst::new(
                        op_id(Op::CmpI),
                        Width::W32,
                        smallvec::smallvec![Operand::Reg(Reg::V(scrut)), Operand::Imm(value as i64)],
                    ));
                } else {
                    let c = sel.alloc_vreg();
                    sel.emit(MInst::new(
                        sel.isa.const_move_op(value as i64),
                        Width::W32,
                        smallvec::smallvec![Operand::Reg(Reg::V(c)), Operand::Imm(value as i64)],
                    ));
                    sel.emit(MInst::new(
                        op_id(Op::CmpRR),
                        Width::W32,
                        smallvec::smallvec![Operand::Reg(Reg::V(scrut)), Operand::Reg(Reg::V(c))],
                    ));
                }
                let last = i + 1 == cases.len();
                let fallthrough = if last { default.index() as BlockId } else { sel.new_block() };
                sel.emit_branch(
                    MInst::new(op_id(Op::BCond), Width::W64, smallvec::smallvec![Operand::Imm(icc(glcc_mir::IcmpPred::Eq) as i64)]),
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

fn is_f64(mir: &Function, v: Value) -> bool {
    matches!(mir.values[v].ty, Ty::Float(FloatKind::F64))
}

fn float_width(mir: &Function, v: Value) -> Width {
    if is_f64(mir, v) {
        Width::W64
    } else {
        Width::W32
    }
}

/// F128 (long double on this target) panics loudly until the quad
/// libcall pass; F80 never appears on aarch64.
fn check_no_quad(ty: &Ty) {
    if matches!(ty, Ty::Float(FloatKind::F128 | FloatKind::F80)) {
        panic!("glcc native backend: {ty:?} arithmetic is not yet supported on aarch64");
    }
}

/// Canonicalize a narrow (sub-32-bit) value into a clean W32 register:
/// signed interpretation sign-extends, unsigned/equality zero-extends.
/// 32-bit and wider values pass through untouched.
fn canonicalize(sel: &mut FuncSel, mir: &Function, v: Value, signed: bool) -> crate::machine::VReg {
    let bits = scalar_bits_of(mir, v);
    let src = sel.value_vreg(mir, v);
    if bits >= 32 {
        return src;
    }
    let dst = sel.alloc_vreg();
    let op = match (signed, bits <= 8) {
        (true, true) => Op::Sxtb,
        (true, false) => Op::Sxth,
        (false, true) => Op::Uxtb,
        (false, false) => Op::Uxth,
    };
    sel.emit(MInst::new(op_id(op), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
    dst
}

/// Integer ALU with immediate folding. No canonicalization: add/sub/
/// mul/logic are correct in the low bits, and narrow consumers
/// canonicalize on their own.
fn alu(sel: &mut FuncSel, mir: &Function, inst: Inst) {
    let kind = mir.insts[inst].kind.clone();
    let dst = result_vreg(sel, mir, inst);
    let width = result_width(sel, mir, inst);
    let mut ops: Vec<Value> = Vec::new();
    kind.for_each_operand(&mut |v| ops.push(v));
    let (a, b) = (ops[0], ops[1]);
    let rhs_const = const_of(mir, b);
    match &kind {
        InstKind::Add(..) => {
            if let Some(c) = rhs_const.filter(|c| (0..=4095).contains(c)) {
                return imm_op(sel, mir, op_id(Op::AddI), width, dst, a, c);
            }
            if let Some(c) = rhs_const.filter(|c| (-4095..=-1).contains(c)) {
                return imm_op(sel, mir, op_id(Op::SubI), width, dst, a, -c);
            }
            rr(sel, mir, op_id(Op::AddRR), width, dst, a, b)
        }
        InstKind::Sub(..) => {
            if let Some(c) = rhs_const.filter(|c| (0..=4095).contains(c)) {
                return imm_op(sel, mir, op_id(Op::SubI), width, dst, a, c);
            }
            rr(sel, mir, op_id(Op::SubRR), width, dst, a, b)
        }
        InstKind::Mul(..) => {
            if let Some(c) = rhs_const.filter(|c| (0..=4095).contains(c)) {
                let av = sel.value_vreg(mir, a);
                sel.emit(MInst::new(
                    op_id(Op::MulI),
                    width,
                    smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(av)), Operand::Imm(c)],
                ));
                return;
            }
            rr(sel, mir, op_id(Op::MulRR), width, dst, a, b)
        }
        // Logic ops have no cheap immediate encoding (bitmask
        // immediates need a mask search): constants materialize into
        // registers.
        InstKind::And(..) => rr(sel, mir, op_id(Op::AndRR), width, dst, a, b),
        InstKind::Or(..) => rr(sel, mir, op_id(Op::OrRR), width, dst, a, b),
        InstKind::Xor(..) => rr(sel, mir, op_id(Op::EorRR), width, dst, a, b),
        _ => unreachable!(),
    }
}

fn imm_op(sel: &mut FuncSel, mir: &Function, op: OpId, width: Width, dst: crate::machine::VReg, a: Value, c: i64) {
    let av = sel.value_vreg(mir, a);
    sel.emit(MInst::new(op, width, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(av)), Operand::Imm(c)]));
}

fn rr(sel: &mut FuncSel, mir: &Function, op: OpId, width: Width, dst: crate::machine::VReg, a: Value, b: Value) {
    let av = sel.value_vreg(mir, a);
    let bv = sel.value_vreg(mir, b);
    sel.emit(MInst::new(op, width, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(av)), Operand::Reg(Reg::V(bv))]));
}

fn const_of(mir: &Function, v: Value) -> Option<i64> {
    match &mir.values[v].def {
        Def::Const(Const::Int { value, .. }) => i64::try_from(*value).ok(),
        Def::Const(Const::Null) => Some(0),
        _ => None,
    }
}

fn shift(sel: &mut FuncSel, mir: &Function, inst: Inst) {
    let kind = mir.insts[inst].kind.clone();
    let dst = result_vreg(sel, mir, inst);
    let width = result_width(sel, mir, inst);
    let mut ops: Vec<Value> = Vec::new();
    kind.for_each_operand(&mut |v| ops.push(v));
    let (a, b) = (ops[0], ops[1]);
    let (rr_op, i_op) = match &kind {
        InstKind::Shl(..) => (Op::ShlRR, Op::ShlI),
        InstKind::LShr(..) => (Op::LshrRR, Op::LshrI),
        InstKind::AShr(..) => (Op::AsrRR, Op::AsrI),
        _ => unreachable!(),
    };
    // Right shifts interpret the value: narrow widths canonicalize
    // first (asr sign-extends; lsl's count may also be narrow but the
    // shifted value is what canonicalization protects).
    let signed = matches!(&kind, InstKind::AShr(..));
    let a2 = canonicalize(sel, mir, a, signed);
    let b2 = canonicalize(sel, mir, b, false);
    if let Some(c) = const_of(mir, b).filter(|c| (0..=63).contains(c)) {
        sel.emit(MInst::new(
            op_id(i_op),
            width,
            smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(a2)), Operand::Imm(c)],
        ));
        return;
    }
    sel.emit(MInst::new(
        op_id(rr_op),
        width,
        smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(a2)), Operand::Reg(Reg::V(b2))],
    ));
}

/// Compare-and-flag for an `Icmp`: canonicalizes narrow operands (by
/// predicate signedness — AArch64 compares read whole W registers),
/// emits the compare, and returns the condition code.
fn emit_icmp(sel: &mut FuncSel, mir: &Function, inst: Inst, pred: glcc_mir::IcmpPred) -> u8 {
    let (pred, a_val, b_val) = cmp_operands(sel, mir, inst, pred);
    let width = value_width(&mir.values[a_val].ty);
    let signed = is_signed_pred(pred);
    let a2 = canonicalize(sel, mir, a_val, signed);
    let b2 = canonicalize(sel, mir, b_val, signed);
    sel.emit(MInst::new(
        op_id(Op::CmpRR),
        width,
        smallvec::smallvec![Operand::Reg(Reg::V(a2)), Operand::Reg(Reg::V(b2))],
    ));
    icc(pred)
}

/// Condition code for a branch/select condition: fuse a single-use
/// compare feeding the condition, else test the boolean itself
/// (zero-extended: bools are 0/1 bytes with possible garbage upper
/// halves from logic ops).
fn cond_cc(sel: &mut FuncSel, mir: &Function, cond: Value) -> u8 {
    let Def::Inst(cmp_inst) = mir.values[cond].def else { return bool_test(sel, mir, cond) };
    let InstKind::Icmp { pred, .. } = mir.insts[cmp_inst].kind else { return bool_test(sel, mir, cond) };
    if mir.use_count(cond) != 1 {
        return bool_test(sel, mir, cond);
    }
    sel.mark_fused(cond);
    emit_icmp(sel, mir, cmp_inst, pred)
}

fn bool_test(sel: &mut FuncSel, mir: &Function, cond: Value) -> u8 {
    let vr = canonicalize(sel, mir, cond, false);
    sel.emit(MInst::new(
        op_id(Op::TstRR),
        Width::W32,
        smallvec::smallvec![Operand::Reg(Reg::V(vr)), Operand::Reg(Reg::V(vr))],
    ));
    enc::cc::NE
}

/// Compare operands for `cmp a, b` (flags describe a - b). A constant
/// on the left swaps sides and mirrors the predicate.
fn cmp_operands(sel: &mut FuncSel, mir: &Function, inst: Inst, pred: glcc_mir::IcmpPred) -> (glcc_mir::IcmpPred, Value, Value) {
    let InstKind::Icmp { lhs, rhs, .. } = mir.insts[inst].kind else { unreachable!() };
    let lhs_const = is_const(mir, lhs);
    let rhs_const = is_const(mir, rhs);
    if lhs_const && !rhs_const {
        (swap(pred), rhs, lhs)
    } else {
        (pred, lhs, rhs)
    }
}

fn is_const(mir: &Function, v: Value) -> bool {
    matches!(mir.values[v].def, Def::Const(_))
}

/// Address of a function/global/extern. Symbols this module defines
/// use adrp+add with the full addend folded into both relocations;
/// everything else (extern declarations, extern data) goes through the
/// GOT — PIE linkers cannot resolve adrp+add against preemptible
/// symbols.
fn addr(sel: &mut FuncSel, mir: &Function, inst: Inst, target: SymTarget) {
    let dst = result_vreg(sel, mir, inst);
    let preemptible = match target {
        SymTarget::Global(g) => sel.module.globals[g.index()].external,
        SymTarget::Extern(_) => true,
        SymTarget::Func(_) | SymTarget::Named(_) => false,
    };
    let mem = crate::machine::MemOperand {
        base: None,
        index: None,
        scale: 0,
        disp: 0,
        sym: Some(SymOperand { target, reloc: crate::machine::RelocKind::PcRel32, addend: 0 }),
    };
    let op = if preemptible { Op::LeaGot } else { Op::Lea };
    sel.emit(MInst::new(op_id(op), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Mem(mem)]));
}

enum CallTarget {
    Direct(SymTarget),
    Indirect(Value),
}

/// Emit the call instruction with arguments/results as fixed-register
/// operands: the allocator inserts the argument moves itself and
/// resolves overlaps between the fixed targets.
fn emit_call_inst(sel: &mut FuncSel, mir: &Function, result: Option<Value>, target: CallTarget, args: &[Value], sig: &glcc_mir::Signature) {
    let mut ops: smallvec::SmallVec<[Operand; 4]> = smallvec::SmallVec::new();
    let mut ret_pieces: Option<(smallvec::SmallVec<[crate::machine::VReg; 2]>, u64)> = None;
    if let Some(res) = result {
        check_no_quad(&sig.ret);
        match sel.isa.abi().ret_loc(&sig.ret) {
            RetLoc::Void => {}
            RetLoc::Reg(preg) => {
                // The call defines a fresh vreg which BECOMES the result
                // value's register.
                let pin = sel.alloc_vreg_class(preg.class);
                sel.set_reg(res, pin);
                ops.push(Operand::FixedReg(Reg::V(pin), preg));
            }
            RetLoc::Regs(pregs, n) => {
                let mut temps = smallvec::SmallVec::<[crate::machine::VReg; 2]>::new();
                for preg in pregs.iter().take(n as usize) {
                    let tmp = sel.alloc_vreg_class(preg.class);
                    ops.push(Operand::FixedReg(Reg::V(tmp), *preg));
                    temps.push(tmp);
                }
                ret_pieces = Some((temps, n as u64));
            }
            RetLoc::Memory => {}
        }
    }
    match target {
        CallTarget::Direct(sym) => {
            ops.push(Operand::Sym(SymOperand { target: sym, reloc: crate::machine::RelocKind::Plt32, addend: 0 }));
        }
        CallTarget::Indirect(v) => {
            // The callee pointer moves through the dedicated pin
            // register (non-allocatable): an ordinary call operand
            // would let the allocator place it in an argument register,
            // which this call's own argument moves clobber.
            let vr = sel.value_vreg(mir, v);
            sel.emit(MInst::new(
                OpId::MOVE,
                Width::W64,
                smallvec::smallvec![Operand::Reg(Reg::P(PReg::gpr(regs::IP0))), Operand::Reg(Reg::V(vr))],
            ));
            ops.push(Operand::Reg(Reg::P(PReg::gpr(regs::IP0))));
        }
    }
    let mut gp = 0usize;
    let mut fp = 0usize;
    let mut idx = 0usize;
    let mut reg_args: smallvec::SmallVec<[PReg; 8]> = smallvec::SmallVec::new();
    let mut stack: Vec<(Value, i64, glcc_mir::Ty)> = Vec::new();
    let mut stack_off = 0i64;
    let mut sret_slot: Option<u32> = None;
    if let Some(res) = result {
        if matches!(sel.isa.abi().ret_loc(&sig.ret), RetLoc::Memory) {
            let ty = mir.values[res].ty.clone();
            let slot = sel.slot(ty.size_bytes().max(1) as u32, glcc_mir::ty::ty_align(&ty).max(1) as u32);
            let addr_vr = sel.alloc_vreg();
            sel.emit(MInst::new(
                op_id(Op::Lea),
                Width::W64,
                smallvec::smallvec![Operand::Reg(Reg::V(addr_vr)), Operand::Mem(crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: 0, sym: None })],
            ));
            // x8 is the indirect-result register and does not come out
            // of the x0-x7 pool.
            ops.push(Operand::FixedReg(Reg::V(addr_vr), PReg::gpr(regs::X8)));
            sel.set_slot(res, slot);
            sret_slot = Some(slot);
        }
    }
    for arg in args.iter() {
        let ty = sig.params.get(idx).unwrap_or(&mir.values[*arg].ty).clone();
        idx += 1;
        check_no_quad(&ty);
        match sel.isa.abi().param_loc(gp, fp, &ty) {
            ParamLoc::Reg(preg) => {
                match preg.class {
                    crate::machine::RegClass::Gpr => gp += 1,
                    crate::machine::RegClass::Vec => fp += 1,
                }
                reg_args.push(preg);
                if abi::by_reference(&ty) {
                    // Large composite: the caller copies it into a stack
                    // temporary and passes the address.
                    let slot = sel.slot(ty.size_bytes().max(1) as u32, glcc_mir::ty::ty_align(&ty).max(1).max(8) as u32);
                    let addr_vr = sel.alloc_vreg();
                    sel.emit(MInst::new(
                        op_id(Op::Lea),
                        Width::W64,
                        smallvec::smallvec![Operand::Reg(Reg::V(addr_vr)), Operand::Mem(crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: 0, sym: None })],
                    ));
                    let src = sel.mem_of(mir, *arg);
                    let dst = crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: 0, sym: None };
                    copy_bytes(sel, &dst, &src, ty.size_bytes(), glcc_mir::ty::ty_align(&ty).max(1).max(8) as u64);
                    ops.push(Operand::Reg(Reg::V(addr_vr)));
                } else {
                    ops.push(Operand::Reg(Reg::V(sel.value_vreg(mir, *arg))));
                }
            }
            ParamLoc::Regs(pregs, n) => {
                // Load each piece from the aggregate's address into a
                // temp pinned to its piece register. HFA pieces sit at
                // member stride; integer chunks at 8 bytes.
                let addr = sel.mem_of(mir, *arg);
                let stride = piece_stride(&ty);
                for (i, preg) in pregs.iter().take(n as usize).enumerate() {
                    let off = addr.disp + (i as i64) * stride as i64;
                    let tmp = sel.alloc_vreg_class(preg.class);
                    if preg.class == crate::machine::RegClass::Vec {
                        let width = if stride >= 8 { Width::W64 } else { Width::W32 };
                        sel.emit(MInst::new(op_id(Op::FLoad), width, smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Mem(crate::machine::MemOperand { disp: off, ..addr.clone() })]));
                    } else {
                        sel.emit(MInst::new(op_id(Op::Load), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Mem(crate::machine::MemOperand { disp: off, ..addr.clone() })]));
                    }
                    match preg.class {
                        crate::machine::RegClass::Gpr => gp += 1,
                        crate::machine::RegClass::Vec => fp += 1,
                    }
                    ops.push(Operand::FixedReg(Reg::V(tmp), *preg));
                }
            }
            ParamLoc::Stack => {
                let align = glcc_mir::ty::ty_align(&ty).max(1).max(8) as i64;
                stack_off = (stack_off + align - 1) / align * align;
                let slot = stack_off;
                stack_off += ty.size_bytes().next_multiple_of(8) as i64;
                stack.push((*arg, slot, ty.clone()));
            }
        }
    }
    let stack_bytes = if stack.is_empty() {
        0
    } else {
        let total = (stack_off as u64).next_multiple_of(16) as i64;
        let sp = PReg::gpr(regs::SP);
        let mut pre = Vec::new();
        abi::sp_adjust(&mut pre, sp, sp, -total);
        for inst in pre {
            sel.emit(inst);
        }
        for (arg, off, ty) in stack.iter() {
            if ty.is_aggregate() {
                let src = sel.mem_of(mir, *arg);
                let dst = crate::machine::MemOperand { base: Some(Reg::P(sp)), index: None, scale: 0, disp: *off, sym: None };
                copy_bytes(sel, &dst, &src, ty.size_bytes(), glcc_mir::ty::ty_align(ty).max(1).max(8) as u64);
                continue;
            }
            let v = sel.value_vreg(mir, *arg);
            if matches!(ty, Ty::Float(_)) {
                let width = if matches!(ty, Ty::Float(FloatKind::F64)) { Width::W64 } else { Width::W32 };
                sel.emit(MInst::new(
                    op_id(Op::FStore),
                    width,
                    smallvec::smallvec![Operand::Reg(Reg::V(v)), Operand::Mem(crate::machine::MemOperand { base: Some(Reg::P(sp)), index: None, scale: 0, disp: *off, sym: None })],
                ));
            } else {
                sel.emit(MInst::new(
                    op_id(Op::Store),
                    value_width(ty),
                    smallvec::smallvec![Operand::Reg(Reg::V(v)), Operand::Mem(crate::machine::MemOperand { base: Some(Reg::P(sp)), index: None, scale: 0, disp: *off, sym: None })],
                ));
            }
        }
        total
    };
    let op = match target {
        CallTarget::Direct(_) => Op::Bl,
        CallTarget::Indirect(_) => Op::Blr,
    };
    let mut call_inst = MInst::new(op_id(op), Width::W64, ops);
    call_inst.call_args = Some(reg_args);
    sel.emit(call_inst);
    let _ = sret_slot;
    if let (Some(res), Some((temps, n))) = (result, ret_pieces) {
        // Unpack the returned pieces into the result's home slot.
        let ty = mir.values[res].ty.clone();
        let slot = sel.slot(ty.size_bytes().next_multiple_of(8).max(1) as u32, glcc_mir::ty::ty_align(&ty).max(1).max(8) as u32);
        sel.set_slot(res, slot);
        let stride = piece_stride(&ty);
        for (i, tmp) in temps.iter().enumerate().take(n as usize) {
            let dst = crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: (i as i64) * stride as i64, sym: None };
            let is_vec = matches!(sel.isa.abi().ret_loc(&sig.ret), RetLoc::Regs(p, _) if p[i].class == crate::machine::RegClass::Vec);
            if is_vec {
                let width = if stride >= 8 { Width::W64 } else { Width::W32 };
                sel.emit(MInst::new(op_id(Op::FStore), width, smallvec::smallvec![Operand::Reg(Reg::V(*tmp)), Operand::Mem(dst)]));
            } else {
                sel.emit(MInst::new(op_id(Op::Store), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(*tmp)), Operand::Mem(dst)]));
            }
        }
    }
    if stack_bytes > 0 {
        let sp = PReg::gpr(regs::SP);
        let mut post = Vec::new();
        abi::sp_adjust(&mut post, sp, sp, stack_bytes);
        for inst in post {
            sel.emit(inst);
        }
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

/// Integer select: csel defines a fresh register (no reuse constraint).
/// Float select rides fcsel when the condition is a fusable compare
/// with a single condition code; pair predicates (eq/ne/le) and plain
/// bool conditions go through GPR copies.
fn select(sel: &mut FuncSel, mir: &Function, inst: Inst) {
    let InstKind::Select { cond, then_val, else_val } = mir.insts[inst].kind else { unreachable!() };
    let dst = result_vreg(sel, mir, inst);
    let then_vr = sel.value_vreg(mir, then_val);
    let else_vr = sel.value_vreg(mir, else_val);
    if matches!(mir.values[mir.insts[inst].result.expect("select has a result")].ty, Ty::Float(_)) {
        let width = float_width(mir, then_val);
        let mut cc: Option<u8> = None;
        if let Some((pred, a, b)) = fusable_fcmp(sel, mir, cond) {
            let fa = sel.value_vreg(mir, a);
            let fb = sel.value_vreg(mir, b);
            sel.emit(MInst::new(op_id(Op::FCmp), width, smallvec::smallvec![Operand::Reg(Reg::V(fa)), Operand::Reg(Reg::V(fb))]));
            if let Some(single) = single_cond_fcmp(pred) {
                sel.emit(MInst::new(
                    op_id(Op::FCsel),
                    width,
                    smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(then_vr)), Operand::Reg(Reg::V(else_vr)), Operand::Imm(single as i64)],
                ));
                return;
            }
            // Pair predicate: materialize the bool from the live flags
            // (fcmp's), then csel through GPR copies below.
            use glcc_mir::FcmpPred;
            let (cc1, cc2, logic) = match pred {
                FcmpPred::Eq => (enc::cc::EQ, Some(enc::cc::VC), Op::AndRR),
                FcmpPred::Ne => (enc::cc::NE, Some(enc::cc::VS), Op::OrRR),
                FcmpPred::Le => (enc::cc::LS, Some(enc::cc::VC), Op::AndRR),
                _ => unreachable!(),
            };
            let t1 = sel.alloc_vreg();
            sel.emit(MInst::new(op_id(Op::Cset), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(t1)), Operand::Imm(cc1 as i64)]));
            let bool_vr = match cc2 {
                Some(cc2) => {
                    let t2 = sel.alloc_vreg();
                    sel.emit(MInst::new(op_id(Op::Cset), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(t2)), Operand::Imm(cc2 as i64)]));
                    let b = sel.alloc_vreg();
                    sel.emit(MInst::new(
                        op_id(logic),
                        Width::W32,
                        smallvec::smallvec![Operand::Reg(Reg::V(b)), Operand::Reg(Reg::V(t1)), Operand::Reg(Reg::V(t2))],
                    ));
                    b
                }
                None => t1,
            };
            sel.emit(MInst::new(
                op_id(Op::TstRR),
                Width::W32,
                smallvec::smallvec![Operand::Reg(Reg::V(bool_vr)), Operand::Reg(Reg::V(bool_vr))],
            ));
            cc = Some(enc::cc::NE);
        }
        let cc = cc.unwrap_or_else(|| cond_cc(sel, mir, cond));
        let g_else = sel.alloc_vreg();
        let g_then = sel.alloc_vreg();
        let wv = if is_f64(mir, then_val) { Width::W64 } else { Width::W32 };
        sel.emit(MInst::new(op_id(Op::VecToGpr), wv, smallvec::smallvec![Operand::Reg(Reg::V(g_else)), Operand::Reg(Reg::V(else_vr))]));
        sel.emit(MInst::new(op_id(Op::VecToGpr), wv, smallvec::smallvec![Operand::Reg(Reg::V(g_then)), Operand::Reg(Reg::V(then_vr))]));
        let g_res = sel.alloc_vreg();
        sel.emit(MInst::new(
            op_id(Op::Csel),
            Width::W64,
            smallvec::smallvec![Operand::Reg(Reg::V(g_res)), Operand::Reg(Reg::V(g_then)), Operand::Reg(Reg::V(g_else)), Operand::Imm(cc as i64)],
        ));
        sel.emit(MInst::new(op_id(Op::GprToVec), wv, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(g_res))]));
        return;
    }
    let cc = cond_cc(sel, mir, cond);
    sel.emit(MInst::new(
        op_id(Op::Csel),
        result_width(sel, mir, inst),
        smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(then_vr)), Operand::Reg(Reg::V(else_vr)), Operand::Imm(cc as i64)],
    ));
}

/// An fcmp feeding a select/branch whose predicate maps to a single
/// condition code (ordered comparisons and NaN tests).
fn fusable_fcmp(sel: &mut FuncSel, mir: &Function, cond: Value) -> Option<(glcc_mir::FcmpPred, Value, Value)> {
    let Def::Inst(cmp_inst) = mir.values[cond].def else { return None };
    let InstKind::Fcmp { pred, .. } = mir.insts[cmp_inst].kind else { return None };
    if mir.use_count(cond) != 1 {
        return None;
    }
    sel.mark_fused(cond);
    let mut ops: Vec<Value> = Vec::new();
    mir.insts[cmp_inst].kind.for_each_operand(&mut |v| ops.push(v));
    Some((pred, ops[0], ops[1]))
}

/// Fcmp predicates expressible as one condition code. eq/ne/le need
/// flag pairs (NaN handling), so they materialize bools instead.
fn single_cond_fcmp(pred: glcc_mir::FcmpPred) -> Option<u8> {
    use glcc_mir::FcmpPred::*;
    match pred {
        Lt => Some(enc::cc::MI),
        Gt => Some(enc::cc::GT),
        Ge => Some(enc::cc::GE),
        Uno => Some(enc::cc::VS),
        Ord => Some(enc::cc::VC),
        _ => None,
    }
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
/// size is a power of two, multiplied into a temp otherwise). GEP
/// indexes arrive 64-bit-clean from lowering.
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
    // power-of-two-sized term rides the Lea's shifted-register scale;
    // anything else is multiplied into a temp and summed.
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
                op_id(Op::MulI),
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
    let mem = crate::machine::MemOperand {
        base: Some(base_reg),
        index: index.map(Reg::V),
        scale,
        disp,
        sym: None,
    };
    sel.emit(MInst::new(op_id(Op::Lea), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Mem(mem)]));
}

/// Integer divide/remainder: sdiv, then msub for the remainder. Narrow
/// operands canonicalize first (the register forms read whole W/X
/// registers).
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
    let result = sel.value_vreg(mir, mir.insts[inst].result.expect("div has a result"));
    let width = result_width(sel, mir, inst);
    let a_vr = canonicalize(sel, mir, a, signed);
    let b_vr = canonicalize(sel, mir, b, signed);
    if !want_rem {
        let op = if signed { Op::SdivRR } else { Op::UdivRR };
        sel.emit(MInst::new(
            op_id(op),
            width,
            smallvec::smallvec![Operand::Reg(Reg::V(result)), Operand::Reg(Reg::V(a_vr)), Operand::Reg(Reg::V(b_vr))],
        ));
        return;
    }
    let op = if signed { Op::Srem } else { Op::Urem };
    sel.emit(MInst::new(
        op_id(op),
        width,
        smallvec::smallvec![Operand::Reg(Reg::V(result)), Operand::Reg(Reg::V(a_vr)), Operand::Reg(Reg::V(b_vr))],
    ));
}

/// memcpy/memset lowered as libcalls against the platform libc. The
/// length (and memset byte) are constants here, materialized into
/// fresh vregs. Arguments ride the call's `call_args` pins: the
/// post-RA move run resolves them as one parallel set.
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
    let mut call = MInst::new(op_id(Op::Bl), Width::W64, ops);
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
        let width = match unit {
            8 => Width::W64,
            4 => Width::W32,
            2 => Width::W16,
            _ => Width::W8,
        };
        let tmp = sel.alloc_vreg();
        sel.emit(MInst::new(
            op_id(Op::Load),
            width,
            smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Mem(crate::machine::MemOperand { disp: src.disp + off, ..src.clone() })],
        ));
        sel.emit(MInst::new(
            op_id(Op::Store),
            width,
            smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Mem(crate::machine::MemOperand { disp: dst.disp + off, ..dst.clone() })],
        ));
        off += unit as i64;
    }
}

// -- float ------------------------------------------------------------------

/// Floating compare: fcmp sets NZCV with unordered = {Z=1, C=1, V=1}.
/// eq/ne/le need a flag pair (NaN handling); lt/gt/ge/uno/ord are
/// single conditions. Cset chain produces the 0/1 byte.
fn fcmp(sel: &mut FuncSel, mir: &Function, inst: Inst) {
    let InstKind::Fcmp { pred, .. } = mir.insts[inst].kind else { unreachable!() };
    let mut ops: Vec<Value> = Vec::new();
    mir.insts[inst].kind.for_each_operand(&mut |v| ops.push(v));
    let (a, b) = (ops[0], ops[1]);
    let width = float_width(mir, a);
    let a = sel.value_vreg(mir, a);
    let b = sel.value_vreg(mir, b);
    sel.emit(MInst::new(op_id(Op::FCmp), width, smallvec::smallvec![Operand::Reg(Reg::V(a)), Operand::Reg(Reg::V(b))]));
    let dst = result_vreg(sel, mir, inst);
    use glcc_mir::FcmpPred::*;
    let (cc1, cc2): (u8, Option<u8>) = match pred {
        Eq => (enc::cc::EQ, Some(enc::cc::VC)), // eq & !unordered
        Ne => (enc::cc::NE, Some(enc::cc::VS)), // ne | unordered
        Lt => (enc::cc::MI, None),
        Le => (enc::cc::LS, Some(enc::cc::VC)), // ls & !unordered
        Gt => (enc::cc::GT, None),
        Ge => (enc::cc::GE, None),
        Uno => (enc::cc::VS, None),
        Ord => (enc::cc::VC, None),
    };
    if let Some(cc2) = cc2 {
        let t1 = sel.alloc_vreg();
        sel.emit(MInst::new(op_id(Op::Cset), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(t1)), Operand::Imm(cc1 as i64)]));
        let t2 = sel.alloc_vreg();
        sel.emit(MInst::new(op_id(Op::Cset), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(t2)), Operand::Imm(cc2 as i64)]));
        let logic = if pred == Ne { Op::OrRR } else { Op::AndRR };
        sel.emit(MInst::new(
            op_id(logic),
            Width::W32,
            smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(t1)), Operand::Reg(Reg::V(t2))],
        ));
        return;
    }
    sel.emit(MInst::new(op_id(Op::Cset), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Imm(cc1 as i64)]));
}

// -- va_list (AAPCS64) -------------------------------------------------------

/// Seed the 32-byte tag {__stack, __gr_top, __vr_top, __gr_offs,
/// __vr_offs}. Offsets are descending and signed; negative means
/// registers remain.
fn va_start(sel: &mut FuncSel, mir: &Function, ap: Value) {
    let save_slot = sel.va_save_slot.expect("va_start without a variadic prelude");
    let mut field = |sel: &mut FuncSel, vr: crate::machine::VReg, disp: i64, width: Width| {
        let mut m = sel.mem_of(mir, ap);
        m.disp += disp;
        sel.emit(MInst::new(op_id(Op::Store), width, smallvec::smallvec![Operand::Reg(Reg::V(vr)), Operand::Mem(m)]));
    };
    let ovf = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::Lea),
        Width::W64,
        smallvec::smallvec![
            Operand::Reg(Reg::V(ovf)),
            Operand::Mem(crate::machine::MemOperand { base: Some(Reg::P(PReg::gpr(regs::X29))), index: None, scale: 0, disp: sel.va_overflow_disp, sym: None }),
        ],
    ));
    field(sel, ovf, 0, Width::W64);
    let gr_top = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::Lea),
        Width::W64,
        smallvec::smallvec![
            Operand::Reg(Reg::V(gr_top)),
            Operand::Mem(crate::machine::MemOperand { base: Some(Reg::FI(save_slot)), index: None, scale: 0, disp: 64, sym: None }),
        ],
    ));
    field(sel, gr_top, 8, Width::W64);
    let vr_top = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::Lea),
        Width::W64,
        smallvec::smallvec![
            Operand::Reg(Reg::V(vr_top)),
            Operand::Mem(crate::machine::MemOperand { base: Some(Reg::FI(save_slot)), index: None, scale: 0, disp: 192, sym: None }),
        ],
    ));
    field(sel, vr_top, 16, Width::W64);
    let goff = sel.alloc_vreg();
    sel.emit(MInst::new(sel.isa.const_move_op(sel.va_gp_off), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(goff)), Operand::Imm(sel.va_gp_off)]));
    field(sel, goff, 24, Width::W32);
    let voff = sel.alloc_vreg();
    sel.emit(MInst::new(sel.isa.const_move_op(sel.va_fp_off), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(voff)), Operand::Imm(sel.va_fp_off)]));
    field(sel, voff, 28, Width::W32);
}

/// One scalar read through a register/stack cursor pair: GP is
/// (offset@24, top@8, step 8); VR is (offset@28, top@16, step 16).
/// Negative offsets read the save area; the stack pointer bumps only
/// on the stack path.
fn va_scalar_path(sel: &mut FuncSel, mir: &Function, ap: Value, off_field: i64, top_field: i64, step: i64, dst: crate::machine::VReg, width: Width, float: bool) {
    let ap_addr = sel.value_vreg(mir, ap);
    let load = |sel: &mut FuncSel, base: crate::machine::Reg, disp: i64, width: Width, dst: crate::machine::VReg, float: bool| {
        let op = if float { Op::FLoad } else { Op::Load };
        sel.emit(MInst::new(
            op_id(op),
            width,
            smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Mem(crate::machine::MemOperand { base: Some(base), index: None, scale: 0, disp, sym: None })],
        ));
    };
    let offs = sel.alloc_vreg();
    load(sel, Reg::V(ap_addr), off_field, Width::W32, offs, false);
    let top = sel.alloc_vreg();
    load(sel, Reg::V(ap_addr), top_field, Width::W64, top, false);
    let stack = sel.alloc_vreg();
    load(sel, Reg::V(ap_addr), 0, Width::W64, stack, false);
    // in_reg = offs < 0 (signed).
    sel.emit(MInst::new(
        op_id(Op::CmpI),
        Width::W32,
        smallvec::smallvec![Operand::Reg(Reg::V(offs)), Operand::Imm(0)],
    ));
    let in_reg = sel.alloc_vreg();
    sel.emit(MInst::new(op_id(Op::Cset), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(in_reg)), Operand::Imm(enc::cc::LT as i64)]));
    let offs_wide = sel.alloc_vreg();
    sel.emit(MInst::new(op_id(Op::Sxtw), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(offs_wide)), Operand::Reg(Reg::V(offs))]));
    let reg_addr = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::AddRR),
        Width::W64,
        smallvec::smallvec![Operand::Reg(Reg::V(reg_addr)), Operand::Reg(Reg::V(top)), Operand::Reg(Reg::V(offs_wide))],
    ));
    let addr = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::Csel),
        Width::W64,
        smallvec::smallvec![Operand::Reg(Reg::V(addr)), Operand::Reg(Reg::V(reg_addr)), Operand::Reg(Reg::V(stack)), Operand::Imm(enc::cc::NE as i64)],
    ));
    load(sel, Reg::V(addr), 0, width, dst, float);
    // Advance the cursor (registers) and the stack pointer (stack path).
    let bumped = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::AddI),
        Width::W32,
        smallvec::smallvec![Operand::Reg(Reg::V(bumped)), Operand::Reg(Reg::V(offs)), Operand::Imm(step)],
    ));
    let new_offs = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::Csel),
        Width::W32,
        smallvec::smallvec![Operand::Reg(Reg::V(new_offs)), Operand::Reg(Reg::V(bumped)), Operand::Reg(Reg::V(offs)), Operand::Imm(enc::cc::NE as i64)],
    ));
    let mut off_m = sel.mem_of(mir, ap);
    off_m.disp += off_field;
    sel.emit(MInst::new(op_id(Op::Store), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(new_offs)), Operand::Mem(off_m)]));
    let stack8 = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::AddI),
        Width::W64,
        smallvec::smallvec![Operand::Reg(Reg::V(stack8)), Operand::Reg(Reg::V(stack)), Operand::Imm(8)],
    ));
    let new_stack = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::Csel),
        Width::W64,
        smallvec::smallvec![Operand::Reg(Reg::V(new_stack)), Operand::Reg(Reg::V(stack)), Operand::Reg(Reg::V(stack8)), Operand::Imm(enc::cc::NE as i64)],
    ));
    let mut stack_m = sel.mem_of(mir, ap);
    sel.emit(MInst::new(op_id(Op::Store), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(new_stack)), Operand::Mem(stack_m)]));
}

/// Round a 16-byte-aligned composite's GP cursor toward zero to an even
/// slot (a 16-aligned argument starts at an even X register).
fn va_gp_align16(sel: &mut FuncSel, mir: &Function, ap: Value) {
    let ap_addr = sel.value_vreg(mir, ap);
    let offs = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::Load),
        Width::W32,
        smallvec::smallvec![Operand::Reg(Reg::V(offs)), Operand::Mem(crate::machine::MemOperand { base: Some(Reg::V(ap_addr)), index: None, scale: 0, disp: 24, sym: None })],
    ));
    sel.emit(MInst::new(
        op_id(Op::CmpI),
        Width::W32,
        smallvec::smallvec![Operand::Reg(Reg::V(offs)), Operand::Imm(0)],
    ));
    let in_regs = sel.alloc_vreg();
    sel.emit(MInst::new(op_id(Op::Cset), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(in_regs)), Operand::Imm(enc::cc::LT as i64)]));
    // (offs + 15) & -16 rounds -56 up to -48 in place.
    let plus = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::AddI),
        Width::W32,
        smallvec::smallvec![Operand::Reg(Reg::V(plus)), Operand::Reg(Reg::V(offs)), Operand::Imm(15)],
    ));
    let mask = sel.alloc_vreg();
    sel.emit(MInst::new(
        sel.isa.const_move_op(-16),
        Width::W32,
        smallvec::smallvec![Operand::Reg(Reg::V(mask)), Operand::Imm(-16)],
    ));
    let rounded = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::AndRR),
        Width::W32,
        smallvec::smallvec![Operand::Reg(Reg::V(rounded)), Operand::Reg(Reg::V(plus)), Operand::Reg(Reg::V(mask))],
    ));
    let new_offs = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::Csel),
        Width::W32,
        smallvec::smallvec![Operand::Reg(Reg::V(new_offs)), Operand::Reg(Reg::V(rounded)), Operand::Reg(Reg::V(offs)), Operand::Imm(enc::cc::NE as i64)],
    ));
    let mut off_m = sel.mem_of(mir, ap);
    off_m.disp += 24;
    sel.emit(MInst::new(op_id(Op::Store), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(new_offs)), Operand::Mem(off_m)]));
}

/// Read the next variadic argument. Scalars read through one
/// register/stack path; aggregates land in a fresh slot whose address
/// is the result (aggregates are addresses in MIR).
fn va_arg(sel: &mut FuncSel, mir: &Function, inst: Inst) -> bool {
    let InstKind::VaArg { ap } = mir.insts[inst].kind else { unreachable!() };
    let res = mir.insts[inst].result.expect("va_arg has a result");
    let res_ty = mir.values[res].ty.clone();
    check_no_quad(&res_ty);
    if matches!(res_ty, Ty::WideInt { .. }) {
        panic!("glcc native backend: va_arg of {res_ty:?} is not yet supported");
    }
    if res_ty.is_aggregate() {
        return va_arg_aggregate(sel, mir, inst, ap);
    }
    let is_float = matches!(res_ty, Ty::Float(_));
    let (off_field, top_field, step) = if is_float { (28i64, 16i64, 16i64) } else { (24, 8, 8) };
    let dst = result_vreg(sel, mir, inst);
    let width = if is_float { float_width(mir, res) } else { value_width(&res_ty) };
    va_scalar_path(sel, mir, ap, off_field, top_field, step, dst, width, is_float);
    true
}

/// Aggregate va_arg: multi-piece arguments follow the whole-argument
/// rule (one fit check gates every piece and the cursor updates
/// together); single pieces read through the scalar path. By-reference
/// aggregates (over 16 bytes) read the pointer from the GP path and
/// copy.
fn va_arg_aggregate(sel: &mut FuncSel, mir: &Function, inst: Inst, ap: Value) -> bool {
    let res = mir.insts[inst].result.expect("va_arg has a result");
    let ty = mir.values[res].ty.clone();
    let size = ty.size_bytes();
    let ap_addr = sel.value_vreg(mir, ap);
    if size > 16 {
        // By reference: the pointer rides the GP path.
        let ptr = sel.alloc_vreg();
        va_scalar_path(sel, mir, ap, 24, 8, 8, ptr, Width::W64, false);
        let slot = sel.slot(size.max(1) as u32, glcc_mir::ty::ty_align(&ty).max(1).max(8) as u32);
        sel.set_slot(res, slot);
        let dst = crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: 0, sym: None };
        let src = crate::machine::MemOperand { base: Some(Reg::V(ptr)), index: None, scale: 0, disp: 0, sym: None };
        copy_bytes(sel, &dst, &src, size, glcc_mir::ty::ty_align(&ty).max(1).max(8) as u64);
        return true;
    }
    // 16-byte-aligned composites start at an even X register.
    if glcc_mir::ty::ty_align(&ty) == 16 {
        va_gp_align16(sel, mir, ap);
    }
    // Piece layout: HFA members at member width, else 8-byte chunks.
    let pieces: Vec<(u64, bool, u64)> = match abi::hfa_pieces(&ty) {
        Some(widths) => {
            let mut off = 0u64;
            widths.iter().map(|&w| { let p = (off, true, w); off += w; p }).collect()
        }
        None => {
            let chunks = size.div_ceil(8);
            (0..chunks).map(|i| (i * 8, false, 8)).collect()
        }
    };
    let slot = sel.slot(size.max(1) as u32, glcc_mir::ty::ty_align(&ty).max(1).max(8) as u32);
    sel.set_slot(res, slot);
    if pieces.len() == 1 {
        // One piece: the scalar path reads it whole.
        let (off, float, width) = pieces[0];
        let tmp = sel.alloc_vreg_class(if float { crate::machine::RegClass::Vec } else { crate::machine::RegClass::Gpr });
        let (off_field, top_field, step) = if float { (28i64, 16i64, 16i64) } else { (24, 8, 8) };
        va_scalar_path(sel, mir, ap, off_field, top_field, step, tmp, if width == 8 { Width::W64 } else { Width::W32 }, float);
        let dst = crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: off as i64, sym: None };
        let (op, w) = if float {
            (Op::FStore, if width == 8 { Width::W64 } else { Width::W32 })
        } else {
            (Op::Store, Width::W64)
        };
        sel.emit(MInst::new(op_id(op), w, smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Mem(dst)]));
        return true;
    }
    // Whole-argument read: gp_end <= 0 && fp_end <= 0 means every
    // piece still sits in the register files.
    let load32 = |sel: &mut FuncSel, disp: i64, dst: crate::machine::VReg| {
        sel.emit(MInst::new(
            op_id(Op::Load),
            Width::W32,
            smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Mem(crate::machine::MemOperand { base: Some(Reg::V(ap_addr)), index: None, scale: 0, disp, sym: None })],
        ));
    };
    let load64 = |sel: &mut FuncSel, disp: i64, dst: crate::machine::VReg| {
        sel.emit(MInst::new(
            op_id(Op::Load),
            Width::W64,
            smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Mem(crate::machine::MemOperand { base: Some(Reg::V(ap_addr)), index: None, scale: 0, disp, sym: None })],
        ));
    };
    let gp_cur = sel.alloc_vreg();
    load32(sel, 24, gp_cur);
    let fp_cur = sel.alloc_vreg();
    load32(sel, 28, fp_cur);
    let gp_base = sel.alloc_vreg();
    load64(sel, 8, gp_base);
    let fp_base = sel.alloc_vreg();
    load64(sel, 16, fp_base);
    let stack = sel.alloc_vreg();
    load64(sel, 0, stack);
    let gp_bytes = pieces.iter().filter(|(_, f, _)| !*f).count() as i64 * 8;
    let fp_bytes = pieces.iter().filter(|(_, f, _)| *f).count() as i64 * 16;
    let gp_end = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::AddI),
        Width::W32,
        smallvec::smallvec![Operand::Reg(Reg::V(gp_end)), Operand::Reg(Reg::V(gp_cur)), Operand::Imm(gp_bytes)],
    ));
    let fp_end = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::AddI),
        Width::W32,
        smallvec::smallvec![Operand::Reg(Reg::V(fp_end)), Operand::Reg(Reg::V(fp_cur)), Operand::Imm(fp_bytes)],
    ));
    sel.emit(MInst::new(op_id(Op::CmpI), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(gp_end)), Operand::Imm(0)]));
    let gp_ok = sel.alloc_vreg();
    sel.emit(MInst::new(op_id(Op::Cset), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(gp_ok)), Operand::Imm(enc::cc::LE as i64)]));
    sel.emit(MInst::new(op_id(Op::CmpI), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(fp_end)), Operand::Imm(0)]));
    let fp_ok = sel.alloc_vreg();
    sel.emit(MInst::new(op_id(Op::Cset), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(fp_ok)), Operand::Imm(enc::cc::LE as i64)]));
    let in_regs = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::AndRR),
        Width::W32,
        smallvec::smallvec![Operand::Reg(Reg::V(in_regs)), Operand::Reg(Reg::V(gp_ok)), Operand::Reg(Reg::V(fp_ok))],
    ));
    // Fetch each piece into the home slot from either path.
    let mut gp_at = 0i64;
    let mut fp_at = 0i64;
    for (off, float, width) in &pieces {
        let (cur, base, piece_at, step) = if *float {
            let at = fp_at;
            fp_at += 16;
            (fp_cur, fp_base, at, 16i64)
        } else {
            let at = gp_at;
            gp_at += 8;
            (gp_cur, gp_base, at, 8)
        };
        let cur_wide = sel.alloc_vreg();
        sel.emit(MInst::new(op_id(Op::Sxtw), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(cur_wide)), Operand::Reg(Reg::V(cur))]));
        let reg_base = sel.alloc_vreg();
        sel.emit(MInst::new(
            op_id(Op::AddRR),
            Width::W64,
            smallvec::smallvec![Operand::Reg(Reg::V(reg_base)), Operand::Reg(Reg::V(base)), Operand::Reg(Reg::V(cur_wide))],
        ));
        let reg_addr = sel.alloc_vreg();
        if piece_at != 0 {
            sel.emit(MInst::new(
                op_id(Op::AddI),
                Width::W64,
                smallvec::smallvec![Operand::Reg(Reg::V(reg_addr)), Operand::Reg(Reg::V(reg_base)), Operand::Imm(piece_at)],
            ));
        } else {
            sel.emit(MInst::new(
                op_id(Op::MovRR),
                Width::W64,
                smallvec::smallvec![Operand::Reg(Reg::V(reg_addr)), Operand::Reg(Reg::V(reg_base))],
            ));
        }
        let stack_addr = sel.alloc_vreg();
        sel.emit(MInst::new(
            op_id(Op::AddI),
            Width::W64,
            smallvec::smallvec![Operand::Reg(Reg::V(stack_addr)), Operand::Reg(Reg::V(stack)), Operand::Imm(*off as i64)],
        ));
        let addr = sel.alloc_vreg();
        sel.emit(MInst::new(
            op_id(Op::Csel),
            Width::W64,
            smallvec::smallvec![Operand::Reg(Reg::V(addr)), Operand::Reg(Reg::V(reg_addr)), Operand::Reg(Reg::V(stack_addr)), Operand::Imm(enc::cc::NE as i64)],
        ));
        let tmp = sel.alloc_vreg_class(if *float { crate::machine::RegClass::Vec } else { crate::machine::RegClass::Gpr });
        let (lop, lw) = if *float {
            (Op::FLoad, if *width == 8 { Width::W64 } else { Width::W32 })
        } else {
            (Op::Load, Width::W64)
        };
        sel.emit(MInst::new(
            op_id(lop),
            lw,
            smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Mem(crate::machine::MemOperand { base: Some(Reg::V(addr)), index: None, scale: 0, disp: 0, sym: None })],
        ));
        let (sop, sw) = if *float {
            (Op::FStore, if *width == 8 { Width::W64 } else { Width::W32 })
        } else {
            (Op::Store, Width::W64)
        };
        let dst = crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: *off as i64, sym: None };
        sel.emit(MInst::new(op_id(sop), sw, smallvec::smallvec![Operand::Reg(Reg::V(tmp)), Operand::Mem(dst)]));
    }
    // Cursor updates: register paths move by their byte counts, the
    // stack pointer only on the stack path.
    let new_gp = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::Csel),
        Width::W32,
        smallvec::smallvec![Operand::Reg(Reg::V(new_gp)), Operand::Reg(Reg::V(gp_end)), Operand::Reg(Reg::V(gp_cur)), Operand::Imm(enc::cc::NE as i64)],
    ));
    let mut gm = sel.mem_of(mir, ap);
    gm.disp += 24;
    sel.emit(MInst::new(op_id(Op::Store), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(new_gp)), Operand::Mem(gm)]));
    let new_fp = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::Csel),
        Width::W32,
        smallvec::smallvec![Operand::Reg(Reg::V(new_fp)), Operand::Reg(Reg::V(fp_end)), Operand::Reg(Reg::V(fp_cur)), Operand::Imm(enc::cc::NE as i64)],
    ));
    let mut fm = sel.mem_of(mir, ap);
    fm.disp += 28;
    sel.emit(MInst::new(op_id(Op::Store), Width::W32, smallvec::smallvec![Operand::Reg(Reg::V(new_fp)), Operand::Mem(fm)]));
    let stack_step = size.next_multiple_of(8) as i64;
    let stack8 = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::AddI),
        Width::W64,
        smallvec::smallvec![Operand::Reg(Reg::V(stack8)), Operand::Reg(Reg::V(stack)), Operand::Imm(stack_step)],
    ));
    let new_stack = sel.alloc_vreg();
    sel.emit(MInst::new(
        op_id(Op::Csel),
        Width::W64,
        smallvec::smallvec![Operand::Reg(Reg::V(new_stack)), Operand::Reg(Reg::V(stack)), Operand::Reg(Reg::V(stack8)), Operand::Imm(enc::cc::NE as i64)],
    ));
    let mut sm = sel.mem_of(mir, ap);
    sel.emit(MInst::new(op_id(Op::Store), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(new_stack)), Operand::Mem(sm)]));
    true
}
