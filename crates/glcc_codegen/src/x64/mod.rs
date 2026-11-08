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
    if mir.sig.variadic {
        panic!("glcc native backend: defining variadic functions is not yet supported");
    }
    let entry_params: Vec<Value> = mir.blocks[mir.entry].params.clone();
    for (idx, &p) in entry_params.iter().enumerate() {
        match sel.isa.abi().param_loc(idx, &mir.values[p].ty) {
            ParamLoc::Reg(preg) => {
                let vr = sel.value_vreg(mir, p);
                sel.emit(MInst::new(
                    OpId::MOVE,
                    Width::W64,
                    smallvec::smallvec![Operand::Reg(Reg::V(vr)), Operand::Reg(Reg::P(preg))],
                ));
            }
            ParamLoc::Stack => panic!("glcc native backend: stack parameters are not yet supported"),
        }
    }
    true
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
            let src = single_operand(sel, mir, inst);
            sel.emit(MInst::new(op_id(Op::MovRR), Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(src))]));
            true
        }
        InstKind::Memcpy { dst, src, len } => {
            mem_intrinsic(sel, mir, inst, "memcpy", &[*dst, *src], Some((*len as i64, 0)));
            true
        }
        InstKind::Memset { dst, byte, len } => {
            mem_intrinsic(sel, mir, inst, "memset", &[*dst, Value::new(0)], Some((*len as i64, *byte as i64)));
            true
        }
        InstKind::AtomicLoad { .. } | InstKind::AtomicStore { .. } | InstKind::AtomicRmw { .. } | InstKind::AtomicCas { .. } | InstKind::AtomicFence => {
            panic!("glcc native backend: atomics are not yet supported")
        }
        InstKind::InlineAsm { .. } => panic!("glcc native backend: inline asm is not yet supported"),
        InstKind::VaStart { .. } | InstKind::VaArg { .. } | InstKind::VaEnd { .. } | InstKind::VaCopy { .. } => {
            panic!("glcc native backend: variadic helpers are not yet supported")
        }
        InstKind::GetFramePointer => panic!("glcc native backend: GetFramePointer is not yet supported"),
        InstKind::EhTypeId { .. } => panic!("glcc native backend: EH is not yet supported"),
        InstKind::Fcmp { .. } | InstKind::FAdd(..) | InstKind::FSub(..) | InstKind::FMul(..) | InstKind::FDiv(..) | InstKind::FNeg(_)
        | InstKind::Ceil(_) | InstKind::Floor(_) | InstKind::Sqrt(_) | InstKind::SItoF(_) | InstKind::UItoF(_) | InstKind::SFtoI(_)
        | InstKind::UFtoI(_) | InstKind::Fptrunc(_) | InstKind::Fpext(_) | InstKind::Splat(_) | InstKind::ExtractLane { .. }
        | InstKind::ExtractValue { .. } | InstKind::InsertValue { .. } | InstKind::Await { .. } => {
            panic!("glcc native backend: floating point/vectors/aggregates are not yet supported")
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
                    let vr = sel.value_vreg(mir, v);
                    sel.emit(MInst::new(OpId::RET, Width::W64, smallvec::smallvec![Operand::Reg(Reg::V(vr))]));
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
    if let Some(res) = result {
        if !matches!(sel.isa.abi().ret_loc(&sig.ret), RetLoc::Void) {
            ops.push(Operand::FixedReg(Reg::V(sel.value_vreg(mir, res)), sel.isa.abi().ret_reg()));
        }
    }
    match target {
        CallTarget::Direct(sym) => {
            ops.push(Operand::Sym(SymOperand { target: sym, reloc: crate::machine::RelocKind::Plt32, addend: 0 }));
        }
        CallTarget::Indirect(v) => {
            ops.push(Operand::Reg(Reg::V(sel.value_vreg(mir, v))));
        }
    }
    if sig.variadic {
        // SysV: %al holds the number of vector registers used (zero
        // while floats are unsupported). The zero is pinned in %rax
        // for the call; a fixed early use under a fixed late def of
        // the same register is the combination regalloc2 supports.
        let zero = sel.alloc_vreg();
        sel.emit(MInst::new(
            sel.isa.const_move_op(0),
            Width::W32,
            smallvec::smallvec![Operand::Reg(Reg::V(zero)), Operand::Imm(0)],
        ));
        ops.push(Operand::FixedReg(Reg::V(zero), PReg::gpr(regs::RAX)));
    }
    for (i, arg) in args.iter().enumerate() {
        let ty = sig.params.get(i).unwrap_or(&mir.values[*arg].ty);
        let preg = match sel.isa.abi().param_loc(i, ty) {
            ParamLoc::Reg(p) => p,
            ParamLoc::Stack => panic!("glcc native backend: stack call arguments are not yet supported"),
        };
        ops.push(Operand::FixedReg(Reg::V(sel.value_vreg(mir, *arg)), preg));
    }
    let op = match target {
        CallTarget::Direct(_) => Op::Call,
        CallTarget::Indirect(_) => Op::CallRm,
    };
    sel.emit(MInst::new(op_id(op), Width::W64, ops));
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
    // Materialize the false value first, then overwrite when taken.
    sel.emit(MInst::new(
        OpId::MOVE,
        result_width(sel, mir, inst),
        smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(else_vr))],
    ));
    let cc = match cond_icc(sel, mir, cond) {
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
    };
    sel.emit(MInst::new(
        op_id(Op::Cmov),
        result_width(sel, mir, inst),
        smallvec::smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(then_vr)), Operand::Imm(cc as i64)],
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
/// length (and memset byte) are constants here; their moves go in
/// reverse slot order so an argument already living in a later
/// argument register is consumed before that register is overwritten.
fn mem_intrinsic(sel: &mut FuncSel, mir: &Function, inst: Inst, name: &'static str, args: &[Value], extra: Option<(i64, i64)>) {
    let arg_regs = sel.isa.abi().arg_regs();
    let mut ops: smallvec::SmallVec<[Operand; 4]> = smallvec::SmallVec::new();
    if let Some(res) = mir.insts[inst].result {
        ops.push(Operand::FixedReg(Reg::V(sel.value_vreg(mir, res)), sel.isa.abi().ret_reg()));
    }
    ops.push(Operand::Sym(SymOperand { target: SymTarget::Named(name), reloc: crate::machine::RelocKind::Plt32, addend: 0 }));

    // (slot, source) moves, emitted last-first.
    let mut moves: Vec<(usize, Operand)> = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        moves.push((i, Operand::Reg(Reg::V(sel.value_vreg(mir, *arg)))));
    }
    if let Some((len, byte)) = extra {
        let mut imm_move = |sel: &mut FuncSel, slot: usize, v: i64, width: Width| {
            let vr = sel.alloc_vreg();
            sel.emit(MInst::new(sel.isa.const_move_op(v), width, smallvec::smallvec![Operand::Reg(Reg::V(vr)), Operand::Imm(v)]));
            moves.push((slot, Operand::Reg(Reg::V(vr))));
        };
        if name == "memset" {
            imm_move(sel, 1, byte, Width::W32);
        }
        imm_move(sel, 2, len, Width::W64);
    }
    for (slot, src) in moves.iter().rev() {
        sel.emit(MInst::new(
            OpId::MOVE,
            Width::W64,
            smallvec::smallvec![Operand::Reg(Reg::P(arg_regs[*slot])), src.clone()],
        ));
    }
    sel.emit(MInst::new(op_id(Op::Call), Width::W64, ops));
}
