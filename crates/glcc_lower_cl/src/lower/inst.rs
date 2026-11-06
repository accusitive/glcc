use std::collections::HashMap;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::immediates::Offset32;
use cranelift_codegen::ir::types::{self};
use cranelift_codegen::ir::{
    AtomicRmwOp, BlockArg, InstBuilder, MemFlagsData, SourceLoc, StackSlotData,
    StackSlotKind, TrapCode,
};
use cranelift_codegen::ir::{Block as ClifBlock, FuncRef, Value as ClifValue};
use cranelift_frontend::{FunctionBuilder, Switch};
use cranelift_module::{FuncId, Linkage, Module};

use glcc_mir::{self as mir};

use crate::abi::{self, Arch, SigLayout};
use crate::memflags;
use crate::ty::clif_ty;
use super::*;

/// Byte offset and element type of one aggregate position: struct field
/// or array element by index.
pub(crate) fn field_offset(ty: &mir::Ty, index: u32) -> (u64, mir::Ty) {
    match ty {
        mir::Ty::Struct { fields, .. } => {
            let offsets = mir::struct_field_offsets(ty);
            let slot = index as usize;
            if slot >= fields.len() {
                panic!("extract step out of range");
            }
            (offsets[slot], fields[slot].clone())
        }
        mir::Ty::Array { elem, .. } => (u64::from(index) * elem.size_bytes(), elem.as_ref().clone()),
        mir::Ty::WideInt { .. } => {
            if index >= ty.limb_count() {
                panic!("limb index out of range");
            }
            (u64::from(index) * u64::from(mir::ty::LIMB_BITS / 8), ty.limb_ty(index))
        }
        _ => panic!("extract step into scalar"),
    }
}

/// Whether a switch table is dense enough that filling its holes with
/// the default target is cheaper than letting the switch helper split
/// it into one table per contiguous run.
pub(crate) fn dense_switch_span(base: i128, max: i128, count: usize) -> bool {
    let span = (max - base) as u128 + 1;
    span <= 4096 && span <= (count as u128).saturating_mul(2) + 8
}


impl<'m> Body<'m> {
/// Lower one inline-asm block. On x86-64, blocks the fork's native
/// `inline_asm` instruction can express ([`crate::asm_native::plan`])
/// lower in place; everything else — and every block on aarch64 and
/// riscv64, where the fork has no inline asm — uses the out-of-line
/// wrapper (see [`crate::asm`]). An `asm goto` block's result is the
/// label selector (0 = fallthrough), which the MIR terminator
/// dispatches on.
pub(crate) fn emit_inline_asm<M: Module>(&mut self, builder: &mut FunctionBuilder, clif: &mut M, block: &mir::AsmBlock, wrappers: &mut Wrappers) -> Option<ClifValue> {
    if self.target.arch == Arch::X86_64 {
        match crate::asm_native::plan(block, self.target.arch, self.func) {
            Ok(native) => return self.emit_native_inline_asm(builder, block, native),
            Err(reason) => wrappers.note_fallback(&reason),
        }
    }
    let site = crate::asm::build(block, self.func, self.target);
    let is_goto = !block.labels.is_empty();
    let sig = crate::asm::wrapper_sig(self.target, is_goto);
    let wrapper = clif
        .declare_function(&site.name, Linkage::Import, &sig)
        .unwrap_or_else(|_| panic!("cannot declare {}", site.name));
    wrappers.add(&site.name, &site.text);
    let wrapper = clif.declare_func_in_func(wrapper, builder.func);
    let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, site.area_size as u32, 4));
    let area = builder.ins().stack_addr(types::I64, slot, 0);
    let area_flags = memflags::trusted();
    let user_flags = memflags::notrap();
    let outputs = block.outputs as usize;
    for (i, op) in block.operands.iter().enumerate() {
        if i < outputs {
            let Some(slot) = site.slots[i] else { continue };
            if slot.addr {
                // A memory output's address crosses in the slot.
                let addr = self.addr_operand(builder, op.value);
                builder.ins().store(area_flags, addr, area, Offset32::new(slot.offset as i32));
            } else if op.constraint.contains('+') {
                // `+` outputs start from the lvalue's current value.
                let ty = clif_ty(&op.ty).expect("asm register operand needs a scalar type");
                let addr = self.addr_operand(builder, op.value);
                let value = builder.ins().load(ty, user_flags, addr, Offset32::new(0));
                builder.ins().store(area_flags, value, area, Offset32::new(slot.offset as i32));
            }
            continue;
        }
        if site.slots[i].is_none() && op.tie.is_none() {
            continue;
        }
        match op.tie {
            Some(tie) => {
                let target = tie as usize;
                if site.slots[target].is_some_and(|slot| slot.addr) {
                    // Tied to a memory output: the input value must be in
                    // the output's location before the template runs.
                    // Aggregates are already copied there by MIR lowering;
                    // scalars store here.
                    if clif_ty(&op.ty).is_some() {
                        let value = self.operand(builder, op.value);
                        let addr = self.operand(builder, block.operands[target].value);
                        builder.ins().store(user_flags, value, addr, Offset32::new(0));
                    }
                } else {
                    let value = self.operand(builder, op.value);
                    let slot = site.slots[target].expect("tied output has a slot");
                    builder.ins().store(area_flags, value, area, Offset32::new(slot.offset as i32));
                }
            }
            None => {
                let value = self.operand(builder, op.value);
                let slot = site.slots[i].expect("asm input has a slot");
                builder.ins().store(area_flags, value, area, Offset32::new(slot.offset as i32));
            }
        }
    }
    let call = builder.ins().call(wrapper, &[area]);
    let selector = is_goto.then(|| builder.inst_results(call)[0]);
    for (i, op) in block.operands.iter().enumerate().take(outputs) {
        let Some(slot) = site.slots[i] else { continue };
        if slot.addr {
            continue;
        }
        let ty = clif_ty(&op.ty).expect("asm register operand needs a scalar type");
        let value = builder.ins().load(ty, area_flags, area, Offset32::new(slot.offset as i32));
        let addr = self.operand(builder, op.value);
        builder.ins().store(user_flags, value, addr, Offset32::new(0));
    }
    selector
}

/// Lower an inline-asm block with the fork's native `inline_asm`
/// instruction: inputs become instruction arguments (widened to the
/// full 64-bit registers the fork passes operands in), outputs become
/// the instruction's `i64` results, reduced back to the C type and
/// stored through the output lvalue's address.
pub(crate) fn emit_native_inline_asm(
    &mut self,
    builder: &mut FunctionBuilder,
    block: &mir::AsmBlock,
    native: crate::asm_native::NativeAsm,
) -> Option<ClifValue> {
    let mut contents = cranelift_codegen::ir::InlineAsmContents::new(native.template, true);
    for constraint in &native.constraints {
        contents = contents.operand(constraint.clone());
    }
    for clobber in &native.clobbers {
        contents = contents.clobber(clobber.clone());
    }
    let asm_ref = builder.create_inline_asm(contents);

    let outputs = block.outputs as usize;
    let mut args: Vec<ClifValue> = Vec::new();
    // The fork consumes input values in GCC order: the read sides of
    // `+` outputs first, then the plain inputs.
    for op in &block.operands[..outputs] {
        if !op.constraint.contains('+') {
            continue;
        }
        let addr = self.addr_operand(builder, op.value);
        let value = load_operand(builder, &op.ty, addr);
        args.push(widen_operand(builder, &op.ty, value));
    }
    for (op, kind) in block.operands[outputs..]
        .iter()
        .zip(&native.operand_kinds[outputs..])
    {
        match *kind {
            crate::asm_native::NativeKind::Reg => {
                let value = self.operand(builder, op.value);
                args.push(widen_operand(builder, &op.ty, value));
            }
            crate::asm_native::NativeKind::Vec => {
                // The register class rides the value's type: the vector
                // value must reach the instruction un-widened.
                let value = self.operand(builder, op.value);
                args.push(value);
            }
            crate::asm_native::NativeKind::Mem => {
                // The fork's "m" semantics: the template sees the memory
                // at the address held in the input value.
                args.push(self.addr_operand(builder, op.value));
            }
            crate::asm_native::NativeKind::Imm => {
                let v = crate::asm::const_int(self.func, op.value);
                let imm = i64::try_from(v)
                    .unwrap_or_else(|_| panic!("asm immediate {v} does not fit in i64"));
                args.push(builder.ins().iconst(types::I64, imm));
            }
        }
    }
    let inst = builder.ins().inline_asm(asm_ref, &args);
    let results: Vec<ClifValue> = builder.inst_results(inst).to_vec();
    for (i, op) in block.operands[..outputs].iter().enumerate() {
        let addr = self.addr_operand(builder, op.value);
        let value = match native.operand_kinds[i] {
            // Vector results come back as the fork's canonical `I8X16`;
            // the store keeps the bits and the C vector type reads them.
            crate::asm_native::NativeKind::Vec => results[i],
            _ => {
                let ty = clif_ty(&op.ty).expect("asm register operand needs a scalar type");
                if ty == types::I64 {
                    results[i]
                } else {
                    builder.ins().ireduce(ty, results[i])
                }
            }
        };
        builder.ins().store(memflags::notrap(), value, addr, Offset32::new(0));
    }
    None
}

pub(crate) fn emit_inst<M: Module>(
    &mut self,
    builder: &mut FunctionBuilder,
    clif: &mut M,
    func_ids: &HashMap<mir::Func, FuncId>,
    extern_ids: &HashMap<mir::Extern, FuncId>,
    data_ids: &HashMap<mir::Global, cranelift_module::DataId>,
    tls_data: &std::collections::HashSet<mir::Global>,
    rt: &Rt,
    layouts: &HashMap<mir::Func, SigLayout>,
    extern_layouts: &HashMap<mir::Extern, SigLayout>,
    inst: mir::Inst,
    wrappers: &mut Wrappers,
) {
    let data = self.func.insts[inst].clone();
    if let Some(debug) = self.debug.as_deref_mut() {
        builder.set_srcloc(SourceLoc::new(debug.intern_location(data.loc)));
    }
    let result_ty = data.result.map(|v| self.func.values[v].ty.clone());
    let lowered: Option<ClifValue> = match data.kind {
        mir::InstKind::Add(a, b) => {
            let x = self.operand(builder, a);
            let y = self.operand(builder, b);
            Some(builder.ins().iadd(x, y))
        }
        mir::InstKind::Sub(a, b) => {
            let x = self.operand(builder, a);
            let y = self.operand(builder, b);
            Some(builder.ins().isub(x, y))
        }
        mir::InstKind::Mul(a, b) => {
            let x = self.operand(builder, a);
            let y = self.operand(builder, b);
            Some(builder.ins().imul(x, y))
        }
        mir::InstKind::Udiv(a, b) => {
            let x = self.operand(builder, a);
            let y = self.operand(builder, b);
            Some(builder.ins().udiv(x, y))
        }
        mir::InstKind::Sdiv(a, b) => {
            let x = self.operand(builder, a);
            let y = self.operand(builder, b);
            if self.target.arch == Arch::X86_64 {
                // C's `/` must fault with SIGFPE on a zero divisor, but
                // Cranelift's `sdiv` traps with `ud2` (SIGILL) to keep
                // the hardware fault reserved for overflow. Dividing the
                // absolute values keeps the fault in hardware; the sign
                // is restored afterwards. `INT_MIN / -1` wraps here
                // instead of faulting, but that quotient is undefined
                // behavior anyway.
                let ty = builder.func.dfg.value_type(x);
                let sign = builder.ins().iconst(ty, i64::from(ty.bits()) - 1);
                let sa = builder.ins().sshr(x, sign);
                let sb = builder.ins().sshr(y, sign);
                let xa = builder.ins().bxor(x, sa);
                let ax = builder.ins().isub(xa, sa);
                let yb = builder.ins().bxor(y, sb);
                let ay = builder.ins().isub(yb, sb);
                let q = builder.ins().udiv(ax, ay);
                let s = builder.ins().bxor(sa, sb);
                let qs = builder.ins().bxor(q, s);
                Some(builder.ins().isub(qs, s))
            } else {
                Some(builder.ins().sdiv(x, y))
            }
        }
        mir::InstKind::Urem(a, b) => {
            let x = self.operand(builder, a);
            let y = self.operand(builder, b);
            Some(builder.ins().urem(x, y))
        }
        mir::InstKind::Srem(a, b) => {
            let x = self.operand(builder, a);
            let y = self.operand(builder, b);
            Some(builder.ins().srem(x, y))
        }
        mir::InstKind::And(a, b) => {
            let x = self.operand(builder, a);
            let y = self.operand(builder, b);
            Some(builder.ins().band(x, y))
        }
        mir::InstKind::Or(a, b) => {
            let x = self.operand(builder, a);
            let y = self.operand(builder, b);
            Some(builder.ins().bor(x, y))
        }
        mir::InstKind::Xor(a, b) => {
            let x = self.operand(builder, a);
            let y = self.operand(builder, b);
            Some(builder.ins().bxor(x, y))
        }
        mir::InstKind::Shl(a, b) => {
            let x = self.operand(builder, a);
            let y = self.operand(builder, b);
            Some(builder.ins().ishl(x, y))
        }
        mir::InstKind::LShr(a, b) => {
            let x = self.operand(builder, a);
            let y = self.operand(builder, b);
            Some(builder.ins().ushr(x, y))
        }
        mir::InstKind::AShr(a, b) => {
            let x = self.operand(builder, a);
            let y = self.operand(builder, b);
            Some(builder.ins().sshr(x, y))
        }
        mir::InstKind::Neg(a) => {
            let x = self.operand(builder, a);
            Some(builder.ins().ineg(x))
        }
        mir::InstKind::Not(a) => {
            let x = self.operand(builder, a);
            Some(builder.ins().bnot(x))
        }
        mir::InstKind::FAdd(a, b) => {
            if self.is_f80(a) {
                let f80 = rt.f80.as_ref().expect("f80 arithmetic needs x86-64");
                Some(self.f80_binop(builder, clif, f80.add, a, b))
            } else if self.is_f128(a) && self.target.arch != Arch::X86_64 {
                // Quad arithmetic through the toolchain trampolines,
                // which share the out-pointer convention.
                let quad = rt.quad.as_ref().expect("quad arithmetic needs trampolines");
                Some(self.f80_binop(builder, clif, quad.add, a, b))
            } else if self.is_f128(a) {
                // x86-64 `__float128` arithmetic through the vendored runtime (see
                // `QuadDirectRt`): only this case reaches here, since
                // quad targets route through trampolines above.
                let quad = rt.quad_direct.as_ref().expect("quad arithmetic needs the soft-float runtime");
                Some(self.f128_direct_binop(builder, clif, quad.add, a, b))
            } else {
                let x = self.operand(builder, a);
                let y = self.operand(builder, b);
                Some(builder.ins().fadd(x, y))
            }
        }
        mir::InstKind::FSub(a, b) => {
            if self.is_f80(a) {
                let f80 = rt.f80.as_ref().expect("f80 arithmetic needs x86-64");
                Some(self.f80_binop(builder, clif, f80.sub, a, b))
            } else if self.is_f128(a) && self.target.arch != Arch::X86_64 {
                // Quad arithmetic through the toolchain trampolines,
                // which share the out-pointer convention.
                let quad = rt.quad.as_ref().expect("quad arithmetic needs trampolines");
                Some(self.f80_binop(builder, clif, quad.sub, a, b))
            } else if self.is_f128(a) {
                // x86-64 `__float128` arithmetic through the vendored runtime (see
                // `QuadDirectRt`): only this case reaches here, since
                // quad targets route through trampolines above.
                let quad = rt.quad_direct.as_ref().expect("quad arithmetic needs the soft-float runtime");
                Some(self.f128_direct_binop(builder, clif, quad.sub, a, b))
            } else {
                let x = self.operand(builder, a);
                let y = self.operand(builder, b);
                Some(builder.ins().fsub(x, y))
            }
        }
        mir::InstKind::FMul(a, b) => {
            if self.is_f80(a) {
                let f80 = rt.f80.as_ref().expect("f80 arithmetic needs x86-64");
                Some(self.f80_binop(builder, clif, f80.mul, a, b))
            } else if self.is_f128(a) && self.target.arch != Arch::X86_64 {
                // Quad arithmetic through the toolchain trampolines,
                // which share the out-pointer convention.
                let quad = rt.quad.as_ref().expect("quad arithmetic needs trampolines");
                Some(self.f80_binop(builder, clif, quad.mul, a, b))
            } else if self.is_f128(a) {
                // x86-64 `__float128` arithmetic through the vendored runtime (see
                // `QuadDirectRt`): only this case reaches here, since
                // quad targets route through trampolines above.
                let quad = rt.quad_direct.as_ref().expect("quad arithmetic needs the soft-float runtime");
                Some(self.f128_direct_binop(builder, clif, quad.mul, a, b))
            } else {
                let x = self.operand(builder, a);
                let y = self.operand(builder, b);
                Some(builder.ins().fmul(x, y))
            }
        }
        mir::InstKind::FDiv(a, b) => {
            if self.is_f80(a) {
                let f80 = rt.f80.as_ref().expect("f80 arithmetic needs x86-64");
                Some(self.f80_binop(builder, clif, f80.div, a, b))
            } else if self.is_f128(a) && self.target.arch != Arch::X86_64 {
                // Quad arithmetic through the toolchain trampolines,
                // which share the out-pointer convention.
                let quad = rt.quad.as_ref().expect("quad arithmetic needs trampolines");
                Some(self.f80_binop(builder, clif, quad.div, a, b))
            } else if self.is_f128(a) {
                // x86-64 `__float128` arithmetic through the vendored runtime (see
                // `QuadDirectRt`): only this case reaches here, since
                // quad targets route through trampolines above.
                let quad = rt.quad_direct.as_ref().expect("quad arithmetic needs the soft-float runtime");
                Some(self.f128_direct_binop(builder, clif, quad.div, a, b))
            } else {
                let x = self.operand(builder, a);
                let y = self.operand(builder, b);
                Some(builder.ins().fdiv(x, y))
            }
        }
        mir::InstKind::FNeg(a) => {
            if self.is_f80(a) {
                // Through the helper's fchs: an F128 fneg would flip
                // the quad sign at bit 127 instead of bit 79.
                let f80 = rt.f80.as_ref().expect("f80 negation needs x86-64");
                let a_addr = self.f80_addr(builder, a);
                let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
                let out = builder.ins().stack_addr(types::I64, slot, 0);
                let callee = clif.declare_func_in_func(f80.neg, builder.func);
                builder.ins().call(callee, &[out, a_addr]);
                Some(f80_result(builder, out))
            } else if self.is_f128(a) {
                // Bit-127 flip through integer ops: no Clif fneg on
                // F128 anywhere (see `quad.rs`).
                let x = self.operand(builder, a);
                Some(crate::quad::quad_neg(builder, x))
            } else {
                let x = self.operand(builder, a);
                Some(builder.ins().fneg(x))
            }
        }
        mir::InstKind::Ceil(a) => {
            if self.is_f80(a) {
                // Through frndint with the control pinned round-up: an
                // F128 ceil would read quad fields, not x87 ones.
                let f80 = rt.f80.as_ref().expect("f80 ceil needs x86-64");
                let a_addr = self.f80_addr(builder, a);
                let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
                let out = builder.ins().stack_addr(types::I64, slot, 0);
                let callee = clif.declare_func_in_func(f80.ceil, builder.func);
                builder.ins().call(callee, &[out, a_addr]);
                Some(f80_result(builder, out))
            } else if self.is_f128(a) {
                // Truncate-and-adjust through integer ops (see
                // `quad.rs`): no Clif ceil on F128 anywhere.
                let x = self.operand(builder, a);
                Some(crate::quad::quad_ceil(builder, x))
            } else {
                let x = self.operand(builder, a);
                Some(builder.ins().ceil(x))
            }
        }
        mir::InstKind::Floor(a) => {
            if self.is_f80(a) {
                // Through frndint pinned round-down (see above).
                let f80 = rt.f80.as_ref().expect("f80 floor needs x86-64");
                let a_addr = self.f80_addr(builder, a);
                let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
                let out = builder.ins().stack_addr(types::I64, slot, 0);
                let callee = clif.declare_func_in_func(f80.floor, builder.func);
                builder.ins().call(callee, &[out, a_addr]);
                Some(f80_result(builder, out))
            } else if self.is_f128(a) {
                // Integer truncate-and-adjust (see `quad.rs`).
                let x = self.operand(builder, a);
                Some(crate::quad::quad_floor(builder, x))
            } else {
                let x = self.operand(builder, a);
                Some(builder.ins().floor(x))
            }
        }
        mir::InstKind::Sqrt(a) => {
            if self.is_f80(a) {
                // Through fsqrt: an F128 sqrt would read quad fields,
                // not x87 ones.
                let f80 = rt.f80.as_ref().expect("f80 sqrt needs x86-64");
                let a_addr = self.f80_addr(builder, a);
                let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
                let out = builder.ins().stack_addr(types::I64, slot, 0);
                let callee = clif.declare_func_in_func(f80.sqrt, builder.func);
                builder.ins().call(callee, &[out, a_addr]);
                Some(f80_result(builder, out))
            } else if self.is_f128(a) {
                if self.target.arch == Arch::X86_64 {
                    todo!("f128 sqrt lowering on x86-64");
                }
                // Through the toolchain `sqrtl` trampoline, like the
                // legacy backend's complex magnitudes.
                let quad = rt.quad.as_ref().expect("quad sqrt needs trampolines");
                let a_addr = self.f80_addr(builder, a);
                let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
                let out = builder.ins().stack_addr(types::I64, slot, 0);
                let callee = clif.declare_func_in_func(quad.sqrt, builder.func);
                builder.ins().call(callee, &[out, a_addr]);
                Some(f80_result(builder, out))
            } else {
                let x = self.operand(builder, a);
                Some(builder.ins().sqrt(x))
            }
        }
        mir::InstKind::Clz(a) => {
            let x = self.operand(builder, a);
            Some(builder.ins().clz(x))
        }
        mir::InstKind::Ctz(a) => {
            let x = self.operand(builder, a);
            Some(builder.ins().ctz(x))
        }
        mir::InstKind::Popcnt(a) => {
            let x = self.operand(builder, a);
            Some(builder.ins().popcnt(x))
        }
        mir::InstKind::EhTypeId { typeinfo } => {
            let mut found = None;
            let mut index = 0u64;
            for block in self.func.blocks.iter() {
                let Some(clauses) = &block.lpad else { continue };
                for clause in clauses {
                    index += 1;
                    if let mir::LpadClause::Catch(g) = clause
                        && *g == typeinfo {
                            found = Some(index);
                        }
                }
            }
            let selector = found.unwrap_or_else(|| panic!("eh.typeid for a typeinfo this function never catches"));
            Some(builder.ins().iconst(types::I32, selector as i64))
        }
        mir::InstKind::Bswap(a) => {
            let x = self.operand(builder, a);
            Some(builder.ins().bswap(x))
        }
        mir::InstKind::Icmp { pred, lhs, rhs } => {
            let x = self.operand(builder, lhs);
            let y = self.operand(builder, rhs);
            let cc = match pred {
                mir::IcmpPred::Eq => IntCC::Equal,
                mir::IcmpPred::Ne => IntCC::NotEqual,
                mir::IcmpPred::Ult => IntCC::UnsignedLessThan,
                mir::IcmpPred::Ule => IntCC::UnsignedLessThanOrEqual,
                mir::IcmpPred::Ugt => IntCC::UnsignedGreaterThan,
                mir::IcmpPred::Uge => IntCC::UnsignedGreaterThanOrEqual,
                mir::IcmpPred::Slt => IntCC::SignedLessThan,
                mir::IcmpPred::Sle => IntCC::SignedLessThanOrEqual,
                mir::IcmpPred::Sgt => IntCC::SignedGreaterThan,
                mir::IcmpPred::Sge => IntCC::SignedGreaterThanOrEqual,
            };
            Some(builder.ins().icmp(cc, x, y))
        }
        mir::InstKind::Fcmp { pred, lhs, rhs } => {
            if self.is_f80(lhs) {
                // Ordered comparisons through the fucomip helpers: an
                // F128 fcmp would read quad fields, not x87 ones.
                // Unordered/ordered reuse the `ne` helper on each side
                // against itself (`ne` is true exactly for NaN, so no
                // new asm or x87 bit-decoding is needed).
                if matches!(pred, mir::FcmpPred::Uno | mir::FcmpPred::Ord) {
                    let f80 = rt.f80.as_ref().expect("f80 comparison needs x86-64");
                    let ne = f80.ne;
                    let a_nan = self.f80_cmp(builder, clif, ne, lhs, lhs);
                    let b_nan = self.f80_cmp(builder, clif, ne, rhs, rhs);
                    let unordered = builder.ins().bor(a_nan, b_nan);
                    if matches!(pred, mir::FcmpPred::Uno) {
                        Some(unordered)
                    } else {
                        let one = builder.ins().iconst(types::I8, 1);
                        Some(builder.ins().bxor(unordered, one))
                    }
                } else {
                    let f80 = rt.f80.as_ref().expect("f80 comparison needs x86-64");
                    let helper = match pred {
                        mir::FcmpPred::Eq => f80.eq,
                        mir::FcmpPred::Ne => f80.ne,
                        mir::FcmpPred::Lt => f80.lt,
                        mir::FcmpPred::Le => f80.le,
                        mir::FcmpPred::Gt => f80.gt,
                        mir::FcmpPred::Ge => f80.ge,
                        mir::FcmpPred::Uno | mir::FcmpPred::Ord => unreachable!(),
                    };
                    Some(self.f80_cmp(builder, clif, helper, lhs, rhs))
                }
            } else if self.is_f128(lhs) {
                // Ordered comparisons as integer sequences over the
                // halves (see `quad.rs`): no Clif fcmp on F128
                // anywhere, on any target.
                let qpred = match pred {
                    mir::FcmpPred::Eq => crate::quad::QuadCmp::Eq,
                    mir::FcmpPred::Ne => crate::quad::QuadCmp::Ne,
                    mir::FcmpPred::Lt => crate::quad::QuadCmp::Lt,
                    mir::FcmpPred::Le => crate::quad::QuadCmp::Le,
                    mir::FcmpPred::Gt => crate::quad::QuadCmp::Gt,
                    mir::FcmpPred::Ge => crate::quad::QuadCmp::Ge,
                    mir::FcmpPred::Uno => crate::quad::QuadCmp::Uno,
                    mir::FcmpPred::Ord => crate::quad::QuadCmp::Ord,
                };
                let x = self.operand(builder, lhs);
                let y = self.operand(builder, rhs);
                let xa = crate::quad::quad_spill(builder, x);
                let (a_lo, a_hi) = crate::quad::quad_load_halves(builder, xa);
                let xb = crate::quad::quad_spill(builder, y);
                let (b_lo, b_hi) = crate::quad::quad_load_halves(builder, xb);
                let r = crate::quad::quad_cmp(builder, qpred, a_lo, a_hi, b_lo, b_hi);
                Some(builder.ins().ireduce(types::I8, r))
            } else {
                let x = self.operand(builder, lhs);
                let y = self.operand(builder, rhs);
                let cc = match pred {
                    mir::FcmpPred::Eq => FloatCC::Equal,
                    mir::FcmpPred::Ne => FloatCC::NotEqual,
                    mir::FcmpPred::Lt => FloatCC::LessThan,
                    mir::FcmpPred::Le => FloatCC::LessThanOrEqual,
                    mir::FcmpPred::Gt => FloatCC::GreaterThan,
                    mir::FcmpPred::Ge => FloatCC::GreaterThanOrEqual,
                    mir::FcmpPred::Uno => FloatCC::Unordered,
                    mir::FcmpPred::Ord => FloatCC::Ordered,
                };
                Some(builder.ins().fcmp(cc, x, y))
            }
        }
        mir::InstKind::Trunc(a) => {
            let dst = result_clif(&result_ty);
            let lowered = self.operand(builder, a);
            let src_bits = mir_int_bits(&self.func.values[a].ty);
            let dst_bits = mir_int_bits(result_ty.as_ref().expect("trunc needs a result type"));
            Some(convert_int(builder, lowered, dst, src_bits, dst_bits, IntConvert::Trunc))
        }
        mir::InstKind::Zext(a) => {
            let dst = result_clif(&result_ty);
            let lowered = self.operand(builder, a);
            let src_bits = mir_int_bits(&self.func.values[a].ty);
            let dst_bits = mir_int_bits(result_ty.as_ref().expect("zext needs a result type"));
            Some(convert_int(builder, lowered, dst, src_bits, dst_bits, IntConvert::Zext))
        }
        mir::InstKind::Sext(a) => {
            let dst = result_clif(&result_ty);
            let lowered = self.operand(builder, a);
            let src_bits = mir_int_bits(&self.func.values[a].ty);
            let dst_bits = mir_int_bits(result_ty.as_ref().expect("sext needs a result type"));
            Some(convert_int(builder, lowered, dst, src_bits, dst_bits, IntConvert::Sext))
        }
        mir::InstKind::Bitcast(a) => {
            let dst = result_clif(&result_ty);
            let lowered = self.operand(builder, a);
            let from = builder.func.dfg.value_type(lowered);
            if from.bytes() == dst.bytes() {
                // A lane-count change needs an explicit byte order;
                // the bit puns are little-endian images.
                let flags = if from.lane_count() != dst.lane_count() {
                    MemFlagsData::new().with_endianness(cranelift_codegen::ir::Endianness::Little)
                } else {
                    MemFlagsData::new()
                };
                Some(builder.ins().bitcast(dst, flags, lowered))
            } else if from.bytes() < dst.bytes() {
                // A narrow pointer widens back to the machine address
                // width, sign-extended like MSVC's default for `__ptr32`.
                Some(builder.ins().sextend(dst, lowered))
            } else {
                Some(builder.ins().ireduce(dst, lowered))
            }
        }
        mir::InstKind::SItoF(a) => {
            let dst_ty = result_ty.clone().expect("sitof needs a result type");
            if matches!(dst_ty, mir::Ty::Float(mir::FloatKind::F80)) {
                // Exact widening through fild: sub-word sources widen
                // value-preservingly to the helper width first.
                let f80 = rt.f80.as_ref().expect("f80 conversion needs x86-64");
                let src_ty = self.func.values[a].ty.clone();
                let bits = mir_int_bits(&src_ty);
                if bits > 64 {
                    // Through the vendored runtime (`__glcc_floattixf`),
                    // captured like a foreign return.
                    let x = self.operand(builder, a);
                    let (lo, hi) = builder.ins().isplit(x);
                    let callee = clif.declare_func_in_func(f80.from_i128, builder.func);
                    builder.ins().call(callee, &[lo, hi]);
                    Some(foreign_f80_result(builder, clif, rt))
                } else {
                    let x = self.operand(builder, a);
                    let src_clif = clif_ty(&src_ty).expect("int source has a Clif type");
                    let (helper, wide) = if bits <= 32 {
                        let w = if src_clif.bytes() < 4 { builder.ins().sextend(types::I32, x) } else { x };
                        (f80.from_i32, w)
                    } else {
                        (f80.from_i64, x)
                    };
                    let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
                    let out = builder.ins().stack_addr(types::I64, slot, 0);
                    let callee = clif.declare_func_in_func(helper, builder.func);
                    builder.ins().call(callee, &[out, wide]);
                    Some(f80_result(builder, out))
                }
            } else if matches!(dst_ty, mir::Ty::Float(mir::FloatKind::F128)) {
                // Integer ops (see `quad.rs`): sub-word sources widen
                // value-preservingly to 64 bits first, 128-bit sources
                // take the rounding path. Works on every target,
                // including x86-64 `__float128`.
                let src_ty = self.func.values[a].ty.clone();
                let bits = mir_int_bits(&src_ty);
                let x = self.operand(builder, a);
                if bits > 64 {
                    let (lo, hi) = crate::quad::quad_from_int128(builder, x, false);
                    Some(crate::quad::quad_store_carrier(builder, lo, hi))
                } else {
                    let src_clif = clif_ty(&src_ty).expect("int source has a Clif type");
                    let w32 = if src_clif.bytes() < 4 { builder.ins().sextend(types::I32, x) } else { x };
                    let wide = if bits <= 32 { builder.ins().sextend(types::I64, w32) } else { w32 };
                    let (lo, hi) = crate::quad::quad_from_int(builder, wide, false);
                    Some(crate::quad::quad_store_carrier(builder, lo, hi))
                }
            } else {
                let dst = result_clif(&result_ty);
                let lowered = self.operand(builder, a);
                Some(builder.ins().fcvt_from_sint(dst, lowered))
            }
        }
        mir::InstKind::UItoF(a) => {
            let dst_ty = result_ty.clone().expect("uitof needs a result type");
            if matches!(dst_ty, mir::Ty::Float(mir::FloatKind::F80)) {
                // Exact like above; unsigned sources zero-extend.
                let f80 = rt.f80.as_ref().expect("f80 conversion needs x86-64");
                let src_ty = self.func.values[a].ty.clone();
                let bits = mir_int_bits(&src_ty);
                if bits > 64 {
                    // Through the vendored runtime (`__glcc_floatuntixf`),
                    // captured like a foreign return.
                    let x = self.operand(builder, a);
                    let (lo, hi) = builder.ins().isplit(x);
                    let callee = clif.declare_func_in_func(f80.from_u128, builder.func);
                    builder.ins().call(callee, &[lo, hi]);
                    Some(foreign_f80_result(builder, clif, rt))
                } else {
                    let x = self.operand(builder, a);
                    let src_clif = clif_ty(&src_ty).expect("int source has a Clif type");
                    let (helper, wide) = if bits <= 32 {
                        let w = if src_clif.bytes() < 4 { builder.ins().uextend(types::I32, x) } else { x };
                        (f80.from_u32, w)
                    } else {
                        (f80.from_u64, x)
                    };
                    let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
                    let out = builder.ins().stack_addr(types::I64, slot, 0);
                    let callee = clif.declare_func_in_func(helper, builder.func);
                    builder.ins().call(callee, &[out, wide]);
                    Some(f80_result(builder, out))
                }
            } else if matches!(dst_ty, mir::Ty::Float(mir::FloatKind::F128)) {
                // Integer ops (see `quad.rs`): unsigned sources
                // zero-extend, 128-bit sources take the rounding path.
                let src_ty = self.func.values[a].ty.clone();
                let bits = mir_int_bits(&src_ty);
                let x = self.operand(builder, a);
                if bits > 64 {
                    let (lo, hi) = crate::quad::quad_from_int128(builder, x, true);
                    Some(crate::quad::quad_store_carrier(builder, lo, hi))
                } else {
                    let src_clif = clif_ty(&src_ty).expect("int source has a Clif type");
                    let w32 = if src_clif.bytes() < 4 { builder.ins().uextend(types::I32, x) } else { x };
                    let wide = if bits <= 32 { builder.ins().uextend(types::I64, w32) } else { w32 };
                    let (lo, hi) = crate::quad::quad_from_int(builder, wide, true);
                    Some(crate::quad::quad_store_carrier(builder, lo, hi))
                }
            } else {
                let dst = result_clif(&result_ty);
                let lowered = self.operand(builder, a);
                Some(builder.ins().fcvt_from_uint(dst, lowered))
            }
        }
        mir::InstKind::SFtoI(a) => {
            if self.is_f80(a) {
                // Truncation toward zero through fistp (which pins the
                // control word itself): sub-word destinations convert
                // at 32 bits, then narrow.
                let f80 = rt.f80.as_ref().expect("f80 conversion needs x86-64");
                let dst_ty = result_ty.clone().expect("sftoi needs a result type");
                let dst_bits = mir_int_bits(&dst_ty);
                if dst_bits > 64 {
                    // Through the vendored runtime (`__glcc_fixxfti`),
                    // matching clang exactly (including out-of-range).
                    let a_addr = self.f80_addr(builder, a);
                    let callee = clif.declare_func_in_func(f80.to_i128, builder.func);
                    let inst = builder.ins().call(callee, &[a_addr]);
                    let res = builder.inst_results(inst).to_vec();
                    Some(builder.ins().iconcat(res[0], res[1]))
                } else {
                    let a_addr = self.f80_addr(builder, a);
                    let (helper, from) = if dst_bits <= 32 { (f80.to_i32, types::I32) } else { (f80.to_i64, types::I64) };
                    let callee = clif.declare_func_in_func(helper, builder.func);
                    let inst = builder.ins().call(callee, &[a_addr]);
                    let r = builder.inst_results(inst)[0];
                    let dst = result_clif(&result_ty);
                    Some(resize_int(builder, r, from, dst))
                }
            } else if self.is_f128(a) {
                // Saturating truncation (see `quad.rs`): sub-word
                // destinations convert at 32 bits, then narrow;
                // 128-bit destinations convert wide directly.
                let dst_ty = result_ty.clone().expect("sftoi needs a result type");
                let dst_bits = mir_int_bits(&dst_ty);
                let x = self.operand(builder, a);
                let xa = crate::quad::quad_spill(builder, x);
                let (lo, hi) = crate::quad::quad_load_halves(builder, xa);
                if dst_bits > 64 {
                    Some(crate::quad::quad_to_s128(builder, lo, hi))
                } else {
                    let r = if dst_bits <= 32 {
                        crate::quad::quad_to_s32(builder, lo, hi)
                    } else {
                        crate::quad::quad_to_s64(builder, lo, hi)
                    };
                    let dst = result_clif(&result_ty);
                    let from = if dst_bits <= 32 { types::I32 } else { types::I64 };
                    Some(resize_int(builder, r, from, dst))
                }
            } else {
                let dst = result_clif(&result_ty);
                let lowered = self.operand(builder, a);
                // x64 converts at 32/64 bits only: sub-word
                // destinations go through I32 and narrow (the value is
                // in range or already saturated).
                let wide = if dst.bits() < 32 { types::I32 } else { dst };
                let converted = builder.ins().fcvt_to_sint(wide, lowered);
                Some(resize_int(builder, converted, wide, dst))
            }
        }
        mir::InstKind::UFtoI(a) => {
            if self.is_f80(a) {
                // Through the 64-bit unsigned conversion, then narrow
                // (gcc-observed: 32-bit results truncate the 64-bit
                // outcome).
                let f80 = rt.f80.as_ref().expect("f80 conversion needs x86-64");
                let dst_ty = result_ty.clone().expect("uftoi needs a result type");
                let dst_bits = mir_int_bits(&dst_ty);
                if dst_bits > 64 {
                    // Through the vendored runtime (`__glcc_fixunsxfti`),
                    // matching clang exactly (including out-of-range).
                    let a_addr = self.f80_addr(builder, a);
                    let callee = clif.declare_func_in_func(f80.to_u128, builder.func);
                    let inst = builder.ins().call(callee, &[a_addr]);
                    let res = builder.inst_results(inst).to_vec();
                    Some(builder.ins().iconcat(res[0], res[1]))
                } else {
                    let a_addr = self.f80_addr(builder, a);
                    let callee = clif.declare_func_in_func(f80.to_u64, builder.func);
                    let inst = builder.ins().call(callee, &[a_addr]);
                    let r = builder.inst_results(inst)[0];
                    let dst = result_clif(&result_ty);
                    Some(resize_int(builder, r, types::I64, dst))
                }
            } else if self.is_f128(a) {
                // Saturating truncation through 32/64/128 bits (see
                // `quad.rs`), then narrow like the platform.
                let dst_ty = result_ty.clone().expect("uftoi needs a result type");
                let dst_bits = mir_int_bits(&dst_ty);
                let x = self.operand(builder, a);
                let xa = crate::quad::quad_spill(builder, x);
                let (lo, hi) = crate::quad::quad_load_halves(builder, xa);
                if dst_bits > 64 {
                    Some(crate::quad::quad_to_u128(builder, lo, hi))
                } else {
                    let r = if dst_bits <= 32 {
                        crate::quad::quad_to_u32(builder, lo, hi)
                    } else {
                        crate::quad::quad_to_u64(builder, lo, hi)
                    };
                    let dst = result_clif(&result_ty);
                    let from = if dst_bits <= 32 { types::I32 } else { types::I64 };
                    Some(resize_int(builder, r, from, dst))
                }
            } else {
                let dst = result_clif(&result_ty);
                let lowered = self.operand(builder, a);
                let wide = if dst.bits() < 32 { types::I32 } else { dst };
                let converted = builder.ins().fcvt_to_uint(wide, lowered);
                Some(resize_int(builder, converted, wide, dst))
            }
        }
        mir::InstKind::Fptrunc(a) => {
            let src_ty = self.func.values[a].ty.clone();
            let dst_ty = result_ty.clone().expect("fptrunc needs a result type");
            if matches!(src_ty, mir::Ty::Float(mir::FloatKind::F80)) {
                // Single rounding through fstp under the live control.
                let f80 = rt.f80.as_ref().expect("f80 narrowing needs x86-64");
                if matches!(dst_ty, mir::Ty::Float(mir::FloatKind::F128)) {
                    // Exact widening to quad through the vendored runtime (x87 has
                    // no quad op): source by value, carrier back.
                    Some(self.f80_extend(builder, clif, f80.extend, a))
                } else {
                    let helper = match dst_ty {
                        mir::Ty::Float(mir::FloatKind::F32) => f80.to_f32,
                        mir::Ty::Float(mir::FloatKind::F64) => f80.to_f64,
                        _ => panic!("long double narrowing to non-float"),
                    };
                    let a_addr = self.f80_addr(builder, a);
                    let callee = clif.declare_func_in_func(helper, builder.func);
                    let inst = builder.ins().call(callee, &[a_addr]);
                    Some(builder.inst_results(inst)[0])
                }
            } else if matches!(src_ty, mir::Ty::Float(mir::FloatKind::F128)) {
                if matches!(dst_ty, mir::Ty::Float(mir::FloatKind::F80)) {
                    // Narrowing to x87 through the vendored runtime, captured like a
                    // foreign return (st(0) has no Clif name).
                    let f80 = rt.f80.as_ref().expect("quad narrowing needs x86-64");
                    Some(self.quad_trunc(builder, clif, rt, f80.trunc, a))
                } else {
                    // Single round-half-even through integer ops (see
                    // `quad.rs`); works on every target.
                    let x = self.operand(builder, a);
                    let xa = crate::quad::quad_spill(builder, x);
                    let (lo, hi) = crate::quad::quad_load_halves(builder, xa);
                    let bits = match dst_ty {
                        mir::Ty::Float(mir::FloatKind::F32) => crate::quad::quad_to_f32(builder, lo, hi),
                        mir::Ty::Float(mir::FloatKind::F64) => crate::quad::quad_to_f64(builder, lo, hi),
                        _ => panic!("quad narrowing to non-float"),
                    };
                    let dst = result_clif(&result_ty);
                    Some(builder.ins().bitcast(dst, MemFlagsData::new(), bits))
                }
            } else {
                let dst = result_clif(&result_ty);
                let lowered = self.operand(builder, a);
                Some(builder.ins().fdemote(dst, lowered))
            }
        }
        mir::InstKind::Fpext(a) => {
            let dst_ty = result_ty.clone().expect("fpext needs a result type");
            if matches!(dst_ty, mir::Ty::Float(mir::FloatKind::F80)) {
                if matches!(self.func.values[a].ty, mir::Ty::Float(mir::FloatKind::F128)) {
                    // Narrowing from quad through the vendored runtime (see above).
                    let f80 = rt.f80.as_ref().expect("quad widening needs x86-64");
                    Some(self.quad_trunc(builder, clif, rt, f80.trunc, a))
                } else {
                    // Exact widening through fld.
                    let f80 = rt.f80.as_ref().expect("f80 widening needs x86-64");
                    let src_ty = self.func.values[a].ty.clone();
                    let helper = match src_ty {
                        mir::Ty::Float(mir::FloatKind::F32) => f80.from_f32,
                        mir::Ty::Float(mir::FloatKind::F64) => f80.from_f64,
                        _ => panic!("widening to long double from non-float"),
                    };
                    let x = self.operand(builder, a);
                    let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
                    let out = builder.ins().stack_addr(types::I64, slot, 0);
                    let callee = clif.declare_func_in_func(helper, builder.func);
                    builder.ins().call(callee, &[out, x]);
                    Some(f80_result(builder, out))
                }
            } else if matches!(dst_ty, mir::Ty::Float(mir::FloatKind::F128)) {
                if matches!(self.func.values[a].ty, mir::Ty::Float(mir::FloatKind::F80)) {
                    // Exact widening from x87 through the vendored runtime.
                    let f80 = rt.f80.as_ref().expect("f80 widening needs x86-64");
                    Some(self.f80_extend(builder, clif, f80.extend, a))
                } else {
                    // Exact widening through integer ops (see `quad.rs`);
                    // works on every target, including x86-64 `__float128`.
                    let src_ty = self.func.values[a].ty.clone();
                    let x = self.operand(builder, a);
                    let (lo, hi) = match src_ty {
                        mir::Ty::Float(mir::FloatKind::F32) => {
                            let b = builder.ins().bitcast(types::I32, MemFlagsData::new(), x);
                            crate::quad::quad_from_f32(builder, b)
                        }
                        mir::Ty::Float(mir::FloatKind::F64) => {
                            let b = builder.ins().bitcast(types::I64, MemFlagsData::new(), x);
                            crate::quad::quad_from_f64(builder, b)
                        }
                        _ => panic!("widening to quad from non-float"),
                    };
                    Some(crate::quad::quad_store_carrier(builder, lo, hi))
                }
            } else {
                let dst = result_clif(&result_ty);
                let lowered = self.operand(builder, a);
                Some(builder.ins().fpromote(dst, lowered))
            }
        }
        mir::InstKind::PtrToInt(a) => {
            let dst = result_clif(&result_ty);
            let x = self.operand(builder, a);
            let from = builder.func.dfg.value_type(x);
            Some(if from == dst {
                x
            } else if from.bytes() == dst.bytes() {
                builder.ins().bitcast(dst, MemFlagsData::new(), x)
            } else if from.bytes() < dst.bytes() {
                // A narrow pointer sign-extends into an integer, its
                // default `__ptr32` semantics.
                builder.ins().sextend(dst, x)
            } else {
                builder.ins().ireduce(dst, x)
            })
        }
        mir::InstKind::IntToPtr(a) => {
            let dst = result_clif(&result_ty);
            let x = self.operand(builder, a);
            let from = builder.func.dfg.value_type(x);
            Some(if from == dst {
                x
            } else if from.bytes() == dst.bytes() {
                builder.ins().bitcast(dst, MemFlagsData::new(), x)
            } else if from.bytes() < dst.bytes() {
                builder.ins().sextend(dst, x)
            } else {
                builder.ins().ireduce(dst, x)
            })
        }
        mir::InstKind::Select { cond, then_val, else_val } => {
            let mut c = self.operand(builder, cond);
            let x = self.operand(builder, then_val);
            let y = self.operand(builder, else_val);
            let data = builder.func.dfg.value_type(x);
            let cond_ty = builder.func.dfg.value_type(c);
            if cond_ty.is_vector() && data.is_vector() {
                // Vector selects are bitwise: cranelift's `select`
                // only takes a scalar truthy condition, so an `fcmp`
                // lane mask rides `bitselect` in the data's own shape.
                if cond_ty != data {
                    c = builder.ins().bitcast(data, MemFlagsData::new().with_endianness(cranelift_codegen::ir::Endianness::Little), c);
                }
                Some(builder.ins().bitselect(c, x, y))
            } else {
                Some(builder.ins().select(c, x, y))
            }
        }
        mir::InstKind::Alloca { pointee, count } => {
            let size = pointee.size_bytes().checked_mul(count).expect("alloca size overflow");
            let slot = stack_slot(
                builder,
                &mir::Ty::Array {
                    elem: Box::new(mir::Ty::int(8)),
                    len: size,
                },
            );
            if let Some(result) = data.result {
                self.slots.insert(result, slot);
            }
            Some(builder.ins().stack_addr(types::I64, slot, 0))
        }
        mir::InstKind::Load { addr, flags } => {
            // This Cranelift has no volatile memory flag: a plain Clif
            // load is fair game for alias analysis (RLE, forwarding).
            // Volatile MIR accesses lower to atomic ops instead, which
            // the analysis never rewrites; see `volatile_load`.
            let x = self.addr_operand(builder, addr);
            let Some(mir_ty) = result_ty.clone() else {
                panic!("load needs a result type");
            };
            if mir_ty.is_aggregate() {
                if flags.volatile {
                    builder.ins().fence();
                }
                let slot = stack_slot(builder, &mir_ty);
                let dst = builder.ins().stack_addr(types::I64, slot, 0);
                if flags.volatile {
                    emit_memcpy(builder, dst, x, mir_ty.size_bytes());
                } else {
                    self.emit_copy(builder, dst, x, mir_ty.size_bytes());
                }
                if flags.volatile {
                    builder.ins().fence();
                }
                Some(dst)
            } else {
                let ty = clif_ty(&mir_ty).unwrap_or_else(|| panic!("load of non-scalar type {mir_ty:?}"));
                if flags.volatile {
                    Some(volatile_load(builder, ty, x))
                } else {
                    Some(builder.ins().load(ty, memflags::notrap(), x, Offset32::new(0)))
                }
            }
        }
        mir::InstKind::Store { addr, val, flags } => {
            let dst = self.addr_operand(builder, addr);
            let src = self.operand(builder, val);
            let mir_ty = self.func.values[val].ty.clone();
            if mir_ty.is_aggregate() {
                if flags.volatile {
                    builder.ins().fence();
                    emit_memcpy(builder, dst, src, mir_ty.size_bytes());
                    builder.ins().fence();
                } else {
                    self.emit_copy(builder, dst, src, mir_ty.size_bytes());
                }
            } else {
                let ty = clif_ty(&mir_ty).unwrap_or_else(|| panic!("store of non-scalar type {mir_ty:?}"));
                if flags.volatile {
                    volatile_store(builder, ty, src, dst);
                } else {
                    builder.ins().store(memflags::notrap(), src, dst, Offset32::new(0));
                }
            }
            None
        }
        mir::InstKind::GetElementPtr { base, indices } => {
            let base_addr = self.addr_operand(builder, base);
            let mir::Ty::Ptr { pointee, .. } = self.func.values[base].ty.clone() else {
                panic!("gep base is not a pointer");
            };
            let addr = self.gep_addr(builder, base_addr, &pointee, &indices);
            // A `__ptr32` result keeps its narrow width in the value.
            let want = result_clif(&result_ty);
            let got = builder.func.dfg.value_type(addr);
            Some(if got == want { addr } else { builder.ins().ireduce(want, addr) })
        }
        mir::InstKind::ExtractValue { agg, index } => {
            let addr = self.operand(builder, agg);
            let mir_ty = self.func.values[agg].ty.clone();
            let (off, elem) = field_offset(&mir_ty, index);
            match clif_ty(&elem) {
                Some(ty) => {
                    let addr = if off == 0 { addr } else { builder.ins().iadd_imm_u(addr, off as i64) };
                    Some(atomic_load_value(builder, ty, addr))
                }
                // Nested aggregates address into the base slot.
                None => Some(if off == 0 { addr } else { builder.ins().iadd_imm_u(addr, off as i64) }),
            }
        }
        mir::InstKind::InsertValue { agg, elem, index } => {
            let base = self.operand(builder, agg);
            let val = self.operand(builder, elem);
            let mir_ty = self.func.values[agg].ty.clone();
            let (off, elem_ty) = field_offset(&mir_ty, index);
            let slot = stack_slot(builder, &mir_ty);
            let dst = builder.ins().stack_addr(types::I64, slot, 0);
            if !self.is_undef(agg) {
                self.emit_copy(builder, dst, base, mir_ty.size_bytes());
            }
            if elem_ty.is_aggregate() {
                self.emit_memcpy_offset(builder, dst, off, val, elem_ty.size_bytes());
            } else {
                builder.ins().store(memflags::notrap(), val, dst, Offset32::new(off as i32));
            }
            Some(dst)
        }
        mir::InstKind::Splat(a) => {
            let scalar = self.operand(builder, a);
            let ty = result_clif(&result_ty);
            Some(builder.ins().splat(ty, scalar))
        }
        mir::InstKind::ExtractLane { vec, lane } => {
            let v = self.operand(builder, vec);
            let ty = clif_ty(&result_ty.expect("extractlane needs a result type"))
                .expect("extractlane result is a scalar");
            Some(builder.ins().extractlane(v, lane))
        }
        mir::InstKind::Memcpy { dst, src, len } => {
            let d = self.addr_operand(builder, dst);
            let s = self.addr_operand(builder, src);
            self.emit_copy(builder, d, s, len);
            None
        }
        mir::InstKind::Memset { dst, byte, len } => {
            let d = self.addr_operand(builder, dst);
            // Splat the byte across 8-byte stores, then mop up the tail.
            let splat = (byte as u64).wrapping_mul(0x0101010101010101);
            let wide = builder.ins().iconst(types::I64, splat as i64);
            let words = len / 8;
            for i in 0..words {
                builder.ins().store(memflags::notrap(), wide, d, Offset32::new((i * 8) as i32));
            }
            let fill = builder.ins().iconst(types::I8, byte as i64);
            for i in (words * 8)..len {
                builder.ins().store(memflags::notrap(), fill, d, Offset32::new(i as i32));
            }
            None
        }
        mir::InstKind::GlobalAddr { global } => {
            let id = data_ids[&global];
            let gv = clif.declare_data_in_func(id, builder.func);
            // Thread-local storage addresses through the TLS
            // descriptor (`elf_gd` model, like the foreign-variadic
            // trampoline); plain globals use the absolute symbol
            // value.
            Some(if tls_data.contains(&global) {
                builder.ins().tls_value(types::I64, gv)
            } else {
                builder.ins().symbol_value(types::I64, gv)
            })
        }
        mir::InstKind::FuncAddr { func } => {
            let id = func_ids[&func];
            let funcref = clif.declare_func_in_func(id, builder.func);
            Some(builder.ins().func_addr(types::I64, funcref))
        }
        mir::InstKind::ExternAddr { target } => {
            let id = extern_ids[&target];
            let funcref = clif.declare_func_in_func(id, builder.func);
            Some(builder.ins().func_addr(types::I64, funcref))
        }
        mir::InstKind::Call { callee, args } => {
            let callee_layout = layouts[&callee].clone();
            emit_call(
                self,
                builder,
                clif,
                func_ids,
                extern_ids,
                data_ids,
                rt,
                layouts,
                extern_layouts,
                CallTarget::Internal(callee),
                &callee_layout,
                &args,
                &result_ty,
                data.result,
                None,
            )
        }
        mir::InstKind::CallExtern { target, args } => {
            let layout = extern_layouts[&target].clone();
            emit_call(
                self,
                builder,
                clif,
                func_ids,
                extern_ids,
                data_ids,
                rt,
                layouts,
                extern_layouts,
                CallTarget::External(target),
                &layout,
                &args,
                &result_ty,
                data.result,
                None,
            )
        }
        // The coroutine pass rewrites every `await` to a poll loop
        // before Clif lowering; one here means the pass was skipped.
        mir::InstKind::Await { .. } => panic!("`await` instruction survived coroutine lowering"),
        mir::InstKind::CallIndirect { callee, args, sig } => {
            let layout = abi::layout_sig(self.target, &sig.params, &sig.ret, sig.variadic);
            let addr = self.addr_operand(builder, callee);
            emit_call(
                self,
                builder,
                clif,
                func_ids,
                extern_ids,
                data_ids,
                rt,
                layouts,
                extern_layouts,
                CallTarget::Indirect { addr, variadic: sig.variadic },
                &layout,
                &args,
                &result_ty,
                data.result,
                None,
            )
        }
        mir::InstKind::VaStart { ap } => {
            let ap_addr = self.addr_operand(builder, ap);
            let va = self.va.expect("va_start outside a variadic function");
            abi::emit_va_start(self.target, builder, ap_addr, &va, &self.layout);
            None
        }
        mir::InstKind::VaArg { ap } => {
            let ap_addr = self.addr_operand(builder, ap);
            let Some(mir_ty) = result_ty.clone() else {
                panic!("va_arg needs a result type");
            };
            // A `va_list` parameter read in a non-variadic function has
            // no prelude of its own: x86-64 and AAPCS64 keep all cursor
            // state in the object itself, so those lower against a
            // dummy context (untouched on these targets). RISC-V folds
            // the caller's spill layout into the walk, which no object
            // carries.
            let (ctx, layout) = match self.va {
                Some(va) => (va, self.layout.clone()),
                None => {
                    if self.target.arch == Arch::Riscv64 {
                        panic!("va_arg outside a variadic function on RISC-V");
                    }
                    (
                        abi::VaCtx {
                            save_base: ap_addr,
                            overflow: ap_addr,
                        },
                        self.layout.clone(),
                    )
                }
            };
            match abi::emit_va_arg(self.target, builder, ap_addr, &ctx, &layout, &mir_ty) {
                abi::VaArgValue::Scalar(v) => Some(v),
                abi::VaArgValue::Addr(a) => Some(a),
            }
        }
        mir::InstKind::VaEnd { .. } => None,
        mir::InstKind::VaCopy { dst, src } => {
            let d = self.addr_operand(builder, dst);
            let s = self.addr_operand(builder, src);
            abi::emit_va_copy(self.target, builder, d, s);
            None
        }
        mir::InstKind::AtomicLoad { addr } => {
            let a = self.addr_operand(builder, addr);
            let Some(mir_ty) = result_ty.clone() else {
                panic!("atomic load needs a result type");
            };
            if mir_ty.is_aggregate() {
                builder.ins().fence();
                let slot = stack_slot(builder, &mir_ty);
                let dst = builder.ins().stack_addr(types::I64, slot, 0);
                emit_memcpy(builder, dst, a, mir_ty.size_bytes());
                builder.ins().fence();
                Some(dst)
            } else {
                let ty = result_clif(&Some(mir_ty));
                Some(atomic_load_value(builder, ty, a))
            }
        }
        mir::InstKind::AtomicStore { addr, val } => {
            let a = self.addr_operand(builder, addr);
            let v = self.operand(builder, val);
            let mir_ty = self.func.values[val].ty.clone();
            if mir_ty.is_aggregate() {
                builder.ins().fence();
                emit_memcpy(builder, a, v, mir_ty.size_bytes());
                builder.ins().fence();
            } else {
                let ty = clif_ty(&mir_ty).unwrap_or_else(|| panic!("atomic store of non-scalar type {mir_ty:?}"));
                atomic_store_value(builder, ty, v, a);
            }
            None
        }
        mir::InstKind::AtomicRmw { op, addr, val } => {
            let ty = result_clif(&result_ty);
            let a = self.addr_operand(builder, addr);
            let v = self.operand(builder, val);
            let rmw = match op {
                mir::AtomicRmwOp::Add => AtomicRmwOp::Add,
                mir::AtomicRmwOp::Sub => AtomicRmwOp::Sub,
                mir::AtomicRmwOp::And => AtomicRmwOp::And,
                mir::AtomicRmwOp::Nand => AtomicRmwOp::Nand,
                mir::AtomicRmwOp::Or => AtomicRmwOp::Or,
                mir::AtomicRmwOp::Xor => AtomicRmwOp::Xor,
                mir::AtomicRmwOp::Xchg => AtomicRmwOp::Xchg,
                mir::AtomicRmwOp::Umin => AtomicRmwOp::Umin,
                mir::AtomicRmwOp::Umax => AtomicRmwOp::Umax,
                mir::AtomicRmwOp::Smin => AtomicRmwOp::Smin,
                mir::AtomicRmwOp::Smax => AtomicRmwOp::Smax,
            };
            Some(builder.ins().atomic_rmw(ty, MemFlagsData::new(), rmw, a, v))
        }
        mir::InstKind::AtomicCas { addr, expected, replacement } => {
            let _ty = result_clif(&result_ty);
            let a = self.addr_operand(builder, addr);
            let e = self.operand(builder, expected);
            let r = self.operand(builder, replacement);
            Some(builder.ins().atomic_cas(MemFlagsData::new(), a, e, r))
        }
        mir::InstKind::AtomicFence => {
            builder.ins().fence();
            None
        }
        mir::InstKind::InlineAsm { block } => self.emit_inline_asm(builder, clif, &block, wrappers),
        mir::InstKind::GetFramePointer => Some(builder.ins().get_frame_pointer(result_clif(&result_ty))),
    };
    match (data.result, lowered) {
        (Some(mir_value), Some(clif_value)) => {
            self.values.insert(mir_value, clif_value);
        }
        (None, None) => {}
        (Some(_), None) => panic!("instruction dropped its result value"),
        (None, Some(_)) => panic!("resultless instruction produced a value"),
    }
}
/// Lower a GEP to address arithmetic: constant steps fold to one
/// displacement, dynamic array indices scale and accumulate.
pub(crate) fn gep_addr(&mut self, builder: &mut FunctionBuilder, base: ClifValue, pointee: &mir::Ty, indices: &[mir::GepIndex]) -> ClifValue {
    let mut ty = pointee.clone();
    let mut fixed: i64 = 0;
    let mut dynamic: Vec<ClifValue> = Vec::new();
    for index in indices {
        match index {
            mir::GepIndex::Field(i) => {
                let mir::Ty::Struct { fields, .. } = &ty else {
                    panic!("field step into non-struct");
                };
                let offsets = mir::struct_field_offsets(&ty);
                let slot = *i as usize;
                if slot >= fields.len() {
                    panic!("field step out of range");
                }
                fixed += offsets[slot] as i64;
                ty = fields[slot].clone();
            }
            mir::GepIndex::ConstIndex(k) => match &ty {
                mir::Ty::Array { elem, .. } => {
                    fixed += (*k * elem.size_bytes()) as i64;
                    ty = elem.as_ref().clone();
                }
                // Element step over a pointed-to scalar: `p++` scales by
                // the pointee size, not by bytes.
                mir::Ty::Int { .. } | mir::Ty::Float(_) | mir::Ty::Bool | mir::Ty::Ptr { .. } | mir::Ty::WideInt { .. } => {
                    fixed += (*k * ty.size_bytes()) as i64;
                }
                _ => panic!("constant step into aggregate"),
            },
            mir::GepIndex::Index(v) => {
                let (scale, next) = match &ty {
                    mir::Ty::Array { elem, .. } => (elem.size_bytes(), elem.as_ref().clone()),
                    // Pointer arithmetic over a single element.
                    mir::Ty::Int { .. } | mir::Ty::Float(_) | mir::Ty::Bool | mir::Ty::Ptr { .. } | mir::Ty::WideInt { .. } => (ty.size_bytes(), ty.clone()),
                    _ => panic!("dynamic step into aggregate"),
                };
                let idx = self.operand(builder, *v);
                // C indices are signed (`p[-1]` walks backwards), so
                // narrow values sign-extend like GCC's `movslq`. (MIR
                // integers carry no signedness; an unsigned index above
                // 2^31 would mis-step, but such indices don't occur
                // while negative ones are common.)
                let wide = match clif_ty(&self.func.values[*v].ty) {
                    Some(t) if t.bytes() < 8 => builder.ins().sextend(types::I64, idx),
                    Some(t) if t.bytes() == 8 => idx,
                    _ => panic!("dynamic gep index must fit in 64 bits"),
                };
                let scaled = if scale == 1 { wide } else { builder.ins().imul_imm_u(wide, scale as i64) };
                dynamic.push(scaled);
                ty = next;
            }
        }
    }
    let mut addr = base;
    if fixed != 0 {
        addr = builder.ins().iadd_imm_u(addr, fixed);
    }
    for scaled in dynamic {
        addr = builder.ins().iadd(addr, scaled);
    }
    addr
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_term<M: Module>(
    &mut self,
    builder: &mut FunctionBuilder,
    clif: &mut M,
    func_ids: &HashMap<mir::Func, FuncId>,
    extern_ids: &HashMap<mir::Extern, FuncId>,
    rt: &Rt,
    layouts: &HashMap<mir::Func, SigLayout>,
    extern_layouts: &HashMap<mir::Extern, SigLayout>,
    term: mir::Terminator,
) {
    use cranelift_codegen::ir::{BlockArg, BlockCall};
    match term {
        mir::Terminator::Br { dest, args } => {
            let target = self.blocks[&dest];
            let lowered = self.lower_branch_args(builder, dest, &args);
            builder.ins().jump(target, &lowered);
        }
        mir::Terminator::CondBr {
            cond,
            then_block,
            else_block,
            then_args,
            else_args,
        } => {
            let test = self.cond_value(builder, cond);
            let then_target = self.blocks[&then_block];
            let else_target = self.blocks[&else_block];
            let then_lowered = self.lower_branch_args(builder, then_block, &then_args);
            let else_lowered = self.lower_branch_args(builder, else_block, &else_args);
            builder.ins().brif(test, then_target, &then_lowered, else_target, &else_lowered);
        }
        mir::Terminator::Switch { scrutinee, cases, default } => {
            let scrut = self.operand(builder, scrutinee);
            let mir_ty = self.func.values[scrutinee].ty.clone();
            let width = match clif_ty(&mir_ty) {
                Some(t) if t.is_int() && t.bytes() <= 8 => t,
                _ => panic!("switch scrutinee must fit in 64 bits"),
            };
            let default_target = self.blocks[&default];
            if !self.func.blocks[default].params.is_empty() {
                panic!("switch target takes block params");
            }
            if cases.is_empty() {
                builder.ins().jump(default_target, &[]);
                return;
            }
            for (_, dest) in &cases {
                if !self.func.blocks[*dest].params.is_empty() {
                    panic!("switch target takes block params");
                }
            }
            // Non-negative cases map directly onto Cranelift's switch
            // helper, which uses jump tables for dense runs and a binary
            // search over sparse ones. Negative or high-bit cases keep
            // the compare chain below.
            let max_case = match width.bits() {
                64 => i64::MAX as i128,
                bits => (1i128 << (bits - 1)) - 1,
            };
            if cases.iter().all(|(value, _)| *value >= 0 && *value <= max_case) {
                let base = cases.iter().map(|(value, _)| *value).min().expect("non-empty cases");
                let max = cases.iter().map(|(value, _)| *value).max().expect("non-empty cases");
                let mut entries: Vec<(i128, ClifBlock)> = cases.iter().map(|(value, dest)| (*value, self.blocks[dest])).collect();
                entries.sort_by_key(|(value, _)| *value);
                // Cranelift's Switch splits the table at holes. When the
                // cases cover nearly the whole span, filling the holes
                // with the default keeps the dispatch to one range check
                // and one table lookup.
                let dense = dense_switch_span(base, max, entries.len());
                let mut switch = Switch::new();
                if dense {
                    let mut next = base;
                    for (value, dest) in entries {
                        while next < value {
                            switch.set_entry((next - base) as u128, default_target);
                            next += 1;
                        }
                        switch.set_entry((value - base) as u128, dest);
                        next = value + 1;
                    }
                    while next <= max {
                        switch.set_entry((next - base) as u128, default_target);
                        next += 1;
                    }
                } else {
                    for (value, dest) in entries {
                        switch.set_entry((value - base) as u128, dest);
                    }
                }
                let index = if base == 0 {
                    scrut
                } else {
                    let delta = builder.ins().iconst(width, base as i64);
                    builder.ins().isub(scrut, delta)
                };
                switch.emit(builder, index, default_target);
                // The helper creates and switches to its own dispatch
                // blocks, so the next source block must force a switch.
                self.current = None;
                return;
            }
            // If-else chain: one test block per case, falling through to
            // the next test and finally the default.
            let mut tests = Vec::new();
            for _ in 0..cases.len() {
                tests.push(builder.create_block());
            }
            builder.ins().jump(tests[0], &[]);
            self.switch_to( &mut *builder, tests[0]);
            for (i, (value, dest)) in cases.into_iter().enumerate() {
                self.switch_to( &mut *builder, tests[i]);
                let target = self.blocks[&dest];
                let probe = builder.ins().iconst(width, value as i64);
                let same = builder.ins().icmp(IntCC::Equal, scrut, probe);
                let otherwise = if i + 1 < tests.len() { tests[i + 1] } else { default_target };
                builder.ins().brif(same, target, &[], otherwise, &[]);
            }
        }
        mir::Terminator::Resume { exn, .. } => {
            // `_Unwind_Resume` never returns; keep the selector unreferenced.
            let exn_clif = self.operand(builder, exn);
            let mut sig = cranelift_codegen::ir::Signature::new(self.target.call_conv());
            sig.params.push(cranelift_codegen::ir::AbiParam::new(types::I64));
            let func_id = clif
                .declare_function("_Unwind_Resume", Linkage::Import, &sig)
                .unwrap_or_else(|e| panic!("cannot declare _Unwind_Resume: {e:?}"));
            let funcref: FuncRef = clif.declare_func_in_func(func_id, builder.func);
            builder.ins().call(funcref, &[exn_clif]);
            builder.ins().trap(TrapCode::user(1).expect("nonzero trap code"));
            self.current = None;
            return;
        }
        mir::Terminator::Invoke { callee, args, sig, normal, normal_args, unwind } => {
            let normal_target = self.blocks[&normal];
            let unwind_target = self.blocks[&unwind];
            // The normal edge passes the call's results first, then the
            // extra branch arguments.
            let returns = usize::from(!sig.ret.is_void());
            assert_eq!(
                self.func.blocks[normal].params.len(),
                returns + normal_args.len(),
                "invoke normal edge arity mismatch"
            );
            let mut edge_args: Vec<BlockArg> = (0..returns as u32).map(|i| BlockArg::TryCallRet(i)).collect();
            edge_args.extend(normal_args.iter().map(|a| BlockArg::Value(self.operand(builder, *a))));
            let payload = self
                .target
                .call_conv()
                .exception_payload_types(types::I64)
                .len();
            let exn_args: Vec<BlockArg> = (0..payload as u32).map(|i| BlockArg::TryCallExn(i)).collect();
            let normal_call = BlockCall::new(normal_target, edge_args, &mut builder.func.dfg.value_lists);
            let unwind_call = BlockCall::new(unwind_target, exn_args, &mut builder.func.dfg.value_lists);
            let (target, callee_layout) = match callee {
                mir::InvokeCallee::Func(f) => (CallTarget::Internal(f), layouts[&f].clone()),
                mir::InvokeCallee::Extern(e) => (CallTarget::External(e), extern_layouts[&e].clone()),
                mir::InvokeCallee::Indirect(v) => {
                    let addr = self.addr_operand(builder, v);
                    let layout = abi::layout_sig(self.target, &sig.params, &sig.ret, sig.variadic);
                    (CallTarget::Indirect { addr, variadic: sig.variadic }, layout)
                }
            };
            let try_targets = Some(super::call::TryTargets { normal: normal_call, unwind: unwind_call });
            let ret_ty = sig.ret.clone();
            emit_call(
                self,
                builder,
                clif,
                func_ids,
                extern_ids,
                &HashMap::new(),
                rt,
                layouts,
                extern_layouts,
                target,
                &callee_layout,
                &args,
                &Some(ret_ty),
                None,
                try_targets,
            );
            // try_call is a terminator: the next source block must force
            // a block switch.
            self.current = None;
            return;
        }
        mir::Terminator::Return { val } => {
            // Clone the piece list first: loading the value below
            // reborrows the self. Multi-piece returns (aggregates and
            // 128-bit scalar pairs) need piece handling either way.
            let ret_pieces = match &self.layout.ret {
                abi::RetLoc::Regs(pieces) if val.is_some() && (self.func.sig.ret.is_aggregate() || pieces.len() > 1) => Some(pieces.clone()),
                _ => None,
            };
            match &self.layout.ret {
                abi::RetLoc::Sret => {
                    let slot = self.sret.expect("aggregate return needs an out-pointer");
                    let addr = self.operand(builder, val.expect("aggregate return needs a value"));
                    self.emit_copy(builder, slot, addr, self.func.sig.ret.size_bytes());
                    builder.ins().return_(&[]);
                }
                abi::RetLoc::Regs(_) if ret_pieces.is_some() => {
                    let pieces = ret_pieces.unwrap();
                    // Multi-piece scalars split one value; aggregates
                    // load each piece from the slot.
                    if pieces.len() > 1 && !self.func.sig.ret.is_aggregate() {
                        let v = self.operand(builder, val.unwrap());
                        let from = builder.func.dfg.value_type(v);
                        let outs = split_value(builder, v, from, pieces.len());
                        builder.ins().return_(&outs);
                    } else {
                        let addr = self.operand(builder, val.unwrap());
                        let mut outs = Vec::with_capacity(pieces.len());
                        for (off, ty) in pieces {
                            outs.push(builder.ins().load(ty, memflags::notrap(), addr, Offset32::new(off as i32)));
                        }
                        builder.ins().return_(&outs);
                    }
                }
                _ => {
                    // A zero-sized aggregate return crosses as nothing;
                    // its value is an address that is simply dropped.
                    if matches!(self.layout.ret, abi::RetLoc::Void) {
                        builder.ins().return_(&[]);
                    } else if let Some(v) = val {
                        let x = self.operand(builder, v);
                        // Non-x86 sub-int returns use W-register
                        // slots: widen the produced value to the slot
                        // width.
                        let from = builder.func.dfg.value_type(x);
                        let x = match &self.layout.ret {
                            abi::RetLoc::Regs(pieces) if pieces.len() == 1 && !from.is_float() && !pieces[0].1.is_float() => resize_int(builder, x, from, pieces[0].1),
                            _ => x,
                        };
                        builder.ins().return_(&[x]);
                    } else {
                        builder.ins().return_(&[]);
                    }
                }
            }
        }
        mir::Terminator::Unreachable => {
            builder.ins().trap(TrapCode::user(1).expect("nonzero trap code"));
        }
    }
}
pub(crate) fn lower_branch_args(&mut self, builder: &mut FunctionBuilder, dest: mir::Block, args: &[mir::Value]) -> Vec<BlockArg> {
    let params = &self.func.blocks[dest].params;
    if params.len() != args.len() {
        panic!("branch arity mismatch");
    }
    args.iter().map(|a| BlockArg::Value(self.operand(builder, *a))).collect()
}
/// Any integer condition as an I8 test: nonzero is true.
pub(crate) fn cond_value(&mut self, builder: &mut FunctionBuilder, cond: mir::Value) -> ClifValue {
    let x = self.operand(builder, cond);
    match &self.func.values[cond].ty {
        mir::Ty::Bool | mir::Ty::Int { bits: 1 } => x,
        mir::Ty::Int { .. } => builder.ins().icmp_imm_u(IntCC::NotEqual, x, 0),
        _ => panic!("branch condition must be an integer"),
    }
}
/// Switch the builder only when the target differs: re-switching the
/// current block trips the builder's filled-check when setup code
/// already emitted there.
pub(crate) fn switch_to(&mut self, builder: &mut FunctionBuilder, target: ClifBlock) {
    if self.current != Some(target) {
        builder.switch_to_block(target);
        self.current = Some(target);
    }
}
}

/// The current value of an output lvalue, for a `+` operand's read side.
fn load_operand(builder: &mut FunctionBuilder, ty: &mir::Ty, addr: ClifValue) -> ClifValue {
    let clif_ty = clif_ty(ty).expect("asm register operand needs a scalar type");
    builder.ins().load(clif_ty, memflags::notrap(), addr, Offset32::new(0))
}

/// Widen an operand value to the 64-bit register the fork passes
/// operands in; upper bits are unspecified, as in GCC.
fn widen_operand(builder: &mut FunctionBuilder, ty: &mir::Ty, value: ClifValue) -> ClifValue {
    match clif_ty(ty) {
        Some(t) if t == types::I64 => value,
        Some(_) => builder.ins().uextend(types::I64, value),
        None => panic!("asm register operand needs a scalar type"),
    }
}
