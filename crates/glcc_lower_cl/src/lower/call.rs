use std::collections::HashMap;

use cranelift_codegen::ir::immediates::Offset32;
use cranelift_codegen::ir::types::{self, Type};
use cranelift_codegen::ir::{
    AbiParam, ArgumentExtension, ArgumentPurpose, FuncRef, InstBuilder, MemFlagsData, Signature, StackSlotData,
    StackSlotKind,
};
use cranelift_codegen::ir::Value as ClifValue;
use cranelift_frontend::FunctionBuilder;
use cranelift_module::{FuncId, Module};

use glcc_mir as mir;

use crate::abi::{self, Arch, Os, SigLayout, Target};
use crate::memflags;
use crate::ty::clif_ty;
use super::*;

/// Which function a call targets: a known module function, a foreign
/// import, or a runtime code address for an indirect call.
#[derive(Clone, Copy)]
pub(crate) enum CallTarget {
    Internal(mir::Func),
    External(mir::Extern),
    Indirect { addr: ClifValue, variadic: bool },
}

/// Emit a call: map MIR arguments onto the callee's slots (named
/// prefix) plus caller-built tail slots (variadic actuals), then route
/// direct, indirect, or through the x86-64 `al` trampoline.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_call<M: Module>(
    body: &mut Body,
    builder: &mut FunctionBuilder,
    clif: &mut M,
    func_ids: &HashMap<mir::Func, FuncId>,
    extern_ids: &HashMap<mir::Extern, FuncId>,
    _data_ids: &HashMap<mir::Global, cranelift_module::DataId>,
    rt: &Rt,
    layouts: &HashMap<mir::Func, SigLayout>,
    _extern_layouts: &HashMap<mir::Extern, SigLayout>,
    target: CallTarget,
    callee_layout: &SigLayout,
    args: &[mir::Value],
    result_ty: &Option<mir::Ty>,
    result_val: Option<mir::Value>,
) -> Option<ClifValue> {
    let (variadic, foreign) = match target {
        CallTarget::Internal(f) => (layouts[&f].variadic, false),
        CallTarget::External(_) => (callee_layout.variadic, true),
        // Indirect calls always go through the per-call signature;
        // variadic ones take the platform's indirect path below.
        CallTarget::Indirect { variadic, .. } => (variadic, true),
    };
    let _ = foreign;
    // A direct foreign call returning x86-64 `long double` leaves it
    // in st(0), not in an F128 register (SysV): declare no Clif
    // return and fetch it through the getret helper below.
    //
    // Limits, all documented rather than silently handled:
    // - Indirect calls assume the internal (xmm0-carrier) convention:
    //   a function pointer to a *foreign* long-double function reads
    //   xmm0 garbage. Address-taken internal functions (the common
    //   case, including callbacks returning other types) are exact.
    // - The reverse direction is unsupported: our address taken on a
    //   long-double-returning function and called from foreign code
    //   writes xmm0 while the caller reads st(0).
    // - A signature mismatch (MIR says F80, the symbol does not return
    //   long double) pops a stale st(0): an ABI mismatch like any
    //   other, loud only in its symptoms.
    let foreign_f80_ret = matches!(target, CallTarget::External(_)) && matches!(result_ty, Some(mir::Ty::Float(mir::FloatKind::F80)));
    let named = callee_layout.params.len();
    // Hidden out-pointer for aggregate returns comes first everywhere.
    let mut clif_args: Vec<ClifValue> = Vec::new();
    let mut sig_tys: Vec<(Type, ArgumentPurpose)> = Vec::new();
    let sret_temp = match &callee_layout.ret {
        abi::RetLoc::Sret => {
            let size = result_ty.as_ref().map(|t| t.size_bytes()).expect("sret needs a result type");
            let slot = stack_slot(
                builder,
                &mir::Ty::Array {
                    elem: Box::new(mir::Ty::int(8)),
                    len: size,
                },
            );
            let addr = builder.ins().stack_addr(types::I64, slot, 0);
            clif_args.push(addr);
            sig_tys.push((types::I64, ArgumentPurpose::StructReturn));
            Some(addr)
        }
        _ => None,
    };
    // Named prefix: fill the callee's slots positionally. Owners
    // spanning slots (128-bit scalar pairs) split once and reuse.
    // Variadic callees pad their signatures to the full register file
    // so the prelude spills register-passed varargs; callers skip those
    // pads and pass the tail actuals directly, or the pads land zeros
    // in exactly the registers the callee reads.
    let mut splits: HashMap<usize, Vec<ClifValue>> = HashMap::new();
    for slot in &callee_layout.slots {
        match slot.kind {
            abi::SlotKind::Sret => continue,
            abi::SlotKind::Pad if variadic => continue,
            abi::SlotKind::Pad => {
                // Pads need type-appropriate zeros (`iconst` on a
                // float type fails verification).
                let zero = if slot.ty.is_float() {
                    match slot.ty {
                        types::F32 => builder.ins().f32const(0.0),
                        _ => builder.ins().f64const(0.0),
                    }
                } else {
                    builder.ins().iconst(slot.ty, 0)
                };
                clif_args.push(zero);
                sig_tys.push((slot.ty, slot.purpose));
            }
            abi::SlotKind::Value { owner, offset } => {
                let arg = args[owner];
                let arg_ty = body.func.values[arg].ty.clone();
                let multi = !arg_ty.is_aggregate()
                    && callee_layout
                        .slots
                        .iter()
                        .filter(|s| matches!(s.kind, abi::SlotKind::Value { owner: o, .. } if o == owner))
                        .count()
                        > 1;
                if multi {
                    if let std::collections::hash_map::Entry::Vacant(e) = splits.entry(owner) {
                        let v = body.operand(builder, arg);
                        let from = builder.func.dfg.value_type(v);
                        e.insert(split_value(builder, v, from, 2));
                    }
                    let idx = (offset / 8) as usize;
                    clif_args.push(splits[&owner][idx]);
                } else {
                    let v = body.call_piece(builder, arg, offset, slot.ty);
                    clif_args.push(v);
                }
                sig_tys.push((slot.ty, slot.purpose));
            }
            abi::SlotKind::StackAddr { owner } | abi::SlotKind::RefAddr { owner } => {
                let arg = args[owner];
                let arg_ty = body.func.values[arg].ty.clone();
                if arg_ty.is_aggregate() {
                    clif_args.push(body.operand(builder, arg));
                } else {
                    // Scalar behind a stack slot (x86-64 `long
                    // double`): spill to a temp first.
                    let v = body.operand(builder, arg);
                    let slot = stack_slot(builder, &arg_ty);
                    let addr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(memflags::trusted(), v, addr, Offset32::new(0));
                    clif_args.push(addr);
                }
                sig_tys.push((slot.ty, slot.purpose));
            }
            abi::SlotKind::Chunk { owner, index } => {
                let base = body.operand(builder, args[owner]);
                let v = builder.ins().load(types::I64, memflags::notrap(), base, Offset32::new((index * 8) as i32));
                clif_args.push(v);
                sig_tys.push((slot.ty, slot.purpose));
            }
        }
    }
    // Variadic tail: each actual classified with continuing counters.
    let mut arg_idx = named;
    if arg_idx < args.len() {
        body.emit_call_tail(builder, callee_layout, args, &mut arg_idx, &mut clif_args, &mut sig_tys);
    }
    let mut sig = Signature::new(body.target.call_conv());
    for (ty, purpose) in &sig_tys {
        sig.params.push(AbiParam {
            value_type: *ty,
            purpose: *purpose,
            extension: ArgumentExtension::None,
        });
    }
    if let abi::RetLoc::Regs(pieces) = &callee_layout.ret {
        // A foreign `long double` return arrives in st(0): no Clif
        // return to declare (fetched through getret at the call).
        if !foreign_f80_ret {
            for (_, ty) in pieces {
                sig.returns.push(AbiParam::new(*ty));
            }
        }
    }
    // Direct calls when the declared signature matches the call-site
    // signature: non-variadic internal functions and non-variadic
    // externs (whose declarations use the same layout). Variadic calls
    // and foreign `long double` returns build a per-call signature and
    // go indirect.
    let indirect = variadic || foreign_f80_ret || matches!(target, CallTarget::Indirect { .. });
    if !indirect {
        let clif_id = match target {
            CallTarget::Internal(f) => func_ids[&f],
            CallTarget::External(e) => extern_ids[&e],
            CallTarget::Indirect { .. } => panic!("indirect call marked direct"),
        };
        let funcref: FuncRef = clif.declare_func_in_func(clif_id, builder.func);
        let inst = builder.ins().call(funcref, &clif_args);
        return body.map_call_results(builder, inst, &callee_layout.ret, sret_temp, result_val);
    }
    let addr = match target {
        CallTarget::Internal(f) => {
            let clif_id = func_ids[&f];
            let funcref = clif.declare_func_in_func(clif_id, builder.func);
            builder.ins().func_addr(types::I64, funcref)
        }
        CallTarget::External(e) => {
            let clif_id = extern_ids[&e];
            let funcref = clif.declare_func_in_func(clif_id, builder.func);
            builder.ins().func_addr(types::I64, funcref)
        }
        CallTarget::Indirect { addr, .. } => addr,
    };
    // x86-64 SysV foreign variadic calls need `%al` set: route through
    // the runtime trampoline with the callee's own signature. Win64 has
    // no such register and calls its variadics directly.
    if foreign && variadic && body.target.arch == Arch::X86_64 && body.target.os != Os::Windows {
        let rt = rt;
        let va_target = rt.va_target.expect("va_target not declared");
        let jmp_va = rt.jmp_va.expect("jmp_va not declared");
        let gv = clif.declare_data_in_func(va_target, builder.func);
        let tptr = builder.ins().tls_value(types::I64, gv);
        builder.ins().store(MemFlagsData::trusted(), addr, tptr, Offset32::new(0));
        let jump = clif.declare_func_in_func(jmp_va, builder.func);
        let jump_addr = builder.ins().func_addr(types::I64, jump);
        let sigref = builder.import_signature(sig);
        let inst = builder.ins().call_indirect(sigref, jump_addr, &clif_args);
        if foreign_f80_ret {
            return Some(foreign_f80_result(builder, clif, rt));
        }
        return body.map_call_results(builder, inst, &callee_layout.ret, sret_temp, result_val);
    }
    let sigref = builder.import_signature(sig);
    let inst = builder.ins().call_indirect(sigref, addr, &clif_args);
    if foreign_f80_ret {
        return Some(foreign_f80_result(builder, clif, rt));
    }
    body.map_call_results(builder, inst, &callee_layout.ret, sret_temp, result_val)
}

/// Fetch a foreign `long double` result from the live `st(0)` into a
/// fresh slot and return its carrier load. Emitted immediately after
/// the foreign call: nothing in Cranelift output touches x87, and a
/// call itself does not, so the value survives intact; `fstpt` pops
/// exactly the one value the callee pushed.
pub(crate) fn foreign_f80_result<M: Module>(builder: &mut FunctionBuilder, clif: &mut M, rt: &Rt) -> ClifValue {
    let f80 = rt.f80.as_ref().expect("foreign long double needs x86-64 helpers");
    let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
    let out = builder.ins().stack_addr(types::I64, slot, 0);
    let callee = clif.declare_func_in_func(f80.getret, builder.func);
    builder.ins().call(callee, &[out]);
    f80_result(builder, out)
}

/// One named argument piece for a call: scalars pass directly (the
/// checker made actual and formal widths agree); aggregate pieces load
/// from the argument slot.

/// Variadic tail emission: each actual classified with counters
/// continuing past the named prefix. Scalars cross directly
/// (RISC-V unnamed floats as integer bits); aggregates follow fixed
/// rules, with one pad-and-chunk group per stack-bound aggregate.
#[allow(clippy::too_many_arguments)]

/// Unnamed floating-point values that cross as integer bits: RISC-V's
/// variadic convention, and Win64's GP mirror of every unnamed
/// argument position.
pub(crate) fn unnamed_fp_in_int_regs(target: Target, mir_ty: &mir::Ty) -> bool {
    matches!(mir_ty, mir::Ty::Float(_)) && (target.arch == Arch::Riscv64 || (target.arch == Arch::X86_64 && target.os == Os::Windows))
}

/// RISC-V unnamed FP values cross as integer bits.
pub(crate) fn riscv_fp_bits(builder: &mut FunctionBuilder, v: ClifValue, ty: Type) -> ClifValue {
    if ty == types::F64 {
        builder.ins().bitcast(types::I64, MemFlagsData::new(), v)
    } else {
        let narrow = builder.ins().bitcast(types::I32, MemFlagsData::new(), v);
        builder.ins().uextend(types::I64, narrow)
    }
}

/// Push one register slot per aggregate piece: load the piece from the
/// aggregate's memory image and, on RISC-V, send FP bits through the
/// integer registers.
pub(crate) fn push_pieces(
    target: Target,
    body: &mut Body,
    builder: &mut FunctionBuilder,
    arg: mir::Value,
    pieces: &[(u64, Type)],
    clif_args: &mut Vec<ClifValue>,
    sig_tys: &mut Vec<(Type, ArgumentPurpose)>,
) {
    for (off, ty) in pieces {
        let base = body.operand(builder, arg);
        let v = builder.ins().load(*ty, memflags::notrap(), base, Offset32::new(*off as i32));
        let (v, ty) = if target.arch == Arch::Riscv64 && ty.is_float() {
            (riscv_fp_bits(builder, v, *ty), types::I64)
        } else {
            (v, *ty)
        };
        clif_args.push(v);
        sig_tys.push((ty, ArgumentPurpose::Normal));
    }
}

/// Reinterpret float bits as an integer of the same width.
pub(crate) fn float_bits_to_int(builder: &mut FunctionBuilder, v: ClifValue, mir_ty: &mir::Ty) -> ClifValue {
    match mir_ty {
        mir::Ty::Float(mir::FloatKind::F32) => {
            let ext = builder.ins().fpromote(types::F64, v);
            builder.ins().bitcast(types::I64, MemFlagsData::new(), ext)
        }
        _ => builder.ins().bitcast(types::I64, MemFlagsData::new(), v),
    }
}

pub(crate) enum TailLoc {
    Scalar(Type),
    Pieces(Vec<(u64, Type)>),
    Stack {
        bytes: u64,
    },
    /// RISC-V stack-bound aggregate: the callee's walk reads one slot
    /// per piece, not the packed memory image, so the pieces go after
    /// the register padding instead of raw chunks.
    StackPieces(Vec<(u64, Type)>),
    /// By-reference aggregate: pass its address.
    Pointer,
}

/// Classify one variadic-tail actual with continuing counters. The
/// flag reports a 16-alignment even-skip, which the caller must
/// materialize as a dummy register argument.
pub(crate) fn tail_classify(target: Target, mir_ty: &mir::Ty, counters: &mut abi::Counters) -> (bool, TailLoc) {
    let (even_pad, loc) = abi::classify_next(target, mir_ty, counters);
    let tail = match loc {
        abi::ParamLoc::Regs(pieces) => {
            if pieces.len() == 1 && !mir_ty.is_aggregate() {
                TailLoc::Scalar(pieces[0].1)
            } else {
                TailLoc::Pieces(pieces)
            }
        }
        abi::ParamLoc::Stack { bytes } => match (target.arch, abi::aggregate_pieces(target, mir_ty)) {
            (Arch::Riscv64, Some(pieces)) if mir_ty.is_aggregate() => TailLoc::StackPieces(pieces),
            _ => TailLoc::Stack { bytes },
        },
        abi::ParamLoc::Pointer => TailLoc::Pointer,
    };
    (even_pad, tail)
}

/// Pad both register files to full with zero dummies.
pub(crate) fn pad_to_full(target: Target, builder: &mut FunctionBuilder, gp: &mut usize, fp: &mut usize, clif_args: &mut Vec<ClifValue>, sig_tys: &mut Vec<(Type, ArgumentPurpose)>) {
    while *gp < target.gp_regs() {
        *gp += 1;
        let zero = builder.ins().iconst(types::I64, 0);
        clif_args.push(zero);
        sig_tys.push((types::I64, ArgumentPurpose::Normal));
    }
    while *fp < target.fp_regs() {
        *fp += 1;
        let zero = builder.ins().f64const(0.0);
        clif_args.push(zero);
        sig_tys.push((types::F64, ArgumentPurpose::Normal));
    }
}

/// Rejoin split scalar pieces (128-bit pairs) into one value: concat
/// the integer lanes, then reinterpret to the value type.
pub(crate) fn join_values(builder: &mut FunctionBuilder, parts: &[(u64, ClifValue)], mir_ty: &mir::Ty) -> ClifValue {
    use cranelift_codegen::ir::InstBuilder;
    assert_eq!(parts.len(), 2, "only pairs split scalars");
    assert_eq!((parts[0].0, parts[1].0), (0, 8));
    let wide = builder.ins().iconcat(parts[0].1, parts[1].1);
    let dst = clif_ty(mir_ty).expect("scalar join needs a Clif type");
    if dst == types::I128 {
        return wide;
    }
    builder.ins().bitcast(dst, MemFlagsData::new(), wide)
}

/// Split one scalar value into integer lanes for multi-slot emission.
pub(crate) fn split_value(builder: &mut FunctionBuilder, v: ClifValue, from_ty: Type, count: usize) -> Vec<ClifValue> {
    use cranelift_codegen::ir::InstBuilder;
    assert_eq!(count, 2, "only pairs split scalars");
    let wide = if from_ty == types::I128 {
        v
    } else {
        builder.ins().bitcast(types::I128, MemFlagsData::new(), v)
    };
    let (lo, hi) = builder.ins().isplit(wide);
    vec![lo, hi]
}


impl<'m> Body<'m> {
pub(crate) fn call_piece(&mut self, builder: &mut FunctionBuilder, arg: mir::Value, offset: u64, slot_ty: Type) -> ClifValue {
    let mir_ty = self.func.values[arg].ty.clone();
    if mir_ty.is_aggregate() {
        let base = self.operand(builder, arg);
        return builder.ins().load(slot_ty, memflags::notrap(), base, Offset32::new(offset as i32));
    }
    let v = self.operand(builder, arg);
    if builder.func.dfg.value_type(v) == slot_ty {
        return v;
    }
    // Width drift between actual and formal (unprototyped edges):
    // resize integers, bitcast floats of equal size.
    let from = builder.func.dfg.value_type(v);
    if !from.is_float() && !slot_ty.is_float() {
        return resize_int(builder, v, from, slot_ty);
    }
    if from.bytes() == slot_ty.bytes() {
        return builder.ins().bitcast(slot_ty, MemFlagsData::new(), v);
    }
    panic!("call argument type mismatch");
}
pub(crate) fn emit_call_tail(
    &mut self,
    builder: &mut FunctionBuilder,
    callee_layout: &SigLayout,
    args: &[mir::Value],
    arg_idx: &mut usize,
    clif_args: &mut Vec<ClifValue>,
    sig_tys: &mut Vec<(Type, ArgumentPurpose)>,
) {
    let target = self.target;
    // Reseed counters from the named prefix both sides share.
    let mut counters = abi::Counters {
        gp: callee_layout.named_gp,
        fp: callee_layout.named_fp,
    };
    let mut padded = false;
    while *arg_idx < args.len() {
        let arg = args[*arg_idx];
        *arg_idx += 1;
        let mir_ty = self.func.values[arg].ty.clone();
        // C promotes sub-int and float varargs; the checker inserts
        // those conversions for C sources (documented precondition for
        // hand-built MIR: pre-promoted actuals).
        if unnamed_fp_in_int_regs(target, &mir_ty) {
            // Unnamed FP travels in integer registers on RISC-V: one
            // for float/double bits, two for quad. Win64 mirrors every
            // unnamed argument into its position's GP register, so the
            // bits ride there too (the XMM copy is never read back).
            let v = self.operand(builder, arg);
            if mir_ty.size_bytes() > 8 {
                // 16-byte values start at an even register (conformant
                // writers pad odd positions; our reader skips the same
                // way, and so does musl): pad the odd slot so positions
                // line up instead of shifting everything after by one.
                if counters.gp % 2 == 1 {
                    counters.gp += 1;
                    let zero = builder.ins().iconst(types::I64, 0);
                    clif_args.push(zero);
                    sig_tys.push((types::I64, ArgumentPurpose::Normal));
                }
                let wide = builder.ins().bitcast(types::I128, MemFlagsData::new(), v);
                let (lo, hi) = builder.ins().isplit(wide);
                counters.gp += 2;
                clif_args.push(lo);
                clif_args.push(hi);
                sig_tys.push((types::I64, ArgumentPurpose::Normal));
                sig_tys.push((types::I64, ArgumentPurpose::Normal));
                continue;
            }
            let bits = float_bits_to_int(builder, v, &mir_ty);
            counters.gp += 1;
            clif_args.push(bits);
            sig_tys.push((types::I64, ArgumentPurpose::Normal));
            continue;
        }
        let (even_pad, loc) = tail_classify(target, &mir_ty, &mut counters);
        if even_pad {
            // 16-aligned composite at an odd GP slot: consume the
            // skipped register with a dummy so the actual layout
            // matches the counters (the callee rounds its cursor the
            // same way).
            let zero = builder.ins().iconst(types::I64, 0);
            clif_args.push(zero);
            sig_tys.push((types::I64, ArgumentPurpose::Normal));
        }
        match loc {
            TailLoc::Scalar(ty) => {
                let mut v = self.operand(builder, arg);
                // Sub-int actuals widen to int for the register file.
                let from = builder.func.dfg.value_type(v);
                if !from.is_float() && from.bytes() < 4 && ty.bytes() == 4 {
                    v = builder.ins().uextend(ty, v);
                }
                clif_args.push(v);
                sig_tys.push((ty, ArgumentPurpose::Normal));
            }
            TailLoc::Pieces(pieces) => {
                push_pieces(target, self, builder, arg, &pieces, clif_args, sig_tys);
            }
            TailLoc::StackPieces(pieces) => {
                if !padded {
                    pad_to_full(target, builder, &mut counters.gp, &mut counters.fp, clif_args, sig_tys);
                    padded = true;
                }
                push_pieces(target, self, builder, arg, &pieces, clif_args, sig_tys);
            }
            TailLoc::Stack { bytes } => {
                if target.arch == Arch::X86_64 {
                    // Stack-bound scalars (x86-64 `long double`) spill
                    // to a temp: `StructArgument` takes the address,
                    // like the named path. Aggregates already are
                    // addresses.
                    let base = if mir_ty.is_aggregate() {
                        self.operand(builder, arg)
                    } else {
                        let v = self.operand(builder, arg);
                        let slot = stack_slot(builder, &mir_ty);
                        let addr = builder.ins().stack_addr(types::I64, slot, 0);
                        builder.ins().store(memflags::trusted(), v, addr, Offset32::new(0));
                        addr
                    };
                    clif_args.push(base);
                    sig_tys.push((types::I64, ArgumentPurpose::StructArgument(bytes as u32)));
                } else {
                    if !padded {
                        pad_to_full(target, builder, &mut counters.gp, &mut counters.fp, clif_args, sig_tys);
                        padded = true;
                    }
                    let base = self.operand(builder, arg);
                    for k in 0..bytes.div_ceil(8) {
                        let v = builder.ins().load(types::I64, memflags::notrap(), base, Offset32::new((k * 8) as i32));
                        clif_args.push(v);
                        sig_tys.push((types::I64, ArgumentPurpose::Normal));
                    }
                }
            }
            TailLoc::Pointer => {
                let base = self.operand(builder, arg);
                clif_args.push(base);
                sig_tys.push((types::I64, ArgumentPurpose::Normal));
            }
        }
    }
}

/// Map call results back: scalars directly, register aggregates
/// reassembled into a fresh slot, sret through its temp slot.
pub(crate) fn map_call_results(
    &mut self,
    builder: &mut FunctionBuilder,
    inst: cranelift_codegen::ir::Inst,
    ret: &abi::RetLoc,
    sret_temp: Option<ClifValue>,
    result_val: Option<mir::Value>,
) -> Option<ClifValue> {
    let results = builder.inst_results(inst).to_vec();
    match ret {
        abi::RetLoc::Void => {
            // A zero-sized aggregate result still exists as a MIR value
            // (its uses are 0-byte copies and address math); hand out a
            // fresh dummy address. Void calls carry no result.
            let size = result_val.map(|v| self.func.values[v].ty.size_bytes())?;
            if size != 0 {
                return None;
            }
            let slot = stack_slot(builder, &mir::Ty::int(8));
            Some(builder.ins().stack_addr(types::I64, slot, 0))
        }
        abi::RetLoc::Sret => sret_temp,
        abi::RetLoc::Regs(pieces) => {
            // Aggregates are addresses everywhere in MIR: a register
            // result reassembles into a fresh slot, even when it fits
            // one piece (a bare small-struct call result stored to a
            // local segfaulted before this). Only true scalars return
            // the value itself.
            let scalar = result_val.map(|v| !self.func.values[v].ty.is_aggregate()).unwrap_or(false);
            if pieces.len() == 1 && scalar {
                let v = results.into_iter().next()?;
                // The callee widened the result to its slot width
                // (`Return`); hand the self back a declared-width value,
                // like entry params get on arrival.
                let mir_val = result_val.expect("scalar result carries a MIR value");
                let mir_ty = self.func.values[mir_val].ty.clone();
                return Some(narrow_to_declared(builder, v, &mir_ty));
            }
            if scalar {
                // Multi-piece scalar (128-bit pair) rejoins to one value.
                assert_eq!(results.len(), 2);
                let wide = builder.ins().iconcat(results[0], results[1]);
                let dst = result_val
                    .map(|v| clif_ty(&self.func.values[v].ty))
                    .unwrap_or(None)
                    .expect("scalar result needs a Clif type");
                if dst == types::I128 {
                    return Some(wide);
                }
                return Some(builder.ins().bitcast(dst, MemFlagsData::new(), wide));
            }
            let size = pieces.iter().map(|(off, ty)| off + ty.bytes() as u64).max().unwrap_or(0);
            let slot = stack_slot(
                builder,
                &mir::Ty::Array {
                    elem: Box::new(mir::Ty::int(8)),
                    len: size,
                },
            );
            let addr = builder.ins().stack_addr(types::I64, slot, 0);
            for ((off, _ty), v) in pieces.iter().zip(results) {
                builder.ins().stack_store(types::I64, v, slot, *off as i32);
            }
            let _ = result_val;
            Some(addr)
        }
    }
}
}
