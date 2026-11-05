
use cranelift_codegen::ir::immediates::{Ieee128, Offset32};
use cranelift_codegen::ir::types::{self, Type};
use cranelift_codegen::ir::{
    ConstantData, InstBuilder, MemFlagsData, StackSlot, StackSlotData,
    StackSlotKind,
};
use cranelift_codegen::ir::Value as ClifValue;
use cranelift_frontend::FunctionBuilder;
use cranelift_module::{FuncId, Module};

use glcc_mir as mir;

use crate::memflags;
use crate::ty::clif_ty;
use super::*;

/// Operand value used as a machine address: a 32-bit pointer widens to
/// the target's 64-bit address width (sign-extended, MSVC's default for
/// `__ptr32`). Ordinary addresses pass through untouched.

/// One volatile scalar load. C requires every volatile access to touch
/// memory; this Cranelift (0.136) has no volatile memory flag, and its
/// alias analysis happily forwards a plain load from a dominating store
/// (or replaces it with an equal earlier load). Atomic load/store are
/// opaque to that analysis — it treats them as fences — so they are what
/// volatile accesses lower to. The ordering is stronger than C's
/// volatile asks for, which is a legal implementation of it.
///
/// 16-byte scalars (F80/F128 carriers, I128) have no atomic lowering on
/// every supported target, so those keep a plain access fenced on both
/// sides: no store reaches across the fence to feed the load and no
/// later load reaches back over it.
pub(crate) fn volatile_load(builder: &mut FunctionBuilder, ty: Type, addr: ClifValue) -> ClifValue {
    if let Some(int_ty) = atomic_ty(ty) {
        let raw = builder.ins().atomic_load(int_ty, MemFlagsData::new(), addr);
        if int_ty == ty { raw } else { builder.ins().bitcast(ty, MemFlagsData::new(), raw) }
    } else {
        builder.ins().fence();
        let value = builder.ins().load(ty, MemFlagsData::new(), addr, Offset32::new(0));
        builder.ins().fence();
        value
    }
}

/// The store counterpart of [`volatile_load`]: atomic for integer-like
/// scalars up to 64 bits (32/64-bit floats bitcast through an integer),
/// fenced plain store otherwise.
pub(crate) fn volatile_store(builder: &mut FunctionBuilder, ty: Type, val: ClifValue, addr: ClifValue) {
    if let Some(int_ty) = atomic_ty(ty) {
        let raw = if int_ty == ty { val } else { builder.ins().bitcast(int_ty, MemFlagsData::new(), val) };
        builder.ins().atomic_store(MemFlagsData::new(), raw, addr);
    } else {
        builder.ins().fence();
        builder.ins().store(MemFlagsData::new(), val, addr, Offset32::new(0));
        builder.ins().fence();
    }
}

/// The integer type an atomic Clif op can carry for a volatile scalar:
/// integers and pointers pass through, 32/64-bit floats go through an
/// integer of the same size. 128-bit scalars return `None` (no atomic
/// lowering is available on every supported target).
pub(crate) fn atomic_ty(ty: Type) -> Option<Type> {
    match ty {
        types::I8 | types::I16 | types::I32 | types::I64 => Some(ty),
        types::F32 => Some(types::I32),
        types::F64 => Some(types::I64),
        _ => None,
    }
}

/// One atomic scalar load (`_Atomic` reads, struct-field extraction).
/// [`atomic_ty`] carries 32/64-bit floats through an integer bitcast
/// (this Cranelift only takes integers in `atomic_load`); every other
/// type loads directly.
pub(crate) fn atomic_load_value(builder: &mut FunctionBuilder, ty: Type, addr: ClifValue) -> ClifValue {
    if let Some(int_ty) = atomic_ty(ty) {
        let raw = builder.ins().atomic_load(int_ty, MemFlagsData::new(), addr);
        if int_ty == ty { raw } else { builder.ins().bitcast(ty, MemFlagsData::new(), raw) }
    } else {
        builder.ins().atomic_load(ty, MemFlagsData::new(), addr)
    }
}

/// The store counterpart of [`atomic_load_value`].
pub(crate) fn atomic_store_value(builder: &mut FunctionBuilder, ty: Type, val: ClifValue, addr: ClifValue) {
    if let Some(int_ty) = atomic_ty(ty) {
        let raw = if int_ty == ty { val } else { builder.ins().bitcast(int_ty, MemFlagsData::new(), val) };
        builder.ins().atomic_store(MemFlagsData::new(), raw, addr);
    } else {
        builder.ins().atomic_store(MemFlagsData::new(), val, addr);
    }
}

/// A compile-time constant as a Clif value. Aggregate-typed values have
/// no constant form besides `undef`, which materializes as a fresh,
/// uninitialized slot whose address is returned.
pub(crate) fn materialize_const(builder: &mut FunctionBuilder, c: &mir::Const, ty: &mir::Ty) -> ClifValue {
    if ty.is_aggregate() {
        let slot = stack_slot(builder, ty);
        let addr = builder.ins().stack_addr(types::I64, slot, 0);
        match c {
            mir::Const::Bytes(content) => store_const_bytes(builder, addr, content),
            // A wide integer constant is a sign-extended scalar pattern
            // in a byte image at the type's own width.
            mir::Const::Int { value, .. } if ty.is_wide_int() => {
                let len = ty.size_bytes() as usize;
                let mut image = vec![0u8; len];
                let raw = value.to_le_bytes();
                let copy = raw.len().min(len);
                image[..copy].copy_from_slice(&raw[..copy]);
                if *value < 0 {
                    image[copy..].fill(0xff);
                }
                store_const_bytes(builder, addr, &image);
            }
            // Fresh slots hold garbage: zeroinit must clear explicitly.
            _ => zero_range(builder, addr, ty.size_bytes()),
        }
        return addr;
    }
    match c {
        mir::Const::Zero | mir::Const::Bytes(_) => {
            panic!("zero/bytes constants need an aggregate type")
        }
        mir::Const::Int { bits, value } => {
            let clif_ty = clif_ty(ty).expect("int constant needs a scalar type");
            if *bits > 64 {
                let raw = *value as u128;
                let lo = builder.ins().iconst(types::I64, (raw as u64) as i64);
                let hi = builder.ins().iconst(types::I64, (raw >> 64) as u64 as i64);
                builder.ins().iconcat(lo, hi)
            } else {
                builder.ins().iconst(clif_ty, *value as i64)
            }
        }
        mir::Const::Float { kind, bits } => match kind {
            mir::FloatKind::F32 => builder.ins().f32const(f32::from_bits(*bits as u32)),
            mir::FloatKind::F64 => builder.ins().f64const(f64::from_bits(*bits as u64)),
            mir::FloatKind::F80 | mir::FloatKind::F128 => {
                let data: ConstantData = Ieee128::with_bits(*bits).into();
                let handle = builder.func.dfg.constants.insert(data);
                builder.ins().f128const(handle)
            }
        },
        // Null is the zero address; undef lowers to zero. A producer
        // must not rely on undef bits either way.
        mir::Const::Null | mir::Const::Undef => {
            let clif_ty = clif_ty(ty).expect("null/undef needs a scalar type");
            zero_value(builder, clif_ty)
        }
    }
}

/// Zero bits of a scalar type. `iconst` only covers 8..64-bit
/// integers, so float and 128-bit forms need their own constants.
pub(crate) fn zero_value(builder: &mut FunctionBuilder, ty: Type) -> ClifValue {
    match ty {
        types::F32 => builder.ins().f32const(0.0),
        types::F64 => builder.ins().f64const(0.0),
        types::F128 => {
            let data: ConstantData = Ieee128::with_bits(0).into();
            let handle = builder.func.dfg.constants.insert(data);
            builder.ins().f128const(handle)
        }
        types::I128 => {
            let lo = builder.ins().iconst(types::I64, 0);
            let hi = builder.ins().iconst(types::I64, 0);
            builder.ins().iconcat(lo, hi)
        }
        ty if ty.is_vector() => {
            // Vector undef: zero the scalar element and splat.
            let scalar = zero_value(builder, ty.lane_type());
            builder.ins().splat(ty, scalar)
        }
        _ => builder.ins().iconst(ty, 0),
    }
}

pub(crate) fn stack_slot(builder: &mut FunctionBuilder, ty: &mir::Ty) -> StackSlot {
    // Zero-sized objects (empty structs, `char x[0]`) still need a
    // distinct address, so they get a byte; Cranelift rejects 0-byte
    // slots.
    let size = ty.size_bytes().max(1);
    if size > u32::MAX as u64 {
        panic!("cannot materialize stack slot of {size} bytes");
    }
    let align = mir::ty::ty_align(ty).max(1);
    debug_assert!(align.is_power_of_two(), "non-power-of-two alignment {align}");
    let align_shift = align.trailing_zeros() as u8;
    builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, size as u32, align_shift))
}

/// Store a static byte image at an address, chunked like `memcpy`.
pub(crate) fn store_const_bytes(builder: &mut FunctionBuilder, dst: ClifValue, content: &[u8]) {
    let mut off = 0usize;
    for (chunk_ty, chunk) in [(types::I64, 8), (types::I32, 4), (types::I16, 2), (types::I8, 1)] {
        while off + chunk <= content.len() {
            let mut buf = [0u8; 8];
            buf[..chunk].copy_from_slice(&content[off..off + chunk]);
            let v = builder.ins().iconst(chunk_ty, i64::from_le_bytes(buf));
            builder.ins().store(memflags::trusted(), v, dst, Offset32::new(off as i32));
            off += chunk;
        }
    }
}

/// Clear a range with 8-byte splats plus byte remainder.
pub(crate) fn zero_range(builder: &mut FunctionBuilder, dst: ClifValue, len: u64) {
    let splat = builder.ins().iconst(types::I64, 0);
    let words = len / 8;
    for i in 0..words {
        builder.ins().store(memflags::trusted(), splat, dst, Offset32::new((i * 8) as i32));
    }
    let zero_byte = builder.ins().iconst(types::I8, 0);
    for i in (words * 8)..len {
        builder.ins().store(memflags::trusted(), zero_byte, dst, Offset32::new(i as i32));
    }
}

pub(crate) fn emit_memcpy(builder: &mut FunctionBuilder, dst: ClifValue, src: ClifValue, len: u64) {
    let mut off = 0u64;
    for (chunk_ty, chunk) in [(types::F128, 16), (types::I64, 8), (types::I32, 4), (types::I16, 2), (types::I8, 1)] {
        while off + chunk <= len {
            let v = builder.ins().load(chunk_ty, memflags::notrap(), src, Offset32::new(off as i32));
            builder.ins().store(memflags::notrap(), v, dst, Offset32::new(off as i32));
            off += chunk;
        }
    }
}

/// Whether a MIR value is an x86-64 80-bit extended float: arithmetic
/// routes through the runtime helpers, everything else through
/// `fstp`/`fucomip` helpers or generic moves. (`F128` rides the same
/// carriers: its arithmetic goes through the quad helpers — vendored
/// soft-float on x86-64, trampolines on quad targets — and every other
/// `F128` op inlines through `quad.rs`).

/// Spill an `F80` scalar (an `F128` carrier value) to a fresh 16-byte
/// slot and return its address for the out-pointer convention.

/// Load the `F128` carrier back from a 16-byte helper out-slot.
pub(crate) fn f80_result(builder: &mut FunctionBuilder, out: ClifValue) -> ClifValue {
    builder.ins().load(types::F128, memflags::trusted(), out, Offset32::new(0))
}

/// Widen an `F80` value to IEEE-quad through the vendored runtime
/// (`__extendxftf2`, exact like the `fld` widenings): the source
/// crosses by value on the stack, the quad result returns in XMM as
/// an `F128` carrier directly.

/// Narrow a quad carrier to `F80` through the vendored runtime (`__glcc_trunctfxf2`,
/// single rounding like the `fstp` narrowings): the carrier rides
/// XMM, the extended result arrives in st(0) and is captured exactly
/// like a foreign `long double` return.

/// One `F80` comparison through its helper, narrowed to the `Bool`
/// (`I8`) result the `Fcmp` instruction carries.

/// Width in bits of an integer MIR type.
pub(crate) fn mir_int_bits(ty: &mir::Ty) -> u32 {
    match ty {
        mir::Ty::Bool => 1,
        mir::Ty::Int { bits } => *bits,
        _ => panic!("not an integer MIR type"),
    }
}

/// Clif type of an instruction result, which must be scalar. Aggregates
/// never reach here: they resolve to addresses along dedicated paths.
pub(crate) fn result_clif(ty: &Option<mir::Ty>) -> Type {
    match ty {
        Some(mir_ty) => clif_ty(mir_ty).expect("result needs a scalar Clif type"),
        None => panic!("instruction needs a result type"),
    }
}

/// Zero- or sign-agnostic width change between integer Clif types.
/// Addresses are plain bit patterns, so extension is always zero fill.
pub(crate) fn resize_int(builder: &mut FunctionBuilder, v: ClifValue, from: Type, to: Type) -> ClifValue {
    if from == to {
        return v;
    }
    if from.bytes() < to.bytes() {
        builder.ins().uextend(to, v)
    } else {
        builder.ins().ireduce(to, v)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum IntConvert {
    Trunc,
    Zext,
    Sext,
}

/// An integer conversion, respecting the MIR bit widths inside one Clif
/// type. `Bool` and sub-byte `Int`s all occupy `i8`, so Cranelift sees no
/// width change for e.g. `zext bool -> int8` even though the conversion
/// is real; extend or reduce those on the MIR widths instead.
pub(crate) fn convert_int(builder: &mut FunctionBuilder, v: ClifValue, to: Type, src_bits: u32, dst_bits: u32, convert: IntConvert) -> ClifValue {
    let from = builder.func.dfg.value_type(v);
    let from_bits = from.bits();
    // Sign-extend a sub-width source (`Bool` or a small `Int`) to its
    // Clif width first: Cranelift would otherwise sign-extend the
    // carrier byte, turning a true boolean into 1 instead of -1.
    let v = match convert {
        IntConvert::Sext if src_bits > 0 && src_bits < from_bits => {
            let shift = builder.ins().iconst(from, i64::from(from_bits - src_bits));
            let up = builder.ins().ishl(v, shift);
            builder.ins().sshr(up, shift)
        }
        _ => v,
    };
    let v = if from != to {
        match convert {
            IntConvert::Trunc => builder.ins().ireduce(to, v),
            IntConvert::Zext => builder.ins().uextend(to, v),
            IntConvert::Sext => builder.ins().sextend(to, v),
        }
    } else {
        v
    };
    let to_bits = to.bits();
    match convert {
        // The source is already canonical for its declared width.
        IntConvert::Zext => v,
        IntConvert::Trunc if dst_bits > 0 && dst_bits < to_bits && to_bits <= 64 => {
            let mask = builder.ins().iconst(to, (1i64 << dst_bits) - 1);
            builder.ins().band(v, mask)
        }
        _ => v,
    }
}

/// Narrow an ABI cover-width value back to its declared width, the
/// inverse of `Return`'s widening. Narrow scalars cross at register
/// width (`int_cover`: W-register covers on AAPCS64/RISC-V, exact
/// widths on x86-64), while bodies carry declared widths so casts and
/// spills lower directly. Non-integer types are passed through: their
/// covers already match.
pub(crate) fn narrow_to_declared(builder: &mut FunctionBuilder, v: ClifValue, mir_ty: &mir::Ty) -> ClifValue {
    if !matches!(mir_ty, mir::Ty::Bool | mir::Ty::Int { .. }) {
        return v;
    }
    let Some(decl) = clif_ty(mir_ty) else {
        return v;
    };
    let got = builder.func.dfg.value_type(v);
    if got == decl {
        return v;
    }
    resize_int(builder, v, got, decl)
}


impl<'m> Body<'m> {
pub(crate) fn f80_cmp<M: Module>(&mut self, builder: &mut FunctionBuilder, clif: &mut M, helper: FuncId, a: mir::Value, b: mir::Value) -> ClifValue {
    let a_addr = self.f80_addr(builder, a);
    let b_addr = self.f80_addr(builder, b);
    let callee = clif.declare_func_in_func(helper, builder.func);
    let inst = builder.ins().call(callee, &[a_addr, b_addr]);
    let r = builder.inst_results(inst)[0];
    builder.ins().ireduce(types::I8, r)
}

pub(crate) fn quad_trunc<M: Module>(&mut self, builder: &mut FunctionBuilder, clif: &mut M, rt: &Rt, helper: FuncId, a: mir::Value) -> ClifValue {
    let x = self.operand(builder, a);
    let callee = clif.declare_func_in_func(helper, builder.func);
    builder.ins().call(callee, &[x]);
    foreign_f80_result(builder, clif, rt)
}

pub(crate) fn f80_extend<M: Module>(&mut self, builder: &mut FunctionBuilder, clif: &mut M, helper: FuncId, a: mir::Value) -> ClifValue {
    let a_addr = self.f80_addr(builder, a);
    let callee = clif.declare_func_in_func(helper, builder.func);
    let inst = builder.ins().call(callee, &[a_addr]);
    builder.inst_results(inst)[0]
}

pub(crate) fn addr_operand(&mut self, builder: &mut FunctionBuilder, v: mir::Value) -> ClifValue {
    let val = self.operand(builder, v);
    // Only a 32-bit address widens; other address operands (a SysV
    // `va_list` tag) have no scalar Clif type and pass through.
    let narrow = match &self.func.values[v].ty {
        mir::Ty::Ptr { bits, .. } => *bits <= 32,
        mir::Ty::Int { bits } => (17..=32).contains(bits),
        _ => false,
    };
    if narrow { builder.ins().sextend(types::I64, val) } else { val }
}
pub(crate) fn is_f80(&self, v: mir::Value) -> bool {
    matches!(self.func.values[v].ty, mir::Ty::Float(mir::FloatKind::F80))
}
pub(crate) fn f80_addr(&mut self, builder: &mut FunctionBuilder, v: mir::Value) -> ClifValue {
    let x = self.operand(builder, v);
    let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
    builder.ins().stack_store(types::I64, x, slot, 0);
    builder.ins().stack_addr(types::I64, slot, 0)
}

pub(crate) fn operand(&mut self, builder: &mut FunctionBuilder, mir_value: mir::Value) -> ClifValue {
    if let Some(resolved) = self.values.get(&mir_value) {
        return *resolved;
    }
    let data = &self.func.values[mir_value];
    let mir::Def::Const(c) = &data.def else {
        panic!("MIR value used before definition: {:?} of type {:?} in {}", data.def, data.ty, self.func.name);
    };
    let c = c.clone();
    let ty = data.ty.clone();
    let block = self.current.expect("operand outside a block");
    if let Some(&cached) = self.consts.get(&(block, mir_value)) {
        return cached;
    }
    let lowered = materialize_const(builder, &c, &ty);
    self.consts.insert((block, mir_value), lowered);
    lowered
}
pub(crate) fn emit_copy(&self, builder: &mut FunctionBuilder, dst: ClifValue, src: ClifValue, len: u64) {
    if len >= LIBC_COPY_THRESHOLD
        && let Some(memcpy) = self.memcpy {
            let len = builder.ins().iconst(types::I64, len as i64);
            builder.ins().call(memcpy, &[dst, src, len]);
            return;
        }
    emit_memcpy(builder, dst, src, len);
}
/// Whether a MIR value is an IEEE-quad float: AArch64/RISC-V `long
/// double` and `__float128` everywhere.
pub(crate) fn is_f128(&self, v: mir::Value) -> bool {
    matches!(self.func.values[v].ty, mir::Ty::Float(mir::FloatKind::F128))
}
/// One `F80` binary operation through its helper: both sides cross as
/// addresses, the result comes back through an out-pointer temp as an
/// `F128` carrier. Rounds once, like the conversions. Also serves the
/// quad trampolines, which share the out-pointer convention exactly.
pub(crate) fn f80_binop<M: Module>(&mut self, builder: &mut FunctionBuilder, clif: &mut M, helper: FuncId, a: mir::Value, b: mir::Value) -> ClifValue {
    let a_addr = self.f80_addr(builder, a);
    let b_addr = self.f80_addr(builder, b);
    let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
    let out = builder.ins().stack_addr(types::I64, slot, 0);
    let callee = clif.declare_func_in_func(helper, builder.func);
    builder.ins().call(callee, &[out, a_addr, b_addr]);
    f80_result(builder, out)
}
/// One x86-64 `__float128` binary operation through the vendored runtime: both
/// carriers pass in XMM registers and the result comes back the same
/// way, so unlike [`f80_binop`] no memory round-trip is needed.
pub(crate) fn f128_direct_binop<M: Module>(&mut self, builder: &mut FunctionBuilder, clif: &mut M, helper: FuncId, a: mir::Value, b: mir::Value) -> ClifValue {
    let x = self.operand(builder, a);
    let y = self.operand(builder, b);
    let callee = clif.declare_func_in_func(helper, builder.func);
    let inst = builder.ins().call(callee, &[x, y]);
    builder.inst_results(inst)[0]
}
pub(crate) fn is_undef(&self, mir_value: mir::Value) -> bool {
    matches!(&self.func.values[mir_value].def, mir::Def::Const(mir::Const::Undef))
}
pub(crate) fn emit_memcpy_offset(&self, builder: &mut FunctionBuilder, dst_base: ClifValue, dst_off: u64, src: ClifValue, len: u64) {
    let dst = if dst_off == 0 { dst_base } else { builder.ins().iadd_imm_u(dst_base, dst_off as i64) };
    self.emit_copy(builder, dst, src, len);
}
}

pub(crate) fn emit_memcpy_pub(builder: &mut FunctionBuilder, dst: ClifValue, src: ClifValue, len: u64) {
    emit_memcpy(builder, dst, src, len)
}
