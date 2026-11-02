//! IEEE-754 quad (`long double` on AArch64/RISC-V, `__float128`
//! everywhere) conversions, comparisons, and negation, inlined as
//! integer IR.
//!
//! Rule for what lives here vs the runtime (`rt/glccrt.c`, plus the
//! vendored `rt/softfloat` on x86-64): `+`/`-`/`*`/`/` call a
//! correctly-rounded quad implementation — the toolchain's `__*tf3`
//! through tiny hand-ASM adapters on AArch64/RISC-V (Cranelift names
//! no `F128` arithmetic, so the adapters are the only way to reach
//! it), the vendored `__glcc_*tf3` directly on x86-64, where the
//! carrier already crosses in XMM registers.
//! Everything else inlines
//! here as bit-exact integer sequences: widenings are exact, integer
//! conversions saturate like the platform (see below), narrowings
//! round half-even explicitly, and ordered comparisons never round —
//! so no live rounding or precision control can leak in.
//!
//! Quad layout is plain IEEE little-endian: `lo` holds fraction bits
//! 63:0, `hi` holds bits 127:64 (sign at 63, exponent at 62:48,
//! fraction-high at 47:0). There is no explicit integer bit, and all
//! 16 bytes are significant (no padding).
//!
//! Conversion semantics follow the platform (probed with zig cc +
//! qemu, not guessed): saturating truncation — in-range values
//! truncate toward zero, positive overflow (and NaN) saturate to the
//! type's maximum, negative overflow to the minimum (0 for unsigned);
//! sub-word integers narrow from the saturated 32-bit result.
//!
//! The (test-only) `ref_*` functions are pure-Rust oracles for the IR
//! builders beneath them (same constants, same order); the tests pin
//! them to probed vectors, and the qemu end-to-end suite checks the
//! builders against the toolchain.

use cranelift_codegen::ir::Value as ClifValue;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::immediates::Offset32;
use cranelift_codegen::ir::types::{self};
use cranelift_codegen::ir::{InstBuilder, StackSlotData, StackSlotKind};
use cranelift_frontend::FunctionBuilder;

use crate::memflags;

/// Quad exponent bias.
/// All-ones quad exponent: infinity and NaN.

/// Assemble halves into a u128.

/// Split a u128 into halves.

/// Pack sign/exp/fraction into halves.

/// Unpack halves into sign, raw exp, and 112-bit fraction.

/// Widen f64 bits to quad (exact; subnormals normalized, SNaN
/// quieted with payload shifted into place).

/// Widen f32 bits to quad (same rules).

/// Integer to quad: `v` is the full 64-bit pattern (callers widen
/// sub-word first), two's complement when signed. Every u64/i64
/// value is exactly representable (113-bit significand).

/// Integer to quad from a full 128-bit pattern (see
/// ref_int_to_quad): magnitudes past 2^113 round half-even, with
/// renormalization on carry (e.g. `u128::MAX` rounds up to 2^128).

/// Unbiased exponent, or `None` for zero/denormal (always
/// fraction-sized) — callers map those to zero directly.

/// Quad to signed 64: saturating truncation (in-range truncates
/// toward zero; positive overflow, +Inf, and NaN saturate to MAX;
/// negative overflow and -Inf to MIN; denormals are zero).

/// Quad to signed 32 (same rules at 32 bits).

/// Quad to unsigned 64: saturating (negatives and -Inf go to 0,
/// not to `u64::MAX` via wrapping).

/// Quad to unsigned 32 (same rules at 32 bits).

/// Quad to signed 128: saturating truncation (same rules at 128
/// bits; `-2^127` is exact through the saturation path, like the
/// 64-bit `-2^63`).
///
/// Note the boundary differs deliberately from the x87 path below:
/// positive `2^127` saturates here, while the vendored
/// `__glcc_fixxfti` (like clang's own lowering) yields the raw bit
/// pattern. Out-of-range float-to-int is UB in C; each flavor
/// matches its target toolchain instead of agreeing with the other.

/// Quad to unsigned 128 (same rules at 128 bits).

/// Whether a quad value is NaN.

/// Whether a quad value is a signed zero.

/// Ordered equality: NaN never equals, signed zeros compare equal.

/// Ordered less-than, by sign then magnitude.

/// Ceiling as halves: smallest integral value `>= x`. NaN, infinities,
/// signed zeros, and already-integral values return bit-identical;
/// `|x| < 1` maps to `1`/`-0` by sign; otherwise truncate the
/// fraction and increment positives when inexact (renormalizing on
/// carry, e.g. `ceil(1.5) == 2`).

/// Floor as halves: largest integral value `<= x`. Mirror of
/// ref_quad_ceil: `|x| < 1` maps to `+0`/`-1`; otherwise
/// truncate and decrement (increment the magnitude of) negatives
/// when inexact.

/// Round a 113-bit significand (with its GRS bits) to a narrower
/// float: `kept_bits` result bits (53 for double, 24 for single),
/// `drop` low bits discarded. Returns the rounded significand
/// (possibly carrying into one extra bit, which the caller
/// renormalizes).

/// Quad to f64 bits: round-half-even; every NaN forces the quiet
/// bit and truncates its payload, overflow to infinity, tiny to
/// subnormal/zero.

/// Quad to f32 bits (same rules at single precision).

/// An `I64` constant as its own statement (builder borrows never
/// nest: every statement below holds at most one builder borrow, so
/// constants and subexpressions are hoisted to `let`s first).
fn q64(builder: &mut FunctionBuilder, v: i64) -> ClifValue {
    builder.ins().iconst(types::I64, v)
}

/// An `I32` constant.
fn q32(builder: &mut FunctionBuilder, v: i32) -> ClifValue {
    builder.ins().iconst(types::I32, v as i64)
}

/// An `I8` boolean constant.
fn q8(builder: &mut FunctionBuilder, v: bool) -> ClifValue {
    builder.ins().iconst(types::I8, i64::from(v))
}

/// Store `(lo, hi)` halves to a fresh 16-byte slot and return the
/// `F128` carrier load: the value form every quad producer makes.
pub(crate) fn quad_store_carrier(builder: &mut FunctionBuilder, lo: ClifValue, hi: ClifValue) -> ClifValue {
    let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
    builder.ins().stack_store(types::I64, lo, slot, 0);
    builder.ins().stack_store(types::I64, hi, slot, 8);
    builder.ins().stack_load(types::I64, types::F128, slot, 0)
}

/// Spill an `F128` carrier value to a fresh 16-byte slot for
/// half-wise processing, returning its address.
pub(crate) fn quad_spill(builder: &mut FunctionBuilder, v: ClifValue) -> ClifValue {
    let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
    builder.ins().stack_store(types::I64, v, slot, 0);
    builder.ins().stack_addr(types::I64, slot, 0)
}

/// Load a quad address as `(lo, hi)` halves (the whole 16 bytes
/// are significant: no padding exists).
pub(crate) fn quad_load_halves(builder: &mut FunctionBuilder, addr: ClifValue) -> (ClifValue, ClifValue) {
    let lo = builder.ins().load(types::I64, memflags::trusted(), addr, Offset32::new(0));
    let hi = builder.ins().load(types::I64, memflags::trusted(), addr, Offset32::new(8));
    (lo, hi)
}

/// Split loaded halves into `(sign, exp15, flo, fhi48)` values.
fn quad_unpack(builder: &mut FunctionBuilder, lo: ClifValue, hi: ClifValue) -> (ClifValue, ClifValue, ClifValue, ClifValue) {
    let k63 = q64(builder, 63);
    let s63 = builder.ins().ushr(hi, k63);
    let k1 = q64(builder, 1);
    let sign = builder.ins().band(s63, k1);
    let k48 = q64(builder, 48);
    let s48 = builder.ins().ushr(hi, k48);
    let k7fff = q64(builder, 0x7fff);
    let exp = builder.ins().band(s48, k7fff);
    let kfmask = q64(builder, 0x0000_ffff_ffff_ffff);
    let fhi = builder.ins().band(hi, kfmask);
    (sign, exp, lo, fhi)
}

/// Unbiased exponent (`I32`) from a masked exp.
fn quad_exp_biased(builder: &mut FunctionBuilder, exp: ClifValue) -> ClifValue {
    let e32 = builder.ins().ireduce(types::I32, exp);
    let kbias = q32(builder, 16383);
    builder.ins().isub(e32, kbias)
}

/// Widen f64 bits to `(lo, hi)` halves (see [`ref_f64_to_quad`]).
pub(crate) fn quad_from_f64(builder: &mut FunctionBuilder, bits: ClifValue) -> (ClifValue, ClifValue) {
    let k63 = q64(builder, 63);
    let s63 = builder.ins().ushr(bits, k63);
    let k1 = q64(builder, 1);
    let sign = builder.ins().band(s63, k1);
    let k52 = q64(builder, 52);
    let s52 = builder.ins().ushr(bits, k52);
    let k7ff = q64(builder, 0x7ff);
    let exp64 = builder.ins().band(s52, k7ff);
    let kmask = q64(builder, 0x000f_ffff_ffff_ffff);
    let mant64 = builder.ins().band(bits, kmask);
    let k0 = q64(builder, 0);
    let is_special = builder.ins().icmp(IntCC::Equal, exp64, k7ff);
    let is_zero_sub = builder.ins().icmp(IntCC::Equal, exp64, k0);
    let mant_is_zero = builder.ins().icmp(IntCC::Equal, mant64, k0);
    let is_zero = builder.ins().band(is_zero_sub, mant_is_zero);
    // Normal: rebias; the 52-bit fraction shifts to bits 111:60,
    // so the low 64 hold bits 59:0 and the high 48 hold 111:64.
    let k15360 = q64(builder, 15360);
    let n_exp = builder.ins().iadd(exp64, k15360);
    let k60 = q64(builder, 60);
    let n_lo = builder.ins().ishl(mant64, k60);
    let k4 = q64(builder, 4);
    let n_hi = builder.ins().ushr(mant64, k4);
    // Subnormal: normalize, then place at bit 111.
    let lz = builder.ins().clz(mant64);
    let k15372 = q64(builder, 15372);
    let s_exp = builder.ins().isub(k15372, lz);
    let normed = builder.ins().ishl(mant64, lz);
    let k48 = q64(builder, 48);
    let s_lo = builder.ins().ishl(normed, k48);
    let k16 = q64(builder, 16);
    let s_hi_raw = builder.ins().ushr(normed, k16);
    let k47c = q64(builder, (1u64 << 47) as i64);
    let kneg1 = q64(builder, -1);
    let not47 = builder.ins().bxor(k47c, kneg1);
    let s_hi = builder.ins().band(s_hi_raw, not47);
    let zs_exp = builder.ins().select(is_zero, k0, s_exp);
    let zs_lo = builder.ins().select(is_zero, k0, s_lo);
    let zs_hi = builder.ins().select(is_zero, k0, s_hi);
    // NaN/Inf: payload shifts into place with its quiet bit
    // untouched (SNaN stays signaling); a zero payload is
    // infinity (fraction stays zero).
    let is_inf = builder.ins().icmp(IntCC::Equal, mant64, k0);
    let sp_lo = builder.ins().ishl(mant64, k60);
    let pay_hi = builder.ins().ushr(mant64, k4);
    let sp_hi = builder.ins().select(is_inf, k0, pay_hi);
    let sp_exp = q64(builder, 0x7fff);
    let z_exp = builder.ins().select(is_zero_sub, zs_exp, n_exp);
    let exp = builder.ins().select(is_special, sp_exp, z_exp);
    let z_lo = builder.ins().select(is_zero_sub, zs_lo, n_lo);
    let lo = builder.ins().select(is_special, sp_lo, z_lo);
    let z_hi = builder.ins().select(is_zero_sub, zs_hi, n_hi);
    let hi48 = builder.ins().select(is_special, sp_hi, z_hi);
    let k63s = q64(builder, 63);
    let sign_shifted = builder.ins().ishl(sign, k63s);
    let k48b = q64(builder, 48);
    let exp_shifted = builder.ins().ishl(exp, k48b);
    let hi_rest = builder.ins().bor(exp_shifted, hi48);
    let hi = builder.ins().bor(sign_shifted, hi_rest);
    (lo, hi)
}

/// Widen f32 bits to `(lo, hi)` halves (see [`ref_f32_to_quad`]).
pub(crate) fn quad_from_f32(builder: &mut FunctionBuilder, bits32: ClifValue) -> (ClifValue, ClifValue) {
    let bits = builder.ins().uextend(types::I64, bits32);
    let k31 = q64(builder, 31);
    let s31 = builder.ins().ushr(bits, k31);
    let k1 = q64(builder, 1);
    let sign = builder.ins().band(s31, k1);
    let k23 = q64(builder, 23);
    let s23 = builder.ins().ushr(bits, k23);
    let kff = q64(builder, 0xff);
    let exp32 = builder.ins().band(s23, kff);
    let kmask = q64(builder, 0x7fffff);
    let mant32 = builder.ins().band(bits, kmask);
    let k0 = q64(builder, 0);
    let is_special = builder.ins().icmp(IntCC::Equal, exp32, kff);
    let is_zero_sub = builder.ins().icmp(IntCC::Equal, exp32, k0);
    let mant_is_zero = builder.ins().icmp(IntCC::Equal, mant32, k0);
    let is_zero = builder.ins().band(is_zero_sub, mant_is_zero);
    // Normal: the 23-bit fraction occupies bits 111:89, so the
    // low half is zero and the high 48 hold mant<<25.
    let k16256 = q64(builder, 16256);
    let n_exp = builder.ins().iadd(exp32, k16256);
    let k25 = q64(builder, 25);
    let n_hi = builder.ins().ishl(mant32, k25);
    // Subnormal: normalize, then place at bit 111.
    let lz = builder.ins().clz(mant32);
    let k16297 = q64(builder, 16297);
    let s_exp = builder.ins().isub(k16297, lz);
    let normed = builder.ins().ishl(mant32, lz);
    let k48 = q64(builder, 48);
    let s_lo = builder.ins().ishl(normed, k48);
    let k16 = q64(builder, 16);
    let s_hi_raw = builder.ins().ushr(normed, k16);
    let k47d = q64(builder, (1u64 << 47) as i64);
    let kneg1d = q64(builder, -1);
    let not47d = builder.ins().bxor(k47d, kneg1d);
    let s_hi = builder.ins().band(s_hi_raw, not47d);
    let zs_exp = builder.ins().select(is_zero, k0, s_exp);
    let zs_lo = builder.ins().select(is_zero, k0, s_lo);
    let zs_hi = builder.ins().select(is_zero, k0, s_hi);
    // NaN/Inf: low half always zero; high half shifts into place
    // with its quiet bit untouched (SNaN stays signaling). The
    // payload moves up: mant<<25 lands the top fraction bits under
    // bit 111 (a right shift would discard them to infinity).
    let is_inf = builder.ins().icmp(IntCC::Equal, mant32, k0);
    let pay_hi = builder.ins().ishl(mant32, k25);
    let sp_hi = builder.ins().select(is_inf, k0, pay_hi);
    let sp_exp = q64(builder, 0x7fff);
    let z_exp = builder.ins().select(is_zero_sub, zs_exp, n_exp);
    let exp = builder.ins().select(is_special, sp_exp, z_exp);
    let z_lo = builder.ins().select(is_zero_sub, zs_lo, k0);
    let lo = builder.ins().select(is_special, k0, z_lo);
    let z_hi = builder.ins().select(is_zero_sub, zs_hi, n_hi);
    let hi48 = builder.ins().select(is_special, sp_hi, z_hi);
    let k63s = q64(builder, 63);
    let sign_shifted = builder.ins().ishl(sign, k63s);
    let k48b = q64(builder, 48);
    let exp_shifted = builder.ins().ishl(exp, k48b);
    let hi_rest = builder.ins().bor(exp_shifted, hi48);
    let hi = builder.ins().bor(sign_shifted, hi_rest);
    (lo, hi)
}

/// Integer (already widened value-preservingly to `I64`) to
/// `(lo, hi)` halves (see [`ref_int_to_quad`]).
pub(crate) fn quad_from_int(builder: &mut FunctionBuilder, v: ClifValue, unsigned: bool) -> (ClifValue, ClifValue) {
    let k0 = q64(builder, 0);
    let k63 = q64(builder, 63);
    let k1 = q64(builder, 1);
    let (sign, mag) = if unsigned {
        (k0, v)
    } else {
        let s63 = builder.ins().ushr(v, k63);
        let s = builder.ins().band(s63, k1);
        let neg = builder.ins().ineg(v);
        let is_neg = builder.ins().icmp(IntCC::NotEqual, s, k0);
        let m = builder.ins().select(is_neg, neg, v);
        (s, m)
    };
    let is_zero = builder.ins().icmp(IntCC::Equal, mag, k0);
    // Fraction excludes the implicit leading 1: clear the top
    // set bit, then shift the remainder to bit 111 (S = 49+lz).
    let lz = builder.ins().clz(mag);
    let top = builder.ins().isub(k63, lz);
    let tbit = builder.ins().ishl(k1, top);
    let mprime = builder.ins().bxor(mag, tbit);
    let k49 = q64(builder, 49);
    let shift = builder.ins().iadd(k49, lz);
    let k64 = q64(builder, 64);
    let s_lt_64 = builder.ins().icmp(IntCC::UnsignedLessThan, shift, k64);
    let lo_full = builder.ins().ishl(mprime, shift);
    let lo_nz = builder.ins().select(s_lt_64, lo_full, k0);
    let sh64 = builder.ins().isub(shift, k64);
    let hi_full = builder.ins().ishl(mprime, sh64);
    let kfmask = q64(builder, 0x0000_ffff_ffff_ffff);
    let hi_masked = builder.ins().band(hi_full, kfmask);
    let hi_nz = builder.ins().select(s_lt_64, k0, hi_masked);
    let rsh = builder.ins().isub(k64, shift);
    let hi_r = builder.ins().ushr(mprime, rsh);
    let hi_r_masked = builder.ins().band(hi_r, kfmask);
    let hi48 = builder.ins().select(s_lt_64, hi_r_masked, hi_nz);
    let e = builder.ins().isub(k63, lz);
    let k16383 = q64(builder, 16383);
    let norm_exp = builder.ins().iadd(e, k16383);
    let exp = builder.ins().select(is_zero, k0, norm_exp);
    let zlo = builder.ins().select(is_zero, k0, lo_nz);
    let zhi = builder.ins().select(is_zero, k0, hi48);
    let k63s = q64(builder, 63);
    let sign_shifted = builder.ins().ishl(sign, k63s);
    let k48b = q64(builder, 48);
    let exp_shifted = builder.ins().ishl(exp, k48b);
    let hi_rest = builder.ins().bor(exp_shifted, zhi);
    let hi = builder.ins().bor(sign_shifted, hi_rest);
    (zlo, hi)
}

/// Integer to quad from a full 128-bit pattern (see
/// ref_int_to_quad128): the value arrives as `I128` (two's
/// complement when signed); magnitudes past 2^113 round half-even
/// with renormalization on carry.
pub(crate) fn quad_from_int128(builder: &mut FunctionBuilder, v: ClifValue, unsigned: bool) -> (ClifValue, ClifValue) {
    let k0_64 = q64(builder, 0);
    let k1_64 = q64(builder, 1);
    let k63 = q64(builder, 63);
    let k64 = q64(builder, 64);
    let k112 = q64(builder, 112);
    let k127 = q64(builder, 127);
    let k0_128 = builder.ins().iconcat(k0_64, k0_64);
    let k1_128 = builder.ins().iconcat(k1_64, k0_64);
    let (_, vhi) = builder.ins().isplit(v);
    let s63 = builder.ins().ushr(vhi, k63);
    let s = builder.ins().band(s63, k1_64);
    let (sign, mag) = if unsigned {
        (k0_64, v)
    } else {
        let neg = builder.ins().ineg(v);
        let is_neg = builder.ins().icmp(IntCC::NotEqual, s, k0_64);
        let m = builder.ins().select(is_neg, neg, v);
        (s, m)
    };
    let is_zero = builder.ins().icmp(IntCC::Equal, mag, k0_128);
    // 128-bit leading zeros from the halves.
    let (mlo, mhi) = builder.ins().isplit(mag);
    let lz_hi = builder.ins().clz(mhi);
    let lz_lo = builder.ins().clz(mlo);
    let hi_is_zero = builder.ins().icmp(IntCC::Equal, mhi, k0_64);
    let lz_lo_plus = builder.ins().iadd(lz_lo, k64);
    let lz = builder.ins().select(hi_is_zero, lz_lo_plus, lz_hi);
    let top = builder.ins().isub(k127, lz);
    // Exact when the magnitude fits 113 bits.
    let k15 = q64(builder, 15);
    let fits = builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, lz, k15);
    let tbit = builder.ins().ishl(k1_128, top);
    let mprime = builder.ins().isub(mag, tbit);
    let esh = builder.ins().isub(k112, top);
    let frac_exact = builder.ins().ishl(mprime, esh);
    // Otherwise round the low bits half-even (GRS at `drop`).
    let drop = builder.ins().isub(top, k112);
    let kept = builder.ins().ushr(mag, drop);
    let k1b = q64(builder, 1);
    let drop1 = builder.ins().isub(drop, k1b);
    let guard_full = builder.ins().ushr(mag, drop1);
    let guard = builder.ins().band(guard_full, k1_128);
    let mask_full = builder.ins().ishl(k1_128, drop1);
    let mask = builder.ins().isub(mask_full, k1_128);
    let rest_full = builder.ins().band(mag, mask);
    let rest_nz = builder.ins().icmp(IntCC::NotEqual, rest_full, k0_128);
    let rest = builder.ins().uextend(types::I128, rest_nz);
    let lsb = builder.ins().band(kept, k1_128);
    // Round bit below the guard; absent when `drop` is 1.
    let k2 = q64(builder, 2);
    let drop_ge2 = builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, drop, k2);
    let drop2 = builder.ins().isub(drop, k2);
    let round_full = builder.ins().ushr(mag, drop2);
    let round_bit = builder.ins().band(round_full, k1_128);
    let round = builder.ins().select(drop_ge2, round_bit, k0_128);
    let r_or_s = builder.ins().bor(round, rest);
    let r_or_l = builder.ins().bor(r_or_s, lsb);
    let inc = builder.ins().band(guard, r_or_l);
    let kept_inc = builder.ins().iadd(kept, inc);
    let k113 = q64(builder, 113);
    let top113 = builder.ins().ishl(k1_128, k113);
    let carried = builder.ins().icmp(IntCC::Equal, kept_inc, top113);
    let kept_n = builder.ins().ushr(kept_inc, k1b);
    let kept1 = builder.ins().select(carried, kept_n, kept_inc);
    let e_inc = builder.ins().iadd(top, k1_64);
    let e_round = builder.ins().select(carried, e_inc, top);
    let one112 = builder.ins().ishl(k1_128, k112);
    let frac_r = builder.ins().isub(kept1, one112);
    let frac = builder.ins().select(fits, frac_exact, frac_r);
    let e = builder.ins().select(fits, top, e_round);
    let k16383 = q64(builder, 16383);
    let norm_exp = builder.ins().iadd(e, k16383);
    let exp = builder.ins().select(is_zero, k0_64, norm_exp);
    // Split the 112-bit fraction back into halves.
    let (flo, _) = builder.ins().isplit(frac);
    let k64b = q64(builder, 64);
    let fhi_shifted = builder.ins().ushr(frac, k64b);
    let (fhi, _) = builder.ins().isplit(fhi_shifted);
    let kfmask = q64(builder, 0x0000_ffff_ffff_ffff);
    let fhi_masked = builder.ins().band(fhi, kfmask);
    let zlo = builder.ins().select(is_zero, k0_64, flo);
    let zhi = builder.ins().select(is_zero, k0_64, fhi_masked);
    let k63s = q64(builder, 63);
    let sign_shifted = builder.ins().ishl(sign, k63s);
    let k48b = q64(builder, 48);
    let exp_shifted = builder.ins().ishl(exp, k48b);
    let hi_rest = builder.ins().bor(exp_shifted, zhi);
    let hi = builder.ins().bor(sign_shifted, hi_rest);
    (zlo, hi)
}
/// Magnitude (integer part) of a normalized quad value at `I64`
/// width: significand with implicit leading 1, shifted down by
/// `112 - E`. Only valid for `0 <= E <= 62` (callers select away
/// every other case first).
fn quad_mag64(builder: &mut FunctionBuilder, flo: ClifValue, fhi: ClifValue, e: ClifValue) -> ClifValue {
    let k112 = q32(builder, 112);
    let k = builder.ins().isub(k112, e);
    let k64 = builder.ins().sextend(types::I64, k);
    let k1 = q64(builder, 1);
    let k48 = q64(builder, 48);
    let one48 = builder.ins().ishl(k1, k48);
    let h = builder.ins().bor(one48, fhi);
    let k64c = q64(builder, 64);
    let ck64 = q32(builder, 64);
    let k_ge = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, k, ck64);
    let sh1 = builder.ins().isub(k64, k64c);
    let mag_hi = builder.ins().ushr(h, sh1);
    let sh2 = builder.ins().isub(k64c, k64);
    let hi_part = builder.ins().ishl(h, sh2);
    let lo_part = builder.ins().ushr(flo, k64);
    let mag_lo = builder.ins().bor(hi_part, lo_part);
    builder.ins().select(k_ge, mag_hi, mag_lo)
}

/// Quad halves to signed-64 bits: saturating truncation (see
/// ref_quad_to_s64).
pub(crate) fn quad_to_s64(builder: &mut FunctionBuilder, lo: ClifValue, hi: ClifValue) -> ClifValue {
    let k0 = q64(builder, 0);
    let (sign, exp, flo, fhi) = quad_unpack(builder, lo, hi);
    let e = quad_exp_biased(builder, exp);
    let is_neg = builder.ins().icmp(IntCC::NotEqual, sign, k0);
    let ck0 = q32(builder, 0);
    let neg_e = builder.ins().icmp(IntCC::SignedLessThan, e, ck0);
    let ck63 = q32(builder, 63);
    let ovf = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, e, ck63);
    let mag = quad_mag64(builder, flo, fhi, e);
    let neg = builder.ins().ineg(mag);
    let res = builder.ins().select(is_neg, neg, mag);
    let max = q64(builder, i64::MAX);
    let min = q64(builder, i64::MIN);
    let sat = builder.ins().select(is_neg, min, max);
    let ovf_res = builder.ins().select(ovf, sat, res);
    builder.ins().select(neg_e, k0, ovf_res)
}

/// Quad halves to signed-32 bits (see [`ref_quad_to_s32`]).
/// Magnitude only needs the top: E <= 30 always shifts by >= 82.
pub(crate) fn quad_to_s32(builder: &mut FunctionBuilder, lo: ClifValue, hi: ClifValue) -> ClifValue {
    let k0 = q64(builder, 0);
    let (sign, exp, _, fhi) = quad_unpack(builder, lo, hi);
    let e = quad_exp_biased(builder, exp);
    let is_neg = builder.ins().icmp(IntCC::NotEqual, sign, k0);
    let ck0 = q32(builder, 0);
    let neg_e = builder.ins().icmp(IntCC::SignedLessThan, e, ck0);
    let ck31 = q32(builder, 31);
    let ovf = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, e, ck31);
    let k112 = q32(builder, 112);
    let k = builder.ins().isub(k112, e);
    let k64 = builder.ins().sextend(types::I64, k);
    let k64c = q64(builder, 64);
    let sh = builder.ins().isub(k64, k64c);
    let k1 = q64(builder, 1);
    let k48 = q64(builder, 48);
    let one48 = builder.ins().ishl(k1, k48);
    let h = builder.ins().bor(one48, fhi);
    let mag = builder.ins().ushr(h, sh);
    let neg = builder.ins().ineg(mag);
    let res = builder.ins().select(is_neg, neg, mag);
    let max = q64(builder, i64::from(i32::MAX));
    let min = q64(builder, i64::from(i32::MIN));
    let sat = builder.ins().select(is_neg, min, max);
    let ovf_res = builder.ins().select(ovf, sat, res);
    let full = builder.ins().select(neg_e, k0, ovf_res);
    builder.ins().ireduce(types::I32, full)
}

/// Quad halves to unsigned-64 bits: saturating truncation (see
/// ref_quad_to_u64). Negatives (including -Inf) go to zero.
pub(crate) fn quad_to_u64(builder: &mut FunctionBuilder, lo: ClifValue, hi: ClifValue) -> ClifValue {
    let k0 = q64(builder, 0);
    let (sign, exp, flo, fhi) = quad_unpack(builder, lo, hi);
    let e = quad_exp_biased(builder, exp);
    let is_neg = builder.ins().icmp(IntCC::NotEqual, sign, k0);
    let ck0 = q32(builder, 0);
    let neg_e = builder.ins().icmp(IntCC::SignedLessThan, e, ck0);
    let ck64 = q32(builder, 64);
    let ovf = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, e, ck64);
    let mag = quad_mag64(builder, flo, fhi, e);
    let max = q64(builder, u64::MAX as i64);
    let ovf_res = builder.ins().select(ovf, max, mag);
    let zero_res = builder.ins().select(neg_e, k0, ovf_res);
    builder.ins().select(is_neg, k0, zero_res)
}

/// Quad halves to unsigned-32 bits (see [`ref_quad_to_u32`]).
pub(crate) fn quad_to_u32(builder: &mut FunctionBuilder, lo: ClifValue, hi: ClifValue) -> ClifValue {
    let k0 = q64(builder, 0);
    let (sign, exp, _, fhi) = quad_unpack(builder, lo, hi);
    let e = quad_exp_biased(builder, exp);
    let is_neg = builder.ins().icmp(IntCC::NotEqual, sign, k0);
    let ck0 = q32(builder, 0);
    let neg_e = builder.ins().icmp(IntCC::SignedLessThan, e, ck0);
    let ck32 = q32(builder, 32);
    let ovf = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, e, ck32);
    // E <= 31 here: magnitude is H >> (48-E).
    let k48 = q32(builder, 48);
    let sh = builder.ins().isub(k48, e);
    let sh64 = builder.ins().sextend(types::I64, sh);
    let k1 = q64(builder, 1);
    let kc48 = q64(builder, 48);
    let one48 = builder.ins().ishl(k1, kc48);
    let h = builder.ins().bor(one48, fhi);
    let mag = builder.ins().ushr(h, sh64);
    let max = q64(builder, 0xffff_ffff);
    let ovf_res = builder.ins().select(ovf, max, mag);
    let zero_res = builder.ins().select(neg_e, k0, ovf_res);
    let full = builder.ins().select(is_neg, k0, zero_res);
    builder.ins().ireduce(types::I32, full)
}

/// Quad halves to signed-128 bits: saturating truncation (see
/// ref_quad_to_s128). The 113-bit significand rides one `I128`
/// lane; shifts past bit 112 go left instead of right (callers with
/// `E >= 127` select saturation first, like the 64-bit path).
pub(crate) fn quad_to_s128(builder: &mut FunctionBuilder, lo: ClifValue, hi: ClifValue) -> ClifValue {
    let k0 = q64(builder, 0);
    let (sign, exp, flo, fhi) = quad_unpack(builder, lo, hi);
    let e = quad_exp_biased(builder, exp);
    let is_neg = builder.ins().icmp(IntCC::NotEqual, sign, k0);
    let ck0 = q32(builder, 0);
    let neg_e = builder.ins().icmp(IntCC::SignedLessThan, e, ck0);
    let ck127 = q32(builder, 127);
    let ovf = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, e, ck127);
    let k1 = q64(builder, 1);
    let kc48 = q64(builder, 48);
    let one48 = builder.ins().ishl(k1, kc48);
    let h = builder.ins().bor(one48, fhi);
    let wide = builder.ins().iconcat(flo, h);
    let k112 = q32(builder, 112);
    let k = builder.ins().isub(k112, e);
    let k64 = builder.ins().sextend(types::I64, k);
    let k_ge0 = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, k, ck0);
    let mag_r = builder.ins().ushr(wide, k64);
    let nk = builder.ins().isub(ck0, k);
    let nk64 = builder.ins().sextend(types::I64, nk);
    let mag_l = builder.ins().ishl(wide, nk64);
    let mag = builder.ins().select(k_ge0, mag_r, mag_l);
    let neg = builder.ins().ineg(mag);
    let res = builder.ins().select(is_neg, neg, mag);
    let max_lo = q64(builder, u64::MAX as i64);
    let max_hi = q64(builder, i64::MAX);
    let max128 = builder.ins().iconcat(max_lo, max_hi);
    let min_hi = q64(builder, i64::MIN);
    let min128 = builder.ins().iconcat(k0, min_hi);
    let sat = builder.ins().select(is_neg, min128, max128);
    let ovf_res = builder.ins().select(ovf, sat, res);
    let k0_128 = builder.ins().iconcat(k0, k0);
    builder.ins().select(neg_e, k0_128, ovf_res)
}

/// Quad halves to unsigned-128 bits: saturating truncation (see
/// ref_quad_to_u128). Negatives go to zero, like the 64-bit
/// path.
pub(crate) fn quad_to_u128(builder: &mut FunctionBuilder, lo: ClifValue, hi: ClifValue) -> ClifValue {
    let k0 = q64(builder, 0);
    let (sign, exp, flo, fhi) = quad_unpack(builder, lo, hi);
    let e = quad_exp_biased(builder, exp);
    let is_neg = builder.ins().icmp(IntCC::NotEqual, sign, k0);
    let ck0 = q32(builder, 0);
    let neg_e = builder.ins().icmp(IntCC::SignedLessThan, e, ck0);
    let ck128 = q32(builder, 128);
    let ovf = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, e, ck128);
    let k1 = q64(builder, 1);
    let kc48 = q64(builder, 48);
    let one48 = builder.ins().ishl(k1, kc48);
    let h = builder.ins().bor(one48, fhi);
    let wide = builder.ins().iconcat(flo, h);
    let k112 = q32(builder, 112);
    let k = builder.ins().isub(k112, e);
    let k64 = builder.ins().sextend(types::I64, k);
    let k_ge0 = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, k, ck0);
    let mag_r = builder.ins().ushr(wide, k64);
    let nk = builder.ins().isub(ck0, k);
    let nk64 = builder.ins().sextend(types::I64, nk);
    let mag_l = builder.ins().ishl(wide, nk64);
    let mag = builder.ins().select(k_ge0, mag_r, mag_l);
    let all1 = q64(builder, u64::MAX as i64);
    let max128 = builder.ins().iconcat(all1, all1);
    let ovf_res = builder.ins().select(ovf, max128, mag);
    let k0_128 = builder.ins().iconcat(k0, k0);
    let zero_res = builder.ins().select(neg_e, k0_128, ovf_res);
    builder.ins().select(is_neg, k0_128, zero_res)
}

/// Quad halves to f64 bits: round-half-even (see
/// ref_quad_to_f64). Every NaN forces the quiet bit and
/// truncates its payload.
pub(crate) fn quad_to_f64(builder: &mut FunctionBuilder, lo: ClifValue, hi: ClifValue) -> ClifValue {
    let k0 = q64(builder, 0);
    let k1 = q64(builder, 1);
    let (sign, exp, flo, fhi) = quad_unpack(builder, lo, hi);
    let k7fff = q64(builder, 0x7fff);
    let is_max = builder.ins().icmp(IntCC::Equal, exp, k7fff);
    let flo_nz = builder.ins().icmp(IntCC::NotEqual, flo, k0);
    let fhi_nz = builder.ins().icmp(IntCC::NotEqual, fhi, k0);
    let frac_nz = builder.ins().bor(flo_nz, fhi_nz);
    let is_nan = builder.ins().band(is_max, frac_nz);
    let is_zero_exp = builder.ins().icmp(IntCC::Equal, exp, k0);
    let k63 = q64(builder, 63);
    let sign63 = builder.ins().ishl(sign, k63);
    let zero_res = sign63;
    // NaN: force the quiet bit, truncate the payload (SNaN
    // payload bits survive shifting, probed).
    let k52m = q64(builder, 0x7ff);
    let k52s = q64(builder, 52);
    let expmax = builder.ins().ishl(k52m, k52s);
    let k4 = q64(builder, 4);
    let pay_hi = builder.ins().ishl(fhi, k4);
    let k60 = q64(builder, 60);
    let pay_lo = builder.ins().ushr(flo, k60);
    let payload = builder.ins().bor(pay_hi, pay_lo);
    let k51c = q64(builder, 51);
    let qbit = builder.ins().ishl(k1, k51c);
    let nan_pay = builder.ins().bor(payload, qbit);
    let nan_hi = builder.ins().bor(expmax, nan_pay);
    let nan_res = builder.ins().bor(sign63, nan_hi);
    // Unbiased double exponent.
    let e = quad_exp_biased(builder, exp);
    let k1023 = q32(builder, 1023);
    let e64 = builder.ins().iadd(e, k1023);
    let k7ff = q32(builder, 0x7ff);
    let ovf = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, e64, k7ff);
    let inf_res = builder.ins().bor(sign63, expmax);
    // Normal split: keep 53, drop 60 (all in the low half).
    let k0b = builder.ins().bor(pay_hi, pay_lo);
    let k59 = q64(builder, 59);
    let flo59 = builder.ins().ushr(flo, k59);
    let guard = builder.ins().band(flo59, k1);
    let k58 = q64(builder, 58);
    let flo58 = builder.ins().ushr(flo, k58);
    let round = builder.ins().band(flo58, k1);
    let ksticky = q64(builder, 0x03ff_ffff_ffff_ffff);
    let rest = builder.ins().band(flo, ksticky);
    let rest_nz = builder.ins().icmp(IntCC::NotEqual, rest, k0);
    let sticky = builder.ins().uextend(types::I64, rest_nz);
    let lsb = builder.ins().band(k0b, k1);
    let r_or_s = builder.ins().bor(round, sticky);
    let r_or_l = builder.ins().bor(r_or_s, lsb);
    let inc = builder.ins().band(guard, r_or_l);
    let kept = builder.ins().iadd(k0b, inc);
    let k53 = q64(builder, 53);
    let k1_53 = builder.ins().ishl(k1, k53);
    let carried = builder.ins().icmp(IntCC::Equal, kept, k1_53);
    let kept_n = builder.ins().ushr(kept, k1);
    let kept1 = builder.ins().select(carried, kept_n, kept);
    let ck1 = q32(builder, 1);
    let e64_inc = builder.ins().iadd(e64, ck1);
    let e64n = builder.ins().select(carried, e64_inc, e64);
    let ovf2 = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, e64n, k7ff);
    let k52mask = q64(builder, 0x000f_ffff_ffff_ffff);
    let mant = builder.ins().band(kept1, k52mask);
    let e64_64 = builder.ins().sextend(types::I64, e64n);
    let enorm_lo = builder.ins().ishl(e64_64, k52s);
    let enorm_hi = builder.ins().bor(enorm_lo, mant);
    let enorm = builder.ins().bor(sign63, enorm_hi);
    let norm_res = builder.ins().select(ovf2, inf_res, enorm);
    // Subnormal: shift the (kept, guard, round, sticky) tuple;
    // s >= 56 shifts everything out (exact zero).
    let k1_32 = q32(builder, 1);
    let s = builder.ins().isub(k1_32, e64);
    let s64 = builder.ins().sextend(types::I64, s);
    let ck56 = q32(builder, 56);
    let s_ge_56 = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, s, ck56);
    let k3 = q64(builder, 3);
    let t0_shifted = builder.ins().ishl(k0b, k3);
    // The tuple above holds only fraction bits; the subnormal scale
    // needs the implicit leading 1 too (bit 55 after the shift),
    // unlike the normal path which carries it in the exponent.
    let k55 = q64(builder, 55);
    let one55 = builder.ins().ishl(k1, k55);
    let t0 = builder.ins().bor(t0_shifted, one55);
    let k2 = q64(builder, 2);
    let t_guard = builder.ins().ishl(guard, k2);
    let t1 = builder.ins().bor(t0, t_guard);
    let t_round = builder.ins().ishl(round, k1);
    let t2 = builder.ins().bor(t1, t_round);
    let t3 = builder.ins().bor(t2, sticky);
    let tsh = builder.ins().ushr(t3, s64);
    let kept_s = builder.ins().ushr(tsh, k3);
    let tsh2 = builder.ins().ushr(tsh, k2);
    let g_s = builder.ins().band(tsh2, k1);
    let tsh1 = builder.ins().ushr(tsh, k1);
    let r_s = builder.ins().band(tsh1, k1);
    let shifted1 = builder.ins().ishl(k1, s64);
    let mask = builder.ins().isub(shifted1, k1);
    let dropped = builder.ins().band(t3, mask);
    let st_extra = builder.ins().icmp(IntCC::NotEqual, dropped, k0);
    let st_extra64 = builder.ins().uextend(types::I64, st_extra);
    let t0b = builder.ins().band(tsh, k1);
    let st_s = builder.ins().bor(t0b, st_extra64);
    let lsb_s = builder.ins().band(kept_s, k1);
    let rs_s = builder.ins().bor(r_s, st_s);
    let rl_s = builder.ins().bor(rs_s, lsb_s);
    let inc_s = builder.ins().band(g_s, rl_s);
    let kept_s2 = builder.ins().iadd(kept_s, inc_s);
    let k52b = q64(builder, 52);
    let k1_52 = builder.ins().ishl(k1, k52b);
    let carried_s = builder.ins().icmp(IntCC::Equal, kept_s2, k1_52);
    let k1_52s = builder.ins().ishl(k1, k52s);
    let min_normal = builder.ins().bor(sign63, k1_52s);
    let sub_full = builder.ins().bor(sign63, kept_s2);
    let sub_res = builder.ins().select(carried_s, min_normal, sub_full);
    let k1c = q32(builder, 1);
    let is_normal = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, e64, k1c);
    let fin_res = builder.ins().select(is_normal, norm_res, sub_res);
    let exp_res = builder.ins().select(ovf, inf_res, fin_res);
    let nan_sel = builder.ins().select(is_nan, nan_res, exp_res);
    let sub_or = builder.ins().select(s_ge_56, zero_res, nan_sel);
    builder.ins().select(is_zero_exp, zero_res, sub_or)
}

/// Quad halves to f32 bits (see [`ref_quad_to_f32`]): the top 24
/// significand bits keep, 89 drop, with the guard in the high
/// half and the sticky spanning both.
pub(crate) fn quad_to_f32(builder: &mut FunctionBuilder, lo: ClifValue, hi: ClifValue) -> ClifValue {
    let k0 = q64(builder, 0);
    let k1 = q64(builder, 1);
    let (sign, exp, flo, fhi) = quad_unpack(builder, lo, hi);
    let k7fff = q64(builder, 0x7fff);
    let is_max = builder.ins().icmp(IntCC::Equal, exp, k7fff);
    let flo_nz = builder.ins().icmp(IntCC::NotEqual, flo, k0);
    let fhi_nz = builder.ins().icmp(IntCC::NotEqual, fhi, k0);
    let frac_nz = builder.ins().bor(flo_nz, fhi_nz);
    let is_nan = builder.ins().band(is_max, frac_nz);
    let is_zero_exp = builder.ins().icmp(IntCC::Equal, exp, k0);
    let k31 = q64(builder, 31);
    let sign32 = builder.ins().ishl(sign, k31);
    let zero_res = sign32;
    let kff = q64(builder, 0xff);
    let k23 = q64(builder, 23);
    let expmax = builder.ins().ishl(kff, k23);
    let k25 = q64(builder, 25);
    let pay24 = builder.ins().ushr(fhi, k25);
    let k22 = q64(builder, 22);
    let qbit = builder.ins().ishl(k1, k22);
    let nan_pay = builder.ins().bor(pay24, qbit);
    let nan32 = builder.ins().bor(expmax, nan_pay);
    let nan_res = builder.ins().bor(sign32, nan32);
    let e = quad_exp_biased(builder, exp);
    let k127 = q32(builder, 127);
    let e32 = builder.ins().iadd(e, k127);
    let kcff = q32(builder, 0xff);
    let ovf = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, e32, kcff);
    let inf_res = builder.ins().bor(sign32, expmax);
    // Normal split: keep 24 (implicit 1 + 23), drop 89.
    let k48 = q64(builder, 48);
    let one48 = builder.ins().ishl(k1, k48);
    let sig_hi = builder.ins().bor(one48, fhi);
    let kept24 = builder.ins().ushr(sig_hi, k25);
    let k24 = q64(builder, 24);
    let sig24 = builder.ins().ushr(sig_hi, k24);
    let guard = builder.ins().band(sig24, k1);
    let sig23 = builder.ins().ushr(sig_hi, k23);
    let round = builder.ins().band(sig23, k1);
    let k23mask = q64(builder, 0x7fffff);
    let rest_hi = builder.ins().band(sig_hi, k23mask);
    let rest_nz_hi = builder.ins().icmp(IntCC::NotEqual, rest_hi, k0);
    let rest_nz = builder.ins().bor(rest_nz_hi, flo_nz);
    let sticky = builder.ins().uextend(types::I64, rest_nz);
    let lsb = builder.ins().band(kept24, k1);
    let r_or_s = builder.ins().bor(round, sticky);
    let r_or_l = builder.ins().bor(r_or_s, lsb);
    let inc = builder.ins().band(guard, r_or_l);
    let kept = builder.ins().iadd(kept24, inc);
    let k1_24 = builder.ins().ishl(k1, k24);
    let carried = builder.ins().icmp(IntCC::Equal, kept, k1_24);
    let kept_n = builder.ins().ushr(kept, k1);
    let kept1 = builder.ins().select(carried, kept_n, kept);
    let ck1 = q32(builder, 1);
    let e32_inc = builder.ins().iadd(e32, ck1);
    let e32n = builder.ins().select(carried, e32_inc, e32);
    let ovf2 = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, e32n, kcff);
    let mant = builder.ins().band(kept1, k23mask);
    let e32_64 = builder.ins().sextend(types::I64, e32n);
    let enorm_lo = builder.ins().ishl(e32_64, k23);
    let enorm_hi = builder.ins().bor(enorm_lo, mant);
    let enorm = builder.ins().bor(sign32, enorm_hi);
    let norm_res = builder.ins().select(ovf2, inf_res, enorm);
    // Subnormal: shift the (kept, guard, round, sticky) tuple;
    // total 89+s exceeds 113 at s >= 25, which is exact zero.
    let k1_32 = q32(builder, 1);
    let s = builder.ins().isub(k1_32, e32);
    let s64 = builder.ins().sextend(types::I64, s);
    let ck25 = q32(builder, 25);
    let s_ge_25 = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, s, ck25);
    let k3 = q64(builder, 3);
    let t0 = builder.ins().ishl(kept24, k3);
    let k2 = q64(builder, 2);
    let t_guard = builder.ins().ishl(guard, k2);
    let t1 = builder.ins().bor(t0, t_guard);
    let t_round = builder.ins().ishl(round, k1);
    let t2 = builder.ins().bor(t1, t_round);
    let t3 = builder.ins().bor(t2, sticky);
    let tsh = builder.ins().ushr(t3, s64);
    let kept_s = builder.ins().ushr(tsh, k3);
    let tsh2 = builder.ins().ushr(tsh, k2);
    let g_s = builder.ins().band(tsh2, k1);
    let tsh1 = builder.ins().ushr(tsh, k1);
    let r_s = builder.ins().band(tsh1, k1);
    let shifted1 = builder.ins().ishl(k1, s64);
    let mask = builder.ins().isub(shifted1, k1);
    let dropped = builder.ins().band(t3, mask);
    let st_extra = builder.ins().icmp(IntCC::NotEqual, dropped, k0);
    let st_extra64 = builder.ins().uextend(types::I64, st_extra);
    let t0b = builder.ins().band(tsh, k1);
    let st_s = builder.ins().bor(t0b, st_extra64);
    let lsb_s = builder.ins().band(kept_s, k1);
    let rs_s = builder.ins().bor(r_s, st_s);
    let rl_s = builder.ins().bor(rs_s, lsb_s);
    let inc_s = builder.ins().band(g_s, rl_s);
    let kept_s2 = builder.ins().iadd(kept_s, inc_s);
    let k1_23 = builder.ins().ishl(k1, k23);
    let carried_s = builder.ins().icmp(IntCC::Equal, kept_s2, k1_23);
    let k1_23s = builder.ins().ishl(k1, k23);
    let min_normal = builder.ins().bor(sign32, k1_23s);
    let sub_full = builder.ins().bor(sign32, kept_s2);
    let sub_res = builder.ins().select(carried_s, min_normal, sub_full);
    let k1c = q32(builder, 1);
    let is_normal = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, e32, k1c);
    let fin_res = builder.ins().select(is_normal, norm_res, sub_res);
    let exp_res = builder.ins().select(ovf, inf_res, fin_res);
    let nan_sel = builder.ins().select(is_nan, nan_res, exp_res);
    let sub_or = builder.ins().select(s_ge_25, zero_res, nan_sel);
    let full = builder.ins().select(is_zero_exp, zero_res, sub_or);
    builder.ins().ireduce(types::I32, full)
}

/// Negate a quad value: flip bit 127 (bit 63 of the high half) with
/// integer ops. Exact, including signed zeros and NaNs.
pub(crate) fn quad_neg(builder: &mut FunctionBuilder, v: ClifValue) -> ClifValue {
    let addr = quad_spill(builder, v);
    let hi = builder.ins().load(types::I64, memflags::trusted(), addr, Offset32::new(8));
    let k63 = q64(builder, 63);
    let k1 = q64(builder, 1);
    let sign = builder.ins().ishl(k1, k63);
    let flipped = builder.ins().bxor(hi, sign);
    builder.ins().store(memflags::trusted(), flipped, addr, Offset32::new(8));
    builder.ins().load(types::F128, memflags::trusted(), addr, Offset32::new(0))
}

/// Round halves toward an integer (see [`ref_quad_ceil`] /
/// ref_quad_floor): truncate the fraction at `112 - E` and, when
/// inexact, increment the magnitude iff rounding away from zero in
/// `ceil`'s direction (`ceil` rounds up positives, `floor` rounds
/// down negatives). NaN, infinities, signed zeros, and integral
/// values select the input back bit-identically; `|x| < 1` selects
/// the signed constants; a carry out of the 112-bit fraction
/// renormalizes to `exp + 1`.
fn quad_round_halves(builder: &mut FunctionBuilder, lo: ClifValue, hi: ClifValue, ceil: bool) -> (ClifValue, ClifValue) {
    let k0 = q64(builder, 0);
    let k1 = q64(builder, 1);
    let cone8 = q8(builder, true);
    let (sign, exp, flo, fhi) = quad_unpack(builder, lo, hi);
    // Specials passthrough bit-identically.
    let k7fff = q64(builder, 0x7fff);
    let is_max = builder.ins().icmp(IntCC::Equal, exp, k7fff);
    let lo_nz = builder.ins().icmp(IntCC::NotEqual, flo, k0);
    let fhi_nz = builder.ins().icmp(IntCC::NotEqual, fhi, k0);
    let frac_nz = builder.ins().bor(lo_nz, fhi_nz);
    let is_nan = builder.ins().band(is_max, frac_nz);
    let not_frac = builder.ins().bxor(frac_nz, cone8);
    let is_inf = builder.ins().band(is_max, not_frac);
    let is_zero_exp = builder.ins().icmp(IntCC::Equal, exp, k0);
    let lo_z = builder.ins().icmp(IntCC::Equal, flo, k0);
    let hi_z = builder.ins().icmp(IntCC::Equal, fhi, k0);
    let frac_z = builder.ins().band(lo_z, hi_z);
    let is_zero = builder.ins().band(is_zero_exp, frac_z);
    let special12 = builder.ins().bor(is_nan, is_inf);
    let is_special = builder.ins().bor(special12, is_zero);
    // |x| < 1 selects the signed constants below.
    let is_denorm_nz = builder.ins().band(is_zero_exp, frac_nz);
    let e = quad_exp_biased(builder, exp);
    let ck0 = q32(builder, 0);
    let e_neg = builder.ins().icmp(IntCC::SignedLessThan, e, ck0);
    let not_zero_exp = builder.ins().bxor(is_zero_exp, cone8);
    let is_small_norm = builder.ins().band(not_zero_exp, e_neg);
    let is_small = builder.ins().bor(is_denorm_nz, is_small_norm);
    // Already integral.
    let ck112 = q32(builder, 112);
    let is_big = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, e, ck112);
    // Dropped-bit count and branch (low 64 vs high 48).
    let ck112b = q32(builder, 112);
    let k32 = builder.ins().isub(ck112b, e);
    let k64 = builder.ins().sextend(types::I64, k32);
    let k64c = q64(builder, 64);
    let lt64 = builder.ins().icmp(IntCC::UnsignedLessThan, k64, k64c);
    // Low-half masks (valid when k < 64).
    let sh_k = builder.ins().ishl(k1, k64);
    let mask_lt = builder.ins().isub(sh_k, k1);
    let all1 = q64(builder, -1);
    let not_mask_lt = builder.ins().bxor(mask_lt, all1);
    let lo_t_lt = builder.ins().band(flo, not_mask_lt);
    let dropped_lt = builder.ins().band(flo, mask_lt);
    let sticky_lt = builder.ins().icmp(IntCC::NotEqual, dropped_lt, k0);
    // High-half masks (valid when k >= 64, t = k - 64 in 0..48).
    let t64 = builder.ins().isub(k64, k64c);
    let sh_t = builder.ins().ishl(k1, t64);
    let mask_h = builder.ins().isub(sh_t, k1);
    let not_mask_h = builder.ins().bxor(mask_h, all1);
    let fhi_t_ge = builder.ins().band(fhi, not_mask_h);
    let fhi_drop = builder.ins().band(fhi, mask_h);
    let fhi_drop_nz = builder.ins().icmp(IntCC::NotEqual, fhi_drop, k0);
    let sticky_ge = builder.ins().bor(lo_nz, fhi_drop_nz);
    let lo_t = builder.ins().select(lt64, lo_t_lt, k0);
    let fhi_t = builder.ins().select(lt64, fhi, fhi_t_ge);
    let sticky = builder.ins().select(lt64, sticky_lt, sticky_ge);
    let exact = builder.ins().bxor(sticky, cone8);
    // Rounding direction: ceil increments positives, floor negatives.
    let is_pos = builder.ins().icmp(IntCC::Equal, sign, k0);
    let dir = if ceil { is_pos } else { builder.ins().bxor(is_pos, cone8) };
    let do_inc = builder.ins().band(sticky, dir);
    // Incremented magnitude, per branch, with renormalization.
    let lo_r_lt = builder.ins().iadd(lo_t_lt, sh_k);
    let carry_lt = builder.ins().icmp(IntCC::UnsignedLessThan, lo_r_lt, lo_t_lt);
    let carry64_lt = builder.ins().uextend(types::I64, carry_lt);
    let fhi_r_lt = builder.ins().iadd(fhi, carry64_lt);
    let fhi_r_ge = builder.ins().iadd(fhi_t_ge, sh_t);
    let top48 = q64(builder, (1u64 << 48) as i64);
    let ovf_lt = builder.ins().icmp(IntCC::Equal, fhi_r_lt, top48);
    let ovf_ge = builder.ins().icmp(IntCC::Equal, fhi_r_ge, top48);
    let renorm_inc = builder.ins().select(lt64, ovf_lt, ovf_ge);
    let renorm = builder.ins().band(do_inc, renorm_inc);
    let exp_p1 = builder.ins().iadd(exp, k1);
    let lo_inc = builder.ins().select(lt64, lo_r_lt, k0);
    let fhi_inc = builder.ins().select(lt64, fhi_r_lt, fhi_r_ge);
    let lo_r = builder.ins().select(renorm, k0, lo_inc);
    let fhi_r = builder.ins().select(renorm, k0, fhi_inc);
    let exp_r = builder.ins().select(renorm, exp_p1, exp);
    let lo_main = builder.ins().select(do_inc, lo_r, lo_t);
    let fhi_main = builder.ins().select(do_inc, fhi_r, fhi_t);
    let exp_main = builder.ins().select(do_inc, exp_r, exp);
    // Small constants: ceil -> 1 / -0, floor -> +0 / -1.
    let hi_one = q64(builder, 0x3fff_0000_0000_0000u64 as i64);
    let hi_negzero = q64(builder, 0x8000_0000_0000_0000u64 as i64);
    let hi_negone = q64(builder, 0xbfff_0000_0000_0000u64 as i64);
    let (small_lo, small_hi) = if ceil {
        (k0, builder.ins().select(is_pos, hi_one, hi_negzero))
    } else {
        (k0, builder.ins().select(is_pos, k0, hi_negone))
    };
    // Priority nest: special first, then small, then main-or-input.
    // `use_main` needs the input back for big/exact cases.
    let not_special = builder.ins().bxor(is_special, cone8);
    let not_small = builder.ins().bxor(is_small, cone8);
    let not_big = builder.ins().bxor(is_big, cone8);
    let not_exact = builder.ins().bxor(exact, cone8);
    let use_main_a = builder.ins().band(not_special, not_small);
    let use_main_b = builder.ins().band(not_big, not_exact);
    let use_main = builder.ins().band(use_main_a, use_main_b);
    let lo_mid = builder.ins().select(use_main, lo_main, lo);
    let fhi_mid = builder.ins().select(use_main, fhi_main, fhi);
    let exp_mid = builder.ins().select(use_main, exp_main, exp);
    let lo_pre = builder.ins().select(is_small, small_lo, lo_mid);
    // Reassemble hi: the small path selects a complete constant;
    // otherwise rebuild from sign, exponent, and fraction (which
    // equals the input hi whenever the main path is off).
    let k63 = q64(builder, 63);
    let sign_shifted = builder.ins().ishl(sign, k63);
    let k48 = q64(builder, 48);
    let exp_shifted = builder.ins().ishl(exp_mid, k48);
    let exp_frac = builder.ins().bor(exp_shifted, fhi_mid);
    let hi_mid = builder.ins().bor(sign_shifted, exp_frac);
    let hi_pre = builder.ins().select(is_small, small_hi, hi_mid);
    let lo_final = builder.ins().select(is_special, lo, lo_pre);
    let hi_final = builder.ins().select(is_special, hi, hi_pre);
    (lo_final, hi_final)
}

/// Ceil a quad carrier value (see [`ref_quad_ceil`]).
pub(crate) fn quad_ceil(builder: &mut FunctionBuilder, v: ClifValue) -> ClifValue {
    let addr = quad_spill(builder, v);
    let (lo, hi) = quad_load_halves(builder, addr);
    let (nlo, nhi) = quad_round_halves(builder, lo, hi, true);
    quad_store_carrier(builder, nlo, nhi)
}

/// Floor a quad carrier value (see [`ref_quad_floor`]).
pub(crate) fn quad_floor(builder: &mut FunctionBuilder, v: ClifValue) -> ClifValue {
    let addr = quad_spill(builder, v);
    let (lo, hi) = quad_load_halves(builder, addr);
    let (nlo, nhi) = quad_round_halves(builder, lo, hi, false);
    quad_store_carrier(builder, nlo, nhi)
}

/// Whether loaded halves are NaN (`I8`): max exponent with a
/// nonzero 112-bit fraction.
fn quad_is_nan(builder: &mut FunctionBuilder, exp: ClifValue, lo: ClifValue, hi: ClifValue) -> ClifValue {
    let k0 = q64(builder, 0);
    let k7fff = q64(builder, 0x7fff);
    let is_max = builder.ins().icmp(IntCC::Equal, exp, k7fff);
    let kfmask = q64(builder, 0x0000_ffff_ffff_ffff);
    let fhi = builder.ins().band(hi, kfmask);
    let lo_nz = builder.ins().icmp(IntCC::NotEqual, lo, k0);
    let hi_nz = builder.ins().icmp(IntCC::NotEqual, fhi, k0);
    let frac_nz = builder.ins().bor(lo_nz, hi_nz);
    builder.ins().band(is_max, frac_nz)
}

/// Whether loaded halves are a signed zero (`I8`).
fn quad_is_zero(builder: &mut FunctionBuilder, exp: ClifValue, lo: ClifValue, hi: ClifValue) -> ClifValue {
    let k0 = q64(builder, 0);
    let exp_z = builder.ins().icmp(IntCC::Equal, exp, k0);
    let kfmask = q64(builder, 0x0000_ffff_ffff_ffff);
    let fhi = builder.ins().band(hi, kfmask);
    let lo_z = builder.ins().icmp(IntCC::Equal, lo, k0);
    let hi_z = builder.ins().icmp(IntCC::Equal, fhi, k0);
    let frac_z = builder.ins().band(lo_z, hi_z);
    builder.ins().band(exp_z, frac_z)
}

/// Ordered equality as `I32` 0/1 (see [`ref_quad_eq`]).
pub(crate) fn quad_eq(builder: &mut FunctionBuilder, a_lo: ClifValue, a_hi: ClifValue, b_lo: ClifValue, b_hi: ClifValue) -> ClifValue {
    let (a_s, a_e, _, _) = quad_unpack(builder, a_lo, a_hi);
    let (b_s, b_e, _, _) = quad_unpack(builder, b_lo, b_hi);
    let nana = quad_is_nan(builder, a_e, a_lo, a_hi);
    let nanb = quad_is_nan(builder, b_e, b_lo, b_hi);
    let unordered = builder.ins().bor(nana, nanb);
    let cone8 = q8(builder, true);
    let ordered = builder.ins().bxor(unordered, cone8);
    let a_zero = quad_is_zero(builder, a_e, a_lo, a_hi);
    let b_zero = quad_is_zero(builder, b_e, b_lo, b_hi);
    let both_zero = builder.ins().band(a_zero, b_zero);
    let same_s = builder.ins().icmp(IntCC::Equal, a_s, b_s);
    let same_e = builder.ins().icmp(IntCC::Equal, a_e, b_e);
    let same_lo = builder.ins().icmp(IntCC::Equal, a_lo, b_lo);
    let same_hi = builder.ins().icmp(IntCC::Equal, a_hi, b_hi);
    let same_lh = builder.ins().band(same_lo, same_hi);
    let same_elh = builder.ins().band(same_e, same_lh);
    let same = builder.ins().band(same_s, same_elh);
    let same_or_zero = builder.ins().bor(both_zero, same);
    let eq = builder.ins().band(ordered, same_or_zero);
    builder.ins().uextend(types::I32, eq)
}

/// Ordered less-than as `I32` 0/1 (see [`ref_quad_lt`]).
pub(crate) fn quad_lt(builder: &mut FunctionBuilder, a_lo: ClifValue, a_hi: ClifValue, b_lo: ClifValue, b_hi: ClifValue) -> ClifValue {
    let (a_s, a_e, a_flo, a_fhi) = quad_unpack(builder, a_lo, a_hi);
    let (b_s, b_e, b_flo, b_fhi) = quad_unpack(builder, b_lo, b_hi);
    let nana = quad_is_nan(builder, a_e, a_lo, a_hi);
    let nanb = quad_is_nan(builder, b_e, b_lo, b_hi);
    let unordered = builder.ins().bor(nana, nanb);
    let cone8 = q8(builder, true);
    let ordered = builder.ins().bxor(unordered, cone8);
    let a_zero = quad_is_zero(builder, a_e, a_lo, a_hi);
    let b_zero = quad_is_zero(builder, b_e, b_lo, b_hi);
    let both_zero = builder.ins().band(a_zero, b_zero);
    let not_zero = builder.ins().bxor(both_zero, cone8);
    let k0 = q64(builder, 0);
    let signs_differ = builder.ins().icmp(IntCC::NotEqual, a_s, b_s);
    let a_neg = builder.ins().icmp(IntCC::NotEqual, a_s, k0);
    // Magnitude order: (exp, fhi, flo) lexicographic, unsigned.
    let ea_lt = builder.ins().icmp(IntCC::UnsignedLessThan, a_e, b_e);
    let ea_eq = builder.ins().icmp(IntCC::Equal, a_e, b_e);
    let ea_gt = builder.ins().icmp(IntCC::UnsignedLessThan, b_e, a_e);
    let ha_lt = builder.ins().icmp(IntCC::UnsignedLessThan, a_fhi, b_fhi);
    let ha_eq = builder.ins().icmp(IntCC::Equal, a_fhi, b_fhi);
    let ha_gt = builder.ins().icmp(IntCC::UnsignedLessThan, b_fhi, a_fhi);
    let lo_lt = builder.ins().icmp(IntCC::UnsignedLessThan, a_flo, b_flo);
    let m_lo_lt = builder.ins().band(ha_eq, lo_lt);
    let mag_lt_hi = builder.ins().bor(ha_lt, m_lo_lt);
    let ea_eq_m = builder.ins().band(ea_eq, mag_lt_hi);
    let mag_lt = builder.ins().bor(ea_lt, ea_eq_m);
    let lo_gt = builder.ins().icmp(IntCC::UnsignedLessThan, b_flo, a_flo);
    let m_lo_gt = builder.ins().band(ha_eq, lo_gt);
    let mag_gt_hi = builder.ins().bor(ha_gt, m_lo_gt);
    let ea_eq_g = builder.ins().band(ea_eq, mag_gt_hi);
    let mag_gt = builder.ins().bor(ea_gt, ea_eq_g);
    let same_sign = builder.ins().select(a_neg, mag_gt, mag_lt);
    let body = builder.ins().select(signs_differ, a_neg, same_sign);
    let both_nz = builder.ins().band(ordered, not_zero);
    let lt = builder.ins().band(both_nz, body);
    builder.ins().uextend(types::I32, lt)
}

/// Unordered (`I8`): either side NaN.
pub(crate) fn quad_unordered(builder: &mut FunctionBuilder, a_lo: ClifValue, a_hi: ClifValue, b_lo: ClifValue, b_hi: ClifValue) -> ClifValue {
    let (_, a_e, _, _) = quad_unpack(builder, a_lo, a_hi);
    let (_, b_e, _, _) = quad_unpack(builder, b_lo, b_hi);
    let nana = quad_is_nan(builder, a_e, a_lo, a_hi);
    let nanb = quad_is_nan(builder, b_e, b_lo, b_hi);
    builder.ins().bor(nana, nanb)
}

/// Ordered (`I8`): neither side NaN.
pub(crate) fn quad_ordered(builder: &mut FunctionBuilder, a_lo: ClifValue, a_hi: ClifValue, b_lo: ClifValue, b_hi: ClifValue) -> ClifValue {
    let unordered = quad_unordered(builder, a_lo, a_hi, b_lo, b_hi);
    let cone8 = q8(builder, true);
    builder.ins().bxor(unordered, cone8)
}

/// Which ordered comparison to lower over quad halves. Every case
/// derives from equality and less-than, computing only the side it
/// needs (mirrors the legacy backend's shape).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuadCmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Uno,
    Ord,
}

/// One ordered comparison over quad halves as `I32` 0/1.
pub(crate) fn quad_cmp(builder: &mut FunctionBuilder, op: QuadCmp, a_lo: ClifValue, a_hi: ClifValue, b_lo: ClifValue, b_hi: ClifValue) -> ClifValue {
    let one = builder.ins().iconst(types::I32, 1);
    match op {
        QuadCmp::Eq => quad_eq(builder, a_lo, a_hi, b_lo, b_hi),
        QuadCmp::Ne => {
            let eq = quad_eq(builder, a_lo, a_hi, b_lo, b_hi);
            builder.ins().bxor(eq, one)
        }
        QuadCmp::Lt => quad_lt(builder, a_lo, a_hi, b_lo, b_hi),
        QuadCmp::Le => {
            let eq = quad_eq(builder, a_lo, a_hi, b_lo, b_hi);
            let lt = quad_lt(builder, a_lo, a_hi, b_lo, b_hi);
            builder.ins().bor(lt, eq)
        }
        QuadCmp::Gt => quad_lt(builder, b_lo, b_hi, a_lo, a_hi),
        QuadCmp::Ge => {
            let eq = quad_eq(builder, a_lo, a_hi, b_lo, b_hi);
            let gt = quad_lt(builder, b_lo, b_hi, a_lo, a_hi);
            builder.ins().bor(gt, eq)
        }
        QuadCmp::Uno => {
            let unordered = quad_unordered(builder, a_lo, a_hi, b_lo, b_hi);
            builder.ins().uextend(types::I32, unordered)
        }
        QuadCmp::Ord => {
            let ordered = quad_ordered(builder, a_lo, a_hi, b_lo, b_hi);
            builder.ins().uextend(types::I32, ordered)
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use tests::*;
