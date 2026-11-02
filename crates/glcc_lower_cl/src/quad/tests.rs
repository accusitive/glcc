use super::*;

/// Quad exponent bias.
pub(crate) const Q_BIAS: i32 = 16383;

/// All-ones quad exponent: infinity and NaN.
pub(crate) const Q_MAX_EXP: u16 = 0x7fff;

/// Assemble halves into a u128.
#[cfg(test)]
fn join(lo: u64, hi: u64) -> u128 {
    ((hi as u128) << 64) | lo as u128
}

/// Split a u128 into halves.
#[cfg(test)]
fn split(v: u128) -> (u64, u64) {
    (v as u64, (v >> 64) as u64)
}

/// Pack sign/exp/fraction into halves.
#[cfg(test)]
fn pack(sign: u128, exp: u16, frac: u128) -> (u64, u64) {
    debug_assert!(sign <= 1);
    debug_assert!(exp <= Q_MAX_EXP);
    debug_assert!(frac >> 112 == 0);
    let hi = (sign << 63) | ((exp as u128) << 48) | (frac >> 64);
    split(join(frac as u64, hi as u64))
}

/// Unpack halves into sign, raw exp, and 112-bit fraction.
#[cfg(test)]
fn unpack(lo: u64, hi: u64) -> (u128, u16, u128) {
    let sign = (hi >> 63) as u128;
    let exp = ((hi >> 48) & 0x7fff) as u16;
    let frac = join(lo, hi & 0x0000_ffff_ffff_ffff);
    (sign, exp, frac)
}

/// Widen f64 bits to quad (exact; subnormals normalized, SNaN
/// quieted with payload shifted into place).
pub(crate) fn ref_f64_to_quad(v: u64) -> (u64, u64) {
    let sign = (v >> 63) as u128;
    let exp = ((v >> 52) & 0x7ff) as i32;
    let mant = (v & 0x000f_ffff_ffff_ffff) as u128;
    if exp == 0x7ff {
        if mant == 0 {
            return pack(sign, Q_MAX_EXP, 0);
        }
        // Payload shifts into place with its quiet bit untouched:
        // SNaN stays signaling (probed).
        return pack(sign, Q_MAX_EXP, mant << 60);
    }
    if exp == 0 {
        if mant == 0 {
            return pack(sign, 0, 0);
        }
        let lz = mant.leading_zeros() - 64;
        let e = 63 - lz as i32 - 1074;
        return pack(sign, (e + Q_BIAS) as u16, ((mant << lz) << 48) & !(1 << 111));
    }
    pack(sign, (exp + 15360) as u16, mant << 60)
}

/// Widen f32 bits to quad (same rules).
pub(crate) fn ref_f32_to_quad(v: u32) -> (u64, u64) {
    let v = v as u128;
    let sign = v >> 31;
    let exp = ((v >> 23) & 0xff) as i32;
    let mant = v & 0x7fffff;
    if exp == 0xff {
        if mant == 0 {
            return pack(sign, Q_MAX_EXP, 0);
        }
        return pack(sign, Q_MAX_EXP, mant << 89);
    }
    if exp == 0 {
        if mant == 0 {
            return pack(sign, 0, 0);
        }
        let lz = mant.leading_zeros() - 64;
        let e = 63 - lz as i32 - 149;
        return pack(sign, (e + Q_BIAS) as u16, ((mant << lz) << 48) & !(1 << 111));
    }
    pack(sign, (exp + 16256) as u16, mant << 89)
}

/// Integer to quad: `v` is the full 64-bit pattern (callers widen
/// sub-word first), two's complement when signed. Every u64/i64
/// value is exactly representable (113-bit significand).
pub(crate) fn ref_int_to_quad(v: u64, unsigned: bool) -> (u64, u64) {
    let sign = if unsigned || v >> 63 == 0 { 0 } else { 1 };
    let mag = if sign == 1 { v.wrapping_neg() as u128 } else { v as u128 };
    if mag == 0 {
        return (0, 0);
    }
    let lz = mag.leading_zeros();
    let e = 127 - lz as i32;
    // Fraction excludes the implicit leading 1: clear the top set
    // bit, then shift the remainder to bit 111.
    let top = 127 - lz;
    let frac = (mag - (1u128 << top)) << (112 - top);
    pack(sign, (e + Q_BIAS) as u16, frac)
}

/// Integer to quad from a full 128-bit pattern (see
/// ref_int_to_quad): magnitudes past 2^113 round half-even, with
/// renormalization on carry (e.g. `u128::MAX` rounds up to 2^128).
pub(crate) fn ref_int_to_quad128(v: u128, unsigned: bool) -> (u64, u64) {
    let sign = if unsigned || v >> 127 == 0 { 0 } else { 1 };
    let mag = if sign == 1 { v.wrapping_neg() } else { v };
    if mag == 0 {
        return (0, 0);
    }
    let lz = mag.leading_zeros();
    let e = 127 - lz as i32;
    let top = 127 - lz;
    if lz >= 15 {
        // Fits the 113-bit significand: exact like the 64-bit path.
        let frac = (mag - (1u128 << top)) << (112 - top);
        return pack(sign, (e + Q_BIAS) as u16, frac);
    }
    // Round the low bits half-even (GRS at `drop`).
    let drop = top - 112;
    let mut kept = mag >> drop;
    let guard = (mag >> (drop - 1)) & 1;
    let rest = mag & ((1u128 << (drop - 1)) - 1);
    if guard == 1 && (rest != 0 || kept & 1 == 1) {
        kept += 1;
    }
    if kept == 1 << 113 {
        // The integer rounded up to 2^(e+1): renormalize.
        return pack(sign, (e + 1 + Q_BIAS) as u16, 0);
    }
    pack(sign, (e + Q_BIAS) as u16, kept - (1u128 << 112))
}

/// Unbiased exponent, or `None` for zero/denormal (always
/// fraction-sized) — callers map those to zero directly.
#[cfg(test)]
fn unbiased(exp: u16) -> Option<i32> {
    if exp == 0 { None } else { Some(exp as i32 - Q_BIAS) }
}

/// Quad to signed 64: saturating truncation (in-range truncates
/// toward zero; positive overflow, +Inf, and NaN saturate to MAX;
/// negative overflow and -Inf to MIN; denormals are zero).
pub(crate) fn ref_quad_to_s64(lo: u64, hi: u64) -> u64 {
    let (sign, exp, frac) = unpack(lo, hi);
    // NaN saturates by sign exactly like infinity (probed); exp max
    // always overflows, so no special case is needed.
    let Some(e) = unbiased(exp) else {
        return 0;
    };
    if e < 0 {
        return 0;
    }
    if e >= 63 {
        return if sign == 1 { 1 << 63 } else { u64::MAX >> 1 };
    }
    // Normalized: reattach the implicit leading 1, then shift.
    let signif = (1u128 << 112) | frac;
    let mag = (signif >> (112 - e as u32)) as u64;
    if sign == 1 { mag.wrapping_neg() } else { mag }
}

/// Quad to signed 32 (same rules at 32 bits).
pub(crate) fn ref_quad_to_s32(lo: u64, hi: u64) -> u32 {
    let (sign, exp, frac) = unpack(lo, hi);
    let Some(e) = unbiased(exp) else {
        return 0;
    };
    if e < 0 {
        return 0;
    }
    if e >= 31 {
        return if sign == 1 { 1 << 31 } else { u32::MAX >> 1 };
    }
    let signif = (1u128 << 112) | frac;
    let mag = (signif >> (112 - e as u32)) as u64;
    (if sign == 1 { mag.wrapping_neg() } else { mag }) as u32
}

/// Quad to unsigned 64: saturating (negatives and -Inf go to 0,
/// not to `u64::MAX` via wrapping).
pub(crate) fn ref_quad_to_u64(lo: u64, hi: u64) -> u64 {
    let (sign, exp, frac) = unpack(lo, hi);
    if sign == 1 {
        return 0;
    }
    let Some(e) = unbiased(exp) else {
        return 0;
    };
    if e < 0 {
        return 0;
    }
    if e >= 64 {
        return u64::MAX;
    }
    let signif = (1u128 << 112) | frac;
    (signif >> (112 - e as u32)) as u64
}

/// Quad to unsigned 32 (same rules at 32 bits).
pub(crate) fn ref_quad_to_u32(lo: u64, hi: u64) -> u32 {
    let (sign, exp, frac) = unpack(lo, hi);
    if sign == 1 {
        return 0;
    }
    let Some(e) = unbiased(exp) else {
        return 0;
    };
    if e < 0 {
        return 0;
    }
    if e >= 32 {
        return u32::MAX;
    }
    let signif = (1u128 << 112) | frac;
    (signif >> (112 - e as u32)) as u32
}

/// Quad to signed 128: saturating truncation (same rules at 128
/// bits; `-2^127` is exact through the saturation path, like the
/// 64-bit `-2^63`).
///
/// Note the boundary differs deliberately from the x87 path below:
/// positive `2^127` saturates here, while the vendored
/// `__glcc_fixxfti` (like clang's own lowering) yields the raw bit
/// pattern. Out-of-range float-to-int is UB in C; each flavor
/// matches its target toolchain instead of agreeing with the other.
pub(crate) fn ref_quad_to_s128(lo: u64, hi: u64) -> u128 {
    let (sign, exp, frac) = unpack(lo, hi);
    let Some(e) = unbiased(exp) else {
        return 0;
    };
    if e < 0 {
        return 0;
    }
    if e >= 127 {
        return if sign == 1 { 1 << 127 } else { u128::MAX >> 1 };
    }
    let signif = (1u128 << 112) | frac;
    // E past 112 shifts left instead of right.
    let mag = if e <= 112 { signif >> (112 - e as u32) } else { signif << (e as u32 - 112) };
    if sign == 1 { mag.wrapping_neg() } else { mag }
}

/// Quad to unsigned 128 (same rules at 128 bits).
pub(crate) fn ref_quad_to_u128(lo: u64, hi: u64) -> u128 {
    let (sign, exp, frac) = unpack(lo, hi);
    if sign == 1 {
        return 0;
    }
    let Some(e) = unbiased(exp) else {
        return 0;
    };
    if e < 0 {
        return 0;
    }
    if e >= 128 {
        return u128::MAX;
    }
    let signif = (1u128 << 112) | frac;
    if e <= 112 { signif >> (112 - e as u32) } else { signif << (e as u32 - 112) }
}

/// Whether a quad value is NaN.
pub(crate) fn ref_quad_is_nan(lo: u64, hi: u64) -> bool {
    let (_, exp, frac) = unpack(lo, hi);
    exp == Q_MAX_EXP && frac != 0
}

/// Whether a quad value is a signed zero.
pub(crate) fn ref_quad_is_zero(lo: u64, hi: u64) -> bool {
    let (_, exp, frac) = unpack(lo, hi);
    exp == 0 && frac == 0
}

/// Ordered equality: NaN never equals, signed zeros compare equal.
pub(crate) fn ref_quad_eq(a_lo: u64, a_hi: u64, b_lo: u64, b_hi: u64) -> bool {
    if ref_quad_is_nan(a_lo, a_hi) || ref_quad_is_nan(b_lo, b_hi) {
        return false;
    }
    if ref_quad_is_zero(a_lo, a_hi) && ref_quad_is_zero(b_lo, b_hi) {
        return true;
    }
    a_lo == b_lo && a_hi == b_hi
}

/// Ordered less-than, by sign then magnitude.
pub(crate) fn ref_quad_lt(a_lo: u64, a_hi: u64, b_lo: u64, b_hi: u64) -> bool {
    if ref_quad_is_nan(a_lo, a_hi) || ref_quad_is_nan(b_lo, b_hi) {
        return false;
    }
    if ref_quad_is_zero(a_lo, a_hi) && ref_quad_is_zero(b_lo, b_hi) {
        return false;
    }
    let (sa, _, _) = unpack(a_lo, a_hi);
    let (sb, _, _) = unpack(b_lo, b_hi);
    if sa != sb {
        return sa == 1;
    }
    // Same sign: compare magnitude (exp, then full fraction).
    let ea = (a_hi >> 48) & 0x7fff;
    let eb = (b_hi >> 48) & 0x7fff;
    let fa = join(a_lo, a_hi & 0x0000_ffff_ffff_ffff);
    let fb = join(b_lo, b_hi & 0x0000_ffff_ffff_ffff);
    let ord = (ea, fa).cmp(&(eb, fb));
    if sa == 0 { ord == std::cmp::Ordering::Less } else { ord == std::cmp::Ordering::Greater }
}

/// Ceiling as halves: smallest integral value `>= x`. NaN, infinities,
/// signed zeros, and already-integral values return bit-identical;
/// `|x| < 1` maps to `1`/`-0` by sign; otherwise truncate the
/// fraction and increment positives when inexact (renormalizing on
/// carry, e.g. `ceil(1.5) == 2`).
pub(crate) fn ref_quad_ceil(lo: u64, hi: u64) -> (u64, u64) {
    let (sign, exp, frac) = unpack(lo, hi);
    if exp == Q_MAX_EXP || (exp == 0 && frac == 0) {
        return (lo, hi);
    }
    if exp == 0 || (exp as i32 - Q_BIAS) < 0 {
        if sign == 0 {
            return pack(0, Q_BIAS as u16, 0);
        }
        return pack(1, 0, 0);
    }
    let e = exp as i32 - Q_BIAS;
    if e >= 112 {
        return (lo, hi);
    }
    let k = (112 - e) as u32;
    let mask = (1u128 << k) - 1;
    if frac & mask == 0 {
        return (lo, hi);
    }
    let trunc = frac & !mask;
    if sign == 0 {
        let rounded = trunc + (1u128 << k);
        if rounded >> 112 != 0 {
            return pack(0, exp + 1, 0);
        }
        return pack(0, exp, rounded);
    }
    pack(sign, exp, trunc)
}

/// Floor as halves: largest integral value `<= x`. Mirror of
/// ref_quad_ceil: `|x| < 1` maps to `+0`/`-1`; otherwise
/// truncate and decrement (increment the magnitude of) negatives
/// when inexact.
pub(crate) fn ref_quad_floor(lo: u64, hi: u64) -> (u64, u64) {
    let (sign, exp, frac) = unpack(lo, hi);
    if exp == Q_MAX_EXP || (exp == 0 && frac == 0) {
        return (lo, hi);
    }
    if exp == 0 || (exp as i32 - Q_BIAS) < 0 {
        if sign == 0 {
            return pack(0, 0, 0);
        }
        return pack(1, Q_BIAS as u16, 0);
    }
    let e = exp as i32 - Q_BIAS;
    if e >= 112 {
        return (lo, hi);
    }
    let k = (112 - e) as u32;
    let mask = (1u128 << k) - 1;
    if frac & mask == 0 {
        return (lo, hi);
    }
    let trunc = frac & !mask;
    if sign == 1 {
        let rounded = trunc + (1u128 << k);
        if rounded >> 112 != 0 {
            return pack(1, exp + 1, 0);
        }
        return pack(1, exp, rounded);
    }
    pack(sign, exp, trunc)
}

/// Round a 113-bit significand (with its GRS bits) to a narrower
/// float: `kept_bits` result bits (53 for double, 24 for single),
/// `drop` low bits discarded. Returns the rounded significand
/// (possibly carrying into one extra bit, which the caller
/// renormalizes).
#[cfg(test)]
fn round_grs(signif: u128, drop: u32, kept_bits: u32) -> u128 {
    debug_assert!(drop + kept_bits == 113);
    let kept = signif >> drop;
    let guard = (signif >> (drop - 1)) & 1;
    let rest = signif & ((1 << (drop - 1)) - 1);
    let lsb = kept & 1;
    if guard == 1 && (rest != 0 || lsb == 1) { kept + 1 } else { kept }
}

/// Quad to f64 bits: round-half-even; every NaN forces the quiet
/// bit and truncates its payload, overflow to infinity, tiny to
/// subnormal/zero.
pub(crate) fn ref_quad_to_f64(lo: u64, hi: u64) -> u64 {
    let sign = hi >> 63;
    let exp = ((hi >> 48) & 0x7fff) as i32;
    let frac = join(lo, hi & 0x0000_ffff_ffff_ffff);
    if exp == 0x7fff {
        if frac == 0 {
            return (sign << 63) | (0x7ff << 52);
        }
        // Any NaN (quiet or signaling): force the quiet bit, truncate
        // the payload (probed: SNaN payload bits survive shifting).
        return (sign << 63) | (0x7ff << 52) | (1 << 51) | (((frac >> 60) as u64) & 0x000f_ffff_ffff_ffff);
    }
    if exp == 0 {
        // Denormal quad (or zero): far below double range.
        return sign << 63;
    }
    let e64 = exp - Q_BIAS + 1023;
    let signif = (1u128 << 112) | frac;
    if e64 >= 0x7ff {
        return (sign << 63) | (0x7ff << 52);
    }
    if e64 >= 1 {
        let mut kept = round_grs(signif, 60, 53);
        let mut e = e64;
        if kept == 1 << 53 {
            kept >>= 1;
            e += 1;
            if e >= 0x7ff {
                return (sign << 63) | (0x7ff << 52);
            }
        }
        return (sign << 63) | ((e as u64) << 52) | ((kept as u64) & 0x000f_ffff_ffff_ffff);
    }
    // Subnormal: shift the whole significand down, rounding once.
    // Rounding can carry into the smallest normal (E64 == 1).
    let s = (1 - e64) as u32;
    let total = 60 + s;
    if total > 113 {
        return sign << 63;
    }
    let r = round_grs(signif, total, 53 - s);
    if r == 1 << 52 {
        return (sign << 63) | ((1u64) << 52);
    }
    (sign << 63) | ((r as u64) & 0x000f_ffff_ffff_ffff)
}

/// Quad to f32 bits (same rules at single precision).
pub(crate) fn ref_quad_to_f32(lo: u64, hi: u64) -> u32 {
    let sign = hi >> 63;
    let exp = ((hi >> 48) & 0x7fff) as i32;
    let frac = join(lo, hi & 0x0000_ffff_ffff_ffff);
    if exp == 0x7fff {
        if frac == 0 {
            return (sign << 63) as u32 | (0xff << 23);
        }
        return (sign << 63) as u32 | (0xff << 23) | (1 << 22) | ((((frac >> 89) as u64) & 0x007f_ffff) as u32);
    }
    if exp == 0 {
        return (sign << 63) as u32;
    }
    let e32 = exp - Q_BIAS + 127;
    let signif = (1u128 << 112) | frac;
    if e32 >= 0xff {
        return (sign << 63) as u32 | (0xff << 23);
    }
    if e32 >= 1 {
        let mut kept = round_grs(signif, 89, 24);
        let mut e = e32;
        if kept == 1 << 24 {
            kept >>= 1;
            e += 1;
            if e >= 0xff {
                return (sign << 63) as u32 | (0xff << 23);
            }
        }
        return (sign << 63) as u32 | ((e as u32) << 23) | ((kept as u32) & 0x007f_ffff);
    }
    let s = (1 - e32) as u32;
    let total = 89 + s;
    if total > 113 {
        return (sign << 63) as u32;
    }
    let r = round_grs(signif, total, 24 - s);
    if r == 1 << 23 {
        return (sign << 63) as u32 | (1u32 << 23);
    }
    (sign << 63) as u32 | ((r as u32) & 0x007f_ffff)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quad of an f64 value, via the f64 bits (oracle-checked).
    fn qf(v: u64) -> (u64, u64) {
        ref_f64_to_quad(v)
    }

    #[test]
    fn f64_widening_vectors() {
        let cases: &[(u64, u64, u64)] = &[
            (0x0000_0000_0000_0000, 0x0000_0000_0000_0000, 0x0000_0000_0000_0000),
            (0x8000_0000_0000_0000, 0x0000_0000_0000_0000, 0x8000_0000_0000_0000),
            (0x3ff8_0000_0000_0000, 0x0000_0000_0000_0000, 0x3fff_8000_0000_0000), // 1.5
            (0x3fb9_9999_9999_999a, 0xa000_0000_0000_0000, 0x3ffb_9999_9999_9999), // 0.1
            (0x7feF_ffff_ffff_ffff, 0xf000_0000_0000_0000, 0x43fe_ffff_ffff_ffff), // max
            (0x0010_0000_0000_0000, 0x0000_0000_0000_0000, 0x3c01_0000_0000_0000), // min normal
            (0x0000_0000_0000_0001, 0x0000_0000_0000_0000, 0x3bcd_0000_0000_0000), // min subnormal
            (0x7ff0_0000_0000_0000, 0x0000_0000_0000_0000, 0x7fff_0000_0000_0000), // inf
            (0xfff0_0000_0000_0000, 0x0000_0000_0000_0000, 0xffff_0000_0000_0000), // -inf
            (0x7ff8_0000_0000_0000, 0x0000_0000_0000_0000, 0x7fff_8000_0000_0000), // qnan
            (0x7ff0_0000_0000_0001, 0x1000_0000_0000_0000, 0x7fff_0000_0000_0000), // snan, still signaling
        ];
        for (v, lo, hi) in cases {
            assert_eq!(ref_f64_to_quad(*v), (*lo, *hi), "f64 {v:#x}");
        }
    }

    #[test]
    fn f32_widening_vectors() {
        let cases: &[(u32, u64, u64)] = &[
            (0x0000_0000, 0x0000_0000_0000_0000, 0x0000_0000_0000_0000),
            (0x8000_0000, 0x0000_0000_0000_0000, 0x8000_0000_0000_0000),
            (0x3fc0_0000, 0x0000_0000_0000_0000, 0x3fff_8000_0000_0000), // 1.5f
            (0x3dcc_cccd, 0x0000_0000_0000_0000, 0x3ffb_9999_9a00_0000), // 0.1f
            (0x0080_0000, 0x0000_0000_0000_0000, 0x3f81_0000_0000_0000), // min normal
            (0x0000_0001, 0x0000_0000_0000_0000, 0x3f6a_0000_0000_0000), // min subnormal
            (0x7f80_0000, 0x0000_0000_0000_0000, 0x7fff_0000_0000_0000), // inf
            (0x7fc0_0000, 0x0000_0000_0000_0000, 0x7fff_8000_0000_0000), // qnan
            (0x7f80_0001, 0x0000_0000_0000_0000, 0x7fff_0000_0200_0000), // snan, still signaling
        ];
        for (v, lo, hi) in cases {
            assert_eq!(ref_f32_to_quad(*v), (*lo, *hi), "f32 {v:#x}");
        }
    }

    #[test]
    fn int_widening_vectors() {
        let cases: &[(u64, bool, u64, u64)] = &[
            (0, false, 0x0000_0000_0000_0000, 0x0000_0000_0000_0000),
            (1, false, 0x0000_0000_0000_0000, 0x3fff_0000_0000_0000),
            (0xffff_ffff_ffff_ffff, false, 0x0000_0000_0000_0000, 0xbfff_0000_0000_0000), // -1
            (0x7fff_ffff_ffff_ffff, false, 0xfffc_0000_0000_0000, 0x403d_ffff_ffff_ffff), // i64max
            (0x8000_0000_0000_0000, false, 0x0000_0000_0000_0000, 0xc03e_0000_0000_0000), // i64min
            (0xffff_ffff_ffff_ffff, true, 0xfffe_0000_0000_0000, 0x403e_ffff_ffff_ffff),  // u64max
            (0x8000_0000_0000_0000, true, 0x0000_0000_0000_0000, 0x403e_0000_0000_0000),  // 2^63
        ];
        for (v, unsigned, lo, hi) in cases {
            assert_eq!(ref_int_to_quad(*v, *unsigned), (*lo, *hi), "int {v:#x}");
        }
    }

    #[test]
    fn int128_widening_vectors() {
        let cases: &[(u128, bool, u64, u64)] = &[
            (0, false, 0x0000_0000_0000_0000, 0x0000_0000_0000_0000),
            (0, true, 0x0000_0000_0000_0000, 0x0000_0000_0000_0000),
            (1, false, 0x0000_0000_0000_0000, 0x3fff_0000_0000_0000),
            (u128::MAX, false, 0x0000_0000_0000_0000, 0xbfff_0000_0000_0000),      // -1
            (u128::MAX, true, 0x0000_0000_0000_0000, 0x407f_0000_0000_0000),       // 2^128
            (1 << 127, false, 0x0000_0000_0000_0000, 0xc07e_0000_0000_0000),       // i128::MIN
            (1 << 127, true, 0x0000_0000_0000_0000, 0x407e_0000_0000_0000),        // 2^127
            ((1 << 113) - 1, false, 0xffff_ffff_ffff_ffff, 0x406f_ffff_ffff_ffff), // max exact
            (1 << 113, false, 0x0000_0000_0000_0000, 0x4070_0000_0000_0000),
            ((1 << 113) + 1, false, 0x0000_0000_0000_0000, 0x4070_0000_0000_0000), // half-even down
            ((1 << 113) + 3, false, 0x0000_0000_0000_0002, 0x4070_0000_0000_0000), // half-even up
            ((1 << 64) - 1, true, 0xfffe_0000_0000_0000, 0x403e_ffff_ffff_ffff),
            (1 << 64, true, 0x0000_0000_0000_0000, 0x403f_0000_0000_0000),
        ];
        for (v, unsigned, lo, hi) in cases {
            assert_eq!(ref_int_to_quad128(*v, *unsigned), (*lo, *hi), "int128 {v:#x}");
        }
    }

    #[test]
    fn int_narrowing_vectors() {
        // Saturating truncation, probed on both targets.
        assert_eq!(ref_quad_to_s64(0, 0x3fff_0000_0000_0000), 1); // 1.0
        assert_eq!(ref_quad_to_s64(0, 0xbfff_0000_0000_0000) as i64, -1);
        assert_eq!(ref_quad_to_s64(0, 0x403e_0000_0000_0000), i64::MAX as u64); // 2^63 saturates
        assert_eq!(ref_quad_to_s64(0, 0xc03e_0000_0000_0000), 1 << 63); // -2^63 exact
        assert_eq!(ref_quad_to_s64(0, 0x7fff_0000_0000_0000), i64::MAX as u64); // +inf
        assert_eq!(ref_quad_to_s64(0, 0xffff_0000_0000_0000), 1 << 63); // -inf
        assert_eq!(ref_quad_to_s64(1, 0x7fff_8000_0000_0000), i64::MAX as u64); // +nan
        assert_eq!(ref_quad_to_s64(1, 0xffff_8000_0000_0000), 1 << 63); // -nan saturates by sign
        assert_eq!(ref_quad_to_u64(0, 0x403e_0000_0000_0000), 1 << 63); // 2^63 exact
        assert_eq!(ref_quad_to_u64(0xffff_ffff_ffff_ffff, 0x403e_ffff_ffff_ffff), u64::MAX); // 2^64-1
        assert_eq!(ref_quad_to_u64(0, 0x403f_0000_0000_0000), u64::MAX); // 2^64 saturates
        assert_eq!(ref_quad_to_u64(0, 0xbfff_0000_0000_0000), 0); // -1.0
        assert_eq!(ref_quad_to_u64(1, 0xffff_8000_0000_0000), 0); // -nan
        assert_eq!(ref_quad_to_s32(0, 0x4020_0000_0000_0000), i32::MAX as u32); // 2^31 saturates
        assert_eq!(ref_quad_to_s32(0, 0xc020_0000_0000_0000), 1 << 31); // -2^31 exact
        assert_eq!(ref_quad_to_u32(0, 0x401f_ffff_ffff_ffff), u32::MAX); // 2^32-1 exact-ish
        // Sub-word narrows the saturated 32-bit result.
        assert_eq!(ref_quad_to_u32(0, 0x403e_0000_0000_0000) as u16, 0xffff); // huge -> MAX -> narrow
        assert_eq!(ref_quad_to_u32(0, 0x403e_0000_0000_0000) as u16, 0xffff);
        // 128-bit saturation follows the same rules at 127/128.
        assert_eq!(ref_quad_to_s128(0, 0x3fff_0000_0000_0000), 1); // 1.0
        assert_eq!(ref_quad_to_s128(0, 0xbfff_0000_0000_0000), u128::MAX); // -1.0
        assert_eq!(ref_quad_to_s128(0, 0x407e_0000_0000_0000), u128::MAX >> 1); // 2^127 saturates
        assert_eq!(ref_quad_to_s128(0, 0xc07e_0000_0000_0000), 1 << 127); // -2^127 exact
        assert_eq!(ref_quad_to_s128(0, 0x7fff_0000_0000_0000), u128::MAX >> 1); // +inf
        assert_eq!(ref_quad_to_s128(0, 0xffff_0000_0000_0000), 1 << 127); // -inf
        assert_eq!(ref_quad_to_s128(1, 0x7fff_8000_0000_0000), u128::MAX >> 1); // +nan
        assert_eq!(ref_quad_to_u128(0, 0x407e_0000_0000_0000), 1 << 127); // 2^127 exact
        assert_eq!(ref_quad_to_u128(0, 0x407f_0000_0000_0000), u128::MAX); // 2^128 saturates
        assert_eq!(ref_quad_to_u128(0, 0xbfff_0000_0000_0000), 0); // -1.0
        assert_eq!(ref_quad_to_u128(0, 0x3ffe_0000_0000_0000), 0); // 0.5 truncates
    }

    #[test]
    fn narrowing_vectors() {
        // (quad lo/hi, f64 bits, f32 bits), probed.
        let one = (0u64, 0x3fff_0000_0000_0000u64);
        assert_eq!(ref_quad_to_f64(one.0, one.1), 0x3ff0_0000_0000_0000);
        assert_eq!(ref_quad_to_f32(one.0, one.1), 0x3f80_0000);
        // Halfway double (1 + 2^-53, even lsb) rounds down.
        let half = (1u64 << 59, 0x3fff_0000_0000_0000u64);
        assert_eq!(ref_quad_to_f64(half.0, half.1), 0x3ff0_0000_0000_0000);
        // Halfway + sticky rounds up.
        let above = ((1u64 << 59) | (1u64 << 52), 0x3fff_0000_0000_0000u64);
        assert_eq!(ref_quad_to_f64(above.0, above.1), 0x3ff0_0000_0000_0001);
        // Quad max overflows to infinity.
        let big = (0xffff_ffff_ffff_ffffu64, 0x7ffe_ffff_ffff_ffffu64);
        assert_eq!(ref_quad_to_f64(big.0, big.1), 0x7ff0_0000_0000_0000);
        assert_eq!(ref_quad_to_f32(big.0, big.1), 0x7f80_0000);
        // Tiny normal underflows to zero.
        let tiny = (0u64, 0x0001_0000_0000_0000u64);
        assert_eq!(ref_quad_to_f64(tiny.0, tiny.1), 0);
        // SNaN with surviving payload: quieted + truncated (probed).
        assert_eq!(ref_quad_to_f64(1, 0x7fff_0000_0000_0000), 0x7ff8_0000_0000_0000);
        assert_eq!(ref_quad_to_f64(0x1234_5678_9abc_def0, 0x7fff_badc_0ffe_e0d0), 0x7ffb_adc0_ffee_0d01);
        assert_eq!(ref_quad_to_f32(0x1234_5678_9abc_def0, 0x7fff_badc_0ffe_e0d0), 0x7fdd_6e07);
    }

    #[test]
    fn comparison_vectors() {
        let one = qf(0x3ff0_0000_0000_0000);
        let two = qf(0x4000_0000_0000_0000);
        let negone = qf(0xbff0_0000_0000_0000);
        let pzero = (0u64, 0u64);
        let nzero = (0u64, 0x8000_0000_0000_0000u64);
        let inf = qf(0x7ff0_0000_0000_0000);
        let nan = qf(0x7ff8_0000_0000_0000);
        assert!(ref_quad_eq(one.0, one.1, one.0, one.1));
        assert!(!ref_quad_eq(one.0, one.1, two.0, two.1));
        assert!(ref_quad_eq(pzero.0, pzero.1, nzero.0, nzero.1));
        assert!(!ref_quad_eq(nan.0, nan.1, nan.0, nan.1));
        assert!(ref_quad_lt(one.0, one.1, two.0, two.1));
        assert!(ref_quad_lt(negone.0, negone.1, one.0, one.1));
        assert!(ref_quad_lt(negone.0, negone.1, nzero.0, nzero.1));
        assert!(!ref_quad_lt(one.0, one.1, one.0, one.1));
        assert!(!ref_quad_lt(pzero.0, pzero.1, nzero.0, nzero.1));
        assert!(ref_quad_lt(one.0, one.1, inf.0, inf.1));
        assert!(!ref_quad_lt(nan.0, nan.1, one.0, one.1));
    }

    #[test]
    fn rounding_vectors() {
        let q = ref_f64_to_quad;
        let one = q(0x3ff0_0000_0000_0000);
        let negone = q(0xbff0_0000_0000_0000);
        let pzero = (0u64, 0u64);
        let nzero = (0u64, 0x8000_0000_0000_0000u64);
        let inf = q(0x7ff0_0000_0000_0000);
        let nan = q(0x7ff8_0000_0000_0000);
        let minus_one = pack(1, Q_BIAS as u16, 0);
        let plus_one = pack(0, Q_BIAS as u16, 0);
        let plus_two = q(0x4000_0000_0000_0000);
        // Integral passthrough, including signed zeros and specials.
        assert_eq!(ref_quad_ceil(pzero.0, pzero.1), pzero);
        assert_eq!(ref_quad_ceil(nzero.0, nzero.1), nzero);
        assert_eq!(ref_quad_floor(pzero.0, pzero.1), pzero);
        assert_eq!(ref_quad_floor(nzero.0, nzero.1), nzero);
        assert_eq!(ref_quad_ceil(one.0, one.1), one);
        assert_eq!(ref_quad_floor(one.0, one.1), one);
        assert_eq!(ref_quad_ceil(inf.0, inf.1), inf);
        assert_eq!(ref_quad_floor(nan.0, nan.1), nan);
        // |x| < 1.
        let half = q(0x3fe0_0000_0000_0000);
        let neghalf = q(0xbfe0_0000_0000_0000);
        assert_eq!(ref_quad_ceil(half.0, half.1), plus_one);
        assert_eq!(ref_quad_ceil(neghalf.0, neghalf.1), nzero);
        assert_eq!(ref_quad_floor(half.0, half.1), pzero);
        assert_eq!(ref_quad_floor(neghalf.0, neghalf.1), minus_one);
        // Denormal nonzero behaves like |x| < 1.
        let tzx = (1u64, 0u64);
        let tzn = (1u64, 0x8000_0000_0000_0000u64);
        assert_eq!(ref_quad_ceil(tzx.0, tzx.1), plus_one);
        assert_eq!(ref_quad_ceil(tzn.0, tzn.1), nzero);
        assert_eq!(ref_quad_floor(tzx.0, tzx.1), pzero);
        assert_eq!(ref_quad_floor(tzn.0, tzn.1), minus_one);
        // Fractions with carry into a new exponent.
        let onehalf = q(0x3ff8_0000_0000_0000);
        assert_eq!(ref_quad_ceil(onehalf.0, onehalf.1), plus_two);
        assert_eq!(ref_quad_floor(onehalf.0, onehalf.1), one);
        let negonehalf = q(0xbff8_0000_0000_0000);
        assert_eq!(ref_quad_ceil(negonehalf.0, negonehalf.1), negone);
        assert_eq!(ref_quad_floor(negonehalf.0, negonehalf.1), q(0xc000_0000_0000_0000));
        // Large integral values pass through (e >= 112).
        let big = (0u64, 0x403e_0000_0000_0000u64);
        assert_eq!(ref_quad_ceil(big.0, big.1), big);
        assert_eq!(ref_quad_floor(big.0, big.1), big);
        // Cross-check oracles against f64 for exactly-representable inputs.
        for v in [
            0x3ff0_0000_0000_0000u64,
            0x3ff8_0000_0000_0000,
            0xbff8_0000_0000_0000,
            0x3fe0_0000_0000_0000,
            0x40c8_0000_0000_0000,
            0xc0c8_0000_0000_0000,
        ] {
            let (lo, hi) = q(v);
            let f = f64::from_bits(v);
            let (elo, ehi) = q(f.ceil().to_bits());
            let (flo, fhi) = q(f.floor().to_bits());
            assert_eq!(ref_quad_ceil(lo, hi), (elo, ehi), "ceil of {v:#x}");
            assert_eq!(ref_quad_floor(lo, hi), (flo, fhi), "floor of {v:#x}");
        }
    }
}

