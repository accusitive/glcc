//! Magic multipliers for division by a constant.
//!
//! Both forms expand to a quotient sequence computed in a type with
//! twice the division's width, where the products never wrap:
//!
//! Unsigned: with `l` the bit length of `d` and `m = ceil(2^(n+l)/d)`,
//! `floor(x*m/2^(n+l))` equals `x/d` for every `n`-bit `x`, because the
//! multiplier's rounding error stays below `2^-l * x/(d*2^n) < 1/d`.
//! `m` has its top bit at `2^n`, and the low bits `e = m - 2^n` alone
//! give the quotient: the wide product `x*e`, shifted right by `n`,
//! added to `x`, and shifted right by `l`, reconstructs the high part
//! of `x*m` exactly.
//!
//! Signed: with `s` the smallest shift whose rounding error
//! `m*d - 2^(n+s)` is below `2^s`, `floor(x*m/2^(n+s))` equals the
//! toward-zero quotient for `x >= 0` and one less for `x < 0`, so the
//! sequence adds the shifted-in sign bit back. The multiplier here is
//! `e = m - 2^n`, which may be negative; the wide product is
//! interpreted signed.

/// The multiplier `e` and shift for unsigned division by `d`:
/// `(x + (x*e >> n)) >> shift`, truncated to `n` bits, is `x / d` for
/// every `n`-bit `x`. `d` must be at least two and fit in `n` bits; a
/// power of two takes the shift rewrite instead.
pub(super) fn unsigned_magic(d: u128, n: u32) -> Option<(u128, u32)> {
    if d < 2 || n == 0 || n > 64 || d >= 1u128 << n || d.count_ones() == 1 {
        return None;
    }
    let shift = bit_length(d);
    let m = ceil_pow2_over(n + shift, d);
    let e = m - (1u128 << n);
    (e != 0 && e < 1u128 << n).then_some((e, shift))
}

/// The multiplier `e` and shift for signed division by the positive
/// constant `d`: `(x + (x*e >>s n)) >>s shift`, minus `(x >>s 2n-1)`,
/// truncated to `n` bits, is `x / d` for every signed `n`-bit `x`. `e`
/// may be negative and the wide product is interpreted signed. `d` must
/// be at least two; a power of two takes the shift rewrite instead.
pub(super) fn signed_magic(d: i128, n: u32) -> Option<(i128, u32)> {
    if d < 2 || n == 0 || n > 64 || d >= 1i128 << (n - 1) || d.count_ones() == 1 {
        return None;
    }
    let d = d as u128;
    // The error condition always holds by `s = bit_length(d)`: the error
    // is at most `d - 1`, below `2^s`.
    for shift in 1..=bit_length(d) {
        let p = n + shift;
        debug_assert!(p < 128, "signed divisors keep the exponent below 128");
        let m = ceil_pow2_over(p, d);
        let error = m * d - (1u128 << p);
        if error < 1u128 << shift {
            return Some((m as i128 - (1i128 << n), shift));
        }
    }
    None
}

fn bit_length(d: u128) -> u32 {
    128 - (d - 1).leading_zeros()
}

/// `ceil(2^p / d)` for a `d` that does not divide `2^p`, valid up to
/// `p == 128` because `d` is not a power of two.
fn ceil_pow2_over(p: u32, d: u128) -> u128 {
    let numerator = if p >= 128 { u128::MAX } else { (1u128 << p) - 1 };
    numerator / d + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Emulate the emitted sequence with wrapping arithmetic at twice
    /// `n`'s width, the way the backend defines it.
    fn run_unsigned(x: u128, e: u128, shift: u32, n: u32) -> u128 {
        let wide = 2 * n;
        let m = if wide >= 128 { u128::MAX } else { (1u128 << wide) - 1 };
        let xw = x & m;
        let product = xw.wrapping_mul(e) & m;
        let sum = xw.wrapping_add(product >> n) & m;
        (sum >> shift) & ((1u128 << n) - 1)
    }

    fn run_signed(x: i128, e: i128, shift: u32, n: u32) -> i128 {
        let wide = 2 * n;
        let m = if wide >= 128 { u128::MAX } else { (1u128 << wide) - 1 };
        let wrap = |v: i128| -> i128 {
            let t = (v as u128) & m;
            if t >> (wide - 1) & 1 == 1 { (t | !m) as i128 } else { t as i128 }
        };
        let ashr = |v: i128, amount: u32| -> i128 { wrap(v) >> amount };
        let xw = wrap(x);
        let product = wrap(xw.wrapping_mul(e));
        let sum = wrap(xw.wrapping_add(ashr(product, n)));
        let biased = ashr(sum, shift);
        wrap(biased.wrapping_sub(ashr(xw, wide - 1)))
    }

    #[test]
    fn unsigned_magic_covers_every_eight_bit_division() {
        for d in 2u128..1 << 8 {
            if d.count_ones() == 1 {
                continue;
            }
            let (e, shift) = unsigned_magic(d, 8).unwrap_or_else(|| panic!("no magic for {d}"));
            for x in 0u128..1 << 8 {
                assert_eq!(run_unsigned(x, e, shift, 8), x / d, "{x} / {d}");
            }
        }
    }

    #[test]
    fn signed_magic_covers_every_eight_bit_division() {
        for d in 2i128..1 << 7 {
            if d.count_ones() == 1 {
                continue;
            }
            let (e, shift) = signed_magic(d, 8).unwrap_or_else(|| panic!("no magic for {d}"));
            for x in -(1i128 << 7)..(1i128 << 7) {
                assert_eq!(run_signed(x, e, shift, 8), x.wrapping_div(d), "{x} / {d}");
            }
        }
    }

    #[test]
    fn magic_covers_the_scalar_widths_at_the_boundaries() {
        let unsigned_edges = |d: u128, n: u32| {
            let (e, shift) = unsigned_magic(d, n).unwrap_or_else(|| panic!("no magic u{n} / {d}"));
            let span = 1u128 << n;
            let edges = [0, 1, d - 1, d, d + 1, span / d * d, span / d * d - 1, span - 1, span / 2, span / 2 + 1];
            for x in edges.into_iter().filter(|x: &u128| *x < span) {
                assert_eq!(run_unsigned(x, e, shift, n), x / d, "u{n}: {x} / {d}");
            }
        };
        for n in [8, 16, 32, 64] {
            for d in [3u128, 7, 10, 100, (1u128 << n) / 3, (1u128 << n) - 1] {
                if d < 1u128 << n && d.count_ones() != 1 {
                    unsigned_edges(d, n);
                }
            }
        }

        let signed_edges = |d: i128, n: u32| {
            let (e, shift) = signed_magic(d, n).unwrap_or_else(|| panic!("no magic s{n} / {d}"));
            let half = 1i128 << (n - 1);
            let edges = [0, 1, -1, d - 1, 1 - d, d, -d, d + 1, -d - 1, half / d * d, -(half / d * d), half - 1, -half];
            for x in edges.into_iter().filter(|x: &i128| x.abs() <= half) {
                assert_eq!(run_signed(x, e, shift, n), x.wrapping_div(d), "s{n}: {x} / {d}");
            }
        };
        for n in [8, 16, 32, 64] {
            for d in [3i128, 7, 10, 100, (1i128 << (n - 1)) / 3, (1i128 << (n - 1)) - 1] {
                if d.count_ones() != 1 {
                    signed_edges(d, n);
                }
            }
        }
    }

    #[test]
    fn powers_of_two_and_small_divisors_are_rejected() {
        for d in [0u128, 1, 2, 4, 1 << 63, 1 << 64] {
            assert_eq!(unsigned_magic(d, 64), None, "{d}");
        }
        assert_eq!(unsigned_magic(10, 3), None, "divisor exceeds the width");
        assert_eq!(unsigned_magic((1u128 << 64) - 1, 64).map(|(_, s)| s), Some(64));

        for d in [0i128, 1, 2, 4, -3] {
            assert_eq!(signed_magic(d, 64), None, "{d}");
        }
        assert_eq!(signed_magic(1i128 << 63, 64), None, "divisor exceeds the signed range");
        assert_eq!(signed_magic(3, 4).map(|(_, s)| s), Some(1));
    }
}
