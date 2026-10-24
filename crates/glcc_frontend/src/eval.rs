//! Constant evaluation for integer (and float) constant expressions.
//!
//! Literals lex raw (spelling + radix + suffix), so their first job here
//! is value resolution: [`int_constant`] maps a spelling to its C17
//! §6.4.4.1 type *and* value (decimal `3000000000` is `long`, hex
//! `0xFFFFFFFF` is `unsigned int` — the suffix alone does not decide),
//! [`char_constant`] processes escapes, and [`float_constant`] parses
//! decimal and hex floats directly in their suffix precision (`f` is
//! binary32, no suffix is binary64, `l` is the target's `long
//! double`).
//! Values live in `i128`/`FloatValue`: every in-range integer literal
//! fits (`unsigned long long` max sits below `i128::MAX`).
//!
//! [`eval_const`] folds checked expressions: literals, enumeration
//! constants, integer/float arithmetic under the usual arithmetic
//! conversions (integers wrapping to the common type, matching
//! two's-complement targets; floats rounding once in the common
//! precision, matching the target), comparisons, `!`/`~`/unary `+`/`-`,
//! ternaries (taken branch only), string lengths for `sizeof`, and
//! casts/conversions. Anything with an address, a call, a side effect,
//! or a need for layout (`sizeof` of incomplete types, members,
//! subscripts) is not a constant and yields `None`.
//!
//! Target facts (integer ranges, plain-`char` signedness, `long
//! double` precision) come from `Sema::target`, threaded through the
//! helpers as `Target`/`char_signed`/`LongDoubleKind` parameters.

use alloc::{boxed::Box, vec::Vec};

use rustc_apfloat::{
    Float, FloatConvert, Round,
    ieee::{Double, Quad, Single, X87DoubleExtended},
};

use crate::{
    check::{CheckedExpr, CheckedExprKind, corresponding_real, int_rank, int_type_size, is_complex, is_signed_int_in},
    lexer::{CharLiteral, CharPrefix, FloatLiteral, FloatSuffix, IntLiteral, IntSuffix, Interner, StringLiteral, StringPrefix},
    parse::{ArrayBound, BinaryOp, Initializer, Ty, UnaryOp},
    sema::{BindingKind, Sema},
    target::{LongDoubleKind, Target},
};

/// A floating-point constant value in a specific C precision: `float`
/// (binary32), `double` (binary64), `long double` (x86 80-bit
/// extended, or `double` on Windows), or `__float128` (IEEE quad).
///
/// Parsing and arithmetic round once to nearest-even in the value's
/// own precision, via `rustc_apfloat` (the library `rustc` itself uses
/// for constant folding): `1e4932L` stays finite (f80 range) while
/// `1e4932` overflows to infinity (f64 range), and `0.1L` keeps 64
/// mantissa bits instead of 53.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FloatValue {
    F32(Single),
    F64(Double),
    F80(X87DoubleExtended),
    F128(Quad),
}

/// The four C floating precisions [`FloatValue`] covers; private, since
/// callers work in terms of `Ty` (via [`float_kind`]) or whole values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FloatKind {
    F32,
    F64,
    F80,
    F128,
}

/// The [`FloatValue`] precision matching a C floating type (`None`
/// for anything else, including complex types, which never fold).
/// `long double` follows the target: x86 80-bit extended (`F80`),
/// IEEE quad (`F128`, AArch64/RISC-V where `long double` aliases
/// `__float128`), or plain `double` (`F64`, Windows).
fn float_kind(ty: &Ty, long_double: LongDoubleKind) -> Option<FloatKind> {
    match ty.unqualified() {
        Ty::Float => Some(FloatKind::F32),
        Ty::Double => Some(FloatKind::F64),
        Ty::LongDouble => Some(match long_double {
            LongDoubleKind::F80 => FloatKind::F80,
            LongDoubleKind::F128 => FloatKind::F128,
            LongDoubleKind::F64 => FloatKind::F64,
        }),
        Ty::Float128 => Some(FloatKind::F128),
        _ => None,
    }
}

impl FloatValue {
    fn kind(self) -> FloatKind {
        match self {
            FloatValue::F32(_) => FloatKind::F32,
            FloatValue::F64(_) => FloatKind::F64,
            FloatValue::F80(_) => FloatKind::F80,
            FloatValue::F128(_) => FloatKind::F128,
        }
    }

    /// This value widened to quad without loss: quad's 112-bit
    /// mantissa covers f80's 64 bits and the exponent ranges match, so
    /// every `f32`/`f64`/f80 value is exact in quad.
    fn to_quad(self) -> Quad {
        let mut loses = false;
        match self {
            FloatValue::F32(v) => v.convert_r(Round::NearestTiesToEven, &mut loses).value,
            FloatValue::F64(v) => v.convert_r(Round::NearestTiesToEven, &mut loses).value,
            FloatValue::F80(v) => v.convert_r(Round::NearestTiesToEven, &mut loses).value,
            FloatValue::F128(v) => v,
        }
    }

    /// This value rounded once to another precision (nearest-even).
    fn convert_to(self, kind: FloatKind) -> FloatValue {
        if self.kind() == kind {
            return self;
        }
        // Via quad, which holds this value exactly: a single rounding.
        let q = self.to_quad();
        let mut loses = false;
        match kind {
            FloatKind::F32 => FloatValue::F32(q.convert_r(Round::NearestTiesToEven, &mut loses).value),
            FloatKind::F64 => FloatValue::F64(q.convert_r(Round::NearestTiesToEven, &mut loses).value),
            FloatKind::F80 => FloatValue::F80(q.convert_r(Round::NearestTiesToEven, &mut loses).value),
            FloatKind::F128 => FloatValue::F128(q),
        }
    }

    /// Whether the value compares equal to zero (either sign; NaN is
    /// nonzero).
    pub fn is_zero(self) -> bool {
        match self {
            FloatValue::F32(v) => v.is_zero(),
            FloatValue::F64(v) => v.is_zero(),
            FloatValue::F80(v) => v.is_zero(),
            FloatValue::F128(v) => v.is_zero(),
        }
    }

    /// Whether the value is NaN, for `__builtin_isnan` folding.
    pub fn is_nan(self) -> bool {
        match self {
            FloatValue::F32(v) => v.is_nan(),
            FloatValue::F64(v) => v.is_nan(),
            FloatValue::F80(v) => v.is_nan(),
            FloatValue::F128(v) => v.is_nan(),
        }
    }

    /// Whether the value is infinite (either sign), for
    /// `__builtin_isinf` folding.
    pub fn is_infinite(self) -> bool {
        match self {
            FloatValue::F32(v) => v.is_infinite(),
            FloatValue::F64(v) => v.is_infinite(),
            FloatValue::F80(v) => v.is_infinite(),
            FloatValue::F128(v) => v.is_infinite(),
        }
    }

    /// Whether the value is finite (neither NaN nor infinite), for
    /// `__builtin_isfinite` folding.
    pub fn is_finite(self) -> bool {
        !self.is_nan() && !self.is_infinite()
    }

    /// Whether the value is normal (IEEE-754R: not zero, subnormal,
    /// infinite or NaN), for `__builtin_isnormal` folding.
    pub fn is_normal(self) -> bool {
        match self {
            FloatValue::F32(v) => v.is_normal(),
            FloatValue::F64(v) => v.is_normal(),
            FloatValue::F80(v) => v.is_normal(),
            FloatValue::F128(v) => v.is_normal(),
        }
    }

    /// Whether the value is negative (sign bit set, even for NaN).
    pub fn is_negative(self) -> bool {
        match self {
            FloatValue::F32(v) => v.is_negative(),
            FloatValue::F64(v) => v.is_negative(),
            FloatValue::F80(v) => v.is_negative(),
            FloatValue::F128(v) => v.is_negative(),
        }
    }

    /// Negation (a sign flip, even for NaN).
    fn neg(self) -> FloatValue {
        match self {
            FloatValue::F32(v) => FloatValue::F32(-v),
            FloatValue::F64(v) => FloatValue::F64(-v),
            FloatValue::F80(v) => FloatValue::F80(-v),
            FloatValue::F128(v) => FloatValue::F128(-v),
        }
    }

    /// `1.0L` as a folded value, for `long double` increment
    /// lowering (which adds one through the glccrt helper).
    pub fn long_double_one() -> FloatValue {
        Self::long_double_one_in(LongDoubleKind::F80)
    }

    /// Target-parameterized [`FloatValue::long_double_one`].
    pub fn long_double_one_in(kind: LongDoubleKind) -> FloatValue {
        match kind {
            LongDoubleKind::F80 => FloatValue::F80("1.0".parse().unwrap()),
            LongDoubleKind::F128 => FloatValue::F128("1.0".parse().unwrap()),
            LongDoubleKind::F64 => FloatValue::F64("1.0".parse().unwrap()),
        }
    }

    /// `0.0L` as a folded value, for zeroing the real part of an
    /// imaginary `long double` literal (`1.0iL`).
    pub fn long_double_zero() -> FloatValue {
        Self::long_double_zero_in(LongDoubleKind::F80)
    }

    /// Target-parameterized [`FloatValue::long_double_zero`].
    pub fn long_double_zero_in(kind: LongDoubleKind) -> FloatValue {
        match kind {
            LongDoubleKind::F80 => FloatValue::F80("0.0".parse().unwrap()),
            LongDoubleKind::F128 => FloatValue::F128("0.0".parse().unwrap()),
            LongDoubleKind::F64 => FloatValue::F64("0.0".parse().unwrap()),
        }
    }

    /// `+inf` / quiet NaN as `long double` folded values, for
    /// `__builtin_infl` / `__builtin_nanl` lowering.
    pub fn long_double_infinity() -> FloatValue {
        Self::long_double_infinity_in(LongDoubleKind::F80)
    }

    /// Target-parameterized [`FloatValue::long_double_infinity`].
    pub fn long_double_infinity_in(kind: LongDoubleKind) -> FloatValue {
        match kind {
            LongDoubleKind::F80 => FloatValue::F80(X87DoubleExtended::INFINITY),
            LongDoubleKind::F128 => FloatValue::F128(Quad::INFINITY),
            LongDoubleKind::F64 => FloatValue::F64(Double::INFINITY),
        }
    }

    /// Target-parameterized quiet-NaN `long double` (see above).
    pub fn long_double_nan_in(kind: LongDoubleKind) -> FloatValue {
        match kind {
            LongDoubleKind::F80 => FloatValue::F80(X87DoubleExtended::NAN),
            LongDoubleKind::F128 => FloatValue::F128(Quad::NAN),
            LongDoubleKind::F64 => FloatValue::F64(Double::NAN),
        }
    }
    pub fn long_double_nan() -> FloatValue {
        Self::long_double_nan_in(LongDoubleKind::F80)
    }

    /// An integer converted to the given precision (nearest-even).
    fn from_int(value: i128, kind: FloatKind) -> FloatValue {
        match kind {
            FloatKind::F32 => FloatValue::F32(Single::from_i128(value).value),
            FloatKind::F64 => FloatValue::F64(Double::from_i128(value).value),
            FloatKind::F80 => FloatValue::F80(X87DoubleExtended::from_i128(value).value),
            FloatKind::F128 => FloatValue::F128(Quad::from_i128(value).value),
        }
    }

    /// The value truncated toward zero as `i128`, or `None` for NaN,
    /// infinities, and out-of-range values (which have no defined
    /// conversion result).
    fn to_int_trunc(self) -> Option<i128> {
        let (status, value) = match self {
            FloatValue::F32(v) => {
                let r = v.to_i128(128);
                (r.status, r.value)
            }
            FloatValue::F64(v) => {
                let r = v.to_i128(128);
                (r.status, r.value)
            }
            FloatValue::F80(v) => {
                let r = v.to_i128(128);
                (r.status, r.value)
            }
            FloatValue::F128(v) => {
                let r = v.to_i128(128);
                (r.status, r.value)
            }
        };
        if status.contains(rustc_apfloat::Status::INVALID_OP) { None } else { Some(value) }
    }

    /// The value as `f32`, rounded once to nearest-even.
    pub fn to_f32(self) -> f32 {
        match self.convert_to(FloatKind::F32) {
            FloatValue::F32(v) => f32::from_bits(v.to_bits() as u32),
            _ => unreachable!("converted to binary32"),
        }
    }

    /// The value as `f64`, rounded once to nearest-even.
    pub fn to_f64(self) -> f64 {
        match self.convert_to(FloatKind::F64) {
            FloatValue::F64(v) => f64::from_bits(v.to_bits() as u64),
            _ => unreachable!("converted to binary64"),
        }
    }

    /// The value as IEEE-quad bits, rounded once. Used for
    /// `__float128` codegen, which is genuinely quad.
    pub fn to_quad_bits(self) -> u128 {
        self.to_quad().to_bits()
    }

    /// The value as x86 80-bit-extended bytes (the C ABI form of `long
    /// double` on this target): the 10-byte little-endian extended
    /// value (64-bit mantissa with explicit integer bit, then
    /// sign+exponent) followed by 6 zero pad bytes, for 16 bytes
    /// total matching `long_double_size`. `rustc_apfloat`'s X87
    /// `to_bits` already packs exactly this layout (sign at bit 79,
    /// biased exponent at bits 64..79, mantissa below), so this is a
    /// byte-order copy, not a conversion.
    pub fn to_f80_bytes(self) -> [u8; 16] {
        match self.convert_to(FloatKind::F80) {
            FloatValue::F80(v) => v.to_bits().to_le_bytes(),
            _ => unreachable!("converted to 80-bit extended"),
        }
    }
}

/// A folded constant: an integer (always in `i128`, two's-complement
/// for signed values), a floating point value in its own precision, or
/// a complex number as its real and imaginary parts (each an integer
/// or float, never nested complex).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i128),
    Float(FloatValue),
    Complex(Box<Value>, Box<Value>),
}

fn suffix_len(suffix: IntSuffix) -> usize {
    match suffix {
        IntSuffix::None => 0,
        IntSuffix::U | IntSuffix::L => 1,
        IntSuffix::UL | IntSuffix::LL | IntSuffix::I8 | IntSuffix::Wb => 2,
        IntSuffix::ULL | IntSuffix::I16 | IntSuffix::I32 | IntSuffix::I64 | IntSuffix::UI8 | IntSuffix::UWb => 3,
        IntSuffix::UI16 | IntSuffix::UI32 | IntSuffix::UI64 => 4,
    }
}

/// The value and C17 §6.4.4.1 type of an integer literal spelling, or
/// `None` when no `unsigned long long` fits (a constraint violation the
/// caller reports). `long_size` is the target's `sizeof(long)` (4 when
/// LLP64, 8 when LP64).
pub fn int_constant(lit: &IntLiteral, interner: &Interner, long_size: u64) -> Option<(Ty, i128)> {
    let spelling = interner.resolve(lit.spelling)?;
    let digits = spelling.get(..spelling.len().checked_sub(suffix_len(lit.suffix))?)?;
    let (text, radix) = match lit.radix {
        crate::lexer::Radix::Dec => (digits, 10),
        crate::lexer::Radix::Hex => (digits.strip_prefix("0x").or_else(|| digits.strip_prefix("0X"))?, 16),
        crate::lexer::Radix::Oct => (
            digits
                .strip_prefix("0o")
                .or_else(|| digits.strip_prefix("0O"))
                .or_else(|| digits.strip_prefix('0'))
                .unwrap_or(digits),
            8,
        ),
        crate::lexer::Radix::Bin => (digits.strip_prefix("0b").or_else(|| digits.strip_prefix("0B"))?, 2),
    };
    let text = if text.is_empty() { "0" } else { text };
    let magnitude = u128::from_str_radix(text, radix).ok()?;
    // First type in each list whose range holds the magnitude.
    const I8: u128 = 1 << 7;
    const U8: u128 = 1 << 8;
    const I16: u128 = 1 << 15;
    const U16: u128 = 1 << 16;
    const I31: u128 = 1 << 31;
    const U32: u128 = 1 << 32;
    const I63: u128 = 1 << 63;
    const U64: u128 = 1 << 64;
    let (long_bound, ulong_bound) = if long_size == 8 { (I63, U64) } else { (I31, U32) };
    let decimal = matches!(lit.radix, crate::lexer::Radix::Dec);
    if matches!(lit.suffix, IntSuffix::Wb | IntSuffix::UWb) {
        let magnitude_bits = u128::BITS - magnitude.leading_zeros();
        let signed = lit.suffix == IntSuffix::Wb;
        let width = if signed { magnitude_bits + 1 } else { magnitude_bits };
        let width = width.max(if signed { 2 } else { 1 });
        if width > 64 {
            return None;
        }
        return Some((Ty::BitInt { width, signed }, magnitude as i128));
    }
    let table: &[(Ty, u128)] = match (decimal, lit.suffix) {
        (true, IntSuffix::None) => &[(Ty::Int, I31), (Ty::Long, long_bound), (Ty::LongLong, I63)],
        (true, IntSuffix::U) => &[(Ty::UnsignedInt, U32), (Ty::UnsignedLong, ulong_bound), (Ty::UnsignedLongLong, U64)],
        (true, IntSuffix::L) => &[(Ty::Long, long_bound), (Ty::LongLong, I63)],
        (true, IntSuffix::UL) => &[(Ty::UnsignedLong, ulong_bound), (Ty::UnsignedLongLong, U64)],
        (true, IntSuffix::LL) => &[(Ty::LongLong, I63)],
        (true, IntSuffix::ULL) => &[(Ty::UnsignedLongLong, U64)],
        (false, IntSuffix::None) => &[
            (Ty::Int, I31),
            (Ty::UnsignedInt, U32),
            (Ty::Long, long_bound),
            (Ty::UnsignedLong, ulong_bound),
            (Ty::LongLong, I63),
            (Ty::UnsignedLongLong, U64),
        ],
        (false, IntSuffix::U) => &[(Ty::UnsignedInt, U32), (Ty::UnsignedLong, ulong_bound), (Ty::UnsignedLongLong, U64)],
        (false, IntSuffix::L) => &[(Ty::Long, long_bound), (Ty::UnsignedLong, ulong_bound), (Ty::LongLong, I63), (Ty::UnsignedLongLong, U64)],
        (false, IntSuffix::UL) => &[(Ty::UnsignedLong, ulong_bound), (Ty::UnsignedLongLong, U64)],
        (false, IntSuffix::LL) => &[(Ty::LongLong, I63), (Ty::UnsignedLongLong, U64)],
        (false, IntSuffix::ULL) => &[(Ty::UnsignedLongLong, U64)],
        // MSVC's `__intN` suffixes name the type outright.
        (_, IntSuffix::I8) => &[(Ty::SignedChar, I8)],
        (_, IntSuffix::I16) => &[(Ty::Short, I16)],
        (_, IntSuffix::I32) => &[(Ty::Int, I31)],
        // MSVC allows an out-of-range hex `i64` to fall back to unsigned,
        // matching its `LL` treatment (`0xffffffffffffffffi64`).
        (true, IntSuffix::I64) => &[(Ty::LongLong, I63)],
        (false, IntSuffix::I64) => &[(Ty::LongLong, I63), (Ty::UnsignedLongLong, U64)],
        (_, IntSuffix::UI8) => &[(Ty::UnsignedChar, U8)],
        (_, IntSuffix::UI16) => &[(Ty::UnsignedShort, U16)],
        (_, IntSuffix::UI32) => &[(Ty::UnsignedInt, U32)],
        (_, IntSuffix::UI64) => &[(Ty::UnsignedLongLong, U64)],
        (_, IntSuffix::Wb | IntSuffix::UWb) => unreachable!("handled above"),
    };
    let (ty, _) = table.iter().find(|(_, bound)| magnitude < *bound)?;
    // Every surviving magnitude fits `i128` (`u64::MAX < i128::MAX`).
    Some((ty.clone(), magnitude as i128))
}

/// Decode one escape (the character after the backslash) plus any
/// following digits, returning the value and the characters consumed.
/// Unknown escapes keep the character itself (gcc warns and agrees).
pub(crate) fn escape_value(chars: &[char]) -> (u32, usize) {
    // Shared with `crate::sema::string_contents`, which decodes string
    // escapes the same way.
    let (first, rest) = (chars[0], &chars[1..]);
    match first {
        'n' => (b'\n' as u32, 1),
        't' => (b'\t' as u32, 1),
        'r' => (b'\r' as u32, 1),
        'v' => (0x0B, 1),
        'b' => (0x08, 1),
        'f' => (0x0C, 1),
        'a' => (0x07, 1),
        '\\' | '\'' | '"' | '?' => (first as u32, 1),
        '0'..='7' => {
            let mut value = first.to_digit(8).unwrap();
            let mut used = 1;
            for c in rest.iter().take(2) {
                if let Some(d) = c.to_digit(8) {
                    value = value * 8 + d;
                    used += 1;
                } else {
                    break;
                }
            }
            (value, used)
        }
        'x' => {
            if let Some((value, used)) = delimited_escape(rest, 16) {
                return (value, used + 1);
            }
            let mut value = 0u32;
            let mut used = 1;
            for c in rest.iter() {
                if let Some(d) = c.to_digit(16) {
                    value = value.wrapping_mul(16).wrapping_add(d);
                    used += 1;
                } else {
                    break;
                }
            }
            (value, used)
        }
        'o' => {
            if let Some((value, used)) = delimited_escape(rest, 8) {
                return (value, used + 1);
            }
            (first as u32, 1)
        }
        // `\uXXXX` / `\UXXXXXXXX` name a code point (C17 §6.4.3); the
        // lexer has already validated the spelling and range.
        'u' | 'U' => {
            if let Some((value, used)) = delimited_escape(rest, 16) {
                return (value, used + 1);
            }
            let digits = if first == 'u' { 4 } else { 8 };
            let mut value = 0u32;
            let mut used = 1;
            for c in rest.iter().take(digits) {
                if let Some(d) = c.to_digit(16) {
                    value = value.wrapping_mul(16).wrapping_add(d);
                    used += 1;
                } else {
                    break;
                }
            }
            (value, used)
        }
        _ => (first as u32, 1),
    }
}

/// Decode a delimited escape body after its `u`, `U`, `x`, or `o`
/// prefix: `{digits}` in `radix`, returning the value and the number of
/// characters consumed after the prefix. Empty bodies and malformed
/// digits yield `None` (the lexer diagnoses them).
fn delimited_escape(rest: &[char], radix: u32) -> Option<(u32, usize)> {
    if rest.first() != Some(&'{') {
        return None;
    }
    let mut value = 0u32;
    for (i, &c) in rest[1..].iter().enumerate() {
        if c == '}' {
            return (i > 0).then_some((value, i + 2));
        }
        let d = c.to_digit(radix)?;
        value = value.wrapping_mul(radix).wrapping_add(d);
    }
    None
}

/// Whether the characters after a backslash begin a universal character
/// name (`\u` / `\U`).
pub(crate) fn is_ucn(tail: &[char]) -> bool {
    matches!(tail.first(), Some('u' | 'U'))
}

/// Narrow (byte) elements for a UCN's code point: values at or below
/// U+00FF stay one byte, matching a raw source character of the same
/// value (see `decode_source`); wider ones become their UTF-8 bytes,
/// exactly as the raw-character path below narrows.
fn push_narrow_codepoint(out: &mut Vec<u32>, value: u32) {
    match char::from_u32(value) {
        Some(c) if (c as u32) <= 0xFF => out.push(c as u32),
        Some(c) => {
            let mut buf = [0u8; 4];
            out.extend(c.encode_utf8(&mut buf).bytes().map(u32::from));
        }
        None => out.push(value),
    }
}

/// The value of a character constant: escapes processed, multi-character
/// constants folded shift-or like GCC. A single narrow character takes
/// the target's plain-`char` value: sign-extended when `char_signed`
/// (x86-64 and Windows), zero-extended otherwise (AArch64/RISC-V).
/// Anything wider wraps to `int`.
pub fn char_constant(lit: &CharLiteral, interner: &Interner) -> Option<i128> {
    char_constant_in(lit, interner, Target::LP64.char_signed)
}

/// Target-parameterized [`char_constant`].
pub fn char_constant_in(lit: &CharLiteral, interner: &Interner, char_signed: bool) -> Option<i128> {
    let spelling = interner.resolve(lit.spelling)?;
    let prefix_len = match lit.prefix {
        CharPrefix::None => 0,
        CharPrefix::L | CharPrefix::SmallU | CharPrefix::BigU => 1,
        CharPrefix::U8 => 2,
    };
    let body = spelling.get(prefix_len + 1..spelling.len().checked_sub(1)?)?;
    let mut values: Vec<u32> = Vec::new();
    if matches!(lit.prefix, CharPrefix::None | CharPrefix::U8) {
        // Narrow: GCC combines raw bytes, so each source character is
        // its Latin-1 byte (or UTF-8 bytes for a wider `char`).
        let chars: Vec<char> = body.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '\\' {
                let tail = &chars[i + 1..];
                if tail.is_empty() {
                    return None;
                }
                let (v, used_chars) = escape_value(tail);
                if is_ucn(tail) {
                    push_narrow_codepoint(&mut values, v);
                } else {
                    values.push(v);
                }
                i += 1 + used_chars;
            } else {
                if (chars[i] as u32) <= 0xFF {
                    values.push(chars[i] as u32);
                } else {
                    let mut buf = [0u8; 4];
                    values.extend(chars[i].encode_utf8(&mut buf).as_bytes().iter().map(|&b| b as u32));
                }
                i += 1;
            }
        }
    } else {
        // Wide: code points, not bytes.
        let chars: Vec<char> = body.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '\\' {
                if i + 1 >= chars.len() {
                    return None;
                }
                let (v, used) = escape_value(&chars[i + 1..]);
                values.push(v);
                i += 1 + used;
            } else {
                values.push(chars[i] as u32);
                i += 1;
            }
        }
    }
    if values.len() == 1 && lit.prefix == CharPrefix::U8 {
        // A UTF-8 code unit is unsigned.
        return Some(values[0] as u8 as i128);
    }
    if values.len() == 1 && lit.prefix == CharPrefix::None {
        // A single narrow character has `char` range: sign- or
        // zero-extend by the target's plain-`char` signedness.
        if char_signed {
            return Some(values[0] as u8 as i8 as i128);
        }
        return Some(values[0] as u8 as i128);
    }
    if values.len() == 1 {
        // A single wide character is its full code point (`wchar_t`):
        // escapes are not narrowed to a byte (`L'\400'` is 256).
        return Some(values[0] as i128);
    }
    if values.is_empty() {
        return None;
    }
    let folded = values.iter().fold(0u32, |acc, &b| acc.wrapping_shl(8) | (b & 0xFF));
    Some(folded as i32 as i128)
}

/// The value of a float constant, parsed directly in its suffix
/// precision (`f` is binary32, no suffix is binary64, `l` is the
/// target's `long double`: x86 80-bit extended, IEEE quad on
/// AArch64/RISC-V, plain `double` on Windows), or `None` on a
/// malformed spelling. Decimal and hex spellings both round once to
/// nearest-even; overflow yields infinity, matching the target's
/// arithmetic.
pub fn float_constant(lit: &FloatLiteral, interner: &Interner) -> Option<FloatValue> {
    float_constant_in(lit, interner, LongDoubleKind::F80)
}

/// Target-parameterized [`float_constant`].
pub fn float_constant_in(lit: &FloatLiteral, interner: &Interner, long_double: LongDoubleKind) -> Option<FloatValue> {
    let spelling = interner.resolve(lit.spelling)?;
    let text = match lit.suffix {
        FloatSuffix::None => spelling,
        FloatSuffix::F | FloatSuffix::L => spelling.get(..spelling.len().checked_sub(1)?)?,
    };
    // `radix` only records how the lexer classified the spelling;
    // parsing sniffs the `0x` prefix itself, so both spellings share one
    // path (octal never reaches here: the lexer only emits floats for
    // decimal and hex shapes).
    match lit.radix {
        crate::lexer::Radix::Dec | crate::lexer::Radix::Hex => {}
        crate::lexer::Radix::Oct | crate::lexer::Radix::Bin => return None,
    }
    match lit.suffix {
        FloatSuffix::F => text.parse::<Single>().ok().map(FloatValue::F32),
        FloatSuffix::None => text.parse::<Double>().ok().map(FloatValue::F64),
        FloatSuffix::L => match long_double {
            LongDoubleKind::F80 => text.parse::<X87DoubleExtended>().ok().map(FloatValue::F80),
            LongDoubleKind::F128 => text.parse::<Quad>().ok().map(FloatValue::F128),
            LongDoubleKind::F64 => text.parse::<Double>().ok().map(FloatValue::F64),
        },
    }
}

/// Element type of a concatenated string literal's character array:
/// `char` for narrow and `u8` literals, `wchar_t` for `L` (`int` on
/// x86-64/RISC-V, `unsigned int` on AArch64, `unsigned short` on
/// Windows — see [`string_elem_ty_in`]), `unsigned short` for `u`
/// (char16_t), `unsigned int` for `U` (char32_t). Adjacent literals of
/// different prefixes concatenate into the widest kind (`L"a" "b"` is
/// wide, C11 §6.4.5p5); incompatible kinds (`L` with `u`/`U`, or `u`
/// with `U`) yield `None`.
pub fn string_elem_ty(parts: &[StringLiteral]) -> Option<Ty> {
    string_elem_ty_in(parts, crate::target::WChar { size: 4, signed: true })
}

/// [`string_elem_ty`] with the target's `wchar_t` shape: `L` literals
/// take the `Ty` matching `__WCHAR_TYPE__`.
pub fn string_elem_ty_in(parts: &[StringLiteral], wchar: crate::target::WChar) -> Option<Ty> {
    string_elem_ty_std(parts, wchar, false)
}

/// [`string_elem_ty_in`] with the C23 `char8_t` rule: a UTF-8 string
/// literal is an array of `unsigned char` instead of plain `char`.
pub fn string_elem_ty_std(parts: &[StringLiteral], wchar: crate::target::WChar, c23: bool) -> Option<Ty> {
    let mut wide = false;
    let mut char16 = false;
    let mut char32 = false;
    let mut utf8 = false;
    for part in parts {
        match part.prefix {
            StringPrefix::None => {}
            StringPrefix::U8 => utf8 = true,
            StringPrefix::L => wide = true,
            StringPrefix::SmallU => char16 = true,
            StringPrefix::BigU => char32 = true,
        }
    }
    match (wide, char16, char32) {
        (true, false, false) => Some(wchar.ty()),
        (true, _, _) | (false, true, true) => None,
        (false, true, false) => Some(Ty::UnsignedShort),
        (false, false, true) => Some(Ty::UnsignedInt),
        (false, false, false) if c23 && utf8 => Some(Ty::UnsignedChar),
        (false, false, false) => Some(Ty::Char),
    }
}

/// Width of one string-literal element in bytes, matching
/// [`string_elem_ty`].
fn string_elem_width(elem: &Ty) -> u64 {
    match elem {
        Ty::Char | Ty::SignedChar | Ty::UnsignedChar => 1,
        Ty::Short | Ty::UnsignedShort => 2,
        _ => 4,
    }
}

/// Number of elements in a concatenated string literal's contents,
/// without the terminating NUL: the bound `char s[] = "ab"` completes
/// to this plus one, and `sizeof` multiplies by the element width.
pub fn string_length(parts: &[StringLiteral], interner: &Interner) -> Option<u64> {
    string_elems(parts, interner).map(|values| values.len() as u64)
}

/// Element values of a concatenated string literal's contents, without
/// the terminating NUL: narrow literals contribute one byte per source
/// character or escape (hex escapes truncate mod 256, like GCC), wide
/// ones one code point each, and adjacent literals concatenate. A
/// narrow part in a wide concatenation widens per source character
/// (`L"a" "b"` is `L"ab"`), one code point each.
fn string_elems(parts: &[StringLiteral], interner: &Interner) -> Option<Vec<u32>> {
    let elem = string_elem_ty(parts)?;
    let wide = !matches!(elem, Ty::Char);
    let mut values = Vec::new();
    for part in parts {
        let spelling = interner.resolve(part.spelling)?;
        let prefix_len = part.prefix.as_str().len();
        let body = spelling.get(prefix_len + 1..spelling.len().checked_sub(1)?)?;
        let chars: Vec<char> = body.chars().collect();
        let part_wide = !matches!(part.prefix, StringPrefix::None | StringPrefix::U8);
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '\\' {
                let tail = &chars[i + 1..];
                if tail.is_empty() {
                    return None;
                }
                let (v, used_chars) = escape_value(tail);
                if is_ucn(tail) {
                    // A UCN names a code point: wide elements keep it
                    // whole, narrow ones narrow it like the raw character.
                    if part_wide || wide {
                        values.push(v);
                    } else {
                        push_narrow_codepoint(&mut values, v);
                    }
                } else {
                    values.push(if part_wide { v } else { v & 0xFF });
                }
                i += 1 + used_chars;
            } else if part_wide || wide {
                // Code points, not bytes.
                values.push(chars[i] as u32);
                i += 1;
            } else {
                // Source bytes arrive as `char`s: values up to U+00FF
                // are Latin-1 bytes (see [`crate::decode_source`]); a
                // wider `char` is a real multibyte source character
                // and contributes its UTF-8 bytes, like passing the
                // source bytes through would.
                let c = chars[i];
                let mut buf = [0u8; 4];
                if (c as u32) <= 0xFF {
                    values.push(c as u32);
                } else {
                    values.extend(c.encode_utf8(&mut buf).as_bytes().iter().map(|&b| b as u32));
                }
                i += 1;
            }
        }
    }
    Some(values)
}

/// The byte image of a concatenated string literal's contents, without
/// the terminating NUL, for a default LP64 `wchar_t` (see
/// [`string_bytes_in`]).
pub fn string_bytes(parts: &[StringLiteral], interner: &Interner) -> Option<Vec<u8>> {
    string_bytes_in(parts, interner, crate::target::WChar { size: 4, signed: true })
}

/// [`string_bytes`] with the target's `wchar_t` shape: narrow literals
/// are one byte per element, wide literals encode each element
/// little-endian in its width (two bytes per `L` element on Windows,
/// where `wchar_t` is a 16-bit `unsigned short`). The backend consumes
/// this for data emission. Malformed spellings and mixed prefixes
/// yield `None`.
pub fn string_bytes_in(parts: &[StringLiteral], interner: &Interner, wchar: crate::target::WChar) -> Option<Vec<u8>> {
    let elem = string_elem_ty_in(parts, wchar)?;
    let width = string_elem_width(&elem) as usize;
    let values = string_elems(parts, interner)?;
    let mut bytes = Vec::with_capacity(values.len() * width);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes()[..width]);
    }
    Some(bytes)
}

/// Alignment of a `__alignof__` operand: the type's natural alignment,
/// with `void` aligning to 1 like GNU (`__alignof__(void)`).
pub fn align_of(sema: &Sema, ty: &Ty, interner: &Interner) -> Option<u64> {
    // `void` and function types are incomplete but align to 1 (GNU); a
    // function's attribute alignment comes from its declaration (see
    // `align_of_expr`).
    if matches!(ty.unqualified(), Ty::Void | Ty::Func { .. }) {
        return Some(1);
    }
    sema.type_align(ty, interner)
}

/// Size of a `sizeof` operand: the type's size, with `void` and
/// functions sized 1 like GNU (`sizeof(void)` and `sizeof f` for a
/// function `f` are 1 in GCC and clang, with a warning).
pub fn sizeof_of(sema: &Sema, ty: &Ty, interner: &Interner) -> Option<u64> {
    if matches!(ty.unqualified(), Ty::Void | Ty::Func { .. }) {
        return Some(1);
    }
    sema.type_size(ty, interner)
}

/// The constant element count of an array type, if its top-level
/// extent is fixed. Variable extents need runtime values.
fn count_of(ty: &Ty) -> Option<u64> {
    match ty.unqualified() {
        Ty::Array {
            bound: Some(ArrayBound::Fixed(n)),
            ..
        } => Some(*n),
        _ => None,
    }
}

/// The alignment an `__alignof__` *expression* reports. An `aligned`
/// attribute on the declaration sets it: for a function it is 1
/// without one (like GCC), for an object the type's alignment applies
/// when there is no attribute.
pub fn align_of_expr(sema: &Sema, operand: &CheckedExpr, interner: &Interner) -> Option<u64> {
    let declared = match operand.kind {
        CheckedExprKind::Ident(id) => sema.decls[id.0 as usize].align,
        _ => None,
    };
    if matches!(operand.ty.unqualified(), Ty::Func { .. }) {
        return Some(declared.unwrap_or(1));
    }
    declared.or_else(|| align_of(sema, &operand.ty, interner))
}

/// Fold a checked expression to a constant, or `None` when it is not a
/// constant expression.
pub fn eval_const(expr: &CheckedExpr, sema: &Sema, interner: &Interner) -> Option<Value> {
    eval_const_in(expr, sema, interner, false)
}

/// Fold a checked expression as a static initializer: like
/// [`eval_const`], but conditions that are non-null address constants
/// also fold (`&obj ? a : b`, `"lit" ? a : b`), including under `!`,
/// `&&`, and `||`. Address constants are not integer constant
/// expressions, so other constant contexts stay strict (C17 §6.6p7
/// admits them in initializers for static storage; GCC folds there and
/// tcctest.c relies on it). A weak symbol may resolve to zero, so its
/// address never folds.
pub fn eval_static_init(expr: &CheckedExpr, sema: &Sema, interner: &Interner) -> Option<Value> {
    eval_const_in(expr, sema, interner, true)
}

/// Whether `expr` is a constant expression suitable for a C23
/// `constexpr` initializer: foldable arithmetic or a non-null address
/// constant. A null pointer constant folds through the cast path.
pub fn is_constant_initializer(expr: &CheckedExpr, sema: &Sema, interner: &Interner) -> bool {
    if eval_static_init(expr, sema, interner).is_some() || address_nonzero(expr, sema) {
        return true;
    }
    // A cast null pointer constant does not fold to a `Value`, but is
    // still a constant initializer.
    match &expr.kind {
        CheckedExprKind::Cast { ty, expr } | CheckedExprKind::Convert { expr, ty } if matches!(ty.unqualified(), Ty::Pointer { .. }) => {
            matches!(eval_const(expr, sema, interner), Some(Value::Int(0)))
        }
        CheckedExprKind::Ident(id) => matches!(sema.decls.get(id.0 as usize).map(|binding| &binding.kind), Some(BindingKind::Func)),
        _ => false,
    }
}

/// Whether every leaf of an initializer is a constant expression.
/// Static-storage compound literals (C23 `(static T){...}`) require
/// this just like file-scope objects do.
pub fn initializer_is_constant(init: &Initializer, sema: &Sema, interner: &Interner) -> bool {
    match init {
        Initializer::Expr(expr) => is_constant_initializer(expr, sema, interner),
        Initializer::List { items, .. } => items.iter().all(|item| initializer_is_constant(&item.init, sema, interner)),
    }
}

fn eval_const_in(expr: &CheckedExpr, sema: &Sema, interner: &Interner, addr_conds: bool) -> Option<Value> {
    match &expr.kind {
        CheckedExprKind::Int(lit) => {
            let (_, v) = int_constant(lit, interner, sema.target.long_size)?;
            if is_complex(&expr.ty) {
                // GNU imaginary literal (`1i`): the value is the
                // imaginary part, over a zero real part.
                Some(Value::Complex(Box::new(Value::Int(0)), Box::new(Value::Int(v))))
            } else {
                Some(Value::Int(v))
            }
        }
        CheckedExprKind::Float(lit) => {
            let v = float_constant_in(lit, interner, sema.target.long_double)?;
            if is_complex(&expr.ty) {
                // GNU imaginary literal (`1.0i`): zero real part in
                // the same precision as the value.
                let zero = FloatValue::from_int(0, float_kind(&corresponding_real(&expr.ty), sema.target.long_double)?);
                Some(Value::Complex(Box::new(Value::Float(zero)), Box::new(Value::Float(v))))
            } else {
                Some(Value::Float(v))
            }
        }
        CheckedExprKind::Char(lit) => char_constant_in(lit, interner, sema.target.char_signed).map(Value::Int),
        CheckedExprKind::Str(_) => None,
        // Suspending is a runtime event, never a constant.
        CheckedExprKind::Await(_) => None,
        CheckedExprKind::Ident(id) => match &sema.decls.get(id.0 as usize)?.kind {
            BindingKind::EnumConst { value } => Some(Value::Int(*value)),
            _ => sema.constexprs.get(id).cloned(),
        },
        CheckedExprKind::Unary(op, operand) => eval_unary(*op, operand, sema, interner, addr_conds),
        CheckedExprKind::Binary(op, lhs, rhs) => eval_binary(*op, lhs, rhs, sema, interner, addr_conds),
        CheckedExprKind::Ternary { cond, then, els } => {
            let nonzero = match eval_const_in(cond, sema, interner, addr_conds) {
                Some(v) => truthy(&v),
                // Where initializers allow address constants, an
                // address condition is known non-null. Anything else
                // stays unknown — a weak symbol's address may be zero,
                // so it never decides the arm.
                None if addr_conds && address_nonzero(cond, sema) => true,
                None => return None,
            };
            if nonzero {
                eval_const_in(then, sema, interner, addr_conds)
            } else {
                eval_const_in(els, sema, interner, addr_conds)
            }
        }
        CheckedExprKind::Elvis { cond, els } => {
            // The condition's own value (converted to the result type,
            // like the lowering does) when it folds nonzero.
            let value = eval_const_in(cond, sema, interner, addr_conds)?;
            if truthy(&value) {
                eval_cast(&expr.ty, cond, sema, interner, addr_conds)
            } else {
                eval_const_in(els, sema, interner, addr_conds)
            }
        }
        CheckedExprKind::Cast { ty, expr } | CheckedExprKind::Convert { expr, ty } => {
            eval_cast(ty, expr, sema, interner, addr_conds)
        }
        CheckedExprKind::Generic { index, assocs, .. } => {
            assocs
                .get(*index)
                .and_then(|(_, expr)| eval_const(expr, sema, interner))
        }
        CheckedExprKind::GenericTy { index, assocs, .. } => {
            assocs
                .get(*index)
                .and_then(|(_, expr)| eval_const(expr, sema, interner))
        }
        // Folded at check time from the member layout.
        CheckedExprKind::Offsetof { offset, .. } => Some(Value::Int(*offset as i128)),
        // Folded at check time from the operand's constness.
        CheckedExprKind::ConstantP(b) => Some(Value::Int(*b as i128)),
        // Folded at check time from the operand's type.
        CheckedExprKind::ClassifyType(code) => Some(Value::Int(*code as i128)),
        // Classifications fold over constant operands (integers never
        // reach here; they fold at check time).
        CheckedExprKind::FpClass { op, arg } => match eval_const(arg, sema, interner)? {
            Value::Float(f) => Some(Value::Int(match op {
                crate::check::FpPredicate::IsNan => i128::from(f.is_nan()),
                crate::check::FpPredicate::IsInf => i128::from(f.is_infinite()),
                // Sign-distinguishing like GCC (`1` / `-1` / `0`).
                crate::check::FpPredicate::IsInfSign if !f.is_infinite() => 0,
                crate::check::FpPredicate::IsInfSign => i128::from(if f.is_negative() { -1 } else { 1 }),
                crate::check::FpPredicate::IsFinite => i128::from(f.is_finite()),
                crate::check::FpPredicate::IsNormal => i128::from(f.is_normal()),
                // `-0.0` counts, so the sign bit decides.
                crate::check::FpPredicate::SignBit => i128::from(f.is_negative()),
            })),
            Value::Int(_) => Some(Value::Int(0)),
            // Classifications never see complex operands (rejected at
            // check time); a complex constant is not a classification.
            Value::Complex(..) => None,
        },
        // The five codes select on the operand's category, so a
        // constant operand folds to the constant code.
        CheckedExprKind::FpClassify { codes, arg } => {
            let Value::Float(f) = eval_const(arg, sema, interner)? else {
                return None;
            };
            let index = if f.is_nan() {
                0
            } else if f.is_infinite() {
                1
            } else if f.is_normal() {
                2
            } else if f.is_zero() {
                4
            } else {
                3
            };
            eval_const(&codes[index], sema, interner)
        }
        CheckedExprKind::Call { .. }
        | CheckedExprKind::VaArg { .. }
        | CheckedExprKind::Atomic { .. }
        | CheckedExprKind::Overflow { .. }
        | CheckedExprKind::Alloca { .. }
        | CheckedExprKind::FrameAddress { .. }
        | CheckedExprKind::Clz { .. }
        // Statement expressions are never constant expressions.
        | CheckedExprKind::StmtExpr { .. }
        | CheckedExprKind::Subscript { .. }
        | CheckedExprKind::Member { .. }
        | CheckedExprKind::CompoundLiteral { .. }
        | CheckedExprKind::Assign(..)
        | CheckedExprKind::CompoundAssign { .. }
        // A static assertion has type void and no value.
        | CheckedExprKind::StaticAssert => None,
        // `sizeof` folds when layout knows the size (complete types,
        // string literals included); variable-length arrays and opaque
        // builtins never fold.
        CheckedExprKind::SizeofTy(ty) => sizeof_of(sema, ty, interner)
            .map(|size| Value::Int(size as i128)),
        CheckedExprKind::AlignofTy(ty) => align_of(sema, ty, interner)
            .map(|align| Value::Int(align as i128)),
        CheckedExprKind::AlignofExpr(operand) => align_of_expr(sema, operand, interner)
            .map(|align| Value::Int(align as i128)),
        CheckedExprKind::SizeofExpr(operand) => sizeof_of(sema, &operand.ty, interner)
            .map(|size| Value::Int(size as i128)),
        CheckedExprKind::CountofTy(ty) => count_of(ty).map(|n| Value::Int(n as i128)),
        CheckedExprKind::CountofExpr(operand) => count_of(&operand.ty).map(|n| Value::Int(n as i128)),
        // A label address is only meaningful to `goto *`, and is
        // resolved at lowering, never folded.
        CheckedExprKind::LabelAddr(_) => None,
    }
}

/// Integer promotions for evaluation: sub-`int` ranks become `int`
/// (LP64 `int` represents every such value, mirroring
/// `Sema::integer_promote` without building nodes). Complex over a
/// sub-`int` component promotes the component, likewise. Also used to
/// find the type duplicate `case` labels compare in.
///
/// Public because MIR lowering needs the same promoted type (and the
/// same `const_as` values) the checker used, so dispatch agrees with
/// duplicate detection exactly.
pub fn promoted(ty: &Ty) -> Ty {
    match ty.unqualified() {
        Ty::BitInt { .. } => ty.unqualified().clone(),
        Ty::Complex(inner) => match int_rank(inner) {
            Some(rank) if rank < int_rank(&Ty::Int).expect("Int has a rank") => Ty::Complex(Box::new(Ty::Int)),
            _ => ty.clone(),
        },
        _ => match int_rank(ty) {
            Some(rank) if rank < int_rank(&Ty::Int).expect("Int has a rank") => Ty::Int,
            _ => ty.clone(),
        },
    }
}

/// Width in bits of an integer type (`None` for non-integers).
fn width(ty: &Ty, target: Target) -> Option<u32> {
    match ty.unqualified() {
        Ty::BitInt { width, .. } => Some(*width),
        _ => int_type_size(ty, &target).map(|s| s * 8),
    }
}

/// The low `width(ty)` bits of `v` as an unsigned pattern.
fn to_bits(v: i128, ty: &Ty, target: Target) -> Option<u128> {
    let bits = width(ty, target)?;
    let mask = if bits == 128 { u128::MAX } else { (1u128 << bits) - 1 };
    Some((v as u128) & mask)
}

/// A bit pattern as the mathematical value of an integer type:
/// sign-extended when signed, `_Bool` collapsing to 0/1. Plain `char`
/// follows `char_signed` (see [`char_constant_in`]).
fn from_bits(bits: u128, ty: &Ty, target: Target) -> Option<i128> {
    let bitlen = width(ty, target)?;
    let mask = if bitlen == 128 { u128::MAX } else { (1u128 << bitlen) - 1 };
    let bits = bits & mask;
    if *ty == Ty::Bool {
        return Some((bits != 0) as i128);
    }
    if is_signed_int_in(ty, target.char_signed)? && bitlen > 0 && (bits >> (bitlen - 1)) & 1 == 1 {
        Some((bits | !mask) as i128)
    } else {
        Some(bits as i128)
    }
}

/// Wrap a value to an integer type's range (modular arithmetic for
/// unsigned, two's-complement wrap for signed, 0/1 for `_Bool`).
fn wrap(v: i128, ty: &Ty, target: Target) -> Option<i128> {
    from_bits(to_bits(v, ty, target)?, ty, target)
}

/// Whether a folded constant reads as nonzero: integers against zero,
/// floats against either zero (NaN is nonzero), complex against either
/// part. Shared with the backend's constant-condition folding.
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Int(i) => *i != 0,
        Value::Float(f) => !f.is_zero(),
        // A complex number is nonzero when either part is.
        Value::Complex(re, im) => truthy(re) || truthy(im),
    }
}

/// The zero imaginary-part filler for a complex component type:
/// integer zero, or floating zero in the component's precision.
fn complex_zero(part: &Ty, long_double: LongDoubleKind) -> Option<Value> {
    if crate::check::is_integer(part) {
        Some(Value::Int(0))
    } else {
        Some(Value::Float(FloatValue::from_int(0, float_kind(part, long_double)?)))
    }
}

/// Convert an integer or float constant value to a real target type
/// (integer or floating), wrapping integers modularly.
fn cast_scalar(target: &Ty, v: Value, target_desc: Target) -> Option<Value> {
    match (target, v) {
        (t, Value::Int(i)) if crate::check::is_integer(t) => wrap(i, t, target_desc).map(Value::Int),
        (t, Value::Float(f)) if crate::check::is_integer(t) => f.to_int_trunc().and_then(|n| wrap(n, t, target_desc)).map(Value::Int),
        (t, Value::Int(i)) if float_kind(t, target_desc.long_double).is_some() => Some(Value::Float(FloatValue::from_int(i, float_kind(t, target_desc.long_double)?))),
        (t, Value::Float(f)) if float_kind(t, target_desc.long_double).is_some() => Some(Value::Float(f.convert_to(float_kind(t, target_desc.long_double)?))),
        _ => None,
    }
}

fn eval_unary(op: UnaryOp, operand: &CheckedExpr, sema: &Sema, interner: &Interner, addr_conds: bool) -> Option<Value> {
    match op {
        UnaryOp::Pos => eval_const_in(operand, sema, interner, addr_conds),
        UnaryOp::Neg => {
            let v = eval_const_in(operand, sema, interner, addr_conds)?;
            let ty = promoted(&operand.ty);
            match v {
                Value::Int(i) => wrap(i.wrapping_neg(), &ty, sema.target).map(Value::Int),
                Value::Float(f) => Some(Value::Float(f.neg())),
                // Component-wise negation.
                Value::Complex(re, im) => {
                    let part = corresponding_real(&ty);
                    Some(Value::Complex(Box::new(neg_part(*re, &part, sema.target)?), Box::new(neg_part(*im, &part, sema.target)?)))
                }
            }
        }
        UnaryOp::BitNot => {
            let v = eval_const_in(operand, sema, interner, addr_conds)?;
            // GNU `~x` on a complex operand is the conjugate: the real
            // part kept, the imaginary part negated.
            if let Value::Complex(re, im) = v {
                let part = corresponding_real(&promoted(&operand.ty));
                return Some(Value::Complex(re, Box::new(neg_part(*im, &part, sema.target)?)));
            }
            let Value::Int(i) = v else {
                return None;
            };
            let ty = promoted(&operand.ty);
            if crate::check::is_integer(&ty) {
                wrap(!i, &ty, sema.target).map(Value::Int)
            } else {
                None
            }
        }
        // A non-null address operand is known truthy where initializers
        // allow address constants, so `!&obj` folds to 0.
        UnaryOp::Not => match eval_const_in(operand, sema, interner, addr_conds) {
            Some(v) => Some(Value::Int(!truthy(&v) as i128)),
            None if addr_conds && address_nonzero(operand, sema) => Some(Value::Int(0)),
            None => None,
        },
        UnaryOp::Real | UnaryOp::Imag => match eval_const_in(operand, sema, interner, addr_conds)? {
            Value::Complex(re, im) => Some(if matches!(op, UnaryOp::Real) { *re } else { *im }),
            _ => None,
        },
        UnaryOp::Deref | UnaryOp::AddrOf | UnaryOp::PreInc | UnaryOp::PreDec | UnaryOp::PostInc | UnaryOp::PostDec => None,
    }
}

/// Whether a condition is an address constant that is never null at
/// runtime: a string or compound literal, or the address of an object
/// or function that is not weak. Only static initializers fold these
/// (see [`eval_static_init`]).
fn address_nonzero(expr: &CheckedExpr, sema: &Sema) -> bool {
    match &expr.kind {
        CheckedExprKind::Str(_) | CheckedExprKind::CompoundLiteral { .. } => true,
        CheckedExprKind::Cast { expr, .. } | CheckedExprKind::Convert { expr, .. } => address_nonzero(expr, sema),
        CheckedExprKind::Unary(UnaryOp::AddrOf, operand) => address_of_nonzero(operand, sema),
        _ => false,
    }
}

/// `&operand` is never null when the operand reaches a non-weak object
/// or function through members and array indexing: the address is the
/// object's plus a constant offset. A pointer base's value is unknown,
/// so only array bases qualify there.
fn address_of_nonzero(operand: &CheckedExpr, sema: &Sema) -> bool {
    match &operand.kind {
        CheckedExprKind::Ident(id) => {
            let binding = &sema.decls[id.0 as usize];
            !binding.weak && !binding.tls && matches!(binding.kind, BindingKind::Var | BindingKind::Func)
        }
        CheckedExprKind::Member { base, .. } => address_of_nonzero(base, sema),
        CheckedExprKind::Subscript { base, .. } => matches!(base.ty.unqualified(), Ty::Array { .. }) && address_of_nonzero(base, sema),
        _ => false,
    }
}

/// Negate one complex part value: floats flip the sign, integers wrap
/// in the component type.
fn neg_part(v: Value, part: &Ty, target: Target) -> Option<Value> {
    match v {
        Value::Int(i) => wrap(i.wrapping_neg(), part, target).map(Value::Int),
        Value::Float(f) => Some(Value::Float(f.neg())),
        Value::Complex(..) => None,
    }
}

fn eval_cast(ty: &Ty, operand: &CheckedExpr, sema: &Sema, interner: &Interner, addr_conds: bool) -> Option<Value> {
    let v = eval_const_in(operand, sema, interner, addr_conds)?;
    // Qualification is value-transparent: `const float` folds like `float`.
    let t = ty.unqualified();
    // A complex target takes the value as its real part over a zero
    // imaginary part; a complex source converts component-wise.
    if let Ty::Complex(inner) = t {
        let zero = complex_zero(inner, sema.target.long_double)?;
        return match v {
            Value::Complex(re, im) => Some(Value::Complex(
                Box::new(cast_scalar(inner, *re, sema.target)?),
                Box::new(cast_scalar(inner, *im, sema.target)?),
            )),
            scalar => Some(Value::Complex(Box::new(cast_scalar(inner, scalar, sema.target)?), Box::new(zero))),
        };
    }
    match v {
        // A complex to `_Bool` is nonzero when either part is (a bare
        // real-part conversion would miss a pure-imaginary value).
        Value::Complex(..) if t == &Ty::Bool => Some(Value::Int(truthy(&v) as i128)),
        // Any other complex to real conversion keeps the real part,
        // discarding the imaginary part.
        Value::Complex(re, _) => cast_scalar(t, *re, sema.target),
        scalar => cast_scalar(t, scalar, sema.target),
    }
}

/// Fold an integer constant expression to the value it has as type
/// `ty` (wrapping modularly, like a conversion): `switch` compares
/// duplicate `case` labels after conversion to the promoted
/// controlling type, so `case -1:` duplicates `case 4294967295:`
/// under an `unsigned` control.
///
/// Public for the same reason as [`promoted`]: lowering dispatches on
/// these exact values.
pub fn const_as(ty: &Ty, expr: &CheckedExpr, sema: &Sema, interner: &Interner) -> Option<i128> {
    match eval_cast(ty, expr, sema, interner, false)? {
        Value::Int(n) => Some(n),
        Value::Float(_) | Value::Complex(..) => None,
    }
}

fn eval_binary(op: BinaryOp, lhs: &CheckedExpr, rhs: &CheckedExpr, sema: &Sema, interner: &Interner, addr_conds: bool) -> Option<Value> {
    use BinaryOp::*;
    // Logical operators short-circuit, so an unevaluated operand need
    // not fold: `2 || 1 / 0` is the constant 1.
    if matches!(op, And | Or) {
        let l = eval_const_in(lhs, sema, interner, addr_conds)?;
        match op {
            And if !truthy(&l) => return Some(Value::Int(0)),
            Or if truthy(&l) => return Some(Value::Int(1)),
            _ => {}
        }
        let r = eval_const_in(rhs, sema, interner, addr_conds)?;
        return Some(Value::Int(truthy(&r) as i128));
    }
    let l = eval_const_in(lhs, sema, interner, addr_conds)?;
    let r = eval_const_in(rhs, sema, interner, addr_conds)?;
    let common = Sema::<'_>::uac_common_type_in(&promoted(&lhs.ty), &promoted(&rhs.ty), &sema.target);
    let cs = sema.target.char_signed;
    let ld = sema.target.long_double;
    match op {
        Shl | Shr => {
            let (Value::Int(a), Value::Int(b)) = (l, r) else {
                return None;
            };
            let ty = promoted(&lhs.ty);
            if !crate::check::is_integer(&ty) {
                return None;
            }
            let bits = width(&ty, sema.target)?;
            if b < 0 || b >= bits as i128 {
                return None;
            }
            let shifted = if matches!(op, Shl) {
                to_bits(a, &ty, sema.target)?.wrapping_shl(b as u32)
            } else if is_signed_int_in(&ty, cs)? {
                // Arithmetic shift, matching the target.
                from_bits(a as u128, &ty, sema.target)?.wrapping_shr(b as u32) as u128
            } else {
                to_bits(a, &ty, sema.target)?.wrapping_shr(b as u32)
            };
            from_bits(shifted, &ty, sema.target).map(Value::Int)
        }
        Eq | Ne | Lt | Le | Gt | Ge => {
            // Complex operands compare only for equality (ordered
            // forms are rejected at check time): equal when both parts
            // compare equal in the common component type.
            if is_complex(&common) {
                if !matches!(op, Eq | Ne) {
                    return None;
                }
                let part = corresponding_real(&common);
                let (re1, im1) = complex_parts(&l, &part, sema.target)?;
                let (re2, im2) = complex_parts(&r, &part, sema.target)?;
                let eq = scalar_eq(&re1, &re2, &part, sema.target)? && scalar_eq(&im1, &im2, &part, sema.target)?;
                return Some(Value::Int((if matches!(op, Eq) { eq } else { !eq }) as i128));
            }
            let result = match (&l, &r) {
                (Value::Int(a), Value::Int(b)) => {
                    if !crate::check::is_integer(&common) {
                        return None;
                    }
                    if is_signed_int_in(&common, cs)? {
                        let a = from_bits(*a as u128, &common, sema.target)?;
                        let b = from_bits(*b as u128, &common, sema.target)?;
                        compare_signed(op, a, b)
                    } else {
                        let a = to_bits(*a, &common, sema.target)?;
                        let b = to_bits(*b, &common, sema.target)?;
                        compare_u128(op, a, b)
                    }
                }
                // Floats (and int/float mixes) compare in the common
                // precision: operands convert to it first, per the usual
                // arithmetic conversions.
                (Value::Float(a), Value::Float(b)) => {
                    let kind = float_kind(&common, ld)?;
                    compare_float(op, a.convert_to(kind), b.convert_to(kind))
                }
                (Value::Int(a), Value::Float(b)) => {
                    let kind = float_kind(&common, ld)?;
                    compare_float(op, FloatValue::from_int(*a, kind), b.convert_to(kind))
                }
                (Value::Float(a), Value::Int(b)) => {
                    let kind = float_kind(&common, ld)?;
                    compare_float(op, a.convert_to(kind), FloatValue::from_int(*b, kind))
                }
                // Complex values never reach a real comparison (their
                // common type is complex, handled above).
                (Value::Complex(..), _) | (_, Value::Complex(..)) => return None,
            };
            Some(Value::Int(result as i128))
        }
        And | Or => {
            unreachable!("logical operators short-circuit above")
        }
        Comma => None,
        _ => {
            // Complex arithmetic is component-wise in the common
            // component type (only `+`/`-`/`*`/`/` reach here; the rest
            // are rejected at check time).
            if is_complex(&common) {
                return eval_complex_op(op, &l, &r, &common, sema.target);
            }
            match (l, r) {
                (Value::Int(a), Value::Int(b)) => {
                    if !crate::check::is_integer(&common) {
                        return None;
                    }
                    eval_int_op(op, a, b, &common, sema.target)
                }
                // Floats (and int/float mixes) compute in the common
                // precision, per the usual arithmetic conversions.
                (Value::Float(a), Value::Float(b)) => {
                    let kind = float_kind(&common, ld)?;
                    eval_float_op(op, a.convert_to(kind), b.convert_to(kind))
                }
                (Value::Int(a), Value::Float(b)) => {
                    let kind = float_kind(&common, ld)?;
                    eval_float_op(op, FloatValue::from_int(a, kind), b.convert_to(kind))
                }
                (Value::Float(a), Value::Int(b)) => {
                    let kind = float_kind(&common, ld)?;
                    eval_float_op(op, a.convert_to(kind), FloatValue::from_int(b, kind))
                }
                // Complex values take the `eval_complex_op` path above
                // (their common type is complex).
                (Value::Complex(..), _) | (_, Value::Complex(..)) => return None,
            }
        }
    }
}

/// Real and imaginary parts of a constant value for a complex
/// operation: complex values convert part-wise to the common
/// component type, scalars lift with a zero imaginary part.
fn complex_parts(v: &Value, part: &Ty, target: Target) -> Option<(Value, Value)> {
    match v {
        Value::Complex(re, im) => Some((cast_scalar(part, (**re).clone(), target)?, cast_scalar(part, (**im).clone(), target)?)),
        scalar => Some((cast_scalar(part, scalar.clone(), target)?, complex_zero(part, target.long_double)?)),
    }
}

/// Whether two scalar part values compare equal in a component type:
/// exact for integers (by signedness), IEEE for floats (so NaN parts
/// compare unequal).
fn scalar_eq(a: &Value, b: &Value, part: &Ty, target: Target) -> Option<bool> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => {
            if !crate::check::is_integer(part) {
                return None;
            }
            if is_signed_int_in(part, target.char_signed)? {
                let x = from_bits(*x as u128, part, target)?;
                let y = from_bits(*y as u128, part, target)?;
                Some(compare_signed(BinaryOp::Eq, x, y))
            } else {
                Some(compare_u128(BinaryOp::Eq, to_bits(*x, part, target)?, to_bits(*y, part, target)?))
            }
        }
        (Value::Float(x), Value::Float(y)) => {
            let kind = float_kind(part, target.long_double)?;
            Some(compare_float(BinaryOp::Eq, x.convert_to(kind), y.convert_to(kind)))
        }
        (Value::Int(x), Value::Float(y)) => {
            let kind = float_kind(part, target.long_double)?;
            Some(compare_float(BinaryOp::Eq, FloatValue::from_int(*x, kind), y.convert_to(kind)))
        }
        (Value::Float(x), Value::Int(y)) => {
            let kind = float_kind(part, target.long_double)?;
            Some(compare_float(BinaryOp::Eq, x.convert_to(kind), FloatValue::from_int(*y, kind)))
        }
        _ => None,
    }
}

/// One complex `+`/`-`/`*`/`/` in the common component type:
/// `(a+bi)+(c+di)` adds part-wise, `(a+bi)(c+di)` is
/// `(ac-bd)+(ad+bc)i`, and division scales by `c²+d²`. Float parts
/// round once per component operation; integer parts wrap modularly.
fn eval_complex_op(op: BinaryOp, l: &Value, r: &Value, common: &Ty, target: Target) -> Option<Value> {
    use BinaryOp::*;
    if !matches!(op, Add | Sub | Mul | Div) {
        return None;
    }
    let part = corresponding_real(common);
    let (a, b) = complex_parts(l, &part, target)?;
    let (c, d) = complex_parts(r, &part, target)?;
    if float_kind(&part, target.long_double).is_some() {
        let kind = float_kind(&part, target.long_double)?;
        let (a, b, c, d) = (as_float(&a, kind)?, as_float(&b, kind)?, as_float(&c, kind)?, as_float(&d, kind)?);
        let (re, im) = match op {
            Add => (eval_float_part(Add, a, c)?, eval_float_part(Add, b, d)?),
            Sub => (eval_float_part(Sub, a, c)?, eval_float_part(Sub, b, d)?),
            Mul => {
                let ac = eval_float_part(Mul, a, c)?;
                let bd = eval_float_part(Mul, b, d)?;
                let ad = eval_float_part(Mul, a, d)?;
                let bc = eval_float_part(Mul, b, c)?;
                (eval_float_part(Sub, ac, bd)?, eval_float_part(Add, ad, bc)?)
            }
            Div => {
                let ac = eval_float_part(Mul, a, c)?;
                let bd = eval_float_part(Mul, b, d)?;
                let bc = eval_float_part(Mul, b, c)?;
                let ad = eval_float_part(Mul, a, d)?;
                let cc = eval_float_part(Mul, c, c)?;
                let dd = eval_float_part(Mul, d, d)?;
                let den = eval_float_part(Add, cc, dd)?;
                let rn = eval_float_part(Add, ac, bd)?;
                let rni = eval_float_part(Sub, bc, ad)?;
                (eval_float_part(Div, rn, den)?, eval_float_part(Div, rni, den)?)
            }
            _ => return None,
        };
        Some(Value::Complex(Box::new(Value::Float(re)), Box::new(Value::Float(im))))
    } else if crate::check::is_integer(&part) {
        let (a, b, c, d) = (as_int(&a)?, as_int(&b)?, as_int(&c)?, as_int(&d)?);
        let (re, im) = match op {
            Add => (eval_int_part(Add, a, c, &part, target)?, eval_int_part(Add, b, d, &part, target)?),
            Sub => (eval_int_part(Sub, a, c, &part, target)?, eval_int_part(Sub, b, d, &part, target)?),
            Mul => {
                let ac = eval_int_part(Mul, a, c, &part, target)?;
                let bd = eval_int_part(Mul, b, d, &part, target)?;
                let ad = eval_int_part(Mul, a, d, &part, target)?;
                let bc = eval_int_part(Mul, b, c, &part, target)?;
                (eval_int_part(Sub, ac, bd, &part, target)?, eval_int_part(Add, ad, bc, &part, target)?)
            }
            Div => {
                let ac = eval_int_part(Mul, a, c, &part, target)?;
                let bd = eval_int_part(Mul, b, d, &part, target)?;
                let bc = eval_int_part(Mul, b, c, &part, target)?;
                let ad = eval_int_part(Mul, a, d, &part, target)?;
                let cc = eval_int_part(Mul, c, c, &part, target)?;
                let dd = eval_int_part(Mul, d, d, &part, target)?;
                let den = eval_int_part(Add, cc, dd, &part, target)?;
                if den == 0 {
                    return None;
                }
                let rn = eval_int_part(Add, ac, bd, &part, target)?;
                let rni = eval_int_part(Sub, bc, ad, &part, target)?;
                (eval_int_part(Div, rn, den, &part, target)?, eval_int_part(Div, rni, den, &part, target)?)
            }
            _ => return None,
        };
        Some(Value::Complex(Box::new(Value::Int(re)), Box::new(Value::Int(im))))
    } else {
        None
    }
}

/// One part value as a float in the given precision.
fn as_float(v: &Value, kind: FloatKind) -> Option<FloatValue> {
    match v {
        Value::Int(i) => Some(FloatValue::from_int(*i, kind)),
        Value::Float(f) => Some(f.convert_to(kind)),
        Value::Complex(..) => None,
    }
}

/// One part value as a raw integer.
fn as_int(v: &Value) -> Option<i128> {
    match v {
        Value::Int(i) => Some(*i),
        Value::Float(_) | Value::Complex(..) => None,
    }
}

/// One float component operation, rounded once.
fn eval_float_part(op: BinaryOp, a: FloatValue, b: FloatValue) -> Option<FloatValue> {
    match eval_float_op(op, a, b)? {
        Value::Float(f) => Some(f),
        _ => None,
    }
}

/// One integer component operation, wrapped to the component type.
fn eval_int_part(op: BinaryOp, a: i128, b: i128, part: &Ty, target: Target) -> Option<i128> {
    match eval_int_op(op, a, b, part, target)? {
        Value::Int(n) => Some(n),
        _ => None,
    }
}

fn compare_u128(op: BinaryOp, a: u128, b: u128) -> bool {
    use BinaryOp::*;
    match op {
        Eq => a == b,
        Ne => a != b,
        Lt => a < b,
        Le => a <= b,
        Gt => a > b,
        Ge => a >= b,
        _ => false,
    }
}

/// Compare two same-precision floats. A `None` ordering is an
/// unordered (NaN) pair, where only `!=` holds, per IEEE 754.
fn compare_float(op: BinaryOp, a: FloatValue, b: FloatValue) -> bool {
    use BinaryOp::*;
    let ord = match (a, b) {
        (FloatValue::F32(x), FloatValue::F32(y)) => x.partial_cmp(&y),
        (FloatValue::F64(x), FloatValue::F64(y)) => x.partial_cmp(&y),
        (FloatValue::F80(x), FloatValue::F80(y)) => x.partial_cmp(&y),
        (FloatValue::F128(x), FloatValue::F128(y)) => x.partial_cmp(&y),
        _ => return false,
    };
    match ord {
        None => matches!(op, Ne),
        Some(core::cmp::Ordering::Equal) => matches!(op, Eq | Le | Ge),
        Some(core::cmp::Ordering::Less) => matches!(op, Ne | Lt | Le),
        Some(core::cmp::Ordering::Greater) => matches!(op, Ne | Gt | Ge),
    }
}

fn compare_signed(op: BinaryOp, a: i128, b: i128) -> bool {
    compare_u128(op, a as u128 ^ (1u128 << 127), b as u128 ^ (1u128 << 127))
}

/// Float `+`/`-`/`*`/`/` in one precision (both sides already
/// converted by the caller); anything else (`%`, shifts, bitwise) is
/// not a float operation. Division by zero yields infinity (or NaN),
/// matching the target.
fn eval_float_op(op: BinaryOp, a: FloatValue, b: FloatValue) -> Option<Value> {
    use BinaryOp::*;
    macro_rules! binop {
        ($a:expr, $b:expr, $op:tt) => {
            match ($a, $b) {
                (FloatValue::F32(x), FloatValue::F32(y)) => FloatValue::F32((x $op y).value),
                (FloatValue::F64(x), FloatValue::F64(y)) => FloatValue::F64((x $op y).value),
                (FloatValue::F80(x), FloatValue::F80(y)) => FloatValue::F80((x $op y).value),
                (FloatValue::F128(x), FloatValue::F128(y)) => FloatValue::F128((x $op y).value),
                _ => return None,
            }
        };
    }
    let r = match op {
        Add => binop!(a, b, +),
        Sub => binop!(a, b, -),
        Mul => binop!(a, b, *),
        Div => binop!(a, b, /),
        _ => return None,
    };
    Some(Value::Float(r))
}

fn eval_int_op(op: BinaryOp, a: i128, b: i128, common: &Ty, target: Target) -> Option<Value> {
    use BinaryOp::*;
    let cs = target.char_signed;
    if is_signed_int_in(common, cs)? {
        let (a, b) = (from_bits(a as u128, common, target)?, from_bits(b as u128, common, target)?);
        let r = match op {
            Add => a.wrapping_add(b),
            Sub => a.wrapping_sub(b),
            Mul => a.wrapping_mul(b),
            Div => {
                if b == 0 {
                    return None;
                }
                a.wrapping_div(b)
            }
            Rem => {
                if b == 0 {
                    return None;
                }
                a.wrapping_rem(b)
            }
            BitAnd => (a as u128 & b as u128) as i128,
            BitOr => (a as u128 | b as u128) as i128,
            BitXor => (a as u128 ^ b as u128) as i128,
            _ => return None,
        };
        wrap(r, common, target).map(Value::Int)
    } else {
        let (a, b) = (to_bits(a, common, target)?, to_bits(b, common, target)?);
        let r = match op {
            Add => a.wrapping_add(b),
            Sub => a.wrapping_sub(b),
            Mul => a.wrapping_mul(b),
            Div => {
                if b == 0 {
                    return None;
                }
                a.wrapping_div(b)
            }
            Rem => {
                if b == 0 {
                    return None;
                }
                a.wrapping_rem(b)
            }
            BitAnd => a & b,
            BitOr => a | b,
            BitXor => a ^ b,
            _ => return None,
        };
        from_bits(r, common, target).map(Value::Int)
    }
}
