//! Implicit builtin prototypes: the spellings no header declares,
//! with the exact prototype GCC gives them. `declare_builtin_if_called`
//! (in the parser) declares these on first call so the later check
//! resolves them like any declared function.

use alloc::vec::Vec;
use alloc::boxed::Box;

use crate::parse::{PtrSize, Ty, TypeQualifiers};

/// Prototype of an implicit builtin, or `None` when `name` is not one.
pub(crate) fn builtin_prototype(name: &str) -> Option<(Ty, Vec<Ty>, bool)> {
    let (ret, params, variadic) = match name {
        "__builtin_va_start" | "__builtin_c23_va_start" => (Ty::Void, vec![Ty::BuiltinVaList], true),
        "__builtin_va_end" => (Ty::Void, vec![Ty::BuiltinVaList], false),
        "__builtin_va_copy" => (Ty::Void, vec![Ty::BuiltinVaList, Ty::BuiltinVaList], false),
        "__builtin_clz" => (Ty::Int, vec![Ty::UnsignedInt], false),
        "__builtin_clzl" => (Ty::Int, vec![Ty::UnsignedLong], false),
        "__builtin_clzll" => (Ty::Int, vec![Ty::UnsignedLongLong], false),
        "__builtin_ctz" => (Ty::Int, vec![Ty::UnsignedInt], false),
        "__builtin_ctzl" => (Ty::Int, vec![Ty::UnsignedLong], false),
        "__builtin_ctzll" => (Ty::Int, vec![Ty::UnsignedLongLong], false),
        "__builtin_popcount" => (Ty::Int, vec![Ty::UnsignedInt], false),
        "__builtin_popcountl" => (Ty::Int, vec![Ty::UnsignedLong], false),
        "__builtin_popcountll" => (Ty::Int, vec![Ty::UnsignedLongLong], false),
        "__builtin_clrsb" => (Ty::Int, vec![Ty::Int], false),
        "__builtin_clrsbl" => (Ty::Int, vec![Ty::Long], false),
        "__builtin_clrsbll" => (Ty::Int, vec![Ty::LongLong], false),
        "__builtin_parity" => (Ty::Int, vec![Ty::UnsignedInt], false),
        "__builtin_parityl" => (Ty::Int, vec![Ty::UnsignedLong], false),
        "__builtin_parityll" => (Ty::Int, vec![Ty::UnsignedLongLong], false),
        "__builtin_copysign" => (Ty::Double, vec![Ty::Double, Ty::Double], false),
        "__builtin_copysignf" => (Ty::Float, vec![Ty::Float, Ty::Float], false),
        "__builtin_copysignl" => (Ty::LongDouble, vec![Ty::LongDouble, Ty::LongDouble], false),
        "__builtin_inff" => (Ty::Float, vec![], false),
        "__builtin_inf" => (Ty::Double, vec![], false),
        "__builtin_infl" => (Ty::LongDouble, vec![], false),
        "__builtin_huge_valf" => (Ty::Float, vec![], false),
        "__builtin_huge_val" => (Ty::Double, vec![], false),
        "__builtin_huge_vall" => (Ty::LongDouble, vec![], false),
        "__builtin_nanf" => (
            Ty::Float,
            vec![Ty::Pointer {
                qualifiers: TypeQualifiers::empty(),
                size: PtrSize::Default,
                pointee: Box::new(Ty::Char),
            }],
            false,
        ),
        "__builtin_nan" | "__builtin_nanl" => (
            if name == "__builtin_nanl" {
                Ty::LongDouble
            } else {
                Ty::Double
            },
            vec![Ty::Pointer {
                qualifiers: TypeQualifiers::empty(),
                size: PtrSize::Default,
                pointee: Box::new(Ty::Char),
            }],
            false,
        ),
        "__builtin_floor" | "__builtin_ceil" => (Ty::Double, vec![Ty::Double], false),
        "__builtin_floorf" | "__builtin_ceilf" => (Ty::Float, vec![Ty::Float], false),
        "__builtin_floorl" | "__builtin_ceill" => (Ty::LongDouble, vec![Ty::LongDouble], false),
        "__builtin_abs" => (Ty::Int, vec![Ty::Int], false),
        "__builtin_labs" => (Ty::Long, vec![Ty::Long], false),
        "__builtin_llabs" => (Ty::LongLong, vec![Ty::LongLong], false),
        "__builtin_fabs" => (Ty::Double, vec![Ty::Double], false),
        "__builtin_fabsf" => (Ty::Float, vec![Ty::Float], false),
        "__builtin_fabsl" => (Ty::LongDouble, vec![Ty::LongDouble], false),
        "__builtin_trap" => (Ty::Int, vec![], false),
        "__builtin_bswap16" => (Ty::UnsignedShort, vec![Ty::UnsignedShort], false),
        "__builtin_bswap32" => (Ty::UnsignedInt, vec![Ty::UnsignedInt], false),
        "__builtin_bswap64" => (Ty::UnsignedLongLong, vec![Ty::UnsignedLongLong], false),
        // Complex conjugates: `__builtin_conjf` over
        // `float _Complex`, `__builtin_conj` over
        // `double _Complex`, `__builtin_conjl` over
        // `long double _Complex`.
        "__builtin_conjf" => (Ty::Complex(Box::new(Ty::Float)), vec![Ty::Complex(Box::new(Ty::Float))], false),
        "__builtin_conj" => (Ty::Complex(Box::new(Ty::Double)), vec![Ty::Complex(Box::new(Ty::Double))], false),
        "__builtin_conjl" => (Ty::Complex(Box::new(Ty::LongDouble)), vec![Ty::Complex(Box::new(Ty::LongDouble))], false),
        // Complex accessors (`__builtin_creal/cimag/cabs` and
        // `f`/`l` variants): the real part, the imaginary part,
        // and the magnitude of a complex number.
        "__builtin_crealf" => (Ty::Float, vec![Ty::Complex(Box::new(Ty::Float))], false),
        "__builtin_creal" => (Ty::Double, vec![Ty::Complex(Box::new(Ty::Double))], false),
        "__builtin_creall" => (Ty::LongDouble, vec![Ty::Complex(Box::new(Ty::LongDouble))], false),
        "__builtin_cimagf" => (Ty::Float, vec![Ty::Complex(Box::new(Ty::Float))], false),
        "__builtin_cimag" => (Ty::Double, vec![Ty::Complex(Box::new(Ty::Double))], false),
        "__builtin_cimagl" => (Ty::LongDouble, vec![Ty::Complex(Box::new(Ty::LongDouble))], false),
        "__builtin_cabsf" => (Ty::Float, vec![Ty::Complex(Box::new(Ty::Float))], false),
        "__builtin_cabs" => (Ty::Double, vec![Ty::Complex(Box::new(Ty::Double))], false),
        "__builtin_cabsl" => (Ty::LongDouble, vec![Ty::Complex(Box::new(Ty::LongDouble))], false),
        _ => return None,
    };
    Some((ret, params, variadic))
}
