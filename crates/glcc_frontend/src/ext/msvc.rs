//! MSVC-family extension spellings (`__declspec`, calling
//! conventions, `__intN`, `__forceinline`, pointer-size attributes).

/// MSVC keywords the lexer keeps as identifiers and the parser gates
/// through [`LangOpts::ms`](crate::lang::LangOpts::ms).
pub const KEYWORDS: &[&str] = &[
    "__int8",
    "__int16",
    "__int32",
    "__int64",
    "__declspec",
    "_declspec",
    "__pragma",
    "__cdecl",
    "__stdcall",
    "__fastcall",
    "__thiscall",
    "__vectorcall",
    "__clrcall",
    "__forceinline",
    "__ptr32",
    "__ptr64",
    "__unaligned",
];

pub fn is_keyword(word: &str) -> bool {
    KEYWORDS.contains(&word)
}

/// MSVC's `__forceinline`.
pub fn is_inline_spelling(word: &str) -> bool {
    word == "__forceinline"
}

/// MSVC's pointer-size attributes, which follow the `*` they modify.
pub const PTR_SIZE_SPELLINGS: &[(&str, crate::parse::PtrSize)] =
    &[("__ptr32", crate::parse::PtrSize::Ptr32), ("__ptr64", crate::parse::PtrSize::Ptr64)];
