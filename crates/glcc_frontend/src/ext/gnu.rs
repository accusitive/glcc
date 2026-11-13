//! GNU-family extension spellings (`__attribute__`, `__asm__`,
//! alternate qualifier spellings, `__inline__`, ...). The catch-all
//! family: anything [`super::spelling_family`] leaves unclassified
//! gates through [`LangOpts::gnu`](crate::lang::LangOpts::gnu).

use crate::parse::TypeQualifier;

/// GNU alternate qualifier spellings, accepted everywhere the
/// standard spelling is. System headers use these pervasively
/// (`void *memcpy(void *__restrict, ...)`).
pub const QUALIFIER_SPELLINGS: &[(&str, TypeQualifier)] = &[
    ("__const__", TypeQualifier::Const),
    ("__const", TypeQualifier::Const),
    ("__restrict__", TypeQualifier::Restrict),
    ("__restrict", TypeQualifier::Restrict),
    ("__volatile__", TypeQualifier::Volatile),
    ("__volatile", TypeQualifier::Volatile),
];

pub fn qualifier_of(word: &str) -> Option<TypeQualifier> {
    QUALIFIER_SPELLINGS.iter().find(|(spelling, _)| *spelling == word).map(|(_, q)| *q)
}

/// GCC x86's named address spaces (`__seg_fs`/`__seg_gs` and their
/// kernel-mode spellings), which the kernel's per-cpu code qualifies
/// pointee types with under `CONFIG_CC_HAS_NAMED_AS`. Their effect
/// (segment-relative addressing) is erased here; see `peek_named_as`.
pub const NAMED_AS_SPELLINGS: &[&str] = &["__seg_fs", "__seg_gs", "__seg_kfs", "__seg_kgs"];

pub fn is_named_as_spelling(word: &str) -> bool {
    NAMED_AS_SPELLINGS.contains(&word)
}

/// Alternate `inline` spellings (`static __inline` definitions in
/// system headers).
pub fn is_inline_spelling(word: &str) -> bool {
    matches!(word, "__inline__" | "__inline")
}
