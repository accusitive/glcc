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

/// Alternate `inline` spellings (`static __inline` definitions in
/// system headers).
pub fn is_inline_spelling(word: &str) -> bool {
    matches!(word, "__inline__" | "__inline")
}
