//! clang-family extension spellings.

/// clang's `_Float32`/`_Float64` type spellings.
pub const FLOAT_SPELLINGS: &[&str] = &["_Float32", "_Float64"];

pub fn is_float_spelling(word: &str) -> bool {
    FLOAT_SPELLINGS.contains(&word)
}
