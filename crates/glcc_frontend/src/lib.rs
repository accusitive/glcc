//! glcc frontend: C lexer, parser, semantic analysis, constant
//! evaluation, and target-aware layout.
//!
//! The backend drives this crate by holding a [`sema::Sema`], feeding
//! source through [`parse::Parser`], and consuming the checked bodies
//! (`sema.bodies`), initializers (`sema.inits`), and declarations
//! (`sema.decls`) — or the s-expression [`dump`] for debugging.
//!
//! `std` is enabled by default. With `--no-default-features` the crate
//! is `no_std` + `alloc`: hash maps come from `hashbrown`, diagnostic
//! rendering to stderr is disabled (panics still carry the message),
//! and the diagnostic-emitted flag becomes process-global instead of
//! thread-local.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod check;
pub mod dump;
pub mod eval;
pub mod lang;
pub mod layout;
pub mod lexer;
pub mod parse;
pub mod sema;
pub mod span;
pub mod target;

use alloc::string::String;

/// Preprocessed C source bytes as the frontend's string view.
///
/// C source is a byte stream, not text: a `"\xff"` literal's raw byte
/// is not valid UTF-8, and compilers pass such bytes through verbatim.
/// Valid UTF-8 decodes unchanged so diagnostics stay readable;
/// anything else falls back to Latin-1, where each byte is one `char`
/// value. String and character lowering interprets `char` values up
/// to U+00FF as their Latin-1 byte, so raw high bytes round-trip
/// (`eval::string_bytes` / `eval::char_constant_in`).
pub fn decode_source(bytes: &[u8]) -> String {
    match core::str::from_utf8(bytes) {
        Ok(text) => String::from(text),
        Err(_) => bytes.iter().map(|&b| b as char).collect(),
    }
}

#[cfg(all(test, feature = "std"))]
mod tests;
