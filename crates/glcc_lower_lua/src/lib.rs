//! `glcc_lower_lua`: MIR to portable Lua 5.1 source.
//!
//! This backend exists for experiments, teaching, and quick "compile C to
//! something runnable anywhere" demos. It is not a replacement for the
//! Cranelift backend: C addresses are emulated in a Lua byte table, integers
//! are Lua numbers, and no native ABI is exposed.
//!
//! [`lower_module`] returns a complete Lua script. When the module defines
//! `main`, the script calls it at the end (passing `argc`/`argv` when the
//! signature asks for them).

mod lower;
mod runtime;

pub use lower::lower_module;
