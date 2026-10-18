//! `glcc_lower_mir`: checked C to MIR lowering.
//!
//! This is the new backend frontend: it walks
//! [`glcc_frontend::sema::Sema`]'s entities and checked bodies the way
//! the Cranelift backend used to, but emits MIR instead of Clif, so
//! optimization passes run before `glcc_lower_cl` takes it the rest of
//! the way down.
//!
//! Frame discipline is deliberately `-O0`: every parameter is spilled to
//! a slot on entry and every local lives in memory, so control flow
//! needs no SSA value plumbing (branches carry no arguments). The
//! `Mem2Reg` pass promotes what escape and volatility analysis allows.
//!
//! Anything not yet lowered fails loudly with `todo!`, never silently.

pub mod debug;
pub mod lower;
pub mod ty;

pub use lower::{Lowerer, lower_translation_unit, naked_asm_text};
