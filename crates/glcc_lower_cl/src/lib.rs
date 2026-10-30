//! `glcc_lower_cl`: MIR to Cranelift lowering.
//!
//! [`lower_module`] takes a [`glcc_mir::Module`] plus any
//! [`cranelift_module::Module`] (object for the real pipeline, JIT in
//! tests) and defines every function and global in it.
//!
//! Lowering is mechanical by construction: MIR mirrors Clif blocks and
//! block params, so control flow transcribes directly. The one real gap
//! is aggregates, which have MIR types but no Clif type. The convention
//! is uniform: **every aggregate-typed [`glcc_mir::Value`] lowers to the
//! address of a stack slot**, never to a Clif value.
//!
//! - [`glcc_mir::InstKind::Alloca`] creates the slot and yields its address.
//! - SSA aggregates ([`glcc_mir::InstKind::InsertValue`], aggregate
//!   [`glcc_mir::InstKind::Load`], aggregate calls) copy into fresh slots via
//!   unrolled loads/stores and yield the fresh address.
//! - [`glcc_mir::InstKind::ExtractValue`] loads a scalar element or computes
//!   an element address for a nested aggregate.
//! - Aggregate params arrive as caller-memory pointers and are copied
//!   into locals; aggregate returns use a hidden caller-provided
//!   out-pointer prepended to the signature (see [`ty::LoweredSig`]).
//!
//! Malformed or unrepresentable input panics with a plain message, like
//! the Cranelift backend: MIR reaching here has passed [`glcc_mir::verify`].

pub mod abi;
pub mod asm;
mod asm_native;
mod debug;
pub mod lower;
mod memflags;
pub mod quad;
pub mod ty;

pub use abi::{Arch, Os, Target};
pub use lower::{NarrowReloc, NarrowTarget, WinUnwindEntry, lower_module, lower_module_collecting, lower_module_selected, lower_module_with_debug};
pub use ty::{LoweredSig, clif_ty, lower_sig};
