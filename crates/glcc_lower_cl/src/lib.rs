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
mod debug;
pub mod ehframe;
pub mod initarray;
pub mod lower;
mod memflags;
pub mod quad;
pub mod ty;

pub use abi::{Arch, Os, Target};
pub use ehframe::{emit_unwind, FuncEh};
pub use initarray::emit_init_array;
pub use lower::{NarrowReloc, NarrowTarget, WinUnwindEntry, lower_module, lower_module_collecting, lower_module_collecting_with_ids, lower_module_selected, lower_module_with_debug};
pub use ty::{LoweredSig, clif_ty, lower_sig};

/// Cranelift settings every embedder of this backend should start
/// from, merged into the JIT builder or flag builder before any
/// embedder-specific flags.
///
/// Stack probing is always on: aggregate SSA values lower to one
/// stack slot per `InsertValue`, so wide-`_BitInt` code can frame
/// well past Cranelift's 4KB probe threshold, and without a probe the
/// prologue jumps straight past the single Windows guard page into
/// uncommitted stack (an undeliverable access violation). Linux
/// survives the same code only because the kernel grows the stack
/// VMA for in-frame faults.
pub fn codegen_flags() -> Vec<(&'static str, &'static str)> {
    vec![
        ("preserve_frame_pointers", "true"),
        ("enable_probestack", "true"),
        ("probestack_strategy", "inline"),
    ]
}
