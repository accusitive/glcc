//! DWARF emission for the MIR compilation pipeline.
//!
//! `glcc_lower_cl` collects [`LoweredDebug`] while it defines every
//! Clif function and data object; [`emit_with_debug`] appends the
//! `.debug_*` sections to the finished `cranelift-object` product.

mod debug;

pub use debug::{
    emit_with_debug, FuncDebug, GlobalDebug, LoweredDebug, ScopeDebug, SrcLocRow, VarDebug,
    VarLoc,
};
