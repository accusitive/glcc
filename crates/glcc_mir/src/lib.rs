//! `glcc_mir`: thin SSA layer above Cranelift with aggregate types.
//!
//! MIR sits between the checked frontend (`glcc_lower_mir` lowers C
//! to it) and the Cranelift lowering (`glcc_lower_cl`), giving our own
//! optimizations a place to run: passes like inlining, constant
//! folding, DCE, and CFG simplification.
//!
//! Lowering out of (and into) MIR is mechanical by design: scalar ops
//! map 1:1 onto Cranelift, aggregates become addressed memory.
//!
//! Aggregates are first-class: struct/array values exist in SSA and in
//! memory, navigated by `GetElementPtr`. Lowering explodes them to
//! Cranelift stack slots, offsets, and scalar ops.
//!
//! Temporary driver: `crates/mir_driver` builds demo functions, runs the
//! pass pipeline, and shows the Cranelift mapping.

pub mod builder;
pub mod coroutine;
pub mod debug;
pub mod display;
pub mod ir;
pub mod optimize;
pub mod pass;
pub mod passes;
pub mod ty;
pub mod verify;

pub use builder::{FunctionBuilder, ModuleBuilder};
pub use debug::{BaseEncoding, DebugType, DebugTypeId, Enumerator, File, FunctionDebug, GlobalDebug, Location, Member, MemberLocation, Qualifier, ScopeDebug, VarDebug, VarHome};
pub use ir::{
    AsmBlock, AsmOperand, AtomicRmwOp, Block, BlockData, Const, DataReloc, Def, Extern, ExternData, ExternEffect, FcmpPred, Func, Function, Global, GlobalData, IcmpPred, Inst,
    InstData, InstKind, MemFlags, Module, Terminator, Value, ValueData, ValueOrigin,
};
pub use optimize::{OptLevel, optimize};
pub use pass::{FunctionPass, ModulePass, PassManager};
pub use ty::{FloatKind, GepIndex, Signature, Ty, dense_field_offsets, struct_field_offsets};
