//! Language-neutral debug metadata.
//!
//! MIR is meant to be largely independent of C, so debug info rides on
//! plain data rather than frontend entities: source positions are
//! `(file, line, column)` triples over a per-module file table, and
//! types are described structurally in [`DebugType`] with names and
//! DWARF-shaped details filled in by lowering. Passes carry the
//! records along; only the final DWARF writer interprets them.

use crate::ir::{Global, Value};

/// One source position: file index into [`crate::ir::Module::files`],
/// 1-based line, 1-based column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Location {
    pub file: u32,
    pub line: u32,
    pub column: u32,
}

/// One source file. The path is stored as written; the DWARF writer
/// splits it into a directory and a file name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct File {
    pub path: String,
}

/// Base encoding of a scalar debug type, mirroring `DW_ATE_*` where a
/// target-independent choice exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseEncoding {
    Boolean,
    Signed,
    Unsigned,
    SignedChar,
    UnsignedChar,
    Float,
    ComplexFloat,
}

/// Type qualifier DIE wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qualifier {
    Const,
    Volatile,
    Restrict,
}

/// Where a member lives inside its aggregate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemberLocation {
    /// Byte offset from the aggregate start.
    Byte(u64),
    /// Bit-field: struct-relative bit offset and width.
    Bit { offset: u64, size: u64 },
}

/// One aggregate member. Nameless non-bit-field members are anonymous
/// struct/union members; nameless bit-fields never reach here.
#[derive(Debug, Clone)]
pub struct Member {
    pub name: Option<String>,
    pub ty: DebugTypeId,
    pub location: MemberLocation,
}

/// One enum constant.
#[derive(Debug, Clone)]
pub struct Enumerator {
    pub name: String,
    pub value: i128,
}

/// Index into [`crate::ir::Module::debug_types`].
pub type DebugTypeId = u32;

/// DWARF-shaped type description. Structurally identical types share
/// one entry, so ids compare by deduplication at insertion, not by
/// structural equality here.
#[derive(Debug, Clone)]
pub enum DebugType {
    Base {
        name: &'static str,
        encoding: BaseEncoding,
        size: u64,
    },
    Pointer {
        /// `None` for `void *`, like gcc.
        pointee: Option<DebugTypeId>,
    },
    Qualified {
        qualifier: Qualifier,
        inner: DebugTypeId,
    },
    Array {
        elem: DebugTypeId,
        /// `None` for `[]`, VLA, and `[*]` bounds.
        count: Option<u64>,
    },
    Struct {
        name: Option<String>,
        is_union: bool,
        size: u64,
        /// Incomplete definition: emitted as a declaration DIE with no
        /// members.
        declaration: bool,
        members: Vec<Member>,
    },
    Enum {
        name: Option<String>,
        repr: DebugTypeId,
        size: u64,
        signed: bool,
        declaration: bool,
        enumerators: Vec<Enumerator>,
    },
    Func {
        /// `None` for a void return.
        ret: Option<DebugTypeId>,
        params: Vec<DebugTypeId>,
        variadic: bool,
    },
}

/// Where one variable lives between the MIR and machine levels.
#[derive(Debug, Clone, Copy)]
pub enum VarHome {
    /// The slot allocated by the named `Alloca` result value.
    Slot(Value),
    /// Module storage.
    Global(Global),
}

/// One variable or parameter of a function.
#[derive(Debug, Clone)]
pub struct VarDebug {
    pub name: Option<String>,
    pub decl: Option<Location>,
    pub debug_ty: Option<DebugTypeId>,
    pub home: VarHome,
}

/// One lexical block. `start` is the opening brace (inclusive) and
/// `end` the closing one (exclusive); machine ranges are derived from
/// rows inside that range at emission time.
#[derive(Debug, Clone)]
pub struct ScopeDebug {
    pub start: Location,
    pub end: Location,
    pub vars: Vec<VarDebug>,
    pub children: Vec<ScopeDebug>,
}

/// Debug record of one defined function: declaration position, ordered
/// parameters, whole-body variables, and nested blocks.
#[derive(Debug, Clone, Default)]
pub struct FunctionDebug {
    pub decl: Option<Location>,
    pub params: Vec<VarDebug>,
    pub vars: Vec<VarDebug>,
    pub scopes: Vec<ScopeDebug>,
}

/// Debug record of one module-level object (file-scope or block-scope
/// static). The type lives on the owning `GlobalData`; `name` is the
/// source spelling (the symbol name may carry a `DeclId` suffix).
#[derive(Debug, Clone, Default)]
pub struct GlobalDebug {
    pub name: Option<String>,
    pub decl: Option<Location>,
    pub debug_ty: Option<DebugTypeId>,
}
