//! Expression checking: annotate `parse::Expr` with types.
//!
//! [`Sema::check_expr`] elaborates a parsed expression into a
//! [`CheckedExpr`], inserting [`CheckedExprKind::Convert`] nodes wherever
//! C requires an implicit conversion (the `Promote` analogue this design
//! is adapted from). Binary operators meet in
//! [`Sema::usual_arithmetic_conversions`], which needs operand types from
//! checking just as checking needs it for operator result types.
//!
//! Scope note: annotation only. Integer literal types come from their
//! values (see `crate::eval`); `sizeof` folds through `crate::layout`
//! where sizes are known. String lengths evaluate escape-aware (see
//! `crate::eval::string_length`), so `sizeof("ab")` and `char s[]`
//! completion fold. Target facts (per-target sizes, `size_t`,
//! plain-`char` signedness) come from `Sema::target`; the few
//! target-free helpers default to LP64. Brace-list initializers
//! check member-by-member and need no layout, only member types
//! and positions.

use alloc::vec::Vec;
use alloc::{boxed::Box, format, vec};

use string_interner::symbol::SymbolU32;

use crate::{
    lexer::{CharPrefix, FloatSuffix, Interner},
    parse::{ArrayBound, AssignOp, BinaryOp, BlockItem, Designator, Expr, GenericAssoc, InitItem, Initializer, Ty, TypeQualifier, UnaryOp},
    sema::{Binding, BindingKind, Member, Sema, StructId, types_compatible},
    span::{CompileError, HasSpan, Span},
    target::{Arch, Os},
};

/// An expression annotated with its type. Conversion nodes make implicit
/// conversions explicit so later phases never re-derive them.
//
// Note: derived equality includes spans, so only span-erased trees (see
// `erased`) may be compared — in practice just array bounds in `Ty`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckedExpr {
    pub ty: Ty,
    pub span: Span,
    pub kind: CheckedExprKind,
}

impl HasSpan for CheckedExpr {
    fn span(&self) -> Span {
        self.span
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RmwOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Nand,
    Min,
    Max,
}

/// Which floating-point classification a `FpClass` node computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpPredicate {
    IsNan,
    /// `__builtin_isinf`: `1` for either infinity, like the `isinf`
    /// macro.
    IsInf,
    /// `__builtin_isinf_sign`: `1` for `+inf`, `-1` for `-inf`, `0`
    /// otherwise (what `<math.h>`'s `isinf` macro expands to).
    IsInfSign,
    /// `__builtin_isfinite`: `1` for anything but NaN and infinities.
    IsFinite,
    /// `__builtin_isnormal`: `1` for finite nonzero values that are not
    /// subnormal.
    IsNormal,
    /// `__builtin_signbit`: the sign bit, so `-0.0` counts.
    SignBit,
}

/// Which GCC overflow builtin an `Overflow` node computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowOp {
    Add,
    Sub,
    Mul,
}

/// The GCC typed overflow builtins (`__builtin_sadd_overflow`,
/// `__builtin_umulll_overflow`, ...): `[su](add|sub|mul)(l|ll)?`
/// between `__builtin_` and `_overflow`. Each is the generic builtin
/// with its operand type pinned (`s`/`u` and the width suffix).
fn typed_overflow_builtin(name: &str) -> Option<(OverflowOp, Ty)> {
    let rest = name.strip_prefix("__builtin_")?;
    let (signed, rest) = match rest.as_bytes().first()? {
        b's' => (true, &rest[1..]),
        b'u' => (false, &rest[1..]),
        _ => return None,
    };
    let (op, rest) = if let Some(rest) = rest.strip_prefix("add") {
        (OverflowOp::Add, rest)
    } else if let Some(rest) = rest.strip_prefix("sub") {
        (OverflowOp::Sub, rest)
    } else if let Some(rest) = rest.strip_prefix("mul") {
        (OverflowOp::Mul, rest)
    } else {
        return None;
    };
    let width = rest.strip_suffix("_overflow")?;
    let ty = match (signed, width) {
        (true, "") => Ty::Int,
        (true, "l") => Ty::Long,
        (true, "ll") => Ty::LongLong,
        (false, "") => Ty::UnsignedInt,
        (false, "l") => Ty::UnsignedLong,
        (false, "ll") => Ty::UnsignedLongLong,
        _ => return None,
    };
    Some((op, ty))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicOp {
    LoadN {
        sized: Option<u8>,
    },
    Load,
    /// C11 `__c11_atomic_init(ptr, val)`: a store with no memory order.
    Init,
    StoreN {
        sized: Option<u8>,
    },
    Store,
    ExchangeN {
        sized: Option<u8>,
    },
    Exchange,
    CmpXchgN {
        sized: Option<u8>,
    },
    CmpXchg,
    Rmw {
        op: RmwOp,
        fetch: bool,
        sized: Option<u8>,
    },
    Fence {
        signal: bool,
    },
    LockFree {
        always: bool,
    },
    TestAndSet,
    Clear,
}

/// How a source pointee converts to a target pointee by assignment
/// (see `Sema::ptr_assign_compatible`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PointeeConversion {
    /// Incompatible.
    No,
    /// Compatible, qualifiers added at most at the outermost level.
    Yes,
    /// Compatible, but qualifiers added deeper (a GCC warning).
    Warns,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckedExprKind {
    /// A fully resolved variable or function: just its binding.
    Ident(crate::sema::DeclId),
    Int(crate::lexer::IntLiteral),
    Float(crate::lexer::FloatLiteral),
    Char(crate::lexer::CharLiteral),
    Str(Vec<crate::lexer::StringLiteral>),
    Unary(UnaryOp, Box<CheckedExpr>),
    Binary(BinaryOp, Box<CheckedExpr>, Box<CheckedExpr>),
    Assign(AssignOp, Box<CheckedExpr>, Box<CheckedExpr>),
    /// `lhs op= rhs` with `op` other than `=`: the lvalue evaluates
    /// once (its address), its current value converts to `op_ty`,
    /// combines with the `rhs` (converted to `rhs_ty`), and the result
    /// converts back to the lvalue's type (C17 §6.5.16.2). Keeping the
    /// operands separate is what allows the single evaluation:
    /// rewriting to `lhs = lhs op rhs` would re-run the lvalue's side
    /// effects.
    CompoundAssign {
        op: AssignOp,
        lhs: Box<CheckedExpr>,
        rhs: Box<CheckedExpr>,
        /// Boxed to keep the variant small: unboxed types would make
        /// it the largest variant and grow every checker frame.
        op_ty: Box<Ty>,
        rhs_ty: Box<Ty>,
    },
    Ternary {
        cond: Box<CheckedExpr>,
        then: Box<CheckedExpr>,
        els: Box<CheckedExpr>,
    },
    /// GNU `x ?: y`: the condition's value is the result when nonzero
    /// (evaluated once — the lowering keeps it in a slot), `els`
    /// otherwise.
    Elvis {
        cond: Box<CheckedExpr>,
        els: Box<CheckedExpr>,
    },
    Call {
        func: Box<CheckedExpr>,
        args: Vec<CheckedExpr>,
    },
    /// `__builtin_await(handle)` (glcc extension): suspend until the
    /// awaited future completes, yielding its result. The operand is a
    /// value of some `_Future(T)` handle type; this node's type is `T`.
    Await(Box<CheckedExpr>),
    /// A GCC atomic builtin call (`__atomic_load_n`, `__atomic_store_n`):
    /// the operation plus all checked operands (pointer first), since
    /// no declared function stands behind it.
    Atomic {
        op: AtomicOp,
        args: Vec<CheckedExpr>,
    },
    /// A GCC overflow builtin (`__builtin_mul_overflow` and friends):
    /// the operation, both operands, and the pointer receiving the
    /// wrapped result. The pointed-to type fixes the result's width
    /// and signedness; the operands keep their own types.
    Overflow {
        op: OverflowOp,
        lhs: Box<CheckedExpr>,
        rhs: Box<CheckedExpr>,
        res: Box<CheckedExpr>,
        /// The pointed-to type the result converts to.
        res_ty: Ty,
    },
    /// `alloca`/`__builtin_alloca`: the requested byte count (already
    /// converted to `size_t`). The result is `void *`.
    Alloca {
        size: Box<CheckedExpr>,
    },
    /// `__builtin_frame_address`/`__builtin_return_address`: the frame
    /// or return address of the `level`-th enclosing frame, as
    /// `void *`. The level is a constant (GCC requires one); lowering
    /// walks the frame-pointer chain.
    FrameAddress {
        level: u64,
        returns_address: bool,
    },
    /// A type-generic bit count (`__builtin_clzg`, `__builtin_ctzg`):
    /// the operand keeps its own width (no promotion), with an
    /// optional zero-fallback value (already converted to `int`).
    Clz {
        ctz: bool,
        arg: Box<CheckedExpr>,
        fallback: Option<Box<CheckedExpr>>,
    },
    /// A folded `__builtin_constant_p(arg)`: whether `arg` is a
    /// compile-time constant. The argument is checked but discarded
    /// (GCC never evaluates it, side effects included), so only the
    /// folded 0/1 remains.
    ConstantP(bool),
    /// A folded `__builtin_classify_type(arg)`: GCC's type-class code
    /// for the argument's type. The argument is checked but discarded
    /// (GCC never evaluates it), so only the folded code remains.
    ClassifyType(u8),
    /// A floating-point classification (`__builtin_isnan`,
    /// `__builtin_isinf`, ..., `__builtin_signbit`): single evaluation,
    /// unlike an `x != x` desugaring. The operand keeps its own type;
    /// integers never reach here (they fold to `ConstantP(false)` at
    /// check time).
    FpClass {
        op: FpPredicate,
        arg: Box<CheckedExpr>,
    },
    /// `__builtin_fpclassify(nan, inf, normal, subnormal, zero, x)`: the
    /// `int` code for `x`'s category, supplied by the caller (the
    /// `<math.h>` macro passes `FP_NAN`...`FP_ZERO`). Single evaluation
    /// of `x`; the codes evaluate in order (they are constants in
    /// practice).
    FpClassify {
        codes: [Box<CheckedExpr>; 5],
        arg: Box<CheckedExpr>,
    },
    Subscript {
        base: Box<CheckedExpr>,
        index: Box<CheckedExpr>,
    },
    Member {
        base: Box<CheckedExpr>,
        /// Path of member indices from the base struct: a direct field
        /// is one element; fields promoting through anonymous
        /// struct/union members append one index per level crossed.
        /// Codegen maps this through layout to an offset.
        path: Vec<usize>,
        arrow: bool,
        /// Total byte offset of the member from the base object, summing
        /// each anonymous level's offset the way `codegen` lowers them.
        /// (Anonymous nesting means `path` alone no longer suffices:
        /// the last index is relative to its enclosing struct.)
        offset: u64,
        /// `(bit index within the unit, width)` when the final member
        /// is a bit-field. Loads/stores must mask, not treat it as a
        /// plain scalar lvalue.
        bitfield: Option<(u32, u32)>,
        /// The member's declared type, before promotion of the field's
        /// value type (bit-fields promote on use). Layout/store relies
        /// on the declared width: a `char` bit-field must extract from
        /// a 1-byte unit, not a promoted 32-bit one.
        decl_ty: Ty,
    },
    Cast {
        ty: Ty,
        expr: Box<CheckedExpr>,
    },
    Generic {
        ctrl: Box<CheckedExpr>,
        /// Index of the selected association in `assocs`.
        index: usize,
        assocs: Vec<(Option<Ty>, CheckedExpr)>,
    },
    VaArg {
        ap: Box<CheckedExpr>,
        ty: Ty,
    },
    StmtExpr {
        items: Vec<BlockItem>,
    },
    /// A folded `offsetof`: the byte offset with the struct type kept
    /// for fidelity.
    Offsetof {
        ty: Ty,
        offset: u64,
    },
    CompoundLiteral {
        ty: Ty,
        init: Box<Initializer>,
    },
    Convert {
        expr: Box<CheckedExpr>,
        ty: Ty,
    },
    SizeofExpr(Box<CheckedExpr>),
    SizeofTy(Ty),
    AlignofExpr(Box<CheckedExpr>),
    AlignofTy(Ty),
    /// GNU label address `&&label`: typed `void *`; the label it names
    /// is validated with the function's `goto`s and dispatched on at
    /// lowering.
    LabelAddr(SymbolU32),
}

// ---- small type predicates (free functions; no Sema state needed) ----

/// Integer conversion rank (C17 §6.3.1.1p1), or `None` for non-integers.
pub(crate) fn int_rank(ty: &Ty) -> Option<u8> {
    Some(match ty.unqualified() {
        Ty::Bool => 0,
        Ty::Char | Ty::SignedChar | Ty::UnsignedChar => 1,
        Ty::Short | Ty::UnsignedShort => 2,
        Ty::Int | Ty::UnsignedInt | Ty::Enum(_) => 3,
        Ty::Long | Ty::UnsignedLong => 4,
        Ty::LongLong | Ty::UnsignedLongLong => 5,
        Ty::Int128 | Ty::UnsignedInt128 => 6,
        _ => return None,
    })
}

/// Whether an integer type is signed. Plain `char` follows the target
/// (`char_signed`; implementation-defined per C17 §6.2.5p15): callers
/// pass `sema.target.char_signed` where the `Sema` counts.
pub(crate) fn is_signed_int_in(ty: &Ty, char_signed: bool) -> Option<bool> {
    Some(match ty.unqualified() {
        Ty::Bool | Ty::UnsignedChar | Ty::UnsignedShort | Ty::UnsignedInt | Ty::UnsignedLong | Ty::UnsignedLongLong | Ty::UnsignedInt128 => false,
        Ty::Short | Ty::Int | Ty::Long | Ty::LongLong | Ty::Int128 => true,
        Ty::SignedChar => true,
        Ty::Char => char_signed,
        Ty::Enum(_) => true,
        _ => return None,
    })
}

/// The bit-field precision of an expression, if any: gcc performs an
/// operation on bit-fields in the precision of the widest operand, so
/// `unsigned long long b : 40` keeps 40 bits through `b << 32` and
/// `b.u33 + b.u33` wraps at 33. Only fields wider than 32 bits carry
/// the precision — a narrower field promotes to `int`, which drops it
/// — and a binary operation with a non-bit-field operand lands in the
/// common type's full precision. Shifts take the left operand's.
pub fn bitfield_precision(expr: &CheckedExpr) -> Option<u32> {
    use crate::parse::{BinaryOp, UnaryOp};
    match &expr.kind {
        CheckedExprKind::Member { bitfield: Some((_, width)), .. } if *width > 32 => Some(*width),
        CheckedExprKind::Unary(op, operand) => match op {
            UnaryOp::Neg | UnaryOp::BitNot | UnaryOp::PreInc | UnaryOp::PreDec | UnaryOp::PostInc | UnaryOp::PostDec => bitfield_precision(operand),
            _ => None,
        },
        CheckedExprKind::Binary(op, lhs, rhs) => match op {
            BinaryOp::Shl | BinaryOp::Shr => bitfield_precision(lhs),
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem | BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor => {
                match (bitfield_precision(lhs), bitfield_precision(rhs)) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    _ => None,
                }
            }
            _ => None,
        },
        CheckedExprKind::Assign(_, lhs, _) => bitfield_precision(lhs),
        CheckedExprKind::CompoundAssign { lhs, .. } => bitfield_precision(lhs),
        _ => None,
    }
}

/// Byte size of an integer type under `target`. Only `long` varies
/// between the supported ABIs (LP64 vs Windows LLP64); the rest of the
/// table is shared. Plain `char` signedness is the other per-target
/// difference here, handled by `is_signed_int_in` instead.
pub(crate) fn int_type_size(ty: &Ty, target: &crate::target::Target) -> Option<u32> {
    let size: u64 = match ty.unqualified() {
        Ty::Bool => target.bool_size,
        Ty::Char | Ty::SignedChar | Ty::UnsignedChar => target.char_size,
        Ty::Short | Ty::UnsignedShort => target.short_size,
        Ty::Int | Ty::UnsignedInt => target.int_size,
        Ty::Long | Ty::UnsignedLong => target.long_size,
        Ty::LongLong | Ty::UnsignedLongLong => target.long_long_size,
        Ty::Int128 | Ty::UnsignedInt128 => target.int128_size,
        Ty::Enum(_) => target.enum_size,
        _ => return None,
    };
    Some(size as u32)
}

pub fn is_integer(ty: &Ty) -> bool {
    int_rank(ty).is_some()
}

fn is_real_float(ty: &Ty) -> bool {
    matches!(ty.unqualified(), Ty::Float | Ty::Double | Ty::LongDouble | Ty::Float128)
}

pub fn is_complex(ty: &Ty) -> bool {
    matches!(ty.unqualified(), Ty::Complex(_))
}

/// The corresponding real type: complex maps to its component, anything
/// else maps to itself.
pub(crate) fn corresponding_real(ty: &Ty) -> Ty {
    match ty.unqualified() {
        Ty::Complex(inner) => inner.as_ref().clone(),
        other => other.clone(),
    }
}

fn complex_of(real: Ty) -> Ty {
    Ty::Complex(Box::new(real))
}

fn is_arithmetic(ty: &Ty) -> bool {
    is_integer(ty) || is_real_float(ty) || is_complex(ty)
}

fn is_scalar(ty: &Ty) -> bool {
    is_arithmetic(ty) || matches!(ty, Ty::Pointer { .. })
}

fn is_void_ptr(ty: &Ty) -> bool {
    matches!(ty.unqualified(), Ty::Pointer { pointee, .. } if pointee.unqualified() == &Ty::Void)
}

/// Qualifiers on what a pointer points to (`const` in `const int *`);
/// empty for unqualified pointees. Only the outermost pointee level
/// matters for the `void *` composite below.
fn pointer_pointee_quals(ty: &Ty) -> Vec<TypeQualifier> {
    let mut pointee = ty;
    while let Ty::Qualified { inner, .. } = pointee {
        pointee = inner;
    }
    if let Ty::Pointer { pointee, .. } = pointee {
        match &**pointee {
            Ty::Qualified { quals, .. } => quals.clone(),
            _ => Vec::new(),
        }
    } else {
        Vec::new()
    }
}

/// Composite of a `void *`-involved ternary (C17 §6.5.15): `void *`
/// with the union of both pointees' qualifiers, so
/// `cond ? void_ptr : const_int_ptr` is `const void *` like GCC.
/// Outer pointer qualifiers drop (rvalues), matching GCC.
fn void_ptr_composite(a: &Ty, b: &Ty) -> Ty {
    let mut quals = pointer_pointee_quals(a);
    for q in pointer_pointee_quals(b) {
        if !quals.contains(&q) {
            quals.push(q);
        }
    }
    quals.sort_by_key(|q| *q as u8);
    Ty::Pointer {
        qualifiers: Vec::new(),
        pointee: Box::new(Ty::qualified(quals, Ty::Void)),
    }
}

/// The unsigned counterpart of a signed integer type.
fn corresponding_unsigned(ty: &Ty) -> Ty {
    match ty.unqualified() {
        Ty::Char | Ty::SignedChar => Ty::UnsignedChar,
        Ty::Short => Ty::UnsignedShort,
        Ty::Int => Ty::UnsignedInt,
        Ty::Long => Ty::UnsignedLong,
        Ty::LongLong => Ty::UnsignedLongLong,
        Ty::Int128 => Ty::UnsignedInt128,
        other => panic!("not a signed integer type: {other:?}"),
    }
}

/// Resolving a field through anonymous members: found with its member
/// path, missing, or present in several anonymous members at once.
enum FieldHit {
    Found(Vec<usize>),
    Miss,
    Ambiguous,
}

/// Whether a checked expression may appear left of `=` or under `&`:
/// variables (never functions, typedefs, or enumeration constants),
/// dereferences, subscripts, and members of lvalues (conversions look
/// through to their operand).
pub fn is_lvalue(expr: &CheckedExpr, decls: &[Binding]) -> bool {
    match &expr.kind {
        CheckedExprKind::Ident(id) => decls
            .get(id.0 as usize)
            .is_some_and(|binding| !matches!(binding.kind, BindingKind::Func | BindingKind::EnumConst { .. })),
        CheckedExprKind::Unary(crate::parse::UnaryOp::Deref, _) => true,
        // A complex part of an lvalue is itself an lvalue
        // (`__real__ x = 1.0`), like a struct member of one.
        CheckedExprKind::Unary(crate::parse::UnaryOp::Real, inner) | CheckedExprKind::Unary(crate::parse::UnaryOp::Imag, inner) => is_lvalue(inner, decls),
        CheckedExprKind::Subscript { .. } => true,
        // Compound literals are lvalues (C17 §6.5.2.5p5).
        CheckedExprKind::CompoundLiteral { .. } => true,
        CheckedExprKind::Member { base, arrow, .. } => *arrow || is_lvalue(base, decls),
        CheckedExprKind::Convert { expr, .. } => is_lvalue(expr, decls),
        _ => false,
    }
}

/// Whether `expr` designates a bit-field: a `Member` whose final field
/// carries a width. Conversions wrap the value without changing what it
/// designates, so they look through (like [`is_lvalue`]).
pub fn designates_bitfield(expr: &CheckedExpr) -> bool {
    match &expr.kind {
        CheckedExprKind::Member { bitfield, .. } => bitfield.is_some(),
        CheckedExprKind::Convert { expr, .. } => designates_bitfield(expr),
        _ => false,
    }
}

/// What a constraint letter selects: how operand lowering treats it
/// (an address, a value, a pinned register) or why it cannot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LetterClass {
    /// A general-purpose register; `named` pins one (`"a"` is `rax`).
    Register(Option<&'static str>),
    /// A register with a high-byte form (x86-64 `"Q"`).
    HighByte,
    /// A memory operand: the operand's address goes to the template in
    /// a scratch register.
    Memory,
    /// An immediate integer constant, substituted textually.
    Immediate,
    /// A letter this machine has but this frontend cannot honor; the
    /// string names what it selects.
    Unsupported(&'static str),
}

/// The letters of a constraint body, as tokens: single characters,
/// except that an AArch64 `U`-prefixed form takes its lowercase suffix
/// with it (`Ump`, `Ush`, ...).
pub fn constraint_letters(body: &str) -> Vec<&str> {
    let mut letters = Vec::new();
    let mut rest = body;
    while let Some(first) = rest.chars().next() {
        let len = if first == 'U' {
            // `U` plus its lowercase suffix (`Ump`, `Ush`, ...).
            1 + rest[1..].find(|c: char| !c.is_ascii_lowercase()).unwrap_or(rest.len() - 1)
        } else {
            first.len_utf8()
        };
        letters.push(&rest[..len]);
        rest = &rest[len..];
    }
    letters
}

/// The class of one constraint letter on `arch`, or `None` when the
/// machine has no such letter. Machine letters are
/// architecture-specific: `"Q"` is a high-byte register on x86-64, a
/// base-register memory operand on AArch64, and no letter on RISC-V.
pub fn constraint_letter(arch: Arch, letter: &str) -> Option<LetterClass> {
    use Arch::*;
    let class = match (arch, letter) {
        // Registers.
        (_, "r" | "q" | "g" | "X") => LetterClass::Register(None),
        (X86_64, "Q") => LetterClass::HighByte,
        (X86_64, "a" | "A") => LetterClass::Register(Some("rax")),
        (X86_64, "b") => LetterClass::Register(Some("rbx")),
        (X86_64, "c") => LetterClass::Register(Some("rcx")),
        (X86_64, "d") => LetterClass::Register(Some("rdx")),
        (X86_64, "S") => LetterClass::Register(Some("rsi")),
        (X86_64, "D") => LetterClass::Register(Some("rdi")),
        // AArch64 `Uci`/`Ucj` (registers 8-11 and 12-15, for SVE ZA
        // indexing): any register does for plain uses.
        (Aarch64, "Uci" | "Ucj") => LetterClass::Register(None),
        // Memory.
        (_, "m" | "o") => LetterClass::Memory,
        (Aarch64, "Q") => LetterClass::Memory,
        (Aarch64, _) if letter.starts_with('U') => LetterClass::Memory,
        (Riscv64, "A") => LetterClass::Memory,
        // Immediates.
        (_, "i" | "n" | "s" | "I" | "J" | "K" | "L" | "M" | "N") => LetterClass::Immediate,
        (X86_64 | Aarch64, "Z") => LetterClass::Immediate,
        (X86_64, "e") => LetterClass::Immediate,
        // Letters these targets have that this frontend cannot honor.
        (Aarch64 | Riscv64, "S") => LetterClass::Unsupported("a symbolic address"),
        (Aarch64, "k") => LetterClass::Unsupported("the stack pointer"),
        (Aarch64, "w" | "x" | "y") => LetterClass::Unsupported("a floating-point or SIMD register"),
        (Aarch64, "Upl" | "Upa") => LetterClass::Unsupported("an SVE predicate register"),
        (_, "f") => LetterClass::Unsupported("a floating-point register"),
        _ => return None,
    };
    Some(class)
}

/// Whether an asm constraint's only alternative is memory: it names a
/// memory class and no register class, so the operand is an addressable
/// object's address, not a value. (A `"rm"` operand may take the
/// register alternative instead, and a matching constraint like `"0"`
/// inherits its target's kind.)
pub fn memory_only_constraint(constraint: &str, arch: Arch) -> bool {
    let body = constraint.trim_start_matches(['=', '+', '&']);
    let is_tie = !body.is_empty() && body.chars().all(|c| c.is_ascii_digit());
    let mut memory = false;
    let mut register = false;
    for letter in constraint_letters(body) {
        match constraint_letter(arch, letter) {
            Some(LetterClass::Memory) => memory = true,
            Some(LetterClass::Register(_) | LetterClass::HighByte) => register = true,
            _ => {}
        }
    }
    !is_tie && memory && !register
}

/// Whether an asm constraint names an immediate class (`i`, `n`, `s`
/// or a range letter). The operand may still have a register or memory
/// alternative (`"Ir"`): lowering then folds a constant expression so
/// the wrapper can substitute it, and the wrapper falls back to the
/// other alternative for anything else.
pub fn has_immediate_constraint(constraint: &str, arch: Arch) -> bool {
    let body = constraint.trim_start_matches(['=', '+', '&']);
    constraint_letters(body)
        .iter()
        .any(|letter| constraint_letter(arch, letter) == Some(LetterClass::Immediate))
}

impl Sema {
    // ---- usual arithmetic conversions (C17 §6.3.1.8) ----

    /// Promote a sub-`int` integer operand to `int` (or to `unsigned int`
    /// when `int` cannot represent every value — unreachable with
    /// the fixed widths of every supported target, kept for
    /// narrower-`int` targets). Non-integers
    /// (floats, pointers, structs) pass through untouched. Enums
    /// promote to their representation: plain `int` for ordinary
    /// enums, wider for sentinel values like `DW_CIE_ID_32`.
    pub fn integer_promote(&self, expr: CheckedExpr) -> CheckedExpr {
        // Complex over a sub-`int` component promotes component-wise
        // (`__complex__ char` computes as `__complex__ int`), mirroring
        // the scalar rule below.
        if let Ty::Complex(inner) = expr.ty.unqualified() {
            if let Some(rank) = int_rank(inner) {
                if rank < int_rank(&Ty::Int).expect("Int has a rank") {
                    let int_size = int_type_size(&Ty::Int, &self.target).expect("Int has a size");
                    let target = if int_type_size(inner, &self.target).expect("rank implies size") < int_size {
                        Ty::Int
                    } else {
                        Ty::UnsignedInt
                    };
                    return self.convert(expr, Ty::Complex(Box::new(target)));
                }
            }
            return expr;
        }
        if let Ty::Enum(id) = expr.ty.unqualified() {
            let repr = self.enums[id.0 as usize].repr.clone();
            if repr == Ty::Int {
                return expr;
            }
            return self.convert(expr, repr);
        }
        let Some(rank) = int_rank(&expr.ty) else {
            return expr;
        };
        if rank >= int_rank(&Ty::Int).expect("Int has a rank") {
            return expr;
        }
        let int_size = int_type_size(&Ty::Int, &self.target).expect("Int has a size");
        let target = if int_type_size(&expr.ty, &self.target).expect("rank implies size") < int_size {
            Ty::Int
        } else {
            Ty::UnsignedInt
        };
        self.convert(expr, target)
    }

    /// Wrap `expr` in a conversion to `ty`, or return it unchanged when
    /// it already has that type.
    pub fn convert(&self, expr: CheckedExpr, ty: Ty) -> CheckedExpr {
        if expr.ty == ty {
            return expr;
        }
        let span = expr.span;
        CheckedExpr {
            ty: ty.clone(),
            span,
            kind: CheckedExprKind::Convert { expr: Box::new(expr), ty },
        }
    }

    /// Default argument promotions for unprototyped/variadic call
    /// arguments: decay plus integer promotions plus `float` → `double`.
    fn default_arg_promote(&self, expr: CheckedExpr) -> CheckedExpr {
        let expr = self.decay_checked(expr);
        let expr = self.integer_promote(expr);
        if expr.ty == Ty::Float { self.convert(expr, Ty::Double) } else { expr }
    }

    /// The common type of two promoted integer operands, following C17
    /// §6.3.1.8 rank/signedness/size rules. Anything non-integer falls
    /// back to the left side (callers only invoke this on arithmetic
    /// operands). Operands are always integer-promoted first (see
    /// `usual_arithmetic_conversions`), so sub-`int` types — the only
    /// ones whose signedness varies by target — never reach here;
    /// `target` is threaded anyway so the rules cannot drift.
    fn uac_integer_common(left: &Ty, right: &Ty, target: &crate::target::Target) -> Ty {
        if left == right {
            return left.clone();
        }
        let (Some(left_rank), Some(right_rank)) = (int_rank(left), int_rank(right)) else {
            return left.clone();
        };
        let left_signed = is_signed_int_in(left, target.char_signed).expect("rank implies signedness");
        let right_signed = is_signed_int_in(right, target.char_signed).expect("rank implies signedness");
        // Same signedness: the greater rank wins.
        if left_signed == right_signed {
            if left_rank >= right_rank {
                return left.clone();
            }
            return right.clone();
        }
        // Unsigned with rank at least the other's: the unsigned type wins.
        if !left_signed && left_rank >= right_rank {
            return left.clone();
        }
        if !right_signed && right_rank >= left_rank {
            return right.clone();
        }
        // Signed that can represent every value of the unsigned side wins.
        let left_size = int_type_size(left, target).expect("rank implies size");
        let right_size = int_type_size(right, target).expect("rank implies size");
        if left_signed && left_size > right_size {
            return left.clone();
        }
        if right_signed && right_size > left_size {
            return right.clone();
        }
        // Otherwise both convert to the unsigned counterpart of the signed type.
        if left_signed { corresponding_unsigned(left) } else { corresponding_unsigned(right) }
    }

    /// The common type of two arithmetic operands: the float/complex
    /// ladder first (so a `float`/`float` pair yields `float` rather
    /// than the left side), then the integer rules. LP64 default for
    /// target-free callers (unit tests); production paths use
    /// [`Sema::uac_common_type_in`] with the compilation target.
    pub fn uac_common_type(left: &Ty, right: &Ty) -> Ty {
        Self::uac_common_type_in(left, right, &crate::target::Target::LP64)
    }

    /// Target-parameterized [`Sema::uac_common_type`].
    pub fn uac_common_type_in(left: &Ty, right: &Ty, target: &crate::target::Target) -> Ty {
        let (left, right) = (left.unqualified(), right.unqualified());
        let complex = is_complex(left) || is_complex(right);
        let l = corresponding_real(left);
        let r = corresponding_real(right);
        let real = if l == Ty::Float128 || r == Ty::Float128 {
            Ty::Float128
        } else if l == Ty::LongDouble || r == Ty::LongDouble {
            Ty::LongDouble
        } else if l == Ty::Double || r == Ty::Double {
            Ty::Double
        } else if l == Ty::Float || r == Ty::Float {
            Ty::Float
        } else {
            let common = Self::uac_integer_common(&l, &r, target);
            return if complex { complex_of(common) } else { common };
        };
        if complex { complex_of(real) } else { real }
    }

    /// Promote both operands, convert each to their common type, and
    /// return the pair plus that type.
    pub fn usual_arithmetic_conversions(&self, left: CheckedExpr, right: CheckedExpr) -> (CheckedExpr, CheckedExpr, Ty) {
        let left = self.integer_promote(left);
        let right = self.integer_promote(right);
        let common = Self::uac_common_type_in(&left.ty, &right.ty, &self.target);
        let left = self.convert(left, common.clone());
        let right = self.convert(right, common.clone());
        (left, right, common)
    }

    // ---- expression checking ----

    /// The function type a designator is called through: identical to
    /// its declared type except that an async function's result is its
    /// `_Future(T)` handle. Every value-level use of the designator
    /// (decay under `&`, calls, assignment to pointers) sees this type,
    /// so calling an async function — directly or through a pointer —
    /// always yields the handle to await.
    fn as_callable_ty(&self, ty: Ty, kind: &CheckedExprKind) -> Ty {
        let CheckedExprKind::Ident(id) = kind else {
            return ty;
        };
        if !self.decls[id.0 as usize].is_async {
            return ty;
        }
        let Ty::Func {
            ret,
            params,
            variadic,
            prototyped,
        } = ty
        else {
            return ty;
        };
        let handle = self.future_handle_of(&ret).expect("async function without a registered future handle");
        Ty::Func {
            ret: Box::new(handle),
            params,
            variadic,
            prototyped,
        }
    }

    /// Array-to-pointer and function-to-pointer decay (C17 §6.3.2.1p3/4),
    /// plus lvalue conversion's qualifier drop: a value's top-level
    /// qualifiers vanish when read (`const int` reads as `int`).
    /// Applied at use sites — never to `sizeof`/`&` operands, which see
    /// the un-decayed type. Qualification on array elements survives on
    /// the pointee (`const int[3]` decays to `const int *`); a
    /// qualifier on the array itself has no decayed form and drops.
    fn decay_checked(&self, expr: CheckedExpr) -> CheckedExpr {
        let CheckedExpr { ty, span, kind } = expr;
        let expr = CheckedExpr {
            ty: ty.unqualified().clone(),
            span,
            kind,
        };
        let ty = match &expr.ty {
            Ty::Array { elem, quals, .. } => Ty::Pointer {
                qualifiers: quals.clone(),
                pointee: elem.clone(),
            },
            Ty::Func { .. } => Ty::Pointer {
                qualifiers: Vec::new(),
                pointee: Box::new(self.as_callable_ty(expr.ty.clone(), &expr.kind)),
            },
            _ => return expr,
        };
        let CheckedExpr { span, kind, .. } = expr;
        // A `_Generic`'s value is its selected arm's value: decay that
        // arm alongside the node so both carry one type.
        let kind = match kind {
            CheckedExprKind::Generic { ctrl, index, assocs } => {
                let mut assocs = assocs;
                if let Some((_, arm)) = assocs.get_mut(index) {
                    *arm = self.decay_checked(arm.clone());
                }
                CheckedExprKind::Generic { ctrl, index, assocs }
            }
            kind => kind,
        };
        CheckedExpr { ty, span, kind }
    }

    /// Whether a checked expression may appear left of `=` or under `&`:
    /// variables (never functions, typedefs, or enumeration constants),
    /// dereferences, subscripts, and members of lvalues (conversions look
    /// through to their operand).
    fn is_lvalue(&self, expr: &CheckedExpr) -> bool {
        is_lvalue(expr, &self.decls)
    }

    /// Whether `expr` designates a bit-field: a `Member` whose final
    /// field carries a width. Conversions wrap the value without
    /// changing what it designates, so they look through (like
    /// `is_lvalue`).
    fn designates_bitfield(expr: &CheckedExpr) -> bool {
        designates_bitfield(expr)
    }

    /// Record the declaration whose address `&expr` takes: the `Ident`
    /// at the base of the operand's lvalue (`&x` marks `x`, `&s.f`
    /// and `&a[i]` mark `s`/`a`). `&*p` marks nothing (it is just
    /// `p`), and compound literals have no declaration to mark.
    /// Callers check validity first; non-lvalue shapes are ignored.
    fn note_address_taken(&self, expr: &CheckedExpr) {
        match &expr.kind {
            CheckedExprKind::Ident(id) => self.mark_address_taken(*id),
            CheckedExprKind::Unary(crate::parse::UnaryOp::Real, inner) | CheckedExprKind::Unary(crate::parse::UnaryOp::Imag, inner) => self.note_address_taken(inner),
            CheckedExprKind::Member { base, .. } => self.note_address_taken(base),
            CheckedExprKind::Subscript { base, .. } => self.note_address_taken(base),
            CheckedExprKind::Convert { expr, .. } => self.note_address_taken(expr),
            _ => {}
        }
    }

    /// A null pointer constant is any integer constant expression with
    /// value zero (C17 §6.3.2.3p1): `0`, `0L`, `(0)`, `'\0'`, and folded
    /// forms like `1 - 1` all qualify — as does any of those cast to
    /// unqualified `void *` (`(void *)0`). A cast to a qualified
    /// `void *` does not count, matching GCC.
    fn is_null_pointer_constant(&self, expr: &CheckedExpr, interner: &Interner) -> bool {
        if matches!(&expr.ty, Ty::Pointer { pointee, .. } if pointee.as_ref() == &Ty::Void) {
            if let CheckedExprKind::Cast { expr, .. } | CheckedExprKind::Convert { expr, .. } = &expr.kind {
                return self.is_null_pointer_constant(expr, interner);
            }
        }
        matches!(crate::eval::eval_const(expr, self, interner), Some(crate::eval::Value::Int(0)))
    }

    /// Convert `expr` as if by assignment to `target`: arithmetic to
    /// arithmetic, pointers to compatible (or `void *`) pointers, and
    /// null constants to pointers. Arrays and functions decay first
    /// (C17 §6.3.2.1), so string literals, array values, and function
    /// designators convert to their pointer types. One level of added
    /// pointer qualifiers is allowed (`char **` converts to
    /// `char *const *`, as `execvp` needs); anything else fails.
    fn convert_for_assignment(&self, target: &Ty, expr: CheckedExpr, interner: &Interner) -> Result<CheckedExpr, CompileError> {
        let expr = self.decay_checked(expr);
        // Top-level qualifiers on the source rvalue are meaningless
        // (the value is copied): a `const char *const` member of a
        // const object converts to `const char *`, like GCC. (The
        // target keeps its qualifiers for the result type; dropping a
        // qualifier *below* the top level still fails below.)
        let CheckedExpr { ty, span, kind } = expr;
        let expr = CheckedExpr {
            ty: ty.unqualified().clone(),
            span,
            kind,
        };
        // The same holds for a source pointer's own qualifier set
        // (`char *const` converts like `char *`); only qualifiers at
        // or below the pointee participate in compatibility.
        let expr = match expr.ty {
            Ty::Pointer { pointee, .. } => {
                let (span, kind) = (expr.span, expr.kind);
                CheckedExpr {
                    ty: Ty::Pointer { qualifiers: Vec::new(), pointee },
                    span,
                    kind,
                }
            }
            _ => expr,
        };
        if expr.ty == *target {
            return Ok(expr);
        }
        // Same struct ignoring top-level qualifiers: the value is
        // copied, so adding `const` to the target (or copying from
        // a qualified source, already stripped above) is fine. Kept
        // as a retype rather than a `Convert` node, which has no
        // lowering for aggregates. Pointers keep their own
        // qualifier-addition rules below.
        if matches!(expr.ty.unqualified(), Ty::Struct(_)) && expr.ty.unqualified() == target.unqualified() {
            let CheckedExpr { span, kind, .. } = expr;
            return Ok(CheckedExpr { ty: target.clone(), span, kind });
        }
        let span = expr.span;
        if is_arithmetic(&expr.ty) && is_arithmetic(target) {
            return Ok(self.convert(expr, target.clone()));
        }
        if let (Ty::Pointer { .. }, Ty::Pointer { .. }) = (target, &expr.ty) {
            if self.ptr_assign_compatible(target, &expr.ty, span) || is_void_ptr(target) || is_void_ptr(&expr.ty) {
                return Ok(self.convert(expr, target.clone()));
            }
            // Mismatched pointees (`float *` from `int *`), dropped
            // qualifiers, and object/function pointer mixes are only
            // diagnosed by GCC and clang (`-Wincompatible-pointer-types`,
            // `-Wdiscarded-qualifiers`), and `-w` silences them: every
            // pointer is one machine word, so the bits move untouched.
            self.warnings.borrow_mut().push(CompileError::warning("incompatible pointer types in assignment", span));
            return Ok(self.convert(expr, target.clone()));
        }
        // Windows `va_list` is a plain `char *` (MSVC's own headers
        // define it so, and the backend models it the same way): the
        // builtin spelling and pointers interconvert freely there.
        // Retyped rather than converted — like the struct case above,
        // so lvalue uses (`va_start(ap, x)`) keep their place.
        if self.target.os == Os::Windows && matches!((target, &expr.ty), (Ty::BuiltinVaList, Ty::Pointer { .. }) | (Ty::Pointer { .. }, Ty::BuiltinVaList)) {
            let CheckedExpr { span, kind, .. } = expr;
            return Ok(CheckedExpr { ty: target.clone(), span, kind });
        }
        if matches!(target, Ty::Pointer { .. }) && self.is_null_pointer_constant(&expr, interner) {
            return Ok(self.convert(expr, target.clone()));
        }
        // GCC accepts integer/pointer mixes with a warning
        // (`-Wint-conversion`, disabled in the torture suite): a
        // runtime address is just its bits, which the backend moves
        // untouched (integers and pointers share one representation).
        if matches!(target, Ty::Pointer { .. }) && is_integer(&expr.ty) {
            self.warnings
                .borrow_mut()
                .push(CompileError::new("assignment makes pointer from integer without a cast", span));
            return Ok(self.convert(expr, target.clone()));
        }
        if is_integer(target) && matches!(&expr.ty, Ty::Pointer { .. }) {
            self.warnings
                .borrow_mut()
                .push(CompileError::new("assignment makes integer from pointer without a cast", span));
            return Ok(self.convert(expr, target.clone()));
        }
        Err(CompileError::new("incompatible types in assignment", span))
    }

    /// Whether `source` converts to `target` by assignment (C17
    /// §6.5.16.1p1): pointers whose pointed-to types are compatible,
    /// where the target may add qualifiers. Adding `const` (or
    /// `volatile`) to the immediate pointee is silent (`char *`
    /// converts to `const char *`, as `execvp` needs); adding
    /// qualifiers deeper (`char **` to `const char **`) is a GCC
    /// warning, recorded for the parser to render. Dropping a
    /// qualifier or mismatched pointees stay incompatible. (Base
    /// qualifiers like `const char` live in `Ty::Qualified`, so they
    /// reach this check; only pointer-owned qualifiers like `*const`
    /// compare as sets.)
    fn ptr_assign_compatible(&self, target: &Ty, source: &Ty, span: Span) -> bool {
        match self.ptr_compat_kind(target, source) {
            PointeeConversion::No => false,
            PointeeConversion::Yes => true,
            PointeeConversion::Warns => {
                self.warnings.borrow_mut().push(CompileError::warning("assignment from incompatible pointer type", span));
                true
            }
        }
    }

    /// Pointee compatibility without diagnosing: the pure query behind
    /// `ptr_assign_compatible` (which adds the warning). Ternary
    /// composition needs both directions silently to prefer an exact
    /// (`Yes`) fit over a warning one.
    fn ptr_compat_kind(&self, target: &Ty, source: &Ty) -> PointeeConversion {
        let (Ty::Pointer { qualifiers: qt, pointee: pt }, Ty::Pointer { qualifiers: qs, pointee: ps }) = (target, source) else {
            return PointeeConversion::No;
        };
        if !qs.iter().all(|q| qt.contains(q)) {
            return PointeeConversion::No;
        }
        self.pointee_conversion(pt, ps, 0)
    }

    /// Whether an enum and its representation integer type meet as
    /// pointees. C17 §6.7.2.2p4 makes `enum e` compatible with the
    /// integer type that holds its values, so `int *` and `enum e *`
    /// (for a `int`-represented enum) mix silently, like GCC. Two
    /// enums stay distinct types even with equal representations.
    fn enum_repr_pair(&self, t: &Ty, s: &Ty) -> bool {
        match (t.unqualified(), s.unqualified()) {
            (Ty::Enum(id), other) | (other, Ty::Enum(id)) => !matches!(other, Ty::Enum(_)) && *other == self.enums[id.0 as usize].repr,
            _ => false,
        }
    }

    /// How a source pointee converts to a target pointee: qualifiers
    /// may only be added, and silently only at the outermost level.
    fn pointee_conversion(&self, target: &Ty, source: &Ty, depth: usize) -> PointeeConversion {
        match (target, source) {
            (Ty::Pointer { qualifiers: qt, pointee: pt }, Ty::Pointer { qualifiers: qs, pointee: ps }) => {
                if !qs.iter().all(|q| qt.contains(q)) {
                    return PointeeConversion::No;
                }
                self.pointee_conversion(pt, ps, depth + 1)
            }
            // Arrays convert element-wise when their bounds agree (an
            // absent bound combines with anything, like `types_compatible`):
            // `const char (*)[4]` from `char (*)[4]` is silent, matching
            // the `const char *` from `char *` rule, so element qualifiers
            // added here count as the outermost level.
            (Ty::Array { elem: et, quals: qt, bound: bt }, Ty::Array { elem: es, quals: qs, bound: bs }) => {
                if !qs.iter().all(|q| qt.contains(q)) {
                    return PointeeConversion::No;
                }
                if !Self::array_bounds_assign_compatible(bt, bs) {
                    return PointeeConversion::No;
                }
                self.pointee_conversion(et, es, depth)
            }
            (Ty::Qualified { quals: qt, inner: it }, Ty::Qualified { quals: qs, inner: is }) => {
                if !qs.iter().all(|q| qt.contains(q)) {
                    return PointeeConversion::No;
                }
                self.pointee_conversion(it, is, depth + 1)
            }
            // The target adds qualifiers: silent at the outermost
            // level (`const char *` from `char *`), a warning deeper
            // (`const char **` from `char **`, like GCC).
            (Ty::Qualified { inner, .. }, s) => match self.pointee_conversion(inner, s, depth + 1) {
                PointeeConversion::No => PointeeConversion::No,
                _ if depth == 0 => PointeeConversion::Yes,
                _ => PointeeConversion::Warns,
            },
            // The source is qualified where the target is not: dropping
            // qualifiers (`const char *` to `char *`) is a GCC warning,
            // accepted like the suite (built with `-w`) expects.
            (_, Ty::Qualified { .. }) => PointeeConversion::Warns,
            (t, s) => {
                if types_compatible(t, s) || self.enum_repr_pair(t, s) {
                    PointeeConversion::Yes
                } else if Self::signedness_only_pair(t, s, &self.target) {
                    // Pointee types differing only in signedness
                    // (`long *` from `unsigned long *`): a GCC warning,
                    // bit-identical underneath.
                    PointeeConversion::Warns
                } else {
                    PointeeConversion::No
                }
            }
        }
    }

    /// Whether two array bounds agree for pointer assignment (mirrors
    /// `types_compatible`'s bound rule): an absent bound combines with
    /// anything, otherwise both extents must match — except a VLA,
    /// which matches any bound (C17 §6.7.6.2p6; see
    /// `sema::bounds_compatible`).
    fn array_bounds_assign_compatible(a: &Option<ArrayBound>, b: &Option<ArrayBound>) -> bool {
        match (a, b) {
            (None, _) | (_, None) => true,
            (Some(ArrayBound::Star), _) | (_, Some(ArrayBound::Star)) => true,
            (Some(ArrayBound::Vla(_)), _) | (_, Some(ArrayBound::Vla(_))) => true,
            (Some(ArrayBound::Fixed(x)), Some(ArrayBound::Fixed(y))) => x == y,
        }
    }

    /// Whether two integer types differ only in signedness (`int`
    /// vs `unsigned`, `char` vs `unsigned char`): same size, same
    /// rank, opposite signedness. `_Bool` and enums are not sign pairs
    /// (kept strict). The rank guard keeps same-sized-but-unrelated
    /// pairs like `long`/`unsigned int` under LLP64 out. Plain `char`
    /// follows `char_signed` (unsigned on AArch64/RISC-V, so there
    /// `char` pairs with `unsigned char`, not `signed char`).
    fn signedness_only_pair(t: &Ty, s: &Ty, target: &crate::target::Target) -> bool {
        if matches!(t.unqualified(), Ty::Bool | Ty::Enum(_)) || matches!(s.unqualified(), Ty::Bool | Ty::Enum(_)) {
            return false;
        }
        let (Some(t_sign), Some(s_sign)) = (is_signed_int_in(t, target.char_signed), is_signed_int_in(s, target.char_signed)) else {
            return false;
        };
        t_sign != s_sign && int_rank(t) == int_rank(s) && int_type_size(t, target) == int_type_size(s, target)
    }

    /// The diagnostic for writing a `const`-qualified object: names
    /// the variable when the target is one (like GCC's
    /// "assignment of read-only variable 'x'"), else the location.
    fn readonly_error(&self, expr: &CheckedExpr, interner: &Interner) -> CompileError {
        match &expr.kind {
            CheckedExprKind::Ident(id) => {
                let text = self.decls.get(id.0 as usize).and_then(|b| b.name).and_then(|n| interner.resolve(n)).unwrap_or("?");
                CompileError::new(format!("assignment of read-only variable '{text}'"), expr.span)
            }
            _ => CompileError::new("assignment of read-only location", expr.span),
        }
    }

    /// Check a condition expression: valid and scalar (C17 §6.8.4.1,
    /// §6.8.5.2). Callers discard the tree; the diagnostics are the
    /// point.
    pub fn check_condition(&self, expr: &Expr, interner: &Interner) -> Result<CheckedExpr, CompileError> {
        let checked = self.decay_checked(self.check_expr(expr, interner)?);
        if !is_scalar(&checked.ty) {
            return Err(CompileError::new("condition must have scalar type", checked.span));
        }
        Ok(checked)
    }

    /// Check an expression that must have integer type: `switch`
    /// controlling expressions and `case` labels (C17 §6.8.1p1,
    /// §6.8.4.2). `what` names the position for the message.
    pub fn check_integer_expr(&self, expr: &Expr, interner: &Interner, what: &str) -> Result<CheckedExpr, CompileError> {
        let checked = self.check_expr(expr, interner)?;
        if !is_integer(&checked.ty) {
            return Err(CompileError::new(format!("{what} must have integer type"), checked.span));
        }
        Ok(checked)
    }

    /// Check an initializer against its object's type, returning the
    /// checked form: scalar expressions convert as if by assignment;
    /// brace lists check member-by-member in declaration order (offsets
    /// need no layout — only member types and positions).
    pub fn check_initializer(&self, target: &Ty, init: &Initializer, interner: &Interner) -> Result<Initializer, CompileError> {
        match init {
            // Values arrive already checked (parsers check inline, where
            // scopes are live); here they only convert to the target.
            Initializer::Expr(expr) => {
                // `char s[] = "abc";` — a string literal initializes an
                // array of its own character type directly (C17
                // §6.7.9p14), with or without braces (the braced form
                // lands in `check_array_init` instead).
                if let Ty::Array { elem, bound, .. } = target.unqualified() {
                    if matches!(expr.ty.unqualified(), Ty::Array { .. }) {
                        if let CheckedExprKind::Str(parts) = &expr.kind {
                            let lit_elem = crate::eval::string_elem_ty_in(parts, self.target.wchar());
                            if lit_elem.is_some_and(|lit| Self::string_elem_matches(elem, &lit)) {
                                // Same overlong-literal warning the
                                // braced path gets; only the fit check
                                // differs.
                                self.check_string_fit(parts, bound, expr.span(), interner);
                                return Ok(Initializer::Expr(expr.clone()));
                            }
                        }
                    }
                }
                // A scalar against a struct/union (`.m = 0` for a mutex,
                // `= 5` for a one-member struct) initializes the first
                // member recursively — the elided-braces rule applied
                // to a designated target. Strings are excluded (a
                // struct never initializes from one), as are
                // struct-valued expressions (whole-struct assignment
                // converts directly).
                if matches!(target.unqualified(), Ty::Struct(_)) && is_scalar(&expr.ty) && !matches!(&expr.kind, CheckedExprKind::Str(_)) {
                    let single = vec![InitItem {
                        designators: Vec::new(),
                        init: Box::new(Initializer::Expr(expr.clone())),
                        span: expr.span(),
                    }];
                    let span = expr.span();
                    return self.check_initializer(target, &Initializer::List { items: single, span }, interner);
                }
                let converted = self.convert_for_assignment(target, expr.clone(), interner)?;
                Ok(Initializer::Expr(converted))
            }
            Initializer::List { items, span } => {
                let items = self.check_init_list(target, items, interner)?;
                // `char s[] = { "abc" };` — the braces are redundant
                // around a string literal initializing the character
                // array itself (C17 §6.7.9p14); collapse to the bare
                // form so sizing and lowering share one shape. The
                // element must match, or the item was already consumed
                // by the element (`char x[2][3] = { "ab" }`).
                if let Ty::Array { elem, .. } = target.unqualified() {
                    if let [item] = items.as_slice() {
                        if item.designators.is_empty() {
                            if let Initializer::Expr(expr) = &*item.init {
                                if let CheckedExprKind::Str(parts) = &expr.kind {
                                    if crate::eval::string_elem_ty_in(parts, self.target.wchar()).is_some_and(|lit| Self::string_elem_matches(elem, &lit)) {
                                        return Ok(Initializer::Expr(expr.clone()));
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(Initializer::List { items, span: *span })
            }
        }
    }

    /// Check a brace-enclosed list against `target`, resolving each item
    /// to the member or element type its designators (or position) select.
    fn check_init_list(&self, target: &Ty, items: &[InitItem], interner: &Interner) -> Result<Vec<InitItem>, CompileError> {
        // Qualification on the object itself never changes the
        // initializer shape (`const struct S s = {...}` initializes
        // like `struct S`).
        match target.unqualified() {
            Ty::Struct(id) => self.check_struct_init(*id, items, interner),
            Ty::Array { .. } => self.check_array_init(target.unqualified(), items, interner),
            _ if is_scalar(target) => {
                // Redundant braces around a scalar are legal (`int x =
                // {{1}}`), but only for a single bare item.
                let [item] = items else {
                    return Err(CompileError::new(
                        "excess elements in scalar initializer",
                        items.get(1).map(|item| item.span).unwrap_or(Span::dummy()),
                    ));
                };
                if !item.designators.is_empty() {
                    return Err(CompileError::new("designator in scalar initializer", item.span));
                }
                let checked = self.check_initializer(target, &item.init, interner)?;
                Ok(vec![InitItem {
                    designators: Vec::new(),
                    init: Box::new(checked),
                    span: item.span,
                }])
            }
            _ => Err(CompileError::new(
                "invalid initializer for type",
                items.first().map(|item| item.span).unwrap_or(Span::dummy()),
            )),
        }
    }

    /// Check a brace list against a struct or union type. Bare items
    /// fill positionally with C brace elision: the cursor is a member
    /// path that descends into anonymous struct members, so `{1, 2}`
    /// for `{ struct { int a, b; }; int c; }` initializes `a` then
    /// `b`; an anonymous union is not transparent (`{1, 2}` for
    /// `{ union { int a, b; }; int c; }` is `a` then `c`, matching
    /// GCC).
    /// Designators (`.field`, possibly chained like `.a.b`, and `[i]`
    /// into member arrays) resolve to a member whose type checks the
    /// item; afterwards the cursor continues after the outermost
    /// designated member. Unions reject a second *bare* item, but
    /// designated items may repeat. Array designator positions stay
    /// loose (checked integer-typed, not tracked).
    fn check_struct_init(&self, id: StructId, items: &[InitItem], interner: &Interner) -> Result<Vec<InitItem>, CompileError> {
        let def = &self.structs[id.0 as usize];
        if !def.complete {
            return Err(CompileError::new(
                "initializer for incomplete type",
                items.first().map(|item| item.span).unwrap_or(Span::dummy()),
            ));
        }
        let is_union = def.is_union;
        let mut cursor: Vec<usize> = Vec::new();
        self.normalize_cursor(id, &mut cursor);
        let mut bare = 0u32;
        let mut checked = Vec::with_capacity(items.len());
        let mut i = 0;
        while i < items.len() {
            let item = &items[i];
            if item.designators.is_empty() {
                if is_union {
                    if bare > 0 {
                        return Err(CompileError::new("excess elements in union initializer", item.span));
                    }
                    bare += 1;
                }
                // A braced list initializes the outermost anonymous
                // aggregate itself (`{{"a", "b"}}` fills the anonymous
                // member's fields), not its first leaf: climb past the
                // transparent descent below. Scalars keep leaf
                // behavior (brace elision into the first member).
                if matches!(&*item.init, Initializer::List { .. }) {
                    while cursor.len() > 1 {
                        cursor.pop();
                    }
                }
                let Some(member) = self.member_at(id, &cursor) else {
                    return Err(CompileError::new("too many initializers for struct", item.span));
                };
                let member_ty = member.ty.clone();
                // Brace elision (C17 §6.7.9p20): a bare scalar where a
                // subaggregate sits consumes this and following bare
                // scalars into that member (`{'X', 8, 9}` for `{char;
                // int[2];}`). The gathered run is checked as one
                // braced list, exactly like explicit braces.
                if self.needs_elision(&member_ty, &item.init) {
                    let j = self.elision_span(&member_ty, items, i);
                    if j > i {
                        let sub = items[i..j].to_vec();
                        let span = sub.first().map(|it| it.span).unwrap_or(item.span).merge(sub.last().map(|it| it.span).unwrap_or(item.span));
                        let init = self.check_initializer(&member_ty, &Initializer::List { items: sub, span }, interner)?;
                        self.advance_cursor(id, &mut cursor);
                        checked.push(InitItem {
                            designators: Vec::new(),
                            init: Box::new(init),
                            span: item.span,
                        });
                        i = j;
                        continue;
                    }
                }
                let init = self.check_initializer(&member_ty, &item.init, interner)?;
                self.advance_cursor(id, &mut cursor);
                checked.push(InitItem {
                    designators: Vec::new(),
                    init: Box::new(init),
                    span: item.span,
                });
                i += 1;
            } else {
                let (target_ty, outer, expanded) = self.resolve_designation(id, &item.designators, interner)?;
                let init = self.check_initializer(&target_ty, &item.init, interner)?;
                cursor = vec![outer + 1];
                self.normalize_cursor(id, &mut cursor);
                checked.push(InitItem {
                    designators: expanded,
                    init: Box::new(init),
                    span: item.span,
                });
                i += 1;
            }
        }
        Ok(checked)
    }

    /// Whether a bare item elides into an aggregate member/element: a
    /// scalar expression where an array or struct sits. Nested lists
    /// recurse normally, and a string literal initializes a character
    /// array directly (no elision).
    fn needs_elision(&self, member_ty: &Ty, init: &Initializer) -> bool {
        let Initializer::Expr(expr) = init else {
            return false;
        };
        match member_ty.unqualified() {
            Ty::Array { elem, .. } => {
                if let CheckedExprKind::Str(parts) = &expr.kind {
                    if let Some(lit_elem) = crate::eval::string_elem_ty_in(parts, self.target.wchar()) {
                        return !Self::string_elem_matches(elem, &lit_elem);
                    }
                }
                true
            }
            // A struct-valued expression of the member's own type
            // initializes the whole member (C17 §6.7.9p13), not the
            // first scalar of an elided run; anything else elides.
            Ty::Struct(_) => expr.ty.unqualified() != member_ty.unqualified(),
            _ => false,
        }
    }

    /// Scalar slots inside a type: what one elided run may fill. `None`
    /// is unbounded (`[]`, or containing one): the run takes every
    /// following bare item.
    /// How many items `items[i..]` fill of `ty` under brace elision:
    /// each scalar takes one bare item, a string literal fills a whole
    /// character array, and aggregates recurse member by member — a
    /// union takes only its first member's share. Stops at designators
    /// and braced items. The result is where the next member's items
    /// begin.
    fn elision_span(&self, ty: &Ty, items: &[InitItem], i: usize) -> usize {
        let mut j = i;
        match ty.unqualified() {
            Ty::Array { elem, bound, .. } => {
                if self.string_item(items, j).is_some_and(|lit| Self::string_elem_matches(elem, &lit)) {
                    return j + 1;
                }
                match bound {
                    Some(ArrayBound::Fixed(n)) => {
                        for _ in 0..*n {
                            let next = self.elision_span(elem, items, j);
                            if next == j {
                                break;
                            }
                            j = next;
                        }
                    }
                    // Unbounded (`[]` or a flexible member): every
                    // following bare item is an element.
                    _ => {
                        while Self::bare_item(items, j) {
                            j += 1;
                        }
                    }
                }
            }
            Ty::Struct(id) => {
                let Some(def) = self.structs.get(id.0 as usize) else {
                    return j;
                };
                let members = if def.is_union { &def.members[..def.members.len().min(1)] } else { &def.members };
                for member in members {
                    let next = self.elision_span(&member.ty, items, j);
                    if next == j {
                        break;
                    }
                    j = next;
                }
            }
            _ => {
                if Self::bare_item(items, j) {
                    j += 1;
                }
            }
        }
        j
    }

    /// Whether `items[j]` is a bare expression item: brace elision
    /// feeds only un-designated expression items to a member.
    fn bare_item(items: &[InitItem], j: usize) -> bool {
        items.get(j).is_some_and(|item| item.designators.is_empty() && matches!(&*item.init, Initializer::Expr(_)))
    }

    /// Element type of the bare string-literal item at `items[j]`, if
    /// any: a string fills a whole character array as one item.
    fn string_item(&self, items: &[InitItem], j: usize) -> Option<Ty> {
        let item = items.get(j)?;
        if !item.designators.is_empty() {
            return None;
        }
        let Initializer::Expr(expr) = &*item.init else {
            return None;
        };
        let CheckedExprKind::Str(parts) = &expr.kind else {
            return None;
        };
        crate::eval::string_elem_ty_in(parts, self.target.wchar())
    }

    /// The member at a cursor path, walking nested structs. `None` when
    /// the path runs past the end (or through a missing definition).
    fn member_at(&self, mut id: StructId, cursor: &[usize]) -> Option<&Member> {
        let mut found = None;
        for &index in cursor {
            let member = self.structs.get(id.0 as usize)?.members.get(index)?;
            if let Ty::Struct(next) = member.ty.unqualified() {
                id = *next;
            }
            found = Some(member);
        }
        found
    }

    /// Descend a cursor into anonymous *struct* members (which bare
    /// items initialize transparently); an anonymous union is not
    /// transparent — its initializer covers the first member only —
    /// so the cursor stops at it. An empty cursor starts at the first
    /// member; out-of-range and incomplete stops stay put for the
    /// caller to diagnose; a local visited set guards pathological
    /// recursive types.
    fn normalize_cursor(&self, id: StructId, cursor: &mut Vec<usize>) {
        if cursor.is_empty() {
            cursor.push(0);
        }
        let mut visiting = Vec::new();
        loop {
            let Some(member) = self.member_at(id, cursor) else {
                break;
            };
            if member.name.is_some() {
                break;
            }
            let Ty::Struct(inner) = member.ty.unqualified() else {
                break;
            };
            if visiting.contains(inner) {
                break;
            }
            if !self.structs.get(inner.0 as usize).is_some_and(|d| d.complete && !d.is_union) {
                break;
            }
            visiting.push(*inner);
            cursor.push(0);
        }
    }

    /// Advance a (valid, normalized) cursor past its member, carrying
    /// past exhausted levels; running past the top level parks at
    /// `[member count]`, which reads as past-the-end.
    fn advance_cursor(&self, root: StructId, cursor: &mut Vec<usize>) {
        if let Some(last) = cursor.last_mut() {
            *last += 1;
        } else {
            cursor.push(0);
            return;
        }
        loop {
            let parent_len = cursor.len() - 1;
            let count = self.level_count(root, &cursor[..parent_len]);
            match count {
                Some(n) if cursor[parent_len] < n => break,
                _ => {
                    cursor.pop();
                    if let Some(parent) = cursor.last_mut() {
                        *parent += 1;
                    } else {
                        cursor.push(self.structs.get(root.0 as usize).map(|d| d.members.len()).unwrap_or(0));
                        return;
                    }
                }
            }
        }
        self.normalize_cursor(root, cursor);
    }

    /// Member count at the struct reached by walking `prefix` from
    /// `root` (cursor paths cross only structs, so anything else is a
    /// broken path).
    fn level_count(&self, root: StructId, prefix: &[usize]) -> Option<usize> {
        let mut id = root;
        for &index in prefix {
            let member = self.structs.get(id.0 as usize)?.members.get(index)?;
            let Ty::Struct(next) = member.ty.unqualified() else {
                return None;
            };
            id = *next;
        }
        Some(self.structs.get(id.0 as usize)?.members.len())
    }

    /// Resolve a struct-initializer item's designators to the
    /// initialized type plus the outermost member index (where a
    /// following bare item continues).
    fn resolve_designation(&self, id: StructId, designators: &[Designator], interner: &Interner) -> Result<(Ty, usize, Vec<Designator>), CompileError> {
        let (target, Some(outer), expanded) = self.walk_designation(Ty::Struct(id), designators, interner)? else {
            unreachable!("a designation always resolves an outermost member")
        };
        Ok((target, outer, expanded))
    }

    /// Walk a designator chain from `start` to the designated type.
    /// `.field` searches the current struct, descending through
    /// anonymous members; `[i]` selects into an array (checked
    /// integer-typed, like array initializers). Also returns the
    /// outermost member index of the first `.field` hop (`None` for
    /// array starts, where no member continues), and the chain with
    /// anonymous-member hops made explicit: the backend addresses
    /// members positionally and never searches by name.
    fn walk_designation(&self, start: Ty, designators: &[Designator], interner: &Interner) -> Result<(Ty, Option<usize>, Vec<Designator>), CompileError> {
        let mut target = start;
        let mut outer: Option<usize> = None;
        let mut expanded: Vec<Designator> = Vec::with_capacity(designators.len());
        let mut visiting = Vec::new();
        for designator in designators {
            match (designator, target.unqualified()) {
                (Designator::Field { name, span }, Ty::Struct(cid)) => {
                    let path = match self.find_field(*cid, *name, &mut visiting) {
                        FieldHit::Found(path) => path,
                        FieldHit::Miss => {
                            let text = interner.resolve(*name).unwrap_or("?");
                            return Err(CompileError::new(format!("no member named '{text}'"), *span));
                        }
                        FieldHit::Ambiguous => {
                            let text = interner.resolve(*name).unwrap_or("?");
                            return Err(CompileError::new(format!("ambiguous member '{text}' (found in several anonymous members)"), *span));
                        }
                    };
                    if outer.is_none() {
                        outer = Some(path[0]);
                    }
                    // Hops through anonymous members become explicit
                    // member indices: the backend addresses members
                    // positionally and never searches by name.
                    for &index in &path[..path.len() - 1] {
                        expanded.push(Designator::AnonField { index: index as u32, span: *span });
                    }
                    expanded.push(Designator::Field { name: *name, span: *span });
                    for &index in &path {
                        let Ty::Struct(cur) = target.unqualified() else {
                            unreachable!("designator path crosses only structs")
                        };
                        target = self.structs[cur.0 as usize].members[index].ty.clone();
                    }
                }
                (Designator::AnonField { index, span }, Ty::Struct(cid)) => {
                    if outer.is_none() {
                        outer = Some(*index as usize);
                    }
                    let Some(member) = self.structs[cid.0 as usize].members.get(*index as usize) else {
                        return Err(CompileError::new("no such member", *span));
                    };
                    target = member.ty.clone();
                    expanded.push(designator.clone());
                }
                (Designator::Field { span, .. } | Designator::AnonField { span, .. }, Ty::Array { .. }) => {
                    return Err(CompileError::new("field designator in array initializer", *span));
                }
                (Designator::Field { name, .. }, _) => {
                    let text = interner.resolve(*name).unwrap_or("?");
                    return Err(CompileError::new(format!("no member named '{text}'"), designator.span()));
                }
                (Designator::AnonField { span, .. }, _) => {
                    return Err(CompileError::new("field designator for non-struct type", *span));
                }
                (Designator::Index { index, span }, Ty::Array { elem, .. }) => {
                    if !is_integer(&index.ty) {
                        return Err(CompileError::new("array designator must have integer type", *span));
                    }
                    target = elem.as_ref().clone();
                    expanded.push(designator.clone());
                }
                (Designator::Index { span, .. }, Ty::Struct(_)) => {
                    return Err(CompileError::new("array designator in struct initializer", *span));
                }
                (Designator::Index { span, .. }, _) => {
                    return Err(CompileError::new("array designator for non-array type", *span));
                }
            }
        }
        Ok((target, outer, expanded))
    }

    /// Whether a type is variably modified (C17 §6.7.6p3): an array
    /// with a `[*]` bound or a bound that is not a non-negative
    /// integer constant, or (recursively) containing one. Function
    /// types stop the search: prototypes may be variably modified
    /// (`void f(int n, int a[n]);` is fine at file scope).
    pub(crate) fn is_variably_modified(&self, ty: &Ty, interner: &Interner) -> bool {
        match ty {
            Ty::Array {
                bound: Some(ArrayBound::Star) | Some(ArrayBound::Vla(_)),
                ..
            } => true,
            Ty::Array { elem, .. } => self.is_variably_modified(elem, interner),
            Ty::Pointer { pointee, .. } => self.is_variably_modified(pointee, interner),
            Ty::Qualified { inner, .. } => self.is_variably_modified(inner, interner),
            Ty::Func { .. } => false,
            _ => false,
        }
    }

    /// Whether a type is itself a variable length array type
    /// (C17 §6.7.6.2p3): an array with a `[*]`/non-constant bound, or
    /// an array whose element type is one. Pointers stop the search:
    /// a pointer to a VLA is variably modified but not a VLA, so
    /// having static storage is fine for it (C17 §6.7.6.2p2).
    pub(crate) fn is_vla_type(&self, ty: &Ty, interner: &Interner) -> bool {
        match ty {
            Ty::Array {
                bound: Some(ArrayBound::Star) | Some(ArrayBound::Vla(_)),
                ..
            } => true,
            Ty::Array { elem, .. } => self.is_vla_type(elem, interner),
            Ty::Qualified { inner, .. } => self.is_vla_type(inner, interner),
            _ => false,
        }
    }

    /// Check a brace list against an array type. Positional items fill in
    /// order; `[i]` designators select positions. Indices fold here: an
    /// index past the bound is a constraint violation, while plain
    /// excess elements are a GCC/clang warning (an error only under
    /// `-pedantic-errors`), recorded for the parser to render. When an
    /// index does not fold, positions are untrackable and excess checks
    /// are skipped. A single string literal initializes a character
    /// array (C17 §6.7.9p14) with fit validated through `string_length`
    /// (an overlong literal warns, like GCC). A variable-length array
    /// cannot be initialized — except with an empty brace list, which
    /// GCC accepts.
    fn check_array_init(&self, target: &Ty, items: &[InitItem], interner: &Interner) -> Result<Vec<InitItem>, CompileError> {
        let Ty::Array { elem, bound, .. } = target else {
            unreachable!("array initializer needs an array type")
        };
        if let [item] = items {
            if item.designators.is_empty() && matches!(&*item.init, Initializer::Expr(expr) if matches!(expr.ty.unqualified(), Ty::Array { .. })) {
                if let Initializer::Expr(expr) = &*item.init {
                    if let CheckedExprKind::Str(parts) = &expr.kind {
                        let lit_elem = crate::eval::string_elem_ty_in(parts, self.target.wchar());
                        if lit_elem.is_some_and(|lit| Self::string_elem_matches(elem, &lit)) {
                            self.check_string_fit(parts, bound, item.span, interner);
                            return Ok(vec![item.clone()]);
                        }
                    }
                }
            }
        }
        if !items.is_empty() && self.is_variably_modified(target, interner) {
            return Err(CompileError::new(
                "variable-sized object may not be initialized",
                items.first().map(|item| item.span).unwrap_or(Span::dummy()),
            ));
        }
        // A known bound limits positions; `[]`, `[*]`, and VLAs impose
        // none.
        let count = match bound {
            Some(ArrayBound::Fixed(n)) => Some(*n),
            _ => None,
        };
        let mut checked = Vec::with_capacity(items.len());
        let mut next: u64 = 0;
        let mut trackable = true;
        let mut i = 0;
        while i < items.len() {
            let item = &items[i];
            let mut pos = next;
            let mut designated = false;
            let mut target = elem.as_ref().clone();
            let mut designators = item.designators.clone();
            if let Some((first, rest)) = item.designators.split_first() {
                let Designator::Index { index, span } = first else {
                    return Err(CompileError::new("field designator in array initializer", first.span()));
                };
                if !is_integer(&index.ty) {
                    return Err(CompileError::new("array designator must have integer type", *span));
                }
                designated = true;
                match crate::eval::eval_const(index, self, interner) {
                    Some(crate::eval::Value::Int(n)) if n >= 0 => {
                        pos = u64::try_from(n).unwrap_or(u64::MAX);
                    }
                    _ => trackable = false,
                }
                // The first index selects into this array; deeper
                // designators (`[i].field`, `[i][j]`) belong to the
                // element and resolve (and check the value) there.
                if !rest.is_empty() {
                    let (ty, _, expanded) = self.walk_designation(target, rest, interner)?;
                    target = ty;
                    designators = Vec::with_capacity(1 + expanded.len());
                    designators.push(first.clone());
                    designators.extend(expanded);
                }
            }
            if trackable {
                if let Some(n) = count {
                    if pos >= n {
                        if designated {
                            return Err(CompileError::new("array index in initializer exceeds array bounds", item.span));
                        }
                        // Plain excess is a warning under both GCC and
                        // clang (an error only with `-pedantic-errors`).
                        // Drop the element rather than carrying it: the
                        // backend lowers items by offset with no bound of
                        // its own, so keeping it writes past the object.
                        self.warnings.borrow_mut().push(CompileError::warning("excess elements in array initializer", item.span));
                        i += 1;
                        continue;
                    }
                }
                next = pos.saturating_add(1);
            }
            // Brace elision into an aggregate element (`struct P a[] =
            // {1, 2, ...}`): gather this and following bare scalars as
            // one element's braced list, consuming a single position.
            if item.designators.is_empty() && self.needs_elision(elem, &item.init) {
                let j = self.elision_span(elem, items, i);
                if j > i {
                    let sub = items[i..j].to_vec();
                    let span = sub.first().map(|it| it.span).unwrap_or(item.span).merge(sub.last().map(|it| it.span).unwrap_or(item.span));
                    let init = self.check_initializer(elem, &Initializer::List { items: sub, span }, interner)?;
                    checked.push(InitItem {
                        designators: Vec::new(),
                        init: Box::new(init),
                        span: item.span,
                    });
                    i = j;
                    continue;
                }
            }
            let init = self.check_initializer(&target, &item.init, interner)?;
            checked.push(InitItem {
                designators,
                init: Box::new(init),
                span: item.span,
            });
            i += 1;
        }
        Ok(checked)
    }

    /// Whether a string literal with element type `lit_elem`
    /// initializes an array of `target_elem` directly: the same
    /// element type, or any two narrow character types
    /// (`unsigned char s[] = "x"`).
    fn string_elem_matches(target_elem: &Ty, lit_elem: &Ty) -> bool {
        let narrow = |t: &Ty| matches!(t.unqualified(), Ty::Char | Ty::SignedChar | Ty::UnsignedChar);
        (narrow(target_elem) && narrow(lit_elem)) || target_elem.unqualified() == lit_elem.unqualified()
    }

    /// Validate a string literal against a character array bound (C17
    /// §6.7.9p14): the bytes must fit, dropping the NUL when exactly
    /// full. An overlong literal is a GCC warning, recorded for the
    /// parser to render; an unknown length or bound skips silently.
    fn check_string_fit(&self, parts: &[crate::lexer::StringLiteral], bound: &Option<ArrayBound>, span: Span, interner: &Interner) {
        let Some(len) = crate::eval::string_length(parts, interner) else {
            return;
        };
        let Some(ArrayBound::Fixed(n)) = bound else {
            return;
        };
        let n = *n;
        if len > n {
            self.warnings
                .borrow_mut()
                .push(CompileError::new("initializer-string for array of characters is too long", span));
        }
    }

    /// Complete a compound literal's array of unknown size from its
    /// initializer (C17 §6.5.2.5p3), like `complete_array_bound` does
    /// for a declaration's `[]`: `(int[]){1, 2}` is `int[3]`, and
    /// `(char[]){"ab"}` is `char[3]` (NUL included). Only the
    /// outermost dimension completes, and only when every designator
    /// folds; anything else keeps the incomplete type for the caller
    /// to reject.
    pub(crate) fn complete_compound_ty(&self, ty: &Ty, init: &Initializer, interner: &Interner) -> Ty {
        let Ty::Array { elem, quals, bound: None } = ty else {
            return ty.clone();
        };
        // `(char[]){"ab"}`: the single string form sizes with its NUL.
        // The braced shape collapses to the bare `Expr` one at check
        // time (see `check_initializer`), so accept either.
        let string = match init {
            Initializer::Expr(expr) => Some(expr),
            Initializer::List { items, .. } => match items.as_slice() {
                [item] if item.designators.is_empty() => match &*item.init {
                    Initializer::Expr(expr) => Some(expr),
                    _ => None,
                },
                _ => None,
            },
        };
        if let Some(expr) = string {
            if let CheckedExprKind::Str(parts) = &expr.kind {
                let lit_elem = crate::eval::string_elem_ty_in(parts, self.target.wchar());
                if lit_elem.is_some_and(|lit| Self::string_elem_matches(elem, &lit)) {
                    match crate::eval::string_length(parts, interner) {
                        Some(len) => {
                            return Ty::Array {
                                elem: elem.clone(),
                                quals: quals.clone(),
                                bound: Some(ArrayBound::Fixed(len.saturating_add(1))),
                            };
                        }
                        None => return ty.clone(),
                    }
                }
            }
        }
        let Initializer::List { items, .. } = init else {
            return ty.clone();
        };
        // Mirror `complete_array_bound`'s tracking: bare items fill in
        // order, `[i]` selects; anything untrackable leaves the type
        // incomplete.
        let mut next: u64 = 0;
        let mut max: u64 = 0;
        for item in items {
            let mut pos = next;
            let mut designated = false;
            for designator in &item.designators {
                match designator {
                    Designator::Index { index, .. } if !designated => {
                        designated = true;
                        match crate::eval::eval_const(index, self, interner) {
                            Some(crate::eval::Value::Int(n)) if n >= 0 => {
                                pos = u64::try_from(n).unwrap_or(u64::MAX);
                            }
                            _ => return ty.clone(),
                        }
                    }
                    Designator::Index { .. } => {}
                    Designator::Field { .. } | Designator::AnonField { .. } => return ty.clone(),
                }
            }
            max = max.max(pos.saturating_add(1));
            next = pos.saturating_add(1);
        }
        Ty::Array {
            elem: elem.clone(),
            quals: quals.clone(),
            bound: Some(ArrayBound::Fixed(max)),
        }
    }

    /// Check a `return` value against the enclosing function's return
    /// type, converting as if by assignment. In a `void` function the
    /// value is legal only when it is itself `void` (`return f();`,
    /// C17 §6.8.6.4p1); anything else is a constraint violation.
    pub fn check_return(&self, ret: &Ty, expr: &Expr, interner: &Interner) -> Result<CheckedExpr, CompileError> {
        let checked = self.check_expr(expr, interner)?;
        if ret.unqualified() == &Ty::Void {
            if checked.ty.unqualified() == &Ty::Void {
                return Ok(checked);
            }
            return Err(CompileError::new("return with a value in void function", expr.span()));
        }
        self.convert_for_assignment(ret, checked, interner)
    }

    /// Annotate an expression with types, inserting conversions. Takes the
    /// interner for identifier spellings in diagnostics; Sema itself
    /// stays interner-free.
    pub fn check_expr(&self, expr: &Expr, interner: &Interner) -> Result<CheckedExpr, CompileError> {
        let span = expr.span();
        match expr {
            Expr::Ident { name, .. } => {
                let Some(id) = self.lookup_symbol(*name) else {
                    let text = interner.resolve(*name).unwrap_or("?");
                    return Err(CompileError::new(format!("use of undeclared identifier '{text}'"), span));
                };
                let binding = &self.decls[id.0 as usize];
                if binding.kind == BindingKind::Typedef {
                    let text = interner.resolve(*name).unwrap_or("?");
                    return Err(CompileError::new(format!("typedef name '{text}' used as a value"), span));
                }
                Ok(CheckedExpr {
                    ty: binding.ty.clone(),
                    span,
                    kind: CheckedExprKind::Ident(id),
                })
            }
            Expr::Int { lit, .. } => {
                // The type comes from the value, not the suffix alone:
                // decimal `3000000000` is `long`, hex `0xFFFFFFFF` is
                // `unsigned int` (C17 §6.4.4.1). A GNU imaginary suffix
                // (`1i`) makes the value the imaginary part of a
                // complex number over that same type.
                let (ty, _) = crate::eval::int_constant(lit, interner, self.target.long_size).ok_or_else(|| CompileError::new("integer constant out of range", span))?;
                let ty = if lit.imag { Ty::Complex(Box::new(ty)) } else { ty };
                Ok(CheckedExpr {
                    ty,
                    span,
                    kind: CheckedExprKind::Int(*lit),
                })
            }
            Expr::Float { lit, .. } => {
                let base = match lit.suffix {
                    FloatSuffix::None => Ty::Double,
                    FloatSuffix::F => Ty::Float,
                    FloatSuffix::L => Ty::LongDouble,
                };
                // A GNU imaginary suffix (`1.0i`, `1.0fi`) makes the
                // value the imaginary part of a complex number.
                let ty = if lit.imag { Ty::Complex(Box::new(base)) } else { base };
                Ok(CheckedExpr {
                    ty,
                    span,
                    kind: CheckedExprKind::Float(*lit),
                })
            }
            // A narrow character constant checks as `int` (C17
            // §6.4.4.4p10); prefixed ones take their provided type
            // (`wchar_t` for `L`, `char16_t`/`char32_t` for `u`/`U`).
            Expr::Char { lit, .. } => Ok(CheckedExpr {
                ty: match lit.prefix {
                    CharPrefix::None => Ty::Int,
                    CharPrefix::L => self.target.wchar().ty(),
                    CharPrefix::SmallU => Ty::UnsignedShort,
                    CharPrefix::BigU => Ty::UnsignedInt,
                },
                span,
                kind: CheckedExprKind::Char(*lit),
            }),
            // Array of the literal's element type (`char` for narrow,
            // `int` for `L`, ...), bounded by its escape-aware element
            // count plus the NUL (`sizeof "ab"` is 3, including in
            // lowering, which reads the type rather than folding). A
            // malformed spelling keeps the incomplete type for the
            // caller to reject.
            Expr::Str { parts, .. } => Ok(CheckedExpr {
                ty: Ty::Array {
                    elem: Box::new(crate::eval::string_elem_ty_in(parts, self.target.wchar()).unwrap_or(Ty::Char)),
                    quals: Vec::new(),
                    bound: crate::eval::string_length(parts, interner).map(|len| ArrayBound::Fixed(len + 1)),
                },
                span,
                kind: CheckedExprKind::Str(parts.clone()),
            }),
            Expr::Unary { op, operand, .. } => self.check_unary(*op, operand, interner, span),
            Expr::Binary { op, lhs, rhs, .. } => self.check_binary(*op, lhs, rhs, interner, span),
            Expr::Assign { op, lhs, rhs, .. } => self.check_assign(*op, lhs, rhs, interner, span),
            Expr::Ternary { cond, then, els, .. } => self.check_ternary(cond, then, els, interner, span),
            Expr::Elvis { cond, els, .. } => self.check_elvis(cond, els, interner, span),
            Expr::Call { func, args, .. } => self.check_call(func, args, interner, span),
            Expr::Subscript { base, index, .. } => self.check_subscript(base, index, interner, span),
            Expr::Member { base, field, arrow, .. } => self.check_member(base, *field, *arrow, interner, span),
            Expr::Cast { ty, expr: operand, .. } => self.check_cast(ty, operand, interner, span),
            Expr::Generic { ctrl, assocs, .. } => self.check_generic(ctrl, assocs, interner, span),
            Expr::VaArg { ap, ty, .. } => {
                let ap = self.decay_checked(self.check_expr(ap, interner)?);
                // MSVC's `va_list` is a plain `char *` (and the backend
                // models it so on Windows), so pointers count there too.
                let va_list = ap.ty == Ty::BuiltinVaList || (self.target.os == Os::Windows && matches!(ap.ty, Ty::Pointer { .. }));
                if !va_list {
                    return Err(CompileError::new("`va_arg` needs a `va_list` argument", ap.span));
                }
                // The type must be a complete object type (neither
                // `void`, a function, nor an array). Promotable types
                // (`char`, `float`) stay accepted like GCC, which only
                // warns for those.
                match ty {
                    Ty::Void | Ty::Func { .. } | Ty::Array { .. } => {
                        return Err(CompileError::new("invalid type in `va_arg`", span));
                    }
                    t if !self.is_complete(t) => {
                        return Err(CompileError::new("invalid type in `va_arg`", span));
                    }
                    _ => {}
                }
                Ok(CheckedExpr {
                    ty: ty.clone(),
                    span,
                    kind: CheckedExprKind::VaArg { ap: Box::new(ap), ty: ty.clone() },
                })
            }
            Expr::Offsetof { ty, designators, .. } => {
                let offset = self.check_offsetof(ty, designators, interner, span)?;
                Ok(CheckedExpr {
                    ty: self.target.size_type(),
                    span,
                    kind: CheckedExprKind::Offsetof { ty: ty.clone(), offset },
                })
            }
            Expr::StmtExpr { items, .. } => {
                // Items checked inline at parse (declarations declared,
                // statements checked); the value is the last expression
                // statement, if any — `void` otherwise. Labels around it
                // do not hide the value (see `parse::stmt_expr_value`).
                let ty = match crate::parse::stmt_expr_value(items) {
                    Some(expr) => expr.ty.clone(),
                    None => Ty::Void,
                };
                Ok(CheckedExpr {
                    ty,
                    span,
                    kind: CheckedExprKind::StmtExpr { items: items.clone() },
                })
            }
            Expr::CompoundLiteral { ty, init, .. } => {
                // An array of unknown size completes from the
                // initializer (see `complete_compound_ty`), like a
                // declaration's `[]` does.
                let ty = self.complete_compound_ty(ty, init, interner);
                // Constraints (C17 §6.5.2.5p1): a complete object type,
                // never variably modified and never a function (GCC's
                // message is kept verbatim). `void` and other bad
                // shapes fail in the initializer check below.
                if matches!(ty.unqualified(), Ty::Func { .. }) {
                    return Err(CompileError::new("compound literal has function type", span));
                }
                if self.is_variably_modified(&ty, interner) {
                    return Err(CompileError::new("variable-sized object may not be initialized", span));
                }
                if !self.is_complete(&ty) || matches!(ty.unqualified(), Ty::Array { bound: None, .. }) {
                    return Err(CompileError::new("compound literal of incomplete type", span));
                }
                let init = self.check_initializer(&ty, init, interner)?;
                Ok(CheckedExpr {
                    ty: ty.clone(),
                    span,
                    kind: CheckedExprKind::CompoundLiteral {
                        ty: ty.clone(),
                        init: Box::new(init),
                    },
                })
            }
            Expr::SizeofExpr { expr: operand, .. } => {
                // The operand still checks (undeclared names must error);
                // `sizeof` itself needs a complete, non-void type — a
                // variable-length array keeps its node but never folds.
                let operand = self.check_expr(operand, interner)?;
                self.check_sizeof_operand(&operand.ty, operand.span)?;
                Ok(CheckedExpr {
                    ty: self.target.size_type(),
                    span,
                    kind: CheckedExprKind::SizeofExpr(Box::new(operand)),
                })
            }
            Expr::SizeofTy { ty, .. } => {
                self.check_sizeof_operand(ty, span)?;
                Ok(CheckedExpr {
                    ty: self.target.size_type(),
                    span,
                    kind: CheckedExprKind::SizeofTy(ty.clone()),
                })
            }
            Expr::AlignofTy { ty, .. } => {
                self.check_alignof_operand(ty, span)?;
                Ok(CheckedExpr {
                    ty: self.target.size_type(),
                    span,
                    kind: CheckedExprKind::AlignofTy(ty.clone()),
                })
            }
            Expr::AlignofExpr { expr: operand, .. } => {
                // The operand still checks (undeclared names must error).
                let operand = self.check_expr(operand, interner)?;
                self.check_alignof_operand(&operand.ty, operand.span)?;
                Ok(CheckedExpr {
                    ty: self.target.size_type(),
                    span,
                    kind: CheckedExprKind::AlignofExpr(Box::new(operand)),
                })
            }
            Expr::LabelAddr { label, .. } => Ok(CheckedExpr {
                ty: Ty::Pointer {
                    qualifiers: Vec::new(),
                    pointee: Box::new(Ty::Void),
                },
                span,
                kind: CheckedExprKind::LabelAddr(*label),
            }),
        }
    }

    /// Validate a `sizeof` operand's type. `void` and functions have no
    /// size in C, but GNU sizes them 1 (`sizeof(void)`, `sizeof f` for
    /// a function `f`) with a warning; incomplete types have no known
    /// size yet and stay errors.
    fn check_sizeof_operand(&self, ty: &Ty, span: Span) -> Result<(), CompileError> {
        match ty.unqualified() {
            Ty::Func { .. } => {
                self.warnings
                    .borrow_mut()
                    .push(CompileError::new("invalid application of `sizeof` to a function type", span));
                return Ok(());
            }
            Ty::Void => {
                self.warnings.borrow_mut().push(CompileError::warning("invalid application of `sizeof` to a void type", span));
                return Ok(());
            }
            _ => {}
        }
        if !self.is_complete(ty) {
            return Err(CompileError::new("invalid application of `sizeof` to an incomplete type", span));
        }
        Ok(())
    }

    /// Validate a `__alignof__` operand's type: functions have no
    /// alignment, and incomplete types have no known alignment yet.
    /// `void` is allowed and aligns to 1 (GNU `__alignof__(void)`).
    fn check_alignof_operand(&self, ty: &Ty, span: Span) -> Result<(), CompileError> {
        // `void` is not a complete type but does align to 1 (GNU), and
        // a function type aligns to 1 as well (its `aligned` attribute
        // is reported through `align_of_expr`).
        if matches!(ty.unqualified(), Ty::Void | Ty::Func { .. }) {
            return Ok(());
        }
        if !self.is_complete(ty) {
            return Err(CompileError::new("invalid application of `__alignof__` to an incomplete type", span));
        }
        Ok(())
    }

    fn check_unary(&self, op: UnaryOp, operand: &Expr, interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        use UnaryOp::*;
        match op {
            Pos | Neg => {
                let operand = self.decay_checked(self.check_expr(operand, interner)?);
                if !is_arithmetic(&operand.ty) {
                    return Err(CompileError::new("unary operand must have arithmetic type", operand.span));
                }
                let operand = self.integer_promote(operand);
                let ty = operand.ty.clone();
                Ok(CheckedExpr {
                    ty,
                    span,
                    kind: CheckedExprKind::Unary(op, Box::new(operand)),
                })
            }
            Not => {
                let operand = self.decay_checked(self.check_expr(operand, interner)?);
                if !is_scalar(&operand.ty) {
                    return Err(CompileError::new("operand of `!` must have scalar type", operand.span));
                }
                Ok(CheckedExpr {
                    ty: Ty::Int,
                    span,
                    kind: CheckedExprKind::Unary(op, Box::new(operand)),
                })
            }
            BitNot => {
                let operand = self.decay_checked(self.check_expr(operand, interner)?);
                // GNU `~x` on a complex operand is the conjugate
                // (negated imaginary part), not a bitwise complement.
                if is_complex(&operand.ty) {
                    let ty = operand.ty.clone();
                    return Ok(CheckedExpr {
                        ty,
                        span,
                        kind: CheckedExprKind::Unary(op, Box::new(operand)),
                    });
                }
                if !is_integer(&operand.ty) {
                    return Err(CompileError::new("operand of `~` must have integer type", operand.span));
                }
                let operand = self.integer_promote(operand);
                let ty = operand.ty.clone();
                Ok(CheckedExpr {
                    ty,
                    span,
                    kind: CheckedExprKind::Unary(op, Box::new(operand)),
                })
            }
            Deref => {
                let operand = self.decay_checked(self.check_expr(operand, interner)?);
                let Ty::Pointer { pointee, .. } = operand.ty.clone() else {
                    return Err(CompileError::new("indirection requires a pointer operand", operand.span));
                };
                Ok(CheckedExpr {
                    ty: *pointee,
                    span,
                    kind: CheckedExprKind::Unary(op, Box::new(operand)),
                })
            }
            AddrOf => {
                let operand = self.check_expr(operand, interner)?;
                let is_func = matches!(
                    operand.kind,
                    CheckedExprKind::Ident(id)
                        if self
                            .decls
                            .get(id.0 as usize)
                            .is_some_and(|b| b.kind == BindingKind::Func)
                );
                // String literals have static storage duration, so
                // `&"lit"` is the address of the literal object (a
                // `char (*)[N]` like GCC; the literal's own type is
                // already the bounded array). It is accepted with a
                // warning since the bare literal already decays to
                // `char *` and the `&` is usually redundant.
                let operand = operand;
                if matches!(operand.kind, CheckedExprKind::Str(_)) {
                    self.warnings.borrow_mut().push(CompileError::warning("taking the address of a string literal", span));
                }
                let is_str = matches!(operand.kind, CheckedExprKind::Str(_));
                if !is_func && !is_str && !self.is_lvalue(&operand) {
                    return Err(CompileError::new("cannot take the address of an rvalue", operand.span));
                }
                // A bit-field has no addressable object (C17 §6.5.3.2p1:
                // the `&` operand is an lvalue other than a bit-field),
                // like GCC/Clang's "cannot take address of bit-field".
                if Self::designates_bitfield(&operand) {
                    return Err(CompileError::new("cannot take the address of a bit-field", operand.span));
                }
                self.note_address_taken(&operand);
                let ty = Ty::Pointer {
                    qualifiers: Vec::new(),
                    pointee: Box::new(self.as_callable_ty(operand.ty.clone(), &operand.kind)),
                };
                Ok(CheckedExpr {
                    ty,
                    span,
                    kind: CheckedExprKind::Unary(op, Box::new(operand)),
                })
            }
            PreInc | PreDec | PostInc | PostDec => {
                // Modifiable-lvalue checks run before decay: arrays
                // decay to (scalar) pointers but are never modifiable,
                // and decay drops the `const` this must diagnose.
                let operand = self.check_expr(operand, interner)?;
                if !self.is_lvalue(&operand) {
                    return Err(CompileError::new("operand of `++`/`--` must be an lvalue", operand.span));
                }
                if operand.ty.is_const() {
                    return Err(self.readonly_error(&operand, interner));
                }
                if matches!(operand.ty.unqualified(), Ty::Array { .. }) {
                    return Err(CompileError::new("operand of `++`/`--` must be an lvalue", operand.span));
                }
                let operand = self.decay_checked(operand);
                if !is_scalar(&operand.ty) {
                    return Err(CompileError::new("operand of `++`/`--` must have scalar type", operand.span));
                }
                // Increment operands are real or pointer (C17
                // §6.5.3.1p1): complex has no `++`/`--`.
                if is_complex(&operand.ty) {
                    return Err(CompileError::new("operand of `++`/`--` must have real or pointer type", operand.span));
                }
                let ty = operand.ty.clone();
                Ok(CheckedExpr {
                    ty,
                    span,
                    kind: CheckedExprKind::Unary(op, Box::new(operand)),
                })
            }
            Real | Imag => {
                let operand = self.check_expr(operand, interner)?;
                let is_const = operand.ty.is_const();
                let operand = self.decay_checked(operand);
                let Ty::Complex(inner) = operand.ty.unqualified().clone() else {
                    return Err(CompileError::new("operand of `__real__`/`__imag__` must have complex type", operand.span));
                };
                // A part of a `const` object is itself `const`, like a
                // member of one: `__real__ cx = 1` fails for const `cx`.
                let mut ty = *inner;
                if is_const {
                    ty = Ty::qualified(vec![TypeQualifier::Const], ty);
                }
                Ok(CheckedExpr {
                    ty,
                    span,
                    kind: CheckedExprKind::Unary(op, Box::new(operand)),
                })
            }
        }
    }

    fn check_binary(&self, op: BinaryOp, lhs: &Expr, rhs: &Expr, interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        use BinaryOp::*;
        let lhs = self.check_expr(lhs, interner)?;
        let rhs = self.check_expr(rhs, interner)?;
        let build = |l: CheckedExpr, r: CheckedExpr, ty: Ty| CheckedExpr {
            ty,
            span,
            kind: CheckedExprKind::Binary(op, Box::new(l), Box::new(r)),
        };
        match op {
            // Shifts promote independently; the result takes the
            // promoted left type (C17 §6.5.7p3).
            Shl | Shr => {
                let (l, r) = (self.decay_checked(lhs), self.decay_checked(rhs));
                if !is_integer(&l.ty) || !is_integer(&r.ty) {
                    return Err(CompileError::new("shift operands must have integer type", span));
                }
                let l = self.integer_promote(l);
                let r = self.integer_promote(r);
                let ty = l.ty.clone();
                Ok(build(l, r, ty))
            }
            // Comparisons yield `int`; operands stay put, except a
            // null against a pointer converts to that pointer type
            // (C17 §6.5.9p2) so the backend compares equal widths.
            Eq | Ne => {
                let (l, r) = (self.decay_checked(lhs), self.decay_checked(rhs));
                if is_arithmetic(&l.ty) && is_arithmetic(&r.ty) {
                    let (l, r, _) = self.usual_arithmetic_conversions(l, r);
                    return Ok(build(l, r, Ty::Int));
                }
                if matches!(&l.ty, Ty::Pointer { .. }) && matches!(&r.ty, Ty::Pointer { .. }) {
                    return Ok(build(l, r, Ty::Int));
                }
                if matches!(&l.ty, Ty::Pointer { .. }) && self.is_null_pointer_constant(&r, interner) {
                    let ty = l.ty.clone();
                    return Ok(build(l, self.convert(r, ty), Ty::Int));
                }
                if matches!(&r.ty, Ty::Pointer { .. }) && self.is_null_pointer_constant(&l, interner) {
                    let ty = r.ty.clone();
                    return Ok(build(self.convert(l, ty), r, Ty::Int));
                }
                Err(CompileError::new("invalid operands to `==`/`!=`", span))
            }
            Lt | Le | Gt | Ge => {
                let (l, r) = (self.decay_checked(lhs), self.decay_checked(rhs));
                // Ordered comparisons need real operands: complex only
                // compares for equality (C17 §6.5.8p1, §6.5.9p2).
                if is_complex(&l.ty) || is_complex(&r.ty) {
                    return Err(CompileError::new("invalid operands to relational operator", span));
                }
                if is_arithmetic(&l.ty) && is_arithmetic(&r.ty) {
                    let (l, r, _) = self.usual_arithmetic_conversions(l, r);
                    return Ok(build(l, r, Ty::Int));
                }
                if matches!(&l.ty, Ty::Pointer { .. }) && matches!(&r.ty, Ty::Pointer { .. }) {
                    return Ok(build(l, r, Ty::Int));
                }
                Err(CompileError::new("invalid operands to relational operator", span))
            }
            And | Or => {
                let (l, r) = (self.decay_checked(lhs), self.decay_checked(rhs));
                if !is_scalar(&l.ty) || !is_scalar(&r.ty) {
                    return Err(CompileError::new("logical operands must have scalar type", span));
                }
                Ok(build(l, r, Ty::Int))
            }
            Comma => {
                let l = self.decay_checked(lhs);
                let r = self.decay_checked(rhs);
                let ty = r.ty.clone();
                Ok(build(l, r, ty))
            }
            // Additive and multiplicative operators, with pointer
            // arithmetic folded in (C17 §6.5.6).
            _ => {
                let (l, r) = (self.decay_checked(lhs), self.decay_checked(rhs));
                // Complex operands only add, subtract, multiply, and
                // divide: `%`, shifts, and bitwise operators need real
                // (integer) operands.
                if (is_complex(&l.ty) || is_complex(&r.ty)) && !matches!(op, Add | Sub | Mul | Div) {
                    return Err(CompileError::new("invalid operands to binary operator", span));
                }
                // `%` needs integer operands (C17 §6.5.5p2): float
                // remainders are `fmod`, not an operator. The backend
                // keeps its `todo!()` until a second frontend needs it.
                if matches!(op, Rem) && (!is_integer(&l.ty) || !is_integer(&r.ty)) {
                    return Err(CompileError::new("invalid operands to binary operator", span));
                }
                if is_arithmetic(&l.ty) && is_arithmetic(&r.ty) {
                    let (l, r, ty) = self.usual_arithmetic_conversions(l, r);
                    return Ok(build(l, r, ty));
                }
                let l_ptr = matches!(&l.ty, Ty::Pointer { .. });
                let r_ptr = matches!(&r.ty, Ty::Pointer { .. });
                let l_int = is_integer(&l.ty);
                let r_int = is_integer(&r.ty);
                if matches!(op, Add | Sub) && l_ptr && r_int {
                    let r = self.integer_promote(r);
                    let ty = l.ty.clone();
                    return Ok(build(l, r, ty));
                }
                if matches!(op, Add) && l_int && r_ptr {
                    let l = self.integer_promote(l);
                    let ty = r.ty.clone();
                    return Ok(build(l, r, ty));
                }
                if matches!(op, Sub) && l_ptr && r_ptr {
                    // `ptrdiff_t` is `long` on LP64, `long long` on
                    // Windows (LLP64).
                    let ty = self.target.ptrdiff_type();
                    return Ok(build(l, r, ty));
                }
                Err(CompileError::new("invalid operands to binary operator", span))
            }
        }
    }

    /// An `asm` output operand: a writable lvalue, like the left side of
    /// an assignment.
    pub fn check_asm_output(&self, expr: &Expr, interner: &Interner) -> Result<CheckedExpr, CompileError> {
        let checked = self.check_expr(expr, interner)?;
        if !self.is_lvalue(&checked) {
            return Err(CompileError::new("asm output is not an lvalue", checked.span));
        }
        if checked.ty.is_const() {
            return Err(self.readonly_error(&checked, interner));
        }
        Ok(checked)
    }

    fn check_assign(&self, op: AssignOp, lhs: &Expr, rhs: &Expr, interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let l = self.check_expr(lhs, interner)?;
        if !self.is_lvalue(&l) {
            return Err(CompileError::new("assignment to a non-lvalue", l.span));
        }
        // A `const`-qualified object is not assignable (C17 §6.5.16p2),
        // by plain, compound, or increment assignment alike.
        if l.ty.is_const() {
            return Err(self.readonly_error(&l, interner));
        }
        let ty = l.ty.clone();
        if op == AssignOp::Assign {
            let r = self.convert_for_assignment(&ty, self.check_expr(rhs, interner)?, interner)?;
            return Ok(CheckedExpr {
                ty,
                span,
                kind: CheckedExprKind::Assign(op, Box::new(l), Box::new(r)),
            });
        }
        // `a op= b` ≡ `a = (a) op (b)`, with the operation itself under
        // the usual arithmetic conversions — except pointer arithmetic,
        // which keeps the pointer and promotes only the integer side.
        let r = self.check_expr(rhs, interner)?;
        let l_dec = self.decay_checked(l.clone());
        let r_dec = self.decay_checked(r);
        // Complex operands only combine with `+`/`-`/`*`/`/` (see
        // `check_binary`): `%=` and the shift/bitwise forms need real
        // operands.
        if is_complex(&l_dec.ty) && !matches!(op, AssignOp::Add | AssignOp::Sub | AssignOp::Mul | AssignOp::Div) {
            return Err(CompileError::new("invalid operands to compound assignment", span));
        }
        // `%=` needs integer operands (C17 §6.5.5p2), like `%` above.
        if matches!(op, AssignOp::Rem) && (!is_integer(&l_dec.ty) || !is_integer(&r_dec.ty)) {
            return Err(CompileError::new("invalid operands to compound assignment", span));
        }
        let ptr_op = matches!(&l_dec.ty, Ty::Pointer { .. }) && matches!(op, AssignOp::Add | AssignOp::Sub) && is_integer(&r_dec.ty);
        let (op_l, op_r) = if ptr_op {
            (l_dec, self.integer_promote(r_dec.clone()))
        } else if is_arithmetic(&l_dec.ty) && is_arithmetic(&r_dec.ty) {
            let (a, b, _) = self.usual_arithmetic_conversions(l_dec, r_dec.clone());
            (a, b)
        } else {
            return Err(CompileError::new("invalid operands to compound assignment", span));
        };
        // The operation stays uncombined so the lowering reads the
        // lvalue exactly once; the operand types carry the conversions
        // computed above (the rhs converts in the lowering too).
        Ok(CheckedExpr {
            ty,
            span,
            kind: CheckedExprKind::CompoundAssign {
                op,
                lhs: Box::new(l),
                rhs: Box::new(r_dec),
                op_ty: Box::new(op_l.ty.clone()),
                rhs_ty: Box::new(op_r.ty.clone()),
            },
        })
    }

    fn check_ternary(&self, cond: &Expr, then: &Expr, els: &Expr, interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        // Array and function operands decay (C17 §6.3.2.1), so
        // `cond ? array : pointer` and `cond ? f : g` compare as
        // pointers below.
        let cond = self.decay_checked(self.check_expr(cond, interner)?);
        let then = self.decay_checked(self.check_expr(then, interner)?);
        let els = self.decay_checked(self.check_expr(els, interner)?);
        self.check_conditional(cond, then, els, interner, span, false)
    }

    /// GNU `x ?: y`: like a ternary whose then-arm is the condition
    /// itself, except that the condition is evaluated once (the
    /// lowering keeps its value in a slot). The result type comes from
    /// the same rules as the ternary's.
    fn check_elvis(&self, cond: &Expr, els: &Expr, interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let cond = self.decay_checked(self.check_expr(cond, interner)?);
        let els = self.decay_checked(self.check_expr(els, interner)?);
        let then = cond.clone();
        self.check_conditional(cond, then, els, interner, span, true)
    }

    /// The common tail of `?:`: the condition must be a scalar, and the
    /// arms convert to a shared result type. `elvis` picks the node
    /// shape (see [`Self::conditional_node`]).
    fn check_conditional(&self, cond: CheckedExpr, then: CheckedExpr, els: CheckedExpr, interner: &Interner, span: Span, elvis: bool) -> Result<CheckedExpr, CompileError> {
        if !is_scalar(&cond.ty) {
            return Err(CompileError::new("ternary condition must have scalar type", cond.span));
        }
        if is_arithmetic(&then.ty) && is_arithmetic(&els.ty) {
            let (then, els, ty) = self.usual_arithmetic_conversions(self.decay_checked(then), self.decay_checked(els));
            return Ok(self.conditional_node(cond, then, els, ty, span, elvis));
        }
        // A single void arm is a GNU extension (`c ? (void)0 : f()`):
        // the other arm's value is discarded and the result is void.
        if then.ty.unqualified() == &Ty::Void || els.ty.unqualified() == &Ty::Void {
            return Ok(self.conditional_node(cond, then, els, Ty::Void, span, elvis));
        }
        // Pointers: identical, `void *`-involved, or null-constant mixes.
        // Qualified mixes compose toward the more qualified side
        // (`cond ? char_ptr : const_char_ptr` is `const char *`), like
        // GCC and like assignment conversion: an exact (`Yes`) fit in
        // either direction beats a warning (`Warns`) fit, so a
        // qualifier-dropping direction never steals the composite.
        if matches!(&then.ty, Ty::Pointer { .. }) && matches!(&els.ty, Ty::Pointer { .. }) {
            if crate::sema::types_compatible(&then.ty, &els.ty) {
                let ty = then.ty.clone();
                return Ok(self.conditional_node(cond, then, els, ty, span, elvis));
            }
            // A null-constant arm (`(void *)0`) keeps the other arm's
            // pointer type (C17 §6.5.15p7), and that rule beats the
            // `void *` composition so `(i == 0 ? (void *)0 : s)->i`
            // stays `struct condstruct *` (tcctest.c). Two null
            // constants are both `void *`, so they return at the
            // compatibility check above.
            let then_null = self.is_null_pointer_constant(&then, interner);
            let els_null = self.is_null_pointer_constant(&els, interner);
            if then_null != els_null {
                let ty = if els_null { then.ty.clone() } else { els.ty.clone() };
                return Ok(self.conditional_node(cond, then, els, ty, span, elvis));
            }
            // `void *` mixes compose silently (C17 §6.5.15) before
            // assignment-compatibility is consulted: otherwise a
            // qualifier on either side (`const void *` vs `int *`,
            // `void *` vs `const int *`) falls into the warning path.
            if is_void_ptr(&then.ty) || is_void_ptr(&els.ty) {
                let ty = void_ptr_composite(&then.ty, &els.ty);
                return Ok(self.conditional_node(cond, then, els, ty, span, elvis));
            }
            let fwd = self.ptr_compat_kind(&then.ty, &els.ty);
            let rev = self.ptr_compat_kind(&els.ty, &then.ty);
            let take_then = fwd == PointeeConversion::Yes || (fwd == PointeeConversion::Warns && rev != PointeeConversion::Yes);
            if take_then && fwd != PointeeConversion::No {
                if fwd == PointeeConversion::Warns {
                    self.warnings.borrow_mut().push(CompileError::warning("pointer type mismatch in conditional expression", span));
                }
                let ty = then.ty.clone();
                return Ok(self.conditional_node(cond, then, els, ty, span, elvis));
            }
            if rev != PointeeConversion::No {
                if rev == PointeeConversion::Warns {
                    self.warnings.borrow_mut().push(CompileError::warning("pointer type mismatch in conditional expression", span));
                }
                let ty = els.ty.clone();
                return Ok(self.conditional_node(cond, then, els, ty, span, elvis));
            }
        }
        if (matches!(&then.ty, Ty::Pointer { .. }) && self.is_null_pointer_constant(&els, interner))
            || (matches!(&els.ty, Ty::Pointer { .. }) && self.is_null_pointer_constant(&then, interner))
        {
            let ty = if matches!(&then.ty, Ty::Pointer { .. }) { then.ty.clone() } else { els.ty.clone() };
            return Ok(self.conditional_node(cond, then, els, ty, span, elvis));
        }
        // Same struct/union on both sides keeps that type
        // (qualification aside: `cond ? cs : s` takes the left side,
        // like GCC's composite).
        if matches!(then.ty.unqualified(), Ty::Struct(_)) && then.ty.unqualified() == els.ty.unqualified() {
            let ty = then.ty.clone();
            return Ok(self.conditional_node(cond, then, els, ty, span, elvis));
        }
        Err(CompileError::new("incompatible operand types in ternary", span))
    }

    /// The node for a checked `?:`: `Ternary` for `c ? a : b`, `Elvis`
    /// for `c ?: b` — where the condition's own value is the result, so
    /// the converted then-arm is dropped (the lowering re-converts the
    /// condition value it already holds).
    fn conditional_node(&self, cond: CheckedExpr, then: CheckedExpr, els: CheckedExpr, ty: Ty, span: Span, elvis: bool) -> CheckedExpr {
        let kind = if elvis {
            CheckedExprKind::Elvis {
                cond: Box::new(cond),
                els: Box::new(els),
            }
        } else {
            CheckedExprKind::Ternary {
                cond: Box::new(cond),
                then: Box::new(then),
                els: Box::new(els),
            }
        };
        CheckedExpr { ty, span, kind }
    }

    fn check_call(&self, func: &Expr, args: &[Expr], interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        // GCC atomic builtins type-check against their arguments (the
        // types are generic over the pointed-to type), so they check
        // inline instead of resolving to a declared prototype.
        if let Expr::Ident { name, .. } = func {
            if let Some(text) = interner.resolve(*name) {
                if text.starts_with("__atomic_") {
                    return self.check_atomic(text, args, interner, span);
                }
                if text.starts_with("__c11_atomic_") {
                    return self.check_c11_atomic(text, args, interner, span);
                }
            }
            match interner.resolve(*name) {
                // Type-generic bit counts (`__builtin_clzg`,
                // `__builtin_ctzg`) likewise check inline: the first
                // argument keeps its own width (no promotion), with an
                // optional `int` zero-fallback second argument.
                Some("__builtin_clzg") => return self.check_clzg(args, interner, span, false),
                Some("__builtin_ctzg") => return self.check_clzg(args, interner, span, true),
                Some("__builtin_constant_p") => return self.check_constant_p(args, interner, span),
                Some("__builtin_isnan" | "__builtin_isnanf" | "__builtin_isnanl") => {
                    return self.check_fp_class(args, interner, span, FpPredicate::IsNan, "__builtin_isnan");
                }
                Some("__builtin_isinf" | "__builtin_isinff" | "__builtin_isinfl") => {
                    return self.check_fp_class(args, interner, span, FpPredicate::IsInf, "__builtin_isinf");
                }
                Some("__builtin_isinf_sign") => {
                    return self.check_fp_class(args, interner, span, FpPredicate::IsInfSign, "__builtin_isinf_sign");
                }
                Some("__builtin_isfinite") => {
                    return self.check_fp_class(args, interner, span, FpPredicate::IsFinite, "__builtin_isfinite");
                }
                Some("__builtin_isnormal") => {
                    return self.check_fp_class(args, interner, span, FpPredicate::IsNormal, "__builtin_isnormal");
                }
                Some("__builtin_signbit" | "__builtin_signbitf" | "__builtin_signbitl") => {
                    return self.check_fp_class(args, interner, span, FpPredicate::SignBit, "__builtin_signbit");
                }
                Some("__builtin_fpclassify") => return self.check_fpclassify(args, interner, span),
                Some("__builtin_classify_type") => return self.check_classify_type(args, interner, span),
                Some("__builtin_choose_expr") => return self.check_choose_expr(args, interner, span),
                Some("__builtin_prefetch") => return self.check_prefetch(args, interner, span),
                Some("alloca" | "__builtin_alloca") => {
                    return self.check_alloca(args, interner, span);
                }
                Some("__builtin_await") => return self.check_await(args, interner, span),
                Some("__builtin_frame_address") => {
                    return self.check_frame_address(args, interner, span, false);
                }
                Some("__builtin_return_address") => {
                    return self.check_frame_address(args, interner, span, true);
                }
                Some("__builtin_add_overflow") => {
                    return self.check_overflow(OverflowOp::Add, args, interner, span, None);
                }
                Some("__builtin_sub_overflow") => {
                    return self.check_overflow(OverflowOp::Sub, args, interner, span, None);
                }
                Some("__builtin_mul_overflow") => {
                    return self.check_overflow(OverflowOp::Mul, args, interner, span, None);
                }
                Some(name) => {
                    if let Some((op, operand_ty)) = typed_overflow_builtin(name) {
                        return self.check_overflow(op, args, interner, span, Some(operand_ty));
                    }
                }
                _ => {}
            }
        }
        let callee = self.decay_checked(self.check_expr(func, interner)?);
        let (ret, params, variadic, prototyped) = match &callee.ty {
            Ty::Func {
                ret,
                params,
                variadic,
                prototyped,
            } => (ret.clone(), params.clone(), *variadic, *prototyped),
            Ty::Pointer { pointee, .. } => match &**pointee {
                Ty::Func {
                    ret,
                    params,
                    variadic,
                    prototyped,
                } => (ret.clone(), params.clone(), *variadic, *prototyped),
                _ => {
                    return Err(CompileError::new("called object is not a function", callee.span));
                }
            },
            _ => {
                return Err(CompileError::new("called object is not a function", callee.span));
            }
        };
        let mut checked = Vec::with_capacity(args.len());
        if prototyped {
            if !variadic && args.len() != params.len() {
                return Err(CompileError::new(format!("wrong number of arguments: expected {}, got {}", params.len(), args.len()), span));
            }
            if variadic && args.len() < params.len() {
                return Err(CompileError::new(
                    format!("not enough arguments: expected at least {}, got {}", params.len(), args.len()),
                    span,
                ));
            }
            for (i, arg) in args.iter().enumerate() {
                let arg = self.check_expr(arg, interner)?;
                if let Some(param) = params.get(i) {
                    checked.push(self.convert_for_assignment(param, arg, interner)?);
                } else {
                    checked.push(self.default_arg_promote(arg));
                }
            }
        } else {
            for arg in args {
                checked.push(self.default_arg_promote(self.check_expr(arg, interner)?));
            }
        }
        Ok(CheckedExpr {
            ty: *ret,
            span,
            kind: CheckedExprKind::Call {
                func: Box::new(callee),
                args: checked,
            },
        })
    }

    /// `__builtin_await(handle)` (glcc extension): checks the operand
    /// to be a `_Future(T)` handle and yields a node of type `T`. Only
    /// valid inside an async function — elsewhere there is no
    /// coroutine to suspend, and the lowering rejects it anyway.
    fn check_await(&self, args: &[Expr], interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        if !self.async_enabled {
            return Err(CompileError::new("`__builtin_await` requires the glcc async extension (-fglcc-async)", span));
        }
        if !self.checking_async_fn.get() {
            return Err(CompileError::new("`__builtin_await` is only valid inside an async function", span));
        }
        if args.len() != 1 {
            return Err(CompileError::new(format!("`__builtin_await` takes exactly one argument, got {}", args.len()), span));
        }
        let arg = self.check_expr(&args[0], interner)?;
        let Some(result) = self.future_result_of(&arg.ty) else {
            return Err(CompileError::new("argument of `__builtin_await` is not a future", arg.span));
        };
        Ok(CheckedExpr {
            ty: result,
            span,
            kind: CheckedExprKind::Await(Box::new(arg)),
        })
    }

    /// All `__atomic_*` builtins: the `_n` forms plus the generic
    /// (pointer-to-value) forms, the sized `_1`/`_2`/`_4`/`_8`/`_16`
    /// libatomic spellings (checked like `_n` with the size verified),
    /// the read-modify-write family (`__atomic_fetch_add`,
    /// `__atomic_add_fetch`, ...), fences, lock-free queries, and
    /// `test_and_set`/`clear`. Memory orders take any integer (GCC
    /// further requires constants, which codegen ignores by mapping
    /// everything to seq-cst).
    fn check_atomic(&self, name: &str, args: &[Expr], interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let rest = &name["__atomic_".len()..];
        let (base, sized) = match rest.rsplit_once('_') {
            Some((b, "1")) => (b, Some(1u8)),
            Some((b, "2")) => (b, Some(2u8)),
            Some((b, "4")) => (b, Some(4u8)),
            Some((b, "8")) => (b, Some(8u8)),
            Some((b, "16")) => (b, Some(16u8)),
            Some((b, "n")) => (b, None),
            _ => (rest, None),
        };
        // `_n` was stripped above; `sized` is Some only for `_1` etc.
        // `is_n` marks the explicit `_n` spelling (vs generic with no
        // suffix) for the four ops that have both.
        let is_n = rest.ends_with("_n") || sized.is_some();
        match base {
            "load" if is_n => return self.check_atomic_load_n(args, interner, span, sized, name),
            "load" => return self.check_atomic_load(args, interner, span),
            "store" if is_n => return self.check_atomic_store_n(args, interner, span, sized, name),
            "store" => return self.check_atomic_store(args, interner, span),
            "exchange" if is_n => return self.check_atomic_exchange_n(args, interner, span, sized, name),
            "exchange" => return self.check_atomic_exchange(args, interner, span),
            "compare_exchange" if is_n => return self.check_atomic_cmpxchg_n(args, interner, span, sized, name),
            "compare_exchange" => return self.check_atomic_cmpxchg(args, interner, span),
            "fetch_add" | "fetch_sub" | "fetch_and" | "fetch_or" | "fetch_xor" | "fetch_nand" | "fetch_min" | "fetch_max" => {
                return self.check_atomic_rmw(base, true, args, interner, span, sized, name);
            }
            "add_fetch" | "sub_fetch" | "and_fetch" | "or_fetch" | "xor_fetch" | "nand_fetch" | "min_fetch" | "max_fetch" => {
                return self.check_atomic_rmw(base, false, args, interner, span, sized, name);
            }
            "thread_fence" => return self.check_atomic_fence(args, interner, span, false, name),
            "signal_fence" => return self.check_atomic_fence(args, interner, span, true, name),
            "always_lock_free" => return self.check_atomic_lock_free(args, interner, span, true, name),
            "is_lock_free" => return self.check_atomic_lock_free(args, interner, span, false, name),
            "test_and_set" => return self.check_atomic_test_and_set(args, interner, span),
            "clear" => return self.check_atomic_clear(args, interner, span),
            _ => {}
        }
        Err(CompileError::new(format!("unsupported atomic builtin `{name}`"), span))
    }

    /// The clang C11 atomics (`__c11_atomic_*`), what `<stdatomic.h>`
    /// expands to. Unlike the GCC spellings these carry no `_n`/size
    /// suffix and `compare_exchange` drops the `weak` flag (strong vs
    /// weak is in the name), so each maps onto the checker for the
    /// corresponding GCC form.
    fn check_c11_atomic(&self, name: &str, args: &[Expr], interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let base = &name["__c11_atomic_".len()..];
        match base {
            "load" => return self.check_atomic_load_n(args, interner, span, None, name),
            "store" => return self.check_atomic_store_n(args, interner, span, None, name),
            "exchange" => return self.check_atomic_exchange_n(args, interner, span, None, name),
            "init" => return self.check_c11_atomic_init(args, interner, span, name),
            "compare_exchange_strong" | "compare_exchange_weak" => {
                return self.check_c11_atomic_cmpxchg(args, interner, span, name);
            }
            "fetch_add" | "fetch_sub" | "fetch_and" | "fetch_or" | "fetch_xor" | "fetch_nand" | "fetch_min" | "fetch_max" => {
                return self.check_atomic_rmw(base, true, args, interner, span, None, name);
            }
            "thread_fence" => return self.check_atomic_fence(args, interner, span, false, name),
            "signal_fence" => return self.check_atomic_fence(args, interner, span, true, name),
            "is_lock_free" => return self.check_atomic_lock_free(args, interner, span, false, name),
            _ => {}
        }
        Err(CompileError::new(format!("unsupported atomic builtin `{name}`"), span))
    }

    /// C11 `void __c11_atomic_init(T *ptr, T val)`: like a store with
    /// no memory order. `_Atomic` erases to its base type, so `ptr` is
    /// an ordinary pointer to the object.
    fn check_c11_atomic_init(&self, args: &[Expr], interner: &Interner, span: Span, what: &str) -> Result<CheckedExpr, CompileError> {
        let [ptr, val] = args else {
            return Err(CompileError::new(format!("`{what}` takes a pointer and a value"), span));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let Ty::Pointer { pointee, .. } = ptr.ty.clone() else {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        };
        let op_ty = self.atomic_n_value_ty(&pointee, None, interner, what, ptr.span)?;
        let val = self.convert_for_assignment(&op_ty, self.check_expr(val, interner)?, interner)?;
        Ok(CheckedExpr {
            ty: Ty::Void,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::Init,
                args: vec![ptr, val],
            },
        })
    }

    /// C11 `bool __c11_atomic_compare_exchange_strong|weak(T *ptr,
    /// T *expected, T desired, int success, int failure)`: like the GCC
    /// `_n` form without the `weak` flag (already in the name).
    fn check_c11_atomic_cmpxchg(&self, args: &[Expr], interner: &Interner, span: Span, what: &str) -> Result<CheckedExpr, CompileError> {
        let [ptr, expected, desired, success, failure] = args else {
            return Err(CompileError::new(
                format!("`{what}` takes a pointer, an expected pointer, a desired value, and two memory orders"),
                span,
            ));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let Ty::Pointer { pointee, .. } = ptr.ty.clone() else {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        };
        let op_ty = self.atomic_n_value_ty(&pointee, None, interner, what, ptr.span)?;
        let expected = self.decay_checked(self.check_expr(expected, interner)?);
        let Some(esize) = self.ptr_pointee_size(&expected.ty, interner) else {
            return Err(CompileError::new(format!("`{what}` needs an expected pointer"), expected.span));
        };
        let psize = self.type_size(&op_ty, interner).unwrap_or(0);
        if esize != psize {
            return Err(CompileError::new(format!("`{what}` expected pointer disagrees in size"), expected.span));
        }
        let desired = self.convert_for_assignment(&op_ty, self.check_expr(desired, interner)?, interner)?;
        let success = self.check_atomic_order(success, interner, what)?;
        let failure = self.check_atomic_order(failure, interner, what)?;
        Ok(CheckedExpr {
            ty: Ty::Bool,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::CmpXchgN { sized: None },
                args: vec![ptr, expected, desired, success, failure],
            },
        })
    }

    fn check_atomic_order(&self, e: &Expr, interner: &Interner, what: &str) -> Result<CheckedExpr, CompileError> {
        let order = self.decay_checked(self.check_expr(e, interner)?);
        if !is_integer(&order.ty) {
            return Err(CompileError::new(format!("`{what}` needs an integer memory order"), order.span));
        }
        Ok(order)
    }

    fn check_atomic_sized(&self, pointee: &Ty, sized: Option<u8>, interner: &Interner, what: &str, pointee_span: Span) -> Result<(), CompileError> {
        if let Some(n) = sized {
            match self.type_size(pointee, interner) {
                Some(size) if size == u64::from(n) => Ok(()),
                _ => Err(CompileError::new(format!("`{what}` needs a {n}-byte object"), pointee_span)),
            }
        } else {
            Ok(())
        }
    }

    /// Unsigned integer type holding exactly `n` bytes: the operative
    /// type of a sized atomic (`_1`/`_2`/`_4`/`_8`/`_16`) through
    /// `void *`, which carries no size of its own.
    fn sized_atomic_ty(n: u8) -> Ty {
        match n {
            1 => Ty::UnsignedChar,
            2 => Ty::UnsignedShort,
            4 => Ty::UnsignedInt,
            8 => Ty::UnsignedLongLong,
            _ => Ty::UnsignedInt128,
        }
    }

    /// Value type an `_n`/sized atomic operates on: the pointee
    /// itself, except through `void *` (as libstdc++ passes the sized
    /// forms, e.g. `__atomic_load_8((const volatile void *)p, o)`),
    /// where the suffix gives the unsigned integer type. A bare `_n`
    /// through `void *` has no size at all and is rejected up front
    /// (rather than panicking the backend on a `void` result).
    fn atomic_n_value_ty(&self, pointee: &Ty, sized: Option<u8>, interner: &Interner, what: &str, span: Span) -> Result<Ty, CompileError> {
        if pointee.unqualified() == &Ty::Void {
            match sized {
                Some(n) => return Ok(Self::sized_atomic_ty(n)),
                None => return Err(CompileError::new(format!("`{what}` needs a pointer to an object type"), span)),
            }
        }
        self.check_atomic_sized(pointee, sized, interner, what, span)?;
        Ok(pointee.clone())
    }

    /// `T __atomic_load_n(T *ptr, int order)` (and `__atomic_load_8`
    /// etc, which verify the size): the pointer gives the result type.
    fn check_atomic_load_n(&self, args: &[Expr], interner: &Interner, span: Span, sized: Option<u8>, what: &str) -> Result<CheckedExpr, CompileError> {
        let [ptr, order] = args else {
            return Err(CompileError::new(format!("`{what}` takes a pointer and a memory order"), span));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let Ty::Pointer { pointee, .. } = ptr.ty.clone() else {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        };
        let op_ty = self.atomic_n_value_ty(&pointee, sized, interner, what, ptr.span)?;
        let order = self.check_atomic_order(order, interner, what)?;
        Ok(CheckedExpr {
            ty: op_ty,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::LoadN { sized },
                args: vec![ptr, order],
            },
        })
    }

    /// `void __atomic_store_n(T *ptr, T val, int order)`: the value
    /// converts to the pointed-to type as if by assignment.
    fn check_atomic_store_n(&self, args: &[Expr], interner: &Interner, span: Span, sized: Option<u8>, what: &str) -> Result<CheckedExpr, CompileError> {
        let [ptr, val, order] = args else {
            return Err(CompileError::new(format!("`{what}` takes a pointer, a value, and a memory order"), span));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let Ty::Pointer { pointee, .. } = ptr.ty.clone() else {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        };
        let op_ty = self.atomic_n_value_ty(&pointee, sized, interner, what, ptr.span)?;
        let val = self.convert_for_assignment(&op_ty, self.check_expr(val, interner)?, interner)?;
        let order = self.check_atomic_order(order, interner, what)?;
        Ok(CheckedExpr {
            ty: Ty::Void,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::StoreN { sized },
                args: vec![ptr, val, order],
            },
        })
    }

    /// `void __atomic_load(T *ptr, T *ret, int order)`: generic load
    /// through pointers; both point at the same-sized object.
    fn check_atomic_load(&self, args: &[Expr], interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let [ptr, ret, order] = args else {
            return Err(CompileError::new("`__atomic_load` takes a pointer, a result pointer, and a memory order", span));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let ret = self.decay_checked(self.check_expr(ret, interner)?);
        let (Some(psize), Some(rsize)) = (self.ptr_pointee_size(&ptr.ty, interner), self.ptr_pointee_size(&ret.ty, interner)) else {
            return Err(CompileError::new("`__atomic_load` needs pointer arguments", span));
        };
        if psize != rsize {
            return Err(CompileError::new("`__atomic_load` pointers disagree in size", span));
        }
        let order = self.check_atomic_order(order, interner, "__atomic_load")?;
        Ok(CheckedExpr {
            ty: Ty::Void,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::Load,
                args: vec![ptr, ret, order],
            },
        })
    }

    /// `void __atomic_store(T *ptr, T *val, int order)`: generic store.
    fn check_atomic_store(&self, args: &[Expr], interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let [ptr, val, order] = args else {
            return Err(CompileError::new("`__atomic_store` takes a pointer, a value pointer, and a memory order", span));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let val = self.decay_checked(self.check_expr(val, interner)?);
        let (Some(psize), Some(vsize)) = (self.ptr_pointee_size(&ptr.ty, interner), self.ptr_pointee_size(&val.ty, interner)) else {
            return Err(CompileError::new("`__atomic_store` needs pointer arguments", span));
        };
        if psize != vsize {
            return Err(CompileError::new("`__atomic_store` pointers disagree in size", span));
        }
        let order = self.check_atomic_order(order, interner, "__atomic_store")?;
        Ok(CheckedExpr {
            ty: Ty::Void,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::Store,
                args: vec![ptr, val, order],
            },
        })
    }

    fn ptr_pointee_size(&self, ty: &Ty, interner: &Interner) -> Option<u64> {
        match ty.unqualified() {
            Ty::Pointer { pointee, .. } => self.type_size(pointee, interner),
            _ => None,
        }
    }

    /// `T __atomic_exchange_n(T *ptr, T val, int order)`: returns the
    /// previous contents.
    fn check_atomic_exchange_n(&self, args: &[Expr], interner: &Interner, span: Span, sized: Option<u8>, what: &str) -> Result<CheckedExpr, CompileError> {
        let [ptr, val, order] = args else {
            return Err(CompileError::new(format!("`{what}` takes a pointer, a value, and a memory order"), span));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let Ty::Pointer { pointee, .. } = ptr.ty.clone() else {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        };
        let op_ty = self.atomic_n_value_ty(&pointee, sized, interner, what, ptr.span)?;
        let val = self.convert_for_assignment(&op_ty, self.check_expr(val, interner)?, interner)?;
        let order = self.check_atomic_order(order, interner, what)?;
        Ok(CheckedExpr {
            ty: op_ty,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::ExchangeN { sized },
                args: vec![ptr, val, order],
            },
        })
    }

    /// `void __atomic_exchange(T *ptr, T *val, T *ret, int order)`.
    fn check_atomic_exchange(&self, args: &[Expr], interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let [ptr, val, ret, order] = args else {
            return Err(CompileError::new(
                "`__atomic_exchange` takes a pointer, a value pointer, a result pointer, and a memory order",
                span,
            ));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let val = self.decay_checked(self.check_expr(val, interner)?);
        let ret = self.decay_checked(self.check_expr(ret, interner)?);
        let (Some(a), Some(b), Some(c)) = (
            self.ptr_pointee_size(&ptr.ty, interner),
            self.ptr_pointee_size(&val.ty, interner),
            self.ptr_pointee_size(&ret.ty, interner),
        ) else {
            return Err(CompileError::new("`__atomic_exchange` needs pointer arguments", span));
        };
        if a != b || a != c {
            return Err(CompileError::new("`__atomic_exchange` pointers disagree in size", span));
        }
        let order = self.check_atomic_order(order, interner, "__atomic_exchange")?;
        Ok(CheckedExpr {
            ty: Ty::Void,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::Exchange,
                args: vec![ptr, val, ret, order],
            },
        })
    }

    /// `bool __atomic_compare_exchange_n(T *ptr, T *expected, T desired,
    /// bool weak, int success, int failure)`: on failure writes the
    /// current value back to `*expected`.
    fn check_atomic_cmpxchg_n(&self, args: &[Expr], interner: &Interner, span: Span, sized: Option<u8>, what: &str) -> Result<CheckedExpr, CompileError> {
        let [ptr, expected, desired, weak, success, failure] = args else {
            return Err(CompileError::new(
                format!("`{what}` takes a pointer, an expected pointer, a desired value, a weak flag, and two memory orders"),
                span,
            ));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let Ty::Pointer { pointee, .. } = ptr.ty.clone() else {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        };
        let op_ty = self.atomic_n_value_ty(&pointee, sized, interner, what, ptr.span)?;
        let expected = self.decay_checked(self.check_expr(expected, interner)?);
        let Some(esize) = self.ptr_pointee_size(&expected.ty, interner) else {
            return Err(CompileError::new(format!("`{what}` needs an expected pointer"), expected.span));
        };
        let psize = self.type_size(&op_ty, interner).unwrap_or(0);
        if esize != psize {
            return Err(CompileError::new(format!("`{what}` expected pointer disagrees in size"), expected.span));
        }
        let desired = self.convert_for_assignment(&op_ty, self.check_expr(desired, interner)?, interner)?;
        let weak = self.decay_checked(self.check_expr(weak, interner)?);
        if !is_integer(&weak.ty) {
            return Err(CompileError::new(format!("`{what}` needs an integer weak flag"), weak.span));
        }
        let success = self.check_atomic_order(success, interner, what)?;
        let failure = self.check_atomic_order(failure, interner, what)?;
        Ok(CheckedExpr {
            ty: Ty::Bool,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::CmpXchgN { sized },
                args: vec![ptr, expected, desired, weak, success, failure],
            },
        })
    }

    /// `bool __atomic_compare_exchange(T *ptr, T *expected, T *desired,
    /// bool weak, int success, int failure)`.
    fn check_atomic_cmpxchg(&self, args: &[Expr], interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let what = "__atomic_compare_exchange";
        let [ptr, expected, desired, weak, success, failure] = args else {
            return Err(CompileError::new(
                format!("`{what}` takes a pointer, an expected pointer, a desired pointer, a weak flag, and two memory orders"),
                span,
            ));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let expected = self.decay_checked(self.check_expr(expected, interner)?);
        let desired = self.decay_checked(self.check_expr(desired, interner)?);
        let (Some(a), Some(b), Some(c)) = (
            self.ptr_pointee_size(&ptr.ty, interner),
            self.ptr_pointee_size(&expected.ty, interner),
            self.ptr_pointee_size(&desired.ty, interner),
        ) else {
            return Err(CompileError::new(format!("`{what}` needs pointer arguments"), span));
        };
        if a != b || a != c {
            return Err(CompileError::new(format!("`{what}` pointers disagree in size"), span));
        }
        let weak = self.decay_checked(self.check_expr(weak, interner)?);
        if !is_integer(&weak.ty) {
            return Err(CompileError::new(format!("`{what}` needs an integer weak flag"), weak.span));
        }
        let success = self.check_atomic_order(success, interner, what)?;
        let failure = self.check_atomic_order(failure, interner, what)?;
        Ok(CheckedExpr {
            ty: Ty::Bool,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::CmpXchg,
                args: vec![ptr, expected, desired, weak, success, failure],
            },
        })
    }

    /// `T __atomic_fetch_add(T *ptr, T val, int order)` (old value) and
    /// `T __atomic_add_fetch(T *ptr, T val, int order)` (new value),
    /// plus the sized spellings. Pointer targets act on `uintptr_t`
    /// (no scaling); `_Bool` targets are rejected like GCC.
    fn check_atomic_rmw(&self, base: &str, fetch: bool, args: &[Expr], interner: &Interner, span: Span, sized: Option<u8>, what: &str) -> Result<CheckedExpr, CompileError> {
        let op = match base {
            "fetch_add" | "add_fetch" => RmwOp::Add,
            "fetch_sub" | "sub_fetch" => RmwOp::Sub,
            "fetch_and" | "and_fetch" => RmwOp::And,
            "fetch_or" | "or_fetch" => RmwOp::Or,
            "fetch_xor" | "xor_fetch" => RmwOp::Xor,
            "fetch_nand" | "nand_fetch" => RmwOp::Nand,
            "fetch_min" | "min_fetch" => RmwOp::Min,
            "fetch_max" | "max_fetch" => RmwOp::Max,
            _ => return Err(CompileError::new(format!("unsupported atomic builtin `{what}`"), span)),
        };
        let [ptr, val, order] = args else {
            return Err(CompileError::new(format!("`{what}` takes a pointer, a value, and a memory order"), span));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let Ty::Pointer { pointee, .. } = ptr.ty.clone() else {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        };
        let op_ty = self.atomic_n_value_ty(&pointee, sized, interner, what, ptr.span)?;
        if matches!(op_ty.unqualified(), Ty::Bool) {
            return Err(CompileError::new(format!("`{what}` must not operate on `_Bool`"), ptr.span));
        }
        if matches!(op, RmwOp::Min | RmwOp::Max) && !is_integer(&op_ty) {
            return Err(CompileError::new(format!("`{what}` needs an integer target"), ptr.span));
        }
        let val = match op_ty.unqualified() {
            Ty::Pointer { .. } => {
                let v = self.decay_checked(self.check_expr(val, interner)?);
                if !is_integer(&v.ty) {
                    return Err(CompileError::new(format!("`{what}` needs an integer value for a pointer target"), v.span));
                }
                v
            }
            _ => {
                if !is_integer(&op_ty) {
                    return Err(CompileError::new(format!("`{what}` needs an integer or pointer target"), ptr.span));
                }
                self.convert_for_assignment(&op_ty, self.check_expr(val, interner)?, interner)?
            }
        };
        let order = self.check_atomic_order(order, interner, what)?;
        Ok(CheckedExpr {
            ty: op_ty,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::Rmw { op, fetch, sized },
                args: vec![ptr, val, order],
            },
        })
    }

    /// `void __atomic_thread_fence(int order)` /
    /// `void __atomic_signal_fence(int order)`.
    fn check_atomic_fence(&self, args: &[Expr], interner: &Interner, span: Span, signal: bool, what: &str) -> Result<CheckedExpr, CompileError> {
        let [order] = args else {
            return Err(CompileError::new(format!("`{what}` takes a memory order"), span));
        };
        let order = self.check_atomic_order(order, interner, what)?;
        Ok(CheckedExpr {
            ty: Ty::Void,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::Fence { signal },
                args: vec![order],
            },
        })
    }

    /// `bool __atomic_always_lock_free(size_t size[, void *ptr])` (size
    /// must be constant) and `bool __atomic_is_lock_free(...)` (may be
    /// dynamic; constant sizes fold here, dynamic ones lower to a size
    /// comparison).
    fn check_atomic_lock_free(&self, args: &[Expr], interner: &Interner, span: Span, always: bool, what: &str) -> Result<CheckedExpr, CompileError> {
        if args.len() < 1 || args.len() > 2 {
            return Err(CompileError::new(format!("`{what}` takes a size and an optional pointer"), span));
        }
        let size = self.decay_checked(self.check_expr(&args[0], interner)?);
        if !is_integer(&size.ty) {
            return Err(CompileError::new(format!("`{what}` needs an integer size"), size.span));
        }
        let mut checked = vec![size];
        if let Some(ptr) = args.get(1) {
            let p = self.decay_checked(self.check_expr(ptr, interner)?);
            match &p.ty {
                t if is_integer(t) => {}
                Ty::Pointer { .. } => {}
                _ => return Err(CompileError::new(format!("`{what}` needs a pointer or zero second argument"), p.span)),
            }
            checked.push(p);
        }
        if always
            && let [size] = &checked[..1]
            && crate::eval::eval_const(size, self, interner).is_none()
        {
            return Err(CompileError::new(format!("`{what}` needs a constant size"), span));
        }
        if let [size] = &checked[..]
            && let Some(crate::eval::Value::Int(n)) = crate::eval::eval_const(size, self, interner)
        {
            let lock_free = matches!(n, 1 | 2 | 4 | 8);
            return Ok(CheckedExpr {
                ty: Ty::Bool,
                span,
                kind: CheckedExprKind::ConstantP(lock_free),
            });
        }
        Ok(CheckedExpr {
            ty: Ty::Bool,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::LockFree { always },
                args: checked,
            },
        })
    }

    /// `bool __atomic_test_and_set(void *ptr, int order)`.
    fn check_atomic_test_and_set(&self, args: &[Expr], interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let what = "__atomic_test_and_set";
        let [ptr, order] = args else {
            return Err(CompileError::new(format!("`{what}` takes a pointer and a memory order"), span));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        if !matches!(ptr.ty.unqualified(), Ty::Pointer { .. }) {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        }
        let order = self.check_atomic_order(order, interner, what)?;
        Ok(CheckedExpr {
            ty: Ty::Bool,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::TestAndSet,
                args: vec![ptr, order],
            },
        })
    }

    /// `void __atomic_clear(bool *ptr, int order)`.
    fn check_atomic_clear(&self, args: &[Expr], interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let what = "__atomic_clear";
        let [ptr, order] = args else {
            return Err(CompileError::new(format!("`{what}` takes a pointer and a memory order"), span));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        if !matches!(ptr.ty.unqualified(), Ty::Pointer { .. }) {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        }
        let order = self.check_atomic_order(order, interner, what)?;
        Ok(CheckedExpr {
            ty: Ty::Void,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::Clear,
                args: vec![ptr, order],
            },
        })
    }

    /// `int __builtin_clzg(x[, fallback])` (and `ctzg`): the first
    /// argument is any unsigned integer type (standard, extended, or
    /// bit-precise) and keeps its width — no promotion — so the count
    /// is within the argument's own bits. With two arguments, a zero
    /// first argument yields the second (an `int`); with one, zero is
    /// undefined (like the fixed-width `clz`, which shares the
    /// backend lowering shape).
    fn check_clzg(&self, args: &[Expr], interner: &Interner, span: Span, ctz: bool) -> Result<CheckedExpr, CompileError> {
        let what = if ctz { "`__builtin_ctzg`" } else { "`__builtin_clzg`" };
        if args.len() < 1 || args.len() > 2 {
            return Err(CompileError::new(format!("{what} takes one unsigned argument and an optional `int` fallback"), span));
        }
        let arg = self.decay_checked(self.check_expr(&args[0], interner)?);
        if !matches!(
            arg.ty.unqualified(),
            Ty::Bool | Ty::UnsignedChar | Ty::UnsignedShort | Ty::UnsignedInt | Ty::UnsignedLong | Ty::UnsignedLongLong | Ty::UnsignedInt128
        ) {
            return Err(CompileError::new(format!("{what} requires an unsigned integer argument"), arg.span));
        }
        let fallback = match args.get(1) {
            Some(fb) => Some(Box::new(self.convert_for_assignment(&Ty::Int, self.check_expr(fb, interner)?, interner)?)),
            None => None,
        };
        Ok(CheckedExpr {
            ty: Ty::Int,
            span,
            kind: CheckedExprKind::Clz {
                ctz,
                arg: Box::new(arg),
                fallback,
            },
        })
    }

    /// `int __builtin_constant_p(exp)`: whether `exp` is known to be
    /// constant at compile time. The operand checks normally but is
    /// then discarded (GCC never evaluates it, side effects
    /// included); only the folded 0/1 remains, so the node itself is
    /// always constant.
    fn check_constant_p(&self, args: &[Expr], interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let [arg] = args else {
            return Err(CompileError::new("`__builtin_constant_p` takes exactly one argument", span));
        };
        let checked = self.decay_checked(self.check_expr(arg, interner)?);
        // A string literal is a constant-pool entry: GCC reports it
        // constant even though its value is an address (`&global` is
        // not).
        let is_const = matches!(checked.kind, CheckedExprKind::Str(_)) || crate::eval::eval_const(&checked, self, interner).is_some();
        Ok(CheckedExpr {
            ty: Ty::Int,
            span,
            kind: CheckedExprKind::ConstantP(is_const),
        })
    }

    /// The single-argument floating-point classifications
    /// (`__builtin_isnan`, `__builtin_isinf`, `__builtin_isinf_sign`,
    /// `__builtin_isfinite`, `__builtin_isnormal`): single evaluation of
    /// the operand, unlike an `x != x` desugaring. Integers are never
    /// NaN, infinite or subnormal, so they fold to `0` immediately
    /// (`signbit` is the exception: its `double` parameter makes the
    /// sign of `-1` count); anything non-arithmetic is rejected.
    fn check_fp_class(&self, args: &[Expr], interner: &Interner, span: Span, op: FpPredicate, what: &str) -> Result<CheckedExpr, CompileError> {
        let [arg] = args else {
            return Err(CompileError::new(format!("`{what}` takes exactly one argument"), span));
        };
        let arg = self.decay_checked(self.check_expr(arg, interner)?);
        if !is_arithmetic(&arg.ty) {
            return Err(CompileError::new(format!("`{what}` needs an arithmetic argument"), arg.span));
        }
        if is_integer(&arg.ty) && op != FpPredicate::SignBit {
            return Ok(CheckedExpr {
                ty: Ty::Int,
                span,
                kind: CheckedExprKind::ConstantP(false),
            });
        }
        // `__builtin_signbit`'s parameter is a `double`, so an integer
        // converts first and its sign decides (`signbit(-1)` is true).
        let arg = if is_integer(&arg.ty) {
            self.convert_for_assignment(&Ty::Double, arg, interner)?
        } else {
            arg
        };
        Ok(CheckedExpr {
            ty: Ty::Int,
            span,
            kind: CheckedExprKind::FpClass { op, arg: Box::new(arg) },
        })
    }

    /// `int __builtin_fpclassify(nan, inf, normal, subnormal, zero, x)`:
    /// the `int` code for `x`'s category, with the five codes supplied
    /// by the caller (`<math.h>`'s `fpclassify` macro passes `FP_NAN`,
    /// `FP_INFINITE`, `FP_NORMAL`, `FP_SUBNORMAL` and `FP_ZERO`). GCC
    /// wants an integer constant for each code; any `int` expression is
    /// accepted here. The operand must be real floating point (GCC
    /// rejects integers and complex; integers would have a constant
    /// category) and evaluates once.
    fn check_fpclassify(&self, args: &[Expr], interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let [nan, inf, normal, subnormal, zero, arg] = args else {
            return Err(CompileError::new("`__builtin_fpclassify` takes five `int` codes and a floating-point operand", span));
        };
        let code = |expr: &Expr| -> Result<CheckedExpr, CompileError> {
            let checked = self.check_expr(expr, interner)?;
            self.convert_for_assignment(&Ty::Int, checked, interner)
        };
        let codes = [
            Box::new(code(nan)?),
            Box::new(code(inf)?),
            Box::new(code(normal)?),
            Box::new(code(subnormal)?),
            Box::new(code(zero)?),
        ];
        let arg = self.decay_checked(self.check_expr(arg, interner)?);
        if !is_real_float(&arg.ty) {
            return Err(CompileError::new("`__builtin_fpclassify` needs a floating-point operand", arg.span));
        }
        Ok(CheckedExpr {
            ty: Ty::Int,
            span,
            kind: CheckedExprKind::FpClassify { codes, arg: Box::new(arg) },
        })
    }

    /// `__builtin_choose_expr(cond, a, b)`: `cond` must fold to an
    /// integer constant; the result is the chosen operand with its own
    /// type, and only that one is checked — GCC leaves the other
    /// unevaluated, so it need not even be valid.
    fn check_choose_expr(&self, args: &[Expr], interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let [cond, then_expr, else_expr] = args else {
            return Err(CompileError::new("`__builtin_choose_expr` takes three arguments", span));
        };
        let checked = self.decay_checked(self.check_expr(cond, interner)?);
        let Some(value) = crate::eval::eval_const(&checked, self, interner) else {
            return Err(CompileError::new("`__builtin_choose_expr` condition is not a constant", span));
        };
        self.check_expr(if crate::eval::truthy(&value) { then_expr } else { else_expr }, interner)
    }

    /// `int __builtin_classify_type(x)`: the type class of the
    /// argument's type, folded to a constant. The argument is checked
    /// (undeclared names and type errors still surface) but never
    /// evaluated, like GCC documents.
    fn check_classify_type(&self, args: &[Expr], interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let [arg] = args else {
            return Err(CompileError::new("`__builtin_classify_type` takes exactly one argument", span));
        };
        let arg = self.check_expr(arg, interner)?;
        Ok(CheckedExpr {
            ty: Ty::Int,
            span,
            kind: CheckedExprKind::ClassifyType(self.type_class(&arg.ty)),
        })
    }

    /// `void __builtin_prefetch(addr[, rw[, locality]])`: a hint, so
    /// only the address is kept (evaluated and dropped through a
    /// `void` cast); the access/locality arguments are checked and
    /// discarded, like GCC's constant-only ones.
    fn check_prefetch(&self, args: &[Expr], interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let Some((addr, rest)) = args.split_first() else {
            return Err(CompileError::new("`__builtin_prefetch` needs an address", span));
        };
        if rest.len() > 2 {
            return Err(CompileError::new("`__builtin_prefetch` takes at most three arguments", span));
        }
        let addr = self.decay_checked(self.check_expr(addr, interner)?);
        if !matches!(addr.ty.unqualified(), Ty::Pointer { .. }) {
            return Err(CompileError::new("`__builtin_prefetch` needs a pointer address", addr.span));
        }
        for arg in rest {
            self.check_expr(arg, interner)?;
        }
        Ok(CheckedExpr {
            ty: Ty::Void,
            span,
            kind: CheckedExprKind::Cast {
                expr: Box::new(addr),
                ty: Ty::Void,
            },
        })
    }

    /// GCC's overflow builtins: the generic `__builtin_add_overflow`
    /// and friends plus the typed spellings (`__builtin_sadd_overflow`
    /// family, recognized by `typed_overflow_builtin`). Two integer
    /// operands and a pointer receiving the wrapped result. The
    /// pointed-to type fixes the width and signedness the result
    /// converts to; the operation itself is exact, and the builtin
    /// returns whether the exact value fit. A typed spelling pins the
    /// operand type, converting operands like prototyped parameters.
    fn check_overflow(&self, op: OverflowOp, args: &[Expr], interner: &Interner, span: Span, operand_ty: Option<Ty>) -> Result<CheckedExpr, CompileError> {
        let [lhs, rhs, res] = args else {
            return Err(CompileError::new("overflow builtin needs two operands and a result pointer", span));
        };
        let mut lhs = self.check_expr(lhs, interner)?;
        let mut rhs = self.check_expr(rhs, interner)?;
        for operand in [&lhs, &rhs] {
            if int_rank(&operand.ty).is_none() {
                return Err(CompileError::new("overflow builtin operands must be integers", operand.span));
            }
        }
        if let Some(ty) = &operand_ty {
            lhs = self.convert_for_assignment(ty, lhs, interner)?;
            rhs = self.convert_for_assignment(ty, rhs, interner)?;
        }
        let res = self.decay_checked(self.check_expr(res, interner)?);
        let Ty::Pointer { pointee, .. } = res.ty.unqualified() else {
            return Err(CompileError::new("overflow builtin result must be a pointer to integer", res.span));
        };
        let res_ty = pointee.as_ref().clone();
        if int_rank(&res_ty).is_none() {
            return Err(CompileError::new("overflow builtin result must be a pointer to integer", res.span));
        }
        Ok(CheckedExpr {
            ty: Ty::Int,
            span,
            kind: CheckedExprKind::Overflow {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                res: Box::new(res),
                res_ty,
            },
        })
    }

    /// `alloca`/`__builtin_alloca`: allocate `size` bytes that stay
    /// live until the enclosing function returns. The frontend only
    /// marks the call; the heap block and its cleanup lower in MIR.
    fn check_alloca(&self, args: &[Expr], interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let [size] = args else {
            return Err(CompileError::new("`alloca` needs exactly one argument", span));
        };
        let size = self.check_expr(size, interner)?;
        let size = self.convert_for_assignment(&self.target.size_type(), size, interner)?;
        Ok(CheckedExpr {
            ty: Ty::Pointer {
                qualifiers: Vec::new(),
                pointee: Box::new(Ty::Void),
            },
            span,
            kind: CheckedExprKind::Alloca { size: Box::new(size) },
        })
    }

    /// `void *__builtin_frame_address(unsigned level)` and
    /// `__builtin_return_address`: the frame or return address of the
    /// `level`-th enclosing frame. GCC requires the level to be a
    /// constant integer, so a non-constant one is rejected here;
    /// lowering walks the frame-pointer chain from the current frame.
    fn check_frame_address(&self, args: &[Expr], interner: &Interner, span: Span, returns_address: bool) -> Result<CheckedExpr, CompileError> {
        let what = if returns_address { "__builtin_return_address" } else { "__builtin_frame_address" };
        let [level] = args else {
            return Err(CompileError::new(format!("`{what}` needs exactly one argument"), span));
        };
        let level = self.decay_checked(self.check_expr(level, interner)?);
        let Some(crate::eval::Value::Int(n)) = crate::eval::eval_const(&level, self, interner) else {
            return Err(CompileError::new(format!("`{what}` level must be a constant integer"), level.span));
        };
        let Ok(level) = u64::try_from(n) else {
            return Err(CompileError::new(format!("`{what}` level must be a non-negative constant integer"), level.span));
        };
        Ok(CheckedExpr {
            ty: Ty::Pointer {
                qualifiers: Vec::new(),
                pointee: Box::new(Ty::Void),
            },
            span,
            kind: CheckedExprKind::FrameAddress { level, returns_address },
        })
    }

    /// GCC's `__builtin_classify_type` codes, in GCC's `typeclass`
    /// order: void 0, integer 1, char 2, enum 3, bool 4, pointer 5,
    /// reference 6, offset 7, real 8, complex 9, function 10, method
    /// 11, record 12, union 13, array 14, string 15. C `char`s are
    /// integers here, like GCC's own classification.
    fn type_class(&self, ty: &Ty) -> u8 {
        match ty.unqualified() {
            Ty::Void => 0,
            Ty::Char
            | Ty::SignedChar
            | Ty::UnsignedChar
            | Ty::Short
            | Ty::UnsignedShort
            | Ty::Int
            | Ty::UnsignedInt
            | Ty::Long
            | Ty::UnsignedLong
            | Ty::LongLong
            | Ty::UnsignedLongLong
            | Ty::Int128
            | Ty::UnsignedInt128 => 1,
            Ty::Enum(_) => 3,
            Ty::Bool => 4,
            Ty::Pointer { .. } => 5,
            Ty::Float | Ty::Double | Ty::LongDouble | Ty::Float128 => 8,
            Ty::Complex(_) => 9,
            Ty::Func { .. } => 10,
            // `__builtin_va_list` is an array of one tag on the
            // glcc targets.
            Ty::BuiltinVaList => 14,
            Ty::Struct(id) if self.structs[id.0 as usize].is_union => 13,
            Ty::Struct(_) => 12,
            Ty::Array { .. } => 14,
            // `unqualified` strips these, so this is unreachable.
            Ty::Qualified { inner, .. } => self.type_class(inner),
        }
    }

    fn check_subscript(&self, base: &Expr, index: &Expr, interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let base = self.decay_checked(self.check_expr(base, interner)?);
        let index = self.decay_checked(self.check_expr(index, interner)?);
        // Either side may be the pointer (`i[a]` is legal).
        let (ptr, idx) = match (&base.ty, &index.ty) {
            (Ty::Pointer { .. }, t) if is_integer(t) => (base, index),
            (t, Ty::Pointer { .. }) if is_integer(t) => (index, base),
            _ => {
                return Err(CompileError::new("subscript requires a pointer and an integer", span));
            }
        };
        let idx = self.integer_promote(idx);
        let Ty::Pointer { pointee, .. } = ptr.ty.clone() else {
            unreachable!("pointer selected above")
        };
        Ok(CheckedExpr {
            ty: *pointee,
            span,
            kind: CheckedExprKind::Subscript {
                base: Box::new(ptr),
                index: Box::new(idx),
            },
        })
    }

    fn check_member(&self, base: &Expr, field: SymbolU32, arrow: bool, interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let mut checked = self.check_expr(base, interner)?;
        if arrow {
            checked = self.decay_checked(checked);
            if !matches!(&checked.ty, Ty::Pointer { .. }) {
                return Err(CompileError::new("`->` requires a pointer to struct or union", checked.span));
            }
        }
        let target_ty = if arrow {
            match checked.ty.unqualified() {
                Ty::Pointer { pointee, .. } => (**pointee).clone(),
                _ => unreachable!("pointer checked above"),
            }
        } else {
            checked.ty.clone()
        };
        let Ty::Struct(id) = target_ty.unqualified() else {
            return Err(CompileError::new("member access requires a struct or union", span));
        };
        let id = *id;
        let mut visiting = Vec::new();
        let path = match self.find_field(id, field, &mut visiting) {
            FieldHit::Found(path) => path,
            FieldHit::Miss => {
                let text = interner.resolve(field).unwrap_or("?");
                return Err(CompileError::new(format!("no member named '{text}'"), span));
            }
            FieldHit::Ambiguous => {
                let text = interner.resolve(field).unwrap_or("?");
                return Err(CompileError::new(format!("ambiguous member '{text}' (found in several anonymous members)"), span));
            }
        };
        let mut ty = Ty::Struct(id);
        for &index in &path {
            let Ty::Struct(cur) = ty.unqualified() else {
                unreachable!("member path crosses only structs")
            };
            ty = self.structs[cur.0 as usize].members[index].ty.clone();
        }
        let decl_ty = ty.clone();
        // Total offset from the base object: each anonymous level
        // adds its containing member's offset, which anonymous
        // splicing does NOT fold into the innermost entry alone.
        let mut offset: u64 = 0;
        let mut level_ty = Ty::Struct(id);
        let mut bitfield: Option<(u32, u32)> = None;
        for (level, &index) in path.iter().enumerate() {
            let Ty::Struct(cur) = level_ty.unqualified() else {
                unreachable!("member path crosses only structs")
            };
            let layout = self.struct_layout(*cur, interner).expect("member access on a complete type");
            offset += layout.member_offsets[index];
            let member = &self.structs[cur.0 as usize].members[index];
            if level + 1 == path.len() {
                if let Some(width_expr) = &member.bitfield {
                    let decl_size = self.type_size(&member.ty, interner).expect("bit-field has a declared type");
                    let w = self.bit_width(width_expr, decl_size * 8, interner).expect("bit-field width was checked to fit");
                    // The unit byte this field starts in, relative to
                    // the containing struct: `member_offsets` holds the
                    // byte of the unit start, and `members` the exact
                    // bit, keyed by the accessed name.
                    if let Some(ml) = layout.members.iter().find(|m| m.name == member.name.unwrap()) {
                        bitfield = Some((ml.bit.unwrap_or(0), w as u32));
                    }
                }
            }
            level_ty = member.ty.clone();
        }
        // C17 §6.3.1.1: a counter-intuitive bit-field's expression
        // type is its integral promotion, NOT its declared type —
        // `unsigned int f : 7` reads as `int`, so `i % bit.u` is a
        // signed divide. Past 32 bits no int type suffices, so the
        // declared type stands (`long l : 40` reads as `long`,
        // matching GCC's full-width value).
        if let Some((_, width)) = &bitfield {
            if *width <= 32 {
                ty = if ty.is_unsigned_in(self.target.char_signed) {
                    if *width >= 32 { Ty::UnsignedInt } else { Ty::Int }
                } else {
                    Ty::Int
                };
            }
        }
        // A member of a `const`-qualified object is itself `const`
        // (C17 §6.7.3p5): `const struct S s; s.x = 1;` fails, as does
        // `p->x = 1;` through a `const struct S *p`.
        if target_ty.is_const() {
            ty = Ty::qualified(vec![TypeQualifier::Const], ty);
        }
        Ok(CheckedExpr {
            ty,
            span,
            kind: CheckedExprKind::Member {
                base: Box::new(checked),
                path,
                arrow,
                offset,
                bitfield,
                decl_ty,
            },
        })
    }

    /// Resolve `field` to a member path, descending into anonymous
    /// struct/union members (C11 §6.7.2.1p13). `visiting` guards
    /// pathological recursive types.
    fn find_field(&self, id: StructId, field: SymbolU32, visiting: &mut Vec<StructId>) -> FieldHit {
        if visiting.contains(&id) {
            return FieldHit::Miss;
        }
        visiting.push(id);
        let result = self.find_field_inner(id, field, visiting);
        visiting.pop();
        result
    }

    fn find_field_inner(&self, id: StructId, field: SymbolU32, visiting: &mut Vec<StructId>) -> FieldHit {
        let def = &self.structs[id.0 as usize];
        if let Some(index) = def.members.iter().position(|m| m.name == Some(field)) {
            return FieldHit::Found(vec![index]);
        }
        let mut found: Option<Vec<usize>> = None;
        for (index, member) in def.members.iter().enumerate() {
            if member.name.is_some() {
                continue;
            }
            let Ty::Struct(inner) = member.ty.unqualified() else {
                continue;
            };
            if !self.structs.get(inner.0 as usize).is_some_and(|d| d.complete) {
                continue;
            }
            match self.find_field(*inner, field, visiting) {
                FieldHit::Miss => {}
                FieldHit::Ambiguous => return FieldHit::Ambiguous,
                FieldHit::Found(mut sub) => {
                    if found.is_some() {
                        return FieldHit::Ambiguous;
                    }
                    sub.insert(0, index);
                    found = Some(sub);
                }
            }
        }
        match found {
            Some(path) => FieldHit::Found(path),
            None => FieldHit::Miss,
        }
    }

    /// Check a `_Generic` selection (C11 §6.5.1.1): the controlling
    /// expression decays, then the first compatible type-name wins,
    /// else `default` (which must exist). Duplicate type-names and a
    /// second `default` are constraint violations. The result has the
    /// selected expression's type and is never an lvalue.
    fn check_generic(&self, ctrl: &Expr, assocs: &[GenericAssoc], interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let ctrl = self.decay_checked(self.check_expr(ctrl, interner)?);
        let mut checked: Vec<(Option<Ty>, CheckedExpr)> = Vec::with_capacity(assocs.len());
        let mut default = None;
        let mut selected = None;
        for assoc in assocs {
            let expr = self.check_expr(&assoc.expr, interner)?;
            match &assoc.ty {
                None => {
                    if default.replace(checked.len()).is_some() {
                        return Err(CompileError::new("duplicate `default` in `_Generic`", assoc.span));
                    }
                }
                Some(ty) => {
                    if checked.iter().any(|(t, _)| t.as_ref() == Some(ty)) {
                        return Err(CompileError::new("duplicate type in `_Generic`", assoc.span));
                    }
                    if selected.is_none() && types_compatible(&ctrl.ty, ty) {
                        selected = Some(checked.len());
                    }
                }
            }
            checked.push((assoc.ty.clone(), expr));
        }
        let index = selected.or(default).ok_or_else(|| CompileError::new("no matching generic association", span))?;
        let ty = checked[index].1.ty.clone();
        Ok(CheckedExpr {
            ty,
            span,
            kind: CheckedExprKind::Generic {
                ctrl: Box::new(ctrl),
                index,
                assocs: checked,
            },
        })
    }

    /// Byte offset of a `__builtin_offsetof(TYPE, .a.b[i])` designator
    /// (GCC's `offsetof`): each `.field` descends through the current
    /// struct or union (anonymous members included), each `[i]` needs
    /// an integer constant index into a member array. Bit-fields have
    /// no addressable offset and are rejected, like GCC.
    fn check_offsetof(&self, ty: &Ty, designators: &[Designator], interner: &Interner, span: Span) -> Result<u64, CompileError> {
        let Ty::Struct(_) = ty.unqualified() else {
            return Err(CompileError::new("`offsetof` needs a struct or union type", span));
        };
        if !self.is_complete(ty) {
            return Err(CompileError::new("`offsetof` of an incomplete type", span));
        }
        let mut cur = ty.clone();
        let mut offset: u64 = 0;
        let mut visiting = Vec::new();
        for designator in designators {
            match designator {
                Designator::Field { name, span } => {
                    let Ty::Struct(cid) = cur.unqualified() else {
                        return Err(CompileError::new("invalid `.field` designator in `offsetof`", *span));
                    };
                    let path = match self.find_field(*cid, *name, &mut visiting) {
                        FieldHit::Found(path) => path,
                        FieldHit::Miss => {
                            let text = interner.resolve(*name).unwrap_or("?");
                            return Err(CompileError::new(format!("no member named '{text}'"), *span));
                        }
                        FieldHit::Ambiguous => {
                            let text = interner.resolve(*name).unwrap_or("?");
                            return Err(CompileError::new(format!("ambiguous member '{text}' (found in several anonymous members)"), *span));
                        }
                    };
                    let mut at = *cid;
                    let mut final_is_bitfield = false;
                    for &index in &path {
                        let layout = self
                            .struct_layout(at, interner)
                            .ok_or_else(|| CompileError::new("cannot compute layout for `offsetof`", *span))?;
                        // `member_offsets` runs parallel to the
                        // definition's members (anonymous members and
                        // unnamed widths included), so the path index
                        // addresses it directly.
                        let off = layout
                            .member_offsets
                            .get(index)
                            .copied()
                            .ok_or_else(|| CompileError::new("cannot compute layout for `offsetof`", *span))?;
                        offset = offset.checked_add(off).ok_or_else(|| CompileError::new("cannot compute layout for `offsetof`", *span))?;
                        let member = &self.structs[at.0 as usize].members[index];
                        final_is_bitfield = member.bitfield.is_some();
                        cur = member.ty.clone();
                        if let Ty::Struct(next) = cur.unqualified() {
                            at = *next;
                        }
                    }
                    if final_is_bitfield {
                        return Err(CompileError::new("`offsetof` of bit-field", *span));
                    }
                }
                Designator::Index { index, span } => {
                    let Ty::Array { elem, .. } = cur.unqualified() else {
                        return Err(CompileError::new("invalid `[i]` designator in `offsetof`", *span));
                    };
                    if !is_integer(&index.ty) {
                        return Err(CompileError::new("array designator must have integer type", *span));
                    }
                    let n = match crate::eval::eval_const(index, self, interner) {
                        Some(crate::eval::Value::Int(n)) if n >= 0 => n as u64,
                        _ => {
                            return Err(CompileError::new("offsetof index is not an integer constant expression", *span));
                        }
                    };
                    let elem_size = self
                        .type_size(elem, interner)
                        .ok_or_else(|| CompileError::new("cannot compute layout for `offsetof`", *span))?;
                    offset = offset
                        .checked_add(n.checked_mul(elem_size).ok_or_else(|| CompileError::new("cannot compute layout for `offsetof`", *span))?)
                        .ok_or_else(|| CompileError::new("cannot compute layout for `offsetof`", *span))?;
                    cur = elem.as_ref().clone();
                }
                Designator::AnonField { span, .. } => {
                    return Err(CompileError::new("invalid `.field` designator in `offsetof`", *span));
                }
            }
        }
        Ok(offset)
    }

    fn check_cast(&self, ty: &Ty, operand: &Expr, interner: &Interner, span: Span) -> Result<CheckedExpr, CompileError> {
        let operand = self.decay_checked(self.check_expr(operand, interner)?);
        // `(void)e` discards the value: legal for any operand, including
        // a `void` one (EDG lowerings wrap `void` calls this way).
        if ty.unqualified() == &Ty::Void {
            return Ok(CheckedExpr {
                ty: ty.clone(),
                span,
                kind: CheckedExprKind::Cast {
                    ty: ty.clone(),
                    expr: Box::new(operand),
                },
            });
        }
        if is_scalar(&operand.ty) && is_scalar(ty) {
            // Complex converts only with arithmetic: neither side of a
            // complex/pointer cast is meaningful.
            let complex = is_complex(&operand.ty) || is_complex(ty);
            let pointer = matches!(operand.ty.unqualified(), Ty::Pointer { .. }) || matches!(ty.unqualified(), Ty::Pointer { .. });
            if complex && pointer {
                return Err(CompileError::new("invalid operands to cast", span));
            }
            return Ok(CheckedExpr {
                ty: ty.clone(),
                span,
                kind: CheckedExprKind::Cast {
                    ty: ty.clone(),
                    expr: Box::new(operand),
                },
            });
        }
        // A cast to a compatible struct/union type is a no-op, like
        // GCC (`(struct S)s`, qualified or not); the value's
        // representation is unchanged.
        if matches!(ty.unqualified(), Ty::Struct(_)) && types_compatible(ty.unqualified(), operand.ty.unqualified()) {
            return Ok(CheckedExpr {
                ty: ty.clone(),
                span,
                kind: CheckedExprKind::Cast {
                    ty: ty.clone(),
                    expr: Box::new(operand),
                },
            });
        }
        // A cast to a union type is a GNU extension ("cast to a union
        // type"): the operand's type must match one of the members, and
        // the value initializes that member. Desugar to a compound
        // literal so lowering reuses the initializer path.
        if let Ty::Struct(id) = ty.unqualified() {
            let def = &self.structs[id.0 as usize];
            if def.is_union {
                let matched = def.members.iter().find_map(|member| {
                    let name = member.name?;
                    let compatible = types_compatible(member.ty.unqualified(), operand.ty.unqualified()) || types_compatible(operand.ty.unqualified(), member.ty.unqualified());
                    compatible.then_some(name)
                });
                let Some(name) = matched else {
                    return Err(CompileError::new("cast to union type from type not present in union", span));
                };
                return Ok(CheckedExpr {
                    ty: ty.clone(),
                    span,
                    kind: CheckedExprKind::CompoundLiteral {
                        ty: ty.clone(),
                        init: Box::new(Initializer::List {
                            items: vec![InitItem {
                                designators: vec![Designator::Field { name, span }],
                                init: Box::new(Initializer::Expr(operand)),
                                span,
                            }],
                            span,
                        }),
                    },
                });
            }
        }
        Err(CompileError::new("invalid operands to cast", span))
    }
}
