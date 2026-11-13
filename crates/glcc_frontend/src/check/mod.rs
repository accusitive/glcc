//! Expression checking: annotate `parse::Expr<'a>` with types.
//!
//! [`Sema::check_expr`] elaborates a parsed expression into a
//! [`CheckedExpr<'a>`], inserting [`CheckedExprKind::Convert`] nodes wherever
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
    diag::{DiagCode, did_you_mean},
    lexer::{CharPrefix, FloatSuffix, Interner},
    parse::{ArrayBound, AssignOp, BinaryOp, BlockItem, Designator, Expr, GenericAssoc, InitItem, Initializer, PtrSize, StdAttributes, Ty, TypeQualifiers, UnaryOp},
    sema::{Binding, BindingKind, DeclId, Member, Sema, StructId, types_compatible},
    span::{CompileError, HasSpan, Span},
    target::{Arch, Os},
};

mod atomic;
mod gnu;
mod ia32;
mod msvc;
mod prototypes;
mod sync;

pub use atomic::{AtomicOp, RmwOp};
pub use gnu::{FpPredicate, OverflowOp};
pub use ia32::{ComiPred, CmpPred, CvtfTy, Ia32Op, PsBitOp, SsOp, Vec128BinOp, Vec128PackKind, Vec128ShufMode, Vec128ShiftDir};
pub(crate) use prototypes::builtin_prototype;
use gnu::typed_overflow_builtin;

/// An expression annotated with its type. Conversion nodes make implicit
/// conversions explicit so later phases never re-derive them.
//
// Note: derived equality includes spans, so only span-erased trees (see
// `erased`) may be compared — in practice just array bounds in `Ty`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckedExpr<'a> {
    pub ty: Ty,
    pub span: Span,
    pub kind: CheckedExprKind<'a>,
}

impl HasSpan for CheckedExpr<'_> {
    fn span(&self) -> Span {
        self.span
    }
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
pub enum CheckedExprKind<'a> {
    /// A fully resolved variable or function: just its binding.
    Ident(crate::sema::DeclId),
    Int(crate::lexer::IntLiteral),
    Float(crate::lexer::FloatLiteral),
    Char(crate::lexer::CharLiteral),
    Str(&'a [crate::lexer::StringLiteral]),
    Unary(UnaryOp, &'a CheckedExpr<'a>),
    Binary(BinaryOp, &'a CheckedExpr<'a>, &'a CheckedExpr<'a>),
    Assign(AssignOp, &'a CheckedExpr<'a>, &'a CheckedExpr<'a>),
    /// `lhs op= rhs` with `op` other than `=`: the lvalue evaluates
    /// once (its address), its current value converts to `op_ty`,
    /// combines with the `rhs` (converted to `rhs_ty`), and the result
    /// converts back to the lvalue's type (C17 §6.5.16.2). Keeping the
    /// operands separate is what allows the single evaluation:
    /// rewriting to `lhs = lhs op rhs` would re-run the lvalue's side
    /// effects.
    CompoundAssign {
        op: AssignOp,
        lhs: &'a CheckedExpr<'a>,
        rhs: &'a CheckedExpr<'a>,
        /// Boxed to keep the variant small: unboxed types would make
        /// it the largest variant and grow every checker frame.
        op_ty: Box<Ty>,
        rhs_ty: Box<Ty>,
    },
    Ternary {
        cond: &'a CheckedExpr<'a>,
        then: &'a CheckedExpr<'a>,
        els: &'a CheckedExpr<'a>,
    },
    /// GNU `x ?: y`: the condition's value is the result when nonzero
    /// (evaluated once — the lowering keeps it in a slot), `els`
    /// otherwise.
    Elvis {
        cond: &'a CheckedExpr<'a>,
        els: &'a CheckedExpr<'a>,
    },
    Call {
        func: &'a CheckedExpr<'a>,
        args: &'a [CheckedExpr<'a>],
    },
    /// `__builtin_await(handle)` (glcc extension): suspend until the
    /// awaited future completes, yielding its result. The operand is a
    /// value of some `_Future(T)` handle type; this node's type is `T`.
    Await(&'a CheckedExpr<'a>),
    /// A GCC atomic builtin call (`__atomic_load_n`, `__atomic_store_n`):
    /// the operation plus all checked operands (pointer first), since
    /// no declared function stands behind it.
    Atomic {
        op: AtomicOp,
        args: &'a [CheckedExpr<'a>],
    },
    /// A GCC x86 `__builtin_ia32_*` builtin (see `ia32::Ia32Op`).
    Ia32 {
        op: Ia32Op,
        args: &'a [CheckedExpr<'a>],
    },
    /// A GCC overflow builtin (`__builtin_mul_overflow` and friends):
    /// the operation, both operands, and the pointer receiving the
    /// wrapped result. The pointed-to type fixes the result's width
    /// and signedness; the operands keep their own types.
    Overflow {
        op: OverflowOp,
        lhs: &'a CheckedExpr<'a>,
        rhs: &'a CheckedExpr<'a>,
        res: &'a CheckedExpr<'a>,
        /// The pointed-to type the result converts to.
        res_ty: Ty,
    },
    /// `alloca`/`__builtin_alloca`: the requested byte count (already
    /// converted to `size_t`). The result is `void *`.
    Alloca {
        size: &'a CheckedExpr<'a>,
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
        arg: &'a CheckedExpr<'a>,
        fallback: Option<&'a CheckedExpr<'a>>,
    },
    /// A byte-order reversal (`__builtin_bswap16/32/64`): the operand
    /// is already converted to the prototype's parameter type, and
    /// `bits` is that width. Constant operands fold in `eval_const`.
    Bswap {
        arg: &'a CheckedExpr<'a>,
        bits: u32,
    },
    /// A type-generic C23/C2y rotate
    /// (`__builtin_stdc_rotate_left`/`_right`): the value keeps its
    /// declared unsigned width, and `bits` is that width (a bit-field's
    /// declared width or a `_BitInt`'s precision). The count keeps its
    /// own integer type so a wide count is reduced modulo `bits` without
    /// truncation.
    Rotate {
        left: bool,
        arg: &'a CheckedExpr<'a>,
        count: &'a CheckedExpr<'a>,
        bits: u32,
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
        arg: &'a CheckedExpr<'a>,
    },
    /// `__builtin_fpclassify(nan, inf, normal, subnormal, zero, x)`: the
    /// `int` code for `x`'s category, supplied by the caller (the
    /// `<math.h>` macro passes `FP_NAN`...`FP_ZERO`). Single evaluation
    /// of `x`; the codes evaluate in order (they are constants in
    /// practice).
    FpClassify {
        codes: [&'a CheckedExpr<'a>; 5],
        arg: &'a CheckedExpr<'a>,
    },
    Subscript {
        base: &'a CheckedExpr<'a>,
        index: &'a CheckedExpr<'a>,
    },
    /// GNU `__builtin_shuffle(v, mask)` / `(v0, v1, mask)`: lane `i`
    /// of the result is lane `mask[i]` of `v` (indices below the lane
    /// count) or of `v1` (above it). The last argument is the mask; it
    /// is an integer vector whose lane width matches the operands'.
    Shuffle {
        args: &'a [CheckedExpr<'a>],
    },
    /// Clang `__builtin_shufflevector(v0[, v1], idx...)`: lane `i` of
    /// the result is lane `idx[i]` of the concatenated operands;
    /// `-1` and out-of-range indices are undef, like clang's poison.
    /// The index list is variadic and constant, so the operands alone
    /// are carried (one or two vectors).
    ShuffleVector {
        args: &'a [CheckedExpr<'a>],
        indices: &'a [i128],
    },
    Member {
        base: &'a CheckedExpr<'a>,
        /// Path of member indices from the base struct: a direct field
        /// is one element; fields promoting through anonymous
        /// struct/union members append one index per level crossed.
        /// Codegen maps this through layout to an offset.
        path: &'a [usize],
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
        expr: &'a CheckedExpr<'a>,
    },
    Generic {
        ctrl: &'a CheckedExpr<'a>,
        /// Index of the selected association in `assocs`.
        index: usize,
        assocs: &'a [(Option<Ty>, CheckedExpr<'a>)],
    },
    /// `_Generic` whose controlling operand is a type-name (C2y).
    GenericTy {
        ty: Ty,
        index: usize,
        assocs: &'a [(Option<Ty>, CheckedExpr<'a>)],
    },
    /// C2y static assertion in expression position: type void, no
    /// runtime effect.
    StaticAssert,
    VaArg {
        ap: &'a CheckedExpr<'a>,
        ty: Ty,
    },
    StmtExpr {
        items: &'a [BlockItem<'a>],
    },
    /// A folded `offsetof`: the byte offset with the struct type kept
    /// for fidelity.
    Offsetof {
        ty: Ty,
        offset: u64,
    },
    CompoundLiteral {
        ty: Ty,
        init: &'a Initializer<'a>,
        storage: crate::parse::CompoundStorage,
    },
    Convert {
        expr: &'a CheckedExpr<'a>,
        ty: Ty,
    },
    SizeofExpr(&'a CheckedExpr<'a>),
    SizeofTy(Ty),
    CountofExpr(&'a CheckedExpr<'a>),
    CountofTy(Ty),
    AlignofExpr(&'a CheckedExpr<'a>),
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
        Ty::BitInt { width, .. } => match width {
            0..=8 => 1,
            9..=16 => 2,
            _ => 3,
        },
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
        Ty::BitInt { signed, .. } => *signed,
        _ => return None,
    })
}

/// The bit-field precision of an expression, if any: gcc performs an
/// operation on bit-fields in the precision of the widest operand, so
/// `unsigned long long b : 40` keeps 40 bits through `b << 32` and
/// `b.u33 + b.u33` wraps at 33. A non-bit-field operand contributes
/// the width of its (promoted) type: `b - 8` wraps at 40 bits because
/// `int` is narrower, while `b - 8ull` computes at 64. Only fields
/// wider than 32 bits carry a reduced precision — a narrower field
/// promotes to `int`, whose 32 bits are already the common width.
/// Shifts take the left operand's.
pub fn bitfield_precision<'a>(expr: &CheckedExpr<'a>, target: &crate::target::Target) -> Option<u32> {
    use crate::parse::{BinaryOp, UnaryOp};
    match &expr.kind {
        CheckedExprKind::Member { bitfield: Some((_, width)), .. } if *width > 32 => Some(*width),
        CheckedExprKind::Unary(op, operand) => match op {
            UnaryOp::Neg | UnaryOp::BitNot | UnaryOp::PreInc | UnaryOp::PreDec | UnaryOp::PostInc | UnaryOp::PostDec => bitfield_precision(operand, target),
            _ => None,
        },
        CheckedExprKind::Binary(op, lhs, rhs) => match op {
            BinaryOp::Shl | BinaryOp::Shr => bitfield_precision(lhs, target),
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem | BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor => {
                Some(operand_precision(lhs, target)?.max(operand_precision(rhs, target)?))
            }
            _ => None,
        },
        CheckedExprKind::Assign(_, lhs, _) => bitfield_precision(lhs, target),
        CheckedExprKind::CompoundAssign { lhs, .. } => bitfield_precision(lhs, target),
        // The usual arithmetic conversions wrap operands in `Convert`
        // nodes; the reduced precision rides along (an explicit `Cast`
        // is a separate node and resets it).
        CheckedExprKind::Convert { expr, .. } => bitfield_precision(expr, target),
        _ => None,
    }
}

/// An integer operand's arithmetic precision: a bit-field's declared
/// width, or the width of its type after the integer promotions. The
/// usual arithmetic conversions wrap operands in `Convert` nodes;
/// those do not widen the contribution, but a conversion that *is* an
/// integer promotion does (a `char` operand counts as `int`).
fn operand_precision<'a>(expr: &CheckedExpr<'a>, target: &crate::target::Target) -> Option<u32> {
    let int_rank = |ty: &Ty| int_rank(ty).unwrap_or(u8::MAX);
    let mut cur = expr;
    let mut promoted = false;
    while let CheckedExprKind::Convert { expr: inner, ty } = &cur.kind {
        if matches!(ty.unqualified(), Ty::Int | Ty::UnsignedInt) && int_rank(&inner.ty) < int_rank(&Ty::Int) {
            promoted = true;
        }
        cur = inner;
    }
    if !is_integer(&cur.ty) {
        return None;
    }
    if let Some(bits) = bitfield_precision(cur, target) {
        return Some(bits);
    }
    if promoted {
        return int_type_size(&Ty::Int, target).map(|bytes| bytes * 8);
    }
    int_type_size(&cur.ty, target).map(|bytes| bytes * 8)
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
        Ty::BitInt { width, .. } => crate::layout::bitint_size(*width),
        Ty::Enum(_) => target.enum_size,
        _ => return None,
    };
    Some(size as u32)
}

pub fn is_integer(ty: &Ty) -> bool {
    int_rank(ty).is_some()
}

/// Whether `ty` is an unsigned integer type suitable for a rotate:
/// standard unsigned types and unsigned bit-precise types, excluding
/// `_Bool` (GCC rejects its boolean type).
fn is_unsigned_integer(ty: &Ty) -> bool {
    matches!(
        ty.unqualified(),
        Ty::UnsignedChar | Ty::UnsignedShort | Ty::UnsignedInt | Ty::UnsignedLong | Ty::UnsignedLongLong | Ty::UnsignedInt128
    ) || matches!(ty.unqualified(), Ty::BitInt { signed: false, .. })
}

fn is_real_float(ty: &Ty) -> bool {
    matches!(ty.unqualified(), Ty::Float | Ty::Double | Ty::LongDouble | Ty::Float128)
}

pub fn is_complex(ty: &Ty) -> bool {
    matches!(ty.unqualified(), Ty::Complex(_))
}

/// Whether `ty` is a GNU vector type (`vector_size`).
pub fn is_vector(ty: &Ty) -> bool {
    matches!(ty.unqualified(), Ty::Vector { .. })
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
    is_arithmetic(ty) || matches!(ty.unqualified(), Ty::Pointer { .. } | Ty::Nullptr)
}

fn is_void_ptr(ty: &Ty) -> bool {
    matches!(ty.unqualified(), Ty::Pointer { pointee, .. } if pointee.unqualified() == &Ty::Void)
}

/// Qualifiers on what a pointer points to (`const` in `const int *`);
/// empty for unqualified pointees. Only the outermost pointee level
/// matters for the `void *` composite below.
fn pointer_pointee_quals(ty: &Ty) -> TypeQualifiers {
    let mut pointee = ty;
    while let Ty::Qualified { inner, .. } = pointee {
        pointee = inner;
    }
    if let Ty::Pointer { pointee, .. } = pointee {
        match &**pointee {
            Ty::Qualified { quals, .. } => *quals,
            _ => TypeQualifiers::empty(),
        }
    } else {
        TypeQualifiers::empty()
    }
}

/// Composite of a `void *`-involved ternary (C17 §6.5.15): `void *`
/// with the union of both pointees' qualifiers, so
/// `cond ? void_ptr : const_int_ptr` is `const void *` like GCC.
/// Outer pointer qualifiers drop (rvalues), matching GCC.
fn void_ptr_composite(a: &Ty, b: &Ty) -> Ty {
    let quals = pointer_pointee_quals(a) | pointer_pointee_quals(b);
    Ty::Pointer {
        qualifiers: TypeQualifiers::empty(),
        size: PtrSize::Default,
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
        Ty::BitInt { width, .. } => Ty::BitInt { width: *width, signed: false },
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
pub fn is_lvalue<'a>(expr: &CheckedExpr<'a>, decls: &[Binding]) -> bool {
    match &expr.kind {
        CheckedExprKind::Ident(id) => decls
            .get(id.0 as usize)
            .is_some_and(|binding| !matches!(binding.kind, BindingKind::Func | BindingKind::EnumConst { .. })),
        CheckedExprKind::Unary(crate::parse::UnaryOp::Deref, _) => true,
        // A complex part of an lvalue is itself an lvalue
        // (`__real__ x = 1.0`), like a struct member of one.
        CheckedExprKind::Unary(crate::parse::UnaryOp::Real, inner) | CheckedExprKind::Unary(crate::parse::UnaryOp::Imag, inner) => is_lvalue(inner, decls),
        // Array/pointer subscripts are lvalues, and so are vector
        // lanes (clang's intrinsic headers assign `__a[0] += __b[0]`;
        // GCC keeps them rvalues, but the lvalue reading is a strict
        // superset of the programs it accepts).
        CheckedExprKind::Subscript { .. } => true,
        // Compound<'a> literals are lvalues (C17 §6.5.2.5p5).
        CheckedExprKind::CompoundLiteral { .. } => true,
        CheckedExprKind::Member { base, arrow, .. } => *arrow || is_lvalue(base, decls),
        CheckedExprKind::Convert { expr, .. } => is_lvalue(expr, decls),
        _ => false,
    }
}

/// Whether `expr` designates a bit-field: a `Member` whose final field
/// carries a width. Conversions wrap the value without changing what it
/// designates, so they look through (like [`is_lvalue`]).
pub fn designates_bitfield<'a>(expr: &CheckedExpr<'a>) -> bool {
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

impl<'a> Sema<'a> {
    // ---- usual arithmetic conversions (C17 §6.3.1.8) ----

    /// Promote a sub-`int` integer operand to `int` (or to `unsigned int`
    /// when `int` cannot represent every value — unreachable with
    /// the fixed widths of every supported target, kept for
    /// narrower-`int` targets). Non-integers
    /// (floats, pointers, structs) pass through untouched. Enums
    /// promote to their representation: plain `int` for ordinary
    /// enums, wider for sentinel values like `DW_CIE_ID_32`.
    pub fn integer_promote(&self, expr: CheckedExpr<'a>) -> CheckedExpr<'a> {
        // Complex over a sub-`int` component promotes component-wise
        // (`__complex__ char` computes as `__complex__ int`), mirroring
        // the scalar rule below.
        if let Ty::Complex(inner) = expr.ty.unqualified() {
            if let Some(rank) = int_rank(inner)
                && rank < int_rank(&Ty::Int).expect("Int has a rank") {
                    let int_size = int_type_size(&Ty::Int, &self.target).expect("Int has a size");
                    let target = if int_type_size(inner, &self.target).expect("rank implies size") < int_size {
                        Ty::Int
                    } else {
                        Ty::UnsignedInt
                    };
                    return self.convert(expr, Ty::Complex(Box::new(target)));
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
        // C23 bit-precise integers do not undergo the integer
        // promotions; their width is already known and UAC compares it
        // directly against the standard operand.
        if matches!(expr.ty.unqualified(), Ty::BitInt { .. }) {
            return expr;
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
    pub fn convert(&self, expr: CheckedExpr<'a>, ty: Ty) -> CheckedExpr<'a> {
        if expr.ty == ty {
            return expr;
        }
        let span = expr.span;
        CheckedExpr {
            ty: ty.clone(),
            span,
            kind: CheckedExprKind::Convert { expr: self.alloc(expr), ty },
        }
    }

    /// Default argument promotions for unprototyped/variadic call
    /// arguments: decay plus integer promotions plus `float` → `double`.
    fn default_arg_promote(&self, expr: CheckedExpr<'a>) -> CheckedExpr<'a> {
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
        let left_bitint = matches!(left.unqualified(), Ty::BitInt { .. });
        let right_bitint = matches!(right.unqualified(), Ty::BitInt { .. });
        if left_bitint || right_bitint {
            // Standard operands have already been promoted, so these are
            // int/unsigned/long/... widths. A bit-precise operand keeps
            // its declared precision.
            let precision = |ty: &Ty| match ty.unqualified() {
                Ty::BitInt { width, .. } => u64::from(*width),
                other => u64::from(int_type_size(other, target).expect("rank implies size") * 8),
            };
            let (left_bits, right_bits) = (precision(left), precision(right));
            if left_bits != right_bits {
                return if left_bits > right_bits { left.clone() } else { right.clone() };
            }
            // Equal precision: a standard type outranks a bit-precise
            // one, and an unsigned type wins over a signed one.
            if left_signed == right_signed {
                return if left_bitint && !right_bitint { right.clone() } else { left.clone() };
            }
            return match (left_bitint, right_bitint) {
                (true, false) if right_signed => corresponding_unsigned(right),
                (true, false) => right.clone(),
                (false, true) if left_signed => corresponding_unsigned(left),
                (false, true) => left.clone(),
                _ => {
                    if left_signed {
                        corresponding_unsigned(left)
                    } else {
                        left.clone()
                    }
                }
            };
        }
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
        // For bit-precise integers the value width, not the padded storage
        // width, decides representability.
        let precision = |ty: &Ty| match ty.unqualified() {
            Ty::BitInt { width, .. } => u64::from(*width),
            other => u64::from(int_type_size(other, target).expect("rank implies size") * 8),
        };
        let left_size = precision(left);
        let right_size = precision(right);
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
    pub fn usual_arithmetic_conversions(&self, left: CheckedExpr<'a>, right: CheckedExpr<'a>) -> (CheckedExpr<'a>, CheckedExpr<'a>, Ty) {
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
    fn decay_checked(&self, expr: CheckedExpr<'a>) -> CheckedExpr<'a> {
        let CheckedExpr { ty, span, kind } = expr;
        let expr = CheckedExpr {
            ty: ty.unqualified().clone(),
            span,
            kind,
        };
        let ty = match &expr.ty {
            Ty::Array { elem, quals, .. } => Ty::Pointer {
                qualifiers: *quals,
                size: PtrSize::Default,
                pointee: elem.clone(),
            },
            Ty::Func { .. } => Ty::Pointer {
                qualifiers: TypeQualifiers::empty(),
                size: PtrSize::Default,
                pointee: Box::new(self.as_callable_ty(expr.ty.clone(), &expr.kind)),
            },
            _ => return expr,
        };
        let CheckedExpr { span, kind, .. } = expr;
        // A `_Generic`'s value is its selected arm's value: decay that
        // arm alongside the node so both carry one type.
        let kind = match kind {
            CheckedExprKind::Generic { ctrl, index, assocs } => {
                let assocs = self.decay_generic_assoc(assocs, index);
                CheckedExprKind::Generic { ctrl, index, assocs }
            }
            CheckedExprKind::GenericTy { ty, index, assocs } => {
                let assocs = self.decay_generic_assoc(assocs, index);
                CheckedExprKind::GenericTy { ty, index, assocs }
            }
            kind => kind,
        };
        CheckedExpr { ty, span, kind }
    }

    /// Rebuild a `_Generic` association slice with the selected arm
    /// decayed alongside the node (arena slices are immutable).
    fn decay_generic_assoc(&self, assocs: &'a [(Option<Ty>, CheckedExpr<'a>)], index: usize) -> &'a [(Option<Ty>, CheckedExpr<'a>)] {
        let mut rebuilt = assocs.to_vec();
        if let Some((_, arm)) = rebuilt.get_mut(index) {
            *arm = self.decay_checked(arm.clone());
        }
        self.alloc_slice(rebuilt)
    }

    /// C23 `auto x = init`: the deduced type is the initializer's type
    /// after lvalue conversion (arrays and functions decay).
    pub(crate) fn auto_deduced_type(&self, init: &CheckedExpr<'a>) -> Ty {
        self.decay_checked(init.clone()).ty
    }

    /// Whether a checked expression may appear left of `=` or under `&`:
    /// variables (never functions, typedefs, or enumeration constants),
    /// dereferences, subscripts, and members of lvalues (conversions look
    /// through to their operand).
    fn is_lvalue(&self, expr: &CheckedExpr<'a>) -> bool {
        is_lvalue(expr, &self.decls)
    }

    /// Whether `expr` designates a bit-field: a `Member` whose final
    /// field carries a width. Conversions wrap the value without
    /// changing what it designates, so they look through (like
    /// `is_lvalue`).
    fn designates_bitfield(expr: &CheckedExpr<'a>) -> bool {
        designates_bitfield(expr)
    }

    /// Record the declaration whose address `&expr` takes: the `Ident`
    /// at the base of the operand's lvalue (`&x` marks `x`, `&s.f`
    /// and `&a[i]` mark `s`/`a`). `&*p` marks nothing (it is just
    /// `p`), and compound literals have no declaration to mark.
    /// Callers check validity first; non-lvalue shapes are ignored.
    fn note_address_taken(&self, expr: &CheckedExpr<'a>) {
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
    fn is_null_pointer_constant(&self, expr: &CheckedExpr<'a>, interner: &Interner) -> bool {
        if matches!(expr.ty.unqualified(), Ty::Nullptr) {
            return true;
        }
        if matches!(&expr.ty, Ty::Pointer { pointee, .. } if pointee.as_ref() == &Ty::Void)
            && let CheckedExprKind::Cast { expr, .. } | CheckedExprKind::Convert { expr, .. } = &expr.kind {
                return self.is_null_pointer_constant(expr, interner);
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
    fn convert_for_assignment(&self, target: &Ty, expr: CheckedExpr<'a>, interner: &Interner) -> Result<CheckedExpr<'a>, CompileError> {
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
            Ty::Pointer { pointee, size, .. } => {
                let (span, kind) = (expr.span, expr.kind);
                CheckedExpr {
                    ty: Ty::Pointer {
                        qualifiers: TypeQualifiers::empty(),
                        size,
                        pointee,
                    },
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
        // Same-shaped GNU vector, ignoring top-level qualifiers: like
        // the struct case, a value copy with no `Convert` lowering for
        // the aggregate shape (the MIR value moves bit-exact).
        if matches!(expr.ty.unqualified(), Ty::Vector { .. }) && expr.ty.unqualified() == target.unqualified() {
            let CheckedExpr { span, kind, .. } = expr;
            return Ok(CheckedExpr { ty: target.clone(), span, kind });
        }
        // Same-size GNU vectors of different spellings retype too
        // (GCC: vectors of the same overall size assign; clang's
        // intrinsic headers return `__v4si`-class values as
        // `__m128i`).
        if matches!(expr.ty.unqualified(), Ty::Vector { .. })
            && matches!(target.unqualified(), Ty::Vector { .. })
            && self.type_size(&expr.ty, interner) == self.type_size(target, interner)
        {
            let CheckedExpr { span, kind, .. } = expr;
            return Ok(CheckedExpr { ty: target.clone(), span, kind });
        }
        let span = expr.span;
        if matches!(target.unqualified(), Ty::Nullptr) {
            let from_same = matches!(expr.ty.unqualified(), Ty::Nullptr);
            let from_null = from_same
                || (is_integer(&expr.ty) && matches!(crate::eval::eval_const(&expr, self, interner), Some(crate::eval::Value::Int(0))))
                || (is_void_ptr(&expr.ty) && self.is_null_pointer_constant(&expr, interner));
            if !from_null {
                return Err(CompileError::new("conversion to `nullptr_t` requires `nullptr` or a null pointer constant", span));
            }
            if from_same {
                let CheckedExpr { span, kind, .. } = expr;
                return Ok(CheckedExpr { ty: target.clone(), span, kind });
            }
            return Ok(self.convert(expr, target.clone()));
        }
        if matches!(expr.ty.unqualified(), Ty::Nullptr) {
            if matches!(target.unqualified(), Ty::Pointer { .. } | Ty::Bool) {
                return Ok(self.convert(expr, target.clone()));
            }
            return Err(CompileError::new("cannot convert `nullptr_t` to a non-pointer type", span));
        }
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
        let (Ty::Pointer { qualifiers: qt, pointee: pt, .. }, Ty::Pointer { qualifiers: qs, pointee: ps, .. }) = (target, source) else {
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
            (Ty::Pointer { qualifiers: qt, pointee: pt, .. }, Ty::Pointer { qualifiers: qs, pointee: ps, .. }) => {
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
    fn readonly_error(&self, expr: &CheckedExpr<'a>, interner: &Interner) -> CompileError {
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
    pub fn check_condition(&self, expr: &Expr<'a>, interner: &Interner) -> Result<CheckedExpr<'a>, CompileError> {
        let checked = self.decay_checked(self.check_expr(expr, interner)?);
        if !is_scalar(&checked.ty) {
            return Err(CompileError::new("condition must have scalar type", checked.span));
        }
        Ok(checked)
    }

    /// Check an expression that must have integer type: `switch`
    /// controlling expressions and `case` labels (C17 §6.8.1p1,
    /// §6.8.4.2). `what` names the position for the message.
    pub fn check_integer_expr(&self, expr: &Expr<'a>, interner: &Interner, what: &str) -> Result<CheckedExpr<'a>, CompileError> {
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
    pub fn check_initializer(&self, target: &Ty, init: &Initializer<'a>, interner: &Interner) -> Result<Initializer<'a>, CompileError> {
        match init {
            // Values arrive already checked (parsers check inline, where
            // scopes are live); here they only convert to the target.
            Initializer::Expr(expr) => {
                // `char s[] = "abc";` — a string literal initializes an
                // array of its own character type directly (C17
                // §6.7.9p14), with or without braces (the braced form
                // lands in `check_array_init` instead).
                if let Ty::Array { elem, bound, .. } = target.unqualified()
                    && matches!(expr.ty.unqualified(), Ty::Array { .. })
                        && let CheckedExprKind::Str(parts) = &expr.kind {
                            let lit_elem = self.string_elem_ty(parts);
                            if lit_elem.is_some_and(|lit| Self::string_elem_matches(elem, &lit)) {
                                // Same overlong-literal warning the
                                // braced path gets; only the fit check
                                // differs.
                                self.check_string_fit(parts, bound, expr.span(), interner);
                                return Ok(Initializer::Expr(expr.clone()));
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
                        designators: &[],
                        init: self.alloc(Initializer::Expr(expr.clone())),
                        span: expr.span(),
                    }];
                    let span = expr.span();
                    return self.check_initializer(
                        target,
                        &Initializer::List {
                            items: self.alloc_slice(single),
                            span,
                        },
                        interner,
                    );
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
                if let Ty::Array { elem, .. } = target.unqualified()
                    && let [item] = items.as_slice()
                        && item.designators.is_empty()
                            && let Initializer::Expr(expr) = item.init
                                && let CheckedExprKind::Str(parts) = &expr.kind
                                    && self.string_elem_ty(parts).is_some_and(|lit| Self::string_elem_matches(elem, &lit)) {
                                        return Ok(Initializer::Expr(expr.clone()));
                                    }
                Ok(Initializer::List {
                    items: self.alloc_slice(items),
                    span: *span,
                })
            }
        }
    }

    /// Check a brace-enclosed list against `target`, resolving each item
    /// to the member or element type its designators (or position) select.
    fn check_init_list(&self, target: &Ty, items: &[InitItem<'a>], interner: &Interner) -> Result<Vec<InitItem<'a>>, CompileError> {
        // Qualification on the object itself never changes the
        // initializer shape (`const struct S s = {...}` initializes
        // like `struct S`).
        match target.unqualified() {
            Ty::Struct(id) => self.check_struct_init(*id, items, interner),
            Ty::Array { .. } => self.check_array_init(target.unqualified(), items, interner),
            // GNU vector initializer: at most one scalar item per lane
            // in order (GCC zero-fills the missing tail); designators
            // have no vector form.
            Ty::Vector { elem, len } => {
                let len = *len as usize;
                if items.len() > len {
                    return Err(CompileError::new("excess elements in vector initializer", items[len].span));
                }
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    if !item.designators.is_empty() {
                        return Err(CompileError::new("designator in vector initializer", item.span));
                    }
                    let checked = self.check_initializer(elem, item.init, interner)?;
                    out.push(InitItem {
                        designators: &[],
                        init: self.alloc(checked),
                        span: item.span,
                    });
                }
                Ok(out)
            }
            _ if is_scalar(target) => {
                // Redundant braces around a scalar are legal (`int x =
                // {{1}}`), but only for a single bare item.
                let [item] = items else {
                    return Err(CompileError::new(
                        "excess elements in scalar initializer",
                        items.get(1).map(|item| item.span).unwrap_or_default(),
                    ));
                };
                if !item.designators.is_empty() {
                    return Err(CompileError::new("designator in scalar initializer", item.span));
                }
                let checked = self.check_initializer(target, item.init, interner)?;
                Ok(vec![InitItem {
                    designators: &[],
                    init: self.alloc(checked),
                    span: item.span,
                }])
            }
            _ => Err(CompileError::new(
                "invalid initializer for type",
                items.first().map(|item| item.span).unwrap_or_default(),
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
    fn check_struct_init(&self, id: StructId, items: &[InitItem<'a>], interner: &Interner) -> Result<Vec<InitItem<'a>>, CompileError> {
        let def = &self.structs[id.0 as usize];
        if !def.complete {
            return Err(CompileError::new(
                "initializer for incomplete type",
                items.first().map(|item| item.span).unwrap_or_default(),
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
                if matches!(item.init, Initializer::List { .. }) {
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
                if self.needs_elision(&member_ty, item.init) {
                    let j = self.elision_span(&member_ty, items, i);
                    if j > i {
                        let sub = items[i..j].to_vec();
                        let span = sub.first().map(|it| it.span).unwrap_or(item.span).merge(sub.last().map(|it| it.span).unwrap_or(item.span));
                        let init = self.check_initializer(
                            &member_ty,
                            &Initializer::List {
                                items: self.alloc_slice(sub),
                                span,
                            },
                            interner,
                        )?;
                        let designators = self.alloc_slice(self.cursor_designators(id, &cursor, item.span));
                        self.advance_cursor(id, &mut cursor);
                        checked.push(InitItem {
                            designators,
                            init: self.alloc(init),
                            span: item.span,
                        });
                        i = j;
                        continue;
                    }
                }
                let init = self.check_initializer(&member_ty, item.init, interner)?;
                let designators = self.alloc_slice(self.cursor_designators(id, &cursor, item.span));
                self.advance_cursor(id, &mut cursor);
                checked.push(InitItem {
                    designators,
                    init: self.alloc(init),
                    span: item.span,
                });
                i += 1;
            } else {
                let (target_ty, outer, expanded) = self.resolve_designation(id, item.designators, interner)?;
                let init = self.check_initializer(&target_ty, item.init, interner)?;
                cursor = vec![outer + 1];
                self.normalize_cursor(id, &mut cursor);
                checked.push(InitItem {
                    designators: self.alloc_slice(expanded),
                    init: self.alloc(init),
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
    fn needs_elision(&self, member_ty: &Ty, init: &Initializer<'a>) -> bool {
        let Initializer::Expr(expr) = init else {
            return false;
        };
        match member_ty.unqualified() {
            Ty::Array { elem, .. } => {
                if let CheckedExprKind::Str(parts) = &expr.kind
                    && let Some(lit_elem) = self.string_elem_ty(parts) {
                        return !Self::string_elem_matches(elem, &lit_elem);
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
    fn elision_span(&self, ty: &Ty, items: &[InitItem<'a>], i: usize) -> usize {
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
    fn bare_item(items: &[InitItem<'a>], j: usize) -> bool {
        items.get(j).is_some_and(|item| item.designators.is_empty() && matches!(item.init, Initializer::Expr(_)))
    }

    /// Element type of a string literal under the selected dialect
    /// (C23 makes UTF-8 literals arrays of `unsigned char`).
    fn string_elem_ty(&self, parts: &[crate::lexer::StringLiteral]) -> Option<Ty> {
        crate::eval::string_elem_ty_std(parts, self.target.wchar(), self.lang.standard_includes(crate::lang::Std::C23))
    }

    /// Element type of the bare string-literal item at `items[j]`, if
    /// any: a string fills a whole character array as one item.
    fn string_item(&self, items: &[InitItem<'a>], j: usize) -> Option<Ty> {
        let item = items.get(j)?;
        if !item.designators.is_empty() {
            return None;
        }
        let Initializer::Expr(expr) = item.init else {
            return None;
        };
        let CheckedExprKind::Str(parts) = &expr.kind else {
            return None;
        };
        self.string_elem_ty(parts)
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

    /// Explicit member path for a positional item at `cursor`. The
    /// backend addresses members positionally and does not descend
    /// anonymous aggregates on its own, so bare items that land inside
    /// one carry those hops as designators. Plain root members stay
    /// bare: their position still covers them, and positional items
    /// skip unnamed bit-fields.
    fn cursor_designators(&self, root: StructId, cursor: &[usize], span: Span) -> Vec<Designator<'a>> {
        if let [index] = cursor
            && self.structs[root.0 as usize]
                .members
                .get(*index)
                .is_some_and(|member| member.name.is_some() || member.bitfield.is_some())
        {
            return Vec::new();
        }
        let mut designators = Vec::with_capacity(cursor.len());
        let mut id = root;
        for (depth, &index) in cursor.iter().enumerate() {
            let Some(member) = self.structs[id.0 as usize].members.get(index) else {
                break;
            };
            if depth + 1 == cursor.len() {
                designators.push(match member.name {
                    Some(name) => Designator::Field { name, span },
                    None => Designator::AnonField { index: index as u32, span },
                });
            } else {
                designators.push(Designator::AnonField { index: index as u32, span });
                let Ty::Struct(next) = member.ty.unqualified() else {
                    break;
                };
                id = *next;
            }
        }
        designators
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
    fn resolve_designation(&self, id: StructId, designators: &[Designator<'a>], interner: &Interner) -> Result<(Ty, usize, Vec<Designator<'a>>), CompileError> {
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
    fn walk_designation(&self, start: Ty, designators: &[Designator<'a>], interner: &Interner) -> Result<(Ty, Option<usize>, Vec<Designator<'a>>), CompileError> {
        let mut target = start;
        let mut outer: Option<usize> = None;
        let mut expanded: Vec<Designator<'a>> = Vec::with_capacity(designators.len());
        let mut visiting = Vec::new();
        for designator in designators {
            match (designator, target.unqualified()) {
                (Designator::Field { name, span }, Ty::Struct(cid)) => {
                    let path = match self.find_field(*cid, *name, &mut visiting) {
                        FieldHit::Found(path) => self.alloc_slice(path),
                        FieldHit::Miss => return Err(self.no_member_error(*cid, *name, *span, interner)),
                        FieldHit::Ambiguous => {
                            let text = interner.resolve(*name).unwrap_or("?");
                            return Err(CompileError::new(format!("ambiguous member '{text}' (found in several anonymous members)"), *span).with_code(DiagCode::AmbiguousMember));
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
                    for &index in path {
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
                    return Err(CompileError::new(format!("no member named '{text}'"), designator.span()).with_code(DiagCode::NoSuchMember));
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
    fn check_array_init(&self, target: &Ty, items: &[InitItem<'a>], interner: &Interner) -> Result<Vec<InitItem<'a>>, CompileError> {
        let Ty::Array { elem, bound, .. } = target else {
            unreachable!("array initializer needs an array type")
        };
        if let [item] = items
            && item.designators.is_empty() && matches!(item.init, Initializer::Expr (expr) if matches!(expr.ty.unqualified(), Ty::Array { .. }))
                && let Initializer::Expr(expr) = item.init
                    && let CheckedExprKind::Str(parts) = &expr.kind {
                        let lit_elem = self.string_elem_ty(parts);
                        if lit_elem.is_some_and(|lit| Self::string_elem_matches(elem, &lit)) {
                            self.check_string_fit(parts, bound, item.span, interner);
                            return Ok(vec![item.clone()]);
                        }
                    }
        if !items.is_empty() && self.is_variably_modified(target, interner) {
            return Err(CompileError::new(
                "variable-sized object may not be initialized",
                items.first().map(|item| item.span).unwrap_or_default(),
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
            let mut designators = item.designators.to_vec();
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
                if let Some(n) = count
                    && pos >= n {
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
                next = pos.saturating_add(1);
            }
            // Brace elision into an aggregate element (`struct P a[] =
            // {1, 2, ...}`): gather this and following bare scalars as
            // one element's braced list, consuming a single position.
            if item.designators.is_empty() && self.needs_elision(elem, item.init) {
                let j = self.elision_span(elem, items, i);
                if j > i {
                    let sub = items[i..j].to_vec();
                    let span = sub.first().map(|it| it.span).unwrap_or(item.span).merge(sub.last().map(|it| it.span).unwrap_or(item.span));
                    let init = self.check_initializer(
                        elem,
                        &Initializer::List {
                            items: self.alloc_slice(sub),
                            span,
                        },
                        interner,
                    )?;
                    checked.push(InitItem {
                        designators: &[],
                        init: self.alloc(init),
                        span: item.span,
                    });
                    i = j;
                    continue;
                }
            }
            let init = self.check_initializer(&target, item.init, interner)?;
            checked.push(InitItem {
                designators: self.alloc_slice(designators),
                init: self.alloc(init),
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
    pub(crate) fn complete_compound_ty(&self, ty: &Ty, init: &Initializer<'a>, interner: &Interner) -> Ty {
        let Ty::Array { elem, quals, bound: None } = ty else {
            return ty.clone();
        };
        // `(char[]){"ab"}`: the single string form sizes with its NUL.
        // The braced shape collapses to the bare `Expr<'a>` one at check
        // time (see `check_initializer`), so accept either.
        let string = match init {
            Initializer::Expr(expr) => Some(expr),
            Initializer::List { items, .. } => match items {
                [item] if item.designators.is_empty() => match item.init {
                    Initializer::Expr(expr) => Some(expr),
                    _ => None,
                },
                _ => None,
            },
        };
        if let Some(expr) = string
            && let CheckedExprKind::Str(parts) = &expr.kind {
                let lit_elem = self.string_elem_ty(parts);
                if lit_elem.is_some_and(|lit| Self::string_elem_matches(elem, &lit)) {
                    match crate::eval::string_length(parts, interner) {
                        Some(len) => {
                            return Ty::Array {
                                elem: elem.clone(),
                                quals: *quals,
                                bound: Some(ArrayBound::Fixed(len.saturating_add(1))),
                            };
                        }
                        None => return ty.clone(),
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
        for item in items.iter() {
            let mut pos = next;
            let mut designated = false;
            for designator in item.designators {
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
            quals: *quals,
            bound: Some(ArrayBound::Fixed(max)),
        }
    }

    /// The ordinary identifier whose spelling is closest to `name`,
    /// plus a noun for its kind (used in the suggestion). Candidates
    /// include bindings from scopes that have already closed (the
    /// scope archive), which the suggestion marks out-of-scope.
    fn similar_symbol(&self, name: SymbolU32, interner: &Interner) -> Option<(DeclId, String)> {
        let needle = interner.resolve(name)?;
        let best = did_you_mean(needle, self.visible_symbols().filter_map(|symbol| interner.resolve(symbol)))?;
        let symbol = self.visible_symbols().find(|symbol| interner.resolve(*symbol) == Some(best))?;
        let (id, live) = match self.lookup_symbol(symbol) {
            Some(id) => (id, true),
            // The closest spelling only binds in scopes that have
            // already closed; still worth pointing at.
            None => (self.archived_symbol(symbol)?, false),
        };
        let kind = match self.decls[id.0 as usize].kind {
            BindingKind::Typedef => "type",
            BindingKind::Func => "function",
            BindingKind::Var => "variable",
            BindingKind::EnumConst { .. } => "enum constant",
        };
        let noun = if live {
            match kind {
                "enum constant" => "an enum constant".to_string(),
                other => format!("a {other}"),
            }
        } else {
            format!("an out-of-scope {kind}")
        };
        Some((id, noun))
    }

    /// Member names reachable from `id`, descending through anonymous
    /// struct/union members (which promote their members).
    fn collect_field_candidates(&self, id: StructId, out: &mut Vec<(SymbolU32, Span)>, visiting: &mut Vec<StructId>) {
        if visiting.contains(&id) {
            return;
        }
        let Some(def) = self.structs.get(id.0 as usize) else {
            return;
        };
        visiting.push(id);
        for member in &def.members {
            match member.name {
                Some(name) => out.push((name, member.span)),
                None => {
                    if let Ty::Struct(inner) = member.ty.unqualified() {
                        self.collect_field_candidates(*inner, out, visiting);
                    }
                }
            }
        }
        visiting.pop();
    }

    fn similar_field(&self, id: StructId, name: SymbolU32, interner: &Interner) -> Option<(SymbolU32, Span)> {
        let needle = interner.resolve(name)?;
        let mut candidates = Vec::new();
        self.collect_field_candidates(id, &mut candidates, &mut Vec::new());
        let best = did_you_mean(needle, candidates.iter().filter_map(|(symbol, _)| interner.resolve(*symbol)))?;
        candidates.into_iter().find(|(symbol, _)| interner.resolve(*symbol) == Some(best))
    }

    fn no_member_error(&self, id: StructId, name: SymbolU32, span: Span, interner: &Interner) -> CompileError {
        let text = interner.resolve(name).unwrap_or("?");
        let mut err = CompileError::new(format!("no member named '{text}'"), span).with_code(DiagCode::NoSuchMember);
        if let Some((similar, similar_span)) = self.similar_field(id, name, interner) {
            let similar = interner.resolve(similar).unwrap_or("?");
            err = err
                .with_secondary(similar_span, format!("`{similar}` is defined here"))
                .with_help(format!("a field with a similar name exists: `{similar}`"));
        }
        err
    }

    /// Check a `return` value against the enclosing function's return
    /// type, converting as if by assignment. In a `void` function the
    /// value is legal only when it is itself `void` (`return f();`,
    /// C17 §6.8.6.4p1); anything else is a constraint violation.
    pub fn check_return(&self, ret: &Ty, expr: &Expr<'a>, interner: &Interner) -> Result<CheckedExpr<'a>, CompileError> {
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
    pub fn check_expr(&self, expr: &Expr<'a>, interner: &Interner) -> Result<CheckedExpr<'a>, CompileError> {
        let span = expr.span();
        match expr {
            Expr::Ident { name, .. } => {
                let Some(id) = self.lookup_symbol(*name) else {
                    let text = interner.resolve(*name).unwrap_or("?");
                    let mut err = CompileError::new(format!("use of undeclared identifier '{text}'"), span).with_code(DiagCode::UndeclaredIdentifier);
                    if let Some((id, kind)) = self.similar_symbol(*name, interner) {
                        let binding = &self.decls[id.0 as usize];
                        if let Some(similar) = binding.name.and_then(|symbol| interner.resolve(symbol)) {
                            err = err
                                .with_secondary(binding.span, format!("`{similar}` is defined here"))
                                .with_help(format!("{kind} with a similar name exists: `{similar}`"));
                        }
                    }
                    return Err(err);
                };
                let binding = &self.decls[id.0 as usize];
                if binding.kind == BindingKind::Typedef {
                    let text = interner.resolve(*name).unwrap_or("?");
                    return Err(CompileError::new(format!("typedef name '{text}' used as a value"), span).with_code(DiagCode::TypedefUsedAsValue));
                }
                if self.std_attrs.get(&id).is_some_and(|attrs| attrs.contains(StdAttributes::DEPRECATED)) {
                    let text = interner.resolve(*name).unwrap_or("?");
                    self.warnings.borrow_mut().push(CompileError::warning(format!("'{text}' is deprecated"), span));
                }
                Ok(CheckedExpr {
                    ty: binding.ty.clone(),
                    span,
                    kind: CheckedExprKind::Ident(id),
                })
            }
            Expr::Int { lit, .. } => {
                // C2y's `0o`/`0O` octal spelling; GNU and clang accept
                // it as an extension in older dialects.
                if lit.radix == crate::lexer::Radix::Oct
                    && interner.resolve(lit.spelling).is_some_and(|s| s.starts_with("0o") || s.starts_with("0O"))
                    && !(self.lang.standard_includes(crate::lang::Std::C2y) || self.lang.gnu || self.lang.clang)
                {
                    return Err(CompileError::new("`0o` octal constants require C2y", span));
                }
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
            Expr::Char { lit, .. } => {
                if lit.prefix == CharPrefix::U8 && !self.lang.standard_includes(crate::lang::Std::C23) {
                    return Err(CompileError::new("u8 character constants need C23", span));
                }
                Ok(CheckedExpr {
                    ty: match lit.prefix {
                        CharPrefix::None => Ty::Int,
                        CharPrefix::L => self.target.wchar().ty(),
                        CharPrefix::SmallU => Ty::UnsignedShort,
                        CharPrefix::BigU => Ty::UnsignedInt,
                        CharPrefix::U8 => Ty::UnsignedChar,
                    },
                    span,
                    kind: CheckedExprKind::Char(*lit),
                })
            }
            // Array of the literal's element type (`char` for narrow,
            // `int` for `L`, ...), bounded by its escape-aware element
            // count plus the NUL (`sizeof "ab"` is 3, including in
            // lowering, which reads the type rather than folding). A
            // malformed spelling keeps the incomplete type for the
            // caller to reject.
            Expr::Str { parts, .. } => Ok(CheckedExpr {
                ty: Ty::Array {
                    elem: Box::new(self.string_elem_ty(parts).unwrap_or(Ty::Char)),
                    quals: TypeQualifiers::empty(),
                    bound: crate::eval::string_length(parts, interner).map(|len| ArrayBound::Fixed(len + 1)),
                },
                span,
                kind: CheckedExprKind::Str(parts),
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
            Expr::GenericTy { ty, assocs, .. } => self.check_generic_ty(ty, assocs, interner, span),
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
                    kind: CheckedExprKind::VaArg {
                        ap: self.alloc(ap),
                        ty: ty.clone(),
                    },
                })
            }
            Expr::Offsetof { ty, designators, .. } => {
                let (offset, runtime) = self.check_offsetof(ty, designators, interner, span)?;
                let size_ty = self.target.size_type();
                // A runtime index keeps the folded prefix in the
                // `Offsetof` node and adds the index terms as ordinary
                // arithmetic, so lowering needs no new node kind.
                let kind = match runtime {
                    None => CheckedExprKind::Offsetof { ty: ty.clone(), offset },
                    Some(term) => CheckedExprKind::Binary(
                        BinaryOp::Add,
                        self.alloc(CheckedExpr {
                            ty: size_ty.clone(),
                            span,
                            kind: CheckedExprKind::Offsetof { ty: ty.clone(), offset },
                        }),
                        term,
                    ),
                };
                Ok(CheckedExpr { ty: size_ty, span, kind })
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
                    kind: CheckedExprKind::StmtExpr { items: items },
                })
            }
            Expr::CompoundLiteral { ty, init, storage, .. } => {
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
                if !storage.is_automatic() && !crate::eval::initializer_is_constant(&init, self, interner) {
                    return Err(CompileError::new("initializer element is not constant", span));
                }
                Ok(CheckedExpr {
                    ty: ty.clone(),
                    span,
                    kind: CheckedExprKind::CompoundLiteral {
                        ty: ty.clone(),
                        init: self.alloc(init),
                        storage: *storage,
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
                    kind: CheckedExprKind::SizeofExpr(self.alloc(operand)),
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
            Expr::CountofExpr { expr: operand, .. } => {
                let operand = self.check_expr(operand, interner)?;
                self.check_countof_operand(&operand.ty, operand.span)?;
                Ok(CheckedExpr {
                    ty: self.target.size_type(),
                    span,
                    kind: CheckedExprKind::CountofExpr(self.alloc(operand)),
                })
            }
            Expr::CountofTy { ty, .. } => {
                self.check_countof_operand(ty, span)?;
                Ok(CheckedExpr {
                    ty: self.target.size_type(),
                    span,
                    kind: CheckedExprKind::CountofTy(ty.clone()),
                })
            }
            Expr::StaticAssert { .. } => Ok(CheckedExpr {
                ty: Ty::Void,
                span,
                kind: CheckedExprKind::StaticAssert,
            }),
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
                    kind: CheckedExprKind::AlignofExpr(self.alloc(operand)),
                })
            }
            Expr::LabelAddr { label, .. } => Ok(CheckedExpr {
                ty: Ty::Pointer {
                    qualifiers: TypeQualifiers::empty(),
                    size: PtrSize::Default,
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
                self.warnings
                    .borrow_mut()
                    .push(CompileError::warning("invalid application of `sizeof` to a void type", span));
                return Ok(());
            }
            _ => {}
        }
        if !self.is_complete(ty) {
            return Err(CompileError::new("invalid application of `sizeof` to an incomplete type", span));
        }
        Ok(())
    }

    /// Validate a `_Countof` operand: a complete array type, with a
    /// known or variable top-level extent (but not `[]` or `[*]`).
    fn check_countof_operand(&self, ty: &Ty, span: Span) -> Result<(), CompileError> {
        match ty.unqualified() {
            Ty::Array {
                bound: Some(ArrayBound::Fixed(_) | ArrayBound::Vla(_)),
                ..
            } => Ok(()),
            Ty::Array {
                bound: Some(ArrayBound::Star) | None,
                ..
            } => Err(CompileError::new("invalid application of `_Countof` to an incomplete type", span)),
            _ => Err(CompileError::new("`_Countof` requires an argument of array type", span)),
        }
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

    fn check_unary(&self, op: UnaryOp, operand: &Expr<'a>, interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        use UnaryOp::*;
        match op {
            Pos | Neg => {
                let operand = self.decay_checked(self.check_expr(operand, interner)?);
                // GNU vectors negate lane-wise (`-v`); no promotion.
                if !is_arithmetic(&operand.ty) && !is_vector(&operand.ty) {
                    return Err(CompileError::new("unary operand must have arithmetic type", operand.span));
                }
                let operand = self.integer_promote(operand);
                let ty = operand.ty.clone();
                Ok(CheckedExpr {
                    ty,
                    span,
                    kind: CheckedExprKind::Unary(op, self.alloc(operand)),
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
                    kind: CheckedExprKind::Unary(op, self.alloc(operand)),
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
                        kind: CheckedExprKind::Unary(op, self.alloc(operand)),
                    });
                }
                if !is_integer(&operand.ty)
                    && !(is_vector(&operand.ty)
                        && matches!(operand.ty.unqualified(), Ty::Vector { elem, .. } if is_integer(elem)))
                {
                    return Err(CompileError::new("operand of `~` must have integer type", operand.span));
                }
                let operand = self.integer_promote(operand);
                let ty = operand.ty.clone();
                Ok(CheckedExpr {
                    ty,
                    span,
                    kind: CheckedExprKind::Unary(op, self.alloc(operand)),
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
                    kind: CheckedExprKind::Unary(op, self.alloc(operand)),
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
                    qualifiers: TypeQualifiers::empty(),
                    size: PtrSize::Default,
                    pointee: Box::new(self.as_callable_ty(operand.ty.clone(), &operand.kind)),
                };
                Ok(CheckedExpr {
                    ty,
                    span,
                    kind: CheckedExprKind::Unary(op, self.alloc(operand)),
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
                // Increment operands are real, complex, or pointer
                // (C17 §6.5.3.1p1; C2y adds complex, stepping the real
                // part). GNU and clang accept complex as an extension.
                if is_complex(&operand.ty) && !(self.lang.standard_includes(crate::lang::Std::C2y) || self.lang.gnu || self.lang.clang) {
                    return Err(CompileError::new("increment or decrement of a complex operand requires C2y", operand.span));
                }
                let ty = operand.ty.clone();
                Ok(CheckedExpr {
                    ty,
                    span,
                    kind: CheckedExprKind::Unary(op, self.alloc(operand)),
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
                    ty = Ty::qualified(TypeQualifiers::CONST, ty);
                }
                Ok(CheckedExpr {
                    ty,
                    span,
                    kind: CheckedExprKind::Unary(op, self.alloc(operand)),
                })
            }
        }
    }

    /// GNU vector extension operands for a binary operator (GCC "Vector
    /// Extensions"): two same-shaped vectors compute lane-wise, and one
    /// arithmetic scalar splats across the lanes (converted to the
    /// element type first). Comparisons produce the same vector shape
    /// with 0/`-1` lanes, like GCC. Returns `None` when neither operand
    /// is a vector, so scalar paths proceed untouched.
    fn check_vector_binary(&self, op: BinaryOp, l: CheckedExpr<'a>, r: CheckedExpr<'a>, interner: &Interner, span: Span) -> Option<Result<CheckedExpr<'a>, CompileError>> {
        use BinaryOp::*;
        if !is_vector(&l.ty) && !is_vector(&r.ty) {
            return None;
        }
        let supported = matches!(
            op,
            Add | Sub | Mul | Div | Rem | BitAnd | BitOr | BitXor | Shl | Shr | Lt | Le | Gt | Ge | Eq | Ne
        );
        if !supported {
            return Some(Err(CompileError::new("invalid operands to binary operator", span)));
        }
        let (v, s, vector_first) = if is_vector(&l.ty) { (l, r, true) } else { (r, l, false) };
        // `%`, shifts, and bitwise operators need integer lanes, like
        // their scalar counterparts.
        if matches!(op, Rem | BitAnd | BitOr | BitXor | Shl | Shr)
            && let Ty::Vector { elem, .. } = v.ty.unqualified()
            && !is_integer(elem)
        {
            return Some(Err(CompileError::new("invalid operands to binary operator", span)));
        }
        let (vec_ty, elem) = match v.ty.unqualified() {
            Ty::Vector { elem, len } => {
                // Comparisons mask lane-wise: the result is a signed
                // integer vector of the operand shape (GCC: "a signed
                // vector of the same size"), so `v4sf < v4sf` yields
                // `v4si` masks, not float lanes.
                let result_ty = if matches!(op, Lt | Le | Gt | Ge | Eq | Ne) {
                    let mask_elem = match self.type_size(elem, interner) {
                        Some(1) => Ty::SignedChar,
                        Some(2) => Ty::Short,
                        Some(4) => Ty::Int,
                        _ => Ty::LongLong,
                    };
                    Ty::Vector {
                        elem: Box::new(mask_elem),
                        len: *len,
                    }
                } else {
                    v.ty.clone()
                };
                (result_ty, elem.as_ref().clone())
            }
            _ => unreachable!("vector operand selected above"),
        };
        if is_vector(&s.ty) {
            if s.ty.unqualified() != v.ty.unqualified() {
                return Some(Err(CompileError::new("vector operands must have the same shape", span)));
            }
            let node = if vector_first { (v, s) } else { (s, v) };
            return Some(Ok(CheckedExpr {
                ty: vec_ty,
                span,
                kind: CheckedExprKind::Binary(op, self.alloc(node.0), self.alloc(node.1)),
            }));
        }
        if !is_arithmetic(&s.ty) {
            return Some(Err(CompileError::new("invalid operands to binary operator", span)));
        }
        let s = self.convert(s, elem);
        let node = if vector_first { (v, s) } else { (s, v) };
        Some(Ok(CheckedExpr {
            ty: vec_ty,
            span,
            kind: CheckedExprKind::Binary(op, self.alloc(node.0), self.alloc(node.1)),
        }))
    }

    fn check_binary(&self, op: BinaryOp, lhs: &Expr<'a>, rhs: &Expr<'a>, interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        use BinaryOp::*;
        let lhs = self.check_expr(lhs, interner)?;
        let rhs = self.check_expr(rhs, interner)?;
        let build = |l: CheckedExpr<'a>, r: CheckedExpr<'a>, ty: Ty| CheckedExpr {
            ty,
            span,
            kind: CheckedExprKind::Binary(op, self.alloc(l), self.alloc(r)),
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
                if let Some(result) = self.check_vector_binary(op, l.clone(), r.clone(), interner, span) {
                    return result;
                }
                if is_arithmetic(&l.ty) && is_arithmetic(&r.ty) {
                    let (l, r, _) = self.usual_arithmetic_conversions(l, r);
                    return Ok(build(l, r, Ty::Int));
                }
                let l_null = matches!(l.ty.unqualified(), Ty::Nullptr);
                let r_null = matches!(r.ty.unqualified(), Ty::Nullptr);
                if l_null || r_null {
                    if l_null && r_null {
                        return Ok(build(l, r, Ty::Int));
                    }
                    if l_null {
                        if matches!(r.ty.unqualified(), Ty::Pointer { .. }) {
                            let ty = r.ty.clone();
                            return Ok(build(self.convert(l, ty), r, Ty::Int));
                        }
                        if self.is_null_pointer_constant(&r, interner) {
                            return Ok(build(l, self.convert(r, Ty::Nullptr), Ty::Int));
                        }
                    } else {
                        if matches!(l.ty.unqualified(), Ty::Pointer { .. }) {
                            let ty = l.ty.clone();
                            return Ok(build(l, self.convert(r, ty), Ty::Int));
                        }
                        if self.is_null_pointer_constant(&l, interner) {
                            return Ok(build(self.convert(l, Ty::Nullptr), r, Ty::Int));
                        }
                    }
                    return Err(CompileError::new("invalid operands to `==`/`!=`", span));
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
                if let Some(result) = self.check_vector_binary(op, l.clone(), r.clone(), interner, span) {
                    return result;
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
                // GNU vectors compute lane-wise; the scalar operand
                // splats (see `check_vector_binary`).
                if let Some(result) = self.check_vector_binary(op, l.clone(), r.clone(), interner, span) {
                    return result;
                }
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
    pub fn check_asm_output(&self, expr: &Expr<'a>, interner: &Interner) -> Result<CheckedExpr<'a>, CompileError> {
        let checked = self.check_expr(expr, interner)?;
        if !self.is_lvalue(&checked) {
            return Err(CompileError::new("asm output is not an lvalue", checked.span));
        }
        if checked.ty.is_const() {
            return Err(self.readonly_error(&checked, interner));
        }
        Ok(checked)
    }

    fn check_assign(&self, op: AssignOp, lhs: &Expr<'a>, rhs: &Expr<'a>, interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
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
                kind: CheckedExprKind::Assign(op, self.alloc(l), self.alloc(r)),
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
        } else if is_vector(&l_dec.ty) || is_vector(&r_dec.ty) {
            self.vector_compound_assign(op, l_dec, r_dec.clone(), span)?
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
                lhs: self.alloc(l),
                rhs: self.alloc(r_dec),
                op_ty: Box::new(op_l.ty.clone()),
                rhs_ty: Box::new(op_r.ty.clone()),
            },
        })
    }

    /// GNU vector compound assignment (`v += w`, `v *= 2`): the left
    /// side is the vector, the right a same-shaped vector or a scalar
    /// splatting into the element type. The op operands carry the
    /// conversions; the assignment's own type stays the lvalue's.
    fn vector_compound_assign(&self, op: AssignOp, l: CheckedExpr<'a>, r: CheckedExpr<'a>, span: Span) -> Result<(CheckedExpr<'a>, CheckedExpr<'a>), CompileError> {
        if !matches!(
            op,
            AssignOp::Add | AssignOp::Sub | AssignOp::Mul | AssignOp::Div | AssignOp::Rem | AssignOp::BitAnd | AssignOp::BitOr | AssignOp::BitXor | AssignOp::Shl | AssignOp::Shr
        ) {
            return Err(CompileError::new("invalid operands to compound assignment", span));
        }
        if !is_vector(&l.ty) {
            return Err(CompileError::new("compound assignment to a scalar with a vector operand", span));
        }
        let elem = match l.ty.unqualified() {
            Ty::Vector { elem, .. } => elem.as_ref().clone(),
            _ => unreachable!("vector left operand checked above"),
        };
        let r = if is_vector(&r.ty) {
            if r.ty.unqualified() != l.ty.unqualified() {
                return Err(CompileError::new("vector operands must have the same shape", span));
            }
            r
        } else if is_arithmetic(&r.ty) {
            self.convert(r, elem)
        } else {
            return Err(CompileError::new("invalid operands to compound assignment", span));
        };
        Ok((l, r))
    }

    fn check_ternary(&self, cond: &Expr<'a>, then: &Expr<'a>, els: &Expr<'a>, interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
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
    fn check_elvis(&self, cond: &Expr<'a>, els: &Expr<'a>, interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let cond = self.decay_checked(self.check_expr(cond, interner)?);
        let els = self.decay_checked(self.check_expr(els, interner)?);
        let then = cond.clone();
        self.check_conditional(cond, then, els, interner, span, true)
    }

    /// The common tail of `?:`: the condition must be a scalar, and the
    /// arms convert to a shared result type. `elvis` picks the node
    /// shape (see [`Self::conditional_node`]).
    fn check_conditional(
        &self,
        cond: CheckedExpr<'a>,
        then: CheckedExpr<'a>,
        els: CheckedExpr<'a>,
        interner: &Interner,
        span: Span,
        elvis: bool,
    ) -> Result<CheckedExpr<'a>, CompileError> {
        if !is_scalar(&cond.ty) {
            return Err(CompileError::new("ternary condition must have scalar type", cond.span));
        }
        if is_arithmetic(&then.ty) && is_arithmetic(&els.ty) {
            let (then, els, ty) = self.usual_arithmetic_conversions(self.decay_checked(then), self.decay_checked(els));
            return Ok(self.conditional_node(cond, then, els, ty, span, elvis));
        }
        if matches!(then.ty.unqualified(), Ty::Nullptr) && matches!(els.ty.unqualified(), Ty::Nullptr) {
            return Ok(self.conditional_node(cond, then, els, Ty::Nullptr, span, elvis));
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
                    self.warnings
                        .borrow_mut()
                        .push(CompileError::warning("pointer type mismatch in conditional expression", span));
                }
                let ty = then.ty.clone();
                return Ok(self.conditional_node(cond, then, els, ty, span, elvis));
            }
            if rev != PointeeConversion::No {
                if rev == PointeeConversion::Warns {
                    self.warnings
                        .borrow_mut()
                        .push(CompileError::warning("pointer type mismatch in conditional expression", span));
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
    fn conditional_node(&self, cond: CheckedExpr<'a>, then: CheckedExpr<'a>, els: CheckedExpr<'a>, ty: Ty, span: Span, elvis: bool) -> CheckedExpr<'a> {
        let kind = if elvis {
            CheckedExprKind::Elvis {
                cond: self.alloc(cond),
                els: self.alloc(els),
            }
        } else {
            CheckedExprKind::Ternary {
                cond: self.alloc(cond),
                then: self.alloc(then),
                els: self.alloc(els),
            }
        };
        CheckedExpr { ty, span, kind }
    }

    fn check_call(&self, func: &Expr<'a>, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
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
                // GCC's x86 `__builtin_ia32_*` builtins.
                if text.starts_with("__builtin_ia32_") {
                    return match self.check_ia32(text, args, interner, span) {
                        Some(result) => result,
                        None => {
                            Err(CompileError::new(format!("unsupported builtin `{text}`"), span).with_code(DiagCode::UnsupportedAtomicBuiltin))
                        }
                    };
                }
                // The legacy `__sync` atomics (GCC §7.9.2), which the
                // docs define in terms of the `__atomic` builtins.
                if text.starts_with("__sync_") {
                    return match self.check_sync(text, args, interner, span) {
                        Some(result) => result,
                        None => {
                            Err(CompileError::new(format!("unsupported builtin `{text}`"), span).with_code(DiagCode::UnsupportedAtomicBuiltin))
                        }
                    };
                }
                // MSVC atomics from `<intrin.h>` (the `8`/`16`/`64`/
                // `Pointer` widths share the one shape).
                if text.starts_with("_InterlockedCompareExchange") {
                    return self.check_interlocked_cmpxchg(args, interner, span, text);
                }
            }
            match interner.resolve(*name) {
                // Type-generic bit counts (`__builtin_clzg`,
                // `__builtin_ctzg`) likewise check inline: the first
                // argument keeps its own width (no promotion), with an
                // optional `int` zero-fallback second argument.
                Some("__builtin_clzg") => return self.check_clzg(args, interner, span, false),
                Some("__builtin_ctzg") => return self.check_clzg(args, interner, span, true),
                Some(n @ ("__builtin_bswap16" | "__builtin_bswap32" | "__builtin_bswap64")) => {
                    return self.check_bswap(n, args, interner, span);
                }
                Some("__builtin_stdc_rotate_left") => return self.check_rotate(args, interner, span, true),
                Some("__builtin_stdc_rotate_right") => return self.check_rotate(args, interner, span, false),
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
                Some("__builtin_shuffle") => return self.check_shuffle(args, interner, span),
                Some("__builtin_shufflevector") => return self.check_shufflevector(args, interner, span),
                Some("__builtin_convertvector") => return self.check_convertvector(args, interner, span),
                Some("__builtin_nontemporal_store") => return self.check_nontemporal_store(args, interner, span),
                Some(n @ ("__builtin_elementwise_add_sat" | "__builtin_elementwise_sub_sat" | "__builtin_elementwise_min" | "__builtin_elementwise_max" | "__builtin_elementwise_sqrt")) => {
                    return self.check_elementwise(n, args, interner, span);
                }
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
                func: self.alloc(callee),
                args: self.alloc_slice(checked),
            },
        })
    }

    fn check_subscript(&self, base: &Expr<'a>, index: &Expr<'a>, interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        // GNU vector subscript reads (`v[i]`, the rvalue of lane `i`):
        // reads only, like GCC — a subscripted vector is not a
        // per-lane lvalue, so `v[i] = x` fails as an assignment to a
        // non-lvalue. Indexing works on rvalues too (`(a + b)[0]`).
        {
            let base = self.check_expr(base, interner)?;
            let index = self.decay_checked(self.check_expr(index, interner)?);
            if let Ty::Vector { elem, .. } = base.ty.unqualified() {
                if !is_integer(&index.ty) {
                    return Err(CompileError::new("vector subscript requires an integer index", span));
                }
                let index = self.integer_promote(index);
                return Ok(CheckedExpr {
                    ty: elem.as_ref().clone(),
                    span,
                    kind: CheckedExprKind::Subscript {
                        base: self.alloc(base),
                        index: self.alloc(index),
                    },
                });
            }
        }
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
                base: self.alloc(ptr),
                index: self.alloc(idx),
            },
        })
    }

    fn check_member(&self, base: &Expr<'a>, field: SymbolU32, arrow: bool, interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
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
            FieldHit::Miss => return Err(self.no_member_error(id, field, span, interner)),
            FieldHit::Ambiguous => {
                let text = interner.resolve(field).unwrap_or("?");
                return Err(CompileError::new(format!("ambiguous member '{text}' (found in several anonymous members)"), span).with_code(DiagCode::AmbiguousMember));
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
            if level + 1 == path.len()
                && let Some(width_expr) = &member.bitfield {
                    let w = *width_expr;
                    // The unit byte this field starts in, relative to
                    // the containing struct: `member_offsets` holds the
                    // byte of the unit start, and `members` the exact
                    // bit, keyed by the accessed name.
                    if let Some(ml) = layout.members.iter().find(|m| m.name == member.name.unwrap()) {
                        bitfield = Some((ml.bit.unwrap_or(0), w as u32));
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
        if let Some((_, width)) = &bitfield
            && *width <= 32 {
                ty = if ty.is_unsigned_in(self.target.char_signed) {
                    if *width >= 32 { Ty::UnsignedInt } else { Ty::Int }
                } else {
                    Ty::Int
                };
            }
        // A member of a `const`-qualified object is itself `const`
        // (C17 §6.7.3p5): `const struct S s; s.x = 1;` fails, as does
        // `p->x = 1;` through a `const struct S *p`.
        if target_ty.is_const() {
            ty = Ty::qualified(TypeQualifiers::CONST, ty);
        }
        Ok(CheckedExpr {
            ty,
            span,
            kind: CheckedExprKind::Member {
                base: self.alloc(checked),
                path: self.alloc_slice(path),
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
    fn check_generic(&self, ctrl: &Expr<'a>, assocs: &[GenericAssoc<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let ctrl = self.decay_checked(self.check_expr(ctrl, interner)?);
        let (index, checked) = self.check_generic_assocs(&ctrl.ty, assocs, interner, span)?;
        let ty = checked[index].1.ty.clone();
        Ok(CheckedExpr {
            ty,
            span,
            kind: CheckedExprKind::Generic {
                ctrl: self.alloc(ctrl),
                index,
                assocs: self.alloc_slice(checked),
            },
        })
    }

    /// C2y `_Generic ( type-name , ... )`: the type operand replaces the
    /// controlling expression, so no lvalue conversion or evaluation is
    /// involved.
    fn check_generic_ty(&self, ty: &Ty, assocs: &[GenericAssoc<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let (index, checked) = self.check_generic_assocs(ty, assocs, interner, span)?;
        let selected_ty = checked[index].1.ty.clone();
        Ok(CheckedExpr {
            ty: selected_ty,
            span,
            kind: CheckedExprKind::GenericTy {
                ty: ty.clone(),
                index,
                assocs: self.alloc_slice(checked),
            },
        })
    }

    /// Resolve the selected association for a controlling type. The
    /// first compatible type-name wins, else `default` (which must
    /// exist). Duplicate type-names and a second `default` are
    /// constraint violations.
    fn check_generic_assocs(
        &self,
        ctrl_ty: &Ty,
        assocs: &[GenericAssoc<'a>],
        interner: &Interner,
        span: Span,
    ) -> Result<(usize, Vec<(Option<Ty>, CheckedExpr<'a>)>), CompileError> {
        let mut checked: Vec<(Option<Ty>, CheckedExpr<'a>)> = Vec::with_capacity(assocs.len());
        let mut default = None;
        let mut selected = None;
        for assoc in assocs {
            let expr = self.check_expr(assoc.expr, interner)?;
            match &assoc.ty {
                None => {
                    if default.replace(checked.len()).is_some() {
                        return Err(CompileError::new("duplicate `default` in `_Generic`", assoc.span));
                    }
                }
                Some(ty) => {
                    if checked
                        .iter()
                        .filter_map(|(t, _)| t.as_ref())
                        .any(|previous| types_compatible(previous, ty) || self.enum_repr_pair(previous, ty))
                    {
                        return Err(CompileError::new("duplicate type in `_Generic`", assoc.span));
                    }
                    if selected.is_none() && (types_compatible(ctrl_ty, ty) || self.enum_repr_pair(ctrl_ty, ty)) {
                        selected = Some(checked.len());
                    }
                }
            }
            checked.push((assoc.ty.clone(), expr));
        }
        let index = selected.or(default).ok_or_else(|| CompileError::new("no matching generic association", span))?;
        Ok((index, checked))
    }

    /// Byte offset of a `__builtin_offsetof(TYPE, .a.b[i])` designator
    /// (GCC's `offsetof`): each `.field` descends through the current
    /// struct or union (anonymous members included), each `[i]` steps
    /// into a member array. Bit-fields have no addressable offset and
    /// are rejected, like GCC.
    ///
    /// A constant index folds into the byte offset; a runtime one is
    /// the GCC extension, lowered to `prefix + i * sizeof(elem)` as
    /// plain arithmetic (returned as the second component) so any
    /// context that truly needs a constant still rejects it.
    fn check_offsetof(&self, ty: &Ty, designators: &[Designator<'a>], interner: &Interner, span: Span) -> Result<(u64, Option<&'a CheckedExpr<'a>>), CompileError> {
        let Ty::Struct(_) = ty.unqualified() else {
            return Err(CompileError::new("`offsetof` needs a struct or union type", span));
        };
        if !self.is_complete(ty) {
            return Err(CompileError::new("`offsetof` of an incomplete type", span));
        }
        let mut cur = ty.clone();
        let mut offset: u64 = 0;
        let mut runtime: Option<&'a CheckedExpr<'a>> = None;
        let mut visiting = Vec::new();
        for designator in designators {
            match designator {
                Designator::Field { name, span } => {
                    let Ty::Struct(cid) = cur.unqualified() else {
                        return Err(CompileError::new("invalid `.field` designator in `offsetof`", *span));
                    };
                    let path = match self.find_field(*cid, *name, &mut visiting) {
                        FieldHit::Found(path) => self.alloc_slice(path),
                        FieldHit::Miss => return Err(self.no_member_error(*cid, *name, *span, interner)),
                        FieldHit::Ambiguous => {
                            let text = interner.resolve(*name).unwrap_or("?");
                            return Err(CompileError::new(format!("ambiguous member '{text}' (found in several anonymous members)"), *span).with_code(DiagCode::AmbiguousMember));
                        }
                    };
                    let mut at = *cid;
                    let mut final_is_bitfield = false;
                    for &index in path {
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
                    let elem_size = self
                        .type_size(elem, interner)
                        .ok_or_else(|| CompileError::new("cannot compute layout for `offsetof`", *span))?;
                    match crate::eval::eval_const(index, self, interner) {
                        Some(crate::eval::Value::Int(n)) if n >= 0 => {
                            offset = offset
                                .checked_add((n as u64).checked_mul(elem_size).ok_or_else(|| CompileError::new("cannot compute layout for `offsetof`", *span))?)
                                .ok_or_else(|| CompileError::new("cannot compute layout for `offsetof`", *span))?;
                        }
                        // A negative constant index stays rejected like GCC.
                        Some(crate::eval::Value::Int(_)) => {
                            return Err(CompileError::new("offsetof index is not an integer constant expression", *span));
                        }
                        None => {
                            let size_ty = self.target.size_type();
                            let idx = self.convert(self.decay_checked(index.clone()), size_ty.clone());
                            let mul = CheckedExpr {
                                ty: size_ty.clone(),
                                span: *span,
                                kind: CheckedExprKind::Binary(
                                    BinaryOp::Mul,
                                    self.alloc(idx),
                                    self.alloc(CheckedExpr {
                                        ty: size_ty.clone(),
                                        span: *span,
                                        kind: CheckedExprKind::SizeofTy(elem.as_ref().clone()),
                                    }),
                                ),
                            };
                            runtime = Some(match runtime {
                                None => self.alloc(mul),
                                Some(prev) => self.alloc(CheckedExpr {
                                    ty: size_ty.clone(),
                                    span: *span,
                                    kind: CheckedExprKind::Binary(BinaryOp::Add, prev, self.alloc(mul)),
                                }),
                            });
                        }
                        _ => {
                            return Err(CompileError::new("offsetof index is not an integer constant expression", *span));
                        }
                    }
                    cur = elem.as_ref().clone();
                }
                Designator::AnonField { span, .. } => {
                    return Err(CompileError::new("invalid `.field` designator in `offsetof`", *span));
                }
            }
        }
        Ok((offset, runtime))
    }

    fn check_cast(&self, ty: &Ty, operand: &Expr<'a>, interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let operand = self.decay_checked(self.check_expr(operand, interner)?);
        // `(void)e` discards the value: legal for any operand, including
        // a `void` one (EDG lowerings wrap `void` calls this way).
        if ty.unqualified() == &Ty::Void {
            return Ok(CheckedExpr {
                ty: ty.clone(),
                span,
                kind: CheckedExprKind::Cast {
                    ty: ty.clone(),
                    expr: self.alloc(operand),
                },
            });
        }
        let target_nullptr = matches!(ty.unqualified(), Ty::Nullptr);
        let operand_nullptr = matches!(operand.ty.unqualified(), Ty::Nullptr);
        if target_nullptr {
            let null_operand = operand_nullptr
                || (is_integer(&operand.ty) && matches!(crate::eval::eval_const(&operand, self, interner), Some(crate::eval::Value::Int(0))))
                || (is_void_ptr(&operand.ty) && self.is_null_pointer_constant(&operand, interner));
            if !null_operand {
                return Err(CompileError::new("cannot cast to `nullptr_t` from a non-null pointer constant", span));
            }
            return Ok(CheckedExpr {
                ty: ty.clone(),
                span,
                kind: CheckedExprKind::Cast {
                    ty: ty.clone(),
                    expr: self.alloc(operand),
                },
            });
        }
        if operand_nullptr {
            if !matches!(ty.unqualified(), Ty::Pointer { .. } | Ty::Bool) {
                return Err(CompileError::new("cannot cast `nullptr_t` to a non-pointer type", span));
            }
            return Ok(CheckedExpr {
                ty: ty.clone(),
                span,
                kind: CheckedExprKind::Cast {
                    ty: ty.clone(),
                    expr: self.alloc(operand),
                },
            });
        }
        // GNU vector casts: between vectors of the same size, and
        // between a vector and an arithmetic scalar of the same size —
        // the bits move untouched (`(__m64)__i` in mmintrin.h). Scalars
        // keep the ordinary scalar-cast rules below.
        if (is_vector(&operand.ty) || is_vector(ty))
            && !is_complex(&operand.ty)
            && !is_complex(ty)
            && (is_arithmetic(&operand.ty) || is_vector(&operand.ty))
            && (is_arithmetic(ty) || is_vector(ty))
            && self.type_size(&operand.ty, interner) == self.type_size(ty, interner)
        {
            return Ok(CheckedExpr {
                ty: ty.clone(),
                span,
                kind: CheckedExprKind::Cast {
                    ty: ty.clone(),
                    expr: self.alloc(operand),
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
                    expr: self.alloc(operand),
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
                    expr: self.alloc(operand),
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
                        init: self.alloc(Initializer::List {
                            items: self.alloc_slice([InitItem {
                                designators: self.alloc_slice([Designator::Field { name, span }]),
                                init: self.alloc(Initializer::Expr(operand)),
                                span,
                            }]),
                            span,
                        }),
                        storage: crate::parse::CompoundStorage::default(),
                    },
                });
            }
        }
        Err(CompileError::new("invalid operands to cast", span))
    }
}
