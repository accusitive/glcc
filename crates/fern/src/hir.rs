//! Typed HIR: the output of checking, the input of lowering.
//!
//! There is one [`HirFn`] per top-level function, top-level value, and
//! instance method. Types mention the body's `generics` as `Ty::Param`;
//! checking finalizes the program as `Ty::Var`-free, and lowering
//! substitutes concrete types per monomorphization.
//!
//! fern is pure: expressions have no sequencing effects, so the HIR is
//! a plain expression tree. Everything a lowering needs about control
//! flow — pattern matches, guards, binds — is one node.

use crate::ast::{BinOp, UnOp};
use crate::defs::{Bound, ClassId, DataId, FnId, ValueId};
use crate::span::Span;
use crate::ty::{Kind, Ty};

/// Which item body a [`HirFn`] came from; the monomorphization key
/// together with a type-argument vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BodyKey {
    Fn(FnId),
    Value(ValueId),
    Method { instance: usize, method: usize },
}

#[derive(Debug, Clone, Default)]
pub struct HirProgram {
    pub bodies: Vec<HirFn>,
    /// Closure bodies, referenced from HIR by index.
    pub closures: Vec<HirClosure>,
}

#[derive(Debug, Clone)]
pub struct HirFn {
    pub key: BodyKey,
    /// Display name, e.g. `map` or `List::map`.
    pub name: String,
    /// The body's type parameters, in order; their instantiation is the
    /// monomorphization key.
    pub generics: Vec<(String, Kind)>,
    /// Obligations assumed inside the body (an instance's context).
    pub bounds: Vec<Bound>,
    /// One pattern per parameter, in order.
    pub params: Vec<HirPat>,
    pub ret: Ty,
    pub locals: Vec<HirLocal>,
    pub body: HirExpr,
    pub span: Span,
}

/// A closure body. Closures are monomorphized together with their
/// enclosing body: types use the enclosing body's parameters.
#[derive(Debug, Clone)]
pub struct HirClosure {
    /// Index into [`HirProgram::closures`]; part of the mono key.
    pub id: u32,
    pub name: String,
    /// The enclosing body's generics, in order.
    pub generics: Vec<(String, Kind)>,
    /// One pattern per call parameter, in order.
    pub params: Vec<HirPat>,
    pub ret: Ty,
    /// Locals of the closure body, captures included.
    pub locals: Vec<HirLocal>,
    /// Copies taken from the enclosing body at creation time.
    pub captures: Vec<HirCapture>,
    pub body: HirExpr,
    pub span: Span,
}

/// One captured local: `outer` is a local of the enclosing body, and
/// `inner` the local of the closure body that holds the copy.
#[derive(Debug, Clone)]
pub struct HirCapture {
    pub outer: usize,
    pub inner: usize,
    pub ty: Ty,
}

#[derive(Debug, Clone)]
pub struct HirLocal {
    pub name: String,
    pub ty: Ty,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirExpr {
    pub kind: HirExprKind,
    pub ty: Ty,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum HirExprKind {
    Const(ConstValue),
    Local(usize),
    /// A constructor call; `args` match the constructor's fields.
    Ctor {
        data_id: DataId,
        ctor: usize,
        args: Vec<HirExpr>,
    },
    /// A globally named function or value used as a value. Lowering
    /// wraps it in a parameterless closure thunk.
    Fn(FnRef),
    /// A closure value: the listed locals of the enclosing body are
    /// copied into the environment, in the closure's capture order.
    Closure {
        id: u32,
        captures: Vec<usize>,
    },
    /// A fully applied call of a known target.
    CallDirect {
        target: FnRef,
        args: Vec<HirExpr>,
    },
    /// A fully applied call through a closure value.
    Call {
        callee: Box<HirExpr>,
        args: Vec<HirExpr>,
    },
    Tuple(Vec<HirExpr>),
    Let {
        binds: Vec<(HirPat, HirExpr)>,
        body: Box<HirExpr>,
    },
    If {
        cond: Box<HirExpr>,
        then_body: Box<HirExpr>,
        else_body: Box<HirExpr>,
    },
    Match {
        scrutinee: Box<HirExpr>,
        arms: Vec<HirArm>,
    },
    Bin {
        op: BinOp,
        lhs: Box<HirExpr>,
        rhs: Box<HirExpr>,
    },
    Un {
        op: UnOp,
        operand: Box<HirExpr>,
    },
}

#[derive(Debug, Clone)]
pub struct HirArm {
    pub pattern: HirPat,
    pub guard: Option<HirExpr>,
    pub body: HirExpr,
}

#[derive(Debug, Clone)]
pub struct HirPat {
    pub kind: HirPatKind,
    pub ty: Ty,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum HirPatKind {
    Wild,
    Bind(usize),
    Int(i64),
    Bool(bool),
    Ctor { data_id: DataId, ctor: usize, fields: Vec<HirPat> },
    Tuple(Vec<HirPat>),
}

#[derive(Debug, Clone)]
pub enum FnRef {
    /// A top-level function; `args` instantiate its generics.
    Fn { id: FnId, args: Vec<Ty> },
    /// A top-level value; `args` instantiate its generics.
    Value { id: ValueId, args: Vec<Ty> },
    /// An instance method; `args` instantiate the instance's generics
    /// followed by the method signature's own parameters.
    Instance { instance: usize, method: usize, args: Vec<Ty> },
    /// A class method whose dispatch is still open; lowering resolves it
    /// through the solver once the class arguments are concrete. `args`
    /// holds the class arguments first, then the method's own
    /// instantiation (the same tail [`FnRef::Instance`] carries).
    Class { class_id: ClassId, method: usize, args: Vec<Ty> },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstValue {
    Int(i64),
    Float(f64),
    Bool(bool),
}
