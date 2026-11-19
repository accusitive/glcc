//! The prism shader profile (`-fprism`).
//!
//! Prism is a C subset whose constructs must be expressible on the
//! shader backends (see `crates/glcc_prism` and `docs/prism.md`):
//! no pointers, no recursion, no dynamic allocation, 32-bit float
//! math only, structured control flow. This module is the profile
//! pass: after parsing, it walks every declaration, type, statement,
//! and checked expression in the translation unit and rejects
//! whatever a shader cannot say. The driver calls
//! [`Sema::prism_validate`] while the parser is still alive, so
//! errors render through the usual diagnostic machinery.
//!
//! The walk is deliberately whole-`Sema` (not fused into per-expr
//! checking): one choke point keeps the profile rules together, and
//! `Sema` already holds every checked body. Rules that need parse
//! attributes (`[[prism::stage]]`/`[[prism::semantic]]` on entry
//! points and interface members) read them from `Sema::prism_attrs`
//! / `Member::prism`, recorded by the parser.

use alloc::{format, vec::Vec};

use hashbrown::{HashMap, HashSet};

use string_interner::symbol::SymbolU32;

use super::{CheckedExpr, CheckedExprKind, is_arithmetic, is_vector};
use crate::{
    lexer::Interner,
    parse::{ArrayBound, AssignOp, BlockItem, Compound, Expr, ExternalDeclaration, ForInit, Initializer, PrismStage, Stmt, TranslationUnit, Ty, UnaryOp},
    sema::{BindingKind, DeclId, Sema, StructId},
    span::{CompileError, HasSpan, Span},
};

/// The reserved tag of the opaque `sampler2D` type `prism.h` declares.
/// Resources only ever appear as file-scope objects of this type.
const SAMPLER2D_TAG: &str = "__prism_sampler2d";

/// Functions the compiler expands to shader operations in prism mode
/// (declared by `prism.h`, lowered by name in `glcc_lower_mir`):
/// never real calls out of the shader.
const PRISM_SCALAR_BUILTINS: &[&str] = &[
    "min", "max", "sqrt", "sqrtf", "floor", "floorf", "ceil", "ceilf", "fabs", "fabsf",
    "splat2", "splat3", "splat4", "min2", "min3", "min4", "max2", "max3", "max4", "abs2", "abs3", "abs4", "mul4x4",
];

/// Parse a swizzle spelling (`.xyzw` / `.rgba` aliases, 1-4 lanes).
/// `None` when any character is not a lane name.
pub(crate) fn parse_swizzle(text: &str) -> Option<Vec<u8>> {
    if text.is_empty() || text.len() > 4 {
        return None;
    }
    let mut lanes = Vec::with_capacity(text.len());
    for ch in text.chars() {
        lanes.push(match ch {
            'x' | 'r' => 0,
            'y' | 'g' => 1,
            'z' | 'b' => 2,
            'w' | 'a' => 3,
            _ => return None,
        });
    }
    Some(lanes)
}

impl<'a> Sema<'a> {
    /// Build a swizzle read over `base` (a vector): the result is a
    /// vector of `lanes.len()` lanes of the same element type — or the
    /// element itself for a single lane (`v.x` reads as `float`).
    pub(super) fn swizzle_node(&self, base: CheckedExpr<'a>, lanes: &[u8], span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let (elem, len) = match base.ty.unqualified() {
            Ty::Vector { elem, len } => ((**elem).clone(), *len),
            _ => return Err(CompileError::new("swizzles apply to vector operands", span)),
        };
        if lanes.iter().any(|&lane| u32::from(lane) >= len) {
            return Err(CompileError::new(format!("swizzle lane out of range for a {len}-lane vector"), span));
        }
        let ty = if lanes.len() == 1 {
            elem
        } else {
            Ty::Vector { elem: Box::new(elem), len: lanes.len() as u32 }
        };
        Ok(CheckedExpr {
            ty,
            span,
            kind: CheckedExprKind::Swizzle { base: self.alloc(base), lanes: self.alloc_slice(lanes.to_vec()) },
        })
    }

    /// `v.xy = e` (`-fprism`): the swizzled assignment becomes a
    /// whole-vector store whose value shuffles the old contents with
    /// the assigned lanes. Compound forms stay unsupported (the read
    /// and the write are one lowering step each; a compound op would
    /// need both under one node).
    pub(super) fn check_swizzle_store(
        &self,
        base: &CheckedExpr<'a>,
        lanes: &[u8],
        op: AssignOp,
        rhs: &Expr<'a>,
        interner: &Interner,
        span: Span,
        swizzle: &CheckedExpr<'a>,
    ) -> Result<CheckedExpr<'a>, CompileError> {
        if op != AssignOp::Assign {
            return Err(CompileError::new(
                "compound assignment to a swizzle is not supported; write `v.xy = v.xy + e`",
                span,
            ));
        }
        if !self.is_lvalue(base) {
            return Err(CompileError::new("assignment to a non-lvalue", base.span));
        }
        if base.ty.is_const() {
            return Err(self.readonly_error(base, interner));
        }
        let rhs = self.check_expr(rhs, interner)?;
        let elem = match swizzle.ty.unqualified() {
            Ty::Vector { elem, .. } => (**elem).clone(),
            ty => ty.clone(),
        };
        // HLSL-style scalar stores splat: `v.xy = 1.0f` writes both
        // lanes. Vector rhs must match the swizzle's width.
        let rhs = if is_arithmetic(&rhs.ty) && !is_vector(&rhs.ty) {
            let converted = self.convert(rhs, elem);
            CheckedExpr {
                ty: swizzle.ty.clone(),
                span: converted.span,
                kind: CheckedExprKind::Splat { expr: self.alloc(converted) },
            }
        } else {
            self.convert_for_assignment(&swizzle.ty, rhs, interner)?
        };
        Ok(CheckedExpr {
            ty: swizzle.ty.clone(),
            span,
            kind: CheckedExprKind::SwizzleStore {
                base: self.alloc(base.clone()),
                lanes: self.alloc_slice(lanes.to_vec()),
                rhs: self.alloc(rhs),
            },
        })
    }

    /// `tex2D(sampler, coord)` (`-fprism`): the sampler argument must
    /// name a file-scope `sampler2D` object (resources do not flow as
    /// values), and the coordinates a `float2`. The node carries the
    /// resource binding directly.
    pub(super) fn check_tex2d(&self, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let [sampler, coord] = args else {
            return Err(CompileError::new("`tex2D` takes (sampler2D, float2)", span));
        };
        let Expr::Ident { name, .. } = sampler else {
            return Err(CompileError::new("`tex2D`'s first argument must name a file-scope `sampler2D`", sampler.span()));
        };
        let Some(id) = self.lookup_symbol(*name) else {
            let text = interner.resolve(*name).unwrap_or("?");
            return Err(CompileError::new(format!("unknown sampler '{text}'"), sampler.span()));
        };
        let binding = &self.decls[id.0 as usize];
        if binding.kind != BindingKind::Var || !self.is_prism_sampler_ty(&binding.ty, interner) {
            return Err(CompileError::new("`tex2D`'s first argument must be a file-scope `sampler2D`", sampler.span()));
        }
        let coord = self.check_expr(coord, interner)?;
        match coord.ty.unqualified() {
            Ty::Vector { elem, len } if **elem == Ty::Float && *len == 2 => {}
            _ => {
                return Err(CompileError::new("`tex2D` coordinates must be a `float2`", coord.span));
            }
        }
        Ok(CheckedExpr {
            ty: Ty::Vector { elem: Box::new(Ty::Float), len: 4 },
            span,
            kind: CheckedExprKind::PrismSample { global: id, coord: self.alloc(coord) },
        })
    }

    /// Whether `sym` resolves to a function with a definition in this
    /// translation unit: a user-defined `tex2D` beats the builtin.
    pub(super) fn callee_is_defined(&self, sym: SymbolU32) -> bool {
        self.lookup_symbol(sym)
            .and_then(|id| self.decls.get(id.0 as usize))
            .is_some_and(|b| b.kind == BindingKind::Func && b.defined)
    }
}

impl Sema<'_> {
    /// Validate the whole translation unit against the prism profile.
    /// Returns driver-renderable errors; an empty result means the
    /// unit may proceed to MIR lowering.
    pub fn prism_validate(&self, tu: &TranslationUnit, interner: &Interner) -> Vec<CompileError> {
        let mut errs = Vec::new();
        let mut check = PrismCheck {
            sema: self,
            interner,
            errs: &mut errs,
            calls: HashMap::default(),
            current_fn: None,
        };
        check.run(tu);
        errs
    }

    /// Whether `ty` names the opaque sampler type (`sampler2D` and any
    /// typedef of it). The backend treats file-scope objects of this
    /// type as host-bound resources.
    pub fn is_prism_sampler_ty(&self, ty: &Ty, interner: &Interner) -> bool {
        let Ty::Struct(id) = ty.unqualified() else { return false };
        self.structs[id.0 as usize]
            .tag
            .is_some_and(|tag| interner.resolve(tag) == Some(SAMPLER2D_TAG))
    }
}

struct PrismCheck<'a, 's> {
    sema: &'s Sema<'a>,
    interner: &'s Interner,
    errs: &'s mut Vec<CompileError>,
    /// Call edges by caller: callee plus the call's span, for the
    /// recursion diagnostic.
    calls: HashMap<DeclId, Vec<(DeclId, Span)>>,
    /// The function currently being walked (recursion edges and
    /// diagnostics).
    current_fn: Option<DeclId>,
}

impl PrismCheck<'_, '_> {
    fn name(&self, sym: SymbolU32) -> &str {
        self.interner.resolve(sym).unwrap_or("?")
    }

    fn name_of(&self, id: DeclId) -> &str {
        self.sema.decls[id.0 as usize].name.map(|n| self.name(n)).unwrap_or("?")
    }

    fn is_sampler(&self, ty: &Ty) -> bool {
        self.sema.is_prism_sampler_ty(ty, self.interner)
    }

    fn err(&mut self, msg: impl Into<String>, span: Span) {
        self.errs.push(CompileError::new(msg, span));
    }

    fn run(&mut self, tu: &TranslationUnit) {
        if let Some((_, span)) = self.sema.toplevel_asm.first() {
            self.err("inline asm is not allowed in prism shaders", *span);
        }
        // Every struct type ever declared: members must be shader-safe
        // (anonymous members are their own StructDefs, so one flat walk
        // covers nesting).
        for def in &self.sema.structs {
            for member in &def.members {
                self.check_value_ty(&member.ty, member.span, "struct member");
            }
        }
        for ed in &tu.external_declarations {
            let id = match ed {
                ExternalDeclaration::Var(id) | ExternalDeclaration::Func(id) => *id,
            };
            let binding = &self.sema.decls[id.0 as usize];
            match binding.kind {
                BindingKind::Func => {
                    self.check_fn(&binding.ty, binding.span);
                    if binding.defined {
                        self.walk_fn(id.0);
                    }
                }
                BindingKind::Var => {
                    self.check_value_ty(&binding.ty, binding.span, "object");
                    if binding.tls {
                        self.err("thread-local storage is not allowed in prism shaders", binding.span);
                    }
                    if self.is_sampler(&binding.ty) {
                        // A file-scope `sampler2D x;` is a tentative
                        // definition (so `defined` is already true);
                        // what the host forbids is an initializer.
                        if self.sema.inits.contains_key(&id) {
                            self.err(
                                format!("sampler '{}' must be declared without an initializer; the host binds it", self.name_of(id)),
                                binding.span,
                            );
                        }
                    } else if let Some(init) = self.sema.inits.get(&id) {
                        self.walk_init(init);
                    }
                }
                BindingKind::Typedef | BindingKind::EnumConst { .. } => {}
            }
        }
        self.check_recursion();
        self.check_entry_points(tu.span);
    }

    /// A function's own signature: return type and parameter types are
    /// value positions, so function-pointer parameters are rejected
    /// here; variadic prototypes have no shader meaning.
    fn check_fn(&mut self, ty: &Ty, span: Span) {
        let Ty::Func { ret, params, variadic, prototyped: _ } = ty.unqualified() else { return };
        if *variadic {
            self.err("variadic functions are not allowed in prism shaders", span);
        }
        self.check_value_ty(ret, span, "return type");
        for param in params {
            self.check_value_ty(param, span, "parameter");
        }
        if self.is_sampler(ret) {
            self.err("samplers cannot be returned from prism functions", span);
        }
    }

    /// Recursive type check for a value position: object, parameter,
    /// return, member, cast, or initializer target. `what` names the
    /// position for diagnostics.
    fn check_value_ty(&mut self, ty: &Ty, span: Span, what: &str) {
        match ty.unqualified() {
            Ty::Pointer { .. } => {
                self.err(format!("pointers are not allowed in prism shaders ({what})"), span);
            }
            Ty::Func { .. } => {
                self.err(format!("function pointers are not allowed in prism shaders ({what})"), span);
            }
            Ty::Double | Ty::LongDouble => {
                self.err(format!("double precision is not supported in prism shaders ({what}); use float"), span);
            }
            Ty::Float128 => {
                self.err(format!("`__float128` is not allowed in prism shaders ({what})"), span);
            }
            Ty::Complex(_) => {
                self.err(format!("`_Complex` is not allowed in prism shaders ({what})"), span);
            }
            Ty::BuiltinVaList { .. } => {
                self.err(format!("`va_list` is not allowed in prism shaders ({what})"), span);
            }
            Ty::Long | Ty::UnsignedLong | Ty::LongLong | Ty::UnsignedLongLong | Ty::Int128 | Ty::UnsignedInt128 => {
                self.err(format!("64-bit integers are not supported in prism shaders ({what})"), span);
            }
            Ty::BitInt { width, .. } if *width > 32 => {
                self.err(format!("bit-precise integers wider than 32 bits are not supported in prism shaders ({what})"), span);
            }
            Ty::Vector { elem, .. } => self.check_value_ty(elem, span, what),
            Ty::Array { elem, bound, .. } => {
                if matches!(bound, Some(ArrayBound::Vla(_) | ArrayBound::Star)) {
                    self.err(format!("variable-length arrays are not allowed in prism shaders ({what})"), span);
                }
                self.check_value_ty(elem, span, what);
            }
            // Unreachable through `unqualified()`, kept for exhaustiveness.
            Ty::Qualified { inner, .. } => self.check_value_ty(inner, span, what),
            Ty::Void | Ty::Bool | Ty::Char | Ty::SignedChar | Ty::UnsignedChar | Ty::Short | Ty::UnsignedShort
            | Ty::Int | Ty::UnsignedInt | Ty::BitInt { .. } | Ty::Nullptr | Ty::Float | Ty::Struct(_) | Ty::Enum(_) => {}
        }
    }

    fn walk_fn(&mut self, id: u32) {
        let did = DeclId(id);
        let Some(body) = self.sema.bodies.get(&did) else { return };
        let saved = self.current_fn;
        self.current_fn = Some(did);
        for param in body.params {
            let binding = &self.sema.decls[param.0 as usize];
            self.check_value_ty(&binding.ty, binding.span, "parameter");
            if self.is_sampler(&binding.ty) {
                self.err("samplers cannot be passed as parameters; reference the file-scope sampler directly", binding.span);
            }
        }
        self.walk_compound(&body.block);
        self.current_fn = saved;
    }

    fn walk_decl_ids(&mut self, ids: &[DeclId], span: Span) {
        for id in ids {
            let binding = &self.sema.decls[id.0 as usize];
            self.check_value_ty(&binding.ty, span, "object");
            if self.is_sampler(&binding.ty) {
                self.err("samplers can only be declared at file scope", span);
            } else if let Some(init) = self.sema.inits.get(id) {
                self.walk_init(init);
            }
        }
    }

    fn walk_compound(&mut self, compound: &Compound<'_>) {
        for item in compound.items {
            match item {
                BlockItem::Decl { ids, span } => self.walk_decl_ids(ids, *span),
                BlockItem::Stmt(stmt) => self.walk_stmt(stmt),
            }
        }
    }

    fn walk_stmt(&mut self, stmt: &Stmt<'_>) {
        match stmt {
            Stmt::Compound(compound) => self.walk_compound(compound),
            Stmt::Decl { ids, span } => self.walk_decl_ids(ids, *span),
            Stmt::Empty { .. } | Stmt::Break { .. } | Stmt::Continue { .. } => {}
            Stmt::Expr { expr, .. } => self.walk_expr(expr),
            Stmt::Asm(asm) => self.err("inline asm is not allowed in prism shaders", asm.span),
            Stmt::Return { expr, span } => {
                if let Some(expr) = expr {
                    if self.is_sampler(&expr.ty) {
                        self.err("samplers cannot be returned from prism functions", *span);
                    }
                    self.walk_expr(expr);
                }
            }
            Stmt::If { cond, then, els, .. } => {
                self.walk_expr(cond);
                self.walk_stmt(then);
                if let Some(els) = els {
                    self.walk_stmt(els);
                }
            }
            Stmt::While { cond, body, .. } => {
                self.walk_expr(cond);
                self.walk_stmt(body);
            }
            Stmt::DoWhile { body, cond, .. } => {
                self.walk_stmt(body);
                self.walk_expr(cond);
            }
            Stmt::For { init, cond, step, body, .. } => {
                match init {
                    ForInit::Decl(ids) => self.walk_decl_ids(ids, stmt.span()),
                    ForInit::Expr(expr) => self.walk_expr(expr),
                    ForInit::Empty => {}
                }
                if let Some(cond) = cond {
                    self.walk_expr(cond);
                }
                if let Some(step) = step {
                    self.walk_expr(step);
                }
                self.walk_stmt(body);
            }
            Stmt::Switch { cond, body, .. } => {
                self.walk_expr(cond);
                self.walk_stmt(body);
            }
            Stmt::Goto { span, .. } => {
                self.err("'goto' is not allowed in prism shaders", *span);
            }
            Stmt::GotoComputed { span, .. } => {
                self.err("computed goto is not allowed in prism shaders", *span);
            }
            Stmt::Label { stmt, .. } => self.walk_stmt(stmt),
            Stmt::Case { expr, hi, stmt, .. } => {
                if let Some(expr) = expr {
                    self.walk_expr(expr);
                }
                if let Some(hi) = hi {
                    self.walk_expr(hi);
                }
                self.walk_stmt(stmt);
            }
        }
    }

    fn walk_init(&mut self, init: &Initializer<'_>) {
        match init {
            Initializer::Expr(expr) => self.walk_expr(expr),
            Initializer::List { items, .. } => {
                for item in items.iter() {
                    self.walk_init(item.init);
                }
            }
        }
    }

    fn walk_expr(&mut self, expr: &CheckedExpr<'_>) {
        match &expr.kind {
            CheckedExprKind::Ident(_) => {}
            CheckedExprKind::Unary(UnaryOp::AddrOf, inner) => {
                self.err("the address-of operator is not allowed in prism shaders (no pointers exist)", expr.span);
                self.walk_expr(inner);
            }
            CheckedExprKind::Unary(UnaryOp::Deref, inner) => {
                self.err("pointer dereference is not allowed in prism shaders", expr.span);
                self.walk_expr(inner);
            }
            CheckedExprKind::Unary(_, inner) => self.walk_expr(inner),
            CheckedExprKind::Binary(_, lhs, rhs) => {
                self.walk_expr(lhs);
                self.walk_expr(rhs);
            }
            CheckedExprKind::Assign(_, lhs, rhs) => {
                self.check_assign_target(lhs, expr.span);
                self.walk_expr(lhs);
                self.walk_expr(rhs);
            }
            CheckedExprKind::CompoundAssign { lhs, rhs, .. } => {
                self.check_assign_target(lhs, expr.span);
                self.walk_expr(lhs);
                self.walk_expr(rhs);
            }
            CheckedExprKind::Ternary { cond, then, els } => {
                self.walk_expr(cond);
                self.walk_expr(then);
                self.walk_expr(els);
            }
            CheckedExprKind::Elvis { cond, els } => {
                self.walk_expr(cond);
                self.walk_expr(els);
            }
            CheckedExprKind::Call { func, args } => self.walk_call(func, args, expr.span),
            CheckedExprKind::Member { base, arrow, .. } => {
                if *arrow {
                    self.err("'->' is not allowed in prism shaders (no pointers exist)", expr.span);
                }
                self.walk_expr(base);
            }
            CheckedExprKind::Subscript { base, index } => {
                if matches!(base.ty.unqualified(), Ty::Pointer { .. }) {
                    self.err("subscripting a pointer is not allowed in prism shaders", expr.span);
                }
                self.walk_expr(base);
                self.walk_expr(index);
            }
            CheckedExprKind::Cast { ty, expr: inner } => {
                self.check_value_ty(ty, expr.span, "cast");
                self.walk_expr(inner);
            }
            CheckedExprKind::Convert { expr: inner, ty } => {
                self.check_value_ty(ty, expr.span, "conversion");
                self.walk_expr(inner);
            }
            CheckedExprKind::CompoundLiteral { init, .. } => self.walk_init(init),
            CheckedExprKind::Alloca { .. } => {
                self.err("alloca is not allowed in prism shaders (no dynamic allocation)", expr.span);
            }
            CheckedExprKind::FrameAddress { .. } => {
                self.err("frame addresses are not allowed in prism shaders", expr.span);
            }
            CheckedExprKind::VaArg { .. } => {
                self.err("va_arg is not allowed in prism shaders", expr.span);
            }
            CheckedExprKind::LabelAddr(..) => {
                self.err("label addresses are not allowed in prism shaders", expr.span);
            }
            CheckedExprKind::Await(_) => {
                self.err("await is not allowed in prism shaders", expr.span);
            }
            CheckedExprKind::StmtExpr { items } => {
                for item in *items {
                    match item {
                        BlockItem::Decl { ids, span } => self.walk_decl_ids(ids, *span),
                        BlockItem::Stmt(stmt) => self.walk_stmt(stmt),
                    }
                }
            }
            // Leaves: literals, folded builtins, type queries.
            CheckedExprKind::Int(_)
            | CheckedExprKind::Float(_)
            | CheckedExprKind::Char(_)
            | CheckedExprKind::Str(_)
            | CheckedExprKind::ConstantP(_)
            | CheckedExprKind::ObjectSize(_)
            | CheckedExprKind::ClassifyType(_)
            | CheckedExprKind::StaticAssert
            | CheckedExprKind::Offsetof { .. }
            | CheckedExprKind::SizeofTy(_)
            | CheckedExprKind::CountofTy(_)
            | CheckedExprKind::AlignofTy(_)
            | CheckedExprKind::GenericTy { .. } => {}
            CheckedExprKind::SizeofExpr(inner)
            | CheckedExprKind::CountofExpr(inner)
            | CheckedExprKind::AlignofExpr(inner) => self.walk_expr(inner),
            CheckedExprKind::Shuffle { args } | CheckedExprKind::ShuffleVector { args, .. } => {
                for arg in args.iter() {
                    self.walk_expr(arg);
                }
            }
            CheckedExprKind::Atomic { args, .. } | CheckedExprKind::Ia32 { args, .. } => {
                for arg in args.iter() {
                    self.walk_expr(arg);
                }
            }
            CheckedExprKind::Overflow { lhs, rhs, res, .. } => {
                self.err("overflow builtins are not allowed in prism shaders", expr.span);
                self.walk_expr(lhs);
                self.walk_expr(rhs);
                self.walk_expr(res);
            }
            CheckedExprKind::Clz { arg, fallback, .. } => {
                self.walk_expr(arg);
                if let Some(fallback) = fallback {
                    self.walk_expr(fallback);
                }
            }
            CheckedExprKind::Bswap { arg, .. } => self.walk_expr(arg),
            CheckedExprKind::Rotate { arg, count, .. } => {
                self.walk_expr(arg);
                self.walk_expr(count);
            }
            CheckedExprKind::FpClass { arg, .. } => self.walk_expr(arg),
            CheckedExprKind::FpClassify { codes, arg } => {
                for code in codes.iter().copied() {
                    self.walk_expr(code);
                }
                self.walk_expr(arg);
            }
            CheckedExprKind::Generic { ctrl, assocs, .. } => {
                self.walk_expr(ctrl);
                for (_, assoc) in assocs.iter() {
                    self.walk_expr(assoc);
                }
            }
            // prism nodes: a swizzle read is a lane permutation of its
            // base; a swizzle store writes through the base lvalue
            // (uniforms are rejected at the assignment below); a
            // sample names a file-scope resource directly.
            CheckedExprKind::Swizzle { base, .. } => self.walk_expr(base),
            CheckedExprKind::SwizzleStore { base, lanes: _, rhs } => {
                self.check_assign_target(base, expr.span);
                self.walk_expr(base);
                self.walk_expr(rhs);
            }
            CheckedExprKind::PrismSample { global: _, coord } => self.walk_expr(coord),
            CheckedExprKind::Splat { expr } => self.walk_expr(expr),
        }
    }

    fn check_assign_target(&mut self, lhs: &CheckedExpr<'_>, span: Span) {
        if self.is_sampler(&lhs.ty) {
            self.err("samplers cannot be assigned; they are host-bound resources", span);
        }
    }

    fn walk_call(&mut self, func: &CheckedExpr<'_>, args: &[CheckedExpr<'_>], span: Span) {
        // A sampler argument can only be legal through `tex2D`, which
        // the checker intercepts into its own node before a `Call`
        // exists — any remaining call taking a sampler is misuse.
        for arg in args {
            if self.is_sampler(&arg.ty) {
                self.err("samplers cannot be passed as parameters; reference the file-scope sampler directly", arg.span());
            }
        }
        let CheckedExprKind::Ident(id) = func.kind else {
            self.err("calling through anything but a named function is not allowed in prism shaders", span);
            self.walk_expr(func);
            return;
        };
        let binding = &self.sema.decls[id.0 as usize];
        if binding.kind != BindingKind::Func {
            self.walk_expr(func);
            return;
        }
        if !binding.defined {
            let name = self.name_of(id);
            // Compiler builtins stay allowed: they lower to MIR
            // operations or fold at check time, never to calls out of
            // the shader. `prism.h`'s scalar builtins (`min`, `sqrtf`,
            // ...) ride the same path in prism mode.
            let allowed = name.starts_with("__builtin_")
                || (self.sema.lang.prism && PRISM_SCALAR_BUILTINS.contains(&name));
            if !allowed {
                self.err(
                    format!("call to unsupported function '{name}': prism shaders cannot call external code"),
                    span,
                );
            }
            return;
        }
        if let Some(caller) = self.current_fn {
            self.calls.entry(caller).or_default().push((id, span));
        }
    }

    /// Reject call-graph cycles: shaders are acyclic.
    fn check_recursion(&mut self) {
        let mut state: HashMap<DeclId, u8> = HashMap::default();
        for caller in self.calls.keys().copied().collect::<Vec<_>>() {
            if let Some(span) = self.find_cycle(caller, &mut state, HashSet::default()) {
                self.err("recursion is not allowed in prism shaders", span);
                return;
            }
        }
    }

    fn find_cycle(&self, node: DeclId, state: &mut HashMap<DeclId, u8>, mut path: HashSet<DeclId>) -> Option<Span> {
        match state.get(&node) {
            // Back edge into the current path: report at the call that
            // closes the cycle.
            Some(1) => {
                let edges = self.calls.get(&node)?;
                return edges.iter().map(|(_, span)| *span).next();
            }
            Some(_) => return None,
            None => {}
        }
        state.insert(node, 1);
        path.insert(node);
        for (callee, _) in self.calls.get(&node).map(Vec::as_slice).unwrap_or(&[]) {
            if let Some(span) = self.find_cycle(*callee, state, path.clone()) {
                return Some(span);
            }
        }
        state.insert(node, 2);
        None
    }

    /// Entry points (`[[prism::vertex]]` / `[[prism::fragment]]`): at
    /// least one per unit, each defined with external linkage, taking
    /// and returning interface structs whose scalar/vector leaves all
    /// carry a semantic.
    fn check_entry_points(&mut self, tu_span: Span) {
        let mut stages: Vec<(DeclId, PrismStage, Span)> = Vec::new();
        for (id, attrs) in &self.sema.prism_attrs {
            if let Some(stage) = attrs.stage {
                let binding = &self.sema.decls[id.0 as usize];
                stages.push((*id, stage, binding.span));
            }
        }
        if stages.is_empty() {
            self.err(
                "no prism entry point: mark one with [[prism::vertex]] or [[prism::fragment]]",
                tu_span,
            );
            return;
        }
        for (id, _stage, span) in stages {
            let binding = &self.sema.decls[id.0 as usize];
            let name = self.name_of(id).to_string();
            if !binding.defined {
                self.err(format!("entry point '{name}' has no definition"), span);
                continue;
            }
            if binding.linkage == crate::sema::Linkage::Internal {
                self.err(format!("entry point '{name}' must have external linkage (drop `static`)"), span);
            }
            let Ty::Func { ret, params, .. } = binding.ty.unqualified() else {
                continue;
            };
            self.check_interface_ty(ret, span, format!("'{name}' return type"));
            for param in params {
                self.check_interface_ty(param, span, format!("'{name}' parameter"));
            }
        }
    }

    /// An entry point's inputs and outputs are interface structs (or
    /// `void` for a no-input parameter list); every scalar or vector
    /// leaf names its hardware binding.
    fn check_interface_ty(&mut self, ty: &Ty, span: Span, what: String) {
        match ty.unqualified() {
            Ty::Void => {}
            Ty::Struct(id) => self.check_interface_leaves(*id, span),
            _ => {
                self.err(format!("{what} must be an interface struct with [[prism::semantic]] members"), span);
            }
        }
    }

    fn check_interface_leaves(&mut self, id: StructId, span: Span) {
        let member_count = self.sema.structs[id.0 as usize].members.len();
        for index in 0..member_count {
            let member = &self.sema.structs[id.0 as usize].members[index];
            if member.name.is_none()
                && let Ty::Struct(inner) = member.ty.unqualified()
            {
                self.check_interface_leaves(*inner, member.span);
                continue;
            }
            if member.prism.semantic.is_none() {
                let name = member.name.map(|n| self.name(n)).unwrap_or("?");
                self.err(format!("interface member '{name}' needs a [[prism::semantic(...)]]"), member.span.merge(span));
            }
        }
    }
}
