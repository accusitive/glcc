//! Item collection: the surface program to resolved definitions.
//!
//! Responsibilities:
//!
//! * *Kinds*: every syntactic type expression resolves to a semantic
//!   `Ty` while kinds are inferred through a shared union-find over
//!   kind variables. A data or class type declares its parameter kinds
//!   by usage — `data Free f a` discovers that `f` is `* -> *` where it
//!   applies `f` inside a constructor — and later items refine those
//!   kinds forward, so declaration order does not matter. Leftover kind
//!   variables default to `*`.
//! * *Structure*: datatypes, classes, and instances land here with
//!   resolved signatures; function and value bodies stay in the AST for
//!   the checker ([`crate::check`]), which generalizes them in one pass
//!   over the items in declaration order.
//!
//! Prelude items carry `Span`s tagged with [`crate::PRELUDE_TAG`];
//! every later pass treats those as built-in.

use std::collections::HashMap;

use crate::ast;
use crate::span::{Diag, KResult, Span};
use crate::ty::{Kind, Ty};

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u32);

        impl $name {
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

id_type!(DataId);
id_type!(ClassId);
id_type!(FnId);
id_type!(ValueId);

/// A value scheme once the checker generalized it.
#[derive(Debug, Clone)]
pub struct Scheme {
    /// Globally unique parameter names (`t$<varid>`), in first-use
    /// order, with their kinds. `$` cannot appear in source names, so
    /// scheme parameters never collide with user-written ones.
    pub vars: Vec<(String, Kind)>,
    pub ty: Ty,
}

impl Scheme {
    pub fn monomorphic(ty: Ty) -> Scheme {
        Scheme {
            vars: Vec::new(),
            ty,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CtorDef {
    pub span: Span,
    pub name: String,
    pub args: Vec<Ty>,
}

#[derive(Debug, Clone)]
pub struct DataDef {
    pub span: Span,
    pub name: String,
    /// Parameter names with kinds, in declaration order.
    pub params: Vec<(String, Kind)>,
    pub ctors: Vec<CtorDef>,
}

#[derive(Debug, Clone)]
pub struct MethodDef {
    pub span: Span,
    pub name: String,
    /// Every free type variable of the signature, in first-use order,
    /// with kinds. Scheme parameter names are taken verbatim from here,
    /// so call sites and instance bodies use the same names.
    pub vars: Vec<(String, Kind)>,
    pub ty: Ty,
}

#[derive(Debug, Clone)]
pub struct ClassDef {
    pub span: Span,
    pub name: String,
    /// Class parameters, in declaration order, with kinds.
    pub params: Vec<(String, Kind)>,
    pub methods: Vec<MethodDef>,
}

impl ClassDef {
    pub fn method(&self, name: &str) -> Option<usize> {
        self.methods.iter().position(|m| m.name == name)
    }

    pub fn param_kind(&self, name: &str) -> Option<&Kind> {
        self.params
            .iter()
            .find(|(p, _)| p == name)
            .map(|(_, k)| k)
    }
}

/// `when C args` on an instance: `args` are the class's parameters in
/// declaration order (the first one is the "self" position).
#[derive(Debug, Clone)]
pub struct Bound {
    pub class_id: ClassId,
    pub args: Vec<Ty>,
    pub span: Span,
}

/// One method body of an instance; the checker walks each once and
/// lowering repeats it per instance.
#[derive(Debug, Clone)]
pub struct ImplMethod {
    pub name: String,
    pub params: Vec<ast::Pat>,
    pub body: ast::Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct InstanceDef {
    pub span: Span,
    pub class_id: ClassId,
    /// Every free type parameter of the head and the constraints, in
    /// first-use order, with kinds resolved.
    pub generics: Vec<(String, Kind)>,
    /// The class's parameter instantiation: instantiating the class
    /// sets parameter `i` to this type. Classes have at least one
    /// parameter, so `args[0]` is the dispatching type.
    pub args: Vec<Ty>,
    pub bounds: Vec<Bound>,
    pub methods: Vec<ImplMethod>,
}

impl InstanceDef {
    /// The dispatching class argument (`args[0]`).
    pub fn self_ty(&self) -> &Ty {
        &self.args[0]
    }
}

#[derive(Debug, Clone)]
pub struct FnDef {
    pub span: Span,
    pub name: String,
    pub params: Vec<ast::Pat>,
    pub body: ast::Expr,
    /// Filled by the checker after the body generalizes.
    pub scheme: Option<Scheme>,
    pub is_prelude: bool,
}

#[derive(Debug, Clone)]
pub struct ValueDef {
    pub span: Span,
    pub name: String,
    pub body: ast::Expr,
    pub scheme: Option<Scheme>,
    pub is_prelude: bool,
}

pub struct Defs {
    pub data: Vec<DataDef>,
    pub data_ids: HashMap<String, DataId>,
    pub classes: Vec<ClassDef>,
    pub class_ids: HashMap<String, ClassId>,
    pub instances: Vec<InstanceDef>,
    pub fns: Vec<FnDef>,
    pub fn_ids: HashMap<String, FnId>,
    pub values: Vec<ValueDef>,
    pub value_ids: HashMap<String, ValueId>,
    /// Constructor name → (its data type, constructor index).
    pub ctor_ids: HashMap<String, (DataId, u32)>,
    /// Saturated argument counts for type constructors: data types, the
    /// primitives from [`Ty::intros()`], and class heads.
    pub arity: HashMap<String, usize>,
    /// Resolved kinds of type heads (`Named` forms only).
    pub kinds_of: HashMap<String, Kind>,
}

impl Defs {
    pub fn data_by_name(&self, name: &str) -> Option<DataId> {
        self.data_ids.get(name).copied()
    }

    pub fn class_by_name(&self, name: &str) -> Option<ClassId> {
        self.class_ids.get(name).copied()
    }

    pub fn fn_by_name(&self, name: &str) -> Option<FnId> {
        self.fn_ids.get(name).copied()
    }

    pub fn value_by_name(&self, name: &str) -> Option<ValueId> {
        self.value_ids.get(name).copied()
    }

    pub fn ctor_by_name(&self, name: &str) -> Option<(DataId, u32)> {
        self.ctor_ids.get(name).copied()
    }

    pub fn arity_of(&self, name: &str) -> Option<usize> {
        self.arity.get(name).copied()
    }

    pub fn kind_of_head(&self, name: &str) -> Option<&Kind> {
        self.kinds_of.get(name)
    }

    /// A prelude item's span is tagged beyond any source offset.
    pub fn is_prelude_span(&self, span: Span) -> bool {
        span.lo >= crate::PRELUDE_TAG
    }
}

// ------------------------------------------------------------------ kinds --

/// Kind vocabulary while inferring; only [`Kind`] survives outside this
/// module.
#[derive(Debug, Clone)]
pub enum RawKind {
    Star,
    Arrow(Box<RawKind>, Box<RawKind>),
    Var(u32),
}

impl RawKind {
    pub fn display(&self) -> String {
        match self {
            RawKind::Star => "*".to_string(),
            RawKind::Arrow(a, b) => format!("{} -> {}", a.display(), b.display()),
            RawKind::Var(v) => format!("k{v}"),
        }
    }
}

/// Union-find over kind variables, shared by the whole collection.
/// Kind expressions carry kind variable ids by value; the table here
/// owns their bindings. The invariant maintained by [`Kinds::link`] is
/// that the binding graph is acyclic.
#[derive(Default)]
struct Kinds {
    nodes: Vec<Option<RawKind>>,
}

impl Kinds {
    fn var(&mut self) -> RawKind {
        self.nodes.push(None);
        RawKind::Var((self.nodes.len() - 1) as u32)
    }

    /// Follows variable bindings until the outermost constructor.
    fn shallow(&self, k: &RawKind) -> RawKind {
        let mut cur = k.clone();
        loop {
            match cur {
                RawKind::Var(v) => match self.nodes[v as usize].clone() {
                    Some(bound) => cur = bound,
                    None => return RawKind::Var(v),
                },
                other => return other,
            }
        }
    }

    /// Whether `v` is reachable from `k` through bindings.
    fn occurs(&self, v: u32, k: &RawKind) -> bool {
        match self.shallow(k) {
            RawKind::Star => false,
            RawKind::Arrow(a, b) => self.occurs(v, &a) || self.occurs(v, &b),
            RawKind::Var(w) => w == v,
        }
    }

    fn link(&mut self, v: u32, k: &RawKind, span: Span) -> KResult<()> {
        match self.nodes[v as usize].clone() {
            Some(existing) => self.unify(&existing, k, span),
            None => {
                if self.occurs(v, k) {
                    return Err(Diag::new(
                        format!("the kind `{}` would contain itself", k.display()),
                        span,
                    ));
                }
                self.nodes[v as usize] = Some(k.clone());
                Ok(())
            }
        }
    }

    fn unify(&mut self, a: &RawKind, b: &RawKind, span: Span) -> KResult<()> {
        let a = self.shallow(a);
        let b = self.shallow(b);
        match (&a, &b) {
            (RawKind::Var(x), RawKind::Var(y)) if x == y => Ok(()),
            (RawKind::Var(x), _) => self.link(*x, &b, span),
            (_, RawKind::Var(y)) => self.link(*y, &a, span),
            (RawKind::Arrow(a1, b1), RawKind::Arrow(a2, b2)) => {
                self.unify(a1, a2, span)?;
                self.unify(b1, b2, span)
            }
            (RawKind::Star, RawKind::Star) => Ok(()),
            _ => Err(Diag::new(
                format!("kinds do not match: `{}` vs `{}`", a.display(), b.display()),
                span,
            )),
        }
    }

    /// Result kind of applying `head` to an argument of kind `arg`.
    fn apply(&mut self, head: &RawKind, arg: &RawKind, span: Span) -> KResult<RawKind> {
        let arg = self.shallow(arg);
        match self.shallow(head) {
            RawKind::Arrow(dom, rest) => {
                self.unify(&dom, &arg, span)?;
                Ok((*rest).clone())
            }
            RawKind::Var(v) => {
                let dom = self.var();
                let rest = self.var();
                self.link(
                    v,
                    &RawKind::Arrow(Box::new(dom.clone()), Box::new(rest.clone())),
                    span,
                )?;
                self.unify(&dom, &arg, span)?;
                Ok(rest)
            }
            other => Err(Diag::new(
                format!(
                    "cannot apply an argument of kind `{}` to a type of kind `{}`",
                    arg.display(),
                    other.display()
                ),
                span,
            )),
        }
    }

    /// Final kind of a raw kind; leftover kind variables default to `*`.
    fn to_kind(&self, k: &RawKind) -> Kind {
        match self.shallow(k) {
            RawKind::Star => Kind::star(),
            RawKind::Arrow(a, b) => Kind::arrow(self.to_kind(&a), self.to_kind(&b)),
            RawKind::Var(_) => Kind::star(),
        }
    }

    /// The `i`th domain of an arrow chain, following bindings.
    fn nth_domain(&self, k: &RawKind, i: usize) -> Option<RawKind> {
        let mut cur = self.shallow(k);
        for _ in 0..i {
            cur = match cur {
                RawKind::Arrow(_, rest) => self.shallow(&rest),
                _ => return None,
            };
        }
        match cur {
            RawKind::Arrow(dom, _) => Some((*dom).clone()),
            _ => None,
        }
    }
}

// --------------------------------------------------------------- resolver --

struct ResolveCtx {
    kinds: Kinds,
    /// Raw kinds of type heads, shared across the collection.
    kinds_of: HashMap<String, RawKind>,
    /// Saturated argument counts of type heads.
    arity: HashMap<String, usize>,
    /// Kind variables currently in item scope, one per type parameter.
    scope: HashMap<String, RawKind>,
}

impl ResolveCtx {
    fn kind_of_name(&self, name: &str, span: Span) -> KResult<RawKind> {
        self.kinds_of
            .get(name)
            .cloned()
            .ok_or_else(|| Diag::new(format!("cannot find type `{name}`"), span))
    }

    fn kind_of_ty(&mut self, ty: &Ty, span: Span) -> KResult<RawKind> {
        match ty {
            Ty::Named { name, args } => {
                let mut cur = self.kind_of_name(name, span)?;
                for arg in args {
                    let arg_kind = self.kind_of_ty(arg, span)?;
                    cur = self.kinds.apply(&cur, &arg_kind, span)?;
                }
                Ok(cur)
            }
            Ty::App(h, a) => {
                let head_kind = self.kind_of_ty(h, span)?;
                let arg_kind = self.kind_of_ty(a, span)?;
                self.kinds.apply(&head_kind, &arg_kind, span)
            }
            Ty::Arrow(..) | Ty::Tuple(_) => Ok(RawKind::Star),
            Ty::Param(p) => {
                if !self.scope.contains_key(p) {
                    let k = self.kinds.var();
                    self.scope.insert(p.clone(), k);
                }
                Ok(self.scope[p].clone())
            }
            Ty::Var(_) => Err(Diag::new(
                "internal: inference variable in a type signature",
                span,
            )),
        }
    }

    /// The kind of the `i`th parameter of the registered head `name`.
    fn peel_param(&self, name: &str, i: usize, span: Span) -> KResult<RawKind> {
        let head = self.kind_of_name(name, span)?;
        self.kinds.nth_domain(&head, i).ok_or_else(|| {
            Diag::new(format!("internal: `{name}` has no parameter {i}"), span)
        })
    }

    fn star_check(&mut self, ty: &Ty, span: Span) -> KResult<()> {
        let kind = self.kind_of_ty(ty, span)?;
        self.kinds.unify(&kind, &RawKind::Star, span)
    }

    fn resolve(&mut self, node: &ast::TyExpr) -> KResult<Ty> {
        match &node.kind {
            ast::TyExprKind::Name(name) => {
                if self.scope.contains_key(name) {
                    Ok(Ty::Param(name.clone()))
                } else if self.kinds_of.contains_key(name) {
                    Ok(Ty::named0(name))
                } else {
                    Err(Diag::new(format!("cannot find type `{name}`"), node.span))
                }
            }
            ast::TyExprKind::Paren(inner) => self.resolve(inner),
            ast::TyExprKind::Tuple(items) => {
                let mut tys = Vec::with_capacity(items.len());
                for item in items {
                    let ty = self.resolve(item)?;
                    self.star_check(&ty, item.span)?;
                    tys.push(ty);
                }
                Ok(Ty::Tuple(tys))
            }
            ast::TyExprKind::Arrow(a, b) => {
                let src = self.resolve(a)?;
                let dst = self.resolve(b)?;
                self.star_check(&src, a.span)?;
                self.star_check(&dst, b.span)?;
                Ok(Ty::arrow(src, dst))
            }
            ast::TyExprKind::App(f, a) => {
                let head = self.resolve(f)?;
                let arg = self.resolve(a)?;
                self.apply_ty(&head, &arg, node.span)
            }
        }
    }

    /// Fold an application into a saturated `Named` when the head has
    /// room for one more argument; otherwise build an `App` (which
    /// fails kind checking further up unless the head is a variable).
    fn apply_ty(&mut self, head: &Ty, arg: &Ty, span: Span) -> KResult<Ty> {
        let head_kind = self.kind_of_ty(head, span)?;
        let arg_kind = self.kind_of_ty(arg, span)?;
        self.kinds.apply(&head_kind, &arg_kind, span)?;
        match head {
            Ty::Named { name, args }
                if self.arity.get(name).is_some_and(|n| args.len() < *n) =>
            {
                let mut args = args.clone();
                args.push(arg.clone());
                Ok(Ty::named(name, args))
            }
            other => Ok(Ty::app(other.clone(), arg.clone())),
        }
    }
}

fn split_chain(node: &ast::TyExpr) -> (&ast::TyExpr, Vec<&ast::TyExpr>) {
    let mut args = Vec::new();
    let mut head = node;
    while let ast::TyExprKind::App(f, a) = &head.kind {
        args.push(&**a);
        head = &**f;
    }
    args.reverse();
    (head, args)
}

/// Names a bound introduces: `when C a b`.
fn parse_bound(ctx: &mut ResolveCtx, defs: &Defs, node: &ast::TyExpr) -> KResult<(ClassId, Vec<Ty>)> {
    let (head, args) = split_chain(node);
    let name = match &head.kind {
        ast::TyExprKind::Name(name) => name.clone(),
        _ => return Err(Diag::new("expected a class name in this bound", head.span)),
    };
    let Some(class_id) = defs.class_by_name(&name) else {
        return Err(Diag::new(
            format!("`{name}` is not a class (used as an instance bound)"),
            head.span,
        ));
    };
    let expected = defs.classes[class_id.index()].params.len();
    if args.len() != expected {
        return Err(Diag::new(
            format!(
                "class `{name}` takes {expected} type argument{}, but this bound has {}",
                if expected == 1 { "" } else { "s" },
                args.len()
            ),
            node.span,
        ));
    }
    let mut cur = ctx.kind_of_name(&name, node.span)?;
    let mut parts = Vec::with_capacity(args.len());
    for arg in args {
        let ty = ctx.resolve(arg)?;
        let kind = ctx.kind_of_ty(&ty, arg.span)?;
        cur = ctx.kinds.apply(&cur, &kind, arg.span)?;
        parts.push(ty);
    }
    ctx.kinds.unify(&cur, &RawKind::Star, node.span)?;
    Ok((class_id, parts))
}

/// The class-argument list of an instance head. Unary classes take the
/// whole written type; n-ary classes take an application chain of n
/// arguments.
fn instance_args(
    ctx: &mut ResolveCtx,
    defs: &Defs,
    class_id: ClassId,
    node: &ast::TyExpr,
) -> KResult<Vec<Ty>> {
    let class = &defs.classes[class_id.index()];
    let expected = class.params.len();
    if expected == 1 {
        return Ok(vec![ctx.resolve(node)?]);
    }
    let (head, args) = split_chain(node);
    if args.len() + 1 != expected {
        return Err(Diag::new(
            format!(
                "class `{}` takes {expected} type arguments, but this instance head has {}",
                class.name,
                args.len() + 1
            ),
            node.span,
        ));
    }
    let mut parts = Vec::with_capacity(expected);
    parts.push(ctx.resolve(head)?);
    for arg in args {
        parts.push(ctx.resolve(arg)?);
    }
    Ok(parts)
}

fn free_type_vars(node: &ast::TyExpr, kinds_of: &HashMap<String, RawKind>) -> Vec<String> {
    node.free_tyvars()
        .into_iter()
        .filter(|name| !kinds_of.contains_key(name))
        .collect()
}

fn fresh_head(kinds: &mut Kinds, params: usize) -> RawKind {
    let mut domains = Vec::with_capacity(params);
    for _ in 0..params {
        domains.push(kinds.var());
    }
    let mut head = RawKind::Star;
    for dom in domains.into_iter().rev() {
        head = RawKind::Arrow(Box::new(dom), Box::new(head));
    }
    head
}

// --------------------------------------------------------------- collect --

/// Function and value names become object-file symbols, a namespace
/// shared with the runtime (`rt/gc.c`), whose internals are all
/// `fern_`-prefixed. Constructors cannot collide: their names must be
/// capitalized.
fn check_reserved(name: &str, span: Span) -> KResult<()> {
    if name.starts_with("fern_") {
        return Err(Diag::new(
            format!("`{name}` is reserved: `fern_` names belong to the runtime"),
            span,
        ));
    }
    Ok(())
}

pub fn collect(program: &ast::Program) -> KResult<Defs> {
    let mut defs = Defs {
        data: Vec::new(),
        data_ids: HashMap::new(),
        classes: Vec::new(),
        class_ids: HashMap::new(),
        instances: Vec::new(),
        fns: Vec::new(),
        fn_ids: HashMap::new(),
        values: Vec::new(),
        value_ids: HashMap::new(),
        ctor_ids: HashMap::new(),
        arity: HashMap::new(),
        kinds_of: HashMap::new(),
    };

    let mut kinds = Kinds::default();
    let mut kinds_of: HashMap<String, RawKind> = HashMap::new();
    let mut arity: HashMap<String, usize> = HashMap::new();

    for prim in Ty::intros() {
        kinds_of.insert(prim.to_string(), RawKind::Star);
        arity.insert(prim.to_string(), 0);
    }

    // Register every item name so forward references resolve later, and
    // reject duplicates.
    for item in &program.items {
        match &item.kind {
            ast::ItemKind::Data(d) => {
                if Ty::intros().contains(&d.name.as_str()) {
                    return Err(Diag::new(
                        format!("`{}` is a built-in type", d.name),
                        d.span,
                    ));
                }
                if kinds_of.contains_key(&d.name) {
                    return Err(Diag::new(
                        format!("duplicate type name `{}`", d.name),
                        d.span,
                    ));
                }
                let id = DataId(defs.data.len() as u32);
                for (tag, ctor) in d.ctors.iter().enumerate() {
                    if defs.ctor_ids.contains_key(&ctor.name) {
                        return Err(Diag::new(
                            format!("duplicate constructor `{}`", ctor.name),
                            d.span,
                        ));
                    }
                    defs.ctor_ids.insert(ctor.name.clone(), (id, tag as u32));
                }
                defs.data.push(DataDef {
                    span: d.span,
                    name: d.name.clone(),
                    params: d.params.iter().map(|p| (p.clone(), Kind::star())).collect(),
                    ctors: Vec::new(),
                });
                defs.data_ids.insert(d.name.clone(), id);
                let head = fresh_head(&mut kinds, d.params.len());
                kinds_of.insert(d.name.clone(), head);
                arity.insert(d.name.clone(), d.params.len());
            }
            ast::ItemKind::Class(c) => {
                if Ty::intros().contains(&c.name.as_str()) {
                    return Err(Diag::new(
                        format!("`{}` is a built-in type", c.name),
                        c.span,
                    ));
                }
                if kinds_of.contains_key(&c.name) {
                    return Err(Diag::new(
                        format!("duplicate type name `{}`", c.name),
                        c.span,
                    ));
                }
                if c.params.is_empty() {
                    return Err(Diag::new(
                        format!("class `{}` needs at least one type parameter", c.name),
                        c.span,
                    ));
                }
                let id = ClassId(defs.classes.len() as u32);
                defs.classes.push(ClassDef {
                    span: c.span,
                    name: c.name.clone(),
                    params: c.params.iter().map(|p| (p.clone(), Kind::star())).collect(),
                    methods: c
                        .methods
                        .iter()
                        .map(|m| MethodDef {
                            span: m.ty.span,
                            name: m.name.clone(),
                            vars: Vec::new(),
                            ty: Ty::unit_ty(),
                        })
                        .collect(),
                });
                defs.class_ids.insert(c.name.clone(), id);
                let head = fresh_head(&mut kinds, c.params.len());
                kinds_of.insert(c.name.clone(), head);
                arity.insert(c.name.clone(), c.params.len());
            }
            ast::ItemKind::Instance(..) => {}
            ast::ItemKind::Fun(f) => {
                check_reserved(&f.name, f.span)?;
                if defs.fn_ids.contains_key(&f.name) || defs.value_ids.contains_key(&f.name) {
                    return Err(Diag::new(
                        format!("`{}` is already defined", f.name),
                        f.span,
                    ));
                }
                let id = FnId(defs.fns.len() as u32);
                defs.fns.push(FnDef {
                    span: f.span,
                    name: f.name.clone(),
                    params: f.params.clone(),
                    body: f.body.clone(),
                    scheme: None,
                    is_prelude: f.span.lo >= crate::PRELUDE_TAG,
                });
                defs.fn_ids.insert(f.name.clone(), id);
            }
            ast::ItemKind::Let(v) => {
                check_reserved(&v.name, v.span)?;
                if defs.fn_ids.contains_key(&v.name) || defs.value_ids.contains_key(&v.name) {
                    return Err(Diag::new(
                        format!("`{}` is already defined", v.name),
                        v.span,
                    ));
                }
                let id = ValueId(defs.values.len() as u32);
                defs.values.push(ValueDef {
                    span: v.span,
                    name: v.name.clone(),
                    body: v.body.clone(),
                    scheme: None,
                    is_prelude: v.span.lo >= crate::PRELUDE_TAG,
                });
                defs.value_ids.insert(v.name.clone(), id);
            }
        }
    }

    let mut ctx = ResolveCtx {
        kinds,
        kinds_of,
        arity,
        scope: HashMap::new(),
    };

    // Data and class signatures resolve first, so instance heads see
    // class parameter kinds fully constrained by the method signatures.
    let mut method_var_kinds: HashMap<ClassId, Vec<Vec<(String, RawKind)>>> = HashMap::new();
    for item in &program.items {
        match &item.kind {
            ast::ItemKind::Data(d) => {
                let mut scope = HashMap::new();
                for (i, p) in d.params.iter().enumerate() {
                    scope.insert(p.clone(), ctx.peel_param(&d.name, i, d.span)?);
                }
                ctx.scope = scope;
                let data_id = defs.data_ids[&d.name];
                let mut ctors = Vec::with_capacity(d.ctors.len());
                for ctor in &d.ctors {
                    let mut args = Vec::with_capacity(ctor.fields.len());
                    for field in &ctor.fields {
                        let ty = ctx.resolve(field)?;
                        ctx.star_check(&ty, field.span)?;
                        args.push(ty);
                    }
                    ctors.push(CtorDef {
                        span: d.span,
                        name: ctor.name.clone(),
                        args,
                    });
                }
                defs.data[data_id.index()].ctors = ctors;
            }
            ast::ItemKind::Class(c) => {
                let class_id = defs.class_ids[&c.name];
                ctx.scope = HashMap::new();
                for (i, p) in c.params.iter().enumerate() {
                    ctx.scope.insert(p.clone(), ctx.peel_param(&c.name, i, c.span)?);
                }
                let mut vars_per_method = Vec::with_capacity(c.methods.len());
                for (mi, m) in c.methods.iter().enumerate() {
                    let free = free_type_vars(&m.ty, &ctx.kinds_of);
                    let mut vars = Vec::with_capacity(free.len());
                    for name in free {
                        if !ctx.scope.contains_key(&name) {
                            let k = ctx.kinds.var();
                            ctx.scope.insert(name.clone(), k);
                        }
                        vars.push((name.clone(), ctx.scope[&name].clone()));
                    }
                    let ty = ctx.resolve(&m.ty)?;
                    ctx.star_check(&ty, m.ty.span)?;
                    defs.classes[class_id.index()].methods[mi].ty = ty;
                    vars_per_method.push(vars);
                }
                method_var_kinds.insert(class_id, vars_per_method);
            }
            ast::ItemKind::Instance(..) | ast::ItemKind::Fun(..) | ast::ItemKind::Let(..) => {}
        }
    }

    let mut instance_generics_raw: Vec<Vec<(String, RawKind)>> = Vec::new();
    for item in &program.items {
        let ast::ItemKind::Instance(inst) = &item.kind else {
            continue;
        };
        let Some(class_id) = defs.class_by_name(&inst.class) else {
            return Err(Diag::new(
                format!("cannot find class `{}`", inst.class),
                inst.span,
            ));
        };

        let mut free_names: Vec<String> = Vec::new();
        for node in std::iter::once(&inst.self_ty).chain(inst.ctx.iter()) {
            for name in free_type_vars(node, &ctx.kinds_of) {
                if !free_names.contains(&name) {
                    free_names.push(name);
                }
            }
        }
        let mut scope = HashMap::new();
        let mut generics = Vec::with_capacity(free_names.len());
        for name in free_names {
            let k = ctx.kinds.var();
            scope.insert(name.clone(), k.clone());
            generics.push((name, k));
        }
        ctx.scope = scope;

        let parts = instance_args(&mut ctx, &defs, class_id, &inst.self_ty)?;
        for part in &parts {
            if let Ty::Param(name) = part {
                return Err(Diag::new(
                    format!(
                        "the instance head argument `{name}` is a bare type parameter; \
                         instance heads must apply a type constructor"
                    ),
                    inst.span,
                ));
            }
        }
        if defs
            .instances
            .iter()
            .any(|prev| prev.class_id == class_id && prev.args == parts)
        {
            let display = parts
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            return Err(Diag::new(
                format!("duplicate instance for `{} {display}`", inst.class),
                inst.span,
            ));
        }
        let mut cur = ctx.kind_of_name(&inst.class, inst.span)?;
        for part in &parts {
            let kind = ctx.kind_of_ty(part, inst.span)?;
            cur = ctx.kinds.apply(&cur, &kind, inst.span)?;
        }
        ctx.kinds.unify(&cur, &RawKind::Star, inst.span)?;

        let mut bounds = Vec::with_capacity(inst.ctx.len());
        for bound in &inst.ctx {
            let (bound_class, args) = parse_bound(&mut ctx, &defs, bound)?;
            bounds.push(Bound {
                class_id: bound_class,
                args,
                span: bound.span,
            });
        }

        let methods = inst
            .methods
            .iter()
            .map(|m| ImplMethod {
                name: m.name.clone(),
                params: m.params.clone(),
                body: m.body.clone(),
                span: m.span,
            })
            .collect();

        defs.instances.push(InstanceDef {
            span: inst.span,
            class_id,
            generics: Vec::new(),
            args: parts,
            bounds,
            methods,
        });
        instance_generics_raw.push(generics);
    }

    let ResolveCtx {
        kinds,
        kinds_of,
        arity,
        ..
    } = ctx;
    defs.arity = arity;

    defs.kinds_of = kinds_of
        .iter()
        .map(|(name, raw)| (name.clone(), kinds.to_kind(raw)))
        .collect();

    for d in &mut defs.data {
        if let Some(head) = kinds_of.get(&d.name) {
            for (i, (_, kind)) in d.params.iter_mut().enumerate() {
                if let Some(dom) = kinds.nth_domain(head, i) {
                    *kind = kinds.to_kind(&dom);
                }
            }
        }
    }
    for (cidx, c) in defs.classes.iter_mut().enumerate() {
        if let Some(head) = kinds_of.get(&c.name) {
            for (i, (_, kind)) in c.params.iter_mut().enumerate() {
                if let Some(dom) = kinds.nth_domain(head, i) {
                    *kind = kinds.to_kind(&dom);
                }
            }
        }
        if let Some(vars_per_method) = method_var_kinds.get(&ClassId(cidx as u32)) {
            for (m, method) in c.methods.iter_mut().enumerate() {
                if let Some(vars) = vars_per_method.get(m) {
                    method.vars = vars
                        .iter()
                        .map(|(name, raw)| (name.clone(), kinds.to_kind(raw)))
                        .collect();
                }
            }
        }
    }
    for (inst, generics) in defs.instances.iter_mut().zip(&instance_generics_raw) {
        inst.generics = generics
            .iter()
            .map(|(name, raw)| (name.clone(), kinds.to_kind(raw)))
            .collect();
    }

    Ok(defs)
}
