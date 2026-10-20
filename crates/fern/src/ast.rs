//! The surface syntax tree.

use crate::span::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct ModuleId(pub u32);

impl ModuleId {
    pub const ROOT: ModuleId = ModuleId(0);

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone)]
pub struct ModuleInfo {
    pub path: Vec<String>,
}

/// A whole fern program: modules, then their items in source order.
#[derive(Debug, Clone, Default)]
pub struct Program {
    pub modules: Vec<ModuleInfo>,
    pub items: Vec<Item>,
}

impl Program {
    pub fn single(items: Vec<Item>) -> Program {
        Program {
            modules: vec![ModuleInfo { path: Vec::new() }],
            items,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Item {
    pub module: ModuleId,
    pub span: Span,
    pub kind: ItemKind,
}

#[derive(Debug, Clone)]
pub enum ItemKind {
    Data(DataDecl),
    Class(ClassDecl),
    Instance(InstanceDecl),
    Extern(ExternDecl),
    Fun(FnDecl),
    Let(ValueDecl),
    Import(ImportDecl),
    Use(UseDecl),
}

/// `import geo::point` or `import geo::point as p`.
#[derive(Debug, Clone)]
pub struct ImportDecl {
    pub span: Span,
    pub path: Vec<String>,
    pub alias: Option<String>,
}

/// `use math::sqrt` or `use math::sqrt as root`.
#[derive(Debug, Clone)]
pub struct UseDecl {
    pub span: Span,
    pub path: Vec<String>,
    pub alias: Option<String>,
}

/// `extern "symbol" name : ty`. The optional string renames the linked
/// symbol; without it the symbol and the fern name are the same.
#[derive(Debug, Clone)]
pub struct ExternDecl {
    pub span: Span,
    pub name: String,
    pub symbol: Option<String>,
    pub ty: TyExpr,
}

#[derive(Debug, Clone)]
pub struct DataDecl {
    pub span: Span,
    pub name: String,
    pub params: Vec<String>,
    pub ctors: Vec<CtorDecl>,
}

#[derive(Debug, Clone)]
pub struct CtorDecl {
    pub name: String,
    pub fields: Vec<TyExpr>,
    /// `Rule { name : T, ... }`; `None` for positional constructors.
    pub field_names: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct ClassDecl {
    pub span: Span,
    pub name: String,
    pub params: Vec<String>,
    pub methods: Vec<MethodSig>,
}

#[derive(Debug, Clone)]
pub struct MethodSig {
    pub name: String,
    pub ty: TyExpr,
}

#[derive(Debug, Clone)]
pub struct InstanceDecl {
    pub span: Span,
    pub class: String,
    /// Haskell-style `context =>`: constraints from the instance head.
    pub ctx: Vec<TyExpr>,
    pub self_ty: TyExpr,
    pub methods: Vec<MethodBody>,
}

#[derive(Debug, Clone)]
pub struct MethodBody {
    pub name: String,
    pub params: Vec<Pat>,
    pub body: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FnParam {
    pub pat: Pat,
    pub ty: TyExpr,
}

#[derive(Debug, Clone)]
pub struct FnDecl {
    pub span: Span,
    pub name: String,
    pub params: Vec<FnParam>,
    pub ret: TyExpr,
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub struct ValueDecl {
    pub span: Span,
    pub name: String,
    pub ty: Option<TyExpr>,
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub struct TyExpr {
    pub kind: TyExprKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum TyExprKind {
    Name(String),
    /// `Pair a b`: juxtaposition, each argument is one atom. Left-assoc.
    App(Box<TyExpr>, Box<TyExpr>),
    /// `a -> b`: right-assoc.
    Arrow(Box<TyExpr>, Box<TyExpr>),
    /// Grouping; transparent for resolution.
    Paren(Box<TyExpr>),
    /// `(a, b)` and `()`: the unit type is the zero-tuple.
    Tuple(Vec<TyExpr>),
}

impl TyExpr {
    pub fn kind(&self) -> &TyExprKind {
        &self.kind
    }

    pub fn display(&self) -> String {
        match &self.kind {
            TyExprKind::Name(n) => n.clone(),
            TyExprKind::Paren(t) => format!("({})", t.display()),
            TyExprKind::Tuple(items) => format!("({})", items.iter().map(|t| t.display()).collect::<Vec<_>>().join(", ")),
            TyExprKind::App(f, a) => format!("{} {}", f.display(), a.display()),
            TyExprKind::Arrow(a, b) => format!("{} -> {}", a.display(), b.display()),
        }
    }

    pub fn free_tyvars(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.collect_free(&mut out);
        out
    }

    fn collect_free(&self, out: &mut Vec<String>) {
        match &self.kind {
            TyExprKind::Name(n) => {
                if !out.contains(n) {
                    out.push(n.clone())
                }
            }
            TyExprKind::Paren(t) => t.collect_free(out),
            TyExprKind::Tuple(items) => {
                for t in items {
                    t.collect_free(out);
                }
            }
            TyExprKind::App(f, a) => {
                f.collect_free(out);
                a.collect_free(out);
            }
            TyExprKind::Arrow(a, b) => {
                a.collect_free(out);
                b.collect_free(out);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Cons,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pat {
    Wild,
    /// Pattern binding a name: lowercase (or lambda parameter).
    Bind(String),
    /// Grouping; transparent.
    Paren(Box<Pat>),
    Ctor {
        name: String,
        fields: Vec<Pat>,
    },
    Tuple(Vec<Pat>),
    /// `[a, b, c]`: sugar, desugared by the checker.
    List(Vec<Pat>),
    /// `a :: b`: sugar over the `Cons` constructor, desugared by the
    /// checker.
    Comp(Box<Pat>, Box<Pat>),
    Int(i64),
    Bool(bool),
}

impl Pat {
    pub fn binds(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.collect_binds(&mut out);
        out
    }

    pub fn is_irrefutable(&self) -> bool {
        match self {
            Pat::Wild | Pat::Bind(_) | Pat::Paren(_) => true,
            Pat::Tuple(fs) => fs.iter().all(|f| f.is_irrefutable()),
            Pat::Ctor { .. } | Pat::List(_) | Pat::Comp(..) | Pat::Int(_) | Pat::Bool(_) => false,
        }
    }

    pub fn collect_binds(&self, out: &mut Vec<String>) {
        match self {
            Pat::Bind(n) => out.push(n.clone()),
            Pat::Wild | Pat::Int(_) | Pat::Bool(_) => {}
            Pat::Paren(inner) => inner.collect_binds(out),
            Pat::Ctor { fields, .. } => {
                for f in fields {
                    f.collect_binds(out);
                }
            }
            Pat::Tuple(fs) => {
                for f in fs {
                    f.collect_binds(out);
                }
            }
            Pat::List(fs) => {
                for f in fs {
                    f.collect_binds(out);
                }
            }
            Pat::Comp(l, r) => {
                l.collect_binds(out);
                r.collect_binds(out);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    Int(i64),
    Float(f64),
    BoolConst(bool),
    Str(String),
    Unit,
    /// `Just(x)`: constructor call.
    CtorCall {
        name: String,
        args: Vec<Expr>,
    },
    /// `Nothing`: bare constructor application, zero args.
    Ctor {
        name: String,
    },
    /// A value reference (function name or local binding)
    Var(String),
    /// `f(x, y)`
    Call(Box<Expr>, Vec<Expr>),
    /// `Rule { targets = ..., recipe = ... }`: a record constructor.
    Record {
        ctor: String,
        fields: Vec<(String, Expr)>,
    },
    /// `state.rules`: a record field, or a module member before resolution.
    Field {
        base: Box<Expr>,
        field: String,
    },
    /// `expr?`: unwrap a `Result`/`Maybe`, early-returning the residual.
    Try(Box<Expr>),
    /// `Eq::eq(x, y)` — a class-method call.
    ClassSel {
        class: String,
        method: String,
        args: Vec<Expr>,
    },
    /// `(x, y) -> body`
    Lambda {
        params: Vec<Pat>,
        body: Box<Expr>,
    },
    /// `let x : T = e in body`
    Let {
        binds: Vec<(Pat, Expr)>,
        body: Box<Expr>,
    },
    /// `match e { case pat => e; ... }`
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },
    Tuple(Vec<Expr>),
    /// `[a, b, c]`: marble-toned sugar; the checker desugars.
    List(Vec<Expr>),
    If {
        cond: Box<Expr>,
        then_body: Box<Expr>,
        else_body: Box<Expr>,
    },
    /// `{ ... }`: a statement block.
    Block(Box<Expr>),
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    UnOp(UnOp, Box<Expr>),
    /// `$ command` (status) or `$(command)` (capture).
    Shell {
        command: Box<ShellCommand>,
        capture: bool,
    },
}

#[derive(Debug, Clone)]
pub struct ShellCommand {
    pub stages: Vec<ShellStage>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ShellStage {
    pub words: Vec<ShellWord>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ShellWord {
    pub parts: Vec<ShellWordPart>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ShellWordPart {
    Lit(String),
    /// `${expr}`: parsed with the surrounding file's spans.
    Interp(Expr),
    /// `$(command)`: the captured stdout.
    Capture(Box<ShellCommand>),
}

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pat: Pat,
    pub guard: Option<Expr>,
    pub body: Expr,
}
