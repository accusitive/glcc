//! Semantic types for fern.
//!
//! `Ty` is the single type representation used through checking, HIR,
//! and lowering. Higher-kinded types are first-class: a type expression
//! `f a` resolves to a [`Ty::App`] when the head is still a bare
//! variable (class variables), and to a saturated `Named { args }` form
//! whenever a named type constructor gets all of its arguments. Kinds
//! are inferred once in [`crate::defs`] and carried by the scopes that
//! introduce type parameters; inference variables snapshot a kind when
//! they are created.
//!
//! fern has no structural sum types: every user datatype is a nominal
//! `Named` type whose constructors are tagged (lowered as tagged
//! enums), plus the anonymous [`Ty::Tuple`] product (`()` being the
//! zero-tuple) and the primitive function type [`Ty::Arrow`].

use std::collections::HashMap;
use std::fmt;

use crate::span::Span;

/// The kind of a type, e.g. `*` or `* -> *`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    Star_,
    /// `k1 -> k2`, so `Star_ -> Star_` is the kind of `List`.
    Arrow(Box<Kind>, Box<Kind>),
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Kind::Star_ => write!(f, "*"),
            Kind::Arrow(a, b) => write!(f, "{a} -> {b}"),
        }
    }
}

impl Kind {
    pub fn star() -> Kind {
        Kind::Star_
    }

    pub fn arrow(src: Kind, dst: Kind) -> Kind {
        Kind::Arrow(Box::new(src), Box::new(dst))
    }

    /// Result kind of applying a `k1 -> k2` head to `arg`; errors when
    /// the head wants something else.
    pub fn apply(&self, arg: &Kind, span: Span) -> Result<Kind, crate::span::Diag> {
        match self {
            Kind::Arrow(a, b) if (&**a) == arg => Ok((**b).clone()),
            other => Err(crate::span::Diag::new(
                format!("cannot apply a type of kind `{other}` to a `{arg}` type"),
                span,
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    /// A nominal type application, saturated or (only when fewer than
    /// the full number of arguments are known) partially applied.
    Named { name: String, args: Vec<Ty> },
    /// A type application whose head is a variable (`f a`).
    App(Box<Ty>, Box<Ty>),
    /// A function type `a -> b`.
    Arrow(Box<Ty>, Box<Ty>),
    /// An anonymous product; `Tuple(vec![])` is the unit type.
    Tuple(Vec<Ty>),
    /// An inference variable owned by [`crate::infer::InferCtx`].
    Var(u32),
    /// The name of a type parameter in scope.
    Param(String),
}

impl Ty {
    pub fn named0(name: &str) -> Ty {
        Ty::Named {
            name: name.to_string(),
            args: Vec::new(),
        }
    }

    pub fn named(name: &str, args: Vec<Ty>) -> Ty {
        Ty::Named {
            name: name.to_string(),
            args,
        }
    }

    pub fn app(head: Ty, arg: Ty) -> Ty {
        Ty::App(Box::new(head), Box::new(arg))
    }

    pub fn arrow(src: Ty, dst: Ty) -> Ty {
        Ty::Arrow(Box::new(src), Box::new(dst))
    }

    pub fn unit_ty() -> Ty {
        Ty::Tuple(Vec::new())
    }

    /// Primitive type names with special lowering behavior.
    pub fn intros() -> [&'static str; 4] {
        ["Int", "Float", "Bool", "Unit"]
    }

    pub fn int_ty() -> Ty {
        Ty::named0("Int")
    }

    pub fn float_ty() -> Ty {
        Ty::named0("Float")
    }

    pub fn bool_ty() -> Ty {
        Ty::named0("Bool")
    }

    pub fn unit_named_ty() -> Ty {
        Ty::named0("Unit")
    }

    pub fn is_unit(&self) -> bool {
        matches!(self, Ty::Tuple(vs) if vs.is_empty())
    }

    /// Substitute type parameters. `Var`s pass through untouched.
    pub fn substitute(&self, map: &HashMap<String, Ty>) -> Ty {
        match self {
            Ty::Named { name, args } => Ty::Named {
                name: name.clone(),
                args: args.iter().map(|a| a.substitute(map)).collect(),
            },
            Ty::App(h, a) => Ty::app(h.substitute(map), a.substitute(map)),
            Ty::Arrow(s, d) => Ty::arrow(s.substitute(map), d.substitute(map)),
            Ty::Tuple(vs) => Ty::Tuple(vs.iter().map(|v| v.substitute(map)).collect()),
            Ty::Param(name) => map.get(name).cloned().unwrap_or_else(|| self.clone()),
            Ty::Var(_) => self.clone(),
        }
    }

    pub fn contains_var(&self) -> bool {
        match self {
            Ty::Var(_) => true,
            Ty::App(h, a) | Ty::Arrow(h, a) => h.contains_var() || a.contains_var(),
            Ty::Named { args, .. } | Ty::Tuple(args) => args.iter().any(|a| a.contains_var()),
            Ty::Param(_) => false,
        }
    }

    pub fn contains_param(&self) -> bool {
        match self {
            Ty::Param(_) => true,
            Ty::App(h, a) | Ty::Arrow(h, a) => h.contains_param() || a.contains_param(),
            Ty::Named { args, .. } | Ty::Tuple(args) => args.iter().any(|a| a.contains_param()),
            Ty::Var(_) => false,
        }
    }

    pub fn is_concrete(&self) -> bool {
        !self.contains_var() && !self.contains_param()
    }

    /// Symbol mangling for MIR instance names; only meaningful on
    /// concrete types. `$` separates components.
    pub fn mangle(&self) -> String {
        match self {
            Ty::Named { name, args } => {
                if args.is_empty() {
                    name.clone()
                } else {
                    let mut out = name.clone();
                    for a in args {
                        out.push('$');
                        out.push_str(&a.mangle());
                    }
                    out
                }
            }
            Ty::App(h, a) => format!("{}${}", h.mangle(), a.mangle()),
            Ty::Arrow(s, d) => format!("fn${}${}", s.mangle(), d.mangle()),
            Ty::Tuple(vs) => {
                let mut out = String::from("tuple");
                for v in vs {
                    out.push('$');
                    out.push_str(&v.mangle());
                }
                out
            }
            Ty::Param(name) => name.clone(),
            Ty::Var(v) => format!("t{v}"),
        }
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ty::Named { name, args } if args.is_empty() => write!(f, "{name}"),
            Ty::Named { name, args } => {
                write!(f, "{name}")?;
                for a in args {
                    write!(f, " {}", operand_of(a))?;
                }
                Ok(())
            }
            Ty::App(h, a) => write!(f, "{} {}", self_operand(h), operand_of(a)),
            Ty::Arrow(s, d) => write!(f, "{} -> {d}", self_operand(s)),
            Ty::Tuple(vs) => {
                write!(f, "(")?;
                for (i, v) in vs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, ")")
            }
            Ty::Param(name) => write!(f, "{name}"),
            Ty::Var(v) => write!(f, "t{v}"),
        }
    }
}

/// Render a type in argument position (parenthesized when its display
/// could run on with a following application).
fn operand_of(ty: &Ty) -> String {
    match ty {
        Ty::Named { args, .. } | Ty::Tuple(args) if !args.is_empty() => format!("({ty})"),
        Ty::Arrow(..) | Ty::App(..) => format!("({ty})"),
        other => format!("{other}"),
    }
}

/// Render a type in head position; only arrows need parens there.
fn self_operand(ty: &Ty) -> String {
    if matches!(ty, Ty::Arrow(..)) {
        format!("({ty})")
    } else {
        format!("{ty}")
    }
}
