//! Type inference: a trail-based union-find over `Ty::Var`.
//!
//! Inference is kind-agnostic: kinds are checked once in
//! [`crate::defs`], and every type that inference sees is well-kinded by
//! construction. What inference adds over plain HM unification is
//! *higher-kinded applications*: two type applications unify by peeling
//! arguments from the right, so `f a` unifies with `Free g a` by
//! binding `f := Free g`. That is what makes class dispatch over
//! constructors like `Functor f` work.
//!
//! Rigid type parameters ([`Ty::Param`], the variables of a scheme in
//! scope) only unify with themselves; inference variables never bind
//! them.

use std::collections::HashMap;

use crate::defs::Scheme;
use crate::span::{Diag, KResult, Span};
use crate::ty::{Kind, Ty};

pub struct InferCtx {
    vars: Vec<VarState>,
    trail: Vec<(u32, Option<Ty>)>,
    /// Current generalization level; see [`InferCtx::generalize`].
    level: usize,
}

struct VarState {
    link: Option<Ty>,
    level: usize,
}

/// A snapshot of inference state. Rolling back restores bindings and
/// removes variables the probe created.
#[derive(Clone, Copy)]
pub struct Mark {
    vars: usize,
    trail: usize,
}

impl Default for InferCtx {
    fn default() -> Self {
        Self::new()
    }
}

impl InferCtx {
    pub fn new() -> InferCtx {
        InferCtx {
            vars: Vec::new(),
            trail: Vec::new(),
            level: 0,
        }
    }

    pub fn fresh(&mut self) -> Ty {
        self.vars.push(VarState { link: None, level: self.level });
        Ty::Var((self.vars.len() - 1) as u32)
    }

    /// Raises the generalization level while checking a value that may
    /// be generalized on its own (a let-bound lambda).
    pub fn enter_level(&mut self) {
        self.level += 1;
    }

    pub fn leave_level(&mut self) {
        debug_assert!(self.level > 0);
        self.level -= 1;
    }

    pub fn mark(&self) -> Mark {
        Mark {
            vars: self.vars.len(),
            trail: self.trail.len(),
        }
    }

    pub fn rollback(&mut self, mark: Mark) {
        while self.trail.len() > mark.trail {
            let (v, state) = self.trail.pop().unwrap();
            self.vars[v as usize].link = state;
        }
        self.vars.truncate(mark.vars);
    }

    /// Follows variable bindings at the head; inner variables stay.
    pub fn shallow(&self, ty: &Ty) -> Ty {
        let mut cur = ty.clone();
        let mut steps = 0;
        while let Ty::Var(v) = cur {
            match &self.vars[v as usize].link {
                Some(next) => cur = next.clone(),
                None => break,
            }
            steps += 1;
            if steps > 1000 {
                return cur;
            }
        }
        cur
    }

    /// Fully resolves a type, replacing every bound variable.
    pub fn resolve(&self, ty: &Ty) -> Ty {
        match self.shallow(ty) {
            Ty::Named { name, args } => Ty::Named {
                name,
                args: args.iter().map(|a| self.resolve(a)).collect(),
            },
            Ty::App(h, a) => Ty::app(self.resolve(&h), self.resolve(&a)),
            Ty::Arrow(s, d) => Ty::arrow(self.resolve(&s), self.resolve(&d)),
            Ty::Tuple(vs) => Ty::Tuple(vs.iter().map(|v| self.resolve(v)).collect()),
            other => other,
        }
    }

    /// The unbound head variable of `ty`, if `ty` is one.
    pub fn unbound_var(&self, ty: &Ty) -> Option<u32> {
        match self.shallow(ty) {
            Ty::Var(v) if self.vars[v as usize].link.is_none() => Some(v),
            _ => None,
        }
    }

    /// All unbound variables of `ty`, in first-use order.
    pub fn unbound_vars(&self, ty: &Ty) -> Vec<u32> {
        let mut out = Vec::new();
        self.collect_unbound(ty, &mut out);
        out
    }

    fn collect_unbound(&self, ty: &Ty, out: &mut Vec<u32>) {
        match self.shallow(ty) {
            Ty::Var(v) => {
                if !out.contains(&v) {
                    out.push(v);
                }
            }
            Ty::Named { args, .. } | Ty::Tuple(args) => {
                for a in &args {
                    self.collect_unbound(a, out);
                }
            }
            Ty::App(h, a) | Ty::Arrow(h, a) => {
                self.collect_unbound(&h, out);
                self.collect_unbound(&a, out);
            }
            Ty::Param(_) => {}
        }
    }

    fn bind(&mut self, v: u32, ty: &Ty) {
        // Level discipline (Rémy): a variable carries the minimum level
        // of everything it is linked with, so generalization can tell
        // local variables from ones shared with the enclosing scope.
        let m = self.vars[v as usize].level.min(self.min_level(ty));
        self.vars[v as usize].level = m;
        self.lower_level(ty, m);
        self.trail.push((v, self.vars[v as usize].link.clone()));
        self.vars[v as usize].link = Some(ty.clone());
    }

    fn min_level(&self, ty: &Ty) -> usize {
        match self.shallow(ty) {
            Ty::Var(w) => self.vars[w as usize].level,
            Ty::Named { args, .. } | Ty::Tuple(args) => args.iter().map(|a| self.min_level(a)).min().unwrap_or(usize::MAX),
            Ty::App(h, a) | Ty::Arrow(h, a) => self.min_level(&h).min(self.min_level(&a)),
            Ty::Param(_) => usize::MAX,
        }
    }

    fn lower_level(&mut self, ty: &Ty, lvl: usize) {
        match self.shallow(ty) {
            Ty::Var(w) => {
                if self.vars[w as usize].level > lvl {
                    self.vars[w as usize].level = lvl;
                }
            }
            Ty::Named { args, .. } | Ty::Tuple(args) => {
                for a in &args {
                    self.lower_level(a, lvl);
                }
            }
            Ty::App(h, a) | Ty::Arrow(h, a) => {
                self.lower_level(&h, lvl);
                self.lower_level(&a, lvl);
            }
            Ty::Param(_) => {}
        }
    }

    fn occurs(&self, v: u32, ty: &Ty) -> bool {
        match self.shallow(ty) {
            Ty::Var(w) => w == v,
            Ty::Named { args, .. } | Ty::Tuple(args) => args.iter().any(|a| self.occurs(v, a)),
            Ty::App(h, a) | Ty::Arrow(h, a) => self.occurs(v, &h) || self.occurs(v, &a),
            Ty::Param(_) => false,
        }
    }

    /// Splits a type into its application head and arguments: `Free g`
    /// becomes (`Free`, `[g])`, a bare name becomes (itself, `[]`).
    fn spine(&self, ty: &Ty) -> (Ty, Vec<Ty>) {
        match self.shallow(ty) {
            Ty::App(h, a) => {
                let (head, mut args) = self.spine(&h);
                args.push(self.resolve(&a));
                (head, args)
            }
            Ty::Named { name, args } if !args.is_empty() => (Ty::named0(&name), args.iter().map(|a| self.resolve(a)).collect()),
            other => (other, Vec::new()),
        }
    }

    pub fn unify(&mut self, a: &Ty, b: &Ty, span: Span) -> KResult<()> {
        let a = self.shallow(a);
        let b = self.shallow(b);
        match (&a, &b) {
            (Ty::Var(x), Ty::Var(y)) if x == y => Ok(()),
            (Ty::Var(x), other) => {
                if self.occurs(*x, other) {
                    return Err(Diag::new("recursive type", span));
                }
                self.bind(*x, other);
                Ok(())
            }
            (other, Ty::Var(y)) => {
                if self.occurs(*y, other) {
                    return Err(Diag::new("recursive type", span));
                }
                self.bind(*y, other);
                Ok(())
            }
            _ => {
                let (h1, args1) = self.spine(&a);
                let (h2, args2) = self.spine(&b);
                self.unify_spines(h1, args1, h2, args2, span)
            }
        }
    }

    /// Unifies two application spines from the right, so a shorter
    /// application can match a longer one through its (variable) head:
    /// `f a` against `Free g a` peels `a` and binds `f := Free g`.
    fn unify_spines(&mut self, h1: Ty, mut args1: Vec<Ty>, h2: Ty, mut args2: Vec<Ty>, span: Span) -> KResult<()> {
        while let (Some(x), Some(y)) = (args1.last(), args2.last()) {
            let (x, y) = (x.clone(), y.clone());
            self.unify(&x, &y, span)?;
            args1.pop();
            args2.pop();
        }
        match (args1.is_empty(), args2.is_empty()) {
            (true, true) => self.unify_heads(&h1, &h2, span),
            (true, false) if matches!(h1, Ty::Var(_)) => {
                let rhs = rebuild(h2, args2);
                self.unify(&h1, &rhs, span)
            }
            (false, true) if matches!(h2, Ty::Var(_)) => {
                let lhs = rebuild(h1, args1);
                self.unify(&lhs, &h2, span)
            }
            _ => Err(mismatch(&h1, &h2, span)),
        }
    }

    fn unify_heads(&mut self, h1: &Ty, h2: &Ty, span: Span) -> KResult<()> {
        match (h1, h2) {
            (Ty::Var(_), _) | (_, Ty::Var(_)) => self.unify(h1, h2, span),
            (Ty::Named { name: n1, .. }, Ty::Named { name: n2, .. }) if n1 == n2 => Ok(()),
            (Ty::Param(p1), Ty::Param(p2)) if p1 == p2 => Ok(()),
            (Ty::Arrow(s1, d1), Ty::Arrow(s2, d2)) => {
                self.unify(s1, s2, span)?;
                self.unify(d1, d2, span)
            }
            (Ty::Tuple(v1), Ty::Tuple(v2)) if v1.len() == v2.len() => {
                for (x, y) in v1.iter().zip(v2.iter()) {
                    self.unify(x, y, span)?;
                }
                Ok(())
            }
            _ => Err(mismatch(h1, h2, span)),
        }
    }

    /// Instantiates every listed scheme parameter with a fresh variable,
    /// also returning those fresh arguments in parameter order.
    pub fn instantiate_args(&mut self, vars: &[(String, Kind)], ty: &Ty) -> (Ty, Vec<Ty>) {
        let mut map = HashMap::with_capacity(vars.len());
        let mut args = Vec::with_capacity(vars.len());
        for (name, _) in vars {
            let v = self.fresh();
            map.insert(name.clone(), v.clone());
            args.push(v);
        }
        (ty.substitute(&map), args)
    }

    /// Instantiates every listed scheme parameter with a fresh variable.
    pub fn instantiate(&mut self, vars: &[(String, Kind)], ty: &Ty) -> Ty {
        self.instantiate_args(vars, ty).0
    }

    pub fn instantiate_scheme(&mut self, scheme: &Scheme) -> Ty {
        self.instantiate(&scheme.vars, &scheme.ty)
    }

    /// Quantifies the unbound variables of `ty` that were created at
    /// the current level or deeper; variables shared with the enclosing
    /// scope (lower level) stay free. Top-level bodies call this at
    /// level zero, quantifying everything. The variables become rigid
    /// parameters named `t$<varid>`, which no source identifier can
    /// collide with.
    pub fn generalize(&self, ty: &Ty) -> Scheme {
        let vars: Vec<u32> = self.unbound_vars(ty).into_iter().filter(|v| self.vars[*v as usize].level >= self.level).collect();
        Scheme {
            ty: self.rigidify(ty, &vars),
            vars: vars.iter().map(|v| (format!("t${v}"), Kind::star())).collect(),
        }
    }

    /// Every currently unbound variable, in creation order.
    pub fn unbound_ids(&self) -> Vec<u32> {
        self.vars.iter().enumerate().filter(|(_, state)| state.link.is_none()).map(|(i, _)| i as u32).collect()
    }

    /// Substitutes the unbound variables `ids` by the rigid parameters
    /// that `generalize` names them with, across a whole type.
    pub fn rigidify(&self, ty: &Ty, ids: &[u32]) -> Ty {
        // Resolve first: a type can mention several ids of one
        // equivalence class, and only the root is in `ids`.
        let resolved = self.resolve(ty);
        let mut map = HashMap::with_capacity(ids.len());
        for v in ids {
            map.insert(*v, Ty::Param(format!("t${v}")));
        }
        subst_vars(&resolved, &map)
    }
}

/// Rebuilds a type from a head and arguments, folding back into the
/// saturated `Named` form when the head is a type constructor.
fn rebuild(head: Ty, args: Vec<Ty>) -> Ty {
    if let Ty::Named { name, args: none } = &head {
        if none.is_empty() {
            return Ty::named(name, args);
        }
    }
    let mut out = head;
    for arg in args {
        out = Ty::app(out, arg);
    }
    out
}

fn subst_vars(ty: &Ty, map: &HashMap<u32, Ty>) -> Ty {
    match ty {
        Ty::Var(v) => map.get(v).cloned().unwrap_or_else(|| ty.clone()),
        Ty::Named { name, args } => Ty::Named {
            name: name.clone(),
            args: args.iter().map(|a| subst_vars(a, map)).collect(),
        },
        Ty::App(h, a) => Ty::app(subst_vars(h, map), subst_vars(a, map)),
        Ty::Arrow(s, d) => Ty::arrow(subst_vars(s, map), subst_vars(d, map)),
        Ty::Tuple(vs) => Ty::Tuple(vs.iter().map(|v| subst_vars(v, map)).collect()),
        Ty::Param(_) => ty.clone(),
    }
}

fn mismatch(a: &Ty, b: &Ty, span: Span) -> Diag {
    Diag::new(format!("mismatched types: expected `{a}`, found `{b}`"), span)
}
