//! Checking: Hindley-Milner inference with class obligations, producing
//! typed HIR.
//!
//! Top-level items are checked in declaration order. A function or value
//! may call itself (its in-progress type is bound while its body is
//! checked) but not items declared later; instance method bodies are
//! checked against the class signature with the class parameters
//! substituted for the instance head, and the instance's bounds become
//! the environment for the class solver.
//!
//! Class-method calls (`Eq::eq(x, y)`) instantiate the method signature,
//! record a `C a1 ... an` obligation, and ask the solver how to dispatch:
//! a concrete instance turns into a direct call, while a goal that still
//! mentions a body type parameter stays a [`FnRef::Class`] that lowering
//! resolves after monomorphization.

use std::collections::{HashMap, HashSet};

use crate::ast::{self, BinOp, UnOp};
use crate::defs::{Bound, Defs, FnId, Scheme, ValueId};
use crate::hir::{BodyKey, ConstValue, FnRef, HirArm, HirCapture, HirClosure, HirExpr, HirExprKind, HirFn, HirLocal, HirPat, HirPatKind, HirProgram, TryResidual};
use crate::infer::InferCtx;
use crate::solve::{Obligation, Solution, Solver};
use crate::span::{Diag, KResult, Span};
use crate::ty::{Kind, Ty};

pub fn check(defs: &mut Defs, program: &ast::Program) -> KResult<HirProgram> {
    let (hir, fn_schemes, value_schemes) = {
        let mut state = Check::new(defs);
        let hir = state.run(program)?;
        (hir, state.fn_schemes, state.value_schemes)
    };
    for (def, scheme) in defs.fns.iter_mut().zip(fn_schemes) {
        def.scheme = scheme;
    }
    for (def, scheme) in defs.values.iter_mut().zip(value_schemes) {
        def.scheme = scheme;
    }
    Ok(hir)
}

/// Check only the items from `first_item` on, reusing the schemes and
/// HIR of everything before it.
///
/// `start_hir` must hold the bodies of the prefix items (and may hold
/// unreachable bodies from earlier analyses; they are kept as-is).
/// `defs.fns`/`defs.values` must already carry the schemes of the
/// prefix, which [`Defs`] collection leaves as `None`.
/// Recheck only `selected` items, keeping the bodies of everything else
/// from `start_hir`. The item list and its IDs must have the same shape as
/// the one `start_hir` was checked from.
pub fn check_selected(defs: &mut Defs, program: &ast::Program, selected: &HashSet<usize>, mut start_hir: HirProgram) -> KResult<HirProgram> {
    let mut fn_ids = HashSet::new();
    let mut value_ids = HashSet::new();
    let mut instances = HashSet::new();
    let (mut fn_index, mut value_index, mut instance_index) = (0usize, 0usize, 0usize);
    for (index, item) in program.items.iter().enumerate() {
        match &item.kind {
            ast::ItemKind::Fun(_) => {
                if selected.contains(&index) {
                    fn_ids.insert(FnId(fn_index as u32));
                }
                fn_index += 1;
            }
            ast::ItemKind::Let(_) => {
                if selected.contains(&index) {
                    value_ids.insert(ValueId(value_index as u32));
                }
                value_index += 1;
            }
            ast::ItemKind::Instance(_) => {
                if selected.contains(&index) {
                    instances.insert(instance_index);
                }
                instance_index += 1;
            }
            ast::ItemKind::Data(_) | ast::ItemKind::Class(_) | ast::ItemKind::Extern(_) | ast::ItemKind::Import(_) | ast::ItemKind::Use(_) => {}
        }
    }
    start_hir.bodies.retain(|body| match body.key {
        BodyKey::Fn(id) => !fn_ids.contains(&id),
        BodyKey::Value(id) => !value_ids.contains(&id),
        BodyKey::Method { instance, .. } => !instances.contains(&instance),
    });

    let (hir, fn_schemes, value_schemes) = {
        let mut state = Check::new(defs);
        state.fn_schemes = defs.fns.iter().map(|f| f.scheme.clone()).collect();
        state.value_schemes = defs.values.iter().map(|v| v.scheme.clone()).collect();
        state.hir = start_hir;
        let hir = state.run_selected(program, selected)?;
        (hir, state.fn_schemes, state.value_schemes)
    };
    for (def, scheme) in defs.fns.iter_mut().zip(fn_schemes) {
        def.scheme = scheme;
    }
    for (def, scheme) in defs.values.iter_mut().zip(value_schemes) {
        def.scheme = scheme;
    }
    Ok(hir)
}

pub fn check_from(defs: &mut Defs, program: &ast::Program, first_item: usize, start_hir: HirProgram) -> KResult<HirProgram> {
    let (hir, fn_schemes, value_schemes) = {
        let mut state = Check::new(defs);
        state.fn_schemes = defs.fns.iter().map(|f| f.scheme.clone()).collect();
        state.value_schemes = defs.values.iter().map(|v| v.scheme.clone()).collect();
        state.hir = start_hir;
        let hir = state.run_from(program, first_item)?;
        (hir, state.fn_schemes, state.value_schemes)
    };
    for (def, scheme) in defs.fns.iter_mut().zip(fn_schemes) {
        def.scheme = scheme;
    }
    for (def, scheme) in defs.values.iter_mut().zip(value_schemes) {
        def.scheme = scheme;
    }
    Ok(hir)
}

struct Check<'d> {
    defs: &'d Defs,
    solver: Solver<'d>,
    infer: InferCtx,
    hir: HirProgram,
    fn_schemes: Vec<Option<Scheme>>,
    value_schemes: Vec<Option<Scheme>>,
    /// Bodies being checked: index 0 is the current top-level body,
    /// later entries are enclosing closures.
    bodies: Vec<BodyState>,
    /// The bounds in scope for the current body.
    env: Vec<Bound>,
    /// Return types of the enclosing bodies, innermost last.
    returns: Vec<Ty>,
    /// The name and in-progress type of the item being checked, when it
    /// may refer to itself.
    self_ref: Option<SelfRef>,
    /// Whether the body being checked is an async function body.
    in_async: bool,
}

enum SelfRef {
    Fn(FnId, String, Ty),
    Value(ValueId, String, Ty),
}

fn ctor_short(name: &str) -> &str {
    name.rsplit("::").next().unwrap_or(name)
}

struct TryInfo {
    name: String,
    success: Ty,
    error: Option<Ty>,
    ok_ctor: usize,
    err_ctor: usize,
    is_maybe: bool,
}

struct BodyState {
    locals: Vec<HirLocal>,
    /// Lexical scopes, innermost last: name to local id.
    scopes: Vec<HashMap<String, usize>>,
    /// For closure bodies: captures discovered so far.
    captures: Vec<HirCapture>,
}

impl BodyState {
    fn new() -> BodyState {
        BodyState {
            locals: Vec::new(),
            scopes: vec![HashMap::new()],
            captures: Vec::new(),
        }
    }

    fn lookup(&self, name: &str) -> Option<usize> {
        self.scopes.iter().rev().find_map(|scope| scope.get(name).copied())
    }

    /// The local of this body that holds the copy of `outer`, adding
    /// the capture when it is new.
    fn capture(&mut self, name: &str, outer: usize, ty: Ty, span: Span) -> usize {
        if let Some(capture) = self.captures.iter().find(|c| c.outer == outer) {
            return capture.inner;
        }
        let inner = self.locals.len();
        self.locals.push(HirLocal {
            name: name.to_string(),
            ty: ty.clone(),
            span,
        });
        self.captures.push(HirCapture { outer, inner, ty });
        inner
    }
}

impl<'d> Check<'d> {
    fn new(defs: &'d Defs) -> Check<'d> {
        Check {
            defs,
            solver: Solver::new(defs),
            infer: InferCtx::new(),
            hir: HirProgram::default(),
            fn_schemes: Vec::new(),
            value_schemes: Vec::new(),
            bodies: Vec::new(),
            env: Vec::new(),
            returns: Vec::new(),
            self_ref: None,
            in_async: false,
        }
    }

    fn run(&mut self, program: &ast::Program) -> KResult<HirProgram> {
        self.fn_schemes = self.defs.fns.iter().map(|f| f.scheme.clone()).collect();
        self.value_schemes = self.defs.values.iter().map(|v| v.scheme.clone()).collect();
        self.run_from(program, 0)
    }

    fn run_selected(&mut self, program: &ast::Program, selected: &HashSet<usize>) -> KResult<HirProgram> {
        let (mut fn_index, mut value_index, mut instance_index) = (0usize, 0usize, 0usize);
        for (index, item) in program.items.iter().enumerate() {
            match &item.kind {
                ast::ItemKind::Data(_) | ast::ItemKind::Class(_) | ast::ItemKind::Extern(_) | ast::ItemKind::Import(_) | ast::ItemKind::Use(_) => {}
                ast::ItemKind::Instance(_) => {
                    if selected.contains(&index) {
                        self.check_instance(instance_index)?;
                    }
                    instance_index += 1;
                }
                ast::ItemKind::Fun(f) => {
                    if selected.contains(&index) {
                        self.check_fn(FnId(fn_index as u32), f)?;
                    }
                    fn_index += 1;
                }
                ast::ItemKind::Let(v) => {
                    if selected.contains(&index) {
                        self.check_value(ValueId(value_index as u32), v)?;
                    }
                    value_index += 1;
                }
            }
        }
        Ok(std::mem::take(&mut self.hir))
    }

    fn run_from(&mut self, program: &ast::Program, first_item: usize) -> KResult<HirProgram> {
        let (mut fn_index, mut value_index, mut instance_index) = (0usize, 0usize, 0usize);
        for item in &program.items[..first_item] {
            match &item.kind {
                ast::ItemKind::Instance(_) => instance_index += 1,
                ast::ItemKind::Fun(_) => fn_index += 1,
                ast::ItemKind::Let(_) => value_index += 1,
                ast::ItemKind::Data(_) | ast::ItemKind::Class(_) | ast::ItemKind::Extern(_) | ast::ItemKind::Import(_) | ast::ItemKind::Use(_) => {}
            }
        }
        for item in &program.items[first_item..] {
            match &item.kind {
                ast::ItemKind::Data(_) | ast::ItemKind::Class(_) | ast::ItemKind::Extern(_) | ast::ItemKind::Import(_) | ast::ItemKind::Use(_) => {}
                ast::ItemKind::Instance(_) => {
                    self.check_instance(instance_index)?;
                    instance_index += 1;
                }
                ast::ItemKind::Fun(f) => {
                    self.check_fn(FnId(fn_index as u32), f)?;
                    fn_index += 1;
                }
                ast::ItemKind::Let(v) => {
                    self.check_value(ValueId(value_index as u32), v)?;
                    value_index += 1;
                }
            }
        }
        Ok(std::mem::take(&mut self.hir))
    }

    // ----------------------------------------------------------- items --

    fn check_fn(&mut self, id: FnId, f: &ast::FnDecl) -> KResult<()> {
        let span = f.span;
        let name = f.name.clone();
        let def = &self.defs.fns[id.index()];
        let declared = def.scheme.clone().expect("function signatures are resolved during collection");
        let param_tys: Vec<Ty> = def.params.iter().map(|p| p.ty.clone()).collect();
        let ret = def.ret.clone();
        let fn_ty = declared.ty.clone();
        let outer_async = self.in_async;
        self.in_async = f.is_async;
        let checked = self.check_root(BodyKey::Fn(id), name, span, declared.vars.clone(), Vec::new(), false, Some(declared.clone()), &mut |this| {
            check_param_names(f.params.iter().map(|p| &p.pat), span)?;
            let mut params = Vec::new();
            for (param, ty) in f.params.iter().zip(&param_tys) {
                params.push(this.check_pat(&param.pat, ty, span)?);
            }
            this.self_ref = Some(SelfRef::Fn(id, f.name.clone(), fn_ty.clone()));
            this.returns.push(ret.clone());
            let body = this.check_expr(&f.body, &ret);
            this.returns.pop();
            Ok((params, ret.clone(), body?, fn_ty.clone()))
        });
        self.in_async = outer_async;
        let (_, mut hir_fn) = checked?;
        hir_fn.is_async = f.is_async;
        self.fn_schemes[id.index()] = Some(declared);
        self.hir.bodies.push(hir_fn);
        Ok(())
    }

    fn check_value(&mut self, id: ValueId, v: &ast::ValueDecl) -> KResult<()> {
        let span = v.span;
        let name = v.name.clone();
        if let Some(declared) = self.defs.values[id.index()].scheme.clone() {
            let ty = declared.ty.clone();
            let (_, hir_fn) = self.check_root(
                BodyKey::Value(id),
                name,
                span,
                declared.vars.clone(),
                Vec::new(),
                false,
                Some(declared.clone()),
                &mut |this| {
                    this.self_ref = Some(SelfRef::Value(id, v.name.clone(), ty.clone()));
                    this.returns.push(ty.clone());
                    let body = this.check_expr(&v.body, &ty);
                    this.returns.pop();
                    Ok((Vec::new(), ty.clone(), body?, ty.clone()))
                },
            )?;
            self.value_schemes[id.index()] = Some(declared);
            self.hir.bodies.push(hir_fn);
            return Ok(());
        }

        let (scheme, hir_fn) = self.check_root(BodyKey::Value(id), name, span, Vec::new(), Vec::new(), false, None, &mut |this| {
            let ty = this.infer.fresh();
            this.self_ref = Some(SelfRef::Value(id, v.name.clone(), ty.clone()));
            this.returns.push(ty.clone());
            let body = this.check_expr(&v.body, &ty);
            this.returns.pop();
            Ok((Vec::new(), ty.clone(), body?, ty))
        })?;
        self.value_schemes[id.index()] = Some(scheme);
        self.hir.bodies.push(hir_fn);
        Ok(())
    }

    fn check_instance(&mut self, index: usize) -> KResult<()> {
        let inst = self.defs.instances[index].clone();
        let class = self.defs.classes[inst.class_id.index()].clone();

        for m in &inst.methods {
            if class.method(&m.name).is_none() {
                return Err(Diag::new(format!("class `{}` has no method `{}`", class.name, m.name), m.span));
            }
        }

        self.env = inst.bounds.clone();
        let result = (|| {
            for (mi, method) in class.methods.iter().enumerate() {
                let Some(impl_m) = inst.methods.iter().find(|m| m.name == method.name) else {
                    continue;
                };
                self.check_instance_method(index, &inst, &class, mi, method, impl_m)?;
            }
            Ok(())
        })();
        self.env = Vec::new();
        result
    }

    fn check_instance_method(
        &mut self,
        index: usize,
        inst: &crate::defs::InstanceDef,
        class: &crate::defs::ClassDef,
        method_index: usize,
        method: &crate::defs::MethodDef,
        impl_m: &crate::defs::ImplMethod,
    ) -> KResult<()> {
        let mut map = HashMap::new();
        for (i, (pname, _)) in class.params.iter().enumerate() {
            map.insert(pname.clone(), inst.args[i].clone());
        }
        let sig = method.ty.substitute(&map);

        let class_params: HashSet<&str> = class.params.iter().map(|(n, _)| n.as_str()).collect();
        let mut generics = inst.generics.clone();
        for (name, kind) in &method.vars {
            if !class_params.contains(name.as_str()) {
                generics.push((name.clone(), kind.clone()));
            }
        }

        let span = impl_m.span;
        let name = format!("{}::{}", class.name, method.name);
        let bounds = inst.bounds.clone();
        let (_, hir_fn) = self.check_root(
            BodyKey::Method {
                instance: index,
                method: method_index,
            },
            name,
            span,
            generics,
            bounds,
            true,
            None,
            &mut |this| {
                check_param_names(impl_m.params.iter(), span)?;
                let (param_tys, ret) = split_arrows(&sig);
                if param_tys.len() != impl_m.params.len() {
                    return Err(Diag::new(
                        format!(
                            "method `{}` takes {} parameter(s), but the instance gives {}",
                            method.name,
                            param_tys.len(),
                            impl_m.params.len()
                        ),
                        span,
                    ));
                }
                let mut params = Vec::new();
                for (p, want) in impl_m.params.iter().zip(param_tys.iter()) {
                    params.push(this.check_pat(p, want, span)?);
                }
                this.returns.push(ret.clone());
                let body = this.check_expr(&impl_m.body, &ret);
                this.returns.pop();
                Ok((params, ret, body?, sig.clone()))
            },
        )?;
        self.hir.bodies.push(hir_fn);
        Ok(())
    }

    /// The shared frame around a top-level body: pushes the body state,
    /// runs `build`, then generalizes, finalizes types, and assembles
    /// the [`HirFn`] plus the body's scheme.
    #[allow(clippy::too_many_arguments)]
    fn check_root(
        &mut self,
        key: BodyKey,
        name: String,
        span: Span,
        generics: Vec<(String, Kind)>,
        bounds: Vec<Bound>,
        is_method: bool,
        declared: Option<Scheme>,
        build: &mut dyn FnMut(&mut Check<'d>) -> KResult<(Vec<HirPat>, Ty, HirExpr, Ty)>,
    ) -> KResult<(Scheme, HirFn)> {
        let closure_start = self.hir.closures.len();
        self.bodies.push(BodyState::new());
        self.self_ref = None;
        let built = build(self);
        let (params, ret, body, fn_ty) = match built {
            Ok(built) => built,
            Err(diag) => {
                self.bodies.pop();
                return Err(diag);
            }
        };
        let state = self.bodies.pop().unwrap();

        let scheme = declared.unwrap_or_else(|| self.infer.generalize(&fn_ty));
        let item_generics = if is_method { generics } else { scheme.vars.clone() };
        let allowed: HashSet<String> = item_generics.iter().map(|(n, _)| n.clone()).collect();
        let ids = self.infer.unbound_ids();
        let self_ref = self.self_ref.take();

        let mut params = params;
        for p in &mut params {
            finalize_pat(&self.infer, &ids, &allowed, p)?;
        }
        let mut ret = ret;
        finalize_ty(&self.infer, &ids, &allowed, &mut ret, span)?;
        let mut body = body;
        finalize_expr(&self.infer, &ids, &allowed, &self_ref, &item_generics, &mut body)?;
        let mut locals = state.locals;
        for l in &mut locals {
            finalize_ty(&self.infer, &ids, &allowed, &mut l.ty, l.span)?;
        }
        for c in &mut self.hir.closures[closure_start..] {
            c.generics = item_generics.clone();
            for p in &mut c.params {
                finalize_pat(&self.infer, &ids, &allowed, p)?;
            }
            finalize_ty(&self.infer, &ids, &allowed, &mut c.ret, c.span)?;
            for l in &mut c.locals {
                finalize_ty(&self.infer, &ids, &allowed, &mut l.ty, l.span)?;
            }
            for capture in &mut c.captures {
                finalize_ty(&self.infer, &ids, &allowed, &mut capture.ty, c.span)?;
            }
            finalize_expr(&self.infer, &ids, &allowed, &self_ref, &item_generics, &mut c.body)?;
        }

        let hir_fn = HirFn {
            key,
            name,
            generics: item_generics,
            bounds,
            params,
            ret,
            locals,
            body,
            span,
            is_async: false,
        };
        Ok((scheme, hir_fn))
    }

    // ---------------------------------------------------- expressions --

    fn synth(&mut self, e: &ast::Expr) -> KResult<HirExpr> {
        let span = e.span;
        match &e.kind {
            ast::ExprKind::Int(v) => Ok(HirExpr {
                kind: HirExprKind::Const(ConstValue::Int(*v)),
                ty: Ty::int_ty(),
                span,
            }),
            ast::ExprKind::Float(v) => Ok(HirExpr {
                kind: HirExprKind::Const(ConstValue::Float(*v)),
                ty: Ty::float_ty(),
                span,
            }),
            ast::ExprKind::BoolConst(v) => Ok(HirExpr {
                kind: HirExprKind::Const(ConstValue::Bool(*v)),
                ty: Ty::bool_ty(),
                span,
            }),
            ast::ExprKind::Str(v) => Ok(HirExpr {
                kind: HirExprKind::Const(ConstValue::Str(v.clone())),
                ty: Ty::named0("String"),
                span,
            }),
            ast::ExprKind::Unit => Ok(HirExpr {
                kind: HirExprKind::Tuple(Vec::new()),
                ty: Ty::unit_ty(),
                span,
            }),
            ast::ExprKind::CtorCall { name, args } => self.synthesize_ctor(name, args, span),
            ast::ExprKind::Ctor { name } => self.synthesize_ctor(name, &[], span),
            ast::ExprKind::Var(name) => self.synthesize_var(name, span),
            ast::ExprKind::Call(callee, args) => self.synthesize_call(callee, args, span),
            ast::ExprKind::Record { ctor, fields } => self.synthesize_record(ctor, fields, span),
            ast::ExprKind::Field { base, field } => self.synthesize_field(base, field, span),
            ast::ExprKind::Try(inner) => self.synthesize_try(inner, span),
            ast::ExprKind::Await(inner) => self.synthesize_await(inner, span),
            ast::ExprKind::ClassSel { class, method, args } => self.synthesize_class_sel(class, method, args, span),
            ast::ExprKind::Lambda { params, body } => self.synthesize_lambda(params, body, span),
            ast::ExprKind::Let { binds, body } => self.synthesize_let(binds, body, span),
            ast::ExprKind::Match { scrutinee, arms } => self.synthesize_match(scrutinee, arms, span),
            ast::ExprKind::Tuple(items) => {
                let mut hir_items = Vec::with_capacity(items.len());
                let mut tys = Vec::with_capacity(items.len());
                for item in items {
                    let h = self.synth(item)?;
                    tys.push(h.ty.clone());
                    hir_items.push(h);
                }
                Ok(HirExpr {
                    kind: HirExprKind::Tuple(hir_items),
                    ty: Ty::Tuple(tys),
                    span,
                })
            }
            ast::ExprKind::List(items) => self.synthesize_list(items, span),
            ast::ExprKind::If { cond, then_body, else_body } => {
                let cond_hir = self.check_expr(cond, &Ty::bool_ty())?;
                let then_hir = self.synth(then_body)?;
                let ty = then_hir.ty.clone();
                let else_hir = self.check_expr(else_body, &ty)?;
                Ok(HirExpr {
                    kind: HirExprKind::If {
                        cond: Box::new(cond_hir),
                        then_body: Box::new(then_hir),
                        else_body: Box::new(else_hir),
                    },
                    ty,
                    span,
                })
            }
            ast::ExprKind::Block(inner) => self.synth(inner),
            ast::ExprKind::BinOp(op, lhs, rhs) => self.synthesize_binop(*op, lhs, rhs, span),
            ast::ExprKind::UnOp(op, operand) => self.synthesize_unop(*op, operand, span),
            ast::ExprKind::Shell { command, capture } => {
                let desugared = self.desugar_shell(command, *capture)?;
                self.synth(&desugared)
            }
        }
    }

    /// Shell commands are ordinary expressions once their words become
    /// `String` values: a command is `spawn`, a capture is
    /// `spawn_capture`, and a pipeline folds its stages into a `Pipe`.
    fn desugar_shell(&self, command: &ast::ShellCommand, capture: bool) -> KResult<ast::Expr> {
        if command.stages.len() == 1 {
            let stage = &command.stages[0];
            if let Some((name, args)) = self.shell_builtin(stage)? {
                return Ok(shell_call(name, args, command.span));
            }
            let program = self.shell_word(&stage.words[0])?;
            let args = stage.words[1..].iter().map(|word| self.shell_word(word)).collect::<KResult<Vec<_>>>()?;
            let list = ast::Expr {
                kind: ast::ExprKind::List(args),
                span: stage.span,
            };
            let function = if capture { "spawn_capture" } else { "spawn" };
            return Ok(shell_call(function, vec![program, list], command.span));
        }

        let stages = command
            .stages
            .iter()
            .map(|stage| {
                if self.shell_builtin(stage)?.is_some() {
                    return Err(Diag::new("a `#` builtin cannot be a pipeline stage", stage.span));
                }
                let words = stage.words.iter().map(|word| self.shell_word(word)).collect::<KResult<Vec<_>>>()?;
                let list = ast::Expr {
                    kind: ast::ExprKind::List(words),
                    span: stage.span,
                };
                Ok(shell_call("argv", vec![list], stage.span))
            })
            .collect::<KResult<Vec<_>>>()?;
        let list = ast::Expr {
            kind: ast::ExprKind::List(stages),
            span: command.span,
        };
        let pipe = shell_call("shell_pipe_of", vec![list], command.span);
        let function = if capture { "shell_pipeline_capture" } else { "shell_pipeline" };
        Ok(shell_call(function, vec![pipe], command.span))
    }

    /// A `#name` word at the start of a stage is a builtin call rather
    /// than an executable. `#env` and `#set` mirror the parser's names.
    fn shell_builtin(&self, stage: &ast::ShellStage) -> KResult<Option<(String, Vec<ast::Expr>)>> {
        let Some(first) = stage.words.first() else {
            return Ok(None);
        };
        let [ast::ShellWordPart::Lit(word)] = first.parts.as_slice() else {
            return Ok(None);
        };
        let Some(name) = word.strip_prefix('#') else {
            return Ok(None);
        };
        let mut chars = name.chars();
        let Some(head) = chars.next() else {
            return Ok(None);
        };
        if !(head.is_ascii_lowercase() || head == '_') || !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Ok(None);
        }
        let name = crate::prelude::builtin_name(name);
        let args = stage.words[1..].iter().map(|word| self.shell_word(word)).collect::<KResult<Vec<_>>>()?;
        Ok(Some((name, args)))
    }

    fn shell_word(&self, word: &ast::ShellWord) -> KResult<ast::Expr> {
        let mut joined: Option<ast::Expr> = None;
        for part in &word.parts {
            let piece = match part {
                ast::ShellWordPart::Lit(text) => ast::Expr {
                    kind: ast::ExprKind::Str(text.clone()),
                    span: word.span,
                },
                ast::ShellWordPart::Interp(expr) => shell_call("shell_arg", vec![expr.clone()], word.span),
                ast::ShellWordPart::Capture(command) => shell_call("shell_arg", vec![self.desugar_shell(command, true)?], word.span),
            };
            joined = Some(match joined {
                None => piece,
                Some(left) => shell_call("concat", vec![left, piece], word.span),
            });
        }
        Ok(joined.unwrap_or_else(|| ast::Expr {
            kind: ast::ExprKind::Str(String::new()),
            span: word.span,
        }))
    }

    fn check_expr(&mut self, e: &ast::Expr, expected: &Ty) -> KResult<HirExpr> {
        let hir = self.synth(e)?;
        self.infer.unify(expected, &hir.ty, e.span)?;
        Ok(hir)
    }

    fn synthesize_var(&mut self, name: &str, span: Span) -> KResult<HirExpr> {
        if let Some((id, ty)) = self.lookup_local(name, span) {
            return Ok(HirExpr {
                kind: HirExprKind::Local(id),
                ty,
                span,
            });
        }
        if let Some(self_ref) = &self.self_ref {
            match self_ref {
                SelfRef::Fn(id, sname, ty) if sname == name => {
                    return Ok(HirExpr {
                        kind: HirExprKind::Fn(FnRef::Fn { id: *id, args: Vec::new() }),
                        ty: ty.clone(),
                        span,
                    });
                }
                SelfRef::Value(id, sname, ty) if sname == name => {
                    return Ok(HirExpr {
                        kind: HirExprKind::Fn(FnRef::Value { id: *id, args: Vec::new() }),
                        ty: ty.clone(),
                        span,
                    });
                }
                _ => {}
            }
        }
        if let Some(id) = self.defs.fn_by_name(name) {
            let Some(scheme) = self.fn_schemes[id.index()].clone() else {
                return Err(Diag::new(format!("`{name}` is used before it is defined"), span));
            };
            let (ty, args) = self.infer.instantiate_args(&scheme.vars, &scheme.ty);
            return Ok(HirExpr {
                kind: HirExprKind::Fn(FnRef::Fn { id, args }),
                ty,
                span,
            });
        }
        if let Some(id) = self.defs.value_by_name(name) {
            let Some(scheme) = self.value_schemes[id.index()].clone() else {
                return Err(Diag::new(format!("`{name}` is used before it is defined"), span));
            };
            let (ty, args) = self.infer.instantiate_args(&scheme.vars, &scheme.ty);
            return Ok(HirExpr {
                kind: HirExprKind::Fn(FnRef::Value { id, args }),
                ty,
                span,
            });
        }
        if let Some(id) = self.defs.extern_by_name(name) {
            return Ok(HirExpr {
                kind: HirExprKind::Fn(FnRef::Extern { id }),
                ty: self.defs.externs[id.index()].ty.clone(),
                span,
            });
        }
        // A class method used without qualification.
        let mut found: Option<(crate::defs::ClassId, usize)> = None;
        for (ci, class) in self.defs.classes.iter().enumerate() {
            if let Some(mi) = class.method(name) {
                if found.is_some() {
                    return Err(Diag::new(
                        format!(
                            "`{name}` is ambiguous: several classes declare a method with \
                             this name; write `Class::method`"
                        ),
                        span,
                    ));
                }
                found = Some((crate::defs::ClassId(ci as u32), mi));
            }
        }
        if let Some((class_id, mi)) = found {
            return self.class_method_value(class_id, mi, span);
        }
        Err(Diag::new(format!("cannot find `{name}`"), span))
    }

    /// A reference to a class method as a value: the class parameters
    /// and the method's own variables are instantiated fresh, and
    /// dispatch happens once a call site has constrained them.
    fn class_method_value(&mut self, class_id: crate::defs::ClassId, method: usize, span: Span) -> KResult<HirExpr> {
        let class = self.defs.classes[class_id.index()].clone();
        let mdef = class.methods[method].clone();
        let class_names: HashSet<&str> = class.params.iter().map(|(n, _)| n.as_str()).collect();
        let mut vars = class.params.clone();
        for (n, k) in &mdef.vars {
            if !class_names.contains(n.as_str()) {
                vars.push((n.clone(), k.clone()));
            }
        }
        let (ty, args) = self.infer.instantiate_args(&vars, &mdef.ty);
        Ok(HirExpr {
            kind: HirExprKind::Fn(FnRef::Class { class_id, method, args }),
            ty,
            span,
        })
    }

    fn synthesize_ctor(&mut self, name: &str, args: &[ast::Expr], span: Span) -> KResult<HirExpr> {
        let (data_id, tag) = self.defs.ctor_by_name(name).ok_or_else(|| Diag::new(format!("cannot find constructor `{name}`"), span))?;
        let data = self.defs.data[data_id.index()].clone();
        let ctor = data.ctors[tag as usize].clone();
        if ctor.args.len() != args.len() {
            return Err(Diag::new(
                format!("constructor `{name}` takes {} field(s), but {} argument(s) were given", ctor.args.len(), args.len()),
                span,
            ));
        }
        let mut map = HashMap::new();
        let mut fresh = Vec::new();
        for (pname, _) in &data.params {
            let v = self.infer.fresh();
            fresh.push(v.clone());
            map.insert(pname.clone(), v);
        }
        let mut hir_args = Vec::new();
        for (a, want) in args.iter().zip(ctor.args.iter()) {
            let want = want.substitute(&map);
            hir_args.push(self.check_expr(a, &want)?);
        }
        Ok(HirExpr {
            kind: HirExprKind::Ctor {
                data_id,
                ctor: tag as usize,
                args: hir_args,
            },
            ty: Ty::Named {
                name: data.name.clone(),
                args: fresh,
            },
            span,
        })
    }

    fn synthesize_record(&mut self, name: &str, fields: &[(String, ast::Expr)], span: Span) -> KResult<HirExpr> {
        let (data_id, tag) = self.defs.ctor_by_name(name).ok_or_else(|| Diag::new(format!("cannot find constructor `{name}`"), span))?;
        let data = self.defs.data[data_id.index()].clone();
        let ctor = data.ctors[tag as usize].clone();
        let Some(field_names) = &ctor.field_names else {
            return Err(Diag::new(format!("constructor `{name}` does not have named fields"), span));
        };
        let mut values: Vec<Option<&ast::Expr>> = vec![None; field_names.len()];
        for (field, value) in fields {
            let Some(index) = field_names.iter().position(|name| name == field) else {
                return Err(Diag::new(format!("`{name}` has no field `{field}`"), value.span));
            };
            if values[index].replace(value).is_some() {
                return Err(Diag::new(format!("field `{field}` is given twice"), value.span));
            }
        }
        let mut map = HashMap::new();
        let mut fresh = Vec::new();
        for (pname, _) in &data.params {
            let v = self.infer.fresh();
            fresh.push(v.clone());
            map.insert(pname.clone(), v);
        }
        let mut hir_args = Vec::with_capacity(values.len());
        for (index, value) in values.into_iter().enumerate() {
            let Some(value) = value else {
                return Err(Diag::new(format!("record constructor `{name}` is missing field `{}`", field_names[index]), span));
            };
            let want = ctor.args[index].substitute(&map);
            hir_args.push(self.check_expr(value, &want)?);
        }
        Ok(HirExpr {
            kind: HirExprKind::Ctor {
                data_id,
                ctor: tag as usize,
                args: hir_args,
            },
            ty: Ty::Named {
                name: data.name.clone(),
                args: fresh,
            },
            span,
        })
    }

    fn synthesize_field(&mut self, base: &ast::Expr, field: &str, span: Span) -> KResult<HirExpr> {
        let base = self.synth(base)?;
        let ty = self.infer.resolve(&base.ty);
        let Ty::Named { name, args } = &ty else {
            return Err(Diag::new(format!("field access requires a record, found `{ty}`"), span));
        };
        let data_id = self.defs.data_by_name(name).ok_or_else(|| Diag::new(format!("cannot find type `{name}`"), span))?;
        let data = self.defs.data[data_id.index()].clone();
        let ctor = data
            .ctors
            .first()
            .filter(|_| data.ctors.len() == 1)
            .ok_or_else(|| Diag::new(format!("`{name}` is not a record type"), span))?;
        let field_names = ctor.field_names.as_ref().ok_or_else(|| Diag::new(format!("`{name}` is not a record type"), span))?;
        let Some(index) = field_names.iter().position(|name| name == field) else {
            return Err(Diag::new(format!("`{name}` has no field `{field}`"), span));
        };
        if data.params.len() != args.len() {
            return Err(Diag::new(format!("`{name}` is not fully applied"), span));
        }
        let subst: HashMap<String, Ty> = data.params.iter().map(|(n, _)| n.clone()).zip(args.iter().cloned()).collect();
        let ty = ctor.args[index].substitute(&subst);
        Ok(HirExpr {
            kind: HirExprKind::Field { base: Box::new(base), index },
            ty,
            span,
        })
    }

    fn synthesize_call(&mut self, callee: &ast::Expr, args: &[ast::Expr], span: Span) -> KResult<HirExpr> {
        let callee_hir = self.synth(callee)?;
        let callee_ty = callee_hir.ty.clone();
        // A named callee is fully applied to its declared parameters, so
        // a declared one-parameter function may return a function. Values
        // have no declared arity; they must fill their whole arrow spine.
        let declared = match &callee_hir.kind {
            HirExprKind::Fn(target) => self.named_arity(target),
            _ => None,
        };
        // The number of arrows is definite up to a trailing unbound
        // variable, which may still grow into more arrows.
        let (known, open) = match declared {
            Some(n) => (n, false),
            None => self.arity_scan(&callee_ty),
        };
        if args.len() < known || (!open && args.len() != known) {
            let count = if open { format!("at least {known}") } else { format!("{known}") };
            return Err(Diag::new(
                format!("this function takes {count} parameter(s), but {} argument(s) were given", args.len()),
                span,
            ));
        }
        let mut cur = callee_ty;
        let mut params = Vec::new();
        for _ in args {
            let a = self.infer.fresh();
            let r = self.infer.fresh();
            self.infer.unify(&cur, &Ty::arrow(a.clone(), r.clone()), callee.span)?;
            params.push(a);
            cur = r;
        }
        let mut hir_args = Vec::new();
        for (a, want) in args.iter().zip(params.iter()) {
            hir_args.push(self.check_expr(a, want)?);
        }
        let ty = cur;
        let kind = match callee_hir.kind {
            HirExprKind::Fn(FnRef::Class { class_id, method, args }) => self.dispatch_class(class_id, method, args, hir_args, span)?,
            // A top-level `let` is evaluated to a closure value, so calls
            // go through the value rather than to a function body.
            HirExprKind::Fn(FnRef::Value { .. }) => HirExprKind::Call {
                callee: Box::new(callee_hir),
                args: hir_args,
            },
            HirExprKind::Fn(target) => HirExprKind::CallDirect { target, args: hir_args },
            kind => HirExprKind::Call {
                callee: Box::new(HirExpr {
                    kind,
                    ty: callee_hir.ty,
                    span: callee_hir.span,
                }),
                args: hir_args,
            },
        };
        Ok(HirExpr { kind, ty, span })
    }

    /// Resolves a class-method call once its arguments have constrained
    /// the class parameters. A concrete instance becomes a direct call;
    /// a goal discharged by the environment or still open stays a
    /// [`FnRef::Class`] that lowering re-resolves.
    ///
    /// `args` carries the class arguments followed by the method's own
    /// instantiation, matching the layout [`FnRef::Class`] documents.
    fn dispatch_class(&mut self, class_id: crate::defs::ClassId, method: usize, args: Vec<Ty>, hir_args: Vec<HirExpr>, span: Span) -> KResult<HirExprKind> {
        let n_class = self.defs.classes[class_id.index()].params.len();
        let class_args = args[..n_class].to_vec();
        let method_args = args[n_class..].to_vec();
        let goal = Obligation {
            class_id,
            args: class_args.clone(),
        };
        let env = self.env.clone();
        let solution = self.solver.prove(&mut self.infer, &env, &goal, span)?;
        let target = match &solution {
            Solution::Instance { index, args } => {
                let mut all = args.clone();
                all.extend(method_args);
                FnRef::Instance {
                    instance: *index,
                    method,
                    args: all,
                }
            }
            Solution::Bound | Solution::Deferred => FnRef::Class { class_id, method, args },
            _ => return Err(self.solver.describe_failure(&goal, &solution, span)),
        };
        Ok(HirExprKind::CallDirect { target, args: hir_args })
    }

    fn synthesize_class_sel(&mut self, class: &str, method: &str, args: &[ast::Expr], span: Span) -> KResult<HirExpr> {
        let class_id = self.defs.class_by_name(class).ok_or_else(|| Diag::new(format!("cannot find class `{class}`"), span))?;
        let class_def = self.defs.classes[class_id.index()].clone();
        let Some(mi) = class_def.method(method) else {
            return Err(Diag::new(format!("class `{class}` has no method `{method}`"), span));
        };
        let mdef = class_def.methods[mi].clone();

        let class_params: HashSet<&str> = class_def.params.iter().map(|(n, _)| n.as_str()).collect();
        let mut vars = class_def.params.clone();
        for (n, k) in &mdef.vars {
            if !class_params.contains(n.as_str()) {
                vars.push((n.clone(), k.clone()));
            }
        }
        let (sig, all_args) = self.infer.instantiate_args(&vars, &mdef.ty);

        let (params, ret) = split_arrows(&sig);
        if params.len() != args.len() {
            return Err(Diag::new(
                format!("`{class}::{method}` takes {} parameter(s), but {} argument(s) were given", params.len(), args.len()),
                span,
            ));
        }
        let mut hir_args = Vec::new();
        for (a, want) in args.iter().zip(params.iter()) {
            hir_args.push(self.check_expr(a, want)?);
        }

        let kind = self.dispatch_class(class_id, mi, all_args, hir_args, span)?;
        Ok(HirExpr { kind, ty: ret, span })
    }

    fn try_info(&self, ty: &Ty, span: Span) -> KResult<TryInfo> {
        let ty = self.infer.resolve(ty);
        let Ty::Named { name, args } = ty else {
            return Err(Diag::new(format!("`?` requires a `Maybe` or `Result`, found `{ty}`"), span));
        };
        let is_maybe = match name.as_str() {
            "Maybe" => true,
            "std::result::Result" => false,
            _ => return Err(Diag::new(format!("`?` requires a `Maybe` or `Result`, found `{name}`"), span)),
        };
        let arity = if is_maybe { 1 } else { 2 };
        if args.len() != arity {
            return Err(Diag::new(format!("`{name}` is not fully applied in `?`"), span));
        }
        let data_id = self.defs.data_by_name(&name).ok_or_else(|| Diag::new(format!("cannot find type `{name}`"), span))?;
        let data = &self.defs.data[data_id.index()];
        let (ok_name, err_name) = if is_maybe { ("Just", "Nothing") } else { ("Ok", "Err") };
        let ok_ctor = data.ctors.iter().position(|c| ctor_short(&c.name) == ok_name);
        let err_ctor = data.ctors.iter().position(|c| ctor_short(&c.name) == err_name);
        let (Some(ok_ctor), Some(err_ctor)) = (ok_ctor, err_ctor) else {
            return Err(Diag::new(format!("`{name}` does not have the expected constructors"), span));
        };
        Ok(TryInfo {
            name,
            success: args[0].clone(),
            error: if is_maybe { None } else { Some(args[1].clone()) },
            ok_ctor,
            err_ctor,
            is_maybe,
        })
    }

    fn synthesize_try(&mut self, inner: &ast::Expr, span: Span) -> KResult<HirExpr> {
        let value = self.synth(inner)?;
        let input = self.try_info(&value.ty, inner.span)?;
        let ret = self.returns.last().cloned().ok_or_else(|| Diag::new("`?` can only be used inside a function body", span))?;
        let mut ret = self.infer.shallow(&ret);
        if matches!(ret, Ty::Var(_)) {
            let fresh = self.infer.fresh();
            let wrapper = if input.is_maybe {
                Ty::named(&input.name, vec![fresh])
            } else {
                Ty::named(&input.name, vec![fresh, input.error.clone().unwrap()])
            };
            self.infer.unify(&ret, &wrapper, span)?;
            ret = wrapper;
        }
        let want = if input.is_maybe { "Maybe" } else { "Result" };
        let out = self
            .try_info(&ret, span)
            .map_err(|_| Diag::new(format!("`?` on a `{want}` requires the enclosing function to return `{want}`"), span))?;
        if input.is_maybe != out.is_maybe {
            return Err(Diag::new(format!("`?` on a `{want}` requires the enclosing function to return `{want}`"), span));
        }
        let residual = if input.is_maybe {
            TryResidual::Reuse
        } else {
            self.infer.unify(input.error.as_ref().unwrap(), out.error.as_ref().unwrap(), span)?;
            TryResidual::Rebuild {
                err_ctor: input.err_ctor,
                ret_err_ctor: out.err_ctor,
            }
        };
        Ok(HirExpr {
            kind: HirExprKind::Try {
                value: Box::new(value),
                ok_ctor: input.ok_ctor,
                residual,
            },
            ty: input.success,
            span,
        })
    }

    fn synthesize_await(&mut self, inner: &ast::Expr, span: Span) -> KResult<HirExpr> {
        if !self.in_async {
            return Err(Diag::new("`await` is only valid inside an async function", span));
        }
        let value = self.synth(inner)?;
        let value_ty = self.infer.shallow(&value.ty);
        let Some(output) = value_ty.future_inner().cloned() else {
            return Err(Diag::new(format!("`await` requires a `Future`, found `{value_ty}`"), span));
        };
        Ok(HirExpr {
            kind: HirExprKind::Await(Box::new(value)),
            ty: output,
            span,
        })
    }

    fn synthesize_lambda(&mut self, params: &[ast::Pat], body: &ast::Expr, span: Span) -> KResult<HirExpr> {
        check_param_names(params, span)?;
        self.bodies.push(BodyState::new());
        let outer_async = self.in_async;
        self.in_async = false;
        let mut param_pats = Vec::new();
        for p in params {
            let t = self.infer.fresh();
            param_pats.push(self.check_pat(p, &t, span)?);
        }
        let ret = self.infer.fresh();
        self.returns.push(ret.clone());
        let built = self.check_expr(body, &ret);
        self.returns.pop();
        self.in_async = outer_async;
        let body_hir = match built {
            Ok(body_hir) => body_hir,
            Err(diag) => {
                self.bodies.pop();
                return Err(diag);
            }
        };
        let state = self.bodies.pop().unwrap();
        let captures: Vec<usize> = state.captures.iter().map(|c| c.outer).collect();
        let id = self.hir.closures.len() as u32;
        self.hir.closures.push(HirClosure {
            id,
            name: format!("lambda${id}"),
            generics: Vec::new(),
            params: param_pats,
            ret: ret.clone(),
            locals: state.locals,
            captures: state.captures,
            body: body_hir,
            span,
        });
        let ty = self.hir.closures[id as usize].params.iter().rev().fold(ret, |acc, p| Ty::arrow(p.ty.clone(), acc));
        Ok(HirExpr {
            kind: HirExprKind::Closure { id, captures },
            ty,
            span,
        })
    }

    fn synthesize_let(&mut self, binds: &[(ast::Pat, ast::Expr)], body: &ast::Expr, span: Span) -> KResult<HirExpr> {
        self.push_scope();
        let built = (|| {
            let mut hir_binds = Vec::new();
            for (pat, value) in binds {
                let value_hir = self.synth(value)?;
                let ty = value_hir.ty.clone();
                let pat_hir = self.check_pat(pat, &ty, value.span)?;
                hir_binds.push((pat_hir, value_hir));
            }
            let body_hir = self.synth(body)?;
            let ty = body_hir.ty.clone();
            Ok(HirExpr {
                kind: HirExprKind::Let {
                    binds: hir_binds,
                    body: Box::new(body_hir),
                },
                ty,
                span,
            })
        })();
        self.pop_scope();
        built
    }

    fn synthesize_match(&mut self, scrutinee: &ast::Expr, arms: &[ast::MatchArm], span: Span) -> KResult<HirExpr> {
        let scrut = self.synth(scrutinee)?;
        let scrut_ty = scrut.ty.clone();
        let result = self.infer.fresh();
        let mut hir_arms = Vec::new();
        for arm in arms {
            self.push_scope();
            let built = (|| {
                let pat = self.check_pat(&arm.pat, &scrut_ty, arm.body.span)?;
                let guard = match &arm.guard {
                    Some(g) => Some(self.check_expr(g, &Ty::bool_ty())?),
                    None => None,
                };
                let body = self.check_expr(&arm.body, &result)?;
                Ok(HirArm { pattern: pat, guard, body })
            })();
            self.pop_scope();
            hir_arms.push(built?);
        }
        let ty = self.infer.shallow(&result);
        Ok(HirExpr {
            kind: HirExprKind::Match {
                scrutinee: Box::new(scrut),
                arms: hir_arms,
            },
            ty,
            span,
        })
    }

    fn synthesize_list(&mut self, items: &[ast::Expr], span: Span) -> KResult<HirExpr> {
        let (cons_id, cons_tag, nil_tag) = self.list_ctors(span)?;
        let el = self.infer.fresh();
        let mut hir_items = Vec::new();
        for item in items {
            hir_items.push(self.check_expr(item, &el)?);
        }
        let list_ty = Ty::named("List", vec![el.clone()]);
        let mut result = HirExpr {
            kind: HirExprKind::Ctor {
                data_id: cons_id,
                ctor: nil_tag,
                args: Vec::new(),
            },
            ty: list_ty.clone(),
            span,
        };
        for item in hir_items.into_iter().rev() {
            result = HirExpr {
                kind: HirExprKind::Ctor {
                    data_id: cons_id,
                    ctor: cons_tag,
                    args: vec![item, result],
                },
                ty: list_ty.clone(),
                span,
            };
        }
        Ok(result)
    }

    fn synthesize_binop(&mut self, op: BinOp, lhs: &ast::Expr, rhs: &ast::Expr, span: Span) -> KResult<HirExpr> {
        let l = self.synth(lhs)?;
        match op {
            BinOp::Cons => {
                let (cons_id, cons_tag, _) = self.list_ctors(span)?;
                let el = self.infer.fresh();
                self.infer.unify(&l.ty, &el, lhs.span)?;
                let list_ty = Ty::named("List", vec![el]);
                let r = self.check_expr(rhs, &list_ty)?;
                Ok(HirExpr {
                    kind: HirExprKind::Ctor {
                        data_id: cons_id,
                        ctor: cons_tag,
                        args: vec![l, r],
                    },
                    ty: list_ty,
                    span,
                })
            }
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                let r = self.synth(rhs)?;
                self.infer.unify(&l.ty, &r.ty, rhs.span)?;
                self.numeric(&l.ty, lhs.span)?;
                let ty = l.ty.clone();
                Ok(HirExpr {
                    kind: HirExprKind::Bin {
                        op,
                        lhs: Box::new(l),
                        rhs: Box::new(r),
                    },
                    ty,
                    span,
                })
            }
            BinOp::Eq | BinOp::Ne => {
                let r = self.synth(rhs)?;
                self.infer.unify(&l.ty, &r.ty, rhs.span)?;
                self.comparable(&l.ty, lhs.span)?;
                Ok(HirExpr {
                    kind: HirExprKind::Bin {
                        op,
                        lhs: Box::new(l),
                        rhs: Box::new(r),
                    },
                    ty: Ty::bool_ty(),
                    span,
                })
            }
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                let r = self.synth(rhs)?;
                self.infer.unify(&l.ty, &r.ty, rhs.span)?;
                self.numeric(&l.ty, lhs.span)?;
                Ok(HirExpr {
                    kind: HirExprKind::Bin {
                        op,
                        lhs: Box::new(l),
                        rhs: Box::new(r),
                    },
                    ty: Ty::bool_ty(),
                    span,
                })
            }
            BinOp::And | BinOp::Or => {
                self.infer.unify(&l.ty, &Ty::bool_ty(), lhs.span)?;
                let r = self.check_expr(rhs, &Ty::bool_ty())?;
                Ok(HirExpr {
                    kind: HirExprKind::Bin {
                        op,
                        lhs: Box::new(l),
                        rhs: Box::new(r),
                    },
                    ty: Ty::bool_ty(),
                    span,
                })
            }
        }
    }

    fn synthesize_unop(&mut self, op: UnOp, operand: &ast::Expr, span: Span) -> KResult<HirExpr> {
        let o = self.synth(operand)?;
        let ty = match op {
            UnOp::Neg => {
                self.numeric(&o.ty, operand.span)?;
                o.ty.clone()
            }
            UnOp::Not => {
                self.infer.unify(&o.ty, &Ty::bool_ty(), operand.span)?;
                Ty::bool_ty()
            }
        };
        Ok(HirExpr {
            kind: HirExprKind::Un { op, operand: Box::new(o) },
            ty,
            span,
        })
    }

    // ------------------------------------------------------ patterns --

    fn check_pat(&mut self, p: &ast::Pat, expected: &Ty, span: Span) -> KResult<HirPat> {
        self.check_dup_binds(p, span)?;
        self.pat_inner(p, expected, span)
    }

    fn check_dup_binds(&self, p: &ast::Pat, span: Span) -> KResult<()> {
        let mut seen = HashSet::new();
        for name in p.binds() {
            if !seen.insert(name.clone()) {
                return Err(Diag::new(format!("the name `{name}` is bound more than once in this pattern"), span));
            }
        }
        Ok(())
    }

    fn pat_inner(&mut self, p: &ast::Pat, expected: &Ty, span: Span) -> KResult<HirPat> {
        match p {
            ast::Pat::Wild => Ok(HirPat {
                kind: HirPatKind::Wild,
                ty: expected.clone(),
                span,
            }),
            ast::Pat::Bind(name) => {
                let id = self.alloc_local(name, expected.clone(), span);
                Ok(HirPat {
                    kind: HirPatKind::Bind(id),
                    ty: expected.clone(),
                    span,
                })
            }
            ast::Pat::Paren(inner) => self.pat_inner(inner, expected, span),
            ast::Pat::Ctor { name, fields } => {
                let (data_id, tag) = self.defs.ctor_by_name(name).ok_or_else(|| Diag::new(format!("cannot find constructor `{name}`"), span))?;
                let data = self.defs.data[data_id.index()].clone();
                let ctor = data.ctors[tag as usize].clone();
                if ctor.args.len() != fields.len() {
                    return Err(Diag::new(
                        format!("constructor `{name}` takes {} field(s), but the pattern has {}", ctor.args.len(), fields.len()),
                        span,
                    ));
                }
                let mut map = HashMap::new();
                let mut fresh = Vec::new();
                for (pname, _) in &data.params {
                    let v = self.infer.fresh();
                    fresh.push(v.clone());
                    map.insert(pname.clone(), v);
                }
                let want_ty = Ty::Named {
                    name: data.name.clone(),
                    args: fresh,
                };
                self.infer.unify(expected, &want_ty, span)?;
                let mut hir_fields = Vec::new();
                for (f, want) in fields.iter().zip(ctor.args.iter()) {
                    let want = want.substitute(&map);
                    hir_fields.push(self.pat_inner(f, &want, span)?);
                }
                Ok(HirPat {
                    kind: HirPatKind::Ctor {
                        data_id,
                        ctor: tag as usize,
                        fields: hir_fields,
                    },
                    ty: want_ty,
                    span,
                })
            }
            ast::Pat::Tuple(items) => {
                let mut fresh = Vec::new();
                for _ in items {
                    fresh.push(self.infer.fresh());
                }
                self.infer.unify(expected, &Ty::Tuple(fresh.clone()), span)?;
                let mut hir_items = Vec::new();
                for (f, want) in items.iter().zip(fresh) {
                    hir_items.push(self.pat_inner(f, &want, span)?);
                }
                Ok(HirPat {
                    kind: HirPatKind::Tuple(hir_items),
                    ty: expected.clone(),
                    span,
                })
            }
            ast::Pat::List(items) => self.list_pattern(items, expected, span),
            ast::Pat::Comp(head, tail) => self.cons_pattern(head, tail, expected, span),
            ast::Pat::Int(v) => {
                self.infer.unify(expected, &Ty::int_ty(), span)?;
                Ok(HirPat {
                    kind: HirPatKind::Int(*v),
                    ty: Ty::int_ty(),
                    span,
                })
            }
            ast::Pat::Bool(v) => {
                self.infer.unify(expected, &Ty::bool_ty(), span)?;
                Ok(HirPat {
                    kind: HirPatKind::Bool(*v),
                    ty: Ty::bool_ty(),
                    span,
                })
            }
        }
    }

    fn list_ctors(&self, span: Span) -> KResult<(crate::defs::DataId, usize, usize)> {
        let Some((data_id, cons)) = self.defs.ctor_by_name("Cons") else {
            return Err(Diag::new("the list syntax needs the prelude constructor `Cons`", span));
        };
        let Some((_, nil)) = self.defs.ctor_by_name("Nil") else {
            return Err(Diag::new("the list syntax needs the prelude constructor `Nil`", span));
        };
        Ok((data_id, cons as usize, nil as usize))
    }

    fn list_pattern(&mut self, items: &[ast::Pat], expected: &Ty, span: Span) -> KResult<HirPat> {
        let (data_id, cons_tag, nil_tag) = self.list_ctors(span)?;
        let el = self.infer.fresh();
        let list_ty = Ty::named("List", vec![el.clone()]);
        self.infer.unify(expected, &list_ty, span)?;
        self.build_list_pattern(items, 0, &el, &list_ty, data_id, cons_tag, nil_tag, span)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_list_pattern(
        &mut self,
        items: &[ast::Pat],
        index: usize,
        el: &Ty,
        list_ty: &Ty,
        data_id: crate::defs::DataId,
        cons_tag: usize,
        nil_tag: usize,
        span: Span,
    ) -> KResult<HirPat> {
        if index >= items.len() {
            return Ok(HirPat {
                kind: HirPatKind::Ctor {
                    data_id,
                    ctor: nil_tag,
                    fields: Vec::new(),
                },
                ty: list_ty.clone(),
                span,
            });
        }
        let head = self.pat_inner(&items[index], el, span)?;
        let tail = self.build_list_pattern(items, index + 1, el, list_ty, data_id, cons_tag, nil_tag, span)?;
        Ok(HirPat {
            kind: HirPatKind::Ctor {
                data_id,
                ctor: cons_tag,
                fields: vec![head, tail],
            },
            ty: list_ty.clone(),
            span,
        })
    }

    fn cons_pattern(&mut self, head: &ast::Pat, tail: &ast::Pat, expected: &Ty, span: Span) -> KResult<HirPat> {
        let (data_id, cons_tag, _) = self.list_ctors(span)?;
        let el = self.infer.fresh();
        let list_ty = Ty::named("List", vec![el.clone()]);
        self.infer.unify(expected, &list_ty, span)?;
        let head_pat = self.pat_inner(head, &el, span)?;
        let tail_pat = self.pat_inner(tail, &list_ty, span)?;
        Ok(HirPat {
            kind: HirPatKind::Ctor {
                data_id,
                ctor: cons_tag,
                fields: vec![head_pat, tail_pat],
            },
            ty: list_ty,
            span,
        })
    }

    // ------------------------------------------------------- helpers --

    fn lookup_local(&mut self, name: &str, span: Span) -> Option<(usize, Ty)> {
        let mut found: Option<(usize, usize)> = None;
        for (i, body) in self.bodies.iter().enumerate().rev() {
            if let Some(id) = body.lookup(name) {
                found = Some((i, id));
                break;
            }
        }
        let (body_index, mut id) = found?;
        let mut ty = self.bodies[body_index].locals[id].ty.clone();
        for k in body_index + 1..self.bodies.len() {
            let outer = id;
            let body = &mut self.bodies[k];
            id = body.capture(name, outer, ty.clone(), span);
            ty = body.locals[id].ty.clone();
        }
        Some((id, ty))
    }

    fn alloc_local(&mut self, name: &str, ty: Ty, span: Span) -> usize {
        let body = self.bodies.last_mut().expect("a body is being checked");
        let id = body.locals.len();
        body.locals.push(HirLocal { name: name.to_string(), ty, span });
        body.scopes.last_mut().expect("a scope is open").insert(name.to_string(), id);
        id
    }

    fn push_scope(&mut self) {
        if let Some(body) = self.bodies.last_mut() {
            body.scopes.push(HashMap::new());
        }
    }

    fn pop_scope(&mut self) {
        if let Some(body) = self.bodies.last_mut() {
            body.scopes.pop();
        }
    }

    fn numeric(&mut self, ty: &Ty, span: Span) -> KResult<()> {
        match self.infer.shallow(ty) {
            Ty::Var(_) => self.infer.unify(ty, &Ty::int_ty(), span),
            Ty::Named { name, args } if args.is_empty() && (name == "Int" || name == "Float") => Ok(()),
            other => Err(Diag::new(format!("expected a numeric type, found `{other}`"), span)),
        }
    }

    fn comparable(&mut self, ty: &Ty, span: Span) -> KResult<()> {
        match self.infer.shallow(ty) {
            Ty::Var(_) => self.infer.unify(ty, &Ty::int_ty(), span),
            Ty::Named { name, args } if args.is_empty() && (name == "Int" || name == "Float" || name == "Bool" || name == "String") => Ok(()),
            other => Err(Diag::new(format!("cannot compare values of type `{other}`"), span)),
        }
    }

    fn arity_scan(&self, ty: &Ty) -> (usize, bool) {
        let mut cur = self.infer.shallow(ty);
        let mut n = 0;
        loop {
            match cur {
                Ty::Arrow(_, d) => {
                    n += 1;
                    cur = self.infer.shallow(&d);
                }
                Ty::Var(_) => return (n, true),
                _ => return (n, false),
            }
        }
    }

    /// The declared parameter count of a named callee: top-level
    /// functions and instance methods by their patterns, class methods
    /// by their signature. `None` when the callee is a value, which has
    /// no declared parameters.
    fn named_arity(&self, target: &FnRef) -> Option<usize> {
        match target {
            FnRef::Fn { id, .. } => Some(self.defs.fns[id.index()].params.len()),
            FnRef::Value { .. } => None,
            FnRef::Instance { instance, method, .. } => Some(self.defs.instances[*instance].methods[*method].params.len()),
            FnRef::Extern { id } => Some(self.defs.externs[id.index()].params.len()),
            FnRef::Class { class_id, method, .. } => {
                let sig = &self.defs.classes[class_id.index()].methods[*method].ty;
                Some(split_arrows(sig).0.len())
            }
        }
    }
}

fn split_arrows(ty: &Ty) -> (Vec<Ty>, Ty) {
    let mut params = Vec::new();
    let mut cur = ty.clone();
    while let Ty::Arrow(s, d) = cur {
        params.push(*s);
        cur = *d;
    }
    (params, cur)
}

/// Parameter lists bind distinct names: `fun f(x, x)` is an error even
/// though each pattern alone is fine.
fn check_param_names<'a>(params: impl IntoIterator<Item = &'a ast::Pat>, span: Span) -> KResult<()> {
    let mut seen = HashSet::new();
    for p in params {
        for name in p.binds() {
            if !seen.insert(name.clone()) {
                return Err(Diag::new(format!("the parameter `{name}` is bound more than once"), span));
            }
        }
    }
    Ok(())
}

// ------------------------------------------------------------ finalize --

fn finalize_ty(infer: &InferCtx, ids: &[u32], allowed: &HashSet<String>, ty: &mut Ty, span: Span) -> KResult<()> {
    let resolved = infer.resolve(ty);
    *ty = infer.rigidify(&resolved, ids);
    check_unknown_params(ty, allowed, span)
}

fn shell_call(name: impl Into<String>, args: Vec<ast::Expr>, span: Span) -> ast::Expr {
    ast::Expr {
        kind: ast::ExprKind::Call(
            Box::new(ast::Expr {
                kind: ast::ExprKind::Var(name.into()),
                span,
            }),
            args,
        ),
        span,
    }
}

fn check_unknown_params(ty: &Ty, allowed: &HashSet<String>, span: Span) -> KResult<()> {
    match ty {
        Ty::Param(name) if name.starts_with("t$") && !allowed.contains(name) => Err(Diag::new("cannot infer this type; use the value so its type is determined".to_string(), span)),
        Ty::Named { args, .. } | Ty::Tuple(args) => {
            for a in args {
                check_unknown_params(a, allowed, span)?;
            }
            Ok(())
        }
        Ty::App(h, a) | Ty::Arrow(h, a) => {
            check_unknown_params(h, allowed, span)?;
            check_unknown_params(a, allowed, span)
        }
        _ => Ok(()),
    }
}

/// Resolves and rigidifies the types carried by a function reference.
fn finalize_ref(infer: &InferCtx, ids: &[u32], allowed: &HashSet<String>, target: &mut FnRef, span: Span) -> KResult<()> {
    let args = match target {
        FnRef::Fn { args, .. } | FnRef::Value { args, .. } | FnRef::Instance { args, .. } | FnRef::Class { args, .. } => args,
        FnRef::Extern { .. } => return Ok(()),
    };
    for a in args {
        finalize_ty(infer, ids, allowed, a, span)?;
    }
    Ok(())
}

fn fixup_self_ref(self_ref: &Option<SelfRef>, generics: &[(String, Kind)], target: &mut FnRef) {
    let generic_args = || generics.iter().map(|(n, _)| Ty::Param(n.clone())).collect::<Vec<_>>();
    match (self_ref, target) {
        (Some(SelfRef::Fn(id, _, _)), FnRef::Fn { id: tid, args }) if id == tid && args.is_empty() => {
            *args = generic_args();
        }
        (Some(SelfRef::Value(id, _, _)), FnRef::Value { id: tid, args }) if id == tid && args.is_empty() => {
            *args = generic_args();
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_pat(infer: &InferCtx, ids: &[u32], allowed: &HashSet<String>, p: &mut HirPat) -> KResult<()> {
    finalize_ty(infer, ids, allowed, &mut p.ty, p.span)?;
    match &mut p.kind {
        HirPatKind::Ctor { fields, .. } | HirPatKind::Tuple(fields) => {
            for f in fields {
                finalize_pat(infer, ids, allowed, f)?;
            }
        }
        HirPatKind::Wild | HirPatKind::Bind(_) | HirPatKind::Int(_) | HirPatKind::Bool(_) => {}
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn finalize_expr(infer: &InferCtx, ids: &[u32], allowed: &HashSet<String>, self_ref: &Option<SelfRef>, generics: &[(String, Kind)], e: &mut HirExpr) -> KResult<()> {
    finalize_ty(infer, ids, allowed, &mut e.ty, e.span)?;
    match &mut e.kind {
        HirExprKind::Const(_) | HirExprKind::Local(_) => {}
        HirExprKind::Fn(target) => {
            finalize_ref(infer, ids, allowed, target, e.span)?;
            fixup_self_ref(self_ref, generics, target);
        }
        HirExprKind::Closure { .. } => {}
        HirExprKind::CallDirect { target, args } => {
            finalize_ref(infer, ids, allowed, target, e.span)?;
            fixup_self_ref(self_ref, generics, target);
            for a in args {
                finalize_expr(infer, ids, allowed, self_ref, generics, a)?;
            }
        }
        HirExprKind::Call { callee, args } => {
            finalize_expr(infer, ids, allowed, self_ref, generics, callee)?;
            for a in args {
                finalize_expr(infer, ids, allowed, self_ref, generics, a)?;
            }
        }
        HirExprKind::Ctor { args, .. } | HirExprKind::Tuple(args) => {
            for a in args {
                finalize_expr(infer, ids, allowed, self_ref, generics, a)?;
            }
        }
        HirExprKind::Field { base, .. } => {
            finalize_expr(infer, ids, allowed, self_ref, generics, base)?;
        }
        HirExprKind::Try { value, .. } => {
            finalize_expr(infer, ids, allowed, self_ref, generics, value)?;
        }
        HirExprKind::Await(value) => {
            finalize_expr(infer, ids, allowed, self_ref, generics, value)?;
        }
        HirExprKind::Let { binds, body } => {
            for (p, value) in binds {
                finalize_pat(infer, ids, allowed, p)?;
                finalize_expr(infer, ids, allowed, self_ref, generics, value)?;
            }
            finalize_expr(infer, ids, allowed, self_ref, generics, body)?;
        }
        HirExprKind::If { cond, then_body, else_body } => {
            finalize_expr(infer, ids, allowed, self_ref, generics, cond)?;
            finalize_expr(infer, ids, allowed, self_ref, generics, then_body)?;
            finalize_expr(infer, ids, allowed, self_ref, generics, else_body)?;
        }
        HirExprKind::Match { scrutinee, arms } => {
            finalize_expr(infer, ids, allowed, self_ref, generics, scrutinee)?;
            for arm in arms {
                finalize_pat(infer, ids, allowed, &mut arm.pattern)?;
                if let Some(guard) = &mut arm.guard {
                    finalize_expr(infer, ids, allowed, self_ref, generics, guard)?;
                }
                finalize_expr(infer, ids, allowed, self_ref, generics, &mut arm.body)?;
            }
        }
        HirExprKind::Bin { lhs, rhs, .. } => {
            finalize_expr(infer, ids, allowed, self_ref, generics, lhs)?;
            finalize_expr(infer, ids, allowed, self_ref, generics, rhs)?;
        }
        HirExprKind::Un { operand, .. } => {
            finalize_expr(infer, ids, allowed, self_ref, generics, operand)?;
        }
    }
    Ok(())
}
