//! Lowering: checked HIR to MIR, with monomorphization.
//!
//! One MIR function is emitted per reachable body instance. Non-generic
//! bodies are roots; a call with concrete type arguments enqueues the
//! callee's instance (`INSTANCE_LIMIT` caps the total). Checked bodies
//! ran once with generic parameters abstract, so lowering substitutes
//! each instance's type arguments and computes cell layouts on the way.
//!
//! Data values, closures, and tuples are heap cells behind an erased
//! pointer, allocated by `fern_alloc` (`rt/gc.c`), so values never dangle
//! when they escape their frame. Pointer-typed locals and temporaries in
//! flight are rooted through the GC frame protocol (see `gc`).

pub(crate) mod expr;
pub(crate) mod func;
pub(crate) mod gc;
pub(crate) mod types;

use std::collections::HashMap;

use glcc_mir as mir;

use crate::defs::Defs;
use crate::hir::*;
use crate::infer::InferCtx;
use crate::solve::{Obligation, Solution, Solver};
use crate::span::{Diag, KResult, Span};
use crate::ty::Ty;

use types::Types;

/// Hard cap on instantiations, so a runaway generic program fails
/// loudly instead of hanging the compiler.
const INSTANCE_LIMIT: usize = 2048;

/// Which body a pending instance lowers.
enum PendingBody {
    Item(BodyKey),
    Closure(u32),
}

struct Pending {
    body: PendingBody,
    subst: HashMap<String, Ty>,
    func: mir::Func,
    span: Span,
    name: String,
}

pub(crate) struct Lowerer<'d> {
    defs: &'d Defs,
    program: &'d HirProgram,
    pub(crate) types: Types<'d>,
    module: mir::Module,
    bodies: HashMap<BodyKey, &'d HirFn>,
    instances: HashMap<String, mir::Func>,
    thunks: HashMap<String, mir::Func>,
    pending: Vec<Pending>,
    solver: Solver<'d>,
    pub(crate) rt: gc::Runtime,
}

/// Lower a checked program to MIR.
pub fn lower(defs: &Defs, program: &HirProgram) -> KResult<mir::Module> {
    let mut lw = Lowerer::new(defs, program);
    lw.run()?;
    Ok(lw.module)
}

impl<'d> Lowerer<'d> {
    fn new(defs: &'d Defs, program: &'d HirProgram) -> Lowerer<'d> {
        let bodies = program.bodies.iter().map(|b| (b.key, b)).collect();
        let mut module = mir::Module::new();
        let rt = gc::declare(&mut module);
        Lowerer {
            defs,
            program,
            types: Types::new(defs),
            module,
            bodies,
            instances: HashMap::new(),
            thunks: HashMap::new(),
            pending: Vec::new(),
            solver: Solver::new(defs),
            rt,
        }
    }

    fn run(&mut self) -> KResult<()> {
        for body in &self.program.bodies {
            if body.generics.is_empty() {
                let _ = self.declare_body(body.key, HashMap::new())?;
            }
        }
        while let Some(item) = self.pending.pop() {
            self.lower_one(&item)?;
        }
        Ok(())
    }

    fn lower_one(&mut self, item: &Pending) -> KResult<()> {
        let mut func = std::mem::replace(
            &mut self.module.funcs[item.func],
            mir::Function::new("__pending", mir::Signature::new(Vec::new(), mir::Ty::Void)).0,
        );
        let program = self.program;
        let out = match item.body {
            PendingBody::Item(key) => {
                let hir = self
                    .bodies
                    .get(&key)
                    .copied()
                    .expect("pending bodies have checked HIR");
                func::lower_fn(self, &mut func, hir, &item.subst, item.span, item.name.clone())
            }
            PendingBody::Closure(id) => func::lower_closure(
                self,
                &mut func,
                &program.closures[id as usize],
                &item.subst,
                item.span,
                item.name.clone(),
            ),
        };
        self.module.funcs[item.func] = func;
        out
    }

    /// Declare (once per symbol) the MIR function for one instance,
    /// returning the function and its symbol. The symbol comes back
    /// because a body being lowered sits in the module under a dummy
    /// name, and thunks must still key on the real one.
    pub(crate) fn declare_body(
        &mut self,
        key: BodyKey,
        subst: HashMap<String, Ty>,
    ) -> KResult<(mir::Func, String)> {
        let hir = self
            .bodies
            .get(&key)
            .copied()
            .ok_or_else(|| Diag::new("no checked body for this function", Span::new(0, 0)))?;
        let name = self.symbol(hir, &subst)?;
        if let Some(&func) = self.instances.get(&name) {
            return Ok((func, name));
        }
        if self.instances.len() >= INSTANCE_LIMIT {
            return Err(Diag::new(
                format!(
                    "monomorphization limit reached: more than {INSTANCE_LIMIT} function instances"
                ),
                hir.span,
            ));
        }
        let sig = self.instance_sig(hir, &subst)?;
        let (func, _) = mir::Function::new(&name, sig);
        let id = self.module.push_func(func);
        self.instances.insert(name.clone(), id);
        self.pending.push(Pending {
            body: PendingBody::Item(key),
            subst,
            func: id,
            span: hir.span,
            name: name.clone(),
        });
        Ok((id, name))
    }

    /// Declare (once per instance) a closure's MIR function. Its
    /// generics are the enclosing body's, already substituted.
    pub(crate) fn declare_closure(
        &mut self,
        enclosing: &str,
        id: u32,
        subst: &HashMap<String, Ty>,
        span: Span,
    ) -> KResult<mir::Func> {
        let name = format!("{enclosing}$c{id}");
        if let Some(&func) = self.instances.get(&name) {
            return Ok(func);
        }
        if self.instances.len() >= INSTANCE_LIMIT {
            return Err(Diag::new(
                format!(
                    "monomorphization limit reached: more than {INSTANCE_LIMIT} function instances"
                ),
                span,
            ));
        }
        let program = self.program;
        let closure = &program.closures[id as usize];
        let mut params = vec![mir::Ty::ptr(mir::Ty::Void)];
        for p in &closure.params {
            params.push(self.types.get(&p.ty.substitute(subst), p.span)?);
        }
        let ret = self.types.get(&closure.ret.substitute(subst), span)?;
        let (func, _) = mir::Function::new(&name, mir::Signature::new(params, ret));
        let fid = self.module.push_func(func);
        self.instances.insert(name.clone(), fid);
        self.pending.push(Pending {
            body: PendingBody::Closure(id),
            subst: subst.clone(),
            func: fid,
            span: closure.span,
            name,
        });
        Ok(fid)
    }

    /// The thunk that presents a named function as a closure value:
    /// same ABI as real closures, with the environment ignored.
    pub(crate) fn thunk(
        &mut self,
        callee: mir::Func,
        callee_name: &str,
        params: &[mir::Ty],
        ret: &mir::Ty,
        _span: Span,
    ) -> KResult<mir::Func> {
        if let Some(&f) = self.thunks.get(callee_name) {
            return Ok(f);
        }
        let name = format!("{callee_name}$$thunk");
        let mut sig_params = vec![mir::Ty::ptr(mir::Ty::Void)];
        sig_params.extend(params.iter().cloned());
        let sig = mir::Signature::new(sig_params, ret.clone());
        let (mut func, entry) = mir::Function::new(&name, sig);
        let mut args = Vec::with_capacity(params.len());
        for (i, ty) in params.iter().enumerate() {
            let value = func.values.push(mir::ValueData {
                ty: ty.clone(),
                def: mir::Def::Param(entry, (i + 1) as u32),
            });
            func.blocks[entry].params.push(value);
            args.push(value);
        }
        // The environment parameter is unused but must be declared.
        let env = func.values.push(mir::ValueData {
            ty: mir::Ty::ptr(mir::Ty::Void),
            def: mir::Def::Param(entry, 0),
        });
        func.blocks[entry].params.insert(0, env);
        let call = push_inst(
            &mut func,
            entry,
            mir::InstKind::Call {
                callee,
                args: args.into(),
            },
            ret.clone(),
        );
        func.blocks[entry].term = Some(mir::Terminator::Return { val: Some(call) });
        let id = self.module.push_func(func);
        self.thunks.insert(callee_name.to_string(), id);
        Ok(id)
    }

    /// Resolve a call target to a concrete MIR function; the name is
    /// its mangled symbol (used to key thunks).
    pub(crate) fn resolve_callee(
        &mut self,
        target: &FnRef,
        subst: &HashMap<String, Ty>,
        span: Span,
    ) -> KResult<(mir::Func, String)> {
        let (key, concrete) = match target {
            FnRef::Fn { id, args } => (BodyKey::Fn(*id), self.concrete_args(args, subst, span)?),
            FnRef::Value { id, args } => (
                BodyKey::Value(*id),
                self.concrete_args(args, subst, span)?,
            ),
            FnRef::Instance {
                instance,
                method,
                args,
            } => (
                BodyKey::Method {
                    instance: *instance,
                    method: *method,
                },
                self.concrete_args(args, subst, span)?,
            ),
            FnRef::Class {
                class_id,
                method,
                args,
            } => {
                let defs = self.defs;
                let class = &defs.classes[class_id.index()];
                let concrete = self.concrete_args(args, subst, span)?;
                let (class_args, tail) = concrete.split_at(class.params.len());
                if let Some(unresolved) = class_args.iter().find(|t| !t.is_concrete()) {
                    return Err(Diag::new(
                        format!(
                            "cannot resolve `{}::{}` on `{unresolved}`",
                            class.name, class.methods[*method].name
                        ),
                        span,
                    ));
                }
                let goal = Obligation {
                    class_id: *class_id,
                    args: class_args.to_vec(),
                };
                let mut infer = InferCtx::new();
                let (index, found) = match self.solver.prove(&mut infer, &[], &goal, span)? {
                    Solution::Instance { index, args } => (index, args),
                    other => return Err(self.solver.describe_failure(&goal, &other, span)),
                };
                let mut concrete = found;
                concrete.extend(tail.iter().cloned());
                (
                    BodyKey::Method {
                        instance: index,
                        method: *method,
                    },
                    concrete,
                )
            }
        };
        let hir = self.body_of(key, span)?;
        let callee_subst = body_subst(hir, &concrete, span)?;
        let (func, name) = self.declare_body(key, callee_subst)?;
        Ok((func, name))
    }

    fn body_of(&self, key: BodyKey, span: Span) -> KResult<&'d HirFn> {
        self.bodies
            .get(&key)
            .copied()
            .ok_or_else(|| Diag::new("no checked body for this method", span))
    }

    fn concrete_args(
        &self,
        args: &[Ty],
        subst: &HashMap<String, Ty>,
        span: Span,
    ) -> KResult<Vec<Ty>> {
        let out: Vec<Ty> = args.iter().map(|t| t.substitute(subst)).collect();
        if let Some(unresolved) = out.iter().find(|t| !t.is_concrete()) {
            return Err(Diag::new(
                format!("`{unresolved}` reached lowering with unsolved types"),
                span,
            ));
        }
        Ok(out)
    }

    fn instance_sig(&mut self, hir: &HirFn, subst: &HashMap<String, Ty>) -> KResult<mir::Signature> {
        let mut params = Vec::with_capacity(hir.params.len());
        for p in &hir.params {
            params.push(self.types.get(&p.ty.substitute(subst), p.span)?);
        }
        let ret = self.types.get(&hir.ret.substitute(subst), hir.span)?;
        Ok(mir::Signature::new(params, ret))
    }

    /// Mangled symbol: `name` for a non-generic body, `name$$T1$T2` for
    /// a generic one, `Head::Class::method$$T1` for an instance method.
    fn symbol(&self, hir: &HirFn, subst: &HashMap<String, Ty>) -> KResult<String> {
        let base = match hir.key {
            BodyKey::Fn(_) | BodyKey::Value(_) => hir.name.clone(),
            BodyKey::Method { instance, method } => {
                let inst = &self.defs.instances[instance];
                let class = &self.defs.classes[inst.class_id.index()];
                let head = inst.args[0].substitute(subst).mangle();
                format!("{}::{}::{}", head, class.name, class.methods[method].name)
            }
        };
        if hir.generics.is_empty() {
            return Ok(base);
        }
        let mut out = base;
        out.push_str("$$");
        for (i, (g, _)) in hir.generics.iter().enumerate() {
            if i > 0 {
                out.push('$');
            }
            let t = subst.get(g).ok_or_else(|| {
                Diag::new(
                    format!("cannot instantiate `{}`: no type for `{g}`", hir.name),
                    hir.span,
                )
            })?;
            out.push_str(&t.mangle());
        }
        Ok(out)
    }
}

fn body_subst(hir: &HirFn, concrete: &[Ty], span: Span) -> KResult<HashMap<String, Ty>> {
    if hir.generics.len() != concrete.len() {
        return Err(Diag::new(
            format!(
                "`{}` expects {} type argument(s), found {}",
                hir.name,
                hir.generics.len(),
                concrete.len()
            ),
            span,
        ));
    }
    Ok(hir
        .generics
        .iter()
        .map(|(g, _)| g.clone())
        .zip(concrete.iter().cloned())
        .collect())
}

fn push_inst(
    func: &mut mir::Function,
    block: mir::Block,
    kind: mir::InstKind,
    ty: mir::Ty,
) -> mir::Value {
    let result = func.values.push(mir::ValueData {
        ty,
        def: mir::Def::Inst(mir::Inst::from_usize(func.insts.len())),
    });
    let inst = func.insts.push(mir::InstData {
        block,
        kind,
        result: Some(result),
        loc: None,
    });
    func.blocks[block].insts.push(inst);
    result
}
