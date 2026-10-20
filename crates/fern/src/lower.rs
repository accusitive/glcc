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

use crate::defs::{Defs, ExternId};
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
    /// The MIR import for each source-level extern.
    externs: HashMap<ExternId, mir::Extern>,
    pending: Vec<Pending>,
    solver: Solver<'d>,
    pub(crate) rt: gc::Runtime,
    /// Suffix for anonymous string globals.
    pub(crate) str_count: u32,
    /// How many source-level externs are already in `module`.
    declared_externs: usize,
    /// Bodies fully lowered, for incremental-lowering statistics.
    lowered: usize,
}

/// The lowering state a session keeps between inputs: the MIR module
/// compiled so far plus the instance/thunk tables that keep a resumed
/// lowering from emitting duplicate symbols.
pub struct LowerCache {
    module: mir::Module,
    instances: HashMap<String, mir::Func>,
    thunks: HashMap<String, mir::Func>,
    externs: HashMap<ExternId, mir::Extern>,
    rt: gc::Runtime,
    str_count: u32,
    declared_externs: usize,
    lowered: usize,
}

impl LowerCache {
    pub fn module(&self) -> &mir::Module {
        &self.module
    }

    /// How many bodies this cache has lowered in total. A caller
    /// comparing the count across a resume sees only the new work.
    pub fn lowered(&self) -> usize {
        self.lowered
    }
}

/// Lower a checked program to MIR.
pub fn lower(defs: &Defs, program: &HirProgram) -> KResult<mir::Module> {
    Ok(lower_with_cache(defs, program)?.module)
}

/// Lower every root of `program` and hand back the state for later
/// [`lower_more`] calls.
pub fn lower_with_cache(defs: &Defs, program: &HirProgram) -> KResult<LowerCache> {
    let mut lw = Lowerer::new(defs, program)?;
    lw.run()?;
    Ok(lw.into_cache())
}

/// Lower the roots in `roots` on top of a previous [`LowerCache`],
/// leaving every already-lowered instance in place.
pub fn lower_more(defs: &Defs, program: &HirProgram, cache: LowerCache, roots: &[BodyKey]) -> KResult<LowerCache> {
    let mut lw = Lowerer::with_cache(defs, program, Some(cache))?;
    lw.run_roots(roots)?;
    Ok(lw.into_cache())
}

impl<'d> Lowerer<'d> {
    fn new(defs: &'d Defs, program: &'d HirProgram) -> KResult<Lowerer<'d>> {
        Self::with_cache(defs, program, None)
    }

    fn with_cache(defs: &'d Defs, program: &'d HirProgram, cache: Option<LowerCache>) -> KResult<Lowerer<'d>> {
        let bodies = program.bodies.iter().map(|b| (b.key, b)).collect();
        let (mut module, mut externs, instances, thunks, rt, str_count, declared_externs, lowered) = match cache {
            Some(cache) => (
                cache.module,
                cache.externs,
                cache.instances,
                cache.thunks,
                cache.rt,
                cache.str_count,
                cache.declared_externs,
                cache.lowered,
            ),
            None => {
                let mut module = mir::Module::new();
                let rt = gc::declare(&mut module);
                (module, HashMap::new(), HashMap::new(), HashMap::new(), rt, 0, 0, 0)
            }
        };
        let mut types = Types::new(defs);
        for (index, def) in defs.externs.iter().enumerate().skip(declared_externs) {
            let mut params = Vec::with_capacity(def.params.len());
            for param in &def.params {
                params.push(types.get(param, def.span)?);
            }
            let ret = types.get(&def.ret, def.span)?;
            let target = module.push_extern(&def.symbol, mir::Signature::new(params, ret));
            externs.insert(ExternId(index as u32), target);
        }
        let declared_externs = declared_externs.max(defs.externs.len());
        Ok(Lowerer {
            defs,
            program,
            types,
            module,
            bodies,
            instances,
            thunks,
            externs,
            pending: Vec::new(),
            solver: Solver::new(defs),
            rt,
            str_count,
            declared_externs,
            lowered,
        })
    }

    fn run(&mut self) -> KResult<()> {
        let roots: Vec<BodyKey> = self.program.bodies.iter().filter(|b| b.generics.is_empty()).map(|b| b.key).collect();
        self.run_roots(&roots)
    }

    pub(crate) fn run_roots(&mut self, roots: &[BodyKey]) -> KResult<()> {
        for &key in roots {
            let body = self
                .bodies
                .get(&key)
                .copied()
                .ok_or_else(|| Diag::new("no checked body for this function", Span::new(0, 0)))?;
            if body.generics.is_empty() {
                let _ = self.declare_body(key, HashMap::new())?;
            }
        }
        while let Some(item) = self.pending.pop() {
            self.lower_one(&item)?;
        }
        Ok(())
    }

    fn into_cache(self) -> LowerCache {
        LowerCache {
            module: self.module,
            instances: self.instances,
            thunks: self.thunks,
            externs: self.externs,
            rt: self.rt,
            str_count: self.str_count,
            declared_externs: self.declared_externs,
            lowered: self.lowered,
        }
    }

    fn lower_one(&mut self, item: &Pending) -> KResult<()> {
        let mut func = std::mem::replace(
            &mut self.module.funcs[item.func],
            mir::Function::new("__pending", mir::Signature::new(Vec::new(), mir::Ty::Void)).0,
        );
        let program = self.program;
        let out = match item.body {
            PendingBody::Item(key) => {
                let hir = self.bodies.get(&key).copied().expect("pending bodies have checked HIR");
                func::lower_fn(self, &mut func, hir, &item.subst, item.span, item.name.clone())
            }
            PendingBody::Closure(id) => func::lower_closure(self, &mut func, &program.closures[id as usize], &item.subst, item.span, item.name.clone()),
        };
        self.module.funcs[item.func] = func;
        self.lowered += 1;
        out
    }

    /// Declare (once per symbol) the MIR function for one instance,
    /// returning the function and its symbol. The symbol comes back
    /// because a body being lowered sits in the module under a dummy
    /// name, and thunks must still key on the real one.
    pub(crate) fn declare_body(&mut self, key: BodyKey, subst: HashMap<String, Ty>) -> KResult<(mir::Func, String)> {
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
                format!("monomorphization limit reached: more than {INSTANCE_LIMIT} function instances"),
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
    pub(crate) fn declare_closure(&mut self, enclosing: &str, id: u32, subst: &HashMap<String, Ty>, span: Span) -> KResult<mir::Func> {
        let name = format!("{enclosing}$c{id}");
        if let Some(&func) = self.instances.get(&name) {
            return Ok(func);
        }
        if self.instances.len() >= INSTANCE_LIMIT {
            return Err(Diag::new(format!("monomorphization limit reached: more than {INSTANCE_LIMIT} function instances"), span));
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
    pub(crate) fn thunk(&mut self, callee: mir::Func, callee_name: &str, params: &[mir::Ty], ret: &mir::Ty, _span: Span) -> KResult<mir::Func> {
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
        let call = push_inst(&mut func, entry, mir::InstKind::Call { callee, args: args.into() }, ret.clone());
        func.blocks[entry].term = Some(mir::Terminator::Return { val: Some(call) });
        let id = self.module.push_func(func);
        self.thunks.insert(callee_name.to_string(), id);
        Ok(id)
    }

    /// The closure thunk for an extern: same ABI as a real closure, but
    /// the body is a direct foreign call and the environment is ignored.
    pub(crate) fn thunk_extern(&mut self, target: mir::Extern, name: &str, params: &[mir::Ty], ret: &mir::Ty) -> KResult<mir::Func> {
        let key = format!("{name}$$extern");
        if let Some(&f) = self.thunks.get(&key) {
            return Ok(f);
        }
        let mut sig_params = vec![mir::Ty::ptr(mir::Ty::Void)];
        sig_params.extend(params.iter().cloned());
        let sig = mir::Signature::new(sig_params, ret.clone());
        let (mut func, entry) = mir::Function::new(&format!("{key}$$thunk"), sig);
        let env = func.values.push(mir::ValueData {
            ty: mir::Ty::ptr(mir::Ty::Void),
            def: mir::Def::Param(entry, 0),
        });
        func.blocks[entry].params.push(env);
        // Unlike a fern callee, the C callee cannot pin its parameters,
        // so the thunk roots every pointer argument for the call.
        let frame = push_inst(
            &mut func,
            entry,
            mir::InstKind::CallExtern {
                target: self.rt.enter,
                args: Default::default(),
            },
            mir::Ty::int(64),
        );
        let mut args = Vec::with_capacity(params.len());
        let mut slot = 0i128;
        for (i, ty) in params.iter().enumerate() {
            let value = func.values.push(mir::ValueData {
                ty: ty.clone(),
                def: mir::Def::Param(entry, (i + 1) as u32),
            });
            func.blocks[entry].params.push(value);
            args.push(value);
            if gc::is_ptr(ty) {
                let slot_value = func.push_const(mir::Const::Int { bits: 64, value: slot });
                push_void_inst(
                    &mut func,
                    entry,
                    mir::InstKind::CallExtern {
                        target: self.rt.pin,
                        args: vec![frame, slot_value, value].into(),
                    },
                );
                slot += 1;
            }
        }
        let call = push_inst(
            &mut func,
            entry,
            mir::InstKind::CallExtern { target, args: args.into() },
            ret.clone(),
        );
        push_void_inst(
            &mut func,
            entry,
            mir::InstKind::CallExtern {
                target: self.rt.leave,
                args: vec![frame].into(),
            },
        );
        func.blocks[entry].term = Some(mir::Terminator::Return { val: Some(call) });
        let id = self.module.push_func(func);
        self.thunks.insert(key, id);
        Ok(id)
    }

    /// Resolve a call target to a concrete MIR function; the name is
    /// its mangled symbol (used to key thunks).
    pub(crate) fn resolve_callee(&mut self, target: &FnRef, subst: &HashMap<String, Ty>, span: Span) -> KResult<(mir::Func, String)> {
        let (key, concrete) = match target {
            FnRef::Extern { .. } => return Err(Diag::new("internal: extern call reached body resolution", span)),
            FnRef::Fn { id, args } => (BodyKey::Fn(*id), self.concrete_args(args, subst, span)?),
            FnRef::Value { id, args } => (BodyKey::Value(*id), self.concrete_args(args, subst, span)?),
            FnRef::Instance { instance, method, args } => (
                BodyKey::Method {
                    instance: *instance,
                    method: *method,
                },
                self.concrete_args(args, subst, span)?,
            ),
            FnRef::Class { class_id, method, args } => {
                let defs = self.defs;
                let class = &defs.classes[class_id.index()];
                let concrete = self.concrete_args(args, subst, span)?;
                let (class_args, tail) = concrete.split_at(class.params.len());
                if let Some(unresolved) = class_args.iter().find(|t| !t.is_concrete()) {
                    return Err(Diag::new(
                        format!("cannot resolve `{}::{}` on `{unresolved}`", class.name, class.methods[*method].name),
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
                (BodyKey::Method { instance: index, method: *method }, concrete)
            }
        };
        let hir = self.body_of(key, span)?;
        let callee_subst = body_subst(hir, &concrete, span)?;
        let (func, name) = self.declare_body(key, callee_subst)?;
        Ok((func, name))
    }

    fn body_of(&self, key: BodyKey, span: Span) -> KResult<&'d HirFn> {
        self.bodies.get(&key).copied().ok_or_else(|| Diag::new("no checked body for this method", span))
    }

    fn concrete_args(&self, args: &[Ty], subst: &HashMap<String, Ty>, span: Span) -> KResult<Vec<Ty>> {
        let out: Vec<Ty> = args.iter().map(|t| t.substitute(subst)).collect();
        if let Some(unresolved) = out.iter().find(|t| !t.is_concrete()) {
            return Err(Diag::new(format!("`{unresolved}` reached lowering with unsolved types"), span));
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
            BodyKey::Fn(_) | BodyKey::Value(_) => hir.name.replace("::", "$"),
            BodyKey::Method { instance, method } => {
                let inst = &self.defs.instances[instance];
                let class = &self.defs.classes[inst.class_id.index()];
                let head = inst.args[0].substitute(subst).mangle();
                format!(
                    "{}::{}::{}",
                    head,
                    class.name.replace("::", "$"),
                    class.methods[method].name
                )
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
            let t = subst
                .get(g)
                .ok_or_else(|| Diag::new(format!("cannot instantiate `{}`: no type for `{g}`", hir.name), hir.span))?;
            out.push_str(&t.mangle());
        }
        Ok(out)
    }
}

fn body_subst(hir: &HirFn, concrete: &[Ty], span: Span) -> KResult<HashMap<String, Ty>> {
    if hir.generics.len() != concrete.len() {
        return Err(Diag::new(
            format!("`{}` expects {} type argument(s), found {}", hir.name, hir.generics.len(), concrete.len()),
            span,
        ));
    }
    Ok(hir.generics.iter().map(|(g, _)| g.clone()).zip(concrete.iter().cloned()).collect())
}

fn push_inst(func: &mut mir::Function, block: mir::Block, kind: mir::InstKind, ty: mir::Ty) -> mir::Value {
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

fn push_void_inst(func: &mut mir::Function, block: mir::Block, kind: mir::InstKind) {
    let inst = func.insts.push(mir::InstData {
        block,
        kind,
        result: None,
        loc: None,
    });
    func.blocks[block].insts.push(inst);
}
