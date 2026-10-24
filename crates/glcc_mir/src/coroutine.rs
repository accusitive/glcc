//! `LowerCoroutines`: rewrite every `__attribute__((async))` function
//! (`Function::coroutine`) into a start/resume pair over a boxed heap
//! frame. Runs once, right after MIR lowering, and only touches
//! functions that carry `coroutine` — everything else passes through
//! unchanged, so the pass is inert without the `-fglcc-async` frontend
//! flag.
//!
//! Layout (the ABI contract, see `docs/async-abi.md`):
//!
//! ```text
//! frame:  [ result@0 | state:u32 | param slots | local slots | handle slots | spills... ]
//! handle: { void *frame; uint32_t (*poll)(void *frame, glcc_ctx *ctx); }
//! poll:   0 = PENDING (frame stays alive, resume after the child completes)
//!         1 = READY   (result lives at frame+0, awaiter reads it then frees the frame)
//! ```
//!
//! The public symbol keeps the original `Func` id and signature (its
//! result is the `_Future(T)` handle): it allocates the frame, stores
//! the incoming parameters, fills the handle (`frame`, `poll`) and
//! returns it. The coroutine body moves to a synthesized
//! `<name>$resume` whose entry is a state dispatch:
//!
//! * state 0 runs the body from the top;
//! * state `k` jumps to the `wait` block of the k-th await, which
//!   polls the awaited child — ready continues right after the await,
//!   pending returns `PENDING`.
//!
//! Every `Alloca` becomes a constant-offset GEP into the frame, every
//! use of an entry parameter a load from its slot, and every value
//! live across a suspension a frame slot written at its definition and
//! read before each use (the coroutine's own SSA values do not
//! survive a return).

use std::collections::{HashMap, HashSet};

use smallvec::SmallVec;

use crate::ir::{Block, Const, Def, Extern, Func, Function, Inst, InstData, InstKind, MemFlags, Module, Terminator, Value, ValueData};
use crate::pass::ModulePass;
use crate::ty::{GepIndex, Signature, Ty, dense_field_offsets, ty_align};

/// The poll result the coroutine has not finished: the frame stays
/// alive and `poll` must be called again once the child completes.
pub const PENDING: i32 = 0;
/// The poll result the coroutine ran to completion: the result (if any)
/// sits at offset 0 of the frame; the awaiter reads it and frees the
/// frame.
pub const READY: i32 = 1;

/// The waker-facing signature of every resume function: the frame
/// (type-erased) plus the context, yielding a poll result.
pub fn poll_sig() -> Signature {
    Signature::new(vec![Ty::ptr(Ty::Void), Ty::ptr(Ty::Void)], Ty::int(32))
}

/// One `__builtin_await` in a coroutine body, with the blocks the
/// rewrite builds for it.
#[derive(Debug, Clone, Copy)]
struct Site {
    block: Block,
    /// Position of the `Await` within `block`.
    pos: usize,
    /// The handle to await: an async call's `_Future(T)`.
    fut: Value,
    /// The value the await produces, if the result type is not void.
    result: Option<Value>,
    /// 1-based resume state; 0 means "run the body from the top".
    state: u32,
    /// Re-entry point on resume: polls the child, then branches to
    /// `after` (ready) or `pending` (not yet).
    wait: Block,
    /// Continuation: loads the child's result, frees the child frame,
    /// then runs the instructions that followed the await.
    after: Block,
    /// Suspension exit: returns `PENDING` to the poller.
    pending: Block,
}

/// The frame schema the pass hands to `build_start` so the start
/// function agrees with the resume body about slots.
struct FrameLayout {
    ty: Ty,
    size: u64,
    align: u64,
    state_field: u32,
    param_fields: Vec<u32>,
}

pub struct LowerCoroutines;

impl ModulePass for LowerCoroutines {
    fn name(&self) -> &'static str {
        "lower-coroutines"
    }

    fn run(&mut self, module: &mut Module) -> bool {
        let work: Vec<Func> = module.funcs.iter_enumerated().filter(|(_, f)| f.coroutine.is_some()).map(|(id, _)| id).collect();
        if work.is_empty() {
            return false;
        }
        for id in work {
            lower_one(module, id);
        }
        // The awaits themselves stay in the arena (their results feed
        // the fresh child-frame loads) but must be in no block list.
        debug_assert!(
            module
                .funcs
                .iter()
                .all(|f| f.blocks.iter().all(|b| b.insts.iter().all(|i| !matches!(f.insts[*i].kind, InstKind::Await { .. })))),
            "await instruction survived coroutine lowering"
        );
        true
    }
}

/// Intern the runtime frame hooks (see `rt/glcc_async.h`); both are
/// weak in the runtime so an embedder can override the allocator.
fn intern_extern(module: &mut Module, name: &str, sig: Signature) -> Extern {
    if let Some((id, _)) = module.externs.iter_enumerated().find(|(_, e)| e.name == name) {
        return id;
    }
    module.push_extern(name, sig)
}

fn frame_alloc_sig() -> Signature {
    Signature::new(vec![Ty::int(64), Ty::int(64)], Ty::ptr(Ty::Void))
}

fn frame_free_sig() -> Signature {
    Signature::new(vec![Ty::ptr(Ty::Void)], Ty::Void)
}

fn lower_one(module: &mut Module, id: Func) {
    // The original function stays the public start symbol (call sites
    // keep referring to this `Func`); its body moves to the resume
    // function cloned here.
    let mut resume = module.funcs[id].clone();
    let result_ty = resume.coroutine.take().expect("coroutine function");
    let handle_ty = resume.sig.ret.clone();
    resume.sig = poll_sig();
    resume.name = format!("{}$resume", resume.name);
    resume.local = true;
    resume.weak = false;
    resume.inline = false;
    resume.debug = None;

    let free = intern_extern(module, "__glcc_frame_free", frame_free_sig());
    let alloc = intern_extern(module, "__glcc_frame_alloc", frame_alloc_sig());

    let layout = lower_body(&mut resume, &result_ty, &handle_ty, free);
    let resume_id = module.push_func(resume);
    build_start(module, id, resume_id, alloc, &layout, &handle_ty);
}

// ---- instruction helpers ----

/// Create an instruction in `block` without appending it to the block's
/// list; callers control instruction order themselves.
fn new_inst(func: &mut Function, block: Block, kind: InstKind, ty: Option<Ty>) -> (Inst, Option<Value>) {
    let result = ty.map(|ty| {
        func.values.push(ValueData {
            ty,
            def: Def::Inst(Inst::from_usize(func.insts.len())),
        })
    });
    let inst = func.insts.push(InstData { block, kind, result, loc: None });
    if let Some(v) = result {
        func.values[v].def = Def::Inst(inst);
    }
    (inst, result)
}

/// Create an instruction and append it to `block`.
fn emit(func: &mut Function, block: Block, kind: InstKind, ty: Option<Ty>) -> Option<Value> {
    let (inst, result) = new_inst(func, block, kind, ty);
    func.blocks[block].insts.push(inst);
    result
}

fn emit_val(func: &mut Function, block: Block, kind: InstKind, ty: Ty) -> Value {
    emit(func, block, kind, Some(ty)).expect("instruction yields a value")
}

fn iconst(func: &mut Function, bits: u32, value: i128) -> Value {
    func.push_const(Const::Int { bits, value })
}

fn gep(func: &mut Function, block: Block, base: Value, steps: &[GepIndex], result: Ty) -> Value {
    emit_val(
        func,
        block,
        InstKind::GetElementPtr {
            base,
            indices: steps.iter().cloned().collect(),
        },
        result,
    )
}

fn load(func: &mut Function, block: Block, addr: Value, ty: Ty) -> Value {
    emit_val(func, block, InstKind::Load { addr, flags: MemFlags::plain() }, ty)
}

fn store(func: &mut Function, block: Block, addr: Value, val: Value) {
    emit(
        func,
        block,
        InstKind::Store {
            addr,
            val,
            flags: MemFlags::plain(),
        },
        None,
    );
}

// ---- the resume body ----

/// Rewrite the cloned coroutine body in place; returns the frame layout
/// the start function must mirror.
fn lower_body(resume: &mut Function, result_ty: &Ty, handle_ty: &Ty, free: Extern) -> FrameLayout {
    let old_entry = resume.entry;

    // Every await, in block/position order (deterministic state ids).
    let mut found: Vec<Site> = Vec::new();
    let mut state = 0u32;
    for (block, data) in resume.blocks.iter_enumerated() {
        for (pos, &inst) in data.insts.iter().enumerate() {
            if let InstKind::Await { fut } = &resume.insts[inst].kind {
                state += 1;
                found.push(Site {
                    block,
                    pos,
                    fut: *fut,
                    result: resume.insts[inst].result,
                    state,
                    wait: Block::from_usize(0),
                    after: Block::from_usize(0),
                    pending: Block::from_usize(0),
                });
            }
        }
    }

    // Entry parameters (stored by the start function) and stack slots
    // (their storage moves into the frame).
    let params: Vec<(Value, Ty)> = resume.blocks[old_entry].params.iter().map(|&v| (v, resume.values[v].ty.clone())).collect();
    let mut allocas: Vec<(Inst, Value, Ty, u64)> = Vec::new();
    for (inst, data) in resume.insts.iter_enumerated() {
        if let InstKind::Alloca { pointee, count } = &data.kind
            && let Some(val) = data.result {
                allocas.push((inst, val, pointee.clone(), *count));
            }
    }

    // Which values must survive each suspension: the live set right
    // after each await. Everything else either stays in SSA (values
    // defined after the await or re-materialized on resume) or already
    // lives in memory.
    let needs = live_after_awaits(resume, &found);
    let awaited: HashSet<Value> = found.iter().filter_map(|s| s.result).collect();
    // An await's result is normally consumed before the next suspension.
    // When it is still live after a later await it must live in the frame
    // too: resuming skips the block that loaded it from the child.
    let spilled_awaits: HashSet<Value> = found
        .iter()
        .enumerate()
        .filter_map(|(i, site)| {
            let result = site.result?;
            needs.iter().enumerate().any(|(j, live)| j != i && live.contains(&result)).then_some(result)
        })
        .collect();
    let mut spill: Vec<Value> = Vec::new();
    {
        let mut all: Vec<Value> = needs.iter().flatten().copied().collect();
        all.sort_by_key(|v| v.index());
        let mut seen = HashSet::new();
        for v in all {
            if (awaited.contains(&v) && !spilled_awaits.contains(&v)) || !seen.insert(v) {
                continue;
            }
            match &resume.values[v].def {
                // Constants re-materialize on demand.
                Def::Const(_) => continue,
                // Entry parameters are replaced by slot loads below.
                Def::Param(b, _) if *b == old_entry => continue,
                // Allocas become frame GEPs, which the dispatch block
                // re-creates on every entry.
                Def::Inst(i) if matches!(&resume.insts[*i].kind, InstKind::Alloca { .. }) => continue,
                _ => {}
            }
            spill.push(v);
        }
    }

    // Frame layout. Offset 0 is the result so a type-erased awaiter can
    // read it without knowing the coroutine (the ABI contract).
    let mut fields: Vec<Ty> = Vec::new();
    let result_field = (!result_ty.is_void()).then(|| {
        fields.push(result_ty.clone());
        (fields.len() - 1) as u32
    });
    let state_field = {
        fields.push(Ty::int(32));
        (fields.len() - 1) as u32
    };
    let param_fields: Vec<u32> = params
        .iter()
        .map(|(_, ty)| {
            fields.push(ty.clone());
            (fields.len() - 1) as u32
        })
        .collect();
    let mut alloca_field: HashMap<Inst, (u32, Ty)> = HashMap::new();
    for (inst, _, pointee, count) in &allocas {
        // Always an array, even for a single element: `[Field(k),
        // ConstIndex(0)]` then reaches `Ptr(pointee)` whatever the
        // pointee is — a bare struct field has no such step, and the
        // layout is identical (size = elem * len, align = elem align).
        fields.push(Ty::Array {
            elem: Box::new(pointee.clone()),
            len: *count,
        });
        alloca_field.insert(*inst, ((fields.len() - 1) as u32, pointee.clone()));
    }
    let handle_fields: Vec<u32> = (0..found.len())
        .map(|_| {
            fields.push(handle_ty.clone());
            (fields.len() - 1) as u32
        })
        .collect();
    let mut spill_field: HashMap<Value, u32> = HashMap::new();
    for v in &spill {
        fields.push(resume.values[*v].ty.clone());
        spill_field.insert(*v, (fields.len() - 1) as u32);
    }
    let offsets = dense_field_offsets(&fields);
    let align = fields.iter().map(ty_align).max().unwrap_or(1).max(1);
    let mut size = 0u64;
    for (f, o) in fields.iter().zip(&offsets) {
        size = size.max(o + f.size_bytes());
    }
    let size = size.next_multiple_of(align);
    let frame_ty = Ty::Struct { fields, offsets, size };

    // The dispatch block becomes the entry: it intercepts every start
    // and every resume, warms up the frame-derived values (parameter
    // loads, slot addresses) and switches on the saved state.
    let dispatch = resume.push_block(vec![Ty::ptr(frame_ty.clone()), Ty::ptr(Ty::Void)]);
    let frame = resume.blocks[dispatch].params[0];
    let ctx = resume.blocks[dispatch].params[1];
    let old_preds: Vec<Block> = resume
        .blocks
        .iter_enumerated()
        .filter(|(b, data)| *b != old_entry && data.term.as_ref().is_some_and(|t| t.successors().contains(&old_entry)))
        .map(|(b, _)| b)
        .collect();
    debug_assert!(old_preds.is_empty(), "entry block is a jump target; the coroutine dispatch cannot replace its parameters");
    resume.entry = dispatch;
    resume.blocks[old_entry].params.clear();

    let state_addr = gep(resume, dispatch, frame, &[GepIndex::Field(state_field)], Ty::ptr(Ty::int(32)));
    let state_val = load(resume, dispatch, state_addr, Ty::int(32));

    // Parameters: the resume entry has none, so every use loads its
    // slot (written by the start function). The prologue stores that
    // lower_mir emitted keep working unchanged on top of these loads.
    for (i, (p, ty)) in params.iter().enumerate() {
        let addr = gep(resume, dispatch, frame, &[GepIndex::Field(param_fields[i])], Ty::ptr(ty.clone()));
        let v = load(resume, dispatch, addr, ty.clone());
        resume.replace_uses(*p, v);
    }

    // Stack slots: constant-offset GEPs into the frame. The GEPs live in
    // dispatch, so they are valid on every entry, start or resume.
    let alloca_set: HashSet<Inst> = allocas.iter().map(|(i, ..)| *i).collect();
    for (inst, val, pointee, _) in &allocas {
        let (field, _) = alloca_field[inst];
        let addr = gep(resume, dispatch, frame, &[GepIndex::Field(field), GepIndex::ConstIndex(0)], Ty::ptr(pointee.clone()));
        resume.replace_uses(*val, addr);
    }
    // Slot addresses warming up in dispatch: handle fields, the result,
    // and the spill slots.
    let result_addr = result_field.map(|f| gep(resume, dispatch, frame, &[GepIndex::Field(f)], Ty::ptr(result_ty.clone())));
    let mut handle_slots: Vec<(Value, Value, Value)> = Vec::new();
    for &hf in &handle_fields {
        let slot = gep(resume, dispatch, frame, &[GepIndex::Field(hf)], Ty::ptr(handle_ty.clone()));
        let f0 = gep(resume, dispatch, frame, &[GepIndex::Field(hf), GepIndex::Field(0)], Ty::ptr(Ty::ptr(Ty::Void)));
        let f1 = gep(resume, dispatch, frame, &[GepIndex::Field(hf), GepIndex::Field(1)], Ty::ptr(Ty::ptr(Ty::Void)));
        handle_slots.push((slot, f0, f1));
    }
    let mut spill_addr: HashMap<Value, Value> = HashMap::new();
    for v in &spill {
        let ty = resume.values[*v].ty.clone();
        let addr = gep(resume, dispatch, frame, &[GepIndex::Field(spill_field[v])], Ty::ptr(ty));
        spill_addr.insert(*v, addr);
    }

    // Completion: every return stores the result (offset 0) and then
    // reports READY. Done before the split so the pending exits the
    // split creates stay untouched (they return PENDING).
    for bid in 0..resume.blocks.len() {
        let block = Block::from_usize(bid);
        let term = resume.blocks[block].term.clone();
        let Some(Terminator::Return { val }) = term else {
            continue;
        };
        if let (Some(raddr), false) = (&result_addr, result_ty.is_void()) {
            let v = val.unwrap_or_else(|| {
                resume.values.push(ValueData {
                    ty: result_ty.clone(),
                    def: Def::Const(Const::Undef),
                })
            });
            store(resume, block, *raddr, v);
        }
        let ready = iconst(resume, 32, READY as i128);
        resume.blocks[block].term = Some(Terminator::Return { val: Some(ready) });
    }

    // Assign the per-await blocks, then cut each block at its awaits.
    let sites: Vec<Site> = found
        .into_iter()
        .map(|s| {
            let wait = resume.push_block(vec![]);
            let after = resume.push_block(vec![]);
            let pending = resume.push_block(vec![]);
            Site { wait, after, pending, ..s }
        })
        .collect();
    resume.blocks[dispatch].term = Some(dispatch_term(&sites, state_val, old_entry));
    let mut child_frames: Vec<Value> = vec![Value::from_usize(0); sites.len()];
    let mut await_loads: HashMap<Value, Value> = HashMap::new();
    let mut k = 0;
    while k < sites.len() {
        let mut j = k;
        while j < sites.len() && sites[j].block == sites[k].block {
            j += 1;
        }
        split_block(resume, &sites[k..j], k, &handle_slots, state_addr, ctx, free, &mut child_frames, &mut await_loads);
        k = j;
    }
    // The awaits are gone and every slot got a frame home: the original
    // `Alloca`s have no remaining uses, so drop them (their positions
    // shifting is fine now that the split is done).
    for bid in 0..resume.blocks.len() {
        let block = Block::from_usize(bid);
        resume.blocks[block].insts.retain(|i| !alloca_set.contains(i));
    }
    // Spilled await results are represented by the load that copies the
    // child's result before its frame is freed.
    for (&result, &load) in &await_loads {
        if let Some(&addr) = spill_addr.get(&result) {
            spill_addr.insert(load, addr);
        }
    }
    spill = spill.iter().map(|v| await_loads.get(v).copied().unwrap_or(*v)).collect();

    // Spills: write the slot at the definition, read it before every
    // use. A single value may be stored once and re-loaded in many
    // places (branches after the suspension each need their own load).
    let mut store_of: HashMap<Value, Inst> = HashMap::new();
    for v in &spill {
        let addr = spill_addr[v];
        match resume.values[*v].def.clone() {
            Def::Inst(def_inst) => {
                let block = resume.insts[def_inst].block;
                let pos = resume.blocks[block]
                    .insts
                    .iter()
                    .position(|&i| i == def_inst)
                    .expect("defining instruction lives in its block");
                let (st, _) = new_inst(
                    resume,
                    block,
                    InstKind::Store {
                        addr,
                        val: *v,
                        flags: MemFlags::plain(),
                    },
                    None,
                );
                resume.blocks[block].insts.insert(pos + 1, st);
                store_of.insert(*v, st);
            }
            Def::Param(block, _) => {
                debug_assert_ne!(block, old_entry, "entry parameters were replaced");
                let (st, _) = new_inst(
                    resume,
                    block,
                    InstKind::Store {
                        addr,
                        val: *v,
                        flags: MemFlags::plain(),
                    },
                    None,
                );
                resume.blocks[block].insts.insert(0, st);
                store_of.insert(*v, st);
            }
            Def::Const(_) => unreachable!("constants are never spilled"),
        }
    }
    for v in &spill {
        let addr = spill_addr[v];
        let ty = resume.values[*v].ty.clone();
        let skip = store_of[v];
        for bid in 0..resume.blocks.len() {
            let block = Block::from_usize(bid);
            let old = std::mem::take(&mut resume.blocks[block].insts);
            let mut rebuilt: Vec<Inst> = Vec::with_capacity(old.len());
            for inst in old {
                if inst == skip {
                    rebuilt.push(inst);
                    continue;
                }
                if inst_uses(resume, inst, *v) {
                    let (ld, lv) = new_inst(resume, block, InstKind::Load { addr, flags: MemFlags::plain() }, Some(ty.clone()));
                    rebuilt.push(ld);
                    let lv = lv.expect("spill load is typed");
                    resume.insts[inst].kind = crate::ir::remap_inst(&resume.insts[inst].kind, |x| if x == *v { lv } else { x });
                }
                rebuilt.push(inst);
            }
            resume.blocks[block].insts = rebuilt;
            let term = resume.blocks[block].term.take();
            if let Some(term) = term {
                let uses = term.operands().contains(v);
                let term = if uses {
                    let (ld, lv) = new_inst(resume, block, InstKind::Load { addr, flags: MemFlags::plain() }, Some(ty.clone()));
                    resume.blocks[block].insts.push(ld);
                    crate::ir::remap_term(&term, |x| if x == *v { lv.expect("spill load is typed") } else { x })
                } else {
                    term
                };
                resume.blocks[block].term = Some(term);
            }
        }
    }

    FrameLayout {
        ty: frame_ty,
        size,
        align,
        state_field,
        param_fields,
    }
}

/// Cut one block at each await: the segment before the k-th await ends
/// by saving the handle and the resume state, then branches to `wait`;
/// the segment after continues in `after`. The awaits themselves are
/// dropped (they belong to no block afterwards). `base` is the global
/// index of `sites[0]`.
#[allow(clippy::too_many_arguments)]
fn split_block(
    func: &mut Function,
    sites: &[Site],
    base: usize,
    handle_slots: &[(Value, Value, Value)],
    state_addr: Value,
    ctx: Value,
    free: Extern,
    child_frames: &mut [Value],
    await_loads: &mut HashMap<Value, Value>,
) {
    let block = sites[0].block;
    let n = sites.len();
    let taken = std::mem::take(&mut func.blocks[block].insts);
    let saved_term = func.blocks[block].term.take();
    // The original terminator is detached while segments are rebuilt, so
    // result replacements have to be replayed on it when it is restored.
    let mut await_results: Vec<(Value, Value)> = Vec::new();

    // Segment inst lists: segment k is everything between await k-1
    // and await k; the final segment runs to the original terminator.
    let mut lists: Vec<Vec<Inst>> = Vec::with_capacity(n + 1);
    let mut cursor = 0;
    for site in sites.iter() {
        lists.push(taken[cursor..site.pos].to_vec());
        cursor = site.pos + 1;
    }
    lists.push(taken[cursor..].to_vec());

    // Poll entries: poll the child once; ready continues, pending
    // suspends (the frame — including this state — stays alive).
    for (i, site) in sites.iter().enumerate() {
        let (_, f0, f1) = handle_slots[base + i];
        let wait = site.wait;
        let fr = load(func, wait, f0, Ty::ptr(Ty::Void));
        let poll = load(func, wait, f1, Ty::ptr(Ty::Void));
        let mut args = SmallVec::new();
        args.push(fr);
        args.push(ctx);
        let r = emit_val(
            func,
            wait,
            InstKind::CallIndirect {
                callee: poll,
                args,
                sig: poll_sig(),
            },
            Ty::int(32),
        );
        let ready = iconst(func, 32, READY as i128);
        let done = emit_val(
            func,
            wait,
            InstKind::Icmp {
                pred: crate::ir::IcmpPred::Eq,
                lhs: r,
                rhs: ready,
            },
            Ty::Bool,
        );
        func.blocks[wait].term = Some(Terminator::CondBr {
            cond: done,
            then_block: site.after,
            else_block: site.pending,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        let pending = iconst(func, 32, PENDING as i128);
        func.blocks[site.pending].term = Some(Terminator::Return { val: Some(pending) });
        child_frames[base + i] = fr;
    }

    // Segments: the k-th receiving block runs from the (k-1)-th await's
    // continuation to the k-th await, then suspends.
    for i in 0..=n {
        let target = if i == 0 { block } else { sites[i - 1].after };
        let mut list: Vec<Inst> = Vec::new();
        if i > 0 {
            let site = &sites[i - 1];
            let child = child_frames[base + i - 1];
            // Copy the awaited result out *before* freeing the child's
            // frame. The type is the await's own, not the coroutine's:
            // a `void` coroutine can still await a value it uses.
            if let Some(res) = site.result {
                let res_ty = func.values[res].ty.clone();
                let (ld, lv) = new_inst(
                    func,
                    target,
                    InstKind::Load {
                        addr: child,
                        flags: MemFlags::plain(),
                    },
                    Some(res_ty),
                );
                let lv = lv.expect("result load is typed");
                func.replace_uses(res, lv);
                await_results.push((res, lv));
                await_loads.insert(res, lv);
                list.push(ld);
            }
            let mut args = SmallVec::new();
            args.push(child);
            let (call, _) = new_inst(func, target, InstKind::CallExtern { target: free, args }, None);
            list.push(call);
        }
        for &inst in &lists[i] {
            func.insts[inst].block = target;
            list.push(inst);
        }
        if i < n {
            let site = &sites[i];
            let (slot, _, _) = handle_slots[base + i];
            let (save, _) = new_inst(
                func,
                target,
                InstKind::Store {
                    addr: slot,
                    val: site.fut,
                    flags: MemFlags::plain(),
                },
                None,
            );
            list.push(save);
            let state = iconst(func, 32, site.state as i128);
            let (mark, _) = new_inst(
                func,
                target,
                InstKind::Store {
                    addr: state_addr,
                    val: state,
                    flags: MemFlags::plain(),
                },
                None,
            );
            list.push(mark);
            func.blocks[target].insts = list;
            func.blocks[target].term = Some(Terminator::Br {
                dest: site.wait,
                args: Default::default(),
            });
        } else {
            func.blocks[target].insts = list;
            let term = saved_term.clone().map(|mut term| {
                for &(old, new) in &await_results {
                    term = crate::ir::remap_term(&term, |v| if v == old { new } else { v });
                }
                term
            });
            func.blocks[target].term = term;
        }
    }
}

/// The dispatch terminator: state 0 falls through to the original entry,
/// state `k` re-enters the k-th await's poll block. A coroutine with no
/// awaits needs no state at all.
fn dispatch_term(sites: &[Site], state: Value, old_entry: Block) -> Terminator {
    if sites.is_empty() {
        return Terminator::Br {
            dest: old_entry,
            args: Default::default(),
        };
    }
    Terminator::Switch {
        scrutinee: state,
        cases: sites.iter().map(|s| (s.state as i128, s.wait)).collect(),
        default: old_entry,
    }
}

/// Liveness for the values that must survive a suspension: the live set
/// at the point right after each await, computed on the pre-split CFG.
fn live_after_awaits(func: &Function, sites: &[Site]) -> Vec<HashSet<Value>> {
    let n = func.blocks.len();
    let succs: Vec<Vec<Block>> = (0..n)
        .map(|b| {
            let block = Block::from_usize(b);
            func.blocks[block].term.as_ref().map(|t| t.successors().iter().copied().collect()).unwrap_or_default()
        })
        .collect();
    let mut live_in: Vec<HashSet<Value>> = vec![HashSet::new(); n];
    let mut changed = true;
    while changed {
        changed = false;
        for b in 0..n {
            let block = Block::from_usize(b);
            let mut live: HashSet<Value> = HashSet::new();
            for s in &succs[b] {
                live.extend(live_in[s.index()].iter().copied());
            }
            if let Some(term) = &func.blocks[block].term {
                term.for_each_operand(&mut |o| {
                    live.insert(o);
                });
            }
            for &inst in func.blocks[block].insts.iter().rev() {
                let data = &func.insts[inst];
                if let Some(res) = data.result {
                    live.remove(&res);
                }
                data.kind.for_each_operand(&mut |o| {
                    live.insert(o);
                });
            }
            for &p in &func.blocks[block].params {
                live.remove(&p);
            }
            if live != live_in[b] {
                live_in[b] = live;
                changed = true;
            }
        }
    }
    sites
        .iter()
        .map(|site| {
            let mut live: HashSet<Value> = HashSet::new();
            for s in &succs[site.block.index()] {
                live.extend(live_in[s.index()].iter().copied());
            }
            let term = func.blocks[site.block].term.as_ref();
            if let Some(term) = term {
                term.for_each_operand(&mut |o| {
                    live.insert(o);
                });
            }
            for &inst in func.blocks[site.block].insts[(site.pos + 1)..].iter().rev() {
                let data = &func.insts[inst];
                if let Some(res) = data.result {
                    live.remove(&res);
                }
                data.kind.for_each_operand(&mut |o| {
                    live.insert(o);
                });
            }
            live
        })
        .collect()
}

/// Whether `inst` mentions `v`.
fn inst_uses(func: &Function, inst: Inst, v: Value) -> bool {
    let mut uses = false;
    func.insts[inst].kind.for_each_operand(&mut |o| {
        if o == v {
            uses = true;
        }
    });
    uses
}

// ---- the start function ----

fn build_start(module: &mut Module, id: Func, resume_id: Func, alloc: Extern, layout: &FrameLayout, handle_ty: &Ty) {
    let func = &mut module.funcs[id];
    func.coroutine = None;
    func.inline = false;
    func.debug = None;
    let params: Vec<Ty> = func.sig.params.clone();

    func.values = Default::default();
    func.insts = Default::default();
    func.blocks = Default::default();
    let entry = func.push_block(params.clone());
    func.entry = entry;

    let Ty::Struct { fields, .. } = handle_ty else {
        panic!("async result handle is not a struct");
    };
    assert_eq!(fields.len(), 2, "async result handle is two words");

    let size = iconst(func, 64, layout.size as i128);
    let align = iconst(func, 64, layout.align as i128);
    let mut args = SmallVec::new();
    args.push(size);
    args.push(align);
    let raw = emit_val(func, entry, InstKind::CallExtern { target: alloc, args }, Ty::ptr(Ty::Void));
    let fp = emit_val(func, entry, InstKind::Bitcast(raw), Ty::ptr(layout.ty.clone()));

    let state_addr = gep(func, entry, fp, &[GepIndex::Field(layout.state_field)], Ty::ptr(Ty::int(32)));
    let zero = iconst(func, 32, 0);
    store(func, entry, state_addr, zero);

    for (i, ty) in params.iter().enumerate() {
        let addr = gep(func, entry, fp, &[GepIndex::Field(layout.param_fields[i])], Ty::ptr(ty.clone()));
        store(func, entry, addr, func.blocks[entry].params[i]);
    }

    // The handle is returned by value: build it in a slot and load it
    // (aggregate values lower to addresses; the load copies the pair).
    let slot = emit_val(
        func,
        entry,
        InstKind::Alloca {
            pointee: handle_ty.clone(),
            count: 1,
        },
        Ty::ptr(handle_ty.clone()),
    );
    let f0 = gep(func, entry, slot, &[GepIndex::Field(0)], Ty::ptr(Ty::ptr(Ty::Void)));
    store(func, entry, f0, raw);
    let f1 = gep(func, entry, slot, &[GepIndex::Field(1)], Ty::ptr(Ty::ptr(Ty::Void)));
    let poll = emit_val(func, entry, InstKind::FuncAddr { func: resume_id }, Ty::ptr(Ty::Void));
    store(func, entry, f1, poll);
    let handle = load(func, entry, slot, handle_ty.clone());
    func.blocks[entry].term = Some(Terminator::Return { val: Some(handle) });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::FunctionBuilder;
    use crate::verify::verify_func;

    /// The `_Future(T)` handle shape the frontend synthesizes: two
    /// type-erased words.
    fn handle_ty() -> Ty {
        Ty::Struct {
            fields: vec![Ty::ptr(Ty::Void), Ty::ptr(Ty::Void)],
            offsets: vec![0, 8],
            size: 16,
        }
    }

    /// A handle-typed value. Aggregate values are addresses, so the
    /// rewrite never inspects where it came from.
    fn handle_value(func: &mut Function) -> Value {
        func.values.push(ValueData {
            ty: handle_ty(),
            def: Def::Const(Const::Undef),
        })
    }

    fn add_param(func: &mut Function, ty: Ty) -> Value {
        let entry = func.entry;
        let i = func.blocks[entry].params.len() as u32;
        let v = func.values.push(ValueData { ty, def: Def::Param(entry, i) });
        func.blocks[entry].params.push(v);
        v
    }

    fn await_value(func: &mut Function, block: Block) -> Value {
        let fut = handle_value(func);
        emit(func, block, InstKind::Await { fut }, Some(Ty::int(32))).expect("await yields the child result")
    }

    /// `int f(int x)`: `x + 1` is live across the suspension; the
    /// awaited result is added to it and returned.
    fn awaiting_function() -> Function {
        let (mut func, entry) = Function::new("f", Signature::new(vec![Ty::int(32)], handle_ty()));
        func.coroutine = Some(Ty::int(32));
        let x = add_param(&mut func, Ty::int(32));
        let one = iconst(&mut func, 32, 1);
        let keep = {
            let mut b = FunctionBuilder::new(&mut func, entry);
            b.add(Ty::int(32), x, one)
        };
        let res = await_value(&mut func, entry);
        let sum = {
            let mut b = FunctionBuilder::new(&mut func, entry);
            b.add(Ty::int(32), res, keep)
        };
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        func
    }

    /// `int g(int x) { int v[2]; v[0] = x; int r = __builtin_await(h); return v[0] + r; }`:
    /// a stack slot whose address outlives the suspension.
    fn slot_function() -> Function {
        let (mut func, entry) = Function::new("g", Signature::new(vec![Ty::int(32)], handle_ty()));
        func.coroutine = Some(Ty::int(32));
        let x = add_param(&mut func, Ty::int(32));
        let slot = emit(&mut func, entry, InstKind::Alloca { pointee: Ty::int(32), count: 2 }, Some(Ty::ptr(Ty::int(32)))).expect("alloca yields an address");
        store(&mut func, entry, slot, x);
        let res = await_value(&mut func, entry);
        let loaded = load(&mut func, entry, slot, Ty::int(32));
        let sum = {
            let mut b = FunctionBuilder::new(&mut func, entry);
            b.add(Ty::int(32), loaded, res)
        };
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        func
    }

    /// `int h(void) { int a = __builtin_await(f0); int b = __builtin_await(f1); return a + b; }`:
    /// the first result is still live after the second suspension.
    fn two_awaits_function() -> Function {
        let (mut func, entry) = Function::new("h", Signature::new(vec![], handle_ty()));
        func.coroutine = Some(Ty::int(32));
        let a = await_value(&mut func, entry);
        let b = await_value(&mut func, entry);
        let sum = {
            let mut builder = FunctionBuilder::new(&mut func, entry);
            builder.add(Ty::int(32), a, b)
        };
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        func
    }

    fn lowered(func: Function) -> Module {
        let mut module = Module::new();
        module.push_func(func);
        assert!(LowerCoroutines.run(&mut module), "the pass rewrites");
        assert_eq!(module.funcs.len(), 2, "one resume function appears");
        module
    }

    fn resume_of(module: &Module) -> Func {
        module
            .funcs
            .iter_enumerated()
            .find(|(_, f)| f.name.ends_with("$resume"))
            .map(|(id, _)| id)
            .expect("resume function")
    }

    fn returns_of(func: &Function) -> Vec<Option<Const>> {
        func.blocks
            .iter()
            .filter_map(|b| match &b.term {
                Some(Terminator::Return { val }) => Some(val.map(|v| func.const_value(v).cloned().unwrap_or(Const::Undef))),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn lowers_into_a_start_and_resume_pair() {
        let module = lowered(awaiting_function());
        let start = &module.funcs[Func::from_usize(0)];
        let resume_id = resume_of(&module);
        let resume = &module.funcs[resume_id];

        // The public symbol keeps its name, gains the handle result and
        // becomes a one-block frame setup.
        assert_eq!(start.name, "f");
        assert!(start.coroutine.is_none(), "the flag is consumed");
        assert_eq!(start.sig, Signature::new(vec![Ty::int(32)], handle_ty()));
        assert_eq!(start.blocks.len(), 1);
        let Some(Terminator::Return { val: Some(handle) }) = start.blocks[start.entry].term.clone() else {
            panic!("start returns the handle");
        };
        assert_eq!(start.values[handle].ty, handle_ty());
        let kinds: Vec<InstKind> = start.blocks[start.entry].insts.iter().map(|i| start.insts[*i].kind.clone()).collect();
        assert!(kinds.iter().any(|k| matches!(k, InstKind::CallExtern { .. })), "the start allocates the frame");
        assert!(
            kinds.iter().any(|k| matches!(k, InstKind::FuncAddr { func } if *func == resume_id)),
            "the handle points at the resume entry"
        );

        // The body moves to `f$resume`, poll-shaped.
        assert_eq!(resume.name, "f$resume");
        assert_eq!(resume.sig, poll_sig());
        assert_eq!(resume.blocks[resume.entry].params.len(), 2);
        let Some(Terminator::Switch { cases, default, .. }) = resume.blocks[resume.entry].term.clone() else {
            panic!("dispatch switches on the resume state");
        };
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].0, 1, "states are 1-based");
        assert_eq!(default.index(), 0, "state 0 falls back to the body top");
        let wait = cases[0].1;
        assert!(
            resume.blocks[wait].insts.iter().any(|i| matches!(resume.insts[*i].kind, InstKind::CallIndirect { .. })),
            "the wait block polls the child"
        );

        let returns = returns_of(resume);
        assert!(
            returns.contains(&Some(Const::Int { bits: 32, value: READY as i128 })),
            "completion reports READY: {returns:?}"
        );
        assert!(
            returns.contains(&Some(Const::Int { bits: 32, value: PENDING as i128 })),
            "suspension reports PENDING: {returns:?}"
        );

        assert!(verify_func(start).is_empty(), "{:?}", verify_func(start));
        assert!(verify_func(resume).is_empty(), "{:?}", verify_func(resume));
    }

    #[test]
    fn spills_values_that_outlive_a_suspension() {
        let module = lowered(awaiting_function());
        let resume = &module.funcs[resume_of(&module)];

        // The live `x + 1` gets a frame slot: a store of an add result
        // where it is defined...
        let spilled = resume.blocks.iter().any(|b| {
            b.insts.iter().any(|i| match &resume.insts[*i].kind {
                InstKind::Store { val, .. } => match resume.values[*val].def {
                    Def::Inst(def) => matches!(resume.insts[def].kind, InstKind::Add(..)),
                    _ => false,
                },
                _ => false,
            })
        });
        assert!(spilled, "live value stored to the frame");

        // ... and a load back from a frame slot (a GEP off the frame
        // parameter) where it is used after the suspension.
        let reloaded = resume.blocks.iter().any(|b| {
            b.insts.iter().any(|i| match &resume.insts[*i].kind {
                InstKind::Load { addr, .. } => match resume.values[*addr].def {
                    Def::Inst(def) => matches!(resume.insts[def].kind, InstKind::GetElementPtr { .. }),
                    _ => false,
                },
                _ => false,
            })
        });
        assert!(reloaded, "live value reloaded from the frame");
    }

    #[test]
    fn await_results_live_across_a_later_await_are_spilled() {
        let module = lowered(two_awaits_function());
        let resume = &module.funcs[resume_of(&module)];

        let from_frame = |v: Value| match resume.values[v].def {
            Def::Inst(inst) => match &resume.insts[inst].kind {
                InstKind::Load { addr, .. } => match resume.values[*addr].def {
                    Def::Inst(gep) => match &resume.insts[gep].kind {
                        InstKind::GetElementPtr { base, .. } => resume.values[*base].def == Def::Param(resume.entry, 0),
                        _ => false,
                    },
                    _ => false,
                },
                _ => false,
            },
            _ => false,
        };

        let spilled = resume.blocks.iter().any(|b| {
            b.insts.iter().any(|i| match resume.insts[*i].kind {
                InstKind::Add(lhs, rhs) => from_frame(lhs) || from_frame(rhs),
                _ => false,
            })
        });
        assert!(spilled, "the first await's result is reloaded from the frame for the add");
    }

    #[test]
    fn moves_stack_slots_into_the_frame() {
        let module = lowered(slot_function());
        let resume = &module.funcs[resume_of(&module)];

        assert!(
            !resume
                .blocks
                .iter()
                .any(|b| b.insts.iter().any(|i| matches!(resume.insts[*i].kind, InstKind::Alloca { .. }))),
            "slots live in the frame, not on the stack"
        );

        // Every use of the slot is one GEP off the frame parameter, so
        // the store and the load after the suspension share it.
        let mut addrs: Vec<Value> = Vec::new();
        for b in resume.blocks.iter() {
            for &i in &b.insts {
                match &resume.insts[i].kind {
                    InstKind::Store { addr, .. } | InstKind::Load { addr, .. } => {
                        if let Def::Inst(def) = resume.values[*addr].def {
                            if let InstKind::GetElementPtr { base, indices } = &resume.insts[def].kind {
                                if matches!(indices.as_slice(), [GepIndex::Field(_), GepIndex::ConstIndex(0)]) && resume.values[*base].def == Def::Param(resume.entry, 0) {
                                    addrs.push(*addr);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        assert!(addrs.len() >= 2, "the slot is addressed twice: {addrs:?}");
        assert!(addrs.iter().all(|a| *a == addrs[0]), "one frame GEP is shared: {addrs:?}");
    }

    #[test]
    fn coroutine_without_awaits_still_becomes_a_pair() {
        let (mut func, entry) = Function::new("h", Signature::new(vec![Ty::int(32)], handle_ty()));
        func.coroutine = Some(Ty::int(32));
        let x = add_param(&mut func, Ty::int(32));
        let one = iconst(&mut func, 32, 1);
        let sum = {
            let mut b = FunctionBuilder::new(&mut func, entry);
            b.add(Ty::int(32), x, one)
        };
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });

        let module = lowered(func);
        let start = &module.funcs[Func::from_usize(0)];
        let resume = &module.funcs[resume_of(&module)];

        assert_eq!(start.sig, Signature::new(vec![Ty::int(32)], handle_ty()));
        assert_eq!(resume.blocks.len(), 2, "dispatch plus the body");
        assert!(matches!(resume.blocks[resume.entry].term, Some(Terminator::Br { .. })), "no states without suspensions");
        assert!(verify_func(start).is_empty());
        assert!(verify_func(resume).is_empty());
    }

    #[test]
    fn await_result_passed_to_a_branch_is_remapped() {
        let (mut func, entry) = Function::new("tail", Signature::new(vec![], handle_ty()));
        func.coroutine = Some(Ty::int(32));
        let join = func.push_block(vec![Ty::int(32)]);
        let joined = func.blocks[join].params[0];
        let res = await_value(&mut func, entry);
        func.blocks[entry].term = Some(Terminator::Br {
            dest: join,
            args: [res].into_iter().collect(),
        });
        func.blocks[join].term = Some(Terminator::Return { val: Some(joined) });

        let module = lowered(func);
        let resume = &module.funcs[resume_of(&module)];
        let passes_the_load = resume.blocks.iter().any(|b| {
            let Some(Terminator::Br { args, .. }) = &b.term else {
                return false;
            };
            args.iter().any(|&arg| match resume.values[arg].def {
                Def::Inst(inst) => matches!(resume.insts[inst].kind, InstKind::Load { .. }),
                _ => false,
            })
        });
        assert!(passes_the_load, "the branch passes the resumed load, not the stale await value");
        assert!(verify_func(resume).is_empty(), "{:?}", verify_func(resume));
    }

    #[test]
    fn leaves_plain_functions_alone() {
        let mut module = Module::new();
        let mut func = Function::new("plain", Signature::new(vec![], Ty::int(32))).0;
        let (one, two) = (0i128, 2i128);
        let a = iconst(&mut func, 32, one);
        let b = iconst(&mut func, 32, two);
        let entry = func.entry;
        let sum = {
            let mut fb = FunctionBuilder::new(&mut func, entry);
            fb.add(Ty::int(32), a, b)
        };
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        module.push_func(func);

        assert!(!LowerCoroutines.run(&mut module));
        assert_eq!(module.funcs.len(), 1, "no resume is synthesized");
    }
}
