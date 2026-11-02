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
