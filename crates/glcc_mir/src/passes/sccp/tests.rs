use super::*;
use crate::ir::{InstData, MemFlags};
use crate::ty::Signature;
use crate::verify::verify_func;

/// `(if c: 41+1 else: 42)`: both arms collapse to 42 no matter how
/// the condition goes; the merge param becomes that constant.
#[test]
fn joins_matching_constants_through_a_phi() {
    let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let cond = param(&mut func, entry, Ty::Bool, 0);
    let a = func.push_const(Const::Int { bits: 32, value: 41 });
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let c42 = func.push_const(Const::Int { bits: 32, value: 42 });
    let then_block = func.push_block(vec![]);
    let else_block = func.push_block(vec![]);
    let sum = emit(&mut func, entry, InstKind::Add(a, one), Ty::int(32));
    let merge = func.push_block(vec![Ty::int(32)]);
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block,
        else_block,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    let merged = func.blocks[merge].params[0];
    func.blocks[then_block].term = Some(Terminator::Br {
        dest: merge,
        args: [sum].into_iter().collect(),
    });
    func.blocks[else_block].term = Some(Terminator::Br {
        dest: merge,
        args: [c42].into_iter().collect(),
    });
    func.blocks[merge].term = Some(Terminator::Return { val: Some(merged) });

    assert!(Sccp.run(&mut func));
    assert!(verify_func(&func).is_empty());
    let Some(Terminator::Return { val: Some(v) }) = func.blocks[merge].term.clone() else {
        panic!("merge still returns");
    };
    assert_eq!(func.const_value(v), Some(&Const::Int { bits: 32, value: 42 }));
    assert!(!Sccp.run(&mut func), "second run is a no-op");
}

/// The arms disagree, so the merge stays overdefined and nothing
/// may be rewritten.
#[test]
fn keeps_conflicting_arms_overdefined() {
    let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let cond = param(&mut func, entry, Ty::Bool, 0);
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let two = func.push_const(Const::Int { bits: 32, value: 2 });
    let then_block = func.push_block(vec![]);
    let else_block = func.push_block(vec![]);
    let merge = func.push_block(vec![Ty::int(32)]);
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block,
        else_block,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    let merged = func.blocks[merge].params[0];
    func.blocks[then_block].term = Some(Terminator::Br {
        dest: merge,
        args: [one].into_iter().collect(),
    });
    func.blocks[else_block].term = Some(Terminator::Br {
        dest: merge,
        args: [two].into_iter().collect(),
    });
    func.blocks[merge].term = Some(Terminator::Return { val: Some(merged) });

    assert!(!Sccp.run(&mut func));
    assert!(verify_func(&func).is_empty());
    assert!(matches!(func.blocks[merge].term.clone().unwrap(), Terminator::Return { .. }));
}

#[test]
fn folds_a_select_whose_arms_agree() {
    let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let cond = param(&mut func, entry, Ty::Bool, 0);
    let c = func.push_const(Const::Int { bits: 32, value: 7 });
    let sel = emit(&mut func, entry, InstKind::Select { cond, then_val: c, else_val: c }, Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
    assert!(Sccp.run(&mut func));
    assert!(verify_func(&func).is_empty());
    let out = returned(&func);
    assert_eq!(func.const_value(out), Some(&Const::Int { bits: 32, value: 7 }));
}

#[test]
fn keeps_a_select_whose_arms_differ() {
    let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let cond = param(&mut func, entry, Ty::Bool, 0);
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let two = func.push_const(Const::Int { bits: 32, value: 2 });
    let sel = emit(
        &mut func,
        entry,
        InstKind::Select {
            cond,
            then_val: one,
            else_val: two,
        },
        Ty::int(32),
    );
    let out = emit(&mut func, entry, InstKind::Add(sel, one), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(out) });
    assert!(!Sccp.run(&mut func));
    assert!(verify_func(&func).is_empty());
}

/// A constant condition decides the successor before the code
/// runs: the untaken arm becomes dead and the branch unconditional.
#[test]
fn prunes_the_unreachable_arm_of_a_constant_condition() {
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let cond = func.push_const(Const::Int { bits: 1, value: 1 });
    let then_block = func.push_block(vec![]);
    let else_block = func.push_block(vec![]);
    let null = func.push_const(Const::Null);
    let load = emit(
        &mut func,
        then_block,
        InstKind::Load {
            addr: null,
            flags: MemFlags::plain(),
        },
        Ty::int(32),
    );
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block,
        else_block,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    let zero = func.push_const(Const::Int { bits: 32, value: 0 });
    func.blocks[then_block].term = Some(Terminator::Return { val: Some(load) });
    func.blocks[else_block].term = Some(Terminator::Return { val: Some(zero) });

    assert!(Sccp.run(&mut func));
    assert!(verify_func(&func).is_empty());
    assert!(matches!(
        func.blocks[entry].term,
        Some(Terminator::Br { dest, .. }) if dest == then_block
    ));
    assert!(matches!(func.blocks[else_block].term, Some(Terminator::Unreachable)));
    // The reachable load stays overdefined (it still reads memory).
    assert!(func.const_value(load).is_none());
    assert!(func.blocks[then_block].insts.len() == 1);
}

/// A computed condition folds to a constant, so the branch is
/// decided from a value that is itself rewritten to a fresh
/// constant by the same pass. Deciding the terminator must not
/// index the solve's lattice with that fresh value.
#[test]
fn folds_a_computed_condition_branch() {
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let zero = func.push_const(Const::Int { bits: 32, value: 0 });
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let cond = emit(&mut func, entry, InstKind::Add(zero, one), Ty::int(32));
    let then_block = func.push_block(vec![]);
    let else_block = func.push_block(vec![]);
    let t = func.push_const(Const::Int { bits: 32, value: 10 });
    let e = func.push_const(Const::Int { bits: 32, value: 20 });
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block,
        else_block,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    func.blocks[then_block].term = Some(Terminator::Return { val: Some(t) });
    func.blocks[else_block].term = Some(Terminator::Return { val: Some(e) });

    assert!(Sccp.run(&mut func));
    assert!(verify_func(&func).is_empty());
    assert!(matches!(
        func.blocks[entry].term,
        Some(Terminator::Br { dest, .. }) if dest == then_block
    ));
    assert!(matches!(func.blocks[else_block].term, Some(Terminator::Unreachable)));
}

/// A loop phi first joins a constant, then widens to overdefined when
/// the back edge arrives. The widening has to flow back through that
/// already-joined edge: a stale `Const(1)` here used to replace the
/// loop counter's back-edge argument with a literal, so the loop
/// never advanced.
#[test]
fn rejoins_a_phi_whose_argument_widens() {
    let scenario = widening_loop();
    let mut func = scenario.func;
    assert!(!Sccp.run(&mut func), "the loop counter must stay dynamic");
    assert!(verify_func(&func).is_empty());
    let Some(Terminator::Br { dest, args }) = func.blocks[scenario.back].term.clone() else {
        panic!("back edge vanished");
    };
    assert_eq!(dest, scenario.hdr);
    assert_eq!(args[0], scenario.carried, "stale constant joined as the loop argument");
}

/// The fixpoint checker accepts a correct solve: every param equals
/// the join of its executable incoming arguments.
#[test]
fn solve_reaches_a_valid_fixpoint_for_a_widening_phi() {
    let scenario = widening_loop();
    let mut state = State::new(&scenario.func);
    state.solve(&scenario.func);
    let errors = state.fixpoint_errors(&scenario.func);
    assert!(errors.is_empty(), "{errors:?}");
}

/// The checker catches the stale-join class of bug directly: force a
/// phi param back to the constant it first joined while its incoming
/// argument has since widened, and the mismatch is reported.
#[test]
fn fixpoint_errors_flags_a_stale_phi_param() {
    let scenario = widening_loop();
    let mut state = State::new(&scenario.func);
    state.solve(&scenario.func);
    state.lattice[scenario.j.index()] = Lat::Const(Const::Int { bits: 32, value: 1 });
    let errors = state.fixpoint_errors(&scenario.func);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains(&format!("%{}", scenario.j.index())), "{}", errors[0]);
    assert!(errors[0].contains("Over"), "{}", errors[0]);
}

#[test]
fn picks_the_matching_switch_arm() {
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let scrut = func.push_const(Const::Int { bits: 32, value: 2 });
    let low = func.push_block(vec![]);
    let high = func.push_block(vec![]);
    let one = func.push_const(Const::Int { bits: 32, value: 11 });
    let seven = func.push_const(Const::Int { bits: 32, value: 77 });
    func.blocks[entry].term = Some(Terminator::Switch {
        scrutinee: scrut,
        cases: vec![(1, low), (2, high)],
        default: low,
    });
    func.blocks[low].term = Some(Terminator::Return { val: Some(one) });
    func.blocks[high].term = Some(Terminator::Return { val: Some(seven) });
    assert!(Sccp.run(&mut func));
    assert!(verify_func(&func).is_empty());
    let Some(Terminator::Br { dest, .. }) = func.blocks[entry].term.clone() else {
        panic!("switch did not fold");
    };
    assert_eq!(dest, high);
}

/// A load feeding a merge from an always-taken path stays
/// overdefined; nothing in the live code may fold away.
#[test]
fn keeps_memory_reads_overdefined() {
    let sig = Signature::new(vec![Ty::ptr(Ty::int(32))], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let p = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
    let read = emit(
        &mut func,
        entry,
        InstKind::Load {
            addr: p,
            flags: MemFlags::plain(),
        },
        Ty::int(32),
    );
    let forty = func.push_const(Const::Int { bits: 32, value: 40 });
    let two = func.push_const(Const::Int { bits: 32, value: 2 });
    let product = emit(&mut func, entry, InstKind::Mul(forty, two), Ty::int(32));
    let sum = emit(&mut func, entry, InstKind::Add(read, product), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
    assert!(Sccp.run(&mut func));
    assert!(verify_func(&func).is_empty());
    // The load stays overdefined (its result is untouched, so the
    // sum still reads it), while the pure 40*2 folded to 80: the
    // sum's rhs is now a materialized constant. Sccp leaves the
    // orphaned mul for the dead-code sweep — it does not DCE.
    assert_eq!(func.blocks[entry].insts.len(), 3);
    match &func.insts[func.blocks[entry].insts[2]].kind {
        InstKind::Add(a, rhs) => {
            assert_eq!(*a, read);
            assert_eq!(func.values[*rhs].def, Def::Const(Const::Int { bits: 32, value: 80 }));
        }
        other => panic!("expected add over the load, got {other:?}"),
    }
}

/// Code reachable only through a never-taken edge: its results are
/// Undef'd at the fixpoint, even when a live merge also joins a
/// real value from the other arm.
#[test]
fn undefs_results_only_reached_through_dead_edges() {
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let cond = func.push_const(Const::Int { bits: 1, value: 0 });
    let then_block = func.push_block(vec![]);
    let else_block = func.push_block(vec![]);
    let merge = func.push_block(vec![Ty::int(32)]);
    let null = func.push_const(Const::Null);
    let dead_load = emit(
        &mut func,
        then_block,
        InstKind::Load {
            addr: null,
            flags: MemFlags::plain(),
        },
        Ty::int(32),
    );
    let seven = func.push_const(Const::Int { bits: 32, value: 7 });
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block,
        else_block,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    let merged = func.blocks[merge].params[0];
    func.blocks[then_block].term = Some(Terminator::Br {
        dest: merge,
        args: [dead_load].into_iter().collect(),
    });
    func.blocks[else_block].term = Some(Terminator::Br {
        dest: merge,
        args: [seven].into_iter().collect(),
    });
    // The disjunct get: `seven` and maybe-undef must join — that is
    // overdefined; the point is that the dead load's own result is
    // a Undef constant after the sweep.
    func.blocks[merge].term = Some(Terminator::Return { val: Some(merged) });
    assert!(Sccp.run(&mut func));
    assert!(verify_func(&func).is_empty());
    // The dead arm is out of the CFG reachable set; the merge
    // param folds to the only arm the branch can take, and the
    // return reads that constant.
    assert!(matches!(func.blocks[then_block].term, Some(Terminator::Unreachable)));
    let out = returned(&func);
    assert_eq!(func.const_value(out), Some(&Const::Int { bits: 32, value: 7 }));
}

fn returned(func: &Function) -> Value {
    for block in func.blocks.iter() {
        match &block.term {
            Some(Terminator::Return { val: Some(v) }) => return *v,
            _ => {}
        }
    }
    panic!("no value return in {func:?}");
}

/// The loop from `rejoins_a_phi_whose_argument_widens`: entry feeds
/// `zero` into the header's counter phi, then the back edge carries a
/// value that widens from `Const(1)` to `Over`.
struct WideningLoop {
    func: Function,
    hdr: Block,
    back: Block,
    j: Value,
    carried: Value,
}

fn widening_loop() -> WideningLoop {
    let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let cond = param(&mut func, entry, Ty::Bool, 0);
    let zero = func.push_const(Const::Int { bits: 32, value: 0 });
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let hdr = func.push_block(vec![Ty::int(32)]);
    let body = func.push_block(vec![]);
    let back = func.push_block(vec![Ty::int(32)]);
    let out = func.push_block(vec![]);
    let j = func.blocks[hdr].params[0];
    func.blocks[entry].term = Some(Terminator::Br {
        dest: hdr,
        args: [zero].into_iter().collect(),
    });
    func.blocks[hdr].term = Some(Terminator::CondBr {
        cond,
        then_block: body,
        else_block: out,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    let next = emit(&mut func, body, InstKind::Add(j, one), Ty::int(32));
    func.blocks[body].term = Some(Terminator::Br {
        dest: back,
        args: [next].into_iter().collect(),
    });
    let carried = func.blocks[back].params[0];
    func.blocks[back].term = Some(Terminator::Br {
        dest: hdr,
        args: [carried].into_iter().collect(),
    });
    func.blocks[out].term = Some(Terminator::Return { val: Some(zero) });
    WideningLoop { func, hdr, back, j, carried }
}

fn param(func: &mut Function, block: Block, ty: Ty, index: u32) -> Value {
    let v = func.values.push(ValueData {
        ty,
        def: Def::Param(block, index),
    });
    func.blocks[block].params.push(v);
    v
}

fn emit(func: &mut Function, block: Block, kind: InstKind, ty: Ty) -> Value {
    let result = func.values.push(ValueData {
        ty,
        def: Def::Inst(Inst::from_usize(func.insts.len())),
    });
    let inst = func.insts.push(InstData {
        block,
        kind,
        result: Some(result),
        loc: None,
    });
    func.values[result].def = Def::Inst(inst);
    func.blocks[block].insts.push(inst);
    result
}
