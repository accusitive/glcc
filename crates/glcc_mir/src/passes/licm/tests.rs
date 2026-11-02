use super::*;
use crate::ir::{Const, IcmpPred, InstData, InstKind, Terminator, ValueData};
use crate::ty::{Signature, Ty};
use crate::verify::verify_func;

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

fn kinds_in(func: &Function, block: Block) -> Vec<InstKind> {
    func.blocks[block].insts.iter().map(|i| func.insts[*i].kind.clone()).collect()
}

/// `while (b) { tripled := x*3; bumped := tripled+1; use_cmp(read,
/// bumped); }`: the mul, its add user, and nothing else hoist. The
/// in-loop load stays, and the comparison reading both sides
/// (bumped from outside after the hoist, the load inside) stays.
#[test]
fn hoists_invariants_to_the_preheader() {
    let sig = Signature::new(vec![Ty::int(32), Ty::ptr(Ty::int(32)), Ty::Bool], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let x = param(&mut func, entry, Ty::int(32), 0);
    let p = param(&mut func, entry, Ty::ptr(Ty::int(32)), 1);
    let cond = param(&mut func, entry, Ty::Bool, 2);
    let header = func.push_block(vec![]);
    let latch = func.push_block(vec![]);
    let exit = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::Br {
        dest: header,
        args: Default::default(),
    });
    func.blocks[header].term = Some(Terminator::CondBr {
        cond,
        then_block: latch,
        else_block: exit,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    let c3 = func.push_const(Const::Int { bits: 32, value: 3 });
    let c1 = func.push_const(Const::Int { bits: 32, value: 1 });
    let tripled = emit(&mut func, latch, InstKind::Mul(x, c3), Ty::int(32));
    let bumped = emit(&mut func, latch, InstKind::Add(tripled, c1), Ty::int(32));
    let read = emit(
        &mut func,
        latch,
        InstKind::Load {
            addr: p,
            flags: crate::ir::MemFlags::plain(),
        },
        Ty::int(32),
    );
    let use_cmp = emit(
        &mut func,
        latch,
        InstKind::Icmp {
            pred: IcmpPred::Ne,
            lhs: bumped,
            rhs: read,
        },
        Ty::Bool,
    );
    // The comparison feeds the back edge, closing the loop chain.
    let walk = func.push_block(vec![]);
    func.blocks[latch].term = Some(Terminator::CondBr {
        cond: use_cmp,
        then_block: walk,
        else_block: header,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    func.blocks[walk].term = Some(Terminator::Br {
        dest: header,
        args: Default::default(),
    });
    func.blocks[exit].term = Some(Terminator::Return { val: None });

    assert!(Licm.run(&mut func));
    assert!(verify_func(&func).is_empty());
    let entry_kinds = kinds_in(&func, entry);
    assert!(entry_kinds.iter().any(|k| matches!(k, InstKind::Mul(..))), "mul hoisted: {entry_kinds:?}");
    assert!(entry_kinds.iter().any(|k| matches!(k, InstKind::Add(..))), "add hoisted: {entry_kinds:?}");
    let latch_kinds = kinds_in(&func, latch);
    assert!(matches!(latch_kinds[0], InstKind::Load { .. }), "{latch_kinds:?}");
    assert!(matches!(latch_kinds[1], InstKind::Icmp { .. }), "{latch_kinds:?}");
    assert_eq!(latch_kinds.len(), 2);
    assert!(!Licm.run(&mut func), "second run is a no-op");
    assert!(verify_func(&func).is_empty());
}

/// Division may trap: it is not pure, so it and everything reading
/// it stay in the loop whatever their operands are.
#[test]
fn keeps_trapping_arithmetic_in_the_loop() {
    let sig = Signature::new(vec![Ty::int(32), Ty::Bool], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let x = param(&mut func, entry, Ty::int(32), 0);
    let cond = param(&mut func, entry, Ty::Bool, 1);
    let header = func.push_block(vec![]);
    let latch = func.push_block(vec![]);
    let exit = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::Br {
        dest: header,
        args: Default::default(),
    });
    let c3 = func.push_const(Const::Int { bits: 32, value: 3 });
    let quotient = emit(&mut func, header, InstKind::Sdiv(x, c3), Ty::int(32));
    let _twice = emit(&mut func, header, InstKind::Add(quotient, quotient), Ty::int(32));
    func.blocks[header].term = Some(Terminator::CondBr {
        cond,
        then_block: latch,
        else_block: exit,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    func.blocks[latch].term = Some(Terminator::Br {
        dest: header,
        args: Default::default(),
    });
    func.blocks[exit].term = Some(Terminator::Return { val: None });

    assert!(!Licm.run(&mut func));
    assert!(verify_func(&func).is_empty());
    assert_eq!(kinds_in(&func, entry).len(), 0);
    assert_eq!(kinds_in(&func, header).len(), 2);
}

/// Two ways into the loop: there is no preheader, so nothing may
/// move (some loop entry would never see the hoisted work).
#[test]
fn skips_loops_with_several_entry_edges() {
    let sig = Signature::new(vec![Ty::int(32), Ty::Bool], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let x = param(&mut func, entry, Ty::int(32), 0);
    let cond = param(&mut func, entry, Ty::Bool, 1);
    let second_entry = func.push_block(vec![]);
    let header = func.push_block(vec![]);
    let latch = func.push_block(vec![]);
    let exit = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block: header,
        else_block: second_entry,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    let skip = func.push_const(Const::Int { bits: 1, value: 0 });
    func.blocks[second_entry].term = Some(Terminator::CondBr {
        cond: skip,
        then_block: header,
        else_block: exit,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    let c3 = func.push_const(Const::Int { bits: 32, value: 3 });
    let tripled = emit(&mut func, latch, InstKind::Mul(x, c3), Ty::int(32));
    let _rising = emit(
        &mut func,
        latch,
        InstKind::Icmp {
            pred: IcmpPred::Ne,
            lhs: tripled,
            rhs: x,
        },
        Ty::Bool,
    );
    func.blocks[header].term = Some(Terminator::CondBr {
        cond,
        then_block: latch,
        else_block: exit,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    func.blocks[latch].term = Some(Terminator::Br {
        dest: header,
        args: Default::default(),
    });
    func.blocks[exit].term = Some(Terminator::Return { val: None });

    assert!(!Licm.run(&mut func));
    assert!(verify_func(&func).is_empty());
    assert_eq!(kinds_in(&func, latch).len(), 2);
}

/// An operand defined inside the loop stays behind until its own
/// chain can hoist: `x ^ x` leaves first, then `mul` follows it in
/// the same sweep, and the comparison (which reads the mul in the
/// loop) never does.
#[test]
fn hoists_unblocked_chains_within_one_sweep() {
    let sig = Signature::new(vec![Ty::int(32), Ty::Bool], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let x = param(&mut func, entry, Ty::int(32), 0);
    let cond = param(&mut func, entry, Ty::Bool, 1);
    let header = func.push_block(vec![]);
    let latch = func.push_block(vec![]);
    let exit = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::Br {
        dest: header,
        args: Default::default(),
    });
    let shaken = emit(&mut func, latch, InstKind::Xor(x, x), Ty::int(32));
    let squared = emit(&mut func, latch, InstKind::Mul(shaken, shaken), Ty::int(32));
    // Dead in the CFG (nothing reads it), so it never hoists.
    let _rising = emit(
        &mut func,
        latch,
        InstKind::Icmp {
            pred: IcmpPred::Ne,
            lhs: squared,
            rhs: x,
        },
        Ty::Bool,
    );
    func.blocks[header].term = Some(Terminator::CondBr {
        cond,
        then_block: latch,
        else_block: exit,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    func.blocks[latch].term = Some(Terminator::Br {
        dest: header,
        args: Default::default(),
    });
    func.blocks[exit].term = Some(Terminator::Return { val: None });

    assert!(Licm.run(&mut func));
    assert!(verify_func(&func).is_empty());
    let entry_kinds = kinds_in(&func, entry);
    assert_eq!(entry_kinds.len(), 2, "{entry_kinds:?}");
    let latch_kinds = kinds_in(&func, latch);
    assert!(matches!(latch_kinds.last(), Some(InstKind::Icmp { .. })));
    assert_eq!(latch_kinds.len(), 1, "{latch_kinds:?}");
}

/// A self-loop (`while (1) double := x+x;`): the back edge is the
/// header's own edge, the loop body is just the header, and the
/// invariant leaves through the entry.
#[test]
fn hoists_a_load_from_a_nonescaping_slot() {
    let sig = Signature::new(vec![Ty::Bool], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let cond = param(&mut func, entry, Ty::Bool, 0);
    let slot = emit(&mut func, entry, InstKind::Alloca { pointee: Ty::int(32), count: 1 }, Ty::ptr(Ty::int(32)));
    let header = func.push_block(vec![]);
    let latch = func.push_block(vec![]);
    let exit = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::Br {
        dest: header,
        args: Default::default(),
    });
    func.blocks[header].term = Some(Terminator::CondBr {
        cond,
        then_block: latch,
        else_block: exit,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    let read = emit(
        &mut func,
        latch,
        InstKind::Load {
            addr: slot,
            flags: crate::ir::MemFlags::plain(),
        },
        Ty::int(32),
    );
    let zero = func.push_const(Const::Int { bits: 32, value: 0 });
    let keep_going = emit(
        &mut func,
        latch,
        InstKind::Icmp {
            pred: IcmpPred::Ne,
            lhs: read,
            rhs: zero,
        },
        Ty::Bool,
    );
    func.blocks[latch].term = Some(Terminator::CondBr {
        cond: keep_going,
        then_block: header,
        else_block: exit,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    func.blocks[exit].term = Some(Terminator::Return { val: None });

    assert!(Licm.run(&mut func));
    assert!(verify_func(&func).is_empty());
    let entry_kinds = kinds_in(&func, entry);
    assert!(entry_kinds.iter().any(|k| matches!(k, InstKind::Load { .. })), "{entry_kinds:?}");
    let latch_kinds = kinds_in(&func, latch);
    assert!(latch_kinds.iter().all(|k| !matches!(k, InstKind::Load { .. })), "{latch_kinds:?}");
    assert!(!Licm.run(&mut func), "second run is a no-op");
}

#[test]
fn keeps_a_load_the_loop_can_overwrite() {
    let sig = Signature::new(vec![Ty::Bool], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let cond = param(&mut func, entry, Ty::Bool, 0);
    let slot = emit(&mut func, entry, InstKind::Alloca { pointee: Ty::int(32), count: 1 }, Ty::ptr(Ty::int(32)));
    let header = func.push_block(vec![]);
    let latch = func.push_block(vec![]);
    let exit = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::Br {
        dest: header,
        args: Default::default(),
    });
    func.blocks[header].term = Some(Terminator::CondBr {
        cond,
        then_block: latch,
        else_block: exit,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    let zero = func.push_const(Const::Int { bits: 32, value: 0 });
    let store = func.insts.push(InstData {
        block: latch,
        kind: InstKind::Store {
            addr: slot,
            val: zero,
            flags: crate::ir::MemFlags::plain(),
        },
        result: None,
        loc: None,
    });
    func.blocks[latch].insts.push(store);
    let read = emit(
        &mut func,
        latch,
        InstKind::Load {
            addr: slot,
            flags: crate::ir::MemFlags::plain(),
        },
        Ty::int(32),
    );
    let keep_going = emit(
        &mut func,
        latch,
        InstKind::Icmp {
            pred: IcmpPred::Ne,
            lhs: read,
            rhs: zero,
        },
        Ty::Bool,
    );
    func.blocks[latch].term = Some(Terminator::CondBr {
        cond: keep_going,
        then_block: header,
        else_block: exit,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    func.blocks[exit].term = Some(Terminator::Return { val: None });

    assert!(!Licm.run(&mut func));
    assert!(verify_func(&func).is_empty());
    assert!(kinds_in(&func, latch).iter().any(|k| matches!(k, InstKind::Load { .. })));
}

#[test]
fn keeps_volatile_loads_in_the_loop() {
    let sig = Signature::new(vec![Ty::Bool], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let cond = param(&mut func, entry, Ty::Bool, 0);
    let slot = emit(&mut func, entry, InstKind::Alloca { pointee: Ty::int(32), count: 1 }, Ty::ptr(Ty::int(32)));
    let header = func.push_block(vec![]);
    let latch = func.push_block(vec![]);
    let exit = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::Br {
        dest: header,
        args: Default::default(),
    });
    func.blocks[header].term = Some(Terminator::CondBr {
        cond,
        then_block: latch,
        else_block: exit,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    let read = emit(
        &mut func,
        latch,
        InstKind::Load {
            addr: slot,
            flags: crate::ir::MemFlags::vol(),
        },
        Ty::int(32),
    );
    let zero = func.push_const(Const::Int { bits: 32, value: 0 });
    let keep_going = emit(
        &mut func,
        latch,
        InstKind::Icmp {
            pred: IcmpPred::Ne,
            lhs: read,
            rhs: zero,
        },
        Ty::Bool,
    );
    func.blocks[latch].term = Some(Terminator::CondBr {
        cond: keep_going,
        then_block: header,
        else_block: exit,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    func.blocks[exit].term = Some(Terminator::Return { val: None });

    assert!(!Licm.run(&mut func));
    assert!(verify_func(&func).is_empty());
    assert!(kinds_in(&func, latch).iter().any(|k| matches!(k, InstKind::Load { .. })));
}

#[test]
fn hoists_from_a_self_loop() {
    let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let x = param(&mut func, entry, Ty::int(32), 0);
    let header = func.push_block(vec![]);
    let exit = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::Br {
        dest: header,
        args: Default::default(),
    });
    let cond = func.push_const(Const::Int { bits: 1, value: 1 });
    let c2 = func.push_const(Const::Int { bits: 32, value: 2 });
    let double = emit(&mut func, header, InstKind::Mul(x, c2), Ty::int(32));
    func.blocks[header].term = Some(Terminator::CondBr {
        cond,
        then_block: header,
        else_block: exit,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    func.blocks[exit].term = Some(Terminator::Return { val: Some(double) });
    assert!(Licm.run(&mut func));
    assert!(verify_func(&func).is_empty());
    // The entry grabbed the only computation; the header is empty.
    assert_eq!(kinds_in(&func, entry).len(), 1);
    assert_eq!(kinds_in(&func, header).len(), 0);
    assert!(!Licm.run(&mut func), "second run is a no-op");
}
