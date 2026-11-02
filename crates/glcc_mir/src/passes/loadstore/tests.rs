use super::*;
use crate::ir::{Const, Def, Extern, Inst, InstData, MemFlags, Terminator, ValueData};
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

fn ptr_func() -> (Function, Block, Value) {
    let sig = Signature::new(vec![Ty::ptr(Ty::int(32))], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let p = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
    (func, entry, p)
}

fn int_const(func: &mut Function, value: i128) -> Value {
    func.push_const(Const::Int { bits: 32, value })
}

fn run(func: &mut Function) -> bool {
    let changed = LoadStoreForward.run(func);
    let errors = verify_func(func);
    assert!(errors.is_empty(), "verifier: {errors:?}");
    changed
}

fn returned(func: &Function) -> Value {
    match &func.blocks[func.entry].term {
        Some(Terminator::Return { val: Some(value) }) => *value,
        other => panic!("expected a value return, got {other:?}"),
    }
}

#[test]
fn store_then_load_forwards_the_stored_value() {
    let (mut func, entry, p) = ptr_func();
    let five = int_const(&mut func, 5);
    emit(
        &mut func,
        entry,
        InstKind::Store {
            addr: p,
            val: five,
            flags: MemFlags::plain(),
        },
        Ty::Void,
    );
    let loaded = emit(
        &mut func,
        entry,
        InstKind::Load {
            addr: p,
            flags: MemFlags::plain(),
        },
        Ty::int(32),
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    assert!(run(&mut func));
    assert_eq!(func.blocks[entry].insts.len(), 1, "load stays removed");
    assert_eq!(returned(&func), five);
    assert!(!run(&mut func), "second run is a no-op");
}

#[test]
fn duplicate_loads_share_one_result() {
    let (mut func, entry, p) = ptr_func();
    let first = emit(
        &mut func,
        entry,
        InstKind::Load {
            addr: p,
            flags: MemFlags::plain(),
        },
        Ty::int(32),
    );
    let second = emit(
        &mut func,
        entry,
        InstKind::Load {
            addr: p,
            flags: MemFlags::plain(),
        },
        Ty::int(32),
    );
    let sum = emit(&mut func, entry, InstKind::Add(first, second), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });

    assert!(run(&mut func));
    assert_eq!(func.blocks[entry].insts.len(), 2, "second load removed");
    assert!(matches!(func.insts[func.blocks[entry].insts[1]].kind, InstKind::Add(a, b) if a == first && b == first));
}

#[test]
fn a_may_aliasing_store_invalidates_loads() {
    let (mut func, entry, p) = ptr_func();
    let q = param(&mut func, entry, Ty::ptr(Ty::int(32)), 1);
    let first = emit(
        &mut func,
        entry,
        InstKind::Load {
            addr: p,
            flags: MemFlags::plain(),
        },
        Ty::int(32),
    );
    let zero = int_const(&mut func, 0);
    emit(
        &mut func,
        entry,
        InstKind::Store {
            addr: q,
            val: zero,
            flags: MemFlags::plain(),
        },
        Ty::Void,
    );
    let second = emit(
        &mut func,
        entry,
        InstKind::Load {
            addr: p,
            flags: MemFlags::plain(),
        },
        Ty::int(32),
    );
    let sum = emit(&mut func, entry, InstKind::Add(first, second), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });

    assert!(!run(&mut func), "the second load must stay");
    assert_eq!(func.blocks[entry].insts.len(), 4);
}

#[test]
fn independent_stack_slots_keep_their_values() {
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let a = emit(&mut func, entry, InstKind::Alloca { pointee: Ty::int(32), count: 1 }, Ty::ptr(Ty::int(32)));
    let b = emit(&mut func, entry, InstKind::Alloca { pointee: Ty::int(32), count: 1 }, Ty::ptr(Ty::int(32)));
    let five = int_const(&mut func, 5);
    let eight = int_const(&mut func, 8);
    emit(
        &mut func,
        entry,
        InstKind::Store {
            addr: a,
            val: five,
            flags: MemFlags::plain(),
        },
        Ty::Void,
    );
    emit(
        &mut func,
        entry,
        InstKind::Store {
            addr: b,
            val: eight,
            flags: MemFlags::plain(),
        },
        Ty::Void,
    );
    let loaded = emit(
        &mut func,
        entry,
        InstKind::Load {
            addr: a,
            flags: MemFlags::plain(),
        },
        Ty::int(32),
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    assert!(run(&mut func));
    assert_eq!(returned(&func), five);
}

#[test]
fn orphaned_address_use_does_not_block_forwarding() {
    let sig = Signature::new(vec![Ty::ptr(Ty::int(32))], Ty::int(32));
    let (mut func, entry) = Function::new("g", sig);
    let p = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
    let slot = emit(&mut func, entry, InstKind::Alloca { pointee: Ty::int(32), count: 1 }, Ty::ptr(Ty::int(32)));
    let five = int_const(&mut func, 5);
    emit(
        &mut func,
        entry,
        InstKind::Store {
            addr: slot,
            val: five,
            flags: MemFlags::plain(),
        },
        Ty::Void,
    );
    let call = emit(
        &mut func,
        entry,
        InstKind::CallExtern {
            target: Extern::from_usize(0),
            args: [slot].into_iter().collect(),
        },
        Ty::Void,
    );
    let call_inst = match &func.values[call].def {
        crate::ir::Def::Inst(inst) => *inst,
        _ => panic!("call result is an instruction"),
    };
    func.blocks[entry].insts.retain(|inst| *inst != call_inst);

    let nine = int_const(&mut func, 9);
    emit(
        &mut func,
        entry,
        InstKind::Store {
            addr: p,
            val: nine,
            flags: MemFlags::plain(),
        },
        Ty::Void,
    );
    let loaded = emit(
        &mut func,
        entry,
        InstKind::Load {
            addr: slot,
            flags: MemFlags::plain(),
        },
        Ty::int(32),
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    assert!(run(&mut func), "the dead call no longer makes the slot escape");
    assert_eq!(returned(&func), five);
}

#[test]
fn neutral_extern_does_not_block_forwarding() {
    let (mut func, entry, p) = ptr_func();
    func.set_extern_effects(&[crate::ir::ExternEffect::neutral()]);
    let five = int_const(&mut func, 5);
    emit(
        &mut func,
        entry,
        InstKind::Store {
            addr: p,
            val: five,
            flags: MemFlags::plain(),
        },
        Ty::Void,
    );
    emit(
        &mut func,
        entry,
        InstKind::CallExtern {
            target: Extern::from_usize(0),
            args: Default::default(),
        },
        Ty::Void,
    );
    let loaded = emit(
        &mut func,
        entry,
        InstKind::Load {
            addr: p,
            flags: MemFlags::plain(),
        },
        Ty::int(32),
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    assert!(run(&mut func));
    assert_eq!(returned(&func), five);
}

#[test]
fn allocator_results_do_not_alias() {
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("alloc", sig);
    func.set_extern_effects(&[crate::ir::ExternEffect::allocator()]);
    let size = func.push_const(Const::Int { bits: 64, value: 8 });
    let ptr_ty = Ty::ptr(Ty::int(32));
    let target = Extern::from_usize(0);
    let a = emit(
        &mut func,
        entry,
        InstKind::CallExtern {
            target,
            args: [size].into_iter().collect(),
        },
        ptr_ty.clone(),
    );
    let b = emit(
        &mut func,
        entry,
        InstKind::CallExtern {
            target,
            args: [size].into_iter().collect(),
        },
        ptr_ty,
    );
    let five = int_const(&mut func, 5);
    let seven = int_const(&mut func, 7);
    emit(
        &mut func,
        entry,
        InstKind::Store {
            addr: a,
            val: five,
            flags: MemFlags::plain(),
        },
        Ty::Void,
    );
    emit(
        &mut func,
        entry,
        InstKind::Store {
            addr: b,
            val: seven,
            flags: MemFlags::plain(),
        },
        Ty::Void,
    );
    let loaded = emit(
        &mut func,
        entry,
        InstKind::Load {
            addr: a,
            flags: MemFlags::plain(),
        },
        Ty::int(32),
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    assert!(run(&mut func));
    assert_eq!(returned(&func), five);
}

#[test]
fn pass_manager_copies_extern_effects_onto_functions() {
    let sig = Signature::new(vec![], Ty::int(32));
    let mut module = crate::ir::Module::new();
    let target = module.push_extern("neutral", Signature::new(vec![], Ty::Void));
    module.externs[target].effect = crate::ir::ExternEffect::neutral();
    let (mut func, entry) = Function::new("caller", sig);
    let p = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
    let five = int_const(&mut func, 5);
    emit(
        &mut func,
        entry,
        InstKind::Store {
            addr: p,
            val: five,
            flags: MemFlags::plain(),
        },
        Ty::Void,
    );
    emit(&mut func, entry, InstKind::CallExtern { target, args: Default::default() }, Ty::Void);
    let loaded = emit(
        &mut func,
        entry,
        InstKind::Load {
            addr: p,
            flags: MemFlags::plain(),
        },
        Ty::int(32),
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });
    module.push_func(func);

    let mut pipeline = crate::pass::PassManager::new().add_func_pass(LoadStoreForward);
    pipeline.run(&mut module);
    assert_eq!(returned(&module.funcs[0]), five);
}

#[test]
fn default_extern_is_not_treated_as_a_fresh_allocator() {
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("external_alloc", sig);
    let size = func.push_const(Const::Int { bits: 64, value: 8 });
    let ptr_ty = Ty::ptr(Ty::int(32));
    let target = Extern::from_usize(0);
    let a = emit(
        &mut func,
        entry,
        InstKind::CallExtern {
            target,
            args: [size].into_iter().collect(),
        },
        ptr_ty.clone(),
    );
    let b = emit(
        &mut func,
        entry,
        InstKind::CallExtern {
            target,
            args: [size].into_iter().collect(),
        },
        ptr_ty,
    );
    let five = int_const(&mut func, 5);
    let seven = int_const(&mut func, 7);
    emit(
        &mut func,
        entry,
        InstKind::Store {
            addr: a,
            val: five,
            flags: MemFlags::plain(),
        },
        Ty::Void,
    );
    emit(
        &mut func,
        entry,
        InstKind::Store {
            addr: b,
            val: seven,
            flags: MemFlags::plain(),
        },
        Ty::Void,
    );
    let loaded = emit(
        &mut func,
        entry,
        InstKind::Load {
            addr: a,
            flags: MemFlags::plain(),
        },
        Ty::int(32),
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    assert!(!run(&mut func), "an unannotated extern may return an aliasing pointer");
    assert_eq!(returned(&func), loaded);
}

#[test]
fn volatile_and_call_barriers_keep_the_load() {
    let (mut func, entry, p) = ptr_func();
    let five = int_const(&mut func, 5);
    emit(
        &mut func,
        entry,
        InstKind::Store {
            addr: p,
            val: five,
            flags: MemFlags::plain(),
        },
        Ty::Void,
    );
    let volatile = emit(&mut func, entry, InstKind::Load { addr: p, flags: MemFlags::vol() }, Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(volatile) });
    assert!(!run(&mut func), "volatile load is never forwarded");

    let sig = Signature::new(vec![Ty::ptr(Ty::int(32))], Ty::int(32));
    let (mut func, entry) = Function::new("g", sig);
    let p = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
    let first = emit(
        &mut func,
        entry,
        InstKind::Load {
            addr: p,
            flags: MemFlags::plain(),
        },
        Ty::int(32),
    );
    let target = Extern::from_usize(0);
    emit(&mut func, entry, InstKind::CallExtern { target, args: Default::default() }, Ty::Void);
    let second = emit(
        &mut func,
        entry,
        InstKind::Load {
            addr: p,
            flags: MemFlags::plain(),
        },
        Ty::int(32),
    );
    let sum = emit(&mut func, entry, InstKind::Add(first, second), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
    assert!(!run(&mut func), "a call may write through p");
}
