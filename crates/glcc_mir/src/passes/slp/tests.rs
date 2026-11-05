use super::*;
use smallvec::smallvec;
use crate::ir::{Const, Terminator};
use crate::ty::{FloatKind, GepIndex, Signature};
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

fn emit_unit(func: &mut Function, block: Block, kind: InstKind) {
    let inst = func.insts.push(InstData {
        block,
        kind,
        result: None,
        loc: None,
    });
    func.blocks[block].insts.push(inst);
}

fn iconst(func: &mut Function, bits: u32, value: i128) -> Value {
    func.push_const(Const::Int { bits, value })
}

fn run(func: &mut Function) -> bool {
    let changed = Slp.run(func);
    let errors = verify_func(func);
    assert!(errors.is_empty(), "verifier: {errors:?}");
    changed
}

fn stores(func: &Function) -> Vec<&InstKind> {
    func.blocks[func.entry]
        .insts
        .iter()
        .filter(|inst| matches!(&func.insts[**inst].kind, InstKind::Store { .. }))
        .map(|inst| &func.insts[*inst].kind)
        .collect()
}

fn has_vector_value(func: &Function, elem: Ty, lanes: u32) -> bool {
    let want = Ty::vector(elem, lanes);
    func.values.iter().any(|data| data.ty == want)
}

/// `dst[i] = src[i] + 1` for four i32 lanes, as straight-line code.
fn four_lane_add() -> (Function, Block, Value, Value) {
    let sig = Signature::new(vec![Ty::ptr(Ty::int(32)), Ty::ptr(Ty::int(32))], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let src = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
    let dst = param(&mut func, entry, Ty::ptr(Ty::int(32)), 1);
    for i in 0u64..4 {
        let from = emit(&mut func, entry, InstKind::GetElementPtr { base: src, indices: smallvec![GepIndex::ConstIndex(i)] }, Ty::ptr(Ty::int(32)));
        let to = emit(&mut func, entry, InstKind::GetElementPtr { base: dst, indices: smallvec![GepIndex::ConstIndex(i)] }, Ty::ptr(Ty::int(32)));
        let loaded = emit(&mut func, entry, InstKind::Load { addr: from, flags: MemFlags::plain() }, Ty::int(32));
        let one = iconst(&mut func, 32, 1);
        let bumped = emit(&mut func, entry, InstKind::Add(loaded, one), Ty::int(32));
        emit_unit(&mut func, entry, InstKind::Store { addr: to, val: bumped, flags: MemFlags::plain() });
    }
    func.blocks[entry].term = Some(Terminator::Return { val: None });
    (func, entry, src, dst)
}

#[test]
fn adjacent_add_chain_vectorizes() {
    let (mut func, ..) = four_lane_add();
    assert!(run(&mut func));
    assert!(has_vector_value(&func, Ty::int(32), 4), "v4xi32 value missing");
    let kept = stores(&func);
    assert_eq!(kept.len(), 1, "four scalar stores should consolidate: {kept:?}");
    let last = *func.blocks[func.entry].insts.last().unwrap();
    let InstKind::Store { val, .. } = &func.insts[last].kind else {
        panic!("last inst is not a store: {:?}", func.insts[last].kind)
    };
    assert!(matches!(func.values[*val].ty, Ty::Vector { .. }));
    assert!(!run(&mut func), "second run is a no-op");
}

#[test]
fn eight_lanes_form_two_groups() {
    let sig = Signature::new(vec![Ty::ptr(Ty::int(32)), Ty::ptr(Ty::int(32))], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let src = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
    let dst = param(&mut func, entry, Ty::ptr(Ty::int(32)), 1);
    for i in 0u64..8 {
        let from = emit(&mut func, entry, InstKind::GetElementPtr { base: src, indices: smallvec![GepIndex::ConstIndex(i)] }, Ty::ptr(Ty::int(32)));
        let to = emit(&mut func, entry, InstKind::GetElementPtr { base: dst, indices: smallvec![GepIndex::ConstIndex(i)] }, Ty::ptr(Ty::int(32)));
        let loaded = emit(&mut func, entry, InstKind::Load { addr: from, flags: MemFlags::plain() }, Ty::int(32));
        emit_unit(&mut func, entry, InstKind::Store { addr: to, val: loaded, flags: MemFlags::plain() });
    }
    func.blocks[entry].term = Some(Terminator::Return { val: None });
    assert!(run(&mut func));
    assert_eq!(stores(&func).len(), 2, "eight scalar stores should form two vector stores");
}

#[test]
fn one_splat_covers_repeated_stores() {
    let sig = Signature::new(vec![Ty::ptr(Ty::int(32))], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let dst = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
    let x = param(&mut func, entry, Ty::int(32), 1);
    for i in 0u64..4 {
        let to = emit(&mut func, entry, InstKind::GetElementPtr { base: dst, indices: smallvec![GepIndex::ConstIndex(i)] }, Ty::ptr(Ty::int(32)));
        emit_unit(&mut func, entry, InstKind::Store { addr: to, val: x, flags: MemFlags::plain() });
    }
    func.blocks[entry].term = Some(Terminator::Return { val: None });
    assert!(run(&mut func));
    assert!(has_vector_value(&func, Ty::int(32), 4));
    assert_eq!(stores(&func).len(), 1);
}

/// Two i64 stores of one value to consecutive offsets, in reverse block
/// order: the lower offset is stored last, so its address is defined
/// after the group's first seed. The packed store must splice in after
/// that address's def, not at the last seed's slot.
#[test]
fn reversed_seed_order_splices_after_address_def() {
    let sig = Signature::new(vec![Ty::ptr(Ty::int(64)), Ty::int(64)], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let dst = param(&mut func, entry, Ty::ptr(Ty::int(64)), 0);
    let x = param(&mut func, entry, Ty::int(64), 1);
    // Higher offset first, then address math and the lower-offset store.
    let hi = emit(&mut func, entry, InstKind::GetElementPtr { base: dst, indices: smallvec![GepIndex::ConstIndex(2)] }, Ty::ptr(Ty::int(64)));
    emit_unit(&mut func, entry, InstKind::Store { addr: hi, val: x, flags: MemFlags::plain() });
    let eight = iconst(&mut func, 64, 8);
    let base = emit(&mut func, entry, InstKind::PtrToInt(dst), Ty::int(64));
    let sum = emit(&mut func, entry, InstKind::Add(base, eight), Ty::int(64));
    let lo = emit(&mut func, entry, InstKind::IntToPtr(sum), Ty::ptr(Ty::int(64)));
    emit_unit(&mut func, entry, InstKind::Store { addr: lo, val: x, flags: MemFlags::plain() });
    func.blocks[entry].term = Some(Terminator::Return { val: None });

    assert!(run(&mut func));
    assert!(has_vector_value(&func, Ty::int(64), 2));
    assert_eq!(stores(&func).len(), 1);

    // Every operand of the surviving store must be defined earlier in
    // the block (or be a param/const).
    let last = *func.blocks[func.entry].insts.last().unwrap();
    let InstKind::Store { addr, .. } = &func.insts[last].kind else {
        panic!("last inst is not a store: {:?}", func.insts[last].kind)
    };
    let addr_pos = func.blocks[func.entry].insts.iter().position(|&inst| func.insts[inst].result == Some(*addr));
    let store_pos = func.blocks[func.entry].insts.iter().position(|&inst| inst == last);
    assert!(addr_pos.is_some_and(|p| p < store_pos.unwrap()), "store's address defined after the store");
}

#[test]
fn non_adjacent_stores_stay_scalar() {
    let sig = Signature::new(vec![Ty::ptr(Ty::int(32)), Ty::ptr(Ty::int(32))], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let src = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
    let dst = param(&mut func, entry, Ty::ptr(Ty::int(32)), 1);
    // Stride 8 bytes: no run of adjacent i32 lanes.
    for i in 0u64..4 {
        let from = emit(&mut func, entry, InstKind::GetElementPtr { base: src, indices: smallvec![GepIndex::ConstIndex(i * 2)] }, Ty::ptr(Ty::int(32)));
        let to = emit(&mut func, entry, InstKind::GetElementPtr { base: dst, indices: smallvec![GepIndex::ConstIndex(i * 2)] }, Ty::ptr(Ty::int(32)));
        let loaded = emit(&mut func, entry, InstKind::Load { addr: from, flags: MemFlags::plain() }, Ty::int(32));
        emit_unit(&mut func, entry, InstKind::Store { addr: to, val: loaded, flags: MemFlags::plain() });
    }
    func.blocks[entry].term = Some(Terminator::Return { val: None });
    assert!(!run(&mut func));
    assert_eq!(stores(&func).len(), 4);
}

#[test]
fn mixed_ops_stay_scalar() {
    let sig = Signature::new(vec![Ty::ptr(Ty::int(32)), Ty::ptr(Ty::int(32))], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let src = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
    let dst = param(&mut func, entry, Ty::ptr(Ty::int(32)), 1);
    for i in 0u64..4 {
        let from = emit(&mut func, entry, InstKind::GetElementPtr { base: src, indices: smallvec![GepIndex::ConstIndex(i)] }, Ty::ptr(Ty::int(32)));
        let to = emit(&mut func, entry, InstKind::GetElementPtr { base: dst, indices: smallvec![GepIndex::ConstIndex(i)] }, Ty::ptr(Ty::int(32)));
        let loaded = emit(&mut func, entry, InstKind::Load { addr: from, flags: MemFlags::plain() }, Ty::int(32));
        let one = iconst(&mut func, 32, 1);
        let mixed = if i % 2 == 0 {
            emit(&mut func, entry, InstKind::Add(loaded, one), Ty::int(32))
        } else {
            emit(&mut func, entry, InstKind::Sub(loaded, one), Ty::int(32))
        };
        emit_unit(&mut func, entry, InstKind::Store { addr: to, val: mixed, flags: MemFlags::plain() });
    }
    func.blocks[entry].term = Some(Terminator::Return { val: None });
    assert!(!run(&mut func));
}

#[test]
fn interleaved_read_blocks_the_group() {
    // b[0], b[2], then a read of b[2], then b[1], b[3]: the read sees the
    // new b[2] before consolidation moves that store later.
    let sig = Signature::new(vec![Ty::ptr(Ty::int(32)), Ty::ptr(Ty::int(32))], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let src = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
    let dst = param(&mut func, entry, Ty::ptr(Ty::int(32)), 1);
    let mut addrs = Vec::new();
    for i in 0u64..4 {
        let to = emit(&mut func, entry, InstKind::GetElementPtr { base: dst, indices: smallvec![GepIndex::ConstIndex(i)] }, Ty::ptr(Ty::int(32)));
        addrs.push(to);
    }
    let x = param(&mut func, entry, Ty::int(32), 2);
    // Store order: lanes 0, 2, read of lane 2, lanes 1, 3.
    emit_unit(&mut func, entry, InstKind::Store { addr: addrs[0], val: x, flags: MemFlags::plain() });
    emit_unit(&mut func, entry, InstKind::Store { addr: addrs[2], val: x, flags: MemFlags::plain() });
    let peeked = emit(&mut func, entry, InstKind::Load { addr: addrs[2], flags: MemFlags::plain() }, Ty::int(32));
    emit_unit(&mut func, entry, InstKind::Store { addr: addrs[1], val: x, flags: MemFlags::plain() });
    emit_unit(&mut func, entry, InstKind::Store { addr: addrs[3], val: x, flags: MemFlags::plain() });
    func.blocks[entry].term = Some(Terminator::Return { val: Some(peeked) });
    assert!(!run(&mut func));
    let _ = src;
}

#[test]
fn call_between_seeds_blocks_global_group() {
    // Stores to a global with a call in between: the callee may read the
    // global, so consolidation would move stores across it.
    let sig = Signature::new(vec![], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    // A dangling global id is fine: the pass only reads the address's
    // object identity, never the module.
    let global = crate::ir::Global::from_usize(0);
    let base = emit(&mut func, entry, InstKind::GlobalAddr { global }, Ty::ptr(Ty::int(32)));
    let callee = crate::ir::Func::from_usize(0);
    let mut addrs = Vec::new();
    for i in 0u64..4 {
        let to = emit(&mut func, entry, InstKind::GetElementPtr { base, indices: smallvec![GepIndex::ConstIndex(i)] }, Ty::ptr(Ty::int(32)));
        addrs.push(to);
    }
    let x = iconst(&mut func, 32, 7);
    emit_unit(&mut func, entry, InstKind::Store { addr: addrs[0], val: x, flags: MemFlags::plain() });
    emit_unit(&mut func, entry, InstKind::Store { addr: addrs[1], val: x, flags: MemFlags::plain() });
    emit_unit(&mut func, entry, InstKind::Call { callee, args: smallvec![] });
    emit_unit(&mut func, entry, InstKind::Store { addr: addrs[2], val: x, flags: MemFlags::plain() });
    emit_unit(&mut func, entry, InstKind::Store { addr: addrs[3], val: x, flags: MemFlags::plain() });
    func.blocks[entry].term = Some(Terminator::Return { val: None });
    assert!(!run(&mut func));
}

#[test]
fn volatile_store_breaks_the_run() {
    let sig = Signature::new(vec![Ty::ptr(Ty::int(32))], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let dst = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
    let x = param(&mut func, entry, Ty::int(32), 1);
    for i in 0u64..4 {
        let to = emit(&mut func, entry, InstKind::GetElementPtr { base: dst, indices: smallvec![GepIndex::ConstIndex(i)] }, Ty::ptr(Ty::int(32)));
        let mut flags = MemFlags::plain();
        flags.volatile = i == 1;
        emit_unit(&mut func, entry, InstKind::Store { addr: to, val: x, flags });
    }
    func.blocks[entry].term = Some(Terminator::Return { val: None });
    assert!(!run(&mut func));
}

#[test]
fn float_lanes_vectorize() {
    let sig = Signature::new(vec![Ty::ptr(Ty::Float(FloatKind::F32)), Ty::ptr(Ty::Float(FloatKind::F32))], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let src = param(&mut func, entry, Ty::ptr(Ty::Float(FloatKind::F32)), 0);
    let dst = param(&mut func, entry, Ty::ptr(Ty::Float(FloatKind::F32)), 1);
    for i in 0u64..4 {
        let from = emit(&mut func, entry, InstKind::GetElementPtr { base: src, indices: smallvec![GepIndex::ConstIndex(i)] }, Ty::ptr(Ty::Float(FloatKind::F32)));
        let to = emit(&mut func, entry, InstKind::GetElementPtr { base: dst, indices: smallvec![GepIndex::ConstIndex(i)] }, Ty::ptr(Ty::Float(FloatKind::F32)));
        let loaded = emit(&mut func, entry, InstKind::Load { addr: from, flags: MemFlags::plain() }, Ty::Float(FloatKind::F32));
        let half = func.push_const(Const::Float { kind: FloatKind::F32, bits: (0.5f32).to_bits() as u128 });
        let scaled = emit(&mut func, entry, InstKind::FMul(loaded, half), Ty::Float(FloatKind::F32));
        emit_unit(&mut func, entry, InstKind::Store { addr: to, val: scaled, flags: MemFlags::plain() });
    }
    func.blocks[entry].term = Some(Terminator::Return { val: None });
    assert!(run(&mut func));
    assert!(has_vector_value(&func, Ty::Float(FloatKind::F32), 4));
}
