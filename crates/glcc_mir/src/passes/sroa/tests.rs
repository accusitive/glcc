use super::*;
use crate::ir::{IcmpPred, MemFlags, Terminator};
use crate::ty::Signature;
use crate::verify::verify_func;

fn param(func: &mut Function, block: Block, ty: Ty, index: u32) -> Value {
    let value = func.values.push(ValueData {
        ty,
        def: Def::Param(block, index),
    });
    func.blocks[block].params.push(value);
    value
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

fn emit_unit(func: &mut Function, block: Block, kind: InstKind) -> Inst {
    let inst = func.insts.push(InstData {
        block,
        kind,
        result: None,
        loc: None,
    });
    func.blocks[block].insts.push(inst);
    inst
}

fn iconst(func: &mut Function, bits: u32, value: i128) -> Value {
    func.push_const(crate::ir::Const::Int { bits, value })
}

fn alloca(func: &mut Function, block: Block, ty: Ty) -> (Inst, Value) {
    let result = func.values.push(ValueData {
        ty: Ty::ptr(ty.clone()),
        def: Def::Inst(Inst::from_usize(func.insts.len())),
    });
    let inst = func.insts.push(InstData {
        block,
        kind: InstKind::Alloca { pointee: ty, count: 1 },
        result: Some(result),
        loc: None,
    });
    func.values[result].def = Def::Inst(inst);
    func.blocks[block].insts.push(inst);
    (inst, result)
}

fn gep(func: &mut Function, block: Block, base: Value, indices: Vec<GepIndex>, ty: Ty) -> Value {
    emit(
        func,
        block,
        InstKind::GetElementPtr {
            base,
            indices: indices.into_iter().collect(),
        },
        Ty::ptr(ty),
    )
}

fn load(func: &mut Function, block: Block, addr: Value, ty: Ty) -> Value {
    emit(func, block, InstKind::Load { addr, flags: MemFlags::plain() }, ty)
}

fn store(func: &mut Function, block: Block, addr: Value, val: Value) -> Inst {
    emit_unit(
        func,
        block,
        InstKind::Store {
            addr,
            val,
            flags: MemFlags::plain(),
        },
    )
}

fn run(func: &mut Function) -> bool {
    let changed = Sroa.run(func);
    let errors = verify_func(func);
    assert!(errors.is_empty(), "verifier: {errors:?}");
    changed
}

fn resident_kinds(func: &Function) -> impl Iterator<Item = &InstKind> {
    func.blocks.iter().flat_map(|block| block.insts.iter()).map(|inst| &func.insts[*inst].kind)
}

fn has_alloca(func: &Function) -> bool {
    resident_kinds(func).any(|kind| matches!(kind, InstKind::Alloca { .. }))
}

fn has_aggregate_alloca(func: &Function) -> bool {
    resident_kinds(func).any(|kind| matches!(kind, InstKind::Alloca { pointee, .. } if pointee.is_aggregate()))
}

fn has_memory_ops(func: &Function) -> bool {
    resident_kinds(func).any(|kind| matches!(kind, InstKind::Alloca { .. } | InstKind::Load { .. } | InstKind::Store { .. }))
}

fn point_ty() -> Ty {
    Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(32)],
        offsets: vec![0, 4],
        size: 8,
    }
}

#[test]
fn splits_struct_fields_and_mem2reg_finishes() {
    let point = point_ty();
    let sig = Signature::new(vec![Ty::int(32), Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("fields", sig);
    let a = param(&mut func, entry, Ty::int(32), 0);
    let b = param(&mut func, entry, Ty::int(32), 1);
    let (_, root) = alloca(&mut func, entry, point.clone());
    let x = gep(&mut func, entry, root, vec![GepIndex::Field(0)], Ty::int(32));
    let y = gep(&mut func, entry, root, vec![GepIndex::Field(1)], Ty::int(32));
    store(&mut func, entry, x, a);
    store(&mut func, entry, y, b);
    let xv = load(&mut func, entry, x, Ty::int(32));
    let yv = load(&mut func, entry, y, Ty::int(32));
    let sum = emit(&mut func, entry, InstKind::Add(xv, yv), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });

    assert!(run(&mut func));
    assert!(!has_aggregate_alloca(&func), "aggregate alloca survived");
    assert!(verify_func(&func).is_empty());
    assert!(!run(&mut func), "second run is a no-op");

    let mut mem2reg = super::super::Mem2Reg;
    assert!(mem2reg.run(&mut func));
    assert!(!has_memory_ops(&func), "memory traffic survived mem2reg");
    assert!(verify_func(&func).is_empty());
}

#[test]
fn splits_fixed_array_elements() {
    let array = Ty::Array {
        elem: Box::new(Ty::int(32)),
        len: 3,
    };
    let sig = Signature::new(vec![Ty::int(32), Ty::int(32), Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("array", sig);
    let a = param(&mut func, entry, Ty::int(32), 0);
    let b = param(&mut func, entry, Ty::int(32), 1);
    let c = param(&mut func, entry, Ty::int(32), 2);
    let (_, root) = alloca(&mut func, entry, array);
    for (index, value) in [(0, a), (1, b), (2, c)] {
        let addr = gep(&mut func, entry, root, vec![GepIndex::ConstIndex(index)], Ty::int(32));
        store(&mut func, entry, addr, value);
    }
    let first = gep(&mut func, entry, root, vec![GepIndex::ConstIndex(0)], Ty::int(32));
    let third = gep(&mut func, entry, root, vec![GepIndex::ConstIndex(2)], Ty::int(32));
    let x = load(&mut func, entry, first, Ty::int(32));
    let z = load(&mut func, entry, third, Ty::int(32));
    let sum = emit(&mut func, entry, InstKind::Add(x, z), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });

    assert!(run(&mut func));
    assert!(!has_aggregate_alloca(&func));
    let mut mem2reg = super::super::Mem2Reg;
    assert!(mem2reg.run(&mut func));
    assert!(!has_memory_ops(&func));
    assert!(verify_func(&func).is_empty());
}

#[test]
fn follows_integer_address_arithmetic() {
    let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("arithmetic", sig);
    let value = param(&mut func, entry, Ty::int(32), 0);
    let (_, root) = alloca(&mut func, entry, point_ty());
    let base = emit(&mut func, entry, InstKind::PtrToInt(root), Ty::int(64));
    let four = iconst(&mut func, 64, 4);
    let at_y = emit(&mut func, entry, InstKind::Add(base, four), Ty::int(64));
    let addr = emit(&mut func, entry, InstKind::IntToPtr(at_y), Ty::ptr(Ty::int(32)));
    store(&mut func, entry, addr, value);
    let loaded = load(&mut func, entry, addr, Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    assert!(run(&mut func));
    assert!(!has_aggregate_alloca(&func));
    let mut mem2reg = super::super::Mem2Reg;
    assert!(mem2reg.run(&mut func));
    assert!(!has_memory_ops(&func));
    assert!(verify_func(&func).is_empty());
}

#[test]
fn follows_bitcast_addresses() {
    let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("bitcast", sig);
    let value = param(&mut func, entry, Ty::int(32), 0);
    let (_, root) = alloca(&mut func, entry, point_ty());
    let field = gep(&mut func, entry, root, vec![GepIndex::Field(0)], Ty::int(32));
    let addr = emit(&mut func, entry, InstKind::Bitcast(field), Ty::ptr(Ty::Void));
    store(&mut func, entry, addr, value);
    let loaded = load(&mut func, entry, addr, Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    assert!(run(&mut func));
    assert!(!has_aggregate_alloca(&func));
    let mut mem2reg = super::super::Mem2Reg;
    assert!(mem2reg.run(&mut func));
    assert!(!has_memory_ops(&func));
    assert!(verify_func(&func).is_empty());
}

#[test]
fn leaves_dynamic_index_in_memory() {
    let array = Ty::Array {
        elem: Box::new(Ty::int(32)),
        len: 4,
    };
    let sig = Signature::new(vec![Ty::int(32), Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("dynamic", sig);
    let index = param(&mut func, entry, Ty::int(32), 0);
    let value = param(&mut func, entry, Ty::int(32), 1);
    let (_, root) = alloca(&mut func, entry, array);
    let addr = gep(&mut func, entry, root, vec![GepIndex::Index(index)], Ty::int(32));
    store(&mut func, entry, addr, value);
    let loaded = load(&mut func, entry, addr, Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    assert!(!run(&mut func));
    assert!(has_alloca(&func));
}

#[test]
fn leaves_escaped_address_in_memory() {
    let sig = Signature::new(vec![], Ty::Void);
    let (mut func, entry) = Function::new("escape", sig);
    let (_, root) = alloca(&mut func, entry, point_ty());
    let addr = gep(&mut func, entry, root, vec![GepIndex::Field(0)], Ty::int(32));
    emit_unit(
        &mut func,
        entry,
        InstKind::CallExtern {
            target: crate::ir::Extern::from_usize(0),
            args: [addr].into_iter().collect(),
        },
    );
    func.blocks[entry].term = Some(Terminator::Return { val: None });

    assert!(!run(&mut func));
    assert!(has_alloca(&func));
}

#[test]
fn leaves_overlapping_leaves_in_memory() {
    let sig = Signature::new(vec![], Ty::int(8));
    let (mut func, entry) = Function::new("overlap", sig);
    let (_, root) = alloca(&mut func, entry, point_ty());
    let wide = gep(&mut func, entry, root, vec![GepIndex::Field(0)], Ty::int(32));
    let byte = gep(&mut func, entry, root, vec![GepIndex::Field(0)], Ty::int(8));
    let one = iconst(&mut func, 32, 1);
    store(&mut func, entry, wide, one);
    let loaded = load(&mut func, entry, byte, Ty::int(8));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    assert!(!run(&mut func));
    assert!(has_alloca(&func));
}

#[test]
fn leaves_aggregate_load_in_memory() {
    let point = point_ty();
    let sig = Signature::new(vec![], point.clone());
    let (mut func, entry) = Function::new("whole", sig);
    let (_, root) = alloca(&mut func, entry, point.clone());
    let loaded = load(&mut func, entry, root, point);
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    assert!(!run(&mut func));
    assert!(has_alloca(&func));
}

#[test]
fn leaves_volatile_access_in_memory() {
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("volatile", sig);
    let value = iconst(&mut func, 32, 7);
    let (_, root) = alloca(&mut func, entry, point_ty());
    let addr = gep(&mut func, entry, root, vec![GepIndex::Field(0)], Ty::int(32));
    emit_unit(
        &mut func,
        entry,
        InstKind::Store {
            addr,
            val: value,
            flags: MemFlags::vol(),
        },
    );
    let loaded = load(&mut func, entry, addr, Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    assert!(!run(&mut func));
    assert!(has_alloca(&func));
}

#[test]
fn leaves_setjmp_functions_in_memory() {
    let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("jmp", sig);
    let value = param(&mut func, entry, Ty::int(32), 0);
    let (_, root) = alloca(&mut func, entry, point_ty());
    let addr = gep(&mut func, entry, root, vec![GepIndex::Field(0)], Ty::int(32));
    store(&mut func, entry, addr, value);
    let loaded = load(&mut func, entry, addr, Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });
    func.returns_twice = true;

    assert!(!run(&mut func));
    assert!(has_alloca(&func));
}

#[test]
fn leaves_stored_address_in_memory() {
    let holder = Ty::Struct {
        fields: vec![Ty::int(32), Ty::ptr(Ty::int(32))],
        offsets: vec![0, 8],
        size: 16,
    };
    let sig = Signature::new(vec![], Ty::Void);
    let (mut func, entry) = Function::new("self", sig);
    let (_, root) = alloca(&mut func, entry, holder);
    let x = gep(&mut func, entry, root, vec![GepIndex::Field(0)], Ty::int(32));
    let y = gep(&mut func, entry, root, vec![GepIndex::Field(1)], Ty::ptr(Ty::int(32)));
    emit_unit(
        &mut func,
        entry,
        InstKind::Store {
            addr: y,
            val: x,
            flags: MemFlags::plain(),
        },
    );
    func.blocks[entry].term = Some(Terminator::Return { val: None });

    assert!(!run(&mut func));
    assert!(has_alloca(&func));
}

#[test]
fn removes_unused_aggregate_alloca() {
    let sig = Signature::new(vec![], Ty::Void);
    let (mut func, entry) = Function::new("unused", sig);
    alloca(&mut func, entry, point_ty());
    func.blocks[entry].term = Some(Terminator::Return { val: None });

    assert!(run(&mut func));
    assert!(!has_alloca(&func));
    assert!(verify_func(&func).is_empty());
    assert!(!run(&mut func));
}

#[test]
fn scalarizes_nested_fields() {
    let inner = Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(8)],
        offsets: vec![0, 4],
        size: 8,
    };
    let outer = Ty::Struct {
        fields: vec![Ty::int(8), inner],
        offsets: vec![0, 8],
        size: 16,
    };
    let sig = Signature::new(vec![Ty::int(8)], Ty::int(8));
    let (mut func, entry) = Function::new("nested", sig);
    let value = param(&mut func, entry, Ty::int(8), 0);
    let (_, root) = alloca(&mut func, entry, outer);
    let addr = gep(&mut func, entry, root, vec![GepIndex::Field(1), GepIndex::Field(1)], Ty::int(8));
    store(&mut func, entry, addr, value);
    let loaded = load(&mut func, entry, addr, Ty::int(8));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    assert!(run(&mut func));
    assert!(!has_aggregate_alloca(&func));
    let mut mem2reg = super::super::Mem2Reg;
    assert!(mem2reg.run(&mut func));
    assert!(!has_memory_ops(&func));
    assert!(verify_func(&func).is_empty());
}

#[test]
fn keeps_icmp_on_address_in_memory() {
    let sig = Signature::new(vec![], Ty::Bool);
    let (mut func, entry) = Function::new("cmp", sig);
    let (_, root) = alloca(&mut func, entry, point_ty());
    let addr = gep(&mut func, entry, root, vec![GepIndex::Field(0)], Ty::int(32));
    let zero_int = iconst(&mut func, 64, 0);
    let zero = emit(&mut func, entry, InstKind::IntToPtr(zero_int), Ty::ptr(Ty::int(32)));
    let cmp = emit(
        &mut func,
        entry,
        InstKind::Icmp {
            pred: IcmpPred::Eq,
            lhs: addr,
            rhs: zero,
        },
        Ty::Bool,
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });

    assert!(!run(&mut func));
    assert!(has_alloca(&func));
}
