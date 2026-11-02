use super::*;
use crate::builder::FunctionBuilder;
use crate::ir::{AsmBlock, AsmOperand, Module};
use crate::ty::{Signature, Ty};

fn const_func() -> Function {
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("consts", sig);
    let mut b = FunctionBuilder::new(&mut func, entry);
    let two = b.iconst(32, 2);
    let three = b.iconst(32, 3);
    let four = b.iconst(32, 4);
    let mul = b.mul(Ty::int(32), three, four);
    let sum = b.add(Ty::int(32), two, mul);
    b.ret(Some(sum));
    func
}

#[test]
fn folds_constant_arithmetic() {
    let mut func = const_func();
    let mut fold = ConstFold;
    fold.run(&mut func);
    let term = func.blocks[func.entry].term.clone();
    assert!(matches!(term, Some(Terminator::Return { .. })));
    let Terminator::Return { val } = term.unwrap() else { unreachable!() };
    let c = func.const_value(val.unwrap()).unwrap();
    assert_eq!(c, &Const::Int { bits: 32, value: 14 });
}

#[test]
fn int_const_folds_address_arithmetic() {
    let (mut func, entry) = Function::new("address", Signature::new(vec![], Ty::int(64)));
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let widened = func.values.push(ValueData {
        ty: Ty::int(64),
        def: Def::Inst(Inst::from_usize(0)),
    });
    let sext = func.insts.push(InstData {
        block: entry,
        kind: InstKind::Sext(one),
        result: Some(widened),
        loc: None,
    });
    func.values[widened].def = Def::Inst(sext);
    let four = func.push_const(Const::Int { bits: 64, value: 4 });
    let scaled = func.values.push(ValueData {
        ty: Ty::int(64),
        def: Def::Inst(Inst::from_usize(0)),
    });
    let mul = func.insts.push(InstData {
        block: entry,
        kind: InstKind::Mul(widened, four),
        result: Some(scaled),
        loc: None,
    });
    func.values[scaled].def = Def::Inst(mul);
    assert_eq!(int_const(&func, scaled), Some((64, 4)));
}

#[test]
fn folds_constant_branch() {
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("branch", sig);
    let then_b = func.push_block(vec![]);
    let else_b = func.push_block(vec![]);
    let one = func.push_const(Const::Int { bits: 1, value: 1 });
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond: one,
        then_block: then_b,
        else_block: else_b,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    let mut fold = ConstFold;
    assert!(fold.run(&mut func));
    assert!(matches!(func.blocks[entry].term, Some(Terminator::Br { .. })));
}

#[test]
fn fold_replacement_keeps_result_type() {
    let ptr_ty = Ty::ptr(Ty::int(8));
    let sig = Signature::new(vec![], ptr_ty.clone());
    let (mut func, entry) = Function::new("sel", sig);
    let one = func.push_const(Const::Int { bits: 1, value: 1 });
    let undef = func.values.push(crate::ir::ValueData {
        ty: ptr_ty.clone(),
        def: Def::Const(Const::Undef),
    });
    let null = func.values.push(crate::ir::ValueData {
        ty: ptr_ty.clone(),
        def: Def::Const(Const::Null),
    });
    let result = func.values.push(crate::ir::ValueData {
        ty: ptr_ty.clone(),
        def: Def::Inst(crate::ir::Inst::from_usize(0)),
    });
    let inst = func.insts.push(crate::ir::InstData {
        block: entry,
        kind: InstKind::Select {
            cond: one,
            then_val: undef,
            else_val: null,
        },
        result: Some(result),
        loc: None,
    });
    func.values[result].def = Def::Inst(inst);
    func.blocks[entry].insts.push(inst);
    func.blocks[entry].term = Some(Terminator::Return { val: Some(result) });
    let mut fold = ConstFold;
    assert!(fold.run(&mut func));
    let Some(Terminator::Return { val: Some(val) }) = func.blocks[entry].term else {
        panic!("entry still returns");
    };
    assert_eq!(func.const_value(val), Some(&Const::Undef));
    assert_eq!(func.values[val].ty, ptr_ty);
}

fn aggregate_copy_func(volatile: bool) -> (Function, crate::ir::Inst, crate::ir::Inst) {
    let ty = Ty::Struct {
        fields: vec![Ty::int(64), Ty::int(64)],
        offsets: vec![0, 8],
        size: 16,
    };
    let sig = Signature::new(vec![Ty::ptr(ty.clone()), Ty::ptr(ty.clone())], Ty::Void);
    let (mut func, entry) = Function::new("copy", sig);
    let dst = func.values.push(crate::ir::ValueData {
        ty: Ty::ptr(ty.clone()),
        def: Def::Param(entry, 0),
    });
    let src = func.values.push(crate::ir::ValueData {
        ty: Ty::ptr(ty.clone()),
        def: Def::Param(entry, 1),
    });
    func.blocks[entry].params.extend([dst, src]);
    let flags = if volatile { crate::ir::MemFlags::vol() } else { crate::ir::MemFlags::plain() };
    let result = func.values.push(crate::ir::ValueData {
        ty: ty.clone(),
        def: Def::Inst(Inst::from_usize(0)),
    });
    let load = func.insts.push(crate::ir::InstData {
        block: entry,
        kind: InstKind::Load { addr: src, flags },
        result: Some(result),
        loc: None,
    });
    func.values[result].def = Def::Inst(load);
    func.blocks[entry].insts.push(load);
    let store = func.insts.push(crate::ir::InstData {
        block: entry,
        kind: InstKind::Store { addr: dst, val: result, flags },
        result: None,
        loc: None,
    });
    func.blocks[entry].insts.push(store);
    func.blocks[entry].term = Some(Terminator::Return { val: None });
    (func, load, store)
}

#[test]
fn forwards_adjacent_aggregate_load_store() {
    let (mut func, load, store) = aggregate_copy_func(false);
    let mut pass = ForwardAggregateCopy;
    assert!(pass.run(&mut func));
    assert!(matches!(func.insts[load].kind, InstKind::Memcpy { len: 16, .. }));
    assert_eq!(func.insts[load].result, None);
    assert!(!func.blocks[func.entry].insts.contains(&store));
    assert!(crate::verify::verify_func(&func).is_empty());
    assert!(!pass.run(&mut func), "second run is a no-op");
}

#[test]
fn expands_local_scalar_memcpy_for_sroa() {
    let sig = Signature::new(vec![Ty::ptr(Ty::int(8))], Ty::int(64));
    let (mut func, entry) = Function::new("scalar-copy", sig);
    let src = entry_param(&mut func, entry, Ty::ptr(Ty::int(8)), 0);
    let slot = stack_slot(&mut func, entry, Ty::int(64));
    let void_ptr = push_value(&mut func, entry, InstKind::Bitcast(slot), Ty::ptr(Ty::Void));
    let copy = push_unit(&mut func, entry, InstKind::Memcpy { dst: void_ptr, src, len: 8 });
    let loaded = push_scalar_load(&mut func, entry, slot, Ty::int(64));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    let mut forward = ForwardAggregateCopy;
    assert!(forward.run(&mut func));
    assert!(!func.blocks[entry].insts.contains(&copy));
    assert!(!resident_kinds(&func).iter().any(|kind| matches!(kind, InstKind::Memcpy { .. })));

    let mut dce = DeadCodeElim;
    dce.run(&mut func);
    let mut sroa = Sroa;
    let mut mem2reg = Mem2Reg;
    sroa.run(&mut func);
    mem2reg.run(&mut func);
    assert!(!resident_kinds(&func).iter().any(|kind| matches!(kind, InstKind::Alloca { .. } | InstKind::Store { .. })));
    assert!(crate::verify::verify_func(&func).is_empty());
}

#[test]
fn expands_local_aggregate_copies_for_sroa() {
    let ty = Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(32)],
        offsets: vec![0, 4],
        size: 8,
    };
    let (mut func, entry) = Function::new("copy", Signature::new(vec![], Ty::int(32)));
    let dst = stack_slot(&mut func, entry, ty.clone());
    let src = stack_slot(&mut func, entry, ty.clone());
    let copy = push_unit(&mut func, entry, InstKind::Memcpy { dst, src, len: 8 });
    let field = push_value(
        &mut func,
        entry,
        InstKind::GetElementPtr {
            base: dst,
            indices: [crate::ty::GepIndex::Field(0)].into_iter().collect(),
        },
        Ty::ptr(Ty::int(32)),
    );
    let loaded = push_scalar_load(&mut func, entry, field, Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    let mut forward = ForwardAggregateCopy;
    assert!(forward.run(&mut func));
    assert!(!func.blocks[entry].insts.contains(&copy));
    assert!(!func.blocks[entry].insts.iter().any(|inst| matches!(func.insts[*inst].kind, InstKind::Memcpy { .. })));

    let mut sroa = Sroa;
    let mut mem2reg = Mem2Reg;
    assert!(sroa.run(&mut func));
    assert!(mem2reg.run(&mut func));
    assert!(!has_memory_ops(&func));
    assert!(crate::verify::verify_func(&func).is_empty());
}

#[test]
fn expands_padded_aggregate_copies_for_sroa() {
    let ty = Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(32)],
        offsets: vec![0, 8],
        size: 16,
    };
    let (mut func, entry) = Function::new("padded", Signature::new(vec![], Ty::int(32)));
    let dst = stack_slot(&mut func, entry, ty.clone());
    let src = stack_slot(&mut func, entry, ty);
    let copy = push_unit(&mut func, entry, InstKind::Memcpy { dst, src, len: 16 });
    let field = push_value(
        &mut func,
        entry,
        InstKind::GetElementPtr {
            base: dst,
            indices: [crate::ty::GepIndex::Field(1)].into_iter().collect(),
        },
        Ty::ptr(Ty::int(32)),
    );
    let loaded = push_scalar_load(&mut func, entry, field, Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    let mut forward = ForwardAggregateCopy;
    assert!(forward.run(&mut func));
    assert!(!func.blocks[entry].insts.contains(&copy));

    let mut sroa = Sroa;
    let mut mem2reg = Mem2Reg;
    assert!(sroa.run(&mut func));
    assert!(mem2reg.run(&mut func));
    assert!(!has_memory_ops(&func));
    assert!(crate::verify::verify_func(&func).is_empty());
}

#[test]
fn expands_aggregate_parameter_copy_and_return() {
    let ty = Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(32)],
        offsets: vec![0, 4],
        size: 8,
    };
    let (mut func, entry) = Function::new("copy_param", Signature::new(vec![ty.clone()], ty.clone()));
    let input = func.values.push(ValueData {
        ty: ty.clone(),
        def: Def::Param(entry, 0),
    });
    func.blocks[entry].params.push(input);
    let slot = stack_slot(&mut func, entry, ty.clone());
    push_unit(
        &mut func,
        entry,
        InstKind::Store {
            addr: slot,
            val: input,
            flags: MemFlags::plain(),
        },
    );
    let loaded = push_value(
        &mut func,
        entry,
        InstKind::Load {
            addr: slot,
            flags: MemFlags::plain(),
        },
        ty,
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    let mut forward = ForwardAggregateCopy;
    assert!(forward.run(&mut func));
    assert!(!resident_kinds(&func).iter().any(|kind| matches!(kind, InstKind::Memcpy { .. })));

    let mut sroa = Sroa;
    let mut mem2reg = Mem2Reg;
    assert!(sroa.run(&mut func));
    assert!(mem2reg.run(&mut func));
    assert!(!has_memory_ops(&func));
    assert!(crate::verify::verify_func(&func).is_empty());
}

#[test]
fn expands_aggregate_load_consumed_by_field_extracts() {
    let ty = Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(32)],
        offsets: vec![0, 4],
        size: 8,
    };
    let (mut func, entry) = Function::new("load_fields", Signature::new(vec![], Ty::int(32)));
    let slot = stack_slot(&mut func, entry, ty.clone());
    let aggregate = push_value(
        &mut func,
        entry,
        InstKind::Load {
            addr: slot,
            flags: MemFlags::plain(),
        },
        ty,
    );
    let field = push_value(&mut func, entry, InstKind::ExtractValue { agg: aggregate, index: 1 }, Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(field) });

    let mut forward = ForwardAggregateCopy;
    assert!(forward.run(&mut func));
    assert!(!resident_kinds(&func).iter().any(|kind| matches!(kind, InstKind::ExtractValue { .. })));

    let mut sroa = Sroa;
    let mut mem2reg = Mem2Reg;
    assert!(sroa.run(&mut func));
    assert!(mem2reg.run(&mut func));
    assert!(!has_memory_ops(&func));
    assert!(crate::verify::verify_func(&func).is_empty());
}

#[test]
fn expands_local_zero_memsets_for_sroa() {
    let ty = Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(32)],
        offsets: vec![0, 4],
        size: 8,
    };
    let (mut func, entry) = Function::new("zero", Signature::new(vec![], Ty::int(32)));
    let slot = stack_slot(&mut func, entry, ty.clone());
    let memset = push_unit(&mut func, entry, InstKind::Memset { dst: slot, byte: 0, len: 8 });
    let field = push_value(
        &mut func,
        entry,
        InstKind::GetElementPtr {
            base: slot,
            indices: [crate::ty::GepIndex::Field(1)].into_iter().collect(),
        },
        Ty::ptr(Ty::int(32)),
    );
    let loaded = push_scalar_load(&mut func, entry, field, Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    let mut forward = ForwardAggregateCopy;
    assert!(forward.run(&mut func));
    assert!(!func.blocks[entry].insts.contains(&memset));
    assert!(!resident_kinds(&func).iter().any(|kind| matches!(kind, InstKind::Memset { .. })));

    let mut sroa = Sroa;
    let mut mem2reg = Mem2Reg;
    assert!(sroa.run(&mut func));
    assert!(mem2reg.run(&mut func));
    assert!(!has_memory_ops(&func));
    let Some(Terminator::Return { val: Some(value) }) = func.blocks[entry].term else {
        panic!("entry still returns the field");
    };
    assert_eq!(func.const_value(value), Some(&Const::Int { bits: 32, value: 0 }));
    assert!(crate::verify::verify_func(&func).is_empty());
}

#[test]
fn keeps_nonzero_and_escaping_memsets() {
    let ty = Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(32)],
        offsets: vec![0, 4],
        size: 8,
    };
    let (mut func, entry) = Function::new("nonzero", Signature::new(vec![], Ty::Void));
    let slot = stack_slot(&mut func, entry, ty.clone());
    let memset = push_unit(&mut func, entry, InstKind::Memset { dst: slot, byte: 0xFF, len: 8 });
    func.blocks[entry].term = Some(Terminator::Return { val: None });
    let mut forward = ForwardAggregateCopy;
    assert!(!forward.run(&mut func));
    assert!(func.blocks[entry].insts.contains(&memset));

    let (mut func, entry) = Function::new("escape", Signature::new(vec![], Ty::ptr(ty.clone())));
    let slot = stack_slot(&mut func, entry, ty);
    let memset = push_unit(&mut func, entry, InstKind::Memset { dst: slot, byte: 0, len: 8 });
    func.blocks[entry].term = Some(Terminator::Return { val: Some(slot) });
    assert!(!forward.run(&mut func));
    assert!(func.blocks[entry].insts.contains(&memset));
}

#[test]
fn keeps_aggregate_copies_when_a_slot_escapes() {
    let ty = Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(32)],
        offsets: vec![0, 4],
        size: 8,
    };
    let (mut func, entry) = Function::new("escape", Signature::new(vec![], Ty::ptr(ty.clone())));
    let dst = stack_slot(&mut func, entry, ty.clone());
    let src = stack_slot(&mut func, entry, ty);
    let copy = push_unit(&mut func, entry, InstKind::Memcpy { dst, src, len: 8 });
    func.blocks[entry].term = Some(Terminator::Return { val: Some(dst) });

    let mut forward = ForwardAggregateCopy;
    assert!(!forward.run(&mut func));
    assert!(func.blocks[entry].insts.contains(&copy));
    assert!(crate::verify::verify_func(&func).is_empty());
}

#[test]
fn keeps_volatile_aggregate_copies() {
    let (mut func, load, _) = aggregate_copy_func(true);
    let mut pass = ForwardAggregateCopy;
    assert!(!pass.run(&mut func));
    assert!(matches!(func.insts[load].kind, InstKind::Load { .. }));
}

fn stack_slot(func: &mut Function, block: Block, ty: Ty) -> Value {
    let mut builder = FunctionBuilder::new(func, block);
    builder.alloca(ty, 1)
}

fn push_unit(func: &mut Function, block: Block, kind: InstKind) -> Inst {
    let inst = func.insts.push(crate::ir::InstData {
        block,
        kind,
        result: None,
        loc: None,
    });
    func.blocks[block].insts.push(inst);
    inst
}

fn push_scalar_load(func: &mut Function, block: Block, addr: Value, ty: Ty) -> Value {
    let mut builder = FunctionBuilder::new(func, block);
    builder.load(addr, ty)
}

fn entry_param(func: &mut Function, block: Block, ty: Ty, index: u32) -> Value {
    let value = func.values.push(crate::ir::ValueData {
        ty,
        def: Def::Param(block, index),
    });
    func.blocks[block].params.push(value);
    value
}

fn push_value(func: &mut Function, block: Block, kind: InstKind, ty: Ty) -> Value {
    let result = func.values.push(crate::ir::ValueData {
        ty,
        def: Def::Inst(Inst::from_usize(0)),
    });
    let inst = func.insts.push(crate::ir::InstData {
        block,
        kind,
        result: Some(result),
        loc: None,
    });
    func.values[result].def = Def::Inst(inst);
    func.blocks[block].insts.push(inst);
    result
}

#[test]
fn removes_store_overwritten_in_the_same_block() {
    let (mut func, entry) = Function::new("dse", Signature::new(vec![], Ty::Void));
    let slot = stack_slot(&mut func, entry, Ty::int(32));
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let two = func.push_const(Const::Int { bits: 32, value: 2 });
    let first = push_unit(
        &mut func,
        entry,
        InstKind::Store {
            addr: slot,
            val: one,
            flags: crate::ir::MemFlags::plain(),
        },
    );
    let second = push_unit(
        &mut func,
        entry,
        InstKind::Store {
            addr: slot,
            val: two,
            flags: crate::ir::MemFlags::plain(),
        },
    );
    func.blocks[entry].term = Some(Terminator::Return { val: None });

    let mut pass = DeadStoreElim;
    assert!(pass.run(&mut func));
    assert!(!func.blocks[entry].insts.contains(&first));
    assert!(func.blocks[entry].insts.contains(&second));
    assert!(crate::verify::verify_func(&func).is_empty());
}

#[test]
fn removes_store_overwritten_across_a_neutral_extern() {
    let (mut func, entry) = Function::new("dse", Signature::new(vec![], Ty::Void));
    func.set_extern_effects(&[crate::ir::ExternEffect::neutral()]);
    let slot = stack_slot(&mut func, entry, Ty::int(32));
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let two = func.push_const(Const::Int { bits: 32, value: 2 });
    let first = push_unit(
        &mut func,
        entry,
        InstKind::Store {
            addr: slot,
            val: one,
            flags: crate::ir::MemFlags::plain(),
        },
    );
    push_unit(
        &mut func,
        entry,
        InstKind::CallExtern {
            target: crate::ir::Extern::from_usize(0),
            args: Default::default(),
        },
    );
    let second = push_unit(
        &mut func,
        entry,
        InstKind::Store {
            addr: slot,
            val: two,
            flags: crate::ir::MemFlags::plain(),
        },
    );
    func.blocks[entry].term = Some(Terminator::Return { val: None });

    let mut pass = DeadStoreElim;
    assert!(pass.run(&mut func));
    assert!(!func.blocks[entry].insts.contains(&first));
    assert!(func.blocks[entry].insts.contains(&second));
}

#[test]
fn removes_store_overwritten_before_a_later_load() {
    let (mut func, entry) = Function::new("dse", Signature::new(vec![], Ty::Void));
    let slot = stack_slot(&mut func, entry, Ty::int(32));
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let two = func.push_const(Const::Int { bits: 32, value: 2 });
    let first = push_unit(
        &mut func,
        entry,
        InstKind::Store {
            addr: slot,
            val: one,
            flags: crate::ir::MemFlags::plain(),
        },
    );
    let second = push_unit(
        &mut func,
        entry,
        InstKind::Store {
            addr: slot,
            val: two,
            flags: crate::ir::MemFlags::plain(),
        },
    );
    let loaded = push_scalar_load(&mut func, entry, slot, Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    let mut pass = DeadStoreElim;
    assert!(pass.run(&mut func));
    assert!(!func.blocks[entry].insts.contains(&first));
    assert!(func.blocks[entry].insts.contains(&second));
}

#[test]
fn keeps_store_read_before_the_overwrite() {
    let (mut func, entry) = Function::new("dse", Signature::new(vec![], Ty::Void));
    let slot = stack_slot(&mut func, entry, Ty::int(32));
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let two = func.push_const(Const::Int { bits: 32, value: 2 });
    let first = push_unit(
        &mut func,
        entry,
        InstKind::Store {
            addr: slot,
            val: one,
            flags: crate::ir::MemFlags::plain(),
        },
    );
    let loaded = push_scalar_load(&mut func, entry, slot, Ty::int(32));
    let second = push_unit(
        &mut func,
        entry,
        InstKind::Store {
            addr: slot,
            val: two,
            flags: crate::ir::MemFlags::plain(),
        },
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    let mut pass = DeadStoreElim;
    assert!(!pass.run(&mut func));
    assert!(func.blocks[entry].insts.contains(&first));
    assert!(func.blocks[entry].insts.contains(&second));
}

#[test]
fn removes_overwritten_copy_from_a_different_object() {
    let ty = Ty::Struct {
        fields: vec![Ty::int(64), Ty::int(64)],
        offsets: vec![0, 8],
        size: 16,
    };
    let (mut func, entry) = Function::new("dse", Signature::new(vec![], Ty::Void));
    let dst = stack_slot(&mut func, entry, ty.clone());
    let first_src = stack_slot(&mut func, entry, ty.clone());
    let second_src = stack_slot(&mut func, entry, ty.clone());
    let first = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: first_src, len: 16 });
    let second = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: second_src, len: 16 });
    func.blocks[entry].term = Some(Terminator::Return { val: None });

    let mut pass = DeadStoreElim;
    assert!(pass.run(&mut func));
    assert!(!func.blocks[entry].insts.contains(&first));
    assert!(func.blocks[entry].insts.contains(&second));
    assert!(crate::verify::verify_func(&func).is_empty());
}

#[test]
fn removes_copy_from_unknown_source_for_local_destination() {
    let ty = Ty::Struct {
        fields: vec![Ty::int(64), Ty::int(64)],
        offsets: vec![0, 8],
        size: 16,
    };
    let sig = Signature::new(vec![Ty::ptr(ty.clone()), Ty::ptr(ty.clone())], Ty::Void);
    let (mut func, entry) = Function::new("dse", sig);
    let first_src = entry_param(&mut func, entry, Ty::ptr(ty.clone()), 0);
    let second_src = entry_param(&mut func, entry, Ty::ptr(ty.clone()), 1);
    let dst = stack_slot(&mut func, entry, ty);
    let first = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: first_src, len: 16 });
    let second = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: second_src, len: 16 });
    func.blocks[entry].term = Some(Terminator::Return { val: None });

    let mut pass = DeadStoreElim;
    assert!(pass.run(&mut func));
    assert!(!func.blocks[entry].insts.contains(&first));
    assert!(func.blocks[entry].insts.contains(&second));
}

#[test]
fn removes_copy_overwritten_before_field_load() {
    let ty = Ty::Struct {
        fields: vec![Ty::Array {
            elem: Box::new(Ty::int(64)),
            len: 4,
        }],
        offsets: vec![0],
        size: 32,
    };
    let sig = Signature::new(vec![Ty::ptr(ty.clone()), Ty::ptr(ty.clone())], Ty::int(64));
    let (mut func, entry) = Function::new("dse", sig);
    let first_src = entry_param(&mut func, entry, Ty::ptr(ty.clone()), 0);
    let second_src = entry_param(&mut func, entry, Ty::ptr(ty.clone()), 1);
    let dst = stack_slot(&mut func, entry, ty);
    let first = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: first_src, len: 32 });
    let second = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: second_src, len: 32 });
    let field = push_value(
        &mut func,
        entry,
        InstKind::GetElementPtr {
            base: dst,
            indices: vec![crate::ty::GepIndex::Field(0), crate::ty::GepIndex::ConstIndex(0)].into_iter().collect(),
        },
        Ty::ptr(Ty::int(64)),
    );
    let as_int = push_value(&mut func, entry, InstKind::PtrToInt(field), Ty::int(64));
    let as_ptr = push_value(&mut func, entry, InstKind::IntToPtr(as_int), Ty::ptr(Ty::int(64)));
    let loaded = push_scalar_load(&mut func, entry, as_ptr, Ty::int(64));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    let mut pass = DeadStoreElim;
    assert!(pass.run(&mut func));
    assert!(!func.blocks[entry].insts.contains(&first));
    assert!(func.blocks[entry].insts.contains(&second));
}

#[test]
fn keeps_copy_when_address_is_stored_as_integer() {
    let ty = Ty::Struct {
        fields: vec![Ty::int(64), Ty::int(64)],
        offsets: vec![0, 8],
        size: 16,
    };
    let sig = Signature::new(vec![Ty::ptr(Ty::int(64)), Ty::ptr(ty.clone()), Ty::ptr(ty.clone())], Ty::Void);
    let (mut func, entry) = Function::new("dse", sig);
    let sink = entry_param(&mut func, entry, Ty::ptr(Ty::int(64)), 0);
    let first_src = entry_param(&mut func, entry, Ty::ptr(ty.clone()), 1);
    let second_src = entry_param(&mut func, entry, Ty::ptr(ty.clone()), 2);
    let dst = stack_slot(&mut func, entry, ty);
    let address = push_value(&mut func, entry, InstKind::PtrToInt(dst), Ty::int(64));
    push_unit(
        &mut func,
        entry,
        InstKind::Store {
            addr: sink,
            val: address,
            flags: crate::ir::MemFlags::plain(),
        },
    );
    let first = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: first_src, len: 16 });
    let second = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: second_src, len: 16 });
    func.blocks[entry].term = Some(Terminator::Return { val: None });

    let mut pass = DeadStoreElim;
    assert!(!pass.run(&mut func));
    assert!(func.blocks[entry].insts.contains(&first));
    assert!(func.blocks[entry].insts.contains(&second));
}

#[test]
fn keeps_copy_when_local_address_escapes() {
    let ty = Ty::Struct {
        fields: vec![Ty::int(64), Ty::int(64)],
        offsets: vec![0, 8],
        size: 16,
    };
    let sig = Signature::new(vec![Ty::ptr(Ty::ptr(ty.clone())), Ty::ptr(ty.clone()), Ty::ptr(ty.clone())], Ty::Void);
    let (mut func, entry) = Function::new("dse", sig);
    let sink = entry_param(&mut func, entry, Ty::ptr(Ty::ptr(ty.clone())), 0);
    let first_src = entry_param(&mut func, entry, Ty::ptr(ty.clone()), 1);
    let second_src = entry_param(&mut func, entry, Ty::ptr(ty.clone()), 2);
    let dst = stack_slot(&mut func, entry, ty);
    push_unit(
        &mut func,
        entry,
        InstKind::Store {
            addr: sink,
            val: dst,
            flags: crate::ir::MemFlags::plain(),
        },
    );
    let first = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: first_src, len: 16 });
    let second = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: second_src, len: 16 });
    func.blocks[entry].term = Some(Terminator::Return { val: None });

    let mut pass = DeadStoreElim;
    assert!(!pass.run(&mut func));
    assert!(func.blocks[entry].insts.contains(&first));
    assert!(func.blocks[entry].insts.contains(&second));
}

#[test]
fn keeps_copy_read_before_the_overwrite() {
    let ty = Ty::Struct {
        fields: vec![Ty::int(64), Ty::int(64)],
        offsets: vec![0, 8],
        size: 16,
    };
    let (mut func, entry) = Function::new("dse", Signature::new(vec![], Ty::Void));
    let dst = stack_slot(&mut func, entry, ty.clone());
    let first_src = stack_slot(&mut func, entry, ty.clone());
    let second_src = stack_slot(&mut func, entry, ty.clone());
    let first = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: first_src, len: 16 });
    let loaded = push_scalar_load(&mut func, entry, dst, Ty::int(64));
    let second = push_unit(&mut func, entry, InstKind::Memcpy { dst, src: second_src, len: 16 });
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    let mut pass = DeadStoreElim;
    assert!(!pass.run(&mut func));
    assert!(func.blocks[entry].insts.contains(&first));
    assert!(func.blocks[entry].insts.contains(&second));
}

#[test]
fn removes_dead_pure_insts() {
    let mut func = const_func();
    let dead_ty = Ty::int(32);
    let x = func.push_const(Const::Int { bits: 32, value: 99 });
    let y = func.push_const(Const::Int { bits: 32, value: 1 });
    let dead_val = func.values.push(crate::ir::ValueData {
        ty: dead_ty.clone(),
        def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
    });
    let dead_inst = func.insts.push(crate::ir::InstData {
        block: func.entry,
        kind: InstKind::Add(x, y),
        result: Some(dead_val),
        loc: None,
    });
    func.values[dead_val].def = Def::Inst(dead_inst);
    func.blocks[func.entry].insts.push(dead_inst);
    let mut dce = DeadCodeElim;
    assert!(dce.run(&mut func));
    assert!(!dce.run(&mut func));
}

#[test]
fn dce_keeps_unused_va_arg() {
    // An unused `VaArg` still consumes one argument, so removing it
    // would shift every later read of the list.
    let ap_ty = Ty::BuiltinVaList { size: 24, align: 8 };
    let sig = Signature::new(vec![Ty::ptr(ap_ty.clone())], Ty::Void);
    let (mut func, entry) = Function::new("va", sig);
    let ap = func.values.push(crate::ir::ValueData {
        ty: Ty::ptr(ap_ty),
        def: Def::Param(entry, 0),
    });
    func.blocks[entry].params.push(ap);
    let consumed = func.values.push(crate::ir::ValueData {
        ty: Ty::int(32),
        def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
    });
    let inst = func.insts.push(crate::ir::InstData {
        block: entry,
        kind: InstKind::VaArg { ap },
        result: Some(consumed),
        loc: None,
    });
    func.values[consumed].def = Def::Inst(inst);
    func.blocks[entry].insts.push(inst);
    func.blocks[entry].term = Some(Terminator::Return { val: None });
    let mut dce = DeadCodeElim;
    assert!(!dce.run(&mut func), "an unused va_arg must survive DCE");
    assert!(crate::verify::verify_func(&func).is_empty());
}

#[test]
fn drops_unreachable_blocks() {
    let sig = Signature::new(vec![], Ty::Void);
    let (mut func, entry) = Function::new("cfg", sig);
    let live = func.push_block(vec![]);
    let dead = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::Br {
        dest: live,
        args: Default::default(),
    });
    func.blocks[live].term = Some(Terminator::Return { val: None });
    func.blocks[dead].term = Some(Terminator::Return { val: None });
    let mut simp = SimplifyCfg;
    assert!(simp.run(&mut func));
    assert!(matches!(func.blocks[dead].term, Some(Terminator::Unreachable)));
}

#[test]
fn folds_identical_branch_arms() {
    let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("same-arm", sig);
    let cond = param(&mut func, entry, Ty::Bool);
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let shared = func.push_block(vec![Ty::int(32)]);
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block: shared,
        else_block: shared,
        then_args: [one].into_iter().collect(),
        else_args: [one].into_iter().collect(),
    });
    let p = func.blocks[shared].params[0];
    func.blocks[shared].term = Some(Terminator::Return { val: Some(p) });

    let mut simp = SimplifyCfg;
    assert!(simp.run(&mut func));
    assert!(crate::verify::verify_func(&func).is_empty());
    let Terminator::Return { val: Some(v) } = func.blocks[entry].term.clone().unwrap() else {
        panic!("branch was not folded");
    };
    assert_eq!(func.const_value(v), Some(&Const::Int { bits: 32, value: 1 }));
}

#[test]
fn folds_fork_to_select() {
    let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("fork", sig);
    let cond = param(&mut func, entry, Ty::Bool);
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let two = func.push_const(Const::Int { bits: 32, value: 2 });
    let shared = func.push_block(vec![Ty::int(32)]);
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block: shared,
        else_block: shared,
        then_args: [one].into_iter().collect(),
        else_args: [two].into_iter().collect(),
    });
    let p = func.blocks[shared].params[0];
    func.blocks[shared].term = Some(Terminator::Return { val: Some(p) });

    let mut simp = SimplifyCfg;
    assert!(simp.run(&mut func));
    assert!(crate::verify::verify_func(&func).is_empty());
    let selects: Vec<InstKind> = func.blocks[entry]
        .insts
        .iter()
        .map(|i| func.insts[*i].kind.clone())
        .filter(|k| matches!(k, InstKind::Select { .. }))
        .collect();
    assert_eq!(selects.len(), 1, "fork becomes one select: {selects:?}");
    assert!(matches!(func.blocks[entry].term, Some(Terminator::Return { val: Some(_) })));
}

#[test]
fn threads_empty_trampoline() {
    let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("thread", sig);
    let cond = param(&mut func, entry, Ty::Bool);
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let two = func.push_const(Const::Int { bits: 32, value: 2 });
    let left = func.push_block(vec![]);
    let right = func.push_block(vec![]);
    let join = func.push_block(vec![Ty::int(32)]);
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block: left,
        else_block: right,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    func.blocks[left].term = Some(Terminator::Br {
        dest: join,
        args: [one].into_iter().collect(),
    });
    func.blocks[right].term = Some(Terminator::Br {
        dest: join,
        args: [two].into_iter().collect(),
    });
    let p = func.blocks[join].params[0];
    func.blocks[join].term = Some(Terminator::Return { val: Some(p) });

    let mut simp = SimplifyCfg;
    assert!(simp.run(&mut func));
    assert!(crate::verify::verify_func(&func).is_empty());
    assert!(func.blocks[left].insts.is_empty());
    assert!(func.blocks[left].params.is_empty());
    assert!(matches!(func.blocks[left].term, Some(Terminator::Unreachable)));
    assert!(matches!(func.blocks[right].term, Some(Terminator::Unreachable)));
    assert!(func.blocks[entry].insts.iter().any(|i| matches!(func.insts[*i].kind, InstKind::Select { .. })));
}

#[test]
fn merges_identical_blocks() {
    let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("identical", sig);
    let cond = param(&mut func, entry, Ty::Bool);
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let left = func.push_block(vec![]);
    let right = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block: left,
        else_block: right,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    func.blocks[left].term = Some(Terminator::Return { val: Some(one) });
    func.blocks[right].term = Some(Terminator::Return { val: Some(one) });

    let mut simp = SimplifyCfg;
    assert!(simp.run(&mut func));
    assert!(crate::verify::verify_func(&func).is_empty());
    assert!(matches!(func.blocks[right].term, Some(Terminator::Unreachable)));
    let Terminator::Return { val: Some(v) } = func.blocks[entry].term.clone().unwrap() else {
        panic!("identical arms did not fold");
    };
    assert_eq!(v, one);
}

#[test]
fn merges_single_predecessor_chain() {
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("chain", sig);
    let mid = func.push_block(vec![]);
    let end = func.push_block(vec![Ty::int(32)]);
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let two = func.push_const(Const::Int { bits: 32, value: 2 });
    let sum = emit_bin(&mut func, mid, InstKind::Add(one, two), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Br {
        dest: mid,
        args: Default::default(),
    });
    func.blocks[mid].term = Some(Terminator::Br {
        dest: end,
        args: [sum].into_iter().collect(),
    });
    let q = func.blocks[end].params[0];
    func.blocks[end].term = Some(Terminator::Return { val: Some(q) });

    let mut simp = SimplifyCfg;
    assert!(simp.run(&mut func));
    assert!(crate::verify::verify_func(&func).is_empty());
    assert_eq!(func.blocks[entry].insts.len(), 1);
    let Terminator::Return { val: Some(v) } = func.blocks[entry].term.clone().unwrap() else {
        panic!("chain did not sink into the entry block");
    };
    assert_eq!(v, sum);
    assert!(matches!(func.blocks[mid].term, Some(Terminator::Unreachable)));
    assert!(matches!(func.blocks[end].term, Some(Terminator::Unreachable)));
    assert!(!simp.run(&mut func), "second run is a no-op");
}

#[test]
fn collapses_empty_diamond() {
    let sig = Signature::new(vec![], Ty::Void);
    let (mut func, entry) = Function::new("diamond", sig);
    let then_block = func.push_block(vec![]);
    let else_block = func.push_block(vec![]);
    let merge = func.push_block(vec![]);
    let cond = func.push_const(Const::Int { bits: 1, value: 1 });
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block,
        else_block,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    func.blocks[then_block].term = Some(Terminator::Br {
        dest: merge,
        args: Default::default(),
    });
    func.blocks[else_block].term = Some(Terminator::Br {
        dest: merge,
        args: Default::default(),
    });
    func.blocks[merge].term = Some(Terminator::Return { val: None });

    let mut simp = SimplifyCfg;
    assert!(simp.run(&mut func));
    assert!(crate::verify::verify_func(&func).is_empty());
    assert!(matches!(func.blocks[entry].term, Some(Terminator::Return { val: None })));
    assert!(matches!(func.blocks[then_block].term, Some(Terminator::Unreachable)));
    assert!(matches!(func.blocks[else_block].term, Some(Terminator::Unreachable)));
    assert!(matches!(func.blocks[merge].term, Some(Terminator::Unreachable)));
}

#[test]
fn if_converts_pure_diamond() {
    let sig = Signature::new(vec![Ty::Bool, Ty::int(32), Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("diamond", sig);
    let cond = param(&mut func, entry, Ty::Bool);
    let x = param(&mut func, entry, Ty::int(32));
    let y = param(&mut func, entry, Ty::int(32));
    let arm = func.push_block(vec![]);
    let merge = func.push_block(vec![Ty::int(32)]);
    let result = func.blocks[merge].params[0];
    let sum = emit_bin(&mut func, arm, InstKind::Add(x, y), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block: arm,
        else_block: merge,
        then_args: Default::default(),
        else_args: [x].into_iter().collect(),
    });
    func.blocks[arm].term = Some(Terminator::Br {
        dest: merge,
        args: [sum].into_iter().collect(),
    });
    func.blocks[merge].term = Some(Terminator::Return { val: Some(result) });

    let mut simp = SimplifyCfg;
    assert!(simp.run(&mut func));
    assert!(crate::verify::verify_func(&func).is_empty());
    assert!(matches!(func.blocks[arm].term, Some(Terminator::Unreachable)));
    let select = func.blocks[entry]
        .insts
        .iter()
        .find_map(|inst| match &func.insts[*inst].kind {
            InstKind::Select { cond: c, then_val, else_val } if *c == cond && *else_val == x => Some((*inst, *then_val)),
            _ => None,
        })
        .expect("diamond arm became a select");
    let Def::Inst(cloned_sum) = func.values[select.1].def else {
        panic!("select arm was not a computed value");
    };
    assert!(matches!(
        func.insts[cloned_sum].kind,
        InstKind::Add(a, b) if a == x && b == y
    ));

    assert!(simp.run(&mut func));
    assert!(crate::verify::verify_func(&func).is_empty());
    let Terminator::Return { val: Some(value) } = func.blocks[entry].term.clone().unwrap() else {
        panic!("join did not merge into entry");
    };
    assert!(matches!(
        func.values[value].def,
        Def::Inst(inst) if matches!(func.insts[inst].kind, InstKind::Select { .. })
    ));
}

#[test]
fn if_converts_a_diamond_with_two_pure_arms() {
    let sig = Signature::new(vec![Ty::Bool, Ty::int(32), Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("two-arms", sig);
    let cond = param(&mut func, entry, Ty::Bool);
    let x = param(&mut func, entry, Ty::int(32));
    let y = param(&mut func, entry, Ty::int(32));
    let then_block = func.push_block(vec![]);
    let else_block = func.push_block(vec![]);
    let merge = func.push_block(vec![Ty::int(32)]);
    let result = func.blocks[merge].params[0];
    let sum = emit_bin(&mut func, then_block, InstKind::Add(x, y), Ty::int(32));
    let difference = emit_bin(&mut func, else_block, InstKind::Sub(x, y), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block,
        else_block,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    func.blocks[then_block].term = Some(Terminator::Br {
        dest: merge,
        args: [sum].into_iter().collect(),
    });
    func.blocks[else_block].term = Some(Terminator::Br {
        dest: merge,
        args: [difference].into_iter().collect(),
    });
    func.blocks[merge].term = Some(Terminator::Return { val: Some(result) });

    let mut simp = SimplifyCfg;
    assert!(simp.run(&mut func));
    assert!(crate::verify::verify_func(&func).is_empty());
    assert!(matches!(func.blocks[then_block].term, Some(Terminator::Unreachable)));
    assert!(matches!(func.blocks[else_block].term, Some(Terminator::Unreachable)));
    let select = func.blocks[entry]
        .insts
        .iter()
        .find_map(|inst| match &func.insts[*inst].kind {
            InstKind::Select { cond: c, then_val, else_val } if *c == cond => Some((*then_val, *else_val)),
            _ => None,
        })
        .expect("both diamond arms became one select");
    let Def::Inst(then_inst) = func.values[select.0].def else {
        panic!("then arm was not cloned");
    };
    let Def::Inst(else_inst) = func.values[select.1].def else {
        panic!("else arm was not cloned");
    };
    assert!(matches!(func.insts[then_inst].kind, InstKind::Add(a, b) if a == x && b == y));
    assert!(matches!(func.insts[else_inst].kind, InstKind::Sub(a, b) if a == x && b == y));
}

#[test]
fn if_conversion_substitutes_arm_parameters() {
    let sig = Signature::new(vec![Ty::Bool, Ty::int(32), Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("param-diamond", sig);
    let cond = param(&mut func, entry, Ty::Bool);
    let x = param(&mut func, entry, Ty::int(32));
    let y = param(&mut func, entry, Ty::int(32));
    let arm = func.push_block(vec![Ty::int(32)]);
    let param_in_arm = func.blocks[arm].params[0];
    let merge = func.push_block(vec![Ty::int(32)]);
    let result = func.blocks[merge].params[0];
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let sum = emit_bin(&mut func, arm, InstKind::Add(param_in_arm, one), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block: arm,
        else_block: merge,
        then_args: [x].into_iter().collect(),
        else_args: [y].into_iter().collect(),
    });
    func.blocks[arm].term = Some(Terminator::Br {
        dest: merge,
        args: [sum].into_iter().collect(),
    });
    func.blocks[merge].term = Some(Terminator::Return { val: Some(result) });

    let mut simp = SimplifyCfg;
    assert!(simp.run(&mut func));
    assert!(crate::verify::verify_func(&func).is_empty());
    assert!(matches!(func.blocks[arm].term, Some(Terminator::Unreachable)));
    let select = func.blocks[entry]
        .insts
        .iter()
        .find_map(|inst| match &func.insts[*inst].kind {
            InstKind::Select { cond: c, then_val, else_val } if *c == cond && *else_val == y => Some(*then_val),
            _ => None,
        })
        .expect("parameterized diamond arm became a select");
    let Def::Inst(select_inst) = func.values[select].def else {
        panic!("select arm was not computed");
    };
    let InstKind::Add(lhs, rhs) = func.insts[select_inst].kind else {
        panic!("arm computation was not cloned");
    };
    assert_eq!(lhs, x);
    assert_eq!(func.const_value(rhs), Some(&Const::Int { bits: 32, value: 1 }));
}

#[test]
fn does_not_if_convert_trapping_arms() {
    let sig = Signature::new(vec![Ty::Bool, Ty::int(32), Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("trap", sig);
    let cond = param(&mut func, entry, Ty::Bool);
    let x = param(&mut func, entry, Ty::int(32));
    let y = param(&mut func, entry, Ty::int(32));
    let arm = func.push_block(vec![]);
    let merge = func.push_block(vec![Ty::int(32)]);
    let result = func.blocks[merge].params[0];
    let quotient = emit_bin(&mut func, arm, InstKind::Sdiv(x, y), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block: arm,
        else_block: merge,
        then_args: Default::default(),
        else_args: [x].into_iter().collect(),
    });
    func.blocks[arm].term = Some(Terminator::Br {
        dest: merge,
        args: [quotient].into_iter().collect(),
    });
    func.blocks[merge].term = Some(Terminator::Return { val: Some(result) });

    let mut simp = SimplifyCfg;
    assert!(!simp.run(&mut func));
    let Def::Inst(quotient_inst) = func.values[quotient].def else {
        panic!("expected a division instruction");
    };
    assert!(func.blocks[arm].insts.contains(&quotient_inst));
    assert!(crate::verify::verify_func(&func).is_empty());
}

#[test]
fn keeps_loop_headers_and_entries() {
    // The header has two preds, so the one-pred latch cannot sink
    // into it.
    let sig = Signature::new(vec![Ty::Bool], Ty::Void);
    let (mut func, entry) = Function::new("loop", sig);
    let header = func.push_block(vec![]);
    let body = func.push_block(vec![]);
    let exit = func.push_block(vec![]);
    let cond = param(&mut func, entry, Ty::Bool);
    func.blocks[entry].term = Some(Terminator::Br {
        dest: header,
        args: Default::default(),
    });
    func.blocks[header].term = Some(Terminator::CondBr {
        cond,
        then_block: body,
        else_block: exit,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    func.blocks[body].term = Some(Terminator::Br {
        dest: header,
        args: Default::default(),
    });
    func.blocks[exit].term = Some(Terminator::Return { val: None });
    let mut simp = SimplifyCfg;
    assert!(!simp.run(&mut func));
}

#[test]
fn keeps_entry_block() {
    // The only pred of the entry is a latch: sinking the entry into
    // it would change where execution starts.
    let sig = Signature::new(vec![Ty::Bool], Ty::Void);
    let (mut func, entry) = Function::new("entry-loop", sig);
    let body = func.push_block(vec![]);
    let exit = func.push_block(vec![]);
    let cond = param(&mut func, entry, Ty::Bool);
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block: body,
        else_block: exit,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    func.blocks[body].term = Some(Terminator::Br {
        dest: entry,
        args: Default::default(),
    });
    func.blocks[exit].term = Some(Terminator::Return { val: None });
    let mut simp = SimplifyCfg;
    assert!(!simp.run(&mut func));
}

#[test]
fn sinks_params_through_successors() {
    let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("sink", sig);
    let x = func.values.push(crate::ir::ValueData {
        ty: Ty::int(32),
        def: Def::Param(entry, 0),
    });
    func.blocks[entry].params.push(x);
    let mid = func.push_block(vec![Ty::int(32)]);
    let end = func.push_block(vec![Ty::int(32)]);
    func.blocks[entry].term = Some(Terminator::Br {
        dest: mid,
        args: [x].into_iter().collect(),
    });
    let p = func.blocks[mid].params[0];
    func.blocks[mid].term = Some(Terminator::Br {
        dest: end,
        args: [p].into_iter().collect(),
    });
    let q = func.blocks[end].params[0];
    func.blocks[end].term = Some(Terminator::Return { val: Some(q) });

    let mut simp = SimplifyCfg;
    assert!(simp.run(&mut func));
    assert!(crate::verify::verify_func(&func).is_empty());
    let Terminator::Return { val: Some(v) } = func.blocks[entry].term.clone().unwrap() else {
        panic!("sink did not collapse the chain");
    };
    assert_eq!(v, x, "param must resolve back to the entry argument");
}

#[test]
fn sinks_branch_only_work_into_the_using_arm() {
    // Both computations feed only the `then` return; the `else`
    // path must not run them.
    let sig = Signature::new(vec![Ty::int(32), Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("sink", sig);
    let x = param(&mut func, entry, Ty::int(32));
    let take = param(&mut func, entry, Ty::Bool);
    let add = emit_bin(&mut func, entry, InstKind::Add(x, x), Ty::int(32));
    let mul = emit_bin(&mut func, entry, InstKind::Mul(add, add), Ty::int(32));
    let then_block = func.push_block(vec![]);
    let else_block = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond: take,
        then_block,
        else_block,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    func.blocks[then_block].term = Some(Terminator::Return { val: Some(mul) });
    let fallback = func.push_const(Const::Int { bits: 32, value: 123 });
    func.blocks[else_block].term = Some(Terminator::Return { val: Some(fallback) });

    let mut simp = SimplifyCfg;
    assert!(simp.run(&mut func));
    assert!(crate::verify::verify_func(&func).is_empty());
    assert!(func.blocks[entry].insts.is_empty(), "entry keeps only the branch");
    let kinds: Vec<InstKind> = func.blocks[then_block].insts.iter().map(|i| func.insts[*i].kind.clone()).collect();
    assert!(matches!(kinds[0], InstKind::Add(..)), "add sinks first: {kinds:?}");
    assert!(matches!(kinds[1], InstKind::Mul(..)), "mul follows its operand: {kinds:?}");
    assert!(!simp.run(&mut func), "second run is a no-op");
}

#[test]
fn keeps_work_used_by_both_arms() {
    let sig = Signature::new(vec![Ty::int(32), Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let x = param(&mut func, entry, Ty::int(32));
    let take = param(&mut func, entry, Ty::Bool);
    let neg = emit_bin(&mut func, entry, InstKind::Neg(x), Ty::int(32));
    let then_block = func.push_block(vec![]);
    let else_block = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond: take,
        then_block,
        else_block,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    func.blocks[then_block].term = Some(Terminator::Return { val: Some(neg) });
    func.blocks[else_block].term = Some(Terminator::Return { val: Some(neg) });
    let mut simp = SimplifyCfg;
    assert!(simp.run(&mut func));
    assert_eq!(func.blocks[entry].insts.len(), 1, "shared work stays in the entry");
    let Terminator::Return { val: Some(v) } = func.blocks[entry].term.clone().unwrap() else {
        panic!("identical arms should collapse");
    };
    assert_eq!(v, neg);
    assert!(!simp.run(&mut func), "second run is a no-op");
}

#[test]
fn keeps_work_out_of_blocks_with_several_predecessors() {
    // `v` is needed inside the loop, but moving it to the header
    // would re-run it every iteration; it must stay in the
    // preheader.
    let sig = Signature::new(vec![Ty::int(32), Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let x = param(&mut func, entry, Ty::int(32));
    let cond = param(&mut func, entry, Ty::Bool);
    let v = emit_bin(&mut func, entry, InstKind::Add(x, x), Ty::int(32));
    let header = func.push_block(vec![]);
    let body = func.push_block(vec![]);
    let exit = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::Br {
        dest: header,
        args: Default::default(),
    });
    func.blocks[header].term = Some(Terminator::CondBr {
        cond,
        then_block: body,
        else_block: exit,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    emit_bin(&mut func, body, InstKind::Add(v, x), Ty::int(32));
    func.blocks[body].term = Some(Terminator::Br {
        dest: header,
        args: Default::default(),
    });
    let zero = func.push_const(Const::Int { bits: 32, value: 0 });
    func.blocks[exit].term = Some(Terminator::Return { val: Some(zero) });
    let mut simp = SimplifyCfg;
    assert!(!simp.run(&mut func));
    assert_eq!(func.blocks[entry].insts.len(), 1, "the add stays in the preheader");
}

#[test]
fn inlines_small_callee() {
    let mut module = Module::new();
    let callee_sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
    let (mut callee, centry) = Function::new("add_one", callee_sig);
    let arg = callee.values.push(crate::ir::ValueData {
        ty: Ty::int(32),
        def: Def::Param(centry, 0),
    });
    callee.blocks[centry].params.push(arg);
    let one = callee.push_const(Const::Int { bits: 32, value: 1 });
    let sum = callee.values.push(crate::ir::ValueData {
        ty: Ty::int(32),
        def: Def::Inst(crate::ir::Inst::from_usize(0)),
    });
    let inst = callee.insts.push(crate::ir::InstData {
        block: centry,
        kind: InstKind::Add(arg, one),
        result: Some(sum),
        loc: None,
    });
    callee.values[sum].def = Def::Inst(inst);
    callee.blocks[centry].insts.push(inst);
    callee.blocks[centry].term = Some(Terminator::Return { val: Some(sum) });
    let callee_id = module.push_func(callee);

    let caller_sig = Signature::new(vec![], Ty::int(32));
    let (mut caller, entry) = Function::new("caller", caller_sig);
    let five = caller.push_const(Const::Int { bits: 32, value: 41 });
    let call_ty = Ty::int(32);
    let call_val = caller.values.push(crate::ir::ValueData {
        ty: call_ty.clone(),
        def: Def::Inst(crate::ir::Inst::from_usize(0)),
    });
    let call = caller.insts.push(crate::ir::InstData {
        block: entry,
        kind: InstKind::Call {
            callee: callee_id,
            args: [five].into_iter().collect(),
        },
        result: Some(call_val),
        loc: None,
    });
    caller.values[call_val].def = Def::Inst(call);
    caller.blocks[entry].insts.push(call);
    caller.blocks[entry].term = Some(Terminator::Return { val: Some(call_val) });
    module.push_func(caller);

    let mut inline = Inline::small();
    assert!(inline.run(&mut module));
    let caller = &module.funcs[crate::ir::Func::from_usize(1)];
    let still_calls = caller
        .blocks
        .iter()
        .flat_map(|b| b.insts.iter())
        .any(|i| matches!(caller.insts[*i].kind, InstKind::Call { .. }));
    assert!(!still_calls);
    // Inlined `41 + 1` must fold to 42: the callee constant has to be
    // materialized fresh in the caller arena, not reused by index.
    let caller_id = crate::ir::Func::from_usize(1);
    let mut fold = ConstFold;
    fold.run(&mut module.funcs[caller_id]);
    let ret = match &module.funcs[caller_id].blocks[entry].term {
        Some(Terminator::Return { val }) => val.unwrap(),
        _ => panic!("caller returns a value"),
    };
    assert_eq!(module.funcs[caller_id].const_value(ret), Some(&Const::Int { bits: 32, value: 42 }));
}

#[test]
fn inlining_keeps_undef_type() {
    let mut module = Module::new();
    let callee_sig = Signature::new(vec![], Ty::int(32));
    let (mut callee, centry) = Function::new("undef_callee", callee_sig);
    let merge = callee.push_block(vec![Ty::int(32)]);
    let undef = callee.values.push(crate::ir::ValueData {
        ty: Ty::int(32),
        def: Def::Const(Const::Undef),
    });
    callee.blocks[centry].term = Some(Terminator::Br {
        dest: merge,
        args: [undef].into_iter().collect(),
    });
    let param = callee.blocks[merge].params[0];
    callee.blocks[merge].term = Some(Terminator::Return { val: Some(param) });
    callee.inline = true;
    let callee_id = module.push_func(callee);

    let caller_sig = Signature::new(vec![], Ty::int(32));
    let (mut caller, entry) = Function::new("caller", caller_sig);
    let call_val = caller.values.push(crate::ir::ValueData {
        ty: Ty::int(32),
        def: Def::Inst(crate::ir::Inst::from_usize(0)),
    });
    let call = caller.insts.push(crate::ir::InstData {
        block: entry,
        kind: InstKind::Call {
            callee: callee_id,
            args: Default::default(),
        },
        result: Some(call_val),
        loc: None,
    });
    caller.values[call_val].def = Def::Inst(call);
    caller.blocks[entry].insts.push(call);
    caller.blocks[entry].term = Some(Terminator::Return { val: Some(call_val) });
    module.push_func(caller);

    let mut inline = Inline::small();
    assert!(inline.run(&mut module));
    let caller = &module.funcs[crate::ir::Func::from_usize(1)];
    let mut found = false;
    for block in caller.blocks.iter() {
        if let Some(Terminator::Br { args, .. }) = &block.term {
            for &arg in args {
                if caller.const_value(arg) == Some(&Const::Undef) {
                    found = true;
                    assert_eq!(caller.values[arg].ty, Ty::int(32));
                }
            }
        }
    }
    assert!(found, "inlined branch keeps its undef argument");
}

#[test]
fn inlining_maps_forward_referenced_values() {
    let mut module = Module::new();
    // Blocks clone in arena order, which is not a def-before-use order:
    // earlier inlinings move a block's tail into a fresh successor, so a
    // later-arena block can define a value used by an earlier-arena one.
    let callee_sig = Signature::new(vec![], Ty::Void);
    let (mut callee, entry) = Function::new("late_def", callee_sig);
    let use_block = callee.push_block(vec![]);
    let def_block = callee.push_block(vec![]);
    let exit = callee.push_block(vec![]);
    let one = callee.push_const(Const::Int { bits: 64, value: 1 });
    let null = callee.values.push(crate::ir::ValueData {
        ty: Ty::ptr(Ty::int(64)),
        def: Def::Const(Const::Null),
    });
    let sum = callee.values.push(crate::ir::ValueData {
        ty: Ty::int(64),
        def: Def::Inst(crate::ir::Inst::from_usize(0)),
    });
    let store = callee.insts.push(crate::ir::InstData {
        block: use_block,
        kind: InstKind::Store {
            addr: null,
            val: sum,
            flags: crate::ir::MemFlags::plain(),
        },
        result: None,
        loc: None,
    });
    callee.blocks[use_block].insts.push(store);
    let add = callee.insts.push(crate::ir::InstData {
        block: def_block,
        kind: InstKind::Add(one, one),
        result: Some(sum),
        loc: None,
    });
    callee.values[sum].def = Def::Inst(add);
    callee.blocks[def_block].insts.push(add);
    callee.blocks[entry].term = Some(Terminator::Br {
        dest: def_block,
        args: Default::default(),
    });
    callee.blocks[def_block].term = Some(Terminator::Br {
        dest: use_block,
        args: Default::default(),
    });
    callee.blocks[use_block].term = Some(Terminator::Br {
        dest: exit,
        args: Default::default(),
    });
    callee.blocks[exit].term = Some(Terminator::Return { val: None });
    callee.inline = true;
    let callee_id = module.push_func(callee);

    // Caller value 2 differs in type from the callee's, so an aliased
    // operand shows up as a type mismatch rather than as silent
    // corruption.
    let caller_sig = Signature::new(vec![], Ty::Void);
    let (mut caller, centry) = Function::new("caller", caller_sig);
    caller.push_const(Const::Int { bits: 64, value: 1 });
    caller.push_const(Const::Int { bits: 64, value: 2 });
    caller.push_const(Const::Int { bits: 32, value: 3 });
    let call = caller.insts.push(crate::ir::InstData {
        block: centry,
        kind: InstKind::Call {
            callee: callee_id,
            args: Default::default(),
        },
        result: None,
        loc: None,
    });
    caller.blocks[centry].insts.push(call);
    caller.blocks[centry].term = Some(Terminator::Return { val: None });
    module.push_func(caller);

    let mut inline = Inline::small();
    assert!(inline.run(&mut module));
    let caller = &module.funcs[crate::ir::Func::from_usize(1)];
    let mut stores = 0;
    for block in caller.blocks.iter() {
        for inst in &block.insts {
            if let InstKind::Store { val, .. } = &caller.insts[*inst].kind {
                stores += 1;
                assert_eq!(caller.values[*val].ty, Ty::int(64));
                assert!(matches!(caller.values[*val].def, Def::Inst(_)));
            }
        }
    }
    assert_eq!(stores, 1, "inlined body still stores");
}

#[test]
fn inlines_callee_with_inline_asm() {
    let mut module = Module::new();
    let callee_sig = Signature::new(vec![], Ty::int(32));
    let (mut callee, centry) = Function::new("with_asm", callee_sig);
    let five = callee.push_const(Const::Int { bits: 32, value: 5 });
    let block = AsmBlock {
        template: "movl $%c0, %0".to_string(),
        operands: vec![AsmOperand {
            value: five,
            constraint: "=r".to_string(),
            ty: Ty::int(32),
            tie: None,
            name: None,
        }],
        outputs: 1,
        clobbers: Vec::new(),
        labels: Vec::new(),
        is_volatile: true,
        is_basic: false,
    };
    let inst = callee.insts.push(crate::ir::InstData {
        block: centry,
        kind: InstKind::InlineAsm { block: Box::new(block) },
        result: None,
        loc: None,
    });
    callee.blocks[centry].insts.push(inst);
    callee.blocks[centry].term = Some(Terminator::Return { val: Some(five) });
    let callee_id = module.push_func(callee);

    let caller_sig = Signature::new(vec![], Ty::int(32));
    let (mut caller, entry) = Function::new("caller", caller_sig);
    let call_val = caller.values.push(crate::ir::ValueData {
        ty: Ty::int(32),
        def: Def::Inst(crate::ir::Inst::from_usize(0)),
    });
    let call = caller.insts.push(crate::ir::InstData {
        block: entry,
        kind: InstKind::Call {
            callee: callee_id,
            args: [].into_iter().collect(),
        },
        result: Some(call_val),
        loc: None,
    });
    caller.values[call_val].def = Def::Inst(call);
    caller.blocks[entry].insts.push(call);
    caller.blocks[entry].term = Some(Terminator::Return { val: Some(call_val) });
    module.push_func(caller);

    let mut inline = Inline::small();
    assert!(inline.run(&mut module));
    let caller = &module.funcs[crate::ir::Func::from_usize(1)];
    let live: Vec<Inst> = caller.blocks.iter().flat_map(|b| b.insts.iter().copied()).collect();
    assert!(!live.iter().any(|inst| matches!(caller.insts[*inst].kind, InstKind::Call { .. })));
    // The operand must point at a caller value (a fresh constant),
    // never at the callee's arena.
    let operand = live
        .iter()
        .find_map(|inst| match &caller.insts[*inst].kind {
            InstKind::InlineAsm { block } => Some(block.operands[0].value),
            _ => None,
        })
        .expect("inline asm survives inlining");
    assert_eq!(caller.const_value(operand), Some(&Const::Int { bits: 32, value: 5 }));
}

fn branchy_callee(marked_inline: bool) -> (Module, crate::ir::Func) {
    // `pick(x) = x > 0 ? 1 : 0`: entry splits to two returns.
    let mut module = Module::new();
    let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
    let (mut callee, entry) = Function::new("pick", sig);
    callee.inline = marked_inline;
    let arg = callee.values.push(crate::ir::ValueData {
        ty: Ty::int(32),
        def: Def::Param(entry, 0),
    });
    callee.blocks[entry].params.push(arg);
    let zero = callee.push_const(Const::Int { bits: 32, value: 0 });
    let cond = callee.values.push(crate::ir::ValueData {
        ty: Ty::Bool,
        def: Def::Inst(crate::ir::Inst::from_usize(callee.insts.len())),
    });
    let cmp = callee.insts.push(crate::ir::InstData {
        block: entry,
        kind: InstKind::Icmp {
            pred: IcmpPred::Sgt,
            lhs: arg,
            rhs: zero,
        },
        result: Some(cond),
        loc: None,
    });
    callee.values[cond].def = Def::Inst(cmp);
    callee.blocks[entry].insts.push(cmp);
    let then_b = callee.push_block(vec![]);
    let else_b = callee.push_block(vec![]);
    callee.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block: then_b,
        else_block: else_b,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    let one = callee.push_const(Const::Int { bits: 32, value: 1 });
    callee.blocks[then_b].term = Some(Terminator::Return { val: Some(one) });
    let zero_ret = callee.push_const(Const::Int { bits: 32, value: 0 });
    callee.blocks[else_b].term = Some(Terminator::Return { val: Some(zero_ret) });
    let callee_id = module.push_func(callee);

    let caller_sig = Signature::new(vec![], Ty::int(32));
    let (mut caller, centry) = Function::new("caller", caller_sig);
    let five = caller.push_const(Const::Int { bits: 32, value: 5 });
    let call_val = caller.values.push(crate::ir::ValueData {
        ty: Ty::int(32),
        def: Def::Inst(crate::ir::Inst::from_usize(caller.insts.len())),
    });
    let call = caller.insts.push(crate::ir::InstData {
        block: centry,
        kind: InstKind::Call {
            callee: callee_id,
            args: [five].into_iter().collect(),
        },
        result: Some(call_val),
        loc: None,
    });
    caller.values[call_val].def = Def::Inst(call);
    caller.blocks[centry].insts.push(call);
    caller.blocks[centry].term = Some(Terminator::Return { val: Some(call_val) });
    module.push_func(caller);
    (module, callee_id)
}

fn caller_still_calls(module: &Module, caller: crate::ir::Func) -> bool {
    let func = &module.funcs[caller];
    func.blocks
        .iter()
        .flat_map(|b| b.insts.iter())
        .any(|i| matches!(func.insts[*i].kind, InstKind::Call { .. }))
}

#[test]
fn inlines_marked_multi_block_callee() {
    let (mut module, _) = branchy_callee(true);
    let caller = crate::ir::Func::from_usize(1);
    let mut inline = Inline::small();
    assert!(inline.run(&mut module));
    assert!(!caller_still_calls(&module, caller));
    assert!(crate::verify::verify_module(&module).is_empty());
}

#[test]
fn inlines_small_unmarked_multi_block_callee() {
    let (mut module, _) = branchy_callee(false);
    let caller = crate::ir::Func::from_usize(1);
    let mut inline = Inline::small();
    assert!(inline.run(&mut module));
    assert!(!caller_still_calls(&module, caller));
    assert!(crate::verify::verify_module(&module).is_empty());
}

#[test]
fn refuses_oversized_unmarked_multi_block_callee() {
    let (mut module, _) = branchy_callee(false);
    let caller = crate::ir::Func::from_usize(1);
    let mut inline = Inline::small();
    inline.max_insts = 0;
    inline.max_hot_insts = 0;
    inline.max_single_insts = 0;
    assert!(!inline.run(&mut module));
    assert!(caller_still_calls(&module, caller));
}

#[test]
fn inlines_hot_multi_block_callee() {
    let (mut module, _) = branchy_callee(false);
    let caller = crate::ir::Func::from_usize(1);
    let entry = module.funcs[caller].entry;
    let call = module.funcs[caller].blocks[entry]
        .insts
        .iter()
        .copied()
        .find(|inst| matches!(module.funcs[caller].insts[*inst].kind, InstKind::Call { .. }))
        .expect("call site");
    let result = module.funcs[caller].insts[call].result.expect("call result");
    let (callee, args) = match &module.funcs[caller].insts[call].kind {
        InstKind::Call { callee, args } => (*callee, args.clone()),
        _ => unreachable!("found a call"),
    };
    // A second call site keeps `call_counts` above one, so the
    // single-call budget cannot mask the hot-call budget.
    let body = module.funcs[caller].push_block(vec![]);
    let exit = module.funcs[caller].push_block(vec![]);
    let cond = module.funcs[caller].push_const(Const::Int { bits: 1, value: 1 });
    module.funcs[caller].blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block: body,
        else_block: exit,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    let next_inst = crate::ir::Inst::from_usize(module.funcs[caller].insts.len());
    let second = module.funcs[caller].values.push(crate::ir::ValueData {
        ty: Ty::int(32),
        def: Def::Inst(next_inst),
    });
    let extra = module.funcs[caller].insts.push(crate::ir::InstData {
        block: body,
        kind: InstKind::Call { callee, args },
        result: Some(second),
        loc: None,
    });
    module.funcs[caller].values[second].def = Def::Inst(extra);
    module.funcs[caller].blocks[body].insts.push(extra);
    module.funcs[caller].blocks[body].term = Some(Terminator::Br {
        dest: entry,
        args: Default::default(),
    });
    module.funcs[caller].blocks[exit].term = Some(Terminator::Return { val: Some(result) });

    let mut inline = Inline::small();
    inline.max_insts = 0;
    inline.max_single_insts = 0;
    assert!(inline.run(&mut module));
    assert!(!caller_still_calls(&module, caller));
    assert!(crate::verify::verify_module(&module).is_empty());
}

#[test]
fn inlines_single_call_callee_on_the_larger_budget() {
    let (mut module, _) = branchy_callee(false);
    let caller = crate::ir::Func::from_usize(1);
    let mut inline = Inline::small();
    inline.max_insts = 0;
    inline.max_hot_insts = 0;
    assert!(inline.run(&mut module));
    assert!(!caller_still_calls(&module, caller));
    assert!(crate::verify::verify_module(&module).is_empty());
}

#[test]
fn inline_only_config_inlines_marked_but_not_plain() {
    let (mut marked, _) = branchy_callee(true);
    let mut only = Inline::inline_only();
    assert!(only.run(&mut marked));
    assert!(!caller_still_calls(&marked, crate::ir::Func::from_usize(1)));

    // A small unmarked single-block callee inlines under `small`
    // but not under `inline_only` (one real inst, over the zero
    // heuristic budget but under the keyword budget).
    for (make_marked, expect_inline_only) in [(false, false), (true, true)] {
        let mut module = Module::new();
        let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
        let (mut callee, centry) = Function::new("add_one", sig);
        callee.inline = make_marked;
        let arg = callee.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Param(centry, 0),
        });
        callee.blocks[centry].params.push(arg);
        let one = callee.push_const(Const::Int { bits: 32, value: 1 });
        let sum = callee.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Inst(crate::ir::Inst::from_usize(callee.insts.len())),
        });
        let add = callee.insts.push(crate::ir::InstData {
            block: centry,
            kind: InstKind::Add(arg, one),
            result: Some(sum),
            loc: None,
        });
        callee.values[sum].def = Def::Inst(add);
        callee.blocks[centry].insts.push(add);
        callee.blocks[centry].term = Some(Terminator::Return { val: Some(sum) });
        let callee_id = module.push_func(callee);
        let csig = Signature::new(vec![], Ty::int(32));
        let (mut caller, entry) = Function::new("caller", csig);
        let five = caller.push_const(Const::Int { bits: 32, value: 5 });
        let rv = caller.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Inst(crate::ir::Inst::from_usize(caller.insts.len())),
        });
        let call = caller.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Call {
                callee: callee_id,
                args: [five].into_iter().collect(),
            },
            result: Some(rv),
            loc: None,
        });
        caller.values[rv].def = Def::Inst(call);
        caller.blocks[entry].insts.push(call);
        caller.blocks[entry].term = Some(Terminator::Return { val: Some(rv) });
        module.push_func(caller);
        let mut only = Inline::inline_only();
        assert_eq!(only.run(&mut module), expect_inline_only, "marked={make_marked}");
    }
}

#[test]
fn marked_callee_gets_larger_budget() {
    // 30 chained adds: over the lowered heuristic budgets but under
    // the 256-inst keyword budget.
    for marked in [false, true] {
        let mut module = Module::new();
        let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
        let (mut callee, centry) = Function::new("long", sig);
        callee.inline = marked;
        let arg = callee.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Param(centry, 0),
        });
        callee.blocks[centry].params.push(arg);
        let one = callee.push_const(Const::Int { bits: 32, value: 1 });
        let mut acc = arg;
        for _ in 0..30 {
            let next = callee.values.push(crate::ir::ValueData {
                ty: Ty::int(32),
                def: Def::Inst(crate::ir::Inst::from_usize(callee.insts.len())),
            });
            let inst = callee.insts.push(crate::ir::InstData {
                block: centry,
                kind: InstKind::Add(acc, one),
                result: Some(next),
                loc: None,
            });
            callee.values[next].def = Def::Inst(inst);
            callee.blocks[centry].insts.push(inst);
            acc = next;
        }
        callee.blocks[centry].term = Some(Terminator::Return { val: Some(acc) });
        let callee_id = module.push_func(callee);
        let csig = Signature::new(vec![], Ty::int(32));
        let (mut caller, entry) = Function::new("caller", csig);
        let five = caller.push_const(Const::Int { bits: 32, value: 0 });
        let rv = caller.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Inst(crate::ir::Inst::from_usize(caller.insts.len())),
        });
        let call = caller.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Call {
                callee: callee_id,
                args: [five].into_iter().collect(),
            },
            result: Some(rv),
            loc: None,
        });
        caller.values[rv].def = Def::Inst(call);
        caller.blocks[entry].insts.push(call);
        caller.blocks[entry].term = Some(Terminator::Return { val: Some(rv) });
        module.push_func(caller);
        let mut inline = Inline::small();
        inline.max_insts = 24;
        inline.max_hot_insts = 24;
        inline.max_single_insts = 0;
        assert_eq!(inline.run(&mut module), marked, "marked={marked} should control the larger budget");
        assert_eq!(caller_still_calls(&module, crate::ir::Func::from_usize(1)), !marked);
    }
}

#[test]
fn refuses_recursive_callee() {
    let mut module = Module::new();
    let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
    let (mut callee, centry) = Function::new("rec", sig);
    callee.inline = true;
    let arg = callee.values.push(crate::ir::ValueData {
        ty: Ty::int(32),
        def: Def::Param(centry, 0),
    });
    callee.blocks[centry].params.push(arg);
    let callee_id = module.push_func(callee);
    // Back-patch a self-call: `rec(arg)`.
    {
        let func = &mut module.funcs[callee_id];
        let rv = func.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
        });
        let call = func.insts.push(crate::ir::InstData {
            block: centry,
            kind: InstKind::Call {
                callee: callee_id,
                args: [arg].into_iter().collect(),
            },
            result: Some(rv),
            loc: None,
        });
        func.values[rv].def = Def::Inst(call);
        func.blocks[centry].insts.push(call);
        func.blocks[centry].term = Some(Terminator::Return { val: Some(rv) });
    }
    let csig = Signature::new(vec![], Ty::int(32));
    let (mut caller, entry) = Function::new("caller", csig);
    let five = caller.push_const(Const::Int { bits: 32, value: 1 });
    let rv = caller.values.push(crate::ir::ValueData {
        ty: Ty::int(32),
        def: Def::Inst(crate::ir::Inst::from_usize(caller.insts.len())),
    });
    let call = caller.insts.push(crate::ir::InstData {
        block: entry,
        kind: InstKind::Call {
            callee: callee_id,
            args: [five].into_iter().collect(),
        },
        result: Some(rv),
        loc: None,
    });
    caller.values[rv].def = Def::Inst(call);
    caller.blocks[entry].insts.push(call);
    caller.blocks[entry].term = Some(Terminator::Return { val: Some(rv) });
    module.push_func(caller);
    let mut inline = Inline::small();
    assert!(!inline.run(&mut module));
    assert!(caller_still_calls(&module, crate::ir::Func::from_usize(1)));
}

#[test]
fn refuses_mutual_recursion() {
    // A calls B and B calls A: inlining either direction would copy
    // a call back into its own caller, so both must be refused.
    let mut module = Module::new();
    let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
    let (mut a, a_entry) = Function::new("a", sig.clone());
    a.inline = true;
    let a_arg = a.values.push(crate::ir::ValueData {
        ty: Ty::int(32),
        def: Def::Param(a_entry, 0),
    });
    a.blocks[a_entry].params.push(a_arg);
    let a_id = module.push_func(a);
    let (mut b, b_entry) = Function::new("b", sig);
    b.inline = true;
    let b_arg = b.values.push(crate::ir::ValueData {
        ty: Ty::int(32),
        def: Def::Param(b_entry, 0),
    });
    b.blocks[b_entry].params.push(b_arg);
    let b_id = module.push_func(b);
    // Back-patch the cross-calls: `a(x) = b(x)`, `b(x) = a(x)`.
    for (self_id, entry, arg, other) in [(a_id, a_entry, a_arg, b_id), (b_id, b_entry, b_arg, a_id)] {
        let func = &mut module.funcs[self_id];
        let rv = func.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
        });
        let call = func.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Call {
                callee: other,
                args: [arg].into_iter().collect(),
            },
            result: Some(rv),
            loc: None,
        });
        func.values[rv].def = Def::Inst(call);
        func.blocks[entry].insts.push(call);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(rv) });
    }
    let mut inline = Inline::small();
    assert!(!inline.run(&mut module));
    assert!(caller_still_calls(&module, a_id));
    assert!(caller_still_calls(&module, b_id));
}

fn emit_alloca(func: &mut Function, block: crate::ir::Block, pointee: Ty) -> Value {
    let slot = func.values.push(crate::ir::ValueData {
        ty: Ty::ptr(pointee.clone()),
        def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
    });
    let inst = func.insts.push(crate::ir::InstData {
        block,
        kind: InstKind::Alloca { pointee, count: 1 },
        result: Some(slot),
        loc: None,
    });
    func.values[slot].def = Def::Inst(inst);
    func.blocks[block].insts.push(inst);
    slot
}

fn emit_store(func: &mut Function, block: crate::ir::Block, addr: Value, val: Value, vol: bool) -> crate::ir::Inst {
    let flags = if vol { crate::ir::MemFlags::vol() } else { crate::ir::MemFlags::plain() };
    let inst = func.insts.push(crate::ir::InstData {
        block,
        kind: InstKind::Store { addr, val, flags },
        result: None,
        loc: None,
    });
    func.blocks[block].insts.push(inst);
    inst
}

fn emit_load(func: &mut Function, block: crate::ir::Block, addr: Value, ty: Ty, vol: bool) -> Value {
    let flags = if vol { crate::ir::MemFlags::vol() } else { crate::ir::MemFlags::plain() };
    let out = func.values.push(crate::ir::ValueData {
        ty,
        def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
    });
    let inst = func.insts.push(crate::ir::InstData {
        block,
        kind: InstKind::Load { addr, flags },
        result: Some(out),
        loc: None,
    });
    func.values[out].def = Def::Inst(inst);
    func.blocks[block].insts.push(inst);
    out
}

fn resident_kinds(func: &Function) -> Vec<InstKind> {
    func.blocks.iter().flat_map(|b| b.insts.iter()).map(|i| func.insts[*i].kind.clone()).collect()
}

fn param(func: &mut Function, block: crate::ir::Block, ty: Ty) -> Value {
    let index = func.blocks[block].params.len() as u32;
    let value = func.values.push(crate::ir::ValueData {
        ty,
        def: Def::Param(block, index),
    });
    func.blocks[block].params.push(value);
    value
}

fn has_memory_ops(func: &Function) -> bool {
    resident_kinds(func)
        .iter()
        .any(|kind| matches!(kind, InstKind::Alloca { .. } | InstKind::Load { .. } | InstKind::Store { .. }))
}

fn br_args(func: &Function, block: crate::ir::Block) -> Vec<Value> {
    match func.blocks[block].term.clone() {
        Some(Terminator::Br { args, .. }) => args.to_vec(),
        other => panic!("expected br, got {other:?}"),
    }
}

fn emit_bin(func: &mut Function, block: crate::ir::Block, kind: InstKind, ty: Ty) -> Value {
    let v = func.values.push(crate::ir::ValueData {
        ty,
        def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
    });
    let inst = func.insts.push(crate::ir::InstData {
        block,
        kind,
        result: Some(v),
        loc: None,
    });
    func.values[v].def = Def::Inst(inst);
    func.blocks[block].insts.push(inst);
    v
}

#[test]
fn ashr_sign_extends_wide_literals() {
    // The frontend pushes `0x8000_0000_0000_0000` as a positive
    // `i128`; as a 64-bit pattern it is the minimum i64, so the
    // shift must treat it as negative.
    let sig = Signature::new(vec![], Ty::int(64));
    let (mut func, entry) = Function::new("ashr", sig);
    let min = func.push_const(Const::Int { bits: 64, value: 1i128 << 63 });
    let shift = func.push_const(Const::Int { bits: 64, value: 63 });
    let ashr = emit_bin(&mut func, entry, InstKind::AShr(min, shift), Ty::int(64));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(ashr) });
    let mut fold = ConstFold;
    fold.run(&mut func);
    let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
        unreachable!()
    };
    assert_eq!(func.const_value(val.unwrap()), Some(&Const::Int { bits: 64, value: -1 }));
}

#[test]
fn folds_shifts_neg_not() {
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("shifts", sig);
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let three = func.push_const(Const::Int { bits: 32, value: 3 });
    let two = func.push_const(Const::Int { bits: 32, value: 2 });
    let sixteen = func.push_const(Const::Int { bits: 32, value: 16 });
    let neg8 = func.push_const(Const::Int { bits: 32, value: -8 });
    let shl = emit_bin(&mut func, entry, InstKind::Shl(one, three), Ty::int(32));
    let lshr = emit_bin(&mut func, entry, InstKind::LShr(sixteen, two), Ty::int(32));
    let ashr = emit_bin(&mut func, entry, InstKind::AShr(neg8, two), Ty::int(32));
    let sum1 = emit_bin(&mut func, entry, InstKind::Add(shl, lshr), Ty::int(32));
    let sum = emit_bin(&mut func, entry, InstKind::Add(sum1, ashr), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
    let mut fold = ConstFold;
    fold.run(&mut func);
    // 8 + 4 + (-2) = 10.
    let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
        unreachable!()
    };
    assert_eq!(func.const_value(val.unwrap()), Some(&Const::Int { bits: 32, value: 10 }));

    // Negation and complement fold with wraparound.
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("unary", sig);
    let five = func.push_const(Const::Int { bits: 32, value: 5 });
    let zero = func.push_const(Const::Int { bits: 32, value: 0 });
    let neg = emit_bin(&mut func, entry, InstKind::Neg(five), Ty::int(32));
    let not = emit_bin(&mut func, entry, InstKind::Not(zero), Ty::int(32));
    let sum = emit_bin(&mut func, entry, InstKind::Add(neg, not), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
    fold.run(&mut func);
    // -5 + (-1) = -6.
    let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
        unreachable!()
    };
    assert_eq!(func.const_value(val.unwrap()), Some(&Const::Int { bits: 32, value: -6 }));
}

#[test]
fn refuses_trapping_folds() {
    // Division by zero and oversized shifts keep their
    // instructions: folding them would delete the trap.
    let sig = Signature::new(vec![], Ty::Void);
    let (mut func, entry) = Function::new("traps", sig);
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let zero = func.push_const(Const::Int { bits: 32, value: 0 });
    let wide = func.push_const(Const::Int { bits: 32, value: 32 });
    for kind in [
        InstKind::Sdiv(one, zero),
        InstKind::Udiv(one, zero),
        InstKind::Srem(one, zero),
        InstKind::Urem(one, zero),
        InstKind::Shl(one, wide),
        InstKind::LShr(one, wide),
        InstKind::AShr(one, wide),
    ] {
        let v = func.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(crate::ir::InstData {
            block: entry,
            kind,
            result: Some(v),
            loc: None,
        });
        func.values[v].def = Def::Inst(inst);
        func.blocks[entry].insts.push(inst);
    }
    func.blocks[entry].term = Some(Terminator::Return { val: None });
    let mut fold = ConstFold;
    assert!(!fold.run(&mut func));
    assert_eq!(func.blocks[entry].insts.len(), 7);
}

#[test]
fn folds_div_rem() {
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("div", sig);
    let n7 = func.push_const(Const::Int { bits: 32, value: -7 });
    let two = func.push_const(Const::Int { bits: 32, value: 2 });
    let div = emit_bin(&mut func, entry, InstKind::Sdiv(n7, two), Ty::int(32));
    let rem = emit_bin(&mut func, entry, InstKind::Srem(n7, two), Ty::int(32));
    let sum = emit_bin(&mut func, entry, InstKind::Add(div, rem), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
    let mut fold = ConstFold;
    fold.run(&mut func);
    // -7 / 2 = -3 (truncated), -7 % 2 = -1.
    let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
        unreachable!()
    };
    assert_eq!(func.const_value(val.unwrap()), Some(&Const::Int { bits: 32, value: -4 }));
}

#[test]
fn folds_int_conversions() {
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("convs", sig);
    // 0x1FF truncates to -1 in 8 bits; -1 zero-extends to 255.
    let wide = func.push_const(Const::Int { bits: 32, value: 0x1FF });
    let neg1_8 = func.push_const(Const::Int { bits: 8, value: -1 });
    let trunc = {
        let v = func.values.push(crate::ir::ValueData {
            ty: Ty::int(8),
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Trunc(wide),
            result: Some(v),
            loc: None,
        });
        func.values[v].def = Def::Inst(inst);
        func.blocks[entry].insts.push(inst);
        v
    };
    let zext = {
        let v = func.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Zext(neg1_8),
            result: Some(v),
            loc: None,
        });
        func.values[v].def = Def::Inst(inst);
        func.blocks[entry].insts.push(inst);
        v
    };
    let sext = {
        let v = func.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Sext(neg1_8),
            result: Some(v),
            loc: None,
        });
        func.values[v].def = Def::Inst(inst);
        func.blocks[entry].insts.push(inst);
        v
    };
    // (trunc as i32) + zext + sext = -1 + 255 + -1 = 253.
    let t32 = {
        let v = func.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Sext(trunc),
            result: Some(v),
            loc: None,
        });
        func.values[v].def = Def::Inst(inst);
        func.blocks[entry].insts.push(inst);
        v
    };
    let mut b = FunctionBuilder::new(&mut func, entry);
    let s1 = b.add(Ty::int(32), t32, zext);
    let sum = b.add(Ty::int(32), s1, sext);
    b.ret(Some(sum));
    let mut fold = ConstFold;
    fold.run(&mut func);
    let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
        unreachable!()
    };
    assert_eq!(func.const_value(val.unwrap()), Some(&Const::Int { bits: 32, value: 253 }));
}

#[test]
fn folds_switch_on_constant() {
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("switch", sig);
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let b0 = func.push_block(vec![]);
    let b1 = func.push_block(vec![]);
    let bdef = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::Switch {
        scrutinee: one,
        cases: vec![(0, b0), (1, b1)],
        default: bdef,
    });
    for (block, val) in [(b0, 10), (b1, 20), (bdef, 30)] {
        let c = func.push_const(Const::Int { bits: 32, value: val });
        func.blocks[block].term = Some(Terminator::Return { val: Some(c) });
    }
    let mut fold = ConstFold;
    assert!(fold.run(&mut func));
    assert!(matches!(
        func.blocks[entry].term,
        Some(Terminator::Br { dest, .. }) if dest == b1
    ));
}

#[test]
fn folds_float_conversions() {
    use crate::ty::FloatKind;
    #[derive(Clone, Copy)]
    enum Conv {
        SItoF,
        UItoF,
        SFtoI,
        UFtoI,
        Fptrunc,
        Fpext,
    }
    let cases = [
        (
            Const::Int { bits: 32, value: -3 },
            Conv::SItoF,
            Ty::Float(FloatKind::F64),
            Const::Float {
                kind: FloatKind::F64,
                bits: (-3.0f64).to_bits() as u128,
            },
        ),
        (
            Const::Int { bits: 32, value: -1 },
            Conv::UItoF,
            Ty::Float(FloatKind::F64),
            Const::Float {
                kind: FloatKind::F64,
                bits: (u32::MAX as f64).to_bits() as u128,
            },
        ),
        (
            Const::Float {
                kind: FloatKind::F64,
                bits: 2.9f64.to_bits() as u128,
            },
            Conv::SFtoI,
            Ty::int(32),
            Const::Int { bits: 32, value: 2 },
        ),
        (
            Const::Float {
                kind: FloatKind::F64,
                bits: (-2.9f64).to_bits() as u128,
            },
            Conv::SFtoI,
            Ty::int(32),
            Const::Int { bits: 32, value: -2 },
        ),
        (
            Const::Float {
                kind: FloatKind::F64,
                bits: 3.9f64.to_bits() as u128,
            },
            Conv::UFtoI,
            Ty::int(32),
            Const::Int { bits: 32, value: 3 },
        ),
        (
            Const::Float {
                kind: FloatKind::F64,
                bits: 1.5f64.to_bits() as u128,
            },
            Conv::Fptrunc,
            Ty::Float(FloatKind::F32),
            Const::Float {
                kind: FloatKind::F32,
                bits: 1.5f32.to_bits() as u128,
            },
        ),
        (
            Const::Float {
                kind: FloatKind::F32,
                bits: (-1.5f32).to_bits() as u128,
            },
            Conv::Fpext,
            Ty::Float(FloatKind::F64),
            Const::Float {
                kind: FloatKind::F64,
                bits: (-1.5f64).to_bits() as u128,
            },
        ),
    ];
    for (input, conv, result_ty, expected) in cases {
        let (mut func, entry) = Function::new("conv", Signature::new(vec![], result_ty.clone()));
        let input = func.push_const(input);
        let kind = match conv {
            Conv::SItoF => InstKind::SItoF(input),
            Conv::UItoF => InstKind::UItoF(input),
            Conv::SFtoI => InstKind::SFtoI(input),
            Conv::UFtoI => InstKind::UFtoI(input),
            Conv::Fptrunc => InstKind::Fptrunc(input),
            Conv::Fpext => InstKind::Fpext(input),
        };
        let result = emit_bin(&mut func, entry, kind, result_ty);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(result) });
        let mut fold = ConstFold;
        assert!(fold.run(&mut func), "conversion did not fold");
        let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
            unreachable!()
        };
        assert_eq!(func.const_value(val.unwrap()), Some(&expected));
    }
}

#[test]
fn folds_float_arithmetic() {
    use crate::ty::FloatKind;
    #[derive(Clone, Copy)]
    enum Arith {
        Add,
        Sub,
        Mul,
        Div,
    }
    let cases = [
        (FloatKind::F64, 1.5f64.to_bits() as u128, 2.25f64.to_bits() as u128, Arith::Add, 3.75f64.to_bits() as u128),
        (FloatKind::F64, 5.0f64.to_bits() as u128, 1.25f64.to_bits() as u128, Arith::Sub, 3.75f64.to_bits() as u128),
        (FloatKind::F64, 1.5f64.to_bits() as u128, 4.0f64.to_bits() as u128, Arith::Mul, 6.0f64.to_bits() as u128),
        (FloatKind::F64, 1.5f64.to_bits() as u128, 2.0f64.to_bits() as u128, Arith::Div, 0.75f64.to_bits() as u128),
        (FloatKind::F32, 1.5f32.to_bits() as u128, 2.25f32.to_bits() as u128, Arith::Add, 3.75f32.to_bits() as u128),
        (
            FloatKind::F32,
            1.0f32.to_bits() as u128,
            0.0f32.to_bits() as u128,
            Arith::Div,
            f32::INFINITY.to_bits() as u128,
        ),
    ];
    for (kind, left, right, op, expected) in cases {
        let ty = Ty::Float(kind);
        let (mut func, entry) = Function::new("float", Signature::new(vec![], ty.clone()));
        let l = func.push_const(Const::Float { kind, bits: left });
        let r = func.push_const(Const::Float { kind, bits: right });
        let arith = match op {
            Arith::Add => InstKind::FAdd(l, r),
            Arith::Sub => InstKind::FSub(l, r),
            Arith::Mul => InstKind::FMul(l, r),
            Arith::Div => InstKind::FDiv(l, r),
        };
        let result = emit_bin(&mut func, entry, arith, ty);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(result) });
        let mut fold = ConstFold;
        assert!(fold.run(&mut func), "float arithmetic did not fold");
        let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
            unreachable!()
        };
        let expected = Const::Float { kind, bits: expected };
        assert_eq!(func.const_value(val.unwrap()), Some(&expected));
    }
}

#[test]
fn keeps_undefined_float_to_int_conversions() {
    use crate::ty::FloatKind;
    let cases = [
        (f64::NAN, false, Ty::int(32)),
        (f64::INFINITY, false, Ty::int(32)),
        (1e300, false, Ty::int(32)),
        (-1.0, true, Ty::int(32)),
        (4294967296.0, true, Ty::int(32)),
    ];
    for (value, unsigned, result_ty) in cases {
        let (mut func, entry) = Function::new("conv", Signature::new(vec![], result_ty.clone()));
        let input = func.push_const(Const::Float {
            kind: FloatKind::F64,
            bits: value.to_bits() as u128,
        });
        let kind = if unsigned { InstKind::UFtoI(input) } else { InstKind::SFtoI(input) };
        let result = emit_bin(&mut func, entry, kind, result_ty);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(result) });
        let mut fold = ConstFold;
        assert!(!fold.run(&mut func), "undefined conversion {value} was folded");
        assert_eq!(func.const_value(result), None);
    }
}

#[test]
fn folds_fneg_and_fcmp() {
    use crate::ty::FloatKind;
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("floats", sig);
    // -(-1.5f) == 1.5f: exact bitwise negation, then an
    // equality fold over the two constants.
    let neg_in = func.push_const(Const::Float {
        kind: FloatKind::F32,
        bits: 0xBFC0_0000,
    });
    let pos = func.push_const(Const::Float {
        kind: FloatKind::F32,
        bits: 0x3FC0_0000,
    });
    let neg = {
        let v = func.values.push(crate::ir::ValueData {
            ty: Ty::Float(FloatKind::F32),
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::FNeg(neg_in),
            result: Some(v),
            loc: None,
        });
        func.values[v].def = Def::Inst(inst);
        func.blocks[entry].insts.push(inst);
        v
    };
    // NaN != 1.5 and NaN is unordered: both fold to 1.
    let nan = func.push_const(Const::Float {
        kind: FloatKind::F64,
        bits: 0x7FF8_0000_0000_0000,
    });
    let one5 = func.push_const(Const::Float {
        kind: FloatKind::F64,
        bits: 0x3FF8_0000_0000_0000,
    });
    let mut flags = Vec::new();
    for (pred, l, r) in [
        (crate::ir::FcmpPred::Eq, neg, pos),
        (crate::ir::FcmpPred::Ne, nan, one5),
        (crate::ir::FcmpPred::Uno, nan, one5),
    ] {
        let v = func.values.push(crate::ir::ValueData {
            ty: Ty::Bool,
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Fcmp { pred, lhs: l, rhs: r },
            result: Some(v),
            loc: None,
        });
        func.values[v].def = Def::Inst(inst);
        func.blocks[entry].insts.push(inst);
        let w = func.values.push(crate::ir::ValueData {
            ty: Ty::int(32),
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
        });
        let wi = func.insts.push(crate::ir::InstData {
            block: entry,
            kind: InstKind::Zext(v),
            result: Some(w),
            loc: None,
        });
        func.values[w].def = Def::Inst(wi);
        func.blocks[entry].insts.push(wi);
        flags.push(w);
    }
    let mut b = FunctionBuilder::new(&mut func, entry);
    let s1 = b.add(Ty::int(32), flags[0], flags[1]);
    let sum = b.add(Ty::int(32), s1, flags[2]);
    b.ret(Some(sum));
    let mut fold = ConstFold;
    fold.run(&mut func);
    // 1 + 1 + 1.
    let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
        unreachable!()
    };
    assert_eq!(func.const_value(val.unwrap()), Some(&Const::Int { bits: 32, value: 3 }));
}

#[test]
fn mem2reg_promotes_entry_spill() {
    // `f(x) = x`: entry spills the param, the body loads it.
    let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("id", sig);
    let arg = func.values.push(crate::ir::ValueData {
        ty: Ty::int(32),
        def: Def::Param(entry, 0),
    });
    func.blocks[entry].params.push(arg);
    let slot = emit_alloca(&mut func, entry, Ty::int(32));
    emit_store(&mut func, entry, slot, arg, false);
    let body = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::Br {
        dest: body,
        args: Default::default(),
    });
    let loaded = emit_load(&mut func, body, slot, Ty::int(32), false);
    func.blocks[body].term = Some(Terminator::Return { val: Some(loaded) });
    let mut pass = Mem2Reg;
    assert!(pass.run(&mut func));
    assert!(crate::verify::verify_func(&func).is_empty());
    // No memory traffic left: the return reads the parameter.
    for kind in resident_kinds(&func) {
        assert!(
            !matches!(kind, InstKind::Alloca { .. } | InstKind::Load { .. } | InstKind::Store { .. }),
            "memory op survived: {kind:?}"
        );
    }
    let Terminator::Return { val } = func.blocks[body].term.clone().unwrap() else {
        unreachable!()
    };
    assert_eq!(val.unwrap(), arg);
}

#[test]
fn mem2reg_keeps_slots_in_setjmp_functions() {
    // A `setjmp` call site re-enters with the registers of the
    // first return, so a slot written after it must stay memory.
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("jmp", sig);
    let slot = emit_alloca(&mut func, entry, Ty::int(32));
    let seven = func.push_const(Const::Int { bits: 32, value: 7 });
    emit_store(&mut func, entry, slot, seven, false);
    let loaded = emit_load(&mut func, entry, slot, Ty::int(32), false);
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });
    func.returns_twice = true;
    let mut pass = Mem2Reg;
    assert!(!pass.run(&mut func), "slots promoted in a setjmp function");
    assert!(crate::verify::verify_func(&func).is_empty());
}

#[test]
fn mem2reg_single_block_chain() {
    // store 1; store 2; load: forwards 2 and kills the first
    // store, then the slot and the second store as dead.
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("chain", sig);
    let slot = emit_alloca(&mut func, entry, Ty::int(32));
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let two = func.push_const(Const::Int { bits: 32, value: 2 });
    emit_store(&mut func, entry, slot, one, false);
    emit_store(&mut func, entry, slot, two, false);
    let loaded = emit_load(&mut func, entry, slot, Ty::int(32), false);
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });
    let mut pass = Mem2Reg;
    assert!(pass.run(&mut func));
    assert!(crate::verify::verify_func(&func).is_empty());
    for kind in resident_kinds(&func) {
        assert!(
            !matches!(kind, InstKind::Alloca { .. } | InstKind::Load { .. } | InstKind::Store { .. }),
            "memory op survived: {kind:?}"
        );
    }
    let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
        unreachable!()
    };
    assert_eq!(func.const_value(val.unwrap()), Some(&Const::Int { bits: 32, value: 2 }));
}

#[test]
fn mem2reg_refuses_escaped_slot() {
    // The slot address feeds a GEP: untracked uses stay in memory.
    use crate::ty::GepIndex;
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("esc", sig);
    let slot = emit_alloca(&mut func, entry, Ty::int(32));
    let seven = func.push_const(Const::Int { bits: 32, value: 7 });
    emit_store(&mut func, entry, slot, seven, false);
    let gep = func.values.push(crate::ir::ValueData {
        ty: Ty::ptr(Ty::int(32)),
        def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
    });
    let gi = func.insts.push(crate::ir::InstData {
        block: entry,
        kind: InstKind::GetElementPtr {
            base: slot,
            indices: [GepIndex::ConstIndex(0)].into_iter().collect(),
        },
        result: Some(gep),
        loc: None,
    });
    func.values[gep].def = Def::Inst(gi);
    func.blocks[entry].insts.push(gi);
    let loaded = emit_load(&mut func, entry, gep, Ty::int(32), false);
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });
    let mut pass = Mem2Reg;
    assert!(!pass.run(&mut func));
    assert!(
        resident_kinds(&func)
            .iter()
            .any(|k| matches!(k, InstKind::Load { .. } | InstKind::Store { .. } | InstKind::Alloca { .. }))
    );
}

#[test]
fn mem2reg_refuses_volatile_slot() {
    // Volatile accesses are observable: the slot stays in memory.
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("vol", sig);
    let slot = emit_alloca(&mut func, entry, Ty::int(32));
    let seven = func.push_const(Const::Int { bits: 32, value: 7 });
    emit_store(&mut func, entry, slot, seven, true);
    let loaded = emit_load(&mut func, entry, slot, Ty::int(32), true);
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });
    let mut pass = Mem2Reg;
    assert!(!pass.run(&mut func));
    assert!(
        resident_kinds(&func)
            .iter()
            .any(|k| matches!(k, InstKind::Load { .. } | InstKind::Store { .. } | InstKind::Alloca { .. }))
    );
}

#[test]
fn escaping_addresses_marks_non_memory_uses() {
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("esc2", sig);
    let slot = emit_alloca(&mut func, entry, Ty::int(32));
    let seven = func.push_const(Const::Int { bits: 32, value: 7 });
    emit_store(&mut func, entry, slot, seven, false);
    let loaded = emit_load(&mut func, entry, slot, Ty::int(32), false);
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });
    let escaped = escaping_addresses(&func);
    assert!(!escaped[slot.index()]);
    assert!(escaped[loaded.index()]);
}

#[test]
fn orphaned_address_use_does_not_block_mem2reg() {
    let (mut func, entry) = Function::new("m2r", Signature::new(vec![], Ty::int(32)));
    let slot = emit_alloca(&mut func, entry, Ty::int(32));
    let call = push_unit(
        &mut func,
        entry,
        InstKind::CallExtern {
            target: crate::ir::Extern::from_usize(0),
            args: [slot].into_iter().collect(),
        },
    );
    func.blocks[entry].insts.retain(|inst| *inst != call);

    let seven = func.push_const(Const::Int { bits: 32, value: 7 });
    emit_store(&mut func, entry, slot, seven, false);
    let loaded = emit_load(&mut func, entry, slot, Ty::int(32), false);
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    let mut pass = Mem2Reg;
    assert!(pass.run(&mut func), "the dead call no longer makes the slot escape");
    assert!(!has_memory_ops(&func));
    assert!(crate::verify::verify_func(&func).is_empty());
}

#[test]
fn aggregates_express_alloca_gep_store_load() {
    use crate::ty::GepIndex;
    use crate::verify::verify_func;
    let sig = Signature::new(vec![], Ty::Void);
    let (mut func, entry) = Function::new("agg", sig);
    let point = Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(32)],
        offsets: vec![0, 4],
        size: 8,
    };
    let slot = func.values.push(crate::ir::ValueData {
        ty: Ty::ptr(point.clone()),
        def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
    });
    let alloca = func.insts.push(crate::ir::InstData {
        block: entry,
        kind: InstKind::Alloca { pointee: point, count: 1 },
        result: Some(slot),
        loc: None,
    });
    func.values[slot].def = Def::Inst(alloca);
    func.blocks[entry].insts.push(alloca);
    let field_ptr = func.values.push(crate::ir::ValueData {
        ty: Ty::ptr(Ty::int(32)),
        def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
    });
    let gep = func.insts.push(crate::ir::InstData {
        block: entry,
        kind: InstKind::GetElementPtr {
            base: slot,
            indices: [GepIndex::Field(1)].into_iter().collect(),
        },
        result: Some(field_ptr),
        loc: None,
    });
    func.values[field_ptr].def = Def::Inst(gep);
    func.blocks[entry].insts.push(gep);
    let seven = func.push_const(Const::Int { bits: 32, value: 7 });
    let store = func.insts.push(crate::ir::InstData {
        block: entry,
        kind: InstKind::Store {
            addr: field_ptr,
            val: seven,
            flags: crate::ir::MemFlags::plain(),
        },
        result: None,
        loc: None,
    });
    func.blocks[entry].insts.push(store);
    let loaded = func.values.push(crate::ir::ValueData {
        ty: Ty::int(32),
        def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
    });
    let load = func.insts.push(crate::ir::InstData {
        block: entry,
        kind: InstKind::Load {
            addr: field_ptr,
            flags: crate::ir::MemFlags::plain(),
        },
        result: Some(loaded),
        loc: None,
    });
    func.values[loaded].def = Def::Inst(load);
    func.blocks[entry].insts.push(load);
    func.blocks[entry].term = Some(Terminator::Return { val: None });
    assert!(verify_func(&func).is_empty());
}

#[test]
fn mem2reg_places_phi_at_branch_merge() {
    // if (c) x = 1; else x = 2; return x;
    let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("diamond", sig);
    let cond = param(&mut func, entry, Ty::Bool);
    let slot = emit_alloca(&mut func, entry, Ty::int(32));
    let then_block = func.push_block(vec![]);
    let else_block = func.push_block(vec![]);
    let merge = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block,
        else_block,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let two = func.push_const(Const::Int { bits: 32, value: 2 });
    emit_store(&mut func, then_block, slot, one, false);
    func.blocks[then_block].term = Some(Terminator::Br {
        dest: merge,
        args: Default::default(),
    });
    emit_store(&mut func, else_block, slot, two, false);
    func.blocks[else_block].term = Some(Terminator::Br {
        dest: merge,
        args: Default::default(),
    });
    let loaded = emit_load(&mut func, merge, slot, Ty::int(32), false);
    func.blocks[merge].term = Some(Terminator::Return { val: Some(loaded) });
    assert!(crate::verify::verify_func(&func).is_empty());

    let mut pass = Mem2Reg;
    assert!(pass.run(&mut func));
    assert!(!has_memory_ops(&func));
    assert!(crate::verify::verify_func(&func).is_empty());
    // Exactly one phi at the merge, fed by both edges.
    let params = func.blocks[merge].params.clone();
    assert_eq!(params.len(), 1);
    assert_eq!(br_args(&func, then_block), vec![one]);
    assert_eq!(br_args(&func, else_block), vec![two]);
    let Terminator::Return { val } = func.blocks[merge].term.clone().unwrap() else {
        unreachable!()
    };
    assert_eq!(val.unwrap(), params[0]);
    // Already promoted: nothing left to do.
    assert!(!pass.run(&mut func));
}

#[test]
fn mem2reg_carries_value_around_a_loop() {
    // x = 0; while (n > 0) { x = x + 1; n = n - 1; } return x;
    let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("loop", sig);
    let n = param(&mut func, entry, Ty::int(32));
    let x_slot = emit_alloca(&mut func, entry, Ty::int(32));
    let n_slot = emit_alloca(&mut func, entry, Ty::int(32));
    let header = func.push_block(vec![]);
    let body = func.push_block(vec![]);
    let exit = func.push_block(vec![]);
    let zero = func.push_const(Const::Int { bits: 32, value: 0 });
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    emit_store(&mut func, entry, x_slot, zero, false);
    emit_store(&mut func, entry, n_slot, n, false);
    func.blocks[entry].term = Some(Terminator::Br {
        dest: header,
        args: Default::default(),
    });
    let nv = emit_load(&mut func, header, n_slot, Ty::int(32), false);
    let pos = emit_bin(
        &mut func,
        header,
        InstKind::Icmp {
            pred: IcmpPred::Sgt,
            lhs: nv,
            rhs: zero,
        },
        Ty::Bool,
    );
    func.blocks[header].term = Some(Terminator::CondBr {
        cond: pos,
        then_block: body,
        else_block: exit,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    let xv = emit_load(&mut func, body, x_slot, Ty::int(32), false);
    let x1 = emit_bin(&mut func, body, InstKind::Add(xv, one), Ty::int(32));
    emit_store(&mut func, body, x_slot, x1, false);
    let nv2 = emit_load(&mut func, body, n_slot, Ty::int(32), false);
    let n1 = emit_bin(&mut func, body, InstKind::Sub(nv2, one), Ty::int(32));
    emit_store(&mut func, body, n_slot, n1, false);
    func.blocks[body].term = Some(Terminator::Br {
        dest: header,
        args: Default::default(),
    });
    let result = emit_load(&mut func, exit, x_slot, Ty::int(32), false);
    func.blocks[exit].term = Some(Terminator::Return { val: Some(result) });
    assert!(crate::verify::verify_func(&func).is_empty());

    let mut pass = Mem2Reg;
    assert!(pass.run(&mut func));
    assert!(!has_memory_ops(&func));
    assert!(crate::verify::verify_func(&func).is_empty());
    // Header phis for both slots, in allocation order: x, then n.
    let params = func.blocks[header].params.clone();
    assert_eq!(params.len(), 2);
    assert_eq!(br_args(&func, entry), vec![zero, n]);
    assert_eq!(br_args(&func, body), vec![x1, n1]);
    let Terminator::Return { val } = func.blocks[exit].term.clone().unwrap() else {
        unreachable!()
    };
    assert_eq!(val.unwrap(), params[0]);
}

#[test]
fn mem2reg_chains_phis_across_nested_merges() {
    // x = 0; if (a) x = 1; if (b) x = 2; return x;
    let sig = Signature::new(vec![Ty::Bool, Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("nested", sig);
    let a = param(&mut func, entry, Ty::Bool);
    let b_cond = param(&mut func, entry, Ty::Bool);
    let slot = emit_alloca(&mut func, entry, Ty::int(32));
    let then_a = func.push_block(vec![]);
    let merge1 = func.push_block(vec![]);
    let then_b = func.push_block(vec![]);
    let merge2 = func.push_block(vec![]);
    let zero = func.push_const(Const::Int { bits: 32, value: 0 });
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let two = func.push_const(Const::Int { bits: 32, value: 2 });
    emit_store(&mut func, entry, slot, zero, false);
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond: a,
        then_block: then_a,
        else_block: merge1,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    emit_store(&mut func, then_a, slot, one, false);
    func.blocks[then_a].term = Some(Terminator::Br {
        dest: merge1,
        args: Default::default(),
    });
    func.blocks[merge1].term = Some(Terminator::CondBr {
        cond: b_cond,
        then_block: then_b,
        else_block: merge2,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    emit_store(&mut func, then_b, slot, two, false);
    func.blocks[then_b].term = Some(Terminator::Br {
        dest: merge2,
        args: Default::default(),
    });
    let loaded = emit_load(&mut func, merge2, slot, Ty::int(32), false);
    func.blocks[merge2].term = Some(Terminator::Return { val: Some(loaded) });

    let mut pass = Mem2Reg;
    assert!(pass.run(&mut func));
    assert!(!has_memory_ops(&func));
    assert!(crate::verify::verify_func(&func).is_empty());
    let params1 = func.blocks[merge1].params.clone();
    let params2 = func.blocks[merge2].params.clone();
    assert_eq!(params1.len(), 1);
    assert_eq!(params2.len(), 1);
    // merge2's phi takes merge1's phi on the fallthrough edge and
    // the fresh constant on the other.
    let merge1_else = match func.blocks[merge1].term.clone().unwrap() {
        Terminator::CondBr { else_args, .. } => else_args.to_vec(),
        other => panic!("expected condbr, got {other:?}"),
    };
    assert_eq!(merge1_else, vec![params1[0]]);
    assert_eq!(br_args(&func, then_b), vec![two]);
    let Terminator::Return { val } = func.blocks[merge2].term.clone().unwrap() else {
        unreachable!()
    };
    assert_eq!(val.unwrap(), params2[0]);
}

#[test]
fn mem2reg_reads_undef_before_the_first_store() {
    // if (c) x = 1; return x; — the else edge has no definition.
    let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("maybe", sig);
    let cond = param(&mut func, entry, Ty::Bool);
    let slot = emit_alloca(&mut func, entry, Ty::int(32));
    let then_block = func.push_block(vec![]);
    let merge = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block,
        else_block: merge,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    emit_store(&mut func, then_block, slot, one, false);
    func.blocks[then_block].term = Some(Terminator::Br {
        dest: merge,
        args: Default::default(),
    });
    let loaded = emit_load(&mut func, merge, slot, Ty::int(32), false);
    func.blocks[merge].term = Some(Terminator::Return { val: Some(loaded) });

    let mut pass = Mem2Reg;
    assert!(pass.run(&mut func));
    assert!(!has_memory_ops(&func));
    assert!(crate::verify::verify_func(&func).is_empty());
    assert_eq!(func.blocks[merge].params.len(), 1);
    assert_eq!(br_args(&func, then_block), vec![one]);
    let else_args = match func.blocks[entry].term.clone().unwrap() {
        Terminator::CondBr { else_args, .. } => else_args.to_vec(),
        other => panic!("expected condbr, got {other:?}"),
    };
    assert_eq!(else_args.len(), 1);
    assert_eq!(func.const_value(else_args[0]), Some(&Const::Undef));
}

#[test]
fn mem2reg_collapses_replacement_chains() {
    // store 1; v1 = load; store v1; v2 = load; return v2
    // v2 is replaced by v1, which itself resolves to 1.
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("chain", sig);
    let slot = emit_alloca(&mut func, entry, Ty::int(32));
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    emit_store(&mut func, entry, slot, one, false);
    let v1 = emit_load(&mut func, entry, slot, Ty::int(32), false);
    emit_store(&mut func, entry, slot, v1, false);
    let v2 = emit_load(&mut func, entry, slot, Ty::int(32), false);
    func.blocks[entry].term = Some(Terminator::Return { val: Some(v2) });

    let mut pass = Mem2Reg;
    assert!(pass.run(&mut func));
    assert!(!has_memory_ops(&func));
    let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
        unreachable!()
    };
    assert_eq!(func.const_value(val.unwrap()), Some(&Const::Int { bits: 32, value: 1 }));
}

#[test]
fn mem2reg_forwards_across_arena_order() {
    // `mid` stores after the block that loads it in arena order;
    // only dominance matters (the Clif lowering fills dominators
    // first), so the load must be replaced with `mid`'s value.
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("forward", sig);
    let slot = emit_alloca(&mut func, entry, Ty::int(32));
    let late = func.push_block(vec![]);
    let mid = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::Br {
        dest: mid,
        args: Default::default(),
    });
    let three = func.push_const(Const::Int { bits: 32, value: 3 });
    let four = func.push_const(Const::Int { bits: 32, value: 4 });
    let seven = emit_bin(&mut func, mid, InstKind::Add(three, four), Ty::int(32));
    emit_store(&mut func, mid, slot, seven, false);
    func.blocks[mid].term = Some(Terminator::Br {
        dest: late,
        args: Default::default(),
    });
    let loaded = emit_load(&mut func, late, slot, Ty::int(32), false);
    func.blocks[late].term = Some(Terminator::Return { val: Some(loaded) });

    let mut pass = Mem2Reg;
    assert!(pass.run(&mut func));
    assert!(!has_memory_ops(&func));
    assert!(crate::verify::verify_func(&func).is_empty());
    let Terminator::Return { val } = func.blocks[late].term.clone().unwrap() else {
        unreachable!()
    };
    assert_eq!(val.unwrap(), seven);
}

#[test]
fn mem2reg_feeds_unreachable_edges_placeholders() {
    // A dead block still branches to the phi block: its edge keeps
    // arity with an undef placeholder.
    let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("dead-edge", sig);
    let cond = param(&mut func, entry, Ty::Bool);
    let slot = emit_alloca(&mut func, entry, Ty::int(32));
    let then_block = func.push_block(vec![]);
    let merge = func.push_block(vec![]);
    let dead = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::CondBr {
        cond,
        then_block,
        else_block: merge,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    emit_store(&mut func, then_block, slot, one, false);
    func.blocks[then_block].term = Some(Terminator::Br {
        dest: merge,
        args: Default::default(),
    });
    let loaded = emit_load(&mut func, merge, slot, Ty::int(32), false);
    func.blocks[merge].term = Some(Terminator::Return { val: Some(loaded) });
    func.blocks[dead].term = Some(Terminator::Br {
        dest: merge,
        args: Default::default(),
    });

    let mut pass = Mem2Reg;
    assert!(pass.run(&mut func));
    assert!(!has_memory_ops(&func));
    assert!(crate::verify::verify_func(&func).is_empty());
    let dead_args = br_args(&func, dead);
    assert_eq!(dead_args.len(), 1);
    assert_eq!(func.const_value(dead_args[0]), Some(&Const::Undef));
}

#[test]
fn mem2reg_leaves_switch_target_phi_slots_in_memory() {
    // The refused slot's phi would land on the switch's default
    // target, which cannot carry arguments; the other slot still
    // promotes in the same run.
    let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("sw", sig);
    let scrut = param(&mut func, entry, Ty::int(32));
    let refused = emit_alloca(&mut func, entry, Ty::int(32));
    let simple = emit_alloca(&mut func, entry, Ty::int(32));
    let case1 = func.push_block(vec![]);
    let default = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::Switch {
        scrutinee: scrut,
        cases: vec![(1, case1)],
        default,
    });
    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let seven = func.push_const(Const::Int { bits: 32, value: 7 });
    emit_store(&mut func, case1, refused, one, false);
    func.blocks[case1].term = Some(Terminator::Br {
        dest: default,
        args: Default::default(),
    });
    emit_store(&mut func, entry, simple, seven, false);
    let a = emit_load(&mut func, default, refused, Ty::int(32), false);
    let b = emit_load(&mut func, default, simple, Ty::int(32), false);
    let sum = emit_bin(&mut func, default, InstKind::Add(a, b), Ty::int(32));
    func.blocks[default].term = Some(Terminator::Return { val: Some(sum) });

    let mut pass = Mem2Reg;
    assert!(pass.run(&mut func));
    assert!(crate::verify::verify_func(&func).is_empty());
    // What remains is exactly the refused slot's traffic.
    let memory_ops = resident_kinds(&func)
        .iter()
        .filter(|kind| matches!(kind, InstKind::Alloca { .. } | InstKind::Load { .. } | InstKind::Store { .. }))
        .count();
    assert_eq!(memory_ops, 3);
}

#[test]
fn mem2reg_waits_for_simplify_cfg_on_dead_blocks() {
    // A load in an unreachable block blocks promotion until
    // SimplifyCfg clears it; the pipeline then promotes cleanly.
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("dead", sig);
    let slot = emit_alloca(&mut func, entry, Ty::int(32));
    let seven = func.push_const(Const::Int { bits: 32, value: 7 });
    emit_store(&mut func, entry, slot, seven, false);
    let live = emit_load(&mut func, entry, slot, Ty::int(32), false);
    let dead = func.push_block(vec![]);
    let dead_load = emit_load(&mut func, dead, slot, Ty::int(32), false);
    func.blocks[dead].term = Some(Terminator::Return { val: Some(dead_load) });
    func.blocks[entry].term = Some(Terminator::Return { val: Some(live) });

    let mut pass = Mem2Reg;
    assert!(!pass.run(&mut func));
    assert!(has_memory_ops(&func));

    let mut simplify = SimplifyCfg;
    assert!(simplify.run(&mut func));
    assert!(pass.run(&mut func));
    assert!(!has_memory_ops(&func));
    assert!(crate::verify::verify_func(&func).is_empty());
    let Terminator::Return { val } = func.blocks[entry].term.clone().unwrap() else {
        unreachable!()
    };
    assert_eq!(func.const_value(val.unwrap()), Some(&Const::Int { bits: 32, value: 7 }));
}

#[test]
fn mem2reg_drops_untouched_allocas() {
    let sig = Signature::new(vec![], Ty::Void);
    let (mut func, entry) = Function::new("unused", sig);
    emit_alloca(&mut func, entry, Ty::int(32));
    emit_alloca(&mut func, entry, Ty::ptr(Ty::int(32)));
    func.blocks[entry].term = Some(Terminator::Return { val: None });

    let mut pass = Mem2Reg;
    assert!(pass.run(&mut func));
    assert!(!has_memory_ops(&func));
    assert!(!pass.run(&mut func));
}

#[test]
fn mem2reg_leaves_aggregate_slots_in_memory() {
    // Aggregate slots stay addressed memory: promotion is scalar.
    let point = Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(32)],
        offsets: vec![0, 4],
        size: 8,
    };
    let sig = Signature::new(vec![], point.clone());
    let (mut func, entry) = Function::new("agg", sig);
    let slot = emit_alloca(&mut func, entry, point.clone());
    let value = func.values.push(crate::ir::ValueData {
        ty: point.clone(),
        def: Def::Const(Const::Undef),
    });
    emit_store(&mut func, entry, slot, value, false);
    let loaded = emit_load(&mut func, entry, slot, point, false);
    func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

    let mut pass = Mem2Reg;
    assert!(!pass.run(&mut func));
    assert!(has_memory_ops(&func));
}
