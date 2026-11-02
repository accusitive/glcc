use super::*;
use crate::ir::Terminator;
use crate::ty::Signature;
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

fn iconst(func: &mut Function, bits: u32, value: i128) -> Value {
    func.push_const(Const::Int { bits, value })
}

fn returned(func: &Function) -> Value {
    match &func.blocks[func.entry].term {
        Some(Terminator::Return { val: Some(v) }) => *v,
        other => panic!("expected a value return, got {other:?}"),
    }
}

fn run(func: &mut Function) -> bool {
    let changed = InstCombine.run(func);
    let errors = verify_func(func);
    assert!(errors.is_empty(), "verifier: {errors:?}");
    changed
}

fn int_func() -> (Function, Block, Value) {
    let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let x = param(&mut func, entry, Ty::int(32), 0);
    (func, entry, x)
}

#[test]
fn add_of_zero_uses_the_other_operand() {
    let (mut func, entry, x) = int_func();
    let zero = iconst(&mut func, 32, 0);
    let sum = emit(&mut func, entry, InstKind::Add(x, zero), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
    assert!(run(&mut func));
    assert_eq!(returned(&func), x);
    assert!(!run(&mut func), "second run is a no-op");
}

#[test]
fn mul_by_zero_becomes_a_typed_zero() {
    let (mut func, entry, x) = int_func();
    let zero = iconst(&mut func, 32, 0);
    let product = emit(&mut func, entry, InstKind::Mul(x, zero), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(product) });
    assert!(run(&mut func));
    let out = returned(&func);
    assert_eq!(func.values[out].ty, Ty::int(32));
    assert_eq!(func.const_value(out), Some(&Const::Int { bits: 32, value: 0 }));
}

#[test]
fn self_subtraction_and_xor_fold_to_zero() {
    let (mut func, entry, x) = int_func();
    let out = emit(&mut func, entry, InstKind::Sub(x, x), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(out) });
    assert!(run(&mut func));
    assert_eq!(func.const_value(returned(&func)), Some(&Const::Int { bits: 32, value: 0 }));

    let (mut func, entry, x) = int_func();
    let out = emit(&mut func, entry, InstKind::Xor(x, x), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(out) });
    assert!(run(&mut func));
    assert_eq!(func.const_value(returned(&func)), Some(&Const::Int { bits: 32, value: 0 }));
}

#[test]
fn division_by_one_keeps_the_dividend() {
    for build in [InstKind::Udiv, InstKind::Sdiv] {
        let (mut func, entry, x) = int_func();
        let one = iconst(&mut func, 32, 1);
        let quotient = emit(&mut func, entry, build(x, one), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(quotient) });
        assert!(run(&mut func));
        assert_eq!(returned(&func), x);
    }
}

#[test]
fn remainder_by_one_folds_to_zero() {
    for build in [InstKind::Urem, InstKind::Srem] {
        let (mut func, entry, x) = int_func();
        let one = iconst(&mut func, 32, 1);
        let remainder = emit(&mut func, entry, build(x, one), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(remainder) });
        assert!(run(&mut func));
        let out = returned(&func);
        assert_eq!(func.values[out].ty, Ty::int(32));
        assert_eq!(func.const_value(out), Some(&Const::Int { bits: 32, value: 0 }));
    }
}

#[test]
fn select_with_equal_arms_keeps_one() {
    let sig = Signature::new(vec![Ty::Bool, Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let cond = param(&mut func, entry, Ty::Bool, 0);
    let x = param(&mut func, entry, Ty::int(32), 1);
    let sel = emit(&mut func, entry, InstKind::Select { cond, then_val: x, else_val: x }, Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
    assert!(run(&mut func));
    assert_eq!(returned(&func), x);
}

#[test]
fn zero_index_gep_retypes_with_a_bitcast() {
    let array = Ty::Array {
        elem: Box::new(Ty::int(8)),
        len: 4,
    };
    let source_ty = Ty::ptr(array);
    let dest_ty = Ty::ptr(Ty::int(8));
    let sig = Signature::new(vec![source_ty.clone()], dest_ty.clone());
    let (mut func, entry) = Function::new("f", sig);
    let source = param(&mut func, entry, source_ty, 0);
    let gep = emit(
        &mut func,
        entry,
        InstKind::GetElementPtr {
            base: source,
            indices: smallvec::smallvec![GepIndex::ConstIndex(0)],
        },
        dest_ty,
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(gep) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("pointer conversion");
    match &func.insts[inst].kind {
        InstKind::Bitcast(ptr) => assert_eq!(*ptr, source),
        other => panic!("expected a pointer bitcast, got {other:?}"),
    }
}

#[test]
fn int_to_ptr_of_a_converted_pointer_retypes_without_round_tripping() {
    let array = Ty::Array {
        elem: Box::new(Ty::int(8)),
        len: 4,
    };
    let source_ty = Ty::ptr(array);
    let dest_ty = Ty::ptr(Ty::int(8));
    let sig = Signature::new(vec![source_ty.clone()], dest_ty.clone());
    let (mut func, entry) = Function::new("f", sig);
    let source = param(&mut func, entry, source_ty, 0);
    let int = emit(&mut func, entry, InstKind::PtrToInt(source), Ty::int(64));
    let cast = emit(&mut func, entry, InstKind::IntToPtr(int), dest_ty);
    func.blocks[entry].term = Some(Terminator::Return { val: Some(cast) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("pointer conversion");
    match &func.insts[inst].kind {
        InstKind::Bitcast(ptr) => assert_eq!(*ptr, source),
        other => panic!("expected a pointer bitcast, got {other:?}"),
    }
}

#[test]
fn select_with_a_condition_arm_becomes_bitwise() {
    let sig = Signature::new(vec![Ty::Bool, Ty::Bool], Ty::Bool);
    let (mut func, entry) = Function::new("f", sig);
    let cond = param(&mut func, entry, Ty::Bool, 0);
    let x = param(&mut func, entry, Ty::Bool, 1);
    let sel = emit(
        &mut func,
        entry,
        InstKind::Select {
            cond,
            then_val: x,
            else_val: cond,
        },
        Ty::Bool,
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt boolean op");
    assert!(matches!(func.insts[inst].kind, InstKind::And(a, b) if a == cond && b == x));

    let sig = Signature::new(vec![Ty::Bool, Ty::Bool], Ty::Bool);
    let (mut func, entry) = Function::new("f", sig);
    let cond = param(&mut func, entry, Ty::Bool, 0);
    let x = param(&mut func, entry, Ty::Bool, 1);
    let sel = emit(
        &mut func,
        entry,
        InstKind::Select {
            cond,
            then_val: cond,
            else_val: x,
        },
        Ty::Bool,
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt boolean op");
    assert!(matches!(func.insts[inst].kind, InstKind::Or(a, b) if a == cond && b == x));
}

#[test]
fn select_of_zero_extended_arms_becomes_bitwise() {
    let sig = Signature::new(vec![Ty::Bool, Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let cond = param(&mut func, entry, Ty::Bool, 0);
    let x = param(&mut func, entry, Ty::Bool, 1);
    let cond_wide = emit(&mut func, entry, InstKind::Zext(cond), Ty::int(32));
    let x_wide = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
    let sel = emit(
        &mut func,
        entry,
        InstKind::Select {
            cond,
            then_val: x_wide,
            else_val: cond_wide,
        },
        Ty::int(32),
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt bitwise op");
    assert!(matches!(func.insts[inst].kind, InstKind::And(a, b) if a == cond_wide && b == x_wide));

    let sig = Signature::new(vec![Ty::Bool, Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let cond = param(&mut func, entry, Ty::Bool, 0);
    let x = param(&mut func, entry, Ty::Bool, 1);
    let cond_wide = emit(&mut func, entry, InstKind::Zext(cond), Ty::int(32));
    let x_wide = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
    let sel = emit(
        &mut func,
        entry,
        InstKind::Select {
            cond,
            then_val: cond_wide,
            else_val: x_wide,
        },
        Ty::int(32),
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt bitwise op");
    assert!(matches!(func.insts[inst].kind, InstKind::Or(a, b) if a == cond_wide && b == x_wide));
}

#[test]
fn extract_of_insert_uses_the_inserted_field() {
    let ty = Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(32)],
        offsets: vec![0, 4],
        size: 8,
    };
    let sig = Signature::new(vec![ty.clone(), Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let base = param(&mut func, entry, ty.clone(), 0);
    let x = param(&mut func, entry, Ty::int(32), 1);
    let inserted = emit(&mut func, entry, InstKind::InsertValue { agg: base, elem: x, index: 0 }, ty);
    let extracted = emit(&mut func, entry, InstKind::ExtractValue { agg: inserted, index: 0 }, Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(extracted) });
    assert!(run(&mut func));
    assert_eq!(returned(&func), x);
}

#[test]
fn extract_of_insert_reads_other_fields_from_the_base() {
    let ty = Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(32)],
        offsets: vec![0, 4],
        size: 8,
    };
    let sig = Signature::new(vec![ty.clone(), Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let base = param(&mut func, entry, ty.clone(), 0);
    let x = param(&mut func, entry, Ty::int(32), 1);
    let inserted = emit(&mut func, entry, InstKind::InsertValue { agg: base, elem: x, index: 0 }, ty);
    let extracted = emit(&mut func, entry, InstKind::ExtractValue { agg: inserted, index: 1 }, Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(extracted) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("field extraction");
    match &func.insts[inst].kind {
        InstKind::ExtractValue { agg, index } => {
            assert_eq!(*agg, base);
            assert_eq!(*index, 1);
        }
        other => panic!("expected extraction from the base, got {other:?}"),
    }
}

#[test]
fn rebuilding_an_aggregate_from_its_fields_is_the_original_value() {
    let ty = Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(32)],
        offsets: vec![0, 4],
        size: 8,
    };
    let sig = Signature::new(vec![ty.clone()], ty.clone());
    let (mut func, entry) = Function::new("f", sig);
    let base = param(&mut func, entry, ty.clone(), 0);
    let undef = func.values.push(ValueData {
        ty: ty.clone(),
        def: Def::Const(Const::Undef),
    });
    let first = emit(&mut func, entry, InstKind::ExtractValue { agg: base, index: 0 }, Ty::int(32));
    let second = emit(&mut func, entry, InstKind::ExtractValue { agg: base, index: 1 }, Ty::int(32));
    let with_first = emit(
        &mut func,
        entry,
        InstKind::InsertValue {
            agg: undef,
            elem: first,
            index: 0,
        },
        ty.clone(),
    );
    let rebuilt = emit(
        &mut func,
        entry,
        InstKind::InsertValue {
            agg: with_first,
            elem: second,
            index: 1,
        },
        ty,
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(rebuilt) });
    assert!(run(&mut func));
    assert_eq!(returned(&func), base);
}

#[test]
fn rebuilding_with_one_changed_field_uses_the_source() {
    let ty = Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(32)],
        offsets: vec![0, 4],
        size: 8,
    };
    let sig = Signature::new(vec![ty.clone(), Ty::int(32)], ty.clone());
    let (mut func, entry) = Function::new("f", sig);
    let base = param(&mut func, entry, ty.clone(), 0);
    let changed = param(&mut func, entry, Ty::int(32), 1);
    let undef = func.values.push(ValueData {
        ty: ty.clone(),
        def: Def::Const(Const::Undef),
    });
    let first = emit(&mut func, entry, InstKind::ExtractValue { agg: base, index: 0 }, Ty::int(32));
    let with_first = emit(
        &mut func,
        entry,
        InstKind::InsertValue {
            agg: undef,
            elem: first,
            index: 0,
        },
        ty.clone(),
    );
    let rebuilt = emit(
        &mut func,
        entry,
        InstKind::InsertValue {
            agg: with_first,
            elem: changed,
            index: 1,
        },
        ty,
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(rebuilt) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("aggregate value");
    match &func.insts[inst].kind {
        InstKind::InsertValue { agg, elem, index } => {
            assert_eq!(*agg, base);
            assert_eq!(*elem, changed);
            assert_eq!(*index, 1);
        }
        other => panic!("expected a field update on the source, got {other:?}"),
    }
}

#[test]
fn rebuilding_with_several_changed_fields_keeps_the_source() {
    let ty = Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(32), Ty::int(32)],
        offsets: vec![0, 4, 8],
        size: 12,
    };
    let sig = Signature::new(vec![ty.clone(), Ty::int(32), Ty::int(32)], ty.clone());
    let (mut func, entry) = Function::new("f", sig);
    let base = param(&mut func, entry, ty.clone(), 0);
    let first = param(&mut func, entry, Ty::int(32), 1);
    let second = param(&mut func, entry, Ty::int(32), 2);
    let undef = func.values.push(ValueData {
        ty: ty.clone(),
        def: Def::Const(Const::Undef),
    });
    let unchanged = emit(&mut func, entry, InstKind::ExtractValue { agg: base, index: 2 }, Ty::int(32));
    let with_first = emit(
        &mut func,
        entry,
        InstKind::InsertValue {
            agg: undef,
            elem: first,
            index: 0,
        },
        ty.clone(),
    );
    let with_second = emit(
        &mut func,
        entry,
        InstKind::InsertValue {
            agg: with_first,
            elem: second,
            index: 1,
        },
        ty.clone(),
    );
    let rebuilt = emit(
        &mut func,
        entry,
        InstKind::InsertValue {
            agg: with_second,
            elem: unchanged,
            index: 2,
        },
        ty,
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(rebuilt) });
    assert!(run(&mut func));
    let outer = def_inst(&func, returned(&func)).expect("aggregate value");
    let (inner, elem, index) = match &func.insts[outer].kind {
        InstKind::InsertValue { agg, elem, index } => (*agg, *elem, *index),
        other => panic!("expected a field update on the source, got {other:?}"),
    };
    assert_eq!(index, 1);
    assert_eq!(elem, second);
    let inner = def_inst(&func, inner).expect("first field update");
    match &func.insts[inner].kind {
        InstKind::InsertValue { agg, elem, index } => {
            assert_eq!(*agg, base);
            assert_eq!(*elem, first);
            assert_eq!(*index, 0);
        }
        other => panic!("expected the unchanged source as the base, got {other:?}"),
    }
}

#[test]
fn partial_rebuild_is_not_the_original_aggregate() {
    let ty = Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(32)],
        offsets: vec![0, 4],
        size: 8,
    };
    let sig = Signature::new(vec![ty.clone()], ty.clone());
    let (mut func, entry) = Function::new("f", sig);
    let base = param(&mut func, entry, ty.clone(), 0);
    let undef = func.values.push(ValueData {
        ty: ty.clone(),
        def: Def::Const(Const::Undef),
    });
    let first = emit(&mut func, entry, InstKind::ExtractValue { agg: base, index: 0 }, Ty::int(32));
    let rebuilt = emit(
        &mut func,
        entry,
        InstKind::InsertValue {
            agg: undef,
            elem: first,
            index: 0,
        },
        ty.clone(),
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(rebuilt) });
    assert!(!run(&mut func));
    assert_eq!(returned(&func), rebuilt);
}

#[test]
fn insert_of_an_extracted_field_is_the_original_value() {
    let ty = Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(32)],
        offsets: vec![0, 4],
        size: 8,
    };
    let sig = Signature::new(vec![ty.clone()], ty.clone());
    let (mut func, entry) = Function::new("f", sig);
    let base = param(&mut func, entry, ty.clone(), 0);
    let field = emit(&mut func, entry, InstKind::ExtractValue { agg: base, index: 0 }, Ty::int(32));
    let inserted = emit(&mut func, entry, InstKind::InsertValue { agg: base, elem: field, index: 0 }, ty);
    func.blocks[entry].term = Some(Terminator::Return { val: Some(inserted) });
    assert!(run(&mut func));
    assert_eq!(returned(&func), base);
}

#[test]
fn zext_of_a_boolean_comparison_uses_the_boolean_value() {
    let sig = Signature::new(vec![Ty::Bool, Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let a = param(&mut func, entry, Ty::Bool, 0);
    let b = param(&mut func, entry, Ty::Bool, 1);
    let a_wide = emit(&mut func, entry, InstKind::Zext(a), Ty::int(32));
    let b_wide = emit(&mut func, entry, InstKind::Zext(b), Ty::int(32));
    let both = emit(&mut func, entry, InstKind::And(a_wide, b_wide), Ty::int(32));
    let zero = iconst(&mut func, 32, 0);
    let not_zero = emit(
        &mut func,
        entry,
        InstKind::Icmp {
            pred: IcmpPred::Ne,
            lhs: both,
            rhs: zero,
        },
        Ty::Bool,
    );
    match zext_boolean_value(&func, not_zero, &Ty::int(32)) {
        Some(Plan::Value(value)) => assert_eq!(value, both),
        Some(_) => panic!("expected the boolean value"),
        None => panic!("nonzero boolean comparison was not recognized"),
    }

    let is_zero = emit(
        &mut func,
        entry,
        InstKind::Icmp {
            pred: IcmpPred::Eq,
            lhs: both,
            rhs: zero,
        },
        Ty::Bool,
    );
    match zext_boolean_value(&func, is_zero, &Ty::int(32)) {
        Some(Plan::XorConst(value, 1)) => assert_eq!(value, both),
        _ => panic!("expected an inverted boolean"),
    }
}

#[test]
fn select_with_constant_condition_picks_an_arm() {
    let (mut func, entry, x) = int_func();
    let cond = iconst(&mut func, 1, 1);
    let other = iconst(&mut func, 32, 99);
    let sel = emit(
        &mut func,
        entry,
        InstKind::Select {
            cond,
            then_val: x,
            else_val: other,
        },
        Ty::int(32),
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
    assert!(run(&mut func));
    assert_eq!(returned(&func), x);
}

/// `if (x < y)` lowers to `icmp` -> `zext` -> `icmp.Ne 0`; the
/// round-trip through `i32` must collapse back to the condition.
#[test]
fn zext_compared_against_zero_collapses() {
    for (pred, truthy) in [(IcmpPred::Ne, true), (IcmpPred::Eq, false)] {
        let sig = Signature::new(vec![Ty::int(32), Ty::int(32)], Ty::Bool);
        let (mut func, entry) = Function::new("f", sig);
        let a = param(&mut func, entry, Ty::int(32), 0);
        let b = param(&mut func, entry, Ty::int(32), 1);
        let cmp = emit(
            &mut func,
            entry,
            InstKind::Icmp {
                pred: IcmpPred::Slt,
                lhs: a,
                rhs: b,
            },
            Ty::Bool,
        );
        let wide = emit(&mut func, entry, InstKind::Zext(cmp), Ty::int(32));
        let zero = iconst(&mut func, 32, 0);
        let test = emit(&mut func, entry, InstKind::Icmp { pred, lhs: wide, rhs: zero }, Ty::Bool);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(test) });
        assert!(run(&mut func), "{pred:?}");
        let out = returned(&func);
        if truthy {
            assert_eq!(out, cmp, "{pred:?} keeps the condition");
        } else {
            assert_ne!(out, cmp, "{pred:?} negates the condition");
            let inst = def_inst(&func, out).expect("rebuilt comparison");
            match &func.insts[inst].kind {
                InstKind::Icmp { pred, lhs, rhs } => {
                    assert_eq!(*pred, IcmpPred::Eq);
                    assert_eq!(*lhs, cmp);
                    assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 1, value: 0 }));
                }
                other => panic!("expected icmp, got {other:?}"),
            }
        }
    }
}

#[test]
fn zext_compared_against_zero_works_with_either_side() {
    let sig = Signature::new(vec![Ty::int(32), Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let a = param(&mut func, entry, Ty::int(32), 0);
    let b = param(&mut func, entry, Ty::int(32), 1);
    let cmp = emit(
        &mut func,
        entry,
        InstKind::Icmp {
            pred: IcmpPred::Sgt,
            lhs: a,
            rhs: b,
        },
        Ty::Bool,
    );
    let wide = emit(&mut func, entry, InstKind::Zext(cmp), Ty::int(64));
    let zero = iconst(&mut func, 64, 0);
    let test = emit(
        &mut func,
        entry,
        InstKind::Icmp {
            pred: IcmpPred::Ne,
            lhs: zero,
            rhs: wide,
        },
        Ty::Bool,
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(test) });
    assert!(run(&mut func));
    assert_eq!(returned(&func), cmp);
}

#[test]
fn truncation_of_a_widened_value_round_trips() {
    let sig = Signature::new(vec![Ty::int(8)], Ty::int(8));
    let (mut func, entry) = Function::new("f", sig);
    let x = param(&mut func, entry, Ty::int(8), 0);
    let wide = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
    let narrow = emit(&mut func, entry, InstKind::Trunc(wide), Ty::int(8));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(narrow) });
    assert!(run(&mut func));
    assert_eq!(returned(&func), x);
}

/// `(unsigned char)(x & 0xff)`: the mask covers the truncation width.
#[test]
fn truncation_absorbs_a_covering_mask() {
    let sig = Signature::new(vec![Ty::int(32)], Ty::int(8));
    let (mut func, entry) = Function::new("f", sig);
    let x = param(&mut func, entry, Ty::int(32), 0);
    let mask = iconst(&mut func, 32, 255);
    let masked = emit(&mut func, entry, InstKind::And(x, mask), Ty::int(32));
    let narrow = emit(&mut func, entry, InstKind::Trunc(masked), Ty::int(8));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(narrow) });
    assert!(run(&mut func));
    let out = returned(&func);
    let inst = def_inst(&func, out).expect("rebuilt truncation");
    match &func.insts[inst].kind {
        InstKind::Trunc(operand) => assert_eq!(*operand, x),
        other => panic!("expected truncation, got {other:?}"),
    }
}

#[test]
fn nested_widening_collapses_to_one_step() {
    let sig = Signature::new(vec![Ty::int(8)], Ty::int(64));
    let (mut func, entry) = Function::new("f", sig);
    let x = param(&mut func, entry, Ty::int(8), 0);
    let wide = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
    let wider = emit(&mut func, entry, InstKind::Zext(wide), Ty::int(64));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(wider) });
    assert!(run(&mut func));
    let out = returned(&func);
    assert_eq!(func.values[out].ty, Ty::int(64));
    let inst = def_inst(&func, out).expect("rebuilt widening");
    match &func.insts[inst].kind {
        InstKind::Zext(operand) => assert_eq!(*operand, x),
        other => panic!("expected widening, got {other:?}"),
    }
}

#[test]
fn self_comparisons_fold_to_constants() {
    for (pred, truth) in [
        (IcmpPred::Eq, true),
        (IcmpPred::Ne, false),
        (IcmpPred::Ult, false),
        (IcmpPred::Ule, true),
        (IcmpPred::Sgt, false),
        (IcmpPred::Sge, true),
        (IcmpPred::Slt, false),
        (IcmpPred::Sle, true),
        (IcmpPred::Ugt, false),
        (IcmpPred::Uge, true),
    ] {
        let (mut func, entry, x) = int_func();
        let cmp = emit(&mut func, entry, InstKind::Icmp { pred, lhs: x, rhs: x }, Ty::Bool);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
        assert!(run(&mut func), "{pred:?}");
        let out = returned(&func);
        assert_eq!(func.values[out].ty, Ty::Bool, "{pred:?}");
        assert_eq!(
            func.const_value(out),
            Some(&Const::Int {
                bits: 1,
                value: i128::from(truth)
            }),
            "{pred:?}"
        );
    }
}

#[test]
fn doubled_unary_operators_cancel() {
    let (mut func, entry, x) = int_func();
    let a = emit(&mut func, entry, InstKind::Neg(x), Ty::int(32));
    let b = emit(&mut func, entry, InstKind::Neg(a), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(b) });
    assert!(run(&mut func));
    assert_eq!(returned(&func), x);

    let (mut func, entry, x) = int_func();
    let a = emit(&mut func, entry, InstKind::Not(x), Ty::int(32));
    let b = emit(&mut func, entry, InstKind::Not(a), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(b) });
    assert!(run(&mut func));
    assert_eq!(returned(&func), x);
}

#[test]
fn dead_results_wait_for_dce() {
    // Nothing uses the sum, so there is nothing to rewrite; DCE in
    // the pipeline removes the instruction, and rewriting it here
    // would only spin the fixpoint.
    let sig = Signature::new(vec![], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let x = iconst(&mut func, 32, 1);
    let zero = iconst(&mut func, 32, 0);
    let _ = emit(&mut func, entry, InstKind::Add(x, zero), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: None });
    assert!(!run(&mut func));
}

#[test]
fn doubled_add_becomes_a_shift() {
    let (mut func, entry, x) = int_func();
    let sum = emit(&mut func, entry, InstKind::Add(x, x), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt shift");
    match &func.insts[inst].kind {
        InstKind::Shl(lhs, amount) => {
            assert_eq!(*lhs, x);
            assert_eq!(func.const_value(*amount), Some(&Const::Int { bits: 32, value: 1 }));
        }
        other => panic!("expected shift, got {other:?}"),
    }
}

#[test]
fn signed_power_of_two_division_avoids_division() {
    for remainder in [false, true] {
        let (mut func, entry, x) = int_func();
        let four = iconst(&mut func, 32, 4);
        let kind = if remainder { InstKind::Srem(x, four) } else { InstKind::Sdiv(x, four) };
        let out = emit(&mut func, entry, kind, Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(out) });
        assert!(run(&mut func));
        assert!(
            !func.blocks[entry]
                .insts
                .iter()
                .any(|inst| matches!(func.insts[*inst].kind, InstKind::Sdiv(..) | InstKind::Srem(..)))
        );
        assert!(crate::verify::verify_func(&func).is_empty());
    }
}

#[test]
fn reassociates_constant_additions() {
    let (mut func, entry, x) = int_func();
    let seven = iconst(&mut func, 32, 7);
    let five = iconst(&mut func, 32, 5);
    let inner = emit(&mut func, entry, InstKind::Add(x, seven), Ty::int(32));
    let outer = emit(&mut func, entry, InstKind::Add(inner, five), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(outer) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt addition");
    match &func.insts[inst].kind {
        InstKind::Add(lhs, rhs) => {
            assert_eq!(*lhs, x);
            assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 32, value: 12 }));
        }
        other => panic!("expected addition, got {other:?}"),
    }
}

#[test]
fn reassociation_wraps_like_the_original_operations() {
    let (mut func, entry, x) = int_func();
    let max = iconst(&mut func, 32, i128::from(i32::MAX));
    let one = iconst(&mut func, 32, 1);
    let inner = emit(&mut func, entry, InstKind::Add(x, max), Ty::int(32));
    let outer = emit(&mut func, entry, InstKind::Add(inner, one), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(outer) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt addition");
    match &func.insts[inst].kind {
        InstKind::Add(lhs, rhs) => {
            assert_eq!(*lhs, x);
            assert_eq!(
                func.const_value(*rhs),
                Some(&Const::Int {
                    bits: 32,
                    value: i128::from(i32::MIN)
                })
            );
        }
        other => panic!("expected addition, got {other:?}"),
    }
}

#[test]
fn reassociates_constant_multiplications() {
    let (mut func, entry, x) = int_func();
    let six = iconst(&mut func, 32, 6);
    let four = iconst(&mut func, 32, 4);
    let inner = emit(&mut func, entry, InstKind::Mul(x, six), Ty::int(32));
    let outer = emit(&mut func, entry, InstKind::Mul(inner, four), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(outer) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt product");
    match &func.insts[inst].kind {
        InstKind::Mul(lhs, rhs) => {
            assert_eq!(*lhs, x);
            assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 32, value: 24 }));
        }
        other => panic!("expected product, got {other:?}"),
    }
}

#[test]
fn nested_shifts_combine() {
    for (mode, build) in [
        (ShiftMode::Left, InstKind::Shl as fn(Value, Value) -> InstKind),
        (ShiftMode::LogicalRight, InstKind::LShr as fn(Value, Value) -> InstKind),
        (ShiftMode::ArithmeticRight, InstKind::AShr as fn(Value, Value) -> InstKind),
    ] {
        let (mut func, entry, x) = int_func();
        let two = iconst(&mut func, 32, 2);
        let three = iconst(&mut func, 32, 3);
        let inner = emit(&mut func, entry, build(x, two), Ty::int(32));
        let outer = emit(&mut func, entry, build(inner, three), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(outer) });
        assert!(run(&mut func), "{mode:?}");
        let inst = def_inst(&func, returned(&func)).expect("rebuilt shift");
        match &func.insts[inst].kind {
            InstKind::Shl(lhs, by) => {
                assert_eq!(mode, ShiftMode::Left);
                assert_eq!(*lhs, x);
                assert_eq!(func.const_value(*by), Some(&Const::Int { bits: 32, value: 5 }));
            }
            InstKind::LShr(lhs, by) => {
                assert_eq!(mode, ShiftMode::LogicalRight);
                assert_eq!(*lhs, x);
                assert_eq!(func.const_value(*by), Some(&Const::Int { bits: 32, value: 5 }));
            }
            InstKind::AShr(lhs, by) => {
                assert_eq!(mode, ShiftMode::ArithmeticRight);
                assert_eq!(*lhs, x);
                assert_eq!(func.const_value(*by), Some(&Const::Int { bits: 32, value: 5 }));
            }
            other => panic!("expected shift, got {other:?}"),
        }
    }
}

#[test]
fn shared_inner_shifts_are_not_duplicated() {
    let (mut func, entry, x) = int_func();
    let two = iconst(&mut func, 32, 2);
    let three = iconst(&mut func, 32, 3);
    let inner = emit(&mut func, entry, InstKind::Shl(x, two), Ty::int(32));
    let outer = emit(&mut func, entry, InstKind::Shl(inner, three), Ty::int(32));
    let sum = emit(&mut func, entry, InstKind::Add(outer, inner), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
    assert!(!run(&mut func));
    let inst = def_inst(&func, outer).expect("outer shift");
    match &func.insts[inst].kind {
        InstKind::Shl(lhs, _) => assert_eq!(*lhs, inner),
        other => panic!("expected shift, got {other:?}"),
    }
}

#[test]
fn oversized_shift_chains_are_left_alone() {
    let (mut func, entry, x) = int_func();
    let wide = iconst(&mut func, 32, 30);
    let five = iconst(&mut func, 32, 5);
    let inner = emit(&mut func, entry, InstKind::Shl(x, wide), Ty::int(32));
    let outer = emit(&mut func, entry, InstKind::Shl(inner, five), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(outer) });
    assert!(!run(&mut func));
    let inst = def_inst(&func, outer).expect("outer shift");
    match &func.insts[inst].kind {
        InstKind::Shl(lhs, _) => assert_eq!(*lhs, inner),
        other => panic!("expected shift, got {other:?}"),
    }
}

#[test]
fn bitwise_constant_chains_combine() {
    for (build, left, right, combined, result) in [
        (InstKind::And as fn(Value, Value) -> InstKind, 0xF0, 0x3C, 0x30, "and"),
        (InstKind::Or as fn(Value, Value) -> InstKind, 0xFFFF0000, 0xFFFF, -1, "or"),
        (InstKind::Xor as fn(Value, Value) -> InstKind, 0xF0, 0x3C, 0xCC, "xor"),
    ] {
        let (mut func, entry, x) = int_func();
        let first = iconst(&mut func, 32, left);
        let second = iconst(&mut func, 32, right);
        let inner = emit(&mut func, entry, build(x, first), Ty::int(32));
        let outer = emit(&mut func, entry, build(inner, second), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(outer) });
        assert!(run(&mut func), "{result}");
        let out = returned(&func);
        if combined == -1 {
            assert_eq!(func.const_value(out), Some(&Const::Int { bits: 32, value: -1 }), "{result}");
            continue;
        }
        let inst = def_inst(&func, out).expect("rebuilt bitwise op");
        match (&func.insts[inst].kind, result) {
            (InstKind::And(lhs, rhs), "and") | (InstKind::Or(lhs, rhs), "or") | (InstKind::Xor(lhs, rhs), "xor") => {
                assert_eq!(*lhs, x);
                assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 32, value: combined }), "{result}");
            }
            (other, _) => panic!("expected {result}, got {other:?}"),
        }
    }
}

#[test]
fn and_chain_with_disjoint_masks_folds_to_zero() {
    let (mut func, entry, x) = int_func();
    let high = iconst(&mut func, 32, 0xF0);
    let low = iconst(&mut func, 32, 0x0F);
    let inner = emit(&mut func, entry, InstKind::And(x, high), Ty::int(32));
    let outer = emit(&mut func, entry, InstKind::And(inner, low), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(outer) });
    assert!(run(&mut func));
    assert_eq!(func.const_value(returned(&func)), Some(&Const::Int { bits: 32, value: 0 }));
}

#[test]
fn xor_chain_with_all_ones_becomes_not() {
    let (mut func, entry, x) = int_func();
    let high = iconst(&mut func, 32, 0xFFFFFF00);
    let rest = iconst(&mut func, 32, 0xFF);
    let inner = emit(&mut func, entry, InstKind::Xor(x, high), Ty::int(32));
    let outer = emit(&mut func, entry, InstKind::Xor(inner, rest), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(outer) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt negation");
    match &func.insts[inst].kind {
        InstKind::Not(operand) => assert_eq!(*operand, x),
        other => panic!("expected not, got {other:?}"),
    }
}

#[test]
fn shift_pairs_fold_to_masks() {
    let cases = [
        (InstKind::Shl as fn(Value, Value) -> InstKind, InstKind::LShr as fn(Value, Value) -> InstKind, 2, 0x3FFFFFFF),
        (InstKind::LShr as fn(Value, Value) -> InstKind, InstKind::Shl as fn(Value, Value) -> InstKind, 3, -8),
        (InstKind::AShr as fn(Value, Value) -> InstKind, InstKind::Shl as fn(Value, Value) -> InstKind, 3, -8),
    ];
    for (inner, outer, amount, expected) in cases {
        let (mut func, entry, x) = int_func();
        let by = iconst(&mut func, 32, amount);
        let shifted = emit(&mut func, entry, inner(x, by), Ty::int(32));
        let masked = emit(&mut func, entry, outer(shifted, by), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(masked) });
        assert!(run(&mut func));
        let inst = def_inst(&func, returned(&func)).expect("rebuilt mask");
        match &func.insts[inst].kind {
            InstKind::And(lhs, rhs) => {
                assert_eq!(*lhs, x);
                assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 32, value: expected }));
            }
            other => panic!("expected and, got {other:?}"),
        }
    }
}

#[test]
fn arithmetic_shift_after_left_shift_is_not_a_mask() {
    let (mut func, entry, x) = int_func();
    let two = iconst(&mut func, 32, 2);
    let shifted = emit(&mut func, entry, InstKind::Shl(x, two), Ty::int(32));
    let out = emit(&mut func, entry, InstKind::AShr(shifted, two), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(out) });
    assert!(!run(&mut func));
}

#[test]
fn unsigned_comparisons_against_zero_fold() {
    for (pred, truth) in [(IcmpPred::Ult, false), (IcmpPred::Uge, true)] {
        let (mut func, entry, x) = int_func();
        let zero = iconst(&mut func, 32, 0);
        let cmp = emit(&mut func, entry, InstKind::Icmp { pred, lhs: x, rhs: zero }, Ty::Bool);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
        assert!(run(&mut func), "{pred:?}");
        assert_eq!(
            func.const_value(returned(&func)),
            Some(&Const::Int {
                bits: 1,
                value: i128::from(truth)
            })
        );
    }

    for (pred, equal) in [(IcmpPred::Ule, true), (IcmpPred::Ugt, false)] {
        let (mut func, entry, x) = int_func();
        let zero = iconst(&mut func, 32, 0);
        let cmp = emit(&mut func, entry, InstKind::Icmp { pred, lhs: x, rhs: zero }, Ty::Bool);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
        assert!(run(&mut func), "{pred:?}");
        let inst = def_inst(&func, returned(&func)).expect("rebuilt comparison");
        match &func.insts[inst].kind {
            InstKind::Icmp { pred: narrowed, lhs, rhs } => {
                assert_eq!(*narrowed, if equal { IcmpPred::Eq } else { IcmpPred::Ne });
                assert_eq!(*lhs, x);
                assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 32, value: 0 }));
            }
            other => panic!("expected comparison, got {other:?}"),
        }
    }
}

#[test]
fn zero_extension_comparisons_narrow() {
    let sig = Signature::new(vec![Ty::int(8), Ty::int(8)], Ty::Bool);
    let (mut func, entry) = Function::new("f", sig);
    let x = param(&mut func, entry, Ty::int(8), 0);
    let y = param(&mut func, entry, Ty::int(8), 1);
    let left = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
    let right = emit(&mut func, entry, InstKind::Zext(y), Ty::int(32));
    let cmp = emit(
        &mut func,
        entry,
        InstKind::Icmp {
            pred: IcmpPred::Slt,
            lhs: left,
            rhs: right,
        },
        Ty::Bool,
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt comparison");
    match &func.insts[inst].kind {
        InstKind::Icmp { pred, lhs, rhs } => {
            assert_eq!(*pred, IcmpPred::Ult);
            assert_eq!((*lhs, *rhs), (x, y));
        }
        other => panic!("expected comparison, got {other:?}"),
    }
}

#[test]
fn sign_extension_comparisons_keep_their_order() {
    let sig = Signature::new(vec![Ty::int(8), Ty::int(8)], Ty::Bool);
    let (mut func, entry) = Function::new("f", sig);
    let x = param(&mut func, entry, Ty::int(8), 0);
    let y = param(&mut func, entry, Ty::int(8), 1);
    let left = emit(&mut func, entry, InstKind::Sext(x), Ty::int(32));
    let right = emit(&mut func, entry, InstKind::Sext(y), Ty::int(32));
    let cmp = emit(
        &mut func,
        entry,
        InstKind::Icmp {
            pred: IcmpPred::Ugt,
            lhs: left,
            rhs: right,
        },
        Ty::Bool,
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt comparison");
    match &func.insts[inst].kind {
        InstKind::Icmp { pred, lhs, rhs } => {
            assert_eq!(*pred, IcmpPred::Ugt);
            assert_eq!((*lhs, *rhs), (x, y));
        }
        other => panic!("expected comparison, got {other:?}"),
    }
}

#[test]
fn zero_extension_comparisons_with_constants_narrow() {
    let sig = Signature::new(vec![Ty::int(8)], Ty::Bool);
    let (mut func, entry) = Function::new("f", sig);
    let x = param(&mut func, entry, Ty::int(8), 0);
    let wide = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
    let constant = iconst(&mut func, 32, 200);
    let cmp = emit(
        &mut func,
        entry,
        InstKind::Icmp {
            pred: IcmpPred::Slt,
            lhs: wide,
            rhs: constant,
        },
        Ty::Bool,
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt comparison");
    match &func.insts[inst].kind {
        InstKind::Icmp { pred, lhs, rhs } => {
            assert_eq!(*pred, IcmpPred::Ult);
            assert_eq!(*lhs, x);
            assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 8, value: -56 }));
        }
        other => panic!("expected comparison, got {other:?}"),
    }

    for (pred, truth) in [(IcmpPred::Slt, true), (IcmpPred::Eq, false)] {
        let sig = Signature::new(vec![Ty::int(8)], Ty::Bool);
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(8), 0);
        let wide = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
        let constant = iconst(&mut func, 32, 300);
        let cmp = emit(&mut func, entry, InstKind::Icmp { pred, lhs: wide, rhs: constant }, Ty::Bool);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
        assert!(run(&mut func));
        assert_eq!(
            func.const_value(returned(&func)),
            Some(&Const::Int {
                bits: 1,
                value: i128::from(truth)
            })
        );
    }
}

#[test]
fn sign_extension_comparisons_with_constants_narrow() {
    let sig = Signature::new(vec![Ty::int(8)], Ty::Bool);
    let (mut func, entry) = Function::new("f", sig);
    let x = param(&mut func, entry, Ty::int(8), 0);
    let wide = emit(&mut func, entry, InstKind::Sext(x), Ty::int(32));
    let constant = iconst(&mut func, 32, 100);
    let cmp = emit(
        &mut func,
        entry,
        InstKind::Icmp {
            pred: IcmpPred::Slt,
            lhs: wide,
            rhs: constant,
        },
        Ty::Bool,
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt comparison");
    match &func.insts[inst].kind {
        InstKind::Icmp { pred, lhs, rhs } => {
            assert_eq!(*pred, IcmpPred::Slt);
            assert_eq!(*lhs, x);
            assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 8, value: 100 }));
        }
        other => panic!("expected comparison, got {other:?}"),
    }

    let sig = Signature::new(vec![Ty::int(8)], Ty::Bool);
    let (mut func, entry) = Function::new("f", sig);
    let x = param(&mut func, entry, Ty::int(8), 0);
    let wide = emit(&mut func, entry, InstKind::Sext(x), Ty::int(32));
    let constant = iconst(&mut func, 32, -200);
    let cmp = emit(
        &mut func,
        entry,
        InstKind::Icmp {
            pred: IcmpPred::Slt,
            lhs: wide,
            rhs: constant,
        },
        Ty::Bool,
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
    assert!(run(&mut func));
    assert_eq!(func.const_value(returned(&func)), Some(&Const::Int { bits: 1, value: 0 }));
}

#[test]
fn constant_on_the_left_narrows_the_comparison() {
    let sig = Signature::new(vec![Ty::int(8)], Ty::Bool);
    let (mut func, entry) = Function::new("f", sig);
    let x = param(&mut func, entry, Ty::int(8), 0);
    let wide = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
    let constant = iconst(&mut func, 32, 200);
    let cmp = emit(
        &mut func,
        entry,
        InstKind::Icmp {
            pred: IcmpPred::Sgt,
            lhs: constant,
            rhs: wide,
        },
        Ty::Bool,
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt comparison");
    match &func.insts[inst].kind {
        InstKind::Icmp { pred, lhs, rhs } => {
            assert_eq!(*pred, IcmpPred::Ult);
            assert_eq!(*lhs, x);
            assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 8, value: -56 }));
        }
        other => panic!("expected comparison, got {other:?}"),
    }
}

#[test]
fn zero_extended_boolean_comparisons_fold() {
    for (pred, constant, expect_value) in [(IcmpPred::Eq, 1, true), (IcmpPred::Ne, 1, false), (IcmpPred::Eq, 0, false), (IcmpPred::Ne, 0, true)] {
        let sig = Signature::new(vec![Ty::Bool], Ty::Bool);
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::Bool, 0);
        let wide = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
        let constant = iconst(&mut func, 32, constant);
        let cmp = emit(&mut func, entry, InstKind::Icmp { pred, lhs: wide, rhs: constant }, Ty::Bool);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
        assert!(run(&mut func), "{pred:?}");
        if expect_value {
            assert_eq!(returned(&func), x, "{pred:?}");
        } else {
            let inst = def_inst(&func, returned(&func)).expect("negated boolean");
            match &func.insts[inst].kind {
                InstKind::Icmp { pred: IcmpPred::Eq, lhs, rhs } => {
                    assert_eq!(*lhs, x);
                    assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 1, value: 0 }));
                }
                other => panic!("expected negated boolean, got {other:?}"),
            }
        }
    }

    for (pred, truth) in [(IcmpPred::Eq, false), (IcmpPred::Ne, true)] {
        let sig = Signature::new(vec![Ty::Bool], Ty::Bool);
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::Bool, 0);
        let wide = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
        let constant = iconst(&mut func, 32, 2);
        let cmp = emit(&mut func, entry, InstKind::Icmp { pred, lhs: wide, rhs: constant }, Ty::Bool);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
        assert!(run(&mut func));
        assert_eq!(
            func.const_value(returned(&func)),
            Some(&Const::Int {
                bits: 1,
                value: i128::from(truth)
            })
        );
    }
}

#[test]
fn bitwise_operations_on_extended_booleans_use_boolean_ops() {
    for op in [Bitwise::And, Bitwise::Or, Bitwise::Xor] {
        let sig = Signature::new(vec![Ty::Bool, Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let a = param(&mut func, entry, Ty::Bool, 0);
        let b = param(&mut func, entry, Ty::Bool, 1);
        let a_wide = emit(&mut func, entry, InstKind::Zext(a), Ty::int(32));
        let b_wide = emit(&mut func, entry, InstKind::Zext(b), Ty::int(32));
        let kind = match op {
            Bitwise::And => InstKind::And(a_wide, b_wide),
            Bitwise::Or => InstKind::Or(a_wide, b_wide),
            Bitwise::Xor => InstKind::Xor(a_wide, b_wide),
        };
        let out = emit(&mut func, entry, kind, Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(out) });
        assert!(run(&mut func));
        let outer = def_inst(&func, returned(&func)).expect("extension");
        let inner = match &func.insts[outer].kind {
            InstKind::Zext(inner) => *inner,
            other => panic!("expected boolean extension, got {other:?}"),
        };
        let inner = def_inst(&func, inner).expect("boolean operation");
        match (&func.insts[inner].kind, op) {
            (InstKind::And(x, y), Bitwise::And) | (InstKind::Or(x, y), Bitwise::Or) | (InstKind::Xor(x, y), Bitwise::Xor) => {
                assert_eq!(*x, a);
                assert_eq!(*y, b);
            }
            (other, _) => panic!("expected boolean operation, got {other:?}"),
        }
    }
}

#[test]
fn select_of_boolean_constants_becomes_an_extension() {
    for (then_value, else_value, signed, invert) in [(1, 0, false, false), (0, 1, false, true), (-1, 0, true, false), (0, -1, true, true)] {
        let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let cond = param(&mut func, entry, Ty::Bool, 0);
        let then_val = iconst(&mut func, 32, then_value);
        let else_val = iconst(&mut func, 32, else_value);
        let sel = emit(&mut func, entry, InstKind::Select { cond, then_val, else_val }, Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
        assert!(run(&mut func), "select {then_value}/{else_value}");
        let outer = def_inst(&func, returned(&func)).expect("extension");
        let inner = match &func.insts[outer].kind {
            InstKind::Zext(inner) | InstKind::Sext(inner) => *inner,
            other => panic!("expected extension, got {other:?}"),
        };
        assert_eq!(matches!(func.insts[outer].kind, InstKind::Sext(_)), signed);
        if invert {
            let inner = def_inst(&func, inner).expect("negated condition");
            match &func.insts[inner].kind {
                InstKind::Icmp { pred: IcmpPred::Eq, lhs, rhs } => {
                    assert_eq!(*lhs, cond);
                    assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 1, value: 0 }));
                }
                other => panic!("expected negated condition, got {other:?}"),
            }
        } else {
            assert_eq!(inner, cond);
        }
    }
}

#[test]
fn select_of_boolean_constants_stays_boolean() {
    let sig = Signature::new(vec![Ty::Bool], Ty::Bool);
    let (mut func, entry) = Function::new("f", sig);
    let cond = param(&mut func, entry, Ty::Bool, 0);
    let then_val = func.push_const(Const::Int { bits: 1, value: 1 });
    let else_val = func.push_const(Const::Int { bits: 1, value: 0 });
    let sel = emit(&mut func, entry, InstKind::Select { cond, then_val, else_val }, Ty::Bool);
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
    assert!(run(&mut func));
    assert_eq!(returned(&func), cond);

    let sig = Signature::new(vec![Ty::Bool], Ty::Bool);
    let (mut func, entry) = Function::new("f", sig);
    let cond = param(&mut func, entry, Ty::Bool, 0);
    let then_val = func.push_const(Const::Int { bits: 1, value: 0 });
    let else_val = func.push_const(Const::Int { bits: 1, value: 1 });
    let sel = emit(&mut func, entry, InstKind::Select { cond, then_val, else_val }, Ty::Bool);
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("negated boolean");
    match &func.insts[inst].kind {
        InstKind::Icmp { pred: IcmpPred::Eq, lhs, rhs } => {
            assert_eq!(*lhs, cond);
            assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 1, value: 0 }));
        }
        other => panic!("expected negated boolean, got {other:?}"),
    }
}

#[test]
fn xor_with_all_ones_becomes_not() {
    let (mut func, entry, x) = int_func();
    let ones = iconst(&mut func, 32, -1);
    let out = emit(&mut func, entry, InstKind::Xor(x, ones), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(out) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt negation");
    match &func.insts[inst].kind {
        InstKind::Not(operand) => assert_eq!(*operand, x),
        other => panic!("expected not, got {other:?}"),
    }
}

#[test]
fn multiply_by_a_power_of_two_shifts() {
    for constant_on_the_left in [false, true] {
        let (mut func, entry, x) = int_func();
        let eight = iconst(&mut func, 32, 8);
        let product = if constant_on_the_left {
            emit(&mut func, entry, InstKind::Mul(eight, x), Ty::int(32))
        } else {
            emit(&mut func, entry, InstKind::Mul(x, eight), Ty::int(32))
        };
        func.blocks[entry].term = Some(Terminator::Return { val: Some(product) });
        assert!(run(&mut func), "constant on left: {constant_on_the_left}");
        let inst = def_inst(&func, returned(&func)).expect("rebuilt shift");
        match &func.insts[inst].kind {
            InstKind::Shl(lhs, amount) => {
                assert_eq!(*lhs, x);
                assert_eq!(func.const_value(*amount), Some(&Const::Int { bits: 32, value: 3 }));
            }
            other => panic!("expected shift, got {other:?}"),
        }
    }
}

#[test]
fn multiply_by_minus_one_negates() {
    let (mut func, entry, x) = int_func();
    let minus_one = iconst(&mut func, 32, -1);
    let product = emit(&mut func, entry, InstKind::Mul(x, minus_one), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(product) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt negation");
    match &func.insts[inst].kind {
        InstKind::Neg(operand) => assert_eq!(*operand, x),
        other => panic!("expected negation, got {other:?}"),
    }
}

#[test]
fn zero_minus_x_negates() {
    let (mut func, entry, x) = int_func();
    let zero = iconst(&mut func, 32, 0);
    let difference = emit(&mut func, entry, InstKind::Sub(zero, x), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(difference) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt negation");
    match &func.insts[inst].kind {
        InstKind::Neg(operand) => assert_eq!(*operand, x),
        other => panic!("expected negation, got {other:?}"),
    }
}

#[test]
fn unsigned_division_by_a_power_of_two_shifts() {
    let (mut func, entry, x) = int_func();
    let eight = iconst(&mut func, 32, 8);
    let quotient = emit(&mut func, entry, InstKind::Udiv(x, eight), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(quotient) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt shift");
    match &func.insts[inst].kind {
        InstKind::LShr(lhs, amount) => {
            assert_eq!(*lhs, x);
            assert_eq!(func.const_value(*amount), Some(&Const::Int { bits: 32, value: 3 }));
        }
        other => panic!("expected logical shift, got {other:?}"),
    }
}

#[test]
fn unsigned_remainder_by_a_power_of_two_masks() {
    let (mut func, entry, x) = int_func();
    let eight = iconst(&mut func, 32, 8);
    let remainder = emit(&mut func, entry, InstKind::Urem(x, eight), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(remainder) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt mask");
    match &func.insts[inst].kind {
        InstKind::And(lhs, amount) => {
            assert_eq!(*lhs, x);
            assert_eq!(func.const_value(*amount), Some(&Const::Int { bits: 32, value: 7 }));
        }
        other => panic!("expected mask, got {other:?}"),
    }
}

#[test]
fn pointer_integer_round_trips_cancel() {
    let ptr_ty = Ty::ptr(Ty::int(8));
    let sig = Signature::new(vec![ptr_ty.clone()], ptr_ty.clone());
    let (mut func, entry) = Function::new("f", sig);
    let p = param(&mut func, entry, ptr_ty.clone(), 0);
    let int = emit(&mut func, entry, InstKind::PtrToInt(p), Ty::int(64));
    let back = emit(&mut func, entry, InstKind::IntToPtr(int), ptr_ty);
    func.blocks[entry].term = Some(Terminator::Return { val: Some(back) });
    assert!(run(&mut func));
    assert_eq!(returned(&func), p);

    let sig = Signature::new(vec![Ty::int(64)], Ty::int(64));
    let (mut func, entry) = Function::new("g", sig);
    let int = param(&mut func, entry, Ty::int(64), 0);
    let ptr_ty = Ty::ptr(Ty::int(8));
    let p = emit(&mut func, entry, InstKind::IntToPtr(int), ptr_ty);
    let back = emit(&mut func, entry, InstKind::PtrToInt(p), Ty::int(64));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(back) });
    assert!(run(&mut func));
    assert_eq!(returned(&func), int);
}

#[test]
fn negated_comparison_inverts_the_predicate() {
    let sig = Signature::new(vec![Ty::int(32), Ty::int(32)], Ty::Bool);
    let (mut func, entry) = Function::new("f", sig.clone());
    let a = param(&mut func, entry, Ty::int(32), 0);
    let b = param(&mut func, entry, Ty::int(32), 1);
    let less = emit(
        &mut func,
        entry,
        InstKind::Icmp {
            pred: IcmpPred::Slt,
            lhs: a,
            rhs: b,
        },
        Ty::Bool,
    );
    let zero = iconst(&mut func, 1, 0);
    let not_less = emit(
        &mut func,
        entry,
        InstKind::Icmp {
            pred: IcmpPred::Eq,
            lhs: less,
            rhs: zero,
        },
        Ty::Bool,
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(not_less) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt comparison");
    match &func.insts[inst].kind {
        InstKind::Icmp { pred, lhs, rhs } => {
            assert_eq!(*pred, IcmpPred::Sge);
            assert_eq!((*lhs, *rhs), (a, b));
        }
        other => panic!("expected comparison, got {other:?}"),
    }

    let (mut func, entry) = Function::new("g", sig);
    let a = param(&mut func, entry, Ty::int(32), 0);
    let b = param(&mut func, entry, Ty::int(32), 1);
    let less = emit(
        &mut func,
        entry,
        InstKind::Icmp {
            pred: IcmpPred::Slt,
            lhs: a,
            rhs: b,
        },
        Ty::Bool,
    );
    let zero = iconst(&mut func, 1, 0);
    let truthy = emit(
        &mut func,
        entry,
        InstKind::Icmp {
            pred: IcmpPred::Ne,
            lhs: less,
            rhs: zero,
        },
        Ty::Bool,
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(truthy) });
    assert!(run(&mut func));
    assert_eq!(returned(&func), less);
}

#[test]
fn utf8_ascii_bound_folds_to_signed_positive() {
    let sig = Signature::new(vec![Ty::int(8)], Ty::Bool);
    let (mut func, entry) = Function::new("f", sig);
    let x = param(&mut func, entry, Ty::int(8), 0);
    let wide = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
    let one = iconst(&mut func, 32, 1);
    let sub = emit(&mut func, entry, InstKind::Sub(wide, one), Ty::int(32));
    let narrow = emit(&mut func, entry, InstKind::Trunc(sub), Ty::int(8));
    let limit = iconst(&mut func, 8, 127);
    let cmp = emit(
        &mut func,
        entry,
        InstKind::Icmp {
            pred: IcmpPred::Ult,
            lhs: narrow,
            rhs: limit,
        },
        Ty::Bool,
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });
    assert!(run(&mut func));
    let inst = def_inst(&func, returned(&func)).expect("rebuilt comparison");
    match &func.insts[inst].kind {
        InstKind::Icmp { pred, lhs, rhs } => {
            assert_eq!(*pred, IcmpPred::Sgt);
            assert_eq!(*lhs, x);
            assert_eq!(func.const_value(*rhs), Some(&Const::Int { bits: 8, value: 0 }));
        }
        other => panic!("expected comparison, got {other:?}"),
    }
}

#[test]
fn leaves_unknown_operands_alone() {
    let sig = Signature::new(vec![Ty::int(32), Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let a = param(&mut func, entry, Ty::int(32), 0);
    let b = param(&mut func, entry, Ty::int(32), 1);
    let sum = emit(&mut func, entry, InstKind::Add(a, b), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
    assert!(!run(&mut func));
}
