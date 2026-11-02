use super::*;
use crate::ir::{Block, Const, Def, InstData, Terminator, ValueData};
use crate::ty::{FloatKind, Signature};
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
    for block in func.blocks.iter() {
        match &block.term {
            Some(Terminator::Return { val: Some(v) }) => return *v,
            _ => {}
        }
    }
    panic!("no value return in {func:?}");
}

fn run(func: &mut Function) -> bool {
    let changed = Cse.run(func);
    let errors = verify_func(func);
    assert!(errors.is_empty(), "verifier: {errors:?}");
    changed
}

fn block_insts(func: &Function, block: Block) -> Vec<Inst> {
    func.blocks[block].insts.clone()
}

fn int_func() -> (Function, Block, Value, Value) {
    let sig = Signature::new(vec![Ty::int(32), Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let a = param(&mut func, entry, Ty::int(32), 0);
    let b = param(&mut func, entry, Ty::int(32), 1);
    (func, entry, a, b)
}

#[test]
fn merges_duplicate_in_one_block() {
    let (mut func, entry, a, b) = int_func();
    let first = emit(&mut func, entry, InstKind::Add(a, b), Ty::int(32));
    let second = emit(&mut func, entry, InstKind::Add(a, b), Ty::int(32));
    let sum = emit(&mut func, entry, InstKind::Add(first, second), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
    assert!(run(&mut func));
    // Both operands of the sum now name the first addition.
    let inst = crate::ir::Inst::from_usize(0);
    let _ = inst;
    let Some(inst) = def_inst(&func, sum) else {
        panic!("sum has no defining instruction");
    };
    match &func.insts[inst].kind {
        InstKind::Add(x, y) => {
            assert_eq!(*x, first);
            assert_eq!(*y, first);
        }
        other => panic!("expected add, got {other:?}"),
    }
    assert_eq!(block_insts(&func, entry).len(), 2, "duplicate must leave the block");
    assert!(!run(&mut func), "second run is a no-op");
}

#[test]
fn merges_commutative_operands() {
    let (mut func, entry, a, b) = int_func();
    let first = emit(&mut func, entry, InstKind::Add(a, b), Ty::int(32));
    let second = emit(&mut func, entry, InstKind::Add(b, a), Ty::int(32));
    let sum = emit(&mut func, entry, InstKind::Add(first, second), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
    assert!(run(&mut func));
    let Some(inst) = def_inst(&func, sum) else {
        panic!("sum has no defining instruction");
    };
    match &func.insts[inst].kind {
        InstKind::Add(x, y) => assert_eq!(*x, *y),
        other => panic!("expected add, got {other:?}"),
    }
}

#[test]
fn keeps_ordered_comparisons_apart() {
    // `a < b` and `b < a` differ, so they must not merge.
    let (mut func, entry, a, b) = int_func();
    let lt = emit(
        &mut func,
        entry,
        InstKind::Icmp {
            pred: IcmpPred::Slt,
            lhs: a,
            rhs: b,
        },
        Ty::Bool,
    );
    let gt = emit(
        &mut func,
        entry,
        InstKind::Icmp {
            pred: IcmpPred::Slt,
            lhs: b,
            rhs: a,
        },
        Ty::Bool,
    );
    let sum = emit(&mut func, entry, InstKind::Xor(lt, gt), Ty::Bool);
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
    assert!(!run(&mut func));
}

#[test]
fn merges_across_dominating_blocks() {
    let (mut func, entry, a, b) = int_func();
    let first = emit(&mut func, entry, InstKind::Add(a, b), Ty::int(32));
    let body = func.push_block(vec![]);
    func.blocks[entry].term = Some(Terminator::Br {
        dest: body,
        args: Default::default(),
    });
    let second = emit(&mut func, body, InstKind::Add(a, b), Ty::int(32));
    func.blocks[body].term = Some(Terminator::Return { val: Some(second) });
    assert!(run(&mut func));
    assert_eq!(returned(&func), first);
    assert!(block_insts(&func, body).is_empty());
}

#[test]
fn keeps_sibling_branch_expressions_apart() {
    // Neither arm dominates the other, so both additions stay.
    let sig = Signature::new(vec![Ty::int(32), Ty::int(32), Ty::Bool], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let a = param(&mut func, entry, Ty::int(32), 0);
    let b = param(&mut func, entry, Ty::int(32), 1);
    let cond = param(&mut func, entry, Ty::Bool, 2);
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
    let t1 = emit(&mut func, then_block, InstKind::Add(a, b), Ty::int(32));
    func.blocks[then_block].term = Some(Terminator::Br {
        dest: merge,
        args: [t1].into_iter().collect(),
    });
    let t2 = emit(&mut func, else_block, InstKind::Add(a, b), Ty::int(32));
    func.blocks[else_block].term = Some(Terminator::Br {
        dest: merge,
        args: [t2].into_iter().collect(),
    });
    let merged = func.blocks[merge].params[0];
    func.blocks[merge].term = Some(Terminator::Return { val: Some(merged) });
    assert!(!run(&mut func));
    assert_eq!(block_insts(&func, then_block).len(), 1);
    assert_eq!(block_insts(&func, else_block).len(), 1);
}

#[test]
fn does_not_merge_memory_reads() {
    // A load sees whatever the last store left; only one alias
    // analysis could say the two loads agree.
    let sig = Signature::new(vec![Ty::ptr(Ty::int(32))], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let p = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
    let l1 = emit(
        &mut func,
        entry,
        InstKind::Load {
            addr: p,
            flags: crate::ir::MemFlags::plain(),
        },
        Ty::int(32),
    );
    let l2 = emit(
        &mut func,
        entry,
        InstKind::Load {
            addr: p,
            flags: crate::ir::MemFlags::plain(),
        },
        Ty::int(32),
    );
    let sum = emit(&mut func, entry, InstKind::Add(l1, l2), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
    assert!(!run(&mut func));
}

#[test]
fn does_not_merge_va_args() {
    // Each `VaArg` advances the list; merging two would drop one
    // advance.
    let ap_ty = Ty::BuiltinVaList { size: 24, align: 8 };
    let sig = Signature::new(vec![Ty::ptr(ap_ty.clone())], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let ap = param(&mut func, entry, Ty::ptr(ap_ty), 0);
    let v1 = emit(&mut func, entry, InstKind::VaArg { ap }, Ty::int(32));
    let v2 = emit(&mut func, entry, InstKind::VaArg { ap }, Ty::int(32));
    let sum = emit(&mut func, entry, InstKind::Add(v1, v2), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
    assert!(!run(&mut func));
}

#[test]
fn merges_identical_geps() {
    let sig = Signature::new(vec![Ty::ptr(Ty::int(32)), Ty::int(32)], Ty::int(32));
    let (mut func, entry) = Function::new("f", sig);
    let base = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
    let index = param(&mut func, entry, Ty::int(32), 1);
    let indices: SmallVec<[GepIndex; 4]> = [GepIndex::Index(index)].into_iter().collect();
    let g1 = emit(&mut func, entry, InstKind::GetElementPtr { base, indices: indices.clone() }, Ty::ptr(Ty::int(32)));
    let g2 = emit(&mut func, entry, InstKind::GetElementPtr { base, indices }, Ty::ptr(Ty::int(32)));
    let p1 = emit(&mut func, entry, InstKind::PtrToInt(g1), Ty::int(64));
    let p2 = emit(&mut func, entry, InstKind::PtrToInt(g2), Ty::int(64));
    let diff = emit(&mut func, entry, InstKind::Sub(p1, p2), Ty::int(64));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(diff) });
    assert!(run(&mut func));
    // Both ptrtoint's now name the first GEP, and the subtraction
    // of a value with itself dies to inst-combine/DCE later.
    let Some(sub) = def_inst(&func, diff) else {
        panic!("difference has no defining instruction");
    };
    match &func.insts[sub].kind {
        InstKind::Sub(x, y) => {
            assert_eq!(*x, *y, "both sides must share the first GEP's address");
            let x = def_inst(&func, *x).expect("ptrtoint");
            match &func.insts[x].kind {
                InstKind::PtrToInt(g) => assert_eq!(*g, g1),
                other => panic!("expected ptrtoint, got {other:?}"),
            }
        }
        other => panic!("expected sub, got {other:?}"),
    }
}

#[test]
fn merges_selects_and_unary_ops() {
    let (mut func, entry, a, b) = int_func();
    let cond = iconst(&mut func, 1, 1);
    let s1 = emit(&mut func, entry, InstKind::Select { cond, then_val: a, else_val: b }, Ty::int(32));
    let neg1 = emit(&mut func, entry, InstKind::Neg(a), Ty::int(32));
    let neg2 = emit(&mut func, entry, InstKind::Neg(a), Ty::int(32));
    let s2 = emit(&mut func, entry, InstKind::Select { cond, then_val: a, else_val: b }, Ty::int(32));
    let sum1 = emit(&mut func, entry, InstKind::Add(s1, s2), Ty::int(32));
    let sum = emit(&mut func, entry, InstKind::Add(sum1, neg1), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
    assert!(run(&mut func));
    assert_eq!(block_insts(&func, entry).len(), 4, "one select, one neg, two adds");
    let _ = neg2;
}

#[test]
fn distinguishes_result_types() {
    // `zext x` to i32 and to i64 are different expressions.
    let sig = Signature::new(vec![Ty::int(8)], Ty::int(64));
    let (mut func, entry) = Function::new("f", sig);
    let x = param(&mut func, entry, Ty::int(8), 0);
    let z32 = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
    let z64 = emit(&mut func, entry, InstKind::Zext(x), Ty::int(64));
    let wide = emit(&mut func, entry, InstKind::Zext(z32), Ty::int(64));
    let sum = emit(&mut func, entry, InstKind::Add(wide, z64), Ty::int(64));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
    assert!(!run(&mut func));
}

#[test]
fn merges_float_binops_commutatively() {
    let sig = Signature::new(vec![Ty::Float(FloatKind::F64), Ty::Float(FloatKind::F64)], Ty::Float(FloatKind::F64));
    let (mut func, entry) = Function::new("f", sig);
    let a = param(&mut func, entry, Ty::Float(FloatKind::F64), 0);
    let b = param(&mut func, entry, Ty::Float(FloatKind::F64), 1);
    let m1 = emit(&mut func, entry, InstKind::FMul(a, b), Ty::Float(FloatKind::F64));
    let m2 = emit(&mut func, entry, InstKind::FMul(b, a), Ty::Float(FloatKind::F64));
    let sum = emit(&mut func, entry, InstKind::FAdd(m1, m2), Ty::Float(FloatKind::F64));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
    assert!(run(&mut func));
    let Some(inst) = def_inst(&func, sum) else {
        panic!("sum has no defining instruction");
    };
    match &func.insts[inst].kind {
        InstKind::FAdd(x, y) => assert_eq!(*x, *y),
        other => panic!("expected fadd, got {other:?}"),
    }
}

fn def_inst(func: &Function, v: Value) -> Option<Inst> {
    match &func.values[v].def {
        Def::Inst(i) if i.index() < func.insts.len() => Some(*i),
        _ => None,
    }
}
