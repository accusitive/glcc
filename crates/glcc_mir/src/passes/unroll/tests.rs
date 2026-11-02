use super::*;
use crate::ir::{Const, Func, InstData, InstKind, Module, ValueData};
use crate::optimize::{OptLevel, optimize};
use crate::ty::{Signature, Ty};

fn emit(func: &mut Function, block: Block, kind: InstKind, ty: Ty) -> Value {
    let result = func.values.push(ValueData {
        ty,
        def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
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

fn counted_loop(start: i128, bound: i128) -> (Module, Func) {
    counted_loop_with(start, bound, true)
}

/// `body_on_true` selects which header successor is the loop body:
/// the usual `i < bound` continue test, or the goto shape whose
/// condition is the exit test (`i >= bound` with the body as the
/// else-successor).
fn counted_loop_with(start: i128, bound: i128, body_on_true: bool) -> (Module, Func) {
    let mut module = Module::new();
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("main", sig);
    let header = func.push_block(vec![Ty::int(32), Ty::int(32)]);
    let body = func.push_block(vec![]);
    let exit = func.push_block(vec![Ty::int(32)]);
    let sum = func.blocks[header].params[0];
    let i = func.blocks[header].params[1];

    let zero = func.push_const(Const::Int { bits: 32, value: 0 });
    let start = func.push_const(Const::Int { bits: 32, value: start });
    func.blocks[entry].term = Some(Terminator::Br {
        dest: header,
        args: [start, zero].into_iter().collect(),
    });

    let bound = func.push_const(Const::Int { bits: 32, value: bound });
    let body_args: SmallVec<[Value; 4]> = Default::default();
    let exit_args: SmallVec<[Value; 4]> = [sum].into_iter().collect();
    let (pred, then_block, then_args, else_block, else_args) = if body_on_true {
        (IcmpPred::Slt, body, body_args, exit, exit_args)
    } else {
        (IcmpPred::Sge, exit, exit_args, body, body_args)
    };
    let cond = emit(
        &mut func,
        header,
        InstKind::Icmp {
            pred,
            lhs: i,
            rhs: bound,
        },
        Ty::Bool,
    );
    func.blocks[header].term = Some(Terminator::CondBr {
        cond,
        then_block,
        else_block,
        then_args,
        else_args,
    });

    let one = func.push_const(Const::Int { bits: 32, value: 1 });
    let next_sum = emit(&mut func, body, InstKind::Add(sum, one), Ty::int(32));
    let next_i = emit(&mut func, body, InstKind::Add(i, one), Ty::int(32));
    func.blocks[body].term = Some(Terminator::Br {
        dest: header,
        args: [next_sum, next_i].into_iter().collect(),
    });

    let result = func.blocks[exit].params[0];
    func.blocks[exit].term = Some(Terminator::Return { val: Some(result) });
    let func = module.push_func(func);
    (module, func)
}

#[test]
fn a_small_counted_loop_folds_to_its_sum() {
    let (mut module, func) = counted_loop(100, 2);
    optimize(&mut module, OptLevel::Speed);
    let main = &module.funcs[func];
    let Terminator::Return { val: Some(value) } = main.blocks[main.entry].term.clone().unwrap() else {
        panic!("expected a value return");
    };
    assert_eq!(main.const_value(value), Some(&Const::Int { bits: 32, value: 102 }));
    assert!(crate::verify::verify_module(&module).is_empty());
}

#[test]
fn a_counted_loop_whose_header_tests_the_exit_folds_to_its_sum() {
    let (mut module, func) = counted_loop_with(100, 2, false);
    optimize(&mut module, OptLevel::Speed);
    let main = &module.funcs[func];
    let Terminator::Return { val: Some(value) } = main.blocks[main.entry].term.clone().unwrap() else {
        panic!("expected a value return");
    };
    assert_eq!(main.const_value(value), Some(&Const::Int { bits: 32, value: 102 }));
    assert!(crate::verify::verify_module(&module).is_empty());
}

#[test]
fn a_large_counted_loop_stays_rolled() {
    let (mut module, func) = counted_loop(0, 100);
    let mut pass = LoopUnroll;
    assert!(!pass.run(&mut module.funcs[func]));
    assert!(crate::verify::verify_module(&module).is_empty());
}
