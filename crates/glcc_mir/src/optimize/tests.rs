use super::*;
use crate::ir::{Block, Const, Def, Function, Inst, InstData, InstKind, Terminator, Value, ValueData};
use crate::ty::{Signature, Ty};

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

fn returned(func: &Function) -> Value {
    let Terminator::Return { val: Some(value) } = func.blocks[func.entry].term.clone().unwrap() else {
        panic!("expected a value return");
    };
    value
}

#[test]
fn pipeline_composes_constants_and_strength_reduction() {
    let mut module = Module::new();
    let constant = module.push_global("constant", Ty::int(32), Some(Const::Int { bits: 32, value: 4 }));
    module.globals[constant].local = true;
    let dead_global = module.push_global("unused", Ty::int(32), Some(Const::Int { bits: 32, value: 9 }));
    module.globals[dead_global].local = true;

    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("read_twice", sig.clone());
    let addr = emit(&mut func, entry, InstKind::GlobalAddr { global: constant }, Ty::ptr(Ty::int(32)));
    let left = emit(&mut func, entry, InstKind::Load { addr, flags: Default::default() }, Ty::int(32));
    let right = emit(&mut func, entry, InstKind::Load { addr, flags: Default::default() }, Ty::int(32));
    let sum = emit(&mut func, entry, InstKind::Add(left, right), Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
    module.push_func(func);

    let (mut dead, dead_entry) = Function::new("unused_function", sig);
    let zero = dead.push_const(Const::Int { bits: 32, value: 0 });
    dead.blocks[dead_entry].term = Some(Terminator::Return { val: Some(zero) });
    dead.local = true;
    module.push_func(dead);

    optimize(&mut module, OptLevel::Speed);

    assert_eq!(module.funcs.len(), 1);
    assert_eq!(module.funcs[0].name, "read_twice");
    assert!(module.globals.is_empty(), "the constant was fully propagated");
    assert_eq!(module.funcs[0].const_value(returned(&module.funcs[0])), Some(&Const::Int { bits: 32, value: 8 }));
    assert!(crate::verify::verify_module(&module).is_empty());
}

#[test]
fn pipeline_removes_inlined_and_propagated_internal_definitions() {
    let mut module = Module::new();
    let sig = Signature::new(vec![], Ty::int(32));

    let (mut helper, helper_entry) = Function::new("helper", sig.clone());
    let seven = helper.push_const(Const::Int { bits: 32, value: 7 });
    helper.blocks[helper_entry].term = Some(Terminator::Return { val: Some(seven) });
    helper.local = true;
    let helper_id = module.push_func(helper);

    let (mut caller, caller_entry) = Function::new("caller", sig.clone());
    let call = emit(
        &mut caller,
        caller_entry,
        InstKind::Call {
            callee: helper_id,
            args: Default::default(),
        },
        Ty::int(32),
    );
    caller.blocks[caller_entry].term = Some(Terminator::Return { val: Some(call) });
    module.push_func(caller);

    let seed = module.push_global("seed", Ty::int(32), Some(Const::Int { bits: 32, value: 17 }));
    module.globals[seed].local = true;
    let (mut reader, reader_entry) = Function::new("read_seed", sig);
    let addr = emit(&mut reader, reader_entry, InstKind::GlobalAddr { global: seed }, Ty::ptr(Ty::int(32)));
    let loaded = emit(&mut reader, reader_entry, InstKind::Load { addr, flags: Default::default() }, Ty::int(32));
    reader.blocks[reader_entry].term = Some(Terminator::Return { val: Some(loaded) });
    module.push_func(reader);

    optimize(&mut module, OptLevel::Speed);

    assert_eq!(module.funcs.len(), 2);
    assert!(module.globals.is_empty(), "the propagated static was dead");
    let caller = module.funcs.iter().find(|func| func.name == "caller").expect("caller survives");
    let reader = module.funcs.iter().find(|func| func.name == "read_seed").expect("reader survives");
    assert_eq!(caller.const_value(returned(caller)), Some(&Const::Int { bits: 32, value: 7 }));
    assert_eq!(reader.const_value(returned(reader)), Some(&Const::Int { bits: 32, value: 17 }));
    assert!(crate::verify::verify_module(&module).is_empty());
}
