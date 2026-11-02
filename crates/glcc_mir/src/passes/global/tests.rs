use super::*;
use crate::ir::{Block, Terminator};
use crate::ty::Signature;
use crate::verify::verify_module;

fn emit(func: &mut Function, block: Block, kind: InstKind, ty: Ty) -> Value {
    let result = func.values.push(ValueData {
        ty,
        def: Def::Inst(Inst::from_usize(func.insts.len())),
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

fn emit_void(func: &mut Function, block: Block, kind: InstKind) -> Inst {
    let inst = func.insts.push(crate::ir::InstData {
        block,
        kind,
        result: None,
        loc: None,
    });
    func.blocks[block].insts.push(inst);
    inst
}

fn int32_global(module: &mut Module, name: &str, value: i128) -> Global {
    let global = module.push_global(name, Ty::int(32), Some(Const::Int { bits: 32, value }));
    module.globals[global].local = true;
    global
}

fn load_global(module: &mut Module, global: Global) -> (Func, Value) {
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("read", sig);
    let addr = emit(&mut func, entry, InstKind::GlobalAddr { global }, Ty::ptr(Ty::int(32)));
    let value = emit(&mut func, entry, InstKind::Load { addr, flags: Default::default() }, Ty::int(32));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(value) });
    let func = module.push_func(func);
    (func, value)
}

fn returned(func: &Function) -> Value {
    let Terminator::Return { val: Some(value) } = func.blocks[func.entry].term.clone().unwrap() else {
        panic!("expected a value return");
    };
    value
}

#[test]
fn propagates_read_only_local_initializers() {
    let mut module = Module::new();
    let global = int32_global(&mut module, "c", 42);
    let (func, _) = load_global(&mut module, global);

    assert!(GlobalConstProp.run(&mut module));
    assert_eq!(module.funcs[func].const_value(returned(&module.funcs[func])), Some(&Const::Int { bits: 32, value: 42 }));
    assert!(
        module.funcs[func].blocks[module.funcs[func].entry]
            .insts
            .iter()
            .all(|inst| !matches!(module.funcs[func].insts[*inst].kind, InstKind::Load { .. }))
    );
    assert!(verify_module(&module).is_empty());
    assert!(!GlobalConstProp.run(&mut module), "second run is a no-op");
}

#[test]
fn merges_identical_read_only_globals() {
    let mut module = Module::new();
    let ty = Ty::Array {
        elem: Box::new(Ty::int(8)),
        len: 5,
    };
    let first = module.push_global("__str.1", ty.clone(), Some(Const::Bytes(b"same\0".to_vec())));
    module.globals[first].local = true;
    module.globals[first].writable = false;
    module.globals[first].align = Some(1);
    let second = module.push_global("__str.2", ty, Some(Const::Bytes(b"same\0".to_vec())));
    module.globals[second].local = true;
    module.globals[second].writable = false;
    module.globals[second].align = Some(8);

    for (name, global) in [("one", first), ("two", second)] {
        let (mut func, entry) = Function::new(name, Signature::new(vec![], Ty::ptr(Ty::int(8))));
        let addr = emit(&mut func, entry, InstKind::GlobalAddr { global }, Ty::ptr(module.globals[global].ty.clone()));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(addr) });
        module.push_func(func);
    }

    assert!(MergeReadOnlyGlobals.run(&mut module));
    assert_eq!(module.globals.len(), 1);
    assert_eq!(module.globals[Global::from_usize(0)].align, Some(8));
    for func in module.funcs.iter() {
        let entry = func.entry;
        let addr = func.blocks[entry].insts[0];
        assert!(matches!(
            func.insts[addr].kind,
            InstKind::GlobalAddr { global } if global == Global::from_usize(0)
        ));
    }
    assert!(verify_module(&module).is_empty());
    assert!(!MergeReadOnlyGlobals.run(&mut module), "second run is a no-op");
}

#[test]
fn keeps_writable_globals_separate() {
    let mut module = Module::new();
    let ty = Ty::Array {
        elem: Box::new(Ty::int(8)),
        len: 5,
    };
    for name in ["one", "two"] {
        let global = module.push_global(name, ty.clone(), Some(Const::Bytes(b"same\0".to_vec())));
        module.globals[global].local = true;
        module.globals[global].align = Some(1);
    }
    assert!(!MergeReadOnlyGlobals.run(&mut module));
    assert_eq!(module.globals.len(), 2);
}

fn load_at_byte_offset(module: &mut Module, global: Global, offset: i128) -> (Func, Value) {
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new("read_at", sig);
    let pointee = module.globals[global].ty.clone();
    let addr = emit(&mut func, entry, InstKind::GlobalAddr { global }, Ty::ptr(pointee));
    let base = emit(&mut func, entry, InstKind::PtrToInt(addr), Ty::int(64));
    let four = func.push_const(Const::Int { bits: 64, value: offset });
    let offset_addr = emit(&mut func, entry, InstKind::Add(base, four), Ty::int(64));
    let ptr = emit(&mut func, entry, InstKind::IntToPtr(offset_addr), Ty::ptr(Ty::int(32)));
    let value = emit(
        &mut func,
        entry,
        InstKind::Load {
            addr: ptr,
            flags: Default::default(),
        },
        Ty::int(32),
    );
    func.blocks[entry].term = Some(Terminator::Return { val: Some(value) });
    let func = module.push_func(func);
    (func, value)
}

#[test]
fn propagates_byte_initializers_through_constant_offsets() {
    let mut module = Module::new();
    let ty = Ty::Array {
        elem: Box::new(Ty::int(32)),
        len: 2,
    };
    let init = Const::Bytes(vec![1, 0, 0, 0, 2, 0, 0, 0]);
    let global = module.push_global("table", ty, Some(init));
    module.globals[global].local = true;
    module.globals[global].writable = false;
    let (func, _) = load_at_byte_offset(&mut module, global, 4);

    assert!(GlobalConstProp.run(&mut module));
    assert_eq!(module.funcs[func].const_value(returned(&module.funcs[func])), Some(&Const::Int { bits: 32, value: 2 }));
    assert!(
        module.funcs[func].blocks[module.funcs[func].entry]
            .insts
            .iter()
            .all(|inst| !matches!(module.funcs[func].insts[*inst].kind, InstKind::Load { .. }))
    );
    assert!(verify_module(&module).is_empty());
}

fn struct_with_two_ints() -> Ty {
    Ty::Struct {
        fields: vec![Ty::int(32), Ty::int(32)],
        offsets: vec![0, 4],
        size: 8,
    }
}

fn copy_global_into_local(module: &mut Module, global: Global, ty: Ty) -> Func {
    let (mut func, entry) = Function::new("copy", Signature::new(vec![], Ty::int(32)));
    let dst = emit(&mut func, entry, InstKind::Alloca { pointee: ty.clone(), count: 1 }, Ty::ptr(ty.clone()));
    let src = emit(&mut func, entry, InstKind::GlobalAddr { global }, Ty::ptr(ty.clone()));
    emit_void(&mut func, entry, InstKind::Memcpy { dst, src, len: ty.size_bytes() });
    let zero = func.push_const(Const::Int { bits: 32, value: 0 });
    func.blocks[entry].term = Some(Terminator::Return { val: Some(zero) });
    module.push_func(func)
}

fn stored_values(func: &Function) -> Vec<(Value, Value)> {
    func.blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter_map(|inst| match func.insts[*inst].kind {
            InstKind::Store { addr, val, .. } => Some((addr, val)),
            _ => None,
        })
        .collect()
}

#[test]
fn expands_copies_from_read_only_initializers() {
    let mut module = Module::new();
    let ty = struct_with_two_ints();
    let global = module.push_global("pair", ty.clone(), Some(Const::Bytes(vec![1, 0, 0, 0, 2, 0, 0, 0])));
    module.globals[global].local = true;
    module.globals[global].writable = false;
    let func = copy_global_into_local(&mut module, global, ty);
    let local_inst = module.funcs[func].blocks[module.funcs[func].entry].insts[0];
    let local = module.funcs[func].insts[local_inst].result.expect("alloca result");

    assert!(GlobalConstProp.run(&mut module));
    let func = &module.funcs[func];
    assert!(
        func.blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .all(|inst| !matches!(func.insts[*inst].kind, InstKind::Memcpy { .. }))
    );
    let stores = stored_values(func);
    assert_eq!(stores.len(), 2);
    for (index, (addr, value)) in stores.into_iter().enumerate() {
        let Def::Inst(addr_inst) = func.values[addr].def else {
            panic!("field store address is a GEP");
        };
        match &func.insts[addr_inst].kind {
            InstKind::GetElementPtr { base, indices } => {
                assert_eq!(*base, local);
                assert_eq!(indices.as_slice(), &[GepIndex::Field(index as u32)]);
            }
            other => panic!("field store goes through a field GEP, got {other:?}"),
        }
        assert_eq!(
            func.const_value(value),
            Some(&Const::Int {
                bits: 32,
                value: index as i128 + 1
            })
        );
    }
    assert!(verify_module(&module).is_empty());
    assert!(!GlobalConstProp.run(&mut module), "second run is a no-op");
}

#[test]
fn expands_zero_initialized_copies() {
    let mut module = Module::new();
    let ty = struct_with_two_ints();
    let global = module.push_global("zeroed", ty.clone(), Some(Const::Zero));
    module.globals[global].local = true;
    module.globals[global].writable = false;
    let func = copy_global_into_local(&mut module, global, ty);

    assert!(GlobalConstProp.run(&mut module));
    let func = &module.funcs[func];
    let stores = stored_values(func);
    assert_eq!(stores.len(), 2);
    assert!(stores.into_iter().all(|(_, value)| func.const_value(value) == Some(&Const::Int { bits: 32, value: 0 })));
    assert!(verify_module(&module).is_empty());
}

#[test]
fn keeps_copies_over_relocated_initializers() {
    let mut module = Module::new();
    let pointer = Ty::ptr(Ty::int(32));
    let ty = Ty::Struct {
        fields: vec![pointer.clone(), Ty::int(32)],
        offsets: vec![0, 8],
        size: 16,
    };
    let target = int32_global(&mut module, "target", 7);
    let global = module.push_global("relocated", ty.clone(), Some(Const::Bytes(vec![0; 16])));
    module.globals[global].local = true;
    module.globals[global].writable = false;
    module.globals[global].relocs.push((0, DataReloc::Global { target, addend: 0 }));
    let func = copy_global_into_local(&mut module, global, ty);

    assert!(!GlobalConstProp.run(&mut module));
    assert!(stored_values(&module.funcs[func]).is_empty());
    assert!(
        module.funcs[func]
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .any(|inst| matches!(module.funcs[func].insts[*inst].kind, InstKind::Memcpy { .. }))
    );
}

#[test]
fn does_not_fold_loads_overlapping_relocations() {
    let mut module = Module::new();
    let ty = Ty::Array {
        elem: Box::new(Ty::int(32)),
        len: 2,
    };
    let global = module.push_global("relocated", ty, Some(Const::Bytes(vec![0; 8])));
    module.globals[global].local = true;
    module.globals[global].writable = false;
    let target = int32_global(&mut module, "target", 7);
    module.globals[global].relocs.push((4, DataReloc::Global { target, addend: 0 }));
    let (func, value) = load_at_byte_offset(&mut module, global, 4);

    assert!(!GlobalConstProp.run(&mut module));
    assert_eq!(returned(&module.funcs[func]), value);
}

#[test]
fn keeps_loads_from_written_storage() {
    let mut module = Module::new();
    let global = int32_global(&mut module, "c", 42);
    let (func, value) = load_global(&mut module, global);
    let entry = module.funcs[func].entry;
    let addr = emit(&mut module.funcs[func], entry, InstKind::GlobalAddr { global }, Ty::ptr(Ty::int(32)));
    let nine = module.funcs[func].push_const(Const::Int { bits: 32, value: 9 });
    emit_void(
        &mut module.funcs[func],
        entry,
        InstKind::Store {
            addr,
            val: nine,
            flags: Default::default(),
        },
    );
    assert!(!GlobalConstProp.run(&mut module));
    assert_eq!(returned(&module.funcs[func]), value);
}

#[test]
fn keeps_loads_when_the_address_escapes() {
    let mut module = Module::new();
    let global = int32_global(&mut module, "c", 42);
    let (func, value) = load_global(&mut module, global);
    let entry = module.funcs[func].entry;
    let addr = emit(&mut module.funcs[func], entry, InstKind::GlobalAddr { global }, Ty::ptr(Ty::int(32)));
    emit_void(&mut module.funcs[func], entry, InstKind::Memcpy { dst: addr, src: addr, len: 0 });
    assert!(!GlobalConstProp.run(&mut module));
    assert_eq!(returned(&module.funcs[func]), value);
}

#[test]
fn keeps_loads_from_external_storage() {
    let mut module = Module::new();
    let global = int32_global(&mut module, "c", 42);
    module.globals[global].local = false;
    let (func, value) = load_global(&mut module, global);
    assert!(!GlobalConstProp.run(&mut module));
    assert_eq!(returned(&module.funcs[func]), value);
}

fn void_function(module: &mut Module, name: &str, local: bool, value: i128) -> Func {
    let sig = Signature::new(vec![], Ty::int(32));
    let (mut func, entry) = Function::new(name, sig);
    let c = func.push_const(Const::Int { bits: 32, value });
    func.blocks[entry].term = Some(Terminator::Return { val: Some(c) });
    func.local = local;
    module.push_func(func)
}

#[test]
fn orphaned_references_do_not_keep_internal_definitions() {
    let mut module = Module::new();
    let dead = void_function(&mut module, "dead_fn", true, 0);
    let keep = void_function(&mut module, "keep", false, 0);
    let entry = module.funcs[keep].entry;
    let call = emit(
        &mut module.funcs[keep],
        entry,
        InstKind::Call {
            callee: dead,
            args: Default::default(),
        },
        Ty::int(32),
    );
    let call_inst = match &module.funcs[keep].values[call].def {
        Def::Inst(inst) => *inst,
        other => panic!("call result is an instruction, got {other:?}"),
    };
    module.funcs[keep].blocks[entry].insts.retain(|inst| *inst != call_inst);

    assert!(DeadModuleItems.run(&mut module));
    assert_eq!(module.funcs.len(), 1);
    assert_eq!(module.funcs[Func::from_usize(0)].name, "keep");
    assert!(verify_module(&module).is_empty());
}

#[test]
fn drops_unreferenced_internal_definitions() {
    let mut module = Module::new();
    let dead_global = int32_global(&mut module, "dead", 1);
    let live_global = int32_global(&mut module, "live", 7);

    let dead_func = void_function(&mut module, "dead_fn", true, 0);
    let live_func = void_function(&mut module, "live_fn", true, 0);
    let keep = void_function(&mut module, "keep", false, 0);

    let live_entry = module.funcs[live_func].entry;
    let addr = emit(&mut module.funcs[live_func], live_entry, InstKind::GlobalAddr { global: live_global }, Ty::ptr(Ty::int(32)));
    let loaded = emit(&mut module.funcs[live_func], live_entry, InstKind::Load { addr, flags: Default::default() }, Ty::int(32));
    module.funcs[live_func].blocks[live_entry].term = Some(Terminator::Return { val: Some(loaded) });

    let keep_entry = module.funcs[keep].entry;
    let call = emit(
        &mut module.funcs[keep],
        keep_entry,
        InstKind::Call {
            callee: live_func,
            args: Default::default(),
        },
        Ty::int(32),
    );
    module.funcs[keep].blocks[keep_entry].term = Some(Terminator::Return { val: Some(call) });

    assert!(DeadModuleItems.run(&mut module));
    assert_eq!(module.funcs.len(), 2);
    assert_eq!(module.funcs[Func::from_usize(0)].name, "live_fn");
    assert_eq!(module.funcs[Func::from_usize(1)].name, "keep");
    assert_eq!(module.globals.len(), 1);
    assert_eq!(module.globals[Global::from_usize(0)].name, "live");

    let keep_entry = module.funcs[Func::from_usize(1)].entry;
    let Some(InstKind::Call { callee, .. }) = module.funcs[Func::from_usize(1)].blocks[keep_entry]
        .insts
        .first()
        .map(|i| &module.funcs[Func::from_usize(1)].insts[*i].kind)
    else {
        panic!("call was not remapped");
    };
    assert_eq!(*callee, Func::from_usize(0));

    let live_entry = module.funcs[Func::from_usize(0)].entry;
    let Some(InstKind::GlobalAddr { global }) = module.funcs[Func::from_usize(0)].blocks[live_entry]
        .insts
        .first()
        .map(|i| &module.funcs[Func::from_usize(0)].insts[*i].kind)
    else {
        panic!("global address was not remapped");
    };
    assert_eq!(*global, Global::from_usize(0));

    assert!(verify_module(&module).is_empty());
    assert!(!DeadModuleItems.run(&mut module), "second run is a no-op");
    let _ = (dead_func, dead_global);
}

#[test]
fn initializer_references_keep_internal_functions() {
    let mut module = Module::new();
    let target = void_function(&mut module, "target", true, 0);
    let slot = module.push_global("slot", Ty::int(64), Some(Const::Zero));
    module.globals[slot].relocs.push((0, DataReloc::Func { target }));

    assert!(!DeadModuleItems.run(&mut module));
    assert_eq!(module.funcs.len(), 1);
    assert_eq!(module.globals.len(), 1);
    let DataReloc::Func { target } = module.globals[slot].relocs[0].1 else {
        panic!("function relocation disappeared");
    };
    assert_eq!(target, Func::from_usize(0));
}

#[test]
fn inline_assembly_disables_removal() {
    let mut module = Module::new();
    let dead = void_function(&mut module, "dead", true, 0);
    let entry = module.funcs[dead].entry;
    emit_void(
        &mut module.funcs[dead],
        entry,
        InstKind::InlineAsm {
            block: Box::new(crate::ir::AsmBlock {
                template: String::new(),
                operands: Vec::new(),
                outputs: 0,
                clobbers: Vec::new(),
                labels: Vec::new(),
                is_volatile: false,
                is_basic: true,
            }),
        },
    );

    assert!(!DeadModuleItems.run(&mut module));
    assert_eq!(module.funcs.len(), 1);
}

#[test]
fn init_functions_are_roots_and_get_remapped() {
    let mut module = Module::new();
    let dead = void_function(&mut module, "dead_fn", true, 0);
    let (mut ctor_func, ctor_entry) = Function::new("__sti_ctor", Signature::new(vec![], Ty::Void));
    ctor_func.blocks[ctor_entry].term = Some(Terminator::Return { val: None });
    ctor_func.local = true;
    let ctor = module.push_func(ctor_func);
    let keep = void_function(&mut module, "keep", false, 0);
    module.init_funcs.push(ctor);

    assert!(DeadModuleItems.run(&mut module));
    assert_eq!(module.funcs.len(), 2, "the unreferenced local constructor survives");
    let names: Vec<&str> = module.funcs.iter().map(|func| func.name.as_str()).collect();
    assert_eq!(names, vec!["__sti_ctor", "keep"]);
    assert_eq!(module.init_funcs, vec![Func::from_usize(0)], "init entries follow the renumbered functions");
    assert!(verify_module(&module).is_empty());
    assert!(!DeadModuleItems.run(&mut module), "second run is a no-op");
    let _ = keep;
    let _ = dead;
}
