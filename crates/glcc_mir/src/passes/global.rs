//! Module-level constant propagation and dead definition removal.
//!
//! [`GlobalConstProp`] replaces loads from read-only local storage with the
//! initializer's value. A local object qualifies only when no instruction
//! writes to it and its address never escapes; inline assembly disables the
//! rewrite because its symbol references are invisible in the IR.
//!
//! [`DeadModuleItems`] removes internal functions and objects that no live
//! definition references. External-linkage definitions are roots. Function
//! and object liveness is solved together because a live object's
//! initializer can reference a function, and a live function can address an
//! object.

use std::collections::HashMap;

use index_vec::IndexVec;

use crate::debug::{FunctionDebug, ScopeDebug, VarHome};
use crate::ir::{Const, DataReloc, Def, Func, Function, Global, Inst, InstKind, Module, Value, ValueData};
use crate::pass::ModulePass;
use crate::ty::Ty;

/// Replace loads from read-only local storage with their initializer.
pub struct GlobalConstProp;

impl ModulePass for GlobalConstProp {
    fn name(&self) -> &'static str {
        "global-const-prop"
    }

    fn run(&mut self, module: &mut Module) -> bool {
        if has_inline_asm(module) {
            return false;
        }

        let nglobals = module.globals.len();
        let mut written = vec![false; nglobals];
        let mut escaped = vec![false; nglobals];
        for global in module.globals.iter() {
            for reloc in global.relocs.iter().map(|(_, r)| r).chain(global.relocs32.iter().map(|(_, r)| r)) {
                if let DataReloc::Global { target, .. } = reloc {
                    escaped[target.index()] = true;
                }
            }
        }

        let mut sites: Vec<(usize, Inst, Global)> = Vec::new();
        for (func_id, func) in module.funcs.iter().enumerate() {
            let resident = func.resident_mask();
            let addresses = global_addresses(func, &resident);
            for (inst_id, inst) in func.insts.iter_enumerated() {
                if !resident[inst_id.index()] {
                    continue;
                }
                match &inst.kind {
                    InstKind::Load { addr, flags } => {
                        if !flags.volatile {
                            if let Some(global) = addresses.get(addr) {
                                sites.push((func_id, inst_id, *global));
                            }
                        }
                    }
                    InstKind::Store { addr, val, .. } => {
                        if let Some(global) = addresses.get(addr) {
                            written[global.index()] = true;
                        }
                        if let Some(global) = addresses.get(val) {
                            escaped[global.index()] = true;
                        }
                    }
                    kind => kind.for_each_operand(&mut |value| {
                        if let Some(global) = addresses.get(&value) {
                            escaped[global.index()] = true;
                        }
                    }),
                }
            }
            for block in func.blocks.iter() {
                if let Some(term) = &block.term {
                    term.for_each_operand(&mut |value| {
                        if let Some(global) = addresses.get(&value) {
                            escaped[global.index()] = true;
                        }
                    });
                }
            }
        }

        let candidates: Vec<bool> = module
            .globals
            .iter()
            .enumerate()
            .map(|(id, global)| {
                global.local
                    && !global.external
                    && !global.weak
                    && !global.tls
                    && !written[id]
                    && !escaped[id]
                    && matches!(
                        &global.init,
                        Some(Const::Int { .. } | Const::Float { .. } | Const::Null | Const::Zero)
                    )
            })
            .collect();

        let mut replacements = Vec::new();
        for (func_id, inst_id, global) in sites {
            let func = &module.funcs[func_id];
            if !candidates[global.index()] {
                continue;
            }
            let Some(result) = func.insts[inst_id].result else {
                continue;
            };
            let init = module.globals[global.index()].init.clone().unwrap();
            if init_matches(&init, &func.values[result].ty) {
                replacements.push((func_id, inst_id, init));
            }
        }

        let mut changed = false;
        for (func_id, inst_id, init) in replacements {
            let func = &mut module.funcs[func_id];
            let Some(result) = func.insts[inst_id].result else {
                continue;
            };
            let block = func.insts[inst_id].block;
            let ty = func.values[result].ty.clone();
            let fresh = func.values.push(ValueData {
                ty,
                def: Def::Const(init),
            });
            func.replace_uses(result, fresh);
            func.blocks[block].insts.retain(|i| *i != inst_id);
            changed = true;
        }
        changed
    }
}

/// Remove unreferenced internal functions and objects.
pub struct DeadModuleItems;

impl ModulePass for DeadModuleItems {
    fn name(&self) -> &'static str {
        "dead-module-items"
    }

    fn run(&mut self, module: &mut Module) -> bool {
        if has_inline_asm(module) {
            return false;
        }

        let nfuncs = module.funcs.len();
        let nglobals = module.globals.len();
        let mut func_calls = vec![Vec::new(); nfuncs];
        let mut func_globals = vec![Vec::new(); nfuncs];
        for (id, func) in module.funcs.iter().enumerate() {
            let resident = func.resident_mask();
            for (inst_id, inst) in func.insts.iter_enumerated() {
                if !resident[inst_id.index()] {
                    continue;
                }
                match &inst.kind {
                    InstKind::Call { callee, .. } => func_calls[id].push(*callee),
                    InstKind::FuncAddr { func } => func_calls[id].push(*func),
                    InstKind::GlobalAddr { global } => func_globals[id].push(*global),
                    _ => {}
                }
            }
            if let Some(debug) = &func.debug {
                collect_debug_globals(debug, &mut func_globals[id]);
            }
        }

        let mut global_funcs = vec![Vec::new(); nglobals];
        let mut global_globals = vec![Vec::new(); nglobals];
        for (id, global) in module.globals.iter().enumerate() {
            for reloc in global.relocs.iter().map(|(_, r)| r).chain(global.relocs32.iter().map(|(_, r)| r)) {
                match reloc {
                    DataReloc::Func { target } => global_funcs[id].push(*target),
                    DataReloc::Global { target, .. } => global_globals[id].push(*target),
                    DataReloc::FuncExtern { .. } => {}
                }
            }
        }

        let mut live_funcs: Vec<bool> = module
            .funcs
            .iter()
            .map(|func| !func.local || func.weak)
            .collect();
        let mut live_globals: Vec<bool> = module
            .globals
            .iter()
            .map(|global| !global.local || global.external || global.weak)
            .collect();

        loop {
            let mut changed = false;
            for id in 0..nfuncs {
                if !live_funcs[id] {
                    continue;
                }
                for target in &func_calls[id] {
                    if !live_funcs[target.index()] {
                        live_funcs[target.index()] = true;
                        changed = true;
                    }
                }
                for target in &func_globals[id] {
                    if !live_globals[target.index()] {
                        live_globals[target.index()] = true;
                        changed = true;
                    }
                }
            }
            for id in 0..nglobals {
                if !live_globals[id] {
                    continue;
                }
                for target in &global_funcs[id] {
                    if !live_funcs[target.index()] {
                        live_funcs[target.index()] = true;
                        changed = true;
                    }
                }
                for target in &global_globals[id] {
                    if !live_globals[target.index()] {
                        live_globals[target.index()] = true;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        let (func_map, keep_funcs) = build_map(&live_funcs);
        let (global_map, keep_globals) = build_map(&live_globals);
        if keep_funcs == nfuncs && keep_globals == nglobals {
            return false;
        }

        let old_funcs = std::mem::take(&mut module.funcs);
        let mut new_funcs = Vec::with_capacity(keep_funcs);
        for (old, mut func) in old_funcs.into_iter().enumerate() {
            if !live_funcs[old] {
                continue;
            }
            remap_function(&mut func, &func_map, &global_map);
            new_funcs.push(func);
        }
        module.funcs = IndexVec::from_vec(new_funcs);

        let old_globals = std::mem::take(&mut module.globals);
        let mut new_globals = Vec::with_capacity(keep_globals);
        for (old, mut global) in old_globals.into_iter().enumerate() {
            if !live_globals[old] {
                continue;
            }
            remap_global(&mut global, &func_map, &global_map);
            new_globals.push(global);
        }
        module.globals = IndexVec::from_vec(new_globals);
        true
    }
}

fn has_inline_asm(module: &Module) -> bool {
    module.funcs.iter().any(|func| {
        let resident = func.resident_mask();
        func.insts
            .iter_enumerated()
            .any(|(id, inst)| resident[id.index()] && matches!(inst.kind, InstKind::InlineAsm { .. }))
    })
}

fn global_addresses(func: &Function, resident: &[bool]) -> HashMap<Value, Global> {
    let mut addresses = HashMap::new();
    for (id, inst) in func.insts.iter_enumerated() {
        if !resident[id.index()] {
            continue;
        }
        if let InstKind::GlobalAddr { global } = inst.kind {
            if let Some(result) = inst.result {
                addresses.insert(result, global);
            }
        }
    }
    addresses
}

fn init_matches(init: &Const, ty: &Ty) -> bool {
    match init {
        Const::Int { bits, .. } => {
            matches!(ty, Ty::Int { bits: load_bits } if load_bits == bits)
                || matches!(ty, Ty::Bool) && *bits == 1
        }
        Const::Float { kind, .. } => matches!(ty, Ty::Float(load_kind) if load_kind == kind),
        Const::Null => matches!(ty, Ty::Ptr { .. }),
        Const::Zero => ty.is_scalar(),
        _ => false,
    }
}

fn build_map<T: index_vec::Idx>(live: &[bool]) -> (Vec<Option<T>>, usize) {
    let mut map = Vec::with_capacity(live.len());
    let mut next = 0;
    for &is_live in live {
        if is_live {
            map.push(Some(T::from_usize(next)));
            next += 1;
        } else {
            map.push(None);
        }
    }
    (map, next)
}

fn remap_function(func: &mut Function, funcs: &[Option<Func>], globals: &[Option<Global>]) {
    // Orphaned instructions never execute and no longer feed liveness,
    // so a target they name may already be gone.
    let resident = func.resident_mask();
    for (id, inst) in func.insts.iter_mut().enumerate() {
        if !resident[id] {
            continue;
        }
        match &mut inst.kind {
            InstKind::Call { callee, .. } => {
                *callee = funcs[callee.index()].expect("live caller references a live function");
            }
            InstKind::FuncAddr { func: target } => {
                *target = funcs[target.index()].expect("live function addresses a live function");
            }
            InstKind::GlobalAddr { global } => {
                *global = globals[global.index()].expect("live function addresses a live object");
            }
            _ => {}
        }
    }
    if let Some(debug) = &mut func.debug {
        remap_debug(debug, globals);
    }
}

fn remap_global(global: &mut crate::ir::GlobalData, funcs: &[Option<Func>], globals: &[Option<Global>]) {
    for reloc in global.relocs.iter_mut().map(|(_, r)| r).chain(global.relocs32.iter_mut().map(|(_, r)| r)) {
        match reloc {
            DataReloc::Func { target } => {
                *target = funcs[target.index()].expect("live object references a live function");
            }
            DataReloc::Global { target, .. } => {
                *target = globals[target.index()].expect("live object references a live object");
            }
            DataReloc::FuncExtern { .. } => {}
        }
    }
}

fn collect_debug_globals(debug: &FunctionDebug, out: &mut Vec<Global>) {
    for var in debug.params.iter().chain(debug.vars.iter()) {
        if let VarHome::Global(global) = var.home {
            out.push(global);
        }
    }
    for scope in &debug.scopes {
        collect_scope_globals(scope, out);
    }
}

fn collect_scope_globals(scope: &ScopeDebug, out: &mut Vec<Global>) {
    for var in &scope.vars {
        if let VarHome::Global(global) = var.home {
            out.push(global);
        }
    }
    for child in &scope.children {
        collect_scope_globals(child, out);
    }
}

fn remap_debug(debug: &mut FunctionDebug, globals: &[Option<Global>]) {
    for var in debug.params.iter_mut().chain(debug.vars.iter_mut()) {
        remap_home(&mut var.home, globals);
    }
    for scope in &mut debug.scopes {
        remap_scope(scope, globals);
    }
}

fn remap_scope(scope: &mut ScopeDebug, globals: &[Option<Global>]) {
    for var in &mut scope.vars {
        remap_home(&mut var.home, globals);
    }
    for child in &mut scope.children {
        remap_scope(child, globals);
    }
}

fn remap_home(home: &mut VarHome, globals: &[Option<Global>]) {
    if let VarHome::Global(global) = home {
        *global = globals[global.index()].expect("live function debug references a live object");
    }
}

#[cfg(test)]
mod tests {
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
        let addr = emit(
            &mut func,
            entry,
            InstKind::GlobalAddr { global },
            Ty::ptr(Ty::int(32)),
        );
        let value = emit(
            &mut func,
            entry,
            InstKind::Load {
                addr,
                flags: Default::default(),
            },
            Ty::int(32),
        );
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
        assert_eq!(
            module.funcs[func].const_value(returned(&module.funcs[func])),
            Some(&Const::Int { bits: 32, value: 42 })
        );
        assert!(module.funcs[func].blocks[module.funcs[func].entry]
            .insts
            .iter()
            .all(|inst| !matches!(module.funcs[func].insts[*inst].kind, InstKind::Load { .. })));
        assert!(verify_module(&module).is_empty());
        assert!(!GlobalConstProp.run(&mut module), "second run is a no-op");
    }

    #[test]
    fn keeps_loads_from_written_storage() {
        let mut module = Module::new();
        let global = int32_global(&mut module, "c", 42);
        let (func, value) = load_global(&mut module, global);
        let entry = module.funcs[func].entry;
        let addr = emit(
            &mut module.funcs[func],
            entry,
            InstKind::GlobalAddr { global },
            Ty::ptr(Ty::int(32)),
        );
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
        let addr = emit(
            &mut module.funcs[func],
            entry,
            InstKind::GlobalAddr { global },
            Ty::ptr(Ty::int(32)),
        );
        emit_void(
            &mut module.funcs[func],
            entry,
            InstKind::Memcpy {
                dst: addr,
                src: addr,
                len: 0,
            },
        );
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
        let addr = emit(
            &mut module.funcs[live_func],
            live_entry,
            InstKind::GlobalAddr {
                global: live_global,
            },
            Ty::ptr(Ty::int(32)),
        );
        let loaded = emit(
            &mut module.funcs[live_func],
            live_entry,
            InstKind::Load {
                addr,
                flags: Default::default(),
            },
            Ty::int(32),
        );
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
        let Some(InstKind::Call { callee, .. }) =
            module.funcs[Func::from_usize(1)].blocks[keep_entry].insts.first().map(|i| &module.funcs[Func::from_usize(1)].insts[*i].kind)
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
}
