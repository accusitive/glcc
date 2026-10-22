//! The canonical MIR optimization pipeline.
//!
//! Every consumer (the C driver, `fernc`, the explorer, and the runtime)
//! runs this function instead of assembling its own pass list, so a new
//! pass or a changed order reaches all of them at once.

use crate::ir::Module;
use crate::pass::PassManager;
use crate::passes;

/// Optimization selector shared by every consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptLevel {
    /// `-O0` (default): no optimization, best debug fidelity.
    #[default]
    None,
    /// `-O1`/`-O2`/`-O3`/`-Og`/`-Ofast`/bare `-O`: optimized.
    Speed,
    /// `-Os`/`-Oz`: optimize for size.
    Size,
}

impl OptLevel {
    /// Cranelift `opt_level` setting value.
    pub fn flag(self) -> &'static str {
        match self {
            OptLevel::None => "none",
            OptLevel::Speed => "speed",
            OptLevel::Size => "speed_and_size",
        }
    }

    /// Whether any MIR optimization runs at this level.
    pub fn optimize(self) -> bool {
        self != OptLevel::None
    }

    /// Parse Godbolt-style values: `0`, `1`, `2`, `3`, `s`, `z`, `fast`.
    pub fn from_str(value: &str) -> OptLevel {
        match value.trim() {
            "1" | "2" | "3" | "fast" => OptLevel::Speed,
            "s" | "z" | "size" => OptLevel::Size,
            _ => OptLevel::None,
        }
    }
}

/// Run the canonical MIR optimization pipeline at `level`.
pub fn optimize(module: &mut Module, level: OptLevel) -> Vec<(&'static str, bool)> {
    let mut pm = PassManager::new();
    if !level.optimize() {
        return pm.run(module);
    }
    pm = pm
        .add_module_pass(passes::Inline::small())
        .add_module_pass(passes::LowerLibcMemory)
        .add_module_pass(passes::GlobalConstProp)
        .add_module_pass(passes::MergeReadOnlyGlobals)
        .add_module_pass(passes::DeadModuleItems);
    let mut report = function_pipeline(pm).run(module);
    // Function passes turn aggregate loads and stores into copies, so
    // give module-level propagation another look at the settled IR. A
    // copied read-only global can then expose its field initializers.
    for _ in 0..3 {
        let mut globals = PassManager::new()
            .add_module_pass(passes::GlobalConstProp)
            .add_module_pass(passes::DeadModuleItems);
        let module_report = globals.run(module);
        let changed = module_report.iter().any(|(_, changed)| *changed);
        report.extend(module_report);
        if !changed {
            break;
        }
        report.extend(function_pipeline(PassManager::new()).run(module));
    }
    // Function passes can orphan the last reference to an internal
    // object (a load that `GlobalConstProp` replaced, a call `Inline`
    // removed), so sweep module liveness once more after they settle.
    let mut settle = PassManager::new().add_module_pass(passes::DeadModuleItems);
    report.extend(settle.run(module));
    report
}

fn function_pipeline(pm: PassManager) -> PassManager {
    pm.add_func_pass(passes::Sroa)
        .add_func_pass(passes::Mem2Reg)
        .add_func_pass(passes::ConstFold)
        .add_func_pass(passes::Sccp)
        .add_func_pass(passes::ForwardAggregateCopy)
        .add_func_pass(passes::DeadStoreElim)
        .add_func_pass(passes::LoadStoreForward)
        .add_func_pass(passes::InstCombine)
        .add_func_pass(passes::ConstMerge)
        .add_func_pass(passes::Cse)
        .add_func_pass(passes::Licm)
        .add_func_pass(passes::SimplifyCfg)
        .add_func_pass(passes::DeadCodeElim)
}

#[cfg(test)]
mod tests {
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
        let addr = emit(
            &mut func,
            entry,
            InstKind::GlobalAddr {
                global: constant,
            },
            Ty::ptr(Ty::int(32)),
        );
        let left = emit(
            &mut func,
            entry,
            InstKind::Load {
                addr,
                flags: Default::default(),
            },
            Ty::int(32),
        );
        let right = emit(
            &mut func,
            entry,
            InstKind::Load {
                addr,
                flags: Default::default(),
            },
            Ty::int(32),
        );
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
        assert_eq!(
            module.funcs[0].const_value(returned(&module.funcs[0])),
            Some(&Const::Int { bits: 32, value: 8 })
        );
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
        let addr = emit(
            &mut reader,
            reader_entry,
            InstKind::GlobalAddr { global: seed },
            Ty::ptr(Ty::int(32)),
        );
        let loaded = emit(
            &mut reader,
            reader_entry,
            InstKind::Load {
                addr,
                flags: Default::default(),
            },
            Ty::int(32),
        );
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
}
