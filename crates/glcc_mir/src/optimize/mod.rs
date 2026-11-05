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
///
/// Wide-integer legalization is not an optimization: the backend only
/// lowers scalars, so it runs first at every level, before any pass
/// that assumes `Ty::Int` fits in a register.
pub fn optimize(module: &mut Module, level: OptLevel) -> Vec<(&'static str, bool)> {
    let mut report = PassManager::new()
        .add_module_pass(passes::LegalizeWideInts::new())
        .run(module);
    if !level.optimize() {
        return report;
    }
    let pm = PassManager::new()
        .add_module_pass(passes::Inline::small())
        .add_module_pass(passes::LowerLibcMemory)
        .add_module_pass(passes::GlobalConstProp)
        .add_module_pass(passes::MergeReadOnlyGlobals)
        .add_module_pass(passes::DeadModuleItems);
    report.extend(function_pipeline(pm).run(module));
    // Function passes turn aggregate loads and stores into copies, so
    // give module-level propagation another look at the settled IR. A
    // copied read-only global can then expose its field initializers.
    for _ in 0..3 {
        let mut globals = PassManager::new().add_module_pass(passes::GlobalConstProp).add_module_pass(passes::DeadModuleItems);
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
        .add_func_pass(passes::LoopVec)
        .add_func_pass(passes::LoopUnroll)
        .add_func_pass(passes::ForwardAggregateCopy)
        .add_func_pass(passes::DeadStoreElim)
        .add_func_pass(passes::LoadStoreForward)
        .add_func_pass(passes::InstCombine)
        .add_func_pass(passes::Slp)
        .add_func_pass(passes::ConstMerge)
        .add_func_pass(passes::Cse)
        .add_func_pass(passes::Licm)
        .add_func_pass(passes::SimplifyCfg)
        .add_func_pass(passes::DeadCodeElim)
}

#[cfg(test)]
mod tests;

