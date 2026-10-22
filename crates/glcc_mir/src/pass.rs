use crate::ir::{Function, Module};

/// A rewrite over one function. Returns true when it changed the IR; the
/// manager repeats passes to a fixpoint.
pub trait FunctionPass {
    fn name(&self) -> &'static str;
    fn run(&mut self, func: &mut Function) -> bool;
}

/// A rewrite over the whole module (inlining, dead-function elimination).
pub trait ModulePass {
    fn name(&self) -> &'static str;
    fn run(&mut self, module: &mut Module) -> bool;
}

/// Ordered pipeline of function and module passes.
#[derive(Default)]
pub struct PassManager {
    func_passes: Vec<Box<dyn FunctionPass>>,
    module_passes: Vec<Box<dyn ModulePass>>,
}

impl PassManager {
    pub fn new() -> Self {
        PassManager::default()
    }

    pub fn add_func_pass(mut self, pass: impl FunctionPass + 'static) -> Self {
        self.func_passes.push(Box::new(pass));
        self
    }

    pub fn add_module_pass(mut self, pass: impl ModulePass + 'static) -> Self {
        self.module_passes.push(Box::new(pass));
        self
    }

    /// Run module passes once, then function passes per function to a
    /// fixpoint (bounded so a buggy pass cannot loop forever).
    pub fn run(&mut self, module: &mut Module) -> Vec<(&'static str, bool)> {
        let mut report = Vec::new();
        for pass in self.module_passes.iter_mut() {
            report.push((pass.name(), pass.run(module)));
        }
        let effects: Vec<_> = module.externs.iter().map(|data| data.effect).collect();
        for func in module.funcs.iter_mut() {
            func.set_extern_effects(&effects);
            for _ in 0..16 {
                let mut changed = false;
                for pass in self.func_passes.iter_mut() {
                    changed |= pass.run(func);
                }
                if !changed {
                    break;
                }
            }
        }
        report
    }

    /// Run function passes on one function to a fixpoint.
    pub fn run_on_func(&mut self, func: &mut crate::ir::Function) {
        for _ in 0..16 {
            let mut changed = false;
            for pass in self.func_passes.iter_mut() {
                changed |= pass.run(func);
            }
            if !changed {
                break;
            }
        }
    }
}
