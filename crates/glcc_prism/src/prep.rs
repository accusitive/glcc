//! MIR preparation: from the driver's optimized module to the SSA
//! form the PrismIR builder consumes.
//!
//! Shader targets want fully promoted SSA — no allocas, no aggregate
//! memory traffic — regardless of the `-O` level the driver ran, so
//! this pipeline always runs: inline every call (the profile forbids
//! recursion and external calls, so this terminates and is exact),
//! expand aggregate copies, then scalarize and promote all locals.
//! Uniform and sampler globals stay as globals: they are resources,
//! read through the interface, not memory the shader manages.
//!
//! The remaining profile assertions (`validate`) run here too: a
//! prism module reaching this crate must not contain the operations
//! no shader can say (EH, atomics, va_list, inline asm, indirect
//! calls) — the frontend checker already rejected them at the source
//! level, so surviving instances are lowering bugs, not diagnostics.

use std::collections::HashMap;
use std::format;

use glcc_mir::ir::{InstKind, Module, Terminator};
use glcc_mir::passes::DomInfo;
use glcc_mir::pass::{FunctionPass, ModulePass, PassManager};
use glcc_mir::passes;

/// Prepare a copy of `module` for PrismIR construction.
pub fn prepare(module: &Module) -> Result<Module, String> {
    let mut module = module.clone();
    prepare_in_place(&mut module)?;
    Ok(module)
}

fn prepare_in_place(module: &mut Module) -> Result<(), String> {
    for func in module.funcs.iter_mut() {
        shim_entry(func);
    }
    // Shaders inline everything: mark every defined non-entry callee
    // force-inline (dropping the size budget) and clear `noinline`,
    // which has no meaning in a shader — the profile's acyclicity
    // guarantees termination.
    for func in module.funcs.iter_mut() {
        func.force_inline = true;
        func.noinline = false;
    }
    let mut turns = 0;
    loop {
        let changed = passes::Inline::small().run(module);
        if !changed {
            break;
        }
        turns += 1;
        if turns > 256 {
            return Err("prism: inlining did not settle".to_string());
        }
    }
    passes::LowerLibcMemory.run(module);

    if std::env::var_os("GLCC_PRISM_DEBUG").is_some() {
        eprintln!("=== prep: after inline+libc ===\n{}", glcc_mir::display::dump_module(module));
    }
    let mut pm = PassManager::new()
        .add_module_pass(passes::GlobalConstProp)
        .add_func_pass(passes::Sroa)
        .add_func_pass(passes::Mem2Reg)
        .add_func_pass(passes::ConstFold)
        .add_func_pass(passes::Sccp)
        .add_func_pass(passes::InstCombine)
        .add_func_pass(passes::Cse)
        .add_func_pass(passes::SimplifyCfg)
        .add_func_pass(passes::DeadCodeElim);
    pm.run(module);
    if std::env::var_os("GLCC_PRISM_DEBUG").is_some() {
        eprintln!("=== prep: after pipeline ===\n{}", glcc_mir::display::dump_module(module));
    }

    if std::env::var_os("GLCC_PRISM_DEBUG").is_some() {
        eprintln!("{}", glcc_mir::display::dump_module(module));
    }
    validate_profile(module)?;
    let errors = glcc_mir::verify::verify_module(module);
    if !errors.is_empty() {
        return Err(format!("prism: prepared MIR failed verification: {}", errors[0]));
    }
    Ok(())
}

/// A `do { ... } while` whose body starts at the function's entry
/// makes the entry block a loop header: [`glcc_mir::passes::Mem2Reg`]
/// refuses phis there and the loop's locals would stay in memory. Give
/// every such function a fresh empty entry block branching into the
/// real one, so the real entry can carry the loop's block params. The
/// signature parameters stay where they are; the shim branch passes
/// them along.
fn shim_entry(func: &mut glcc_mir::ir::Function) {
    if func.debug.is_some() || func.blocks.is_empty() {
        return;
    }
    let old = func.entry;
    let reachable = func.reachable_blocks();
    let dom = DomInfo::compute(func, &reachable);
    let has_back_edge = dom.preds[old.index()].iter().any(|&pred| dom.dominates(old, pred, old));
    if !has_back_edge {
        return;
    }
    let new_entry = func.push_block(vec![]);
    let args = func.blocks[old].params.iter().copied().collect();
    func.blocks[new_entry].term = Some(Terminator::Br { dest: old, args });
    func.entry = new_entry;
}

/// Profile assertions on the prepared module — the operations no
/// shader can say. The frontend rejects these at the source level;
/// survivors mean a lowering bug.
fn validate_profile(module: &Module) -> Result<(), String> {
    for func in module.funcs.iter() {
        for (id, inst) in func.insts.iter_enumerated() {
            let resident = func.blocks[inst.block].insts.contains(&id);
            if !resident {
                continue;
            }
            let banned = matches!(
                &func.insts[id].kind,
                InstKind::Call { .. }
                    | InstKind::CallIndirect { .. }
                    | InstKind::CallExtern { .. }
                    | InstKind::Alloca { .. }
                    | InstKind::Memcpy { .. }
                    | InstKind::Memset { .. }
                    | InstKind::VaStart { .. }
                    | InstKind::VaArg { .. }
                    | InstKind::VaEnd { .. }
                    | InstKind::VaCopy { .. }
                    | InstKind::Await { .. }
                    | InstKind::AtomicLoad { .. }
                    | InstKind::AtomicStore { .. }
                    | InstKind::AtomicRmw { .. }
                    | InstKind::AtomicCas { .. }
                    | InstKind::AtomicFence
                    | InstKind::InlineAsm { .. }
                    | InstKind::GetFramePointer
                    | InstKind::EhTypeId { .. }
            );
            if banned {
                return Err(format!(
                    "prism: '{}' contains an operation no shader can express ({}); this is a lowering bug",
                    func.name,
                    glcc_mir::display::inst_str(&func, &func.insts[id].kind)
                ));
            }
        }
        for block in func.blocks.iter() {
            match &block.term {
                Some(glcc_mir::ir::Terminator::Return { .. }) | Some(glcc_mir::ir::Terminator::Unreachable) | None => {}
                Some(glcc_mir::ir::Terminator::Resume { .. }) | Some(glcc_mir::ir::Terminator::Invoke { .. }) => {
                    return Err(format!("prism: '{}' carries exception control flow", func.name));
                }
                _ => {}
            }
        }
    }
    Ok(())
}
