use crate::ir::{Function, InstKind, LpadClause, Module, Terminator};
use crate::ty::{Signature, Ty};

/// Structural check only: operand existence, arity, terminator presence.
/// Dominance is left to construction discipline (the builder appends in
/// order) so passes stay simple.
pub fn verify_func(func: &Function) -> Vec<String> {
    let mut errors = Vec::new();
    if func.blocks.is_empty() {
        errors.push(format!("{}: no blocks", func.name));
        return errors;
    }
    if func.entry.index() >= func.blocks.len() {
        errors.push(format!("{}: bad entry block", func.name));
    }
    for (id, block) in func.blocks.iter_enumerated() {
        if block.term.is_none() {
            errors.push(format!("{}: block {id:?} unterminated", func.name));
        }
        for param in &block.params {
            if param.index() >= func.values.len() {
                errors.push(format!("{}: bad block param", func.name));
            }
        }
        if let Some(clauses) = &block.lpad {
            if block.params.len() != 2 {
                errors.push(format!(
                    "{}: landing pad block {id:?} takes {} params, expected the two payload values",
                    func.name,
                    block.params.len()
                ));
            }
            if clauses.is_empty() {
                errors.push(format!("{}: landing pad block {id:?} has no clauses", func.name));
            }
        }
        for inst in &block.insts {
            if inst.index() >= func.insts.len() {
                errors.push(format!("{}: bad inst reference", func.name));
                continue;
            }
            let data = &func.insts[*inst];
            if data.block != id {
                errors.push(format!("{}: inst {inst:?} in wrong block", func.name));
            }
            for op in data.kind.operands() {
                if op.index() >= func.values.len() {
                    errors.push(format!("{}: inst {inst:?} bad operand", func.name));
                }
            }
            if let Some(result) = data.result
                && result.index() >= func.values.len() {
                    errors.push(format!("{}: inst {inst:?} bad result", func.name));
                }
            check_inst_types(func, *inst, &data.kind, &mut errors);
        }
        if let Some(term) = &block.term {
            for op in term.operands() {
                if op.index() >= func.values.len() {
                    errors.push(format!("{}: terminator bad operand", func.name));
                }
            }
            for succ in term.successors() {
                if succ.index() >= func.blocks.len() {
                    errors.push(format!("{}: bad successor", func.name));
                }
            }
            check_term(func, term, &mut errors);
        }
    }
    check_eh(func, &mut errors);
    errors
}

/// Exception-handling well-formedness: landing pads carry the two payload
/// params and are reached only through `unwind` edges; EH ops require a
/// personality; `eh.typeid` names a typeinfo this function can catch.
fn check_eh(func: &Function, errors: &mut Vec<String>) {
    // Normal edges (br/condbr/switch and the invoke normal edge) must not
    // target a landing pad; collect every normal successor.
    let mut normal_succs = std::collections::HashSet::new();
    for block in func.blocks.iter() {
        let Some(term) = &block.term else { continue };
        match term {
            Terminator::Br { dest, .. } => {
                normal_succs.insert(*dest);
            }
            Terminator::CondBr { then_block, else_block, .. } => {
                normal_succs.insert(*then_block);
                normal_succs.insert(*else_block);
            }
            Terminator::Switch { cases, default, .. } => {
                normal_succs.insert(*default);
                for (_, dest) in cases {
                    normal_succs.insert(*dest);
                }
            }
            Terminator::Invoke { normal, .. } => {
                normal_succs.insert(*normal);
            }
            Terminator::Return { .. } | Terminator::Resume { .. } | Terminator::Unreachable => {}
        }
    }
    for (id, _) in func.blocks.iter_enumerated() {
        if normal_succs.contains(&id) {
            if func.blocks[id].lpad.is_some() {
                errors.push(format!(
                    "{}: landing pad block {id:?} is targeted by a normal edge; only unwind edges may reach it",
                    func.name
                ));
            }
        }
    }
    let has_eh = func.blocks.iter().any(|b| b.lpad.is_some() || matches!(b.term, Some(Terminator::Invoke { .. }) | Some(Terminator::Resume { .. })))
        || func.insts.iter().any(|i| matches!(i.kind, InstKind::EhTypeId { .. }));
    if has_eh && func.personality.is_none() {
        errors.push(format!(
            "{}: function has EH constructs but no personality",
            func.name
        ));
    }
    for (inst, data) in func.insts.iter_enumerated() {
        let InstKind::EhTypeId { typeinfo } = data.kind else {
            continue;
        };
        let known = func.blocks.iter().any(|b| {
            b.lpad
                .as_ref()
                .is_some_and(|cs| cs.iter().any(|c| matches!(c, LpadClause::Catch(g) if *g == typeinfo)))
        });
        if !known {
            errors.push(format!(
                "{}: eh.typeid at inst {inst:?} names a typeinfo absent from every landing pad's clause list",
                func.name
            ));
        }
    }
}

fn check_inst_types(func: &Function, inst: crate::ir::Inst, kind: &InstKind, errors: &mut Vec<String>) {
    let ty_of = |v: crate::ir::Value| func.values[v].ty.clone();
    match kind {
        InstKind::Call { callee, args } => {
            let _ = (callee, args);
        }
        InstKind::CallIndirect { callee, args, .. } => {
            let _ = (callee, args);
        }
        InstKind::AtomicLoad { .. }
        | InstKind::AtomicStore { .. }
        | InstKind::AtomicRmw { .. }
        | InstKind::AtomicCas { .. }
        | InstKind::AtomicFence
        | InstKind::GetFramePointer => {}
        InstKind::GetElementPtr { base, indices } => {
            let mut cur = ty_of(*base);
            if let Ty::Ptr { pointee, .. } = cur {
                cur = pointee.as_ref().clone();
            } else {
                errors.push(format!("{}: gep on non-pointer {inst:?}", func.name));
                return;
            }
            for idx in indices.iter() {
                match Signature::gep_step(&cur, idx) {
                    Some(next) => cur = next,
                    None => {
                        errors.push(format!("{}: bad gep step {idx:?} on {cur:?}", func.name));
                        return;
                    }
                }
            }
        }
        InstKind::Load { .. } | InstKind::Store { .. } => {}
        _ => {}
    }
    let _ = term_ok(func);
}

fn term_ok(_func: &Function) -> bool {
    true
}

fn check_term(func: &Function, term: &Terminator, errors: &mut Vec<String>) {
    match term {
        Terminator::Br { dest, args } => {
            if args.len() != func.blocks[*dest].params.len() {
                errors.push(format!("{}: br arity mismatch", func.name));
            }
        }
        Terminator::CondBr {
            then_block,
            else_block,
            then_args,
            else_args,
            ..
        } => {
            if then_args.len() != func.blocks[*then_block].params.len() || else_args.len() != func.blocks[*else_block].params.len() {
                errors.push(format!("{}: condbr arity mismatch", func.name));
            }
        }
        Terminator::Return { val } => {
            let is_void = func.sig.ret.is_void();
            if is_void != val.is_none() {
                errors.push(format!("{}: return arity mismatch", func.name));
            }
        }
        Terminator::Invoke { args: _, sig, normal, normal_args, unwind, .. } => {
            // The normal edge passes the call's results first, then the
            // extra branch arguments.
            let returns = usize::from(!sig.ret.is_void());
            if func.blocks[*normal].params.len() != returns + normal_args.len() {
                errors.push(format!(
                    "{}: invoke normal edge b{} takes {} params for {} call result(s)",
                    func.name,
                    normal.index(),
                    func.blocks[*normal].params.len(),
                    returns
                ));
            }
            if func.blocks[*unwind].lpad.is_none() {
                errors.push(format!(
                    "{}: invoke unwind target b{} is not a landing pad",
                    func.name,
                    unwind.index()
                ));
            }
        }
        Terminator::Switch { .. } | Terminator::Resume { .. } | Terminator::Unreachable => {}
    }
}

pub fn verify_module(module: &Module) -> Vec<String> {
    let mut errors = Vec::new();
    for func in module.funcs.iter() {
        errors.extend(verify_func(func));
    }
    let mut seen = std::collections::HashSet::new();
    for func in &module.init_funcs {
        if func.index() >= module.funcs.len() {
            errors.push(format!(".init target f{} is out of range", func.index()));
            continue;
        }
        if !seen.insert(*func) {
            errors.push(format!("duplicate .init entry @{}", module.funcs[*func].name));
            continue;
        }
        let sig = &module.funcs[*func].sig;
        if !sig.params.is_empty() || !sig.ret.is_void() {
            errors.push(format!(
                ".init target @{} must take no parameters and return void, got {:?} -> {:?}",
                module.funcs[*func].name, sig.params, sig.ret,
            ));
        }
    }
    errors
}

#[cfg(test)]
mod tests;

