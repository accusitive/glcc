use crate::ir::{Function, InstKind, Module, Terminator};
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
    errors
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
        Terminator::Switch { .. } | Terminator::Unreachable => {}
    }
}

pub fn verify_module(module: &Module) -> Vec<String> {
    let mut errors = Vec::new();
    for func in module.funcs.iter() {
        errors.extend(verify_func(func));
    }
    errors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ty::{Signature, Ty};

    #[test]
    fn unterminated_block_fails() {
        let sig = Signature::new(vec![], Ty::Void);
        let (func, _) = Function::new("empty", sig);
        assert!(!verify_func(&func).is_empty());
    }

    #[test]
    fn terminated_block_passes() {
        let sig = Signature::new(vec![], Ty::Void);
        let (mut func, entry) = Function::new("ret", sig);
        func.blocks[entry].term = Some(Terminator::Return { val: None });
        assert!(verify_func(&func).is_empty());
    }
}
