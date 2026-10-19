//! Constant unification: every value denoting one constant collapses
//! into the first value with that type and bit pattern.
//!
//! Lowering and inlining materialize a fresh [`Def::Const`] value per
//! occurrence, so `x * 13` written three times reaches the optimizer
//! as `mul x, c1`, `mul x, c2`, `mul x, c3` with distinct operand
//! ids. Consumers that compare operands structurally — [`Cse`](super::Cse)
//! above all — cannot see that the expressions are identical until
//! the duplicates share one value.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::ir::{Const, Def, Function, Value};
use crate::pass::FunctionPass;
use crate::ty::Ty;

/// Merge duplicate constant values into one representative.
#[derive(Default)]
pub struct ConstMerge;

impl FunctionPass for ConstMerge {
    fn name(&self) -> &'static str {
        "const-merge"
    }

    fn run(&mut self, func: &mut Function) -> bool {
        let mut duplicates: HashMap<Value, Value> = HashMap::new();
        {
            // Scoped so the borrow of `func.values` ends before the
            // rewrite below.
            let mut canon: HashMap<(&Ty, &Const), Value> = HashMap::new();
            for (i, value) in func.values.iter().enumerate() {
                let Def::Const(c) = &value.def else {
                    continue;
                };
                let v = Value::from_usize(i);
                match canon.entry((&value.ty, c)) {
                    Entry::Occupied(e) => {
                        duplicates.insert(v, *e.get());
                    }
                    Entry::Vacant(e) => {
                        e.insert(v);
                    }
                }
            }
        }
        // Values survive rewrites in the arena, so a duplicate whose
        // uses are already gone must not report progress forever.
        let uses = func.count_resident_uses();
        duplicates.retain(|dup, _| uses.get(dup).is_some_and(|n| *n > 0));
        if duplicates.is_empty() {
            return false;
        }
        // Every duplicate maps straight to the representative, so one
        // remap sweep is enough (no chains).
        func.remap_operands(&duplicates);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Block, Inst, InstData, InstKind, Terminator, ValueData};
    use crate::ty::Signature;
    use crate::verify::verify_func;

    fn param(func: &mut Function, block: Block, ty: Ty, index: u32) -> Value {
        let v = func.values.push(ValueData {
            ty,
            def: Def::Param(block, index),
        });
        func.blocks[block].params.push(v);
        v
    }

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

    fn run(func: &mut Function) -> bool {
        let changed = ConstMerge.run(func);
        let errors = verify_func(func);
        assert!(errors.is_empty(), "verifier: {errors:?}");
        changed
    }

    fn def_inst(func: &Function, v: Value) -> Inst {
        match &func.values[v].def {
            Def::Inst(i) => *i,
            other => panic!("expected instruction, got {other:?}"),
        }
    }

    #[test]
    fn merges_duplicate_constants() {
        let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(32), 0);
        let c1 = func.push_const(Const::Int { bits: 32, value: 13 });
        let c2 = func.push_const(Const::Int { bits: 32, value: 13 });
        let m1 = emit(&mut func, entry, InstKind::Mul(x, c1), Ty::int(32));
        let m2 = emit(&mut func, entry, InstKind::Mul(x, c2), Ty::int(32));
        let sum = emit(&mut func, entry, InstKind::Add(m1, m2), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        assert!(run(&mut func));
        for mul in [m1, m2] {
            match &func.insts[def_inst(&func, mul)].kind {
                InstKind::Mul(_, c) => assert_eq!(*c, c1, "both muls read the first 13"),
                other => panic!("expected mul, got {other:?}"),
            }
        }
        assert!(!run(&mut func), "second run is a no-op");
    }

    #[test]
    fn keeps_differently_typed_constants_apart() {
        // Zero as `int` and zero as `long`: same bit pattern, two
        // distinct values.
        let sig = Signature::new(vec![], Ty::int(64));
        let (mut func, entry) = Function::new("f", sig);
        let c32 = func.push_const(Const::Int { bits: 32, value: 0 });
        let c64 = func.push_const(Const::Int { bits: 64, value: 0 });
        let wide = emit(&mut func, entry, InstKind::Zext(c32), Ty::int(64));
        let sum = emit(&mut func, entry, InstKind::Add(wide, c64), Ty::int(64));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        assert_ne!(func.values[c32].ty, func.values[c64].ty);
        assert!(!run(&mut func));
    }

    #[test]
    fn rewrites_terminator_operands() {
        let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let cond = param(&mut func, entry, Ty::Bool, 0);
        let c1 = func.push_const(Const::Int { bits: 32, value: 7 });
        let c2 = func.push_const(Const::Int { bits: 32, value: 7 });
        let then_block = func.push_block(vec![Ty::int(32)]);
        let else_block = func.push_block(vec![Ty::int(32)]);
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block,
            else_block,
            then_args: [c1].into_iter().collect(),
            else_args: [c2].into_iter().collect(),
        });
        let then_param = func.blocks[then_block].params[0];
        let else_param = func.blocks[else_block].params[0];
        func.blocks[then_block].term = Some(Terminator::Return { val: Some(then_param) });
        func.blocks[else_block].term = Some(Terminator::Return { val: Some(else_param) });
        assert!(run(&mut func));
        let Some(Terminator::CondBr { then_args, else_args, .. }) = &func.blocks[entry].term else {
            panic!("expected conditional branch");
        };
        assert_eq!(then_args[0], else_args[0]);
    }
}
