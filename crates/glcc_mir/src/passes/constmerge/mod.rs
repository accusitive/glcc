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
        duplicates.retain(|dup, _| uses.get(dup.index()).copied().unwrap_or(0) > 0);
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
mod tests;

