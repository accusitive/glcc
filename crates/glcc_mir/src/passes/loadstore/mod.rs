//! Forward memory values inside a basic block.
//!
//! Two loads from the same address with no intervening write produce the
//! same value, and a load after a store to the same address produces the
//! stored value. Lowering leaves both patterns everywhere (`*p = 5;
//! return *p;`, `a[i] = 7; return a[i];`), and neither `Cse` nor
//! `DeadStoreElim` touches them: `Cse` only numbers pure expressions and
//! `DeadStoreElim` only removes writes, never reads.
//!
//! Addresses compare by `address_info` when both resolve to the same
//! stack slot, global, or allocation at a constant offset, and by SSA
//! identity otherwise. Writes invalidate every potentially aliasing
//! entry using the same stack-slot escape rules as `DeadStoreElim`.
//! Calls, inline asm, volatile accesses, and aggregate copies are
//! barriers unless an extern's [`ExternEffect`](crate::ir::ExternEffect)
//! promises it does not write program-visible memory.

use std::collections::HashSet;

use crate::ir::{Block, Function, InstKind, Value};
use crate::pass::FunctionPass;

use super::{address_info, non_escaping_stack_slots, ranges_may_overlap};

/// Store-to-load forwarding and redundant load elimination.
#[derive(Default)]
pub struct LoadStoreForward;

struct Entry {
    addr: Value,
    len: u64,
    value: Value,
}

enum Action {
    Keep,
    Replace(Value),
    Barrier,
}

impl FunctionPass for LoadStoreForward {
    fn name(&self) -> &'static str {
        "load-store-forward"
    }

    fn run(&mut self, func: &mut Function) -> bool {
        if func.blocks.is_empty() {
            return false;
        }
        let non_escaping = non_escaping_stack_slots(func);
        let mut changed = false;
        for index in 0..func.blocks.len() {
            let block = Block::from_usize(index);
            changed |= forward_block(func, block, &non_escaping);
        }
        changed
    }
}

fn forward_block(func: &mut Function, block: Block, non_escaping: &HashSet<Value>) -> bool {
    let insts = std::mem::take(&mut func.blocks[block].insts);
    let mut entries: Vec<Entry> = Vec::new();
    let mut out = Vec::with_capacity(insts.len());
    let mut changed = false;
    for inst in insts {
        let action = match &func.insts[inst].kind {
            InstKind::Load { addr, flags } => {
                let Some(result) = func.insts[inst].result else {
                    out.push(inst);
                    continue;
                };
                if flags.volatile {
                    Action::Barrier
                } else {
                    let ty = &func.values[result].ty;
                    let forwarded = entries
                        .iter()
                        .rev()
                        .find(|entry| same_location(func, entry.addr, *addr) && func.values[entry.value].ty == *ty)
                        .map(|entry| entry.value);
                    match forwarded {
                        Some(value) => Action::Replace(value),
                        None => {
                            entries.push(Entry {
                                addr: *addr,
                                len: ty.size_bytes(),
                                value: result,
                            });
                            Action::Keep
                        }
                    }
                }
            }
            InstKind::Store { addr, val, flags } => {
                if flags.volatile {
                    Action::Barrier
                } else {
                    let len = func.values[*val].ty.size_bytes();
                    if len == 0 {
                        Action::Keep
                    } else {
                        entries.retain(|entry| !ranges_may_overlap(func, non_escaping, *addr, len, entry.addr, entry.len));
                        entries.push(Entry { addr: *addr, len, value: *val });
                        Action::Keep
                    }
                }
            }
            // Division may trap but does not touch memory; treating it as
            // a barrier would discard valid loads for no safety gain.
            InstKind::Udiv(..) | InstKind::Sdiv(..) | InstKind::Urem(..) | InstKind::Srem(..) => Action::Keep,
            InstKind::CallExtern { target, .. } if !func.extern_effect(*target).writes => Action::Keep,
            kind if kind.is_pure() => Action::Keep,
            _ => Action::Barrier,
        };
        match action {
            Action::Keep => out.push(inst),
            Action::Replace(value) => {
                let result = func.insts[inst].result.expect("load result");
                func.replace_uses(result, value);
                changed = true;
            }
            Action::Barrier => {
                entries.clear();
                out.push(inst);
            }
        }
    }
    func.blocks[block].insts = out;
    changed
}

/// Whether two addresses name the same object at the same offset. Bare
/// SSA identity covers parameter pointers; the object-plus-offset form
/// covers distinct `getelementptr`/`ptrtoint` spellings of one slot.
fn same_location(func: &Function, a: Value, b: Value) -> bool {
    if a == b {
        return true;
    }
    match (address_info(func, a), address_info(func, b)) {
        (Some(a), Some(b)) => a.object == b.object && a.offset.is_some() && a.offset == b.offset,
        _ => false,
    }
}

#[cfg(test)]
mod tests;

