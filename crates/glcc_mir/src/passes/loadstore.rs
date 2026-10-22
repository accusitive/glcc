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
//! stack slot or global at a constant offset, and by SSA identity
//! otherwise. Writes invalidate every potentially aliasing entry using
//! the same stack-slot escape rules as `DeadStoreElim`. Calls, inline
//! asm, volatile accesses, and aggregate copies are barriers.

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
                        entries.push(Entry {
                            addr: *addr,
                            len,
                            value: *val,
                        });
                        Action::Keep
                    }
                }
            }
            // Division may trap but does not touch memory; treating it as
            // a barrier would discard valid loads for no safety gain.
            InstKind::Udiv(..) | InstKind::Sdiv(..) | InstKind::Urem(..) | InstKind::Srem(..) => Action::Keep,
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
mod tests {
    use super::*;
    use crate::ir::{Const, Def, Extern, Inst, InstData, MemFlags, Terminator, ValueData};
    use crate::ty::{Signature, Ty};
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

    fn ptr_func() -> (Function, Block, Value) {
        let sig = Signature::new(vec![Ty::ptr(Ty::int(32))], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let p = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
        (func, entry, p)
    }

    fn int_const(func: &mut Function, value: i128) -> Value {
        func.push_const(Const::Int { bits: 32, value })
    }

    fn run(func: &mut Function) -> bool {
        let changed = LoadStoreForward.run(func);
        let errors = verify_func(func);
        assert!(errors.is_empty(), "verifier: {errors:?}");
        changed
    }

    fn returned(func: &Function) -> Value {
        match &func.blocks[func.entry].term {
            Some(Terminator::Return { val: Some(value) }) => *value,
            other => panic!("expected a value return, got {other:?}"),
        }
    }

    #[test]
    fn store_then_load_forwards_the_stored_value() {
        let (mut func, entry, p) = ptr_func();
        let five = int_const(&mut func, 5);
        emit(&mut func, entry, InstKind::Store { addr: p, val: five, flags: MemFlags::plain() }, Ty::Void);
        let loaded = emit(&mut func, entry, InstKind::Load { addr: p, flags: MemFlags::plain() }, Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        assert!(run(&mut func));
        assert_eq!(func.blocks[entry].insts.len(), 1, "load stays removed");
        assert_eq!(returned(&func), five);
        assert!(!run(&mut func), "second run is a no-op");
    }

    #[test]
    fn duplicate_loads_share_one_result() {
        let (mut func, entry, p) = ptr_func();
        let first = emit(&mut func, entry, InstKind::Load { addr: p, flags: MemFlags::plain() }, Ty::int(32));
        let second = emit(&mut func, entry, InstKind::Load { addr: p, flags: MemFlags::plain() }, Ty::int(32));
        let sum = emit(&mut func, entry, InstKind::Add(first, second), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });

        assert!(run(&mut func));
        assert_eq!(func.blocks[entry].insts.len(), 2, "second load removed");
        assert!(matches!(func.insts[func.blocks[entry].insts[1]].kind, InstKind::Add(a, b) if a == first && b == first));
    }

    #[test]
    fn a_may_aliasing_store_invalidates_loads() {
        let (mut func, entry, p) = ptr_func();
        let q = param(&mut func, entry, Ty::ptr(Ty::int(32)), 1);
        let first = emit(&mut func, entry, InstKind::Load { addr: p, flags: MemFlags::plain() }, Ty::int(32));
        let zero = int_const(&mut func, 0);
        emit(&mut func, entry, InstKind::Store { addr: q, val: zero, flags: MemFlags::plain() }, Ty::Void);
        let second = emit(&mut func, entry, InstKind::Load { addr: p, flags: MemFlags::plain() }, Ty::int(32));
        let sum = emit(&mut func, entry, InstKind::Add(first, second), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });

        assert!(!run(&mut func), "the second load must stay");
        assert_eq!(func.blocks[entry].insts.len(), 4);
    }

    #[test]
    fn independent_stack_slots_keep_their_values() {
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let a = emit(&mut func, entry, InstKind::Alloca { pointee: Ty::int(32), count: 1 }, Ty::ptr(Ty::int(32)));
        let b = emit(&mut func, entry, InstKind::Alloca { pointee: Ty::int(32), count: 1 }, Ty::ptr(Ty::int(32)));
        let five = int_const(&mut func, 5);
        let eight = int_const(&mut func, 8);
        emit(&mut func, entry, InstKind::Store { addr: a, val: five, flags: MemFlags::plain() }, Ty::Void);
        emit(&mut func, entry, InstKind::Store { addr: b, val: eight, flags: MemFlags::plain() }, Ty::Void);
        let loaded = emit(&mut func, entry, InstKind::Load { addr: a, flags: MemFlags::plain() }, Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        assert!(run(&mut func));
        assert_eq!(returned(&func), five);
    }

    #[test]
    fn orphaned_address_use_does_not_block_forwarding() {
        let sig = Signature::new(vec![Ty::ptr(Ty::int(32))], Ty::int(32));
        let (mut func, entry) = Function::new("g", sig);
        let p = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
        let slot = emit(&mut func, entry, InstKind::Alloca { pointee: Ty::int(32), count: 1 }, Ty::ptr(Ty::int(32)));
        let five = int_const(&mut func, 5);
        emit(&mut func, entry, InstKind::Store { addr: slot, val: five, flags: MemFlags::plain() }, Ty::Void);
        let call = emit(
            &mut func,
            entry,
            InstKind::CallExtern {
                target: Extern::from_usize(0),
                args: [slot].into_iter().collect(),
            },
            Ty::Void,
        );
        let call_inst = match &func.values[call].def {
            crate::ir::Def::Inst(inst) => *inst,
            _ => panic!("call result is an instruction"),
        };
        func.blocks[entry].insts.retain(|inst| *inst != call_inst);

        let nine = int_const(&mut func, 9);
        emit(&mut func, entry, InstKind::Store { addr: p, val: nine, flags: MemFlags::plain() }, Ty::Void);
        let loaded = emit(&mut func, entry, InstKind::Load { addr: slot, flags: MemFlags::plain() }, Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        assert!(run(&mut func), "the dead call no longer makes the slot escape");
        assert_eq!(returned(&func), five);
    }

    #[test]
    fn volatile_and_call_barriers_keep_the_load() {
        let (mut func, entry, p) = ptr_func();
        let five = int_const(&mut func, 5);
        emit(&mut func, entry, InstKind::Store { addr: p, val: five, flags: MemFlags::plain() }, Ty::Void);
        let volatile = emit(&mut func, entry, InstKind::Load { addr: p, flags: MemFlags::vol() }, Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(volatile) });
        assert!(!run(&mut func), "volatile load is never forwarded");

        let sig = Signature::new(vec![Ty::ptr(Ty::int(32))], Ty::int(32));
        let (mut func, entry) = Function::new("g", sig);
        let p = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
        let first = emit(&mut func, entry, InstKind::Load { addr: p, flags: MemFlags::plain() }, Ty::int(32));
        let target = Extern::from_usize(0);
        emit(&mut func, entry, InstKind::CallExtern { target, args: Default::default() }, Ty::Void);
        let second = emit(&mut func, entry, InstKind::Load { addr: p, flags: MemFlags::plain() }, Ty::int(32));
        let sum = emit(&mut func, entry, InstKind::Add(first, second), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        assert!(!run(&mut func), "a call may write through p");
    }
}
