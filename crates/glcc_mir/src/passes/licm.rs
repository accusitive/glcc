//! Loop invariant code motion.
//!
//! A computation whose operands cannot change between iterations does
//! not belong inside the loop. This pass finds natural loops (back
//! edges `n -> h`, including self-edges, where the header `h`
//! dominates `n`), and hoists pure instructions whose operands are
//! defined outside the loop — or by an earlier hoist of the same
//! sweep — into the loop's preheader: the unique out-of-loop
//! predecessor of the header.
//!
//! Loops without a single entry edge are left alone:
//! [`SimplifyCfg`](super::SimplifyCfg)'s `merge_single_pred_blocks`
//! routinely creates proper preheaders first, so this stance only
//! costs hoists on multi-entry loops.
//!
//! Only [`InstKind::is_pure`](crate::ir::InstKind) instructions move:
//! loads, calls, atomics, and trapping arithmetic (`Udiv`/`Sdiv`/
//! `Urem`/`Srem`) are not pure, so moving them could erase a trap or
//! reorder a memory effect. Pure, non-trapping work may run when the
//! loop itself would not, which C never observes.
//!
//! SSA makes the rewrite mechanical: the preheader dominates every
//! loop block, so a hoisted definition still dominates its uses, and
//! operands hoisted earlier in the same sweep now live in the
//! preheader (outside every loop block), which the invariance test
//! picks up in the same run.

use std::collections::{HashMap, HashSet};

use crate::ir::{Block, Def, Function, Inst, Value};
use crate::pass::FunctionPass;

use super::DomInfo;

/// Hoist loop-invariant pure instructions out of natural loops.
#[derive(Default)]
pub struct Licm;

impl FunctionPass for Licm {
    fn name(&self) -> &'static str {
        "licm"
    }

    fn run(&mut self, func: &mut Function) -> bool {
        if func.blocks.is_empty() {
            return false;
        }
        let reachable = func.reachable_blocks();
        let dom = DomInfo::compute(func, &reachable);
        let loops = natural_loops(func, &dom);
        // Outer loops first: an operand hoisted out of an inner loop
        // this run would otherwise only move out of the outer one on a
        // later fixpoint turn, and one hoistable out of an outer loop
        // in one run should not settle for the inner one.
        let mut loops: Vec<(Block, HashSet<Block>)> = loops;
        loops.sort_by_key(|(header, _)| header.index());
        let uses = func.count_resident_uses();
        let mut changed = false;
        for (header, loop_blocks) in loops {
            let Some(preheader) = preheader_of(func, &dom, header, &loop_blocks) else {
                continue;
            };
            changed |= hoist_loop(func, preheader, &loop_blocks, &uses);
        }
        changed
    }
}

/// Every edge `n -> h` with `h` dominating `n` is a back edge; the
/// natural loop is `h` (the real loop header, the edge destination)
/// plus everything that can reach `n` without passing through `h`.
/// Predecessors explored from `n` stop at `h` itself (already in the
/// set), which is exactly that reachability.
fn natural_loops(func: &Function, dom: &DomInfo) -> Vec<(Block, HashSet<Block>)> {
    let mut loops = Vec::new();
    for (idx, block) in func.blocks.iter().enumerate() {
        let n = Block::from_usize(idx);
        let Some(term) = &block.term else {
            continue;
        };
        for succ in term.successors() {
            // A back edge is `n -> h` where the DESTINATION `h`
            // dominates the source `n`, not the other way around.
            if !dominates(dom, succ, n, func.entry) {
                continue;
            }
            let h = succ;
            let mut set: HashSet<Block> = [h].into_iter().collect();
            let mut work = vec![n];
            while let Some(member) = work.pop() {
                if !set.insert(member) {
                    continue;
                }
                work.extend(dom.preds[member.index()].iter().copied());
            }
            loops.push((h, set));
        }
    }
    loops
}

/// The loop's preheader: the unique out-of-loop predecessor of the
/// header. The CFG makes that block dominate the header, and this
/// rechecks it against a pred-bug paranoia before using it.
fn preheader_of(
    func: &Function,
    dom: &DomInfo,
    header: Block,
    loop_blocks: &HashSet<Block>,
) -> Option<Block> {
    let mut outside: Vec<Block> = loop_blocks
        .iter()
        .flat_map(|b| dom.preds[b.index()].iter().copied())
        .collect();
    outside.retain(|p| !loop_blocks.contains(p));
    outside.sort_by_key(|b| b.index());
    outside.dedup();
    if outside.len() != 1 {
        return None;
    }
    let preheader = outside[0];
    if !dominates(dom, preheader, header, func.entry) {
        return None;
    }
    if func.blocks[preheader].term.is_none() {
        return None;
    }
    Some(preheader)
}

/// Hoist one loop's invariants. Candidates repeat: an instruction
/// hoisted earlier in this sweep leaves the loop blocks behind, so its
/// own users become invariant until the sweep stops moving.
fn hoist_loop(
    func: &mut Function,
    preheader: Block,
    loop_blocks: &HashSet<Block>,
    uses: &HashMap<Value, usize>,
) -> bool {
    let mut changed = false;
    let mut order: Vec<Block> = loop_blocks.iter().copied().collect();
    order.sort_by_key(|b| b.index());
    loop {
        let mut candidates: Vec<Inst> = Vec::new();
        for &block in &order {
            for i in 0..func.blocks[block].insts.len() {
                let inst = func.blocks[block].insts[i];
                if should_hoist(func, inst, loop_blocks, uses) {
                    candidates.push(inst);
                }
            }
        }
        if candidates.is_empty() {
            return changed;
        }
        for inst in candidates {
            let block = func.insts[inst].block;
            if let Some(pos) = func.blocks[block].insts.iter().position(|i| *i == inst) {
                func.blocks[block].insts.remove(pos);
            }
            func.insts[inst].block = preheader;
            func.blocks[preheader].insts.push(inst);
        }
        changed = true;
    }
}

fn should_hoist(
    func: &Function,
    inst: Inst,
    loop_blocks: &HashSet<Block>,
    uses: &HashMap<Value, usize>,
) -> bool {
    let Some(result) = func.insts[inst].result else {
        return false;
    };
    if uses.get(&result).copied().unwrap_or(0) == 0 {
        return false;
    }
    if !func.insts[inst].kind.is_pure() {
        return false;
    }
    let mut invariant = true;
    func.insts[inst].kind.for_each_operand(&mut |op| {
        if let Some(def) = def_block(func, op) {
            if loop_blocks.contains(&def) {
                invariant = false;
            }
        }
    });
    invariant
}

/// The block an operand's definition lives in; constants have none.
fn def_block(func: &Function, v: Value) -> Option<Block> {
    match &func.values[v].def {
        Def::Const(_) => None,
        Def::Param(block, _) => Some(*block),
        Def::Inst(inst) => Some(func.insts[*inst].block),
    }
}

/// Dominator walk between a pair of blocks.
fn dominates(dom: &DomInfo, ancestor: Block, node: Block, entry: Block) -> bool {
    let mut cur = node;
    loop {
        if cur == ancestor {
            return true;
        }
        if cur == entry {
            return false;
        }
        match dom.idom[cur.index()] {
            Some(next) => cur = next,
            None => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Const, IcmpPred, InstData, InstKind, Terminator, ValueData};
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

    fn kinds_in(func: &Function, block: Block) -> Vec<InstKind> {
        func.blocks[block]
            .insts
            .iter()
            .map(|i| func.insts[*i].kind.clone())
            .collect()
    }

    /// `while (b) { tripled := x*3; bumped := tripled+1; use_cmp(read,
    /// bumped); }`: the mul, its add user, and nothing else hoist. The
    /// in-loop load stays, and the comparison reading both sides
    /// (bumped from outside after the hoist, the load inside) stays.
    #[test]
    fn hoists_invariants_to_the_preheader() {
        let sig = Signature::new(vec![Ty::int(32), Ty::ptr(Ty::int(32)), Ty::Bool], Ty::Void);
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(32), 0);
        let p = param(&mut func, entry, Ty::ptr(Ty::int(32)), 1);
        let cond = param(&mut func, entry, Ty::Bool, 2);
        let header = func.push_block(vec![]);
        let latch = func.push_block(vec![]);
        let exit = func.push_block(vec![]);
        func.blocks[entry].term = Some(Terminator::Br {
            dest: header,
            args: Default::default(),
        });
        func.blocks[header].term = Some(Terminator::CondBr {
            cond,
            then_block: latch,
            else_block: exit,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        let c3 = func.push_const(Const::Int { bits: 32, value: 3 });
        let c1 = func.push_const(Const::Int { bits: 32, value: 1 });
        let tripled = emit(&mut func, latch, InstKind::Mul(x, c3), Ty::int(32));
        let bumped = emit(&mut func, latch, InstKind::Add(tripled, c1), Ty::int(32));
        let read = emit(
            &mut func,
            latch,
            InstKind::Load {
                addr: p,
                flags: crate::ir::MemFlags::plain(),
            },
            Ty::int(32),
        );
        let use_cmp = emit(
            &mut func,
            latch,
            InstKind::Icmp {
                pred: IcmpPred::Ne,
                lhs: bumped,
                rhs: read,
            },
            Ty::Bool,
        );
        // The comparison feeds the back edge, closing the loop chain.
        let walk = func.push_block(vec![]);
        func.blocks[latch].term = Some(Terminator::CondBr {
            cond: use_cmp,
            then_block: walk,
            else_block: header,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        func.blocks[walk].term = Some(Terminator::Br {
            dest: header,
            args: Default::default(),
        });
        func.blocks[exit].term = Some(Terminator::Return { val: None });

        assert!(Licm.run(&mut func));
        assert!(verify_func(&func).is_empty());
        let entry_kinds = kinds_in(&func, entry);
        assert!(
            entry_kinds.iter().any(|k| matches!(k, InstKind::Mul(..))),
            "mul hoisted: {entry_kinds:?}"
        );
        assert!(
            entry_kinds.iter().any(|k| matches!(k, InstKind::Add(..))),
            "add hoisted: {entry_kinds:?}"
        );
        let latch_kinds = kinds_in(&func, latch);
        assert!(matches!(latch_kinds[0], InstKind::Load { .. }), "{latch_kinds:?}");
        assert!(matches!(latch_kinds[1], InstKind::Icmp { .. }), "{latch_kinds:?}");
        assert_eq!(latch_kinds.len(), 2);
        assert!(!Licm.run(&mut func), "second run is a no-op");
        assert!(verify_func(&func).is_empty());
    }

    /// Division may trap: it is not pure, so it and everything reading
    /// it stay in the loop whatever their operands are.
    #[test]
    fn keeps_trapping_arithmetic_in_the_loop() {
        let sig = Signature::new(vec![Ty::int(32), Ty::Bool], Ty::Void);
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(32), 0);
        let cond = param(&mut func, entry, Ty::Bool, 1);
        let header = func.push_block(vec![]);
        let latch = func.push_block(vec![]);
        let exit = func.push_block(vec![]);
        func.blocks[entry].term = Some(Terminator::Br {
            dest: header,
            args: Default::default(),
        });
        let c3 = func.push_const(Const::Int { bits: 32, value: 3 });
        let quotient = emit(&mut func, header, InstKind::Sdiv(x, c3), Ty::int(32));
        let _twice = emit(&mut func, header, InstKind::Add(quotient, quotient), Ty::int(32));
        func.blocks[header].term = Some(Terminator::CondBr {
            cond,
            then_block: latch,
            else_block: exit,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        func.blocks[latch].term = Some(Terminator::Br {
            dest: header,
            args: Default::default(),
        });
        func.blocks[exit].term = Some(Terminator::Return { val: None });

        assert!(!Licm.run(&mut func));
        assert!(verify_func(&func).is_empty());
        assert_eq!(kinds_in(&func, entry).len(), 0);
        assert_eq!(kinds_in(&func, header).len(), 2);
    }

    /// Two ways into the loop: there is no preheader, so nothing may
    /// move (some loop entry would never see the hoisted work).
    #[test]
    fn skips_loops_with_several_entry_edges() {
        let sig = Signature::new(vec![Ty::int(32), Ty::Bool], Ty::Void);
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(32), 0);
        let cond = param(&mut func, entry, Ty::Bool, 1);
        let second_entry = func.push_block(vec![]);
        let header = func.push_block(vec![]);
        let latch = func.push_block(vec![]);
        let exit = func.push_block(vec![]);
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block: header,
            else_block: second_entry,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        let skip = func.push_const(Const::Int { bits: 1, value: 0 });
        func.blocks[second_entry].term = Some(Terminator::CondBr {
            cond: skip,
            then_block: header,
            else_block: exit,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        let c3 = func.push_const(Const::Int { bits: 32, value: 3 });
        let tripled = emit(&mut func, latch, InstKind::Mul(x, c3), Ty::int(32));
        let _rising = emit(
            &mut func,
            latch,
            InstKind::Icmp {
                pred: IcmpPred::Ne,
                lhs: tripled,
                rhs: x,
            },
            Ty::Bool,
        );
        func.blocks[header].term = Some(Terminator::CondBr {
            cond,
            then_block: latch,
            else_block: exit,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        func.blocks[latch].term = Some(Terminator::Br {
            dest: header,
            args: Default::default(),
        });
        func.blocks[exit].term = Some(Terminator::Return { val: None });

        assert!(!Licm.run(&mut func));
        assert!(verify_func(&func).is_empty());
        assert_eq!(kinds_in(&func, latch).len(), 2);
    }

    /// An operand defined inside the loop stays behind until its own
    /// chain can hoist: `x ^ x` leaves first, then `mul` follows it in
    /// the same sweep, and the comparison (which reads the mul in the
    /// loop) never does.
    #[test]
    fn hoists_unblocked_chains_within_one_sweep() {
        let sig = Signature::new(vec![Ty::int(32), Ty::Bool], Ty::Void);
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(32), 0);
        let cond = param(&mut func, entry, Ty::Bool, 1);
        let header = func.push_block(vec![]);
        let latch = func.push_block(vec![]);
        let exit = func.push_block(vec![]);
        func.blocks[entry].term = Some(Terminator::Br {
            dest: header,
            args: Default::default(),
        });
        let shaken = emit(&mut func, latch, InstKind::Xor(x, x), Ty::int(32));
        let squared = emit(&mut func, latch, InstKind::Mul(shaken, shaken), Ty::int(32));
        // Dead in the CFG (nothing reads it), so it never hoists.
        let _rising = emit(
            &mut func,
            latch,
            InstKind::Icmp {
                pred: IcmpPred::Ne,
                lhs: squared,
                rhs: x,
            },
            Ty::Bool,
        );
        func.blocks[header].term = Some(Terminator::CondBr {
            cond,
            then_block: latch,
            else_block: exit,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        func.blocks[latch].term = Some(Terminator::Br {
            dest: header,
            args: Default::default(),
        });
        func.blocks[exit].term = Some(Terminator::Return { val: None });

        assert!(Licm.run(&mut func));
        assert!(verify_func(&func).is_empty());
        let entry_kinds = kinds_in(&func, entry);
        assert_eq!(entry_kinds.len(), 2, "{entry_kinds:?}");
        let latch_kinds = kinds_in(&func, latch);
        assert!(matches!(latch_kinds.last(), Some(InstKind::Icmp { .. })));
        assert_eq!(latch_kinds.len(), 1, "{latch_kinds:?}");
    }

    /// A self-loop (`while (1) double := x+x;`): the back edge is the
    /// header's own edge, the loop body is just the header, and the
    /// invariant leaves through the entry.
    #[test]
    fn hoists_from_a_self_loop() {
        let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(32), 0);
        let header = func.push_block(vec![]);
        let exit = func.push_block(vec![]);
        func.blocks[entry].term = Some(Terminator::Br {
            dest: header,
            args: Default::default(),
        });
        let cond = func.push_const(Const::Int { bits: 1, value: 1 });
        let c2 = func.push_const(Const::Int { bits: 32, value: 2 });
        let double = emit(&mut func, header, InstKind::Mul(x, c2), Ty::int(32));
        func.blocks[header].term = Some(Terminator::CondBr {
            cond,
            then_block: header,
            else_block: exit,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        func.blocks[exit].term = Some(Terminator::Return { val: Some(double) });
        assert!(Licm.run(&mut func));
        assert!(verify_func(&func).is_empty());
        // The entry grabbed the only computation; the header is empty.
        assert_eq!(kinds_in(&func, entry).len(), 1);
        assert_eq!(kinds_in(&func, header).len(), 0);
        assert!(!Licm.run(&mut func), "second run is a no-op");
    }
}
