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

use std::collections::HashSet;

use crate::ir::{Block, Def, Function, Inst, InstKind, Value};
use crate::pass::FunctionPass;

use super::{DomInfo, MemoryObject, address_info, non_escaping_stack_slots, ranges_may_overlap};

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
        if loops.is_empty() {
            return false;
        }
        loops.sort_by_key(|(header, _)| header.index());
        let uses = func.count_resident_uses();
        let non_escaping = non_escaping_stack_slots(func);
        let mut changed = false;
        for (header, loop_blocks) in loops {
            let Some(preheader) = preheader_of(func, &dom, header, &loop_blocks) else {
                continue;
            };
            changed |= hoist_loop(func, preheader, &loop_blocks, &uses, &non_escaping);
        }
        changed
    }
}

/// Every edge `n -> h` with `h` dominating `n` is a back edge; the
/// natural loop is `h` (the real loop header, the edge destination)
/// plus everything that can reach `n` without passing through `h`.
/// Predecessors explored from `n` stop at `h` itself (already in the
/// set), which is exactly that reachability.
pub(super) fn natural_loops(func: &Function, dom: &DomInfo) -> Vec<(Block, HashSet<Block>)> {
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
fn preheader_of(func: &Function, dom: &DomInfo, header: Block, loop_blocks: &HashSet<Block>) -> Option<Block> {
    let mut outside: Vec<Block> = loop_blocks.iter().flat_map(|b| dom.preds[b.index()].iter().copied()).collect();
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
    func.blocks[preheader].term.as_ref()?;
    Some(preheader)
}

/// Hoist one loop's invariants. Candidates repeat: an instruction
/// hoisted earlier in this sweep leaves the loop blocks behind, so its
/// own users become invariant until the sweep stops moving.
fn hoist_loop(func: &mut Function, preheader: Block, loop_blocks: &HashSet<Block>, uses: &[u32], non_escaping: &HashSet<Value>) -> bool {
    let mut changed = false;
    let mut order: Vec<Block> = loop_blocks.iter().copied().collect();
    order.sort_by_key(|b| b.index());
    loop {
        let mut candidates: Vec<Inst> = Vec::new();
        for &block in &order {
            for i in 0..func.blocks[block].insts.len() {
                let inst = func.blocks[block].insts[i];
                if should_hoist(func, inst, loop_blocks, uses, non_escaping) {
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

fn should_hoist(func: &Function, inst: Inst, loop_blocks: &HashSet<Block>, uses: &[u32], non_escaping: &HashSet<Value>) -> bool {
    let Some(result) = func.insts[inst].result else {
        return false;
    };
    if uses.get(result.index()).copied().unwrap_or(0) == 0 {
        return false;
    }
    let kind = &func.insts[inst].kind;
    if !kind.is_pure() && !hoistable_load(func, inst, loop_blocks, non_escaping) {
        return false;
    }
    let mut invariant = true;
    kind.for_each_operand(&mut |op| {
        if let Some(def) = def_block(func, op)
            && loop_blocks.contains(&def) {
                invariant = false;
            }
    });
    invariant
}

/// A non-volatile load from a non-escaping stack slot whose bytes no loop
/// instruction writes. Such a load cannot fault, and no callee can reach
/// the slot's memory.
fn hoistable_load(func: &Function, inst: Inst, loop_blocks: &HashSet<Block>, non_escaping: &HashSet<Value>) -> bool {
    let InstKind::Load { addr, flags } = &func.insts[inst].kind else {
        return false;
    };
    if flags.volatile {
        return false;
    }
    let Some(info) = address_info(func, *addr) else {
        return false;
    };
    let MemoryObject::Stack(slot) = info.object else {
        return false;
    };
    if !non_escaping.contains(&slot) {
        return false;
    }
    let Some(result) = func.insts[inst].result else {
        return false;
    };
    let len = func.values[result].ty.size_bytes();
    if len == 0 {
        return false;
    }
    for &block in loop_blocks {
        for &other in &func.blocks[block].insts {
            if write_may_alias(func, other, *addr, len, non_escaping) {
                return false;
            }
        }
    }
    true
}

fn write_may_alias(func: &Function, inst: Inst, addr: Value, len: u64, non_escaping: &HashSet<Value>) -> bool {
    match &func.insts[inst].kind {
        InstKind::Store { addr: write, val, .. } => ranges_may_overlap(func, non_escaping, *write, func.values[*val].ty.size_bytes(), addr, len),
        InstKind::Memcpy { dst, len: write_len, .. } | InstKind::Memset { dst, len: write_len, .. } => ranges_may_overlap(func, non_escaping, *dst, *write_len, addr, len),
        InstKind::AtomicStore { addr: write, val } => ranges_may_overlap(func, non_escaping, *write, func.values[*val].ty.size_bytes(), addr, len),
        InstKind::AtomicRmw { addr: write, val, .. } => ranges_may_overlap(func, non_escaping, *write, func.values[*val].ty.size_bytes(), addr, len),
        InstKind::AtomicCas { addr: write, replacement, .. } => ranges_may_overlap(func, non_escaping, *write, func.values[*replacement].ty.size_bytes(), addr, len),
        InstKind::InlineAsm { .. } | InstKind::VaStart { .. } | InstKind::VaArg { .. } | InstKind::VaEnd { .. } | InstKind::VaCopy { .. } => true,
        _ => false,
    }
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
mod tests;

