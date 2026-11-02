//! Scalar replacement of aggregates.
//!
//! Local structs and arrays are lowered to one aggregate `Alloca` plus
//! `GetElementPtr` accesses. [`Mem2Reg`](super::Mem2Reg) only promotes
//! scalar slots, so without this pass those aggregates stay in memory
//! and every field read/write is a real load/store.
//!
//! This pass splits a non-escaping aggregate slot into one scalar slot
//! per accessed leaf. The replacement is only attempted when every use
//! of the slot is a tracked address computation ending in a scalar
//! load/store at a constant byte offset. Dynamic indices, aggregate
//! loads/stores, volatile/atomic accesses, and any address that escapes
//! into a call, a store value, or a terminator leave the slot in memory.
//! Overlapping leaves (unions, type punning) are refused as well: the
//! split slots must not alias each other.
//!
//! After the rewrite the new scalar slots are ordinary `Mem2Reg`
//! candidates, so the pipeline runs this pass immediately before
//! `Mem2Reg`.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ir::{Block, Def, Function, Inst, InstData, InstKind, Value, ValueData};
use crate::pass::FunctionPass;
use crate::ty::{GepIndex, Ty};

use super::{MemoryObject, address_info};

/// Cap the number of split slots so a large constant-indexed array
/// cannot turn one aggregate into an unbounded pile of stack slots.
const MAX_LEAVES: usize = 128;

/// Split non-escaping aggregate stack slots into scalar slots.
#[derive(Default)]
pub struct Sroa;

impl FunctionPass for Sroa {
    fn name(&self) -> &'static str {
        "sroa"
    }

    fn run(&mut self, func: &mut Function) -> bool {
        // A `longjmp` re-enters with the registers of the first
        // `setjmp` return, so locals written in between must stay in
        // memory (the same rule `Mem2Reg` follows).
        if func.returns_twice || func.blocks.is_empty() {
            return false;
        }
        let reachable = func.reachable_blocks();
        let resident = func.resident_mask();
        if !func
            .insts
            .iter_enumerated()
            .any(|(id, inst)| resident[id.index()] && matches!(&inst.kind, InstKind::Alloca { pointee, count } if *count == 1 && pointee.is_aggregate()))
        {
            return false;
        }
        let mut plans = Vec::new();
        for (id, inst) in func.insts.iter_enumerated() {
            if !resident[id.index()] || !reachable[inst.block.index()] {
                continue;
            }
            let InstKind::Alloca { pointee, count } = &inst.kind else {
                continue;
            };
            if *count != 1 || !pointee.is_aggregate() {
                continue;
            }
            let Some(root) = inst.result else {
                continue;
            };
            if let Some(plan) = plan_scalarization(func, root, id, pointee, &resident, &reachable) {
                plans.push(plan);
            }
        }
        if plans.is_empty() {
            return false;
        }
        apply(func, plans);
        true
    }
}

struct Plan {
    alloca: Inst,
    block: Block,
    loc: Option<crate::debug::Location>,
    /// Accessed leaves, ordered by byte offset. Empty when the slot is
    /// entirely unused.
    leaves: Vec<Leaf>,
    /// Address computations that become dead after the rewrite.
    derived: HashSet<Inst>,
}

#[derive(Debug)]
struct Leaf {
    ty: Ty,
    accesses: Vec<Inst>,
}

fn plan_scalarization(func: &Function, root: Value, alloca: Inst, pointee: &Ty, resident: &[bool], reachable: &[bool]) -> Option<Plan> {
    let mut derived: HashMap<Value, Option<u64>> = HashMap::new();
    derived.insert(root, Some(0));
    let mut derived_insts = HashSet::new();
    for (value, data) in func.values.iter_enumerated() {
        if value == root {
            continue;
        }
        let Def::Inst(id) = data.def else {
            continue;
        };
        if !resident[id.index()] || !reachable[func.insts[id].block.index()] {
            continue;
        }
        let Some(info) = address_info(func, value) else {
            continue;
        };
        if info.object != MemoryObject::Stack(root) {
            continue;
        }
        derived.insert(value, info.offset);
        derived_insts.insert(id);
    }
    if derived.values().any(Option::is_none) {
        return None;
    }

    let total = pointee.size_bytes();
    let mut leaves: BTreeMap<u64, Leaf> = BTreeMap::new();
    for (id, inst) in func.insts.iter_enumerated() {
        if !resident[id.index()] || !reachable[inst.block.index()] {
            continue;
        }
        match &inst.kind {
            InstKind::Load { addr, flags } => {
                if !derived.contains_key(addr) {
                    continue;
                }
                if flags.volatile {
                    return None;
                }
                let result = inst.result?;
                let ty = func.values[result].ty.clone();
                let offset = derived[addr]?;
                record_access(&mut leaves, offset, ty, id, total)?;
            }
            InstKind::Store { addr, val, flags } => {
                if derived.contains_key(val) {
                    return None;
                }
                if !derived.contains_key(addr) {
                    continue;
                }
                if flags.volatile {
                    return None;
                }
                let ty = func.values[*val].ty.clone();
                let offset = derived[addr]?;
                record_access(&mut leaves, offset, ty, id, total)?;
            }
            InstKind::GetElementPtr { base, indices } => {
                if derived.contains_key(base) {
                    let result = inst.result?;
                    if !derived.contains_key(&result) {
                        return None;
                    }
                }
                for index in indices.iter() {
                    if let GepIndex::Index(value) = index
                        && derived.contains_key(value) {
                            return None;
                        }
                }
            }
            InstKind::Bitcast(inner) => {
                if derived.contains_key(inner) {
                    let result = inst.result?;
                    if !derived.contains_key(&result) {
                        return None;
                    }
                    // A width-changing bitcast (MSVC `__ptr32`) is not
                    // address-preserving, so it cannot feed the split.
                    if func.values[*inner].ty.size_bytes() != func.values[result].ty.size_bytes() {
                        return None;
                    }
                }
            }
            InstKind::PtrToInt(inner) => {
                if derived.contains_key(inner) {
                    let result = inst.result?;
                    if !derived.contains_key(&result) || func.values[*inner].ty.ptr_width() != Some(64) {
                        return None;
                    }
                }
            }
            InstKind::IntToPtr(inner) => {
                if derived.contains_key(inner) {
                    let result = inst.result?;
                    if !derived.contains_key(&result) || func.values[result].ty.ptr_width() != Some(64) {
                        return None;
                    }
                }
            }
            InstKind::Add(a, b) | InstKind::Sub(a, b) => {
                let a_derived = derived.contains_key(a);
                let b_derived = derived.contains_key(b);
                if a_derived || b_derived {
                    if a_derived && b_derived {
                        return None;
                    }
                    let result = inst.result?;
                    if !derived.contains_key(&result) {
                        return None;
                    }
                }
            }
            other => {
                let mut uses_derived = false;
                other.for_each_operand(&mut |value| {
                    if derived.contains_key(&value) {
                        uses_derived = true;
                    }
                });
                if uses_derived {
                    return None;
                }
            }
        }
    }
    for (idx, block) in func.blocks.iter().enumerate() {
        if !reachable[idx] {
            continue;
        }
        let Some(term) = &block.term else {
            continue;
        };
        let mut uses_derived = false;
        term.for_each_operand(&mut |value| {
            if derived.contains_key(&value) {
                uses_derived = true;
            }
        });
        if uses_derived {
            return None;
        }
    }

    if leaves.len() > MAX_LEAVES {
        return None;
    }
    let mut previous_end = 0;
    for (&offset, leaf) in &leaves {
        let end = offset.checked_add(leaf.ty.size_bytes())?;
        if end > total || offset < previous_end {
            return None;
        }
        previous_end = end;
    }

    Some(Plan {
        alloca,
        block: func.insts[alloca].block,
        loc: func.insts[alloca].loc,
        leaves: leaves.into_values().collect(),
        derived: derived_insts,
    })
}

fn record_access(leaves: &mut BTreeMap<u64, Leaf>, offset: u64, ty: Ty, inst: Inst, total: u64) -> Option<()> {
    let size = ty.size_bytes();
    if !ty.is_scalar() || size == 0 || offset.checked_add(size)? > total {
        return None;
    }
    match leaves.entry(offset) {
        Entry::Vacant(entry) => {
            entry.insert(Leaf { ty, accesses: vec![inst] });
        }
        Entry::Occupied(mut entry) => {
            let leaf = entry.get_mut();
            if leaf.ty != ty {
                return None;
            }
            leaf.accesses.push(inst);
        }
    }
    Some(())
}

fn apply(func: &mut Function, plans: Vec<Plan>) {
    let mut replacements: HashMap<Inst, Vec<Inst>> = HashMap::new();
    let mut remove: HashSet<Inst> = HashSet::new();
    let mut retarget: HashMap<Inst, Value> = HashMap::new();
    for plan in plans {
        remove.insert(plan.alloca);
        remove.extend(plan.derived);
        for leaf in plan.leaves {
            let value = func.values.push(ValueData {
                ty: Ty::ptr(leaf.ty.clone()),
                def: Def::Inst(Inst::from_usize(func.insts.len())),
            });
            let alloc = func.insts.push(InstData {
                block: plan.block,
                kind: InstKind::Alloca { pointee: leaf.ty, count: 1 },
                result: Some(value),
                loc: plan.loc,
            });
            func.values[value].def = Def::Inst(alloc);
            replacements.entry(plan.alloca).or_default().push(alloc);
            for access in leaf.accesses {
                retarget.insert(access, value);
            }
        }
    }

    for (inst, slot) in retarget {
        match &mut func.insts[inst].kind {
            InstKind::Load { addr, .. } => *addr = slot,
            InstKind::Store { addr, .. } => *addr = slot,
            _ => unreachable!("SROA records only loads and stores"),
        }
    }

    for idx in 0..func.blocks.len() {
        let block = Block::from_usize(idx);
        let old = std::mem::take(&mut func.blocks[block].insts);
        let mut out = Vec::with_capacity(old.len());
        for inst in old {
            if let Some(allocs) = replacements.get(&inst) {
                out.extend(allocs.iter().copied());
            } else if !remove.contains(&inst) {
                out.push(inst);
            }
        }
        func.blocks[block].insts = out;
    }
}

#[cfg(test)]
mod tests;

