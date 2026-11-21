//! Scalar replacement of aggregates.
//!
//! Local structs and arrays are lowered to one aggregate `Alloca` plus
//! `GetElementPtr` accesses. [`Mem2Reg`](super::Mem2Reg) only promotes
//! scalar and vector slots, so without this pass those aggregates stay
//! in memory and every field read/write is a real load/store.
//!
//! This pass splits a non-escaping aggregate slot into one scalar or
//! vector slot per accessed leaf. The replacement is only attempted
//! when every use of the slot is a tracked address computation ending
//! in a load/store at a constant byte offset. Dynamic indices,
//! volatile accesses, and any address that escapes into a call, a
//! store value, or a terminator leave the slot in memory.
//! Overlapping leaves (unions, type punning) are refused as well: the
//! split slots must not alias each other.
//!
//! Whole-aggregate loads and stores split too, when the slot's other
//! accesses agree with the aggregate's field boundaries: an aggregate
//! store becomes an `ExtractValue`/leaf-store sequence, an aggregate
//! load a leaf-load/`InsertValue` reassembly. This is what lets an
//! interface struct (`prism` entry points take and return one) flow
//! through registers like any other value. Aggregates whose boundaries
//! disagree with the other accesses (punned unions, sub-leaf accesses)
//! stay in memory.
//!
//! After the rewrite the new scalar slots are ordinary `Mem2Reg`
//! candidates, so the pipeline runs this pass immediately before
//! `Mem2Reg`.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ir::{Block, Const, Def, Function, Inst, InstData, InstKind, MemFlags, Value, ValueData};
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
    /// Split leaves, ordered by byte offset. Includes leaves that only
    /// exist because an aggregate access needs them (their `accesses`
    /// is empty; the aggregate rewrite produces their loads/stores).
    leaves: Vec<Leaf>,
    /// Aggregate-typed loads and stores to rewrite into leaf accesses.
    aggs: Vec<AggAccess>,
    /// Address computations that become dead after the rewrite.
    derived: HashSet<Inst>,
}

#[derive(Debug)]
struct Leaf {
    ty: Ty,
    /// Direct (leaf-typed) loads and stores at this offset; their
    /// addresses are retargeted to the split slot.
    accesses: Vec<Inst>,
}

/// One aggregate load/store, with how it decomposes into the plan's
/// leaves.
#[derive(Debug)]
struct AggAccess {
    inst: Inst,
    is_store: bool,
    /// `(leaf index, path from this aggregate's root to the leaf)`.
    pieces: Vec<(usize, Vec<u32>)>,
}

/// How one load/store at a tracked address decomposes.
enum Access {
    Leaf,
    Aggregate,
    Refuse,
}

fn classify(ty: &Ty) -> Access {
    if ty.is_scalar() || matches!(ty, Ty::Vector { .. }) {
        return Access::Leaf;
    }
    match ty {
        Ty::Struct { fields, .. } if !fields.is_empty() => Access::Aggregate,
        Ty::Array { len, .. } if *len > 0 => Access::Aggregate,
        _ => Access::Refuse,
    }
}

/// One leaf span of a decomposed aggregate.
struct Piece {
    offset: u64,
    ty: Ty,
    path: Vec<u32>,
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
    let mut aggs: Vec<(Inst, Ty, bool)> = Vec::new();
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
                record_access(&mut leaves, &mut aggs, id, offset, ty, false, total)?;
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
                record_access(&mut leaves, &mut aggs, id, offset, ty, true, total)?;
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

    let decompositions = resolve_aggregates(&mut leaves, &aggs)?;
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

    // Attach each aggregate to its plan-leaf indices. Leaves are keyed
    // by offset, which pieces share.
    let leaf_index: HashMap<(u64, &Ty), usize> = leaves.iter().enumerate().map(|(index, (&offset, leaf))| ((offset, &leaf.ty), index)).collect();
    let mut resolved = Vec::with_capacity(aggs.len());
    for ((inst, _, is_store), pieces) in aggs.iter().zip(decompositions) {
        let mut mapped = Vec::with_capacity(pieces.len());
        for (offset, ty, path) in pieces {
            let index = *leaf_index.get(&(offset, &ty))?;
            mapped.push((index, path));
        }
        resolved.push(AggAccess { inst: *inst, is_store: *is_store, pieces: mapped });
    }

    Some(Plan {
        alloca,
        block: func.insts[alloca].block,
        loc: func.insts[alloca].loc,
        leaves: leaves.into_values().collect(),
        aggs: resolved,
        derived: derived_insts,
    })
}

/// Decompose every aggregate access into leaf spans and merge them
/// with the direct leaf accesses. Spans must tile each aggregate
/// exactly and must not partially overlap each other or any direct
/// leaf: a mismatch means the accesses do not share one field layout
/// (punning), which cannot be rewritten. Returns, per aggregate, its
/// `(offset, ty, path)` decomposition in offset order.
fn resolve_aggregates(leaves: &mut BTreeMap<u64, Leaf>, aggs: &[(Inst, Ty, bool)]) -> Option<Vec<Vec<(u64, Ty, Vec<u32>)>>> {
    let mut decompositions = Vec::with_capacity(aggs.len());
    for (_, ty, _) in aggs {
        let mut pieces = Vec::new();
        decompose_ty(ty, 0, &mut Vec::new(), &mut pieces, 0)?;
        decompositions.push(pieces);
    }
    // Spans from all aggregates must be pairwise disjoint or identical.
    let mut all: Vec<&Piece> = decompositions.iter().flatten().collect();
    all.sort_by_key(|piece| (piece.offset, piece.ty.size_bytes()));
    let mut previous_end = 0;
    let mut previous: Option<&Piece> = None;
    for piece in &all {
        let end = piece.offset.checked_add(piece.ty.size_bytes())?;
        if let Some(prev) = previous {
            if piece.offset < previous_end
                && (piece.offset != prev.offset || end != previous_end || piece.ty != prev.ty)
            {
                return None;
            }
        }
        previous = Some(piece);
        previous_end = end;
    }
    // Direct leaves must sit inside an identical piece or miss all of them.
    for (&offset, leaf) in leaves.iter() {
        let end = offset.checked_add(leaf.ty.size_bytes())?;
        for piece in &all {
            let piece_end = piece.offset + piece.ty.size_bytes();
            if offset < piece_end && piece.offset < end && (offset != piece.offset || end != piece_end || leaf.ty != piece.ty) {
                return None;
            }
        }
    }
    // Pieces not directly accessed become leaves of their own.
    for piece in &all {
        match leaves.entry(piece.offset) {
            Entry::Vacant(entry) => {
                entry.insert(Leaf { ty: piece.ty.clone(), accesses: Vec::new() });
            }
            Entry::Occupied(entry) => {
                if entry.get().ty != piece.ty {
                    return None;
                }
            }
        }
    }
    Some(
        decompositions
            .into_iter()
            .map(|pieces| pieces.into_iter().map(|piece| (piece.offset, piece.ty, piece.path)).collect())
            .collect(),
    )
}

/// Flatten `ty` into its scalar/vector leaf spans, tracking the field
/// path from the aggregate root.
fn decompose_ty(ty: &Ty, offset: u64, path: &mut Vec<u32>, out: &mut Vec<Piece>, depth: u32) -> Option<()> {
    const MAX_DEPTH: u32 = 16;
    if ty.is_scalar() || matches!(ty, Ty::Vector { .. }) {
        if ty.size_bytes() == 0 {
            return None;
        }
        out.push(Piece { offset, ty: ty.clone(), path: path.clone() });
        return Some(());
    }
    if depth >= MAX_DEPTH {
        return None;
    }
    match ty {
        Ty::Struct { fields, offsets, .. } => {
            for (index, field) in fields.iter().enumerate() {
                path.push(index as u32);
                decompose_ty(field, offset + offsets[index], path, out, depth + 1)?;
                path.pop();
            }
            Some(())
        }
        Ty::Array { elem, len } => {
            let stride = elem.size_bytes();
            if stride == 0 {
                return None;
            }
            for index in 0..*len {
                if index > u32::MAX as u64 {
                    return None;
                }
                path.push(index as u32);
                decompose_ty(elem, offset + index * stride, path, out, depth + 1)?;
                path.pop();
            }
            Some(())
        }
        _ => None,
    }
}

fn record_access(leaves: &mut BTreeMap<u64, Leaf>, aggs: &mut Vec<(Inst, Ty, bool)>, inst: Inst, offset: u64, ty: Ty, is_store: bool, total: u64) -> Option<()> {
    let size = ty.size_bytes();
    if size == 0 || offset.checked_add(size)? > total {
        return None;
    }
    match classify(&ty) {
        Access::Leaf => match leaves.entry(offset) {
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
        },
        Access::Aggregate => aggs.push((inst, ty, is_store)),
        Access::Refuse => return None,
    }
    Some(())
}

/// Everything the block rewrite needs for one aggregate access; the
/// leaf ops are emitted at the original instruction's position.
struct AggRewrite {
    is_store: bool,
    flags: MemFlags,
    loc: Option<crate::debug::Location>,
    agg_ty: Ty,
    /// The stored aggregate (stores) / the result to replace (loads).
    operand: Option<Value>,
    /// `(slot address, leaf ty, path)` per piece.
    pieces: Vec<(Value, Ty, Vec<u32>)>,
}

fn apply(func: &mut Function, plans: Vec<Plan>) {
    let mut replacements: HashMap<Inst, Vec<Inst>> = HashMap::new();
    let mut remove: HashSet<Inst> = HashSet::new();
    let mut retarget: HashMap<Inst, Value> = HashMap::new();
    let mut value_replacements: HashMap<Value, Value> = HashMap::new();
    let mut aggs: HashMap<Inst, AggRewrite> = HashMap::new();
    for plan in plans {
        remove.insert(plan.alloca);
        remove.extend(plan.derived);
        let mut slots = Vec::with_capacity(plan.leaves.len());
        for leaf in &plan.leaves {
            let value = func.values.push(ValueData {
                ty: Ty::ptr(leaf.ty.clone()),
                def: Def::Inst(Inst::from_usize(func.insts.len())),
            });
            let alloc = func.insts.push(InstData {
                block: plan.block,
                kind: InstKind::Alloca { pointee: leaf.ty.clone(), count: 1 },
                result: Some(value),
                loc: plan.loc,
            });
            func.values[value].def = Def::Inst(alloc);
            replacements.entry(plan.alloca).or_default().push(alloc);
            slots.push(value);
        }
        for (leaf_index, leaf) in plan.leaves.iter().enumerate() {
            for access in &leaf.accesses {
                retarget.insert(*access, slots[leaf_index]);
            }
        }
        for agg in plan.aggs {
            let kind = &func.insts[agg.inst].kind;
            let (flags, operand) = match kind {
                InstKind::Load { flags, .. } => (*flags, func.insts[agg.inst].result),
                InstKind::Store { flags, val, .. } => (*flags, Some(*val)),
                _ => unreachable!("SROA aggregates are loads and stores"),
            };
            let pieces = agg
                .pieces
                .iter()
                .map(|&(index, ref path)| {
                    let ty = plan.leaves[index].ty.clone();
                    (slots[index], ty, path.clone())
                })
                .collect();
            aggs.insert(
                agg.inst,
                AggRewrite {
                    is_store: agg.is_store,
                    flags,
                    loc: func.insts[agg.inst].loc,
                    agg_ty: func.values[operand.expect("aggregate access carries a value")].ty.clone(),
                    operand,
                    pieces,
                },
            );
            remove.insert(agg.inst);
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
                continue;
            }
            if let Some(agg) = aggs.get(&inst) {
                rewrite_aggregate(func, block, &mut out, agg, &mut value_replacements);
                continue;
            }
            if !remove.contains(&inst) {
                out.push(inst);
            }
        }
        func.blocks[block].insts = out;
    }

    if !value_replacements.is_empty() {
        func.remap_operands(&value_replacements);
    }
}

fn rewrite_aggregate(func: &mut Function, block: Block, out: &mut Vec<Inst>, agg: &AggRewrite, value_replacements: &mut HashMap<Value, Value>) {
    let loc = agg.loc;
    if agg.is_store {
        let stored = agg.operand.expect("aggregate store carries its value");
        for (slot, _, path) in &agg.pieces {
            let extracted = emit_extract_chain(func, block, out, stored, path, loc);
            let inst = func.insts.push(InstData {
                block,
                kind: InstKind::Store { addr: *slot, val: extracted, flags: agg.flags },
                result: None,
                loc,
            });
            out.push(inst);
        }
        return;
    }
    let result = agg.operand.expect("aggregate load has a result");
    let mut chain: Option<Value> = None;
    for (slot, leaf_ty, path) in &agg.pieces {
        let value = func.values.push(ValueData {
            ty: leaf_ty.clone(),
            def: Def::Inst(Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(InstData {
            block,
            kind: InstKind::Load { addr: *slot, flags: agg.flags },
            result: Some(value),
            loc,
        });
        func.values[value].def = Def::Inst(inst);
        out.push(inst);
        chain = Some(insert_leaf(func, block, out, chain, value, path, &agg.agg_ty, loc));
    }
    value_replacements.insert(result, chain.expect("aggregate decomposes into at least one leaf"));
}

/// `extractvalue` chain reading the leaf at `path` out of an
/// aggregate: `path = [0, 1]` gives `extractvalue(extractvalue(v, 0), 1)`.
fn emit_extract_chain(func: &mut Function, block: Block, out: &mut Vec<Inst>, agg: Value, path: &[u32], loc: Option<crate::debug::Location>) -> Value {
    let mut cur = agg;
    for &index in path {
        let ty = elem_ty(&func.values[cur].ty, index);
        cur = emit_extract(func, block, out, cur, index, &ty, loc);
    }
    cur
}

/// Grow the reassembly chain with one leaf at `path`. The chain starts
/// at an `undef` of the aggregate type (when `chain` is `None`) or at
/// the previous piece's chain, which the path digs into and rewrites.
fn insert_leaf(func: &mut Function, block: Block, out: &mut Vec<Inst>, chain: Option<Value>, leaf: Value, path: &[u32], root_ty: &Ty, loc: Option<crate::debug::Location>) -> Value {
    let cur = match chain {
        Some(v) => v,
        None => push_undef(func, root_ty),
    };
    if path.len() == 1 {
        return emit_insert(func, block, out, cur, leaf, path[0], root_ty, loc);
    }
    let sub_ty = elem_ty(root_ty, path[0]);
    let sub = match chain {
        Some(v) => emit_extract(func, block, out, v, path[0], &sub_ty, loc),
        None => push_undef(func, &sub_ty),
    };
    let updated = insert_leaf(func, block, out, Some(sub), leaf, &path[1..], &sub_ty, loc);
    emit_insert(func, block, out, cur, updated, path[0], root_ty, loc)
}

/// Element/field type at `index` of an aggregate.
fn elem_ty(ty: &Ty, index: u32) -> Ty {
    match ty {
        Ty::Struct { fields, .. } => fields[index as usize].clone(),
        Ty::Array { elem, .. } => (**elem).clone(),
        _ => unreachable!("aggregate path through non-aggregate"),
    }
}

fn push_undef(func: &mut Function, ty: &Ty) -> Value {
    func.values.push(ValueData {
        ty: ty.clone(),
        def: Def::Const(Const::Undef),
    })
}

fn emit_insert(func: &mut Function, block: Block, out: &mut Vec<Inst>, agg: Value, elem: Value, index: u32, ty: &Ty, loc: Option<crate::debug::Location>) -> Value {
    let value = func.values.push(ValueData {
        ty: ty.clone(),
        def: Def::Inst(Inst::from_usize(func.insts.len())),
    });
    let inst = func.insts.push(InstData {
        block,
        kind: InstKind::InsertValue { agg, elem, index },
        result: Some(value),
        loc,
    });
    func.values[value].def = Def::Inst(inst);
    out.push(inst);
    value
}

fn emit_extract(func: &mut Function, block: Block, out: &mut Vec<Inst>, agg: Value, index: u32, ty: &Ty, loc: Option<crate::debug::Location>) -> Value {
    let value = func.values.push(ValueData {
        ty: ty.clone(),
        def: Def::Inst(Inst::from_usize(func.insts.len())),
    });
    let inst = func.insts.push(InstData {
        block,
        kind: InstKind::ExtractValue { agg, index },
        result: Some(value),
        loc,
    });
    func.values[value].def = Def::Inst(inst);
    out.push(inst);
    value
}

#[cfg(test)]
mod tests;
