//! SLP vectorization: pack isomorphic scalar chains into 128-bit vectors.
//!
//! Seeds are runs of scalar stores to consecutive offsets of one memory
//! object (typically produced by [`LoopUnroll`](super::LoopUnroll) fully
//! unrolling a small counted loop, or by fixed-size elementwise code).
//! From each seed group the pass walks the stored values' def trees and
//! packs them when every lane matches: the same pure elementwise op with
//! packable operands, leaf loads from consecutive offsets, or one value
//! broadcast to every lane. Anything else leaves the group scalar.
//!
//! The rewrite moves memory accesses: the seed stores consolidate into
//! one vector store spliced after the last seed (and every in-block def
//! the packed tree reads) with a fresh vector load inserted there, while
//! every scalar load stays where it is. A group is therefore only taken
//! when no other memory access or call touches (or may touch) an
//! involved offset range inside that window — see [`Group::safe_against`].

use std::collections::{HashMap, HashSet};

use crate::ir::{Block, Def, Function, Inst, InstData, InstKind, MemFlags, Value, ValueData};
use crate::pass::FunctionPass;
use crate::ty::Ty;

use super::alias::{Object, resolve};
use super::vector::{ElemOp, elem_op, vector_lanes};
use super::{MemoryObject, escaping_addresses};

/// How one lane group's value tree packs.
enum PackNode {
    /// One vector load covering every lane, based at the lowest lane's
    /// address.
    Load { addr: Value },
    /// One scalar broadcast to all lanes.
    Splat(Value),
    /// The same elementwise op in every lane, over packable operands.
    Op(ElemOp, Vec<PackNode>),
}

/// Where a scalar memory access sits for the safety sweep. Its byte range
/// is known only when the address offset is; an access with no resolvable
/// object (a call, inline asm) may touch anything.
#[derive(Clone, Copy)]
struct Access {
    /// The instruction behind the access, or `None` for a fresh vector
    /// load that does not exist yet.
    inst: Option<Inst>,
    pos: usize,
    writes: bool,
    object: Option<Object>,
    offset: Option<u64>,
    size: u64,
}

impl Access {
    fn overlaps(&self, other: &Access) -> bool {
        match (self.object, other.object) {
            // Opaque roots and unresolved addresses may alias anything.
            (Some(Object::Opaque(_)), _) | (_, Some(Object::Opaque(_))) => true,
            (None, _) | (_, None) => true,
            (Some(Object::Known(a)), Some(Object::Known(b))) if a == b => {
                let (Some(x), Some(y)) = (self.offset, other.offset) else {
                    return true;
                };
                x < y.saturating_add(other.size) && y < x.saturating_add(self.size)
            }
            (Some(Object::Known(_)), Some(Object::Known(_))) => false,
        }
    }
}

/// A viable seed group: which scalar stores it replaces, the packed tree
/// to emit, and where.
struct Group {
    /// The seed stores, consumed by the rewrite.
    seeds: Vec<Inst>,
    /// The leaf loads the packed tree replaces with one vector load.
    leaves: Vec<Inst>,
    /// Memory ranges the fresh vector load reads.
    reads: Vec<Access>,
    /// Memory ranges the seed stores wrote.
    writes: Vec<Access>,
    /// Position in the block where the packed tree is spliced: past the
    /// last seed and every in-block def the tree reads.
    emit_at: usize,
    node: PackNode,
    elem: Ty,
    lanes: u32,
    addr: Value,
    flags: MemFlags,
}

impl Group {
    /// Whether consolidating the seed stores and inserting the vector
    /// load at `emit_at` preserves every other access's view of memory.
    /// Scalar leaf loads do not move, so only accesses between the
    /// earliest involved position and `emit_at` can be reordered across:
    /// a write crossing any involved range changes what the fresh vector
    /// load sees or what survives the store, and a read crossing a seed
    /// range sees a store that now happens later. Accesses with no
    /// resolvable object (calls, inline asm) count as crossing unless
    /// every involved object is a stack slot that never escapes.
    fn safe_against(&self, others: &[Access], escapes: &[bool], own: &[Inst]) -> bool {
        let involved = self.reads.iter().chain(&self.writes);
        let window = involved.clone().map(|a| a.pos).min().unwrap_or(self.emit_at);
        let stack_only = involved.clone().all(|a| match a.object {
            Some(Object::Known(MemoryObject::Stack(v))) => !escapes[v.index()],
            _ => false,
        });
        for other in others {
            if other.pos < window || other.pos > self.emit_at || other.inst.is_some_and(|inst| own.contains(&inst)) {
                continue;
            }
            if !self.reads.iter().chain(&self.writes).any(|a| other.overlaps(a)) {
                continue;
            }
            if other.writes {
                return false;
            }
            if self.writes.iter().any(|a| other.overlaps(a)) {
                return false;
            }
            if other.object.is_none() && !stack_only {
                return false;
            }
        }
        true
    }
}

/// SLP vectorization of straight-line scalar chains.
#[derive(Default)]
pub struct Slp;

impl FunctionPass for Slp {
    fn name(&self) -> &'static str {
        "slp"
    }

    fn run(&mut self, func: &mut Function) -> bool {
        let escapes = escaping_addresses(func);
        let mut changed = false;
        for idx in 0..func.blocks.len() {
            let block = Block::from_usize(idx);
            let old = std::mem::take(&mut func.blocks[block].insts);
            let mut groups = find_groups(func, block, &old, &escapes);
            let mut skipped: HashSet<Inst> = HashSet::new();
            let mut splice: HashMap<usize, Vec<Inst>> = HashMap::new();
            for group in groups.drain(..) {
                let emit_at = group.emit_at;
                skipped.extend(&group.seeds);
                let insts = materialize(func, block, group);
                splice.entry(emit_at).or_default().extend(insts);
            }
            let mut out = Vec::with_capacity(old.len());
            let len = old.len();
            for (pos, inst) in old.into_iter().enumerate() {
                if let Some(mut insts) = splice.remove(&pos) {
                    out.append(&mut insts);
                }
                if skipped.remove(&inst) {
                    changed = true;
                    continue;
                }
                out.push(inst);
            }
            // A splice point of `len` means the tree lands at the very end
            // of the block.
            if let Some(mut insts) = splice.remove(&len) {
                out.append(&mut insts);
            }
            func.blocks[block].insts = out;
        }
        changed
    }
}

/// Seed runs of consecutive-offset stores, packed into groups and filtered
/// by the safety check. A store already holding a vector value is not a
/// seed (no supported element shape), which keeps the pass idempotent.
fn find_groups(func: &Function, block: Block, insts: &[Inst], escapes: &[bool]) -> Vec<Group> {
    // Result value -> position, so packed leaf loads know where they read.
    let mut pos_of: HashMap<Value, usize> = HashMap::new();
    let mut stores: Vec<(usize, Inst, Value, Value, MemFlags)> = Vec::new();
    for (pos, &inst) in insts.iter().enumerate() {
        match &func.insts[inst].kind {
            InstKind::Store { addr, val, flags } if !flags.volatile => {
                stores.push((pos, inst, *addr, *val, *flags));
            }
            _ => {
                if let Some(result) = func.insts[inst].result {
                    pos_of.insert(result, pos);
                }
            }
        }
    }
    // Group candidate seeds by object and element type, then find runs of
    // consecutive offsets inside each.
    let mut by_key: HashMap<(Object, Ty), Vec<(usize, Inst, Value, Value, MemFlags, u64)>> = HashMap::new();
    for (pos, inst, addr, val, flags) in stores {
        let Some((object, Some(offset))) = resolve(func, addr) else { continue };
        let elem = func.values[val].ty.clone();
        by_key.entry((object, elem)).or_default().push((pos, inst, addr, val, flags, offset));
    }
    let mut candidates: Vec<Group> = Vec::new();
    for ((_, elem), mut group) in by_key {
        let Some(lanes) = vector_lanes(&elem) else { continue };
        let step = elem.size_bytes();
        group.sort_by_key(|&(_, _, _, _, _, off)| off);
        let mut run: Vec<(usize, Inst, Value, Value, MemFlags, u64)> = Vec::new();
        for entry in group {
            if let Some(last) = run.last() {
                if entry.5 != last.5 + step {
                    chunk_run(func, block, std::mem::take(&mut run), &elem, lanes, &pos_of, &mut candidates);
                }
            }
            run.push(entry);
        }
        chunk_run(func, block, run, &elem, lanes, &pos_of, &mut candidates);
    }
    // A leaf load reading bytes a later seed of the same group writes saw
    // the post-store value; the fresh vector load sees the pre-store one.
    candidates.retain(|group| {
        group.reads.iter().all(|read| group.writes.iter().all(|write| write.pos >= read.pos || !read.overlaps(write)))
    });
    // Other accesses in the block decide the rest.
    let others: Vec<Access> = insts
        .iter()
        .enumerate()
        .filter_map(|(pos, &inst)| inst_access(func, inst, pos))
        .collect();
    candidates.retain(|group| {
        let own: Vec<Inst> = group.seeds.iter().chain(&group.leaves).copied().collect();
        group.safe_against(&others, escapes, &own)
    });
    // Emit in block order and drop any group whose seeds another group
    // already claimed.
    candidates.sort_by_key(|group| group.emit_at);
    let mut taken: HashSet<Inst> = HashSet::new();
    candidates.retain(|group| {
        if group.seeds.iter().any(|seed| taken.contains(seed)) {
            return false;
        }
        taken.extend(&group.seeds);
        true
    });
    candidates
}

/// Cut one consecutive run into full lane-count chunks, each becoming a
/// group when its stored values pack.
#[allow(clippy::type_complexity)]
fn chunk_run(
    func: &Function,
    block: Block,
    run: Vec<(usize, Inst, Value, Value, MemFlags, u64)>,
    elem: &Ty,
    lanes: u32,
    pos_of: &HashMap<Value, usize>,
    out: &mut Vec<Group>,
) {
    for chunk in run.chunks(lanes as usize) {
        if chunk.len() != lanes as usize {
            break;
        }
        let mut emit_at = chunk.last().unwrap().0;
        let addr = chunk[0].2;
        let values: Vec<Value> = chunk.iter().map(|&(_, _, _, val, _, _)| val).collect();
        let mut leaves: Vec<Inst> = Vec::new();
        let mut reads: Vec<Access> = Vec::new();
        let Some(node) = pack(func, block, &values, elem, 0, pos_of, &mut leaves, &mut reads) else {
            continue;
        };
        // Seeds are offset-sorted, so block order need not match: the
        // packed store can consume values defined after the last seed's
        // slot. Splice after every in-block def the tree reads instead.
        let mut roots = vec![addr];
        scalar_roots(&node, &mut roots);
        for root in roots {
            if let Some(&pos) = pos_of.get(&root) {
                emit_at = emit_at.max(pos + 1);
            }
        }
        let writes = chunk
            .iter()
            .map(|&(pos, inst, addr, val, _, offset)| Access {
                inst: Some(inst),
                pos,
                writes: true,
                object: resolve(func, addr).map(|(object, _)| object),
                offset: Some(offset),
                size: func.values[val].ty.size_bytes(),
            })
            .collect();
        out.push(Group {
            seeds: chunk.iter().map(|&(_, inst, _, _, _, _)| inst).collect(),
            leaves,
            reads,
            writes,
            emit_at,
            node,
            elem: elem.clone(),
            lanes,
            addr,
            flags: chunk.last().unwrap().4,
        });
    }
}

/// One scalar memory access of the block, for the safety sweep.
fn inst_access(func: &Function, inst: Inst, pos: usize) -> Option<Access> {
    let (writes, addr, size) = match &func.insts[inst].kind {
        InstKind::Load { addr, .. } => (false, *addr, func.values[func.insts[inst].result?].ty.size_bytes()),
        InstKind::Store { addr, val, .. } => (true, *addr, func.values[*val].ty.size_bytes()),
        InstKind::Memcpy { dst, .. } | InstKind::Memset { dst, .. } => (true, *dst, u64::MAX),
        InstKind::AtomicLoad { addr } => (false, *addr, func.values[func.insts[inst].result?].ty.size_bytes()),
        InstKind::AtomicStore { addr, val } => (true, *addr, func.values[*val].ty.size_bytes()),
        InstKind::AtomicRmw { addr, .. } | InstKind::AtomicCas { addr, .. } => (true, *addr, u64::MAX),
        // A call or inline-asm block may read and write any memory the
        // function can name.
        InstKind::Call { .. } | InstKind::CallIndirect { .. } | InstKind::CallExtern { .. } | InstKind::InlineAsm { .. } => {
            return Some(Access { inst: Some(inst), pos, writes: true, object: None, offset: None, size: u64::MAX });
        }
        _ => return None,
    };
    let object = resolve(func, addr).map(|(object, _)| object);
    Some(Access {
        inst: Some(inst),
        pos,
        writes,
        object,
        offset: resolve(func, addr).and_then(|(_, offset)| offset),
        size,
    })
}

/// Try to pack one lane group's values into a [`PackNode`]. `leaves`
/// collects the leaf load instructions the packed tree replaces and
/// `reads` the byte ranges the fresh vector loads will cover.
#[allow(clippy::too_many_arguments)]
fn pack(
    func: &Function,
    block: Block,
    values: &[Value],
    elem: &Ty,
    depth: u32,
    pos_of: &HashMap<Value, usize>,
    leaves: &mut Vec<Inst>,
    reads: &mut Vec<Access>,
) -> Option<PackNode> {
    if depth > 6 {
        return None;
    }
    // One value everywhere: broadcast it.
    if values.iter().all(|&v| v == values[0]) {
        return Some(PackNode::Splat(values[0]));
    }
    // Identical constants broadcast too; mixed constants have no packed
    // form without a constant-vector construct.
    let consts: Option<Vec<_>> = values.iter().map(|&v| const_of(func, v)).collect();
    if let Some(consts) = consts {
        if consts.iter().all(|c| Some(c) == consts.first()) {
            return Some(PackNode::Splat(values[0]));
        }
        return None;
    }
    // Adjacent loads become one vector load.
    if let Some(addr) = pack_loads(func, block, values, elem, pos_of, leaves, reads) {
        return Some(PackNode::Load { addr });
    }
    // The same elementwise op in every lane, defined in this block, over
    // packable operands. Packed ops are pure and get duplicated when a
    // lane result still has scalar users; `DeadCodeElim` reaps what the
    // rewrite orphans.
    let defs: Option<Vec<(Inst, ElemOp)>> = values
        .iter()
        .map(|&v| match &func.values[v].def {
            Def::Inst(inst) if func.insts[*inst].block == block && func.insts[*inst].result == Some(v) => {
                elem_op(&func.insts[*inst].kind).map(|op| (*inst, op))
            }
            _ => None,
        })
        .collect();
    let defs = defs?;
    let op = defs[0].1;
    if defs.iter().any(|&(_, o)| o != op) {
        return None;
    }
    let mut slots: [Vec<Value>; 2] = [Vec::with_capacity(values.len()), Vec::with_capacity(values.len())];
    for &(inst, op) in &defs {
        let [a, b] = op.operands(&func.insts[inst].kind)?;
        slots[0].push(a);
        slots[1].push(b);
    }
    let children = slots
        .into_iter()
        .map(|slot| pack(func, block, &slot, elem, depth + 1, pos_of, leaves, reads))
        .collect::<Option<Vec<_>>>()?;
    Some(PackNode::Op(op, children))
}

/// All lanes loading from the same object at consecutive offsets become a
/// vector load based at the lowest offset.
fn pack_loads(
    func: &Function,
    block: Block,
    values: &[Value],
    elem: &Ty,
    pos_of: &HashMap<Value, usize>,
    leaves: &mut Vec<Inst>,
    reads: &mut Vec<Access>,
) -> Option<Value> {
    let step = elem.size_bytes();
    let mut base: Option<(Object, u64, Value)> = None;
    for &v in values {
        let Def::Inst(inst) = func.values[v].def else { return None };
        let data = &func.insts[inst];
        let InstKind::Load { addr, flags } = &data.kind else { return None };
        if flags.volatile || data.block != block || data.result != Some(v) {
            return None;
        }
        let Some((object, Some(offset))) = resolve(func, *addr) else { return None };
        match base {
            None => base = Some((object, offset, *addr)),
            // Lanes through one opaque root share a base; lanes through
            // distinct roots may hold the same address and never pack.
            // `expected` runs one step ahead of the lanes seen so far.
            Some((base_object, expected, _)) if base_object == object && offset == expected => {}
            _ => return None,
        }
        if let Some((_, expected, _)) = &mut base {
            *expected += step;
        }
    }
    let (object, offset, addr) = base?;
    let pos = values.iter().filter_map(|&v| pos_of.get(&v)).min().copied().unwrap_or(0);
    leaves.extend(values.iter().filter_map(|&v| match &func.values[v].def {
        Def::Inst(inst) => Some(*inst),
        _ => None,
    }));
    reads.push(Access {
        inst: None,
        pos,
        writes: false,
        object: Some(object),
        offset: Some(offset),
        size: step * values.len() as u64,
    });
    Some(addr)
}

/// The constant a value is defined as, if it is one.
fn const_of(func: &Function, v: Value) -> Option<crate::ir::Const> {
    match &func.values[v].def {
        Def::Const(c) => Some(c.clone()),
        _ => None,
    }
}

/// The scalar values the packed tree reads as-is: splat sources and the
/// addresses of vector loads. Op nodes consume only tree-internal results.
fn scalar_roots(node: &PackNode, out: &mut Vec<Value>) {
    match node {
        PackNode::Splat(v) => out.push(*v),
        PackNode::Load { addr } => out.push(*addr),
        PackNode::Op(_, children) => children.iter().for_each(|child| scalar_roots(child, out)),
    }
}

/// Emit a group's packed tree as real instructions, in topological order,
/// ending with the vector store. The caller splices the returned ids into
/// the block at the group's emit position.
fn materialize(func: &mut Function, block: Block, group: Group) -> Vec<Inst> {
    let ty = Ty::vector(group.elem.clone(), group.lanes);
    let mut out = Vec::new();
    {
        let mut emit = |func: &mut Function, kind: InstKind, result: Option<Ty>| {
            let result = result.map(|ty| func.values.push(ValueData { ty, def: Def::Inst(Inst::from_usize(func.insts.len())) }));
            let inst = func.insts.push(InstData { block, kind, result, loc: None });
            out.push(inst);
            result
        };
        let value = emit_node(func, &group.node, ty, &mut emit);
        emit(func, InstKind::Store { addr: group.addr, val: value, flags: group.flags }, None);
    }
    out
}

/// Depth-first emission of one [`PackNode`], returning its vector value.
fn emit_node(func: &mut Function, node: &PackNode, ty: Ty, emit: &mut impl FnMut(&mut Function, InstKind, Option<Ty>) -> Option<Value>) -> Value {
    match node {
        PackNode::Splat(v) => emit(func, InstKind::Splat(*v), Some(ty)).unwrap(),
        PackNode::Load { addr } => emit(func, InstKind::Load { addr: *addr, flags: MemFlags::plain() }, Some(ty)).unwrap(),
        PackNode::Op(op, children) => {
            let values: Vec<Value> = children.iter().map(|child| emit_node(func, child, ty.clone(), emit)).collect();
            emit(func, op.kind(values[0], values[1]), Some(ty)).unwrap()
        }
    }
}

#[cfg(test)]
mod tests;
