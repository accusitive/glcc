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
                    if let GepIndex::Index(value) = index {
                        if derived.contains_key(value) {
                            return None;
                        }
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
        loc: func.insts[alloca].loc.clone(),
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
                loc: plan.loc.clone(),
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
mod tests {
    use super::*;
    use crate::ir::{IcmpPred, MemFlags, Terminator};
    use crate::ty::Signature;
    use crate::verify::verify_func;

    fn param(func: &mut Function, block: Block, ty: Ty, index: u32) -> Value {
        let value = func.values.push(ValueData {
            ty,
            def: Def::Param(block, index),
        });
        func.blocks[block].params.push(value);
        value
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

    fn emit_unit(func: &mut Function, block: Block, kind: InstKind) -> Inst {
        let inst = func.insts.push(InstData {
            block,
            kind,
            result: None,
            loc: None,
        });
        func.blocks[block].insts.push(inst);
        inst
    }

    fn iconst(func: &mut Function, bits: u32, value: i128) -> Value {
        func.push_const(crate::ir::Const::Int { bits, value })
    }

    fn alloca(func: &mut Function, block: Block, ty: Ty) -> (Inst, Value) {
        let result = func.values.push(ValueData {
            ty: Ty::ptr(ty.clone()),
            def: Def::Inst(Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(InstData {
            block,
            kind: InstKind::Alloca { pointee: ty, count: 1 },
            result: Some(result),
            loc: None,
        });
        func.values[result].def = Def::Inst(inst);
        func.blocks[block].insts.push(inst);
        (inst, result)
    }

    fn gep(func: &mut Function, block: Block, base: Value, indices: Vec<GepIndex>, ty: Ty) -> Value {
        emit(
            func,
            block,
            InstKind::GetElementPtr {
                base,
                indices: indices.into_iter().collect(),
            },
            Ty::ptr(ty),
        )
    }

    fn load(func: &mut Function, block: Block, addr: Value, ty: Ty) -> Value {
        emit(func, block, InstKind::Load { addr, flags: MemFlags::plain() }, ty)
    }

    fn store(func: &mut Function, block: Block, addr: Value, val: Value) -> Inst {
        emit_unit(
            func,
            block,
            InstKind::Store {
                addr,
                val,
                flags: MemFlags::plain(),
            },
        )
    }

    fn run(func: &mut Function) -> bool {
        let changed = Sroa.run(func);
        let errors = verify_func(func);
        assert!(errors.is_empty(), "verifier: {errors:?}");
        changed
    }

    fn resident_kinds(func: &Function) -> impl Iterator<Item = &InstKind> {
        func.blocks.iter().flat_map(|block| block.insts.iter()).map(|inst| &func.insts[*inst].kind)
    }

    fn has_alloca(func: &Function) -> bool {
        resident_kinds(func).any(|kind| matches!(kind, InstKind::Alloca { .. }))
    }

    fn has_aggregate_alloca(func: &Function) -> bool {
        resident_kinds(func).any(|kind| matches!(kind, InstKind::Alloca { pointee, .. } if pointee.is_aggregate()))
    }

    fn has_memory_ops(func: &Function) -> bool {
        resident_kinds(func).any(|kind| matches!(kind, InstKind::Alloca { .. } | InstKind::Load { .. } | InstKind::Store { .. }))
    }

    fn point_ty() -> Ty {
        Ty::Struct {
            fields: vec![Ty::int(32), Ty::int(32)],
            offsets: vec![0, 4],
            size: 8,
        }
    }

    #[test]
    fn splits_struct_fields_and_mem2reg_finishes() {
        let point = point_ty();
        let sig = Signature::new(vec![Ty::int(32), Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("fields", sig);
        let a = param(&mut func, entry, Ty::int(32), 0);
        let b = param(&mut func, entry, Ty::int(32), 1);
        let (_, root) = alloca(&mut func, entry, point.clone());
        let x = gep(&mut func, entry, root, vec![GepIndex::Field(0)], Ty::int(32));
        let y = gep(&mut func, entry, root, vec![GepIndex::Field(1)], Ty::int(32));
        store(&mut func, entry, x, a);
        store(&mut func, entry, y, b);
        let xv = load(&mut func, entry, x, Ty::int(32));
        let yv = load(&mut func, entry, y, Ty::int(32));
        let sum = emit(&mut func, entry, InstKind::Add(xv, yv), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });

        assert!(run(&mut func));
        assert!(!has_aggregate_alloca(&func), "aggregate alloca survived");
        assert!(verify_func(&func).is_empty());
        assert!(!run(&mut func), "second run is a no-op");

        let mut mem2reg = super::super::Mem2Reg;
        assert!(mem2reg.run(&mut func));
        assert!(!has_memory_ops(&func), "memory traffic survived mem2reg");
        assert!(verify_func(&func).is_empty());
    }

    #[test]
    fn splits_fixed_array_elements() {
        let array = Ty::Array {
            elem: Box::new(Ty::int(32)),
            len: 3,
        };
        let sig = Signature::new(vec![Ty::int(32), Ty::int(32), Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("array", sig);
        let a = param(&mut func, entry, Ty::int(32), 0);
        let b = param(&mut func, entry, Ty::int(32), 1);
        let c = param(&mut func, entry, Ty::int(32), 2);
        let (_, root) = alloca(&mut func, entry, array);
        for (index, value) in [(0, a), (1, b), (2, c)] {
            let addr = gep(&mut func, entry, root, vec![GepIndex::ConstIndex(index)], Ty::int(32));
            store(&mut func, entry, addr, value);
        }
        let first = gep(&mut func, entry, root, vec![GepIndex::ConstIndex(0)], Ty::int(32));
        let third = gep(&mut func, entry, root, vec![GepIndex::ConstIndex(2)], Ty::int(32));
        let x = load(&mut func, entry, first, Ty::int(32));
        let z = load(&mut func, entry, third, Ty::int(32));
        let sum = emit(&mut func, entry, InstKind::Add(x, z), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });

        assert!(run(&mut func));
        assert!(!has_aggregate_alloca(&func));
        let mut mem2reg = super::super::Mem2Reg;
        assert!(mem2reg.run(&mut func));
        assert!(!has_memory_ops(&func));
        assert!(verify_func(&func).is_empty());
    }

    #[test]
    fn follows_integer_address_arithmetic() {
        let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("arithmetic", sig);
        let value = param(&mut func, entry, Ty::int(32), 0);
        let (_, root) = alloca(&mut func, entry, point_ty());
        let base = emit(&mut func, entry, InstKind::PtrToInt(root), Ty::int(64));
        let four = iconst(&mut func, 64, 4);
        let at_y = emit(&mut func, entry, InstKind::Add(base, four), Ty::int(64));
        let addr = emit(&mut func, entry, InstKind::IntToPtr(at_y), Ty::ptr(Ty::int(32)));
        store(&mut func, entry, addr, value);
        let loaded = load(&mut func, entry, addr, Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        assert!(run(&mut func));
        assert!(!has_aggregate_alloca(&func));
        let mut mem2reg = super::super::Mem2Reg;
        assert!(mem2reg.run(&mut func));
        assert!(!has_memory_ops(&func));
        assert!(verify_func(&func).is_empty());
    }

    #[test]
    fn follows_bitcast_addresses() {
        let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("bitcast", sig);
        let value = param(&mut func, entry, Ty::int(32), 0);
        let (_, root) = alloca(&mut func, entry, point_ty());
        let field = gep(&mut func, entry, root, vec![GepIndex::Field(0)], Ty::int(32));
        let addr = emit(&mut func, entry, InstKind::Bitcast(field), Ty::ptr(Ty::Void));
        store(&mut func, entry, addr, value);
        let loaded = load(&mut func, entry, addr, Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        assert!(run(&mut func));
        assert!(!has_aggregate_alloca(&func));
        let mut mem2reg = super::super::Mem2Reg;
        assert!(mem2reg.run(&mut func));
        assert!(!has_memory_ops(&func));
        assert!(verify_func(&func).is_empty());
    }

    #[test]
    fn leaves_dynamic_index_in_memory() {
        let array = Ty::Array {
            elem: Box::new(Ty::int(32)),
            len: 4,
        };
        let sig = Signature::new(vec![Ty::int(32), Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("dynamic", sig);
        let index = param(&mut func, entry, Ty::int(32), 0);
        let value = param(&mut func, entry, Ty::int(32), 1);
        let (_, root) = alloca(&mut func, entry, array);
        let addr = gep(&mut func, entry, root, vec![GepIndex::Index(index)], Ty::int(32));
        store(&mut func, entry, addr, value);
        let loaded = load(&mut func, entry, addr, Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        assert!(!run(&mut func));
        assert!(has_alloca(&func));
    }

    #[test]
    fn leaves_escaped_address_in_memory() {
        let sig = Signature::new(vec![], Ty::Void);
        let (mut func, entry) = Function::new("escape", sig);
        let (_, root) = alloca(&mut func, entry, point_ty());
        let addr = gep(&mut func, entry, root, vec![GepIndex::Field(0)], Ty::int(32));
        emit_unit(
            &mut func,
            entry,
            InstKind::CallExtern {
                target: crate::ir::Extern::from_usize(0),
                args: [addr].into_iter().collect(),
            },
        );
        func.blocks[entry].term = Some(Terminator::Return { val: None });

        assert!(!run(&mut func));
        assert!(has_alloca(&func));
    }

    #[test]
    fn leaves_overlapping_leaves_in_memory() {
        let sig = Signature::new(vec![], Ty::int(8));
        let (mut func, entry) = Function::new("overlap", sig);
        let (_, root) = alloca(&mut func, entry, point_ty());
        let wide = gep(&mut func, entry, root, vec![GepIndex::Field(0)], Ty::int(32));
        let byte = gep(&mut func, entry, root, vec![GepIndex::Field(0)], Ty::int(8));
        let one = iconst(&mut func, 32, 1);
        store(&mut func, entry, wide, one);
        let loaded = load(&mut func, entry, byte, Ty::int(8));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        assert!(!run(&mut func));
        assert!(has_alloca(&func));
    }

    #[test]
    fn leaves_aggregate_load_in_memory() {
        let point = point_ty();
        let sig = Signature::new(vec![], point.clone());
        let (mut func, entry) = Function::new("whole", sig);
        let (_, root) = alloca(&mut func, entry, point.clone());
        let loaded = load(&mut func, entry, root, point);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        assert!(!run(&mut func));
        assert!(has_alloca(&func));
    }

    #[test]
    fn leaves_volatile_access_in_memory() {
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("volatile", sig);
        let value = iconst(&mut func, 32, 7);
        let (_, root) = alloca(&mut func, entry, point_ty());
        let addr = gep(&mut func, entry, root, vec![GepIndex::Field(0)], Ty::int(32));
        emit_unit(
            &mut func,
            entry,
            InstKind::Store {
                addr,
                val: value,
                flags: MemFlags::vol(),
            },
        );
        let loaded = load(&mut func, entry, addr, Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        assert!(!run(&mut func));
        assert!(has_alloca(&func));
    }

    #[test]
    fn leaves_setjmp_functions_in_memory() {
        let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("jmp", sig);
        let value = param(&mut func, entry, Ty::int(32), 0);
        let (_, root) = alloca(&mut func, entry, point_ty());
        let addr = gep(&mut func, entry, root, vec![GepIndex::Field(0)], Ty::int(32));
        store(&mut func, entry, addr, value);
        let loaded = load(&mut func, entry, addr, Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });
        func.returns_twice = true;

        assert!(!run(&mut func));
        assert!(has_alloca(&func));
    }

    #[test]
    fn leaves_stored_address_in_memory() {
        let holder = Ty::Struct {
            fields: vec![Ty::int(32), Ty::ptr(Ty::int(32))],
            offsets: vec![0, 8],
            size: 16,
        };
        let sig = Signature::new(vec![], Ty::Void);
        let (mut func, entry) = Function::new("self", sig);
        let (_, root) = alloca(&mut func, entry, holder);
        let x = gep(&mut func, entry, root, vec![GepIndex::Field(0)], Ty::int(32));
        let y = gep(&mut func, entry, root, vec![GepIndex::Field(1)], Ty::ptr(Ty::int(32)));
        emit_unit(
            &mut func,
            entry,
            InstKind::Store {
                addr: y,
                val: x,
                flags: MemFlags::plain(),
            },
        );
        func.blocks[entry].term = Some(Terminator::Return { val: None });

        assert!(!run(&mut func));
        assert!(has_alloca(&func));
    }

    #[test]
    fn removes_unused_aggregate_alloca() {
        let sig = Signature::new(vec![], Ty::Void);
        let (mut func, entry) = Function::new("unused", sig);
        alloca(&mut func, entry, point_ty());
        func.blocks[entry].term = Some(Terminator::Return { val: None });

        assert!(run(&mut func));
        assert!(!has_alloca(&func));
        assert!(verify_func(&func).is_empty());
        assert!(!run(&mut func));
    }

    #[test]
    fn scalarizes_nested_fields() {
        let inner = Ty::Struct {
            fields: vec![Ty::int(32), Ty::int(8)],
            offsets: vec![0, 4],
            size: 8,
        };
        let outer = Ty::Struct {
            fields: vec![Ty::int(8), inner],
            offsets: vec![0, 8],
            size: 16,
        };
        let sig = Signature::new(vec![Ty::int(8)], Ty::int(8));
        let (mut func, entry) = Function::new("nested", sig);
        let value = param(&mut func, entry, Ty::int(8), 0);
        let (_, root) = alloca(&mut func, entry, outer);
        let addr = gep(&mut func, entry, root, vec![GepIndex::Field(1), GepIndex::Field(1)], Ty::int(8));
        store(&mut func, entry, addr, value);
        let loaded = load(&mut func, entry, addr, Ty::int(8));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(loaded) });

        assert!(run(&mut func));
        assert!(!has_aggregate_alloca(&func));
        let mut mem2reg = super::super::Mem2Reg;
        assert!(mem2reg.run(&mut func));
        assert!(!has_memory_ops(&func));
        assert!(verify_func(&func).is_empty());
    }

    #[test]
    fn keeps_icmp_on_address_in_memory() {
        let sig = Signature::new(vec![], Ty::Bool);
        let (mut func, entry) = Function::new("cmp", sig);
        let (_, root) = alloca(&mut func, entry, point_ty());
        let addr = gep(&mut func, entry, root, vec![GepIndex::Field(0)], Ty::int(32));
        let zero_int = iconst(&mut func, 64, 0);
        let zero = emit(&mut func, entry, InstKind::IntToPtr(zero_int), Ty::ptr(Ty::int(32)));
        let cmp = emit(
            &mut func,
            entry,
            InstKind::Icmp {
                pred: IcmpPred::Eq,
                lhs: addr,
                rhs: zero,
            },
            Ty::Bool,
        );
        func.blocks[entry].term = Some(Terminator::Return { val: Some(cmp) });

        assert!(!run(&mut func));
        assert!(has_alloca(&func));
    }
}
