//! Module-level constant propagation and dead definition removal.
//!
//! [`GlobalConstProp`] replaces loads from read-only local storage with the
//! initializer's value. A local object qualifies only when no instruction
//! writes to it and its address never escapes; inline assembly disables the
//! rewrite because its symbol references are invisible in the IR.
//!
//! [`MergeReadOnlyGlobals`] folds identical read-only internal objects (string
//! literals) into one definition.
//!
//! [`DeadModuleItems`] removes internal functions and objects that no live
//! definition references. External-linkage definitions are roots. Function
//! and object liveness is solved together because a live object's
//! initializer can reference a function, and a live function can address an
//! object.

use std::collections::{HashMap, HashSet};

use index_vec::IndexVec;

use crate::debug::{FunctionDebug, ScopeDebug, VarHome};
use crate::ir::{Const, DataReloc, Def, Func, Function, Global, GlobalData, Inst, InstKind, MemFlags, Module, Value, ValueData};
use crate::pass::ModulePass;
use crate::ty::{GepIndex, Ty};

use super::{MemoryObject, address_info};

/// Replace loads from read-only local storage with their initializer.
pub struct GlobalConstProp;

impl ModulePass for GlobalConstProp {
    fn name(&self) -> &'static str {
        "global-const-prop"
    }

    fn run(&mut self, module: &mut Module) -> bool {
        if has_inline_asm(module) {
            return false;
        }

        let nglobals = module.globals.len();
        let mut written = vec![false; nglobals];
        let mut escaped = vec![false; nglobals];
        for global in module.globals.iter() {
            for reloc in global.relocs.iter().map(|(_, r)| r).chain(global.relocs32.iter().map(|(_, r)| r)) {
                if let DataReloc::Global { target, .. } = reloc {
                    escaped[target.index()] = true;
                }
            }
        }

        let mut sites: Vec<(usize, Inst, Global, u64)> = Vec::new();
        for (func_id, func) in module.funcs.iter().enumerate() {
            let resident = func.resident_mask();
            let addresses = global_addresses(func, &resident);
            for (inst_id, inst) in func.insts.iter_enumerated() {
                if !resident[inst_id.index()] {
                    continue;
                }
                let resolved = inst.result.and_then(|result| global_root(func, result, &addresses));
                match &inst.kind {
                    InstKind::Load { addr, flags } => {
                        if !flags.volatile
                            && let Some((global, offset)) = global_at(func, *addr, &addresses) {
                                sites.push((func_id, inst_id, global, offset));
                            }
                    }
                    InstKind::Store { addr, val, .. } => {
                        if let Some(global) = global_root(func, *addr, &addresses) {
                            written[global.index()] = true;
                        }
                        if let Some(global) = global_root(func, *val, &addresses) {
                            escaped[global.index()] = true;
                        }
                    }
                    InstKind::Memcpy { dst, .. } => {
                        if let Some(global) = global_root(func, *dst, &addresses) {
                            written[global.index()] = true;
                        }
                    }
                    InstKind::Memset { dst, .. } => {
                        if let Some(global) = global_root(func, *dst, &addresses) {
                            written[global.index()] = true;
                        }
                    }
                    InstKind::GetElementPtr { .. } | InstKind::Bitcast(_) | InstKind::PtrToInt(_) | InstKind::IntToPtr(_) | InstKind::Add(_, _) | InstKind::Sub(_, _)
                        if resolved.is_some() => {}
                    kind => kind.for_each_operand(&mut |value| {
                        if let Some(global) = global_root(func, value, &addresses) {
                            escaped[global.index()] = true;
                        }
                    }),
                }
            }
            for block in func.blocks.iter() {
                if let Some(term) = &block.term {
                    term.for_each_operand(&mut |value| {
                        if let Some(global) = global_root(func, value, &addresses) {
                            escaped[global.index()] = true;
                        }
                    });
                }
            }
        }

        let candidates: Vec<bool> = module
            .globals
            .iter()
            .enumerate()
            .map(|(id, global)| {
                global.local
                    && !global.external
                    && !global.weak
                    && !global.tls
                    && !written[id]
                    && !escaped[id]
                    && matches!(&global.init, Some(Const::Int { .. } | Const::Float { .. } | Const::Null | Const::Zero | Const::Bytes(_)))
            })
            .collect();

        let mut replacements = Vec::new();
        for (func_id, inst_id, global, offset) in sites {
            let func = &module.funcs[func_id];
            if !candidates[global.index()] {
                continue;
            }
            let Some(result) = func.insts[inst_id].result else {
                continue;
            };
            let global = &module.globals[global.index()];
            if let Some(init) = constant_at(global, &func.values[result].ty, offset) {
                replacements.push((func_id, inst_id, init));
            }
        }

        let mut changed = false;
        for (func_id, inst_id, init) in replacements {
            let func = &mut module.funcs[func_id];
            let Some(result) = func.insts[inst_id].result else {
                continue;
            };
            let block = func.insts[inst_id].block;
            let ty = func.values[result].ty.clone();
            let fresh = func.values.push(ValueData { ty, def: Def::Const(init) });
            func.replace_uses(result, fresh);
            func.blocks[block].insts.retain(|i| *i != inst_id);
            changed = true;
        }
        changed |= expand_global_memcpys(module, &candidates);
        changed
    }
}

/// Copying a whole local aggregate from a read-only global leaves a
/// `memcpy` that hides the individual scalar initializers. Split it into
/// field stores so `Sroa` can promote the destination and the loads fold
/// away. The source global must not escape; a global only read by a copy
/// still qualifies.
fn expand_global_memcpys(module: &mut Module, candidates: &[bool]) -> bool {
    let globals = &module.globals;
    let mut changed = false;
    for func in module.funcs.iter_mut() {
        let resident = func.resident_mask();
        let addresses = global_addresses(func, &resident);
        let non_escaping = super::non_escaping_stack_slots(func);
        for block_idx in 0..func.blocks.len() {
            let block = crate::ir::Block::from_usize(block_idx);
            let insts = func.blocks[block].insts.clone();
            let mut rebuilt = Vec::with_capacity(insts.len());
            let mut expanded = false;
            for inst in insts {
                let copy = match &func.insts[inst].kind {
                    InstKind::Memcpy { dst, src, len } => Some((*dst, *src, *len)),
                    _ => None,
                };
                let Some((dst, src, len)) = copy else {
                    rebuilt.push(inst);
                    continue;
                };
                let Some(leaves) = global_copy_leaves(func, dst, src, len, &addresses, &non_escaping, globals, candidates) else {
                    rebuilt.push(inst);
                    continue;
                };
                let Ty::Ptr { bits, .. } = func.values[dst].ty else {
                    unreachable!("local copy destinations are pointers");
                };
                for (path, ty, init) in leaves {
                    let addr_ty = Ty::ptr_bits(ty.clone(), bits);
                    let (addr, addr_inst) = super::make_value(
                        func,
                        block,
                        InstKind::GetElementPtr {
                            base: dst,
                            indices: path.into_iter().collect(),
                        },
                        addr_ty,
                    );
                    rebuilt.push(addr_inst);
                    let value = func.values.push(ValueData { ty, def: Def::Const(init) });
                    rebuilt.push(super::make_unit(
                        func,
                        block,
                        InstKind::Store {
                            addr,
                            val: value,
                            flags: MemFlags::plain(),
                        },
                    ));
                }
                expanded = true;
                changed = true;
            }
            if expanded {
                func.blocks[block].insts = rebuilt;
            }
        }
    }
    changed
}

fn global_copy_leaves(
    func: &Function,
    dst: Value,
    src: Value,
    len: u64,
    addresses: &HashMap<Value, Global>,
    non_escaping: &HashSet<Value>,
    globals: &IndexVec<Global, GlobalData>,
    candidates: &[bool],
) -> Option<Vec<(Vec<GepIndex>, Ty, Const)>> {
    let Ty::Ptr { pointee, .. } = &func.values[dst].ty else {
        return None;
    };
    if !pointee.is_aggregate() || pointee.size_bytes() != len {
        return None;
    }
    let Some(super::AddressInfo {
        object: MemoryObject::Stack(root),
        ..
    }) = address_info(func, dst)
    else {
        return None;
    };
    if !non_escaping.contains(&root) {
        return None;
    }
    let (global, source_offset) = global_at(func, src, addresses)?;
    if !candidates[global.index()] {
        return None;
    }
    let global = &globals[global.index()];
    let init = global.init.as_ref()?;
    let mut leaves = Vec::new();
    for (path, ty) in super::aggregate_leaves(pointee)? {
        let offset = source_offset.checked_add(leaf_offset(pointee, &path)?)?;
        let value = match init {
            Const::Zero => super::zero_const_for(&ty),
            _ => constant_at(global, &ty, offset)?,
        };
        leaves.push((path, ty, value));
    }
    Some(leaves)
}

fn leaf_offset(root: &Ty, path: &[GepIndex]) -> Option<u64> {
    let mut ty = root;
    let mut offset = 0u64;
    for step in path {
        match (ty, step) {
            (Ty::Struct { fields, offsets, .. }, GepIndex::Field(index)) => {
                let index = *index as usize;
                offset = offset.checked_add(*offsets.get(index)?)?;
                ty = fields.get(index)?;
            }
            (Ty::Array { elem, .. }, GepIndex::ConstIndex(index)) => {
                offset = offset.checked_add(elem.size_bytes().checked_mul(*index)?)?;
                ty = elem;
            }
            _ => return None,
        }
    }
    Some(offset)
}

/// Fold identical read-only internal objects into one definition.
///
/// Only objects that cannot be written and whose debug info never names them
/// qualify, so merging cannot change observable storage: string literals are
/// the intended case. The survivor takes the strictest alignment.
pub struct MergeReadOnlyGlobals;

impl ModulePass for MergeReadOnlyGlobals {
    fn name(&self) -> &'static str {
        "merge-read-only-globals"
    }

    fn run(&mut self, module: &mut Module) -> bool {
        let nglobals = module.globals.len();
        let mut protected = HashSet::new();
        for func in module.funcs.iter() {
            if let Some(debug) = &func.debug {
                let mut named = Vec::new();
                collect_debug_globals(debug, &mut named);
                protected.extend(named);
            }
        }

        let mut canonical: Vec<Option<Global>> = vec![None; nglobals];
        {
            let mut first: HashMap<(&Ty, &Const), Global> = HashMap::new();
            for (id, global) in module.globals.iter_enumerated() {
                if !mergeable(global) || protected.contains(&id) {
                    continue;
                }
                let Some(init) = &global.init else { continue };
                match first.entry((&global.ty, init)) {
                    std::collections::hash_map::Entry::Occupied(entry) => canonical[id.index()] = Some(*entry.get()),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(id);
                    }
                }
            }
        }
        if canonical.iter().all(Option::is_none) {
            return false;
        }

        for (old, canonical) in canonical.iter().enumerate() {
            let Some(canonical) = canonical else { continue };
            let merged = match (module.globals[*canonical].align, module.globals[old].align) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (align, None) | (None, align) => align,
            };
            module.globals[*canonical].align = merged;
        }

        let mut survivor: Vec<Option<Global>> = vec![None; nglobals];
        let mut kept = 0;
        for (old, slot) in survivor.iter_mut().enumerate() {
            if canonical[old].is_none() {
                *slot = Some(Global::from_usize(kept));
                kept += 1;
            }
        }
        let targets: Vec<Global> = (0..nglobals)
            .map(|old| survivor[canonical[old].unwrap_or(Global::from_usize(old)).index()].expect("canonical object survives"))
            .collect();

        for func in module.funcs.iter_mut() {
            let resident = func.resident_mask();
            for (id, inst) in func.insts.iter_mut().enumerate() {
                if !resident[id] {
                    continue;
                }
                if let InstKind::GlobalAddr { global } = &mut inst.kind {
                    *global = targets[global.index()];
                }
            }
            if let Some(debug) = &mut func.debug {
                let homes: Vec<Option<Global>> = targets.iter().map(|target| Some(*target)).collect();
                remap_debug(debug, &homes);
            }
        }

        let old_globals = std::mem::take(&mut module.globals);
        let mut new_globals = Vec::with_capacity(kept);
        for (old, mut global) in old_globals.into_iter().enumerate() {
            if canonical[old].is_some() {
                continue;
            }
            for reloc in global.relocs.iter_mut().map(|(_, reloc)| reloc).chain(global.relocs32.iter_mut().map(|(_, reloc)| reloc)) {
                if let DataReloc::Global { target, .. } = reloc {
                    *target = targets[target.index()];
                }
            }
            new_globals.push(global);
        }
        module.globals = IndexVec::from_vec(new_globals);
        true
    }
}

fn mergeable(global: &GlobalData) -> bool {
    global.local
        && !global.external
        && !global.weak
        && !global.tls
        && !global.writable
        && global.debug.is_none()
        && global.section.is_none()
        && global.init.is_some()
        && global.relocs.is_empty()
        && global.relocs32.is_empty()
}

/// Remove unreferenced internal functions and objects.
pub struct DeadModuleItems;

impl ModulePass for DeadModuleItems {
    fn name(&self) -> &'static str {
        "dead-module-items"
    }

    fn run(&mut self, module: &mut Module) -> bool {
        if has_inline_asm(module) {
            return false;
        }

        let nfuncs = module.funcs.len();
        let nglobals = module.globals.len();
        let mut func_calls = vec![Vec::new(); nfuncs];
        let mut func_globals = vec![Vec::new(); nfuncs];
        for (id, func) in module.funcs.iter().enumerate() {
            let resident = func.resident_mask();
            for (inst_id, inst) in func.insts.iter_enumerated() {
                if !resident[inst_id.index()] {
                    continue;
                }
                match &inst.kind {
                    InstKind::Call { callee, .. } => func_calls[id].push(*callee),
                    InstKind::FuncAddr { func } => func_calls[id].push(*func),
                    InstKind::GlobalAddr { global } => func_globals[id].push(*global),
                    _ => {}
                }
            }
            if let Some(debug) = &func.debug {
                collect_debug_globals(debug, &mut func_globals[id]);
            }
        }

        let mut global_funcs = vec![Vec::new(); nglobals];
        let mut global_globals = vec![Vec::new(); nglobals];
        for (id, global) in module.globals.iter().enumerate() {
            for reloc in global.relocs.iter().map(|(_, r)| r).chain(global.relocs32.iter().map(|(_, r)| r)) {
                match reloc {
                    DataReloc::Func { target } => global_funcs[id].push(*target),
                    DataReloc::Global { target, .. } => global_globals[id].push(*target),
                    DataReloc::FuncExtern { .. } => {}
                }
            }
        }

        let mut live_funcs: Vec<bool> = module.funcs.iter().map(|func| !func.local || func.weak).collect();
        let mut live_globals: Vec<bool> = module.globals.iter().map(|global| !global.local || global.external || global.weak).collect();
        // Constructors run from `.init_array` without any MIR-level
        // reference: they are roots.
        for func in &module.init_funcs {
            live_funcs[func.index()] = true;
        }

        loop {
            let mut changed = false;
            for id in 0..nfuncs {
                if !live_funcs[id] {
                    continue;
                }
                for target in &func_calls[id] {
                    if !live_funcs[target.index()] {
                        live_funcs[target.index()] = true;
                        changed = true;
                    }
                }
                for target in &func_globals[id] {
                    if !live_globals[target.index()] {
                        live_globals[target.index()] = true;
                        changed = true;
                    }
                }
            }
            for id in 0..nglobals {
                if !live_globals[id] {
                    continue;
                }
                for target in &global_funcs[id] {
                    if !live_funcs[target.index()] {
                        live_funcs[target.index()] = true;
                        changed = true;
                    }
                }
                for target in &global_globals[id] {
                    if !live_globals[target.index()] {
                        live_globals[target.index()] = true;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        let (func_map, keep_funcs) = build_map(&live_funcs);
        let (global_map, keep_globals) = build_map(&live_globals);
        if keep_funcs == nfuncs && keep_globals == nglobals {
            return false;
        }

        let old_funcs = std::mem::take(&mut module.funcs);
        let mut new_funcs = Vec::with_capacity(keep_funcs);
        for (old, mut func) in old_funcs.into_iter().enumerate() {
            if !live_funcs[old] {
                continue;
            }
            remap_function(&mut func, &func_map, &global_map);
            new_funcs.push(func);
        }
        module.funcs = IndexVec::from_vec(new_funcs);

        for func in module.init_funcs.iter_mut() {
            *func = func_map[func.index()].expect("init function survives (it is a root)");
        }

        let old_globals = std::mem::take(&mut module.globals);
        let mut new_globals = Vec::with_capacity(keep_globals);
        for (old, mut global) in old_globals.into_iter().enumerate() {
            if !live_globals[old] {
                continue;
            }
            remap_global(&mut global, &func_map, &global_map);
            new_globals.push(global);
        }
        module.globals = IndexVec::from_vec(new_globals);
        true
    }
}

fn has_inline_asm(module: &Module) -> bool {
    module.funcs.iter().any(|func| {
        let resident = func.resident_mask();
        func.insts
            .iter_enumerated()
            .any(|(id, inst)| resident[id.index()] && matches!(inst.kind, InstKind::InlineAsm { .. }))
    })
}

fn global_addresses(func: &Function, resident: &[bool]) -> HashMap<Value, Global> {
    let mut addresses = HashMap::new();
    for (id, inst) in func.insts.iter_enumerated() {
        if !resident[id.index()] {
            continue;
        }
        if let InstKind::GlobalAddr { global } = inst.kind
            && let Some(result) = inst.result {
                addresses.insert(result, global);
            }
    }
    addresses
}

fn global_root(func: &Function, value: Value, addresses: &HashMap<Value, Global>) -> Option<Global> {
    match address_info(func, value)?.object {
        MemoryObject::Global(root) => addresses.get(&root).copied(),
        MemoryObject::Stack(_) | MemoryObject::Heap(_) => None,
    }
}

fn global_at(func: &Function, value: Value, addresses: &HashMap<Value, Global>) -> Option<(Global, u64)> {
    let info = address_info(func, value)?;
    let MemoryObject::Global(root) = info.object else {
        return None;
    };
    Some((*addresses.get(&root)?, info.offset?))
}

fn constant_at(global: &GlobalData, ty: &Ty, offset: u64) -> Option<Const> {
    let init = global.init.as_ref()?;
    if offset == 0 && init_matches(init, ty) {
        return Some(init.clone());
    }
    let Const::Bytes(bytes) = init else {
        return None;
    };
    // Bounds and width first, then any relocation touching the load.
    let size = ty.size_bytes();
    let len = usize::try_from(size).ok()?;
    let start = usize::try_from(offset).ok()?;
    let end = start.checked_add(len)?;
    let slice = bytes.get(start..end)?;
    if reloc_overlaps(global, offset, size) {
        return None;
    }
    let mut raw = [0u8; 16];
    if len > raw.len() {
        return None;
    }
    raw[..len].copy_from_slice(slice);
    // Global images are little-endian (lower_mir stores integers and
    // float bits with `to_le_bytes`), matching the supported targets.
    match ty {
        Ty::Int { bits } if *bits >= 8 && *bits <= 128 && u64::from(*bits) / 8 == size => Some(
            Const::Int {
                bits: *bits,
                value: i128::from_le_bytes(raw),
            }
            .canonical(),
        ),
        Ty::Bool => match slice {
            [0] => Some(Const::Int { bits: 1, value: 0 }),
            [1] => Some(Const::Int { bits: 1, value: 1 }),
            _ => None,
        },
        Ty::Float(kind) => Some(Const::Float {
            kind: *kind,
            bits: u128::from_le_bytes(raw),
        }),
        Ty::Ptr { .. } if slice.iter().all(|byte| *byte == 0) => Some(Const::Null),
        _ => None,
    }
}

fn reloc_overlaps(global: &GlobalData, offset: u64, size: u64) -> bool {
    let end = offset.saturating_add(size);
    let overlaps = |at: u64, width: u64| at < end && offset < at.saturating_add(width);
    global.relocs.iter().any(|(at, _)| overlaps(*at, 8)) || global.relocs32.iter().any(|(at, _)| overlaps(*at, 4))
}

fn init_matches(init: &Const, ty: &Ty) -> bool {
    match init {
        Const::Int { bits, .. } => matches!(ty, Ty::Int { bits: load_bits } if load_bits == bits) || matches!(ty, Ty::Bool) && *bits == 1,
        Const::Float { kind, .. } => matches!(ty, Ty::Float(load_kind) if load_kind == kind),
        Const::Null => matches!(ty, Ty::Ptr { .. }),
        Const::Zero => ty.is_scalar(),
        _ => false,
    }
}

fn build_map<T: index_vec::Idx>(live: &[bool]) -> (Vec<Option<T>>, usize) {
    let mut map = Vec::with_capacity(live.len());
    let mut next = 0;
    for &is_live in live {
        if is_live {
            map.push(Some(T::from_usize(next)));
            next += 1;
        } else {
            map.push(None);
        }
    }
    (map, next)
}

fn remap_function(func: &mut Function, funcs: &[Option<Func>], globals: &[Option<Global>]) {
    // Orphaned instructions never execute and no longer feed liveness,
    // so a target they name may already be gone.
    let resident = func.resident_mask();
    for (id, inst) in func.insts.iter_mut().enumerate() {
        if !resident[id] {
            continue;
        }
        match &mut inst.kind {
            InstKind::Call { callee, .. } => {
                *callee = funcs[callee.index()].expect("live caller references a live function");
            }
            InstKind::FuncAddr { func: target } => {
                *target = funcs[target.index()].expect("live function addresses a live function");
            }
            InstKind::GlobalAddr { global } => {
                *global = globals[global.index()].expect("live function addresses a live object");
            }
            _ => {}
        }
    }
    if let Some(debug) = &mut func.debug {
        remap_debug(debug, globals);
    }
}

fn remap_global(global: &mut crate::ir::GlobalData, funcs: &[Option<Func>], globals: &[Option<Global>]) {
    for reloc in global.relocs.iter_mut().map(|(_, r)| r).chain(global.relocs32.iter_mut().map(|(_, r)| r)) {
        match reloc {
            DataReloc::Func { target } => {
                *target = funcs[target.index()].expect("live object references a live function");
            }
            DataReloc::Global { target, .. } => {
                *target = globals[target.index()].expect("live object references a live object");
            }
            DataReloc::FuncExtern { .. } => {}
        }
    }
}

fn collect_debug_globals(debug: &FunctionDebug, out: &mut Vec<Global>) {
    for var in debug.params.iter().chain(debug.vars.iter()) {
        if let VarHome::Global(global) = var.home {
            out.push(global);
        }
    }
    for scope in &debug.scopes {
        collect_scope_globals(scope, out);
    }
}

fn collect_scope_globals(scope: &ScopeDebug, out: &mut Vec<Global>) {
    for var in &scope.vars {
        if let VarHome::Global(global) = var.home {
            out.push(global);
        }
    }
    for child in &scope.children {
        collect_scope_globals(child, out);
    }
}

fn remap_debug(debug: &mut FunctionDebug, globals: &[Option<Global>]) {
    for var in debug.params.iter_mut().chain(debug.vars.iter_mut()) {
        remap_home(&mut var.home, globals);
    }
    for scope in &mut debug.scopes {
        remap_scope(scope, globals);
    }
}

fn remap_scope(scope: &mut ScopeDebug, globals: &[Option<Global>]) {
    for var in &mut scope.vars {
        remap_home(&mut var.home, globals);
    }
    for child in &mut scope.children {
        remap_scope(child, globals);
    }
}

fn remap_home(home: &mut VarHome, globals: &[Option<Global>]) {
    if let VarHome::Global(global) = home {
        *global = globals[global.index()].expect("live function debug references a live object");
    }
}

#[cfg(test)]
mod tests;

