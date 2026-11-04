use std::collections::HashMap;

use cranelift_codegen::ir::types::{self};
use cranelift_module::{FuncId, Linkage, Module};

use glcc_mir as mir;

use crate::abi::{self, Target};

#[cfg(test)]
mod tests;

mod body;
mod call;
mod data;
mod inst;
mod mem;
mod rt;

pub(crate) use body::*;
pub(crate) use call::*;
pub(crate) use data::*;
pub(crate) use mem::*;
pub(crate) use rt::*;

/// Windows x64 unwind record for one defined function: the serialized
/// `UNWIND_INFO` (`.xdata` content) plus the function size its
/// `.pdata` `RUNTIME_FUNCTION` entry needs. The driver's `winunwind`
/// module lays these out as sections; other targets collect none.
use crate::ehframe::FuncEh;

pub struct WinUnwindEntry {
    pub func: FuncId,
    /// Function size in bytes: begin address + size is the RUNTIME_FUNCTION
    /// end address.
    pub size: u32,
    /// Serialized UNWIND_INFO, 4-byte aligned sections entries.
    pub xdata: Vec<u8>,
}

/// A 32-bit (`__ptr32`) address patch in a global's image. The
/// Cranelift data machinery only emits pointer-width relocations, so
/// these are reported to the object driver, which patches the emitted
/// object with a 4-byte absolute relocation at `offset` within the
/// object `container`. `addend` is a byte displacement.
pub struct NarrowReloc {
    pub container: cranelift_module::DataId,
    pub offset: u64,
    pub target: NarrowTarget,
    pub addend: i64,
}

/// The symbol a [`NarrowReloc`] resolves to.
pub enum NarrowTarget {
    Data(cranelift_module::DataId),
    Func(FuncId),
}

/// Lower a whole MIR module into a Cranelift module for `target`,
/// defining every function and global. Returns the Clif id of each MIR
/// function.
///
/// Linkage follows the IR flags: `local` functions and globals emit
/// `Local` (internal linkage, anonymous objects), weak definitions
/// emit `Preemptible` (ELF `STB_WEAK`), the rest `Export`. Tests need
/// stable symbol names either way.
pub fn lower_module<M: Module>(mir_module: &mir::Module, clif: &mut M, target: Target) -> HashMap<mir::Func, FuncId> {
    let mut asm_text = String::new();
    let mut eh = Vec::new();
    lower_module_impl(mir_module, clif, target, &mut asm_text, None, None, &mut eh, &|_| true, &|_| true).0
}

/// [`lower_module`] for a module that shares a [`Module`] with code
/// compiled by an earlier call.
///
/// Every symbol is declared, so calls into earlier code resolve, but a
/// function or global is only *defined* when its predicate returns
/// true. Callers pass false for symbols that are already finalized in
/// `clif`; the declaration then reuses the existing id.
pub fn lower_module_selected<M: Module>(
    mir_module: &mir::Module,
    clif: &mut M,
    target: Target,
    define_func: &dyn Fn(&str) -> bool,
    define_data: &dyn Fn(&str) -> bool,
) -> HashMap<mir::Func, FuncId> {
    let mut asm_text = String::new();
    let mut eh = Vec::new();
    lower_module_impl(mir_module, clif, target, &mut asm_text, None, None, &mut eh, define_func, define_data).0
}

/// [`lower_module`], also appending the generated inline-asm wrapper
/// text to `asm_text`. The driver assembles it into the TU's `.s`.
/// Returns the 32-bit (`__ptr32`) address patches the object driver
/// must apply itself.
pub fn lower_module_collecting<M: Module>(mir_module: &mir::Module, clif: &mut M, target: Target, asm_text: &mut String, unwind: &mut Vec<WinUnwindEntry>, eh: &mut Vec<FuncEh>) -> Vec<NarrowReloc> {
    lower_module_collecting_with_ids(mir_module, clif, target, asm_text, unwind, eh).1
}

/// [`lower_module_collecting`], also returning the Clif id of every MIR
/// function: the object driver needs the ids of the module's
/// constructors ([`mir::Module::init_funcs`]) to emit `.init_array`
/// (see [`crate::initarray::emit_init_array`]).
pub fn lower_module_collecting_with_ids<M: Module>(
    mir_module: &mir::Module,
    clif: &mut M,
    target: Target,
    asm_text: &mut String,
    unwind: &mut Vec<WinUnwindEntry>,
    eh: &mut Vec<FuncEh>,
) -> (HashMap<mir::Func, FuncId>, Vec<NarrowReloc>) {
    lower_module_impl(mir_module, clif, target, asm_text, None, Some(unwind), eh, &|_| true, &|_| true)
}

/// [`lower_module_collecting`], also collecting the debug records the
/// DWARF writer needs: machine sizes and source maps of every defined
/// function, frame-relative variable homes, and module-level objects.
pub fn lower_module_with_debug<M: Module>(
    mir_module: &mir::Module,
    clif: &mut M,
    target: Target,
    asm_text: &mut String,
    unwind: &mut Vec<WinUnwindEntry>,
    eh: &mut Vec<FuncEh>,
) -> (HashMap<mir::Func, FuncId>, glcc_dwarf::LoweredDebug, Vec<NarrowReloc>) {
    let mut debug = glcc_dwarf::LoweredDebug::new();
    let (ids, narrow) = lower_module_impl(mir_module, clif, target, asm_text, Some(&mut debug), Some(unwind), eh, &|_| true, &|_| true);
    (ids, debug, narrow)
}

pub(crate) fn lower_module_impl<M: Module>(
    mir_module: &mir::Module,
    clif: &mut M,
    target: Target,
    asm_text: &mut String,
    mut debug: Option<&mut glcc_dwarf::LoweredDebug>,
    mut unwind: Option<&mut Vec<WinUnwindEntry>>,
    eh: &mut Vec<FuncEh>,
    define_func: &dyn Fn(&str) -> bool,
    define_data: &dyn Fn(&str) -> bool,
) -> (HashMap<mir::Func, FuncId>, Vec<NarrowReloc>) {
    let mut wrappers = Wrappers::default();
    let mut layouts = HashMap::new();
    for (id, func) in mir_module.funcs.iter_enumerated() {
        layouts.insert(id, abi::layout_sig(target, &func.sig.params, &func.sig.ret, func.sig.variadic));
    }
    let mut ids = HashMap::new();
    for (id, func) in mir_module.funcs.iter_enumerated() {
        let sig = abi::clif_signature(&layouts[&id], target.call_conv());
        let linkage = definition_linkage(func.local, func.weak);
        let clif_id = clif.declare_function(&func.name, linkage, &sig).unwrap_or_else(|e| panic!("cannot declare {}: {e}", func.name));
        ids.insert(id, clif_id);
    }
    // Imports for declared externs use plain named-only signatures;
    // each call site builds its own signature from actuals. The
    // call-site layout keeps the declared variadic-ness on every
    // target: it selects the indirect per-call signature, since the
    // actuals may outnumber the named params. On x86-64 it also routes
    // the call through the `%al` trampoline (see `emit_call`).
    //
    // Only imports the module actually references are declared: an
    // unused declaration would leave an undefined object-file symbol
    // behind, and MSVC's linker rejects undefined COFF symbols outright.
    let mut used_externs = std::collections::HashSet::new();
    let mut used_globals = std::collections::HashSet::new();
    for func in mir_module.funcs.iter() {
        for inst in &func.insts {
            match &inst.kind {
                mir::InstKind::CallExtern { target, .. } => {
                    used_externs.insert(*target);
                }
                mir::InstKind::ExternAddr { target } => {
                    used_externs.insert(*target);
                }
                mir::InstKind::GlobalAddr { global } => {
                    used_globals.insert(*global);
                }
                mir::InstKind::EhTypeId { typeinfo } => {
                    used_globals.insert(*typeinfo);
                }
                _ => {}
            }
        }
        // Invokes reference externs/functions; landing-pad catch clauses
        // name the typeinfo globals the LSDA's type table will point at.
        for block in func.blocks.iter() {
            if let Some(clauses) = &block.lpad {
                for clause in clauses {
                    if let mir::LpadClause::Catch(global) = clause {
                        used_globals.insert(*global);
                    }
                }
            }
            if let Some(mir::Terminator::Invoke { callee, .. }) = &block.term
                && let mir::InvokeCallee::Extern(target) = callee {
                    used_externs.insert(*target);
                }
        }
        if let Some(personality) = func.personality {
            used_externs.insert(personality);
        }
    }
    for global in mir_module.globals.iter() {
        for (_, reloc) in global.relocs.iter().chain(&global.relocs32) {
            match reloc {
                mir::DataReloc::Global { target, .. } => {
                    used_globals.insert(*target);
                }
                mir::DataReloc::FuncExtern { target } => {
                    used_externs.insert(*target);
                }
                mir::DataReloc::Func { .. } => {}
                // Deferred relocations are resolved during parsing.
                mir::DataReloc::Deferred { .. } => {}
            }
        }
    }
    let mut extern_ids = HashMap::new();
    let mut extern_layouts = HashMap::new();
    for (id, ext) in mir_module.externs.iter_enumerated() {
        if !used_externs.contains(&id) {
            continue;
        }
        let layout = abi::layout_sig(target, &ext.sig.params, &ext.sig.ret, false);
        let sig = abi::clif_signature(&layout, target.call_conv());
        let clif_id = clif
            .declare_function(&ext.name, import_linkage(ext.weak), &sig)
            .unwrap_or_else(|e| panic!("cannot declare extern {}: {e}", ext.name));
        extern_ids.insert(id, clif_id);
        extern_layouts.insert(id, abi::layout_sig(target, &ext.sig.params, &ext.sig.ret, ext.sig.variadic));
    }
    let mut data_ids = HashMap::new();
    // Thread-local globals address through `tls_value`, not
    // `symbol_value`; collecting them once keeps `emit_inst` from
    // re-scanning the module for every `GlobalAddr`.
    let mut tls_data = std::collections::HashSet::new();
    for (id, global) in mir_module.globals.iter_enumerated() {
        // Declare all data first: relocations may point forward. Unused
        // extern declarations stay undeclared for the same reason as
        // unused function imports.
        if global.external && !used_globals.contains(&id) {
            continue;
        }
        data_ids.insert(id, declare_global(global, clif));
        if global.tls {
            tls_data.insert(id);
        }
    }
    let mut narrow = Vec::new();
    for (id, global) in mir_module.globals.iter_enumerated() {
        if let Some(&did) = data_ids.get(&id)
            && define_data(&global.name) {
                define_global(global, did, &data_ids, &ids, &extern_ids, clif, &mut narrow);
            }
    }
    if let Some(debug) = debug.as_deref_mut() {
        for (id, global) in mir_module.globals.iter_enumerated() {
            crate::debug::collect_global_debug(debug, global, data_ids.get(&id).copied());
        }
    }
    // x86-64 foreign variadic calls route through the runtime
    // trampoline; declare its symbols only when such a call exists.
    let rt = Rt::declare_if_needed(mir_module, target, clif);
    let memcpy_id = mir_module
        .funcs
        .iter()
        .any(needs_libc_memcpy)
        .then(|| {
            // A user prototype may describe a different signature. In that
            // case fall back to the inline copy rather than conflicting
            // with the user's declaration.
            clif.declare_function("memcpy", Linkage::Import, &rt_sig(target, vec![types::I64, types::I64, types::I64], vec![types::I64]))
                .ok()
        })
        .flatten();
    for (id, func) in mir_module.funcs.iter_enumerated() {
        if !define_func(&func.name) {
            continue;
        }
        lower_function(
            id,
            func,
            &mir_module.globals,
            target,
            &layouts,
            &extern_layouts,
            &ids,
            &extern_ids,
            &data_ids,
            &tls_data,
            &rt,
            memcpy_id,
            clif,
            &mut wrappers,
            debug.as_deref_mut(),
            unwind.as_deref_mut(),
            eh,
        );
    }
    asm_text.push_str(&wrappers.text);
    (ids, narrow)
}

pub(crate) fn is_large_aggregate(ty: &mir::Ty) -> bool {
    ty.is_aggregate() && ty.size_bytes() >= LIBC_COPY_THRESHOLD
}

pub(crate) fn needs_libc_memcpy(func: &mir::Function) -> bool {
    if is_large_aggregate(&func.sig.ret) || func.sig.params.iter().any(is_large_aggregate) {
        return true;
    }
    func.insts.iter().any(|inst| match &inst.kind {
        mir::InstKind::Memcpy { len, .. } => *len >= LIBC_COPY_THRESHOLD,
        mir::InstKind::Load { .. } => inst.result.is_some_and(|v| is_large_aggregate(&func.values[v].ty)),
        mir::InstKind::Store { val, .. } => is_large_aggregate(&func.values[*val].ty),
        mir::InstKind::InsertValue { agg, .. } => is_large_aggregate(&func.values[*agg].ty),
        _ => false,
    })
}

/// Inline-asm wrapper text collected while lowering one module,
/// deduplicated by symbol: the wrapper name is content-addressed, so
/// identical blocks share one definition.
#[derive(Default)]
pub struct Wrappers {
    pub(crate) seen: std::collections::HashSet<String>,
    fallbacks: std::collections::HashSet<String>,
    pub(crate) text: String,
}

impl Wrappers {
    pub(crate) fn add(&mut self, name: &str, text: &str) {
        if self.seen.insert(name.to_string()) {
            self.text.push_str(text);
        }
    }

    /// One note per distinct reason per module, so a unit full of
    /// `"+r"` blocks reports once.
    pub(crate) fn note_fallback(&mut self, reason: &str) {
        if self.fallbacks.insert(reason.to_string()) {
            eprintln!("glcc: note: inline asm lowered to the out-of-line wrapper: {reason}");
        }
    }
}

/// Cranelift linkage of one MIR definition: internal stays `Local`,
/// a weak definition is `Preemptible` (ELF `STB_WEAK`: a strong
/// definition in another TU wins, duplicates resolve), everything
/// else exports.
fn definition_linkage(local: bool, weak: bool) -> Linkage {
    if local {
        Linkage::Local
    } else if weak {
        Linkage::Preemptible
    } else {
        Linkage::Export
    }
}

/// Cranelift linkage of one undefined import: a weak declaration is
/// `Preemptible` too, which object marks `STB_WEAK`, so the linker
/// resolves it to zero when nothing defines it (GCC's behavior for
/// `extern ... __attribute__((weak))`).
fn import_linkage(weak: bool) -> Linkage {
    if weak { Linkage::Preemptible } else { Linkage::Import }
}

/// Whether any `F80` value flows through the module: signatures,
/// values, allocation points, and global storage. Over-approximating
/// is harmless (unused imports emit no relocations), under-
/// approximating would leave helper calls undeclared.
fn uses_f80(mir_module: &mir::Module) -> bool {
    fn ty_uses(ty: &mir::Ty) -> bool {
        match ty {
            mir::Ty::Float(mir::FloatKind::F80) => true,
            mir::Ty::Ptr { pointee, .. } => ty_uses(pointee),
            mir::Ty::Array { elem, .. } => ty_uses(elem),
            mir::Ty::Struct { fields, .. } => fields.iter().any(ty_uses),
            _ => false,
        }
    }
    mir_module.globals.iter().any(|g| ty_uses(&g.ty))
        || mir_module
            .funcs
            .iter()
            .any(|f| f.sig.params.iter().any(ty_uses) || ty_uses(&f.sig.ret) || f.values.iter().any(|v| ty_uses(&v.ty)))
}

/// Whether any `F128` value flows through the module (same shapes as
/// [`uses_f80`]): gates the quad-trampoline imports on non-x86-64
/// targets, whose `long double` lowers to `F128`.
fn uses_f128(mir_module: &mir::Module) -> bool {
    fn ty_uses(ty: &mir::Ty) -> bool {
        match ty {
            mir::Ty::Float(mir::FloatKind::F128) => true,
            mir::Ty::Ptr { pointee, .. } => ty_uses(pointee),
            mir::Ty::Array { elem, .. } => ty_uses(elem),
            mir::Ty::Struct { fields, .. } => fields.iter().any(ty_uses),
            _ => false,
        }
    }
    mir_module.globals.iter().any(|g| ty_uses(&g.ty))
        || mir_module
            .funcs
            .iter()
            .any(|f| f.sig.params.iter().any(ty_uses) || ty_uses(&f.sig.ret) || f.values.iter().any(|v| ty_uses(&v.ty)))
}

/// Copies at least this large call libc `memcpy` instead of unrolling
/// one load/store pair per 16 bytes. Smaller copies keep the inline
/// sequence to avoid call overhead on common small aggregates.
const LIBC_COPY_THRESHOLD: u64 = 256;

