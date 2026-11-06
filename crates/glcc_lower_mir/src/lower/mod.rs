use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

use glcc_frontend as fe;
use glcc_frontend::span::HasSpan;
use glcc_mir as mir;

use crate::ty::{complex_component, int_bits, is_complex, is_signed, lower_ty};

/// Address patches gathered while folding one global image: byte
/// offset, the relocation, and the patch width in bits (64 for ordinary
/// pointers, 32 for `__ptr32`). Split into `relocs`/`relocs32` once the
/// fold completes.
pub(crate) type GlobalRelocs = Vec<(u64, mir::DataReloc, u32)>;

/// Width of the address patch a pointer destination needs: its own
/// pointer width (`32` for `__ptr32`), or 64 for the integer-at-pointer-
/// width address case.
pub(crate) fn pointer_reloc_bits(ty: &mir::Ty) -> u32 {
    match ty {
        mir::Ty::Ptr { bits, .. } => *bits,
        _ => 64,
    }
}

/// Split a fold's address patches into pointer-width and 32-bit
/// (`__ptr32`) lists. Later initializers overwrite earlier ones at the
/// same offset (`{ [0 ... 2] = f, [0] = g }`), so keep only the last
/// patch per offset, in first-seen order.
pub(crate) fn split_global_relocs(relocs: GlobalRelocs) -> (Vec<(u64, mir::DataReloc)>, Vec<(u64, mir::DataReloc)>) {
    let mut seen = HashSet::new();
    let mut native = Vec::new();
    let mut narrow = Vec::new();
    for (off, reloc, bits) in relocs.into_iter().rev() {
        if !seen.insert(off) {
            continue;
        }
        if bits <= 32 {
            narrow.push((off, reloc));
        } else {
            native.push((off, reloc));
        }
    }
    native.reverse();
    narrow.reverse();
    (native, narrow)
}

/// Shared lowering context: checked semantic facts plus global symbols.
/// Everything else (the MIR function under construction and its frame)
/// threads through as explicit arguments, so nothing aliases.
pub struct Lowerer<'s, 'i> {
    pub(crate) sema: &'s fe::sema::Sema<'s>,
    pub(crate) interner: &'i fe::lexer::Interner,
    /// Unique suffix for synthesized anonymous globals (strings,
    /// compound literals).
    pub(crate) str_count: Cell<u64>,
    /// Debug type table dedup: frontend type to emitted entry.
    pub(crate) debug_ty_cache: RefCell<Vec<(fe::parse::Ty, mir::DebugTypeId)>>,
}

/// Which `<complex.h>`-style accessor a call lowers to inline.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ComplexBuiltin {
    Real,
    Imag,
    Conj,
    Abs,
}

/// A `va_*` builtin call target.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum VaBuiltin {
    Start,
    End,
    Copy,
    /// MSVC's `vadefs.h` spells `va_start` as `__va_start(&ap, x)` —
    /// the same operation as [`VaBuiltin::Start`], with the list
    /// passed by pointer.
    MsvcStart,
}

/// A single-instruction compiler builtin call target: bit counts,
/// byte swaps, float rounding, traps, and fences.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScalarBuiltin {
    Abs,
    Clz,
    Ctz,
    Clrsb,
    Parity,
    Popcnt,
    Copysign,
    Bswap,
    Ceil,
    Floor,
    Trap,
    Sync,
}

/// One function frame: a slot address per local, the block new
/// instructions append to, the loop stack for continue, the combined
/// break stack (loops and `switch`es push; `break` takes the top),
/// the open `switch`es collecting case targets, and the
/// function-scoped `goto` labels resolved during the single pass.
pub(crate) struct Frame {
    pub(crate) slots: HashMap<fe::sema::DeclId, mir::Value>,
    pub(crate) current: Option<mir::Block>,
    /// Source position of the statement being lowered, stamped onto
    /// every emitted instruction.
    pub(crate) loc: Option<mir::Location>,
    pub(crate) loops: Vec<LoopTarget>,
    pub(crate) breaks: Vec<BreakTarget>,
    pub(crate) switches: Vec<SwitchTarget>,
    /// Labels immediately prefixing the loop or switch being lowered
    /// (C2y N3355), consumed by that loop's target push.
    pub(crate) pending_loop_labels: Vec<string_interner::symbol::SymbolU32>,
    pub(crate) labels: HashMap<string_interner::symbol::SymbolU32, mir::Block>,
    /// The computed-goto dispatch block and its target-id parameter,
    /// created on the first `goto *expr` and filled in once the whole
    /// body is lowered (every label block exists by then).
    pub(crate) computed_goto: Option<(mir::Block, mir::Value)>,
    /// Head of the function's `alloca` block list, created on the first
    /// `alloca` call and drained at every return.
    pub(crate) alloca_head: Option<mir::Value>,
    /// Cached extents of every VLA object, by `ArrayBound::Vla` index,
    /// captured at the declaration (see `lower_vla_object`).
    pub(crate) vla_extents: HashMap<fe::sema::DeclId, HashMap<u32, mir::Value>>,
    /// The pointer slots of the function's VLA objects, drained at every
    /// return. Each slot holds the site's current allocation or null.
    pub(crate) vla_slots: Vec<mir::Value>,
}

pub(crate) struct LoopTarget {
    pub(crate) cont: mir::Block,
    pub(crate) labels: Vec<string_interner::symbol::SymbolU32>,
}

pub(crate) struct BreakTarget {
    pub(crate) block: mir::Block,
    pub(crate) labels: Vec<string_interner::symbol::SymbolU32>,
}

/// One open `switch`: the promoted controlling type both sides
/// compare in and the promoted scrutinee value, plus the case
/// targets collected while the body lowers (the dispatch block itself
/// stays a local: it is terminated once every case exists).
pub(crate) struct SwitchTarget {
    pub(crate) promoted: fe::parse::Ty,
    pub(crate) scrutinee: mir::Value,
    pub(crate) cases: Vec<(i128, mir::Block)>,
    /// GNU `case lo ... hi:` targets, tested with comparisons before
    /// the exact-value dispatch.
    pub(crate) ranges: Vec<(i128, i128, mir::Block)>,
    pub(crate) default: Option<mir::Block>,
}

/// A bit-field store target: the covering unit type plus field
/// positions. Loads compute their own sign from the declared
/// type; stores need only these.
#[derive(Debug, Clone)]
pub(crate) struct BitfieldTarget {
    pub(crate) unit_ty: fe::parse::Ty,
    pub(crate) bit: u32,
    pub(crate) width: u32,
}

/// The object an address-constant expression roots at: a declared
/// global (or function), a string literal, or a compound literal. The
/// last two carry no declaration and materialize their own objects.
pub(crate) enum AddrRoot<'a> {
    Decl(fe::sema::DeclId),
    Str(&'a [fe::lexer::StringLiteral]),
    Compound(&'a fe::parse::Ty, &'a fe::parse::Initializer<'a>),
}

/// Assembler text for `__attribute__((naked))` definitions: their
/// bodies cannot be lowered (there is no compiler prologue to hold C
/// code), so each becomes a whole function in the per-TU `.s` output.
/// The parser guarantees the body holds only basic asm statements.
pub fn naked_asm_text(sema: &fe::sema::Sema, interner: &fe::lexer::Interner, tu: &fe::parse::TranslationUnit) -> String {
    let lx = Lowerer {
        sema,
        interner,
        str_count: Cell::new(0),
        debug_ty_cache: RefCell::new(Vec::new()),
    };
    let mut out = String::new();
    let mut emitted: std::collections::HashSet<fe::sema::DeclId> = std::collections::HashSet::new();
    for decl in &tu.external_declarations {
        let fe::parse::ExternalDeclaration::Func(id) = decl else { continue };
        let binding = &sema.decls[id.0 as usize];
        if !binding.naked || binding.alias.is_some() {
            continue;
        }
        let Some(body) = sema.bodies.get(id) else { continue };
        if !emitted.insert(*id) {
            continue;
        }
        let name = lx.symbol_name(binding, *id);
        if let Some(section) = binding.section {
            out.push_str(".section ");
            out.push_str(interner.resolve(section).expect("section name symbol"));
            out.push_str(",\"ax\"\n");
        }
        // The definition lands in its own object, so the C object's
        // reference needs a global symbol even for `static` functions.
        out.push_str(".globl ");
        out.push_str(&name);
        out.push('\n');
        out.push_str(".type ");
        out.push_str(&name);
        out.push_str(", @function\n");
        out.push_str(&name);
        out.push_str(":\n");
        for item in body.block.items {
            match item {
                fe::parse::BlockItem::Stmt(fe::parse::Stmt::Asm(asm)) => {
                    out.push_str(&asm.template);
                    out.push('\n');
                }
                fe::parse::BlockItem::Stmt(fe::parse::Stmt::Empty { .. }) => {}
                _ => panic!("naked function body may only contain basic asm statements"),
            }
        }
        out.push_str(".size ");
        out.push_str(&name);
        out.push_str(", .-");
        out.push_str(&name);
        out.push('\n');
        if binding.section.is_some() {
            out.push_str(".previous\n");
        }
    }
    out
}

/// `inline` definitions whose bodies are dropped at declaration and
/// revived as TU-local copies only when referenced (see the revival
/// pass in `lower_translation_unit`). Three cases:
///
/// - `static inline` (clang's intrinsics headers spell every `_mm_*`
///   this way): nothing observable, so emitting only referenced
///   bodies matches GCC/clang and keeps an included-but-unused
///   header from lowering hundreds of functions per TU.
/// - MSVC's headers define every CRT wrapper as a non-static
///   `__inline` and rely on COMDAT folding (or the identical CRT
///   export) to avoid duplicate definitions; glcc emits neither, so
///   on the Windows target calls resolve to the CRT export.
/// - `__attribute__((gnu_inline))` + `extern inline` (GCC's headers
///   spell every intrinsic `extern __inline` this way) provides no
///   external definition at all under GCC semantics — emitting each
///   body would both bloat every TU and collide across them.
fn skips_extern_inline(sema: &fe::sema::Sema, binding: &fe::sema::Binding) -> bool {
    if binding.inline && binding.linkage == fe::sema::Linkage::Internal {
        return true;
    }
    binding.inline && binding.linkage == fe::sema::Linkage::External && (sema.target.os == fe::target::Os::Windows || binding.gnu_inline)
}

/// Lower a whole checked translation unit to a MIR module.
pub fn lower_translation_unit(sema: &fe::sema::Sema, interner: &fe::lexer::Interner, tu: &fe::parse::TranslationUnit) -> mir::Module {
    let lx = Lowerer {
        sema,
        interner,
        str_count: Cell::new(0),
        debug_ty_cache: RefCell::new(Vec::new()),
    };
    let mut module = mir::ModuleBuilder::new();
    let mut funcs: HashMap<fe::sema::DeclId, mir::Func> = HashMap::new();
    let mut externs: HashMap<fe::sema::DeclId, mir::Extern> = HashMap::new();
    let mut globals: HashMap<fe::sema::DeclId, mir::Global> = HashMap::new();
    // Globals whose initializers still need folding once every
    // entity's id exists (address constants may point forward).
    let mut define_order: Vec<fe::sema::DeclId> = Vec::new();
    for decl in &tu.external_declarations {
        // A prototype and its definition share one `DeclId` but yield
        // two entries; declaring twice would leak an empty shell the
        // body never replaces, so each entity declares once.
        match decl {
            fe::parse::ExternalDeclaration::Func(id) => {
                let binding = &sema.decls[id.0 as usize];
                // An `alias` declaration denotes another entity: its
                // uses map to that entity's function below.
                if binding.alias.is_some() {
                    continue;
                }
                if sema.bodies.contains_key(id) {
                    if funcs.contains_key(id) {
                        continue;
                    }
                    // MSVC's headers define every CRT wrapper as a
                    // non-static `__inline` and rely on COMDAT folding
                    // (or the identical CRT export) to avoid duplicate
                    // definitions; glcc emits neither, so the body is
                    // dropped and calls resolve to the CRT exports.
                    // Referenced wrappers are revived after lowering
                    // (see the revival pass below).
                    if skips_extern_inline(sema, binding) {
                        if !externs.contains_key(id) {
                            let sig = lx.func_sig(binding);
                            let name = lx.symbol_name(binding, *id);
                            let eid = module.module.push_extern(&name, sig);
                            module.module.externs[eid].weak = false;
                            externs.insert(*id, eid);
                        }
                        continue;
                    }
                    if binding.naked {
                        // Naked definitions are emitted as assembler
                        // text; callers link against the symbol.
                        if !externs.contains_key(id) {
                            let sig = lx.func_sig(binding);
                            let name = lx.symbol_name(binding, *id);
                            let eid = module.module.push_extern(&name, sig);
                            module.module.externs[eid].weak = binding.weak;
                            externs.insert(*id, eid);
                        }
                        continue;
                    }
                    let sig = lx.func_sig(binding);
                    let name = lx.symbol_name(binding, *id);
                    let (fid, _) = module.declare(&name, sig);
                    // Internal linkage stays TU-local so identical
                    // `static` definitions never collide at link.
                    // Weak definitions (including EDG's C++ vague
                    // linkage) stay preemptible. The C `inline` hint
                    // rides along for the `inline` pass (see
                    // `Function::inline`).
                    module.module.funcs[fid].local = binding.linkage == fe::sema::Linkage::Internal;
                    module.module.funcs[fid].weak = binding.weak;
                    module.module.funcs[fid].inline = binding.inline;
                    module.module.funcs[fid].force_inline = binding.forceinline;
                    // An async definition's body is a coroutine over
                    // its declared result; `LowerCoroutines` splits it
                    // after lowering (see `Function::coroutine`).
                    if binding.is_async {
                        let result = lower_ty(lx.declared_ret(binding), lx.sema, lx.interner);
                        module.module.funcs[fid].coroutine = Some(result);
                    }
                    funcs.insert(*id, fid);
                } else {
                    // Prototype without a body: a foreign import for
                    // `CallExtern` (C library, runtime helpers).
                    if externs.contains_key(id) {
                        continue;
                    }
                    let sig = lx.func_sig(binding);
                    let name = lx.symbol_name(binding, *id);
                    let eid = module.module.push_extern(&name, sig);
                    module.module.externs[eid].weak = binding.weak;
                    externs.insert(*id, eid);
                }
            }
            fe::parse::ExternalDeclaration::Var(id) => {
                let binding = &sema.decls[id.0 as usize];
                if binding.defined {
                    if globals.contains_key(id) {
                        continue;
                    }
                    let gid = lx.declare_global(&mut module, *id);
                    globals.insert(*id, gid);
                    define_order.push(*id);
                }
            }
        }
    }
    // Bodiless functions missed above (implicit C89 declarations
    // never reach the declare pass): every referenced prototype needs
    // an import, so sweep the rest now while the maps stay immutable
    // for lowering. A declaration naming a function defined in this TU
    // — a block-scope prototype whose file-scope definition comes
    // later, say — resolves to that definition: external linkage makes
    // them one entity, however many `DeclId`s spell it.
    for (i, binding) in sema.decls.iter().enumerate() {
        if !matches!(binding.kind, fe::sema::BindingKind::Func) {
            continue;
        }
        let id = fe::sema::DeclId(i as u32);
        if sema.bodies.contains_key(&id) || funcs.contains_key(&id) || externs.contains_key(&id) || binding.alias.is_some() {
            continue;
        }
        let defined = binding
            .name
            .and_then(|name| funcs.iter().find_map(|(&did, &target)| (sema.decls[did.0 as usize].name == Some(name)).then_some(target)));
        if let Some(target) = defined {
            funcs.insert(id, target);
            continue;
        }
        let sig = lx.func_sig(binding);
        let name = lx.symbol_name(binding, id);
        let eid = module.module.push_extern(&name, sig);
        module.module.externs[eid].weak = binding.weak;
        externs.insert(id, eid);
    }
    // Block-scope `static` objects live in module globals, folded
    // once in `define_global` so calls never re-run their
    // initializers. Pure `extern` declarations (file-scope like
    // `stderr`, or block-scope with no file-scope entity) likewise
    // declare their import here, so uses resolve to a symbol instead
    // of a slot. The scans cover every declaration in the TU upfront,
    // keeping the function-lowering maps immutable.
    for (i, binding) in sema.decls.iter().enumerate() {
        if binding.block_static {
            let id = fe::sema::DeclId(i as u32);
            if let std::collections::hash_map::Entry::Vacant(e) = globals.entry(id) {
                let gid = lx.declare_block_static(&mut module, id);
                e.insert(gid);
                define_order.push(id);
            }
        }
        // A pure `extern` declaration introduces no storage: it reuses
        // the (possibly imported) global entity, mirroring the legacy
        // backend's `ensure_global_data` on first reference.
        if binding.kind == fe::sema::BindingKind::Var && binding.linkage == fe::sema::Linkage::External && !binding.defined && !binding.tentative && binding.alias.is_none() {
            let id = fe::sema::DeclId(i as u32);
            globals.entry(id).or_insert_with(|| {
                let gid = lx.lower_data_extern(&mut module, id);
                gid
            });
        }
    }
    // `__attribute__((alias("a")))` declarations denote the target
    // entity, so every use resolves to the target's global, import, or
    // function; no symbol is emitted for the alias spelling itself.
    for (i, binding) in sema.decls.iter().enumerate() {
        let Some(target) = binding.alias else { continue };
        let id = fe::sema::DeclId(i as u32);
        if let Some(&gid) = globals.get(&target) {
            globals.insert(id, gid);
        } else if let Some(&fid) = funcs.get(&target) {
            funcs.insert(id, fid);
        } else if let Some(&eid) = externs.get(&target) {
            externs.insert(id, eid);
        }
    }
    // Fold initializers now that every function, global, and import
    // id exists for address constants to resolve against.
    for id in define_order {
        let gid = globals[&id];
        lx.define_global(&mut module, &funcs, &externs, &globals, id, gid);
    }
    let mut lowered: HashSet<fe::sema::DeclId> = HashSet::new();
    for decl in &tu.external_declarations {
        if let fe::parse::ExternalDeclaration::Func(id) = decl {
            // Naked definitions come from the per-TU assembler text.
            if sema.decls[id.0 as usize].naked {
                continue;
            }
            // MSVC header `__inline` bodies were dropped at declaration
            // (see `skips_msvc_extern_inline`); referenced ones are
            // revived after this loop, with the import standing in for
            // the rest.
            if sema.bodies.contains_key(id) && skips_extern_inline(sema, &sema.decls[id.0 as usize]) {
                continue;
            }
            // Duplicate entries (prototype plus definition) lower once.
            if sema.bodies.contains_key(id) && lowered.insert(*id) {
                // Lower into a fresh function, then install it over the
                // predeclared shell: `&mut module` and `&mut func` stay
                // distinct objects throughout lowering.
                let binding = &sema.decls[id.0 as usize];
                let name = lx.symbol_name(binding, *id);
                let (mut func, _) = mir::Function::new(&name, lx.func_sig(binding));
                func.local = binding.linkage == fe::sema::Linkage::Internal;
                func.weak = binding.weak;
                func.inline = binding.inline;
                func.force_inline = binding.forceinline;
                func.noinline = binding.noinline;
                // The fresh `Function` replaces the predeclared shell,
                // so facts recorded at declaration time ride along: an
                // async body lowers as a coroutine over its declared
                // result (`func_sig` above handed the symbol the handle
                // return instead).
                func.coroutine = module.module.funcs[funcs[id]].coroutine.clone();
                let mut frame = Frame {
                    slots: HashMap::new(),
                    current: None,
                    loc: None,
                    loops: Vec::new(),
                    breaks: Vec::new(),
                    switches: Vec::new(),
                    pending_loop_labels: Vec::new(),
                    labels: HashMap::new(),
                    computed_goto: None,
                    alloca_head: None,
                    vla_extents: HashMap::new(),
                    vla_slots: Vec::new(),
                };
                lx.lower_function(&mut module, &mut func, &mut frame, &funcs, &externs, &globals, *id);
                module.module.funcs[funcs[id]] = func;
            }
        }
    }
    // Revive referenced MSVC `__inline` wrappers whose bodies were
    // dropped at declaration. Names like `sprintf` fold into the inline
    // body under cl and mingw and have no UCRT import to fall back on,
    // so the body is lowered here as a TU-local copy: the `__inline`
    // the header asked for, spelled `static inline`. Bodies may
    // reference further dropped wrappers, so this runs to a fixpoint.
    let mut revived: HashSet<mir::Extern> = HashSet::new();
    loop {
        let mut used: HashSet<mir::Extern> = HashSet::new();
        for func in &module.module.funcs {
            for block in &func.blocks {
                for &inst in &block.insts {
                    match &func.insts[inst].kind {
                        mir::InstKind::CallExtern { target, .. } | mir::InstKind::ExternAddr { target } => {
                            used.insert(*target);
                        }
                        _ => {}
                    }
                }
            }
        }
        for global in &module.module.globals {
            for (_, reloc) in global.relocs.iter().chain(&global.relocs32) {
                if let mir::DataReloc::FuncExtern { target } = reloc {
                    used.insert(*target);
                }
            }
        }
        // Sort for reproducible object layout; the source map is a
        // `HashMap` with no order of its own.
        let mut pending: Vec<(fe::sema::DeclId, mir::Extern)> = Vec::new();
        for (id, eid) in &externs {
            if used.contains(eid) && !revived.contains(eid) && sema.bodies.contains_key(id) && skips_extern_inline(sema, &sema.decls[id.0 as usize]) {
                pending.push((*id, *eid));
            }
        }
        if pending.is_empty() {
            break;
        }
        pending.sort_by_key(|(id, _)| id.0);
        for (decl_id, extern_id) in pending {
            let binding = &sema.decls[decl_id.0 as usize];
            let name = lx.symbol_name(binding, decl_id);
            let (fid, _) = module.declare(&name, lx.func_sig(binding));
            let (mut func, _) = mir::Function::new(&name, lx.func_sig(binding));
            func.local = true;
            func.inline = binding.inline;
            func.force_inline = binding.forceinline;
            func.noinline = binding.noinline;
            func.coroutine = module.module.funcs[fid].coroutine.clone();
            let mut frame = Frame {
                slots: HashMap::new(),
                current: None,
                loc: None,
                loops: Vec::new(),
                breaks: Vec::new(),
                switches: Vec::new(),
                pending_loop_labels: Vec::new(),
                labels: HashMap::new(),
                computed_goto: None,
                alloca_head: None,
                vla_extents: HashMap::new(),
                vla_slots: Vec::new(),
            };
            // Map the declaration before lowering so the body's own
            // calls (and later lookups) bind to the copy.
            funcs.insert(decl_id, fid);
            lx.lower_function(&mut module, &mut func, &mut frame, &funcs, &externs, &globals, decl_id);
            module.module.funcs[fid] = func;
            // Calls lowered before this revival bound the extern;
            // rebind every reference to the copy.
            for caller in module.module.funcs.iter_mut() {
                let inst_ids: Vec<mir::Inst> = caller.blocks.iter().flat_map(|b| b.insts.iter().copied()).collect();
                for inst in inst_ids {
                    let replacement = match &caller.insts[inst].kind {
                        mir::InstKind::CallExtern { target, args } if *target == extern_id => Some(mir::InstKind::Call { callee: fid, args: args.clone() }),
                        mir::InstKind::ExternAddr { target } if *target == extern_id => Some(mir::InstKind::FuncAddr { func: fid }),
                        _ => None,
                    };
                    if let Some(kind) = replacement {
                        caller.insts[inst].kind = kind;
                    }
                }
            }
            for global in module.module.globals.iter_mut() {
                for (_, reloc) in global.relocs.iter_mut().chain(&mut global.relocs32) {
                    if matches!(reloc, mir::DataReloc::FuncExtern { target } if *target == extern_id) {
                        *reloc = mir::DataReloc::Func { target: fid };
                    }
                }
            }
            revived.insert(extern_id);
        }
    }
    // One `.init_array`/`.fini_array` per TU collecting the defined
    // constructor/destructor functions, now that every body exists.
    lx.emit_init_fini_arrays(&mut module, &funcs);
    module.finish()
}

mod addr;
mod atomic;
mod builtin;
mod complex;
mod data;
mod emit;
mod expr;
mod fp;
mod function;
mod ia32;
mod init;
mod vector;
mod vla;
#[cfg(test)]
mod tests;

impl<'s, 'i> Lowerer<'s, 'i> {
    pub(crate) fn symbol_name(&self, binding: &fe::sema::Binding, id: fe::sema::DeclId) -> String {
        // An asm label renames the emitted symbol (EDG uses this for
        // overloaded C library declarations); the C spelling stays the
        // declaration's lookup key.
        match binding.asm_label.or(binding.name) {
            Some(sym) => self.interner.resolve(sym).unwrap_or("<anon>").to_string(),
            // Block-scope entities share spellings; the id disambiguates.
            None => format!("<anon>.{}", id.0),
        }
    }

    /// The declared (C-level) return type of a function binding.
    pub(crate) fn declared_ret<'b>(&self, binding: &'b fe::sema::Binding) -> &'b fe::parse::Ty {
        let fe::parse::Ty::Func { ret, .. } = binding.ty.unqualified() else {
            panic!("function binding without a Func type");
        };
        ret
    }

    pub(crate) fn func_sig(&self, binding: &fe::sema::Binding) -> mir::Signature {
        let fe::parse::Ty::Func { params, variadic, prototyped, .. } = binding.ty.unqualified() else {
            panic!("function binding without a Func type");
        };
        // An async function's public symbol returns its boxed
        // `_Future(T)` handle, not the declared result (`LowerCoroutines`
        // turns the body into start/resume pair); callers type their
        // calls from the handle, so both sides of the ABI agree.
        let ret = if binding.is_async {
            let handle = self
                .sema
                .future_handle_of(self.declared_ret(binding))
                .expect("async function without a registered future handle");
            lower_ty(&handle, self.sema, self.interner)
        } else {
            lower_ty(self.declared_ret(binding), self.sema, self.interner)
        };
        let mut sig = mir::Signature::new(params.iter().map(|p| lower_ty(p, self.sema, self.interner)).collect(), ret);
        // Unprototyped declarations (`int f()`, C89 implicit
        // `extern int f()`) fix no parameter list: callers supply any
        // arguments, default-promoted, so their signature takes the
        // variadic shape. A definition with no parameters (`int f()`)
        // never reads its arguments (no `...`, so no `va_start`),
        // though — the variadic register save area would be dead
        // weight, so it keeps a fixed signature.
        let has_body = binding.defined;
        sig.variadic = *variadic || (!*prototyped && (!has_body || !params.is_empty()));
        sig
    }

}

pub(crate) fn float_kind(ty: &mir::Ty) -> mir::FloatKind {
    match ty {
        mir::Ty::Float(kind) => *kind,
        _ => panic!("not a float MIR type"),
    }
}

/// Whether the outermost level of a frontend type carries `volatile`.
/// `Ty::qualified` distributes qualifiers onto what they qualify, so
/// the outermost level is exactly the accessed object's own.
pub(crate) fn ty_is_volatile(ty: &fe::parse::Ty) -> bool {
    use fe::parse::TypeQualifiers;
    const VOL: TypeQualifiers = TypeQualifiers::VOLATILE;
    match ty {
        fe::parse::Ty::Qualified { quals, .. } => quals.contains(VOL),
        fe::parse::Ty::Pointer { qualifiers, .. } => qualifiers.contains(VOL),
        fe::parse::Ty::Array { quals, .. } => quals.contains(VOL),
        _ => false,
    }
}

/// Whether reading/writing `e` touches volatile storage: its own
/// (lvalue) type's qualifier, or the object it designates. The
/// checker's lvalue conversion strips top-level qualifiers from value
/// expressions, so a read of `*p` for a `volatile int *p` arrives here
/// typed `int`; resolving the designated object — the declared type
/// for an identifier, the pointee for a dereference or subscript, the
/// base object for a member — recovers what the access actually
/// touches. Wrappers that drop or retarget qualifiers (`__real__`,
/// `__imag__`, casts, assignment conversions) recurse.
pub(crate) fn access_volatile(e: &fe::check::CheckedExpr, decls: &[fe::sema::Binding]) -> bool {
    if ty_is_volatile(&e.ty) {
        return true;
    }
    match &e.kind {
        fe::check::CheckedExprKind::Ident(id) => ty_is_volatile(&decls[id.0 as usize].ty),
        fe::check::CheckedExprKind::Unary(fe::parse::UnaryOp::Deref, inner) => pointee_volatile(&inner.ty),
        fe::check::CheckedExprKind::Subscript { base, .. } => pointee_volatile(&base.ty),
        fe::check::CheckedExprKind::Member { base, arrow, decl_ty, .. } => {
            ty_is_volatile(decl_ty) || if *arrow { pointee_volatile(&base.ty) } else { access_volatile(base, decls) }
        }
        fe::check::CheckedExprKind::Unary(op, inner) if matches!(op, fe::parse::UnaryOp::Real | fe::parse::UnaryOp::Imag) => access_volatile(inner, decls),
        fe::check::CheckedExprKind::Cast { expr: inner, .. } | fe::check::CheckedExprKind::Convert { expr: inner, .. } => access_volatile(inner, decls),
        _ => false,
    }
}

/// Qualifiers of the object a pointer (or decayed array) designates.
/// The pointee is not touched by lvalue conversion, unlike the value
/// expression's own top-level qualifiers.
pub(crate) fn pointee_volatile(ty: &fe::parse::Ty) -> bool {
    match ty.unqualified() {
        fe::parse::Ty::Pointer { pointee, .. } => ty_is_volatile(pointee),
        fe::parse::Ty::Array { elem, .. } => ty_is_volatile(elem),
        _ => false,
    }
}

/// A scalar builtin whose calls take exactly one argument: a
/// declaration with any other shape names a user function, not the
/// builtin (GCC disables the builtin the same way).
pub(crate) fn unary_builtin(binding: &fe::sema::Binding, builtin: ScalarBuiltin) -> Option<ScalarBuiltin> {
    match binding.ty.unqualified() {
        fe::parse::Ty::Func { params, .. } if params.len() == 1 => Some(builtin),
        _ => None,
    }
}

/// The binary operation a compound assignment performs.
pub(crate) fn compound_bin_op(op: &fe::parse::AssignOp) -> fe::parse::BinaryOp {
    use fe::parse::AssignOp;
    use fe::parse::BinaryOp;
    match op {
        AssignOp::Add => BinaryOp::Add,
        AssignOp::Sub => BinaryOp::Sub,
        AssignOp::Mul => BinaryOp::Mul,
        AssignOp::Div => BinaryOp::Div,
        AssignOp::Rem => BinaryOp::Rem,
        AssignOp::Shl => BinaryOp::Shl,
        AssignOp::Shr => BinaryOp::Shr,
        AssignOp::BitAnd => BinaryOp::BitAnd,
        AssignOp::BitXor => BinaryOp::BitXor,
        AssignOp::BitOr => BinaryOp::BitOr,
        AssignOp::Assign => panic!("plain assignment is not a compound operation"),
    }
}

/// Whether a MIR type lowers to a Clif scalar (and so fits the atomic
/// scalar instructions): integers, bools, floats, and pointers.
/// Aggregates copy plain bytes instead.
pub(crate) fn is_mir_scalar(ty: &mir::Ty) -> bool {
    matches!(ty, mir::Ty::Bool | mir::Ty::Int { .. } | mir::Ty::Float(_) | mir::Ty::Ptr { .. })
}

pub(crate) fn float_bits(value: fe::eval::FloatValue, ty: &mir::Ty) -> u128 {
    match float_kind(ty) {
        mir::FloatKind::F32 => u128::from(value.to_f32().to_bits()),
        mir::FloatKind::F64 => u128::from(value.to_f64().to_bits()),
        mir::FloatKind::F128 => value.to_quad_bits(),
        mir::FloatKind::F80 => u128::from_le_bytes(value.to_f80_bytes()),
    }
}

/// The integer form a float bitcasts to, and the position of its sign
/// bit: 31/63 for the IEEE binary32/64, 127 for binary128, and 79 for
/// x87 long doubles (whose register form keeps the 80-bit pattern
/// padded to 128 bits).
pub(crate) fn float_int_form(ty: &mir::Ty) -> (mir::Ty, u32) {
    match float_kind(ty) {
        mir::FloatKind::F32 => (mir::Ty::int(32), 31),
        mir::FloatKind::F64 => (mir::Ty::int(64), 63),
        mir::FloatKind::F80 => (mir::Ty::int(128), 79),
        mir::FloatKind::F128 => (mir::Ty::int(128), 127),
    }
}

/// Bit pattern of the kind's positive infinity (every exponent bit).
pub(crate) fn float_inf_bits(ty: &mir::Ty) -> u128 {
    match float_kind(ty) {
        mir::FloatKind::F32 => u128::from(0x7f800000u32),
        mir::FloatKind::F64 => u128::from(0x7ff0000000000000u64),
        mir::FloatKind::F80 => float_bits(fe::eval::FloatValue::long_double_infinity_in(fe::target::LongDoubleKind::F80), ty),
        mir::FloatKind::F128 => float_bits(fe::eval::FloatValue::long_double_infinity_in(fe::target::LongDoubleKind::F128), ty),
    }
}

/// Bit pattern of the kind's smallest normal: exponent field one with
/// a zero mantissa — the x87 form sets its explicit integer bit (bit
/// 63) and puts the exponent's low bit at 64; binary128 likewise at
/// 112.
pub(crate) fn float_min_normal_bits(ty: &mir::Ty) -> u128 {
    match float_kind(ty) {
        mir::FloatKind::F32 => u128::from(f32::MIN_POSITIVE.to_bits()),
        mir::FloatKind::F64 => u128::from(f64::MIN_POSITIVE.to_bits()),
        mir::FloatKind::F80 => (1u128 << 64) | (1u128 << 63),
        mir::FloatKind::F128 => 1u128 << 112,
    }
}

// These tests execute generated code in-process, so the harness runs
// the host target end to end: the frontend lowers for
// `fe::target::Target::native()` and the JIT uses
// `glcc_lower_cl::Target::native()`.

/// Positional index of a named struct member.
pub(crate) fn member_index(sema: &fe::sema::Sema, id: fe::sema::StructId, name: string_interner::symbol::SymbolU32) -> u32 {
    sema.structs[id.0 as usize]
        .members
        .iter()
        .position(|m| m.name == Some(name))
        .unwrap_or_else(|| panic!("unknown struct member")) as u32
}

/// The member a struct-level designator names: `Field` searches by
/// name, `AnonField` is a hop through an anonymous member that the
/// checker already resolved to an index.
pub(crate) fn designated_member(sema: &fe::sema::Sema, id: fe::sema::StructId, des: &fe::parse::Designator) -> u32 {
    match des {
        fe::parse::Designator::Field { name, .. } => member_index(sema, id, *name),
        fe::parse::Designator::AnonField { index, .. } => *index,
        fe::parse::Designator::Index { .. } => panic!("index designator into struct"),
    }
}
