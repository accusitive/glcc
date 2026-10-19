use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

use glcc_frontend as fe;
use glcc_frontend::span::HasSpan;
use glcc_mir as mir;

use crate::ty::{complex_component, int_bits, is_complex, is_signed, lower_ty};

/// Shared lowering context: checked semantic facts plus global symbols.
/// Everything else (the MIR function under construction and its frame)
/// threads through as explicit arguments, so nothing aliases.
pub struct Lowerer<'s, 'i> {
    pub(crate) sema: &'s fe::sema::Sema,
    pub(crate) interner: &'i fe::lexer::Interner,
    /// Unique suffix for synthesized anonymous globals (strings,
    /// compound literals).
    pub(crate) str_count: Cell<u64>,
    /// Debug type table dedup: frontend type to emitted entry.
    pub(crate) debug_ty_cache: RefCell<Vec<(fe::parse::Ty, mir::DebugTypeId)>>,
}

/// Which `<complex.h>`-style accessor a call lowers to inline.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ComplexBuiltin {
    Real,
    Imag,
    Conj,
    Abs,
}

/// A `va_*` builtin call target.
#[derive(Clone, Copy, PartialEq, Eq)]
enum VaBuiltin {
    Start,
    End,
    Copy,
}

/// A single-instruction compiler builtin call target: bit counts,
/// byte swaps, float rounding, traps, and fences.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ScalarBuiltin {
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
struct Frame {
    slots: HashMap<fe::sema::DeclId, mir::Value>,
    current: Option<mir::Block>,
    /// Source position of the statement being lowered, stamped onto
    /// every emitted instruction.
    loc: Option<mir::Location>,
    loops: Vec<LoopTarget>,
    breaks: Vec<mir::Block>,
    switches: Vec<SwitchTarget>,
    labels: HashMap<string_interner::symbol::SymbolU32, mir::Block>,
    /// The computed-goto dispatch block and its target-id parameter,
    /// created on the first `goto *expr` and filled in once the whole
    /// body is lowered (every label block exists by then).
    computed_goto: Option<(mir::Block, mir::Value)>,
    /// Head of the function's `alloca` block list, created on the first
    /// `alloca` call and drained at every return.
    alloca_head: Option<mir::Value>,
    /// Cached extents of every VLA object, by `ArrayBound::Vla` index,
    /// captured at the declaration (see `lower_vla_object`).
    vla_extents: HashMap<fe::sema::DeclId, HashMap<u32, mir::Value>>,
    /// The pointer slots of the function's VLA objects, drained at every
    /// return. Each slot holds the site's current allocation or null.
    vla_slots: Vec<mir::Value>,
}

struct LoopTarget {
    cont: mir::Block,
}

/// One open `switch`: the promoted controlling type both sides
/// compare in and the promoted scrutinee value, plus the case
/// targets collected while the body lowers (the dispatch block itself
/// stays a local: it is terminated once every case exists).
struct SwitchTarget {
    promoted: fe::parse::Ty,
    scrutinee: mir::Value,
    cases: Vec<(i128, mir::Block)>,
    /// GNU `case lo ... hi:` targets, tested with comparisons before
    /// the exact-value dispatch.
    ranges: Vec<(i128, i128, mir::Block)>,
    default: Option<mir::Block>,
}

/// A bit-field store target: the covering unit type plus field
/// positions. Loads compute their own sign from the declared
/// type; stores need only these.
#[derive(Debug, Clone)]
struct BitfieldTarget {
    unit_ty: fe::parse::Ty,
    bit: u32,
    width: u32,
}

/// The object an address-constant expression roots at: a declared
/// global (or function), a string literal, or a compound literal. The
/// last two carry no declaration and materialize their own objects.
enum AddrRoot<'a> {
    Decl(fe::sema::DeclId),
    Str(&'a [fe::lexer::StringLiteral]),
    Compound(&'a fe::parse::Ty, &'a fe::parse::Initializer),
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
        for item in &body.block.items {
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

/// MSVC's headers define every CRT wrapper as a non-static `__inline`
/// and rely on COMDAT folding (or the identical CRT export) to avoid
/// duplicate definitions; glcc emits neither, so on the Windows target
/// the body is dropped at declaration and calls resolve to the CRT
/// export. Wrappers that are actually referenced and have no export to
/// resolve to (`sprintf`) are revived as TU-local copies after
/// lowering; see the revival pass in `lower_translation_unit`.
fn skips_msvc_extern_inline(sema: &fe::sema::Sema, binding: &fe::sema::Binding) -> bool {
    sema.target.os == fe::target::Os::Windows && binding.inline && binding.linkage == fe::sema::Linkage::External
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
                    if skips_msvc_extern_inline(sema, binding) {
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
            if !globals.contains_key(&id) {
                let gid = lx.declare_block_static(&mut module, id);
                globals.insert(id, gid);
                define_order.push(id);
            }
        }
        // A pure `extern` declaration introduces no storage: it reuses
        // the (possibly imported) global entity, mirroring the legacy
        // backend's `ensure_global_data` on first reference.
        if binding.kind == fe::sema::BindingKind::Var && binding.linkage == fe::sema::Linkage::External && !binding.defined && !binding.tentative && binding.alias.is_none() {
            let id = fe::sema::DeclId(i as u32);
            if !globals.contains_key(&id) {
                let gid = lx.lower_data_extern(&mut module, id);
                globals.insert(id, gid);
            }
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
            if sema.bodies.contains_key(id) && skips_msvc_extern_inline(sema, &sema.decls[id.0 as usize]) {
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
            for (_, reloc) in &global.relocs {
                if let mir::DataReloc::FuncExtern { target } = reloc {
                    used.insert(*target);
                }
            }
        }
        // Sort for reproducible object layout; the source map is a
        // `HashMap` with no order of its own.
        let mut pending: Vec<(fe::sema::DeclId, mir::Extern)> = Vec::new();
        for (id, eid) in &externs {
            if used.contains(eid) && !revived.contains(eid) && sema.bodies.contains_key(id) && skips_msvc_extern_inline(sema, &sema.decls[id.0 as usize]) {
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
            func.noinline = binding.noinline;
            func.coroutine = module.module.funcs[fid].coroutine.clone();
            let mut frame = Frame {
                slots: HashMap::new(),
                current: None,
                loc: None,
                loops: Vec::new(),
                breaks: Vec::new(),
                switches: Vec::new(),
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
                for (_, reloc) in &mut global.relocs {
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

impl<'s, 'i> Lowerer<'s, 'i> {
    fn symbol_name(&self, binding: &fe::sema::Binding, id: fe::sema::DeclId) -> String {
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
    fn declared_ret<'b>(&self, binding: &'b fe::sema::Binding) -> &'b fe::parse::Ty {
        let fe::parse::Ty::Func { ret, .. } = binding.ty.unqualified() else {
            panic!("function binding without a Func type");
        };
        &**ret
    }

    fn func_sig(&self, binding: &fe::sema::Binding) -> mir::Signature {
        let fe::parse::Ty::Func { params, variadic, .. } = binding.ty.unqualified() else {
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
        sig.variadic = *variadic;
        sig
    }

    /// Declare a file-scope object's storage; the initializer folds
    /// in `define_global` once every entity's id exists, since address
    /// constants may point forward.
    fn declare_global(&self, module: &mut mir::ModuleBuilder, id: fe::sema::DeclId) -> mir::Global {
        let binding = &self.sema.decls[id.0 as usize];
        let name = self.symbol_name(binding, id);
        let local = binding.linkage == fe::sema::Linkage::Internal;
        self.declare_global_with_name(module, id, &name, local)
    }

    /// A block-scope `static` object has static storage duration: it
    /// lives in a module global defined at load time, not in a
    /// per-call slot. The name carries the `DeclId` (two blocks may
    /// share a spelling) mirroring the legacy backend's
    /// `ensure_global_data`, and the initializer folds once in
    /// `define_global` so calls never re-run it. Thread-locals share
    /// the path: `declare_global_with_name` marks the TLS flag.
    fn declare_block_static(&self, module: &mut mir::ModuleBuilder, id: fe::sema::DeclId) -> mir::Global {
        let binding = &self.sema.decls[id.0 as usize];
        // An asm label names the symbol outright, without the `DeclId`.
        let name = match (binding.asm_label, binding.name) {
            (Some(sym), _) => self.interner.resolve(sym).unwrap_or("<anon>").to_string(),
            (None, Some(sym)) => format!("{}.{}", self.interner.resolve(sym).unwrap_or("<anon>"), id.0),
            (None, None) => format!("<anon>.{}", id.0),
        };
        self.declare_global_with_name(module, id, &name, true)
    }

    /// A pure `extern` object declaration names storage defined
    /// elsewhere: declare the import so uses lower to a symbol. A
    /// thread-local import keeps its TLS flag so `GlobalAddr` goes
    /// through `tls_value` (`elf_gd` model).
    fn lower_data_extern(&self, module: &mut mir::ModuleBuilder, id: fe::sema::DeclId) -> mir::Global {
        let binding = &self.sema.decls[id.0 as usize];
        let name = self.symbol_name(binding, id);
        let mir_ty = lower_ty(&binding.ty, self.sema, self.interner);
        let gid = module.module.push_data_extern(&name, mir_ty);
        module.module.globals[gid].tls = binding.tls;
        // A weak declaration stays weak when undefined: the object's
        // symbol is `STB_WEAK` and the linker may resolve it to zero.
        module.module.globals[gid].weak = binding.weak;
        module.module.globals[gid].debug = Some(self.global_debug(module, id));
        gid
    }

    /// Startup/shutdown hooks (`__attribute__((constructor))`): one
    /// `.init_array` / `.fini_array` per TU collecting the defined
    /// functions' addresses in increasing priority, then declaration
    /// order (bare attributes are 65535). Local linkage: the sections
    /// merge across TUs while the symbols never collide. Mirrors the
    /// legacy backend's `emit_init_fini_arrays`.
    fn emit_init_fini_arrays(&self, module: &mut mir::ModuleBuilder, funcs: &HashMap<fe::sema::DeclId, mir::Func>) {
        let mut ctors: Vec<(u64, fe::sema::DeclId)> = Vec::new();
        let mut dtors: Vec<(u64, fe::sema::DeclId)> = Vec::new();
        for (i, binding) in self.sema.decls.iter().enumerate() {
            let id = fe::sema::DeclId(i as u32);
            if binding.kind != fe::sema::BindingKind::Func || !funcs.contains_key(&id) {
                continue;
            }
            if let Some(priority) = binding.constructor {
                ctors.push((priority, id));
            }
            if let Some(priority) = binding.destructor {
                dtors.push((priority, id));
            }
        }
        ctors.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.0.cmp(&b.1.0)));
        dtors.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.0.cmp(&b.1.0)));
        self.emit_hook_array(module, funcs, "__glcc_init_array", ".init_array", &ctors);
        self.emit_hook_array(module, funcs, "__glcc_fini_array", ".fini_array", &dtors);
    }

    /// One hook array: a pointer-sized slot per function holding its
    /// link-time address reloc, in a custom section the linker
    /// collects (`KEEP(*(SORT...))` in the default script, like GCC's
    /// own). LP64 assumes an 8-byte function pointer, as does the
    /// legacy backend.
    fn emit_hook_array(&self, module: &mut mir::ModuleBuilder, funcs: &HashMap<fe::sema::DeclId, mir::Func>, name: &str, section: &str, entries: &[(u64, fe::sema::DeclId)]) {
        if entries.is_empty() {
            return;
        }
        const PTR_SIZE: u64 = 8;
        let mut relocs = Vec::with_capacity(entries.len());
        for (i, (_, id)) in entries.iter().enumerate() {
            relocs.push((i as u64 * PTR_SIZE, mir::DataReloc::Func { target: funcs[id] }));
        }
        let ty = mir::Ty::Array {
            elem: Box::new(mir::Ty::ptr(mir::Ty::Void)),
            len: entries.len() as u64,
        };
        let init = mir::Const::Bytes(vec![0u8; entries.len() * PTR_SIZE as usize]);
        let gid = module.module.push_global(name, ty, Some(init));
        let global = &mut module.module.globals[gid];
        global.local = true;
        global.align = Some(PTR_SIZE);
        global.section = Some(section.to_string());
        global.relocs = relocs;
    }

    /// Shell storage for one defined object; `define_global` fills
    /// the type and image. Unsized arrays size from their initializer,
    /// so the shell type is provisional until the fold completes it.
    /// Linkage, weak, TLS, and alignment come from the binding's
    /// declared type; the alignment stays with the type's element
    /// when an unsized array completes from its initializer.
    fn declare_global_with_name(&self, module: &mut mir::ModuleBuilder, id: fe::sema::DeclId, name: &str, local: bool) -> mir::Global {
        let binding = &self.sema.decls[id.0 as usize];
        let mir_ty = lower_ty(&binding.ty, self.sema, self.interner);
        let gid = module.module.push_global(name, mir_ty, None);
        let global = &mut module.module.globals[gid];
        global.local = local;
        global.weak = binding.weak;
        global.tls = binding.tls;
        // An object's `aligned` attribute sets its alignment, raising
        // or lowering it (like GCC); the type decides without one.
        global.align = Some(
            binding
                .align
                .unwrap_or_else(|| self.sema.type_align(&binding.ty, self.interner).expect("defined object with an alignable type")),
        );
        if let Some(section) = binding.section {
            global.section = Some(self.interner.resolve(section).expect("section name symbol").to_string());
        }
        module.module.globals[gid].debug = Some(self.global_debug(module, id));
        gid
    }

    /// Fold one defined object's initializer into its shell: constant
    /// bytes inline, addresses as link-time relocations over the
    /// zeroed image (C's remainder rule, covering padding too).
    fn define_global(
        &self,
        module: &mut mir::ModuleBuilder,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        id: fe::sema::DeclId,
        gid: mir::Global,
    ) {
        let binding = &self.sema.decls[id.0 as usize];
        // Unsized arrays complete from their initializer: strings size
        // from the literal, lists from the element count.
        if matches!(binding.ty.unqualified(), fe::parse::Ty::Array { bound: None, .. }) {
            if let Some(init) = self.sema.inits.get(&id) {
                let init = init.clone();
                let (mir_ty, image, relocs) = self.sized_unsized(module, funcs, externs, globals, binding, &init);
                let shell = &mut module.module.globals[gid];
                shell.ty = mir_ty;
                shell.init = Some(mir::Const::Bytes(image));
                shell.relocs = relocs;
            }
            return;
        }
        let mut mir_ty = lower_ty(&binding.ty, self.sema, self.interner);
        let mut size = self.sema.type_size(&binding.ty, self.interner).expect("sized global type") as usize;
        // An initialized flexible array member grows the object past
        // `sizeof` (GCC extension): storage carries the larger size
        // while the declared type keeps its layout.
        if let Some(init) = self.sema.inits.get(&id) {
            let extent = self.init_extent(&binding.ty, init);
            if extent > size as u64 {
                if let mir::Ty::Struct { fields, offsets, .. } = &mir_ty {
                    mir_ty = mir::Ty::Struct {
                        fields: fields.clone(),
                        offsets: offsets.clone(),
                        size: extent,
                    };
                }
                size = extent as usize;
            }
        }
        let (init, relocs) = match self.sema.inits.get(&id) {
            None => (None, Vec::new()),
            Some(fe::parse::Initializer::Expr(e)) if !mir_ty.is_aggregate() => {
                match fe::eval::eval_static_init(e, self.sema, self.interner) {
                    // Pointer targets fold through the image path
                    // below: a null folds to zero bytes, an address to
                    // a reloc.
                    Some(folded) if !matches!(mir_ty, mir::Ty::Ptr(_)) => (Some(self.const_of_folded(&folded, &mir_ty)), Vec::new()),
                    _ => {
                        let mut image = vec![0u8; size];
                        let mut relocs = Vec::new();
                        let init = self.sema.inits.get(&id).unwrap().clone();
                        self.fold_global_bytes(module, &mut image, 0, &binding.ty, &mir_ty, &init, None, funcs, externs, globals, &mut relocs);
                        (Some(mir::Const::Bytes(image)), relocs)
                    }
                }
            }
            Some(init) => {
                let mut image = vec![0u8; size];
                let mut relocs = Vec::new();
                let init = init.clone();
                self.fold_global_bytes(module, &mut image, 0, &binding.ty, &mir_ty, &init, None, funcs, externs, globals, &mut relocs);
                (Some(mir::Const::Bytes(image)), relocs)
            }
        };
        let shell = &mut module.module.globals[gid];
        shell.ty = mir_ty;
        shell.init = init;
        // Later initializers overwrite earlier ones at the same offset
        // (`{ [0 ... 2] = f, [0] = g }` keeps `g`); keep the last reloc
        // per offset in first-seen order, matching the image fold.
        let mut seen = HashSet::new();
        let mut deduped: Vec<(u64, mir::DataReloc)> = Vec::with_capacity(relocs.len());
        for (off, reloc) in relocs.into_iter().rev() {
            if seen.insert(off) {
                deduped.push((off, reloc));
            }
        }
        deduped.reverse();
        shell.relocs = deduped;
    }

    /// Bytes an initializer actually covers. Usually the type's own
    /// size, but an initialized flexible array member extends the
    /// object past `sizeof` (GCC extension), so storage sizes itself
    /// from this rather than from the type alone.
    fn init_extent(&self, fe_ty: &fe::parse::Ty, init: &fe::parse::Initializer) -> u64 {
        self.designated_extent(fe_ty, &[], init)
    }

    /// Extent of an initializer under a designator prefix: the
    /// prefix walks to the selected member or element, then the
    /// initializer's own extent follows.
    fn designated_extent(&self, fe_ty: &fe::parse::Ty, designators: &[fe::parse::Designator], init: &fe::parse::Initializer) -> u64 {
        if let Some((first, rest)) = designators.split_first() {
            return match (fe_ty.unqualified(), first) {
                (fe::parse::Ty::Struct(id), fe::parse::Designator::Field { .. } | fe::parse::Designator::AnonField { .. }) => {
                    let def = &self.sema.structs[id.0 as usize];
                    let slot = designated_member(self.sema, *id, first) as usize;
                    let Some(member) = def.members.get(slot) else {
                        return 0;
                    };
                    // Overlaid union members start at zero.
                    let at = if def.is_union {
                        0
                    } else {
                        match self.sema.struct_layout(*id, self.interner) {
                            Some(layout) => layout.member_offsets[slot],
                            None => return 0,
                        }
                    };
                    at + self.designated_extent(&member.ty, rest, init)
                }
                (fe::parse::Ty::Array { elem, .. }, fe::parse::Designator::Index { index, .. }) => {
                    let Some(fe::eval::Value::Int(k)) = fe::eval::eval_const(index, self.sema, self.interner) else {
                        return 0;
                    };
                    let elem_size = self.sema.type_size(elem, self.interner).unwrap_or(0);
                    k.max(0) as u64 * elem_size + self.designated_extent(elem, rest, init).max(elem_size)
                }
                _ => 0,
            };
        }
        match init {
            fe::parse::Initializer::Expr(e) => {
                let fe::parse::Ty::Array { bound: None, .. } = fe_ty.unqualified() else {
                    return 0;
                };
                // A flexible member sized from a string literal.
                match Self::peek_global_str(e) {
                    Some(parts) => fe::eval::string_bytes_in(parts, self.interner, self.sema.target.wchar())
                        .map(|bytes| bytes.len() as u64)
                        .unwrap_or(0),
                    None => 0,
                }
            }
            fe::parse::Initializer::List { items, .. } => {
                let mut end = 0u64;
                let mut cursor = 0usize;
                for (i, item) in items.iter().enumerate() {
                    let (at, elem_ty, stride) = match fe_ty.unqualified() {
                        fe::parse::Ty::Array { elem, bound: None, .. } => {
                            let elem_size = self.sema.type_size(elem, self.interner).unwrap_or(0);
                            let at = if item.designators.is_empty() { i as u64 * elem_size } else { 0 };
                            (at, elem.as_ref(), elem_size)
                        }
                        fe::parse::Ty::Struct(id) => {
                            let def = &self.sema.structs[id.0 as usize];
                            if def.is_union {
                                return 0;
                            }
                            let Some(layout) = self.sema.struct_layout(*id, self.interner) else {
                                return 0;
                            };
                            let slot = match item.designators.first() {
                                Some(des @ (fe::parse::Designator::Field { .. } | fe::parse::Designator::AnonField { .. })) => designated_member(self.sema, *id, des) as usize,
                                Some(fe::parse::Designator::Index { .. }) => cursor,
                                None => {
                                    let slot = cursor;
                                    cursor += 1;
                                    slot
                                }
                            };
                            let Some(member) = def.members.get(slot) else {
                                continue;
                            };
                            (layout.member_offsets[slot], &member.ty, 0)
                        }
                        _ => return 0,
                    };
                    // Every array element occupies its own size; the
                    // last one may extend further (a flexible member
                    // of a struct element).
                    let sub = self.designated_extent(elem_ty, item.designators.get(1..).unwrap_or(&[]), &item.init).max(stride);
                    end = end.max(at + sub);
                }
                end
            }
        }
    }

    /// Complete an unsized array from a string or list initializer.
    /// List elements may hold address constants, folded to relocations
    /// through the shared image walk.
    #[allow(clippy::too_many_arguments)]
    fn sized_unsized(
        &self,
        module: &mut mir::ModuleBuilder,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        binding: &fe::sema::Binding,
        init: &fe::parse::Initializer,
    ) -> (mir::Ty, Vec<u8>, Vec<(u64, mir::DataReloc)>) {
        let fe::parse::Ty::Array { elem, .. } = binding.ty.unqualified() else {
            panic!("unsized non-array");
        };
        let elem_mir = lower_ty(elem, self.sema, self.interner);
        match init {
            fe::parse::Initializer::Expr(e) => {
                let fe::check::CheckedExprKind::Str(parts) = &e.kind else {
                    panic!("unsized array needs a literal initializer");
                };
                let mut image = fe::eval::string_bytes_in(parts, self.interner, self.sema.target.wchar()).expect("checked string folds to bytes");
                let width = elem_mir.size_bytes();
                let len = image.len() as u64 / width + 1;
                image.resize(len as usize * width as usize, 0);
                (mir::Ty::Array { elem: Box::new(elem_mir), len }, image, Vec::new())
            }
            fe::parse::Initializer::List { items, .. } => {
                let mut len = 0u64;
                let mut cursor = 0u64;
                for item in items {
                    match item.designators.first() {
                        Some(fe::parse::Designator::Index { index, .. }) => match fe::eval::eval_const(index, self.sema, self.interner) {
                            Some(fe::eval::Value::Int(k)) => {
                                len = len.max((k as u64) + 1);
                                cursor = (k as u64) + 1;
                            }
                            _ => panic!("non-constant array designator"),
                        },
                        Some(_) => panic!("field designator into array"),
                        None => {
                            cursor += 1;
                            len = len.max(cursor);
                        }
                    }
                }
                let mir_ty = mir::Ty::Array { elem: Box::new(elem_mir), len };
                let elem_size = self.sema.type_size(elem, self.interner).expect("sized array element") as usize;
                let mut image = vec![0u8; len as usize * elem_size];
                let mut relocs = Vec::new();
                self.fold_global_bytes(module, &mut image, 0, &binding.ty, &mir_ty, init, None, funcs, externs, globals, &mut relocs);
                (mir_ty, image, relocs)
            }
        }
    }

    /// Fold a constant initializer into a global byte image: constant
    /// bytes inline, address constants as link-time relocations over
    /// the zeroed image (C's remainder rule is the zeroed buffer this
    /// patches, covering padding too).
    #[allow(clippy::too_many_arguments)]
    fn fold_global_bytes(
        &self,
        module: &mut mir::ModuleBuilder,
        image: &mut [u8],
        base: u64,
        fe_ty: &fe::parse::Ty,
        mir_ty: &mir::Ty,
        init: &fe::parse::Initializer,
        bf: Option<BitfieldTarget>,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        relocs: &mut Vec<(u64, mir::DataReloc)>,
    ) {
        match init {
            fe::parse::Initializer::Expr(e) => {
                // Bit-field targets read-modify-write the covering
                // unit bytes; only constant integers occur here (C
                // requires constant static initializers).
                if let Some(t) = bf {
                    let Some(fe::eval::Value::Int(n)) = fe::eval::eval_static_init(e, self.sema, self.interner) else {
                        panic!("constant bit-field initializer");
                    };
                    let len = lower_ty(&t.unit_ty, self.sema, self.interner).size_bytes() as usize;
                    let unit_bits = len as u32 * 8;
                    let mask: u128 = if t.width >= unit_bits { u128::MAX } else { (1u128 << t.width) - 1 };
                    let mut buf = [0u8; 16];
                    buf[..len].copy_from_slice(&image[base as usize..base as usize + len]);
                    let mut unit = u128::from_le_bytes(buf);
                    unit = (unit & !(mask << t.bit)) | (((n as u128) & mask) << t.bit);
                    buf = unit.to_le_bytes();
                    image[base as usize..base as usize + len].copy_from_slice(&buf[..len]);
                    return;
                }
                // String into a character array: raw bytes, NUL-padded
                // by the zeroed image. Conversions around the literal
                // are transparent.
                if let Some(parts) = Self::peek_global_str(e) {
                    if let fe::parse::Ty::Array { elem, bound, .. } = fe_ty.unqualified() {
                        if let Some(lit_elem) = fe::eval::string_elem_ty_in(parts, self.sema.target.wchar()) {
                            let elem_mir = lower_ty(elem, self.sema, self.interner);
                            let lit_mir = lower_ty(&lit_elem, self.sema, self.interner);
                            if elem_mir == lit_mir {
                                let width = elem_mir.size_bytes();
                                let content = fe::eval::string_bytes_in(parts, self.interner, self.sema.target.wchar()).expect("checked string folds to bytes");
                                let room = match bound {
                                    Some(fe::parse::ArrayBound::Fixed(n)) => n.saturating_mul(width),
                                    _ => content.len() as u64 + width,
                                };
                                let end = base.saturating_add(room.min(content.len() as u64));
                                let room_in_image = (image.len() as u64).saturating_sub(base);
                                let end = end.min(base.saturating_add(room_in_image));
                                image[base as usize..end as usize].copy_from_slice(&content[..(end - base) as usize]);
                                return;
                            }
                        }
                    }
                    if matches!(fe_ty.unqualified(), fe::parse::Ty::Pointer { .. }) {
                        let (target, ..) = self.materialize_string_global(module, parts);
                        relocs.push((base, mir::DataReloc::Global { target, addend: 0 }));
                        return;
                    }
                }
                // A compound literal in a static initializer (C17
                // §6.5.2.5p5: static storage duration at file scope).
                // As a value of object type its bytes patch inline; as
                // a pointer its static object materializes anonymously
                // and the address records as a reloc.
                if let fe::check::CheckedExprKind::CompoundLiteral { ty: lit_ty, init: lit_init } = &e.kind {
                    if matches!(fe_ty.unqualified(), fe::parse::Ty::Pointer { .. }) {
                        let target = self.materialize_global_compound(module, funcs, externs, globals, lit_ty, lit_init);
                        relocs.push((base, mir::DataReloc::Global { target, addend: 0 }));
                    } else {
                        self.fold_global_bytes(
                            module,
                            image,
                            base,
                            lit_ty,
                            &lower_ty(lit_ty, self.sema, self.interner),
                            lit_init,
                            None,
                            funcs,
                            externs,
                            globals,
                            relocs,
                        );
                    }
                    return;
                }
                // A label address (or arithmetic on one) in a static
                // initializer (`static void *tab[] = {&&a, &&b}`,
                // `static int d[] = {&&b - &&a}`): the object holds the
                // ids `&&label` computes, so `goto *tab[i]` dispatches
                // and differences stay consecutive.
                if let Some(n) = self.fold_label_const(e) {
                    let len = mir_ty.size_bytes() as usize;
                    image[base as usize..base as usize + len].copy_from_slice(&n.to_le_bytes()[..len]);
                    return;
                }
                match fe::eval::eval_static_init(e, self.sema, self.interner) {
                    Some(fe::eval::Value::Int(n)) => {
                        // Through conversions this covers enums, `_Bool`,
                        // chars, and null pointers alike. Sizes come from
                        // the target type, so pointer-width zeros land
                        // for nulls.
                        let len = mir_ty.size_bytes() as usize;
                        image[base as usize..base as usize + len].copy_from_slice(&n.to_le_bytes()[..len]);
                        return;
                    }
                    Some(fe::eval::Value::Float(f)) => {
                        let bits = float_bits(f, mir_ty);
                        let len = mir_ty.size_bytes() as usize;
                        image[base as usize..base as usize + len].copy_from_slice(&bits.to_le_bytes()[..len]);
                        return;
                    }
                    Some(fe::eval::Value::Complex(re, im)) => {
                        // A folded complex initializer (`= 1.0 + 2.0i`):
                        // the parts patch inline at the component
                        // offsets, real first.
                        if !is_complex(fe_ty) {
                            panic!("complex constant for a non-complex object");
                        }
                        let comp_fe = complex_component(fe_ty);
                        let comp_size = self.sema.type_size(&comp_fe, self.interner).expect("complex component has a size");
                        self.write_complex_part(image, base, &comp_fe, &re);
                        self.write_complex_part(image, base + comp_size, &comp_fe, &im);
                        return;
                    }
                    None => {}
                }
                // A null pointer constant spelled as a cast (`(T *)0`,
                // and `NULL` from real headers): fold the operand and
                // store the (zero) address bytes.
                if matches!(fe_ty.unqualified(), fe::parse::Ty::Pointer { .. }) {
                    if let fe::check::CheckedExprKind::Cast { expr, .. } | fe::check::CheckedExprKind::Convert { expr, .. } = &e.kind {
                        if let Some(fe::eval::Value::Int(n)) = fe::eval::eval_const(expr, self.sema, self.interner) {
                            let len = mir_ty.size_bytes() as usize;
                            image[base as usize..base as usize + len].copy_from_slice(&n.to_le_bytes()[..len]);
                            return;
                        }
                    }
                }
                // Addresses, pointer arithmetic, and (through
                // conversions) string literals for pointer targets.
                if matches!(fe_ty.unqualified(), fe::parse::Ty::Pointer { .. }) {
                    self.reloc_global_addr(module, funcs, externs, globals, relocs, base, e);
                    return;
                }
                // An address constant held in an integer (`(long)&obj`,
                // at pointer width): a link-time value like the pointer
                // path. Only at pointer width, which never truncates.
                if matches!(mir_ty, mir::Ty::Int { bits: 64 }) && self.is_global_address(e) {
                    self.reloc_global_addr(module, funcs, externs, globals, relocs, base, e);
                    return;
                }
                panic!("non-constant global initializer");
            }
            fe::parse::Initializer::List { items, .. } => match (fe_ty.unqualified(), mir_ty) {
                // A flexible array member holds no elements in its
                // type (its MIR form is a pointer): initializing one
                // extends the object, so elements fold by element
                // stride past the struct.
                (fe::parse::Ty::Array { elem, bound: None, .. }, mir::Ty::Ptr(_)) => {
                    let elem_mir = lower_ty(elem, self.sema, self.interner);
                    let elem_size = elem_mir.size_bytes();
                    for (i, item) in items.iter().enumerate() {
                        if !item.designators.is_empty() {
                            panic!("designator into flexible array member");
                        }
                        self.fold_global_bytes(
                            module,
                            image,
                            base + i as u64 * elem_size,
                            elem.as_ref(),
                            &elem_mir,
                            &item.init,
                            None,
                            funcs,
                            externs,
                            globals,
                            relocs,
                        );
                    }
                }
                (fe::parse::Ty::Array { .. }, mir::Ty::Array { .. }) => {
                    let mut cursor = 0u64;
                    for item in items {
                        let (off, elem_fe, elem_mir, next, bf) = self.global_designated(base, fe_ty, mir_ty, &item.designators, cursor);
                        cursor = next;
                        self.fold_global_bytes(module, image, off, &elem_fe, &elem_mir, &item.init, bf, funcs, externs, globals, relocs);
                    }
                }
                (fe::parse::Ty::Struct(_), mir::Ty::Struct { .. }) => {
                    let mut cursor = 0u64;
                    for item in items {
                        let (off, elem_fe, elem_mir, next, bf) = self.global_designated(base, fe_ty, mir_ty, &item.designators, cursor);
                        cursor = next;
                        self.fold_global_bytes(module, image, off, &elem_fe, &elem_mir, &item.init, bf, funcs, externs, globals, relocs);
                    }
                }
                // Unions overlay at offset zero; the blob holds bytes.
                (fe::parse::Ty::Struct(id), mir::Ty::Array { .. }) if self.sema.structs[id.0 as usize].is_union => {
                    let mut cursor = 0u64;
                    for item in items {
                        let (off, elem_fe, elem_mir, next, bf) = self.global_designated(base, fe_ty, mir_ty, &item.designators, cursor);
                        cursor = next;
                        self.fold_global_bytes(module, image, off, &elem_fe, &elem_mir, &item.init, bf, funcs, externs, globals, relocs);
                    }
                }
                _ if items.len() == 1 && items[0].designators.is_empty() => {
                    self.fold_global_bytes(module, image, base, fe_ty, mir_ty, &items[0].init, bf, funcs, externs, globals, relocs);
                }
                _ => panic!("braced list for scalar type"),
            },
        }
    }

    /// See through assignment-inserted conversions to a string literal.
    fn peek_global_str(e: &fe::check::CheckedExpr) -> Option<&[fe::lexer::StringLiteral]> {
        let mut cur = e;
        loop {
            match &cur.kind {
                fe::check::CheckedExprKind::Str(parts) => return Some(parts),
                fe::check::CheckedExprKind::Cast { expr, .. } | fe::check::CheckedExprKind::Convert { expr, .. } => cur = expr,
                _ => return None,
            }
        }
    }

    /// Fold a static-initializer constant built from label addresses
    /// (`&&label`, and arithmetic mixing them with integers): ids stand
    /// in for addresses, and pointer steps scale by the pointee size
    /// exactly like the runtime arithmetic path, so `&&b - &&a` counts
    /// labels like GCC's byte offsets count jumps. Plain integer
    /// constants fold here too.
    fn fold_label_const(&self, e: &fe::check::CheckedExpr) -> Option<i128> {
        if let Some(fe::eval::Value::Int(n)) = fe::eval::eval_const(e, self.sema, self.interner) {
            return Some(n);
        }
        match &e.kind {
            fe::check::CheckedExprKind::LabelAddr(name) => self.sema.label_ids.get(name).map(|id| *id as i128),
            fe::check::CheckedExprKind::Cast { expr, .. } | fe::check::CheckedExprKind::Convert { expr, .. } => self.fold_label_const(expr),
            fe::check::CheckedExprKind::Binary(op, lhs, rhs) => {
                let l = self.fold_label_const(lhs)?;
                let r = self.fold_label_const(rhs)?;
                let lhs_ptr = matches!(lhs.ty.unqualified(), fe::parse::Ty::Pointer { .. });
                let rhs_ptr = matches!(rhs.ty.unqualified(), fe::parse::Ty::Pointer { .. });
                match op {
                    fe::parse::BinaryOp::Add if lhs_ptr => Some(l + r * self.pointee_step(&lhs.ty)?),
                    fe::parse::BinaryOp::Add if rhs_ptr => Some(r + l * self.pointee_step(&rhs.ty)?),
                    fe::parse::BinaryOp::Add => Some(l + r),
                    // Pointer difference scales down; pointer minus
                    // integer scales up; integer minus integer is plain.
                    fe::parse::BinaryOp::Sub if lhs_ptr && rhs_ptr => Some((l - r) / self.pointee_step(&lhs.ty)?),
                    fe::parse::BinaryOp::Sub if lhs_ptr => Some(l - r * self.pointee_step(&lhs.ty)?),
                    fe::parse::BinaryOp::Sub => Some(l - r),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// The byte step of pointer arithmetic: the pointee size, with GNU
    /// `void`/function pointees stepping one byte.
    fn pointee_step(&self, ptr_ty: &fe::parse::Ty) -> Option<i128> {
        let fe::parse::Ty::Pointer { pointee, .. } = ptr_ty.unqualified() else {
            return None;
        };
        Some(match pointee.unqualified() {
            fe::parse::Ty::Void | fe::parse::Ty::Func { .. } => 1,
            _ => self.sema.type_size(pointee, self.interner)? as i128,
        })
    }

    /// Emit a file-scope compound literal (`(T){...}`) into a fresh
    /// anonymous object, returning its global for an address reloc.
    /// The object has static storage duration, so its address is an
    /// address constant wherever a global initializer needs one.
    #[allow(clippy::too_many_arguments)]
    fn materialize_global_compound(
        &self,
        module: &mut mir::ModuleBuilder,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        ty: &fe::parse::Ty,
        init: &fe::parse::Initializer,
    ) -> mir::Global {
        let size = self.sema.type_size(ty, self.interner).expect("compound literal of complete type") as usize;
        let mir_ty = lower_ty(ty, self.sema, self.interner);
        let mut image = vec![0u8; size];
        let mut relocs = Vec::new();
        self.fold_global_bytes(module, &mut image, 0, ty, &mir_ty, init, None, funcs, externs, globals, &mut relocs);
        let n = self.str_count.get();
        self.str_count.set(n + 1);
        let gid = module.module.push_global(&format!("__compound.{n}"), mir_ty, Some(mir::Const::Bytes(image)));
        let global = &mut module.module.globals[gid];
        global.local = true;
        global.align = Some(self.sema.type_align(ty, self.interner).expect("compound literal of complete type"));
        global.relocs = relocs;
        gid
    }

    /// Record one pointer address reloc at `off`: `&var`, `&arr[i]`,
    /// decayed arrays and functions, string literals, and `&x ± c`.
    /// Conversions around the address are transparent. A null pointer
    /// constant is just zero bytes, and the image is pre-zeroed.
    #[allow(clippy::too_many_arguments)]
    fn reloc_global_addr(
        &self,
        module: &mut mir::ModuleBuilder,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        relocs: &mut Vec<(u64, mir::DataReloc)>,
        off: u64,
        e: &fe::check::CheckedExpr,
    ) {
        if matches!(fe::eval::eval_const(e, self.sema, self.interner), Some(fe::eval::Value::Int(0))) {
            return;
        }
        match &e.kind {
            fe::check::CheckedExprKind::Cast { expr, .. } | fe::check::CheckedExprKind::Convert { expr, .. } => {
                self.reloc_global_addr(module, funcs, externs, globals, relocs, off, expr);
            }
            fe::check::CheckedExprKind::Str(parts) => {
                let (target, ..) = self.materialize_string_global(module, parts);
                relocs.push((off, mir::DataReloc::Global { target, addend: 0 }));
            }
            fe::check::CheckedExprKind::Unary(op, inner) => {
                assert!(matches!(op, fe::parse::UnaryOp::AddrOf), "only address constants reach relocs");
                self.reloc_global_pointed(module, funcs, externs, globals, relocs, off, inner, 0);
            }
            fe::check::CheckedExprKind::Ident(id) => {
                let binding = &self.sema.decls[id.0 as usize];
                match binding.kind {
                    fe::sema::BindingKind::Func => {
                        if let Some(target) = funcs.get(id) {
                            relocs.push((off, mir::DataReloc::Func { target: *target }));
                        } else if let Some(target) = externs.get(id) {
                            relocs.push((off, mir::DataReloc::FuncExtern { target: *target }));
                        } else {
                            panic!("address of a function before its declaration");
                        }
                    }
                    _ => {
                        // Decayed array (or its `&`): the object's base.
                        let Some(target) = globals.get(id) else {
                            panic!("address of undefined storage");
                        };
                        relocs.push((off, mir::DataReloc::Global { target: *target, addend: 0 }));
                    }
                }
            }
            fe::check::CheckedExprKind::Binary(op, lhs, rhs) => {
                // `&x + c` / `&x - c` (and decayed equivalents): one
                // side is an address, the other a folded integer.
                let (addr_side, const_side, neg) = match op {
                    fe::parse::BinaryOp::Add => (lhs.as_ref(), rhs.as_ref(), false),
                    fe::parse::BinaryOp::Sub => (lhs.as_ref(), rhs.as_ref(), true),
                    _ => panic!("non-constant global initializer"),
                };
                let c = match fe::eval::eval_const(const_side, self.sema, self.interner) {
                    Some(fe::eval::Value::Int(n)) => n,
                    _ => panic!("non-constant global initializer"),
                };
                let elem = match addr_side.ty.unqualified() {
                    fe::parse::Ty::Pointer { pointee, .. } => self.sema.type_size(pointee, self.interner).unwrap() as i128,
                    _ => 1,
                };
                let addend = if neg { -c * elem } else { c * elem };
                // The address side roots at a declared object, a string
                // literal, or a compound literal; casts and conversions
                // are looked through there.
                self.reloc_global_pointed(module, funcs, externs, globals, relocs, off, addr_side, addend as i64);
            }
            _ => panic!("non-constant global initializer"),
        }
    }

    /// The pointer *value* of a constant pointer expression: the root
    /// object plus a byte displacement (`&lvalue`, decayed arrays and
    /// functions, `p ± k`, `p->m`, `p[i]`). A scalar variable's value
    /// is not an address constant and panics like the other shapes.
    fn global_ptr_root<'a>(&self, expr: &'a fe::check::CheckedExpr) -> (AddrRoot<'a>, i64) {
        match &expr.kind {
            fe::check::CheckedExprKind::Ident(id) => {
                let binding = &self.sema.decls[id.0 as usize];
                let decays = matches!(binding.kind, fe::sema::BindingKind::Func) || matches!(binding.ty.unqualified(), fe::parse::Ty::Array { .. });
                assert!(decays, "non-constant global initializer");
                (AddrRoot::Decl(*id), 0)
            }
            fe::check::CheckedExprKind::Str(parts) => (AddrRoot::Str(parts), 0),
            fe::check::CheckedExprKind::CompoundLiteral { ty, init } => (AddrRoot::Compound(ty, init), 0),
            fe::check::CheckedExprKind::Cast { expr, .. } | fe::check::CheckedExprKind::Convert { expr, .. } => self.global_ptr_root(expr),
            fe::check::CheckedExprKind::Unary(op, inner) if matches!(op, fe::parse::UnaryOp::AddrOf) => self.global_pointed_root(inner, 0),
            fe::check::CheckedExprKind::Binary(op, lhs, rhs) if matches!(op, fe::parse::BinaryOp::Add | fe::parse::BinaryOp::Sub) => {
                let neg = matches!(op, fe::parse::BinaryOp::Sub);
                let (ptr, int) = if matches!(lhs.ty.unqualified(), fe::parse::Ty::Pointer { .. }) {
                    (lhs.as_ref(), rhs.as_ref())
                } else {
                    (rhs.as_ref(), lhs.as_ref())
                };
                let k = match fe::eval::eval_const(int, self.sema, self.interner) {
                    Some(fe::eval::Value::Int(n)) => n,
                    _ => panic!("non-constant global initializer"),
                };
                let stride = self.ptr_stride(&ptr.ty);
                let (root, disp) = self.global_ptr_root(ptr);
                let addend = if neg { -k * stride } else { k * stride };
                (root, disp + addend as i64)
            }
            fe::check::CheckedExprKind::Member { base, offset, arrow: true, .. } => {
                let (root, disp) = self.global_ptr_root(base);
                (root, disp + *offset as i64)
            }
            fe::check::CheckedExprKind::Subscript { base, index } => {
                let i = match fe::eval::eval_const(index, self.sema, self.interner) {
                    Some(fe::eval::Value::Int(n)) => n,
                    _ => panic!("non-constant global initializer"),
                };
                let stride = self.ptr_stride(&base.ty);
                let (root, disp) = self.global_ptr_root(base);
                (root, disp + (i * stride) as i64)
            }
            _ => panic!("non-constant global initializer"),
        }
    }

    /// The stride of pointer arithmetic on an array or pointer type.
    fn ptr_stride(&self, ty: &fe::parse::Ty) -> i128 {
        let elem = match ty.unqualified() {
            fe::parse::Ty::Array { elem, .. } => elem.as_ref(),
            fe::parse::Ty::Pointer { pointee, .. } => pointee.as_ref(),
            _ => panic!("non-constant global initializer"),
        };
        self.sema.type_size(elem, self.interner).unwrap() as i128
    }

    /// The object a `&...` points at: a bare global, `&arr[i]`
    /// (multi-dimensional too), or `&s.member` (members nest; unions
    /// sit at zero). Strings and compound literals materialize their
    /// own objects.
    #[allow(clippy::too_many_arguments)]
    fn reloc_global_pointed(
        &self,
        module: &mut mir::ModuleBuilder,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        relocs: &mut Vec<(u64, mir::DataReloc)>,
        off: u64,
        pointed: &fe::check::CheckedExpr,
        extra: i64,
    ) {
        let (root, disp) = self.global_pointed_root(pointed, extra);
        match root {
            AddrRoot::Str(parts) => {
                let (target, ..) = self.materialize_string_global(module, parts);
                relocs.push((off, mir::DataReloc::Global { target, addend: disp }));
            }
            AddrRoot::Compound(lit_ty, lit_init) => {
                let target = self.materialize_global_compound(module, funcs, externs, globals, lit_ty, lit_init);
                relocs.push((off, mir::DataReloc::Global { target, addend: disp }));
            }
            AddrRoot::Decl(id) => {
                let binding = &self.sema.decls[id.0 as usize];
                if matches!(binding.kind, fe::sema::BindingKind::Func) {
                    assert!(disp == 0, "arithmetic on a function address");
                    if let Some(target) = funcs.get(&id) {
                        relocs.push((off, mir::DataReloc::Func { target: *target }));
                    } else if let Some(target) = externs.get(&id) {
                        relocs.push((off, mir::DataReloc::FuncExtern { target: *target }));
                    } else {
                        panic!("address of a function before its declaration");
                    }
                    return;
                }
                let Some(target) = globals.get(&id) else {
                    panic!("address of undefined storage");
                };
                relocs.push((off, mir::DataReloc::Global { target: *target, addend: disp }));
            }
        }
    }

    /// Root object and byte displacement of a `&...` operand: nested
    /// members add each layer's own total `offset`, subscripts stride
    /// by their level's element size, conversions look through.
    /// Anything else is not an address constant.
    fn global_pointed_root<'a>(&self, pointed: &'a fe::check::CheckedExpr, extra: i64) -> (AddrRoot<'a>, i64) {
        match &pointed.kind {
            fe::check::CheckedExprKind::Ident(id) => (AddrRoot::Decl(*id), extra),
            fe::check::CheckedExprKind::Str(parts) => (AddrRoot::Str(parts), extra),
            fe::check::CheckedExprKind::CompoundLiteral { ty, init } => (AddrRoot::Compound(ty, init), extra),
            fe::check::CheckedExprKind::Cast { expr, .. } | fe::check::CheckedExprKind::Convert { expr, .. } => self.global_pointed_root(expr, extra),
            fe::check::CheckedExprKind::Unary(op, inner) if matches!(op, fe::parse::UnaryOp::AddrOf) => self.global_pointed_root(inner, extra),
            fe::check::CheckedExprKind::Member { base, offset, arrow, .. } => {
                // `&(p->m)`: the base is a pointer value, so walk it as
                // one; a plain `s.m` recurses into the object.
                let (root, inner) = if *arrow {
                    let (root, inner) = self.global_ptr_root(base);
                    (root, inner + extra)
                } else {
                    self.global_pointed_root(base, extra)
                };
                (root, inner + *offset as i64)
            }
            fe::check::CheckedExprKind::Subscript { base, index } => {
                // `&arr[i]`: the address is constant only for a static
                // array object. The base often arrives already decayed
                // to a pointer; the element then comes from the
                // pointee. Multi-dimensional arrays recurse: each level
                // strides by its own element.
                let i = match fe::eval::eval_const(index, self.sema, self.interner) {
                    Some(fe::eval::Value::Int(n)) => n,
                    _ => panic!("non-constant global initializer"),
                };
                let stride = self.ptr_stride(&base.ty);
                let (root, inner) = match base.ty.unqualified() {
                    fe::parse::Ty::Array { .. } => self.global_pointed_root(base, extra),
                    _ => {
                        let (root, inner) = self.global_ptr_root(base);
                        (root, inner + extra)
                    }
                };
                (root, inner + (i * stride) as i64)
            }
            _ => panic!("non-constant global initializer"),
        }
    }

    /// Whether `e` is an address constant (through conversions): the
    /// shapes `reloc_global_addr` accepts. Scalar idents are excluded
    /// (a variable's *value* is not constant); functions and decayed
    /// arrays are addresses. Keep in sync with `reloc_global_addr`.
    fn is_global_address(&self, e: &fe::check::CheckedExpr) -> bool {
        match &e.kind {
            fe::check::CheckedExprKind::Cast { expr, .. } | fe::check::CheckedExprKind::Convert { expr, .. } => self.is_global_address(expr),
            fe::check::CheckedExprKind::Unary(op, _) => {
                matches!(op, fe::parse::UnaryOp::AddrOf)
            }
            fe::check::CheckedExprKind::Ident(id) => {
                let binding = &self.sema.decls[id.0 as usize];
                matches!(binding.kind, fe::sema::BindingKind::Func) || matches!(binding.ty.unqualified(), fe::parse::Ty::Array { .. })
            }
            fe::check::CheckedExprKind::Binary(op, _, _) => {
                matches!(op, fe::parse::BinaryOp::Add | fe::parse::BinaryOp::Sub)
            }
            fe::check::CheckedExprKind::Subscript { .. } | fe::check::CheckedExprKind::Member { .. } | fe::check::CheckedExprKind::Str(_) => true,
            fe::check::CheckedExprKind::CompoundLiteral { ty, .. } => {
                matches!(ty.unqualified(), fe::parse::Ty::Array { .. })
            }
            _ => false,
        }
    }

    fn const_of_folded(&self, folded: &fe::eval::Value, mir_ty: &mir::Ty) -> mir::Const {
        match folded {
            fe::eval::Value::Int(n) => mir::Const::Int {
                bits: int_bits(mir_ty),
                value: *n,
            },
            fe::eval::Value::Float(f) => mir::Const::Float {
                kind: float_kind(mir_ty),
                bits: float_bits(*f, mir_ty),
            },
            // Complex constants never reach scalar consts: literals
            // materialize through slots, and globals fold through the
            // byte image above.
            fe::eval::Value::Complex(..) => {
                unreachable!("complex constants materialize through slots, never scalar consts")
            }
        }
    }

    /// Patch one folded complex part into a global image at `off`:
    /// integers as little-endian bytes, floats as their target bits.
    fn write_complex_part(&self, image: &mut [u8], off: u64, component: &fe::parse::Ty, v: &fe::eval::Value) {
        let comp_mir = lower_ty(component, self.sema, self.interner);
        let len = comp_mir.size_bytes() as usize;
        match v {
            fe::eval::Value::Int(n) => {
                image[off as usize..off as usize + len].copy_from_slice(&n.to_le_bytes()[..len]);
            }
            fe::eval::Value::Float(f) => {
                let bits = float_bits(*f, &comp_mir);
                image[off as usize..off as usize + len].copy_from_slice(&bits.to_le_bytes()[..len]);
            }
            _ => panic!("complex part does not match its component type"),
        }
    }

    fn lower_function(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        id: fe::sema::DeclId,
    ) {
        let body = self.sema.bodies[&id].block.clone();
        let binding = &self.sema.decls[id.0 as usize];
        let entry = func.entry;
        frame.current = Some(entry);
        // Entry params are typed by the signature — the ABI's view —
        // while the body's binding may be narrower (K&R declarations
        // pass promoted values); the spill converts.
        let fe::parse::Ty::Func { params: sig_params, .. } = binding.ty.unqualified() else {
            panic!("function definition without a Func type");
        };
        // Entry block params correspond positionally to signature params.
        for (i, param_id) in self.sema.bodies[&id].params.iter().enumerate() {
            let param_binding = &self.sema.decls[param_id.0 as usize];
            if let Some(loc) = self.debug_location(module, param_binding.span) {
                frame.loc = Some(loc);
            }
            let sig_fe = sig_params.get(i);
            let param_ty = match sig_fe {
                Some(ty) => lower_ty(ty, self.sema, self.interner),
                None => lower_ty(&param_binding.ty, self.sema, self.interner),
            };
            let param = func.values.push(mir::ValueData {
                ty: param_ty,
                def: mir::Def::Param(entry, i as u32),
            });
            func.blocks[entry].params.push(param);
            // A `va_list` parameter arrives as a pointer to the
            // caller's tag object (the array-typed parameter decayed).
            // Alias it, don't spill the pointer into a slot (which
            // `va_arg` would misread as cursors): the tag itself is
            // the caller's object, so cursors stay shared as C
            // requires. Mirrors the legacy backend. MSVC's `va_list`
            // is a plain pointer: the parameter is the cursor itself,
            // spilled like any scalar so `va_arg` advances the copy.
            if self.va_list_is_array() && matches!(param_binding.ty.unqualified(), fe::parse::Ty::BuiltinVaList) {
                frame.slots.insert(*param_id, param);
                continue;
            }
            // -O0 frame: every parameter is spilled to a slot on entry.
            // Volatility rides the spill: a `volatile` parameter's
            // slot stays volatile for every later load.
            let body_ty = lower_ty(&param_binding.ty, self.sema, self.interner);
            let slot = self.alloca(func, frame, body_ty);
            let vol = ty_is_volatile(&param_binding.ty);
            let value = match sig_fe {
                Some(from) => self.convert(func, frame, param, from, &param_binding.ty),
                None => param,
            };
            self.store(func, frame, slot, value, vol);
            frame.slots.insert(*param_id, slot);
        }
        self.lower_compound(module, func, frame, funcs, externs, globals, &body);
        // Falling off the end: void returns, `main` returns 0 per C23,
        // anything else returns an undefined value. Returning garbage
        // like GCC does (rather than trapping) keeps legacy code that
        // ignores the result working; the value is UB either way.
        // An async body returns its *declared* result type (the coroutine
        // pass turns these returns into result-slot stores).
        let result_ty = func.coroutine.clone().unwrap_or_else(|| func.sig.ret.clone());
        if frame.current.is_some() {
            if result_ty.is_void() {
                self.terminate(func, frame, mir::Terminator::Return { val: None });
            } else if binding.name.is_some_and(|n| self.interner.resolve(n) == Some("main")) {
                let zero = self.iconst(func, 32, 0);
                self.terminate(func, frame, mir::Terminator::Return { val: Some(zero) });
            } else {
                let undef = func.values.push(mir::ValueData {
                    ty: result_ty,
                    def: mir::Def::Const(mir::Const::Undef),
                });
                self.terminate(func, frame, mir::Terminator::Return { val: Some(undef) });
            }
        }
        // Computed gotos dispatch here: one switch over the function's
        // labels, keyed by the ids `&&label` computes. Filled after the
        // body so every label block exists; labels whose address was
        // never taken get cases too (harmless, and it keeps label
        // addresses in static initializers working, since those fold
        // before the body is lowered).
        self.drain_allocas(module, func, frame);
        if let Some((dispatch, param)) = frame.computed_goto.take() {
            let dead = func.push_block(vec![]);
            func.blocks[dead].term = Some(mir::Terminator::Unreachable);
            let mut cases: Vec<(i128, mir::Block)> = Vec::new();
            for (name, target) in &frame.labels {
                if let Some(id) = self.sema.label_ids.get(name) {
                    cases.push((*id as i128, *target));
                }
            }
            cases.sort_by_key(|(id, _)| *id);
            func.blocks[dispatch].term = Some(mir::Terminator::Switch {
                scrutinee: param,
                cases,
                default: dead,
            });
        }
        frame.loc = self.debug_location(module, binding.span);
        func.debug = Some(self.build_function_debug(module, &frame.slots, &self.sema.bodies[&id], id, globals));
        let _ = funcs;
    }

    fn lower_compound(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        compound: &fe::parse::Compound,
    ) {
        for item in &compound.items {
            match item {
                fe::parse::BlockItem::Decl { ids, .. } => {
                    for id in ids {
                        self.lower_local(module, func, frame, funcs, externs, globals, *id);
                    }
                }
                fe::parse::BlockItem::Stmt(stmt) => self.lower_stmt(module, func, frame, funcs, externs, globals, stmt),
            }
        }
    }

    fn lower_local(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        id: fe::sema::DeclId,
    ) {
        let binding = &self.sema.decls[id.0 as usize];
        if let Some(loc) = self.debug_location(module, binding.span) {
            frame.loc = Some(loc);
        }
        if binding.block_static {
            // Static storage: the module global was folded upfront, so
            // uses already resolve through `place_of_var` to a
            // `GlobalAddr`. No slot and no per-call initialization.
            debug_assert!(globals.contains_key(&id), "block-static global missing");
            return;
        }
        // A pure `extern` declaration (including a block-scope one with
        // no file-scope entity) introduces no storage: it reuses the
        // import declared upfront, so no slot is allocated. Function
        // declarations are the same: calls resolve through
        // `funcs`/`externs`, never through a slot.
        if matches!(binding.kind, fe::sema::BindingKind::Func) {
            return;
        }
        if binding.linkage == fe::sema::Linkage::External && !binding.defined && !binding.tentative {
            debug_assert!(globals.contains_key(&id), "extern import missing");
            return;
        }
        if Self::ty_is_vla_object(&binding.ty) {
            self.lower_vla_object(module, func, frame, funcs, externs, globals, id);
            return;
        }
        let mir_ty = lower_ty(&binding.ty, self.sema, self.interner);
        let slot = self.alloca(func, frame, mir_ty);
        frame.slots.insert(id, slot);
        if let Some(init) = self.sema.inits.get(&id) {
            self.lower_init(module, func, frame, funcs, externs, globals, slot, &binding.ty, init);
        }
    }

    /// Lower one `asm` statement into a [`mir::InstKind::InlineAsm`]
    /// block: inputs evaluate in order, then output addresses. The
    /// backend turns the block into a call to a generated wrapper that
    /// runs the template (see `glcc_lower_cl::asm`). A bit-field output
    /// round-trips through a unit-typed temp: the wrapper stores whole
    /// units, and the field merges back afterwards.
    fn lower_asm_stmt(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        asm: &fe::parse::AsmStmt,
    ) {
        let mut values = Vec::with_capacity(asm.inputs.len());
        for input in &asm.inputs {
            let arch = self.sema.target.arch;
            let value = if fe::check::memory_only_constraint(&input.constraint, arch) {
                self.place(module, func, frame, funcs, externs, globals, &input.expr).0
            } else if fe::check::has_immediate_constraint(&input.constraint, arch) {
                // `"i"`/`"n"`/`"I"`/... take integer constant
                // expressions (`-1`, `sizeof x`, `1 << 3`), not just
                // literals: fold here so the wrapper substitutes a
                // constant. Anything else (including a runtime value
                // for a `"Ir"`-style operand) falls through to the
                // normal path, where a register or memory alternative
                // carries it.
                match fe::eval::eval_const(&input.expr, self.sema, self.interner) {
                    Some(fe::eval::Value::Int(n)) => {
                        let mir_ty = lower_ty(&input.expr.ty, self.sema, self.interner);
                        let bits = match mir_ty {
                            mir::Ty::Int { bits } => bits.max(8),
                            _ => 64,
                        };
                        func.push_const(mir::Const::Int { bits, value: n })
                    }
                    _ => self.expr_value(module, func, frame, funcs, externs, globals, &input.expr),
                }
            } else {
                self.expr_value(module, func, frame, funcs, externs, globals, &input.expr)
            };
            values.push(value);
        }
        // Every output lvalue's address is evaluated, tie or not:
        // `asm("" : "=m"(a[i++]))` still bumps `i`.
        struct BitfieldOut {
            temp: mir::Value,
            addr: mir::Value,
            decl_ty: fe::parse::Ty,
            bit: u32,
            width: u32,
            volatile: bool,
        }
        let mut operands = Vec::with_capacity(asm.outputs.len() + asm.inputs.len());
        let mut conversion: Vec<(fe::parse::Ty, Option<fe::parse::Ty>)> = Vec::with_capacity(asm.outputs.len());
        let mut bitfield_outs: Vec<BitfieldOut> = Vec::new();
        for output in &asm.outputs {
            let (addr, _) = self.place(module, func, frame, funcs, externs, globals, &output.expr);
            let name = output.name.and_then(|sym| self.interner.resolve(sym)).map(str::to_string);
            let bitfield = match &output.expr.kind {
                fe::check::CheckedExprKind::Member {
                    bitfield: Some((bit, width)),
                    decl_ty,
                    ..
                } => Some((*bit, *width, decl_ty.clone())),
                _ => None,
            };
            match bitfield {
                Some((bit, width, decl_ty)) => {
                    let unit_mir = lower_ty(&decl_ty, self.sema, self.interner);
                    let temp = self.alloca(func, frame, unit_mir.clone());
                    let volatile = access_volatile(&output.expr, &self.sema.decls);
                    if output.constraint.contains('+') {
                        let field = self.bitfield_load(func, frame, addr, &decl_ty, &output.expr.ty, bit, width, is_signed(&decl_ty, self.sema), volatile);
                        let unit = self.convert(func, frame, field, &output.expr.ty, &decl_ty);
                        self.store(func, frame, temp, unit, false);
                    }
                    bitfield_outs.push(BitfieldOut {
                        temp,
                        addr,
                        decl_ty: decl_ty.clone(),
                        bit,
                        width,
                        volatile,
                    });
                    conversion.push((output.expr.ty.clone(), Some(decl_ty.clone())));
                    operands.push(mir::AsmOperand {
                        value: temp,
                        constraint: output.constraint.clone(),
                        ty: unit_mir,
                        tie: None,
                        name,
                    });
                }
                None => {
                    let ty = lower_ty(&output.expr.ty, self.sema, self.interner);
                    conversion.push((output.expr.ty.clone(), None));
                    operands.push(mir::AsmOperand {
                        value: addr,
                        constraint: output.constraint.clone(),
                        ty,
                        tie: None,
                        name,
                    });
                }
            }
        }
        for (i, input) in asm.inputs.iter().enumerate() {
            let name = input.name.and_then(|sym| self.interner.resolve(sym)).map(str::to_string);
            let mir_ty = lower_ty(&input.expr.ty, self.sema, self.interner);
            let aggregate = mir_ty.is_aggregate();
            let memory = fe::check::memory_only_constraint(&input.constraint, self.sema.target.arch);
            let value = match input.tie() {
                Some(n) if aggregate => {
                    // An aggregate shares only a memory output: its value
                    // is copied into the output's location, and both
                    // operands name that one address.
                    if !fe::check::memory_only_constraint(&asm.outputs[n].constraint, self.sema.target.arch) {
                        panic!("asm register operand is wider than a register");
                    }
                    self.store(func, frame, operands[n].value, values[i], false);
                    values[i]
                }
                Some(n) => {
                    let (member_ty, unit_ty) = &conversion[n];
                    let value = self.convert(func, frame, values[i], &input.expr.ty, member_ty);
                    match unit_ty {
                        Some(unit_ty) => self.convert(func, frame, value, member_ty, unit_ty),
                        None => value,
                    }
                }
                None => values[i],
            };
            // An aggregate in a register constraint crosses as its
            // address, like an array operand in GCC.
            let ty = if aggregate && !memory { mir::Ty::ptr(mir::Ty::Void) } else { mir_ty };
            operands.push(mir::AsmOperand {
                value,
                constraint: input.constraint.clone(),
                ty,
                tie: input.tie().map(|n| n as u32),
                name,
            });
        }
        let labels: Vec<String> = asm.labels.iter().map(|label| self.interner.resolve(*label).unwrap_or("?").to_string()).collect();
        let block = mir::AsmBlock {
            template: asm.template.clone(),
            operands,
            outputs: asm.outputs.len() as u32,
            clobbers: asm.clobbers.clone(),
            labels,
            is_volatile: asm.is_volatile,
            is_basic: asm.is_basic,
        };
        // `asm goto` needs the wrapper's selector as an instruction
        // result; plain asm ignores it.
        let selector = if asm.is_goto {
            Some(self.emit(func, frame, mir::InstKind::InlineAsm { block: Box::new(block) }, mir::Ty::int(64)))
        } else {
            self.emit_unit(func, frame, mir::InstKind::InlineAsm { block: Box::new(block) });
            None
        };
        for out in bitfield_outs {
            let unit_mir = lower_ty(&out.decl_ty, self.sema, self.interner);
            let value = self.load(func, frame, out.temp, unit_mir, false);
            self.bitfield_store(func, frame, out.addr, &out.decl_ty, value, out.bit, out.width, out.volatile);
        }
        if let Some(selector) = selector {
            // The wrapper returns 0 when the template fell through and
            // `i + 1` when it jumped to `labels[i]`. Dispatch after the
            // outputs merge so both paths observe the stored results.
            let mut cases = Vec::with_capacity(asm.labels.len());
            for (i, name) in asm.labels.iter().enumerate() {
                let target = match frame.labels.get(name) {
                    Some(block) => *block,
                    None => {
                        let block = func.push_block(vec![]);
                        frame.labels.insert(*name, block);
                        block
                    }
                };
                cases.push(((i + 1) as i128, target));
            }
            let fallthrough = func.push_block(vec![]);
            self.terminate(
                func,
                frame,
                mir::Terminator::Switch {
                    scrutinee: selector,
                    cases,
                    default: fallthrough,
                },
            );
            frame.current = Some(fallthrough);
        }
    }

    /// Lower a statement whose value becomes an enclosing statement
    /// expression's result: an expression statement, or a chain of
    /// labels ending in one. Each label defines its jump block exactly
    /// like `lower_stmt`'s label arm, so computed gotos still find it.
    /// Anything else lowers for effect and yields no value.
    fn lower_value_stmt(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        stmt: &fe::parse::Stmt,
    ) -> Option<mir::Value> {
        if let Some(loc) = self.debug_location(module, stmt.span()) {
            frame.loc = Some(loc);
        }
        match stmt {
            fe::parse::Stmt::Expr { expr, .. } => Some(self.expr_value(module, func, frame, funcs, externs, globals, expr)),
            fe::parse::Stmt::Label { name, stmt, .. } => {
                let target = match frame.labels.get(name) {
                    Some(block) => *block,
                    None => {
                        let block = func.push_block(vec![]);
                        frame.labels.insert(*name, block);
                        block
                    }
                };
                if frame.current.is_some() {
                    self.branch(func, frame, target, vec![]);
                }
                frame.current = Some(target);
                self.lower_value_stmt(module, func, frame, funcs, externs, globals, stmt)
            }
            _ => {
                self.lower_stmt(module, func, frame, funcs, externs, globals, stmt);
                None
            }
        }
    }

    fn lower_stmt(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        stmt: &fe::parse::Stmt,
    ) {
        if let Some(loc) = self.debug_location(module, stmt.span()) {
            frame.loc = Some(loc);
        }
        match stmt {
            fe::parse::Stmt::Empty { .. } => {}
            fe::parse::Stmt::Expr { expr, .. } => {
                // Ensure a current block exists even for dead code.
                self.live_block(func, frame);
                self.expr_value(module, func, frame, funcs, externs, globals, expr);
            }
            fe::parse::Stmt::Asm(asm) => {
                self.lower_asm_stmt(module, func, frame, funcs, externs, globals, asm);
            }
            fe::parse::Stmt::Return { expr, .. } => {
                let val = expr.as_ref().map(|e| self.expr_value(module, func, frame, funcs, externs, globals, e));
                // A void return carries no value (a `return f();` of a
                // void `f`), and a bare `return;` in a value-returning
                // function (C89 legacy code) returns an undefined value,
                // like falling off the end. An async body returns its
                // declared result type, not the handle its symbol yields
                // (the coroutine pass rewrites the return).
                let result_ty = func.coroutine.clone().unwrap_or_else(|| func.sig.ret.clone());
                let val = if result_ty.is_void() {
                    None
                } else {
                    Some(val.unwrap_or_else(|| {
                        func.values.push(mir::ValueData {
                            ty: result_ty,
                            def: mir::Def::Const(mir::Const::Undef),
                        })
                    }))
                };
                self.terminate(func, frame, mir::Terminator::Return { val });
            }
            fe::parse::Stmt::Compound(inner) => self.lower_compound(module, func, frame, funcs, externs, globals, inner),
            fe::parse::Stmt::If { cond, then, els, .. } => {
                let test = self.bool_of(module, func, frame, funcs, externs, globals, cond);
                let then_block = func.push_block(vec![]);
                let else_block = func.push_block(vec![]);
                let merge = func.push_block(vec![]);
                self.terminate(
                    func,
                    frame,
                    mir::Terminator::CondBr {
                        cond: test,
                        then_block,
                        else_block,
                        then_args: Default::default(),
                        else_args: Default::default(),
                    },
                );
                frame.current = Some(then_block);
                self.lower_stmt(module, func, frame, funcs, externs, globals, then);
                // Every open path needs an explicit edge to the merge:
                // MIR blocks never fall through.
                let mut live = false;
                if frame.current.is_some() {
                    self.branch(func, frame, merge, vec![]);
                    live = true;
                }
                frame.current = Some(else_block);
                if let Some(els) = els {
                    self.lower_stmt(module, func, frame, funcs, externs, globals, els);
                }
                if frame.current.is_some() {
                    self.branch(func, frame, merge, vec![]);
                    live = true;
                }
                self.join(func, frame, merge, live);
            }
            fe::parse::Stmt::While { cond, body, .. } => {
                let header = func.push_block(vec![]);
                let loop_body = func.push_block(vec![]);
                let exit = func.push_block(vec![]);
                self.branch(func, frame, header, vec![]);
                frame.current = Some(header);
                let test = self.bool_of(module, func, frame, funcs, externs, globals, cond);
                self.terminate(
                    func,
                    frame,
                    mir::Terminator::CondBr {
                        cond: test,
                        then_block: loop_body,
                        else_block: exit,
                        then_args: Default::default(),
                        else_args: Default::default(),
                    },
                );
                frame.loops.push(LoopTarget { cont: header });
                frame.breaks.push(exit);
                frame.current = Some(loop_body);
                self.lower_stmt(module, func, frame, funcs, externs, globals, body);
                frame.breaks.pop();
                frame.loops.pop();
                if frame.current.is_some() {
                    self.branch(func, frame, header, vec![]);
                }
                frame.current = Some(exit);
            }
            fe::parse::Stmt::DoWhile { body, cond, .. } => {
                let loop_body = func.push_block(vec![]);
                let check = func.push_block(vec![]);
                let exit = func.push_block(vec![]);
                self.branch(func, frame, loop_body, vec![]);
                frame.loops.push(LoopTarget { cont: check });
                frame.breaks.push(exit);
                frame.current = Some(loop_body);
                self.lower_stmt(module, func, frame, funcs, externs, globals, body);
                frame.breaks.pop();
                frame.loops.pop();
                if frame.current.is_some() {
                    self.branch(func, frame, check, vec![]);
                }
                frame.current = Some(check);
                let test = self.bool_of(module, func, frame, funcs, externs, globals, cond);
                self.terminate(
                    func,
                    frame,
                    mir::Terminator::CondBr {
                        cond: test,
                        then_block: loop_body,
                        else_block: exit,
                        then_args: Default::default(),
                        else_args: Default::default(),
                    },
                );
                frame.current = Some(exit);
            }
            fe::parse::Stmt::For { init, cond, step, body, .. } => {
                match init {
                    fe::parse::ForInit::Decl(ids) => {
                        for id in ids {
                            self.lower_local(module, func, frame, funcs, externs, globals, *id);
                        }
                    }
                    fe::parse::ForInit::Expr(e) => {
                        self.live_block(func, frame);
                        self.expr_value(module, func, frame, funcs, externs, globals, e);
                    }
                    fe::parse::ForInit::Empty => {}
                }
                let header = func.push_block(vec![]);
                let loop_body = func.push_block(vec![]);
                let step_block = func.push_block(vec![]);
                let exit = func.push_block(vec![]);
                self.branch(func, frame, header, vec![]);
                frame.current = Some(header);
                match cond {
                    Some(c) => {
                        let test = self.bool_of(module, func, frame, funcs, externs, globals, c);
                        self.terminate(
                            func,
                            frame,
                            mir::Terminator::CondBr {
                                cond: test,
                                then_block: loop_body,
                                else_block: exit,
                                then_args: Default::default(),
                                else_args: Default::default(),
                            },
                        );
                    }
                    None => self.branch(func, frame, loop_body, vec![]),
                }
                frame.loops.push(LoopTarget { cont: step_block });
                frame.breaks.push(exit);
                frame.current = Some(loop_body);
                self.lower_stmt(module, func, frame, funcs, externs, globals, body);
                frame.breaks.pop();
                frame.loops.pop();
                if frame.current.is_some() {
                    self.branch(func, frame, step_block, vec![]);
                }
                frame.current = Some(step_block);
                if let Some(s) = step {
                    self.expr_value(module, func, frame, funcs, externs, globals, s);
                }
                // The step block always falls into the header: it is only
                // reachable via branches above, never fallen into, so an
                // open step always ends here.
                if frame.current.is_some() {
                    self.branch(func, frame, header, vec![]);
                }
                frame.current = Some(exit);
            }
            fe::parse::Stmt::Break { .. } => {
                // Loops and `switch`es share one stack so `break`
                // takes whichever encloses most tightly; the checker
                // rejects strays, so an empty stack is a lowering bug.
                let target = frame.breaks.last().copied().expect("break outside a loop or switch");
                self.branch(func, frame, target, vec![]);
            }
            fe::parse::Stmt::Continue { .. } => {
                let target = frame.loops.last().expect("continue outside a loop").cont;
                self.branch(func, frame, target, vec![]);
            }
            fe::parse::Stmt::Switch { cond, body, .. } => {
                // C compares in the promoted controlling type, and the
                // checker's duplicate-`case` detection ran in exactly
                // that type: promote the scrutinee explicitly (the
                // tree carries no conversion node) so dispatch agrees
                // with it. A narrow `char` switch would otherwise
                // miscompare high cases.
                let promoted = fe::eval::promoted(&cond.ty);
                let scrutinee = self.expr_value(module, func, frame, funcs, externs, globals, cond);
                let scrutinee = self.convert(func, frame, scrutinee, &cond.ty, &promoted);
                let dispatch = self.live_block(func, frame);
                let body_entry = func.push_block(vec![]);
                let exit = func.push_block(vec![]);
                frame.switches.push(SwitchTarget {
                    promoted,
                    scrutinee,
                    cases: Vec::new(),
                    ranges: Vec::new(),
                    default: None,
                });
                frame.breaks.push(exit);
                frame.current = Some(body_entry);
                self.lower_stmt(module, func, frame, funcs, externs, globals, body);
                frame.breaks.pop();
                let ctx = frame.switches.pop().expect("switch pushed by this arm");
                // A trailing open path falls out of the switch.
                if frame.current.is_some() {
                    self.branch(func, frame, exit, vec![]);
                }
                // Patch the dispatch now that every case block exists.
                // Cases may hide in nested compounds, selections, and
                // loops (Duff's device); all recorded into this
                // context while the body lowered.
                frame.current = Some(dispatch);
                let default = ctx.default.unwrap_or(exit);
                let bits = int_bits(&lower_ty(&ctx.promoted, self.sema, self.interner));
                // GNU case ranges test first, each in its own block
                // (`lo <= x <= hi`, compared in the promoted type's
                // signedness); the exact-value dispatch follows.
                let signed = is_signed(&ctx.promoted, self.sema);
                let mut cur = dispatch;
                for (lo, hi, target) in ctx.ranges.iter() {
                    let next = func.push_block(vec![]);
                    frame.current = Some(cur);
                    let lo_v = self.iconst(func, bits, *lo);
                    let hi_v = self.iconst(func, bits, *hi);
                    let ge = self.emit(
                        func,
                        frame,
                        mir::InstKind::Icmp {
                            pred: if signed { mir::IcmpPred::Sge } else { mir::IcmpPred::Uge },
                            lhs: ctx.scrutinee,
                            rhs: lo_v,
                        },
                        mir::Ty::Bool,
                    );
                    let le = self.emit(
                        func,
                        frame,
                        mir::InstKind::Icmp {
                            pred: if signed { mir::IcmpPred::Sle } else { mir::IcmpPred::Ule },
                            lhs: ctx.scrutinee,
                            rhs: hi_v,
                        },
                        mir::Ty::Bool,
                    );
                    let inside = self.emit(func, frame, mir::InstKind::And(ge, le), mir::Ty::Bool);
                    self.terminate(
                        func,
                        frame,
                        mir::Terminator::CondBr {
                            cond: inside,
                            then_block: *target,
                            else_block: next,
                            then_args: Default::default(),
                            else_args: Default::default(),
                        },
                    );
                    cur = next;
                }
                frame.current = Some(cur);
                if bits <= 64 {
                    self.terminate(
                        func,
                        frame,
                        mir::Terminator::Switch {
                            scrutinee: ctx.scrutinee,
                            cases: ctx.cases,
                            default,
                        },
                    );
                } else {
                    self.switch_128(func, frame, ctx, default);
                }
                frame.current = Some(exit);
            }
            fe::parse::Stmt::Goto { label, .. } => {
                // Labels resolve during the single pass: a forward
                // jump creates its block early, and the later `Label`
                // reuses it. The checker guarantees the label exists
                // in this function.
                let target = match frame.labels.get(label) {
                    Some(block) => *block,
                    None => {
                        let block = func.push_block(vec![]);
                        frame.labels.insert(*label, block);
                        block
                    }
                };
                self.branch(func, frame, target, vec![]);
            }
            fe::parse::Stmt::GotoComputed { expr, .. } => {
                // Dispatch on the label id: branch to the function's
                // dispatch block with the computed value, which the
                // switch there routes to the label. Created lazily so
                // functions without computed gotos pay nothing.
                let value = self.expr_value(module, func, frame, funcs, externs, globals, expr);
                let (dispatch, _) = match frame.computed_goto {
                    Some(pair) => pair,
                    None => {
                        let block = func.push_block(vec![mir::Ty::int(64)]);
                        let param = func.blocks[block].params[0];
                        frame.computed_goto = Some((block, param));
                        (block, param)
                    }
                };
                self.branch(func, frame, dispatch, vec![value]);
            }
            fe::parse::Stmt::Label { name, stmt, .. } => {
                // Execution falls through into a label; only an
                // already-terminated path (after `return`/`goto`)
                // arrives solely by jump.
                let target = match frame.labels.get(name) {
                    Some(block) => *block,
                    None => {
                        let block = func.push_block(vec![]);
                        frame.labels.insert(*name, block);
                        block
                    }
                };
                if frame.current.is_some() {
                    self.branch(func, frame, target, vec![]);
                }
                frame.current = Some(target);
                self.lower_stmt(module, func, frame, funcs, externs, globals, stmt);
            }
            fe::parse::Stmt::Case { expr, hi, stmt, .. } => {
                // Like a label, but recorded into the innermost open
                // `switch` for the dispatch terminator. The checker
                // rejects strays and duplicates, so those are backstop
                // panics; values fold through the same `const_as` the
                // duplicate check used.
                let target = func.push_block(vec![]);
                let value = match expr {
                    None => None,
                    Some(label) => {
                        let ctx = frame.switches.last().expect("case outside a switch");
                        Some(fe::eval::const_as(&ctx.promoted, label, self.sema, self.interner).expect("case label must fold"))
                    }
                };
                let upper = hi.as_ref().map(|label| {
                    let ctx = frame.switches.last().expect("case outside a switch");
                    fe::eval::const_as(&ctx.promoted, label, self.sema, self.interner).expect("case range bound must fold")
                });
                {
                    let ctx = frame.switches.last_mut().expect("case outside a switch");
                    match (value, upper) {
                        (None, _) => {
                            if ctx.default.replace(target).is_some() {
                                panic!("duplicate `default` in `switch`");
                            }
                        }
                        (Some(v), None) => ctx.cases.push((v, target)),
                        (Some(lo), Some(hi)) => ctx.ranges.push((lo, hi, target)),
                    }
                }
                if frame.current.is_some() {
                    self.branch(func, frame, target, vec![]);
                }
                frame.current = Some(target);
                self.lower_stmt(module, func, frame, funcs, externs, globals, stmt);
            }
        }
    }

    /// Join two branch paths at `merge`: live when either side falls
    /// through, otherwise sealed as unreachable so it still verifies.
    fn join(&self, func: &mut mir::Function, frame: &mut Frame, merge: mir::Block, live: bool) {
        if live {
            frame.current = Some(merge);
        } else {
            frame.current = Some(merge);
            self.terminate(func, frame, mir::Terminator::Unreachable);
        }
    }

    fn branch(&self, func: &mut mir::Function, frame: &mut Frame, dest: mir::Block, args: Vec<mir::Value>) {
        self.terminate(
            func,
            frame,
            mir::Terminator::Br {
                dest,
                args: args.into_iter().collect(),
            },
        );
    }

    /// A 128-bit `switch` as an if-else chain: the MIR `Switch`
    /// terminator only covers scrutinees the Clif emission can
    /// compare directly (64 bits and under), so a wide controlling
    /// expression compares through ordinary conditional branches.
    /// Enters with the dispatch block current; every test compares
    /// full 128-bit values for equality.
    fn switch_128(&self, func: &mut mir::Function, frame: &mut Frame, ctx: SwitchTarget, default: mir::Block) {
        if ctx.cases.is_empty() {
            self.branch(func, frame, default, vec![]);
            return;
        }
        let tests: Vec<mir::Block> = ctx.cases.iter().map(|_| func.push_block(vec![])).collect();
        self.branch(func, frame, tests[0], vec![]);
        for (i, (value, dest)) in ctx.cases.into_iter().enumerate() {
            frame.current = Some(tests[i]);
            let probe = self.iconst(func, 128, value);
            let same = self.emit(
                func,
                frame,
                mir::InstKind::Icmp {
                    pred: mir::IcmpPred::Eq,
                    lhs: ctx.scrutinee,
                    rhs: probe,
                },
                mir::Ty::Bool,
            );
            let otherwise = if i + 1 < tests.len() { tests[i + 1] } else { default };
            self.terminate(
                func,
                frame,
                mir::Terminator::CondBr {
                    cond: same,
                    then_block: dest,
                    else_block: otherwise,
                    then_args: Default::default(),
                    else_args: Default::default(),
                },
            );
        }
    }

    /// Rvalue of an expression as a MIR value.
    fn expr_value(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        expr: &fe::check::CheckedExpr,
    ) -> mir::Value {
        let mir_ty = lower_ty(&expr.ty, self.sema, self.interner);
        match &expr.kind {
            fe::check::CheckedExprKind::Int(_) | fe::check::CheckedExprKind::Char(_) | fe::check::CheckedExprKind::Float(_) => {
                // A GNU imaginary literal (`1i`, `1.0i`) is an
                // `Int`/`Float` node with complex type: zero real
                // part, literal imaginary part.
                if is_complex(&expr.ty) {
                    return self.complex_literal(func, frame, expr);
                }
                let folded = fe::eval::eval_const(expr, self.sema, self.interner).unwrap_or_else(|| panic!("literal must fold"));
                func.push_const(self.const_of_folded(&folded, &mir_ty))
            }
            fe::check::CheckedExprKind::Ident(id) => {
                let binding = &self.sema.decls[id.0 as usize];
                match binding.kind {
                    fe::sema::BindingKind::EnumConst { value } => self.iconst(func, int_bits(&mir_ty), value),
                    fe::sema::BindingKind::Func => {
                        if let Some(target) = funcs.get(id) {
                            let target = *target;
                            self.emit(func, frame, mir::InstKind::FuncAddr { func: target }, mir_ty)
                        } else {
                            // Prototype-only functions are foreign
                            // imports; their address is an import
                            // address like any other.
                            let Some(target) = externs.get(id) else {
                                let binding = &self.sema.decls[id.0 as usize];
                                let name = binding.name.and_then(|n| self.interner.resolve(n)).unwrap_or("<anon>");
                                panic!("address of undefined function '{name}'");
                            };
                            let target = *target;
                            self.emit(func, frame, mir::InstKind::ExternAddr { target }, mir_ty)
                        }
                    }
                    fe::sema::BindingKind::Var => {
                        let addr = self.place_of_var(func, frame, globals, *id);
                        let array = self.expr_denotes_array(expr);
                        self.rvalue_of_place(func, frame, addr, array, mir_ty, access_volatile(expr, &self.sema.decls))
                    }
                    _ => panic!("identifier is not a value"),
                }
            }
            fe::check::CheckedExprKind::Cast { ty, expr: inner } | fe::check::CheckedExprKind::Convert { expr: inner, ty } => {
                let from_ty = &inner.ty;
                let from = self.expr_value(module, func, frame, funcs, externs, globals, inner);
                self.convert(func, frame, from, from_ty, ty)
            }
            fe::check::CheckedExprKind::Unary(op, inner) => {
                use fe::parse::UnaryOp::*;
                if matches!(op, PreInc | PreDec | PostInc | PostDec) {
                    let (addr, _) = self.place(module, func, frame, funcs, externs, globals, inner);
                    let inner_ty = lower_ty(&inner.ty, self.sema, self.interner);
                    // Pointer steps scale by the frontend pointee size
                    // through byte arithmetic: the MIR pointee is
                    // erased (structs become `void`), so a GEP element
                    // step cannot know the stride.
                    if matches!(inner_ty, mir::Ty::Ptr(_)) {
                        let fe::parse::Ty::Pointer { pointee, .. } = inner.ty.unqualified().clone() else {
                            panic!("pointer step without a pointer");
                        };
                        let scale = match pointee.unqualified() {
                            fe::parse::Ty::Void | fe::parse::Ty::Func { .. } => 1,
                            _ => self.sema.type_size(&pointee, self.interner).expect("step over sized pointee"),
                        };
                        let mir::Ty::Ptr(pointee_mir) = &inner_ty else {
                            panic!("pointer step without a pointer");
                        };
                        let pointee_mir = pointee_mir.as_ref().clone();
                        let cur = self.load(func, frame, addr, inner_ty.clone(), access_volatile(inner, &self.sema.decls));
                        let one = self.iconst(func, 64, 1);
                        let mag = if matches!(op, PreInc | PostInc) {
                            one
                        } else {
                            self.emit(func, frame, mir::InstKind::Neg(one), mir::Ty::int(64))
                        };
                        let (next, _) = self.scaled_index_addr(func, frame, cur, mag, &fe::parse::Ty::LongLong, scale, pointee_mir);
                        self.store(func, frame, addr, next, access_volatile(inner, &self.sema.decls));
                        return if matches!(op, PreInc | PreDec) { next } else { cur };
                    }
                    // Bit-fields step through extraction and RMW: the
                    // unit address holds the whole unit, so a plain
                    // load would step the packed neighbors.
                    if let fe::check::CheckedExprKind::Member {
                        bitfield: Some((bit, width)),
                        decl_ty,
                        ..
                    } = &inner.kind
                    {
                        let cur = self.bitfield_load(
                            func,
                            frame,
                            addr,
                            decl_ty,
                            &inner.ty,
                            *bit,
                            *width,
                            is_signed(decl_ty, self.sema),
                            access_volatile(inner, &self.sema.decls),
                        );
                        let one = self.iconst(func, int_bits(&inner_ty), 1);
                        let next = if matches!(op, PreInc | PostInc) {
                            self.emit(func, frame, mir::InstKind::Add(cur, one), inner_ty.clone())
                        } else {
                            self.emit(func, frame, mir::InstKind::Sub(cur, one), inner_ty.clone())
                        };
                        let unit = self.convert(func, frame, next, &inner.ty, decl_ty);
                        self.bitfield_store(func, frame, addr, decl_ty, unit, *bit, *width, access_volatile(inner, &self.sema.decls));
                        // `++f` yields the value stored, narrowed to the
                        // field: the step above wraps at the unit width.
                        return if matches!(op, PreInc | PreDec) {
                            self.reduce_to_bits(func, frame, next, *width, is_signed(decl_ty, self.sema), inner_ty)
                        } else {
                            cur
                        };
                    }
                    let cur = self.load(func, frame, addr, inner_ty.clone(), access_volatile(inner, &self.sema.decls));
                    let one = match &inner_ty {
                        mir::Ty::Float(_) => {
                            // 1.0 at the operand width: long double uses
                            // the target representation (f80 bytes or
                            // quad bits) through the shared const fold.
                            let bits = match float_kind(&inner_ty) {
                                mir::FloatKind::F32 => u128::from(1.0f32.to_bits()),
                                mir::FloatKind::F64 => u128::from(1.0f64.to_bits()),
                                mir::FloatKind::F80 => float_bits(fe::eval::FloatValue::long_double_one_in(fe::target::LongDoubleKind::F80), &inner_ty),
                                mir::FloatKind::F128 => float_bits(fe::eval::FloatValue::long_double_one_in(fe::target::LongDoubleKind::F128), &inner_ty),
                            };
                            func.push_const(mir::Const::Float {
                                kind: float_kind(&inner_ty),
                                bits,
                            })
                        }
                        _ => self.iconst(func, int_bits(&inner_ty), 1),
                    };
                    let kind = match (op, &inner_ty) {
                        (PreInc | PostInc, mir::Ty::Float(_)) => mir::InstKind::FAdd(cur, one),
                        (PreDec | PostDec, mir::Ty::Float(_)) => mir::InstKind::FSub(cur, one),
                        (PreInc | PostInc, _) => mir::InstKind::Add(cur, one),
                        _ => mir::InstKind::Sub(cur, one),
                    };
                    let next = self.emit(func, frame, kind, inner_ty);
                    self.store(func, frame, addr, next, access_volatile(inner, &self.sema.decls));
                    // Prefix yields the new value, postfix the old one.
                    return if matches!(op, PreInc | PreDec) { next } else { cur };
                }
                let operand_ty = lower_ty(&inner.ty, self.sema, self.interner);
                // The operand evaluates exactly once: value ops take it
                // here, while `AddrOf`/`Deref` address through `place`
                // or a single evaluation below. Eagerly taking it above
                // this match double-evaluated side effects (`&a[j++]`
                // stepped twice and stored the wrong address).
                match op {
                    fe::parse::UnaryOp::Pos => self.expr_value(module, func, frame, funcs, externs, globals, inner),
                    fe::parse::UnaryOp::Neg => {
                        if is_complex(&inner.ty) {
                            let value = self.expr_value(module, func, frame, funcs, externs, globals, inner);
                            return self.complex_neg(func, frame, value, &inner.ty);
                        }
                        let value = self.expr_value(module, func, frame, funcs, externs, globals, inner);
                        match operand_ty {
                            mir::Ty::Float(_) => self.emit(func, frame, mir::InstKind::FNeg(value), mir_ty),
                            _ => {
                                let neg = self.emit(func, frame, mir::InstKind::Neg(value), mir_ty.clone());
                                self.reduce_bitfield_precision(func, frame, neg, expr, mir_ty)
                            }
                        }
                    }
                    fe::parse::UnaryOp::BitNot => {
                        // GNU `~x` on a complex operand is the
                        // conjugate (negated imaginary part), not a
                        // bitwise complement.
                        if is_complex(&inner.ty) {
                            let value = self.expr_value(module, func, frame, funcs, externs, globals, inner);
                            return self.complex_conj(func, frame, value, &inner.ty);
                        }
                        let value = self.expr_value(module, func, frame, funcs, externs, globals, inner);
                        let not = self.emit(func, frame, mir::InstKind::Not(value), mir_ty.clone());
                        self.reduce_bitfield_precision(func, frame, not, expr, mir_ty)
                    }
                    fe::parse::UnaryOp::Not => {
                        if is_complex(&inner.ty) {
                            let value = self.expr_value(module, func, frame, funcs, externs, globals, inner);
                            let nz = self.complex_nonzero(func, frame, value, &inner.ty);
                            let one = self.iconst(func, 1, 1);
                            let is_zero = self.emit(func, frame, mir::InstKind::Xor(nz, one), mir::Ty::Bool);
                            return self.convert_bool(func, frame, is_zero, &mir_ty);
                        }
                        let value = self.expr_value(module, func, frame, funcs, externs, globals, inner);
                        let zero = match &operand_ty {
                            mir::Ty::Bool | mir::Ty::Int { .. } => self.iconst(func, int_bits(&operand_ty), 0),
                            mir::Ty::Ptr(_) => self.iconst(func, 64, 0),
                            mir::Ty::Float(kind) => func.push_const(mir::Const::Float { kind: *kind, bits: 0 }),
                            _ => todo!("logical negation lowering"),
                        };
                        let test = match &operand_ty {
                            mir::Ty::Float(_) => self.emit(
                                func,
                                frame,
                                mir::InstKind::Fcmp {
                                    pred: mir::FcmpPred::Eq,
                                    lhs: value,
                                    rhs: zero,
                                },
                                mir::Ty::Bool,
                            ),
                            _ => self.emit(
                                func,
                                frame,
                                mir::InstKind::Icmp {
                                    pred: mir::IcmpPred::Eq,
                                    lhs: value,
                                    rhs: zero,
                                },
                                mir::Ty::Bool,
                            ),
                        };
                        self.convert_bool(func, frame, test, &mir_ty)
                    }
                    fe::parse::UnaryOp::AddrOf => {
                        // Bit-fields have no addressable storage (C17
                        // §6.5.3.2p1); the checker should reject this,
                        // so reaching here is a loud backstop.
                        if matches!(&inner.kind, fe::check::CheckedExprKind::Member { bitfield: Some(_), .. }) {
                            panic!("cannot take address of bit-field");
                        }
                        match &inner.kind {
                            // Function designators decay without a load.
                            fe::check::CheckedExprKind::Ident(id) if matches!(self.sema.decls[id.0 as usize].kind, fe::sema::BindingKind::Func) => {
                                if let Some(target) = funcs.get(id) {
                                    let target = *target;
                                    self.emit(func, frame, mir::InstKind::FuncAddr { func: target }, mir_ty)
                                } else {
                                    let Some(target) = externs.get(id) else {
                                        panic!("address of undefined function");
                                    };
                                    let target = *target;
                                    self.emit(func, frame, mir::InstKind::ExternAddr { target }, mir_ty)
                                }
                            }
                            _ => self.place(module, func, frame, funcs, externs, globals, inner).0,
                        }
                    }
                    fe::parse::UnaryOp::Deref => {
                        let ptr = self.expr_value(module, func, frame, funcs, externs, globals, inner);
                        let fe::parse::Ty::Pointer { pointee, .. } = inner.ty.unqualified().clone() else {
                            panic!("dereference of non-pointer");
                        };
                        if matches!(pointee.unqualified(), fe::parse::Ty::Func { .. }) {
                            // A dereferenced function pointer is a
                            // function designator, which decays right
                            // back to the pointer: the value IS the
                            // address.
                            return ptr;
                        }
                        let array = self.expr_denotes_array(expr);
                        self.rvalue_of_place(func, frame, ptr, array, mir_ty, access_volatile(expr, &self.sema.decls))
                    }
                    fe::parse::UnaryOp::Real | fe::parse::UnaryOp::Imag => {
                        // `__real__ x` / `__imag__ x` select a
                        // component of a complex operand. An lvalue
                        // base addresses its own storage (stores land
                        // in the right half); an rvalue base (a call,
                        // a ternary, an arithmetic result) extracts
                        // off the pair value, which has no address.
                        let index = if matches!(op, fe::parse::UnaryOp::Real) { 0 } else { 1 };
                        match &inner.kind {
                            fe::check::CheckedExprKind::Ident(_)
                            | fe::check::CheckedExprKind::Member { .. }
                            | fe::check::CheckedExprKind::Subscript { .. }
                            | fe::check::CheckedExprKind::CompoundLiteral { .. }
                            | fe::check::CheckedExprKind::Unary(fe::parse::UnaryOp::Deref, _)
                            | fe::check::CheckedExprKind::Unary(fe::parse::UnaryOp::Real | fe::parse::UnaryOp::Imag, _) => {
                                let (addr, _) = self.place(module, func, frame, funcs, externs, globals, expr);
                                self.load(func, frame, addr, mir_ty, access_volatile(inner, &self.sema.decls))
                            }
                            _ => {
                                let agg = self.expr_value(module, func, frame, funcs, externs, globals, inner);
                                self.complex_extract(func, frame, agg, &inner.ty, index)
                            }
                        }
                    }
                    _ => todo!("unary operator lowering"),
                }
            }
            fe::check::CheckedExprKind::Binary(op, lhs, rhs) => {
                use fe::parse::BinaryOp::*;
                // Short-circuit operators must not evaluate the right
                // side eagerly.
                if matches!(op, And | Or) {
                    return self.logical(module, func, frame, funcs, externs, globals, op, lhs, rhs, &mir_ty);
                }
                // Comma evaluates both sides for effects and yields the
                // right one, whatever their types (either side may be
                // a pointer without implying pointer arithmetic).
                if matches!(op, Comma) {
                    self.expr_value(module, func, frame, funcs, externs, globals, lhs);
                    return self.expr_value(module, func, frame, funcs, externs, globals, rhs);
                }
                let lhs_ty = lower_ty(&lhs.ty, self.sema, self.interner);
                let lhs_ptr = matches!(lhs_ty, mir::Ty::Ptr(_));
                let rhs_ptr = matches!(lower_ty(&rhs.ty, self.sema, self.interner), mir::Ty::Ptr(_));
                if lhs_ptr || rhs_ptr {
                    return self.pointer_binary(module, func, frame, funcs, externs, globals, op, lhs, rhs, &mir_ty);
                }
                // Complex operands compute part-wise: `==`/`!=`
                // conjoin part equality, arithmetic follows the C
                // pair formulas. Both sides share the operand type
                // through the usual arithmetic conversions, so
                // either side names it.
                if is_complex(&lhs.ty) || is_complex(&rhs.ty) {
                    let x = self.expr_value(module, func, frame, funcs, externs, globals, lhs);
                    let y = self.expr_value(module, func, frame, funcs, externs, globals, rhs);
                    if matches!(op, Eq | Ne) {
                        return self.complex_cmp(func, frame, op, x, y, &lhs.ty, &mir_ty);
                    }
                    return self.complex_binop(func, frame, op, x, y, &lhs.ty);
                }
                // Comparisons finish through `Bool` conversion directly.
                if matches!(op, Lt | Le | Gt | Ge | Eq | Ne) {
                    let x = self.expr_value(module, func, frame, funcs, externs, globals, lhs);
                    let y = self.expr_value(module, func, frame, funcs, externs, globals, rhs);
                    let (signed, float) = (is_signed(&lhs.ty, self.sema), matches!(lhs_ty, mir::Ty::Float(_)));
                    let kind = match (op, float, signed) {
                        (Lt, true, _) => mir::InstKind::Fcmp {
                            pred: mir::FcmpPred::Lt,
                            lhs: x,
                            rhs: y,
                        },
                        (Le, true, _) => mir::InstKind::Fcmp {
                            pred: mir::FcmpPred::Le,
                            lhs: x,
                            rhs: y,
                        },
                        (Gt, true, _) => mir::InstKind::Fcmp {
                            pred: mir::FcmpPred::Gt,
                            lhs: x,
                            rhs: y,
                        },
                        (Ge, true, _) => mir::InstKind::Fcmp {
                            pred: mir::FcmpPred::Ge,
                            lhs: x,
                            rhs: y,
                        },
                        (Eq, true, _) => mir::InstKind::Fcmp {
                            pred: mir::FcmpPred::Eq,
                            lhs: x,
                            rhs: y,
                        },
                        (Ne, true, _) => mir::InstKind::Fcmp {
                            pred: mir::FcmpPred::Ne,
                            lhs: x,
                            rhs: y,
                        },
                        (Lt, false, true) => mir::InstKind::Icmp {
                            pred: mir::IcmpPred::Slt,
                            lhs: x,
                            rhs: y,
                        },
                        (Le, false, true) => mir::InstKind::Icmp {
                            pred: mir::IcmpPred::Sle,
                            lhs: x,
                            rhs: y,
                        },
                        (Gt, false, true) => mir::InstKind::Icmp {
                            pred: mir::IcmpPred::Sgt,
                            lhs: x,
                            rhs: y,
                        },
                        (Ge, false, true) => mir::InstKind::Icmp {
                            pred: mir::IcmpPred::Sge,
                            lhs: x,
                            rhs: y,
                        },
                        (Eq, false, _) => mir::InstKind::Icmp {
                            pred: mir::IcmpPred::Eq,
                            lhs: x,
                            rhs: y,
                        },
                        (Ne, false, _) => mir::InstKind::Icmp {
                            pred: mir::IcmpPred::Ne,
                            lhs: x,
                            rhs: y,
                        },
                        _ => self.int_cmp(op, signed, x, y),
                    };
                    let test = self.emit(func, frame, kind, mir::Ty::Bool);
                    return self.convert_bool(func, frame, test, &mir_ty);
                }
                let x = self.expr_value(module, func, frame, funcs, externs, globals, lhs);
                let y = self.expr_value(module, func, frame, funcs, externs, globals, rhs);
                let kind = match (op, &lhs_ty) {
                    (Add, mir::Ty::Float(_)) => mir::InstKind::FAdd(x, y),
                    (Sub, mir::Ty::Float(_)) => mir::InstKind::FSub(x, y),
                    (Mul, mir::Ty::Float(_)) => mir::InstKind::FMul(x, y),
                    (Div, mir::Ty::Float(_)) => mir::InstKind::FDiv(x, y),
                    (Add, _) => mir::InstKind::Add(x, y),
                    (Sub, _) => mir::InstKind::Sub(x, y),
                    (Mul, _) => mir::InstKind::Mul(x, y),
                    (Div, _) => {
                        if is_signed(&lhs.ty, self.sema) {
                            mir::InstKind::Sdiv(x, y)
                        } else {
                            mir::InstKind::Udiv(x, y)
                        }
                    }
                    (Rem, mir::Ty::Float(_)) => todo!("float remainder lowering"),
                    (Rem, _) => {
                        if is_signed(&lhs.ty, self.sema) {
                            mir::InstKind::Srem(x, y)
                        } else {
                            mir::InstKind::Urem(x, y)
                        }
                    }
                    (Shl, _) => mir::InstKind::Shl(x, y),
                    (Shr, _) => {
                        if is_signed(&lhs.ty, self.sema) {
                            mir::InstKind::AShr(x, y)
                        } else {
                            mir::InstKind::LShr(x, y)
                        }
                    }
                    (BitAnd, _) => mir::InstKind::And(x, y),
                    (BitXor, _) => mir::InstKind::Xor(x, y),
                    (BitOr, _) => mir::InstKind::Or(x, y),
                    _ => todo!("binary operator lowering"),
                };
                let value = self.emit(func, frame, kind, mir_ty.clone());
                self.reduce_bitfield_precision(func, frame, value, expr, mir_ty)
            }
            fe::check::CheckedExprKind::Assign(fe::parse::AssignOp::Assign, lhs, rhs) => {
                if let fe::check::CheckedExprKind::Member {
                    bitfield: Some((bit, width)),
                    decl_ty,
                    ..
                } = &lhs.kind
                {
                    // Bit-field stores read-modify-write the covering
                    // unit; the assignment evaluates to the stored
                    // (width-clamped) value, so re-read it.
                    let (addr, _) = self.place(module, func, frame, funcs, externs, globals, lhs);
                    let rhs_val = self.expr_value(module, func, frame, funcs, externs, globals, rhs);
                    let narrowed = self.convert(func, frame, rhs_val, &rhs.ty, &lhs.ty);
                    let unit = self.convert(func, frame, narrowed, &lhs.ty, decl_ty);
                    self.bitfield_store(func, frame, addr, decl_ty, unit, *bit, *width, access_volatile(lhs, &self.sema.decls));
                    return self.bitfield_load(
                        func,
                        frame,
                        addr,
                        decl_ty,
                        &lhs.ty,
                        *bit,
                        *width,
                        is_signed(decl_ty, self.sema),
                        access_volatile(lhs, &self.sema.decls),
                    );
                }
                let (addr, _) = self.place(module, func, frame, funcs, externs, globals, lhs);
                let rhs_val = self.expr_value(module, func, frame, funcs, externs, globals, rhs);
                let narrowed = self.convert(func, frame, rhs_val, &rhs.ty, &lhs.ty);
                self.store(func, frame, addr, narrowed, access_volatile(lhs, &self.sema.decls));
                // An assignment evaluates to the stored value.
                narrowed
            }
            fe::check::CheckedExprKind::CompoundAssign { op, lhs, rhs, op_ty, rhs_ty } => {
                // The lvalue evaluates once (C17 §6.5.16.2): its
                // address serves the read, the combination, and the
                // store. Bit-fields read-modify-write the covering
                // unit and evaluate to the stored (width-clamped)
                // value.
                let (addr, _) = self.place(module, func, frame, funcs, externs, globals, lhs);
                let mir_ty = lower_ty(&expr.ty, self.sema, self.interner);
                let bitfield = match &lhs.kind {
                    fe::check::CheckedExprKind::Member {
                        bitfield: Some((bit, width)),
                        decl_ty,
                        ..
                    } => Some((*bit, *width, decl_ty.clone())),
                    _ => None,
                };
                // The right operand evaluates before the lvalue is read
                // (GCC's order): `x[0] |= foo ()` sees the store `foo`
                // made, so the compound result is 131, not 3.
                let rhs_val = self.expr_value(module, func, frame, funcs, externs, globals, rhs);
                let rhs_val = self.convert(func, frame, rhs_val, &rhs.ty, rhs_ty);
                let cur = match &bitfield {
                    Some((bit, width, unit_ty)) => self.bitfield_load(
                        func,
                        frame,
                        addr,
                        unit_ty,
                        &lhs.ty,
                        *bit,
                        *width,
                        is_signed(unit_ty, self.sema),
                        access_volatile(lhs, &self.sema.decls),
                    ),
                    None => self.load(func, frame, addr, mir_ty.clone(), access_volatile(lhs, &self.sema.decls)),
                };
                let cur = self.convert(func, frame, cur, &lhs.ty, op_ty);
                let combined = self.compound_op(module, func, frame, funcs, externs, globals, op, cur, op_ty, rhs_val, rhs_ty, &mir_ty);
                let narrowed = self.convert(func, frame, combined, op_ty, &lhs.ty);
                match &bitfield {
                    Some((bit, width, unit_ty)) => {
                        let unit = self.convert(func, frame, narrowed, &lhs.ty, unit_ty);
                        self.bitfield_store(func, frame, addr, unit_ty, unit, *bit, *width, access_volatile(lhs, &self.sema.decls));
                        self.bitfield_load(
                            func,
                            frame,
                            addr,
                            unit_ty,
                            &lhs.ty,
                            *bit,
                            *width,
                            is_signed(unit_ty, self.sema),
                            access_volatile(lhs, &self.sema.decls),
                        )
                    }
                    None => {
                        self.store(func, frame, addr, narrowed, access_volatile(lhs, &self.sema.decls));
                        narrowed
                    }
                }
            }
            fe::check::CheckedExprKind::Call { func: callee, args } => {
                // A call naming a function entity (`f()`, `(*f)()`) is
                // direct; anything else (a function-pointer variable,
                // member, or arbitrary expression) evaluates to a code
                // address for an indirect call.
                let direct = match &callee.kind {
                    fe::check::CheckedExprKind::Ident(id) => matches!(self.sema.decls[id.0 as usize].kind, fe::sema::BindingKind::Func).then_some(*id),
                    fe::check::CheckedExprKind::Unary(fe::parse::UnaryOp::Deref, inner) => match &inner.kind {
                        fe::check::CheckedExprKind::Ident(id) => matches!(self.sema.decls[id.0 as usize].kind, fe::sema::BindingKind::Func).then_some(*id),
                        _ => None,
                    },
                    _ => None,
                };
                if let Some(id) = direct {
                    // `va_start`/`va_end`/`va_copy` are calls to
                    // compiler builtins by name; they lower to dedicated
                    // MIR ops.
                    if let Some(builtin) = self.va_builtin(id) {
                        return self.lower_va_builtin(module, func, frame, funcs, externs, globals, builtin, args, &mir_ty);
                    }
                    // Single-instruction compiler builtins
                    // (`__builtin_clzll`, `__builtin_ceil`, ...): no
                    // library symbol stands behind them, so they lower
                    // inline like the legacy backend instead of calling
                    // out.
                    if let Some(builtin) = self.scalar_builtin(id) {
                        return self.lower_scalar_builtin(module, func, frame, funcs, externs, globals, builtin, args, &expr.ty, &mir_ty);
                    }
                    // NaN/infinity constructors (`__builtin_nanf`,
                    // `__builtin_inff`, ...): no library symbol stands
                    // behind them either (the runtime ships no stubs),
                    // so they fold to constants inline. The tag argument
                    // of `nan*` is a string literal without effects.
                    if let Some(value) = self.fp_const_builtin(id, &mir_ty, func) {
                        return value;
                    }
                    // Complex accessors and conjugates
                    // (`__builtin_creal/cimag/conj/cabs` and the glibc
                    // `creal/cimag/conj/cabs` spellings): lowered
                    // directly off the components, so they work
                    // without libm. Only exact signatures intercept,
                    // so same-named user functions still call out
                    // normally.
                    if self.complex_builtin(id).is_some() {
                        if let Some(value) = self.lower_complex_builtin(module, func, frame, funcs, externs, globals, id, args) {
                            return value;
                        }
                    }
                    if let Some(target) = funcs.get(&id) {
                        let target = *target;
                        return self.emit_call(
                            module,
                            func,
                            frame,
                            funcs,
                            externs,
                            globals,
                            args,
                            &mir_ty,
                            mir::InstKind::Call {
                                callee: target,
                                args: Default::default(),
                            },
                        );
                    }
                    // Prototype-only functions are foreign imports.
                    let Some(target) = externs.get(&id) else {
                        let binding = &self.sema.decls[id.0 as usize];
                        let name = binding.name.and_then(|n| self.interner.resolve(n)).unwrap_or("<anon>");
                        panic!("call to undefined function '{name}'");
                    };
                    let target = *target;
                    // The runtime `setjmp`: the function can be
                    // re-entered here by `longjmp`, so passes must keep
                    // its slots in memory.
                    if self.sema.decls[id.0 as usize].name.and_then(|n| self.interner.resolve(n)) == Some("__glccrt_setjmp") {
                        func.returns_twice = true;
                    }
                    return self.emit_call(
                        module,
                        func,
                        frame,
                        funcs,
                        externs,
                        globals,
                        args,
                        &mir_ty,
                        mir::InstKind::CallExtern { target, args: Default::default() },
                    );
                }
                let callee_addr = self.expr_value(module, func, frame, funcs, externs, globals, callee);
                let sig = self.indirect_sig(callee);
                return self.emit_call(
                    module,
                    func,
                    frame,
                    funcs,
                    externs,
                    globals,
                    args,
                    &mir_ty,
                    mir::InstKind::CallIndirect {
                        callee: callee_addr,
                        args: Default::default(),
                        sig,
                    },
                );
            }
            fe::check::CheckedExprKind::Await(inner) => {
                // The handle is any expression (typically an async
                // call, but a stored `_Future(T)` works too); evaluate
                // it, then mark the suspension. The coroutine pass
                // rewrites the marker into the poll loop and the
                // value into a load of the completed child's frame.
                let fut = self.expr_value(module, func, frame, funcs, externs, globals, inner);
                if mir_ty.is_void() {
                    self.emit_unit(func, frame, mir::InstKind::Await { fut });
                    self.iconst(func, 32, 0)
                } else {
                    self.emit(func, frame, mir::InstKind::Await { fut }, mir_ty)
                }
            }
            fe::check::CheckedExprKind::Ternary { cond, then, els } => {
                // A void ternary (`c ? (void)0 : f()`) has no value to
                // route: evaluate both arms for side effects only and
                // yield the same dummy a void call does.
                let slot = if mir_ty.is_void() {
                    None
                } else {
                    // Allocate the result slot before branching: it must
                    // live in a block dominating both arms.
                    Some(self.alloca(func, frame, mir_ty.clone()))
                };
                let test = self.bool_of(module, func, frame, funcs, externs, globals, cond);
                let then_block = func.push_block(vec![]);
                let else_block = func.push_block(vec![]);
                let merge = func.push_block(vec![]);
                self.terminate(
                    func,
                    frame,
                    mir::Terminator::CondBr {
                        cond: test,
                        then_block,
                        else_block,
                        then_args: Default::default(),
                        else_args: Default::default(),
                    },
                );
                frame.current = Some(then_block);
                let then_val = self.expr_value(module, func, frame, funcs, externs, globals, then);
                if let Some(slot) = slot {
                    self.store_ternary_arm(func, frame, slot, then_val, &then.ty, &expr.ty);
                }
                let then_open = frame.current.is_some();
                if then_open {
                    self.branch(func, frame, merge, vec![]);
                }
                frame.current = Some(else_block);
                let else_val = self.expr_value(module, func, frame, funcs, externs, globals, els);
                if let Some(slot) = slot {
                    self.store_ternary_arm(func, frame, slot, else_val, &els.ty, &expr.ty);
                }
                let else_open = frame.current.is_some();
                if else_open {
                    self.branch(func, frame, merge, vec![]);
                }
                self.join(func, frame, merge, then_open || else_open);
                match slot {
                    Some(slot) => self.load(func, frame, slot, mir_ty, false),
                    None => self.iconst(func, 32, 0),
                }
            }
            fe::check::CheckedExprKind::Elvis { cond, els } => {
                // The condition evaluates once: its value (converted to
                // the result type) goes into the slot before the branch
                // and comes back out when nonzero.
                let slot = self.alloca(func, frame, mir_ty.clone());
                let cond_val = self.expr_value(module, func, frame, funcs, externs, globals, cond);
                let test = self.bool_of_value(func, frame, cond_val, &cond.ty);
                self.store_ternary_arm(func, frame, slot, cond_val, &cond.ty, &expr.ty);
                let then_block = func.push_block(vec![]);
                let else_block = func.push_block(vec![]);
                let merge = func.push_block(vec![]);
                self.terminate(
                    func,
                    frame,
                    mir::Terminator::CondBr {
                        cond: test,
                        then_block,
                        else_block,
                        then_args: Default::default(),
                        else_args: Default::default(),
                    },
                );
                frame.current = Some(then_block);
                let then_open = frame.current.is_some();
                if then_open {
                    self.branch(func, frame, merge, vec![]);
                }
                frame.current = Some(else_block);
                let else_val = self.expr_value(module, func, frame, funcs, externs, globals, els);
                self.store_ternary_arm(func, frame, slot, else_val, &els.ty, &expr.ty);
                let else_open = frame.current.is_some();
                if else_open {
                    self.branch(func, frame, merge, vec![]);
                }
                self.join(func, frame, merge, then_open || else_open);
                self.load(func, frame, slot, mir_ty, false)
            }
            fe::check::CheckedExprKind::Member {
                bitfield: Some((bit, width)),
                decl_ty,
                ..
            } => {
                // Bit-field reads extract from the covering unit; the
                // address resolves like any member.
                let (addr, _) = self.place(module, func, frame, funcs, externs, globals, expr);
                self.bitfield_load(
                    func,
                    frame,
                    addr,
                    decl_ty,
                    &expr.ty,
                    *bit,
                    *width,
                    is_signed(decl_ty, self.sema),
                    access_volatile(expr, &self.sema.decls),
                )
            }
            fe::check::CheckedExprKind::Member { .. } => {
                let (addr, _) = self.place(module, func, frame, funcs, externs, globals, expr);
                let array = self.expr_denotes_array(expr);
                self.rvalue_of_place(func, frame, addr, array, mir_ty, access_volatile(expr, &self.sema.decls))
            }
            fe::check::CheckedExprKind::Subscript { .. } => {
                let (addr, _) = self.place(module, func, frame, funcs, externs, globals, expr);
                let array = self.expr_denotes_array(expr);
                self.rvalue_of_place(func, frame, addr, array, mir_ty, access_volatile(expr, &self.sema.decls))
            }
            fe::check::CheckedExprKind::CompoundLiteral { .. } => {
                let (addr, _) = self.place(module, func, frame, funcs, externs, globals, expr);
                let array = self.expr_denotes_array(expr);
                self.rvalue_of_place(func, frame, addr, array, mir_ty, access_volatile(expr, &self.sema.decls))
            }
            fe::check::CheckedExprKind::Str(parts) => {
                let addr = self.string_literal(module, func, frame, parts);
                // String literals decay to their element pointer at
                // creation: every use wants the element pointer, never
                // the array.
                self.emit(
                    func,
                    frame,
                    mir::InstKind::GetElementPtr {
                        base: addr,
                        indices: [mir::GepIndex::ConstIndex(0)].into_iter().collect(),
                    },
                    mir_ty,
                )
            }
            fe::check::CheckedExprKind::LabelAddr(name) => {
                // The label's "address" is its id: computed gotos
                // dispatch on it, and arithmetic between ids of one
                // function stays consecutive (`&&b - &&a` counts
                // labels, like GCC's byte offsets count jumps).
                let Some(id) = self.sema.label_ids.get(name).copied() else {
                    let text = self.interner.resolve(*name).unwrap_or("?");
                    panic!("label address taken for undefined label '{text}'");
                };
                self.iconst(func, 64, id as i128)
            }
            fe::check::CheckedExprKind::VaArg { ap, .. } => {
                let (addr, _) = self.place(module, func, frame, funcs, externs, globals, ap);
                self.emit(func, frame, mir::InstKind::VaArg { ap: addr }, mir_ty)
            }
            fe::check::CheckedExprKind::StmtExpr { items } => {
                // GNU statement expression: lower items in order like
                // a block; the value is the trailing expression
                // statement. Labels around it hand off to
                // `lower_value_stmt`, which still defines their jump
                // blocks (`&&label` scans `frame.labels`).
                // (Shadowing is safe: every declaration gets a fresh
                // `DeclId`, so inner homes never clobber outer
                // ones.)
                let n = items.len();
                let has_value = fe::parse::stmt_expr_value(items).is_some();
                let mut result = None;
                for (i, item) in items.iter().enumerate() {
                    match item {
                        fe::parse::BlockItem::Decl { ids, .. } => {
                            for id in ids {
                                self.lower_local(module, func, frame, funcs, externs, globals, *id);
                            }
                        }
                        fe::parse::BlockItem::Stmt(stmt) => {
                            if i + 1 == n && has_value {
                                result = self.lower_value_stmt(module, func, frame, funcs, externs, globals, stmt);
                                continue;
                            }
                            self.lower_stmt(module, func, frame, funcs, externs, globals, stmt);
                        }
                    }
                }
                // No trailing expression: void-like dummy, like void
                // calls and void ternary arms.
                result.unwrap_or_else(|| self.iconst(func, 32, 0))
            }
            fe::check::CheckedExprKind::SizeofTy(ty) | fe::check::CheckedExprKind::AlignofTy(ty) => {
                let is_size = matches!(expr.kind, fe::check::CheckedExprKind::SizeofTy(_));
                if is_size && Self::ty_is_vm(ty) {
                    return self.vla_type_size(module, func, frame, funcs, externs, globals, ty);
                }
                let laid = if is_size {
                    fe::eval::sizeof_of(self.sema, ty, self.interner)
                } else {
                    fe::eval::align_of(self.sema, ty, self.interner)
                };
                let Some(n) = laid else {
                    todo!("layout of variable-length type");
                };
                self.iconst(func, int_bits(&mir_ty), n as i128)
            }
            fe::check::CheckedExprKind::SizeofExpr(inner) | fe::check::CheckedExprKind::AlignofExpr(inner) => {
                let is_size = matches!(expr.kind, fe::check::CheckedExprKind::SizeofExpr(_));
                if is_size && Self::ty_is_vm(&inner.ty) {
                    return self.vla_sizeof(module, func, frame, funcs, externs, globals, inner);
                }
                let laid = if is_size {
                    fe::eval::sizeof_of(self.sema, &inner.ty, self.interner)
                } else {
                    fe::eval::align_of_expr(self.sema, inner, self.interner)
                };
                let Some(n) = laid else {
                    todo!("layout of variable-length type");
                };
                self.iconst(func, int_bits(&mir_ty), n as i128)
            }
            fe::check::CheckedExprKind::Offsetof { offset, .. } => self.iconst(func, int_bits(&mir_ty), *offset as i128),
            fe::check::CheckedExprKind::Generic { index, assocs, .. } => {
                let Some((_, chosen)) = assocs.get(*index) else {
                    panic!("generic association index out of range");
                };
                self.expr_value(module, func, frame, funcs, externs, globals, chosen)
            }
            fe::check::CheckedExprKind::ConstantP(b) => self.iconst(func, int_bits(&mir_ty), i128::from(*b)),
            fe::check::CheckedExprKind::ClassifyType(code) => self.iconst(func, int_bits(&mir_ty), i128::from(*code)),
            fe::check::CheckedExprKind::FpClass { op, arg } => {
                return self.lower_fp_class(module, func, frame, funcs, externs, globals, op, arg, &mir_ty);
            }
            fe::check::CheckedExprKind::FpClassify { codes, arg } => {
                return self.lower_fp_classify(module, func, frame, funcs, externs, globals, codes, arg, &mir_ty);
            }
            fe::check::CheckedExprKind::Atomic { op, args } => {
                return self.lower_atomic(module, func, frame, funcs, externs, globals, op, args, &expr.ty, &mir_ty);
            }
            fe::check::CheckedExprKind::Overflow { op, lhs, rhs, res, res_ty } => {
                return self.lower_overflow(module, func, frame, funcs, externs, globals, op, lhs, rhs, res, res_ty, &mir_ty);
            }
            fe::check::CheckedExprKind::Alloca { size } => {
                return self.lower_alloca(module, func, frame, funcs, externs, globals, size);
            }
            fe::check::CheckedExprKind::FrameAddress { level, returns_address } => {
                // [%rbp] holds the caller's frame pointer and [%rbp + 8]
                // the return address, so level N walks N links. GCC
                // requires a constant level (checked), so this unrolls.
                let ptr_ty = mir::Ty::ptr(mir::Ty::Void);
                let mut fp = self.emit(func, frame, mir::InstKind::GetFramePointer, ptr_ty.clone());
                for _ in 0..*level {
                    fp = self.load(func, frame, fp, ptr_ty.clone(), false);
                }
                if *returns_address {
                    let (addr, _) = self.ptr_offset_addr(func, frame, fp, 8, mir::Ty::Void);
                    fp = self.load(func, frame, addr, ptr_ty, false);
                }
                return fp;
            }
            fe::check::CheckedExprKind::Clz { ctz, arg, fallback } => {
                // Type-generic count at the operand's own width (no
                // promotion): the count converts to the declared `int`,
                // and a zero operand yields the fallback when given.
                let operand_mir = lower_ty(&arg.ty, self.sema, self.interner);
                let value = self.expr_value(module, func, frame, funcs, externs, globals, arg);
                let count = if matches!(operand_mir, mir::Ty::Bool) {
                    // A `_Bool` is one bit: its only nonzero value has
                    // no leading or trailing zeros, and zero counts
                    // one (the byte-wide `clz` would say seven).
                    let one = self.iconst(func, int_bits(&operand_mir), 1);
                    self.emit(func, frame, mir::InstKind::Sub(one, value), operand_mir.clone())
                } else {
                    let kind = if *ctz { mir::InstKind::Ctz(value) } else { mir::InstKind::Clz(value) };
                    self.emit(func, frame, kind, operand_mir.clone())
                };
                let count = self.convert(func, frame, count, &arg.ty, &expr.ty);
                let Some(fallback) = fallback else {
                    return count;
                };
                let nonzero = self.bool_of_value(func, frame, value, &arg.ty);
                let fallback = self.expr_value(module, func, frame, funcs, externs, globals, fallback);
                self.emit(
                    func,
                    frame,
                    mir::InstKind::Select {
                        cond: nonzero,
                        then_val: count,
                        else_val: fallback,
                    },
                    mir_ty,
                )
            }
            other => todo!("expression lowering for {other:?}"),
        }
    }

    /// Floating-point classification (`__builtin_isnan`, `isinf`,
    /// `isinf_sign`, `isfinite`, `isnormal`, `signbit`): `isnan(x)` is
    /// `x != x` (only NaN is unordered with itself); the infinity tests
    /// compare the magnitude against `+inf`, and the finite/normal
    /// tests combine that with the smallest normal. Non-float operands
    /// are never NaN, infinite or subnormal (constant ones fold at
    /// check time), but still evaluate for their side effects.
    #[allow(clippy::too_many_arguments)]
    fn lower_fp_class(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        op: &fe::check::FpPredicate,
        arg: &fe::check::CheckedExpr,
        mir_ty: &mir::Ty,
    ) -> mir::Value {
        let arg_mir = lower_ty(&arg.ty, self.sema, self.interner);
        if !matches!(arg_mir, mir::Ty::Float(_)) {
            // Integers fold to `ConstantP(false)` at check time, so
            // only complex reaches here: like the legacy backend
            // (whose scalar type query panics), classification over a
            // pair has no lowering yet.
            if is_complex(&arg.ty) {
                panic!("floating classification of a complex operand");
            }
            let _ = self.expr_value(module, func, frame, funcs, externs, globals, arg);
            return self.iconst(func, int_bits(mir_ty), 0);
        }
        let value = self.expr_value(module, func, frame, funcs, externs, globals, arg);
        match op {
            fe::check::FpPredicate::IsNan => {
                let test = self.emit(
                    func,
                    frame,
                    mir::InstKind::Fcmp {
                        pred: mir::FcmpPred::Ne,
                        lhs: value,
                        rhs: value,
                    },
                    mir::Ty::Bool,
                );
                self.convert_bool(func, frame, test, mir_ty)
            }
            fe::check::FpPredicate::IsInf => {
                let magnitude = self.float_magnitude(func, frame, value, &arg_mir);
                let inf = func.push_const(mir::Const::Float {
                    kind: float_kind(&arg_mir),
                    bits: float_inf_bits(&arg_mir),
                });
                let test = self.emit(
                    func,
                    frame,
                    mir::InstKind::Fcmp {
                        pred: mir::FcmpPred::Eq,
                        lhs: magnitude,
                        rhs: inf,
                    },
                    mir::Ty::Bool,
                );
                self.convert_bool(func, frame, test, mir_ty)
            }
            fe::check::FpPredicate::IsInfSign => {
                let pinf = func.push_const(mir::Const::Float {
                    kind: float_kind(&arg_mir),
                    bits: float_inf_bits(&arg_mir),
                });
                let ninf = self.emit(func, frame, mir::InstKind::FNeg(pinf), arg_mir);
                let is_pos = self.emit(
                    func,
                    frame,
                    mir::InstKind::Fcmp {
                        pred: mir::FcmpPred::Eq,
                        lhs: value,
                        rhs: pinf,
                    },
                    mir::Ty::Bool,
                );
                let is_neg = self.emit(
                    func,
                    frame,
                    mir::InstKind::Fcmp {
                        pred: mir::FcmpPred::Eq,
                        lhs: value,
                        rhs: ninf,
                    },
                    mir::Ty::Bool,
                );
                let pos = self.convert_bool(func, frame, is_pos, &mir::Ty::int(32));
                let neg = self.convert_bool(func, frame, is_neg, &mir::Ty::int(32));
                let diff = self.emit(func, frame, mir::InstKind::Sub(pos, neg), mir::Ty::int(32));
                self.convert_int_width(func, frame, diff, 32, mir_ty)
            }
            fe::check::FpPredicate::IsFinite => {
                let magnitude = self.float_magnitude(func, frame, value, &arg_mir);
                let inf = func.push_const(mir::Const::Float {
                    kind: float_kind(&arg_mir),
                    bits: float_inf_bits(&arg_mir),
                });
                let not_inf = self.emit(
                    func,
                    frame,
                    mir::InstKind::Fcmp {
                        pred: mir::FcmpPred::Ne,
                        lhs: magnitude,
                        rhs: inf,
                    },
                    mir::Ty::Bool,
                );
                // NaN is "not inf" (unordered `!=` is true), so it needs
                // the ordering test to stay out.
                let ordered = self.emit(
                    func,
                    frame,
                    mir::InstKind::Fcmp {
                        pred: mir::FcmpPred::Ord,
                        lhs: value,
                        rhs: value,
                    },
                    mir::Ty::Bool,
                );
                let test = self.emit(func, frame, mir::InstKind::And(ordered, not_inf), mir::Ty::Bool);
                self.convert_bool(func, frame, test, mir_ty)
            }
            fe::check::FpPredicate::IsNormal => {
                let magnitude = self.float_magnitude(func, frame, value, &arg_mir);
                let kind = float_kind(&arg_mir);
                let inf = func.push_const(mir::Const::Float {
                    kind,
                    bits: float_inf_bits(&arg_mir),
                });
                let min_normal = func.push_const(mir::Const::Float {
                    kind,
                    bits: float_min_normal_bits(&arg_mir),
                });
                let zero = func.push_const(mir::Const::Float { kind, bits: 0 });
                let ge_min = self.emit(
                    func,
                    frame,
                    mir::InstKind::Fcmp {
                        pred: mir::FcmpPred::Ge,
                        lhs: magnitude,
                        rhs: min_normal,
                    },
                    mir::Ty::Bool,
                );
                let not_inf = self.emit(
                    func,
                    frame,
                    mir::InstKind::Fcmp {
                        pred: mir::FcmpPred::Ne,
                        lhs: magnitude,
                        rhs: inf,
                    },
                    mir::Ty::Bool,
                );
                let nonzero = self.emit(
                    func,
                    frame,
                    mir::InstKind::Fcmp {
                        pred: mir::FcmpPred::Ne,
                        lhs: value,
                        rhs: zero,
                    },
                    mir::Ty::Bool,
                );
                let rest = self.emit(func, frame, mir::InstKind::And(not_inf, nonzero), mir::Ty::Bool);
                let test = self.emit(func, frame, mir::InstKind::And(ge_min, rest), mir::Ty::Bool);
                self.convert_bool(func, frame, test, mir_ty)
            }
            fe::check::FpPredicate::SignBit => {
                // The sign bit is the top bit of the integer form:
                // bit 31/63/127 of the IEEE widths, and bit 79 for
                // x87 long doubles (whose register form keeps the
                // 80-bit pattern padded to 128 bits). Shifting it
                // down tests one bit without caring where it sits.
                let (int_ty, top) = float_int_form(&arg_mir);
                let bits = self.emit(func, frame, mir::InstKind::Bitcast(value), int_ty.clone());
                let shift = self.iconst(func, int_bits(&int_ty), i128::from(top));
                let sign = self.emit(func, frame, mir::InstKind::LShr(bits, shift), int_ty.clone());
                let zero = self.iconst(func, int_bits(&int_ty), 0);
                let test = self.emit(
                    func,
                    frame,
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Ne,
                        lhs: sign,
                        rhs: zero,
                    },
                    mir::Ty::Bool,
                );
                self.convert_bool(func, frame, test, mir_ty)
            }
        }
    }

    /// `__builtin_fpclassify(nan, inf, normal, subnormal, zero, x)`: the
    /// category code for `x`, selecting among the five `int` codes. The
    /// categories come from quiet comparisons: NaN is unordered with
    /// itself, infinity is `|x| == +inf`, zero is `x == 0`, and normal
    /// is a magnitude of at least the smallest normal; subnormal is the
    /// remaining case. The magnitude tests are false for NaN, so the
    /// select order (NaN first) resolves every overlap.
    #[allow(clippy::too_many_arguments)]
    fn lower_fp_classify(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        codes: &[Box<fe::check::CheckedExpr>; 5],
        arg: &fe::check::CheckedExpr,
        mir_ty: &mir::Ty,
    ) -> mir::Value {
        let code: Vec<mir::Value> = codes.iter().map(|code| self.expr_value(module, func, frame, funcs, externs, globals, code)).collect();
        let arg_mir = lower_ty(&arg.ty, self.sema, self.interner);
        let value = self.expr_value(module, func, frame, funcs, externs, globals, arg);
        let kind = float_kind(&arg_mir);
        let magnitude = self.float_magnitude(func, frame, value, &arg_mir);
        let inf = func.push_const(mir::Const::Float {
            kind,
            bits: float_inf_bits(&arg_mir),
        });
        let min_normal = func.push_const(mir::Const::Float {
            kind,
            bits: float_min_normal_bits(&arg_mir),
        });
        let zero = func.push_const(mir::Const::Float { kind, bits: 0 });
        let is_nan = self.emit(
            func,
            frame,
            mir::InstKind::Fcmp {
                pred: mir::FcmpPred::Ne,
                lhs: value,
                rhs: value,
            },
            mir::Ty::Bool,
        );
        let is_inf = self.emit(
            func,
            frame,
            mir::InstKind::Fcmp {
                pred: mir::FcmpPred::Eq,
                lhs: magnitude,
                rhs: inf,
            },
            mir::Ty::Bool,
        );
        let is_zero = self.emit(
            func,
            frame,
            mir::InstKind::Fcmp {
                pred: mir::FcmpPred::Eq,
                lhs: value,
                rhs: zero,
            },
            mir::Ty::Bool,
        );
        let ge_min = self.emit(
            func,
            frame,
            mir::InstKind::Fcmp {
                pred: mir::FcmpPred::Ge,
                lhs: magnitude,
                rhs: min_normal,
            },
            mir::Ty::Bool,
        );
        let not_inf = self.emit(
            func,
            frame,
            mir::InstKind::Fcmp {
                pred: mir::FcmpPred::Ne,
                lhs: magnitude,
                rhs: inf,
            },
            mir::Ty::Bool,
        );
        let nonzero = self.emit(
            func,
            frame,
            mir::InstKind::Fcmp {
                pred: mir::FcmpPred::Ne,
                lhs: value,
                rhs: zero,
            },
            mir::Ty::Bool,
        );
        let finite = self.emit(func, frame, mir::InstKind::And(not_inf, nonzero), mir::Ty::Bool);
        let is_normal = self.emit(func, frame, mir::InstKind::And(ge_min, finite), mir::Ty::Bool);
        let mut result = code[3];
        for (cond, code) in [(is_normal, code[2]), (is_zero, code[4]), (is_inf, code[1]), (is_nan, code[0])] {
            result = self.emit(
                func,
                frame,
                mir::InstKind::Select {
                    cond,
                    then_val: code,
                    else_val: result,
                },
                mir_ty.clone(),
            );
        }
        result
    }

    /// The magnitude of a float value, its sign bit cleared (like
    /// `fabs`, but bit-level so `-0.0` comes back positive and NaN
    /// keeps its payload).
    fn float_magnitude(&self, func: &mut mir::Function, frame: &mut Frame, value: mir::Value, operand_mir: &mir::Ty) -> mir::Value {
        let (int_ty, top) = float_int_form(operand_mir);
        let bits = int_bits(&int_ty);
        let raw = self.emit(func, frame, mir::InstKind::Bitcast(value), int_ty.clone());
        let one = self.iconst(func, bits, 1);
        let shift = self.iconst(func, bits, i128::from(top));
        let sign = self.emit(func, frame, mir::InstKind::Shl(one, shift), int_ty.clone());
        let set = self.emit(func, frame, mir::InstKind::And(raw, sign), int_ty.clone());
        let cleared = self.emit(func, frame, mir::InstKind::Xor(raw, set), int_ty.clone());
        self.emit(func, frame, mir::InstKind::Bitcast(cleared), operand_mir.clone())
    }

    /// One operand of an overflow builtin widened to the i128 the
    /// exact computation runs in, extended by its own signedness.
    fn overflow_operand(&self, func: &mut mir::Function, frame: &mut Frame, value: mir::Value, ty: &fe::parse::Ty) -> mir::Value {
        let wide = mir::Ty::int(128);
        if int_bits(&lower_ty(ty, self.sema, self.interner)) == 128 {
            return value;
        }
        let kind = if is_signed(ty, self.sema) {
            mir::InstKind::Sext(value)
        } else {
            mir::InstKind::Zext(value)
        };
        self.emit(func, frame, kind, wide)
    }

    /// A comparison yielding a `Bool`.
    fn icmp(&self, func: &mut mir::Function, frame: &mut Frame, pred: mir::IcmpPred, lhs: mir::Value, rhs: mir::Value) -> mir::Value {
        self.emit(func, frame, mir::InstKind::Icmp { pred, lhs, rhs }, mir::Ty::Bool)
    }

    /// GCC's overflow builtins: the operation runs exactly in i128
    /// (every operand value and product of 64-bit or narrower types
    /// fits), the wrapped result stores through the pointer, and the
    /// return value says whether the exact value fit the pointed-to
    /// type. The one exception is `u64 * u64`, whose exact product
    /// exceeds i128: it stays a u128 and compares unsigned.
    #[allow(clippy::too_many_arguments)]
    fn lower_overflow(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        op: &fe::check::OverflowOp,
        lhs: &fe::check::CheckedExpr,
        rhs: &fe::check::CheckedExpr,
        res: &fe::check::CheckedExpr,
        res_ty: &fe::parse::Ty,
        mir_ty: &mir::Ty,
    ) -> mir::Value {
        let a = self.expr_value(module, func, frame, funcs, externs, globals, lhs);
        let b = self.expr_value(module, func, frame, funcs, externs, globals, rhs);
        let addr = self.expr_value(module, func, frame, funcs, externs, globals, res);
        let a = self.overflow_operand(func, frame, a, &lhs.ty);
        let b = self.overflow_operand(func, frame, b, &rhs.ty);
        let wide = mir::Ty::int(128);
        let value = match op {
            fe::check::OverflowOp::Add => self.emit(func, frame, mir::InstKind::Add(a, b), wide.clone()),
            fe::check::OverflowOp::Sub => self.emit(func, frame, mir::InstKind::Sub(a, b), wide.clone()),
            fe::check::OverflowOp::Mul => self.emit(func, frame, mir::InstKind::Mul(a, b), wide.clone()),
        };
        let res_mir = lower_ty(res_ty, self.sema, self.interner);
        let res_bits = int_bits(&res_mir);
        let signed = is_signed(res_ty, self.sema);
        let unsigned_product = matches!(op, fe::check::OverflowOp::Mul)
            && !is_signed(&lhs.ty, self.sema)
            && !is_signed(&rhs.ty, self.sema)
            && int_bits(&lower_ty(&lhs.ty, self.sema, self.interner)) == 64
            && int_bits(&lower_ty(&rhs.ty, self.sema, self.interner)) == 64;
        let overflow = if unsigned_product {
            // A non-negative exact u128: only the result type's
            // maximum can be exceeded, compared unsigned because the
            // i128 bits of large products read as negative.
            if res_bits == 128 {
                self.iconst(func, 1, 0)
            } else {
                let max = if signed { (1i128 << (res_bits - 1)) - 1 } else { (1i128 << res_bits) - 1 };
                let hi = self.iconst(func, 128, max);
                self.icmp(func, frame, mir::IcmpPred::Ugt, value, hi)
            }
        } else if res_bits == 128 {
            // Every i128 fits a 128-bit result, and a signed one
            // always does; only unsigned results reject negatives.
            if signed {
                self.iconst(func, 1, 0)
            } else {
                let zero = self.iconst(func, 128, 0);
                self.icmp(func, frame, mir::IcmpPred::Slt, value, zero)
            }
        } else {
            let (lo, hi) = if signed {
                (-(1i128 << (res_bits - 1)), (1i128 << (res_bits - 1)) - 1)
            } else {
                (0, (1i128 << res_bits) - 1)
            };
            let hi = self.iconst(func, 128, hi);
            let lo = self.iconst(func, 128, lo);
            let too_high = self.icmp(func, frame, mir::IcmpPred::Sgt, value, hi);
            let too_low = self.icmp(func, frame, mir::IcmpPred::Slt, value, lo);
            self.emit(func, frame, mir::InstKind::Or(too_low, too_high), mir::Ty::Bool)
        };
        let stored = if res_bits == 128 {
            value
        } else {
            self.emit(func, frame, mir::InstKind::Trunc(value), res_mir)
        };
        self.store(func, frame, addr, stored, false);
        self.convert_bool(func, frame, overflow, mir_ty)
    }

    /// The module extern for `name`, declaring it on first use so a
    /// source declaration of the same function is reused instead of
    /// declared twice.
    fn extern_fn(&self, module: &mut mir::ModuleBuilder, name: &str, params: Vec<mir::Ty>, ret: mir::Ty) -> mir::Extern {
        for (id, ext) in module.module.externs.iter_enumerated() {
            if ext.name == name {
                return id;
            }
        }
        module.module.push_extern(name, mir::Signature { params, ret, variadic: false })
    }

    /// Head slot of the `alloca` block list, created empty in the entry
    /// block on first use: one list per call, however many times a loop
    /// re-runs the `alloca` that fills it.
    fn alloca_head(&self, func: &mut mir::Function, frame: &mut Frame) -> mir::Value {
        if let Some(slot) = frame.alloca_head {
            return slot;
        }
        let slot = self.alloca(func, frame, mir::Ty::ptr(mir::Ty::Void));
        let null = func.push_const(mir::Const::Null);
        let entry = func.entry;
        let inst = func.insts.push(mir::InstData {
            block: entry,
            kind: mir::InstKind::Store {
                addr: slot,
                val: null,
                flags: mir::MemFlags::plain(),
            },
            result: None,
            loc: frame.loc,
        });
        func.blocks[entry].insts.push(inst);
        frame.alloca_head = Some(slot);
        slot
    }

    /// `alloca`: a heap block whose first word links the previous one,
    /// so every return can drain the list. Sixteen bytes of header keep
    /// malloc's 16-byte alignment for the caller's bytes.
    fn lower_alloca(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        size: &fe::check::CheckedExpr,
    ) -> mir::Value {
        let ptr_ty = mir::Ty::ptr(mir::Ty::Void);
        let byte_ty = mir::Ty::ptr(mir::Ty::int(8));
        let head = self.alloca_head(func, frame);
        let size = self.expr_value(module, func, frame, funcs, externs, globals, size);
        let header = self.iconst(func, 64, 16);
        let total = self.emit(func, frame, mir::InstKind::Add(size, header), mir::Ty::int(64));
        let malloc = self.extern_fn(module, "malloc", vec![mir::Ty::int(64)], ptr_ty.clone());
        let block = self.emit(
            func,
            frame,
            mir::InstKind::CallExtern {
                target: malloc,
                args: [total].into_iter().collect(),
            },
            ptr_ty.clone(),
        );
        let prev = self.load(func, frame, head, ptr_ty.clone(), false);
        self.store(func, frame, block, prev, false);
        self.store(func, frame, head, block, false);
        let bytes = self.emit(func, frame, mir::InstKind::Bitcast(block), byte_ty);
        let sixteen = self.iconst(func, 64, 16);
        self.emit(
            func,
            frame,
            mir::InstKind::GetElementPtr {
                base: bytes,
                indices: [mir::GepIndex::Index(sixteen)].into_iter().collect(),
            },
            ptr_ty,
        )
    }

    /// Free the `alloca` list at every return. Returns keep their
    /// value through a block parameter; the walk itself is a loop over
    /// the links, so any number of allocations drains.
    fn drain_allocas(&self, module: &mut mir::ModuleBuilder, func: &mut mir::Function, frame: &mut Frame) {
        let head = frame.alloca_head;
        let slots = frame.vla_slots.clone();
        if head.is_none() && slots.is_empty() {
            return;
        }
        let ptr_ty = mir::Ty::ptr(mir::Ty::Void);
        let free = self.extern_fn(module, "free", vec![ptr_ty.clone()], mir::Ty::Void);
        let null = func.push_const(mir::Const::Null);
        let returns: Vec<mir::Block> = func
            .blocks
            .iter_enumerated()
            .filter(|(_, b)| matches!(b.term, Some(mir::Terminator::Return { .. })))
            .map(|(id, _)| id)
            .collect();
        for block in returns {
            let Some(mir::Terminator::Return { val }) = func.blocks[block].term.take() else {
                unreachable!("filtered to returns");
            };
            // VLA objects die with the function: free each site's live
            // allocation before the return.
            if !slots.is_empty() {
                frame.current = Some(block);
                for slot in &slots {
                    let p = self.load(func, frame, *slot, ptr_ty.clone(), false);
                    self.emit_unit(
                        func,
                        frame,
                        mir::InstKind::CallExtern {
                            target: free,
                            args: [p].into_iter().collect(),
                        },
                    );
                }
            }
            let Some(head) = head else {
                // No `alloca` list: restore the plain return.
                frame.current = Some(block);
                self.terminate(func, frame, mir::Terminator::Return { val });
                continue;
            };
            let ret_ty: Vec<mir::Ty> = val.map(|v| func.values[v].ty.clone()).into_iter().collect();
            let has_val = !ret_ty.is_empty();
            let args: Vec<mir::Value> = val.into_iter().collect();
            let check = func.push_block(ret_ty.clone());
            let body = func.push_block(Vec::new());
            let done = func.push_block(ret_ty);
            func.blocks[block].term = Some(mir::Terminator::Br {
                dest: check,
                args: args.iter().copied().collect(),
            });
            frame.current = Some(check);
            let head_val = self.load(func, frame, head, ptr_ty.clone(), false);
            let live = self.icmp(func, frame, mir::IcmpPred::Ne, head_val, null);
            self.terminate(
                func,
                frame,
                mir::Terminator::CondBr {
                    cond: live,
                    then_block: body,
                    else_block: done,
                    then_args: Default::default(),
                    else_args: args.iter().copied().collect(),
                },
            );
            frame.current = Some(body);
            let head_val = self.load(func, frame, head, ptr_ty.clone(), false);
            let next = self.load(func, frame, head_val, ptr_ty.clone(), false);
            self.emit_unit(
                func,
                frame,
                mir::InstKind::CallExtern {
                    target: free,
                    args: [head_val].into_iter().collect(),
                },
            );
            self.store(func, frame, head, next, false);
            let args: Vec<mir::Value> = if has_val { vec![func.blocks[check].params[0]] } else { Vec::new() };
            self.branch(func, frame, check, args);
            frame.current = Some(done);
            let val = if has_val { Some(func.blocks[done].params[0]) } else { None };
            self.terminate(func, frame, mir::Terminator::Return { val });
        }
    }

    /// Whether a type is variably modified: an array with a `[*]` or
    /// runtime bound at any depth outside a function type (mirrors
    /// `check::Sema::is_variably_modified`).
    fn ty_is_vm(ty: &fe::parse::Ty) -> bool {
        match ty.unqualified() {
            fe::parse::Ty::Array {
                bound: Some(fe::parse::ArrayBound::Star) | Some(fe::parse::ArrayBound::Vla(_)),
                ..
            } => true,
            fe::parse::Ty::Array { elem, .. } => Self::ty_is_vm(elem),
            fe::parse::Ty::Pointer { pointee, .. } => Self::ty_is_vm(pointee),
            _ => false,
        }
    }

    /// Whether `ty` names a VLA *object*: an array type (after
    /// stripping qualifiers) that is variably modified. Pointers to
    /// VLAs are not objects — an `int (*p)[n]` is eight static bytes.
    fn ty_is_vla_object(ty: &fe::parse::Ty) -> bool {
        matches!(ty.unqualified(), fe::parse::Ty::Array { .. }) && Self::ty_is_vm(ty)
    }

    /// Every `ArrayBound::Vla` index in a type, outermost dimension
    /// first (`int a[m][n]` yields `[m, n]`).
    fn vla_ids_in(ty: &fe::parse::Ty) -> Vec<u32> {
        let mut out = Vec::new();
        Self::collect_vla_ids(ty, &mut out);
        out
    }

    fn collect_vla_ids(ty: &fe::parse::Ty, out: &mut Vec<u32>) {
        if let fe::parse::Ty::Array { elem, bound, .. } = ty.unqualified() {
            if let Some(fe::parse::ArrayBound::Vla(id)) = bound {
                out.push(*id);
            }
            Self::collect_vla_ids(elem, out);
        }
    }

    /// One stored bound expression, evaluated now and widened to
    /// pointer width by its own signedness.
    fn vla_extent(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        id: u32,
    ) -> mir::Value {
        let bound = self.sema.vla_bounds[id as usize].clone();
        let v = self.expr_value(module, func, frame, funcs, externs, globals, &bound);
        self.convert(func, frame, v, &bound.ty, &fe::parse::Ty::LongLong)
    }

    /// Byte size of a variably modified type from an extent map
    /// (`ArrayBound::Vla` index to value): the ultimate element's
    /// static size times every fixed count times every mapped extent.
    fn vla_size_with(&self, func: &mut mir::Function, frame: &mut Frame, ty: &fe::parse::Ty, extents: &HashMap<u32, mir::Value>) -> mir::Value {
        let mut fixed: u64 = 1;
        let mut cur = ty.clone();
        let mut dyn_ids = Vec::new();
        while let fe::parse::Ty::Array { elem, bound, .. } = cur.unqualified().clone() {
            match bound {
                Some(fe::parse::ArrayBound::Fixed(n)) => fixed = fixed.wrapping_mul(n),
                Some(fe::parse::ArrayBound::Vla(id)) => dyn_ids.push(id),
                // Incomplete (`[]`) and `[*]` sizes are a frontend
                // constraint violation away.
                Some(fe::parse::ArrayBound::Star) | None => {
                    panic!("runtime size of an unsized array")
                }
            }
            cur = *elem;
        }
        let elem = self.sema.type_size(&cur, self.interner).expect("vla over a sized element");
        let mut total = self.iconst(func, 64, fixed.wrapping_mul(elem) as i128);
        for id in dyn_ids {
            let e = extents[&id];
            total = self.emit(func, frame, mir::InstKind::Mul(total, e), mir::Ty::int(64));
        }
        total
    }

    /// Byte size of a variably modified type, evaluating each bound
    /// expression where it is used (for `sizeof` on type-names and
    /// strides through pointers to VLAs).
    fn vla_type_size(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        ty: &fe::parse::Ty,
    ) -> mir::Value {
        let mut extents = HashMap::new();
        for id in Self::vla_ids_in(ty) {
            if extents.contains_key(&id) {
                continue;
            }
            let v = self.vla_extent(module, func, frame, funcs, externs, globals, id);
            extents.insert(id, v);
        }
        self.vla_size_with(func, frame, ty, &extents)
    }

    /// Cached extents when `e` names a VLA object: walk through row
    /// subscripts (and transparent conversions) to the declared array.
    /// Anything else (`*p`, plain pointers) evaluates bounds fresh at
    /// the use site instead.
    fn vla_cached_extents(&self, frame: &Frame, e: &fe::check::CheckedExpr) -> Option<HashMap<u32, mir::Value>> {
        let mut cur = e;
        loop {
            match &cur.kind {
                fe::check::CheckedExprKind::Subscript { base, .. } => cur = base,
                fe::check::CheckedExprKind::Cast { expr, .. } | fe::check::CheckedExprKind::Convert { expr, .. } => cur = expr,
                fe::check::CheckedExprKind::Ident(id) => {
                    return frame.vla_extents.get(id).cloned();
                }
                _ => return None,
            }
        }
    }

    /// Byte size of a `sizeof` operand with variably modified type:
    /// cached extents for an object reference, fresh evaluation
    /// otherwise.
    fn vla_sizeof(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        e: &fe::check::CheckedExpr,
    ) -> mir::Value {
        if let Some(map) = self.vla_cached_extents(frame, e) {
            self.vla_size_with(func, frame, &e.ty, &map)
        } else {
            self.vla_type_size(module, func, frame, funcs, externs, globals, &e.ty)
        }
    }

    /// The constant stride a pointer to `pointee` steps by: `void` and
    /// function pointees step one byte (GNU).
    fn pointee_size(&self, pointee: &fe::parse::Ty) -> u64 {
        match pointee.unqualified() {
            fe::parse::Ty::Void | fe::parse::Ty::Func { .. } => 1,
            _ => self.sema.type_size(pointee, self.interner).unwrap_or_else(|| panic!("sized pointee: {pointee:?}")),
        }
    }

    /// The byte stride `ptr` steps by, as a value: `pointee_size` for
    /// sized pointees, a runtime size for variably modified ones
    /// (`int (*)[h][w]`), with extents cached for a named VLA object and
    /// evaluated fresh through pointers.
    #[allow(clippy::too_many_arguments)]
    fn pointee_stride(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        ptr: &fe::check::CheckedExpr,
        pointee: &fe::parse::Ty,
    ) -> mir::Value {
        if Self::ty_is_vm(pointee) {
            return match self.vla_cached_extents(frame, ptr) {
                Some(map) => self.vla_size_with(func, frame, pointee, &map),
                None => self.vla_type_size(module, func, frame, funcs, externs, globals, pointee),
            };
        }
        self.iconst(func, 64, self.pointee_size(pointee) as i128)
    }

    /// Free one heap block through libc.
    fn call_free(&self, module: &mut mir::ModuleBuilder, func: &mut mir::Function, frame: &mut Frame, ptr: mir::Value) {
        let ptr_ty = mir::Ty::ptr(mir::Ty::Void);
        let free = self.extern_fn(module, "free", vec![ptr_ty], mir::Ty::Void);
        self.emit_unit(
            func,
            frame,
            mir::InstKind::CallExtern {
                target: free,
                args: [ptr].into_iter().collect(),
            },
        );
    }

    /// Allocate one VLA object: elaborate each bound in source order,
    /// `malloc` the total, and home the pointer in a fresh slot. The
    /// slot starts null (in the entry block) and this site's previous
    /// allocation is freed first, so a loop body — or a `goto` back
    /// before the declaration — re-elaborates without leaking. The
    /// slot is drained at every return.
    fn lower_vla_object(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        id: fe::sema::DeclId,
    ) {
        let ty = self.sema.decls[id.0 as usize].ty.clone();
        let ptr_ty = mir::Ty::ptr(mir::Ty::Void);
        let slot = self.alloca(func, frame, ptr_ty.clone());
        // The null init lands in the entry block (like every slot):
        // the declaration site may sit in a loop, where the store must
        // run once, not per iteration.
        let null = func.push_const(mir::Const::Null);
        let entry = func.entry;
        let inst = func.insts.push(mir::InstData {
            block: entry,
            kind: mir::InstKind::Store {
                addr: slot,
                val: null,
                flags: mir::MemFlags::plain(),
            },
            result: None,
            loc: frame.loc,
        });
        func.blocks[entry].insts.push(inst);
        frame.slots.insert(id, slot);
        frame.vla_slots.push(slot);
        let prev = self.load(func, frame, slot, ptr_ty.clone(), false);
        self.call_free(module, func, frame, prev);
        let mut extents = HashMap::new();
        for vla in Self::vla_ids_in(&ty) {
            let v = self.vla_extent(module, func, frame, funcs, externs, globals, vla);
            extents.insert(vla, v);
        }
        let total = self.vla_size_with(func, frame, &ty, &extents);
        let malloc = self.extern_fn(module, "malloc", vec![mir::Ty::int(64)], ptr_ty.clone());
        let base = self.emit(
            func,
            frame,
            mir::InstKind::CallExtern {
                target: malloc,
                args: [total].into_iter().collect(),
            },
            ptr_ty,
        );
        self.store(func, frame, slot, base, false);
        frame.vla_extents.insert(id, extents);
    }

    /// Address operand of an atomic builtin: the argument already
    /// denotes an address (`&x`, a decayed array), so its rvalue is
    /// the address itself, mirroring the legacy backend's
    /// evaluate-then-address convention.
    fn atomic_addr(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        e: &fe::check::CheckedExpr,
    ) -> mir::Value {
        self.expr_value(module, func, frame, funcs, externs, globals, e)
    }

    /// Result slot for a value-producing atomic instruction.
    fn atomic_result(&self, func: &mut mir::Function, ty: mir::Ty) -> mir::Value {
        func.values.push(mir::ValueData {
            ty,
            def: mir::Def::Inst(mir::Inst::from_usize(func.insts.len())),
        })
    }

    /// Install an atomic instruction, returning its result (or the unit
    /// value statements discard for resultless ones).
    fn atomic_finish(&self, func: &mut mir::Function, frame: &mut Frame, kind: mir::InstKind, result: Option<mir::Value>) -> mir::Value {
        let block = self.live_block(func, frame);
        let inst = func.insts.push(mir::InstData {
            block,
            kind,
            result,
            loc: frame.loc,
        });
        if let Some(v) = result {
            func.values[v].def = mir::Def::Inst(inst);
        }
        func.blocks[block].insts.push(inst);
        result.unwrap_or_else(|| self.iconst(func, 32, 0))
    }

    /// Pointee of an atomic pointer operand.
    fn atomic_pointee_of(e: &fe::check::CheckedExpr) -> fe::parse::Ty {
        match e.ty.unqualified() {
            fe::parse::Ty::Pointer { pointee, .. } => pointee.as_ref().clone(),
            _ => panic!("atomic operand without pointer type"),
        }
    }

    /// Lower a GCC atomic builtin call to MIR atomic instructions,
    /// mirroring the legacy backend: memory orders are ignored
    /// (sequentially consistent), trailing order/size operands still
    /// evaluate for their side effects, and non-scalar pointees copy
    /// plain bytes. Builtins returning `void` fabricate the unit value
    /// statements discard.
    #[allow(clippy::too_many_arguments)]
    fn lower_atomic(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        op: &fe::check::AtomicOp,
        args: &[fe::check::CheckedExpr],
        fe_ty: &fe::parse::Ty,
        mir_ty: &mir::Ty,
    ) -> mir::Value {
        use fe::check::AtomicOp;
        // Address of a pointer-typed operand.
        match op {
            AtomicOp::LoadN { .. } => {
                let ptr = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[0]);
                let _ = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let result = self.atomic_result(func, mir_ty.clone());
                self.atomic_finish(func, frame, mir::InstKind::AtomicLoad { addr: ptr }, Some(result))
            }
            AtomicOp::Load => {
                let ptr = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[0]);
                let ret = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[1]);
                let _ = self.expr_value(module, func, frame, funcs, externs, globals, &args[2]);
                let pointed = Self::atomic_pointee_of(&args[0]);
                if is_mir_scalar(&lower_ty(&pointed, self.sema, self.interner)) {
                    let loaded = self.emit(func, frame, mir::InstKind::AtomicLoad { addr: ptr }, lower_ty(&pointed, self.sema, self.interner));
                    self.store(func, frame, ret, loaded, ty_is_volatile(&pointed));
                } else {
                    let size = self.sema.type_size(&pointed, self.interner).expect("atomic object of complete type");
                    self.emit_unit(func, frame, mir::InstKind::Memcpy { dst: ret, src: ptr, len: size });
                }
                self.iconst(func, 32, 0)
            }
            AtomicOp::StoreN { .. } => {
                let ptr = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[0]);
                let val = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let _ = self.expr_value(module, func, frame, funcs, externs, globals, &args[2]);
                self.emit_unit(func, frame, mir::InstKind::AtomicStore { addr: ptr, val });
                self.iconst(func, 32, 0)
            }
            AtomicOp::Init => {
                let ptr = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[0]);
                let val = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                self.emit_unit(func, frame, mir::InstKind::AtomicStore { addr: ptr, val });
                self.iconst(func, 32, 0)
            }
            AtomicOp::Store => {
                let ptr = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[0]);
                let val_ptr = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[1]);
                let _ = self.expr_value(module, func, frame, funcs, externs, globals, &args[2]);
                let pointed = Self::atomic_pointee_of(&args[0]);
                if is_mir_scalar(&lower_ty(&pointed, self.sema, self.interner)) {
                    let pointed_mir = lower_ty(&pointed, self.sema, self.interner);
                    let v = self.load(func, frame, val_ptr, pointed_mir, ty_is_volatile(&Self::atomic_pointee_of(&args[1])));
                    self.emit_unit(func, frame, mir::InstKind::AtomicStore { addr: ptr, val: v });
                } else {
                    let size = self.sema.type_size(&pointed, self.interner).expect("atomic object of complete type");
                    self.emit_unit(
                        func,
                        frame,
                        mir::InstKind::Memcpy {
                            dst: ptr,
                            src: val_ptr,
                            len: size,
                        },
                    );
                }
                self.iconst(func, 32, 0)
            }
            AtomicOp::ExchangeN { .. } => {
                let ptr = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[0]);
                let val = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let _ = self.expr_value(module, func, frame, funcs, externs, globals, &args[2]);
                let result = self.atomic_result(func, mir_ty.clone());
                self.atomic_finish(
                    func,
                    frame,
                    mir::InstKind::AtomicRmw {
                        op: mir::AtomicRmwOp::Xchg,
                        addr: ptr,
                        val,
                    },
                    Some(result),
                )
            }
            AtomicOp::Exchange => {
                let ptr = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[0]);
                let val_ptr = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[1]);
                let ret = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[2]);
                let _ = self.expr_value(module, func, frame, funcs, externs, globals, &args[3]);
                let pointed = Self::atomic_pointee_of(&args[0]);
                let pointed_mir = lower_ty(&pointed, self.sema, self.interner);
                let v = self.load(func, frame, val_ptr, pointed_mir, ty_is_volatile(&Self::atomic_pointee_of(&args[1])));
                let old = self.emit(
                    func,
                    frame,
                    mir::InstKind::AtomicRmw {
                        op: mir::AtomicRmwOp::Xchg,
                        addr: ptr,
                        val: v,
                    },
                    lower_ty(&args[0].ty, self.sema, self.interner),
                );
                self.store(func, frame, ret, old, ty_is_volatile(&Self::atomic_pointee_of(&args[2])));
                self.iconst(func, 32, 0)
            }
            AtomicOp::CmpXchgN { .. } => {
                let ptr = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[0]);
                let expected_ptr = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[1]);
                let desired = self.expr_value(module, func, frame, funcs, externs, globals, &args[2]);
                for arg in &args[3..] {
                    let _ = self.expr_value(module, func, frame, funcs, externs, globals, arg);
                }
                let desired_mir = lower_ty(&args[2].ty, self.sema, self.interner);
                let exp = self.load(func, frame, expected_ptr, desired_mir.clone(), ty_is_volatile(&Self::atomic_pointee_of(&args[1])));
                let old = self.emit(
                    func,
                    frame,
                    mir::InstKind::AtomicCas {
                        addr: ptr,
                        expected: exp,
                        replacement: desired,
                    },
                    desired_mir,
                );
                self.store(func, frame, expected_ptr, old, ty_is_volatile(&Self::atomic_pointee_of(&args[1])));
                let ok = self.emit(
                    func,
                    frame,
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Eq,
                        lhs: old,
                        rhs: exp,
                    },
                    mir::Ty::Bool,
                );
                self.convert_bool(func, frame, ok, mir_ty)
            }
            AtomicOp::CmpXchg => {
                let ptr = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[0]);
                let expected_ptr = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[1]);
                let desired_ptr = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[2]);
                for arg in &args[3..] {
                    let _ = self.expr_value(module, func, frame, funcs, externs, globals, arg);
                }
                let pointed = Self::atomic_pointee_of(&args[0]);
                let pointed_mir = lower_ty(&pointed, self.sema, self.interner);
                let exp = self.load(func, frame, expected_ptr, pointed_mir.clone(), ty_is_volatile(&Self::atomic_pointee_of(&args[1])));
                let des = self.load(func, frame, desired_ptr, pointed_mir.clone(), ty_is_volatile(&Self::atomic_pointee_of(&args[2])));
                let old = self.emit(
                    func,
                    frame,
                    mir::InstKind::AtomicCas {
                        addr: ptr,
                        expected: exp,
                        replacement: des,
                    },
                    pointed_mir,
                );
                self.store(func, frame, expected_ptr, old, ty_is_volatile(&Self::atomic_pointee_of(&args[1])));
                let ok = self.emit(
                    func,
                    frame,
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Eq,
                        lhs: old,
                        rhs: exp,
                    },
                    mir::Ty::Bool,
                );
                self.convert_bool(func, frame, ok, mir_ty)
            }
            AtomicOp::Rmw { op, fetch, .. } => {
                let ptr = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[0]);
                let val = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let _ = self.expr_value(module, func, frame, funcs, externs, globals, &args[2]);
                // The operand converts to the result width first.
                let v = self.convert(func, frame, val, &args[1].ty, fe_ty);
                let signed = is_signed(fe_ty, self.sema);
                let rmw = match op {
                    fe::check::RmwOp::Add => mir::AtomicRmwOp::Add,
                    fe::check::RmwOp::Sub => mir::AtomicRmwOp::Sub,
                    fe::check::RmwOp::And => mir::AtomicRmwOp::And,
                    fe::check::RmwOp::Or => mir::AtomicRmwOp::Or,
                    fe::check::RmwOp::Xor => mir::AtomicRmwOp::Xor,
                    fe::check::RmwOp::Nand => mir::AtomicRmwOp::Nand,
                    fe::check::RmwOp::Min => {
                        if signed {
                            mir::AtomicRmwOp::Smin
                        } else {
                            mir::AtomicRmwOp::Umin
                        }
                    }
                    fe::check::RmwOp::Max => {
                        if signed {
                            mir::AtomicRmwOp::Smax
                        } else {
                            mir::AtomicRmwOp::Umax
                        }
                    }
                };
                let old = self.emit(func, frame, mir::InstKind::AtomicRmw { op: rmw, addr: ptr, val: v }, mir_ty.clone());
                if *fetch {
                    return old;
                }
                let combined = match op {
                    fe::check::RmwOp::Add => mir::InstKind::Add(old, v),
                    fe::check::RmwOp::Sub => mir::InstKind::Sub(old, v),
                    fe::check::RmwOp::And => mir::InstKind::And(old, v),
                    fe::check::RmwOp::Or => mir::InstKind::Or(old, v),
                    fe::check::RmwOp::Xor => mir::InstKind::Xor(old, v),
                    fe::check::RmwOp::Nand => {
                        let both = self.emit(func, frame, mir::InstKind::And(old, v), mir_ty.clone());
                        mir::InstKind::Not(both)
                    }
                    // `old op v` for the new value: min = old < v ? old : v.
                    fe::check::RmwOp::Min | fe::check::RmwOp::Max => {
                        let pred = match (matches!(op, fe::check::RmwOp::Min), signed) {
                            (true, true) => mir::IcmpPred::Slt,
                            (true, false) => mir::IcmpPred::Ult,
                            (false, true) => mir::IcmpPred::Sgt,
                            (false, false) => mir::IcmpPred::Ugt,
                        };
                        let cond = self.icmp(func, frame, pred, old, v);
                        mir::InstKind::Select { cond, then_val: old, else_val: v }
                    }
                };
                self.emit(func, frame, combined, mir_ty.clone())
            }
            AtomicOp::Fence { .. } => {
                let _ = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                self.emit_unit(func, frame, mir::InstKind::AtomicFence);
                self.iconst(func, 32, 0)
            }
            AtomicOp::LockFree { .. } => {
                let size_v = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                if args.len() > 1 {
                    let _ = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                }
                let t = lower_ty(&args[0].ty, self.sema, self.interner);
                let one = self.iconst(func, int_bits(&t), 1);
                let mut acc = self.emit(
                    func,
                    frame,
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Eq,
                        lhs: size_v,
                        rhs: one,
                    },
                    mir::Ty::Bool,
                );
                for n in [2i128, 4, 8] {
                    let c = self.iconst(func, int_bits(&t), n);
                    let eq = self.emit(
                        func,
                        frame,
                        mir::InstKind::Icmp {
                            pred: mir::IcmpPred::Eq,
                            lhs: size_v,
                            rhs: c,
                        },
                        mir::Ty::Bool,
                    );
                    acc = self.emit(func, frame, mir::InstKind::Or(acc, eq), mir::Ty::Bool);
                }
                self.convert_bool(func, frame, acc, mir_ty)
            }
            AtomicOp::TestAndSet => {
                let ptr = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[0]);
                let _ = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let one = self.iconst(func, 8, 1);
                let old = self.emit(
                    func,
                    frame,
                    mir::InstKind::AtomicRmw {
                        op: mir::AtomicRmwOp::Xchg,
                        addr: ptr,
                        val: one,
                    },
                    mir::Ty::int(8),
                );
                let zero = self.iconst(func, 8, 0);
                let was_set = self.emit(
                    func,
                    frame,
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Ne,
                        lhs: old,
                        rhs: zero,
                    },
                    mir::Ty::Bool,
                );
                self.convert_bool(func, frame, was_set, mir_ty)
            }
            AtomicOp::Clear => {
                let ptr = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[0]);
                let _ = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let zero = self.iconst(func, 8, 0);
                self.emit_unit(func, frame, mir::InstKind::AtomicStore { addr: ptr, val: zero });
                self.iconst(func, 32, 0)
            }
        }
    }

    /// Short-circuit `&&`/`||`: the right side evaluates only on the
    /// deciding path. The decided value crosses through a slot, since
    /// this frame never passes block arguments.
    fn logical(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        op: &fe::parse::BinaryOp,
        lhs: &fe::check::CheckedExpr,
        rhs: &fe::check::CheckedExpr,
        mir_ty: &mir::Ty,
    ) -> mir::Value {
        let is_and = matches!(op, fe::parse::BinaryOp::And);
        let slot = self.alloca(func, frame, mir_ty.clone());
        let first = self.bool_of(module, func, frame, funcs, externs, globals, lhs);
        let rhs_block = func.push_block(vec![]);
        let bypass = func.push_block(vec![]);
        let merge = func.push_block(vec![]);
        let (go, stop) = if is_and { (rhs_block, bypass) } else { (bypass, rhs_block) };
        self.terminate(
            func,
            frame,
            mir::Terminator::CondBr {
                cond: first,
                then_block: go,
                else_block: stop,
                then_args: Default::default(),
                else_args: Default::default(),
            },
        );
        // Bypass edge: the first value already decides. Both sides
        // widen from `Bool` to the C result width before storing.
        frame.current = Some(bypass);
        let first_wide = self.convert_bool(func, frame, first, mir_ty);
        self.store(func, frame, slot, first_wide, false);
        self.branch(func, frame, merge, vec![]);
        // Deciding edge: evaluate the right side.
        frame.current = Some(rhs_block);
        let second = self.bool_of(module, func, frame, funcs, externs, globals, rhs);
        let second_wide = self.convert_bool(func, frame, second, mir_ty);
        self.store(func, frame, slot, second_wide, false);
        self.branch(func, frame, merge, vec![]);
        frame.current = Some(merge);
        self.load(func, frame, slot, mir_ty.clone(), false)
    }

    /// Store one ternary arm into the result slot: scalars convert
    /// to the result type first (`cond ? p : 0` must store a full
    /// null pointer, not a 4-byte zero over garbage), aggregates
    /// copy whole (their value is the address).
    #[allow(clippy::too_many_arguments)]
    fn store_ternary_arm(&self, func: &mut mir::Function, frame: &mut Frame, slot: mir::Value, val: mir::Value, arm_fe_ty: &fe::parse::Ty, res_fe_ty: &fe::parse::Ty) {
        let arm_mir = lower_ty(arm_fe_ty, self.sema, self.interner);
        let res_mir = lower_ty(res_fe_ty, self.sema, self.interner);
        if arm_mir.is_aggregate() {
            self.emit_unit(
                func,
                frame,
                mir::InstKind::Memcpy {
                    dst: slot,
                    src: val,
                    len: res_mir.size_bytes(),
                },
            );
            return;
        }
        let narrowed = self.convert(func, frame, val, arm_fe_ty, res_fe_ty);
        self.store(func, frame, slot, narrowed, false);
    }

    /// Which `va_*` builtin a call target names, if any.
    fn va_builtin(&self, id: fe::sema::DeclId) -> Option<VaBuiltin> {
        let binding = &self.sema.decls[id.0 as usize];
        if !matches!(binding.kind, fe::sema::BindingKind::Func) {
            return None;
        }
        match binding.name.and_then(|n| self.interner.resolve(n)) {
            Some("__builtin_va_start") => Some(VaBuiltin::Start),
            Some("__builtin_va_end") => Some(VaBuiltin::End),
            Some("__builtin_va_copy") => Some(VaBuiltin::Copy),
            _ => None,
        }
    }

    /// A single-instruction compiler builtin call target: bit counts,
    /// byte swaps, float rounding, traps, and fences. These have no
    /// library symbol behind them, so they lower inline like the
    /// legacy backend instead of calling out.
    /// Which single-instruction builtin a call target names, if any.
    fn scalar_builtin(&self, id: fe::sema::DeclId) -> Option<ScalarBuiltin> {
        let binding = &self.sema.decls[id.0 as usize];
        if !matches!(binding.kind, fe::sema::BindingKind::Func) {
            return None;
        }
        match binding.name.and_then(|n| self.interner.resolve(n)) {
            Some("__builtin_clz" | "__builtin_clzl" | "__builtin_clzll") => Some(ScalarBuiltin::Clz),
            Some("__builtin_ctz" | "__builtin_ctzl" | "__builtin_ctzll") => Some(ScalarBuiltin::Ctz),
            Some("__builtin_clrsb" | "__builtin_clrsbl" | "__builtin_clrsbll") => Some(ScalarBuiltin::Clrsb),
            Some("__builtin_parity" | "__builtin_parityl" | "__builtin_parityll") => Some(ScalarBuiltin::Parity),
            Some("copysign" | "copysignf" | "copysignl" | "__builtin_copysign" | "__builtin_copysignf" | "__builtin_copysignl") => match binding.ty.unqualified() {
                fe::parse::Ty::Func { params, .. } if params.len() == 2 => Some(ScalarBuiltin::Copysign),
                _ => None,
            },
            Some("__builtin_popcount" | "__builtin_popcountl" | "__builtin_popcountll") => Some(ScalarBuiltin::Popcnt),
            Some("__builtin_bswap16" | "__builtin_bswap32" | "__builtin_bswap64") => Some(ScalarBuiltin::Bswap),
            Some("__builtin_ceil" | "__builtin_ceill") => Some(ScalarBuiltin::Ceil),
            Some("__builtin_floor" | "__builtin_floorl") => Some(ScalarBuiltin::Floor),
            Some("__builtin_trap") => Some(ScalarBuiltin::Trap),
            Some("__sync_synchronize") => Some(ScalarBuiltin::Sync),
            // The `abs` family and the libm rounding builtins expand
            // inline, like GCC: a program may still define the names,
            // but direct calls use the builtin. A declaration with a
            // different shape disables the expansion.
            Some("abs" | "__builtin_abs") => unary_builtin(binding, ScalarBuiltin::Abs),
            Some("labs" | "__builtin_labs") => unary_builtin(binding, ScalarBuiltin::Abs),
            Some("llabs" | "__builtin_llabs") => unary_builtin(binding, ScalarBuiltin::Abs),
            Some("fabs" | "fabsf" | "fabsl" | "__builtin_fabs" | "__builtin_fabsf" | "__builtin_fabsl") => unary_builtin(binding, ScalarBuiltin::Abs),
            Some("floor" | "floorf" | "floorl" | "__builtin_floorf") => unary_builtin(binding, ScalarBuiltin::Floor),
            Some("ceil" | "ceilf" | "ceill" | "__builtin_ceilf") => unary_builtin(binding, ScalarBuiltin::Ceil),
            _ => None,
        }
    }

    /// A NaN/infinity builtin call target folded to a constant of
    /// the result type, or `None`. Covers every spelling the parser
    /// declares (`nanf`/`nan`/`nanl`, `inff`/`inf`/`infl`,
    /// `huge_valf`/`huge_val`/`huge_vall`).
    fn fp_const_builtin(&self, id: fe::sema::DeclId, mir_ty: &mir::Ty, func: &mut mir::Function) -> Option<mir::Value> {
        let binding = &self.sema.decls[id.0 as usize];
        if !matches!(binding.kind, fe::sema::BindingKind::Func) {
            return None;
        }
        let name = binding.name.and_then(|n| self.interner.resolve(n))?;
        let nan = matches!(name, "__builtin_nanf" | "__builtin_nan" | "__builtin_nanl");
        let inf = matches!(
            name,
            "__builtin_inff" | "__builtin_inf" | "__builtin_infl" | "__builtin_huge_valf" | "__builtin_huge_val" | "__builtin_huge_vall"
        );
        if !nan && !inf {
            return None;
        }
        let mir::Ty::Float(kind) = mir_ty else {
            return None;
        };
        let bits = match kind {
            mir::FloatKind::F32 => u128::from(if nan { 0x7fc00000u32 } else { 0x7f800000u32 }),
            mir::FloatKind::F64 => u128::from(if nan { 0x7ff8000000000000u64 } else { 0x7ff0000000000000u64 }),
            mir::FloatKind::F80 | mir::FloatKind::F128 => {
                let ld = if matches!(kind, mir::FloatKind::F80) {
                    fe::target::LongDoubleKind::F80
                } else {
                    fe::target::LongDoubleKind::F128
                };
                let value = if nan {
                    fe::eval::FloatValue::long_double_nan_in(ld)
                } else {
                    fe::eval::FloatValue::long_double_infinity_in(ld)
                };
                float_bits(value, mir_ty)
            }
        };
        Some(func.push_const(mir::Const::Float { kind: *kind, bits }))
    }

    /// Lower a single-instruction builtin call: evaluate the operand,
    /// apply the operation at the operand width, and convert to the
    /// declared result width (64-bit counts narrow to `int`).
    #[allow(clippy::too_many_arguments)]
    fn lower_scalar_builtin(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        builtin: ScalarBuiltin,
        args: &[fe::check::CheckedExpr],
        fe_ty: &fe::parse::Ty,
        _mir_ty: &mir::Ty,
    ) -> mir::Value {
        // `trap` terminates the block, so materialize the dummy result
        // first: anything appended after a terminator panics.
        if matches!(builtin, ScalarBuiltin::Trap) {
            let dummy = self.iconst(func, 32, 0);
            self.terminate(func, frame, mir::Terminator::Unreachable);
            return dummy;
        }
        if matches!(builtin, ScalarBuiltin::Sync) {
            for arg in args {
                self.expr_value(module, func, frame, funcs, externs, globals, arg);
            }
            self.emit_unit(func, frame, mir::InstKind::AtomicFence);
            return self.iconst(func, 32, 0);
        }
        // `abs(x)` / `fabs(x)`: the magnitude with a single evaluation.
        // Floats clear the sign bit (a select would keep `-0.0`);
        // integers are `x < 0 ? -x : x`, where the negative extreme
        // wraps, like the machine sequence GCC emits.
        if matches!(builtin, ScalarBuiltin::Abs) {
            let arg = &args[0];
            let operand_mir = lower_ty(&arg.ty, self.sema, self.interner);
            let value = self.expr_value(module, func, frame, funcs, externs, globals, arg);
            if let mir::Ty::Float(_) = operand_mir {
                let out = self.float_magnitude(func, frame, value, &operand_mir);
                return self.convert(func, frame, out, &arg.ty, fe_ty);
            }
            let bits = int_bits(&operand_mir);
            let zero = self.iconst(func, bits, 0);
            let neg = self.emit(func, frame, mir::InstKind::Neg(value), operand_mir.clone());
            let is_neg = self.emit(
                func,
                frame,
                mir::InstKind::Icmp {
                    pred: mir::IcmpPred::Slt,
                    lhs: value,
                    rhs: zero,
                },
                mir::Ty::Bool,
            );
            let raw = self.emit(
                func,
                frame,
                mir::InstKind::Select {
                    cond: is_neg,
                    then_val: neg,
                    else_val: value,
                },
                operand_mir.clone(),
            );
            return self.convert(func, frame, raw, &arg.ty, fe_ty);
        }
        // `clrsb(x)`: leading redundant sign bits. `x ^ (x >> w-1)` is
        // `x` for a non-negative operand and `~x` for a negative one
        // (the shift is arithmetic), so `clz` of it counts the sign
        // extension; `clz(0) == w` makes a zero operand count `w - 1`.
        if matches!(builtin, ScalarBuiltin::Clrsb) {
            let arg = &args[0];
            let operand_mir = lower_ty(&arg.ty, self.sema, self.interner);
            let value = self.expr_value(module, func, frame, funcs, externs, globals, arg);
            let bits = (operand_mir.size_bytes() * 8) as u32;
            let shift = self.iconst(func, bits, (bits - 1) as i128);
            let sign = self.emit(func, frame, mir::InstKind::AShr(value, shift), operand_mir.clone());
            let folded = self.emit(func, frame, mir::InstKind::Xor(value, sign), operand_mir.clone());
            let leading = self.emit(func, frame, mir::InstKind::Clz(folded), operand_mir.clone());
            let one = self.iconst(func, bits, 1);
            let raw = self.emit(func, frame, mir::InstKind::Sub(leading, one), operand_mir.clone());
            return self.convert(func, frame, raw, &arg.ty, fe_ty);
        }
        // `parity(x)`: `int` 0/1 of the low bit of the popcount.
        if matches!(builtin, ScalarBuiltin::Parity) {
            let arg = &args[0];
            let operand_mir = lower_ty(&arg.ty, self.sema, self.interner);
            let value = self.expr_value(module, func, frame, funcs, externs, globals, arg);
            let bits = (operand_mir.size_bytes() * 8) as u32;
            let count = self.emit(func, frame, mir::InstKind::Popcnt(value), operand_mir.clone());
            let one = self.iconst(func, bits, 1);
            let raw = self.emit(func, frame, mir::InstKind::And(count, one), operand_mir.clone());
            return self.convert(func, frame, raw, &arg.ty, fe_ty);
        }
        // `copysign(x, y)`: x's magnitude with y's sign bit. Bit-level:
        // `(bits_x ^ (bits_x & mask)) | (bits_y & mask)` where the mask
        // is the top bit of the integer form (bit 79 for x87 long
        // doubles, whose register form keeps the 80-bit pattern padded
        // to 128 bits).
        if matches!(builtin, ScalarBuiltin::Copysign) {
            let (x, y) = (&args[0], &args[1]);
            let operand_mir = lower_ty(&x.ty, self.sema, self.interner);
            let (int_ty, top) = float_int_form(&operand_mir);
            let bits = int_bits(&int_ty);
            let xv = self.expr_value(module, func, frame, funcs, externs, globals, x);
            let yv = self.expr_value(module, func, frame, funcs, externs, globals, y);
            let xb = self.emit(func, frame, mir::InstKind::Bitcast(xv), int_ty.clone());
            let yb = self.emit(func, frame, mir::InstKind::Bitcast(yv), int_ty.clone());
            let one = self.iconst(func, bits, 1);
            let shift = self.iconst(func, bits, i128::from(top));
            let mask = self.emit(func, frame, mir::InstKind::Shl(one, shift), int_ty.clone());
            let x_sign = self.emit(func, frame, mir::InstKind::And(xb, mask), int_ty.clone());
            let cleared = self.emit(func, frame, mir::InstKind::Xor(xb, x_sign), int_ty.clone());
            let y_sign = self.emit(func, frame, mir::InstKind::And(yb, mask), int_ty.clone());
            let combined = self.emit(func, frame, mir::InstKind::Or(cleared, y_sign), int_ty.clone());
            let raw = self.emit(func, frame, mir::InstKind::Bitcast(combined), operand_mir.clone());
            return self.convert(func, frame, raw, &x.ty, fe_ty);
        }
        let arg = &args[0];
        let operand_mir = lower_ty(&arg.ty, self.sema, self.interner);
        let value = self.expr_value(module, func, frame, funcs, externs, globals, arg);
        let kind = match builtin {
            ScalarBuiltin::Clz => mir::InstKind::Clz(value),
            ScalarBuiltin::Ctz => mir::InstKind::Ctz(value),
            ScalarBuiltin::Popcnt => mir::InstKind::Popcnt(value),
            ScalarBuiltin::Bswap => mir::InstKind::Bswap(value),
            ScalarBuiltin::Ceil => mir::InstKind::Ceil(value),
            ScalarBuiltin::Floor => mir::InstKind::Floor(value),
            ScalarBuiltin::Trap | ScalarBuiltin::Sync | ScalarBuiltin::Abs | ScalarBuiltin::Clrsb | ScalarBuiltin::Parity | ScalarBuiltin::Copysign => {
                unreachable!("handled above")
            }
        };
        let raw = self.emit(func, frame, kind, operand_mir.clone());
        self.convert(func, frame, raw, &arg.ty, fe_ty)
    }

    /// Which complex accessor `id` names, if any: the real part,
    /// imaginary part, conjugate, or magnitude getters (plain, `f`/`l`
    /// suffixed, and `__builtin_` spellings).
    fn complex_builtin(&self, id: fe::sema::DeclId) -> Option<ComplexBuiltin> {
        let binding = &self.sema.decls[id.0 as usize];
        if !matches!(binding.kind, fe::sema::BindingKind::Func) {
            return None;
        }
        Some(match binding.name.and_then(|n| self.interner.resolve(n))? {
            "creal" | "crealf" | "creall" | "__builtin_creal" | "__builtin_crealf" | "__builtin_creall" => ComplexBuiltin::Real,
            "cimag" | "cimagf" | "cimagl" | "__builtin_cimag" | "__builtin_cimagf" | "__builtin_cimagl" => ComplexBuiltin::Imag,
            "conj" | "conjf" | "conjl" | "__builtin_conj" | "__builtin_conjf" | "__builtin_conjl" => ComplexBuiltin::Conj,
            "cabs" | "cabsf" | "cabsl" | "__builtin_cabs" | "__builtin_cabsf" | "__builtin_cabsl" => ComplexBuiltin::Abs,
            _ => return None,
        })
    }

    /// Lower a complex accessor call directly off the components, or
    /// `None` when the callee's signature is not the exact glibc one
    /// (so same-named user functions still call out normally).
    /// Compiler builtins keep their declared prototypes, which always
    /// match; magnitudes over integer components never match (no
    /// glibc signature takes them, mirroring the legacy backend).
    #[allow(clippy::too_many_arguments)]
    fn lower_complex_builtin(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        id: fe::sema::DeclId,
        args: &[fe::check::CheckedExpr],
    ) -> Option<mir::Value> {
        let op = self.complex_builtin(id)?;
        let name = self.sema.decls[id.0 as usize].name.and_then(|n| self.interner.resolve(n))?;
        // Compiler builtins need no signature check: the parser
        // declares their exact prototypes.
        if !name.starts_with("__builtin_") {
            let binding = &self.sema.decls[id.0 as usize];
            let fe::parse::Ty::Func { params, ret, .. } = binding.ty.unqualified() else {
                return None;
            };
            let [param] = params.as_slice() else {
                return None;
            };
            let fe::parse::Ty::Complex(inner) = param.unqualified() else {
                return None;
            };
            let comp = inner.as_ref().clone();
            let want_ret = match op {
                ComplexBuiltin::Conj => param.clone(),
                ComplexBuiltin::Real | ComplexBuiltin::Imag | ComplexBuiltin::Abs => comp.clone(),
            };
            if ret.unqualified() != want_ret.unqualified() {
                return None;
            }
            if matches!(op, ComplexBuiltin::Abs) && !matches!(comp.unqualified(), fe::parse::Ty::Float | fe::parse::Ty::Double | fe::parse::Ty::LongDouble) {
                return None;
            }
        }
        let arg = args.first()?;
        if !is_complex(&arg.ty) {
            return None;
        }
        let lowered = self.expr_value(module, func, frame, funcs, externs, globals, arg);
        match op {
            ComplexBuiltin::Real | ComplexBuiltin::Imag => {
                let index = if matches!(op, ComplexBuiltin::Real) { 0 } else { 1 };
                Some(self.complex_extract(func, frame, lowered, &arg.ty, index))
            }
            ComplexBuiltin::Conj => Some(self.complex_conj(func, frame, lowered, &arg.ty)),
            ComplexBuiltin::Abs => {
                let comp_fe = complex_component(&arg.ty);
                let comp_mir = lower_ty(&comp_fe, self.sema, self.interner);
                if !matches!(comp_mir, mir::Ty::Float(_)) {
                    return None;
                }
                let a = self.complex_extract(func, frame, lowered, &arg.ty, 0);
                let b = self.complex_extract(func, frame, lowered, &arg.ty, 1);
                let aa = self.emit(func, frame, mir::InstKind::FMul(a, a), comp_mir.clone());
                let bb = self.emit(func, frame, mir::InstKind::FMul(b, b), comp_mir.clone());
                let sum = self.emit(func, frame, mir::InstKind::FAdd(aa, bb), comp_mir.clone());
                Some(self.emit(func, frame, mir::InstKind::Sqrt(sum), comp_mir))
            }
        }
    }

    /// Lower a `va_*` builtin call to its MIR instruction. The list
    /// operand lowers to its address either way.
    fn lower_va_builtin(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        builtin: VaBuiltin,
        args: &[fe::check::CheckedExpr],
        mir_ty: &mir::Ty,
    ) -> mir::Value {
        let kind = match builtin {
            VaBuiltin::Start => mir::InstKind::VaStart {
                ap: self.place(module, func, frame, funcs, externs, globals, &args[0]).0,
            },
            VaBuiltin::End => mir::InstKind::VaEnd {
                ap: self.place(module, func, frame, funcs, externs, globals, &args[0]).0,
            },
            VaBuiltin::Copy => mir::InstKind::VaCopy {
                dst: self.place(module, func, frame, funcs, externs, globals, &args[0]).0,
                src: self.place(module, func, frame, funcs, externs, globals, &args[1]).0,
            },
        };
        // Builtins are `void`; fabricate the unit value statements discard.
        if mir_ty.is_void() {
            self.emit_unit(func, frame, kind);
            self.iconst(func, 32, 0)
        } else {
            self.emit(func, frame, kind, mir_ty.clone())
        }
    }

    /// Lower call arguments and install a call instruction of any
    /// flavor: `template` carries the callee addressing with empty
    /// args, filled here. Void calls fabricate a unit value the
    /// statement layer discards.
    #[allow(clippy::too_many_arguments)]
    fn emit_call(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        args: &[fe::check::CheckedExpr],
        mir_ty: &mir::Ty,
        template: mir::InstKind,
    ) -> mir::Value {
        let mut lowered_args = Vec::with_capacity(args.len());
        for arg in args {
            lowered_args.push(self.expr_value(module, func, frame, funcs, externs, globals, arg));
        }
        let kind = match template {
            mir::InstKind::Call { callee, .. } => {
                // A definition without a prototype fixes its own
                // arity: extra arguments still evaluate, but only the
                // declared parameters pass, and missing ones pass
                // undefined values (C89 untyped assembly). Variadic
                // callees take their tail as-is.
                let sig = &module.module.funcs[callee].sig;
                if !sig.variadic && lowered_args.len() > sig.params.len() {
                    lowered_args.truncate(sig.params.len());
                }
                while !sig.variadic && lowered_args.len() < sig.params.len() {
                    let ty = sig.params[lowered_args.len()].clone();
                    let undef = func.values.push(mir::ValueData {
                        ty,
                        def: mir::Def::Const(mir::Const::Undef),
                    });
                    lowered_args.push(undef);
                }
                mir::InstKind::Call {
                    callee,
                    args: lowered_args.into_iter().collect(),
                }
            }
            mir::InstKind::CallExtern { target, .. } => mir::InstKind::CallExtern {
                target,
                args: lowered_args.into_iter().collect(),
            },
            mir::InstKind::CallIndirect { callee, sig, .. } => mir::InstKind::CallIndirect {
                callee,
                args: lowered_args.into_iter().collect(),
                sig,
            },
            _ => panic!("emit_call needs a call template"),
        };
        let has_result = !mir_ty.is_void();
        let result = has_result.then(|| {
            func.values.push(mir::ValueData {
                ty: mir_ty.clone(),
                def: mir::Def::Inst(mir::Inst::from_usize(func.insts.len())),
            })
        });
        let block = self.live_block(func, frame);
        let inst = func.insts.push(mir::InstData {
            block,
            kind,
            result,
            loc: frame.loc,
        });
        if let Some(v) = result {
            func.values[v].def = mir::Def::Inst(inst);
        }
        func.blocks[block].insts.push(inst);
        result.unwrap_or_else(|| self.iconst(func, 32, 0))
    }

    /// The callee signature of an indirect call, from the pointed-to
    /// function type. The callee arrives as a pointer to function
    /// (through decay or a designator dereference); a bare function
    /// type covers designator positions.
    fn indirect_sig(&self, callee: &fe::check::CheckedExpr) -> mir::Signature {
        let func_ty = match callee.ty.unqualified() {
            fe::parse::Ty::Pointer { pointee, .. } => match pointee.unqualified() {
                fe::parse::Ty::Func { .. } => pointee.as_ref().clone(),
                _ => panic!("indirect call to non-function"),
            },
            fe::parse::Ty::Func { .. } => callee.ty.clone(),
            _ => panic!("indirect call to non-function"),
        };
        let fe::parse::Ty::Func { ret, params, variadic, .. } = func_ty.unqualified() else {
            panic!("indirect call to non-function");
        };
        let mut sig = mir::Signature::new(
            params.iter().map(|p| lower_ty(p, self.sema, self.interner)).collect(),
            lower_ty(ret, self.sema, self.interner),
        );
        sig.variadic = *variadic;
        sig
    }

    /// GEP with a properly typed result: walks `pointee_ty` through
    /// the steps so verification sees the structural type, not `Void`.
    fn gep(&self, func: &mut mir::Function, frame: &mut Frame, base: mir::Value, pointee_ty: &mir::Ty, indices: Vec<mir::GepIndex>) -> (mir::Value, mir::Ty) {
        let mut cur = pointee_ty.clone();
        for idx in &indices {
            cur = mir::Signature::gep_step(&cur, idx).unwrap_or_else(|| panic!("bad gep step"));
        }
        let v = self.emit(
            func,
            frame,
            mir::InstKind::GetElementPtr {
                base,
                indices: indices.into_iter().collect(),
            },
            mir::Ty::ptr(cur.clone()),
        );
        (v, cur)
    }

    /// Index a pointer by an element size known only to the frontend
    /// (struct elements, whose MIR pointees erase): `base + idx *
    /// scale` as address arithmetic, retyped to the element.
    #[allow(clippy::too_many_arguments)]
    fn scaled_index_addr(
        &self,
        func: &mut mir::Function,
        frame: &mut Frame,
        ptr: mir::Value,
        idx: mir::Value,
        idx_ty: &fe::parse::Ty,
        scale: u64,
        obj: mir::Ty,
    ) -> (mir::Value, mir::Ty) {
        let base = self.emit(func, frame, mir::InstKind::PtrToInt(ptr), mir::Ty::int(64));
        let wide = self.convert(func, frame, idx, idx_ty, &fe::parse::Ty::LongLong);
        let step = if scale == 1 {
            wide
        } else {
            let n = self.iconst(func, 64, scale as i128);
            self.emit(func, frame, mir::InstKind::Mul(wide, n), mir::Ty::int(64))
        };
        let sum = self.emit(func, frame, mir::InstKind::Add(base, step), mir::Ty::int(64));
        let out = self.emit(func, frame, mir::InstKind::IntToPtr(sum), mir::Ty::ptr(obj.clone()));
        (out, obj)
    }

    /// `scaled_index_addr` with a runtime stride: a VLA row's byte size
    /// is a value, not a layout constant.
    fn scaled_index_addr_dyn(
        &self,
        func: &mut mir::Function,
        frame: &mut Frame,
        ptr: mir::Value,
        idx: mir::Value,
        idx_ty: &fe::parse::Ty,
        scale: mir::Value,
        obj: mir::Ty,
    ) -> (mir::Value, mir::Ty) {
        let base = self.emit(func, frame, mir::InstKind::PtrToInt(ptr), mir::Ty::int(64));
        let wide = self.convert(func, frame, idx, idx_ty, &fe::parse::Ty::LongLong);
        let step = self.emit(func, frame, mir::InstKind::Mul(wide, scale), mir::Ty::int(64));
        let sum = self.emit(func, frame, mir::InstKind::Add(base, step), mir::Ty::int(64));
        let out = self.emit(func, frame, mir::InstKind::IntToPtr(sum), mir::Ty::ptr(obj.clone()));
        (out, obj)
    }

    /// Whether member descent must use byte offsets: any union level
    /// (overlaid members share offset zero), or any struct level
    /// carrying bit-fields (coalesced units shift field indices).
    /// Conservative and always correct: byte offsets are authoritative
    /// either way, so plain structs keep their field steps only when
    /// no level needs them. Only structs can appear mid-path (the
    /// checker rejects anything else), so anything else ends the walk
    /// negatively.
    fn descent_needs_bytes(&self, base_ty: &fe::parse::Ty, path: &[usize]) -> bool {
        let mut ty = base_ty.clone();
        for &index in path {
            let fe::parse::Ty::Struct(id) = ty.unqualified() else {
                return false;
            };
            let def = &self.sema.structs[id.0 as usize];
            if def.is_union || def.members.iter().any(|m| m.bitfield.is_some()) {
                return true;
            }
            let Some(member) = def.members.get(index) else {
                return false;
            };
            ty = member.ty.clone();
        }
        false
    }

    /// Bit-field read: load the covering unit, shift the field to
    /// bit zero, mask to its width, and sign-extend into the value
    /// type when the field is signed. Masks compute build-side, so no
    /// shift ever overflows its type. Signedness follows the declared
    /// field type, not the promoted value type: narrow unsigned
    /// fields (`unsigned y:5`) promote to signed `int` but still read
    /// zero-extended.
    /// GCC reduces an operation on bit-fields to the widest operand's
    /// precision: `unsigned long long b : 40` wraps at 40 bits through
    /// `<<`, `+`, `-`, and so on (see `check::bitfield_precision`).
    fn reduce_bitfield_precision(&self, func: &mut mir::Function, frame: &mut Frame, value: mir::Value, expr: &fe::check::CheckedExpr, mir_ty: mir::Ty) -> mir::Value {
        let Some(bits) = fe::check::bitfield_precision(expr) else {
            return value;
        };
        self.reduce_to_bits(func, frame, value, bits, is_signed(&expr.ty, self.sema), mir_ty)
    }

    /// A value narrowed to `bits` of `mir_ty`, sign-extended when the
    /// type is signed: the stored value of a bit-field, in a register.
    fn reduce_to_bits(&self, func: &mut mir::Function, frame: &mut Frame, value: mir::Value, bits: u32, signed: bool, mir_ty: mir::Ty) -> mir::Value {
        let width = int_bits(&mir_ty);
        if bits >= width {
            return value;
        }
        if signed {
            // Sign-extend from the field width: shift up, shift back.
            let shift = self.iconst(func, width, (width - bits) as i128);
            let up = self.emit(func, frame, mir::InstKind::Shl(value, shift), mir_ty.clone());
            self.emit(func, frame, mir::InstKind::AShr(up, shift), mir_ty)
        } else {
            let mask = self.iconst(func, width, ((1u128 << bits) - 1) as i128);
            self.emit(func, frame, mir::InstKind::And(value, mask), mir_ty)
        }
    }

    /// Storage unit of a bit-field: the declared type's byte size in
    /// bits. `_Bool` is the exception — its value type is 1 bit, but
    /// its unit is the byte the layout packs the field into.
    fn bitfield_unit_ty(&self, ty: &fe::parse::Ty) -> mir::Ty {
        match lower_ty(ty, self.sema, self.interner) {
            mir::Ty::Bool => mir::Ty::int(8),
            unit => unit,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn bitfield_load(
        &self,
        func: &mut mir::Function,
        frame: &mut Frame,
        addr: mir::Value,
        unit_fe: &fe::parse::Ty,
        val_fe: &fe::parse::Ty,
        bit: u32,
        width: u32,
        signed: bool,
        vol: bool,
    ) -> mir::Value {
        let unit_mir = self.bitfield_unit_ty(unit_fe);
        let val_mir = lower_ty(val_fe, self.sema, self.interner);
        let unit_bits = int_bits(&unit_mir);
        let val_bits = int_bits(&val_mir);
        let flags = if vol { mir::MemFlags::vol() } else { mir::MemFlags::plain() };
        let raw = self.emit(func, frame, mir::InstKind::Load { addr, flags }, unit_mir.clone());
        let shifted = if bit == 0 {
            raw
        } else {
            let b = self.iconst(func, unit_bits, bit as i128);
            self.emit(func, frame, mir::InstKind::LShr(raw, b), unit_mir.clone())
        };
        let mask: u128 = if width >= unit_bits { u128::MAX } else { (1u128 << width) - 1 };
        let m = self.iconst(func, unit_bits, mask as i128);
        let masked = self.emit(func, frame, mir::InstKind::And(shifted, m), unit_mir.clone());
        let widened = if unit_bits == val_bits {
            masked
        } else if unit_bits < val_bits {
            self.emit(func, frame, mir::InstKind::Zext(masked), val_mir.clone())
        } else {
            self.emit(func, frame, mir::InstKind::Trunc(masked), val_mir.clone())
        };
        if !signed || width >= val_bits {
            return widened;
        }
        let sh = self.iconst(func, val_bits, (val_bits - width) as i128);
        let up = self.emit(func, frame, mir::InstKind::Shl(widened, sh), val_mir.clone());
        self.emit(func, frame, mir::InstKind::AShr(up, sh), val_mir)
    }

    /// Bit-field write: read-modify-write the covering unit, clamping
    /// the value to the field width first. The value arrives at the
    /// declared width (callers convert); signedness is irrelevant
    /// under the mask, so stores need none.
    fn bitfield_store(&self, func: &mut mir::Function, frame: &mut Frame, addr: mir::Value, unit_fe: &fe::parse::Ty, value: mir::Value, bit: u32, width: u32, vol: bool) {
        let unit_mir = self.bitfield_unit_ty(unit_fe);
        let unit_bits = int_bits(&unit_mir);
        // `_Bool` is a 1-bit value type whose registers hold a byte —
        // exactly its unit's width. Other values arrive at the
        // declared width (callers convert), which the unit matches.
        let value_bits = if matches!(func.values[value].ty, mir::Ty::Bool) {
            unit_bits
        } else {
            int_bits(&func.values[value].ty)
        };
        let value = if value_bits < unit_bits {
            self.emit(func, frame, mir::InstKind::Zext(value), unit_mir.clone())
        } else if value_bits > unit_bits {
            self.emit(func, frame, mir::InstKind::Trunc(value), unit_mir.clone())
        } else {
            value
        };
        let flags = if vol { mir::MemFlags::vol() } else { mir::MemFlags::plain() };
        let raw = self.emit(func, frame, mir::InstKind::Load { addr, flags }, unit_mir.clone());
        let mask: u128 = if width >= unit_bits { u128::MAX } else { (1u128 << width) - 1 };
        let m = self.iconst(func, unit_bits, mask as i128);
        let clamped = self.emit(func, frame, mir::InstKind::And(value, m), unit_mir.clone());
        let field = if bit == 0 {
            clamped
        } else {
            let b = self.iconst(func, unit_bits, bit as i128);
            self.emit(func, frame, mir::InstKind::Shl(clamped, b), unit_mir.clone())
        };
        // Holes everywhere outside the field: `mask << bit` cannot
        // overflow (the checker fits the field in its unit).
        let holes = self.iconst(func, unit_bits, !(mask << bit) as i128);
        let cleared = self.emit(func, frame, mir::InstKind::And(raw, holes), unit_mir.clone());
        let merged = self.emit(func, frame, mir::InstKind::Or(cleared, field), unit_mir.clone());
        self.store(func, frame, addr, merged, vol);
    }

    /// Bit-field store target for a member: unit type plus positions
    /// from the frontend layout, or `None` for ordinary members.
    /// Shared by every initializer and designator walk so the layout
    /// query lives once. Unnamed bit-fields resolve to `None`: they
    /// are unreachable by name, and positional items skip them.
    fn bitfield_target_of(&self, id: fe::sema::StructId, index: usize) -> Option<BitfieldTarget> {
        let def = &self.sema.structs[id.0 as usize];
        let member = def.members.get(index)?;
        let width_expr = member.bitfield.as_ref()?;
        let layout = self.sema.struct_layout(id, self.interner)?;
        let decl_size = self.sema.type_size(&member.ty, self.interner)?;
        let width = self.sema.bit_width(width_expr, decl_size * 8, self.interner)?;
        let name = member.name?;
        let ml = layout.members.iter().find(|m| m.name == name)?;
        Some(BitfieldTarget {
            unit_ty: member.ty.clone(),
            bit: ml.bit.unwrap_or(0),
            width: width as u32,
        })
    }

    /// Initialize one bit-field member: evaluate the initializer to
    /// the unit width in a temp (reusing the ordinary init paths,
    /// which know scalars and braces), then read-modify-write it.
    #[allow(clippy::too_many_arguments)]
    fn init_bitfield(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        addr: mir::Value,
        target: &BitfieldTarget,
        init: &fe::parse::Initializer,
    ) {
        let tmp = self.alloca(func, frame, lower_ty(&target.unit_ty, self.sema, self.interner));
        self.lower_init(module, func, frame, funcs, externs, globals, tmp, &target.unit_ty, init);
        let v = self.load(func, frame, tmp, lower_ty(&target.unit_ty, self.sema, self.interner), false);
        self.bitfield_store(func, frame, addr, &target.unit_ty, v, target.bit, target.width, ty_is_volatile(&target.unit_ty));
    }

    /// Address a member through a pointer: byte arithmetic off the
    /// pointer value by a frontend-computed offset. The result
    /// retypes to the addressed object so further direct steps (as in
    /// `p->s.x`) resolve against its complete type.
    fn ptr_offset_addr(&self, func: &mut mir::Function, frame: &mut Frame, ptr: mir::Value, offset: u64, obj: mir::Ty) -> (mir::Value, mir::Ty) {
        let base = self.emit(func, frame, mir::InstKind::PtrToInt(ptr), mir::Ty::int(64));
        let addr = if offset == 0 {
            base
        } else {
            let off = self.iconst(func, 64, offset as i128);
            self.emit(func, frame, mir::InstKind::Add(base, off), mir::Ty::int(64))
        };
        let out = self.emit(func, frame, mir::InstKind::IntToPtr(addr), mir::Ty::ptr(obj.clone()));
        (out, obj)
    }

    /// Retype an address value without moving it: overlaid storage
    /// (union members, brace-elided initializers) shares the base
    /// address, but MIR steps need the member's type on the value.
    fn retype_addr(&self, func: &mut mir::Function, frame: &mut Frame, addr: mir::Value, ty: mir::Ty) -> mir::Value {
        self.ptr_offset_addr(func, frame, addr, 0, ty).0
    }

    /// Address of an lvalue expression plus the addressed object type:
    /// variables resolve to slots or globals, member/subscript steps
    /// lower to GEPs, dereference to the pointer itself, and compound
    /// literals materialize a fresh slot.
    fn place(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        expr: &fe::check::CheckedExpr,
    ) -> (mir::Value, mir::Ty) {
        match &expr.kind {
            fe::check::CheckedExprKind::Ident(id) => {
                let binding_ty = self.sema.decls[id.0 as usize].ty.clone();
                let addr = self.place_of_var(func, frame, globals, *id);
                (addr, lower_ty(&binding_ty, self.sema, self.interner))
            }
            fe::check::CheckedExprKind::Member {
                base,
                path,
                arrow,
                bitfield: _,
                offset,
                decl_ty,
                ..
            } => {
                // Bit-fields need no address special case: the unit
                // address resolves like any member (byte path via the
                // rule below), and loads/stores branch on the flag at
                // their own sites.
                // Through a pointer (`p->x`, and `(*p).x` which parses
                // to a dereference base), the address is byte
                // arithmetic off the pointer value by the checker's
                // authoritative offset. Symbolic field steps cannot
                // resolve there: struct pointees erase to `void`
                // (self-referential definitions have no finite
                // expansion), so the base value's MIR type carries no
                // fields. Union crossings and bit-field structs join
                // the byte path for the dual reasons (overlaid members
                // share offset zero; coalesced units shift field
                // indices). Direct plain structs keep field steps below.
                //
                // The object type comes from the declared member type,
                // never the node's own type: array members decay to
                // pointers in value position (`u.arr[i]`), and loading
                // through the decayed type would read the address
                // itself as data.
                if *arrow || matches!(base.kind, fe::check::CheckedExprKind::Unary(fe::parse::UnaryOp::Deref, _)) || self.descent_needs_bytes(&base.ty, path) {
                    // Arrow takes the pointer value itself; dereference
                    // and direct bases take their address (an rvalue
                    // would copy the whole object and strand stores).
                    let ptr = if *arrow {
                        self.expr_value(module, func, frame, funcs, externs, globals, base)
                    } else {
                        self.place(module, func, frame, funcs, externs, globals, base).0
                    };
                    let obj = lower_ty(decl_ty, self.sema, self.interner);
                    let (addr, obj) = self.ptr_offset_addr(func, frame, ptr, *offset as u64, obj);
                    return (addr, obj);
                }
                let (mut addr, mut obj) = self.place(module, func, frame, funcs, externs, globals, base);
                for field in path {
                    let (next, ty) = self.gep(func, frame, addr, &obj, vec![mir::GepIndex::Field(*field as u32)]);
                    addr = next;
                    obj = ty;
                }
                (addr, obj)
            }
            fe::check::CheckedExprKind::Subscript { base, index } => {
                let base_ty = base.ty.unqualified().clone();
                // The base address: array storage itself, or the pointer
                // value for pointer bases.
                let addr = match &base_ty {
                    fe::parse::Ty::Array { .. } => self.place(module, func, frame, funcs, externs, globals, base).0,
                    fe::parse::Ty::Pointer { .. } => self.expr_value(module, func, frame, funcs, externs, globals, base),
                    _ => panic!("subscript of non-array, non-pointer"),
                };
                // Step by the frontend element size through byte
                // arithmetic. MIR-level stepping cannot see it: struct
                // pointees erase to `void`, unsized arrays have no
                // array form (their object chain would stride by
                // pointer size), and checker decay may leave extra
                // pointer levels. `void` steps one byte (GNU).
                let elem_fe = match self.subscript_step_ty(base) {
                    Some(t) => t,
                    // Calls, ternaries, and conversions yielding
                    // pointers have no declared shape to consult: step
                    // over the node's own pointee, as before.
                    None => match base.ty.unqualified() {
                        fe::parse::Ty::Array { elem, .. } => elem.as_ref().clone(),
                        fe::parse::Ty::Pointer { pointee, .. } => pointee.as_ref().clone(),
                        _ => panic!("subscript of non-array, non-pointer"),
                    },
                };
                let obj = lower_ty(&elem_fe, self.sema, self.interner);
                let idx = self.expr_value(module, func, frame, funcs, externs, globals, index);
                if Self::ty_is_vm(&elem_fe) {
                    // A row of a VLA steps by a runtime extent: cached
                    // for a declared object, fresh through pointers.
                    let scale = match self.vla_cached_extents(frame, base) {
                        Some(map) => self.vla_size_with(func, frame, &elem_fe, &map),
                        None => self.vla_type_size(module, func, frame, funcs, externs, globals, &elem_fe),
                    };
                    return self.scaled_index_addr_dyn(func, frame, addr, idx, &index.ty, scale, obj);
                }
                let scale = match elem_fe.unqualified() {
                    fe::parse::Ty::Void | fe::parse::Ty::Func { .. } => 1,
                    _ => self.sema.type_size(&elem_fe, self.interner).expect("index into sized element"),
                };
                return self.scaled_index_addr(func, frame, addr, idx, &index.ty, scale, obj);
            }
            fe::check::CheckedExprKind::Unary(fe::parse::UnaryOp::Deref, inner) => {
                let ptr = self.expr_value(module, func, frame, funcs, externs, globals, inner);
                let fe::parse::Ty::Pointer { pointee, .. } = inner.ty.unqualified().clone() else {
                    panic!("dereference of non-pointer");
                };
                (ptr, lower_ty(&pointee, self.sema, self.interner))
            }
            // `__real__ x` / `__imag__ x` select a component place:
            // field 0 (real) or 1 (imaginary) of the pair. Every
            // caller addressing through here holds an lvalue base
            // (the checker rejects stores, increments, and
            // address-of on rvalue parts), so the base always has
            // its own storage.
            fe::check::CheckedExprKind::Unary(op, inner) if matches!(op, fe::parse::UnaryOp::Real | fe::parse::UnaryOp::Imag) => {
                let base = match &inner.kind {
                    fe::check::CheckedExprKind::Ident(_)
                    | fe::check::CheckedExprKind::Member { .. }
                    | fe::check::CheckedExprKind::Subscript { .. }
                    | fe::check::CheckedExprKind::CompoundLiteral { .. }
                    | fe::check::CheckedExprKind::Unary(fe::parse::UnaryOp::Deref, _)
                    | fe::check::CheckedExprKind::Unary(fe::parse::UnaryOp::Real | fe::parse::UnaryOp::Imag, _) => {
                        self.place(module, func, frame, funcs, externs, globals, inner).0
                    }
                    _ => panic!("address of a complex rvalue part"),
                };
                let pair = lower_ty(&inner.ty, self.sema, self.interner);
                let index = if matches!(op, fe::parse::UnaryOp::Real) { 0 } else { 1 };
                let (addr, obj) = self.gep(func, frame, base, &pair, vec![mir::GepIndex::Field(index)]);
                (addr, obj)
            }
            fe::check::CheckedExprKind::CompoundLiteral { ty, init } => {
                let mir_ty = lower_ty(ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, mir_ty.clone());
                self.lower_init(module, func, frame, funcs, externs, globals, slot, ty, init.as_ref());
                (slot, mir_ty)
            }
            // Aggregate rvalues (call results, statement expressions,
            // ternaries): copy into a temporary so their members have
            // stable storage to address. The checker rejects stores
            // through such a place.
            _ => {
                let mir_ty = lower_ty(&expr.ty, self.sema, self.interner);
                assert!(mir_ty.is_aggregate(), "address of a non-lvalue");
                let slot = self.alloca(func, frame, mir_ty.clone());
                let val = self.expr_value(module, func, frame, funcs, externs, globals, expr);
                self.emit_unit(
                    func,
                    frame,
                    mir::InstKind::Memcpy {
                        dst: slot,
                        src: val,
                        len: mir_ty.size_bytes(),
                    },
                );
                (slot, mir_ty)
            }
        }
    }

    /// Rvalue of a place address: arrays decay to the address itself
    /// (C never loads an array by value); everything else loads, which
    /// copies aggregates per the MIR convention. Decay keys off the
    /// declared object type, since the checker pre-decays array
    /// expressions to pointers.
    fn rvalue_of_place(&self, func: &mut mir::Function, frame: &mut Frame, addr: mir::Value, obj_is_array: bool, mir_ty: mir::Ty, vol: bool) -> mir::Value {
        if obj_is_array { addr } else { self.load(func, frame, addr, mir_ty, vol) }
    }

    /// Whether `va_list` behaves as GCC's array-of-one-tag (value is
    /// the tag's address, never a load). MSVC's `va_list` is a plain
    /// pointer: its value is the cursor it holds.
    fn va_list_is_array(&self) -> bool {
        !matches!(self.sema.target.os, fe::target::Os::Windows)
    }

    /// Whether an expression denotes array storage decayed to a
    /// pointer (its value is its address, never a load): array idents,
    /// members, and rows, through assignment-inserted conversions. The
    /// checker decays some nodes to pointers (same kind, new type),
    /// so declared types — not node types — decide array-backed
    /// bases. MIR twin of the legacy backend's `expr_denotes_array`.
    fn expr_denotes_array(&self, e: &fe::check::CheckedExpr) -> bool {
        // A `va_list` names its tag object the way an array names its
        // storage: the value is the address, never a load — whether it
        // arrives as an ident, a member, or an element. Where the ABI
        // spells `va_list` as a plain pointer (Win64), its value is a
        // load instead.
        match e.ty.unqualified() {
            fe::parse::Ty::Array { .. } => return true,
            fe::parse::Ty::BuiltinVaList => return self.va_list_is_array(),
            _ => {}
        }
        match &e.kind {
            fe::check::CheckedExprKind::Cast { expr, .. } | fe::check::CheckedExprKind::Convert { expr, .. } => self.expr_denotes_array(expr),
            // A decayed array literal (`(int[]){1, 2}` as an `int *`):
            // the kind keeps the pre-decay array type while the node
            // carries the pointer, so the object is the address.
            fe::check::CheckedExprKind::CompoundLiteral { ty, .. } => {
                matches!(ty.unqualified(), fe::parse::Ty::Array { .. })
            }
            fe::check::CheckedExprKind::Subscript { base, .. } => match self.subscript_step_ty(base) {
                Some(t) => match t.unqualified() {
                    fe::parse::Ty::Array { .. } => true,
                    fe::parse::Ty::BuiltinVaList => self.va_list_is_array(),
                    _ => false,
                },
                None => false,
            },
            // Dereference to an object type (`*app` for a `va_list`,
            // `*p` for an array): the value is the address. The
            // checker decays array-typed dereferences to pointers in
            // subscript and assignment positions (same `kind`, new
            // `ty`), so the operand's pointee must be consulted.
            fe::check::CheckedExprKind::Unary(fe::parse::UnaryOp::Deref, ptr) => {
                matches!(ptr.ty.unqualified(), fe::parse::Ty::Pointer { pointee, .. }
                    if matches!(pointee.unqualified(), fe::parse::Ty::Array { .. })
                        || (matches!(pointee.unqualified(), fe::parse::Ty::BuiltinVaList) && self.va_list_is_array()))
            }
            _ => self.subscript_base_is_array(e),
        }
    }

    /// Element type a subscript steps over, before decay: for `m[i]`
    /// of `int m[2][3]` that's the row `int[3]`, even though the
    /// checker decayed the node itself to `int *`. `None` when the
    /// base is not array-backed (plain pointers step by the node's
    /// own pointee type).
    fn subscript_step_ty(&self, base: &fe::check::CheckedExpr) -> Option<fe::parse::Ty> {
        let ty = match &base.kind {
            fe::check::CheckedExprKind::Ident(id) => self.sema.decls[id.0 as usize].ty.clone(),
            fe::check::CheckedExprKind::Member { base: inner, path, .. } => self.member_path_ty(inner, path)?,
            // `m[i][j]`: step over elements of `m[i]`'s row.
            fe::check::CheckedExprKind::Subscript { base: inner, .. } => {
                let row = self.subscript_step_ty(inner)?;
                return match row.unqualified() {
                    fe::parse::Ty::Array { elem, .. } => Some(elem.as_ref().clone()),
                    _ => None,
                };
            }
            // `(*p)[i]`: the checker decayed `*p` from array to
            // pointer-to-element (same `kind`, new `ty`), so the
            // declaration cannot name the step. The decayed pointer
            // itself does.
            fe::check::CheckedExprKind::Unary(fe::parse::UnaryOp::Deref, _) => {
                return match base.ty.unqualified() {
                    fe::parse::Ty::Pointer { pointee, .. } => Some(pointee.as_ref().clone()),
                    fe::parse::Ty::Array { elem, .. } => Some(elem.as_ref().clone()),
                    _ => None,
                };
            }
            _ => return None,
        };
        match ty.unqualified() {
            fe::parse::Ty::Array { elem, .. } => Some(elem.as_ref().clone()),
            // A pointer operand steps over its pointee: `pp[i]` moves
            // by `sizeof *pp`, same as `arr[i]` moves by the element.
            fe::parse::Ty::Pointer { pointee, .. } => Some(pointee.as_ref().clone()),
            _ => None,
        }
    }

    /// Whether a subscript's base denotes an array object, so `b[i]`
    /// steps over array storage rather than a loaded pointer.
    fn subscript_base_is_array(&self, e: &fe::check::CheckedExpr) -> bool {
        let va_array = self.va_list_is_array();
        match &e.kind {
            // `arr[i]`: the slot holds the elements themselves. A
            // `va_list` ident behaves the same way where the ABI models
            // it as an array: its value is the tag object's address,
            // never a load.
            fe::check::CheckedExprKind::Ident(id) => match self.sema.decls[id.0 as usize].ty.unqualified() {
                fe::parse::Ty::Array { .. } => true,
                fe::parse::Ty::BuiltinVaList => va_array,
                _ => false,
            },
            fe::check::CheckedExprKind::Member { base: inner, path, .. } => match self.member_path_ty(inner, path).as_ref().map(|t| t.unqualified()) {
                Some(fe::parse::Ty::Array { .. }) => true,
                Some(fe::parse::Ty::BuiltinVaList) => va_array,
                _ => false,
            },
            // `a[i][j]`: the row itself is address-backed iff its own
            // base denotes an array object — recurse, since the
            // checker decays the row to a pointer (same `kind`, new
            // `ty`), which a type test cannot see through. A loaded
            // pointer row (`pp[i][j]`) recurses to `false`.
            fe::check::CheckedExprKind::Subscript { base: inner, .. } => self.subscript_base_is_array(inner),
            _ => false,
        }
    }

    /// Declared type at the end of a member path, for array-storage
    /// questions: walks struct definitions from the base, so checker
    /// decay along the way cannot hide an array member.
    fn member_path_ty(&self, inner: &fe::check::CheckedExpr, path: &[usize]) -> Option<fe::parse::Ty> {
        let mut ty: fe::parse::Ty = match inner.ty.unqualified() {
            fe::parse::Ty::Struct(id) => fe::parse::Ty::Struct(*id),
            fe::parse::Ty::Pointer { pointee, .. } => (**pointee).unqualified().clone(),
            _ => return None,
        };
        for &index in path {
            let fe::parse::Ty::Struct(cur) = ty.unqualified() else {
                return None;
            };
            ty = self.sema.structs[cur.0 as usize].members[index].ty.clone();
        }
        Some(ty)
    }

    /// Any scalar condition as an I8 test: nonzero counts as true.
    fn bool_of(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        expr: &fe::check::CheckedExpr,
    ) -> mir::Value {
        let value = self.expr_value(module, func, frame, funcs, externs, globals, expr);
        self.bool_of_value(func, frame, value, &expr.ty)
    }

    /// Truth value of an already-lowered scalar (for `?:` with no
    /// then-arm, whose condition value doubles as the result).
    fn bool_of_value(&self, func: &mut mir::Function, frame: &mut Frame, value: mir::Value, fe_ty: &fe::parse::Ty) -> mir::Value {
        match lower_ty(fe_ty, self.sema, self.interner) {
            mir::Ty::Bool | mir::Ty::Int { bits: 1 } => value,
            mir::Ty::Ptr(_) => {
                let null = self.iconst(func, 64, 0);
                self.emit(
                    func,
                    frame,
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Ne,
                        lhs: value,
                        rhs: null,
                    },
                    mir::Ty::Bool,
                )
            }
            mir::Ty::Int { .. } => {
                // Compare at the operand's own width: narrowing to
                // I32 here dropped the high half of 64-bit conditions
                // (`while (1ULL << 32)` never ran).
                let mir_ty = lower_ty(fe_ty, self.sema, self.interner);
                let zero = self.iconst(func, int_bits(&mir_ty), 0);
                self.emit(
                    func,
                    frame,
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Ne,
                        lhs: value,
                        rhs: zero,
                    },
                    mir::Ty::Bool,
                )
            }
            // Floats (including long double) test against +0.0: NaN
            // is nonzero like every other float, and -0.0 compares
            // equal to +0.0, so `!=` is the truth test.
            mir::Ty::Float(kind) => {
                let zero = func.push_const(mir::Const::Float { kind, bits: 0 });
                self.emit(
                    func,
                    frame,
                    mir::InstKind::Fcmp {
                        pred: mir::FcmpPred::Ne,
                        lhs: value,
                        rhs: zero,
                    },
                    mir::Ty::Bool,
                )
            }
            // A complex condition holds when either part is nonzero
            // (NaN parts count as nonzero, like scalar floats).
            _ if is_complex(fe_ty) => self.complex_nonzero(func, frame, value, fe_ty),
            _ => todo!("non-integer condition lowering"),
        }
    }

    /// A string literal as a fresh `char[N]` global holding the decoded
    /// bytes plus terminator. Wide literals are rejected by the decoder.
    fn string_literal(&self, module: &mut mir::ModuleBuilder, func: &mut mir::Function, frame: &mut Frame, parts: &[fe::lexer::StringLiteral]) -> mir::Value {
        let (global, elem, len) = self.materialize_string_global(module, parts);
        self.emit(
            func,
            frame,
            mir::InstKind::GlobalAddr { global },
            mir::Ty::ptr(mir::Ty::Array { elem: Box::new(elem), len }),
        )
    }

    /// Emit a string literal into a fresh anonymous global, returning
    /// it with its element type and element count for an address reloc
    /// or a `GlobalAddr`.
    fn materialize_string_global(&self, module: &mut mir::ModuleBuilder, parts: &[fe::lexer::StringLiteral]) -> (mir::Global, mir::Ty, u64) {
        let elem = fe::eval::string_elem_ty_in(parts, self.sema.target.wchar()).expect("checked literal has one prefix kind");
        let elem_mir = lower_ty(&elem, self.sema, self.interner);
        let width = elem_mir.size_bytes();
        let mut image = fe::eval::string_bytes_in(parts, self.interner, self.sema.target.wchar()).expect("checked string folds to bytes");
        let len = image.len() as u64 / width + 1;
        image.resize(len as usize * width as usize, 0);
        let n = self.str_count.get();
        self.str_count.set(n + 1);
        let global = module.module.push_global(
            &format!("__str.{n}"),
            mir::Ty::Array {
                elem: Box::new(elem_mir.clone()),
                len,
            },
            Some(mir::Const::Bytes(image)),
        );
        let data = &mut module.module.globals[global];
        data.local = true;
        data.align = Some(width.max(1));
        (global, elem_mir, len)
    }

    /// Whether an initializer writes every byte that can affect the
    /// object's value, making the zero-fill before it redundant.
    ///
    /// Only positional, recursively complete lists qualify: named
    /// designators and bit-field structs keep the zero-fill because
    /// they touch only part of the object (bit-fields read-modify-write
    /// their unnamed neighbors).
    fn initializer_covers_object(&self, fe_ty: &fe::parse::Ty, mir_ty: &mir::Ty, init: &fe::parse::Initializer) -> bool {
        match init {
            fe::parse::Initializer::Expr(e) => match mir_ty {
                mir::Ty::Array { elem: mir_elem, .. } => {
                    let fe::check::CheckedExprKind::Str(parts) = &e.kind else {
                        return false;
                    };
                    let Some(lit_elem) = fe::eval::string_elem_ty_in(parts, self.sema.target.wchar()) else {
                        return false;
                    };
                    if lower_ty(&lit_elem, self.sema, self.interner) != **mir_elem {
                        return false;
                    }
                    let lit_len = fe::eval::string_bytes_in(parts, self.interner, self.sema.target.wchar())
                        .expect("checked string folds to bytes")
                        .len() as u64
                        + mir_elem.size_bytes();
                    lit_len >= mir_ty.size_bytes()
                }
                // An expression initializer for a scalar or aggregate
                // value copies the complete object.
                _ => true,
            },
            fe::parse::Initializer::List { items, .. } => {
                if items.iter().any(|item| !item.designators.is_empty()) {
                    return false;
                }
                match (fe_ty.unqualified(), mir_ty) {
                    (fe::parse::Ty::Array { elem, .. }, mir::Ty::Array { elem: mir_elem, len }) => {
                        items.len() as u64 == *len && items.iter().all(|item| self.initializer_covers_object(elem, mir_elem, &item.init))
                    }
                    (fe::parse::Ty::Struct(id), mir::Ty::Struct { fields, .. }) => {
                        let def = &self.sema.structs[id.0 as usize];
                        if def.is_union || def.members.iter().any(|member| member.bitfield.is_some()) {
                            return false;
                        }
                        items.len() == fields.len()
                            && items
                                .iter()
                                .zip(fields)
                                .zip(&def.members)
                                .all(|((item, field), member)| self.initializer_covers_object(&member.ty, field, &item.init))
                    }
                    // `int x = {1}` is a one-element list around a
                    // scalar initializer.
                    _ if items.len() == 1 => self.initializer_covers_object(fe_ty, mir_ty, &items[0].init),
                    _ => false,
                }
            }
        }
    }

    /// Initialize a local slot from an initializer: zero the whole
    /// object first (C's remainder rule, covering padding too), then
    /// store each provided element.
    fn lower_init(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        slot: mir::Value,
        fe_ty: &fe::parse::Ty,
        init: &fe::parse::Initializer,
    ) {
        let mir_ty = lower_ty(fe_ty, self.sema, self.interner);
        match init {
            fe::parse::Initializer::Expr(e) => {
                // `char s[] = "abc"`: zero the object (the remainder
                // past the literal must be NUL), then copy the literal
                // bytes that fit. The literal may be shorter than the
                // array (`char s[2] = ""`), never longer in bytes than
                // the literal plus its terminator.
                if let (fe::check::CheckedExprKind::Str(parts), mir::Ty::Array { elem, .. }) = (&e.kind, &mir_ty) {
                    let lit_elem = fe::eval::string_elem_ty_in(parts, self.sema.target.wchar()).map(|lit| lower_ty(&lit, self.sema, self.interner));
                    if lit_elem.as_ref() == Some(&**elem) {
                        let lit_len = fe::eval::string_bytes_in(parts, self.interner, self.sema.target.wchar())
                            .expect("checked string folds to bytes")
                            .len() as u64
                            + elem.size_bytes();
                        let len = mir_ty.size_bytes();
                        if !self.initializer_covers_object(fe_ty, &mir_ty, init) {
                            self.emit_unit(func, frame, mir::InstKind::Memset { dst: slot, byte: 0, len });
                        }
                        let src = self.string_literal(module, func, frame, parts);
                        let src_addr = self.emit(
                            func,
                            frame,
                            mir::InstKind::GetElementPtr {
                                base: src,
                                indices: [mir::GepIndex::ConstIndex(0)].into_iter().collect(),
                            },
                            mir::Ty::ptr(mir::Ty::int(8)),
                        );
                        self.emit_unit(
                            func,
                            frame,
                            mir::InstKind::Memcpy {
                                dst: slot,
                                src: src_addr,
                                len: len.min(lit_len),
                            },
                        );
                        return;
                    }
                }
                let value = self.expr_value(module, func, frame, funcs, externs, globals, e);
                let narrowed = self.convert(func, frame, value, &e.ty, fe_ty);
                self.store(func, frame, slot, narrowed, ty_is_volatile(fe_ty));
            }
            fe::parse::Initializer::List { items, .. } => {
                if !self.initializer_covers_object(fe_ty, &mir_ty, init) {
                    self.emit_unit(
                        func,
                        frame,
                        mir::InstKind::Memset {
                            dst: slot,
                            byte: 0,
                            len: mir_ty.size_bytes(),
                        },
                    );
                }
                match fe_ty.unqualified() {
                    fe::parse::Ty::Array { elem, .. } => {
                        let mut index = 0u64;
                        for item in items {
                            index = self.array_init_item(module, func, frame, funcs, externs, globals, slot, elem, item, index);
                        }
                    }
                    fe::parse::Ty::Struct(id) => {
                        let mut field = 0usize;
                        for item in items {
                            field = self.struct_init_item(module, func, frame, funcs, externs, globals, slot, *id, item, field);
                        }
                    }
                    // `int x = {1}` is legal C: bare scalar braces.
                    _ if items.len() == 1 && items[0].designators.is_empty() => {
                        self.lower_init(module, func, frame, funcs, externs, globals, slot, fe_ty, &items[0].init);
                    }
                    _ => panic!("braced list for scalar type"),
                }
            }
        }
    }

    /// One initializer item for a bit-field struct: positional items
    /// skip unnamed bit-fields (which consume no value, matching
    /// GCC); every target addresses by layout byte offset, and
    /// bit-field targets read-modify-write through `init_bitfield`.
    #[allow(clippy::too_many_arguments)]
    fn bitfield_struct_init_item(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        slot: mir::Value,
        struct_id: fe::sema::StructId,
        item: &fe::parse::InitItem,
        cursor: usize,
    ) -> usize {
        let def = &self.sema.structs[struct_id.0 as usize];
        let (index, rest) = match item.designators.split_first() {
            None => {
                let mut i = cursor;
                while def.members.get(i).is_some_and(|m| m.name.is_none() && m.bitfield.is_some()) {
                    i += 1;
                }
                (i, &[][..])
            }
            Some((des @ (fe::parse::Designator::Field { .. } | fe::parse::Designator::AnonField { .. }), rest)) => (designated_member(self.sema, struct_id, des) as usize, rest),
            Some((fe::parse::Designator::Index { .. }, _)) => {
                panic!("index designator into struct")
            }
        };
        let member = def.members.get(index).expect("excess initializer for struct").clone();
        let layout = self.sema.struct_layout(struct_id, self.interner).expect("layout of initialized struct");
        let (addr, _) = self.ptr_offset_addr(func, frame, slot, layout.member_offsets[index], lower_ty(&member.ty, self.sema, self.interner));
        let next = match item.designators.first() {
            Some(des @ (fe::parse::Designator::Field { .. } | fe::parse::Designator::AnonField { .. })) => designated_member(self.sema, struct_id, des) as usize + 1,
            Some(_) => cursor,
            None => index + 1,
        };
        if rest.is_empty() {
            if let Some(t) = self.bitfield_target_of(struct_id, index) {
                self.init_bitfield(module, func, frame, funcs, externs, globals, addr, &t, &item.init);
            } else {
                self.lower_init(module, func, frame, funcs, externs, globals, addr, &member.ty, &item.init);
            }
            return next;
        }
        let (addr, fe, bf) = self.init_addr(func, frame, addr, member.ty.clone(), rest);
        match bf {
            Some(t) => self.init_bitfield(module, func, frame, funcs, externs, globals, addr, &t, &item.init),
            None => self.lower_init(module, func, frame, funcs, externs, globals, addr, &fe, &item.init),
        }
        next
    }

    /// Walk an explicit designator list from a slot address, returning
    /// the addressed subobject and its frontend type.
    fn init_addr(
        &self,
        func: &mut mir::Function,
        frame: &mut Frame,
        mut addr: mir::Value,
        mut fe_ty: fe::parse::Ty,
        designators: &[fe::parse::Designator],
    ) -> (mir::Value, fe::parse::Ty, Option<BitfieldTarget>) {
        // Mirror of the walk in MIR types for well-typed GEPs.
        let mut mir_ty = lower_ty(&fe_ty, self.sema, self.interner);
        let mut bf = None;
        for (di, des) in designators.iter().enumerate() {
            // Bit-fields have no addressable subobjects: anything
            // below one is invalid C rejected above, backstopped here.
            let rest_empty = di + 1 == designators.len();
            match des {
                fe::parse::Designator::Field { .. } | fe::parse::Designator::AnonField { .. } => {
                    let fe::parse::Ty::Struct(id) = fe_ty.unqualified() else {
                        panic!("field designator into non-struct");
                    };
                    let id = *id;
                    let index = designated_member(self.sema, id, des);
                    // Overlaid union members share the base address;
                    // only the type advances.
                    if self.sema.structs[id.0 as usize].is_union {
                        fe_ty = self.sema.structs[id.0 as usize].members[index as usize].ty.clone();
                        mir_ty = lower_ty(&fe_ty, self.sema, self.interner);
                        addr = self.retype_addr(func, frame, addr, mir_ty.clone());
                        bf = self.bitfield_target_of(id, index as usize);
                        if bf.is_some() && !rest_empty {
                            panic!("designator into bit-field");
                        }
                        continue;
                    }
                    // Coalesced units shift field indices: address
                    // bit-field levels by layout bytes instead.
                    if self.sema.structs[id.0 as usize].members.iter().any(|m| m.bitfield.is_some()) {
                        let layout = self.sema.struct_layout(id, self.interner).expect("layout of initialized struct");
                        fe_ty = self.sema.structs[id.0 as usize].members[index as usize].ty.clone();
                        let (addr2, _) = self.ptr_offset_addr(func, frame, addr, layout.member_offsets[index as usize], lower_ty(&fe_ty, self.sema, self.interner));
                        addr = addr2;
                        mir_ty = lower_ty(&fe_ty, self.sema, self.interner);
                        bf = self.bitfield_target_of(id, index as usize);
                        if bf.is_some() && !rest_empty {
                            panic!("designator into bit-field");
                        }
                        continue;
                    }
                    (addr, mir_ty) = self.gep(func, frame, addr, &mir_ty, vec![mir::GepIndex::Field(index)]);
                    fe_ty = self.sema.structs[id.0 as usize].members[index as usize].ty.clone();
                }
                fe::parse::Designator::Index { index, .. } => {
                    let Some(fe::eval::Value::Int(k)) = fe::eval::eval_const(index, self.sema, self.interner) else {
                        panic!("non-constant array designator");
                    };
                    let fe::parse::Ty::Array { elem, .. } = fe_ty.unqualified() else {
                        panic!("index designator into non-array");
                    };
                    (addr, mir_ty) = self.gep(func, frame, addr, &mir_ty, vec![mir::GepIndex::ConstIndex(k as u64)]);
                    fe_ty = elem.as_ref().clone();
                }
            }
        }
        (addr, fe_ty, bf)
    }

    /// One array initializer item: positional or designated.
    fn array_init_item(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        slot: mir::Value,
        elem_ty: &fe::parse::Ty,
        item: &fe::parse::InitItem,
        cursor: u64,
    ) -> u64 {
        let (addr, ty, bf) = if item.designators.is_empty() {
            // Typed GEP (not a `Ptr(Void)` raw emit): nested
            // initializers reuse this address as their base, and both
            // verification and lowering resolve steps against the base
            // value's type.
            let mir::Ty::Ptr(inner) = func.values[slot].ty.clone() else {
                panic!("array initializer slot is not an address");
            };
            let (addr, _) = self.gep(func, frame, slot, inner.as_ref(), vec![mir::GepIndex::ConstIndex(cursor)]);
            (addr, elem_ty.clone(), None)
        } else {
            let array_ty = fe::parse::Ty::Array {
                elem: Box::new(elem_ty.clone()),
                quals: vec![],
                bound: None,
            };
            self.init_addr(func, frame, slot, array_ty, &item.designators)
        };
        match bf {
            Some(t) => self.init_bitfield(module, func, frame, funcs, externs, globals, addr, &t, &item.init),
            None => self.lower_init(module, func, frame, funcs, externs, globals, addr, &ty, &item.init),
        }
        match item.designators.first() {
            Some(fe::parse::Designator::Index { index, .. }) => match fe::eval::eval_const(index, self.sema, self.interner) {
                Some(fe::eval::Value::Int(k)) => (k as u64) + 1,
                _ => cursor,
            },
            Some(_) => cursor,
            None => cursor + 1,
        }
    }

    /// One struct initializer item: positional or `.field =` designated.
    fn struct_init_item(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        slot: mir::Value,
        struct_id: fe::sema::StructId,
        item: &fe::parse::InitItem,
        cursor: usize,
    ) -> usize {
        // Overlaid members share offset zero: the slot itself is every
        // member's address, with the type selected by position or
        // designator. Deeper designators walk from the member.
        if self.sema.structs[struct_id.0 as usize].is_union {
            return self.union_init_item(module, func, frame, funcs, externs, globals, slot, struct_id, item, cursor);
        }
        // Bit-field structs address by layout bytes (coalesced units
        // shift field indices), with stores read-modify-writing.
        if self.sema.structs[struct_id.0 as usize].members.iter().any(|m| m.bitfield.is_some()) {
            return self.bitfield_struct_init_item(module, func, frame, funcs, externs, globals, slot, struct_id, item, cursor);
        }
        let (addr, ty, bf) = if item.designators.is_empty() {
            // Same typed-base discipline as above: the address feeds
            // nested initializers and must resolve steps.
            let mir_struct = lower_ty(&fe::parse::Ty::Struct(struct_id), self.sema, self.interner);
            let (addr, _) = self.gep(func, frame, slot, &mir_struct, vec![mir::GepIndex::Field(cursor as u32)]);
            let ty = self.sema.structs[struct_id.0 as usize].members[cursor].ty.clone();
            (addr, ty, None)
        } else {
            let struct_ty = fe::parse::Ty::Struct(struct_id);
            self.init_addr(func, frame, slot, struct_ty, &item.designators)
        };
        match bf {
            Some(t) => self.init_bitfield(module, func, frame, funcs, externs, globals, addr, &t, &item.init),
            None => self.lower_init(module, func, frame, funcs, externs, globals, addr, &ty, &item.init),
        }
        match item.designators.first() {
            Some(des @ (fe::parse::Designator::Field { .. } | fe::parse::Designator::AnonField { .. })) => designated_member(self.sema, struct_id, des) as usize + 1,
            Some(_) => cursor,
            None => cursor + 1,
        }
    }

    /// One union initializer item: the slot itself at offset zero,
    /// typed by position or by the leading designator, with deeper
    /// designators walking from the selected member. Bit-field targets
    /// read-modify-write; positional items skip unnamed bit-fields
    /// like GCC.
    #[allow(clippy::too_many_arguments)]
    fn union_init_item(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        slot: mir::Value,
        struct_id: fe::sema::StructId,
        item: &fe::parse::InitItem,
        cursor: usize,
    ) -> usize {
        let def = &self.sema.structs[struct_id.0 as usize];
        let (index, rest) = match item.designators.split_first() {
            None => {
                let mut i = cursor;
                while def.members.get(i).is_some_and(|m| m.name.is_none() && m.bitfield.is_some()) {
                    i += 1;
                }
                (i, &[][..])
            }
            Some((des @ (fe::parse::Designator::Field { .. } | fe::parse::Designator::AnonField { .. }), rest)) => (designated_member(self.sema, struct_id, des) as usize, rest),
            Some((fe::parse::Designator::Index { .. }, _)) => {
                panic!("index designator into union")
            }
        };
        let member_fe = def.members.get(index).expect("excess initializer for union").ty.clone();
        let next = match item.designators.first() {
            Some(des @ (fe::parse::Designator::Field { .. } | fe::parse::Designator::AnonField { .. })) => designated_member(self.sema, struct_id, des) as usize + 1,
            Some(_) => cursor,
            None => index + 1,
        };
        let member_mir = lower_ty(&member_fe, self.sema, self.interner);
        // The union's storage is the member's storage: retype the
        // address so MIR steps see the member, not the union image.
        let slot = self.retype_addr(func, frame, slot, member_mir);
        if rest.is_empty() {
            if let Some(t) = self.bitfield_target_of(struct_id, index) {
                self.init_bitfield(module, func, frame, funcs, externs, globals, slot, &t, &item.init);
            } else {
                self.lower_init(module, func, frame, funcs, externs, globals, slot, &member_fe, &item.init);
            }
        } else {
            let (addr, fe, bf) = self.init_addr(func, frame, slot, member_fe, rest);
            match bf {
                Some(t) => self.init_bitfield(module, func, frame, funcs, externs, globals, addr, &t, &item.init),
                None => self.lower_init(module, func, frame, funcs, externs, globals, addr, &fe, &item.init),
            }
        }
        next
    }

    /// Resolve an initializer item's designators to an absolute byte
    /// offset plus the subobject's types and the next positional cursor.
    /// Empty designators take the cursor position; a trailing index or
    /// field continues after it. The fifth return names a bit-field
    /// final target for read-modify-write folding (else `None`).
    fn global_designated(
        &self,
        base: u64,
        fe_ty: &fe::parse::Ty,
        mir_ty: &mir::Ty,
        designators: &[fe::parse::Designator],
        cursor: u64,
    ) -> (u64, fe::parse::Ty, mir::Ty, u64, Option<BitfieldTarget>) {
        // Positional items skip unnamed bit-fields (which consume no
        // value, matching GCC); designators name their member.
        let positional = |id: fe::sema::StructId, cursor: u64| -> usize {
            let def = &self.sema.structs[id.0 as usize];
            let mut i = cursor as usize;
            while def.members.get(i).is_some_and(|m| m.name.is_none() && m.bitfield.is_some()) {
                i += 1;
            }
            i
        };
        if designators.is_empty() {
            return match (fe_ty.unqualified(), mir_ty) {
                (fe::parse::Ty::Array { elem, .. }, mir::Ty::Array { elem: mir_elem, .. }) => {
                    let elem_size = self.sema.type_size(elem, self.interner).expect("sized array element");
                    (base + cursor * elem_size, elem.as_ref().clone(), mir_elem.as_ref().clone(), cursor + 1, None)
                }
                // Bit-field structs address by layout bytes; the flag
                // marks bit-field targets for read-modify-write.
                (fe::parse::Ty::Struct(id), _) if self.sema.structs[id.0 as usize].members.iter().any(|m| m.bitfield.is_some()) && !self.sema.structs[id.0 as usize].is_union => {
                    let slot = positional(*id, cursor);
                    let member_fe = self.sema.structs[id.0 as usize].members[slot].ty.clone();
                    let layout = self.sema.struct_layout(*id, self.interner).expect("layout of initialized struct");
                    let member_mir = lower_ty(&member_fe, self.sema, self.interner);
                    (
                        base + layout.member_offsets[slot],
                        member_fe,
                        member_mir,
                        slot as u64 + 1,
                        self.bitfield_target_of(*id, slot),
                    )
                }
                (fe::parse::Ty::Struct(id), mir::Ty::Struct { fields, .. }) => {
                    let offsets = mir::struct_field_offsets(mir_ty);
                    let slot = cursor as usize;
                    (
                        base + offsets[slot],
                        self.sema.structs[id.0 as usize].members[slot].ty.clone(),
                        fields[slot].clone(),
                        cursor + 1,
                        None,
                    )
                }
                // Union members overlay at the base: position selects
                // the type, the offset never advances.
                (fe::parse::Ty::Struct(id), mir::Ty::Array { .. }) if self.sema.structs[id.0 as usize].is_union => {
                    let slot = positional(*id, cursor);
                    let member_fe = self.sema.structs[id.0 as usize].members.get(slot).expect("excess initializer for union").ty.clone();
                    let member_mir = lower_ty(&member_fe, self.sema, self.interner);
                    (base, member_fe, member_mir, slot as u64 + 1, self.bitfield_target_of(*id, slot))
                }
                _ => panic!("positional initializer into scalar"),
            };
        }
        let mut off = base;
        let mut fe = fe_ty.clone();
        let mut mi = mir_ty.clone();
        let mut bf = None;
        for (di, des) in designators.iter().enumerate() {
            let rest_empty = di + 1 == designators.len();
            match des {
                fe::parse::Designator::Field { .. } | fe::parse::Designator::AnonField { .. } => {
                    let fe::parse::Ty::Struct(id) = fe.unqualified() else {
                        panic!("field designator into non-struct");
                    };
                    let id = *id;
                    let slot = designated_member(self.sema, id, des) as usize;
                    // Union members overlay: no offset, member type by
                    // name with a freshly lowered MIR type.
                    if self.sema.structs[id.0 as usize].is_union {
                        fe = self.sema.structs[id.0 as usize].members[slot].ty.clone();
                        mi = lower_ty(&fe, self.sema, self.interner);
                        bf = self.bitfield_target_of(id, slot);
                        if bf.is_some() && !rest_empty {
                            panic!("designator into bit-field");
                        }
                        continue;
                    }
                    // Coalesced units shift field indices: address
                    // bit-field levels by layout bytes instead.
                    if self.sema.structs[id.0 as usize].members.iter().any(|m| m.bitfield.is_some()) {
                        let layout = self.sema.struct_layout(id, self.interner).expect("layout of initialized struct");
                        off += layout.member_offsets[slot];
                        fe = self.sema.structs[id.0 as usize].members[slot].ty.clone();
                        mi = lower_ty(&fe, self.sema, self.interner);
                        bf = self.bitfield_target_of(id, slot);
                        if bf.is_some() && !rest_empty {
                            panic!("designator into bit-field");
                        }
                        continue;
                    }
                    let mir::Ty::Struct { fields, .. } = &mi else {
                        panic!("field designator into non-struct");
                    };
                    off += mir::struct_field_offsets(&mi)[slot];
                    fe = self.sema.structs[id.0 as usize].members[slot].ty.clone();
                    mi = fields[slot].clone();
                }
                fe::parse::Designator::Index { index, .. } => {
                    let Some(fe::eval::Value::Int(k)) = fe::eval::eval_const(index, self.sema, self.interner) else {
                        panic!("non-constant array designator");
                    };
                    // A flexible array member decays to a pointer in
                    // MIR, but its elements still address by element
                    // stride from the member's offset.
                    if let (fe::parse::Ty::Array { elem, bound: None, .. }, mir::Ty::Ptr(_)) = (fe.unqualified(), &mi) {
                        let elem_size = self.sema.type_size(elem, self.interner).expect("sized array element");
                        off += (k as u64) * elem_size;
                        fe = elem.as_ref().clone();
                        mi = lower_ty(&fe, self.sema, self.interner);
                        continue;
                    }
                    let (fe::parse::Ty::Array { elem, .. }, mir::Ty::Array { elem: mir_elem, .. }) = (fe.unqualified(), &mi) else {
                        panic!("index designator into non-array");
                    };
                    let elem_size = self.sema.type_size(elem, self.interner).expect("sized array element");
                    off += (k as u64) * elem_size;
                    fe = elem.as_ref().clone();
                    mi = mir_elem.as_ref().clone();
                }
            }
        }
        let next = if designators.len() == 1 {
            match &designators[0] {
                fe::parse::Designator::Index { index, .. } => match fe::eval::eval_const(index, self.sema, self.interner) {
                    Some(fe::eval::Value::Int(k)) => (k as u64) + 1,
                    _ => cursor,
                },
                des @ (fe::parse::Designator::Field { .. } | fe::parse::Designator::AnonField { .. }) => {
                    // Only exact when the field addresses this level;
                    // nested paths keep the cursor (pathological).
                    match (fe_ty.unqualified(), mir_ty) {
                        (fe::parse::Ty::Struct(id), mir::Ty::Struct { .. }) => designated_member(self.sema, *id, des) as u64 + 1,
                        _ => cursor,
                    }
                }
            }
        } else {
            cursor
        };
        (off, fe, mi, next, bf)
    }

    fn place_of_var(&self, func: &mut mir::Function, frame: &mut Frame, globals: &HashMap<fe::sema::DeclId, mir::Global>, id: fe::sema::DeclId) -> mir::Value {
        if let Some(slot) = frame.slots.get(&id) {
            // A VLA object's home is the slot holding its heap pointer:
            // the address of the object is the loaded value.
            if Self::ty_is_vla_object(&self.sema.decls[id.0 as usize].ty) {
                return self.load(func, frame, *slot, mir::Ty::ptr(mir::Ty::Void), false);
            }
            return *slot;
        }
        let Some(global) = globals.get(&id) else {
            panic!("variable has no address in this frame");
        };
        let pointee = lower_ty(&self.sema.decls[id.0 as usize].ty, self.sema, self.interner);
        self.emit(func, frame, mir::InstKind::GlobalAddr { global: *global }, mir::Ty::ptr(pointee))
    }

    /// Explicit conversion between scalar MIR values, directed by the
    /// frontend types on each side. Complex on either side desugars
    /// through [`Self::convert_complex`]: the pair type erases the
    /// complex/real distinction, so this must branch on the frontend
    /// types first.
    fn convert(&self, func: &mut mir::Function, frame: &mut Frame, from: mir::Value, from_ty: &fe::parse::Ty, to_ty: &fe::parse::Ty) -> mir::Value {
        let from_mir = lower_ty(from_ty, self.sema, self.interner);
        let to_mir = lower_ty(to_ty, self.sema, self.interner);
        if from_mir == to_mir {
            return from;
        }
        // Casts to `void` discard: keep the value for the expression.
        if to_mir.is_void() {
            return from;
        }
        if is_complex(from_ty) || is_complex(to_ty) {
            return self.convert_complex(func, frame, from, from_ty, to_ty);
        }
        let kind = match (&from_mir, &to_mir) {
            (mir::Ty::Bool, mir::Ty::Int { .. }) => {
                // `Bool` occupies one byte and holds 0 or 1, so it is
                // already the representation of an 8-bit integer.
                if int_bits(&to_mir) == 8 {
                    return from;
                }
                mir::InstKind::Zext(from)
            }
            (mir::Ty::Int { .. }, mir::Ty::Int { .. }) => {
                let (from_bits, to_bits) = (int_bits(&from_mir), int_bits(&to_mir));
                if to_bits < from_bits {
                    mir::InstKind::Trunc(from)
                } else if is_signed(from_ty, self.sema) {
                    mir::InstKind::Sext(from)
                } else {
                    mir::InstKind::Zext(from)
                }
            }
            (mir::Ty::Int { .. } | mir::Ty::Bool, mir::Ty::Float(_)) => {
                if is_signed(from_ty, self.sema) {
                    mir::InstKind::SItoF(from)
                } else {
                    mir::InstKind::UItoF(from)
                }
            }
            (mir::Ty::Int { .. }, mir::Ty::Bool) => {
                // Truth test, not a truncation (which would misread
                // values like 256 as false): nonzero counts as true,
                // like `bool_of`.
                let zero = self.iconst(func, int_bits(&from_mir), 0);
                return self.emit(
                    func,
                    frame,
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Ne,
                        lhs: from,
                        rhs: zero,
                    },
                    to_mir,
                );
            }
            (mir::Ty::Float(_), mir::Ty::Bool) => {
                // Truth test, not an integer conversion (which is
                // undefined for NaN and out-of-range values): nonzero
                // counts as true, like `bool_of`.
                let zero = func.push_const(mir::Const::Float {
                    kind: float_kind(&from_mir),
                    bits: 0,
                });
                return self.emit(
                    func,
                    frame,
                    mir::InstKind::Fcmp {
                        pred: mir::FcmpPred::Ne,
                        lhs: from,
                        rhs: zero,
                    },
                    to_mir,
                );
            }
            (mir::Ty::Float(_), mir::Ty::Int { .. }) => {
                if is_signed(to_ty, self.sema) {
                    mir::InstKind::SFtoI(from)
                } else {
                    mir::InstKind::UFtoI(from)
                }
            }
            (mir::Ty::Float(_), mir::Ty::Float(_)) => {
                if to_mir.size_bytes() > from_mir.size_bytes() {
                    mir::InstKind::Fpext(from)
                } else {
                    mir::InstKind::Fptrunc(from)
                }
            }
            (mir::Ty::Ptr(_), mir::Ty::Bool) => {
                // Null test, not a truncation of the address (which
                // would misread addresses like 0x100 as null): nonzero
                // counts as true, like `bool_of`.
                let null = self.iconst(func, 64, 0);
                return self.emit(
                    func,
                    frame,
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Ne,
                        lhs: from,
                        rhs: null,
                    },
                    to_mir,
                );
            }
            (mir::Ty::Ptr(_), mir::Ty::Int { .. }) => mir::InstKind::PtrToInt(from),
            (mir::Ty::Ptr(_), mir::Ty::Ptr(_)) => mir::InstKind::Bitcast(from),
            (mir::Ty::Int { .. } | mir::Ty::Bool, mir::Ty::Ptr(_)) => {
                // MIR integer types are signless, so the source's
                // signedness applies here (C17 6.3.2.3p5-6): widen
                // narrow sources to pointer width (LP64), sign- or
                // zero-extending as the frontend type dictates.
                let src = if int_bits(&from_mir) < 64 {
                    let kind = if is_signed(from_ty, self.sema) {
                        mir::InstKind::Sext(from)
                    } else {
                        mir::InstKind::Zext(from)
                    };
                    self.emit(func, frame, kind, mir::Ty::int(64))
                } else {
                    from
                };
                mir::InstKind::IntToPtr(src)
            }
            _ => todo!("conversion lowering"),
        };
        self.emit(func, frame, kind, to_mir)
    }

    /// One component of a complex value: `ExtractValue` off the pair,
    /// which lowers to a load from the value's slot. Component
    /// *addresses* (for stores through `__real__`/`__imag__`) use
    /// field GEPs in `place` instead: values carry the pair struct
    /// type and have no address of their own.
    fn complex_extract(&self, func: &mut mir::Function, frame: &mut Frame, agg: mir::Value, fe_ty: &fe::parse::Ty, index: u32) -> mir::Value {
        let comp_mir = lower_ty(&complex_component(fe_ty), self.sema, self.interner);
        self.emit(func, frame, mir::InstKind::ExtractValue { agg, index }, comp_mir)
    }

    /// Assemble a complex value from its component values: two
    /// `InsertValue` steps over an `undef` base, which lowers to
    /// stores into a fresh slot (no initial copy).
    fn complex_build(&self, func: &mut mir::Function, frame: &mut Frame, fe_ty: &fe::parse::Ty, re: mir::Value, im: mir::Value) -> mir::Value {
        let pair = lower_ty(fe_ty, self.sema, self.interner);
        let base = func.values.push(mir::ValueData {
            ty: pair.clone(),
            def: mir::Def::Const(mir::Const::Undef),
        });
        let with_re = self.emit(func, frame, mir::InstKind::InsertValue { agg: base, elem: re, index: 0 }, pair.clone());
        self.emit(func, frame, mir::InstKind::InsertValue { agg: with_re, elem: im, index: 1 }, pair)
    }

    /// The zero of a complex component type: integer zero, or
    /// floating `+0.0` in the component's precision (all-zero bits,
    /// which is `+0.0` for every supported float representation).
    fn complex_zero(&self, func: &mut mir::Function, comp_mir: &mir::Ty) -> mir::Value {
        match comp_mir {
            mir::Ty::Float(kind) => func.push_const(mir::Const::Float { kind: *kind, bits: 0 }),
            _ => self.iconst(func, int_bits(comp_mir), 0),
        }
    }

    /// Lower a conversion with complex on either side (`Cast` or
    /// `Convert`): complex to complex (part-wise), real to complex
    /// (value as the real part, zero imaginary part), complex to real
    /// (real part, imaginary discarded), and complex to `_Bool`
    /// (nonzero when either part is). Same-type conversions never
    /// reach here (the caller returns the value untouched).
    fn convert_complex(&self, func: &mut mir::Function, frame: &mut Frame, from: mir::Value, from_ty: &fe::parse::Ty, to_ty: &fe::parse::Ty) -> mir::Value {
        let from_cx = is_complex(from_ty);
        let to_cx = is_complex(to_ty);
        match (from_cx, to_cx) {
            (true, true) => {
                let from_comp = complex_component(from_ty);
                let to_comp = complex_component(to_ty);
                let re = self.complex_extract(func, frame, from, from_ty, 0);
                let im = self.complex_extract(func, frame, from, from_ty, 1);
                let new_re = self.convert(func, frame, re, &from_comp, &to_comp);
                let new_im = self.convert(func, frame, im, &from_comp, &to_comp);
                self.complex_build(func, frame, to_ty, new_re, new_im)
            }
            (false, true) => {
                let to_comp = complex_component(to_ty);
                let new_re = self.convert(func, frame, from, from_ty, &to_comp);
                let to_comp_mir = lower_ty(&to_comp, self.sema, self.interner);
                let zero = self.complex_zero(func, &to_comp_mir);
                self.complex_build(func, frame, to_ty, new_re, zero)
            }
            (true, false) => {
                if matches!(to_ty.unqualified(), fe::parse::Ty::Bool) {
                    let nz = self.complex_nonzero(func, frame, from, from_ty);
                    return self.convert_bool(func, frame, nz, &lower_ty(to_ty, self.sema, self.interner));
                }
                let from_comp = complex_component(from_ty);
                let re = self.complex_extract(func, frame, from, from_ty, 0);
                self.convert(func, frame, re, &from_comp, to_ty)
            }
            (false, false) => unreachable!("complex conversion without a complex side"),
        }
    }

    /// Whether a complex value is nonzero, as a `Bool`: either part
    /// nonzero (NaN parts count as nonzero, like scalar floats).
    fn complex_nonzero(&self, func: &mut mir::Function, frame: &mut Frame, agg: mir::Value, fe_ty: &fe::parse::Ty) -> mir::Value {
        let comp_fe = complex_component(fe_ty);
        let comp_mir = lower_ty(&comp_fe, self.sema, self.interner);
        let re = self.complex_extract(func, frame, agg, fe_ty, 0);
        let im = self.complex_extract(func, frame, agg, fe_ty, 1);
        let (re_nz, im_nz) = match &comp_mir {
            mir::Ty::Float(kind) => {
                let zero = func.push_const(mir::Const::Float { kind: *kind, bits: 0 });
                (
                    self.emit(
                        func,
                        frame,
                        mir::InstKind::Fcmp {
                            pred: mir::FcmpPred::Ne,
                            lhs: re,
                            rhs: zero,
                        },
                        mir::Ty::Bool,
                    ),
                    self.emit(
                        func,
                        frame,
                        mir::InstKind::Fcmp {
                            pred: mir::FcmpPred::Ne,
                            lhs: im,
                            rhs: zero,
                        },
                        mir::Ty::Bool,
                    ),
                )
            }
            _ => {
                let zero = self.iconst(func, int_bits(&comp_mir), 0);
                (
                    self.emit(
                        func,
                        frame,
                        mir::InstKind::Icmp {
                            pred: mir::IcmpPred::Ne,
                            lhs: re,
                            rhs: zero,
                        },
                        mir::Ty::Bool,
                    ),
                    self.emit(
                        func,
                        frame,
                        mir::InstKind::Icmp {
                            pred: mir::IcmpPred::Ne,
                            lhs: im,
                            rhs: zero,
                        },
                        mir::Ty::Bool,
                    ),
                )
            }
        };
        self.emit(func, frame, mir::InstKind::Or(re_nz, im_nz), mir::Ty::Bool)
    }

    /// Negate one loaded component value.
    fn complex_neg_part(&self, func: &mut mir::Function, frame: &mut Frame, v: mir::Value, comp_mir: &mir::Ty) -> mir::Value {
        match comp_mir {
            mir::Ty::Float(_) => self.emit(func, frame, mir::InstKind::FNeg(v), comp_mir.clone()),
            _ => self.emit(func, frame, mir::InstKind::Neg(v), comp_mir.clone()),
        }
    }

    /// Unary `-x` on a complex value: both parts negated.
    fn complex_neg(&self, func: &mut mir::Function, frame: &mut Frame, agg: mir::Value, fe_ty: &fe::parse::Ty) -> mir::Value {
        let comp_fe = complex_component(fe_ty);
        let comp_mir = lower_ty(&comp_fe, self.sema, self.interner);
        let re = self.complex_extract(func, frame, agg, fe_ty, 0);
        let im = self.complex_extract(func, frame, agg, fe_ty, 1);
        let neg_re = self.complex_neg_part(func, frame, re, &comp_mir);
        let neg_im = self.complex_neg_part(func, frame, im, &comp_mir);
        self.complex_build(func, frame, fe_ty, neg_re, neg_im)
    }

    /// GNU `~x` on a complex value: the conjugate (real part kept,
    /// imaginary part negated).
    fn complex_conj(&self, func: &mut mir::Function, frame: &mut Frame, agg: mir::Value, fe_ty: &fe::parse::Ty) -> mir::Value {
        let comp_fe = complex_component(fe_ty);
        let comp_mir = lower_ty(&comp_fe, self.sema, self.interner);
        let re = self.complex_extract(func, frame, agg, fe_ty, 0);
        let im = self.complex_extract(func, frame, agg, fe_ty, 1);
        let neg_im = self.complex_neg_part(func, frame, im, &comp_mir);
        self.complex_build(func, frame, fe_ty, re, neg_im)
    }

    /// Materialize a complex literal (a GNU imaginary literal like
    /// `1i` or `1.0i`, whose checked form is an `Int`/`Float` node
    /// with complex type): zero real part, literal imaginary part.
    fn complex_literal(&self, func: &mut mir::Function, frame: &mut Frame, expr: &fe::check::CheckedExpr) -> mir::Value {
        let comp_fe = complex_component(&expr.ty);
        let comp_mir = lower_ty(&comp_fe, self.sema, self.interner);
        let folded = fe::eval::eval_const(expr, self.sema, self.interner).unwrap_or_else(|| panic!("complex literal must fold"));
        let fe::eval::Value::Complex(re, im) = folded else {
            panic!("complex literal without complex parts");
        };
        let re_v = func.push_const(self.const_of_folded(&re, &comp_mir));
        let im_v = func.push_const(self.const_of_folded(&im, &comp_mir));
        self.complex_build(func, frame, &expr.ty, re_v, im_v)
    }

    /// One complex `+`/`-`/`*`/`/` (both sides already share the
    /// operand type through the usual arithmetic conversions): the
    /// parts compute per the C formulas into a fresh value.
    /// `(a+bi)+(c+di)` adds part-wise, `(a+bi)(c+di)` is
    /// `(ac-bd)+(ad+bc)i`, and `(a+bi)/(c+di)` is
    /// `((ac+bd)+(bc-ad)i)/(cc+dd)`. Integer parts wrap; integer
    /// division follows the component's signedness; float parts
    /// (including `long double`, whose MIR ops lower through the
    /// runtime helpers) round once per component operation.
    fn complex_binop(&self, func: &mut mir::Function, frame: &mut Frame, op: &fe::parse::BinaryOp, lhs: mir::Value, rhs: mir::Value, fe_ty: &fe::parse::Ty) -> mir::Value {
        let comp_fe = complex_component(fe_ty);
        let comp_mir = lower_ty(&comp_fe, self.sema, self.interner);
        let a = self.complex_extract(func, frame, lhs, fe_ty, 0);
        let b = self.complex_extract(func, frame, lhs, fe_ty, 1);
        let c = self.complex_extract(func, frame, rhs, fe_ty, 0);
        let d = self.complex_extract(func, frame, rhs, fe_ty, 1);
        let (re, im) = match op {
            fe::parse::BinaryOp::Add => (self.complex_add(func, frame, a, c, &comp_mir, true), self.complex_add(func, frame, b, d, &comp_mir, true)),
            fe::parse::BinaryOp::Sub => (self.complex_add(func, frame, a, c, &comp_mir, false), self.complex_add(func, frame, b, d, &comp_mir, false)),
            fe::parse::BinaryOp::Mul => {
                let ac = self.complex_mul(func, frame, a, c, &comp_mir);
                let bd = self.complex_mul(func, frame, b, d, &comp_mir);
                let ad = self.complex_mul(func, frame, a, d, &comp_mir);
                let bc = self.complex_mul(func, frame, b, c, &comp_mir);
                (
                    self.complex_add(func, frame, ac, bd, &comp_mir, false),
                    self.complex_add(func, frame, ad, bc, &comp_mir, true),
                )
            }
            fe::parse::BinaryOp::Div => {
                let ac = self.complex_mul(func, frame, a, c, &comp_mir);
                let bd = self.complex_mul(func, frame, b, d, &comp_mir);
                let bc = self.complex_mul(func, frame, b, c, &comp_mir);
                let ad = self.complex_mul(func, frame, a, d, &comp_mir);
                let cc = self.complex_mul(func, frame, c, c, &comp_mir);
                let dd = self.complex_mul(func, frame, d, d, &comp_mir);
                let den = self.complex_add(func, frame, cc, dd, &comp_mir, true);
                let rn = self.complex_add(func, frame, ac, bd, &comp_mir, true);
                let rni = self.complex_add(func, frame, bc, ad, &comp_mir, false);
                (
                    self.complex_div(func, frame, rn, den, &comp_fe, &comp_mir),
                    self.complex_div(func, frame, rni, den, &comp_fe, &comp_mir),
                )
            }
            _ => unreachable!("frontend denies non-arithmetic complex operators"),
        };
        self.complex_build(func, frame, fe_ty, re, im)
    }

    /// One component addition or subtraction.
    fn complex_add(&self, func: &mut mir::Function, frame: &mut Frame, x: mir::Value, y: mir::Value, comp_mir: &mir::Ty, add: bool) -> mir::Value {
        let kind = match (add, comp_mir) {
            (true, mir::Ty::Float(_)) => mir::InstKind::FAdd(x, y),
            (false, mir::Ty::Float(_)) => mir::InstKind::FSub(x, y),
            (true, _) => mir::InstKind::Add(x, y),
            (false, _) => mir::InstKind::Sub(x, y),
        };
        self.emit(func, frame, kind, comp_mir.clone())
    }

    /// One component multiplication.
    fn complex_mul(&self, func: &mut mir::Function, frame: &mut Frame, x: mir::Value, y: mir::Value, comp_mir: &mir::Ty) -> mir::Value {
        let kind = match comp_mir {
            mir::Ty::Float(_) => mir::InstKind::FMul(x, y),
            _ => mir::InstKind::Mul(x, y),
        };
        self.emit(func, frame, kind, comp_mir.clone())
    }

    /// One component division: float parts round once, integer parts
    /// divide by the signedness of the component (truncating, like
    /// scalar division).
    fn complex_div(&self, func: &mut mir::Function, frame: &mut Frame, x: mir::Value, y: mir::Value, comp_fe: &fe::parse::Ty, comp_mir: &mir::Ty) -> mir::Value {
        let kind = match comp_mir {
            mir::Ty::Float(_) => mir::InstKind::FDiv(x, y),
            _ if is_signed(comp_fe, self.sema) => mir::InstKind::Sdiv(x, y),
            _ => mir::InstKind::Udiv(x, y),
        };
        self.emit(func, frame, kind, comp_mir.clone())
    }

    /// A complex `==`/`!=` (the only comparisons complex supports):
    /// equal when both parts compare equal. Float parts use ordered
    /// equality (NaN never equals, so `!=` holds); the result widens
    /// to the consuming `int` type.
    #[allow(clippy::too_many_arguments)]
    fn complex_cmp(
        &self,
        func: &mut mir::Function,
        frame: &mut Frame,
        op: &fe::parse::BinaryOp,
        lhs: mir::Value,
        rhs: mir::Value,
        fe_ty: &fe::parse::Ty,
        res_mir: &mir::Ty,
    ) -> mir::Value {
        let eq = matches!(op, fe::parse::BinaryOp::Eq);
        let comp_fe = complex_component(fe_ty);
        let comp_mir = lower_ty(&comp_fe, self.sema, self.interner);
        let a = self.complex_extract(func, frame, lhs, fe_ty, 0);
        let b = self.complex_extract(func, frame, lhs, fe_ty, 1);
        let c = self.complex_extract(func, frame, rhs, fe_ty, 0);
        let d = self.complex_extract(func, frame, rhs, fe_ty, 1);
        let (re_eq, im_eq) = match &comp_mir {
            mir::Ty::Float(_) => (
                self.emit(
                    func,
                    frame,
                    mir::InstKind::Fcmp {
                        pred: mir::FcmpPred::Eq,
                        lhs: a,
                        rhs: c,
                    },
                    mir::Ty::Bool,
                ),
                self.emit(
                    func,
                    frame,
                    mir::InstKind::Fcmp {
                        pred: mir::FcmpPred::Eq,
                        lhs: b,
                        rhs: d,
                    },
                    mir::Ty::Bool,
                ),
            ),
            _ => (
                self.emit(
                    func,
                    frame,
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Eq,
                        lhs: a,
                        rhs: c,
                    },
                    mir::Ty::Bool,
                ),
                self.emit(
                    func,
                    frame,
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Eq,
                        lhs: b,
                        rhs: d,
                    },
                    mir::Ty::Bool,
                ),
            ),
        };
        let both = self.emit(func, frame, mir::InstKind::And(re_eq, im_eq), mir::Ty::Bool);
        let r = if eq {
            both
        } else {
            let one = self.iconst(func, 1, 1);
            self.emit(func, frame, mir::InstKind::Xor(both, one), mir::Ty::Bool)
        };
        self.convert_bool(func, frame, r, res_mir)
    }

    /// Apply a compound assignment's operation to the lvalue's current
    /// value (`x`, converted to `op_ty`) and the converted rhs (`y`,
    /// of type `rhs_ty`), producing a value of `op_ty` (MIR type
    /// `mir_ty`). Pointer arithmetic scales by the pointee size through
    /// byte arithmetic, like `pointer_binary`.
    #[allow(clippy::too_many_arguments)]
    fn compound_op(
        &self,
        _module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        _funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        _externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        _globals: &HashMap<fe::sema::DeclId, mir::Global>,
        op: &fe::parse::AssignOp,
        x: mir::Value,
        op_ty: &fe::parse::Ty,
        y: mir::Value,
        rhs_ty: &fe::parse::Ty,
        mir_ty: &mir::Ty,
    ) -> mir::Value {
        use fe::parse::BinaryOp::*;
        if let fe::parse::Ty::Pointer { pointee, .. } = op_ty.unqualified() {
            // GNU void-pointer arithmetic: `void` (and function)
            // pointees step by one byte.
            let scale = match pointee.unqualified() {
                fe::parse::Ty::Void | fe::parse::Ty::Func { .. } => 1,
                _ => self.sema.type_size(pointee, self.interner).expect("sized pointee"),
            };
            let index = if matches!(op, fe::parse::AssignOp::Sub) {
                self.emit(func, frame, mir::InstKind::Neg(y), lower_ty(rhs_ty, self.sema, self.interner))
            } else {
                y
            };
            let mir::Ty::Ptr(obj) = mir_ty.clone() else {
                panic!("compound pointer assignment without a pointer result");
            };
            let (addr, _) = self.scaled_index_addr(func, frame, x, index, rhs_ty, scale, obj.as_ref().clone());
            return addr;
        }
        let bin_op = compound_bin_op(op);
        if is_complex(op_ty) {
            return self.complex_binop(func, frame, &bin_op, x, y, op_ty);
        }
        let kind = match (bin_op, lower_ty(op_ty, self.sema, self.interner)) {
            (Add, mir::Ty::Float(_)) => mir::InstKind::FAdd(x, y),
            (Sub, mir::Ty::Float(_)) => mir::InstKind::FSub(x, y),
            (Mul, mir::Ty::Float(_)) => mir::InstKind::FMul(x, y),
            (Div, mir::Ty::Float(_)) => mir::InstKind::FDiv(x, y),
            (Add, _) => mir::InstKind::Add(x, y),
            (Sub, _) => mir::InstKind::Sub(x, y),
            (Mul, _) => mir::InstKind::Mul(x, y),
            (Div, _) => {
                if is_signed(op_ty, self.sema) {
                    mir::InstKind::Sdiv(x, y)
                } else {
                    mir::InstKind::Udiv(x, y)
                }
            }
            (Rem, mir::Ty::Float(_)) => todo!("float remainder lowering"),
            (Rem, _) => {
                if is_signed(op_ty, self.sema) {
                    mir::InstKind::Srem(x, y)
                } else {
                    mir::InstKind::Urem(x, y)
                }
            }
            (Shl, _) => mir::InstKind::Shl(x, y),
            (Shr, _) => {
                if is_signed(op_ty, self.sema) {
                    mir::InstKind::AShr(x, y)
                } else {
                    mir::InstKind::LShr(x, y)
                }
            }
            (BitAnd, _) => mir::InstKind::And(x, y),
            (BitXor, _) => mir::InstKind::Xor(x, y),
            (BitOr, _) => mir::InstKind::Or(x, y),
            _ => todo!("compound assignment operator lowering"),
        };
        // The combination runs in the operator's type (`long double x`
        // with `int g` in `g += x`), not the lvalue's: the caller
        // narrows the result afterwards.
        let result_ty = lower_ty(op_ty, self.sema, self.interner);
        self.emit(func, frame, kind, result_ty)
    }

    /// Pointer-involving binary operators: scaled `p + i` / `p - i`
    /// through GEP element steps, `p - q` as a scaled byte difference,
    /// and unsigned comparisons.
    fn pointer_binary(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        op: &fe::parse::BinaryOp,
        lhs: &fe::check::CheckedExpr,
        rhs: &fe::check::CheckedExpr,
        mir_ty: &mir::Ty,
    ) -> mir::Value {
        use fe::parse::BinaryOp::*;
        match op {
            Add => {
                // Either side may carry the pointer. Steps scale by
                // the frontend pointee size through byte arithmetic:
                // the MIR pointee is erased, so a GEP element step
                // cannot know the stride.
                let (ptr_expr, idx_expr) = match (lhs.ty.unqualified(), rhs.ty.unqualified()) {
                    (fe::parse::Ty::Pointer { .. }, _) => (lhs, rhs),
                    (_, fe::parse::Ty::Pointer { .. }) => (rhs, lhs),
                    _ => panic!("pointer addition without a pointer"),
                };
                let fe::parse::Ty::Pointer { pointee, .. } = ptr_expr.ty.unqualified() else {
                    panic!("pointer addition without a pointer");
                };
                let base = self.expr_value(module, func, frame, funcs, externs, globals, ptr_expr);
                let index = self.expr_value(module, func, frame, funcs, externs, globals, idx_expr);
                let mir::Ty::Ptr(obj) = mir_ty.clone() else {
                    panic!("pointer addition without a pointer result");
                };
                let (sum, _) = if Self::ty_is_vm(pointee) {
                    let scale = self.pointee_stride(module, func, frame, funcs, externs, globals, ptr_expr, pointee);
                    self.scaled_index_addr_dyn(func, frame, base, index, &idx_expr.ty, scale, obj.as_ref().clone())
                } else {
                    self.scaled_index_addr(func, frame, base, index, &idx_expr.ty, self.pointee_size(pointee), obj.as_ref().clone())
                };
                sum
            }
            Sub => {
                let lhs_ptr = matches!(lhs.ty.unqualified(), fe::parse::Ty::Pointer { .. });
                let rhs_ptr = matches!(rhs.ty.unqualified(), fe::parse::Ty::Pointer { .. });
                if lhs_ptr && rhs_ptr {
                    // `p - q`: byte difference scaled down by the element
                    // size, which a variably modified pointee computes at
                    // runtime (`int (*)[h][w]`).
                    let fe::parse::Ty::Pointer { pointee, .. } = lhs.ty.unqualified() else {
                        panic!("pointer difference without a pointer");
                    };
                    let scale = self.pointee_stride(module, func, frame, funcs, externs, globals, lhs, pointee);
                    let x = self.expr_value(module, func, frame, funcs, externs, globals, lhs);
                    let y = self.expr_value(module, func, frame, funcs, externs, globals, rhs);
                    let wide_x = self.emit(func, frame, mir::InstKind::PtrToInt(x), mir::Ty::int(64));
                    let wide_y = self.emit(func, frame, mir::InstKind::PtrToInt(y), mir::Ty::int(64));
                    let diff = self.emit(func, frame, mir::InstKind::Sub(wide_x, wide_y), mir::Ty::int(64));
                    let quot = self.emit(func, frame, mir::InstKind::Sdiv(diff, scale), mir::Ty::int(64));
                    return self.convert_int_width(func, frame, quot, 64, mir_ty);
                }
                if !lhs_ptr {
                    panic!("pointer subtraction without a pointer minuend");
                }
                let fe::parse::Ty::Pointer { pointee, .. } = lhs.ty.unqualified() else {
                    panic!("pointer subtraction without a pointer minuend");
                };
                let base = self.expr_value(module, func, frame, funcs, externs, globals, lhs);
                let index = self.expr_value(module, func, frame, funcs, externs, globals, rhs);
                let neg = self.emit(func, frame, mir::InstKind::Neg(index), lower_ty(&rhs.ty, self.sema, self.interner));
                let mir::Ty::Ptr(obj) = mir_ty.clone() else {
                    panic!("pointer subtraction without a pointer result");
                };
                let (diff, _) = if Self::ty_is_vm(pointee) {
                    let scale = self.pointee_stride(module, func, frame, funcs, externs, globals, lhs, pointee);
                    self.scaled_index_addr_dyn(func, frame, base, neg, &rhs.ty, scale, obj.as_ref().clone())
                } else {
                    self.scaled_index_addr(func, frame, base, neg, &rhs.ty, self.pointee_size(pointee), obj.as_ref().clone())
                };
                diff
            }
            Lt | Le | Gt | Ge | Eq | Ne => {
                let x = self.expr_value(module, func, frame, funcs, externs, globals, lhs);
                let y = self.expr_value(module, func, frame, funcs, externs, globals, rhs);
                let pred = match op {
                    Lt => mir::IcmpPred::Ult,
                    Le => mir::IcmpPred::Ule,
                    Gt => mir::IcmpPred::Ugt,
                    Ge => mir::IcmpPred::Uge,
                    Eq => mir::IcmpPred::Eq,
                    Ne => mir::IcmpPred::Ne,
                    _ => panic!("not a comparison operator"),
                };
                let test = self.emit(func, frame, mir::InstKind::Icmp { pred, lhs: x, rhs: y }, mir::Ty::Bool);
                self.convert_bool(func, frame, test, mir_ty)
            }
            _ => panic!("pointer operator lowering: {op:?}"),
        }
    }

    /// Resize an integer value to a result type by width.
    fn convert_int_width(&self, func: &mut mir::Function, frame: &mut Frame, value: mir::Value, from_bits: u32, to_mir: &mir::Ty) -> mir::Value {
        let to_bits = int_bits(to_mir);
        if to_bits == from_bits {
            return value;
        }
        if to_bits < from_bits {
            return self.emit(func, frame, mir::InstKind::Trunc(value), to_mir.clone());
        }
        self.emit(func, frame, mir::InstKind::Sext(value), to_mir.clone())
    }

    /// Unsigned integer comparison. Signed and float cases are built
    /// inline at the use site; this covers what remains.
    fn int_cmp(&self, op: &fe::parse::BinaryOp, signed: bool, x: mir::Value, y: mir::Value) -> mir::InstKind {
        use fe::parse::BinaryOp::*;
        let pred = match (op, signed) {
            (Lt, false) => mir::IcmpPred::Ult,
            (Le, false) => mir::IcmpPred::Ule,
            (Gt, false) => mir::IcmpPred::Ugt,
            (Ge, false) => mir::IcmpPred::Uge,
            (Lt, true) => mir::IcmpPred::Slt,
            (Le, true) => mir::IcmpPred::Sle,
            (Gt, true) => mir::IcmpPred::Sgt,
            (Ge, true) => mir::IcmpPred::Sge,
            _ => panic!("not a comparison operator"),
        };
        mir::InstKind::Icmp { pred, lhs: x, rhs: y }
    }

    fn convert_bool(&self, func: &mut mir::Function, frame: &mut Frame, test: mir::Value, to_mir: &mir::Ty) -> mir::Value {
        if *to_mir == mir::Ty::Bool {
            return test;
        }
        // `Bool` is one byte; widening to `int` zero-extends.
        self.emit(func, frame, mir::InstKind::Zext(test), to_mir.clone())
    }

    fn iconst(&self, func: &mut mir::Function, bits: u32, value: i128) -> mir::Value {
        func.push_const(mir::Const::Int { bits, value })
    }

    /// A stack slot for `ty`, always created in the entry block (like
    /// clang -O0): the slot address dominates every use, including
    /// across loop back-edges and case/goto edges that skip the
    /// declaring block. Mid-function allocas left addresses Cranelift
    /// rejects as non-dominating.
    fn alloca(&self, func: &mut mir::Function, frame: &mut Frame, ty: mir::Ty) -> mir::Value {
        let entry = func.entry;
        let ptr = mir::Ty::ptr(ty.clone());
        let result = func.values.push(mir::ValueData {
            ty: ptr,
            def: mir::Def::Inst(mir::Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(mir::InstData {
            block: entry,
            kind: mir::InstKind::Alloca { pointee: ty, count: 1 },
            result: Some(result),
            loc: frame.loc,
        });
        func.values[result].def = mir::Def::Inst(inst);
        func.blocks[entry].insts.push(inst);
        result
    }

    fn load(&self, func: &mut mir::Function, frame: &mut Frame, addr: mir::Value, ty: mir::Ty, vol: bool) -> mir::Value {
        let flags = if vol { mir::MemFlags::vol() } else { mir::MemFlags::plain() };
        self.emit(func, frame, mir::InstKind::Load { addr, flags }, ty)
    }

    fn store(&self, func: &mut mir::Function, frame: &mut Frame, addr: mir::Value, val: mir::Value, vol: bool) {
        let flags = if vol { mir::MemFlags::vol() } else { mir::MemFlags::plain() };
        self.emit_unit(func, frame, mir::InstKind::Store { addr, val, flags });
    }

    fn emit(&self, func: &mut mir::Function, frame: &mut Frame, kind: mir::InstKind, ty: mir::Ty) -> mir::Value {
        let block = self.live_block(func, frame);
        let result = func.values.push(mir::ValueData {
            ty,
            def: mir::Def::Inst(mir::Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(mir::InstData {
            block,
            kind,
            result: Some(result),
            loc: frame.loc,
        });
        func.values[result].def = mir::Def::Inst(inst);
        func.blocks[block].insts.push(inst);
        result
    }

    fn emit_unit(&self, func: &mut mir::Function, frame: &mut Frame, kind: mir::InstKind) {
        let block = self.live_block(func, frame);
        let inst = func.insts.push(mir::InstData {
            block,
            kind,
            result: None,
            loc: frame.loc,
        });
        func.blocks[block].insts.push(inst);
    }

    fn terminate(&self, func: &mut mir::Function, frame: &mut Frame, term: mir::Terminator) {
        let block = self.live_block(func, frame);
        frame.current.take();
        func.blocks[block].term = Some(term);
    }

    /// The block instructions append to, opening an orphan past a
    /// terminator so unreachable source still lowers to valid MIR.
    fn live_block(&self, func: &mut mir::Function, frame: &mut Frame) -> mir::Block {
        if let Some(block) = frame.current {
            return block;
        }
        let block = func.push_block(vec![]);
        frame.current = Some(block);
        block
    }
}

fn float_kind(ty: &mir::Ty) -> mir::FloatKind {
    match ty {
        mir::Ty::Float(kind) => *kind,
        _ => panic!("not a float MIR type"),
    }
}

/// Whether the outermost level of a frontend type carries `volatile`.
/// `Ty::qualified` distributes qualifiers onto what they qualify, so
/// the outermost level is exactly the accessed object's own.
fn ty_is_volatile(ty: &fe::parse::Ty) -> bool {
    const VOL: fe::parse::TypeQualifier = fe::parse::TypeQualifier::Volatile;
    match ty {
        fe::parse::Ty::Qualified { quals, .. } => quals.contains(&VOL),
        fe::parse::Ty::Pointer { qualifiers, .. } => qualifiers.contains(&VOL),
        fe::parse::Ty::Array { quals, .. } => quals.contains(&VOL),
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
fn access_volatile(e: &fe::check::CheckedExpr, decls: &[fe::sema::Binding]) -> bool {
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
fn pointee_volatile(ty: &fe::parse::Ty) -> bool {
    match ty.unqualified() {
        fe::parse::Ty::Pointer { pointee, .. } => ty_is_volatile(pointee),
        fe::parse::Ty::Array { elem, .. } => ty_is_volatile(elem),
        _ => false,
    }
}

/// A scalar builtin whose calls take exactly one argument: a
/// declaration with any other shape names a user function, not the
/// builtin (GCC disables the builtin the same way).
fn unary_builtin(binding: &fe::sema::Binding, builtin: ScalarBuiltin) -> Option<ScalarBuiltin> {
    match binding.ty.unqualified() {
        fe::parse::Ty::Func { params, .. } if params.len() == 1 => Some(builtin),
        _ => None,
    }
}

/// The binary operation a compound assignment performs.
fn compound_bin_op(op: &fe::parse::AssignOp) -> fe::parse::BinaryOp {
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
fn is_mir_scalar(ty: &mir::Ty) -> bool {
    matches!(ty, mir::Ty::Bool | mir::Ty::Int { .. } | mir::Ty::Float(_) | mir::Ty::Ptr(_))
}

fn float_bits(value: fe::eval::FloatValue, ty: &mir::Ty) -> u128 {
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
fn float_int_form(ty: &mir::Ty) -> (mir::Ty, u32) {
    match float_kind(ty) {
        mir::FloatKind::F32 => (mir::Ty::int(32), 31),
        mir::FloatKind::F64 => (mir::Ty::int(64), 63),
        mir::FloatKind::F80 => (mir::Ty::int(128), 79),
        mir::FloatKind::F128 => (mir::Ty::int(128), 127),
    }
}

/// Bit pattern of the kind's positive infinity (every exponent bit).
fn float_inf_bits(ty: &mir::Ty) -> u128 {
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
fn float_min_normal_bits(ty: &mir::Ty) -> u128 {
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
#[cfg(test)]
mod tests {
    use cranelift_jit::{JITBuilder, JITModule};

    use glcc_frontend as fe;
    use glcc_mir as mir;

    use super::{lower_translation_unit, naked_asm_text};

    /// The driver's optimization pipeline (`src/driver.rs`).
    fn run_opt_pipeline(module: &mut mir::Module) {
        let mut pm = mir::pass::PassManager::new()
            .add_module_pass(mir::passes::Inline::small())
            .add_func_pass(mir::passes::Mem2Reg)
            .add_func_pass(mir::passes::ConstFold)
            .add_func_pass(mir::passes::Sccp)
            .add_func_pass(mir::passes::ForwardAggregateCopy)
            .add_func_pass(mir::passes::InstCombine)
            .add_func_pass(mir::passes::ConstMerge)
            .add_func_pass(mir::passes::Cse)
            .add_func_pass(mir::passes::Licm)
            .add_func_pass(mir::passes::SimplifyCfg)
            .add_func_pass(mir::passes::DeadCodeElim);
        pm.run(module);
    }

    /// The optimized pipeline collapses C integer-promotion round
    /// trips: `x > 0` lowers to `icmp` -> `zext` -> `icmp != 0`, and a
    /// branch must end up testing the comparison directly.
    #[test]
    fn optimized_pipeline_simplifies_conditions() {
        let src = "int check(int x) { if (x > 0 && x < 100) return 1; return 0; }\n\
                   int main() { return check(5); }";
        let mut module = lower_module(src);
        run_opt_pipeline(&mut module);
        let dump = mir::display::dump_module(&module);
        let check = module.funcs.iter().find(|f| f.name == "check").expect("check lowered");
        let zext_results: std::collections::HashSet<mir::Value> = check
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .filter_map(|inst| match &check.insts[*inst].kind {
                mir::InstKind::Zext(_) => check.insts[*inst].result,
                _ => None,
            })
            .collect();
        for inst in check.blocks.iter().flat_map(|block| block.insts.iter()) {
            if let mir::InstKind::Icmp { lhs, rhs, .. } = &check.insts[*inst].kind {
                assert!(!zext_results.contains(lhs) && !zext_results.contains(rhs), "promotion survived in check:\n{dump}");
            }
        }
        assert_eq!(jit_run_main(module), 1);
    }

    /// `(unsigned char)(x & 0xff)` keeps only the low byte: the mask is
    /// redundant under the truncation, and the result stays correct.
    #[test]
    fn optimized_pipeline_absorbs_covering_masks() {
        let src = "unsigned char low(unsigned x) { return (unsigned char)(x & 0xff); }\n\
                   int main() { return low(0x1ff); }";
        let mut module = lower_module(src);
        run_opt_pipeline(&mut module);
        let dump = mir::display::dump_module(&module);
        let low = module.funcs.iter().find(|f| f.name == "low").expect("low lowered");
        let has_and = low
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .any(|inst| matches!(low.insts[*inst].kind, mir::InstKind::And(..)));
        assert!(!has_and, "mask survived in low:\n{dump}");
        assert_eq!(jit_run_main(module), 0xff);
    }

    /// Two identical additions collapse to one shared value; the
    /// product keeps computing `(a + b) * (a + b)`.
    #[test]
    fn optimized_pipeline_eliminates_redundant_expressions() {
        let src = "int twice(int a, int b) { int x = a + b; int y = a + b; return x * y; }\n\
                   int main() { return twice(3, 4); }";
        let mut module = lower_module(src);
        run_opt_pipeline(&mut module);
        let dump = mir::display::dump_module(&module);
        let twice = module.funcs.iter().find(|f| f.name == "twice").expect("twice lowered");
        let adds = twice
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .filter(|inst| matches!(twice.insts[**inst].kind, mir::InstKind::Add(..)))
            .count();
        assert_eq!(adds, 1, "duplicate addition survived in twice:\n{dump}");
        assert_eq!(jit_run_main(module), 49);
    }

    /// The dead `if (0)` arm folds away, and its join block sinks into
    /// the entry, leaving a single return block.
    #[test]
    fn optimized_pipeline_sinks_dead_join_blocks() {
        let src = "int f(int x) { if (0) x = 1; return x; }\n\
                   int main() { return f(9); }";
        let mut module = lower_module(src);
        run_opt_pipeline(&mut module);
        let dump = mir::display::dump_module(&module);
        let f = module.funcs.iter().find(|f| f.name == "f").expect("f lowered");
        let live = f.blocks.iter().filter(|block| !matches!(block.term, Some(mir::Terminator::Unreachable))).count();
        assert_eq!(live, 1, "join block survived in f:\n{dump}");
        assert_eq!(jit_run_main(module), 9);
    }

    /// `__builtin_clrsbll` of the minimum i64 is zero; its lowering
    /// shifts the literal right by 63, which must sign-extend the
    /// pattern rather than shift the positive `i128` the parser sees.
    #[test]
    fn optimized_pipeline_computes_clrsb_of_min_i64() {
        let src = "int main() { return __builtin_clrsbll(0x8000000000000000ULL); }";
        let mut module = lower_module(src);
        run_opt_pipeline(&mut module);
        let dump = mir::display::dump_module(&module);
        assert_eq!(jit_run_main(module), 0, "clrsb of i64 min:\n{dump}");
    }

    /// The `setjmp` builtin lowers to a runtime call whose containing
    /// function must keep its slots addressable for the `longjmp` path.
    #[test]
    fn setjmp_builtin_marks_the_function() {
        let module = lower_module(
            "void f(void *b) { if (__builtin_setjmp(b)) __builtin_longjmp(b, 1); }\n\
             int g(void) { return 1; }",
        );
        let f = module.funcs.iter().find(|f| f.name == "f").expect("f lowered");
        assert!(f.returns_twice, "setjmp call did not mark f");
        let g = module.funcs.iter().find(|f| f.name == "g").expect("g lowered");
        assert!(!g.returns_twice);
    }

    /// `__attribute__((noinline))` on a prototype reaches the
    /// definition's MIR, so the inliner leaves the call site alone.
    #[test]
    fn noinline_prototype_blocks_inlining() {
        let src = "int g(int x) __attribute__((noinline));\n\
                   int g(int x) { return x + 1; }\n\
                   int main() { return g(41); }";
        let mut module = lower_module(src);
        let g = module.funcs.iter().find(|f| f.name == "g").expect("g lowered");
        assert!(g.noinline, "prototype attribute did not reach the definition");
        run_opt_pipeline(&mut module);
        let dump = mir::display::dump_module(&module);
        let main_fn = module.funcs.iter().find(|f| f.name == "main").expect("main lowered");
        let calls = main_fn
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .any(|inst| matches!(main_fn.insts[*inst].kind, mir::InstKind::Call { .. }));
        assert!(calls, "noinline call was inlined:\n{dump}");
        assert_eq!(jit_run_main(module), 42);
    }

    /// `x / 1` and `x % 1` fold away at -O2. The dividend loads a
    /// mutable global, so ConstFold cannot reach them: only the
    /// identity rules can.
    #[test]
    fn optimized_pipeline_folds_division_by_one() {
        let src = "int g = -7;\n\
                   int q(int x) { return x / 1; }\n\
                   int r(int x) { return x % 1; }\n\
                   int main() { return q(g) + r(g); }";
        let mut module = lower_module(src);
        run_opt_pipeline(&mut module);
        let dump = mir::display::dump_module(&module);
        let divides = module.funcs.iter().any(|f| {
            f.blocks.iter().flat_map(|b| b.insts.iter()).any(|inst| {
                matches!(
                    f.insts[*inst].kind,
                    mir::InstKind::Udiv(..) | mir::InstKind::Sdiv(..) | mir::InstKind::Urem(..) | mir::InstKind::Srem(..)
                )
            })
        });
        assert!(!divides, "division by one survived:\n{dump}");
        assert_eq!(jit_run_main(module), -7);
    }

    /// Lowering materializes one constant per literal occurrence, so
    /// `const-merge` must unify them before `cse` can recognize
    /// repeated expressions — in one block (`local`) and across a
    /// dominating branch (`arms`).
    #[test]
    fn optimized_pipeline_unifies_constants_before_cse() {
        let src = "int local(int x, int y) {\n\
                       int a = x * 13 + y * 5;\n\
                       int b = x * 13 + y * 5;\n\
                       return a + b;\n\
                   }\n\
                   int arms(int x, int y, int c) {\n\
                       int common = (x + 5) * (y - 2);\n\
                       if (c)\n\
                           return common + (x + 5) * (y - 2);\n\
                       return common - (x + 5) * (y - 2);\n\
                   }\n\
                   int main() { return local(2, 3) + arms(2, 3, 1); }";
        let mut module = lower_module(src);
        run_opt_pipeline(&mut module);
        let dump = mir::display::dump_module(&module);
        fn muls(module: &mir::Module, name: &str) -> usize {
            let func = module.funcs.iter().find(|f| f.name == name).expect("function is lowered");
            func.blocks
                .iter()
                .flat_map(|block| block.insts.iter())
                .filter(|inst| matches!(&func.insts[**inst].kind, mir::InstKind::Mul(..)))
                .count()
        }
        assert_eq!(muls(&module, "local"), 2, "one mul per distinct product:\n{dump}");
        assert_eq!(muls(&module, "arms"), 1, "both arms reuse the common product:\n{dump}");
        assert_eq!(jit_run_main(module), 96);
    }

    /// Work only the taken arm needs sinks into that arm instead of
    /// running on both paths.
    #[test]
    fn optimized_pipeline_sinks_branch_only_work() {
        let src = "int sink(int x, int take) {\n\
                       int product = (x + 11) * (x - 3);\n\
                       int mix = x * 7 + 9;\n\
                       if (take)\n\
                           return product + mix;\n\
                       return 123;\n\
                   }\n\
                   int main() { return sink(4, 0); }";
        let mut module = lower_module(src);
        run_opt_pipeline(&mut module);
        let dump = mir::display::dump_module(&module);
        let sink = module.funcs.iter().find(|f| f.name == "sink").expect("sink is lowered");
        assert!(sink.blocks[sink.entry].insts.len() <= 1, "the entry keeps only the branch test:\n{dump}");
        assert_eq!(jit_run_main(module), 123);
    }

    /// C text all the way to MIR for the host target (the JIT below
    /// executes in-process, so target and host must agree).
    fn lower_module(src: &str) -> mir::Module {
        lower_module_for(src, fe::target::Target::native())
    }

    /// `lower_module` for an explicit target.
    fn lower_module_for(src: &str, target: fe::target::Target) -> mir::Module {
        let mut interner = fe::lexer::Interner::new();
        let mut sema = fe::sema::Sema::new();
        sema.target = target;
        let tokens = fe::lexer::Lexer::new(src, &mut interner).lex_all();
        let mut parser = fe::parse::Parser::new(src, tokens, &mut sema, &mut interner);
        let tu = parser.parse_translation_unit();
        lower_translation_unit(&sema, &interner, &tu)
    }

    /// MSVC's headers define every CRT wrapper as a non-static
    /// `__inline` and rely on COMDAT folding or the identical CRT
    /// export. On the Windows target the body is dropped at
    /// declaration; a referenced wrapper is revived as a TU-local
    /// copy (names like `sprintf` have no export to resolve to).
    #[test]
    fn windows_msvc_extern_inline_definitions_become_imports() {
        let module = lower_module_for(
            "__inline int wrap(void) { return 7; } int main(void) { return wrap(); }",
            fe::target::Target::WINDOWS_X86_64,
        );
        let wrap = module.funcs.iter().find(|f| f.name == "wrap").expect("referenced wrapper revived");
        assert!(wrap.local, "the revived copy is TU-local");
        assert!(module.externs.iter().any(|e| e.name == "wrap"));
        assert!(module.funcs.iter().any(|f| f.name == "main"));

        // An unreferenced wrapper stays dropped: declaration-only.
        let module = lower_module_for("__inline int unused(void) { return 7; } int main(void) { return 0; }", fe::target::Target::WINDOWS_X86_64);
        assert!(!module.funcs.iter().any(|f| f.name == "unused"));
        assert!(module.externs.iter().any(|e| e.name == "unused"));

        // Static `__inline` bodies stay TU-local definitions.
        let module = lower_module_for(
            "static __inline int local(void) { return 7; } int main(void) { return local(); }",
            fe::target::Target::WINDOWS_X86_64,
        );
        assert!(module.funcs.iter().any(|f| f.name == "local"));

        // Other targets keep the plain C behavior: the definition
        // emits (glcc has no COMDAT, but ELF libc exports no twin).
        let module = lower_module_for("__inline int wrap(void) { return 7; } int main(void) { return wrap(); }", fe::target::Target::LP64);
        assert!(module.funcs.iter().any(|f| f.name == "wrap"));
    }

    /// `(MIR module, assembler text)` for `src`.
    fn lower_with_asm(src: &str) -> (mir::Module, String) {
        let mut interner = fe::lexer::Interner::new();
        let mut sema = fe::sema::Sema::new();
        let tokens = fe::lexer::Lexer::new(src, &mut interner).lex_all();
        let mut parser = fe::parse::Parser::new(src, tokens, &mut sema, &mut interner);
        let tu = parser.parse_translation_unit();
        let module = lower_translation_unit(&sema, &interner, &tu);
        (module, naked_asm_text(&sema, &interner, &tu))
    }

    #[test]
    fn asm_labels_rename_emitted_symbols() {
        // A GNU asm label names the symbol while the C spelling stays
        // the declaration key: EDG emits `_Z6strchrPci(...)
        // __asm__("strchr")` for the mangled C library overloads.
        let module = lower_module(
            "extern int renamed(int) __asm__(\"target\");\n\
             int defined(void) __asm__(\"elsewhere\") { return 1; }\n\
             int caller(void) { return defined() + renamed(1); }",
        );
        assert!(!module.funcs.iter().any(|f| f.name == "defined"));
        assert!(module.funcs.iter().any(|f| f.name == "elsewhere"));
        assert!(!module.externs.iter().any(|e| e.name == "renamed"));
        let (target_id, _) = module.externs.iter_enumerated().find(|(_, e)| e.name == "target").expect("import carries the label");
        let caller = module.funcs.iter().find(|f| f.name == "caller").unwrap();
        assert!(
            caller
                .insts
                .iter()
                .any(|i| matches!(&i.kind, mir::InstKind::CallExtern { target, .. } if *target == target_id)),
            "the call resolves to the labeled import"
        );
        assert!(
            caller.insts.iter().any(|i| matches!(
                &i.kind,
                mir::InstKind::Call { callee, .. } if module.funcs[*callee].name == "elsewhere"
            )),
            "the call resolves to the labeled definition"
        );
    }

    #[test]
    fn asm_labels_survive_to_the_jit() {
        // The C name keeps calling the labeled definition, and an
        // import still resolves through its label (libc's `rand`).
        assert_eq!(run_main("int f(void) __asm__(\"g\") { return 40; } int main(void) { return f() + 2; }"), 42);
        assert_eq!(run_main("int roll(void) __asm__(\"rand\"); int main(void) { return roll() >= 0; }"), 1);
    }

    #[test]
    fn naked_functions_emit_asm_text_and_import() {
        let (module, text) = lower_with_asm("__attribute__((naked)) int f(void) { __asm__(\"mov $42, %eax\\n\\tret\"); }\nint main(void){ return f(); }");
        assert!(text.contains(".globl f\n"), "{text}");
        assert!(text.contains("f:\n"), "{text}");
        assert!(text.contains("mov $42, %eax"), "{text}");
        // The C side calls an import; the body is not lowered.
        assert!(!module.funcs.iter().any(|f| f.name == "f"));
        assert!(module.externs.iter().any(|e| e.name == "f"));
    }

    #[test]
    fn section_attributes_reach_globals_and_naked_text() {
        let (module, text) = lower_with_asm(
            "__attribute__((section(\".vectors\"))) const unsigned long vec[2] = {1, 2};\
             __attribute__((section(\".noinit\"))) char scratch[8];\
             __attribute__((naked, section(\".mystart\"))) void boot(void) { __asm__(\"ret\"); }\
             int main(void){ return (int)vec[0] + scratch[0]; }",
        );
        assert_eq!(global_named(&module, "vec").section.as_deref(), Some(".vectors"));
        assert_eq!(global_named(&module, "scratch").section.as_deref(), Some(".noinit"));
        assert!(text.contains(".section .mystart,\"ax\"\n.globl boot\n"), "{text}");
        assert!(text.contains(".previous\n"), "{text}");
    }

    /// C text all the way to a JIT call returning `i32`.
    fn run_main(src: &str) -> i32 {
        jit_run_main(lower_module(src))
    }

    /// JIT `mir_module` and call its `main`.
    fn jit_run_main(mir_module: mir::Module) -> i32 {
        unsafe {
            let errors = mir::verify::verify_module(&mir_module);
            assert!(errors.is_empty(), "verify errors: {errors:?}");
            // The variadic prelude reads the frame pointer, like the
            // driver (which sets preserve_frame_pointers), and Win64
            // F128/I128 follow LLVM's ABI there.
            let mut flags = vec![("preserve_frame_pointers", "true")];
            if cfg!(windows) {
                flags.push(("enable_llvm_abi_extensions", "true"));
            }
            let builder = JITBuilder::with_flags(&flags, cranelift_module::default_libcall_names()).expect("jit builder");
            let mut jit = JITModule::new(builder);
            let ids = glcc_lower_cl::lower_module(&mir_module, &mut jit, glcc_lower_cl::Target::native());
            jit.finalize_definitions().expect("jit finalize");
            let main = *ids.iter().find(|(id, _)| mir_module.funcs[**id].name == "main").map(|(_, fid)| fid).expect("main function");
            let addr = jit.get_finalized_function(main);
            let func: extern "C" fn() -> i32 = std::mem::transmute(addr);
            func()
        }
    }

    #[test]
    fn vla_objects() {
        // A runtime bound allocates, indexes, and round-trips values.
        assert_eq!(run_main("int main(){int n = 5; int a[n]; for (int i = 0; i < n; i++) a[i] = i * i; return a[4];}"), 16);
        // Rows of a multidimensional VLA stride by the runtime extent.
        assert_eq!(
            run_main("int main(){int n = 3, m = 4; int a[n][m]; int i, j; for (i = 0; i < n; i++) for (j = 0; j < m; j++) a[i][j] = i * m + j; return a[2][3];}"),
            11
        );
        // The declaration captures its extents: later writes to the
        // bound variable do not resize the object.
        assert_eq!(run_main("int main(){int n = 4; int a[n]; n = 99; return sizeof a;}"), 16);
        // `sizeof` on a type-name evaluates the bounds where it is used.
        assert_eq!(run_main("int main(){int n = 6; typedef int t[n]; return sizeof(t);}"), 24);
        // Through a pointer-to-VLA the bound re-evaluates: both the row
        // stride and `sizeof *a` read the parameter.
        assert_eq!(
            run_main("int bar(int n, char a[2][n]){a[1][n - 1] = 7; return sizeof *a;} int main(){char x[2][5]; return bar(5, x);}"),
            5
        );
        // Pointer arithmetic steps by the runtime extent: `p + 2 - p`,
        // `*p + 3 - *p`, `p - i`, and element addresses all scale with
        // the variably modified pointee (tcctest.c's `c99_vla_test_2`).
        assert_eq!(run_main("int main(){int m = 4; int a[3][4]; int (*p)[m] = a; return (p + 2) - p;}"), 2);
        assert_eq!(run_main("int main(){int m = 4; int a[3][4]; int (*p)[m] = a; return (*p + 3) - *p;}"), 3);
        assert_eq!(run_main("int main(){int m = 4; int a[3][4]; int (*p)[m] = a; p = p + 2; p = p - 1; return p - a;}"), 1);
        assert_eq!(run_main("int main(){int m = 4; int a[3][4]; int (*p)[m] = a; return &p[2][0] - &p[0][0];}"), 8);
    }

    #[test]
    fn static_initializers_fold_address_conditions() {
        // Address constants fold as ternary conditions in initializers
        // (they are not integer constant expressions; GCC folds them
        // there — tcctest.c's `sinit23`).
        assert_eq!(
            run_main("int obj; int x[] = { \"astring\" ? sizeof(\"astring\") : -1, &obj ? 42 : -1 }; int main(){ return x[0] * 10 + x[1]; }"),
            122
        );
        assert_eq!(run_main("int obj; int x = !&obj ? 1 : 2; int main(){ return x; }"), 2);
        // Logical operators short-circuit an unevaluated operand:
        // `2 || 1 / 0` is the constant 1 (`sinit24`).
        assert_eq!(run_main("int x = 2 || 1 / 0; int main(){ return x; }"), 1);
        assert_eq!(run_main("int x = 0 && 1 / 0; int main(){ return x; }"), 0);
    }

    #[test]
    #[should_panic(expected = "non-constant global initializer")]
    fn weak_address_conditions_do_not_fold() {
        // A weak symbol may resolve to zero, so its address is no
        // initializer constant (GCC: "initializer element is not
        // constant"; tcctest.c's comment).
        run_main("extern int w __attribute__((weak)); int x = &w ? 1 : 2; int main(){ return x; }");
    }

    #[test]
    fn braced_string_initializers_are_the_bare_shape() {
        // The braced lone-string form is the bare string with braces
        // (C17 §6.7.9p14): sized with the NUL and lowered the same way
        // (tcctest.c's `str_ag2`).
        assert_eq!(run_main("char s[] = { \"b\" }; int main(){ return sizeof(s) + s[0] - 'b'; }"), 2);
        assert_eq!(run_main("char s[4] = { \"abc\" }; int main(){ return s[2] == 'c'; }"), 1);
        // Locals and compound literals take the same shape.
        assert_eq!(run_main("int main(){ char s[] = { \"xy\" }; return sizeof(s) + s[1] - 'y'; }"), 3);
        assert_eq!(run_main("int main(){ char *s = (char[]){ \"ab\" }; return s[1] == 'b'; }"), 1);
        // A nested array still initializes its first row from the
        // string; the braces are not redundant there.
        assert_eq!(run_main("char m[2][3] = { \"ab\" }; int main(){ return m[0][1] == 'b' && m[1][0] == 0; }"), 1);
    }

    #[test]
    fn full_initializers_skip_the_redundant_zero_fill() {
        let main_body = |src| {
            let dump = mir::display::dump_module(&lower_module(src));
            dump.split("func @main").nth(1).expect("main is lowered").to_string()
        };

        let full = main_body("int main(){ long long a[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}; return (int)a[15]; }");
        assert!(!full.contains("memset"), "{full}");

        let partial = main_body("int main(){ long long a[16] = {1}; return (int)a[15]; }");
        assert!(partial.contains("memset"), "{partial}");

        let designated = main_body("struct S{int a,b;}; int main(){ struct S s = {.a = 1}; return s.a; }");
        assert!(designated.contains("memset"), "{designated}");

        let full_string = main_body("int main(){ char s[4] = \"abc\"; return s[3]; }");
        assert!(!full_string.contains("memset"), "{full_string}");

        let partial_string = main_body("int main(){ char s[8] = \"abc\"; return s[7]; }");
        assert!(partial_string.contains("memset"), "{partial_string}");
    }

    #[test]
    fn vla_frees_at_every_return() {
        // The declaration site frees its own previous block; each of the
        // two returns drains the live allocation. No drain loop: the
        // only branch is the source `if`.
        let module = lower_module("int f(int n){ if (n) { int a[n]; a[0] = 1; return a[0]; } return 2; } int main(){return f(1);}");
        let free = module.externs.iter().position(|e| e.name == "free").expect("free import");
        let dump = mir::display::dump_module(&module);
        let f = dump.split("func @main").next().unwrap();
        assert_eq!(f.matches(&format!("call.ext e{free}(")).count(), 3, "{f}");
        assert_eq!(f.matches("brif").count(), 1, "{f}");
    }

    #[test]
    fn vla_loop_reuses_its_storage() {
        // Each iteration frees the previous block before allocating, so
        // `malloc` hands back the same storage.
        assert_eq!(
            run_main("int main(){int i; char *first = 0; for (i = 0; i < 4; i++){ char a[4096]; if (i == 0) first = a; else if (a != first) return 1; } return 0;}"),
            0
        );
    }

    #[test]
    fn arithmetic_return() {
        assert_eq!(run_main("int main(){return 2+3*4;}"), 14);
        assert_eq!(run_main("int main(){return (10-3)*2;}"), 14);
        assert_eq!(run_main("int main(){return -5+12;}"), 7);
    }

    #[test]
    fn switch_dispatch() {
        let body = "int f(int x){switch(x){case 1: return 10; case 2: case 3: return 20; default: return 30;}} int main(){return f(1)*1000+f(2)*100+f(3)+f(9);}";
        // 10, 20, 20, 30.
        assert_eq!(run_main(body), 10 * 1000 + 20 * 100 + 20 + 30);
        // Fallthrough accumulates without `break`.
        assert_eq!(
            run_main("int main(){int s = 0; switch(2){case 1: s += 1; case 2: s += 10; case 3: s += 100; default: s += 1000;} return s;}"),
            1110
        );
        // `break` leaves the switch, not the function.
        assert_eq!(run_main("int main(){int s = 0; switch(1){case 1: s = 7; break; default: s = 9;} return s;}"), 7);
        // Empty body and bare default still dispatch.
        assert_eq!(run_main("int main(){switch(5){} return 1;}"), 1);
        assert_eq!(run_main("int main(){switch(5){default: return 4;}}"), 4);
    }

    #[test]
    fn switch_promotion() {
        // Narrow controlling expressions compare post-promotion: a
        // signed `char` -1 must not match `case 255` (spelled out
        // because plain `char` is unsigned on aarch64).
        assert_eq!(
            run_main("int main(){signed char c = -1; switch(c){case 255: return 1; case -1: return 2; default: return 3;}}"),
            2
        );
        // Plain `char` follows the target ABI: signed on x86-64 (so -1
        // matches `case -1`), unsigned on aarch64 (0xFF matches
        // `case 255`).
        let plain_char = if fe::target::Target::native().char_signed { 2 } else { 1 };
        assert_eq!(
            run_main("int main(){char c = -1; switch(c){case 255: return 1; case -1: return 2; default: return 3;}}"),
            plain_char
        );
        assert_eq!(
            run_main("int main(){unsigned char c = 255; switch(c){case 255: return 1; case -1: return 2; default: return 3;}}"),
            1
        );
        assert_eq!(run_main("int main(){short s = 4464; switch(s){case 4464: return 1; default: return 2;}}"), 1);
    }

    #[test]
    fn switch_nesting_and_loops() {
        // `break` in a loop in a switch exits the loop; `break` in a
        // switch in a loop exits the switch; `continue` still finds
        // the loop across the switch.
        assert_eq!(
            run_main("int main(){int s = 0; switch(1){case 1: for(int i = 0; i < 10; i++){if(i == 3) break; s += 1;} s += 100;} return s;}"),
            103
        );
        assert_eq!(
            run_main("int main(){int s = 0; for(int i = 0; i < 5; i++){switch(i){case 2: break; default: s += 1;}} return s;}"),
            4
        );
        assert_eq!(
            run_main("int main(){int s = 0; for(int i = 0; i < 5; i++){switch(i){case 1: case 3: continue; default: s += 1;}} return s;}"),
            3
        );
        // Nested switches dispatch independently.
        assert_eq!(
            run_main("int main(){int s = 0; switch(1){case 1: switch(2){case 2: s = 12; break; default: s = 10;} break; default: s = 1;} return s;}"),
            12
        );
    }

    #[test]
    fn switch_duff() {
        // Cases may live inside loops in the switch body.
        assert_eq!(
            run_main("int main(){int n = 0; int i = 0; switch(2){case 0: i++; case 2: i += 10; for(; i < 12; i++){n++; if(i == 11) break;} } return n * 100 + i;}"),
            211
        );
    }

    #[test]
    fn switch_wide() {
        // 128-bit controlling expressions compare full values through
        // a conditional chain rather than the 64-bit switch.
        assert_eq!(
            run_main("int main(){__int128 x = ((__int128)1 << 100) + 5; switch(x){case 5: return 1; default: return 2;}}"),
            2
        );
        assert_eq!(
            run_main("int main(){__int128 x = ((__int128)1 << 100) + 5; switch(x){case 5: return 1; case (((__int128)1 << 100) + 5): return 3; default: return 2;}}"),
            3
        );
    }

    #[test]
    fn goto_labels() {
        // Forward and backward jumps, including over declarations.
        assert_eq!(run_main("int main(){goto mid; int x = 1; mid: return 42;}"), 42);
        assert_eq!(run_main("int main(){int i = 0; int s = 0; again: i++; s += i; if(i < 10) goto again; return s;}"), 55);
        // Jumping out of a loop, and into a switch body.
        assert_eq!(run_main("int main(){for(int i = 0; i < 100; i++){if(i == 4) goto done;} done: return 7;}"), 7);
        assert_eq!(run_main("int main(){goto inner; switch(1){case 1: return 1; inner: return 2; default: return 3;}}"), 2);
    }

    #[test]
    fn computed_goto() {
        // A label address held in a local, dispatched on.
        assert_eq!(run_main("int main(){void *p = &&l; goto *p; l: return 5;}"), 5);
        // Jumping between labels through a table, with arithmetic on
        // the addresses (`&&b - &&a` counts labels).
        assert_eq!(
            run_main(
                "int main(){static void *t[] = {&&a, &&b, &&c}; int i = 0, s = 0; goto *t[i++]; \
                 a: s += 10; goto *t[i++]; b: s += 100; goto *t[i++]; c: return s;}"
            ),
            110
        );
        // Label ids also serve static integer initializers, and label
        // differences stay consecutive.
        assert_eq!(
            run_main(
                "int main(){static int d[] = {&&b - &&a, &&c - &&a}; void *p = &&a + d[1]; \
                 goto *p; a: return 1; b: return 2; c: return 3;}"
            ),
            3
        );
    }

    #[test]
    fn computed_goto_in_statement_expression_scope() {
        // A label defined beside a statement expression's `goto` stays
        // local to it: the jump resolves inside.
        assert_eq!(run_main("int main(){int x = 0; ({ l: x++; if (x < 3) goto l; }); return x;}"), 3);
    }

    #[test]
    fn local_labels_shadow_function_labels() {
        // GNU `__label__` labels are block scoped: the inner LBL loops
        // once, then the `goto` outside the inner block targets the
        // function-scoped label of the same spelling
        // (tcctest.c's `statement_expr_test`).
        assert_eq!(
            run_main("int main(){ int i = 0; ({ { __label__ LBL; LBL: if (i++ == 0) goto LBL; } goto LBL; }); LBL: return i; }"),
            2
        );
        // A statement-expression label set that cycles through itself
        // and only ends at the block's fallthrough (tcctest.c's
        // `local_label_test`): colliding with the enclosing labels made
        // this loop forever.
        assert_eq!(
            run_main(
                "int main(){ int a; goto l1; \
                 l2: a = 1 + ({ __label__ l1, l2, l3, l4; goto l1; \
                 l4: goto l3; l2: goto l4; l1: goto l2; l3:; 1; }); \
                 return a; \
                 l4: goto l2; \
                 l1: goto l4; }"
            ),
            2
        );
    }

    #[test]
    fn statement_expression_value_through_label() {
        // A trailing label does not hide the value (GCC: the value of
        // the last statement; tcctest.c relies on this for
        // `({ __label__ here; here: (unsigned long)&&here; })`), and
        // the label still defines its jump target for computed gotos.
        assert_eq!(run_main("int main(){int x = ({ l: 42; }); return x;}"), 42);
        assert_eq!(run_main("int main(){int x = ({ a: b: 43; }); return x;}"), 43);
        assert_eq!(
            run_main("int main(){int s = 0; int v = ({ void *p; l: p = &&l, s < 2 ? ({ s++; goto *p; 0; }) : 66; }); return v + s;}"),
            68
        );
    }

    #[test]
    fn member_access_on_aggregate_rvalues() {
        assert_eq!(
            run_main("struct S{int x;int y[2];}; struct S f(void){struct S s; s.x=3; s.y[1]=4; return s;} int main(){return f().x + f().y[1];}"),
            7
        );
        assert_eq!(
            run_main("struct S{int x;}; int main(){struct S a,b; a.x=1; b.x=2; return (1 ? a : b).x + ({ struct S t; t.x=4; t; }).x;}"),
            5
        );
    }

    #[test]
    fn alloca_memory_is_usable() {
        assert_eq!(run_main("int main(){char *p = __builtin_alloca(4); p[0] = 3; return p[0];}"), 3);
        assert_eq!(run_main("int main(){int *p = (int *)alloca(sizeof(int) * 4); p[3] = 7; return p[3];}"), 7);
        // malloc's 16-byte alignment survives the header.
        assert_eq!(run_main("int main(){return ((unsigned long)__builtin_alloca(1) % 16) == 0 ? 5 : 6;}"), 5);
    }

    #[test]
    fn alloca_frees_at_every_return() {
        // Every return routes through its own drain loop, so an
        // allocation from an earlier loop iteration is freed even when
        // the return precedes the `alloca` in source order.
        let module = lower_module("int f(int n){ if (n) { char *p = __builtin_alloca(8); p[0] = 1; return p[0]; } return 2; } int main(){return f(1);}");
        let frees = module.externs.iter().filter(|e| e.name == "free").count();
        assert_eq!(frees, 1, "one shared `free` import");
        let free = module.externs.iter().position(|e| e.name == "free").expect("free import");
        let dump = mir::display::dump_module(&module);
        let f = dump.split("func @main").next().unwrap();
        assert_eq!(f.matches(&format!("call.ext e{free}(")).count(), 2, "{f}");
        // Two drain checks on top of the source `if`.
        assert_eq!(f.matches("brif").count(), 3, "{f}");
    }

    #[test]
    fn overflow_builtins() {
        // The wrapped result stores through the pointer; the return
        // value says whether the exact value fit.
        assert_eq!(
            run_main("int main(){int r; return __builtin_add_overflow(2147483647, 1, &r) + (r == (-2147483647 - 1)) * 2;}"),
            3
        );
        assert_eq!(
            run_main("int main(){unsigned char r; return __builtin_sub_overflow((unsigned char)3, (unsigned char)5, &r) + (r == 254) * 2;}"),
            3
        );
        // `u64 * u64` overflows i128: the exact u128 product compares
        // unsigned against the result type's maximum.
        assert_eq!(
            run_main("int main(){unsigned long r; return __builtin_mul_overflow(0x100000000ul, 0x100000000ul, &r) + (r == 0) * 2;}"),
            3
        );
        assert_eq!(
            run_main("int main(){long long r; return __builtin_mul_overflow(0x100000000ull, 0x80000000ull, &r) + (r == (-9223372036854775807LL - 1)) * 2;}"),
            3
        );
        assert_eq!(
            run_main("int main(){long long r; return __builtin_mul_overflow(0x100000000ull, 0x7FFFFFFFull, &r) + (r == 0x7FFFFFFF00000000LL) * 2;}"),
            2
        );
        // Signed operands multiply exactly in i128.
        assert_eq!(
            run_main("int main(){long long r; return __builtin_mul_overflow(-4611686018427387904LL, 4LL, &r) + (r == 0) * 2;}"),
            3
        );
    }

    #[test]
    fn frame_address_builtins_walk_the_frame_chain() {
        // Level 0 is the function's own frame, so the caller's locals
        // sit above it and level 1 (the caller's frame) above them.
        // Both levels of return address are live and distinct.
        assert_eq!(
            run_main(
                "int probe(int *a, int *b) {\
                     char *fp0 = (char *) __builtin_frame_address(0);\
                     char *fp1 = (char *) __builtin_frame_address(1);\
                     if ((char *) a <= fp0) return 1;\
                     if ((char *) b <= fp0) return 2;\
                     if ((char *) a >= fp1) return 3;\
                     if ((char *) b >= fp1) return 4;\
                     if (!__builtin_return_address(0)) return 5;\
                     if (__builtin_return_address(1) == __builtin_return_address(0)) return 6;\
                     return 0;\
                 }\
                 int main() { int a = 1; int b = 2; return probe(&a, &b); }"
            ),
            0
        );
    }

    #[test]
    fn sizeof_string_literal_lowers() {
        // The literal's type carries its bound (NUL included), so
        // `sizeof` lowers even where it does not fold.
        assert_eq!(run_main("int main(){return (int) sizeof \"ab\" + (int) sizeof \"a\\n\";}"), 6);
    }

    #[test]
    fn packed_and_over_aligned_structs_lower() {
        // `packed` members drop padding (so `d` sits at offset 2 and
        // the loads are unaligned) while an `aligned` struct keeps its
        // padded size and stride.
        assert_eq!(
            run_main(
                "struct P { short s __attribute__((packed)); double d __attribute__((packed)); };\
                 int main(){ struct P p; p.s = 3; p.d = 4.5; return p.s + (int) p.d + (sizeof(struct P) == 10); }"
            ),
            8
        );
        assert_eq!(
            run_main(
                "struct X { int a; int b; } __attribute__((aligned(32)));\
                 struct Y { struct X x[2]; int c; };\
                 struct Y y[2];\
                 int main(){ if (((unsigned long) &y[0]) % 32) return 1; return (int) ((char *) &y[1] - (char *) &y[0]); }"
            ),
            96
        );
    }

    #[test]
    fn alias_attribute_shares_storage() {
        // The alias names the target's storage: writes through either
        // spelling land in the same object, and no symbol is emitted
        // for the alias.
        assert_eq!(
            run_main(
                "int a[4]; extern int b[4] __attribute__((alias(\"a\")));\
                 int main(){ b[2] = 1; a[2] = 2; return b[2]; }"
            ),
            2
        );
        // Function aliases resolve to the target's definition.
        assert_eq!(
            run_main(
                "int g(void) { return 7; } extern int f(void) __attribute__((alias(\"g\")));\
                 int main(){ return f(); }"
            ),
            7
        );
    }

    #[test]
    fn bit_and_sign_builtins() {
        // `clrsb` counts the redundant sign bits; zero counts `w - 1`.
        assert_eq!(run_main("int main(){ return __builtin_clrsb(0); }"), 31);
        assert_eq!(run_main("int main(){ return __builtin_clrsb(1); }"), 30);
        assert_eq!(run_main("int main(){ return __builtin_clrsb(0x0F); }"), 27);
        assert_eq!(run_main("int main(){ return __builtin_clrsb(-1); }"), 31);
        assert_eq!(run_main("int main(){ return __builtin_clrsbll(1LL); }"), 62);
        // `parity` is the low bit of the popcount.
        assert_eq!(run_main("int main(){ return __builtin_parity(7); }"), 1);
        assert_eq!(run_main("int main(){ return __builtin_parity(6); }"), 0);
        // `copysign` takes the magnitude of the first operand and the
        // sign bit of the second, at the bit level.
        assert_eq!(run_main("int main(){ return __builtin_copysign(1.0, -2.0) < 0.0 ? 1 : 2; }"), 1);
        assert_eq!(run_main("int main(){ return __builtin_copysign(-1.0, 2.0) > 0.0 ? 1 : 2; }"), 1);
        assert_eq!(run_main("int main(){ return __builtin_copysignf(-1.0f, 2.0f) > 0.0f ? 1 : 2; }"), 1);
        // The `__builtin_` spellings rewrite to the libc names, so plain
        // calls expand inline too, like the other libm builtins.
        assert_eq!(
            run_main(
                "double copysign(double, double);\n\
                 int main(){ return copysign(-1.0, 2.0) > 0.0 ? 1 : 2; }"
            ),
            1
        );
        // `__builtin_choose_expr` and `?:` evaluate one arm only.
        assert_eq!(run_main("int main(){ int y = 0; int z = __builtin_choose_expr(1, 3, y++); return z + y; }"), 3);
        assert_eq!(run_main("int main(){ int y = 0; return (y ?: 4); }"), 4);
        assert_eq!(run_main("int main(){ int y = 2; return (y ?: 4); }"), 2);
    }

    #[test]
    fn case_ranges_and_member_address_constants() {
        // GNU case ranges dispatch with comparisons in the promoted
        // type's signedness, alongside exact cases.
        assert_eq!(
            run_main(
                "int f(unsigned long long x) { switch (x) { case 1 ... 5: return 1; case 6: return 2; case 100 ... 200: return 3; default: return 4; } }\n\
                 int main() { return f(3) + f(6) * 10 + f(150) * 100 + f(7) * 1000; }"
            ),
            1 + 20 + 300 + 4000
        );
        assert_eq!(
            run_main(
                "int g(int x) { switch (x) { case -5 ... -1: return 1; case 0: return 2; default: return 3; } }\n\
                 int main() { return g(-3) + g(0) * 10 + g(1) * 100; }"
            ),
            1 + 20 + 300
        );
        // `&p->m` and `p + k` are address constants: the root object
        // plus the member offset and stride.
        assert_eq!(
            run_main(
                "typedef struct { int a; char *name; } FOO;\n\
                 FOO items[] = { {1, \"1\"}, {2, \"2\"}, {0, 0} };\n\
                 int *second = (int *)&((items + 1)->a);\n\
                 int *first = (int *)&(items->a);\n\
                 int main() { return *second * 10 + *first; }"
            ),
            21
        );
    }

    #[test]
    fn abs_and_rounding_builtins_expand_inline() {
        // `abs`/`labs`/`llabs` compute the magnitude without a call,
        // even when the program declares the name: GCC expands the
        // builtin at direct calls, which is what makes tests that
        // define an aborting `llabs` still pass.
        assert_eq!(
            run_main(
                "long long llabs(long long);\n\
                 int main() { long long v = -3; return (int)(llabs(v) * 100 + __builtin_labs(-2L) * 10 + __builtin_abs(-1)); }"
            ),
            321
        );
        // Floats clear the sign bit, so `-0.0` comes back positive.
        assert_eq!(
            run_main(
                "double fabs(double);\n\
                 int main() { double z = -0.0; double r = fabs(z); return (r == 0.0 && !__builtin_signbit(r)) ? 0 : 1; }"
            ),
            0
        );
        // `floor`/`ceil` lower to MIR rounding instructions.
        assert_eq!(
            run_main(
                "double floor(double);\n\
                 int main() { double a = 1.7; double b = -1.2; return (int)(floor(a) * 10 + __builtin_ceil(b) + 2); }"
            ),
            11
        );
    }

    #[test]
    fn compound_assignment_evaluates_the_lvalue_once() {
        // The lvalue's side effects run exactly once (C17 §6.5.16.2):
        // its address feeds the read, the operation, and the store.
        assert_eq!(
            run_main(
                "int main() {\n\
                 int a[4] = {1, 2, 3, 4};\n\
                 int i = 0;\n\
                 int s = 0;\n\
                 a[i++] += 10;\n\
                 s += a[0] * 100 + i;\n\
                 i = 0;\n\
                 a[i++] *= 2;\n\
                 s += a[0] * 10000 + i;\n\
                 return s;\n\
                 }"
            ),
            11 * 100 + 1 + 22 * 10000 + 1
        );
        // Pointer targets step once as well.
        assert_eq!(run_main("int main() { char b[4] = {0, 0, 0, 0}; char *p = b; *p++ += 3; return b[0] * 10 + (p - b); }"), 31);
        // Bit-field targets read-modify-write their covering unit and
        // evaluate to the stored (width-clamped) value.
        assert_eq!(
            run_main(
                "struct S { unsigned f : 4; };\n\
                 int main() { struct S s = {3}; int i = 0; int arr[2] = {0, 0}; arr[i++] += 1; s.f += 5; return arr[0] * 100 + (int)s.f * 10 + i; }"
            ),
            100 + 80 + 1
        );
        // Pointer arithmetic scales by the pointee size.
        assert_eq!(
            run_main("int main() { int a[4] = {1, 2, 3, 4}; int *p = a; p += 2; *p -= 1; return a[2] * 10 + (p - a); }"),
            2 * 10 + 2
        );
        // The combination runs in the operator's type even when the
        // lvalue is narrower: `g += x` sums as `long double` and
        // converts back to `int` for the store.
        assert_eq!(
            run_main(
                "int g;\n\
                 void h(double x) { g += x; }\n\
                 int main() { h(3.75); return g; }"
            ),
            3
        );
        let module = lower_module("int g;\nvoid h(long double x) { g += x; }");
        let ld = mir::Ty::Float(match fe::target::Target::native().long_double {
            fe::target::LongDoubleKind::F80 => mir::FloatKind::F80,
            fe::target::LongDoubleKind::F128 => mir::FloatKind::F128,
            fe::target::LongDoubleKind::F64 => mir::FloatKind::F64,
        });
        let mut sums = 0;
        for func in &module.funcs {
            for inst in &func.insts {
                if matches!(&inst.kind, mir::InstKind::FAdd(..)) {
                    let result = inst.result.expect("FAdd has a result");
                    assert_eq!(func.values[result].ty, ld, "FAdd result type");
                    sums += 1;
                }
            }
        }
        assert_eq!(sums, 1);
    }

    #[test]
    fn old_style_calls_take_declared_arity() {
        // A definition without a prototype fixes its own arity: extra
        // arguments still evaluate (their side effects happen), but
        // only the declared parameters pass.
        assert_eq!(run_main("f(){return 3;} int main(){int x = 0; return f(x = 1) + x;}"), 4);
        assert_eq!(run_main("int g; g2(){g = 7; return 1;} int main(){return g2(1, 2) + g;}"), 8);
    }

    #[test]
    fn builtin_fp_class_classify_and_prefetch() {
        // `signbit` sees the sign bit (`-0.0` counts) and converts
        // integers to `double` first.
        assert_eq!(run_main("int main(){return __builtin_signbit(-0.0) + 2 * __builtin_signbit(0.0);}"), 1);
        assert_eq!(run_main("int main(){return __builtin_signbit(-1) ? 3 : 4;}"), 3);
        // Long doubles classify through their own sign bit (the JIT
        // lacks the x87 negate helper, so this stays positive).
        assert_eq!(run_main("int main(){long double x = 2.5L; return __builtin_signbit(x) ? 5 : 6;}"), 6);
        // Classify-type codes fold from the type alone, so a long
        // double field is real (the 20040709 pattern).
        assert_eq!(run_main("int main(){return __builtin_classify_type(1.5);}"), 8);
        assert_eq!(run_main("int main(){return __builtin_classify_type(1);}"), 1);
        assert_eq!(
            run_main("struct W { long double l; }; int main(){struct W w; return __builtin_classify_type(w.l) == 8 ? 7 : 8;}"),
            7
        );
        // The prefetch address still evaluates; the hint itself is
        // nothing.
        assert_eq!(run_main("int main(){int *p = 0; __builtin_prefetch(p, 0, 3); return 0;}"), 0);
    }

    #[test]
    fn recursive_structs() {
        // Self-reference through pointers: traversal, multi-hop
        // arrows, and dereference-then-dot all resolve per hop.
        let list = "struct Node{int v; struct Node *next;};\
            int sum(struct Node *p){int s = 0; while(p){s += p->v; p = p->next;} return s;}\
            int main(){struct Node n1; struct Node n2; struct Node n3;\
            n1.v = 10; n2.v = 20; n3.v = 30;\
            n1.next = &n2; n2.next = &n3; n3.next = 0;\
            return sum(&n1) * 100 + n1.next->next->v + (*n2.next).v;}";
        // 60, 30, 30.
        assert_eq!(run_main(list), 60 * 100 + 30 + 30);
        // Struct arrays through pointers scale by element size.
        assert_eq!(
            run_main(
                "struct P{int x; int y;}; struct P a[3] = {{1,2},{3,4},{5,6}};\
                int at(struct P *p, int i){return p[i].x + p[i].y;}\
                int main(){return at(a, 0) + at(a, 1) * 10 + at(a, 2) * 100;}"
            ),
            3 + 7 * 10 + 11 * 100
        );
        // Mutual recursion across two definitions.
        assert_eq!(
            run_main(
                "struct A{struct B *b; int x;}; struct B{struct A *a; int y;};\
                int main(){struct A a; struct B b;\
                a.x = 5; b.y = 7; a.b = &b; b.a = &a;\
                return a.b->y * 10 + b.a->x;}"
            ),
            75
        );
        // Pointers to never-defined structs pass through opaquely.
        assert_eq!(
            run_main(
                "int isnull(struct Opaque *p){return p == 0;}\
                int main(){return isnull(0) + isnull((void *)1);}"
            ),
            1
        );
    }

    #[test]
    #[should_panic(expected = "embeds itself by value")]
    fn recursive_struct_by_value_panics() {
        // Built below the parser (which rejects by-value recursion with
        // its own diagnostic): the lowering backstop still fires.
        let mut interner = fe::lexer::Interner::new();
        let mut sema = fe::sema::Sema::new();
        let tag = interner.get_or_intern("S");
        let span = fe::span::Span::dummy();
        let id = sema.define_tag(tag, false, span).unwrap();
        let v = interner.get_or_intern("v");
        sema.complete_tag(
            id,
            vec![fe::sema::Member {
                name: Some(v),
                ty: fe::parse::Ty::Struct(id),
                bitfield: None,
                packed: false,
                align_override: None,
                span,
            }],
        )
        .unwrap();
        let _ = crate::ty::lower_ty(&fe::parse::Ty::Struct(id), &sema, &interner);
    }

    #[test]
    fn call_result_aggregate_assign() {
        // A register-sized aggregate call result is an address like
        // any other aggregate value: storing it to a local must copy
        // from the reassembled slot, not the raw register.
        assert_eq!(
            run_main(
                "struct P{int x; int y;}; struct P mk(int n){struct P p; p.x = n; p.y = n+1; return p;}\
                int main(){struct P p = mk(10); return p.x + p.y;}"
            ),
            21
        );
        assert_eq!(
            run_main(
                "union N{int a; int b;}; union N mk(int n){union N u; u.b = n; return u;}\
                int main(){union N w = mk(7); return w.a;}"
            ),
            7
        );
    }

    #[test]
    fn unions() {
        // Overlaid members share storage; access resolves per member.
        assert_eq!(run_main("union N{int a; int b;}; int main(){union N u; u.a = 42; return u.a + u.b;}"), 84);
        assert_eq!(run_main("union N{int a; int b;}; int main(){union N u; u.b = 42; return u.a;}"), 42);
        // By value in both directions, including larger blobs.
        assert_eq!(
            run_main(
                "union N{int a; int b;}; int get(union N u){return u.a;}\
                int main(){union N u; u.b = 9; return get(u);}"
            ),
            9
        );
        assert_eq!(
            run_main(
                "union Big{long l[3]; int i;}; long sum(union Big u){return u.l[0]+u.l[1]+u.l[2];}\
                int main(){union Big u; u.l[0] = 1; u.l[1] = 2; u.l[2] = 3; return (int)sum(u);}"
            ),
            6
        );
        // Nested, designated, and global initializers address offset
        // zero with the selected member's type.
        assert_eq!(run_main("union N{int a; int b;}; int main(){union N u = {5}; return u.a + u.b;}"), 10);
        assert_eq!(run_main("union N{int a; int b;}; int main(){union N u = {.b = 6}; return u.a;}"), 6);
        assert_eq!(run_main("union N{int a; int b;}; union N g = {7}; int main(){return g.a + g.b;}"), 14);
        assert_eq!(
            run_main("union N{int a; int b;}; struct W{union N u; int t;}; int main(){struct W w; w.u.a = 3; w.t = 4; return w.u.b + w.t;}"),
            7
        );
    }

    #[test]
    fn bitfields() {
        // Reads extract and sign-extend; unsigned fields zero-extend
        // even though narrow ones promote to signed `int`.
        assert_eq!(
            run_main("struct S{int x:3; unsigned y:5;}; int main(){struct S s; s.x = -3; s.y = 17; return s.x * 100 + s.y;}"),
            -300 + 17
        );
        assert_eq!(
            run_main("struct S{int x:3; int y:5;}; int main(){struct S s; s.x = 1; s.y = 2; s.x++; ++s.y; return s.x * 100 + s.y;}"),
            203
        );
        // Assignment evaluates to the stored value; out-of-range
        // stores wrap like GCC (which warns and yields -12 here).
        assert_eq!(run_main("struct S{int x:3; int y:5;}; int main(){struct S s; int q = (s.y = 20); return q;}"), -12);
        // Shared units preserve neighbors; `:0` and unnamed widths
        // shape layout without values.
        assert_eq!(
            run_main("struct S{int a:3; int :5; int b:8;}; int main(){struct S s; s.a = -1; s.b = -56; return s.a * 100 + s.b;}"),
            -156
        );
        assert_eq!(
            run_main("struct S{int a:4; int :0; int b:4;}; int main(){struct S s; s.a = 1; s.b = 2; return s.a * 10 + s.b;}"),
            12
        );
        // Chars and wide fields across unit widths.
        assert_eq!(
            run_main("struct S{char c:2; long long l:40;}; int main(){struct S s; s.c = 1; s.l = 1000000000L; return s.c + (int)(s.l / 1000000000L);}"),
            2
        );
        assert_eq!(
            run_main("struct M{char c:2; long long l:40;}; int main(){struct M m; m.c = 1; m.l = 5; return m.c + (int)m.l + (int)sizeof(struct M);}"),
            1 + 5 + 8
        );
        assert_eq!(run_main("struct W{unsigned long long u:40;}; int main(){struct W w; w.u = 5; return (int)w.u;}"), 5);
        assert_eq!(run_main("struct S{int z:17;}; int main(){struct S s; s.z = -1000; return s.z;}"), -1000);
        // Positional (skipping unnamed), designated, nested, and
        // global initializers.
        assert_eq!(run_main("struct S{int x:3; int y:5;}; int main(){struct S s = {5, 20}; return s.x + s.y;}"), -3 + -12);
        assert_eq!(run_main("struct S{int x:3; int y:5;}; int main(){struct S s = {.y = 9}; return s.x + s.y;}"), 9);
        assert_eq!(
            run_main("struct U{int a:3; int :5; int b:8;}; int main(){struct U u = {7, 200}; return u.a + u.b;}"),
            -1 + -56
        );
        assert_eq!(run_main("struct S{int x:3; int y:5;}; struct S g = {5, 20}; int main(){return g.x + g.y;}"), -3 + -12);
        assert_eq!(
            run_main("struct I{int v;}; struct S{struct I i; int x:3;}; int main(){struct S s; s.i.v = 1; s.x = 2; return s.i.v + s.x;}"),
            3
        );
        // By value across calls, exercising unit classification.
        assert_eq!(
            run_main(
                "struct S{int x:3; int y:5;}; int get(struct S s){return s.x + s.y;}\
                int main(){struct S s; s.x = 1; s.y = 2; return get(s);}"
            ),
            3
        );
        // Compound literals, arrays of bit-field structs, and nested
        // designated initializers compose through the same paths.
        assert_eq!(run_main("struct S{int x:3; int y:5;}; int main(){return ((struct S){1, 2}).x + ((struct S){1, 2}).y;}"), 3);
        assert_eq!(
            run_main("struct S{int x:3; int y:5;}; int main(){struct S a[2] = {{1, 2}, {3, 4}}; return a[0].x + a[0].y + a[1].x + a[1].y;}"),
            10
        );
        assert_eq!(
            run_main("struct I{int v;}; struct S{struct I i; int x:3;}; int main(){struct S s = {.i.v = 7, .x = 2}; return s.i.v + s.x;}"),
            9
        );
    }

    /// C text all the way to MIR verification, without executing:
    /// for shapes the JIT cannot link (long-double arithmetic needs
    /// the `glccrt` helpers) and for declarations that only need to
    /// lower through the verifier.
    fn lower(src: &str) {
        let mir_module = lower_module(src);
        let errors = mir::verify::verify_module(&mir_module);
        assert!(errors.is_empty(), "verify errors in {src}: {errors:?}");
    }

    /// The module global with this exact symbol name.
    fn global_named<'m>(module: &'m mir::Module, name: &str) -> &'m mir::GlobalData {
        module.globals.iter().find(|g| g.name == name).unwrap_or_else(|| panic!("no global named {name}"))
    }

    #[test]
    fn block_scope_function_declarations_resolve_to_definitions() {
        // `float f();` inside a function names the file-scope `f`
        // defined later in the TU: calls go straight to the
        // definition, not to an import of the same name (930603-1).
        let src = "int main(){ float fx(), inita(); return (int)(fx(2.0) + inita()); }\
                   float fx(x) float x; { return x + 0.25; }\
                   float inita(){ return 3.0; }";
        let module = lower_module(src);
        assert!(module.externs.iter().all(|e| e.name != "fx" && e.name != "inita"), "block-scope prototypes became imports");
        assert_eq!(run_main(src), 5);
    }

    #[test]
    fn union_member_initializers_step_through_the_member_type() {
        // The union's storage is the member's: initializing a struct
        // member (or a member below a union designator) steps with the
        // member's type, not the union image (20071029-1, strlen-5).
        assert_eq!(
            run_main(
                "union U { struct { int f1; int f2; } f; char s[16]; long a; };\
                 int main(){ union U u = { { 3, 4 } }; union U v = { .f = { 5, 6 } }; return u.f.f1 + u.f.f2 + v.f.f1 + v.f.f2; }"
            ),
            3 + 4 + 5 + 6
        );
        assert_eq!(
            run_main(
                "struct S { union { struct { int a; } i; long l; } u; int z; };\
                 int main(){ struct S s = { .u.i.a = 7, .z = 2 }; return s.u.i.a + s.z; }"
            ),
            9
        );
    }

    #[test]
    fn address_constants_into_string_literals() {
        // A string literal is an object, so `"foo" + 1` and
        // `&("abc"[2])` are address constants with an addend
        // (pr53084, 921019-1).
        let module = lower_module(
            "static const char *const foo[] = {\"foo\" + 1};\
             static char *x = &(\"abc\"[2]);\
             int main(){ return foo[0][0] == 'o' && *x == 'c' ? 0 : 1; }",
        );
        let (_, reloc) = &global_named(&module, "foo").relocs[0];
        assert!(matches!(reloc, mir::DataReloc::Global { addend: 1, .. }), "{reloc:?}");
        let (_, reloc) = &global_named(&module, "x").relocs[0];
        assert!(matches!(reloc, mir::DataReloc::Global { addend: 2, .. }), "{reloc:?}");
        assert_eq!(
            run_main(
                "static const char *const foo[] = {\"foo\" + 1};\
                 static char *x = &(\"abc\"[2]);\
                 int main(){ return foo[0][0] == 'o' && *x == 'c' ? 0 : 1; }"
            ),
            0
        );
    }

    #[test]
    fn tls_globals_carry_the_tls_flag() {
        let module = lower_module(
            "_Thread_local int x = 1;\
             extern _Thread_local int y;\
             int f(void){ static _Thread_local int z = 2; return z; }\
             int main(){ return x + f() + y; }",
        );
        assert!(global_named(&module, "x").tls, "file-scope TLS");
        assert!(global_named(&module, "y").tls, "extern TLS import");
        assert!(global_named(&module, "y").external);
        let block: Vec<_> = module.globals.iter().filter(|g| g.tls && g.local && g.name.starts_with("z.")).collect();
        assert_eq!(block.len(), 1, "block-scope static TLS");
    }

    #[test]
    fn weak_marks_functions_and_globals() {
        let module = lower_module(
            "__attribute__((weak)) int w = 3;\
             __attribute__((weak)) int f(void){ return w; }\
             int g(void){ return f(); }\
             int main(){ return g() + w; }",
        );
        let w = global_named(&module, "w");
        assert!(w.weak);
        assert!(!w.local, "weak is preemptible, not internal");
        let weak_func = module.funcs.iter().find(|f| f.name == "f").unwrap();
        assert!(weak_func.weak);
        let strong = module.funcs.iter().find(|f| f.name == "main").unwrap();
        assert!(!strong.weak && !strong.local);
        // A weak declaration with no definition stays weak on the
        // import (function and object), matching GCC's `STB_WEAK`
        // undefined symbols.
        let module = lower_module(
            "extern int __attribute__((weak)) v;\
             extern int __attribute__((weak)) f(void);\
             int main(){ return (f ? f() : 7) + (v ? v : 9); }",
        );
        let v = global_named(&module, "v");
        assert!(v.external && v.weak, "weak extern object");
        let f = module.externs.iter().find(|e| e.name == "f").expect("f import");
        assert!(f.weak, "weak extern function");
    }

    #[test]
    fn definitions_carry_natural_alignment() {
        let module = lower_module(
            "double d = 1.5;\
             char buf[3];\
             struct S { char c; long l; } s;\
             char page[16] __attribute__((aligned(4096)));\
             static char slot[8] __attribute__((aligned(64)));\
             long tight __attribute__((aligned(1)));\
             int main(){ return (int)d + buf[0] + s.c + page[0] + slot[0] + (int)tight; }",
        );
        assert_eq!(global_named(&module, "d").align, Some(8));
        assert_eq!(global_named(&module, "buf").align, Some(1));
        assert_eq!(global_named(&module, "s").align, Some(fe::target::Target::native().long_align));
        // `aligned` sets the alignment, raising or lowering it (GCC).
        assert_eq!(global_named(&module, "page").align, Some(4096));
        assert_eq!(global_named(&module, "slot").align, Some(64));
        assert_eq!(global_named(&module, "tight").align, Some(1));
    }

    #[test]
    fn constructors_and_destructors_build_hook_arrays() {
        let module = lower_module(
            "int a; int b;\
             void c2(void) __attribute__((constructor(200))) { a = 2; }\
             void c3(void) __attribute__((constructor)) { a = 3; }\
             void c1(void) __attribute__((constructor(100))) { a = 1; }\
             void d1(void) __attribute__((destructor)) { b = 1; }\
             void plain(void){}\
             int main(){ return a + b; }",
        );
        let ctor = global_named(&module, "__glcc_init_array");
        assert!(ctor.local);
        assert_eq!(ctor.section.as_deref(), Some(".init_array"));
        assert_eq!(ctor.align, Some(8));
        assert_eq!(ctor.relocs.iter().map(|(off, _)| *off).collect::<Vec<_>>(), [0, 8, 16]);
        let names: Vec<&str> = ctor
            .relocs
            .iter()
            .map(|(_, r)| match r {
                mir::DataReloc::Func { target } => module.funcs[*target].name.as_str(),
                other => panic!("non-function hook reloc {other:?}"),
            })
            .collect();
        // Priority order first (bare constructors are 65535), then
        // declaration order; `plain` is never a hook.
        assert_eq!(names, ["c1", "c2", "c3"]);
        let dtor = global_named(&module, "__glcc_fini_array");
        assert_eq!(dtor.section.as_deref(), Some(".fini_array"));
        assert_eq!(dtor.relocs.len(), 1);
    }

    #[test]
    fn undeclared_or_absent_hooks_emit_no_arrays() {
        // A constructor with no body never runs: no array entry, and
        // with no defined hooks at all no array object exists.
        let module = lower_module(
            "void never(void) __attribute__((constructor));\
             int main(){ return 0; }",
        );
        assert!(!module.globals.iter().any(|g| g.name.starts_with("__glcc_")));
    }

    #[test]
    fn complex_literals() {
        assert_eq!(run_main("int main(){double _Complex x = 1.0 + 2.0i; return (int)(__real__ x + __imag__ x);}"), 3);
        assert_eq!(run_main("int main(){double _Complex x = 2.0i; return (int)(__real__ x + __imag__ x);}"), 2);
        lower("int main(){double _Complex x = 2.0i; return (int)(__real__ x + __imag__ x);}");
        assert_eq!(run_main("int main(){double _Complex x = 2.0i; return (int)(__real__ x + __imag__ x);}"), 2);
        assert_eq!(run_main("int main(){float _Complex x = 1.0fi; return (int)(__real__ x + __imag__ x);}"), 1);
        assert_eq!(run_main("int main(){__complex__ int x = 1 + 2i; return __real__ x + __imag__ x;}"), 3);
        assert_eq!(run_main("int main(){__complex__ double x = 1j; return (int)(__real__ x + __imag__ x);}"), 1);
    }

    #[test]
    fn complex_arithmetic() {
        // (1+2i)+(3+4i) = 4+6i.
        assert_eq!(
            run_main("int main(){double _Complex a = 1.0 + 2.0i; double _Complex b = 3.0 + 4.0i; double _Complex c = a + b; return (int)__real__ c * 100 + (int)__imag__ c;}"),
            406
        );
        // (1+2i)-(3+4i) = -2-2i.
        assert_eq!(
            run_main("int main(){double _Complex a = 1.0 + 2.0i; double _Complex b = 3.0 + 4.0i; double _Complex c = a - b; return (int)__real__ c * 100 + (int)__imag__ c;}"),
            -202
        );
        // (1+2i)(3+4i) = -5+10i.
        assert_eq!(
            run_main("int main(){double _Complex a = 1.0 + 2.0i; double _Complex b = 3.0 + 4.0i; double _Complex c = a * b; return (int)__real__ c * 100 + (int)__imag__ c;}"),
            -490
        );
        // (4+0i)/(2+0i) = 2+0i, exact in float.
        assert_eq!(
            run_main("int main(){double _Complex a = 4.0; double _Complex b = 2.0; double _Complex c = a / b; return (int)__real__ c * 100 + (int)__imag__ c;}"),
            200
        );
        // Mixed complex/real through the usual conversions.
        assert_eq!(
            run_main(
                "float __complex__ p(float __complex__ a, float __complex__ b){return a + b;} int main(){float __complex__ x = 1.0f + 2.0fi; return (int)__real__ p(x, 3.0f) * 10 + (int)__imag__ p(x, 3.0f);}"
            ),
            42
        );
        // Integer complex wraps and truncates like scalars.
        assert_eq!(
            run_main("int main(){__complex__ int a = 1 + 2i; __complex__ int b = 3 + 4i; __complex__ int c = a * b; return __real__ c * 100 + __imag__ c;}"),
            -490
        );
        assert_eq!(run_main("int main(){__complex__ int a = 7; __complex__ int b = 2; return __real__(a / b);}"), 3);
        assert_eq!(run_main("int main(){__complex__ int a = -7; __complex__ int b = 2; return __real__(a / b);}"), -3);
        assert_eq!(
            run_main("int main(){__complex__ unsigned a = 7; __complex__ unsigned b = 2; return (int)__real__(a / b);}"),
            3
        );
        // Unary minus, identity, and conjugate.
        assert_eq!(
            run_main("int main(){double _Complex a = 1.0 + 2.0i; double _Complex c = -a + +a + ~a; return (int)__real__ c * 100 + (int)__imag__ c;}"),
            98
        );
        // Compound assignment (the checker pre-combines the op).
        assert_eq!(
            run_main("int main(){double _Complex a = 1.0 + 2.0i; a += 1.0; a -= 1.0i; a *= 2.0; a /= 2.0; return (int)__real__ a * 10 + (int)__imag__ a;}"),
            21
        );
    }

    #[test]
    fn complex_comparisons_and_conditions() {
        assert_eq!(
            run_main(
                "int f(double _Complex a, double _Complex b){return (a == b) + (a != b) * 10;} int main(){double _Complex a = 1.0 + 2.0i; return f(a, a) * 100 + f(a, 1.0 + 3.0i);}"
            ),
            110
        );
        // Ordered equality: +0.0 equals -0.0 in both parts.
        assert_eq!(run_main("int main(){if ((__complex__ double)0.0 != (__complex__ double)(-0.0)) return 1; return 0;}"), 0);
        // A complex condition holds when either part is nonzero.
        assert_eq!(
            run_main("int f(double _Complex a){if (a) return 1; return 0;} int main(){return f(0.0) * 100 + f(1.0i) * 10 + f(1.0);}"),
            11
        );
        assert_eq!(
            run_main("int f(double _Complex a){return !a;} int main(){return f(0.0) * 10 + f(1.0i) + f(1.0 + 1.0i) * 100;}"),
            10
        );
        assert_eq!(
            run_main("int main(){double _Complex z = 0.0; double _Complex o = 1.0; return (z && o) + (o && z) * 10 + (o && o) * 100 + (z || o);;}"),
            101
        );
        // Integer components compare exactly.
        assert_eq!(
            run_main("int f(__complex__ int a, __complex__ int b){return (a == b) + (a != b);} int main(){__complex__ int a = 1 + 2i; return f(a, a) * 10 + f(a, 1 + 3i);}"),
            11
        );
    }

    #[test]
    fn complex_conversions() {
        // Real to complex: value as the real part, zero imaginary.
        assert_eq!(
            run_main("int main(){double x = 2.5; double _Complex c = x; return (int)__real__ c * 10 + (int)__imag__ c;}"),
            20
        );
        // Complex to real: the real part, imaginary discarded.
        assert_eq!(run_main("int main(){double _Complex c = 2.5 + 3.5i; double r = c; int n = c; return (int)r * 10 + n;}"), 22);
        // Complex to `_Bool`: nonzero when either part is.
        assert_eq!(
            run_main("int main(){double _Complex z = 0.0; double _Complex i = 1.0i; _Bool a = z; _Bool b = i; return a * 10 + b;}"),
            1
        );
        // Complex to complex across precisions.
        assert_eq!(
            run_main("int main(){float _Complex f = 1.5f + 2.5fi; double _Complex d = f; return (int)__real__ d + (int)__imag__ d;}"),
            3
        );
        assert_eq!(
            run_main("int main(){double _Complex d = 1.5 + 2.5i; float _Complex f = d; return (int)__real__ f + (int)__imag__ f;}"),
            3
        );
        assert_eq!(
            run_main("int main(){double _Complex d = 1.0 + 2.0i; __complex__ int n = d; return __real__ n * 100 + __imag__ n;}"),
            102
        );
        assert_eq!(
            run_main("int main(){int n = 7; double _Complex c = (double _Complex)n; return (int)__real__ c + (int)__imag__ c;}"),
            7
        );
    }

    #[test]
    fn complex_real_imag_lvalues() {
        assert_eq!(
            run_main("double f(double _Complex x){return __real__ x + __imag__ x;} int main(){double _Complex a = 0.0; __real__ a = 1.5; __imag__ a = 2.5; return (int)f(a);}"),
            4
        );
        assert_eq!(
            run_main("void g(double _Complex *p){__real__ *p = 1.0; __imag__ *p = 2.0;} int main(){double _Complex a = 0.0; g(&a); return (int)(__real__ a + __imag__ a);}"),
            3
        );
        assert_eq!(
            run_main("double *f(double _Complex *p){return &__real__ *p;} int main(){double _Complex a = 5.0 + 6.0i; *f(&a) = 7.0; return (int)__real__ a + (int)__imag__ a;}"),
            13
        );
        assert_eq!(
            run_main("int main(){double _Complex a[2]; __real__ a[0] = 1.0; __imag__ a[1] = 2.0; return (int)(__real__ a[0] + __imag__ a[1]);}"),
            3
        );
        // Parts step like real lvalues.
        assert_eq!(
            run_main("int main(){double _Complex a = 1.0 + 2.0i; __real__ a += 10.0; ++__imag__ a; return (int)__real__ a * 100 + (int)__imag__ a;}"),
            1103
        );
        // Parts of rvalues (calls, arithmetic) extract off the pair.
        assert_eq!(
            run_main("double _Complex f(void){return 1.0 + 2.0i;} int main(){return (int)__real__ f() * 10 + (int)__imag__ f();}"),
            12
        );
        assert_eq!(
            run_main("int main(){double _Complex a = 1.0 + 2.0i; double _Complex b = 3.0 + 4.0i; return (int)__real__(a + b) * 10 + (int)__imag__(a + b);}"),
            46
        );
        assert_eq!(run_main("int main(){double _Complex a = 1.0 + 2.0i; return (int)__real__ +a + (int)__imag__ -a;}"), -1);
    }

    #[test]
    fn complex_calls_ternary_and_aggregates() {
        assert_eq!(
            run_main("double _Complex id(double _Complex x){return x;} int main(){double _Complex c = id(1.0 + 2.0i) + id(3.0); return (int)__real__ c * 10 + (int)__imag__ c;}"),
            42
        );
        assert_eq!(
            run_main("int main(){int c = 1; double _Complex a = 5.0 + 6.0i; double _Complex r = c ? a : 2.0 * a; return (int)__real__ r + (int)__imag__ r;}"),
            11
        );
        assert_eq!(
            run_main(
                "double _Complex g(void); double _Complex f(void){double _Complex x = g(); return x;} double _Complex g(void){return 1.0 + 2.0i;} int main(){return (int)__real__ f() + (int)__imag__ f();}"
            ),
            3
        );
        // Complex inside structs and arrays, by value and by address.
        assert_eq!(
            run_main("struct S{double _Complex v; int k;}; int main(){struct S s; s.v = 1.0i; s.k = 3; return (int)__imag__ s.v + s.k;}"),
            4
        );
        assert_eq!(
            run_main(
                "struct S{double _Complex v; int k;}; struct S id(struct S s){return s;} int main(){struct S s; s.v = 1.0 + 2.0i; s.k = 5; struct S t = id(s); return (int)__real__ t.v + (int)__imag__ t.v + t.k;}"
            ),
            8
        );
        assert_eq!(
            run_main("int main(){double _Complex a[2]; a[0] = 1.0; a[1] = 2.0i; return (int)(__real__ a[0] + __imag__ a[1]);}"),
            3
        );
    }

    #[test]
    fn complex_builtins() {
        assert_eq!(
            run_main("int main(){double _Complex x = 1.0 + 2.0i; double _Complex y = __builtin_conj(x); return (int)__real__ y * 10 + (int)__imag__ y;}"),
            8
        );
        assert_eq!(
            run_main("int main(){float _Complex x = 1.0f + 2.0fi; float _Complex y = __builtin_conjf(x); return (int)__real__ y * 10 + (int)__imag__ y;}"),
            8
        );
        // 3-4-5 triangle: exact under the naive magnitude.
        assert_eq!(
            run_main("int main(){double _Complex x = 3.0 + 4.0i; return (int)(__builtin_creal(x) + __builtin_cimag(x) + __builtin_cabs(x));}"),
            12
        );
        assert_eq!(
            run_main("int main(){float _Complex x = 3.0f + 4.0fi; return (int)(__builtin_crealf(x) + __builtin_cimagf(x) + __builtin_cabsf(x));}"),
            12
        );
        // Glibc spellings with exact signatures lower inline.
        assert_eq!(
            run_main(
                "double creal(double _Complex); double cimag(double _Complex); double cabs(double _Complex); int main(){double _Complex x = 3.0 + 4.0i; return (int)(creal(x) + cimag(x) + cabs(x));}"
            ),
            12
        );
        assert_eq!(
            run_main("double _Complex conj(double _Complex); int main(){double _Complex x = 1.0 + 2.0i; return (int)__real__ conj(x) * 10 + (int)__imag__ conj(x);}"),
            8
        );
        // `f`-suffixed glibc spellings, and `~` over float complex.
        assert_eq!(
            run_main(
                "float crealf(float _Complex); float cimagf(float _Complex); float cabsf(float _Complex); int main(){float _Complex x = 3.0f + 4.0fi; return (int)(crealf(x) + cimagf(x) + cabsf(x));}"
            ),
            12
        );
        assert_eq!(
            run_main(
                "float _Complex conjf(float _Complex); int main(){float _Complex x = 1.0f + 2.0fi; float _Complex y = conjf(x) + ~x; return (int)__real__ y * 10 + (int)__imag__ y;}"
            ),
            16
        );
        // Same-named user functions with other signatures still call out.
        lower("int conj(int x); int f(void){return conj(1);}");
        lower("float creal(int x); float f(void){return creal(1);}");
    }

    #[test]
    fn complex_globals() {
        assert_eq!(run_main("double _Complex g = 1.0 + 2.0i; int main(){return (int)__real__ g + (int)__imag__ g;}"), 3);
        assert_eq!(run_main("float _Complex g; int main(){g = 3.0fi; return (int)__imag__ g + (int)__real__ g;}"), 3);
        assert_eq!(run_main("__complex__ int g = 4 + 5i; int main(){return __real__ g + __imag__ g;}"), 9);
        // Remainder-zeroed globals read back their parts.
        assert_eq!(run_main("double _Complex g = 5.0i; int main(){return (int)__real__ g * 10 + (int)__imag__ g;}"), 5);
        // Block-scope statics fold once and persist across calls.
        assert_eq!(
            run_main("int f(){static double _Complex c = 1.0 + 2.0i; __real__ c += 1.0; return (int)__real__ c;} int main(){f(); return f();}"),
            3
        );
    }

    #[test]
    fn flexible_array_member_globals() {
        // An initialized flexible array member grows the object past
        // `sizeof` (GCC extension); the declared type keeps its size.
        assert_eq!(
            run_main("struct F { int i; int j[]; }; struct F x = { 1, { 2, 3 } }; int main(){return x.i + x.j[0] + x.j[1];}"),
            6
        );
        assert_eq!(run_main("struct F { int i; int j[]; }; struct F x = { 1, { 2, 3 } }; int main(){return sizeof x;}"), 4);
        // A string literal fills a flexible char member, and a
        // designator into one sizes the object by the element it
        // lands on.
        assert_eq!(
            run_main("struct C { char c; char p[]; }; struct C s = { 'o', \"wx\" }; int main(){return s.p[0] + s.p[1] - 'w' - 'x';}"),
            0
        );
        assert_eq!(run_main("struct F { int i; int j[]; }; struct F x = { .j[1] = 9 }; int main(){return x.j[0] + x.j[1];}"), 9);
        // An empty flexible initializer adds nothing.
        assert_eq!(run_main("struct F { int i; int j[]; }; struct F x = { .i = 4, .j = {} }; int main(){return x.i;}"), 4);
    }

    #[test]
    fn complex_long_double_lowers() {
        // Long-double arithmetic needs the `glccrt` helpers the JIT
        // cannot link, so these only lower through the verifier.
        lower("long double _Complex f(void){long double _Complex x = 1.0iL; return x;}");
        lower("long double _Complex f(long double _Complex a, long double _Complex b){return a + b + a - b + a * b + a / b;}");
        lower("int f(long double _Complex a, long double _Complex b){return (a == b) + (a != b);}");
        lower("long double _Complex f(double _Complex x){return x;}");
        lower("double _Complex f(long double _Complex x){return x;}");
        lower("long double f(long double _Complex x){return __builtin_creall(x) + __builtin_cimagl(x);}");
        lower("long double _Complex f(long double _Complex x){return __builtin_conjl(x);}");
        lower("long double f(long double _Complex x){return __builtin_cabsl(x);}");
        lower("int f(long double _Complex a){if (a) return 1; return 0;}");
        lower("int f(long double _Complex a){return !a;}");
        lower("long double _Complex f(long double _Complex a){return -a + +a + ~a;}");
        lower("long double _Complex g = 1.0L + 2.0iL; long double _Complex f(void){return g;}");
        lower("__complex__ long f(void){__complex__ long x = 3 + 4i; return x;}");
        lower("long double _Complex f(int c, long double _Complex a){return c ? a : 2.0L * a;}");
    }

    #[test]
    fn locals_and_global() {
        assert_eq!(run_main("int main(){int x = 6; int y = 7; return x+y;}"), 13);
        assert_eq!(run_main("int g = 41; int main(){return g+1;}"), 42);
        assert_eq!(run_main("int g; int main(){return g;}"), 0);
    }

    #[test]
    fn block_static_locals() {
        // Zero-init without `=`, and persistence across calls.
        assert_eq!(run_main("int f(){static int c = 0; c += 1; return c;} int main(){f(); return f();}"), 2);
        assert_eq!(run_main("int main(){static int z; return z;}"), 0);
        assert_eq!(run_main("int main(){static int n = 5; n += 1; return n;}"), 6);
        // Aggregates fold once; unsized arrays complete from the init.
        assert_eq!(run_main("int f(){static int a[3] = {1,2,3}; return a[1];} int main(){return f();}"), 2);
        assert_eq!(run_main("struct P{int x; int y;}; int main(){static struct P p = {1,2}; return p.x + p.y;}"), 3);
        assert_eq!(run_main("int main(){static char s[] = \"hi\"; return s[0] + s[1];}"), 209);
        // Addressable storage shared across calls.
        assert_eq!(run_main("int *f(){static int x = 7; return &x;} int main(){*f() = 10; return *f();}"), 10);
        // Same spelling in different scopes is separate storage.
        assert_eq!(
            run_main(
                "int f(){static int x = 1; x += 10; return x;} int g(){static int x = 2; x += 100; return x;}\
                int main(){int a = f(); int b = g(); return a + b;}"
            ),
            11 + 102
        );
    }

    #[test]
    fn padded_struct_layout() {
        // Trailing padding counts in the size: member access, array
        // stride, assignment, and globals all agree with C layout.
        assert_eq!(run_main("struct P{char c; int x;}; int main(){struct P p; p.c = 1; p.x = 2; return p.c + p.x;}"), 3);
        assert_eq!(
            run_main(
                "struct P{char c; int x;}; int main(){struct P a[3]; a[0].x = 1; a[1].x = 10; a[2].x = 100;\
                a[0].c = 0; a[1].c = 0; a[2].c = 0; return a[0].x + a[1].x + a[2].x;}"
            ),
            111
        );
        assert_eq!(
            run_main("struct P{char c; int x;}; int main(){struct P a = {7, 8}; struct P b; b = a; return b.c + b.x;}"),
            15
        );
        assert_eq!(run_main("struct P{char c; int x;}; struct P g = {3, 4}; int main(){return g.c + g.x;}"), 7);
        assert_eq!(
            run_main(
                "struct R{char t[3]; int v;}; struct R rs[2] = {{\"ab\", 1}, {\"cd\", 2}};\
                int main(){return rs[0].v * 10 + rs[1].v + rs[1].t[0];}"
            ),
            10 + 2 + 99
        );
        assert_eq!(run_main("struct P{char c; int x;}; int main(){return (int)sizeof(struct P);}"), 8);
    }

    #[test]
    fn subscript_rows_decay() {
        // `a[i]` of a 2D array is the row's address, not a load of
        // its first element — even though the checker pre-decays the
        // base (and the row) to pointers.
        assert_eq!(
            run_main(
                "int sum(const int *r){return r[0]+r[1]+r[2];}\
                int main(){int m[2][3] = {{1,2,3},{4,5,6}}; return sum(m[0]) + sum(m[1]);}"
            ),
            21
        );
        assert_eq!(
            run_main(
                "const char *pick(const char **t, int i){return t[i];}\
                int main(){const char *t[2] = {\"ab\", \"c\"}; return (int)(pick(t, 1)[0]);}"
            ),
            99
        );
        assert_eq!(
            run_main(
                "int first(int *p){return p[0];}\
                int main(){int a[4] = {7, 8, 9, 10}; return first(a + 2);}"
            ),
            9
        );
    }

    #[test]
    fn unary_operands_evaluate_once() {
        // `&a[j++]` must step once with the pre-increment index: the
        // address operator used to evaluate its operand twice (once
        // as a value, once for the address), double-stepping.
        assert_eq!(
            run_main(
                "int main(){int a[4] = {10, 20, 30, 40}; int j = 0;\
                int *p = &a[j++]; return j * 100 + *p;}"
            ),
            110
        );
        assert_eq!(
            run_main(
                "int main(){int a[4] = {10, 20, 30, 40}; int j = 0;\
                int x = (*&a[j++]); return j * 100 + x;}"
            ),
            110
        );
        assert_eq!(
            run_main(
                "int main(){int a[4] = {10, 20, 30, 40}; int j = 0;\
                int x = *(a + (j++)); return j * 100 + x;}"
            ),
            110
        );
    }

    #[test]
    fn null_pointer_globals() {
        // Every null spelling folds to zero bytes: plain `0`,
        // `(void *)0`, and inside lists, plus `_Bool` globals which
        // share the width computation.
        assert_eq!(run_main("static int *p = 0; int main(){return p != 0;}"), 0);
        assert_eq!(run_main("static void *q = (void *)0; int main(){return q != 0;}"), 0);
        assert_eq!(
            run_main(
                "static const char *cmd[3] = {\"a\", ((void *)0), \"b\"};\
                int main(){return (cmd[1] != 0) + (cmd[0][0] != 'a');}"
            ),
            0
        );
        assert_eq!(run_main("static _Bool t = 1; static _Bool f; int main(){return t + f * 2;}"), 1);
    }

    #[test]
    fn atomic_load_store_rmw() {
        // Relaxed atomics behave like their plain counterparts here;
        // the narrowing and compare-exchange reporting matter. Orders
        // ride as raw integers (the harness lacks the macros).
        assert_eq!(
            run_main(
                "int main(){int x = 0; __atomic_store_n(&x, 41, 0);\
                return __atomic_load_n(&x, 0);}"
            ),
            41
        );
        assert_eq!(
            run_main(
                "int main(){int x = 7; int e = __atomic_exchange_n(&x, 3, 0);\
                return e * 100 + x;}"
            ),
            700 + 3
        );
        assert_eq!(
            run_main(
                "int main(){int x = 7; int old = __atomic_fetch_add(&x, 10, 0);\
                return old * 100 + x;}"
            ),
            700 + 17
        );
        assert_eq!(
            run_main(
                "int main(){int x = 17; int e = 17;\
                int ok = __atomic_compare_exchange_n(&x, &e, 99, 0, 0, 0);\
                return ok * 10000 + x * 100 + e;}"
            ),
            10000 + 9900 + 17
        );
        assert_eq!(
            run_main(
                "int main(){int x = 17; int e = 18;\
                int ok = __atomic_compare_exchange_n(&x, &e, 99, 0, 0, 0);\
                return ok * 10000 + x * 100 + e;}"
            ),
            1700 + 17
        );
    }

    #[test]
    fn atomic_c11_builtins() {
        // The clang C11 spellings `<stdatomic.h>` expands to: the same
        // behavior as the GCC forms, minus the `_n` suffix and the
        // explicit weak flag.
        assert_eq!(
            run_main(
                "int main(){int x = 0; __c11_atomic_store(&x, 41, 5);\
                return __c11_atomic_load(&x, 5);}"
            ),
            41
        );
        assert_eq!(run_main("int main(){int x; __c11_atomic_init(&x, 7); return x;}"), 7);
        assert_eq!(
            run_main(
                "int main(){int x = 7; int e = __c11_atomic_exchange(&x, 3, 5);\
                return e * 100 + x;}"
            ),
            700 + 3
        );
        assert_eq!(
            run_main(
                "int main(){int x = 7; int old = __c11_atomic_fetch_add(&x, 10, 5);\
                return old * 100 + x;}"
            ),
            700 + 17
        );
        assert_eq!(
            run_main(
                "int main(){int x = 17; int e = 17;\
                int ok = __c11_atomic_compare_exchange_strong(&x, &e, 99, 5, 5);\
                return ok * 10000 + x * 100 + e;}"
            ),
            10000 + 9900 + 17
        );
        assert_eq!(
            run_main(
                "int main(){int x = 17; int e = 18;\
                int ok = __c11_atomic_compare_exchange_weak(&x, &e, 99, 5, 5);\
                return ok * 10000 + x * 100 + e;}"
            ),
            1700 + 17
        );
        assert_eq!(
            run_main(
                "int main(){int x = 5;\
                int a = __c11_atomic_fetch_max(&x, 9, 5); int b = __c11_atomic_fetch_min(&x, 2, 5);\
                return a * 100 + b;}"
            ),
            500 + 9
        );
        assert_eq!(
            run_main(
                "int main(){int x = 2;\
                int c = __c11_atomic_fetch_nand(&x, 1, 5); return c * 100 + (x == -1);}"
            ),
            200 + 1
        );
        assert_eq!(
            run_main(
                "int main(){int x = 0; __c11_atomic_thread_fence(5); __c11_atomic_signal_fence(5);\
                return __c11_atomic_is_lock_free(sizeof(int)) + (x == 0);}"
            ),
            2
        );
    }

    #[test]
    fn atomic_min_max_signedness() {
        // `min`/`max` compare by the target's signedness (the MIR
        // `Smin`/`Umax` split), and the `min_fetch`/`max_fetch` forms
        // compute the new value with a select.
        assert_eq!(
            run_main(
                "int main(){int x = -5;\
                int a = __atomic_fetch_max(&x, -9, 5); int b = __atomic_fetch_min(&x, -9, 5);\
                return (a == -5) + (x == -9) * 2;}"
            ),
            1 + 2
        );
        assert_eq!(
            run_main(
                "int main(){unsigned u = 3u;\
                unsigned c = __atomic_max_fetch(&u, 10u, 5);\
                return (c == 10) + (u == 10) * 2;}"
            ),
            1 + 2
        );
        assert_eq!(
            run_main(
                "int main(){int x = -5;\
                int r = __atomic_min_fetch(&x, -9, 5);\
                return (r == -9) + (x == -9) * 2;}"
            ),
            1 + 2
        );
    }

    #[test]
    fn struct_pointer_arithmetic() {
        // Steps over struct pointees scale by the frontend element
        // size (the MIR pointee is erased): `p++`, `p + i`, `p - i`,
        // and `p[i]` all agree.
        assert_eq!(
            run_main(
                "struct P{char c; int x;}; int main(){struct P a[3]; struct P *p = a;\
                p->x = 1; p++; p->x = 10; p += 1; p->x = 100; p -= 2;\
                return a[0].x + a[1].x + a[2].x + (p == a); }"
            ),
            111 + 1
        );
        assert_eq!(
            run_main(
                "struct P{char c; int x;}; int get(struct P *p, int i){return (p + i)->x;}\
                int main(){struct P a[2]; a[0].x = 5; a[1].x = 6; return get(a, 0) + get(a, 1);}"
            ),
            11
        );
        assert_eq!(
            run_main(
                "struct P{char c; int x;}; int main(){struct P a[3]; struct P *p = a + 2;\
                return (p - a) * 100 + (p - 2 == a);}"
            ),
            200 + 1
        );
    }

    #[test]
    fn va_list_forwarding() {
        // `va_arg` through a `va_list` parameter reads the caller's
        // object: parameters alias it rather than spilling the pointer
        // into a slot (which reads as cursor garbage).
        assert_eq!(
            run_main(
                "int fwd(int n, __builtin_va_list ap){int s = 0;\
                for (int i = 0; i < n; i++) s += __builtin_va_arg(ap, int); return s;}\
                int sum(int n, ...){__builtin_va_list ap; __builtin_va_start(ap, n);\
                int s = fwd(n, ap); __builtin_va_end(ap); return s;}\
                int main(){return sum(4, 10, 20, 30, 40);}"
            ),
            100
        );
    }

    #[test]
    fn va_arg_reads_overflow_aggregates_whole() {
        // A 9-byte struct spans two eightbytes, so the third struct
        // vararg (after two register pairs) cannot fit its registers
        // and the SysV ABI sends the whole argument to the overflow
        // area. Reading piecewise would take the first eightbyte from
        // the save area and the second from the stack.
        assert_eq!(
            run_main(
                "struct S{char x[9];}; int f(int n, ...){__builtin_va_list ap;\
                __builtin_va_start(ap, n); int s = 0;\
                for (int i = 0; i < n; i++){struct S t = __builtin_va_arg(ap, struct S);\
                s = s * 3 + t.x[0] + t.x[8];} __builtin_va_end(ap); return s;}\
                int main(){struct S a = {\"ABCDEFGHI\"}; return f(6, a, a, a, a, a, a);}"
            ),
            50232
        );
        // Same rule in the vector file: five three-float structs need
        // ten XMM slots, so the fifth crosses wholly on the stack.
        assert_eq!(
            run_main(
                "struct F{float a, b, c;}; int g(int n, ...){__builtin_va_list ap;\
                __builtin_va_start(ap, n); int s = 0;\
                for (int i = 0; i < n; i++){struct F t = __builtin_va_arg(ap, struct F);\
                s = s * 2 + (int)t.a + (int)t.c;} __builtin_va_end(ap); return s;}\
                int main(){struct F f = {13.1f, 13.2f, 13.3f}; return g(5, f, f, f, f, f);}"
            ),
            806
        );
    }

    #[test]
    fn scalar_builtins_lower_inline() {
        // Single-instruction builtins have no library symbol: they
        // lower inline instead of calling out.
        assert_eq!(run_main("int main(){return __builtin_clzll(0x1000000000000000ULL);}"), 3);
        assert_eq!(run_main("int main(){return __builtin_ctz(8) + __builtin_popcount(7);}"), 6);
        assert_eq!(run_main("int main(){return (int)__builtin_ceil(1.5) * 10 + (int)__builtin_floor(1.5);}"), 21);
        assert_eq!(run_main("int main(){return (__builtin_bswap32(0x12345678U) == 0x78563412U);}"), 1);
    }

    #[test]
    fn type_generic_bit_counts_count_at_operand_width() {
        // `__builtin_clzg`/`__builtin_ctzg`: the operand keeps its own
        // width (no promotion), the count converts to `int`, and a zero
        // operand yields the fallback when one is given.
        assert_eq!(run_main("int main(){return __builtin_clzg((unsigned char)0x10);}"), 3);
        assert_eq!(run_main("int main(){return __builtin_clzg(0x10ULL);}"), 59);
        assert_eq!(run_main("int main(){return __builtin_ctzg((unsigned short)0x100);}"), 8);
        assert_eq!(run_main("int main(){return __builtin_clzg(1U, 42);}"), 31);
        assert_eq!(run_main("int main(){return __builtin_clzg(0U, 42);}"), 42);
        assert_eq!(run_main("int main(){return __builtin_ctzg(0x80U, 42);}"), 7);
        assert_eq!(run_main("int main(){return __builtin_ctzg(0U, 42);}"), 42);
        // A `_Bool` operand counts one bit: its nonzero value has no
        // leading or trailing zeros, and zero counts the fallback (the
        // byte-wide count would say eight).
        assert_eq!(run_main("int main(){return __builtin_clzg((_Bool)1);}"), 0);
        assert_eq!(run_main("int main(){return __builtin_ctzg((_Bool)0, 42);}"), 42);
        // 128-bit operands keep their width too.
        assert_eq!(run_main("int main(){return __builtin_clzg((unsigned __int128)1);}"), 127);
    }

    #[test]
    fn fp_classifications_lower_to_comparisons() {
        // `isinf` is 1/0 like GCC; `isinf_sign` distinguishes the sign
        // (what `<math.h>`'s `isinf` macro expands to).
        assert_eq!(
            run_main("int main(){double p=1.0/0.0, n=-1.0/0.0; return __builtin_isinf(p)*100 + __builtin_isinf(n)*10 + __builtin_isinf(2.0);}"),
            110
        );
        assert_eq!(
            run_main("int main(){double p=1.0/0.0, n=-1.0/0.0; return __builtin_isinf_sign(p)*100 + __builtin_isinf_sign(n)*10 + __builtin_isinf_sign(2.0);}"),
            90
        );
        // `isfinite` rejects NaN and the infinities; `isnormal` also
        // rejects zero and subnormals.
        assert_eq!(
            run_main("int main(){double p=1.0/0.0; return __builtin_isfinite(1.5) + __builtin_isfinite(0.0)*10 + __builtin_isfinite(p)*100 + __builtin_isfinite(p-p)*1000;}"),
            11
        );
        assert_eq!(
            run_main("int main(){double p=1.0/0.0; return __builtin_isnormal(1.5) + __builtin_isnormal(5e-324)*10 + __builtin_isnormal(0.0)*100 + __builtin_isnormal(p)*1000;}"),
            1
        );
        // `fpclassify` picks the code for the category; the codes here
        // are glibc's `FP_NAN`...`FP_ZERO` (0, 1, 4, 3, 2).
        assert_eq!(
            run_main(
                "int main(){double p=1.0/0.0; return __builtin_fpclassify(0,1,4,3,2,0.0/0.0) + __builtin_fpclassify(0,1,4,3,2,p)*10 \
                 + __builtin_fpclassify(0,1,4,3,2,1.5)*100 + __builtin_fpclassify(0,1,4,3,2,5e-324)*1000 + __builtin_fpclassify(0,1,4,3,2,0.0)*10000;}"
            ),
            23410
        );
        // The float forms use the same tests at 32-bit width.
        assert_eq!(
            run_main(
                "int main(){float p=1.0f/0.0f; return __builtin_isinf_sign(p)*10 + __builtin_isfinite(1.5f) \
                 + __builtin_isnormal(0.0f)*100 + __builtin_fpclassify(0,1,4,3,2,5e-45f)*1000;}"
            ),
            3011
        );
        // 80-bit long doubles classify through the x87 comparisons.
        lower(
            "int f(long double x){ return __builtin_isinf(x) + __builtin_isinf_sign(x) + __builtin_isfinite(x) \
             + __builtin_isnormal(x) + __builtin_fpclassify(0,1,4,3,2,x); }",
        );
    }

    #[test]
    fn ternary_arms_convert_to_result() {
        // `cond ? ptr : 0` must store a full null pointer, not a
        // 4-byte zero over stack garbage.
        assert_eq!(run_main("int x = 5; int main(){int *p = x ? &x : 0; return *p;}"), 5);
        assert_eq!(run_main("int x = 0; int main(){int *p = x ? &x : 0; return p == 0;}"), 1);
        assert_eq!(run_main("int main(){int a = 1, b = 2; int c = 0; int r = c ? a : b; return r;}"), 2);
        // A null-constant `(void *)0` arm keeps the other arm's
        // pointer type, so member access survives (tcctest.c shape).
        assert_eq!(
            run_main("struct S{int i;}; int f(int i, struct S *s){return (i == 0 ? (void *)0 : s)->i;} int main(){struct S s; s.i = 7; return f(1, &s);}"),
            7
        );
    }

    #[test]
    fn inline_flag_reaches_mir() {
        // The C `inline` hint rides the binding into MIR so the
        // `inline` pass can spend a larger budget on it.
        for (src, name, expect_inline) in [
            ("inline int f(void){return 1;} int main(){return f();}", "f", true),
            ("int f(void){return 1;} int main(){return f();}", "f", false),
            ("static __inline__ int f(void){return 1;} int main(){return f();}", "f", true),
        ] {
            let mut interner = fe::lexer::Interner::new();
            let mut sema = fe::sema::Sema::new();
            let tokens = fe::lexer::Lexer::new(src, &mut interner).lex_all();
            let mut parser = fe::parse::Parser::new(src, tokens, &mut sema, &mut interner);
            let tu = parser.parse_translation_unit();
            drop(parser);
            let module = lower_translation_unit(&sema, &interner, &tu);
            let func = module.funcs.iter().find(|f| f.name == name).expect("function lowered");
            assert_eq!(func.inline, expect_inline, "{src}");
            assert!(mir::verify::verify_module(&module).is_empty(), "{src}");
        }
    }

    #[test]
    fn inline_end_to_end_through_jit() {
        // `inline` callees (including multi-block bodies) execute
        // correctly, whether or not the pass inlines them here.
        assert_eq!(run_main("inline int add1(int x){return x+1;} int main(){return add1(41);}"), 42);
        assert_eq!(run_main("inline int pick(int x){if(x>0) return 1; return 0;} int main(){return pick(5)*10+pick(-3);}"), 10);
    }

    #[test]
    fn opt_pipeline_promotes_spills() {
        // The driver pipeline (inline, mem2reg, fold, simplify, DCE)
        // removes the -O0 spill slots from a trivial function while
        // keeping verification clean.
        use mir::pass::{FunctionPass, ModulePass};
        for src in [
            "int f(int x){return x+1;} int main(){return f(41);}",
            "inline int pick(int x){if(x>0) return 1; return 0;} int main(){return pick(5);}",
            "volatile int v; int main(){v = 1; return v;}",
        ] {
            let mut interner = fe::lexer::Interner::new();
            let mut sema = fe::sema::Sema::new();
            let tokens = fe::lexer::Lexer::new(src, &mut interner).lex_all();
            let mut parser = fe::parse::Parser::new(src, tokens, &mut sema, &mut interner);
            let tu = parser.parse_translation_unit();
            drop(parser);
            let mut module = lower_translation_unit(&sema, &interner, &tu);
            let mut inline = mir::passes::Inline::small();
            inline.run(&mut module);
            for func in module.funcs.iter_mut() {
                for _ in 0..16 {
                    let mut changed = false;
                    changed |= mir::passes::Mem2Reg.run(func);
                    changed |= mir::passes::ConstFold.run(func);
                    changed |= mir::passes::InstCombine.run(func);
                    changed |= mir::passes::Cse.run(func);
                    changed |= mir::passes::SimplifyCfg.run(func);
                    changed |= mir::passes::DeadCodeElim.run(func);
                    if !changed {
                        break;
                    }
                }
            }
            assert!(mir::verify::verify_module(&module).is_empty(), "verify errors in {src}");
        }
        // The non-volatile cases leave no spill traffic behind.
        let src = "int f(int x){int y = x; return y+1;} int main(){return f(41);}";
        let mut interner = fe::lexer::Interner::new();
        let mut sema = fe::sema::Sema::new();
        let tokens = fe::lexer::Lexer::new(src, &mut interner).lex_all();
        let mut parser = fe::parse::Parser::new(src, tokens, &mut sema, &mut interner);
        let tu = parser.parse_translation_unit();
        drop(parser);
        let mut module = lower_translation_unit(&sema, &interner, &tu);
        run_opt_pipeline(&mut module);
        assert!(mir::verify::verify_module(&module).is_empty());
        let f = module.funcs.iter().find(|f| f.name == "f").expect("f lowered");
        assert!(
            !f.blocks
                .iter()
                .flat_map(|b| b.insts.iter())
                .any(|i| matches!(f.insts[*i].kind, mir::InstKind::Alloca { .. } | mir::InstKind::Load { .. } | mir::InstKind::Store { .. })),
            "spills survived in f"
        );
        // ...while a volatile global access keeps its memory traffic.
        let src = "volatile int v; int main(){v = 1; return v;}";
        let mut interner = fe::lexer::Interner::new();
        let mut sema = fe::sema::Sema::new();
        let tokens = fe::lexer::Lexer::new(src, &mut interner).lex_all();
        let mut parser = fe::parse::Parser::new(src, tokens, &mut sema, &mut interner);
        let tu = parser.parse_translation_unit();
        drop(parser);
        let mut module = lower_translation_unit(&sema, &interner, &tu);
        run_opt_pipeline(&mut module);
        assert!(mir::verify::verify_module(&module).is_empty());
        let main = module.funcs.iter().find(|f| f.name == "main").expect("main lowered");
        assert!(
            main.blocks.iter().flat_map(|b| b.insts.iter()).any(|i| matches!(
                main.insts[*i].kind,
                mir::InstKind::Load { flags, .. } | mir::InstKind::Store { flags, .. } if flags.volatile
            )),
            "volatile traffic was optimized out"
        );
    }

    /// The checker's value conversion strips top-level qualifiers, so a
    /// volatile read through `*p`, `a[i]`, `s->x`, or a parameter
    /// reaches lowering typed as a plain value; lowering must resolve
    /// the designated object and keep the `.vol` flag. Without it the
    /// read is a plain load that Cranelift forwards a dominating store
    /// into.
    #[test]
    fn volatile_lvalue_reads_keep_their_flag() {
        let src = "struct S { int x; };\n\
                   struct B { int x : 3; };\n\
                   int param(volatile int n) { return n + 2; }\n\
                   int deref(volatile int *p) { return *p + 2; }\n\
                   int sub(volatile int *a) { return a[0] + 2; }\n\
                   int member(volatile struct S *s) { return s->x + 2; }\n\
                   int dot(void) { volatile struct S s; s.x = 1; return s.x + 2; }\n\
                   int bitfield(volatile struct B *b) { b->x = 3; return b->x + 2; }\n\
                   int local(void) { volatile int v = 1; return v + 2; }\n\
                   void store(volatile int *p, int v) { *p = v; }\n\
                   int plain(int n) { return n + 2; }\n\
                   int main(void) { return local() + dot(); }";
        let volatile_accesses = |module: &mir::Module, name: &str| -> (usize, usize) {
            let f = module.funcs.iter().find(|f| f.name == name).unwrap_or_else(|| panic!("{name} lowered"));
            let mut loads = 0;
            let mut stores = 0;
            for inst in f.blocks.iter().flat_map(|b| b.insts.iter()) {
                match &f.insts[*inst].kind {
                    mir::InstKind::Load { flags, .. } if flags.volatile => loads += 1,
                    mir::InstKind::Store { flags, .. } if flags.volatile => stores += 1,
                    _ => {}
                }
            }
            (loads, stores)
        };
        let forms = [
            ("param", true, true),
            ("deref", true, false),
            ("sub", true, false),
            ("member", true, false),
            ("dot", true, true),
            ("bitfield", true, true),
            ("local", true, true),
            ("store", false, true),
        ];
        let mut module = lower_module(src);
        for (name, want_load, want_store) in forms {
            let (loads, stores) = volatile_accesses(&module, name);
            assert_eq!(loads > 0, want_load, "volatile loads in {name} before optimization: {loads}");
            assert_eq!(stores > 0, want_store, "volatile stores in {name} before optimization: {stores}");
        }
        assert_eq!(volatile_accesses(&module, "plain"), (0, 0), "plain parameter marked volatile");
        run_opt_pipeline(&mut module);
        assert!(mir::verify::verify_module(&module).is_empty());
        for (name, want_load, want_store) in forms {
            let (loads, stores) = volatile_accesses(&module, name);
            assert_eq!(loads > 0, want_load, "volatile loads in {name} after optimization: {loads}");
            assert_eq!(stores > 0, want_store, "volatile stores in {name} after optimization: {stores}");
        }
        assert_eq!(volatile_accesses(&module, "plain"), (0, 0), "plain parameter marked volatile after optimization");
        assert_eq!(jit_run_main(module), 6);
    }

    #[test]
    fn negative_subscript_indices() {
        // C indices are signed: `p[-1]` walks backwards ( lowered
        // with a sign-extending step, not zero-extending).
        assert_eq!(
            run_main(
                "int main(){char s[] = \"abc\"; char *p = s + 2;\
                return (p[0] == 'c') + (p[-1] == 'b') + (p[-2] == 'a');}"
            ),
            3
        );
        assert_eq!(
            run_main(
                "int main(){int a[4] = {10, 20, 30, 40}; int *q = a + 3;\
                return (q[-3] == 10) + (q[0] == 40);}"
            ),
            2
        );
    }

    #[test]
    fn flex_array_member_subscript() {
        // Flexible array members have no MIR array form: indexing
        // must stride by the frontend element size, not pointer size.
        assert_eq!(
            run_main(
                "struct F{int n; unsigned vals[];};\
                int at(struct F *f, int i){return (int)f->vals[i];}\
                int main(){static unsigned backing[8] = {0, 11, 22, 33};\
                struct F *f = (struct F *)backing;\
                f->n = 3; return at(f, 0) + at(f, 1) + at(f, 2);}"
            ),
            66
        );
    }

    #[test]
    fn assignment_forms() {
        assert_eq!(run_main("int main(){int x = 1; x += 5; x = x * 2; return x;}"), 12);
        assert_eq!(run_main("int main(){int x = 10; x -= 3; x /= 7; return x;}"), 1);
        assert_eq!(run_main("int g = 0; int main(){g = 9; return g;}"), 9);
    }

    #[test]
    fn asm_empty_templates() {
        // Matching constraint: the input ties to the output's location.
        let (module, _) = lower_with_asm("int main(){int x=7; int y; __asm__(\"\" : \"=r\"(y) : \"0\"(x)); return y;}");
        let main = module.funcs.iter().find(|f| f.name == "main").expect("main");
        let asm = main
            .insts
            .iter()
            .find_map(|inst| match &inst.kind {
                mir::InstKind::InlineAsm { block } => Some(block.clone()),
                _ => None,
            })
            .expect("inline asm instruction");
        assert_eq!(asm.template, "");
        assert_eq!(asm.outputs, 1);
        assert_eq!(asm.operands.len(), 2);
        assert_eq!(asm.operands[0].constraint, "=r");
        assert_eq!(asm.operands[1].constraint, "0");
        assert_eq!(asm.operands[1].tie, Some(0));
        // Inputs evaluate in order before the block: `x=3` still runs.
        let (module, _) = lower_with_asm("int main(){int x=0; asm(\"\" : : \"r\"(x=3)); return x;}");
        let main = module.funcs.iter().find(|f| f.name == "main").expect("main");
        assert!(main.insts.iter().any(|inst| matches!(inst.kind, mir::InstKind::InlineAsm { .. })));
    }

    #[test]
    fn asm_immediate_operands_fold_constants() {
        // `"i"` takes any integer constant expression, not just a
        // literal: `-5` must reach the wrapper as a constant.
        let (module, _) = lower_with_asm("int main(){int y; asm(\"movl %1, %0\" : \"=r\"(y) : \"i\"(-5)); return y;}");
        let main = module.funcs.iter().find(|f| f.name == "main").expect("main");
        let block = main
            .insts
            .iter()
            .find_map(|inst| match &inst.kind {
                mir::InstKind::InlineAsm { block } => Some(block.clone()),
                _ => None,
            })
            .expect("inline asm instruction");
        assert_eq!(main.const_value(block.operands[1].value), Some(&mir::Const::Int { bits: 32, value: -5 }));
    }

    #[test]
    fn asm_immediate_or_register_alternatives() {
        // `"Ir"` folds a constant (the wrapper substitutes it) but
        // materializes a runtime value for the register alternative.
        fn operand_value(src: &str) -> Option<mir::Const> {
            let (module, _) = lower_with_asm(src);
            let main = module.funcs.iter().find(|f| f.name == "main").expect("main");
            let block = main
                .insts
                .iter()
                .find_map(|inst| match &inst.kind {
                    mir::InstKind::InlineAsm { block } => Some(block.clone()),
                    _ => None,
                })
                .expect("inline asm instruction");
            main.const_value(block.operands[1].value).cloned()
        }
        assert_eq!(
            operand_value("int main(){int y; asm(\"add %1, %0\" : \"+r\"(y) : \"Ir\"(5)); return y;}"),
            Some(mir::Const::Int { bits: 32, value: 5 })
        );
        assert_eq!(operand_value("int main(){int y, x; asm(\"add %1, %0\" : \"+r\"(y) : \"Ir\"(x)); return y;}"), None);
    }

    #[test]
    fn asm_goto_dispatches_on_the_selector() {
        let (module, _) = lower_with_asm("int main(){int x=0; asm goto(\"testl %0, %0; jnz %l[lab]\" : : \"r\"(x) : : lab); return 1; lab: return 2;}");
        let main = module.funcs.iter().find(|f| f.name == "main").expect("main");
        let asm = main
            .insts
            .iter()
            .find_map(|inst| match &inst.kind {
                mir::InstKind::InlineAsm { block } => Some((inst, block.clone())),
                _ => None,
            })
            .expect("inline asm instruction");
        assert_eq!(asm.1.labels, vec!["lab".to_string()]);
        let selector = asm.0.result.expect("asm goto yields a selector");
        let switch = main
            .blocks
            .iter()
            .find_map(|block| match &block.term {
                Some(mir::Terminator::Switch { scrutinee, cases, default }) if *scrutinee == selector => Some((cases.clone(), *default)),
                _ => None,
            })
            .expect("selector switch");
        assert_eq!(switch.0.len(), 1);
        assert_eq!(switch.0[0].0, 1);
        // The default is the fallthrough, the case is the `lab` block.
        let returns = |block: mir::Block, value: i128| {
            matches!(
                main.blocks[block].term,
                Some(mir::Terminator::Return { val: Some(v) })
                    if main.const_value(v) == Some(&mir::Const::Int { bits: 32, value })
            )
        };
        assert!(returns(switch.1, 1), "fallthrough returns 1");
        assert!(returns(switch.0[0].1, 2), "the label block returns 2");
    }

    #[test]
    fn calls_and_recursion() {
        assert_eq!(run_main("int sq(int v){return v*v;} int main(){return sq(7);}"), 49);
        assert_eq!(run_main("int fib(int n){if(n<2) return n; return fib(n-1)+fib(n-2);} int main(){return fib(10);}"), 55);
    }

    #[test]
    fn comparisons_and_logic() {
        assert_eq!(run_main("int main(){return (3<5)+(5<3)+(4==4);}"), 2);
        assert_eq!(run_main("int main(){return (7>=7)+(6!=6)+(2>3);}"), 1);
        // Short-circuit must not evaluate the trapping side.
        assert_eq!(run_main("int main(){return (0 && (1/0));}"), 0);
        assert_eq!(run_main("int main(){return (1 || (1/0));}"), 1);
        assert_eq!(run_main("int main(){int x = 5; return (x>3) ? 10 : 20;}"), 10);
        assert_eq!(run_main("int main(){int x = 1; return (x>3) ? 10 : 20;}"), 20);
    }

    #[test]
    fn loops() {
        assert_eq!(run_main("int main(){int s = 0; int i = 1; while(i<=100){s += i; i++;} return s;}"), 5050);
        assert_eq!(run_main("int main(){int s = 0; int i = 1; do { s += i; i++; } while(i<=10); return s;}"), 55);
        assert_eq!(run_main("int main(){int s = 0; for(int i = 1; i<=10; i++){s += i;} return s;}"), 55);
        assert_eq!(
            run_main("int main(){int s = 0; for(int i = 0; i<10; i++){if(i%2==0) continue; if(i>7) break; s += i;} return s;}"),
            16
        );
    }

    #[test]
    fn wide_conditions_test_full_width() {
        // Conditions test the whole value: narrowing to I32 dropped
        // the high half, so `while (1ULL << 32)` never ran.
        assert_eq!(run_main("int main(){unsigned long long v = 1ULL << 32; int n = 0; while(v){n++; v >>= 33;} return n;}"), 1);
        assert_eq!(run_main("int main(){long long v = -9223372036854775807LL - 1; if(v){return 1;} return 0;}"), 1);
        assert_eq!(run_main("int main(){unsigned long long v = 1ULL << 32; return v ? 7 : 8;}"), 7);
        assert_eq!(run_main("int main(){unsigned long long v = 0; return v ? 7 : 8;}"), 8);
    }

    #[test]
    fn address_and_deref() {
        assert_eq!(run_main("int main(){int x = 5; int *p = &x; *p += 2; return x;}"), 7);
        assert_eq!(run_main("int main(){int x = 5; int *p = &x; int y = *p; return y;}"), 5);
    }

    #[test]
    fn pointer_arithmetic() {
        assert_eq!(
            run_main("int sum(int *p, int n){int s = 0; for(int i = 0; i<n; i++) s += p[i]; return s;} int a[4] = {1,2,3,4}; int main(){return sum(a, 4);}"),
            10
        );
        assert_eq!(run_main("int main(){int a[4]; int *p = &a[0]; int *q = &a[3]; return q-p;}"), 3);
        assert_eq!(run_main("int main(){int a[3] = {10,20,30}; int *p = a+1; return *p + *(p+1);}"), 50);
    }

    #[test]
    fn strings() {
        assert_eq!(
            run_main("int slen(char *s){int n = 0; while(*s){n++; s++;} return n;} int main(){return slen(\"hello\");}"),
            5
        );
        assert_eq!(run_main("int main(){char s[] = \"abc\"; return s[0]+s[1]+s[2];}"), 294);
        assert_eq!(run_main("char g[] = \"hi\"; int main(){return g[0]+g[1];}"), 209);
    }

    #[test]
    fn sizes() {
        let t = fe::target::Target::native();
        assert_eq!(
            run_main("int main(){return sizeof(int)+sizeof(char)+sizeof(long);}"),
            (t.int_size + t.char_size + t.long_size) as i32
        );
        assert_eq!(run_main("int main(){int a[10]; return sizeof(a)/sizeof(a[0]);}"), 10);
    }

    #[test]
    fn structs() {
        assert_eq!(run_main("struct P{int x; int y;}; int main(){struct P p; p.x = 3; p.y = 4; return p.x+p.y;}"), 7);
        assert_eq!(run_main("struct P{int x; int y;}; int main(){struct P p = {3,4}; return p.x+p.y;}"), 7);
        assert_eq!(
            run_main("struct P{int x; int y;}; int get(struct P p){return p.x+p.y;} int main(){return get((struct P){3,4});}"),
            7
        );
        assert_eq!(run_main("struct P{int x; int y;}; struct P g = {.y = 5}; int main(){return g.x+g.y;}"), 5);
    }

    #[test]
    fn bool_conversions() {
        // Integer to `_Bool` is a truth test, not a truncation: 256
        // is nonzero, so true.
        assert_eq!(run_main("int main(){_Bool b = (_Bool)256; return b;}"), 1);
        assert_eq!(run_main("int main(){_Bool b = (_Bool)0; return b;}"), 0);
        assert_eq!(run_main("int main(){long x = -1; _Bool b = x; return b;}"), 1);
        // Pointers test against null: a live address is true even
        // when its low byte is zero; null is false.
        assert_eq!(run_main("int main(){int x = 1; int *p = &x; _Bool b = (_Bool)p; return b;}"), 1);
        assert_eq!(run_main("int main(){int *p = (int *)0; _Bool b = (_Bool)p; return b;}"), 0);
    }

    /// Integer-to-pointer casts extend by the source's signedness (C17
    /// 6.3.2.3p5-6): MIR integer types are signless, so the conversion
    /// has to sign-extend signed narrow sources to pointer width and
    /// zero-extend unsigned ones. `(void *)-1` must be the all-ones
    /// pattern, not `0xffffffff` (SQLite's `sqlite3Utf8CharLen(z, -1)`
    /// sentinel leans on this).
    #[test]
    fn integer_to_pointer_casts_respect_signedness() {
        assert_eq!(run_main("int main(){const void *p = (const void*)-1; return (p == (const void*)-1) + ((long)p == -1);}"), 2);
        // Plain `char` is signed on x86-64 but unsigned on aarch64 and
        // riscv64, so spell out the signedness this check needs.
        assert_eq!(
            run_main(
                "int main(){volatile int i = -1; volatile short s = -1; volatile signed char c = -1;\
                return ((long)(void*)i == -1) + ((long)(void*)s == -1) * 2 + ((long)(void*)c == -1) * 4;}"
            ),
            7
        );
        // Unsigned narrow sources stay zero-extended, including the
        // full-width unsigned value whose sign bit is set.
        assert_eq!(
            run_main(
                "int main(){volatile unsigned u = 4294967295u; volatile unsigned short h = 65535; volatile unsigned char b = 255;\
                return ((unsigned long)(void*)u == 4294967295UL) + ((unsigned long)(void*)h == 65535UL) * 2 + ((unsigned long)(void*)b == 255UL) * 4;}"
            ),
            7
        );
        assert_eq!(run_main("int main(){volatile long l = -1; return (long)(void*)l == -1;}"), 1);
    }

    /// The conversion widens every `inttoptr` source to pointer width
    /// before the cast, so the backend never re-decides the extension.
    #[test]
    fn integer_to_pointer_source_widens_to_pointer_width() {
        let module = lower_module(
            "void sink(const void *p); void f(volatile int i, volatile unsigned u, volatile short s, volatile char c) {\
                sink((const void*)i); sink((const void*)u); sink((const void*)s); sink((const void*)c); }",
        );
        let mut seen = 0;
        for func in module.funcs.iter().filter(|f| f.name == "f") {
            for inst in func.insts.iter() {
                if let mir::InstKind::IntToPtr(src) = &inst.kind {
                    assert_eq!(func.values[*src].ty, mir::Ty::int(64));
                    seen += 1;
                }
            }
        }
        assert_eq!(seen, 4, "casts lowered through inttoptr");
    }

    /// Plain `char` takes its signedness from the target (signed on
    /// x86-64, unsigned on aarch64 and riscv64); `signed char` and
    /// `unsigned char` never do. The widening before `inttoptr` follows.
    #[test]
    fn int_to_pointer_extension_follows_target_char_signedness() {
        let src = "void sink(const void *p); void f(volatile char c, volatile signed char s, volatile unsigned char u) {\
                    sink((const void*)c); sink((const void*)s); sink((const void*)u); }";
        let cases = [
            (fe::target::Target::X86_64, ["sext", "sext", "zext"]),
            (fe::target::Target::AARCH64, ["zext", "sext", "zext"]),
        ];
        for (target, expected) in cases {
            let module = lower_module_for(src, target);
            let mut exts = Vec::new();
            for func in module.funcs.iter().filter(|f| f.name == "f") {
                for inst in func.insts.iter() {
                    let mir::InstKind::IntToPtr(v) = &inst.kind else {
                        continue;
                    };
                    let def = func.insts.iter().find(|i| i.result == Some(*v)).expect("inttoptr source widens to pointer width first");
                    exts.push(match &def.kind {
                        mir::InstKind::Sext(_) => "sext",
                        mir::InstKind::Zext(_) => "zext",
                        _ => panic!("inttoptr source is not a sext/zext"),
                    });
                }
            }
            assert_eq!(exts, expected, "extension before inttoptr on {target:?}");
        }
    }

    /// The optimized pipeline keeps the conversion values intact with a
    /// load in the way (nothing to constant-fold).
    #[test]
    fn optimized_pipeline_preserves_pointer_cast_signedness() {
        let src = "int main(){volatile int i = -1; volatile unsigned u = 4294967295u; volatile unsigned short h = 65535;\
                   return ((long)(void*)i != -1)\
                        + ((unsigned long)(void*)u != 4294967295UL) * 2\
                        + ((long)(void*)h != 65535) * 4;}";
        let mut module = lower_module(src);
        run_opt_pipeline(&mut module);
        let dump = mir::display::dump_module(&module);
        assert_eq!(jit_run_main(module), 0, "casts changed values:\n{dump}");
    }

    #[test]
    fn bool_to_byte_convert_is_a_no_op() {
        // `_Bool` registers hold a byte, so a conversion to an 8-bit
        // integer is already the target representation: no
        // zero-extension (which Cranelift rejects at equal widths).
        assert_eq!(run_main("unsigned char f(_Bool b){ return (unsigned char)b; } int main(){ return f(1) + f(0); }"), 1);
        assert_eq!(run_main("int f(_Bool b){ char c = b; return c; } int main(){ return f(1); }"), 1);
    }

    #[test]
    fn debug_records_describe_functions_scopes_and_globals() {
        let module = lower_module(
            "int g = 7;\n\
             extern int ext;\n\
             static int counter;\n\
             struct P { int x; unsigned y : 3; };\n\
             struct P p = { 1, 2 };\n\
             int add(int a, int b) {\n\
                 int sum = a + b;\n\
                 for (int i = 0; i < 2; i++) { sum += i; }\n\
                 { int shadow = sum; sum = shadow; }\n\
                 return sum;\n\
             }\n\
             int main(void) { int x = 3; return add(x, g); }",
        );
        assert_eq!(module.files.len(), 1, "one source file");
        assert_eq!(module.files[0].path, "<__glcc_source>");
        assert!(module.funcs.iter().all(|f| f.debug.is_some()), "every defined function carries a debug record");

        fn collect(vars: &[mir::VarDebug], scopes: &[mir::ScopeDebug], out: &mut Vec<String>) {
            out.extend(vars.iter().filter_map(|v| v.name.clone()));
            for scope in scopes {
                collect(&scope.vars, &scope.children, out);
            }
        }

        let add = module.funcs.iter().find(|f| f.name == "add").unwrap();
        let dbg = add.debug.as_ref().expect("add debug record");
        assert!(dbg.decl.is_some());
        assert_eq!(dbg.params.iter().filter_map(|p| p.name.as_deref()).collect::<Vec<_>>(), ["a", "b"]);
        let mut names = Vec::new();
        collect(&dbg.vars, &dbg.scopes, &mut names);
        for expected in ["sum", "i", "shadow"] {
            assert!(names.contains(&expected.to_string()), "missing {expected}: {names:?}");
        }
        fn check_scopes(scopes: &[mir::ScopeDebug]) {
            for scope in scopes {
                assert!(scope.start.line <= scope.end.line, "{scope:?}");
                check_scopes(&scope.children);
            }
        }
        check_scopes(&dbg.scopes);
        // Every instruction maps to a line of the body.
        for inst in add.insts.iter() {
            let loc = inst.loc.expect("instruction location");
            assert_eq!(loc.file, 0);
            assert!((6..=11).contains(&loc.line), "line {}", loc.line);
        }

        // The struct global carries member positions, including the
        // bit-field as a bit offset and width.
        let p = global_named(&module, "p");
        let gd = p.debug.as_ref().expect("p debug record");
        match &module.debug_types[gd.debug_ty.expect("p type") as usize] {
            mir::DebugType::Struct { name, members, .. } => {
                assert_eq!(name.as_deref(), Some("P"));
                assert_eq!(members.len(), 2);
                assert_eq!(members[0].name.as_deref(), Some("x"));
                assert_eq!(members[0].location, mir::MemberLocation::Byte(0));
                assert_eq!(members[1].name.as_deref(), Some("y"));
                assert_eq!(members[1].location, mir::MemberLocation::Bit { offset: 32, size: 3 });
            }
            other => panic!("struct P lowered as {other:?}"),
        }

        // Defined, tentative-static, and imported objects all carry
        // records, like the legacy collector.
        assert!(global_named(&module, "g").debug.is_some());
        assert!(global_named(&module, "counter").debug.is_some());
        assert!(global_named(&module, "ext").debug.is_some());
        let sum = dbg.vars.iter().find(|v| v.name.as_deref() == Some("sum")).unwrap();
        assert!(matches!(sum.home, mir::VarHome::Slot(_)));
    }
}

/// Positional index of a named struct member.
fn member_index(sema: &fe::sema::Sema, id: fe::sema::StructId, name: string_interner::symbol::SymbolU32) -> u32 {
    sema.structs[id.0 as usize]
        .members
        .iter()
        .position(|m| m.name == Some(name))
        .unwrap_or_else(|| panic!("unknown struct member")) as u32
}

/// The member a struct-level designator names: `Field` searches by
/// name, `AnonField` is a hop through an anonymous member that the
/// checker already resolved to an index.
fn designated_member(sema: &fe::sema::Sema, id: fe::sema::StructId, des: &fe::parse::Designator) -> u32 {
    match des {
        fe::parse::Designator::Field { name, .. } => member_index(sema, id, *name),
        fe::parse::Designator::AnonField { index, .. } => *index,
        fe::parse::Designator::Index { .. } => panic!("index designator into struct"),
    }
}
