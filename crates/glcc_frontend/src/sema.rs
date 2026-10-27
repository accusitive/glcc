use alloc::vec::Vec;
use alloc::{boxed::Box, string::String, vec};
use core::cell::{Cell, RefCell};

use hashbrown::{HashMap, HashSet};

use string_interner::symbol::SymbolU32;

use crate::check::{CheckedExpr, CheckedExprKind};
use crate::eval::escape_value;
use crate::lang::LangOpts;
use crate::layout::StructLayout;
use crate::lexer::{Interner, StringLiteral, StringPrefix};
use crate::parse::{ArrayBound, AsmOperand, AsmStmt, BlockItem, Compound, Designator, ForInit, FuncBody, InitItem, Initializer, PtrSize, Stmt, Ty, TypeQualifiers};
use crate::span::{CompileError, HasSpan, Span};
use crate::target::Target;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeclId(pub u32);

// A struct or union type is nominal: `Ty::Struct` carries only this id,
// while members and completeness live in the `StructDef` behind it. That
// keeps a forward-declared and a completed `struct S` the *same* `Ty`, so
// redeclaration merging in `Sema::declare` never sees a changed type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructId(pub u32);

// An enum type is nominal like a struct/union: `Ty::Enum` carries only
// this id, while the enumerators and completeness live in the `EnumDef`
// behind it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnumId(pub u32);

// One enumeration constant: its `= value` folded while parsing, or
// `None` to continue the sequence. Uses of the constant check with type
// `int` (C17 §6.4.4.3).
#[derive(Debug)]
pub struct Enumerator {
    pub name: SymbolU32,
    /// Explicit value, already folded; `None` continues the sequence.
    pub value: Option<i128>,
    pub span: Span,
}

impl HasSpan for Enumerator {
    fn span(&self) -> Span {
        self.span
    }
}

#[derive(Debug)]
pub struct EnumDef {
    pub tag: Option<SymbolU32>,       // `None`: anonymous enum
    pub enumerators: Vec<Enumerator>, // empty until the body is parsed
    pub complete: bool,
    /// Smallest type holding every value (C23 §6.7.2.2p4 ladder:
    /// `int`, `unsigned int`, `long`, `unsigned long`): `int` for
    /// ordinary enums, wider for sentinel values like
    /// `DW_CIE_ID_32`, or the declared fixed underlying type in C23.
    /// Fixed at completion; layout, conversions, and promotions all
    /// key off it.
    pub repr: Ty,
    /// C23 `enum E : type`: the declared underlying type, retained so
    /// later declarations must agree with it.
    pub fixed: Option<Ty>,
    pub span: Span,
}

impl HasSpan for EnumDef {
    fn span(&self) -> Span {
        self.span
    }
}

/// One member of a struct/union. `name` is `None` for anonymous
/// struct/union members (whose own members promote for access) and for
/// unnamed bit-fields (`int : 3;`, `unsigned : 0;`). `bitfield` is the
/// width in bits, folded and range-checked while parsing (see the member
/// parser). GNU `packed`/`aligned` on the member adjust layout (see
/// `crate::layout`); all other attributes are ignored at parse.
#[derive(Debug, Clone)]
pub struct Member {
    pub name: Option<SymbolU32>,
    pub ty: Ty,
    /// Bit-field width in bits, already folded and range-checked.
    pub bitfield: Option<u64>,
    /// `__attribute__((packed))` on this member: no padding before it.
    pub packed: bool,
    /// `__attribute__((aligned(n)))` on this member: a minimum
    /// alignment (GCC semantics).
    pub align_override: Option<u64>,
    pub span: Span,
}

impl HasSpan for Member {
    fn span(&self) -> Span {
        self.span
    }
}

#[derive(Debug)]
pub struct StructDef {
    pub tag: Option<SymbolU32>, // `None`: anonymous struct/union
    pub is_union: bool,
    pub members: Vec<Member>, // empty until the body is parsed
    pub complete: bool,
    /// `__attribute__((packed))` on the definition: every member
    /// packs (alignment 1 unless a member raises it).
    pub packed: bool,
    /// `__attribute__((aligned(n)))` on the definition: a minimum
    /// overall alignment (GCC semantics).
    pub align_override: Option<u64>,
    /// Memoized `struct_layout` result. Layouts are pure functions of
    /// the definition (plus target, fixed per `Sema`), so the first
    /// successful computation sticks; `complete_tag` and the
    /// attribute application below it clear the entry they precede.
    pub layout_cache: RefCell<Option<StructLayout>>,
    pub span: Span,
}

impl HasSpan for StructDef {
    fn span(&self) -> Span {
        self.span
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingKind {
    Var,
    Func,
    Typedef,
    /// An enumeration constant: type `int` (C17 §6.4.4.3), no linkage,
    /// not assignable and not addressable. The value is folded at
    /// declaration (see `crate::eval`), so later enumerators and
    /// constant expressions can use it.
    EnumConst {
        value: i128,
    },
}

/// Linkage of an identifier (C17 §6.2.2).
///
/// - `NoLinkage`: block-scope objects without `extern`, parameters,
///   and `typedef` names. Each declaration is a distinct entity;
///   inner scopes shadow outer ones.
/// - `Internal`: file-scope `static` (and block-scope `extern`
///   referring to one). One entity per translation unit.
/// - `External`: file-scope without `static`, and block-scope
///   `extern` (or function declarations without storage, which act
///   as if declared with `extern`). One entity for the whole program;
///   a block-scope `extern` reuses the visible file-scope entity
///   instead of shadowing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    NoLinkage,
    Internal,
    External,
}
#[derive(Debug)]
pub struct Binding {
    pub name: Option<SymbolU32>, // the spelling, for diagnostics only
    pub ty: Ty,                  // single source of truth for the type
    pub kind: BindingKind,       // Var, Func, Typedef, or EnumConst
    pub linkage: Linkage,
    /// Block-scope `static` object: static storage duration despite
    /// having no linkage (C17 §6.2.4p3). The backend gives it a local
    /// data symbol rather than a stack slot, and every declaration in
    /// the block names a distinct object.
    pub block_static: bool,
    /// Thread-local storage (`_Thread_local`, `__thread`): the backend
    /// declares the symbol with TLS and accesses it via `tls_value`.
    /// Combines with `static`/`extern`; a block-scope thread-local
    /// without `extern` gets static storage like `block_static`.
    pub tls: bool,
    pub defined: bool, // external definition seen (`= init` or function body)
    // File-scope `int x;` without `extern` or initializer: not yet a
    // definition, but becomes one at end of TU if none appears
    // (C17 §6.9.2). `extern int x;` is neither defined nor tentative.
    pub tentative: bool,
    // `__attribute__((constructor[(prio)]))` / `(destructor)`: startup
    // and shutdown hooks, run via `.init_array` / `.fini_array` in
    // increasing priority (bare attributes are 65535, like GCC), then
    // declaration order. Set from any declaration or the definition;
    // first wins on merge (the extern prototype usually carries it).
    pub constructor: Option<u64>,
    pub destructor: Option<u64>,
    /// `__attribute__((weak))`: emit with weak linkage. Merged with OR:
    /// any declaration may mark the symbol weak.
    pub weak: bool,
    /// `__attribute__((alias("name")))`: this declaration denotes the
    /// named entity's storage (GCC requires the target to be declared
    /// in the same translation unit). Uses lower to the target's
    /// global/function, so no symbol is emitted for the alias name.
    pub alias: Option<DeclId>,
    /// `__attribute__((naked))`: no compiler prologue/epilogue; the
    /// definition's body is emitted as assembler text instead of being
    /// lowered. Merged with OR like `inline`, so a prototype may carry
    /// it for a definition that omits the attribute.
    pub naked: bool,
    /// `__attribute__((async))` (glcc extension, `-fglcc-async`): the
    /// function compiles as a stackless coroutine. `name` returns the
    /// boxed `_Future(T)` handle and `name$resume` the coroutine body.
    /// Merged with OR like `naked`, so a prototype may carry it for a
    /// definition that omits the attribute (cross-TU headers).
    pub is_async: bool,
    /// C `inline` function specifier (`inline`, `__inline__`,
    /// `__inline`, `__forceinline`): an optimization hint requesting call-site
    /// inlining. Recorded on function bindings only (the parser
    /// rejects it elsewhere); merged with OR like `weak`, so an
    /// `inline` prototype marks the definition too. The MIR
    /// lowering copies it to `Function::inline`, where the `inline`
    /// pass uses a larger budget for marked callees.
    pub inline: bool,
    /// `__attribute__((noinline))`: the definition must not be inlined
    /// at its call sites. Merged with OR like `inline`, so a prototype
    /// may carry it for a definition that omits the attribute; the MIR
    /// lowering copies it to `Function::noinline`.
    pub noinline: bool,
    /// `__attribute__((aligned(n)))` on this declaration: the alignment
    /// from any of its declarations (merged with max, like GCC). The
    /// attribute *sets* the object's alignment — raising or lowering it
    /// (`long x __attribute__((aligned(1)))` is 1-aligned, matching
    /// gcc); without one the type's alignment applies. `__alignof__(f)`
    /// reports it for functions (1 without one, like GCC).
    pub align: Option<u64>,
    /// `__attribute__((section("name")))`: emit the object (or a naked
    /// function) into that section instead of `.data`/`.bss`/`.text`.
    /// Merged with first-wins across declarations. Functions with a
    /// compiler-generated body cannot honor it (the object writer has
    /// no per-function section), which the parser warns about.
    pub section: Option<SymbolU32>,
    /// `__asm__("name")` on a declarator (GNU asm labels): the printed
    /// symbol name, while the C spelling stays the declaration's lookup
    /// key. Merged with first-wins across declarations. EDG emits these
    /// for renamed C++ entities — overloaded C library declarations
    /// come out as `_Z6strchrPci(...) __asm__("strchr")`, and the
    /// backend must reference `strchr` rather than the mangled name.
    pub asm_label: Option<SymbolU32>,
    pub span: Span,
}

impl HasSpan for Binding {
    fn span(&self) -> Span {
        self.span
    }
}

#[derive(Debug)]
pub struct Sema<'a> {
    /// Bump arena owning every parsed and checked syntax node. The
    /// driver creates it before the `Sema` and keeps it alive for the
    /// whole frontend run.
    pub arena: &'a bumpalo::Bump,
    pub decls: Vec<Binding>,
    pub structs: Vec<StructDef>,
    pub enums: Vec<EnumDef>,
    pub scopes: ScopeStack,
    /// The target being compiled for: every size, alignment, and
    /// representation choice in layout and `sizeof` reads this
    /// (LP64 x86-64 by default; the target-free helpers in
    /// `crate::check` and `crate::eval` default to LP64 too).
    pub target: Target,
    /// Dialect the parser gates spellings against (permissive default;
    /// the driver sets it from `-std=`/`-f*-extensions`).
    pub lang: LangOpts,
    /// `extern inline` definitions emit weak symbols. Off for C (GNU89
    /// strong-standalone semantics); the C++ driver enables it because
    /// EDG lowers every C++ `inline` function to `extern inline`, which
    /// needs vague linkage or multi-TU links fail on duplicates.
    pub weak_extern_inline: bool,
    /// `-fglcc-async`: enables the async extension (`__attribute__((async))`,
    /// `_Future(T)`, `__builtin_await`). Without it those spellings
    /// diagnose rather than compiling as something else.
    pub async_enabled: bool,
    /// Handle structs synthesized for `_Future(T)`, keyed by result
    /// type: one hidden `{ void *frame; void *poll; }` struct per `T`,
    /// so every mention of the same `_Future(T)` is one nominal type.
    pub future_handles: RefCell<Vec<(Ty, StructId)>>,
    /// True while the body of an `__attribute__((async))` definition is
    /// being checked; `__builtin_await` is only valid there. Interior
    /// mutability because checking takes `&self`.
    pub(crate) checking_async_fn: Cell<bool>,
    /// Structs currently being laid out (see `crate::layout`): guards
    /// against recursive types, which the parser rejects but which can
    /// still be built below it.
    pub(crate) visiting: RefCell<Vec<StructId>>,
    /// Non-fatal diagnostics collected during checking (overlong string
    /// initializers, which are a GCC warning): the parser renders and
    /// clears them after each declaration. Interior mutability because
    /// checking takes `&self` throughout.
    pub(crate) warnings: RefCell<Vec<CompileError>>,
    /// Parsed function bodies by function `DeclId`. Only definitions
    /// (`defined`) have entries.
    pub bodies: HashMap<DeclId, FuncBody<'a>>,
    /// Object initializers by object `DeclId`: the *checked* form
    /// (implicit conversions inserted), which later phases consume.
    /// Only definitions with `= ...` have entries.
    pub inits: HashMap<DeclId, Initializer<'a>>,
    /// C23 `constexpr` values, usable in constant expressions such as
    /// array bounds even though they live in objects.
    pub constexprs: HashMap<DeclId, crate::eval::Value>,
    /// C23 `[[...]]` attributes attached to declarations.
    pub std_attrs: HashMap<DeclId, crate::parse::StdAttributes>,
    /// GNU label addresses (`&&label`) and computed `goto`: one small
    /// id per label spelling, assigned when the label is defined (so
    /// ids run in definition order) and keyed by spelling, with the
    /// first definition winning. Ids of one function stay consecutive
    /// like GCC's byte offsets, keeping `&&b - &&a` arithmetic
    /// meaningful; keying by spelling alone lets a label address in a
    /// static initializer resolve without knowing its function.
    pub label_ids: HashMap<SymbolU32, u64>,
    /// Checked VLA bound expressions by `ArrayBound::Vla` index, in
    /// parse order. `Ty` stays free of source positions (only the
    /// index travels in the type); the backend evaluates these wherever
    /// a runtime size is needed. Bounds checked in prototype scope may
    /// reference prototype bindings — a definition remaps those to its
    /// parameters (see `remap_vla_idents`).
    pub vla_bounds: Vec<CheckedExpr<'a>>,
    /// Declarations whose address is ever taken with `&`: populated
    /// during checking (`check_unary`'s `AddrOf` case walks the
    /// operand's lvalue base to its `Ident`). Interior mutability
    /// because checking takes `&self` throughout (like `warnings`).
    /// The backend reads this (via `is_address_taken`) to decide
    /// which locals must live in memory.
    pub address_taken: RefCell<HashSet<DeclId>>,
    /// Original (pre-preprocessor) line per TU-buffer line, from
    /// linemarkers (`# 26 "tcc.h"` means the next line is 26). The
    /// lexer keeps spans buffer-relative (see `SourceFiles`), so the
    /// backend maps through this for DWARF (`-g`); diagnostics map
    /// the same way at render time. Empty when unparsed.
    pub orig_lines: Vec<usize>,
    /// File-scope `asm("...")` templates in source order (decoded).
    /// They go into the per-TU assembler text, not the object the
    /// Cranelift backend emits.
    pub toplevel_asm: Vec<(String, Span)>,
}
#[derive(Debug, Default)]
pub struct ScopeStack {
    global: Scope,
    scopes: Vec<Scope>,
}
impl ScopeStack {
    pub fn current_mut(&mut self) -> &mut Scope {
        match self.scopes.last_mut() {
            Some(s) => s,
            None => &mut self.global,
        }
    }
    pub fn current(&self) -> &Scope {
        match self.scopes.last() {
            Some(s) => s,
            None => &self.global,
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = &Scope> {
        self.scopes.iter().rev().chain(core::iter::once(&self.global))
    }
    /// Depth of pushed block/prototype scopes: `0` is file scope.
    pub fn depth(&self) -> usize {
        self.scopes.len()
    }
}
#[derive(Debug, Default)]
pub struct Scope {
    pub ordinary: HashMap<SymbolU32, DeclId>,
    // Struct/union tags live in their own namespace (C17 §6.2.1): a tag
    // and an ordinary identifier may share a spelling without interacting
    // (`struct test; int test;` declares two unrelated things).
    pub tags: HashMap<SymbolU32, StructId>,
    // Enum tags share that namespace (a `struct S` and an `enum S` in
    // one scope conflict); they get their own map only so each kind
    // resolves to its own definition table.
    pub enum_tags: HashMap<SymbolU32, EnumId>,
}
impl Sema<'static> {
    /// A test-only `Sema` whose arena lives for the rest of the process.
    #[doc(hidden)]
    pub fn leak_for_tests() -> Self {
        Self::leak_with_target_for_tests(Target::LP64)
    }

    /// A test-only `Sema` for another target.
    #[doc(hidden)]
    pub fn leak_with_target_for_tests(target: Target) -> Self {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        Self::with_target(target, arena)
    }
}

impl<'a> Sema<'a> {
    /// Move a checked node into the shared arena.
    pub(crate) fn alloc<T>(&self, value: T) -> &'a T {
        self.arena.alloc(value)
    }

    /// Freeze an iterator of nodes into an arena slice.
    pub(crate) fn alloc_slice<T>(&self, items: impl IntoIterator<Item = T>) -> &'a [T] {
        let mut out = bumpalo::collections::Vec::new_in(self.arena);
        out.extend(items);
        out.into_bump_slice()
    }

    pub fn new(arena: &'a bumpalo::Bump) -> Self {
        Self::with_target(Target::LP64, arena)
    }
    /// A `Sema` for another target (layout and `sizeof` follow it).
    pub fn with_target(target: Target, arena: &'a bumpalo::Bump) -> Self {
        Self {
            arena,
            decls: vec![],
            structs: vec![],
            enums: vec![],
            scopes: ScopeStack::default(),
            target,
            lang: LangOpts::default(),
            weak_extern_inline: false,
            async_enabled: false,
            future_handles: RefCell::new(Vec::new()),
            checking_async_fn: Cell::new(false),
            visiting: RefCell::new(Vec::new()),
            warnings: RefCell::new(Vec::new()),
            bodies: HashMap::new(),
            inits: HashMap::new(),
            constexprs: HashMap::new(),
            std_attrs: HashMap::new(),
            label_ids: HashMap::new(),
            vla_bounds: Vec::new(),
            address_taken: RefCell::new(HashSet::new()),
            orig_lines: Vec::new(),
            toplevel_asm: Vec::new(),
        }
    }
    /// The handle type for `_Future(result)`: a two-word type-erased
    /// boxed frame, synthesized on first mention and interned by the
    /// result type (see `future_handles`).
    pub fn future_handle(&mut self, result: Ty) -> Ty {
        if let Some(ty) = self.future_handle_of(&result) {
            return ty;
        }
        let ptr = || Ty::Pointer {
            qualifiers: TypeQualifiers::empty(),
            size: PtrSize::Default,
            pointee: Box::new(Ty::Void),
        };
        let span = Span::dummy();
        let id = StructId(self.structs.len() as u32);
        self.structs.push(StructDef {
            tag: None,
            is_union: false,
            members: vec![
                Member {
                    name: None,
                    ty: ptr(),
                    bitfield: None,
                    packed: false,
                    align_override: None,
                    span,
                },
                Member {
                    name: None,
                    ty: ptr(),
                    bitfield: None,
                    packed: false,
                    align_override: None,
                    span,
                },
            ],
            complete: true,
            packed: false,
            align_override: None,
            layout_cache: RefCell::new(None),
            span,
        });
        self.future_handles.borrow_mut().push((result, id));
        Ty::Struct(id)
    }
    /// The handle type for `_Future(result)`, if one was synthesized.
    /// Every async declaration makes one, so a call site of an async
    /// function always finds its handle here.
    pub fn future_handle_of(&self, result: &Ty) -> Option<Ty> {
        self.future_handles.borrow().iter().find(|(r, _)| r == result).map(|(_, id)| Ty::Struct(*id))
    }
    /// The result type of a `_Future(T)` handle type: `None` unless
    /// `ty` is a future handle.
    pub fn future_result_of(&self, ty: &Ty) -> Option<Ty> {
        let Ty::Struct(id) = ty.unqualified() else {
            return None;
        };
        self.future_handles.borrow().iter().find(|(_, sid)| sid == id).map(|(r, _)| r.clone())
    }
    /// True at file scope (no pushed block/prototype scopes).
    pub fn is_file_scope(&self) -> bool {
        self.scopes.depth() == 0
    }
    pub fn push_scope(&mut self) {
        self.scopes.scopes.push(Scope::default());
    }
    pub fn pop_scope(&mut self) {
        assert!(!self.scopes.scopes.is_empty());
        self.scopes.scopes.pop();
    }
    // Record `name` in the current scope, merging with a prior
    // declaration of the same entity when C says they are one:
    //
    // - A binding *with* linkage (`Internal`/`External`) merges with
    //   the innermost visible binding with linkage of the same name
    //   (C17 §6.2.2p4): file-scope redeclarations meet in `global`,
    //   and a block-scope `extern` (or a block-scope function without
    //   storage, which acts as `extern`) reuses the file-scope entity
    //   instead of shadowing it. No new scope entry is made; the
    //   existing `DeclId` is returned.
    // - A binding with `NoLinkage` is always a new entity shadowing
    //   outer scopes; only the current scope is checked, and any hit
    //   is a same-scope redefinition error — except a `typedef`
    //   repeated with a compatible type, which merges (C11 onwards).
    //
    // Merging checks kind, linkage, and type compatibility (function
    // types use C17 §6.2.7/§6.7.6.3 compatibility, so `int f();`
    // merges with `int f(int);`). The stored type becomes the
    // composite. Two non-tentative definitions of one object, or two
    // bodies for one function, fail. Failures carry both spans, so the
    // renderer can point at the conflicting and the previous
    // declaration.
    pub fn declare(&mut self, name: SymbolU32, binding: Binding) -> Result<DeclId, CompileError> {
        if binding.linkage == Linkage::NoLinkage {
            if let Some(existing) = self.lookup_symbol_in_current_scope(name) {
                let prev = &self.decls[existing.0 as usize];
                // `typedef int T; typedef int T;` with the same type
                // merges; anything else in one scope redeclares.
                if binding.kind == BindingKind::Typedef && prev.kind == BindingKind::Typedef && prev.linkage == binding.linkage && types_compatible(&prev.ty, &binding.ty) {
                    return Ok(existing);
                }
                return Err(CompileError::new("conflicting redeclaration", binding.span).with_secondary(prev.span, "previously declared here"));
            }
            let id = DeclId(self.decls.len() as u32);
            self.decls.push(binding);
            self.scopes.current_mut().ordinary.insert(name, id);
            return Ok(id);
        }
        // Has linkage: a same-scope `NoLinkage` entry (a local or a
        // `typedef`) conflicts — gcc's "extern declaration follows
        // declaration with no linkage" and "redeclared as different
        // kind of symbol".
        if let Some(current) = self.lookup_symbol_in_current_scope(name)
            && self.decls[current.0 as usize].linkage == Linkage::NoLinkage {
                return Err(CompileError::new("conflicting redeclaration", binding.span).with_secondary(self.decls[current.0 as usize].span, "previously declared here"));
            }
        if let Some(existing) = self.lookup_visible_with_linkage(name) {
            let prev = &self.decls[existing.0 as usize];
            if prev.kind != binding.kind {
                return Err(CompileError::new("conflicting redeclaration", binding.span).with_secondary(prev.span, "previously declared here"));
            }
            if prev.linkage != binding.linkage {
                return Err(CompileError::new("conflicting redeclaration", binding.span).with_secondary(prev.span, "previously declared here"));
            }
            if prev.tls != binding.tls {
                return Err(CompileError::new("conflicting thread-local specifier", binding.span).with_secondary(prev.span, "previously declared here"));
            }
            if !types_compatible(&prev.ty, &binding.ty) {
                return Err(CompileError::new("conflicting redeclaration", binding.span).with_secondary(prev.span, "previously declared here"));
            }
            if prev.defined && binding.defined {
                if binding.kind == BindingKind::Func {
                    return Err(CompileError::new("redefinition of function", binding.span).with_secondary(prev.span, "previous definition here"));
                } else {
                    return Err(CompileError::new("redefinition", binding.span).with_secondary(prev.span, "previous definition here"));
                }
            }
            let composite = composite_type(prev.ty.clone(), binding.ty.clone());
            let merged_defined = prev.defined || binding.defined;
            let merged_tentative = !merged_defined && (prev.tentative || binding.tentative);
            let stored = &mut self.decls[existing.0 as usize];
            stored.ty = composite;
            stored.defined = merged_defined;
            stored.tentative = merged_tentative;
            // Constructor/destructor attributes ride along: set from
            // any declaration or the definition, first wins (the
            // extern prototype usually carries it).
            stored.constructor = stored.constructor.or(binding.constructor);
            stored.destructor = stored.destructor.or(binding.destructor);
            // Weakness is sticky the other way: any declaration marks it.
            stored.weak |= binding.weak;
            // `inline` merges the same way: any declaration marks it.
            stored.inline |= binding.inline;
            // An alias target may arrive with any declaration; first wins.
            stored.alias = stored.alias.or(binding.alias);
            // `naked` merges like `inline`: any declaration marks it.
            stored.naked |= binding.naked;
            // `noinline` merges the same way.
            stored.noinline |= binding.noinline;
            // `async` merges the same way: any declaration marks it.
            stored.is_async |= binding.is_async;
            // Alignment attributes are sticky the strictest way.
            stored.align = match (stored.align, binding.align) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            };
            // A section may arrive with any declaration; first wins.
            stored.section = stored.section.or(binding.section);
            // Same for an asm label.
            stored.asm_label = stored.asm_label.or(binding.asm_label);
            // The declaration still introduces the name in the current
            // scope: a block-scope `extern int v;` names the file-scope
            // object for the rest of the block, past a shadowing local
            // (C17 §6.2.2p4).
            self.scopes.current_mut().ordinary.insert(name, existing);
            return Ok(existing);
        }
        let id = DeclId(self.decls.len() as u32);
        self.decls.push(binding);
        self.scopes.current_mut().ordinary.insert(name, id);
        Ok(id)
    }
    // Mark an existing binding as defined, clearing tentative: a
    // definition overrides prior tentative declarations. The parser sets
    // both flags itself in `declare` so merges see them together, so
    // this wrapper has no in-tree caller outside tests.
    #[allow(dead_code)]
    pub fn define(&mut self, id: DeclId) {
        self.decls[id.0 as usize].defined = true;
        self.decls[id.0 as usize].tentative = false;
    }
    /// Record that `id`'s address was taken (`&id`, `&s.field`,
    /// `&a[i]`, `&func`, ...). Takes `&self` so checking can call it
    /// without threading `&mut`.
    pub fn mark_address_taken(&self, id: DeclId) {
        self.address_taken.borrow_mut().insert(id);
    }
    /// Whether `id`'s address was ever taken (see `address_taken`).
    pub fn is_address_taken(&self, id: DeclId) -> bool {
        self.address_taken.borrow().contains(&id)
    }
    /// Point VLA bound expressions at a replacement binding: rewrite
    /// every `Ident(old)` to `Ident(new)` throughout `vla_bounds`.
    ///
    /// A definition re-declares its parameters in function scope, but
    /// later parameters' bounds were checked in prototype scope
    /// against the prototype bindings (`void f(int n, int a[m][n])`).
    /// Without the rewrite the backend would lower a dead `DeclId`.
    /// `DeclId`s are unique per parse, so only this declarator's
    /// bounds can mention `old`.
    /// Point VLA bound expressions at replacement bindings in one
    /// simultaneous pass: `map` rewrites every `Ident(from)` to
    /// `Ident(to)` throughout `vla_bounds`. Simultaneous matters —
    /// sequential rewrites cascade when a replacement id equals a
    /// later source id (K&R parameters reuse truncated temporaries).
    pub fn remap_vla_idents(&mut self, map: &HashMap<DeclId, DeclId>) {
        if map.is_empty() {
            return;
        }
        let mut bounds = Vec::with_capacity(self.vla_bounds.len());
        for bound in &self.vla_bounds {
            bounds.push(remap_expr_ident(bound, self.arena, &mut self.inits, map));
        }
        self.vla_bounds = bounds;
    }
    /// Declare a compiler builtin (GCC's `__builtin_va_start` and
    /// friends) at file scope, no matter where the use sits: builtins
    /// are translation-unit entities, so a use inside a function body
    /// must not leave a block-scoped binding behind. Redeclaring
    /// reuses the existing entity.
    pub fn declare_global(&mut self, name: SymbolU32, binding: Binding) -> Result<DeclId, CompileError> {
        if let Some(existing) = self.scopes.global.ordinary.get(&name).copied() {
            return Ok(existing);
        }
        let id = DeclId(self.decls.len() as u32);
        self.decls.push(binding);
        self.scopes.global.ordinary.insert(name, id);
        Ok(id)
    }
    pub fn lookup_symbol_in_current_scope(&self, sym: SymbolU32) -> Option<DeclId> {
        self.scopes.current().ordinary.get(&sym).copied()
    }
    pub fn lookup_symbol(&self, sym: SymbolU32) -> Option<DeclId> {
        self.scopes.iter().find_map(|scope| scope.ordinary.get(&sym).copied())
    }
    /// Ordinary identifiers visible from the current scope, innermost
    /// first. May repeat a spelling when an inner scope shadows an
    /// outer one; used only for "did you mean" suggestions.
    pub fn visible_symbols(&self) -> impl Iterator<Item = SymbolU32> + '_ {
        self.scopes.iter().flat_map(|scope| scope.ordinary.keys().copied())
    }
    /// Innermost visible binding *with* linkage (C17 §6.2.2p4's "prior
    /// declaration ... visible"): skipped `NoLinkage` locals do not
    /// count, so a block-scope `extern` finds the file-scope entity
    /// past a shadowing local.
    pub fn lookup_visible_with_linkage(&self, sym: SymbolU32) -> Option<DeclId> {
        for scope in self.scopes.iter() {
            if let Some(&id) = scope.ordinary.get(&sym)
                && self.decls[id.0 as usize].linkage != Linkage::NoLinkage {
                    return Some(id);
                }
        }
        None
    }
    /// Prior linkage visible for an `extern` (or linkage-adopting)
    /// declaration: the linkage it inherits (C17 §6.2.2p4), or `None`
    /// when nothing with linkage is visible.
    pub fn visible_linkage(&self, sym: SymbolU32) -> Option<Linkage> {
        self.lookup_visible_with_linkage(sym).map(|id| self.decls[id.0 as usize].linkage)
    }

    // ---- struct/union tags ----

    // Look a tag up in the current scope only.
    pub fn lookup_tag(&self, tag: SymbolU32) -> Option<StructId> {
        self.scopes.current().tags.get(&tag).copied()
    }

    // A specifier *without* a body mentions `tag` (C17 §6.7.2.3): reuse the
    // tag visible in the innermost scope that has one — a block-scope
    // `struct S *p;` refers to the file-scope `struct S` — and only if none
    // is visible, implicitly declare an incomplete type in the current
    // scope.
    pub fn reference_tag(&mut self, tag: SymbolU32, is_union: bool, span: Span) -> Result<StructId, CompileError> {
        if let Some(id) = self.scopes.iter().find_map(|scope| scope.tags.get(&tag).copied()) {
            self.check_tag_kind(id, is_union, span)?;
            return Ok(id);
        }
        // No struct/union tag in sight: an enum tag in this scope would
        // share the namespace and conflicts.
        self.check_no_enum_tag(tag, span)?;
        Ok(self.declare_tag(tag, is_union, span))
    }

    // A *definition* (`tag { ... }`): work in the current scope so an inner
    // definition can shadow an outer one, and insert the tag before the body
    // parses so `struct Node { struct Node *next; };` sees itself. A tag
    // that is already complete here is a redefinition (C17 §6.7.2.3).
    pub fn define_tag(&mut self, tag: SymbolU32, is_union: bool, span: Span) -> Result<StructId, CompileError> {
        if let Some(id) = self.lookup_tag(tag) {
            self.check_tag_kind(id, is_union, span)?;
            if self.structs[id.0 as usize].complete {
                return Err(CompileError::new("redefinition of struct/union tag", span).with_secondary(self.structs[id.0 as usize].span, "previous definition here"));
            }
            // Point the tag at its definition site (it may have been
            // forward-declared elsewhere first).
            self.structs[id.0 as usize].span = span;
            return Ok(id);
        }
        self.check_no_enum_tag(tag, span)?;
        Ok(self.declare_tag(tag, is_union, span))
    }

    // Insert an incomplete tag into the current scope.
    fn declare_tag(&mut self, tag: SymbolU32, is_union: bool, span: Span) -> StructId {
        let id = StructId(self.structs.len() as u32);
        self.structs.push(StructDef {
            tag: Some(tag),
            is_union,
            members: vec![],
            complete: false,
            packed: false,
            align_override: None,
            layout_cache: RefCell::new(None),
            span,
        });
        self.scopes.current_mut().tags.insert(tag, id);
        id
    }

    // Anonymous structs never enter a scope: each occurrence is a distinct
    // type (C17 §6.7.2.3).
    pub fn new_anonymous_struct(&mut self, is_union: bool, span: Span) -> StructId {
        let id = StructId(self.structs.len() as u32);
        self.structs.push(StructDef {
            tag: None,
            is_union,
            members: vec![],
            complete: false,
            packed: false,
            align_override: None,
            layout_cache: RefCell::new(None),
            span,
        });
        id
    }

    // Attach the parsed member list, completing the type. Completing an
    // already-complete tag again is a redefinition (C17 §6.7.2.3).
    pub fn complete_tag(&mut self, id: StructId, members: Vec<Member>) -> Result<(), CompileError> {
        let def = &mut self.structs[id.0 as usize];
        if def.complete {
            return Err(CompileError::new("redefinition of struct/union tag", def.span));
        }
        def.members = members;
        def.complete = true;
        def.layout_cache = RefCell::new(None);
        Ok(())
    }

    /// Drop a definition's memoized layout after `packed` /
    /// `align_override` change post-completion (see the parser's
    /// struct specifier): the cache must never predate the effects.
    pub fn invalidate_layout(&mut self, id: StructId) {
        self.structs[id.0 as usize].layout_cache = RefCell::new(None);
    }

    fn check_tag_kind(&self, id: StructId, is_union: bool, span: Span) -> Result<(), CompileError> {
        if self.structs[id.0 as usize].is_union != is_union {
            return Err(CompileError::new("conflicting redeclaration", span).with_secondary(self.structs[id.0 as usize].span, "previously declared here"));
        }
        Ok(())
    }

    // ---- enum tags (sharing the struct/union tag namespace) ----

    // Look an enum tag up in the current scope only.
    pub fn lookup_enum_tag(&self, tag: SymbolU32) -> Option<EnumId> {
        self.scopes.current().enum_tags.get(&tag).copied()
    }

    // A use of `enum tag` without a body: reuse a visible enum tag, else
    // implicitly declare an incomplete one here (erroring when a
    // struct/union tag owns the spelling in this scope).
    pub fn reference_enum_tag(&mut self, tag: SymbolU32, span: Span) -> Result<EnumId, CompileError> {
        if let Some(id) = self.scopes.iter().find_map(|scope| scope.enum_tags.get(&tag).copied()) {
            return Ok(id);
        }
        self.check_no_struct_tag(tag, span)?;
        Ok(self.declare_enum_tag(tag, span))
    }

    // An `enum tag { ... }` definition: like struct definitions, it works
    // in the current scope (so an inner definition shadows an outer
    // tag) and re-completing a complete tag here is a redefinition.
    pub fn define_enum_tag(&mut self, tag: SymbolU32, span: Span) -> Result<EnumId, CompileError> {
        if let Some(id) = self.lookup_enum_tag(tag) {
            if self.enums[id.0 as usize].complete {
                return Err(CompileError::new("redefinition of enum tag", span).with_secondary(self.enums[id.0 as usize].span, "previous definition here"));
            }
            self.enums[id.0 as usize].span = span;
            return Ok(id);
        }
        self.check_no_struct_tag(tag, span)?;
        Ok(self.declare_enum_tag(tag, span))
    }

    // Insert an incomplete enum tag into the current scope.
    fn declare_enum_tag(&mut self, tag: SymbolU32, span: Span) -> EnumId {
        let id = EnumId(self.enums.len() as u32);
        self.enums.push(EnumDef {
            tag: Some(tag),
            enumerators: vec![],
            complete: false,
            // Unknown until the body parses; plain `int` is right
            // for every enum without out-of-range sentinels.
            repr: Ty::Int,
            fixed: None,
            span,
        });
        self.scopes.current_mut().enum_tags.insert(tag, id);
        id
    }

    // Anonymous enums never enter a scope: each occurrence is a distinct
    // type (C17 §6.7.2.3).
    pub fn new_anonymous_enum(&mut self, span: Span) -> EnumId {
        let id = EnumId(self.enums.len() as u32);
        self.enums.push(EnumDef {
            tag: None,
            enumerators: vec![],
            complete: false,
            repr: Ty::Int,
            fixed: None,
            span,
        });
        id
    }

    /// Record or check a C23 fixed underlying type on a (possibly still
    /// incomplete) enum. Redeclarations must repeat the same type, and
    /// a tag once declared with one may not shed it.
    pub fn set_enum_fixed(&mut self, id: EnumId, fixed: Option<Ty>, span: Span) -> Result<(), CompileError> {
        let def = &mut self.enums[id.0 as usize];
        match (&def.fixed, fixed) {
            (Some(previous), Some(new)) if *previous != new => {
                Err(CompileError::new("enumeration redeclared with a different underlying type", span).with_secondary(def.span, "previous declaration here"))
            }
            (Some(_), None) => Err(CompileError::new("enumeration previously declared with fixed underlying type", span).with_secondary(def.span, "previous declaration here")),
            (None, Some(_)) if def.complete => {
                Err(CompileError::new("enumeration previously declared with nonfixed underlying type", span).with_secondary(def.span, "previous declaration here"))
            }
            (_, fixed @ Some(_)) => {
                def.fixed = fixed;
                Ok(())
            }
            (None, None) => Ok(()),
        }
    }

    // Attach the parsed enumerator list, completing the type.
    // Completing an already-complete tag again is a redefinition.
    pub fn complete_enum_tag(&mut self, id: EnumId, enumerators: Vec<Enumerator>, repr: Ty, fixed: Option<Ty>) -> Result<(), CompileError> {
        self.set_enum_fixed(id, fixed, self.enums[id.0 as usize].span)?;
        let def = &mut self.enums[id.0 as usize];
        if def.complete {
            return Err(CompileError::new("redefinition of enum tag", def.span));
        }
        def.enumerators = enumerators;
        def.repr = repr;
        def.complete = true;
        Ok(())
    }

    // A struct/union tag in this scope blocks an enum of the same
    // spelling (and vice versa): one tag namespace (C17 §6.2.3).
    fn check_no_struct_tag(&self, tag: SymbolU32, span: Span) -> Result<(), CompileError> {
        if let Some(id) = self.scopes.current().tags.get(&tag) {
            return Err(CompileError::new("conflicting redeclaration", span).with_secondary(self.structs[id.0 as usize].span, "previously declared here"));
        }
        Ok(())
    }

    fn check_no_enum_tag(&self, tag: SymbolU32, span: Span) -> Result<(), CompileError> {
        if let Some(id) = self.scopes.current().enum_tags.get(&tag) {
            return Err(CompileError::new("conflicting redeclaration", span).with_secondary(self.enums[id.0 as usize].span, "previously declared here"));
        }
        Ok(())
    }
}

// ---- type compatibility (C17 §6.2.7, §6.7.6.3p15) ----

// Whether two declared types may be the same entity redeclared: equal
// primitives/pointers/structs, and for functions compatible returns
// with either side unprototyped (`f()`, old-style) matching any
// prototype, or two prototypes agreeing on arity, parameter types,
// and variadicity. Qualification must match exactly (`const int x;`
// against `int x;` conflicts, like GCC) — except on function
// parameters, where top-level qualifiers are dropped (see below);
// composite declarations merge it (see `composite_type`).
pub fn types_compatible(a: &Ty, b: &Ty) -> bool {
    match (a, b) {
        (Ty::Qualified { quals: qa, inner: ia }, Ty::Qualified { quals: qb, inner: ib }) => qa == qb && types_compatible(ia, ib),
        (
            Ty::Func {
                ret: ra,
                params: pa,
                variadic: va,
                prototyped: pra,
            },
            Ty::Func {
                ret: rb,
                params: pb,
                variadic: vb,
                prototyped: prb,
            },
        ) => {
            if !types_compatible(ra, rb) {
                return false;
            }
            // Unprototyped (`f()`, `f(a, b)`) supplies no parameter
            // information: compatible with any prototype of the same
            // return type (C17 §6.7.6.3p15).
            if !pra || !prb {
                return true;
            }
            if va != vb || pa.len() != pb.len() {
                return false;
            }
            // Each parameter compares with its top-level qualifiers
            // dropped (see `unqualified_param_ty`); tcc.h redeclares
            // libc functions this way, like GCC.
            pa.iter().zip(pb.iter()).all(|(x, y)| types_compatible(&unqualified_param_ty(x), &unqualified_param_ty(y)))
        }
        (
            Ty::Pointer {
                qualifiers: qa,
                size: sa,
                pointee: pa,
            },
            Ty::Pointer {
                qualifiers: qb,
                size: sb,
                pointee: pb,
            },
        ) => qa == qb && sa == sb && types_compatible(pa, pb),
        (Ty::Array { elem: ea, quals: qa, bound: ba }, Ty::Array { elem: eb, quals: qb, bound: bb }) => qa == qb && types_compatible(ea, eb) && bounds_compatible(ba, bb),
        _ => a == b,
    }
}
/// GCC's `__builtin_types_compatible_p` relation: [`types_compatible`]
/// with top-level qualifiers ignored and array bounds disregarded
/// (`int` ≡ `const int`, `int[5]` ≡ `int[]`, `int[5]` ≡ `int[3]`).
pub fn types_compatible_p(a: &Ty, b: &Ty) -> bool {
    match (a.unqualified(), b.unqualified()) {
        (Ty::Array { elem: ea, quals: qa, .. }, Ty::Array { elem: eb, quals: qb, .. }) => qa == qb && types_compatible_p(ea, eb),
        (a, b) => types_compatible(a, b),
    }
}
/// A parameter type with its top-level qualifiers dropped, for
/// redeclaration compatibility (C17 §6.7.6.3p15): `char **` and
/// `char **__restrict` parameters are compatible, like GCC. Only the
/// outermost level is erased — pointee qualification still counts
/// (`const char *` against `char *` still conflicts).
fn unqualified_param_ty(ty: &Ty) -> Ty {
    match ty {
        Ty::Qualified { inner, .. } => inner.unqualified().clone(),
        Ty::Pointer { pointee, size, .. } => Ty::Pointer {
            qualifiers: TypeQualifiers::empty(),
            size: *size,
            pointee: pointee.clone(),
        },
        _ => ty.clone(),
    }
}
/// Whether two array bounds may describe the same object (C17 §6.7.6.2p6
/// read loosely): an absent (`[]`) or `[*]` bound combines with anything,
/// while two present expressions must spell the same bound. (No constant
/// folding: `a[1+1]` against `a[2]` counts as incompatible.)
fn bounds_compatible(a: &Option<ArrayBound>, b: &Option<ArrayBound>) -> bool {
    match (a, b) {
        (None, _) | (_, None) => true,
        (Some(ArrayBound::Star), _) | (_, Some(ArrayBound::Star)) => true,
        (Some(ArrayBound::Fixed(x)), Some(ArrayBound::Fixed(y))) => x == y,
        // A VLA matches any VLA, and (per C17 §6.7.6.2p6) any fixed
        // bound: `int[n]` and `int[4]` are compatible, with undefined
        // behavior only if the extents differ at runtime. This is what
        // lets a fixed array reach a VLA parameter (`f(3, 4, b)` for
        // `f(int m, int n, int a[m][n])`).
        (Some(ArrayBound::Vla(_)), _) | (_, Some(ArrayBound::Vla(_))) => true,
    }
}

// The composite of two compatible declarations (C17 §6.2.7p4): the
// prototyped side wins when one function type is unprototyped, so
// `int f(); int f(int);` leaves `int f(int);` behind.
pub fn composite_type(a: Ty, b: Ty) -> Ty {
    match (a, b) {
        (
            Ty::Func {
                ret: ra,
                params: pa,
                variadic: va,
                prototyped: pra,
            },
            Ty::Func {
                ret: rb,
                params: pb,
                variadic: vb,
                prototyped: prb,
            },
        ) => {
            let ret = Box::new(composite_type(*ra, *rb));
            match (pra, prb) {
                (false, false) => Ty::Func {
                    ret,
                    params: vec![],
                    variadic: false,
                    prototyped: false,
                },
                (false, true) => Ty::Func {
                    ret,
                    params: pb,
                    variadic: vb,
                    prototyped: true,
                },
                (true, false) => Ty::Func {
                    ret,
                    params: pa,
                    variadic: va,
                    prototyped: true,
                },
                (true, true) => {
                    let params = pa.into_iter().zip(pb).map(|(x, y)| composite_type(x, y)).collect();
                    Ty::Func {
                        ret,
                        params,
                        variadic: va,
                        prototyped: true,
                    }
                }
            }
        }
        (
            Ty::Pointer {
                qualifiers: qa,
                size,
                pointee: pa,
            },
            Ty::Pointer { pointee: pb, .. },
        ) => Ty::Pointer {
            qualifiers: qa,
            size,
            pointee: Box::new(composite_type(*pa, *pb)),
        },
        (Ty::Qualified { quals: qa, inner: ia }, Ty::Qualified { quals: qb, inner: ib }) => {
            // Compatible qualified types (checked by the caller) merge
            // qualifier sets — equal here, kept as-is — and compose
            // the inner types.
            let mut quals = qa;
            quals |= qb;
            Ty::qualified(quals, composite_type(*ia, *ib))
        }
        (Ty::Array { elem: ea, quals: qa, bound: ba }, Ty::Array { elem: eb, bound: bb, .. }) => Ty::Array {
            elem: Box::new(composite_type(*ea, *eb)),
            quals: qa,
            bound: composite_bound(ba, bb),
        },
        (a, _) => a,
    }
}

/// The composite of two compatible bounds: a specified bound wins over
/// `[]`, and over `[*]`; two fixed bounds are already known equal.
fn composite_bound(a: Option<ArrayBound>, b: Option<ArrayBound>) -> Option<ArrayBound> {
    match (a, b) {
        (None, b) => b,
        (a, None) => a,
        (Some(ArrayBound::Star), b) => b,
        (a, Some(ArrayBound::Star)) => a,
        (Some(_), b @ Some(_)) => b,
    }
}

/// Decoded contents of concatenated narrow string literals: prefix
/// and surrounding quotes stripped, escapes processed (`"\n"`
/// becomes a real newline), adjacent literals joined.
///
/// Narrow prefixes only (`""`/`u8`); wide prefixes (`L`/`u`/`U`)
/// and malformed spellings yield `None`, like `eval::string_bytes`.
///
/// NOTE: this is lossy for high-byte escapes. C's `"\xff"` is the
/// single byte `0xFF`, which is not valid UTF-8 on its own, so no
/// Rust `String` can hold it — here it decodes to U+00FF (`'ÿ'`).
/// Anything needing exact C bytes (data emission, `sizeof` math)
/// must use `eval::string_bytes` instead.
pub fn string_contents(parts: &[StringLiteral], interner: &Interner) -> Option<String> {
    let mut out = String::new();
    for part in parts {
        match part.prefix {
            StringPrefix::None | StringPrefix::U8 => {}
            StringPrefix::L | StringPrefix::SmallU | StringPrefix::BigU => return None,
        }
        let spelling = interner.resolve(part.spelling)?;
        let prefix_len = part.prefix.as_str().len();
        let body = spelling.get(prefix_len + 1..spelling.len().checked_sub(1)?)?;
        // Byte-walk like `eval::string_bytes` so escape boundaries
        // match, but push decoded `char`s instead of bytes.
        let mut i = 0;
        while i < body.len() {
            if body.as_bytes()[i] != b'\\' {
                let c = body[i..].chars().next()?;
                out.push(c);
                i += c.len_utf8();
                continue;
            }
            let tail: Vec<char> = body[i + 1..].chars().collect();
            if tail.is_empty() {
                return None;
            }
            let (v, used) = escape_value(&tail);
            // A UCN names a code point directly; every other escape
            // truncates mod 256 (like GCC), so any value is a scalar.
            let ch = if crate::eval::is_ucn(&tail) { char::from_u32(v)? } else { char::from_u32(v & 0xFF)? };
            out.push(ch);
            i += 1 + tail[..used].iter().map(|c| c.len_utf8()).sum::<usize>();
        }
    }
    Some(out)
}

/// Rewrite one `Ident(old)` to `Ident(new)` across a checked
/// expression (see `Sema::remap_vla_idents`). Total over
/// `CheckedExprKind`: leaves hold no bindings, and types carry none
/// (VLA bounds travel by index in `Sema::vla_bounds`, which the caller
/// already walks). Declarations nested in GNU statement expressions
/// keep their bindings; only their initializers' expression nodes are
/// rewritten, via `inits`.
fn remap_expr_ident<'a>(expr: &CheckedExpr<'a>, arena: &'a bumpalo::Bump, inits: &mut HashMap<DeclId, Initializer<'a>>, map: &HashMap<DeclId, DeclId>) -> CheckedExpr<'a> {
    let kind = match &expr.kind {
        CheckedExprKind::Ident(id) => CheckedExprKind::Ident(map.get(id).copied().unwrap_or(*id)),
        CheckedExprKind::Int(lit) => CheckedExprKind::Int(*lit),
        CheckedExprKind::Float(lit) => CheckedExprKind::Float(*lit),
        CheckedExprKind::Char(lit) => CheckedExprKind::Char(*lit),
        CheckedExprKind::Str(parts) => CheckedExprKind::Str(parts),
        CheckedExprKind::ConstantP(value) => CheckedExprKind::ConstantP(*value),
        CheckedExprKind::ClassifyType(code) => CheckedExprKind::ClassifyType(*code),
        CheckedExprKind::Offsetof { ty, offset } => CheckedExprKind::Offsetof { ty: ty.clone(), offset: *offset },
        CheckedExprKind::SizeofTy(ty) => CheckedExprKind::SizeofTy(ty.clone()),
        CheckedExprKind::CountofTy(ty) => CheckedExprKind::CountofTy(ty.clone()),
        CheckedExprKind::AlignofTy(ty) => CheckedExprKind::AlignofTy(ty.clone()),
        CheckedExprKind::StaticAssert => CheckedExprKind::StaticAssert,
        CheckedExprKind::LabelAddr(label) => CheckedExprKind::LabelAddr(*label),
        CheckedExprKind::FrameAddress { level, returns_address } => CheckedExprKind::FrameAddress {
            level: *level,
            returns_address: *returns_address,
        },
        CheckedExprKind::Unary(op, e) => CheckedExprKind::Unary(*op, arena.alloc(remap_expr_ident(e, arena, inits, map))),
        CheckedExprKind::Await(e) => CheckedExprKind::Await(arena.alloc(remap_expr_ident(e, arena, inits, map))),
        CheckedExprKind::Cast { ty, expr: e } => CheckedExprKind::Cast {
            ty: ty.clone(),
            expr: arena.alloc(remap_expr_ident(e, arena, inits, map)),
        },
        CheckedExprKind::Convert { expr: e, ty } => CheckedExprKind::Convert {
            expr: arena.alloc(remap_expr_ident(e, arena, inits, map)),
            ty: ty.clone(),
        },
        CheckedExprKind::SizeofExpr(e) => CheckedExprKind::SizeofExpr(arena.alloc(remap_expr_ident(e, arena, inits, map))),
        CheckedExprKind::CountofExpr(e) => CheckedExprKind::CountofExpr(arena.alloc(remap_expr_ident(e, arena, inits, map))),
        CheckedExprKind::AlignofExpr(e) => CheckedExprKind::AlignofExpr(arena.alloc(remap_expr_ident(e, arena, inits, map))),
        CheckedExprKind::VaArg { ap, ty } => CheckedExprKind::VaArg {
            ap: arena.alloc(remap_expr_ident(ap, arena, inits, map)),
            ty: ty.clone(),
        },
        CheckedExprKind::Member {
            base,
            path,
            arrow,
            offset,
            bitfield,
            decl_ty,
        } => CheckedExprKind::Member {
            base: arena.alloc(remap_expr_ident(base, arena, inits, map)),
            path,
            arrow: *arrow,
            offset: *offset,
            bitfield: *bitfield,
            decl_ty: decl_ty.clone(),
        },
        CheckedExprKind::Binary(op, l, r) => CheckedExprKind::Binary(
            *op,
            arena.alloc(remap_expr_ident(l, arena, inits, map)),
            arena.alloc(remap_expr_ident(r, arena, inits, map)),
        ),
        CheckedExprKind::Assign(op, l, r) => CheckedExprKind::Assign(
            *op,
            arena.alloc(remap_expr_ident(l, arena, inits, map)),
            arena.alloc(remap_expr_ident(r, arena, inits, map)),
        ),
        CheckedExprKind::Subscript { base, index } => CheckedExprKind::Subscript {
            base: arena.alloc(remap_expr_ident(base, arena, inits, map)),
            index: arena.alloc(remap_expr_ident(index, arena, inits, map)),
        },
        CheckedExprKind::CompoundAssign { op, lhs, rhs, op_ty, rhs_ty } => CheckedExprKind::CompoundAssign {
            op: *op,
            lhs: arena.alloc(remap_expr_ident(lhs, arena, inits, map)),
            rhs: arena.alloc(remap_expr_ident(rhs, arena, inits, map)),
            op_ty: op_ty.clone(),
            rhs_ty: rhs_ty.clone(),
        },
        CheckedExprKind::Ternary { cond, then, els } => CheckedExprKind::Ternary {
            cond: arena.alloc(remap_expr_ident(cond, arena, inits, map)),
            then: arena.alloc(remap_expr_ident(then, arena, inits, map)),
            els: arena.alloc(remap_expr_ident(els, arena, inits, map)),
        },
        CheckedExprKind::Elvis { cond, els } => CheckedExprKind::Elvis {
            cond: arena.alloc(remap_expr_ident(cond, arena, inits, map)),
            els: arena.alloc(remap_expr_ident(els, arena, inits, map)),
        },
        CheckedExprKind::Call { func, args } => CheckedExprKind::Call {
            func: arena.alloc(remap_expr_ident(func, arena, inits, map)),
            args: remap_expr_slice(args, arena, inits, map),
        },
        CheckedExprKind::Atomic { op, args } => CheckedExprKind::Atomic {
            op: *op,
            args: remap_expr_slice(args, arena, inits, map),
        },
        CheckedExprKind::Overflow { op, lhs, rhs, res, res_ty } => CheckedExprKind::Overflow {
            op: *op,
            lhs: arena.alloc(remap_expr_ident(lhs, arena, inits, map)),
            rhs: arena.alloc(remap_expr_ident(rhs, arena, inits, map)),
            res: arena.alloc(remap_expr_ident(res, arena, inits, map)),
            res_ty: res_ty.clone(),
        },
        CheckedExprKind::Alloca { size } => CheckedExprKind::Alloca {
            size: arena.alloc(remap_expr_ident(size, arena, inits, map)),
        },
        CheckedExprKind::Clz { ctz, arg, fallback } => CheckedExprKind::Clz {
            ctz: *ctz,
            arg: arena.alloc(remap_expr_ident(arg, arena, inits, map)),
            fallback: fallback.map(|e| &*arena.alloc(remap_expr_ident(e, arena, inits, map))),
        },
        CheckedExprKind::Rotate { left, arg, count, bits } => CheckedExprKind::Rotate {
            left: *left,
            arg: arena.alloc(remap_expr_ident(arg, arena, inits, map)),
            count: arena.alloc(remap_expr_ident(count, arena, inits, map)),
            bits: *bits,
        },
        CheckedExprKind::FpClass { op, arg } => CheckedExprKind::FpClass {
            op: *op,
            arg: arena.alloc(remap_expr_ident(arg, arena, inits, map)),
        },
        CheckedExprKind::FpClassify { codes, arg } => CheckedExprKind::FpClassify {
            codes: core::array::from_fn(|i| &*arena.alloc(remap_expr_ident(codes[i], arena, inits, map))),
            arg: arena.alloc(remap_expr_ident(arg, arena, inits, map)),
        },
        CheckedExprKind::Generic { ctrl, index, assocs } => CheckedExprKind::Generic {
            ctrl: arena.alloc(remap_expr_ident(ctrl, arena, inits, map)),
            index: *index,
            assocs: remap_assocs(assocs, arena, inits, map),
        },
        CheckedExprKind::GenericTy { ty, index, assocs } => CheckedExprKind::GenericTy {
            ty: ty.clone(),
            index: *index,
            assocs: remap_assocs(assocs, arena, inits, map),
        },
        CheckedExprKind::StmtExpr { items } => CheckedExprKind::StmtExpr {
            items: remap_block_items(items, arena, inits, map),
        },
        CheckedExprKind::CompoundLiteral { ty, init, storage } => CheckedExprKind::CompoundLiteral {
            ty: ty.clone(),
            init: arena.alloc(remap_init_ident(init, arena, inits, map)),
            storage: *storage,
        },
    };
    CheckedExpr {
        ty: expr.ty.clone(),
        span: expr.span,
        kind,
    }
}

fn remap_expr_slice<'a>(
    exprs: &'a [CheckedExpr<'a>],
    arena: &'a bumpalo::Bump,
    inits: &mut HashMap<DeclId, Initializer<'a>>,
    map: &HashMap<DeclId, DeclId>,
) -> &'a [CheckedExpr<'a>] {
    let mut out = bumpalo::collections::Vec::new_in(arena);
    for expr in exprs {
        out.push(remap_expr_ident(expr, arena, inits, map));
    }
    out.into_bump_slice()
}

fn remap_assocs<'a>(
    assocs: &'a [(Option<Ty>, CheckedExpr<'a>)],
    arena: &'a bumpalo::Bump,
    inits: &mut HashMap<DeclId, Initializer<'a>>,
    map: &HashMap<DeclId, DeclId>,
) -> &'a [(Option<Ty>, CheckedExpr<'a>)] {
    let mut out = bumpalo::collections::Vec::new_in(arena);
    for (ty, expr) in assocs {
        out.push((ty.clone(), remap_expr_ident(expr, arena, inits, map)));
    }
    out.into_bump_slice()
}

fn remap_inits<'a>(ids: &[DeclId], arena: &'a bumpalo::Bump, inits: &mut HashMap<DeclId, Initializer<'a>>, map: &HashMap<DeclId, DeclId>) {
    for id in ids {
        if let Some(init) = inits.remove(id) {
            let init = remap_init_ident(&init, arena, inits, map);
            inits.insert(*id, init);
        }
    }
}

/// Rewrite across an initializer: the expression form plus every
/// nested list item, drilling through designators into their index
/// expressions.
fn remap_init_ident<'a>(init: &Initializer<'a>, arena: &'a bumpalo::Bump, inits: &mut HashMap<DeclId, Initializer<'a>>, map: &HashMap<DeclId, DeclId>) -> Initializer<'a> {
    match init {
        Initializer::Expr(expr) => Initializer::Expr(remap_expr_ident(expr, arena, inits, map)),
        Initializer::List { items, span } => {
            let mut out = bumpalo::collections::Vec::new_in(arena);
            for item in *items {
                let mut designators = bumpalo::collections::Vec::new_in(arena);
                for designator in item.designators {
                    designators.push(match designator {
                        Designator::Field { name, span } => Designator::Field { name: *name, span: *span },
                        Designator::AnonField { index, span } => Designator::AnonField { index: *index, span: *span },
                        Designator::Index { index, span } => Designator::Index {
                            index: remap_expr_ident(index, arena, inits, map),
                            span: *span,
                        },
                    });
                }
                out.push(InitItem {
                    designators: designators.into_bump_slice(),
                    init: arena.alloc(remap_init_ident(item.init, arena, inits, map)),
                    span: item.span,
                });
            }
            Initializer::List {
                items: out.into_bump_slice(),
                span: *span,
            }
        }
    }
}

/// Rewrite across block items: statements plus the initializers of
/// declarations nested in GNU statement expressions.
fn remap_block_items<'a>(items: &'a [BlockItem<'a>], arena: &'a bumpalo::Bump, inits: &mut HashMap<DeclId, Initializer<'a>>, map: &HashMap<DeclId, DeclId>) -> &'a [BlockItem<'a>] {
    let mut out = bumpalo::collections::Vec::new_in(arena);
    for item in items {
        out.push(match item {
            BlockItem::Decl { ids, span } => {
                remap_inits(ids, arena, inits, map);
                BlockItem::Decl { ids, span: *span }
            }
            BlockItem::Stmt(stmt) => BlockItem::Stmt(remap_stmt_ident(stmt, arena, inits, map)),
        });
    }
    out.into_bump_slice()
}

/// Rewrite across a statement: every expression position plus `for`
/// initializers.
fn remap_stmt_ident<'a>(stmt: &Stmt<'a>, arena: &'a bumpalo::Bump, inits: &mut HashMap<DeclId, Initializer<'a>>, map: &HashMap<DeclId, DeclId>) -> Stmt<'a> {
    match stmt {
        Stmt::Compound(compound) => Stmt::Compound(Compound {
            span: compound.span,
            items: remap_block_items(compound.items, arena, inits, map),
        }),
        Stmt::Decl { ids, span } => {
            remap_inits(ids, arena, inits, map);
            Stmt::Decl { ids, span: *span }
        }
        Stmt::Empty { span } => Stmt::Empty { span: *span },
        Stmt::Break { label, span } => Stmt::Break { label: *label, span: *span },
        Stmt::Continue { label, span } => Stmt::Continue { label: *label, span: *span },
        Stmt::Goto { label, span } => Stmt::Goto { label: *label, span: *span },
        Stmt::GotoComputed { expr, span } => Stmt::GotoComputed {
            expr: remap_expr_ident(expr, arena, inits, map),
            span: *span,
        },
        Stmt::Expr { expr, span } => Stmt::Expr {
            expr: remap_expr_ident(expr, arena, inits, map),
            span: *span,
        },
        Stmt::Asm(asm) => {
            let mut remap_operands = |ops: &'a [AsmOperand<'a>]| {
                let mut out = bumpalo::collections::Vec::new_in(arena);
                for op in ops {
                    out.push(AsmOperand {
                        name: op.name,
                        constraint: op.constraint.clone(),
                        expr: remap_expr_ident(&op.expr, arena, inits, map),
                        span: op.span,
                    });
                }
                out.into_bump_slice()
            };
            Stmt::Asm(AsmStmt {
                template: asm.template.clone(),
                outputs: remap_operands(asm.outputs),
                inputs: remap_operands(asm.inputs),
                clobbers: asm.clobbers,
                labels: asm.labels,
                flags: asm.flags,
                span: asm.span,
            })
        }
        Stmt::Return { expr, span } => Stmt::Return {
            expr: expr.as_ref().map(|e| remap_expr_ident(e, arena, inits, map)),
            span: *span,
        },
        Stmt::If { cond, then, els, span } => Stmt::If {
            cond: remap_expr_ident(cond, arena, inits, map),
            then: arena.alloc(remap_stmt_ident(then, arena, inits, map)),
            els: els.map(|e| &*arena.alloc(remap_stmt_ident(e, arena, inits, map))),
            span: *span,
        },
        Stmt::While { cond, body, span } => Stmt::While {
            cond: remap_expr_ident(cond, arena, inits, map),
            body: arena.alloc(remap_stmt_ident(body, arena, inits, map)),
            span: *span,
        },
        Stmt::DoWhile { body, cond, span } => Stmt::DoWhile {
            body: arena.alloc(remap_stmt_ident(body, arena, inits, map)),
            cond: remap_expr_ident(cond, arena, inits, map),
            span: *span,
        },
        Stmt::For { init, cond, step, body, span } => {
            let init = match init {
                ForInit::Decl(ids) => {
                    remap_inits(ids, arena, inits, map);
                    ForInit::Decl(ids)
                }
                ForInit::Expr(expr) => ForInit::Expr(remap_expr_ident(expr, arena, inits, map)),
                ForInit::Empty => ForInit::Empty,
            };
            Stmt::For {
                init,
                cond: cond.as_ref().map(|e| remap_expr_ident(e, arena, inits, map)),
                step: step.as_ref().map(|e| remap_expr_ident(e, arena, inits, map)),
                body: arena.alloc(remap_stmt_ident(body, arena, inits, map)),
                span: *span,
            }
        }
        Stmt::Switch { cond, body, span } => Stmt::Switch {
            cond: remap_expr_ident(cond, arena, inits, map),
            body: arena.alloc(remap_stmt_ident(body, arena, inits, map)),
            span: *span,
        },
        Stmt::Label { name, names_loop, stmt, span } => Stmt::Label {
            name: *name,
            names_loop: *names_loop,
            stmt: arena.alloc(remap_stmt_ident(stmt, arena, inits, map)),
            span: *span,
        },
        Stmt::Case { expr, hi, stmt, span } => Stmt::Case {
            expr: expr.as_ref().map(|e| remap_expr_ident(e, arena, inits, map)),
            hi: hi.as_ref().map(|e| remap_expr_ident(e, arena, inits, map)),
            stmt: arena.alloc(remap_stmt_ident(stmt, arena, inits, map)),
            span: *span,
        },
    }
}
