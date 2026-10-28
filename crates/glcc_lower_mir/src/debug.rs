//! Frontend-to-MIR debug lowering.
//!
//! Turns C spans and types into the language-neutral records of
//! [`glcc_mir::debug`]: file table entries, per-instruction locations,
//! DWARF-shaped type descriptions, and variable/scope records for
//! functions and module-level objects. Layout facts (offsets, bit
//! fields, enum values) are resolved here, where `Sema` is at hand, so
//! the DWARF writer never needs the frontend.

use std::collections::HashMap;

use glcc_frontend as fe;
use glcc_frontend::span::HasSpan;
use glcc_mir as mir;

use crate::lower::Lowerer;

impl<'s, 'i> Lowerer<'s, 'i> {
    /// Source location of a span, or `None` for synthesized nodes and
    /// spans without a file. Lines are 1-based; columns are converted
    /// from the lexer's 0-based offsets to DWARF's 1-based convention.
    pub(crate) fn debug_location(&self, module: &mut mir::ModuleBuilder, span: fe::span::Span) -> Option<mir::Location> {
        if span.is_dummy() || span.start_ln == 0 {
            return None;
        }
        let path = self.interner.resolve(span.file?)?;
        if path.is_empty() {
            return None;
        }
        let file = module.intern_file(path);
        Some(mir::Location {
            file,
            line: self.map_orig_line(span.start_ln) as u32,
            column: span.start_col as u32 + 1,
        })
    }

    /// Buffer-relative line to original source line through the parser's
    /// linemarker table (`SourceFiles::orig_lines`), like the legacy
    /// backend's `map_span`.
    fn map_orig_line(&self, ln: u64) -> u64 {
        if ln == 0 {
            return 0;
        }
        let idx = (ln - 1) as usize;
        self.sema.orig_lines.get(idx).copied().unwrap_or(ln as usize) as u64
    }

    /// Both ends of a span as locations, for lexical block ranges.
    pub(crate) fn debug_span_range(&self, module: &mut mir::ModuleBuilder, span: fe::span::Span) -> Option<(mir::Location, mir::Location)> {
        let start = self.debug_location(module, span)?;
        if span.end_ln == 0 {
            return None;
        }
        let end = mir::Location {
            file: start.file,
            line: self.map_orig_line(span.end_ln) as u32,
            column: span.end_col as u32 + 1,
        };
        Some((start, end))
    }

    /// DWARF-shaped description of a frontend type, deduplicated across
    /// the module through `module.debug_types`. `None` for types with no
    /// useful description (`void`, incomplete structs without a
    /// definition, integer complex).
    pub(crate) fn lower_debug_ty(&self, module: &mut mir::ModuleBuilder, ty: &fe::parse::Ty) -> Option<mir::DebugTypeId> {
        let mut cache = self.debug_ty_cache.borrow_mut();
        lower_ty_debug(&mut cache, module, self.sema, self.interner, ty)
    }

    /// Debug record of one lowered function. `slots` maps local
    /// declarations to their spilled slot addresses.
    pub(crate) fn build_function_debug(
        &self,
        module: &mut mir::ModuleBuilder,
        slots: &HashMap<fe::sema::DeclId, mir::Value>,
        body: &fe::parse::FuncBody,
        id: fe::sema::DeclId,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
    ) -> mir::FunctionDebug {
        let binding = &self.sema.decls[id.0 as usize];
        let mut debug = mir::FunctionDebug {
            decl: self.debug_location(module, binding.span),
            ..Default::default()
        };
        for param_id in body.params {
            if let Some(var) = self.var_debug(module, slots, globals, *param_id) {
                debug.params.push(var);
            }
        }
        let mut root = ScopeBuild {
            span: body.block.span,
            vars: Vec::new(),
            children: Vec::new(),
        };
        collect_items(body.block.items, &mut root);
        let (vars, scopes) = self.scope_debug(module, slots, globals, &root);
        debug.vars = vars;
        debug.scopes = scopes;
        debug
    }

    /// Debug record of a file-scope or block-scope object.
    pub(crate) fn global_debug(&self, module: &mut mir::ModuleBuilder, decl_id: fe::sema::DeclId) -> mir::GlobalDebug {
        let binding = &self.sema.decls[decl_id.0 as usize];
        mir::GlobalDebug {
            name: binding.name.and_then(|s| self.interner.resolve(s)).map(str::to_string),
            decl: self.debug_location(module, binding.span),
            debug_ty: self.lower_debug_ty(module, &binding.ty),
        }
    }

    /// One variable or parameter record, or `None` when it has no
    /// storage to describe (functions, `extern` declarations,
    /// `va_list` aliases).
    fn var_debug(
        &self,
        module: &mut mir::ModuleBuilder,
        slots: &HashMap<fe::sema::DeclId, mir::Value>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        id: fe::sema::DeclId,
    ) -> Option<mir::VarDebug> {
        let binding = &self.sema.decls[id.0 as usize];
        if !matches!(binding.kind, fe::sema::BindingKind::Var) {
            return None;
        }
        if matches!(binding.ty.unqualified(), fe::parse::Ty::BuiltinVaList) {
            return None;
        }
        let home = if let Some(slot) = slots.get(&id) {
            mir::VarHome::Slot(*slot)
        } else {
            let global = globals.get(&id)?;
            mir::VarHome::Global(*global)
        };
        let name = binding.name.and_then(|s| self.interner.resolve(s)).map(str::to_string);
        let decl = self.debug_location(module, binding.span);
        let debug_ty = self.lower_debug_ty(module, &binding.ty);
        Some(mir::VarDebug { name, decl, debug_ty, home })
    }

    /// Variables and ranged child scopes of one lexical scope. Scopes
    /// whose span carries no source position (synthesized nodes) hoist
    /// their contents into the parent, like the DWARF writer does for
    /// scopes that generated no code.
    fn scope_debug(
        &self,
        module: &mut mir::ModuleBuilder,
        slots: &HashMap<fe::sema::DeclId, mir::Value>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        scope: &ScopeBuild,
    ) -> (Vec<mir::VarDebug>, Vec<mir::ScopeDebug>) {
        let mut vars: Vec<mir::VarDebug> = scope.vars.iter().filter_map(|id| self.var_debug(module, slots, globals, *id)).collect();
        let mut children = Vec::new();
        for child in &scope.children {
            let (cvars, cchildren) = self.scope_debug(module, slots, globals, child);
            match self.debug_span_range(module, child.span) {
                Some((start, end)) => children.push(mir::ScopeDebug {
                    start,
                    end,
                    vars: cvars,
                    children: cchildren,
                }),
                None => {
                    vars.extend(cvars);
                    children.extend(cchildren);
                }
            }
        }
        (vars, children)
    }
}

/// One lexical scope under construction. A port of the legacy
/// backend's walk: `for` initializers form their own scope spanning the
/// whole statement; statement expressions are walked but open no
/// scope.
struct ScopeBuild {
    span: fe::span::Span,
    vars: Vec<fe::sema::DeclId>,
    children: Vec<ScopeBuild>,
}

fn collect_items(items: &[fe::parse::BlockItem], scope: &mut ScopeBuild) {
    for item in items {
        match item {
            fe::parse::BlockItem::Decl { ids, .. } => scope.vars.extend(ids.iter().copied()),
            fe::parse::BlockItem::Stmt(stmt) => collect_stmt(stmt, scope),
        }
    }
}

fn collect_stmt(stmt: &fe::parse::Stmt, scope: &mut ScopeBuild) {
    match stmt {
        fe::parse::Stmt::Compound(compound) => {
            let mut child = ScopeBuild {
                span: compound.span,
                vars: Vec::new(),
                children: Vec::new(),
            };
            collect_items(compound.items, &mut child);
            scope.children.push(child);
        }
        fe::parse::Stmt::For { init, body, .. } => {
            let mut fscope = ScopeBuild {
                span: stmt.span(),
                vars: Vec::new(),
                children: Vec::new(),
            };
            if let fe::parse::ForInit::Decl(ids) = init {
                fscope.vars.extend(ids.iter().copied());
            }
            collect_stmt(body, &mut fscope);
            scope.children.push(fscope);
        }
        fe::parse::Stmt::If { then, els, .. } => {
            collect_stmt(then, scope);
            if let Some(els) = els {
                collect_stmt(els, scope);
            }
        }
        fe::parse::Stmt::While { body, .. } | fe::parse::Stmt::DoWhile { body, .. } | fe::parse::Stmt::Switch { body, .. } => {
            collect_stmt(body, scope);
        }
        fe::parse::Stmt::Label { stmt, .. } | fe::parse::Stmt::Case { stmt, .. } => {
            collect_stmt(stmt, scope);
        }
        _ => {}
    }
}

/// Scalar types with a direct DWARF base encoding. Sizes are the LP64
/// target assumptions the rest of the backend shares (signed `char`,
/// 8-byte pointers, 16-byte `long double` slot).
fn scalar_info(ty: &fe::parse::Ty) -> Option<(&'static str, mir::BaseEncoding, u64)> {
    use fe::parse::Ty;
    use mir::BaseEncoding::*;
    Some(match ty {
        Ty::Bool => ("_Bool", Boolean, 1),
        Ty::Char => ("char", SignedChar, 1),
        Ty::SignedChar => ("signed char", SignedChar, 1),
        Ty::UnsignedChar => ("unsigned char", UnsignedChar, 1),
        Ty::Short => ("short", Signed, 2),
        Ty::UnsignedShort => ("unsigned short", Unsigned, 2),
        Ty::Int => ("int", Signed, 4),
        Ty::UnsignedInt => ("unsigned int", Unsigned, 4),
        Ty::Long => ("long", Signed, 8),
        Ty::UnsignedLong => ("unsigned long", Unsigned, 8),
        Ty::LongLong => ("long long", Signed, 8),
        Ty::UnsignedLongLong => ("unsigned long long", Unsigned, 8),
        Ty::Int128 => ("__int128", Signed, 16),
        Ty::UnsignedInt128 => ("unsigned __int128", Unsigned, 16),
        Ty::Nullptr => ("nullptr_t", Unsigned, 8),
        Ty::Float => ("float", Float, 4),
        Ty::Double => ("double", Float, 8),
        // x87 80-bit extended in a 16-byte slot (LP64), like gcc.
        Ty::LongDouble => ("long double", Float, 16),
        Ty::Float128 => ("__float128", Float, 16),
        Ty::Complex(inner) => match inner.as_ref() {
            Ty::Float => ("float _Complex", ComplexFloat, 8),
            Ty::Double => ("double _Complex", ComplexFloat, 16),
            Ty::LongDouble => ("long double _Complex", ComplexFloat, 32),
            // Integer complex has no DWARF base encoding; omit the DIE.
            _ => return None,
        },
        _ => return None,
    })
}

fn push_type(module: &mut mir::ModuleBuilder, ty: mir::DebugType) -> mir::DebugTypeId {
    let id = module.module.debug_types.len() as mir::DebugTypeId;
    module.module.debug_types.push(ty);
    id
}

/// Structural type lowering, deduplicated through `cache` by frontend
/// type equality. Structs and enums insert their entry before members
/// so pointers back to the same type terminate.
fn lower_ty_debug(
    cache: &mut Vec<(fe::parse::Ty, mir::DebugTypeId)>,
    module: &mut mir::ModuleBuilder,
    sema: &fe::sema::Sema,
    interner: &fe::lexer::Interner,
    ty: &fe::parse::Ty,
) -> Option<mir::DebugTypeId> {
    use fe::parse::Ty;
    if let Some((_, id)) = cache.iter().find(|(t, _)| t == ty) {
        return Some(*id);
    }
    // Qualified types wrap the inner DIE in qualifier DIEs, innermost
    // first (`const int` is const-of-int).
    if let Ty::Qualified { quals, inner } = ty {
        let mut id = lower_ty_debug(cache, module, sema, interner, inner)?;
        for q in quals.iter_qualifiers() {
            let qualifier = match q {
                fe::parse::TypeQualifier::Const => mir::Qualifier::Const,
                fe::parse::TypeQualifier::Volatile => mir::Qualifier::Volatile,
                fe::parse::TypeQualifier::Restrict => mir::Qualifier::Restrict,
            };
            id = push_type(module, mir::DebugType::Qualified { qualifier, inner: id });
        }
        cache.push((ty.clone(), id));
        return Some(id);
    }
    if let Ty::BitInt { width, signed } = ty {
        let size = fe::layout::bitint_size(*width);
        let encoding = if *signed {
            mir::BaseEncoding::Signed
        } else {
            mir::BaseEncoding::Unsigned
        };
        // Clang names the DIE after the declared precision, and marks
        // the padding when the storage is wider than that precision.
        let name = format!("{}_BitInt({width})", if *signed { "" } else { "unsigned " });
        let id = push_type(
            module,
            mir::DebugType::Base {
                name: name.into(),
                encoding,
                size,
                bit_size: (*width as u64 != size * 8).then_some(*width as u64),
            },
        );
        cache.push((ty.clone(), id));
        return Some(id);
    }
    if let Some((name, encoding, size)) = scalar_info(ty) {
        let id = push_type(
            module,
            mir::DebugType::Base {
                name: name.into(),
                encoding,
                size,
                bit_size: None,
            },
        );
        cache.push((ty.clone(), id));
        return Some(id);
    }
    match ty {
        Ty::Void => None,
        Ty::Qualified { .. } => unreachable!("handled above"),
        Ty::Pointer { qualifiers, pointee, .. } => {
            // `void *` carries no pointee type, like gcc.
            let pointee_id = match pointee.unqualified() {
                Ty::Void => None,
                _ => lower_ty_debug(cache, module, sema, interner, pointee),
            };
            let mut id = push_type(module, mir::DebugType::Pointer { pointee: pointee_id });
            for q in qualifiers.iter_qualifiers() {
                let qualifier = match q {
                    fe::parse::TypeQualifier::Const => mir::Qualifier::Const,
                    fe::parse::TypeQualifier::Volatile => mir::Qualifier::Volatile,
                    fe::parse::TypeQualifier::Restrict => mir::Qualifier::Restrict,
                };
                id = push_type(module, mir::DebugType::Qualified { qualifier, inner: id });
            }
            cache.push((ty.clone(), id));
            Some(id)
        }
        Ty::Array { elem, bound, .. } => {
            let elem_id = lower_ty_debug(cache, module, sema, interner, elem)?;
            let count = match bound {
                Some(fe::parse::ArrayBound::Fixed(n)) if *n > 0 => Some(*n),
                _ => None,
            };
            let id = push_type(module, mir::DebugType::Array { elem: elem_id, count });
            cache.push((ty.clone(), id));
            Some(id)
        }
        Ty::Struct(sid) => {
            let def = sema.structs.get(sid.0 as usize)?;
            let name = def.tag.and_then(|s| interner.resolve(s)).map(str::to_string);
            let id = push_type(
                module,
                mir::DebugType::Struct {
                    name,
                    is_union: def.is_union,
                    size: 0,
                    declaration: false,
                    members: Vec::new(),
                },
            );
            cache.push((ty.clone(), id));
            let Some(layout) = sema.struct_layout(*sid, interner) else {
                if let mir::DebugType::Struct { declaration, .. } = &mut module.module.debug_types[id as usize] {
                    *declaration = true;
                }
                return Some(id);
            };
            if let mir::DebugType::Struct { size, .. } = &mut module.module.debug_types[id as usize] {
                *size = layout.size;
            }
            // `members` parallels named definition members only, so
            // index by name (robust to layout changes).
            let by_name: HashMap<_, _> = layout.members.iter().map(|m| (m.name, m)).collect();
            let mut members = Vec::new();
            for (i, member) in def.members.iter().enumerate() {
                let member_ty = lower_ty_debug(cache, module, sema, interner, &member.ty);
                let Some(member_ty) = member_ty else {
                    continue;
                };
                let Some(sym) = member.name else {
                    // Unnamed bit-fields shape padding only; anonymous
                    // members below still get nameless DIEs.
                    if member.bitfield.is_some() {
                        continue;
                    }
                    members.push(mir::Member {
                        name: None,
                        ty: member_ty,
                        location: mir::MemberLocation::Byte(layout.member_offsets[i]),
                    });
                    continue;
                };
                let name = interner.resolve(sym).map(str::to_string);
                let location = match member.bitfield {
                    Some(width) => match by_name.get(&sym) {
                        Some(ml) => {
                            let bit = ml.bit.unwrap_or(0);
                            mir::MemberLocation::Bit {
                                offset: ml.offset * 8 + u64::from(bit),
                                size: width,
                            }
                        }
                        None => mir::MemberLocation::Byte(layout.member_offsets[i]),
                    },
                    None => {
                        let offset = by_name.get(&sym).map(|m| m.offset).unwrap_or(layout.member_offsets[i]);
                        mir::MemberLocation::Byte(offset)
                    }
                };
                members.push(mir::Member { name, ty: member_ty, location });
            }
            if let mir::DebugType::Struct { members: slot, .. } = &mut module.module.debug_types[id as usize] {
                *slot = members;
            }
            Some(id)
        }
        Ty::Enum(eid) => {
            let def = sema.enums.get(eid.0 as usize)?;
            let name = def.tag.and_then(|s| interner.resolve(s)).map(str::to_string);
            let repr = lower_ty_debug(cache, module, sema, interner, &def.repr)?;
            let signed = !matches!(
                def.repr.unqualified(),
                Ty::UnsignedInt | Ty::UnsignedLong | Ty::UnsignedLongLong | Ty::UnsignedShort | Ty::UnsignedChar
            );
            let size = sema.type_size(&def.repr, interner).unwrap_or(4);
            let id = push_type(
                module,
                mir::DebugType::Enum {
                    name,
                    repr,
                    size,
                    signed,
                    declaration: !def.complete,
                    enumerators: Vec::new(),
                },
            );
            cache.push((ty.clone(), id));
            if !def.complete {
                return Some(id);
            }
            // Values fold exactly like the parser: explicit constants
            // evaluate, implicit ones continue the sequence.
            let mut enumerators = Vec::new();
            let mut prev: Option<i128> = None;
            for e in &def.enumerators {
                let value = match e.value {
                    Some(value) => value,
                    None => prev.map(|p| p.wrapping_add(1)).unwrap_or(0),
                };
                prev = Some(value);
                let Some(name) = interner.resolve(e.name) else {
                    continue;
                };
                enumerators.push(mir::Enumerator { name: name.to_string(), value });
            }
            if let mir::DebugType::Enum { enumerators: slot, .. } = &mut module.module.debug_types[id as usize] {
                *slot = enumerators;
            }
            Some(id)
        }
        Ty::Func { ret, params, variadic, .. } => {
            let ret = match ret.unqualified() {
                Ty::Void => None,
                _ => lower_ty_debug(cache, module, sema, interner, ret),
            };
            let mut param_ids = Vec::new();
            for param in params {
                // `void` in `f(void)` is not a real parameter.
                if matches!(param.unqualified(), Ty::Void) {
                    continue;
                }
                if let Some(id) = lower_ty_debug(cache, module, sema, interner, param) {
                    param_ids.push(id);
                }
            }
            let id = push_type(
                module,
                mir::DebugType::Func {
                    ret,
                    params: param_ids,
                    variadic: *variadic,
                },
            );
            cache.push((ty.clone(), id));
            Some(id)
        }
        Ty::BuiltinVaList => {
            // Opaque target type without a dedicated DWARF encoding;
            // describe it as a byte bag so variables still print raw.
            let size = sema.type_size(ty, interner).unwrap_or(8);
            let id = push_type(
                module,
                mir::DebugType::Base {
                    name: "__builtin_va_list".into(),
                    encoding: mir::BaseEncoding::Unsigned,
                    size,
                    bit_size: None,
                },
            );
            cache.push((ty.clone(), id));
            Some(id)
        }
        Ty::Float128 => None,
        // Scalars were handled by `scalar_info` above.
        _ => None,
    }
}
