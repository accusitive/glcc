use super::*;

impl<'s, 'i> Lowerer<'s, 'i> {
    /// Declare a file-scope object's storage; the initializer folds
    /// in `define_global` once every entity's id exists, since address
    /// constants may point forward.
    pub(crate) fn declare_global(&self, module: &mut mir::ModuleBuilder, id: fe::sema::DeclId) -> mir::Global {
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
    pub(crate) fn declare_block_static(&self, module: &mut mir::ModuleBuilder, id: fe::sema::DeclId) -> mir::Global {
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
    pub(crate) fn lower_data_extern(&self, module: &mut mir::ModuleBuilder, id: fe::sema::DeclId) -> mir::Global {
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
    pub(crate) fn emit_init_fini_arrays(&self, module: &mut mir::ModuleBuilder, funcs: &HashMap<fe::sema::DeclId, mir::Func>) {
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
    pub(crate) fn emit_hook_array(&self, module: &mut mir::ModuleBuilder, funcs: &HashMap<fe::sema::DeclId, mir::Func>, name: &str, section: &str, entries: &[(u64, fe::sema::DeclId)]) {
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
    pub(crate) fn declare_global_with_name(&self, module: &mut mir::ModuleBuilder, id: fe::sema::DeclId, name: &str, local: bool) -> mir::Global {
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
    pub(crate) fn define_global(
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
                let (relocs, relocs32) = split_global_relocs(relocs);
                let shell = &mut module.module.globals[gid];
                shell.ty = mir_ty;
                shell.init = Some(mir::Const::Bytes(image));
                shell.relocs = relocs;
                shell.relocs32 = relocs32;
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
                    Some(folded) if !matches!(mir_ty, mir::Ty::Ptr { .. }) => (Some(self.const_of_folded(&folded, &mir_ty)), Vec::new()),
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
        let (relocs, relocs32) = split_global_relocs(relocs);
        let shell = &mut module.module.globals[gid];
        shell.ty = mir_ty;
        shell.init = init;
        shell.relocs = relocs;
        shell.relocs32 = relocs32;
    }

    /// Bytes an initializer actually covers. Usually the type's own
    /// size, but an initialized flexible array member extends the
    /// object past `sizeof` (GCC extension), so storage sizes itself
    /// from this rather than from the type alone.
    pub(crate) fn init_extent(&self, fe_ty: &fe::parse::Ty, init: &fe::parse::Initializer) -> u64 {
        self.designated_extent(fe_ty, &[], init)
    }

    /// Extent of an initializer under a designator prefix: the
    /// prefix walks to the selected member or element, then the
    /// initializer's own extent follows.
    pub(crate) fn designated_extent(&self, fe_ty: &fe::parse::Ty, designators: &[fe::parse::Designator], init: &fe::parse::Initializer) -> u64 {
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
                    let sub = self.designated_extent(elem_ty, item.designators.get(1..).unwrap_or(&[]), item.init).max(stride);
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
    pub(crate) fn sized_unsized(
        &self,
        module: &mut mir::ModuleBuilder,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        binding: &fe::sema::Binding,
        init: &fe::parse::Initializer,
    ) -> (mir::Ty, Vec<u8>, GlobalRelocs) {
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
                for item in *items {
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
    pub(crate) fn fold_global_bytes(
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
        relocs: &mut GlobalRelocs,
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
                    if let fe::parse::Ty::Array { elem, bound, .. } = fe_ty.unqualified()
                        && let Some(lit_elem) = fe::eval::string_elem_ty_in(parts, self.sema.target.wchar()) {
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
                    if matches!(fe_ty.unqualified(), fe::parse::Ty::Pointer { .. }) {
                        let (target, ..) = self.materialize_string_global(module, parts);
                        relocs.push((base, mir::DataReloc::Global { target, addend: 0 }, pointer_reloc_bits(mir_ty)));
                        return;
                    }
                }
                // A compound literal in a static initializer (C17
                // §6.5.2.5p5: static storage duration at file scope).
                // As a value of object type its bytes patch inline; as
                // a pointer its static object materializes anonymously
                // and the address records as a reloc.
                if let fe::check::CheckedExprKind::CompoundLiteral { ty: lit_ty, init: lit_init, .. } = &e.kind {
                    if matches!(fe_ty.unqualified(), fe::parse::Ty::Pointer { .. }) {
                        let target = self.materialize_global_compound(module, funcs, externs, globals, lit_ty, lit_init);
                        relocs.push((base, mir::DataReloc::Global { target, addend: 0 }, pointer_reloc_bits(mir_ty)));
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
                if matches!(fe_ty.unqualified(), fe::parse::Ty::Pointer { .. })
                    && let fe::check::CheckedExprKind::Cast { expr, .. } | fe::check::CheckedExprKind::Convert { expr, .. } = &e.kind
                        && let Some(fe::eval::Value::Int(n)) = fe::eval::eval_const(expr, self.sema, self.interner) {
                            let len = mir_ty.size_bytes() as usize;
                            image[base as usize..base as usize + len].copy_from_slice(&n.to_le_bytes()[..len]);
                            return;
                        }
                // Addresses, pointer arithmetic, and (through
                // conversions) string literals for pointer targets.
                if matches!(fe_ty.unqualified(), fe::parse::Ty::Pointer { .. }) {
                    self.reloc_global_addr(module, funcs, externs, globals, relocs, base, e, pointer_reloc_bits(mir_ty));
                    return;
                }
                // An address constant held in an integer (`(long)&obj`,
                // at pointer width): a link-time value like the pointer
                // path. Only at pointer width, which never truncates.
                if matches!(mir_ty, mir::Ty::Int { bits: 64 }) && self.is_global_address(e) {
                    self.reloc_global_addr(module, funcs, externs, globals, relocs, base, e, 64);
                    return;
                }
                panic!("non-constant global initializer");
            }
            fe::parse::Initializer::List { items, .. } => match (fe_ty.unqualified(), mir_ty) {
                // A flexible array member holds no elements in its
                // type (its MIR form is a pointer): initializing one
                // extends the object, so elements fold by element
                // stride past the struct.
                (fe::parse::Ty::Array { elem, bound: None, .. }, mir::Ty::Ptr { .. }) => {
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
                            item.init,
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
                    for item in *items {
                        let (off, elem_fe, elem_mir, next, bf) = self.global_designated(base, fe_ty, mir_ty, item.designators, cursor);
                        cursor = next;
                        self.fold_global_bytes(module, image, off, &elem_fe, &elem_mir, item.init, bf, funcs, externs, globals, relocs);
                    }
                }
                (fe::parse::Ty::Struct(_), mir::Ty::Struct { .. }) => {
                    let mut cursor = 0u64;
                    for item in *items {
                        let (off, elem_fe, elem_mir, next, bf) = self.global_designated(base, fe_ty, mir_ty, item.designators, cursor);
                        cursor = next;
                        self.fold_global_bytes(module, image, off, &elem_fe, &elem_mir, item.init, bf, funcs, externs, globals, relocs);
                    }
                }
                // Unions overlay at offset zero; the blob holds bytes.
                (fe::parse::Ty::Struct(id), mir::Ty::Array { .. }) if self.sema.structs[id.0 as usize].is_union => {
                    let mut cursor = 0u64;
                    for item in *items {
                        let (off, elem_fe, elem_mir, next, bf) = self.global_designated(base, fe_ty, mir_ty, item.designators, cursor);
                        cursor = next;
                        self.fold_global_bytes(module, image, off, &elem_fe, &elem_mir, item.init, bf, funcs, externs, globals, relocs);
                    }
                }
                _ if items.len() == 1 && items[0].designators.is_empty() => {
                    self.fold_global_bytes(module, image, base, fe_ty, mir_ty, items[0].init, bf, funcs, externs, globals, relocs);
                }
                _ => panic!("braced list for scalar type"),
            },
        }
    }

    /// See through assignment-inserted conversions to a string literal.
    pub(crate) fn peek_global_str<'a>(e: &'a fe::check::CheckedExpr<'a>) -> Option<&'a [fe::lexer::StringLiteral]> {
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
    pub(crate) fn fold_label_const(&self, e: &fe::check::CheckedExpr) -> Option<i128> {
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
    pub(crate) fn pointee_step(&self, ptr_ty: &fe::parse::Ty) -> Option<i128> {
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
    pub(crate) fn materialize_global_compound(
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
        let (relocs, relocs32) = split_global_relocs(relocs);
        global.relocs = relocs;
        global.relocs32 = relocs32;
        gid
    }

    /// Record one pointer address reloc at `off`: `&var`, `&arr[i]`,
    /// decayed arrays and functions, string literals, and `&x ± c`.
    /// Conversions around the address are transparent. A null pointer
    /// constant is just zero bytes, and the image is pre-zeroed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reloc_global_addr(
        &self,
        module: &mut mir::ModuleBuilder,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        relocs: &mut GlobalRelocs,
        off: u64,
        e: &fe::check::CheckedExpr,
        bits: u32,
    ) {
        if matches!(fe::eval::eval_const(e, self.sema, self.interner), Some(fe::eval::Value::Int(0))) {
            return;
        }
        match &e.kind {
            fe::check::CheckedExprKind::Cast { expr, .. } | fe::check::CheckedExprKind::Convert { expr, .. } => {
                self.reloc_global_addr(module, funcs, externs, globals, relocs, off, expr, bits);
            }
            fe::check::CheckedExprKind::Str(parts) => {
                let (target, ..) = self.materialize_string_global(module, parts);
                relocs.push((off, mir::DataReloc::Global { target, addend: 0 }, bits));
            }
            fe::check::CheckedExprKind::Unary(op, inner) => {
                assert!(matches!(op, fe::parse::UnaryOp::AddrOf), "only address constants reach relocs");
                self.reloc_global_pointed(module, funcs, externs, globals, relocs, off, inner, 0, bits);
            }
            fe::check::CheckedExprKind::Ident(id) => {
                let binding = &self.sema.decls[id.0 as usize];
                match binding.kind {
                    fe::sema::BindingKind::Func => {
                        if let Some(target) = funcs.get(id) {
                            relocs.push((off, mir::DataReloc::Func { target: *target }, bits));
                        } else if let Some(target) = externs.get(id) {
                            relocs.push((off, mir::DataReloc::FuncExtern { target: *target }, bits));
                        } else {
                            panic!("address of a function before its declaration");
                        }
                    }
                    _ => {
                        // Decayed array (or its `&`): the object's base.
                        let Some(target) = globals.get(id) else {
                            panic!("address of undefined storage");
                        };
                        relocs.push((off, mir::DataReloc::Global { target: *target, addend: 0 }, bits));
                    }
                }
            }
            fe::check::CheckedExprKind::Binary(op, lhs, rhs) => {
                // `&x + c` / `&x - c` (and decayed equivalents): one
                // side is an address, the other a folded integer.
                let (addr_side, const_side, neg) = match op {
                    fe::parse::BinaryOp::Add => (*lhs, *rhs, false),
                    fe::parse::BinaryOp::Sub => (*lhs, *rhs, true),
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
                self.reloc_global_pointed(module, funcs, externs, globals, relocs, off, addr_side, addend as i64, bits);
            }
            _ => panic!("non-constant global initializer"),
        }
    }

    /// The pointer *value* of a constant pointer expression: the root
    /// object plus a byte displacement (`&lvalue`, decayed arrays and
    /// functions, `p ± k`, `p->m`, `p[i]`). A scalar variable's value
    /// is not an address constant and panics like the other shapes.
    pub(crate) fn global_ptr_root<'a>(&self, expr: &'a fe::check::CheckedExpr) -> (AddrRoot<'a>, i64) {
        match &expr.kind {
            fe::check::CheckedExprKind::Ident(id) => {
                let binding = &self.sema.decls[id.0 as usize];
                let decays = matches!(binding.kind, fe::sema::BindingKind::Func) || matches!(binding.ty.unqualified(), fe::parse::Ty::Array { .. });
                assert!(decays, "non-constant global initializer");
                (AddrRoot::Decl(*id), 0)
            }
            fe::check::CheckedExprKind::Str(parts) => (AddrRoot::Str(parts), 0),
            fe::check::CheckedExprKind::CompoundLiteral { ty, init, .. } => (AddrRoot::Compound(ty, init), 0),
            fe::check::CheckedExprKind::Cast { expr, .. } | fe::check::CheckedExprKind::Convert { expr, .. } => self.global_ptr_root(expr),
            fe::check::CheckedExprKind::Unary(op, inner) if matches!(op, fe::parse::UnaryOp::AddrOf) => self.global_pointed_root(inner, 0),
            fe::check::CheckedExprKind::Binary(op, lhs, rhs) if matches!(op, fe::parse::BinaryOp::Add | fe::parse::BinaryOp::Sub) => {
                let neg = matches!(op, fe::parse::BinaryOp::Sub);
                let (ptr, int) = if matches!(lhs.ty.unqualified(), fe::parse::Ty::Pointer { .. }) {
                    (*lhs, *rhs)
                } else {
                    (*rhs, *lhs)
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
    pub(crate) fn ptr_stride(&self, ty: &fe::parse::Ty) -> i128 {
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
    pub(crate) fn reloc_global_pointed(
        &self,
        module: &mut mir::ModuleBuilder,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        relocs: &mut GlobalRelocs,
        off: u64,
        pointed: &fe::check::CheckedExpr,
        extra: i64,
        bits: u32,
    ) {
        let (root, disp) = self.global_pointed_root(pointed, extra);
        match root {
            AddrRoot::Str(parts) => {
                let (target, ..) = self.materialize_string_global(module, parts);
                relocs.push((off, mir::DataReloc::Global { target, addend: disp }, bits));
            }
            AddrRoot::Compound(lit_ty, lit_init) => {
                let target = self.materialize_global_compound(module, funcs, externs, globals, lit_ty, lit_init);
                relocs.push((off, mir::DataReloc::Global { target, addend: disp }, bits));
            }
            AddrRoot::Decl(id) => {
                let binding = &self.sema.decls[id.0 as usize];
                if matches!(binding.kind, fe::sema::BindingKind::Func) {
                    assert!(disp == 0, "arithmetic on a function address");
                    if let Some(target) = funcs.get(&id) {
                        relocs.push((off, mir::DataReloc::Func { target: *target }, bits));
                    } else if let Some(target) = externs.get(&id) {
                        relocs.push((off, mir::DataReloc::FuncExtern { target: *target }, bits));
                    } else {
                        panic!("address of a function before its declaration");
                    }
                    return;
                }
                let Some(target) = globals.get(&id) else {
                    panic!("address of undefined storage");
                };
                relocs.push((off, mir::DataReloc::Global { target: *target, addend: disp }, bits));
            }
        }
    }

    /// Root object and byte displacement of a `&...` operand: nested
    /// members add each layer's own total `offset`, subscripts stride
    /// by their level's element size, conversions look through.
    /// Anything else is not an address constant.
    pub(crate) fn global_pointed_root<'a>(&self, pointed: &'a fe::check::CheckedExpr, extra: i64) -> (AddrRoot<'a>, i64) {
        match &pointed.kind {
            fe::check::CheckedExprKind::Ident(id) => (AddrRoot::Decl(*id), extra),
            fe::check::CheckedExprKind::Str(parts) => (AddrRoot::Str(parts), extra),
            fe::check::CheckedExprKind::CompoundLiteral { ty, init, .. } => (AddrRoot::Compound(ty, init), extra),
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
    pub(crate) fn is_global_address(&self, e: &fe::check::CheckedExpr) -> bool {
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

    pub(crate) fn const_of_folded(&self, folded: &fe::eval::Value, mir_ty: &mir::Ty) -> mir::Const {
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
    pub(crate) fn write_complex_part(&self, image: &mut [u8], off: u64, component: &fe::parse::Ty, v: &fe::eval::Value) {
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

    /// A string literal as a fresh `char[N]` global holding the decoded
    /// bytes plus terminator. Wide literals are rejected by the decoder.
    pub(crate) fn string_literal(&self, module: &mut mir::ModuleBuilder, func: &mut mir::Function, frame: &mut Frame, parts: &[fe::lexer::StringLiteral]) -> mir::Value {
        let (global, elem, len) = self.materialize_string_global(module, parts);
        self.emit(
            func,
            frame,
            mir::InstKind::GlobalAddr { global },
            mir::Ty::ptr(mir::Ty::Array { elem: Box::new(elem), len }),
        )
    }

    /// Emit a string literal into a fresh read-only anonymous global,
    /// returning it with its element type and element count for an
    /// address reloc or a `GlobalAddr`.
    pub(crate) fn materialize_string_global(&self, module: &mut mir::ModuleBuilder, parts: &[fe::lexer::StringLiteral]) -> (mir::Global, mir::Ty, u64) {
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
        data.writable = false;
        data.align = Some(width.max(1));
        (global, elem_mir, len)
    }

    /// Resolve an initializer item's designators to an absolute byte
    /// offset plus the subobject's types and the next positional cursor.
    /// Empty designators take the cursor position; a trailing index or
    /// field continues after it. The fifth return names a bit-field
    /// final target for read-modify-write folding (else `None`).
    pub(crate) fn global_designated(
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
                    if let (fe::parse::Ty::Array { elem, bound: None, .. }, mir::Ty::Ptr { .. }) = (fe.unqualified(), &mi) {
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
        // A following bare item continues after the outermost member
        // the chain names, however deep the rest of the path goes.
        let next = match designators.first() {
            Some(fe::parse::Designator::Index { index, .. }) => match fe::eval::eval_const(index, self.sema, self.interner) {
                Some(fe::eval::Value::Int(k)) => (k as u64) + 1,
                _ => cursor,
            },
            Some(des @ (fe::parse::Designator::Field { .. } | fe::parse::Designator::AnonField { .. })) => match (fe_ty.unqualified(), mir_ty) {
                (fe::parse::Ty::Struct(id), mir::Ty::Struct { .. }) => designated_member(self.sema, *id, des) as u64 + 1,
                _ => cursor,
            },
            None => cursor,
        };
        (off, fe, mi, next, bf)
    }

}
