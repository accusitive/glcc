use super::*;

impl<'s, 'i> Lowerer<'s, 'i> {
    /// Whether an initializer writes every byte that can affect the
    /// object's value, making the zero-fill before it redundant.
    ///
    /// Only positional, recursively complete lists qualify: named
    /// designators and bit-field structs keep the zero-fill because
    /// they touch only part of the object (bit-fields read-modify-write
    /// their unnamed neighbors).
    pub(crate) fn initializer_covers_object(&self, fe_ty: &fe::parse::Ty, mir_ty: &mir::Ty, init: &fe::parse::Initializer) -> bool {
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
                        items.len() as u64 == *len && items.iter().all(|item| self.initializer_covers_object(elem, mir_elem, item.init))
                    }
                    // Vector lanes cover the object only when the list
                    // fills every lane with expression items (the 8-byte
                    // shape lowers to an i64, so the FE type decides).
                    (fe::parse::Ty::Vector { elem, len }, _) => {
                        items.len() as u64 == u64::from(*len)
                            && items.iter().all(|item| matches!(item.init, fe::parse::Initializer::Expr(_)) && self.initializer_covers_object(elem, &lower_ty(elem, self.sema, self.interner), item.init))
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
                                .all(|((item, field), member)| self.initializer_covers_object(&member.ty, field, item.init))
                    }
                    // `int x = {1}` is a one-element list around a
                    // scalar initializer.
                    _ if items.len() == 1 => self.initializer_covers_object(fe_ty, mir_ty, items[0].init),
                    _ => false,
                }
            }
        }
    }

    /// Initialize a local slot from an initializer: zero the whole
    /// object first (C's remainder rule, covering padding too), then
    /// store each provided element.
    pub(crate) fn lower_init(
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
                        for item in *items {
                            index = self.array_init_item(module, func, frame, funcs, externs, globals, slot, elem, item, index);
                        }
                    }
                    fe::parse::Ty::Struct(id) => {
                        let mut field = 0usize;
                        for item in *items {
                            field = self.struct_init_item(module, func, frame, funcs, externs, globals, slot, *id, item, field);
                        }
                    }
                    // GNU vector lanes fill in order (the checker caps
                    // the count at the lane number); a partial list
                    // zero-fills through the memset above.
                    fe::parse::Ty::Vector { elem, .. } => {
                        let shape = self.vector_shape(fe_ty);
                        for (i, item) in items.iter().enumerate() {
                            let idx = self.iconst(func, 64, i as i128);
                            let addr = self.vector_lane_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, &shape);
                            self.lower_init(module, func, frame, funcs, externs, globals, addr, elem, item.init);
                        }
                    }
                    // `int x = {1}` is legal C: bare scalar braces.
                    _ if items.len() == 1 && items[0].designators.is_empty() => {
                        self.lower_init(module, func, frame, funcs, externs, globals, slot, fe_ty, items[0].init);
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
    pub(crate) fn bitfield_struct_init_item(
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
                self.init_bitfield(module, func, frame, funcs, externs, globals, addr, &t, item.init);
            } else {
                self.lower_init(module, func, frame, funcs, externs, globals, addr, &member.ty, item.init);
            }
            return next;
        }
        let (addr, fe, bf) = self.init_addr(func, frame, addr, member.ty.clone(), rest);
        match bf {
            Some(t) => self.init_bitfield(module, func, frame, funcs, externs, globals, addr, &t, item.init),
            None => self.lower_init(module, func, frame, funcs, externs, globals, addr, &fe, item.init),
        }
        next
    }

    /// Walk an explicit designator list from a slot address, returning
    /// the addressed subobject and its frontend type.
    pub(crate) fn init_addr(
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
    pub(crate) fn array_init_item(
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
            let mir::Ty::Ptr { pointee: inner, .. } = func.values[slot].ty.clone() else {
                panic!("array initializer slot is not an address");
            };
            let (addr, _) = self.gep(func, frame, slot, inner.as_ref(), vec![mir::GepIndex::ConstIndex(cursor)]);
            (addr, elem_ty.clone(), None)
        } else {
            let array_ty = fe::parse::Ty::Array {
                elem: Box::new(elem_ty.clone()),
                quals: fe::parse::TypeQualifiers::empty(),
                bound: None,
            };
            self.init_addr(func, frame, slot, array_ty, item.designators)
        };
        match bf {
            Some(t) => self.init_bitfield(module, func, frame, funcs, externs, globals, addr, &t, item.init),
            None => self.lower_init(module, func, frame, funcs, externs, globals, addr, &ty, item.init),
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
    pub(crate) fn struct_init_item(
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
            self.init_addr(func, frame, slot, struct_ty, item.designators)
        };
        match bf {
            Some(t) => self.init_bitfield(module, func, frame, funcs, externs, globals, addr, &t, item.init),
            None => self.lower_init(module, func, frame, funcs, externs, globals, addr, &ty, item.init),
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
    pub(crate) fn union_init_item(
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
                self.init_bitfield(module, func, frame, funcs, externs, globals, slot, &t, item.init);
            } else {
                self.lower_init(module, func, frame, funcs, externs, globals, slot, &member_fe, item.init);
            }
        } else {
            let (addr, fe, bf) = self.init_addr(func, frame, slot, member_fe, rest);
            match bf {
                Some(t) => self.init_bitfield(module, func, frame, funcs, externs, globals, addr, &t, item.init),
                None => self.lower_init(module, func, frame, funcs, externs, globals, addr, &fe, item.init),
            }
        }
        next
    }

}
