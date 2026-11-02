use super::*;

impl<'s, 'i> Lowerer<'s, 'i> {
    /// GEP with a properly typed result: walks `pointee_ty` through
    /// the steps so verification sees the structural type, not `Void`.
    pub(crate) fn gep(&self, func: &mut mir::Function, frame: &mut Frame, base: mir::Value, pointee_ty: &mir::Ty, indices: Vec<mir::GepIndex>) -> (mir::Value, mir::Ty) {
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
    pub(crate) fn scaled_index_addr(
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

    /// The negated index of a pointer subtraction `p - i`, as an
    /// `i64`. The operand is widened with its own signedness before
    /// negation: negating a narrow unsigned index first (`unsigned i`
    /// -> `i32` `0xFFFFFFE4`) and zero-extending that loses the sign
    /// and adds 2^32 to the address.
    pub(crate) fn pointer_sub_index(&self, func: &mut mir::Function, frame: &mut Frame, index: mir::Value, index_ty: &fe::parse::Ty) -> mir::Value {
        let wide = self.convert(func, frame, index, index_ty, &fe::parse::Ty::LongLong);
        self.emit(func, frame, mir::InstKind::Neg(wide), mir::Ty::int(64))
    }

    /// `scaled_index_addr` with a runtime stride: a VLA row's byte size
    /// is a value, not a layout constant.
    pub(crate) fn scaled_index_addr_dyn(
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
    pub(crate) fn descent_needs_bytes(&self, base_ty: &fe::parse::Ty, path: &[usize]) -> bool {
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
    pub(crate) fn reduce_bitfield_precision(&self, func: &mut mir::Function, frame: &mut Frame, value: mir::Value, expr: &fe::check::CheckedExpr, mir_ty: mir::Ty) -> mir::Value {
        let Some(bits) = fe::check::bitfield_precision(expr, &self.sema.target) else {
            return value;
        };
        self.reduce_to_bits(func, frame, value, bits, is_signed(&expr.ty, self.sema), mir_ty)
    }

    /// A value narrowed to `bits` of `mir_ty`, sign-extended when the
    /// type is signed: the stored value of a bit-field, in a register.
    pub(crate) fn reduce_to_bits(&self, func: &mut mir::Function, frame: &mut Frame, value: mir::Value, bits: u32, signed: bool, mir_ty: mir::Ty) -> mir::Value {
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
            let mask = self.low_bits_mask(func, frame, width, bits);
            self.emit(func, frame, mir::InstKind::And(value, mask), mir_ty)
        }
    }

    /// Constant mask with the low `bits` bits set in a value of
    /// `width` bits. Widths past 128 build the mask with `not 0` and a
    /// shift so it never has to fit an i128 constant.
    pub(crate) fn low_bits_mask(&self, func: &mut mir::Function, frame: &mut Frame, width: u32, bits: u32) -> mir::Value {
        debug_assert!(bits < width);
        if bits >= 128 {
            let ty = mir::Ty::int(width);
            let zero = self.iconst(func, width, 0);
            let all = self.emit(func, frame, mir::InstKind::Not(zero), ty.clone());
            let shift = self.iconst(func, width, i128::from(width - bits));
            return self.emit(func, frame, mir::InstKind::LShr(all, shift), ty);
        }
        let mask = if bits == 128 { u128::MAX } else { (1u128 << bits) - 1 };
        self.iconst(func, width, mask as i128)
    }

    /// Storage unit of a bit-field: the declared type's byte size in
    /// bits. `_Bool` is the exception — its value type is 1 bit, but
    /// its unit is the byte the layout packs the field into.
    pub(crate) fn bitfield_unit_ty(&self, ty: &fe::parse::Ty) -> mir::Ty {
        match lower_ty(ty, self.sema, self.interner) {
            mir::Ty::Bool => mir::Ty::int(8),
            unit => unit,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bitfield_load(
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
    pub(crate) fn bitfield_store(&self, func: &mut mir::Function, frame: &mut Frame, addr: mir::Value, unit_fe: &fe::parse::Ty, value: mir::Value, bit: u32, width: u32, vol: bool) {
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
    pub(crate) fn bitfield_target_of(&self, id: fe::sema::StructId, index: usize) -> Option<BitfieldTarget> {
        let def = &self.sema.structs[id.0 as usize];
        let member = def.members.get(index)?;
        let width = member.bitfield?;
        let layout = self.sema.struct_layout(id, self.interner)?;
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
    pub(crate) fn init_bitfield(
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
    pub(crate) fn ptr_offset_addr(&self, func: &mut mir::Function, frame: &mut Frame, ptr: mir::Value, offset: u64, obj: mir::Ty) -> (mir::Value, mir::Ty) {
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
    pub(crate) fn retype_addr(&self, func: &mut mir::Function, frame: &mut Frame, addr: mir::Value, ty: mir::Ty) -> mir::Value {
        self.ptr_offset_addr(func, frame, addr, 0, ty).0
    }

    /// Address of an lvalue expression plus the addressed object type:
    /// variables resolve to slots or globals, member/subscript steps
    /// lower to GEPs, dereference to the pointer itself, and compound
    /// literals materialize a fresh slot.
    pub(crate) fn place(
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
                    let (addr, obj) = self.ptr_offset_addr(func, frame, ptr, *offset, obj);
                    return (addr, obj);
                }
                let (mut addr, mut obj) = self.place(module, func, frame, funcs, externs, globals, base);
                for &field in *path {
                    let (next, ty) = self.gep(func, frame, addr, &obj, vec![mir::GepIndex::Field(field as u32)]);
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
                self.scaled_index_addr(func, frame, addr, idx, &index.ty, scale, obj)
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
            fe::check::CheckedExprKind::CompoundLiteral { ty, init, storage } => {
                let mir_ty = lower_ty(ty, self.sema, self.interner);
                if storage.is_automatic() {
                    let slot = self.alloca(func, frame, mir_ty.clone());
                    self.lower_init(module, func, frame, funcs, externs, globals, slot, ty, init);
                    return (slot, mir_ty);
                }
                // A static or thread-local compound literal is one
                // anonymous object for the whole program (or thread),
                // not a fresh temporary on each evaluation.
                let target = self.materialize_global_compound(module, funcs, externs, globals, ty, init);
                if storage.contains(fe::parse::CompoundStorage::THREAD_LOCAL) {
                    module.module.globals[target].tls = true;
                }
                let addr = self.emit(func, frame, mir::InstKind::GlobalAddr { global: target }, mir::Ty::ptr(mir_ty.clone()));
                (addr, mir_ty)
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
    pub(crate) fn rvalue_of_place(&self, func: &mut mir::Function, frame: &mut Frame, addr: mir::Value, obj_is_array: bool, mir_ty: mir::Ty, vol: bool) -> mir::Value {
        if obj_is_array { addr } else { self.load(func, frame, addr, mir_ty, vol) }
    }

}
