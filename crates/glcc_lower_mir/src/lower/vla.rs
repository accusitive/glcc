use super::*;

impl<'s, 'i> Lowerer<'s, 'i> {
    /// Whether a type is variably modified: an array with a `[*]` or
    /// runtime bound at any depth outside a function type (mirrors
    /// `check::Sema::is_variably_modified`).
    pub(crate) fn ty_is_vm(ty: &fe::parse::Ty) -> bool {
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
    pub(crate) fn ty_is_vla_object(ty: &fe::parse::Ty) -> bool {
        matches!(ty.unqualified(), fe::parse::Ty::Array { .. }) && Self::ty_is_vm(ty)
    }

    /// Every `ArrayBound::Vla` index in a type, outermost dimension
    /// first (`int a[m][n]` yields `[m, n]`).
    pub(crate) fn vla_ids_in(ty: &fe::parse::Ty) -> Vec<u32> {
        let mut out = Vec::new();
        Self::collect_vla_ids(ty, &mut out);
        out
    }

    pub(crate) fn collect_vla_ids(ty: &fe::parse::Ty, out: &mut Vec<u32>) {
        if let fe::parse::Ty::Array { elem, bound, .. } = ty.unqualified() {
            if let Some(fe::parse::ArrayBound::Vla(id)) = bound {
                out.push(*id);
            }
            Self::collect_vla_ids(elem, out);
        }
    }

    /// One stored bound expression, evaluated now and widened to
    /// pointer width by its own signedness.
    pub(crate) fn vla_extent(
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
    pub(crate) fn vla_size_with(&self, func: &mut mir::Function, frame: &mut Frame, ty: &fe::parse::Ty, extents: &HashMap<u32, mir::Value>) -> mir::Value {
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
    pub(crate) fn vla_type_size(
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
    pub(crate) fn vla_cached_extents(&self, frame: &Frame, e: &fe::check::CheckedExpr) -> Option<HashMap<u32, mir::Value>> {
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

    /// The element count of an array type: a constant for a fixed
    /// extent, otherwise the cached or freshly evaluated VLA extent.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn countof_value(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        ty: &fe::parse::Ty,
        cached: Option<&HashMap<u32, mir::Value>>,
        result_ty: &fe::parse::Ty,
    ) -> mir::Value {
        match ty.unqualified() {
            fe::parse::Ty::Array {
                bound: Some(fe::parse::ArrayBound::Fixed(n)),
                ..
            } => {
                let mir_ty = lower_ty(result_ty, self.sema, self.interner);
                self.iconst(func, int_bits(&mir_ty), *n as i128)
            }
            fe::parse::Ty::Array {
                bound: Some(fe::parse::ArrayBound::Vla(id)),
                ..
            } => {
                let value = cached
                    .and_then(|map| map.get(id).copied())
                    .unwrap_or_else(|| self.vla_extent(module, func, frame, funcs, externs, globals, *id));
                self.convert(func, frame, value, &fe::parse::Ty::LongLong, result_ty)
            }
            _ => panic!("`_Countof` operand without an array extent"),
        }
    }

    /// Byte size of a `sizeof` operand with variably modified type:
    /// cached extents for an object reference, fresh evaluation
    /// otherwise.
    pub(crate) fn vla_sizeof(
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
    pub(crate) fn pointee_size(&self, pointee: &fe::parse::Ty) -> u64 {
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
    pub(crate) fn pointee_stride(
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
    pub(crate) fn call_free(&self, module: &mut mir::ModuleBuilder, func: &mut mir::Function, frame: &mut Frame, ptr: mir::Value) {
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
    pub(crate) fn lower_vla_object(
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

}
