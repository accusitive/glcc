use super::*;

impl<'s, 'i> Lowerer<'s, 'i> {
    /// Address operand of an atomic builtin: the argument already
    /// denotes an address (`&x`, a decayed array), so its rvalue is
    /// the address itself, mirroring the legacy backend's
    /// evaluate-then-address convention.
    pub(crate) fn atomic_addr(
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
    pub(crate) fn atomic_result(&self, func: &mut mir::Function, ty: mir::Ty) -> mir::Value {
        func.values.push(mir::ValueData {
            ty,
            def: mir::Def::Inst(mir::Inst::from_usize(func.insts.len())),
        })
    }

    /// Install an atomic instruction, returning its result (or the unit
    /// value statements discard for resultless ones).
    pub(crate) fn atomic_finish(&self, func: &mut mir::Function, frame: &mut Frame, kind: mir::InstKind, result: Option<mir::Value>) -> mir::Value {
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
    pub(crate) fn atomic_pointee_of(e: &fe::check::CheckedExpr) -> fe::parse::Ty {
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
    pub(crate) fn lower_atomic(
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
            AtomicOp::InterlockedCmpxchg => {
                let ptr = self.atomic_addr(module, func, frame, funcs, externs, globals, &args[0]);
                let desired = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let expected = self.expr_value(module, func, frame, funcs, externs, globals, &args[2]);
                let pointed = Self::atomic_pointee_of(&args[0]);
                let pointed_mir = lower_ty(&pointed, self.sema, self.interner);
                let old = self.emit(
                    func,
                    frame,
                    mir::InstKind::AtomicCas {
                        addr: ptr,
                        expected,
                        replacement: desired,
                    },
                    pointed_mir,
                );
                self.convert(func, frame, old, &pointed, fe_ty)
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

}
