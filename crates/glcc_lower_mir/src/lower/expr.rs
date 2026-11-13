use super::*;

impl<'s, 'i> Lowerer<'s, 'i> {
    /// Rvalue of an expression as a MIR value.
    pub(crate) fn expr_value(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        expr: &fe::check::CheckedExpr,
    ) -> mir::Value {
        let value = self.expr_value_inner(module, func, frame, funcs, externs, globals, expr);
        self.normalize_bitint(func, frame, value, &expr.ty)
    }

    /// Keep a `_BitInt(N)` value sign- or zero-extended from its
    /// declared precision. MIR carries the padded storage width, so
    /// arithmetic on a value narrower than its storage must mask here
    /// to wrap at N bits rather than at the storage width.
    pub(crate) fn normalize_bitint(&self, func: &mut mir::Function, frame: &mut Frame, value: mir::Value, ty: &fe::parse::Ty) -> mir::Value {
        let ty = match ty.unqualified() {
            fe::parse::Ty::Enum(id) => self.sema.enums[id.0 as usize].repr.unqualified(),
            other => other,
        };
        let fe::parse::Ty::BitInt { width, signed } = ty else {
            return value;
        };
        let mir_ty = lower_ty(ty, self.sema, self.interner);
        let storage = int_bits(&mir_ty);
        if *width >= storage {
            return value;
        }
        let int_ty = mir::Ty::int(storage);
        if *signed {
            let shift = self.iconst(func, storage, i128::from(storage - *width));
            let up = self.emit(func, frame, mir::InstKind::Shl(value, shift), int_ty.clone());
            self.emit(func, frame, mir::InstKind::AShr(up, shift), int_ty)
        } else {
            let mask = self.low_bits_mask(func, frame, storage, *width);
            self.emit(func, frame, mir::InstKind::And(value, mask), int_ty)
        }
    }

    /// `expr_value` without the `_BitInt` normalization wrapper; every
    /// recursive value request goes through the wrapper above.
    pub(crate) fn expr_value_inner(
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
                    if matches!(inner_ty, mir::Ty::Ptr { .. }) {
                        let fe::parse::Ty::Pointer { pointee, .. } = inner.ty.unqualified().clone() else {
                            panic!("pointer step without a pointer");
                        };
                        let scale = match pointee.unqualified() {
                            fe::parse::Ty::Void | fe::parse::Ty::Func { .. } => 1,
                            _ => self.sema.type_size(&pointee, self.interner).expect("step over sized pointee"),
                        };
                        let mir::Ty::Ptr { pointee: pointee_mir, .. } = &inner_ty else {
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
                    if is_complex(&inner.ty) {
                        // C2y `++`/`--` on a complex lvalue steps the
                        // real part by one and leaves the imaginary part
                        // untouched.
                        let cur = self.load(func, frame, addr, inner_ty.clone(), access_volatile(inner, &self.sema.decls));
                        let comp_fe = complex_component(&inner.ty);
                        let comp_mir = lower_ty(&comp_fe, self.sema, self.interner);
                        let re = self.complex_extract(func, frame, cur, &inner.ty, 0);
                        let im = self.complex_extract(func, frame, cur, &inner.ty, 1);
                        let one = self.one_value(func, &comp_mir);
                        let next_re = self.complex_add(func, frame, re, one, &comp_mir, matches!(op, PreInc | PostInc));
                        let next = self.complex_build(func, frame, &inner.ty, next_re, im);
                        self.store(func, frame, addr, next, access_volatile(inner, &self.sema.decls));
                        return if matches!(op, PreInc | PreDec) { next } else { cur };
                    }
                    let cur = self.load(func, frame, addr, inner_ty.clone(), access_volatile(inner, &self.sema.decls));
                    let one = self.one_value(func, &inner_ty);
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
                        if fe::check::is_vector(&inner.ty) {
                            let shape = self.vector_shape(&inner.ty);
                            return self.vector_neg(func, frame, value, &shape, &mir_ty);
                        }
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
                        if fe::check::is_vector(&inner.ty) {
                            let shape = self.vector_shape(&inner.ty);
                            return self.vector_bitnot(func, frame, value, &shape, &mir_ty);
                        }
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
                            mir::Ty::Ptr { .. } => self.iconst(func, 64, 0),
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
                let lhs_ptr = matches!(lhs_ty, mir::Ty::Ptr { .. });
                let rhs_ptr = matches!(lower_ty(&rhs.ty, self.sema, self.interner), mir::Ty::Ptr { .. });
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
                // GNU vectors compute lane-wise: the scalar operand
                // splats in, comparisons yield integer mask vectors
                // (see `vector_binary`).
                if fe::check::is_vector(&lhs.ty) || fe::check::is_vector(&rhs.ty) {
                    return self.vector_binary(module, func, frame, funcs, externs, globals, *op, lhs, rhs, &mir_ty);
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
                    if self.complex_builtin(id).is_some()
                        && let Some(value) = self.lower_complex_builtin(module, func, frame, funcs, externs, globals, id, args) {
                            return value;
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
                self.emit_call(
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
                )
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
            fe::check::CheckedExprKind::Shuffle { args } => {
                // Lane i of the result: the mask lane selects within
                // `v0` below the lane count, within `v1` above it. The
                // selectors are runtime values, so both operands
                // materialize to slots and each lane loads through a
                // computed address.
                let (v0, v1, mask) = match args.len() {
                    2 => (&args[0], None, &args[1]),
                    3 => (&args[0], Some(&args[1]), &args[2]),
                    _ => panic!("shuffle with two or three operands"),
                };
                let shape = self.vector_shape(&v0.ty);
                let elem_size = shape.elem_mir.size_bytes();
                let a_val = self.expr_value(module, func, frame, funcs, externs, globals, v0);
                let a_slot = self.alloca(func, frame, lower_ty(&v0.ty, self.sema, self.interner));
                self.store(func, frame, a_slot, a_val, false);
                let b_slot = match v1 {
                    Some(v1) => {
                        let b_val = self.expr_value(module, func, frame, funcs, externs, globals, v1);
                        let b_slot = self.alloca(func, frame, lower_ty(&v1.ty, self.sema, self.interner));
                        self.store(func, frame, b_slot, b_val, false);
                        b_slot
                    }
                    None => a_slot,
                };
                let mask_val = self.expr_value(module, func, frame, funcs, externs, globals, mask);
                let lane_ty = shape.elem_mir.clone();
                let out_slot = self.alloca(func, frame, lower_ty(&expr.ty, self.sema, self.interner));
                for lane in 0..shape.lanes {
                    let sel = self.emit(func, frame, mir::InstKind::ExtractLane { vec: mask_val, lane: lane as u8 }, lane_ty.clone());
                    let lanes_i64 = self.emit(func, frame, mir::InstKind::Zext(sel), mir::Ty::int(64));
                    let lane_count = self.iconst(func, 64, i128::from(shape.lanes));
                    let in_v0 = self.emit(
                        func,
                        frame,
                        mir::InstKind::Icmp {
                            pred: mir::IcmpPred::Ult,
                            lhs: lanes_i64,
                            rhs: lane_count,
                        },
                        mir::Ty::Bool,
                    );
                    let idx0 = lanes_i64;
                    let sub = self.iconst(func, 64, i128::from(shape.lanes));
                    let idx1 = self.emit(func, frame, mir::InstKind::Sub(lanes_i64, sub), mir::Ty::int(64));
                    let addr0 = self.scaled_index_addr(func, frame, a_slot, idx0, &fe::parse::Ty::LongLong, elem_size, lane_ty.clone()).0;
                    let addr1 = self.scaled_index_addr(func, frame, b_slot, idx1, &fe::parse::Ty::LongLong, elem_size, lane_ty.clone()).0;
                    let from0 = self.load(func, frame, addr0, lane_ty.clone(), false);
                    let from1 = self.load(func, frame, addr1, lane_ty.clone(), false);
                    let val = self.emit(func, frame, mir::InstKind::Select { cond: in_v0, then_val: from0, else_val: from1 }, lane_ty.clone());
                    let dst = self.iconst(func, 64, i128::from(lane));
                    let out_addr = self.scaled_index_addr(func, frame, out_slot, dst, &fe::parse::Ty::LongLong, elem_size, lane_ty.clone()).0;
                    self.store(func, frame, out_addr, val, false);
                }
                self.load(func, frame, out_slot, mir_ty, false)
            }
            fe::check::CheckedExprKind::ShuffleVector { args, indices } => {
                // Lane `i` takes the constant `indices[i]` of the
                // concatenated operands; `-1` and out-of-range are
                // undef. Both shapes (first-class and bit-pun) ride
                // byte-indexed slot loads.
                let shape = self.vector_shape(&args[0].ty);
                let elem_size = shape.elem_mir.size_bytes();
                let a_val = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let a_slot = self.alloca(func, frame, lower_ty(&args[0].ty, self.sema, self.interner));
                self.store(func, frame, a_slot, a_val, false);
                let b_slot = if args.len() > 1 {
                    let b_val = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                    let b_slot = self.alloca(func, frame, lower_ty(&args[1].ty, self.sema, self.interner));
                    self.store(func, frame, b_slot, b_val, false);
                    b_slot
                } else {
                    a_slot
                };
                let out_slot = self.alloca(func, frame, mir_ty.clone());
                let total_lanes = shape.lanes as u64 * if args.len() > 1 { 2 } else { 1 };
                for (lane, idx) in indices.iter().enumerate() {
                    let val = if *idx >= 0 && (*idx as u64) < total_lanes {
                        let (slot, within) = if (*idx as u64) < shape.lanes as u64 {
                            (a_slot, *idx as u64)
                        } else {
                            (b_slot, *idx as u64 - shape.lanes as u64)
                        };
                        let sel = self.iconst(func, 64, within as i128);
                        let addr = self.scaled_index_addr(func, frame, slot, sel, &fe::parse::Ty::LongLong, elem_size, shape.elem_mir.clone()).0;
                        self.load(func, frame, addr, shape.elem_mir.clone(), false)
                    } else {
                        func.values.push(mir::ValueData {
                            ty: shape.elem_mir.clone(),
                            def: mir::Def::Const(mir::Const::Undef),
                        })
                    };
                    let dst = self.iconst(func, 64, lane as i128);
                    let out_addr = self.scaled_index_addr(func, frame, out_slot, dst, &fe::parse::Ty::LongLong, elem_size, shape.elem_mir.clone()).0;
                    self.store(func, frame, out_addr, val, false);
                }
                self.load(func, frame, out_slot, mir_ty, false)
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
                            for id in *ids {
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
            fe::check::CheckedExprKind::StaticAssert => self.iconst(func, 32, 0),
            fe::check::CheckedExprKind::CountofTy(ty) => self.countof_value(module, func, frame, funcs, externs, globals, ty, None, &expr.ty),
            fe::check::CheckedExprKind::CountofExpr(inner) => {
                let cached = self.vla_cached_extents(frame, inner);
                self.countof_value(module, func, frame, funcs, externs, globals, &inner.ty, cached.as_ref(), &expr.ty)
            }
            fe::check::CheckedExprKind::Offsetof { offset, .. } => self.iconst(func, int_bits(&mir_ty), *offset as i128),
            fe::check::CheckedExprKind::Generic { index, assocs, .. } => {
                let Some((_, chosen)) = assocs.get(*index) else {
                    panic!("generic association index out of range");
                };
                self.expr_value(module, func, frame, funcs, externs, globals, chosen)
            }
            fe::check::CheckedExprKind::GenericTy { index, assocs, .. } => {
                let Some((_, chosen)) = assocs.get(*index) else {
                    panic!("generic association index out of range");
                };
                self.expr_value(module, func, frame, funcs, externs, globals, chosen)
            }
            fe::check::CheckedExprKind::ConstantP(b) => self.iconst(func, int_bits(&mir_ty), i128::from(*b)),
            fe::check::CheckedExprKind::ClassifyType(code) => self.iconst(func, int_bits(&mir_ty), i128::from(*code)),
            fe::check::CheckedExprKind::FpClass { op, arg } => {
                self.lower_fp_class(module, func, frame, funcs, externs, globals, op, arg, &mir_ty)
            }
            fe::check::CheckedExprKind::FpClassify { codes, arg } => {
                self.lower_fp_classify(module, func, frame, funcs, externs, globals, codes, arg, &mir_ty)
            }
            fe::check::CheckedExprKind::Atomic { op, args } => {
                self.lower_atomic(module, func, frame, funcs, externs, globals, op, args, &expr.ty, &mir_ty)
            }
            fe::check::CheckedExprKind::Ia32 { op, args } => {
                self.lower_ia32(module, func, frame, funcs, externs, globals, op, args, &expr.ty)
            }
            fe::check::CheckedExprKind::Overflow { op, lhs, rhs, res, res_ty } => {
                self.lower_overflow(module, func, frame, funcs, externs, globals, op, lhs, rhs, res, res_ty, &mir_ty)
            }
            fe::check::CheckedExprKind::Alloca { size } => {
                self.lower_alloca(module, func, frame, funcs, externs, globals, size)
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
                fp
            }
            fe::check::CheckedExprKind::Clz { ctz, arg, fallback } => {
                // Type-generic count at the operand's own width (no
                // promotion): the count converts to the declared `int`,
                // and a zero operand yields the fallback when given.
                let operand_mir = lower_ty(&arg.ty, self.sema, self.interner);
                let operand_bits = match arg.ty.unqualified() {
                    fe::parse::Ty::BitInt { width, .. } => *width,
                    _ => int_bits(&operand_mir),
                };
                let value = self.expr_value(module, func, frame, funcs, externs, globals, arg);
                let count = if matches!(operand_mir, mir::Ty::Bool) {
                    // A `_Bool` is one bit: its only nonzero value has
                    // no leading or trailing zeros, and zero counts
                    // one (the byte-wide `clz` would say seven).
                    let one = self.iconst(func, int_bits(&operand_mir), 1);
                    self.emit(func, frame, mir::InstKind::Sub(one, value), operand_mir.clone())
                } else {
                    let kind = if *ctz { mir::InstKind::Ctz(value) } else { mir::InstKind::Clz(value) };
                    let count = self.emit(func, frame, kind, operand_mir.clone());
                    if !*ctz && operand_bits < int_bits(&operand_mir) {
                        let extra = self.iconst(func, int_bits(&operand_mir), i128::from(int_bits(&operand_mir) - operand_bits));
                        self.emit(func, frame, mir::InstKind::Sub(count, extra), operand_mir.clone())
                    } else {
                        count
                    }
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
            fe::check::CheckedExprKind::Bswap { arg, bits } => {
                // The operand was already converted to the builtin's
                // parameter width at check time (the result type is
                // the same unsigned type), so the swap emits at that
                // width with no further adjustment.
                let operand_mir = lower_ty(&arg.ty, self.sema, self.interner);
                let value = self.expr_value(module, func, frame, funcs, externs, globals, arg);
                let swapped = self.emit(func, frame, mir::InstKind::Bswap(value), operand_mir);
                self.convert(func, frame, swapped, &arg.ty, &expr.ty)
            }
            fe::check::CheckedExprKind::Rotate { left, arg, count, bits } => {
                // Rotate at the operand's declared width. Reduce the
                // count modulo that width in the count's own type: a
                // narrow `_BitInt` width or a bit-field width is not the
                // storage width, and converting first would drop count
                // bits that matter modulo the width.
                let operand_mir = lower_ty(&arg.ty, self.sema, self.interner);
                let value = self.expr_value(module, func, frame, funcs, externs, globals, arg);
                let count_ty = count.ty.clone();
                let count_mir = lower_ty(&count_ty, self.sema, self.interner);
                let count_storage = int_bits(&count_mir);
                let count = self.expr_value(module, func, frame, funcs, externs, globals, count);
                let work_ty = fe::parse::Ty::UnsignedLongLong;
                let work_mir = mir::Ty::int(64);
                let width = self.iconst(func, 64, i128::from(*bits));
                // The count is unsigned after checking, so a 64-bit
                // work type holds every count up to 64 bits exactly.
                // Wider counts split into halves: 128-bit remainder
                // has no backend lowering.
                let shift = if count_storage > 64 {
                    let half = self.iconst(func, count_storage, 64);
                    let high = self.emit(func, frame, mir::InstKind::LShr(count, half), count_mir);
                    let low64 = self.convert(func, frame, count, &count_ty, &work_ty);
                    let high64 = self.convert(func, frame, high, &count_ty, &work_ty);
                    let low_mod = self.emit(func, frame, mir::InstKind::Urem(low64, width), work_mir.clone());
                    let high_mod = self.emit(func, frame, mir::InstKind::Urem(high64, width), work_mir.clone());
                    let factor = self.iconst(func, 64, ((1u128 << 64) % u128::from(*bits)) as i128);
                    let scaled = self.emit(func, frame, mir::InstKind::Mul(high_mod, factor), work_mir.clone());
                    let sum = self.emit(func, frame, mir::InstKind::Add(scaled, low_mod), work_mir.clone());
                    self.emit(func, frame, mir::InstKind::Urem(sum, width), work_mir.clone())
                } else {
                    let count = if count_storage < 64 {
                        self.convert(func, frame, count, &count_ty, &work_ty)
                    } else {
                        count
                    };
                    self.emit(func, frame, mir::InstKind::Urem(count, width), work_mir.clone())
                };
                let complement = self.emit(func, frame, mir::InstKind::Sub(width, shift), work_mir.clone());
                let complement = self.emit(func, frame, mir::InstKind::Urem(complement, width), work_mir);
                let shift = self.convert(func, frame, shift, &work_ty, &arg.ty);
                let complement = self.convert(func, frame, complement, &work_ty, &arg.ty);
                let (near, far) = if *left { (shift, complement) } else { (complement, shift) };
                let low = self.emit(func, frame, mir::InstKind::Shl(value, near), operand_mir.clone());
                let high = self.emit(func, frame, mir::InstKind::LShr(value, far), operand_mir.clone());
                let raw = self.emit(func, frame, mir::InstKind::Or(low, high), operand_mir.clone());
                // A bit-field or `_BitInt` narrower than its storage
                // keeps only its declared precision.
                let raw = if *bits < int_bits(&operand_mir) {
                    let mask = self.low_bits_mask(func, frame, int_bits(&operand_mir), *bits);
                    self.emit(func, frame, mir::InstKind::And(raw, mask), operand_mir.clone())
                } else {
                    raw
                };
                self.convert(func, frame, raw, &arg.ty, &expr.ty)
            }
            other => todo!("expression lowering for {other:?}"),
        }
    }

    /// Short-circuit `&&`/`||`: the right side evaluates only on the
    /// deciding path. The decided value crosses through a slot, since
    /// this frame never passes block arguments.
    pub(crate) fn logical(
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
    pub(crate) fn store_ternary_arm(&self, func: &mut mir::Function, frame: &mut Frame, slot: mir::Value, val: mir::Value, arm_fe_ty: &fe::parse::Ty, res_fe_ty: &fe::parse::Ty) {
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

    /// Whether `va_list` behaves as GCC's array-of-one-tag (value is
    /// the tag's address, never a load). MSVC's `va_list` is a plain
    /// pointer: its value is the cursor it holds.
    pub(crate) fn va_list_is_array(&self) -> bool {
        !matches!(self.sema.target.os, fe::target::Os::Windows)
    }

    /// Whether an expression denotes array storage decayed to a
    /// pointer (its value is its address, never a load): array idents,
    /// members, and rows, through assignment-inserted conversions. The
    /// checker decays some nodes to pointers (same kind, new type),
    /// so declared types — not node types — decide array-backed
    /// bases. MIR twin of the legacy backend's `expr_denotes_array`.
    pub(crate) fn expr_denotes_array(&self, e: &fe::check::CheckedExpr) -> bool {
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
                // `(p - 1)[i]`, `f()[i]`: the base is a pointer-valued
                // expression whose step no declaration names. A
                // pointer-to-array base still addresses the row — the
                // decayed node's value is the row's address, never a
                // load.
                None => matches!(
                    base.ty.unqualified(),
                    fe::parse::Ty::Pointer { pointee, .. } if matches!(pointee.unqualified(), fe::parse::Ty::Array { .. })
                ),
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
    pub(crate) fn subscript_step_ty(&self, base: &fe::check::CheckedExpr) -> Option<fe::parse::Ty> {
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
    pub(crate) fn subscript_base_is_array(&self, e: &fe::check::CheckedExpr) -> bool {
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
    pub(crate) fn member_path_ty(&self, inner: &fe::check::CheckedExpr, path: &[usize]) -> Option<fe::parse::Ty> {
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
    pub(crate) fn bool_of(
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
    pub(crate) fn bool_of_value(&self, func: &mut mir::Function, frame: &mut Frame, value: mir::Value, fe_ty: &fe::parse::Ty) -> mir::Value {
        match lower_ty(fe_ty, self.sema, self.interner) {
            mir::Ty::Bool | mir::Ty::Int { bits: 1 } => value,
            mir::Ty::Ptr { .. } => {
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

    /// Explicit conversion between scalar MIR values, directed by the
    /// frontend types on each side. Complex on either side desugars
    /// through [`Self::convert_complex`]: the pair type erases the
    /// complex/real distinction, so this must branch on the frontend
    /// types first.
    pub(crate) fn convert(&self, func: &mut mir::Function, frame: &mut Frame, from: mir::Value, from_ty: &fe::parse::Ty, to_ty: &fe::parse::Ty) -> mir::Value {
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
            (mir::Ty::Ptr { .. }, mir::Ty::Bool) => {
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
            (mir::Ty::Ptr { .. }, mir::Ty::Int { .. }) => mir::InstKind::PtrToInt(from),
            (mir::Ty::Ptr { .. }, mir::Ty::Ptr { .. }) => mir::InstKind::Bitcast(from),
            // Same-size vector and bit-punned image casts move the
            // bits whole (the SSE intrinsics pass `__m128`-class types
            // between the lane spellings).
            (mir::Ty::Vector { .. }, _) | (_, mir::Ty::Vector { .. }) if from_mir.size_bytes() == to_mir.size_bytes() => mir::InstKind::Bitcast(from),
            (mir::Ty::Int { .. } | mir::Ty::Bool, mir::Ty::Ptr { .. }) => {
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

    /// Apply a compound assignment's operation to the lvalue's current
    /// value (`x`, converted to `op_ty`) and the converted rhs (`y`,
    /// of type `rhs_ty`), producing a value of `op_ty` (MIR type
    /// `mir_ty`). Pointer arithmetic scales by the pointee size through
    /// byte arithmetic, like `pointer_binary`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compound_op(
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
            let (index, index_ty) = if matches!(op, fe::parse::AssignOp::Sub) {
                (self.pointer_sub_index(func, frame, y, rhs_ty), fe::parse::Ty::LongLong)
            } else {
                (y, rhs_ty.clone())
            };
            let mir::Ty::Ptr { pointee: obj, .. } = mir_ty.clone() else {
                panic!("compound pointer assignment without a pointer result");
            };
            let (addr, _) = self.scaled_index_addr(func, frame, x, index, &index_ty, scale, obj.as_ref().clone());
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
    pub(crate) fn pointer_binary(
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
                let mir::Ty::Ptr { pointee: obj, .. } = mir_ty.clone() else {
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
                let neg = self.pointer_sub_index(func, frame, index, &rhs.ty);
                let mir::Ty::Ptr { pointee: obj, .. } = mir_ty.clone() else {
                    panic!("pointer subtraction without a pointer result");
                };
                let (diff, _) = if Self::ty_is_vm(pointee) {
                    let scale = self.pointee_stride(module, func, frame, funcs, externs, globals, lhs, pointee);
                    self.scaled_index_addr_dyn(func, frame, base, neg, &fe::parse::Ty::LongLong, scale, obj.as_ref().clone())
                } else {
                    self.scaled_index_addr(func, frame, base, neg, &fe::parse::Ty::LongLong, self.pointee_size(pointee), obj.as_ref().clone())
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
    pub(crate) fn convert_int_width(&self, func: &mut mir::Function, frame: &mut Frame, value: mir::Value, from_bits: u32, to_mir: &mir::Ty) -> mir::Value {
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
    pub(crate) fn int_cmp(&self, op: &fe::parse::BinaryOp, signed: bool, x: mir::Value, y: mir::Value) -> mir::InstKind {
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

    pub(crate) fn convert_bool(&self, func: &mut mir::Function, frame: &mut Frame, test: mir::Value, to_mir: &mir::Ty) -> mir::Value {
        if *to_mir == mir::Ty::Bool {
            return test;
        }
        // `Bool` is one byte; widening to `int` zero-extends.
        self.emit(func, frame, mir::InstKind::Zext(test), to_mir.clone())
    }


    /// Lower call arguments and install a call instruction of any
    /// flavor: `template` carries the callee addressing with empty
    /// args, filled here. Void calls fabricate a unit value the
    /// statement layer discards.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_call(
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
    pub(crate) fn indirect_sig(&self, callee: &fe::check::CheckedExpr) -> mir::Signature {
        let func_ty = match callee.ty.unqualified() {
            fe::parse::Ty::Pointer { pointee, .. } => match pointee.unqualified() {
                fe::parse::Ty::Func { .. } => pointee.as_ref().clone(),
                _ => panic!("indirect call to non-function"),
            },
            fe::parse::Ty::Func { .. } => callee.ty.clone(),
            _ => panic!("indirect call to non-function"),
        };
        let fe::parse::Ty::Func { ret, params, variadic, prototyped } = func_ty.unqualified() else {
            panic!("indirect call to non-function");
        };
        let mut sig = mir::Signature::new(
            params.iter().map(|p| lower_ty(p, self.sema, self.interner)).collect(),
            lower_ty(ret, self.sema, self.interner),
        );
        sig.variadic = *variadic || !*prototyped;
        sig
    }

}
