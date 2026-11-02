use super::*;

impl<'s, 'i> Lowerer<'s, 'i> {
    /// One component of a complex value: `ExtractValue` off the pair,
    /// which lowers to a load from the value's slot. Component
    /// *addresses* (for stores through `__real__`/`__imag__`) use
    /// field GEPs in `place` instead: values carry the pair struct
    /// type and have no address of their own.
    pub(crate) fn complex_extract(&self, func: &mut mir::Function, frame: &mut Frame, agg: mir::Value, fe_ty: &fe::parse::Ty, index: u32) -> mir::Value {
        let comp_mir = lower_ty(&complex_component(fe_ty), self.sema, self.interner);
        self.emit(func, frame, mir::InstKind::ExtractValue { agg, index }, comp_mir)
    }

    /// Assemble a complex value from its component values: two
    /// `InsertValue` steps over an `undef` base, which lowers to
    /// stores into a fresh slot (no initial copy).
    pub(crate) fn complex_build(&self, func: &mut mir::Function, frame: &mut Frame, fe_ty: &fe::parse::Ty, re: mir::Value, im: mir::Value) -> mir::Value {
        let pair = lower_ty(fe_ty, self.sema, self.interner);
        let base = func.values.push(mir::ValueData {
            ty: pair.clone(),
            def: mir::Def::Const(mir::Const::Undef),
        });
        let with_re = self.emit(func, frame, mir::InstKind::InsertValue { agg: base, elem: re, index: 0 }, pair.clone());
        self.emit(func, frame, mir::InstKind::InsertValue { agg: with_re, elem: im, index: 1 }, pair)
    }

    /// The constant one at a scalar type's width: `1` for integers,
    /// `1.0` for floats (long double uses the target representation
    /// through the shared const fold). Complex increments step their
    /// real component with this.
    pub(crate) fn one_value(&self, func: &mut mir::Function, ty: &mir::Ty) -> mir::Value {
        match ty {
            mir::Ty::Float(_) => {
                let bits = match float_kind(ty) {
                    mir::FloatKind::F32 => u128::from(1.0f32.to_bits()),
                    mir::FloatKind::F64 => u128::from(1.0f64.to_bits()),
                    mir::FloatKind::F80 => float_bits(fe::eval::FloatValue::long_double_one_in(fe::target::LongDoubleKind::F80), ty),
                    mir::FloatKind::F128 => float_bits(fe::eval::FloatValue::long_double_one_in(fe::target::LongDoubleKind::F128), ty),
                };
                func.push_const(mir::Const::Float { kind: float_kind(ty), bits })
            }
            _ => self.iconst(func, int_bits(ty), 1),
        }
    }

    /// The zero of a complex component type: integer zero, or
    /// floating `+0.0` in the component's precision (all-zero bits,
    /// which is `+0.0` for every supported float representation).
    pub(crate) fn complex_zero(&self, func: &mut mir::Function, comp_mir: &mir::Ty) -> mir::Value {
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
    pub(crate) fn convert_complex(&self, func: &mut mir::Function, frame: &mut Frame, from: mir::Value, from_ty: &fe::parse::Ty, to_ty: &fe::parse::Ty) -> mir::Value {
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
    pub(crate) fn complex_nonzero(&self, func: &mut mir::Function, frame: &mut Frame, agg: mir::Value, fe_ty: &fe::parse::Ty) -> mir::Value {
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
    pub(crate) fn complex_neg_part(&self, func: &mut mir::Function, frame: &mut Frame, v: mir::Value, comp_mir: &mir::Ty) -> mir::Value {
        match comp_mir {
            mir::Ty::Float(_) => self.emit(func, frame, mir::InstKind::FNeg(v), comp_mir.clone()),
            _ => self.emit(func, frame, mir::InstKind::Neg(v), comp_mir.clone()),
        }
    }

    /// Unary `-x` on a complex value: both parts negated.
    pub(crate) fn complex_neg(&self, func: &mut mir::Function, frame: &mut Frame, agg: mir::Value, fe_ty: &fe::parse::Ty) -> mir::Value {
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
    pub(crate) fn complex_conj(&self, func: &mut mir::Function, frame: &mut Frame, agg: mir::Value, fe_ty: &fe::parse::Ty) -> mir::Value {
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
    pub(crate) fn complex_literal(&self, func: &mut mir::Function, frame: &mut Frame, expr: &fe::check::CheckedExpr) -> mir::Value {
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
    pub(crate) fn complex_binop(&self, func: &mut mir::Function, frame: &mut Frame, op: &fe::parse::BinaryOp, lhs: mir::Value, rhs: mir::Value, fe_ty: &fe::parse::Ty) -> mir::Value {
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
    pub(crate) fn complex_add(&self, func: &mut mir::Function, frame: &mut Frame, x: mir::Value, y: mir::Value, comp_mir: &mir::Ty, add: bool) -> mir::Value {
        let kind = match (add, comp_mir) {
            (true, mir::Ty::Float(_)) => mir::InstKind::FAdd(x, y),
            (false, mir::Ty::Float(_)) => mir::InstKind::FSub(x, y),
            (true, _) => mir::InstKind::Add(x, y),
            (false, _) => mir::InstKind::Sub(x, y),
        };
        self.emit(func, frame, kind, comp_mir.clone())
    }

    /// One component multiplication.
    pub(crate) fn complex_mul(&self, func: &mut mir::Function, frame: &mut Frame, x: mir::Value, y: mir::Value, comp_mir: &mir::Ty) -> mir::Value {
        let kind = match comp_mir {
            mir::Ty::Float(_) => mir::InstKind::FMul(x, y),
            _ => mir::InstKind::Mul(x, y),
        };
        self.emit(func, frame, kind, comp_mir.clone())
    }

    /// One component division: float parts round once, integer parts
    /// divide by the signedness of the component (truncating, like
    /// scalar division).
    pub(crate) fn complex_div(&self, func: &mut mir::Function, frame: &mut Frame, x: mir::Value, y: mir::Value, comp_fe: &fe::parse::Ty, comp_mir: &mir::Ty) -> mir::Value {
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
    pub(crate) fn complex_cmp(
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

}
