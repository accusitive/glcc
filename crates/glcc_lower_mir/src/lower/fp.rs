use super::*;

impl<'s, 'i> Lowerer<'s, 'i> {
    /// Floating-point classification (`__builtin_isnan`, `isinf`,
    /// `isinf_sign`, `isfinite`, `isnormal`, `signbit`): `isnan(x)` is
    /// `x != x` (only NaN is unordered with itself); the infinity tests
    /// compare the magnitude against `+inf`, and the finite/normal
    /// tests combine that with the smallest normal. Non-float operands
    /// are never NaN, infinite or subnormal (constant ones fold at
    /// check time), but still evaluate for their side effects.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lower_fp_class(
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
    pub(crate) fn lower_fp_classify(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        codes: &[&fe::check::CheckedExpr; 5],
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
    pub(crate) fn float_magnitude(&self, func: &mut mir::Function, frame: &mut Frame, value: mir::Value, operand_mir: &mir::Ty) -> mir::Value {
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
    pub(crate) fn overflow_operand(&self, func: &mut mir::Function, frame: &mut Frame, value: mir::Value, ty: &fe::parse::Ty) -> mir::Value {
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

    /// GCC's overflow builtins: the operation runs exactly in i128
    /// (every operand value and product of 64-bit or narrower types
    /// fits), the wrapped result stores through the pointer, and the
    /// return value says whether the exact value fit the pointed-to
    /// type. The one exception is `u64 * u64`, whose exact product
    /// exceeds i128: it stays a u128 and compares unsigned.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lower_overflow(
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
        let res_bits = match res_ty.unqualified() {
            fe::parse::Ty::BitInt { width, .. } => *width,
            _ => int_bits(&res_mir),
        };
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

}
