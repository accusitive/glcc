use super::*;

impl<'s, 'i> Lowerer<'s, 'i> {
    /// The module extern for `name`, declaring it on first use so a
    /// source declaration of the same function is reused instead of
    /// declared twice.
    pub(crate) fn extern_fn(&self, module: &mut mir::ModuleBuilder, name: &str, params: Vec<mir::Ty>, ret: mir::Ty) -> mir::Extern {
        for (id, ext) in module.module.externs.iter_enumerated() {
            if ext.name == name {
                return id;
            }
        }
        module.module.push_extern(name, mir::Signature { params, ret, variadic: false })
    }

    /// Which `va_*` builtin a call target names, if any.
    pub(crate) fn va_builtin(&self, id: fe::sema::DeclId) -> Option<VaBuiltin> {
        let binding = &self.sema.decls[id.0 as usize];
        if !matches!(binding.kind, fe::sema::BindingKind::Func) {
            return None;
        }
        match binding.name.and_then(|n| self.interner.resolve(n)) {
            Some("__builtin_va_start" | "__builtin_c23_va_start") => Some(VaBuiltin::Start),
            Some("__builtin_va_end") => Some(VaBuiltin::End),
            Some("__builtin_va_copy") => Some(VaBuiltin::Copy),
            Some("__va_start") => Some(VaBuiltin::MsvcStart),
            _ => None,
        }
    }

    /// A single-instruction compiler builtin call target: bit counts,
    /// byte swaps, float rounding, traps, and fences. These have no
    /// library symbol behind them, so they lower inline like the
    /// legacy backend instead of calling out.
    /// Which single-instruction builtin a call target names, if any.
    pub(crate) fn scalar_builtin(&self, id: fe::sema::DeclId) -> Option<ScalarBuiltin> {
        let binding = &self.sema.decls[id.0 as usize];
        if !matches!(binding.kind, fe::sema::BindingKind::Func) {
            return None;
        }
        match binding.name.and_then(|n| self.interner.resolve(n)) {
            Some("__builtin_clz" | "__builtin_clzl" | "__builtin_clzll") => Some(ScalarBuiltin::Clz),
            Some("__builtin_ctz" | "__builtin_ctzl" | "__builtin_ctzll") => Some(ScalarBuiltin::Ctz),
            Some("__builtin_clrsb" | "__builtin_clrsbl" | "__builtin_clrsbll") => Some(ScalarBuiltin::Clrsb),
            Some("__builtin_parity" | "__builtin_parityl" | "__builtin_parityll") => Some(ScalarBuiltin::Parity),
            Some("copysign" | "copysignf" | "copysignl" | "__builtin_copysign" | "__builtin_copysignf" | "__builtin_copysignl") => match binding.ty.unqualified() {
                fe::parse::Ty::Func { params, .. } if params.len() == 2 => Some(ScalarBuiltin::Copysign),
                _ => None,
            },
            Some("__builtin_popcount" | "__builtin_popcountl" | "__builtin_popcountll") => Some(ScalarBuiltin::Popcnt),
            Some("__builtin_bswap16" | "__builtin_bswap32" | "__builtin_bswap64") => Some(ScalarBuiltin::Bswap),
            // MSVC's `<intrin.h>` spellings of the same byte swaps.
            Some("_byteswap_ushort" | "_byteswap_ulong" | "_byteswap_uint64") => Some(ScalarBuiltin::Bswap),
            Some("__builtin_ceil" | "__builtin_ceill") => Some(ScalarBuiltin::Ceil),
            Some("__builtin_floor" | "__builtin_floorl") => Some(ScalarBuiltin::Floor),
            Some("__builtin_trap") => Some(ScalarBuiltin::Trap),
            Some("__sync_synchronize" | "_ReadWriteBarrier") => Some(ScalarBuiltin::Sync),
            // The `abs` family and the libm rounding builtins expand
            // inline, like GCC: a program may still define the names,
            // but direct calls use the builtin. A declaration with a
            // different shape disables the expansion.
            Some("abs" | "__builtin_abs") => unary_builtin(binding, ScalarBuiltin::Abs),
            Some("labs" | "__builtin_labs") => unary_builtin(binding, ScalarBuiltin::Abs),
            Some("llabs" | "__builtin_llabs") => unary_builtin(binding, ScalarBuiltin::Abs),
            Some("fabs" | "fabsf" | "fabsl" | "__builtin_fabs" | "__builtin_fabsf" | "__builtin_fabsl") => unary_builtin(binding, ScalarBuiltin::Abs),
            Some("floor" | "floorf" | "floorl" | "__builtin_floorf") => unary_builtin(binding, ScalarBuiltin::Floor),
            Some("ceil" | "ceilf" | "ceill" | "__builtin_ceilf") => unary_builtin(binding, ScalarBuiltin::Ceil),
            _ => None,
        }
    }

    /// A NaN/infinity builtin call target folded to a constant of
    /// the result type, or `None`. Covers every spelling the parser
    /// declares (`nanf`/`nan`/`nanl`, `inff`/`inf`/`infl`,
    /// `huge_valf`/`huge_val`/`huge_vall`).
    pub(crate) fn fp_const_builtin(&self, id: fe::sema::DeclId, mir_ty: &mir::Ty, func: &mut mir::Function) -> Option<mir::Value> {
        let binding = &self.sema.decls[id.0 as usize];
        if !matches!(binding.kind, fe::sema::BindingKind::Func) {
            return None;
        }
        let name = binding.name.and_then(|n| self.interner.resolve(n))?;
        let nan = matches!(name, "__builtin_nanf" | "__builtin_nan" | "__builtin_nanl");
        let inf = matches!(
            name,
            "__builtin_inff" | "__builtin_inf" | "__builtin_infl" | "__builtin_huge_valf" | "__builtin_huge_val" | "__builtin_huge_vall"
        );
        if !nan && !inf {
            return None;
        }
        let mir::Ty::Float(kind) = mir_ty else {
            return None;
        };
        let bits = match kind {
            mir::FloatKind::F32 => u128::from(if nan { 0x7fc00000u32 } else { 0x7f800000u32 }),
            mir::FloatKind::F64 => u128::from(if nan { 0x7ff8000000000000u64 } else { 0x7ff0000000000000u64 }),
            mir::FloatKind::F80 | mir::FloatKind::F128 => {
                let ld = if matches!(kind, mir::FloatKind::F80) {
                    fe::target::LongDoubleKind::F80
                } else {
                    fe::target::LongDoubleKind::F128
                };
                let value = if nan {
                    fe::eval::FloatValue::long_double_nan_in(ld)
                } else {
                    fe::eval::FloatValue::long_double_infinity_in(ld)
                };
                float_bits(value, mir_ty)
            }
        };
        Some(func.push_const(mir::Const::Float { kind: *kind, bits }))
    }

    /// Lower a single-instruction builtin call: evaluate the operand,
    /// apply the operation at the operand width, and convert to the
    /// declared result width (64-bit counts narrow to `int`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lower_scalar_builtin(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        builtin: ScalarBuiltin,
        args: &[fe::check::CheckedExpr],
        fe_ty: &fe::parse::Ty,
        _mir_ty: &mir::Ty,
    ) -> mir::Value {
        // `trap` terminates the block, so materialize the dummy result
        // first: anything appended after a terminator panics.
        if matches!(builtin, ScalarBuiltin::Trap) {
            let dummy = self.iconst(func, 32, 0);
            self.terminate(func, frame, mir::Terminator::Unreachable);
            return dummy;
        }
        if matches!(builtin, ScalarBuiltin::Sync) {
            for arg in args {
                self.expr_value(module, func, frame, funcs, externs, globals, arg);
            }
            self.emit_unit(func, frame, mir::InstKind::AtomicFence);
            return self.iconst(func, 32, 0);
        }
        // `abs(x)` / `fabs(x)`: the magnitude with a single evaluation.
        // Floats clear the sign bit (a select would keep `-0.0`);
        // integers are `x < 0 ? -x : x`, where the negative extreme
        // wraps, like the machine sequence GCC emits.
        if matches!(builtin, ScalarBuiltin::Abs) {
            let arg = &args[0];
            let operand_mir = lower_ty(&arg.ty, self.sema, self.interner);
            let value = self.expr_value(module, func, frame, funcs, externs, globals, arg);
            if let mir::Ty::Float(_) = operand_mir {
                let out = self.float_magnitude(func, frame, value, &operand_mir);
                return self.convert(func, frame, out, &arg.ty, fe_ty);
            }
            let bits = int_bits(&operand_mir);
            let zero = self.iconst(func, bits, 0);
            let neg = self.emit(func, frame, mir::InstKind::Neg(value), operand_mir.clone());
            let is_neg = self.emit(
                func,
                frame,
                mir::InstKind::Icmp {
                    pred: mir::IcmpPred::Slt,
                    lhs: value,
                    rhs: zero,
                },
                mir::Ty::Bool,
            );
            let raw = self.emit(
                func,
                frame,
                mir::InstKind::Select {
                    cond: is_neg,
                    then_val: neg,
                    else_val: value,
                },
                operand_mir.clone(),
            );
            return self.convert(func, frame, raw, &arg.ty, fe_ty);
        }
        // `clrsb(x)`: leading redundant sign bits. `x ^ (x >> w-1)` is
        // `x` for a non-negative operand and `~x` for a negative one
        // (the shift is arithmetic), so `clz` of it counts the sign
        // extension; `clz(0) == w` makes a zero operand count `w - 1`.
        if matches!(builtin, ScalarBuiltin::Clrsb) {
            let arg = &args[0];
            let operand_mir = lower_ty(&arg.ty, self.sema, self.interner);
            let value = self.expr_value(module, func, frame, funcs, externs, globals, arg);
            let bits = (operand_mir.size_bytes() * 8) as u32;
            let shift = self.iconst(func, bits, (bits - 1) as i128);
            let sign = self.emit(func, frame, mir::InstKind::AShr(value, shift), operand_mir.clone());
            let folded = self.emit(func, frame, mir::InstKind::Xor(value, sign), operand_mir.clone());
            let leading = self.emit(func, frame, mir::InstKind::Clz(folded), operand_mir.clone());
            let one = self.iconst(func, bits, 1);
            let raw = self.emit(func, frame, mir::InstKind::Sub(leading, one), operand_mir.clone());
            return self.convert(func, frame, raw, &arg.ty, fe_ty);
        }
        // `parity(x)`: `int` 0/1 of the low bit of the popcount.
        if matches!(builtin, ScalarBuiltin::Parity) {
            let arg = &args[0];
            let operand_mir = lower_ty(&arg.ty, self.sema, self.interner);
            let value = self.expr_value(module, func, frame, funcs, externs, globals, arg);
            let bits = (operand_mir.size_bytes() * 8) as u32;
            let count = self.emit(func, frame, mir::InstKind::Popcnt(value), operand_mir.clone());
            let one = self.iconst(func, bits, 1);
            let raw = self.emit(func, frame, mir::InstKind::And(count, one), operand_mir.clone());
            return self.convert(func, frame, raw, &arg.ty, fe_ty);
        }
        // `copysign(x, y)`: x's magnitude with y's sign bit. Bit-level:
        // `(bits_x ^ (bits_x & mask)) | (bits_y & mask)` where the mask
        // is the top bit of the integer form (bit 79 for x87 long
        // doubles, whose register form keeps the 80-bit pattern padded
        // to 128 bits).
        if matches!(builtin, ScalarBuiltin::Copysign) {
            let (x, y) = (&args[0], &args[1]);
            let operand_mir = lower_ty(&x.ty, self.sema, self.interner);
            let (int_ty, top) = float_int_form(&operand_mir);
            let bits = int_bits(&int_ty);
            let xv = self.expr_value(module, func, frame, funcs, externs, globals, x);
            let yv = self.expr_value(module, func, frame, funcs, externs, globals, y);
            let xb = self.emit(func, frame, mir::InstKind::Bitcast(xv), int_ty.clone());
            let yb = self.emit(func, frame, mir::InstKind::Bitcast(yv), int_ty.clone());
            let one = self.iconst(func, bits, 1);
            let shift = self.iconst(func, bits, i128::from(top));
            let mask = self.emit(func, frame, mir::InstKind::Shl(one, shift), int_ty.clone());
            let x_sign = self.emit(func, frame, mir::InstKind::And(xb, mask), int_ty.clone());
            let cleared = self.emit(func, frame, mir::InstKind::Xor(xb, x_sign), int_ty.clone());
            let y_sign = self.emit(func, frame, mir::InstKind::And(yb, mask), int_ty.clone());
            let combined = self.emit(func, frame, mir::InstKind::Or(cleared, y_sign), int_ty.clone());
            let raw = self.emit(func, frame, mir::InstKind::Bitcast(combined), operand_mir.clone());
            return self.convert(func, frame, raw, &x.ty, fe_ty);
        }
        let arg = &args[0];
        let operand_mir = lower_ty(&arg.ty, self.sema, self.interner);
        let value = self.expr_value(module, func, frame, funcs, externs, globals, arg);
        let kind = match builtin {
            ScalarBuiltin::Clz => mir::InstKind::Clz(value),
            ScalarBuiltin::Ctz => mir::InstKind::Ctz(value),
            ScalarBuiltin::Popcnt => mir::InstKind::Popcnt(value),
            ScalarBuiltin::Bswap => mir::InstKind::Bswap(value),
            ScalarBuiltin::Ceil => mir::InstKind::Ceil(value),
            ScalarBuiltin::Floor => mir::InstKind::Floor(value),
            ScalarBuiltin::Trap | ScalarBuiltin::Sync | ScalarBuiltin::Abs | ScalarBuiltin::Clrsb | ScalarBuiltin::Parity | ScalarBuiltin::Copysign => {
                unreachable!("handled above")
            }
        };
        let raw = self.emit(func, frame, kind, operand_mir.clone());
        self.convert(func, frame, raw, &arg.ty, fe_ty)
    }

    /// Which complex accessor `id` names, if any: the real part,
    /// imaginary part, conjugate, or magnitude getters (plain, `f`/`l`
    /// suffixed, and `__builtin_` spellings).
    pub(crate) fn complex_builtin(&self, id: fe::sema::DeclId) -> Option<ComplexBuiltin> {
        let binding = &self.sema.decls[id.0 as usize];
        if !matches!(binding.kind, fe::sema::BindingKind::Func) {
            return None;
        }
        Some(match binding.name.and_then(|n| self.interner.resolve(n))? {
            "creal" | "crealf" | "creall" | "__builtin_creal" | "__builtin_crealf" | "__builtin_creall" => ComplexBuiltin::Real,
            "cimag" | "cimagf" | "cimagl" | "__builtin_cimag" | "__builtin_cimagf" | "__builtin_cimagl" => ComplexBuiltin::Imag,
            "conj" | "conjf" | "conjl" | "__builtin_conj" | "__builtin_conjf" | "__builtin_conjl" => ComplexBuiltin::Conj,
            "cabs" | "cabsf" | "cabsl" | "__builtin_cabs" | "__builtin_cabsf" | "__builtin_cabsl" => ComplexBuiltin::Abs,
            _ => return None,
        })
    }

    /// Lower a complex accessor call directly off the components, or
    /// `None` when the callee's signature is not the exact glibc one
    /// (so same-named user functions still call out normally).
    /// Compiler builtins keep their declared prototypes, which always
    /// match; magnitudes over integer components never match (no
    /// glibc signature takes them, mirroring the legacy backend).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lower_complex_builtin(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        id: fe::sema::DeclId,
        args: &[fe::check::CheckedExpr],
    ) -> Option<mir::Value> {
        let op = self.complex_builtin(id)?;
        let name = self.sema.decls[id.0 as usize].name.and_then(|n| self.interner.resolve(n))?;
        // Compiler builtins need no signature check: the parser
        // declares their exact prototypes.
        if !name.starts_with("__builtin_") {
            let binding = &self.sema.decls[id.0 as usize];
            let fe::parse::Ty::Func { params, ret, .. } = binding.ty.unqualified() else {
                return None;
            };
            let [param] = params.as_slice() else {
                return None;
            };
            let fe::parse::Ty::Complex(inner) = param.unqualified() else {
                return None;
            };
            let comp = inner.as_ref().clone();
            let want_ret = match op {
                ComplexBuiltin::Conj => param.clone(),
                ComplexBuiltin::Real | ComplexBuiltin::Imag | ComplexBuiltin::Abs => comp.clone(),
            };
            if ret.unqualified() != want_ret.unqualified() {
                return None;
            }
            if matches!(op, ComplexBuiltin::Abs) && !matches!(comp.unqualified(), fe::parse::Ty::Float | fe::parse::Ty::Double | fe::parse::Ty::LongDouble) {
                return None;
            }
        }
        let arg = args.first()?;
        if !is_complex(&arg.ty) {
            return None;
        }
        let lowered = self.expr_value(module, func, frame, funcs, externs, globals, arg);
        match op {
            ComplexBuiltin::Real | ComplexBuiltin::Imag => {
                let index = if matches!(op, ComplexBuiltin::Real) { 0 } else { 1 };
                Some(self.complex_extract(func, frame, lowered, &arg.ty, index))
            }
            ComplexBuiltin::Conj => Some(self.complex_conj(func, frame, lowered, &arg.ty)),
            ComplexBuiltin::Abs => {
                let comp_fe = complex_component(&arg.ty);
                let comp_mir = lower_ty(&comp_fe, self.sema, self.interner);
                if !matches!(comp_mir, mir::Ty::Float(_)) {
                    return None;
                }
                let a = self.complex_extract(func, frame, lowered, &arg.ty, 0);
                let b = self.complex_extract(func, frame, lowered, &arg.ty, 1);
                let aa = self.emit(func, frame, mir::InstKind::FMul(a, a), comp_mir.clone());
                let bb = self.emit(func, frame, mir::InstKind::FMul(b, b), comp_mir.clone());
                let sum = self.emit(func, frame, mir::InstKind::FAdd(aa, bb), comp_mir.clone());
                Some(self.emit(func, frame, mir::InstKind::Sqrt(sum), comp_mir))
            }
        }
    }

    /// Lower a `va_*` builtin call to its MIR instruction. The list
    /// operand lowers to its address either way.
    pub(crate) fn lower_va_builtin(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        builtin: VaBuiltin,
        args: &[fe::check::CheckedExpr],
        mir_ty: &mir::Ty,
    ) -> mir::Value {
        let kind = match builtin {
            VaBuiltin::Start => mir::InstKind::VaStart {
                ap: self.place(module, func, frame, funcs, externs, globals, &args[0]).0,
            },
            // `__va_start(&ap, x)`: peel the `&` so the list place is
            // the operand, exactly like `__builtin_va_start(ap, x)`.
            VaBuiltin::MsvcStart => {
                let fe::check::CheckedExprKind::Unary(fe::parse::UnaryOp::AddrOf, ap_expr) = &args[0].kind else {
                    let span = args[0].span;
                    panic!("__va_start expects &va_list as its first argument at {span:?}");
                };
                mir::InstKind::VaStart {
                    ap: self.place(module, func, frame, funcs, externs, globals, ap_expr).0,
                }
            }
            VaBuiltin::End => mir::InstKind::VaEnd {
                ap: self.place(module, func, frame, funcs, externs, globals, &args[0]).0,
            },
            VaBuiltin::Copy => mir::InstKind::VaCopy {
                dst: self.place(module, func, frame, funcs, externs, globals, &args[0]).0,
                src: self.place(module, func, frame, funcs, externs, globals, &args[1]).0,
            },
        };
        // Builtins are `void`; fabricate the unit value statements discard.
        if mir_ty.is_void() {
            self.emit_unit(func, frame, kind);
            self.iconst(func, 32, 0)
        } else {
            self.emit(func, frame, kind, mir_ty.clone())
        }
    }

}
