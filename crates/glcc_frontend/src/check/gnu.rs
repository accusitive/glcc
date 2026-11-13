//! One-off GNU builtins that check inline instead of resolving to a
//! declared prototype: type-generic queries (`clzg`, `ctzg`,
//! `constant_p`, `classify_type`, the float classifiers,
//! `fpclassify`, rotation), expression forms whose operands must not
//! undergo default promotions (`choose_expr`, `overflow`), and the
//! few that touch the frame (`alloca`, `prefetch`, frame/return
//! addresses, the async `await`).

use super::*;

/// Which floating-point classification a `FpClass` node computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpPredicate {
    IsNan,
    /// `__builtin_isinf`: `1` for either infinity, like the `isinf`
    /// macro.
    IsInf,
    /// `__builtin_isinf_sign`: `1` for `+inf`, `-1` for `-inf`, `0`
    /// otherwise (what `<math.h>`'s `isinf` macro expands to).
    IsInfSign,
    /// `__builtin_isfinite`: `1` for anything but NaN and infinities.
    IsFinite,
    /// `__builtin_isnormal`: `1` for finite nonzero values that are not
    /// subnormal.
    IsNormal,
    /// `__builtin_signbit`: the sign bit, so `-0.0` counts.
    SignBit,
}

/// Which GCC overflow builtin an `Overflow` node computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowOp {
    Add,
    Sub,
    Mul,
}

/// The GCC typed overflow builtins (`__builtin_sadd_overflow`,
/// `__builtin_umulll_overflow`, ...): `[su](add|sub|mul)(l|ll)?`
/// between `__builtin_` and `_overflow`. Each is the generic builtin
/// with its operand type pinned (`s`/`u` and the width suffix).
pub(super) fn typed_overflow_builtin(name: &str) -> Option<(OverflowOp, Ty)> {
    let rest = name.strip_prefix("__builtin_")?;
    let (signed, rest) = match rest.as_bytes().first()? {
        b's' => (true, &rest[1..]),
        b'u' => (false, &rest[1..]),
        _ => return None,
    };
    let (op, rest) = if let Some(rest) = rest.strip_prefix("add") {
        (OverflowOp::Add, rest)
    } else if let Some(rest) = rest.strip_prefix("sub") {
        (OverflowOp::Sub, rest)
    } else {
        let rest = rest.strip_prefix("mul")?;
        (OverflowOp::Mul, rest)
    };
    let width = rest.strip_suffix("_overflow")?;
    let ty = match (signed, width) {
        (true, "") => Ty::Int,
        (true, "l") => Ty::Long,
        (true, "ll") => Ty::LongLong,
        (false, "") => Ty::UnsignedInt,
        (false, "l") => Ty::UnsignedLong,
        (false, "ll") => Ty::UnsignedLongLong,
        _ => return None,
    };
    Some((op, ty))
}

impl<'a> Sema<'a> {
    /// `__builtin_bswap16/32/64(x)`: byte-order reversal at the
    /// prototype's fixed width. The operand converts to the parameter
    /// type *before* anything else — `bswap16(0x0800u)` swaps a `u16`,
    /// i.e. `0x0008`, regardless of the source literal's width — and
    /// constant operands fold in `eval_const` (the kernel's ethertype
    /// `case` labels are `bswap` of literals).
    pub(super) fn check_bswap(&self, name: &str, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let (bits, param) = match name {
            "__builtin_bswap16" => (16, Ty::UnsignedShort),
            "__builtin_bswap32" => (32, Ty::UnsignedInt),
            _ => (64, Ty::UnsignedLongLong),
        };
        let [arg] = args else {
            return Err(CompileError::new(format!("`{name}` takes exactly one argument"), span));
        };
        let checked = self.convert_for_assignment(&param, self.check_expr(arg, interner)?, interner)?;
        Ok(CheckedExpr {
            ty: param,
            span,
            kind: CheckedExprKind::Bswap {
                arg: self.alloc(checked),
                bits,
            },
        })
    }

    /// `__builtin_await(handle)` (glcc extension): checks the operand
    /// to be a `_Future(T)` handle and yields a node of type `T`. Only
    /// valid inside an async function — elsewhere there is no
    /// coroutine to suspend, and the lowering rejects it anyway.
    pub(super) fn check_await(&self, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        if !self.async_enabled {
            return Err(CompileError::new("`__builtin_await` requires the glcc async extension (-fglcc-async)", span));
        }
        if !self.checking_async_fn.get() {
            return Err(CompileError::new("`__builtin_await` is only valid inside an async function", span));
        }
        if args.len() != 1 {
            return Err(CompileError::new(format!("`__builtin_await` takes exactly one argument, got {}", args.len()), span));
        }
        let arg = self.check_expr(&args[0], interner)?;
        let Some(result) = self.future_result_of(&arg.ty) else {
            return Err(CompileError::new("argument of `__builtin_await` is not a future", arg.span));
        };
        Ok(CheckedExpr {
            ty: result,
            span,
            kind: CheckedExprKind::Await(self.alloc(arg)),
        })
    }

    /// `int __builtin_clzg(x[, fallback])` (and `ctzg`): the first
    /// argument is any unsigned integer type (standard, extended, or
    /// bit-precise) and keeps its width — no promotion — so the count
    /// is within the argument's own bits. With two arguments, a zero
    /// first argument yields the second (an `int`); with one, zero is
    /// undefined (like the fixed-width `clz`, which shares the
    /// backend lowering shape).
    pub(super) fn check_clzg(&self, args: &[Expr<'a>], interner: &Interner, span: Span, ctz: bool) -> Result<CheckedExpr<'a>, CompileError> {
        let what = if ctz { "`__builtin_ctzg`" } else { "`__builtin_clzg`" };
        if args.is_empty() || args.len() > 2 {
            return Err(CompileError::new(format!("{what} takes one unsigned argument and an optional `int` fallback"), span));
        }
        let arg = self.decay_checked(self.check_expr(&args[0], interner)?);
        if !matches!(
            arg.ty.unqualified(),
            Ty::Bool | Ty::UnsignedChar | Ty::UnsignedShort | Ty::UnsignedInt | Ty::UnsignedLong | Ty::UnsignedLongLong | Ty::UnsignedInt128
        ) && !matches!(arg.ty.unqualified(), Ty::BitInt { signed: false, .. })
        {
            return Err(CompileError::new(format!("{what} requires an unsigned integer argument"), arg.span));
        }
        let fallback = match args.get(1) {
            Some(fb) => Some(self.alloc(self.convert_for_assignment(&Ty::Int, self.check_expr(fb, interner)?, interner)?)),
            None => None,
        };
        Ok(CheckedExpr {
            ty: Ty::Int,
            span,
            kind: CheckedExprKind::Clz {
                ctz,
                arg: self.alloc(arg),
                fallback,
            },
        })
    }

    /// `__builtin_stdc_rotate_left`/`_right`: type-generic rotation of
    /// an unsigned integer at its own width. A bit-field keeps its
    /// declared type and width (GCC rotates `unsigned s : 5` over five
    /// bits rather than the promoted `int`). The count may be any
    /// integer except an enum and keeps enough width that reducing it
    /// modulo the operand width does not lose count bits.
    pub(super) fn check_rotate(&self, args: &[Expr<'a>], interner: &Interner, span: Span, left: bool) -> Result<CheckedExpr<'a>, CompileError> {
        let what = if left { "`__builtin_stdc_rotate_left`" } else { "`__builtin_stdc_rotate_right`" };
        let [arg, count] = args else {
            return Err(CompileError::new(format!("{what} takes exactly two arguments"), span));
        };
        let arg = self.decay_checked(self.check_expr(arg, interner)?);
        let (ty, bits) = match &arg.kind {
            CheckedExprKind::Member {
                bitfield: Some((_, width)),
                decl_ty,
                ..
            } => {
                if !is_unsigned_integer(decl_ty) {
                    return Err(CompileError::new(format!("{what} requires an unsigned integer argument"), arg.span));
                }
                (decl_ty.unqualified().clone(), *width)
            }
            _ => {
                if !is_unsigned_integer(&arg.ty) {
                    return Err(CompileError::new(format!("{what} requires an unsigned integer argument"), arg.span));
                }
                let ty = arg.ty.unqualified().clone();
                let bits = match ty {
                    Ty::BitInt { width, .. } => width,
                    ref ty => u32::from(int_type_size(ty, &self.target).expect("unsigned integer has a size")) * 8,
                };
                (ty, bits)
            }
        };
        let count = self.check_expr(count, interner)?;
        let count_ty = count.ty.unqualified().clone();
        if !is_integer(&count_ty) || matches!(count_ty, Ty::Enum(_)) {
            return Err(CompileError::new(format!("{what} requires an integer count"), count.span));
        }
        // Keep the count's value: narrow types widen losslessly to
        // `unsigned int`, while wide signed types switch to their
        // unsigned counterpart so the modulo stays bit-exact.
        let count_ty = if is_signed_int_in(&count_ty, self.target.char_signed).unwrap_or(false) {
            corresponding_unsigned(&count_ty)
        } else {
            count_ty
        };
        let count_ty = if int_type_size(&count_ty, &self.target).is_some_and(|size| size < 4) {
            Ty::UnsignedInt
        } else {
            count_ty
        };
        let count = self.convert_for_assignment(&count_ty, count, interner)?;
        Ok(CheckedExpr {
            ty,
            span,
            kind: CheckedExprKind::Rotate {
                left,
                arg: self.alloc(arg),
                count: self.alloc(count),
                bits,
            },
        })
    }

    /// `int __builtin_constant_p(exp)`: whether `exp` is known to be
    /// constant at compile time. The operand checks normally but is
    /// then discarded (GCC never evaluates it, side effects
    /// included); only the folded 0/1 remains, so the node itself is
    /// always constant.
    pub(super) fn check_constant_p(&self, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let [arg] = args else {
            return Err(CompileError::new("`__builtin_constant_p` takes exactly one argument", span));
        };
        let checked = self.decay_checked(self.check_expr(arg, interner)?);
        // A string literal is a constant-pool entry: GCC reports it
        // constant even though its value is an address (`&global` is
        // not).
        let is_const = matches!(checked.kind, CheckedExprKind::Str(_)) || crate::eval::eval_const(&checked, self, interner).is_some();
        Ok(CheckedExpr {
            ty: Ty::Int,
            span,
            kind: CheckedExprKind::ConstantP(is_const),
        })
    }

    /// The single-argument floating-point classifications
    /// (`__builtin_isnan`, `__builtin_isinf`, `__builtin_isinf_sign`,
    /// `__builtin_isfinite`, `__builtin_isnormal`): single evaluation of
    /// the operand, unlike an `x != x` desugaring. Integers are never
    /// NaN, infinite or subnormal, so they fold to `0` immediately
    /// (`signbit` is the exception: its `double` parameter makes the
    /// sign of `-1` count); anything non-arithmetic is rejected.
    pub(super) fn check_fp_class(&self, args: &[Expr<'a>], interner: &Interner, span: Span, op: FpPredicate, what: &str) -> Result<CheckedExpr<'a>, CompileError> {
        let [arg] = args else {
            return Err(CompileError::new(format!("`{what}` takes exactly one argument"), span));
        };
        let arg = self.decay_checked(self.check_expr(arg, interner)?);
        if !is_arithmetic(&arg.ty) {
            return Err(CompileError::new(format!("`{what}` needs an arithmetic argument"), arg.span));
        }
        if is_integer(&arg.ty) && op != FpPredicate::SignBit {
            return Ok(CheckedExpr {
                ty: Ty::Int,
                span,
                kind: CheckedExprKind::ConstantP(false),
            });
        }
        // `__builtin_signbit`'s parameter is a `double`, so an integer
        // converts first and its sign decides (`signbit(-1)` is true).
        let arg = if is_integer(&arg.ty) {
            self.convert_for_assignment(&Ty::Double, arg, interner)?
        } else {
            arg
        };
        Ok(CheckedExpr {
            ty: Ty::Int,
            span,
            kind: CheckedExprKind::FpClass { op, arg: self.alloc(arg) },
        })
    }

    /// `int __builtin_fpclassify(nan, inf, normal, subnormal, zero, x)`:
    /// the `int` code for `x`'s category, with the five codes supplied
    /// by the caller (`<math.h>`'s `fpclassify` macro passes `FP_NAN`,
    /// `FP_INFINITE`, `FP_NORMAL`, `FP_SUBNORMAL` and `FP_ZERO`). GCC
    /// wants an integer constant for each code; any `int` expression is
    /// accepted here. The operand must be real floating point (GCC
    /// rejects integers and complex; integers would have a constant
    /// category) and evaluates once.
    pub(super) fn check_fpclassify(&self, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let [nan, inf, normal, subnormal, zero, arg] = args else {
            return Err(CompileError::new("`__builtin_fpclassify` takes five `int` codes and a floating-point operand", span));
        };
        let code = |expr: &Expr<'a>| -> Result<CheckedExpr<'a>, CompileError> {
            let checked = self.check_expr(expr, interner)?;
            self.convert_for_assignment(&Ty::Int, checked, interner)
        };
        let codes = [
            self.alloc(code(nan)?),
            self.alloc(code(inf)?),
            self.alloc(code(normal)?),
            self.alloc(code(subnormal)?),
            self.alloc(code(zero)?),
        ];
        let arg = self.decay_checked(self.check_expr(arg, interner)?);
        if !is_real_float(&arg.ty) {
            return Err(CompileError::new("`__builtin_fpclassify` needs a floating-point operand", arg.span));
        }
        Ok(CheckedExpr {
            ty: Ty::Int,
            span,
            kind: CheckedExprKind::FpClassify { codes, arg: self.alloc(arg) },
        })
    }

    /// `__builtin_choose_expr(cond, a, b)`: `cond` must fold to an
    /// integer constant; the result is the chosen operand with its own
    /// type, and only that one is checked — GCC leaves the other
    /// unevaluated, so it need not even be valid.
    pub(super) fn check_choose_expr(&self, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let [cond, then_expr, else_expr] = args else {
            return Err(CompileError::new("`__builtin_choose_expr` takes three arguments", span));
        };
        let checked = self.decay_checked(self.check_expr(cond, interner)?);
        let Some(value) = crate::eval::eval_const(&checked, self, interner) else {
            return Err(CompileError::new("`__builtin_choose_expr` condition is not a constant", span));
        };
        self.check_expr(if crate::eval::truthy(&value) { then_expr } else { else_expr }, interner)
    }

    /// `int __builtin_classify_type(x)`: the type class of the
    /// argument's type, folded to a constant. The argument is checked
    /// (undeclared names and type errors still surface) but never
    /// evaluated, like GCC documents.
    pub(super) fn check_classify_type(&self, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let [arg] = args else {
            return Err(CompileError::new("`__builtin_classify_type` takes exactly one argument", span));
        };
        let arg = self.check_expr(arg, interner)?;
        Ok(CheckedExpr {
            ty: Ty::Int,
            span,
            kind: CheckedExprKind::ClassifyType(self.type_class(&arg.ty)),
        })
    }

    /// GNU `__builtin_shuffle(v, mask)` / `(v0, v1, mask)`: lane `i`
    /// of the result comes from lane `mask[i]` of `v0` (indices below
    /// the lane count) or `v1` (above it). The operands are same-shaped
    /// vectors and the mask an integer vector with the same lane count;
    /// GCC requires the lane indices be in range, which is not checked
    /// here (out-of-range lanes are UB, like GCC).
    pub(super) fn check_shuffle(&self, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let (argc, checked) = match args.len() {
            2 => (2, args.iter().map(|a| self.check_expr(a, interner)).collect::<Result<Vec<_>, _>>()?),
            3 => (3, args.iter().map(|a| self.check_expr(a, interner)).collect::<Result<Vec<_>, _>>()?),
            _ => return Err(CompileError::new("`__builtin_shuffle` takes two or three arguments", span)),
        };
        let (v0, mask) = (&checked[0], &checked[argc - 1]);
        let (Ty::Vector { elem, len }, Ty::Vector { elem: mask_elem, len: mask_len }) = (v0.ty.unqualified(), mask.ty.unqualified()) else {
            return Err(CompileError::new("`__builtin_shuffle` needs vector operands", span));
        };
        if argc == 3 {
            let v1 = &checked[1];
            let Ty::Vector { elem: e1, len: l1 } = v1.ty.unqualified() else {
                return Err(CompileError::new("`__builtin_shuffle` needs vector operands", span));
            };
            if e1 != elem || l1 != len {
                return Err(CompileError::new("`__builtin_shuffle` vectors must have the same shape", span));
            }
        }
        if mask_len != len || !is_integer(mask_elem) {
            return Err(CompileError::new("`__builtin_shuffle` mask must be an integer vector of the same lane count", span));
        }
        Ok(CheckedExpr {
            ty: v0.ty.clone(),
            span,
            kind: CheckedExprKind::Shuffle {
                args: self.alloc_slice(checked),
            },
        })
    }

    /// Clang `__builtin_shufflevector(v0[, v1], idx...)`: one lane per
    /// constant index over the concatenated operands (`-1` and
    /// out-of-range indices are undef, like clang's poison). The
    /// two-vector form is recognized by its second argument being a
    /// vector; the single-vector form indexes `v0` alone.
    pub(super) fn check_shufflevector(&self, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        if args.len() < 2 {
            return Err(CompileError::new("`__builtin_shufflevector` takes at least two arguments", span));
        }
        let checked = args.iter().map(|a| self.check_expr(a, interner)).collect::<Result<Vec<_>, _>>()?;
        let Ty::Vector { elem, .. } = checked[0].ty.unqualified() else {
            return Err(CompileError::new("`__builtin_shufflevector` needs a vector operand", span));
        };
        let (operand_count, first_idx) = match checked[1].ty.unqualified() {
            Ty::Vector { .. } => (2, 2),
            _ => (1, 1),
        };
        if checked.len() <= first_idx {
            return Err(CompileError::new("`__builtin_shufflevector` needs at least one index", span));
        }
        if operand_count == 2 {
            let Ty::Vector { elem: e1, .. } = checked[1].ty.unqualified() else {
                return Err(CompileError::new("`__builtin_shufflevector` needs vector operands", span));
            };
            if e1 != elem {
                return Err(CompileError::new("`__builtin_shufflevector` vectors must have the same element type", span));
            }
        }
        let mut indices = Vec::with_capacity(checked.len() - first_idx);
        for arg in &checked[first_idx..] {
            let crate::eval::Value::Int(idx) = crate::eval::eval_const(arg, self, interner).ok_or_else(|| CompileError::new("`__builtin_shufflevector` indices must be constants", arg.span))? else {
                return Err(CompileError::new("`__builtin_shufflevector` indices must be integers", arg.span));
            };
            indices.push(idx);
        }
        Ok(CheckedExpr {
            ty: Ty::Vector { elem: elem.clone(), len: indices.len() as u32 },
            span,
            kind: CheckedExprKind::ShuffleVector {
                args: self.alloc_slice(checked[..operand_count].iter().cloned()),
                indices: self.alloc_slice(indices),
            },
        })
    }

    /// Clang `__builtin_convertvector(v, T)`: element-wise conversion
    /// to the target vector type (same lane count, any element
    /// kinds). The type argument arrives as a typedef identifier, so
    /// it resolves through the symbol table, not the expression
    /// checker.
    pub(super) fn check_convertvector(&self, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        if args.len() != 2 {
            return Err(CompileError::new("`__builtin_convertvector` takes two arguments", span));
        }
        let v = self.check_expr(&args[0], interner)?;
        let Ty::Vector { len: src_len, .. } = v.ty.unqualified() else {
            return Err(CompileError::new("`__builtin_convertvector` needs a vector operand", v.span));
        };
        let Expr::Ident { name, .. } = &args[1] else {
            return Err(CompileError::new("`__builtin_convertvector` target must be a type name", span));
        };
        let Some(id) = self.lookup_symbol(*name) else {
            return Err(CompileError::new("`__builtin_convertvector` target must be a type name", span));
        };
        let binding = &self.decls[id.0 as usize];
        if binding.kind != BindingKind::Typedef {
            return Err(CompileError::new("`__builtin_convertvector` target must be a type name", span));
        }
        let ty = binding.ty.clone();
        let Ty::Vector { len: dst_len, .. } = ty.unqualified() else {
            return Err(CompileError::new("`__builtin_convertvector` target must be a vector type", span));
        };
        if src_len != dst_len {
            return Err(CompileError::new("`__builtin_convertvector` lane counts must match", span));
        }
        Ok(CheckedExpr {
            ty,
            span,
            kind: CheckedExprKind::Ia32 {
                op: Ia32Op::ConvertVector,
                args: self.alloc_slice([v]),
            },
        })
    }

    /// Clang `__builtin_nontemporal_store(v, p)`: a non-temporal
    /// store, an ordinary store in our lowering.
    pub(super) fn check_nontemporal_store(&self, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let [v, p] = args else {
            return Err(CompileError::new("`__builtin_nontemporal_store` takes two arguments", span));
        };
        let v = self.check_expr(v, interner)?;
        let p = self.decay_checked(self.check_expr(p, interner)?);
        if !matches!(p.ty.unqualified(), Ty::Pointer { .. }) {
            return Err(CompileError::new("`__builtin_nontemporal_store` needs a pointer target", p.span));
        }
        Ok(CheckedExpr {
            ty: Ty::Void,
            span,
            kind: CheckedExprKind::Ia32 {
                op: Ia32Op::NtStore,
                args: self.alloc_slice([v, p]),
            },
        })
    }

    /// Clang `__builtin_elementwise_*`: the lane-wise operation over
    /// any GNU vector shape, its signedness taken from the element
    /// type (`sqrt` also takes a scalar).
    pub(super) fn check_elementwise(&self, name: &str, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let suffix = name.strip_prefix("__builtin_elementwise_").unwrap_or("");
        let checked = args.iter().map(|a| self.check_expr(a, interner)).collect::<Result<Vec<_>, _>>()?;
        let vector_elem = |c: &CheckedExpr<'a>| match c.ty.unqualified() {
            Ty::Vector { elem, .. } => Some((**elem).clone()),
            _ => None,
        };
        let (op, ret) = match (suffix, checked.len()) {
            ("sqrt", 1) => (Ia32Op::Vec128Sqrt, checked[0].ty.clone()),
            (_, 2) => {
                let (Some(e0), Some(e1)) = (vector_elem(&checked[0]), vector_elem(&checked[1])) else {
                    return Err(CompileError::new(format!("`{name}` needs vector operands"), span));
                };
                if e0 != e1 {
                    return Err(CompileError::new(format!("`{name}` operands must have the same element type"), span));
                }
                let unsigned = !is_signed_int_in(&e0, self.target.char_signed).unwrap_or(true);
                let op = match suffix {
                    "add_sat" => Ia32Op::Vec128Sat { add: true, unsigned },
                    "sub_sat" => Ia32Op::Vec128Sat { add: false, unsigned },
                    "min" => Ia32Op::Vec128MinMax { min: true, signed: !unsigned },
                    "max" => Ia32Op::Vec128MinMax { min: false, signed: !unsigned },
                    _ => unreachable!("dispatch selected a two-argument elementwise builtin"),
                };
                (op, checked[0].ty.clone())
            }
            _ => return Err(CompileError::new(format!("`{name}` takes one or two arguments"), span)),
        };
        Ok(CheckedExpr {
            ty: ret,
            span,
            kind: CheckedExprKind::Ia32 {
                op,
                args: self.alloc_slice(checked),
            },
        })
    }

    /// `void __builtin_prefetch(addr[, rw[, locality]])`: a hint, so
    /// only the address is kept (evaluated and dropped through a
    /// `void` cast); the access/locality arguments are checked and
    /// discarded, like GCC's constant-only ones.
    pub(super) fn check_prefetch(&self, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let Some((addr, rest)) = args.split_first() else {
            return Err(CompileError::new("`__builtin_prefetch` needs an address", span));
        };
        if rest.len() > 2 {
            return Err(CompileError::new("`__builtin_prefetch` takes at most three arguments", span));
        }
        let addr = self.decay_checked(self.check_expr(addr, interner)?);
        if !matches!(addr.ty.unqualified(), Ty::Pointer { .. }) {
            return Err(CompileError::new("`__builtin_prefetch` needs a pointer address", addr.span));
        }
        for arg in rest {
            self.check_expr(arg, interner)?;
        }
        Ok(CheckedExpr {
            ty: Ty::Void,
            span,
            kind: CheckedExprKind::Cast {
                expr: self.alloc(addr),
                ty: Ty::Void,
            },
        })
    }

    /// GCC's overflow builtins: the generic `__builtin_add_overflow`
    /// and friends plus the typed spellings (`__builtin_sadd_overflow`
    /// family, recognized by `typed_overflow_builtin`). Two integer
    /// operands and a pointer receiving the wrapped result. The
    /// pointed-to type fixes the width and signedness the result
    /// converts to; the operation itself is exact, and the builtin
    /// returns whether the exact value fit. A typed spelling pins the
    /// operand type, converting operands like prototyped parameters.
    pub(super) fn check_overflow(&self, op: OverflowOp, args: &[Expr<'a>], interner: &Interner, span: Span, operand_ty: Option<Ty>) -> Result<CheckedExpr<'a>, CompileError> {
        let [lhs, rhs, res] = args else {
            return Err(CompileError::new("overflow builtin needs two operands and a result pointer", span));
        };
        let mut lhs = self.check_expr(lhs, interner)?;
        let mut rhs = self.check_expr(rhs, interner)?;
        for operand in [&lhs, &rhs] {
            if int_rank(&operand.ty).is_none() {
                return Err(CompileError::new("overflow builtin operands must be integers", operand.span));
            }
        }
        if let Some(ty) = &operand_ty {
            lhs = self.convert_for_assignment(ty, lhs, interner)?;
            rhs = self.convert_for_assignment(ty, rhs, interner)?;
        }
        let res = self.decay_checked(self.check_expr(res, interner)?);
        let Ty::Pointer { pointee, .. } = res.ty.unqualified() else {
            return Err(CompileError::new("overflow builtin result must be a pointer to integer", res.span));
        };
        let res_ty = pointee.as_ref().clone();
        if int_rank(&res_ty).is_none() {
            return Err(CompileError::new("overflow builtin result must be a pointer to integer", res.span));
        }
        Ok(CheckedExpr {
            ty: Ty::Int,
            span,
            kind: CheckedExprKind::Overflow {
                op,
                lhs: self.alloc(lhs),
                rhs: self.alloc(rhs),
                res: self.alloc(res),
                res_ty,
            },
        })
    }

    /// `alloca`/`__builtin_alloca`: allocate `size` bytes that stay
    /// live until the enclosing function returns. The frontend only
    /// marks the call; the heap block and its cleanup lower in MIR.
    pub(super) fn check_alloca(&self, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let [size] = args else {
            return Err(CompileError::new("`alloca` needs exactly one argument", span));
        };
        let size = self.check_expr(size, interner)?;
        let size = self.convert_for_assignment(&self.target.size_type(), size, interner)?;
        Ok(CheckedExpr {
            ty: Ty::Pointer {
                qualifiers: TypeQualifiers::empty(),
                size: PtrSize::Default,
                pointee: Box::new(Ty::Void),
            },
            span,
            kind: CheckedExprKind::Alloca { size: self.alloc(size) },
        })
    }

    /// `void *__builtin_frame_address(unsigned level)` and
    /// `__builtin_return_address`: the frame or return address of the
    /// `level`-th enclosing frame. GCC requires the level to be a
    /// constant integer, so a non-constant one is rejected here;
    /// lowering walks the frame-pointer chain from the current frame.
    pub(super) fn check_frame_address(&self, args: &[Expr<'a>], interner: &Interner, span: Span, returns_address: bool) -> Result<CheckedExpr<'a>, CompileError> {
        let what = if returns_address { "__builtin_return_address" } else { "__builtin_frame_address" };
        let [level] = args else {
            return Err(CompileError::new(format!("`{what}` needs exactly one argument"), span));
        };
        let level = self.decay_checked(self.check_expr(level, interner)?);
        let Some(crate::eval::Value::Int(n)) = crate::eval::eval_const(&level, self, interner) else {
            return Err(CompileError::new(format!("`{what}` level must be a constant integer"), level.span));
        };
        let Ok(level) = u64::try_from(n) else {
            return Err(CompileError::new(format!("`{what}` level must be a non-negative constant integer"), level.span));
        };
        Ok(CheckedExpr {
            ty: Ty::Pointer {
                qualifiers: TypeQualifiers::empty(),
                size: PtrSize::Default,
                pointee: Box::new(Ty::Void),
            },
            span,
            kind: CheckedExprKind::FrameAddress { level, returns_address },
        })
    }

    /// GCC's `__builtin_classify_type` codes, in GCC's `typeclass`
    /// order: void 0, integer 1, char 2, enum 3, bool 4, pointer 5,
    /// reference 6, offset 7, real 8, complex 9, function 10, method
    /// 11, record 12, union 13, array 14, string 15. C `char`s are
    /// integers here, like GCC's own classification.
    pub(super) fn type_class(&self, ty: &Ty) -> u8 {
        match ty.unqualified() {
            Ty::Void => 0,
            Ty::Char
            | Ty::SignedChar
            | Ty::UnsignedChar
            | Ty::Short
            | Ty::UnsignedShort
            | Ty::Int
            | Ty::UnsignedInt
            | Ty::Long
            | Ty::UnsignedLong
            | Ty::LongLong
            | Ty::UnsignedLongLong
            | Ty::Int128
            | Ty::UnsignedInt128
            | Ty::BitInt { .. } => 1,
            Ty::Nullptr => 5,
            Ty::Enum(_) => 3,
            Ty::Bool => 4,
            Ty::Pointer { .. } => 5,
            Ty::Float | Ty::Double | Ty::LongDouble | Ty::Float128 => 8,
            Ty::Complex(_) => 9,
            // Vectors are past the manual's table (GCC/clang answer 18
            // for `__m128` in practice).
            Ty::Vector { .. } => 18,
            Ty::Func { .. } => 10,
            // `__builtin_va_list` is an array of one tag on the
            // glcc targets.
            Ty::BuiltinVaList => 14,
            Ty::Struct(id) if self.structs[id.0 as usize].is_union => 13,
            Ty::Struct(_) => 12,
            Ty::Array { .. } => 14,
            // `unqualified` strips these, so this is unreachable.
            Ty::Qualified { inner, .. } => self.type_class(inner),
        }
    }
}
