//! Checking for the legacy `__sync` atomic builtins (GCC §7.9.2, the
//! Intel Itanium ABI spellings). The GNU docs define them in terms of
//! the `__atomic` builtins, so checking maps each name onto an
//! [`AtomicOp`] node from `super::atomic`. There are no memory-order
//! arguments: every operation is a full barrier, i.e. sequentially
//! consistent — what the lowerer does with orders anyway. Each
//! accepts any pointer whose pointee is an integer or pointer type 1,
//! 2, 4, or 8 bytes wide (`_Bool` excluded); pointer operands act as
//! `uintptr_t` (unscaled), which falls out of the atomic ops never
//! scaling addresses.

use super::*;

/// The read-modify-write operation a `__sync` name selects and
/// whether it returns the old (`fetch_and_add`) or the new
/// (`add_and_fetch`) value.
fn sync_rmw(rest: &str) -> Option<(RmwOp, bool)> {
    let (op, fetch) = if let Some(op) = rest.strip_prefix("fetch_and_") {
        (op, true)
    } else {
        (rest.strip_suffix("_and_fetch")?, false)
    };
    let op = match op {
        "add" => RmwOp::Add,
        "sub" => RmwOp::Sub,
        "and" => RmwOp::And,
        "or" => RmwOp::Or,
        "xor" => RmwOp::Xor,
        "nand" => RmwOp::Nand,
        _ => return None,
    };
    Some((op, fetch))
}

impl<'a> Sema<'a> {
    /// A legacy `__sync_*` call, or `None` when `name` is not one.
    pub(super) fn check_sync(&self, name: &str, args: &[Expr<'a>], interner: &Interner, span: Span) -> Option<Result<CheckedExpr<'a>, CompileError>> {
        let rest = name.strip_prefix("__sync_")?;
        if rest == "synchronize" {
            return Some(self.sync_fence(args, interner, span, name));
        }
        if let Some((op, fetch)) = sync_rmw(rest) {
            return Some(self.sync_rmw_call(op, fetch, args, interner, span, name));
        }
        match rest {
            "val_compare_and_swap" => Some(self.sync_cmpxchg(false, args, interner, span, name)),
            "bool_compare_and_swap" => Some(self.sync_cmpxchg(true, args, interner, span, name)),
            "lock_test_and_set" => Some(self.sync_xchg(args, interner, span, name)),
            "lock_release" => Some(self.sync_release(args, interner, span, name)),
            _ => None,
        }
    }

    /// The type a `__sync` operation works on, validating GCC's
    /// constraints: an integer or pointer object 1, 2, 4, or 8 bytes
    /// wide, and not `_Bool`.
    fn sync_op_ty(&self, pointee: &Ty, interner: &Interner, what: &str, span: Span) -> Result<Ty, CompileError> {
        let unqual = pointee.unqualified();
        let usable = !matches!(unqual, Ty::Bool)
            && (matches!(unqual, Ty::Pointer { .. }) || is_integer(pointee))
            && matches!(self.type_size(pointee, interner), Some(1 | 2 | 4 | 8));
        if !usable {
            return Err(CompileError::new(
                format!("`{what}` needs a pointer to an integer or pointer type of 1, 2, 4, or 8 bytes"),
                span,
            ));
        }
        Ok(pointee.clone())
    }

    /// The pointer operand shared by every `__sync` form.
    fn sync_ptr(&self, ptr: &Expr<'a>, interner: &Interner, what: &str) -> Result<(CheckedExpr<'a>, Ty), CompileError> {
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let Ty::Pointer { pointee, .. } = ptr.ty.clone() else {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        };
        let op_ty = self.sync_op_ty(&pointee, interner, what, ptr.span)?;
        Ok((ptr, op_ty))
    }

    /// The value operand, and the ignored trailing arguments (GCC's
    /// "list of variables protected by the memory barrier"): checked
    /// so their side effects still happen, like the `__atomic` order
    /// operands they shadow.
    fn sync_val_and_extras(&self, val: &Expr<'a>, extras: &[Expr<'a>], op_ty: &Ty, interner: &Interner, _what: &str) -> Result<(CheckedExpr<'a>, Vec<CheckedExpr<'a>>), CompileError> {
        // Pointer targets take either an integer (the documented
        // `uintptr_t` treatment, unscaled) or a pointer value: the
        // Itanium signature overloads the value to the pointed-to
        // type, and mingw's `_InterlockedExchangePointer` swaps
        // pointers through `__sync_lock_test_and_set`.
        let val = match op_ty.unqualified() {
            Ty::Pointer { .. } => {
                let v = self.decay_checked(self.check_expr(val, interner)?);
                if is_integer(&v.ty) {
                    v
                } else {
                    self.convert_for_assignment(op_ty, v, interner)?
                }
            }
            _ => self.convert_for_assignment(op_ty, self.check_expr(val, interner)?, interner)?,
        };
        let extras = extras.iter().map(|e| self.check_expr(e, interner).map(|v| self.decay_checked(v))).collect::<Result<Vec<_>, _>>()?;
        Ok((val, extras))
    }

    /// `type __sync_fetch_and_op(T *ptr, T value, ...)` and
    /// `type __sync_op_and_fetch(T *ptr, T value, ...)`: atomic
    /// `{ tmp = *ptr; *ptr op= value; return tmp or *ptr; }` (nand:
    /// `~(tmp & value)`, the GCC 4.4 form).
    fn sync_rmw_call(&self, op: RmwOp, fetch: bool, args: &[Expr<'a>], interner: &Interner, span: Span, what: &str) -> Result<CheckedExpr<'a>, CompileError> {
        let [ptr, val, extras @ ..] = args else {
            return Err(CompileError::new(format!("`{what}` takes a pointer and a value"), span));
        };
        let (ptr, op_ty) = self.sync_ptr(ptr, interner, what)?;
        let (val, extras) = self.sync_val_and_extras(val, extras, &op_ty, interner, what)?;
        Ok(CheckedExpr {
            ty: op_ty,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::Rmw { op, fetch, sized: None },
                args: self.alloc_slice([ptr, val].into_iter().chain(extras)),
            },
        })
    }

    /// `type __sync_{val,bool}_compare_and_swap(T *ptr, T oldval,
    /// T newval, ...)`: atomic compare-and-swap; the `val` form
    /// returns the previous contents, the `bool` form whether the
    /// swap happened.
    fn sync_cmpxchg(&self, want_bool: bool, args: &[Expr<'a>], interner: &Interner, span: Span, what: &str) -> Result<CheckedExpr<'a>, CompileError> {
        let [ptr, expected, desired, extras @ ..] = args else {
            return Err(CompileError::new(format!("`{what}` takes a pointer, the expected value, and the new value"), span));
        };
        let (ptr, op_ty) = self.sync_ptr(ptr, interner, what)?;
        let expected = self.convert_for_assignment(&op_ty, self.check_expr(expected, interner)?, interner)?;
        let desired = self.convert_for_assignment(&op_ty, self.check_expr(desired, interner)?, interner)?;
        let extras = extras.iter().map(|e| self.check_expr(e, interner).map(|v| self.decay_checked(v))).collect::<Result<Vec<_>, _>>()?;
        Ok(CheckedExpr {
            ty: if want_bool { Ty::Bool } else { op_ty },
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::SyncCmpXchg { want_bool },
                args: self.alloc_slice([ptr, expected, desired].into_iter().chain(extras)),
            },
        })
    }

    /// `type __sync_lock_test_and_set(T *ptr, T value, ...)`: an
    /// atomic exchange returning the old value (an acquire barrier;
    /// the lowerer's seq-cst exchange is stricter).
    fn sync_xchg(&self, args: &[Expr<'a>], interner: &Interner, span: Span, what: &str) -> Result<CheckedExpr<'a>, CompileError> {
        let [ptr, val, extras @ ..] = args else {
            return Err(CompileError::new(format!("`{what}` takes a pointer and a value"), span));
        };
        let (ptr, op_ty) = self.sync_ptr(ptr, interner, what)?;
        let (val, extras) = self.sync_val_and_extras(val, extras, &op_ty, interner, what)?;
        Ok(CheckedExpr {
            ty: op_ty,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::ExchangeN { sized: None },
                args: self.alloc_slice([ptr, val].into_iter().chain(extras)),
            },
        })
    }

    /// `void __sync_lock_release(T *ptr, ...)`: release the lock
    /// taken by `lock_test_and_set` by storing zero — the
    /// `AtomicOp::Clear` store, generalized to the pointee width in
    /// the lowerer.
    fn sync_release(&self, args: &[Expr<'a>], interner: &Interner, span: Span, what: &str) -> Result<CheckedExpr<'a>, CompileError> {
        let [ptr, extras @ ..] = args else {
            return Err(CompileError::new(format!("`{what}` takes a pointer"), span));
        };
        let (ptr, _) = self.sync_ptr(ptr, interner, what)?;
        let extras = extras.iter().map(|e| self.check_expr(e, interner).map(|v| self.decay_checked(v))).collect::<Result<Vec<_>, _>>()?;
        Ok(CheckedExpr {
            ty: Ty::Void,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::Clear,
                args: self.alloc_slice([ptr].into_iter().chain(extras)),
            },
        })
    }

    /// `void __sync_synchronize(...)`: a full memory barrier. The
    /// ignored trailing barrier list still evaluates, like the other
    /// `__sync` forms.
    fn sync_fence(&self, args: &[Expr<'a>], interner: &Interner, span: Span, _what: &str) -> Result<CheckedExpr<'a>, CompileError> {
        let extras = args.iter().map(|e| self.check_expr(e, interner).map(|v| self.decay_checked(v))).collect::<Result<Vec<_>, _>>()?;
        Ok(CheckedExpr {
            ty: Ty::Void,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::Fence { signal: false },
                args: self.alloc_slice(extras),
            },
        })
    }
}
