//! Checking for the memory-model-aware atomic builtins: the GCC
//! `__atomic_*` spellings (the `_n` value forms, the generic pointer
//! forms, the sized libatomic spellings, the fetch/put
//! read-modify-write family, fences, lock-free queries,
//! `test_and_set`/`clear`) and the clang `__c11_atomic_*` spellings
//! `<stdatomic.h>` expands to. Everything checks into an
//! [`AtomicOp`] node; the lowerer maps the node onto MIR atomic
//! instructions and treats every memory order as sequentially
//! consistent.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RmwOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Nand,
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicOp {
    LoadN {
        sized: Option<u8>,
    },
    Load,
    /// C11 `__c11_atomic_init(ptr, val)`: a store with no memory order.
    Init,
    StoreN {
        sized: Option<u8>,
    },
    Store,
    ExchangeN {
        sized: Option<u8>,
    },
    Exchange,
    CmpXchgN {
        sized: Option<u8>,
    },
    CmpXchg,
    /// MSVC `_InterlockedCompareExchange*(ptr, exchange, comparand)`:
    /// a strong compare-exchange taking values (not pointers) and
    /// yielding the initial value.
    InterlockedCmpxchg,
    /// Legacy `__sync_val_compare_and_swap` /
    /// `__sync_bool_compare_and_swap`: the same value-taking
    /// compare-exchange, returning the previous value or (with
    /// `want_bool`) whether the swap happened.
    SyncCmpXchg {
        want_bool: bool,
    },
    Rmw {
        op: RmwOp,
        fetch: bool,
        sized: Option<u8>,
    },
    Fence {
        signal: bool,
    },
    LockFree {
        always: bool,
    },
    TestAndSet,
    Clear,
}

impl<'a> Sema<'a> {
    /// All `__atomic_*` builtins: the `_n` forms plus the generic
    /// (pointer-to-value) forms, the sized `_1`/`_2`/`_4`/`_8`/`_16`
    /// libatomic spellings (checked like `_n` with the size verified),
    /// the read-modify-write family (`__atomic_fetch_add`,
    /// `__atomic_add_fetch`, ...), fences, lock-free queries, and
    /// `test_and_set`/`clear`. Memory orders take any integer (GCC
    /// further requires constants, which codegen ignores by mapping
    /// everything to seq-cst).
    pub(super) fn check_atomic(&self, name: &str, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let rest = &name["__atomic_".len()..];
        let (base, sized) = match rest.rsplit_once('_') {
            Some((b, "1")) => (b, Some(1u8)),
            Some((b, "2")) => (b, Some(2u8)),
            Some((b, "4")) => (b, Some(4u8)),
            Some((b, "8")) => (b, Some(8u8)),
            Some((b, "16")) => (b, Some(16u8)),
            Some((b, "n")) => (b, None),
            _ => (rest, None),
        };
        // `_n` was stripped above; `sized` is Some only for `_1` etc.
        // `is_n` marks the explicit `_n` spelling (vs generic with no
        // suffix) for the four ops that have both.
        let is_n = rest.ends_with("_n") || sized.is_some();
        match base {
            "load" if is_n => return self.check_atomic_load_n(args, interner, span, sized, name),
            "load" => return self.check_atomic_load(args, interner, span),
            "store" if is_n => return self.check_atomic_store_n(args, interner, span, sized, name),
            "store" => return self.check_atomic_store(args, interner, span),
            "exchange" if is_n => return self.check_atomic_exchange_n(args, interner, span, sized, name),
            "exchange" => return self.check_atomic_exchange(args, interner, span),
            "compare_exchange" if is_n => return self.check_atomic_cmpxchg_n(args, interner, span, sized, name),
            "compare_exchange" => return self.check_atomic_cmpxchg(args, interner, span),
            "fetch_add" | "fetch_sub" | "fetch_and" | "fetch_or" | "fetch_xor" | "fetch_nand" | "fetch_min" | "fetch_max" => {
                return self.check_atomic_rmw(base, true, args, interner, span, sized, name);
            }
            "add_fetch" | "sub_fetch" | "and_fetch" | "or_fetch" | "xor_fetch" | "nand_fetch" | "min_fetch" | "max_fetch" => {
                return self.check_atomic_rmw(base, false, args, interner, span, sized, name);
            }
            "thread_fence" => return self.check_atomic_fence(args, interner, span, false, name),
            "signal_fence" => return self.check_atomic_fence(args, interner, span, true, name),
            "always_lock_free" => return self.check_atomic_lock_free(args, interner, span, true, name),
            "is_lock_free" => return self.check_atomic_lock_free(args, interner, span, false, name),
            "test_and_set" => return self.check_atomic_test_and_set(args, interner, span),
            "clear" => return self.check_atomic_clear(args, interner, span),
            _ => {}
        }
        Err(CompileError::new(format!("unsupported atomic builtin `{name}`"), span).with_code(DiagCode::UnsupportedAtomicBuiltin))
    }

    /// The clang C11 atomics (`__c11_atomic_*`), what `<stdatomic.h>`
    /// expands to. Unlike the GCC spellings these carry no `_n`/size
    /// suffix and `compare_exchange` drops the `weak` flag (strong vs
    /// weak is in the name), so each maps onto the checker for the
    /// corresponding GCC form.
    pub(super) fn check_c11_atomic(&self, name: &str, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let base = &name["__c11_atomic_".len()..];
        match base {
            "load" => return self.check_atomic_load_n(args, interner, span, None, name),
            "store" => return self.check_atomic_store_n(args, interner, span, None, name),
            "exchange" => return self.check_atomic_exchange_n(args, interner, span, None, name),
            "init" => return self.check_c11_atomic_init(args, interner, span, name),
            "compare_exchange_strong" | "compare_exchange_weak" => {
                return self.check_c11_atomic_cmpxchg(args, interner, span, name);
            }
            "fetch_add" | "fetch_sub" | "fetch_and" | "fetch_or" | "fetch_xor" | "fetch_nand" | "fetch_min" | "fetch_max" => {
                return self.check_atomic_rmw(base, true, args, interner, span, None, name);
            }
            "thread_fence" => return self.check_atomic_fence(args, interner, span, false, name),
            "signal_fence" => return self.check_atomic_fence(args, interner, span, true, name),
            "is_lock_free" => return self.check_atomic_lock_free(args, interner, span, false, name),
            _ => {}
        }
        Err(CompileError::new(format!("unsupported atomic builtin `{name}`"), span).with_code(DiagCode::UnsupportedAtomicBuiltin))
    }

    /// C11 `void __c11_atomic_init(T *ptr, T val)`: like a store with
    /// no memory order. `_Atomic` erases to its base type, so `ptr` is
    /// an ordinary pointer to the object.
    pub(super) fn check_c11_atomic_init(&self, args: &[Expr<'a>], interner: &Interner, span: Span, what: &str) -> Result<CheckedExpr<'a>, CompileError> {
        let [ptr, val] = args else {
            return Err(CompileError::new(format!("`{what}` takes a pointer and a value"), span));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let Ty::Pointer { pointee, .. } = ptr.ty.clone() else {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        };
        let op_ty = self.atomic_n_value_ty(&pointee, None, interner, what, ptr.span)?;
        let val = self.convert_for_assignment(&op_ty, self.check_expr(val, interner)?, interner)?;
        Ok(CheckedExpr {
            ty: Ty::Void,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::Init,
                args: self.alloc_slice([ptr, val]),
            },
        })
    }

    /// C11 `bool __c11_atomic_compare_exchange_strong|weak(T *ptr,
    /// T *expected, T desired, int success, int failure)`: like the GCC
    /// `_n` form without the `weak` flag (already in the name).
    pub(super) fn check_c11_atomic_cmpxchg(&self, args: &[Expr<'a>], interner: &Interner, span: Span, what: &str) -> Result<CheckedExpr<'a>, CompileError> {
        let [ptr, expected, desired, success, failure] = args else {
            return Err(CompileError::new(
                format!("`{what}` takes a pointer, an expected pointer, a desired value, and two memory orders"),
                span,
            ));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let Ty::Pointer { pointee, .. } = ptr.ty.clone() else {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        };
        let op_ty = self.atomic_n_value_ty(&pointee, None, interner, what, ptr.span)?;
        let expected = self.decay_checked(self.check_expr(expected, interner)?);
        let Some(esize) = self.ptr_pointee_size(&expected.ty, interner) else {
            return Err(CompileError::new(format!("`{what}` needs an expected pointer"), expected.span));
        };
        let psize = self.type_size(&op_ty, interner).unwrap_or(0);
        if esize != psize {
            return Err(CompileError::new(format!("`{what}` expected pointer disagrees in size"), expected.span));
        }
        let desired = self.convert_for_assignment(&op_ty, self.check_expr(desired, interner)?, interner)?;
        let success = self.check_atomic_order(success, interner, what)?;
        let failure = self.check_atomic_order(failure, interner, what)?;
        Ok(CheckedExpr {
            ty: Ty::Bool,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::CmpXchgN { sized: None },
                args: self.alloc_slice([ptr, expected, desired, success, failure]),
            },
        })
    }

    pub(super) fn check_atomic_order(&self, e: &Expr<'a>, interner: &Interner, what: &str) -> Result<CheckedExpr<'a>, CompileError> {
        let order = self.decay_checked(self.check_expr(e, interner)?);
        if !is_integer(&order.ty) {
            return Err(CompileError::new(format!("`{what}` needs an integer memory order"), order.span));
        }
        Ok(order)
    }

    pub(super) fn check_atomic_sized(&self, pointee: &Ty, sized: Option<u8>, interner: &Interner, what: &str, pointee_span: Span) -> Result<(), CompileError> {
        if let Some(n) = sized {
            match self.type_size(pointee, interner) {
                Some(size) if size == u64::from(n) => Ok(()),
                _ => Err(CompileError::new(format!("`{what}` needs a {n}-byte object"), pointee_span)),
            }
        } else {
            Ok(())
        }
    }

    /// Unsigned integer type holding exactly `n` bytes: the operative
    /// type of a sized atomic (`_1`/`_2`/`_4`/`_8`/`_16`) through
    /// `void *`, which carries no size of its own.
    pub(super) fn sized_atomic_ty(n: u8) -> Ty {
        match n {
            1 => Ty::UnsignedChar,
            2 => Ty::UnsignedShort,
            4 => Ty::UnsignedInt,
            8 => Ty::UnsignedLongLong,
            _ => Ty::UnsignedInt128,
        }
    }

    /// Value type an `_n`/sized atomic operates on: the pointee
    /// itself, except through `void *` (as libstdc++ passes the sized
    /// forms, e.g. `__atomic_load_8((const volatile void *)p, o)`),
    /// where the suffix gives the unsigned integer type. A bare `_n`
    /// through `void *` has no size at all and is rejected up front
    /// (rather than panicking the backend on a `void` result).
    pub(super) fn atomic_n_value_ty(&self, pointee: &Ty, sized: Option<u8>, interner: &Interner, what: &str, span: Span) -> Result<Ty, CompileError> {
        if pointee.unqualified() == &Ty::Void {
            match sized {
                Some(n) => return Ok(Self::sized_atomic_ty(n)),
                None => return Err(CompileError::new(format!("`{what}` needs a pointer to an object type"), span)),
            }
        }
        self.check_atomic_sized(pointee, sized, interner, what, span)?;
        Ok(pointee.clone())
    }

    /// `T __atomic_load_n(T *ptr, int order)` (and `__atomic_load_8`
    /// etc, which verify the size): the pointer gives the result type.
    pub(super) fn check_atomic_load_n(&self, args: &[Expr<'a>], interner: &Interner, span: Span, sized: Option<u8>, what: &str) -> Result<CheckedExpr<'a>, CompileError> {
        let [ptr, order] = args else {
            return Err(CompileError::new(format!("`{what}` takes a pointer and a memory order"), span));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let Ty::Pointer { pointee, .. } = ptr.ty.clone() else {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        };
        let op_ty = self.atomic_n_value_ty(&pointee, sized, interner, what, ptr.span)?;
        let order = self.check_atomic_order(order, interner, what)?;
        Ok(CheckedExpr {
            ty: op_ty,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::LoadN { sized },
                args: self.alloc_slice([ptr, order]),
            },
        })
    }

    /// `void __atomic_store_n(T *ptr, T val, int order)`: the value
    /// converts to the pointed-to type as if by assignment.
    pub(super) fn check_atomic_store_n(&self, args: &[Expr<'a>], interner: &Interner, span: Span, sized: Option<u8>, what: &str) -> Result<CheckedExpr<'a>, CompileError> {
        let [ptr, val, order] = args else {
            return Err(CompileError::new(format!("`{what}` takes a pointer, a value, and a memory order"), span));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let Ty::Pointer { pointee, .. } = ptr.ty.clone() else {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        };
        let op_ty = self.atomic_n_value_ty(&pointee, sized, interner, what, ptr.span)?;
        let val = self.convert_for_assignment(&op_ty, self.check_expr(val, interner)?, interner)?;
        let order = self.check_atomic_order(order, interner, what)?;
        Ok(CheckedExpr {
            ty: Ty::Void,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::StoreN { sized },
                args: self.alloc_slice([ptr, val, order]),
            },
        })
    }

    /// `void __atomic_load(T *ptr, T *ret, int order)`: generic load
    /// through pointers; both point at the same-sized object.
    pub(super) fn check_atomic_load(&self, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let [ptr, ret, order] = args else {
            return Err(CompileError::new("`__atomic_load` takes a pointer, a result pointer, and a memory order", span));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let ret = self.decay_checked(self.check_expr(ret, interner)?);
        let (Some(psize), Some(rsize)) = (self.ptr_pointee_size(&ptr.ty, interner), self.ptr_pointee_size(&ret.ty, interner)) else {
            return Err(CompileError::new("`__atomic_load` needs pointer arguments", span));
        };
        if psize != rsize {
            return Err(CompileError::new("`__atomic_load` pointers disagree in size", span));
        }
        let order = self.check_atomic_order(order, interner, "__atomic_load")?;
        Ok(CheckedExpr {
            ty: Ty::Void,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::Load,
                args: self.alloc_slice([ptr, ret, order]),
            },
        })
    }

    /// `void __atomic_store(T *ptr, T *val, int order)`: generic store.
    pub(super) fn check_atomic_store(&self, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let [ptr, val, order] = args else {
            return Err(CompileError::new("`__atomic_store` takes a pointer, a value pointer, and a memory order", span));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let val = self.decay_checked(self.check_expr(val, interner)?);
        let (Some(psize), Some(vsize)) = (self.ptr_pointee_size(&ptr.ty, interner), self.ptr_pointee_size(&val.ty, interner)) else {
            return Err(CompileError::new("`__atomic_store` needs pointer arguments", span));
        };
        if psize != vsize {
            return Err(CompileError::new("`__atomic_store` pointers disagree in size", span));
        }
        let order = self.check_atomic_order(order, interner, "__atomic_store")?;
        Ok(CheckedExpr {
            ty: Ty::Void,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::Store,
                args: self.alloc_slice([ptr, val, order]),
            },
        })
    }

    pub(super) fn ptr_pointee_size(&self, ty: &Ty, interner: &Interner) -> Option<u64> {
        match ty.unqualified() {
            Ty::Pointer { pointee, .. } => self.type_size(pointee, interner),
            _ => None,
        }
    }

    /// `T __atomic_exchange_n(T *ptr, T val, int order)`: returns the
    /// previous contents.
    pub(super) fn check_atomic_exchange_n(&self, args: &[Expr<'a>], interner: &Interner, span: Span, sized: Option<u8>, what: &str) -> Result<CheckedExpr<'a>, CompileError> {
        let [ptr, val, order] = args else {
            return Err(CompileError::new(format!("`{what}` takes a pointer, a value, and a memory order"), span));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let Ty::Pointer { pointee, .. } = ptr.ty.clone() else {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        };
        let op_ty = self.atomic_n_value_ty(&pointee, sized, interner, what, ptr.span)?;
        let val = self.convert_for_assignment(&op_ty, self.check_expr(val, interner)?, interner)?;
        let order = self.check_atomic_order(order, interner, what)?;
        Ok(CheckedExpr {
            ty: op_ty,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::ExchangeN { sized },
                args: self.alloc_slice([ptr, val, order]),
            },
        })
    }

    /// `void __atomic_exchange(T *ptr, T *val, T *ret, int order)`.
    pub(super) fn check_atomic_exchange(&self, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let [ptr, val, ret, order] = args else {
            return Err(CompileError::new(
                "`__atomic_exchange` takes a pointer, a value pointer, a result pointer, and a memory order",
                span,
            ));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let val = self.decay_checked(self.check_expr(val, interner)?);
        let ret = self.decay_checked(self.check_expr(ret, interner)?);
        let (Some(a), Some(b), Some(c)) = (
            self.ptr_pointee_size(&ptr.ty, interner),
            self.ptr_pointee_size(&val.ty, interner),
            self.ptr_pointee_size(&ret.ty, interner),
        ) else {
            return Err(CompileError::new("`__atomic_exchange` needs pointer arguments", span));
        };
        if a != b || a != c {
            return Err(CompileError::new("`__atomic_exchange` pointers disagree in size", span));
        }
        let order = self.check_atomic_order(order, interner, "__atomic_exchange")?;
        Ok(CheckedExpr {
            ty: Ty::Void,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::Exchange,
                args: self.alloc_slice([ptr, val, ret, order]),
            },
        })
    }

    /// `bool __atomic_compare_exchange_n(T *ptr, T *expected, T desired,
    /// bool weak, int success, int failure)`: on failure writes the
    /// current value back to `*expected`.
    pub(super) fn check_atomic_cmpxchg_n(&self, args: &[Expr<'a>], interner: &Interner, span: Span, sized: Option<u8>, what: &str) -> Result<CheckedExpr<'a>, CompileError> {
        let [ptr, expected, desired, weak, success, failure] = args else {
            return Err(CompileError::new(
                format!("`{what}` takes a pointer, an expected pointer, a desired value, a weak flag, and two memory orders"),
                span,
            ));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let Ty::Pointer { pointee, .. } = ptr.ty.clone() else {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        };
        let op_ty = self.atomic_n_value_ty(&pointee, sized, interner, what, ptr.span)?;
        let expected = self.decay_checked(self.check_expr(expected, interner)?);
        let Some(esize) = self.ptr_pointee_size(&expected.ty, interner) else {
            return Err(CompileError::new(format!("`{what}` needs an expected pointer"), expected.span));
        };
        let psize = self.type_size(&op_ty, interner).unwrap_or(0);
        if esize != psize {
            return Err(CompileError::new(format!("`{what}` expected pointer disagrees in size"), expected.span));
        }
        let desired = self.convert_for_assignment(&op_ty, self.check_expr(desired, interner)?, interner)?;
        let weak = self.decay_checked(self.check_expr(weak, interner)?);
        if !is_integer(&weak.ty) {
            return Err(CompileError::new(format!("`{what}` needs an integer weak flag"), weak.span));
        }
        let success = self.check_atomic_order(success, interner, what)?;
        let failure = self.check_atomic_order(failure, interner, what)?;
        Ok(CheckedExpr {
            ty: Ty::Bool,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::CmpXchgN { sized },
                args: self.alloc_slice([ptr, expected, desired, weak, success, failure]),
            },
        })
    }

    /// `bool __atomic_compare_exchange(T *ptr, T *expected, T *desired,
    /// bool weak, int success, int failure)`.
    pub(super) fn check_atomic_cmpxchg(&self, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let what = "__atomic_compare_exchange";
        let [ptr, expected, desired, weak, success, failure] = args else {
            return Err(CompileError::new(
                format!("`{what}` takes a pointer, an expected pointer, a desired pointer, a weak flag, and two memory orders"),
                span,
            ));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let expected = self.decay_checked(self.check_expr(expected, interner)?);
        let desired = self.decay_checked(self.check_expr(desired, interner)?);
        let (Some(a), Some(b), Some(c)) = (
            self.ptr_pointee_size(&ptr.ty, interner),
            self.ptr_pointee_size(&expected.ty, interner),
            self.ptr_pointee_size(&desired.ty, interner),
        ) else {
            return Err(CompileError::new(format!("`{what}` needs pointer arguments"), span));
        };
        if a != b || a != c {
            return Err(CompileError::new(format!("`{what}` pointers disagree in size"), span));
        }
        let weak = self.decay_checked(self.check_expr(weak, interner)?);
        if !is_integer(&weak.ty) {
            return Err(CompileError::new(format!("`{what}` needs an integer weak flag"), weak.span));
        }
        let success = self.check_atomic_order(success, interner, what)?;
        let failure = self.check_atomic_order(failure, interner, what)?;
        Ok(CheckedExpr {
            ty: Ty::Bool,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::CmpXchg,
                args: self.alloc_slice([ptr, expected, desired, weak, success, failure]),
            },
        })
    }

    /// `T __atomic_fetch_add(T *ptr, T val, int order)` (old value) and
    /// `T __atomic_add_fetch(T *ptr, T val, int order)` (new value),
    /// plus the sized spellings. Pointer targets act on `uintptr_t`
    /// (no scaling); `_Bool` targets are rejected like GCC.
    pub(super) fn check_atomic_rmw(
        &self,
        base: &str,
        fetch: bool,
        args: &[Expr<'a>],
        interner: &Interner,
        span: Span,
        sized: Option<u8>,
        what: &str,
    ) -> Result<CheckedExpr<'a>, CompileError> {
        let op = match base {
            "fetch_add" | "add_fetch" => RmwOp::Add,
            "fetch_sub" | "sub_fetch" => RmwOp::Sub,
            "fetch_and" | "and_fetch" => RmwOp::And,
            "fetch_or" | "or_fetch" => RmwOp::Or,
            "fetch_xor" | "xor_fetch" => RmwOp::Xor,
            "fetch_nand" | "nand_fetch" => RmwOp::Nand,
            "fetch_min" | "min_fetch" => RmwOp::Min,
            "fetch_max" | "max_fetch" => RmwOp::Max,
            _ => return Err(CompileError::new(format!("unsupported atomic builtin `{what}`"), span).with_code(DiagCode::UnsupportedAtomicBuiltin)),
        };
        let [ptr, val, order] = args else {
            return Err(CompileError::new(format!("`{what}` takes a pointer, a value, and a memory order"), span));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let Ty::Pointer { pointee, .. } = ptr.ty.clone() else {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        };
        let op_ty = self.atomic_n_value_ty(&pointee, sized, interner, what, ptr.span)?;
        if matches!(op_ty.unqualified(), Ty::Bool) {
            return Err(CompileError::new(format!("`{what}` must not operate on `_Bool`"), ptr.span));
        }
        if matches!(op, RmwOp::Min | RmwOp::Max) && !is_integer(&op_ty) {
            return Err(CompileError::new(format!("`{what}` needs an integer target"), ptr.span));
        }
        let val = match op_ty.unqualified() {
            Ty::Pointer { .. } => {
                let v = self.decay_checked(self.check_expr(val, interner)?);
                if !is_integer(&v.ty) {
                    return Err(CompileError::new(format!("`{what}` needs an integer value for a pointer target"), v.span));
                }
                v
            }
            _ => {
                if !is_integer(&op_ty) {
                    return Err(CompileError::new(format!("`{what}` needs an integer or pointer target"), ptr.span));
                }
                self.convert_for_assignment(&op_ty, self.check_expr(val, interner)?, interner)?
            }
        };
        let order = self.check_atomic_order(order, interner, what)?;
        Ok(CheckedExpr {
            ty: op_ty,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::Rmw { op, fetch, sized },
                args: self.alloc_slice([ptr, val, order]),
            },
        })
    }

    /// `void __atomic_thread_fence(int order)` /
    /// `void __atomic_signal_fence(int order)`.
    pub(super) fn check_atomic_fence(&self, args: &[Expr<'a>], interner: &Interner, span: Span, signal: bool, what: &str) -> Result<CheckedExpr<'a>, CompileError> {
        let [order] = args else {
            return Err(CompileError::new(format!("`{what}` takes a memory order"), span));
        };
        let order = self.check_atomic_order(order, interner, what)?;
        Ok(CheckedExpr {
            ty: Ty::Void,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::Fence { signal },
                args: self.alloc_slice([order]),
            },
        })
    }

    /// `bool __atomic_always_lock_free(size_t size[, void *ptr])` (size
    /// must be constant) and `bool __atomic_is_lock_free(...)` (may be
    /// dynamic; constant sizes fold here, dynamic ones lower to a size
    /// comparison).
    pub(super) fn check_atomic_lock_free(&self, args: &[Expr<'a>], interner: &Interner, span: Span, always: bool, what: &str) -> Result<CheckedExpr<'a>, CompileError> {
        if args.is_empty() || args.len() > 2 {
            return Err(CompileError::new(format!("`{what}` takes a size and an optional pointer"), span));
        }
        let size = self.decay_checked(self.check_expr(&args[0], interner)?);
        if !is_integer(&size.ty) {
            return Err(CompileError::new(format!("`{what}` needs an integer size"), size.span));
        }
        let mut checked = vec![size];
        if let Some(ptr) = args.get(1) {
            let p = self.decay_checked(self.check_expr(ptr, interner)?);
            match &p.ty {
                t if is_integer(t) => {}
                Ty::Pointer { .. } => {}
                _ => return Err(CompileError::new(format!("`{what}` needs a pointer or zero second argument"), p.span)),
            }
            checked.push(p);
        }
        if always
            && let [size] = &checked[..1]
            && crate::eval::eval_const(size, self, interner).is_none()
        {
            return Err(CompileError::new(format!("`{what}` needs a constant size"), span));
        }
        if let [size] = &checked[..]
            && let Some(crate::eval::Value::Int(n)) = crate::eval::eval_const(size, self, interner)
        {
            let lock_free = matches!(n, 1 | 2 | 4 | 8);
            return Ok(CheckedExpr {
                ty: Ty::Bool,
                span,
                kind: CheckedExprKind::ConstantP(lock_free),
            });
        }
        Ok(CheckedExpr {
            ty: Ty::Bool,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::LockFree { always },
                args: self.alloc_slice(checked),
            },
        })
    }

    /// `bool __atomic_test_and_set(void *ptr, int order)`.
    pub(super) fn check_atomic_test_and_set(&self, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let what = "__atomic_test_and_set";
        let [ptr, order] = args else {
            return Err(CompileError::new(format!("`{what}` takes a pointer and a memory order"), span));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        if !matches!(ptr.ty.unqualified(), Ty::Pointer { .. }) {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        }
        let order = self.check_atomic_order(order, interner, what)?;
        Ok(CheckedExpr {
            ty: Ty::Bool,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::TestAndSet,
                args: self.alloc_slice([ptr, order]),
            },
        })
    }

    /// `void __atomic_clear(bool *ptr, int order)`.
    pub(super) fn check_atomic_clear(&self, args: &[Expr<'a>], interner: &Interner, span: Span) -> Result<CheckedExpr<'a>, CompileError> {
        let what = "__atomic_clear";
        let [ptr, order] = args else {
            return Err(CompileError::new(format!("`{what}` takes a pointer and a memory order"), span));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        if !matches!(ptr.ty.unqualified(), Ty::Pointer { .. }) {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        }
        let order = self.check_atomic_order(order, interner, what)?;
        Ok(CheckedExpr {
            ty: Ty::Void,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::Clear,
                args: self.alloc_slice([ptr, order]),
            },
        })
    }
}
