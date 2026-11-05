//! MSVC builtin checking (`<intrin.h>` spellings).

use super::*;

impl<'a> Sema<'a> {
    /// MSVC `T _InterlockedCompareExchange*(T volatile *dest, T exchange,
    /// T comparand)`: a strong compare-exchange that takes the expected
    /// value directly (not a pointer) and returns the object's initial
    /// value. The pointer's pointee gives the operand type, so the
    /// `...8`/`16`/`64`/`Pointer` variants need no special casing.
    pub(super) fn check_interlocked_cmpxchg(&self, args: &[Expr<'a>], interner: &Interner, span: Span, what: &str) -> Result<CheckedExpr<'a>, CompileError> {
        let [ptr, exchange, comparand] = args else {
            return Err(CompileError::new(format!("`{what}` takes a pointer, an exchange value, and a comparand"), span));
        };
        let ptr = self.decay_checked(self.check_expr(ptr, interner)?);
        let Ty::Pointer { pointee, .. } = ptr.ty.clone() else {
            return Err(CompileError::new(format!("`{what}` needs a pointer argument"), ptr.span));
        };
        let op_ty = self.atomic_n_value_ty(&pointee, None, interner, what, ptr.span)?;
        let exchange = self.convert_for_assignment(&op_ty, self.check_expr(exchange, interner)?, interner)?;
        let comparand = self.convert_for_assignment(&op_ty, self.check_expr(comparand, interner)?, interner)?;
        Ok(CheckedExpr {
            ty: op_ty,
            span,
            kind: CheckedExprKind::Atomic {
                op: AtomicOp::InterlockedCmpxchg,
                args: self.alloc_slice([ptr, exchange, comparand]),
            },
        })
    }
}
