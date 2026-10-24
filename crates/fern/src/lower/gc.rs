//! The GC runtime ABI: frames, root slots, and allocation.
//!
//! The collector itself lives in C (`rt/gc.c`, compiled by
//! [`crate::runtime`]) and owns the heap, the root array, and the depth
//! bookkeeping. Generated code reaches it through four functions declared
//! here:
//!
//! ```text
//!   u64   fern_enter(void)                        frame cookie
//!   void  fern_pin(u64 frame, u64 slot, void *p)  publish or retire a root
//!   void  fern_leave(u64 frame)                   drop the frame's roots
//!   void *fern_alloc(u64 size)                    allocate a payload
//! ```
//!
//! A frame is a cookie handed out by `fern_enter`: the root depth at
//! entry. Every body numbers its slots from zero -- first one per
//! pointer-typed local, then one per temporary in flight -- and pins a
//! pointer into its slot wherever the value may live across a later
//! allocation; a null pin retires a temporary. Frames nest safely: a
//! callee's cookie is the depth at its call, so its slots sit above every
//! live caller's and below every pending callee's, and `fern_leave` puts
//! the depth back before the body returns. Thunks allocate and pin
//! nothing, so they carry no frame.

use glcc_mir as mir;

use crate::span::KResult;

use super::func::FnCtx;

pub(crate) struct Runtime {
    pub(crate) enter: mir::Extern,
    pub(crate) pin: mir::Extern,
    pub(crate) leave: mir::Extern,
    pub(crate) alloc: mir::Extern,
    /// Content equality for NUL-terminated strings.
    pub(crate) str_eq: mir::Extern,
}

/// Declare the runtime's entry points on the module.
pub(crate) fn declare(module: &mut mir::Module) -> Runtime {
    let word = mir::Ty::int(64);
    let ptr = mir::Ty::ptr(mir::Ty::Void);
    let enter = module.push_extern("fern_enter", mir::Signature::new(vec![], word.clone()));
    let pin = module.push_extern("fern_pin", mir::Signature::new(vec![word.clone(), word.clone(), ptr.clone()], mir::Ty::Void));
    let leave = module.push_extern("fern_leave", mir::Signature::new(vec![word], mir::Ty::Void));
    let alloc = module.push_extern("fern_alloc", mir::Signature::new(vec![mir::Ty::int(64)], ptr.clone()));
    let str_eq = module.push_extern("fern_str_eq", mir::Signature::new(vec![ptr.clone(), ptr], mir::Ty::Bool));

    module.externs[enter].effect = mir::ExternEffect::neutral();
    module.externs[pin].effect = mir::ExternEffect::neutral();
    module.externs[leave].effect = mir::ExternEffect::neutral();
    module.externs[alloc].effect = mir::ExternEffect::allocator();
    module.externs[str_eq].effect = mir::ExternEffect::read_only();

    Runtime { enter, pin, leave, alloc, str_eq }
}

/// Whether a value of this MIR type can point into the heap.
pub(crate) fn is_ptr(ty: &mir::Ty) -> bool {
    matches!(ty, mir::Ty::Ptr { .. })
}

impl<'l, 'd> FnCtx<'l, 'd> {
    /// Prologue: take this body's frame cookie. Runs before locals are
    /// declared, so every pin has a frame to name.
    pub(super) fn open_frame(&mut self) -> KResult<()> {
        let enter = self.lw.rt.enter;
        let frame = self.emit(
            mir::InstKind::CallExtern {
                target: enter,
                args: Default::default(),
            },
            mir::Ty::int(64),
        )?;
        self.frame = Some(frame);
        Ok(())
    }

    /// Epilogue: drop the frame's roots, just before the body returns.
    pub(super) fn close_frame(&mut self) -> KResult<()> {
        let frame = self.frame.expect("close_frame without open_frame");
        let leave = self.lw.rt.leave;
        self.emit_void(mir::InstKind::CallExtern {
            target: leave,
            args: vec![frame].into(),
        })
    }

    /// Reserve a root slot for a local whose value is a pointer; locals
    /// stay rooted for the whole body.
    pub(super) fn reserve_local_root(&mut self, ty: &mir::Ty) -> Option<u32> {
        if !is_ptr(ty) {
            return None;
        }
        let slot = self.local_roots;
        self.local_roots += 1;
        Some(slot)
    }

    /// The current temporary mark, for releasing several holds at once.
    pub(super) fn mark(&self) -> u32 {
        self.temps
    }

    /// Hold `val` across later allocations, when it is a pointer. Returns
    /// the mark to hand back to [`Self::release`]; non-pointers are
    /// skipped, so callers can hold unconditionally.
    pub(super) fn hold(&mut self, val: mir::Value) -> KResult<u32> {
        let mark = self.temps;
        if !is_ptr(&self.func.values[val].ty) {
            return Ok(mark);
        }
        let slot = self.local_roots + self.temps;
        self.temps += 1;
        self.pin(slot, val)?;
        Ok(mark)
    }

    /// Retire every temporary held since `mark`.
    pub(super) fn release(&mut self, mark: u32) -> KResult<()> {
        while self.temps > mark {
            self.temps -= 1;
            self.unpin(self.local_roots + self.temps)?;
        }
        Ok(())
    }

    /// Allocate `size` bytes, 8-aligned, and return the payload pointer.
    pub(super) fn call_alloc(&mut self, size: mir::Value) -> KResult<mir::Value> {
        let alloc = self.lw.rt.alloc;
        self.emit(
            mir::InstKind::CallExtern {
                target: alloc,
                args: vec![size].into(),
            },
            mir::Ty::ptr(mir::Ty::Void),
        )
    }

    pub(super) fn pin(&mut self, slot: u32, val: mir::Value) -> KResult<()> {
        let frame = self.frame.expect("pin outside a GC frame");
        let slot = self.iconst(64, i128::from(slot));
        let pin = self.lw.rt.pin;
        self.call_runtime(pin, vec![frame, slot, val])
    }

    pub(super) fn unpin(&mut self, slot: u32) -> KResult<()> {
        let frame = self.frame.expect("unpin outside a GC frame");
        let slot = self.iconst(64, i128::from(slot));
        let null = self.func.push_const(mir::Const::Null);
        let pin = self.lw.rt.pin;
        self.call_runtime(pin, vec![frame, slot, null])
    }

    /// A void runtime call; code after a diverging expression is
    /// unreachable, so pins there are simply dropped.
    fn call_runtime(&mut self, target: mir::Extern, args: Vec<mir::Value>) -> KResult<()> {
        if self.terminated {
            return Ok(());
        }
        self.emit_void(mir::InstKind::CallExtern { target, args: args.into() })
    }
}
