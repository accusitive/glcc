use super::*;

impl<'s, 'i> Lowerer<'s, 'i> {
    /// A comparison yielding a `Bool`.
    pub(crate) fn icmp(&self, func: &mut mir::Function, frame: &mut Frame, pred: mir::IcmpPred, lhs: mir::Value, rhs: mir::Value) -> mir::Value {
        self.emit(func, frame, mir::InstKind::Icmp { pred, lhs, rhs }, mir::Ty::Bool)
    }

    pub(crate) fn iconst(&self, func: &mut mir::Function, bits: u32, value: i128) -> mir::Value {
        func.push_const(mir::Const::Int { bits, value })
    }

    /// A stack slot for `ty`, always created in the entry block (like
    /// clang -O0): the slot address dominates every use, including
    /// across loop back-edges and case/goto edges that skip the
    /// declaring block. Mid-function allocas left addresses Cranelift
    /// rejects as non-dominating.
    pub(crate) fn alloca(&self, func: &mut mir::Function, frame: &mut Frame, ty: mir::Ty) -> mir::Value {
        let entry = func.entry;
        let ptr = mir::Ty::ptr(ty.clone());
        let result = func.values.push(mir::ValueData {
            ty: ptr,
            def: mir::Def::Inst(mir::Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(mir::InstData {
            block: entry,
            kind: mir::InstKind::Alloca { pointee: ty, count: 1 },
            result: Some(result),
            loc: frame.loc,
        });
        func.values[result].def = mir::Def::Inst(inst);
        func.blocks[entry].insts.push(inst);
        result
    }

    /// Allocate `ty` with at least `align` bytes of alignment. A plain
    /// stack slot only guarantees the type's natural alignment, so an
    /// over-aligned local gets a byte buffer and a rounded pointer
    /// computed in the entry block.
    pub(crate) fn alloca_aligned(&self, func: &mut mir::Function, frame: &mut Frame, ty: mir::Ty, align: u64) -> mir::Value {
        debug_assert!(align.is_power_of_two());
        let size = ty.size_bytes().max(1);
        let bytes = size.checked_add(align - 1).expect("aligned alloca size overflow");
        let raw = self.alloca(
            func,
            frame,
            mir::Ty::Array {
                elem: Box::new(mir::Ty::int(8)),
                len: bytes,
            },
        );
        let bits = (self.sema.target.pointer_size * 8) as u32;
        let int_ty = mir::Ty::int(bits);
        let addr = self.emit(func, frame, mir::InstKind::PtrToInt(raw), int_ty.clone());
        let bias = self.iconst(func, bits, i128::from(align - 1));
        let biased = self.emit(func, frame, mir::InstKind::Add(addr, bias), int_ty.clone());
        let mask = self.iconst(func, bits, i128::from(!(align - 1)));
        let aligned = self.emit(func, frame, mir::InstKind::And(biased, mask), int_ty);
        self.emit(func, frame, mir::InstKind::IntToPtr(aligned), mir::Ty::ptr(ty))
    }

    pub(crate) fn load(&self, func: &mut mir::Function, frame: &mut Frame, addr: mir::Value, ty: mir::Ty, vol: bool) -> mir::Value {
        let flags = if vol { mir::MemFlags::vol() } else { mir::MemFlags::plain() };
        self.emit(func, frame, mir::InstKind::Load { addr, flags }, ty)
    }

    pub(crate) fn store(&self, func: &mut mir::Function, frame: &mut Frame, addr: mir::Value, val: mir::Value, vol: bool) {
        let flags = if vol { mir::MemFlags::vol() } else { mir::MemFlags::plain() };
        self.emit_unit(func, frame, mir::InstKind::Store { addr, val, flags });
    }

    pub(crate) fn emit(&self, func: &mut mir::Function, frame: &mut Frame, kind: mir::InstKind, ty: mir::Ty) -> mir::Value {
        let block = self.live_block(func, frame);
        let result = func.values.push(mir::ValueData {
            ty,
            def: mir::Def::Inst(mir::Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(mir::InstData {
            block,
            kind,
            result: Some(result),
            loc: frame.loc,
        });
        func.values[result].def = mir::Def::Inst(inst);
        func.blocks[block].insts.push(inst);
        result
    }

    pub(crate) fn emit_unit(&self, func: &mut mir::Function, frame: &mut Frame, kind: mir::InstKind) {
        let block = self.live_block(func, frame);
        let inst = func.insts.push(mir::InstData {
            block,
            kind,
            result: None,
            loc: frame.loc,
        });
        func.blocks[block].insts.push(inst);
    }

    pub(crate) fn terminate(&self, func: &mut mir::Function, frame: &mut Frame, term: mir::Terminator) {
        let block = self.live_block(func, frame);
        frame.current.take();
        func.blocks[block].term = Some(term);
    }

    /// The block instructions append to, opening an orphan past a
    /// terminator so unreachable source still lowers to valid MIR.
    pub(crate) fn live_block(&self, func: &mut mir::Function, frame: &mut Frame) -> mir::Block {
        if let Some(block) = frame.current {
            return block;
        }
        let block = func.push_block(vec![]);
        frame.current = Some(block);
        block
    }
}
