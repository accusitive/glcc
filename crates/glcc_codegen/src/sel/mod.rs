//! Instruction selection: the generic driver plus the declarative
//! pattern table. Selection walks MIR blocks in order; each instruction
//! goes through the target's `select_custom` hook first, then the
//! ordered pattern table. Patterns map one MIR instruction to one or
//! more machine instructions; fused shapes (compare + branch, calls)
//! belong in the hook, where they see their context.

use std::collections::HashMap;

use glcc_mir::{Const, Def, Function, Inst, InstKind, Value};
use smallvec::{SmallVec, smallvec};

use crate::isa::TargetIsa;
use crate::machine::{BlockId, MBlock, MFunction, MInst, OpId, Operand, PReg, Reg, RegClass, VReg};

/// One selection pattern: match a MIR instruction of `kind` whose
/// result type and operand shapes satisfy the predicates, then emit the
/// template list. Patterns are tried in table order; first match wins,
/// so more specific patterns go first.
pub struct Pattern {
    pub kind: InstKind,
    /// Result must be a scalar integer/pointer/bool of at most this
    /// many bits (excludes aggregates; `None` accepts any type).
    pub res_max: Option<u32>,
    pub lhs: Shape,
    pub rhs: Shape,
    pub emit: Vec<Emit>,
}

/// Operand shape predicates.
#[derive(Clone, Copy)]
pub enum Shape {
    Any,
    /// A constant with value in the inclusive range.
    ConstIn(i64, i64),
    ConstIs(i64),
    /// Must not be a constant (so a rematerialization doesn't obscure a
    /// cheaper register form).
    NonConst,
}

/// One machine instruction emitted by a pattern.
pub struct Emit {
    pub op: OpId,
    pub width: EmitWidth,
    pub slots: Vec<SlotSel>,
}

pub enum EmitWidth {
    /// Width of the MIR result type; patterns emitting into a
    /// result-less instruction must use another variant.
    Result,
    /// Width of operand `n`'s type (e.g. a store's stored value).
    OfOpnd(u8),
    Fixed(crate::machine::Width),
}

/// A slot of the emitted instruction, mapped from the matched MIR
/// instruction. Operand indices follow `for_each_operand` order.
pub enum SlotSel {
    /// The result value's register.
    Result,
    Opnd(u8),
    /// Operand `n`'s constant value as an immediate (patterns using it
    /// must predicate the shape with `Shape::ConstIn`).
    OpndImm(u8),
    Imm(i64),
    /// Memory reference through operand `n`'s address (a vreg base, or
    /// a frame slot directly for allocas).
    MemOf(u8),
}

pub struct PatternTable {
    pub patterns: &'static [Pattern],
}

/// Where a MIR value lives during selection: a virtual register, or a
/// frame slot for an alloca (used directly as memory bases; vregs only
/// materialize when the address itself is needed).
#[derive(Clone, Copy, Debug)]
pub enum ValLoc {
    Reg(VReg),
    Slot(u32),
}

/// Per-function selection state. The target's hooks and the pattern
/// emitter both go through `FuncSel` for value registers and emission.
pub struct FuncSel<'a> {
    pub isa: &'static dyn TargetIsa,
    /// The enclosing module (signatures of callees and externs).
    pub module: &'a glcc_mir::Module,
    /// The function being selected, for asm symbol retargeting.
    pub func_id: Option<glcc_mir::Func>,
    pub f: MFunction,
    locs: HashMap<Value, ValLoc>,
    next_vreg: u32,
    vreg_class: Vec<crate::machine::RegClass>,
    cur: BlockId,
    /// The MIR entry block: materializations there dominate the whole
    /// function, so their vregs may be cached globally (see value_loc).
    mir_entry: BlockId,
    /// Constants already materialized in the current block, by value.
    block_consts: HashMap<Value, VReg>,
    /// Variadic definitions: frame slot of the 176-byte register save
    /// area (6*8 GP + 8*16 FP), set by the Entry hook.
    pub va_save_slot: Option<u32>,
    /// Byte offset of the overflow arguments relative to rbp.
    pub va_overflow_disp: i64,
    /// Initial gp_offset / fp_offset fields of the va_list tag.
    pub va_gp_off: i64,
    pub va_fp_off: i64,
    /// For memory-class (sret) returns: the vreg holding the hidden
    /// out-pointer, pinned at entry.
    pub sret_loc: Option<VReg>,
    /// Values whose computation was folded into a consumer (compare +
    /// branch fusion).
    fused: std::collections::HashSet<Value>,
}

impl<'a> FuncSel<'a> {
    fn new(isa: &'static dyn TargetIsa, module: &'a glcc_mir::Module, func_id: Option<glcc_mir::Func>, mir: &Function) -> Self {
        let mut f = MFunction {
            name: mir.name.clone(),
            local: mir.local,
            weak: mir.weak,
            blocks: Vec::with_capacity(mir.blocks.len()),
            frame: Default::default(),
            num_vregs: 0,
            vreg_class: Vec::new(),
        };
        for _ in mir.blocks.iter() {
            f.blocks.push(MBlock {
                params: Vec::new(),
                insts: Vec::new(),
                succs: SmallVec::new(),
            });
        }
        FuncSel {
            isa,
            module,
            func_id,
            f,
            locs: HashMap::new(),
            next_vreg: 0,
            vreg_class: Vec::new(),
            cur: 0,
            mir_entry: mir.entry.index() as BlockId,
            block_consts: HashMap::new(),
            va_save_slot: None,
            va_overflow_disp: 0,
            va_gp_off: 0,
            va_fp_off: 48,
            sret_loc: None,
            fused: std::collections::HashSet::new(),
        }
    }

    pub fn alloc_vreg(&mut self) -> VReg {
        self.alloc_vreg_class(crate::machine::RegClass::Gpr)
    }

    pub fn alloc_vreg_class(&mut self, class: crate::machine::RegClass) -> VReg {
        let v = VReg(self.next_vreg);
        self.next_vreg += 1;
        self.vreg_class.push(class);
        v
    }

    fn value_class(ty: &glcc_mir::Ty) -> crate::machine::RegClass {
        match ty {
            glcc_mir::Ty::Float(_) => crate::machine::RegClass::Vec,
            _ => crate::machine::RegClass::Gpr,
        }
    }



    /// The register holding `v`, creating (and for constants,
    /// materializing in the current block) as needed.
    pub fn value_vreg(&mut self, mir: &Function, v: Value) -> VReg {
        match self.value_loc(mir, v) {
            ValLoc::Reg(vr) => vr,
            ValLoc::Slot(slot) => {
                // Materialize the slot's address once; later uses of
                // the value reuse it.
                let vr = self.alloc_vreg();
                self.emit(MInst::new(
                    self.isa.lea_op(),
                    crate::machine::Width::W64,
                    smallvec![
                        Operand::Reg(Reg::V(vr)),
                        Operand::Mem(crate::machine::MemOperand { base: Some(Reg::FI(slot)), index: None, scale: 0, disp: 0, sym: None })
                    ],
                ));
                self.locs.insert(v, ValLoc::Reg(vr));
                vr
            }
        }
    }

    pub fn value_loc(&mut self, mir: &Function, v: Value) -> ValLoc {
        if let Some(loc) = self.locs.get(&v) {
            return *loc;
        }
        let data = &mir.values[v];
        match data.def {
            Def::Const(_) => {
                if let Some(vr) = self.block_consts.get(&v) {
                    return ValLoc::Reg(*vr);
                }
                let vr = self.alloc_vreg_class(Self::value_class(&mir.values[v].ty));
                if vr.0 == 9 && std::env::var_os("GLCC_RA_DEBUG").is_some() {
                    eprintln!("SEL DEBUG: MIR const {:?} (def {:?}) -> vreg 9 (materialize)", v, mir.values[v].def);
                }
                self.materialize_const(mir, v, vr);
                self.block_consts.insert(v, vr);
                // Only an entry-block materialization dominates the
                // whole function; elsewhere the vreg is valid within
                // this block alone (block_consts clears per block).
                if self.cur == self.mir_entry {
                    self.locs.insert(v, ValLoc::Reg(vr));
                }
                ValLoc::Reg(vr)
            }
            Def::Inst(_) | Def::Param(..) => {
                let vr = self.alloc_vreg_class(Self::value_class(&mir.values[v].ty));
                if vr.0 == 9 && std::env::var_os("GLCC_RA_DEBUG").is_some() {
                    eprintln!("SEL DEBUG: MIR value {:?} (def {:?}) -> vreg 9", v, mir.values[v].def);
                }
                self.locs.insert(v, ValLoc::Reg(vr));
                ValLoc::Reg(vr)
            }
        }
    }

    /// Memory operand reading `v`'s address as its base: allocas turn
    /// into frame-slot bases directly, everything else into a vreg.
    pub fn mem_of(&mut self, mir: &Function, v: Value) -> crate::machine::MemOperand {
        let base = match self.value_loc(mir, v) {
            ValLoc::Slot(s) => Reg::FI(s),
            ValLoc::Reg(vr) => Reg::V(vr),
        };
        crate::machine::MemOperand { base: Some(base), index: None, scale: 0, disp: 0, sym: None }
    }

    fn materialize_const(&mut self, mir: &Function, v: Value, dst: VReg) {
        let ty = &mir.values[v].ty;
        // Float constants ride a 64-bit integer immediate into a GPR,
        // then a GPR→XMM bitcast move; there is no SSE immediate form.
        if let Def::Const(Const::Float { bits, .. }) = &mir.values[v].def {
            let bits = *bits as i64;
            let gpr = self.alloc_vreg();
            self.emit(MInst::new(
                self.isa.const_move_op(bits),
                crate::machine::Width::W64,
                smallvec![Operand::Reg(Reg::V(gpr)), Operand::Imm(bits)],
            ));
            self.emit(MInst::new(
                self.isa.gpr_to_vec_op(),
                crate::machine::Width::W64,
                smallvec![Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(gpr))],
            ));
            return;
        }
        let value = match &mir.values[v].def {
            Def::Const(Const::Int { value, .. }) => i64::try_from(*value).unwrap_or_else(|_| {
                panic!("glcc native backend: 65+-bit constant not yet supported")
            }),
            // Undef and Null are both zero: an undef read is allowed
            // to yield any value of the type.
            Def::Const(Const::Null) | Def::Const(Const::Undef) => 0,
            d => panic!("glcc native backend: unsupported constant {d:?}"),
        };
        let bits = match ty {
            glcc_mir::Ty::Int { bits } | glcc_mir::Ty::Ptr { bits, .. } => *bits,
            glcc_mir::Ty::Bool => 8,
            t => panic!("glcc native backend: unsupported constant type {t:?}"),
        };
        let width = match bits {
            b if b <= 8 => crate::machine::Width::W8,
            b if b <= 16 => crate::machine::Width::W16,
            b if b <= 32 => crate::machine::Width::W32,
            _ => crate::machine::Width::W64,
        };
        let op = self.isa.const_move_op(value);
        self.emit(MInst::new(op, width, smallvec![Operand::Reg(Reg::V(dst)), Operand::Imm(value)]));
    }

    /// Append to the current block.
    pub fn emit(&mut self, inst: MInst) {
        self.f.blocks[self.cur as usize].insts.push(inst);
    }

    pub fn cur_block(&self) -> BlockId {
        self.cur
    }

    pub fn slot(&mut self, size: u32, align: u32) -> u32 {
        self.f.frame.slots.push(crate::machine::Slot { size, align, offset: 0 });
        (self.f.frame.slots.len() - 1) as u32
    }

    /// Emit a block-ending branch instruction. `targets` and per-edge
    /// argument vregs flow to the allocator and the encoder.
    pub fn emit_branch(&mut self, inst: MInst, targets: &[BlockId], args: &[&[VReg]]) {
        assert_eq!(targets.len(), args.len(), "branch targets/args mismatch");
        let b = self.cur as usize;
        self.f.blocks[b].succs = targets.iter().copied().collect();
        self.f.blocks[b].insts.push(MInst {
            branch: Some(crate::machine::BranchInfo {
                targets: targets.iter().copied().collect(),
                args: args.iter().map(|a| a.iter().copied().collect()).collect(),
            }),
            ..inst
        });
    }

    /// Debug: note direct alloc_vreg callers.
    pub fn alloc_vreg_note(&mut self, tag: &'static str) -> VReg {
        let v = self.alloc_vreg();
        if v.0 == 9 && std::env::var_os("GLCC_RA_DEBUG").is_some() {
            eprintln!("SEL DEBUG: {tag} -> vreg 9");
        }
        v
    }

    /// Append an empty block (used for synthesized control flow, e.g.
    /// switch chains); the caller emits into it with `set_cur`.
    pub fn new_block(&mut self) -> BlockId {
        self.f.blocks.push(MBlock { params: Vec::new(), insts: Vec::new(), succs: SmallVec::new() });
        (self.f.blocks.len() - 1) as BlockId
    }

    pub fn set_cur(&mut self, block: BlockId) {
        self.cur = block;
    }

    /// Map a MIR value to a frame slot (allocas).
    pub fn set_slot(&mut self, v: Value, slot: u32) {
        self.locs.insert(v, ValLoc::Slot(slot));
    }

    /// Pre-assign a MIR value to a vreg (definitions selected at their
    /// definition site, e.g. alloca addresses in the entry block, where
    /// the definition dominates every use).
    pub fn set_reg(&mut self, v: Value, vr: VReg) {
        self.locs.insert(v, ValLoc::Reg(vr));
    }

    /// MIR instructions already lowered by fusion (their result is
    /// consumed by the fusing instruction, so the pattern table must
    /// skip them).
    pub fn mark_fused(&mut self, v: Value) {
        self.fused.insert(v);
    }

    pub fn is_fused(&self, v: Value) -> bool {
        self.fused.contains(&v)
    }
}

/// Select a whole MIR function into machine IR.
pub fn select_function<'a>(isa: &'static dyn TargetIsa, module: &'a glcc_mir::Module, func_id: Option<glcc_mir::Func>, mir: &Function) -> MFunction {
    let mut sel = FuncSel::new(isa, module, func_id, mir);

    // Block parameters: every MIR block param gets a vreg up front, so
    // uses never depend on block walk order. Entry params are excluded:
    // regalloc2 forbids entry liveins, so the ABI moves (emitted by the
    // target's Entry hook) are their only definitions.
    for (b, data) in mir.blocks.iter_enumerated() {
        if b == mir.entry {
            continue;
        }
        for &p in &data.params {
            let vr = sel.alloc_vreg_class(FuncSel::value_class(&mir.values[p].ty));
            if vr.0 == 9 && std::env::var_os("GLCC_RA_DEBUG").is_some() {
                eprintln!("SEL DEBUG: block param {p:?} of block {b:?} -> vreg 9");
            }
            sel.locs.insert(p, ValLoc::Reg(vr));
            sel.f.blocks[b.index()].params.push(vr);
        }
    }

    isa.select_custom(&mut sel, mir, crate::isa::SelItem::Entry);

    for (b, data) in mir.blocks.iter_enumerated() {
        sel.cur = b.index() as BlockId;
        sel.block_consts.clear();
        for &inst in &data.insts {
            // Fused-away instructions (compare folded into a branch)
            // were already lowered by their consumer.
            if mir.insts[inst].result.is_some_and(|r| sel.is_fused(r)) {
                continue;
            }
            if isa.select_custom(&mut sel, mir, crate::isa::SelItem::Inst(inst)) {
                continue;
            }
            if try_pattern(&mut sel, mir, isa.patterns(), inst) {
                continue;
            }
            panic!("glcc native backend: cannot select {:?} in function {}", mir.insts[inst].kind, mir.name);
        }
        if !isa.select_custom(&mut sel, mir, crate::isa::SelItem::Term(b)) {
            panic!("glcc native backend: unhandled terminator in block {b:?}");
        }
    }
    sel.f.num_vregs = sel.next_vreg;
    sel.f.vreg_class = std::mem::take(&mut sel.vreg_class);
    split_critical_edges(&mut sel.f);
    prune_dead_blocks(&mut sel.f);
    sel.f
}

/// Drop blocks unreachable from the entry. The register allocator only
/// walks reachable blocks, so dead blocks would break its liveness
/// invariants; MIR routinely contains them (arms of `if` that both
/// return leave the join unreachable).
fn prune_dead_blocks(f: &mut MFunction) {
    let mut order: Vec<BlockId> = Vec::with_capacity(f.blocks.len());
    let mut remap = vec![u32::MAX; f.blocks.len()];
    let mut stack = vec![0u32];
    remap[0] = 0;
    while let Some(b) = stack.pop() {
        order.push(b);
        for &succ in &f.blocks[b as usize].succs {
            if remap[succ as usize] == u32::MAX {
                remap[succ as usize] = 0;
                stack.push(succ);
            }
        }
    }
    for (new, &old) in order.iter().enumerate() {
        remap[old as usize] = new as BlockId;
    }
    if order.len() == f.blocks.len() {
        return;
    }
    let mut blocks = std::mem::take(&mut f.blocks);
    for &old in &order {
        let b = &mut blocks[old as usize];
        for s in b.succs.iter_mut() {
            *s = remap[*s as usize];
        }
        for inst in b.insts.iter_mut() {
            if let Some(br) = &mut inst.branch {
                for t in br.targets.iter_mut() {
                    *t = remap[*t as usize];
                }
            }
        }
    }
    f.blocks = order.iter().map(|old| std::mem::take(&mut blocks[*old as usize])).collect();
}

fn try_pattern(sel: &mut FuncSel, mir: &Function, table: &PatternTable, inst: Inst) -> bool {
    let data = &mir.insts[inst];
    let mut operands: SmallVec<[Value; 4]> = SmallVec::new();
    data.kind.for_each_operand(&mut |v| operands.push(v));

    for p in table.patterns {
        // Match on the variant only: payload fields (dummy values in
        // the table rows) are predicates' business, not the key's.
        if std::mem::discriminant(&p.kind) != std::mem::discriminant(&data.kind) {
            continue;
        }
        if let Some(max) = p.res_max {
            let Some(res) = data.result else { continue };
            let bits = scalar_bits(&mir.values[res].ty);
            if bits == 0 || bits > max {
                continue;
            }
        }
        if !shape_matches(mir, operands.first().copied(), &p.lhs)
            || !shape_matches(mir, operands.get(1).copied(), &p.rhs)
        {
            continue;
        }
        for e in &p.emit {
            let width = match e.width {
                EmitWidth::Result => {
                    let res = data.result.expect("pattern emitting Result width without a result");
                    value_width(&mir.values[res].ty)
                }
                EmitWidth::OfOpnd(n) => value_width(&mir.values[operands[n as usize]].ty),
                EmitWidth::Fixed(w) => w,
            };
            let slots: SmallVec<[Operand; 4]> = e.slots
                .iter()
                .map(|s| match s {
                    SlotSel::Result => {
                        let res = data.result.expect("pattern emitting Result slot without a result");
                        Operand::Reg(Reg::V(sel.value_vreg(mir, res)))
                    }
                    SlotSel::Opnd(i) => Operand::Reg(Reg::V(sel.value_vreg(mir, operands[*i as usize]))),
                    SlotSel::OpndImm(i) => {
                        let v = operands[*i as usize];
                        let value = match &mir.values[v].def {
                            Def::Const(Const::Int { value, .. }) => i64::try_from(*value)
                                .unwrap_or_else(|_| panic!("glcc native backend: 65+-bit immediate not yet supported")),
                            Def::Const(Const::Null) => 0,
                            d => panic!("glcc native backend: OpndImm on non-constant {d:?}"),
                        };
                        Operand::Imm(value)
                    }
                    SlotSel::Imm(v) => Operand::Imm(*v),
                    SlotSel::MemOf(i) => Operand::Mem(sel.mem_of(mir, operands[*i as usize])),
                })
                .collect();
            sel.emit(MInst::new(e.op, width, slots));
        }
        return true;
    }
    false
}

fn shape_matches(mir: &Function, v: Option<Value>, shape: &Shape) -> bool {
    let Some(v) = v else { return matches!(shape, Shape::Any) };
    let is_const = matches!(mir.values[v].def, Def::Const(_));
    let value = match &mir.values[v].def {
        Def::Const(Const::Int { value, .. }) => Some(*value),
        Def::Const(Const::Null) => Some(0),
        _ => None,
    };
    match shape {
        Shape::Any => true,
        Shape::ConstIn(lo, hi) => value.is_some_and(|c| c >= i128::from(*lo) && c <= i128::from(*hi)),
        Shape::ConstIs(c) => value == Some(i128::from(*c)),
        Shape::NonConst => !is_const,
    }
}

/// Bits of a scalar integer/pointer/bool type, 0 for anything else.
pub fn scalar_bits(ty: &glcc_mir::Ty) -> u32 {
    match ty {
        glcc_mir::Ty::Int { bits } | glcc_mir::Ty::Ptr { bits, .. } => *bits,
        glcc_mir::Ty::Bool => 8,
        glcc_mir::Ty::Float(glcc_mir::FloatKind::F32) => 32,
        glcc_mir::Ty::Float(glcc_mir::FloatKind::F64) => 64,
        _ => 0,
    }
}

pub fn value_width(ty: &glcc_mir::Ty) -> crate::machine::Width {
    match scalar_bits(ty) {
        b if (1..=8).contains(&b) => crate::machine::Width::W8,
        b if (9..=16).contains(&b) => crate::machine::Width::W16,
        b if (17..=32).contains(&b) => crate::machine::Width::W32,
        b if (33..=64).contains(&b) => crate::machine::Width::W64,
        b => panic!("glcc native backend: unsupported scalar width {b} for type {:?}", ty),
    }
}

/// regalloc2 requires critical edges (multi-way branch into a block
/// with several predecessors) to be split: insert a forwarding block
/// per edge. Targets' branch emission never sees these; they only
/// forward block params.
fn split_critical_edges(f: &mut MFunction) {
    let npreds: Vec<usize> = {
        let mut p = vec![0usize; f.blocks.len()];
        for b in &f.blocks {
            for &s in &b.succs {
                p[s as usize] += 1;
            }
        }
        p
    };
    let original = f.blocks.len();
    for bi in 0..original {
        if f.blocks[bi].succs.len() <= 1 {
            continue;
        }
        for edge in 0..f.blocks[bi].succs.len() {
            let succ = f.blocks[bi].succs[edge];
            if npreds[succ as usize] <= 1 {
                continue;
            }
            // Thread the edge's original argument vregs through the
            // forwarding block unchanged: they are defined in bi and
            // feed succ's block params. Fresh vregs here would have no
            // definition at all (the split edge would carry them into
            // their own defining block), and regalloc2 then reports
            // them as live-in to the entry.
            let args = f.blocks[bi].insts.last().and_then(|i| i.branch.as_ref()).map(|br| br.args[edge].clone()).unwrap_or_default();
            let new: BlockId = f.blocks.len() as u32;
            let mut jmp = MInst::new(OpId::JMP, crate::machine::Width::W64, smallvec![]);
            jmp.branch = Some(crate::machine::BranchInfo {
                targets: smallvec![succ],
                args: vec![args],
            });
            f.blocks.push(MBlock {
                params: Vec::new(),
                insts: vec![jmp],
                succs: smallvec![succ],
            });
            let last = f.blocks[bi].insts.last_mut().expect("multi-succ block ends in a branch");
            let br = last.branch.as_mut().expect("multi-succ block ends in a branch");
            br.targets[edge] = new;
            // The forwarding block has no params: the original args
            // move to its jump.
            br.args[edge] = smallvec::smallvec![];
            f.blocks[bi].succs[edge] = new;
        }
    }
}
