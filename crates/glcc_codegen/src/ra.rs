//! Register allocation over the machine IR, via `regalloc2`. The
//! adapter derives each instruction's operand list and constraints from
//! the target's opcode table, so targets never talk to the allocator
//! directly — they only describe their opcodes.

use regalloc2::{
    Allocation, Edit, Function as RaFunctionTrait, Inst as RaInst, InstRange, MachineEnv,
    Operand as RaOperand, OperandConstraint, OperandKind, OperandPos, Output, PReg as RaPReg,
    PRegSet, RegClass as RaRegClass, RegallocOptions, VReg as RaVReg,
};
use smallvec::SmallVec;

use crate::isa::{Constraint, OpcodeTable, RegInfo, SlotKind, TargetIsa};
use crate::machine::{BlockId, MBlock, MFunction, MInst, OpId, Operand, PReg, Reg, RegClass, VReg};

/// Run allocation over `f` and rewrite it in place: every virtual
/// register operand resolves to a physical register or spill slot, the
/// allocator's moves are interleaved, and its spill slots are appended
/// to the frame. User frame slots keep their indices.
pub fn allocate(f: &mut MFunction, isa: &'static dyn TargetIsa) {
    if std::env::var_os("GLCC_RA_VERBOSE").is_some() {
        unsafe {
            let _ = log::set_logger_racy(&VERBOSE_LOGGER);
        }
        log::set_max_level(log::LevelFilter::Trace);
    }
    let env = machine_env(isa.reginfo());
    if std::env::var_os("GLCC_NATIVE_TRACE").is_some() {
        eprintln!("TRACE: regalloc2::run {}", f.name);
    }
    let out = {
        let ra = Adapter::new(f, isa);
        let mut options = RegallocOptions::default();
        options.verbose_log = std::env::var_os("GLCC_RA_VERBOSE").is_some();
        options.validate_ssa = cfg!(debug_assertions);
        regalloc2::run(&ra, &env, &options).expect("regalloc2")
    };

    if std::env::var_os("GLCC_NATIVE_TRACE").is_some() {
        eprintln!("TRACE: regalloc2 done {}", f.name);
    }
    let reginfo = isa.reginfo();
    let resolve = |alloc: Allocation| -> Reg {
        match alloc.kind() {
            regalloc2::AllocationKind::None => panic!("glcc native backend: unresolved allocation"),
            regalloc2::AllocationKind::Reg => {
                let preg = alloc.as_reg().expect("reg allocation");
                let class = match preg.class() {
                    regalloc2::RegClass::Int => crate::machine::RegClass::Gpr,
                    _ => crate::machine::RegClass::Vec,
                };
                Reg::P(crate::machine::PReg { hw: preg.hw_enc() as u8, class })
            }
            regalloc2::AllocationKind::Stack => Reg::Spill(alloc.as_stack().expect("stack allocation").index() as u32),
        }
    };

    // Edits regalloc2 places after a block-ending branch implement the
    // successor block params' moves; on ISAs where the branch transfers
    // control immediately (all of ours), they must execute before it.
    let mut ends_in_branch = Vec::with_capacity(f.blocks.len());
    if std::env::var_os("GLCC_RA_DEBUG").is_some() {
        for (i, offs) in out.inst_alloc_offsets.iter().enumerate() {
            let end = out.inst_alloc_offsets.get(i + 1).map(|o| *o as usize).unwrap_or(out.allocs.len());
            let n = end - *offs as usize;
            let want = ra_operands_of(f, isa, i);
            if n != want {
                let mut seen = 0usize;
                'outer: for b in &f.blocks {
                    for inst in &b.insts {
                        if seen == i {
                            eprintln!("RA DEBUG: inst {i} has {n} allocs but {want} operands: {:?}", inst);
                            break 'outer;
                        }
                        seen += 1;
                    }
                }
            }
        }
    }
    let plans: Vec<Vec<MInst>> = {
        let ra = Adapter::new(f, isa);
        f.blocks
            .iter()
            .enumerate()
            .map(|(bi, _)| {
                ends_in_branch.push(f.blocks[bi].insts.last().is_some_and(|i| i.branch.is_some()));
                let mut insts = Vec::with_capacity(f.blocks[bi].insts.len());
                let seq_debug = std::env::var_os("GLCC_RA_SEQ").is_some();
                for item in out.block_insts_and_edits(&ra, regalloc2::Block::new(bi)) {
                    if seq_debug {
                        match item {
                            regalloc2::InstOrEdit::Inst(i) => eprintln!("RA SEQ b{bi}: inst {}", i.index()),
                            regalloc2::InstOrEdit::Edit(Edit::Move { from, to }) => {
                                eprintln!("RA SEQ b{bi}: edit {:?} -> {:?}", from, to)
                            }
                            _ => eprintln!("RA SEQ b{bi}: other edit"),
                        }
                    }
                    match item {
                        regalloc2::InstOrEdit::Inst(i) => {
                            let (mb, mi) = ra.inst_loc[i.index()];
                            let mut inst = f.blocks[mb as usize].insts[mi].clone();
                            apply_allocs(&mut inst, out.inst_allocs(i), &resolve);
                            insts.push(inst);
                        }
                        regalloc2::InstOrEdit::Edit(Edit::Move { from, to }) => {
                            insts.push(MInst::new(
                                OpId::MOVE,
                                crate::machine::Width::W64,
                                smallvec::smallvec![Operand::Reg(resolve(*to)), Operand::Reg(resolve(*from))],
                            ));
                        }
                    }
                }
                insts
            })
            .collect()
    };
    // Insert call-argument moves: each call's register arguments go
    // into their ABI registers immediately before the call, as one
    // parallel-move set (a pin's destination may be another pin's
    // source). The moves' sources and targets are physical post-RA.
    let reginfo = isa.reginfo();
    let resolve_run = |pending: &mut Vec<(Reg, Reg)>| -> Vec<(Reg, Reg)> {
        let mut moves: Vec<(Reg, Reg)> = Vec::with_capacity(pending.len());
        while let Some(i) = (0..pending.len()).find(|&i| {
            let (dst, _) = pending[i];
            !pending.iter().enumerate().any(|(j, (_, src))| j != i && *src == dst)
        }) {
            moves.push(pending.remove(i));
        }
        let mut guard = 0usize;
        while !pending.is_empty() {
            guard += 1;
            if guard > 100 {
                eprintln!("ARGMOVE HANG: pending={:?} moves={:?}", pending, moves);
                panic!("resolve_run spin");
            }
            let (dst, src) = pending[0];
            let scratch = match reginfo.scratch[match src {
                Reg::P(p) => p.class as usize,
                _ => RegClass::Gpr as usize,
            }] {
                Some(p) => p,
                None => panic!("glcc native backend: no scratch register for call-arg move cycle"),
            };
            moves.push((Reg::P(scratch), src));
            pending[0] = (dst, Reg::P(scratch));
            while let Some(i) = (0..pending.len()).find(|&i| {
                let (dst, _) = pending[i];
                !pending.iter().enumerate().any(|(j, (_, src))| j != i && *src == dst)
            }) {
                moves.push(pending.remove(i));
            }
        }
        moves
    };
    let mut plans = plans;
    if std::env::var_os("GLCC_NATIVE_TRACE").is_some() {
        eprintln!("TRACE: inserting argmoves {}", f.name);
    }
    if std::env::var_os("GLCC_NO_ARGMOVES").is_none() {
    for plan in &mut plans {
        let mut out: Vec<MInst> = Vec::with_capacity(plan.len() + 8);
        for inst in plan.drain(..) {
            if let Some(pregs) = &inst.call_args {
                let regs: Vec<Reg> = inst
                    .ops
                    .iter()
                    .filter_map(|o| match o {
                        Operand::Reg(r @ Reg::P(_)) => Some(*r),
                        _ => None,
                    })
                    .collect();
                let args = &regs[regs.len() - pregs.len()..];
                let mut pending: Vec<(Reg, Reg)> = Vec::new();
                for (arg, preg) in args.iter().zip(pregs.iter()) {
                    if *arg != Reg::P(*preg) {
                        pending.push((Reg::P(*preg), *arg));
                    }
                }
                if pending.len() == 1 {
                    out.push(MInst::new(
                        OpId::MOVE,
                        crate::machine::Width::W64,
                        smallvec::smallvec![Operand::Reg(pending[0].0), Operand::Reg(pending[0].1)],
                    ));
                } else if pending.len() > 1 {
                    for (dst, src) in resolve_run(&mut pending) {
                        out.push(MInst::new(
                            OpId::MOVE,
                            crate::machine::Width::W64,
                            smallvec::smallvec![Operand::Reg(dst), Operand::Reg(src)],
                        ));
                    }
                }
            }
            out.push(inst);
        }
        *plan = out;
    }
    if std::env::var_os("GLCC_NATIVE_TRACE").is_some() {
        eprintln!("TRACE: argmoves done {}", f.name);
    }
    }

    for (bi, plan) in plans.into_iter().enumerate() {
        let block = &mut f.blocks[bi];
        if ends_in_branch[bi] {
            // Edits regalloc2 places after a block-ending branch
            // implement the successor block params' moves; on ISAs
            // where the branch transfers control immediately (all of
            // ours), they must execute before it.
            let branch_at = plan.iter().rposition(|i| i.branch.is_some()).expect("branch block keeps its branch");
            let mut ordered = plan;
            let branch = ordered.remove(branch_at);
            ordered.push(branch);
            block.insts = ordered;
        } else {
            block.insts = plan;
        }
    }

    if std::env::var_os("GLCC_NATIVE_TRACE").is_some() {
        eprintln!("TRACE: spillslots {} = {}", f.name, out.num_spillslots);
    }
    f.frame.spill_start = f.frame.slots.len() as u32;
    for _ in 0..out.num_spillslots {
        f.frame.slots.push(crate::machine::Slot { size: 8, align: 8, offset: 0 });
    }

    if std::env::var_os("GLCC_NATIVE_TRACE").is_some() {
        eprintln!("TRACE: writeback done {}", f.name);
    }
    resolve_entry_pins(f, isa);
}

/// The entry block's leading register-to-register MOVEs are the ABI
/// parameter pins: one pin's destination may be another pin's source,
/// so they must execute as the parallel move they semantically are.
/// (Call-argument moves are inserted separately, one resolved set per
/// call.)
fn resolve_entry_pins(f: &mut MFunction, isa: &'static dyn TargetIsa) {
    if std::env::var_os("GLCC_NATIVE_TRACE").is_some() {
        eprintln!("TRACE: entry pins {}", f.name);
    }
    let reginfo = isa.reginfo();
    let resolve_run = |pending: &mut Vec<(Reg, Reg)>| -> Vec<(Reg, Reg)> {
        let mut moves: Vec<(Reg, Reg)> = Vec::with_capacity(pending.len());
        let mut guard0 = 0usize;
        while let Some(i) = (0..pending.len()).find(|&i| {
            let (dst, _) = pending[i];
            matches!(dst, Reg::Spill(_)) || !pending.iter().enumerate().any(|(j, (_, src))| j != i && *src == dst)
        }) {
            guard0 += 1;
            if guard0 > 1000 {
                panic!("first find spin: {:?}", pending);
            }
            moves.push(pending.remove(i));
        }
        while !pending.is_empty() {
            // Cycle: stash one source in the scratch register of its
            // class and continue with the copy retargeted at it.
            let (dst, src) = pending[0];
            let scratch = match reginfo.scratch[match src {
                Reg::P(p) => p.class as usize,
                _ => RegClass::Gpr as usize,
            }] {
                Some(p) => p,
                None => panic!("glcc native backend: no scratch register for move cycle"),
            };
            moves.push((Reg::P(scratch), src));
            pending[0] = (dst, Reg::P(scratch));
            while let Some(i) = (0..pending.len()).find(|&i| {
                let (dst, _) = pending[i];
                matches!(dst, Reg::Spill(_)) || !pending.iter().enumerate().any(|(j, (_, src))| j != i && *src == dst)
            }) {
                moves.push(pending.remove(i));
            }
        }
        moves
    };

    let block = &mut f.blocks[0];
    // The pins are the entry block's LEADING run of reg-to-reg MOVEs;
    // later MOVEs (call-argument moves) belong to other parallel sets
    // and must not be merged into this one.
    let mut pending: Vec<(Reg, Reg)> = Vec::new();
    let mut run_end = 0usize;
    while run_end < block.insts.len() && block.insts[run_end].op == OpId::MOVE {
        match (&block.insts[run_end].ops[0], &block.insts[run_end].ops[1]) {
            (Operand::Reg(dst), Operand::Reg(Reg::P(src))) => {
                if *dst != Reg::P(*src) {
                    pending.push((*dst, Reg::P(*src)));
                }
                run_end += 1;
            }
            _ => break,
        }
    }
    if pending.len() >= 2 {
        let encoded: Vec<MInst> = resolve_run(&mut pending)
            .into_iter()
            .filter(|(dst, src)| dst != src)
            .map(|(dst, src)| {
                MInst::new(
                    OpId::MOVE,
                    crate::machine::Width::W64,
                    smallvec::smallvec![Operand::Reg(dst), Operand::Reg(src)],
                )
            })
            .collect();
        let tail = block.insts.split_off(run_end);
        block.insts.truncate(0);
        block.insts.extend(encoded);
        block.insts.extend(tail);
    }
}

/// Operand count regalloc2 saw for flattened inst `i` (debug aid).
fn ra_operands_of(f: &MFunction, isa: &'static dyn TargetIsa, i: usize) -> usize {
    let mut n = 0usize;
    let mut seen = 0usize;
    for b in &f.blocks {
        for inst in &b.insts {
            if seen == i {
                n = count_ra_operands(f, isa, inst);
            }
            seen += 1;
        }
    }
    n
}

fn count_ra_operands(_f: &MFunction, _isa: &'static dyn TargetIsa, inst: &MInst) -> usize {
    // What apply_allocs will map: every Reg::V in the operand list,
    // plus vreg bases/indices inside Mem operands.
    let mut n = 0usize;
    for op in &inst.ops {
        match op {
            Operand::Reg(Reg::V(_)) => n += 1,
            Operand::Mem(m) => {
                for r in [m.base, m.index] {
                    if matches!(r, Some(Reg::V(_))) {
                        n += 1;
                    }
                }
            }
            _ => {}
        }
    }
    n
}

fn apply_allocs(inst: &mut MInst, allocs: &[Allocation], resolve: &impl Fn(Allocation) -> Reg) {
    // Allocations arrive in the order regalloc2 saw operands; walk the
    // instruction rebuilding only the vreg operands in that order.
    let ops_snapshot: Vec<Operand> = inst.ops.to_vec();
    let mut next = 0usize;
    let mut map = |r: &mut Reg| {
        if let Reg::V(_) = *r {
            if next >= allocs.len() {
                panic!(
                    "glcc native backend: operand/alloc mismatch on opcode {:?} ops {:?} ({next} vreg operands, {} allocs)",
                    inst.op,
                    ops_snapshot,
                    allocs.len()
                );
            }
            *r = resolve(allocs[next]);
            next += 1;
        }
    };
    for op in inst.ops.iter_mut() {
        match op {
            Operand::Reg(r) | Operand::FixedReg(r, _) => map(r),
            Operand::Mem(m) => {
                if let Some(base) = &mut m.base {
                    map(base);
                }
                if let Some(index) = &mut m.index {
                    map(index);
                }
            }
            _ => {}
        }
    }
}

fn machine_env(reginfo: &RegInfo) -> MachineEnv {
    let set = |regs: &[PReg]| -> PRegSet {
        let mut s = PRegSet::empty();
        for &p in regs {
            s.add(RaPReg::new(p.hw as usize, ra_class(p.class)));
        }
        s
    };
    MachineEnv {
        preferred_regs_by_class: [set(&reginfo.preferred[0]), set(&reginfo.preferred[1]), PRegSet::empty()],
        non_preferred_regs_by_class: [set(&reginfo.non_preferred[0]), set(&reginfo.non_preferred[1]), PRegSet::empty()],
        scratch_by_class: reginfo.scratch.map(|p| p.map(|p| RaPReg::new(p.hw as usize, ra_class(p.class)))),
        fixed_stack_slots: Vec::new(),
    }
}

fn ra_class(c: RegClass) -> RaRegClass {
    match c {
        RegClass::Gpr => RaRegClass::Int,
        // Scalar floats ride regalloc2's Float class (index 1, matching
        // the MachineEnv's per-class arrays).
        RegClass::Vec => RaRegClass::Float,
    }
}

struct Adapter<'a> {
    f: &'a MFunction,
    ops: &'static OpcodeTable,
    reginfo: &'static RegInfo,
    ret_preg: PReg,
    vec_ret_preg: PReg,
    /// regalloc2 instruction id → (block, index within block).
    inst_loc: Vec<(BlockId, usize)>,
    block_start: Vec<usize>,
    /// Per-ra-inst regalloc2 operands, flattened.
    operands: Vec<RaOperand>,
    op_offsets: Vec<u32>,
    /// Per-ra-inst per-successor branch args, empty for non-branches.
    branch_args: Vec<Vec<SmallVec<[RaVReg; 4]>>>,
    clobbers: Vec<PRegSet>,
    succs: Vec<Vec<regalloc2::Block>>,
    preds: Vec<Vec<regalloc2::Block>>,
    block_params: Vec<Vec<RaVReg>>,
}

impl<'a> Adapter<'a> {
    /// Register class the selection assigned to `v` (floats live in
    /// the Vec class; everything else is Gpr).
    fn vclass(&self, v: VReg) -> RaRegClass {
        ra_class(self.f.vreg_class[v.0 as usize])
    }

    fn new(f: &'a MFunction, isa: &'static dyn TargetIsa) -> Self {
        let ops = isa.opcodes();
        let reginfo = isa.reginfo();
        let mut a = Adapter {
            f,
            ops,
            reginfo,
            ret_preg: isa.abi().ret_reg(),
            vec_ret_preg: isa.abi().vec_ret_reg(),
            inst_loc: Vec::new(),
            block_start: Vec::new(),
            operands: Vec::new(),
            op_offsets: Vec::new(),
            branch_args: Vec::new(),
            clobbers: Vec::new(),
            succs: f.blocks.iter().map(|b| b.succs.iter().map(|s| regalloc2::Block::new(*s as usize)).collect()).collect(),
            preds: vec![Vec::new(); f.blocks.len()],
            block_params: Vec::new(),
        };
        a.block_params = f
            .blocks
            .iter()
            .map(|b| b.params.iter().map(|v| RaVReg::new(v.0 as usize, a.vclass(*v))).collect())
            .collect();
        for (bi, block) in f.blocks.iter().enumerate() {
            for &s in &block.succs {
                a.preds[s as usize].push(regalloc2::Block::new(bi));
            }
        }
        for (bi, block) in f.blocks.iter().enumerate() {
            a.block_start.push(a.inst_loc.len());
            for (ii, inst) in block.insts.iter().enumerate() {
                a.inst_loc.push((bi as BlockId, ii));
                let start = a.operands.len() as u32;
                a.collect_operands(inst);
                a.op_offsets.push(start);
                let mut clob = PRegSet::empty();
                let spec = ops.spec(inst.op);
                if spec.flags.call {
                    // A call result pinned in a register (FixedReg def)
                    // is written by the call: that preg must not also
                    // be a clobber (regalloc2 forbids the collision).
                    let mut reserved: Vec<PReg> = Vec::new();
                    let mut past_target = false;
                    for op in &inst.ops {
                        match op {
                            // The target (direct symbol or indirect
                            // register) separates result defs from
                            // argument uses.
                            Operand::Sym(_) | Operand::Reg(_) => past_target = true,
                            Operand::FixedReg(_, preg) if !past_target => reserved.push(*preg),
                            _ => {}
                        }
                    }
                    for &p in spec.flags.clobbers {
                        if reserved.contains(&p) {
                            continue;
                        }
                        clob.add(RaPReg::new(p.hw as usize, ra_class(p.class)));
                    }
                }
                a.clobbers.push(clob);
                a.branch_args.push(inst.branch.as_ref().map(|br| {
                    br.args.iter().map(|edge| edge.iter().map(|v| RaVReg::new(v.0 as usize, a.vclass(*v))).collect()).collect()
                }).unwrap_or_default());
            }
        }

        if std::env::var_os("GLCC_RA_DEBUG2").is_some() {
            let mut start = 0usize;
            for (ri, chunk) in a.op_offsets.iter().enumerate() {
                let end = a.op_offsets.get(ri + 1).copied().unwrap_or(a.operands.len() as u32) as usize;
                let (bi, ii) = a.inst_loc[ri];
                eprintln!("RA2: inst {ri} (b{bi}.{ii}) op {:?}", f.blocks[bi as usize].insts[ii].op);
                for op in &a.operands[*chunk as usize..end] {
                    let _ = start;
                    eprintln!("      {:?} v{} c{:?}", op.kind(), op.vreg().vreg(), op.constraint());
                }
                start += 1;
            }
        }
        if std::env::var_os("GLCC_RA_DEBUG").is_some() {
            eprintln!("RA DEBUG: vreg_class = {:?}", f.vreg_class);
            let mut refs = vec![0usize; f.num_vregs as usize];
            for op in a.operands.iter() {
                refs[op.vreg().vreg()] += 1;
            }
            for (bi, b) in f.blocks.iter().enumerate() {
                for v in &b.params {
                    refs[v.0 as usize] += 1;
                }
            }
            for br in &a.branch_args {
                for edge in br {
                    for v in edge {
                        refs[v.vreg()] += 1;
                    }
                }
            }
            for (v, n) in refs.iter().enumerate() {
                if *n == 0 {
                    eprintln!("RA DEBUG: vreg {v} referenced {n} times");
                }
            }
        }
        a
    }

    /// Build the regalloc2 operand list of one machine instruction.
    /// Framework pseudos are handled here; target opcodes go through
    /// their slot table. Physical registers in register slots are
    /// invisible to the allocator (the value they carry is guaranteed
    /// by the ABI or the encoding, not by allocation); frame-index
    /// bases are resolved post-RA.
    fn collect_operands(&mut self, inst: &MInst) {
        match inst.op {
            OpId::MOVE => {
                let dst = match &inst.ops[0] {
                    Operand::Reg(Reg::V(v)) => Some(*v),
                    _ => None,
                };
                let (Some(dst), Operand::Reg(src)) = (dst, &inst.ops[1]) else {
                    // Physical destination (call argument setup): only
                    // the source use is tracked.
                    if let Operand::Reg(Reg::V(v)) = &inst.ops[1] {
                        self.operands.push(RaOperand::new(
                            RaVReg::new(v.0 as usize, self.vclass(*v)),
                            OperandConstraint::Any,
                            OperandKind::Use,
                            OperandPos::Early,
                        ));
                    }
                    return;
                };
                match src {
                    // Move from a physical register (entry parameter,
                    // call result): the move itself defines the value,
                    // so the destination is unconstrained.
                    Reg::P(_) => self.operands.push(RaOperand::new(
                        RaVReg::new(dst.0 as usize, self.vclass(dst)),
                        OperandConstraint::Any,
                        OperandKind::Def,
                        OperandPos::Early,
                    )),
                    Reg::V(src) => {
                        // A plain copy: dst unconstrained, src read as
                        // is. (A Reuse-tied def here would assert in
                        // regalloc2 when the copy feeds a fixed use.)
                        self.operands.push(RaOperand::new(
                            RaVReg::new(dst.0 as usize, self.vclass(dst)),
                            OperandConstraint::Any,
                            OperandKind::Def,
                            OperandPos::Early,
                        ));
                        self.operands.push(RaOperand::new(
                            RaVReg::new(src.0 as usize, self.vclass(*src)),
                            OperandConstraint::Any,
                            OperandKind::Use,
                            OperandPos::Early,
                        ));
                    }
                    // Allocator-emitted move out of a spill slot: the
                    // destination is unconstrained.
                    Reg::Spill(_) => self.operands.push(RaOperand::new(
                        RaVReg::new(dst.0 as usize, self.vclass(dst)),
                        OperandConstraint::Any,
                        OperandKind::Def,
                        OperandPos::Early,
                    )),
                    Reg::FI(_) => panic!("glcc native backend: MOVE from frame slot"),
                }
                return;
            }
            OpId::RET => {
                // One return value pinned to the ABI return register;
                // aggregate returns pin each piece register explicitly.
                for op in inst.ops.iter() {
                    match op {
                        Operand::Reg(Reg::V(v)) => {
                            let class = self.vclass(*v);
                            let ret = if class == RaRegClass::Vector { self.vec_ret_preg } else { self.ret_preg };
                            self.operands.push(RaOperand::new(
                                RaVReg::new(v.0 as usize, class),
                                OperandConstraint::FixedReg(RaPReg::new(ret.hw as usize, class)),
                                OperandKind::Use,
                                OperandPos::Late,
                            ));
                        }
                        Operand::FixedReg(Reg::V(v), preg) => {
                            self.operands.push(RaOperand::new(
                                RaVReg::new(v.0 as usize, ra_class(preg.class)),
                                OperandConstraint::FixedReg(RaPReg::new(preg.hw as usize, ra_class(preg.class))),
                                OperandKind::Use,
                                OperandPos::Late,
                            ));
                        }
                        other => panic!("glcc native backend: ret operand {other:?}"),
                    }
                }
                return;
            }
            OpId::SPILL | OpId::RELOAD | OpId::JMP | OpId::UNREACHABLE => return,
            _ => {}
        }

        let spec = self.ops.spec(inst.op);
        if spec.flags.call {
            // Calls take their arguments/results as dynamically-many
            // pinned operands: FixedReg uses in ABI order, plus the
            // result def in the return register.
            // FixedReg operands before the target are defs (the call
            // result in the return register), the rest are uses.
            let mut past_target = false;
            for op in &inst.ops {
                match op {
                    Operand::Sym(_) => past_target = true,
                    Operand::FixedReg(Reg::V(v), preg) => {
                        let (kind, pos) = if past_target {
                            (OperandKind::Use, OperandPos::Early)
                        } else {
                            (OperandKind::Def, OperandPos::Late)
                        };
                        self.operands.push(RaOperand::new(
                            RaVReg::new(v.0 as usize, ra_class(preg.class)),
                            OperandConstraint::FixedReg(RaPReg::new(preg.hw as usize, ra_class(preg.class))),
                            kind,
                            pos,
                        ));
                    }
                    // The indirect callee is an ordinary register use.
                    Operand::Reg(Reg::V(v)) => {
                        past_target = true;
                        self.operands.push(RaOperand::new(
                            RaVReg::new(v.0 as usize, self.vclass(*v)),
                            OperandConstraint::Reg,
                            OperandKind::Use,
                            OperandPos::Early,
                        ));
                    }
                    Operand::Reg(Reg::P(_)) | Operand::Imm(_) => {}
                    other => panic!("glcc native backend: call operand {other:?}"),
                }
            }
            return;
        }
        for (slot, op) in spec.slots.iter().zip(&inst.ops) {
            match (slot.kind, op) {
                (SlotKind::Reg(_), Operand::Reg(Reg::V(v))) => {
                    let constraint = match slot.constraint {
                        Constraint::Any => OperandConstraint::Any,
                        Constraint::Reg => OperandConstraint::Reg,
                        Constraint::Fixed(p) => OperandConstraint::FixedReg(RaPReg::new(p.hw as usize, ra_class(p.class))),
                        Constraint::Reuse(n) => OperandConstraint::Reuse(n as usize),
                        Constraint::ImmRange(..) => unreachable!("imm range on register slot"),
                    };
                    // Reuse defs land late (regalloc2 requires it);
                    // other defs land late so they may reuse an early
                    // use's register.
                    let (kind, pos) = if slot.def {
                        (OperandKind::Def, OperandPos::Late)
                    } else {
                        (OperandKind::Use, OperandPos::Early)
                    };
                    self.operands.push(RaOperand::new(RaVReg::new(v.0 as usize, self.vclass(*v)), constraint, kind, pos));
                }
                (SlotKind::Reg(_), Operand::Reg(_)) => {}
                // Reg-or-memory x86 operand forms: the memory side
                // contributes no operand of its own (its bases are
                // collected below).
                (SlotKind::Reg(_), Operand::Mem(_)) => {}
                (SlotKind::Imm, Operand::Imm(_)) | (SlotKind::FrameIndex, Operand::FrameIndex(_)) | (SlotKind::Sym, Operand::Sym(_)) => {}
                (SlotKind::Mem, Operand::Mem(_)) => {}
                _ => panic!("glcc native backend: opcode {} slot {slot:?} got {op:?}", spec.name),
            }
        }
        // Memory operands' base/index registers are ordinary uses.
        for op in &inst.ops {
            if let Operand::Mem(m) = op {
                for r in [m.base, m.index] {
                    if let Some(Reg::V(v)) = r {
                        self.operands.push(RaOperand::new(
                            RaVReg::new(v.0 as usize, self.vclass(v)),
                            OperandConstraint::Reg,
                            OperandKind::Use,
                            OperandPos::Early,
                        ));
                    }
                }
            }
        }
    }
}

impl<'a> RaFunctionTrait for Adapter<'a> {
    fn num_insts(&self) -> usize {
        self.inst_loc.len()
    }

    fn num_blocks(&self) -> usize {
        self.f.blocks.len()
    }

    fn entry_block(&self) -> regalloc2::Block {
        regalloc2::Block::new(0)
    }

    fn block_insns(&self, block: regalloc2::Block) -> InstRange {
        let b = block.index();
        let start = self.block_start[b];
        let end = if b + 1 < self.block_start.len() { self.block_start[b + 1] } else { self.inst_loc.len() };
        InstRange::new(RaInst::new(start), RaInst::new(end))
    }

    fn block_succs(&self, block: regalloc2::Block) -> &[regalloc2::Block] {
        &self.succs[block.index()]
    }

    fn block_preds(&self, block: regalloc2::Block) -> &[regalloc2::Block] {
        &self.preds[block.index()]
    }

    fn block_params(&self, block: regalloc2::Block) -> &[RaVReg] {
        &self.block_params[block.index()]
    }

    fn is_ret(&self, insn: RaInst) -> bool {
        let op = self.mir_inst(insn).op;
        op == OpId::RET || op == OpId::UNREACHABLE
    }

    fn is_branch(&self, insn: RaInst) -> bool {
        self.mir_inst(insn).branch.is_some()
    }

    fn branch_blockparams(&self, _block: regalloc2::Block, insn: RaInst, succ_idx: usize) -> &[RaVReg] {
        &self.branch_args[insn.index()][succ_idx]
    }

    fn inst_operands(&self, insn: RaInst) -> &[RaOperand] {
        let start = self.op_offsets[insn.index()] as usize;
        let end = if insn.index() + 1 < self.op_offsets.len() {
            self.op_offsets[insn.index() + 1] as usize
        } else {
            self.operands.len()
        };
        &self.operands[start..end]
    }

    fn inst_clobbers(&self, insn: RaInst) -> PRegSet {
        self.clobbers[insn.index()]
    }

    fn num_vregs(&self) -> usize {
        self.f.num_vregs as usize
    }

    fn spillslot_size(&self, class: RaRegClass) -> usize {
        match class {
            RaRegClass::Float | RaRegClass::Vector => 2,
            _ => 1,
        }
    }
}

impl<'a> Adapter<'a> {
    fn mir_inst(&self, insn: RaInst) -> &MInst {
        let (bi, ii) = self.inst_loc[insn.index()];
        &self.f.blocks[bi as usize].insts[ii]
    }
}

static VERBOSE_LOGGER: VerboseLogger = VerboseLogger;
struct VerboseLogger;
impl log::Log for VerboseLogger {
    fn enabled(&self, meta: &log::Metadata) -> bool {
        meta.level() <= log::Level::Trace
    }
    fn log(&self, record: &log::Record) {
        eprintln!("RA: {}", record.args());
    }
    fn flush(&self) {}
}
