//! MIR + interface → PrismIR: the structurizer and op lowering.
//!
//! The prism profile guarantees a reducible, goto-free CFG, so the
//! MIR control flow graph reconstructs into a region tree directly:
//! post-dominators give each branch its join, dominator back edges
//! give the loops, and every remaining edge is a break, continue, or
//! return. SSA joins live in the region statements (see
//! [`crate::ir::Stmt`]): an `if` carries one value per arm into its
//! join registers, a loop carries its per-iteration registers and
//! gathers one value per `break` into its exit registers.
//!
//! Ops lower width-typed: MIR's float and integer forms collapse into
//! one lane-wise [`Alu`] per operation (the operand register types
//! decide float vs integer), source swizzles absorb `ExtractLane`,
//! `FDiv` becomes `rcp` + `mul` (era ISAs have no divide), and the
//! fusion pass at the end sinks single-use copies and recovers `mad`
//! and the `dp3`/`dp4` dot products the frontend's `mul4x4` lowering
//! spread into extract/mul/add trees.

use std::collections::{HashMap, HashSet};
#[allow(unused_imports)]
use std::vec;
use std::format;

use smallvec::{smallvec, SmallVec};

use glcc_mir::ir::{Block, Const, Def, FcmpPred, Function, Global, IcmpPred, Inst, InstKind, Module, Terminator, Value};
use glcc_mir::GepIndex;
use glcc_mir::passes::DomInfo;

use crate::interface::Interface;
use crate::ir::{Alu, Body, CmpCode, Lod, Op, Program, RegTy, Scalar, Setp, Src, Stmt, Swizzle, Tex, VReg};

/// One MIR value's PrismIR shape: a register, an inline interface
/// source (inputs and uniforms read directly as operands), a
/// flattened aggregate of leaves, or an undefined value.
#[derive(Debug, Clone)]
enum PVal {
    V(VReg),
    Src(Src),
    Agg(Vec<PVal>),
    Undef,
    /// A global's address: samplers resolve through it at the sample;
    /// uniforms through loads. Any other use is a lowering bug.
    Global(Global),
}

pub fn build(interface: &Interface, module: &Module, func: &Function) -> Result<Program, String> {
    let mut program = Program {
        name: interface.entry.clone(),
        stage: interface.stage,
        interface: interface.clone(),
        regs: Vec::new(),
        body: Body::new(),
    };
    let mut b = FnBuilder {
        interface,
        module,
        program: &mut program,
        func,
        dom: DomInfo::compute(func, &func.reachable_blocks()),
        visited: HashSet::new(),
        orig_entry: Block::from_usize(0),
    };
    b.orig_entry = real_entry(func);

    // The entry parameters are the interface inputs: one aggregate of
    // input-register sources per parameter, flattened in leaf order
    // across all of them.
    let mut env: HashMap<Value, PVal> = HashMap::new();
    let mut input_index = 0u32;
    for param in func.blocks[b.orig_entry].params.clone() {
        let count = agg_leaf_count(&func.values[param].ty) as u32;
        if input_index + count > interface.inputs.len() as u32 {
            return Err("prism: entry parameters do not match the interface".to_string());
        }
        let sources: Vec<PVal> = (input_index..input_index + count)
            .map(|i| PVal::Src(Src::Input { input: i, swizzle: None, negate: false }))
            .collect();
        env.insert(param, PVal::Agg(sources));
        input_index += count;
    }
    if input_index != interface.inputs.len() as u32 {
        return Err("prism: entry parameters do not match the interface".to_string());
    }

    let stops: HashMap<Block, StopKind> = HashMap::new();
    let walk = b.walk(func.entry, &[], &mut env, &stops)?;
    if !walk.exits.is_empty() {
        return Err("prism: control flow leaves the entry point without a return".to_string());
    }
    program.body = walk.body;
    fuse(&mut program);
    Ok(program)
}

/// Why the structurizer stopped at a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StopKind {
    /// An `if` join in the current scope: arms record their edge args,
    /// no statement.
    Join,
    /// A loop header: the edge is a `continue` of that loop.
    Continue,
    /// A loop exit: the edge is a `break` of that loop.
    Break,
    /// Another `switch` case's body: reaching it is fallthrough,
    /// which prism does not support yet.
    CaseEntry,
}

/// What one structured walk produced.
struct Walked {
    body: Body,
    /// Stops reached from inside this subtree, with the edge args per
    /// incoming edge, in discovery order. Entries for the consuming
    /// scope are removed as they are consumed; entries for
    /// enclosing-scope stops propagate up.
    exits: HashMap<Block, Vec<SmallVec<[VReg; 4]>>>,
}

/// One level of a `switch` chain: the body built so far (this case's
/// `setp` and `if`, with the deeper chain nested in the `if`'s else)
/// plus what the chain's fall-through contributes to the switch join.
struct ChainOut {
    body: Body,
    join_entry: Option<SmallVec<[VReg; 4]>>,
    exits: HashMap<Block, Vec<SmallVec<[VReg; 4]>>>,
}

struct FnBuilder<'a> {
    interface: &'a Interface,
    module: &'a Module,
    program: &'a mut Program,
    func: &'a Function,
    dom: DomInfo,
    visited: HashSet<Block>,
    /// The real entry block: `func.entry` may be a prep-inserted shim
    /// (`br` into this block) so the real entry can host loop phis.
    orig_entry: Block,
}

impl FnBuilder<'_> {
    fn push_reg(&mut self, ty: RegTy) -> VReg {
        self.program.push_reg(ty)
    }

    fn reg_ty(&self, v: Value) -> RegTy {
        mir_reg_ty(&self.func.values[v].ty)
    }

    /// Materialize a value shape into a register (a `mov` when it is
    /// an inline source or an undefined leaf), emitting into `body`.
    fn materialize(&mut self, val: &PVal, ty: RegTy, body: &mut Body) -> Result<VReg, String> {
        match val {
            PVal::V(v) => Ok(*v),
            PVal::Undef => {
                let dst = self.push_reg(ty);
                body.stmts.push(Stmt::Op(Op { dst, kind: Alu::Mov, srcs: smallvec![Src::Undef] }));
                Ok(dst)
            }
            PVal::Src(src) => {
                let dst = self.push_reg(ty);
                body.stmts.push(Stmt::Op(Op { dst, kind: Alu::Mov, srcs: smallvec![src.clone()] }));
                Ok(dst)
            }
            PVal::Global(global) => Err(format!("prism: global '{}' used as a value (materialize)", self.global_name(*global))),
            PVal::Agg(_) => Err("prism: an aggregate flowed where a value is required".to_string()),
        }
    }

    /// Flatten a value shape into the register list a leaf position
    /// sequence needs (interface outputs, exit values).
    fn flatten_regs(&mut self, val: &PVal, tys: &[RegTy], body: &mut Body) -> Result<SmallVec<[VReg; 8]>, String> {
        let mut out = SmallVec::new();
        match val {
            PVal::Agg(leaves) => {
                for (index, leaf) in leaves.iter().enumerate() {
                    let ty = tys.get(index).ok_or("prism: aggregate shape mismatch")?.clone();
                    match leaf {
                        PVal::Agg(_) => return Err("prism: nested aggregate where a leaf is required".to_string()),
                        flat => out.push(self.materialize(flat, ty, body)?),
                    }
                }
            }
            flat => {
                let ty = tys.first().ok_or("prism: aggregate shape mismatch")?.clone();
                out.push(self.materialize(flat, ty, body)?);
            }
        }
        Ok(out)
    }

    /// Walk one structured region starting at `block`, entered with
    /// `entry_args`. Returns the statements built and the stops the
    /// subtree reached.
    fn walk(
        &mut self,
        start: Block,
        start_args: &[Value],
        env: &mut HashMap<Value, PVal>,
        stops: &HashMap<Block, StopKind>,
    ) -> Result<Walked, String> {
        self.walk_inner(start, start_args, env, stops, false)
    }

    /// Region entries (`entering`) skip the stop check: a loop's body
    /// walk starts at its own header, which is the `continue` target.
    fn walk_inner(
        &mut self,
        start: Block,
        start_args: &[Value],
        env: &mut HashMap<Value, PVal>,
        stops: &HashMap<Block, StopKind>,
        entering: bool,
    ) -> Result<Walked, String> {
        // An edge straight into a stop: record it (or emit its
        // statement) without entering the block.
        if !entering
            && let Some(kind) = stops.get(&start)
        {
            let mut body = Body::new();
            let args = self.edge_args(start_args, env, &mut body)?;
            return self.finish_stop(*kind, start, args, body, HashMap::new());
        }

        let mut body = Body::new();
        let mut exits: HashMap<Block, Vec<SmallVec<[VReg; 4]>>> = HashMap::new();
        let mut block = start;
        let mut entry_args: Vec<Value> = start_args.to_vec();
        let mut first = entering;
        loop {
            let natural = self.natural_loop(block);
            if !natural.is_empty() && !first {
                let built = self.build_loop(block, &entry_args, natural, env, stops)?;
                body.stmts.extend(built.body.stmts);
                merge_exits(&mut exits, built.exits);
                return Ok(Walked { body, exits });
            }
            first = false;
            if !self.visited.insert(block) {
                return Err("prism: control flow is not tree-structured (a block merged outside a join)".to_string());
            }
            // Single-predecessor block parameters rename directly.
            let params = self.func.blocks[block].params.clone();
            for (param, arg) in params.iter().zip(entry_args.iter()) {
                let value = self.pval_of(*arg, env)?;
                env.insert(*param, value);
            }
            let insts = self.func.blocks[block].insts.clone();
            for inst in insts {
                self.lower_inst(inst, env, &mut body)?;
            }
            let Some(term) = self.func.blocks[block].term.clone() else {
                return Err("prism: block without a terminator".to_string());
            };
            match term {
                Terminator::Return { val } => {
                    let outputs: Vec<RegTy> = self.interface.outputs.iter().map(|leaf| leaf.ty.clone()).collect();
                    let val = match val {
                        Some(v) => self.pval_of(v, env)?,
                        None => {
                            if outputs.is_empty() {
                                PVal::Agg(Vec::new())
                            } else {
                                return Err("prism: entry returned nothing but the interface has outputs".to_string());
                            }
                        }
                    };
                    let regs = self.flatten_regs(&val, &outputs, &mut body)?;
                    body.stmts.push(Stmt::Exit(regs));
                    return Ok(Walked { body, exits });
                }
                Terminator::Unreachable => return Ok(Walked { body, exits }),
                Terminator::Br { dest, args } => match stops.get(&dest) {
                    Some(kind) => {
                        let args = self.edge_args(&args, env, &mut body)?;
                        return self.finish_stop(*kind, dest, args, body, exits);
                    }
                    None => {
                        block = dest;
                        entry_args = args.to_vec();
                    }
                },
                Terminator::CondBr { cond, then_block, else_block, then_args, else_args } => {
                    if then_block == else_block {
                        // Degenerate branch: one continuation.
                        match stops.get(&then_block) {
                            Some(kind) => {
                                let args = self.edge_args(&then_args, env, &mut body)?;
                                return self.finish_stop(*kind, then_block, args, body, exits);
                            }
                            None => {
                                block = then_block;
                                entry_args = then_args.to_vec();
                                continue;
                            }
                        }
                    }
                    let join = self.branch_join(block, &[then_block, else_block], stops);
                    let mut arm_stops = stops.clone();
                    let mut join_is_new = false;
                    if let Some(join) = join
                        && !stops.contains_key(&join)
                    {
                        arm_stops.insert(join, StopKind::Join);
                        join_is_new = true;
                    }
                    let pred = self.setp_for_cond(cond, env, &mut body)?;
                    let mut then_env = env.clone();
                    let mut else_env = env.clone();
                    let then_walk = self.walk(then_block, &then_args, &mut then_env, &arm_stops)?;
                    let else_walk = self.walk(else_block, &else_args, &mut else_env, &arm_stops)?;
                    let mut then_body = then_walk.body;
                    let mut els_body = else_walk.body;
                    let mut then_exits = then_walk.exits;
                    let mut else_exits = else_walk.exits;
                    if !join_is_new {
                        // The arms reconverge at an enclosing scope's
                        // stop (or never reconverge): their statements
                        // (including breaks and returns) still belong
                        // under this if.
                        body.stmts.push(Stmt::If { pred, then: then_body, els: els_body, results: Vec::new() });
                        merge_exits(&mut exits, then_exits);
                        merge_exits(&mut exits, else_exits);
                        return Ok(Walked { body, exits });
                    }
                    let join = join.expect("join known");
                    let then_entry = then_exits.remove(&join).ok_or("prism: branch join not reached from the then arm")?;
                    let else_entry = else_exits.remove(&join).ok_or("prism: branch join not reached from the else arm")?;
                    let (then_entry, else_entry) = (single_entry(then_entry)?, single_entry(else_entry)?);
                    let params = self.func.blocks[join].params.clone();
                    if params.len() != then_entry.len() || params.len() != else_entry.len() {
                        return Err("prism: join arity mismatch".to_string());
                    }
                    let mut results = Vec::with_capacity(params.len());
                    for (index, param) in params.iter().enumerate() {
                        let join_reg = self.push_reg(self.reg_ty(*param));
                        results.push((join_reg, then_entry[index], else_entry[index]));
                        env.insert(*param, PVal::V(join_reg));
                    }
                    body.stmts.push(Stmt::If { pred, then: then_body, els: els_body, results });
                    merge_exits(&mut exits, then_exits);
                    merge_exits(&mut exits, else_exits);
                    block = join;
                    entry_args = Vec::new();
                }
                Terminator::Switch { scrutinee, cases, default } => {
                    if cases.is_empty() {
                        // `switch` with no cases is its default.
                        return self.walk(default, &[], env, stops).map(|walked| Walked {
                            body: {
                                let mut body = body;
                                body.stmts.extend(walked.body.stmts);
                                body
                            },
                            exits: walked.exits,
                        });
                    }
                    let join = self.branch_join(
                        block,
                        &cases.iter().map(|(_, b)| *b).chain([default]).collect::<Vec<_>>(),
                        stops,
                    );
                    let mut arm_stops = stops.clone();
                    let mut join_is_new = false;
                    if let Some(join) = join
                        && !stops.contains_key(&join)
                    {
                        arm_stops.insert(join, StopKind::Join);
                        join_is_new = true;
                    }
                    // Every other case's body is off-limits (no
                    // fallthrough).
                    for (_, target) in &cases {
                        arm_stops.entry(*target).or_insert(StopKind::CaseEntry);
                    }
                    arm_stops.entry(default).or_insert(StopKind::CaseEntry);
                    let scrutinee = self.src_of(scrutinee, env)?;
                    let join_regs: Vec<VReg> = match (join_is_new, join) {
                        (true, Some(join)) => self.func.blocks[join].params.iter().map(|p| self.push_reg(self.reg_ty(*p))).collect(),
                        _ => Vec::new(),
                    };
                    let chain = self.switch_chain(&scrutinee, &cases, default, env, &arm_stops, join, &join_regs)?;
                    body.stmts.extend(chain.body.stmts);
                    if !join_is_new {
                        merge_exits(&mut exits, chain.exits);
                        return Ok(Walked { body, exits });
                    }
                    let join = join.expect("join known");
                    for (param, reg) in self.func.blocks[join].params.iter().zip(&join_regs) {
                        env.insert(*param, PVal::V(*reg));
                    }
                    merge_exits(&mut exits, chain.exits);
                    block = join;
                    entry_args = Vec::new();
                }
                Terminator::Invoke { .. } | Terminator::Resume { .. } => {
                    return Err("prism: exception control flow in a shader".to_string());
                }
            }
        }
    }

    /// Record a stop edge: joins record their args; loop edges emit
    /// their statement.
    fn finish_stop(
        &mut self,
        kind: StopKind,
        block: Block,
        args: SmallVec<[VReg; 4]>,
        mut body: Body,
        mut exits: HashMap<Block, Vec<SmallVec<[VReg; 4]>>>,
    ) -> Result<Walked, String> {
        match kind {
            StopKind::Join => {
                exits.entry(block).or_default().push(args);
            }
            StopKind::Continue => body.stmts.push(Stmt::Continue(args)),
            StopKind::Break => body.stmts.push(Stmt::Break(args)),
            StopKind::CaseEntry => {
                return Err("prism: switch fallthrough is not supported; end each case with break".to_string());
            }
        }
        Ok(Walked { body, exits })
    }

    /// The nested `if`-chain of a `switch`: one `setp`+`if` per case,
    /// the default as the innermost else. Returns the chain body, the
    /// join args the chain's fall-through contributes (the default's
    /// edge args, when the join is in scope), and the outer stops the
    /// chain's arms hit.
    fn switch_chain(
        &mut self,
        scrutinee: &Src,
        cases: &[(i128, Block)],
        default: Block,
        env: &mut HashMap<Value, PVal>,
        arm_stops: &HashMap<Block, StopKind>,
        join: Option<Block>,
        join_regs: &[VReg],
    ) -> Result<ChainOut, String> {
        let Some(((value, target), rest)) = cases.split_first() else {
            let mut own_stops = arm_stops.clone();
            own_stops.remove(&default);
            let walk = self.walk(default, &[], env, &own_stops)?;
            let mut exits = walk.exits;
            let join_entry = match join {
                Some(join) => {
                    let entry = exits.remove(&join).ok_or("prism: switch default does not reach the switch join")?;
                    Some(single_entry(entry)?)
                }
                None => None,
            };
            return Ok(ChainOut { body: walk.body, join_entry, exits });
        };
        let mut own_stops = arm_stops.clone();
        own_stops.remove(target);
        let then_walk = self.walk(*target, &[], env, &own_stops)?;
        let mut chain = self.switch_chain(scrutinee, rest, default, env, arm_stops, join, join_regs)?;
        let mut then_exits = then_walk.exits;
        let pred = self.push_reg(RegTy::Pred);
        let mut body = Body::new();
        body.stmts.push(Stmt::Setp(Setp {
            dst: pred,
            code: CmpCode::Eq,
            a: scrutinee.clone(),
            b: Src::ConstI32(*value as i32),
        }));
        let mut results = Vec::new();
        if let Some(join) = join {
            let then_entry = then_exits.remove(&join).ok_or("prism: switch case does not reach the switch join")?;
            let then_entry = single_entry(then_entry)?;
            let else_entry = chain.join_entry.clone().ok_or("prism: switch chain lost its join entry")?;
            if join_regs.len() != then_entry.len() || join_regs.len() != else_entry.len() {
                return Err("prism: switch join arity mismatch".to_string());
            }
            for (index, reg) in join_regs.iter().enumerate() {
                results.push((*reg, then_entry[index], else_entry[index]));
            }
        }
        body.stmts.push(Stmt::If { pred, then: then_walk.body, els: chain.body, results });
        merge_exits(&mut chain.exits, then_exits);
        Ok(ChainOut { body, join_entry: chain.join_entry, exits: chain.exits })
    }

    /// The join block of a multi-way branch: the first block both arm
    /// flows reach without crossing an enclosing stop, in dominance
    /// order. Arms that terminate (return/break) contribute nothing;
    /// when the surviving flows do not reconverge inside the scope,
    /// their blocks fold into the arms themselves.
    fn branch_join(&self, branch: Block, targets: &[Block], stops: &HashMap<Block, StopKind>) -> Option<Block> {
        let reachable = self.func.reachable_blocks();
        let mut common: Option<HashSet<Block>> = None;
        for target in targets {
            let reach = self.reach_restricted(*target, stops, &reachable);
            common = Some(match common {
                None => reach,
                Some(prev) => prev.intersection(&reach).copied().collect(),
            });
        }
        let common = common?;
        let rpo = glcc_mir::passes::reverse_postorder(self.func, &reachable);
        let mut rpo_index = vec![usize::MAX; self.func.blocks.len()];
        for (i, &b) in rpo.iter().enumerate() {
            rpo_index[b.index()] = i;
        }
        let mut candidates: Vec<Block> = common
            .into_iter()
            .filter(|b| !stops.contains_key(b) && *b != branch)
            .collect();
        candidates.sort_by_key(|b| rpo_index[b.index()]);
        // The join is the earliest candidate the other candidates all
        // flow into (the innermost convergence of nested joins).
        for (position, candidate) in candidates.iter().enumerate() {
            let reach = self.reach_restricted(*candidate, stops, &reachable);
            if candidates.iter().enumerate().all(|(other, b)| other == position || reach.contains(b)) {
                return Some(*candidate);
            }
        }
        None
    }

    /// Blocks reachable from `start` without crossing a stop: the
    /// region this arm's flow lives in.
    fn reach_restricted(&self, start: Block, stops: &HashMap<Block, StopKind>, reachable: &[bool]) -> HashSet<Block> {
        let mut seen: HashSet<Block> = HashSet::new();
        let mut work = vec![start];
        seen.insert(start);
        while let Some(block) = work.pop() {
            if block != start && stops.contains_key(&block) {
                continue;
            }
            if let Some(term) = &self.func.blocks[block].term {
                for succ in term.successors() {
                    if reachable[succ.index()] && seen.insert(succ) {
                        work.push(succ);
                    }
                }
            }
        }
        seen
    }

    /// The natural-loop block set of `block`, empty when `block` is
    /// not a loop header.
    fn natural_loop(&self, header: Block) -> Vec<Block> {
        let mut work: Vec<Block> = self.dom.preds[header.index()]
            .iter()
            .copied()
            .filter(|&pred| self.dom.dominates(header, pred, self.func.entry))
            .collect();
        if work.is_empty() {
            return Vec::new();
        }
        let mut set = vec![header];
        let mut seen: HashSet<Block> = HashSet::new();
        seen.insert(header);
        while let Some(block) = work.pop() {
            if !seen.insert(block) {
                continue;
            }
            set.push(block);
            for &pred in &self.dom.preds[block.index()] {
                work.push(pred);
            }
        }
        set
    }

    fn build_loop(
        &mut self,
        header: Block,
        entry_args: &[Value],
        natural: Vec<Block>,
        env: &mut HashMap<Value, PVal>,
        stops: &HashMap<Block, StopKind>,
    ) -> Result<Walked, String> {
        let loop_set: HashSet<Block> = natural.into_iter().collect();
        let mut exit_targets: Vec<Block> = Vec::new();
        for block in &loop_set {
            if let Some(term) = &self.func.blocks[*block].term {
                for succ in term.successors() {
                    if !loop_set.contains(&succ) && !exit_targets.contains(&succ) {
                        exit_targets.push(succ);
                    }
                }
            }
        }
        if exit_targets.len() > 1 {
            return Err("prism: loop exits to more than one block".to_string());
        }
        let exit = exit_targets.into_iter().next();

        let mut body = Body::new();
        let mut exits: HashMap<Block, Vec<SmallVec<[VReg; 4]>>> = HashMap::new();

        // Carried registers: the header's parameters, initialized from
        // the entry edge. The signature parameters of the original
        // entry (a `do`-while whose body starts there) pass through
        // unchanged — they are the interface, not loop state.
        let params = self.func.blocks[header].params.clone();
        let sig_params = if header == self.orig_entry { self.func.sig.params.len() } else { 0 };
        let mut carried = Vec::new();
        for (index, param) in params.iter().enumerate() {
            let arg = entry_args.get(index).ok_or("prism: loop entry arity mismatch")?;
            if index < sig_params {
                let value = self.pval_of(*arg, env)?;
                env.insert(*param, value);
                continue;
            }
            let ty = self.reg_ty(*param);
            let phi = self.push_reg(ty);
            let init = self.pval_of(*arg, env)?;
            let init = self.materialize(&init, ty, &mut body)?;
            carried.push((phi, init));
            env.insert(*param, PVal::V(phi));
        }

        let mut body_stops = stops.clone();
        body_stops.insert(header, StopKind::Continue);
        if let Some(exit) = exit {
            body_stops.insert(exit, StopKind::Break);
        }

        let body_walk = self.walk_inner(header, &[], env, &body_stops, true)?;
        for (block, entries) in body_walk.exits {
            if Some(block) == exit || block == header {
                // This loop's own break/continue edges: their
                // statements were emitted inside the body.
                continue;
            }
            exits.entry(block).or_default().extend(entries);
        }

        let results: Vec<VReg> = exit
            .map(|exit| self.func.blocks[exit].params.iter().map(|param| self.push_reg(self.reg_ty(*param))).collect())
            .unwrap_or_default();
        body.stmts.push(Stmt::Loop { carried, body: body_walk.body, results: results.clone() });

        if let Some(exit) = exit {
            if stops.contains_key(&exit) {
                // The exit block belongs to an enclosing scope (an
                // `if` join or an outer stop): the loop's exit
                // registers are this subtree's contribution to it.
                exits.entry(exit).or_default().push(results.iter().copied().collect());
                return Ok(Walked { body, exits });
            }
            for (param, reg) in self.func.blocks[exit].params.iter().zip(&results) {
                env.insert(*param, PVal::V(*reg));
            }
            let continuation = self.walk(exit, &[], env, stops)?;
            body.stmts.extend(continuation.body.stmts);
            for (block, entries) in continuation.exits {
                exits.entry(block).or_default().extend(entries);
            }
        }
        Ok(Walked { body, exits })
    }

    /// Edge arguments materialized into registers.
    fn edge_args(&mut self, args: &[Value], env: &mut HashMap<Value, PVal>, body: &mut Body) -> Result<SmallVec<[VReg; 4]>, String> {
        let mut out = SmallVec::new();
        for arg in args {
            let val = self.pval_of(*arg, env)?;
            let ty = self.reg_ty(*arg);
            out.push(self.materialize(&val, ty, body)?);
        }
        Ok(out)
    }

    /// Predicate for a branch condition: a direct comparison lowers to
    /// `setp` with the comparison's own code; any other boolean (an
    /// `and`/`or`/`select` chain, a widened comparison) tests as
    /// `!= 0`.
    fn setp_for_cond(&mut self, cond: Value, env: &mut HashMap<Value, PVal>, body: &mut Body) -> Result<VReg, String> {
        let (code, a, b) = match self.def_kind(cond) {
            Some(InstKind::Icmp { pred, lhs, rhs }) => (icmp_code(pred), lhs, Some(rhs)),
            Some(InstKind::Fcmp { pred, lhs, rhs }) => (fcmp_code(pred), lhs, Some(rhs)),
            _ => (CmpCode::Ne, cond, None),
        };
        let a = self.src_of(a, env)?;
        let b = match b {
            Some(b) => self.src_of(b, env)?,
            None => Src::ConstI32(0),
        };
        let dst = self.push_reg(RegTy::Pred);
        body.stmts.push(Stmt::Setp(Setp { dst, code, a, b }));
        Ok(dst)
    }

    fn def_kind(&self, v: Value) -> Option<InstKind> {
        match &self.func.values[v].def {
            Def::Inst(id) => Some(self.func.insts[*id].kind.clone()),
            _ => None,
        }
    }

    /// The value shape of a MIR value: the environment first, then
    /// constants and globals from the value's own definition (branch
    /// and loop edges carry constant arguments).
    fn pval_of(&self, v: Value, env: &HashMap<Value, PVal>) -> Result<PVal, String> {
        if let Some(pval) = env.get(&v) {
            return Ok(pval.clone());
        }
        match &self.func.values[v].def {
            Def::Const(Const::Int { value, .. }) => Ok(PVal::Src(Src::ConstI32(*value as i32))),
            Def::Const(Const::Float { kind: glcc_mir::ty::FloatKind::F32, bits }) => Ok(PVal::Src(Src::ConstF32(f32::from_bits(*bits as u32)))),
            Def::Const(Const::Zero) => Ok(PVal::Src(Src::ConstF32(0.0))),
            Def::Const(Const::Undef) => Ok(PVal::Undef),
            Def::Const(_) => Err("prism: unsupported constant on an edge".to_string()),
            Def::Inst(id) => match &self.func.insts[*id].kind {
                InstKind::GlobalAddr { global } => Ok(PVal::Global(*global)),
                _ => Err("prism: operand defined outside the reachable region".to_string()),
            },
            _ => Err("prism: operand defined outside the reachable region".to_string()),
        }
    }

    /// One source operand for a MIR value.
    fn src_of(&self, v: Value, env: &HashMap<Value, PVal>) -> Result<Src, String> {
        if let Some(pval) = env.get(&v) {
            return match pval {
                PVal::V(reg) => Ok(Src::VReg { v: *reg, swizzle: None, negate: false }),
                PVal::Src(src) => Ok(src.clone()),
                PVal::Undef => Ok(Src::Undef),
                PVal::Agg(_) => Err("prism: an aggregate flowed into an operand".to_string()),
                PVal::Global(global) => Err(format!("prism: global '{}' used as a value", self.global_name(*global))),
            };
        }
        // Constants read inline.
        match &self.func.values[v].def {
            Def::Const(Const::Int { value, .. }) => Ok(Src::ConstI32(*value as i32)),
            Def::Const(Const::Float { kind: glcc_mir::ty::FloatKind::F32, bits }) => {
                Ok(Src::ConstF32(f32::from_bits(*bits as u32)))
            }
            Def::Const(Const::Zero) => Ok(Src::ConstF32(0.0)),
            _ => Err("prism: operand defined outside the reachable region".to_string()),
        }
    }

    fn lower_inst(&mut self, inst: Inst, env: &mut HashMap<Value, PVal>, body: &mut Body) -> Result<(), String> {
        let data = self.func.insts[inst].clone();
        let Some(result) = data.result else {
            return Ok(());
        };
        let kind = data.kind;
        let pval = match &kind {
            InstKind::ExtractValue { agg, index } => {
                let base = self.pval_of(*agg, env)?;
                match base {
                    PVal::Agg(leaves) => leaves.get(*index as usize).cloned().ok_or("prism: aggregate leaf out of range")?,
                    PVal::Undef => PVal::Undef,
                    _ => return Err("prism: extract from a non-aggregate".to_string()),
                }
            }
            InstKind::InsertValue { agg, elem, index } => {
                let base = match env.get(agg).cloned() {
                    Some(base) => base,
                    None => {
                        let leaves = agg_leaf_count(&self.func.values[*agg].ty);
                        PVal::Agg(vec![PVal::Undef; leaves])
                    }
                };
                let PVal::Agg(mut leaves) = base else {
                    return Err("prism: insert into a non-aggregate".to_string());
                };
                let elem_val = self.pval_of(*elem, env)?;
                let slot = leaves.get_mut(*index as usize).ok_or("prism: aggregate leaf out of range")?;
                *slot = elem_val;
                PVal::Agg(leaves)
            }
            InstKind::ExtractLane { vec, lane } => {
                let src = self.src_of(*vec, env)?;
                PVal::Src(reswizzle(src, Some(Swizzle::from_slice(&[*lane])), false))
            }
            InstKind::Load { addr, .. } => self.lower_load(*addr, env)?,
            InstKind::GlobalAddr { global } => PVal::Global(*global),
            InstKind::SampleImage { image, coord } => {
                let sampler = self.sampler_of(*image, env)?;
                let coord = self.src_of(*coord, env)?;
                let dst = self.push_reg(mir_reg_ty(&self.func.values[result].ty));
                body.stmts.push(Stmt::Tex(Tex { dst, sampler, coord, lod: Lod::Implicit }));
                PVal::V(dst)
            }
            _ => {
                let dst = self.lower_alu(&kind, result, env, body)?;
                PVal::V(dst)
            }
        };
        env.insert(result, pval);
        Ok(())
    }

    /// Loads read uniforms as interface sources. A whole-object load
    /// of a multi-leaf uniform flattens into its leaf sources; geps
    /// with constant steps pick the leaf at that path; a single
    /// dynamic step indexes a vector-element array relatively.
    fn lower_load(&mut self, addr: Value, env: &mut HashMap<Value, PVal>) -> Result<PVal, String> {
        let (global, path, dynamic) = self.resolve_uniform_addr(addr)?;
        let name = self.global_name(global);
        let Some(object) = self.interface.uniform_by_name(&name) else {
            return Err(format!("prism: load from '{name}', which is not a uniform"));
        };
        if let Some(index) = dynamic {
            let base_leaf = self
                .interface
                .uniform_leaf(object, &path)
                .ok_or("prism: dynamic uniform index into a non-array")?;
            if base_leaf.ty.lanes() == 1 {
                return Err("prism: dynamic indexing of scalar uniform arrays is not supported".to_string());
            }
            let index_src = self.src_of(index, env)?;
            let index_reg = index_src.vreg().ok_or("prism: dynamic uniform index must be a register")?;
            return Ok(PVal::Src(Src::Uniform {
                object,
                reg: base_leaf.creg,
                swizzle: None,
                negate: false,
                rel: Some(index_reg),
            }));
        }
        let uniform = &self.interface.uniforms[object as usize];
        let src_for = |leaf: &crate::interface::UniformLeaf| {
            Src::Uniform {
                object,
                reg: leaf.creg,
                swizzle: (leaf.lane > 0).then(|| Swizzle::from_slice(&[leaf.lane])),
                negate: false,
                rel: None,
            }
        };
        if let Some(leaf) = self.interface.uniform_leaf(object, &path) {
            return Ok(PVal::Src(src_for(leaf)));
        }
        // A partial path (a whole sub-object): flatten its leaves.
        let leaves: Vec<PVal> = uniform
            .leaves
            .iter()
            .filter(|leaf| leaf.path.starts_with(&path))
            .map(|leaf| PVal::Src(src_for(leaf)))
            .collect();
        if leaves.is_empty() {
            return Err(format!("prism: uniform '{name}' member not found"));
        }
        Ok(PVal::Agg(leaves))
    }

    /// Resolve a load address to a uniform global, a constant member
    /// path, and an optional dynamic array index.
    fn resolve_uniform_addr(&self, addr: Value) -> Result<(Global, Vec<u32>, Option<Value>), String> {
        let mut path: Vec<u32> = Vec::new();
        let mut dynamic: Option<Value> = None;
        let mut current = addr;
        loop {
            match &self.func.values[current].def {
                Def::Inst(id) => match &self.func.insts[*id].kind {
                    InstKind::GlobalAddr { global } => return Ok((*global, path, dynamic)),
                    InstKind::GetElementPtr { base, indices } => {
                        for index in indices.iter().rev() {
                            match *index {
                                GepIndex::Field(i) => path.insert(0, i),
                                GepIndex::ConstIndex(i) => path.insert(0, i as u32),
                                GepIndex::Index(v) => {
                                    if dynamic.is_some() {
                                        return Err("prism: multi-dimensional dynamic uniform indexing is not supported".to_string());
                                    }
                                    dynamic = Some(v);
                                    path.push(0);
                                }
                            }
                        }
                        current = *base;
                    }
                    _ => return Err("prism: load from a non-global address".to_string()),
                },
                _ => return Err("prism: load from a non-global address".to_string()),
            }
        }
    }

    fn global_name(&self, global: Global) -> String {
        self.module.globals[global].name.clone()
    }

    fn sampler_of(&self, image: Value, env: &HashMap<Value, PVal>) -> Result<u32, String> {
        let global = match env.get(&image) {
            Some(PVal::Global(global)) => *global,
            _ => match &self.func.values[image].def {
                Def::Inst(id) => match &self.func.insts[*id].kind {
                    InstKind::GlobalAddr { global } => *global,
                    _ => return Err("prism: sample from a non-global image".to_string()),
                },
                _ => return Err("prism: sample from a non-global image".to_string()),
            },
        };
        let name = self.global_name(global);
        self.interface
            .sampler_by_name(&name)
            .ok_or_else(|| format!("prism: sample from '{name}', which is not a declared sampler"))
    }

    /// The operation forms. `result` carries the MIR result type.
    fn lower_alu(&mut self, kind: &InstKind, result: Value, env: &mut HashMap<Value, PVal>, body: &mut Body) -> Result<VReg, String> {
        let ty = mir_reg_ty(&self.func.values[result].ty);
        match kind {
            InstKind::Add(a, b) | InstKind::FAdd(a, b) => {
                let (a, b) = self.two(*a, *b, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Add, smallvec![a, b], body))
            }
            InstKind::Sub(a, b) | InstKind::FSub(a, b) => {
                let (a, b) = self.two(*a, *b, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Sub, smallvec![a, b], body))
            }
            InstKind::Mul(a, b) | InstKind::FMul(a, b) => {
                let (a, b) = self.two(*a, *b, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Mul, smallvec![a, b], body))
            }
            InstKind::FDiv(a, b) => {
                // Era ISAs have no divide: reciprocal, then multiply.
                let b_src = self.src_of(*b, env)?;
                let rcp = self.push_reg(ty.clone());
                body.stmts.push(Stmt::Op(Op { dst: rcp, kind: Alu::Rcp, srcs: smallvec![b_src] }));
                let a_src = self.src_of(*a, env)?;
                let rcp_src = Src::VReg { v: rcp, swizzle: None, negate: false };
                Ok(self.emit_op(ty.clone(), Alu::Mul, smallvec![a_src, rcp_src], body))
            }
            InstKind::Sdiv(a, b) => {
                let (a, b) = self.two(*a, *b, env)?;
                Ok(self.emit_op(ty.clone(), Alu::IDiv { unsigned: false }, smallvec![a, b], body))
            }
            InstKind::Udiv(a, b) => {
                let (a, b) = self.two(*a, *b, env)?;
                Ok(self.emit_op(ty.clone(), Alu::IDiv { unsigned: true }, smallvec![a, b], body))
            }
            InstKind::Srem(a, b) => {
                let (a, b) = self.two(*a, *b, env)?;
                Ok(self.emit_op(ty.clone(), Alu::IRem { unsigned: false }, smallvec![a, b], body))
            }
            InstKind::Urem(a, b) => {
                let (a, b) = self.two(*a, *b, env)?;
                Ok(self.emit_op(ty.clone(), Alu::IRem { unsigned: true }, smallvec![a, b], body))
            }
            InstKind::And(a, b) => {
                let (a, b) = self.two(*a, *b, env)?;
                Ok(self.emit_op(ty.clone(), Alu::And, smallvec![a, b], body))
            }
            InstKind::Or(a, b) => {
                let (a, b) = self.two(*a, *b, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Or, smallvec![a, b], body))
            }
            InstKind::Xor(a, b) => {
                let (a, b) = self.two(*a, *b, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Xor, smallvec![a, b], body))
            }
            InstKind::Shl(a, b) => {
                let (a, b) = self.two(*a, *b, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Shl, smallvec![a, b], body))
            }
            InstKind::LShr(a, b) => {
                let (a, b) = self.two(*a, *b, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Shr { arithmetic: false }, smallvec![a, b], body))
            }
            InstKind::AShr(a, b) => {
                let (a, b) = self.two(*a, *b, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Shr { arithmetic: true }, smallvec![a, b], body))
            }
            InstKind::Not(a) => {
                let a = self.src_of(*a, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Not, smallvec![a], body))
            }
            InstKind::Neg(a) | InstKind::FNeg(a) => {
                let a = self.src_of(*a, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Neg, smallvec![a], body))
            }
            InstKind::Min { a, b } => {
                let (a, b) = self.two(*a, *b, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Min, smallvec![a, b], body))
            }
            InstKind::Max { a, b } => {
                let (a, b) = self.two(*a, *b, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Max, smallvec![a, b], body))
            }
            InstKind::Sqrt(a) => {
                let a = self.src_of(*a, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Sqrt, smallvec![a], body))
            }
            InstKind::Floor(a) => {
                let a = self.src_of(*a, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Floor, smallvec![a], body))
            }
            InstKind::Ceil(a) => {
                let a = self.src_of(*a, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Ceil, smallvec![a], body))
            }
            InstKind::Icmp { pred, lhs, rhs } => {
                let (a, b) = self.two(*lhs, *rhs, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Cmp(icmp_code(*pred)), smallvec![a, b], body))
            }
            InstKind::Fcmp { pred, lhs, rhs } => {
                let (a, b) = self.two(*lhs, *rhs, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Cmp(fcmp_code(*pred)), smallvec![a, b], body))
            }
            InstKind::Zext(a) | InstKind::Sext(a) | InstKind::Trunc(a) => {
                let src_ty = mir_reg_ty(&self.func.values[*a].ty);
                let a = self.src_of(*a, env)?;
                let op_kind = if src_ty == ty {
                    Alu::Mov
                } else {
                    match kind {
                        InstKind::Zext(_) => Alu::Zext,
                        InstKind::Sext(_) => Alu::Sext,
                        _ => Alu::Trunc,
                    }
                };
                Ok(self.emit_op(ty.clone(), op_kind, smallvec![a], body))
            }
            InstKind::SItoF(a) => {
                let a = self.src_of(*a, env)?;
                Ok(self.emit_op(ty.clone(), Alu::IToF { unsigned: false }, smallvec![a], body))
            }
            InstKind::UItoF(a) => {
                let a = self.src_of(*a, env)?;
                Ok(self.emit_op(ty.clone(), Alu::IToF { unsigned: true }, smallvec![a], body))
            }
            InstKind::SFtoI(a) => {
                let a = self.src_of(*a, env)?;
                Ok(self.emit_op(ty.clone(), Alu::FToI { unsigned: false }, smallvec![a], body))
            }
            InstKind::UFtoI(a) => {
                let a = self.src_of(*a, env)?;
                Ok(self.emit_op(ty.clone(), Alu::FToI { unsigned: true }, smallvec![a], body))
            }
            InstKind::Bitcast(a) => {
                let a = self.src_of(*a, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Bitcast, smallvec![a], body))
            }
            InstKind::Select { cond, then_val, else_val } => {
                let cond = self.src_of(*cond, env)?;
                let a = self.src_of(*then_val, env)?;
                let b = self.src_of(*else_val, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Sel, smallvec![cond, a, b], body))
            }
            InstKind::Splat(v) => {
                // A scalar source feeds every lane of the vector dst.
                let v = self.src_of(*v, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Mov, smallvec![v], body))
            }
            InstKind::ShuffleVector { lhs, rhs, lanes } => {
                let a = self.src_of(*lhs, env)?;
                let b = self.src_of(*rhs, env)?;
                Ok(self.emit_op(ty.clone(), Alu::Shuffle { lanes: lanes.clone() }, smallvec![a, b], body))
            }
            other => Err(format!(
                "prism: operation {} has no shader form",
                glcc_mir::display::inst_str(self.func, other)
            )),
        }
    }

    fn emit_op(&mut self, ty: RegTy, kind: Alu, srcs: SmallVec<[Src; 3]>, body: &mut Body) -> VReg {
        let dst = self.push_reg(ty);
        body.stmts.push(Stmt::Op(Op { dst, kind, srcs }));
        dst
    }

    fn two(&mut self, a: Value, b: Value, env: &mut HashMap<Value, PVal>) -> Result<(Src, Src), String> {
        let a = self.src_of(a, env)?;
        let b = self.src_of(b, env)?;
        Ok((a, b))
    }
}

/// The real entry block, past any prep-inserted shim (`br` into the
/// real entry with the signature parameters as edge args).
fn real_entry(func: &Function) -> Block {
    let mut block = func.entry;
    loop {
        if !func.blocks[block].params.is_empty() {
            return block;
        }
        match &func.blocks[block].term {
            Some(Terminator::Br { dest, .. }) => block = *dest,
            _ => return block,
        }
    }
}

/// (breaks, continues) in a loop body, pre-order — a test helper for
/// structurizer shapes.
#[cfg(test)]
pub fn test_controls(body: &Body) -> (usize, usize) {
    let controls = super::isa::test_controls_of(body);
    controls
}

fn single_entry(entries: Vec<SmallVec<[VReg; 4]>>) -> Result<SmallVec<[VReg; 4]>, String> {
    if entries.len() != 1 {
        return Err("prism: a join has more than one incoming edge from one arm".to_string());
    }
    Ok(entries.into_iter().next().expect("one entry"))
}

fn merge_exits(exits: &mut HashMap<Block, Vec<SmallVec<[VReg; 4]>>>, walked: HashMap<Block, Vec<SmallVec<[VReg; 4]>>>) {
    for (block, entries) in walked {
        exits.entry(block).or_default().extend(entries);
    }
}

fn agg_leaf_count(ty: &glcc_mir::Ty) -> usize {
    match ty {
        glcc_mir::Ty::Struct { fields, .. } => fields.iter().map(agg_leaf_count).sum(),
        glcc_mir::Ty::Array { elem, len } => agg_leaf_count(elem) * *len as usize,
        _ => 1,
    }
}

fn mir_reg_ty(ty: &glcc_mir::Ty) -> RegTy {
    match ty {
        glcc_mir::Ty::Bool => RegTy::Scalar(Scalar::i32()),
        glcc_mir::Ty::Int { bits } => RegTy::Scalar(Scalar::Int { bits: *bits as u8 }),
        glcc_mir::Ty::Float(glcc_mir::ty::FloatKind::F32) => RegTy::Scalar(Scalar::F32),
        glcc_mir::Ty::Vector { elem, lanes } => RegTy::Vector {
            elem: match &**elem {
                glcc_mir::Ty::Float(glcc_mir::ty::FloatKind::F32) => Scalar::F32,
                glcc_mir::Ty::Int { bits } => Scalar::Int { bits: *bits as u8 },
                other => panic!("prism vector lane type {other:?}"),
            },
            lanes: *lanes as u8,
        },
        other => panic!("prism value type {other:?} is not a scalar or vector"),
    }
}

fn icmp_code(pred: IcmpPred) -> CmpCode {
    match pred {
        IcmpPred::Eq => CmpCode::Eq,
        IcmpPred::Ne => CmpCode::Ne,
        IcmpPred::Slt | IcmpPred::Ult => CmpCode::Lt,
        IcmpPred::Sle | IcmpPred::Ule => CmpCode::Le,
        IcmpPred::Sgt | IcmpPred::Ugt => CmpCode::Gt,
        IcmpPred::Sge | IcmpPred::Uge => CmpCode::Ge,
    }
}

fn fcmp_code(pred: FcmpPred) -> CmpCode {
    match pred {
        FcmpPred::Eq => CmpCode::Eq,
        FcmpPred::Ne => CmpCode::Ne,
        FcmpPred::Lt => CmpCode::Lt,
        FcmpPred::Le => CmpCode::Le,
        FcmpPred::Gt => CmpCode::Gt,
        FcmpPred::Ge => CmpCode::Ge,
        FcmpPred::Uno | FcmpPred::Ord => CmpCode::Eq,
    }
}

/// Rebuild a source with a new swizzle and negate flag.
fn reswizzle(src: Src, swizzle: Option<Swizzle>, negate: bool) -> Src {
    match src {
        Src::VReg { v, negate: old, .. } => Src::VReg { v, swizzle, negate: old ^ negate },
        Src::Input { input, negate: old, .. } => Src::Input { input, swizzle, negate: old ^ negate },
        Src::Uniform { object, reg, negate: old, rel, .. } => Src::Uniform { object, reg, swizzle, negate: old ^ negate, rel },
        Src::Undef => Src::Undef,
        Src::ConstF32(value) => Src::ConstF32(if negate { -value } else { value }),
        Src::ConstI32(value) => Src::ConstI32(if negate { -value } else { value }),
    }
}

trait SrcExt {
    fn vreg(&self) -> Option<VReg>;
}

impl SrcExt for Src {
    fn vreg(&self) -> Option<VReg> {
        match self {
            Src::VReg { v, .. } => Some(*v),
            _ => None,
        }
    }
}

/// Fusion peepholes over the built program:
///
/// 1. sink single-use `mov`s into their consumer's operand (source
///    swizzles compose; shuffles and phi positions keep their
///    registers),
/// 2. dot-shaped add trees of lane products become `dp3`/`dp4` — the
///    exact shape `mul4x4`'s lowering spreads across the function,
/// 3. a multiply feeding a float add/sub becomes `mad` (the subtract
///    negates the addend).
fn fuse(program: &mut Program) {
    sink_movs(program);
    let uses = count_uses(program);
    fuse_trees(program, &uses, FuseKind::Dot);
    let uses = count_uses(program);
    fuse_trees(program, &uses, FuseKind::Mad);
}

/// Rebuild a source with its negation flipped.
fn negate_src(src: Src) -> Src {
    match src {
        Src::VReg { v, swizzle, negate } => Src::VReg { v, swizzle, negate: !negate },
        Src::Input { input, swizzle, negate } => Src::Input { input, swizzle, negate: !negate },
        Src::Uniform { object, reg, swizzle, negate, rel } => Src::Uniform { object, reg, swizzle, negate: !negate, rel },
        Src::ConstF32(value) => Src::ConstF32(-value),
        Src::ConstI32(value) => Src::ConstI32(-value),
        Src::Undef => Src::Undef,
    }
}

/// One operand identity for dot-term matching: the base a lane reads
/// from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Base {
    V(VReg),
    Uniform(u32, u8),
    Input(u32),
}

fn src_base(src: &Src) -> Option<(Base, u8)> {
    match src {
        Src::VReg { v, swizzle: Some(lanes), negate: false } if lanes.len() == 1 => Some((Base::V(*v), lanes[0])),
        Src::Uniform { object, reg, swizzle: Some(lanes), negate: false, rel: None } if lanes.len() == 1 => {
            Some((Base::Uniform(*object, *reg), lanes[0]))
        }
        Src::Input { input, swizzle: Some(lanes), negate: false } if lanes.len() == 1 => Some((Base::Input(*input), lanes[0])),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FuseKind {
    Dot,
    Mad,
}

fn fuse_trees(program: &mut Program, uses: &HashMap<VReg, u32>, kind: FuseKind) {
    fuse_body(&mut program.body, &program.regs, uses, kind);
}

fn fuse_body(body: &mut Body, regs: &[RegTy], uses: &HashMap<VReg, u32>, kind: FuseKind) {
    // Ops by dst within this straight-line region; dot/mad trees must
    // live in one.
    let mut by_dst: HashMap<VReg, usize> = HashMap::new();
    for (index, stmt) in body.stmts.iter().enumerate() {
        if let Stmt::Op(op) = stmt {
            by_dst.insert(op.dst, index);
        }
    }
    let mut consumed: Vec<VReg> = Vec::new();
    for index in (0..body.stmts.len()).rev() {
        let (is_sub, srcs) = match &body.stmts[index] {
            Stmt::Op(Op { kind: Alu::Add, srcs, .. }) => (false, srcs.clone()),
            Stmt::Op(Op { kind: Alu::Sub, srcs, .. }) => (true, srcs.clone()),
            _ => continue,
        };
        // Float-only shapes: both operands' register types are float.
        if !srcs.iter().all(|src| match src {
            Src::VReg { v, .. } => regs.get(v.0 as usize).is_some_and(|ty| ty.is_f32()),
            Src::ConstF32(_) | Src::Input { .. } | Src::Uniform { .. } => true,
            _ => false,
        }) {
            continue;
        }
        match kind {
            FuseKind::Dot => {
                if let Some((width, a, b, _terms)) = dot_tree(&body.stmts, &by_dst, uses, index) {
                    // Collect the tree's nodes before the rewrite
                    // replaces the root's sources.
                    let dead = tree_dsts(&body.stmts, &by_dst, uses, index);
                    let alu = if width == 4 { Alu::Dp4 } else { Alu::Dp3 };
                    let a = whole_src(a, width);
                    let b = whole_src(b, width);
                    if let Stmt::Op(op) = &mut body.stmts[index] {
                        op.kind = alu;
                        op.srcs = smallvec![a, b];
                    }
                    consumed.extend(dead);
                }
            }
            FuseKind::Mad => {
                let is_mul = |src: &Src| match src {
                    Src::VReg { v, .. } => by_dst
                        .get(v)
                        .is_some_and(|&i| matches!(&body.stmts[i], Stmt::Op(Op { kind: Alu::Mul, .. })))
                        && uses.get(v).copied().unwrap_or(0) == 1,
                    _ => false,
                };
                let mul_sides: Vec<usize> = srcs.iter().enumerate().filter(|(_, src)| is_mul(src)).map(|(i, _)| i).collect();
                // Both sides multiply: a dot-tree fragment — leave it
                // (the dot pass already had its chance; fusing half a
                // dot would hide the shape).
                let mul_side = if mul_sides.len() == 1 { mul_sides[0] } else { usize::MAX };
                let _ = &is_mul;
                if mul_side != usize::MAX {
                    let side = mul_side;
                    let mul_index = match &srcs[side] {
                        Src::VReg { v, .. } => by_dst[v],
                        _ => unreachable!("matched above"),
                    };
                    let (a, b) = match &body.stmts[mul_index] {
                        Stmt::Op(Op { srcs: mul_srcs, .. }) => (mul_srcs[0].clone(), mul_srcs[1].clone()),
                        _ => unreachable!("matched above"),
                    };
                    let addend = srcs[1 - side].clone();
                    let addend = if is_sub { negate_src(addend) } else { addend };
                    if let Stmt::Op(op) = &mut body.stmts[index] {
                        op.kind = Alu::Mad;
                        op.srcs = smallvec![a, b, addend];
                    }
                    if let Stmt::Op(Op { dst, .. }) = &body.stmts[mul_index] {
                        consumed.push(*dst);
                    }
                }
            }
        }
    }
    if !consumed.is_empty() {
        let consumed: HashSet<VReg> = consumed.into_iter().collect();
        for stmt in body.stmts.iter_mut() {
            if let Stmt::Op(op) = stmt
                && consumed.contains(&op.dst)
            {
                // Marked for removal below; neutralize first so the
                // filter sees a stable shape.
                op.kind = Alu::Mov;
                op.srcs = smallvec![Src::Undef];
            }
        }
        // Remove consumed ops only when nothing else reads them (the
        // use counts guaranteed single use).
        let live: Vec<Stmt> = body
            .stmts
            .iter()
            .filter(|stmt| match stmt {
                Stmt::Op(Op { dst, kind: Alu::Mov, srcs }) if srcs.len() == 1 && matches!(srcs[0], Src::Undef) => {
                    !consumed.contains(dst)
                }
                _ => true,
            })
            .cloned()
            .collect();
        body.stmts = live;
    }
    for stmt in &mut body.stmts {
        match stmt {
            Stmt::If { then, els, .. } => {
                fuse_body(then, regs, uses, kind);
                fuse_body(els, regs, uses, kind);
            }
            Stmt::Loop { body: inner, .. } => fuse_body(inner, regs, uses, kind),
            _ => {}
        }
    }
}

/// The whole-vector form of a dot operand: the base with an identity
/// swizzle over `width` lanes.
fn whole_src(base: Base, width: u8) -> Src {
    let swizzle = Some((0..width).collect::<Swizzle>());
    match base {
        Base::V(v) => Src::VReg { v, swizzle, negate: false },
        Base::Uniform(object, reg) => Src::Uniform { object, reg, swizzle, negate: false, rel: None },
        Base::Input(input) => Src::Input { input, swizzle, negate: false },
    }
}

/// Flatten the add tree at `index` into its multiply terms; a dot
/// needs 3 or 4 single-use terms pairing the same two bases on
/// distinct lanes.
#[allow(clippy::type_complexity)]
fn dot_tree(
    stmts: &[Stmt],
    by_dst: &HashMap<VReg, usize>,
    uses: &HashMap<VReg, u32>,
    index: usize,
) -> Option<(u8, Base, Base, Vec<(u8, u8)>)> {
    let mut terms: Vec<((Base, u8), (Base, u8))> = Vec::new();
    if !collect_terms(stmts, by_dst, uses, index, &mut terms) {
        return None;
    }
    if terms.len() != 3 && terms.len() != 4 {
        return None;
    }
    // Canonical operand order per term, then require one common left
    // base and one common right base.
    let mut pairs = Vec::new();
    for ((a, la), (b, lb)) in terms {
        if a <= b {
            pairs.push(((a, la), (b, lb)));
        } else {
            pairs.push(((b, lb), (a, la)));
        }
    }
    let ((left, _), (right, _)) = pairs[0];
    let mut left_lanes = Vec::new();
    let mut right_lanes = Vec::new();
    for ((a, la), (b, lb)) in &pairs {
        if *a != left || *b != right {
            return None;
        }
        left_lanes.push(*la);
        right_lanes.push(*lb);
    }
    let width = pairs.len() as u8;
    for lanes in [&left_lanes, &right_lanes] {
        let mut sorted = lanes.clone();
        sorted.sort();
        if sorted.iter().enumerate().any(|(i, lane)| *lane != i as u8) || *lanes.iter().max().unwrap_or(&0) >= width {
            return None;
        }
    }
    Some((width, left, right, pairs.iter().map(|((_, la), (_, lb))| (*la, *lb)).collect()))
}

/// Recursively flatten an add tree into its multiply terms. Every
/// intermediate node must be single-use; leaves are single-use float
/// multiplies of one-lane sources.
fn collect_terms(
    stmts: &[Stmt],
    by_dst: &HashMap<VReg, usize>,
    uses: &HashMap<VReg, u32>,
    index: usize,
    terms: &mut Vec<((Base, u8), (Base, u8))>,
) -> bool {
    let (kind, srcs) = match &stmts[index] {
        Stmt::Op(Op { kind, srcs, .. }) => (kind, srcs),
        _ => return false,
    };
    match kind {
        Alu::Add => {
            for src in srcs {
                match src {
                    Src::VReg { v, .. } => {
                        let Some(&child) = by_dst.get(v) else { return false };
                        if uses.get(v).copied().unwrap_or(0) != 1 {
                            return false;
                        }
                        if !collect_terms(stmts, by_dst, uses, child, terms) {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }
            true
        }
        Alu::Mul => {
            if srcs.len() != 2 {
                return false;
            }
            let (Some(a), Some(b)) = (src_base(&srcs[0]), src_base(&srcs[1])) else {
                return false;
            };
            // Each mul here was reached through a single-use edge, so
            // its dst has exactly that one use.
            terms.push((a, b));
            true
        }
        _ => false,
    }
}

/// The dsts a fused tree consumes: every intermediate add except the
/// root, plus every multiply.
fn tree_dsts(stmts: &[Stmt], by_dst: &HashMap<VReg, usize>, uses: &HashMap<VReg, u32>, index: usize) -> Vec<VReg> {
    let mut dsts = Vec::new();
    collect_dsts(stmts, by_dst, uses, index, &mut dsts, true);
    dsts
}

fn collect_dsts(stmts: &[Stmt], by_dst: &HashMap<VReg, usize>, uses: &HashMap<VReg, u32>, index: usize, dsts: &mut Vec<VReg>, root: bool) {
    let (kind, srcs, dst) = match &stmts[index] {
        Stmt::Op(Op { kind, srcs, dst, .. }) => (kind, srcs, *dst),
        _ => return,
    };
    match kind {
        Alu::Add => {
            if !root {
                dsts.push(dst);
            }
            for src in srcs {
                if let Src::VReg { v, .. } = src
                    && let Some(&child) = by_dst.get(v)
                    && uses.get(v).copied().unwrap_or(0) == 1
                {
                    collect_dsts(stmts, by_dst, uses, child, dsts, false);
                }
            }
        }
        Alu::Mul => dsts.push(dst),
        _ => {}
    }
}

/// Global VReg use counts (operand positions only; definitions do not
/// count).
fn count_uses(program: &Program) -> HashMap<VReg, u32> {
    let mut uses: HashMap<VReg, u32> = HashMap::new();
    fn count_src(src: &Src, uses: &mut HashMap<VReg, u32>) {
        match src {
            Src::VReg { v, .. } => *uses.entry(*v).or_insert(0) += 1,
            Src::Uniform { rel: Some(rel), .. } => {
                *uses.entry(*rel).or_insert(0) += 1;
            }
            _ => {}
        }
    }
    fn count_body(body: &Body, uses: &mut HashMap<VReg, u32>) {
        for stmt in &body.stmts {
            match stmt {
                Stmt::Op(op) => op.srcs.iter().for_each(|src| count_src(src, uses)),
                Stmt::Tex(tex) => count_src(&tex.coord, uses),
                Stmt::Setp(setp) => {
                    count_src(&setp.a, uses);
                    count_src(&setp.b, uses);
                }
                Stmt::If { pred, then, els, results } => {
                    *uses.entry(*pred).or_insert(0) += 1;
                    for (_, then_v, else_v) in results {
                        *uses.entry(*then_v).or_insert(0) += 1;
                        *uses.entry(*else_v).or_insert(0) += 1;
                    }
                    count_body(then, uses);
                    count_body(els, uses);
                }
                Stmt::Loop { carried, body: inner, results: _ } => {
                    for (_, init) in carried {
                        *uses.entry(*init).or_insert(0) += 1;
                    }
                    count_body(inner, uses);
                }
                Stmt::Break(args) | Stmt::Continue(args) => {
                    for arg in args {
                        *uses.entry(*arg).or_insert(0) += 1;
                    }
                }
                Stmt::Exit(args) => {
                    for arg in args {
                        *uses.entry(*arg).or_insert(0) += 1;
                    }
                }
            }
        }
    }
    count_body(&program.body, &mut uses);
    uses
}

/// Sink single-use `mov`s into operand positions. Shuffles keep their
/// sources (their lane indices assume the source's width), and phi
/// incoming values must stay registers.
fn sink_movs(program: &mut Program) {
    let uses = count_uses(program);
    sink_body(&mut program.body, &uses);
    // Drop movs whose single use was rewritten away (or that were
    // dead to begin with).
    let after = count_uses(program);
    let sunk = mov_dsts(&program.body)
        .into_iter()
        .filter(|dst| after.get(dst).copied().unwrap_or(0) == 0)
        .collect::<Vec<_>>();
    if !sunk.is_empty() {
        remove_movs(&mut program.body, &sunk);
    }
}

/// Every single-source `mov` destination in the program.
fn mov_dsts(body: &Body) -> Vec<VReg> {
    let mut dsts = Vec::new();
    for stmt in &body.stmts {
        match stmt {
            Stmt::Op(Op { dst, kind: Alu::Mov, srcs }) if srcs.len() == 1 => dsts.push(*dst),
            Stmt::If { then, els, .. } => {
                dsts.extend(mov_dsts(then));
                dsts.extend(mov_dsts(els));
            }
            Stmt::Loop { body: inner, .. } => dsts.extend(mov_dsts(inner)),
            _ => {}
        }
    }
    dsts
}

fn sink_body(body: &mut Body, uses: &HashMap<VReg, u32>) {
    // Single-use movs defined in this region.
    let mut movs: HashMap<VReg, Src> = HashMap::new();
    for stmt in &body.stmts {
        if let Stmt::Op(Op { dst, kind: Alu::Mov, srcs }) = stmt
            && srcs.len() == 1
            && uses.get(dst).copied().unwrap_or(0) == 1
        {
            movs.insert(*dst, srcs[0].clone());
        }
    }
    if !movs.is_empty() {
        rewrite_srcs(body, &movs);
    }
    for stmt in &mut body.stmts {
        match stmt {
            Stmt::If { then, els, .. } => {
                sink_body(then, uses);
                sink_body(els, uses);
            }
            Stmt::Loop { body: inner, .. } => sink_body(inner, uses),
            _ => {}
        }
    }
}

/// Rewrite operand positions of one region (and its nested regions'
/// operand positions are handled by their own sink pass — this pass
/// rewrites only the statements directly in `body`).
fn rewrite_srcs(body: &mut Body, movs: &HashMap<VReg, Src>) {
    let resolve = |src: &Src| -> Src {
        let Src::VReg { v, swizzle, negate } = src else { return src.clone() };
        let Some(def) = movs.get(v) else { return src.clone() };
        compose_src(def, swizzle.as_ref(), *negate)
    };
    for stmt in &mut body.stmts {
        match stmt {
            // Shuffles keep their sources: their lane indices assume
            // the sources' full widths.
            Stmt::Op(Op { kind: Alu::Shuffle { .. }, .. }) => {}
            Stmt::Op(op) => op.srcs = op.srcs.iter().map(&resolve).collect(),
            Stmt::Tex(tex) => tex.coord = resolve(&tex.coord),
            Stmt::Setp(setp) => {
                setp.a = resolve(&setp.a);
                setp.b = resolve(&setp.b);
            }
            _ => {}
        }
    }
}

/// Compose a sunk mov's source with the use site's swizzle and negate.
fn compose_src(def: &Src, swizzle: Option<&Swizzle>, negate: bool) -> Src {
    let composed_swizzle = |def_swizzle: &Option<Swizzle>| -> Option<Swizzle> {
        match (def_swizzle, swizzle) {
            (None, None) => None,
            (None, Some(use_swizzle)) => Some(use_swizzle.clone()),
            (Some(def_swizzle), None) => Some(def_swizzle.clone()),
            (Some(def_swizzle), Some(use_swizzle)) => Some(use_swizzle.iter().map(|&lane| def_swizzle[lane as usize]).collect()),
        }
    };
    match def {
        Src::VReg { v, swizzle: def_swizzle, negate: def_negate } => Src::VReg {
            v: *v,
            swizzle: composed_swizzle(def_swizzle),
            negate: *def_negate ^ negate,
        },
        Src::Input { input, swizzle: def_swizzle, negate: def_negate } => Src::Input {
            input: *input,
            swizzle: composed_swizzle(def_swizzle),
            negate: *def_negate ^ negate,
        },
        Src::Uniform { object, reg, swizzle: def_swizzle, negate: def_negate, rel } => Src::Uniform {
            object: *object,
            reg: *reg,
            swizzle: composed_swizzle(def_swizzle),
            negate: *def_negate ^ negate,
            rel: *rel,
        },
        Src::ConstF32(value) => Src::ConstF32(if negate { -value } else { *value }),
        Src::ConstI32(value) => Src::ConstI32(if negate { -value } else { *value }),
        Src::Undef => Src::Undef,
    }
}

fn remove_movs(body: &mut Body, sunk: &[VReg]) {
    let sunk: HashSet<VReg> = sunk.iter().copied().collect();
    filter_body(body, &sunk);
}

fn filter_body(body: &mut Body, sunk: &HashSet<VReg>) {
    body.stmts.retain(|stmt| match stmt {
        Stmt::Op(Op { dst, kind: Alu::Mov, srcs }) => !(sunk.contains(dst) && srcs.len() == 1),
        _ => true,
    });
    for stmt in &mut body.stmts {
        match stmt {
            Stmt::If { then, els, .. } => {
                filter_body(then, sunk);
                filter_body(els, sunk);
            }
            Stmt::Loop { body: inner, .. } => filter_body(inner, sunk),
            _ => {}
        }
    }
}


