//! Sparse conditional constant propagation.
//!
//! [`ConstFold`](super::ConstFold) folds an instruction only when its
//! operands are already constants in the arena. This pass reproduces
//! that locally *and* goes further over the CFG: values become the
//! lattice `unknown -> constant -> overdefined`, blocks become
//! executable one edge at a time, and phi-equivalent block params join
//! the branch arguments that reach them.
//!
//! That turns into three kinds of rewrite once the worklists drain:
//!
//! - a result that settled at `constant` gets its uses replaced by a
//!   materialized constant — even when the folding depended on paths
//!   the program provably cannot take;
//! - a result still `unknown` at the fixpoint is only referenced from
//!   blocks no edge proved executable; its uses become `Undef` and the
//!   later dead-code sweeps remove it;
//! - a conditional branch or switch on a resolved condition folds to
//!   its one reachable destination, and blocks no edge reached get
//!   `Unreachable` terminators ([`SimplifyCfg`](super::SimplifyCfg)
//!   deletes them and their instructions).

use std::collections::{HashMap, VecDeque};

use crate::ir::{Block, Const, Def, FcmpPred, Function, Inst, InstKind, Terminator, Value, ValueData, ValueOrigin};
use crate::pass::FunctionPass;
use crate::ty::{FloatKind, Ty};

use super::{eval_icmp, fold_ashr_bits, fold_div_bits, fold_fcmp_bits, fold_int2_bits, fold_neg_bits, fold_not_bits, fold_shift_bits, int_ty_bits, mask, wrap};

/// One value's optimization status.
#[derive(Debug, Clone, PartialEq)]
enum Lat {
    /// Nothing known yet; a value left here at the fixpoint is unreachable.
    Unknown,
    /// Every reached execution sees this value.
    Const(Const),
    /// The value varies (or its def reads memory or may trap): leave it.
    Over,
}

/// Sparse conditional constant propagation over one function.
#[derive(Default)]
pub struct Sccp;

impl FunctionPass for Sccp {
    fn name(&self) -> &'static str {
        "sccp"
    }

    fn run(&mut self, func: &mut Function) -> bool {
        if func.blocks.is_empty() {
            return false;
        }
        let mut state = State::new(func);
        state.solve(func);
        rewrite(func, &state.lattice, &state.executable)
    }
}

/// Worklists and lattice of one solve. `func` is threaded through the
/// walk rather than held in the state, so borrows stay disjoint.
struct State {
    lattice: Vec<Lat>,
    /// Blocks with at least one proven-executable incoming edge.
    executable: Vec<bool>,
    edge_todo: VecDeque<(Block, Block)>,
    bb_todo: VecDeque<Block>,
    /// Blocks already in `bb_todo`, so a hot param is not enqueued once
    /// per use.
    queued_bbs: Vec<bool>,
    /// Resident instruction users of each value; a value update must
    /// re-evaluate them.
    users: Vec<Vec<Inst>>,
    /// Blocks whose terminator mentions each value (branch conditions,
    /// switch scrutinees, and edge arguments).
    term_users: Vec<Vec<Block>>,
}

impl State {
    fn new(func: &Function) -> State {
        let nvalues = func.values.len();
        let mut lattice = vec![Lat::Over; nvalues];
        let mut users: Vec<Vec<Inst>> = vec![Vec::new(); nvalues];
        let mut term_users: Vec<Vec<Block>> = vec![Vec::new(); nvalues];
        // Defs first: constants are known, entry params are function
        // arguments (overdefined immediately), other params and
        // instruction results stay unknown until proven otherwise.
        for (i, data) in func.values.iter().enumerate() {
            match &data.def {
                Def::Const(c) => lattice[i] = Lat::Const(c.clone()),
                Def::Param(block, _) if *block == func.entry => lattice[i] = Lat::Over,
                _ => lattice[i] = Lat::Unknown,
            }
        }
        for block in func.blocks.iter() {
            for &inst in &block.insts {
                func.insts[inst].kind.for_each_operand(&mut |op| users[op.index()].push(inst));
            }
        }
        for (idx, block) in func.blocks.iter().enumerate() {
            let here = Block::from_usize(idx);
            if let Some(term) = &block.term {
                term.for_each_operand(&mut |op| {
                    if !term_users[op.index()].contains(&here) {
                        term_users[op.index()].push(here);
                    }
                });
            }
        }
        State {
            lattice,
            executable: vec![false; func.blocks.len()],
            edge_todo: VecDeque::new(),
            bb_todo: VecDeque::new(),
            queued_bbs: vec![false; func.blocks.len()],
            users,
            term_users,
        }
    }

    fn solve(&mut self, func: &Function) {
        self.executable[func.entry.index()] = true;
        self.enqueue_bb(func.entry);
        while !(self.edge_todo.is_empty() && self.bb_todo.is_empty()) {
            while let Some((from, to)) = self.edge_todo.pop_front() {
                self.visit_edge(func, from, to);
            }
            while let Some(block) = self.bb_todo.pop_front() {
                self.queued_bbs[block.index()] = false;
                if self.executable[block.index()] {
                    self.visit_block(func, block);
                }
            }
        }
        // Debug builds prove the fixpoint the rewrite assumes; release
        // skips the check entirely.
        if cfg!(debug_assertions) {
            let errors = self.fixpoint_errors(func);
            debug_assert!(errors.is_empty(), "sccp fixpoint violated: {errors:?}");
        }
    }

    fn mark_edge(&mut self, from: Block, to: Block) {
        // No dedup: an edge must re-join its arguments into the target's
        // params every time the source is re-visited, because an
        // argument that widened after the first join would otherwise be
        // stuck at its stale lattice. `join_param` only propagates real
        // changes, so repeated joins still reach a fixpoint.
        self.edge_todo.push_back((from, to));
    }

    fn enqueue_bb(&mut self, block: Block) {
        if !self.queued_bbs[block.index()] {
            self.queued_bbs[block.index()] = true;
            self.bb_todo.push_back(block);
        }
    }

    /// The edge from `from` to `to` is proven: `to` is executable from
    /// here on, and each of its params joins this edge's argument (all
    /// edges together produce the join a phi needs).
    fn visit_edge(&mut self, func: &Function, from: Block, to: Block) {
        if !self.executable[to.index()] {
            self.executable[to.index()] = true;
            self.enqueue_bb(to);
        }
        for (i, param) in func.blocks[to].params.iter().enumerate() {
            let Some(arg) = edge_arg(func, from, to, i) else {
                continue;
            };
            self.join_param(func, *param, arg);
        }
    }

    /// Check the fixpoint invariant: every block-param's lattice must
    /// equal the join of the arguments on its executable incoming edges.
    /// A stale join (a param stuck at a constant its incoming arguments
    /// have since widened past) violates this and is the class of
    /// miscompile that `mark_edge`'s no-dedup rule fixes; this reports
    /// it instead of leaving a silently wrong rewrite.
    fn fixpoint_errors(&self, func: &Function) -> Vec<String> {
        let mut errors = Vec::new();
        for block_idx in 0..func.blocks.len() {
            let block = Block::from_usize(block_idx);
            if !self.executable[block_idx] || block == func.entry {
                continue;
            }
            for (i, param) in func.blocks[block].params.clone().iter().enumerate() {
                let mut expected = Lat::Unknown;
                for from_idx in 0..func.blocks.len() {
                    let from = Block::from_usize(from_idx);
                    if !self.executable[from_idx] {
                        continue;
                    }
                    let Some(arg) = edge_arg(func, from, block, i) else {
                        continue;
                    };
                    expected = lub(&expected, &self.lattice[arg.index()]);
                }
                let stored = &self.lattice[param.index()];
                if stored != &expected {
                    errors.push(format!(
                        "{}: b{block_idx} param %{} is {stored:?}, expected {expected:?}",
                        func.name,
                        param.index()
                    ));
                } else if matches!(stored, Lat::Unknown) {
                    errors.push(format!(
                        "{}: b{block_idx} param %{} never joined (stayed Unknown)",
                        func.name,
                        param.index()
                    ));
                }
            }
        }
        errors
    }

    /// One more argument joins a phi parameter; the param belongs to
    /// the join of all incoming edges, so it can only widen.
    fn join_param(&mut self, func: &Function, param: Value, arg: Value) {
        let incoming = self.lattice[arg.index()].clone();
        let merged = match (&self.lattice[param.index()], incoming) {
            (Lat::Unknown, other) => other,
            (Lat::Over, _) => Lat::Over,
            (Lat::Const(a), Lat::Const(b)) if *a == b => Lat::Const(a.clone()),
            (Lat::Const(_), _) => Lat::Over,
        };
        self.assign(func, param, merged);
    }

    fn assign(&mut self, func: &Function, v: Value, lat: Lat) {
        if self.lattice[v.index()] == lat {
            return;
        }
        self.lattice[v.index()] = lat;
        self.propagate(func, v);
    }

    /// A widened value feeds its instruction users immediately and its
    /// terminator users through the block queue.
    fn propagate(&mut self, func: &Function, v: Value) {
        // The user lists are fixed while solving, so walk them by index
        // instead of cloning a vector per assignment.
        let mut i = 0;
        while i < self.users[v.index()].len() {
            let user = self.users[v.index()][i];
            i += 1;
            self.eval_inst(func, user);
        }
        let mut i = 0;
        while i < self.term_users[v.index()].len() {
            let block = self.term_users[v.index()][i];
            i += 1;
            self.enqueue_bb(block);
        }
    }

    /// A block with at least one proven edge: run its instructions
    /// top-down, then follow its terminator's now-decided edges.
    fn visit_block(&mut self, func: &Function, block: Block) {
        for i in 0..func.blocks[block].insts.len() {
            self.eval_inst(func, func.blocks[block].insts[i]);
        }
        let Some(term) = &func.blocks[block].term else {
            return;
        };
        match term {
            Terminator::Br { dest, .. } => self.mark_edge(block, *dest),
            Terminator::CondBr { cond, then_block, else_block, .. } => match branch_targets(&self.lattice[cond.index()], *then_block, *else_block) {
                EdgeSet::None => {}
                EdgeSet::Only(dest) => self.mark_edge(block, dest),
                EdgeSet::Both => {
                    self.mark_edge(block, *then_block);
                    self.mark_edge(block, *else_block);
                }
            },
            Terminator::Switch { scrutinee, cases, default } => match &self.lattice[scrutinee.index()] {
                Lat::Unknown => {}
                Lat::Const(_) => {
                    self.mark_edge(block, switch_take(&self.lattice[scrutinee.index()], cases, *default));
                }
                _ => {
                    self.mark_edge(block, *default);
                    for (_, dest) in cases {
                        self.mark_edge(block, *dest);
                    }
                }
            },
            Terminator::Return { .. } | Terminator::Unreachable => {}
        }
    }

    /// Recompute one instruction's result lattice from its operands.
    /// Monotone, because operand lattices only ever widen.
    fn eval_inst(&mut self, func: &Function, inst: Inst) {
        let Some(result) = func.insts[inst].result else {
            return;
        };
        let next = eval(&func.insts[inst].kind, &func.values[result].ty, &self.lattice);
        self.assign(func, result, next);
    }
}

/// The edge's `i`th argument, or `None` for edges without a branch arm
/// there (a `Switch` target, whose arms carry no arguments) and for
/// argument lists shorter than the params.
fn edge_arg(func: &Function, from: Block, to: Block, i: usize) -> Option<Value> {
    let term = func.blocks[from].term.as_ref()?;
    let args = match term {
        Terminator::Br { dest, args } if *dest == to => args,
        Terminator::CondBr {
            then_block,
            then_args,
            else_block,
            else_args,
            ..
        } => {
            if *then_block == to {
                then_args
            } else if *else_block == to {
                else_args
            } else {
                return None;
            }
        }
        _ => return None,
    };
    args.get(i).copied()
}

/// Least upper bound of two lattice values: any `Unknown` yields the
/// other, `Over` absorbs, and only equal constants keep a constant.
fn lub(a: &Lat, b: &Lat) -> Lat {
    match (a, b) {
        (Lat::Unknown, other) | (other, Lat::Unknown) => other.clone(),
        (Lat::Over, _) | (_, Lat::Over) => Lat::Over,
        (Lat::Const(x), Lat::Const(y)) if x == y => Lat::Const(x.clone()),
        (Lat::Const(_), Lat::Const(_)) => Lat::Over,
    }
}

fn lat_int(value: &Lat) -> Option<(u32, i128)> {
    match value {
        Lat::Const(Const::Int { bits, value }) => Some((*bits, *value)),
        _ => None,
    }
}

enum EdgeSet {
    /// The condition is still unknown; no edges proven yet.
    None,
    Only(Block),
    Both,
}

fn branch_targets(lat: &Lat, then_block: Block, else_block: Block) -> EdgeSet {
    match lat {
        Lat::Unknown => EdgeSet::None,
        Lat::Const(Const::Int { value: 0, .. }) => EdgeSet::Only(else_block),
        Lat::Const(_) => EdgeSet::Only(then_block),
        Lat::Over => EdgeSet::Both,
    }
}

/// The switch destination for a resolved scrutinee: the case whose key
/// matches the low bits of the scrutinee's own width (lowering promotes
/// the scrutinee first, so case keys share that width), or the default.
fn switch_take(lat: &Lat, cases: &[(i128, Block)], default: Block) -> Block {
    match lat {
        Lat::Const(Const::Int { bits, value }) => {
            let m = mask(*bits);
            let v = (*value as u128) & m;
            match cases.iter().find(|(k, _)| (*k as u128) & m == v) {
                Some((_, b)) => *b,
                None => default,
            }
        }
        _ => default,
    }
}

/// Lattice arithmetic for one instruction kind: `Unknown` operands stay
/// unknown; anything the bits-level fold helpers accept becomes a
/// constant and everything else (memory, calls, atomics, the `Va*`
/// state machine, float arithmetic) is overdefined.
fn eval(kind: &InstKind, result_ty: &Ty, lats: &[Lat]) -> Lat {
    let at = |v: Value| lats[v.index()].clone();
    /// A folded `Const` becomes a lattice value; `None` means the fold
    /// provider refused (trap condition, width mismatch) — overdefined.
    fn folded(c: Option<Const>) -> Lat {
        match c {
            Some(c) => Lat::Const(c),
            None => Lat::Over,
        }
    }
    let int2 = |a: Value, b: Value, f: fn(i128, i128) -> i128| match (lat_int(&at(a)), lat_int(&at(b))) {
        (Some((bits, x)), Some((_, y))) => folded(fold_int2_bits(bits, x, y, f)),
        _ => Lat::Over,
    };
    match kind {
        InstKind::Add(a, b) => int2(*a, *b, |x, y| x.wrapping_add(y)),
        InstKind::Sub(a, b) => int2(*a, *b, |x, y| x.wrapping_sub(y)),
        InstKind::Mul(a, b) => int2(*a, *b, |x, y| x.wrapping_mul(y)),
        InstKind::And(a, b) => int2(*a, *b, |x, y| x & y),
        InstKind::Or(a, b) => int2(*a, *b, |x, y| x | y),
        InstKind::Xor(a, b) => int2(*a, *b, |x, y| x ^ y),
        InstKind::Shl(a, b) => match (lat_int(&at(*a)), lat_int(&at(*b))) {
            (Some((bits, x)), Some((_, s))) => folded(fold_shift_bits(bits, x as u128, s, |x, s| x << s)),
            _ => Lat::Over,
        },
        InstKind::LShr(a, b) => match (lat_int(&at(*a)), lat_int(&at(*b))) {
            (Some((bits, x)), Some((_, s))) => folded(fold_shift_bits(bits, x as u128, s, |x, s| x >> s)),
            _ => Lat::Over,
        },
        InstKind::AShr(a, b) => match (lat_int(&at(*a)), lat_int(&at(*b))) {
            (Some((bits, x)), Some((_, s))) => folded(fold_ashr_bits(bits, x, s)),
            _ => Lat::Over,
        },
        InstKind::Udiv(a, b) => div_pair(&at(*a), &at(*b), false, false),
        InstKind::Sdiv(a, b) => div_pair(&at(*a), &at(*b), true, false),
        InstKind::Urem(a, b) => div_pair(&at(*a), &at(*b), false, true),
        InstKind::Srem(a, b) => div_pair(&at(*a), &at(*b), true, true),
        InstKind::Neg(a) => folded(lat_int(&at(*a)).and_then(|(bits, x)| fold_neg_bits(bits, x))),
        InstKind::Not(a) => folded(lat_int(&at(*a)).and_then(|(bits, x)| fold_not_bits(bits, x))),
        InstKind::FNeg(a) => match at(*a) {
            Lat::Const(Const::Float { kind, bits }) => {
                // F80 rides a 16-byte carrier with the x87 sign at bit
                // 79, not the top.
                let sign = match kind {
                    FloatKind::F32 => 1u128 << 31,
                    FloatKind::F64 => 1u128 << 63,
                    FloatKind::F80 => 1u128 << 79,
                    FloatKind::F128 => 1u128 << 127,
                };
                folded(Some(Const::Float { kind, bits: bits ^ sign }))
            }
            _ => Lat::Over,
        },
        InstKind::Trunc(a) => conv(&at(*a), result_ty, |x, _, tb| wrap(tb, x)),
        InstKind::Zext(a) => conv(&at(*a), result_ty, |x, ob, tb| wrap(tb, ((x as u128) & mask(ob)) as i128)),
        InstKind::Sext(a) => conv(&at(*a), result_ty, |x, _, tb| wrap(tb, x)),
        InstKind::Bitcast(a) => {
            let Some((obits, x)) = lat_int(&at(*a)) else {
                return Lat::Over;
            };
            let Some(tbits) = int_ty_bits(result_ty) else {
                return Lat::Over;
            };
            if obits != tbits {
                return Lat::Over;
            }
            folded(Some(Const::Int {
                bits: tbits,
                value: wrap(tbits, x),
            }))
        }
        InstKind::Icmp { pred, lhs, rhs } => match (lat_int(&at(*lhs)), lat_int(&at(*rhs))) {
            (Some((bits, l)), Some((_, r))) => folded(Some(Const::Int {
                bits: 1,
                value: i128::from(eval_icmp(*pred, bits, l, r)),
            })),
            _ => Lat::Over,
        },
        InstKind::Fcmp { pred, lhs, rhs } => float_cmp(*pred, &at(*lhs), &at(*rhs)),
        InstKind::Select { cond, then_val, else_val } => match &lats[cond.index()] {
            Lat::Unknown => Lat::Unknown,
            Lat::Const(Const::Int { value: 0, .. }) => at(*else_val),
            Lat::Const(_) => at(*then_val),
            Lat::Over => joined_lattice(at(*then_val), at(*else_val)),
        },
        _ => Lat::Over,
    }
}

/// A conversion fold over one integer lattice on the way to
/// `result_ty`'s width.
fn conv(a: &Lat, result_ty: &Ty, f: impl FnOnce(i128, u32, u32) -> i128) -> Lat {
    let Some((obits, x)) = lat_int(a) else {
        return Lat::Over;
    };
    let Some(tbits) = int_ty_bits(result_ty) else {
        return Lat::Over;
    };
    if tbits == 0 || tbits > 128 {
        return Lat::Over;
    }
    Lat::Const(Const::Int {
        bits: tbits,
        value: f(x, obits, tbits),
    })
}

/// A zero divisor or oversized width must keep the runtime trap, so an
/// unfolded division is overdefined, never a constant.
fn div_pair(a: &Lat, b: &Lat, signed: bool, is_rem: bool) -> Lat {
    match (lat_int(a), lat_int(b)) {
        (Some((bits, x)), Some((_, y))) => match fold_div_bits(bits, x, y, signed, is_rem) {
            Some(c) => Lat::Const(c),
            None => Lat::Over,
        },
        _ => Lat::Over,
    }
}

/// IEEE `f32`/`f64` comparisons with NaN semantics, everything wider or
/// mismatched stays overdefined.
fn float_cmp(pred: FcmpPred, lhs: &Lat, rhs: &Lat) -> Lat {
    let truth = match (lhs, rhs) {
        (Lat::Const(Const::Float { kind: FloatKind::F32, bits: lb }), Lat::Const(Const::Float { kind: FloatKind::F32, bits: rb })) => {
            fold_fcmp_bits(pred, f32::from_bits(*lb as u32) as f64, f32::from_bits(*rb as u32) as f64)
        }
        (Lat::Const(Const::Float { kind: FloatKind::F64, bits: lb }), Lat::Const(Const::Float { kind: FloatKind::F64, bits: rb })) => {
            fold_fcmp_bits(pred, f64::from_bits(*lb as u64), f64::from_bits(*rb as u64))
        }
        _ => return Lat::Over,
    };
    Lat::Const(Const::Int {
        bits: 1,
        value: i128::from(truth),
    })
}

/// Join of two select arms when the condition is overdefined.
fn joined_lattice(t: Lat, e: Lat) -> Lat {
    match (&t, &e) {
        (Lat::Const(a), Lat::Const(b)) if a == b => Lat::Const(a.clone()),
        (Lat::Const(_), Lat::Const(_)) | (Lat::Over, _) | (_, Lat::Over) => Lat::Over,
        // One arm unknown, the other constant: optimistic until every
        // path settles.
        (Lat::Unknown, other) | (other, Lat::Unknown) => other.clone(),
    }
}

/// Materialize the fixpoint: constant-lattice results get their uses
/// rewritten to a fresh constant of the value's own type, still-unknown
/// results become a typed `Undef`, only branches whose condition
/// resolved get rewritten, and only blocks with a proven edge keep
/// their terminator.
fn rewrite(func: &mut Function, lattice: &[Lat], executable: &[bool]) -> bool {
    let mut map: HashMap<Value, Value> = HashMap::new();
    let mut changed = false;

    // Only values still referenced from instructions or terminators
    // matter. Replacing the def of an unused one (e.g. a param whose
    // uses all folded away last run) pushes fresh arena values forever
    // and would make reruns non-idempotent.
    let mut used = vec![false; func.values.len()];
    for block_idx in 0..func.blocks.len() {
        let block = Block::from_usize(block_idx);
        for &inst in &func.blocks[block].insts {
            func.insts[inst].kind.for_each_operand(&mut |op| used[op.index()] = true);
        }
        if let Some(term) = &func.blocks[block].term {
            term.for_each_operand(&mut |op| used[op.index()] = true);
        }
    }

    for block_idx in 0..func.blocks.len() {
        let block = Block::from_usize(block_idx);
        for i in 0..func.blocks[block].params.len() {
            let param = func.blocks[block].params[i];
            if used[param.index()] {
                note_value(func, &mut map, lattice, param);
            }
        }
        for i in 0..func.blocks[block].insts.len() {
            let inst = func.blocks[block].insts[i];
            if let Some(result) = func.insts[inst].result {
                if used[result.index()] {
                    note_value(func, &mut map, lattice, result);
                }
            }
        }
    }
    // Decide terminators while `lattice` still covers every operand:
    // `remap_operands` below pushes fresh constants past the lattice's
    // end, so indexing it with a remapped condition would go out of
    // bounds. The arguments a fold picks are still the original values
    // and get rewritten by the same sweep.
    for block_idx in 0..func.blocks.len() {
        let block = Block::from_usize(block_idx);
        if !executable[block_idx] {
            // Never reached: dead. Instructions stay for now so that
            // SimplifyCfg has something to delete once the sweep above
            // has replaced every use, including in the dead edges'
            // argument lists.
            if !matches!(func.blocks[block].term, Some(Terminator::Unreachable)) {
                func.blocks[block].term = Some(Terminator::Unreachable);
                changed = true;
            }
            continue;
        }
        let Some(term) = func.blocks[block].term.clone() else {
            continue;
        };
        match term {
            Terminator::CondBr {
                cond,
                then_block,
                then_args,
                else_block,
                else_args,
                ..
            } => {
                let folded = match branch_targets(&lattice[cond.index()], then_block, else_block) {
                    EdgeSet::Only(dest) => {
                        let args = if dest == then_block { then_args } else { else_args };
                        Some(Terminator::Br { dest, args })
                    }
                    EdgeSet::None | EdgeSet::Both => None,
                };
                if let Some(t) = folded {
                    func.blocks[block].term = Some(t);
                    changed = true;
                }
            }
            // A switch on a resolved scrutinee falls straight to its
            // one reachable arm, which carries no arguments; an
            // overdefined scrutinee keeps the whole switch.
            Terminator::Switch { scrutinee, cases, default } => {
                if matches!(&lattice[scrutinee.index()], Lat::Const(_)) {
                    let dest = switch_take(&lattice[scrutinee.index()], &cases, default);
                    func.blocks[block].term = Some(Terminator::Br { dest, args: Default::default() });
                    changed = true;
                }
            }
            _ => {}
        }
    }

    if !map.is_empty() {
        func.remap_operands(&map);
        changed = true;
    }
    changed
}

/// Materialize one value's lattice into the replacement map, skipping
/// constants that already live as `Def::Const` in the arena.
fn note_value(func: &mut Function, map: &mut HashMap<Value, Value>, lattice: &[Lat], v: Value) {
    let lat = lattice[v.index()].clone();
    match (&lat, &func.values[v].def) {
        (Lat::Const(c), Def::Const(existing)) if c == existing => {}
        (Lat::Const(c), _) => {
            let origin = value_origin(func, v);
            let ty = func.values[v].ty.clone();
            let fresh = func.values.push(ValueData { ty, def: Def::Const(c.clone()) });
            map.insert(v, fresh);
            if let Some(origin) = origin {
                func.origins.insert(fresh, origin);
            }
        }
        (Lat::Unknown, _) => {
            let origin = value_origin(func, v);
            let ty = func.values[v].ty.clone();
            let fresh = func.values.push(ValueData {
                ty,
                def: Def::Const(Const::Undef),
            });
            map.insert(v, fresh);
            if let Some(origin) = origin {
                func.origins.insert(fresh, origin);
            }
        }
        (Lat::Over, _) => {}
    }
}

/// Debug provenance for the value `note_value` is about to replace in
/// place of `v`: the defining instruction's pre-fold text (its operands
/// are still the original values at this point) and source location, or
/// a block-param spelling. Constants need no origin.
fn value_origin(func: &Function, v: Value) -> Option<ValueOrigin> {
    match &func.values[v].def {
        Def::Inst(inst) => Some(ValueOrigin {
            pass: "sccp",
            text: crate::display::inst_str(func, &func.insts[*inst].kind),
            loc: func.insts[*inst].loc,
        }),
        Def::Param(block, i) => Some(ValueOrigin {
            pass: "sccp",
            text: format!("b{}.param{i}", block.index()),
            loc: None,
        }),
        Def::Const(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{InstData, MemFlags};
    use crate::ty::Signature;
    use crate::verify::verify_func;

    /// `(if c: 41+1 else: 42)`: both arms collapse to 42 no matter how
    /// the condition goes; the merge param becomes that constant.
    #[test]
    fn joins_matching_constants_through_a_phi() {
        let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let cond = param(&mut func, entry, Ty::Bool, 0);
        let a = func.push_const(Const::Int { bits: 32, value: 41 });
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let c42 = func.push_const(Const::Int { bits: 32, value: 42 });
        let then_block = func.push_block(vec![]);
        let else_block = func.push_block(vec![]);
        let sum = emit(&mut func, entry, InstKind::Add(a, one), Ty::int(32));
        let merge = func.push_block(vec![Ty::int(32)]);
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block,
            else_block,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        let merged = func.blocks[merge].params[0];
        func.blocks[then_block].term = Some(Terminator::Br {
            dest: merge,
            args: [sum].into_iter().collect(),
        });
        func.blocks[else_block].term = Some(Terminator::Br {
            dest: merge,
            args: [c42].into_iter().collect(),
        });
        func.blocks[merge].term = Some(Terminator::Return { val: Some(merged) });

        assert!(Sccp.run(&mut func));
        assert!(verify_func(&func).is_empty());
        let Some(Terminator::Return { val: Some(v) }) = func.blocks[merge].term.clone() else {
            panic!("merge still returns");
        };
        assert_eq!(func.const_value(v), Some(&Const::Int { bits: 32, value: 42 }));
        assert!(!Sccp.run(&mut func), "second run is a no-op");
    }

    /// The arms disagree, so the merge stays overdefined and nothing
    /// may be rewritten.
    #[test]
    fn keeps_conflicting_arms_overdefined() {
        let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let cond = param(&mut func, entry, Ty::Bool, 0);
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let two = func.push_const(Const::Int { bits: 32, value: 2 });
        let then_block = func.push_block(vec![]);
        let else_block = func.push_block(vec![]);
        let merge = func.push_block(vec![Ty::int(32)]);
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block,
            else_block,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        let merged = func.blocks[merge].params[0];
        func.blocks[then_block].term = Some(Terminator::Br {
            dest: merge,
            args: [one].into_iter().collect(),
        });
        func.blocks[else_block].term = Some(Terminator::Br {
            dest: merge,
            args: [two].into_iter().collect(),
        });
        func.blocks[merge].term = Some(Terminator::Return { val: Some(merged) });

        assert!(!Sccp.run(&mut func));
        assert!(verify_func(&func).is_empty());
        assert!(matches!(func.blocks[merge].term.clone().unwrap(), Terminator::Return { .. }));
    }

    #[test]
    fn folds_a_select_whose_arms_agree() {
        let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let cond = param(&mut func, entry, Ty::Bool, 0);
        let c = func.push_const(Const::Int { bits: 32, value: 7 });
        let sel = emit(&mut func, entry, InstKind::Select { cond, then_val: c, else_val: c }, Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sel) });
        assert!(Sccp.run(&mut func));
        assert!(verify_func(&func).is_empty());
        let out = returned(&func);
        assert_eq!(func.const_value(out), Some(&Const::Int { bits: 32, value: 7 }));
    }

    #[test]
    fn keeps_a_select_whose_arms_differ() {
        let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let cond = param(&mut func, entry, Ty::Bool, 0);
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let two = func.push_const(Const::Int { bits: 32, value: 2 });
        let sel = emit(
            &mut func,
            entry,
            InstKind::Select {
                cond,
                then_val: one,
                else_val: two,
            },
            Ty::int(32),
        );
        let out = emit(&mut func, entry, InstKind::Add(sel, one), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(out) });
        assert!(!Sccp.run(&mut func));
        assert!(verify_func(&func).is_empty());
    }

    /// A constant condition decides the successor before the code
    /// runs: the untaken arm becomes dead and the branch unconditional.
    #[test]
    fn prunes_the_unreachable_arm_of_a_constant_condition() {
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let cond = func.push_const(Const::Int { bits: 1, value: 1 });
        let then_block = func.push_block(vec![]);
        let else_block = func.push_block(vec![]);
        let null = func.push_const(Const::Null);
        let load = emit(
            &mut func,
            then_block,
            InstKind::Load {
                addr: null,
                flags: MemFlags::plain(),
            },
            Ty::int(32),
        );
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block,
            else_block,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        let zero = func.push_const(Const::Int { bits: 32, value: 0 });
        func.blocks[then_block].term = Some(Terminator::Return { val: Some(load) });
        func.blocks[else_block].term = Some(Terminator::Return { val: Some(zero) });

        assert!(Sccp.run(&mut func));
        assert!(verify_func(&func).is_empty());
        assert!(matches!(
            func.blocks[entry].term,
            Some(Terminator::Br { dest, .. }) if dest == then_block
        ));
        assert!(matches!(func.blocks[else_block].term, Some(Terminator::Unreachable)));
        // The reachable load stays overdefined (it still reads memory).
        assert!(func.const_value(load).is_none());
        assert!(func.blocks[then_block].insts.len() == 1);
    }

    /// A computed condition folds to a constant, so the branch is
    /// decided from a value that is itself rewritten to a fresh
    /// constant by the same pass. Deciding the terminator must not
    /// index the solve's lattice with that fresh value.
    #[test]
    fn folds_a_computed_condition_branch() {
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let zero = func.push_const(Const::Int { bits: 32, value: 0 });
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let cond = emit(&mut func, entry, InstKind::Add(zero, one), Ty::int(32));
        let then_block = func.push_block(vec![]);
        let else_block = func.push_block(vec![]);
        let t = func.push_const(Const::Int { bits: 32, value: 10 });
        let e = func.push_const(Const::Int { bits: 32, value: 20 });
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block,
            else_block,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        func.blocks[then_block].term = Some(Terminator::Return { val: Some(t) });
        func.blocks[else_block].term = Some(Terminator::Return { val: Some(e) });

        assert!(Sccp.run(&mut func));
        assert!(verify_func(&func).is_empty());
        assert!(matches!(
            func.blocks[entry].term,
            Some(Terminator::Br { dest, .. }) if dest == then_block
        ));
        assert!(matches!(func.blocks[else_block].term, Some(Terminator::Unreachable)));
    }

    /// A loop phi first joins a constant, then widens to overdefined when
    /// the back edge arrives. The widening has to flow back through that
    /// already-joined edge: a stale `Const(1)` here used to replace the
    /// loop counter's back-edge argument with a literal, so the loop
    /// never advanced.
    #[test]
    fn rejoins_a_phi_whose_argument_widens() {
        let scenario = widening_loop();
        let mut func = scenario.func;
        assert!(!Sccp.run(&mut func), "the loop counter must stay dynamic");
        assert!(verify_func(&func).is_empty());
        let Some(Terminator::Br { dest, args }) = func.blocks[scenario.back].term.clone() else {
            panic!("back edge vanished");
        };
        assert_eq!(dest, scenario.hdr);
        assert_eq!(args[0], scenario.carried, "stale constant joined as the loop argument");
    }

    /// The fixpoint checker accepts a correct solve: every param equals
    /// the join of its executable incoming arguments.
    #[test]
    fn solve_reaches_a_valid_fixpoint_for_a_widening_phi() {
        let scenario = widening_loop();
        let mut state = State::new(&scenario.func);
        state.solve(&scenario.func);
        let errors = state.fixpoint_errors(&scenario.func);
        assert!(errors.is_empty(), "{errors:?}");
    }

    /// The checker catches the stale-join class of bug directly: force a
    /// phi param back to the constant it first joined while its incoming
    /// argument has since widened, and the mismatch is reported.
    #[test]
    fn fixpoint_errors_flags_a_stale_phi_param() {
        let scenario = widening_loop();
        let mut state = State::new(&scenario.func);
        state.solve(&scenario.func);
        state.lattice[scenario.j.index()] = Lat::Const(Const::Int { bits: 32, value: 1 });
        let errors = state.fixpoint_errors(&scenario.func);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains(&format!("%{}", scenario.j.index())), "{}", errors[0]);
        assert!(errors[0].contains("Over"), "{}", errors[0]);
    }

    #[test]
    fn picks_the_matching_switch_arm() {
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let scrut = func.push_const(Const::Int { bits: 32, value: 2 });
        let low = func.push_block(vec![]);
        let high = func.push_block(vec![]);
        let one = func.push_const(Const::Int { bits: 32, value: 11 });
        let seven = func.push_const(Const::Int { bits: 32, value: 77 });
        func.blocks[entry].term = Some(Terminator::Switch {
            scrutinee: scrut,
            cases: vec![(1, low), (2, high)],
            default: low,
        });
        func.blocks[low].term = Some(Terminator::Return { val: Some(one) });
        func.blocks[high].term = Some(Terminator::Return { val: Some(seven) });
        assert!(Sccp.run(&mut func));
        assert!(verify_func(&func).is_empty());
        let Some(Terminator::Br { dest, .. }) = func.blocks[entry].term.clone() else {
            panic!("switch did not fold");
        };
        assert_eq!(dest, high);
    }

    /// A load feeding a merge from an always-taken path stays
    /// overdefined; nothing in the live code may fold away.
    #[test]
    fn keeps_memory_reads_overdefined() {
        let sig = Signature::new(vec![Ty::ptr(Ty::int(32))], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let p = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
        let read = emit(
            &mut func,
            entry,
            InstKind::Load {
                addr: p,
                flags: MemFlags::plain(),
            },
            Ty::int(32),
        );
        let forty = func.push_const(Const::Int { bits: 32, value: 40 });
        let two = func.push_const(Const::Int { bits: 32, value: 2 });
        let product = emit(&mut func, entry, InstKind::Mul(forty, two), Ty::int(32));
        let sum = emit(&mut func, entry, InstKind::Add(read, product), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        assert!(Sccp.run(&mut func));
        assert!(verify_func(&func).is_empty());
        // The load stays overdefined (its result is untouched, so the
        // sum still reads it), while the pure 40*2 folded to 80: the
        // sum's rhs is now a materialized constant. Sccp leaves the
        // orphaned mul for the dead-code sweep — it does not DCE.
        assert_eq!(func.blocks[entry].insts.len(), 3);
        match &func.insts[func.blocks[entry].insts[2]].kind {
            InstKind::Add(a, rhs) => {
                assert_eq!(*a, read);
                assert_eq!(func.values[*rhs].def, Def::Const(Const::Int { bits: 32, value: 80 }));
            }
            other => panic!("expected add over the load, got {other:?}"),
        }
    }

    /// Code reachable only through a never-taken edge: its results are
    /// Undef'd at the fixpoint, even when a live merge also joins a
    /// real value from the other arm.
    #[test]
    fn undefs_results_only_reached_through_dead_edges() {
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let cond = func.push_const(Const::Int { bits: 1, value: 0 });
        let then_block = func.push_block(vec![]);
        let else_block = func.push_block(vec![]);
        let merge = func.push_block(vec![Ty::int(32)]);
        let null = func.push_const(Const::Null);
        let dead_load = emit(
            &mut func,
            then_block,
            InstKind::Load {
                addr: null,
                flags: MemFlags::plain(),
            },
            Ty::int(32),
        );
        let seven = func.push_const(Const::Int { bits: 32, value: 7 });
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block,
            else_block,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        let merged = func.blocks[merge].params[0];
        func.blocks[then_block].term = Some(Terminator::Br {
            dest: merge,
            args: [dead_load].into_iter().collect(),
        });
        func.blocks[else_block].term = Some(Terminator::Br {
            dest: merge,
            args: [seven].into_iter().collect(),
        });
        // The disjunct get: `seven` and maybe-undef must join — that is
        // overdefined; the point is that the dead load's own result is
        // a Undef constant after the sweep.
        func.blocks[merge].term = Some(Terminator::Return { val: Some(merged) });
        assert!(Sccp.run(&mut func));
        assert!(verify_func(&func).is_empty());
        // The dead arm is out of the CFG reachable set; the merge
        // param folds to the only arm the branch can take, and the
        // return reads that constant.
        assert!(matches!(func.blocks[then_block].term, Some(Terminator::Unreachable)));
        let out = returned(&func);
        assert_eq!(func.const_value(out), Some(&Const::Int { bits: 32, value: 7 }));
    }

    fn returned(func: &Function) -> Value {
        for block in func.blocks.iter() {
            match &block.term {
                Some(Terminator::Return { val: Some(v) }) => return *v,
                _ => {}
            }
        }
        panic!("no value return in {func:?}");
    }

    /// The loop from `rejoins_a_phi_whose_argument_widens`: entry feeds
    /// `zero` into the header's counter phi, then the back edge carries a
    /// value that widens from `Const(1)` to `Over`.
    struct WideningLoop {
        func: Function,
        hdr: Block,
        back: Block,
        j: Value,
        carried: Value,
    }

    fn widening_loop() -> WideningLoop {
        let sig = Signature::new(vec![Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let cond = param(&mut func, entry, Ty::Bool, 0);
        let zero = func.push_const(Const::Int { bits: 32, value: 0 });
        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let hdr = func.push_block(vec![Ty::int(32)]);
        let body = func.push_block(vec![]);
        let back = func.push_block(vec![Ty::int(32)]);
        let out = func.push_block(vec![]);
        let j = func.blocks[hdr].params[0];
        func.blocks[entry].term = Some(Terminator::Br {
            dest: hdr,
            args: [zero].into_iter().collect(),
        });
        func.blocks[hdr].term = Some(Terminator::CondBr {
            cond,
            then_block: body,
            else_block: out,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        let next = emit(&mut func, body, InstKind::Add(j, one), Ty::int(32));
        func.blocks[body].term = Some(Terminator::Br {
            dest: back,
            args: [next].into_iter().collect(),
        });
        let carried = func.blocks[back].params[0];
        func.blocks[back].term = Some(Terminator::Br {
            dest: hdr,
            args: [carried].into_iter().collect(),
        });
        func.blocks[out].term = Some(Terminator::Return { val: Some(zero) });
        WideningLoop {
            func,
            hdr,
            back,
            j,
            carried,
        }
    }

    fn param(func: &mut Function, block: Block, ty: Ty, index: u32) -> Value {
        let v = func.values.push(ValueData {
            ty,
            def: Def::Param(block, index),
        });
        func.blocks[block].params.push(v);
        v
    }

    fn emit(func: &mut Function, block: Block, kind: InstKind, ty: Ty) -> Value {
        let result = func.values.push(ValueData {
            ty,
            def: Def::Inst(Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(InstData {
            block,
            kind,
            result: Some(result),
            loc: None,
        });
        func.values[result].def = Def::Inst(inst);
        func.blocks[block].insts.push(inst);
        result
    }
}
