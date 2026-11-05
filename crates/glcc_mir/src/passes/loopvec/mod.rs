//! Loop vectorization: turn counted loops over memory into a vector
//! loop plus a scalar epilogue.
//!
//! The shape the pass accepts is the one the C frontends emit: a
//! preheader entering the header, a header whose only branch is the
//! `icmp` of the induction variable against a loop-invariant bound
//! (`i <u n` or `i <s n`, either orientation of the `brif`), and a latch
//! that adds one to the induction variable and jumps back. All body
//! computation must be elementwise ops over unit-stride loads of one
//! element type, loop-invariant values, and constants, feeding
//! unit-stride stores — anything else (calls, carried values beyond the
//! induction variable, backward dependences) keeps the loop scalar.
//!
//! The rewrite guards the vector loop the way correctness demands:
//! when loads and stores through pointer parameters cannot be proven
//! disjoint (`dst`/`src` may alias), one runtime range check routes to
//! the scalar loop instead. Constant-trip loops are left to
//! [`LoopUnroll`](super::LoopUnroll) + SLP, which run after this pass.

mod affine;

use std::collections::{HashMap, HashSet};

use smallvec::smallvec;
use smallvec::SmallVec;

use crate::ir::{Block, Def, Function, IcmpPred, Inst, InstData, InstKind, MemFlags, Terminator, Value, ValueData, remap_inst};
use crate::pass::FunctionPass;
use crate::ty::Ty;

use super::DomInfo;
use super::alias::{Object, resolve};
use super::int_const;
use super::vector::{ElemOp, elem_op, vector_lanes};

use affine::affine_of;

/// Largest loop body (in instructions) worth vectorizing.
const MAX_LOOP_INSTS: usize = 64;

/// Constant trips at or below this stay with [`LoopUnroll`](super::LoopUnroll).
const MAX_TRIP_COUNT: usize = 8;

/// Vectorize counted natural loops with runtime trip counts.
#[derive(Default)]
pub struct LoopVec;

impl FunctionPass for LoopVec {
    fn name(&self) -> &'static str {
        "loop-vec"
    }

    fn run(&mut self, func: &mut Function) -> bool {
        if func.blocks.is_empty() {
            return false;
        }
        let reachable = func.reachable_blocks();
        let dom = DomInfo::compute(func, &reachable);
        let mut loops = super::licm::natural_loops(func, &dom);
        loops.retain(|(header, _)| reachable[header.index()]);
        loops.sort_by_key(|(header, blocks)| (blocks.len(), header.index()));
        for (header, blocks) in loops {
            let Some(shape) = analyze(func, &dom, header, &blocks) else {
                continue;
            };
            let Some(plan) = plan(func, &shape) else {
                continue;
            };
            apply(func, plan);
            return true;
        }
        false
    }
}

/// One integer reduction carried in a header param: `param += arg`
/// (or `-=`) every iteration.
#[derive(Clone, Copy)]
struct Reduction {
    /// Header param slot of the accumulator.
    index: usize,
    param: Value,
    /// `Add` or `Sub` only: float reductions would reassociate.
    op: ElemOp,
    /// The value folded in each iteration (a load, usually).
    arg: Value,
}

/// The counted-loop shape the pass transforms.
struct Shape {
    header: Block,
    latch: Block,
    preheader: Block,
    exit: Block,
    exit_args: SmallVec<[Value; 4]>,
    init_args: SmallVec<[Value; 4]>,
    /// Header param holding the induction variable, and its width.
    iv: Value,
    iv_bits: u32,
    /// Continue predicate, normalized so the loop runs while it holds:
    /// `Ult` (`i <u n`) or `Slt` (`i <s n`), induction variable first.
    pred: IcmpPred,
    /// Loop-invariant bound the induction variable is compared against.
    bound: Value,
    /// Whether the header's branch-true arm is the latch (else it is the
    /// exit, and the raw predicate is the negation of `pred`).
    cond_continues: bool,
    /// The header icmp's original predicate, for the scalar clone.
    raw_pred: IcmpPred,
    /// Integer accumulators folded every iteration, in param order.
    reductions: Vec<Reduction>,
}

fn analyze(func: &Function, dom: &DomInfo, header: Block, set: &HashSet<Block>) -> Option<Shape> {
    if set.len() != 2 || func.returns_twice {
        return None;
    }
    let latch = *set.iter().find(|block| **block != header)?;
    if !func.blocks[latch].params.is_empty() {
        return None;
    }
    let total_insts = set.iter().map(|block| func.blocks[*block].insts.len()).sum::<usize>();
    if total_insts > MAX_LOOP_INSTS {
        return None;
    }
    // The latch is one straight-line block jumping back.
    let Terminator::Br { dest, args: back_args } = func.blocks[latch].term.clone()? else {
        return None;
    };
    if dest != header {
        return None;
    }
    let preheader = preheader_of(func, dom, header, set)?;
    let Terminator::Br { dest, args: init_args } = func.blocks[preheader].term.clone()? else {
        return None;
    };
    if dest != header {
        return None;
    }

    let Terminator::CondBr { cond, then_block, else_block, then_args, else_args } = func.blocks[header].term.clone()? else {
        return None;
    };
    let (body_args, exit, exit_args, cond_continues) = if then_block == latch && else_block != latch {
        (then_args, else_block, else_args, true)
    } else if else_block == latch && then_block != latch {
        (else_args, then_block, then_args, false)
    } else {
        return None;
    };
    if !body_args.is_empty() {
        return None;
    }
    // The header computes nothing but the loop test; its params are the
    // induction variable plus integer reduction accumulators.
    let params = &func.blocks[header].params;
    if func.blocks[header].insts.len() != 1 || params.is_empty() || params.len() != init_args.len() || params.len() != back_args.len() {
        return None;
    }
    let cond_inst = match &func.values[cond].def {
        Def::Inst(inst) if func.insts[*inst].block == header && func.blocks[header].insts[0] == *inst => *inst,
        _ => return None,
    };
    let InstKind::Icmp { pred, lhs, rhs } = func.insts[cond_inst].kind else {
        return None;
    };
    // The branch may test the continue condition or the exit condition;
    // normalize to the continue predicate.
    let continue_pred = if cond_continues { pred } else { negate(pred)? };

    let mut iv = None;
    let mut reductions = Vec::new();
    for (index, param) in params.iter().enumerate() {
        let is_iv = int_const(func, init_args[index]).is_some_and(|(_, v)| v == 0)
            && steps_by_one(func, *param, back_args[index])
            && lhs == *param
            && rhs != *param
            && matches!(continue_pred, IcmpPred::Ult | IcmpPred::Slt);
        if is_iv {
            if iv.is_some() {
                return None;
            }
            // The bound must be defined outside the loop.
            if matches!(&func.values[rhs].def, Def::Inst(inst) if set.contains(&func.insts[*inst].block)) {
                return None;
            }
            // Small constant trips belong to LoopUnroll + SLP, which
            // fold them to straight-line code; a vector loop would
            // defeat constant folding.
            if let Some((_, v)) = int_const(func, rhs) {
                let trip = match continue_pred {
                    IcmpPred::Ult => v as u128,
                    _ => v.max(0) as u128,
                };
                if trip <= MAX_TRIP_COUNT as u128 {
                    return None;
                }
            }
            let Ty::Int { bits } = func.values[*param].ty else {
                return None;
            };
            iv = Some((*param, bits));
        } else {
            reductions.push(reduction_shape(func, index, *param, back_args[index])?);
        }
    }
    let (iv, iv_bits) = iv?;
    let bound = rhs;
    Some(Shape {
        header,
        latch,
        preheader,
        exit,
        exit_args,
        init_args,
        iv,
        iv_bits,
        pred: continue_pred,
        bound: rhs,
        cond_continues,
        raw_pred: pred,
        reductions,
    })
}

/// The reduction a header param must implement to be carried: `param
/// += arg` (either order) or `param -= arg`, integer-only so the
/// vector epilogue's reassociation is exact in modular arithmetic.
fn reduction_shape(func: &Function, index: usize, param: Value, update: Value) -> Option<Reduction> {
    if !matches!(func.values[param].ty, Ty::Int { .. }) {
        return None;
    }
    let Def::Inst(inst) = func.values[update].def else {
        return None;
    };
    let (op, arg) = match &func.insts[inst].kind {
        InstKind::Add(a, b) if *a == param => (ElemOp::Add, *b),
        InstKind::Add(a, b) if *b == param => (ElemOp::Add, *a),
        InstKind::Sub(a, b) if *a == param => (ElemOp::Sub, *b),
        _ => return None,
    };
    Some(Reduction { index, param, op, arg })
}

fn negate(pred: IcmpPred) -> Option<IcmpPred> {
    Some(match pred {
        IcmpPred::Ult => IcmpPred::Uge,
        IcmpPred::Uge => IcmpPred::Ult,
        IcmpPred::Slt => IcmpPred::Sge,
        IcmpPred::Sge => IcmpPred::Slt,
        _ => return None,
    })
}

fn steps_by_one(func: &Function, param: Value, update: Value) -> bool {
    let Def::Inst(inst) = func.values[update].def else { return false };
    match &func.insts[inst].kind {
        InstKind::Add(a, b) => {
            (*a == param && int_const(func, *b).is_some_and(|(_, v)| v == 1))
                || (*b == param && int_const(func, *a).is_some_and(|(_, v)| v == 1))
        }
        _ => false,
    }
}

/// Whether `value` is defined inside the loop: a loop-block param or an
/// inst residing in one.
fn loop_value(func: &Function, value: Value, header: Block, latch: Block) -> bool {
    match &func.values[value].def {
        Def::Const(_) => false,
        Def::Param(block, _) => *block == header || *block == latch,
        Def::Inst(inst) => {
            let block = func.insts[*inst].block;
            block == header || block == latch
        }
        _ => false,
    }
}

fn preheader_of(func: &Function, dom: &DomInfo, header: Block, set: &HashSet<Block>) -> Option<Block> {
    let mut outside: Vec<Block> = set.iter().flat_map(|block| dom.preds[block.index()].iter().copied()).collect();
    outside.retain(|block| !set.contains(block));
    outside.sort_by_key(|block| block.index());
    outside.dedup();
    if outside.len() != 1 {
        return None;
    }
    let preheader = outside[0];
    if !dom.dominates(preheader, header, func.entry) || func.blocks[preheader].term.is_none() {
        return None;
    }
    Some(preheader)
}

// ---------------------------------------------------------------------------
// Plan: legality and the vector body's value trees
// ---------------------------------------------------------------------------

/// One memory access the plan vectorizes or reasons about.
struct Access {
    /// The object base the access addresses within.
    root: Value,
    /// Byte offset from `root` to the access's lowest address.
    abs_off: i64,
    /// Bytes per element.
    elem_size: u64,
    /// True for a stride-0 load, whose range is one element instead of
    /// the whole trip.
    invariant: bool,
}

/// How one in-loop scalar value becomes a vector value.
enum VNode {
    /// Vector load for the unit-stride access at this plan index.
    VLoad(usize),
    /// Invariant-address load, read once and broadcast to all lanes.
    ILoad(usize),
    /// Elementwise op over the scalar operand values.
    Op(ElemOp, Value, Value),
}

struct Plan {
    shape: Shape,
    /// Vector element type and its lane count; every unit-stride access
    /// shares it.
    elem: Ty,
    lanes: u32,
    /// Vectorized stores in body order: address access plus the stored
    /// scalar value.
    stores: Vec<(Access, Value)>,
    /// Loads the trees consume: unit-stride or invariant.
    loads: Vec<Access>,
    /// In-loop scalar values mapped to how their vector form builds.
    trees: HashMap<Value, VNode>,
    /// Distinct invariant pointers the preheader must `ptrtoint` for
    /// address math.
    roots: Vec<Value>,
    /// (store, load) pairs whose disjointness only a runtime range
    /// check can establish, as indices into stores-then-loads.
    alias_pairs: Vec<(usize, usize)>,
}

fn plan(func: &Function, shape: &Shape) -> Option<Plan> {
    let in_loop = |v: Value| loop_value(func, v, shape.header, shape.latch);
    let mut stores: Vec<(Access, Value)> = Vec::new();
    let mut loads: Vec<Access> = Vec::new();
    let mut trees: HashMap<Value, VNode> = HashMap::new();
    let mut elem: Option<Ty> = None;
    for &inst in &func.blocks[shape.latch].insts {
        let InstKind::Store { addr, val, flags } = &func.insts[inst].kind else {
            continue;
        };
        if flags.volatile {
            return None;
        }
        let elem_ty = func.values[*val].ty.clone();
        let es = elem_ty.size_bytes();
        let aff = affine_of(func, *addr, shape.iv, &in_loop)?;
        let root = aff.root?;
        if aff.iv_coeff != es as i64 || aff.base_off % es as i64 != 0 || es == 0 {
            return None;
        }
        // One element type per loop: all unit accesses share it.
        match &elem {
            Some(prev) if *prev != elem_ty => return None,
            Some(_) => {}
            None => {
                vector_lanes(&elem_ty)?;
                elem = Some(elem_ty.clone());
            }
        }
        let (root, abs) = object_offset(func, root, aff.base_off)?;
        let access = Access { root, abs_off: abs, elem_size: es, invariant: false };
        value_tree(func, shape, *val, &elem_ty, &in_loop, &mut loads, &mut trees)?;
        stores.push((access, *val));
    }
    // Reductions without stores anchor the element type themselves.
    let elem = match elem {
        Some(e) => e,
        None => {
            let red = shape.reductions.first()?;
            let ty = func.values[red.arg].ty.clone();
            vector_lanes(&ty)?;
            ty
        }
    };
    for red in &shape.reductions {
        if func.values[red.arg].ty != elem {
            return None;
        }
        value_tree(func, shape, red.arg, &elem, &in_loop, &mut loads, &mut trees)?;
    }
    let lanes = vector_lanes(&elem)?;

    // Insts outside the trees (address math feeding load addresses, the
    // induction update) simply don't reach the vector body; their uses
    // remap to the scalar clone. Anything with side effects must.
    for &inst in &func.blocks[shape.latch].insts {
        if func.insts[inst].result.is_some_and(|result| trees.contains_key(&result))
            || matches!(&func.insts[inst].kind, InstKind::Store { .. })
        {
            continue;
        }
        if !func.insts[inst].kind.is_pure() {
            return None;
        }
    }

    // Dependences between the accesses the vector body performs.
    let mut alias_pairs: Vec<(usize, usize)> = Vec::new();
    for (si, (s, _)) in stores.iter().enumerate() {
        for (li, l) in loads.iter().enumerate() {
            match relation(func, s, l) {
                Relation::Safe => {}
                Relation::Check => alias_pairs.push((si, stores.len() + li)),
                Relation::Bail => return None,
            }
        }
        for (other, (o, _)) in stores.iter().enumerate() {
            if other != si && !stores_compatible(func, s, o) {
                return None;
            }
        }
    }
    let mut roots: Vec<Value> = Vec::new();
    for (access, _) in &stores {
        push_root(&mut roots, access.root);
    }
    for access in &loads {
        push_root(&mut roots, access.root);
    }
    Some(Plan {
        shape: Shape {
            header: shape.header,
            latch: shape.latch,
            preheader: shape.preheader,
            exit: shape.exit,
            exit_args: shape.exit_args.clone(),
            init_args: shape.init_args.clone(),
            iv: shape.iv,
            iv_bits: shape.iv_bits,
            pred: shape.pred,
            bound: shape.bound,
            cond_continues: shape.cond_continues,
            raw_pred: shape.raw_pred,
            reductions: shape.reductions.clone(),
        },
        elem,
        lanes,
        stores,
        loads,
        trees,
        roots,
        alias_pairs,
    })
}

fn push_root(roots: &mut Vec<Value>, root: Value) {
    if !roots.contains(&root) {
        roots.push(root);
    }
}

/// Object base and offset of an access: the affine root may be a
/// derived address (`inttoptr(base + n)`, a GEP) that already carries an
/// offset, so decompose it first — emission adds `abs_off` to the base
/// pointer, and the offset baked into the root must not be added twice.
fn object_offset(func: &Function, root: Value, base_off: i64) -> Option<(Value, i64)> {
    let (object, off) = resolve(func, root)?;
    let off = i64::try_from(off?).ok()?;
    Some((object.base_value(), off.checked_add(base_off)?))
}

/// The vector form of one in-loop scalar value, memoized so shared
/// subexpressions build once. Constants and invariants need no entry.
fn value_tree(func: &Function, shape: &Shape, value: Value, elem: &Ty, in_loop: &impl Fn(Value) -> bool, loads: &mut Vec<Access>, trees: &mut HashMap<Value, VNode>) -> Option<()> {
    // Constants and invariants need no entry: emission broadcasts them
    // directly.
    if trees.contains_key(&value) || !in_loop(value) {
        return Some(());
    }
    let node = build_tree(func, shape, value, elem, in_loop, loads, trees)?;
    trees.insert(value, node);
    Some(())
}

fn build_tree(func: &Function, shape: &Shape, value: Value, elem: &Ty, in_loop: &impl Fn(Value) -> bool, loads: &mut Vec<Access>, trees: &mut HashMap<Value, VNode>) -> Option<VNode> {
    let Def::Inst(inst) = func.values[value].def else {
        // The induction variable and anything else carried is off
        // limits; invariants and constants never reach here.
        return None;
    };
    if func.insts[inst].block != shape.latch {
        return None;
    }
    match &func.insts[inst].kind {
        InstKind::Load { addr, flags } => {
            if flags.volatile || func.values[value].ty != *elem {
                return None;
            }
            let aff = affine_of(func, *addr, shape.iv, in_loop)?;
            let root = aff.root?;
            let es = elem.size_bytes();
            if aff.iv_coeff == es as i64 && aff.base_off % es as i64 == 0 {
                let (root, abs) = object_offset(func, root, aff.base_off)?;
                loads.push(Access { root, abs_off: abs, elem_size: es, invariant: false });
                Some(VNode::VLoad(loads.len() - 1))
            } else if aff.iv_coeff == 0 {
                // Invariant address: read once per vector iteration.
                let (root, abs) = object_offset(func, root, aff.base_off)?;
                loads.push(Access { root, abs_off: abs, elem_size: es, invariant: true });
                Some(VNode::ILoad(loads.len() - 1))
            } else {
                None
            }
        }
        kind => {
            let op = elem_op(kind)?;
            let (a, b) = operands_of(kind, op)?;
            value_tree(func, shape, a, elem, in_loop, loads, trees)?;
            value_tree(func, shape, b, elem, in_loop, loads, trees)?;
            Some(VNode::Op(op, a, b))
        }
    }
}

fn operands_of(kind: &InstKind, op: ElemOp) -> Option<(Value, Value)> {
    if elem_op(kind) != Some(op) {
        return None;
    }
    Some(match kind {
        InstKind::Add(a, b)
        | InstKind::Sub(a, b)
        | InstKind::Mul(a, b)
        | InstKind::And(a, b)
        | InstKind::Or(a, b)
        | InstKind::Xor(a, b)
        | InstKind::Shl(a, b)
        | InstKind::FAdd(a, b)
        | InstKind::FSub(a, b)
        | InstKind::FMul(a, b) => (*a, *b),
        _ => unreachable!("elem_op matched a binary op"),
    })
}

/// How one store relates to one load of the vector body.
enum Relation {
    /// No overlap is possible.
    Safe,
    /// Overlap is possible; a runtime range check decides.
    Check,
    /// Overlap would change semantics in a way no check fixes.
    Bail,
}

fn object_of(func: &Function, access: &Access) -> Option<Object> {
    resolve(func, access.root).map(|(object, _)| object)
}

fn relation(func: &Function, s: &Access, l: &Access) -> Relation {
    let (Some(so), Some(lo)) = (object_of(func, s), object_of(func, l)) else {
        return Relation::Bail;
    };
    if so != lo {
        return match (&so, &lo) {
            // Distinct named objects never overlap; anything involving
            // an opaque root needs the runtime check.
            (Object::Known(_), Object::Known(_)) => Relation::Safe,
            _ => Relation::Check,
        };
    }
    if s.invariant && l.invariant {
        // The same fixed address rewritten every iteration: the
        // store-then-load order inside the body decides, and the vector
        // body always loads first.
        return Relation::Bail;
    }
    if s.invariant || l.invariant {
        // A fixed address against the sweeping range: the runtime check
        // compares one element against the whole trip.
        return Relation::Check;
    }
    // Same memory, both sweeping: a forward or zero distance keeps each
    // load ahead of the store that would clobber it; a backward one is
    // a carried dependence.
    let d = l.abs_off - s.abs_off;
    if d >= 0 && d % s.elem_size as i64 == 0 {
        Relation::Safe
    } else {
        Relation::Bail
    }
}

/// Whether two stores in the loop can coexist: same address (the later
/// one wins, order preserved) or provably distinct objects. Overlapping
/// but different stores reorder ambiguously.
fn stores_compatible(func: &Function, a: &Access, b: &Access) -> bool {
    let (Some(ao), Some(bo)) = (object_of(func, a), object_of(func, b)) else {
        return false;
    };
    ao != bo || a.abs_off == b.abs_off
}

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

/// Appends instructions to one block during the rewrite.
struct Emit<'f> {
    func: &'f mut Function,
    block: Block,
}

impl Emit<'_> {
    fn push(&mut self, kind: InstKind, ty: Ty) -> Value {
        let result = self.func.values.push(ValueData {
            ty,
            def: Def::Inst(Inst::from_usize(self.func.insts.len())),
        });
        let inst = self.func.insts.push(InstData {
            block: self.block,
            kind,
            result: Some(result),
            loc: None,
        });
        self.func.values[result].def = Def::Inst(inst);
        self.func.blocks[self.block].insts.push(inst);
        result
    }

    fn push_unit(&mut self, kind: InstKind) {
        let inst = self.func.insts.push(InstData {
            block: self.block,
            kind,
            result: None,
            loc: None,
        });
        self.func.blocks[self.block].insts.push(inst);
    }

    fn iconst(&mut self, bits: u32, value: i128) -> Value {
        self.func.push_const(crate::ir::Const::Int { bits, value })
    }
}

fn apply(func: &mut Function, plan: Plan) {
    let Plan {
        shape:
            Shape {
                header,
                latch,
                preheader,
                exit,
                exit_args,
                init_args,
                iv,
                iv_bits,
                pred,
                bound,
                cond_continues,
                raw_pred,
                reductions,
            },
        elem,
        lanes,
        stores,
        loads,
        trees,
        roots,
        alias_pairs,
    } = plan;
    let original_header = func.blocks[header].clone();
    let original_latch = func.blocks[latch].clone();
    let iv_ty = func.values[iv].ty.clone();
    let log2 = lanes.trailing_zeros();
    let es64 = (lanes as u64) * elem.size_bytes();
    let vty = Ty::vector(elem.clone(), lanes);

    // Preheader math: normalized trip count, vector trip count, pointer
    // bases, and the runtime alias check when disjointness is not
    // structural.
    let mut alias_ok: Option<Value> = None;
    let mut root_ints: HashMap<Value, Value> = HashMap::new();
    let (vec_trip, shift) = {
        let mut pre = Emit { func, block: preheader };
        let zero = pre.iconst(iv_bits, 0);
        let nn = match pred {
            IcmpPred::Slt => {
                let pos = pre.push(InstKind::Icmp { pred: IcmpPred::Sgt, lhs: bound, rhs: zero }, Ty::Bool);
                pre.push(InstKind::Select { cond: pos, then_val: bound, else_val: zero }, iv_ty.clone())
            }
            _ => bound,
        };
        let shift = pre.iconst(iv_bits, log2 as i128);
        let vec_trip = pre.push(InstKind::LShr(nn, shift), iv_ty.clone());
        let nn64 = if iv_bits < 64 {
            pre.push(InstKind::Zext(nn), Ty::int(64))
        } else {
            nn
        };
        for root in &roots {
            let int = pre.push(InstKind::PtrToInt(*root), Ty::int(64));
            root_ints.insert(*root, int);
        }
        for (si, li) in &alias_pairs {
            let s = pair_access(&stores, &loads, *si);
            let l = pair_access(&stores, &loads, *li);
            let s_start = base_addr(&mut pre, &root_ints, s.root, s.abs_off);
            let s_range = range(&mut pre, nn64, s);
            let s_end = plus(&mut pre, s_start, s_range);
            let l_start = base_addr(&mut pre, &root_ints, l.root, l.abs_off);
            let l_range = range(&mut pre, nn64, l);
            let l_end = plus(&mut pre, l_start, l_range);
            let ahead = pre.push(InstKind::Icmp { pred: IcmpPred::Ule, lhs: s_end, rhs: l_start }, Ty::Bool);
            let behind = pre.push(InstKind::Icmp { pred: IcmpPred::Ule, lhs: l_end, rhs: s_start }, Ty::Bool);
            let cond = pre.push(InstKind::Or(ahead, behind), Ty::Bool);
            alias_ok = Some(match alias_ok {
                Some(prev) => pre.push(InstKind::And(prev, cond), Ty::Bool),
                None => cond,
            });
        }
        (vec_trip, shift)
    };

    // New control flow: a guarded vector loop feeding the scalar clone.
    let acc_tys: Vec<Ty> = reductions.iter().map(|_| vty.clone()).collect();
    let vec_header = func.push_block([iv_ty.clone()].into_iter().chain(acc_tys.clone()).collect());
    let vec_body = func.push_block(vec![]);
    let vec_exit = func.push_block(vec![]);
    let scalar_header = func.push_block(func.blocks[header].params.iter().map(|p| func.values[*p].ty.clone()).collect());
    let scalar_latch = func.push_block(vec![]);
    let vec_i = func.blocks[vec_header].params[0];
    let acc_vecs: Vec<Value> = func.blocks[vec_header].params[1..].to_vec();
    let scalar_params = func.blocks[scalar_header].params.clone();
    let scalar_iv = scalar_params[0];

    {
        let mut vh = Emit { func, block: vec_header };
        let more = vh.push(InstKind::Icmp { pred: IcmpPred::Ult, lhs: vec_i, rhs: vec_trip }, Ty::Bool);
        vh.func.blocks[vec_header].term = Some(Terminator::CondBr {
            cond: more,
            then_block: vec_body,
            else_block: vec_exit,
            then_args: SmallVec::new(),
            else_args: SmallVec::new(),
        });
    }

    // Vector body: tree values (loads first, via memoized emission),
    // then the stores in body order, then the induction bump.
    let mut vectors: HashMap<Value, Value> = HashMap::new();
    {
        let mut vb = Emit { func, block: vec_body };
        let vec_i64 = if iv_bits < 64 {
            vb.push(InstKind::Zext(vec_i), Ty::int(64))
        } else {
            vec_i
        };
        let stride = vb.iconst(64, es64 as i128);
        let swept = vb.push(InstKind::Mul(vec_i64, stride), Ty::int(64));
        for (access, val) in &stores {
            let vector = emit_tree(&mut vb, *val, &trees, &loads, elem.clone(), vty.clone(), &root_ints, swept, &mut vectors);
            let addr = sweep_addr(&mut vb, &root_ints, access, swept, elem.clone());
            vb.push_unit(InstKind::Store { addr, val: vector, flags: MemFlags::plain() });
        }
        let mut raccs: Vec<Value> = Vec::new();
        for (red, acc) in reductions.iter().zip(&acc_vecs) {
            let varg = emit_tree(&mut vb, red.arg, &trees, &loads, elem.clone(), vty.clone(), &root_ints, swept, &mut vectors);
            raccs.push(vb.push(red.op.kind(*acc, varg), vty.clone()));
        }
        let one = vb.iconst(iv_bits, 1);
        let incr = vb.push(InstKind::Add(vec_i, one), iv_ty.clone());
        let mut back = smallvec![incr];
        back.extend(raccs);
        vb.func.blocks[vec_body].term = Some(Terminator::Br {
            dest: vec_header,
            args: back,
        });
    }

    {
        let mut ve = Emit { func, block: vec_exit };
        let start = ve.push(InstKind::Shl(vec_trip, shift), iv_ty.clone());
        // init + sum of lanes: exact for integer add and subtract in
        // modular arithmetic, since each lane accumulates
        // `0 (op) arg[i] arg[i+lanes] ...`.
        let mut exit_args_new = smallvec![start];
        if !reductions.is_empty() {
            let Ty::Int { bits: elem_bits } = elem else { unreachable!("reductions are integer") };
            for (red, acc) in reductions.iter().zip(&acc_vecs) {
                let mut total = ve.push(InstKind::ExtractLane { vec: *acc, lane: 0 }, elem.clone());
                for lane in 1..lanes {
                    let part = ve.push(InstKind::ExtractLane { vec: *acc, lane: lane as u8 }, elem.clone());
                    total = ve.push(InstKind::Add(total, part), elem.clone());
                }
                let init = init_args[red.index];
                let final_acc = ve.push(InstKind::Add(init, total), Ty::int(elem_bits));
                exit_args_new.push(final_acc);
            }
        }
        ve.func.blocks[vec_exit].term = Some(Terminator::Br {
            dest: scalar_header,
            args: exit_args_new,
        });
    }

    // Preheader routes to the vector loop, falling back to the scalar
    // clone when the runtime alias check fires. Vector accumulators
    // start at zero; the epilogue adds the original init back.
    let mut acc_inits: Vec<Value> = Vec::new();
    if !reductions.is_empty() {
        let mut pre = Emit { func, block: preheader };
        let Ty::Int { bits: elem_bits } = elem else { unreachable!("reductions are integer") };
        let zero_elem = pre.iconst(elem_bits, 0);
        for _ in &reductions {
            acc_inits.push(pre.push(InstKind::Splat(zero_elem), vty.clone()));
        }
    }
    let entry = func.push_const(crate::ir::Const::Int { bits: iv_bits, value: 0 });
    let mut vec_entry_args = smallvec![entry];
    vec_entry_args.extend(acc_inits.iter().copied());
    func.blocks[preheader].term = Some(match alias_ok {
        Some(ok) => Terminator::CondBr {
            cond: ok,
            then_block: vec_header,
            else_block: scalar_header,
            then_args: vec_entry_args,
            else_args: init_args.clone(),
        },
        None => Terminator::Br {
            dest: vec_header,
            args: vec_entry_args,
        },
    });

    // Scalar clone of the original loop, entered with the original init
    // args (alias-check failure) or the vector loop's leftovers.
    let mut map: HashMap<Value, Value> = HashMap::new();
    for (old, new) in func.blocks[header].params.iter().zip(&scalar_params) {
        map.insert(*old, *new);
    }
    for &old in &original_latch.insts {
        let data = &func.insts[old];
        let kind = remap_inst(&data.kind, |v| map.get(&v).copied().unwrap_or(v));
        let old_result = data.result;
        let loc = data.loc;
        let inst = func.insts.push(InstData {
            block: scalar_latch,
            kind,
            result: None,
            loc,
        });
        let result = old_result.map(|old_result| {
            let ty = func.values[old_result].ty.clone();
            let result = func.values.push(ValueData { ty, def: Def::Inst(inst) });
            map.insert(old_result, result);
            result
        });
        func.insts[inst].result = result;
        func.blocks[scalar_latch].insts.push(inst);
    }
    let backs = match &original_latch.term {
        Some(Terminator::Br { args, .. }) => args.clone(),
        _ => unreachable!("counted loop latch jumps back"),
    };
    let new_backs: SmallVec<[Value; 4]> = backs.iter().map(|v| map.get(v).copied().unwrap_or(*v)).collect();
    func.blocks[scalar_latch].term = Some(Terminator::Br {
        dest: scalar_header,
        args: new_backs,
    });
    {
        let icmp = func.insts[original_header.insts[0]].result.expect("header test has a result");
        let mut sh = Emit { func, block: scalar_header };
        let cond = sh.push(InstKind::Icmp { pred: raw_pred, lhs: scalar_iv, rhs: bound }, Ty::Bool);
        map.insert(icmp, cond);
        let mapped: SmallVec<[Value; 4]> = exit_args.iter().map(|v| map.get(v).copied().unwrap_or(*v)).collect();
        let term = if cond_continues {
            Terminator::CondBr {
                cond,
                then_block: scalar_latch,
                else_block: exit,
                then_args: SmallVec::new(),
                else_args: mapped,
            }
        } else {
            Terminator::CondBr {
                cond,
                then_block: exit,
                else_block: scalar_latch,
                then_args: mapped,
                else_args: SmallVec::new(),
            }
        };
        sh.func.blocks[scalar_header].term = Some(term);
    }

    // Uses outside the loop now see the clone's values.
    func.remap_operands(&map);
    for block in [header, latch] {
        func.blocks[block].insts.clear();
        func.blocks[block].term = Some(Terminator::Unreachable);
    }
}

/// The stores' and loads' access list indexed the way `alias_pairs`
/// refers to it.
fn pair_access<'a>(stores: &'a [(Access, Value)], loads: &'a [Access], index: usize) -> &'a Access {
    if index < stores.len() {
        &stores[index].0
    } else {
        &loads[index - stores.len()]
    }
}

/// Range in bytes one access covers over the whole trip: one element
/// for invariant loads, the full sweep for unit-stride accesses.
fn range(pre: &mut Emit, nn64: Value, access: &Access) -> Value {
    if access.invariant {
        pre.iconst(64, access.elem_size as i128)
    } else {
        let bytes = pre.iconst(64, access.elem_size as i128);
        pre.push(InstKind::Mul(nn64, bytes), Ty::int(64))
    }
}

/// `root_int + off` in 64-bit address arithmetic.
fn base_addr(pre: &mut Emit, root_ints: &HashMap<Value, Value>, root: Value, off: i64) -> Value {
    let off_v = pre.iconst(64, off as i128);
    match root_ints.get(&root) {
        Some(int) => pre.push(InstKind::Add(*int, off_v), Ty::int(64)),
        None => off_v,
    }
}

fn plus(pre: &mut Emit, base: Value, delta: Value) -> Value {
    pre.push(InstKind::Add(base, delta), Ty::int(64))
}

/// Emission of one scalar value's vector form, memoized in `vectors`.
fn emit_tree(
    vb: &mut Emit,
    value: Value,
    trees: &HashMap<Value, VNode>,
    loads: &[Access],
    elem: Ty,
    vty: Ty,
    root_ints: &HashMap<Value, Value>,
    swept: Value,
    vectors: &mut HashMap<Value, Value>,
) -> Value {
    if let Some(built) = vectors.get(&value) {
        return *built;
    }
    let vector = match trees.get(&value) {
        Some(VNode::VLoad(index)) => {
            let addr = sweep_addr(vb, root_ints, &loads[*index], swept, elem.clone());
            vb.push(InstKind::Load { addr, flags: MemFlags::plain() }, vty.clone())
        }
        Some(VNode::ILoad(index)) => {
            let access = &loads[*index];
            let base = base_addr(vb, root_ints, access.root, access.abs_off);
            let addr = vb.push(InstKind::IntToPtr(base), Ty::ptr(elem.clone()));
            let scalar = vb.push(InstKind::Load { addr, flags: MemFlags::plain() }, elem.clone());
            vb.push(InstKind::Splat(scalar), vty.clone())
        }
        Some(VNode::Op(op, a, b)) => {
            let va = emit_tree(vb, *a, trees, loads, elem.clone(), vty.clone(), root_ints, swept, vectors);
            let vbb = emit_tree(vb, *b, trees, loads, elem.clone(), vty.clone(), root_ints, swept, vectors);
            vb.push(op.kind(va, vbb), vty.clone())
        }
        // Invariants and constants broadcast directly.
        None => vb.push(InstKind::Splat(value), vty.clone()),
    };
    vectors.insert(value, vector);
    vector
}

/// The per-vector-iteration address of a unit-stride access:
/// `inttoptr(root + base_off + vec_i * lanes * elem_size)`.
fn sweep_addr(vb: &mut Emit, root_ints: &HashMap<Value, Value>, access: &Access, swept: Value, elem: Ty) -> Value {
    let base = base_addr(vb, root_ints, access.root, access.abs_off);
    let total = plus(vb, base, swept);
    vb.push(InstKind::IntToPtr(total), Ty::ptr(elem))
}

#[cfg(test)]
mod tests;
