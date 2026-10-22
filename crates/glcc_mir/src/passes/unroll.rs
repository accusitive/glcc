//! Full unrolling for small counted loops.
//!
//! A natural loop can be replaced by straight-line code when its trip
//! count is known: the induction variable starts at a constant, the
//! latch updates it by a constant step, and the header compares it
//! against a constant. This pass emits one copy of the loop per
//! iteration plus a final header copy whose exit edge replaces the
//! loop. Constant propagation then folds the copies, so a loop like
//! `for (int i = 0; i < 2; i++) sum += 1;` becomes `sum += 2`.
//!
//! Only loops with one latch and one exit at the header are considered.
//! A `break`, a return, or any other edge leaving the loop keeps the
//! dynamic trip count, and loops above [`MAX_TRIP_COUNT`] stay rolled
//! so a large constant bound cannot blow up the function.

use std::collections::{HashMap, HashSet};

use smallvec::SmallVec;

use crate::ir::{remap_inst, Block, BlockData, Def, Function, IcmpPred, InstData, InstKind, Terminator, Value, ValueData};
use crate::pass::FunctionPass;

use super::licm::natural_loops;
use super::{eval_icmp, int_const, DomInfo};

/// Largest trip count expanded into straight-line code.
const MAX_TRIP_COUNT: usize = 8;

/// Largest loop body (in instructions) worth expanding. Big bodies are
/// left rolled even when the trip count is small.
const MAX_LOOP_INSTS: usize = 64;

/// Fully unroll counted natural loops.
#[derive(Default)]
pub struct LoopUnroll;

impl FunctionPass for LoopUnroll {
    fn name(&self) -> &'static str {
        "loop-unroll"
    }

    fn run(&mut self, func: &mut Function) -> bool {
        if func.blocks.is_empty() {
            return false;
        }
        let reachable = func.reachable_blocks();
        let dom = DomInfo::compute(func, &reachable);
        let mut loops = natural_loops(func, &dom);
        loops.retain(|(header, _)| reachable[header.index()]);
        // Inner loops first: unrolling an inner loop after duplicating an
        // outer one would expand the same body once per outer iteration.
        loops.sort_by_key(|(header, blocks)| (blocks.len(), header.index()));
        for (header, blocks) in loops {
            let Some(counted) = analyze(func, &dom, header, &blocks) else {
                continue;
            };
            unroll(func, &counted);
            return true;
        }
        false
    }
}

struct CountedLoop {
    header: Block,
    loop_blocks: Vec<Block>,
    preheader: Block,
    exit: Block,
    body: Block,
    body_args: SmallVec<[Value; 4]>,
    exit_args: SmallVec<[Value; 4]>,
    init_args: SmallVec<[Value; 4]>,
    order: Vec<Block>,
    trip_count: usize,
}

fn analyze(func: &Function, dom: &DomInfo, header: Block, set: &HashSet<Block>) -> Option<CountedLoop> {
    if set.len() < 2 || func.returns_twice {
        return None;
    }
    let preheader = preheader_of(func, dom, header, set)?;
    let Terminator::Br { dest, args: init_args } = func.blocks[preheader].term.clone()? else {
        return None;
    };
    if dest != header {
        return None;
    }

    let (body, body_args, exit, exit_args) = header_edges(func, header, set)?;
    let mut backedges = Vec::new();
    for &block in set {
        if block == header {
            continue;
        }
        let term = func.blocks[block].term.as_ref()?;
        match term {
            Terminator::Br { dest, args } => {
                if !set.contains(dest) {
                    // A break or an early return: the trip count is not
                    // determined by the header alone.
                    return None;
                }
                if *dest == header {
                    backedges.push((block, args.clone()));
                }
            }
            Terminator::CondBr { then_block, else_block, .. } => {
                if !set.contains(then_block) || !set.contains(else_block) || *then_block == header || *else_block == header {
                    return None;
                }
            }
            Terminator::Switch { cases, default, .. } => {
                if !set.contains(default) || *default == header || cases.iter().any(|(_, dest)| !set.contains(dest) || *dest == header) {
                    return None;
                }
            }
            Terminator::Return { .. } | Terminator::Unreachable => return None,
        }
    }
    if backedges.len() != 1 {
        return None;
    }
    let (_, back_args) = backedges.pop()?;

    let header_params = &func.blocks[header].params;
    if init_args.len() != header_params.len() || back_args.len() != header_params.len() {
        return None;
    }
    if set.iter().map(|block| func.blocks[*block].insts.len()).sum::<usize>() > MAX_LOOP_INSTS {
        return None;
    }

    let Terminator::CondBr { cond, .. } = func.blocks[header].term.as_ref()? else {
        return None;
    };
    let (pred, lhs, rhs) = icmp_of_cond(func, header, *cond)?;
    let trip_count = header_params
        .iter()
        .enumerate()
        .find_map(|(index, param)| trip_count(func, *param, pred, lhs, rhs, init_args[index], back_args[index]))
        .filter(|count| *count <= MAX_TRIP_COUNT)?;

    let order = loop_order(func, header, set)?;
    let mut loop_blocks: Vec<Block> = set.iter().copied().collect();
    loop_blocks.sort_by_key(|block| block.index());

    Some(CountedLoop {
        header,
        loop_blocks,
        preheader,
        exit,
        body,
        body_args,
        exit_args,
        init_args,
        order,
        trip_count,
    })
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

/// The header's conditional branch split into its in-loop successor
/// (the body) and its out-of-loop successor (the exit), with the edge
/// arguments belonging to each.
fn header_edges(func: &Function, header: Block, set: &HashSet<Block>) -> Option<(Block, SmallVec<[Value; 4]>, Block, SmallVec<[Value; 4]>)> {
    let Terminator::CondBr {
        then_block,
        else_block,
        then_args,
        else_args,
        ..
    } = func.blocks[header].term.clone()?
    else {
        return None;
    };
    match (set.contains(&then_block), set.contains(&else_block)) {
        (true, false) => Some((then_block, then_args, else_block, else_args)),
        (false, true) => Some((else_block, else_args, then_block, then_args)),
        _ => None,
    }
}

fn icmp_of_cond(func: &Function, header: Block, cond: Value) -> Option<(IcmpPred, Value, Value)> {
    let Def::Inst(inst) = func.values[cond].def else {
        return None;
    };
    if func.insts[inst].block != header {
        return None;
    }
    let InstKind::Icmp { pred, lhs, rhs } = func.insts[inst].kind else {
        return None;
    };
    Some((pred, lhs, rhs))
}

/// Trip count of a loop whose header compares `param` with a constant
/// and whose latch updates it by a constant step. The simulation stops
/// at the unroll cap, so a non-terminating update is rejected.
fn trip_count(func: &Function, param: Value, pred: IcmpPred, lhs: Value, rhs: Value, init: Value, back: Value) -> Option<usize> {
    let step = update_step(func, param, back)?;
    let (bits, mut iv) = int_const(func, init)?;
    let (_, bound) = if lhs == param {
        int_const(func, rhs)?
    } else if rhs == param {
        int_const(func, lhs)?
    } else {
        return None;
    };
    let mut count = 0usize;
    loop {
        let live = if lhs == param {
            eval_icmp(pred, bits, iv, bound)
        } else {
            eval_icmp(pred, bits, bound, iv)
        };
        if !live {
            return Some(count);
        }
        count += 1;
        if count > MAX_TRIP_COUNT {
            return None;
        }
        iv = super::wrap(bits, iv.wrapping_add(step));
    }
}

fn update_step(func: &Function, param: Value, update: Value) -> Option<i128> {
    let Def::Inst(inst) = func.values[update].def else {
        return None;
    };
    match &func.insts[inst].kind {
        InstKind::Add(a, b) if *a == param => int_const(func, *b).map(|(_, value)| value),
        InstKind::Add(a, b) if *b == param => int_const(func, *a).map(|(_, value)| value),
        InstKind::Sub(a, b) if *a == param => int_const(func, *b).map(|(_, value)| -value),
        _ => None,
    }
}

/// Topological order of the loop's forward edges. The latch edge back
/// to the header is skipped; any other cycle means the loop is not
/// shaped like one counted loop and is rejected.
fn loop_order(func: &Function, header: Block, set: &HashSet<Block>) -> Option<Vec<Block>> {
    fn visit(func: &Function, block: Block, header: Block, set: &HashSet<Block>, state: &mut HashMap<Block, u8>, order: &mut Vec<Block>) -> Option<()> {
        match state.get(&block).copied().unwrap_or(0) {
            1 => return None,
            2 => return Some(()),
            _ => {}
        }
        state.insert(block, 1);
        order.push(block);
        if let Some(term) = &func.blocks[block].term {
            for succ in term.successors() {
                if succ == header || !set.contains(&succ) {
                    continue;
                }
                visit(func, succ, header, set, state, order)?;
            }
        }
        state.insert(block, 2);
        Some(())
    }

    let mut order = Vec::with_capacity(set.len());
    visit(func, header, header, set, &mut HashMap::new(), &mut order)?;
    (order.len() == set.len()).then_some(order)
}

fn unroll(func: &mut Function, counted: &CountedLoop) {
    let original: HashMap<Block, BlockData> = counted.loop_blocks.iter().map(|block| (*block, func.blocks[*block].clone())).collect();
    let header_params = original[&counted.header].params.clone();
    let header_tys: Vec<_> = header_params.iter().map(|p| func.values[*p].ty.clone()).collect();

    let mut header_clones = Vec::with_capacity(counted.trip_count + 1);
    for _ in 0..=counted.trip_count {
        header_clones.push(func.push_block(header_tys.clone()));
    }

    let mut block_maps: Vec<HashMap<Block, Block>> = Vec::with_capacity(counted.trip_count);
    let mut value_maps: Vec<HashMap<Value, Value>> = Vec::with_capacity(counted.trip_count);
    for iteration in 0..counted.trip_count {
        let mut blocks = HashMap::new();
        blocks.insert(counted.header, header_clones[iteration]);
        let mut values = HashMap::new();
        let new_params = func.blocks[header_clones[iteration]].params.clone();
        values.extend(header_params.iter().copied().zip(new_params));

        for &block in &counted.order {
            if block == counted.header {
                continue;
            }
            let tys = original[&block].params.iter().map(|p| func.values[*p].ty.clone()).collect();
            let clone = func.push_block(tys);
            blocks.insert(block, clone);
            let new_params = func.blocks[clone].params.clone();
            values.extend(original[&block].params.iter().copied().zip(new_params));
        }
        block_maps.push(blocks);
        value_maps.push(values);
    }

    for iteration in 0..counted.trip_count {
        clone_insts(func, &original, &counted.order, &block_maps[iteration], &mut value_maps[iteration]);
    }

    let mut tail_values: HashMap<Value, Value> = HashMap::new();
    let tail_params = func.blocks[header_clones[counted.trip_count]].params.clone();
    tail_values.extend(header_params.iter().copied().zip(tail_params));
    clone_block_insts(func, &original, counted.header, header_clones[counted.trip_count], &mut tail_values);

    for iteration in 0..counted.trip_count {
        let blocks = &block_maps[iteration];
        let values = &value_maps[iteration];
        for &block in &counted.order {
            if block == counted.header {
                continue;
            }
            let term = original[&block].term.as_ref().expect("loop block terminated");
            let term = clone_term(term, blocks, values, counted.header, header_clones[iteration + 1]);
            func.blocks[blocks[&block]].term = Some(term);
        }
        let args = map_values(&counted.body_args, values);
        func.blocks[header_clones[iteration]].term = Some(Terminator::Br {
            dest: blocks[&counted.body],
            args,
        });
    }

    let args = map_values(&counted.exit_args, &tail_values);
    func.blocks[header_clones[counted.trip_count]].term = Some(Terminator::Br { dest: counted.exit, args });
    func.blocks[counted.preheader].term = Some(Terminator::Br {
        dest: header_clones[0],
        args: counted.init_args.clone(),
    });

    let mut external = HashMap::new();
    for param in &header_params {
        if let Some(new) = tail_values.get(param) {
            external.insert(*param, *new);
        }
    }
    for &inst in &original[&counted.header].insts {
        if let Some(result) = func.insts[inst].result {
            if let Some(new) = tail_values.get(&result) {
                external.insert(result, *new);
            }
        }
    }
    if let Some(values) = value_maps.last() {
        for &block in &counted.loop_blocks {
            if block == counted.header {
                continue;
            }
            for param in &original[&block].params {
                if let Some(new) = values.get(param) {
                    external.insert(*param, *new);
                }
            }
            for &inst in &original[&block].insts {
                if let Some(result) = func.insts[inst].result {
                    if let Some(new) = values.get(&result) {
                        external.insert(result, *new);
                    }
                }
            }
        }
    }

    for &block in &counted.loop_blocks {
        func.blocks[block].insts.clear();
        func.blocks[block].term = Some(Terminator::Unreachable);
    }
    func.remap_operands(&external);
}

fn clone_insts(func: &mut Function, original: &HashMap<Block, BlockData>, order: &[Block], blocks: &HashMap<Block, Block>, values: &mut HashMap<Value, Value>) {
    for &block in order {
        clone_block_insts(func, original, block, blocks[&block], values);
    }
}

fn clone_block_insts(func: &mut Function, original: &HashMap<Block, BlockData>, block: Block, clone: Block, values: &mut HashMap<Value, Value>) {
    for &old in &original[&block].insts {
        let data = &func.insts[old];
        let kind = remap_inst(&data.kind, |value| values.get(&value).copied().unwrap_or(value));
        let old_result = data.result;
        let loc = data.loc.clone();
        let inst = func.insts.push(InstData {
            block: clone,
            kind,
            result: None,
            loc,
        });
        let result = old_result.map(|old_result| {
            let ty = func.values[old_result].ty.clone();
            let result = func.values.push(ValueData { ty, def: Def::Inst(inst) });
            values.insert(old_result, result);
            result
        });
        func.insts[inst].result = result;
        func.blocks[clone].insts.push(inst);
    }
}

fn clone_term(term: &Terminator, blocks: &HashMap<Block, Block>, values: &HashMap<Value, Value>, header: Block, next_header: Block) -> Terminator {
    let block = |dest: Block| {
        if dest == header {
            next_header
        } else {
            blocks[&dest]
        }
    };
    let value = |value: Value| values.get(&value).copied().unwrap_or(value);
    match term {
        Terminator::Br { dest, args } => Terminator::Br {
            dest: block(*dest),
            args: map_values(args, values),
        },
        Terminator::CondBr {
            cond,
            then_block,
            else_block,
            then_args,
            else_args,
        } => Terminator::CondBr {
            cond: value(*cond),
            then_block: block(*then_block),
            else_block: block(*else_block),
            then_args: map_values(then_args, values),
            else_args: map_values(else_args, values),
        },
        Terminator::Switch { scrutinee, cases, default } => Terminator::Switch {
            scrutinee: value(*scrutinee),
            cases: cases.iter().map(|(case, dest)| (*case, block(*dest))).collect(),
            default: block(*default),
        },
        Terminator::Return { .. } | Terminator::Unreachable => {
            unreachable!("counted loops have no early exits")
        }
    }
}

fn map_values(values: &[Value], map: &HashMap<Value, Value>) -> SmallVec<[Value; 4]> {
    values.iter().map(|value| map.get(value).copied().unwrap_or(*value)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Const, Func, InstData, InstKind, Module, ValueData};
    use crate::optimize::{optimize, OptLevel};
    use crate::ty::{Signature, Ty};

    fn emit(func: &mut Function, block: Block, kind: InstKind, ty: Ty) -> Value {
        let result = func.values.push(ValueData {
            ty,
            def: Def::Inst(crate::ir::Inst::from_usize(func.insts.len())),
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

    fn counted_loop(start: i128, bound: i128) -> (Module, Func) {
        let mut module = Module::new();
        let sig = Signature::new(vec![], Ty::int(32));
        let (mut func, entry) = Function::new("main", sig);
        let header = func.push_block(vec![Ty::int(32), Ty::int(32)]);
        let body = func.push_block(vec![]);
        let exit = func.push_block(vec![Ty::int(32)]);
        let sum = func.blocks[header].params[0];
        let i = func.blocks[header].params[1];

        let zero = func.push_const(Const::Int { bits: 32, value: 0 });
        let start = func.push_const(Const::Int { bits: 32, value: start });
        func.blocks[entry].term = Some(Terminator::Br {
            dest: header,
            args: [start, zero].into_iter().collect(),
        });

        let bound = func.push_const(Const::Int { bits: 32, value: bound });
        let cond = emit(
            &mut func,
            header,
            InstKind::Icmp {
                pred: IcmpPred::Slt,
                lhs: i,
                rhs: bound,
            },
            Ty::Bool,
        );
        func.blocks[header].term = Some(Terminator::CondBr {
            cond,
            then_block: body,
            else_block: exit,
            then_args: Default::default(),
            else_args: [sum].into_iter().collect(),
        });

        let one = func.push_const(Const::Int { bits: 32, value: 1 });
        let next_sum = emit(&mut func, body, InstKind::Add(sum, one), Ty::int(32));
        let next_i = emit(&mut func, body, InstKind::Add(i, one), Ty::int(32));
        func.blocks[body].term = Some(Terminator::Br {
            dest: header,
            args: [next_sum, next_i].into_iter().collect(),
        });

        let result = func.blocks[exit].params[0];
        func.blocks[exit].term = Some(Terminator::Return { val: Some(result) });
        let func = module.push_func(func);
        (module, func)
    }

    #[test]
    fn a_small_counted_loop_folds_to_its_sum() {
        let (mut module, func) = counted_loop(100, 2);
        optimize(&mut module, OptLevel::Speed);
        let main = &module.funcs[func];
        let Terminator::Return { val: Some(value) } = main.blocks[main.entry].term.clone().unwrap() else {
            panic!("expected a value return");
        };
        assert_eq!(main.const_value(value), Some(&Const::Int { bits: 32, value: 102 }));
        assert!(crate::verify::verify_module(&module).is_empty());
    }

    #[test]
    fn a_large_counted_loop_stays_rolled() {
        let (mut module, func) = counted_loop(0, 100);
        let mut pass = LoopUnroll;
        assert!(!pass.run(&mut module.funcs[func]));
        assert!(crate::verify::verify_module(&module).is_empty());
    }
}
