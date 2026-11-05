use super::*;
use crate::ir::{Const, Terminator};
use crate::ty::Signature;
use crate::verify::verify_func;
use smallvec::smallvec;

fn param(func: &mut Function, block: Block, ty: Ty, index: u32) -> Value {
    let v = func.values.push(ValueData { ty, def: Def::Param(block, index) });
    func.blocks[block].params.push(v);
    v
}

fn emit(func: &mut Function, block: Block, kind: InstKind, ty: Ty) -> Value {
    let result = func.values.push(ValueData { ty, def: Def::Inst(Inst::from_usize(func.insts.len())) });
    let inst = func.insts.push(InstData { block, kind, result: Some(result), loc: None });
    func.values[result].def = Def::Inst(inst);
    func.blocks[block].insts.push(inst);
    result
}

fn emit_unit(func: &mut Function, block: Block, kind: InstKind) {
    let inst = func.insts.push(InstData { block, kind, result: None, loc: None });
    func.blocks[block].insts.push(inst);
}

fn iconst(func: &mut Function, bits: u32, value: i128) -> Value {
    func.push_const(Const::Int { bits, value })
}

/// `dst[i] = src[i] + 1` with a runtime bound, in the exact shape the C
/// frontend emits: ptrtoint bases, sext/shl index math, inttoptr
/// addresses, two-block loop, `i <s n`.
fn runtime_add_loop() -> (Function, Block, Value, Value, Value) {
    let sig = Signature::new(vec![Ty::ptr(Ty::int(32)), Ty::ptr(Ty::int(32)), Ty::int(32)], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let dst = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
    let src = param(&mut func, entry, Ty::ptr(Ty::int(32)), 1);
    let n = param(&mut func, entry, Ty::int(32), 2);
    let dsti = emit(&mut func, entry, InstKind::PtrToInt(dst), Ty::int(64));
    let srci = emit(&mut func, entry, InstKind::PtrToInt(src), Ty::int(64));
    let header = func.push_block(vec![Ty::int(32)]);
    let iv = func.blocks[header].params[0];
    func.blocks[entry].term = Some(Terminator::Br { dest: header, args: smallvec::smallvec![iconst(&mut func, 32, 0)] });
    let body = func.push_block(vec![]);
    let exit = func.push_block(vec![]);
    let cond = emit(&mut func, header, InstKind::Icmp { pred: IcmpPred::Slt, lhs: iv, rhs: n }, Ty::Bool);
    func.blocks[header].term = Some(Terminator::CondBr { cond, then_block: body, else_block: exit, then_args: smallvec::smallvec![], else_args: smallvec::smallvec![] });
    let sext = emit(&mut func, body, InstKind::Sext(iv), Ty::int(64));
    let two = iconst(&mut func, 64, 2);
    let scaled = emit(&mut func, body, InstKind::Shl(sext, two), Ty::int(64));
    let daddr = emit(&mut func, body, InstKind::Add(dsti, scaled), Ty::int(64));
    let dptr = emit(&mut func, body, InstKind::IntToPtr(daddr), Ty::ptr(Ty::int(32)));
    let saddr = emit(&mut func, body, InstKind::Add(srci, scaled), Ty::int(64));
    let sptr = emit(&mut func, body, InstKind::IntToPtr(saddr), Ty::ptr(Ty::int(32)));
    let loaded = emit(&mut func, body, InstKind::Load { addr: sptr, flags: MemFlags::plain() }, Ty::int(32));
    let one = iconst(&mut func, 32, 1);
    let bumped = emit(&mut func, body, InstKind::Add(loaded, one), Ty::int(32));
    emit_unit(&mut func, body, InstKind::Store { addr: dptr, val: bumped, flags: MemFlags::plain() });
    let next = emit(&mut func, body, InstKind::Add(iv, one), Ty::int(32));
    func.blocks[body].term = Some(Terminator::Br { dest: header, args: smallvec::smallvec![next] });
    func.blocks[exit].term = Some(Terminator::Return { val: None });
    (func, header, dst, src, n)
}

#[test]
fn debug_shape() {
    let (mut func, ..) = runtime_add_loop();
    let errors = verify_func(&func);
    assert!(errors.is_empty(), "verifier: {errors:?}");
    let reachable = func.reachable_blocks();
    let dom = DomInfo::compute(&func, &reachable);
    let loops = super::super::licm::natural_loops(&func, &dom);
    eprintln!("loops: {:?}", loops.iter().map(|(h, s)| (h.index(), s.len())).collect::<Vec<_>>());
    let Some((header, set)) = loops.first() else { panic!("no loop found") };
    let shape = analyze(&func, &dom, *header, set);
    eprintln!("shape ok: {}", shape.is_some());
    assert!(shape.is_some(), "analyze failed");
    let p = plan(&func, shape.as_ref().unwrap());
    eprintln!("plan ok: {}", p.is_some());
    assert!(p.is_some(), "plan failed");
}

fn run(func: &mut Function) -> bool {
    let mut pass = LoopVec;
    let changed = crate::pass::FunctionPass::run(&mut pass, func);
    let errors = verify_func(func);
    assert!(errors.is_empty(), "verifier: {errors:?}");
    changed
}

fn vector_values(func: &Function) -> usize {
    func.values.iter().filter(|d| matches!(d.ty, Ty::Vector { .. })).count()
}

#[test]
fn runtime_bound_loop_vectorizes() {
    let (mut func, ..) = runtime_add_loop();
    assert!(run(&mut func));
    assert!(vector_values(&func) > 0, "no vector values emitted");
    assert!(!run(&mut func), "second run is a no-op");
    // The vector loop must test its induction variable unsigned against
    // the vector trip count; the scalar clone must survive.
    let mut saw_vector_icmp = false;
    let mut saw_scalar_icmp = false;
    for data in func.insts.iter() {
        if let InstKind::Icmp { pred, .. } = &data.kind {
            match pred {
                IcmpPred::Ult => saw_vector_icmp = true,
                IcmpPred::Slt => saw_scalar_icmp = true,
                _ => {}
            }
        }
    }
    assert!(saw_vector_icmp, "no unsigned vector-loop test");
    assert!(saw_scalar_icmp, "scalar clone lost its test");
}

#[test]
fn empty_and_exact_trips_keep_structures() {
    // Structural only: n = 0 and n = 8 route through the same guards.
    let (mut func, ..) = runtime_add_loop();
    assert!(run(&mut func));
    assert_eq!(func.blocks.len(), 9, "4 original (3 retired) + vector header/body/exit + scalar header/latch");
}

#[test]
fn backward_dependence_keeps_loop_scalar() {
    // dst[i + 1] = src[i] with src aliasing dst: the store sweeps one
    // element ahead of the load, so each load would see the previous
    // iteration's store — a carried dependence no runtime check fixes.
    let sig = Signature::new(vec![Ty::ptr(Ty::int(32)), Ty::ptr(Ty::int(32)), Ty::int(32)], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let dst = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
    let src = param(&mut func, entry, Ty::ptr(Ty::int(32)), 1);
    let n = param(&mut func, entry, Ty::int(32), 2);
    let dsti = emit(&mut func, entry, InstKind::PtrToInt(dst), Ty::int(64));
    let srci = emit(&mut func, entry, InstKind::PtrToInt(src), Ty::int(64));
    let header = func.push_block(vec![Ty::int(32)]);
    let iv = func.blocks[header].params[0];
    func.blocks[entry].term = Some(Terminator::Br { dest: header, args: smallvec![iconst(&mut func, 32, 0)] });
    let body = func.push_block(vec![]);
    let exit = func.push_block(vec![]);
    let cond = emit(&mut func, header, InstKind::Icmp { pred: IcmpPred::Slt, lhs: iv, rhs: n }, Ty::Bool);
    func.blocks[header].term = Some(Terminator::CondBr { cond, then_block: body, else_block: exit, then_args: smallvec![], else_args: smallvec![] });
    let sext = emit(&mut func, body, InstKind::Sext(iv), Ty::int(64));
    let two = iconst(&mut func, 64, 2);
    let scaled = emit(&mut func, body, InstKind::Shl(sext, two), Ty::int(64));
    let four = iconst(&mut func, 64, 4);
    let daddr = emit(&mut func, body, InstKind::Add(dsti, scaled), Ty::int(64));
    let bump = emit(&mut func, body, InstKind::Add(daddr, four), Ty::int(64));
    let dptr = emit(&mut func, body, InstKind::IntToPtr(bump), Ty::ptr(Ty::int(32)));
    let sptr = emit(&mut func, body, InstKind::IntToPtr(daddr), Ty::ptr(Ty::int(32)));
    let loaded = emit(&mut func, body, InstKind::Load { addr: sptr, flags: MemFlags::plain() }, Ty::int(32));
    let one = iconst(&mut func, 32, 1);
    let bumped = emit(&mut func, body, InstKind::Add(loaded, one), Ty::int(32));
    emit_unit(&mut func, body, InstKind::Store { addr: dptr, val: bumped, flags: MemFlags::plain() });
    let next = emit(&mut func, body, InstKind::Add(iv, one), Ty::int(32));
    func.blocks[body].term = Some(Terminator::Br { dest: header, args: smallvec![next] });
    func.blocks[exit].term = Some(Terminator::Return { val: None });
    assert!(!run(&mut func));
}

#[test]
fn call_in_body_keeps_loop_scalar() {
    let (mut func, ..) = runtime_add_loop();
    let callee = crate::ir::Func::from_usize(0);
    let body = Block::from_usize(2);
    let junk = emit(&mut func, body, InstKind::Call { callee, args: smallvec![] }, Ty::int(32));
    // Give the call's result a use so it is not dead.
    let exit_unused = emit(&mut func, body, InstKind::Add(junk, junk), Ty::int(32));
    let _ = exit_unused;
    // The extra Add is dead; the call itself is impure and must block.
    assert!(!run(&mut func));
}

#[test]
fn unsigned_bounds_vectorize_too() {
    let sig = Signature::new(vec![Ty::ptr(Ty::int(32)), Ty::ptr(Ty::int(32)), Ty::int(64)], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let dst = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
    let src = param(&mut func, entry, Ty::ptr(Ty::int(32)), 1);
    let n = param(&mut func, entry, Ty::int(64), 2);
    let dsti = emit(&mut func, entry, InstKind::PtrToInt(dst), Ty::int(64));
    let srci = emit(&mut func, entry, InstKind::PtrToInt(src), Ty::int(64));
    let header = func.push_block(vec![Ty::int(64)]);
    let iv = func.blocks[header].params[0];
    func.blocks[entry].term = Some(Terminator::Br { dest: header, args: smallvec![iconst(&mut func, 64, 0)] });
    let body = func.push_block(vec![]);
    let exit = func.push_block(vec![]);
    let cond = emit(&mut func, header, InstKind::Icmp { pred: IcmpPred::Ult, lhs: iv, rhs: n }, Ty::Bool);
    func.blocks[header].term = Some(Terminator::CondBr { cond, then_block: body, else_block: exit, then_args: smallvec![], else_args: smallvec![] });
    let two = iconst(&mut func, 64, 2);
    let scaled = emit(&mut func, body, InstKind::Shl(iv, two), Ty::int(64));
    let daddr = emit(&mut func, body, InstKind::Add(dsti, scaled), Ty::int(64));
    let dptr = emit(&mut func, body, InstKind::IntToPtr(daddr), Ty::ptr(Ty::int(32)));
    let saddr = emit(&mut func, body, InstKind::Add(srci, scaled), Ty::int(64));
    let sptr = emit(&mut func, body, InstKind::IntToPtr(saddr), Ty::ptr(Ty::int(32)));
    let loaded = emit(&mut func, body, InstKind::Load { addr: sptr, flags: MemFlags::plain() }, Ty::int(32));
    let one = iconst(&mut func, 64, 1);
    let const_one = iconst(&mut func, 32, 1);
    let bumped = emit(&mut func, body, InstKind::Add(loaded, const_one), Ty::int(32));
    emit_unit(&mut func, body, InstKind::Store { addr: dptr, val: bumped, flags: MemFlags::plain() });
    let next = emit(&mut func, body, InstKind::Add(iv, one), Ty::int(64));
    func.blocks[body].term = Some(Terminator::Br { dest: header, args: smallvec![next] });
    func.blocks[exit].term = Some(Terminator::Return { val: None });
    assert!(run(&mut func));
    assert!(vector_values(&func) > 0);
}

/// `sum += a[i]` with a runtime bound, plus `dst[i] = a[i]` so the loop
/// carries both a store and a reduction.
fn reduction_loop() -> (Function, Block, Value, Value, Value) {
    let sig = Signature::new(vec![Ty::ptr(Ty::int(32)), Ty::ptr(Ty::int(32)), Ty::int(32), Ty::int(32)], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    let a = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
    let dst = param(&mut func, entry, Ty::ptr(Ty::int(32)), 1);
    let sum_init = param(&mut func, entry, Ty::int(32), 2);
    let n = param(&mut func, entry, Ty::int(32), 3);
    let ai = emit(&mut func, entry, InstKind::PtrToInt(a), Ty::int(64));
    let dsti = emit(&mut func, entry, InstKind::PtrToInt(dst), Ty::int(64));
    let header = func.push_block(vec![Ty::int(32), Ty::int(32)]);
    let iv = func.blocks[header].params[0];
    let sum = func.blocks[header].params[1];
    func.blocks[entry].term = Some(Terminator::Br { dest: header, args: smallvec![iconst(&mut func, 32, 0), sum_init] });
    let body = func.push_block(vec![]);
    let exit = func.push_block(vec![]);
    let cond = emit(&mut func, header, InstKind::Icmp { pred: IcmpPred::Slt, lhs: iv, rhs: n }, Ty::Bool);
    func.blocks[header].term = Some(Terminator::CondBr { cond, then_block: body, else_block: exit, then_args: smallvec![], else_args: smallvec![] });
    let sext = emit(&mut func, body, InstKind::Sext(iv), Ty::int(64));
    let two = iconst(&mut func, 64, 2);
    let scaled = emit(&mut func, body, InstKind::Shl(sext, two), Ty::int(64));
    let aaddr = emit(&mut func, body, InstKind::Add(ai, scaled), Ty::int(64));
    let aptr = emit(&mut func, body, InstKind::IntToPtr(aaddr), Ty::ptr(Ty::int(32)));
    let loaded = emit(&mut func, body, InstKind::Load { addr: aptr, flags: MemFlags::plain() }, Ty::int(32));
    let daddr = emit(&mut func, body, InstKind::Add(dsti, scaled), Ty::int(64));
    let dptr = emit(&mut func, body, InstKind::IntToPtr(daddr), Ty::ptr(Ty::int(32)));
    emit_unit(&mut func, body, InstKind::Store { addr: dptr, val: loaded, flags: MemFlags::plain() });
    let acc = emit(&mut func, body, InstKind::Add(sum, loaded), Ty::int(32));
    let one32 = iconst(&mut func, 32, 1);
    let next = emit(&mut func, body, InstKind::Add(iv, one32), Ty::int(32));
    func.blocks[body].term = Some(Terminator::Br { dest: header, args: smallvec![next, acc] });
    func.blocks[exit].term = Some(Terminator::Return { val: None });
    (func, header, a, dst, sum)
}

#[test]
fn reduction_carries_through_the_epilogue() {
    let (mut func, ..) = reduction_loop();
    assert!(run(&mut func));
    // The vector loop must carry a v4xi32 accumulator; the epilogue
    // folds the lanes back into the scalar sum.
    assert!(vector_values(&func) > 0);
    let mut saw_extract = false;
    for data in func.insts.iter() {
        if matches!(&data.kind, InstKind::ExtractLane { .. }) {
            saw_extract = true;
        }
    }
    assert!(saw_extract, "no lane reduction in the epilogue");
}
