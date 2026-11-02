use super::*;
use crate::debug::File;
use crate::ir::{Block, Const, Def, Inst, InstData, Value, ValueData, ValueOrigin};
use crate::ty::{GepIndex, Signature};

fn param(func: &mut Function, block: Block, ty: Ty) -> Value {
    let index = func.blocks[block].params.len() as u32;
    let value = func.values.push(ValueData {
        ty,
        def: Def::Param(block, index),
    });
    func.blocks[block].params.push(value);
    value
}

fn emit(func: &mut Function, block: Block, kind: InstKind, ty: Ty) -> Value {
    let inst = Inst::from_usize(func.insts.len());
    let value = func.values.push(ValueData { ty, def: Def::Inst(inst) });
    func.insts.push(InstData {
        block,
        kind,
        result: Some(value),
        loc: None,
    });
    func.blocks[block].insts.push(inst);
    value
}

#[test]
fn full_dump_annotates_located_instructions() {
    let sig = Signature::new(vec![], Ty::Void);
    let (mut f, entry) = Function::new("t", sig);
    let located = emit(&mut f, entry, InstKind::Alloca { pointee: Ty::int(32), count: 1 }, Ty::ptr(Ty::int(32)));
    if let Def::Inst(inst) = &f.values[located].def {
        f.insts[*inst].loc = Some(Location { file: 0, line: 4, column: 7 });
    } else {
        panic!("emit produces an instruction");
    }
    let bare = emit(&mut f, entry, InstKind::GetFramePointer, Ty::ptr(Ty::Void));
    if let Def::Inst(inst) = &f.values[bare].def {
        assert_eq!(f.insts[*inst].loc, None);
    } else {
        panic!("emit produces an instruction");
    }
    f.blocks[entry].term = Some(Terminator::Return { val: None });

    let mut module = Module::default();
    module.files.push(File { path: "t.c".to_string() });
    module.push_func(f);

    let plain = dump_module(&module);
    assert!(!plain.contains("t.c"), "{plain}");

    let full = dump_module_full(&module);
    let located_line = full.lines().find(|l| l.contains("alloca")).expect("alloca line");
    assert!(located_line.ends_with("; t.c:4:7"), "{located_line}");
    let bare_line = full.lines().find(|l| l.contains("get.frame.pointer")).expect("frame pointer line");
    assert!(!bare_line.contains(';'), "{bare_line}");
}

#[test]
fn full_dump_annotates_materialized_constants() {
    let sig = Signature::new(vec![Ty::int(32)], Ty::int(32));
    let (mut f, entry) = Function::new("t", sig);
    let base = param(&mut f, entry, Ty::int(32));
    let one = f.push_const(Const::Int { bits: 32, value: 1 });
    let sum = emit(&mut f, entry, InstKind::Add(base, one), Ty::int(32));
    f.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
    // Simulate a pass materializing `one` in place of an earlier def.
    f.origins.insert(
        one,
        ValueOrigin {
            pass: "sccp",
            text: "add %j i32(1)".to_string(),
            loc: Some(Location { file: 0, line: 7, column: 3 }),
        },
    );

    let mut module = Module::default();
    module.files.push(File { path: "t.c".to_string() });
    let one_name = val_name(&f, one);
    let base_name = val_name(&f, base);
    module.push_func(f);

    let plain = dump_module(&module);
    assert!(!plain.contains("<-"), "compact dump gained provenance:\n{plain}");

    let full = dump_module_full(&module);
    let annotation = format!("{one_name} <- sccp: add %j i32(1) @ t.c:7:3");
    assert!(full.contains(&annotation), "missing annotation:\n{full}");
    // The un-origined parameter must not be annotated.
    assert!(!full.contains(&format!("{base_name} <- ")), "{full}");
    // Compact output is still a substring of the full dump.
    for line in plain.lines() {
        assert!(full.contains(line), "compact line missing:\n{line}");
    }
}

#[test]
fn branches_print_targets_and_arguments() {
    let sig = Signature::new(vec![], Ty::Void);
    let (mut f, entry) = Function::new("t", sig);
    let scrut = param(&mut f, entry, Ty::int(32));
    let b1 = f.push_block(vec![]);
    let b2 = f.push_block(vec![]);
    let b3 = f.push_block(vec![]);
    let b4 = f.push_block(vec![Ty::ptr(Ty::int(32))]);
    let b5 = f.push_block(vec![]);
    f.blocks[entry].term = Some(Terminator::Switch {
        scrutinee: scrut,
        cases: vec![(0, b1), (1, b2)],
        default: b3,
    });
    let base = emit(&mut f, b1, InstKind::Alloca { pointee: Ty::int(32), count: 1 }, Ty::ptr(Ty::int(32)));
    let idx = f.push_const(Const::Int { bits: 64, value: 5 });
    let gep = emit(
        &mut f,
        b1,
        InstKind::GetElementPtr {
            base,
            indices: vec![GepIndex::Field(1), GepIndex::ConstIndex(3), GepIndex::Index(idx)].into(),
        },
        Ty::ptr(Ty::int(32)),
    );
    f.blocks[b1].term = Some(Terminator::Br {
        dest: b4,
        args: [gep].into_iter().collect(),
    });
    f.blocks[b2].term = Some(Terminator::Br {
        dest: b4,
        args: [idx].into_iter().collect(),
    });
    let p = f.blocks[b4].params[0];
    let cond = f.push_const(Const::Int { bits: 1, value: 0 });
    f.blocks[b4].term = Some(Terminator::CondBr {
        cond,
        then_block: b5,
        else_block: b3,
        then_args: Default::default(),
        else_args: [p].into_iter().collect(),
    });
    f.blocks[b5].term = Some(Terminator::Return { val: None });
    f.blocks[b3].term = Some(Terminator::Return { val: None });

    let mut text = String::new();
    dump_function(&mut text, &f).unwrap();
    let gep_line = format!("gep {} [.1, 3, {}]", val_name(&f, base), val_name(&f, idx));
    assert!(text.contains(&gep_line), "{text}");
    assert!(text.contains(&format!("br b4({})", val_name(&f, gep))), "{text}");
    assert!(text.contains(&format!("br b4({})", val_name(&f, idx))), "{text}");
    assert!(text.contains(&format!("switch {} [0: b1, 1: b2] default b3", val_name(&f, scrut))), "{text}");
    assert!(text.contains(&format!("brif {} b5 b3({})", val_name(&f, cond), val_name(&f, p))), "{text}");
}

#[test]
fn eh_constructs_dump() {
    use crate::ir::{InvokeCallee, LpadClause};
    let sig = Signature::new(vec![Ty::int(32)], Ty::Void);
    let (mut f, entry) = Function::new("catcher", sig);
    f.personality = Some(crate::ir::Extern::from_usize(7));
    let x = param(&mut f, entry, Ty::int(32));
    let normal = f.push_block(vec![Ty::int(32)]);
    let lpad = f.push_block(vec![Ty::ptr(Ty::Void), Ty::int(64)]);
    let cleanup = f.push_block(vec![]);
    f.blocks[lpad].lpad = Some(smallvec::smallvec![LpadClause::Catch(crate::ir::Global::from_usize(3)), LpadClause::CatchAll]);
    f.blocks[entry].term = Some(Terminator::Invoke {
        callee: InvokeCallee::Extern(crate::ir::Extern::from_usize(2)),
        args: smallvec::smallvec![x],
        sig: Signature::new(vec![Ty::int(32)], Ty::int(32)),
        normal,
        normal_args: Default::default(),
        unwind: lpad,
    });
    let exn = f.blocks[lpad].params[0];
    let sel = f.blocks[lpad].params[1];
    f.blocks[normal].term = Some(Terminator::Return { val: None });
    f.blocks[lpad].term = Some(Terminator::Resume { exn, sel });
    f.blocks[cleanup].term = Some(Terminator::Unreachable);

    let mut module = Module::default();
    module.push_func(f);
    let dump = dump_module(&module);
    assert!(dump.contains("func @catcher personality=e7:"), "{dump}");
    assert!(dump.contains(".lpad [catch g3, catchall]"), "{dump}");
    assert!(dump.contains("invoke.ext e2 sig=(i32) -> i32 (%0) to b1 unwind b2"), "{dump}");
    assert!(dump.contains("resume %2 %3"), "{dump}");
}
