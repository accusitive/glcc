use super::*;
use crate::ir::{Const, Def, Function, Inst, Terminator, ValueData};
use crate::ty::{Signature, Ty};
use crate::verify::verify_func;

fn push_value(func: &mut Function, block: Block, kind: InstKind, ty: Ty) -> Value {
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

fn iconst(func: &mut Function, bits: u32, value: i128) -> Value {
    func.push_const(Const::Int { bits, value })
}

fn int_to_ptr(func: &mut Function, block: Block, value: i128, ty: Ty) -> Value {
    let c = iconst(func, 64, value);
    push_value(func, block, InstKind::IntToPtr(c), ty)
}

fn push_call(func: &mut Function, block: Block, target: crate::ir::Extern, args: &[Value], ty: Ty) -> (Value, Inst) {
    let result = func.values.push(ValueData {
        ty,
        def: Def::Inst(Inst::from_usize(func.insts.len())),
    });
    let call = func.insts.push(InstData {
        block,
        kind: InstKind::CallExtern {
            target,
            args: args.iter().copied().collect(),
        },
        result: Some(result),
        loc: None,
    });
    func.values[result].def = Def::Inst(call);
    func.blocks[block].insts.push(call);
    (result, call)
}

fn module_with(extern_name: &str) -> (Module, crate::ir::Func, Block, crate::ir::Extern) {
    let mut module = Module::new();
    let sig = Signature::new(vec![Ty::ptr(Ty::int(8)), Ty::ptr(Ty::int(8)), Ty::int(64)], Ty::ptr(Ty::int(8)));
    let ext = module.push_extern(extern_name, sig);
    let (mut func, entry) = Function::new("caller", Signature::new(vec![Ty::int(64)], Ty::Void));
    let len = func.values.push(ValueData {
        ty: Ty::int(64),
        def: Def::Param(entry, 0),
    });
    func.blocks[entry].params.push(len);
    func.blocks[entry].term = Some(Terminator::Return { val: None });
    let id = module.funcs.push(func);
    (module, id, entry, ext)
}

#[test]
fn lowers_constant_memcpy() {
    let (mut module, func_id, entry, ext) = module_with("memcpy");
    let ptr_ty = Ty::ptr(Ty::int(8));
    let (result, call) = {
        let f = &mut module.funcs[func_id];
        let dst = int_to_ptr(f, entry, 1, ptr_ty.clone());
        let src = int_to_ptr(f, entry, 2, ptr_ty.clone());
        let len = iconst(f, 64, 8);
        push_call(f, entry, ext, &[dst, src, len], ptr_ty)
    };

    assert!(LowerLibcMemory.run(&mut module));
    let f = &module.funcs[func_id];
    assert!(!f.blocks[entry].insts.contains(&call));
    let kinds: Vec<_> = f.blocks.iter().flat_map(|block| block.insts.iter()).map(|inst| &f.insts[*inst].kind).collect();
    assert!(kinds.iter().any(|kind| matches!(kind, InstKind::Memcpy { len: 8, .. })));
    assert!(!kinds.iter().any(|kind| matches!(kind, InstKind::CallExtern { .. })));
    assert_eq!(f.use_count(result), 0, "the call result is replaced by dst");
    assert!(verify_func(f).is_empty());
}

#[test]
fn keeps_non_constant_memcpy() {
    let (mut module, func_id, entry, ext) = module_with("memcpy");
    let ptr_ty = Ty::ptr(Ty::int(8));
    let (_, call) = {
        let f = &mut module.funcs[func_id];
        let dst = int_to_ptr(f, entry, 1, ptr_ty.clone());
        let src = int_to_ptr(f, entry, 2, ptr_ty.clone());
        let len = f.blocks[entry].params[0];
        push_call(f, entry, ext, &[dst, src, len], ptr_ty)
    };

    assert!(!LowerLibcMemory.run(&mut module));
    assert!(matches!(module.funcs[func_id].insts[call].kind, InstKind::CallExtern { .. }));
}

#[test]
fn lowers_constant_memset() {
    let (mut module, func_id, entry, ext) = module_with("memset");
    let ptr_ty = Ty::ptr(Ty::int(8));
    {
        let f = &mut module.funcs[func_id];
        let dst = int_to_ptr(f, entry, 1, ptr_ty.clone());
        let byte = iconst(f, 32, 0xab);
        let len = iconst(f, 64, 16);
        push_call(f, entry, ext, &[dst, byte, len], ptr_ty);
    }

    assert!(LowerLibcMemory.run(&mut module));
    let f = &module.funcs[func_id];
    let kinds: Vec<_> = f.blocks.iter().flat_map(|block| block.insts.iter()).map(|inst| &f.insts[*inst].kind).collect();
    assert!(kinds.iter().any(|kind| matches!(kind, InstKind::Memset { byte: 0xab, len: 16, .. })));
    assert!(!kinds.iter().any(|kind| matches!(kind, InstKind::CallExtern { .. })));
    assert!(verify_func(f).is_empty());
}
