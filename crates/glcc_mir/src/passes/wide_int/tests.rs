use super::*;
use crate::ir::Module;
use crate::ty::Signature;

fn param(func: &mut Function, block: Block, ty: Ty, index: u32) -> Value {
    let value = func.values.push(ValueData {
        ty,
        def: Def::Param(block, index),
    });
    func.blocks[block].params.push(value);
    value
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

fn ty_has_legalizable_int(ty: &Ty) -> bool {
    match ty {
        Ty::Int { bits } => *bits > MAX_SCALAR_INT_BITS,
        Ty::Ptr { pointee, .. } => ty_has_legalizable_int(pointee),
        Ty::Array { elem, .. } => ty_has_legalizable_int(elem),
        Ty::Struct { fields, .. } => fields.iter().any(ty_has_legalizable_int),
        _ => false,
    }
}

fn module_with_binop(divide: bool) -> Module {
    let mut module = Module::new();
    let sig = Signature::new(vec![Ty::int(256), Ty::int(256)], Ty::int(256));
    let (mut func, entry) = Function::new("binop", sig);
    let a = param(&mut func, entry, Ty::int(256), 0);
    let b = param(&mut func, entry, Ty::int(256), 1);
    let kind = if divide { InstKind::Udiv(a, b) } else { InstKind::Add(a, b) };
    let value = emit(&mut func, entry, kind, Ty::int(256));
    func.blocks[entry].term = Some(Terminator::Return { val: Some(value) });
    module.push_func(func);
    module
}

#[test]
fn rewrites_wide_addition_into_limbs() {
    let mut module = module_with_binop(false);
    assert!(LegalizeWideInts::new().run(&mut module));
    assert!(crate::verify::verify_module(&module).is_empty());

    let func = &module.funcs[0];
    assert_eq!(func.sig.ret, Ty::WideInt { bits: 256 });
    assert!(func.values.iter().all(|v| !ty_has_legalizable_int(&v.ty)));
    assert!(func.insts.iter().any(|i| matches!(&i.kind, InstKind::InsertValue { .. })));
    assert!(func.insts.iter().any(|i| matches!(&i.kind, InstKind::Icmp { pred: IcmpPred::Ult, .. })));
    assert!(!func.insts.iter().any(|i| matches!(&i.kind, InstKind::Add(..)) && i.result.is_some_and(|r| func.values[r].ty == Ty::int(256))));
}

#[test]
fn synthesizes_one_division_helper_per_operation() {
    let mut module = module_with_binop(true);
    assert!(LegalizeWideInts::new().run(&mut module));
    assert!(crate::verify::verify_module(&module).is_empty());
    assert!(module.funcs.iter().any(|f| f.name == "__glcc_wide_udiv$256"));
    assert!(!module.funcs.iter().any(|f| f.name == "__glcc_wide_urem$256"));
}

#[test]
fn optimize_at_o0_still_legalizes() {
    let mut module = module_with_binop(false);
    let report = crate::optimize::optimize(&mut module, crate::optimize::OptLevel::None);
    assert!(report.iter().any(|(name, changed)| *name == "legalize-wide-ints" && *changed));
    assert!(crate::verify::verify_module(&module).is_empty());
    assert!(module.funcs[0].values.iter().all(|v| !ty_has_legalizable_int(&v.ty)));
}
