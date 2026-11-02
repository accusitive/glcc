use super::*;
use crate::ir::Block;
use crate::ty::{Signature, Ty};

#[test]
fn unterminated_block_fails() {
    let sig = Signature::new(vec![], Ty::Void);
    let (func, _) = Function::new("empty", sig);
    assert!(!verify_func(&func).is_empty());
}

#[test]
fn terminated_block_passes() {
    let sig = Signature::new(vec![], Ty::Void);
    let (mut func, entry) = Function::new("ret", sig);
    func.blocks[entry].term = Some(Terminator::Return { val: None });
    assert!(verify_func(&func).is_empty());
}

const EH_MODULE: &str = r#"
.extern @thrower sig=(i32) -> i32
.extern @__gxx_personality_v0 sig=() -> i32
.dataextern @_ZTIi ty=ptr(void)

.func @catcher sig=(i32) -> void personality=@__gxx_personality_v0
.params %0:i32
b0:
  invoke.ext @thrower sig=(i32) -> i32, i32 %0 to b1 unwind b2
b1(%1:i32):
  ret
b2(%2:ptr(void), %3:i64) .lpad [catch @_ZTIi]:
  %4:i32 = eh.typeid @_ZTIi
  %5:bool = icmp.eq %3, %4
  brif %5 b3 b4
b3():
  ret
b4():
  resume %2, %3
"#;

#[test]
fn eh_module_parses_and_verifies() {
    let module = crate::parse_module(EH_MODULE).expect("EH module parses");
    assert!(verify_module(&module).is_empty(), "{:?}", verify_module(&module));

    let f = &module.funcs[module.funcs.len() - 1];
    assert!(f.personality.is_some());
    let lpad = &f.blocks[Block::from_usize(2)];
    assert_eq!(lpad.lpad.as_ref().map(|c| c.len()), Some(1));
    assert_eq!(f.blocks[Block::from_usize(0)].params.len(), 1);
    match &f.blocks[Block::from_usize(0)].term {
        Some(Terminator::Invoke { normal, unwind, .. }) => {
            assert_eq!(normal.index(), 1);
            assert_eq!(unwind.index(), 2);
        }
        other => panic!("expected invoke terminator, got {other:?}"),
    }
    let eh_inst = f.blocks[Block::from_usize(2)].insts[0];
    match &f.insts[eh_inst].kind {
        InstKind::EhTypeId { .. } => {}
        other => panic!("expected eh.typeid, got {other:?}"),
    }
}

#[test]
fn invoke_unwind_target_must_be_lpad() {
    let text = EH_MODULE.replace("unwind b2", "unwind b5").replace(
        "b4():\n  resume %2, %3",
        "b4():\n  resume %2, %3\nb5(%6:ptr(void), %7:i64):\n  ret",
    );
    let err = crate::parse_module(&text).expect_err("verify failure surfaces through parse");
    assert!(err.message.contains("not a landing pad"), "{err:?}");
}

#[test]
fn lpad_needs_personality() {
    let text = EH_MODULE.replace(" personality=@__gxx_personality_v0", "");
    let err = crate::parse_module(&text).expect_err("verify failure surfaces through parse");
    assert!(err.message.contains("no personality"), "{err:?}");
}

#[test]
fn lpad_rejects_normal_edge() {
    let text = EH_MODULE.replace("brif %5 b3 b4", "brif %5 b2(%2, %3) b4");
    let err = crate::parse_module(&text).expect_err("verify failure surfaces through parse");
    assert!(err.message.contains("targeted by a normal edge"), "{err:?}");
}

#[test]
fn lpad_takes_exactly_two_params() {
    use crate::ir::{Def, LpadClause, ValueData};
    let sig = Signature::new(vec![], Ty::Void);
    let (mut func, entry) = Function::new("f", sig);
    func.personality = Some(crate::ir::Extern::from_usize(0));
    func.blocks[entry].term = Some(Terminator::Return { val: None });
    let lpad = func.push_block(vec![Ty::ptr(Ty::Void)]);
    func.blocks[lpad].lpad = Some(smallvec::smallvec![LpadClause::CatchAll]);
    let exn = func.values.push(ValueData { ty: Ty::ptr(Ty::Void), def: Def::Param(lpad, 0) });
    let sel = func.values.push(ValueData { ty: Ty::int(64), def: Def::Param(lpad, 1) });
    func.blocks[lpad].term = Some(Terminator::Resume { exn, sel });
    let errors = verify_func(&func);
    assert!(errors.iter().any(|e| e.contains("two payload values")), "{errors:?}");
}

#[test]
fn eh_typeid_requires_matching_clause() {
    let text = EH_MODULE
        .replace(".dataextern @_ZTIi ty=ptr(void)", ".dataextern @_ZTIi ty=ptr(void)\n.dataextern @_ZTIx ty=ptr(void)")
        .replace("%4:i32 = eh.typeid @_ZTIi", "%4:i32 = eh.typeid @_ZTIx");
    let err = crate::parse_module(&text).expect_err("verify failure surfaces through parse");
    assert!(err.message.contains("absent from every landing pad"), "{err:?}");
}
