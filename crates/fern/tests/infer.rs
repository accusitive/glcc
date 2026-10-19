//! Inference engine tests: higher-kinded unification, rigidity, and
//! scheme instantiation.

use fern::infer::InferCtx;
use fern::span::Span;
use fern::ty::Ty;

fn span() -> Span {
    Span::new(0, 1)
}

fn named(name: &str, args: Vec<Ty>) -> Ty {
    Ty::named(name, args)
}

#[test]
fn binds_variables_and_resolves() {
    let mut ctx = InferCtx::new();
    let a = ctx.fresh();
    ctx.unify(&a, &Ty::int_ty(), span()).expect("unify");
    assert_eq!(ctx.resolve(&a), Ty::int_ty());
}

#[test]
fn arrow_mismatch_reports() {
    let mut ctx = InferCtx::new();
    let a = ctx.fresh();
    let b = ctx.fresh();
    let lhs = Ty::arrow(a, b);
    let rhs = Ty::arrow(Ty::int_ty(), Ty::bool_ty());
    ctx.unify(&lhs, &rhs, span()).expect("unify");
    assert_eq!(ctx.resolve(&lhs), Ty::arrow(Ty::int_ty(), Ty::bool_ty()));

    let err = ctx.unify(&Ty::int_ty(), &Ty::bool_ty(), span()).unwrap_err();
    assert!(err.msg.contains("mismatched types"), "{}", err.msg);
}

#[test]
fn partial_application_matches_saturated_constructor() {
    let mut ctx = InferCtx::new();
    let f = ctx.fresh();
    let a = ctx.fresh();
    let g = ctx.fresh();

    // `f a` against `Free g a` binds `f := Free g`.
    let lhs = Ty::app(f.clone(), a.clone());
    let rhs = named("Free", vec![g.clone(), a.clone()]);
    ctx.unify(&lhs, &rhs, span()).expect("unify");
    assert_eq!(ctx.resolve(&f), named("Free", vec![g.clone()]));
    assert!(ctx.unbound_var(&a).is_some());

    // And the other way around.
    let mut ctx = InferCtx::new();
    let f = ctx.fresh();
    let b = ctx.fresh();
    ctx.unify(&named("Free", vec![Ty::int_ty(), b.clone()]), &Ty::app(f.clone(), b), span()).expect("unify");
    assert_eq!(ctx.resolve(&f), named("Free", vec![Ty::int_ty()]));
}

#[test]
fn equal_length_spines_unify_heads() {
    let mut ctx = InferCtx::new();
    let f = ctx.fresh();
    let a = ctx.fresh();
    let b = ctx.fresh();
    let g = ctx.fresh();
    let h = ctx.fresh();

    // `f a b` against `Free g h`: f := Free, a := g, b := h.
    let lhs = Ty::app(Ty::app(f.clone(), a.clone()), b.clone());
    let rhs = named("Free", vec![g.clone(), h.clone()]);
    ctx.unify(&lhs, &rhs, span()).expect("unify");
    assert_eq!(ctx.resolve(&f), named("Free", vec![]));
    assert_eq!(ctx.resolve(&a), ctx.resolve(&g));
    assert_eq!(ctx.resolve(&b), ctx.resolve(&h));
}

#[test]
fn named_types_unify_structurally() {
    let mut ctx = InferCtx::new();
    let a = ctx.fresh();
    ctx.unify(&named("List", vec![a.clone()]), &named("List", vec![Ty::int_ty()]), span()).expect("unify");
    assert_eq!(ctx.resolve(&a), Ty::int_ty());

    let err = ctx.unify(&named("List", vec![a]), &named("Maybe", vec![Ty::int_ty()]), span()).unwrap_err();
    assert!(err.msg.contains("mismatched types"), "{}", err.msg);
}

#[test]
fn rigid_parameters_do_not_bind() {
    let mut ctx = InferCtx::new();
    let rigid = Ty::Param("a".to_string());
    let err = ctx.unify(&rigid, &Ty::int_ty(), span()).unwrap_err();
    assert!(err.msg.contains("mismatched types"), "{}", err.msg);

    let v = ctx.fresh();
    ctx.unify(&v, &rigid, span()).expect("flexible var takes rigid type");
    assert_eq!(ctx.resolve(&v), rigid);
    assert!(ctx.unify(&v, &Ty::int_ty(), span()).is_err());

    let other = Ty::Param("b".to_string());
    assert!(ctx.unify(&other, &rigid, span()).is_err());
    assert!(ctx.unify(&other, &other, span()).is_ok());
}

#[test]
fn occurs_check_rejects_cycles() {
    let mut ctx = InferCtx::new();
    let a = ctx.fresh();
    let err = ctx.unify(&a, &named("List", vec![a.clone()]), span()).unwrap_err();
    assert!(err.msg.contains("recursive type"), "{}", err.msg);
}

#[test]
fn rollback_restores_state() {
    let mut ctx = InferCtx::new();
    let a = ctx.fresh();
    ctx.unify(&a, &Ty::int_ty(), span()).expect("unify");

    let mark = ctx.mark();
    let b = ctx.fresh();
    ctx.unify(&b, &Ty::bool_ty(), span()).expect("bind b");
    ctx.rollback(mark);

    assert_eq!(ctx.resolve(&a), Ty::int_ty());
    // The variable the probe created is gone; the next fresh one takes
    // its id.
    let c = ctx.fresh();
    assert_eq!(c, b);
}

#[test]
fn levels_keep_outer_variables_free() {
    let mut ctx = InferCtx::new();
    let outer = ctx.fresh();
    ctx.enter_level();
    let inner = ctx.fresh();
    ctx.unify(&inner, &outer, span()).expect("connect");
    let local = ctx.fresh();
    let local_ty = Ty::Tuple(vec![local.clone()]);
    let scheme = ctx.generalize(&local_ty);
    ctx.leave_level();

    assert_eq!(scheme.vars.len(), 1);
    assert_eq!(scheme.vars[0].0, format!("t${}", 2));
    assert!(ctx.unbound_var(&outer).is_some());
    assert!(scheme.ty.contains_param());
}

#[test]
fn generalize_then_instantiate() {
    let mut ctx = InferCtx::new();
    let a = ctx.fresh();
    let ty = Ty::arrow(a.clone(), a.clone());
    let scheme = ctx.generalize(&ty);
    assert_eq!(scheme.vars.len(), 1);
    assert!(!scheme.ty.contains_var());
    assert!(scheme.ty.contains_param());

    let inst = ctx.instantiate_scheme(&scheme);
    ctx.unify(&inst, &Ty::arrow(Ty::int_ty(), Ty::int_ty()), span()).expect("same-type instantiation");
    let inst2 = ctx.instantiate_scheme(&scheme);
    ctx.unify(&inst2, &Ty::arrow(Ty::bool_ty(), Ty::bool_ty()), span())
        .expect("second instantiation is independent");
    let inst3 = ctx.instantiate_scheme(&scheme);
    assert!(ctx.unify(&inst3, &Ty::arrow(Ty::int_ty(), Ty::bool_ty()), span()).is_err());
}
