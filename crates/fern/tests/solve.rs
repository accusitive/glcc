//! Solver tests: instance dispatch, environment bounds, higher-kinded
//! heads, and deferral.

use fern::defs::{self, Bound, Defs};
use fern::infer::InferCtx;
use fern::solve::{Obligation, Solution, Solver};
use fern::span::Span;
use fern::ty::Ty;
use fern::{lexer, parser};

fn collect(src: &str) -> Defs {
    let tokens = lexer::lex(src).expect("lex");
    let program = parser::parse(tokens).expect("parse");
    defs::collect(&program).expect("collect")
}

fn span() -> Span {
    Span::new(0, 1)
}

fn goal(defs: &Defs, name: &str, args: Vec<Ty>) -> Obligation {
    Obligation {
        class_id: defs.class_by_name(name).expect("class"),
        args,
    }
}

fn prove(defs: &Defs, env: &[Bound], g: &Obligation) -> Solution {
    Solver::new(defs)
        .prove(&mut InferCtx::new(), env, g, span())
        .expect("prove")
}

fn instance(solution: Solution) -> (usize, Vec<Ty>) {
    match solution {
        Solution::Instance { index, args } => (index, args),
        other => panic!("expected an instance, got {other:?}"),
    }
}

const SUMS: &str = "\
class Sum a { add : a -> a -> a }
instance Sum Int {}
";

#[test]
fn concrete_goal_picks_its_instance() {
    let defs = collect(SUMS);
    let (index, args) = instance(prove(
        &defs,
        &[],
        &goal(&defs, "Sum", vec![Ty::int_ty()]),
    ));
    assert_eq!(index, 0);
    assert!(args.is_empty());
}

#[test]
fn unknown_arguments_defer() {
    let defs = collect(SUMS);
    let mut infer = InferCtx::new();
    let t = infer.fresh();
    let solution = Solver::new(&defs)
        .prove(&mut infer, &[], &goal(&defs, "Sum", vec![t]), span())
        .expect("prove");
    assert!(matches!(solution, Solution::Deferred));
}

#[test]
fn environment_bounds_discharge_goals() {
    let defs = collect(SUMS);
    let cid = defs.class_by_name("Sum").expect("Sum");
    let env = [Bound {
        class_id: cid,
        args: vec![Ty::Param("a".into())],
        span: span(),
    }];
    let solver = Solver::new(&defs);

    let rigid = Obligation {
        class_id: cid,
        args: vec![Ty::Param("a".into())],
    };
    let solution = solver
        .prove(&mut InferCtx::new(), &env, &rigid, span())
        .expect("prove");
    assert!(matches!(solution, Solution::Bound));

    // An unknown argument is bound to the bound's type.
    let mut infer = InferCtx::new();
    let t = infer.fresh();
    let unknown = Obligation {
        class_id: cid,
        args: vec![t.clone()],
    };
    let solution = solver
        .prove(&mut infer, &env, &unknown, span())
        .expect("prove");
    assert!(matches!(solution, Solution::Bound));
    assert_eq!(infer.resolve(&t), Ty::Param("a".into()));
}

#[test]
fn bounds_recurse_through_instances() {
    let defs = collect("\
class Sum a { add : a -> a -> a }
data List a = Nil | Cons(a, List a)
instance Sum Int {}
instance Sum (List a) when Sum a {}
");
    let list = |inner: Ty| Ty::named("List", vec![inner]);

    let (index, args) = instance(prove(
        &defs,
        &[],
        &goal(&defs, "Sum", vec![list(list(Ty::int_ty()))]),
    ));
    assert_eq!(index, 1);
    assert_eq!(args, vec![list(Ty::int_ty())]);

    assert!(matches!(
        prove(&defs, &[], &goal(&defs, "Sum", vec![list(Ty::float_ty())])),
        Solution::Unsatisfied
    ));
}

#[test]
fn hkt_goals_dispatch_through_partial_applications() {
    let defs = collect("\
class Functor f { map : (a -> b) -> f a -> f b }
data List a = Nil | Cons(a, List a)
data Free f a = Pure(a) | Wrap(f a, Free f a)
instance Functor List {}
instance Functor (Free g) when Functor g {}
");
    let cid = defs.class_by_name("Functor").expect("Functor");
    let solver = Solver::new(&defs);

    // `Functor (Free List)`: the inner argument is itself a constructor.
    let free_list = Ty::named("Free", vec![Ty::named("List", vec![])]);
    let (index, args) = instance(
        solver
            .prove(&mut InferCtx::new(), &[], &goal(&defs, "Functor", vec![free_list]), span())
            .expect("prove"),
    );
    assert_eq!(index, 1);
    assert_eq!(args, vec![Ty::named("List", vec![])]);

    // `Functor (Free g)` with a rigid `g` resolves through the bound.
    let rigid = Ty::Param("g".into());
    let env = [Bound {
        class_id: cid,
        args: vec![rigid.clone()],
        span: span(),
    }];
    let g = Obligation {
        class_id: cid,
        args: vec![Ty::named("Free", vec![rigid.clone()])],
    };
    let (index, args) = instance(
        solver
            .prove(&mut InferCtx::new(), &env, &g, span())
            .expect("prove"),
    );
    assert_eq!(index, 1);
    assert_eq!(args, vec![rigid]);
}

#[test]
fn two_matching_instances_are_ambiguous() {
    let defs = collect("\
class Eq a { eq : a -> a -> Bool }
data List a = Nil | Cons(a, List a)
instance Eq (List a) {}
instance Eq (List b) {}
");
    let solution = prove(
        &defs,
        &[],
        &goal(&defs, "Eq", vec![Ty::named("List", vec![Ty::int_ty()])]),
    );
    match solution {
        Solution::Ambiguous { instances } => assert_eq!(instances.len(), 2),
        other => panic!("expected ambiguity, got {other:?}"),
    }
}

#[test]
fn instance_generics_substitute_positionally() {
    let defs = collect("\
class Co a b { co : a -> b }
data Pair a b = Pair(a, b)
instance Co (Pair a b) Int {}
");
    let (_, args) = instance(prove(
        &defs,
        &[],
        &goal(
            &defs,
            "Co",
            vec![
                Ty::named("Pair", vec![Ty::int_ty(), Ty::bool_ty()]),
                Ty::int_ty(),
            ],
        ),
    ));
    assert_eq!(args, vec![Ty::int_ty(), Ty::bool_ty()]);
}
