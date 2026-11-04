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
    let mut program = parser::parse(tokens).expect("parse");
    fern::resolve::qualify(&mut program).expect("resolve");
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
    Solver::new(defs).prove(&mut InferCtx::new(), env, g, span()).expect("prove")
}

fn instance(solution: Solution) -> (usize, Vec<Ty>) {
    match solution {
        Solution::Instance { index, args } => (index, args),
        other => panic!("expected an instance, got {other:?}"),
    }
}

const SUMS: &str = "\
class Sum<A> { add: (A, A) -> A }
instance Sum<Int> {}
";

#[test]
fn concrete_goal_picks_its_instance() {
    let defs = collect(SUMS);
    let (index, args) = instance(prove(&defs, &[], &goal(&defs, "Sum", vec![Ty::int_ty()])));
    assert_eq!(index, 0);
    assert!(args.is_empty());
}

#[test]
fn unknown_arguments_defer() {
    let defs = collect(SUMS);
    let mut infer = InferCtx::new();
    let t = infer.fresh();
    let solution = Solver::new(&defs).prove(&mut infer, &[], &goal(&defs, "Sum", vec![t]), span()).expect("prove");
    assert!(matches!(solution, Solution::Deferred));
}

#[test]
fn environment_bounds_discharge_goals() {
    let defs = collect(SUMS);
    let cid = defs.class_by_name("Sum").expect("Sum");
    let env = [Bound {
        class_id: cid,
        args: vec![Ty::Param("A".into())],
        span: span(),
    }];
    let solver = Solver::new(&defs);

    let rigid = Obligation {
        class_id: cid,
        args: vec![Ty::Param("A".into())],
    };
    let solution = solver.prove(&mut InferCtx::new(), &env, &rigid, span()).expect("prove");
    assert!(matches!(solution, Solution::Bound));

    // An unknown argument is bound to the bound's type.
    let mut infer = InferCtx::new();
    let t = infer.fresh();
    let unknown = Obligation {
        class_id: cid,
        args: vec![t.clone()],
    };
    let solution = solver.prove(&mut infer, &env, &unknown, span()).expect("prove");
    assert!(matches!(solution, Solution::Bound));
    assert_eq!(infer.resolve(&t), Ty::Param("A".into()));
}

#[test]
fn bounds_recurse_through_instances() {
    let defs = collect(
        "\
class Sum<A> { add: (A, A) -> A }
type List<A> = | Nil | Cons(A, List<A>)
instance Sum<Int> {}
instance<A> Sum<List<A>> when Sum<A> {}
",
    );
    let list = |inner: Ty| Ty::named("List", vec![inner]);

    let (index, args) = instance(prove(&defs, &[], &goal(&defs, "Sum", vec![list(list(Ty::int_ty()))])));
    assert_eq!(index, 1);
    assert_eq!(args, vec![list(Ty::int_ty())]);

    assert!(matches!(prove(&defs, &[], &goal(&defs, "Sum", vec![list(Ty::float_ty())])), Solution::Unsatisfied));
}

#[test]
fn hkt_goals_dispatch_through_partial_applications() {
    let defs = collect(
        "\
class Functor<F> { map: for<A, B> (A -> B) -> F<A> -> F<B> }
type List<A> = | Nil | Cons(A, List<A>)
type Free<F, A> = | Pure(A) | Wrap(F<A>, Free<F, A>)
instance Functor<List> {}
instance<G> Functor<Free<G>> when Functor<G> {}
",
    );
    let cid = defs.class_by_name("Functor").expect("Functor");
    let solver = Solver::new(&defs);

    // `Functor (Free List)`: the inner argument is itself a constructor.
    let free_list = Ty::named("Free", vec![Ty::named("List", vec![])]);
    let (index, args) = instance(solver.prove(&mut InferCtx::new(), &[], &goal(&defs, "Functor", vec![free_list]), span()).expect("prove"));
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
    let (index, args) = instance(solver.prove(&mut InferCtx::new(), &env, &g, span()).expect("prove"));
    assert_eq!(index, 1);
    assert_eq!(args, vec![rigid]);
}

#[test]
fn two_matching_instances_are_ambiguous() {
    let defs = collect(
        "\
class Eq<A> { eq: (A, A) -> Bool }
type List<A> = | Nil | Cons(A, List<A>)
instance<A> Eq<List<A>> {}
instance<B> Eq<List<B>> {}
",
    );
    let solution = prove(&defs, &[], &goal(&defs, "Eq", vec![Ty::named("List", vec![Ty::int_ty()])]));
    match solution {
        Solution::Ambiguous { instances } => assert_eq!(instances.len(), 2),
        other => panic!("expected ambiguity, got {other:?}"),
    }
}

#[test]
fn instance_generics_substitute_positionally() {
    let defs = collect(
        "\
class Co<A, B> { co: (A, B) -> B }
type Pair<A, B> = | Pair(A, B)
instance<A, B> Co<Pair<A, B>, Int> {}
",
    );
    let (_, args) = instance(prove(
        &defs,
        &[],
        &goal(&defs, "Co", vec![Ty::named("Pair", vec![Ty::int_ty(), Ty::bool_ty()]), Ty::int_ty()]),
    ));
    assert_eq!(args, vec![Ty::int_ty(), Ty::bool_ty()]);
}
