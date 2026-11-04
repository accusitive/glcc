//! Checker tests: inference, generalization, class dispatch, deferred
//! obligations, closures, and error cases.

use fern::check::check;
use fern::defs::{self, Defs};
use fern::hir::{FnRef, HirExpr, HirExprKind, HirFn, HirProgram};
use fern::span::Diag;
use fern::ty::Ty;
use fern::{lexer, parser};

fn check_src(src: &str) -> Result<(Defs, HirProgram), Diag> {
    let tokens = lexer::lex(src).expect("lex");
    let mut program = parser::parse(tokens).expect("parse");
    fern::resolve::qualify(&mut program)?;
    let mut defs = defs::collect(&program).expect("collect");
    let hir = check(&mut defs, &program)?;
    Ok((defs, hir))
}

fn check_ok(src: &str) -> (Defs, HirProgram) {
    check_src(src).expect("check")
}

fn check_err(src: &str) -> String {
    match check_src(src) {
        Ok(_) => panic!("expected a check error"),
        Err(diag) => diag.msg,
    }
}

fn body<'a>(hir: &'a HirProgram, name: &str) -> &'a HirFn {
    hir.bodies.iter().find(|b| b.name == name).unwrap_or_else(|| panic!("no body named `{name}`"))
}

fn is_generic_name(name: &str) -> bool {
    name.starts_with("t$")
}

fn refs(e: &HirExpr, out: &mut Vec<FnRef>) {
    match &e.kind {
        HirExprKind::Const(_) | HirExprKind::Local(_) | HirExprKind::Closure { .. } => {}
        HirExprKind::Fn(target) => out.push(target.clone()),
        HirExprKind::CallDirect { target, args } => {
            out.push(target.clone());
            for a in args {
                refs(a, out);
            }
        }
        HirExprKind::Call { callee, args } => {
            refs(callee, out);
            for a in args {
                refs(a, out);
            }
        }
        HirExprKind::Ctor { args, .. } | HirExprKind::Tuple(args) => {
            for a in args {
                refs(a, out);
            }
        }
        HirExprKind::Field { base, .. } => refs(base, out),
        HirExprKind::Try { value, .. } => refs(value, out),
        HirExprKind::Await(value) => refs(value, out),
        HirExprKind::Let { binds, body } => {
            for (_, value) in binds {
                refs(value, out);
            }
            refs(body, out);
        }
        HirExprKind::If { cond, then_body, else_body } => {
            refs(cond, out);
            refs(then_body, out);
            refs(else_body, out);
        }
        HirExprKind::Match { scrutinee, arms } => {
            refs(scrutinee, out);
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    refs(guard, out);
                }
                refs(&arm.body, out);
            }
        }
        HirExprKind::Bin { lhs, rhs, .. } => {
            refs(lhs, out);
            refs(rhs, out);
        }
        HirExprKind::Un { operand, .. } => refs(operand, out),
    }
}

const LISTS: &str = "\
type List<A> = | Nil | Cons(A, List<A>)
class Functor<F> { map: for<A, B> (A -> B) -> F<A> -> F<B> }
instance Functor<List> {
    map(f, xs) = match xs {
        case Nil => Nil
        case Cons(y, ys) => Cons(f(y), map(f, ys))
    }
}
";

#[test]
fn identity_is_generalized() {
    let (defs, _) = check_ok("fn id(x) : for<A> A -> A = x");
    let scheme = defs.fn_by_name("id").and_then(|id| defs.fns[id.index()].scheme.as_ref()).expect("id has a scheme");
    assert_eq!(scheme.vars.len(), 1);
    assert_eq!(scheme.vars[0].0, "A");
    assert!(!is_generic_name(&scheme.vars[0].0));
}

#[test]
fn one_scheme_many_instantiations() {
    check_ok("type Pair<A, B> = | Pair(A, B)\nfn id(x) : for<A> A -> A = x\nfn p() : -> Pair<Int, Bool> = Pair(id(1), id(true))");
}

#[test]
fn higher_kinded_dispatch_defers_for_a_type_variable() {
    let (_, hir) = check_ok(format!("{LISTS}fn inc_all(xs) : for<F> F<Int> -> F<Int> = map(|x| x + 1, xs)").as_str());
    let inc_all = body(&hir, "inc_all");
    assert_eq!(inc_all.generics.len(), 1, "the constructor stays a parameter");
    let mut found = Vec::new();
    refs(&inc_all.body, &mut found);
    assert!(
        found.iter().any(|r| matches!(r, FnRef::Class { .. })),
        "the Functor goal is deferred to lowering: {found:?}"
    );
}

#[test]
fn higher_kinded_dispatch_resolves_on_a_concrete_type() {
    let (_, hir) = check_ok(format!("{LISTS}fn one() : -> List<Int> = map(|x| x + 1, Cons(1, Nil))").as_str());
    let one = body(&hir, "one");
    let mut found = Vec::new();
    refs(&one.body, &mut found);
    assert!(
        found.iter().any(|r| matches!(r, FnRef::Instance { instance: 0, method: 0, .. })),
        "the List instance is chosen: {found:?}"
    );
}

#[test]
fn instance_bounds_discharge_goals() {
    let src = "\
class Sum<A> { addp: (A, A) -> A }
instance Sum<Int> { addp(x, y) = x + y }
type Box<A> = | Box(A)
instance<A> Sum<Box<A>> when Sum<A> { addp(Box(x), Box(y)) = Box(addp(x, y)) }
";
    let (_, hir) = check_ok(src);
    let method = hir
        .bodies
        .iter()
        .find(|b| matches!(b.key, fern::hir::BodyKey::Method { instance: 1, method: 0 }))
        .expect("the Box instance method");
    let mut found = Vec::new();
    refs(&method.body, &mut found);
    assert!(
        found.iter().any(|r| matches!(
            r,
            FnRef::Class { args, .. } if args.first() == Some(&Ty::Param("A".into()))
        )),
        "the bound is kept for lowering: {found:?}"
    );
}

#[test]
fn instances_are_checked_in_item_order() {
    let src = "\
class Sum<A> { addp: (A, A) -> A }
instance Sum<Int> { addp(x, y) = x + y }
fn sum2() : -> Int = addp(1, 2)
";
    let (_, hir) = check_ok(src);
    let sum2 = body(&hir, "sum2");
    let mut found = Vec::new();
    refs(&sum2.body, &mut found);
    assert!(
        found.iter().any(|r| matches!(
            r,
            FnRef::Instance { method: 0, args, .. } if args.is_empty()
        )),
        "a concrete goal calls the instance directly: {found:?}"
    );
}

#[test]
fn class_sel_works_at_top_level() {
    check_ok(format!("{LISTS}fn two() : -> List<Int> = Functor::map(|x| x + 2, Cons(1, Nil))").as_str());
}

#[test]
fn self_recursion_references_its_own_generics() {
    let (_, hir) = check_ok("fn dup(b, x) : for<A> (Bool, A) -> A = if b then x else dup(false, x)");
    let dup_body = body(&hir, "dup");
    assert_eq!(dup_body.generics.len(), 1);
    let mut found = Vec::new();
    refs(&dup_body.body, &mut found);
    assert!(
        found.iter().any(|r| matches!(r, FnRef::Fn { args, .. } if args.len() == 1)),
        "the self call passes the body's own generics: {found:?}"
    );
}

#[test]
fn strictly_typed_recursion() {
    check_ok("fn fact(n) : Int -> Int = if n == 0 then 1 else n * fact(n - 1)");
}

#[test]
fn closures_capture_enclosing_locals() {
    let (_, hir) = check_ok("fn adder(n) : Int -> Int -> Int = |x| x + n");
    assert_eq!(hir.closures.len(), 1);
    assert_eq!(hir.closures[0].captures.len(), 1);
    assert_eq!(hir.closures[0].captures[0].outer, 0);
    assert_eq!(hir.closures[0].captures[0].ty, Ty::int_ty());
    let adder = body(&hir, "adder");
    match &adder.body.kind {
        HirExprKind::Closure { id, captures } => {
            assert_eq!(*id, 0);
            assert_eq!(captures.len(), 1);
        }
        other => panic!("expected a closure, got {other:?}"),
    }
}

#[test]
fn nested_closures_capture_through_the_chain() {
    let (_, hir) = check_ok("fn f(a) : Int -> Int -> Int -> Int = |y| |z| a + y + z");
    assert_eq!(hir.closures.len(), 2);
    // Lambdas are recorded as they complete, so the inner one is first.
    let inner = &hir.closures[0];
    let outer = &hir.closures[1];
    assert_eq!(outer.captures.len(), 1);
    assert_eq!(inner.captures.len(), 2);
    assert_eq!(inner.captures[0].outer, outer.captures[0].inner);
}

#[test]
fn local_lets_are_monomorphic_and_ordered() {
    check_ok("fn f(x) : Int -> Int = { let y = x + 1; y * 2 }");
}

#[test]
fn matcher_unifies_arm_types() {
    check_ok(
        "\
type Maybe<A> = | Nothing | Just(A)
fn or_zero(m) : Maybe<Int> -> Int = match m { case Nothing => 0 case Just(x) => x }
",
    );
}

#[test]
fn list_sugar_round_trips() {
    let defs_src = "type List<A> = | Nil | Cons(A, List<A>)\n";
    check_ok(format!("{defs_src}fn xs() : -> List<Int> = [1, 2, 3]\nfn empty() : for<A> -> List<A> = []").as_str());
    check_ok(format!("{defs_src}fn head2(xs) : List<Int> -> Int = match xs {{ case [a, b] => a case _ => 0 }}").as_str());
    check_ok(format!("{defs_src}fn conses(xs) : List<Int> -> Int = match xs {{ case a :: rest => a case _ => 0 }}").as_str());
}

#[test]
fn integer_and_bool_patterns_work() {
    check_ok("fn sign(n) : Int -> Int = match n { case 0 => 0 case _ => 1 }");
    check_ok("fn flip(b) : Bool -> Bool = match b { case true => false case false => true }");
}

#[test]
fn forward_references_are_allowed() {
    let (_, hir) = check_ok("fn f(x) : Int -> Int = g(x)\nfn g(x) : Int -> Int = x");
    assert!(hir.bodies.iter().any(|body| body.name == "f"));
    assert!(hir.bodies.iter().any(|body| body.name == "g"));
}

#[test]
fn mutual_recursion_uses_declared_signatures() {
    check_ok(
        "fn even(n) : Int -> Bool = if n == 0 then true else odd(n - 1)\n\
         fn odd(n) : Int -> Bool = if n == 0 then false else even(n - 1)",
    );
}

#[test]
fn a_signature_cannot_be_specialized_by_its_body() {
    let msg = check_err("fn bad(x) : for<A> A -> A = 1");
    assert!(msg.contains("mismatched types"), "{msg}");
}

#[test]
fn annotated_values_are_checked_against_their_type() {
    check_ok("let n : Int = 1");
    let msg = check_err("let n : Int = true");
    assert!(msg.contains("mismatched types"), "{msg}");
}

#[test]
fn an_annotated_value_can_be_used_before_its_declaration() {
    check_ok("fn f() : -> Int = n\nlet n : Int = 1");
}

#[test]
fn error_unknown_name() {
    let msg = check_err("fn f(x) : Int -> Int = zz(x)");
    assert!(msg.contains("unknown value `zz`"), "{msg}");
}

#[test]
fn error_arity_mismatch() {
    let msg = check_err("fn id(x) : for<A> A -> A = x\nfn f() : -> Int = id()");
    assert!(msg.contains("takes 1 parameter(s), but 0 argument(s)"), "{msg}");
}

#[test]
fn closure_returning_function_is_called_by_declared_arity() {
    check_ok("fn mk(x) : Int -> Int -> Int = |y| x + y\nfn f() : -> Int = mk(1)(2)");
    check_ok("fn mk(x) : Int -> Int -> Int = |y| x + y\nfn f() : -> Int = { let g = mk(1); g(2) }");
}

#[test]
fn error_partial_application() {
    let msg = check_err("fn add(x, y) : (Int, Int) -> Int = x + y\nfn f() : -> Int = add(1)(2)");
    assert!(msg.contains("takes 2 parameter(s), but 1 argument(s)"), "{msg}");
}

#[test]
fn error_duplicate_pattern_bind() {
    let msg = check_err("fn f(x, x) : (Int, Int) -> Int = x");
    assert!(msg.contains("bound more than once"), "{msg}");
}

#[test]
fn error_operator_operand() {
    let msg = check_err("fn f(x) : Bool -> Int = x + true");
    assert!(msg.contains("expected a numeric type"), "{msg}");
    check_ok("fn lt(x) : Int -> Bool = x < x");
    check_ok("fn lt_f(x) : Float -> Bool = x < 1.5");
    check_ok("fn add_f(x) : Float -> Float = x + 2.5");
    check_ok("fn eq_f(x) : Float -> Bool = x == 1.5");
}

#[test]
fn error_ambiguous_class_method() {
    let src = "\
class Sum<A> { addp: (A, A) -> A }
class Zero<A> { zero: A }
instance Sum<Int> { addp(x, y) = x + y }
instance Zero<Int> { zero() = 0 }
fn z() : -> Int = Zero::zero()
";
    check_ok(src);
    let src = "\
class A<T> { shrunk: T -> T }
class B<T> { shrunk: T -> T }
fn z(x) : for<T> T -> T = shrunk(x)
";
    let msg = check_err(src);
    assert!(msg.contains("ambiguous"), "{msg}");
}

#[test]
fn error_unusable_unknown() {
    let msg = check_err("type List<A> = | Nil | Cons(A, List<A>)\nfn f(n) : Int -> Int = { let empty = Nil; n }");
    assert!(msg.contains("cannot infer"), "{msg}");
}

#[test]
fn try_requires_a_matching_return_type() {
    let msg = check_err(
        "type Maybe<A> = | Nothing | Just(A)\n\
         fn f(m) : Maybe<Int> -> Int = { let x = m?; x }",
    );
    assert!(msg.contains("requires the enclosing function to return `Maybe`"), "{msg}");

    let msg = check_err(
        "type Maybe<A> = | Nothing | Just(A)\n\
         fn f(x) : Int -> Int = { let y = x?; y }",
    );
    assert!(msg.contains("`?` requires a `Maybe` or `Result`"), "{msg}");
}

#[test]
fn record_fields_are_checked_by_name() {
    check_ok("type P = { x: Int, y: Int }\nfn f(p) : P -> Int = p.x + p.y");

    let msg = check_err("type P = { x: Int }\nfn f() : -> P = P { }");
    assert!(msg.contains("missing field `x`"), "{msg}");

    let msg = check_err("type P = { x: Int }\nfn f() : -> P = P { y = 1 }");
    assert!(msg.contains("has no field `y`"), "{msg}");

    let msg = check_err("type P = { x: Int }\nfn f() : -> P = P { x = 1, x = 2 }");
    assert!(msg.contains("given twice"), "{msg}");
}

#[test]
fn generic_top_level_values_generalize() {
    let (defs, _) = check_ok("type List<A> = | Nil | Cons(A, List<A>)\nlet empty = Nil");
    let scheme = defs
        .value_by_name("empty")
        .and_then(|id| defs.values[id.index()].scheme.as_ref())
        .expect("empty has a scheme");
    assert_eq!(scheme.vars.len(), 1);
}

#[test]
fn async_calls_return_futures() {
    let (defs, hir) = check_ok("async fn f(x) : Int -> Int = x\nfn g() : -> Future<Int> = f(1)");
    assert!(body(&hir, "f").is_async);
    let scheme = defs.fn_by_name("f").and_then(|id| defs.fns[id.index()].scheme.as_ref()).expect("f has a scheme");
    let Ty::Arrow(_, ret) = &scheme.ty else {
        panic!("f should be a function");
    };
    assert!(ret.is_future(), "async calls must return Future: {ret:?}");
}

#[test]
fn await_unwraps_a_future() {
    check_ok("async fn f(x) : Int -> Int = x\nasync fn g() : -> Int = f(1).await + 1");
    check_ok("async fn f(x) : Int -> Int = x\nfn g() : -> Future<Int> = f(1)");
}

#[test]
fn await_rejects_non_futures_and_sync_contexts() {
    let msg = check_err("fn f() : -> Int = 1.await");
    assert!(msg.contains("only valid inside an async function"), "{msg}");

    let msg = check_err("async fn f() : -> Int = 1.await");
    assert!(msg.contains("requires a `Future`"), "{msg}");

    let msg = check_err("async fn f() : -> Int = { let g = |x| f().await; 1 }");
    assert!(msg.contains("only valid inside an async function"), "{msg}");
}

#[test]
fn future_is_a_unary_type() {
    check_ok("fn pass(f) : Future<Int> -> Future<Int> = f");
}

#[test]
fn unresolved_goal_over_a_type_parameter_defers() {
    let src = "\
class Sum<A> { addp: (A, A) -> A }
type Box<A> = | Box(A)
instance Sum<Int> { addp(x, y) = x + y }
fn f(b) : for<A> A -> A = addp(b, b)
";
    let (_, hir) = check_ok(src);
    let f = body(&hir, "f");
    let mut found = Vec::new();
    refs(&f.body, &mut found);
    assert!(found.iter().any(|r| matches!(r, FnRef::Class { .. })), "lowering must pick the instance: {found:?}");
}
