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
    let program = parser::parse(tokens).expect("parse");
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
data List a = Nil | Cons(a, List a)
class Functor f { map : (a -> b) -> f a -> f b }
instance Functor List {
    map(f, xs) = match xs {
        case Nil => Nil
        case Cons(y, ys) => Cons(f(y), map(f, ys))
    }
}
";

#[test]
fn identity_is_generalized() {
    let (defs, _) = check_ok("fun id(x : a) : a = x");
    let scheme = defs.fn_by_name("id").and_then(|id| defs.fns[id.index()].scheme.as_ref()).expect("id has a scheme");
    assert_eq!(scheme.vars.len(), 1);
    assert_eq!(scheme.vars[0].0, "a");
    assert!(!is_generic_name(&scheme.vars[0].0));
}

#[test]
fn one_scheme_many_instantiations() {
    check_ok("data Pair a b = Pair(a, b)\nfun id(x : a) : a = x\nfun p() : Pair Int Bool = Pair(id(1), id(true))");
}

#[test]
fn higher_kinded_dispatch_defers_for_a_type_variable() {
    let (_, hir) = check_ok(format!("{LISTS}fun inc_all(xs : f Int) : f Int = map(x -> x + 1, xs)").as_str());
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
    let (_, hir) = check_ok(format!("{LISTS}fun one() : List Int = map(x -> x + 1, Cons(1, Nil))").as_str());
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
class Sum a { addp : a -> a -> a }
instance Sum Int { addp(x, y) = x + y }
data Box a = Box(a)
instance Sum (Box a) when Sum a { addp(Box(x), Box(y)) = Box(addp(x, y)) }
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
            FnRef::Class { args, .. } if args.first() == Some(&Ty::Param("a".into()))
        )),
        "the bound is kept for lowering: {found:?}"
    );
}

#[test]
fn instances_are_checked_in_item_order() {
    let src = "\
class Sum a { addp : a -> a -> a }
instance Sum Int { addp(x, y) = x + y }
fun sum2() : Int = addp(1, 2)
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
    check_ok(format!("{LISTS}fun two() : List Int = Functor::map(x -> x + 2, Cons(1, Nil))").as_str());
}

#[test]
fn self_recursion_references_its_own_generics() {
    let (_, hir) = check_ok("fun dup(b : Bool, x : a) : a = if b then x else dup(false, x)");
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
    check_ok("fun fact(n : Int) : Int = if n == 0 then 1 else n * fact(n - 1)");
}

#[test]
fn closures_capture_enclosing_locals() {
    let (_, hir) = check_ok("fun adder(n : Int) : Int -> Int = x -> x + n");
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
    let (_, hir) = check_ok("fun f(a : Int) : Int -> Int -> Int = y -> z -> a + y + z");
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
    check_ok("fun f(x : Int) : Int = { let y = x + 1; y * 2 }");
}

#[test]
fn matcher_unifies_arm_types() {
    check_ok(
        "\
data Maybe a = Nothing | Just(a)
fun or_zero(m : Maybe Int) : Int = match m { case Nothing => 0 case Just(x) => x }
",
    );
}

#[test]
fn list_sugar_round_trips() {
    let defs_src = "data List a = Nil | Cons(a, List a)\n";
    check_ok(format!("{defs_src}fun xs() : List Int = [1, 2, 3]\nfun empty() : List a = []").as_str());
    check_ok(format!("{defs_src}fun head2(xs : List Int) : Int = match xs {{ case [a, b] => a case _ => 0 }}").as_str());
    check_ok(format!("{defs_src}fun conses(xs : List Int) : Int = match xs {{ case a :: rest => a case _ => 0 }}").as_str());
}

#[test]
fn integer_and_bool_patterns_work() {
    check_ok("fun sign(n : Int) : Int = match n { case 0 => 0 case _ => 1 }");
    check_ok("fun flip(b : Bool) : Bool = match b { case true => false case false => true }");
}

#[test]
fn forward_references_are_allowed() {
    let (_, hir) = check_ok("fun f(x : Int) : Int = g(x)\nfun g(x : Int) : Int = x");
    assert!(hir.bodies.iter().any(|body| body.name == "f"));
    assert!(hir.bodies.iter().any(|body| body.name == "g"));
}

#[test]
fn mutual_recursion_uses_declared_signatures() {
    check_ok(
        "fun even(n : Int) : Bool = if n == 0 then true else odd(n - 1)\n\
         fun odd(n : Int) : Bool = if n == 0 then false else even(n - 1)",
    );
}

#[test]
fn a_signature_cannot_be_specialized_by_its_body() {
    let msg = check_err("fun bad(x : a) : a = 1");
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
    check_ok("fun f() : Int = n\nlet n : Int = 1");
}

#[test]
fn error_unknown_name() {
    let msg = check_err("fun f(x : Int) : Int = zz(x)");
    assert!(msg.contains("cannot find `zz`"), "{msg}");
}

#[test]
fn error_arity_mismatch() {
    let msg = check_err("fun id(x : a) : a = x\nfun f() : Int = id()");
    assert!(msg.contains("takes 1 parameter(s), but 0 argument(s)"), "{msg}");
}

#[test]
fn closure_returning_function_is_called_by_declared_arity() {
    check_ok("fun mk(x : Int) : Int -> Int = y -> x + y\nfun f() : Int = mk(1)(2)");
    check_ok("fun mk(x : Int) : Int -> Int = y -> x + y\nfun f() : Int = { let g = mk(1); g(2) }");
}

#[test]
fn error_partial_application() {
    let msg = check_err("fun add(x : Int, y : Int) : Int = x + y\nfun f() : Int = add(1)(2)");
    assert!(msg.contains("takes 2 parameter(s), but 1 argument(s)"), "{msg}");
}

#[test]
fn error_duplicate_pattern_bind() {
    let msg = check_err("fun f(x : Int, x : Int) : Int = x");
    assert!(msg.contains("bound more than once"), "{msg}");
}

#[test]
fn error_operator_operand() {
    let msg = check_err("fun f(x : Bool) : Int = x + true");
    assert!(msg.contains("expected a numeric type"), "{msg}");
    check_ok("fun lt(x : Int) : Bool = x < x");
    check_ok("fun lt_f(x : Float) : Bool = x < 1.5");
    check_ok("fun add_f(x : Float) : Float = x + 2.5");
    check_ok("fun eq_f(x : Float) : Bool = x == 1.5");
}

#[test]
fn error_ambiguous_class_method() {
    let src = "\
class Sum a { addp : a -> a -> a }
class Zero a { zero : a }
instance Sum Int { addp(x, y) = x + y }
instance Zero Int { zero() = 0 }
fun z() : Int = Zero::zero()
";
    check_ok(src);
    let src = "\
class A a { shrunk : a -> a }
class B a { shrunk : a -> a }
fun z(x : a) : a = shrunk(x)
";
    let msg = check_err(src);
    assert!(msg.contains("ambiguous"), "{msg}");
}

#[test]
fn error_unusable_unknown() {
    let msg = check_err("data List a = Nil | Cons(a, List a)\nfun f(n : Int) : Int = { let empty = Nil; n }");
    assert!(msg.contains("cannot infer"), "{msg}");
}

#[test]
fn try_requires_a_matching_return_type() {
    let msg = check_err(
        "data Maybe a = Nothing | Just(a)\n\
         fun f(m : Maybe Int) : Int = { let x = m?; x }",
    );
    assert!(msg.contains("requires the enclosing function to return `Maybe`"), "{msg}");

    let msg = check_err(
        "data Maybe a = Nothing | Just(a)\n\
         fun f(x : Int) : Int = { let y = x?; y }",
    );
    assert!(msg.contains("`?` requires a `Maybe` or `Result`"), "{msg}");
}

#[test]
fn record_fields_are_checked_by_name() {
    check_ok("data P = P { x : Int, y : Int }\nfun f(p : P) : Int = p.x + p.y");

    let msg = check_err("data P = P { x : Int }\nfun f() : P = P { }");
    assert!(msg.contains("missing field `x`"), "{msg}");

    let msg = check_err("data P = P { x : Int }\nfun f() : P = P { y = 1 }");
    assert!(msg.contains("has no field `y`"), "{msg}");

    let msg = check_err("data P = P { x : Int }\nfun f() : P = P { x = 1, x = 2 }");
    assert!(msg.contains("given twice"), "{msg}");
}

#[test]
fn generic_top_level_values_generalize() {
    let (defs, _) = check_ok("data List a = Nil | Cons(a, List a)\nlet empty = Nil");
    let scheme = defs
        .value_by_name("empty")
        .and_then(|id| defs.values[id.index()].scheme.as_ref())
        .expect("empty has a scheme");
    assert_eq!(scheme.vars.len(), 1);
}

#[test]
fn unresolved_goal_over_a_type_parameter_defers() {
    let src = "\
class Sum a { addp : a -> a -> a }
data Box a = Box(a)
instance Sum Int { addp(x, y) = x + y }
fun f(b : a) : a = addp(b, b)
";
    let (_, hir) = check_ok(src);
    let f = body(&hir, "f");
    let mut found = Vec::new();
    refs(&f.body, &mut found);
    assert!(found.iter().any(|r| matches!(r, FnRef::Class { .. })), "lowering must pick the instance: {found:?}");
}
