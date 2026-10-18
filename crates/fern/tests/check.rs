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
    hir.bodies
        .iter()
        .find(|b| b.name == name)
        .unwrap_or_else(|| panic!("no body named `{name}`"))
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
        HirExprKind::Let { binds, body } => {
            for (_, value) in binds {
                refs(value, out);
            }
            refs(body, out);
        }
        HirExprKind::If {
            cond,
            then_body,
            else_body,
        } => {
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
    let (defs, _) = check_ok("fun id(x) = x");
    let scheme = defs
        .fn_by_name("id")
        .and_then(|id| defs.fns[id.index()].scheme.as_ref())
        .expect("id has a scheme");
    assert_eq!(scheme.vars.len(), 1);
    assert!(is_generic_name(&scheme.vars[0].0));
}

#[test]
fn one_scheme_many_instantiations() {
    check_ok("data Pair a b = Pair(a, b)\nfun id(x) = x\nfun p() = Pair(id(1), id(true))");
}

#[test]
fn higher_kinded_dispatch_defers_for_a_type_variable() {
    let (_, hir) = check_ok(format!("{LISTS}fun inc_all(xs) = map(x -> x + 1, xs)").as_str());
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
    let (_, hir) = check_ok(format!("{LISTS}fun one() = map(x -> x + 1, Cons(1, Nil))").as_str());
    let one = body(&hir, "one");
    let mut found = Vec::new();
    refs(&one.body, &mut found);
    assert!(
        found
            .iter()
            .any(|r| matches!(r, FnRef::Instance { instance: 0, method: 0, .. })),
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
fun sum2() = addp(1, 2)
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
    check_ok(format!("{LISTS}fun two() = Functor::map(x -> x + 2, Cons(1, Nil))").as_str());
}

#[test]
fn self_recursion_references_its_own_generics() {
    let (_, hir) = check_ok("fun dup(b, x) = if b then x else dup(false, x)");
    let dup_body = body(&hir, "dup");
    assert_eq!(dup_body.generics.len(), 1);
    let mut found = Vec::new();
    refs(&dup_body.body, &mut found);
    assert!(
        found
            .iter()
            .any(|r| matches!(r, FnRef::Fn { args, .. } if args.len() == 1)),
        "the self call passes the body's own generics: {found:?}"
    );
}

#[test]
fn strictly_typed_recursion() {
    check_ok("fun fact(n) = if n == 0 then 1 else n * fact(n - 1)");
}

#[test]
fn closures_capture_enclosing_locals() {
    let (_, hir) = check_ok("fun adder(n) = x -> x + n");
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
    let (_, hir) = check_ok("fun f(a) = y -> z -> a + y + z");
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
    check_ok("fun f(x) = { let y = x + 1; y * 2 }");
}

#[test]
fn matcher_unifies_arm_types() {
    check_ok("\
data Maybe a = Nothing | Just(a)
fun or_zero(m) = match m { case Nothing => 0 case Just(x) => x }
");
}

#[test]
fn list_sugar_round_trips() {
    let defs_src = "data List a = Nil | Cons(a, List a)\n";
    check_ok(format!("{defs_src}fun xs() = [1, 2, 3]\nfun empty() = []").as_str());
    check_ok(format!("{defs_src}fun head2(xs) = match xs {{ case [a, b] => a case _ => 0 }}").as_str());
    check_ok(format!("{defs_src}fun conses(xs) = match xs {{ case a :: rest => a case _ => 0 }}").as_str());
}

#[test]
fn integer_and_bool_patterns_work() {
    check_ok("fun sign(n) = match n { case 0 => 0 case _ => 1 }");
    check_ok("fun flip(b) = match b { case true => false case false => true }");
}

#[test]
fn error_use_before_defined() {
    let msg = check_err("fun f(x) = g(x)\nfun g(x) = x");
    assert!(msg.contains("used before it is defined"), "{msg}");
}

#[test]
fn error_unknown_name() {
    let msg = check_err("fun f(x) = zz(x)");
    assert!(msg.contains("cannot find `zz`"), "{msg}");
}

#[test]
fn error_arity_mismatch() {
    let msg = check_err("fun id(x) = x\nfun f() = id()");
    assert!(
        msg.contains("takes 1 parameter(s), but 0 argument(s)"),
        "{msg}"
    );
}

#[test]
fn closure_returning_function_is_called_by_declared_arity() {
    check_ok("fun mk(x) = y -> x + y\nfun f() = mk(1)(2)");
    check_ok("fun mk(x) = y -> x + y\nfun f() = { let g = mk(1); g(2) }");
}

#[test]
fn error_partial_application() {
    let msg = check_err("fun add(x, y) = x + y\nfun f() = add(1)(2)");
    assert!(
        msg.contains("takes 2 parameter(s), but 1 argument(s)"),
        "{msg}"
    );
}

#[test]
fn error_duplicate_pattern_bind() {
    let msg = check_err("fun f(x, x) = x");
    assert!(msg.contains("bound more than once"), "{msg}");
}

#[test]
fn error_operator_operand() {
    let msg = check_err("fun f(x) = x + true");
    assert!(msg.contains("expected a numeric type"), "{msg}");
    check_ok("fun f(x) = x < x");
    check_ok("fun f(x) = x < 1.5");
    check_ok("fun f(x) = x + 2.5");
    check_ok("fun f(x) = x == 1.5");
}

#[test]
fn error_ambiguous_class_method() {
    let src = "\
class Sum a { addp : a -> a -> a }
class Zero a { zero : a }
instance Sum Int { addp(x, y) = x + y }
instance Zero Int { zero() = 0 }
fun z() = Zero::zero()
";
    check_ok(src);
    let src = "\
class A a { shrunk : a -> a }
class B a { shrunk : a -> a }
fun z(x) = shrunk(x)
";
    let msg = check_err(src);
    assert!(msg.contains("ambiguous"), "{msg}");
}

#[test]
fn error_unusable_unknown() {
    let msg = check_err(
        "data List a = Nil | Cons(a, List a)\nfun f(n) = { let empty = Nil; n }",
    );
    assert!(msg.contains("cannot infer"), "{msg}");
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
fun f(b) = addp(b, b)
";
    let (_, hir) = check_ok(src);
    let f = body(&hir, "f");
    let mut found = Vec::new();
    refs(&f.body, &mut found);
    assert!(
        found.iter().any(|r| matches!(r, FnRef::Class { .. })),
        "lowering must pick the instance: {found:?}"
    );
}
