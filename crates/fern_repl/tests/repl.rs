//! REPL core tests: definitions persist, redefinitions win, and failed
//! input rolls back to the last good program.

#![cfg(not(windows))]

use fern_repl::{input_incomplete, is_definition, Repl};

#[test]
fn defines_and_calls_a_function() {
    let mut repl = Repl::new();
    let sig = repl.define("fun twice(x) = x * 2").expect("defines");
    assert_eq!(sig, vec!["twice : Int -> Int"]);
    assert_eq!(repl.eval_expr("twice(21)").expect("runs"), "42");
}

#[test]
fn a_later_definition_wins() {
    let mut repl = Repl::new();
    repl.define("fun f(x) = x + 1").expect("defines");
    repl.define("fun f(x) = x * 10").expect("redefines");
    assert_eq!(repl.eval_expr("f(4)").expect("runs"), "40");
}

#[test]
fn let_values_persist() {
    let mut repl = Repl::new();
    repl.define("let a = 1 + 2").expect("defines");
    assert_eq!(repl.eval_expr("a").expect("runs"), "3");
}

#[test]
fn a_failed_definition_rolls_back() {
    let mut repl = Repl::new();
    repl.define("fun good(x) = x + 1").expect("defines");
    let err = repl.define("fun bad(x) = x + true").expect_err("rejects");
    assert!(err[0].msg.contains("Bool"), "{:?}", err[0].msg);
    assert_eq!(repl.eval_expr("good(41)").expect("still works"), "42");
}

#[test]
fn adt_match_and_recursion() {
    let mut repl = Repl::new();
    repl.define("data Color = Red | Green | Blue").expect("data");
    repl.define(
        "fun to_int(c) = match c { case Red => 0 case Green => 1 case Blue => 2 }",
    )
    .expect("matches");
    assert_eq!(repl.eval_expr("to_int(Green)").expect("runs"), "1");

    repl.define("fun fac(n) = if n == 0 then 1 else n * fac(n - 1)")
        .expect("recursion");
    assert_eq!(repl.eval_expr("fac(6)").expect("runs"), "720");
}

#[test]
fn polymorphic_signature_uses_letters() {
    let mut repl = Repl::new();
    let sig = repl.define("fun id(x) = x").expect("defines");
    assert!(sig[0].contains("forall a"), "{sig:?}");
    assert!(!sig[0].contains("t$"), "{sig:?}");
}

#[test]
fn closures_and_higher_order_functions() {
    let mut repl = Repl::new();
    repl.define("fun sum(xs) = match xs { case Nil => 0 case Cons(x, rest) => x + sum(rest) }")
        .expect("defines");
    repl.define("fun map(f, xs) = match xs { case Nil => Nil case Cons(x, rest) => Cons(f(x), map(f, rest)) }")
        .expect("defines");
    assert_eq!(
        repl.eval_expr("sum(map((x) -> x * 2, [1, 2, 3]))").expect("runs"),
        "12"
    );
}

#[test]
fn clear_forgets_definitions() {
    let mut repl = Repl::new();
    repl.define("fun f(x) = x").expect("defines");
    repl.clear();
    let err = repl.eval_expr("f(1)").expect_err("f is gone");
    assert!(err[0].msg.contains("cannot find"), "{:?}", err[0].msg);
}

#[test]
fn string_literals_evaluate() {
    let mut repl = Repl::new();
    assert_eq!(repl.eval_expr("\"hello\"").expect("runs"), "hello");
    assert_eq!(repl.eval_expr("\"a\\nb\"").expect("runs"), "a\nb");
    repl.define("fun greeting(name) = name").expect("defines");
    assert_eq!(
        repl.eval_expr("greeting(\"fern\")").expect("runs"),
        "fern"
    );
}

#[test]
fn externs_and_shell_primitives() {
    let mut repl = Repl::new();
    assert_eq!(repl.eval_expr("\"a\" == \"a\"").expect("runs"), "true");
    assert_eq!(
        repl.eval_expr("spawn_status(\"true\", [])").expect("runs"),
        "0"
    );
    assert_eq!(
        repl.eval_expr("spawn_status(\"false\", [])").expect("runs"),
        "1"
    );
    assert_eq!(
        repl.eval_expr("spawn_capture(\"echo\", [\"hi\"])")
            .expect("runs"),
        "hi\n"
    );
    repl.define("extern \"strlen\" c_strlen : String -> Int")
        .expect("declares an extern");
    assert_eq!(repl.eval_expr("c_strlen(\"abcd\")").expect("runs"), "4");
    assert_eq!(
        repl.eval_expr("str_join(\"|\", str_split(\",\", \"a,b\"))")
            .expect("runs"),
        "a|b"
    );
    assert_eq!(
        repl.eval_expr("set_env(\"FERN_PHASE_B\", \"yes\")")
            .expect("runs"),
        "true"
    );
    assert_eq!(
        repl.eval_expr("getenv(\"FERN_PHASE_B\")").expect("runs"),
        "yes"
    );
}

#[test]
fn classifies_input_for_the_driver() {
    assert!(is_definition("fun f(x) = x"));
    assert!(is_definition("let a = 1"));
    assert!(!is_definition("1 + 2"));
    assert!(input_incomplete("fun f(x) = {"));
    assert!(!input_incomplete("fun f(x) = x"));
}
