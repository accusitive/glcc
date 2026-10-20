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
fn shell_commands_capture_and_pipe() {
    let mut repl = Repl::new();
    assert_eq!(repl.eval_expr("$ true").expect("runs"), "true");
    assert_eq!(repl.eval_expr("$ false").expect("runs"), "false");
    assert_eq!(repl.eval_expr("$(echo hi)").expect("runs"), "hi\n");
    assert_eq!(
        repl.eval_expr("$(echo hi | tr a-z A-Z)").expect("runs"),
        "HI\n"
    );
    assert_eq!(
        repl.eval_expr("$(printf \"[%s]\" ${\"a b\"})")
            .expect("runs"),
        "[a b]"
    );
    assert_eq!(
        repl.eval_expr("($ #env HOME) == getenv(\"HOME\")")
            .expect("runs"),
        "true"
    );
    assert_eq!(
        repl.eval_expr("$ true && $ false").expect("runs"),
        "false"
    );
    assert_eq!(
        repl.eval_expr("$(printf hi) |> str_len").expect("runs"),
        "2"
    );
}

#[test]
fn shell_commands_work_in_definitions() {
    let mut repl = Repl::new();
    let sig = repl
        .define("let shell_ok = $ true")
        .expect("defines a command value");
    assert_eq!(sig, vec!["shell_ok : Bool"]);
    assert_eq!(repl.eval_expr("shell_ok").expect("runs"), "true");
    repl.define("fun upper(s) = $(printf \"%s\" ${s} | tr a-z A-Z)")
        .expect("defines a capture");
    assert_eq!(repl.eval_expr("upper(\"abc\")").expect("runs"), "ABC");
}

#[test]
fn incomplete_shell_input_is_recognized() {
    assert!(input_incomplete("$ echo hi |"));
    assert!(input_incomplete("$ echo \"oops"));
    assert!(input_incomplete("$(echo hi"));
    assert!(input_incomplete("$ echo ${x"));
    assert!(input_incomplete("$ echo hi &&"));
    assert!(input_incomplete("1 +"));
    assert!(!input_incomplete("$ echo hi"));
    assert!(!input_incomplete("$(echo hi)"));
    assert!(!input_incomplete("$ true && $ false"));
    assert!(!input_incomplete(""));
    assert!(!input_incomplete("   \n"));
    assert!(!input_incomplete("-- just a comment"));
}

#[test]
fn classifies_input_for_the_driver() {
    assert!(is_definition("fun f(x) = x"));
    assert!(is_definition("let a = 1"));
    assert!(!is_definition("1 + 2"));
    assert!(input_incomplete("fun f(x) = {"));
    assert!(!input_incomplete("fun f(x) = x"));
}

#[test]
fn redefining_after_an_evaluation_uses_the_new_body() {
    let mut repl = Repl::new();
    repl.define("fun f(x) = x + 1").expect("defines");
    assert_eq!(repl.eval_expr("f(1)").expect("runs"), "2");
    repl.define("fun f(x) = x + 2").expect("redefines");
    assert_eq!(repl.eval_expr("f(1)").expect("runs the new body"), "3");
}

#[test]
fn new_monomorphizations_work_across_evaluations() {
    let mut repl = Repl::new();
    repl.define("fun id(x) = x").expect("defines");
    assert_eq!(repl.eval_expr("id(1)").expect("runs"), "1");
    assert_eq!(repl.eval_expr("id(true)").expect("runs"), "true");
}

#[test]
fn a_heavy_library_is_checked_and_lowered_incrementally() {
    let mut repl = Repl::new();
    let mut library = String::new();
    for i in 0..150 {
        library.push_str(&format!("fun f{i}(x) = x + {i}\n"));
    }
    repl.define(&library).expect("defines a library");

    // The first expression pays for the library once.
    assert_eq!(repl.eval_expr("f149(1)").expect("runs"), "150");

    // The next one must not: only its own wrapper is lowered.
    assert_eq!(repl.eval_expr("f0(41)").expect("runs"), "41");
    let stats = repl.lower_stats();
    assert!(!stats.full, "the library lowering is reused");
    assert_eq!(stats.bodies, 1, "only the new expression is emitted");
}
