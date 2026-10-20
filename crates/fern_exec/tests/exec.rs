//! End-to-end tests for the persistent JIT image.
//!
//! These exercise the property the REPL depends on: a later program can
//! call functions compiled by an earlier `load`, and only the new
//! symbols are defined.

#![cfg(not(windows))]

use fern::ty::Ty;
use fern_exec::{ExecSession, Value};

fn files(session: &fern::Session, file: fern::FileId) -> [fern::FileId; 2] {
    [session.prelude(), file]
}

#[test]
fn compiles_and_calls_a_function() {
    let mut session = fern::Session::new();
    let file = session.add_file("t.fern", "fun main() : Int = 1");
    let module = session.compile_shared(&files(&session, file)).expect("compiles");
    let mut exec = ExecSession::new();
    exec.load(&module);
    assert_eq!(exec.call("main", &Ty::int_ty()), Value::Int(1));
}

#[test]
fn a_later_program_calls_an_earlier_function() {
    let mut session = fern::Session::new();
    let file = session.add_file("t.fern", "fun twice(x : Int) : Int = x * 2");
    let module = session.compile_shared(&files(&session, file)).expect("compiles");
    let mut exec = ExecSession::new();
    exec.load(&module);
    assert!(exec.contains("twice"));

    session.set_text(file, "fun twice(x : Int) : Int = x * 2\nfun main() : Int = twice(21)");
    let module = session.compile_shared(&files(&session, file)).expect("compiles");
    exec.load(&module);
    assert_eq!(exec.call("main", &Ty::int_ty()), Value::Int(42));
}

#[test]
fn a_later_program_can_use_an_earlier_string() {
    let mut session = fern::Session::new();
    let file = session.add_file("t.fern", "fun hello() : String = \"hello\"");
    let module = session.compile_shared(&files(&session, file)).expect("compiles");
    let mut exec = ExecSession::new();
    exec.load(&module);
    assert_eq!(exec.call("hello", &Ty::named0("String")), Value::String("hello".to_string()));

    session.set_text(file, "fun hello() : String = \"hello\"\nfun greet() : String = concat(hello(), \"!\")");
    let module = session.compile_shared(&files(&session, file)).expect("compiles");
    exec.load(&module);
    assert_eq!(exec.call("greet", &Ty::named0("String")), Value::String("hello!".to_string()));
}

#[test]
fn reset_drops_the_previous_bodies() {
    let mut session = fern::Session::new();
    let file = session.add_file("t.fern", "fun main() : Int = 1");
    let module = session.compile_shared(&files(&session, file)).expect("compiles");
    let mut exec = ExecSession::new();
    exec.load(&module);
    assert_eq!(exec.call("main", &Ty::int_ty()), Value::Int(1));

    let mut other = fern::Session::new();
    let file = other.add_file("t.fern", "fun main() : Int = 2");
    let module = other.compile_shared(&files(&other, file)).expect("compiles");
    exec.reset();
    exec.load(&module);
    assert_eq!(exec.call("main", &Ty::int_ty()), Value::Int(2));
}

#[test]
fn many_loads_keep_calling_old_code() {
    let mut session = fern::Session::new();
    let file = session.add_file("t.fern", "fun a() : Int = 1");
    let mut exec = ExecSession::new();
    for step in 2..=6 {
        let text = format!("fun a() : Int = 1\nfun f{step}() : Int = a() + {step}");
        session.set_text(file, text);
        let module = session.compile_shared(&files(&session, file)).expect("compiles");
        exec.load(&module);
        assert_eq!(exec.call(&format!("f{step}"), &Ty::int_ty()), Value::Int(step + 1));
    }
}
