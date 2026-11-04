//! Async end-to-end tests for the persistent JIT image.

use fern::ty::Ty;
use fern_exec::{ExecSession, Value};

fn run(src: &str) -> Value {
    let mut session = fern::Session::new();
    let file = session.add_file("t.fern", src);
    let module = session.compile_shared(&[session.prelude(), file]).expect("compiles");
    let mut exec = ExecSession::new();
    exec.load(&module);
    exec.call("main", &Ty::int_ty())
}

#[test]
fn async_and_await_run_to_completion() {
    assert_eq!(run("async fn inc(x) : Int -> Int = x + 1\nfn main() : -> Int = block_on_int(inc(41))"), Value::Int(42));
    assert_eq!(
        run("async fn inc(x) : Int -> Int = x + 1\n\
             async fn twice(x) : Int -> Int = { let a = inc(x).await; let b = inc(a).await; b }\n\
             fn main() : -> Int = block_on_int(twice(1))"),
        Value::Int(3)
    );
    assert_eq!(
        run("async fn inc(x) : Int -> Int = x + 1\n\
             async fn both(x) : Int -> Int = { let a = inc(x).await; let b = inc(x).await; a + b }\n\
             fn main() : -> Int = block_on_int(both(1))"),
        Value::Int(4)
    );
}

#[test]
fn a_tail_await_returns_the_child_result() {
    assert_eq!(
        run("async fn inc(x) : Int -> Int = x + 1\n\
             async fn tail(x) : Int -> Int = inc(x).await\n\
             fn main() : -> Int = block_on_int(tail(41))"),
        Value::Int(42)
    );
}

#[test]
fn async_frames_survive_collection() {
    let src = "\
        async fn sum(n) : Int -> Int = if n == 0 then 0 else { let xs = [n]; n + sum(n - 1).await }\n\
        fn main() : -> Int = block_on_int(sum(200))";
    assert_eq!(run(src), Value::Int(200 * 201 / 2));
}

#[test]
fn unit_futures_wait_without_a_result() {
    let src = "\
        async fn ping(x) : Int -> Unit = { let _ = x; () }\n\
        fn main() : -> Int = { let _ = block_on_unit(ping(1)); 7 }";
    assert_eq!(run(src), Value::Int(7));
}

#[test]
fn appending_a_caller_lowers_the_new_async_call() {
    let mut session = fern::Session::new();
    let file = session.add_file("t.fern", "async fn inc(x) : Int -> Int = x + 1");
    let module = session.compile_shared(&[session.prelude(), file]).expect("compiles");
    let mut exec = ExecSession::new();
    exec.load(&module);

    session.set_text(file, "async fn inc(x) : Int -> Int = x + 1\nfn main() : -> Int = block_on_int(inc(41))");
    let module = session.compile_shared(&[session.prelude(), file]).expect("compiles");
    exec.load(&module);
    assert_eq!(exec.call("main", &Ty::int_ty()), Value::Int(42));
}
