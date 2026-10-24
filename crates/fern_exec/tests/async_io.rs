//! End-to-end tests for the `std::io` futures.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use fern::ty::Ty;
use fern_exec::{ExecSession, Value};

fn run(src: &str) -> Value {
    let mut session = fern::Session::new();
    let io = session.load_import(vec!["std".to_string(), "io".to_string()]).expect("std::io");
    let file = session.add_file("t.fern", src);
    let module = session.compile_shared(&[session.prelude(), io, file]).expect("compiles");
    let mut exec = ExecSession::new();
    exec.load(&module);
    exec.call("main", &Ty::int_ty())
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fern-{name}-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn fern_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

#[test]
fn round_trips_a_file() {
    let dir = temp_dir("async-io-roundtrip");
    let path = fern_path(&dir.join("data.txt"));
    let src = format!(
        "import std::io\n\
         async fun roundtrip(p : String) : String = {{\n\
         let _ = await io.write_file(p, \"hello async io\");\n\
         await io.read_file(p)\n\
         }}\n\
         fun main() : Int = if block_on_string(roundtrip(\"{path}\")) == \"hello async io\" then 1 else 0"
    );
    assert_eq!(run(&src), Value::Int(1));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn reads_a_file_larger_than_one_chunk() {
    let dir = temp_dir("async-io-large");
    let path = dir.join("large.txt");
    fs::write(&path, "x".repeat(200_000)).expect("write large file");
    let src = format!(
        "import std::io\n\
         fun main() : Int = str_len(block_on_string(io.read_file(\"{}\")))",
        fern_path(&path)
    );
    assert_eq!(run(&src), Value::Int(200_000));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn sleep_suspends_the_driver() {
    let src = "import std::io\n\
               async fun nap() : Int = { let _ = await io.sleep(25); 5 }\n\
               fun main() : Int = block_on_int(nap())";
    let start = Instant::now();
    assert_eq!(run(src), Value::Int(5));
    assert!(start.elapsed() >= Duration::from_millis(20), "sleep returned after {:?}", start.elapsed());
}

#[test]
fn block_on_handles_scalar_results() {
    let src = "import std::io\n\
               async fun truth() : Bool = true\n\
               async fun half() : Float = 0.5\n\
               async fun text() : String = \"ok\"\n\
               fun main() : Int = if block_on_bool(truth()) && block_on_float(half()) == 0.5 && block_on_string(text()) == \"ok\" then 1 else 0";
    assert_eq!(run(src), Value::Int(1));
}
