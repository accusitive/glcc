//! End-to-end driver tests: `fernc` writes an object, the system linker
//! links it against a small C harness, and the result runs.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const FERNC: &str = env!("CARGO_BIN_EXE_fernc");

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fernc-test-{}-{name}", std::process::id()));
    fs::create_dir_all(&dir).expect("temp dir");
    dir
}

fn have_cc() -> bool {
    Command::new("cc").arg("--version").output().map(|out| out.status.success()).unwrap_or(false)
}

fn run_fernc(source: &str, name: &str, extra: &[&str]) -> (PathBuf, PathBuf) {
    let dir = temp_dir(name);
    let fern_path = dir.join("main.fern");
    fs::write(&fern_path, source).expect("write source");
    let object_path = dir.join("main.o");
    let mut command = Command::new(FERNC);
    command.arg(&fern_path).arg("-o").arg(&object_path);
    for arg in extra {
        command.arg(arg);
    }
    let out = command.output().expect("run fernc");
    assert!(
        out.status.success(),
        "fernc failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (dir, object_path)
}

/// Run `fernc` on an in-memory source and return stdout.
fn run_fernc_stdout(source: &str, name: &str, extra: &[&str]) -> String {
    let dir = temp_dir(name);
    let fern_path = dir.join("main.fern");
    fs::write(&fern_path, source).expect("write source");
    let mut command = Command::new(FERNC);
    command.arg(&fern_path);
    for arg in extra {
        command.arg(arg);
    }
    let out = command.output().expect("run fernc");
    assert!(out.status.success(), "fernc failed:\n{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Link the object with a C harness that calls `answer()` and prints it.
fn link_and_run(dir: &Path, object: &Path) -> (String, i32) {
    let c_path = dir.join("main.c");
    fs::write(
        &c_path,
        "#include <stdio.h>\nlong answer(void);\nint main(void) { printf(\"%ld\\n\", answer()); return 0; }\n",
    )
    .expect("write c harness");
    let exe = dir.join("program");
    let out = Command::new("cc").arg(&c_path).arg(object).arg("-o").arg(&exe).output().expect("run cc");
    assert!(out.status.success(), "link failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let out = Command::new(&exe).output().expect("run program");
    (String::from_utf8_lossy(&out.stdout).into_owned(), out.status.code().unwrap_or(-1))
}

const PROGRAM: &str = r#"
fun range(n : Int) : List Int = if n == 0 then Nil else Cons(n, range(n - 1))
fun sum(xs : List Int) : Int = match xs { case Nil => 0 case Cons(x, xs) => x + sum(xs) }
fun twice(f : a -> a, x : a) : a = f(f(x))
fun inc(x : Int) : Int = x + 1
fun fib(n : Int) : Int = if n < 2 then n else fib(n - 1) + fib(n - 2)

class Functor f { map : (a -> b) -> f a -> f b }

instance Functor List {
    map(f, xs) = match xs { case Nil => Nil case Cons(x, xs) => Cons(f(x), map(f, xs)) }
}

fun answer() : Int = sum(map(x -> x * 2, range(10))) + twice(inc, 0) + if eq([1], [1]) then 1 else 0
"#;

#[test]
fn emits_an_object_that_links_and_runs() {
    if !have_cc() {
        return;
    }
    // 2 * (1 + .. + 10) = 110, plus twice(inc, 0) = 2, plus 1.
    let (dir, object) = run_fernc(PROGRAM, "plain", &[]);
    assert!(object.exists(), "no object written");
    let (stdout, code) = link_and_run(&dir, &object);
    assert_eq!(code, 0, "program failed: {stdout}");
    assert_eq!(stdout, "113\n");
}

#[test]
fn optimized_object_runs_the_same() {
    if !have_cc() {
        return;
    }
    let (dir, object) = run_fernc(PROGRAM, "opt", &["-O1"]);
    let (stdout, code) = link_and_run(&dir, &object);
    assert_eq!(code, 0, "program failed: {stdout}");
    assert_eq!(stdout, "113\n");
}

const FIB: &str = "fun fib(n : Int) : Int = if n < 2 then n else fib(n - 1) + fib(n - 2)\n";

#[test]
fn emit_mir_writes_the_module_to_stdout() {
    let text = run_fernc_stdout(FIB, "mir", &["--emit=mir"]);
    assert!(text.contains("func @fib:"), "no fib in dump:\n{text}");
    let text = run_fernc_stdout(FIB, "mir-flag", &["--emit-mir"]);
    assert!(text.contains("func @fib:"), "no fib in dump:\n{text}");
}

#[test]
fn emit_full_mir_keeps_every_compact_dump_line() {
    let plain = run_fernc_stdout(FIB, "full-mir-plain", &["--emit-mir"]);
    let full = run_fernc_stdout(FIB, "full-mir", &["--emit-full-mir"]);
    assert!(full.contains("func @fib:"), "no fib in dump:\n{full}");
    // fern lowering does not populate instruction locations yet, so the
    // full dump is currently identical; when locations land, every
    // compact line must survive with a `; file:line:column` suffix.
    for line in plain.lines() {
        assert!(full.contains(line), "compact line missing from full dump:\n{line}");
    }
}

#[test]
fn emit_full_mir_honors_the_output_path() {
    let dir = temp_dir("mir-out");
    let fern_path = dir.join("main.fern");
    fs::write(&fern_path, FIB).expect("write source");
    let dump_path = dir.join("main.mir");
    let out = Command::new(FERNC)
        .arg(&fern_path)
        .arg("--emit-full-mir")
        .arg("-o")
        .arg(&dump_path)
        .output()
        .expect("run fernc");
    assert!(out.status.success(), "fernc failed:\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(out.stdout.is_empty(), "dump leaked to stdout");
    let text = fs::read_to_string(&dump_path).expect("read dump");
    assert!(text.contains("func @fib:"), "no fib in dump:\n{text}");
}

#[test]
fn emit_check_stops_before_object_emission() {
    let dir = temp_dir("check");
    let fern_path = dir.join("main.fern");
    fs::write(&fern_path, "fun main() : Int = 1\n").expect("write source");
    let out = Command::new(FERNC).arg(&fern_path).arg("--emit=check").output().expect("run fernc");
    assert!(out.status.success());
    assert!(!dir.join("main.o").exists());
}

#[test]
fn type_errors_are_reported_with_a_location() {
    let dir = temp_dir("error");
    let fern_path = dir.join("main.fern");
    fs::write(&fern_path, "fun main() : Int = 1 + true\n").expect("write source");
    let out = Command::new(FERNC).arg(&fern_path).arg("-o").arg(dir.join("main.o")).output().expect("run fernc");
    assert!(!out.status.success(), "expected a compile error");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("error:"), "no error message:\n{stderr}");
    assert!(stderr.contains("main.fern"), "no file in message:\n{stderr}");
    assert!(stderr.contains("^"), "no caret in message:\n{stderr}");
}

#[test]
fn brainfuck_example_links_and_runs() {
    if !have_cc() {
        return;
    }
    let dir = temp_dir("brainfuck");
    let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/fern/brainfuck/brainfuck.fern");
    let object = dir.join("brainfuck.o");
    let out = Command::new(FERNC).arg(&example).arg("-O1").arg("-o").arg(&object).output().expect("run fernc");
    assert!(out.status.success(), "fernc failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let (stdout, code) = link_and_run(&dir, &object);
    assert_eq!(code, 0, "program failed: {stdout}");
    // "Hello World!\n", '*', '@', a NUL byte, and "He", folded into a
    // base-31 checksum mod 1_000_000_007 (seed 7).
    assert_eq!(stdout, "49729618\n");
}

#[test]
fn brainfuck_demo_prints_and_runs_files() {
    if !have_cc() {
        return;
    }
    let dir = temp_dir("brainfuck-demo");
    let example_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/fern/brainfuck");
    let object = dir.join("brainfuck.o");
    let out = Command::new(FERNC)
        .arg(example_dir.join("brainfuck.fern"))
        .arg("-O1")
        .arg("-o")
        .arg(&object)
        .output()
        .expect("run fernc");
    assert!(out.status.success(), "fernc failed:\n{}", String::from_utf8_lossy(&out.stderr));

    let exe = dir.join("brainfuck");
    let out = Command::new("cc")
        .arg(example_dir.join("driver.c"))
        .arg(&object)
        .arg("-o")
        .arg(&exe)
        .output()
        .expect("run cc");
    assert!(out.status.success(), "link failed:\n{}", String::from_utf8_lossy(&out.stderr));

    let out = Command::new(&exe).output().expect("run demo");
    assert!(out.status.success(), "demo failed: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("hello => Hello World!\\n"), "missing hello:\n{stdout}");
    assert!(stdout.contains("forty => *"), "missing forty:\n{stdout}");
    assert!(stdout.contains("cat => He"), "missing cat:\n{stdout}");
    assert!(stdout.contains("checksum = 49729618"), "bad checksum:\n{stdout}");

    let program = dir.join("program.bf");
    fs::write(&program, "++++++++[>++++++++<-]>+.").expect("write program");
    let out = Command::new(&exe).arg(&program).output().expect("run file");
    assert!(out.status.success(), "file mode failed: {}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8_lossy(&out.stdout), "A");

    let out = Command::new(&exe).arg(dir.join("missing.bf")).output().expect("run missing");
    assert!(!out.status.success(), "missing file should fail");
    assert!(String::from_utf8_lossy(&out.stdout).contains("cannot read"));
}

#[test]
fn module_imports_link_and_run() {
    if !have_cc() {
        return;
    }
    let dir = temp_dir("modules");
    let main_path = dir.join("main.fern");
    fs::write(
        &main_path,
        "import math\nfun answer() : Int = math.double(math.origin()) + 1\n",
    )
    .expect("write main");
    fs::write(
        dir.join("math.fern"),
        "fun double(x : Int) : Int = x * 2\nfun origin() : Int = 20\n",
    )
    .expect("write math");
    let object = dir.join("main.o");
    let out = Command::new(FERNC)
        .arg(&main_path)
        .arg("-o")
        .arg(&object)
        .output()
        .expect("run fernc");
    assert!(
        out.status.success(),
        "fernc failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (stdout, code) = link_and_run(&dir, &object);
    assert_eq!(code, 0, "program failed: {stdout}");
    assert_eq!(stdout, "41\n");
}

#[test]
fn same_named_modules_link_and_run() {
    if !have_cc() {
        return;
    }
    let dir = temp_dir("module-names");
    let source = |n: i64| {
        format!(
            "data Tree a = Leaf(a)\n\
             fun make() : Tree Int = Leaf({n})\n\
             fun get(t : Tree Int) : Int = match t {{ case Leaf(x) => x }}\n"
        )
    };
    let main_path = dir.join("main.fern");
    fs::write(
        &main_path,
        "import a\nimport b\nfun answer() : Int = a.get(a.make()) + b.get(b.make())\n",
    )
    .expect("write main");
    fs::write(dir.join("a.fern"), source(20)).expect("write a");
    fs::write(dir.join("b.fern"), source(22)).expect("write b");
    let object = dir.join("main.o");
    let out = Command::new(FERNC)
        .arg(&main_path)
        .arg("-o")
        .arg(&object)
        .output()
        .expect("run fernc");
    assert!(
        out.status.success(),
        "fernc failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (stdout, code) = link_and_run(&dir, &object);
    assert_eq!(code, 0, "program failed: {stdout}");
    assert_eq!(stdout, "42\n");
}
