//! End-to-end driver tests: `fernc` writes an object, the system linker
//! links it against a small C harness, and the result runs.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const FERNC: &str = env!("CARGO_BIN_EXE_fernc");

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fernc-test-{}-{name}", std::process::id()));
    fs::create_dir_all(&dir).expect("temp dir");
    dir
}

/// glcc, this repository's own C compiler, is the link driver for the
/// harnesses. Integration tests of a workspace member do not get
/// `CARGO_BIN_EXE_glcc`, so find the sibling of `fernc` in the target
/// directory. `cargo test --workspace` builds both.
fn glcc_binary() -> PathBuf {
    if let Some(path) = std::env::var_os("CARGO_BIN_EXE_glcc") {
        return PathBuf::from(path);
    }
    let candidate =
        Path::new(FERNC).parent().expect("fernc has a parent").join(format!("glcc{}", std::env::consts::EXE_SUFFIX));
    assert!(
        candidate.is_file(),
        "glcc not found at {}; build it with `cargo build -p glcc`",
        candidate.display()
    );
    candidate
}

fn have_cc() -> bool {
    Command::new(glcc_binary())
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Keep the fern object distinct from the C harness's own output.
fn object_path(dir: &Path) -> PathBuf {
    dir.join(if cfg!(windows) { "fern.obj" } else { "fern.o" })
}

fn exe_path(dir: &Path, name: &str) -> PathBuf {
    if cfg!(windows) { dir.join(format!("{name}.exe")) } else { dir.join(name) }
}

fn link_with_c(c_path: &Path, object: &Path, exe: &Path) -> Output {
    Command::new(glcc_binary())
        .arg(c_path)
        .arg(object)
        .arg("-o")
        .arg(exe)
        .output()
        .expect("run glcc")
}

fn run_fernc(source: &str, name: &str, extra: &[&str]) -> (PathBuf, PathBuf) {
    let dir = temp_dir(name);
    let fern_path = dir.join("main.fern");
    fs::write(&fern_path, source).expect("write source");
    let object_path = object_path(&dir);
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
    run_linked(dir, object, None)
}

/// Like [`link_and_run`], with `input` piped to the program's stdin.
fn link_and_run_with_stdin(dir: &Path, object: &Path, input: &str) -> (String, i32) {
    run_linked(dir, object, Some(input))
}

fn run_linked(dir: &Path, object: &Path, input: Option<&str>) -> (String, i32) {
    let c_path = dir.join("harness.c");
    fs::write(
        &c_path,
        "#include <stdio.h>\nlong long answer(void);\nint main(void) { printf(\"%lld\\n\", answer()); return 0; }\n",
    )
    .expect("write c harness");
    let exe = exe_path(dir, "program");
    let out = link_with_c(&c_path, object, &exe);
    assert!(out.status.success(), "link failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let out = match input {
        Some(input) => {
            let mut child = Command::new(&exe)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .expect("spawn program");
            child.stdin.take().expect("piped stdin").write_all(input.as_bytes()).expect("write stdin");
            child.wait_with_output().expect("run program")
        }
        None => Command::new(&exe).output().expect("run program"),
    };
    let stdout = String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n");
    (stdout, out.status.code().unwrap_or(-1))
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

#[test]
fn async_io_round_trips_a_file() {
    if !have_cc() {
        return;
    }
    let dir = temp_dir("async-io");
    let path = dir.join("data.txt").to_string_lossy().replace('\\', "/");
    let src = format!(
        "import std::io\n\
         \n\
         async fun roundtrip(p : String) : String = {{\n\
         let _ = await io.write_file(p, \"hello native io\");\n\
         await io.read_file(p)\n\
         }}\n\
         \n\
         fun answer() : Int = if block_on_string(roundtrip(\"{path}\")) == \"hello native io\" then 1 else 0\n"
    );
    let (dir, object) = run_fernc(&src, "async-io", &[]);
    assert!(object.exists(), "no object written");
    let (stdout, code) = link_and_run(&dir, &object);
    assert_eq!(code, 0, "program failed: {stdout}");
    assert_eq!(stdout, "1\n");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn async_read_line_reads_piped_stdin() {
    if !have_cc() {
        return;
    }
    let src = "import std::io\n\
               async fun line() : String = await io.read_line()\n\
               fun answer() : Int = if block_on_string(line()) == \"hello stdin\" then 1 else 0\n";
    let (dir, object) = run_fernc(src, "async-stdin-line", &[]);
    let (stdout, code) = link_and_run_with_stdin(&dir, &object, "hello stdin\n");
    assert_eq!(code, 0, "program failed: {stdout}");
    assert_eq!(stdout, "1\n");
}

#[test]
fn async_read_stdin_reads_piped_input() {
    if !have_cc() {
        return;
    }
    let src = "import std::io\n\
               async fun all() : String = await io.read_stdin(64)\n\
               fun answer() : Int = if block_on_string(all()) == \"hello stdin\" then 1 else 0\n";
    let (dir, object) = run_fernc(src, "async-stdin-all", &[]);
    let (stdout, code) = link_and_run_with_stdin(&dir, &object, "hello stdin");
    assert_eq!(code, 0, "program failed: {stdout}");
    assert_eq!(stdout, "1\n");
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
fn try_and_records_link_and_run() {
    if !have_cc() {
        return;
    }
    let dir = temp_dir("try-records");
    let source = dir.join("main.fern");
    fs::write(
        &source,
        "import std::result\n\
         \n\
         data Failure = Failure { message : String }\n\
         data Box = Box { value : Int }\n\
         \n\
         fun parse(ok : Bool) : result.Result Int Failure =\n\
             if ok then result.Ok(3) else result.Err(Failure { message = \"bad\" })\n\
         fun label(ok : Bool) : result.Result String Failure = {\n\
             let n = parse(ok)?;\n\
             result.Ok(concat(\"n=\", str_of_int(n)))\n\
         }\n\
         fun nested(ok : Bool) : result.Result String Failure = {\n\
             let s = label(ok)?;\n\
             result.Ok(s)\n\
         }\n\
         fun answer() : Int = {\n\
             let b = Box { value = 4 };\n\
             (match nested(true) {\n\
                 case result.Ok(s) => if s == \"n=3\" then b.value else 0\n\
                 case result.Err(_) => 0\n\
             }) + (match nested(false) {\n\
                 case result.Ok(_) => 0\n\
                 case result.Err(Failure(message)) => if message == \"bad\" then 2 else 0\n\
                 case result.Err(_) => 0\n\
             })\n\
         }\n",
    )
    .expect("write try-records program");

    let object = object_path(&dir);
    let out = Command::new(FERNC).arg(&source).arg("-O1").arg("-o").arg(&object).output().expect("run fernc");
    assert!(
        out.status.success(),
        "fernc failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let (stdout, code) = link_and_run(&dir, &object);
    assert_eq!(code, 0, "program failed: {stdout}");
    assert_eq!(stdout, "6\n");
}

#[test]
fn type_errors_are_reported_with_a_location() {
    let dir = temp_dir("error");
    let fern_path = dir.join("main.fern");
    fs::write(&fern_path, "fun main() : Int = 1 + true\n").expect("write source");
    let out = Command::new(FERNC).arg(&fern_path).arg("-o").arg(object_path(&dir)).output().expect("run fernc");
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
    let object = object_path(&dir);
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
    let object = object_path(&dir);
    let out = Command::new(FERNC)
        .arg(example_dir.join("brainfuck.fern"))
        .arg("-O1")
        .arg("-o")
        .arg(&object)
        .output()
        .expect("run fernc");
    assert!(out.status.success(), "fernc failed:\n{}", String::from_utf8_lossy(&out.stderr));

    let exe = exe_path(&dir, "brainfuck");
    let out = link_with_c(&example_dir.join("driver.c"), &object, &exe);
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
fn makefile_runner_builds_and_skips_up_to_date_targets() {
    if !have_cc() {
        return;
    }
    let dir = temp_dir("make-runner");
    let _ = fs::remove_file(dir.join("out.txt"));
    let tool_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/fernmake");
    let object = object_path(&dir);
    let out = Command::new(FERNC)
        .arg(tool_dir.join("make.fern"))
        .arg("-O1")
        .arg("-o")
        .arg(&object)
        .output()
        .expect("run fernc");
    assert!(out.status.success(), "fernc failed:\n{}", String::from_utf8_lossy(&out.stderr));

    let exe = exe_path(&dir, "fernmake");
    let out = link_with_c(&tool_dir.join("driver.c"), &object, &exe);
    assert!(out.status.success(), "link failed:\n{}", String::from_utf8_lossy(&out.stderr));

    fs::write(dir.join("source.txt"), "payload\n").expect("write source");
    fs::write(dir.join("common.mk"), "COPY ?= cp\nOUT = out.txt\n").expect("write included makefile");
    fs::write(
        dir.join("rules.mk"),
        "include common.mk\n\
         SHELL = sh\n\
         .SHELLFLAGS = -c\n\
         all: $(OUT)\n\
         \n\
         $(OUT): source.txt\n\
         \t$(COPY) $< $@\n\
         \tprintf \"built %s from %s\\n\" $@ $^\n\
         \n\
         report: $(OUT)\n\
         \tprintf \"report\\n\"\n\
         \n\
         .PHONY: all report clean\n\
         clean:\n\
         \trm -f $(OUT)\n",
    )
    .expect("write makefile");

    let run = |target: &str| {
        let mut command = Command::new(&exe);
        command.current_dir(&dir).arg("rules.mk");
        if !target.is_empty() {
            command.arg(target);
        }
        let out = command.output().expect("run fernmake");
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        (stdout, stderr, out.status.success())
    };

    let (stdout, stderr, ok) = run("");
    assert!(ok, "first build failed:\n{stdout}\n{stderr}");
    assert!(stdout.contains("cp source.txt out.txt"), "missing copy:\n{stdout}\n{stderr}");
    assert!(stdout.contains("built out.txt from source.txt"), "missing automatic vars:\n{stdout}\n{stderr}");
    assert_eq!(fs::read_to_string(dir.join("out.txt")).expect("read output"), "payload\n");

    let (stdout, stderr, ok) = run("");
    assert!(ok, "second build failed:\n{stdout}\n{stderr}");
    assert!(!stdout.contains("cp "), "rebuilt an up-to-date target:\n{stdout}\n{stderr}");

    let (stdout, stderr, ok) = run("report");
    assert!(ok, "report failed:\n{stdout}\n{stderr}");
    assert!(stdout.contains("report"), "missing report target:\n{stdout}\n{stderr}");
    assert!(!stdout.contains("cp "), "report rebuilt its prerequisite:\n{stdout}\n{stderr}");

    let (stdout, stderr, ok) = run("clean");
    assert!(ok, "clean failed:\n{stdout}\n{stderr}");
    assert!(!dir.join("out.txt").exists(), "clean left the output");

    fs::write(
        dir.join("features.mk"),
        concat!(
            "SHELL_WORD := $(shell printf 'from shell\\n')\n",
            "PICK = $(firstword $(wildcard source.txt) fallback)\n",
            "MISSING = $(firstword $(wildcard absent.txt) fallback)\n",
            "OVERRIDE ?= default\n",
            "LINE = one \\\n",
            "       two\n",
            "WHERE = $(dir $(realpath $(firstword $(MAKEFILE_LIST))))\n",
            "HERE = $(abspath $(WHERE).)\n",
            "\n",
            "all: build/features.txt\n",
            "\n",
            "build/features.txt: source.txt features.mk\n",
            "\tmkdir -p $(dir $@)\n",
            "\t@printf '%s\\n' '$(SHELL_WORD)' '$(PICK)' '$(MISSING)' '$(LINE)' '$(WHERE)' '$(HERE)' '$(OVERRIDE)' > $@\n",
        ),
    )
    .expect("write feature makefile");

    let out = Command::new(&exe)
        .current_dir(&dir)
        .arg("features.mk")
        .env("OVERRIDE", "from-env")
        .output()
        .expect("run feature makefile");
    assert!(
        out.status.success(),
        "feature build failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!stdout.contains("printf"), "`@` recipe was echoed:\n{stdout}");

    let features = fs::read_to_string(dir.join("build/features.txt")).expect("read features");
    let mut lines = features.lines();
    assert_eq!(lines.next(), Some("from shell"));
    assert_eq!(lines.next(), Some("source.txt"));
    assert_eq!(lines.next(), Some("fallback"));
    assert_eq!(lines.next(), Some("one two"));
    let where_line = lines.next();
    let here_line = lines.next();
    // Git Bash's `realpath` and Rust's `canonicalize` spell Windows paths
    // differently; Makefile.win does not use the shell path helpers.
    #[cfg(not(windows))]
    {
        let expected_dir = fs::canonicalize(&dir).expect("canonical temp dir");
        let expected_where = format!("{}/", expected_dir.display());
        let expected_here = expected_dir.to_str().expect("utf8 temp dir");
        assert_eq!(where_line, Some(expected_where.as_str()));
        assert_eq!(here_line, Some(expected_here));
    }
    #[cfg(windows)]
    {
        assert!(where_line.is_some(), "realpath produced no line");
        assert!(here_line.is_some(), "abspath produced no line");
    }
    assert_eq!(lines.next(), Some("from-env"));
}

#[test]
fn stdlib_modules_link_and_run() {
    if !have_cc() {
        return;
    }
    let dir = temp_dir("stdlib-modules");
    let source = dir.join("main.fern");
    fs::write(
        &source,
        "import std::iter\n\
         import std::list\n\
         import std::option\n\
         import std::path\n\
         import std::result\n\
         import std::string\n\
         \n\
         fun use_result(r : result.Result Int String) : Int = result.unwrap_or(0, r)\n\
         fun ext_len(p : String) : Int = match path.extension(p) {\n\
             case Nothing => 0\n\
             case Just(ext) => str_len(ext)\n\
         }\n\
         fun answer() : Int = {\n\
             let xs = list.range(1, 6);\n\
             let ys = list.map(x -> x * 2, list.filter(x -> x > 2, xs));\n\
             let repeated = iter.collect(iter.take(5, iter.repeat(7)));\n\
             let flat = iter.collect(iter.take(5, iter.flat_map(x -> iter.range(0, x), iter.range(1, 10))));\n\
             let zipped = list.zip(list.range(0, 3), list.range(10, 20));\n\
             let dropped = iter.collect(iter.take(2, iter.drop_while(x -> x < 5, iter.range(0, 10))));\n\
             let p = path.join(\"examples/fern\", \"make/main.fern\");\n\
             let text = string.replace(\"world\", \"fern\", string.trim(\"  hello world  \"));\n\
             if list.contains(6, ys) && eq(list.sum(ys), 24) && eq(list.sum(repeated), 35)\n\
                && eq(list.sum(flat), 2) && eq(list.length(zipped), 3) && eq(list.sum(dropped), 11)\n\
                && eq(str_len(text), 10) && eq(ext_len(p), 4) && eq(path.basename(\"/\"), \"/\")\n\
                && eq(option.unwrap_or_else(u -> 3, Nothing), 3)\n\
                && eq(use_result(option.to_result(\"bad\", Just(7))), 7) then 42 else 0\n\
         }\n",
    )
    .expect("write stdlib program");

    let object = object_path(&dir);
    let out = Command::new(FERNC)
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .output()
        .expect("run fernc");
    assert!(
        out.status.success(),
        "fernc failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let (stdout, code) = link_and_run(&dir, &object);
    assert_eq!(code, 0, "program failed: {stdout}");
    assert_eq!(stdout, "42\n");
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
    let object = object_path(&dir);
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
    let object = object_path(&dir);
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
