use std::io::Write as _;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};

use glcc_lower_lua::lower_module;

static NEXT: AtomicU32 = AtomicU32::new(0);

fn have_lua() -> bool {
    Command::new("lua5.1").arg("-v").output().is_ok()
}

fn run_mir(mir: &str) -> std::process::Output {
    run_mir_with(mir, b"", &[])
}

fn run_mir_with(mir: &str, stdin: &[u8], extra_args: &[&str]) -> std::process::Output {
    let module = glcc_mir::parse_module(mir).unwrap_or_else(|e| panic!("MIR parse failed: {e}\n{mir}"));
    let lua = lower_module(&module);
    let path = format!("/tmp/opencode/glcc-lua-{}-{}.lua", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed));
    std::fs::write(&path, lua.as_bytes()).unwrap();
    let mut child = Command::new("lua5.1")
        .args(extra_args)
        .arg(&path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("running lua5.1: {e}\n--- generated ---\n{lua}"));
    child.stdin.take().unwrap().write_all(stdin).unwrap();
    let output = child.wait_with_output().unwrap();
    if !output.stderr.is_empty() {
        panic!(
            "lua wrote to stderr: {:?}\n--- generated ---\n{lua}\n--- stderr ---\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    output
}

#[test]
fn arithmetic_runs_and_exits_with_main_result() {
    if !have_lua() {
        return;
    }
    let out = run_mir(
        r#"
.func @main sig=() -> i32
  %0:i32 = add 2, 3
  %1:i32 = mul %0, 7
  ret %1
"#,
    );
    assert_eq!(out.status.code(), Some(35), "stderr: {}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn loops_and_stack_memory() {
    if !have_lua() {
        return;
    }
    let out = run_mir(
        r#"
.func @main sig=() -> i32
  %0:ptr(i32) = alloca i32
  %1:ptr(i32) = alloca i32
  store %0, 0
  store %1, 0
  br b1
b1:
  %2:i32 = load %1
  %3:bool = icmp.slt %2, 10
  brif %3 b2 b3
b2:
  %4:i32 = load %0
  %5:i32 = load %1
  %6:i32 = add %4, %5
  store %0, %6
  %7:i32 = add %5, 1
  store %1, %7
  br b1
b3:
  %8:i32 = load %0
  ret %8
"#,
    );
    assert_eq!(out.status.code(), Some(45), "stderr: {}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn calls_and_gep() {
    if !have_lua() {
        return;
    }
    let out = run_mir(
        r#"
.func @add sig=(i32, i32) -> i32
.params %0:i32, %1:i32
  %2:i32 = add %0, %1
  ret %2
.func @main sig=() -> i32
  %0:ptr([3 x i32]) = alloca [3 x i32]
  %1:ptr(i32) = gep %0 0, 0
  %2:ptr(i32) = gep %0 0, 1
  %3:ptr(i32) = gep %0 0, 2
  store %1, 10
  store %2, 14
  store %3, 18
  %4:i32 = load %1
  %5:i32 = load %2
  %6:i32 = call @add i32 %4, i32 %5
  %7:i32 = load %3
  %8:i32 = call @add i32 %6, i32 %7
  ret %8
"#,
    );
    assert_eq!(out.status.code(), Some(42), "stderr: {}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn extern_and_global_string() {
    if !have_lua() {
        return;
    }
    let out = run_mir(
        r#"
.extern @puts sig=(ptr(void)) -> i32
.global @msg ty=[6 x i8] init=bytes<68656c6c6f00>
.func @main sig=() -> i32
  %0:ptr(void) = globaladdr @msg
  call.ext @puts ptr(void) %0
  ret 0
"#,
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "hello\n");
    assert_eq!(out.status.code(), Some(0));
}

#[test]
fn many_functions_do_not_exhaust_chunk_locals() {
    if !have_lua() {
        return;
    }
    let mut mir = String::new();
    for i in 0..250 {
        mir.push_str(&format!(".func @f{i} sig=() -> i32\n  ret {i}\n"));
    }
    mir.push_str(".func @main sig=() -> i32\n  %0:i32 = call @f249\n  ret %0\n");
    let out = run_mir(&mir);
    assert_eq!(out.status.code(), Some(249), "stderr: {}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn unreachable_blocks_are_not_emitted() {
    if !have_lua() {
        return;
    }
    let mir = r#"
.func @main sig=() -> i32
  br b1
b1:
  ret 7
b2:
  ret 99
"#;
    let module = glcc_mir::parse_module(mir).unwrap();
    let lua = lower_module(&module);
    assert!(!lua.contains("-- b2"), "unreachable block was emitted:\n{lua}");
    assert!(!lua.contains("pc == 2"), "unreachable block was emitted:\n{lua}");
    let out = run_mir(mir);
    assert_eq!(out.status.code(), Some(7), "stderr: {}", String::from_utf8_lossy(&out.stderr));
}

const GETCHAR_TWICE: &str = r#"
.extern @getchar sig=() -> i32
.func @main sig=() -> i32
  %0:i32 = call.ext @getchar
  %1:i32 = call.ext @getchar
  %2:i32 = add %0, %1
  ret %2
"#;

#[test]
fn getchar_reads_stdin_bytes() {
    if !have_lua() {
        return;
    }
    let out = run_mir_with(GETCHAR_TWICE, b"AB", &[]);
    assert_eq!(out.status.code(), Some(65 + 66), "stderr: {}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn getchar_drains_line_from_read() {
    if !have_lua() {
        return;
    }
    let out = run_mir_with(GETCHAR_TWICE, b"", &["-e", "read = function() return 'AB' end"]);
    assert_eq!(out.status.code(), Some(65 + 66), "stderr: {}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn getchar_returns_eof_at_end_of_input() {
    if !have_lua() {
        return;
    }
    let out = run_mir_with(
        r#"
.extern @getchar sig=() -> i32
.func @main sig=() -> i32
  %0:i32 = call.ext @getchar
  ret %0
"#,
        b"",
        &[],
    );
    assert_eq!(out.status.code(), Some(255), "stderr: {}", String::from_utf8_lossy(&out.stderr));
}
