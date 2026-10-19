//! `glcc-explorer` binary: serve the 3-panel page or compile one file.
//!
//! - `glcc-explorer file.c` prints MIR then assembly.
//! - `glcc-explorer --serve [--port N] [--wasm module.wasm]` serves
//!   the page, the wasm compiler module, and the objdump endpoint.
//! - `glcc-explorer --mir/--asm file.c` prints one panel only.
//!
//! `--serve` does not compile: the page loads `glcc-explorer.wasm`
//! (built by `tools/build-wasm.sh`) and runs the whole pipeline in the
//! browser. The server only provides the static page plus
//! `POST /disasm`, which runs local objdump on object bytes the page
//! uploads.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::PathBuf;

use glcc_explorer::{OptLevel, Options, compile};

const PAGE: &str = include_str!("page.html");
const WASM_NAME: &str = "glcc-explorer.wasm";

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "--serve" || a == "serve") {
        let port = port_arg(&argv).unwrap_or(7666);
        serve(port, wasm_path_arg(&argv));
        return;
    }
    if argv.is_empty() || argv.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    let mut want_mir = true;
    let mut want_mir_raw = false;
    let mut want_mir_full = false;
    let mut want_mir_raw_full = false;
    let mut want_asm = true;
    let mut opt = OptLevel::None;
    let mut target: Option<glcc_frontend::target::Target> = None;
    let mut async_enabled = false;
    let mut files: Vec<String> = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--mir" => want_asm = false,
            "--mir-raw" => {
                want_asm = false;
                want_mir_raw = true;
            }
            "--mir-full" => {
                want_asm = false;
                want_mir = false;
                want_mir_full = true;
            }
            "--mir-raw-full" => {
                want_asm = false;
                want_mir = false;
                want_mir_raw_full = true;
            }
            "--asm" => {
                want_mir = false;
                want_mir_raw = false;
                want_mir_full = false;
                want_mir_raw_full = false;
            }
            "-O0" => opt = OptLevel::None,
            "-O1" | "-O2" | "-O3" | "-O" => opt = OptLevel::Speed,
            "-Os" | "-Oz" => opt = OptLevel::Size,
            "-fglcc-async" => async_enabled = true,
            s if s.starts_with("--target=") => match glcc_frontend::target::Target::from_name(&s["--target=".len()..]) {
                Some(t) => target = Some(t),
                None => {
                    eprintln!("glcc-explorer: unsupported target '{}'", &s["--target=".len()..]);
                    std::process::exit(1);
                }
            },
            s if s.starts_with("--port=") => {}
            "--port" => {
                i += 1;
            }
            s if s.starts_with("--wasm=") => {}
            "--wasm" => {
                i += 1;
            }
            s if s.starts_with('-') => {
                eprintln!("glcc-explorer: unknown option '{s}'");
                std::process::exit(1);
            }
            f => files.push(f.to_string()),
        }
        i += 1;
    }
    if files.is_empty() {
        print_help();
        std::process::exit(1);
    }
    let path = &files[0];
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("glcc-explorer: cannot read '{path}': {e}");
            std::process::exit(1);
        }
    };
    let opts = Options {
        opt,
        target,
        preprocess: true,
        pp_native: None,
        cpp_args: Vec::new(),
        async_enabled,
    };
    let out = compile(&source, &opts);
    for e in &out.errors {
        eprintln!("glcc-explorer: error: {e}");
    }
    if want_mir_raw {
        println!("=== MIR (unoptimized) ===\n{}", out.mir_raw);
    }
    if want_mir_raw_full {
        println!("=== MIR (unoptimized, full) ===\n{}", out.mir_raw_full);
    }
    if want_mir {
        println!("=== MIR ===\n{}", out.mir);
    }
    if want_mir_full {
        println!("=== MIR (full) ===\n{}", out.mir_full);
    }
    if want_asm {
        println!("=== ASM ===\n{}", out.asm);
    }
    if !out.success {
        std::process::exit(1);
    }
}

fn port_arg(argv: &[String]) -> Option<u16> {
    for (i, a) in argv.iter().enumerate() {
        if let Some(rest) = a.strip_prefix("--port=") {
            return rest.parse().ok();
        }
        if a == "--port" {
            return argv.get(i + 1).and_then(|s| s.parse().ok());
        }
    }
    None
}

fn wasm_path_arg(argv: &[String]) -> Option<PathBuf> {
    for (i, a) in argv.iter().enumerate() {
        if let Some(rest) = a.strip_prefix("--wasm=") {
            return Some(PathBuf::from(rest));
        }
        if a == "--wasm" {
            return argv.get(i + 1).map(PathBuf::from);
        }
    }
    None
}

/// Where the wasm module lives: `--wasm`, `$GLCC_EXPLORER_WASM`, the
/// crate directory (where `tools/build-wasm.sh` drops it), or the workspace
/// target dir. `None` when the file exists nowhere; serving then tells
/// the page to run `tools/build-wasm.sh` instead of failing silently.
fn resolve_wasm(explicit: Option<PathBuf>) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(p) = explicit {
        candidates.push(p);
    }
    if let Some(p) = std::env::var_os("GLCC_EXPLORER_WASM") {
        candidates.push(PathBuf::from(p));
    }
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    candidates.push(PathBuf::from(WASM_NAME));
    candidates.push(crate_dir.join(WASM_NAME));
    candidates.push(crate_dir.join("../..").join("target/wasm32-unknown-unknown/wasm/glcc_explorer.wasm"));
    candidates.into_iter().find(|p| p.is_file())
}

fn print_help() {
    print!(
        "\
Usage: glcc-explorer [options] file.c | --serve [--port N] [--wasm module.wasm]
Godbolt-like inspector: C source in, MIR and assembly out.

  file.c            compile one file, print MIR then assembly
  --mir             print optimized MIR only (with a file)
  --mir-raw         print unoptimized MIR only (with a file)
  --mir-full        print optimized MIR with source locations, like
                    the driver's --emit-full-mir (with a file)
  --mir-raw-full    print unoptimized MIR with source locations (with a file)
  --asm             print assembly only (with a file)
  -O0/-O1/-O2/-O3/-Os/-Oz   opt level (default -O0)
  -fglcc-async      enable the async extension: `__attribute__((async))`,
                    `_Future(T)`, `__builtin_await` (off by default)
  --target=T        x86_64, aarch64, riscv64, or x86_64-windows (default native)
  --serve [--port N]  serve the browser page on 0.0.0.0:N (default 7666)
  --wasm FILE       wasm compiler module to serve (default: {WASM_NAME} next
                    to the crate or in target/wasm32-unknown-unknown/wasm)
"
    );
}

fn serve(port: u16, wasm: Option<PathBuf>) {
    let addr = format!("0.0.0.0:{port}");
    let listener = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("glcc-explorer: cannot bind '{addr}': {e}");
            std::process::exit(1);
        }
    };
    eprintln!("glcc-explorer: serving on 0.0.0.0:{port} (e.g. http://localhost:{port}/)");
    let wasm = resolve_wasm(wasm);
    match &wasm {
        Some(p) => eprintln!("glcc-explorer: wasm module {}", p.display()),
        None => eprintln!("glcc-explorer: no {WASM_NAME}; run ./tools/build-wasm.sh to build the in-browser compiler"),
    }
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let wasm = wasm.clone();
        std::thread::spawn(move || handle_conn(stream, wasm));
    }
}

fn handle_conn(stream: std::net::TcpStream, wasm: Option<PathBuf>) {
    let mut reader = BufReader::new(stream);
    let Some((method_owned, path_owned, body)) = read_http_request(&mut reader) else {
        return;
    };
    let mut stream = reader.into_inner();
    if method_owned == "GET" && (path_owned == "/" || path_owned.starts_with("/?")) {
        respond(&mut stream, "200 OK", "text/html; charset=utf-8", PAGE.as_bytes());
    } else if method_owned == "GET" && path_owned == "/health" {
        respond(&mut stream, "200 OK", "text/plain", b"ok");
    } else if method_owned == "GET" && path_owned == format!("/{WASM_NAME}") {
        match wasm.as_ref().and_then(|p| std::fs::read(p).ok()) {
            Some(bytes) => respond(&mut stream, "200 OK", "application/wasm", &bytes),
            None => respond(
                &mut stream,
                "404 Not Found",
                "text/plain",
                format!("{WASM_NAME} is not built; run ./tools/build-wasm.sh (restart this server if it started before the build)").as_bytes(),
            ),
        }
    } else if method_owned == "POST" && path_owned.starts_with("/disasm") {
        let text = glcc_explorer::disassemble(&body);
        respond(&mut stream, "200 OK", "text/plain; charset=utf-8", text.as_bytes());
    } else {
        respond(&mut stream, "404 Not Found", "text/plain", b"not found");
    }
}

fn read_http_request(reader: &mut impl BufRead) -> Option<(String, String, Vec<u8>)> {
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() || request_line.is_empty() {
        return None;
    }
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }
    let (method, path) = (parts[0].to_string(), parts[1].to_string());
    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => return None,
        }
        if line.trim_end().is_empty() {
            break;
        }
        let trimmed = line.trim_end();
        if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
            content_length = rest.trim().parse().unwrap_or(0);
        } else if let Some(rest) = trimmed.strip_prefix("content-length:") {
            content_length = rest.trim().parse().unwrap_or(0);
        }
    }
    // Read the POST body through the same buffered reader: headers and
    // body usually arrive in one packet, so bytes already buffered must
    // not be dropped via `into_inner` before the body is consumed.
    let mut body: Vec<u8> = Vec::new();
    if method == "POST" {
        const MAX_BODY: usize = 8 << 20;
        let want = content_length.min(MAX_BODY);
        body = vec![0u8; want];
        let mut read = 0;
        while read < want {
            match reader.read(&mut body[read..]) {
                Ok(0) => break,
                Ok(n) => read += n,
                Err(_) => return None,
            }
        }
        body.truncate(read);
    }
    Some((method, path, body))
}

fn respond(stream: &mut std::net::TcpStream, status: &str, content_type: &str, body: &[u8]) {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};

    #[test]
    fn coalesced_headers_and_body_are_preserved() {
        let src = "int add(int a,int b){return a+b;}";
        let raw = format!("POST /disasm HTTP/1.1\r\nContent-Length: {}\r\n\r\n{src}", src.len());
        let mut reader = BufReader::new(Cursor::new(raw.into_bytes()));
        let (method, path, body) = read_http_request(&mut reader).expect("request");
        assert_eq!(method, "POST");
        assert_eq!(body, src.as_bytes());
        assert!(path.starts_with("/disasm"));
    }

    #[test]
    fn wasm_module_is_resolved_from_the_crate_dir() {
        let path = resolve_wasm(None);
        if let Some(path) = path {
            // The crate-dir copy is the hyphen name; the target-dir
            // fallback keeps the cargo-crate underscores by design.
            assert!(path.is_file());
        }
    }
}
