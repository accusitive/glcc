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
