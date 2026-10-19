//! Differential corpus vendored from clang's preprocessor test suite
//! (`clang/test/Preprocessor` in a local LLVM checkout).
//!
//! Regenerate with:
//!   ./tools/pp-diff.bash --vendor=crates/glcc_pp/tests/clang_pp
//!
//! `corpus.tsv` rows are `input <TAB> RUN-flags <TAB> expected-file`;
//! goldens are raw `clang -E` output and both sides are compared after
//! dropping `# <n> ...` linemarker lines (same contract as
//! `tools/pp-diff.bash`). They are ground truth: do not edit them by hand —
//! fix `glcc_pp` (or regenerate with a newer clang and accept the
//! churn).

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use glcc_pp::{PpOptions, preprocess};

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/clang_pp")
}

/// Standard macros clang keeps under `-undef` (per `-std`), captured at
/// vendor time so the goldens are host-independent.
fn load_predefines(root: &Path, std_key: &str) -> Vec<(String, String)> {
    let path = root.join("predefines").join(format!("{std_key}.txt"));
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut out = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("#define ") {
            let (name, value) = rest.split_once(' ').unwrap_or((rest, ""));
            out.push((name.to_string(), value.to_string()));
        }
    }
    out
}

#[derive(Default)]
struct RunFlags {
    std: Option<String>,
    std_key: String,
    defines: Vec<(String, Option<String>)>,
    undefs: Vec<String>,
    include_paths: Vec<PathBuf>,
    preincludes: Vec<String>,
}

fn apply_valued(f: &mut RunFlags, flag: &str, val: &str, root: &Path) {
    match flag {
        "-D" => {
            let (name, value) = match val.split_once('=') {
                Some((n, v)) => (n.to_string(), Some(v.to_string())),
                None => (val.to_string(), None),
            };
            f.defines.push((name, value));
        }
        "-U" => f.undefs.push(val.to_string()),
        "-I" | "-isystem" | "-iquote" => {
            // Vendoring ran clang from the corpus root, so relative
            // search paths are relative to the vendored inputs.
            let p = PathBuf::from(val);
            f.include_paths.push(if p.is_absolute() { p } else { root.join("inputs").join(p) });
        }
        "-include" => f.preincludes.push(val.to_string()),
        _ => unreachable!("unsupported flag {flag}"),
    }
}

fn parse_flags(flags: &str, root: &Path) -> RunFlags {
    let mut f = RunFlags {
        std_key: "default".to_string(),
        ..RunFlags::default()
    };
    let toks: Vec<&str> = flags.split_whitespace().collect();
    let mut i = 0;
    while i < toks.len() {
        let t = toks[i];
        match t {
            "-E" | "-undef" => {}
            "-o" => i += 1,
            _ if t.starts_with("-std=") => {
                f.std_key = t["-std=".len()..].to_string();
                f.std = Some(t.to_string());
            }
            "-D" | "-U" | "-I" | "-isystem" | "-iquote" | "-include" => {
                i += 1;
                apply_valued(&mut f, t, toks.get(i).copied().unwrap_or(""), root);
            }
            _ if t.starts_with("-include") => apply_valued(&mut f, "-include", &t["-include".len()..], root),
            _ if t.starts_with("-D") => apply_valued(&mut f, "-D", &t[2..], root),
            _ if t.starts_with("-U") => apply_valued(&mut f, "-U", &t[2..], root),
            _ if t.starts_with("-I") => apply_valued(&mut f, "-I", &t[2..], root),
            other => panic!("unsupported flag {other} in corpus.tsv"),
        }
        i += 1;
    }
    f
}

fn is_marker(line: &str) -> bool {
    line.strip_prefix("# ").is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit()))
}

fn strip_markers(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        if !is_marker(line.trim_end_matches('\n')) {
            out.push_str(line);
        }
    }
    out
}

/// The goldens are `clang -E` output with LF endings, but a checkout
/// with `core.autocrlf=true` rewrites the vendored corpus to CRLF.
/// Compare on normalized endings so the fixture is host-independent.
fn normalize(text: &str) -> String {
    text.replace("\r\n", "\n")
}

fn report(input: &str, expected: &str, got: &str, verbose: bool) -> String {
    let e: Vec<&str> = expected.lines().collect();
    let g: Vec<&str> = got.lines().collect();
    let mut out = String::new();
    let _ = writeln!(out, "=== {input}");
    let mut shown = 0;
    for i in 0..e.len().max(g.len()) {
        let (el, gl) = (e.get(i).copied(), g.get(i).copied());
        if el != gl {
            let _ = writeln!(out, "  line {}:", i + 1);
            let _ = writeln!(out, "    exp: {}", el.map_or("<missing>".to_string(), |s| format!("{s:?}")));
            let _ = writeln!(out, "    got: {}", gl.map_or("<missing>".to_string(), |s| format!("{s:?}")));
            shown += 1;
            if shown == 4 && !verbose {
                let _ = writeln!(out, "  ...");
                break;
            }
        }
    }
    out
}

#[test]
fn clang_corpus_matches() {
    let root = corpus_dir();
    let tsv = fs::read_to_string(root.join("corpus.tsv")).expect("corpus.tsv missing; run ./tools/pp-diff.bash --vendor=...");
    let filter = std::env::var("CLANG_CORPUS_FILTER").ok();
    let verbose = std::env::var("CLANG_CORPUS_VERBOSE").is_ok();
    let mut failures: Vec<String> = Vec::new();
    let mut runs = 0;
    for line in tsv.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let mut cols = line.split('\t');
        let input_name = cols.next().unwrap();
        if filter.as_ref().is_some_and(|f| !input_name.contains(f)) {
            continue;
        }
        let flags = parse_flags(cols.next().unwrap_or(""), &root);
        let expected_name = cols.next().unwrap();
        let input = root.join("inputs").join(input_name);
        let src = normalize(&fs::read_to_string(&input).unwrap_or_else(|e| panic!("{}: {e}", input.display())));
        let mut opts = PpOptions {
            linemarkers: true,
            std: flags.std,
            defines: flags.defines,
            undefs: flags.undefs,
            include_paths: flags.include_paths,
            preincludes: flags.preincludes,
            filename: input_name.to_string(),
            ..PpOptions::default()
        };
        // Quoted includes resolve against the vendored inputs dir, not
        // the test runner's cwd.
        opts.include_paths.insert(0, root.join("inputs"));
        opts.predefined = load_predefines(&root, &flags.std_key);
        runs += 1;
        let got = match preprocess(&src, input_name, &opts) {
            Ok(s) => normalize(&strip_markers(&s)),
            Err(e) => {
                failures.push(format!("=== {input_name}\n  error: {e}"));
                continue;
            }
        };
        let expected = normalize(&strip_markers(&fs::read_to_string(root.join(expected_name)).unwrap()));
        if got != expected {
            failures.push(report(input_name, &expected, &got, verbose));
        }
    }
    assert!(failures.is_empty(), "{} of {runs} clang corpus runs failed:\n\n{}", failures.len(), failures.join("\n"));
}
