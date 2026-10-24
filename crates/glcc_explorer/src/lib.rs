//! `glcc_explorer`: Godbolt-style inspector for the glcc pipeline.
//!
//! Give it C source, get back the MIR dump and disassembled machine code.
//! The library owns the compile (`compile`); the `glcc-explorer` binary
//! serves a three-panel page (source / MIR / assembly) over a std-only
//! HTTP server so this crate adds no new dependencies.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use glcc_frontend::lexer::{Interner, Lexer};
use glcc_frontend::parse::Parser;
use glcc_frontend::sema::Sema;
use glcc_frontend::target::Target as FrontendTarget;

#[cfg(target_arch = "wasm32")]
mod wasm;

/// Optimization selector, re-exported from `glcc_mir` so the explorer
/// and every other consumer share one definition.
pub use glcc_mir::OptLevel;

/// What to compile and how.
#[derive(Debug, Clone, Default)]
pub struct Options {
    pub opt: OptLevel,
    pub target: Option<FrontendTarget>,
    /// Run the preprocessor first when the source has preprocessor
    /// lines. On by default; disable for pure-language inputs or
    /// offline use.
    pub preprocess: bool,
    /// Which preprocessor to run: None prefers the native `glcc_pp`
    /// (falling back per `GLCC_PP`).
    pub pp_native: Option<bool>,
    /// Extra flags forwarded to `clang -E` (e.g. `-Iinc -DFOO=1`).
    pub cpp_args: Vec<String>,
    /// `-fglcc-async`: `__attribute__((async))` becomes a stackless
    /// coroutine, and `_Future(T)`/`__builtin_await` become usable.
    /// Off by default, exactly like the driver flag.
    pub async_enabled: bool,
}

impl Options {
    pub fn target_or_native(&self) -> FrontendTarget {
        self.target.unwrap_or_else(FrontendTarget::native)
    }
}

/// One compile: MIR text, assembly text, and any diagnostics.
///
/// `mir` is the post-pass (optimized) dump and matches the generated
/// assembly; `mir_raw` is the pre-pass dump straight from lowering.
/// The `_full` variants annotate every located instruction with its
/// source position, like the driver's `--emit-full-mir`. With
/// `-fglcc-async` the raw dump still carries `await` instructions:
/// the coroutine split into start/resume pairs happens between the two
/// views, before optimization, as in the driver.
#[derive(Debug, Clone)]
pub struct Output {
    pub success: bool,
    pub mir: String,
    pub mir_raw: String,
    pub mir_full: String,
    pub mir_raw_full: String,
    pub asm: String,
    pub errors: Vec<String>,
    /// Machine-readable diagnostics (the frontend's JSON object:
    /// `{"diagnostics":[...]}`), for the editor integration. Empty
    /// string when the failure happened before the frontend ran.
    pub diagnostics: String,
    pub object_len: usize,
    /// Emitted object (relocatable ELF) bytes. Empty when codegen did
    /// not run; the wasm build ships these to the disassembly endpoint.
    pub object: Vec<u8>,
}

/// Front-end + MIR + codegen output before any disassembly: the wasm
/// build's half of [`compile`].
#[derive(Debug, Clone)]
pub struct Prepared {
    pub mir: String,
    pub mir_raw: String,
    pub mir_full: String,
    pub mir_raw_full: String,
    pub object: Vec<u8>,
    pub errors: Vec<String>,
    pub diagnostics: String,
}

/// Compile C source to MIR + object bytes. Never panics: frontend and
/// backend panics are caught and reported in `errors`.
///
/// On failure the `Output` carries whatever was produced before the
/// failing stage (partial MIR, earlier errors).
pub fn prepare(source: &str, opts: &Options) -> Result<Prepared, Output> {
    let target = opts.target_or_native();
    let mut errors: Vec<String> = Vec::new();

    let text: String = if opts.preprocess && needs_preprocess(source) {
        match run_preprocessor(source, target, &opts.cpp_args, use_native_pp(opts)) {
            Ok(t) => t,
            Err(e) => {
                return Err(Output {
                    success: false,
                    mir: String::new(),
                    mir_raw: String::new(),
                    mir_full: String::new(),
                    mir_raw_full: String::new(),
                    asm: String::new(),
                    errors: vec![e],
                    diagnostics: String::new(),
                    object_len: 0,
                    object: Vec::new(),
                });
            }
        }
    } else {
        source.to_string()
    };

    let (mut mir_module, mir_raw, mir_raw_full, diagnostics) = match lower_to_mir(&text, target, opts.async_enabled) {
        Ok(v) => v,
        Err(e) => {
            return Err(Output {
                success: false,
                mir: e.partial_mir.clone(),
                mir_raw: e.partial_mir,
                mir_full: String::new(),
                mir_raw_full: String::new(),
                asm: String::new(),
                errors: e.errors,
                diagnostics: e.diagnostics,
                object_len: 0,
                object: Vec::new(),
            });
        }
    };
    run_mir_passes(&mut mir_module, opts.opt);
    let mir_text = glcc_mir::display::dump_module(&mir_module);
    let mir_full = glcc_mir::display::dump_module_full(&mir_module);
    errors.extend(mir_module_verify_notes(&mir_module));

    let object_bytes = match lower_to_object(&mir_module, target, opts.opt) {
        Ok(b) => b,
        Err(e) => {
            return Err(Output {
                success: false,
                mir: mir_text,
                mir_raw: mir_raw.clone(),
                mir_full,
                mir_raw_full,
                asm: String::new(),
                errors: [errors, vec![e]].concat(),
                diagnostics: diagnostics.clone(),
                object_len: 0,
                object: Vec::new(),
            });
        }
    };

    Ok(Prepared {
        mir: mir_text,
        mir_raw,
        mir_full,
        mir_raw_full,
        object: object_bytes,
        errors,
        diagnostics,
    })
}

/// Compile C source to MIR + assembly. Never panics: frontend and
/// backend panics are caught and reported in `errors`.
pub fn compile(source: &str, opts: &Options) -> Output {
    match prepare(source, opts) {
        Ok(p) => Output {
            success: p.errors.is_empty(),
            asm: disassemble(&p.object),
            mir: p.mir,
            mir_raw: p.mir_raw,
            mir_full: p.mir_full,
            mir_raw_full: p.mir_raw_full,
            errors: p.errors,
            diagnostics: p.diagnostics,
            object_len: p.object.len(),
            object: p.object,
        },
        Err(o) => o,
    }
}

/// [`prepare`] wrapped as an [`Output`], with the assembly panel left
/// empty: the wasm page asks the server's disassembly endpoint for it
/// after shipping the object bytes out.
pub fn compile_object(source: &str, opts: &Options) -> Output {
    match prepare(source, opts) {
        Ok(p) => Output {
            success: p.errors.is_empty(),
            mir: p.mir,
            mir_raw: p.mir_raw,
            mir_full: p.mir_full,
            mir_raw_full: p.mir_raw_full,
            asm: String::new(),
            errors: p.errors,
            diagnostics: p.diagnostics,
            object_len: p.object.len(),
            object: p.object,
        },
        Err(o) => o,
    }
}

/// Target code used across the wasm ABI (`glcc_compile` argument):
/// 0 = x86_64, 1 = aarch64, 2 = riscv64.
pub fn target_from_code(code: u32) -> FrontendTarget {
    match code {
        1 => FrontendTarget::AARCH64,
        2 => FrontendTarget::RISCV64,
        _ => FrontendTarget::X86_64,
    }
}

/// Optimization code used across the wasm ABI: 0 = -O0, 1 = -O2-ish
/// (`speed`), 2 = -Oz-ish (`size`).
pub fn opt_from_code(code: u32) -> OptLevel {
    match code {
        1 => OptLevel::Speed,
        2 => OptLevel::Size,
        _ => OptLevel::None,
    }
}

struct MirFailure {
    errors: Vec<String>,
    partial_mir: String,
    diagnostics: String,
}

fn needs_preprocess(source: &str) -> bool {
    source.lines().any(|l| l.trim_start().starts_with('#'))
}

/// clang-compatible binary for the non-native path (`GLCC_PP` names a
/// clang build; the native `glcc_pp` is preferred, see [`use_native_pp`]).
fn preprocessor_binary() -> String {
    std::env::var("GLCC_PP").unwrap_or_else(|_| "clang".to_string())
}

/// Whether the native `glcc_pp` handles this compile:
/// `Options::pp_native` wins when set; otherwise unset, empty, or
/// `native` values of `GLCC_PP` mean the native path.
fn use_native_pp(opts: &Options) -> bool {
    match opts.pp_native {
        Some(v) => v,
        None => match std::env::var_os("GLCC_PP") {
            None => true,
            Some(v) => v.is_empty() || v == "native",
        },
    }
}

fn run_preprocessor(source: &str, target: FrontendTarget, extra: &[String], native: bool) -> Result<String, String> {
    if native {
        let cpp: Vec<std::ffi::OsString> = extra.iter().map(std::ffi::OsString::from).collect();
        let mut popts = glcc_pp::cpp_flags_to_opts(&cpp, "input.c");
        popts.linemarkers = true;
        popts.target = Some(target);
        popts.predefined = glcc_pp::predefines(Some(target.triple()), None);
        if !popts.no_std_includes {
            popts.include_paths.extend(glcc_pp::native_include_paths(Some(target.triple()), None));
        }
        return glcc_pp::preprocess(source, "input.c", &popts).map_err(|e| e.to_string());
    }
    let dir = temp_file("glcc-explorer-pp-", ".c")?;
    std::fs::write(&dir, source).map_err(|e| format!("cannot write temp source: {e}"))?;
    let mut cmd = Command::new(preprocessor_binary());
    cmd.arg("-E");
    if !target.is_native() {
        cmd.arg(format!("--target={}", target.triple()));
    }
    for a in extra {
        cmd.arg(a);
    }
    cmd.arg(&dir);
    let out = cmd.output().map_err(|e| format!("cannot run preprocessor: {e}"))?;
    let _ = std::fs::remove_file(&dir);
    if !out.status.success() {
        return Err(format!("preprocessing failed:\n{}", String::from_utf8_lossy(&out.stderr).trim_end()));
    }
    Ok(glcc_frontend::decode_source(&out.stdout))
}

fn lower_to_mir(source: &str, target: FrontendTarget, async_enabled: bool) -> Result<(glcc_mir::Module, String, String, String), MirFailure> {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(glcc_mir::Module, String, String, String), (Vec<String>, String)> {
        let mut interner = Interner::new();
        let arena = glcc_frontend::bumpalo::Bump::new();
        let mut sema = Sema::with_target(target, &arena);
        sema.async_enabled = async_enabled;
        let tokens = Lexer::new(source, &mut interner).lex_all();
        let mut parser = Parser::new(source, tokens, &arena, &mut sema, &mut interner);
        // The explorer reports diagnostics through `Output::errors` and
        // `Output::diagnostics`, not codespan's stderr rendering.
        parser.emit_human = false;
        let tu = parser.parse_translation_unit();
        let diagnostics = parser.diagnostics_json();
        if parser.has_errors() {
            let errors = parser
                .diagnostics()
                .into_iter()
                .filter(|d| d.is_error())
                .map(|d| d.message)
                .collect();
            return Err((errors, diagnostics));
        }
        drop(parser);
        let mut module = glcc_lower_mir::lower_translation_unit(&sema, &interner, &tu);
        let text = glcc_mir::display::dump_module(&module);
        let text_full = glcc_mir::display::dump_module_full(&module);
        // Driver-equivalent ordering: coroutines become start/resume
        // pairs before any other pass and regardless of the opt level
        // (`await` has no lowering of its own); `text` stays pre-split
        // so the raw panels show the awaits.
        glcc_mir::pass::PassManager::new().add_module_pass(glcc_mir::coroutine::LowerCoroutines).run(&mut module);
        Ok((module, text, text_full, diagnostics))
    }));
    match outcome {
        Ok(Ok((module, text, text_full, diagnostics))) => {
            let verify = glcc_mir::verify::verify_module(&module);
            if verify.is_empty() {
                Ok((module, text, text_full, diagnostics))
            } else {
                Err(MirFailure {
                    errors: verify,
                    partial_mir: text,
                    diagnostics: String::new(),
                })
            }
        }
        Ok(Err((errors, diagnostics))) => Err(MirFailure {
            errors,
            partial_mir: String::new(),
            diagnostics,
        }),
        Err(payload) => Err(MirFailure {
            errors: vec![format!("frontend error: {}", panic_message(&payload))],
            partial_mir: String::new(),
            diagnostics: String::new(),
        }),
    }
}

fn mir_module_verify_notes(_module: &glcc_mir::Module) -> Vec<String> {
    Vec::new()
}

/// Driver-equivalent MIR pipeline (`glcc_mir::optimize`), so the
/// explorer always matches the other consumers. `-O0` runs no passes,
/// so the optimized view matches the raw dump.
fn run_mir_passes(module: &mut glcc_mir::Module, opt: OptLevel) {
    glcc_mir::optimize(module, opt);
}

fn lower_to_object(module: &glcc_mir::Module, target: FrontendTarget, opt: OptLevel) -> Result<Vec<u8>, String> {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let cl_target = match target.arch {
            glcc_frontend::target::Arch::X86_64 => glcc_lower_cl::Arch::X86_64,
            glcc_frontend::target::Arch::Aarch64 => glcc_lower_cl::Arch::Aarch64,
            glcc_frontend::target::Arch::Riscv64 => glcc_lower_cl::Arch::Riscv64,
        };
        let cl_os = match target.os {
            glcc_frontend::target::Os::Linux => glcc_lower_cl::Os::Linux,
            glcc_frontend::target::Os::Windows => glcc_lower_cl::Os::Windows,
        };
        let cl_target = glcc_lower_cl::Target { arch: cl_target, os: cl_os };
        use cranelift_codegen::settings::Configurable;
        let mut flag_builder = cranelift_codegen::settings::builder();
        flag_builder.set("opt_level", opt.flag()).expect("known opt level");
        flag_builder.set("is_pic", "true").expect("pic");
        flag_builder.set("preserve_frame_pointers", "true").expect("frame pointers");
        flag_builder.set("tls_model", "elf_gd").expect("tls model");
        let flags = cranelift_codegen::settings::Flags::new(flag_builder);
        // `cranelift_native` only knows the host arch; inside wasm the
        // host is wasm32, so every target goes through the triples.
        let isa: cranelift_codegen::isa::OwnedTargetIsa = if target.is_native() && !cfg!(target_arch = "wasm32") {
            cranelift_native::builder().expect("native isa").finish(flags).expect("isa flags")
        } else {
            cranelift_codegen::isa::lookup(target.triple().parse().expect("known target triple"))
                .expect("known isa")
                .finish(flags)
                .expect("isa flags")
        };
        let builder = cranelift_object::ObjectBuilder::new(isa, "glcc-explorer", cranelift_module::default_libcall_names()).expect("object builder");
        let mut obj = cranelift_object::ObjectModule::new(builder);
        glcc_lower_cl::lower_module(module, &mut obj, cl_target);
        obj.finish().emit().expect("object emit")
    }));
    match outcome {
        Ok(bytes) => Ok(bytes),
        Err(payload) => Err(format!("codegen error: {}", panic_message(&payload))),
    }
}

/// Disassemble object bytes via `objdump`, falling back to
/// `llvm-objdump` (GNU binutils is usually host-target-only, so an
/// aarch64 or riscv64 object needs LLVM's multi-target objdump).
/// A missing disassembler becomes the asm panel text instead of an
/// error, so the wasm page still shows MIR and diagnostics.
pub fn disassemble(object: &[u8]) -> String {
    let path = match temp_file("glcc-explorer-obj-", ".o") {
        Ok(p) => p,
        Err(e) => return format!("cannot stage object for disassembly: {e}"),
    };
    if let Err(e) = std::fs::write(&path, object) {
        return format!("cannot write temp object: {e}");
    }
    let attempts: [(&str, &[&str]); 6] = [
        ("objdump", &["-d", "-M", "intel", "--no-show-raw-insn", "-t"]),
        ("llvm-objdump", &["-d", "--no-show-raw-insn", "--x86-asm-syntax=intel", "-t"]),
        ("objdump", &["-d", "--no-show-raw-insn"]),
        ("llvm-objdump", &["-d", "--no-show-raw-insn"]),
        ("objdump", &["-d"]),
        ("llvm-objdump", &["-d"]),
    ];
    let mut failures = String::new();
    let mut result = None;
    for (bin, args) in attempts {
        match run_disassembler(bin, args, &path) {
            Ok(text) => {
                result = Some(text);
                break;
            }
            Err(e) => failures = e,
        }
    }
    let _ = std::fs::remove_file(&path);
    match result {
        Some(text) => text,
        None if failures.is_empty() => "objdump not found; install binutils to see assembly".to_string(),
        None => failures,
    }
}

/// Run one disassembler attempt. `Err` reports the tool's message so
/// the caller can try the next one and surface the last failure.
fn run_disassembler(bin: &str, args: &[&str], path: &Path) -> Result<String, String> {
    let out = match Command::new(bin).args(args).arg(path).output() {
        Ok(out) => out,
        Err(e) => return Err(format!("{bin} failed:\n{e}")),
    };
    let stdout = String::from_utf8_lossy(&out.stdout);
    if out.status.success() && !stdout.contains("can't disassemble") {
        Ok(stdout.into_owned())
    } else {
        let stderr = String::from_utf8_lossy(&out.stderr);
        Err(format!("{bin} failed:\n{}", if stderr.trim().is_empty() { stdout.trim() } else { stderr.trim() }))
    }
}

fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else {
        "unknown panic".to_string()
    }
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_file(prefix: &str, suffix: &str) -> Result<PathBuf, String> {
    let base = std::env::temp_dir();
    for _ in 0..100 {
        let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let name = format!("{prefix}{}-{}{suffix}", std::process::id(), n);
        let path = base.join(name);
        if !path.exists() {
            return Ok(path);
        }
    }
    Err("cannot pick a temp file name".to_string())
}

/// Escape a string for embedding in JSON `"..."`.
pub fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Render an `Output` as a JSON object for the HTTP server.
pub fn output_json(o: &Output) -> String {
    let mut err = String::from("[");
    for (i, e) in o.errors.iter().enumerate() {
        if i > 0 {
            err.push(',');
        }
        err.push('"');
        err.push_str(&escape_json(e));
        err.push('"');
    }
    err.push(']');
    let diagnostics = if o.diagnostics.is_empty() {
        "{\"diagnostics\":[]}"
    } else {
        &o.diagnostics
    };
    format!(
        "{{\"success\":{},\"mir\":\"{}\",\"mir_raw\":\"{}\",\"mir_full\":\"{}\",\"mir_raw_full\":\"{}\",\"asm\":\"{}\",\"errors\":{},\"diagnostics\":{},\"object_len\":{}}}",
        o.success,
        escape_json(&o.mir),
        escape_json(&o.mir_raw),
        escape_json(&o.mir_full),
        escape_json(&o.mir_raw_full),
        escape_json(&o.asm),
        err,
        diagnostics,
        o.object_len
    )
}

/// Write helper for the server's static page.
#[allow(dead_code)]
pub fn write_all(mut w: impl Write, bytes: &[u8]) -> std::io::Result<()> {
    w.write_all(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> Options {
        Options {
            preprocess: false,
            ..Options::default()
        }
    }

    /// The asm panel degrades to a message when no disassembler is on
    /// `PATH`, so only assert on real disassembly when one is present.
    fn disassembler_available() -> bool {
        ["objdump", "llvm-objdump"]
            .iter()
            .any(|bin| Command::new(bin).arg("--version").output().map(|o| o.status.success()).unwrap_or(false))
    }

    #[test]
    fn simple_add_shows_mir_and_asm() {
        let out = compile("int add(int a, int b) { return a + b; }", &opts());
        assert!(out.success, "errors: {:?}", out.errors);
        assert!(out.mir.contains("func @add"), "mir was:\n{}", out.mir);
        assert!(out.mir_raw.contains("func @add"), "raw mir was:\n{}", out.mir_raw);
        if disassembler_available() {
            assert!(out.asm.contains("add"), "asm was:\n{}", out.asm);
        }
        assert!(out.object_len > 0);
    }

    #[test]
    fn o0_optimized_view_matches_the_raw_dump() {
        let out = compile("int add(int a, int b) { return a + b; }", &opts());
        assert!(out.success, "errors: {:?}", out.errors);
        assert!(out.mir_raw.contains("alloca"), "raw mir was:\n{}", out.mir_raw);
        assert_eq!(out.mir, out.mir_raw);
    }

    #[test]
    fn optimized_mir_is_post_pass() {
        let opts = Options { opt: OptLevel::Speed, ..opts() };
        let out = compile("int add(int a, int b) { return a + b; }", &opts);
        assert!(out.success, "errors: {:?}", out.errors);
        assert!(out.mir_raw.contains("alloca"), "raw mir was:\n{}", out.mir_raw);
        assert!(!out.mir.contains("alloca"), "optimized mir was:\n{}", out.mir);
    }

    #[test]
    fn frontend_error_is_reported_not_panicked() {
        let out = compile("int x = ;", &opts());
        assert!(!out.success);
        assert!(!out.errors.is_empty());
    }

    #[test]
    fn comments_compile_without_preprocessing() {
        let src = "// leading\nint add(int a, int b) {\n    /* sum */ return a + b; // tail\n}\n";
        let out = compile(src, &opts());
        assert!(out.success, "errors: {:?}", out.errors);
        assert!(out.mir.contains("func @add"), "mir was:\n{}", out.mir);
    }

    #[test]
    fn opt_levels_parse() {
        assert_eq!(OptLevel::from_str("0"), OptLevel::None);
        assert_eq!(OptLevel::from_str("2"), OptLevel::Speed);
        assert_eq!(OptLevel::from_str("z"), OptLevel::Size);
    }

    #[test]
    fn native_pp_expands_object_macros_without_clang() {
        let opts = Options {
            preprocess: true,
            pp_native: Some(true),
            ..Options::default()
        };
        let out = compile("#define V 41\nint x = V + 1;", &opts);
        assert!(out.success, "errors: {:?}", out.errors);
        assert!(out.mir.contains("42") || out.object_len > 0, "mir was:\n{}", out.mir);
    }

    #[test]
    fn output_json_escapes() {
        let o = Output {
            success: true,
            mir: "a\"b\n".to_string(),
            mir_raw: String::new(),
            mir_full: String::new(),
            mir_raw_full: String::new(),
            asm: String::new(),
            errors: vec![],
            diagnostics: String::new(),
            object_len: 0,
            object: Vec::new(),
        };
        let j = output_json(&o);
        assert!(j.contains("a\\\"b\\n"));
        assert!(j.contains("\"diagnostics\":{\"diagnostics\":[]}"), "{j}");
    }

    /// A frontend failure is reported through both the plain `errors`
    /// list and the machine-readable diagnostics object, so an editor
    /// gets positions without parsing the message text.
    #[test]
    fn frontend_error_carries_structured_diagnostics() {
        let out = compile("int x = ;", &opts());
        assert!(!out.success);
        assert!(out.errors.iter().any(|e| e.contains("expected expression")), "errors: {:?}", out.errors);
        assert!(out.diagnostics.contains("\"severity\":\"error\""), "diagnostics: {}", out.diagnostics);
        assert!(out.diagnostics.contains("\"line\":1"), "diagnostics: {}", out.diagnostics);
        let json = output_json(&out);
        assert!(json.contains("\"diagnostics\":{\"diagnostics\":["), "{json}");
    }

    /// Warnings ride the same diagnostics object (so the editor can show
    /// them as markers) without failing the compile.
    #[test]
    fn warnings_ride_the_diagnostics_object() {
        let out = compile("int x = sizeof(void);", &opts());
        assert!(out.success, "errors: {:?}", out.errors);
        assert!(out.diagnostics.contains("\"severity\":\"warning\""), "diagnostics: {}", out.diagnostics);
        assert!(output_json(&out).contains("\"severity\":\"warning\""), "json: {}", output_json(&out));
    }

    /// The full MIR views carry `; path:line:column` annotation on the
    /// located instructions, like the driver's `--emit-full-mir`.
    #[test]
    fn full_mir_carries_source_locations() {
        let out = compile("int add(int a, int b) {\n    return a + b;\n}\n", &opts());
        assert!(out.success, "errors: {:?}", out.errors);
        assert!(out.mir_full.contains(" ; "), "full mir was:\n{}", out.mir_full);
        assert!(out.mir_full.contains("<__glcc_source>:2:5"), "full mir was:\n{}", out.mir_full);
        assert!(!out.mir.contains(" ; "), "compact mir must stay bare:\n{}", out.mir);
        assert!(out.mir_raw_full.contains("<__glcc_source>:2:5"), "raw full mir was:\n{}", out.mir_raw_full);
    }

    #[test]
    fn compile_object_skips_disassembly_but_keeps_the_object() {
        let out = compile_object("int add(int a, int b) { return a + b; }", &opts());
        assert!(out.success, "errors: {:?}", out.errors);
        assert!(out.asm.is_empty());
        assert_eq!(out.object.len(), out.object_len);
        assert!(out.object_len > 0);
        assert!(!out.mir.is_empty());
    }

    #[test]
    fn wasm_codes_map_to_opts_and_targets() {
        assert_eq!(opt_from_code(0), OptLevel::None);
        assert_eq!(opt_from_code(1), OptLevel::Speed);
        assert_eq!(opt_from_code(2), OptLevel::Size);
        assert_eq!(target_from_code(0), FrontendTarget::X86_64);
        assert_eq!(target_from_code(1), FrontendTarget::AARCH64);
        assert_eq!(target_from_code(2), FrontendTarget::RISCV64);
    }

    #[test]
    fn every_target_code_emits_an_object() {
        for code in 0..3 {
            let opts = Options {
                target: Some(target_from_code(code)),
                ..opts()
            };
            let out = compile_object("int add(int a, int b) { return a + b; }", &opts);
            assert!(out.success, "target {code} errors: {:?}", out.errors);
            assert!(out.object_len > 0, "target {code} emitted no object");
        }
    }

    #[test]
    fn async_pipeline_lowers_coroutines() {
        let opts = Options { async_enabled: true, ..opts() };
        let src = "_Future(int) add1(int x);\n\
                   __attribute__((async)) int twice(int x) {\n\
                       int r = __builtin_await(add1(x));\n\
                       return r * 2;\n\
                   }\n";
        let out = compile(src, &opts);
        assert!(out.success, "errors: {:?}", out.errors);
        assert!(out.mir_raw.contains("await"), "raw mir was:\n{}", out.mir_raw);
        assert!(out.mir.contains("$resume"), "mir was:\n{}", out.mir);
        assert!(out.object_len > 0);
    }

    #[test]
    fn async_flag_off_rejects_futures() {
        let out = compile("_Future(int) f(void);", &opts());
        assert!(!out.success, "errors: {:?}", out.errors);
        assert!(out.errors.iter().any(|e| e.contains("-fglcc-async")), "errors: {:?}", out.errors);
    }

    #[test]
    fn async_attribute_is_inert_without_the_flag() {
        let out = compile("__attribute__((async)) int one(void) { return 1; }", &opts());
        assert!(out.success, "errors: {:?}", out.errors);
        assert!(!out.mir.contains("$resume"), "mir was:\n{}", out.mir);
    }
}
