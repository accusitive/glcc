//! `glcc` command driver: a GCC-shaped command line over the
//! frontend/backend crates.
//!
//! Parsing is a hand-rolled sequential scan rather than clap: link
//! order is significant (`a.o -lfoo b.o` must forward in argv
//! order), unknown options must warn-and-continue for makefile
//! tolerance, and `-x` is a state machine over positionals. All
//! three fight an option-grouping parser; a classifier loop gets
//! them for free, with no new dependencies.

use std::ffi::{OsStr, OsString};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::winunwind;

/// Optimization selector, re-exported from `glcc_mir` so the driver and
/// every other consumer share one definition.
pub use glcc_mir::OptLevel;

use glcc_frontend::{
    decode_source,
    lang::{LangOpts, Std},
    lexer::{Interner, Lexer},
    parse::Parser,
    sema::Sema,
    span::take_diagnostic_emitted,
    target::{Arch, Os, Target},
};

/// Preprocessor binary (`clang` unless `GLCC_PP` overrides); only used
/// when the native preprocessor is not in effect.
fn preprocessor() -> OsString {
    std::env::var_os("GLCC_PP").unwrap_or_else(|| glcc_pp::find_clang().map(PathBuf::into_os_string).unwrap_or_else(|| OsString::from("clang")))
}

/// Whether the native `glcc_pp` preprocessor is in effect for `job`:
/// `--pp=native|clang` wins when given. Otherwise the native
/// preprocessor is the default, unless `GLCC_PP` names another binary
/// or the job asks for a clang-only preprocessor mode.
fn use_native_pp(job: &Job) -> bool {
    match job.pp_native {
        Some(v) => v,
        None => native_pp_default(std::env::var_os("GLCC_PP").as_deref()) && pp_flags_native_compatible(&job.cpp),
    }
}

/// `GLCC_PP` resolution when `--pp=` is absent: unset, empty, or
/// `native` means `glcc_pp`; anything else (typically a clang path)
/// selects the `clang -E` path.
fn native_pp_default(env: Option<&OsStr>) -> bool {
    match env {
        None => true,
        Some(v) => v.is_empty() || v == OsStr::new("native"),
    }
}

/// Preprocessor-output modes the native pp does not implement: those
/// jobs keep the `clang -E` path (`-P` is handled by the caller).
fn pp_flags_native_compatible(cpp: &[OsString]) -> bool {
    const INCOMPATIBLE: [&str; 6] = ["-dM", "-dD", "-dN", "-H", "-C", "-CC"];
    !cpp.iter().any(|a| INCOMPATIBLE.iter().any(|f| a.as_os_str() == OsStr::new(f)))
}

/// Link-driver binary (`cc` unless `GLCC_CC` overrides): supplies
/// crt files and default libraries, so linking looks like gcc's.
fn link_driver() -> OsString {
    std::env::var_os("GLCC_CC").unwrap_or_else(|| OsString::from("cc"))
}

/// Whether the final link drives MSVC's `link.exe`: a Windows target
/// on a Windows host, unless `GLCC_CC` names a gcc-shaped driver.
/// Object inputs are COFF either way.
fn msvc_link(job: &Job) -> bool {
    cfg!(windows) && matches!(job.target.os, glcc_frontend::target::Os::Windows) && std::env::var_os("GLCC_CC").is_none()
}

/// MSVC linker binary; `GLCC_LD` overrides (e.g. `lld-link`), the same
/// knob `-fuse-ld` feeds on the gcc-shaped path.
fn msvc_linker() -> OsString {
    std::env::var_os("GLCC_LD").unwrap_or_else(|| msvc_tool("link"))
}

/// The MSVC librarian, for `-o *.a` bundles.
fn msvc_librarian() -> OsString {
    std::env::var_os("GLCC_AR").unwrap_or_else(|| msvc_tool("lib"))
}

/// An MSVC build tool by base name: the full path from the discovered
/// toolchain `bin` directory when one is found (the tools are not on
/// `PATH` outside a Developer Command Prompt), else the bare name.
fn msvc_tool(name: &str) -> OsString {
    if let Some(bin) = glcc_pp::msvc_dirs().and_then(|dirs| dirs.bin.as_ref()) {
        let cand = bin.join(format!("{name}.exe"));
        if cand.is_file() {
            return cand.into_os_string();
        }
    }
    OsString::from(format!("{name}.exe"))
}

/// `link.exe` command line for one final link. The MSVC CRT supplies
/// startup (`mainCRTStartup` calls `main`) and the C library, and the
/// Win32 import libraries cover common API calls (`user32` holds e.g.
/// `CharUpperW`); glcc's own codegen may still need helpers the CRT does
/// not provide (builtin setjmp/longjmp, inline quad arithmetic), which
/// the staged Win64 runtime library supplies, appended last.
fn msvc_link_command(job: &Job, objects: &dyn Fn(usize) -> OsString, out: &OsStr, asm_objects: &[PathBuf]) -> (OsString, Vec<OsString>) {
    // Library search paths: `LIB` when a Developer Command Prompt set
    // it, else the discovered MSVC + Windows SDK layout.
    let libpaths = if std::env::var_os("LIB").is_some() {
        Vec::new()
    } else {
        glcc_pp::msvc_dirs().map(|dirs| dirs.libs.clone()).unwrap_or_default()
    };
    let (linker, mut args) = msvc_link_command_with(job, objects, out, asm_objects, msvc_linker(), &libpaths);
    // The runtime stays out of `_with` (whose tests are hermetic);
    // discovery is environment-sensitive, like the linker and LIB paths.
    if wants_glcc_runtime(job) {
        match find_glcc_runtime("glccrt-win64.lib") {
            Some(rt) => args.push(rt.into_os_string()),
            None => eprintln!("glcc: warning: no glccrt-win64.lib (run `cargo build` to stage it, or set GLCC_RT); builtin setjmp/longjmp and quad arithmetic will not link"),
        }
    }
    (linker, args)
}

fn msvc_link_command_with(
    job: &Job,
    objects: &dyn Fn(usize) -> OsString,
    out: &OsStr,
    asm_objects: &[PathBuf],
    linker: OsString,
    libpaths: &[PathBuf],
) -> (OsString, Vec<OsString>) {
    let mut args: Vec<OsString> = vec![OsString::from("/nologo")];
    if job.shared {
        args.push(OsString::from("/dll"));
    }
    args.push(OsString::from(format!("/out:{}", Path::new(out).display())));
    args.push(OsString::from(if job.shared { "/subsystem:windows" } else { "/subsystem:console" }));
    let mut items = job.order.iter();
    while let Some(item) = items.next() {
        match item {
            OrderItem::Input(i) => args.push(objects(*i)),
            OrderItem::LinkArg(a) if a == "-e" || a == "-u" => {
                let prefix = if a == "-e" { "/entry:" } else { "/include:" };
                if let Some(OrderItem::LinkArg(val)) = items.next() {
                    args.push(OsString::from(format!("{prefix}{}", val.to_string_lossy())));
                }
            }
            OrderItem::LinkArg(a) => push_msvc_link_arg(&mut args, a),
        }
    }
    for obj in asm_objects {
        args.push(obj.as_os_str().to_os_string());
    }
    for l in &job.link_libs {
        push_msvc_link_arg(&mut args, l);
    }
    for dir in libpaths {
        args.push(OsString::from(format!("/libpath:{}", dir.display())));
    }
    if wants_glcc_runtime(job) {
        args.push(OsString::from("libcmt.lib"));
        args.push(OsString::from("kernel32.lib"));
        args.push(OsString::from("user32.lib"));
    }
    (linker, args)
}

/// GCC-shaped link flag → `link.exe` spelling. `-lfoo` names
/// `foo.lib` (`-l:foo.lib` is passed through exactly), `-Ldir` adds a
/// library search path, and flags the CRT already provides or that
/// only exist on ELF drop out; anything else passes through verbatim.
fn push_msvc_link_arg(args: &mut Vec<OsString>, a: &OsStr) {
    const GCC_ONLY: [&str; 21] = [
        "-static",
        "-static-pie",
        "-pie",
        "-no-pie",
        "-nopie",
        "-pthread",
        "-pthreads",
        "-s",
        "-shared",
        "-nostdlib",
        "-nostartfiles",
        "-nodefaultlibs",
        "-static-libgcc",
        "-shared-libgcc",
        "-static-libstdc++",
        "-lm",
        "-lpthread",
        "-lrt",
        "-ldl",
        "-lc",
        "-lgcc_s",
    ];
    let s = a.to_string_lossy();
    if GCC_ONLY.contains(&s.as_ref()) {
        return;
    }
    if let Some(lib) = s.strip_prefix("-l") {
        args.push(OsString::from(match lib.strip_prefix(':') {
            Some(exact) => exact.to_string(),
            None => format!("{lib}.lib"),
        }));
    } else if let Some(dir) = s.strip_prefix("-L") {
        args.push(OsString::from(format!("/libpath:{dir}")));
    } else if let Some(pieces) = s.strip_prefix("-Wl,") {
        for piece in pieces.split(',') {
            push_msvc_link_arg(args, OsStr::new(piece));
        }
    } else {
        args.push(a.to_os_string());
    }
}

/// Where the staged glcc runtime lives, by flavor file name (`glccrt.o`
/// on ELF hosts, `glccrt-win64.lib` on Windows): `$GLCC_RT` names the
/// file directly when set, otherwise `~/.local/lib/glcc/<name>` first,
/// then `/usr/lib/glcc/<name>`, where an ELF host's `make install`
/// puts it.
fn glcc_runtime_candidates(name: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(rt) = std::env::var_os("GLCC_RT") {
        if !rt.is_empty() {
            out.push(PathBuf::from(rt));
        }
    }
    if let Some(home) = home_dir() {
        out.push(home.join(".local/lib/glcc").join(name));
    }
    out.push(PathBuf::from("/usr/lib/glcc").join(name));
    out
}

/// The user's home directory: `HOME`, then `USERPROFILE` (Windows), so
/// the staged runtime is found on every host.
fn home_dir() -> Option<PathBuf> {
    for key in ["HOME", "USERPROFILE"] {
        if let Some(v) = std::env::var_os(key) {
            if !v.is_empty() {
                return Some(PathBuf::from(v));
            }
        }
    }
    None
}

/// First candidate that exists on disk, if any.
fn find_glcc_runtime(name: &str) -> Option<PathBuf> {
    glcc_runtime_candidates(name).into_iter().find(|p| p.is_file())
}

/// Whether the link should pull the glcc runtime (`glccrt.o` on ELF
/// hosts, `glccrt-win64.lib` on Windows): never under
/// `-nostdlib`/`-nostartfiles`, like the system crt files.
fn wants_glcc_runtime(job: &Job) -> bool {
    !job.order.iter().any(|item| match item {
        OrderItem::LinkArg(a) => a == "-nostdlib" || a == "-nostartfiles",
        OrderItem::Input(_) => false,
    })
}

/// Whether the staged runtime reaches libm at link time: the aarch64
/// and riscv64 quad trampolines branch to the toolchain's `sqrtl`
/// (`rt/glccrt.c`), while the x86-64 soft-float combine is
/// self-contained.
fn quad_runtime_needs_libm(arch: Arch) -> bool {
    matches!(arch, Arch::Aarch64 | Arch::Riscv64)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Compile and link (default).
    Link,
    /// `-c`: objects only.
    Compile,
    /// `-E`: preprocessed source to stdout (or `-o`).
    Preprocess,
    /// `-M`/`-MM`: dependency rules.
    Deps,
    /// `-fsyntax-only`: check only.
    SyntaxOnly,
    /// `--emit-mir`: print MIR to stdout (or `-o`).
    EmitMir,
    /// `--emit-full-mir`: like `EmitMir`, with the source location of
    /// every located instruction annotated.
    EmitFullMir,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flavor {
    /// `glcc`: C sources, no C++ support.
    C,
    /// `glcpp`: C++ sources via EDG's front end (like g++, `.c` also
    /// compiles as C++).
    Cpp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    C,
    Preprocessed,
    Assembler,
    /// C++ source translated to C by EDG before glcc's C compilation.
    Cpp,
    /// C source translated to C by EDG's C frontend (`--edg-c`).
    EdgC,
}

/// A compilable input: `.c`/`.i`/`.s`/`.S`, or anything under `-x`.
#[derive(Debug, Clone)]
pub struct Input {
    pub path: OsString,
    pub lang: Lang,
}

/// One link-line entry, in original argv order: objects substitute
/// for their source positions, flags forward where they stood.
#[derive(Debug, Clone)]
pub enum OrderItem {
    Input(usize),
    LinkArg(OsString),
}

#[derive(Debug)]
pub struct Job {
    pub action: Action,
    /// which front end this invocation targets
    pub flavor: Flavor,
    /// EDG checkout root (the submodule dir); `Cpp` needs this.
    pub edg_root: Option<PathBuf>,
    pub deps_sys_excluded: bool,
    /// `-MD` (Some(false)) / `-MMD` (Some(true)); orthogonal to action.
    pub depgen: Option<bool>,
    pub dep_file: Option<OsString>,   // `-MF`
    pub dep_target: Option<OsString>, // `-MT`
    pub dep_phony: bool,              // `-MP`
    pub dep_extra: Vec<OsString>,     // `-MQ t`, `-MG`, forwarded
    pub inputs: Vec<Input>,
    pub order: Vec<OrderItem>,
    pub output: Option<OsString>,
    /// Flags forwarded to `clang -E` (and `cc -c` for `.S`), in order.
    pub cpp: Vec<OsString>,
    /// Raw front-end arguments (includes, defines, sysroot) for cpfe.
    pub fe_args: Vec<OsString>,
    /// Libraries always appended to the final link line.
    pub link_libs: Vec<OsString>,
    /// `-Wa,` pieces, reassembled for assembler invocations.
    pub as_args: Vec<OsString>,
    pub verbose: bool,
    pub dry_run: bool, // `-###`
    pub shared: bool,
    /// `--edg-c`: route C inputs through EDG's C frontend (glcpp only).
    pub force_edg_c: bool,
    pub use_ld: Option<OsString>, // `-fuse-ld=X`
    pub relocatable: bool,        // `-r`
    /// DWARF emission, on by default like `gcc -g` workflows expect:
    /// `-g0` disables, any other `-g*` (re)enables.
    pub debug: bool,
    /// Optimization level, default `-O0` (Cranelift `none`): any `-O*`
    /// selects codegen, last wins like gcc.
    pub opt_level: OptLevel,
    /// Cranelift `is_pic`: on by default so objects can go into the
    /// PIE/shared links the system driver produces. `-fno-pic` and
    /// `-fno-pie` select direct addressing; the PIC/PIE spelling
    /// variants share one switch, last wins.
    pub pic: bool,
    /// Keep intermediate files (`-save-temps[=cwd|=obj]`): `<stem>.i`
    /// preprocessed C for every C input, plus `<stem>.int.c` raw
    /// EDG-lowered C for C++ inputs.
    pub save_temps: SaveTemps,
    /// Compilation target (`--target=`; native by default): selects the
    /// preprocessor triple, the frontend's sizes, and the Cranelift ISA.
    pub target: Target,
    /// `--pp=native` (Some(true)) / `--pp=clang` (Some(false)); None
    /// prefers `glcc_pp` (see `use_native_pp`).
    pub pp_native: Option<bool>,
    /// `-std=`/`-ansi`: the language level the frontend parses (and
    /// the preprocessor sees, via the unchanged forwarding). `None`
    /// keeps the permissive default level.
    pub lang_std: Option<Std>,
    /// `-f[no-]gnu-extensions`: GNU-family spellings. `None` keeps the
    /// fully-permissive default (`true`).
    pub ext_gnu: Option<bool>,
    /// `-f[no-]ms-extensions`: MSVC-family spellings (`__declspec`,
    /// calling conventions, `__intN`).
    pub ext_ms: Option<bool>,
    /// `-f[no-]clang-extensions`: clang-family spellings
    /// (`_Float32`/`_Float64`).
    pub ext_clang: Option<bool>,
    /// `-fglcc-async`: enables the async extension (`__attribute__((async))`,
    /// `_Future(T)`, `__builtin_await`). Off by default so no existing
    /// code can accidentally compile as coroutines.
    pub async_enabled: bool,
    /// `--error-format=json|human`: how frontend diagnostics are
    /// reported. `human` (the default) renders them through codespan to
    /// stderr; `json` prints the machine-readable form for editors.
    pub error_format: ErrorFormat,
    pub warnings: Vec<String>,
}

/// Diagnostic rendering selected by `--error-format`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ErrorFormat {
    /// codespan-rendered diagnostics on stderr (the default).
    #[default]
    Human,
    /// One JSON object with a `diagnostics` array, for editor integration.
    Json,
}

impl ErrorFormat {
    fn parse(text: &str) -> Result<Self, String> {
        match text {
            "human" => Ok(ErrorFormat::Human),
            "json" => Ok(ErrorFormat::Json),
            _ => Err(format!("invalid argument to '--error-format' (expected 'human' or 'json')")),
        }
    }
}

/// Where `-save-temps` files go: `Off` discards them (the default),
/// `Cwd` writes them to the working directory (plain `-save-temps`
/// and `-save-temps=cwd`), `Obj` beside each `-c` object output
/// (`-save-temps=obj`; without persistent objects, e.g. link flow,
/// this falls back to the working directory).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SaveTemps {
    #[default]
    Off,
    Cwd,
    Obj,
}

/// The frontend's dialect options for one job: `-std=`/`-ansi` and the
/// `-f[no-]{gnu,ms,clang}-extensions` toggles on top of the permissive
/// default. (The glcc async extension is gated separately by
/// `Job::async_enabled` / `-fglcc-async`.)
pub fn language(job: &Job) -> LangOpts {
    let mut lang = LangOpts::default();
    if let Some(std) = job.lang_std {
        lang.std = std;
    }
    if let Some(gnu) = job.ext_gnu {
        lang.gnu = gnu;
    }
    if let Some(ms) = job.ext_ms {
        lang.ms = ms;
    }
    if let Some(clang) = job.ext_clang {
        lang.clang = clang;
    }
    lang
}

impl Default for Job {
    fn default() -> Self {
        Job {
            action: Action::Link,
            flavor: Flavor::C,
            edg_root: None,
            deps_sys_excluded: false,
            depgen: None,
            dep_file: None,
            dep_target: None,
            dep_phony: false,
            dep_extra: Vec::new(),
            inputs: Vec::new(),
            order: Vec::new(),
            output: None,
            cpp: Vec::new(),
            fe_args: Vec::new(),
            link_libs: Vec::new(),
            as_args: Vec::new(),
            verbose: false,
            dry_run: false,
            shared: false,
            force_edg_c: false,
            use_ld: None,
            relocatable: false,
            debug: true,
            opt_level: OptLevel::None,
            pic: true,
            save_temps: SaveTemps::Off,
            target: Target::native(),
            lang_std: None,
            ext_gnu: None,
            ext_ms: None,
            ext_clang: None,
            pp_native: None,
            async_enabled: false,
            error_format: ErrorFormat::Human,
            warnings: Vec::new(),
        }
    }
}

/// The extra runtime libraries a flavor always links when producing
/// executables (on top of the user's `-l` list and system defaults).
/// `Cpp` mirrors g++/clang++, which append `-lstdc++ -lm`.
fn default_link_libs(flavor: Flavor) -> Vec<OsString> {
    match flavor {
        Flavor::C => Vec::new(),
        Flavor::Cpp => vec![OsString::from("-lstdc++"), OsString::from("-lm"), OsString::from("-lgcc_s"), OsString::from("-lpthread")],
    }
}

impl Job {
    fn warn(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }

    /// `--target=<triple>` forwarding for clang/cc invocations: empty
    /// when building native, so native command lines are byte-identical
    /// to before. Cross linking additionally needs a matching
    /// `GLCC_CC` (a native `cc` cannot link foreign objects).
    fn target_args(&self) -> Vec<OsString> {
        if self.target.is_native() {
            Vec::new()
        } else {
            vec![OsString::from(format!("--target={}", self.target.triple()))]
        }
    }

    /// Record a compilable input, tracking its link-line position.
    fn push_compilable(&mut self, path: OsString, lang: Lang) {
        let idx = self.inputs.len();
        self.inputs.push(Input { path, lang });
        self.order.push(OrderItem::Input(idx));
    }
}

/// Split `arg` after an ASCII `prefix`: the attached remainder, or
/// the next argv element. Errors when neither exists.
fn attached_or_next(args: &[OsString], pos: &mut usize, prefix_len: usize, flag: &str) -> Result<OsString, String> {
    let raw = args[*pos].as_encoded_bytes();
    if raw.len() > prefix_len {
        return Ok(strip_prefix(&args[*pos], prefix_len));
    }
    *pos += 1;
    args.get(*pos).cloned().ok_or_else(|| format!("missing argument to '{flag}'"))
}

#[cfg(unix)]
fn strip_prefix(arg: &OsStr, n: usize) -> OsString {
    use std::os::unix::ffi::{OsStrExt, OsStringExt};
    OsString::from_vec(arg.as_bytes()[n..].to_vec())
}

#[cfg(not(unix))]
fn strip_prefix(arg: &OsStr, n: usize) -> OsString {
    OsString::from(arg.to_string_lossy()[n..].to_string())
}

/// Value-taking flags. Order matters only where one flag prefixes
/// another (`-iwithprefixbefore` before `-iwithprefix`): an attached
/// remainder (`-oout`) resolves exactly like the separate spelling
/// (`-o out`), and both spellings end up in `handle_valued`.
const VALUED_FLAGS: &[&str] = &[
    "-Xlinker",
    "-Xpreprocessor",
    "-iwithprefixbefore",
    "-include",
    "-imacros",
    "-isystem",
    "-idirafter",
    "-iquote",
    "-iwithprefix",
    "-iprefix",
    "-MF",
    "-MT",
    "-MQ",
    "-o",
    "-x",
    "-e",
    "-u",
    "-z",
    "-T",
    "-l",
    "-L",
    "-I",
    "-D",
    "-U",
];

/// Which valued flag `s` names, if any: bare or attached.
fn split_valued(s: &str) -> Option<&'static str> {
    VALUED_FLAGS.iter().find(|f| s == **f || s.starts_with(**f)).copied()
}

/// Apply one valued flag with its already-resolved value.
fn handle_valued(job: &mut Job, lang: &mut Option<Lang>, flag: &str, val: OsString) -> Result<(), String> {
    match flag {
        "-o" => job.output = Some(val),
        "-x" => *lang = parse_lang(&val, job.flavor)?,
        "-e" | "-u" | "-z" | "-T" => {
            job.order.push(OrderItem::LinkArg(OsString::from(flag)));
            job.order.push(OrderItem::LinkArg(val));
        }
        "-Xlinker" => {
            job.order.push(OrderItem::LinkArg(OsString::from("-Xlinker")));
            job.order.push(OrderItem::LinkArg(val));
        }
        "-Xpreprocessor" => job.cpp.push(val),
        "-MF" => job.dep_file = Some(val),
        "-MT" => job.dep_target = Some(val),
        "-MQ" => {
            job.dep_extra.push(OsString::from("-MQ"));
            job.dep_extra.push(val);
        }
        "-include" | "-imacros" | "-isystem" | "-idirafter" | "-iquote" | "-iprefix" | "-iwithprefix" | "-iwithprefixbefore" => {
            job.cpp.push(OsString::from(flag));
            job.cpp.push(val);
        }
        "-l" | "-L" => {
            let mut joined = OsString::from(flag);
            joined.push(val);
            job.order.push(OrderItem::LinkArg(joined));
        }
        "-I" | "-D" | "-U" => {
            // `-I` values forwarded to cpfe must be absolute: the
            // front end is invoked with a scratch cwd so the `.ti`
            // it writes to cwd stays out of the user's directory.
            let val = if flag == "-I" {
                Path::new(&val).canonicalize().map(|p| p.as_os_str().to_os_string()).unwrap_or(val)
            } else {
                val
            };
            let mut joined = OsString::from(flag);
            joined.push(&val);
            job.cpp.push(joined.clone());
            job.fe_args.push(joined);
        }
        _ => unreachable!("splitter only yields valued flags"),
    }
    Ok(())
}

/// Classify a path by extension. C++ spellings are rejected outright
/// for the C flavor (there is no C++ frontend); in the Cpp flavor they
/// route to EDG and plain `.c` compiles as C++ (like g++), or as C
/// through EDG's C frontend under `--edg-c`.
fn classify_extension(path: &OsStr, flavor: Flavor, force_edg_c: bool) -> Result<FileKind, String> {
    let name = Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let ext = Path::new(path).extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
    match ext.as_str() {
        "c" => Ok(FileKind::Source(match (flavor, force_edg_c) {
            (Flavor::C, _) => Lang::C,
            (Flavor::Cpp, true) => Lang::EdgC,
            // g++ compiles `.c` as C++.
            (Flavor::Cpp, false) => Lang::Cpp,
        })),
        "i" => Ok(FileKind::Source(match (flavor, force_edg_c) {
            (Flavor::Cpp, true) => Lang::EdgC,
            _ => Lang::Preprocessed,
        })),
        "s" | "S" => Ok(FileKind::Source(Lang::Assembler)),
        "o" | "obj" | "lo" => Ok(FileKind::LinkObject),
        "a" => Ok(FileKind::LinkArchive),
        "so" => Ok(FileKind::LinkShared),
        "C" | "cc" | "cpp" | "cxx" | "c++" | "CPP" | "cp" | "ii" | "hh" | "hpp" | "hxx" | "h" => match flavor {
            Flavor::C => Err(format!("C++ input '{}' is not supported", Path::new(path).display())),
            Flavor::Cpp => Ok(FileKind::Source(Lang::Cpp)),
        },
        _ if name.contains(".so.") => Ok(FileKind::LinkShared),
        _ => Err(format!("unrecognized file extension for '{}'; use -x to specify its language", Path::new(path).display())),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileKind {
    Source(Lang),
    LinkObject,
    LinkArchive,
    LinkShared,
}

fn push_path(job: &mut Job, path: OsString, forced: Option<Lang>) -> Result<(), String> {
    if let Some(lang) = forced {
        // `-x c` names the C language; under `--edg-c` that means
        // EDG's C frontend rather than the native pipeline.
        let lang = match (lang, job.flavor, job.force_edg_c) {
            (Lang::C, Flavor::Cpp, true) => Lang::EdgC,
            _ => lang,
        };
        job.push_compilable(path, lang);
        return Ok(());
    }
    match classify_extension(&path, job.flavor, job.force_edg_c)? {
        FileKind::Source(lang) => job.push_compilable(path, lang),
        FileKind::LinkObject | FileKind::LinkArchive | FileKind::LinkShared => job.order.push(OrderItem::LinkArg(path)),
    }
    Ok(())
}

/// Parse one `-x` language name.
fn parse_lang(name: &OsStr, flavor: Flavor) -> Result<Option<Lang>, String> {
    match name.to_str() {
        Some("c") => Ok(Some(Lang::C)),
        Some("cpp-output") => Ok(Some(Lang::Preprocessed)),
        Some("assembler") => Ok(Some(Lang::Assembler)),
        Some("none") => Ok(None),
        Some("c++") => match flavor {
            Flavor::C => Err("C++ input is not supported".into()),
            Flavor::Cpp => Ok(Some(Lang::Cpp)),
        },
        Some("c-header" | "c++-header") => Err("precompiled headers are not supported".into()),
        _ => Err(format!("language not recognized: '{}'", name.to_string_lossy())),
    }
}

/// Parse argv (already response-file-expanded) into a `Job`.
/// Unknown `-...` options warn and are dropped; `-o` is last-wins;
/// the last of `-c`/`-E`/`-M`/`-MM`/`-fsyntax-only`/`--emit-mir`/
/// `--emit-full-mir` picks the action.
pub fn parse_args(args: &[OsString]) -> Result<Job, String> {
    parse_args_flavor(args, Flavor::C)
}

pub fn parse_args_flavor(args: &[OsString], flavor: Flavor) -> Result<Job, String> {
    let mut job = Job {
        flavor,
        link_libs: default_link_libs(flavor),
        ..Job::default()
    };
    // `--edg-c` is position-independent (like `-o`'s last-wins, not
    // like `-x`'s running state): pre-scan so a `.c` before the flag
    // classifies the same as one after it. Stops at `--`, where
    // everything after is a filename.
    job.force_edg_c = args.iter().take_while(|a| a.as_os_str() != "--").any(|a| a.as_os_str() == "--edg-c");
    if job.force_edg_c && flavor == Flavor::C {
        return Err("--edg-c requires glcpp (EDG is not linked into glcc)".into());
    }
    let mut lang: Option<Lang> = None;
    let mut end_opts = false;
    let mut pos = 0;
    while pos < args.len() {
        let arg = &args[pos];
        if end_opts {
            push_path(&mut job, arg.clone(), lang)?;
            pos += 1;
            continue;
        }
        if arg == "--" {
            end_opts = true;
            pos += 1;
            continue;
        }
        let raw = arg.as_encoded_bytes();
        if raw == b"-" {
            return Err("reading from standard input is not supported; pass a file".into());
        }
        if !(raw.len() > 1 && raw[0] == b'-') {
            push_path(&mut job, arg.clone(), lang)?;
            pos += 1;
            continue;
        }
        let s = arg.to_string_lossy();
        match s.as_ref() {
            "-c" => job.action = Action::Compile,
            "-E" => job.action = Action::Preprocess,
            "-S" => return Err("-S (assembly output) is not supported by glcc".into()),
            "-M" => {
                job.action = Action::Deps;
                job.deps_sys_excluded = false;
            }
            "-MM" => {
                job.action = Action::Deps;
                job.deps_sys_excluded = true;
            }
            "-MD" => job.depgen = Some(false),
            "-MMD" => job.depgen = Some(true),
            "-MP" => job.dep_phony = true,
            "-MG" => job.dep_extra.push(OsString::from("-MG")),
            "-fsyntax-only" => job.action = Action::SyntaxOnly,
            "--emit-mir" => job.action = Action::EmitMir,
            "--emit-full-mir" => job.action = Action::EmitFullMir,
            // Accepted by the pre-scan above (which also rejects it
            // for the C flavor); nothing position-dependent to do here.
            "--edg-c" => {}
            "-v" | "--verbose" => job.verbose = true,
            "-###" | "--dry-run" => {
                job.dry_run = true;
                job.verbose = true;
            }
            "-static" => job.order.push(OrderItem::LinkArg(arg.clone())),
            "-shared" => {
                job.shared = true;
                job.order.push(OrderItem::LinkArg(arg.clone()));
            }
            "-static-pie" | "-pie" | "-no-pie" | "-nopie" | "-pthread" | "-pthreads" | "-s" | "-r" | "--relocatable" | "-nostdlib" | "-nostartfiles" | "-nodefaultlibs"
            | "-static-libgcc" | "-shared-libgcc" | "-static-libstdc++" => {
                if s.as_ref() == "-r" || s.as_ref() == "--relocatable" {
                    job.relocatable = true;
                }
                job.order.push(OrderItem::LinkArg(arg.clone()));
            }
            "-undef" | "-nostdinc" | "-nostdinc++" | "-H" | "-C" | "-CC" | "-P" | "-dM" | "-dD" | "-dN" | "-trigraphs" | "-no-trigraphs" => job.cpp.push(arg.clone()),
            "-pedantic" | "-pedantic-errors" | "-w" | "-pipe" | "-time" | "-Q" => {}
            "-o" | "-x" | "-e" | "-u" | "-z" | "-T" | "-Xlinker" | "-Xpreprocessor" | "-MF" | "-MT" | "-MQ" | "-include" | "-imacros" | "-isystem" | "-idirafter" | "-iquote"
            | "-l" | "-L" | "-I" | "-D" | "-U" => {
                let flag = s.into_owned();
                let val = attached_or_next(args, &mut pos, flag.len(), &flag)?;
                handle_valued(&mut job, &mut lang, &flag, val)?;
            }
            _ => {
                // `-save-temps[=cwd|=obj]`: keep intermediates (last wins).
                if s == "-save-temps" || s == "-save-temps=cwd" {
                    job.save_temps = SaveTemps::Cwd;
                } else if s == "-save-temps=obj" {
                    job.save_temps = SaveTemps::Obj;
                } else if s.starts_with("-save-temps") {
                    return Err("invalid argument to '-save-temps' (expected 'cwd' or 'obj')".into());
                } else if let Some(flag) = split_valued(&s) {
                    handle_valued(&mut job, &mut lang, flag, strip_prefix(arg, flag.len()))?;
                } else if let Some(rest) = s.strip_prefix("--sysroot=") {
                    job.cpp.push(OsString::from(format!("--sysroot={rest}")));
                    job.order.push(OrderItem::LinkArg(OsString::from(format!("--sysroot={rest}"))));
                } else if s.starts_with("--param") {
                    // Nothing to tune, but a separate-spelling value must
                    // still be consumed so it is not read as an input file.
                    if !s.contains('=') {
                        let _ = attached_or_next(args, &mut pos, s.len(), "--param")?;
                    }
                } else if let Some(rest) = s.strip_prefix("--target=") {
                    match Target::from_name(rest) {
                        Some(t) => job.target = t,
                        None => return Err(format!("unsupported target '{rest}' (expected e.g. x86_64, aarch64, riscv64, or x86_64-windows)")),
                    }
                } else if s == "--target" {
                    let val = attached_or_next(args, &mut pos, s.len(), "--target")?;
                    let text = val.to_string_lossy().into_owned();
                    match Target::from_name(&text) {
                        Some(t) => job.target = t,
                        None => return Err(format!("unsupported target '{text}' (expected e.g. x86_64, aarch64, riscv64, or x86_64-windows)")),
                    }
                } else if s == "--pp" {
                    let val = attached_or_next(args, &mut pos, s.len(), "--pp")?;
                    let text = val.to_string_lossy().into_owned();
                    match text.as_str() {
                        "native" => job.pp_native = Some(true),
                        "clang" => job.pp_native = Some(false),
                        _ => return Err(format!("invalid argument to '--pp' (expected 'native' or 'clang')")),
                    }
                } else if let Some(rest) = s.strip_prefix("--pp=") {
                    match rest {
                        "native" => job.pp_native = Some(true),
                        "clang" => job.pp_native = Some(false),
                        _ => return Err(format!("invalid argument to '--pp' (expected 'native' or 'clang')")),
                    }
                } else if s == "--error-format" {
                    let val = attached_or_next(args, &mut pos, s.len(), "--error-format")?;
                    job.error_format = ErrorFormat::parse(&val.to_string_lossy())?;
                } else if let Some(rest) = s.strip_prefix("--error-format=") {
                    job.error_format = ErrorFormat::parse(rest)?;
                } else if s.starts_with("-std=") || s == "-ansi" {
                    if s == "-ansi" {
                        // `-ansi` means C90, like gcc.
                        job.lang_std = Some(Std::C89);
                    } else {
                        let value = s.strip_prefix("-std=").expect("checked prefix");
                        match Std::parse(value) {
                            Some(std) => job.lang_std = Some(std),
                            None => return Err(format!("invalid argument to '-std=' (unrecognized standard '{value}')")),
                        }
                    }
                    // The preprocessor still sees the flag itself
                    // (`__STDC_VERSION__`, its include search).
                    job.cpp.push(arg.clone());
                } else if s == "-O0" {
                    job.opt_level = OptLevel::None;
                } else if s == "-Os" || s == "-Oz" {
                    job.opt_level = OptLevel::Size;
                } else if s.starts_with("-O") || s == "-G" {
                    // `-O1`/`-O2`/`-O3`/`-Og`/`-Ofast`/bare `-O` (and
                    // legacy `-G`): optimized. Last `-O*` wins.
                    job.opt_level = OptLevel::Speed;
                } else if s.starts_with("-g") {
                    // Debug knobs: `-g` (any level but `-g0`) emits
                    // DWARF; `-g0` disables it again (last wins, like gcc).
                    if s == "-g0" {
                        job.debug = false;
                    } else {
                        job.debug = true;
                    }
                } else if let Some(rest) = s.strip_prefix("-Wl,") {
                    // Before `-W` below: `-Wl,...` starts with `-W`.
                    let _ = rest;
                    job.order.push(OrderItem::LinkArg(arg.clone()));
                } else if let Some(rest) = s.strip_prefix("-Wa,") {
                    if !rest.is_empty() {
                        job.as_args.push(OsString::from(rest));
                    }
                } else if let Some(rest) = s.strip_prefix("-Wp,") {
                    // `-Wp,-MD,foo.d` sets depfile generation like
                    // `-MD -MF foo.d`; anything else forwards comma-split.
                    let parts: Vec<&str> = rest.split(',').collect();
                    match parts.as_slice() {
                        ["-MD", file] | ["-MMD", file] => {
                            job.depgen = Some(parts[0] == "-MMD");
                            job.dep_file = Some(OsString::from(*file));
                        }
                        _ => {
                            for p in parts {
                                job.cpp.push(OsString::from(p));
                            }
                        }
                    }
                } else if s.starts_with("-W") {
                    // Diagnostics we never emit need no configuration.
                } else if let Some(rest) = s.strip_prefix("-fuse-ld=") {
                    job.use_ld = Some(OsString::from(rest));
                    job.order.push(OrderItem::LinkArg(arg.clone()));
                } else if s.starts_with("-flto") {
                    job.warn("link-time optimization is not supported; continuing without it");
                } else if s.starts_with("-fsanitize") {
                    job.warn(format!("sanitizers are not supported; ignoring '{s}'"));
                } else if s.starts_with("-fprofile") || s == "-ftest-coverage" || s == "--coverage" {
                    job.warn(format!("coverage/profiling instrumentation is not supported; ignoring '{s}'"));
                } else if s == "-pg" || s == "-p" {
                    job.warn(format!("profiling instrumentation is not supported; ignoring '{s}'"));
                } else if s == "-fglcc-async" {
                    job.async_enabled = true;
                } else if s == "-fgnu-extensions" {
                    job.ext_gnu = Some(true);
                } else if s == "-fno-gnu-extensions" {
                    job.ext_gnu = Some(false);
                } else if s == "-fms-extensions" {
                    job.ext_ms = Some(true);
                } else if s == "-fno-ms-extensions" {
                    job.ext_ms = Some(false);
                } else if s == "-fclang-extensions" {
                    job.ext_clang = Some(true);
                } else if s == "-fno-clang-extensions" {
                    job.ext_clang = Some(false);
                } else if matches!(s.as_ref(), "-fpic" | "-fPIC" | "-fpie" | "-fPIE") {
                    // Cranelift has one PIC mode: PIE is PIC without
                    // symbol preemption, which this backend does not
                    // model separately.
                    job.pic = true;
                    // Also lets clang -E define __PIC__/__PIE__ to match.
                    job.cpp.push(arg.clone());
                } else if matches!(s.as_ref(), "-fno-pic" | "-fno-PIC" | "-fno-pie" | "-fno-PIE") {
                    job.pic = false;
                    job.cpp.push(arg.clone());
                } else if s.starts_with("-f") {
                    // Spelling and machine-independent codegen flags:
                    // nothing to configure.
                } else if s == "-m32" || s == "-mx32" {
                    return Err("32-bit targets are not supported; glcc always builds LP64".into());
                } else if s.starts_with("-m") {
                    // `-m64`, `-march=`, `-mtune=`: the target is
                    // fixed native 64-bit.
                } else if s == "-B" || s.starts_with("-B") || s.starts_with("-V") || s.starts_with("-specs") {
                    // Linker search paths, version selection, specs
                    // files: nothing to honor.
                } else if s.starts_with("-A") || s.starts_with("-d") {
                    job.warn(format!("unrecognized option '{s}'; ignoring"));
                } else {
                    job.warn(format!("unrecognized option '{s}'; ignoring"));
                }
            }
        }
        pos += 1;
    }
    Ok(job)
}

/// Split response-file text like gcc: whitespace-separated,
/// honoring single/double quotes and backslash escapes.
fn split_response(text: &str) -> Vec<OsString> {
    let mut out = Vec::new();
    let mut cur = OsString::new();
    let mut quote: Option<char> = None;
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        match (quote, c) {
            (None, '\'') | (None, '"') => quote = Some(c),
            (Some(q), c) if c == q => quote = None,
            (None, c) if c.is_whitespace() => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            (_, '\\') => {
                if let Some(e) = chars.next() {
                    cur.push(e.to_string());
                }
            }
            _ => cur.push(c.to_string()),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Expand `@file` arguments in place (depth-capped against cycles).
pub fn expand_response_files(args: Vec<OsString>, read: &dyn Fn(&Path) -> std::io::Result<String>) -> Result<Vec<OsString>, String> {
    fn expand(args: Vec<OsString>, read: &dyn Fn(&Path) -> std::io::Result<String>, depth: usize, out: &mut Vec<OsString>) -> Result<(), String> {
        if depth > 16 {
            return Err("response files nested too deeply".into());
        }
        for arg in args {
            let raw = arg.as_encoded_bytes();
            if raw.starts_with(b"@") && raw.len() > 1 {
                let name = strip_prefix(&arg, 1);
                let path = Path::new(&name);
                let text = read(path).map_err(|e| format!("cannot read response file '{}': {e}", path.display()))?;
                expand(split_response(&text), read, depth + 1, out)?;
            } else {
                out.push(arg);
            }
        }
        Ok(())
    }
    let mut out = Vec::with_capacity(args.len());
    expand(args, read, 0, &mut out)?;
    Ok(out)
}

// ---------- execution ----------

/// Driver failure: a message for the user, or silence (a frontend
/// diagnostic or the panic hook already said everything).
pub enum Fail {
    Msg(String),
    Quiet,
}

type Outcome<T> = std::result::Result<T, Fail>;

fn fail(msg: impl Into<String>) -> Fail {
    Fail::Msg(msg.into())
}

struct Ctx {
    verbose: bool,
    dry_run: bool,
}

fn quote_arg(arg: &OsStr) -> String {
    let s = arg.to_string_lossy();
    if s.is_empty()
        || s.chars()
            .any(|c| matches!(c, ' ' | '\t' | '"' | '\'' | '\\' | '$' | '`' | '!' | '(' | ')' | '&' | ';' | '<' | '>' | '|' | '*'))
    {
        format!("'{}'", s.replace('\'', "'\\''"))
    } else {
        s.into_owned()
    }
}

fn fmt_cmd(prog: &OsStr, args: &[OsString]) -> String {
    let mut out = quote_arg(prog);
    for a in args {
        out.push(' ');
        out.push_str(&quote_arg(a));
    }
    out
}

/// Run a subcommand, echoing it first under `-v`/`-###`.
/// `Ok(None)` means dry run: printed only.
fn run_cmd(ctx: &Ctx, prog: &OsStr, args: &[OsString]) -> std::io::Result<Option<std::process::Output>> {
    if ctx.verbose || ctx.dry_run {
        eprintln!("+ {}", fmt_cmd(prog, args));
    }
    if ctx.dry_run {
        return Ok(None);
    }
    Command::new(prog).args(args).output().map(Some)
}

/// A private scratch dir for this invocation's temporaries,
/// removed on drop (best effort).
struct TempDir {
    path: PathBuf,
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

impl TempDir {
    fn create() -> std::io::Result<TempDir> {
        let base = std::env::temp_dir();
        for _ in 0..100 {
            let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = base.join(format!("glcc-{}-{n}", std::process::id()));
            match std::fs::create_dir(&dir) {
                Ok(()) => return Ok(TempDir { path: dir }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
        Err(std::io::Error::new(std::io::ErrorKind::AlreadyExists, "cannot create temporary directory"))
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Preprocess one file: `clang -E [--target=T] <cpp...> <input>`, stdout captured.
fn preprocess_cmd(job: &Job, input: &OsStr) -> (OsString, Vec<OsString>) {
    let mut args = vec![OsString::from("-E")];
    args.extend(job.target_args());
    args.extend(job.cpp.iter().cloned());
    args.push(input.to_os_string());
    (preprocessor(), args)
}

/// Last `-std=`/`-ansi` in the preprocessor flags, for the native
/// predefine seed (`-std` changes `__STDC_VERSION__` and friends).
fn pp_std_flag(job: &Job) -> Option<String> {
    let mut found = None;
    for a in &job.cpp {
        let s = a.to_string_lossy();
        if s.starts_with("-std=") || s == "-ansi" {
            found = Some(s.into_owned());
        }
    }
    found
}

fn run_preprocess(ctx: &Ctx, job: &Job, input: &OsStr) -> Outcome<Vec<u8>> {
    if use_native_pp(job) {
        if ctx.dry_run {
            return Ok(Vec::new());
        }
        let bytes = std::fs::read(input).map_err(|e| fail(format!("cannot read '{}': {e}", Path::new(input).display())))?;
        let text = decode_source(&bytes);
        let mut opts = glcc_pp::cpp_flags_to_opts(&job.cpp, &input.to_string_lossy());
        opts.linemarkers = !job.cpp.iter().any(|a| a == "-P");
        opts.target = Some(job.target);
        let std = pp_std_flag(job);
        opts.std = std.clone();
        // GCC drops all builtins under `-undef`; otherwise seed the
        // native preprocessor with exactly what `clang -E` predefines.
        if !job.cpp.iter().any(|a| a == "-undef") {
            let mut predefined = glcc_pp::predefines(Some(job.target.triple()), std.as_deref());
            if !job.pic {
                predefined.retain(|(name, _)| !matches!(name.as_str(), "__PIC__" | "__pic__" | "__PIE__" | "__pie__"));
            }
            opts.predefined = predefined;
        }
        // The default system search (resource headers, libc) comes from
        // the same clang seed; `-nostdinc` keeps only the `-I` paths.
        if !opts.no_std_includes {
            let native = glcc_pp::native_include_paths(Some(job.target.triple()), opts.std.as_deref());
            if job.target.os == Os::Windows {
                opts.clang_resource_dir = glcc_pp::clang_resource_dir(&native);
            }
            opts.include_paths.extend(native);
        }
        let (text, warnings) = glcc_pp::preprocess_with_warnings(&text, &input.to_string_lossy(), &opts)
            .map_err(|e| fail(format!("preprocessing '{}' failed:\n{e}", Path::new(input).display())))?;
        for warning in warnings {
            eprintln!("{warning}");
        }
        return Ok(text.into_bytes());
    }
    let (prog, args) = preprocess_cmd(job, input);
    let out = run_cmd(ctx, &prog, &args).map_err(|e| fail(format!("cannot run preprocessor '{}': {e}", prog.to_string_lossy())))?;
    let Some(out) = out else {
        return Ok(Vec::new());
    };
    if !out.status.success() {
        return Err(fail(format!(
            "preprocessing '{}' failed:\n{}",
            Path::new(input).display(),
            String::from_utf8_lossy(&out.stderr).trim_end()
        )));
    }
    Ok(out.stdout)
}

/// Frontend and optimization pipeline: preprocessed C in, verified
/// MIR plus the per-TU assembler text out. Frontend diagnostics are
/// accumulated by the parser and reported (codespan or JSON, per
/// `error_format`) before returning `Err`; a later codegen/verification
/// panic is caught by the callers and reported as `Quiet`.
fn lower_source_to_mir(src: &str, flavor: Flavor, target: Target, opt_level: OptLevel, async_enabled: bool, lang: LangOpts, error_format: ErrorFormat) -> std::result::Result<(glcc_mir::Module, String), ()> {
    // The driver owns the interner, the lexer output, and the
    // sema: the lexer fills spellings, the parser borrows all of
    // it, and `(sema, interner)` thread on into MIR lowering.
    let mut interner = Interner::new();
    let mut sema = Sema::with_target(target);
    sema.lang = lang;
    sema.async_enabled = async_enabled;
    // C++ `inline` has vague linkage: EDG lowers every C++ inline
    // function to a GNU89 `extern inline` definition, which must
    // emit weak or multi-TU links fail on duplicates. Plain C
    // keeps GNU89 strong-standalone semantics.
    sema.weak_extern_inline = flavor == Flavor::Cpp;
    let tokens = Lexer::new(src, &mut interner).lex_all();
    let mut parser = Parser::new(src, tokens, &mut sema, &mut interner);
    // `json` consumers render `Parser::diagnostics_json` themselves.
    parser.emit_human = error_format == ErrorFormat::Human;
    let tu = parser.parse_translation_unit();
    let has_errors = parser.has_errors();
    if error_format == ErrorFormat::Json && (!parser.diagnostics().is_empty()) {
        // Warnings are useful even when compilation succeeds.
        eprintln!("{}", parser.diagnostics_json());
    }
    if has_errors {
        return Err(());
    }
    drop(parser);
    // File-scope asm becomes assembler text for the per-TU `.s`
    // pipeline: the Cranelift object cannot carry raw text.
    let mut asm_text = String::new();
    for (template, _) in &sema.toplevel_asm {
        asm_text.push_str(template);
        asm_text.push('\n');
    }
    asm_text.push_str(&glcc_lower_mir::naked_asm_text(&sema, &interner, &tu));
    let mut mir_module = glcc_lower_mir::lower_translation_unit(&sema, &interner, &tu);
    // Coroutines become start/resume pairs before any optimization,
    // and regardless of the opt level: `await` has no lowering of
    // its own, and the pass is inert without `-fglcc-async` (no
    // function carries a coroutine flag then).
    glcc_mir::pass::PassManager::new()
        .add_module_pass(glcc_mir::coroutine::LowerCoroutines)
        .run(&mut mir_module);
    // Optimization pipeline (optimized builds only). The pass list
    // lives in `glcc_mir::optimize`, shared by every consumer; `-O0`
    // runs no MIR passes for best debug fidelity.
    glcc_mir::optimize(&mut mir_module, opt_level);
    let errors = glcc_mir::verify::verify_module(&mir_module);
    if !errors.is_empty() {
        for e in &errors {
            eprintln!("glcc: mir error: {e}");
        }
        panic!("mir verification failed");
    }
    Ok((mir_module, asm_text))
}

/// Compile one preprocessed TU to object bytes through the new
/// pipeline: frontend check, MIR lowering and verification, then
/// Clif lowering to a native (or `--target=`) object. Frontend
/// diagnostics are reported by `lower_source_to_mir`; any panic here
/// (codegen or MIR verification) is `Quiet`: the panic hook or the
/// rendered diagnostic already spoke.
fn compile_source(src: &str, flavor: Flavor, target: Target, debug: bool, pic: bool, opt_level: OptLevel, async_enabled: bool, lang: LangOpts, error_format: ErrorFormat) -> std::result::Result<(Vec<u8>, String), ()> {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (mir_module, mut asm_text) = lower_source_to_mir(src, flavor, target, opt_level, async_enabled, lang, error_format)?;
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
        flag_builder.set("opt_level", opt_level.flag()).expect("known opt level");
        flag_builder.set("is_pic", if pic { "true" } else { "false" }).expect("pic");
        flag_builder.set("preserve_frame_pointers", "true").expect("frame pointers");
        // Cranelift's MachInst verifier is a dev-time safety net; our own
        // MIR verifier still runs in every build.
        flag_builder.set("enable_verifier", if cfg!(debug_assertions) { "true" } else { "false" }).expect("verifier");
        // TLS access (`tls_value` for the x86-64 foreign-variadic
        // trampoline) has no lowering unless a model is selected;
        // `elf_gd` lowers on all supported ISAs, like the legacy
        // backend.
        flag_builder.set("tls_model", "elf_gd").expect("tls model");
        // MSVC has no 128-bit float and no i128: Win64 passes `F128`
        // (indirectly) and splits `I128` the way LLVM does. Our own
        // runtime is compiled with the same setting, so internal
        // `__glcc_*` calls agree by construction.
        if matches!(target.os, glcc_frontend::target::Os::Windows) {
            flag_builder.set("enable_llvm_abi_extensions", "true").expect("llvm abi extensions");
        }
        let flags = cranelift_codegen::settings::Flags::new(flag_builder);
        let isa: cranelift_codegen::isa::OwnedTargetIsa = if target.is_native() {
            cranelift_native::builder().expect("native isa").finish(flags).expect("isa flags")
        } else {
            cranelift_codegen::isa::lookup(target.triple().parse().expect("known target triple"))
                .expect("known isa")
                .finish(flags)
                .expect("isa flags")
        };
        let mut builder = cranelift_object::ObjectBuilder::new(isa, "glcc", cranelift_module::default_libcall_names()).expect("object builder");
        // Always emit `.eh_frame` so backtraces work with or without
        // `-g`: the DWARF variable locations are CFA-relative, so gdb
        // needs the CFI to resolve them too.
        builder.unwind_info(true);
        let mut obj = cranelift_object::ObjectModule::new(builder);
        let mut unwind = Vec::new();
        Ok(if debug {
            let (_, lowered, narrow) = glcc_lower_cl::lower_module_with_debug(&mir_module, &mut obj, cl_target, &mut asm_text, &mut unwind);
            let mut product = obj.finish();
            if let Err(e) = winunwind::emit(&mut product, &unwind) {
                eprintln!("glcc: warning: {e}");
            }
            apply_narrow_relocs(&mut product, &narrow);
            (glcc_dwarf::emit_with_debug(product, &lowered, &mir_module), asm_text)
        } else {
            let narrow = glcc_lower_cl::lower_module_collecting(&mir_module, &mut obj, cl_target, &mut asm_text, &mut unwind);
            let mut product = obj.finish();
            if let Err(e) = winunwind::emit(&mut product, &unwind) {
                eprintln!("glcc: warning: {e}");
            }
            apply_narrow_relocs(&mut product, &narrow);
            (product.emit().expect("object emit"), asm_text)
        })
    }));
    match outcome {
        Ok(Ok(compiled)) => Ok(compiled),
        Ok(Err(())) => Err(()),
        Err(_) => Err(()),
    }
}

/// Patch a `__ptr32` object's 32-bit address relocations into the
/// emitted object. Cranelift's data descriptor only offers pointer-width
/// relocations, so `define_global` leaves these fields zeroed and hands
/// them here; the object writer turns each into a format-appropriate
/// 4-byte absolute reloc (`R_X86_64_32`, `IMAGE_REL_AMD64_ADDR32`) over
/// the zeroed image.
fn apply_narrow_relocs(product: &mut cranelift_object::ObjectProduct, narrow: &[glcc_lower_cl::NarrowReloc]) {
    use glcc_lower_cl::NarrowTarget;
    use object::write::{Relocation, SymbolId, SymbolSection};
    use object::{RelocationEncoding, RelocationFlags, RelocationKind};

    for reloc in narrow {
        let container = product.data_symbol(reloc.container);
        let SymbolSection::Section(section) = product.object.symbol(container).section else {
            panic!("__ptr32 reloc container has no section");
        };
        let place = product.object.symbol(container).value + reloc.offset;
        let symbol: SymbolId = match reloc.target {
            NarrowTarget::Data(id) => product.data_symbol(id),
            NarrowTarget::Func(id) => product.function_symbol(id),
        };
        let relocation = Relocation {
            offset: place,
            symbol,
            addend: reloc.addend,
            flags: RelocationFlags::Generic {
                kind: RelocationKind::Absolute,
                encoding: RelocationEncoding::Generic,
                size: 32,
            },
        };
        product.object.add_relocation(section, relocation).expect("__ptr32 relocation");
    }
}

/// MIR text for `--emit-mir`/`--emit-full-mir`, sharing
/// `compile_source`'s panic handling: the frontend speaks for itself,
/// so a caught panic becomes `Quiet` instead of double-reporting.
fn dump_mir_source(src: &str, flavor: Flavor, target: Target, opt_level: OptLevel, async_enabled: bool, lang: LangOpts, error_format: ErrorFormat, full: bool) -> std::result::Result<String, ()> {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (module, _) = lower_source_to_mir(src, flavor, target, opt_level, async_enabled, lang, error_format)?;
        Ok::<String, ()>(if full {
            glcc_mir::display::dump_module_full(&module)
        } else {
            glcc_mir::display::dump_module(&module)
        })
    }));
    match outcome {
        Ok(Ok(text)) => Ok(text),
        Ok(Err(())) => Err(()),
        Err(_) => Err(()),
    }
}

/// Parse and check only (`-fsyntax-only`).
fn check_source(src: &str, target: Target, async_enabled: bool, lang: LangOpts, error_format: ErrorFormat) -> std::result::Result<(), ()> {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut interner = Interner::new();
        let mut sema = Sema::with_target(target);
        sema.lang = lang;
        sema.async_enabled = async_enabled;
        let tokens = Lexer::new(src, &mut interner).lex_all();
        let mut parser = Parser::new(src, tokens, &mut sema, &mut interner);
        parser.emit_human = error_format == ErrorFormat::Human;
        let _ = parser.parse_translation_unit();
        let has_errors = parser.has_errors();
        if error_format == ErrorFormat::Json && !parser.diagnostics().is_empty() {
            eprintln!("{}", parser.diagnostics_json());
        }
        if has_errors {
            return Err(());
        }
        Ok(())
    }));
    match outcome {
        Ok(Ok(())) => Ok(()),
        Ok(Err(())) => Err(()),
        Err(_) => Err(()),
    }
}

/// Dedicated assembler for raw `.s` input and glcc's generated asm
/// text: `$GLCC_AS` when set, the host `as` natively, otherwise the
/// target's `<triple>-as` if the cross binutils are installed. `None`
/// falls back to the link driver, which also handles `.S` (whose
/// preprocessing step needs a full driver).
fn assembler_for(job: &Job) -> Option<OsString> {
    if let Some(as_) = std::env::var_os("GLCC_AS") {
        if !as_.is_empty() {
            return Some(as_);
        }
    }
    if job.target.is_native() {
        return tool_available("as").then(|| OsString::from("as"));
    }
    let cross = format!("{}-as", job.target.triple());
    tool_available(&cross).then(|| OsString::from(cross))
}

/// Assemble one `.s`/`.S`: raw `.s` goes straight to the assembler,
/// `.S` to the link driver, whose preprocessor runs first.
fn assemble_cmd(job: &Job, input: &OsStr, output: &Path) -> (OsString, Vec<OsString>) {
    let preprocessed = Path::new(input).extension().is_some_and(|e| e == "S");
    if !preprocessed {
        if let Some(prog) = assembler_for(job) {
            let mut args = Vec::new();
            for p in &job.as_args {
                // `-Wa,a,b` arrives comma-joined; the assembler wants
                // the options as separate arguments.
                for flag in p.to_string_lossy().split(',') {
                    args.push(OsString::from(flag));
                }
            }
            args.push(OsString::from("-o"));
            args.push(output.as_os_str().to_os_string());
            args.push(input.to_os_string());
            return (prog, args);
        }
    }
    let mut args = vec![OsString::from("-c")];
    args.extend(job.target_args());
    if preprocessed {
        args.extend(job.cpp.iter().cloned());
    }
    for p in &job.as_args {
        args.push(OsString::from(format!("-Wa,{}", p.to_string_lossy())));
    }
    args.push(OsString::from("-o"));
    args.push(output.as_os_str().to_os_string());
    args.push(input.to_os_string());
    (link_driver(), args)
}

/// Dependency generation over compilable inputs:
/// `clang -M|-MM [--target=T] <cpp...> <dep-extra...> [-MP] [-MF f] [-MT t] <inputs...>`.
fn deps_cmd(job: &Job, style_mm: bool, inputs: &[OsString]) -> (OsString, Vec<OsString>) {
    let mut args = vec![OsString::from(if style_mm { "-MM" } else { "-M" })];
    args.extend(job.target_args());
    args.extend(job.cpp.iter().cloned());
    args.extend(job.dep_extra.iter().cloned());
    if job.dep_phony {
        args.push(OsString::from("-MP"));
    }
    if let Some(f) = &job.dep_file {
        args.push(OsString::from("-MF"));
        args.push(f.clone());
    }
    if let Some(t) = &job.dep_target {
        args.push(OsString::from("-MT"));
        args.push(t.clone());
    }
    args.extend(inputs.iter().cloned());
    (preprocessor(), args)
}

fn stem_of(path: &OsStr) -> OsString {
    Path::new(path).file_stem().map(|s| s.to_os_string()).unwrap_or_else(|| OsString::from("a"))
}

/// Directory for one input's `-save-temps` files: the working
/// directory, or (under `-save-temps=obj` with `-c`) beside the
/// object file. `None` when temps are off. Same-stem inputs in
/// different directories overwrite each other, like gcc.
fn temps_dir(job: &Job, obj_out: Option<&Path>) -> Option<PathBuf> {
    match job.save_temps {
        SaveTemps::Off => None,
        SaveTemps::Cwd => Some(PathBuf::from(".")),
        SaveTemps::Obj => Some(
            obj_out
                .and_then(|o| o.parent())
                .filter(|p| !p.as_os_str().is_empty())
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from(".")),
        ),
    }
}

/// Write one `-save-temps` file (best effort: an unwritable temp
/// must not fail the build).
fn save_temp_file(dir: &Path, name: &str, bytes: &[u8]) {
    let path = dir.join(name);
    if let Err(e) = std::fs::write(&path, bytes) {
        eprintln!("glcc: warning: cannot write '{}': {e}", path.display());
    }
}

/// Output paths for `-c`: one `-o` for a single compilable input,
/// else `<stem>.o` per input in the working directory.
fn compile_outputs(job: &Job) -> std::result::Result<Vec<PathBuf>, String> {
    let compilable: Vec<usize> = job
        .order
        .iter()
        .filter_map(|item| match item {
            OrderItem::Input(i) => Some(*i),
            OrderItem::LinkArg(_) => None,
        })
        .collect();
    if compilable.len() == 1 {
        if let Some(o) = &job.output {
            return Ok(vec![PathBuf::from(o)]);
        }
    } else if job.output.is_some() {
        return Err("cannot specify -o with -c with multiple files".into());
    }
    Ok(compilable
        .iter()
        .map(|i| {
            let mut stem = stem_of(&job.inputs[*i].path);
            stem.push(".o");
            PathBuf::from(stem)
        })
        .collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Artifact {
    Archive,
    Shared,
    Executable,
    Relocatable,
}

/// What the link step produces: `-r` and `-shared` say it
/// directly, otherwise the `-o` name decides (`.a` archives via
/// `ar`, `.so` links shared), defaulting to an executable.
fn decide_artifact(job: &Job) -> std::result::Result<Artifact, String> {
    if job.relocatable {
        return Ok(Artifact::Relocatable);
    }
    if job.shared {
        if let Some(o) = &job.output {
            if o.as_encoded_bytes().ends_with(b".a") {
                return Err("cannot combine -shared with '.a' output".into());
            }
        }
        return Ok(Artifact::Shared);
    }
    if let Some(o) = &job.output {
        let raw = o.as_encoded_bytes();
        if raw.ends_with(b".a") {
            return Ok(Artifact::Archive);
        }
        if raw.ends_with(b".so") || raw.windows(4).any(|w| w == b".so.") {
            return Ok(Artifact::Shared);
        }
    }
    Ok(Artifact::Executable)
}

/// Whether `name` is an executable on PATH (probed with `--version`).
fn tool_available(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn mold_available() -> bool {
    tool_available("mold")
}

/// Link command. The glcc runtime (when given) goes right after
/// `-o <out>`, before user objects, so it precedes anything that might
/// reference its symbols; `use_mold` is decided by the caller.
fn link_command(job: &Job, objects: &dyn Fn(usize) -> OsString, out: &OsStr, use_mold: bool, runtime: Option<&OsStr>, asm_objects: &[PathBuf]) -> (OsString, Vec<OsString>) {
    let mut args = Vec::new();
    args.extend(job.target_args());
    if use_mold {
        args.push(OsString::from("-fuse-ld=mold"));
    }
    args.push(OsString::from("-o"));
    args.push(out.to_os_string());
    if let Some(rt) = runtime {
        args.push(rt.to_os_string());
        if quad_runtime_needs_libm(job.target.arch) {
            args.push(OsString::from("-lm"));
        }
    }
    for item in &job.order {
        match item {
            OrderItem::Input(i) => args.push(objects(*i)),
            OrderItem::LinkArg(a) => args.push(a.clone()),
        }
    }
    for obj in asm_objects {
        args.push(obj.as_os_str().to_os_string());
    }
    for l in &job.link_libs {
        args.push(l.clone());
    }
    (link_driver(), args)
}

fn is_object_arg(arg: &OsStr) -> bool {
    matches!(Path::new(arg).extension().and_then(|e| e.to_str()), Some("o" | "obj" | "lo"))
}

fn emit_output(path: &OsStr, bytes: &[u8]) -> Outcome<()> {
    std::fs::write(path, bytes).map_err(|e| fail(format!("cannot write '{}': {e}", Path::new(path).display())))
}

/// Write per-TU assembler text as `<stem>.s` in `dir`, assemble it to
/// `<stem>.o` there, and return the object path.
fn assemble_asm_text(ctx: &Ctx, job: &Job, text: &str, stem: &str, dir: &Path) -> Outcome<PathBuf> {
    let s_path = dir.join(format!("{stem}.s"));
    let o_path = dir.join(format!("{stem}.o"));
    std::fs::write(&s_path, text).map_err(|e| fail(format!("cannot write '{}': {e}", s_path.display())))?;
    let (prog, args) = assemble_cmd(job, s_path.as_os_str(), &o_path);
    let out = run_cmd(ctx, &prog, &args).map_err(|e| fail(format!("cannot run assembler driver: {e}")))?.unwrap();
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let err = err.trim_end();
        if err.is_empty() {
            return Err(fail(format!("assembling '{}' failed", s_path.display())));
        }
        return Err(fail(format!("assembling '{}' failed:\n{err}", s_path.display())));
    }
    Ok(o_path)
}

/// Relocatably merge `extra` into `primary` so `-c` still yields one
/// object file.
fn merge_objects(ctx: &Ctx, job: &Job, primary: &Path, extra: &Path) -> Outcome<()> {
    let name = primary.file_name().unwrap_or_default().to_string_lossy().into_owned();
    let tmp = primary.with_file_name(format!("{name}.glcc-merge-{}", std::process::id()));
    let mut args = vec![OsString::from("-r"), OsString::from("-o"), tmp.as_os_str().to_os_string()];
    args.extend(job.target_args());
    args.push(primary.as_os_str().to_os_string());
    args.push(extra.as_os_str().to_os_string());
    let prog = link_driver();
    let out = run_cmd(ctx, &prog, &args).map_err(|e| fail(format!("cannot run linker driver: {e}")))?.unwrap();
    if !out.status.success() {
        let _ = std::fs::remove_file(&tmp);
        let err = String::from_utf8_lossy(&out.stderr);
        let err = err.trim_end();
        if err.is_empty() {
            return Err(fail(format!("relocatable link of '{}' failed", primary.display())));
        }
        return Err(fail(format!("relocatable link of '{}' failed:\n{err}", primary.display())));
    }
    std::fs::rename(&tmp, primary).map_err(|e| fail(format!("cannot replace '{}': {e}", primary.display())))
}

fn emit_stdout(bytes: &[u8]) -> Outcome<()> {
    std::io::stdout().write_all(bytes).map_err(|e| fail(format!("cannot write to stdout: {e}")))
}

/// Side-effect `-MD`/`-MMD` pass after a successful compile: one
/// clang invocation per input (or a single shared one when `-MF`
/// names the file for several inputs).
fn run_depgen(ctx: &Ctx, job: &Job, units: &[(OsString, Option<OsString>)]) -> Outcome<()> {
    let flag = if job.depgen == Some(true) { "-MMD" } else { "-MD" };
    // `-E`: dependencies only — without it clang would link.
    if job.dep_file.is_some() && units.len() > 1 {
        let mut args = vec![OsString::from(flag), OsString::from("-E")];
        args.extend(job.target_args());
        args.extend(job.cpp.iter().cloned());
        args.extend(job.dep_extra.iter().cloned());
        if job.dep_phony {
            args.push(OsString::from("-MP"));
        }
        args.push(OsString::from("-MF"));
        args.push(job.dep_file.clone().unwrap());
        if let Some(t) = &job.dep_target {
            args.push(OsString::from("-MT"));
            args.push(t.clone());
        }
        for (input, _) in units {
            args.push(input.clone());
        }
        let prog = preprocessor();
        let out = run_cmd(ctx, &prog, &args).map_err(|e| fail(format!("cannot run preprocessor: {e}")))?;
        if let Some(out) = out {
            if !out.status.success() {
                return Err(fail(format!("dependency generation failed:\n{}", String::from_utf8_lossy(&out.stderr).trim_end())));
            }
        }
        return Ok(());
    }
    for (input, obj) in units {
        let mut args = vec![OsString::from(flag), OsString::from("-E")];
        args.extend(job.target_args());
        args.extend(job.cpp.iter().cloned());
        args.extend(job.dep_extra.iter().cloned());
        if job.dep_phony {
            args.push(OsString::from("-MP"));
        }
        match (&job.dep_file, units.len()) {
            (Some(f), 1) => {
                args.push(OsString::from("-MF"));
                args.push(f.clone());
            }
            _ => {
                let mut d = stem_of(input);
                d.push(".d");
                args.push(OsString::from("-MF"));
                args.push(d);
            }
        }
        match (&job.dep_target, obj) {
            (Some(t), _) => {
                args.push(OsString::from("-MT"));
                args.push(t.clone());
            }
            (None, Some(o)) => {
                args.push(OsString::from("-MT"));
                args.push(o.clone());
            }
            (None, None) => {
                let mut o = stem_of(input);
                o.push(".o");
                args.push(OsString::from("-MT"));
                args.push(o);
            }
        }
        args.push(input.clone());
        let prog = preprocessor();
        let out = run_cmd(ctx, &prog, &args).map_err(|e| fail(format!("cannot run preprocessor: {e}")))?;
        if let Some(out) = out {
            if !out.status.success() {
                return Err(fail(format!("dependency generation failed:\n{}", String::from_utf8_lossy(&out.stderr).trim_end())));
            }
        }
    }
    Ok(())
}

/// Warn about linker inputs under an action that never links.
fn warn_unused_link_inputs(job: &Job, action: &str) {
    for item in &job.order {
        if let OrderItem::LinkArg(a) = item {
            eprintln!("glcc: warning: '{}': linker input file unused because {} not done", Path::new(a).display(), action);
        }
    }
}

fn has_compilable(job: &Job) -> bool {
    job.order.iter().any(|item| matches!(item, OrderItem::Input(_)))
}

fn do_preprocess(ctx: &Ctx, job: &Job) -> Outcome<()> {
    if !has_compilable(job) {
        warn_unused_link_inputs(job, "preprocessing");
        return Ok(());
    }
    if ctx.dry_run {
        for item in &job.order {
            if let OrderItem::Input(i) = item {
                match job.inputs[*i].lang {
                    Lang::Cpp | Lang::EdgC => eprintln!("+ cpfe --preprocess {}", job.inputs[*i].path.display()),
                    _ if use_native_pp(job) => eprintln!("+ glcc_pp {}", job.inputs[*i].path.display()),
                    _ => {
                        let (prog, args) = preprocess_cmd(job, &job.inputs[*i].path);
                        eprintln!("+ {}", fmt_cmd(&prog, &args));
                    }
                }
            }
        }
        return Ok(());
    }
    let mut combined = Vec::new();
    for item in &job.order {
        if let OrderItem::Input(i) = item {
            let input = &job.inputs[*i];
            match input.lang {
                Lang::Cpp | Lang::EdgC => match cpp_preprocess_text(job, input) {
                    Ok(text) => combined.extend(text.into_bytes()),
                    Err(e) => return Err(e),
                },
                _ => combined.extend(run_preprocess(ctx, job, &input.path)?),
            }
        }
    }
    match &job.output {
        Some(o) => emit_output(o, &combined),
        None => emit_stdout(&combined),
    }
}

fn do_deps(ctx: &Ctx, job: &Job) -> Outcome<()> {
    let inputs: Vec<OsString> = job
        .order
        .iter()
        .filter_map(|item| match item {
            OrderItem::Input(i) => {
                if matches!(job.inputs[*i].lang, Lang::Cpp | Lang::EdgC) {
                    None
                } else {
                    Some(job.inputs[*i].path.clone())
                }
            }
            OrderItem::LinkArg(_) => None,
        })
        .collect();
    if job.inputs.iter().any(|i| matches!(i.lang, Lang::Cpp | Lang::EdgC)) {
        eprintln!("glcc: warning: -M/-MM do not trace EDG inputs yet; listing C inputs only");
    }

    if inputs.is_empty() {
        warn_unused_link_inputs(job, "dependency generation");
        return Ok(());
    }
    let (prog, args) = deps_cmd(job, job.deps_sys_excluded, &inputs);
    let out = run_cmd(ctx, &prog, &args).map_err(|e| fail(format!("cannot run preprocessor: {e}")))?;
    let Some(out) = out else { return Ok(()) };
    if !out.status.success() {
        return Err(fail(format!("dependency generation failed:\n{}", String::from_utf8_lossy(&out.stderr).trim_end())));
    }
    // `-o` names the depfile when `-MF` didn't.
    match (&job.dep_file, &job.output) {
        (None, Some(o)) => emit_output(o, &out.stdout),
        _ => emit_stdout(&out.stdout),
    }
}

/// Which EDG language mode translates an input: C++ (`--c++`) or
/// C (`--c`, under `--edg-c`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EdgMode {
    Cxx,
    C,
}

fn edg_mode(lang: Lang) -> EdgMode {
    match lang {
        Lang::EdgC => EdgMode::C,
        _ => EdgMode::Cxx,
    }
}

/// Shared cpfe invocation: language mode, host version scrapes, and the
/// user's `-I`/`-D`/`-U` flags, plus caller `extra` options.
fn cpfe_argv(job: &Job, input_path: &OsStr, mode: EdgMode, extra: &[OsString]) -> Outcome<Vec<OsString>> {
    let root = job.edg_root.as_ref().ok_or_else(|| fail("edg root not configured (set GLCPP_EDG)"))?;
    let scrape = edg_scrape(root).ok_or_else(|| fail(format!("cannot scrape gcc for EDG at {}", root.display())))?;
    let mut fe: Vec<OsString> = vec![OsString::from("cpfe")];
    let includes = match mode {
        EdgMode::Cxx => {
            fe.push(OsString::from("--c++"));
            &scrape.cxx_includes
        }
        // `-tused` is C++-only (template instantiation); C mode
        // translates straight through.
        EdgMode::C => {
            fe.push(OsString::from("--c"));
            &scrape.c_includes
        }
    };
    fe.push(OsString::from(format!("--gnu_version={}", scrape.version)));
    fe.push(OsString::from("-D_POSIX_SOURCE"));
    fe.push(OsString::from("-D__CHAR_BIT__=8"));
    for d in includes.split(':') {
        if !d.is_empty() {
            fe.push(OsString::from(format!("--sys_include={d}")));
        }
    }
    fe.extend(extra.iter().cloned());
    fe.extend(job.fe_args.iter().cloned());
    let canon = Path::new(input_path)
        .canonicalize()
        .map_err(|e| fail(format!("cannot locate '{}': {e}", Path::new(input_path).display())))?;
    fe.push(canon.into_os_string());
    set_edg_env(root, &scrape);
    Ok(fe)
}

/// Run cpfe on an EDG input (`Lang::Cpp` or `Lang::EdgC`); returns
/// the generated `.int.c` path. cpfe runs with `tu_dir` as cwd so its
/// `<source>.ti` instantiation file lands there rather than in the
/// user's directory.
fn fe_translate(ctx: &Ctx, job: &Job, input: &Input, idx: usize, scratch: &TempDir, gen_c_out: Option<&Path>) -> Outcome<PathBuf> {
    let tu_dir = scratch.path().join(format!("tu{idx}"));
    std::fs::create_dir_all(&tu_dir).map_err(|e| fail(format!("cannot create {}: {e}", tu_dir.display())))?;
    let stem = stem_of(&input.path);
    let stem_s = stem.to_string_lossy().into_owned();
    let gen_c = match gen_c_out {
        Some(p) => p.to_path_buf(),
        None => tu_dir.join(format!("{stem_s}.int.c")),
    };

    let mode = edg_mode(input.lang);
    let mut extra = Vec::new();
    if mode == EdgMode::Cxx {
        extra.push(OsString::from("-tused"));
    }
    if gen_c_out.is_some() || job.action != Action::SyntaxOnly {
        extra.push(OsString::from(format!("--gen_c_file_name={}", gen_c.display())));
    }
    let fe = cpfe_argv(job, &input.path, mode, &extra)?;

    if ctx.verbose || ctx.dry_run {
        eprintln!("+ cpfe {}", fe.iter().map(|a| a.to_string_lossy().into_owned()).collect::<Vec<_>>().join(" "));
    }
    if ctx.dry_run {
        return Ok(gen_c);
    }

    let prev = std::env::current_dir().map_err(|e| fail(format!("cannot get cwd: {e}")))?;
    if let Err(e) = std::env::set_current_dir(&tu_dir) {
        return Err(fail(format!("cannot cd to {}: {e}", tu_dir.display())));
    }
    let rc = edg_frontend(&fe)?;
    let _ = std::env::set_current_dir(&prev);
    if rc != 0 {
        return Err(fail(format!("EDG front end failed for '{}'", Path::new(&input.path).display())));
    }

    Ok(gen_c)
}

/// Compile an EDG input (`Lang::Cpp` or `Lang::EdgC`) via cpfe, then
/// glcc's pipeline. cpfe runs with `tu_dir` as cwd so its
/// `<source>.ti` instantiation file lands there rather than in the
/// user's directory, beside the `.int.c` and the resulting `.o`
/// (prelink locates `.ti` files by object stem).
fn compile_edg_input(ctx: &Ctx, job: &Job, input: &Input, idx: usize, out: &Path, scratch: &TempDir, obj_out: Option<&Path>) -> Outcome<String> {
    let gen_c = fe_translate(ctx, job, input, idx, scratch, None)?;
    // cpfe's `<stem>.ti` instantiation file stays inside the scratch
    // dir (it is only consumed by edg_prelink, which this pipeline
    // does not run), so the user's directory sees just the `.o`.
    let gen_c_os = gen_c.into_os_string();
    let bytes = run_preprocess(ctx, job, &gen_c_os)?;
    let c = decode_source(&bytes);
    // `-save-temps`: the raw EDG output plus exactly what the C
    // frontend parses, written before compiling so they survive
    // frontend errors too.
    if let Some(dir) = temps_dir(job, obj_out) {
        let stem = stem_of(&input.path).to_string_lossy().into_owned();
        match std::fs::read(&gen_c_os) {
            Ok(raw) => save_temp_file(&dir, &format!("{stem}.int.c"), &raw),
            Err(e) => eprintln!("glcc: warning: cannot read '{}': {e}", gen_c_os.display()),
        }
        save_temp_file(&dir, &format!("{stem}.i"), &bytes);
    }
    let (bytes, asm_text) = compile_source(&c, job.flavor, job.target, job.debug, job.pic, job.opt_level, job.async_enabled, language(job), job.error_format).map_err(|()| {
        take_diagnostic_emitted();
        Fail::Quiet
    })?;
    std::fs::write(out, &bytes).map_err(|e| fail(format!("cannot write '{}': {e}", out.display())))?;

    Ok(asm_text)
}

#[derive(Clone)]
struct EdpScrape {
    version: String,
    cxx_includes: String,
    c_includes: String,
}

/// EDG front-end entry points glcpp installs at startup. GLCC never
/// registers these, so C++ inputs only work through the glcpp crate that
/// links edg-sys.
pub struct EdgOps {
    pub frontend: fn(&[OsString]) -> i32,
    pub frontend_capture: fn(&[OsString]) -> (i32, Vec<u8>),
    /// Directory holding the staged `libC.a` (baked in by glcpp's build
    /// script from edg-sys metadata). `None` under glcc, which never
    /// links the C++ runtime.
    pub libc_dir: Option<PathBuf>,
}

static EDG_OPS: OnceLock<EdgOps> = OnceLock::new();

pub fn install_edg_ops(ops: EdgOps) {
    let _ = EDG_OPS.set(ops);
}

fn edg_ops() -> Outcome<&'static EdgOps> {
    EDG_OPS.get().ok_or_else(|| fail("C++ sources require glcpp (EDG is not linked into glcc)"))
}

fn edg_frontend(args: &[OsString]) -> Outcome<i32> {
    Ok((edg_ops()?.frontend)(args))
}

fn edg_frontend_capture(args: &[OsString]) -> Outcome<(i32, Vec<u8>)> {
    Ok((edg_ops()?.frontend_capture)(args))
}

fn edg_scrape(root: &Path) -> Option<EdpScrape> {
    static CACHE: OnceLock<Option<EdpScrape>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            fn run(root: &Path, lang: &str) -> Option<String> {
                let out = Command::new("python3")
                    .arg(root.join("dev_tools/bin/edg-scrape-compiler"))
                    .arg("gcc")
                    .arg("--lang")
                    .arg(lang)
                    .arg("includes")
                    .output()
                    .ok()?;
                if out.status.success() { String::from_utf8(out.stdout).ok() } else { None }
            }
            let cxx = run(root, "c++")?;
            let c = run(root, "c")?;
            let version = Command::new("python3")
                .arg(root.join("dev_tools/bin/edg-scrape-compiler"))
                .arg("gcc")
                .arg("version")
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())?;
            Some(EdpScrape {
                version: version.trim().to_string(),
                cxx_includes: cxx.trim().to_string(),
                c_includes: c.trim().to_string(),
            })
        })
        .as_ref()
        .cloned()
}

fn resolve_edg_root() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("GLCPP_EDG") {
        if !v.is_empty() {
            return Some(PathBuf::from(v));
        }
    }
    let from_manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("edg");
    if from_manifest.is_dir() {
        return Some(from_manifest);
    }
    None
}

/// Directory holding `libC.a` for the C++ link: the baked-in staged
/// path wins, then `$GLCPP_LIB_DIR`, then the install location.
fn resolve_libc_dir() -> Option<PathBuf> {
    if let Some(dir) = EDG_OPS.get().and_then(|ops| ops.libc_dir.as_ref()) {
        if dir.join("libC.a").is_file() {
            return Some(dir.clone());
        }
    }
    if let Ok(v) = std::env::var("GLCPP_LIB_DIR") {
        if !v.is_empty() {
            let dir = PathBuf::from(v);
            if dir.join("libC.a").is_file() {
                return Some(dir);
            }
        }
    }
    if let Some(home) = home_dir() {
        let dir = home.join(".local/lib/glcc");
        if dir.join("libC.a").is_file() {
            return Some(dir);
        }
    }
    None
}

fn set_edg_env(root: &Path, scrape: &EdpScrape) {
    // EDG sniffs its runtime environment around cpfe: base includes and
    // the scraped includes of the host compiler pick the standard library
    // header locations. Single-threaded driver -> unsafe set_var is sound.
    unsafe {
        std::env::set_var("EDG_BASE", root.join("bases/docker/dev-env/gcc"));
        std::env::set_var("EDG_USE_SYSTEM_HEADERS", "1");
        std::env::set_var("EDG_GCC_INCL_SCRAPE", &scrape.cxx_includes);
        std::env::set_var("EDG_GCC_CINCL_SCRAPE", &scrape.c_includes);
        std::env::set_var("EDG_GCC_VER_SCRAPE", &scrape.version);
    }
}

/// Compile one input to object bytes and per-TU assembler text (C via
/// the in-process pipeline, preprocessed input read straight off disk).
fn compile_input_to_bytes(ctx: &Ctx, job: &Job, input: &Input, obj_out: Option<&Path>) -> Outcome<(Vec<u8>, String)> {
    let text = match input.lang {
        Lang::C => {
            let bytes = run_preprocess(ctx, job, &input.path)?;
            // `-save-temps`: keep a copy of exactly what the frontend
            // parses, written before compiling so it survives errors.
            if let Some(dir) = temps_dir(job, obj_out) {
                let stem = stem_of(&input.path).to_string_lossy().into_owned();
                save_temp_file(&dir, &format!("{stem}.i"), &bytes);
            }
            decode_source(&bytes)
        }
        Lang::Preprocessed => {
            let bytes = std::fs::read(&input.path).map_err(|e| fail(format!("cannot read '{}': {e}", Path::new(&input.path).display())))?;
            decode_source(&bytes)
        }
        Lang::Assembler => unreachable!("assembler inputs are assembled, not compiled"),
        Lang::Cpp | Lang::EdgC => unreachable!("EDG inputs are handled by compile_edg_input"),
    };
    compile_source(&text, job.flavor, job.target, job.debug, job.pic, job.opt_level, job.async_enabled, language(job), job.error_format).map_err(|()| {
        // Drain the flag so a caught diagnostic never silences a
        // later, unrelated panic in the hook.
        take_diagnostic_emitted();
        Fail::Quiet
    })
}

/// `glcpp -E`: route EDG inputs through the EDG front end, not clang.
fn cpp_preprocess_text(job: &Job, input: &Input) -> Outcome<String> {
    let fe = cpfe_argv(job, &input.path, edg_mode(input.lang), &[OsString::from("--preprocess")])?;
    let (rc, bytes) = edg_frontend_capture(&fe)?;
    if rc != 0 {
        return Err(fail(format!("EDG front end failed for '{}'", Path::new(&input.path).display())));
    }
    Ok(decode_source(&bytes))
}

fn do_syntax_only(ctx: &Ctx, job: &Job) -> Outcome<()> {
    if !has_compilable(job) {
        warn_unused_link_inputs(job, "checking");
        return Ok(());
    }
    for item in &job.order {
        let OrderItem::Input(i) = item else { continue };
        let input = &job.inputs[*i];
        match input.lang {
            Lang::Assembler => eprintln!("glcc: warning: '{}': assembler input skipped for -fsyntax-only", Path::new(&input.path).display()),
            Lang::C | Lang::Preprocessed => {
                let text = match input.lang {
                    Lang::C => {
                        let bytes = run_preprocess(ctx, job, &input.path)?;
                        if let Some(dir) = temps_dir(job, None) {
                            let stem = stem_of(&input.path).to_string_lossy().into_owned();
                            save_temp_file(&dir, &format!("{stem}.i"), &bytes);
                        }
                        decode_source(&bytes)
                    }
                    _ => {
                        let bytes = std::fs::read(&input.path).map_err(|e| fail(format!("cannot read '{}': {e}", Path::new(&input.path).display())))?;
                        decode_source(&bytes)
                    }
                };
                if ctx.dry_run {
                    continue;
                }
                check_source(&text, job.target, job.async_enabled, language(job), job.error_format).map_err(|()| Fail::Quiet)?;
            }
            Lang::Cpp | Lang::EdgC => {
                if ctx.dry_run {
                    eprintln!("+ cpfe [options] --gen_c_file_name=... {}", input.path.display());
                } else {
                    let scratch = match TempDir::create() {
                        Ok(s) => s,
                        Err(e) => return Err(fail(format!("cannot create temporary directory: {e}"))),
                    };
                    // `-save-temps`: land the lowered C directly in the
                    // temps dir instead of the throwaway scratch dir.
                    let gen_c_out = temps_dir(job, None).and_then(|dir| {
                        let stem = stem_of(&input.path).to_string_lossy().into_owned();
                        std::env::current_dir().ok().map(|cwd| cwd.join(dir).join(format!("{stem}.int.c")))
                    });
                    fe_translate(ctx, job, input, *i, &scratch, gen_c_out.as_deref())?;
                }
            }
        }
    }
    Ok(())
}

/// `--emit-mir`/`--emit-full-mir`: print each C input's MIR to stdout
/// (or `-o`), after the `-O*` pipeline. `-O0` runs no passes, so it
/// prints the raw lowering. `full` adds the per-instruction source
/// locations debug info is built from.
fn do_emit_mir(ctx: &Ctx, job: &Job, full: bool) -> Outcome<()> {
    let flag = if full { "--emit-full-mir" } else { "--emit-mir" };
    if !has_compilable(job) {
        warn_unused_link_inputs(job, "MIR emission");
        return Ok(());
    }
    let mut combined = String::new();
    for item in &job.order {
        let OrderItem::Input(i) = item else { continue };
        let input = &job.inputs[*i];
        let text = match input.lang {
            Lang::Assembler => {
                eprintln!("glcc: warning: '{}': assembler input skipped for {flag}", Path::new(&input.path).display());
                continue;
            }
            Lang::Cpp | Lang::EdgC => return Err(fail(format!("{flag} is only supported for C inputs"))),
            Lang::C => {
                let bytes = run_preprocess(ctx, job, &input.path)?;
                if let Some(dir) = temps_dir(job, None) {
                    let stem = stem_of(&input.path).to_string_lossy().into_owned();
                    save_temp_file(&dir, &format!("{stem}.i"), &bytes);
                }
                decode_source(&bytes)
            }
            Lang::Preprocessed => {
                let bytes = std::fs::read(&input.path).map_err(|e| fail(format!("cannot read '{}': {e}", Path::new(&input.path).display())))?;
                decode_source(&bytes)
            }
        };
        if ctx.dry_run {
            continue;
        }
        let mir = dump_mir_source(&text, job.flavor, job.target, job.opt_level, job.async_enabled, language(job), job.error_format, full).map_err(|()| {
            take_diagnostic_emitted();
            Fail::Quiet
        })?;
        combined.push_str(&mir);
    }
    match &job.output {
        Some(o) => emit_output(o, combined.as_bytes()),
        None => emit_stdout(combined.as_bytes()),
    }
}

/// Compile every input, pairing each with its object path: the
/// `-c` outputs for `Compile`, temp objects (or temp assembled
/// objects) otherwise. Returns `(source, object)` pairs in order.
fn do_compile(ctx: &Ctx, job: &Job, scratch: Option<&TempDir>) -> Outcome<(Vec<(OsString, PathBuf)>, Vec<PathBuf>)> {
    // `-c` has no persistent scratch of its own: create one when a
    // compiled input may need space for its assembler text (EDG always
    // does; C does when file-scope asm is present).
    let owned: Option<TempDir> = if scratch.is_none() && job.inputs.iter().any(|i| !matches!(i.lang, Lang::Assembler)) {
        Some(TempDir::create().map_err(|e| fail(format!("cannot create temporary directory: {e}")))?)
    } else {
        None
    };
    let scratch: Option<&TempDir> = scratch.or(owned.as_ref());
    let fixed: Option<Vec<PathBuf>> = match job.action {
        Action::Compile => Some(compile_outputs(job).map_err(Fail::Msg)?),
        _ => None,
    };
    let mut fixed = fixed.map(|v| v.into_iter());
    let mut pairs: Vec<(OsString, PathBuf)> = Vec::new();
    for item in &job.order {
        if let OrderItem::Input(i) = item {
            let input = &job.inputs[*i];
            let out = match (&mut fixed, input.lang) {
                (Some(outs), _) => outs.next().unwrap(),
                (None, Lang::Cpp | Lang::EdgC) => {
                    let dir = scratch.expect("EDG compile needs scratch space");
                    let mut name = stem_of(&input.path);
                    name.push(".o");
                    let d = dir.path().join(format!("tu{i}"));
                    d.join(name)
                }
                (None, _) => {
                    let dir = scratch.expect("link flow needs scratch space");
                    let mut name = stem_of(&input.path);
                    match input.lang {
                        Lang::Assembler => name.push(".o"),
                        _ => name.push(format!("-{i}.o")),
                    }
                    dir.path().join(name)
                }
            };
            pairs.push((input.path.clone(), out));
        } else if let OrderItem::LinkArg(a) = item {
            if job.action == Action::Compile {
                eprintln!("glcc: warning: '{}': linker input file unused because linking not done", Path::new(a).display());
            }
        }
    }
    if ctx.dry_run {
        for item in &job.order {
            if let OrderItem::Input(i) = item {
                let input = &job.inputs[*i];
                let out = pairs.iter().find(|(p, _)| p == &input.path).unwrap().1.clone();
                match input.lang {
                    Lang::Assembler => {
                        let (prog, args) = assemble_cmd(job, &input.path, &out);
                        eprintln!("+ {}", fmt_cmd(&prog, &args));
                    }
                    Lang::Cpp | Lang::EdgC => {
                        eprintln!("+ cpfe [options] --gen_c_file_name=... {}", input.path.to_string_lossy());
                    }
                    _ => {
                        let (prog, args) = preprocess_cmd(job, &input.path);
                        eprintln!("+ {}", fmt_cmd(&prog, &args));
                    }
                }
            }
        }
        return Ok((Vec::new(), Vec::new()));
    }
    let mut produced: Vec<(OsString, PathBuf)> = Vec::new();
    // Objects from the per-TU assembler text (file-scope asm today,
    // inline-asm wrappers later): linked alongside the primary object,
    // or relocatably merged into it under `-c`.
    let mut asm_objects: Vec<PathBuf> = Vec::new();
    for item in &job.order {
        if let OrderItem::Input(i) = item {
            let input = &job.inputs[*i];
            let out = pairs.iter().find(|(p, _)| p == &input.path).unwrap().1.clone();
            match input.lang {
                Lang::Assembler => {
                    let (prog, args) = assemble_cmd(job, &input.path, &out);
                    let out = run_cmd(ctx, &prog, &args).map_err(|e| fail(format!("cannot run assembler driver: {e}")))?.unwrap();
                    if !out.status.success() {
                        let err = String::from_utf8_lossy(&out.stderr);
                        let err = err.trim_end();
                        if err.is_empty() {
                            return Err(fail(format!("assembling '{}' failed", Path::new(&input.path).display())));
                        }
                        return Err(fail(format!("assembling '{}' failed:\n{}", Path::new(&input.path).display(), err)));
                    }
                }
                _ => {
                    // `-save-temps=obj` anchors temps beside the object
                    // file, which only exists persistently under `-c`;
                    // in link flow the objects are ephemeral, so temps
                    // fall back to the working directory.
                    let obj_for_temps = (job.action == Action::Compile).then(|| out.as_path());
                    let asm_text = match input.lang {
                        Lang::Cpp | Lang::EdgC => {
                            let scratch_ref = scratch.ok_or_else(|| fail("missing scratch for EDG compile"))?;
                            compile_edg_input(ctx, job, input, *i, &out, scratch_ref, obj_for_temps)?
                        }
                        _ => {
                            let (bytes, asm_text) = compile_input_to_bytes(ctx, job, input, obj_for_temps)?;
                            emit_output(out.as_os_str(), &bytes)?;
                            asm_text
                        }
                    };
                    if !asm_text.trim().is_empty() {
                        let dir = scratch.ok_or_else(|| fail("missing scratch for assembler text"))?;
                        let stem = stem_of(&input.path).to_string_lossy().into_owned();
                        let extra = assemble_asm_text(ctx, job, &asm_text, &format!("{stem}-{i}-asm"), dir.path())?;
                        if let Some(tdir) = temps_dir(job, obj_for_temps) {
                            save_temp_file(&tdir, &format!("{stem}-asm.s"), asm_text.as_bytes());
                        }
                        if job.action == Action::Compile {
                            merge_objects(ctx, job, &out, &extra)?;
                        } else {
                            asm_objects.push(extra);
                        }
                    }
                }
            }
            produced.push((input.path.clone(), out));
        }
    }
    Ok((produced, asm_objects))
}

fn do_link(ctx: &Ctx, job: &Job) -> Outcome<()> {
    let scratch = TempDir::create().map_err(|e| fail(format!("cannot create temporary directory: {e}")))?;
    let (produced, asm_objects) = do_compile(ctx, job, Some(&scratch))?;
    if job.depgen.is_some() {
        // `-MT` defaults to the would-be `-c` name, not the temp object.
        let units: Vec<(OsString, Option<OsString>)> = produced
            .iter()
            .filter(|(p, _)| job.inputs.iter().find(|input| &input.path == p).is_none_or(|input| input.lang != Lang::Assembler))
            .map(|(p, _)| {
                let mut o = stem_of(p);
                o.push(".o");
                (p.clone(), Some(o))
            })
            .collect();
        run_depgen(ctx, job, &units)?;
    }
    if ctx.dry_run {
        return Ok(());
    }
    let artifact = decide_artifact(job).map_err(Fail::Msg)?;
    let out: OsString = job.output.clone().unwrap_or_else(|| match (artifact, job.target.os) {
        (Artifact::Executable, glcc_frontend::target::Os::Windows) => OsString::from("a.exe"),
        _ => OsString::from("a.out"),
    });
    match artifact {
        Artifact::Archive => do_archive(ctx, job, &produced, &asm_objects, &out),
        Artifact::Shared | Artifact::Executable | Artifact::Relocatable => do_cc_link(ctx, job, &produced, &asm_objects, &out),
    }
}

/// Bundle objects with `ar` (`-o *.a`), or MSVC's `lib.exe` for the
/// Windows target.
fn do_archive(ctx: &Ctx, job: &Job, produced: &[(OsString, PathBuf)], asm_objects: &[PathBuf], out: &OsStr) -> Outcome<()> {
    let mut objs: Vec<OsString> = produced.iter().map(|(_, o)| o.as_os_str().to_os_string()).collect();
    objs.extend(asm_objects.iter().map(|o| o.as_os_str().to_os_string()));
    for item in &job.order {
        if let OrderItem::LinkArg(a) = item {
            if is_object_arg(a) {
                objs.push(a.clone());
            } else {
                eprintln!("glcc: warning: '{}': not an object file; skipped while creating archive", Path::new(a).display());
            }
        }
    }
    if objs.is_empty() {
        return Err(fail("no object files for archive"));
    }
    if msvc_link(job) {
        let mut args = vec![OsString::from("/nologo"), OsString::from(format!("/out:{}", Path::new(out).display()))];
        args.extend(objs);
        if ctx.verbose {
            eprintln!("+ {}", fmt_cmd(&msvc_librarian(), &args));
        }
        if ctx.dry_run {
            return Ok(());
        }
        let status = Command::new(msvc_librarian())
            .args(&args)
            .status()
            .map_err(|e| fail(format!("cannot run '{}': {e}", msvc_librarian().to_string_lossy())))?;
        if !status.success() {
            return Err(fail("creating archive failed"));
        }
        return Ok(());
    }
    let mut args = vec![OsString::from("crus"), out.to_os_string()];
    args.extend(objs);
    if ctx.verbose {
        eprintln!("+ {}", fmt_cmd(OsStr::new("ar"), &args));
    }
    if ctx.dry_run {
        return Ok(());
    }
    let status = Command::new("ar").args(&args).status().map_err(|e| fail(format!("cannot run 'ar': {e}")))?;
    if !status.success() {
        return Err(fail("creating archive failed"));
    }
    Ok(())
}

/// Shared objects, executables, and `-r` partial links: the system
/// driver supplies crt files and default libraries, mold selected
/// unless overridden or missing.
fn do_cc_link(ctx: &Ctx, job: &Job, produced: &[(OsString, PathBuf)], asm_objects: &[PathBuf], out: &OsStr) -> Outcome<()> {
    let by_index: std::collections::HashMap<usize, OsString> = produced
        .iter()
        .filter_map(|(p, o)| job.inputs.iter().position(|input| &input.path == p).map(|i| (i, o.as_os_str().to_os_string())))
        .collect();
    let (prog, args) = if msvc_link(job) {
        if job.relocatable {
            return Err(fail("-r partial links are not supported for the Windows target yet"));
        }
        msvc_link_command(job, &|i| by_index[&i].clone(), out, asm_objects)
    } else {
        let mut use_mold = false;
        if job.use_ld.is_none() {
            if mold_available() {
                use_mold = true;
            } else {
                eprintln!("glcc: warning: mold not found in PATH; linking with the default linker");
            }
        }
        let runtime = if wants_glcc_runtime(job) { find_glcc_runtime("glccrt.o") } else { None };
        let runtime_arg = runtime.as_ref().map(|p| p.as_os_str().to_os_string());
        link_command(job, &|i| by_index[&i].clone(), out, use_mold, runtime_arg.as_deref(), asm_objects)
    };
    let out = run_cmd(ctx, &prog, &args).map_err(|e| fail(format!("cannot run linker driver: {e}")))?.unwrap();
    if !out.status.success() {
        // `run_cmd` captures rather than inherits stderr, so echo what
        // the linker said: undefined symbols and missing libraries are
        // the only explanation of a failed link. MSVC's `link.exe`
        // prints diagnostics to stdout instead of stderr.
        let stderr = String::from_utf8_lossy(&out.stderr);
        let stderr = stderr.trim_end();
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stdout = stdout.trim_end();
        let msg = match (stderr.is_empty(), stdout.is_empty()) {
            (true, true) => return Err(fail("link failed")),
            (false, true) => stderr.to_string(),
            (true, false) => stdout.to_string(),
            (false, false) => format!("{stderr}\n{stdout}"),
        };
        return Err(fail(format!("link failed:\n{msg}")));
    }
    Ok(())
}

// ---------- entry point ----------

fn driver_name(f: Flavor) -> &'static str {
    match f {
        Flavor::C => "glcc",
        Flavor::Cpp => "glcpp",
    }
}

fn print_help(flavor: Flavor) {
    let prog = driver_name(flavor);
    let edg_c_help = match flavor {
        Flavor::C => "",
        Flavor::Cpp => "  --edg-c            compile C inputs (.c, -x c, .i) with EDG's C\n                     frontend instead of the C++ frontend\n",
    };
    print!(
        "\
Usage: {prog} [options] file...
Drop-in GCC-style driver: preprocesses C++ to C (EDG cpfe, for glcpp), preprocesses C with clang,
compiles with the Cranelift backend, and links via the system driver with mold.

Actions (last of -c/-E/-M/-MM/-fsyntax-only/--emit-mir/--emit-full-mir wins):
  -c                 compile only; one .o per input (<stem>.o, or -o for one)
  -E                 preprocess to stdout (or -o); stops before compiling
  -M/-MM             dependency rules to stdout (or -o); -MM skips system headers
  -MD/-MMD           with -c/link: also write <stem>.d (-MF/-MT/-MP/-MG apply)
  -fsyntax-only      parse and check only; no output
  --emit-mir         print the MIR of each C input to stdout (or -o) and
                     stop; `-O*` selects how much of the pipeline runs
                     first, so `-O0 --emit-mir` prints the raw lowering
  --emit-full-mir    like --emit-mir, but every located instruction gets
                     a `; file:line:column` suffix with the source position
                     DWARF line info is generated from (much larger output)
  -S                 not supported (assembly listing)
  (default)          compile and link an executable (a.out, or -o)
  -save-temps[=cwd|=obj]
                     keep intermediates: <stem>.i (preprocessed C as the
                     frontend parses it) for C inputs, plus <stem>.int.c
                     (raw EDG-lowered C) for C++ inputs; in cwd, or beside
                     each -c object with =obj (link flow has no persistent
                     objects, so =obj falls back to cwd)
{edg_c_help}
Output detection (link step):
  -o FILE            output path; *.a archives via ar, *.so links shared
  -shared            link a shared object
  -r                 relocatable (partial) link
  -static, -pie, -no-pie, -static-pie, -s, -nostdlib, -nostartfiles,
  -nodefaultlibs     forwarded to the link, like gcc

Preprocessor (native glcc_pp by default; clang -E fallback):
  -I/-D/-U (joined or separate), -include/-imacros/-isystem/-idirafter/
  -iquote/-iprefix/-iwithprefix, -nostdinc, -undef, -std=, -ansi,
  -Wp,... (comma-split), -Xpreprocessor
  --pp=native|--pp=clang selects glcc_pp or clang -E per invocation;
  clang keeps -dM/-dD/-dN/-H/-C/-CC/-trigraphs jobs and GLCC_PP values
  that name another binary
Link (forwarded in argv order):
  -L/-l (joined or separate), -Wl,..., -Xlinker, -e/-u/-z/-T,
  --sysroot=, -fuse-ld= (default: mold when installed), -pthread
Codegen (last -O* wins, like gcc; default -O0):
  -O0                no optimization, best debug fidelity
  -O1/-O2/-O3/-Og/-Ofast/bare -O  optimized (Cranelift `speed`)
  -Os/-Oz            optimize for size (Cranelift `speed_and_size`)
  -fno-pic/-fno-pie  direct addressing instead of PIC (default); useful for
                     freestanding or fixed-address images. -fpic/-fPIC and
                     -fpie/-fPIE re-enable PIC; last flag wins.
Extensions (glcc-specific):
  -fglcc-async       stackless coroutines: `__attribute__((async))` functions
                     return a `_Future(T)` handle, `__builtin_await` suspends
                     on one; needs the `glcc_async` runtime (frame hooks are
                     weak, see docs/async-abi.md)
  --error-format=human|json
                     frontend diagnostics: codespan on stderr (default), or
                     one JSON object with 1-based line/column and byte
                     offsets, for editors (Monaco and friends)
Language level:
  -std=<std>, -ansi  set the version the frontend parses (e.g. `-std=c11`
                     requires `_Static_assert`, C89 rejects it); the same
                     gcc/clang spellings are also forwarded to the
                     preprocessor (`__STDC_VERSION__`, include search).
                     Default: permissive C17.
  -f[no-]gnu-extensions
  -f[no-]ms-extensions
  -f[no-]clang-extensions
                     gate the parser's vendor spellings (GNU: `__asm__`,
                     `__attribute__`, `__int128`; MSVC: `__declspec`,
                     `__cdecl`, `__intN`; clang: `_Float32`/`_Float64`).
                     All on by default.
Target (native by default):
  --target=T         compile for T: x86_64, aarch64, riscv64, or
                     x86_64-windows (short names or triples, e.g.
                     aarch64-linux-gnu or x86_64-w64-mingw32); forwards
                     --target= to clang and the link driver, and selects
                     the frontend sizes and the Cranelift ISA. Cross
                     linking needs a matching GLCC_CC. x86_64-windows
                     picks the MSVC ABI and COFF objects; on a Windows
                     host the final link drives link.exe (run from a
                     Developer Command Prompt for its library paths).
Debug (on by default; last -g* wins):
  -g (any level but -g0)  DWARF line info and subprograms
  -g0                  no debuginfo (still emits .eh_frame)
Accepted and ignored: -W*, -f* (except as handled above or below),
  -m64/-march=...,
  -pedantic, -w, -pipe, --param=, -B, -specs=
Warned and ignored: -flto*, -fsanitize*, -fprofile*, --coverage,
  -pg, unknown -... options
Errors: -m32/-mx32 (LP64 only), -x c-header (no PCH), C++ inputs
Environment: GLCC_PP (native preprocessor by default; 'clang' or a
  clang path selects clang -E, overridable per-invocation with
  --pp=native|--pp=clang), GLCC_CC (default cc; point at
  a cross compiler when using --target= for a foreign architecture;
  set it to keep the gcc-shaped link even on the Windows target),
  GLCC_LD (linker for the Windows target; default link.exe, e.g.
  lld-link), GLCC_AR (librarian for the Windows target; default
  lib.exe),
  GLCC_AS (assembler for .s and generated asm; default as, or the
  target's <triple>-as when installed for --target=; .S stays on GLCC_CC),
  GLCC_RT (runtime override for either target: glccrt.o, default
  ~/.local/lib/glcc/glccrt.o then /usr/lib/glcc/glccrt.o; or
  glccrt-win64.lib, default ~/.local/lib/glcc/glccrt-win64.lib)
  GLCPP_EDG (glcpp only: path to the edgcpp/compiler submodule;
  default is the edg/ dir beside the glcc crate)
Examples:
  glcc -O2 -Iinc -DFOO=1 -c foo.c -o foo.o
  glcc foo.o bar.o -Llib -lfoo -o prog
  make CC=glcc CFLAGS='-O2 -Wall'
"
    );
}

fn print_version(flavor: Flavor) {
    println!("{} {} (Cranelift backend; GCC-compatible command driver)", driver_name(flavor), env!("CARGO_PKG_VERSION"));
}

/// Run `cc <flag>` and print its stdout (for -print-search-dirs and
/// friends, where the system driver's answer is the compatible one).
/// `-dumpmachine` answers with the triple being compiled for, not the
/// host's.
fn delegate_cc(flag: &str, target: Target) -> i32 {
    if flag == "-dumpmachine" {
        println!("{}", target.triple());
        return 0;
    }
    match Command::new(link_driver()).arg(flag).output() {
        Ok(out) if out.status.success() => {
            print!("{}", String::from_utf8_lossy(&out.stdout));
            0
        }
        _ => {
            if flag == "-dumpversion" {
                println!("{}", env!("CARGO_PKG_VERSION"));
                return 0;
            }
            eprintln!("glcc: error: cannot query system driver for '{flag}'");
            1
        }
    }
}

/// Standalone queries, honored wherever they appear before `--`.
/// Returns `Some(code)` when the driver should exit immediately.
fn prescan(argv: &[OsString], flavor: Flavor) -> Option<i32> {
    let target = argv_target(argv);
    for a in argv {
        if a == "--" {
            break;
        }
        if a == "--help" {
            print_help(flavor);
            return Some(0);
        }
        if a == "--version" {
            print_version(flavor);
            return Some(0);
        }
        if a == "-dumpmachine" || a == "-dumpversion" {
            return Some(delegate_cc(a.to_str().unwrap(), target));
        }
        if a == "-print-search-dirs" {
            return Some(delegate_cc("-print-search-dirs", target));
        }
        if a.to_string_lossy().starts_with("-print-file-name=") {
            return Some(delegate_cc(&a.to_string_lossy(), target));
        }
    }
    None
}

/// The `--target=` selection in `argv` (stopping at `--`), or native
/// when absent or unparsable (parse errors surface later in `run`).
fn argv_target(argv: &[OsString]) -> Target {
    let mut target = Target::native();
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        if a == "--" {
            break;
        }
        let s = a.to_string_lossy();
        if let Some(rest) = s.strip_prefix("--target=") {
            if let Some(t) = Target::from_name(rest) {
                target = t;
            }
        } else if s == "--target" {
            if let Some(next) = it.next() {
                if let Some(t) = Target::from_name(&next.to_string_lossy()) {
                    target = t;
                }
            }
        }
    }
    target
}

/// Driver entry point: argument vector without `argv[0]`; returns the
/// process exit code.
pub fn run(argv: Vec<OsString>, flavor: Flavor) -> i32 {
    if let Some(code) = prescan(&argv, flavor) {
        return code;
    }
    let expanded = match expand_response_files(argv, &|p| std::fs::read_to_string(p)) {
        Ok(v) => v,
        Err(m) => {
            eprintln!("glcc: error: {m}");
            return 1;
        }
    };
    let mut job = match parse_args_flavor(&expanded, flavor) {
        Ok(j) => j,
        Err(m) => {
            eprintln!("glcc: error: {m}");
            return 1;
        }
    };
    if flavor == Flavor::Cpp && job.edg_root.is_none() {
        job.edg_root = resolve_edg_root();
    }
    if flavor == Flavor::Cpp {
        // The C++ runtime provides exception handling, RTTI, and
        // `new`/`delete`. Its `-L` must precede `-lC`, and both ride at
        // the front of the appended libs (after user objects, like g++).
        match resolve_libc_dir() {
            Some(dir) => {
                job.link_libs.insert(0, OsString::from("-lC"));
                job.link_libs.insert(0, OsString::from(format!("-L{}", dir.display())));
            }
            None if job.action == Action::Link => {
                eprintln!("glcpp: warning: libC.a not found (set GLCPP_LIB_DIR); C++ exception handling will fail to link");
            }
            None => {}
        }
    }
    for w in &job.warnings {
        eprintln!("glcc: warning: {w}");
    }
    if job.inputs.is_empty() && job.order.is_empty() {
        eprintln!("glcc: error: no input files");
        return 1;
    }
    let ctx = Ctx {
        verbose: job.verbose,
        dry_run: job.dry_run,
    };
    let outcome = match job.action {
        Action::Preprocess => do_preprocess(&ctx, &job),
        Action::Deps => do_deps(&ctx, &job),
        Action::SyntaxOnly => do_syntax_only(&ctx, &job),
        Action::EmitMir => do_emit_mir(&ctx, &job, false),
        Action::EmitFullMir => do_emit_mir(&ctx, &job, true),
        Action::Compile => do_compile(&ctx, &job, None).and_then(|(produced, _)| {
            if job.depgen.is_some() {
                // `-MT` defaults to the real outputs here.
                let units: Vec<(OsString, Option<OsString>)> = produced.iter().map(|(p, o)| (p.clone(), Some(o.as_os_str().to_os_string()))).collect();
                run_depgen(&ctx, &job, &units)
            } else {
                Ok(())
            }
        }),
        Action::Link => do_link(&ctx, &job),
    };
    match outcome {
        Ok(()) => 0,
        Err(Fail::Msg(m)) => {
            eprintln!("glcc: error: {m}");
            1
        }
        Err(Fail::Quiet) => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(words: &[&str]) -> Vec<OsString> {
        words.iter().map(OsString::from).collect()
    }

    fn parse(words: &[&str]) -> Job {
        parse_args(&args(words)).expect("parse failed")
    }

    fn parse_cpp(words: &[&str]) -> Job {
        parse_args_flavor(&args(words), Flavor::Cpp).expect("parse failed")
    }

    fn order_kind(job: &Job) -> Vec<String> {
        job.order
            .iter()
            .map(|item| match item {
                OrderItem::Input(i) => format!("in:{}", job.inputs[*i].path.to_string_lossy()),
                OrderItem::LinkArg(a) => format!("flag:{}", a.to_string_lossy()),
            })
            .collect()
    }

    #[test]
    fn compile_single_with_output() {
        let job = parse(&["-c", "foo.c", "-o", "foo.o"]);
        assert_eq!(job.action, Action::Compile);
        assert_eq!(job.inputs.len(), 1);
        assert_eq!(job.inputs[0].lang, Lang::C);
        assert_eq!(job.output.as_deref(), Some(OsStr::new("foo.o")));
        assert_eq!(order_kind(&job), ["in:foo.c"]);
    }

    #[test]
    fn std_and_extension_flags_reach_the_frontends() {
        let job = parse(&["-c", "-std=gnu11", "a.c"]);
        assert_eq!(job.lang_std, Some(Std::Gnu11));
        assert!(job.cpp.iter().any(|a| a.as_encoded_bytes() == b"-std=gnu11"));
        assert_eq!(language(&job).std, Std::Gnu11);

        let job = parse(&["-c", "-ansi", "a.c"]);
        assert_eq!(job.lang_std, Some(Std::C89));

        let job = parse(&["-c", "-std=c2x", "a.c"]);
        assert_eq!(job.lang_std, Some(Std::C23));
        assert!(language(&job).gnu, "flag doesn't imply the GNU gate");
    }

    #[test]
    fn extension_flags_toggle_lang_opts() {
        let job = parse(&["-c", "-fno-gnu-extensions", "-fms-extensions", "-fno-clang-extensions", "a.c"]);
        let lang = language(&job);
        assert!(!lang.gnu);
        assert!(lang.ms);
        assert!(!lang.clang);
        assert_eq!(lang.std, Std::Gnu17, "default level unchanged");

        let job = parse(&["-c", "-fgnu-extensions", "-fno-ms-extensions", "-fclang-extensions", "a.c"]);
        let lang = language(&job);
        assert!(lang.gnu);
        assert!(!lang.ms);
        assert!(lang.clang);
    }

    #[test]
    fn invalid_std_value_is_an_error() {
        let err = parse_args(&args(&["-c", "-std=c1y", "a.c"])).unwrap_err();
        assert!(err.contains("c1y"), "'{err}'");
    }

    #[test]
    fn link_order_is_argv_order() {
        let job = parse(&["a.c", "-lfoo", "b.o", "-Llib", "c.c", "-Wl,-z,relro"]);
        assert_eq!(job.action, Action::Link);
        assert_eq!(order_kind(&job), ["in:a.c", "flag:-lfoo", "flag:b.o", "flag:-Llib", "in:c.c", "flag:-Wl,-z,relro"]);
    }

    #[test]
    fn attached_and_separate_forms() {
        let job = parse(&["-Iinc", "-I", "other", "-DFOO=1", "-D", "BAR", "-oout", "a.c", "-lm", "-L", "lib"]);
        assert_eq!(job.cpp, ["-Iinc", "-Iother", "-DFOO=1", "-DBAR"]);
        assert_eq!(job.output.as_deref(), Some(OsStr::new("out")));
        assert_eq!(order_kind(&job), ["in:a.c", "flag:-lm", "flag:-Llib"]);
    }

    #[test]
    fn action_last_wins() {
        assert_eq!(parse(&["-c", "-E", "a.c"]).action, Action::Preprocess);
        assert_eq!(parse(&["-E", "-c", "a.c"]).action, Action::Compile);
        assert_eq!(parse(&["-c", "-M", "a.c"]).action, Action::Deps);
        assert_eq!(parse(&["-E", "--emit-mir", "a.c"]).action, Action::EmitMir);
        assert_eq!(parse(&["-E", "--emit-full-mir", "a.c"]).action, Action::EmitFullMir);
        assert_eq!(parse(&["--emit-full-mir", "--emit-mir", "a.c"]).action, Action::EmitMir);
        assert_eq!(parse(&["--emit-mir", "--emit-full-mir", "a.c"]).action, Action::EmitFullMir);
        assert_eq!(parse(&["--emit-mir", "-fsyntax-only", "a.c"]).action, Action::SyntaxOnly);
    }

    #[test]
    fn full_mir_annotates_instructions_with_source_locations() {
        let src = "# 1 \"t.c\"\nint add(int a, int b) { return a + b; }\nint main() { return add(2, 3); }\n";
        let plain = dump_mir_source(src, Flavor::C, Target::LP64, OptLevel::None, false, LangOpts::default(), ErrorFormat::Human, false).expect("plain dump");
        let full = dump_mir_source(src, Flavor::C, Target::LP64, OptLevel::None, false, LangOpts::default(), ErrorFormat::Human, true).expect("full dump");
        assert!(!plain.contains("t.c:"), "{plain}");
        assert!(full.contains("; t.c:"), "{full}");
        // The full dump is the compact text plus location suffixes:
        // every compact line survives verbatim.
        for line in plain.lines() {
            assert!(full.contains(line), "compact line missing from full dump:\n{line}");
        }
    }

    #[test]
    fn x_language_override() {
        let job = parse(&["-x", "c", "foo.o", "-x", "none", "bar.o", "-x", "assembler", "baz.c"]);
        assert_eq!(job.inputs.len(), 2);
        assert_eq!(job.inputs[0].lang, Lang::C);
        assert_eq!(job.inputs[1].lang, Lang::Assembler);
        assert_eq!(order_kind(&job), ["in:foo.o", "flag:bar.o", "in:baz.c"]);
        assert!(parse_args(&args(&["-x", "bogus", "a.c"])).is_err());
        assert!(parse_args(&args(&["-x", "c-header", "a.h"])).is_err());
    }

    #[test]
    fn unknown_options_warn_and_continue() {
        // Machine and warning flags are silently accepted (fixed
        // native target, no diagnostics to configure).
        let job = parse(&["-march=native", "-O2", "-Wall", "-fPIC", "-m64", "a.c"]);
        assert_eq!(job.inputs.len(), 1);
        assert!(job.warnings.is_empty());
        assert_eq!(job.cpp, ["-fPIC"]);
        let job = parse(&["-Zunknown-flag", "a.c"]);
        assert_eq!(job.inputs.len(), 1);
        assert_eq!(job.warnings, ["unrecognized option '-Zunknown-flag'; ignoring"]);
    }

    #[test]
    fn debug_flags() {
        // Debug is on by default; `-g0` disables, any other `-g*`
        // (re)enables. Last wins, like gcc.
        assert!(parse(&["a.c"]).debug);
        assert!(parse(&["-g", "a.c"]).debug);
        assert!(parse(&["-g3", "a.c"]).debug);
        assert!(parse(&["-ggdb", "a.c"]).debug);
        assert!(!parse(&["-g0", "a.c"]).debug);
        assert!(!parse(&["-g", "-g0", "a.c"]).debug);
        assert!(parse(&["-g0", "-g", "a.c"]).debug);
        let job = parse(&["-O2", "-g0", "a.c"]);
        assert!(!job.debug);
    }

    /// Byte scan for a section name in an emitted object (ELF section
    /// header string tables spell names in full).
    fn has_section(bytes: &[u8], name: &[u8]) -> bool {
        bytes.windows(name.len()).any(|w| w == name)
    }

    #[test]
    fn debug_flag_controls_debug_sections() {
        let src = "int add(int a, int b) { return a + b; }\nint main() { return add(2, 3); }\n";
        let compile = |debug| {
            compile_source(src, Flavor::C, Target::LP64, debug, true, OptLevel::None, false, LangOpts::default(), ErrorFormat::Human)
                .expect("compiles")
                .0
        };
        let with = compile(true);
        assert!(has_section(&with, b".debug_info"), "missing .debug_info with -g");
        assert!(has_section(&with, b".debug_line"), "missing .debug_line with -g");
        // `cranelift-object` emits unwind only for ELF (`.eh_frame`);
        // the driver's COFF `.pdata`/`.xdata` path lives in winunwind.
        #[cfg(not(windows))]
        assert!(has_section(&with, b".eh_frame"), "missing .eh_frame with -g");
        let without = compile(false);
        assert!(!has_section(&without, b".debug_info"), "unexpected .debug_info with -g0");
        assert!(!has_section(&without, b".debug_line"), "unexpected .debug_line with -g0");
        #[cfg(not(windows))]
        assert!(has_section(&without, b".eh_frame"), "missing .eh_frame with -g0");
        // Optimized builds still carry debug info through the MIR
        // passes (locations ride on instructions).
        let optimized = compile_source(src, Flavor::C, Target::LP64, true, true, OptLevel::Speed, false, LangOpts::default(), ErrorFormat::Human)
            .expect("compiles")
            .0;
        assert!(has_section(&optimized, b".debug_info"), "missing .debug_info at -O2 -g");
        assert!(has_section(&optimized, b".debug_line"), "missing .debug_line at -O2 -g");
    }

    #[test]
    fn debug_sections_survive_empty_functions() {
        // A function with no statements still gets a subprogram and an
        // (empty) line sequence rather than failing emission.
        let src = "void e() {}\nint main() { e(); return 0; }\n";
        let bytes = compile_source(src, Flavor::C, Target::LP64, true, true, OptLevel::None, false, LangOpts::default(), ErrorFormat::Human)
            .expect("compiles")
            .0;
        assert!(has_section(&bytes, b".debug_info"));
        assert!(has_section(&bytes, b".debug_line"));
    }

    /// An `extern` declaration names storage defined elsewhere: its DIE
    /// must not carry a location, or the relocation in `.debug_info`
    /// references an undefined symbol and fails the link even when the
    /// variable is unused (c-testsuite 00094).
    #[test]
    fn debug_extern_declaration_has_no_location() {
        let src = "extern int x;\nint main() { return 0; }\n";
        let bytes = compile_source(src, Flavor::C, Target::LP64, true, true, OptLevel::None, false, LangOpts::default(), ErrorFormat::Human)
            .expect("compiles")
            .0;
        let (dies, _) = read_dies(&bytes);
        let x = find(&dies, gimli::constants::DW_TAG_variable, "x");
        assert!(x.flags.contains(&gimli::constants::DW_AT_declaration));
        assert!(x.flags.contains(&gimli::constants::DW_AT_external));
        assert!(!x.has_location);

        use object::{Object as _, ObjectSection as _, ObjectSymbol as _};
        let obj = object::File::parse(&bytes[..]).expect("object parses");
        for section in obj.sections() {
            if !section.name().unwrap_or("").starts_with(".debug") {
                continue;
            }
            for (_, reloc) in section.relocations() {
                if let object::RelocationTarget::Symbol(idx) = reloc.target() {
                    let name = obj.symbol_by_index(idx).ok().and_then(|s| s.name().ok());
                    assert_ne!(name, Some("x"), "relocation against `extern int x`");
                }
            }
        }
    }

    /// A `__ptr32` object holding an address patches 4 bytes, but
    /// Cranelift's data descriptor only emits pointer-width
    /// relocations, so the driver adds a 32-bit absolute reloc itself
    /// after `finish`.
    #[test]
    fn ptr32_global_initializer_emits_a_32_bit_relocation() {
        let src = "int x;\nint *__ptr32 g = &x;\nint main(void) { return 0; }\n";
        let lang = LangOpts { ms: true, ..LangOpts::default() };
        let bytes = compile_source(src, Flavor::C, Target::LP64, false, true, OptLevel::None, false, lang, ErrorFormat::Human)
            .expect("compiles")
            .0;
        use object::{Object as _, ObjectSection as _, ObjectSymbol as _};
        let obj = object::File::parse(&bytes[..]).expect("object parses");
        let g = obj.symbol_by_name("g").expect("g is defined");
        assert_eq!(g.size(), 4, "a __ptr32 object is 4 bytes");
        let mut relocs = Vec::new();
        for section in obj.sections() {
            for (_, reloc) in section.relocations() {
                if let object::RelocationTarget::Symbol(idx) = reloc.target() {
                    if obj.symbol_by_index(idx).ok().and_then(|s| s.name().ok()) == Some("x") {
                        relocs.push(reloc);
                    }
                }
            }
        }
        assert_eq!(relocs.len(), 1, "one patch for the __ptr32 pointer");
        assert_eq!(relocs[0].size(), 32, "the patch is 32 bits wide");
        assert_eq!(relocs[0].kind(), object::RelocationKind::Absolute);
    }

    /// A variable whose slot the optimizer promoted has no home in the
    /// emitted code. Its DIE must survive with a name and type but no
    /// location, so a debugger reports "optimized out" (and `ptype`
    /// still resolves) instead of "No symbol".
    #[test]
    fn debug_optimized_out_variables_keep_their_die() {
        use gimli::constants::*;
        let src = "__attribute__((noinline)) int f(int a) {\n\
                       int t = a + 1;\n\
                       return t * 2;\n\
                   }\n\
                   int main(void) { return f(3); }\n";
        let bytes = compile_source(src, Flavor::C, Target::LP64, true, true, OptLevel::Speed, false, LangOpts::default(), ErrorFormat::Human)
            .expect("compiles")
            .0;
        let (dies, _) = read_dies(&bytes);
        let param = find(&dies, DW_TAG_formal_parameter, "a");
        let local = find(&dies, DW_TAG_variable, "t");
        assert!(!param.has_location, "promoted parameter kept a location");
        assert!(!local.has_location, "promoted local kept a location");
        assert!(!param.flags.contains(&DW_AT_declaration), "parameter marked a declaration");
        assert!(!local.flags.contains(&DW_AT_declaration), "local marked a declaration");
    }

    /// Windows unwind tables: compiling for x86_64-windows appends
    /// `.pdata` (one 12-byte RUNTIME_FUNCTION per defined function, in
    /// ascending `.text` order, every field
    /// `IMAGE_REL_AMD64_ADDR32NB`) and `.xdata` (serialized
    /// UNWIND_INFO records). Windows' unwinder (`RtlVirtualUnwind`,
    /// `RtlCaptureStackBackTrace`, SEH) walks these tables; without
    /// them stack walks stop at the first glcc frame. Non-Windows
    /// targets emit neither section.
    #[test]
    fn windows_target_emits_unwind_tables() {
        use object::{Object as _, ObjectSection as _, ObjectSymbol as _};

        let src = "int f(int x) { return x + 1; }\nint g(void) { return f(2); }\nint main(void) { return g(); }\n";
        let bytes = compile_source(src, Flavor::C, Target::WINDOWS_X86_64, false, true, OptLevel::None, false, LangOpts::default(), ErrorFormat::Human)
            .expect("compiles")
            .0;
        let obj = object::File::parse(&bytes[..]).expect("object parses");
        assert_eq!(obj.format(), object::BinaryFormat::Coff);

        let pdata = obj.section_by_name(".pdata").expect(".pdata");
        let xdata = obj.section_by_name(".xdata").expect(".xdata");
        let xdata_data = xdata.data().expect("xdata data");
        let rows = pdata.data().expect("pdata data");
        assert_eq!(rows.len(), 3 * 12, "one RUNTIME_FUNCTION per defined function");

        // Rows must be sorted by begin address; definition order is
        // ascending `.text` order, visible as sorted symbol addresses.
        let text = obj.section_by_name(".text").expect(".text");
        let mut funcs: Vec<(u64, String)> = obj
            .symbols()
            .filter(|s| s.kind() == object::SymbolKind::Text)
            .filter(|s| s.section() == object::SymbolSection::Section(text.index()))
            .map(|s| (s.address(), s.name().unwrap_or("").to_string()))
            .collect();
        funcs.sort();
        assert_eq!(funcs.len(), 3, "f, g and main");

        // Relocations at rows[i*12 + {0,4,8}] name begin, end and
        // unwind addresses.
        let relocs: Vec<(u64, object::Relocation)> = pdata.relocations().collect();
        assert_eq!(relocs.len(), 3 * 3, "three fields per row");
        let sym_of = |reloc: &object::Relocation| -> (String, Option<object::SectionIndex>, u64) {
            let object::RelocationTarget::Symbol(idx) = reloc.target() else {
                panic!("expected a symbol relocation target");
            };
            let sym = obj.symbol_by_index(idx).expect("reloc symbol");
            let name = sym.name().unwrap_or("").to_string();
            (name, sym.section().index(), sym.address())
        };
        for (i, (_, func_name)) in funcs.iter().enumerate() {
            let row = i as u64 * 12;
            let at = |field: u64| -> &object::Relocation { &relocs.iter().find(|(o, _)| *o == row + field).unwrap_or_else(|| panic!("reloc at row {i} + {field}")).1 };
            for field in [0, 4, 8] {
                match at(field).flags() {
                    object::RelocationFlags::Coff { typ } => assert_eq!(typ, object::pe::IMAGE_REL_AMD64_ADDR32NB),
                    other => panic!("expected a COFF relocation kind, got {other:?}"),
                }
            }
            let (begin, _, _) = sym_of(at(0));
            let (end, _, _) = sym_of(at(4));
            let (unwind, unwind_section, anchor) = sym_of(at(8));
            assert_eq!(begin, *func_name, "row {i} begins with the {i}th function in .text order");
            assert_eq!(end, begin, "row {i} end address relocates against the same symbol");
            assert_eq!(unwind_section, Some(xdata.index()), "row {i} unwind field points into .xdata");

            let begin_addend = u32::from_le_bytes(rows[row as usize..row as usize + 4].try_into().expect("begin field"));
            let end_addend = u32::from_le_bytes(rows[row as usize + 4..row as usize + 8].try_into().expect("end field"));
            assert_eq!(begin_addend, 0);
            assert!(end_addend > 0, "row {i} end address addend is the function size");
            let anchor = anchor as usize;
            assert!(anchor + 4 <= xdata_data.len(), "row {i} anchor is inside .xdata");
            assert_eq!(xdata_data[anchor] & 0x7, 1, "row {i} anchors a version 1 UNWIND_INFO ({unwind})");
        }

        // Other targets must not grow the sections.
        let bytes = compile_source(src, Flavor::C, Target::AARCH64, false, true, OptLevel::None, false, LangOpts::default(), ErrorFormat::Human)
            .expect("compiles")
            .0;
        let obj = object::File::parse(&bytes[..]).expect("object parses");
        assert_eq!(obj.format(), object::BinaryFormat::Elf);
        assert!(obj.section_by_name(".pdata").is_none());
    }

    /// DIE summary for DWARF read-back: enough to pin names, original
    /// lines, and layout facts without gdb.
    struct DieSummary {
        depth: u32,
        tag: gimli::constants::DwTag,
        name: Option<String>,
        decl_line: Option<u64>,
        udata_attrs: Vec<(gimli::constants::DwAt, u64)>,
        sdata_attrs: Vec<(gimli::constants::DwAt, i64)>,
        flags: Vec<gimli::constants::DwAt>,
        has_location: bool,
    }

    /// Parse `bytes` as an object and summarize every DIE plus the
    /// non-empty line rows of the line program.
    fn read_dies(bytes: &[u8]) -> (Vec<DieSummary>, Vec<u64>) {
        use gimli::read::EndianSlice;
        use object::{Object as _, ObjectSection as _};

        let obj = object::File::parse(bytes).expect("emitted object parses");
        let load = |name: &str| -> Vec<u8> {
            obj.section_by_name(name)
                .and_then(|s| s.uncompressed_data().ok())
                .map(|d| d.into_owned())
                .unwrap_or_default()
        };
        let info = load(".debug_info");
        let abbrev = load(".debug_abbrev");
        let line = load(".debug_line");
        let endian = gimli::RunTimeEndian::Little;
        let dwarf = gimli::read::Dwarf::load(|id: gimli::SectionId| -> Result<EndianSlice<'_, gimli::RunTimeEndian>, gimli::Error> {
            let data: &[u8] = match id {
                gimli::SectionId::DebugInfo => &info,
                gimli::SectionId::DebugAbbrev => &abbrev,
                gimli::SectionId::DebugLine => &line,
                _ => &[],
            };
            Ok(EndianSlice::new(data, endian))
        })
        .expect("DWARF parses");
        let mut out = Vec::new();
        let mut rows = Vec::new();
        let mut units = dwarf.units();
        while let Some(header) = units.next().expect("unit header") {
            let unit = dwarf.unit(header).expect("unit");
            if let Some(program) = unit.line_program.clone() {
                let mut iter = program.rows();
                while let Some((_, row)) = iter.next_row().expect("line row") {
                    if !row.end_sequence()
                        && let Some(n) = row.line()
                    {
                        rows.push(n.get());
                    }
                }
            }
            let mut tree = unit.entries_tree(None).expect("tree");
            let root = tree.root().expect("root");
            fn walk<R: gimli::read::Reader>(node: gimli::read::EntriesTreeNode<'_, '_, R>, depth: u32, out: &mut Vec<DieSummary>) -> Result<(), gimli::Error> {
                let entry = node.entry();
                let mut udata_attrs = Vec::new();
                let mut sdata_attrs = Vec::new();
                let mut flags = Vec::new();
                for attr in entry.attrs() {
                    match attr.value() {
                        gimli::read::AttributeValue::Udata(n) => udata_attrs.push((attr.name(), n)),
                        gimli::read::AttributeValue::Sdata(n) => sdata_attrs.push((attr.name(), n)),
                        gimli::read::AttributeValue::Flag(true) => flags.push(attr.name()),
                        _ => {}
                    }
                }
                let name = entry.attr_value(gimli::constants::DW_AT_name).and_then(|v| match v {
                    gimli::read::AttributeValue::String(s) => s.to_string_lossy().ok().map(|s| s.into_owned()),
                    _ => None,
                });
                let decl_line = entry.attr_value(gimli::constants::DW_AT_decl_line).and_then(|v| match v {
                    gimli::read::AttributeValue::Udata(n) => Some(n),
                    _ => None,
                });
                let has_location = entry.attr_value(gimli::constants::DW_AT_location).is_some();
                out.push(DieSummary {
                    depth,
                    tag: entry.tag(),
                    name,
                    decl_line,
                    udata_attrs,
                    sdata_attrs,
                    flags,
                    has_location,
                });
                let mut children = node.children();
                while let Some(child) = children.next()? {
                    walk(child, depth + 1, out)?;
                }
                Ok(())
            }
            walk(root, 0, &mut out).expect("walk");
        }
        (out, rows)
    }

    fn find<'a>(dies: &'a [DieSummary], tag: gimli::constants::DwTag, name: &str) -> &'a DieSummary {
        dies.iter()
            .find(|d| d.tag == tag && d.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("missing {tag:?} DIE for {name}"))
    }

    fn udata(die: &DieSummary, at: gimli::constants::DwAt) -> u64 {
        die.udata_attrs
            .iter()
            .find(|(name, _)| *name == at)
            .map(|(_, n)| *n)
            .unwrap_or_else(|| panic!("missing {at:?} on {}", die.name.as_deref().unwrap_or("?")))
    }

    /// `DW_AT_const_value` as written for either signed or unsigned
    /// enumerator representations.
    fn const_value(die: &DieSummary) -> i64 {
        use gimli::constants::DW_AT_const_value as AT;
        die.udata_attrs
            .iter()
            .find(|(name, _)| *name == AT)
            .map(|(_, n)| *n as i64)
            .or_else(|| die.sdata_attrs.iter().find(|(name, _)| *name == AT).map(|(_, n)| *n))
            .unwrap_or_else(|| panic!("missing DW_AT_const_value on {}", die.name.as_deref().unwrap_or("?")))
    }

    #[test]
    fn debug_info_records_types_vars_and_original_lines() {
        use gimli::constants::*;
        // Linemarker: the next line is `orig.c:100`, so DWARF must
        // store original lines, not TU-buffer lines.
        let src = "# 100 \"orig.c\"\n\
             struct Point { int x; int y; };\n\
             enum Color { RED, GREEN = 5, BLUE };\n\
             struct Bits { unsigned a : 3; unsigned b : 5; };\n\
             int g;\n\
             int add(int a, int b) {\n\
                 int total = a + b;\n\
                 struct Point p = {total, b};\n\
                 struct Bits bs;\n\
                 bs.a = 1;\n\
                 enum Color col = GREEN;\n\
                 return total + p.x + bs.a + col + g;\n\
             }\n\
             int main() { g = 1; return add(2, 3); }\n";
        let bytes = compile_source(src, Flavor::C, Target::LP64, true, true, OptLevel::None, false, LangOpts::default(), ErrorFormat::Human)
            .expect("compiles")
            .0;
        assert!(has_section(&bytes, b"orig.c"), "line program lost the original file name");
        let (dies, rows) = read_dies(&bytes);
        let add = find(&dies, DW_TAG_subprogram, "add");
        assert_eq!(add.depth, 1);
        assert_eq!(add.decl_line, Some(104));
        assert_eq!(find(&dies, DW_TAG_subprogram, "main").decl_line, Some(112));
        let params: Vec<_> = dies
            .iter()
            .filter(|d| d.tag == DW_TAG_formal_parameter && d.depth == 2)
            .map(|d| d.name.clone().unwrap())
            .collect();
        assert_eq!(params, ["a", "b"]);
        assert_eq!(find(&dies, DW_TAG_variable, "total").decl_line, Some(105));
        let global = find(&dies, DW_TAG_variable, "g");
        assert_eq!(global.depth, 1);
        assert_eq!(global.decl_line, Some(103));
        // Struct members keep byte offsets; bit-fields carry widths and
        // struct-relative bit offsets.
        let members: Vec<_> = dies.iter().filter(|d| d.tag == DW_TAG_member && d.depth == 2).collect();
        let px = members.iter().find(|d| d.name.as_deref() == Some("x")).expect("Point.x");
        let py = members.iter().find(|d| d.name.as_deref() == Some("y")).expect("Point.y");
        assert_eq!(udata(px, DW_AT_data_member_location), 0);
        assert_eq!(udata(py, DW_AT_data_member_location), 4);
        let bit = |name: &str| {
            members
                .iter()
                .find(|d| d.name.as_deref() == Some(name) && d.udata_attrs.iter().any(|(at, _)| *at == DW_AT_bit_size))
                .unwrap_or_else(|| panic!("Bits.{name}"))
        };
        assert_eq!(udata(bit("a"), DW_AT_bit_size), 3);
        assert_eq!(udata(bit("a"), DW_AT_data_bit_offset), 0);
        assert_eq!(udata(bit("b"), DW_AT_bit_size), 5);
        assert_eq!(udata(bit("b"), DW_AT_data_bit_offset), 3);
        // Enumerators fold like the parser: explicit values evaluate,
        // implicit ones continue.
        let enumerator = |name: &str| find(&dies, DW_TAG_enumerator, name);
        assert_eq!(const_value(enumerator("RED")), 0);
        assert_eq!(const_value(enumerator("GREEN")), 5);
        assert_eq!(const_value(enumerator("BLUE")), 6);
        // Line rows carry the same original lines as the DIEs.
        assert!(rows.contains(&104) && rows.contains(&105) && rows.contains(&112), "missing original lines in {rows:?}");
    }

    #[test]
    fn opt_flags() {
        // Default is `-O0`.
        assert_eq!(parse(&["a.c"]).opt_level, OptLevel::None);
        assert_eq!(parse(&["-O0", "a.c"]).opt_level, OptLevel::None);
        assert_eq!(parse(&["-O", "a.c"]).opt_level, OptLevel::Speed);
        assert_eq!(parse(&["-O1", "a.c"]).opt_level, OptLevel::Speed);
        assert_eq!(parse(&["-O2", "a.c"]).opt_level, OptLevel::Speed);
        assert_eq!(parse(&["-O3", "a.c"]).opt_level, OptLevel::Speed);
        assert_eq!(parse(&["-Og", "a.c"]).opt_level, OptLevel::Speed);
        assert_eq!(parse(&["-Ofast", "a.c"]).opt_level, OptLevel::Speed);
        assert_eq!(parse(&["-Os", "a.c"]).opt_level, OptLevel::Size);
        assert_eq!(parse(&["-Oz", "a.c"]).opt_level, OptLevel::Size);
        // Last wins, like gcc.
        assert_eq!(parse(&["-O2", "-O0", "a.c"]).opt_level, OptLevel::None);
        assert_eq!(parse(&["-O0", "-O2", "a.c"]).opt_level, OptLevel::Speed);
        assert_eq!(parse(&["-O2", "-Os", "a.c"]).opt_level, OptLevel::Size);
    }

    #[test]
    fn pic_flags_last_wins() {
        // PIC is the default for PIE/shared links.
        assert!(parse(&["a.c"]).pic);
        for flag in ["-fpic", "-fPIC", "-fpie", "-fPIE"] {
            assert!(parse(&[flag, "a.c"]).pic, "{flag}");
        }
        for flag in ["-fno-pic", "-fno-PIC", "-fno-pie", "-fno-PIE"] {
            assert!(!parse(&[flag, "a.c"]).pic, "{flag}");
        }
        assert!(!parse(&["-fPIC", "-fno-pic", "a.c"]).pic);
        assert!(parse(&["-fno-pic", "-fPIE", "a.c"]).pic);
    }

    #[test]
    fn string_literals_are_emitted_read_only() {
        use object::{Object as _, ObjectSection as _, ObjectSymbol as _};

        let src = "int main(void) { char *p = \"abc\"; char *e = \"\"; return p[0] + e[0]; }";
        let bytes = compile_source(src, Flavor::C, Target::LP64, false, true, OptLevel::None, false, LangOpts::default(), ErrorFormat::Human)
            .expect("compiles")
            .0;
        let obj = object::File::parse(&bytes[..]).expect("object parses");
        let rodata = obj.section_by_name(".rodata").expect(".rodata section");
        let literals: Vec<_> = obj
            .symbols()
            .filter(|symbol| symbol.name().is_ok_and(|name| name.starts_with("__str.")))
            .collect();
        assert_eq!(literals.len(), 2, "both literals are materialized");
        for symbol in literals {
            assert_eq!(
                symbol.section(),
                object::SymbolSection::Section(rodata.index()),
                "{} must be read-only",
                symbol.name().unwrap()
            );
        }
    }

    #[test]
    fn non_pic_globals_avoid_the_got() {
        let reloc_kinds = |pic| {
            use object::{Object as _, ObjectSection as _};
            let src = "int x; int get(void) { return x; }";
            let bytes = compile_source(src, Flavor::C, Target::LP64, false, pic, OptLevel::None, false, LangOpts::default(), ErrorFormat::Human)
                .expect("compiles")
                .0;
            let obj = object::File::parse(&bytes[..]).expect("object parses");
            obj.section_by_name(".text")
                .expect(".text")
                .relocations()
                .map(|(_, reloc)| reloc.kind())
                .collect::<Vec<_>>()
        };
        assert!(reloc_kinds(true).contains(&object::RelocationKind::GotRelative));
        assert!(!reloc_kinds(false).contains(&object::RelocationKind::GotRelative));
    }

    #[test]
    fn target_flag() {
        use glcc_frontend::target::Target;
        // Native by default.
        assert_eq!(parse(&["a.c"]).target, Target::native());
        // Attached and separate spellings, short names and triples.
        assert_eq!(parse(&["--target=aarch64", "a.c"]).target, Target::AARCH64);
        assert_eq!(parse(&["--target", "riscv64", "a.c"]).target, Target::RISCV64);
        assert_eq!(parse(&["--target=aarch64-linux-gnu", "a.c"]).target, Target::AARCH64);
        assert_eq!(parse(&["--target=x86_64", "a.c"]).target, Target::X86_64);
        // Last wins, like -O.
        assert_eq!(parse(&["--target=aarch64", "--target=riscv64", "a.c"]).target, Target::RISCV64);
        // Unknown targets are hard errors.
        assert!(parse_args(&args(&["--target=mips", "a.c"])).is_err());
        assert!(parse_args(&args(&["--target", "bogus", "a.c"])).is_err());
        // Native forwards nothing; cross forwards the triple. The suite
        // runs on x86_64 and aarch64 hosts, so use a target that is
        // cross relative to this host.
        assert!(parse(&["a.c"]).target_args().is_empty());
        let cross = if Target::native() == Target::AARCH64 { Target::RISCV64 } else { Target::AARCH64 };
        let flag = format!("--target={}", cross.name());
        let expected = format!("--target={}", cross.triple());
        assert_eq!(parse(&[flag.as_str(), "a.c"]).target_args(), [expected.as_str()]);
        let (_, pp) = preprocess_cmd(&parse(&[flag.as_str(), "a.c"]), OsStr::new("a.c"));
        let s: Vec<String> = pp.iter().map(|a| a.to_string_lossy().into_owned()).collect();
        assert_eq!(s, ["-E", expected.as_str(), "a.c"]);
    }

    #[test]
    fn hard_errors() {
        assert!(parse_args(&args(&["-m32", "a.c"])).is_err());
        assert!(parse_args(&args(&["-S", "a.c"])).is_err());
        assert!(parse_args(&args(&["-"])).is_err());
        assert!(parse_args(&args(&["-o"])).is_err());
        assert!(parse_args(&args(&["foo.cpp"])).is_err());
        assert!(parse_args(&args(&["foo.txt"])).is_err());
        // Bare `-c` parses; "no input files" is a run-time error.
        assert!(parse_args(&args(&["-c"])).unwrap().inputs.is_empty());
    }

    #[test]
    fn error_format_selects_human_or_json() {
        assert_eq!(parse(&["a.c"]).error_format, ErrorFormat::Human);
        assert_eq!(parse(&["--error-format=json", "a.c"]).error_format, ErrorFormat::Json);
        assert_eq!(parse(&["--error-format", "human", "a.c"]).error_format, ErrorFormat::Human);
        // Last wins, like other repeated options.
        assert_eq!(
            parse(&["--error-format=json", "--error-format=human", "a.c"]).error_format,
            ErrorFormat::Human
        );
        assert!(parse_args(&args(&["--error-format=bogus", "a.c"])).is_err());
    }

    #[test]
    fn save_temps_flags() {
        // Off by default; bare and `=cwd` write to the working
        // directory, `=obj` beside each `-c` object. Last wins.
        assert_eq!(parse(&["a.c"]).save_temps, SaveTemps::Off);
        assert_eq!(parse(&["-save-temps", "a.c"]).save_temps, SaveTemps::Cwd);
        assert_eq!(parse(&["-save-temps=cwd", "a.c"]).save_temps, SaveTemps::Cwd);
        assert_eq!(parse(&["-save-temps=obj", "a.c"]).save_temps, SaveTemps::Obj);
        assert_eq!(parse(&["-save-temps=obj", "-save-temps", "a.c"]).save_temps, SaveTemps::Cwd);
        assert!(parse_args(&args(&["-save-temps=bogus", "a.c"])).is_err());
        assert!(parse_args(&args(&["-save-tempsfoo", "a.c"])).is_err());
    }

    #[test]
    fn pp_selection_flag_and_env() {
        // Auto (`None`) resolves per invocation; explicit flags win.
        assert_eq!(parse(&["a.c"]).pp_native, None);
        assert_eq!(parse(&["--pp=native", "a.c"]).pp_native, Some(true));
        assert_eq!(parse(&["--pp", "clang", "a.c"]).pp_native, Some(false));
        // Last wins, like -O.
        assert_eq!(parse(&["--pp=native", "--pp=clang", "a.c"]).pp_native, Some(false));
        assert!(parse_args(&args(&["--pp=bogus", "a.c"])).is_err());
        // Flag beats env: --pp=clang disables even with GLCC_PP=native.
        // (Env itself is not mutated here; use_native_pp falls back to it.)
        assert!(use_native_pp(&parse(&["--pp=native", "a.c"])));
        assert!(!use_native_pp(&parse(&["--pp=clang", "a.c"])));
        // No `--pp=`, no `GLCC_PP`: the native preprocessor.
        assert!(native_pp_default(None));
        assert!(native_pp_default(Some(OsStr::new(""))));
        assert!(native_pp_default(Some(OsStr::new("native"))));
        assert!(!native_pp_default(Some(OsStr::new("clang"))));
        assert!(!native_pp_default(Some(OsStr::new("/usr/bin/clang-22"))));
        // clang-only preprocessor output modes keep the clang path.
        assert!(!pp_flags_native_compatible(&[OsString::from("-dM")]));
        assert!(!pp_flags_native_compatible(&[OsString::from("-CC")]));
        assert!(pp_flags_native_compatible(&[OsString::from("-P")]));
        assert!(pp_flags_native_compatible(&[OsString::from("-Iinc")]));
    }

    #[test]
    fn edg_c_flag_routes_c_inputs_through_edg_c_frontend() {
        // Without the flag, `.c` compiles as C++ under glcpp.
        assert_eq!(parse_cpp(&["a.c"]).inputs[0].lang, Lang::Cpp);
        assert_eq!(parse_cpp(&["a.i"]).inputs[0].lang, Lang::Preprocessed);
        // The flag is position-independent: before or after the file.
        for argv in [["--edg-c", "a.c"], ["a.c", "--edg-c"]] {
            let job = parse_cpp(&argv);
            assert!(job.force_edg_c);
            assert_eq!(job.inputs[0].lang, Lang::EdgC);
        }
        // Preprocessed and `-x c` inputs route through EDG's C mode too.
        assert_eq!(parse_cpp(&["--edg-c", "a.i"]).inputs[0].lang, Lang::EdgC);
        assert_eq!(parse_cpp(&["--edg-c", "-x", "c", "a.o"]).inputs[0].lang, Lang::EdgC);
        // `-x c` without the flag stays on the native pipeline.
        assert_eq!(parse_cpp(&["-x", "c", "a.o"]).inputs[0].lang, Lang::C);
        // C++ inputs are unaffected by the flag.
        assert_eq!(parse_cpp(&["--edg-c", "a.cc"]).inputs[0].lang, Lang::Cpp);
        // glcc cannot honor it: EDG is not linked in.
        assert!(parse_args(&args(&["--edg-c", "a.c"])).is_err());
    }

    #[test]
    fn file_scope_asm_becomes_assembler_text() {
        let src = "asm(\".globl x\\nx:\\n  ret\\n\");\nint f(void){return 0;}\n";
        let (bytes, asm_text) = compile_source(src, Flavor::C, Target::LP64, false, true, OptLevel::None, false, LangOpts::default(), ErrorFormat::Human).expect("compiles");
        assert!(!bytes.is_empty());
        assert_eq!(asm_text, ".globl x\nx:\n  ret\n\n");
    }

    #[test]
    fn inline_asm_wrappers_become_assembler_text() {
        let src = "int f(int x){ __asm__(\"addl $1, %0\" : \"+r\"(x)); return x; }\n\
                   int g(int x){ __asm__(\"addl $1, %0\" : \"+r\"(x)); return x; }\n";
        let (bytes, asm_text) = compile_source(src, Flavor::C, Target::LP64, false, true, OptLevel::None, false, LangOpts::default(), ErrorFormat::Human).expect("compiles");
        assert!(!bytes.is_empty());
        assert!(asm_text.contains("addl $1, %eax"), "{asm_text}");
        // Identical blocks share one content-addressed wrapper.
        assert_eq!(asm_text.matches(".weak __glcc_asm_").count(), 1, "{asm_text}");
    }

    #[test]
    fn asm_goto_wrapper_returns_a_selector() {
        let src = "int f(int x){ asm goto(\"testl %0, %0; jnz %l[lab]\" : : \"r\"(x) : : lab); return 1; lab: return 2; }\n";
        let (bytes, asm_text) = compile_source(src, Flavor::C, Target::LP64, false, true, OptLevel::None, false, LangOpts::default(), ErrorFormat::Human).expect("compiles");
        assert!(!bytes.is_empty());
        assert!(asm_text.contains("jnz .Lasm"), "{asm_text}");
        assert!(asm_text.contains("movq $1,"), "{asm_text}");
    }

    #[test]
    fn raw_asm_goes_to_the_assembler() {
        if std::env::var_os("GLCC_AS").is_some() {
            return; // the environment picks the program; nothing stable to assert
        }
        let job = parse(&["-Wa,-adhln,-k", "a.s"]);
        let (prog, args) = assemble_cmd(&job, OsStr::new("a.s"), Path::new("a.o"));
        let args: Vec<String> = args.iter().map(|a| a.to_string_lossy().into_owned()).collect();
        if tool_available("as") {
            assert_eq!(prog, "as");
            assert_eq!(args, ["-adhln", "-k", "-o", "a.o", "a.s"]);
        } else {
            assert_eq!(prog, link_driver());
            assert_eq!(args, ["-c", "-Wa,-adhln,-k", "-o", "a.o", "a.s"]);
        }
    }

    #[test]
    fn preprocessed_asm_stays_on_the_link_driver() {
        let job = parse(&["a.S"]);
        let (prog, args) = assemble_cmd(&job, OsStr::new("a.S"), Path::new("a.o"));
        assert_eq!(prog, link_driver());
        let args: Vec<String> = args.iter().map(|a| a.to_string_lossy().into_owned()).collect();
        assert_eq!(args, ["-c", "-o", "a.o", "a.S"]);
    }

    #[test]
    fn link_mode_flags() {
        let job = parse(&["-shared", "-fPIC", "a.c", "-o", "lib.so"]);
        assert!(job.shared);
        assert_eq!(order_kind(&job), ["flag:-shared", "in:a.c"]);
        let job = parse(&["a.c", "-static", "-pthread", "-s"]);
        assert_eq!(order_kind(&job), ["in:a.c", "flag:-static", "flag:-pthread", "flag:-s"]);
        let job = parse(&["-Xlinker", "--no-as-needed", "-lfoo", "a.c"]);
        assert_eq!(order_kind(&job), ["flag:-Xlinker", "flag:--no-as-needed", "flag:-lfoo", "in:a.c"]);
        let job = parse(&["-fuse-ld=lld", "a.c"]);
        assert_eq!(job.use_ld.as_deref(), Some(OsStr::new("lld")));
    }

    #[test]
    fn artifact_detection() {
        let made = |extra: &[&str]| {
            let mut v = vec!["a.c"];
            v.extend(extra);
            parse(&v)
        };
        assert_eq!(decide_artifact(&made(&[])).unwrap(), Artifact::Executable);
        assert_eq!(decide_artifact(&made(&["-o", "prog"])).unwrap(), Artifact::Executable);
        assert_eq!(decide_artifact(&made(&["-o", "lib.a"])).unwrap(), Artifact::Archive);
        assert_eq!(decide_artifact(&made(&["-o", "lib.so"])).unwrap(), Artifact::Shared);
        assert_eq!(decide_artifact(&made(&["-o", "lib.so.1"])).unwrap(), Artifact::Shared);
        assert_eq!(decide_artifact(&made(&["-shared"])).unwrap(), Artifact::Shared);
        assert!(decide_artifact(&made(&["-shared", "-o", "x.a"])).is_err());
        let mut reloc = made(&["-r"]);
        reloc.output = Some(OsString::from("part.o"));
        assert_eq!(decide_artifact(&reloc).unwrap(), Artifact::Relocatable);
    }

    #[test]
    fn compile_output_naming() {
        let job = parse(&["-c", "foo.c", "-o", "bar.o"]);
        assert_eq!(compile_outputs(&job).unwrap(), [PathBuf::from("bar.o")]);
        // Directory components are dropped: `sub/b.c` becomes `b.o`.
        let job = parse(&["-c", "a.c", "sub/b.c"]);
        assert_eq!(compile_outputs(&job).unwrap(), [PathBuf::from("a.o"), PathBuf::from("b.o")]);
        // -o with several inputs is a gcc-style error.
        assert!(compile_outputs(&parse(&["-c", "a.c", "b.c", "-o", "x.o"])).is_err());
    }

    #[test]
    fn dep_options() {
        let job = parse(&["-M", "-MF", "foo.d", "-MT", "foo.o", "-MP", "-MG", "a.c"]);
        assert_eq!(job.action, Action::Deps);
        assert!(!job.deps_sys_excluded);
        assert_eq!(job.dep_file.as_deref(), Some(OsStr::new("foo.d")));
        assert_eq!(job.dep_target.as_deref(), Some(OsStr::new("foo.o")));
        assert!(job.dep_phony);
        assert_eq!(job.dep_extra, ["-MG"]);
        let job = parse(&["-MM", "a.c"]);
        assert!(job.deps_sys_excluded);
        let job = parse(&["-Wp,-MD,foo.d", "a.c"]);
        assert_eq!(job.depgen, Some(false));
        assert_eq!(job.dep_file.as_deref(), Some(OsStr::new("foo.d")));
        let job = parse(&["-Wp,-MMD,foo.d", "a.c"]);
        assert_eq!(job.depgen, Some(true));
    }

    #[test]
    fn response_file_splitting() {
        assert_eq!(split_response("-c 'a b.c' -DFOO=\"a b\"\n-o\tout"), ["-c", "a b.c", "-DFOO=a b", "-o", "out"]);
        assert_eq!(split_response("a\\ b c"), ["a b", "c"]);
    }

    #[test]
    fn response_file_expansion() {
        use std::collections::HashMap;
        let mut files = HashMap::new();
        files.insert(PathBuf::from("args.rsp"), "-O2 @more.rsp -o out".to_string());
        files.insert(PathBuf::from("more.rsp"), "a.c".to_string());
        files.insert(PathBuf::from("loop.rsp"), "@loop.rsp".to_string());
        let read = |p: &Path| files.get(p).cloned().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "nope"));
        let out = expand_response_files(args(&["@args.rsp", "b.c"]), &read).unwrap();
        assert_eq!(out, args(&["-O2", "a.c", "-o", "out", "b.c"]));
        assert!(expand_response_files(args(&["@loop.rsp"]), &read).is_err());
        assert!(expand_response_files(args(&["@missing.rsp"]), &read).is_err());
    }

    #[test]
    fn runtime_candidates_prefer_home_over_system() {
        let cands = glcc_runtime_candidates("glccrt.o");
        let last = cands.last().unwrap();
        assert_eq!(last, &PathBuf::from("/usr/lib/glcc/glccrt.o"));
        if home_dir().is_some() {
            assert!(cands.iter().any(|p| p.ends_with(".local/lib/glcc/glccrt.o")));
        }
        let win = glcc_runtime_candidates("glccrt-win64.lib");
        assert_eq!(win.last().unwrap(), &PathBuf::from("/usr/lib/glcc/glccrt-win64.lib"));
        assert!(win.iter().any(|p| p.ends_with(".local/lib/glcc/glccrt-win64.lib")));
    }

    #[test]
    fn runtime_suppressed_by_nostdlib_or_nostartfiles() {
        assert!(wants_glcc_runtime(&parse(&["a.c"])));
        assert!(!wants_glcc_runtime(&parse(&["a.c", "-nostdlib"])));
        assert!(!wants_glcc_runtime(&parse(&["a.c", "-nostartfiles"])));
        // -nodefaultlibs still wants the runtime (startup files stay).
        assert!(wants_glcc_runtime(&parse(&["a.c", "-nodefaultlibs"])));
    }

    #[test]
    fn link_command_places_runtime_first() {
        let job = parse(&["a.c", "-lfoo"]);
        let mut expect: Vec<String> = job.target_args().iter().map(|a| a.to_string_lossy().into_owned()).collect();
        expect.extend(["-o", "a.out", "/r/glccrt.o"].map(String::from));
        if quad_runtime_needs_libm(job.target.arch) {
            expect.push("-lm".into());
        }
        expect.extend(["a.o", "-lfoo"].map(String::from));
        let (_, with) = link_command(&job, &|_| OsString::from("a.o"), OsStr::new("a.out"), false, Some(OsStr::new("/r/glccrt.o")), &[]);
        let s: Vec<String> = with.iter().map(|a| a.to_string_lossy().into_owned()).collect();
        assert_eq!(s, expect);

        let (_, without) = link_command(&job, &|_| OsString::from("a.o"), OsStr::new("a.out"), false, None, &[]);
        let s: Vec<String> = without.iter().map(|a| a.to_string_lossy().into_owned()).collect();
        let mut expect: Vec<String> = job.target_args().iter().map(|a| a.to_string_lossy().into_owned()).collect();
        expect.extend(["-o", "a.out", "a.o", "-lfoo"].map(String::from));
        assert_eq!(s, expect);
    }

    /// AArch64 and RISC-V quad trampolines branch to libm's `sqrtl`
    /// and get a `-lm` right after the runtime; x86-64's combined
    /// runtime is self-contained and its link line carries none.
    #[test]
    fn link_command_adds_libm_for_quad_runtime() {
        for target in ["aarch64", "riscv64"] {
            let job = parse(&["--target", target, "a.c"]);
            let (_, args) = link_command(&job, &|_| OsString::from("a.o"), OsStr::new("a.out"), false, Some(OsStr::new("/r/glccrt.o")), &[]);
            let s: Vec<String> = args.iter().map(|a| a.to_string_lossy().into_owned()).collect();
            let rt = s.iter().position(|a| a == "/r/glccrt.o").expect("runtime in link line");
            assert_eq!(&s[rt + 1], "-lm", "{target} link line: {s:?}");
        }

        let job = parse(&["--target", "x86_64", "a.c"]);
        let (_, args) = link_command(&job, &|_| OsString::from("a.o"), OsStr::new("a.out"), false, Some(OsStr::new("/r/glccrt.o")), &[]);
        let s: Vec<String> = args.iter().map(|a| a.to_string_lossy().into_owned()).collect();
        assert!(!s.contains(&"-lm".to_string()), "x86-64 link line: {s:?}");
    }

    /// The MSVC link path spells gcc-shaped flags in `link.exe` terms:
    /// `-lfoo` names `foo.lib`, `-Ldir` adds a search path, `-e`/`-u`
    /// pairs become `/entry:`/`/include:`, and gcc-only flags drop out.
    #[test]
    fn msvc_link_command_translates_gcc_flags() {
        let job = parse(&["--target=x86_64-windows", "pre.obj", "-lm", "-lfoo", "-Ldir", "-e", "winstart", "-o", "x.exe"]);
        let (prog, args) = msvc_link_command_with(&job, &|_| OsString::from("a.obj"), OsStr::new("x.exe"), &[], OsString::from("link.exe"), &[]);
        assert_eq!(prog, OsString::from("link.exe"));
        let s: Vec<String> = args.iter().map(|a| a.to_string_lossy().into_owned()).collect();
        assert_eq!(
            s,
            [
                "/nologo",
                "/out:x.exe",
                "/subsystem:console",
                "pre.obj",
                "foo.lib",
                "/libpath:dir",
                "/entry:winstart",
                "libcmt.lib",
                "kernel32.lib",
                "user32.lib"
            ]
        );
    }

    /// `-shared` renames the artifact and the subsystem; `-nostdlib`
    /// drops the CRT defaults like it drops the gcc-side runtime.
    #[test]
    fn msvc_link_command_subsystem_and_defaults() {
        let job = parse(&["--target=x86_64-windows", "-shared", "a.c"]);
        let (_, args) = msvc_link_command_with(&job, &|_| OsString::from("a.obj"), OsStr::new("a.dll"), &[], OsString::from("link.exe"), &[]);
        let s: Vec<String> = args.iter().map(|a| a.to_string_lossy().into_owned()).collect();
        assert_eq!(s, ["/nologo", "/dll", "/out:a.dll", "/subsystem:windows", "a.obj", "libcmt.lib", "kernel32.lib", "user32.lib"]);
        let bare = parse(&["--target=x86_64-windows", "-nostdlib", "a.c"]);
        let (_, args) = msvc_link_command_with(&bare, &|_| OsString::from("a.obj"), OsStr::new("a.exe"), &[], OsString::from("link.exe"), &[]);
        assert!(!args.iter().any(|a| a == OsStr::new("libcmt.lib")));
    }

    #[test]
    fn async_flag_defaults_off_and_parses() {
        assert!(!parse(&["a.c"]).async_enabled);
        assert!(parse(&["-fglcc-async", "a.c"]).async_enabled);
    }

    /// A coroutine compiles into a start/resume pair: the original
    /// symbol allocates the frame, `<name>$resume` polls it.
    #[test]
    fn async_functions_split_into_start_and_resume() {
        let src = "\
__attribute__((async)) int add1(int x) { return x + 1; }
int main(void) { add1(1); return 0; }
";
        let bytes = compile_source(src, Flavor::C, Target::LP64, true, true, OptLevel::None, true, LangOpts::default(), ErrorFormat::Human)
            .expect("compiles")
            .0;
        assert!(has_section(&bytes, b"add1$resume"), "resume symbol");
        assert!(has_section(&bytes, b"__glcc_frame_alloc"), "frame hook");
    }

    /// Awaiting copies the child result out and frees the child frame.
    #[test]
    fn awaiting_coroutines_free_child_frames() {
        let src = "\
__attribute__((async)) int add1(int x) { return x + 1; }
__attribute__((async)) int twice(int x) { return __builtin_await(add1(x)) * 2; }
int main(void) { return 0; }
";
        let bytes = compile_source(src, Flavor::C, Target::LP64, true, true, OptLevel::None, true, LangOpts::default(), ErrorFormat::Human)
            .expect("compiles")
            .0;
        assert!(has_section(&bytes, b"twice$resume"), "resume symbol");
        assert!(has_section(&bytes, b"__glcc_frame_free"), "free hook");
    }

    /// The optimizer runs after the split; starts may inline into their
    /// callers, resumes never do.
    #[test]
    fn coroutines_survive_the_optimizer() {
        let src = "\
__attribute__((async)) int add1(int x) { return x + 1; }
__attribute__((async)) int twice(int x) { return __builtin_await(add1(x)) * 2; }
int main(void) { _Future(int) f = twice(20); (void)f; return 0; }
";
        let bytes = compile_source(src, Flavor::C, Target::LP64, true, true, OptLevel::Speed, true, LangOpts::default(), ErrorFormat::Human)
            .expect("compiles")
            .0;
        assert!(has_section(&bytes, b"twice$resume"), "resume symbol");
    }

    /// Without `-fglcc-async` the attribute is ignored, so no split
    /// occurs and the function stays an ordinary one.
    #[test]
    fn async_extension_off_by_default() {
        let src = "\
__attribute__((async)) int add1(int x) { return x + 1; }
int main(void) { return add1(1); }
";
        let bytes = compile_source(src, Flavor::C, Target::LP64, true, true, OptLevel::None, false, LangOpts::default(), ErrorFormat::Human)
            .expect("compiles")
            .0;
        assert!(!has_section(&bytes, b"$resume"), "no split without the flag");
    }
}
