//! `glcc_pp`: native C preprocessor.
//!
//! The driver and the explorer prefer this preprocessor; `--pp=clang`
//! (or a `GLCC_PP` value other than `native`) selects the `clang -E`
//! path instead. When clang is available it is still queried once per
//! process for the macro and system-include seed; snapshots stand in
//! when it is not.
//!
//! # Architecture
//!
//! Token-level pipeline on glcc's lexer, interner, diagnostics, and
//! consteval:
//!
//! - `glcc_frontend::lexer` in PP mode (`PpToken`): raw spellings,
//!   whitespace preserved, preprocessing numbers kept raw for a later
//!   convert/validate pass into proper literals (which never runs inside
//!   the preprocessor itself).
//! - `tree.rs` (`IntoTokenTree`): directive grouping
//!   and `#if` nesting; malformed groups are `PpError`, never panics.
//! - `expand.rs` (`Expander`): rescan expansion with
//!   blue paint, `defined` in conditions, `#if` through glcc consteval.
//! - `macros.rs` / `substitute.rs`: parameter lists,
//!   prescan, `#` stringify, `##` paste with re-lex, `__VA_ARGS__`.
//! - `prepend.rs`: the rescan queue.
//! - This file: options/flags, `clang -dM -v` macro + include seeding
//!   with per-target snapshots, the file layer (`#include` search,
//!   `#pragma once`, linemarkers), output rendering, tests.
//!
//! Deliberate deviations from a by-the-book C preprocessor:
//! `#include` recurses through the
//! file layer here (so markers/`#pragma once`/search behave); clang
//! `__has_*` builtins, `__LINE__`/`__FILE__`, and output padding match
//! `clang -E`; `#if` garbage selects `#else` instead of erroring.

mod expand;
mod macros;
mod prepend;
mod substitute;
mod timefmt;
mod tree;

pub use glcc_toolchain::{MsvcDirs, find_clang, msvc_dirs};

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};

use glcc_frontend::lang::Std;
use glcc_frontend::lexer::{Interner, Lexer, PpToken, PpWs};
use string_interner::symbol::SymbolU32;

use expand::{Expander, MacroDef, Tok};
use tree::ControlLine;

/// Whether the native driver is selected via the environment: on by
/// default, off when `GLCC_PP` names something other than `native`
/// (typically a clang binary or `clang`).
pub fn is_native_selected() -> bool {
    match std::env::var_os("GLCC_PP") {
        None => true,
        Some(v) => v.is_empty() || v == "native",
    }
}

/// Extract `-D`/`-U`/`-I`/`-isystem`/`-iquote`/`-include` from a driver
/// `cpp` flag list (joined or separate spellings), plus `-nostdinc`
/// (`no_std_includes`); every other flag (`-std=`, ...) is ignored.
pub fn cpp_flags_to_opts(cpp: &[OsString], filename: &str) -> PpOptions {
    let mut opts = PpOptions {
        filename: filename.to_string(),
        ..PpOptions::default()
    };
    let mut i = 0;
    while i < cpp.len() {
        let s = cpp[i].to_string_lossy().into_owned();
        if s == "-D" || s == "-U" || s == "-I" || s == "-isystem" || s == "-iquote" || s == "-include" {
            i += 1;
            if let Some(v) = cpp.get(i) {
                let flag = match s.as_str() {
                    "-isystem" | "-iquote" => "-I",
                    other => other,
                };
                push_joined(&mut opts, flag, &v.to_string_lossy());
            }
        } else if let Some(rest) = s.strip_prefix("-include") {
            push_joined(&mut opts, "-include", rest);
        } else if let Some(rest) = s.strip_prefix("-isystem").or_else(|| s.strip_prefix("-iquote")) {
            push_joined(&mut opts, "-I", rest);
        } else if let Some(rest) = s.strip_prefix("-D") {
            push_joined(&mut opts, "-D", rest);
        } else if let Some(rest) = s.strip_prefix("-U") {
            push_joined(&mut opts, "-U", rest);
        } else if let Some(rest) = s.strip_prefix("-I") {
            push_joined(&mut opts, "-I", rest);
        } else if s == "-nostdinc" {
            opts.no_std_includes = true;
        } else if s == "-trigraphs" {
            opts.trigraphs = Some(true);
        } else if s == "-no-trigraphs" {
            opts.trigraphs = Some(false);
        }
        i += 1;
    }
    opts
}

/// What the seed query learns from the system compiler: predefined
/// macros and clang's default system include search.
#[derive(Clone, Default)]
struct Seed {
    predefined: Vec<(String, String)>,
    includes: Vec<PathBuf>,
}

/// Bootstrap the native preprocessor from the system compiler, so it
/// sees exactly what `clang -E` would: predefined macros (`clang -dM`)
/// plus the default system include directories (`clang -v`), where the
/// resource headers (`stddef.h`, ...) and libc live. `triple` selects
/// `--target=`; `std_flag` is `-std=...`/`-ansi` when the driver was
/// given one. Falls back to [`fallback_predefines`] (clang-22 snapshots
/// in `src/builtins/`) and [`fallback_include_paths`] when clang is
/// unavailable. Results are cached per (triple, std) for the process.
fn seed(triple: Option<&str>, std_flag: Option<&str>) -> Seed {
    static CACHE: std::sync::Mutex<Option<(String, Seed)>> = std::sync::Mutex::new(None);
    let key = format!("{}|{}", triple.unwrap_or(""), std_flag.unwrap_or(""));
    if let Ok(cache) = CACHE.lock()
        && let Some((k, v)) = cache.as_ref()
            && *k == key {
                return v.clone();
            }
    let got = clang_seed(triple, std_flag).unwrap_or_else(|| Seed {
        predefined: fallback_predefines(triple),
        includes: fallback_include_paths(triple),
    });
    if let Ok(mut cache) = CACHE.lock() {
        *cache = Some((key, got.clone()));
    }
    got
}

/// Predefined macros the native preprocessor starts from: exactly what
/// `clang -dM -E` reports for `triple`/`std_flag`.
pub fn native_predefines(triple: Option<&str>, std_flag: Option<&str>) -> Vec<(String, String)> {
    seed(triple, std_flag).predefined
}

/// Predefined macros glcc compiles against: the native seed with a GNU
/// version matching the features glcc implements.
///
/// Clang reports GNU 4.2.1 for compatibility, which sits just below the
/// 4.3/4.8 thresholds many libraries use to select `__builtin_bswap{64,16}`
/// paths. Reporting 4.8 unlocks those while staying below 5.4, whose
/// overflow builtins glcc lowers conservatively. Users can still override
/// the version with `-D`.
pub fn predefines(triple: Option<&str>, std_flag: Option<&str>) -> Vec<(String, String)> {
    let mut pre = native_predefines(triple, std_flag);
    for (name, value) in [("__GNUC__", "4"), ("__GNUC_MINOR__", "8"), ("__GNUC_PATCHLEVEL__", "0")] {
        match pre.iter_mut().find(|(existing, _)| existing == name) {
            Some(slot) => slot.1 = value.to_string(),
            None => pre.push((name.to_string(), value.to_string())),
        }
    }
    pre
}

/// Default system include directories for `triple`/`std_flag`: clang's
/// `#include <...>` search list, without the user's `-I` directories
/// (callers prepend those themselves). Callers must skip this under
/// `-nostdinc` ([`PpOptions::no_std_includes`]).
pub fn native_include_paths(triple: Option<&str>, std_flag: Option<&str>) -> Vec<PathBuf> {
    seed(triple, std_flag).includes
}

#[cfg(not(target_arch = "wasm32"))]
fn clang_binary() -> Option<OsString> {
    match std::env::var_os("GLCC_PP") {
        Some(v) if !v.is_empty() && v != "native" => Some(v),
        // No clang on `PATH` (the usual Windows case outside a shell
        // with the LLVM component): fall back to a discovered one, then
        // to the bare name so the spawn failure stays visible.
        _ => Some(find_clang().map(PathBuf::into_os_string).unwrap_or_else(|| OsString::from("clang"))),
    }
}

#[cfg(target_arch = "wasm32")]
fn clang_seed(triple: Option<&str>, std_flag: Option<&str>) -> Option<Seed> {
    // No processes inside a wasm sandbox: `Command::output()` aborts on
    // wasm32, so the embedded snapshots are the only source there.
    let _ = (triple, std_flag);
    None
}

#[cfg(not(target_arch = "wasm32"))]
fn clang_seed(triple: Option<&str>, std_flag: Option<&str>) -> Option<Seed> {
    let bin = clang_binary()?;
    let mut cmd = std::process::Command::new(bin);
    cmd.args(["-E", "-v", "-dM"]);
    if let Some(s) = std_flag {
        cmd.arg(s);
    }
    if let Some(t) = triple {
        cmd.arg(format!("--target={t}"));
    }
    cmd.args(["-x", "c", null_device()]);
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    Some(Seed {
        predefined: parse_d_m_defined(&text),
        includes: parse_search_dirs(&stderr),
    })
}

/// Empty input for the clang seed query: the platform's null device.
fn null_device() -> &'static str {
    if cfg!(windows) { "NUL" } else { "/dev/null" }
}

/// The `#include <...> search starts here:` block of `clang -E -v`:
/// default directories in search order (framework entries are skipped).
/// Entries are absolute paths on every host (`/usr/...` on POSIX,
/// `C:\...` on Windows).
fn parse_search_dirs(stderr: &str) -> Vec<PathBuf> {
    const START: &str = "#include <...> search starts here:";
    const END: &str = "End of search list.";
    let mut out = Vec::new();
    let mut inside = false;
    for line in stderr.lines() {
        let t = line.trim();
        if t == START {
            inside = true;
        } else if t == END {
            inside = false;
        } else if inside && (Path::new(t).is_absolute() || t.starts_with('/')) && !t.ends_with("(framework directory)") {
            out.push(PathBuf::from(t));
        }
    }
    out
}

/// Best-effort system include search when clang is unavailable: the
/// newest clang resource or gcc include directory (for `stddef.h` and
/// friends), then the usual host directories. Layouts that differ
/// (cross sysroots, vendor toolchains) need explicit `-I` flags.
fn fallback_include_paths(triple: Option<&str>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if cfg!(windows) {
        // No fixed POSIX layout: honor the toolchain's `INCLUDE` when a
        // Developer Command Prompt set it, else use the MSVC + Windows
        // SDK layout discovered from the Visual Studio install.
        match std::env::var_os("INCLUDE") {
            Some(value) => out.extend(std::env::split_paths(&value)),
            None => {
                if let Some(msvc) = msvc_dirs() {
                    out.extend(msvc.includes.iter().cloned());
                }
            }
        }
    } else {
        if let Some(dir) = newest_versioned_dir(Path::new("/usr/lib/clang"), "include") {
            out.push(dir);
        } else if let Some(dir) = newest_versioned_dir(Path::new("/usr/lib/gcc"), "include") {
            out.push(dir);
        }
        out.push(PathBuf::from("/usr/local/include"));
        let host = match triple {
            Some(t) => Some(t.to_string()),
            None => match std::env::consts::ARCH {
                "x86_64" => Some("x86_64-linux-gnu".to_string()),
                "aarch64" => Some("aarch64-linux-gnu".to_string()),
                "riscv64" => Some("riscv64-linux-gnu".to_string()),
                _ => None,
            },
        };
        if let Some(t) = host {
            let multiarch = PathBuf::from(format!("/usr/include/{t}"));
            if multiarch.is_dir() {
                out.push(multiarch);
            }
        }
        out.push(PathBuf::from("/usr/include"));
    }
    out.retain(|d| d.is_dir());
    out
}

/// Newest `<parent>/*/<leaf>` or `<parent>/*/*/<leaf>` (clang keeps
/// versions directly, gcc nests a host triple in between), chosen by
/// numeric-leading directory name.
fn newest_versioned_dir(parent: &Path, leaf: &str) -> Option<PathBuf> {
    let mut best: Option<(u64, PathBuf)> = None;
    for entry in std::fs::read_dir(parent).ok()?.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let direct = dir.join(leaf);
        if direct.is_dir() {
            consider_versioned(&mut best, &dir, direct);
        } else if let Ok(sub) = std::fs::read_dir(&dir) {
            for sub in sub.flatten() {
                let sub = sub.path();
                let cand = sub.join(leaf);
                if cand.is_dir() {
                    consider_versioned(&mut best, &sub, cand);
                }
            }
        }
    }
    best.map(|(_, p)| p)
}

fn consider_versioned(best: &mut Option<(u64, PathBuf)>, name_dir: &Path, candidate: PathBuf) {
    let key = name_dir
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.split(['.', '-', '_']).next())
        .and_then(|n| n.parse::<u64>().ok())
        .unwrap_or(0);
    if best.as_ref().is_none_or(|(k, _)| key > *k) {
        *best = Some((key, candidate));
    }
}

/// Parse `clang -dM` output (`#define NAME [replacement]`) into
/// (name, replacement) pairs; `name` may carry `(params)`.
fn parse_d_m_defined(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let Some(body) = line.strip_prefix("#define ") else { continue };
        let end = body.find([' ', '\t', '(']).unwrap_or(body.len());
        let name = &body[..end];
        if !is_ident(name) {
            continue;
        }
        let rest = &body[end..];
        if rest.starts_with('(') {
            let Some(close) = rest.find(')') else { continue };
            let full = &body[..end + close + 1];
            out.push((full.to_string(), rest[close + 1..].trim_start().to_string()));
        } else {
            out.push((name.to_string(), rest.trim_start().to_string()));
        }
    }
    out
}

/// Snapshot fallback when clang is unavailable: clang-22 `-dM` dumps
/// for glcc's three Linux targets (plus the host arch when `triple`
/// is `None`). Approximate by nature (`__VERSION__` is frozen);
/// prefer the live bootstrap above when exactness matters.
pub fn fallback_predefines(triple: Option<&str>) -> Vec<(String, String)> {
    const X86_64: &str = include_str!("builtins/x86_64-linux-gnu.dM");
    const AARCH64: &str = include_str!("builtins/aarch64-linux-gnu.dM");
    const RISCV64: &str = include_str!("builtins/riscv64-linux-gnu.dM");
    let t = triple.unwrap_or(if cfg!(target_arch = "aarch64") {
        "aarch64-linux-gnu"
    } else if cfg!(target_arch = "riscv64") {
        "riscv64-linux-gnu"
    } else {
        "x86_64-linux-gnu"
    });
    let text = if t.contains("aarch64") {
        AARCH64
    } else if t.contains("riscv") {
        RISCV64
    } else {
        X86_64
    };
    parse_d_m_defined(text)
}

/// `__has_feature` / `__has_extension` answers: a table probed from
/// clang in default C mode (see `has_table`), with the C11 six gated by
/// `-std` like the compiler itself (`tls` is always on). Unknown names
/// are false; real misses surface in differentials.
fn has_feature(which: &str, arg: &str, std: Option<&str>) -> bool {
    const GATED: &[&str] = &["c_alignas", "c_alignof", "c_atomic", "c_generic_selections", "c_static_assert", "c_thread_local"];
    if arg == "tls" {
        return true;
    }
    // Probed across -std=c99..c23: always extensions, never features.
    if which == "__has_extension" && (arg == "c_attributes" || arg == "c_fixed_enum") {
        return true;
    }
    let Some(mut hit) = has_table(which, arg) else {
        return false;
    };
    if hit && which == "__has_feature" && GATED.contains(&arg) && !std_is_modern(std) {
        hit = false;
    }
    hit
}

fn std_is_modern(std: Option<&str>) -> bool {
    match std {
        None => true,
        Some(s) if s == "-ansi" || s.contains("89") || s.contains("90") || s.contains("99") => false,
        _ => true,
    }
}

// 135 entries probed from clang-22 default C mode
fn has_table(which: &str, arg: &str) -> Option<bool> {
    Some(match (which, arg) {
        ("__has_feature", "T") => false,
        ("__has_extension", "T") => false,
        ("__has_feature", "X") => false,
        ("__has_extension", "X") => false,
        ("__has_feature", "__alloc_size__") => false,
        ("__has_extension", "__alloc_size__") => false,
        ("__has_feature", "__cxx_aggregate_nsdmi__") => false,
        ("__has_extension", "__cxx_aggregate_nsdmi__") => false,
        ("__has_feature", "__cxx_binary_literals__") => false,
        ("__has_extension", "__cxx_binary_literals__") => true,
        ("__has_feature", "__cxx_decltype_auto__") => false,
        ("__has_extension", "__cxx_decltype_auto__") => false,
        ("__has_feature", "__cxx_generic_lambdas__") => false,
        ("__has_extension", "__cxx_generic_lambdas__") => false,
        ("__has_feature", "__cxx_init_captures__") => false,
        ("__has_extension", "__cxx_init_captures__") => false,
        ("__has_feature", "__cxx_relaxed_constexpr__") => false,
        ("__has_extension", "__cxx_relaxed_constexpr__") => false,
        ("__has_feature", "__cxx_return_type_deduction__") => false,
        ("__has_extension", "__cxx_return_type_deduction__") => false,
        ("__has_feature", "__cxx_static_assert__") => false,
        ("__has_extension", "__cxx_static_assert__") => false,
        ("__has_feature", "__cxx_variable_templates__") => false,
        ("__has_extension", "__cxx_variable_templates__") => false,
        ("__has_feature", "__x") => false,
        ("__has_extension", "__x") => false,
        ("__has_feature", "address_sanitizer") => false,
        ("__has_extension", "address_sanitizer") => false,
        ("__has_feature", "assume_nonnull") => true,
        ("__has_extension", "assume_nonnull") => true,
        ("__has_feature", "attribute_analyzer_noreturn") => true,
        ("__has_extension", "attribute_analyzer_noreturn") => true,
        ("__has_feature", "attribute_availability_in_templates") => true,
        ("__has_extension", "attribute_availability_in_templates") => true,
        ("__has_feature", "attribute_availability_swift") => true,
        ("__has_extension", "attribute_availability_swift") => true,
        ("__has_feature", "attribute_availability_with_strict") => true,
        ("__has_extension", "attribute_availability_with_strict") => true,
        ("__has_feature", "__attribute_deprecated_with_message__") => true,
        ("__has_extension", "__attribute_deprecated_with_message__") => true,
        ("__has_feature", "__attribute_unavailable_with_message__") => true,
        ("__has_extension", "__attribute_unavailable_with_message__") => true,
        ("__has_feature", "blocks") => false,
        ("__has_extension", "blocks") => false,
        ("__has_feature", "builtin_clz") => false,
        ("__has_extension", "builtin_clz") => false,
        ("__has_feature", "c_alignas") => true,
        ("__has_extension", "c_alignas") => true,
        ("__has_feature", "c_alignof") => true,
        ("__has_extension", "c_alignof") => true,
        ("__has_feature", "c_atomic") => true,
        ("__has_extension", "c_atomic") => true,
        ("__has_feature", "c_generic_selections") => true,
        ("__has_extension", "c_generic_selections") => true,
        ("__has_feature", "c_static_assert") => true,
        ("__has_extension", "c_static_assert") => true,
        ("__has_feature", "c_thread_local") => true,
        ("__has_extension", "c_thread_local") => true,
        ("__has_feature", "c_thread_safety_attributes") => true,
        ("__has_extension", "c_thread_safety_attributes") => true,
        ("__has_feature", "cxx_access_control_sfinae") => false,
        ("__has_extension", "cxx_access_control_sfinae") => false,
        ("__has_feature", "cxx_alias_templates") => false,
        ("__has_extension", "cxx_alias_templates") => false,
        ("__has_feature", "cxx_alignas") => false,
        ("__has_extension", "cxx_alignas") => false,
        ("__has_feature", "cxx_alignof") => false,
        ("__has_extension", "cxx_alignof") => false,
        ("__has_feature", "cxx_atomic") => false,
        ("__has_extension", "cxx_atomic") => false,
        ("__has_feature", "cxx_attributes") => false,
        ("__has_extension", "cxx_attributes") => false,
        ("__has_feature", "cxx_auto_type") => false,
        ("__has_extension", "cxx_auto_type") => false,
        ("__has_feature", "cxx_binary_literals") => false,
        ("__has_extension", "cxx_binary_literals") => true,
        ("__has_feature", "cxx_constexpr") => false,
        ("__has_extension", "cxx_constexpr") => false,
        ("__has_feature", "cxx_constexpr_string_builtins") => false,
        ("__has_extension", "cxx_constexpr_string_builtins") => false,
        ("__has_feature", "cxx_decltype") => false,
        ("__has_extension", "cxx_decltype") => false,
        ("__has_feature", "cxx_decltype_auto") => false,
        ("__has_extension", "cxx_decltype_auto") => false,
        ("__has_feature", "cxx_decltype_incomplete_return_types") => false,
        ("__has_extension", "cxx_decltype_incomplete_return_types") => false,
        ("__has_feature", "cxx_default_function_template_args") => false,
        ("__has_extension", "cxx_default_function_template_args") => false,
        ("__has_feature", "cxx_defaulted_functions") => false,
        ("__has_extension", "cxx_defaulted_functions") => false,
        ("__has_feature", "cxx_delegating_constructors") => false,
        ("__has_extension", "cxx_delegating_constructors") => false,
        ("__has_feature", "cxx_deleted_functions") => false,
        ("__has_extension", "cxx_deleted_functions") => false,
        ("__has_feature", "cxx_exceptions") => false,
        ("__has_extension", "cxx_exceptions") => false,
        ("__has_feature", "cxx_explicit_conversions") => false,
        ("__has_extension", "cxx_explicit_conversions") => false,
        ("__has_feature", "cxx_generalized_initializers") => false,
        ("__has_extension", "cxx_generalized_initializers") => false,
        ("__has_feature", "cxx_generic_lambda") => false,
        ("__has_extension", "cxx_generic_lambda") => false,
        ("__has_feature", "cxx_if_constexpr") => false,
        ("__has_extension", "cxx_if_constexpr") => false,
        ("__has_feature", "cxx_implicit_moves") => false,
        ("__has_extension", "cxx_implicit_moves") => false,
        ("__has_feature", "cxx_inheriting_constructors") => false,
        ("__has_extension", "cxx_inheriting_constructors") => false,
        ("__has_feature", "cxx_init_capture") => false,
        ("__has_extension", "cxx_init_capture") => false,
        ("__has_feature", "cxx_inline_namespaces") => false,
        ("__has_extension", "cxx_inline_namespaces") => false,
        ("__has_feature", "cxx_lambdas") => false,
        ("__has_extension", "cxx_lambdas") => false,
        ("__has_feature", "cxx_local_type_template_args") => false,
        ("__has_extension", "cxx_local_type_template_args") => false,
        ("__has_feature", "cxx_noexcept") => false,
        ("__has_extension", "cxx_noexcept") => false,
        ("__has_feature", "cxx_nonstatic_member_init") => false,
        ("__has_extension", "cxx_nonstatic_member_init") => false,
        ("__has_feature", "cxx_nullptr") => false,
        ("__has_extension", "cxx_nullptr") => false,
        ("__has_feature", "cxx_override_control") => false,
        ("__has_extension", "cxx_override_control") => false,
        ("__has_feature", "cxx_range_for") => false,
        ("__has_extension", "cxx_range_for") => false,
        ("__has_feature", "cxx_raw_string_literals") => false,
        ("__has_extension", "cxx_raw_string_literals") => false,
        ("__has_feature", "cxx_reference_qualified_functions") => false,
        ("__has_extension", "cxx_reference_qualified_functions") => false,
        ("__has_feature", "cxx_relaxed_constexpr") => false,
        ("__has_extension", "cxx_relaxed_constexpr") => false,
        ("__has_feature", "cxx_return_type_deduction") => false,
        ("__has_extension", "cxx_return_type_deduction") => false,
        ("__has_feature", "cxx_rtti") => false,
        ("__has_extension", "cxx_rtti") => false,
        ("__has_feature", "cxx_runtime_array") => false,
        ("__has_extension", "cxx_runtime_array") => false,
        ("__has_feature", "cxx_rvalue_references") => false,
        ("__has_extension", "cxx_rvalue_references") => false,
        ("__has_feature", "cxx_static_assert") => false,
        ("__has_extension", "cxx_static_assert") => false,
        ("__has_feature", "cxx_string_literal_templates") => false,
        ("__has_extension", "cxx_string_literal_templates") => false,
        ("__has_feature", "cxx_strong_enums") => false,
        ("__has_extension", "cxx_strong_enums") => false,
        ("__has_feature", "cxx_thread_local") => false,
        ("__has_extension", "cxx_thread_local") => false,
        ("__has_feature", "cxx_trailing_return") => false,
        ("__has_extension", "cxx_trailing_return") => false,
        ("__has_feature", "cxx_unicode_literals") => false,
        ("__has_extension", "cxx_unicode_literals") => false,
        ("__has_feature", "cxx_unrestricted_unions") => false,
        ("__has_extension", "cxx_unrestricted_unions") => false,
        ("__has_feature", "cxx_user_literals") => false,
        ("__has_extension", "cxx_user_literals") => false,
        ("__has_feature", "cxx_variable_templates") => false,
        ("__has_extension", "cxx_variable_templates") => false,
        ("__has_feature", "cxx_variadic_templates") => false,
        ("__has_extension", "cxx_variadic_templates") => false,
        ("__has_feature", "enumerator_attributes") => true,
        ("__has_extension", "enumerator_attributes") => true,
        ("__has_feature", "extension") => false,
        ("__has_extension", "extension") => false,
        ("__has_feature", "f") => false,
        ("__has_extension", "f") => false,
        ("__has_feature", "feature") => false,
        ("__has_extension", "feature") => false,
        ("__has_feature", "gnu_asm") => false,
        ("__has_extension", "gnu_asm") => true,
        ("__has_feature", "has_nothrow_assign") => false,
        ("__has_extension", "has_nothrow_assign") => false,
        ("__has_feature", "has_nothrow_constructor") => false,
        ("__has_extension", "has_nothrow_constructor") => false,
        ("__has_feature", "has_nothrow_copy") => false,
        ("__has_extension", "has_nothrow_copy") => false,
        ("__has_feature", "has_trivial_assign") => false,
        ("__has_extension", "has_trivial_assign") => false,
        ("__has_feature", "has_trivial_constructor") => false,
        ("__has_extension", "has_trivial_constructor") => false,
        ("__has_feature", "has_trivial_copy") => false,
        ("__has_extension", "has_trivial_copy") => false,
        ("__has_feature", "has_trivial_destructor") => false,
        ("__has_extension", "has_trivial_destructor") => false,
        ("__has_feature", "has_virtual_destructor") => false,
        ("__has_extension", "has_virtual_destructor") => false,
        ("__has_feature", "hwaddress_sanitizer") => false,
        ("__has_extension", "hwaddress_sanitizer") => false,
        ("__has_feature", "is_abstract") => false,
        ("__has_extension", "is_abstract") => false,
        ("__has_feature", "is_assignable") => false,
        ("__has_extension", "is_assignable") => false,
        ("__has_feature", "is_base_of") => false,
        ("__has_extension", "is_base_of") => false,
        ("__has_feature", "is_class") => false,
        ("__has_extension", "is_class") => false,
        ("__has_feature", "is_constructible") => false,
        ("__has_extension", "is_constructible") => false,
        ("__has_feature", "is_convertible") => false,
        ("__has_extension", "is_convertible") => false,
        ("__has_feature", "is_convertible_to") => false,
        ("__has_extension", "is_convertible_to") => false,
        ("__has_feature", "is_empty") => false,
        ("__has_extension", "is_empty") => false,
        ("__has_feature", "is_enum") => false,
        ("__has_extension", "is_enum") => false,
        ("__has_feature", "is_final") => false,
        ("__has_extension", "is_final") => false,
        ("__has_feature", "is_literal") => false,
        ("__has_extension", "is_literal") => false,
        ("__has_feature", "is_nothrow_constructible") => false,
        ("__has_extension", "is_nothrow_constructible") => false,
        ("__has_feature", "is_pod") => false,
        ("__has_extension", "is_pod") => false,
        ("__has_feature", "is_polymorphic") => false,
        ("__has_extension", "is_polymorphic") => false,
        ("__has_feature", "is_standard_layout") => false,
        ("__has_extension", "is_standard_layout") => false,
        ("__has_feature", "is_trivial") => false,
        ("__has_extension", "is_trivial") => false,
        ("__has_feature", "is_trivially_assignable") => false,
        ("__has_extension", "is_trivially_assignable") => false,
        ("__has_feature", "is_trivially_constructible") => false,
        ("__has_extension", "is_trivially_constructible") => false,
        ("__has_feature", "is_trivially_copyable") => false,
        ("__has_extension", "is_trivially_copyable") => false,
        ("__has_feature", "is_union") => false,
        ("__has_extension", "is_union") => false,
        ("__has_feature", "memory_sanitizer") => false,
        ("__has_extension", "memory_sanitizer") => false,
        ("__has_feature", "modules") => false,
        ("__has_extension", "modules") => false,
        ("__has_feature", "name") => false,
        ("__has_extension", "name") => false,
        ("__has_feature", "nullability") => true,
        ("__has_extension", "nullability") => true,
        ("__has_feature", "objc_arc") => false,
        ("__has_extension", "objc_arc") => false,
        ("__has_feature", "objc_arc_weak") => false,
        ("__has_extension", "objc_arc_weak") => false,
        ("__has_feature", "objc_bridge_id") => true,
        ("__has_extension", "objc_bridge_id") => true,
        ("__has_feature", "objc_bridge_id_on_typedefs") => true,
        ("__has_extension", "objc_bridge_id_on_typedefs") => true,
        ("__has_feature", "objc_fixed_enum") => false,
        ("__has_extension", "objc_fixed_enum") => false,
        ("__has_feature", "ptrauth_calls") => false,
        ("__has_extension", "ptrauth_calls") => false,
        ("__has_feature", "ptrauth_function_pointer_type_discrimination") => false,
        ("__has_extension", "ptrauth_function_pointer_type_discrimination") => false,
        ("__has_feature", "ptrauth_intrinsics") => false,
        ("__has_extension", "ptrauth_intrinsics") => false,
        ("__has_feature", "ptrauth_objc_interface_sel") => false,
        ("__has_extension", "ptrauth_objc_interface_sel") => false,
        ("__has_feature", "ptrauth_objc_method_list_pointer") => false,
        ("__has_extension", "ptrauth_objc_method_list_pointer") => false,
        ("__has_feature", "ptrauth_restricted_intptr_qualifier") => false,
        ("__has_extension", "ptrauth_restricted_intptr_qualifier") => false,
        ("__has_feature", "realtime_sanitizer") => false,
        ("__has_extension", "realtime_sanitizer") => false,
        ("__has_feature", "thread_sanitizer") => false,
        ("__has_extension", "thread_sanitizer") => false,
        ("__has_feature", "tls") => true,
        ("__has_extension", "tls") => true,
        ("__has_feature", "undefined_behavior_sanitizer") => false,
        ("__has_extension", "undefined_behavior_sanitizer") => false,
        ("__has_feature", "underlying_type") => false,
        ("__has_extension", "underlying_type") => false,
        ("__has_feature", "xxx") => false,
        ("__has_extension", "xxx") => false,
        _ => return None,
    })
}

/// Options for [`preprocess`]. Built from `-D`/`-U`/`-I` flags.
#[derive(Debug, Clone, Default)]
pub struct PpOptions {
    /// Compiler builtins (`clang -dM` seed or [`fallback_predefines`]),
    /// applied before `-D`/`-U` like GCC's own predefined macros.
    /// Names may carry `(params)` for builtin function-likes.
    pub predefined: Vec<(String, String)>,
    /// `-DNAME` or `-DNAME=replacement` (object- or function-like).
    pub defines: Vec<(String, Option<String>)>,
    /// `-U NAME`.
    pub undefs: Vec<String>,
    /// `-I`/`-isystem`/`-iquote` paths, in order.
    pub include_paths: Vec<PathBuf>,
    /// `-nostdinc`: callers do not append [`native_include_paths`].
    pub no_std_includes: bool,
    /// Raw `-std=`/`-ansi` flag when the driver was given one (selects
    /// `__STDC_VERSION__` in the seed and gates `__has_feature` answers).
    pub std: Option<String>,
    /// `-include` files, processed before the main source.
    pub preincludes: Vec<String>,
    /// Emit `# <n> "file"` linemarkers (needed whenever the output
    /// feeds the real compiler, so diagnostics map back).
    pub linemarkers: bool,
    /// Target for `#if` constant evaluation (char signedness, type
    /// sizes); `None` means native.
    pub target: Option<glcc_frontend::target::Target>,
    /// Clang's resource include directory when it appears in
    /// `include_paths` (see [`clang_resource_dir`]). Windows targets
    /// skip it for `*intrin.h`: clang's copies expand to
    /// `__builtin_ia32_*` operations glcc does not implement, while the
    /// MSVC SDK provides the same header names for its target.
    pub clang_resource_dir: Option<PathBuf>,
    /// Input name used in diagnostics.
    pub filename: String,
    /// `-trigraphs`/`-no-trigraphs`; `None` follows the standard level
    /// (enabled before C23, disabled from C23 on).
    pub trigraphs: Option<bool>,
}

/// The clang resource include directory of a seeded search list: clang
/// searches it first, and it alone ships clang's builtin headers such as
/// `__stddef_max_align_t.h`. `None` when the first entry is not a clang
/// resource directory (e.g. clang was unavailable and a GCC sysroot
/// supplied the fallback list).
pub fn clang_resource_dir(paths: &[PathBuf]) -> Option<PathBuf> {
    paths.first().filter(|p| p.join("__stddef_max_align_t.h").is_file()).cloned()
}

/// Preprocessor failure. The `String` payload already carries the
/// `native pp: ` prefix; there are no panics on this path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PpError(pub String);

impl fmt::Display for PpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PpError {}

/// A `#warning` diagnostic collected during preprocessing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PpWarning {
    /// `file:line` where the directive appeared.
    pub loc: String,
    /// The directive's message text, without the `#warning` keyword.
    pub message: String,
}

impl fmt::Display for PpWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.message.is_empty() {
            write!(f, "{}: warning: #warning", self.loc)
        } else {
            write!(f, "{}: warning: #warning {}", self.loc, self.message)
        }
    }
}

pub(crate) fn err(msg: impl Into<String>) -> PpError {
    PpError(format!("native pp: {}", msg.into()))
}

/// Which file of a header search provided the current file, for
/// `#include_next` (which resumes past it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FoundKind {
    /// Main source or `-include`: search everything.
    Start,
    /// Beside the includer (quoted lookup): skip that directory.
    QuoterDir,
    /// At `include_paths[i]`: resume after it.
    Index(usize),
}

/// Driver file layer: include search, marker output, `#pragma once`
/// set. Macro state lives in the [`Expander`](expand::Expander).
pub(crate) struct Ctx {
    pub opts: PpOptions,
    pub out: String,
    pub once: HashSet<PathBuf>,
    pub depth: u32,
    pub warnings: Vec<PpWarning>,
}

impl Ctx {
    /// Search for a header: quoted tries the includer's directory first,
    /// then `-I` paths; bracketed tries `-I` paths only. With `after`,
    /// resume past the current file (`#include_next`).
    pub fn resolve_header(&self, name: &str, quoted: bool, after: Option<FoundKind>, dir: &Path, loc: &str) -> Result<(PathBuf, FoundKind), PpError> {
        let start = match after {
            None | Some(FoundKind::Start) | Some(FoundKind::QuoterDir) => None,
            Some(FoundKind::Index(i)) => Some(i + 1),
        };
        if quoted && after.is_none() {
            let cand = dir.join(name);
            if cand.is_file() {
                return Ok((cand, FoundKind::QuoterDir));
            }
        }
        for (i, p) in self.opts.include_paths.iter().enumerate() {
            if let Some(s) = start
                && i < s {
                    continue;
                }
            if matches!(after, Some(FoundKind::QuoterDir)) && p == dir {
                continue;
            }
            let cand = p.join(name);
            if cand.is_file() {
                if self.skips_clang_intrinsic(p, name) {
                    continue;
                }
                return Ok((cand, FoundKind::Index(i)));
            }
        }
        Err(err(format!("{loc}: #include '{name}' not found")))
    }

    /// Windows/MSVC programs include `<intrin.h>` for the SDK's own
    /// intrinsic declarations. Clang's resource copy is compiler
    /// builtins for clang, so let the later MSVC/system directories
    /// provide `intrin.h` and the `*intrin.h` files it includes.
    fn skips_clang_intrinsic(&self, dir: &Path, name: &str) -> bool {
        name.ends_with("intrin.h") && self.opts.clang_resource_dir.as_deref() == Some(dir) && matches!(self.opts.target, Some(t) if t.os == glcc_frontend::target::Os::Windows)
    }

    pub fn record_once(&mut self, path: &Path) {
        let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.once.insert(canon);
    }
}

/// One `# <n> "file"` linemarker line at column 0, with an optional
/// clang-style flag suffix (`1` enter, `2` return).
pub(crate) fn marker(n: i64, display: &str) -> String {
    format!("# {n} \"{}\"\n", display.replace('\\', "\\\\").replace('"', "\\\""))
}

pub(crate) fn marker_flag(n: i64, display: &str, flag: u8) -> String {
    format!("# {n} \"{}\" {flag}\n", display.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Clang's `-E` output padding, reproduced so expanded text matches
/// byte-for-byte: at a macro-expansion boundary, emit one space iff
/// the adjoining characters are `(digit-or-dot, +, -, .)` or
/// `(+, -, +, -)`. This keeps `W+1` as `11 +1` (never merging `++`,
/// `--`, exponents, or floats) while `(W)+1` stays `(11)+1`.

fn push_joined(opts: &mut PpOptions, flag: &str, val: &str) {
    match flag {
        "-D" => {
            let (name, rep) = match val.split_once('=') {
                Some((n, r)) => (n.to_string(), Some(r.to_string())),
                None => (val.to_string(), None),
            };
            opts.defines.push((name, rep));
        }
        "-U" => opts.undefs.push(val.to_string()),
        "-I" => opts.include_paths.push(PathBuf::from(val)),
        "-include" => opts.preincludes.push(val.to_string()),
        _ => {}
    }
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c == '_' || c.is_ascii_alphabetic() => chars.all(|c| c == '_' || c.is_ascii_alphanumeric()),
        _ => false,
    }
}

/// Whether translation phase 1 should replace trigraphs. C23 removed
/// them; `-trigraphs`/`-no-trigraphs` override the dialect default.
fn trigraphs_enabled(opts: &PpOptions) -> bool {
    if let Some(enabled) = opts.trigraphs {
        return enabled;
    }
    let Some(flag) = opts.std.as_deref() else {
        return true;
    };
    let text = if flag == "-ansi" { "c89" } else { flag.strip_prefix("-std=").unwrap_or(flag) };
    Std::parse(text).is_none_or(|std| std.level() < Std::C23.level())
}

/// Replace the nine C trigraph sequences, leaving text with no `??`
/// untouched. C23 modes keep the sequences literal.
fn translate_trigraphs(source: &str) -> Cow<'_, str> {
    if !source.contains("??") {
        return Cow::Borrowed(source);
    }
    let bytes = source.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    let mut changed = false;
    while i < bytes.len() {
        if i + 2 < bytes.len() && bytes[i] == b'?' && bytes[i + 1] == b'?' {
            let replacement = match bytes[i + 2] {
                b'=' => Some(b'#'),
                b'/' => Some(b'\\'),
                b'\'' => Some(b'^'),
                b'(' => Some(b'['),
                b')' => Some(b']'),
                b'!' => Some(b'|'),
                b'<' => Some(b'{'),
                b'>' => Some(b'}'),
                b'-' => Some(b'~'),
                _ => None,
            };
            if let Some(replacement) = replacement {
                out.push(replacement);
                i += 3;
                changed = true;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    if changed {
        Cow::Owned(String::from_utf8(out).expect("trigraph replacement preserves UTF-8"))
    } else {
        Cow::Borrowed(source)
    }
}

fn translate_source<'a>(source: &'a str, opts: &PpOptions) -> Cow<'a, str> {
    if trigraphs_enabled(opts) { translate_trigraphs(source) } else { Cow::Borrowed(source) }
}

/// Preprocess `source` and return the expanded text.
///
/// Implements object- and function-like macros, conditionals evaluated
/// through `glcc_frontend`, `#include` with `# <n> "file"` linemarkers
/// (when [`PpOptions::linemarkers`]), `#pragma once`, and `#line`.
/// Anything else outside scope is an `Err` naming the gap, never a
/// silent pass-through.
pub fn preprocess(source: &str, filename: &str, opts: &PpOptions) -> Result<String, PpError> {
    preprocess_with_warnings(source, filename, opts).map(|(text, _)| text)
}

/// [`preprocess`], also returning the `#warning` diagnostics it emitted
/// (including from included files).
pub fn preprocess_with_warnings(source: &str, filename: &str, opts: &PpOptions) -> Result<(String, Vec<PpWarning>), PpError> {
    let source = translate_source(source, opts);
    let mut interner = Interner::new();
    let mut ctx = Ctx {
        opts: opts.clone(),
        out: String::new(),
        once: HashSet::new(),
        depth: 0,
        warnings: Vec::new(),
    };
    let mut macros: HashMap<SymbolU32, MacroDef> = HashMap::new();
    seed_macros(&opts.predefined, filename, &mut interner, &mut macros)?;
    seed_macros(
        &opts
            .defines
            .iter()
            .map(|(n, r)| (n.clone(), r.clone().unwrap_or_else(|| "1".to_string())))
            .collect::<Vec<_>>(),
        filename,
        &mut interner,
        &mut macros,
    )?;
    for u in &opts.undefs {
        let sym = interner.get_or_intern(u.clone());
        macros.remove(&sym);
    }

    let main_path = PathBuf::from(filename);
    let main_dir = main_path.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
    let main_display = interner.get_or_intern(filename);
    // `-include` files come first, as if included at line 1.
    for pre in opts.preincludes.clone() {
        let (target, found) = ctx
            .resolve_header(&pre, true, None, &main_dir, filename)
            .map_err(|_| err(format!("{filename}: -include '{pre}' not found")))?;
        let canon = std::fs::canonicalize(&target).unwrap_or_else(|_| target.clone());
        if ctx.once.contains(&canon) {
            continue;
        }
        let text = std::fs::read_to_string(&target).map_err(|_| err(format!("{filename}: cannot read '{}'", target.display())))?;
        let text = translate_source(&text, opts);
        // An `-include` file reports include level 1 (as if included).
        ctx.depth = 1;
        let r = run_source(
            text.as_ref(),
            &target,
            interner.get_or_intern(target.display().to_string()),
            main_display,
            found,
            true,
            &mut interner,
            &mut macros,
            &mut ctx,
            opts,
        );
        ctx.depth = 0;
        r?;
    }
    if opts.linemarkers {
        ctx.out.push_str(&marker(1, filename));
    }
    run_source(
        source.as_ref(),
        &main_path,
        main_display,
        main_display,
        FoundKind::Start,
        false,
        &mut interner,
        &mut macros,
        &mut ctx,
        opts,
    )?;
    if opts.linemarkers {
        // clang terminates the final line unconditionally (a marker-only
        // tail therefore grows a blank line).
        ctx.out.push('\n');
    } else if !ctx.out.ends_with('\n') {
        ctx.out.push('\n');
    }
    Ok((ctx.out, ctx.warnings))
}

/// Seed the macro table from `(name, replacement)` pairs (predefined
/// builtins and `-D`): each is lexed and parsed exactly like a
/// `#define` tail, so function-like builtins (`__INTMAX_C(c)`) work.
fn seed_macros(pairs: &[(String, String)], filename: &str, interner: &mut Interner, macros: &mut HashMap<SymbolU32, MacroDef>) -> Result<(), PpError> {
    use tree::parse_control_line;
    for (name, rep) in pairs {
        let text = if rep.is_empty() { format!("define {name}") } else { format!("define {name} {rep}") };
        let mut lexer = Lexer::new(&text, interner);
        let toks = lexer.lex_pp_all();
        let file = interner.get_or_intern(filename);
        let rest: Vec<Tok> = toks.into_iter().map(|t| Tok::from_pp(&t, file, false, false)).collect();
        let control = parse_control_line(&rest, filename, 1, interner)?;
        match control {
            ControlLine::DefineObject { macro_name, replacement } => {
                let sym = interner.get_or_intern(macro_name);
                macros.insert(sym, MacroDef::Object(replacement));
            }
            ControlLine::DefineFunction {
                macro_name,
                parameters,
                replacement,
            } => {
                let loc = format!("{filename}: -D{name}");
                let parsed = macros::parse_parameters(&parameters, &loc, interner)?;
                let sym = interner.get_or_intern(macro_name);
                macros.insert(
                    sym,
                    MacroDef::Func {
                        params: parsed.parameters,
                        variadic: parsed.variadic,
                        named_variadic: parsed.named_variadic,
                        body: replacement,
                    },
                );
            }
            _ => return Err(err(format!("{filename}: invalid macro definition '{name}'"))),
        }
    }
    Ok(())
}

/// Lex, group, and expand one file's text, then render. `entered`
/// emits the `# 1 "file" 1` enter marker (for `-include` files; the
/// main source gets its marker from [`preprocess`], includes from the
/// expander's own recursion).
#[allow(clippy::too_many_arguments)]
fn run_source(
    source: &str,
    path: &Path,
    display: SymbolU32,
    base_display: SymbolU32,
    found: FoundKind,
    entered: bool,
    interner: &mut Interner,
    macros: &mut HashMap<SymbolU32, MacroDef>,
    ctx: &mut Ctx,
    opts: &PpOptions,
) -> Result<(), PpError> {
    let mut lexer = Lexer::new(source, interner);
    let toks: Vec<Tok> = lexer.lex_pp_all().into_iter().map(|t| Tok::from_pp(&t, display, false, false)).collect();
    if std::env::var_os("GLCC_PP_DEBUG").is_some() {
        let nl = toks.iter().filter(|t| matches!(t.tok, PpToken::Ws(PpWs::Newline))).count();
        let mx = toks.iter().map(|t| t.line).max().unwrap_or(0);
        eprintln!("DBG lex {}: {} toks, {} newlines, max line {}", path.display(), toks.len(), nl, mx);
    }
    let disp_str = interner.resolve(display).unwrap_or("").to_string();
    let trees = tree::collect_trees(toks, &disp_str, interner)?;
    if entered && opts.linemarkers {
        ctx.out.push_str(&marker_flag(1, &disp_str, 1));
    }
    let mut expander = Expander {
        interner,
        trees: crate::prepend::PrependingPeekableIterator::new(trees.into_iter()),
        output: Vec::new(),
        macros: std::rc::Rc::new(std::mem::take(macros)),
        expanding: HashSet::new(),
        respect_defined: false,
        file_display: display,
        file_path: path.to_path_buf(),
        file_found: found,
        base_display,
        counter: 0,
        undef_builtins: HashSet::new(),
        date_time: None,
        timestamps: HashMap::new(),
        pushed_macros: HashMap::new(),
        line_base: 0,
        display_override: None,
        target: opts.target,
        std_flag: opts.std.clone(),
        depth: 0,
        files: ctx,
        mark_n: 1,
        mark_emit: 0,
        fresh: true,
        prev: None,
        emit_inline: true,
        frames: Vec::new(),
        r_cur_line: 1,
        r_tokens_on_line: false,
        r_directive_on_line: false,
        r_pending_sol: false,
        r_prev: None,
        r_prevprev: None,
        carry: None,
    };
    expander.expand()?;
    *macros = std::rc::Rc::try_unwrap(std::mem::take(&mut expander.macros)).unwrap_or_else(|rc| (*rc).clone());
    drop(expander);
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use super::*;

    fn opts() -> PpOptions {
        PpOptions::default()
    }

    fn pp(src: &str) -> Result<String, PpError> {
        preprocess(src, "t.c", &opts())
    }

    #[test]
    fn warning_directive_is_reported_only_when_active() {
        let (text, warnings) = preprocess_with_warnings("#if 0\n#warning no\n#endif\n#warning \"yes\"\n", "t.c", &opts()).unwrap();
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(warnings[0].loc, "t.c:4");
        assert_eq!(warnings[0].message, "\"yes\"");
        assert!(warnings[0].to_string().contains("warning: #warning \"yes\""));
        assert!(!text.contains("warning"));
    }

    #[test]
    fn pp_lexer_counts_every_newline() {
        use glcc_frontend::lexer::{Lexer, PpToken, PpWs};
        let src = "/* a\nb\nc */\n#define X 1\n// trail\nint v = X;\n";
        let mut interner = Interner::new();
        let mut lexer = Lexer::new(src, &mut interner);
        let toks = lexer.lex_pp_all();
        let nls: Vec<u64> = toks.iter().filter(|t| matches!(t.tok, PpToken::Ws(PpWs::Newline))).map(|t| t.line).collect();
        // 7 physical lines: newlines after lines 1,2 (comment), 3 (comment
        // close), 4 (define), 5 (comment), 6 (code).
        assert_eq!(nls.len(), 6, "newlines: {nls:?}");
        let last_code = toks.iter().rev().find(|t| matches!(t.tok, PpToken::Ident(_))).unwrap();
        assert_eq!(last_code.line, 6);
    }

    #[test]
    fn pp_lexer_amalgamation_markers() {
        use glcc_frontend::lexer::{Lexer, PpToken, PpWs};
        // Amalgamation-style: license comment, defines, an unclosed
        // marker comment closed later, conditionals, trailing content.
        let src = "/* license line 1\n** line 2\n*/\n#ifndef GUARD\n# define X static\n#endif\n/************** Begin **************\n/*\n** text\n*/\nint tail = 1;\n";
        let mut interner = Interner::new();
        let mut lexer = Lexer::new(src, &mut interner);
        let toks = lexer.lex_pp_all();
        let max_line = toks.iter().map(|t| t.line).max().unwrap_or(0);
        assert_eq!(max_line, 11);
        let tail = toks.iter().rev().find(|t| matches!(t.tok, PpToken::Ident(_))).unwrap();
        assert_eq!(tail.line, 11);
        let nls = toks.iter().filter(|t| matches!(t.tok, PpToken::Ws(PpWs::Newline))).count();
        assert_eq!(nls, 11);
    }

    /// C11 6.4.8 pp-number: a sign continues the number only when the
    /// character just consumed was `e`/`E`/`p`/`P`. Hex digits are not
    /// exponent markers: `0xE1+` splits, while `0xE+` joins (the sign,
    /// then the identifier characters, become part of the number).
    #[test]
    fn pp_lexer_sign_needs_trailing_exponent_marker() {
        fn numbers(src: &str) -> Vec<String> {
            use glcc_frontend::lexer::{Lexer, PpToken};
            let mut interner = Interner::new();
            let mut lexer = Lexer::new(src, &mut interner);
            lexer
                .lex_pp_all()
                .iter()
                .filter_map(|t| match t.tok {
                    PpToken::PpNumber(sym) => Some(interner.resolve(sym).unwrap_or("").to_string()),
                    _ => None,
                })
                .collect()
        }
        assert_eq!(numbers("0xE1+rol"), ["0xE1"]);
        assert_eq!(numbers("0xABCDEF+1"), ["0xABCDEF", "1"]);
        assert_eq!(numbers("1E5-2"), ["1E5", "2"]);
        assert_eq!(numbers("1e+5"), ["1e+5"]);
        assert_eq!(numbers("0xE+rol"), ["0xE+rol"]);
    }

    /// Regression: a sticky sign flag used to swallow the macro
    /// invocation after any number containing `e`/`E`/`p`/`P`, so
    /// SQLite's `0x6ED9EBA1+rol(v,5)` never expanded `rol`.
    #[test]
    fn macro_expands_after_number_containing_exponent_marker() {
        let src = "#define rol(x,k) (((x)<<(k))|((x)>>(32-(k))))\nint a = 0xE1+rol(1,2);\nint b = 1E5+rol(3,4);\n";
        assert_eq!(
            pp(src).unwrap(),
            "\nint a = 0xE1+(((1)<<(2))|((1)>>(32-(2))));\nint b = 1E5+(((3)<<(4))|((3)>>(32-(4))));\n"
        );
    }

    #[test]
    fn passthrough_without_directives() {
        assert_eq!(pp("int x = 1;\n").unwrap(), "int x = 1;\n");
    }

    #[test]
    fn object_define_and_undef() {
        let src = "#define FOO 42\nint x = FOO;\n#undef FOO\nint y = FOO;\n";
        assert_eq!(pp(src).unwrap(), "\nint x = 42;\n\nint y = FOO;\n");
    }

    #[test]
    fn define_from_opts_and_chained() {
        let o = PpOptions {
            defines: vec![("A".to_string(), Some("B".to_string())), ("B".to_string(), Some("1".to_string()))],
            ..PpOptions::default()
        };
        assert_eq!(preprocess("int x = A;\n", "t.c", &o).unwrap(), "int x = 1;\n");
    }

    #[test]
    fn self_recursive_define_terminates() {
        let src = "#define X X\nint x = X;\n";
        assert_eq!(pp(src).unwrap(), "\nint x = X;\n");
    }

    #[test]
    fn ifdef_branches() {
        let src = "#define A\n#ifdef A\nint a = 1;\n#else\nint a = 0;\n#endif\n#ifndef B\nint b = 1;\n#endif\n";
        assert_eq!(pp(src).unwrap(), "\n\nint a = 1;\n\n\nint b = 1;\n\n");
    }

    #[test]
    fn if_unsigned_comparison_uses_c_conversions() {
        // Legacy i128 arithmetic would say -1 > 1U is false; C says true.
        let src = "#if -1 > 1U\nint big = 1;\n#else\nint big = 0;\n#endif\n";
        assert_eq!(pp(src).unwrap(), "\nint big = 1;\n\n");
    }

    #[test]
    fn if_sizeof_folds_through_frontend() {
        let src = "#if sizeof(int) >= 2 && sizeof(char) == 1\nint ok = 1;\n#endif\n";
        assert_eq!(pp(src).unwrap(), "\nint ok = 1;\n\n");
    }

    #[test]
    fn if_garbage_selects_else_branch() {
        let src = "#if +++\nint a = 1;\n#else\nint a = 0;\n#endif\n";
        assert_eq!(pp(src).unwrap(), "\nint a = 0;\n\n");
    }

    #[test]
    fn if_undefined_ident_is_zero() {
        let src = "#if NOPE_DEFINED_ANYWHERE\nint a = 1;\n#else\nint a = 0;\n#endif\n";
        assert_eq!(pp(src).unwrap(), "\nint a = 0;\n\n");
    }

    #[test]
    fn if_expression_with_defined() {
        let src = "#define V 3\n#if defined(V) && V > 2\nint ok = 1;\n#else\nint ok = 0;\n#endif\n";
        assert_eq!(pp(src).unwrap(), "\n\nint ok = 1;\n\n");
    }

    #[test]
    fn if_feature_tests() {
        let dir = test_dir("feat");
        write_file(&dir.join("have.h"), "int h = 1;\n");
        let main = dir.join("main.c");
        write_file(
            &main,
            "#if __has_include(\"have.h\")\nint a = 1;\n#else\nint a = 0;\n#endif\n#if __has_include(\"missing.h\")\nint b = 1;\n#else\nint b = 0;\n#endif\n#if __has_feature(address_sanitizer)\nint c = 1;\n#else\nint c = 0;\n#endif\n#if __has_attribute(cold) && defined __has_builtin && __has_builtin(__builtin_expect)\nint d = 1;\n#else\nint d = 0;\n#endif\n#if __has_extension(c_atomic)\nint e = 1;\n#else\nint e = 0;\n#endif\n",
        );
        let got = preprocess(&std::fs::read_to_string(&main).unwrap(), &main.display().to_string(), &opts()).unwrap();
        assert_eq!(got, "\nint a = 1;\n\n\nint b = 0;\n\n\nint c = 0;\n\n\nint d = 1;\n\n\nint e = 1;\n\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn elifdef_and_elifndef_select_branches() {
        let src = "#define B 1\n\
                   #if 0\nint a = 1;\n\
                   #elifdef B\nint b = 1;\n\
                   #elifndef C\nint c = 1;\n\
                   #else\nint d = 1;\n\
                   #endif\n\
                   #if 0\nint e = 1;\n\
                   #elifdef MISSING\nint f = 1;\n\
                   #elifndef ALSO_MISSING\nint g = 1;\n\
                   #else\nint h = 1;\n\
                   #endif\n";
        let got = pp(src).unwrap();
        assert!(got.contains("int b = 1;"), "{got}");
        assert!(!got.contains("int c = 1;"), "{got}");
        assert!(!got.contains("int d = 1;"), "{got}");
        assert!(got.contains("int g = 1;"), "{got}");
        assert!(!got.contains("int f = 1;"), "{got}");
        assert!(!got.contains("int h = 1;"), "{got}");
    }

    #[test]
    fn has_builtin_and_attribute_are_closed_tables() {
        let src = "#if __has_builtin(__builtin_expect)\nint a = 1;\n#else\nint a = 0;\n#endif\n\
                   #if __has_builtin(__builtin_stdc_leading_zeros)\nint b = 1;\n#else\nint b = 0;\n#endif\n\
                   #if __has_builtin(__builtin_stdc_rotate_left) && __has_builtin(__builtin_stdc_rotate_right)\nint r = 1;\n#else\nint r = 0;\n#endif\n\
                   #if __has_attribute(aligned)\nint c = 1;\n#else\nint c = 0;\n#endif\n\
                   #if __has_attribute(no_such_attr)\nint d = 1;\n#else\nint d = 0;\n#endif\n\
                   #if __has_c_attribute(nodiscard)\nint e = 1;\n#else\nint e = 0;\n#endif\n";
        let got = pp(src).unwrap();
        assert!(got.contains("int a = 1;"), "{got}");
        assert!(got.contains("int b = 0;"), "{got}");
        assert!(got.contains("int r = 1;"), "{got}");
        assert!(got.contains("int c = 1;"), "{got}");
        assert!(got.contains("int d = 0;"), "{got}");
        assert!(got.contains("int e = 1;"), "{got}");
    }

    #[test]
    fn has_c_attribute_is_a_closed_table() {
        let src = "#if __has_c_attribute(nodiscard) && __has_c_attribute(deprecated) && !__has_c_attribute(no_such_attr) && __has_c_attribute(gnu::packed)\nint a = 1;\n#else\nint a = 0;\n#endif\n";
        let got = pp(src).unwrap();
        assert!(got.contains("int a = 1;"), "{got}");
    }

    #[test]
    fn has_feature_gated_by_std() {
        assert!(has_feature("__has_feature", "c_atomic", None));
        assert!(has_feature("__has_feature", "c_atomic", Some("-std=c17")));
        assert!(!has_feature("__has_feature", "c_atomic", Some("-std=c99")));
        assert!(has_feature("__has_feature", "tls", Some("-std=c89")));
        assert!(has_feature("__has_extension", "c_attributes", Some("-std=c99")));
        assert!(!has_feature("__has_feature", "address_sanitizer", None));
    }

    #[test]
    fn error_directive_fails() {
        let e = pp("#error boom\n").unwrap_err();
        assert!(e.0.contains("#error"), "was: {e}");
    }

    #[test]
    fn include_quoted_relative_and_bracket_via_I() {
        let dir = test_dir("inc1");
        write_file(&dir.join("sub.h"), "int sub = 1;\n");
        write_file(&dir.join("sys.h"), "int sys = 2;\n");
        let main = dir.join("main.c");
        write_file(&main, "#include \"sub.h\"\n#include <sys.h>\nint m = sub + sys;\n");
        let o = PpOptions {
            include_paths: vec![dir.clone()],
            ..opts()
        };
        let got = preprocess(&std::fs::read_to_string(&main).unwrap(), &main.display().to_string(), &o).unwrap();
        assert_eq!(got, "int sub = 1;\n\nint sys = 2;\n\nint m = sub + sys;\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn windows_target_skips_clang_intrinsics_for_sdk() {
        use glcc_frontend::target::Target;
        let resource = test_dir("intrin-resource");
        let sdk = test_dir("intrin-sdk");
        write_file(&resource.join("intrin.h"), "clang-intrin\n");
        write_file(&sdk.join("intrin.h"), "sdk-intrin\n");
        let with_target = |target| PpOptions {
            include_paths: vec![resource.clone(), sdk.clone()],
            clang_resource_dir: Some(resource.clone()),
            target: Some(target),
            ..opts()
        };
        let windows = preprocess("#include <intrin.h>\n", "t.c", &with_target(Target::WINDOWS_X86_64)).unwrap();
        assert!(windows.contains("sdk-intrin"), "was: {windows}");
        let linux = preprocess("#include <intrin.h>\n", "t.c", &with_target(Target::X86_64)).unwrap();
        assert!(linux.contains("clang-intrin"), "was: {linux}");
        let _ = std::fs::remove_dir_all(&resource);
        let _ = std::fs::remove_dir_all(&sdk);
    }

    #[test]
    fn include_missing_is_precise_error() {
        let e = pp("#include \"nope.h\"\n").unwrap_err();
        assert!(e.0.contains("#include 'nope.h' not found"), "was: {e}");
    }

    #[test]
    fn include_errors_name_the_included_file() {
        let dir = test_dir("inc-err");
        write_file(&dir.join("bad.h"), "#endif\n");
        let main = dir.join("main.c");
        write_file(&main, "#include \"bad.h\"\n");
        let e = preprocess(&std::fs::read_to_string(&main).unwrap(), &main.display().to_string(), &opts()).unwrap_err();
        assert!(e.0.contains("bad.h:1: #endif without #if"), "was: {e}");
        assert!(!e.0.contains("main.c:1: #endif"), "was: {e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tgmath_header_dispatches_by_type() {
        let header_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../include");
        let mut o = opts();
        o.include_paths.push(header_dir);
        o.include_paths.extend(native_include_paths(None, Some("-std=c17")));
        o.predefined = native_predefines(None, Some("-std=c17"));
        let src = "#include <tgmath.h>\ndouble f(float x, long double y) { return sqrt(x) + sqrt(y); }\n";
        let got = preprocess(src, "t.c", &o).unwrap();
        assert!(got.contains("sqrtf"), "was: {got}");
        assert!(got.contains("sqrtl"), "was: {got}");
        assert!(got.contains("csqrtf"), "was: {got}");
        assert!(got.contains("csqrtl"), "was: {got}");
    }

    #[test]
    fn trigraphs_translate_before_c23() {
        assert_eq!(translate_trigraphs("??=??/??'??(??)??!??<??>??-").as_ref(), "#\\^[]|{}~");
        let c17 = PpOptions {
            std: Some("-std=c17".to_string()),
            ..opts()
        };
        let got = preprocess("int a = 1; ??/\nint b = 2;\n", "t.c", &c17).unwrap();
        assert!(got.contains("int a = 1; int b = 2;"), "was: {got}");
        let got = preprocess("int a??(2??) = {0};\n", "t.c", &c17).unwrap();
        assert!(got.contains("int a[2]"), "was: {got}");

        let c23 = PpOptions {
            std: Some("-std=c23".to_string()),
            ..opts()
        };
        let got = preprocess("int a??(2??) = {0};\n", "t.c", &c23).unwrap();
        assert!(!got.contains("int a[2]"), "was: {got}");
        assert_eq!(got, "int a??(2??) = {0};\n");
        assert!(!trigraphs_enabled(&c23));
        assert!(trigraphs_enabled(&PpOptions {
            trigraphs: Some(true),
            ..c23.clone()
        }));
    }

    #[test]
    fn trigraphs_apply_inside_includes() {
        let dir = test_dir("trigraph-include");
        write_file(&dir.join("tri.h"), "int h??(1??) = {2};\n");
        let main = dir.join("main.c");
        write_file(&main, "#include \"tri.h\"\n");
        let c17 = PpOptions {
            include_paths: vec![dir.clone()],
            std: Some("-std=c17".to_string()),
            ..opts()
        };
        let got = preprocess(&std::fs::read_to_string(&main).unwrap(), &main.display().to_string(), &c17).unwrap();
        assert!(got.contains("int h[1]"), "was: {got}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn digraph_directives_and_paste() {
        let src = "%:define X 1\n#define G(a,b) a %:%: b\n#define AB 3\nint a = G(A,B);\nint b<:2:> = <% X, 2 %>;\n";
        let got = pp(src).unwrap();
        assert!(got.contains("int a = 3;"), "was: {got}");
        assert!(got.contains("int b[2]"), "was: {got}");
    }

    #[test]
    fn include_in_inactive_branch_is_not_read() {
        let got = pp("#if 0\n#include \"nope.h\"\n#else\nint x = 1;\n#endif\n").unwrap();
        assert_eq!(got, "\nint x = 1;\n\n");
    }

    #[test]
    fn pragma_once_includes_once() {
        let dir = test_dir("inc2");
        write_file(&dir.join("once.h"), "#pragma once\nint o = 1;\n");
        let main = dir.join("main.c");
        write_file(&main, "#include \"once.h\"\n#include \"once.h\"\n");
        let got = preprocess(&std::fs::read_to_string(&main).unwrap(), &main.display().to_string(), &opts()).unwrap();
        assert_eq!(got, "\nint o = 1;\n\n\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fallback_predefines_cover_type_builtins() {
        let defs = fallback_predefines(Some("x86_64-linux-gnu"));
        assert!(defs.len() > 300, "got {}", defs.len());
        let get = |n: &str| defs.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone());
        assert_eq!(get("__SIZE_TYPE__").as_deref(), Some("long unsigned int"));
        assert_eq!(get("__STDC_VERSION__").as_deref(), Some("201710L"));
        assert_eq!(get("__x86_64__").as_deref(), Some("1"));
        // Function-like builtins keep their parameter lists.
        assert!(defs.iter().any(|(k, _)| k == "__INTMAX_C(c)"), "has INTMAX_C");
    }

    #[test]
    fn predefines_report_the_gnu_version_glcc_targets() {
        let pre = predefines(None, None);
        let get = |name: &str| pre.iter().find(|(existing, _)| existing == name).map(|(_, value)| value.as_str());
        assert_eq!(get("__GNUC__"), Some("4"));
        assert_eq!(get("__GNUC_MINOR__"), Some("8"));
        assert_eq!(get("__GNUC_PATCHLEVEL__"), Some("0"));
    }

    #[test]
    fn predefined_seed_flows_into_macros() {
        let o = PpOptions {
            predefined: vec![("PSEEDED".to_string(), "42".to_string())],
            ..opts()
        };
        assert_eq!(preprocess("int x = PSEEDED;\n", "t.c", &o).unwrap(), "int x = 42;\n");
    }

    #[test]
    fn linemarkers_bracket_file_transitions() {
        let dir = test_dir("inc3");
        write_file(&dir.join("a.h"), "int a = 1;\n");
        let main = dir.join("main.c");
        write_file(&main, "int t = 0;\n#include \"a.h\"\nint b = 2;\n");
        let o = PpOptions { linemarkers: true, ..opts() };
        let got = preprocess(&std::fs::read_to_string(&main).unwrap(), &main.display().to_string(), &o).unwrap();
        let main_d = main.display().to_string();
        let a_d = dir.join("a.h").display().to_string();
        assert!(got.starts_with(&format!("{}int t = 0;\n", marker(1, &main_d))), "was:\n{got}");
        assert!(got.contains(&format!("{}int a = 1;\n", marker_flag(1, &a_d, 1))), "was:\n{got}");
        assert!(got.ends_with(&format!("{}int b = 2;\n", marker_flag(3, &main_d, 2))), "was:\n{got}");
    }

    #[test]
    fn include_via_macro_expansion() {
        let dir = test_dir("inc4");
        write_file(&dir.join("m.h"), "int m = 9;\n");
        let main = dir.join("main.c");
        write_file(&main, "#define H \"m.h\"\n#include H\n");
        let got = preprocess(&std::fs::read_to_string(&main).unwrap(), &main.display().to_string(), &opts()).unwrap();
        assert_eq!(got, "\nint m = 9;\n\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn embed_emits_bytes_and_honours_parameters() {
        let dir = test_dir("embed");
        std::fs::write(dir.join("d.bin"), [1u8, 2, 255]).unwrap();
        write_file(&dir.join("empty.bin"), "");
        let main = dir.join("main.c");
        write_file(
            &main,
            "#define H \"d.bin\"\n\
             const unsigned char a[] = {\n#embed H\n};\n\
             const unsigned char b[] = {\n#embed \"d.bin\" limit(2)\n};\n\
             const unsigned char c[] = {\n#embed \"d.bin\" suffix(, 9)\n};\n\
             const unsigned char d[] = {\n#embed \"empty.bin\" if_empty(7)\n};\n",
        );
        let got = preprocess(&std::fs::read_to_string(&main).unwrap(), &main.display().to_string(), &opts()).unwrap();
        let compact: String = got.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(compact.contains("a[]={1,2,255};"), "{got}");
        assert!(compact.contains("b[]={1,2};"), "{got}");
        assert!(compact.contains("c[]={1,2,255,9};"), "{got}");
        assert!(compact.contains("d[]={7};"), "{got}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn has_embed_reports_found_empty_and_missing() {
        let dir = test_dir("has-embed");
        std::fs::write(dir.join("d.bin"), [1u8, 2]).unwrap();
        write_file(&dir.join("empty.bin"), "");
        let main = dir.join("main.c");
        write_file(
            &main,
            "#define H \"d.bin\"\n\
             #if __has_embed(H) != 1\n#error found\n#endif\n\
             #if __has_embed(\"empty.bin\") != 2\n#error empty\n#endif\n\
             #if __has_embed(\"missing.bin\") != 0\n#error missing\n#endif\n\
             #if __has_embed(\"d.bin\" limit(0)) != 2\n#error limit\n#endif\n",
        );
        preprocess(&std::fs::read_to_string(&main).unwrap(), &main.display().to_string(), &opts()).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn embed_errors_on_missing_resources_and_bad_limits() {
        let dir = test_dir("embed-errors");
        std::fs::write(dir.join("d.bin"), [1u8]).unwrap();
        let missing = dir.join("missing.c");
        write_file(&missing, "#embed \"missing.bin\"\n");
        assert!(preprocess(&std::fs::read_to_string(&missing).unwrap(), &missing.display().to_string(), &opts()).is_err());
        let bad = dir.join("bad.c");
        write_file(&bad, "#embed \"d.bin\" limit(-1)\n");
        let e = preprocess(&std::fs::read_to_string(&bad).unwrap(), &bad.display().to_string(), &opts()).unwrap_err();
        assert!(e.0.contains("limit is negative"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    static TEST_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn test_dir(tag: &str) -> PathBuf {
        let n = TEST_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("glcc-pp-test-{}-{n}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_file(path: &PathBuf, contents: &str) {
        std::fs::write(path, contents).unwrap();
    }

    #[test]
    fn comments_become_spaces_but_strings_survive() {
        let src = "int x = 1; // trailing\n/* block\nspanning */ int y = \"a//b\";\n";
        assert_eq!(pp(src).unwrap(), "int x = 1; \n\n  int y = \"a//b\";\n");
    }

    #[test]
    fn cli_flag_parsing_keeps_d_u_i() {
        let cpp = vec![
            OsString::from("-DFOO=1"),
            OsString::from("-U"),
            OsString::from("BAR"),
            OsString::from("-Iinc"),
            OsString::from("-std=c11"),
            OsString::from("-isystem"),
            OsString::from("sys"),
            OsString::from("-includepre.h"),
        ];
        let o = cpp_flags_to_opts(&cpp, "t.c");
        assert_eq!(o.defines, vec![("FOO".to_string(), Some("1".to_string()))]);
        assert_eq!(o.undefs, vec!["BAR".to_string()]);
        assert_eq!(o.include_paths, vec![PathBuf::from("inc"), PathBuf::from("sys")]);
        assert_eq!(o.preincludes, vec!["pre.h".to_string()]);
    }

    #[test]
    fn function_like_macros_expand() {
        let src = "#define ADD(a, b) ((a) + (b))\nint x = ADD(1, 2);\n";
        assert_eq!(pp(src).unwrap(), "\nint x = ((1) + (2));\n");
    }

    #[test]
    fn function_macros_nest_and_repeat_on_one_line() {
        let src = "#define ADD(a, b) ((a) + (b))\nint x = ADD(1, 2) + ADD(3, 4) + ADD(ADD(1, 1), 0);\n";
        assert_eq!(pp(src).unwrap(), "\nint x = ((1) + (2)) + ((3) + (4)) + ((((1) + (1))) + (0));\n");
    }

    #[test]
    fn function_macro_without_call_is_literal() {
        let src = "#define F(a) ((a))\nint x = F;\nint y = F ;\n";
        assert_eq!(pp(src).unwrap(), "\nint x = F;\nint y = F ;\n");
    }

    #[test]
    fn function_macro_arity_mismatch_errors() {
        let e = pp("#define F(a, b) x\nint y = F(1);\n").unwrap_err();
        assert!(e.0.contains("needs 2 arguments, got 1"), "was: {e}");
    }

    #[test]
    fn stringify_and_glue() {
        let src = "#define S(x) #x\n#define CAT(a, b) a ## b\nconst char *s = S(hello  world);\nint xy = CAT(x, y);\n";
        assert_eq!(pp(src).unwrap(), "\n\nconst char *s = \"hello world\";\nint xy = xy;\n");
    }

    #[test]
    fn variadic_and_comma_elision() {
        let src = "#define LOG(fmt, ...) logf(fmt, __VA_ARGS__)\n#define ELOG(fmt, ...) logf(fmt, ##__VA_ARGS__)\nLOG(\"%d\", 1);\nELOG(\"x\");\n";
        assert_eq!(pp(src).unwrap(), "\n\nlogf(\"%d\", 1);\nlogf(\"x\");\n");
    }

    #[test]
    fn macro_names_inside_strings_do_not_expand() {
        let src = "#define FOO 1\nconst char *s = \"FOO\";\n";
        assert_eq!(pp(src).unwrap(), "\nconst char *s = \"FOO\";\n");
    }

    #[test]
    fn expansion_boundary_spacing_matches_clang() {
        // `W+1` -> `11 +1`, but `(W)+1` -> `(11)+1`, `I+1` -> `foo+1`,
        // and identifiers never trigger padding (`mem0.mutex`).
        let src = "#define W 11\n#define I foo\n#define M mem0\nint a=W+1;\nint b=(W)+1;\nint c=I+1;\nint d=W-W;\nint e=++W;\nint f=M.mutex;\nint g=M+1;\n";
        assert_eq!(
            pp(src).unwrap(),
            "\n\n\nint a=11 +1;\nint b=(11)+1;\nint c=foo+1;\nint d=11 -11;\nint e=++11;\nint f=mem0.mutex;\nint g=mem0+1;\n"
        );
    }

    #[test]
    fn function_body_literals_pad_like_clang() {
        let src = "#define F(x) 11\nint a=F(x)+1;\nint b=1-F(x);\n";
        assert_eq!(pp(src).unwrap(), "\nint a=11 +1;\nint b=1-11;\n");
    }

    #[test]
    fn multiline_calls_expand_whole() {
        let src = "#define ADD(a, b) ((a) + (b))\nint x = ADD(1,\n  2);\nint y = 3;\n";
        assert_eq!(pp(src).unwrap(), "\nint x = ((1) + (2));\nint y = 3;\n");
    }

    #[test]
    fn empty_call_is_one_empty_argument() {
        let src = "#define S(x) #x\nconst char *a = S();\n#define G() 2\nint g = G( );\n";
        assert_eq!(pp(src).unwrap(), "\nconst char *a = \"\";\n\nint g = 2;\n");
    }

    #[test]
    fn line_and_file_expand_at_use_site() {
        let src = "int a = __LINE__;\n#define L __LINE__\nint b = L;\nconst char *f = __FILE__;\n";
        assert_eq!(pp(src).unwrap(), "int a = 1;\n\nint b = 3;\nconst char *f = \"t.c\";\n");
    }

    #[test]
    fn self_paint_stops_at_one_level() {
        let src = "#define F(a) a\nint x = F(F);\n";
        assert_eq!(pp(src).unwrap(), "\nint x = F;\n");
    }

    #[test]
    fn if_with_function_macro() {
        let src = "#define V(x) (x)\n#if V(1) + V(0)\nint ok = 1;\n#endif\n";
        assert_eq!(pp(src).unwrap(), "\n\nint ok = 1;\n\n");
    }

    #[test]
    fn line_directive_expands_macro_operands() {
        let src = "#define L 1000\n#line L\nint a = __LINE__;\n";
        assert_eq!(pp(src).unwrap(), "\n\nint a = 1000;\n");
    }

    #[test]
    fn line_number_can_move_backwards() {
        let src = "int a = __LINE__;\n#line 1\nint b = __LINE__;\n";
        assert_eq!(pp(src).unwrap(), "int a = 1;\n\nint b = 1;\n");
    }

    #[test]
    fn line_directive_rejects_non_number() {
        let err = pp("#line foo\n").unwrap_err();
        assert!(err.0.contains("malformed #line"), "{err}");
    }

    #[test]
    fn gnu_linemarker_remaps_line_and_file() {
        // GNU's preprocessed format: `# <line> "file" [flags]`.
        let src = "int a = __LINE__;\n# 7 \"hello.c\"\nint b = __LINE__;\nconst char *f = __FILE__;\n";
        assert_eq!(pp(src).unwrap(), "int a = 1;\n\nint b = 7;\nconst char *f = \"hello.c\";\n");
    }

    #[test]
    fn gnu_linemarker_bare_form_keeps_file() {
        let src = "int a = __LINE__;\n# 1\nint b = __LINE__;\nconst char *f = __FILE__;\n";
        assert_eq!(pp(src).unwrap(), "int a = 1;\n\nint b = 1;\nconst char *f = \"t.c\";\n");
    }

    #[test]
    fn gnu_linemarker_drops_integer_flags() {
        let src = "# 3 \"sys.h\" 3\nint x = __LINE__;\n";
        assert_eq!(pp(src).unwrap(), "\nint x = 3;\n");
    }

    #[test]
    fn gnu_linemarker_rejects_junk() {
        let err = pp("# 1 foo\n").unwrap_err();
        assert!(err.0.contains("malformed #line"), "{err}");
        let err = pp("# 1 \"f\" x\n").unwrap_err();
        assert!(err.0.contains("malformed #line"), "{err}");
    }

    #[test]
    fn pragma_push_pop_macro() {
        let src = "#define X 1\n#pragma push_macro(\"X\")\n#undef X\n#define X 2\nX\n#pragma pop_macro(\"X\")\nX\n";
        assert_eq!(pp(src).unwrap(), "\n\n\n\n2\n\n1\n");
    }

    #[test]
    fn pragma_pop_macro_restores_undefined() {
        let src = "#pragma push_macro(\"Y\")\n#define Y 1\n#pragma pop_macro(\"Y\")\n#ifdef Y\nD\n#else\nND\n#endif\nY\n";
        assert_eq!(pp(src).unwrap(), "\n\n\n\nND\n\nY\n");
    }

    #[test]
    fn counter_base_file_and_include_level() {
        let src = "a __COUNTER__ b __COUNTER__ c __BASE_FILE__ l __INCLUDE_LEVEL__\n";
        assert_eq!(pp(src).unwrap(), "a 0 b 1 c \"t.c\" l 0\n");
    }

    #[test]
    fn redefine_and_undef_location_macros() {
        let src = "#define __DATE__ 1\n__DATE__\n#undef __DATE__\n#ifdef __DATE__\nD\n#else\nND\n#endif\n__DATE__\n";
        assert_eq!(pp(src).unwrap(), "\n1\n\n\nND\n\n__DATE__\n");
    }

    #[test]
    fn date_time_and_timestamp_format() {
        let out = pp("__DATE__ __TIME__\n").unwrap();
        let out = out.trim();
        let (date, time) = out.split_once("\" \"").unwrap_or_else(|| panic!("date+time: {out:?}"));
        let date = date.trim_start_matches('"');
        let time = time.trim_end_matches('"');
        assert_eq!(date.len(), 11, "date: {date:?}");
        assert_eq!(time.len(), 8, "time: {time:?}");
        assert_eq!(time.as_bytes()[2], b':');
        assert_eq!(time.as_bytes()[5], b':');
        let ts = pp("__TIMESTAMP__\n").unwrap();
        let ts = ts.trim().trim_matches('"');
        assert_eq!(ts.len(), 24, "timestamp: {ts:?}");
        assert_eq!(ts.as_bytes()[13], b':');
        assert_eq!(ts.as_bytes()[16], b':');
    }

    #[test]
    fn va_opt_expands_only_for_a_nonempty_pack() {
        let src = "#define F(...) f(0 __VA_OPT__(,) __VA_ARGS__)\n\
                   #define G(x, ...) g(x __VA_OPT__(, __VA_ARGS__))\n\
                   #define H(...) h(__VA_OPT__(a) b)\n\
                   F();\nF(1, 2);\nG(3);\nG(3, 4);\nH();\nH(1);\n";
        let got = pp(src).unwrap();
        assert!(got.contains("f(0  );"), "{got}");
        assert!(got.contains("f(0 , 1,2);"), "{got}");
        assert!(got.contains("g(3 );"), "{got}");
        assert!(got.contains("g(3 , 4);"), "{got}");
        assert!(got.contains("h( b);"), "{got}");
        assert!(got.contains("h(a b);"), "{got}");
    }

    #[test]
    fn va_opt_pastes_and_keeps_line_starts() {
        let src = "#define CAT(a, ...) a ## __VA_OPT__(b)\n\
                   #define CAT2(...) __VA_OPT__(x) ## y\n\
                   CAT(p,);\nCAT(p,1);\nCAT2();\nCAT2(1);\n\
                   int a;\n#define E(...) __VA_OPT__(x) y\nE()\nint b;\n";
        let got = pp(src).unwrap();
        let lines: Vec<&str> = got.lines().map(|line| line.trim().trim_end_matches(';')).collect();
        for want in ["p", "pb", "y", "xy"] {
            assert!(lines.contains(&want), "missing {want:?}: {got}");
        }
    }

    #[test]
    fn nostdinc_flag_recorded() {
        let cpp = vec![OsString::from("-nostdinc")];
        assert!(cpp_flags_to_opts(&cpp, "t.c").no_std_includes);
        let plain = vec![OsString::from("-I"), OsString::from("inc")];
        assert!(!cpp_flags_to_opts(&plain, "t.c").no_std_includes);
    }

    #[test]
    fn clang_search_list_parsing() {
        let stderr = "#include \"...\" search starts here:\n#include <...> search starts here:\n /usr/lib/clang/22/include\n /usr/local/include\n /Library/Frameworks (framework directory)\n /usr/include\nEnd of search list.\n";
        assert_eq!(
            parse_search_dirs(stderr),
            vec![
                PathBuf::from("/usr/lib/clang/22/include"),
                PathBuf::from("/usr/local/include"),
                PathBuf::from("/usr/include")
            ]
        );
    }

    #[test]
    fn fallback_includes_existing_dirs() {
        let dirs = fallback_include_paths(Some("x86_64-linux-gnu"));
        assert!(dirs.iter().all(|d| d.is_dir()), "got {dirs:?}");
        // The POSIX layout is the only one this fallback knows about.
        #[cfg(not(windows))]
        assert!(dirs.contains(&PathBuf::from("/usr/include")), "got {dirs:?}");
    }

    #[test]
    fn native_include_paths_exist() {
        let dirs = native_include_paths(Some("x86_64-linux-gnu"), None);
        assert!(dirs.iter().all(|d| d.is_dir()), "got {dirs:?}");
        // A POSIX toolchain (or clang) always yields something; a bare
        // Windows host without either may legitimately have none.
        #[cfg(not(windows))]
        assert!(!dirs.is_empty(), "no include paths");
    }
}
