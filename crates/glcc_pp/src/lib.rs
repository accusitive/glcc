//! `glcc_pp`: native C preprocessor.
//!
//! The driver and the explorer prefer this preprocessor; `--pp=clang`
//! (or a `GLCC_PP` value other than `native`) selects the `clang -E`
//! path instead. Predefined macros (`predefine.rs`) and the default
//! system include search (`includes.rs`) are glcc's own, computed
//! from the target — no clang/gcc dump is consulted.
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
//! - `predefine.rs`: the predefined macro table; `includes.rs`: the
//!   default system include search.
//! - This file: options/flags, the file layer (`#include` search,
//!   `#pragma once`, linemarkers), output rendering, tests.
//!
//! Deliberate deviations from a by-the-book C preprocessor:
//! `#include` recurses through the
//! file layer here (so markers/`#pragma once`/search behave); clang
//! `__has_*` builtins, `__LINE__`/`__FILE__`, and output padding match
//! `clang -E`; `#if` garbage selects `#else` instead of erroring.

mod expand;
mod includes;
mod macros;
mod predefine;
mod prepend;
mod substitute;
mod timefmt;
mod tree;

pub use glcc_toolchain::{MsvcDirs, find_clang, msvc_dirs};
pub use includes::include_paths;
pub use predefine::predefines;

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
    /// Predefined macros ([`predefines`]), applied before `-D`/`-U`
    /// like GCC's own predefined macros. Names may carry `(params)`
    /// for builtin function-likes.
    pub predefined: Vec<(String, String)>,
    /// `-DNAME` or `-DNAME=replacement` (object- or function-like).
    pub defines: Vec<(String, Option<String>)>,
    /// `-U NAME`.
    pub undefs: Vec<String>,
    /// `-I`/`-isystem`/`-iquote` paths, in order.
    pub include_paths: Vec<PathBuf>,
    /// `-nostdinc`: callers do not append [`include_paths`].
    pub no_std_includes: bool,
    /// Raw `-std=`/`-ansi` flag when the driver was given one (selects
    /// `__STDC_VERSION__` in [`predefines`] and gates `__has_feature`
    /// answers).
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
    /// Assembler-with-cpp mode (`.S` inputs): an unrecognized `#word`
    /// line is a GAS comment, passed through verbatim instead of an
    /// error. Known directives and `# <n>` linemarkers keep working.
    pub assembler: bool,
    /// `-dM`: instead of the expanded text, emit `#define NAME body`
    /// lines for every macro defined at the end of preprocessing
    /// (builtins and `-D` included), sorted by name like clang.
    pub dump_macros: bool,
}

/// The clang resource include directory of an include list, when the
/// list starts with one (see `includes::include_paths`): it alone
/// ships clang's builtin headers such as `__stddef_max_align_t.h`.
/// `None` when the first entry is not a clang resource directory (e.g.
/// a GCC sysroot supplied the resource headers).
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
    /// Every file a directive resolved to (main source, `-include`,
    /// `#include`, `#include_next`, `#embed`, `__has_include` probes),
    /// in resolution order — the input for `-MD`-style depfiles.
    pub deps: Vec<PathBuf>,
}

impl Ctx {
    /// Search for a header: quoted tries the includer's directory first,
    /// then `-I` paths; bracketed tries `-I` paths only. With `after`,
    /// resume past the current file (`#include_next`).
    pub fn resolve_header(&mut self, name: &str, quoted: bool, after: Option<FoundKind>, dir: &Path, loc: &str) -> Result<(PathBuf, FoundKind), PpError> {
        let start = match after {
            None | Some(FoundKind::Start) | Some(FoundKind::QuoterDir) => None,
            Some(FoundKind::Index(i)) => Some(i + 1),
        };
        if quoted && after.is_none() {
            let cand = dir.join(name);
            if cand.is_file() {
                self.deps.push(cand.clone());
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
                self.deps.push(cand.clone());
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

/// Push a linemarker, keeping it at column 0: a source file may end
/// without a newline (C requires one, so behave as if it were there).
pub(crate) fn push_marker(out: &mut String, s: &str) {
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(s);
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
    preprocess_with_all(source, filename, opts).map(|(text, warnings, _)| (text, warnings))
}

/// [`preprocess_with_warnings`], also returning the files every
/// directive resolved to (main source first, then `-include` files,
/// then headers in resolution order) for depfile generation.
pub fn preprocess_with_all(source: &str, filename: &str, opts: &PpOptions) -> Result<(String, Vec<PpWarning>, Vec<PathBuf>), PpError> {
    let source = translate_source(source, opts);
    let mut interner = Interner::new();
    let mut ctx = Ctx {
        opts: opts.clone(),
        out: String::new(),
        once: HashSet::new(),
        depth: 0,
        warnings: Vec::new(),
        deps: Vec::new(),
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
    ctx.deps.push(main_path.clone());
    // `-include` files come first, as if included at line 1. They are
    // command-line inclusions, so a relative name resolves against the
    // working directory, not the main source's directory (kbuild passes
    // `-include ./include/...`).
    for pre in opts.preincludes.clone() {
        let (target, found) = ctx
            .resolve_header(&pre, true, None, Path::new("."), filename)
            .map_err(|_| err(format!("{filename}: -include '{pre}' not found")))?;
        let canon = std::fs::canonicalize(&target).unwrap_or_else(|_| target.clone());
        ctx.deps.push(canon.clone());
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
        push_marker(&mut ctx.out, &marker(1, filename));
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
    if opts.dump_macros {
        ctx.out = dump_macros(&interner, &macros);
    }
    Ok((ctx.out, ctx.warnings, ctx.deps))
}

/// `-dM`: every macro defined at the end of preprocessing, as
/// `#define` lines sorted by name (clang's ordering).
fn dump_macros(interner: &Interner, macros: &HashMap<SymbolU32, MacroDef>) -> String {
    use substitute::render_raw;
    use std::fmt::Write as _;
    let mut defs: Vec<(String, &MacroDef)> = macros
        .iter()
        .map(|(sym, def)| (interner.resolve(*sym).unwrap_or("").to_string(), def))
        .collect();
    defs.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = String::new();
    for (name, def) in defs {
        match def {
            MacroDef::Object(body) => {
                let _ = write!(out, "#define {name}");
                let body = render_raw(body, interner);
                if !body.is_empty() {
                    let _ = write!(out, " {}", body.trim_end());
                }
                out.push('\n');
            }
            MacroDef::Func { params, variadic, named_variadic, body } => {
                let mut params = params.join(", ");
                if *variadic {
                    if *named_variadic {
                        params.push_str("...");
                    } else if params.is_empty() {
                        params = "...".to_string();
                    } else {
                        params.push_str(", ...");
                    }
                }
                let _ = write!(out, "#define {name}({params})");
                let body = render_raw(body, interner);
                if !body.is_empty() {
                    let _ = write!(out, " {}", body.trim_end());
                }
                out.push('\n');
            }
        }
    }
    out
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
        let control = parse_control_line(&rest, filename, 1, interner, false)?;
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
    let trees = tree::collect_trees(toks, &disp_str, interner, opts.assembler)?;
    if entered && opts.linemarkers {
        push_marker(&mut ctx.out, &marker_flag(1, &disp_str, 1));
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
mod tests;

