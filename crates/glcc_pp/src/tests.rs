#![allow(non_snake_case)]

use super::*;
use glcc_frontend::target::Target;

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
    o.include_paths.extend(include_paths(&Target::native()));
    o.predefined = predefines(&Target::native(), Some("-std=c17"));
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
fn predefines_cover_type_builtins() {
    let defs = predefines(&Target::X86_64, None);
    assert!(defs.len() > 300, "got {}", defs.len());
    let get = |n: &str| defs.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone());
    assert_eq!(get("__SIZE_TYPE__").as_deref(), Some("long unsigned int"));
    assert_eq!(get("__STDC_VERSION__").as_deref(), Some("201710L"));
    assert_eq!(get("__x86_64__").as_deref(), Some("1"));
    assert_eq!(get("__LONG_MAX__").as_deref(), Some("9223372036854775807L"));
    assert_eq!(get("__INT64_C(c)").as_deref(), Some("c##L"));
    // Function-like builtins keep their parameter lists.
    assert!(defs.iter().any(|(k, _)| k == "__INTMAX_C(c)"), "has INTMAX_C");
}

#[test]
fn predefines_windows_are_msvc_mode() {
    let defs = predefines(&Target::WINDOWS_X86_64, None);
    let get = |n: &str| defs.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone());
    assert_eq!(get("_WIN64").as_deref(), Some("1"));
    assert!(get("_MSC_VER").is_some(), "lacks _MSC_VER");
    assert_eq!(get("_INTEGRAL_MAX_BITS").as_deref(), Some("64"));
    // LLP64: `long` stays 4 bytes; the Linux markers are gone.
    assert_eq!(get("__SIZEOF_LONG__").as_deref(), Some("4"));
    assert_eq!(get("__LONG_MAX__").as_deref(), Some("2147483647L"));
    assert_eq!(get("__SIZE_TYPE__").as_deref(), Some("long long unsigned int"));
    assert_eq!(get("__WCHAR_TYPE__").as_deref(), Some("unsigned short"));
    assert!(get("__linux__").is_none());
    assert!(get("__LP64__").is_none());
    // The GNU/Linux markers only appear for the Linux targets.
    let linux = predefines(&Target::X86_64, None);
    assert!(linux.iter().any(|(k, _)| k == "__linux__"));
}

#[test]
fn predefines_follow_the_std_flag() {
    let version = |flag: &str| {
        predefines(&Target::X86_64, Some(flag))
            .iter()
            .find(|(name, _)| name == "__STDC_VERSION__")
            .map(|(_, value)| value.clone())
    };
    assert_eq!(version("-std=c99").as_deref(), Some("199901L"));
    assert_eq!(version("-std=c11").as_deref(), Some("201112L"));
    assert_eq!(version("-std=c23").as_deref(), Some("202311L"));
    assert_eq!(version("-std=gnu2y").as_deref(), Some("202400L"));
    assert_eq!(version("-std=c89"), None);
    assert_eq!(version("-ansi"), None);
}

#[test]
fn predefines_track_the_target_architecture() {
    let aarch64 = predefines(&Target::AARCH64, None);
    let get = |n: &str| aarch64.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone());
    assert_eq!(get("__aarch64__").as_deref(), Some("1"));
    assert_eq!(get("__CHAR_UNSIGNED__").as_deref(), Some("1"));
    assert_eq!(get("__WCHAR_TYPE__").as_deref(), Some("unsigned int"));
    assert_eq!(get("__LDBL_MANT_DIG__").as_deref(), Some("113"));
    let riscv = predefines(&Target::RISCV64, None);
    assert!(riscv.iter().any(|(k, v)| k == "__riscv_xlen" && v == "64"));
    assert!(riscv.iter().any(|(k, v)| k == "__CHAR_UNSIGNED__"));
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
fn include_paths_exist() {
    let dirs = include_paths(&Target::X86_64);
    assert!(dirs.iter().all(|d| d.is_dir()), "got {dirs:?}");
    // A POSIX toolchain always yields something; a bare Windows
    // host without either may legitimately have none.
    #[cfg(not(windows))]
    assert!(!dirs.is_empty(), "no include paths");
}

#[test]
fn include_operand_expands_function_like_macros() {
    // The kernel's trace headers do `#include TRACE_INCLUDE(TRACE_SYSTEM)`:
    // a function-like macro in the operand whose arguments come from the
    // same line. The operand must expand as one sequence, ending in a
    // stringized header name.
    let dir = std::env::temp_dir();
    let header = dir.join("glcc_pp_include_macro_test.h");
    std::fs::write(&header, "int from_header;\n").unwrap();
    let src = format!(
        "#define __STR1(x) #x\n\
         #define __STR(x) __STR1(x)\n\
         #define WI(x) __STR(x)\n\
         #include WI({})\nint y;\n",
        header.display()
    );
    let got = pp(&src).unwrap();
    assert!(got.contains("from_header"), "{got}");
    assert!(got.contains("int y;"), "{got}");
}

#[test]
fn non_assembler_mode_errors_on_unknown_directive() {
    let e = pp("#bogus x\nint y;\n").unwrap_err();
    assert!(e.0.contains("unknown directive '#bogus'"), "{e}");
}

#[test]
fn assembler_mode_passes_unknown_directives_through_verbatim() {
    let o = PpOptions { assembler: true, ..opts() };
    let src = "# Setup for GCC\n#APP\nnop\n#NO_APP\n\t# indented tail\n";
    assert_eq!(preprocess(src, "t.S", &o).unwrap(), src);
}

#[test]
fn assembler_mode_comments_do_not_expand_macros() {
    let o = PpOptions {
        assembler: true,
        defines: vec![("reload".to_string(), Some("KILLED".to_string()))],
        ..opts()
    };
    let got = preprocess("#define K 7\n# reload K now\nint v = K;\n", "t.S", &o).unwrap();
    assert!(got.contains("# reload K now"), "{got}");
    assert!(got.contains("int v = 7;"), "{got}");
}

#[test]
fn assembler_mode_keeps_known_directives_working() {
    let o = PpOptions { assembler: true, ..opts() };
    let src = "#define FOO 3\nint v = FOO;\n#if FOO != 3\n#error wrong\n#endif\n# plain comment\n";
    let got = preprocess(src, "t.S", &o).unwrap();
    assert!(got.contains("int v = 3;"), "{got}");
    assert!(got.contains("# plain comment"), "{got}");
    assert!(!got.contains("#error"), "{got}");
}

#[test]
fn assembler_mode_linemarkers_still_remap() {
    let dir = test_dir("asmlin");
    let inc = dir.join("sub.S");
    write_file(&inc, "int from_inc;\n");
    let main = dir.join("m.S");
    write_file(&main, "# a comment\n#include \"sub.S\"\n# 5 \"foo.S\" 1\nint y;\n");
    let o = PpOptions { linemarkers: true, assembler: true, ..opts() };
    let got = preprocess(&std::fs::read_to_string(&main).unwrap(), &main.display().to_string(), &o).unwrap();
    let main_d = main.display().to_string();
    assert!(got.starts_with(&marker(1, &main_d)), "was:\n{got}");
    assert!(got.contains(" # a comment\n"), "was:\n{got}");
    assert!(got.contains("int from_inc;\n"), "was:\n{got}");
    // The `# 5 "foo.S" 1` input linemarker is consumed (not passed
    // through as a comment): the line number remaps so `int y;`
    // accounts as line 5, exactly like C mode (no file-rename marker).
    assert!(got.contains(&format!("{}\n\nint y;\n", marker_flag(3, &main_d, 2))), "was:\n{got}");
    assert!(!got.contains("foo.S"), "was:\n{got}");
    let _ = std::fs::remove_dir_all(&dir);
}
