//! Representative standard-mode probes. These are deliberately small:
//! each one names a feature the selected standard introduced, so a
//! regression shows up as a failed parse in the matching mode. The
//! original audit matrix lives in `docs/c-standard-audit.md`.

use super::tu_diagnostics_with_lang;
use crate::lang::{LangOpts, Std};

fn lang(std: Std) -> LangOpts {
    LangOpts {
        std,
        gnu: false,
        ms: false,
        clang: false,
        strict: false,
    }
}

fn accepts(name: &str, std: Std, src: &str) {
    let diags = tu_diagnostics_with_lang(src, lang(std));
    assert!(
        diags.iter().all(|d| !d.is_error()),
        "{name} should compile in {std:?}: {diags:#?}"
    );
}

fn rejects(name: &str, std: Std, src: &str) {
    let mut strict = lang(std);
    strict.strict = true;
    let diags = tu_diagnostics_with_lang(src, strict);
    assert!(
        diags.iter().any(|d| d.is_error()),
        "{name} should be rejected in {std:?}: {diags:#?}"
    );
}

#[test]
fn c99_language_probes() {
    let std = Std::C99;
    for (name, src) in [
        ("line comment", "int x; // comment\n"),
        ("long long", "long long x;\n"),
        ("inline", "inline int f(void) { return 0; }\n"),
        ("restrict", "int f(int *restrict p) { return *p; }\n"),
        ("VLA", "int f(int n) { int a[n]; return (int)sizeof a; }\n"),
        ("flexible array member", "struct S { int n; int a[]; };\n"),
        ("designated initializer", "struct S { int a, b; }; struct S s = { .b = 1 };\n"),
        ("compound literal", "int f(void) { return ((int[]){1, 2})[1]; }\n"),
        ("hexadecimal float", "double x = 0x1p2;\n"),
        ("__func__", "const char *f(void) { return __func__; }\n"),
        ("universal character name", "int \\u00e9 = 1;\n"),
        ("enumerator trailing comma", "enum E { A, };\n"),
        ("mixed declarations", "int f(void) { int x = 0; x++; int y = 1; return x + y; }\n"),
        ("for declaration", "int f(void) { for (int i = 0; i < 1; i++) { } return 0; }\n"),
        ("static array parameter", "void f(int a[static 1]) { }\n"),
        ("complex type", "double _Complex z = 1.0;\n"),
        ("digraph", "int a<:1:> = { 1 };\n"),
    ] {
        accepts(name, std, src);
    }
    // A C23 feature must not leak into a strict C99 parse.
    rejects("nullptr is C23", std, "int f(void) { return nullptr == 0; }\n");
}

#[test]
fn c11_language_probes() {
    let std = Std::C11;
    for (name, src) in [
        ("_Static_assert", "_Static_assert(1, \"ok\");\n"),
        ("_Noreturn", "_Noreturn void f(void);\n"),
        ("_Thread_local", "_Thread_local int x;\n"),
        ("_Alignof", "int f(void) { return (int)_Alignof(int); }\n"),
        ("_Generic", "int f(void) { return _Generic(1, int: 1, default: 0); }\n"),
        ("anonymous union", "struct S { union { int a; float b; }; };\n"),
        ("_Bool bit-field", "struct S { _Bool b : 1; };\n"),
        ("_Alignas", "_Alignas(16) int x;\n"),
        ("_Atomic scalar", "_Atomic int x;\n"),
        ("_Atomic aggregate", "struct S { int a; }; _Atomic(struct S) x;\n"),
        ("block static assert", "int f(void) { _Static_assert(1, \"ok\"); return 0; }\n"),
    ] {
        accepts(name, std, src);
    }
    rejects("C23 bool", std, "bool b = true;\n");
}

#[test]
fn c17_language_probes() {
    let std = Std::C17;
    for (name, src) in [
        ("_Static_assert", "_Static_assert(1, \"ok\");\n"),
        ("_Generic", "int f(void) { return _Generic(1, int: 1, default: 0); }\n"),
    ] {
        accepts(name, std, src);
    }
}

#[test]
fn c23_language_probes() {
    let std = Std::C23;
    for (name, src) in [
        ("bool/true/false", "bool b = true; bool c = false;\n"),
        ("nullptr", "int f(void) { return nullptr == 0; }\n"),
        ("constexpr", "constexpr int n = 3;\n"),
        ("auto inference", "auto x = 2.5;\n"),
        ("typeof_unqual", "typeof_unqual(int) x;\n"),
        ("bare static_assert", "static_assert(1);\n"),
        ("bare alignof", "int f(void) { return (int)alignof(int); }\n"),
        ("bare alignas", "alignas(16) int x;\n"),
        ("binary literal", "int x = 0b1010;\n"),
        ("digit separator", "int x = 1'000;\n"),
        ("u8 character", "int f(void) { return _Generic(u8'c', unsigned char: 1, default: 0); }\n"),
        ("u8 string", "int f(void) { return _Generic(u8\"x\"[0], unsigned char: 1, default: 0); }\n"),
        ("wb suffix", "_BitInt(8) x = 1wb;\n"),
        ("_BitInt", "_BitInt(17) x;\n"),
        ("attributes", "[[nodiscard]] int f(void);\n"),
        ("fixed enum", "enum E : unsigned char { A, B };\n"),
        ("compound storage class", "int f(void) { return ((static int[]){1, 2})[0]; }\n"),
        ("label before declaration", "int f(void) { L: int x = 1; return x; }\n"),
        ("label at block end", "int f(void) { goto end; end: }\n"),
        ("nullptr comparison", "int f(void) { return nullptr == nullptr; }\n"),
        ("nullptr generic identity", "int f(void) { return _Generic(nullptr, void *: 0, default: 1); }\n"),
        ("auto nullptr", "int f(void) { auto p = nullptr; return p == nullptr; }\n"),
    ] {
        accepts(name, std, src);
    }
    rejects("C2y _Countof", std, "int f(void) { int a[3]; return (int)_Countof(a); }\n");
    rejects("C2y expression static_assert", std, "int f(void) { return (static_assert(1), 5); }\n");
    rejects("C2y 0o octal", std, "int x = 0o17;\n");
    rejects("C2y complex increment", std, "int f(void) { double _Complex z = 1.0; z++; return 0; }\n");
    rejects("C2y __auto_type", std, "int f(void) { __auto_type x = 1; return x; }\n");
}

#[test]
fn c2y_language_probes() {
    let std = Std::C2y;
    for (name, src) in [
        ("_Countof", "int f(void) { int a[3]; return (int)_Countof(a); }\n"),
        ("if declaration", "int f(void) { if (int x = 1; x) return x; return 0; }\n"),
        ("switch declaration", "int f(int x) { switch (int y = x; y) { case 1: return 1; default: return 0; } }\n"),
        ("named loop", "int f(void) { int r = 0; a: for (int i = 0; i < 1; i++) { r++; break a; } return r; }\n"),
        ("_Generic type operand", "int f(void) { return _Generic(int, int: 1, default: 0); }\n"),
        ("delimited hex escape", "int f(void) { return \"\\x{41}\"[0]; }\n"),
        ("delimited UCN character", "int f(void) { return '\\u{41}'; }\n"),
        ("delimited UCN identifier", "int \\u{e9} = 1;\n"),
        ("bit-precise enum", "enum E : _BitInt(5) { A = 1 };\n"),
        ("expression static_assert", "int f(void) { return (static_assert(1), 5); }\n"),
        ("0o octal literal", "int f(void) { return 0o17; }\n"),
        ("0O octal literal", "int f(void) { return 0O17; }\n"),
        ("octal digit separator", "int f(void) { return 0o1'7; }\n"),
        ("complex increment", "int f(void) { double _Complex z = 1.0; z++; --z; return (int)__real__ z; }\n"),
        ("__auto_type", "int f(void) { __auto_type x = 1; return x; }\n"),
    ] {
        accepts(name, std, src);
    }
}
