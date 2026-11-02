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
    assert_eq!(std_from_code(0), None);
    assert_eq!(std_from_code(1), Some(Std::C89));
    assert_eq!(std_from_code(4), Some(Std::C17));
    assert_eq!(std_from_code(5), Some(Std::C23));
    assert_eq!(std_from_code(6), Some(Std::C2y));
    assert_eq!(std_from_code(7), None);
}

/// `-std=` reaches the parser's dialect gates, not just the
/// preprocessor: C89 rejects `_Static_assert` (added in C11), the
/// default and C23 accept it.
#[test]
fn std_gates_the_parser_dialect() {
    let src = "_Static_assert(1, \"ok\");\nint x;\n";
    let c89 = Options { std: Some(Std::C89), ..opts() };
    let out = compile(src, &c89);
    assert!(!out.success, "C89 must reject _Static_assert, got: {:?}", out.errors);
    let c23 = Options { std: Some(Std::C23), ..opts() };
    let out = compile(src, &c23);
    assert!(out.success, "C23 errors: {:?}", out.errors);
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
