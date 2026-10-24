//! Parser error recovery: a broken item is reported without hiding the
//! rest of the file, and input cut off at EOF is flagged as incomplete.

use fern::ast::{ItemKind, Program};
use fern::lexer::lex;
use fern::parser::{ParseOutput, parse, parse_recovering};

fn parse_src(src: &str) -> ParseOutput {
    match lex(src) {
        Ok(tokens) => parse_recovering(tokens),
        Err(diag) => ParseOutput {
            program: Program::single(Vec::new()),
            incomplete: diag.incomplete,
            diags: vec![diag],
        },
    }
}

fn fun_names(program: &Program) -> Vec<&str> {
    program
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Fun(fun) => Some(fun.name.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn function_parameters_require_type_annotations() {
    let out = parse_src("fun f(x) = x");
    assert!(out.diags.iter().any(|diag| diag.msg.contains("type annotation")), "{:?}", out.diags);
}

#[test]
fn functions_require_a_return_type() {
    let out = parse_src("fun f(x : Int) = x");
    assert!(out.diags.iter().any(|diag| diag.msg.contains("return type")), "{:?}", out.diags);
}

#[test]
fn broken_item_does_not_hide_the_next() {
    let out = parse_src("fun broken( = 1\nfun ok() : Int = 2\n");
    assert!(!out.diags.is_empty(), "expected a diagnostic");
    assert!(fun_names(&out.program).contains(&"ok"), "{:?}", out.program);
}

#[test]
fn error_inside_a_block_skips_nested_item_keywords() {
    let out = parse_src("fun broken() : Int = {\n  let x = ;\n  fun nested() : Int = 1\n}\nfun ok() : Int = 2\n");
    let names = fun_names(&out.program);
    assert!(names.contains(&"ok"), "{names:?}");
    assert!(!names.contains(&"nested"), "{names:?}");
}

#[test]
fn incomplete_block_is_flagged() {
    let out = parse_src("fun main() : Int = {\n");
    assert!(out.incomplete, "{:?}", out.diags);
    assert!(!out.diags.is_empty(), "expected a diagnostic");
}

#[test]
fn incomplete_block_comment_is_flagged() {
    let out = parse_src("/* never closed");
    assert!(out.incomplete, "{:?}", out.diags);
}

#[test]
fn unterminated_string_is_incomplete() {
    let out = parse_src("fun main() : String = \"oops");
    assert!(out.incomplete, "{:?}", out.diags);
    assert!(!out.diags.is_empty(), "expected a diagnostic");
}

#[test]
fn broken_extern_does_not_hide_the_next_item() {
    let out = parse_src("extern broken\nfun ok() : Int = 1\n");
    assert!(!out.diags.is_empty(), "expected a diagnostic");
    assert!(fun_names(&out.program).contains(&"ok"), "{:?}", out.program);
}

#[test]
fn parses_extern_items() {
    let out = parse_src("extern \"puts\" put : String -> Int\n");
    assert!(out.diags.is_empty(), "{:?}", out.diags);
    assert!(matches!(out.program.items[0].kind, ItemKind::Extern(_)));
}

#[test]
fn fail_fast_parse_still_errors() {
    let tokens = lex("fun broken( = 1").expect("lexes");
    assert!(parse(tokens).is_err());
}

#[test]
fn recovery_reports_multiple_items() {
    let out = parse_src("fun bad_one( = 1\nfun good_one() : Int = 10\nfun bad_two( = 2\nfun good_two() : Int = 20\n");
    assert!(out.diags.len() >= 2, "{:?}", out.diags);
    let names = fun_names(&out.program);
    assert!(names.contains(&"good_one"), "{names:?}");
    assert!(names.contains(&"good_two"), "{names:?}");
}

#[test]
fn unterminated_shell_quote_is_incomplete() {
    let out = parse_src("fun main() : Bool = $ echo \"oops\n");
    assert!(out.incomplete, "{:?}", out.diags);
}

#[test]
fn unterminated_shell_capture_is_incomplete() {
    let out = parse_src("fun main() : String = $(echo hi\n");
    assert!(out.incomplete, "{:?}", out.diags);
}

#[test]
fn unterminated_interpolation_is_incomplete() {
    let out = parse_src("fun main() : String = $ echo ${name\n");
    assert!(out.incomplete, "{:?}", out.diags);
}

#[test]
fn trailing_pipe_is_incomplete() {
    let out = parse_src("fun main() : String = $ echo hi |\n");
    assert!(out.incomplete, "{:?}", out.diags);
}

#[test]
fn trailing_operator_is_incomplete() {
    let out = parse_src("fun main() : Bool = $ true &&\n");
    assert!(out.incomplete, "{:?}", out.diags);
}
