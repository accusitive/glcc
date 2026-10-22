use crate::check::{CheckedExpr, CheckedExprKind};
use crate::lang::{LangOpts, Std};
use crate::layout::StructLayout;
use crate::lexer::{CharLiteral, CharPrefix, Constant, FloatLiteral, FloatSuffix, IntLiteral, IntSuffix, Interner, Lexer, Punctuator, Radix, StringPrefix, Token};
use crate::parse::{
    ArrayBound, AssignOp, BinaryOp, BlockItem, Designator, DirectBase, DirectSuffix, Expr, ExternalDeclaration, GnuAttributes, Initializer, Parser, PtrSize, Stmt, Ty,
    TypeQualifier, UnaryOp, resolve_type_specifiers,
};
use crate::sema::Sema;
use crate::span::{CompileError, Severity, Span, byte_range};

/// Parse driver for unit tests, mirroring any real driver: the test
/// owns the sema and the interner, builds the lexer, and hands the
/// tokens to the parser.
fn test_parser<'a>(src: &str, sema: &'a mut Sema, interner: &'a mut Interner) -> Parser<'a> {
    let tokens = Lexer::new(src, &mut *interner).lex_all();
    Parser::new(src, tokens, sema, interner)
}

/// Assert that `parser` recorded an error whose message contains
/// `expected`. The parser accumulates diagnostics and recovers instead of
/// aborting, so error tests inspect the collected diagnostics rather than
/// relying on a panic.
fn assert_parse_error(parser: &Parser, expected: &str) {
    let diagnostics = parser.diagnostics();
    assert!(
        diagnostics
            .iter()
            .any(|d| d.is_error() && d.message.contains(expected)),
        "expected an error containing {expected:?}, got: {:#?}",
        diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

/// Panic with the parser's first error, for helpers that must not
/// silently return a value from a source that failed to parse.
fn finish(parser: &Parser) {
    if let Some(err) = parser.diagnostics().iter().find(|d| d.is_error()) {
        panic!("{}", err.message);
    }
}

/// Assert that `diagnostics` contains an error whose message contains
/// `expected`.
fn assert_diag_contains(diagnostics: &[CompileError], expected: &str) {
    assert!(
        diagnostics.iter().any(|d| d.is_error() && d.message.contains(expected)),
        "expected an error containing {expected:?}, got: {:#?}",
        diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

/// The diagnostics from lexing `src`, without panicking.
fn lex_diagnostics(src: &str) -> Vec<CompileError> {
    let mut interner = Interner::new();
    Lexer::new(src, &mut interner).lex_all().diagnostics().to_vec()
}

/// The diagnostics from parsing `src` as one expression.
fn expr_diagnostics(src: &str) -> Vec<CompileError> {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(src, &mut sema, &mut interner);
    let _ = parser.parse_expression();
    parser.diagnostics()
}

/// The diagnostics from parsing `src` as one declarator.
fn declarator_diagnostics(src: &str) -> Vec<CompileError> {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(src, &mut sema, &mut interner);
    let _ = parser.parse_declarator();
    parser.diagnostics()
}

/// The diagnostics from parsing `src` as a whole translation unit.
fn tu_diagnostics(src: &str) -> Vec<CompileError> {
    tu_diagnostics_with_lang(src, LangOpts::default())
}

/// Translation-unit diagnostics under a specific dialect.
fn tu_diagnostics_with_lang(src: &str, lang: LangOpts) -> Vec<CompileError> {
    let mut sema = Sema::new();
    sema.lang = lang;
    let mut interner = Interner::new();
    let mut parser = test_parser(src, &mut sema, &mut interner);
    parser.parse_translation_unit();
    parser.diagnostics()
}

/// The diagnostics from lowering `src` (a declarator after `int`) to a
/// type, covering both parse-time and `resolve_type_specifiers` errors.
fn lower_to_ty_diagnostics(src: &str) -> Vec<CompileError> {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(&format!("int {src}"), &mut sema, &mut interner);
    let specs = parser.parse_declaration_specifiers();
    let d = parser.parse_declarator();
    let _ = parser.spec_to_ty(&specs, &d);
    parser.diagnostics()
}

/// Assert that resolving `src` as declaration specifiers reports an error
/// containing `expected` (from the parser or `resolve_type_specifiers`).
fn assert_base_error(src: &str, expected: &str) {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(src, &mut sema, &mut interner);
    let specs = parser.parse_declaration_specifiers();
    let diagnostics = parser.diagnostics();
    let resolved = resolve_type_specifiers(&specs.type_specifier, specs.span);
    let parse_hit = diagnostics.iter().any(|d| d.is_error() && d.message.contains(expected));
    let resolve_hit = resolved.as_ref().err().is_some_and(|e| e.message.contains(expected));
    assert!(
        parse_hit || resolve_hit,
        "expected an error containing {expected:?}, got parse {:#?} and resolve {:?}",
        diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>(),
        resolved.err().map(|e| e.message)
    );
}

/// Lex `src` and collect every punctuator spelling; panics on any
/// non-punctuator token so tests stay focused on punctuators.
fn lex_puncts(src: &str) -> Vec<&'static str> {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(src, &mut sema, &mut interner);
    let mut out = Vec::new();
    while let Some(t) = parser.tokens.advance() {
        match &t.val {
            Token::Punctuator(p) => out.push(p.as_str()),
            other => panic!("expected punctuator, got {other:?}"),
        }
    }
    out
}

#[test]
fn single_char_punctuators() {
    assert_eq!(
        lex_puncts("[ ] ( ) { } . & * + - ~ ! / % < > ^ | ? : ; = , #"),
        [
            "[", "]", "(", ")", "{", "}", ".", "&", "*", "+", "-", "~", "!", "/", "%", "<", ">", "^", "|", "?", ":", ";", "=", ",", "#"
        ]
    );
}

#[test]
fn two_char_punctuators() {
    assert_eq!(
        lex_puncts("-> ++ -- << >> <= >= == != && || += -= *= /= %= &= ^= |= ##"),
        [
            "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=", "##"
        ]
    );
}

#[test]
fn three_char_punctuators() {
    assert_eq!(lex_puncts("... <<= >>="), ["...", "<<=", ">>="]);
}

#[test]
fn digraphs_are_not_supported() {
    // Former digraph spellings must split into separate punctuators.
    assert_eq!(lex_puncts("<:"), ["<", ":"]);
    assert_eq!(lex_puncts(":>"), [":", ">"]);
    assert_eq!(lex_puncts("<%"), ["<", "%"]);
    assert_eq!(lex_puncts("%>"), ["%", ">"]);
    assert_eq!(lex_puncts("%:"), ["%", ":"]);
    assert_eq!(lex_puncts("%:%:"), ["%", ":", "%", ":"]);
}

#[test]
fn every_spelling_roundtrips() {
    let spellings = [
        "[", "]", "(", ")", "{", "}", ".", "&", "*", "+", "-", "~", "!", "/", "%", "<", ">", "^", "|", "?", ":", ";", "=", ",", /* "#", */ "->", "++", "--", "<<", ">>", "<=",
        ">=", "==", "!=", "&&", "||", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=", /* "##", */ "...", "<<=", ">>=",
    ];
    assert_eq!(spellings.len(), 46);
    for s in spellings {
        assert_eq!(lex_puncts(s), [s], "roundtrip failed for {s:?}");
    }
}

#[test]
fn maximal_munch() {
    assert_eq!(lex_puncts(".."), [".", "."]);
    assert_eq!(lex_puncts("...."), ["...", "."]);
    assert_eq!(lex_puncts("- >"), ["-", ">"]);
    assert_eq!(lex_puncts(">>="), [">>="]);
    assert_eq!(lex_puncts(">>"), [">>"]);
    assert_eq!(lex_puncts("> ="), [">", "="]);
    assert_eq!(lex_puncts("<<="), ["<<="]);
    assert_eq!(lex_puncts("=="), ["=="]);
    assert_eq!(lex_puncts("= ="), ["=", "="]);
}

#[test]
fn slash_eq_is_not_a_comment() {
    assert_eq!(lex_puncts("/="), ["/="]);
    assert_eq!(lex_puncts("/"), ["/"]);
}

#[test]
fn comments_are_skipped() {
    assert_eq!(lex_puncts("/* c */ + // c\n -"), ["+", "-"]);
    assert_eq!(lex_puncts("/**/+"), ["+"]);
    // A comment separates tokens that would otherwise merge.
    assert_eq!(lex_raw("1/**/2"), ["i:1", "i:2"]);
    // Comment markers inside literals are plain text.
    assert_eq!(lex_raw("\"/*\" '/' // tail"), ["s:\"/*\"", "c:'/'"]);
}

#[test]
fn block_comment_newlines_advance_lines() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("/* a\nb */ x", &mut sema, &mut interner);
    let x = parser.tokens.advance().unwrap().clone();
    match &x.val {
        Token::Identifier(s) => assert_eq!(parser.intern.resolve(*s).unwrap(), "x"),
        other => panic!("{other:?}"),
    }
    assert_eq!((x.loc.ln, x.loc.col), (2, 5));
    assert!(parser.tokens.is_empty());
}

#[test]
fn linemarker_is_skipped() {
    assert_eq!(lex_puncts("# 1 \"foo.h\"\n+"), ["+"]);
}

#[test]
fn non_linemarker_directives_are_skipped() {
    // `clang -E` passes `#pragma` through from system headers; any `#`
    // line without a line number is skipped the same way linemarkers are.
    assert_eq!(lex_puncts("#pragma clang diagnostic push\n+"), ["+"]);
    assert_eq!(lex_puncts("#pragma clang diagnostic ignored \"-Wpadded\"\n# 1 \"foo.h\"\n+"), ["+"]);
    assert_eq!(lex_puncts("#ident \"foo\"\n+"), ["+"]);
    // A directive without a trailing newline still ends the stream.
    assert_eq!(lex_puncts("+\n#pragma once"), ["+"]);
}

/// Lex `src` and collect int/float/char constants as `"i:<spelling>"` /
/// `"f:<spelling>"` / `"c:<spelling>"`; panics on any other token.
fn lex_lits(src: &str) -> Vec<String> {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(src, &mut sema, &mut interner);
    let mut out = Vec::new();
    while let Some(t) = parser.tokens.advance() {
        match &t.val {
            Token::Constant(c) => {
                let (tag, spelling) = match &c {
                    Constant::Int(i) => ("i", i.spelling),
                    Constant::Float(f) => ("f", f.spelling),
                    Constant::Char(c) => ("c", c.spelling),
                };
                out.push(format!("{tag}:{}", parser.intern.resolve(spelling).unwrap()))
            }
            other => panic!("expected constant, got {other:?}"),
        }
    }
    out
}

/// Lex `src` and dump every token's spelling, prefixing constants with
/// `i:`/`f:`/`c:` and string literals with `s:` so mixed
/// constant/punctuator streams can be asserted.
fn lex_raw(src: &str) -> Vec<String> {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(src, &mut sema, &mut interner);
    finish(&parser);
    let mut out = Vec::new();
    while let Some(t) = parser.tokens.advance() {
        match &t.val {
            Token::Constant(c) => {
                let (tag, spelling) = match &c {
                    Constant::Int(i) => ("i", i.spelling),
                    Constant::Float(f) => ("f", f.spelling),
                    Constant::Char(c) => ("c", c.spelling),
                };
                out.push(format!("{tag}:{}", parser.intern.resolve(spelling).unwrap()))
            }
            Token::StringLiteral(s) => out.push(format!("s:{}", parser.intern.resolve(s.spelling).unwrap())),
            Token::Punctuator(p) => out.push(p.as_str().to_string()),
            other => panic!("unexpected token {other:?}"),
        }
    }
    out
}

#[test]
fn decimal_and_octal_ints() {
    assert_eq!(lex_lits("0 1 42 0777"), ["i:0", "i:1", "i:42", "i:0777"]);
}

#[test]
fn int_suffixes() {
    assert_eq!(
        lex_lits("1u 1U 1l 1L 1ll 1LL 1ul 1UL 1ull 1ULL 1lu 1LU 1llu 1LLU"),
        [
            "i:1u", "i:1U", "i:1l", "i:1L", "i:1ll", "i:1LL", "i:1ul", "i:1UL", "i:1ull", "i:1ULL", "i:1lu", "i:1LU", "i:1llu", "i:1LLU"
        ]
    );
}

#[test]
fn hex_ints() {
    assert_eq!(lex_lits("0x0 0X1F 0xdeadBEEF 0x1u 0x10LL"), ["i:0x0", "i:0X1F", "i:0xdeadBEEF", "i:0x1u", "i:0x10LL"]);
}

#[test]
fn decimal_floats() {
    assert_eq!(
        lex_lits("1. 1.5 .5 1e3 1E-3 1.5e+2 1.f 1.5F .5l 1e3f"),
        ["f:1.", "f:1.5", "f:.5", "f:1e3", "f:1E-3", "f:1.5e+2", "f:1.f", "f:1.5F", "f:.5l", "f:1e3f"]
    );
}

#[test]
fn hex_floats() {
    assert_eq!(
        lex_lits("0x1p3 0x1P-3 0x1.8p3 0x1.p3 0xAp+2f"),
        ["f:0x1p3", "f:0x1P-3", "f:0x1.8p3", "f:0x1.p3", "f:0xAp+2f"]
    );
}

#[test]
fn dots_are_not_swallowed() {
    assert_eq!(lex_raw("1..."), ["i:1", "..."]);
    assert_eq!(lex_raw("0x1..."), ["i:0x1", "..."]);
    assert_eq!(lex_raw("1..2"), ["i:1", ".", "f:.2"]);
    assert_eq!(lex_raw(".5"), ["f:.5"]);
}

#[test]
fn c23_binary_literals_and_digit_separators() {
    assert_eq!(
        lex_raw("0b1010 0B11 1'000'000 0xDEAD'BEEF 0b1010'1010"),
        ["i:0b1010", "i:0B11", "i:1000000", "i:0xDEADBEEF", "i:0b10101010"]
    );
}

#[test]
fn literal_metadata() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("42 017 0xF 42ull 1.5 .5f 1e3l 0x1p-2", &mut sema, &mut interner);
    let mut lits = Vec::new();
    while let Some(t) = parser.tokens.advance() {
        match &t.val {
            Token::Constant(c) => lits.push(c.clone()),
            other => panic!("expected constant, got {other:?}"),
        }
    }
    let [a, b, c, d, e, f, g, h] = lits.as_slice() else {
        panic!("expected 8 constants, got {lits:?}")
    };
    let Constant::Int(i) = a else { panic!("{a:?}") };
    assert_eq!((i.radix, i.suffix), (Radix::Dec, IntSuffix::None));
    let Constant::Int(i) = b else { panic!("{b:?}") };
    assert_eq!((i.radix, i.suffix), (Radix::Oct, IntSuffix::None));
    let Constant::Int(i) = c else { panic!("{c:?}") };
    assert_eq!((i.radix, i.suffix), (Radix::Hex, IntSuffix::None));
    let Constant::Int(i) = d else { panic!("{d:?}") };
    assert_eq!((i.radix, i.suffix), (Radix::Dec, IntSuffix::ULL));
    let Constant::Float(x) = e else { panic!("{e:?}") };
    assert_eq!((x.radix, x.suffix), (Radix::Dec, FloatSuffix::None));
    let Constant::Float(x) = f else { panic!("{f:?}") };
    assert_eq!((x.radix, x.suffix), (Radix::Dec, FloatSuffix::F));
    let Constant::Float(x) = g else { panic!("{g:?}") };
    assert_eq!((x.radix, x.suffix), (Radix::Dec, FloatSuffix::L));
    let Constant::Float(x) = h else { panic!("{h:?}") };
    assert_eq!((x.radix, x.suffix), (Radix::Hex, FloatSuffix::None));
}

#[test]
fn msvc_int_suffixes() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let source = "1i8 2i16 3i32 4i64 5ui8 6ui16 7ui32 8ui64 9I64 10UI64 11i";
    let mut parser = test_parser(source, &mut sema, &mut interner);
    let mut lits = Vec::new();
    while let Some(t) = parser.tokens.advance() {
        match &t.val {
            Token::Constant(Constant::Int(i)) => lits.push((i.spelling, i.suffix, i.imag)),
            other => panic!("expected int constant, got {other:?}"),
        }
    }
    let suffixes = lits.iter().map(|(_, s, _)| *s).collect::<Vec<_>>();
    assert_eq!(
        suffixes,
        [
            IntSuffix::I8,
            IntSuffix::I16,
            IntSuffix::I32,
            IntSuffix::I64,
            IntSuffix::UI8,
            IntSuffix::UI16,
            IntSuffix::UI32,
            IntSuffix::UI64,
            IntSuffix::I64,
            IntSuffix::UI64,
            IntSuffix::None,
        ]
    );
    assert!(lits[10].2, "trailing `i` must stay the imaginary suffix");
}

#[test]
fn works_across_lines() {
    assert_eq!(lex_raw("1\\\n..."), ["i:1", "..."]);
    assert_eq!(lex_puncts(".\\\n."), [".", "."]);
    // A splice inside one pp-token is deleted, joining the halves.
    assert_eq!(lex_raw("1.\\\n5"), ["f:1.5"]);
    assert_eq!(lex_raw("0x\\\n10"), ["i:0x10"]);
    assert_eq!(lex_raw("1e\\\n+3"), ["f:1e+3"]);
    // Consecutive splices inside one token collapse the same way.
    assert_eq!(lex_raw("1\\\n\\\n2"), ["i:12"]);
}

#[test]
fn splice_inside_string() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("\"ab\\\ncd\"", &mut sema, &mut interner);
    let s = parser.tokens.advance().unwrap();
    match &s.val {
        Token::StringLiteral(s) => {
            assert_eq!(parser.intern.resolve(s.spelling).unwrap(), "\"abcd\"")
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn splice_updates_location() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("ab\\\ncd 1.\\\n5", &mut sema, &mut interner);
    let ident = parser.tokens.advance().unwrap().clone();
    match &ident.val {
        Token::Identifier(s) => assert_eq!(parser.intern.resolve(*s).unwrap(), "abcd"),
        other => panic!("{other:?}"),
    }
    assert_eq!((ident.loc.ln, ident.loc.col), (1, 0));
    // Tokens are lexed ahead of time: instead of a lexer cursor, the next
    // token in the stream carries the following location.
    assert_eq!(parser.tokens.position(), 1);
    let peeked = parser.tokens.peek().expect("float should be next");
    assert_eq!((peeked.loc.ln, peeked.loc.col), (2, 3));
    let float = parser.tokens.advance().unwrap().clone();
    match &float.val {
        Token::Constant(Constant::Float(x)) => {
            assert_eq!(parser.intern.resolve(x.spelling).unwrap(), "1.5");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!((float.loc.ln, float.loc.col), (2, 3));
    assert!(parser.tokens.is_empty());
    assert_eq!(parser.tokens.position(), 2);
}

/// Punctuator lookahead materializes chars past a splice; consuming them
/// afterwards must still land on the right line/column.
#[test]
fn splice_then_partial_consume() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("1+\\\nx", &mut sema, &mut interner);
    let i = parser.tokens.advance().unwrap().clone();
    match &i.val {
        Token::Constant(Constant::Int(v)) => {
            assert_eq!(parser.intern.resolve(v.spelling).unwrap(), "1")
        }
        other => panic!("{other:?}"),
    }
    let plus = parser.tokens.advance().unwrap().clone();
    assert!(matches!(&plus.val, Token::Punctuator(Punctuator::Plus)));
    let x = parser.tokens.advance().unwrap().clone();
    match &x.val {
        Token::Identifier(s) => assert_eq!(parser.intern.resolve(*s).unwrap(), "x"),
        other => panic!("{other:?}"),
    }
    assert_eq!((x.loc.ln, x.loc.col), (2, 0));
    assert!(parser.tokens.is_empty());
    assert_eq!(parser.tokens.position(), 3);
}

#[test]
fn plain_char_literals() {
    assert_eq!(lex_lits("'a'"), ["c:'a'"]);
    assert_eq!(lex_lits("'ab' 'abc'"), ["c:'ab'", "c:'abc'"]);
}

#[test]
fn char_escapes_stay_raw() {
    // Like string literals, the spelling keeps escapes unprocessed.
    assert_eq!(
        lex_lits(r#"'\n' '\t' '\0' '\x41' '\"' '\\' '\''"#),
        [r#"c:'\n'"#, r#"c:'\t'"#, r#"c:'\0'"#, r#"c:'\x41'"#, r#"c:'\"'"#, r#"c:'\\'"#, r#"c:'\''"#,]
    );
}

#[test]
fn char_prefixes() {
    assert_eq!(lex_lits("L'a' u'a' U'a' u8'a'"), ["c:L'a'", "c:u'a'", "c:U'a'", "c:u8'a'"]);
}

#[test]
fn char_prefix_metadata() {
    use crate::lexer::CharPrefix;
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("'a' L'a' u'a' U'a' u8'a'", &mut sema, &mut interner);
    let mut prefixes = Vec::new();
    while let Some(t) = parser.tokens.advance() {
        match &t.val {
            Token::Constant(Constant::Char(c)) => prefixes.push(c.prefix),
            other => panic!("expected char constant, got {other:?}"),
        }
    }
    assert_eq!(
        prefixes,
        [CharPrefix::None, CharPrefix::L, CharPrefix::SmallU, CharPrefix::BigU, CharPrefix::U8]
    );
}

#[test]
fn prefixed_char_needs_adjacency() {
    // `L 'a'` is an identifier plus a plain char, not one prefixed char.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("L 'a'", &mut sema, &mut interner);
    match &parser.tokens.advance().unwrap().val {
        Token::Identifier(s) => assert_eq!(parser.intern.resolve(*s).unwrap(), "L"),
        other => panic!("{other:?}"),
    }
    match &parser.tokens.advance().unwrap().val {
        Token::Constant(Constant::Char(c)) => {
            assert_eq!(parser.intern.resolve(c.spelling).unwrap(), "'a'")
        }
        other => panic!("{other:?}"),
    }
    // Plain identifiers starting with a prefix letter still lex as such.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("Long unsigned", &mut sema, &mut interner);
    match &parser.tokens.advance().unwrap().val {
        Token::Identifier(s) => assert_eq!(parser.intern.resolve(*s).unwrap(), "Long"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn chars_mix_with_other_tokens() {
    assert_eq!(lex_raw("'a'+'b'"), ["c:'a'", "+", "c:'b'"]);
}

#[test]
fn splice_inside_char() {
    // Phase-2 splice is deleted before lexing, like in strings.
    assert_eq!(lex_lits("'ab\\\ncd'"), ["c:'abcd'"]);
}

#[test]
fn plain_string_literals() {
    // Spellings keep quotes, mirroring char constants.
    assert_eq!(lex_raw(r#""""#), [r#"s:"""#]);
    assert_eq!(lex_raw(r#""abc""#), [r#"s:"abc""#]);
}

#[test]
fn string_escapes_stay_raw() {
    // Escapes stay unprocessed, and an escaped quote must not end the
    // literal early.
    assert_eq!(lex_raw(r#""a\"b""#), [r#"s:"a\"b""#]);
    assert_eq!(lex_raw(r#""\n\t\0\x41\\\'""#), [r#"s:"\n\t\0\x41\\\'""#]);
}

#[test]
fn string_prefixes() {
    assert_eq!(lex_raw(r#"L"a" u"a" U"a" u8"a""#), [r#"s:L"a""#, r#"s:u"a""#, r#"s:U"a""#, r#"s:u8"a""#]);
}

#[test]
fn string_prefix_metadata() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(r#""a" L"a" u"a" U"a" u8"a""#, &mut sema, &mut interner);
    let mut prefixes = Vec::new();
    while let Some(t) = parser.tokens.advance() {
        match &t.val {
            Token::StringLiteral(s) => prefixes.push(s.prefix),
            other => panic!("expected string literal, got {other:?}"),
        }
    }
    assert_eq!(prefixes, [StringPrefix::None, StringPrefix::L, StringPrefix::SmallU, StringPrefix::BigU, StringPrefix::U8,]);
}

#[test]
fn c23_u8_literals_are_unsigned_char() {
    let c23 = LangOpts {
        std: Std::C23,
        gnu: false,
        ms: false,
        clang: false,
    };
    let c17 = LangOpts {
        std: Std::C17,
        gnu: false,
        ms: false,
        clang: false,
    };
    assert!(tu_diagnostics_with_lang(
        "_Static_assert(_Generic(u8\"x\"[0], unsigned char: 1, default: 0), \"string\");\n\
         _Static_assert(_Generic(u8'x', unsigned char: 1, default: 0), \"char\");\n\
         _Static_assert(sizeof(u8\"x\") == 2, \"size\");",
        c23
    )
    .is_empty());
    let diags = tu_diagnostics_with_lang("int x = u8'a';", c17);
    assert!(diags.iter().any(|d| d.is_error() && d.message.contains("u8 character constants need C23")), "{diags:#?}");
}

#[test]
fn prefixed_string_needs_adjacency() {
    // `L "a"` is an identifier plus a plain string, not one prefixed string.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("L \"a\"", &mut sema, &mut interner);
    match &parser.tokens.advance().unwrap().val {
        Token::Identifier(s) => assert_eq!(parser.intern.resolve(*s).unwrap(), "L"),
        other => panic!("{other:?}"),
    }
    match &parser.tokens.advance().unwrap().val {
        Token::StringLiteral(s) => {
            assert_eq!(parser.intern.resolve(s.spelling).unwrap(), "\"a\"")
        }
        other => panic!("{other:?}"),
    }
    // Same for `u8`: spacing breaks the prefix, and `u8x` is an identifier.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("u8 \"a\" u8x", &mut sema, &mut interner);
    match &parser.tokens.advance().unwrap().val {
        Token::Identifier(s) => assert_eq!(parser.intern.resolve(*s).unwrap(), "u8"),
        other => panic!("{other:?}"),
    }
    match &parser.tokens.advance().unwrap().val {
        Token::StringLiteral(s) => {
            assert_eq!(parser.intern.resolve(s.spelling).unwrap(), "\"a\"")
        }
        other => panic!("{other:?}"),
    }
    match &parser.tokens.advance().unwrap().val {
        Token::Identifier(s) => assert_eq!(parser.intern.resolve(*s).unwrap(), "u8x"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn strings_mix_with_other_tokens() {
    assert_eq!(lex_raw(r#""a"+"b""#), [r#"s:"a""#, "+", r#"s:"b""#]);
}

#[test]
fn string_rejects_raw_newline() {
    assert_diag_contains(&lex_diagnostics("\"ab\ncd\""), "unterminated string literal");
}

/// Parse `src` as a single declarator; panics if tokens remain.
fn parse_one_declarator(src: &str) -> crate::parse::Declarator {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(src, &mut sema, &mut interner);
    let d = parser.parse_declarator();
    finish(&parser);
    assert!(parser.tokens.is_empty(), "trailing tokens in {src:?}");
    d
}

fn ident_name(d: &crate::parse::Declarator, parser: &Parser) -> Option<String> {
    match &d.direct.base {
        DirectBase::Ident(s) => Some(parser.intern.resolve(*s).unwrap().to_string()),
        _ => None,
    }
}

#[test]
fn dollar_in_identifiers_lexes() {
    // GNU `$` in identifiers (EDG/linker-generated names like
    // `vec$I...`): lexed as one identifier, leading or not.
    for (src, want) in [("vec$IN3AST8Function3ArgEEE", "vec$IN3AST8Function3ArgEEE"), ("$lead", "$lead"), ("a$b$c", "a$b$c")] {
        let mut sema = Sema::new();
        let mut interner = Interner::new();
        let mut parser = test_parser(src, &mut sema, &mut interner);
        match &parser.tokens.advance().unwrap().val {
            Token::Identifier(s) => assert_eq!(parser.intern.resolve(*s).unwrap(), want),
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn declarator_plain_identifier() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("a", &mut sema, &mut interner);
    let d = parser.parse_declarator();
    assert!(d.pointer.is_empty());
    assert_eq!(ident_name(&d, &parser).as_deref(), Some("a"));
    assert!(d.direct.suffixes.is_empty());
    assert!(parser.tokens.is_empty());
}

#[test]
fn declarator_pointers_with_qualifiers() {
    let d = parse_one_declarator("* const * volatile p");
    assert_eq!(d.pointer.len(), 2);
    assert!(matches!(d.pointer[0].qualifiers.as_slice(), [TypeQualifier::Const]));
    assert!(matches!(d.pointer[1].qualifiers.as_slice(), [TypeQualifier::Volatile]));
    assert!(d.direct.suffixes.is_empty());
}

#[test]
fn declarator_arrays() {
    let d = parse_one_declarator("a[]");
    let [DirectSuffix::Array(a)] = d.direct.suffixes.as_slice() else { panic!("{d:?}") };
    assert!(!a.is_static && a.bound.is_none());

    // `[*]` only parses in prototype scope: a bare declarator rejects
    // it (see `star_outside_prototype_panics`), so parse a prototype.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("(int a[*])", &mut sema, &mut interner);
    parser.tokens.advance();
    let suffix = parser.parse_function_suffix();
    assert!(parser.tokens.is_empty());
    let [param] = suffix.params.as_slice() else { panic!("{suffix:?}") };
    assert!(param.star);
}

#[test]
fn star_outside_prototype_panics() {
    assert_diag_contains(&declarator_diagnostics("a[*]"), "`[*]` outside function prototype scope");
}

#[test]
fn declarator_array_sizes() {
    let d = parse_one_declarator("a[3]");
    let [DirectSuffix::Array(a)] = d.direct.suffixes.as_slice() else { panic!("{d:?}") };
    assert!(!a.is_static && a.qualifiers.is_empty());
    assert_eq!(a.bound, Some(ArrayBound::Fixed(3)), "a constant bound folds to its count");

    // `[*p]` is a deref expression, not the `[*]` star marker, and does
    // not fold to a count — so it stays a VLA. `p` needs declaring
    // first, in block scope, since a runtime bound at file scope is a
    // variably modified type there.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int *p; a[*p]", &mut sema, &mut interner);
    parser.sema.push_scope();
    parser.parse_declaration();
    let d = parser.parse_declarator();
    parser.sema.pop_scope();
    assert!(parser.tokens.is_empty());
    let [DirectSuffix::Array(a)] = d.direct.suffixes.as_slice() else { panic!("{d:?}") };
    assert!(matches!(a.bound, Some(ArrayBound::Vla(_))), "an unfoldable bound is a VLA");

    let d = parse_one_declarator("a[static const 3]");
    let [DirectSuffix::Array(a)] = d.direct.suffixes.as_slice() else { panic!("{d:?}") };
    assert!(a.is_static);
    assert!(matches!(a.qualifiers.as_slice(), [TypeQualifier::Const]));
    assert_eq!(a.bound, Some(ArrayBound::Fixed(3)));

    // Bounds are full assignment expressions over declared names, and
    // fold when they can.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int n; a[n + 1]", &mut sema, &mut interner);
    parser.sema.push_scope();
    parser.parse_declaration();
    let d = parser.parse_declarator();
    parser.sema.pop_scope();
    assert!(parser.tokens.is_empty());
    let [DirectSuffix::Array(a)] = d.direct.suffixes.as_slice() else { panic!("{d:?}") };
    assert!(matches!(a.bound, Some(ArrayBound::Vla(_))));
}

#[test]
fn declarator_functions() {
    let d = parse_one_declarator("f()");
    let [DirectSuffix::Function(f)] = d.direct.suffixes.as_slice() else { panic!("{d:?}") };
    assert!(!f.has_params && !f.variadic);
}

#[test]
fn declarator_function_params() {
    let d = parse_one_declarator("f(void)");
    let [DirectSuffix::Function(f)] = d.direct.suffixes.as_slice() else { panic!("{d:?}") };
    assert!(f.has_params && !f.variadic);

    let d = parse_one_declarator("f(int, ...)");
    let [DirectSuffix::Function(f)] = d.direct.suffixes.as_slice() else { panic!("{d:?}") };
    assert!(f.has_params && f.variadic);
}

#[test]
fn function_parameter_names_and_types_parse() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("f(int a, char *s, ...)", &mut sema, &mut interner);
    let d = parser.parse_declarator();
    let [DirectSuffix::Function(f)] = d.direct.suffixes.as_slice() else { panic!("{d:?}") };
    assert!(f.has_params && f.variadic);
    assert_eq!(f.params.len(), 2);
    let name = |i: usize| parser.intern.resolve(f.params[i].name.expect("parameter name")).unwrap().to_string();
    assert_eq!(name(0), "a");
    assert_eq!(name(1), "s");
    assert_eq!(f.params[0].ty, Ty::Int);
    assert_eq!(
        f.params[1].ty,
        Ty::Pointer {
            qualifiers: vec![],
            size: PtrSize::Default,
            pointee: Box::new(Ty::Char),
        }
    );
}

#[test]
fn old_style_identifier_list_parses() {
    // `f(a, b)` carries an identifier-list: the names take their types
    // from a definition's declaration list, so the list reaches the
    // declarator as `has_params` plus the recorded `idents`.
    let d = parse_one_declarator("f(a, b)");
    let [DirectSuffix::Function(f)] = d.direct.suffixes.as_slice() else { panic!("{d:?}") };
    assert!(f.has_params && !f.variadic);
    assert!(f.params.is_empty());
    assert!(f.is_old_style);
    assert_eq!(f.idents.len(), 2);
}

#[test]
fn typedef_name_beats_identifier_list() {
    // C17 §6.7.6.3p11: an identifier that could be a typedef name is
    // taken as one, so `f(T)` prototypes one unnamed parameter instead
    // of reading as an old-style identifier-list.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("typedef int T; f(T)", &mut sema, &mut interner);
    parser.parse_declaration();
    let d = parser.parse_declarator();
    assert!(parser.tokens.is_empty());
    let [DirectSuffix::Function(f)] = d.direct.suffixes.as_slice() else { panic!("{d:?}") };
    assert!(f.has_params);
    assert_eq!(f.params.len(), 1);
    assert!(f.params[0].name.is_none());
    assert_eq!(f.params[0].ty, Ty::Int);
}

#[test]
fn declarator_paren_and_combined() {
    let d = parse_one_declarator("(*fp)(int)");
    let [DirectSuffix::Function(f)] = d.direct.suffixes.as_slice() else { panic!("{d:?}") };
    assert!(f.has_params);
    let DirectBase::Paren(inner) = &d.direct.base else { panic!("{d:?}") };
    assert_eq!(inner.pointer.len(), 1);
    assert!(matches!(inner.direct.base, DirectBase::Ident(_)));

    // Suffixes associate left to right: `a[3](void)` is (array of) function.
    let d = parse_one_declarator("a[3](void)");
    let [DirectSuffix::Array(_), DirectSuffix::Function(_)] = d.direct.suffixes.as_slice() else {
        panic!("{d:?}")
    };
}

#[test]
fn declarator_abstract() {
    // No identifier: allowed so contexts without a declarator keep parsing.
    let d = parse_one_declarator("");
    assert!(d.pointer.is_empty());
    assert!(matches!(d.direct.base, DirectBase::Abstract));
    assert!(d.direct.suffixes.is_empty());
}

/// Parse `int <src>` specs + declarator and lower to a `Ty`.
fn lower_to_ty(src: &str) -> Ty {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(&format!("int {src}"), &mut sema, &mut interner);
    let specs = parser.parse_declaration_specifiers();
    let d = parser.parse_declarator();
    assert!(parser.tokens.is_empty(), "trailing tokens in {src:?}");
    let ty = parser.spec_to_ty(&specs, &d);
    finish(&parser);
    ty
}

#[test]
fn spec_to_ty_pointers_nest_outside_in() {
    // `**p`: leftmost `*` is outermost.
    let ty = lower_to_ty("**p");
    let Ty::Pointer { pointee, .. } = &ty else { panic!("{ty:?}") };
    assert!(matches!(&**pointee, Ty::Pointer { .. }));
}

#[test]
fn spec_to_ty_keeps_pointer_qualifiers() {
    let ty = lower_to_ty("* const p");
    let Ty::Pointer { qualifiers, .. } = &ty else { panic!("{ty:?}") };
    assert!(matches!(qualifiers.as_slice(), [TypeQualifier::Const]));
}

#[test]
fn spec_to_ty_paren_flips_application() {
    // `(*f)(void)`: pointer to function ...
    let ty = lower_to_ty("(*f)(void)");
    assert!(matches!(ty, Ty::Pointer { pointee, .. } if matches!(*pointee, Ty::Func { .. })));

    // ... but `*f(void)`: function returning pointer.
    let ty = lower_to_ty("*f(void)");
    assert!(matches!(ty, Ty::Func { ret, .. } if matches!(*ret, Ty::Pointer { .. })));
}

#[test]
fn spec_to_ty_variadic() {
    let ty = lower_to_ty("f(int, ...)");
    assert!(matches!(ty, Ty::Func { variadic: true, .. }));
    let ty = lower_to_ty("f(void)");
    assert!(matches!(ty, Ty::Func { variadic: false, .. }));
}

#[test]
fn spec_to_ty_carries_parameter_types() {
    let ty = lower_to_ty("f(int, char *)");
    assert_eq!(
        ty,
        Ty::Func {
            ret: Box::new(Ty::Int),
            params: vec![
                Ty::Int,
                Ty::Pointer {
                    qualifiers: vec![],
                    size: PtrSize::Default,
                    pointee: Box::new(Ty::Char),
                },
            ],
            variadic: false,
            prototyped: true,
        }
    );
    // `f(void)` is prototyped with no parameters; `f()` is
    // unprototyped — same empty list, different prototype flag.
    let void_ty = lower_to_ty("f(void)");
    let empty_ty = lower_to_ty("f()");
    let Ty::Func {
        params: vp, prototyped: vprot, ..
    } = &void_ty
    else {
        panic!("{void_ty:?}")
    };
    let Ty::Func {
        params: ep, prototyped: eprot, ..
    } = &empty_ty
    else {
        panic!("{empty_ty:?}")
    };
    assert!(vp.is_empty() && *vprot);
    assert!(ep.is_empty() && !eprot);
    assert_ne!(void_ty, empty_ty);
}

#[test]
fn spec_to_ty_adjusts_function_parameter_to_pointer() {
    // C17 §6.7.6.3p8: a parameter declared with function type adjusts
    // to pointer to function.
    let ty = lower_to_ty("f(int g(void))");
    let Ty::Func { params, .. } = &ty else { panic!("{ty:?}") };
    let [param] = params.as_slice() else { panic!("{params:?}") };
    let Ty::Pointer { qualifiers, pointee, .. } = param else { panic!("{param:?}") };
    assert!(qualifiers.is_empty());
    assert!(matches!(**pointee, Ty::Func { .. }));
}

#[test]
fn array_parameter_decays_to_pointer() {
    // C17 §6.7.6.3p7: an array parameter adjusts to a pointer, whatever
    // the bound (even absent).
    for src in ["f(int a[10])", "f(int a[])", "f(int a[static 3])"] {
        let ty = lower_to_ty(src);
        let Ty::Func { params, .. } = &ty else { panic!("{ty:?} for {src:?}") };
        let [param] = params.as_slice() else { panic!("{params:?} for {src:?}") };
        let Ty::Pointer { pointee, .. } = param else {
            panic!("{param:?} for {src:?} should have decayed")
        };
        assert_eq!(**pointee, Ty::Int);
    }
    // Bracket qualifiers qualify the adjusted pointer.
    let ty = lower_to_ty("f(int a[const 3])");
    let Ty::Func { params, .. } = &ty else { panic!("{ty:?}") };
    let [Ty::Pointer { qualifiers, pointee, .. }] = params.as_slice() else {
        panic!("{params:?}")
    };
    assert!(matches!(qualifiers.as_slice(), [TypeQualifier::Const]));
    assert_eq!(**pointee, Ty::Int);
}

#[test]
fn void_parameter_must_be_alone() {
    assert_diag_contains(&lower_to_ty_diagnostics("f(void, int)"), "`void` must be the only parameter and unnamed");
}

#[test]
fn named_void_parameter_rejected() {
    assert_diag_contains(&lower_to_ty_diagnostics("f(void x)"), "`void` must be the only parameter and unnamed");
}

#[test]
fn storage_class_in_parameter_rejected() {
    assert_diag_contains(&lower_to_ty_diagnostics("f(static int)"), "storage class specifier in parameter");
}

#[test]
fn ellipsis_without_named_parameter_warns_not_fails() {
    // A bare `f(...)` is a GNU extension (clang warns): accepted as a
    // variadic function with no named parameters.
    let ty = lower_to_ty("f(...)");
    assert!(matches!(ty, Ty::Func { variadic: true, .. }));
}

/// Parse `src` as declaration specifiers and resolve them to one base type.
fn base_of(src: &str) -> Ty {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(src, &mut sema, &mut interner);
    let specs = parser.parse_declaration_specifiers();
    assert!(parser.tokens.is_empty(), "trailing tokens in {src:?}");
    match resolve_type_specifiers(&specs.type_specifier, specs.span) {
        Ok(ty) => ty,
        Err(err) => panic!("{}", err.message),
    }
}

#[test]
fn type_specifiers_resolve() {
    use Ty::*;
    let cases = [
        ("void", Void),
        ("char", Char),
        ("signed char", SignedChar),
        ("unsigned char", UnsignedChar),
        ("short", Short),
        ("short int", Short),
        ("signed short", Short),
        ("unsigned short", UnsignedShort),
        ("unsigned short int", UnsignedShort),
        ("int", Int),
        ("signed", Int),
        ("signed int", Int),
        ("unsigned", UnsignedInt),
        ("unsigned int", UnsignedInt),
        ("long", Long),
        ("long int", Long),
        ("signed long", Long),
        ("unsigned long", UnsignedLong),
        ("long long", LongLong),
        ("long long int", LongLong),
        ("signed long long", LongLong),
        ("unsigned long long", UnsignedLongLong),
        ("unsigned long long int", UnsignedLongLong),
        ("float", Float),
        ("double", Double),
        ("long double", LongDouble),
        // ISO C23 interchange floats alias the ABI-identical types.
        ("_Float32", Float),
        ("_Float64", Double),
        ("_Bool", Bool),
        ("float _Complex", Complex(Box::new(Float))),
        ("double _Complex", Complex(Box::new(Double))),
        ("long double _Complex", Complex(Box::new(LongDouble))),
        ("_Complex", Complex(Box::new(Double))),
        ("__complex__ double", Complex(Box::new(Double))),
        ("int __complex__", Complex(Box::new(Int))),
        ("__complex__ long", Complex(Box::new(Long))),
        ("unsigned _Complex", Complex(Box::new(UnsignedInt))),
        // Order doesn't matter.
        ("int unsigned long", UnsignedLong),
        ("long int", Long),
        ("double long", LongDouble),
        ("char signed", SignedChar),
        ("int short", Short),
    ];
    for (src, expected) in cases {
        assert_eq!(base_of(src), expected, "{src:?}");
    }
}

fn rejects(src: &str) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| base_of(src)));
    assert!(result.is_err(), "{src:?} should be rejected");
}

#[test]
fn type_specifiers_reject_invalid() {
    for src in [
        "",
        "short short",
        "int int",
        "float float",
        "double double",
        "char char",
        "long long long",
        "signed signed",
        "unsigned unsigned",
        "signed unsigned",
        "float int",
        "int float",
        "char int",
        "char short",
        "char long",
        "long char",
        "short long",
        "long short",
        "unsigned float",
        "signed double",
        "long float",
        "short float",
        "float double",
        "void int",
        "void void",
        "signed void",
        "_Bool int",
        "signed _Bool",
        "unsigned _Bool",
        "short _Bool",
        "long _Bool",
        "long long double",
        "double long long",
        "_Bool _Complex",
        "float _Complex _Complex",
    ] {
        rejects(src);
    }
}

use crate::sema::{Binding, BindingKind, DeclId, Linkage};

/// Parse one declaration that must introduce exactly one identifier —
/// the usual single-declarator case. Multi-declarator tests inspect the
/// whole list from `parse_declaration` directly.
fn parse_decl(parser: &mut Parser) -> DeclId {
    let ids = parser.parse_declaration();
    let [id] = ids.as_slice() else { panic!("expected exactly one declarator, got {ids:?}") };
    *id
}

fn declare_var(sema: &mut Sema, intern: &mut Interner, name: &str, ty: Ty) -> crate::sema::DeclId {
    try_declare_var(sema, intern, name, ty).unwrap()
}

fn try_declare_var(sema: &mut Sema, intern: &mut Interner, name: &str, ty: Ty) -> Result<crate::sema::DeclId, CompileError> {
    let symbol = intern.get_or_intern(name);
    sema.declare(
        symbol,
        Binding {
            name: Some(symbol),
            ty,
            kind: BindingKind::Var,
            linkage: Linkage::External,
            block_static: false,
            tls: false,
            defined: false,
            tentative: true,
            constructor: None,
            destructor: None,
            weak: false,
            alias: None,
            inline: false,
            align: None,
            naked: false,
            noinline: false,
            is_async: false,
            section: None,
            asm_label: None,
            span: Span::dummy(),
        },
    )
}

#[test]
fn sema_redeclare_same_type_merges() {
    let mut intern = Interner::new();
    let mut sema = Sema::new();
    let first = declare_var(&mut sema, &mut intern, "x", Ty::Int);
    let second = declare_var(&mut sema, &mut intern, "x", Ty::Int);
    assert_eq!(first, second);
    assert_eq!(sema.decls.len(), 1);
    let x = intern.get_or_intern("x");
    assert_eq!(sema.lookup_symbol(x), Some(first));
    sema.define(first);
    assert!(sema.decls[first.0 as usize].defined);
    // Redeclaring after definition with the same type still merges.
    let third = declare_var(&mut sema, &mut intern, "x", Ty::Int);
    assert_eq!(third, first);
}

#[test]
fn sema_redeclare_conflicting_type_reports() {
    let mut intern = Interner::new();
    let mut sema = Sema::new();
    try_declare_var(&mut sema, &mut intern, "x", Ty::Int).unwrap();
    let err = try_declare_var(&mut sema, &mut intern, "x", Ty::Float).unwrap_err();
    assert!(err.message.contains("conflicting redeclaration"), "{:?}", err.message);
}

#[test]
fn sema_inner_scope_shadows() {
    let mut intern = Interner::new();
    let mut sema = Sema::new();
    let outer = declare_var(&mut sema, &mut intern, "x", Ty::Int);
    sema.push_scope();
    // A block-scope object without `extern` has no linkage: it shadows
    // the file-scope entity instead of merging with it.
    let sym = intern.get_or_intern("x");
    let inner = sema
        .declare(
            sym,
            Binding {
                name: Some(sym),
                ty: Ty::Float,
                kind: BindingKind::Var,
                linkage: Linkage::NoLinkage,
                block_static: false,
                tls: false,
                defined: true,
                tentative: false,
                constructor: None,
                destructor: None,
                weak: false,
                alias: None,
                inline: false,
                align: None,
                naked: false,
                noinline: false,
                is_async: false,
                section: None,
                asm_label: None,
                span: Span::dummy(),
            },
        )
        .unwrap();
    assert_ne!(inner, outer);
    assert_eq!(sema.decls.len(), 2);
    let x = intern.get_or_intern("x");
    assert_eq!(sema.lookup_symbol(x), Some(inner));
    assert_eq!(sema.lookup_symbol_in_current_scope(x), Some(inner));
    sema.pop_scope();
    assert_eq!(sema.lookup_symbol(x), Some(outer));
}

#[test]
fn sema_lookup_miss() {
    let mut intern = Interner::new();
    let sema = Sema::new();
    let x = intern.get_or_intern("x");
    assert_eq!(sema.lookup_symbol(x), None);
    assert_eq!(sema.lookup_symbol_in_current_scope(x), None);
}

#[test]
fn parse_declaration_registers_and_defines() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x;", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    let x = parser.intern.get_or_intern("x");
    assert_eq!(parser.sema.lookup_symbol(x), Some(id));
    let binding = &parser.sema.decls[id.0 as usize];
    assert_eq!(binding.ty, Ty::Int);
    // File-scope `int x;` without initializer is tentative (C17 §6.9.2),
    // not yet a definition.
    assert!(!binding.defined);
    assert!(binding.tentative);
}

#[test]
fn parse_declaration_extern_stays_undefined() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("extern int x;", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    assert!(!parser.sema.decls[id.0 as usize].defined);
}

#[test]
fn typeof_specifier_takes_expression_or_type_name() {
    // `typeof(expr)` keeps the operand's type exactly — arrays and
    // qualifiers do not decay — and `typeof(type-name)` names it
    // directly. Both spell a whole type specifier.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "int i; const int ci; typeof (i) a; typeof (ci) b; typeof (\"xy\") c; typeof (int[5]) d;",
        &mut sema,
        &mut interner,
    );
    let _i = parse_decl(&mut parser);
    let _ci = parse_decl(&mut parser);
    let a = parse_decl(&mut parser);
    let b = parse_decl(&mut parser);
    let c = parse_decl(&mut parser);
    let d = parse_decl(&mut parser);
    assert_eq!(parser.sema.decls[a.0 as usize].ty, Ty::Int);
    assert_eq!(parser.sema.decls[b.0 as usize].ty, Ty::qualified(vec![TypeQualifier::Const], Ty::Int));
    assert_eq!(
        parser.sema.decls[c.0 as usize].ty,
        Ty::Array {
            elem: Box::new(Ty::Char),
            quals: Vec::new(),
            bound: Some(ArrayBound::Fixed(3)),
        }
    );
    assert_eq!(
        parser.sema.decls[d.0 as usize].ty,
        Ty::Array {
            elem: Box::new(Ty::Int),
            quals: Vec::new(),
            bound: Some(ArrayBound::Fixed(5)),
        }
    );
}

#[test]
fn builtin_types_compatible_p_folds_to_int() {
    // Top-level qualifiers and array bounds are ignored; the result is
    // an integer constant expression, so it works as an array bound.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "int a[__builtin_types_compatible_p (int, const int)]; int b[__builtin_types_compatible_p (int, long) + 1]; int c[__builtin_types_compatible_p (int[5], int[])]; int d[__builtin_types_compatible_p (char *, const char *)];",
        &mut sema,
        &mut interner,
    );
    let bound = |parser: &mut Parser| {
        let id = parse_decl(parser);
        match &parser.sema.decls[id.0 as usize].ty {
            Ty::Array {
                bound: Some(ArrayBound::Fixed(n)),
                ..
            } => *n,
            other => panic!("expected a fixed array bound, got {other:?}"),
        }
    };
    assert_eq!(bound(&mut parser), 1);
    assert_eq!(bound(&mut parser), 1);
    assert_eq!(bound(&mut parser), 1);
    assert_eq!(bound(&mut parser), 0);
}

#[test]
fn elvis_operator_shares_the_ternary_type_rules() {
    // `x ?: y` is a ternary whose then-arm is the condition: the same
    // conversions pick the result type.
    let checked = check_with(&["int a;", "int b;"], "a ?: b");
    assert_eq!(checked.ty, Ty::Int);
    assert!(matches!(checked.kind, CheckedExprKind::Elvis { .. }));
    let checked = check_with(&["int a;", "double d;"], "a ?: d");
    assert_eq!(checked.ty, Ty::Double);
    let checked = check_with(&["int *p;", "void *v;"], "p ?: v");
    assert!(matches!(checked.ty.unqualified(), Ty::Pointer { .. }));
    let msg = check_with_err(&["int *p;"], "1 ?: p");
    assert!(msg.contains("incompatible operand types in ternary"), "{msg:?}");
}

#[test]
fn builtin_choose_expr_skips_the_unchosen_arm() {
    // Only the chosen arm is checked — GCC leaves the other
    // unevaluated, so it need not even be valid.
    let checked = check_with(&[], "__builtin_choose_expr (1, 2, not_declared)");
    assert_eq!(checked.ty, Ty::Int);
    let checked = check_with(&["double d;"], "__builtin_choose_expr (0, 1, d)");
    assert_eq!(checked.ty, Ty::Double);
    let msg = check_with_err(&["int a;"], "__builtin_choose_expr (a, 1, 2)");
    assert!(msg.contains("condition is not a constant"), "{msg:?}");
}

#[test]
fn stmt_case_ranges_parse_and_check() {
    // GNU `case lo ... hi:` records both bounds; a plain `case` and
    // `default` keep `hi` empty.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "void f(int x) { switch (x) { case 1 ... 5: break; case 6: break; default: break; } }",
        &mut sema,
        &mut interner,
    );
    let (_, body) = parse_one_body("case range", &mut parser);
    let [BlockItem::Stmt(Stmt::Switch { body: switch_body, .. })] = body.block.items.as_slice() else {
        panic!("{body:?}")
    };
    let Stmt::Compound(inner) = &**switch_body else { panic!("{switch_body:?}") };
    let [range, single, default] = inner.items.as_slice() else { panic!("{inner:?}") };
    let BlockItem::Stmt(Stmt::Case { expr: Some(_), hi: Some(_), .. }) = range else {
        panic!("{range:?}")
    };
    let BlockItem::Stmt(Stmt::Case { expr: Some(_), hi: None, .. }) = single else {
        panic!("{single:?}")
    };
    let BlockItem::Stmt(Stmt::Case { expr: None, hi: None, .. }) = default else {
        panic!("{default:?}")
    };
}

#[test]
fn stmt_case_range_endpoint_duplicate_reports() {
    // Both range endpoints count as covered values for the duplicate
    // check, so `case 5:` after `case 1 ... 5:` is caught.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int x) { switch (x) { case 1 ... 5: break; case 5: break; } }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "duplicate `case` value in `switch`");
}

#[test]
fn member_attributes_may_follow_bitfield_width() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct s { unsigned a : 16; unsigned long b : 32 __attribute__((packed)); } x;", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    let Ty::Struct(tag) = parser.sema.decls[id.0 as usize].ty.unqualified() else {
        panic!("not a struct object")
    };
    let members = &parser.sema.structs[tag.0 as usize].members;
    assert_eq!(members.len(), 2);
    assert!(!members[0].packed);
    assert!(members[1].packed);
    let width = members[1].bitfield.as_ref().expect("bit-field width");
    assert_eq!(crate::eval::eval_const(width, parser.sema, &parser.intern), Some(crate::eval::Value::Int(32)));
}

#[test]
fn alignof_function_takes_its_alignment_attribute() {
    // A function designator's alignment is its `aligned` attribute (1
    // without one, like GCC); a function *type* aligns to 1 too.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "void f(void) __attribute__((aligned(256))); void g(void); __alignof__(f) + __alignof__(g) + __alignof__(int (void))",
        &mut sema,
        &mut interner,
    );
    parser.parse_declaration();
    parser.parse_declaration();
    let expr = parser.parse_expression();
    let checked = parser.sema.check_expr(&expr, &parser.intern).expect("check failed");
    assert_eq!(crate::eval::eval_const(&checked, parser.sema, &parser.intern), Some(crate::eval::Value::Int(258)));
}

#[test]
fn redeclared_restrict_params_merge() {
    // C17 §6.7.6.3p15: top-level parameter qualifiers don't affect
    // compatibility — `char **` vs `char **__restrict` merge into one
    // entity (tcc.h redeclares libc functions this way).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "extern float strtof(const char *, char **); extern float strtof(const char *, char **__restrict);",
        &mut sema,
        &mut interner,
    );
    let first = parse_decl(&mut parser);
    let second = parse_decl(&mut parser);
    assert_eq!(first, second);
}

// ---- struct / union declaration & definition ----

#[test]
fn struct_variable_with_forward_declared_tag() {
    // The pattern in test.c: the tag and the variable share a spelling
    // but live in different namespaces.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct test test;", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    let test = parser.intern.get_or_intern("test");
    let tag = parser.sema.lookup_tag(test).expect("tag declared");
    assert!(!parser.sema.structs[tag.0 as usize].complete);
    let binding = &parser.sema.decls[id.0 as usize];
    assert_eq!(binding.ty, Ty::Struct(tag));
    assert_eq!(binding.kind, BindingKind::Var);
    assert_eq!(parser.sema.lookup_symbol(test), Some(id));
    assert!(parser.tokens.is_empty());
}

#[test]
fn struct_forward_declarations_merge() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S; struct S; struct S *p;", &mut sema, &mut interner);
    assert!(parser.parse_declaration().is_empty());
    let s = parser.intern.get_or_intern("S");
    let first = parser.sema.lookup_tag(s).unwrap();
    assert!(parser.parse_declaration().is_empty());
    assert_eq!(parser.sema.lookup_tag(s), Some(first));
    assert_eq!(parser.sema.structs.len(), 1);
    let id = parse_decl(&mut parser);
    let Ty::Pointer { pointee, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected a pointer")
    };
    assert_eq!(**pointee, Ty::Struct(first));
}

#[test]
fn struct_definition_completes_forward_declaration() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S; struct S { int x; };", &mut sema, &mut interner);
    assert!(parser.parse_declaration().is_empty());
    let s = parser.intern.get_or_intern("S");
    let id = parser.sema.lookup_tag(s).unwrap();
    assert!(!parser.sema.structs[id.0 as usize].complete);
    assert!(parser.parse_declaration().is_empty());
    let def = &parser.sema.structs[id.0 as usize];
    assert!(def.complete);
    assert_eq!(def.members.len(), 1);
    let m = &def.members[0];
    assert_eq!(parser.intern.resolve(m.name.unwrap()).unwrap(), "x");
    assert_eq!(m.ty, Ty::Int);
}

#[test]
fn struct_can_reference_itself_in_body() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct Node { struct Node *next; };", &mut sema, &mut interner);
    assert!(parser.parse_declaration().is_empty());
    let node = parser.intern.get_or_intern("Node");
    let id = parser.sema.lookup_tag(node).unwrap();
    let def = &parser.sema.structs[id.0 as usize];
    assert!(def.complete);
    assert_eq!(def.members.len(), 1);
    let m = &def.members[0];
    assert_eq!(parser.intern.resolve(m.name.unwrap()).unwrap(), "next");
    let Ty::Pointer { pointee, .. } = &m.ty else { panic!("expected a pointer") };
    assert_eq!(**pointee, Ty::Struct(id));
}

#[test]
fn struct_embedding_itself_by_value_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { struct S v; int x; };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "struct member 'v' has incomplete type");
}

/// A `Sema` with one family (or the standard level) turned off.
fn gated_sema(lang: LangOpts) -> Sema {
    let mut sema = Sema::new();
    sema.lang = lang;
    sema
}

#[test]
fn no_gnu_extensions_reject_gnu_spellings() {
    for src in [
        "__attribute__((aligned(1))) int x;",
        "__typeof__(int) y;",
        "__asm__(\"nop\");",
        "__thread int y;",
        "__extension__ typedef int t;",
        "void *__restrict__ p;",
    ] {
        let mut sema = gated_sema(LangOpts {
            gnu: false,
            ..LangOpts::default()
        });
        let mut interner = Interner::new();
        let mut parser = test_parser(src, &mut sema, &mut interner);
        parser.parse_translation_unit();
        assert!(parser.has_errors(), "expected an error for {src:?}");
    }
}

#[test]
fn no_ms_extensions_filter_declspec() {
    let mut sema = gated_sema(LangOpts { ms: false, ..LangOpts::default() });
    let mut interner = Interner::new();
    let mut parser = test_parser("__declspec(noreturn) int f(void);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "declaration for parameter 'f'");
}

#[test]
fn forceinline_requires_ms_extensions() {
    let mut sema = gated_sema(LangOpts { ms: false, ..LangOpts::default() });
    let mut interner = Interner::new();
    let mut parser = test_parser("__forceinline int f(void) { return 1; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert!(parser.has_errors());
}

#[test]
fn ptr_size_requires_ms_extensions() {
    let mut sema = gated_sema(LangOpts { ms: false, ..LangOpts::default() });
    let mut interner = Interner::new();
    let mut parser = test_parser("int * __ptr32 p;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert!(parser.has_errors());
}

#[test]
fn ptr_size_attributes_set_pointer_width() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int * __ptr32 p; int * __ptr64 q; int *r;", &mut sema, &mut interner);
    let p = parse_decl(&mut parser);
    let q = parse_decl(&mut parser);
    let r = parse_decl(&mut parser);
    let ptr_ty = |size| Ty::Pointer {
        qualifiers: vec![],
        size,
        pointee: Box::new(Ty::Int),
    };
    assert_eq!(parser.sema.decls[p.0 as usize].ty, ptr_ty(PtrSize::Ptr32));
    // `__ptr64` is the ordinary 64-bit pointer type on this target.
    assert_eq!(parser.sema.decls[q.0 as usize].ty, ptr_ty(PtrSize::Default));
    assert_eq!(parser.sema.decls[r.0 as usize].ty, ptr_ty(PtrSize::Default));
    let size = |id: DeclId| parser.sema.type_size(&parser.sema.decls[id.0 as usize].ty, &parser.intern);
    assert_eq!(size(p), Some(4));
    assert_eq!(size(q), Some(8));
    assert_eq!(size(r), Some(8));
}

#[test]
fn unaligned_requires_ms_extensions() {
    let mut sema = gated_sema(LangOpts { ms: false, ..LangOpts::default() });
    let mut interner = Interner::new();
    let mut parser = test_parser("int __unaligned *p;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert!(parser.has_errors());
}

#[test]
fn unaligned_qualifier_is_erased() {
    // `UNALIGNED` in the Windows headers expands to `__unaligned`. Our
    // accesses are already unaligned-safe, so it leaves no trace on the
    // type, in either the base-type or the pointer-qualifier position.
    let mut sema = gated_sema(LangOpts { ms: true, ..LangOpts::default() });
    let mut interner = Interner::new();
    let mut parser = test_parser("unsigned short __unaligned *p; unsigned short * __unaligned q;", &mut sema, &mut interner);
    let p = parse_decl(&mut parser);
    let q = parse_decl(&mut parser);
    let plain = Ty::Pointer {
        qualifiers: vec![],
        size: PtrSize::Default,
        pointee: Box::new(Ty::UnsignedShort),
    };
    assert_eq!(parser.sema.decls[p.0 as usize].ty, plain);
    assert_eq!(parser.sema.decls[q.0 as usize].ty, plain);
}

#[test]
fn c23_typeof_unqual_strips_top_level_qualifiers() {
    let c23 = LangOpts {
        std: Std::C23,
        gnu: false,
        ms: false,
        clang: false,
    };
    let decl_ty = |src: &str| {
        let mut sema = gated_sema(c23);
        let mut interner = Interner::new();
        let mut parser = test_parser(src, &mut sema, &mut interner);
        let id = parse_decl(&mut parser);
        parser.sema.decls[id.0 as usize].ty.clone()
    };
    assert_eq!(decl_ty("typeof_unqual(const int) a;"), Ty::Int);
    assert_eq!(
        decl_ty("typeof_unqual(int *const) a;"),
        Ty::Pointer {
            qualifiers: Vec::new(),
            size: PtrSize::Default,
            pointee: Box::new(Ty::Int),
        }
    );
    assert_eq!(
        decl_ty("typeof_unqual(const int *) a;"),
        Ty::Pointer {
            qualifiers: Vec::new(),
            size: PtrSize::Default,
            pointee: Box::new(Ty::qualified(vec![TypeQualifier::Const], Ty::Int)),
        }
    );
    let Ty::Array { elem, quals, bound } = decl_ty("typeof_unqual(const int[3]) a;") else {
        panic!("expected an array");
    };
    assert_eq!(*elem, Ty::Int);
    assert!(quals.is_empty());
    assert_eq!(bound, Some(ArrayBound::Fixed(3)));
}

#[test]
fn c89_static_assert_keyword() {
    let mut sema = gated_sema(LangOpts {
        std: Std::C89,
        ..LangOpts::default()
    });
    let mut interner = Interner::new();
    let mut parser = test_parser("_Static_assert(1, \"x\");", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "no type specifier");
}

#[test]
fn c89_thread_local_standard_spelling() {
    let mut sema = gated_sema(LangOpts {
        std: Std::C89,
        ..LangOpts::default()
    });
    let mut interner = Interner::new();
    let mut parser = test_parser("_Thread_local int y;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "expected `;`, got Keyword(Int)");
}

#[test]
fn gnu_spellings_survive_at_standard_level() {
    // `-std=c99` gates C11/C23 keywords, not the GNU family.
    for src in ["__asm__(\"nop\");", "__attribute__((aligned(1))) int x;", "__thread long long y;"] {
        let mut sema = gated_sema(LangOpts {
            std: Std::C99,
            ..LangOpts::default()
        });
        let mut interner = Interner::new();
        let mut parser = test_parser(src, &mut sema, &mut interner);
        parser.parse_translation_unit();
    }
}

#[test]
fn struct_member_of_never_defined_type_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { struct T v; };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "struct member 'v' has incomplete type");
}

#[test]
fn mutually_recursive_structs_by_value_panic() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct A { struct B v; }; struct B { int x; };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "struct member 'v' has incomplete type");
}

#[test]
fn union_embedding_itself_by_value_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("union U { union U v; };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "union member 'v' has incomplete type");
}

#[test]
fn complete_member_types_keep_working() {
    // Pointers (including self-reference), by-value members of completed
    // types, and a trailing flexible array member are all legal C.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "struct P { int x; }; struct L { int v; struct L *n; struct P p; }; struct F { int n; char data[]; };",
        &mut sema,
        &mut interner,
    );
    parser.parse_translation_unit();
    for tag in ["L", "F"] {
        let sym = parser.intern.get_or_intern(tag);
        let id = parser.sema.lookup_tag(sym).unwrap();
        assert!(parser.sema.structs[id.0 as usize].complete);
    }
}

#[test]
fn struct_members_parse() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct P { int a, b; struct P *next; };", &mut sema, &mut interner);
    assert!(parser.parse_declaration().is_empty());
    let p = parser.intern.get_or_intern("P");
    let id = parser.sema.lookup_tag(p).unwrap();
    let def = &parser.sema.structs[id.0 as usize];
    let names: Vec<&str> = def.members.iter().map(|m| parser.intern.resolve(m.name.unwrap()).unwrap()).collect();
    assert_eq!(names, ["a", "b", "next"]);
    assert!(matches!(def.members[0].ty, Ty::Int));
    assert!(matches!(def.members[1].ty, Ty::Int));
}

#[test]
fn empty_struct_member_declarations_are_skipped() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct E { int a; ; int b; ; };", &mut sema, &mut interner);
    assert!(parser.parse_declaration().is_empty());
    let e = parser.intern.get_or_intern("E");
    let id = parser.sema.lookup_tag(e).unwrap();
    let def = &parser.sema.structs[id.0 as usize];
    let names: Vec<&str> = def.members.iter().map(|m| parser.intern.resolve(m.name.unwrap()).unwrap()).collect();
    assert_eq!(names, ["a", "b"]);
}

#[test]
fn tag_and_ordinary_namespaces_are_separate() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct x; int x;", &mut sema, &mut interner);
    assert!(parser.parse_declaration().is_empty());
    let x = parser.intern.get_or_intern("x");
    assert!(parser.sema.lookup_tag(x).is_some());
    let id = parse_decl(&mut parser);
    assert!(parser.sema.lookup_tag(x).is_some());
    assert_eq!(parser.sema.decls[id.0 as usize].ty, Ty::Int);

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x; struct x;", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    assert!(parser.parse_declaration().is_empty());
    assert_eq!(parser.sema.decls[id.0 as usize].ty, Ty::Int);
}

#[test]
fn member_list_tags_reach_the_enclosing_scope() {
    // gcc and clang both let a tag declared in a member list outlive it
    // (verified against both compilers), so glcc does too.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct O { struct I { int x; } i; };", &mut sema, &mut interner);
    assert!(parser.parse_declaration().is_empty());
    let i = parser.intern.get_or_intern("I");
    assert!(parser.sema.lookup_tag(i).is_some());
}

#[test]
fn anonymous_struct_is_complete_and_unscoped() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct { int x; } a; struct { int x; } b;", &mut sema, &mut interner);
    let a = parse_decl(&mut parser);
    let b = parse_decl(&mut parser);
    let Ty::Struct(id_a) = &parser.sema.decls[a.0 as usize].ty else {
        panic!("expected a struct")
    };
    let Ty::Struct(id_b) = &parser.sema.decls[b.0 as usize].ty else {
        panic!("expected a struct")
    };
    assert_ne!(id_a, id_b);
    let def = &parser.sema.structs[id_a.0 as usize];
    assert!(def.tag.is_none());
    assert!(!def.is_union);
    assert!(def.complete);
    assert_eq!(def.members.len(), 1);
}

#[test]
fn union_tag_parses() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("union U { int a; float b; } u;", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    let u = parser.intern.get_or_intern("U");
    let tag = parser.sema.lookup_tag(u).unwrap();
    assert!(parser.sema.structs[tag.0 as usize].is_union);
    assert_eq!(parser.sema.structs[tag.0 as usize].members.len(), 2);
    assert_eq!(parser.sema.decls[id.0 as usize].ty, Ty::Struct(tag));
}

#[test]
fn tag_only_declaration_declares_no_object() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S;", &mut sema, &mut interner);
    assert!(parser.parse_declaration().is_empty());
    assert_eq!(parser.sema.decls.len(), 0);
    assert_eq!(parser.sema.structs.len(), 1);
}

#[test]
fn struct_specifier_resolves_to_tag_type() {
    let Ty::Struct(_) = base_of("struct S") else { panic!("expected a struct type") };
}

#[test]
fn struct_specifier_cannot_mix_with_primitives() {
    assert_base_error("unsigned struct S", "type specifiers");
}

#[test]
fn struct_redefinition_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int x; }; struct S { int y; };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "redefinition of struct/union tag");
}

#[test]
fn struct_union_kind_mismatch_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S; union S;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "conflicting redeclaration");
}

#[test]
fn anonymous_struct_requires_a_name() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct { int x; };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "declaration does not declare anything");
}

#[test]
fn declaration_without_declarator_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "declaration does not declare anything");
}

#[test]
fn anonymous_struct_needs_a_body() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "anonymous struct/union without a body");
}

#[test]
fn storage_class_in_member_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { static int x; };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "storage class specifier in struct member");
}

#[test]
fn function_member_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int f(); };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "function member");
}

#[test]
fn anonymous_members_nest_and_promote_access() {
    // The <bits/sigcontext.h> shape: the anonymous union stays one
    // nameless member (so it overlaps in layout), while access to its
    // fields promotes through it.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "struct R { __extension__ union { int a; float b; }; int y; }; int f(void) { struct R r; r.a = 1; r.b = 2.0; return r.y; }",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
    let r = parser.intern.get_or_intern("R");
    let tag = parser.sema.lookup_tag(r).unwrap();
    let def = &parser.sema.structs[tag.0 as usize];
    assert_eq!(def.members.len(), 2);
    assert!(def.members[0].name.is_none());
    assert_eq!(parser.intern.resolve(def.members[1].name.unwrap()).unwrap(), "y");

    // Promoted access resolves to a member path through the
    // anonymous union.
    let decls = ["struct R { union { int a; float b; }; int y; };", "struct R r;"];
    let checked = check_with(&decls, "r.a");
    assert!(matches!(
        &checked.kind,
        CheckedExprKind::Member { path, .. } if path == &vec![0, 0]
    ));
    assert_eq!(checked.ty, Ty::Int);
    let checked = check_with(&decls, "r.b");
    assert!(matches!(
        &checked.kind,
        CheckedExprKind::Member { path, .. } if path == &vec![0, 1]
    ));
    assert_eq!(checked.ty, Ty::Float);
    let checked = check_with(&decls, "r.y");
    assert!(matches!(
        &checked.kind,
        CheckedExprKind::Member { path, .. } if path == &vec![1]
    ));
}

#[test]
fn member_without_declarator_declares_nothing() {
    // A named definition, a bare tag, or an enum without a declarator
    // declares no member (gcc warns and accepts); only anonymous
    // struct/unions splice.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct O { struct I { int x; }; struct Q; enum { A }; int y; }; int z = A;", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
    let o = parser.intern.get_or_intern("O");
    let tag = parser.sema.lookup_tag(o).unwrap();
    let def = &parser.sema.structs[tag.0 as usize];
    assert_eq!(def.members.len(), 1);
    assert_eq!(parser.intern.resolve(def.members[0].name.unwrap()).unwrap(), "y");
}

#[test]
fn aggregate_bit_field_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { struct { int x; } : 3; };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "bit-field has non-integer type");
}

#[test]
fn bit_field_members_parse() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int a : 3, b; };", &mut sema, &mut interner);
    assert!(parser.parse_declaration().is_empty());
    let s = parser.intern.get_or_intern("S");
    let tag = parser.sema.lookup_tag(s).unwrap();
    let def = &parser.sema.structs[tag.0 as usize];
    assert_eq!(def.members.len(), 2);
    assert!(def.members[0].bitfield.is_some());
    assert!(matches!(def.members[0].bitfield.as_ref().unwrap().kind, CheckedExprKind::Int(_)));
    assert!(def.members[1].bitfield.is_none());

    // Unnamed widths (`unsigned : 0;`) declare no member name.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { unsigned : 0; int x : 1; };", &mut sema, &mut interner);
    assert!(parser.parse_declaration().is_empty());
    let s = parser.intern.get_or_intern("S");
    let tag = parser.sema.lookup_tag(s).unwrap();
    let def = &parser.sema.structs[tag.0 as usize];
    assert_eq!(def.members.len(), 2);
    assert!(def.members[0].name.is_none());
    assert!(def.members[0].bitfield.is_some());
}

#[test]
fn bit_field_width_must_be_integer() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int a : 1.5; };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "bit-field width must have integer type");
}

#[test]
fn implicit_int_function_definition() {
    // A storage class may precede the implicit `int` (`static inline
    // fu (unsigned short data)` in alias-3 / pr79043); the identifier
    // opens the declarator and the missing specifier defaults to
    // `int`, C89-style.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("static inline fu (unsigned short data) { return data; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

// ---- enums ----

#[test]
fn enum_constants_are_ints() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("enum { A, B, C }; int x = B; int y = A + C;", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
    for (name, expected) in [("A", 0), ("B", 1), ("C", 2)] {
        let sym = parser.intern.get_or_intern(name);
        let id = parser.sema.lookup_symbol(sym).expect("enumerator declared");
        let binding = &parser.sema.decls[id.0 as usize];
        let BindingKind::EnumConst { value } = binding.kind else {
            panic!("expected an enumerator, got {:?}", binding.kind)
        };
        assert_eq!(value, expected);
        assert_eq!(binding.ty, Ty::Int);
    }
}

#[test]
fn enum_explicit_values_and_trailing_comma() {
    // Explicit values check as integer-typed (later ones may use
    // earlier ones); a trailing comma is allowed.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("enum { A = 3, B = A + 1, C, };", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert!(tu.external_declarations.is_empty());
    let def = &parser.sema.enums[0];
    assert!(def.complete);
    assert_eq!(def.enumerators.len(), 3);
    assert!(def.enumerators[0].value.is_some());
    assert!(def.enumerators[2].value.is_none());
}

#[test]
fn enum_tag_forward_and_define() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("enum E; enum E { A }; enum E e;", &mut sema, &mut interner);
    let e = parser.intern.get_or_intern("E");
    assert!(parser.parse_declaration().is_empty());
    let fwd = parser.sema.lookup_enum_tag(e).unwrap();
    assert!(!parser.sema.enums[fwd.0 as usize].complete);
    assert!(parser.parse_declaration().is_empty());
    assert!(parser.sema.enums[fwd.0 as usize].complete);
    let id = parse_decl(&mut parser);
    let Ty::Enum(tag) = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected an enum type")
    };
    assert_eq!(*tag, fwd);
    assert!(parser.tokens.is_empty());
}

#[test]
fn anonymous_enum_declares_no_object() {
    // The dwm.c shape: bare `enum { ... };` declares constants only,
    // contributing nothing to the translation unit.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("enum { A, B }; int a;", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn typedef_enum_is_usable_both_ways() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("typedef enum { A } T; T x;", &mut sema, &mut interner);
    let t = parse_decl(&mut parser);
    assert_eq!(parser.sema.decls[t.0 as usize].kind, BindingKind::Typedef);
    let x = parse_decl(&mut parser);
    let Ty::Enum(_) = &parser.sema.decls[x.0 as usize].ty else {
        panic!("expected an enum type")
    };
}

#[test]
fn enum_values_flow_into_integer_positions() {
    // Enumerators in bounds, designators, case labels, calls, and
    // enum-typed objects passed where integers are expected.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "enum { A, B, Last }; int arr[Last]; int d[] = { [B] = 5 }; enum E { C }; enum E e; int f(int x); int y = f(e); int g(int x) { switch (x) { case A: return B; default: return C; } }",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 6);
}

#[test]
fn inner_enum_definition_shadows_outer() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("enum E { A }; int f(void) { enum E { B }; return B; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    // The bare enum definition contributes nothing to the unit; only
    // the function does. The inner `enum E` shadows the outer tag.
    assert_eq!(tu.external_declarations.len(), 1);
    let e = parser.intern.get_or_intern("E");
    assert!(parser.sema.lookup_enum_tag(e).is_some());
}

#[test]
fn enum_struct_tag_conflict_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S; enum S { A };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "conflicting redeclaration");
}

#[test]
fn struct_enum_tag_conflict_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("enum E { A }; struct E;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "conflicting redeclaration");
}

#[test]
fn enum_redefinition_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("enum E { A }; enum E { B };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "redefinition of enum tag");
}

#[test]
fn enumerator_redeclaration_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("enum { A }; enum { A };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "conflicting redeclaration");
}

#[test]
fn enumerator_assignment_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("enum { A }; int f(void) { A = 1; return 0; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "assignment to a non-lvalue");
}

#[test]
fn enum_specifier_cannot_mix_with_primitives() {
    assert_base_error("unsigned enum E", "`enum` with other type specifiers");
}

#[test]
fn enum_fixed_underlying_type_sizes_and_types() {
    let c23 = LangOpts {
        std: Std::C23,
        gnu: false,
        ms: false,
        clang: false,
    };
    let src = "enum A : unsigned char { A0 = 1, A1 = 255 };\n\
               enum B : long long { B0 = -1, B1 = 0x100000000LL };\n\
               enum F : int;\n\
               enum F : int { F0 = 7 };\n\
               _Static_assert(sizeof(enum A) == 1, \"A size\");\n\
               _Static_assert(sizeof(enum B) == 8, \"B size\");\n\
               _Static_assert(_Generic((enum A)0, unsigned char: 1, default: 0), \"A generic\");\n\
               _Static_assert(_Generic((enum B)0, long long: 1, default: 0), \"B generic\");\n\
               _Static_assert(_Generic(A0, unsigned char: 1, default: 0), \"enumerator generic\");";
    let diagnostics = tu_diagnostics_with_lang(src, c23);
    assert!(diagnostics.iter().all(|d| !d.is_error()), "{diagnostics:#?}");

    let mut sema = gated_sema(c23);
    let mut interner = Interner::new();
    let mut parser = test_parser(src, &mut sema, &mut interner);
    parser.parse_translation_unit();
    let a = parser.intern.get_or_intern("A");
    let id = parser.sema.lookup_enum_tag(a).unwrap();
    assert_eq!(parser.sema.enums[id.0 as usize].repr, Ty::UnsignedChar);
    assert_eq!(parser.sema.enums[id.0 as usize].fixed, Some(Ty::UnsignedChar));
    let a0 = parser.intern.get_or_intern("A0");
    let binding = &parser.sema.decls[parser.sema.lookup_symbol(a0).unwrap().0 as usize];
    assert_eq!(binding.ty, Ty::UnsignedChar);
}

#[test]
fn enum_fixed_underlying_type_rejects_bad_values_and_types() {
    let c23 = LangOpts {
        std: Std::C23,
        gnu: false,
        ms: false,
        clang: false,
    };
    for src in [
        "enum A : unsigned char { A0 = 256 };",
        "enum A : _Bool { A0 = 2 };",
        "enum A : float { A0 };",
    ] {
        let diagnostics = tu_diagnostics_with_lang(src, c23);
        assert!(diagnostics.iter().any(|d| d.is_error()), "{src}: {diagnostics:#?}");
    }
}

#[test]
fn enum_fixed_underlying_type_redeclaration_checks() {
    let c23 = LangOpts {
        std: Std::C23,
        gnu: false,
        ms: false,
        clang: false,
    };
    for src in [
        "enum E : int; enum E { A };",
        "enum E : int; enum E : long { A };",
        "enum E { A }; enum E : int;",
    ] {
        let diagnostics = tu_diagnostics_with_lang(src, c23);
        assert!(diagnostics.iter().any(|d| d.is_error()), "{src}: {diagnostics:#?}");
    }
}

// ---- tag scoping (directly on Sema) ----

#[test]
fn inner_struct_definition_shadows_outer() {
    let mut intern = Interner::new();
    let mut sema = Sema::new();
    let s = intern.get_or_intern("S");
    let outer = sema.define_tag(s, false, Span::dummy()).unwrap();
    sema.complete_tag(outer, vec![]).unwrap();
    sema.push_scope();
    let inner = sema.define_tag(s, false, Span::dummy()).unwrap();
    assert_ne!(inner, outer);
    assert!(!sema.structs[inner.0 as usize].complete);
    sema.pop_scope();
    assert_eq!(sema.lookup_tag(s), Some(outer));
    assert!(sema.structs[outer.0 as usize].complete);
}

#[test]
fn forward_reference_reuses_visible_tag() {
    let mut intern = Interner::new();
    let mut sema = Sema::new();
    let s = intern.get_or_intern("S");
    let outer = sema.define_tag(s, false, Span::dummy()).unwrap();
    sema.push_scope();
    // a block-scope mention without a body refers to the visible tag
    assert_eq!(sema.reference_tag(s, false, Span::dummy()).unwrap(), outer);
    // no tag visible at all → implicit incomplete declaration in this scope
    let fresh = sema.reference_tag(intern.get_or_intern("Q"), false, Span::dummy()).unwrap();
    assert!(!sema.structs[fresh.0 as usize].complete);
    assert_eq!(sema.lookup_tag(intern.get_or_intern("Q")), Some(fresh));
    sema.pop_scope();
    assert!(sema.lookup_tag(intern.get_or_intern("Q")).is_none());
}

// ---- typedefs ----

#[test]
fn typedef_struct_anonymous_declares_alias() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("typedef struct { int i; } Arg; Arg a;", &mut sema, &mut interner);
    let arg = parse_decl(&mut parser);
    let arg_ty = parser.sema.decls[arg.0 as usize].ty.clone();
    assert_eq!(parser.sema.decls[arg.0 as usize].kind, BindingKind::Typedef);
    let Ty::Struct(id) = &arg_ty else { panic!("expected a struct type") };
    let def = &parser.sema.structs[id.0 as usize];
    assert!(def.tag.is_none());
    assert!(def.complete);

    let a = parse_decl(&mut parser);
    assert_eq!(parser.sema.decls[a.0 as usize].kind, BindingKind::Var);
    assert_eq!(parser.sema.decls[a.0 as usize].ty, arg_ty);

    let sym = parser.intern.get_or_intern("Arg");
    assert_eq!(parser.sema.lookup_symbol(sym), Some(arg));
}

#[test]
fn typedef_carries_primitive_type() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("typedef int T; T x;", &mut sema, &mut interner);
    let t = parse_decl(&mut parser);
    assert_eq!(parser.sema.decls[t.0 as usize].kind, BindingKind::Typedef);
    assert_eq!(parser.sema.decls[t.0 as usize].ty, Ty::Int);
    let x = parse_decl(&mut parser);
    assert_eq!(parser.sema.decls[x.0 as usize].kind, BindingKind::Var);
    assert_eq!(parser.sema.decls[x.0 as usize].ty, Ty::Int);
}

#[test]
fn typedef_tag_is_usable_both_ways() {
    // `typedef struct Monitor Monitor;` — same spelling as a tag and a
    // typedef name, no conflict.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("typedef struct Monitor Monitor; Monitor *m;", &mut sema, &mut interner);
    let td = parse_decl(&mut parser);
    assert_eq!(parser.sema.decls[td.0 as usize].kind, BindingKind::Typedef);
    let monitor = parser.intern.get_or_intern("Monitor");
    let tag = parser.sema.lookup_tag(monitor).expect("tag declared");
    let m = parse_decl(&mut parser);
    let Ty::Pointer { pointee, .. } = &parser.sema.decls[m.0 as usize].ty else {
        panic!("expected a pointer")
    };
    assert_eq!(**pointee, Ty::Struct(tag));
}

#[test]
fn typedef_redeclaration_conflict_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("typedef int T; typedef float T;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "conflicting redeclaration");
}

#[test]
fn typedef_and_object_conflict_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("typedef int T; int T;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "conflicting redeclaration");
}

// ---- GNU extensions (system headers) ----

#[test]
fn gnu_extension_marker_is_ignored() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("__extension__ typedef long long int greg_t;", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("__extension__ int x;", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { __extension__ int x; };", &mut sema, &mut interner);
    assert!(parser.parse_declaration().is_empty());
    let s = parser.intern.get_or_intern("S");
    let tag = parser.sema.lookup_tag(s).unwrap();
    assert_eq!(parser.sema.structs[tag.0 as usize].members.len(), 1);
}

#[test]
fn gnu_qualifier_spellings_parse() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int *__restrict p;", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    let Ty::Pointer { qualifiers, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected a pointer")
    };
    assert!(matches!(qualifiers.as_slice(), [TypeQualifier::Restrict]));

    // `__const` in specifiers qualifies the base type; `inline` counts
    // as a function specifier.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("__const int c;", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    assert_eq!(parser.sema.decls[id.0 as usize].ty, Ty::qualified(vec![TypeQualifier::Const], Ty::Int));

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("static __inline int f(void);", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    assert!(matches!(parser.sema.decls[id.0 as usize].ty, Ty::Func { .. }));

    // Leading GNU `inline` opens the declaration too (`__inline__ int
    // f();`, pervasive in system headers and EDG C output) — it must
    // not be misread as an implicit-int declarator named `__inline__`.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("__inline__ int f(void) { return 1; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn gnu_attribute_is_ignored() {
    // The <errno.h> shape: attributes after a function declarator.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "extern int *__errno_location (void) __attribute__ ((__nothrow__)) __attribute__ ((__const__));",
        &mut sema,
        &mut interner,
    );
    let id = parse_decl(&mut parser);
    let Ty::Func { ret, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected a function")
    };
    let Ty::Pointer { pointee, .. } = &**ret else { panic!("expected a pointer return") };
    assert_eq!(**pointee, Ty::Int);

    // Attributes on typedefs, members, parameters, and after a body.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("typedef int T __attribute__ ((__mode__ (__SI__)));", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    assert_eq!(parser.sema.decls[id.0 as usize].ty, Ty::Int);

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int x __attribute__ ((aligned (8))); };", &mut sema, &mut interner);
    assert!(parser.parse_declaration().is_empty());

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int x; } __attribute__ ((aligned (8))) v;", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    assert!(matches!(parser.sema.decls[id.0 as usize].ty, Ty::Struct(_)));

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int x __attribute__ ((unused)));", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    assert!(matches!(parser.sema.decls[id.0 as usize].ty, Ty::Func { .. }));

    // A prefix annotation on its own line, as in <X11/Xlib.h>.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("__attribute__ ((deprecated)) extern int old_fn(void);", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);

    // An annotation between the specifiers and the declarator, as in
    // sqlite (`static int __attribute__((noinline)) f(void);`), plus a
    // statement annotation (`__attribute__((fallthrough));`).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "static int __attribute__ ((noinline)) f(void); int g __attribute__ ((unused)); int h(int x) { switch (x) { case 1: __attribute__ ((fallthrough)); default: return 0; } }",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 3);

    // Annotations interleave specifiers, even before a union body with
    // a nested-brace initializer (the sqlite `aMutex` shape).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "struct M { int t; }; union U { struct M m; char s[4]; }; static __attribute__ ((aligned (8))) union U u[] = { { .m = { 1 } } };",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn constructor_attributes_land_on_binding() {
    // Suffix attributes ride the binding (bare = 65535, explicit
    // priority kept).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "void c(void) __attribute__((constructor(101))); void d(void) __attribute__((destructor)); void e(void) __attribute__((constructor));",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 3);
    for name in ["c", "d", "e"] {
        let sym = parser.intern.get_or_intern(name);
        let id = parser.sema.lookup_symbol(sym).expect("declared");
        let b = &parser.sema.decls[id.0 as usize];
        assert_eq!(b.kind, BindingKind::Func);
        match name {
            "c" => assert_eq!(b.constructor, Some(101)),
            "d" => assert_eq!(b.destructor, Some(65535)),
            _ => assert_eq!(b.constructor, Some(65535)),
        }
    }
}

#[test]
fn constructor_attribute_merges_from_prototype() {
    // EDG's shape: the extern prototype carries the attribute, the
    // definition is bare. First wins on merge.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("extern void s(void) __attribute__((__constructor__)); void s(void) {}", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
    let sym = parser.intern.get_or_intern("s");
    let id = parser.sema.lookup_symbol(sym).unwrap();
    let b = &parser.sema.decls[id.0 as usize];
    assert!(b.defined);
    assert_eq!(b.constructor, Some(65535));
}

#[test]
fn weak_attribute_lands_on_binding() {
    // Suffix, prefix, and `__weak__` spellings all mark the symbol;
    // plain declarations stay strong.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "void a(void) __attribute__((weak)); __attribute__((weak)) void b(void) {} __attribute__((__weak__)) int c;",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 3);
    for name in ["a", "b", "c"] {
        let sym = parser.intern.get_or_intern(name);
        let id = parser.sema.lookup_symbol(sym).expect("declared");
        assert!(parser.sema.decls[id.0 as usize].weak, "{name} should be weak");
    }
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void d(void);", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
    let sym = parser.intern.get_or_intern("d");
    let id = parser.sema.lookup_symbol(sym).unwrap();
    assert!(!parser.sema.decls[id.0 as usize].weak);
}

#[test]
fn alias_attribute_resolves_to_the_target_declaration() {
    // `alias("name")` denotes the named entity's storage: the binding
    // records the target's declaration, and the target's own binding is
    // untouched.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a[10]; extern int b[10] __attribute__((alias(\"a\")));", &mut sema, &mut interner);
    let a = parse_decl(&mut parser);
    let b = parse_decl(&mut parser);
    assert_eq!(parser.sema.decls[b.0 as usize].alias, Some(a));
    assert_eq!(parser.sema.decls[a.0 as usize].alias, None);
    // A later definition merges into the declaration the alias named,
    // so the target id stays stable.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a[10]; extern int b[10] __attribute__((alias(\"a\"))); int a[10] = {1};", &mut sema, &mut interner);
    let a = parse_decl(&mut parser);
    let b = parse_decl(&mut parser);
    let redefined = parse_decl(&mut parser);
    assert_eq!(redefined, a);
    assert!(parser.sema.decls[a.0 as usize].defined);
    assert_eq!(parser.sema.decls[b.0 as usize].alias, Some(a));
}

#[test]
fn alias_target_must_be_declared() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("extern int b __attribute__((alias(\"a\")));", &mut sema, &mut interner);
    parser.parse_declaration();
    assert_parse_error(&parser, "aliased to undefined symbol 'a'");
}

#[test]
fn alias_target_type_must_match() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a; extern long b __attribute__((alias(\"a\")));", &mut sema, &mut interner);
    parser.parse_declaration();
    parser.parse_declaration();
    assert_parse_error(&parser, "alias target has an incompatible type");
}

#[test]
fn weak_attribute_merges_from_prototype() {
    // EDG's shape: the extern prototype carries the attribute, the
    // definition is bare. Any declaration marks it (OR merge).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("extern __attribute__((__weak__)) void s(void); void s(void) {}", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
    let sym = parser.intern.get_or_intern("s");
    let id = parser.sema.lookup_symbol(sym).unwrap();
    let b = &parser.sema.decls[id.0 as usize];
    assert!(b.defined);
    assert!(b.weak);
}

#[test]
fn extern_inline_weak_only_in_cpp_mode() {
    // EDG lowers every C++ `inline` function to a GNU89 `extern
    // inline` definition, which needs vague linkage. Plain C keeps
    // strong-standalone semantics.
    for (cpp_mode, expect_weak) in [(false, false), (true, true)] {
        let mut sema = Sema::new();
        sema.weak_extern_inline = cpp_mode;
        let mut interner = Interner::new();
        let mut parser = test_parser("extern inline int f(void) { return 1; }", &mut sema, &mut interner);
        let tu = parser.parse_translation_unit();
        assert_eq!(tu.external_declarations.len(), 1);
        let sym = parser.intern.get_or_intern("f");
        let id = parser.sema.lookup_symbol(sym).unwrap();
        let b = &parser.sema.decls[id.0 as usize];
        assert!(b.defined);
        assert_eq!(b.weak, expect_weak);
    }
}

#[test]
fn plain_inline_definition_is_weak_in_cpp_mode() {
    // EDG's COMDAT inline definitions drop the `extern` on the
    // definition (`__inline__ ... { ... }`); they still need vague
    // linkage in C++ mode, while `static inline` stays internal.
    let mut sema = Sema::new();
    sema.weak_extern_inline = true;
    let mut interner = Interner::new();
    let mut parser = test_parser("inline int f(void) { return 1; } static inline int g(void) { return 2; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
    let f = parser.intern.get_or_intern("f");
    let f_id = parser.sema.lookup_symbol(f).unwrap();
    assert!(parser.sema.decls[f_id.0 as usize].weak, "plain inline is weak");
    let g = parser.intern.get_or_intern("g");
    let g_id = parser.sema.lookup_symbol(g).unwrap();
    assert!(!parser.sema.decls[g_id.0 as usize].weak, "static inline stays local");
}

#[test]
fn asm_labels_record_the_emitted_symbol() {
    // `__asm__("name")` on a declarator sets the assembler symbol; the
    // C spelling stays the declaration's lookup key. EDG emits these
    // for the C library overloads it mangles
    // (`_Z6strchrPci(...) __asm__("strchr")`).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("extern char *strchr_(const char *, int) __asm__(\"strchr\");", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let name = parser.intern.get_or_intern("strchr_");
    let id = parser.sema.lookup_symbol(name).unwrap();
    let label = parser.sema.decls[id.0 as usize].asm_label.expect("label recorded");
    assert_eq!(parser.intern.resolve(label), Some("strchr"));
}

#[test]
fn asm_label_from_a_prototype_rides_the_definition() {
    // First wins across declarations, like `section`: the extern
    // prototype usually carries the label while the definition omits
    // it.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("extern int f(int) __asm__(\"g\");\nint f(int x) { return x; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let name = parser.intern.get_or_intern("f");
    let id = parser.sema.lookup_symbol(name).unwrap();
    let label = parser.sema.decls[id.0 as usize].asm_label.expect("label merged");
    assert_eq!(parser.intern.resolve(label), Some("g"));
}

#[test]
fn inline_specifier_lands_on_binding() {
    // `inline` and its GNU spellings mark function bindings; plain
    // declarations stay unmarked.
    for src in [
        "inline int f(void) { return 1; }",
        "static inline int f(void) { return 1; }",
        "extern inline int f(void) { return 1; }",
        "static __inline__ int f(void) { return 1; }",
        "__inline int f(void) { return 1; }",
        "__forceinline int f(void) { return 1; }",
    ] {
        let mut sema = Sema::new();
        let mut interner = Interner::new();
        let mut parser = test_parser(src, &mut sema, &mut interner);
        let tu = parser.parse_translation_unit();
        assert_eq!(tu.external_declarations.len(), 1, "{src}");
        let sym = parser.intern.get_or_intern("f");
        let id = parser.sema.lookup_symbol(sym).unwrap();
        assert!(parser.sema.decls[id.0 as usize].inline, "{src} should be inline");
    }
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void) { return 1; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
    let sym = parser.intern.get_or_intern("f");
    let id = parser.sema.lookup_symbol(sym).unwrap();
    assert!(!parser.sema.decls[id.0 as usize].inline);
}

#[test]
fn inline_merges_from_prototype() {
    // Any declaration may carry `inline` (OR merge, like `weak`):
    // a marked prototype marks the later definition and vice versa.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("inline int f(void); int f(void) { return 1; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
    let sym = parser.intern.get_or_intern("f");
    let id = parser.sema.lookup_symbol(sym).unwrap();
    assert!(parser.sema.decls[id.0 as usize].inline);

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void); inline int f(void) { return 1; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
    let sym = parser.intern.get_or_intern("f");
    let id = parser.sema.lookup_symbol(sym).unwrap();
    assert!(parser.sema.decls[id.0 as usize].inline);
}

#[test]
fn gnu_asm_label_is_ignored() {
    // The <stdio.h> `__REDIRECT` shape: an asm rename after the
    // declarator, before the `;`.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "extern int scanf (const char *__restrict __format, ...) __asm__ (\"\" \"__isoc99_scanf\");",
        &mut sema,
        &mut interner,
    );
    let id = parse_decl(&mut parser);
    let Ty::Func { params, variadic, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected a function")
    };
    assert!(*variadic);
    assert_eq!(params.len(), 1);

    // A top-level assembler line declares nothing.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("__asm__ (\".symver foo, foo@VERS\"); int x;", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn gnu_predefined_int128_typedefs() {
    // GCC predefines `__uint128_t`/`__int128_t` with no declaration
    // (used bare by sqlite).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("__uint128_t r; __int128_t s;", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
    let r = parser.intern.get_or_intern("r");
    let id = parser.sema.lookup_symbol(r).unwrap();
    assert_eq!(parser.sema.decls[id.0 as usize].ty, Ty::UnsignedInt128);
}

#[test]
fn gnu_int128_parses() {
    // The <linux/types.h> shape: `__int128` combines only with a sign.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "typedef __signed__ __int128 __s128; typedef unsigned __int128 __u128; __s128 s; __u128 u;",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 4);
    let s = parser.intern.get_or_intern("s");
    let id = parser.sema.lookup_symbol(s).unwrap();
    assert_eq!(parser.sema.decls[id.0 as usize].ty, Ty::Int128);
    let u = parser.intern.get_or_intern("u");
    let id = parser.sema.lookup_symbol(u).unwrap();
    assert_eq!(parser.sema.decls[id.0 as usize].ty, Ty::UnsignedInt128);
    // Sizes, ranks, and conversions follow 16-byte integers.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let parser = test_parser("", &mut sema, &mut interner);
    assert_eq!(parser.sema.type_size(&Ty::Int128, &parser.intern), Some(16));
    assert_eq!(check_with(&["typedef __signed__ __int128 __s128;", "__s128 s;"], "s + 1").ty, Ty::Int128);
}

#[test]
fn int128_with_long_panics() {
    assert_base_error("long __int128", "`__int128` with other type specifiers");
}

#[test]
fn gnu_signed_spelling_parses() {
    // The <asm-generic/int-ll64.h> shape.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("typedef __signed__ char __s8;", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    assert_eq!(parser.sema.decls[id.0 as usize].ty, Ty::SignedChar);
    assert!(parser.tokens.is_empty());
}

#[test]
fn builtin_va_list_is_a_type() {
    // The <stdio.h> shape: `va_list` built on the builtin type.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "typedef __builtin_va_list __gnuc_va_list; typedef __gnuc_va_list va_list; int vfprintf(void *s, const char *f, va_list ap);",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 3);
    let va = parser.intern.get_or_intern("va_list");
    let td = parser.sema.lookup_symbol(va).unwrap();
    assert_eq!(parser.sema.decls[td.0 as usize].ty, Ty::BuiltinVaList);
}

#[test]
fn windows_builtin_va_list_is_char_pointer() {
    use crate::target::Target;
    // On MSVC targets clang models `__builtin_va_list` as `char *`, so
    // the clang <stdarg.h> typedef merges with <vadefs.h>'s `char *`
    // spelling instead of conflicting (both orders).
    for src in [
        "typedef __builtin_va_list va_list; typedef char* va_list;",
        "typedef char* va_list; typedef __builtin_va_list va_list;",
    ] {
        let mut sema = Sema::new();
        sema.target = Target::WINDOWS_X86_64;
        let mut interner = Interner::new();
        let mut parser = test_parser(src, &mut sema, &mut interner);
        let tu = parser.parse_translation_unit();
        assert_eq!(tu.external_declarations.len(), 2);
        let va = parser.intern.get_or_intern("va_list");
        let td = parser.sema.lookup_symbol(va).unwrap();
        assert!(
            matches!(parser.sema.decls[td.0 as usize].ty, Ty::Pointer { .. }),
            "expected char*, got {:?}",
            parser.sema.decls[td.0 as usize].ty
        );
    }
}

#[test]
fn msvc_spellings_and_annotations_parse() {
    // The UCRT header shape: `__int64` spellings, calling-convention
    // keywords, `__declspec`, and `__pragma(pack(...))` around it all.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "__pragma(pack(push, 8)) __declspec(noinline) unsigned __int64* __cdecl f(void); __int32 g(__int16 x); __pragma(pack(pop))",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
    let f = parser.intern.get_or_intern("f");
    let id = parser.sema.lookup_symbol(f).unwrap();
    assert_eq!(
        parser.sema.decls[id.0 as usize].ty,
        Ty::Func {
            ret: Box::new(Ty::Pointer {
                qualifiers: Vec::new(),
                size: PtrSize::Default,
                pointee: Box::new(Ty::UnsignedLongLong)
            }),
            params: vec![],
            variadic: false,
            prototyped: true,
        }
    );
}

#[test]
fn builtin_float128_is_a_type() {
    // The <bits/floatn.h> shape: an opaque quad-precision type that
    // names objects and parameters but takes part in nothing else yet.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("typedef __float128 _Float128; extern __float128 acosf128(__float128);", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
    let f = parser.intern.get_or_intern("_Float128");
    let td = parser.sema.lookup_symbol(f).unwrap();
    assert_eq!(parser.sema.decls[td.0 as usize].ty, Ty::Float128);

    // A bare builtin-typed parameter prototypes (it is not an
    // old-style identifier list).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(__float128);", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    let Ty::Func { params, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected a function")
    };
    assert_eq!(params.as_slice(), &[Ty::Float128]);
}

#[test]
fn builtin_type_cannot_combine() {
    assert_base_error("unsigned __float128", "builtin type with other type specifiers");
}

// ---- conversions and string initializers ----

#[test]
fn string_literal_decays_for_pointer_targets() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "char *p = \"abc\"; int strcmp(const char *, const char *); int f(void) { return strcmp(p, \"-v\"); }",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 3);
    let p = parser.intern.get_or_intern("p");
    let id = parser.sema.lookup_symbol(p).unwrap();
    let Ty::Pointer { pointee, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected a pointer")
    };
    assert_eq!(**pointee, Ty::Char);
}

#[test]
fn function_designator_decays_for_pointer_targets() {
    // Storing a function in a function-pointer object (the
    // `Button.func = spawn` shape from dwm).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "struct Arg { int i; }; void spawn(const struct Arg *); void (*fp)(const struct Arg *) = spawn;",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
}

#[test]
fn function_designator_is_a_condition() {
    // `if (handler[i])` from dwm: a function value decays to a scalar.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "void h(void); void (*handler[2])(void); int f(void) { if (handler[0]) return 1; if (h) return 2; return 0; }",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 3);
}

#[test]
fn pointer_qualifier_addition_in_assignment() {
    // The `execvp(file, argv)` shape from dwm's `spawn`: `char **`
    // converts to `char *const *`, and casts/subscripts feed it.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "void execvp(const char *, char *const *); void s(const void *v) { execvp(((char **)v)[0], (char **)v); }",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
}

#[test]
fn dropping_pointer_qualifiers_warns_only() {
    // GCC/clang diagnose the dropped `volatile` (-Wdiscarded-qualifiers)
    // but still compile the assignment; `-w` silences it.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("char *volatile *p; char **q; void f(void) { q = p; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 3);
}

#[test]
fn qualified_struct_assignment_accepted() {
    // Copying a struct ignores top-level qualifiers (mrustc
    // returns `const struct` by value): adding `const` to the
    // target, or copying from a qualified source, copies bytes.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "struct S { int x; }; const struct S f(struct S *p) { struct S t; t = (*(struct S *)p); return t; } const struct S g; const struct S h = g;",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 3);
}

#[test]
fn distinct_struct_assignment_rejected() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "struct A { int x; }; struct B { int x; }; void f(void) { struct A a; struct B b; a = b; }",
        &mut sema,
        &mut interner,
    );
    parser.parse_translation_unit();
    assert_parse_error(&parser, "incompatible types in assignment");
}

#[test]
fn ternary_decays_array_and_function_operands() {
    // The `ch.res_class ? ch.res_class : broken` shape from dwm: an
    // array and a pointer compare as pointers, as do two functions.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "char *p; char a[4]; void f(void); void g(void); void (*fp)(void); void h(int c) { p = c ? a : p; fp = c ? f : g; }",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 6);
}

#[test]
fn variadic_call_decays_and_promotes() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void h(const char *s, ...); void g(void) { h(\"a\", 1); }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
}

#[test]
fn char_array_from_string_literal() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "char s[] = \"abc\"; static char d[2] = \"0\"; char *tags[] = { \"1\", ((void *)0) };",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 3);
}

#[test]
fn string_literal_for_non_char_array_still_rejected() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x[2] = \"ab\";", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "incompatible types in assignment");
}

// ---- compound literals ----

#[test]
fn compound_assignment_keeps_the_lvalue_once() {
    // The checked form keeps the lvalue and the rhs apart, so the
    // lowering can evaluate the lvalue's address exactly once; the
    // operand types carry the conversions.
    let e = check_with(&["int x;", "long y;"], "x += y");
    let CheckedExprKind::CompoundAssign { op, lhs, rhs, op_ty, rhs_ty } = &e.kind else {
        panic!("{e:?}")
    };
    assert_eq!(*op, AssignOp::Add);
    assert!(matches!(lhs.kind, CheckedExprKind::Ident(_)));
    assert!(matches!(rhs.kind, CheckedExprKind::Ident(_)));
    assert_eq!(**op_ty, Ty::Long);
    assert_eq!(**rhs_ty, Ty::Long);
    // Plain assignment keeps the combined form's shape: just a store.
    let e = check_with(&["int x;"], "x = 1");
    assert!(matches!(e.kind, CheckedExprKind::Assign(AssignOp::Assign, ..)));
}

#[test]
fn compound_literal_array_decays() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int *p = (int[]){ 1, 2, 3 };", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    let Ty::Pointer { pointee, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected a pointer")
    };
    assert_eq!(**pointee, Ty::Int);
    assert!(parser.tokens.is_empty());
}

#[test]
fn compound_literal_shcmd_shape() {
    // The `SHCMD` shape from dwm's config: an array of `const char *`
    // with a null terminator, used as an initializer and as a call
    // argument.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "char *cmd; void f(const char **); const char **p = (const char *[]){ \"/bin/sh\", \"-c\", cmd, ((void *)0) }; void g(void) { f((const char *[]){ \"a\" }); }",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 4);
}

#[test]
fn compound_literal_struct_with_designator() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "struct P { int x, y; }; struct P p = (struct P){ 1, 2 }; struct A { int i; }; struct A a = (struct A){ .i = 1 };",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
}

#[test]
fn compound_literal_direct_sizeof_operand() {
    // `sizeof (type){...}` is `sizeof` of a compound literal, not a
    // parenthesized type-name (tcctest.c: "and as direct sizeof
    // argument (as unary expression)"): the `{` after the closing
    // paren selects the operand branch.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "int n = sizeof (struct { short i; short j; }){ 4, 5 }; int m = sizeof (int[2]){ 1, 2 }; int a = __alignof__ (int){ 1 };",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 3);
    let mut values = Vec::new();
    for decl in &tu.external_declarations {
        let ExternalDeclaration::Var(id) = decl else {
            panic!("expected a declaration, got {decl:?}")
        };
        let Initializer::Expr(e) = parser.sema.inits[id].clone() else {
            panic!("expected an expression initializer");
        };
        values.push(crate::eval::eval_const(&e, parser.sema, parser.intern));
    }
    use crate::eval::Value;
    assert_eq!(values, [Some(Value::Int(4)), Some(Value::Int(8)), Some(Value::Int(4))]);
    assert!(parser.tokens.is_empty());
}

#[test]
fn compound_literal_unknown_array_completes() {
    // `(int[]){1, 2, 3}` completes to `int[3]` (C17 §6.5.2.5p3), so
    // `sizeof` folds to 12 on LP64.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int n = sizeof((int[]){ 1, 2, 3 });", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    let Initializer::Expr(e) = parser.sema.inits[&id].clone() else {
        panic!("expected an expression initializer");
    };
    assert_eq!(crate::eval::eval_const(&e, parser.sema, parser.intern), Some(crate::eval::Value::Int(12)));
    assert!(parser.tokens.is_empty());
    // The braced lone-string form completes too (tcctest.c's
    // `(char []){ "tralala3" }`), collapsing to the bare shape.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int n = sizeof((char[]){ \"ab\" });", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    let Initializer::Expr(e) = parser.sema.inits[&id].clone() else {
        panic!("expected an expression initializer");
    };
    assert_eq!(crate::eval::eval_const(&e, parser.sema, parser.intern), Some(crate::eval::Value::Int(3)));
    assert!(parser.tokens.is_empty());
}

#[test]
fn compound_literal_postfix_attaches_to_literal() {
    // A compound literal is a postfix expression (C17 §6.5.2.5p1):
    // subscripting and member access apply to the object itself,
    // not to a parenthesized group around it.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "struct P { int x, y; }; int f(void) { return (int[]){ 0, 1, 2 }[2] + (struct P){ 1, 2 }.y; }",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
    assert!(parser.tokens.is_empty());
}

#[test]
fn compound_literal_is_assignable_lvalue() {
    // The literal names an object, so it accepts assignment like one.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { (int){ 3 } = 5; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
    assert!(parser.tokens.is_empty());
}

#[test]
fn compound_literal_function_type_rejected() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("typedef int F(void); void f(void) { (F){ 0 }; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "compound literal has function type");
}

#[test]
fn compound_literal_vla_rejected() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int n) { (int[n]){ 1 }; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "variable-sized object may not be initialized");
}

#[test]
fn compound_literal_uncompletable_rejected() {
    // A non-constant designator leaves `(int[])` without a size.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x; void f(void) { (int[]){ [x] = 1 }; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "compound literal of incomplete type");
}

#[test]
fn c23_compound_literal_storage_classes() {
    let c23 = LangOpts {
        std: Std::C23,
        gnu: false,
        ms: false,
        clang: false,
    };
    let ok = "int *f(void) { return &(static int){ 7 }; }\n\
              int *g(void) { return &(static thread_local int){ 7 }; }\n\
              int *h(void) { return &(thread_local int){ 7 }; }\n\
              int main(void) { return *f() + *g() + *h(); }";
    let diagnostics = tu_diagnostics_with_lang(ok, c23);
    assert!(diagnostics.iter().all(|d| !d.is_error()), "{diagnostics:#?}");

    let nonconstant = "int main(void) { int x = 5; int *p = &(static int){ x }; return *p; }";
    let diagnostics = tu_diagnostics_with_lang(nonconstant, c23);
    assert_diag_contains(&diagnostics, "initializer element is not constant");

    // The storage prefix is a C23 spelling; earlier dialects parse the
    // lead as an expression and reject it.
    let c17 = LangOpts {
        std: Std::C17,
        gnu: false,
        ms: false,
        clang: false,
    };
    let diagnostics = tu_diagnostics_with_lang("int main(void) { int *p = &(static int){ 1 }; return *p; }", c17);
    assert!(diagnostics.iter().any(|d| d.is_error()), "{diagnostics:#?}");
}

#[test]
fn builtin_va_start_end_declare_on_call() {
    // util.c's `die`: `va_start`/`va_end` expand to builtins declared
    // nowhere; calling them declares their prototypes.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "typedef __builtin_va_list va_list; void vfprintf(void *, const char *, va_list); void die(const char *fmt, ...) { va_list ap; __builtin_va_start(ap, fmt); vfprintf(((void *)0), fmt, ap); __builtin_va_end(ap); }",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    // The typedef, `vfprintf`, and `die` — the builtins declare
    // bindings but contribute no external declarations themselves.
    assert_eq!(tu.external_declarations.len(), 3);
    for name in ["__builtin_va_start", "__builtin_va_end"] {
        let sym = parser.intern.get_or_intern(name);
        let id = parser.sema.lookup_symbol(sym).expect("builtin declared");
        assert_eq!(parser.sema.decls[id.0 as usize].kind, BindingKind::Func);
    }
}

// ---- constant evaluation (see `crate::eval`) ----

/// Parse one `= ...` declaration and return its stored initializer.

fn checked_init(src: &str) -> Initializer {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(src, &mut sema, &mut interner);
    let ids = parser.parse_declaration();
    assert!(parser.tokens.is_empty(), "trailing tokens in {src:?}");
    parser.sema.inits.get(&ids[0]).expect("init stored").clone()
}

/// The folded value of the enumerator `name` in `src`.
fn enum_value_of(src: &str, name: &str) -> i128 {
    enum_value_of_with_lang(src, name, LangOpts::default())
}

fn enum_value_of_with_lang(src: &str, name: &str, lang: LangOpts) -> i128 {
    let mut sema = gated_sema(lang);
    let mut interner = Interner::new();
    let mut parser = test_parser(src, &mut sema, &mut interner);
    parser.parse_translation_unit();
    let sym = parser.intern.get_or_intern(name);
    let id = parser.sema.lookup_symbol(sym).expect("enumerator declared");
    match parser.sema.decls[id.0 as usize].kind {
        BindingKind::EnumConst { value } => value,
        ref kind => panic!("expected an enumerator, got {kind:?}"),
    }
}

#[test]
fn literal_types_follow_values() {
    // Hex `0xFFFFFFFF` is `unsigned int`, decimal `3000000000` is
    // `long`: when source and target agree no conversion is stored.
    let Initializer::Expr(e) = checked_init("unsigned u = 0xFFFFFFFF;") else {
        panic!("expected an expression init")
    };
    assert_eq!(e.ty, Ty::UnsignedInt);
    assert!(matches!(e.kind, CheckedExprKind::Int(_)));

    let Initializer::Expr(e) = checked_init("long l = 3000000000;") else {
        panic!("expected an expression init")
    };
    assert_eq!(e.ty, Ty::Long);
    assert!(matches!(e.kind, CheckedExprKind::Int(_)));

    // `i = 3000000000` converts `long` down to `int` (stored in the
    // checked tree), while a small literal needs no conversion at all.
    let checked = check_with(&["int i;"], "i = 3000000000");
    assert_eq!(checked.ty, Ty::Int);
    let CheckedExprKind::Assign(_, _, r) = &checked.kind else {
        panic!("expected an assignment, got {:?}", checked.kind)
    };
    assert_eq!(r.ty, Ty::Int);
    let CheckedExprKind::Convert { expr, .. } = &r.kind else {
        panic!("expected a conversion, got {:?}", r.kind)
    };
    assert_eq!(expr.ty, Ty::Long);

    let checked = check_with(&["int i;"], "i = 42");
    let CheckedExprKind::Assign(_, _, r) = &checked.kind else {
        panic!("expected an assignment, got {:?}", checked.kind)
    };
    assert!(matches!(&r.kind, CheckedExprKind::Int(_)));

    let Initializer::Expr(e) = checked_init("int z = 0;") else {
        panic!("expected an expression init")
    };
    assert_eq!(e.ty, Ty::Int);
    assert!(matches!(e.kind, CheckedExprKind::Int(_)));
}

#[test]
fn integer_constant_out_of_range_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x = 0x10000000000000000;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "integer constant out of range");
}

#[test]
fn enumerator_values_fold() {
    let src =
        "enum { A = 1 + 2 * 3, B = A << 2, C = -B, D = 10 / 3, E = 10 % 3, F = (2 > 1 ? 10 : 20), G = 'a', H = 010, I = 0x10, J = 1 << 4, K = ~0 & 0xFF, M = 0xFFFFFFFFu + 1u };";
    for (name, expected) in [
        ("A", 7),
        ("B", 28),
        ("C", -28),
        ("D", 3),
        ("E", 1),
        ("F", 10),
        ("G", 97),
        ("H", 8),
        ("I", 16),
        ("J", 16),
        ("K", 255),
        ("M", 0),
    ] {
        assert_eq!(enum_value_of(src, name), expected, "{name}");
    }
}

#[test]
fn enumerator_rejects_non_constant() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x; enum { A = x };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "not an integer constant expression");
}

#[test]
fn enumerator_rejects_division_by_zero() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("enum { Z = 1 / 0 };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "not an integer constant expression");
}

#[test]
fn enumerator_rejects_bad_shift() {
    // Shifting by the width or more is undefined.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("enum { S = 1 << 40 };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "not an integer constant expression");
}

#[test]
fn enumerator_takes_smallest_fitting_type() {
    // C23 §6.7.2.2p4 ladder: `1ll << 40` needs `long`,
    // `0xffffffffU` needs `unsigned int`.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("enum { A = 1ll << 40, B = 0xffffffffU };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    for (name, ty) in [("A", Ty::Long), ("B", Ty::UnsignedInt)] {
        let sym = parser.intern.get_or_intern(name);
        let id = parser.sema.lookup_symbol(sym).expect("enumerator declared");
        assert_eq!(parser.sema.decls[id.0 as usize].ty, ty);
    }
}

#[test]
fn enumerator_rejects_beyond_unsigned_long() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("enum { A = (unsigned __int128)1 << 100 };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "not representable");
}

#[test]
fn float_enumerator_rejected() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("enum { F = 1.5 };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "not an integer constant expression");
}

#[test]
fn folded_zero_is_null() {
    // Any folded zero — not just a `0` spelling — is a null pointer
    // constant, including `'\\0'`.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int *p = 1 - 1; int *q = '\\0';", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
    assert_eq!(check_with(&["int *p;"], "p == 1 - 1").ty, Ty::Int);
}

#[test]
fn int_literal_values() {
    use crate::eval::int_constant;
    use Ty::*;
    let mut intern = Interner::new();
    let mut lit = |spelling: &str, radix: Radix, suffix: IntSuffix| {
        int_constant(
            &IntLiteral {
                spelling: intern.get_or_intern(spelling),
                radix,
                suffix,
                imag: false,
            },
            &intern,
            8,
        )
    };
    assert_eq!(lit("42", Radix::Dec, IntSuffix::None), Some((Int, 42)));
    assert_eq!(lit("3000000000", Radix::Dec, IntSuffix::None), Some((Long, 3_000_000_000)));
    assert_eq!(lit("0xFFFFFFFF", Radix::Hex, IntSuffix::None), Some((UnsignedInt, 0xFFFF_FFFF)));
    assert_eq!(lit("010", Radix::Oct, IntSuffix::None), Some((Int, 8)));
    assert_eq!(lit("0b1010", Radix::Bin, IntSuffix::None), Some((Int, 10)));
    assert_eq!(lit("0B11111111", Radix::Bin, IntSuffix::None), Some((Int, 255)));
    assert_eq!(lit("1ull", Radix::Dec, IntSuffix::ULL), Some((UnsignedLongLong, 1)));
    assert_eq!(lit("0xFFFFFFFFFFFFFFFF", Radix::Hex, IntSuffix::None), Some((UnsignedLong, u64::MAX as i128)));
    assert_eq!(lit("0x10000000000000000", Radix::Hex, IntSuffix::None), None);
    assert_eq!(
        lit("9223372036854775807i64", Radix::Dec, IntSuffix::I64),
        Some((LongLong, i64::MAX as i128))
    );
    assert_eq!(
        lit("0xFFFFFFFFFFFFFFFFui64", Radix::Hex, IntSuffix::UI64),
        Some((UnsignedLongLong, u64::MAX as i128))
    );
    assert_eq!(lit("127i8", Radix::Dec, IntSuffix::I8), Some((SignedChar, 127)));
    assert_eq!(lit("128i8", Radix::Dec, IntSuffix::I8), None);
    assert_eq!(lit("0xffffui16", Radix::Hex, IntSuffix::UI16), Some((UnsignedShort, 0xffff)));
    assert_eq!(lit("2147483647i32", Radix::Dec, IntSuffix::I32), Some((Int, i32::MAX as i128)));
    assert_eq!(lit("0xffffffffui32", Radix::Hex, IntSuffix::UI32), Some((UnsignedInt, u32::MAX as i128)));
    // C23 `wb`/`uwb` pick the narrowest bit-precise type that fits.
    assert_eq!(lit("0wb", Radix::Dec, IntSuffix::Wb), Some((BitInt { width: 2, signed: true }, 0)));
    assert_eq!(lit("1wb", Radix::Dec, IntSuffix::Wb), Some((BitInt { width: 2, signed: true }, 1)));
    assert_eq!(lit("127wb", Radix::Dec, IntSuffix::Wb), Some((BitInt { width: 8, signed: true }, 127)));
    assert_eq!(lit("128wb", Radix::Dec, IntSuffix::Wb), Some((BitInt { width: 9, signed: true }, 128)));
    assert_eq!(lit("0uwb", Radix::Dec, IntSuffix::UWb), Some((BitInt { width: 1, signed: false }, 0)));
    assert_eq!(lit("255uwb", Radix::Dec, IntSuffix::UWb), Some((BitInt { width: 8, signed: false }, 255)));
    assert_eq!(lit("256uwb", Radix::Dec, IntSuffix::UWb), Some((BitInt { width: 9, signed: false }, 256)));
}

#[test]
fn int_literal_values_llp64() {
    use crate::eval::int_constant;
    use Ty::*;
    let mut intern = Interner::new();
    let mut lit = |spelling: &str, radix: Radix, suffix: IntSuffix| {
        int_constant(
            &IntLiteral {
                spelling: intern.get_or_intern(spelling),
                radix,
                suffix,
                imag: false,
            },
            &intern,
            4,
        )
    };
    assert_eq!(lit("3000000000", Radix::Dec, IntSuffix::None), Some((LongLong, 3_000_000_000)));
    assert_eq!(lit("5000000000L", Radix::Dec, IntSuffix::L), Some((LongLong, 5_000_000_000)));
    assert_eq!(lit("0xFFFFFFFF", Radix::Hex, IntSuffix::None), Some((UnsignedInt, 0xFFFF_FFFF)));
    assert_eq!(lit("0xFFFFFFFFFFFFFFFF", Radix::Hex, IntSuffix::None), Some((UnsignedLongLong, u64::MAX as i128)));
    assert_eq!(lit("0xFFFFFFFFUL", Radix::Hex, IntSuffix::UL), Some((UnsignedLong, 0xFFFF_FFFF)));
    assert_eq!(lit("0x10000000000UL", Radix::Hex, IntSuffix::UL), Some((UnsignedLongLong, 1 << 40)));
}

#[test]
fn static_assert_checks_and_disappears() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let src = "int x;\n_Static_assert(sizeof(int) == 4, \"int is 4\");\nint f(void) { _Static_assert(1 == 1, \"one\"); return x; }";
    let mut parser = test_parser(src, &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
    let f = parser.intern.get_or_intern("f");
    let fid = parser.sema.lookup_symbol(f).expect("function declared");
    let body = parser.sema.bodies[&fid].clone();
    let [BlockItem::Stmt(Stmt::Empty { .. }), BlockItem::Stmt(_)] = body.block.items.as_slice() else {
        panic!("expected the static assert to vanish, got {:?}", body.block.items)
    };
}

#[test]
fn c23_bare_static_assert() {
    let c23 = LangOpts {
        std: Std::C23,
        gnu: false,
        ms: false,
        clang: false,
    };
    assert!(tu_diagnostics_with_lang("static_assert(1);", c23).is_empty());
    assert!(tu_diagnostics_with_lang("void f(void) { static_assert(1, \"ok\"); }", c23).is_empty());
    let diags = tu_diagnostics_with_lang("static_assert(0, \"boom\");", c23);
    assert!(
        diags
            .iter()
            .any(|d| d.is_error() && d.message.contains("static assertion failed: \"boom\"")),
        "missing failure: {diags:#?}"
    );
    assert!(!tu_diagnostics("static_assert(1);").is_empty());
}

#[test]
fn static_assert_failure_reports_message() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { _Static_assert(0, \"boom\"); }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "static assertion failed: \"boom\"");
}

#[test]
fn static_assert_rejects_non_constant() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int n) { _Static_assert(n, \"var\"); }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "needs an integer constant expression");
}

#[test]
fn char_literal_values() {
    use crate::eval::char_constant;
    let mut intern = Interner::new();
    let mut lit = |spelling: &str, prefix: CharPrefix| {
        char_constant(
            &CharLiteral {
                spelling: intern.get_or_intern(spelling),
                prefix,
            },
            &intern,
        )
    };
    assert_eq!(lit("'a'", CharPrefix::None), Some(97));
    assert_eq!(lit("'\\n'", CharPrefix::None), Some(10));
    assert_eq!(lit("'\\x41'", CharPrefix::None), Some(65));
    assert_eq!(lit("'\\0'", CharPrefix::None), Some(0));
    assert_eq!(lit("'ab'", CharPrefix::None), Some(0x6162));
    assert_eq!(lit("L'a'", CharPrefix::L), Some(97));
    assert_eq!(lit("L'\\400'", CharPrefix::L), Some(256));
}

#[test]
fn float_literal_values() {
    use crate::eval::{FloatValue, float_constant};
    let mut intern = Interner::new();
    let mut lit = |spelling: &str, radix: Radix, suffix: FloatSuffix| {
        float_constant(
            &FloatLiteral {
                spelling: intern.get_or_intern(spelling),
                radix,
                suffix,
                imag: false,
            },
            &intern,
        )
    };
    // Doubles keep their exact `f64` values.
    assert_eq!(lit("1.5", Radix::Dec, FloatSuffix::None).map(|v| v.to_f64()), Some(1.5));
    assert_eq!(lit(".5", Radix::Dec, FloatSuffix::None).map(|v| v.to_f64()), Some(0.5));
    assert_eq!(lit("1e3", Radix::Dec, FloatSuffix::None).map(|v| v.to_f64()), Some(1000.0));
    assert_eq!(lit("1E-3", Radix::Dec, FloatSuffix::None).map(|v| v.to_f64()), Some(0.001));
    assert_eq!(lit("0x1p3", Radix::Hex, FloatSuffix::None).map(|v| v.to_f64()), Some(8.0));
    assert_eq!(lit("0x1.8p3", Radix::Hex, FloatSuffix::None).map(|v| v.to_f64()), Some(12.0));
    assert_eq!(lit("0x1P-3", Radix::Hex, FloatSuffix::None).map(|v| v.to_f64()), Some(0.125));
    // Suffixes select the precision: `F` parses in binary32, `L` in
    // 80-bit extended.
    assert!(matches!(lit("1.5F", Radix::Dec, FloatSuffix::F), Some(FloatValue::F32(_))));
    assert_eq!(lit("1.5F", Radix::Dec, FloatSuffix::F).map(|v| v.to_f32()), Some(1.5f32));
    assert!(matches!(lit("1.5L", Radix::Dec, FloatSuffix::L), Some(FloatValue::F80(_))));
    assert_eq!(lit("1.5L", Radix::Dec, FloatSuffix::L).map(|v| v.to_f64()), Some(1.5));
    assert_eq!(lit("0x1.8p3L", Radix::Hex, FloatSuffix::L).map(|v| v.to_f64()), Some(12.0));
    // Extended range: finite in f80, infinite in f64.
    let big_l = lit("1e4932", Radix::Dec, FloatSuffix::L).expect("parses");
    assert_eq!(big_l.to_f64(), f64::INFINITY);
    assert_ne!(big_l.to_quad_bits() >> 112 & 0x7fff, 0x7fff);
    let big_d = lit("1e4932", Radix::Dec, FloatSuffix::None).expect("parses");
    assert_eq!(big_d.to_f64(), f64::INFINITY);
    assert_eq!(big_d.to_quad_bits() >> 112 & 0x7fff, 0x7fff);
    // Extended precision: `0.1L` keeps 64 mantissa bits (the f64
    // rounding of the true value still agrees, but the f80 holds more).
    let tenth = lit("0.1L", Radix::Dec, FloatSuffix::L).expect("parses");
    assert_eq!(tenth.to_f64(), 0.1);
    let tenth_d = lit("0.1", Radix::Dec, FloatSuffix::None).expect("parses");
    assert_ne!(tenth.to_quad_bits(), tenth_d.to_quad_bits());
}

#[test]
fn long_double_f80_bytes() {
    // `to_f80_bytes` is the x86 C ABI form of `long double`: the
    // 10-byte little-endian extended value (mantissa, then
    // sign+exponent) plus 6 zero pad bytes. Pinned against GCC's
    // encoding (`00 04 00 00 00 00 00 80 34 40` for 2^53+1).
    use crate::eval::float_constant;
    let mut intern = Interner::new();
    let mut lit = |spelling: &str, radix: Radix, suffix: FloatSuffix| {
        float_constant(
            &FloatLiteral {
                spelling: intern.get_or_intern(spelling),
                radix,
                suffix,
                imag: false,
            },
            &intern,
        )
        .expect("parses")
        .to_f80_bytes()
    };
    assert_eq!(
        lit("9007199254740993.0L", Radix::Dec, FloatSuffix::L),
        [0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    );
    // 1.5 = 1.1b * 2^0: mantissa 0xC000000000000000, exponent 0x3FFF.
    assert_eq!(
        lit("1.5L", Radix::Dec, FloatSuffix::L),
        [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC0, 0xFF, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    );
    // Zeros keep their sign; infinities use the all-ones exponent with
    // the integer bit set (never a pseudo-infinity).
    assert_eq!(
        lit("-0.0L", Radix::Dec, FloatSuffix::L),
        [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    );
    assert_eq!(
        lit("1e4933L", Radix::Dec, FloatSuffix::L),
        [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xFF, 0x7F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    );
    // A `double` widens exactly (single rounding at parse, none here):
    // 0.5 is 1.0 * 2^-1, so the biased exponent is 0x3FFE.
    assert_eq!(
        lit("0.5", Radix::Dec, FloatSuffix::None),
        [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xFE, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    );
}

#[test]
fn long_double_constant_folding() {
    use crate::eval::{FloatValue, Value, eval_const};
    fn fold(src: &str) -> Option<Value> {
        let mut sema = Sema::new();
        let mut interner = Interner::new();
        let mut parser = test_parser(src, &mut sema, &mut interner);
        let expr = parser.parse_expression();
        assert!(parser.tokens.is_empty(), "trailing tokens in {src:?}");
        let checked = parser.sema.check_expr(&expr, &parser.intern).expect("check failed");
        eval_const(&checked, parser.sema, &parser.intern)
    }
    fn fold_f80(src: &str) -> FloatValue {
        match fold(src) {
            Some(Value::Float(v @ FloatValue::F80(_))) => v,
            other => panic!("{src:?} did not fold to f80, got {other:?}"),
        }
    }
    // `L` arithmetic folds in 80-bit precision, with the `L` type.
    assert_eq!(fold_f80("1.0L + 2.0L").to_f64(), 3.0);
    assert_eq!(fold_f80("-1.5L").to_f64(), -1.5);
    assert_eq!(fold("0.0L ? 2 : 3"), Some(Value::Int(3)));
    assert_eq!(fold("1.0L ? 2 : 3"), Some(Value::Int(2)));
    // Exact in binary: no rounding anywhere.
    assert_eq!(fold("0x1p-64L + 0x1p-64L == 0x1p-63L"), Some(Value::Int(1)));
    // 2^53+1 is exact in f80 (64-bit mantissa) but rounds in f64, so
    // the difference against 2^53 survives only in extended precision.
    assert_eq!(fold("(int)((long double)9007199254740993 - (long double)9007199254740992)"), Some(Value::Int(1)));
    // Extended range: finite over finite, where f64 would fold inf/inf.
    assert_eq!(fold("(int)(1e4932L / 1e4932L)"), Some(Value::Int(1)));
    // Mixed `double`/`long double` compares in `long double`.
    assert_eq!(fold("1e4932L > 1e308"), Some(Value::Int(1)));
    // NaN only compares unequal to anything, itself included.
    assert_eq!(fold("0.0L / 0.0L != 0.0L"), Some(Value::Int(1)));
    assert_eq!(fold("0.0L / 0.0L == 0.0L"), Some(Value::Int(0)));
}

// ---- struct layout and sizeof (see `crate::layout` / `crate::target`) ----

/// Layout of the tag `tag` after parsing `src`.
fn layout_of(src: &str, tag: &str) -> (StructLayout, Interner) {
    // Member names in a layout are interned symbols, so the interner
    // has to be returned alongside it.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(src, &mut sema, &mut interner);
    parser.parse_translation_unit();
    let sym = parser.intern.get_or_intern(tag);
    let id = parser.sema.lookup_tag(sym).expect("tag declared");
    let layout = parser.sema.struct_layout(id, &parser.intern).expect("layout known");
    (layout, interner)
}

/// `(size, align)` of the tag `tag` after parsing `src`.
fn struct_size_of(src: &str, tag: &str) -> (u64, u64) {
    let (layout, _) = layout_of(src, tag);
    (layout.size, layout.align)
}

/// Byte offset of one member in a layout.
fn member_offset_in(src: &str, tag: &str, member: &str) -> u64 {
    let (layout, interner) = layout_of(src, tag);
    layout_entries(&interner, &layout)
        .into_iter()
        .find(|(name, _, _)| *name == member)
        .map(|(_, offset, _)| offset)
        .expect("member laid out")
}

/// `(name, byte offset, bit)` entries of a layout, in order.
fn layout_entries<'a>(parser_intern: &'a Interner, layout: &'a StructLayout) -> Vec<(&'a str, u64, Option<u32>)> {
    layout.members.iter().map(|m| (parser_intern.resolve(m.name).unwrap(), m.offset, m.bit)).collect()
}

#[test]
fn type_sizes_and_aligns() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let parser = test_parser("", &mut sema, &mut interner);
    let size = |ty: &Ty| parser.sema.type_size(ty, &parser.intern);
    let align = |ty: &Ty| parser.sema.type_align(ty, &parser.intern);
    assert_eq!((size(&Ty::Int), align(&Ty::Int)), (Some(4), Some(4)));
    assert_eq!((size(&Ty::Char), align(&Ty::Char)), (Some(1), Some(1)));
    assert_eq!((size(&Ty::Long), align(&Ty::Long)), (Some(8), Some(8)));
    assert_eq!((size(&Ty::Double), align(&Ty::Double)), (Some(8), Some(8)));
    assert_eq!((size(&Ty::LongDouble), align(&Ty::LongDouble)), (Some(16), Some(16)));
    assert_eq!(
        (
            size(&Ty::Pointer {
                qualifiers: vec![],
                size: PtrSize::Default,
                pointee: Box::new(Ty::Char),
            }),
            align(&Ty::Complex(Box::new(Ty::Double)))
        ),
        (Some(8), Some(8))
    );
    assert_eq!(size(&Ty::Void), None);
    assert_eq!(
        size(&Ty::Func {
            ret: Box::new(Ty::Int),
            params: vec![],
            variadic: false,
            prototyped: true,
        }),
        None
    );
    assert_eq!(size(&Ty::BuiltinVaList), Some(24));
}

#[test]
fn layout_plain_struct() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct P { char a; int b; double d; };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let p = parser.intern.get_or_intern("P");
    let id = parser.sema.lookup_tag(p).unwrap();
    let layout = parser.sema.struct_layout(id, &parser.intern).unwrap();
    assert_eq!((layout.size, layout.align), (16, 8));
    let entries = layout_entries(&parser.intern, &layout);
    assert_eq!(entries, [("a", 0, None), ("b", 4, None), ("d", 8, None)]);
}

#[test]
fn layout_nested_structs_arrays_and_unions() {
    assert_eq!(struct_size_of("struct I { int x; }; struct O { char c; struct I i; int arr[3]; };", "O"), (20, 4));
    assert_eq!(struct_size_of("union U { char c; double d; int arr[3]; };", "U"), (16, 8));
    // An anonymous struct member reserves its nested extent.
    assert_eq!(struct_size_of("struct O { struct { double d; char c; }; char e; };", "O"), (24, 8));
    // ... while an anonymous union member overlaps.
    assert_eq!(struct_size_of("union U { struct { int a; char b; }; int c; };", "U"), (8, 4));
}

#[test]
fn layout_covers_objects_near_the_address_space_limit() {
    // GCC caps objects at 2^63 bytes; the bit cursor must not overflow
    // `u64` on the way there (991014-1).
    let (size, align) = struct_size_of("struct H { short buf[(1L << 62) - 256]; int a; };", "H");
    assert_eq!((size, align), (2 * ((1 << 62) - 256) + 4, 4));
    assert_eq!(member_offset_in("struct H { short buf[(1L << 62) - 256]; int a; };", "H", "a"), 2 * ((1 << 62) - 256));
}

#[test]
fn layout_flexible_array_member() {
    // `sizeof` ignores the flexible member; `offsetof` still locates
    // it (the sqlite `WalIterator.aSegment` shape).
    assert_eq!(struct_size_of("struct F { int n; char data[]; };", "F"), (4, 4));
    assert_eq!(struct_size_of("struct G { char data[]; };", "G"), (0, 1));
    assert_eq!(enum_value_of("struct W { int n; char seg[]; }; enum { N = __builtin_offsetof(struct W, seg) };", "N"), 4);
}

#[test]
fn layout_probed_bitfields() {
    // Every shape below was verified against GCC 14 (sizes,
    // alignments, and observable offsets).
    let cases = [
        ("struct A { char c; int : 0; };", "A", (4, 1)),
        ("struct B { char c; int x : 3; int y : 5; char d; };", "B", (4, 4)),
        ("struct C { int a : 3; char c; };", "C", (4, 4)),
        ("struct D { int f : 24; };", "D", (4, 4)),
        ("union U { int a : 3; char c; };", "U", (4, 4)),
        ("struct H { char c; int x : 9; char d; };", "H", (4, 4)),
        ("struct I { char c; int x : 32; char d; };", "I", (12, 4)),
        ("struct J { char c; int : 5; };", "J", (2, 1)),
        ("struct K { int x : 9; };", "K", (4, 4)),
        ("struct L { char c; short s : 3; int i : 20; char d; };", "L", (8, 4)),
        ("struct M { int a : 30; int b : 3; };", "M", (8, 4)),
        ("struct E {};", "E", (0, 1)),
    ];
    for (src, tag, expected) in cases {
        assert_eq!(struct_size_of(src, tag), expected, "{src}");
    }
    // Offsets and bits within the packed shape: a bit-field records
    // the storage unit holding it (so accesses never overhang the
    // struct) and the bit index of its first bit within that unit.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct B { char c; int x : 3; int y : 5; char d; };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let b = parser.intern.get_or_intern("B");
    let id = parser.sema.lookup_tag(b).unwrap();
    let layout = parser.sema.struct_layout(id, &parser.intern).unwrap();
    assert_eq!(
        layout_entries(&parser.intern, &layout),
        [("c", 0, None), ("x", 0, Some(8)), ("y", 0, Some(11)), ("d", 2, None)]
    );
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct H { char c; int x : 9; char d; };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let h = parser.intern.get_or_intern("H");
    let id = parser.sema.lookup_tag(h).unwrap();
    let layout = parser.sema.struct_layout(id, &parser.intern).unwrap();
    assert_eq!(layout_entries(&parser.intern, &layout), [("c", 0, None), ("x", 0, Some(8)), ("d", 3, None)]);
}

#[test]
fn layout_unknown_without_hanging() {
    // Incomplete types have no layout ...
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let s = parser.intern.get_or_intern("S");
    let id = parser.sema.lookup_tag(s).unwrap();
    assert_eq!(parser.sema.struct_layout(id, &parser.intern), None);
    assert_eq!(parser.sema.type_size(&Ty::Struct(id), &parser.intern), None);
    // ... nor do recursive types (rejected by the parser, but still
    // constructible below it — layout must not hang on them).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let s = interner.get_or_intern("S");
    let id = sema.define_tag(s, false, Span::dummy()).unwrap();
    let v = interner.get_or_intern("s");
    sema.complete_tag(
        id,
        vec![crate::sema::Member {
            name: Some(v),
            ty: Ty::Struct(id),
            bitfield: None,
            packed: false,
            align_override: None,
            span: Span::dummy(),
        }],
    )
    .unwrap();
    assert_eq!(sema.struct_layout(id, &interner), None);
}

#[test]
fn layout_follows_a_custom_target() {
    use crate::target::Target;
    let target = Target {
        int_size: 2,
        int_align: 2,
        ..Target::LP64
    };
    let mut sema = Sema::with_target(target);
    let mut interner = Interner::new();
    let mut parser = test_parser("struct P { char c; int x; };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_eq!(parser.sema.type_size(&Ty::Int, &parser.intern), Some(2));
    let p = parser.intern.get_or_intern("P");
    let id = parser.sema.lookup_tag(p).unwrap();
    let layout = parser.sema.struct_layout(id, &parser.intern).unwrap();
    assert_eq!((layout.size, layout.align), (4, 2));
    assert_eq!(layout_entries(&parser.intern, &layout), [("c", 0, None), ("x", 2, None)]);
}

#[test]
fn sizeof_validation() {
    // `void` and functions have no size in C, but GNU sizes them 1
    // (`sizeof(void)`, `sizeof f`) with a warning ...
    assert_eq!(check_with(&[], "sizeof(void)").ty, Ty::UnsignedLong);
    assert_eq!(check_with(&["int f(void);"], "sizeof(f)").ty, Ty::UnsignedLong);
    assert_eq!(enum_value_of("enum { N = sizeof(void) };", "N"), 1);
    // A function *type* is equally sizeless in C (abstract function
    // declarators like `(void)` in type-names do not parse yet, so the
    // type arrives via typedef here), and equally sized 1 by GNU.
    assert_eq!(enum_value_of("typedef int F(void); enum { N = sizeof(F) };", "N"), 1);
    // ... incomplete types have no size at all ...
    let msg = check_with_err(&["struct S;"], "sizeof(struct S)");
    assert!(msg.contains("incomplete type"), "{msg:?}");
    // ... but complete operands keep the `size_t` type.
    assert_eq!(check_with(&[], "sizeof(int)").ty, Ty::UnsignedLong);
    assert_eq!(check_with(&["int a[3];"], "sizeof(a)").ty, Ty::UnsignedLong);
}

#[test]
fn sizeof_folds_in_constants() {
    // Layout feeds the evaluator: `sizeof` folds where sizes are known.
    assert_eq!(enum_value_of("enum { N = sizeof(int) };", "N"), 4);
    assert_eq!(enum_value_of("enum { N = sizeof(char) };", "N"), 1);
    assert_eq!(enum_value_of("enum { N = sizeof(int *) };", "N"), 8);
    assert_eq!(enum_value_of("struct P { int x; char y; }; enum { N = sizeof(struct P) };", "N"), 8);
    assert_eq!(enum_value_of("enum { N = sizeof(int[3]) };", "N"), 12);
}

#[test]
fn negative_bit_field_width_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int x : -1; };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "negative bit-field width");
}

#[test]
fn over_wide_bit_field_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { char c : 9; };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "bit-field width exceeds its type");
}

#[test]
fn named_zero_width_bit_field_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int x : 0; };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "zero width for bit-field");
}

#[test]
fn ambiguous_anonymous_member_panics() {
    let msg = check_with_err(&["struct O { union { int x; }; union { int x; }; };", "struct O o;"], "o.x");
    assert!(msg.contains("ambiguous member"), "{msg:?}");
}

#[test]
fn abstract_function_declarators_parse() {
    // A `(` opening on specifiers is a parameter list, never grouping
    // (verified against gcc): abstract function declarators in
    // type-names, casts, and parameters.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "typedef int T; int f(int (int)); int g(int (int *)); int h(int (void)); int k(int (T)); int e(int (int, char));",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    // The typedef plus the five declarations.
    assert_eq!(tu.external_declarations.len(), 6);
    // `f` takes a function, which adjusts to a function pointer.
    let f = parser.intern.get_or_intern("f");
    let id = parser.sema.lookup_symbol(f).unwrap();
    let Ty::Func { params, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected a function")
    };
    let [Ty::Pointer { pointee, .. }] = params.as_slice() else {
        panic!("expected one pointer parameter, got {params:?}")
    };
    assert!(matches!(**pointee, Ty::Func { prototyped: true, .. }));
    // `(void)` parameters prototype empty.
    let h = parser.intern.get_or_intern("h");
    let id = parser.sema.lookup_symbol(h).unwrap();
    let Ty::Func { params, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected a function")
    };
    let [Ty::Pointer { pointee, .. }] = params.as_slice() else {
        panic!("expected one pointer parameter, got {params:?}")
    };
    let Ty::Func { params, .. } = &**pointee else { panic!("expected a function pointer") };
    assert!(params.is_empty());
}

#[test]
fn attributes_between_pointer_and_declarator() {
    // GNU attributes may sit between pointer levels or between the
    // pointer list and the direct declarator (verified against gcc).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "int * __attribute__((noinline)) f(void); int * __attribute__((x)) * __attribute__((y)) g(int t);",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
    for name in ["f", "g"] {
        let sym = parser.intern.get_or_intern(name);
        let id = parser.sema.lookup_symbol(sym).unwrap();
        assert!(matches!(parser.sema.decls[id.0 as usize].ty, Ty::Func { .. }), "{name} must stay a function");
    }
}

#[test]
fn attributes_in_cast_type_names() {
    // GNU attributes prefixing a cast's type-name (`(ATTR int(*)(void))`
    // in c-testsuite 00210) are no-ops that still leave a type-name
    // lead; the same holds inside an abstract declarator
    // (`int (ATTR *)(void)`, verified against gcc).
    let Expr::Cast { ty, .. } = parse_one_expr("(__attribute__((noinline)) int) x") else {
        panic!("attributed cast")
    };
    assert_eq!(ty, Ty::Int);

    let Expr::Cast { ty, .. } = parse_one_expr("(__attribute__((noinline)) int (*)(void)) f") else {
        panic!("attributed function-pointer cast")
    };
    let Ty::Pointer { pointee, .. } = ty else {
        panic!("expected a function pointer, got {ty:?}")
    };
    assert!(matches!(*pointee, Ty::Func { .. }));

    let Expr::Cast { ty, .. } = parse_one_expr("(int (__attribute__((noinline)) *)(void)) f") else {
        panic!("attributed abstract declarator")
    };
    let Ty::Pointer { pointee, .. } = ty else {
        panic!("expected a function pointer, got {ty:?}")
    };
    assert!(matches!(*pointee, Ty::Func { .. }));
}

#[test]
fn parenthesized_declarator_parameters_parse() {
    // A redundant paren around a parameter's declarator is grouping,
    // not an empty parameter list: `void ((*f)())` declares a
    // function-pointer parameter (verified against gcc).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void q(void ((*f)()), void ((*g)(void)));", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
    let q = parser.intern.get_or_intern("q");
    let id = parser.sema.lookup_symbol(q).unwrap();
    let Ty::Func { params, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected a function")
    };
    assert_eq!(params.len(), 2);
    for p in params {
        let Ty::Pointer { pointee, .. } = p else {
            panic!("expected a function-pointer parameter, got {p:?}")
        };
        assert!(matches!(**pointee, Ty::Func { .. }));
    }
}

#[test]
fn sizeof_abstract_function_type_sizes_one() {
    // `sizeof(int (int))` parses as a function type (like gcc, which
    // warns and sizes it 1).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("unsigned long s = sizeof(int (int));", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

// ---- initializers across anonymous members ----

#[test]
fn designator_descends_into_anonymous_member() {
    // `.b` names a field of the anonymous union; the cursor continues
    // after it, so a following bare item takes the next member.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct R { union { int a; float b; }; int y; } r = { .b = 2.0, .y = 1 };", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct R { union { int a; float b; }; int y; } r = { .a = 1, 2 };", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn brace_elision_crosses_anonymous_members() {
    // Flat items descend into anonymous struct members transparently.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct O { struct { int x; int y; }; int z; } o = { 1, 2, 3 };", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);

    // An anonymous union is not transparent, matching GCC: one item
    // initializes the union, the next moves on to the following
    // member (`{ 1, 2 }` is `a = 1, c = 2`).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct R { union { int a; int b; }; int c; } r = { 1, 2 };", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);

    // ... including nested anonymous structs.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct P { struct { struct { int x; } i; int y; }; int z; } p = { 1, 2, 3 };", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn chained_and_array_designators() {
    // `.i.x` chains through a named member; `.arr[1]` selects into a
    // member array; the two compose.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct I { int x; }; struct O { struct I i; int y; } o = { .i.x = 5 };", &mut sema, &mut interner);
    assert_eq!(parser.parse_translation_unit().external_declarations.len(), 1);

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int arr[3]; int y; } s = { .arr[1] = 5, .y = 6 };", &mut sema, &mut interner);
    assert_eq!(parser.parse_translation_unit().external_declarations.len(), 1);

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct P { int x; }; struct S { struct P ps[2]; } s = { .ps[1].x = 7 };", &mut sema, &mut interner);
    assert_eq!(parser.parse_translation_unit().external_declarations.len(), 1);
}

#[test]
fn obsolete_colon_designators() {
    // GCC's obsolete `field: value` designator spelling parses like
    // `.field = value` (structs, unions, nested brace lists).
    for src in [
        "struct S { int a; int b; } s = { b: 2, a: 1 };",
        "union U { double d; int i[2]; } u = { d: -0.25 };",
        "struct T { int c[3]; } t = { c: {1, 2, 3} };",
    ] {
        let mut sema = Sema::new();
        let mut interner = Interner::new();
        let mut parser = test_parser(src, &mut sema, &mut interner);
        assert_eq!(parser.parse_translation_unit().external_declarations.len(), 1, "{src}");
    }
}

#[test]
fn union_designated_items_may_repeat() {
    // Designated union items never count as excess (only a second
    // *bare* item does).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("union U { int a; float b; } u = { .a = 1, .b = 2.0 };", &mut sema, &mut interner);
    assert_eq!(parser.parse_translation_unit().external_declarations.len(), 1);
}

#[test]
fn union_second_bare_item_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("union U { int a; float b; } u = { 1, 2 };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "excess elements in union initializer");
}

#[test]
fn too_many_bare_initializers_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int a; } s = { 1, 2 };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "too many initializers for struct");
}

#[test]
fn brace_elision_into_array_member() {
    // Bare scalars fill an array member in order (C17 §6.7.9p20).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { char f1; int f2[2]; } object = {'X', 8, 9};", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn brace_elision_through_nested_struct() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct R { struct { long x; } a, b; } r = { 100, 200 };", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn brace_elision_into_array_elements() {
    // Each braced list is one element; bare scalars inside elide
    // through the element's array member.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { unsigned u[4]; } a[] = { { 1, 2, 3, 4 }, { 0 } };", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn brace_elision_single_scalar_into_array() {
    // One scalar initializes the first element, the rest zero-fill.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int a[4]; } s = { 5 };", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn braced_list_initializes_anonymous_member_itself() {
    // A braced list fills the anonymous aggregate's fields; it must
    // not descend to the first leaf (`{"1234", "567"}` is x and y,
    // not two initializers for x).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "union U { struct { char x[4]; char y[4]; }; struct { char z[8]; }; } u = {{\"1234\", \"567\"}};",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn ambiguous_designator_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct O { union { int x; }; union { int x; }; } o = { .x = 1 };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "ambiguous member");
}

#[test]
fn designator_into_array_without_index_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int arr[3]; } s = { .arr.x = 1 };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "field designator in array initializer");
}

#[test]
fn designator_chain_selects_into_array_element() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "struct S { int bit; int other; }; struct S s[2] = { [1].bit = 4, [0].other = 3 };",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn designator_chain_selects_into_nested_arrays() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int m[2][3] = { [1][2] = 5 };", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

// ---- array bounds and case labels are constant ----

#[test]
fn negative_array_bound_panics() {
    // A folded negative bound is rejected ...
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x[-1];", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "negative array bound");
}

#[test]
fn negative_folded_array_bound_panics() {
    // ... even when the negativity only shows after folding ...
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x[1 - 2];", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "negative array bound");
}

#[test]
fn non_constant_bound_stays_variable_length() {
    // ... while a non-constant bound stays a (runtime-sized) array.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int n) { int a[n]; return a[0]; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn zero_array_bound_accepted_as_extension() {
    // `int x[0]` is a constraint violation in strict C, but GCC
    // accepts it — and so do we.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x[0];", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

// ---- variable-length arrays ----

#[test]
fn prototype_params_scope_over_later_bounds() {
    // Parameter names live in prototype scope (C17 §6.2.1p4), so a
    // later bound sees an earlier parameter — the normal VLA prototype
    // shape.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int n, int a[n]); void g(int n, int a[n]) { a[0] = n; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
}

#[test]
fn duplicate_prototype_param_names_panic() {
    // Verified against gcc ("redefinition of parameter").
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int x, int x);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "conflicting redeclaration");
}

#[test]
fn file_scope_vla_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int n; int a[n];", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "variably modified type at file scope");
}

#[test]
fn file_scope_star_bound_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a[*];", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "`[*]` outside function prototype scope");
}

#[test]
fn block_scope_star_bound_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { int a[*]; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "`[*]` outside function prototype scope");
}

#[test]
fn vla_struct_member_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int n; struct S { int a[n]; };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "variably modified struct member");
}

#[test]
fn star_struct_member_panics() {
    // `[*]` needs prototype scope wherever it appears.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int a[*]; };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "`[*]` outside function prototype scope");
}

#[test]
fn vla_function_pointer_member_parses() {
    // The member is a plain pointer; the variably modified prototype
    // sits behind it, where prototypes may be variably modified.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int (*f)(int n, int a[n]); } s;", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn vla_initializer_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int n) { int a[n] = { 1 }; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "variable-sized object may not be initialized");
}

#[test]
fn vla_empty_initializer_parses() {
    // GCC accepts an empty brace list on a VLA.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int n) { int a[n] = {}; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn vla_pointer_and_multidimensional_vla_parse() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int n) { int (*p)[n]; int b[2][n]; (void)p; (void)b; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn vla_bound_expression_retained() {
    // A non-constant bound keeps its checked expression in
    // `Sema::vla_bounds` for runtime evaluation (allocation,
    // `sizeof`, strides).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int n) { int a[n + 1]; (void)a; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
    let f = parser.intern.get_or_intern("f");
    let fid = parser.sema.lookup_symbol(f).expect("function declared");
    let body = parser.sema.bodies[&fid].clone();
    let [BlockItem::Decl { ids, .. }, _] = body.block.items.as_slice() else {
        panic!("expected a declaration followed by a statement, got {:?}", body.block.items)
    };
    let [a] = ids.as_slice() else { panic!("expected one declarator, got {ids:?}") };
    let Ty::Array {
        bound: Some(ArrayBound::Vla(slot)),
        ..
    } = &parser.sema.decls[a.0 as usize].ty
    else {
        panic!("expected a VLA object, got {:?}", parser.sema.decls[a.0 as usize].ty)
    };
    assert!(
        matches!(parser.sema.vla_bounds[*slot as usize].kind, CheckedExprKind::Binary(BinaryOp::Add, _, _)),
        "expected the `n + 1` bound, got {:?}",
        parser.sema.vla_bounds[*slot as usize].kind
    );
}

#[test]
fn vla_param_bound_remapped_to_body_params() {
    // `a[n]` in a definition checks against the prototype-scope `n`;
    // the stored bound must reference the body's parameter instead,
    // or the backend would lower a dead `DeclId`.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int n, int a[2][n]) { a[0][0] = n; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
    let f = parser.intern.get_or_intern("f");
    let fid = parser.sema.lookup_symbol(f).expect("function declared");
    let body = parser.sema.bodies[&fid].clone();
    let [n_id, a_id] = body.params.as_slice() else {
        panic!("expected two params, got {:?}", body.params)
    };
    // The parameter decayed to a pointer to the `int[n]` row.
    let Ty::Pointer { pointee, .. } = &parser.sema.decls[a_id.0 as usize].ty else {
        panic!("expected a decayed array parameter, got {:?}", parser.sema.decls[a_id.0 as usize].ty)
    };
    let Ty::Array {
        bound: Some(ArrayBound::Vla(slot)),
        ..
    } = pointee.as_ref()
    else {
        panic!("expected a VLA row, got {pointee:?}")
    };
    let bound = &parser.sema.vla_bounds[*slot as usize];
    assert_eq!(bound.kind, CheckedExprKind::Ident(*n_id), "bound must name the body parameter, got {:?}", bound.kind);
}

#[test]
fn kr_vla_param_bound_remapped_to_body_params() {
    // K&R parameters check against a temporary scope that is
    // truncated away: the stored bound must reference the body's
    // parameter. The rewrite is simultaneous — sequential rewrites
    // cascade when truncated ids are reused.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int get(m, n, a) int m, n; int a[m][n]; { return a[1][2]; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
    let g = parser.intern.get_or_intern("get");
    let fid = parser.sema.lookup_symbol(g).expect("function declared");
    let body = parser.sema.bodies[&fid].clone();
    let [m_id, n_id, a_id] = body.params.as_slice() else {
        panic!("expected three params, got {:?}", body.params)
    };
    let Ty::Pointer { pointee, .. } = &parser.sema.decls[a_id.0 as usize].ty else {
        panic!("expected a decayed array parameter, got {:?}", parser.sema.decls[a_id.0 as usize].ty)
    };
    let Ty::Array {
        bound: Some(ArrayBound::Vla(slot)),
        ..
    } = pointee.as_ref()
    else {
        panic!("expected a VLA row, got {pointee:?}")
    };
    let bound = &parser.sema.vla_bounds[*slot as usize];
    assert_eq!(bound.kind, CheckedExprKind::Ident(*n_id), "bound must name the body parameter, got {:?}", bound.kind);
    let _ = m_id;
}

#[test]
fn fixed_array_to_vla_param_accepted() {
    // `int[n]` and `int[4]` are compatible (C17 §6.7.6.2p6), so a
    // fixed array reaches a VLA parameter.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "int get(int m, int n, int a[m][n]) { return a[1][2]; } int main(void) { int b[3][4]; return get(3, 4, b); }",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
}

#[test]
fn vla_static_storage_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int n) { static int a[n]; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "variably modified object with static storage duration");
}

#[test]
fn vla_linkage_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int n) { extern int a[n]; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "variably modified object with linkage");
}

#[test]
fn pointer_to_vla_with_static_storage_parses() {
    // Only the VLA type itself needs automatic storage (C17
    // §6.7.6.2p2): a `static` pointer to a VLA is a fixed-size object,
    // which GCC accepts (`static int (*starr)[h][w];` in tcctest.c).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int n) { static int (*p)[n]; (void)p; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
}

#[test]
fn nested_vla_type_static_storage_panics() {
    // An array whose element type is a VLA is itself a VLA type, so
    // static storage stays rejected even though the outer bound is
    // constant.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int n) { static int a[3][n]; (void)a; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "variably modified object with static storage duration");
}

#[test]
fn goto_over_vla_panics() {
    // Jumping past a VLA declaration into its scope would leave
    // storage unallocated (C11 §6.8.6.1p1).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int n) { goto l; int a[n]; l:; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "jump into scope of identifier with variably modified type");
}

#[test]
fn switch_over_vla_panics() {
    // The dispatch jumps over the declaration like a goto.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int n, int x) { switch (x) { int a[n]; case 1: break; } }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "jump into scope of identifier with variably modified type");
}

#[test]
fn goto_with_live_vla_parses() {
    // Jumping out past a live VLA frees it (no diagnostic): the
    // backward loop, the same-block cleanup, and the nested exit.
    for src in [
        "void f(int n) { lab:; int a[n]; (void)a; goto lab; }",
        "void f(int n) { int a[n]; (void)a; goto out; out:; }",
        "void f(int n) { int a[n]; { goto out; } out: (void)a; }",
        "void f(int n, int x) { switch (x) { case 1:; int a[n]; (void)a; break; } }",
    ] {
        let mut sema = Sema::new();
        let mut interner = Interner::new();
        let mut parser = test_parser(src, &mut sema, &mut interner);
        let tu = parser.parse_translation_unit();
        assert_eq!(tu.external_declarations.len(), 1, "{src}");
    }
}

// ---- array initializer lengths ----

#[test]
fn array_initializer_exact_and_short_parse() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a[3] = { 1, 2, 3 }; int b[3] = { 1 };", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
}

#[test]
fn array_initializer_excess_warns_not_fails() {
    // GCC and clang warn (not error) on excess elements — and so the
    // parse succeeds; the warning renders to stderr.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a[3] = { 1, 2, 3, 4 };", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn array_designator_out_of_range_panics() {
    // Out-of-range `[i]` is a hard error under both compilers.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a[3] = { [5] = 1 };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "array index in initializer exceeds array bounds");
}

#[test]
fn array_designators_track_positions() {
    // `[1]` then two bare items fills 1..3 exactly; a repeated
    // designator is fine.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a[3] = { [1] = 1, 2, 3 }; int b[2] = { [1] = 1, [0] = 2 };", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
}

#[test]
fn nested_array_excess_warns() {
    // The inner `{3, 4}` overflows the inner `[2]` (a warning, like
    // GCC), while the outer list fits.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a[2][2] = { { 1, 2 }, { 3, 4, 5 } };", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn string_initializer_fit() {
    // Exact fit drops the NUL (valid C); short literals pad; `[]`
    // sizes to fit; escapes count as one byte each.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "char a[3] = \"abc\"; char b[4] = \"ab\"; char c[] = \"ab\"; char d[4] = \"a\\n\";",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 4);
}

#[test]
fn string_initializer_too_long_warns_not_fails() {
    // A GCC warning, not an error — the parse succeeds.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("char s[3] = \"abcd\";", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn sizeof_string_literal_folds() {
    // `sizeof` counts the NUL; an escape is one byte.
    assert_eq!(enum_value_of("enum { N = sizeof(\"ab\") };", "N"), 3);
    assert_eq!(enum_value_of("enum { N = sizeof(\"a\\n\") };", "N"), 3);
}

#[test]
fn string_literal_types_carry_their_bound() {
    // The literal's type is a complete array (N includes the NUL), so
    // every `sizeof` path folds — not just the constant evaluator's.
    // Wide literals count elements, with the width in the element type.
    let checked = check_with(&[], "\"ab\"");
    assert!(matches!(
        &checked.ty,
        Ty::Array { elem, bound: Some(ArrayBound::Fixed(3)), .. } if **elem == Ty::Char
    ));
    let checked = check_with(&[], "L\"ab\"");
    assert!(matches!(
        &checked.ty,
        Ty::Array { elem, bound: Some(ArrayBound::Fixed(3)), .. } if **elem == Ty::Int
    ));
    assert_eq!(enum_value_of("enum { N = sizeof(L\"ab\") };", "N"), 12);
}

#[test]
fn string_bytes_decode_escapes() {
    // The backend's data-emission API: escapes become single bytes
    // (hex truncated mod 256, like GCC), adjacent literals join, and
    // wide literals encode one little-endian element each.
    use crate::eval::string_bytes;
    use crate::lexer::{StringLiteral, StringPrefix};
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let parser = test_parser("", &mut sema, &mut interner);
    let a = parser.intern.get_or_intern("\"a\\n\\x41\"");
    let b = parser.intern.get_or_intern("\"b\"");
    let parts = vec![
        StringLiteral {
            spelling: a,
            prefix: StringPrefix::None,
        },
        StringLiteral {
            spelling: b,
            prefix: StringPrefix::None,
        },
    ];
    assert_eq!(string_bytes(&parts, &parser.intern).unwrap(), b"a\nAb");
    let w = parser.intern.get_or_intern("L\"ab\"");
    let wide = vec![StringLiteral {
        spelling: w,
        prefix: StringPrefix::L,
    }];
    // Wide literals encode each element little-endian in its width
    // (`L` is a 4-byte `wchar_t` on Linux targets).
    assert_eq!(string_bytes(&wide, &parser.intern).unwrap(), b"a\0\0\0b\0\0\0");
}

#[test]
fn wide_literals_follow_the_target_wchar() {
    // Windows `wchar_t` is a 16-bit `unsigned short` (LLP64): wide
    // elements are two bytes and take `unsigned short` type, while
    // LP64 keeps the 4-byte signed `int`.
    use crate::eval::{string_bytes_in, string_elem_ty_in};
    use crate::lexer::{StringLiteral, StringPrefix};
    use crate::target::{Target, WChar};

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let parser = test_parser("", &mut sema, &mut interner);
    let w = parser.intern.get_or_intern("L\"ab\"");
    let wide = vec![StringLiteral {
        spelling: w,
        prefix: StringPrefix::L,
    }];

    let win = Target::WINDOWS_X86_64.wchar();
    assert_eq!(string_elem_ty_in(&wide, win), Some(Ty::UnsignedShort));
    assert_eq!(string_bytes_in(&wide, &parser.intern, win).unwrap(), b"a\0b\0");

    let lp64 = WChar { size: 4, signed: true };
    assert_eq!(string_elem_ty_in(&wide, lp64), Some(Ty::Int));
    assert_eq!(string_bytes_in(&wide, &parser.intern, lp64).unwrap(), b"a\0\0\0b\0\0\0");
}

#[test]
fn windows_wchar_sizes_and_wide_constant_type() {
    use crate::target::Target;

    fn enum_of(src: &str, name: &str, target: Target) -> i128 {
        let mut sema = Sema::new();
        sema.target = target;
        let mut interner = Interner::new();
        let mut parser = test_parser(src, &mut sema, &mut interner);
        parser.parse_translation_unit();
        let sym = parser.intern.get_or_intern(name);
        let id = parser.sema.lookup_symbol(sym).expect("enumerator declared");
        match parser.sema.decls[id.0 as usize].kind {
            BindingKind::EnumConst { value } => value,
            ref kind => panic!("expected an enumerator, got {kind:?}"),
        }
    }

    let win = Target::WINDOWS_X86_64;
    // `L"ab"` is 3 elements of 2 bytes; `L'a'` is a 2-byte value.
    assert_eq!(enum_of("enum { N = sizeof(L\"ab\") };", "N", win), 6);
    assert_eq!(enum_of("enum { N = sizeof(L'a') };", "N", win), 2);
    let lp64 = Target::X86_64;
    assert_eq!(enum_of("enum { N = sizeof(L\"ab\") };", "N", lp64), 12);
    assert_eq!(enum_of("enum { N = sizeof(L'a') };", "N", lp64), 4);
}

#[test]
fn non_utf8_source_bytes_survive_to_data() {
    // clang -E passes raw source bytes through, so a string holding a
    // lone `\xff` is not valid UTF-8: decoding falls back to Latin-1
    // and the original byte must come back out of `string_bytes`.
    use crate::decode_source;
    use crate::eval::string_bytes;
    use crate::lexer::{StringLiteral, StringPrefix};
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let parser = test_parser("", &mut sema, &mut interner);
    let decoded = decode_source(b"const char g[] = \"\0\xff\";");
    assert!(decoded.contains('\u{ff}'), "raw 0xff must survive decoding");
    let a = parser.intern.get_or_intern("\"\0\u{ff}\"");
    let parts = vec![StringLiteral {
        spelling: a,
        prefix: StringPrefix::None,
    }];
    assert_eq!(string_bytes(&parts, &parser.intern).unwrap(), b"\0\xff");
    // A real multibyte source character contributes its UTF-8 bytes.
    let b = parser.intern.get_or_intern("\"\u{4e2d}\"");
    let parts = vec![StringLiteral {
        spelling: b,
        prefix: StringPrefix::None,
    }];
    assert_eq!(string_bytes(&parts, &parser.intern).unwrap(), "\u{4e2d}".as_bytes());
}

#[test]
fn string_contents_decodes() {
    // The `String` twin of `string_bytes_decode_escapes`: quotes and
    // prefix stripped, escapes decoded, adjacent literals joined,
    // wide literals refused.
    use crate::lexer::{StringLiteral, StringPrefix};
    use crate::sema::string_contents;
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let parser = test_parser("", &mut sema, &mut interner);
    let a = parser.intern.get_or_intern("\"a\\n\\x41\"");
    let b = parser.intern.get_or_intern("\"b\"");
    let parts = vec![
        StringLiteral {
            spelling: a,
            prefix: StringPrefix::None,
        },
        StringLiteral {
            spelling: b,
            prefix: StringPrefix::None,
        },
    ];
    assert_eq!(string_contents(&parts, &parser.intern).unwrap(), "a\nAb");
    let w = parser.intern.get_or_intern("L\"ab\"");
    let wide = vec![StringLiteral {
        spelling: w,
        prefix: StringPrefix::L,
    }];
    assert_eq!(string_contents(&wide, &parser.intern), None);
    // High-byte escapes are lossy by construction (see
    // `string_contents` docs): C's byte 0xFF is U+00FF here.
    let h = parser.intern.get_or_intern("\"\\xff\"");
    let high = vec![StringLiteral {
        spelling: h,
        prefix: StringPrefix::None,
    }];
    assert_eq!(string_contents(&high, &parser.intern).unwrap(), "ÿ");
}

#[test]
fn incomplete_array_completes_from_initializer() {
    // C17 §6.7.9p22: the outermost dimension takes its size from the
    // initializer — which makes `sizeof` fold (dwm's static assertion
    // relies on exactly this).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a[] = { 1, 2, 3 };", &mut sema, &mut interner);
    let ids = parser.parse_declaration();
    let Ty::Array {
        bound: Some(ArrayBound::Fixed(n)),
        ..
    } = &parser.sema.decls[ids[0].0 as usize].ty
    else {
        panic!("expected a completed array");
    };
    assert_eq!(*n, 3);
    // ... including through designators and string literals (NUL
    // included), and `sizeof` folds afterwards.
    assert_eq!(enum_value_of("int a[] = { 1, 2 }; enum { N = sizeof(a) };", "N"), 8);
    assert_eq!(enum_value_of("int b[] = { [2] = 1 }; enum { N = sizeof(b) };", "N"), 12);
    assert_eq!(enum_value_of("char s[] = \"ab\"; enum { N = sizeof(s) };", "N"), 3);
    // The braced lone-string form checks and sizes like the bare one
    // (C17 §6.7.9p14; tcctest.c's `str_ag2`).
    assert_eq!(enum_value_of("char s[] = { \"ab\" }; enum { N = sizeof(s) };", "N"), 3);
    // The dwm static-assertion shape (the tag declaration itself
    // yields no external declaration).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "static const char *tags[] = { \"a\", \"b\" }; struct NumTags { char x[(sizeof(tags) / sizeof(tags[0])) > 31 ? -1 : 1]; };",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn incomplete_array_without_initializer_stays_incomplete() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a[];", &mut sema, &mut interner);
    let ids = parser.parse_declaration();
    let Ty::Array { bound: None, .. } = &parser.sema.decls[ids[0].0 as usize].ty else {
        panic!("expected an incomplete array");
    };
}

// ---- GNU attributes: packed / aligned take effect ----

#[test]
fn attribute_packed_struct_layout() {
    // Verified against gcc: no padding anywhere, alignment 1.
    assert_eq!(struct_size_of("struct __attribute__((packed)) Pk { char c; int x; };", "Pk"), (5, 1));
    // Post-body attributes apply to the type even with a declarator
    // (verified against gcc).
    assert_eq!(struct_size_of("struct S { char c; int x; } __attribute__((packed)) v;", "S"), (5, 1));
    // ... and unions pack the same way.
    assert_eq!(struct_size_of("union __attribute__((packed)) PU { int x; char c[9]; };", "PU"), (9, 1));
}

#[test]
fn attribute_packed_member_layout() {
    // A packed member drops its own alignment (verified against gcc).
    assert_eq!(struct_size_of("struct M { char c; int x __attribute__((packed)); };", "M"), (5, 1));
}

#[test]
fn attribute_aligned_layout() {
    // A member `aligned(n)` is a minimum alignment: `x` moves to 16
    // and the struct follows (verified against gcc).
    assert_eq!(struct_size_of("struct A { char c; int x __attribute__((aligned(16))); };", "A"), (32, 16));
    assert_eq!(member_offset_in("struct A { char c; int x __attribute__((aligned(16))); };", "A", "x"), 16);
    // A struct `aligned(n)` pads the size (verified against gcc).
    assert_eq!(struct_size_of("struct __attribute__((aligned(16))) SA { char c; };", "SA"), (16, 16));
    // Bare `aligned` is the target's maximum useful alignment (16 on
    // x86-64, verified against gcc).
    assert_eq!(struct_size_of("struct B { char c; } __attribute__((aligned));", "B"), (16, 16));
}

#[test]
fn alignof_object_takes_its_alignment_attribute() {
    // An object's `aligned` attribute sets its alignment, raising or
    // lowering it (like GCC); without one the type's alignment applies.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "long tight __attribute__((aligned(1)));\
         char wide __attribute__((aligned(64)));\
         char plain;\
         __alignof__(tight) * 1000 + __alignof__(wide) * 10 + __alignof__(plain)",
        &mut sema,
        &mut interner,
    );
    parser.parse_declaration();
    parser.parse_declaration();
    parser.parse_declaration();
    let expr = parser.parse_expression();
    let checked = parser.sema.check_expr(&expr, &parser.intern).expect("check failed");
    assert_eq!(crate::eval::eval_const(&checked, parser.sema, &parser.intern), Some(crate::eval::Value::Int(1641)));
}

#[test]
fn alignof_folds() {
    // GNU `__alignof__` (and C11 `_Alignof`) fold like `sizeof`,
    // through types and expressions alike.
    assert_eq!(enum_value_of("enum { N = __alignof__(long long) };", "N"), 8);
    assert_eq!(enum_value_of("enum { N = _Alignof(char) };", "N"), 1);
    assert_eq!(enum_value_of("int x; enum { N = __alignof__ x };", "N"), 4);
    // The shape in clang's own headers: an `aligned` member argument.
    assert_eq!(
        struct_size_of("struct M { long long a __attribute__((__aligned__(__alignof__(long long)))); char c; };", "M"),
        (16, 8)
    );
}

#[test]
fn c23_labels_before_declarations_and_at_block_end() {
    let c23 = LangOpts {
        std: Std::C23,
        gnu: false,
        ms: false,
        clang: false,
    };
    let c17 = LangOpts {
        std: Std::C17,
        gnu: false,
        ms: false,
        clang: false,
    };
    assert!(tu_diagnostics_with_lang(
        "int main(void) { goto L; L: int x = 1; return x; }",
        c23
    )
    .is_empty());
    assert!(tu_diagnostics_with_lang("int main(void) { switch (0) { case 1: int x = 1; break; } return 0; }", c23).is_empty());
    assert!(tu_diagnostics_with_lang("int main(void) { L: }", c23).is_empty());
    assert!(!tu_diagnostics_with_lang("int main(void) { L: int x = 1; return x; }", c17).is_empty());
}

#[test]
fn c23_bool_true_false() {
    let c23 = LangOpts {
        std: Std::C23,
        gnu: false,
        ms: false,
        clang: false,
    };
    assert!(tu_diagnostics_with_lang(
        "_Static_assert(sizeof(bool) == 1, \"bool\");\n\
         _Static_assert(sizeof(true) == 1, \"true\");\n\
         _Static_assert(_Generic(true, bool: 1, default: 0), \"type\");\n\
         bool b = true; int main(void) { return false; }",
        c23
    )
    .is_empty());
    assert!(tu_diagnostics("int bool = 1;").is_empty());
}

#[test]
fn c23_nullptr() {
    let c23 = LangOpts {
        std: Std::C23,
        gnu: false,
        ms: false,
        clang: false,
    };
    assert!(tu_diagnostics_with_lang(
        "_Static_assert(sizeof(nullptr) == sizeof(void *), \"size\");\n\
         typedef typeof(nullptr) nullptr_t;\n\
         nullptr_t p = nullptr;\n\
         int main(void) { int *q = nullptr; return q == nullptr; }",
        c23
    )
    .is_empty());
    assert!(tu_diagnostics("int nullptr = 0;").is_empty());
}

#[test]
fn c23_auto_deduces_from_initializer() {
    let c23 = LangOpts {
        std: Std::C23,
        gnu: false,
        ms: false,
        clang: false,
    };
    let src = "int main(void) {\n\
                   auto x = 2.5;\n\
                   auto p = &x;\n\
                   auto s = \"hi\";\n\
                   const int ci = 4;\n\
                   auto y = ci;\n\
                   auto const z = 7;\n\
                   static auto st = 3;\n\
                   _Static_assert(_Generic(x, double: 1, default: 0), \"x\");\n\
                   _Static_assert(_Generic(p, double *: 1, default: 0), \"p\");\n\
                   _Static_assert(_Generic(s, char *: 1, default: 0), \"s\");\n\
                   _Static_assert(_Generic(y, int: 1, default: 0), \"y\");\n\
                   _Static_assert(_Generic(&z, const int *: 1, default: 0), \"z\");\n\
                   return z + st;\n\
               }";
    let diagnostics = tu_diagnostics_with_lang(src, c23);
    assert!(diagnostics.iter().all(|d| !d.is_error()), "{diagnostics:#?}");
}

#[test]
fn c23_constexpr_values_fold_in_constant_contexts() {
    let c23 = LangOpts {
        std: Std::C23,
        gnu: false,
        ms: false,
        clang: false,
    };
    let src = "constexpr int n = 3;\n\
               int a[n];\n\
               constexpr double d = 1.5;\n\
               int b[(int)d];\n\
               constexpr int *p = 0;\n\
               enum { E = n + 2 };\n\
               _Static_assert(n == 3, \"n\");\n\
               int main(void) {\n\
                   constexpr int m = 4;\n\
                   int c[m];\n\
                   _Static_assert(m == 4, \"m\");\n\
                   return (sizeof a / sizeof a[0]) + (sizeof c / sizeof c[0]) + E;\n\
               }";
    let diagnostics = tu_diagnostics_with_lang(src, c23);
    assert!(diagnostics.iter().all(|d| !d.is_error()), "{diagnostics:#?}");
}

#[test]
fn c23_constexpr_rejects_nonconstant_initializers_and_writes() {
    let c23 = LangOpts {
        std: Std::C23,
        gnu: false,
        ms: false,
        clang: false,
    };
    for (src, expected) in [
        ("constexpr int x;", "requires an initializer"),
        ("int y; constexpr int x = y;", "not a constant expression"),
        ("constexpr int *p = (int *)1;", "not a constant expression"),
        ("int main(void) { constexpr int n = 3; n = 4; return n; }", "read-only"),
        ("int main(void) { constexpr int n = 3; return ++n; }", "read-only"),
    ] {
        let diagnostics = tu_diagnostics_with_lang(src, c23);
        assert_diag_contains(&diagnostics, expected);
    }
    // Before C23 the spelling is an ordinary identifier.
    assert!(tu_diagnostics_with_lang("int constexpr = 1;", LangOpts::default()).is_empty());
}

#[test]
fn c23_attributes_parse_and_drive_warnings() {
    let c23 = LangOpts {
        std: Std::C23,
        gnu: false,
        ms: false,
        clang: false,
    };
    let src = "[[nodiscard]] int f(void);\n\
               [[deprecated]] int old(void);\n\
               [[gnu::packed]] struct S { int x; };\n\
               [[maybe_unused, deprecated]] int object;\n\
               int g(int x) { switch (x) { case 1: x++; [[fallthrough]]; default: return x; } }\n\
               int main(void) { [[maybe_unused]] int local = old(); f(); return local + g(1); }";
    let diagnostics = tu_diagnostics_with_lang(src, c23);
    assert!(diagnostics.iter().all(|d| !d.is_error()), "{diagnostics:#?}");
    assert!(
        diagnostics
            .iter()
            .any(|d| d.severity == Severity::Warning && d.message.contains("ignoring return value of 'f'")),
        "{diagnostics:#?}"
    );
    assert!(
        diagnostics
            .iter()
            .any(|d| d.severity == Severity::Warning && d.message.contains("'old' is deprecated")),
        "{diagnostics:#?}"
    );

    // Taking the nodiscard result silences its diagnostic.
    let diagnostics = tu_diagnostics_with_lang("[[nodiscard]] int f(void); int main(void) { return f(); }", c23);
    assert!(diagnostics.iter().all(|d| d.severity != Severity::Warning), "{diagnostics:#?}");

    // The spelling is a C23 declaration form only.
    let c17 = LangOpts {
        std: Std::C17,
        gnu: false,
        ms: false,
        clang: false,
    };
    let diagnostics = tu_diagnostics_with_lang("[[deprecated]] int x;", c17);
    assert!(diagnostics.iter().any(|d| d.is_error()), "{diagnostics:#?}");
}

#[test]
fn c23_auto_rejects_bad_declarations() {
    let c23 = LangOpts {
        std: Std::C23,
        gnu: false,
        ms: false,
        clang: false,
    };
    for (src, expected) in [
        ("int main(void) { auto x; return 0; }", "requires an initializer"),
        ("int main(void) { auto *p = 0; return 0; }", "plain identifier"),
        ("int main(void) { auto a[2] = {1, 2}; return 0; }", "plain identifier"),
        ("int main(void) { auto x = 1, y = 2; return 0; }", "single declarator"),
        ("int main(void) { auto int z = 3; return 0; }", "cannot be combined"),
        ("int main(void) { auto q = {1}; return 0; }", "must be an expression"),
    ] {
        let diagnostics = tu_diagnostics_with_lang(src, c23);
        assert_diag_contains(&diagnostics, expected);
    }
}

#[test]
fn c23_bitint_types_and_suffixes() {
    let c23 = LangOpts {
        std: Std::C23,
        gnu: false,
        ms: false,
        clang: false,
    };
    assert!(tu_diagnostics_with_lang(
        "_Static_assert(sizeof(_BitInt(8)) == 1, \"8-bit storage\");\n\
         _Static_assert(_Alignof(_BitInt(9)) == 2, \"16-bit storage\");\n\
         _Static_assert(sizeof(_BitInt(17)) == 4, \"32-bit storage\");\n\
         _Static_assert(sizeof(unsigned _BitInt(33)) == 8, \"64-bit storage\");\n\
         _Static_assert(_Generic((signed _BitInt(40))1, signed _BitInt(40): 1, default: 0), \"signed type\");\n\
         _Static_assert(_Generic(127wb, _BitInt(8): 1, default: 0), \"wb width\");\n\
         _Static_assert(_Generic(255uwb, unsigned _BitInt(8): 1, default: 0), \"uwb width\");\n\
         int main(void) {\n\
             signed _BitInt(40) a = 1;\n\
             unsigned _BitInt(9) b = 511;\n\
             return (int)(a + b);\n\
         }",
        c23
    )
    .is_empty());
    assert!(tu_diagnostics_with_lang("_BitInt(0) x;", c23).iter().any(|d| d.is_error()));
    assert!(tu_diagnostics_with_lang("_BitInt(65) x;", c23).iter().any(|d| d.is_error()));
    assert!(tu_diagnostics_with_lang("signed _BitInt(1) x;", c23).iter().any(|d| d.is_error()));
    assert!(!tu_diagnostics("_BitInt(8) x;").is_empty());
}

#[test]
fn c23_bitint_usual_arithmetic_conversions() {
    use Ty::*;
    let b8 = BitInt { width: 8, signed: true };
    let u8 = BitInt { width: 8, signed: false };
    let s32 = BitInt { width: 32, signed: true };
    let u32 = BitInt { width: 32, signed: false };
    let s40 = BitInt { width: 40, signed: true };
    let u40 = BitInt { width: 40, signed: false };
    assert_eq!(Sema::uac_common_type(&b8, &Int), Int);
    assert_eq!(Sema::uac_common_type(&u8, &Int), Int);
    assert_eq!(Sema::uac_common_type(&s32, &Int), Int);
    assert_eq!(Sema::uac_common_type(&u32, &Int), UnsignedInt);
    assert_eq!(Sema::uac_common_type(&s40, &Int), s40.clone());
    assert_eq!(Sema::uac_common_type(&u40, &Int), u40.clone());
    assert_eq!(Sema::uac_common_type(&s40, &Long), Long);
    assert_eq!(Sema::uac_common_type(&u40, &Long), Long);
    assert_eq!(Sema::uac_common_type(&b8, &u8), u8.clone());
    assert_eq!(Sema::uac_common_type(&b8, &BitInt { width: 16, signed: true }), BitInt { width: 16, signed: true });
}

#[test]
fn c23_bare_alignof() {
    let c23 = LangOpts {
        std: Std::C23,
        gnu: false,
        ms: false,
        clang: false,
    };
    assert_eq!(enum_value_of_with_lang("enum { N = alignof(long long) };", "N", c23), 8);
    assert_eq!(enum_value_of_with_lang("int x; enum { N = alignof x };", "N", c23), 4);
}

#[test]
fn alignof_void_is_one() {
    // GNU: `__alignof__(void)` folds to 1 (verified against gcc/clang).
    assert_eq!(enum_value_of("enum { N = __alignof__(void) };", "N"), 1);
}

#[test]
fn alignas_specifier_aligns_members() {
    // `_Alignas`/`alignas` on a member (C11 §6.7.5, C23 keyword)
    // raises the member's alignment like `aligned(n)`; values verified
    // against gcc.
    assert_eq!(struct_size_of("struct A { char c; int _Alignas(16) x; };", "A"), (32, 16));
    assert_eq!(struct_size_of("struct B { char c; int alignas(8) y; };", "B"), (16, 8));
}

#[test]
fn alignas_opens_a_declaration_at_file_and_block_scope() {
    let c23 = LangOpts {
        std: Std::C23,
        gnu: false,
        ms: false,
        clang: false,
    };
    assert!(tu_diagnostics("_Alignas(16) int x;").is_empty());
    assert!(tu_diagnostics("void f(void) { _Alignas(16) int y; (void)y; }").is_empty());
    assert!(tu_diagnostics_with_lang("alignas(16) int x;", c23).is_empty());
    assert!(
        tu_diagnostics_with_lang("void f(void) { alignas(16) int y; (void)y; }", c23).is_empty()
    );
}

#[test]
fn single_underscore_attribute_parses() {
    // `__attribute` (no trailing underscores) is accepted by GCC and
    // clang next to `__attribute__`.
    assert_eq!(struct_size_of("struct __attribute((packed)) S { char c; int x; };", "S"), (5, 1));
}

#[test]
fn alignof_incomplete_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S; int x = __alignof__(struct S);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "invalid application of `__alignof__` to an incomplete type");
}

#[test]
fn attribute_forward_declaration_carries_nothing() {
    // Verified against gcc: attributes on a mere forward declaration
    // do not reach the later definition.
    assert_eq!(struct_size_of("struct __attribute__((packed)) Fwd; struct Fwd { char c; int x; };", "Fwd"), (8, 4));
}

#[test]
fn attribute_aligned_non_power_of_two_panics() {
    // Verified against gcc (a hard error there too).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct N { char c; int x __attribute__((aligned(3))); };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "not a power of 2");
}

#[test]
fn attribute_aligned_non_constant_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int n; struct N { char c; int x __attribute__((aligned(n))); };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "needs an integer constant");
}

#[test]
fn attribute_aligned_zero_warns_not_fails() {
    // GCC warns and ignores `aligned(0)` — the parse succeeds.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct Z { char c; int x __attribute__((aligned(0))); };", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 0);
    assert_eq!(struct_size_of("struct Z { char c; int x; };", "Z"), (8, 4));
}

#[test]
fn case_labels_fold() {
    // Labels accept any integer constant expression — literals,
    // enumerators, arithmetic, and `sizeof`.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "enum { A = 3 }; int f(int x) { switch (x) { case A + 1: return 1; case sizeof(char): return 2; case 'a': return 3; default: return 0; } }",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    // The bare enum declares constants only — just the function shows.
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn non_constant_case_label_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int x, int y) { switch (x) { case y: return 1; default: return 0; } }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "case label is not an integer constant expression");
}

#[test]
fn switch_distinct_cases_accepted() {
    // Distinct values — including across nested switches, and labels
    // inside loops (Duff's device) — parse normally.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "int f(int x, int y) { switch (x) { case 1: return 1; switch (y) { case 1: return 2; case 2: return 3; } case 2: return 4; } return 0; }",
        &mut sema,
        &mut interner,
    );
    assert_eq!(parser.parse_translation_unit().external_declarations.len(), 1);

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "int f(int x) { switch (x) { case 1: while (x) { case 2: x--; } return 1; } return 0; }",
        &mut sema,
        &mut interner,
    );
    assert_eq!(parser.parse_translation_unit().external_declarations.len(), 1);
}

#[test]
fn duplicate_case_value_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "int f(int x) { switch (x) { case 1: return 1; case 1 + 0: return 2; } return 0; }",
        &mut sema,
        &mut interner,
    );
    parser.parse_translation_unit();
    assert_parse_error(&parser, "duplicate `case` value in `switch`");
}

#[test]
fn duplicate_case_after_conversion_panics() {
    // Labels compare after conversion to the promoted controlling
    // type (verified against gcc): `-1` duplicates `4294967295u`
    // under `unsigned`.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "int f(unsigned x) { switch (x) { case -1: return 1; case 4294967295u: return 2; } return 0; }",
        &mut sema,
        &mut interner,
    );
    parser.parse_translation_unit();
    assert_parse_error(&parser, "duplicate `case` value in `switch`");
}

#[test]
fn duplicate_default_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "int f(int x) { switch (x) { case 1: return 1; default: return 0; default: return 2; } }",
        &mut sema,
        &mut interner,
    );
    parser.parse_translation_unit();
    assert_parse_error(&parser, "duplicate `default` in `switch`");
}

#[test]
fn builtin_va_arg_takes_a_type() {
    // `va_arg(ap, TYPE)` spells a call but parses a type-name second
    // (the sqlite shape: `*__builtin_va_arg(ap, int*) = val;`).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "typedef __builtin_va_list va_list; int sum(int n, ...) { va_list ap; __builtin_va_start(ap, n); int x = __builtin_va_arg(ap, int); double d = __builtin_va_arg(ap, double); int *p = __builtin_va_arg(ap, int *); __builtin_va_end(ap); return x; }",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
}

#[test]
fn va_arg_without_va_list_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { int x = __builtin_va_arg(0, int); }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "`va_arg` needs a `va_list` argument");
}

#[test]
fn va_arg_with_void_type_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "typedef __builtin_va_list va_list; void f(va_list ap) { __builtin_va_arg(ap, void); }",
        &mut sema,
        &mut interner,
    );
    parser.parse_translation_unit();
    assert_parse_error(&parser, "invalid type in `va_arg`");
}

#[test]
fn generic_selection_resolves() {
    // `_Generic` picks the compatible type-name, else `default`; the
    // result carries the selected expression's type and folds.
    let checked = check_with(&[], "_Generic(0, int: 1, default: 2)");
    assert_eq!(checked.ty, Ty::Int);
    let checked = check_with(&[], "_Generic(0L, int: 1, default: 2)");
    assert_eq!(checked.ty, Ty::Int);
    let checked = check_with(&["const char *z;"], "_Generic(z, const char *: 1, default: 2)");
    assert_eq!(checked.ty, Ty::Int);
    assert_eq!(enum_value_of("enum { N = _Generic(0, int: 5, default: 6) };", "N"), 5);
    assert_eq!(enum_value_of("enum { N = _Generic(0L, int: 5, default: 6) };", "N"), 6);
    // The sqlite shape: a ternary controlling expression with pointer
    // associations compared against null.
    let checked = check_with(
        &["char *z;", "char *strchr(const char *, int);"],
        "_Generic(0 ? z : (void *)1, const void *: (const char *)(strchr(z, 45)), default: strchr(z, 45)) != 0",
    );
    assert_eq!(checked.ty, Ty::Int);
}

#[test]
fn generic_without_match_or_default_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x = _Generic(0L, int: 1);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "no matching generic association");
}

#[test]
fn generic_duplicate_default_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x = _Generic(0, default: 1, default: 2);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "duplicate `default` in `_Generic`");
}

#[test]
fn function_returning_function_pointer_defines_params() {
    // The sqlite shape: `f` is a function of `(int, char *)`
    // returning a function pointer — the *inner* suffix declares the
    // definition's parameters.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("static void (*f(int n, char *s))(void) { (void)n; (void)s; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    let [ExternalDeclaration::Func(id)] = tu.external_declarations.as_slice() else {
        panic!("expected one function")
    };
    let body = parser.sema.bodies.get(id).expect("body recorded");
    assert_eq!(body.params.len(), 2);
}

#[test]
fn builtin_sync_synchronize_declares_on_call() {
    // sqlite's memory barrier: an undeclared nullary builtin call.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("static void sqlite3MemoryBarrier(void) { __sync_synchronize(); }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn libc_builtins_rewrite_to_libc_calls() {
    // `__builtin_X()` for libc X calls X: the builtin name never
    // reaches checking (which would error), `abort`/`strlen` declare
    // implicitly instead.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "void f(void) { __builtin_abort(); __builtin_strlen(\"abc\"); __builtin_memcpy(((void *)0), ((void *)0), 0); }",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
    for name in ["abort", "strlen", "memcpy"] {
        let sym = parser.intern.get_or_intern(name);
        let id = parser.sema.lookup_symbol(sym).expect("implicit libc declaration");
        assert!(matches!(parser.sema.decls[id.0 as usize].kind, BindingKind::Func));
    }
    let builtin = parser.intern.get_or_intern("__builtin_abort");
    assert!(parser.sema.lookup_symbol(builtin).is_none());
}

#[test]
fn runtime_builtins_rewrite_to_runtime_calls() {
    // The setjmp builtins are calls into the assembly runtime,
    // declared with their real prototypes.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void *b) { if (__builtin_setjmp(b)) __builtin_longjmp(b, 1); }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
    for name in ["__glccrt_setjmp", "__glccrt_longjmp"] {
        let sym = parser.intern.get_or_intern(name);
        let id = parser.sema.lookup_symbol(sym).expect("runtime declaration");
        let binding = &parser.sema.decls[id.0 as usize];
        assert!(matches!(binding.kind, BindingKind::Func), "{name}");
    }
    for name in ["__builtin_setjmp", "__builtin_longjmp"] {
        let builtin = parser.intern.get_or_intern(name);
        assert!(parser.sema.lookup_symbol(builtin).is_none(), "{name} leaked");
    }
}

#[test]
fn frame_address_builtins_fold_their_level() {
    // `__builtin_frame_address`/`__builtin_return_address` check inline
    // into a `FrameAddress` node: the level folds (GCC requires a
    // constant integer) and the result is `void *`.
    let checked = check_with(&[], "__builtin_return_address(1 + 1)");
    assert!(matches!(checked.ty, Ty::Pointer { .. }));
    assert!(matches!(checked.kind, CheckedExprKind::FrameAddress { level: 2, returns_address: true }));
    let checked = check_with(&[], "__builtin_frame_address(0)");
    assert!(matches!(checked.kind, CheckedExprKind::FrameAddress { level: 0, returns_address: false }));
    assert!(check_with_err(&["int x;"], "__builtin_return_address(x)").contains("constant"));
}

#[test]
fn builtin_expect_drops_to_its_value() {
    // Branch prediction only: the call becomes its first argument.
    let checked = check_with(&["int x;"], "__builtin_expect(x, 1)");
    assert_eq!(checked.ty, Ty::Int);
    // The expected-value argument is still evaluated (pr85156):
    // `__builtin_expect(x, y++)` becomes `(y++, x)`.
    let checked = check_with(&["int x;", "int y;"], "__builtin_expect(x, y++)");
    assert_eq!(checked.ty, Ty::Int);
    let CheckedExprKind::Binary(BinaryOp::Comma, _, value) = &checked.kind else {
        panic!("expected a comma expression, got {:?}", checked.kind)
    };
    assert_eq!(value.ty, Ty::Int);
}

#[test]
fn builtin_clzg_checks_inline() {
    // Type-generic: the operand keeps its width (uchar stays uchar),
    // the fallback converts to `int`, the result is always `int`.
    let checked = check_with(&["unsigned x;"], "__builtin_clzg(x)");
    assert_eq!(checked.ty, Ty::Int);
    let checked = check_with(&["unsigned char c;"], "__builtin_clzg(c)");
    let CheckedExprKind::Clz { arg, fallback, .. } = &checked.kind else {
        panic!("expected a clz node")
    };
    assert_eq!(arg.ty, Ty::UnsignedChar);
    assert!(fallback.is_none());
    let checked = check_with(&["unsigned x;"], "__builtin_clzg(x, 5)");
    assert_eq!(checked.ty, Ty::Int);
    let msg = check_with_err(&["int x;"], "__builtin_clzg(x)");
    assert!(msg.contains("unsigned integer"), "{msg:?}");
    let msg = check_with_err(&["unsigned x;"], "__builtin_clzg(x, 1, 2)");
    assert!(msg.contains("optional"), "{msg:?}");
}

#[test]
fn builtin_unreachable_rewrites_to_abort() {
    // Abnormal termination if ever reached: the builtin name never
    // reaches checking, `abort` declares implicitly instead.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int x) { if (x) return 1; __builtin_unreachable(); return 0; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
    let abort = parser.intern.get_or_intern("abort");
    assert!(parser.sema.lookup_symbol(abort).is_some());
    let builtin = parser.intern.get_or_intern("__builtin_unreachable");
    assert!(parser.sema.lookup_symbol(builtin).is_none());
}

#[test]
fn builtin_math_calls_check() {
    // Direct builtin math calls (sqlite's shell): declared on use
    // with their C library signatures.
    let checked = check_with(&[], "__builtin_ceil(1.5)");
    assert_eq!(checked.ty, Ty::Double);
    let checked = check_with(&[], "__builtin_floor(1)");
    assert_eq!(checked.ty, Ty::Double);
    let checked = check_with(&[], "__builtin_nanf(\"\")");
    assert_eq!(checked.ty, Ty::Float);
    let checked = check_with(&[], "__builtin_nan(\"\")");
    assert_eq!(checked.ty, Ty::Double);
    let checked = check_with(&[], "__builtin_nanl(\"\")");
    assert_eq!(checked.ty, Ty::LongDouble);
    let checked = check_with(&[], "__builtin_inf()");
    assert_eq!(checked.ty, Ty::Double);
    let checked = check_with(&[], "__builtin_inff()");
    assert_eq!(checked.ty, Ty::Float);
    let checked = check_with(&[], "__builtin_infl()");
    assert_eq!(checked.ty, Ty::LongDouble);
    let checked = check_with(&[], "__builtin_huge_val()");
    assert_eq!(checked.ty, Ty::Double);
    let checked = check_with(&[], "__builtin_huge_valf()");
    assert_eq!(checked.ty, Ty::Float);
    let checked = check_with(&[], "__builtin_huge_vall()");
    assert_eq!(checked.ty, Ty::LongDouble);
}

#[test]
fn builtin_math_calls_rewrite_to_libc_names() {
    // `__builtin_log10` and its width variants are exactly their libc
    // counterparts: the callee rewrites, and the implicit declaration
    // carries the real prototype (the `l` form returns `long double`,
    // not `int`).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "double f(double x){return __builtin_log10(x);}\
         float g(float x){return __builtin_log10f(x);}\
         long double h(long double x){return __builtin_log10l(x);}",
        &mut sema,
        &mut interner,
    );
    parser.parse_translation_unit();
    for (name, ret) in [("log10", Ty::Double), ("log10f", Ty::Float), ("log10l", Ty::LongDouble)] {
        let sym = parser.intern.get_or_intern(name);
        let id = parser.sema.lookup_symbol(sym).unwrap_or_else(|| panic!("{name} should be declared implicitly"));
        let binding = &parser.sema.decls[id.0 as usize];
        let Ty::Func { ret: got, params, .. } = &binding.ty else {
            panic!("{name} should be a function")
        };
        assert_eq!(**got, ret, "{name} return type");
        assert_eq!(params.as_slice(), &[ret], "{name} parameter");
    }
    let builtin = parser.intern.get_or_intern("__builtin_log10");
    assert!(parser.sema.lookup_symbol(builtin).is_none(), "builtin name must not be declared");
}

#[test]
fn builtin_isnan_isinf_check() {
    // Classification builtins (libstdc++ math): single evaluation
    // with an `int` result; integer arguments fold to `0` at once.
    let checked = check_with(&[], "__builtin_isnan(1.5)");
    assert_eq!(checked.ty, Ty::Int);
    let checked = check_with(&[], "__builtin_isnanf(1.5f)");
    assert_eq!(checked.ty, Ty::Int);
    let checked = check_with(&[], "__builtin_isnanl(1.5L)");
    assert_eq!(checked.ty, Ty::Int);
    let checked = check_with(&[], "__builtin_isinf(1.5)");
    assert_eq!(checked.ty, Ty::Int);
    let checked = check_with(&[], "__builtin_isinfl(1.5L)");
    assert_eq!(checked.ty, Ty::Int);
    let checked = check_with(&[], "__builtin_isnan(1)");
    assert!(matches!(checked.kind, CheckedExprKind::ConstantP(false)));
    let checked = check_with(&[], "__builtin_isinf(1)");
    assert!(matches!(checked.kind, CheckedExprKind::ConstantP(false)));
}

#[test]
fn builtin_signbit_check() {
    // `signbit` reads the sign bit, so integers never fold away: the
    // value converts to `double` first (`signbit(-1)` is true).
    let checked = check_with(&[], "__builtin_signbit(-0.0)");
    assert_eq!(checked.ty, Ty::Int);
    assert!(matches!(
        checked.kind,
        CheckedExprKind::FpClass {
            op: crate::check::FpPredicate::SignBit,
            ..
        }
    ));
    let checked = check_with(&[], "__builtin_signbitf(1.5f)");
    assert!(matches!(
        checked.kind,
        CheckedExprKind::FpClass {
            op: crate::check::FpPredicate::SignBit,
            ..
        }
    ));
    let checked = check_with(&[], "__builtin_signbitl(1.5L)");
    assert!(matches!(
        checked.kind,
        CheckedExprKind::FpClass {
            op: crate::check::FpPredicate::SignBit,
            ..
        }
    ));
    let checked = check_with(&[], "__builtin_signbit(-1)");
    let CheckedExprKind::FpClass { arg, .. } = &checked.kind else {
        panic!("expected a signbit node")
    };
    assert_eq!(arg.ty, Ty::Double);
}

#[test]
fn builtin_fp_classifications_check() {
    use crate::check::FpPredicate;
    // `isfinite`/`isnormal`/`isinf_sign` are what `<math.h>` expands
    // its macros to; each is a single-evaluation `int` node.
    for (src, op) in [
        ("__builtin_isinf_sign(1.5)", FpPredicate::IsInfSign),
        ("__builtin_isfinite(1.5)", FpPredicate::IsFinite),
        ("__builtin_isnormal(1.5f)", FpPredicate::IsNormal),
    ] {
        let checked = check_with(&[], src);
        assert_eq!(checked.ty, Ty::Int);
        assert!(matches!(&checked.kind, CheckedExprKind::FpClass { op: got, .. } if *got == op), "{src}");
    }
    // `fpclassify` carries the five codes and a floating operand.
    let checked = check_with(&[], "__builtin_fpclassify(0, 1, 4, 3, 2, 1.5)");
    assert_eq!(checked.ty, Ty::Int);
    let CheckedExprKind::FpClassify { codes, arg } = &checked.kind else {
        panic!("expected a fpclassify node")
    };
    assert_eq!(codes.len(), 5);
    assert_eq!(arg.ty, Ty::Double);
    // The operand must be floating point: an integer has a constant
    // category (GCC rejects it).
    let msg = check_with_err(&[], "__builtin_fpclassify(0, 1, 4, 3, 2, 1)");
    assert!(msg.contains("floating-point"), "{msg:?}");
    let msg = check_with_err(&[], "__builtin_fpclassify(0, 1, 4, 3, 2)");
    assert!(msg.contains("five"), "{msg:?}");
}

#[test]
fn builtin_classify_type_folds() {
    // GCC's type-class codes, folded from the argument's type alone
    // (the argument never evaluates).
    fn class(decls: &[&str], src: &str) -> u8 {
        match check_with(decls, src).kind {
            CheckedExprKind::ClassifyType(code) => code,
            other => panic!("expected a classify-type node, got {other:?}"),
        }
    }
    assert_eq!(class(&[], "__builtin_classify_type(1)"), 1);
    assert_eq!(class(&[], "__builtin_classify_type('a')"), 1);
    assert_eq!(class(&[], "__builtin_classify_type(1.5)"), 8);
    assert_eq!(class(&[], "__builtin_classify_type(1.5f)"), 8);
    assert_eq!(class(&[], "__builtin_classify_type(1.5L)"), 8);
    assert_eq!(class(&[], "__builtin_classify_type(\"x\")"), 14);
    assert_eq!(class(&[], "__builtin_classify_type((void)0)"), 0);
    assert_eq!(class(&[], "__builtin_classify_type((void *)0)"), 5);
    assert_eq!(class(&["struct S { int x; } s;"], "__builtin_classify_type(s)"), 12);
    assert_eq!(class(&["union U { int x; } u;"], "__builtin_classify_type(u)"), 13);
    let msg = check_with_err(&[], "__builtin_classify_type()");
    assert!(msg.contains("exactly one"), "{msg:?}");
}

#[test]
fn builtin_prefetch_is_a_void_hint() {
    // A hint: the address still evaluates (through a `void` cast), and
    // the access/locality arguments are checked and dropped.
    let checked = check_with(&["int *p;"], "__builtin_prefetch(p)");
    assert_eq!(checked.ty, Ty::Void);
    let CheckedExprKind::Cast { ty, .. } = &checked.kind else {
        panic!("expected a void cast")
    };
    assert_eq!(*ty, Ty::Void);
    let checked = check_with(&["int *p;"], "__builtin_prefetch(p, 1, 3)");
    assert_eq!(checked.ty, Ty::Void);
    let checked = check_with(&["int a[4];"], "__builtin_prefetch(a + 1, 0)");
    assert_eq!(checked.ty, Ty::Void);
    let msg = check_with_err(&["int x;"], "__builtin_prefetch(x)");
    assert!(msg.contains("pointer"), "{msg:?}");
    let msg = check_with_err(&["int *p;"], "__builtin_prefetch(p, 1, 2, 3)");
    assert!(msg.contains("three"), "{msg:?}");
    let msg = check_with_err(&[], "__builtin_prefetch()");
    assert!(msg.contains("address"), "{msg:?}");
}

#[test]
fn builtin_alloca_check() {
    // The size converts to `size_t`; the result is `void *`.
    let checked = check_with(&[], "__builtin_alloca(16)");
    assert!(matches!(checked.ty.unqualified(), Ty::Pointer { pointee, .. } if pointee.unqualified() == &Ty::Void));
    let CheckedExprKind::Alloca { size } = &checked.kind else {
        panic!("expected an alloca node")
    };
    assert_eq!(size.ty, Ty::UnsignedLong);
    // The plain name is the builtin too, declared or not.
    let checked = check_with(&["extern void *alloca(unsigned long);"], "alloca(sizeof(int))");
    assert!(matches!(checked.kind, CheckedExprKind::Alloca { .. }));
    let msg = check_with_err(&[], "alloca()");
    assert!(msg.contains("exactly one"), "{msg:?}");
    let msg = check_with_err(&[], "alloca(1, 2)");
    assert!(msg.contains("exactly one"), "{msg:?}");
}

#[test]
fn builtin_overflow_check() {
    // The pointed-to type fixes the result width and signedness; the
    // operands keep their own.
    let checked = check_with(&["int r;"], "__builtin_add_overflow(1, 2, &r)");
    assert_eq!(checked.ty, Ty::Int);
    let CheckedExprKind::Overflow { op, res_ty, .. } = &checked.kind else {
        panic!("expected an overflow node")
    };
    assert_eq!(*op, crate::check::OverflowOp::Add);
    assert_eq!(*res_ty, Ty::Int);
    let checked = check_with(&["long r;"], "__builtin_mul_overflow(1, 2L, &r)");
    let CheckedExprKind::Overflow { op, res_ty, .. } = &checked.kind else {
        panic!("expected an overflow node")
    };
    assert_eq!(*op, crate::check::OverflowOp::Mul);
    assert_eq!(*res_ty, Ty::Long);
    let checked = check_with(&["unsigned char r;"], "__builtin_sub_overflow(1, 2, &r)");
    assert!(matches!(
        checked.kind,
        CheckedExprKind::Overflow {
            op: crate::check::OverflowOp::Sub,
            ..
        }
    ));
    // The result pointer may be any pointer expression.
    let checked = check_with(&["int a[2];"], "__builtin_add_overflow(1, 2, a + 1)");
    assert!(matches!(checked.kind, CheckedExprKind::Overflow { .. }));
    let msg = check_with_err(&[], "__builtin_add_overflow(1, 2)");
    assert!(msg.contains("result pointer"), "{msg:?}");
    let msg = check_with_err(&["double d;"], "__builtin_add_overflow(1, 2, &d)");
    assert!(msg.contains("pointer to integer"), "{msg:?}");
    let msg = check_with_err(&["int r;"], "__builtin_add_overflow(1.5, 2, &r)");
    assert!(msg.contains("integers"), "{msg:?}");
}

#[test]
fn typed_overflow_builtins_check() {
    // The typed GCC spellings pin the operand type: `sadd` is `int`,
    // `saddl` is `long`, `umulll` is `unsigned long long`, ...
    let checked = check_with(&["int r;"], "__builtin_sadd_overflow(1, 2, &r)");
    let CheckedExprKind::Overflow { op, lhs, res_ty, .. } = &checked.kind else {
        panic!("expected an overflow node")
    };
    assert_eq!(*op, crate::check::OverflowOp::Add);
    assert_eq!(*res_ty, Ty::Int);
    assert_eq!(lhs.ty, Ty::Int);
    // Operands convert to the pinned type like prototyped parameters.
    let checked = check_with(&["int r;"], "__builtin_sadd_overflow('a', 2L, &r)");
    let CheckedExprKind::Overflow { lhs, rhs, .. } = &checked.kind else {
        panic!("expected an overflow node")
    };
    assert_eq!(lhs.ty, Ty::Int);
    assert_eq!(rhs.ty, Ty::Int);
    let checked = check_with(&["long r;"], "__builtin_ssubl_overflow(1, 2, &r)");
    let CheckedExprKind::Overflow { op, res_ty, .. } = &checked.kind else {
        panic!("expected an overflow node")
    };
    assert_eq!(*op, crate::check::OverflowOp::Sub);
    assert_eq!(*res_ty, Ty::Long);
    let checked = check_with(&["unsigned long long r;"], "__builtin_umulll_overflow(1, 2, &r)");
    let CheckedExprKind::Overflow { op, res_ty, .. } = &checked.kind else {
        panic!("expected an overflow node")
    };
    assert_eq!(*op, crate::check::OverflowOp::Mul);
    assert_eq!(*res_ty, Ty::UnsignedLongLong);
    // Only the two-operand-and-pointer shape is an overflow builtin.
    let msg = check_with_err(&[], "__builtin_sadd_overflow(1)");
    assert!(msg.contains("result pointer"), "{msg:?}");
    let msg = check_with_err(&["int r;"], "__builtin_sadd_overflow(1.5, 2, &r)");
    assert!(msg.contains("integers"), "{msg:?}");
    // Misspelled suffixes are ordinary (undeclared) functions.
    let msg = check_with_err(&["int r;"], "__builtin_saddx_overflow(1, 2, &r)");
    assert!(msg.contains("undeclared"), "{msg:?}");
}

#[test]
fn atomic_builtins_check_against_arguments() {
    // sqlite's memory-status counters: `__atomic_load_n` returns the
    // pointed-to type, `__atomic_store_n` converts the value like an
    // assignment and returns `void`.
    let checked = check_with(&["int n;"], "__atomic_load_n(&n, 0)");
    assert_eq!(checked.ty, Ty::Int);
    let checked = check_with(&["long n;"], "__atomic_store_n(&n, 1, 0)");
    assert_eq!(checked.ty, Ty::Void);
}

#[test]
fn atomic_load_without_pointer_panics() {
    assert!(check_with_err(&["int n;"], "__atomic_load_n(n, 0)").contains("`__atomic_load_n` needs a pointer argument"));
}

#[test]
fn c11_atomic_builtins_check_against_arguments() {
    // The clang C11 spellings `<stdatomic.h>` expands to: the pointer
    // gives the value type, the `_n`-less shapes match the GCC
    // families, and `compare_exchange` drops the weak flag (strong vs
    // weak is in the name).
    let checked = check_with(&["int n;"], "__c11_atomic_load(&n, 5)");
    assert_eq!(checked.ty, Ty::Int);
    let checked = check_with(&["int n;"], "__c11_atomic_store(&n, 1, 5)");
    assert_eq!(checked.ty, Ty::Void);
    let checked = check_with(&["int n;"], "__c11_atomic_init(&n, 1)");
    assert_eq!(checked.ty, Ty::Void);
    let checked = check_with(&["int n;"], "__c11_atomic_fetch_add(&n, 1, 5)");
    assert_eq!(checked.ty, Ty::Int);
    let checked = check_with(&["int n;", "int e;"], "__c11_atomic_compare_exchange_strong(&n, &e, 1, 5, 5)");
    assert_eq!(checked.ty, Ty::Bool);
    let checked = check_with(&[], "__c11_atomic_is_lock_free(4)");
    assert_eq!(checked.ty, Ty::Bool);
}

#[test]
fn interlocked_compare_exchange_checks_against_arguments() {
    // MSVC `<intrin.h>` atomics: the pointee gives the operand type
    // and the result is the object's initial value, so the `8`/`16`/
    // `64`/`Pointer` widths share the shape.
    let checked = check_with(&["int n;"], "_InterlockedCompareExchange(&n, 1, 0)");
    assert_eq!(checked.ty, Ty::Int);
    let checked = check_with(&["char n;"], "_InterlockedCompareExchange8(&n, 1, 0)");
    assert_eq!(checked.ty, Ty::Char);
    let checked = check_with(&["int *p;"], "_InterlockedCompareExchangePointer((void *volatile *)&p, 0, 0)");
    assert!(matches!(checked.ty, Ty::Pointer { .. }));
}

#[test]
fn interlocked_compare_exchange_without_pointer_panics() {
    assert!(check_with_err(&["int n;"], "_InterlockedCompareExchange(n, 1, 0)").contains("needs a pointer argument"));
}

#[test]
fn c11_atomic_load_without_pointer_panics() {
    assert!(check_with_err(&["int n;"], "__c11_atomic_load(n, 5)").contains("`__c11_atomic_load` needs a pointer argument"));
}

#[test]
fn c11_atomic_min_on_pointer_panics() {
    assert!(check_with_err(&["int *p;"], "__c11_atomic_fetch_min(&p, 1, 5)").contains("`__c11_atomic_fetch_min` needs an integer target"));
}

#[test]
fn declared_name_scopes_over_its_initializer() {
    // `int *p = malloc(sizeof(*p))` sees `p` (C17 §6.2.1p4: scope
    // starts after the declarator) — the sqlite idiom.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "typedef unsigned long size_t; void *malloc(size_t); struct S { int x; }; struct S *p = (struct S *)malloc(sizeof(*p)); int x = x + 0;",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 4);
}

#[test]
fn builtin_offsetof_folds() {
    // `__builtin_offsetof` resolves designators against the layout and
    // folds to a constant (usable anywhere one is).
    assert_eq!(enum_value_of("struct P { char c; int x; }; enum { A = __builtin_offsetof(struct P, x) };", "A"), 4);
    assert_eq!(check_with(&["struct P { char c; int x; };"], "__builtin_offsetof(struct P, x)").ty, Ty::UnsignedLong);
    // Through anonymous members and array elements.
    assert_eq!(
        enum_value_of("struct R { union { int a; char b[4]; }; }; enum { N = __builtin_offsetof(struct R, b[2]) };", "N"),
        2
    );
}

#[test]
fn offsetof_bit_field_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int x : 3; }; int y = __builtin_offsetof(struct S, x);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "`offsetof` of bit-field");
}

#[test]
fn statement_expression_value_is_last_item() {
    // A statement expression runs block items for real (own scope,
    // checked inline) and takes the last expression statement's type.
    assert_eq!(first_value_type_of("int f(void) { return ({ 1; }) + 2; }"), Ty::Int);
    assert_eq!(first_value_type_of("int f(void) { return ({ int y = 5; y + 1; }); }"), Ty::Int);
    assert_eq!(first_value_type_of("void f(void) { ({ int y; }); }"), Ty::Void);
    assert_eq!(first_value_type_of("int f(void) { return __extension__ ({ 1; }); }"), Ty::Int);
    // A trailing label does not hide the value (GCC: the value of the
    // last statement; tcctest.c takes `&&label` this way).
    assert_eq!(first_value_type_of("int f(void) { return ({ l: 42; }); }"), Ty::Int);
    assert_eq!(first_value_type_of("void f(void) { ({ a: b: (char)42; }); }"), Ty::Char);
    // Declarations inside stay scoped.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void) { return ({ int y = 1; }), y; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert!(parser.has_errors());
}

#[test]
fn file_scope_statement_expression_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x = ({ 1; });", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "braced-group within expression allowed only inside a function");
}

// ---- control-flow validity ----

#[test]
fn goto_and_labels_resolve() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int x) { goto done; x++; done: return x; } void g(void) { goto l; l: ; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
}

#[test]
fn label_named_like_a_typedef_resolves_as_label() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("typedef int t; int f(void) { t x = 1; t: x++; if (x < 3) goto t; return x; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert!(sema.label_ids.keys().any(|sym| interner.resolve(*sym) == Some("t")));
}

#[test]
fn undefined_label_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { goto missing; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "label 'missing' used but not defined");
}

#[test]
fn duplicate_label_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { l: goto l; l: ; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "duplicate label 'l'");
}

#[test]
fn label_address_and_computed_goto_parse() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { void *p = &&l; goto *p; l: ; }", &mut sema, &mut interner);
    let (_, body) = parse_one_body("label-addr", &mut parser);
    let [BlockItem::Decl { ids, .. }, BlockItem::Stmt(goto), BlockItem::Stmt(label)] = body.block.items.as_slice() else {
        panic!("{body:?}")
    };
    assert!(matches!(goto, Stmt::GotoComputed { .. }));
    assert!(matches!(label, Stmt::Label { .. }));
    let Initializer::Expr(e) = &sema.inits[&ids[0]] else { panic!("{ids:?}") };
    assert!(matches!(&e.kind, CheckedExprKind::LabelAddr(_)));
    assert_eq!(sema.label_ids.len(), 1);
    assert_eq!(sema.label_ids.values().copied().collect::<Vec<_>>(), [0]);
}

#[test]
fn label_ids_run_in_definition_order() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { a: b: c: ; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let mut ids: Vec<u64> = sema.label_ids.values().copied().collect();
    ids.sort();
    assert_eq!(ids, [0, 1, 2]);
}

#[test]
fn label_inside_statement_expression_resolves() {
    // A statement expression is its own label scope: a `goto` to a
    // label defined beside it resolves inside, and does not escape.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { ({ l: goto l; }); }", &mut sema, &mut interner);
    parser.parse_translation_unit();
}

#[test]
fn label_inside_statement_expression_is_local() {
    // Jumping into a statement expression from outside resolves to
    // nothing, like GCC.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { goto l; ({ l: ; }); }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "label 'l' used but not defined");
}

#[test]
fn gnu_label_declaration_is_consumed() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { __label__ a, b; a: goto a; }", &mut sema, &mut interner);
    let (_, body) = parse_one_body("gnu-label", &mut parser);
    let [empty, label] = body.block.items.as_slice() else { panic!("{body:?}") };
    assert!(matches!(empty, BlockItem::Stmt(Stmt::Empty { .. })));
    assert!(matches!(label, BlockItem::Stmt(Stmt::Label { .. })));
}

#[test]
fn break_outside_loop_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { if (1) break; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "break statement not within loop or switch");
}

#[test]
fn continue_outside_loop_panics() {
    // ... including directly in a `switch` (which is not a loop).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int x) { switch (x) { case 1: continue; default: return 0; } }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "continue statement not within a loop");
}

#[test]
fn break_in_switch_and_loops_parses() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "int f(int x) { while (x) { if (x > 2) break; else continue; } switch (x) { case 1: break; default: break; } return x; }",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn case_outside_switch_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { case 1: ; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "case label not within a switch statement");
}

#[test]
fn statement_expression_jumps_stay_scoped() {
    // Jumping out of a statement expression resolves outside (like
    // GCC); jumping in has no visible label and fails.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int x) { int y = ({ if (x) goto out; 0; }); return y; out: return 1; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);

    // `break` still binds a `switch` inside the expression.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "int f(void) { int y = ({ switch (1) { case 1: break; default: break; } 0; }); return y; }",
        &mut sema,
        &mut interner,
    );
    assert_eq!(parser.parse_translation_unit().external_declarations.len(), 1);
}

#[test]
fn jump_into_statement_expression_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void) { goto inner; int y = ({ inner: 1; 0; }); return y; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "used but not defined");
}

/// The type of the first valued statement (`return expr;` or a lone
/// expression) in a single-function source: statement-expression
/// shapes need function context, since braced groups are rejected
/// outside functions.
fn first_value_type_of(src: &str) -> Ty {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(src, &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    let [ExternalDeclaration::Func(id)] = tu.external_declarations.as_slice() else {
        panic!("expected one function");
    };
    let body = parser.sema.bodies.get(id).expect("body recorded");
    for item in &body.block.items {
        if let BlockItem::Stmt(Stmt::Return { expr: Some(expr), .. }) = item {
            return expr.ty.clone();
        }
        if let BlockItem::Stmt(Stmt::Expr { expr, .. }) = item {
            return expr.ty.clone();
        }
    }
    panic!("no valued statement in {src:?}");
}

#[test]
fn constant_if_drops_the_untaken_branch() {
    use crate::parse::{BlockItem, Stmt};
    // A constant condition removes the branch that can never run, so
    // dead calls (`link_error()` in the torture suite) vanish. A
    // branch holding a label stays: `goto` can jump into it.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "void f(int x) { if (0) x = 1; else x = 2; if (1) x = 3; else x = 4; if (0) { lbl: x = 5; } goto lbl; }",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
    let f = parser.intern.get_or_intern("f");
    let id = parser.sema.lookup_symbol(f).expect("f declared");
    let body = parser.sema.bodies.get(&id).expect("body recorded");
    let [BlockItem::Stmt(t0), BlockItem::Stmt(t1), BlockItem::Stmt(t2), ..] = body.block.items.as_slice() else {
        panic!("expected three ifs, got {:?}", body.block.items)
    };
    // The never-taken then empties; the else stays the else.
    let Stmt::If { then, els: Some(_), .. } = t0 else {
        panic!("expected an if with an else, got {t0:?}")
    };
    assert!(matches!(&**then, Stmt::Empty { .. }), "dead then must empty, got {then:?}");
    // The never-taken else is dropped outright.
    let Stmt::If { els: None, .. } = t1 else {
        panic!("expected the else to be dropped, got {t1:?}")
    };
    // The label keeps the dead branch alive for `goto`.
    let Stmt::If { then, .. } = t2 else { panic!("expected an if, got {t2:?}") };
    assert!(matches!(&**then, Stmt::Compound(_)), "labeled dead branch must survive, got {then:?}");
}

#[test]
fn func_name_identifiers_expand() {
    use crate::check::CheckedExprKind;
    use crate::parse::{BlockItem, Stmt};
    // `__func__` (and friends) expand to the enclosing function's name
    // as a string literal; outside a function they stay undeclared.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "const char *f(void) { return __func__; } const char *g(void) { return __PRETTY_FUNCTION__; }",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
    for (fname, sname) in [("f", "f"), ("g", "g")] {
        let f = parser.intern.get_or_intern(fname);
        let id = parser.sema.lookup_symbol(f).unwrap();
        let body = parser.sema.bodies.get(&id).expect("body recorded");
        let [BlockItem::Stmt(Stmt::Return { expr: Some(expr), .. })] = body.block.items.as_slice() else {
            panic!("expected a single return");
        };
        // The array decays to a pointer, then converts to the `const`
        // return type (the conversion node exists precisely because
        // the `const` is modeled instead of erased).
        assert_eq!(
            expr.ty,
            Ty::Pointer {
                qualifiers: vec![],
                size: PtrSize::Default,
                pointee: Box::new(Ty::qualified(vec![TypeQualifier::Const], Ty::Char)),
            }
        );
        let CheckedExprKind::Convert { expr, .. } = &expr.kind else {
            panic!("expected a conversion, got {:?}", expr.kind);
        };
        let CheckedExprKind::Str(parts) = &expr.kind else {
            panic!("expected a string literal, got {:?}", expr.kind);
        };
        assert_eq!(parts.len(), 1);
        assert_eq!(parser.intern.resolve(parts[0].spelling).unwrap(), format!("\"{sname}\""));
    }
    let msg = check_with_err(&[], "__func__");
    assert!(msg.contains("undeclared"), "{msg:?}");
}

// ---- base type qualifiers are modeled, not erased ----

#[test]
fn const_object_initializes_and_reads() {
    // Initialization is not assignment: `const` objects initialize
    // normally and read through every shape (members, pointers,
    // subscripts, calls).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "const int x = 5; struct S { int m; }; const struct S s = { 1 }; const int a[3] = { 1, 2, 3 }; const char *p = \"ab\"; int f(void) { return x + s.m + a[1] + p[0]; }",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    // The tag declaration yields nothing: x, s, a, p, f.
    assert_eq!(tu.external_declarations.len(), 5);
}

#[test]
fn const_qualified_spellings_keep_qualifiers() {
    // `__const`/`const`/`volatile` wrap the base type (sorted,
    // deduplicated); declarator operators build outside.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("const volatile int x;", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    assert_eq!(
        parser.sema.decls[id.0 as usize].ty,
        Ty::qualified(vec![TypeQualifier::Const, TypeQualifier::Volatile], Ty::Int)
    );
    // `const char *p` is a pointer to const char (the dwm shape).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("const char *p;", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    let Ty::Pointer { pointee, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected a pointer");
    };
    assert_eq!(**pointee, Ty::qualified(vec![TypeQualifier::Const], Ty::Char));
    // ... while `char *const p` is a const pointer instead.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("char *const p;", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    let Ty::Pointer { qualifiers, pointee, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected a pointer");
    };
    assert!(matches!(qualifiers.as_slice(), [TypeQualifier::Const]));
    assert_eq!(**pointee, Ty::Char);
}

#[test]
fn typedef_const_distributes() {
    // `const` on an array typedef qualifies the elements, on a
    // pointer typedef the pointer itself — and on a function typedef
    // it vanishes, silently like GCC.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("typedef int A[3]; const A a;", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
    let a = parser.intern.get_or_intern("a");
    let id = parser.sema.lookup_symbol(a).unwrap();
    let Ty::Array { elem, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected an array");
    };
    assert_eq!(**elem, Ty::qualified(vec![TypeQualifier::Const], Ty::Int));

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("typedef int *P; const P p = (void *)0; int x; void f(void) { *p = x; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 4);

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("typedef void F(void); const F f;", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    assert!(matches!(parser.sema.decls[id.0 as usize].ty, Ty::Func { .. }));
}

#[test]
fn const_assignment_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("const int x = 5; void f(void) { x = 6; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "assignment of read-only variable 'x'");
}

#[test]
fn const_compound_assignment_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("const int x = 5; void f(void) { x += 1; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "assignment of read-only variable 'x'");
}

#[test]
fn const_increment_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("const int x = 5; void f(void) { x++; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "assignment of read-only variable 'x'");
}

#[test]
fn array_increment_panics() {
    // Arrays decay to pointers but are never modifiable lvalues.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a[3]; void f(void) { a++; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "operand of `++`/`--` must be an lvalue");
}

#[test]
fn const_pointer_target_assignment_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("const int *p; void f(void) { *p = 1; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "assignment of read-only location");
}

#[test]
fn const_struct_member_assignment_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int m; }; const struct S s = { 1 }; void f(void) { s.m = 2; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "assignment of read-only location");
}

#[test]
fn const_struct_pointer_member_assignment_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int m; }; void f(const struct S *p) { p->m = 2; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "assignment of read-only location");
}

#[test]
fn const_array_element_assignment_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("const int a[3] = { 1, 2, 3 }; void f(void) { a[0] = 5; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "assignment of read-only location");
}

#[test]
fn volatile_reads_and_writes() {
    // `volatile` never blocks access (unlike `const`).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("volatile int v; int f(void) { v = 1; return v; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
}

#[test]
fn pointer_qualifier_conversion_adds_silently() {
    // Adding `const` to the pointee converts silently (the `execvp`
    // shape); string literals decay into it too.
    let checked = check_with(&["char *s;", "const char *t;", "void execvp(const char *, char *const *);"], "execvp(t, &s)");
    assert_eq!(checked.ty, Ty::Void);
    let checked = check_with(&["const char *t;"], "t = \"ab\"");
    assert!(matches!(checked.ty, Ty::Pointer { .. }));
}

#[test]
fn pointer_qualifier_conversion_deep_add_warns() {
    // Adding qualifiers below the top level is a GCC warning — the
    // parse succeeds (warnings render to stderr).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("char **p; const char **q = p;", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
}

#[test]
fn pointer_qualifier_drop_warns_not_fails() {
    // Dropping `const` (`char *` from `const char *`) is a GCC warning,
    // accepted like the torture suite (built with `-w`) expects.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int *p; const int *q = p; void f(void) { p = q; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 3);
}

#[test]
fn ternary_pointer_mix_composes_const() {
    // `cond ? char_ptr : const_char_ptr` is `const char *`, like GCC.
    let checked = check_with(&["char *a;", "const char *b;", "int c;"], "c ? a : b");
    assert_eq!(
        checked.ty,
        Ty::Pointer {
            qualifiers: vec![],
            size: PtrSize::Default,
            pointee: Box::new(Ty::qualified(vec![TypeQualifier::Const], Ty::Char)),
        }
    );
}

#[test]
fn ternary_void_ptr_mix_composes_silently() {
    // `void *` mixes compose to `void *` (qualifiers unioned) with no
    // "pointer type mismatch" warning, like GCC (C17 §6.5.15).
    let void_ptr = Ty::Pointer {
        qualifiers: vec![],
        size: PtrSize::Default,
        pointee: Box::new(Ty::Void),
    };
    let const_void_ptr = Ty::Pointer {
        qualifiers: vec![],
        size: PtrSize::Default,
        pointee: Box::new(Ty::qualified(vec![TypeQualifier::Const], Ty::Void)),
    };
    let cases: &[(&[&str], &str, Ty)] = &[
        (&["void *v;", "int *p;", "int c;"], "c ? v : p", void_ptr.clone()),
        (&["void *v;", "int *p;", "int c;"], "c ? p : v", void_ptr.clone()),
        (&["const void *v;", "int *p;", "int c;"], "c ? v : p", const_void_ptr.clone()),
        (&["const void *v;", "int *p;", "int c;"], "c ? p : v", const_void_ptr.clone()),
        (&["void *v;", "const int *p;", "int c;"], "c ? v : p", const_void_ptr.clone()),
        (&["void *v;", "const int *p;", "int c;"], "c ? p : v", const_void_ptr.clone()),
    ];
    for (decls, expr_src, want) in cases {
        let mut sema = Sema::new();
        let mut interner = Interner::new();
        let src = format!("{} {expr_src}", decls.join(" "));
        let mut parser = test_parser(&src, &mut sema, &mut interner);
        for _ in *decls {
            parser.parse_declaration();
        }
        let expr = parser.parse_expression();
        let checked = parser
            .sema
            .check_expr(&expr, &parser.intern)
            .unwrap_or_else(|err| panic!("check failed for {expr_src:?}: {}", err.message));
        assert_eq!(checked.ty, *want, "for {expr_src:?}");
        assert!(
            parser.sema.warnings.borrow().is_empty(),
            "unexpected warning for {expr_src:?}: {:?}",
            parser.sema.warnings.borrow()
        );
    }
}

#[test]
fn ternary_null_constant_keeps_the_other_pointer_type() {
    // `(void *)0` is a null pointer constant (C17 §6.3.2.3p3), so the
    // result takes the other arm's pointer type (C17 §6.5.15p7)
    // before any `void *` composition; tcctest.c's
    // `(i == 0 ? (void *)0 : s)->i` needs this.
    let int_ptr = Ty::Pointer {
        qualifiers: vec![],
        size: PtrSize::Default,
        pointee: Box::new(Ty::Int),
    };
    let const_void_ptr = Ty::Pointer {
        qualifiers: vec![],
        size: PtrSize::Default,
        pointee: Box::new(Ty::qualified(vec![TypeQualifier::Const], Ty::Void)),
    };
    let decls = &["int *q;", "int c;"];
    for (expr_src, want) in [
        ("c ? (void *)0 : q", int_ptr.clone()),
        ("c ? q : (void *)0", int_ptr.clone()),
        // A qualified `void *` is not a null pointer constant: the
        // `void *` composite still applies, like GCC.
        ("c ? (const void *)0 : q", const_void_ptr.clone()),
        ("c ? (void *)0 : (const void *)0", const_void_ptr.clone()),
    ] {
        let checked = check_with(decls, expr_src);
        assert_eq!(checked.ty, want, "for {expr_src:?}");
    }
}

#[test]
fn enum_pointer_mixes_with_its_repr() {
    // An enum is compatible with its representation integer type
    // (C17 §6.7.2.2p4), so `int *` and `enum e *` compose and assign
    // silently, like GCC.
    let decls = &["enum e { a = -2147483647 - 1 };", "int *p;", "enum e *q;"];
    let checked = check_with(decls, "1 ? q : p");
    assert!(matches!(checked.ty.unqualified(), Ty::Pointer { pointee, .. } if matches!(pointee.unqualified(), Ty::Enum(_))));
    let checked = check_with(decls, "p = q");
    assert!(matches!(checked.ty.unqualified(), Ty::Pointer { .. }));
    // A different integer type and a second enum stay incompatible.
    let msg = check_with_err(&["enum e { a = -2147483647 - 1 };", "unsigned *u;", "enum e *q;"], "1 ? q : u");
    assert!(msg.contains("incompatible operand types in ternary"), "{msg:?}");
    let msg = check_with_err(&["enum e1 { a = 1 };", "enum e2 { b = 2 };", "enum e1 *q1;", "enum e2 *q2;"], "1 ? q1 : q2");
    assert!(msg.contains("incompatible operand types in ternary"), "{msg:?}");
}

#[test]
fn const_function_parameter_is_readonly() {
    // Parameters carry qualifiers like any object.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(const int x) { x = 1; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "assignment of read-only variable 'x'");
}

#[test]
fn const_return_value_reads() {
    // A `const`-qualified return converts like any value.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("const int f(void); int g(void) { return f() + 1; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
}

#[test]
fn generic_qualified_control_selects_unqualified() {
    // The controlling expression undergoes lvalue conversion (which
    // drops qualifiers), so a qualified control matches the plain
    // association, like GCC.
    assert_eq!(enum_value_of("enum { N = _Generic((const int)0, int: 1, default: 2) };", "N"), 1);
}

#[test]
fn file_scope_vla_typedef_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int n; typedef int B[n];", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "variably modified type at file scope");
}

#[test]
fn block_scope_vla_typedef_parses() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int n) { typedef int B[n]; B x; (void)x; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn const_qualifier_mismatch_redeclaration_panics() {
    // Verified against gcc ("conflicting type qualifiers").
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("const int x; int x;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "conflicting redeclaration");
}

#[test]
fn restrict_on_base_type_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("restrict int *p;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "invalid use of 'restrict'");
}

#[test]
fn restrict_on_pointer_parses() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int *restrict p;", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    let Ty::Pointer { qualifiers, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected a pointer");
    };
    assert!(matches!(qualifiers.as_slice(), [TypeQualifier::Restrict]));
}

// ---- multiple declarators flatten ----

#[test]
fn multiple_declarators_flatten_into_one_each() {
    // `int a, b, *c;` must be indistinguishable from
    // `int a; int b; int *c;` — one `Var` per declarator, and the `*`
    // binds to `c` only.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a, b, *c;", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    let ids: Vec<u32> = tu
        .external_declarations
        .iter()
        .map(|ed| match ed {
            ExternalDeclaration::Var(id) => id.0,
            ExternalDeclaration::Func(_) => panic!("expected a Var, got {ed:?}"),
        })
        .collect();
    assert_eq!(ids, [0, 1, 2]);
    assert_eq!(parser.sema.decls.len(), 3);
    let Ty::Pointer { pointee, .. } = &parser.sema.decls[2].ty else {
        panic!("expected a pointer for `c`")
    };
    assert_eq!(**pointee, Ty::Int);
}

#[test]
fn tag_only_declarations_yield_no_external_declaration() {
    // The tag lives in `Sema`; a specifier-only declaration contributes
    // nothing to the translation unit's declaration list.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S; int a;", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
    assert!(matches!(tu.external_declarations[0], ExternalDeclaration::Var(_)));
}

#[test]
fn struct_definition_with_declarator_list_yields_one_var_each() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int x; } a, b;", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
    assert!(tu.external_declarations.iter().all(|ed| matches!(ed, ExternalDeclaration::Var(_))));
    let s = parser.intern.get_or_intern("S");
    let tag = parser.sema.lookup_tag(s).unwrap();
    assert!(parser.sema.structs[tag.0 as usize].complete);
}

// ---- function parameter lists ----

#[test]
fn function_declaration_lowers_parameters() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int, char *);", &mut sema, &mut interner);
    let id = parse_decl(&mut parser);
    let ty = &parser.sema.decls[id.0 as usize].ty;
    let Ty::Func { ret, params, variadic, .. } = ty else { panic!("{ty:?}") };
    assert_eq!(**ret, Ty::Int);
    assert!(!*variadic);
    assert_eq!(
        params,
        &vec![
            Ty::Int,
            Ty::Pointer {
                qualifiers: vec![],
                size: PtrSize::Default,
                pointee: Box::new(Ty::Char),
            },
        ]
    );
}

#[test]
fn function_redeclaration_compares_parameter_types_only() {
    // Parameter *names* are not part of the type: `int f(int x);`
    // redeclares `int f(int);`.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int); int f(int x);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    // `f` merged, plus `x` in the second prototype's scope.
    assert_eq!(parser.sema.decls.len(), 2);
}

#[test]
fn function_parameter_type_conflict_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int); int f(float);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "conflicting redeclaration");
}

#[test]
fn function_pointer_member_with_parameters_parses() {
    // The shape of `Button.func` in test.c: a member that is a pointer
    // to a function taking a pointer, so the parameter list must parse
    // before the member can be typed.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct B { void (*f)(const void *p); };", &mut sema, &mut interner);
    assert!(parser.parse_declaration().is_empty());
    let b = parser.intern.get_or_intern("B");
    let tag = parser.sema.lookup_tag(b).unwrap();
    let def = &parser.sema.structs[tag.0 as usize];
    let member_ty = &def.members[0].ty;
    let Ty::Pointer { pointee, .. } = member_ty else { panic!("{member_ty:?}") };
    let Ty::Func { ret, params, variadic, .. } = &**pointee else { panic!("{pointee:?}") };
    assert_eq!(**ret, Ty::Void);
    assert!(!*variadic);
    let [param] = params.as_slice() else { panic!("{params:?}") };
    // The `const` of `const void *` qualifies the pointee.
    assert!(matches!(param, Ty::Pointer { pointee, .. } if **pointee == Ty::qualified(vec![TypeQualifier::Const], Ty::Void)));
}

#[test]
fn representative_prototypes_parse() {
    // A spread of parameter shapes: function-pointer callbacks,
    // `register`, struct types, unnamed/abstract parameters, ellipsis,
    // and the empty lists.
    for src in [
        "void on_event(void (*cb)(int code, void *ctx));",
        "int search(register int key, const char *name);",
        "int cmp(struct S a, struct S *b);",
        "unsigned long write(int fd, const void *buf, unsigned long n);",
        "int f(int (*)(void));",
        "int puts(const char *);",
        "int main(void);",
        "int h(int, ...);",
        "void nothing();",
        "void v(void);",
    ] {
        let mut sema = Sema::new();
        let mut interner = Interner::new();
        let mut parser = test_parser(src, &mut sema, &mut interner);
        let ids = parser.parse_declaration();
        assert!(parser.tokens.is_empty(), "trailing tokens for {src:?}");
        let ty = &parser.sema.decls[ids[0].0 as usize].ty;
        assert!(matches!(ty, Ty::Func { .. }), "{src:?} -> {ty:?}");
    }
}

// ---- function declarations / definitions at file scope ----

/// Parse a whole translation unit that must hold exactly one external
/// declaration, returning its id.
fn parse_one_external(src: &str, parser: &mut Parser) -> (DeclId, ExternalDeclarationKind) {
    let tu = parser.parse_translation_unit();
    assert!(parser.tokens.is_empty(), "trailing tokens for {src:?}");
    let [ed] = tu.external_declarations.as_slice() else {
        panic!("expected one external declaration for {src:?}, got {:?}", tu.external_declarations);
    };
    match ed {
        ExternalDeclaration::Var(id) => (*id, ExternalDeclarationKind::Var),
        ExternalDeclaration::Func(id) => (*id, ExternalDeclarationKind::Func),
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ExternalDeclarationKind {
    Var,
    Func,
}

#[test]
fn function_prototype_yields_func_binding() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int);", &mut sema, &mut interner);
    let (id, kind) = parse_one_external("int f(int);", &mut parser);
    assert_eq!(kind, ExternalDeclarationKind::Func);
    let b = &parser.sema.decls[id.0 as usize];
    assert_eq!(b.kind, BindingKind::Func);
    assert_eq!(b.linkage, Linkage::External);
    assert!(!b.defined && !b.tentative);
    let Ty::Func { params, prototyped, variadic, .. } = &b.ty else { panic!("{:?}", b.ty) };
    assert!(*prototyped && !variadic && params == &vec![Ty::Int]);
}

#[test]
fn unprototyped_declaration_is_unprototyped() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f();", &mut sema, &mut interner);
    let (id, kind) = parse_one_external("int f();", &mut parser);
    assert_eq!(kind, ExternalDeclarationKind::Func);
    let Ty::Func { prototyped, params, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected func")
    };
    assert!(!prototyped && params.is_empty());
}

#[test]
fn unprototyped_merges_with_prototype() {
    // C17 §6.2.7p4: the composite takes the prototyped side.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(); int f(int);", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
    assert_eq!(parser.sema.decls.len(), 1);
    let Ty::Func { params, prototyped, .. } = &parser.sema.decls[0].ty else {
        panic!("expected func")
    };
    assert!(*prototyped && params == &vec![Ty::Int]);

    // And the other order keeps the prototype too.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int); int f();", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_eq!(parser.sema.decls.len(), 1);
    let Ty::Func { params, prototyped, .. } = &parser.sema.decls[0].ty else {
        panic!("expected func")
    };
    assert!(*prototyped && params == &vec![Ty::Int]);
}

#[test]
fn prototyped_arity_conflict_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void); int f(int);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "conflicting redeclaration");
}

#[test]
fn prototyped_variadic_conflict_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int); int f(int, ...);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "conflicting redeclaration");
}

#[test]
fn unprototyped_merges_with_variadic() {
    // Unspecified parameters match anything, including `...`.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(); int f(int, ...);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_eq!(parser.sema.decls.len(), 1);
}

#[test]
fn function_definition_defines() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void) { return 0; }", &mut sema, &mut interner);
    let (id, kind) = parse_one_external("int f(void) {}", &mut parser);
    assert_eq!(kind, ExternalDeclarationKind::Func);
    let b = &parser.sema.decls[id.0 as usize];
    assert!(b.defined && !b.tentative);
    assert_eq!(b.linkage, Linkage::External);
}

#[test]
fn parse_function_definition_works_standalone() {
    // specs + declarator in, defined `DeclId` out, body consumed.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("static int f(void) { return 0; }", &mut sema, &mut interner);
    let specs = parser.parse_declaration_specifiers();
    let declarator = parser.parse_declarator();
    let ty = parser.spec_to_ty(&specs, &declarator);
    let id = parser.parse_function_definition(&specs, declarator, ty, &GnuAttributes::default());
    assert!(parser.tokens.is_empty());
    let b = &parser.sema.decls[id.0 as usize];
    assert_eq!(b.kind, BindingKind::Func);
    assert!(b.defined && !b.tentative);
    assert_eq!(b.linkage, Linkage::Internal);
}

#[test]
fn declaration_then_definition_merges() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int); int f(int x) { return x; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
    assert!(matches!(tu.external_declarations[0], ExternalDeclaration::Func(_)));
    assert!(matches!(tu.external_declarations[1], ExternalDeclaration::Func(_)));
    // `f` merged plus its parameter `x` declared in the prototype
    // scope and again in the function scope.
    assert_eq!(parser.sema.decls.len(), 3);
    assert!(parser.sema.decls[0].defined);
    assert_eq!(parser.sema.decls[1].kind, BindingKind::Var);
}

#[test]
fn two_definitions_panic() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void) {} int f(void) {}", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "redefinition of function");
}

#[test]
fn definition_with_conflicting_prototype_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void) {} int f(int);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "conflicting redeclaration");
}

#[test]
fn static_extern_linkage_merges() {
    // `extern` inherits prior internal linkage (C17 §6.2.2p4).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("static int f(void); extern int f(void);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_eq!(parser.sema.decls.len(), 1);
    assert_eq!(parser.sema.decls[0].linkage, Linkage::Internal);

    // `static` definition then a plain declaration adopts internal.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("static int f(void) {} int f(void);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_eq!(parser.sema.decls.len(), 1);
    assert_eq!(parser.sema.decls[0].linkage, Linkage::Internal);
    assert!(parser.sema.decls[0].defined);
}

#[test]
fn extern_then_static_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("extern int f(void); static int f(void);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "conflicting redeclaration");
}

#[test]
fn old_style_definition_parses_and_defines() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    // K&R definitions rewrite to modern prototypes: the
    // declaration-list supplies the parameter types, missing entries
    // default to `int`, and the body sees the parameters.
    let mut parser = test_parser("int f(a) int a; { return a; }", &mut sema, &mut interner);
    let (id, kind) = parse_one_external("old-style", &mut parser);
    assert_eq!(kind, ExternalDeclarationKind::Func);
    let b = &parser.sema.decls[id.0 as usize];
    assert!(b.defined);
    // Rewritten definitions are prototyped with the declared types.
    let Ty::Func { prototyped, params, .. } = &b.ty else { panic!("expected func") };
    assert!(prototyped);
    assert_eq!(params, &vec![Ty::Int]);
    // The body declares one parameter `a` of type `int`.
    let body = parser.sema.bodies.get(&id).expect("body");
    assert_eq!(body.params.len(), 1);
    assert_eq!(parser.sema.decls[body.params[0].0 as usize].ty, Ty::Int);
    // The declaration-list `a` does not leak to file scope.
    let a = parser.intern.get_or_intern("a");
    assert_eq!(parser.sema.lookup_symbol(a), None);
}

#[test]
fn old_style_prototype_panics() {
    // Identifier-lists are definitions only.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(a, b);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "parameter list without types");
}

#[test]
fn kr_missing_param_defaults_to_int() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(a, b) int a; { return a + b; }", &mut sema, &mut interner);
    let (id, _) = parse_one_external("kr-missing", &mut parser);
    let Ty::Func { prototyped, params, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected func")
    };
    assert!(prototyped);
    assert_eq!(params, &vec![Ty::Int, Ty::Int]);
}

#[test]
fn kr_parameters_promote_in_the_signature_only() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(c, s, h) char c; short s; float h; { return 0; }", &mut sema, &mut interner);
    let (id, _) = parse_one_external("kr-promote", &mut parser);
    let Ty::Func { params, prototyped, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected func")
    };
    assert!(prototyped);
    // The signature carries the promoted types: an unprototyped
    // caller passes promoted values, so the ABI slot must hold them.
    assert_eq!(params, &vec![Ty::Int, Ty::Int, Ty::Double]);
    // The body's parameters keep the declared types; the prologue
    // converts, so `sizeof(c)` is 1 and `f(-1)` reads the low byte.
    let body = parser.sema.bodies.get(&id).expect("body recorded");
    let body_tys: Vec<Ty> = body.params.iter().map(|p| parser.sema.decls[p.0 as usize].ty.clone()).collect();
    assert_eq!(body_tys, vec![Ty::Char, Ty::Short, Ty::Float]);
}

#[test]
fn kr_array_and_function_decay() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(a, p) int a[]; int (*p)(void); { return 0; }", &mut sema, &mut interner);
    let (id, _) = parse_one_external("kr-decay", &mut parser);
    let Ty::Func { params, prototyped, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected func")
    };
    assert!(prototyped);
    assert_eq!(params.len(), 2);
    assert!(matches!(&params[0], Ty::Pointer { .. }));
    assert!(matches!(&params[1], Ty::Pointer { .. }));
}

#[test]
fn kr_multi_declarator_list() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(a, b) int a, b; { return a + b; }", &mut sema, &mut interner);
    let (id, _) = parse_one_external("kr-multi", &mut parser);
    let Ty::Func { params, prototyped, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected func")
    };
    assert!(prototyped);
    assert_eq!(params, &vec![Ty::Int, Ty::Int]);
}

#[test]
fn kr_register_allowed() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(a) register int a; { return a; }", &mut sema, &mut interner);
    let (id, _) = parse_one_external("kr-register", &mut parser);
    let Ty::Func { params, prototyped, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected func")
    };
    assert!(prototyped);
    assert_eq!(params, &vec![Ty::Int]);
}

#[test]
fn kr_merges_with_prior_prototype() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int, int); int f(a, b) int a, b; { return a + b; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let f = parser.intern.get_or_intern("f");
    let id = parser.sema.lookup_symbol(f).expect("f");
    let b = &parser.sema.decls[id.0 as usize];
    assert!(b.defined);
    let Ty::Func { params, prototyped, .. } = &b.ty else { panic!("expected func") };
    assert!(prototyped);
    assert_eq!(params, &vec![Ty::Int, Ty::Int]);
}

#[test]
fn kr_arity_mismatch_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int, int, int); int f(a, b) int a, b; { return 0; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "conflicting redeclaration");
}

#[test]
fn kr_extra_declaration_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(a) int a; int b; { return a; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "no such parameter");
}

#[test]
fn kr_duplicate_ident_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(a, a) int a; { return a; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "redefinition of parameter");
}

#[test]
fn kr_duplicate_declaration_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(a) int a; int a; { return a; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "redefinition of parameter");
}

#[test]
fn kr_initializer_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(a) int a = 1; { return a; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "initializer in parameter declaration");
}

#[test]
fn kr_storage_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(a) static int a; { return a; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "storage class specifier in parameter");
}

#[test]
fn kr_void_param_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(a) void a; { return 0; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "declaration of `void` object");
}

#[test]
fn prototype_scope_tags_do_not_leak() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(struct S { int x; } s);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let s = parser.intern.get_or_intern("S");
    assert_eq!(parser.sema.lookup_tag(s), None);
    // ... but the parameter type still names that tag.
    let f = parser.intern.get_or_intern("f");
    let id = parser.sema.lookup_symbol(f).unwrap();
    let Ty::Func { params, .. } = &parser.sema.decls[id.0 as usize].ty else {
        panic!("expected func")
    };
    assert!(matches!(params.as_slice(), [Ty::Struct(_)]));

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(struct Q *p);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let q = parser.intern.get_or_intern("Q");
    // A mere reference forward-declares in the prototype scope, so it
    // is gone afterwards too.
    assert_eq!(parser.sema.lookup_tag(q), None);
}

#[test]
fn tentative_and_definitions() {
    // `int x;` parses as tentative — checked via `parse_declaration`,
    // which does not run end-of-TU finalization ...
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x;", &mut sema, &mut interner);
    let ids = parser.parse_declaration();
    assert!(!parser.sema.decls[ids[0].0 as usize].defined);
    assert!(parser.sema.decls[ids[0].0 as usize].tentative);

    // ... but a whole TU finalizes it to a definition (C17 §6.9.2p2).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x;", &mut sema, &mut interner);
    let (id, kind) = parse_one_external("int x;", &mut parser);
    assert_eq!(kind, ExternalDeclarationKind::Var);
    assert!(parser.sema.decls[id.0 as usize].defined);
    assert!(!parser.sema.decls[id.0 as usize].tentative);

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x = 1;", &mut sema, &mut interner);
    let (id, _) = parse_one_external("int x = 1;", &mut parser);
    assert!(parser.sema.decls[id.0 as usize].defined);
    assert!(!parser.sema.decls[id.0 as usize].tentative);

    // Tentative + tentative merges (then finalizes to defined).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x; int x;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_eq!(parser.sema.decls.len(), 1);
    assert!(parser.sema.decls[0].defined);

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x; int x = 1;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_eq!(parser.sema.decls.len(), 1);
    assert!(parser.sema.decls[0].defined);

    // `extern` without init is a pure declaration (never finalized).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("extern int x;", &mut sema, &mut interner);
    let (id, _) = parse_one_external("extern int x;", &mut parser);
    assert!(!parser.sema.decls[id.0 as usize].defined);
    assert!(!parser.sema.decls[id.0 as usize].tentative);

    // ... even `extern int x = 1;` defines.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("extern int x = 1;", &mut sema, &mut interner);
    let (id, _) = parse_one_external("extern int x = 1;", &mut parser);
    assert!(parser.sema.decls[id.0 as usize].defined);
}

#[test]
fn two_initialized_definitions_panic() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x = 1; int x = 2;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "redefinition");
}

#[test]
fn object_linkage_merges() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("static int x; extern int x;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_eq!(parser.sema.decls.len(), 1);
    assert_eq!(parser.sema.decls[0].linkage, Linkage::Internal);
}

#[test]
fn block_static_object_is_distinguished_from_automatic() {
    // A block-scope `static` has no linkage but static storage duration
    // (C17 §6.2.4p3); `block_static` is what tells the backend to give it
    // a data symbol instead of a stack slot or SSA variable.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void) { static int s = 1; int a; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let s = parser.intern.get_or_intern("s");
    let a = parser.intern.get_or_intern("a");
    let statics: Vec<_> = parser.sema.decls.iter().filter(|b| b.block_static).collect();
    assert_eq!(statics.len(), 1);
    assert_eq!(statics[0].name, Some(s));
    assert_eq!(statics[0].linkage, Linkage::NoLinkage);
    assert!(parser.sema.decls.iter().any(|b| b.name == Some(a) && !b.block_static));
}

#[test]
fn file_scope_static_uses_linkage_not_block_static() {
    // File-scope `static` is threaded through `linkage` (Internal); it
    // must not set `block_static`, or the backend would re-declare it
    // under a mangled local name.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("static int x; int f(void) { return x; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let x = parser.intern.get_or_intern("x");
    let binding = parser.sema.decls.iter().find(|b| b.name == Some(x)).expect("x declared");
    assert_eq!(binding.linkage, Linkage::Internal);
    assert!(!binding.block_static);
}

#[test]
fn object_extern_then_static_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("extern int x; static int x;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "conflicting redeclaration");
}

#[test]
fn var_and_func_conflict() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x; int x(void);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "conflicting redeclaration");
}

#[test]
fn auto_at_file_scope_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("auto int x;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "`auto`/`register` at file scope");
}

#[test]
fn multiple_storage_classes_panic() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("static extern int x;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "multiple storage class");
}

#[test]
fn inline_on_object_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("inline int x;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "`inline` on non-function");
}

#[test]
fn void_object_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void x;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "declaration of `void` object");
}

#[test]
fn function_initializer_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void) = 1;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "function with initializer");
}

#[test]
fn function_returning_function_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void)(void);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "function returning function");
}

#[test]
fn function_definition_with_comma_panics() {
    // A definition takes exactly one declarator (C17 §6.9.1p2).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int g, f(void) {}", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "function definition with multiple declarators");
}

#[test]
fn block_extern_reuses_file_entity() {
    // A block-scope `extern` merges with the file-scope entity instead
    // of shadowing it (C17 §6.2.2p4).
    let mut intern = Interner::new();
    let mut sema = Sema::new();
    let x = intern.get_or_intern("x");
    let file = sema
        .declare(
            x,
            Binding {
                name: Some(x),
                ty: Ty::Int,
                kind: BindingKind::Var,
                linkage: Linkage::External,
                block_static: false,
                tls: false,
                defined: false,
                tentative: true,
                constructor: None,
                destructor: None,
                weak: false,
                alias: None,
                inline: false,
                align: None,
                naked: false,
                noinline: false,
                is_async: false,
                section: None,
                asm_label: None,
                span: Span::dummy(),
            },
        )
        .unwrap();
    sema.push_scope();
    let inner = sema
        .declare(
            x,
            Binding {
                name: Some(x),
                ty: Ty::Int,
                kind: BindingKind::Var,
                linkage: Linkage::External,
                block_static: false,
                tls: false,
                defined: false,
                tentative: false,
                constructor: None,
                destructor: None,
                weak: false,
                alias: None,
                inline: false,
                align: None,
                naked: false,
                noinline: false,
                is_async: false,
                section: None,
                asm_label: None,
                span: Span::dummy(),
            },
        )
        .unwrap();
    assert_eq!(inner, file);
    sema.pop_scope();
}

#[test]
fn block_extern_after_local_reports() {
    // gcc: "extern declaration follows declaration with no linkage".
    let mut intern = Interner::new();
    let mut sema = Sema::new();
    let x = intern.get_or_intern("x");
    sema.push_scope();
    sema.declare(
        x,
        Binding {
            name: Some(x),
            ty: Ty::Int,
            kind: BindingKind::Var,
            linkage: Linkage::NoLinkage,
            block_static: false,
            tls: false,
            defined: true,
            tentative: false,
            constructor: None,
            destructor: None,
            weak: false,
            alias: None,
            inline: false,
            align: None,
            naked: false,
            noinline: false,
            is_async: false,
            section: None,
            asm_label: None,
            span: Span::dummy(),
        },
    )
    .unwrap();
    let err = sema
        .declare(
            x,
            Binding {
                name: Some(x),
                ty: Ty::Int,
                kind: BindingKind::Var,
                linkage: Linkage::External,
                block_static: false,
                tls: false,
                defined: false,
                tentative: false,
                constructor: None,
                destructor: None,
                weak: false,
                alias: None,
                inline: false,
                align: None,
                naked: false,
                noinline: false,
                is_async: false,
                section: None,
                asm_label: None,
                span: Span::dummy(),
            },
        )
        .unwrap_err();
    assert!(err.message.contains("conflicting redeclaration"), "{:?}", err.message);
}

// ---- spans (see `crate::span`) ----

use crate::span::HasSpan;

fn span_of_decl(src: &str) -> crate::span::Span {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(src, &mut sema, &mut interner);
    let ids = parser.parse_declaration();
    assert!(parser.tokens.is_empty(), "trailing tokens for {src:?}");
    parser.sema.decls[ids[0].0 as usize].span
}

#[test]
fn span_covers_whole_declaration() {
    // `int x;`: specs `int` at 1:0-1:3, declarator `x` at 1:4-1:5.
    let span = span_of_decl("int x;");
    assert_eq!((span.start_ln, span.start_col), (1, 0));
    assert_eq!((span.end_ln, span.end_col), (1, 5));
}

#[test]
fn span_translation_unit_covers_input() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x;\nint y;", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!((tu.span.start_ln, tu.span.start_col), (1, 0));
    assert_eq!((tu.span.end_ln, tu.span.end_col), (2, 6));
    // `HasSpan` agrees with the stored span field.
    assert_eq!(tu.span(), tu.span);
}

#[test]
fn span_function_suffix_includes_parens() {
    // `f(int)`: `(` at 1:1-1:2, `)` ending at 1:6.
    let d = parse_one_declarator("f(int)");
    let [DirectSuffix::Function(f)] = d.direct.suffixes.as_slice() else { panic!("{d:?}") };
    assert_eq!((f.span.start_ln, f.span.start_col), (1, 1));
    assert_eq!((f.span.end_ln, f.span.end_col), (1, 6));
}

#[test]
fn span_pointer_covers_star_and_qualifiers() {
    let d = parse_one_declarator("* const p");
    let [ptr] = d.pointer.as_slice() else { panic!("{d:?}") };
    assert_eq!((ptr.span.start_ln, ptr.span.start_col), (1, 0));
    assert_eq!((ptr.span.end_ln, ptr.span.end_col), (1, 7));
}

#[test]
fn span_param_covers_specifiers_and_declarator() {
    let d = parse_one_declarator("f(int a)");
    let [DirectSuffix::Function(f)] = d.direct.suffixes.as_slice() else { panic!("{d:?}") };
    assert_eq!((f.params[0].span.start_ln, f.params[0].span.start_col), (1, 2));
    assert_eq!((f.params[0].span.end_ln, f.params[0].span.end_col), (1, 7));
}

#[test]
fn span_member_covers_member_declaration() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int x; };", &mut sema, &mut interner);
    assert!(parser.parse_declaration().is_empty());
    let s = parser.intern.get_or_intern("S");
    let tag = parser.sema.lookup_tag(s).unwrap();
    let def = &parser.sema.structs[tag.0 as usize];
    // Tag span covers `struct S`; member span covers `int x`.
    assert_eq!((def.span.start_ln, def.span.start_col), (1, 0));
    assert_eq!((def.span.end_ln, def.span.end_col), (1, 8));
    assert_eq!((def.members[0].span.start_ln, def.members[0].span.start_col), (1, 11));
    assert_eq!((def.members[0].span.end_ln, def.members[0].span.end_col), (1, 16));
}

#[test]
fn span_merged_declaration_keeps_first_site() {
    // The stored binding keeps the first declaration's span, so later
    // diagnostics can point at both sites.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(); int f(int);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_eq!(parser.sema.decls.len(), 1);
    let span = parser.sema.decls[0].span;
    assert_eq!((span.start_ln, span.start_col), (1, 0));
    assert_eq!((span.end_ln, span.end_col), (1, 7));
}

#[test]
fn error_message_carries_location() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int); int f(float);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let diagnostics = parser.diagnostics();
    let err = diagnostics.iter().find(|d| d.is_error()).expect("an error is reported");
    assert!(err.message.contains("conflicting redeclaration"), "{:?}", err.message);
    assert_eq!(err.primary.span.start_ln, 1);
}

// ---- codespan-rendered diagnostics ----

#[test]
fn byte_range_maps_lines_and_columns() {
    let src = "int x;\nint y;";
    let at = |sl, sc, el, ec| {
        byte_range(
            src,
            Span {
                file: None,
                start_ln: sl,
                start_col: sc,
                end_ln: el,
                end_col: ec,
            },
        )
    };
    assert_eq!(at(1, 0, 1, 5), 0..5);
    assert_eq!(at(2, 0, 2, 6), 7..13);
    assert_eq!(at(1, 3, 1, 3), 3..3);
    // Out-of-range positions clamp instead of failing.
    assert_eq!(at(1, 0, 1, 100), 0..6);
    assert_eq!(at(9, 0, 9, 4), 13..13);
    // Columns count characters, not bytes.
    let wide = "int π;";
    assert_eq!(
        byte_range(
            wide,
            Span {
                file: None,
                start_ln: 1,
                start_col: 4,
                end_ln: 1,
                end_col: 5,
            },
        ),
        4..6
    );
}

#[test]
fn single_label_error_renders_source_snippet() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let parser = test_parser("int x;\nfloat y;", &mut sema, &mut interner);
    let file = parser.intern.get_or_intern("<__glcc_source>");
    let err = CompileError::new(
        "declaration of `void` object",
        Span {
            file: Some(file),
            start_ln: 2,
            start_col: 0,
            end_ln: 2,
            end_col: 5,
        },
    );
    let out = parser.emit_to_string(&err);
    assert!(out.contains("error"), "{out:?}");
    assert!(out.contains("declaration of `void` object"), "{out:?}");
    assert!(out.contains("float y;"), "{out:?}");
}

#[test]
fn conflicting_redeclaration_renders_both_sites() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let parser = test_parser("int a;\nfloat a;", &mut sema, &mut interner);
    let file = parser.intern.get_or_intern("<__glcc_source>");
    let first = Span {
        file: Some(file),
        start_ln: 1,
        start_col: 0,
        end_ln: 1,
        end_col: 5,
    };
    let second = Span {
        file: Some(file),
        start_ln: 2,
        start_col: 0,
        end_ln: 2,
        end_col: 7,
    };
    let a = parser.intern.get_or_intern("a");
    let ok = parser.sema.declare(
        a,
        Binding {
            name: Some(a),
            ty: Ty::Int,
            kind: BindingKind::Var,
            linkage: Linkage::External,
            block_static: false,
            tls: false,
            defined: false,
            tentative: true,
            constructor: None,
            destructor: None,
            weak: false,
            alias: None,
            inline: false,
            align: None,
            naked: false,
            noinline: false,
            is_async: false,
            section: None,
            asm_label: None,
            span: first,
        },
    );
    assert!(ok.is_ok());
    let err = parser
        .sema
        .declare(
            a,
            Binding {
                name: Some(a),
                ty: Ty::Float,
                kind: BindingKind::Var,
                linkage: Linkage::External,
                block_static: false,
                tls: false,
                defined: false,
                tentative: true,
                constructor: None,
                destructor: None,
                weak: false,
                alias: None,
                inline: false,
                align: None,
                naked: false,
                noinline: false,
                is_async: false,
                section: None,
                asm_label: None,
                span: second,
            },
        )
        .expect_err("conflicting types should fail");
    let out = parser.emit_to_string(&err);
    assert!(out.contains("conflicting redeclaration"), "{out:?}");
    assert!(out.contains("previously declared here"), "{out:?}");
    assert!(out.contains("int a;"), "{out:?}");
    assert!(out.contains("float a;"), "{out:?}");
}

#[test]
fn dummy_span_error_renders_message_only() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let parser = test_parser("int x;", &mut sema, &mut interner);
    let out = parser.emit_to_string(&CompileError::new("mystery error", Span::dummy()));
    assert!(out.contains("mystery error"), "{out:?}");
}

#[test]
fn diagnostic_names_real_file_and_original_line() {
    // Preprocessed TU with linemarkers: buffer line 4 holds `float y;`,
    // whose original location is `b.h:10`.
    let src = "# 1 \"a.c\"\nint x;\n# 10 \"b.h\"\nfloat y;\n";
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let parser = test_parser(src, &mut sema, &mut interner);
    let file = parser.intern.get_or_intern("b.h");
    let err = CompileError::new(
        "bad float",
        Span {
            file: Some(file),
            start_ln: 4,
            start_col: 0,
            end_ln: 4,
            end_col: 8,
        },
    );
    let out = parser.emit_to_string(&err);
    assert!(out.contains("b.h:10"), "{out:?}");
    assert!(!out.contains("__glcc_source"), "{out:?}");
    assert!(out.contains("float y;"), "{out:?}");
}

#[test]
fn diagnostic_labels_span_files() {
    // Primary in `b.h`, secondary in `a.c`: each header names its own file.
    let src = "# 1 \"a.c\"\nint a;\n# 20 \"b.h\"\nfloat a;\n";
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let parser = test_parser(src, &mut sema, &mut interner);
    let a_c = parser.intern.get_or_intern("a.c");
    let b_h = parser.intern.get_or_intern("b.h");
    let err = CompileError::new(
        "conflicting redeclaration",
        Span {
            file: Some(b_h),
            start_ln: 4,
            start_col: 0,
            end_ln: 4,
            end_col: 8,
        },
    )
    .with_secondary(
        Span {
            file: Some(a_c),
            start_ln: 2,
            start_col: 0,
            end_ln: 2,
            end_col: 6,
        },
        "previously declared here",
    );
    let out = parser.emit_to_string(&err);
    assert!(out.contains("b.h:20"), "{out:?}");
    assert!(out.contains("a.c:1"), "{out:?}");
    assert!(out.contains("int a;"), "{out:?}");
    assert!(out.contains("float a;"), "{out:?}");
}

// ---- function bodies ----

/// Parse one function definition and return its stored body.
fn parse_one_body(src: &str, parser: &mut Parser) -> (DeclId, crate::parse::FuncBody) {
    let tu = parser.parse_translation_unit();
    finish(parser);
    assert!(parser.tokens.is_empty(), "trailing tokens for {src:?}");
    let [ExternalDeclaration::Func(id)] = tu.external_declarations.as_slice() else {
        panic!("expected one Func for {src:?}");
    };
    let body = parser.sema.bodies.remove(id).expect("body stored");
    (*id, body)
}

#[test]
fn body_declarations_register_without_leaking() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void) { int x; int y; }", &mut sema, &mut interner);
    let (id, body) = parse_one_body("decls", &mut parser);
    assert_eq!(body.params.len(), 0);
    assert_eq!(body.block.items.len(), 2);
    assert!(body.block.items.iter().all(|item| matches!(
        item,
        BlockItem::Decl { ids, .. } if ids.len() == 1
    )));
    // `x` and `y` registered (f + 2 locals) but their scopes popped.
    assert_eq!(parser.sema.decls.len(), 3);
    let x = parser.intern.get_or_intern("x");
    assert_eq!(parser.sema.lookup_symbol(x), None);
    let _ = id;
}

#[test]
fn body_params_shadow_file_scope() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x; int f(int x, int y) { int z; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
    // File `x`, `f`, params `x`/`y` in the prototype scope and again
    // in the function scope, local `z`.
    assert_eq!(parser.sema.decls.len(), 7);
    let x = parser.intern.get_or_intern("x");
    let file_x = parser.sema.lookup_symbol(x).unwrap();
    assert_eq!(parser.sema.decls[file_x.0 as usize].ty, Ty::Int);
    // The file binding is the first declaration, untouched by the param.
    assert_eq!(file_x.0, 0);
    let f = match tu.external_declarations[1] {
        ExternalDeclaration::Func(id) => id,
        _ => panic!("expected Func"),
    };
    let body = &parser.sema.bodies[&f];
    assert_eq!(body.params.len(), 2);
    assert_eq!(body.block.items.len(), 1);
}

#[test]
fn body_nested_blocks_nest() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { { int a; } int b; }", &mut sema, &mut interner);
    let (_, body) = parse_one_body("nested", &mut parser);
    let [first, second] = body.block.items.as_slice() else { panic!("{body:?}") };
    let BlockItem::Stmt(Stmt::Compound(inner)) = first else { panic!("{first:?}") };
    assert_eq!(inner.items.len(), 1);
    assert!(matches!(&inner.items[0], BlockItem::Decl { .. }));
    assert!(matches!(second, BlockItem::Decl { .. }));
    // `a` does not leak past its block.
    let a = parser.intern.get_or_intern("a");
    assert_eq!(parser.sema.lookup_symbol(a), None);
}

#[test]
fn body_return_and_empty_statements() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void) { return 0; }", &mut sema, &mut interner);
    let (_, body) = parse_one_body("return", &mut parser);
    let [BlockItem::Stmt(Stmt::Return { expr, .. })] = body.block.items.as_slice() else {
        panic!("{body:?}")
    };
    let Some(expr) = expr else { panic!("expected a return value") };
    assert!(matches!(&expr.kind, CheckedExprKind::Int(_)));

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { return; }", &mut sema, &mut interner);
    let (_, body) = parse_one_body("bare return", &mut parser);
    assert!(matches!(body.block.items.as_slice(), [BlockItem::Stmt(Stmt::Return { expr: None, .. })]));

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) {}", &mut sema, &mut interner);
    let (_, body) = parse_one_body("empty", &mut parser);
    assert!(body.block.items.is_empty());

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { ; }", &mut sema, &mut interner);
    let (_, body) = parse_one_body("semi", &mut parser);
    assert!(matches!(body.block.items.as_slice(), [BlockItem::Stmt(Stmt::Empty { .. })]));
}

#[test]
fn body_block_may_shadow_param() {
    // Legal C (unlike C++): a block-scope declaration shadows the parameter.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int x) { int x; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    // `f`, `x` in the prototype scope, `x` in the function scope, and
    // the shadowing block `x`.
    assert_eq!(parser.sema.decls.len(), 4);
}

#[test]
fn body_duplicate_params_panic() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int x, int x) {}", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "conflicting redeclaration");
}

#[test]
fn body_if_with_block_registers_decls() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int n) { if (n) { int y; } return n; }", &mut sema, &mut interner);
    let (_, body) = parse_one_body("if", &mut parser);
    let [first, second] = body.block.items.as_slice() else { panic!("{body:?}") };
    let BlockItem::Stmt(Stmt::If { cond, then, els, .. }) = first else {
        panic!("{first:?}")
    };
    assert!(matches!(&cond.kind, CheckedExprKind::Ident(_)));
    assert!(els.is_none());
    let Stmt::Compound(then_block) = &**then else { panic!("{then:?}") };
    assert_eq!(then_block.items.len(), 1);
    assert!(matches!(second, BlockItem::Stmt(Stmt::Return { .. })));
    // `f`, `n` in the prototype scope, `n` in the function scope, and
    // the `if`-block local `y`.
    assert_eq!(parser.sema.decls.len(), 4);
}

#[test]
fn body_if_else_and_while() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { int a; int b(void); int c(void); if (a) b(); else c(); }", &mut sema, &mut interner);
    let (_, body) = parse_one_body("if-else", &mut parser);
    let [_, _, _, BlockItem::Stmt(Stmt::If { els: Some(_), .. })] = body.block.items.as_slice() else {
        panic!("{body:?}")
    };

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { int a; int b(void); while (a) b(); }", &mut sema, &mut interner);
    let (_, body) = parse_one_body("while", &mut parser);
    assert!(matches!(body.block.items.as_slice(), [_, _, BlockItem::Stmt(Stmt::While { .. })]));

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { int a; int b(void); do b(); while (a); }", &mut sema, &mut interner);
    let (_, body) = parse_one_body("do-while", &mut parser);
    assert!(matches!(body.block.items.as_slice(), [_, _, BlockItem::Stmt(Stmt::DoWhile { .. })]));
}

#[test]
fn body_for_with_declaration_init() {
    use crate::parse::ForInit;

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { for (int i = 0; i < 1; i++) {} }", &mut sema, &mut interner);
    let (_, body) = parse_one_body("for-decl", &mut parser);
    let [BlockItem::Stmt(Stmt::For { init, cond, step, body, .. })] = body.block.items.as_slice() else {
        panic!("{body:?}")
    };
    assert!(matches!(init, ForInit::Decl(ids) if ids.len() == 1));
    assert!(cond.is_some() && step.is_some());
    assert!(matches!(&**body, Stmt::Compound(_)));
    // The loop variable does not leak past the `for`.
    let i = parser.intern.get_or_intern("i");
    assert_eq!(parser.sema.lookup_symbol(i), None);
}

#[test]
fn body_unterminated_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void) { int x;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "unterminated compound statement");
}

#[test]
fn body_unterminated_statement_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void) { if (1)", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "unterminated statement");
}

#[test]
fn body_truncated_expression_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void) { return 0", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "expected `;`");
}

#[test]
fn body_missing_semi_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void) { int x }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "expected `;`");
}

// ---- expressions ----

/// Parse `src` as one full expression; panics if tokens remain.
fn parse_one_expr(src: &str) -> Expr {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(src, &mut sema, &mut interner);
    let expr = parser.parse_expression();
    finish(&parser);
    assert!(parser.tokens.is_empty(), "trailing tokens in {src:?}");
    expr
}

#[test]
fn expr_precedence_and_associativity() {
    // `*` binds tighter than `+`.
    let Expr::Binary { op: BinaryOp::Add, lhs, rhs, .. } = parse_one_expr("a + b * c") else {
        panic!("additive")
    };
    assert!(matches!(&*lhs, Expr::Ident { .. }));
    assert!(matches!(&*rhs, Expr::Binary { op: BinaryOp::Mul, .. }));

    // `-` associates left.
    let Expr::Binary { op: BinaryOp::Sub, lhs, .. } = parse_one_expr("a - b - c") else {
        panic!("left assoc")
    };
    assert!(matches!(&*lhs, Expr::Binary { op: BinaryOp::Sub, .. }));

    // `=` associates right.
    let Expr::Assign {
        op: AssignOp::Assign, lhs, rhs, ..
    } = parse_one_expr("a = b = c")
    else {
        panic!("right assoc")
    };
    assert!(matches!(&*lhs, Expr::Ident { .. }));
    assert!(matches!(&*rhs, Expr::Assign { .. }));

    // `==` binds tighter than `&&`, looser than `<`.
    let Expr::Binary { op: BinaryOp::And, .. } = parse_one_expr("a == b && c") else {
        panic!("equality")
    };
    let Expr::Binary { op: BinaryOp::Eq, lhs, rhs, .. } = parse_one_expr("a < b == c > d") else {
        panic!("relational")
    };
    assert!(matches!(&*lhs, Expr::Binary { op: BinaryOp::Lt, .. }));
    assert!(matches!(&*rhs, Expr::Binary { op: BinaryOp::Gt, .. }));

    // Shifts sit between additives and relationals.
    let Expr::Binary { op: BinaryOp::Shl, lhs, .. } = parse_one_expr("a + b << c") else {
        panic!("shift")
    };
    assert!(matches!(&*lhs, Expr::Binary { op: BinaryOp::Add, .. }));
}

#[test]
fn expr_ternary_and_comma() {
    let Expr::Ternary { cond, then, els, .. } = parse_one_expr("a ? b : c") else {
        panic!("ternary")
    };
    assert!(matches!(&*cond, Expr::Ident { .. }));
    assert!(matches!(&*then, Expr::Ident { .. }));
    assert!(matches!(&*els, Expr::Ident { .. }));

    // The else-branch associates right; the middle allows commas.
    let Expr::Ternary { els, .. } = parse_one_expr("a ? b : c ? d : e") else {
        panic!("right assoc ternary")
    };
    assert!(matches!(&*els, Expr::Ternary { .. }));
    let Expr::Ternary { then, .. } = parse_one_expr("a ? b, c : d") else {
        panic!("comma in middle")
    };
    assert!(matches!(&*then, Expr::Binary { op: BinaryOp::Comma, .. }));

    // Top-level commas group left.
    let Expr::Binary { op: BinaryOp::Comma, lhs, .. } = parse_one_expr("a, b") else {
        panic!("comma")
    };
    assert!(matches!(&*lhs, Expr::Ident { .. }));
}

#[test]
fn expr_unary_forms() {
    for (src, op) in [
        ("-a", UnaryOp::Neg),
        ("+a", UnaryOp::Pos),
        ("!a", UnaryOp::Not),
        ("~a", UnaryOp::BitNot),
        ("*p", UnaryOp::Deref),
        ("&x", UnaryOp::AddrOf),
        ("++i", UnaryOp::PreInc),
        ("--i", UnaryOp::PreDec),
    ] {
        let Expr::Unary { op: found, operand, .. } = parse_one_expr(src) else {
            panic!("{src:?}")
        };
        assert_eq!(found, op);
        assert!(matches!(&*operand, Expr::Ident { .. }));
    }
    // `!` binds tighter than `&&`; unary nests over casts.
    let Expr::Binary { op: BinaryOp::And, lhs, .. } = parse_one_expr("!a && b") else {
        panic!("not/and")
    };
    assert!(matches!(&*lhs, Expr::Unary { op: UnaryOp::Not, .. }));
    let Expr::Unary { op: UnaryOp::Neg, operand, .. } = parse_one_expr("-(int)x") else {
        panic!("negated cast")
    };
    assert!(matches!(&*operand, Expr::Cast { .. }));
}

#[test]
fn expr_postfix_forms() {
    // Chained call / subscript / member.
    let Expr::Call { func, args, .. } = parse_one_expr("a.b[c](d)") else {
        panic!("chained call")
    };
    let Expr::Subscript { base, .. } = func.as_ref() else { panic!("subscript") };
    let Expr::Member { arrow: false, .. } = base.as_ref() else { panic!("member") };
    assert_eq!(args.len(), 1);

    let Expr::Member { arrow: true, .. } = parse_one_expr("p->m") else { panic!("arrow") };
    let Expr::Call { args, .. } = parse_one_expr("f()") else { panic!("empty call") };
    assert!(args.is_empty());
    let Expr::Call { args, .. } = parse_one_expr("f(a, b + 1)") else { panic!("call args") };
    assert_eq!(args.len(), 2);

    // Postfix inc/dec wrap their operand.
    let Expr::Unary { op: UnaryOp::PostInc, .. } = parse_one_expr("i++") else {
        panic!("postinc")
    };
    let Expr::Unary { op: UnaryOp::PostDec, .. } = parse_one_expr("i--") else {
        panic!("postdec")
    };
}

#[test]
fn expr_casts_and_sizeof() {
    let Expr::Cast { ty, expr, .. } = parse_one_expr("(int)x") else { panic!("cast") };
    assert_eq!(ty, Ty::Int);
    assert!(matches!(&*expr, Expr::Ident { .. }));

    let Expr::Cast { ty, .. } = parse_one_expr("(int *)p") else { panic!("pointer cast") };
    assert!(matches!(ty, Ty::Pointer { .. }));

    // Chained casts associate right.
    let Expr::Cast { expr, .. } = parse_one_expr("(int)(float)x") else {
        panic!("chained cast")
    };
    assert!(matches!(&*expr, Expr::Cast { .. }));

    // `(x)` groups when `x` is not a type name.
    assert!(matches!(parse_one_expr("(x)"), Expr::Ident { .. }));

    assert!(matches!(parse_one_expr("sizeof x"), Expr::SizeofExpr { .. }));
    assert!(matches!(parse_one_expr("sizeof(x)"), Expr::SizeofExpr { .. }));
    let Expr::SizeofTy { ty, .. } = parse_one_expr("sizeof(int)") else {
        panic!("sizeof type")
    };
    assert_eq!(ty, Ty::Int);
    assert!(matches!(parse_one_expr("sizeof(int *)"), Expr::SizeofTy { .. }));
}

#[test]
fn expr_typedef_cast_disambiguation() {
    // `(T)` casts when `T` is a typedef name (C17 §6.7.6.3p11 spirit).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("typedef int T; (T)x", &mut sema, &mut interner);
    parser.parse_declaration();
    let expr = parser.parse_expression();
    assert!(parser.tokens.is_empty());
    assert!(matches!(expr, Expr::Cast { .. }));
}

#[test]
fn expr_constants_and_strings() {
    use crate::lexer::{IntSuffix, Radix};

    let Expr::Int { lit, .. } = parse_one_expr("42") else { panic!("int") };
    assert_eq!((lit.radix, lit.suffix), (Radix::Dec, IntSuffix::None));
    assert!(matches!(parse_one_expr("1.5"), Expr::Float { .. }));
    assert!(matches!(parse_one_expr("'a'"), Expr::Char { .. }));

    // Adjacent literals concatenate.
    let Expr::Str { parts, .. } = parse_one_expr("\"a\" \"b\"") else { panic!("strings") };
    assert_eq!(parts.len(), 2);
}

#[test]
fn expr_compound_assignment_ops() {
    for (src, op) in [
        ("a += b", AssignOp::Add),
        ("a -= b", AssignOp::Sub),
        ("a *= b", AssignOp::Mul),
        ("a /= b", AssignOp::Div),
        ("a %= b", AssignOp::Rem),
        ("a <<= b", AssignOp::Shl),
        ("a >>= b", AssignOp::Shr),
        ("a &= b", AssignOp::BitAnd),
        ("a ^= b", AssignOp::BitXor),
        ("a |= b", AssignOp::BitOr),
    ] {
        let Expr::Assign { op: found, .. } = parse_one_expr(src) else { panic!("{src:?}") };
        assert_eq!(found, op);
    }
}

#[test]
fn expr_truncated_binary_panics() {
    assert_diag_contains(&expr_diagnostics("a +"), "expected expression");
}

#[test]
fn expr_ternary_missing_colon_panics() {
    assert_diag_contains(&expr_diagnostics("a ? b"), "expected `:`");
}

#[test]
fn expr_empty_call_arg_panics() {
    assert_diag_contains(&expr_diagnostics("f(a,)"), "expected expression");
}

#[test]
fn expr_member_missing_name_panics() {
    assert_diag_contains(&expr_diagnostics("s."), "expected member name");
}

// ---- statements ----

#[test]
fn stmt_switch_case_default_break_continue() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "void f(int x) { switch (x) { int y(void); int z(void); case 1: y(); break; default: z(); } }",
        &mut sema,
        &mut interner,
    );
    let (_, body) = parse_one_body("switch", &mut parser);
    let [BlockItem::Stmt(Stmt::Switch { cond, body: switch_body, .. })] = body.block.items.as_slice() else {
        panic!("{body:?}")
    };
    assert!(matches!(&cond.kind, CheckedExprKind::Ident(_)));
    let Stmt::Compound(inner) = &**switch_body else { panic!("{switch_body:?}") };
    assert_eq!(inner.items.len(), 5);
    let [_, _, case, brk, default] = inner.items.as_slice() else { panic!("{inner:?}") };
    let BlockItem::Stmt(Stmt::Case { expr: Some(_), stmt, .. }) = case else {
        panic!("{case:?}")
    };
    assert!(matches!(&**stmt, Stmt::Expr { .. }));
    assert!(matches!(brk, BlockItem::Stmt(Stmt::Break { .. })));
    let BlockItem::Stmt(Stmt::Case { expr: None, stmt, .. }) = default else {
        panic!("{default:?}")
    };
    assert!(matches!(&**stmt, Stmt::Expr { .. }));

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { int a; while (a) { continue; } }", &mut sema, &mut interner);
    let (_, body) = parse_one_body("continue", &mut parser);
    let [_, BlockItem::Stmt(Stmt::While { body, .. })] = body.block.items.as_slice() else {
        panic!("{body:?}")
    };
    let Stmt::Compound(inner) = &**body else { panic!("{body:?}") };
    assert!(matches!(inner.items.as_slice(), [BlockItem::Stmt(Stmt::Continue { .. })]));
}

#[test]
fn stmt_goto_and_label() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { goto end; end: return; }", &mut sema, &mut interner);
    let (_, body) = parse_one_body("goto", &mut parser);
    let [goto, label] = body.block.items.as_slice() else { panic!("{body:?}") };
    assert!(matches!(goto, BlockItem::Stmt(Stmt::Goto { .. })));
    let BlockItem::Stmt(Stmt::Label { stmt, .. }) = label else { panic!("{label:?}") };
    assert!(matches!(&**stmt, Stmt::Return { .. }));
}

#[test]
fn stmt_expression_statement() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { int a; int b(int); int c; int d(void); a = 1; b(c), d(); }", &mut sema, &mut interner);
    let (_, body) = parse_one_body("expr-stmt", &mut parser);
    let [_, _, _, _, first, second] = body.block.items.as_slice() else { panic!("{body:?}") };
    let BlockItem::Stmt(Stmt::Expr { expr, .. }) = first else { panic!("{first:?}") };
    assert!(matches!(&expr.kind, CheckedExprKind::Assign(..)));
    let BlockItem::Stmt(Stmt::Expr { expr, .. }) = second else { panic!("{second:?}") };
    let CheckedExprKind::Binary(op, ..) = &expr.kind else { panic!("{expr:?}") };
    assert_eq!(*op, BinaryOp::Comma);
}

// ---- initializers ----

#[test]
fn init_expr_stored_in_sema() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x = 1 + 2;", &mut sema, &mut interner);
    let ids = parser.parse_declaration();
    assert!(parser.tokens.is_empty());
    let init = parser.sema.inits.get(&ids[0]).expect("init stored");
    let Initializer::Expr(expr) = init else { panic!("{init:?}") };
    let CheckedExprKind::Binary(op, ..) = &expr.kind else { panic!("{expr:?}") };
    assert_eq!(*op, BinaryOp::Add);
    assert!(parser.sema.decls[ids[0].0 as usize].defined);

    // No initializer means no entry (tentative definition).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int y;", &mut sema, &mut interner);
    let ids = parser.parse_declaration();
    assert!(!parser.sema.inits.contains_key(&ids[0]));
}

#[test]
fn init_brace_lists_and_designators() {
    // Plain and nested lists.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x = { 5 };", &mut sema, &mut interner);
    let ids = parser.parse_declaration();
    let Initializer::List { items, .. } = parser.sema.inits.get(&ids[0]).expect("init") else {
        panic!("expected a list")
    };
    assert_eq!(items.len(), 1);
    assert!(items[0].designators.is_empty());

    // Designated struct members, out of order plus nesting.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int a; int b; }; struct S s = { .b = 2, .a = { 3 } };", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    let id = match tu.external_declarations.as_slice() {
        [ExternalDeclaration::Var(id)] => *id,
        other => panic!("{other:?}"),
    };
    let Initializer::List { items, .. } = parser.sema.inits.get(&id).expect("init") else {
        panic!("expected a list")
    };
    let [first, second] = items.as_slice() else { panic!("{items:?}") };
    let [Designator::Field { .. }] = first.designators.as_slice() else {
        panic!("{first:?}")
    };
    assert!(matches!(&*first.init, Initializer::Expr(_)));
    assert!(matches!(&*second.init, Initializer::List { .. }));
}

#[test]
fn init_designated_scalar_panics() {
    // Designators need an aggregate; gcc rejects this the same way.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x = { [0] = 5 };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "designator in scalar initializer");
}

#[test]
fn init_missing_comma_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x = { 1 2 };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "expected `,` or `}` in initializer");
}

#[test]
fn init_designator_missing_assign_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("struct S { int a; }; struct S s = { .a 5 };", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "expected `=`, got");
}

#[test]
fn init_empty_value_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x = ;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "expected expression");
}

// ---- array types ----

#[test]
fn array_types_lower() {
    // `a[3]`: bound expression kept.
    let Ty::Array { elem, quals, bound } = lower_to_ty("a[3]") else {
        panic!("expected an array")
    };
    assert_eq!(*elem, Ty::Int);
    assert!(quals.is_empty());
    assert_eq!(bound, Some(ArrayBound::Fixed(3)));

    // `a[]` stays incomplete; qualifiers are kept on the type.
    let Ty::Array { bound: None, quals, .. } = lower_to_ty("a[]") else {
        panic!("expected an incomplete array")
    };
    assert!(quals.is_empty());
    let Ty::Array { quals, .. } = lower_to_ty("a[const 3]") else {
        panic!("expected an array")
    };
    assert!(matches!(quals.as_slice(), [TypeQualifier::Const]));

    // Nesting goes inside out: `a[2][3]` is an array of arrays.
    let Ty::Array { elem, bound, .. } = lower_to_ty("a[2][3]") else {
        panic!("expected an array")
    };
    assert_eq!(bound, Some(ArrayBound::Fixed(2)), "a[2][3] is two rows");
    assert!(matches!(&*elem, Ty::Array { .. }));

    // Pointers compose both ways.
    assert!(matches!(lower_to_ty("*p[3]"), Ty::Array { .. }));
    let Ty::Pointer { pointee, .. } = lower_to_ty("(*p)[3]") else {
        panic!("expected a pointer to array")
    };
    assert!(matches!(&*pointee, Ty::Array { .. }));
}

#[test]
fn array_redeclarations_merge() {
    // Same bounds merge; `[]` takes the specified bound (C17 §6.2.7).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a[3]; int a[3];", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_eq!(parser.sema.decls.len(), 1);

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a[]; int a[3];", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_eq!(parser.sema.decls.len(), 1);
    let Ty::Array { bound: Some(_), .. } = &parser.sema.decls[0].ty else {
        panic!("composite should carry the bound")
    };

    // `[*]` combines like `[]` (verified against gcc).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int [*]); void f(int [3]);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_eq!(parser.sema.decls.len(), 1);
}

#[test]
fn array_bound_conflict_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a[3]; int a[4];", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "conflicting redeclaration");
}

#[test]
fn array_element_conflict_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a[3]; float a[3];", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "conflicting redeclaration");
}

#[test]
fn array_of_functions_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void); int g[3](void);", &mut sema, &mut interner);
    // `g[3](void)`: array of functions (C17 §6.7.6.1p1).
    parser.parse_translation_unit();
    assert_parse_error(&parser, "array of functions");
}

#[test]
fn function_returning_array_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void)[3];", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "function returning array");
}

#[test]
fn star_in_definition_params_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int a[*]) {}", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "`[*]` outside function prototype scope");
}

#[test]
fn star_in_prototype_params_merges() {
    // `[*]` is at home in prototypes (verified against gcc).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int [*]); void f(int [*]);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_eq!(parser.sema.decls.len(), 1);
}

#[test]
fn star_in_nested_prototype_of_definition_is_fine() {
    // The `[*]` sits in `g`'s prototype scope, not `f`'s parameter list
    // (verified against gcc).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(int g(int [*])) { return 0; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    // `f` defined (`g` is declared first, in `f`'s prototype scope).
    let f = parser.intern.get_or_intern("f");
    let id = parser.sema.lookup_symbol(f).unwrap();
    assert!(parser.sema.decls[id.0 as usize].defined);
}

#[test]
fn array_with_initializer_stores_init() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a[2] = { 1, 2 };", &mut sema, &mut interner);
    let ids = parser.parse_declaration();
    let init = parser.sema.inits.get(&ids[0]).expect("init stored");
    let Initializer::List { items, .. } = init else { panic!("expected a list") };
    assert_eq!(items.len(), 2);

    // Index designators select array elements.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a[2] = { [1] = 5 };", &mut sema, &mut interner);
    let ids = parser.parse_declaration();
    let init = parser.sema.inits.get(&ids[0]).expect("init stored");
    let Initializer::List { items, .. } = init else { panic!("expected a list") };
    let [item] = items.as_slice() else { panic!("{items:?}") };
    assert!(matches!(item.designators.as_slice(), [Designator::Index { .. }]));
}

// ---- expression checking (see `crate::check`) ----

/// Declare each of `decls`, then parse and check `expr_src`.
/// Panics if tokens remain or checking fails.
fn check_with(decls: &[&str], expr_src: &str) -> CheckedExpr {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let src = format!("{} {}", decls.join(" "), expr_src);
    let mut parser = test_parser(&src, &mut sema, &mut interner);
    for _ in decls {
        parser.parse_declaration();
    }
    let expr = parser.parse_expression();
    assert!(parser.tokens.is_empty(), "trailing tokens in {expr_src:?}");
    parser
        .sema
        .check_expr(&expr, &parser.intern)
        .unwrap_or_else(|err| panic!("check failed for {expr_src:?}: {}", err.message))
}

/// Like [`check_with`], but checking must fail; returns the message.
fn check_with_err(decls: &[&str], expr_src: &str) -> String {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let src = format!("{} {}", decls.join(" "), expr_src);
    let mut parser = test_parser(&src, &mut sema, &mut interner);
    for _ in decls {
        parser.parse_declaration();
    }
    let expr = parser.parse_expression();
    assert!(parser.tokens.is_empty(), "trailing tokens in {expr_src:?}");
    parser.sema.check_expr(&expr, &parser.intern).expect_err("checking should fail").message
}

#[test]
fn uac_common_type_table() {
    use Ty::*;
    // Already-promoted integers.
    assert_eq!(Sema::uac_common_type(&Int, &Int), Int);
    assert_eq!(Sema::uac_common_type(&Int, &Long), Long);
    assert_eq!(Sema::uac_common_type(&UnsignedInt, &Int), UnsignedInt);
    assert_eq!(Sema::uac_common_type(&Long, &UnsignedInt), Long);
    assert_eq!(Sema::uac_common_type(&UnsignedLong, &Long), UnsignedLong);
    assert_eq!(Sema::uac_common_type(&LongLong, &UnsignedInt), LongLong);
    // Float ladder beats integers, in rank order.
    assert_eq!(Sema::uac_common_type(&Float, &Int), Float);
    assert_eq!(Sema::uac_common_type(&Double, &Float), Double);
    assert_eq!(Sema::uac_common_type(&LongDouble, &Int), LongDouble);
    // Complex follows its real component.
    assert_eq!(Sema::uac_common_type(&Complex(Box::new(Float)), &Int), Complex(Box::new(Float)));
    assert_eq!(Sema::uac_common_type(&Double, &Complex(Box::new(Float))), Complex(Box::new(Double)));
    assert_eq!(Sema::uac_common_type(&Complex(Box::new(Int)), &Complex(Box::new(Long))), Complex(Box::new(Long)));
}

#[test]
fn complex_types_check() {
    use Ty::*;
    let c = |t: Ty| Complex(Box::new(t));
    // GNU imaginary literals take complex types over their base.
    assert_eq!(check_with(&[], "1.0i").ty, c(Double));
    assert_eq!(check_with(&[], "1.0fi").ty, c(Float));
    assert_eq!(check_with(&[], "1.0if").ty, c(Float));
    assert_eq!(check_with(&[], "1.0iL").ty, c(LongDouble));
    assert_eq!(check_with(&[], "1i").ty, c(Int));
    assert_eq!(check_with(&[], "1j").ty, c(Int));
    // Mixed real/complex arithmetic promotes to complex.
    assert_eq!(check_with(&[], "1.0 + 2.0i").ty, c(Double));
    assert_eq!(check_with(&[], "1 + 2.0fi").ty, c(Float));
    assert_eq!(check_with(&["__complex__ int x;", "__complex__ long y;"], "x + y").ty, c(Long));
    // `__real__`/`__imag__` project the component type.
    assert_eq!(check_with(&["double _Complex x;"], "__real__ x").ty, Double);
    assert_eq!(check_with(&["__complex__ int x;"], "__imag__ x").ty, Int);
    // `~` on complex is the conjugate, keeping the type.
    assert_eq!(check_with(&["double _Complex x;"], "~x").ty, c(Double));
    // Equality applies; everything else rejects complex operands.
    assert_eq!(check_with(&["double _Complex x;"], "x == x").ty, Int);
    for (decls, expr) in [
        (vec!["double x;"], "__real__ x"),
        (vec!["double x;"], "__imag__ x"),
        (vec!["double _Complex x;"], "x % 2.0"),
        (vec!["double _Complex x;"], "x & x"),
        (vec!["double _Complex x;"], "x << 1"),
        (vec!["double _Complex x;"], "x < 2.0"),
        (vec!["double _Complex x;"], "x++"),
        (vec!["int *p;"], "(double _Complex)p"),
        (vec!["double _Complex x;"], "(int *)x"),
        (vec!["double _Complex x;"], "x %= 2.0"),
        (vec!["const double _Complex x;"], "__real__ x = 1.0"),
    ] {
        check_with_err(&decls, expr);
    }
}

#[test]
fn complex_constants_fold() {
    use crate::eval::{Value, eval_const};
    fn fold(src: &str) -> Option<Value> {
        let mut sema = Sema::new();
        let mut interner = Interner::new();
        let mut parser = test_parser(src, &mut sema, &mut interner);
        let expr = parser.parse_expression();
        assert!(parser.tokens.is_empty(), "trailing tokens in {src:?}");
        let checked = parser.sema.check_expr(&expr, &parser.intern).expect("check failed");
        eval_const(&checked, parser.sema, &parser.intern)
    }
    fn parts(v: Value) -> (Value, Value) {
        match v {
            Value::Complex(re, im) => (*re, *im),
            other => panic!("not a complex constant: {other:?}"),
        }
    }
    // Imaginary literals fold with a zero real part.
    assert!(matches!(fold("2.0i"), Some(Value::Complex(..))));
    // `1.0 + 2.0i` folds part-wise for static initializers.
    let (re, im) = parts(fold("1.0 + 2.0i").expect("folds"));
    assert!(matches!((re, im), (Value::Float(_), Value::Float(_))));
    // Conjugates and parts fold.
    assert!(matches!(fold("~(1.0 + 2.0i)"), Some(Value::Complex(..))));
    assert!(matches!(fold("__real__ (1.0 + 2.0i)"), Some(Value::Float(_))));
    assert!(matches!(fold("__imag__ (1 + 2i)"), Some(Value::Int(_))));
    // Complex equality folds (ordered, NaN-aware).
    assert_eq!(fold("(1.0 + 2.0i) == (1.0 + 2.0i)"), Some(Value::Int(1)));
    assert_eq!(fold("(1.0 + 2.0i) != (1.0 + 2.0i)"), Some(Value::Int(0)));
    assert_eq!(fold("(1.0 + 2.0i) == (1.0 + 3.0i)"), Some(Value::Int(0)));
    // Casts convert part-wise, `_Bool` reads either part.
    assert!(matches!(fold("(float _Complex)(1.0 + 2.0i)"), Some(Value::Complex(..))));
    assert_eq!(fold("(_Bool)(0.0 + 1.0i)"), Some(Value::Int(1)));
    assert_eq!(fold("(int)(3.0 + 9.0i)"), Some(Value::Int(3)));
}

#[test]
fn integer_promotion_to_int() {
    // Sub-int operands promote; Converts appear exactly where the type
    // actually changes.
    let checked = check_with(&["char c;"], "c + c");
    assert_eq!(checked.ty, Ty::Int);
    let CheckedExprKind::Binary(_, l, r) = &checked.kind else {
        panic!("expected a binary node")
    };
    assert!(matches!(&l.kind, CheckedExprKind::Convert { .. }));
    assert!(matches!(&r.kind, CheckedExprKind::Convert { .. }));

    // No-op promotion inserts nothing.
    let checked = check_with(&["int i;"], "i + 1");
    let CheckedExprKind::Binary(_, l, r) = &checked.kind else {
        panic!("expected a binary node")
    };
    assert!(matches!(&l.kind, CheckedExprKind::Ident(_)));
    assert!(matches!(&r.kind, CheckedExprKind::Int(_)));
}

#[test]
fn check_literal_types() {
    assert_eq!(check_with(&[], "42").ty, Ty::Int);
    assert_eq!(check_with(&[], "42u").ty, Ty::UnsignedInt);
    assert_eq!(check_with(&[], "42L").ty, Ty::Long);
    assert_eq!(check_with(&[], "42ull").ty, Ty::UnsignedLongLong);
    assert_eq!(check_with(&[], "1.5").ty, Ty::Double);
    assert_eq!(check_with(&[], "1.5f").ty, Ty::Float);
    assert_eq!(check_with(&[], "'a'").ty, Ty::Int);
}

#[test]
fn check_identifiers() {
    let checked = check_with(&["int x;"], "x");
    assert_eq!(checked.ty, Ty::Int);
    assert!(matches!(checked.kind, CheckedExprKind::Ident(_)));

    let msg = check_with_err(&[], "zzz");
    assert!(msg.contains("undeclared"), "{msg:?}");

    let msg = check_with_err(&["typedef int T;"], "T + 1");
    assert!(msg.contains("typedef"), "{msg:?}");
}

#[test]
fn check_arithmetic_conversions() {
    // `1 + 2u`: common type `unsigned int`, left side converted.
    let checked = check_with(&[], "1 + 2u");
    assert_eq!(checked.ty, Ty::UnsignedInt);
    let CheckedExprKind::Binary(_, l, r) = &checked.kind else {
        panic!("expected a binary node")
    };
    assert!(matches!(&l.kind, CheckedExprKind::Convert { .. }));
    assert!(matches!(&r.kind, CheckedExprKind::Int(_)));

    // Floats dominate without touching the float side.
    let checked = check_with(&["float f;"], "f + 1");
    assert_eq!(checked.ty, Ty::Float);

    // Shifts promote independently and keep the left type.
    let checked = check_with(&["char c;"], "c << 2");
    assert_eq!(checked.ty, Ty::Int);

    // Invalid mixes fail instead of converting.
    let msg = check_with_err(&["int *p;"], "1.5 + p");
    assert!(msg.contains("invalid operands"), "{msg:?}");

    // `%` needs integer operands (C17 §6.5.5p2): float remainders
    // are `fmod`, not an operator.
    assert_eq!(check_with(&["int a;", "int b;"], "a % b").ty, Ty::Int);
    let msg = check_with_err(&["double d;", "double e;"], "d % e");
    assert!(msg.contains("invalid operands"), "{msg:?}");
    let msg = check_with_err(&["int a;"], "a % 1.5");
    assert!(msg.contains("invalid operands"), "{msg:?}");
}

#[test]
fn check_pointer_arithmetic() {
    // `p + 1` keeps the pointer; the integer promotes in place.
    let checked = check_with(&["int *p;"], "p + 1");
    assert!(matches!(checked.ty, Ty::Pointer { .. }));
    let CheckedExprKind::Binary(_, _, r) = &checked.kind else {
        panic!("expected a binary node")
    };
    assert!(matches!(&r.kind, CheckedExprKind::Int(_)));

    // `p - q` is `long` (LP64 `ptrdiff_t`).
    let checked = check_with(&["int *p;", "int *q;"], "p - q");
    assert_eq!(checked.ty, Ty::Long);

    // Pointers never mix with floats.
    let msg = check_with_err(&["int *p;"], "p + 1.5");
    assert!(msg.contains("invalid operands"), "{msg:?}");
}

#[test]
fn check_comparisons_and_logic() {
    assert_eq!(check_with(&["int a;", "int b;"], "a < b").ty, Ty::Int);
    assert_eq!(check_with(&["int *p;"], "p == 0").ty, Ty::Int);
    assert_eq!(check_with(&["int a;", "int b;"], "a && b").ty, Ty::Int);

    // Non-zero integers never stand in for null pointers.
    let msg = check_with_err(&["int *p;"], "p == 1");
    assert!(msg.contains("invalid operands"), "{msg:?}");

    let msg = check_with_err(&["int a;", "int *p;"], "a < p");
    assert!(msg.contains("invalid operands"), "{msg:?}");
}

#[test]
fn check_unary_forms() {
    assert_eq!(check_with(&["char c;"], "-c").ty, Ty::Int);
    assert_eq!(check_with(&["int x;"], "!x").ty, Ty::Int);
    assert_eq!(check_with(&["int x;"], "++x").ty, Ty::Int);
    // `++` does not promote its operand into a conversion.
    let checked = check_with(&["int x;"], "++x");
    let CheckedExprKind::Unary(_, operand) = &checked.kind else {
        panic!("expected a unary node")
    };
    assert!(matches!(&operand.kind, CheckedExprKind::Ident(_)));

    assert_eq!(
        check_with(&["int x;"], "&x").ty,
        Ty::Pointer {
            qualifiers: Vec::new(),
            size: PtrSize::Default,
            pointee: Box::new(Ty::Int),
        }
    );
    assert_eq!(check_with(&["int *p;"], "*p").ty, Ty::Int);

    let msg = check_with_err(&[], "&3");
    assert!(msg.contains("rvalue"), "{msg:?}");
    let msg = check_with_err(&["int x;"], "*x");
    assert!(msg.contains("pointer"), "{msg:?}");
}

#[test]
fn check_assignment_forms() {
    // `float` converts down to `int` on assignment.
    let checked = check_with(&["int x;"], "x = 1.5");
    assert_eq!(checked.ty, Ty::Int);
    let CheckedExprKind::Assign(_, _, r) = &checked.kind else {
        panic!("expected an assignment node")
    };
    assert!(matches!(&r.kind, CheckedExprKind::Convert { .. }));

    // Null constants assign to pointers silently; other integers and
    // pointers convert with a GCC-style `-Wint-conversion` warning.
    check_with(&["int *p;"], "p = 0");
    let checked = check_with(&["int *p;"], "p = 1");
    let CheckedExprKind::Assign(_, _, r) = &checked.kind else {
        panic!("expected an assignment node")
    };
    assert!(matches!(&r.kind, CheckedExprKind::Convert { .. }));
    assert!(matches!(&r.ty, Ty::Pointer { .. }));
    let checked = check_with(&["int x;", "int *p;"], "x = p");
    let CheckedExprKind::Assign(_, _, r) = &checked.kind else {
        panic!("expected an assignment node")
    };
    assert!(matches!(&r.kind, CheckedExprKind::Convert { .. }));
    assert_eq!(r.ty, Ty::Int);
    let msg = check_with_err(&["int x;"], "1 = x");
    assert!(msg.contains("non-lvalue"), "{msg:?}");

    // Compound assignment keeps the left type.
    let checked = check_with(&["int x;"], "x += 2");
    assert_eq!(checked.ty, Ty::Int);
    let checked = check_with(&["int *p;"], "p += 2");
    assert!(matches!(checked.ty, Ty::Pointer { .. }));
    let msg = check_with_err(&["int x;", "int *p;"], "x += p");
    assert!(msg.contains("compound"), "{msg:?}");
    // `%=` needs integer operands too.
    let msg = check_with_err(&["double d;", "int a;"], "d %= a");
    assert!(msg.contains("compound"), "{msg:?}");
}

#[test]
fn check_null_pointer_comparison_converts() {
    // `p != 0` converts the null to the pointer type (C17 §6.5.9p2)
    // so both sides share a width downstream.
    let checked = check_with(&["int *p;"], "p != 0");
    assert_eq!(checked.ty, Ty::Int);
    let CheckedExprKind::Binary(_, l, r) = &checked.kind else {
        panic!("expected a binary node")
    };
    assert!(matches!(&l.ty, Ty::Pointer { .. }));
    assert!(matches!(&r.kind, CheckedExprKind::Convert { .. }));
    assert!(matches!(&r.ty, Ty::Pointer { .. }));
    let checked = check_with(&["int *p;"], "0 == p");
    let CheckedExprKind::Binary(_, l, r) = &checked.kind else {
        panic!("expected a binary node")
    };
    assert!(matches!(&l.kind, CheckedExprKind::Convert { .. }));
    assert!(matches!(&r.ty, Ty::Pointer { .. }));
}

#[test]
fn check_ternary_forms() {
    assert_eq!(check_with(&["int a;", "int b;", "int c;"], "c ? a : b").ty, Ty::Int);
    assert_eq!(check_with(&["int c;"], "c ? 1 : 2.5").ty, Ty::Double);
    let checked = check_with(&["int *p;", "int c;"], "c ? p : 0");
    assert!(matches!(checked.ty, Ty::Pointer { .. }));

    let msg = check_with_err(&["struct S { int m; };", "struct S s;"], "s ? 1 : 2");
    assert!(msg.contains("scalar"), "{msg:?}");
}

#[test]
fn check_calls() {
    let checked = check_with(&["int f(int, int);"], "f(1, 2)");
    assert_eq!(checked.ty, Ty::Int);

    let msg = check_with_err(&["int f(int, int);"], "f(1)");
    assert!(msg.contains("wrong number"), "{msg:?}");

    // Unprototyped calls apply default promotions (`float` → `double`).
    let checked = check_with(&["int g();"], "g(1.5f)");
    assert_eq!(checked.ty, Ty::Int);
    let CheckedExprKind::Call { args, .. } = &checked.kind else {
        panic!("expected a call node")
    };
    assert!(matches!(&args[0].kind, CheckedExprKind::Convert { .. }));
    assert_eq!(args[0].ty, Ty::Double);

    let msg = check_with_err(&["int x;"], "x(1)");
    assert!(msg.contains("not a function"), "{msg:?}");
}

#[test]
fn check_subscript_orders() {
    assert_eq!(check_with(&["int a[3];", "int i;"], "a[i]").ty, Ty::Int);
    assert_eq!(check_with(&["int a[3];", "int i;"], "i[a]").ty, Ty::Int);
    let msg = check_with_err(&["int a;", "int i;"], "a[i]");
    assert!(msg.contains("pointer"), "{msg:?}");
    let msg = check_with_err(&["int a[3];", "float f;"], "a[f]");
    assert!(msg.contains("integer"), "{msg:?}");
}

#[test]
fn check_member_access() {
    let decls = ["struct S { int m; };", "struct S s;", "struct S *p;"];
    assert_eq!(check_with(&decls, "s.m").ty, Ty::Int);
    assert_eq!(check_with(&decls, "p->m").ty, Ty::Int);
}

#[test]
fn check_member_resolves_to_index() {
    use crate::check::CheckedExprKind;

    // Fields resolve to positions, not names: the second field is index 1
    // even though both share the struct, and `->` resolves the same way.
    let decls = ["struct S { int a; float b; };", "struct S s;", "struct S *p;"];
    let checked = check_with(&decls, "s.b");
    assert!(matches!(
        checked.kind,
        CheckedExprKind::Member {
            path,
            arrow: false,
            ..
        } if path == vec![1]
    ));
    let checked = check_with(&decls, "p->a");
    assert!(matches!(
        checked.kind,
        CheckedExprKind::Member {
            path,
            arrow: true,
            ..
        } if path == vec![0]
    ));
}

#[test]
fn check_member_missing_field() {
    let msg = check_with_err(&["struct S { int m; };", "struct S s;"], "s.n");
    assert!(msg.contains("no member"), "{msg:?}");

    let msg = check_with_err(&["int x;"], "x.m");
    assert!(msg.contains("struct or union"), "{msg:?}");
}

#[test]
fn check_casts_and_sizeof() {
    assert_eq!(check_with(&["float x;"], "(int)x").ty, Ty::Int);
    assert_eq!(check_with(&["int x;"], "(void)x").ty, Ty::Void);
    assert_eq!(check_with(&["int x;"], "sizeof x").ty, Ty::UnsignedLong);
    assert_eq!(check_with(&[], "sizeof(int)").ty, Ty::UnsignedLong);

    let msg = check_with_err(&["struct S { int m; };", "int x;"], "(struct S)x");
    assert!(msg.contains("invalid operands to cast"), "{msg:?}");
}

// ---- statement/initializer checking (checked inline during parsing) ----

#[test]
fn body_undeclared_expression_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { x; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "use of undeclared identifier 'x'");
}

#[test]
fn body_undeclared_return_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void) { return y; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "use of undeclared identifier 'y'");
}

#[test]
fn void_function_returns_void_call() {
    // `return f();` for a void `f` is legal; a non-void value is not.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void); void g(void) { return f(); }", &mut sema, &mut interner);
    parser.parse_translation_unit();
}

#[test]
fn void_function_returns_value_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void g(void) { return 3; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "return with a value in void function");
}

#[test]
fn cast_to_union_initializes_member() {
    // A GNU cast to a union type initializes the member whose type
    // matches the operand.
    let checked = check_with(&["union U { int a; double b; };"], "(union U) 1.5");
    assert!(matches!(checked.kind, CheckedExprKind::CompoundLiteral { .. }));
    let msg = check_with_err(&["union U { int a; };"], "(union U) 1.5");
    assert!(msg.contains("not present in union"), "{msg:?}");
}

#[test]
fn body_struct_condition_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { struct S { int m; }; struct S s; if (s) {} }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "condition must have scalar type");
}

#[test]
fn body_float_switch_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { switch (1.5) {} }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "switch controlling expression must have integer type");
}

#[test]
fn body_float_case_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int x) { switch (x) { case 1.5: break; } }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "case label is not an integer constant expression");
}

#[test]
fn init_undeclared_name_panics() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x = y;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "use of undeclared identifier 'y'");
}

#[test]
fn init_pointer_to_int_warns_not_fails() {
    // Integer/pointer mixes convert with a GCC-style warning, including
    // in initializers (`locp = malloc(..)` with an implicit declaration).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int *p; int x = p;", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 2);
}

#[test]
fn init_narrowing_conversion_accepted() {
    // `=` converts like assignment; the definition still records.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x = 1.5;", &mut sema, &mut interner);
    let ids = parser.parse_declaration();
    assert!(parser.sema.decls[ids[0].0 as usize].defined);
    assert!(parser.sema.inits.contains_key(&ids[0]));
}

#[test]
fn stored_initializers_carry_conversions() {
    // Later phases consume the checked form: implicit conversions are
    // already inserted (`long x = 1` stores a Convert, not a bare Int).
    let init = checked_init("long x = 1;");
    let Initializer::Expr(expr) = &init else { panic!("{init:?}") };
    assert!(matches!(&expr.kind, CheckedExprKind::Convert { .. }));
    assert_eq!(expr.ty, Ty::Long);
}

#[test]
fn gnu_qualifier_spellings_open_declarations() {
    // `__const`/`__volatile`/`__restrict` open declarations and
    // parameter lists (system headers spell qualifiers this way, and
    // `__const` must not read as an old-style identifier list).
    let src = "typedef unsigned long size_t;\n\
               extern size_t strlen (__const char *__s);\n\
               __volatile int a = 0;\n\
               int f (const char *__restrict p) { return p[0]; }\n";
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(src, &mut sema, &mut interner);
    let unit = parser.parse_translation_unit();
    assert_eq!(unit.external_declarations.len(), 4);
}

// ---- AST dump (backend debugging) ----

#[test]
fn dump_golden() {
    // The s-expression dump pins its own format: every declaration,
    // body, and expression with its type. Changes to the format
    // update this string deliberately.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int g = 1; int f(int x) { int y = x + 1L; if (y) { return y; } return 0; }", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(
        crate::dump::dump_translation_unit(parser.sema, &parser.intern, &tu),
        "(var g#0 :int (int 1 :int))\n\
         (func f#2 :fn(int) -> int (block (param x#3 :int) \
         (decl (var y#4 :int (convert int (+ (convert long (ident x#3 :int) :long) (int 1L :long) :long) :int))) \
         (if (ident y#4 :int) (block (return (ident y#4 :int)))) \
         (return (int 0 :int))))"
    );
}

#[test]
fn dump_control_flow_and_members() {
    // Labels, gotos, member paths (by name), and string spellings.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "struct S { int m; }; const char *msg = \"hi\"; int f(struct S *p) { goto out; out: return p->m; }",
        &mut sema,
        &mut interner,
    );
    let tu = parser.parse_translation_unit();
    let dump = crate::dump::dump_translation_unit(parser.sema, &parser.intern, &tu);
    assert!(dump.contains("(goto out)"), "{dump}");
    assert!(dump.contains("(label out (return (member (ident p#"), "{dump}");
    assert!(dump.contains("m :int)))"), "{dump}");
    assert!(dump.contains("(str \"hi\" :"), "{dump}");
}

/// Body items of the single function `f` defined in `src`.
fn body_of_f(src: &str) -> Vec<BlockItem> {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(src, &mut sema, &mut interner);
    parser.parse_translation_unit();
    finish(&parser);
    let f = parser.intern.get_or_intern("f");
    let fid = parser.sema.lookup_symbol(f).expect("function declared");
    parser.sema.bodies[&fid].block.items.clone()
}

#[test]
fn asm_statement_parses_operands() {
    // `__asm__` starts a statement even though the same spelling names
    // symbols after declarators.
    let items = body_of_f("void f(int x){ int y; __asm__(\"\" : \"=r\"(y) : \"0\"(x) : \"memory\"); }");
    let [BlockItem::Decl { .. }, BlockItem::Stmt(Stmt::Asm(asm))] = items.as_slice() else {
        panic!("expected a declaration and an asm statement, got {items:?}");
    };
    assert_eq!(asm.template, "");
    assert!(!asm.is_volatile && !asm.is_goto);
    let [output] = asm.outputs.as_slice() else { panic!("expected one output") };
    assert_eq!(output.constraint, "=r");
    assert_eq!(output.name, None);
    let [input] = asm.inputs.as_slice() else { panic!("expected one input") };
    assert_eq!(input.tie(), Some(0));
    assert_eq!(asm.clobbers, vec!["memory".to_string()]);
    assert!(asm.labels.is_empty());
}

#[test]
fn asm_basic_form_is_the_no_colon_one() {
    // GCC's basic/extended split is the first colon: a basic template
    // passes through verbatim, so `%eax` stays literal rather than
    // becoming an operand reference (tcctest.c's `asm_dot_test`).
    let items = body_of_f("void f(void){ asm(\".text; mov %eax, %ecx\"); __asm__ __volatile__(\"nop\"); }");
    let [BlockItem::Stmt(Stmt::Asm(basic)), BlockItem::Stmt(Stmt::Asm(volatile_basic))] = items.as_slice() else {
        panic!("expected two asm statements, got {items:?}");
    };
    assert!(basic.is_basic);
    assert!(volatile_basic.is_volatile && volatile_basic.is_basic);
    // An empty operand list after a colon is still the extended form.
    let items = body_of_f("void f(void){ asm(\"mov %eax, %ecx\" : : : \"memory\"); }");
    let [BlockItem::Stmt(Stmt::Asm(extended))] = items.as_slice() else {
        panic!("expected an asm statement, got {items:?}");
    };
    assert!(!extended.is_basic);
}

#[test]
fn asm_qualifiers_and_named_operands() {
    let items = body_of_f("void f(int x){ int y; asm volatile(\"\" : [out] \"=r\"(y) : [in] \"r\"(x)); }");
    let [BlockItem::Decl { .. }, BlockItem::Stmt(Stmt::Asm(asm))] = items.as_slice() else {
        panic!("expected a declaration and an asm statement, got {items:?}");
    };
    assert!(asm.is_volatile && !asm.is_goto);
    assert!(asm.outputs[0].name.is_some());
    assert!(asm.inputs[0].name.is_some());
}

#[test]
fn asm_plus_operand() {
    let items = body_of_f("void f(int x){ asm(\"\" : \"+r\"(x)); }");
    let [BlockItem::Stmt(Stmt::Asm(asm))] = items.as_slice() else {
        panic!("expected an asm statement, got {items:?}");
    };
    assert!(asm.outputs[0].is_inout());
    assert!(asm.inputs.is_empty());
}

#[test]
fn asm_goto_labels_recorded() {
    let items = body_of_f("void f(void){ asm goto(\"\" :::: lab); lab: ; }");
    let [BlockItem::Stmt(Stmt::Asm(asm)), BlockItem::Stmt(Stmt::Label { .. })] = items.as_slice() else {
        panic!("expected asm goto then a label, got {items:?}");
    };
    assert!(asm.is_goto);
    assert_eq!(asm.labels.len(), 1);
}

#[test]
fn asm_output_must_be_lvalue() {
    assert_diag_contains(&tu_diagnostics("void f(void){ __asm__(\"\" : \"=r\"(1)); }"), "asm output is not an lvalue");
}

#[test]
fn asm_output_constraint_needs_direction() {
    assert_diag_contains(&tu_diagnostics("void f(int x){ asm(\"\" : \"r\"(x)); }"), "output operand constraint lacks '='");
}

#[test]
fn asm_tie_must_reference_an_output() {
    assert_diag_contains(&tu_diagnostics("void f(int x){ asm(\"\" : \"=r\"(x) : \"3\"(x)); }"), "asm operand number out of range");
}

#[test]
fn asm_nonempty_template_accepted() {
    let items = body_of_f("int f(int x){ int y; asm(\"movl %1, %0\" : \"=r\"(y) : \"r\"(x)); return y; }");
    let [BlockItem::Decl { .. }, BlockItem::Stmt(Stmt::Asm(asm)), ..] = items.as_slice() else {
        panic!("expected a declaration and an asm statement, got {items:?}");
    };
    assert_eq!(asm.template, "movl %1, %0");
}

#[test]
fn asm_goto_template_accepted() {
    let items = body_of_f("int f(int x){ asm goto(\"testl %0, %0; jnz %l[lab]\" : : \"r\"(x) : : lab); return 1; lab: return 2; }");
    let [BlockItem::Stmt(Stmt::Asm(asm)), ..] = items.as_slice() else {
        panic!("expected asm goto first, got {items:?}");
    };
    assert!(asm.is_goto);
    assert_eq!(asm.template, "testl %0, %0; jnz %l[lab]");
    assert_eq!(asm.labels.len(), 1);
}

#[test]
fn asm_goto_reference_must_be_listed() {
    assert_diag_contains(&tu_diagnostics("void f(int x){ asm goto(\"jmp %l[baz]\" : : \"r\"(x) : : lab); lab: ; }"), "undefined named operand 'baz'");
}

#[test]
fn asm_label_reference_needs_goto() {
    assert_diag_contains(&tu_diagnostics("void f(void){ asm(\"jmp %l[lab]\" ::: ); lab: ; }"), "undefined named operand 'lab'");
}

#[test]
fn asm_goto_needs_a_label_list() {
    assert_diag_contains(&tu_diagnostics("void f(void){ asm goto(\"nop\" :::); }"), "asm goto needs at least one label");
}

#[test]
fn asm_labels_require_goto() {
    assert_diag_contains(&tu_diagnostics("void f(void){ asm(\"nop\" ::: : lab); lab: ; }"), "asm label list requires asm goto");
}

#[test]
fn asm_goto_label_must_exist() {
    assert_diag_contains(&tu_diagnostics("void f(int x){ asm goto(\"jmp %l[missing]\" : : \"r\"(x) : : missing); }"), "label 'missing' used but not defined");
}

#[test]
fn asm_memory_input_needs_address() {
    assert_diag_contains(&tu_diagnostics("void f(int x){ asm(\"\" : : \"m\"(x + 1)); }"), "memory input is not directly addressable");
}

#[test]
fn asm_memory_bitfield_rejected() {
    assert_diag_contains(&tu_diagnostics("void f(void){ struct S { unsigned b : 3; } s; asm(\"\" : : \"m\"(s.b)); }"), "cannot take address of bit-field");
}

#[test]
fn asm_memory_letters_follow_the_target() {
    use crate::check::{constraint_letters, memory_only_constraint};
    use crate::target::Arch;
    // x86-64: `"Q"` is a high-byte register, so the operand is a value.
    assert!(!memory_only_constraint("Q", Arch::X86_64));
    assert!(memory_only_constraint("m", Arch::X86_64));
    // AArch64: `"Q"` and the `U..` addressing-mode qualifiers are
    // memory; `"Uci"` is a register.
    assert!(memory_only_constraint("Q", Arch::Aarch64));
    assert!(memory_only_constraint("+Q", Arch::Aarch64));
    assert!(memory_only_constraint("Ump", Arch::Aarch64));
    assert!(!memory_only_constraint("Uci", Arch::Aarch64));
    // RISC-V: `"A"` is an address held in a general-purpose register.
    assert!(memory_only_constraint("A", Arch::Riscv64));
    // A `U` token keeps its lowercase suffix.
    assert_eq!(constraint_letters("UmpQr"), ["Ump", "Q", "r"]);
    // The same letter means memory on one target and a register on
    // another, so the parser must consult the target: a non-lvalue
    // passes on x86-64 ...
    body_of_f("void f(int x){ asm(\"\" : : \"Q\"(x + 1)); }");
}

#[test]
fn asm_aarch64_memory_letter_needs_an_lvalue() {
    // ... but not on AArch64, where `"Q"` is memory.
    use crate::target::Target;
    let mut sema = Sema::with_target(Target::AARCH64);
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(int x){ asm(\"\" : : \"Q\"(x + 1)); }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "memory input is not directly addressable");
}

#[test]
fn asm_output_readonly_rejected() {
    assert_diag_contains(&tu_diagnostics("void f(void){ const int x = 1; __asm__(\"\" : \"=r\"(x)); }"), "assignment of read-only variable 'x'");
}

#[test]
fn file_scope_asm_collected() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("asm(\".globl x\\n.globl y\");\n__asm__(\"nop\");\nint f(void){ return 0; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let templates: Vec<&str> = parser.sema.toplevel_asm.iter().map(|(t, _)| t.as_str()).collect();
    assert_eq!(templates, [".globl x\n.globl y", "nop"]);
}

#[test]
fn file_scope_asm_rejects_operands() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("asm(\"nop\" : : \"memory\");", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "file-scope asm cannot have operands");
}

#[test]
fn naked_attribute_recorded() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("__attribute__((naked)) void f(void) { __asm__(\"nop\"); }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let f = parser.intern.get_or_intern("f");
    let fid = parser.sema.lookup_symbol(f).expect("function declared");
    assert!(parser.sema.decls[fid.0 as usize].naked);
}

#[test]
fn section_attribute_recorded() {
    // The object lands in the named section; a prototype may carry the
    // attribute for a definition that omits it (first wins in `declare`).
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser(
        "__attribute__((section(\".vectors\"))) int table[4];\n\
         __attribute__((section(\".noinit\"))) extern char scratch[8];\n\
         char scratch[8];",
        &mut sema,
        &mut interner,
    );
    parser.parse_translation_unit();
    let table = parser.intern.get_or_intern("table");
    let tid = parser.sema.lookup_symbol(table).expect("table declared");
    let vectors = parser.intern.get_or_intern(".vectors");
    assert_eq!(parser.sema.decls[tid.0 as usize].section, Some(vectors));
    let scratch = parser.intern.get_or_intern("scratch");
    let sid = parser.sema.lookup_symbol(scratch).expect("scratch declared");
    let noinit = parser.intern.get_or_intern(".noinit");
    assert_eq!(parser.sema.decls[sid.0 as usize].section, Some(noinit));
}

#[test]
fn naked_prototype_attribute_reaches_definition() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("__attribute__((naked)) void f(void);\nvoid f(void) { __asm__(\"nop\"); }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let f = parser.intern.get_or_intern("f");
    let fid = parser.sema.lookup_symbol(f).expect("function declared");
    assert!(parser.sema.decls[fid.0 as usize].naked);
}

#[test]
fn naked_body_rejects_c_code() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("__attribute__((naked)) int f(void) { return 1; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "naked function body may only contain basic asm statements");
}

#[test]
fn naked_body_rejects_extended_asm() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("__attribute__((naked)) void f(void) { __asm__(\"\" ::: \"memory\"); }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "naked function body may only contain basic asm statements");
}

// ---- the async extension (`-fglcc-async`) ----

/// [`test_parser`] with the extension enabled, the way `-fglcc-async`
/// reaches the parser.
fn async_parser<'a>(src: &str, sema: &'a mut Sema, interner: &'a mut Interner) -> Parser<'a> {
    sema.async_enabled = true;
    test_parser(src, sema, interner)
}

#[test]
fn future_specifier_needs_the_flag() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(_Future(int) x);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "`_Future` requires the glcc async extension");
}

#[test]
fn await_builtin_needs_the_flag() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void) { return __builtin_await(0); }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "`__builtin_await` requires the glcc async extension");
}

#[test]
fn async_attribute_without_the_flag_is_inert() {
    // No `-fglcc-async`: the attribute warns on stderr and is ignored,
    // so calls still see the declared result.
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("__attribute__((async)) int f(void); f()", &mut sema, &mut interner);
    parser.parse_external_declaration();
    let expr = parser.parse_expression();
    let checked = parser.sema.check_expr(&expr, &parser.intern).unwrap_or_else(|err| panic!("check failed: {}", err.message));
    assert_eq!(checked.ty, Ty::Int);
    assert!(sema.future_handle_of(&Ty::Int).is_none(), "no handle without the flag");
}

#[test]
fn await_outside_a_coroutine_is_rejected() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = async_parser("int f(void) { return __builtin_await(0); }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "`__builtin_await` is only valid inside an async function");
}

#[test]
fn variadic_coroutines_are_rejected() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = async_parser("__attribute__((async)) int f(int x, ...) { return x; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "async function cannot be variadic");
}

#[test]
fn async_main_is_rejected() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = async_parser("__attribute__((async)) int main(void) { return 0; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "async function `main` is not supported");
}

#[test]
fn naked_coroutines_are_rejected() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = async_parser("__attribute__((async)) __attribute__((naked)) void f(void) { }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "async function cannot be naked");
}

#[test]
fn async_on_an_object_is_inert() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = async_parser("__attribute__((async)) int x;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert!(sema.future_handle_of(&Ty::Int).is_none(), "no handle for objects");
}

/// `_Future(T)` is one nominal type per result type, however often it
/// is spelled.
#[test]
fn future_handles_are_interned_per_result_type() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = async_parser("_Future(int) a; _Future(int) b; _Future(char) c;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let int_handle = sema.future_handle_of(&Ty::Int).expect("int handle interned");
    let char_handle = sema.future_handle_of(&Ty::Char).expect("char handle interned");
    assert_ne!(int_handle, char_handle);
    assert_eq!(sema.future_result_of(&int_handle), Some(Ty::Int));
    let decls: Vec<crate::parse::Ty> = sema.decls.iter().filter(|b| b.kind == crate::sema::BindingKind::Var).map(|b| b.ty.clone()).collect();
    assert_eq!(decls.len(), 3, "{decls:?}");
    assert_eq!(decls[0], decls[1], "one handle per result type");
    assert_ne!(decls[0], decls[2]);
}

/// Calling an async function yields its handle, so the result can be
/// held, passed around, and awaited later.
#[test]
fn async_calls_are_typed_as_future_handles() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = async_parser("__attribute__((async)) int f(int x); f(1)", &mut sema, &mut interner);
    parser.parse_external_declaration();
    let expr = parser.parse_expression();
    let checked = parser.sema.check_expr(&expr, &parser.intern).unwrap_or_else(|err| panic!("check failed: {}", err.message));
    let handle = parser.sema.future_handle_of(&Ty::Int).expect("handle interned");
    assert_eq!(checked.ty, handle);
}

/// `__builtin_await` yields the future's result, and rejects operands
/// that are not futures.
#[test]
fn await_yields_the_result_and_rejects_non_futures() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = async_parser("__attribute__((async)) int f(int x); __builtin_await(f(1))", &mut sema, &mut interner);
    parser.parse_external_declaration();
    parser.sema.checking_async_fn.set(true);
    let expr = parser.parse_expression();
    let checked = parser.sema.check_expr(&expr, &parser.intern).unwrap_or_else(|err| panic!("check failed: {}", err.message));
    assert_eq!(checked.ty, Ty::Int);
    assert!(matches!(checked.kind, CheckedExprKind::Await(_)));
    parser.sema.checking_async_fn.set(false);

    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = async_parser("int x; __builtin_await(x)", &mut sema, &mut interner);
    parser.parse_external_declaration();
    parser.sema.checking_async_fn.set(true);
    let expr = parser.parse_expression();
    let err = parser.sema.check_expr(&expr, &parser.intern).expect_err("non-futures are rejected");
    assert!(err.message.contains("is not a future"), "{}", err.message);
}

// ---- C99 gap coverage: `_Imaginary`, universal character names, `_Pragma` ----

/// `_Imaginary` names a conditional C99 type this compiler does not
/// implement; every position that can name a type must diagnose it
/// rather than fall through to an unrelated parse error.
#[test]
fn imaginary_type_specifier_is_diagnosed_at_file_scope() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("_Imaginary float x;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "`_Imaginary` is not supported");
}

#[test]
fn imaginary_type_specifier_is_diagnosed_in_a_block() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { _Imaginary double x = 0.0; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "`_Imaginary` is not supported");
}

#[test]
fn imaginary_type_specifier_is_diagnosed_in_a_cast() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int f(void) { return (int)(_Imaginary float)1.0f; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "`_Imaginary` is not supported");
}

// ---- C11 gap coverage: `_Atomic`, `_Noreturn` ----

/// `_Atomic` is both a type specifier (`_Atomic(T)`) and a type
/// qualifier (`_Atomic T`). Atomic types have their base's
/// representation, which this compiler models by erasing the qualifier.
#[test]
fn atomic_erases_to_its_base_type() {
    use Ty::*;
    for (src, expected) in [
        ("_Atomic int", Int),
        ("_Atomic(int)", Int),
        ("_Atomic unsigned long", UnsignedLong),
        ("_Atomic(unsigned long)", UnsignedLong),
        ("_Atomic _Bool", Bool),
        ("_Atomic(long long)", LongLong),
    ] {
        assert_eq!(base_of(src), expected, "{src:?}");
    }
    assert_eq!(
        base_of("_Atomic(int *)"),
        Ty::Pointer {
            qualifiers: vec![],
            size: PtrSize::Default,
            pointee: Box::new(Int),
        }
    );
}

#[test]
fn atomic_declarations_parse() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("_Atomic int x = 0; typedef _Atomic _Bool atomic_flag;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let x = parser.intern.get_or_intern("x");
    let id = parser.sema.lookup_symbol(x).unwrap();
    assert_eq!(parser.sema.decls[id.0 as usize].ty, Ty::Int);
    let flag = parser.intern.get_or_intern("atomic_flag");
    let id = parser.sema.lookup_symbol(flag).unwrap();
    assert_eq!(parser.sema.decls[id.0 as usize].ty, Ty::Bool);
}

/// `_Noreturn` is a function specifier: it parses and is otherwise a
/// hint this compiler does not consume, but it only applies to functions.
#[test]
fn noreturn_function_specifier_parses() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("_Noreturn void abort(void);", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let abort = parser.intern.get_or_intern("abort");
    let id = parser.sema.lookup_symbol(abort).unwrap();
    assert!(matches!(parser.sema.decls[id.0 as usize].ty, Ty::Func { .. }));
}

#[test]
fn noreturn_on_object_is_rejected() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("_Noreturn int x;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "`_Noreturn` on non-function declaration");
}

#[test]
fn atomic_specifier_combines_with_nothing() {
    assert_base_error("_Atomic(int) long", "`_Atomic` with other type specifiers");
}

/// A UCN may open or continue an identifier (C17 §6.4.2.1); the lexer
/// decodes it to the named character.
#[test]
fn ucn_names_identifier_characters() {
    let cases = [("int \\u00e9 = 1;", "é"), ("int a\\u00e9b = 1;", "aéb")];
    for (src, want) in cases {
        let mut interner = Interner::new();
        let tokens = Lexer::new(src, &mut interner).lex_all();
        let id = match &tokens.iter().nth(1).unwrap().val {
            Token::Identifier(sym) => *sym,
            other => panic!("{other:?}"),
        };
        assert_eq!(interner.resolve(id), Some(want), "{src}");
    }
}

/// The lexer validates the UCN's spelling and range but leaves it raw in
/// the literal; escape decoding happens downstream.
#[test]
fn ucn_in_literals_decodes() {
    // Narrow string: code points at or below U+00FF narrow to one byte.
    let mut interner = Interner::new();
    let tokens = Lexer::new("\"\\u00e9\"", &mut interner).lex_all();
    let lit = match &tokens.iter().next().unwrap().val {
        Token::StringLiteral(s) => *s,
        other => panic!("{other:?}"),
    };
    assert_eq!(crate::eval::string_bytes(&[lit], &interner), Some(vec![0xE9]));

    // Wide string: the code point itself, LE in the target's `wchar_t`.
    let mut interner = Interner::new();
    let tokens = Lexer::new("L\"\\u00e9\"", &mut interner).lex_all();
    let lit = match &tokens.iter().next().unwrap().val {
        Token::StringLiteral(s) => *s,
        other => panic!("{other:?}"),
    };
    assert_eq!(crate::eval::string_bytes(&[lit], &interner), Some(vec![0xE9, 0, 0, 0]));

    // Narrow character constant.
    let mut interner = Interner::new();
    let tokens = Lexer::new("'\\u00e9'", &mut interner).lex_all();
    let lit = match &tokens.iter().next().unwrap().val {
        Token::Constant(Constant::Char(c)) => *c,
        other => panic!("{other:?}"),
    };
    assert_eq!(crate::eval::char_constant_in(&lit, &interner, false), Some(0xE9));
}

#[test]
fn ucn_with_too_few_digits_is_diagnosed() {
    assert_diag_contains(&lex_diagnostics("int \\u12 = 1;"), "invalid universal character name");
}

#[test]
fn ucn_with_non_hex_digits_is_diagnosed() {
    assert_diag_contains(&lex_diagnostics("\"\\uZZZZ\""), "expected 4 hexadecimal digits");
}

#[test]
fn ucn_surrogate_is_diagnosed() {
    assert_diag_contains(&lex_diagnostics("int \\uD800 = 1;"), "is not a valid character");
}

/// `_Pragma ( string-literal )` is a preprocessing operator, so the
/// native preprocessor removes it; raw-source callers must skip it too.
#[test]
fn pragma_operator_is_skipped_at_file_scope() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("_Pragma(\"pack(1)\")\nint x;", &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
}

#[test]
fn pragma_operator_is_skipped_in_a_block() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { _Pragma(\"pack(1)\") int x; (void)x; _Pragma(\"pack()\") }", &mut sema, &mut interner);
    parser.parse_translation_unit();
}

#[test]
fn pragma_operator_without_operand_is_diagnosed() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("_Pragma int x;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "`_Pragma` requires a parenthesized string literal");
}

// ---- error tolerance and machine-readable diagnostics ----

/// A parse error in one file-scope declaration must not stop the parser:
/// the following declaration is still parsed and the diagnostics list
/// holds one error per broken construct.
#[test]
fn recovery_reports_each_broken_declaration() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x = ; long y = ;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let errors = parser.diagnostics().iter().filter(|d| d.is_error()).count();
    assert_eq!(errors, 2, "{:#?}", parser.diagnostics());
}

/// Past the error budget the parser stops instead of cascading: the
/// remaining declarations are never parsed, and one final diagnostic
/// says why.
#[test]
fn recovery_stops_after_error_limit() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut src = String::new();
    for i in 0..(crate::parse::MAX_ERRORS + 5) {
        src.push_str(&format!("int x{i} = ;\n"));
    }
    src.push_str("int ok;\n");
    let mut parser = test_parser(&src, &mut sema, &mut interner);
    parser.parse_translation_unit();
    let diagnostics = parser.diagnostics();
    let errors = diagnostics.iter().filter(|d| d.is_error()).count();
    assert_eq!(errors, crate::parse::MAX_ERRORS + 1, "{diagnostics:#?}");
    assert!(diagnostics.iter().any(|d| d.message.contains("too many errors")), "{diagnostics:#?}");
    let ok = parser.intern.get_or_intern("ok");
    assert!(parser.sema.lookup_symbol(ok).is_none());
}

/// Recovery keeps going inside a function body, where statements (not
/// declarations) are the unit that resynchronizes.
#[test]
fn recovery_reports_each_broken_statement() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("void f(void) { int x = ; int y = ; }", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let errors = parser.diagnostics().iter().filter(|d| d.is_error()).count();
    assert_eq!(errors, 2, "{:#?}", parser.diagnostics());
}

/// After a bad declarator, resynchronization reaches the next
/// declaration, so a later valid one is still recorded in `Sema`.
#[test]
fn recovery_declares_the_next_declaration() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x = ; int ok;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert!(parser.has_errors());
    let ok = parser.intern.get_or_intern("ok");
    assert!(parser.sema.lookup_symbol(ok).is_some());
}

/// Lexical diagnostics are folded into the parser's list (they share the
/// accumulation sink), so a caller sees one ordered stream.
#[test]
fn lexer_diagnostics_reach_the_parser_sink() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int \\u12 = 1;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    assert_parse_error(&parser, "invalid universal character name");
}

/// Warnings are collected alongside errors but tagged so consumers can
/// tell them apart (and they never latch recovery).
#[test]
fn warnings_are_tagged_and_do_not_latch() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x = sizeof(void); int y = ;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let diagnostics = parser.diagnostics();
    assert!(
        diagnostics.iter().any(|d| d.severity == Severity::Warning && d.message.contains("sizeof")),
        "{diagnostics:#?}"
    );
    assert!(diagnostics.iter().any(|d| d.is_error()), "{diagnostics:#?}");
}

/// The JSON form is what an editor consumes: a `diagnostics` array with
/// severity, message, and 1-based line/column plus byte offsets.
#[test]
fn diagnostics_json_shape() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x = ;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let json = parser.diagnostics_json();
    assert!(json.starts_with("{\"diagnostics\":["), "{json}");
    assert!(json.contains("\"severity\":\"error\""), "{json}");
    assert!(json.contains("\"file\":\"<__glcc_source>\""), "{json}");
    assert!(json.contains("\"line\":1"), "{json}");
    assert!(json.contains("\"column\":"), "{json}");
    assert!(json.contains("\"start\":"), "{json}");
    assert!(json.contains("\"end\":"), "{json}");
}

/// Span rendering and JSON agree on the file name and 1-based line for a
/// diagnostic that carries a real (non-dummy) span.
#[test]
fn diagnostics_json_positions_match_the_source() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int x = ;", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let err = parser.diagnostics().into_iter().find(|d| d.is_error()).expect("an error");
    assert_eq!(err.primary.span.start_ln, 1);
    let json = parser.diagnostics_json();
    assert!(json.contains("\"line\":1"));
    assert!(json.contains("\"column\":9"), "{json}");
}

/// A single TU can carry a lexical, a syntactic, and a semantic error;
/// recovery keeps parsing so all three are reported instead of only the
/// first one encountered.
#[test]
fn recovery_reports_lexer_parser_and_sema_errors_together() {
    let mut sema = Sema::new();
    let mut interner = Interner::new();
    let mut parser = test_parser("int a = \\u12;\nint b = ;\nint c = missing;\n", &mut sema, &mut interner);
    parser.parse_translation_unit();
    let messages: Vec<String> = parser.diagnostics().into_iter().map(|d| d.message).collect();
    assert!(messages.iter().any(|m| m.contains("universal character name")), "{messages:?}");
    assert!(messages.iter().any(|m| m.contains("expected expression")), "{messages:?}");
    assert!(messages.iter().any(|m| m.contains("missing")), "{messages:?}");
}
