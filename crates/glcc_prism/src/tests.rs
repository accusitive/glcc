//! PrismIR end-to-end tests: C shader sources through the whole
//! pipeline (parse + profile check → MIR → prep → PrismIR), the ISA
//! round trip, the fusion peepholes, and the interface tables.

use std::string::String;
use std::vec::Vec;
use std::path::PathBuf;

use glcc_frontend as fe;
use glcc_mir as mir;

use crate::ir::{Alu, Program, Stmt};

/// One shader source through the pipeline, returning the ISA text and
/// the built program.
fn compile(src: &str) -> (String, Program) {
    let mut interner = fe::lexer::Interner::new();
    let arena = fe::bumpalo::Bump::new();
    let mut sema = fe::sema::Sema::with_target(fe::target::Target::native(), &arena);
    sema.lang = fe::lang::LangOpts {
        prism: true,
        ..Default::default()
    };
    let tokens = fe::lexer::Lexer::new(src, &mut interner).lex_all();
    let mut parser = fe::parse::Parser::new(src, tokens, sema.arena, &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    let errors = sema.prism_validate(&tu, &interner);
    assert!(errors.is_empty(), "profile errors: {errors:?}");
    let mut module = glcc_lower_mir::lower_translation_unit(&sema, &interner, &tu);
    let _ = &mut module;
    let prepared = crate::prep::prepare(&module).expect("prep");
    let interface = crate::interface::build(&sema, &interner).expect("interface");
    let func = prepared
        .funcs
        .iter()
        .find(|f| f.name == interface.entry)
        .expect("entry lowered");
    let program = crate::build::build(&interface, &prepared, func).expect("build");
    crate::validate::validate(&program).expect("validate");
    let text = crate::isa::print(&program);
    (text, program)
}

/// The prism.h surface a test shader needs (typedefs and builtins;
/// the driver stages the real header, tests inline the essentials).
const PRISM_PRELUDE: &str = r#"
typedef float float2 __attribute__((vector_size(8)));
typedef float float3 __attribute__((vector_size(12)));
typedef float float4 __attribute__((vector_size(16)));
typedef int int2 __attribute__((vector_size(8)));
typedef int int4 __attribute__((vector_size(16)));
typedef struct __prism_sampler2d { unsigned int __prism_opaque[2]; } sampler2D;
typedef struct { float4 c0, c1, c2, c3; } float4x4;
float4 tex2D(sampler2D s, float2 uv);
float2 splat2(float s);
float4 splat4(float s);
float4 mul4x4(float4x4 m, float4 v);
#define mul(m, v) mul4x4(m, v)
"#;

const FRAG_SHADER: &str = r#"
struct FSIn  { [[prism::semantic(TEXCOORD0)]] float2 uv; };
struct FSOut { [[prism::semantic(COLOR0)]] float4 color; };
sampler2D tex;
float4 tint [[prism::uniform]];
[[prism::fragment]]
struct FSOut fs_main(struct FSIn i) {
    struct FSOut o;
    float4 t = tex2D(tex, i.uv);
    float k = 0.0f;
    for (int j = 0; j < 3; j = j + 1) {
        if (t.x > 0.5f) { k = k + t.y; } else { k = k - t.z; }
    }
    o.color = t * tint + splat4(k);
    return o;
}
"#;

const VS_SHADER: &str = r#"
struct VSIn  { [[prism::semantic(POSITION)]] float4 pos; [[prism::semantic(TEXCOORD0)]] float2 uv; };
struct VSOut { [[prism::semantic(POSITION)]] float4 pos; [[prism::semantic(TEXCOORD0)]] float2 uv; };
float4x4 mvp [[prism::uniform]];
[[prism::vertex]]
struct VSOut vs_main(struct VSIn i) {
    struct VSOut o;
    o.pos = mul4x4(mvp, i.pos);
    o.uv = i.uv;
    return o;
}
"#;

fn strip_comments(text: &str) -> String {
    text.lines()
        .filter(|line| !line.trim_start().starts_with(';'))
        .collect::<Vec<_>>()
        .join("\n")
}

/// print → parse → print is the identity (the golden format must
/// round-trip; canonicalization shows up as the first print changing).
#[test]
fn isa_round_trips() {
    for src in [FRAG_SHADER, VS_SHADER] {
        let text = compile(&format!("{PRISM_PRELUDE}\n{src}")).0;
        let parsed = crate::isa::parse(&text).expect("parse");
        let again = crate::isa::print(&parsed);
        assert_eq!(strip_comments(&text), strip_comments(&again), "round trip changed the text");
    }
}

/// mul4x4 lowers to one dp4 per output lane: the shared dot fusion the
/// plan requires.
#[test]
fn mul4x4_fuses_to_dp4() {
    let (_, program) = compile(&format!("{PRISM_PRELUDE}\n{VS_SHADER}"));
    let mut dps = 0;
    let mut stack = vec![&program.body];
    while let Some(body) = stack.pop() {
        for stmt in &body.stmts {
            match stmt {
                Stmt::Op(op) if matches!(op.kind, Alu::Dp4) => dps += 1,
                Stmt::If { then, els, .. } => {
                    stack.push(then);
                    stack.push(els);
                }
                Stmt::Loop { body: inner, .. } => stack.push(inner),
                _ => {}
            }
        }
    }
    assert_eq!(dps, 4, "four dp4 dots expected:\n{}", crate::isa::print(&program));
}

/// A loop with both `break` and `continue` keeps the carried registers
/// and the exit value distinct.
#[test]
fn loop_break_continue() {
    let src = r#"
struct FSIn  { [[prism::semantic(TEXCOORD0)]] float2 uv; };
struct FSOut { [[prism::semantic(COLOR0)]] float4 color; };
[[prism::fragment]]
struct FSOut fs_main(struct FSIn i) {
    struct FSOut o;
    float k = i.uv.x;
    for (int j = 0; j < 8; j = j + 1) {
        if (j == 2) { continue; }
        if (j == 5) { break; }
        k = k * 0.5f;
    }
    o.color = splat4(k);
    return o;
}
"#;
    let (text, program) = compile(&format!("{PRISM_PRELUDE}\n{src}"));
    // One loop; its body carries k and j and contains both controls.
    let mut loops = Vec::new();
    let mut stack = vec![&program.body];
    while let Some(body) = stack.pop() {
        for stmt in &body.stmts {
            if let Stmt::Loop { carried, body: inner, .. } = stmt {
                loops.push((carried.len(), inner.clone()));
            }
        }
    }
    assert_eq!(loops.len(), 1, "one loop:\n{text}");
    let (carried, inner) = &loops[0];
    assert_eq!(*carried, 2, "k and j are carried:\n{text}");
    // Two breaks: the `j == 5` break and the loop condition's exit.
    let counts = crate::build::test_controls(inner);
    assert_eq!(counts.0, 2, "breaks:\n{text}");
    assert_eq!(counts.1, 1, "one continue:\n{text}");
}

/// A `do { } while` whose body starts at the entry block still
/// promotes its locals (the prep entry shim).
#[test]
fn do_while_at_entry() {
    let src = r#"
struct FSIn  { [[prism::semantic(TEXCOORD0)]] float2 uv; };
struct FSOut { [[prism::semantic(COLOR0)]] float4 color; };
[[prism::fragment]]
struct FSOut fs_main(struct FSIn i) {
    struct FSOut o;
    float k = 1.0f;
    do { k = k * 0.5f; } while (k > i.uv.x);
    o.color = splat4(k);
    return o;
}
"#;
    let (text, program) = compile(&format!("{PRISM_PRELUDE}\n{src}"));
    let mut loops = 0;
    let mut stack = vec![&program.body];
    while let Some(body) = stack.pop() {
        for stmt in &body.stmts {
            match stmt {
                Stmt::Loop { .. } => loops += 1,
                Stmt::If { then, els, .. } => {
                    stack.push(then);
                    stack.push(els);
                }
                _ => {}
            }
        }
    }
    assert_eq!(loops, 1, "one loop:\n{text}");
    assert!(text.contains("continue"), "the latch continues:\n{text}");
    assert!(text.contains("break"), "the exit breaks:\n{text}");
    let _ = program;
}

/// Early return inside a branch: each exit carries the full output.
#[test]
fn early_return() {
    let src = r#"
struct FSIn  { [[prism::semantic(TEXCOORD0)]] float2 uv; };
struct FSOut { [[prism::semantic(COLOR0)]] float4 color; };
[[prism::fragment]]
struct FSOut fs_main(struct FSIn i) {
    struct FSOut o;
    if (i.uv.x > 0.5f) {
        o.color = splat4(1.0f);
        return o;
    }
    o.color = splat4(0.0f);
    return o;
}
"#;
    let (_, program) = compile(&format!("{PRISM_PRELUDE}\n{src}"));
    let mut exits = 0;
    let mut stack = vec![&program.body];
    while let Some(body) = stack.pop() {
        for stmt in &body.stmts {
            match stmt {
                Stmt::Exit(_) => exits += 1,
                Stmt::If { then, els, .. } => {
                    stack.push(then);
                    stack.push(els);
                }
                _ => {}
            }
        }
    }
    assert_eq!(exits, 2, "both branches return:\n{}", crate::isa::print(&program));
}

/// Uniform packing: float4x4 takes four consecutive c-registers,
/// scalars pack four per register in declaration order.
#[test]
fn uniform_creg_packing() {
    let src = r#"
struct VSIn  { [[prism::semantic(POSITION)]] float4 pos; };
struct VSOut { [[prism::semantic(POSITION)]] float4 pos; };
float4 a [[prism::uniform]];
float s [[prism::uniform]];
float t [[prism::uniform]];
float4 b [[prism::uniform]];
float u [[prism::uniform]];
float4x4 m [[prism::uniform]];
[[prism::vertex]]
struct VSOut vs_main(struct VSIn i) {
    struct VSOut o;
    o.pos = a * i.pos + splat4(s + t + u);
    return o;
}
"#;
    let mut interner = fe::lexer::Interner::new();
    let arena = fe::bumpalo::Bump::new();
    let mut sema = fe::sema::Sema::with_target(fe::target::Target::native(), &arena);
    sema.lang = fe::lang::LangOpts {
        prism: true,
        ..Default::default()
    };
    let text = format!("{PRISM_PRELUDE}\n{src}");
    let tokens = fe::lexer::Lexer::new(&text, &mut interner).lex_all();
    let mut parser = fe::parse::Parser::new(&text, tokens, sema.arena, &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert!(sema.prism_validate(&tu, &interner).is_empty());
    let interface = crate::interface::build(&sema, &interner).expect("interface");
    let creg = |name: &str| interface.uniforms.iter().find(|u| u.name == name).map(|u| (u.creg, u.leaves.len()));
    assert_eq!(creg("a"), Some((0, 1)), "a takes c0");
    assert_eq!(creg("s"), Some((1, 1)), "s packs into c1");
    assert_eq!(creg("t"), Some((1, 1)), "t shares c1");
    assert_eq!(creg("b"), Some((2, 1)), "b takes c2 (vectors never share)");
    assert_eq!(creg("u"), Some((3, 1)), "u starts a fresh scalar register after b");
    assert_eq!(creg("m"), Some((4, 4)), "m takes c4..c7");
    assert_eq!(interface.uniforms.iter().find(|u| u.name == "m").unwrap().leaves.iter().map(|leaf| leaf.creg).collect::<Vec<_>>(), vec![4, 5, 6, 7]);
}

/// std140 sizes for SPIR-V's UBO form: vec3 strides 16, arrays round
/// their element stride, structs round their size.
#[test]
fn uniform_std140_sizes() {
    assert_eq!(crate::interface::std140_size(&fe::parse::Ty::Float), 4);
    assert_eq!(
        crate::interface::std140_size(&fe::parse::Ty::Vector {
            elem: Box::new(fe::parse::Ty::Float),
            len: 3
        }),
        12
    );
    assert_eq!(
        crate::interface::std140_size(&fe::parse::Ty::Vector {
            elem: Box::new(fe::parse::Ty::Float),
            len: 4
        }),
        16
    );
    assert_eq!(
        crate::interface::std140_size(&fe::parse::Ty::Array {
            elem: Box::new(fe::parse::Ty::Float),
            bound: Some(fe::parse::ArrayBound::Fixed(4)),
            quals: Default::default(),
        }),
        64,
        "std140: array of scalars strides to 16"
    );
}

/// The golden outputs for the two sample shaders (the ISA is the
/// committed artifact; regenerate with GLCC_PRISM_UPDATE_GOLDENS=1).
#[test]
fn goldens() {
    for (name, src) in [("frag", FRAG_SHADER), ("vs", VS_SHADER)] {
        let text = compile(&format!("{PRISM_PRELUDE}\n{src}")).0;
        let golden = golden_path(name);
        if std::env::var_os("GLCC_PRISM_UPDATE_GOLDENS").is_some() {
            std::fs::write(&golden, &text).expect("write golden");
        }
        let expected = std::fs::read_to_string(&golden).expect("golden file");
        assert_eq!(text, expected, "golden drift for {name}");
    }
}

/// A helper holding a uniform swizzle and a sample: prep force-inlines
/// it, so the inliner must remap the prism ops (it panicked on
/// shuffles and samples before), and the resulting shuffles over
/// inline uniform reads must validate against their leaf types.
#[test]
fn helper_with_uniform_swizzle_and_sample() {
    let prelude = r#"
float3 splat3(float s);
float max(float a, float b);
"#;
    let src = r#"
struct FSIn  { [[prism::semantic(NORMAL)]] float3 normal; };
struct FSOut { [[prism::semantic(COLOR0)]] float4 color; };
sampler2D tex;
float4 tint [[prism::uniform]];
float4 light [[prism::uniform]];

static float3 shade(float3 n) {
    float3 l = light.xyz;
    float ndl = l.x * n.x + l.y * n.y + l.z * n.z;
    float4 s = tex2D(tex, splat2(0.5f));
    return n * splat3(max(ndl, 0.0f)) + s.xyz;
}

[[prism::fragment]]
struct FSOut fs_main(struct FSIn i) {
    struct FSOut o;
    o.color = tint * splat4(shade(i.normal).x);
    return o;
}
"#;
    let (text, program) = compile(&format!("{PRISM_PRELUDE}\n{prelude}\n{src}"));
    assert!(text.contains("mul") || text.contains("mad"), "the shaded product survives:\n{text}");
    assert!(text.contains("sample"), "the helper's sample survives:\n{text}");
    assert!(text.contains("shuffle u1.c1"), "the uniform swizzle survives:\n{text}");
    let _ = program;
}

/// validate catches a loop whose continue carries the wrong arity.
#[test]
fn validate_catches_continue_arity() {
    let mut program = empty_program();
    let a = program.push_reg(crate::ir::RegTy::Scalar(crate::ir::Scalar::F32));
    let b = program.push_reg(crate::ir::RegTy::Scalar(crate::ir::Scalar::F32));
    let ia = program.push_reg(crate::ir::RegTy::Scalar(crate::ir::Scalar::F32));
    let ib = program.push_reg(crate::ir::RegTy::Scalar(crate::ir::Scalar::F32));
    program.body.stmts.push(Stmt::Loop {
        carried: vec![(a, ia), (b, ib)],
        body: {
            let mut body = crate::ir::Body::new();
            body.stmts.push(Stmt::Continue(vec![a].into()));
            body
        },
        results: vec![],
    });
    let error = crate::validate::validate(&program).expect_err("arity error");
    assert!(error.contains("continue carries 1 values, the loop carries 2"), "{error}");
}

/// The committed golden file for a test shader, under
/// `crates/glcc_prism/tests/goldens/`.
fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/goldens").join(format!("{name}.isa"))
}

fn empty_program() -> Program {
    Program {
        name: "t".into(),
        stage: crate::ir::Stage::Fragment,
        interface: crate::interface::Interface {
            stage: crate::ir::Stage::Fragment,
            entry: "t".into(),
            inputs: vec![],
            outputs: vec![],
            uniforms: vec![],
            samplers: vec![],
        },
        regs: Vec::new(),
        body: Default::default(),
    }
}


