use super::*;

use cranelift_jit::{JITBuilder, JITModule};

use glcc_frontend as fe;
use glcc_mir as mir;

use super::{lower_translation_unit, naked_asm_text};

/// The driver's optimization pipeline, shared through `glcc_mir`.
fn run_opt_pipeline(module: &mut mir::Module) {
    mir::optimize(module, mir::OptLevel::Speed);
}

/// The optimized pipeline collapses C integer-promotion round
/// trips: `x > 0` lowers to `icmp` -> `zext` -> `icmp != 0`, and a
/// branch must end up testing the comparison directly.
#[test]
fn optimized_pipeline_simplifies_conditions() {
    let src = "int check(int x) { if (x > 0 && x < 100) return 1; return 0; }\n\
               int main() { return check(5); }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    let check = module.funcs.iter().find(|f| f.name == "check").expect("check lowered");
    let zext_results: std::collections::HashSet<mir::Value> = check
        .blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter_map(|inst| match &check.insts[*inst].kind {
            mir::InstKind::Zext(_) => check.insts[*inst].result,
            _ => None,
        })
        .collect();
    for inst in check.blocks.iter().flat_map(|block| block.insts.iter()) {
        if let mir::InstKind::Icmp { lhs, rhs, .. } = &check.insts[*inst].kind {
            assert!(!zext_results.contains(lhs) && !zext_results.contains(rhs), "promotion survived in check:\n{dump}");
        }
    }
    assert_eq!(jit_run_main(module), 1);
}

/// A short-circuit `&&`/`||` whose right side is side-effect-free
/// becomes a select, then a bitwise operation over the normalized
/// comparisons.
#[test]
fn optimized_pipeline_if_converts_boolean_operators() {
    let src = "__attribute__((noinline)) int land(int a, int b) { return (a > 0) && (b > 0); }\n\
               __attribute__((noinline)) int lor(int a, int b) { return (a > 0) || (b > 0); }\n\
               int main() {\n\
                   return land(1, 2) + land(1, -1) + lor(1, -1) + lor(-1, -1);\n\
               }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    for (name, expected) in [("land", "and"), ("lor", "or")] {
        let func = module.funcs.iter().find(|f| f.name == name).expect("function lowered");
        let kinds: Vec<&mir::InstKind> = func.blocks.iter().flat_map(|block| block.insts.iter()).map(|inst| &func.insts[*inst].kind).collect();
        assert!(
            kinds.iter().any(|kind| match (expected, kind) {
                ("and", mir::InstKind::And(..)) | ("or", mir::InstKind::Or(..)) => true,
                _ => false,
            }),
            "{name} did not become a bitwise operation:\n{dump}"
        );
        assert!(!kinds.iter().any(|kind| matches!(kind, mir::InstKind::Select { .. })), "{name} kept a select:\n{dump}");
    }
    assert_eq!(jit_run_main(module), 2);
}

/// Chained `&&`/`||` combines diamonds; speculating each small pure
/// arm makes the whole function branchless and lets the bitwise
/// simplifications see nested boolean values.
#[test]
fn optimized_pipeline_if_converts_chained_boolean_operators() {
    let src = "__attribute__((noinline)) int land3(int a, int b, int c) {\n\
                   return (a > 0) && (b > 0) && (c > 0);\n\
               }\n\
               __attribute__((noinline)) int lor3(int a, int b, int c) {\n\
                   return (a > 0) || (b > 0) || (c > 0);\n\
               }\n\
               int main() {\n\
                   return land3(1, 2, 3) + land3(1, 0, 1)\n\
                       + lor3(0, 0, 1) + lor3(0, 0, 0);\n\
               }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    for name in ["land3", "lor3"] {
        let func = module.funcs.iter().find(|f| f.name == name).expect("function lowered");
        assert!(
            !func.blocks.iter().any(|block| matches!(block.term, Some(mir::Terminator::CondBr { .. }))),
            "{name} kept a conditional branch:\n{dump}"
        );
        assert!(
            !func
                .blocks
                .iter()
                .flat_map(|block| block.insts.iter())
                .any(|inst| { matches!(func.insts[*inst].kind, mir::InstKind::Select { .. }) }),
            "{name} kept a select:\n{dump}"
        );
    }
    assert_eq!(jit_run_main(module), 2);
}

/// Bitwise combinations of normalized comparisons stay boolean until
/// the final extension, avoiding one extension per operand.
#[test]
fn optimized_pipeline_keeps_bitwise_boolean_ops_boolean() {
    let src = "__attribute__((noinline)) int band(int a, int b) { return (a > 0) & (b > 0); }\n\
               __attribute__((noinline)) int bor(int a, int b) { return (a > 0) | (b > 0); }\n\
               __attribute__((noinline)) int bxor(int a, int b) { return (a > 0) ^ (b > 0); }\n\
               int main() {\n\
                   return band(1, 2) + band(1, -1) * 2\n\
                       + bor(1, -1) * 4 + bor(-1, -1) * 8\n\
                       + bxor(1, 2) * 16 + bxor(1, -1) * 32;\n\
               }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    for name in ["band", "bor", "bxor"] {
        let func = module.funcs.iter().find(|f| f.name == name).expect("function lowered");
        let zexts = func
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .filter(|inst| matches!(func.insts[**inst].kind, mir::InstKind::Zext(..)))
            .count();
        assert_eq!(zexts, 1, "{name} did not keep one extension:\n{dump}");
        assert!(
            !func
                .blocks
                .iter()
                .flat_map(|block| block.insts.iter())
                .any(|inst| { matches!(func.insts[*inst].kind, mir::InstKind::Select { .. }) }),
            "{name} kept a select:\n{dump}"
        );
    }
    assert_eq!(jit_run_main(module), 37);
}

/// A ternary producing only 0 and 1 is a boolean extension. The
/// inverted arm must negate with a comparison, not a bitwise `not`,
/// because `Bool` is byte-carried rather than a one-bit integer.
#[test]
fn optimized_pipeline_turns_boolean_ternaries_into_extensions() {
    let src = "__attribute__((noinline)) int yes(int x) { return (x > 0) ? 1 : 0; }\n\
               __attribute__((noinline)) int no(int x) { return (x > 0) ? 0 : 1; }\n\
               int main() {\n\
                   return yes(5) + yes(-1) * 2 + no(5) * 4 + no(-1) * 8;\n\
               }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    for name in ["yes", "no"] {
        let func = module.funcs.iter().find(|f| f.name == name).expect("function lowered");
        let kinds: Vec<&mir::InstKind> = func.blocks.iter().flat_map(|block| block.insts.iter()).map(|inst| &func.insts[*inst].kind).collect();
        assert!(
            kinds.iter().any(|kind| matches!(kind, mir::InstKind::Zext(..))),
            "{name} did not become an extension:\n{dump}"
        );
        assert!(!kinds.iter().any(|kind| matches!(kind, mir::InstKind::Select { .. })), "{name} kept a select:\n{dump}");
        assert!(
            !kinds.iter().any(|kind| matches!(kind, mir::InstKind::Not(..))),
            "{name} negated a byte-carried bool bitwise:\n{dump}"
        );
    }
    assert_eq!(jit_run_main(module), 9);
}

/// The signed form of a boolean ternary sign-extends the one-bit
/// condition, so true becomes -1 even when the carrier grows to a
/// wider Clif type.
#[test]
fn optimized_pipeline_sign_extends_boolean_ternaries() {
    let src = "__attribute__((noinline)) int yes(int x) { return (x > 0) ? -1 : 0; }\n\
               __attribute__((noinline)) int no(int x) { return (x > 0) ? 0 : -1; }\n\
               int main() {\n\
                   int a = yes(5), b = yes(-1), c = no(5), d = no(-1);\n\
                   return (a == -1) + 2 * (b == 0) + 4 * (c == 0) + 8 * (d == -1)\n\
                       + 16 * (a + b + c + d == -2);\n\
               }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    for name in ["yes", "no"] {
        let func = module.funcs.iter().find(|f| f.name == name).expect("function lowered");
        let kinds: Vec<&mir::InstKind> = func.blocks.iter().flat_map(|block| block.insts.iter()).map(|inst| &func.insts[*inst].kind).collect();
        assert!(
            kinds.iter().any(|kind| matches!(kind, mir::InstKind::Sext(..))),
            "{name} did not become a sign extension:\n{dump}"
        );
        assert!(!kinds.iter().any(|kind| matches!(kind, mir::InstKind::Select { .. })), "{name} kept a select:\n{dump}");
    }
    assert_eq!(jit_run_main(module), 31);
}

/// Comparisons against a constant narrow to the source width with
/// the right signedness, and constants outside the source range
/// collapse to a fixed truth value.
#[test]
fn optimized_pipeline_narrows_comparisons_with_constants() {
    let src = "int z(unsigned char x) { return (int)x < 200; }\n\
               int z_high(unsigned char x) { return (int)x < 300; }\n\
               int z_eq_high(unsigned char x) { return (int)x == 300; }\n\
               int s(signed char x) { return (int)x < 100; }\n\
               int s_low(signed char x) { return (int)x < -200; }\n\
               int main() {\n\
                   if (!z(199) || z(200) || z(255)) return 1;\n\
                   if (!s(99) || s(100) || !s(-128)) return 2;\n\
                   if (!z_high(0) || z_eq_high(255) || s_low(0)) return 3;\n\
                   return 0;\n\
               }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    for func in &module.funcs {
        let widened: std::collections::HashSet<mir::Value> = func
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .filter_map(|inst| match &func.insts[*inst].kind {
                mir::InstKind::Zext(_) | mir::InstKind::Sext(_) => func.insts[*inst].result,
                _ => None,
            })
            .collect();
        for inst in func.blocks.iter().flat_map(|block| block.insts.iter()) {
            if let mir::InstKind::Icmp { lhs, rhs, .. } = &func.insts[*inst].kind {
                assert!(
                    !widened.contains(lhs) && !widened.contains(rhs),
                    "constant comparison stayed widened in {}:\n{dump}",
                    func.name
                );
            }
        }
    }
    assert_eq!(jit_run_main(module), 0);
}

/// `(unsigned char)(x & 0xff)` keeps only the low byte: the mask is
/// redundant under the truncation, and the result stays correct.
#[test]
fn optimized_pipeline_absorbs_covering_masks() {
    let src = "unsigned char low(unsigned x) { return (unsigned char)(x & 0xff); }\n\
               int main() { return low(0x1ff); }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    let low = module.funcs.iter().find(|f| f.name == "low").expect("low lowered");
    let has_and = low
        .blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .any(|inst| matches!(low.insts[*inst].kind, mir::InstKind::And(..)));
    assert!(!has_and, "mask survived in low:\n{dump}");
    assert_eq!(jit_run_main(module), 0xff);
}

/// SROA splits non-escaping struct and array locals into scalar
/// slots, and `Mem2Reg` then promotes them: none of the functions
/// keep stack traffic, including the one whose fields change in a
/// loop (which needs a phi per promoted field).
#[test]
fn optimized_pipeline_scalarizes_aggregate_locals() {
    let src = "struct point { int x, y; };\n\
               int sum_points(int a, int b) {\n\
                   struct point q;\n\
                   q.x = a; q.y = b;\n\
                   return q.x + q.y;\n\
               }\n\
               int sum_array(int a, int b, int c) {\n\
                   int v[3];\n\
                   v[0] = a; v[1] = b; v[2] = c;\n\
                   return v[0] * 100 + v[1] * 10 + v[2];\n\
               }\n\
               int sum_loop(int n) {\n\
                   struct point q;\n\
                   q.x = 0; q.y = 0;\n\
                   for (int i = 0; i < n; i++) {\n\
                       q.x = q.x + i;\n\
                       q.y = q.y + q.x;\n\
                   }\n\
                   return q.x * 100 + q.y;\n\
               }\n\
               int main() {\n\
                   return sum_points(1, 2) + sum_array(1, 2, 3) + sum_loop(4);\n\
               }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    for name in ["sum_points", "sum_array", "sum_loop"] {
        let func = module.funcs.iter().find(|f| f.name == name).expect("function is lowered");
        let memory_ops = func
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .filter(|inst| {
                matches!(
                    func.insts[**inst].kind,
                    mir::InstKind::Alloca { .. } | mir::InstKind::Load { .. } | mir::InstKind::Store { .. }
                )
            })
            .count();
        assert_eq!(memory_ops, 0, "stack traffic survived in {name}:\n{dump}");
    }
    assert_eq!(jit_run_main(module), 736);
}

/// Two identical additions collapse to one shared value; the
/// product keeps computing `(a + b) * (a + b)`.
#[test]
fn optimized_pipeline_eliminates_redundant_expressions() {
    let src = "int twice(int a, int b) { int x = a + b; int y = a + b; return x * y; }\n\
               int main() { return twice(3, 4); }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    let twice = module.funcs.iter().find(|f| f.name == "twice").expect("twice lowered");
    let adds = twice
        .blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter(|inst| matches!(twice.insts[**inst].kind, mir::InstKind::Add(..)))
        .count();
    assert_eq!(adds, 1, "duplicate addition survived in twice:\n{dump}");
    assert_eq!(jit_run_main(module), 49);
}

/// The dead `if (0)` arm folds away, and its join block sinks into
/// the entry, leaving a single return block.
#[test]
fn optimized_pipeline_sinks_dead_join_blocks() {
    let src = "int f(int x) { if (0) x = 1; return x; }\n\
               int main() { return f(9); }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    let f = module.funcs.iter().find(|f| f.name == "f").expect("f lowered");
    let live = f.blocks.iter().filter(|block| !matches!(block.term, Some(mir::Terminator::Unreachable))).count();
    assert_eq!(live, 1, "join block survived in f:\n{dump}");
    assert_eq!(jit_run_main(module), 9);
}

/// `__builtin_clrsbll` of the minimum i64 is zero; its lowering
/// shifts the literal right by 63, which must sign-extend the
/// pattern rather than shift the positive `i128` the parser sees.
#[test]
fn optimized_pipeline_computes_clrsb_of_min_i64() {
    let src = "int main() { return __builtin_clrsbll(0x8000000000000000ULL); }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    assert_eq!(jit_run_main(module), 0, "clrsb of i64 min:\n{dump}");
}

/// The `setjmp` builtin lowers to a runtime call whose containing
/// function must keep its slots addressable for the `longjmp` path.
#[test]
fn setjmp_builtin_marks_the_function() {
    let module = lower_module(
        "void f(void *b) { if (__builtin_setjmp(b)) __builtin_longjmp(b, 1); }\n\
         int g(void) { return 1; }",
    );
    let f = module.funcs.iter().find(|f| f.name == "f").expect("f lowered");
    assert!(f.returns_twice, "setjmp call did not mark f");
    let g = module.funcs.iter().find(|f| f.name == "g").expect("g lowered");
    assert!(!g.returns_twice);
}

/// `__attribute__((noinline))` on a prototype reaches the
/// definition's MIR, so the inliner leaves the call site alone.
#[test]
fn noinline_prototype_blocks_inlining() {
    let src = "int g(int x) __attribute__((noinline));\n\
               int g(int x) { return x + 1; }\n\
               int main() { return g(41); }";
    let mut module = lower_module(src);
    let g = module.funcs.iter().find(|f| f.name == "g").expect("g lowered");
    assert!(g.noinline, "prototype attribute did not reach the definition");
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    let main_fn = module.funcs.iter().find(|f| f.name == "main").expect("main lowered");
    let calls = main_fn
        .blocks
        .iter()
        .flat_map(|b| b.insts.iter())
        .any(|inst| matches!(main_fn.insts[*inst].kind, mir::InstKind::Call { .. }));
    assert!(calls, "noinline call was inlined:\n{dump}");
    assert_eq!(jit_run_main(module), 42);
}

/// `x / 1` and `x % 1` fold away at -O2. The dividend loads a
/// mutable global, so ConstFold cannot reach them: only the
/// identity rules can.
#[test]
fn optimized_pipeline_folds_division_by_one() {
    let src = "int g = -7;\n\
               int q(int x) { return x / 1; }\n\
               int r(int x) { return x % 1; }\n\
               int main() { return q(g) + r(g); }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    let divides = module.funcs.iter().any(|f| {
        f.blocks.iter().flat_map(|b| b.insts.iter()).any(|inst| {
            matches!(
                f.insts[*inst].kind,
                mir::InstKind::Udiv(..) | mir::InstKind::Sdiv(..) | mir::InstKind::Urem(..) | mir::InstKind::Srem(..)
            )
        })
    });
    assert!(!divides, "division by one survived:\n{dump}");
    assert_eq!(jit_run_main(module), -7);
}

/// Signed division and remainder by a positive power of two become
/// shifts and masks. Check the rounding cases, including `INT_MIN`.
#[test]
fn optimized_pipeline_rewrites_signed_power_of_two_division() {
    let src = "int q(int x) { return x / 4; }\n\
               int r(int x) { return x % 4; }\n\
               int main() {\n\
                   if (q(-7) != -1 || r(-7) != -3) return 1;\n\
                   if (q(7) != 1 || r(7) != 3) return 2;\n\
                   if (q(-2147483647 - 1) != -536870912) return 3;\n\
                   if (r(-2147483647 - 1) != 0) return 4;\n\
                   return 0;\n\
               }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    let divides = module.funcs.iter().any(|f| {
        f.blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .any(|inst| matches!(f.insts[*inst].kind, mir::InstKind::Sdiv(..) | mir::InstKind::Srem(..)))
    });
    assert!(!divides, "signed power-of-two division survived:\n{dump}");
    assert_eq!(jit_run_main(module), 0);
}

/// Read-only aggregate initializers become constants at the byte
/// offset the load reaches, including through the `sext`/`mul`/`add`
/// address chain array indexing produces.
#[test]
fn optimized_pipeline_folds_read_only_aggregate_loads() {
    let src = "static const int a[3] = {1, 2, 3};\n\
               static const struct P { int x; short y; char z; } p = {4, 5, 6};\n\
               static const double d[2] = {2.5, 4.0};\n\
               __attribute__((noinline)) int fa(void) { return a[1]; }\n\
               __attribute__((noinline)) int fp(void) { return p.x + p.y + p.z; }\n\
               __attribute__((noinline)) int fd(void) { return (int)d[0] + (int)d[1]; }\n\
               int main() { return fa() + fp() + fd(); }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    for name in ["fa", "fp", "fd"] {
        let func = module.funcs.iter().find(|f| f.name == name).expect("function lowered");
        let loads = func
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .any(|inst| matches!(func.insts[*inst].kind, mir::InstKind::Load { .. }));
        assert!(!loads, "{name} still loads read-only storage:\n{dump}");
    }
    assert_eq!(jit_run_main(module), 23);
    assert_eq!(run_main("int main() { return \"abc\"[1]; }"), 98);
}

/// Copying an aggregate and changing one field should keep the
/// source as the base instead of extracting and rebuilding fields
/// that stay the same.
#[test]
fn optimized_pipeline_updates_one_aggregate_field() {
    let src = "struct P { int x; int y; };\n\
               __attribute__((noinline)) struct P inc(struct P p) { struct P q = p; q.y++; return q; }\n\
               int main() { struct P p = {3, 4}; struct P q = inc(p); return q.x * 10 + q.y; }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    let func = module.funcs.iter().find(|f| f.name == "inc").expect("function lowered");
    let rebuilds_first = func
        .blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .any(|inst| matches!(func.insts[*inst].kind, mir::InstKind::ExtractValue { index: 0, .. }));
    assert!(!rebuilds_first, "inc still extracts the unchanged field:\n{dump}");
    assert_eq!(jit_run_main(module), 35);
}

/// Overwriting several fields of a copied aggregate should still
/// use the source for fields that the function leaves alone.
#[test]
fn optimized_pipeline_updates_several_aggregate_fields() {
    let src = "struct P { int x; int y; int z; };\n\
               __attribute__((noinline)) struct P bump(struct P p) { struct P q = p; q.x++; q.y += 2; return q; }\n\
               int main() { struct P p = {3, 4, 5}; struct P q = bump(p); return q.x * 100 + q.y * 10 + q.z; }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    let func = module.funcs.iter().find(|f| f.name == "bump").expect("function lowered");
    let rebuilds_unchanged = func
        .blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .any(|inst| matches!(func.insts[*inst].kind, mir::InstKind::ExtractValue { index: 2, .. }));
    assert!(!rebuilds_unchanged, "bump still extracts the unchanged field:\n{dump}");
    assert_eq!(jit_run_main(module), 465);
}

/// Literal floating-point expressions fold before codegen, so
/// subsequent conversions see a single constant.
#[test]
fn optimized_pipeline_folds_constant_float_arithmetic() {
    let src = "double f(void) { return 1.5 + 2.25 * 4.0; }\n\
               float g(void) { return 1.5f / 2.0f; }\n\
               int main(void) { return (int)f() + (int)g(); }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    for name in ["f", "g"] {
        let func = module.funcs.iter().find(|f| f.name == name).expect("function lowered");
        let arithmetic = func.blocks.iter().flat_map(|block| block.insts.iter()).any(|inst| {
            matches!(
                func.insts[*inst].kind,
                mir::InstKind::FAdd(..) | mir::InstKind::FSub(..) | mir::InstKind::FMul(..) | mir::InstKind::FDiv(..)
            )
        });
        assert!(!arithmetic, "{name} kept literal float arithmetic:\n{dump}");
    }
    assert_eq!(jit_run_main(module), 10);
}

/// Constant float-to-integer conversions are exact within the
/// target range, so an aggregate field assigned a literal does not
/// need a runtime conversion when it is read back as an integer.
#[test]
fn optimized_pipeline_folds_constant_float_conversions() {
    let src = "struct P { int x; double y; };\n\
               __attribute__((noinline)) int f(int x) { struct P p = {0}; p.x = x; p.y = 2.0; return p.x + (int)p.y; }\n\
               int main() { return f(5); }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    let func = module.funcs.iter().find(|f| f.name == "f").expect("function lowered");
    let converted = func.blocks.iter().flat_map(|block| block.insts.iter()).any(|inst| {
        matches!(
            func.insts[*inst].kind,
            mir::InstKind::SFtoI(..) | mir::InstKind::UFtoI(..) | mir::InstKind::Fptrunc(..) | mir::InstKind::Fpext(..)
        )
    });
    assert!(!converted, "literal conversion stayed in f:\n{dump}");
    assert_eq!(jit_run_main(module), 7);
}

/// Returning an aggregate parameter lowers to extracting every
/// field and rebuilding the aggregate. Filling all fields from one
/// source is the source itself, so the identity survives.
#[test]
fn optimized_pipeline_returns_aggregate_identity_directly() {
    let src = "struct P { int x; double y; };\n\
               __attribute__((noinline)) struct P id(struct P p) { return p; }\n\
               int main() { struct P p = {3, 4.0}; struct P q = id(p); return q.x + (int)q.y; }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    let func = module.funcs.iter().find(|f| f.name == "id").expect("function lowered");
    assert!(
        !func
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .any(|inst| { matches!(func.insts[*inst].kind, mir::InstKind::ExtractValue { .. } | mir::InstKind::InsertValue { .. }) }),
        "id still rebuilds its parameter:\n{dump}"
    );
    assert_eq!(jit_run_main(module), 7);
}

/// A local aggregate initialized by copying a read-only global is a
/// bundle of constants. Splitting the copy exposes them to `Sroa`,
/// including when the local is later modified.
#[test]
fn optimized_pipeline_folds_read_only_global_copies() {
    let src = "struct P { int x; short y; char z; };\n\
               static const struct P p = {7, 8, 9};\n\
               __attribute__((noinline)) int px(void) { struct P q = p; return q.x + q.y + q.z; }\n\
               __attribute__((noinline)) int py(void) { struct P q = p; q.x++; return q.x + q.y + q.z; }\n\
               int main() { return px() + py(); }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    for name in ["px", "py"] {
        let func = module.funcs.iter().find(|f| f.name == name).expect("function lowered");
        let hidden = func
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .any(|inst| matches!(func.insts[*inst].kind, mir::InstKind::Memcpy { .. } | mir::InstKind::Load { .. }));
        assert!(!hidden, "{name} still copies or loads the global initializer:\n{dump}");
    }
    assert_eq!(jit_run_main(module), 49);
}

/// Lowering materializes one constant per literal occurrence, so
/// `const-merge` must unify them before `cse` can recognize
/// repeated expressions — in one block (`local`) and across a
/// dominating branch (`arms`).
#[test]
fn optimized_pipeline_unifies_constants_before_cse() {
    let src = "int local(int x, int y) {\n\
                   int a = x * 13 + y * 5;\n\
                   int b = x * 13 + y * 5;\n\
                   return a + b;\n\
               }\n\
               int arms(int x, int y, int c) {\n\
                   int common = (x + 5) * (y - 2);\n\
                   if (c)\n\
                       return common + (x + 5) * (y - 2);\n\
                   return common - (x + 5) * (y - 2);\n\
               }\n\
               int main() { return local(2, 3) + arms(2, 3, 1); }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    fn muls(module: &mir::Module, name: &str) -> usize {
        let func = module.funcs.iter().find(|f| f.name == name).expect("function is lowered");
        func.blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .filter(|inst| matches!(&func.insts[**inst].kind, mir::InstKind::Mul(..)))
            .count()
    }
    assert_eq!(muls(&module, "local"), 2, "one mul per distinct product:\n{dump}");
    assert_eq!(muls(&module, "arms"), 1, "both arms reuse the common product:\n{dump}");
    assert_eq!(jit_run_main(module), 96);
}

/// Work only the taken arm needs sinks into that arm instead of
/// running on both paths.
#[test]
fn optimized_pipeline_sinks_branch_only_work() {
    let src = "int sink(int x, int take) {\n\
                   int product = (x + 11) * (x - 3);\n\
                   int mix = x * 7 + 9;\n\
                   if (take)\n\
                       return product + mix;\n\
                   return 123;\n\
               }\n\
               int main() { return sink(4, 0); }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    let sink = module.funcs.iter().find(|f| f.name == "sink").expect("sink is lowered");
    assert!(sink.blocks[sink.entry].insts.len() <= 1, "the entry keeps only the branch test:\n{dump}");
    assert_eq!(jit_run_main(module), 123);
}

/// A small loop with a constant trip count is fully unrolled, so the
/// header check and back edge disappear and the result folds to a
/// constant.
#[test]
fn optimized_pipeline_unrolls_small_counted_loops() {
    let src = "int main(void) { int sum = 100; for (int i = 0; i < 2; i++) { sum += 1; } return sum; }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    let main = module.funcs.iter().find(|f| f.name == "main").expect("main lowered");
    let live = main.reachable_blocks().into_iter().filter(|live| *live).count();
    assert_eq!(live, 1, "the loop survived:\n{dump}");
    assert_eq!(jit_run_main(module), 102);
}

/// A loop bound that is not constant keeps its back edge: the
/// unroller must not assume a trip count it cannot see.
#[test]
fn optimized_pipeline_keeps_dynamic_loop_bounds() {
    let src = "int sum_to(int n) { int sum = 0; for (int i = 0; i < n; i++) sum += i; return sum; }\n\
               int main() { return sum_to(5); }";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let sum_to = module.funcs.iter().find(|f| f.name == "sum_to").expect("sum_to lowered");
    assert!(
        sum_to.reachable_blocks().into_iter().filter(|live| *live).count() > 1,
        "the dynamic loop was removed:\n{}",
        mir::display::dump_module(&module)
    );
    assert_eq!(jit_run_main(module), 10);
}

/// An over-aligned local gets a slot whose address satisfies the
/// requested alignment; the raw stack slot only guarantees the
/// type's natural alignment.
#[test]
fn over_aligned_local_is_aligned() {
    let src = "int main(void){ char pad[3]; _Alignas(64) int x; return (int)((unsigned long)&x % 64); }";
    assert_eq!(jit_run_main(lower_module(src)), 0);
}

/// C text all the way to MIR for the host target (the JIT below
/// executes in-process, so target and host must agree).
fn lower_module(src: &str) -> mir::Module {
    lower_module_for(src, fe::target::Target::native())
}

/// `lower_module` for an explicit target.
fn lower_module_for(src: &str, target: fe::target::Target) -> mir::Module {
    let mut interner = fe::lexer::Interner::new();
    let mut sema = fe::sema::Sema::leak_for_tests();
    sema.target = target;
    let tokens = fe::lexer::Lexer::new(src, &mut interner).lex_all();
    let mut parser = fe::parse::Parser::new(src, tokens, sema.arena, &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    lower_translation_unit(&sema, &interner, &tu)
}

/// `lower_module` for C2y sources.
fn lower_module_c2y(src: &str) -> mir::Module {
    let mut interner = fe::lexer::Interner::new();
    let mut sema = fe::sema::Sema::leak_for_tests();
    sema.lang.std = fe::lang::Std::C2y;
    let tokens = fe::lexer::Lexer::new(src, &mut interner).lex_all();
    let mut parser = fe::parse::Parser::new(src, tokens, sema.arena, &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    lower_translation_unit(&sema, &interner, &tu)
}

/// `lower_module` for C23 sources.
fn lower_module_c23(src: &str) -> mir::Module {
    let mut interner = fe::lexer::Interner::new();
    let mut sema = fe::sema::Sema::leak_for_tests();
    sema.lang.std = fe::lang::Std::C23;
    let tokens = fe::lexer::Lexer::new(src, &mut interner).lex_all();
    let mut parser = fe::parse::Parser::new(src, tokens, sema.arena, &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    lower_translation_unit(&sema, &interner, &tu)
}

/// `lower_module` with MS extensions on, for `__ptr32`/`__ptr64`.
fn lower_module_ms(src: &str) -> mir::Module {
    let mut interner = fe::lexer::Interner::new();
    let mut sema = fe::sema::Sema::leak_for_tests();
    sema.lang = fe::lang::LangOpts { ms: true, ..Default::default() };
    let tokens = fe::lexer::Lexer::new(src, &mut interner).lex_all();
    let mut parser = fe::parse::Parser::new(src, tokens, sema.arena, &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    lower_translation_unit(&sema, &interner, &tu)
}

/// MSVC's headers define every CRT wrapper as a non-static
/// `__inline` and rely on COMDAT folding or the identical CRT
/// export. On the Windows target the body is dropped at
/// declaration; a referenced wrapper is revived as a TU-local
/// copy (names like `sprintf` have no export to resolve to).
#[test]
fn windows_msvc_extern_inline_definitions_become_imports() {
    let module = lower_module_for(
        "__inline int wrap(void) { return 7; } int main(void) { return wrap(); }",
        fe::target::Target::WINDOWS_X86_64,
    );
    let wrap = module.funcs.iter().find(|f| f.name == "wrap").expect("referenced wrapper revived");
    assert!(wrap.local, "the revived copy is TU-local");
    assert!(module.externs.iter().any(|e| e.name == "wrap"));
    assert!(module.funcs.iter().any(|f| f.name == "main"));

    // An unreferenced wrapper stays dropped: declaration-only.
    let module = lower_module_for("__inline int unused(void) { return 7; } int main(void) { return 0; }", fe::target::Target::WINDOWS_X86_64);
    assert!(!module.funcs.iter().any(|f| f.name == "unused"));
    assert!(module.externs.iter().any(|e| e.name == "unused"));

    // Static `__inline` bodies stay TU-local definitions.
    let module = lower_module_for(
        "static __inline int local(void) { return 7; } int main(void) { return local(); }",
        fe::target::Target::WINDOWS_X86_64,
    );
    assert!(module.funcs.iter().any(|f| f.name == "local"));

    // Other targets keep the plain C behavior: the definition
    // emits (glcc has no COMDAT, but ELF libc exports no twin).
    let module = lower_module_for("__inline int wrap(void) { return 7; } int main(void) { return wrap(); }", fe::target::Target::LP64);
    assert!(module.funcs.iter().any(|f| f.name == "wrap"));
}

/// `(MIR module, assembler text)` for `src`.
fn lower_with_asm(src: &str) -> (mir::Module, String) {
    let mut interner = fe::lexer::Interner::new();
    let mut sema = fe::sema::Sema::leak_for_tests();
    let tokens = fe::lexer::Lexer::new(src, &mut interner).lex_all();
    let mut parser = fe::parse::Parser::new(src, tokens, sema.arena, &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    let module = lower_translation_unit(&sema, &interner, &tu);
    (module, naked_asm_text(&sema, &interner, &tu))
}

#[test]
fn asm_labels_rename_emitted_symbols() {
    // A GNU asm label names the symbol while the C spelling stays
    // the declaration key: EDG emits `_Z6strchrPci(...)
    // __asm__("strchr")` for the mangled C library overloads.
    let module = lower_module(
        "extern int renamed(int) __asm__(\"target\");\n\
         int defined(void) __asm__(\"elsewhere\") { return 1; }\n\
         int caller(void) { return defined() + renamed(1); }",
    );
    assert!(!module.funcs.iter().any(|f| f.name == "defined"));
    assert!(module.funcs.iter().any(|f| f.name == "elsewhere"));
    assert!(!module.externs.iter().any(|e| e.name == "renamed"));
    let (target_id, _) = module.externs.iter_enumerated().find(|(_, e)| e.name == "target").expect("import carries the label");
    let caller = module.funcs.iter().find(|f| f.name == "caller").unwrap();
    assert!(
        caller
            .insts
            .iter()
            .any(|i| matches!(&i.kind, mir::InstKind::CallExtern { target, .. } if *target == target_id)),
        "the call resolves to the labeled import"
    );
    assert!(
        caller.insts.iter().any(|i| matches!(
            &i.kind,
            mir::InstKind::Call { callee, .. } if module.funcs[*callee].name == "elsewhere"
        )),
        "the call resolves to the labeled definition"
    );
}

#[test]
fn asm_labels_survive_to_the_jit() {
    // The C name keeps calling the labeled definition, and an
    // import still resolves through its label (libc's `rand`).
    assert_eq!(run_main("int f(void) __asm__(\"g\") { return 40; } int main(void) { return f() + 2; }"), 42);
    assert_eq!(run_main("int roll(void) __asm__(\"rand\"); int main(void) { return roll() >= 0; }"), 1);
}

#[test]
fn naked_functions_emit_asm_text_and_import() {
    let (module, text) = lower_with_asm("__attribute__((naked)) int f(void) { __asm__(\"mov $42, %eax\\n\\tret\"); }\nint main(void){ return f(); }");
    assert!(text.contains(".globl f\n"), "{text}");
    assert!(text.contains("f:\n"), "{text}");
    assert!(text.contains("mov $42, %eax"), "{text}");
    // The C side calls an import; the body is not lowered.
    assert!(!module.funcs.iter().any(|f| f.name == "f"));
    assert!(module.externs.iter().any(|e| e.name == "f"));
}

#[test]
fn section_attributes_reach_globals_and_naked_text() {
    let (module, text) = lower_with_asm(
        "__attribute__((section(\".vectors\"))) const unsigned long vec[2] = {1, 2};\
         __attribute__((section(\".noinit\"))) char scratch[8];\
         __attribute__((naked, section(\".mystart\"))) void boot(void) { __asm__(\"ret\"); }\
         int main(void){ return (int)vec[0] + scratch[0]; }",
    );
    assert_eq!(global_named(&module, "vec").section.as_deref(), Some(".vectors"));
    assert_eq!(global_named(&module, "scratch").section.as_deref(), Some(".noinit"));
    assert!(text.contains(".section .mystart,\"ax\"\n.globl boot\n"), "{text}");
    assert!(text.contains(".previous\n"), "{text}");
}

#[test]
fn bitint_values_wrap_at_declared_precision() {
    let src = "int main(void) {\n\
               unsigned _BitInt(8) a = 255uwb;\n\
               signed _BitInt(40) b = 0x7fffffffffLL;\n\
               signed _BitInt(33) c = 0x7fffffffffLL;\n\
               unsigned _BitInt(40) d = 0xffffffffffULL;\n\
               signed _BitInt(40) b1 = b + 1;\n\
               return (int)(a + 1uwb == 0)\n\
                   + 2 * (int)(b1 < 0)\n\
                   + 4 * (int)(c < 0)\n\
                   + 8 * (int)(d + 1 == 0)\n\
                   + 16 * (int)(sizeof(a) == 1)\n\
                   + 32 * (int)(_Alignof(d) == 8);\n\
           }";
    assert_eq!(jit_run_main(lower_module_c23(src)), 63);
}

#[test]
fn bitint_arbitrary_width_arithmetic() {
    let src = "int main(void) {\n\
               int ok = 0;\n\
               unsigned _BitInt(256) a = 1;\n\
               a = a << 200;\n\
               if ((a >> 200) == 1 && (a >> 199) == 2) ok |= 1;\n\
               unsigned _BitInt(256) c = a + 5;\n\
               if ((c >> 200) == 1 && (unsigned long long)c == 5) ok |= 2;\n\
               if (c - a == 5) ok |= 4;\n\
               unsigned _BitInt(256) e = a * 3;\n\
               if ((e >> 200) == 3 && e / a == 3 && c % a == 5) ok |= 8;\n\
               _BitInt(200) s = -5;\n\
               if (s * 3 == -15 && (s * 3) / 2 == -7 && (s * 3) % 4 == -3) ok |= 16;\n\
               unsigned _BitInt(4096) big = (unsigned _BitInt(4096))1 << 4095;\n\
               if (big > 0 && (big >> 4095) == 1) ok |= 32;\n\
               unsigned _BitInt(4096) q = ((unsigned _BitInt(4096))123456789 << 1000) / 123456789;\n\
               if (q == ((unsigned _BitInt(4096))1 << 1000)) ok |= 64;\n\
               return ok;\n\
           }";
    assert_eq!(jit_run_main(lower_module_c23(src)), 127);
}

#[test]
fn bitint_float_conversions() {
    let src = "int main(void) {\n\
               int ok = 0;\n\
               unsigned _BitInt(256) a = (unsigned _BitInt(256))1 << 200;\n\
               double d = (double)a;\n\
               if (d > 1.6e60 && d < 1.61e60) ok |= 1;\n\
               _BitInt(256) n = -(_BitInt(256))a;\n\
               if ((double)n < 0.0) ok |= 2;\n\
               unsigned _BitInt(256) b = (unsigned _BitInt(256))1.0e20;\n\
               if ((b >> 64) == 5 && (unsigned long long)b == 7766279631452241920ull) ok |= 4;\n\
               _BitInt(200) s = (_BitInt(200))-12.99;\n\
               if (s == -12) ok |= 8;\n\
               float f = 16777216.0f;\n\
               unsigned _BitInt(200) g = (unsigned _BitInt(200))f;\n\
               if ((g >> 24) == 1) ok |= 16;\n\
               return ok;\n\
           }";
    assert_eq!(jit_run_main(lower_module_c23(src)), 31);
}

#[test]
fn bitint_aggregate_storage() {
    let src = "static _BitInt(256) gn = -7;\n\
               struct S { unsigned _BitInt(256) a; _BitInt(300) b; char tag; };\n\
               int main(void) {\n\
                   int ok = 0;\n\
                   if (gn == -7) ok |= 1;\n\
                   struct S s;\n\
                   s.a = (unsigned _BitInt(256))5 << 200;\n\
                   s.b = -123456789012345678901234567890wb;\n\
                   s.tag = 9;\n\
                   if (s.tag == 9 && (s.a >> 200) == 5 && s.b < 0) ok |= 2;\n\
                   if ((s.b % 1000) == -890) ok |= 4;\n\
                   unsigned _BitInt(256) arr[3];\n\
                   for (int i = 0; i < 3; i++) arr[i] = (unsigned _BitInt(256))i << 128;\n\
                   if ((arr[2] >> 128) == 2) ok |= 8;\n\
                   struct S t = s;\n\
                   t.tag = 1;\n\
                   if (s.tag == 9 && t.tag == 1 && t.a == s.a && t.b == s.b) ok |= 16;\n\
                   return ok;\n\
               }";
    assert_eq!(jit_run_main(lower_module_c23(src)), 31);
}

#[test]
fn bitint_wide_division_synthesizes_a_helper() {
    let mut module = lower_module_c23("int main(void) { unsigned _BitInt(256) a = 7, b = 2; return (int)(a / b) + (int)(a % b); }");
    mir::optimize(&mut module, mir::OptLevel::None);
    assert!(module.funcs.iter().any(|f| f.name == "__glcc_wide_udiv$256"));
    assert!(module.funcs.iter().any(|f| f.name == "__glcc_wide_urem$256"));
    assert_eq!(jit_run_main(module), 4);
}

#[test]
fn nullptr_t_is_distinct_and_pointer_sized() {
    let src = "typedef typeof(nullptr) nullptr_t;\n\
               nullptr_t g = nullptr;\n\
               int main(void) {\n\
                   nullptr_t p = nullptr;\n\
                   int *q = p;\n\
                   p = 0;\n\
                   return (q == nullptr) + (p == nullptr) + (nullptr == 0)\n\
                       + !nullptr + (nullptr ? 0 : 1) + (g == nullptr)\n\
                       + (sizeof(nullptr_t) == sizeof(void *));\n\
               }";
    assert_eq!(jit_run_main(lower_module_c23(src)), 7);
}

/// C text all the way to a JIT call returning `i32`.
fn run_main(src: &str) -> i32 {
    jit_run_main(lower_module(src))
}

/// JIT `mir_module` and call its `main`.
fn jit_run_main(mut mir_module: mir::Module) -> i32 {
    unsafe {
        let errors = mir::verify::verify_module(&mir_module);
        assert!(errors.is_empty(), "verify errors: {errors:?}");
        // The driver runs the shared pipeline before lowering; the
        // JIT helper bypasses it, so legalize wide integers here.
        mir::optimize(&mut mir_module, mir::OptLevel::None);
        // The variadic prelude reads the frame pointer, like the
        // driver (which sets preserve_frame_pointers), and Win64
        // F128/I128 follow LLVM's ABI there.
        let mut flags = glcc_lower_cl::codegen_flags();
        if cfg!(windows) {
            flags.push(("enable_llvm_abi_extensions", "true"));
        }
        let builder = JITBuilder::with_flags(&flags, cranelift_module::default_libcall_names()).expect("jit builder");
        let mut jit = JITModule::new(builder);
        let ids = glcc_lower_cl::lower_module(&mir_module, &mut jit, glcc_lower_cl::Target::native());
        jit.finalize_definitions().expect("jit finalize");
        let main = *ids.iter().find(|(id, _)| mir_module.funcs[**id].name == "main").map(|(_, fid)| fid).expect("main function");
        let addr = jit.get_finalized_function(main);
        let func: extern "C" fn() -> i32 = std::mem::transmute(addr);
        func()
    }
}

#[test]
fn vla_objects() {
    // A runtime bound allocates, indexes, and round-trips values.
    assert_eq!(run_main("int main(){int n = 5; int a[n]; for (int i = 0; i < n; i++) a[i] = i * i; return a[4];}"), 16);
    // Rows of a multidimensional VLA stride by the runtime extent.
    assert_eq!(
        run_main("int main(){int n = 3, m = 4; int a[n][m]; int i, j; for (i = 0; i < n; i++) for (j = 0; j < m; j++) a[i][j] = i * m + j; return a[2][3];}"),
        11
    );
    // The declaration captures its extents: later writes to the
    // bound variable do not resize the object.
    assert_eq!(run_main("int main(){int n = 4; int a[n]; n = 99; return sizeof a;}"), 16);
    // `sizeof` on a type-name evaluates the bounds where it is used.
    assert_eq!(run_main("int main(){int n = 6; typedef int t[n]; return sizeof(t);}"), 24);
    // `_Countof` reads fixed and variable top-level extents; it does
    // not re-evaluate a captured VLA bound or the operand.
    assert_eq!(run_main("int main(){int a[7]; return _Countof(a);}"), 7);
    assert_eq!(run_main("int main(){int n = 5; int a[n]; return _Countof(a);}"), 5);
    assert_eq!(run_main("int main(){int n = 4; int a[3][n]; return _Countof(a) * 10 + _Countof(*a);}"), 34);
    assert_eq!(run_main("int main(){int n = 4; int a[n]; n = 99; return _Countof(a);}"), 4);
    assert_eq!(run_main("int main(){int a[3]; int n = 0; return _Countof(a) + n;}"), 3);
    assert_eq!(run_main("int main(){return _Countof(int[3][4]);}"), 3);
    // Through a pointer-to-VLA the bound re-evaluates: both the row
    // stride and `sizeof *a` read the parameter.
    assert_eq!(
        run_main("int bar(int n, char a[2][n]){a[1][n - 1] = 7; return sizeof *a;} int main(){char x[2][5]; return bar(5, x);}"),
        5
    );
    // Pointer arithmetic steps by the runtime extent: `p + 2 - p`,
    // `*p + 3 - *p`, `p - i`, and element addresses all scale with
    // the variably modified pointee (tcctest.c's `c99_vla_test_2`).
    assert_eq!(run_main("int main(){int m = 4; int a[3][4]; int (*p)[m] = a; return (p + 2) - p;}"), 2);
    assert_eq!(run_main("int main(){int m = 4; int a[3][4]; int (*p)[m] = a; return (*p + 3) - *p;}"), 3);
    assert_eq!(run_main("int main(){int m = 4; int a[3][4]; int (*p)[m] = a; p = p + 2; p = p - 1; return p - a;}"), 1);
    assert_eq!(run_main("int main(){int m = 4; int a[3][4]; int (*p)[m] = a; return &p[2][0] - &p[0][0];}"), 8);
}

#[test]
fn static_initializers_fold_address_conditions() {
    // Address constants fold as ternary conditions in initializers
    // (they are not integer constant expressions; GCC folds them
    // there — tcctest.c's `sinit23`).
    assert_eq!(
        run_main("int obj; int x[] = { \"astring\" ? sizeof(\"astring\") : -1, &obj ? 42 : -1 }; int main(){ return x[0] * 10 + x[1]; }"),
        122
    );
    assert_eq!(run_main("int obj; int x = !&obj ? 1 : 2; int main(){ return x; }"), 2);
    // Logical operators short-circuit an unevaluated operand:
    // `2 || 1 / 0` is the constant 1 (`sinit24`).
    assert_eq!(run_main("int x = 2 || 1 / 0; int main(){ return x; }"), 1);
    assert_eq!(run_main("int x = 0 && 1 / 0; int main(){ return x; }"), 0);
}

#[test]
#[should_panic(expected = "non-constant global initializer")]
fn weak_address_conditions_do_not_fold() {
    // A weak symbol may resolve to zero, so its address is no
    // initializer constant (GCC: "initializer element is not
    // constant"; tcctest.c's comment).
    run_main("extern int w __attribute__((weak)); int x = &w ? 1 : 2; int main(){ return x; }");
}

#[test]
fn braced_string_initializers_are_the_bare_shape() {
    // The braced lone-string form is the bare string with braces
    // (C17 §6.7.9p14): sized with the NUL and lowered the same way
    // (tcctest.c's `str_ag2`).
    assert_eq!(run_main("char s[] = { \"b\" }; int main(){ return sizeof(s) + s[0] - 'b'; }"), 2);
    assert_eq!(run_main("char s[4] = { \"abc\" }; int main(){ return s[2] == 'c'; }"), 1);
    // Locals and compound literals take the same shape.
    assert_eq!(run_main("int main(){ char s[] = { \"xy\" }; return sizeof(s) + s[1] - 'y'; }"), 3);
    assert_eq!(run_main("int main(){ char *s = (char[]){ \"ab\" }; return s[1] == 'b'; }"), 1);
    // A nested array still initializes its first row from the
    // string; the braces are not redundant there.
    assert_eq!(run_main("char m[2][3] = { \"ab\" }; int main(){ return m[0][1] == 'b' && m[1][0] == 0; }"), 1);
}

#[test]
fn full_initializers_skip_the_redundant_zero_fill() {
    let main_body = |src| {
        let dump = mir::display::dump_module(&lower_module(src));
        dump.split("func @main").nth(1).expect("main is lowered").to_string()
    };

    let full = main_body("int main(){ long long a[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}; return (int)a[15]; }");
    assert!(!full.contains("memset"), "{full}");

    let partial = main_body("int main(){ long long a[16] = {1}; return (int)a[15]; }");
    assert!(partial.contains("memset"), "{partial}");

    let designated = main_body("struct S{int a,b;}; int main(){ struct S s = {.a = 1}; return s.a; }");
    assert!(designated.contains("memset"), "{designated}");

    let full_string = main_body("int main(){ char s[4] = \"abc\"; return s[3]; }");
    assert!(!full_string.contains("memset"), "{full_string}");

    let partial_string = main_body("int main(){ char s[8] = \"abc\"; return s[7]; }");
    assert!(partial_string.contains("memset"), "{partial_string}");
}

#[test]
fn vla_frees_at_every_return() {
    // The declaration site frees its own previous block; each of the
    // two returns drains the live allocation. No drain loop: the
    // only branch is the source `if`.
    let module = lower_module("int f(int n){ if (n) { int a[n]; a[0] = 1; return a[0]; } return 2; } int main(){return f(1);}");
    let free = module.externs.iter().position(|e| e.name == "free").expect("free import");
    let dump = mir::display::dump_module(&module);
    let f = dump.split("func @main").next().unwrap();
    assert_eq!(f.matches(&format!("call.ext e{free}(")).count(), 3, "{f}");
    assert_eq!(f.matches("brif").count(), 1, "{f}");
}

#[test]
fn vla_loop_reuses_its_storage() {
    // Each iteration frees the previous block before allocating, so
    // `malloc` hands back the same storage.
    assert_eq!(
        run_main("int main(){int i; char *first = 0; for (i = 0; i < 4; i++){ char a[4096]; if (i == 0) first = a; else if (a != first) return 1; } return 0;}"),
        0
    );
}

#[test]
fn arithmetic_return() {
    assert_eq!(run_main("int main(){return 2+3*4;}"), 14);
    assert_eq!(run_main("int main(){return (10-3)*2;}"), 14);
    assert_eq!(run_main("int main(){return -5+12;}"), 7);
}

#[test]
fn switch_dispatch() {
    let body = "int f(int x){switch(x){case 1: return 10; case 2: case 3: return 20; default: return 30;}} int main(){return f(1)*1000+f(2)*100+f(3)+f(9);}";
    // 10, 20, 20, 30.
    assert_eq!(run_main(body), 10 * 1000 + 20 * 100 + 20 + 30);
    // Fallthrough accumulates without `break`.
    assert_eq!(
        run_main("int main(){int s = 0; switch(2){case 1: s += 1; case 2: s += 10; case 3: s += 100; default: s += 1000;} return s;}"),
        1110
    );
    // `break` leaves the switch, not the function.
    assert_eq!(run_main("int main(){int s = 0; switch(1){case 1: s = 7; break; default: s = 9;} return s;}"), 7);
    // Empty body and bare default still dispatch.
    assert_eq!(run_main("int main(){switch(5){} return 1;}"), 1);
    assert_eq!(run_main("int main(){switch(5){default: return 4;}}"), 4);
}

#[test]
fn switch_promotion() {
    // Narrow controlling expressions compare post-promotion: a
    // signed `char` -1 must not match `case 255` (spelled out
    // because plain `char` is unsigned on aarch64).
    assert_eq!(
        run_main("int main(){signed char c = -1; switch(c){case 255: return 1; case -1: return 2; default: return 3;}}"),
        2
    );
    // Plain `char` follows the target ABI: signed on x86-64 (so -1
    // matches `case -1`), unsigned on aarch64 (0xFF matches
    // `case 255`).
    let plain_char = if fe::target::Target::native().char_signed { 2 } else { 1 };
    assert_eq!(
        run_main("int main(){char c = -1; switch(c){case 255: return 1; case -1: return 2; default: return 3;}}"),
        plain_char
    );
    assert_eq!(
        run_main("int main(){unsigned char c = 255; switch(c){case 255: return 1; case -1: return 2; default: return 3;}}"),
        1
    );
    assert_eq!(run_main("int main(){short s = 4464; switch(s){case 4464: return 1; default: return 2;}}"), 1);
}

#[test]
fn switch_nesting_and_loops() {
    // `break` in a loop in a switch exits the loop; `break` in a
    // switch in a loop exits the switch; `continue` still finds
    // the loop across the switch.
    assert_eq!(
        run_main("int main(){int s = 0; switch(1){case 1: for(int i = 0; i < 10; i++){if(i == 3) break; s += 1;} s += 100;} return s;}"),
        103
    );
    assert_eq!(
        run_main("int main(){int s = 0; for(int i = 0; i < 5; i++){switch(i){case 2: break; default: s += 1;}} return s;}"),
        4
    );
    assert_eq!(
        run_main("int main(){int s = 0; for(int i = 0; i < 5; i++){switch(i){case 1: case 3: continue; default: s += 1;}} return s;}"),
        3
    );
    // Nested switches dispatch independently.
    assert_eq!(
        run_main("int main(){int s = 0; switch(1){case 1: switch(2){case 2: s = 12; break; default: s = 10;} break; default: s = 1;} return s;}"),
        12
    );
}

#[test]
fn switch_duff() {
    // Cases may live inside loops in the switch body.
    assert_eq!(
        run_main("int main(){int n = 0; int i = 0; switch(2){case 0: i++; case 2: i += 10; for(; i < 12; i++){n++; if(i == 11) break;} } return n * 100 + i;}"),
        211
    );
}

#[test]
fn switch_wide() {
    // 128-bit controlling expressions compare full values through
    // a conditional chain rather than the 64-bit switch.
    assert_eq!(
        run_main("int main(){__int128 x = ((__int128)1 << 100) + 5; switch(x){case 5: return 1; default: return 2;}}"),
        2
    );
    assert_eq!(
        run_main("int main(){__int128 x = ((__int128)1 << 100) + 5; switch(x){case 5: return 1; case (((__int128)1 << 100) + 5): return 3; default: return 2;}}"),
        3
    );
}

#[test]
fn goto_labels() {
    // Forward and backward jumps, including over declarations.
    assert_eq!(run_main("int main(){goto mid; int x = 1; mid: return 42;}"), 42);
    assert_eq!(run_main("int main(){int i = 0; int s = 0; again: i++; s += i; if(i < 10) goto again; return s;}"), 55);
    // Jumping out of a loop, and into a switch body.
    assert_eq!(run_main("int main(){for(int i = 0; i < 100; i++){if(i == 4) goto done;} done: return 7;}"), 7);
    assert_eq!(run_main("int main(){goto inner; switch(1){case 1: return 1; inner: return 2; default: return 3;}}"), 2);
}

#[test]
fn label_before_declaration_initializes_on_the_path() {
    assert_eq!(
        jit_run_main(lower_module_c23(
            "int main(){int n = 0; goto L; n = 100; L: int x = 41; n += x;\
             switch (n) { case 41: int y = 1; n += y; break; default: break; } return n;}"
        )),
        42
    );
    // A label at the end of a compound statement still lands there.
    assert_eq!(jit_run_main(lower_module_c23("int main(){goto E; E:}")), 0);
}

#[test]
fn c23_u8_literals_lower() {
    assert_eq!(jit_run_main(lower_module_c23("int main(){return u8'A';}")), 65);
    assert_eq!(jit_run_main(lower_module_c23("int main(){return sizeof(u8\"abc\");}")), 4);
}

#[test]
fn c23_compound_literal_storage_lower() {
    let src = "int *automatic(void) { return &(int){ 7 }; }\n\
               int *persistent(void) { return &(static int){ 7 }; }\n\
               int main(void) {\n\
                   int *p = automatic(); *p = 9; int a = *automatic();\n\
                   int *q = persistent(); *q = 9; int b = *persistent();\n\
                   return a * 10 + b;\n\
               }";
    assert_eq!(jit_run_main(lower_module_c23(src)), 79);
}

#[test]
fn c23_constexpr_values_lower() {
    let src = "constexpr int n = 3;\n\
               int a[n];\n\
               int main(void) { constexpr int m = 4; int c[m]; \
               return (sizeof a / sizeof a[0]) + (sizeof c / sizeof c[0]); }";
    assert_eq!(jit_run_main(lower_module_c23(src)), 7);
}

#[test]
fn c23_auto_deduced_objects_lower() {
    let src = "int main(void) { auto x = 2.5; auto p = &x; auto s = \"hi\"; auto const z = 7; \\\n                   return (sizeof x == 8 && sizeof p == 8 && s[0] == 'h' && *p == 2.5 && z == 7) ? 0 : 1; }";
    assert_eq!(jit_run_main(lower_module_c23(src)), 0);
}

#[test]
fn c23_fixed_enum_underlying_type_lower() {
    let src = "enum A : unsigned char { A1 = 255 };\n\
               enum B : long long { B1 = 0x100000000LL };\n\
               int main(void) { enum A a = A1; enum B b = B1; \
               return (sizeof a == 1 && sizeof b == 8 && a == 255 && b == 0x100000000LL) ? 0 : 1; }";
    assert_eq!(jit_run_main(lower_module_c23(src)), 0);
}

#[test]
fn computed_goto() {
    // A label address held in a local, dispatched on.
    assert_eq!(run_main("int main(){void *p = &&l; goto *p; l: return 5;}"), 5);
    // Jumping between labels through a table, with arithmetic on
    // the addresses (`&&b - &&a` counts labels).
    assert_eq!(
        run_main(
            "int main(){static void *t[] = {&&a, &&b, &&c}; int i = 0, s = 0; goto *t[i++]; \
             a: s += 10; goto *t[i++]; b: s += 100; goto *t[i++]; c: return s;}"
        ),
        110
    );
    // Label ids also serve static integer initializers, and label
    // differences stay consecutive.
    assert_eq!(
        run_main(
            "int main(){static int d[] = {&&b - &&a, &&c - &&a}; void *p = &&a + d[1]; \
             goto *p; a: return 1; b: return 2; c: return 3;}"
        ),
        3
    );
}

#[test]
fn computed_goto_in_statement_expression_scope() {
    // A label defined beside a statement expression's `goto` stays
    // local to it: the jump resolves inside.
    assert_eq!(run_main("int main(){int x = 0; ({ l: x++; if (x < 3) goto l; }); return x;}"), 3);
}

#[test]
fn local_labels_shadow_function_labels() {
    // GNU `__label__` labels are block scoped: the inner LBL loops
    // once, then the `goto` outside the inner block targets the
    // function-scoped label of the same spelling
    // (tcctest.c's `statement_expr_test`).
    assert_eq!(
        run_main("int main(){ int i = 0; ({ { __label__ LBL; LBL: if (i++ == 0) goto LBL; } goto LBL; }); LBL: return i; }"),
        2
    );
    // A statement-expression label set that cycles through itself
    // and only ends at the block's fallthrough (tcctest.c's
    // `local_label_test`): colliding with the enclosing labels made
    // this loop forever.
    assert_eq!(
        run_main(
            "int main(){ int a; goto l1; \
             l2: a = 1 + ({ __label__ l1, l2, l3, l4; goto l1; \
             l4: goto l3; l2: goto l4; l1: goto l2; l3:; 1; }); \
             return a; \
             l4: goto l2; \
             l1: goto l4; }"
        ),
        2
    );
}

#[test]
fn statement_expression_value_through_label() {
    // A trailing label does not hide the value (GCC: the value of
    // the last statement; tcctest.c relies on this for
    // `({ __label__ here; here: (unsigned long)&&here; })`), and
    // the label still defines its jump target for computed gotos.
    assert_eq!(run_main("int main(){int x = ({ l: 42; }); return x;}"), 42);
    assert_eq!(run_main("int main(){int x = ({ a: b: 43; }); return x;}"), 43);
    assert_eq!(
        run_main("int main(){int s = 0; int v = ({ void *p; l: p = &&l, s < 2 ? ({ s++; goto *p; 0; }) : 66; }); return v + s;}"),
        68
    );
}

#[test]
fn member_access_on_aggregate_rvalues() {
    assert_eq!(
        run_main("struct S{int x;int y[2];}; struct S f(void){struct S s; s.x=3; s.y[1]=4; return s;} int main(){return f().x + f().y[1];}"),
        7
    );
    assert_eq!(
        run_main("struct S{int x;}; int main(){struct S a,b; a.x=1; b.x=2; return (1 ? a : b).x + ({ struct S t; t.x=4; t; }).x;}"),
        5
    );
}

#[test]
fn alloca_memory_is_usable() {
    assert_eq!(run_main("int main(){char *p = __builtin_alloca(4); p[0] = 3; return p[0];}"), 3);
    assert_eq!(run_main("int main(){int *p = (int *)alloca(sizeof(int) * 4); p[3] = 7; return p[3];}"), 7);
    // malloc's 16-byte alignment survives the header.
    assert_eq!(run_main("int main(){return ((unsigned long)__builtin_alloca(1) % 16) == 0 ? 5 : 6;}"), 5);
}

#[test]
fn alloca_frees_at_every_return() {
    // Every return routes through its own drain loop, so an
    // allocation from an earlier loop iteration is freed even when
    // the return precedes the `alloca` in source order.
    let module = lower_module("int f(int n){ if (n) { char *p = __builtin_alloca(8); p[0] = 1; return p[0]; } return 2; } int main(){return f(1);}");
    let frees = module.externs.iter().filter(|e| e.name == "free").count();
    assert_eq!(frees, 1, "one shared `free` import");
    let free = module.externs.iter().position(|e| e.name == "free").expect("free import");
    let dump = mir::display::dump_module(&module);
    let f = dump.split("func @main").next().unwrap();
    assert_eq!(f.matches(&format!("call.ext e{free}(")).count(), 2, "{f}");
    // Two drain checks on top of the source `if`.
    assert_eq!(f.matches("brif").count(), 3, "{f}");
}

#[test]
fn overflow_builtins() {
    // The wrapped result stores through the pointer; the return
    // value says whether the exact value fit.
    assert_eq!(
        run_main("int main(){int r; return __builtin_add_overflow(2147483647, 1, &r) + (r == (-2147483647 - 1)) * 2;}"),
        3
    );
    assert_eq!(
        run_main("int main(){unsigned char r; return __builtin_sub_overflow((unsigned char)3, (unsigned char)5, &r) + (r == 254) * 2;}"),
        3
    );
    // `u64 * u64` overflows i128: the exact u128 product compares
    // unsigned against the result type's maximum.
    assert_eq!(
        run_main("int main(){unsigned long r; return __builtin_mul_overflow(0x100000000ul, 0x100000000ul, &r) + (r == 0) * 2;}"),
        3
    );
    assert_eq!(
        run_main("int main(){long long r; return __builtin_mul_overflow(0x100000000ull, 0x80000000ull, &r) + (r == (-9223372036854775807LL - 1)) * 2;}"),
        3
    );
    assert_eq!(
        run_main("int main(){long long r; return __builtin_mul_overflow(0x100000000ull, 0x7FFFFFFFull, &r) + (r == 0x7FFFFFFF00000000LL) * 2;}"),
        2
    );
    // Signed operands multiply exactly in i128.
    assert_eq!(
        run_main("int main(){long long r; return __builtin_mul_overflow(-4611686018427387904LL, 4LL, &r) + (r == 0) * 2;}"),
        3
    );
}

#[test]
fn frame_address_builtins_walk_the_frame_chain() {
    // Level 0 is the function's own frame, so the caller's locals
    // sit above it and level 1 (the caller's frame) above them.
    // Both levels of return address are live and distinct.
    assert_eq!(
        run_main(
            "int probe(int *a, int *b) {\
                 char *fp0 = (char *) __builtin_frame_address(0);\
                 char *fp1 = (char *) __builtin_frame_address(1);\
                 if ((char *) a <= fp0) return 1;\
                 if ((char *) b <= fp0) return 2;\
                 if ((char *) a >= fp1) return 3;\
                 if ((char *) b >= fp1) return 4;\
                 if (!__builtin_return_address(0)) return 5;\
                 if (__builtin_return_address(1) == __builtin_return_address(0)) return 6;\
                 return 0;\
             }\
             int main() { int a = 1; int b = 2; return probe(&a, &b); }"
        ),
        0
    );
}

#[test]
fn sizeof_string_literal_lowers() {
    // The literal's type carries its bound (NUL included), so
    // `sizeof` lowers even where it does not fold.
    assert_eq!(run_main("int main(){return (int) sizeof \"ab\" + (int) sizeof \"a\\n\";}"), 6);
}

#[test]
fn packed_and_over_aligned_structs_lower() {
    // `packed` members drop padding (so `d` sits at offset 2 and
    // the loads are unaligned) while an `aligned` struct keeps its
    // padded size and stride.
    assert_eq!(
        run_main(
            "struct P { short s __attribute__((packed)); double d __attribute__((packed)); };\
             int main(){ struct P p; p.s = 3; p.d = 4.5; return p.s + (int) p.d + (sizeof(struct P) == 10); }"
        ),
        8
    );
    assert_eq!(
        run_main(
            "struct X { int a; int b; } __attribute__((aligned(32)));\
             struct Y { struct X x[2]; int c; };\
             struct Y y[2];\
             int main(){ if (((unsigned long) &y[0]) % 32) return 1; return (int) ((char *) &y[1] - (char *) &y[0]); }"
        ),
        96
    );
}

#[test]
fn alias_attribute_shares_storage() {
    // The alias names the target's storage: writes through either
    // spelling land in the same object, and no symbol is emitted
    // for the alias.
    assert_eq!(
        run_main(
            "int a[4]; extern int b[4] __attribute__((alias(\"a\")));\
             int main(){ b[2] = 1; a[2] = 2; return b[2]; }"
        ),
        2
    );
    // Function aliases resolve to the target's definition.
    assert_eq!(
        run_main(
            "int g(void) { return 7; } extern int f(void) __attribute__((alias(\"g\")));\
             int main(){ return f(); }"
        ),
        7
    );
}

#[test]
fn bit_and_sign_builtins() {
    // `clrsb` counts the redundant sign bits; zero counts `w - 1`.
    assert_eq!(run_main("int main(){ return __builtin_clrsb(0); }"), 31);
    assert_eq!(run_main("int main(){ return __builtin_clrsb(1); }"), 30);
    assert_eq!(run_main("int main(){ return __builtin_clrsb(0x0F); }"), 27);
    assert_eq!(run_main("int main(){ return __builtin_clrsb(-1); }"), 31);
    assert_eq!(run_main("int main(){ return __builtin_clrsbll(1LL); }"), 62);
    // `parity` is the low bit of the popcount.
    assert_eq!(run_main("int main(){ return __builtin_parity(7); }"), 1);
    assert_eq!(run_main("int main(){ return __builtin_parity(6); }"), 0);
    // `copysign` takes the magnitude of the first operand and the
    // sign bit of the second, at the bit level.
    assert_eq!(run_main("int main(){ return __builtin_copysign(1.0, -2.0) < 0.0 ? 1 : 2; }"), 1);
    assert_eq!(run_main("int main(){ return __builtin_copysign(-1.0, 2.0) > 0.0 ? 1 : 2; }"), 1);
    assert_eq!(run_main("int main(){ return __builtin_copysignf(-1.0f, 2.0f) > 0.0f ? 1 : 2; }"), 1);
    // The `__builtin_` spellings rewrite to the libc names, so plain
    // calls expand inline too, like the other libm builtins.
    assert_eq!(
        run_main(
            "double copysign(double, double);\n\
             int main(){ return copysign(-1.0, 2.0) > 0.0 ? 1 : 2; }"
        ),
        1
    );
    // `__builtin_choose_expr` and `?:` evaluate one arm only.
    assert_eq!(run_main("int main(){ int y = 0; int z = __builtin_choose_expr(1, 3, y++); return z + y; }"), 3);
    assert_eq!(run_main("int main(){ int y = 0; return (y ?: 4); }"), 4);
    assert_eq!(run_main("int main(){ int y = 2; return (y ?: 4); }"), 2);
}

#[test]
fn case_ranges_and_member_address_constants() {
    // GNU case ranges dispatch with comparisons in the promoted
    // type's signedness, alongside exact cases.
    assert_eq!(
        run_main(
            "int f(unsigned long long x) { switch (x) { case 1 ... 5: return 1; case 6: return 2; case 100 ... 200: return 3; default: return 4; } }\n\
             int main() { return f(3) + f(6) * 10 + f(150) * 100 + f(7) * 1000; }"
        ),
        1 + 20 + 300 + 4000
    );
    assert_eq!(
        run_main(
            "int g(int x) { switch (x) { case -5 ... -1: return 1; case 0: return 2; default: return 3; } }\n\
             int main() { return g(-3) + g(0) * 10 + g(1) * 100; }"
        ),
        1 + 20 + 300
    );
    // `&p->m` and `p + k` are address constants: the root object
    // plus the member offset and stride.
    assert_eq!(
        run_main(
            "typedef struct { int a; char *name; } FOO;\n\
             FOO items[] = { {1, \"1\"}, {2, \"2\"}, {0, 0} };\n\
             int *second = (int *)&((items + 1)->a);\n\
             int *first = (int *)&(items->a);\n\
             int main() { return *second * 10 + *first; }"
        ),
        21
    );
}

#[test]
fn abs_and_rounding_builtins_expand_inline() {
    // `abs`/`labs`/`llabs` compute the magnitude without a call,
    // even when the program declares the name: GCC expands the
    // builtin at direct calls, which is what makes tests that
    // define an aborting `llabs` still pass.
    assert_eq!(
        run_main(
            "long long llabs(long long);\n\
             int main() { long long v = -3; return (int)(llabs(v) * 100 + __builtin_labs(-2L) * 10 + __builtin_abs(-1)); }"
        ),
        321
    );
    // Floats clear the sign bit, so `-0.0` comes back positive.
    assert_eq!(
        run_main(
            "double fabs(double);\n\
             int main() { double z = -0.0; double r = fabs(z); return (r == 0.0 && !__builtin_signbit(r)) ? 0 : 1; }"
        ),
        0
    );
    // `floor`/`ceil` lower to MIR rounding instructions.
    assert_eq!(
        run_main(
            "double floor(double);\n\
             int main() { double a = 1.7; double b = -1.2; return (int)(floor(a) * 10 + __builtin_ceil(b) + 2); }"
        ),
        11
    );
}

#[test]
fn compound_assignment_evaluates_the_lvalue_once() {
    // The lvalue's side effects run exactly once (C17 §6.5.16.2):
    // its address feeds the read, the operation, and the store.
    assert_eq!(
        run_main(
            "int main() {\n\
             int a[4] = {1, 2, 3, 4};\n\
             int i = 0;\n\
             int s = 0;\n\
             a[i++] += 10;\n\
             s += a[0] * 100 + i;\n\
             i = 0;\n\
             a[i++] *= 2;\n\
             s += a[0] * 10000 + i;\n\
             return s;\n\
             }"
        ),
        11 * 100 + 1 + 22 * 10000 + 1
    );
    // Pointer targets step once as well.
    assert_eq!(run_main("int main() { char b[4] = {0, 0, 0, 0}; char *p = b; *p++ += 3; return b[0] * 10 + (p - b); }"), 31);
    // Bit-field targets read-modify-write their covering unit and
    // evaluate to the stored (width-clamped) value.
    assert_eq!(
        run_main(
            "struct S { unsigned f : 4; };\n\
             int main() { struct S s = {3}; int i = 0; int arr[2] = {0, 0}; arr[i++] += 1; s.f += 5; return arr[0] * 100 + (int)s.f * 10 + i; }"
        ),
        100 + 80 + 1
    );
    // Pointer arithmetic scales by the pointee size.
    assert_eq!(
        run_main("int main() { int a[4] = {1, 2, 3, 4}; int *p = a; p += 2; *p -= 1; return a[2] * 10 + (p - a); }"),
        2 * 10 + 2
    );
    // The combination runs in the operator's type even when the
    // lvalue is narrower: `g += x` sums as `long double` and
    // converts back to `int` for the store.
    assert_eq!(
        run_main(
            "int g;\n\
             void h(double x) { g += x; }\n\
             int main() { h(3.75); return g; }"
        ),
        3
    );
    let module = lower_module("int g;\nvoid h(long double x) { g += x; }");
    let ld = mir::Ty::Float(match fe::target::Target::native().long_double {
        fe::target::LongDoubleKind::F80 => mir::FloatKind::F80,
        fe::target::LongDoubleKind::F128 => mir::FloatKind::F128,
        fe::target::LongDoubleKind::F64 => mir::FloatKind::F64,
    });
    let mut sums = 0;
    for func in &module.funcs {
        for inst in &func.insts {
            if matches!(&inst.kind, mir::InstKind::FAdd(..)) {
                let result = inst.result.expect("FAdd has a result");
                assert_eq!(func.values[result].ty, ld, "FAdd result type");
                sums += 1;
            }
        }
    }
    assert_eq!(sums, 1);
}

#[test]
fn old_style_calls_take_declared_arity() {
    // A definition without a prototype fixes its own arity: extra
    // arguments still evaluate (their side effects happen), but
    // only the declared parameters pass.
    assert_eq!(run_main("f(){return 3;} int main(){int x = 0; return f(x = 1) + x;}"), 4);
    assert_eq!(run_main("int g; g2(){g = 7; return 1;} int main(){return g2(1, 2) + g;}"), 8);
}

#[test]
fn builtin_fp_class_classify_and_prefetch() {
    // `signbit` sees the sign bit (`-0.0` counts) and converts
    // integers to `double` first.
    assert_eq!(run_main("int main(){return __builtin_signbit(-0.0) + 2 * __builtin_signbit(0.0);}"), 1);
    assert_eq!(run_main("int main(){return __builtin_signbit(-1) ? 3 : 4;}"), 3);
    // Long doubles classify through their own sign bit (the JIT
    // lacks the x87 negate helper, so this stays positive).
    assert_eq!(run_main("int main(){long double x = 2.5L; return __builtin_signbit(x) ? 5 : 6;}"), 6);
    // Classify-type codes fold from the type alone, so a long
    // double field is real (the 20040709 pattern).
    assert_eq!(run_main("int main(){return __builtin_classify_type(1.5);}"), 8);
    assert_eq!(run_main("int main(){return __builtin_classify_type(1);}"), 1);
    assert_eq!(
        run_main("struct W { long double l; }; int main(){struct W w; return __builtin_classify_type(w.l) == 8 ? 7 : 8;}"),
        7
    );
    // The prefetch address still evaluates; the hint itself is
    // nothing.
    assert_eq!(run_main("int main(){int *p = 0; __builtin_prefetch(p, 0, 3); return 0;}"), 0);
}

#[test]
fn recursive_structs() {
    // Self-reference through pointers: traversal, multi-hop
    // arrows, and dereference-then-dot all resolve per hop.
    let list = "struct Node{int v; struct Node *next;};\
        int sum(struct Node *p){int s = 0; while(p){s += p->v; p = p->next;} return s;}\
        int main(){struct Node n1; struct Node n2; struct Node n3;\
        n1.v = 10; n2.v = 20; n3.v = 30;\
        n1.next = &n2; n2.next = &n3; n3.next = 0;\
        return sum(&n1) * 100 + n1.next->next->v + (*n2.next).v;}";
    // 60, 30, 30.
    assert_eq!(run_main(list), 60 * 100 + 30 + 30);
    // Struct arrays through pointers scale by element size.
    assert_eq!(
        run_main(
            "struct P{int x; int y;}; struct P a[3] = {{1,2},{3,4},{5,6}};\
            int at(struct P *p, int i){return p[i].x + p[i].y;}\
            int main(){return at(a, 0) + at(a, 1) * 10 + at(a, 2) * 100;}"
        ),
        3 + 7 * 10 + 11 * 100
    );
    // Mutual recursion across two definitions.
    assert_eq!(
        run_main(
            "struct A{struct B *b; int x;}; struct B{struct A *a; int y;};\
            int main(){struct A a; struct B b;\
            a.x = 5; b.y = 7; a.b = &b; b.a = &a;\
            return a.b->y * 10 + b.a->x;}"
        ),
        75
    );
    // Pointers to never-defined structs pass through opaquely.
    assert_eq!(
        run_main(
            "int isnull(struct Opaque *p){return p == 0;}\
            int main(){return isnull(0) + isnull((void *)1);}"
        ),
        1
    );
}

#[test]
#[should_panic(expected = "embeds itself by value")]
fn recursive_struct_by_value_panics() {
    // Built below the parser (which rejects by-value recursion with
    // its own diagnostic): the lowering backstop still fires.
    let mut interner = fe::lexer::Interner::new();
    let mut sema = fe::sema::Sema::leak_for_tests();
    let tag = interner.get_or_intern("S");
    let span = fe::span::Span::dummy();
    let id = sema.define_tag(tag, false, span).unwrap();
    let v = interner.get_or_intern("v");
    sema.complete_tag(
        id,
        vec![fe::sema::Member {
            name: Some(v),
            ty: fe::parse::Ty::Struct(id),
            bitfield: None,
            packed: false,
            align_override: None,
            span,
        }],
    )
    .unwrap();
    let _ = crate::ty::lower_ty(&fe::parse::Ty::Struct(id), &sema, &interner);
}

#[test]
fn call_result_aggregate_assign() {
    // A register-sized aggregate call result is an address like
    // any other aggregate value: storing it to a local must copy
    // from the reassembled slot, not the raw register.
    assert_eq!(
        run_main(
            "struct P{int x; int y;}; struct P mk(int n){struct P p; p.x = n; p.y = n+1; return p;}\
            int main(){struct P p = mk(10); return p.x + p.y;}"
        ),
        21
    );
    assert_eq!(
        run_main(
            "union N{int a; int b;}; union N mk(int n){union N u; u.b = n; return u;}\
            int main(){union N w = mk(7); return w.a;}"
        ),
        7
    );
}

#[test]
fn unions() {
    // Overlaid members share storage; access resolves per member.
    assert_eq!(run_main("union N{int a; int b;}; int main(){union N u; u.a = 42; return u.a + u.b;}"), 84);
    assert_eq!(run_main("union N{int a; int b;}; int main(){union N u; u.b = 42; return u.a;}"), 42);
    // By value in both directions, including larger blobs.
    assert_eq!(
        run_main(
            "union N{int a; int b;}; int get(union N u){return u.a;}\
            int main(){union N u; u.b = 9; return get(u);}"
        ),
        9
    );
    assert_eq!(
        run_main(
            "union Big{long l[3]; int i;}; long sum(union Big u){return u.l[0]+u.l[1]+u.l[2];}\
            int main(){union Big u; u.l[0] = 1; u.l[1] = 2; u.l[2] = 3; return (int)sum(u);}"
        ),
        6
    );
    // Nested, designated, and global initializers address offset
    // zero with the selected member's type.
    assert_eq!(run_main("union N{int a; int b;}; int main(){union N u = {5}; return u.a + u.b;}"), 10);
    assert_eq!(run_main("union N{int a; int b;}; int main(){union N u = {.b = 6}; return u.a;}"), 6);
    assert_eq!(run_main("union N{int a; int b;}; union N g = {7}; int main(){return g.a + g.b;}"), 14);
    assert_eq!(
        run_main("union N{int a; int b;}; struct W{union N u; int t;}; int main(){struct W w; w.u.a = 3; w.t = 4; return w.u.b + w.t;}"),
        7
    );
}

#[test]
fn bitfields() {
    // Reads extract and sign-extend; unsigned fields zero-extend
    // even though narrow ones promote to signed `int`.
    assert_eq!(
        run_main("struct S{int x:3; unsigned y:5;}; int main(){struct S s; s.x = -3; s.y = 17; return s.x * 100 + s.y;}"),
        -300 + 17
    );
    assert_eq!(
        run_main("struct S{int x:3; int y:5;}; int main(){struct S s; s.x = 1; s.y = 2; s.x++; ++s.y; return s.x * 100 + s.y;}"),
        203
    );
    // Assignment evaluates to the stored value; out-of-range
    // stores wrap like GCC (which warns and yields -12 here).
    assert_eq!(run_main("struct S{int x:3; int y:5;}; int main(){struct S s; int q = (s.y = 20); return q;}"), -12);
    // Shared units preserve neighbors; `:0` and unnamed widths
    // shape layout without values.
    assert_eq!(
        run_main("struct S{int a:3; int :5; int b:8;}; int main(){struct S s; s.a = -1; s.b = -56; return s.a * 100 + s.b;}"),
        -156
    );
    assert_eq!(
        run_main("struct S{int a:4; int :0; int b:4;}; int main(){struct S s; s.a = 1; s.b = 2; return s.a * 10 + s.b;}"),
        12
    );
    // Chars and wide fields across unit widths.
    assert_eq!(
        run_main("struct S{char c:2; long long l:40;}; int main(){struct S s; s.c = 1; s.l = 1000000000L; return s.c + (int)(s.l / 1000000000L);}"),
        2
    );
    assert_eq!(
        run_main("struct M{char c:2; long long l:40;}; int main(){struct M m; m.c = 1; m.l = 5; return m.c + (int)m.l + (int)sizeof(struct M);}"),
        1 + 5 + 8
    );
    assert_eq!(run_main("struct W{unsigned long long u:40;}; int main(){struct W w; w.u = 5; return (int)w.u;}"), 5);
    assert_eq!(run_main("struct S{int z:17;}; int main(){struct S s; s.z = -1000; return s.z;}"), -1000);
    // Positional (skipping unnamed), designated, nested, and
    // global initializers.
    assert_eq!(run_main("struct S{int x:3; int y:5;}; int main(){struct S s = {5, 20}; return s.x + s.y;}"), -3 + -12);
    assert_eq!(run_main("struct S{int x:3; int y:5;}; int main(){struct S s = {.y = 9}; return s.x + s.y;}"), 9);
    assert_eq!(
        run_main("struct U{int a:3; int :5; int b:8;}; int main(){struct U u = {7, 200}; return u.a + u.b;}"),
        -1 + -56
    );
    assert_eq!(run_main("struct S{int x:3; int y:5;}; struct S g = {5, 20}; int main(){return g.x + g.y;}"), -3 + -12);
    assert_eq!(
        run_main("struct I{int v;}; struct S{struct I i; int x:3;}; int main(){struct S s; s.i.v = 1; s.x = 2; return s.i.v + s.x;}"),
        3
    );
    // By value across calls, exercising unit classification.
    assert_eq!(
        run_main(
            "struct S{int x:3; int y:5;}; int get(struct S s){return s.x + s.y;}\
            int main(){struct S s; s.x = 1; s.y = 2; return get(s);}"
        ),
        3
    );
    // Compound literals, arrays of bit-field structs, and nested
    // designated initializers compose through the same paths.
    assert_eq!(run_main("struct S{int x:3; int y:5;}; int main(){return ((struct S){1, 2}).x + ((struct S){1, 2}).y;}"), 3);
    assert_eq!(
        run_main("struct S{int x:3; int y:5;}; int main(){struct S a[2] = {{1, 2}, {3, 4}}; return a[0].x + a[0].y + a[1].x + a[1].y;}"),
        10
    );
    assert_eq!(
        run_main("struct I{int v;}; struct S{struct I i; int x:3;}; int main(){struct S s = {.i.v = 7, .x = 2}; return s.i.v + s.x;}"),
        9
    );
    // GCC computes bit-field arithmetic at the widest operand's
    // precision: an `int` or `unsigned` operand keeps a 40-bit
    // field at 40 bits, while `unsigned long long` widens to 64.
    assert_eq!(
        run_main(
            "struct W{unsigned long long u:40;}; int main(){struct W w; w.u = 2; return (unsigned long long)(w.u - 8) == 0xfffffffffaULL;}"
        ),
        1
    );
    assert_eq!(
        run_main("struct W{unsigned long long u:40;}; int main(){struct W w; w.u = 2; return (unsigned long long)((w.u - 8) + 8) == 2;}"),
        1
    );
    assert_eq!(
        run_main(
            "struct W{unsigned long long u:40;}; int main(){struct W w; w.u = 2; return (unsigned long long)(w.u - 8ULL) == 0xfffffffffffffffaULL;}"
        ),
        1
    );
    assert_eq!(
        run_main(
            "struct W{unsigned long long u:40;}; int main(){struct W w; w.u = 2; return ((unsigned long long)(w.u - 8)) + 8 == 0x10000000002ULL;}"
        ),
        1
    );
    assert_eq!(
        run_main(
            "struct W{unsigned long long u:40;}; int main(){struct W w; w.u = 2; unsigned int eight = 8; return (unsigned long long)(w.u - eight) == 0xfffffffffaULL;}"
        ),
        1
    );
}

#[test]
fn unprototyped_signatures_admit_arguments() {
    // A C89 implicit declaration (`extern int f()`) fixes no
    // parameter list: the import's signature must accept the
    // default-promoted actuals.
    let module = lower_module("int main(){return foo(1, 2);}");
    let import = module.externs.iter().find(|e| e.name == "foo").expect("foo import");
    assert!(import.sig.variadic && import.sig.params.is_empty(), "{:?}", import.sig);

    // The same holds for an indirect call through a function
    // pointer declared without a prototype.
    let module = lower_module("int main(){int (*p)() = 0; return p(1, 2);}");
    let indirect = module
        .funcs
        .iter()
        .flat_map(|f| f.insts.iter())
        .find_map(|inst| match &inst.kind {
            mir::InstKind::CallIndirect { sig, .. } => Some(sig),
            _ => None,
        })
        .expect("indirect call");
    assert!(indirect.variadic && indirect.params.is_empty(), "{indirect:?}");
}

#[test]
fn unprototyped_definitions_take_their_own_shape() {
    // A parameterless definition (`int f()`) reads no arguments (no
    // `...`, so no `va_start`), so its signature must not carry the
    // variadic register save area. Declarations without a body keep
    // the variadic shape: callers may pass any argument list.
    let module = lower_module("int main(){return 0;}");
    let main = module.funcs.iter().find(|f| f.name == "main").expect("main");
    assert!(!main.sig.variadic, "{:?}", main.sig);

    let module = lower_module("int div3(int n){return n/3;} int main(){return div3(4);}");
    let div3 = module.funcs.iter().find(|f| f.name == "div3").expect("div3");
    assert!(!div3.sig.variadic, "{:?}", div3.sig);

    // K&R definitions are rewritten into prototyped functions with
    // promoted parameter types, so they are fixed shapes as well.
    let module = lower_module("int f(a) int a; { return a; } int main(){return f(1);}");
    let f = module.funcs.iter().find(|f| f.name == "f").expect("f");
    assert!(!f.sig.variadic && f.sig.params.len() == 1, "{:?}", f.sig);
}

#[test]
fn anonymous_member_initializers_flatten() {
    // Flat positional items descend through anonymous members in
    // both locals and globals (C17 §6.7.9p20).
    assert_eq!(
        run_main("struct S{struct{int a,b;};int c;}; int main(){struct S s={5,6,7}; return s.a+s.b*10+s.c*100;}"),
        765
    );
    assert_eq!(run_main("struct S{struct{int a,b;};int c;} s={5,6,7}; int main(){return s.a+s.b*10+s.c*100;}"), 765);
    // Nested anonymous structs and array members in the same run.
    assert_eq!(
        run_main("struct S{struct{struct{int a;} i; int b[2];};int c;}; int main(){struct S s={1,2,3,4}; return s.i.a+s.b[0]*10+s.b[1]*100+s.c*1000;}"),
        4321
    );
    assert_eq!(
        run_main("struct S{struct{struct{int a;} i; int b[2];};int c;} s={1,2,3,4}; int main(){return s.i.a+s.b[0]*10+s.b[1]*100+s.c*1000;}"),
        4321
    );
    // Braced members keep their own grouping, and flat items
    // resume at the enclosing struct.
    assert_eq!(
        run_main("struct S{struct{int a,b;};int c;}; int main(){struct S s={{1,2},3}; return s.a+s.b*10+s.c*100;}"),
        321
    );
    assert_eq!(run_main("struct S{struct{int a,b;};int c;} s={{1,2},3}; int main(){return s.a+s.b*10+s.c*100;}"), 321);
    // An anonymous union stays one member: the next flat item
    // initializes the following member, not another union field.
    assert_eq!(run_main("struct S{union{int a;int b;};int c;}; int main(){struct S s={1,2}; return s.a*10+s.c;}"), 12);
    assert_eq!(run_main("struct S{union{int a;int b;};int c;} s={1,2}; int main(){return s.a*10+s.c;}"), 12);
}

/// C text all the way to MIR verification, without executing:
/// for shapes the JIT cannot link (long-double arithmetic needs
/// the `glccrt` helpers) and for declarations that only need to
/// lower through the verifier.
fn lower(src: &str) {
    let mir_module = lower_module(src);
    let errors = mir::verify::verify_module(&mir_module);
    assert!(errors.is_empty(), "verify errors in {src}: {errors:?}");
}

/// The module global with this exact symbol name.
fn global_named<'m>(module: &'m mir::Module, name: &str) -> &'m mir::GlobalData {
    module.globals.iter().find(|g| g.name == name).unwrap_or_else(|| panic!("no global named {name}"))
}

#[test]
fn block_scope_function_declarations_resolve_to_definitions() {
    // `float f();` inside a function names the file-scope `f`
    // defined later in the TU: calls go straight to the
    // definition, not to an import of the same name (930603-1).
    let src = "int main(){ float fx(), inita(); return (int)(fx(2.0) + inita()); }\
               float fx(x) float x; { return x + 0.25; }\
               float inita(){ return 3.0; }";
    let module = lower_module(src);
    assert!(module.externs.iter().all(|e| e.name != "fx" && e.name != "inita"), "block-scope prototypes became imports");
    assert_eq!(run_main(src), 5);
}

#[test]
fn union_member_initializers_step_through_the_member_type() {
    // The union's storage is the member's: initializing a struct
    // member (or a member below a union designator) steps with the
    // member's type, not the union image (20071029-1, strlen-5).
    assert_eq!(
        run_main(
            "union U { struct { int f1; int f2; } f; char s[16]; long a; };\
             int main(){ union U u = { { 3, 4 } }; union U v = { .f = { 5, 6 } }; return u.f.f1 + u.f.f2 + v.f.f1 + v.f.f2; }"
        ),
        3 + 4 + 5 + 6
    );
    assert_eq!(
        run_main(
            "struct S { union { struct { int a; } i; long l; } u; int z; };\
             int main(){ struct S s = { .u.i.a = 7, .z = 2 }; return s.u.i.a + s.z; }"
        ),
        9
    );
}

#[test]
fn address_constants_into_string_literals() {
    // A string literal is an object, so `"foo" + 1` and
    // `&("abc"[2])` are address constants with an addend
    // (pr53084, 921019-1).
    let module = lower_module(
        "static const char *const foo[] = {\"foo\" + 1};\
         static char *x = &(\"abc\"[2]);\
         int main(){ return foo[0][0] == 'o' && *x == 'c' ? 0 : 1; }",
    );
    let (_, reloc) = &global_named(&module, "foo").relocs[0];
    assert!(matches!(reloc, mir::DataReloc::Global { addend: 1, .. }), "{reloc:?}");
    let (_, reloc) = &global_named(&module, "x").relocs[0];
    assert!(matches!(reloc, mir::DataReloc::Global { addend: 2, .. }), "{reloc:?}");
    assert_eq!(
        run_main(
            "static const char *const foo[] = {\"foo\" + 1};\
             static char *x = &(\"abc\"[2]);\
             int main(){ return foo[0][0] == 'o' && *x == 'c' ? 0 : 1; }"
        ),
        0
    );
}

#[test]
fn ptr32_global_addresses_become_narrow_relocs() {
    let module = lower_module_ms(
        "int x;\
         int *__ptr32 g = &x;\
         int *h = &x;\
         struct S { int *__ptr32 a; int *b; };\
         struct S s = { &x, &x };\
         int main(){ return 0; }",
    );
    // The `__ptr32` field patches 4 bytes, so it never reaches the
    // pointer-width list the object backend emits directly.
    let g = global_named(&module, "g");
    assert!(g.relocs.is_empty(), "no pointer-width patch for __ptr32");
    assert_eq!(g.relocs32.len(), 1);
    assert_eq!(g.relocs32[0].0, 0);
    assert!(matches!(g.relocs32[0].1, mir::DataReloc::Global { addend: 0, .. }), "{:?}", g.relocs32[0].1);

    let h = global_named(&module, "h");
    assert!(h.relocs32.is_empty());
    assert_eq!(h.relocs.len(), 1);

    // Mixed struct: the 32-bit member at offset 0, the ordinary
    // pointer at offset 8, each in its own list.
    let s = global_named(&module, "s");
    assert_eq!(s.relocs32.iter().map(|(off, _)| *off).collect::<Vec<_>>(), [0]);
    assert_eq!(s.relocs.iter().map(|(off, _)| *off).collect::<Vec<_>>(), [8]);
}

#[test]
fn tls_globals_carry_the_tls_flag() {
    let module = lower_module(
        "_Thread_local int x = 1;\
         extern _Thread_local int y;\
         int f(void){ static _Thread_local int z = 2; return z; }\
         int main(){ return x + f() + y; }",
    );
    assert!(global_named(&module, "x").tls, "file-scope TLS");
    assert!(global_named(&module, "y").tls, "extern TLS import");
    assert!(global_named(&module, "y").external);
    let block: Vec<_> = module.globals.iter().filter(|g| g.tls && g.local && g.name.starts_with("z.")).collect();
    assert_eq!(block.len(), 1, "block-scope static TLS");
}

#[test]
fn weak_marks_functions_and_globals() {
    let module = lower_module(
        "__attribute__((weak)) int w = 3;\
         __attribute__((weak)) int f(void){ return w; }\
         int g(void){ return f(); }\
         int main(){ return g() + w; }",
    );
    let w = global_named(&module, "w");
    assert!(w.weak);
    assert!(!w.local, "weak is preemptible, not internal");
    let weak_func = module.funcs.iter().find(|f| f.name == "f").unwrap();
    assert!(weak_func.weak);
    let strong = module.funcs.iter().find(|f| f.name == "main").unwrap();
    assert!(!strong.weak && !strong.local);
    // A weak declaration with no definition stays weak on the
    // import (function and object), matching GCC's `STB_WEAK`
    // undefined symbols.
    let module = lower_module(
        "extern int __attribute__((weak)) v;\
         extern int __attribute__((weak)) f(void);\
         int main(){ return (f ? f() : 7) + (v ? v : 9); }",
    );
    let v = global_named(&module, "v");
    assert!(v.external && v.weak, "weak extern object");
    let f = module.externs.iter().find(|e| e.name == "f").expect("f import");
    assert!(f.weak, "weak extern function");
}

#[test]
fn definitions_carry_natural_alignment() {
    let module = lower_module(
        "double d = 1.5;\
         char buf[3];\
         struct S { char c; long l; } s;\
         char page[16] __attribute__((aligned(4096)));\
         static char slot[8] __attribute__((aligned(64)));\
         long tight __attribute__((aligned(1)));\
         int main(){ return (int)d + buf[0] + s.c + page[0] + slot[0] + (int)tight; }",
    );
    assert_eq!(global_named(&module, "d").align, Some(8));
    assert_eq!(global_named(&module, "buf").align, Some(1));
    assert_eq!(global_named(&module, "s").align, Some(fe::target::Target::native().long_align));
    // `aligned` sets the alignment, raising or lowering it (GCC).
    assert_eq!(global_named(&module, "page").align, Some(4096));
    assert_eq!(global_named(&module, "slot").align, Some(64));
    assert_eq!(global_named(&module, "tight").align, Some(1));
}

#[test]
fn constructors_and_destructors_build_hook_arrays() {
    let module = lower_module(
        "int a; int b;\
         void c2(void) __attribute__((constructor(200))) { a = 2; }\
         void c3(void) __attribute__((constructor)) { a = 3; }\
         void c1(void) __attribute__((constructor(100))) { a = 1; }\
         void d1(void) __attribute__((destructor)) { b = 1; }\
         void plain(void){}\
         int main(){ return a + b; }",
    );
    let ctor = global_named(&module, "__glcc_init_array");
    assert!(ctor.local);
    assert_eq!(ctor.section.as_deref(), Some(".init_array"));
    assert_eq!(ctor.align, Some(8));
    assert_eq!(ctor.relocs.iter().map(|(off, _)| *off).collect::<Vec<_>>(), [0, 8, 16]);
    let names: Vec<&str> = ctor
        .relocs
        .iter()
        .map(|(_, r)| match r {
            mir::DataReloc::Func { target } => module.funcs[*target].name.as_str(),
            other => panic!("non-function hook reloc {other:?}"),
        })
        .collect();
    // Priority order first (bare constructors are 65535), then
    // declaration order; `plain` is never a hook.
    assert_eq!(names, ["c1", "c2", "c3"]);
    let dtor = global_named(&module, "__glcc_fini_array");
    assert_eq!(dtor.section.as_deref(), Some(".fini_array"));
    assert_eq!(dtor.relocs.len(), 1);
}

#[test]
fn undeclared_or_absent_hooks_emit_no_arrays() {
    // A constructor with no body never runs: no array entry, and
    // with no defined hooks at all no array object exists.
    let module = lower_module(
        "void never(void) __attribute__((constructor));\
         int main(){ return 0; }",
    );
    assert!(!module.globals.iter().any(|g| g.name.starts_with("__glcc_")));
}

#[test]
fn complex_literals() {
    assert_eq!(run_main("int main(){double _Complex x = 1.0 + 2.0i; return (int)(__real__ x + __imag__ x);}"), 3);
    assert_eq!(run_main("int main(){double _Complex x = 2.0i; return (int)(__real__ x + __imag__ x);}"), 2);
    lower("int main(){double _Complex x = 2.0i; return (int)(__real__ x + __imag__ x);}");
    assert_eq!(run_main("int main(){double _Complex x = 2.0i; return (int)(__real__ x + __imag__ x);}"), 2);
    assert_eq!(run_main("int main(){float _Complex x = 1.0fi; return (int)(__real__ x + __imag__ x);}"), 1);
    assert_eq!(run_main("int main(){__complex__ int x = 1 + 2i; return __real__ x + __imag__ x;}"), 3);
    assert_eq!(run_main("int main(){__complex__ double x = 1j; return (int)(__real__ x + __imag__ x);}"), 1);
}

#[test]
fn complex_arithmetic() {
    // (1+2i)+(3+4i) = 4+6i.
    assert_eq!(
        run_main("int main(){double _Complex a = 1.0 + 2.0i; double _Complex b = 3.0 + 4.0i; double _Complex c = a + b; return (int)__real__ c * 100 + (int)__imag__ c;}"),
        406
    );
    // (1+2i)-(3+4i) = -2-2i.
    assert_eq!(
        run_main("int main(){double _Complex a = 1.0 + 2.0i; double _Complex b = 3.0 + 4.0i; double _Complex c = a - b; return (int)__real__ c * 100 + (int)__imag__ c;}"),
        -202
    );
    // (1+2i)(3+4i) = -5+10i.
    assert_eq!(
        run_main("int main(){double _Complex a = 1.0 + 2.0i; double _Complex b = 3.0 + 4.0i; double _Complex c = a * b; return (int)__real__ c * 100 + (int)__imag__ c;}"),
        -490
    );
    // (4+0i)/(2+0i) = 2+0i, exact in float.
    assert_eq!(
        run_main("int main(){double _Complex a = 4.0; double _Complex b = 2.0; double _Complex c = a / b; return (int)__real__ c * 100 + (int)__imag__ c;}"),
        200
    );
    // Mixed complex/real through the usual conversions.
    assert_eq!(
        run_main(
            "float __complex__ p(float __complex__ a, float __complex__ b){return a + b;} int main(){float __complex__ x = 1.0f + 2.0fi; return (int)__real__ p(x, 3.0f) * 10 + (int)__imag__ p(x, 3.0f);}"
        ),
        42
    );
    // Integer complex wraps and truncates like scalars.
    assert_eq!(
        run_main("int main(){__complex__ int a = 1 + 2i; __complex__ int b = 3 + 4i; __complex__ int c = a * b; return __real__ c * 100 + __imag__ c;}"),
        -490
    );
    assert_eq!(run_main("int main(){__complex__ int a = 7; __complex__ int b = 2; return __real__(a / b);}"), 3);
    assert_eq!(run_main("int main(){__complex__ int a = -7; __complex__ int b = 2; return __real__(a / b);}"), -3);
    assert_eq!(
        run_main("int main(){__complex__ unsigned a = 7; __complex__ unsigned b = 2; return (int)__real__(a / b);}"),
        3
    );
    // Unary minus, identity, and conjugate.
    assert_eq!(
        run_main("int main(){double _Complex a = 1.0 + 2.0i; double _Complex c = -a + +a + ~a; return (int)__real__ c * 100 + (int)__imag__ c;}"),
        98
    );
    // Compound assignment (the checker pre-combines the op).
    assert_eq!(
        run_main("int main(){double _Complex a = 1.0 + 2.0i; a += 1.0; a -= 1.0i; a *= 2.0; a /= 2.0; return (int)__real__ a * 10 + (int)__imag__ a;}"),
        21
    );
}

#[test]
fn complex_comparisons_and_conditions() {
    assert_eq!(
        run_main(
            "int f(double _Complex a, double _Complex b){return (a == b) + (a != b) * 10;} int main(){double _Complex a = 1.0 + 2.0i; return f(a, a) * 100 + f(a, 1.0 + 3.0i);}"
        ),
        110
    );
    // Ordered equality: +0.0 equals -0.0 in both parts.
    assert_eq!(run_main("int main(){if ((__complex__ double)0.0 != (__complex__ double)(-0.0)) return 1; return 0;}"), 0);
    // A complex condition holds when either part is nonzero.
    assert_eq!(
        run_main("int f(double _Complex a){if (a) return 1; return 0;} int main(){return f(0.0) * 100 + f(1.0i) * 10 + f(1.0);}"),
        11
    );
    assert_eq!(
        run_main("int f(double _Complex a){return !a;} int main(){return f(0.0) * 10 + f(1.0i) + f(1.0 + 1.0i) * 100;}"),
        10
    );
    assert_eq!(
        run_main("int main(){double _Complex z = 0.0; double _Complex o = 1.0; return (z && o) + (o && z) * 10 + (o && o) * 100 + (z || o);;}"),
        101
    );
    // Integer components compare exactly.
    assert_eq!(
        run_main("int f(__complex__ int a, __complex__ int b){return (a == b) + (a != b);} int main(){__complex__ int a = 1 + 2i; return f(a, a) * 10 + f(a, 1 + 3i);}"),
        11
    );
}

#[test]
fn complex_conversions() {
    // Real to complex: value as the real part, zero imaginary.
    assert_eq!(
        run_main("int main(){double x = 2.5; double _Complex c = x; return (int)__real__ c * 10 + (int)__imag__ c;}"),
        20
    );
    // Complex to real: the real part, imaginary discarded.
    assert_eq!(run_main("int main(){double _Complex c = 2.5 + 3.5i; double r = c; int n = c; return (int)r * 10 + n;}"), 22);
    // Complex to `_Bool`: nonzero when either part is.
    assert_eq!(
        run_main("int main(){double _Complex z = 0.0; double _Complex i = 1.0i; _Bool a = z; _Bool b = i; return a * 10 + b;}"),
        1
    );
    // Complex to complex across precisions.
    assert_eq!(
        run_main("int main(){float _Complex f = 1.5f + 2.5fi; double _Complex d = f; return (int)__real__ d + (int)__imag__ d;}"),
        3
    );
    assert_eq!(
        run_main("int main(){double _Complex d = 1.5 + 2.5i; float _Complex f = d; return (int)__real__ f + (int)__imag__ f;}"),
        3
    );
    assert_eq!(
        run_main("int main(){double _Complex d = 1.0 + 2.0i; __complex__ int n = d; return __real__ n * 100 + __imag__ n;}"),
        102
    );
    assert_eq!(
        run_main("int main(){int n = 7; double _Complex c = (double _Complex)n; return (int)__real__ c + (int)__imag__ c;}"),
        7
    );
}

#[test]
fn complex_real_imag_lvalues() {
    assert_eq!(
        run_main("double f(double _Complex x){return __real__ x + __imag__ x;} int main(){double _Complex a = 0.0; __real__ a = 1.5; __imag__ a = 2.5; return (int)f(a);}"),
        4
    );
    assert_eq!(
        run_main("void g(double _Complex *p){__real__ *p = 1.0; __imag__ *p = 2.0;} int main(){double _Complex a = 0.0; g(&a); return (int)(__real__ a + __imag__ a);}"),
        3
    );
    assert_eq!(
        run_main("double *f(double _Complex *p){return &__real__ *p;} int main(){double _Complex a = 5.0 + 6.0i; *f(&a) = 7.0; return (int)__real__ a + (int)__imag__ a;}"),
        13
    );
    assert_eq!(
        run_main("int main(){double _Complex a[2]; __real__ a[0] = 1.0; __imag__ a[1] = 2.0; return (int)(__real__ a[0] + __imag__ a[1]);}"),
        3
    );
    // Parts step like real lvalues.
    assert_eq!(
        run_main("int main(){double _Complex a = 1.0 + 2.0i; __real__ a += 10.0; ++__imag__ a; return (int)__real__ a * 100 + (int)__imag__ a;}"),
        1103
    );
    // Parts of rvalues (calls, arithmetic) extract off the pair.
    assert_eq!(
        run_main("double _Complex f(void){return 1.0 + 2.0i;} int main(){return (int)__real__ f() * 10 + (int)__imag__ f();}"),
        12
    );
    assert_eq!(
        run_main("int main(){double _Complex a = 1.0 + 2.0i; double _Complex b = 3.0 + 4.0i; return (int)__real__(a + b) * 10 + (int)__imag__(a + b);}"),
        46
    );
    assert_eq!(run_main("int main(){double _Complex a = 1.0 + 2.0i; return (int)__real__ +a + (int)__imag__ -a;}"), -1);
}

#[test]
fn complex_calls_ternary_and_aggregates() {
    assert_eq!(
        run_main("double _Complex id(double _Complex x){return x;} int main(){double _Complex c = id(1.0 + 2.0i) + id(3.0); return (int)__real__ c * 10 + (int)__imag__ c;}"),
        42
    );
    assert_eq!(
        run_main("int main(){int c = 1; double _Complex a = 5.0 + 6.0i; double _Complex r = c ? a : 2.0 * a; return (int)__real__ r + (int)__imag__ r;}"),
        11
    );
    assert_eq!(
        run_main(
            "double _Complex g(void); double _Complex f(void){double _Complex x = g(); return x;} double _Complex g(void){return 1.0 + 2.0i;} int main(){return (int)__real__ f() + (int)__imag__ f();}"
        ),
        3
    );
    // Complex inside structs and arrays, by value and by address.
    assert_eq!(
        run_main("struct S{double _Complex v; int k;}; int main(){struct S s; s.v = 1.0i; s.k = 3; return (int)__imag__ s.v + s.k;}"),
        4
    );
    assert_eq!(
        run_main(
            "struct S{double _Complex v; int k;}; struct S id(struct S s){return s;} int main(){struct S s; s.v = 1.0 + 2.0i; s.k = 5; struct S t = id(s); return (int)__real__ t.v + (int)__imag__ t.v + t.k;}"
        ),
        8
    );
    assert_eq!(
        run_main("int main(){double _Complex a[2]; a[0] = 1.0; a[1] = 2.0i; return (int)(__real__ a[0] + __imag__ a[1]);}"),
        3
    );
}

#[test]
fn complex_builtins() {
    assert_eq!(
        run_main("int main(){double _Complex x = 1.0 + 2.0i; double _Complex y = __builtin_conj(x); return (int)__real__ y * 10 + (int)__imag__ y;}"),
        8
    );
    assert_eq!(
        run_main("int main(){float _Complex x = 1.0f + 2.0fi; float _Complex y = __builtin_conjf(x); return (int)__real__ y * 10 + (int)__imag__ y;}"),
        8
    );
    // 3-4-5 triangle: exact under the naive magnitude.
    assert_eq!(
        run_main("int main(){double _Complex x = 3.0 + 4.0i; return (int)(__builtin_creal(x) + __builtin_cimag(x) + __builtin_cabs(x));}"),
        12
    );
    assert_eq!(
        run_main("int main(){float _Complex x = 3.0f + 4.0fi; return (int)(__builtin_crealf(x) + __builtin_cimagf(x) + __builtin_cabsf(x));}"),
        12
    );
    // Glibc spellings with exact signatures lower inline.
    assert_eq!(
        run_main(
            "double creal(double _Complex); double cimag(double _Complex); double cabs(double _Complex); int main(){double _Complex x = 3.0 + 4.0i; return (int)(creal(x) + cimag(x) + cabs(x));}"
        ),
        12
    );
    assert_eq!(
        run_main("double _Complex conj(double _Complex); int main(){double _Complex x = 1.0 + 2.0i; return (int)__real__ conj(x) * 10 + (int)__imag__ conj(x);}"),
        8
    );
    // `f`-suffixed glibc spellings, and `~` over float complex.
    assert_eq!(
        run_main(
            "float crealf(float _Complex); float cimagf(float _Complex); float cabsf(float _Complex); int main(){float _Complex x = 3.0f + 4.0fi; return (int)(crealf(x) + cimagf(x) + cabsf(x));}"
        ),
        12
    );
    assert_eq!(
        run_main(
            "float _Complex conjf(float _Complex); int main(){float _Complex x = 1.0f + 2.0fi; float _Complex y = conjf(x) + ~x; return (int)__real__ y * 10 + (int)__imag__ y;}"
        ),
        16
    );
    // Same-named user functions with other signatures still call out.
    lower("int conj(int x); int f(void){return conj(1);}");
    lower("float creal(int x); float f(void){return creal(1);}");
}

#[test]
fn complex_globals() {
    assert_eq!(run_main("double _Complex g = 1.0 + 2.0i; int main(){return (int)__real__ g + (int)__imag__ g;}"), 3);
    assert_eq!(run_main("float _Complex g; int main(){g = 3.0fi; return (int)__imag__ g + (int)__real__ g;}"), 3);
    assert_eq!(run_main("__complex__ int g = 4 + 5i; int main(){return __real__ g + __imag__ g;}"), 9);
    // Remainder-zeroed globals read back their parts.
    assert_eq!(run_main("double _Complex g = 5.0i; int main(){return (int)__real__ g * 10 + (int)__imag__ g;}"), 5);
    // Block-scope statics fold once and persist across calls.
    assert_eq!(
        run_main("int f(){static double _Complex c = 1.0 + 2.0i; __real__ c += 1.0; return (int)__real__ c;} int main(){f(); return f();}"),
        3
    );
}

#[test]
fn flexible_array_member_globals() {
    // An initialized flexible array member grows the object past
    // `sizeof` (GCC extension); the declared type keeps its size.
    assert_eq!(
        run_main("struct F { int i; int j[]; }; struct F x = { 1, { 2, 3 } }; int main(){return x.i + x.j[0] + x.j[1];}"),
        6
    );
    assert_eq!(run_main("struct F { int i; int j[]; }; struct F x = { 1, { 2, 3 } }; int main(){return sizeof x;}"), 4);
    // A string literal fills a flexible char member, and a
    // designator into one sizes the object by the element it
    // lands on.
    assert_eq!(
        run_main("struct C { char c; char p[]; }; struct C s = { 'o', \"wx\" }; int main(){return s.p[0] + s.p[1] - 'w' - 'x';}"),
        0
    );
    assert_eq!(run_main("struct F { int i; int j[]; }; struct F x = { .j[1] = 9 }; int main(){return x.j[0] + x.j[1];}"), 9);
    // An empty flexible initializer adds nothing.
    assert_eq!(run_main("struct F { int i; int j[]; }; struct F x = { .i = 4, .j = {} }; int main(){return x.i;}"), 4);
}

#[test]
fn complex_long_double_lowers() {
    // Long-double arithmetic needs the `glccrt` helpers the JIT
    // cannot link, so these only lower through the verifier.
    lower("long double _Complex f(void){long double _Complex x = 1.0iL; return x;}");
    lower("long double _Complex f(long double _Complex a, long double _Complex b){return a + b + a - b + a * b + a / b;}");
    lower("int f(long double _Complex a, long double _Complex b){return (a == b) + (a != b);}");
    lower("long double _Complex f(double _Complex x){return x;}");
    lower("double _Complex f(long double _Complex x){return x;}");
    lower("long double f(long double _Complex x){return __builtin_creall(x) + __builtin_cimagl(x);}");
    lower("long double _Complex f(long double _Complex x){return __builtin_conjl(x);}");
    lower("long double f(long double _Complex x){return __builtin_cabsl(x);}");
    lower("int f(long double _Complex a){if (a) return 1; return 0;}");
    lower("int f(long double _Complex a){return !a;}");
    lower("long double _Complex f(long double _Complex a){return -a + +a + ~a;}");
    lower("long double _Complex g = 1.0L + 2.0iL; long double _Complex f(void){return g;}");
    lower("__complex__ long f(void){__complex__ long x = 3 + 4i; return x;}");
    lower("long double _Complex f(int c, long double _Complex a){return c ? a : 2.0L * a;}");
}

#[test]
fn locals_and_global() {
    assert_eq!(run_main("int main(){int x = 6; int y = 7; return x+y;}"), 13);
    assert_eq!(run_main("int g = 41; int main(){return g+1;}"), 42);
    assert_eq!(run_main("int g; int main(){return g;}"), 0);
}

#[test]
fn block_static_locals() {
    // Zero-init without `=`, and persistence across calls.
    assert_eq!(run_main("int f(){static int c = 0; c += 1; return c;} int main(){f(); return f();}"), 2);
    assert_eq!(run_main("int main(){static int z; return z;}"), 0);
    assert_eq!(run_main("int main(){static int n = 5; n += 1; return n;}"), 6);
    // Aggregates fold once; unsized arrays complete from the init.
    assert_eq!(run_main("int f(){static int a[3] = {1,2,3}; return a[1];} int main(){return f();}"), 2);
    assert_eq!(run_main("struct P{int x; int y;}; int main(){static struct P p = {1,2}; return p.x + p.y;}"), 3);
    assert_eq!(run_main("int main(){static char s[] = \"hi\"; return s[0] + s[1];}"), 209);
    // Addressable storage shared across calls.
    assert_eq!(run_main("int *f(){static int x = 7; return &x;} int main(){*f() = 10; return *f();}"), 10);
    // Same spelling in different scopes is separate storage.
    assert_eq!(
        run_main(
            "int f(){static int x = 1; x += 10; return x;} int g(){static int x = 2; x += 100; return x;}\
            int main(){int a = f(); int b = g(); return a + b;}"
        ),
        11 + 102
    );
}

#[test]
fn padded_struct_layout() {
    // Trailing padding counts in the size: member access, array
    // stride, assignment, and globals all agree with C layout.
    assert_eq!(run_main("struct P{char c; int x;}; int main(){struct P p; p.c = 1; p.x = 2; return p.c + p.x;}"), 3);
    assert_eq!(
        run_main(
            "struct P{char c; int x;}; int main(){struct P a[3]; a[0].x = 1; a[1].x = 10; a[2].x = 100;\
            a[0].c = 0; a[1].c = 0; a[2].c = 0; return a[0].x + a[1].x + a[2].x;}"
        ),
        111
    );
    assert_eq!(
        run_main("struct P{char c; int x;}; int main(){struct P a = {7, 8}; struct P b; b = a; return b.c + b.x;}"),
        15
    );
    assert_eq!(run_main("struct P{char c; int x;}; struct P g = {3, 4}; int main(){return g.c + g.x;}"), 7);
    assert_eq!(
        run_main(
            "struct R{char t[3]; int v;}; struct R rs[2] = {{\"ab\", 1}, {\"cd\", 2}};\
            int main(){return rs[0].v * 10 + rs[1].v + rs[1].t[0];}"
        ),
        10 + 2 + 99
    );
    assert_eq!(run_main("struct P{char c; int x;}; int main(){return (int)sizeof(struct P);}"), 8);
}

#[test]
fn subscript_rows_decay() {
    // `a[i]` of a 2D array is the row's address, not a load of
    // its first element — even though the checker pre-decays the
    // base (and the row) to pointers.
    assert_eq!(
        run_main(
            "int sum(const int *r){return r[0]+r[1]+r[2];}\
            int main(){int m[2][3] = {{1,2,3},{4,5,6}}; return sum(m[0]) + sum(m[1]);}"
        ),
        21
    );
    assert_eq!(
        run_main(
            "const char *pick(const char **t, int i){return t[i];}\
            int main(){const char *t[2] = {\"ab\", \"c\"}; return (int)(pick(t, 1)[0]);}"
        ),
        99
    );
    assert_eq!(
        run_main(
            "int first(int *p){return p[0];}\
            int main(){int a[4] = {7, 8, 9, 10}; return first(a + 2);}"
        ),
        9
    );
}

#[test]
fn unary_operands_evaluate_once() {
    // `&a[j++]` must step once with the pre-increment index: the
    // address operator used to evaluate its operand twice (once
    // as a value, once for the address), double-stepping.
    assert_eq!(
        run_main(
            "int main(){int a[4] = {10, 20, 30, 40}; int j = 0;\
            int *p = &a[j++]; return j * 100 + *p;}"
        ),
        110
    );
    assert_eq!(
        run_main(
            "int main(){int a[4] = {10, 20, 30, 40}; int j = 0;\
            int x = (*&a[j++]); return j * 100 + x;}"
        ),
        110
    );
    assert_eq!(
        run_main(
            "int main(){int a[4] = {10, 20, 30, 40}; int j = 0;\
            int x = *(a + (j++)); return j * 100 + x;}"
        ),
        110
    );
}

#[test]
fn null_pointer_globals() {
    // Every null spelling folds to zero bytes: plain `0`,
    // `(void *)0`, and inside lists, plus `_Bool` globals which
    // share the width computation.
    assert_eq!(run_main("static int *p = 0; int main(){return p != 0;}"), 0);
    assert_eq!(run_main("static void *q = (void *)0; int main(){return q != 0;}"), 0);
    assert_eq!(
        run_main(
            "static const char *cmd[3] = {\"a\", ((void *)0), \"b\"};\
            int main(){return (cmd[1] != 0) + (cmd[0][0] != 'a');}"
        ),
        0
    );
    assert_eq!(run_main("static _Bool t = 1; static _Bool f; int main(){return t + f * 2;}"), 1);
}

#[test]
fn atomic_load_store_rmw() {
    // Relaxed atomics behave like their plain counterparts here;
    // the narrowing and compare-exchange reporting matter. Orders
    // ride as raw integers (the harness lacks the macros).
    assert_eq!(
        run_main(
            "int main(){int x = 0; __atomic_store_n(&x, 41, 0);\
            return __atomic_load_n(&x, 0);}"
        ),
        41
    );
    assert_eq!(
        run_main(
            "int main(){int x = 7; int e = __atomic_exchange_n(&x, 3, 0);\
            return e * 100 + x;}"
        ),
        700 + 3
    );
    assert_eq!(
        run_main(
            "int main(){int x = 7; int old = __atomic_fetch_add(&x, 10, 0);\
            return old * 100 + x;}"
        ),
        700 + 17
    );
    assert_eq!(
        run_main(
            "int main(){int x = 17; int e = 17;\
            int ok = __atomic_compare_exchange_n(&x, &e, 99, 0, 0, 0);\
            return ok * 10000 + x * 100 + e;}"
        ),
        10000 + 9900 + 17
    );
    assert_eq!(
        run_main(
            "int main(){int x = 17; int e = 18;\
            int ok = __atomic_compare_exchange_n(&x, &e, 99, 0, 0, 0);\
            return ok * 10000 + x * 100 + e;}"
        ),
        1700 + 17
    );
}

#[test]
fn atomic_load_store_aggregate() {
    // Aggregates go through a memory copy; a scalar Clif load would
    // panic on the non-scalar result type.
    assert_eq!(
        run_main(
            "struct S{int a; int b;}; int main(){struct S s={3,4}; struct S r;\
            __atomic_store(&s, &s, 0); __atomic_load(&s, &r, 0);\
            return r.a * 10 + r.b;}"
        ),
        34
    );
}

#[test]
fn atomic_c11_builtins() {
    // The clang C11 spellings `<stdatomic.h>` expands to: the same
    // behavior as the GCC forms, minus the `_n` suffix and the
    // explicit weak flag.
    assert_eq!(
        run_main(
            "int main(){int x = 0; __c11_atomic_store(&x, 41, 5);\
            return __c11_atomic_load(&x, 5);}"
        ),
        41
    );
    assert_eq!(run_main("int main(){int x; __c11_atomic_init(&x, 7); return x;}"), 7);
    assert_eq!(
        run_main(
            "int main(){int x = 7; int e = __c11_atomic_exchange(&x, 3, 5);\
            return e * 100 + x;}"
        ),
        700 + 3
    );
    assert_eq!(
        run_main(
            "int main(){int x = 7; int old = __c11_atomic_fetch_add(&x, 10, 5);\
            return old * 100 + x;}"
        ),
        700 + 17
    );
    assert_eq!(
        run_main(
            "int main(){int x = 17; int e = 17;\
            int ok = __c11_atomic_compare_exchange_strong(&x, &e, 99, 5, 5);\
            return ok * 10000 + x * 100 + e;}"
        ),
        10000 + 9900 + 17
    );
    assert_eq!(
        run_main(
            "int main(){int x = 17; int e = 18;\
            int ok = __c11_atomic_compare_exchange_weak(&x, &e, 99, 5, 5);\
            return ok * 10000 + x * 100 + e;}"
        ),
        1700 + 17
    );
    assert_eq!(
        run_main(
            "int main(){int x = 5;\
            int a = __c11_atomic_fetch_max(&x, 9, 5); int b = __c11_atomic_fetch_min(&x, 2, 5);\
            return a * 100 + b;}"
        ),
        500 + 9
    );
    assert_eq!(
        run_main(
            "int main(){int x = 2;\
            int c = __c11_atomic_fetch_nand(&x, 1, 5); return c * 100 + (x == -1);}"
        ),
        200 + 1
    );
    assert_eq!(
        run_main(
            "int main(){int x = 0; __c11_atomic_thread_fence(5); __c11_atomic_signal_fence(5);\
            return __c11_atomic_is_lock_free(sizeof(int)) + (x == 0);}"
        ),
        2
    );
}

#[test]
fn interlocked_compare_exchange_returns_initial_value() {
    // MSVC `<intrin.h>` atomics lower to the same compare-exchange,
    // differing only in taking the comparand by value and returning
    // the object's initial value (not a success flag).
    assert_eq!(
        run_main(
            "int _InterlockedCompareExchange(int volatile *, int, int);\
             int main(){int x = 5; int r = _InterlockedCompareExchange(&x, 9, 5);\
             return (r == 5) + (x == 9) * 2;}"
        ),
        1 + 2
    );
    assert_eq!(
        run_main(
            "int _InterlockedCompareExchange(int volatile *, int, int);\
             int main(){int y = 1; int r = _InterlockedCompareExchange(&y, 7, 0);\
             return (r == 1) + (y == 1) * 2;}"
        ),
        1 + 2
    );
}

#[test]
fn atomic_min_max_signedness() {
    // `min`/`max` compare by the target's signedness (the MIR
    // `Smin`/`Umax` split), and the `min_fetch`/`max_fetch` forms
    // compute the new value with a select.
    assert_eq!(
        run_main(
            "int main(){int x = -5;\
            int a = __atomic_fetch_max(&x, -9, 5); int b = __atomic_fetch_min(&x, -9, 5);\
            return (a == -5) + (x == -9) * 2;}"
        ),
        1 + 2
    );
    assert_eq!(
        run_main(
            "int main(){unsigned u = 3u;\
            unsigned c = __atomic_max_fetch(&u, 10u, 5);\
            return (c == 10) + (u == 10) * 2;}"
        ),
        1 + 2
    );
    assert_eq!(
        run_main(
            "int main(){int x = -5;\
            int r = __atomic_min_fetch(&x, -9, 5);\
            return (r == -9) + (x == -9) * 2;}"
        ),
        1 + 2
    );
}

#[test]
fn struct_pointer_arithmetic() {
    // Steps over struct pointees scale by the frontend element
    // size (the MIR pointee is erased): `p++`, `p + i`, `p - i`,
    // and `p[i]` all agree.
    assert_eq!(
        run_main(
            "struct P{char c; int x;}; int main(){struct P a[3]; struct P *p = a;\
            p->x = 1; p++; p->x = 10; p += 1; p->x = 100; p -= 2;\
            return a[0].x + a[1].x + a[2].x + (p == a); }"
        ),
        111 + 1
    );
    assert_eq!(
        run_main(
            "struct P{char c; int x;}; int get(struct P *p, int i){return (p + i)->x;}\
            int main(){struct P a[2]; a[0].x = 5; a[1].x = 6; return get(a, 0) + get(a, 1);}"
        ),
        11
    );
    assert_eq!(
        run_main(
            "struct P{char c; int x;}; int main(){struct P a[3]; struct P *p = a + 2;\
            return (p - a) * 100 + (p - 2 == a);}"
        ),
        200 + 1
    );
}

#[test]
fn array_row_pointer_arithmetic() {
    // Rows of `int m[2][3]` step by sizeof(int[3]) = 12, so the end
    // pointer `m + 2` is begin + 24. A step mis-sized by the inner
    // element size (+8) walks the wrong rows.
    assert_eq!(
        run_main(
            "int main(){int m[2][3] = {{1,2,3},{4,5,6}}; int sum = 0;\
            for (int (*p)[3] = m; p != m + 2; ++p)\
                for (int j = 0; j < 3; j++) sum += (*p)[j];\
            return sum;}"
        ),
        21
    );
    // Row indexing through a `T(*)[N]` pointer, forwards and
    // backwards from the last row.
    assert_eq!(
        run_main("int main(){int m[2][3] = {{1,0,0},{0,5,0}}; int (*p)[3] = m + 1; return p[0][1] + (p - 1)[0][0];}"),
        6
    );
    // Pointer difference over row pointers counts rows, not bytes.
    assert_eq!(run_main("int main(){int m[2][3]; int (*p)[3] = m; return (m + 2) - p;}"), 2);
    // A 3-D traversal multiplies the strides per dimension.
    assert_eq!(
        run_main(
            "int main(){int c[2][2][2] = {{{1,2},{3,4}},{{5,6},{7,8}}}; int sum = 0;\
            for (int i = 0; i < 2; i++)\
                for (int (*p)[2] = c[i]; p != c[i] + 2; ++p)\
                    for (int j = 0; j < 2; j++) sum += (*p)[j];\
            return sum;}"
        ),
        36
    );
    // Row storage behind a pointer-valued base the declarations
    // cannot name: the subscripted row decays to its address, it
    // is never loaded as a pointer.
    assert_eq!(
        run_main("int main(){int m[2][3] = {{1,2,3},{4,5,6}}; int *flat = &m[0][0]; return ((int (*)[3])flat)[1][2];}"),
        6
    );
}

#[test]
fn va_list_forwarding() {
    // `va_arg` through a `va_list` parameter reads the caller's
    // object: parameters alias it rather than spilling the pointer
    // into a slot (which reads as cursor garbage).
    assert_eq!(
        run_main(
            "int fwd(int n, __builtin_va_list ap){int s = 0;\
            for (int i = 0; i < n; i++) s += __builtin_va_arg(ap, int); return s;}\
            int sum(int n, ...){__builtin_va_list ap; __builtin_va_start(ap, n);\
            int s = fwd(n, ap); __builtin_va_end(ap); return s;}\
            int main(){return sum(4, 10, 20, 30, 40);}"
        ),
        100
    );
}

#[test]
fn c23_va_start_takes_one_argument() {
    // C23's `va_start` spelling passes only the list; clang's header
    // expands it to `__builtin_c23_va_start(ap)`.
    assert_eq!(
        run_main(
            "int sum(int n, ...){__builtin_va_list ap; __builtin_c23_va_start(ap);\
            int s = 0; for (int i = 0; i < n; i++) s += __builtin_va_arg(ap, int);\
            __builtin_va_end(ap); return s;}\
            int main(){return sum(3, 4, 5, 6);}"
        ),
        15
    );
}

#[test]
fn unaligned_qualifier_lowers_away() {
    // `__unaligned` is accepted in MS mode and has no codegen
    // effect: accesses are already emitted unaligned-safe.
    assert_eq!(
        jit_run_main(lower_module_ms(
            "int main(){ unsigned short x = 7;\
             unsigned short __unaligned *p = (unsigned short __unaligned *)&x;\
             return *p; }"
        )),
        7
    );
}

#[test]
fn va_arg_reads_overflow_aggregates_whole() {
    // A 9-byte struct spans two eightbytes, so the third struct
    // vararg (after two register pairs) cannot fit its registers
    // and the SysV ABI sends the whole argument to the overflow
    // area. Reading piecewise would take the first eightbyte from
    // the save area and the second from the stack.
    assert_eq!(
        run_main(
            "struct S{char x[9];}; int f(int n, ...){__builtin_va_list ap;\
            __builtin_va_start(ap, n); int s = 0;\
            for (int i = 0; i < n; i++){struct S t = __builtin_va_arg(ap, struct S);\
            s = s * 3 + t.x[0] + t.x[8];} __builtin_va_end(ap); return s;}\
            int main(){struct S a = {\"ABCDEFGHI\"}; return f(6, a, a, a, a, a, a);}"
        ),
        50232
    );
    // Same rule in the vector file: five three-float structs need
    // ten XMM slots, so the fifth crosses wholly on the stack.
    assert_eq!(
        run_main(
            "struct F{float a, b, c;}; int g(int n, ...){__builtin_va_list ap;\
            __builtin_va_start(ap, n); int s = 0;\
            for (int i = 0; i < n; i++){struct F t = __builtin_va_arg(ap, struct F);\
            s = s * 2 + (int)t.a + (int)t.c;} __builtin_va_end(ap); return s;}\
            int main(){struct F f = {13.1f, 13.2f, 13.3f}; return g(5, f, f, f, f, f);}"
        ),
        806
    );
}

#[test]
fn scalar_builtins_lower_inline() {
    // Single-instruction builtins have no library symbol: they
    // lower inline instead of calling out.
    assert_eq!(run_main("int main(){return __builtin_clzll(0x1000000000000000ULL);}"), 3);
    assert_eq!(run_main("int main(){return __builtin_ctz(8) + __builtin_popcount(7);}"), 6);
    assert_eq!(run_main("int main(){return (int)__builtin_ceil(1.5) * 10 + (int)__builtin_floor(1.5);}"), 21);
    assert_eq!(run_main("int main(){return (__builtin_bswap32(0x12345678U) == 0x78563412U);}"), 1);
    // MSVC's `<intrin.h>` names are the same instructions, so they
    // lower inline too rather than needing a library symbol.
    let msvc = "unsigned short _byteswap_ushort(unsigned short);\
                unsigned int _byteswap_ulong(unsigned int);\
                unsigned long long _byteswap_uint64(unsigned long long);\
                void _ReadWriteBarrier(void);\
                int main(){ _ReadWriteBarrier();\
                  return (_byteswap_ushort(0x1234) == 0x3412)\
                       + (_byteswap_ulong(0x12345678U) == 0x78563412U)\
                       + (_byteswap_uint64(0x123456789ABCDEF0ULL) == 0xF0DEBC9A78563412ULL); }";
    assert_eq!(jit_run_main(lower_module_ms(msvc)), 3);
}

#[test]
fn type_generic_bit_counts_count_at_operand_width() {
    // `__builtin_clzg`/`__builtin_ctzg`: the operand keeps its own
    // width (no promotion), the count converts to `int`, and a zero
    // operand yields the fallback when one is given.
    assert_eq!(run_main("int main(){return __builtin_clzg((unsigned char)0x10);}"), 3);
    assert_eq!(run_main("int main(){return __builtin_clzg(0x10ULL);}"), 59);
    assert_eq!(run_main("int main(){return __builtin_ctzg((unsigned short)0x100);}"), 8);
    assert_eq!(run_main("int main(){return __builtin_clzg(1U, 42);}"), 31);
    assert_eq!(run_main("int main(){return __builtin_clzg(0U, 42);}"), 42);
    assert_eq!(run_main("int main(){return __builtin_ctzg(0x80U, 42);}"), 7);
    assert_eq!(run_main("int main(){return __builtin_ctzg(0U, 42);}"), 42);
    // A `_Bool` operand counts one bit: its nonzero value has no
    // leading or trailing zeros, and zero counts the fallback (the
    // byte-wide count would say eight).
    assert_eq!(run_main("int main(){return __builtin_clzg((_Bool)1);}"), 0);
    assert_eq!(run_main("int main(){return __builtin_ctzg((_Bool)0, 42);}"), 42);
    // 128-bit operands keep their width too.
    assert_eq!(run_main("int main(){return __builtin_clzg((unsigned __int128)1);}"), 127);
    assert_eq!(
        jit_run_main(lower_module_c23("int main(void){ unsigned _BitInt(40) x = 1uwb; return __builtin_clzg(x); }")),
        39
    );
}

#[test]
fn stdc_rotate_builtins_rotate_at_operand_width() {
    assert_eq!(run_main("int main(){return __builtin_stdc_rotate_left(0x12345678U, 4) == 0x23456781U;}"), 1);
    assert_eq!(run_main("int main(){return __builtin_stdc_rotate_right(0x12345678U, 4) == 0x81234567U;}"), 1);
    // Counts reduce modulo the width, including a count equal to
    // the width and one larger than a byte-wide operand.
    assert_eq!(run_main("int main(){return __builtin_stdc_rotate_left(0x12345678U, 32) == 0x12345678U;}"), 1);
    assert_eq!(run_main("int main(){return __builtin_stdc_rotate_left((unsigned char)0x81, 4) == 0x18;}"), 1);
    assert_eq!(run_main("int main(){return __builtin_stdc_rotate_right((unsigned short)0x8001, 1) == 0xC000;}"), 1);
    assert_eq!(
        run_main("int main(){return __builtin_stdc_rotate_left(0x0123456789ABCDEFULL, 8) == 0x23456789ABCDEF01ULL;}"),
        1
    );
    // A `_BitInt(N)` rotates over its value precision, not its
    // padded storage width.
    assert_eq!(
        jit_run_main(lower_module_c2y(
            "int main(void){ unsigned _BitInt(7) x = 1; \
             return (__builtin_stdc_rotate_left(x, 1) == 2) \
                  + (__builtin_stdc_rotate_left(x, 260) == 2) * 2; }"
        )),
        3
    );
    // A wide count reduces modulo the precision before shifting,
    // rather than truncating to `unsigned int` first (2^32 mod 7
    // is 4, so rotating 1 left by four is 16, not 1).
    assert_eq!(
        jit_run_main(lower_module_c2y(
            "int main(void){ unsigned _BitInt(7) x = 1; \
             unsigned long long n = 0x100000000ULL; \
             return (__builtin_stdc_rotate_left(x, n) == 16) \
                  + (__builtin_stdc_rotate_left(x, n + 3) == 1) * 2; }"
        )),
        3
    );
    // A 128-bit count has no native remainder lowering; the two
    // 64-bit halves still reduce exactly ((2^64 + 4) mod 7 is 6).
    assert_eq!(
        jit_run_main(lower_module_c2y(
            "int main(void){ unsigned _BitInt(7) x = 1; \
             unsigned __int128 n = ((unsigned __int128)1 << 64) + 4; \
             return (__builtin_stdc_rotate_left(x, n) == 64) \
                  + (__builtin_stdc_rotate_left(x, n + 1) == 1) * 2; }"
        )),
        3
    );
    // A bit-field rotates over its declared width, and its result
    // keeps the declared unsigned type (not the promoted `int`).
    assert_eq!(
        run_main(
            "struct S { unsigned s : 5; }; \
             int main(void){ struct S s = { 0x12 }; int n = 1; \
             return (__builtin_stdc_rotate_left(s.s, 3) == 0x14) \
                  + (__builtin_stdc_rotate_right(s.s, 2) == 0x14) * 2 \
                  + (__builtin_stdc_rotate_right(s.s, n) == 0x9) * 4; }"
        ),
        7
    );
}

#[test]
fn fp_classifications_lower_to_comparisons() {
    // `isinf` is 1/0 like GCC; `isinf_sign` distinguishes the sign
    // (what `<math.h>`'s `isinf` macro expands to).
    assert_eq!(
        run_main("int main(){double p=1.0/0.0, n=-1.0/0.0; return __builtin_isinf(p)*100 + __builtin_isinf(n)*10 + __builtin_isinf(2.0);}"),
        110
    );
    assert_eq!(
        run_main("int main(){double p=1.0/0.0, n=-1.0/0.0; return __builtin_isinf_sign(p)*100 + __builtin_isinf_sign(n)*10 + __builtin_isinf_sign(2.0);}"),
        90
    );
    // `isfinite` rejects NaN and the infinities; `isnormal` also
    // rejects zero and subnormals.
    assert_eq!(
        run_main("int main(){double p=1.0/0.0; return __builtin_isfinite(1.5) + __builtin_isfinite(0.0)*10 + __builtin_isfinite(p)*100 + __builtin_isfinite(p-p)*1000;}"),
        11
    );
    assert_eq!(
        run_main("int main(){double p=1.0/0.0; return __builtin_isnormal(1.5) + __builtin_isnormal(5e-324)*10 + __builtin_isnormal(0.0)*100 + __builtin_isnormal(p)*1000;}"),
        1
    );
    // `fpclassify` picks the code for the category; the codes here
    // are glibc's `FP_NAN`...`FP_ZERO` (0, 1, 4, 3, 2).
    assert_eq!(
        run_main(
            "int main(){double p=1.0/0.0; return __builtin_fpclassify(0,1,4,3,2,0.0/0.0) + __builtin_fpclassify(0,1,4,3,2,p)*10 \
             + __builtin_fpclassify(0,1,4,3,2,1.5)*100 + __builtin_fpclassify(0,1,4,3,2,5e-324)*1000 + __builtin_fpclassify(0,1,4,3,2,0.0)*10000;}"
        ),
        23410
    );
    // The float forms use the same tests at 32-bit width.
    assert_eq!(
        run_main(
            "int main(){float p=1.0f/0.0f; return __builtin_isinf_sign(p)*10 + __builtin_isfinite(1.5f) \
             + __builtin_isnormal(0.0f)*100 + __builtin_fpclassify(0,1,4,3,2,5e-45f)*1000;}"
        ),
        3011
    );
    // 80-bit long doubles classify through the x87 comparisons.
    lower(
        "int f(long double x){ return __builtin_isinf(x) + __builtin_isinf_sign(x) + __builtin_isfinite(x) \
         + __builtin_isnormal(x) + __builtin_fpclassify(0,1,4,3,2,x); }",
    );
}

#[test]
fn ternary_arms_convert_to_result() {
    // `cond ? ptr : 0` must store a full null pointer, not a
    // 4-byte zero over stack garbage.
    assert_eq!(run_main("int x = 5; int main(){int *p = x ? &x : 0; return *p;}"), 5);
    assert_eq!(run_main("int x = 0; int main(){int *p = x ? &x : 0; return p == 0;}"), 1);
    assert_eq!(run_main("int main(){int a = 1, b = 2; int c = 0; int r = c ? a : b; return r;}"), 2);
    // A null-constant `(void *)0` arm keeps the other arm's
    // pointer type, so member access survives (tcctest.c shape).
    assert_eq!(
        run_main("struct S{int i;}; int f(int i, struct S *s){return (i == 0 ? (void *)0 : s)->i;} int main(){struct S s; s.i = 7; return f(1, &s);}"),
        7
    );
}

#[test]
fn inline_flag_reaches_mir() {
    // The C `inline` hint rides the binding into MIR so the
    // `inline` pass can spend a larger budget on it.
    for (src, name, expect_inline) in [
        ("inline int f(void){return 1;} int main(){return f();}", "f", true),
        ("int f(void){return 1;} int main(){return f();}", "f", false),
        ("static __inline__ int f(void){return 1;} int main(){return f();}", "f", true),
    ] {
        let mut interner = fe::lexer::Interner::new();
        let mut sema = fe::sema::Sema::leak_for_tests();
        let tokens = fe::lexer::Lexer::new(src, &mut interner).lex_all();
        let mut parser = fe::parse::Parser::new(src, tokens, sema.arena, &mut sema, &mut interner);
        let tu = parser.parse_translation_unit();
        drop(parser);
        let module = lower_translation_unit(&sema, &interner, &tu);
        let func = module.funcs.iter().find(|f| f.name == name).expect("function lowered");
        assert_eq!(func.inline, expect_inline, "{src}");
        assert!(mir::verify::verify_module(&module).is_empty(), "{src}");
    }
}

#[test]
fn inline_end_to_end_through_jit() {
    // `inline` callees (including multi-block bodies) execute
    // correctly, whether or not the pass inlines them here.
    assert_eq!(run_main("inline int add1(int x){return x+1;} int main(){return add1(41);}"), 42);
    assert_eq!(run_main("inline int pick(int x){if(x>0) return 1; return 0;} int main(){return pick(5)*10+pick(-3);}"), 10);
}

#[test]
fn opt_pipeline_promotes_spills() {
    // The driver pipeline (inline, mem2reg, fold, simplify, DCE)
    // removes the -O0 spill slots from a trivial function while
    // keeping verification clean.
    use mir::pass::{FunctionPass, ModulePass};
    for src in [
        "int f(int x){return x+1;} int main(){return f(41);}",
        "inline int pick(int x){if(x>0) return 1; return 0;} int main(){return pick(5);}",
        "volatile int v; int main(){v = 1; return v;}",
    ] {
        let mut interner = fe::lexer::Interner::new();
        let mut sema = fe::sema::Sema::leak_for_tests();
        let tokens = fe::lexer::Lexer::new(src, &mut interner).lex_all();
        let mut parser = fe::parse::Parser::new(src, tokens, sema.arena, &mut sema, &mut interner);
        let tu = parser.parse_translation_unit();
        drop(parser);
        let mut module = lower_translation_unit(&sema, &interner, &tu);
        let mut inline = mir::passes::Inline::small();
        inline.run(&mut module);
        for func in module.funcs.iter_mut() {
            for _ in 0..16 {
                let mut changed = false;
                changed |= mir::passes::Sroa.run(func);
                changed |= mir::passes::Mem2Reg.run(func);
                changed |= mir::passes::ConstFold.run(func);
                changed |= mir::passes::InstCombine.run(func);
                changed |= mir::passes::Cse.run(func);
                changed |= mir::passes::SimplifyCfg.run(func);
                changed |= mir::passes::DeadCodeElim.run(func);
                if !changed {
                    break;
                }
            }
        }
        assert!(mir::verify::verify_module(&module).is_empty(), "verify errors in {src}");
    }
    // The non-volatile cases leave no spill traffic behind.
    let src = "int f(int x){int y = x; return y+1;} int main(){return f(41);}";
    let mut interner = fe::lexer::Interner::new();
    let mut sema = fe::sema::Sema::leak_for_tests();
    let tokens = fe::lexer::Lexer::new(src, &mut interner).lex_all();
    let mut parser = fe::parse::Parser::new(src, tokens, sema.arena, &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    drop(parser);
    let mut module = lower_translation_unit(&sema, &interner, &tu);
    run_opt_pipeline(&mut module);
    assert!(mir::verify::verify_module(&module).is_empty());
    let f = module.funcs.iter().find(|f| f.name == "f").expect("f lowered");
    assert!(
        !f.blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .any(|i| matches!(f.insts[*i].kind, mir::InstKind::Alloca { .. } | mir::InstKind::Load { .. } | mir::InstKind::Store { .. })),
        "spills survived in f"
    );
    // ...while a volatile global access keeps its memory traffic.
    let src = "volatile int v; int main(){v = 1; return v;}";
    let mut interner = fe::lexer::Interner::new();
    let mut sema = fe::sema::Sema::leak_for_tests();
    let tokens = fe::lexer::Lexer::new(src, &mut interner).lex_all();
    let mut parser = fe::parse::Parser::new(src, tokens, sema.arena, &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    drop(parser);
    let mut module = lower_translation_unit(&sema, &interner, &tu);
    run_opt_pipeline(&mut module);
    assert!(mir::verify::verify_module(&module).is_empty());
    let main = module.funcs.iter().find(|f| f.name == "main").expect("main lowered");
    assert!(
        main.blocks.iter().flat_map(|b| b.insts.iter()).any(|i| matches!(
            main.insts[*i].kind,
            mir::InstKind::Load { flags, .. } | mir::InstKind::Store { flags, .. } if flags.volatile
        )),
        "volatile traffic was optimized out"
    );
}

/// The checker's value conversion strips top-level qualifiers, so a
/// volatile read through `*p`, `a[i]`, `s->x`, or a parameter
/// reaches lowering typed as a plain value; lowering must resolve
/// the designated object and keep the `.vol` flag. Without it the
/// read is a plain load that Cranelift forwards a dominating store
/// into.
#[test]
fn volatile_lvalue_reads_keep_their_flag() {
    let src = "struct S { int x; };\n\
               struct B { int x : 3; };\n\
               int param(volatile int n) { return n + 2; }\n\
               int deref(volatile int *p) { return *p + 2; }\n\
               int sub(volatile int *a) { return a[0] + 2; }\n\
               int member(volatile struct S *s) { return s->x + 2; }\n\
               int dot(void) { volatile struct S s; s.x = 1; return s.x + 2; }\n\
               int bitfield(volatile struct B *b) { b->x = 3; return b->x + 2; }\n\
               int local(void) { volatile int v = 1; return v + 2; }\n\
               void store(volatile int *p, int v) { *p = v; }\n\
               int plain(int n) { return n + 2; }\n\
               int main(void) { return local() + dot(); }";
    let volatile_accesses = |module: &mir::Module, name: &str| -> (usize, usize) {
        let f = module.funcs.iter().find(|f| f.name == name).unwrap_or_else(|| panic!("{name} lowered"));
        let mut loads = 0;
        let mut stores = 0;
        for inst in f.blocks.iter().flat_map(|b| b.insts.iter()) {
            match &f.insts[*inst].kind {
                mir::InstKind::Load { flags, .. } if flags.volatile => loads += 1,
                mir::InstKind::Store { flags, .. } if flags.volatile => stores += 1,
                _ => {}
            }
        }
        (loads, stores)
    };
    let forms = [
        ("param", true, true),
        ("deref", true, false),
        ("sub", true, false),
        ("member", true, false),
        ("dot", true, true),
        ("bitfield", true, true),
        ("local", true, true),
        ("store", false, true),
    ];
    let mut module = lower_module(src);
    for (name, want_load, want_store) in forms {
        let (loads, stores) = volatile_accesses(&module, name);
        assert_eq!(loads > 0, want_load, "volatile loads in {name} before optimization: {loads}");
        assert_eq!(stores > 0, want_store, "volatile stores in {name} before optimization: {stores}");
    }
    assert_eq!(volatile_accesses(&module, "plain"), (0, 0), "plain parameter marked volatile");
    run_opt_pipeline(&mut module);
    assert!(mir::verify::verify_module(&module).is_empty());
    for (name, want_load, want_store) in forms {
        let (loads, stores) = volatile_accesses(&module, name);
        assert_eq!(loads > 0, want_load, "volatile loads in {name} after optimization: {loads}");
        assert_eq!(stores > 0, want_store, "volatile stores in {name} after optimization: {stores}");
    }
    assert_eq!(volatile_accesses(&module, "plain"), (0, 0), "plain parameter marked volatile after optimization");
    assert_eq!(jit_run_main(module), 6);
}

#[test]
fn negative_subscript_indices() {
    // C indices are signed: `p[-1]` walks backwards ( lowered
    // with a sign-extending step, not zero-extending).
    assert_eq!(
        run_main(
            "int main(){char s[] = \"abc\"; char *p = s + 2;\
            return (p[0] == 'c') + (p[-1] == 'b') + (p[-2] == 'a');}"
        ),
        3
    );
    assert_eq!(
        run_main(
            "int main(){int a[4] = {10, 20, 30, 40}; int *q = a + 3;\
            return (q[-3] == 10) + (q[0] == 40);}"
        ),
        2
    );
}

#[test]
fn flex_array_member_subscript() {
    // Flexible array members have no MIR array form: indexing
    // must stride by the frontend element size, not pointer size.
    assert_eq!(
        run_main(
            "struct F{int n; unsigned vals[];};\
            int at(struct F *f, int i){return (int)f->vals[i];}\
            int main(){static unsigned backing[8] = {0, 11, 22, 33};\
            struct F *f = (struct F *)backing;\
            f->n = 3; return at(f, 0) + at(f, 1) + at(f, 2);}"
        ),
        66
    );
}

#[test]
fn assignment_forms() {
    assert_eq!(run_main("int main(){int x = 1; x += 5; x = x * 2; return x;}"), 12);
    assert_eq!(run_main("int main(){int x = 10; x -= 3; x /= 7; return x;}"), 1);
    assert_eq!(run_main("int g = 0; int main(){g = 9; return g;}"), 9);
}

#[test]
fn asm_empty_templates() {
    // Matching constraint: the input ties to the output's location.
    let (module, _) = lower_with_asm("int main(){int x=7; int y; __asm__(\"\" : \"=r\"(y) : \"0\"(x)); return y;}");
    let main = module.funcs.iter().find(|f| f.name == "main").expect("main");
    let asm = main
        .insts
        .iter()
        .find_map(|inst| match &inst.kind {
            mir::InstKind::InlineAsm { block } => Some(block.clone()),
            _ => None,
        })
        .expect("inline asm instruction");
    assert_eq!(asm.template, "");
    assert_eq!(asm.outputs, 1);
    assert_eq!(asm.operands.len(), 2);
    assert_eq!(asm.operands[0].constraint, "=r");
    assert_eq!(asm.operands[1].constraint, "0");
    assert_eq!(asm.operands[1].tie, Some(0));
    // Inputs evaluate in order before the block: `x=3` still runs.
    let (module, _) = lower_with_asm("int main(){int x=0; asm(\"\" : : \"r\"(x=3)); return x;}");
    let main = module.funcs.iter().find(|f| f.name == "main").expect("main");
    assert!(main.insts.iter().any(|inst| matches!(inst.kind, mir::InstKind::InlineAsm { .. })));
}

#[test]
fn asm_immediate_operands_fold_constants() {
    // `"i"` takes any integer constant expression, not just a
    // literal: `-5` must reach the wrapper as a constant.
    let (module, _) = lower_with_asm("int main(){int y; asm(\"movl %1, %0\" : \"=r\"(y) : \"i\"(-5)); return y;}");
    let main = module.funcs.iter().find(|f| f.name == "main").expect("main");
    let block = main
        .insts
        .iter()
        .find_map(|inst| match &inst.kind {
            mir::InstKind::InlineAsm { block } => Some(block.clone()),
            _ => None,
        })
        .expect("inline asm instruction");
    assert_eq!(main.const_value(block.operands[1].value), Some(&mir::Const::Int { bits: 32, value: -5 }));
}

#[test]
fn asm_immediate_or_register_alternatives() {
    // `"Ir"` folds a constant (the wrapper substitutes it) but
    // materializes a runtime value for the register alternative.
    fn operand_value(src: &str) -> Option<mir::Const> {
        let (module, _) = lower_with_asm(src);
        let main = module.funcs.iter().find(|f| f.name == "main").expect("main");
        let block = main
            .insts
            .iter()
            .find_map(|inst| match &inst.kind {
                mir::InstKind::InlineAsm { block } => Some(block.clone()),
                _ => None,
            })
            .expect("inline asm instruction");
        main.const_value(block.operands[1].value).cloned()
    }
    assert_eq!(
        operand_value("int main(){int y; asm(\"add %1, %0\" : \"+r\"(y) : \"Ir\"(5)); return y;}"),
        Some(mir::Const::Int { bits: 32, value: 5 })
    );
    assert_eq!(operand_value("int main(){int y, x; asm(\"add %1, %0\" : \"+r\"(y) : \"Ir\"(x)); return y;}"), None);
}

#[test]
fn asm_goto_dispatches_on_the_selector() {
    let (module, _) = lower_with_asm("int main(){int x=0; asm goto(\"testl %0, %0; jnz %l[lab]\" : : \"r\"(x) : : lab); return 1; lab: return 2;}");
    let main = module.funcs.iter().find(|f| f.name == "main").expect("main");
    let asm = main
        .insts
        .iter()
        .find_map(|inst| match &inst.kind {
            mir::InstKind::InlineAsm { block } => Some((inst, block.clone())),
            _ => None,
        })
        .expect("inline asm instruction");
    assert_eq!(asm.1.labels, vec!["lab".to_string()]);
    let selector = asm.0.result.expect("asm goto yields a selector");
    let switch = main
        .blocks
        .iter()
        .find_map(|block| match &block.term {
            Some(mir::Terminator::Switch { scrutinee, cases, default }) if *scrutinee == selector => Some((cases.clone(), *default)),
            _ => None,
        })
        .expect("selector switch");
    assert_eq!(switch.0.len(), 1);
    assert_eq!(switch.0[0].0, 1);
    // The default is the fallthrough, the case is the `lab` block.
    let returns = |block: mir::Block, value: i128| {
        matches!(
            main.blocks[block].term,
            Some(mir::Terminator::Return { val: Some(v) })
                if main.const_value(v) == Some(&mir::Const::Int { bits: 32, value })
        )
    };
    assert!(returns(switch.1, 1), "fallthrough returns 1");
    assert!(returns(switch.0[0].1, 2), "the label block returns 2");
}

#[test]
fn calls_and_recursion() {
    assert_eq!(run_main("int sq(int v){return v*v;} int main(){return sq(7);}"), 49);
    assert_eq!(run_main("int fib(int n){if(n<2) return n; return fib(n-1)+fib(n-2);} int main(){return fib(10);}"), 55);
}

#[test]
fn comparisons_and_logic() {
    assert_eq!(run_main("int main(){return (3<5)+(5<3)+(4==4);}"), 2);
    assert_eq!(run_main("int main(){return (7>=7)+(6!=6)+(2>3);}"), 1);
    // Short-circuit must not evaluate the trapping side.
    assert_eq!(run_main("int main(){return (0 && (1/0));}"), 0);
    assert_eq!(run_main("int main(){return (1 || (1/0));}"), 1);
    assert_eq!(run_main("int main(){int x = 5; return (x>3) ? 10 : 20;}"), 10);
    assert_eq!(run_main("int main(){int x = 1; return (x>3) ? 10 : 20;}"), 20);
}

#[test]
fn loops() {
    assert_eq!(run_main("int main(){int s = 0; int i = 1; while(i<=100){s += i; i++;} return s;}"), 5050);
    assert_eq!(run_main("int main(){int s = 0; int i = 1; do { s += i; i++; } while(i<=10); return s;}"), 55);
    assert_eq!(run_main("int main(){int s = 0; for(int i = 1; i<=10; i++){s += i;} return s;}"), 55);
    assert_eq!(
        run_main("int main(){int s = 0; for(int i = 0; i<10; i++){if(i%2==0) continue; if(i>7) break; s += i;} return s;}"),
        16
    );
}

#[test]
fn wide_conditions_test_full_width() {
    // Conditions test the whole value: narrowing to I32 dropped
    // the high half, so `while (1ULL << 32)` never ran.
    assert_eq!(run_main("int main(){unsigned long long v = 1ULL << 32; int n = 0; while(v){n++; v >>= 33;} return n;}"), 1);
    assert_eq!(run_main("int main(){long long v = -9223372036854775807LL - 1; if(v){return 1;} return 0;}"), 1);
    assert_eq!(run_main("int main(){unsigned long long v = 1ULL << 32; return v ? 7 : 8;}"), 7);
    assert_eq!(run_main("int main(){unsigned long long v = 0; return v ? 7 : 8;}"), 8);
}

#[test]
fn address_and_deref() {
    assert_eq!(run_main("int main(){int x = 5; int *p = &x; *p += 2; return x;}"), 7);
    assert_eq!(run_main("int main(){int x = 5; int *p = &x; int y = *p; return y;}"), 5);
}

#[test]
fn pointer_arithmetic() {
    assert_eq!(
        run_main("int sum(int *p, int n){int s = 0; for(int i = 0; i<n; i++) s += p[i]; return s;} int a[4] = {1,2,3,4}; int main(){return sum(a, 4);}"),
        10
    );
    assert_eq!(run_main("int main(){int a[4]; int *p = &a[0]; int *q = &a[3]; return q-p;}"), 3);
    assert_eq!(run_main("int main(){int a[3] = {10,20,30}; int *p = a+1; return *p + *(p+1);}"), 50);
    // Subtracting a narrow unsigned index must widen it with its
    // own signedness *before* negating: negating first and then
    // zero-extending the `i32` bit pattern adds 2^32 to the
    // address. SQLite's `fts3MIBufferFree()` recomputes a base
    // pointer with `p - (u32)`, so zero-extension there crashes.
    let high = "char *p = (char*)0x211E22B8B5Cull;";
    assert_eq!(
        run_main(&format!(
            "int main(){{{high} unsigned int u = 28; char *q = p - u; return q == (char*)0x211E22B8B40ull ? 0 : 1;}}"
        )),
        0
    );
    assert_eq!(
        run_main(&format!("int main(){{{high} unsigned int u = 28; p -= u; return p == (char*)0x211E22B8B40ull ? 0 : 1;}}")),
        0
    );
}

#[test]
fn strings() {
    assert_eq!(
        run_main("int slen(char *s){int n = 0; while(*s){n++; s++;} return n;} int main(){return slen(\"hello\");}"),
        5
    );
    assert_eq!(run_main("int main(){char s[] = \"abc\"; return s[0]+s[1]+s[2];}"), 294);
    assert_eq!(run_main("char g[] = \"hi\"; int main(){return g[0]+g[1];}"), 209);
}

#[test]
fn string_literals_are_read_only() {
    let module = lower_module("int main(void) { char *p = \"hi\"; char *e = \"\"; return p[0] + e[0]; }");
    let literals: Vec<_> = module.globals.iter().filter(|g| g.name.starts_with("__str.")).collect();
    assert_eq!(literals.len(), 2);
    assert!(literals.iter().all(|g| !g.writable));
    assert!(
        literals.iter().any(|g| matches!(&g.init, Some(mir::Const::Bytes(bytes)) if bytes == b"hi\0")),
        "the literal keeps its bytes"
    );
    assert!(
        literals.iter().any(|g| matches!(&g.init, Some(mir::Const::Bytes(bytes)) if bytes == b"\0")),
        "the empty literal keeps its terminator"
    );
}

#[test]
fn c2y_static_assert_expressions_lower() {
    assert_eq!(
        jit_run_main(lower_module_c2y("int main(void) { int x = (static_assert(1), 42); static_assert(1); return x; }")),
        42
    );
    assert_eq!(jit_run_main(lower_module_c2y("int main(void) { return (static_assert(1), _Static_assert(1), 7); }")), 7);
}

#[test]
fn c2y_bit_precise_enums_lower() {
    let src = "enum E : _BitInt(5) { E_MAX = 15, E_MIN = -16 };\n\
               enum U : unsigned _BitInt(5) { U_MAX = 31 };\n\
               int main(void) {\n\
                   enum E e = E_MAX; e = e + 1;\n\
                   enum U u = U_MAX; u = u + 1;\n\
                   return (e == -16) + 2 * (u == 0) + 4 * (sizeof(e) == 1)\n\
                       + 8 * _Generic((enum E)0, enum E: 1, default: 0);\n\
               }";
    assert_eq!(jit_run_main(lower_module_c2y(src)), 15);
}

#[test]
fn c2y_octal_literals_lower() {
    assert_eq!(jit_run_main(lower_module_c2y("int main(void) { return 0o17 + 0O10; }")), 23);
    assert_eq!(jit_run_main(lower_module_c2y("int main(void) { return 0o1'7; }")), 15);
}

#[test]
fn c2y_complex_increment_decrement_lower() {
    // Each step changes only the real part.
    assert_eq!(
        jit_run_main(lower_module_c2y(
            "int main(void) { double _Complex z = 1.0 + 2.0i; ++z; z++; --z; return (int)(__real__ z * 10 + __imag__ z); }"
        )),
        22
    );
    // Postfix yields the old value; prefix yields the new one.
    assert_eq!(
        jit_run_main(lower_module_c2y(
            "int main(void) {\n\
                 double _Complex z = 1.0 + 2.0i;\n\
                 double _Complex old = z++;\n\
                 return (int)(__real__ old * 10 + __imag__ old)\n\
                     + (int)(__real__ ++z * 10 + __imag__ z);\n\
             }"
        )),
        44
    );
    // Integer and single-precision components step at their own
    // width.
    assert_eq!(
        jit_run_main(lower_module_c2y(
            "int main(void) { int _Complex z = 1 + 2i; ++z; return (int)(__real__ z * 10 + __imag__ z); }"
        )),
        22
    );
    assert_eq!(
        jit_run_main(lower_module_c2y(
            "int main(void) { float _Complex z = 1.0f + 2.0fi; --z; return (int)(__real__ z * 10 + __imag__ z); }"
        )),
        2
    );
}

#[test]
fn gnu_auto_type_lowers() {
    assert_eq!(jit_run_main(lower_module("int main(void) { __auto_type x = 7; return x; }")), 7);
    assert_eq!(jit_run_main(lower_module("int main(void) { int a[3] = {1, 2, 3}; __auto_type p = a; return p[2]; }")), 3);
}

#[test]
fn named_loops_lower() {
    assert_eq!(
        run_main(
            "int main(){int r = 0; \
             outer: for (int i = 0; i < 3; i++) { \
               for (int j = 0; j < 3; j++) { \
                 if (i == 1 && j == 1) continue outer; \
                 if (i == 2) break outer; \
                 r++; \
               } \
               r += 100; \
             } return r;}"
        ),
        104
    );
    assert_eq!(
        run_main(
            "int main(){int r = 0; \
             a: b: for (int i = 0; i < 3; i++) { \
               for (int j = 0; j < 3; j++) { \
                 if (j == 1) continue b; \
                 r++; \
               } \
             } return r;}"
        ),
        3
    );
    // A selection declaration does not break the label-to-switch link.
    assert_eq!(
        run_main(
            "int main(){int r = 0; \
             a: switch (int x = 1; x) { \
               case 1: for (int j = 0; j < 3; j++) { r++; break a; } \
               r = 99; \
             } return r;}"
        ),
        1
    );
    // Header loops must not steal the label of the loop they are in.
    assert_eq!(
        run_main(
            "int main(){int r = 0; \
             a: for (({ for (;;) { break; } }); r < 1; ) { \
               r++; if (r == 1) break a; \
             } return r;}"
        ),
        1
    );
}

#[test]
fn selection_declarations_lower() {
    assert_eq!(
        run_main(
            "int main(){int r = 0;                  if (int x = 0) r = 1; else r = x + 2;                  if (int y = 3; y > 2) r += y;                  switch (int z = 4) { case 4: r += z; break; default: r = 99; }                  switch (int w = 5; w) { case 5: r += w; }                  return r;}"
        ),
        14
    );
    // The declared object shadows an outer one and is gone after the
    // selection statement.
    assert_eq!(
        run_main("int main(){int x = 100; if (int x = 0) { return 1; }                  else if (x == 0) { x = 7; } else { return 2; } return x;}"),
        100
    );
    // Initialization still runs when the control expression folds
    // away.
    assert_eq!(run_main("int main(){int n = 0; if (int x = n++) {} return n;}"), 1);
}

#[test]
fn delimited_escapes_lower() {
    assert_eq!(run_main("int main(){char s[] = \"\\o{110}\\x{65}llo\"; return s[0] + s[4];}"), ('H' as i32) + ('o' as i32));
    assert_eq!(run_main("int main(){return (int)sizeof(\"\\u{2021}\");}"), 4);
    assert_eq!(run_main("int main(){return L\"\\u{2021}\"[0] == 0x2021;}"), 1);
}

#[test]
fn generic_type_operand_lowers_selected_arm() {
    assert_eq!(run_main("int main(){return _Generic(int, int: 3, default: 0);}"), 3);
    assert_eq!(run_main("int main(){return _Generic(long, int: 3, default: 4);}"), 4);
}

#[test]
fn sizes() {
    let t = fe::target::Target::native();
    assert_eq!(
        run_main("int main(){return sizeof(int)+sizeof(char)+sizeof(long);}"),
        (t.int_size + t.char_size + t.long_size) as i32
    );
    assert_eq!(run_main("int main(){int a[10]; return sizeof(a)/sizeof(a[0]);}"), 10);
}

#[test]
fn structs() {
    assert_eq!(run_main("struct P{int x; int y;}; int main(){struct P p; p.x = 3; p.y = 4; return p.x+p.y;}"), 7);
    assert_eq!(run_main("struct P{int x; int y;}; int main(){struct P p = {3,4}; return p.x+p.y;}"), 7);
    assert_eq!(
        run_main("struct P{int x; int y;}; int get(struct P p){return p.x+p.y;} int main(){return get((struct P){3,4});}"),
        7
    );
    assert_eq!(run_main("struct P{int x; int y;}; struct P g = {.y = 5}; int main(){return g.x+g.y;}"), 5);
    assert_eq!(
        run_main(
            "struct P{int x; int y;}; struct P bump(struct P p){struct P q = p; q.y++; return q;}\
             int main(){struct P p = {3, 4}; struct P q = bump(p); return q.x * 10 + q.y;}"
        ),
        35
    );
    assert_eq!(
        run_main(
            "struct S{int a[2]; struct I{int v;} inner;}; struct S id(struct S s){return s;}\
             int main(){struct S s = {{2, 3}, {4}}; struct S t = id(s); return t.a[0] + t.a[1] * 10 + t.inner.v * 100;}"
        ),
        432
    );
    assert_eq!(
        run_main("struct P{int x; int y;}; int f(int v){struct P p = {0}; p.x = v; return p.x + p.y;} int main(){return f(7);}"),
        7
    );
    assert_eq!(
        run_main("int f(int v){int a[4] = {0}; a[2] = v; return a[0] + a[1] + a[2] + a[3];} int main(){return f(5);}"),
        5
    );
    assert_eq!(
        run_main("struct P{int x; double y;}; struct P id(struct P p){return p;} int main(){struct P p = {3, 4.0}; struct P q = id(p); return q.x + (int)q.y;}"),
        7
    );
    assert_eq!(
        run_main("struct P{int x; double y;}; int main(){struct P p = {0}; p.x = 5; p.y = 2.0; return p.x + (int)p.y;}"),
        7
    );
}

#[test]
fn bool_conversions() {
    // Integer to `_Bool` is a truth test, not a truncation: 256
    // is nonzero, so true.
    assert_eq!(run_main("int main(){_Bool b = (_Bool)256; return b;}"), 1);
    assert_eq!(run_main("int main(){_Bool b = (_Bool)0; return b;}"), 0);
    assert_eq!(run_main("int main(){long x = -1; _Bool b = x; return b;}"), 1);
    // Pointers test against null: a live address is true even
    // when its low byte is zero; null is false.
    assert_eq!(run_main("int main(){int x = 1; int *p = &x; _Bool b = (_Bool)p; return b;}"), 1);
    assert_eq!(run_main("int main(){int *p = (int *)0; _Bool b = (_Bool)p; return b;}"), 0);
}

/// Integer-to-pointer casts extend by the source's signedness (C17
/// 6.3.2.3p5-6): MIR integer types are signless, so the conversion
/// has to sign-extend signed narrow sources to pointer width and
/// zero-extend unsigned ones. `(void *)-1` must be the all-ones
/// pattern, not `0xffffffff` (SQLite's `sqlite3Utf8CharLen(z, -1)`
/// sentinel leans on this).
#[test]
fn integer_to_pointer_casts_respect_signedness() {
    assert_eq!(run_main("int main(){const void *p = (const void*)-1; return (p == (const void*)-1) + ((long)p == -1);}"), 2);
    // Plain `char` is signed on x86-64 but unsigned on aarch64 and
    // riscv64, so spell out the signedness this check needs.
    assert_eq!(
        run_main(
            "int main(){volatile int i = -1; volatile short s = -1; volatile signed char c = -1;\
            return ((long)(void*)i == -1) + ((long)(void*)s == -1) * 2 + ((long)(void*)c == -1) * 4;}"
        ),
        7
    );
    // Unsigned narrow sources stay zero-extended, including the
    // full-width unsigned value whose sign bit is set.
    assert_eq!(
        run_main(
            "int main(){volatile unsigned u = 4294967295u; volatile unsigned short h = 65535; volatile unsigned char b = 255;\
            return ((unsigned long)(void*)u == 4294967295UL) + ((unsigned long)(void*)h == 65535UL) * 2 + ((unsigned long)(void*)b == 255UL) * 4;}"
        ),
        7
    );
    assert_eq!(run_main("int main(){volatile long l = -1; return (long)(void*)l == -1;}"), 1);
}

/// The conversion widens every `inttoptr` source to pointer width
/// before the cast, so the backend never re-decides the extension.
#[test]
fn integer_to_pointer_source_widens_to_pointer_width() {
    let module = lower_module(
        "void sink(const void *p); void f(volatile int i, volatile unsigned u, volatile short s, volatile char c) {\
            sink((const void*)i); sink((const void*)u); sink((const void*)s); sink((const void*)c); }",
    );
    let mut seen = 0;
    for func in module.funcs.iter().filter(|f| f.name == "f") {
        for inst in func.insts.iter() {
            if let mir::InstKind::IntToPtr(src) = &inst.kind {
                assert_eq!(func.values[*src].ty, mir::Ty::int(64));
                seen += 1;
            }
        }
    }
    assert_eq!(seen, 4, "casts lowered through inttoptr");
}

/// Plain `char` takes its signedness from the target (signed on
/// x86-64, unsigned on aarch64 and riscv64); `signed char` and
/// `unsigned char` never do. The widening before `inttoptr` follows.
#[test]
fn int_to_pointer_extension_follows_target_char_signedness() {
    let src = "void sink(const void *p); void f(volatile char c, volatile signed char s, volatile unsigned char u) {\
                sink((const void*)c); sink((const void*)s); sink((const void*)u); }";
    let cases = [
        (fe::target::Target::X86_64, ["sext", "sext", "zext"]),
        (fe::target::Target::AARCH64, ["zext", "sext", "zext"]),
    ];
    for (target, expected) in cases {
        let module = lower_module_for(src, target);
        let mut exts = Vec::new();
        for func in module.funcs.iter().filter(|f| f.name == "f") {
            for inst in func.insts.iter() {
                let mir::InstKind::IntToPtr(v) = &inst.kind else {
                    continue;
                };
                let def = func.insts.iter().find(|i| i.result == Some(*v)).expect("inttoptr source widens to pointer width first");
                exts.push(match &def.kind {
                    mir::InstKind::Sext(_) => "sext",
                    mir::InstKind::Zext(_) => "zext",
                    _ => panic!("inttoptr source is not a sext/zext"),
                });
            }
        }
        assert_eq!(exts, expected, "extension before inttoptr on {target:?}");
    }
}

/// The optimized pipeline keeps the conversion values intact with a
/// load in the way (nothing to constant-fold).
#[test]
fn optimized_pipeline_preserves_pointer_cast_signedness() {
    let src = "int main(){volatile int i = -1; volatile unsigned u = 4294967295u; volatile unsigned short h = 65535;\
               return ((long)(void*)i != -1)\
                    + ((unsigned long)(void*)u != 4294967295UL) * 2\
                    + ((long)(void*)h != 65535) * 4;}";
    let mut module = lower_module(src);
    run_opt_pipeline(&mut module);
    let dump = mir::display::dump_module(&module);
    assert_eq!(jit_run_main(module), 0, "casts changed values:\n{dump}");
}

#[test]
fn bool_to_byte_convert_is_a_no_op() {
    // `_Bool` registers hold a byte, so a conversion to an 8-bit
    // integer is already the target representation: no
    // zero-extension (which Cranelift rejects at equal widths).
    assert_eq!(run_main("unsigned char f(_Bool b){ return (unsigned char)b; } int main(){ return f(1) + f(0); }"), 1);
    assert_eq!(run_main("int f(_Bool b){ char c = b; return c; } int main(){ return f(1); }"), 1);
}

#[test]
fn debug_records_describe_functions_scopes_and_globals() {
    let module = lower_module(
        "int g = 7;\n\
         extern int ext;\n\
         static int counter;\n\
         struct P { int x; unsigned y : 3; };\n\
         struct P p = { 1, 2 };\n\
         int add(int a, int b) {\n\
             int sum = a + b;\n\
             for (int i = 0; i < 2; i++) { sum += i; }\n\
             { int shadow = sum; sum = shadow; }\n\
             return sum;\n\
         }\n\
         int main(void) { int x = 3; return add(x, g); }",
    );
    assert_eq!(module.files.len(), 1, "one source file");
    assert_eq!(module.files[0].path, "<__glcc_source>");
    assert!(module.funcs.iter().all(|f| f.debug.is_some()), "every defined function carries a debug record");

    fn collect(vars: &[mir::VarDebug], scopes: &[mir::ScopeDebug], out: &mut Vec<String>) {
        out.extend(vars.iter().filter_map(|v| v.name.clone()));
        for scope in scopes {
            collect(&scope.vars, &scope.children, out);
        }
    }

    let add = module.funcs.iter().find(|f| f.name == "add").unwrap();
    let dbg = add.debug.as_ref().expect("add debug record");
    assert!(dbg.decl.is_some());
    assert_eq!(dbg.params.iter().filter_map(|p| p.name.as_deref()).collect::<Vec<_>>(), ["a", "b"]);
    let mut names = Vec::new();
    collect(&dbg.vars, &dbg.scopes, &mut names);
    for expected in ["sum", "i", "shadow"] {
        assert!(names.contains(&expected.to_string()), "missing {expected}: {names:?}");
    }
    fn check_scopes(scopes: &[mir::ScopeDebug]) {
        for scope in scopes {
            assert!(scope.start.line <= scope.end.line, "{scope:?}");
            check_scopes(&scope.children);
        }
    }
    check_scopes(&dbg.scopes);
    // Every instruction maps to a line of the body.
    for inst in add.insts.iter() {
        let loc = inst.loc.expect("instruction location");
        assert_eq!(loc.file, 0);
        assert!((6..=11).contains(&loc.line), "line {}", loc.line);
    }

    // The struct global carries member positions, including the
    // bit-field as a bit offset and width.
    let p = global_named(&module, "p");
    let gd = p.debug.as_ref().expect("p debug record");
    match &module.debug_types[gd.debug_ty.expect("p type") as usize] {
        mir::DebugType::Struct { name, members, .. } => {
            assert_eq!(name.as_deref(), Some("P"));
            assert_eq!(members.len(), 2);
            assert_eq!(members[0].name.as_deref(), Some("x"));
            assert_eq!(members[0].location, mir::MemberLocation::Byte(0));
            assert_eq!(members[1].name.as_deref(), Some("y"));
            assert_eq!(members[1].location, mir::MemberLocation::Bit { offset: 32, size: 3 });
        }
        other => panic!("struct P lowered as {other:?}"),
    }

    // Defined, tentative-static, and imported objects all carry
    // records, like the legacy collector.
    assert!(global_named(&module, "g").debug.is_some());
    assert!(global_named(&module, "counter").debug.is_some());
    assert!(global_named(&module, "ext").debug.is_some());
    let sum = dbg.vars.iter().find(|v| v.name.as_deref() == Some("sum")).unwrap();
    assert!(matches!(sum.home, mir::VarHome::Slot(_)));
}

#[test]
fn bitint_signed_comparison_uses_the_magnitude_of_the_top_limb() {
    let src = "int main(void) {\n\
               int ok = 0;\n\
               _BitInt(300) v = -5;\n\
               if (v < 0 && v <= 0 && !(v > 0) && !(v >= 0)) ok |= 1;\n\
               if (v < -4 && !(v < -5) && v <= -5 && v > -6) ok |= 2;\n\
               _BitInt(300) p = 5;\n\
               if (p > 0 && p >= 0 && !(p < 0) && !(p <= 0)) ok |= 4;\n\
               if (p > 4 && !(p > 5) && p >= 5 && p < 6) ok |= 8;\n\
               _BitInt(300) big = (_BitInt(300))1 << 299;\n\
               _BitInt(300) small = big - 1;\n\
               if (big < 0 && small > 0 && big < small) ok |= 16;\n\
               unsigned _BitInt(300) u = (_BitInt(300))-1;\n\
               if (u > 0 && u == ~(unsigned _BitInt(300))0) ok |= 32;\n\
               return ok;\n\
           }";
    assert_eq!(jit_run_main(lower_module_c23(src)), 63);
}
