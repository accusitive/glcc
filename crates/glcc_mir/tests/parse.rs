//! Tests for the MIR text parser (`glcc_mir::parse`), covering the v1
//! grammar from `docs/edg-mir-backend.md`.

use glcc_mir::display::dump_module;
use glcc_mir::verify::verify_module;
use glcc_mir::{Block, Const, DataReloc, Def, Extern, FloatKind, Func, GepIndex, Global, Inst, InstKind, IcmpPred, Ty, parse_module};

fn ok(text: &str) -> glcc_mir::Module {
    let module = parse_module(text).expect("parse should succeed");
    assert_eq!(verify_module(&module), Vec::<String>::new(), "parsed module must verify");
    module
}

fn err_at(text: &str, line: u32, needle: &str) {
    let error = parse_module(text).expect_err("parse should fail");
    assert_eq!(error.line, line, "wrong error line for: {}", error.message);
    assert!(error.message.contains(needle), "error message `{}` lacks `{needle}`", error.message);
}

#[test]
fn spec_add_example() {
    let module = ok(
        ".func @add sig=(i64, i64) -> i64\n\
         .params %0:i64, %1:i64\n\
         b0:\n\
         \x20 %2:i64 = add %0, %1\n\
         \x20 ret %2\n",
    );
    assert_eq!(module.funcs.len(), 1);
    let func = &module.funcs[Func::from_usize(0)];
    assert_eq!(func.name, "add");
    assert_eq!(func.sig.params, vec![Ty::int(64), Ty::int(64)]);
    assert_eq!(func.sig.ret, Ty::int(64));
    let entry = func.entry;
    assert_eq!(func.blocks[entry].params.len(), 2);
    assert!(matches!(func.values[func.blocks[entry].params[0]].def, Def::Param(b, 0) if b == entry));
    assert_eq!(func.blocks[entry].insts.len(), 1);
    let inst = &func.insts[func.blocks[entry].insts[0]];
    assert!(matches!(&inst.kind, InstKind::Add(a, b) if *a == func.blocks[entry].params[0] && *b == func.blocks[entry].params[1]));
    assert!(matches!(&func.blocks[entry].term, Some(Terminator::Return { val: Some(_) })));
}

use glcc_mir::Terminator;

#[test]
fn comments_and_blank_lines() {
    ok("# leading comment\n\n.func @f sig=() -> void  # trailing\nb0:\n  ret  # done\n");
}

#[test]
fn control_flow_with_label_params() {
    let module = ok(
        ".func @count sig=(i64) -> i64\n\
         .params %0:i64\n\
         b0:\n\
         \x20 %1:i64 = add %0, 0\n\
         \x20 br b1(%1, %0)\n\
         b1(%2:i64, %3:i64):\n\
         \x20 %4:bool = icmp.slt %2, %3\n\
         \x20 brif %4 b2(%2) b3\n\
         b2:\n\
         \x20 %5:i64 = add %2, 1\n\
         \x20 br b1(%5, %3)\n\
         b3:\n\
         \x20 ret %2\n",
    );
    let func = &module.funcs[Func::from_usize(0)];
    assert_eq!(func.blocks.len(), 4);
    assert_eq!(func.blocks[Block::from_usize(1)].params.len(), 2);
    assert!(matches!(&func.blocks[Block::from_usize(1)].term, Some(Terminator::CondBr { .. })));
    assert!(matches!(&func.blocks[Block::from_usize(0)].term, Some(Terminator::Br { dest, .. }) if dest.index() == 1));
}

#[test]
fn forward_branch_materializes_arity_before_label() {
    let module = ok(
        ".func @f sig=(i64) -> i64\n\
         .params %0:i64\n\
         b0:\n\
         \x20 brif %0 b1(%0) b2\n\
         b1(%1:i64):\n\
         \x20 ret %1\n\
         b2:\n\
         \x20 ret 0\n",
    );
    assert_eq!(module.funcs[Func::from_usize(0)].blocks[Block::from_usize(1)].params.len(), 1);
}

#[test]
fn switch_cases_and_default() {
    let module = ok(
        ".func @cls sig=(i64) -> i64\n\
         .params %0:i64\n\
         b0:\n\
         \x20 switch %0 [1: b1, 2: b2, -1: b2] default b3\n\
         b1:\n\
         \x20 ret 10\n\
         b2:\n\
         \x20 ret 20\n\
         b3:\n\
         \x20 ret 0\n",
    );
    let func = &module.funcs[Func::from_usize(0)];
    let (cases, default) = match &func.blocks[Block::from_usize(0)].term {
        Some(Terminator::Switch { cases, default, .. }) => (cases.clone(), *default),
        other => panic!("expected switch, got {other:?}"),
    };
    assert_eq!(cases, vec![(1, Block::from_usize(1)), (2, Block::from_usize(2)), (-1, Block::from_usize(2))]);
    assert_eq!(default.index(), 3);
}

#[test]
fn switch_targets_must_have_no_params() {
    err_at(
        ".func @f sig=(i64) -> void\n\
         .params %0:i64\n\
         b0:\n\
         \x20 switch %0 [1: b1] default b2\n\
         b1(%1:i64):\n\
         \x20 ret\n\
         b2:\n\
         \x20 ret\n",
        5,
        "arguments",
    );
}

#[test]
fn globals_scalar_array_bytes_zeroinit() {
    let module = ok(
        ".global @g1 ty=i64 init=42\n\
         .global @g2 ty=[4 x i8] init=[1, 2, 3, 4]\n\
         .global @g3 ty=struct<12>{0:i64, 8:i32} init=[7, 8]\n\
         .global @g4 ty=i64 init=zeroinit\n\
         .global @g5 ty=ptr(i8) init=bytes<aabbcc>\n\
         .global @g6 ty=i64 init=undef\n\
         .global @g7 local const align=16 ty=i64 init=1\n\
         .global @g8 weak tls ty=i64 init=undef\n\
         .dataextern @ext ty=i64\n",
    );
    assert_eq!(module.globals.len(), 9);
    let g = |i: usize| &module.globals[Global::from_usize(i)];
    assert_eq!(g(0).init, Some(Const::Int { bits: 64, value: 42 }));
    assert_eq!(g(1).init, Some(Const::Bytes(vec![1, 2, 3, 4])));
    let image = match &g(2).init {
        Some(Const::Bytes(image)) => image.clone(),
        other => panic!("expected byte image, got {other:?}"),
    };
    assert_eq!(image.len(), 12);
    assert_eq!(&image[0..8], &7i64.to_le_bytes());
    assert_eq!(&image[8..12], &8i32.to_le_bytes());
    assert_eq!(g(3).init, Some(Const::Zero));
    assert_eq!(g(4).init, Some(Const::Bytes(vec![0xaa, 0xbb, 0xcc])));
    assert_eq!(g(5).init, None);
    assert!(g(6).local);
    assert!(!g(6).writable);
    assert_eq!(g(6).align, Some(16));
    assert!(g(7).weak);
    assert!(g(7).tls);
    assert!(g(8).external);
}

#[test]
fn call_ext_variadic_and_discarded_result() {
    let module = ok(
        ".extern @printf sig=(ptr(i8), ...) -> i32\n\
         .func @main sig=() -> i32\n\
         b0:\n\
         \x20 call.ext @printf ptr(i8) null, i64 5\n\
         \x20 %0:i32 = call.ext @printf ptr(i8) null, i64 6\n\
         \x20 ret %0\n",
    );
    let func = &module.funcs[Func::from_usize(0)];
    assert!(module.externs[Extern::from_usize(0)].sig.variadic);
    assert_eq!(func.blocks[Block::from_usize(0)].insts.len(), 2);
    assert!(func.insts[Inst::from_usize(0)].result.is_none());
    assert!(func.insts[Inst::from_usize(1)].result.is_some());
}

#[test]
fn call_internal_forward_reference() {
    let module = ok(
        ".func @main sig=() -> i64\n\
         b0:\n\
         \x20 %0:i64 = call @later i64 3\n\
         \x20 ret %0\n\
         .func @later sig=(i64) -> i64\n\
         .params %0:i64\n\
         b0:\n\
         \x20 ret %0\n",
    );
    assert_eq!(module.funcs.len(), 2);
    let main = &module.funcs[Func::from_usize(0)];
    match &main.insts[main.blocks[main.entry].insts[0]].kind {
        InstKind::Call { callee, args } => {
            assert_eq!(callee.index(), 1);
            assert_eq!(args.len(), 1);
        }
        other => panic!("expected call, got {other:?}"),
    }
}

#[test]
fn call_void_rejects_result_annotation() {
    err_at(
        ".func @f sig=() -> void\n\
         b0:\n\
         \x20 %0:bool = call @g\n\
         \x20 ret\n\
         .func @g sig=() -> void\n\
         b0:\n\
         \x20 ret\n",
        3,
        "void",
    );
}

#[test]
fn call_indirect() {
    let module = ok(
        ".func @apply sig=(ptr(i64)) -> i64\n\
         .params %0:ptr(i64)\n\
         b0:\n\
         \x20 %1:i64 = call.ind sig=(i64) -> i64 %0, i64 7\n\
         \x20 ret %1\n",
    );
    let func = &module.funcs[Func::from_usize(0)];
    match &func.insts[func.blocks[func.entry].insts[0]].kind {
        InstKind::CallIndirect { callee, args, sig } => {
            assert_eq!(*callee, func.blocks[func.entry].params[0]);
            assert_eq!(args.len(), 1);
            assert_eq!(sig.params, vec![Ty::int(64)]);
        }
        other => panic!("expected call.ind, got {other:?}"),
    }
}

#[test]
fn gep_field_index_const_steps() {
    let module = ok(
        ".func @f sig=(i64) -> i64\n\
         .params %0:i64\n\
         b0:\n\
         \x20 %1:ptr(struct<16>{0:i64, 8:[2 x i32]}) = alloca struct<16>{0:i64, 8:[2 x i32]}\n\
         \x20 %2:ptr([2 x i32]) = gep %1 field 1\n\
         \x20 %3:ptr(i32) = gep %2 index %0, 1\n\
         \x20 %4:i32 = load %3\n\
         \x20 %5:i64 = sext %4\n\
         \x20 ret %5\n",
    );
    let func = &module.funcs[Func::from_usize(0)];
    let block = &func.blocks[func.entry];
    match &func.insts[block.insts[1]].kind {
        InstKind::GetElementPtr { base, indices } => {
            assert_eq!(Some(*base), func.insts[block.insts[0]].result);
            assert!(matches!(&indices[..], [GepIndex::Field(1)]));
        }
        other => panic!("expected gep, got {other:?}"),
    }
    match &func.insts[block.insts[2]].kind {
        InstKind::GetElementPtr { indices, .. } => assert!(matches!(
            &indices[..],
            [GepIndex::Index(_), GepIndex::ConstIndex(1)]
        )),
        other => panic!("expected gep, got {other:?}"),
    }
}

#[test]
fn casts_icmp_fcmp_preds() {
    let module = ok(
        ".func @ops sig=(i32, f64) -> f64\n\
         .params %0:i32, %1:f64\n\
         b0:\n\
         \x20 %2:i64 = sext %0\n\
         \x20 %3:i8 = trunc %0\n\
         \x20 %4:i64 = zext %3\n\
         \x20 %5:f64 = sitof %0\n\
         \x20 %6:i32 = uftoi %1\n\
         \x20 %7:bool = icmp.ule %0, 10\n\
         \x20 %8:bool = fcmp.ord %1, f64:0x0\n\
         \x20 %9:i64 = select %7, %2, 20\n\
         \x20 %10:i64 = not %2\n\
         \x20 %11:i64 = shl %2, 3\n\
         \x20 %12:f64 = fadd %1, f64:0x4000000000000000\n\
         \x20 %13:ptr(i8) = inttoptr %2\n\
         \x20 %14:i64 = ptrtoint %13\n\
         \x20 %15:f64 = fpext %5\n\
         \x20 %16:f64 = fneg %15\n\
         \x20 ret %12\n",
    );
    let func = &module.funcs[Func::from_usize(0)];
    let block = &func.blocks[func.entry];
    assert!(matches!(&func.insts[block.insts[5]].kind, InstKind::Icmp { pred: IcmpPred::Ule, .. }));
    assert!(matches!(&func.insts[block.insts[6]].kind, InstKind::Fcmp { .. }));
    assert!(matches!(&func.insts[block.insts[10]].kind, InstKind::FAdd(..)));
    assert!(matches!(&func.insts[block.insts[12]].kind, InstKind::PtrToInt(..)));
}

#[test]
fn memory_ops_and_flags() {
    let module = ok(
        ".func @mem sig=() -> i64\n\
         b0:\n\
         \x20 %0:ptr(i64) = alloca i64\n\
         \x20 store %0, 42\n\
         \x20 %1:i64 = load volatile %0\n\
         \x20 store volatile %0, %1\n\
         \x20 %2:ptr(i8) = alloca i8 x<256>\n\
         \x20 memcpy %2, %2, len=8\n\
         \x20 memset %2, byte=0, len=16\n\
         \x20 %3:ptr(i32) = alloca [4 x i32] x<2>\n\
         \x20 ret %1\n",
    );
    let func = &module.funcs[Func::from_usize(0)];
    let block = &func.blocks[func.entry];
    assert!(matches!(&func.insts[block.insts[2]].kind, InstKind::Load { flags, .. } if flags.volatile));
    assert!(matches!(&func.insts[block.insts[4]].kind, InstKind::Alloca { count: 256, .. }));
    assert!(matches!(&func.insts[block.insts[6]].kind, InstKind::Memset { byte: 0, len: 16, .. }));
    assert!(matches!(&func.insts[block.insts[7]].kind, InstKind::Alloca { count: 2, .. }));
}

#[test]
fn addr_ops() {
    let module = ok(
        ".global @obj ty=i64 init=0\n\
         .extern @helper sig=(i64) -> i64\n\
         .func @f sig=() -> i64\n\
         b0:\n\
         \x20 %0:ptr(i64) = globaladdr @obj\n\
         \x20 %1:ptr(i8) = funcaddr @f\n\
         \x20 %2:ptr(i8) = externaddr @helper\n\
         \x20 %3:i64 = load %0\n\
         \x20 ret %3\n",
    );
    let func = &module.funcs[Func::from_usize(0)];
    let block = &func.blocks[func.entry];
    assert!(matches!(&func.insts[block.insts[0]].kind, InstKind::GlobalAddr { .. }));
    assert!(matches!(&func.insts[block.insts[1]].kind, InstKind::FuncAddr { .. }));
    assert!(matches!(&func.insts[block.insts[2]].kind, InstKind::ExternAddr { .. }));
}

#[test]
fn lazy_forward_reference() {
    let module = ok(
        ".func @lazy sig=(i64) -> i64\n\
         .params %0:i64\n\
         b0:\n\
         \x20 %1:i64 = add %2, %0\n\
         \x20 br b1\n\
         b1:\n\
         \x20 %2:i64 = add %0, 1\n\
         \x20 ret %1\n",
    );
    let func = &module.funcs[Func::from_usize(0)];
    let block = &func.blocks[func.entry];
    let add = &func.insts[block.insts[0]];
    let (a, _b) = match &add.kind {
        InstKind::Add(a, b) => (*a, *b),
        other => panic!("expected add, got {other:?}"),
    };
    assert!(matches!(&func.values[a].def, Def::Inst(_)));
}

#[test]
fn lazy_block_param_bound_after_use() {
    let module = ok(
        ".func @lazy sig=(i64) -> i64\n\
         .params %0:i64\n\
         b0:\n\
         \x20 brif %0 b1(%2) b2\n\
         b1(%2:i64):\n\
         \x20 ret %2\n\
         b2:\n\
         \x20 ret 0\n",
    );
    let func = &module.funcs[Func::from_usize(0)];
    let param = func.blocks[Block::from_usize(1)].params[0];
    assert!(matches!(&func.values[param].def, Def::Param(b, 0) if b.index() == 1));
    assert_eq!(func.values[param].ty, Ty::int(64));
}

#[test]
fn float_const_bit_patterns() {
    let module = ok(
        ".func @f sig=() -> f64\n\
         b0:\n\
         \x20 %0:f64 = fadd f64:0x3ff0000000000000, f64:0x8000000000000000\n\
         \x20 ret %0\n",
    );
    let func = &module.funcs[Func::from_usize(0)];
    let block = &func.blocks[func.entry];
    let neg = match &func.insts[block.insts[0]].kind {
        InstKind::FAdd(_, b) => func.const_value(*b).unwrap().clone(),
        other => panic!("expected fadd, got {other:?}"),
    };
    assert_eq!(neg, Const::Float { kind: FloatKind::F64, bits: 0x8000000000000000 });
}

#[test]
fn wide_int_and_ptr_bits_types() {
    let module = ok(
        ".func @f sig=(i256, ptr<32>(i8)) -> i256\n\
         .params %0:i256, %1:ptr<32>(i8)\n\
         b0:\n\
         \x20 ret undef\n",
    );
    let func = &module.funcs[Func::from_usize(0)];
    assert_eq!(func.sig.params[0], Ty::WideInt { bits: 256 });
    assert_eq!(func.sig.params[1], Ty::ptr_bits(Ty::int(8), 32));
}

#[test]
fn unary_ops_and_bare_void_ret() {
    let module = ok(
        ".func @f sig=(i64) -> void\n\
         .params %0:i64\n\
         b0:\n\
         \x20 %1:i64 = neg %0\n\
         \x20 %2:i64 = clz %1\n\
         \x20 %3:i64 = ctz %1\n\
         \x20 %4:i64 = popcnt %1\n\
         \x20 %5:i64 = bswap %1\n\
         \x20 %6:f64 = sqrt f64:0x0\n\
         \x20 %7:f64 = ceil %6\n\
         \x20 %8:f64 = floor %6\n\
         \x20 %9:f64 = fneg %6\n\
         \x20 %10:i64 = srem %1, 3\n\
         \x20 ret\n",
    );
    assert_eq!(module.funcs[Func::from_usize(0)].blocks[Block::from_usize(0)].insts.len(), 10);
}

#[test]
fn dump_round_trip_does_not_panic() {
    let module = ok(
        ".global @counter ty=i64 init=0\n\
         .extern @putchar sig=(i32) -> i32\n\
         .func @add sig=(i64, i64) -> i64\n\
         .params %0:i64, %1:i64\n\
         b0:\n\
         \x20 %2:i64 = add %0, %1\n\
         \x20 ret %2\n\
         .func @loop sig=(i64) -> i64\n\
         .params %3:i64\n\
         b0:\n\
         \x20 %4:i64 = mul %3, 0\n\
         \x20 br b1(%3, %4)\n\
         b1(%5:i64, %6:i64):\n\
         \x20 %7:bool = icmp.slt %6, %5\n\
         \x20 brif %7 b2(%6) b3\n\
         b2:\n\
         \x20 %8:i64 = add %6, 1\n\
         \x20 br b1(%5, %8)\n\
         b3:\n\
         \x20 ret %6\n",
    );
    let text = dump_module(&module);
    assert!(text.contains("func @add:"), "dump should mention @add:\n{text}");
    assert!(text.contains("icmp"), "dump should mention icmp:\n{text}");
}

#[test]
fn error_unterminated_block() {
    err_at(
        ".func @f sig=() -> i64\n\
         b0:\n\
         \x20 %0:ptr(i64) = alloca i64\n\
         b1:\n\
         \x20 ret 0\n",
        3,
        "no terminator",
    );
}

#[test]
fn error_double_definition() {
    err_at(
        ".func @f sig=() -> i64\n\
         .params %0:i64\n\
         b0:\n\
         \x20 %0:i64 = add %0, %0\n\
         \x20 ret %0\n",
        4,
        "double definition",
    );
}

#[test]
fn error_double_block_label() {
    err_at(
        ".func @f sig=() -> void\n\
         b0:\n\
         \x20 ret\n\
         b1:\n\
         \x20 ret\n\
         b1:\n\
         \x20 ret\n",
        6,
        "duplicate label",
    );
}

#[test]
fn error_unknown_type() {
    err_at(".func @f sig=(flurb) -> void\nb0:\n  ret\n", 1, "unknown type");
}

#[test]
fn error_bad_terminator_arity() {
    err_at(
        ".func @f sig=(i64) -> void\n\
         .params %0:i64\n\
         b0:\n\
         \x20 brif %0 b1(%0) b2\n\
         b1:\n\
         \x20 ret\n\
         b2:\n\
         \x20 br b1\n",
        8,
        "arguments",
    );
}

#[test]
fn error_use_without_definition() {
    err_at(".func @f sig=() -> i64\nb0:\n  ret %7\n", 3, "never defined");
}

#[test]
fn error_trailing_garbage() {
    err_at(".func @f sig=() -> void\nb0:\n  unreachable extra tokens here\n", 3, "trailing");
}

#[test]
fn error_unknown_instruction() {
    err_at(".func @f sig=() -> void\nb0:\n  frobnicate %0\n", 3, "unknown instruction");
}

#[test]
fn error_unknown_global_reference() {
    err_at(".func @f sig=() -> void\nb0:\n  %0:ptr(i64) = globaladdr @nope\n  ret\n", 3, "unknown global");
}

#[test]
fn error_instruction_after_terminator() {
    err_at(
        ".func @f sig=() -> void\n\
         b0:\n\
         \x20 ret\n\
         \x20 unreachable\n",
        4,
        "already has a terminator",
    );
}

#[test]
fn error_externaddr_needs_extern() {
    err_at(
        ".global @g ty=i64 init=0\n\
         .func @f sig=() -> void\n\
         b0:\n\
         \x20 %0:ptr(i8) = externaddr @g\n\
         \x20 ret\n",
        4,
        "unknown extern",
    );
}

#[test]
fn error_branch_argument_must_be_value() {
    err_at(
        ".func @f sig=() -> void\n\
         b0:\n\
         \x20 br b1(7)\n\
         b1:\n\
         \x20 ret\n",
        3,
        "value reference",
    );
}

#[test]
fn error_scalar_init_for_aggregate() {
    err_at(".global @g ty=[2 x i32] init=5\n", 1, "non-scalar");
}

#[test]
fn error_bad_float_pattern_width() {
    err_at(
        ".func @f sig=() -> f64\n\
         b0:\n\
         \x20 %0:f64 = fadd f64:0x3ff00000000000000, f64:0x0\n\
         \x20 ret %0\n",
        3,
        "bit pattern",
    );
}

#[test]
fn error_missing_sig() {
    err_at(".func @f\nb0:\n  ret\n", 1, "sig=");
}

#[test]
fn error_function_without_body() {
    err_at(".func @f sig=() -> void\n", 1, "no terminator");
}

#[test]
fn error_type_nesting_too_deep() {
    let mut text = String::from(".func @f sig=(");
    for _ in 0..300 {
        text.push_str("ptr<");
    }
    for _ in 0..300 {
        text.push('>');
    }
    text.push_str(") -> void\nb0:\n  ret\n");
    err_at(&text, 1, "too deep");
}

#[test]
fn error_aggregate_init_too_large() {
    err_at(".global @g ty=[8589934592 x i8] init=[1]\n", 1, "4 GiB");
}

#[test]
fn ptr_paren_spelling() {
    let module = ok(
        ".func @f sig=(ptr(i8)) -> ptr(i8)\n\
         .params %0:ptr(i8)\n\
         b0:\n\
         \x20 ret %0\n",
    );
    assert_eq!(module.funcs[Func::from_usize(0)].sig.ret, Ty::ptr(Ty::int(8)));
}

#[test]
fn error_label_param_type_mismatches_branch_arg() {
    err_at(
        ".func @f sig=(i64) -> void\n\
         .params %0:i64\n\
         b0:\n\
         \x20 br b1(%0)\n\
         b1(%1:i32):\n\
         \x20 ret\n",
        5,
        "but declared",
    );
}

#[test]
fn reloc_scalar_init() {
    let module = ok(
        ".global @str ty=[6 x i8] init=bytes<68656c6c6f00>\n\
         .global @p ty=ptr(i8) init=reloc @str\n",
    );
    let p = &module.globals[Global::from_usize(1)];
    assert_eq!(p.init, Some(Const::Bytes(vec![0u8; 8])));
    assert!(matches!(&p.relocs[..], [(0, DataReloc::Global { target, addend: 0 })] if target.index() == 0));
}

#[test]
fn reloc_aggregate_offsets() {
    let module = ok(
        ".global @a ty=i64 init=1\n\
         .global @b ty=i64 init=2\n\
         .global @arr ty=[3 x ptr(i8)] init=[reloc @a, undef, reloc @b]\n\
         .global @s ty=struct<24>{0:i64, 8:ptr(i8), 16:ptr(i8)} init=[7, reloc @b+4, reloc @a]\n",
    );
    let arr = &module.globals[Global::from_usize(2)];
    assert_eq!(arr.init, Some(Const::Bytes(vec![0u8; 24])));
    assert!(matches!(
        &arr.relocs[..],
        [(0, DataReloc::Global { target, addend: 0 }), (16, DataReloc::Global { .. })]
            if target.index() == 0 && arr.relocs[1].0 == 16
    ));
    let s = &module.globals[Global::from_usize(3)];
    assert!(matches!(
        &s.relocs[..],
        [(8, DataReloc::Global { addend: 4, .. }), (16, DataReloc::Global { addend: 0, .. })]
    ));
    let first = match &s.relocs[0].1 {
        DataReloc::Global { target, .. } => target.index(),
        other => panic!("expected global reloc, got {other:?}"),
    };
    assert_eq!(first, 1);
}

#[test]
fn reloc_func_extern_targets() {
    let module = ok(
        ".extern @puts sig=(ptr(i8)) -> i32\n\
         .func @handler sig=() -> void\n\
         b0:\n\
         \x20 ret\n\
         .global @pf ty=ptr(i8) init=reloc @handler\n\
         .global @pe ty=ptr(i8) init=reloc @puts\n",
    );
    let pf = &module.globals[Global::from_usize(0)];
    assert!(matches!(&pf.relocs[..], [(0, DataReloc::Func { target })] if target.index() == 0));
    let pe = &module.globals[Global::from_usize(1)];
    assert!(matches!(&pe.relocs[..], [(0, DataReloc::FuncExtern { target })] if target.index() == 0));
}

#[test]
fn reloc_addend() {
    let module = ok(
        ".global @buf ty=[16 x i8] init=zeroinit\n\
         .global @p ty=ptr(i8) init=reloc @buf+8\n",
    );
    let p = &module.globals[Global::from_usize(1)];
    assert!(matches!(&p.relocs[..], [(0, DataReloc::Global { target, addend: 8 })] if target.index() == 0));
}

#[test]
fn error_reloc_unknown_symbol() {
    err_at(".global @p ty=ptr(i8) init=reloc @missing\n", 1, "unknown relocation symbol");
}

#[test]
fn error_reloc_non_pointer_slot() {
    err_at(
        ".global @a ty=i64 init=1\n\
         .global @s ty=struct<12>{0:i64, 8:i32} init=[1, reloc @a]\n",
        2,
        "non-pointer",
    );
}

#[test]
fn error_reloc_bad_addend() {
    err_at(".global @p ty=ptr(i8) init=reloc @p+x\n", 1, "addend");
}

#[test]
fn error_reloc_missing_symbol() {
    err_at(".global @p ty=ptr(i8) init=reloc\n", 1, "@symbol");
}

#[test]
fn init_registers_constructors_in_order() {
    let module = ok(
        ".func @sti_b sig=() -> void\n\
         b0:\n\
         \x20 ret\n\
         .func @main sig=() -> i32\n\
         b0:\n\
         \x20 ret 0\n\
         .init @sti_b\n\
         .func @sti_a sig=() -> void\n\
         b0:\n\
         \x20 ret\n\
         .init @sti_a\n",
    );
    let names: Vec<&str> = module.init_funcs.iter().map(|f| module.funcs[*f].name.as_str()).collect();
    assert_eq!(names, vec!["sti_b", "sti_a"], "list order is the emission order");
}

#[test]
fn error_init_unknown_symbol() {
    err_at(".init @missing\n", 1, "unknown .init target @missing");
}

#[test]
fn error_init_extern_target() {
    err_at(".extern @f sig=() -> void\n.init @f\n", 2, "is an .extern");
}

#[test]
fn error_init_before_declaration() {
    err_at(".init @f\n.func @f sig=() -> void\nb0:\n  ret\n", 1, "unknown .init target @f");
}

#[test]
fn error_init_duplicate_entry() {
    err_at(
        ".func @f sig=() -> void\nb0:\n  ret\n.init @f\n.init @f\n",
        5,
        "duplicate .init entry @f",
    );
}

#[test]
fn init_shows_up_in_dump() {
    let module = ok(
        ".func @sti sig=() -> void\nb0:\n  ret\n.init @sti\n",
    );
    let dump = dump_module(&module);
    assert!(dump.contains(".init @sti"), "{dump}");
}

#[test]
fn error_init_non_void_target() {
    let module_text = ".func @f sig=() -> i32\nb0:\n  ret 0\n.init @f\n";
    let error = parse_module(module_text).expect_err("parse should fail");
    assert!(error.message.contains("must take no parameters and return void"), "{}", error.message);
}

#[test]
fn vector_types_and_ops() {
    let module = ok(
        ".func @v sig=(ptr(i32)) -> i32\n\
         .params %0:ptr(i32)\n\
         b0:\n\
         \x20 %1:ptr(i32) = gep %0, 0\n\
         \x20 %2:v4xi32 = load %1\n\
         \x20 %3:v4xi32 = splat 7\n\
         \x20 %4:v4xi32 = add %2, %3\n\
         \x20 %5:i32 = extractlane %4, 2\n\
         \x20 ret %5\n",
    );
    let func = &module.funcs[Func::from_usize(0)];
    let kinds: Vec<&InstKind> = func.blocks[func.entry].insts.iter().map(|inst| &func.insts[*inst].kind).collect();
    assert!(matches!(kinds[2], InstKind::Splat(_)), "{kinds:?}");
    assert!(matches!(kinds[4], InstKind::ExtractLane { lane: 2, .. }), "{kinds:?}");
    let splatted = match &kinds[2] {
        InstKind::Splat(v) => *v,
        _ => unreachable!(),
    };
    assert_eq!(func.values[splatted].ty, Ty::int(32));
}
#[test]
fn const_operand_round_trip_probe() {
    let text = ".func @f sig=(i64) -> i64\n.params %0:i64\nb0:\n %1:i64 = add %0, i64(4)\n ret %1\n";
    match glcc_mir::parse_module(text) {
        Ok(_) => println!("parses fine"),
        Err(e) => println!("ERR: {}", e.message),
    }
}
