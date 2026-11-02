use super::*;

fn struct_ty(fields: Vec<mir::Ty>) -> mir::Ty {
    let offsets = mir::dense_field_offsets(&fields);
    let size = fields.iter().map(|f| f.size_bytes()).sum();
    mir::Ty::Struct { fields, offsets, size }
}

fn i32_ty() -> mir::Ty {
    mir::Ty::int(32)
}

fn i64_ty() -> mir::Ty {
    mir::Ty::int(64)
}

fn f32_ty() -> mir::Ty {
    mir::Ty::Float(mir::FloatKind::F32)
}

fn f64_ty() -> mir::Ty {
    mir::Ty::Float(mir::FloatKind::F64)
}

fn pieces_of(loc: &ParamLoc) -> Vec<(u64, Type)> {
    match loc {
        ParamLoc::Regs(pieces) => pieces.clone(),
        _ => panic!("expected registers"),
    }
}

fn param0(target: Target, ty: &mir::Ty) -> ParamLoc {
    let layout = layout_sig(target, std::slice::from_ref(ty), &mir::Ty::Void, false);
    assert_eq!(layout.params.len(), 1);
    layout.params.into_iter().next().unwrap()
}

const X86: Target = Target {
    arch: Arch::X86_64,
    os: Os::Linux,
};
const ARM: Target = Target {
    arch: Arch::Aarch64,
    os: Os::Linux,
};
const RV: Target = Target {
    arch: Arch::Riscv64,
    os: Os::Linux,
};
const WIN64: Target = Target::WINDOWS_X86_64;

// Oracle shapes t2..t10 (8/16/12/32B structs).
fn p8() -> mir::Ty {
    struct_ty(vec![i32_ty(), i32_ty()])
}
fn q16() -> mir::Ty {
    struct_ty(vec![i64_ty(), i64_ty()])
}
fn r16() -> mir::Ty {
    struct_ty(vec![f64_ty(), f64_ty()])
}
fn m16() -> mir::Ty {
    struct_ty(vec![f64_ty(), i32_ty()])
}
fn s8() -> mir::Ty {
    struct_ty(vec![f32_ty(), f32_ty()])
}
fn t16() -> mir::Ty {
    struct_ty(vec![f32_ty(), f32_ty(), f32_ty(), f32_ty()])
}
fn u8() -> mir::Ty {
    struct_ty(vec![i32_ty(), f32_ty()])
}
fn big32() -> mir::Ty {
    struct_ty(vec![i64_ty(), i64_ty(), i64_ty(), i64_ty()])
}
fn twelve() -> mir::Ty {
    struct_ty(vec![i32_ty(), i32_ty(), i32_ty()])
}

#[test]
fn x86_splits_per_eightbyte() {
    use types::*;
    assert_eq!(pieces_of(&param0(X86, &p8())), vec![(0, I64)]);
    assert_eq!(pieces_of(&param0(X86, &q16())), vec![(0, I64), (8, I64)]);
    assert_eq!(pieces_of(&param0(X86, &r16())), vec![(0, F64), (8, F64)]);
    // Mixed split: SSE eightbyte first, INTEGER second.
    assert_eq!(pieces_of(&param0(X86, &m16())), vec![(0, F64), (8, I32)]);
    // Single SSE eightbyte carries both floats.
    assert_eq!(pieces_of(&param0(X86, &s8())), vec![(0, F64)]);
    assert_eq!(pieces_of(&param0(X86, &t16())), vec![(0, F64), (8, F64)]);
    // INTEGER wins ties inside one eightbyte.
    assert_eq!(pieces_of(&param0(X86, &u8())), vec![(0, I64)]);
    assert_eq!(pieces_of(&param0(X86, &twelve())), vec![(0, I64), (8, I32)]);
    assert!(matches!(param0(X86, &big32()), ParamLoc::Stack { .. }));
    assert!(matches!(param0(X86, &mir::Ty::Float(mir::FloatKind::F80)), ParamLoc::Stack { .. }));
    // Quad rides alone.
    assert_eq!(pieces_of(&param0(X86, &mir::Ty::Float(mir::FloatKind::F128))), vec![(0, F128)]);
}

#[test]
fn arm_hfa_or_x_or_pointer() {
    use types::*;
    assert_eq!(pieces_of(&param0(ARM, &p8())), vec![(0, I64)]);
    assert_eq!(pieces_of(&param0(ARM, &q16())), vec![(0, I64), (8, I64)]);
    // HFA at member width.
    assert_eq!(pieces_of(&param0(ARM, &r16())), vec![(0, F64), (8, F64)]);
    assert_eq!(pieces_of(&param0(ARM, &s8())), vec![(0, F32), (4, F32)]);
    assert_eq!(pieces_of(&param0(ARM, &t16())), vec![(0, F32), (4, F32), (8, F32), (12, F32)]);
    // Mixed composites go wholly in X registers.
    assert_eq!(pieces_of(&param0(ARM, &m16())), vec![(0, I64), (8, I64)]);
    assert_eq!(pieces_of(&param0(ARM, &u8())), vec![(0, I64)]);
    assert_eq!(pieces_of(&param0(ARM, &twelve())), vec![(0, I64), (8, I64)]);
    // >16B crosses by reference (one GP register).
    assert!(matches!(param0(ARM, &big32()), ParamLoc::Pointer));
    assert_eq!(pieces_of(&param0(ARM, &mir::Ty::Float(mir::FloatKind::F128))), vec![(0, F128)]);
}

#[test]
fn riscv_field_split() {
    use types::*;
    assert_eq!(pieces_of(&param0(RV, &p8())), vec![(0, I64)]);
    assert_eq!(pieces_of(&param0(RV, &q16())), vec![(0, I64), (8, I64)]);
    assert_eq!(pieces_of(&param0(RV, &r16())), vec![(0, F64), (8, F64)]);
    // Split per field across register files.
    assert_eq!(pieces_of(&param0(RV, &m16())), vec![(8, I64), (0, F64)]);
    assert_eq!(pieces_of(&param0(RV, &s8())), vec![(0, F32), (4, F32)]);
    assert_eq!(pieces_of(&param0(RV, &u8())), vec![(0, I64), (4, F32)]);
    // More than two FP members demote to integer registers.
    assert_eq!(pieces_of(&param0(RV, &t16())), vec![(0, I64), (8, I64)]);
    assert_eq!(pieces_of(&param0(RV, &twelve())), vec![(0, I64), (8, I64)]);
    // Quad in integer registers.
    assert_eq!(pieces_of(&param0(RV, &mir::Ty::Float(mir::FloatKind::F128))), vec![(0, I64), (8, I64)]);
    // >16B crosses by reference.
    assert!(matches!(param0(RV, &big32()), ParamLoc::Pointer));
}

#[test]
fn returns_mirror_params_and_sret() {
    // Small struct returns ride registers.
    for target in [X86, ARM, RV] {
        let layout = layout_sig(target, &[], &p8(), false);
        assert!(matches!(layout.ret, RetLoc::Regs(_)));
        assert!(layout.slots.iter().all(|s| !matches!(s.kind, SlotKind::Sret)));
    }
    // Large struct returns take the hidden out-pointer.
    for target in [X86, ARM, RV] {
        let layout = layout_sig(target, &[], &big32(), false);
        assert!(matches!(layout.ret, RetLoc::Sret));
        assert!(layout.slots.iter().any(|s| matches!(s.kind, SlotKind::Sret)));
    }
}

#[test]
fn variadic_layout_cursors() {
    // x86-64 named (int): gp starts past one register.
    let layout = layout_sig(X86, &[i32_ty()], &i32_ty(), true);
    assert!(layout.variadic);
    assert_eq!(layout.named_gp, 1);
    assert_eq!(layout.named_fp, 0);
    // Padding fills both files in declarations (FP pads are
    // 128-bit so 128-bit actuals survive; see `layout_sig`).
    let gp_pads = layout.slots.iter().filter(|s| matches!(s.kind, SlotKind::Pad) && s.ty == types::I64).count();
    let fp_pads = layout.slots.iter().filter(|s| matches!(s.kind, SlotKind::Pad) && s.ty == types::F128).count();
    assert_eq!(gp_pads, 5);
    assert_eq!(fp_pads, 8);
}

#[test]
fn win64_scalars_use_gp_and_xmm_files() {
    use types::*;
    assert_eq!(pieces_of(&param0(WIN64, &i32_ty())), vec![(0, I32)]);
    assert_eq!(pieces_of(&param0(WIN64, &i64_ty())), vec![(0, I64)]);
    assert_eq!(pieces_of(&param0(WIN64, &f32_ty())), vec![(0, F32)]);
    assert_eq!(pieces_of(&param0(WIN64, &f64_ty())), vec![(0, F64)]);
    assert_eq!(pieces_of(&param0(WIN64, &mir::Ty::ptr(mir::Ty::Void))), vec![(0, I64)]);
    assert_eq!(pieces_of(&param0(WIN64, &mir::Ty::Bool)), vec![(0, I8)]);
}

#[test]
fn win64_aggregates_by_cover_or_reference() {
    use types::*;
    // 1, 2, 4, 8 bytes cross as an integer of that width.
    let s1 = struct_ty(vec![mir::Ty::int(8)]);
    let s2 = struct_ty(vec![mir::Ty::int(16)]);
    let s4 = struct_ty(vec![i32_ty()]);
    assert_eq!(pieces_of(&param0(WIN64, &s1)), vec![(0, I8)]);
    assert_eq!(pieces_of(&param0(WIN64, &s2)), vec![(0, I16)]);
    assert_eq!(pieces_of(&param0(WIN64, &s4)), vec![(0, I32)]);
    assert_eq!(pieces_of(&param0(WIN64, &p8())), vec![(0, I64)]);
    // Every other size crosses by reference, one GP register.
    let s3 = struct_ty(vec![mir::Ty::int(8), mir::Ty::int(16)]);
    let s5 = struct_ty(vec![mir::Ty::int(32), mir::Ty::int(8)]);
    for ty in [s3, s5, twelve(), q16(), r16(), big32()] {
        assert!(matches!(param0(WIN64, &ty), ParamLoc::Pointer), "{ty:?}");
    }
}

#[test]
fn win64_returns_and_sret() {
    // Small aggregates come back in RAX.
    let layout = layout_sig(WIN64, &[], &p8(), false);
    assert_eq!(layout.ret, RetLoc::Regs(vec![(0, types::I64)]));
    let layout = layout_sig(WIN64, &[], &f64_ty(), false);
    assert_eq!(layout.ret, RetLoc::Regs(vec![(0, types::F64)]));
    // Anything else takes the hidden out-pointer.
    let layout = layout_sig(WIN64, &[], &twelve(), false);
    assert_eq!(layout.ret, RetLoc::Sret);
    assert!(layout.slots.iter().any(|s| matches!(s.kind, SlotKind::Sret)));
}

#[test]
fn win64_target_shape() {
    assert_eq!(WIN64.call_conv(), CallConv::WindowsFastcall);
    assert_eq!(X86.call_conv(), CallConv::SystemV);
    assert_eq!(WIN64.triple(), "x86_64-pc-windows-msvc");
    assert_eq!(WIN64.gp_regs(), 4);
    assert_eq!(WIN64.fp_regs(), 4);
    assert_eq!(WIN64.va_list_size(), 8);
    assert_eq!(WIN64.va_save_size(), 0);
    assert_eq!(Target::from_triple("x86_64-pc-windows-msvc"), Some(WIN64));
    assert_eq!(Target::from_triple("x86_64-unknown-linux-gnu"), Some(X86));
    assert_eq!(Target::from_triple("mips64-unknown-linux-gnu"), None);
}

#[test]
fn win64_variadic_pads_fill_positions() {
    // One named int: the three remaining positions take I64 pads
    // (never F128: FP varargs are read from their GP mirror).
    let layout = layout_sig(WIN64, &[i32_ty()], &i32_ty(), true);
    assert!(layout.variadic);
    let pads: Vec<Type> = layout.slots.iter().filter(|s| matches!(s.kind, SlotKind::Pad)).map(|s| s.ty).collect();
    assert_eq!(pads, vec![types::I64, types::I64, types::I64]);
    // FP named args consume positions like any other: mixed
    // (double, int) leaves two pads.
    let layout = layout_sig(WIN64, &[f64_ty(), i32_ty()], &i32_ty(), true);
    let pads = layout.slots.iter().filter(|s| matches!(s.kind, SlotKind::Pad)).count();
    assert_eq!(pads, 2);
    // Four named positions exhaust the register file: no pads.
    let layout = layout_sig(WIN64, &[i32_ty(), i32_ty(), i32_ty(), i32_ty()], &i32_ty(), true);
    assert!(layout.slots.iter().all(|s| !matches!(s.kind, SlotKind::Pad)));
    // The hidden sret pointer occupies position 0 like MSVC's.
    let layout = layout_sig(WIN64, &[i32_ty()], &big32(), true);
    let pads = layout.slots.iter().filter(|s| matches!(s.kind, SlotKind::Pad)).count();
    assert_eq!(pads, 2);
}
