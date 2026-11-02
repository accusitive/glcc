
use cranelift_codegen::ir::types::{self, Type};
use cranelift_codegen::ir::{
    AbiParam, ArgumentExtension, ArgumentPurpose, Signature,
};
use cranelift_module::{FuncId, Linkage, Module};

use glcc_mir as mir;

use crate::abi::{Arch, Os, Target};
use super::*;

/// Lazily declared runtime imports: the x86-64 foreign-variadic
/// pair (the callee address goes into TLS `__glccrt_va_target` and the
/// naked `__glccrt_jmp_va` tail-jumps to it with `%al = 8`), plus the
/// x87 `long double` helpers when the module touches `F80` (other
/// targets never see `F80`: their `long double` lowers to `F128`).
#[derive(Default)]
pub(crate) struct Rt {
    pub(crate) va_target: Option<cranelift_module::DataId>,
    pub(crate) jmp_va: Option<FuncId>,
    pub(crate) f80: Option<F80Rt>,
    pub(crate) quad: Option<QuadRt>,
    pub(crate) quad_direct: Option<QuadDirectRt>,
}

/// x86-64 `long double` (80-bit extended) runtime helpers. Values
/// cross as addresses under the out-pointer convention: no Cranelift
/// FP op may touch the F128 carrier, so even moves through helpers
/// stay bitwise.
pub(crate) struct F80Rt {
    pub(crate) add: FuncId,
    pub(crate) sub: FuncId,
    pub(crate) mul: FuncId,
    pub(crate) div: FuncId,
    pub(crate) neg: FuncId,
    pub(crate) ceil: FuncId,
    pub(crate) floor: FuncId,
    pub(crate) sqrt: FuncId,
    pub(crate) eq: FuncId,
    pub(crate) ne: FuncId,
    pub(crate) lt: FuncId,
    pub(crate) le: FuncId,
    pub(crate) gt: FuncId,
    pub(crate) ge: FuncId,
    pub(crate) to_f32: FuncId,
    pub(crate) to_f64: FuncId,
    pub(crate) from_f32: FuncId,
    pub(crate) from_f64: FuncId,
    pub(crate) to_i32: FuncId,
    pub(crate) to_i64: FuncId,
    pub(crate) to_u64: FuncId,
    /// Quad widening (`__extendxftf2`, exact): x87 value by stack,
    /// quad result in XMM.
    pub(crate) extend: FuncId,
    /// Quad narrowing (`__trunctfxf2`, single rounding): quad in XMM,
    /// x87 result in st(0), captured like a foreign return.
    pub(crate) trunc: FuncId,
    pub(crate) from_i32: FuncId,
    pub(crate) from_u32: FuncId,
    pub(crate) from_i64: FuncId,
    pub(crate) from_u64: FuncId,
    /// int128 -> x87 (`__glcc_floattixf`, `__glcc_floatuntixf` from
    /// the vendored runtime): 128-bit pair in, x87 value in st(0).
    pub(crate) from_i128: FuncId,
    pub(crate) from_u128: FuncId,
    /// x87 -> int128 (`__glcc_fixxfti`, `__glcc_fixunsxfti`): x87
    /// value by stack, 128-bit pair back in registers, matching
    /// clang exactly (including its out-of-range results).
    pub(crate) to_i128: FuncId,
    pub(crate) to_u128: FuncId,
    /// Pops the live `st(0)` a foreign `long double` callee left into
    /// memory (see the `emit_call` foreign-return path).
    pub(crate) getret: FuncId,
}

/// AArch64/RISC-V IEEE-quad (`F128`, their `long double` and
/// `__float128`) arithmetic trampolines (`rt/glccrt.c`): same
/// out-pointer convention as the x87 helpers, adapting to the
/// toolchain's `__*tf3` register convention
/// inside. Cranelift names no `F128` arithmetic, so these adapters
/// are the only way to reach it. Conversions, comparisons, negation,
/// and rounding inline as integer bit-twiddling in [`crate::quad`].
pub(crate) struct QuadRt {
    pub(crate) add: FuncId,
    pub(crate) sub: FuncId,
    pub(crate) mul: FuncId,
    pub(crate) div: FuncId,
    pub(crate) sqrt: FuncId,
}

/// x86-64 `__float128` arithmetic imports (`__glcc_addtf3` and friends
/// from the vendored soft-float runtime, `rt/softfloat`). Unlike the quad-target trampolines
/// above, these need no adapter: `F128` values already cross in XMM
/// registers on x86-64, exactly the `__*tf3` convention, so the
/// carrier passes straight through the call.
pub(crate) struct QuadDirectRt {
    pub(crate) add: FuncId,
    pub(crate) sub: FuncId,
    pub(crate) mul: FuncId,
    pub(crate) div: FuncId,
}

pub(crate) fn rt_sig(target: Target, params: Vec<Type>, returns: Vec<Type>) -> Signature {
    let mut sig = Signature::new(target.call_conv());
    for ty in params {
        sig.params.push(AbiParam::new(ty));
    }
    for ty in returns {
        sig.returns.push(AbiParam::new(ty));
    }
    sig
}

impl Rt {
    pub(crate) fn declare_if_needed<M: Module>(mir_module: &mir::Module, target: Target, clif: &mut M) -> Rt {
        let mut rt = Rt::default();
        if target.arch != Arch::X86_64 {
            if uses_f128(mir_module) {
                let mut import = |name: &str| {
                    clif.declare_function(name, Linkage::Import, &rt_sig(target, vec![types::I64, types::I64, types::I64], vec![]))
                        .unwrap_or_else(|_| panic!("cannot declare {name}"))
                };
                rt.quad = Some(QuadRt {
                    add: import("__glcc_quad_add"),
                    sub: import("__glcc_quad_sub"),
                    mul: import("__glcc_quad_mul"),
                    div: import("__glcc_quad_div"),
                    sqrt: clif
                        .declare_function("__glcc_quad_sqrt", Linkage::Import, &rt_sig(target, vec![types::I64, types::I64], vec![]))
                        .unwrap_or_else(|_| panic!("cannot declare __glcc_quad_sqrt")),
                });
            }
            return rt;
        }
        // The `%al` trampoline is a SysV x86-64 device: Win64 passes
        // variadics natively and needs no runtime support.
        let foreign_variadic = target.os != Os::Windows
            && mir_module.funcs.iter().any(|func| {
                func.insts.iter().any(|inst| match &inst.kind {
                    mir::InstKind::CallExtern { target, .. } => mir_module.externs[*target].sig.variadic,
                    // An indirect variadic call needs the trampoline too.
                    mir::InstKind::CallIndirect { sig, .. } => sig.variadic,
                    _ => false,
                })
            });
        if foreign_variadic {
            let va_target = clif.declare_data("__glccrt_va_target", Linkage::Import, true, true).expect("cannot declare va_target");
            let jmp_va = clif
                .declare_function("__glccrt_jmp_va", Linkage::Import, &cranelift_codegen::ir::Signature::new(target.call_conv()))
                .expect("cannot declare jmp_va");
            rt.va_target = Some(va_target);
            rt.jmp_va = Some(jmp_va);
        }
        if uses_f128(mir_module) {
            let mut import = |name: &str| {
                clif.declare_function(name, Linkage::Import, &rt_sig(target, vec![types::F128, types::F128], vec![types::F128]))
                    .unwrap_or_else(|_| panic!("cannot declare {name}"))
            };
            rt.quad_direct = Some(QuadDirectRt {
                add: import("__glcc_addtf3"),
                sub: import("__glcc_subtf3"),
                mul: import("__glcc_multf3"),
                div: import("__glcc_divtf3"),
            });
        }
        if uses_f80(mir_module) {
            let mut import = |name: &str, params: Vec<Type>, returns: Vec<Type>| {
                clif.declare_function(name, Linkage::Import, &rt_sig(target, params, returns))
                    .unwrap_or_else(|_| panic!("cannot declare {name}"))
            };
            let binop = || (vec![types::I64, types::I64, types::I64], vec![]);
            let (p, r) = binop();
            let add = import("__glccrt_f80_add", p.clone(), r.clone());
            let (p, r) = binop();
            let sub = import("__glccrt_f80_sub", p.clone(), r.clone());
            let (p, r) = binop();
            let mul = import("__glccrt_f80_mul", p.clone(), r.clone());
            let (p, r) = binop();
            let div = import("__glccrt_f80_div", p, r);
            let neg = import("__glccrt_f80_neg", vec![types::I64, types::I64], vec![]);
            let ceil = import("__glccrt_f80_ceil", vec![types::I64, types::I64], vec![]);
            let floor = import("__glccrt_f80_floor", vec![types::I64, types::I64], vec![]);
            let sqrt = import("__glccrt_f80_sqrt", vec![types::I64, types::I64], vec![]);
            let cmp = || (vec![types::I64, types::I64], vec![types::I32]);
            let (p, r) = cmp();
            let eq = import("__glccrt_f80_eq", p.clone(), r.clone());
            let (p, r) = cmp();
            let ne = import("__glccrt_f80_ne", p.clone(), r.clone());
            let (p, r) = cmp();
            let lt = import("__glccrt_f80_lt", p.clone(), r.clone());
            let (p, r) = cmp();
            let le = import("__glccrt_f80_le", p.clone(), r.clone());
            let (p, r) = cmp();
            let gt = import("__glccrt_f80_gt", p.clone(), r.clone());
            let (p, r) = cmp();
            let ge = import("__glccrt_f80_ge", p, r);
            let to_f32 = import("__glccrt_f80_to_f32", vec![types::I64], vec![types::F32]);
            let to_f64 = import("__glccrt_f80_to_f64", vec![types::I64], vec![types::F64]);
            let from_f32 = import("__glccrt_f80_from_f32", vec![types::I64, types::F32], vec![]);
            let from_f64 = import("__glccrt_f80_from_f64", vec![types::I64, types::F64], vec![]);
            let to_i32 = import("__glccrt_f80_to_i32", vec![types::I64], vec![types::I32]);
            let to_i64 = import("__glccrt_f80_to_i64", vec![types::I64], vec![types::I64]);
            let to_u64 = import("__glccrt_f80_to_u64", vec![types::I64], vec![types::I64]);
            // `__trunctfxf2` takes the quad in XMM and returns the
            // x87 value in st(0): no Clif return to declare, the
            // caller captures it through `getret` like foreign calls.
            let trunc = import("__glcc_trunctfxf2", vec![types::F128], vec![]);
            let from_i32 = import("__glccrt_f80_from_i32", vec![types::I64, types::I32], vec![]);
            let from_u32 = import("__glccrt_f80_from_u32", vec![types::I64, types::I32], vec![]);
            let from_i64 = import("__glccrt_f80_from_i64", vec![types::I64, types::I64], vec![]);
            let from_u64 = import("__glccrt_f80_from_u64", vec![types::I64, types::I64], vec![]);
            // int128 -> x87 takes the 128-bit pair in registers and
            // returns in st(0): no Clif return, captured via getret.
            let from_i128 = import("__glcc_floattixf", vec![types::I64, types::I64], vec![]);
            let from_u128 = import("__glcc_floatuntixf", vec![types::I64, types::I64], vec![]);
            let getret = import("__glccrt_f80_getret", vec![types::I64], vec![]);
            // x87 -> int128 takes the value the SysV way and returns
            // the pair in registers. Declared after the last use of
            // `import` above, whose closure borrows `clif`.
            let mut to_pair = |name: &str| {
                let mut sig = Signature::new(target.call_conv());
                sig.params.push(AbiParam {
                    value_type: types::I64,
                    purpose: ArgumentPurpose::StructArgument(16),
                    extension: ArgumentExtension::None,
                });
                sig.returns.push(AbiParam::new(types::I64));
                sig.returns.push(AbiParam::new(types::I64));
                clif.declare_function(name, Linkage::Import, &sig).unwrap_or_else(|_| panic!("cannot declare {name}"))
            };
            let to_i128 = to_pair("__glcc_fixxfti");
            let to_u128 = to_pair("__glcc_fixunsxfti");
            // `__extendxftf2` takes the x87 value the SysV way (16
            // bytes on the stack): same `StructArgument`
            // address-passing our own `long double` parameters use.
            // Declared after the last use of `import` above, whose
            // closure borrows `clif`.
            let extend = {
                let mut sig = Signature::new(target.call_conv());
                sig.params.push(AbiParam {
                    value_type: types::I64,
                    purpose: ArgumentPurpose::StructArgument(16),
                    extension: ArgumentExtension::None,
                });
                sig.returns.push(AbiParam::new(types::F128));
                clif.declare_function("__glcc_extendxftf2", Linkage::Import, &sig)
                    .expect("cannot declare __glcc_extendxftf2")
            };
            rt.f80 = Some(F80Rt {
                add,
                sub,
                mul,
                div,
                neg,
                ceil,
                floor,
                sqrt,
                eq,
                ne,
                lt,
                le,
                gt,
                ge,
                to_f32,
                to_f64,
                from_f32,
                from_f64,
                to_i32,
                to_i64,
                to_u64,
                extend,
                trunc,
                from_i32,
                from_u32,
                from_i64,
                from_u64,
                from_i128,
                from_u128,
                to_i128,
                to_u128,
                getret,
            });
        }
        rt
    }
}

