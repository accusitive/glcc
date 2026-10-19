use std::collections::{BTreeMap, HashMap};

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::immediates::{Ieee128, Offset32};
use cranelift_codegen::ir::types::{self, Type};
use cranelift_codegen::ir::{
    AbiParam, ArgumentExtension, ArgumentPurpose, AtomicRmwOp, BlockArg, ConstantData, FuncRef, InstBuilder, MemFlagsData, Signature, SourceLoc, StackSlot, StackSlotData,
    StackSlotKind, TrapCode,
};
use cranelift_codegen::ir::{Block as ClifBlock, Value as ClifValue};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch};
use cranelift_module::{DataDescription, FuncId, Linkage, Module};

use glcc_mir as mir;

use crate::abi::{self, Arch, Os, SigLayout, Target, VaCtx};
use crate::memflags;
use crate::ty::clif_ty;

/// Windows x64 unwind record for one defined function: the serialized
/// `UNWIND_INFO` (`.xdata` content) plus the function size its
/// `.pdata` `RUNTIME_FUNCTION` entry needs. The driver's `winunwind`
/// module lays these out as sections; other targets collect none.
pub struct WinUnwindEntry {
    pub func: FuncId,
    /// Function size in bytes: begin address + size is the RUNTIME_FUNCTION
    /// end address.
    pub size: u32,
    /// Serialized UNWIND_INFO, 4-byte aligned sections entries.
    pub xdata: Vec<u8>,
}

/// A 32-bit (`__ptr32`) address patch in a global's image. The
/// Cranelift data machinery only emits pointer-width relocations, so
/// these are reported to the object driver, which patches the emitted
/// object with a 4-byte absolute relocation at `offset` within the
/// object `container`. `addend` is a byte displacement.
pub struct NarrowReloc {
    pub container: cranelift_module::DataId,
    pub offset: u64,
    pub target: NarrowTarget,
    pub addend: i64,
}

/// The symbol a [`NarrowReloc`] resolves to.
pub enum NarrowTarget {
    Data(cranelift_module::DataId),
    Func(FuncId),
}

/// Lower a whole MIR module into a Cranelift module for `target`,
/// defining every function and global. Returns the Clif id of each MIR
/// function.
///
/// Linkage follows the IR flags: `local` functions and globals emit
/// `Local` (internal linkage, anonymous objects), weak definitions
/// emit `Preemptible` (ELF `STB_WEAK`), the rest `Export`. Tests need
/// stable symbol names either way.
pub fn lower_module<M: Module>(mir_module: &mir::Module, clif: &mut M, target: Target) -> HashMap<mir::Func, FuncId> {
    let mut asm_text = String::new();
    lower_module_impl(mir_module, clif, target, &mut asm_text, None, None).0
}

/// [`lower_module`], also appending the generated inline-asm wrapper
/// text to `asm_text`. The driver assembles it into the TU's `.s`.
/// Returns the 32-bit (`__ptr32`) address patches the object driver
/// must apply itself.
pub fn lower_module_collecting<M: Module>(
    mir_module: &mir::Module,
    clif: &mut M,
    target: Target,
    asm_text: &mut String,
    unwind: &mut Vec<WinUnwindEntry>,
) -> Vec<NarrowReloc> {
    lower_module_impl(mir_module, clif, target, asm_text, None, Some(unwind)).1
}

/// [`lower_module_collecting`], also collecting the debug records the
/// DWARF writer needs: machine sizes and source maps of every defined
/// function, frame-relative variable homes, and module-level objects.
pub fn lower_module_with_debug<M: Module>(
    mir_module: &mir::Module,
    clif: &mut M,
    target: Target,
    asm_text: &mut String,
    unwind: &mut Vec<WinUnwindEntry>,
) -> (HashMap<mir::Func, FuncId>, glcc_dwarf::LoweredDebug, Vec<NarrowReloc>) {
    let mut debug = glcc_dwarf::LoweredDebug::new();
    let (ids, narrow) = lower_module_impl(mir_module, clif, target, asm_text, Some(&mut debug), Some(unwind));
    (ids, debug, narrow)
}

fn lower_module_impl<M: Module>(
    mir_module: &mir::Module,
    clif: &mut M,
    target: Target,
    asm_text: &mut String,
    mut debug: Option<&mut glcc_dwarf::LoweredDebug>,
    mut unwind: Option<&mut Vec<WinUnwindEntry>>,
) -> (HashMap<mir::Func, FuncId>, Vec<NarrowReloc>) {
    let mut wrappers = Wrappers::default();
    let mut layouts = HashMap::new();
    for (id, func) in mir_module.funcs.iter_enumerated() {
        layouts.insert(id, abi::layout_sig(target, &func.sig.params, &func.sig.ret, func.sig.variadic));
    }
    let mut ids = HashMap::new();
    for (id, func) in mir_module.funcs.iter_enumerated() {
        let sig = abi::clif_signature(&layouts[&id], target.call_conv());
        let linkage = definition_linkage(func.local, func.weak);
        let clif_id = clif.declare_function(&func.name, linkage, &sig).unwrap_or_else(|_| panic!("cannot declare {}", func.name));
        ids.insert(id, clif_id);
    }
    // Imports for declared externs use plain named-only signatures;
    // each call site builds its own signature from actuals. The
    // call-site layout still records variadic-ness on x86-64: variadic
    // foreign calls must route through the `%al` trampoline (see
    // `emit_call`), which only a variadic layout triggers.
    //
    // Only imports the module actually references are declared: an
    // unused declaration would leave an undefined object-file symbol
    // behind, and MSVC's linker rejects undefined COFF symbols outright.
    let mut used_externs = std::collections::HashSet::new();
    let mut used_globals = std::collections::HashSet::new();
    for func in mir_module.funcs.iter() {
        for inst in &func.insts {
            match &inst.kind {
                mir::InstKind::CallExtern { target, .. } => {
                    used_externs.insert(*target);
                }
                mir::InstKind::ExternAddr { target } => {
                    used_externs.insert(*target);
                }
                mir::InstKind::GlobalAddr { global } => {
                    used_globals.insert(*global);
                }
                _ => {}
            }
        }
    }
    for global in mir_module.globals.iter() {
        for (_, reloc) in global.relocs.iter().chain(&global.relocs32) {
            match reloc {
                mir::DataReloc::Global { target, .. } => {
                    used_globals.insert(*target);
                }
                mir::DataReloc::FuncExtern { target } => {
                    used_externs.insert(*target);
                }
                mir::DataReloc::Func { .. } => {}
            }
        }
    }
    let mut extern_ids = HashMap::new();
    let mut extern_layouts = HashMap::new();
    for (id, ext) in mir_module.externs.iter_enumerated() {
        if !used_externs.contains(&id) {
            continue;
        }
        let layout = abi::layout_sig(target, &ext.sig.params, &ext.sig.ret, false);
        let sig = abi::clif_signature(&layout, target.call_conv());
        let clif_id = clif
            .declare_function(&ext.name, import_linkage(ext.weak), &sig)
            .unwrap_or_else(|_| panic!("cannot declare extern {}", ext.name));
        extern_ids.insert(id, clif_id);
        let call_variadic = ext.sig.variadic && target.arch == Arch::X86_64;
        extern_layouts.insert(id, abi::layout_sig(target, &ext.sig.params, &ext.sig.ret, call_variadic));
    }
    let mut data_ids = HashMap::new();
    // Thread-local globals address through `tls_value`, not
    // `symbol_value`; collecting them once keeps `emit_inst` from
    // re-scanning the module for every `GlobalAddr`.
    let mut tls_data = std::collections::HashSet::new();
    for (id, global) in mir_module.globals.iter_enumerated() {
        // Declare all data first: relocations may point forward. Unused
        // extern declarations stay undeclared for the same reason as
        // unused function imports.
        if global.external && !used_globals.contains(&id) {
            continue;
        }
        data_ids.insert(id, declare_global(global, clif));
        if global.tls {
            tls_data.insert(id);
        }
    }
    let mut narrow = Vec::new();
    for (id, global) in mir_module.globals.iter_enumerated() {
        if let Some(&did) = data_ids.get(&id) {
            define_global(global, did, &data_ids, &ids, &extern_ids, clif, &mut narrow);
        }
    }
    if let Some(debug) = debug.as_deref_mut() {
        for (id, global) in mir_module.globals.iter_enumerated() {
            crate::debug::collect_global_debug(debug, global, data_ids.get(&id).copied());
        }
    }
    // x86-64 foreign variadic calls route through the runtime
    // trampoline; declare its symbols only when such a call exists.
    let rt = Rt::declare_if_needed(mir_module, target, clif);
    let memcpy_id = mir_module
        .funcs
        .iter()
        .any(needs_libc_memcpy)
        .then(|| {
            // A user prototype may describe a different signature. In that
            // case fall back to the inline copy rather than conflicting
            // with the user's declaration.
            clif.declare_function("memcpy", Linkage::Import, &rt_sig(target, vec![types::I64, types::I64, types::I64], vec![types::I64]))
                .ok()
        })
        .flatten();
    for (id, func) in mir_module.funcs.iter_enumerated() {
        lower_function(
            id,
            func,
            target,
            &layouts,
            &extern_layouts,
            &ids,
            &extern_ids,
            &data_ids,
            &tls_data,
            &rt,
            memcpy_id,
            clif,
            &mut wrappers,
            debug.as_deref_mut(),
            unwind.as_deref_mut(),
        );
    }
    asm_text.push_str(&wrappers.text);
    (ids, narrow)
}

fn is_large_aggregate(ty: &mir::Ty) -> bool {
    ty.is_aggregate() && ty.size_bytes() >= LIBC_COPY_THRESHOLD
}

fn needs_libc_memcpy(func: &mir::Function) -> bool {
    if is_large_aggregate(&func.sig.ret) || func.sig.params.iter().any(is_large_aggregate) {
        return true;
    }
    func.insts.iter().any(|inst| match &inst.kind {
        mir::InstKind::Memcpy { len, .. } => *len >= LIBC_COPY_THRESHOLD,
        mir::InstKind::Load { .. } => inst.result.is_some_and(|v| is_large_aggregate(&func.values[v].ty)),
        mir::InstKind::Store { val, .. } => is_large_aggregate(&func.values[*val].ty),
        mir::InstKind::InsertValue { agg, .. } => is_large_aggregate(&func.values[*agg].ty),
        _ => false,
    })
}

/// Inline-asm wrapper text collected while lowering one module,
/// deduplicated by symbol: the wrapper name is content-addressed, so
/// identical blocks share one definition.
#[derive(Default)]
pub struct Wrappers {
    seen: std::collections::HashSet<String>,
    text: String,
}

impl Wrappers {
    fn add(&mut self, name: &str, text: &str) {
        if self.seen.insert(name.to_string()) {
            self.text.push_str(text);
        }
    }
}

/// Cranelift linkage of one MIR definition: internal stays `Local`,
/// a weak definition is `Preemptible` (ELF `STB_WEAK`: a strong
/// definition in another TU wins, duplicates resolve), everything
/// else exports.
fn definition_linkage(local: bool, weak: bool) -> Linkage {
    if local {
        Linkage::Local
    } else if weak {
        Linkage::Preemptible
    } else {
        Linkage::Export
    }
}

/// Cranelift linkage of one undefined import: a weak declaration is
/// `Preemptible` too, which object marks `STB_WEAK`, so the linker
/// resolves it to zero when nothing defines it (GCC's behavior for
/// `extern ... __attribute__((weak))`).
fn import_linkage(weak: bool) -> Linkage {
    if weak { Linkage::Preemptible } else { Linkage::Import }
}

/// Lazily declared runtime imports: the x86-64 foreign-variadic
/// pair (the callee address goes into TLS `__glccrt_va_target` and the
/// naked `__glccrt_jmp_va` tail-jumps to it with `%al = 8`), plus the
/// x87 `long double` helpers when the module touches `F80` (other
/// targets never see `F80`: their `long double` lowers to `F128`).
#[derive(Default)]
struct Rt {
    va_target: Option<cranelift_module::DataId>,
    jmp_va: Option<FuncId>,
    f80: Option<F80Rt>,
    quad: Option<QuadRt>,
    quad_direct: Option<QuadDirectRt>,
}

/// x86-64 `long double` (80-bit extended) runtime helpers. Values
/// cross as addresses under the out-pointer convention: no Cranelift
/// FP op may touch the F128 carrier, so even moves through helpers
/// stay bitwise.
struct F80Rt {
    add: FuncId,
    sub: FuncId,
    mul: FuncId,
    div: FuncId,
    neg: FuncId,
    ceil: FuncId,
    floor: FuncId,
    sqrt: FuncId,
    eq: FuncId,
    ne: FuncId,
    lt: FuncId,
    le: FuncId,
    gt: FuncId,
    ge: FuncId,
    to_f32: FuncId,
    to_f64: FuncId,
    from_f32: FuncId,
    from_f64: FuncId,
    to_i32: FuncId,
    to_i64: FuncId,
    to_u64: FuncId,
    /// Quad widening (`__extendxftf2`, exact): x87 value by stack,
    /// quad result in XMM.
    extend: FuncId,
    /// Quad narrowing (`__trunctfxf2`, single rounding): quad in XMM,
    /// x87 result in st(0), captured like a foreign return.
    trunc: FuncId,
    from_i32: FuncId,
    from_u32: FuncId,
    from_i64: FuncId,
    from_u64: FuncId,
    /// int128 -> x87 (`__glcc_floattixf`, `__glcc_floatuntixf` from
    /// the vendored runtime): 128-bit pair in, x87 value in st(0).
    from_i128: FuncId,
    from_u128: FuncId,
    /// x87 -> int128 (`__glcc_fixxfti`, `__glcc_fixunsxfti`): x87
    /// value by stack, 128-bit pair back in registers, matching
    /// clang exactly (including its out-of-range results).
    to_i128: FuncId,
    to_u128: FuncId,
    /// Pops the live `st(0)` a foreign `long double` callee left into
    /// memory (see the `emit_call` foreign-return path).
    getret: FuncId,
}

/// AArch64/RISC-V IEEE-quad (`F128`, their `long double` and
/// `__float128`) arithmetic trampolines (`rt/glccrt.c`): same
/// out-pointer convention as the x87 helpers, adapting to the
/// toolchain's `__*tf3` register convention
/// inside. Cranelift names no `F128` arithmetic, so these adapters
/// are the only way to reach it. Conversions, comparisons, negation,
/// and rounding inline as integer bit-twiddling in [`crate::quad`].
struct QuadRt {
    add: FuncId,
    sub: FuncId,
    mul: FuncId,
    div: FuncId,
    sqrt: FuncId,
}

/// x86-64 `__float128` arithmetic imports (`__glcc_addtf3` and friends
/// from the vendored soft-float runtime, `rt/softfloat`). Unlike the quad-target trampolines
/// above, these need no adapter: `F128` values already cross in XMM
/// registers on x86-64, exactly the `__*tf3` convention, so the
/// carrier passes straight through the call.
struct QuadDirectRt {
    add: FuncId,
    sub: FuncId,
    mul: FuncId,
    div: FuncId,
}

fn rt_sig(target: Target, params: Vec<Type>, returns: Vec<Type>) -> Signature {
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
    fn declare_if_needed<M: Module>(mir_module: &mir::Module, target: Target, clif: &mut M) -> Rt {
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

/// Whether any `F80` value flows through the module: signatures,
/// values, allocation points, and global storage. Over-approximating
/// is harmless (unused imports emit no relocations), under-
/// approximating would leave helper calls undeclared.
fn uses_f80(mir_module: &mir::Module) -> bool {
    fn ty_uses(ty: &mir::Ty) -> bool {
        match ty {
            mir::Ty::Float(mir::FloatKind::F80) => true,
            mir::Ty::Ptr { pointee, .. } => ty_uses(pointee),
            mir::Ty::Array { elem, .. } => ty_uses(elem),
            mir::Ty::Struct { fields, .. } => fields.iter().any(ty_uses),
            _ => false,
        }
    }
    mir_module.globals.iter().any(|g| ty_uses(&g.ty))
        || mir_module
            .funcs
            .iter()
            .any(|f| f.sig.params.iter().any(ty_uses) || ty_uses(&f.sig.ret) || f.values.iter().any(|v| ty_uses(&v.ty)))
}

/// Whether any `F128` value flows through the module (same shapes as
/// [`uses_f80`]): gates the quad-trampoline imports on non-x86-64
/// targets, whose `long double` lowers to `F128`.
fn uses_f128(mir_module: &mir::Module) -> bool {
    fn ty_uses(ty: &mir::Ty) -> bool {
        match ty {
            mir::Ty::Float(mir::FloatKind::F128) => true,
            mir::Ty::Ptr { pointee, .. } => ty_uses(pointee),
            mir::Ty::Array { elem, .. } => ty_uses(elem),
            mir::Ty::Struct { fields, .. } => fields.iter().any(ty_uses),
            _ => false,
        }
    }
    mir_module.globals.iter().any(|g| ty_uses(&g.ty))
        || mir_module
            .funcs
            .iter()
            .any(|f| f.sig.params.iter().any(ty_uses) || ty_uses(&f.sig.ret) || f.values.iter().any(|v| ty_uses(&v.ty)))
}

fn declare_global<M: Module>(global: &mir::GlobalData, clif: &mut M) -> cranelift_module::DataId {
    // Imports name storage defined elsewhere: declare without an image.
    if global.external {
        return clif
            .declare_data(&global.name, import_linkage(global.weak), true, global.tls)
            .unwrap_or_else(|_| panic!("cannot declare extern data {}", global.name));
    }
    let linkage = definition_linkage(global.local, global.weak);
    clif.declare_data(&global.name, linkage, true, global.tls)
        .unwrap_or_else(|_| panic!("cannot declare data {}", global.name))
}

/// Define one declared global: fold the constant image, patch address
/// relocations over it, then apply the declared alignment (every C
/// object's natural `_Alignof`, so `double` never lands in a
/// byte-aligned slot) and any custom section (`.init_array`).
fn define_global<M: Module>(
    global: &mir::GlobalData,
    id: cranelift_module::DataId,
    data_ids: &HashMap<mir::Global, cranelift_module::DataId>,
    func_ids: &HashMap<mir::Func, FuncId>,
    extern_ids: &HashMap<mir::Extern, FuncId>,
    clif: &mut M,
    narrow: &mut Vec<NarrowReloc>,
) {
    if global.external {
        return;
    }
    // Zero-sized objects (empty structs, `T x[0]` members) still need a
    // symbol; one byte of storage keeps Cranelift's data machinery
    // happy without observable effect.
    let size = global.ty.size_bytes().max(1);
    let mut bytes = vec![0u8; size as usize];
    if let Some(init) = &global.init {
        write_const(&mut bytes, 0, init);
    }
    let mut desc = DataDescription::new();
    // An all-zero object with no custom section is uninitialized storage:
    // defining it as such keeps it out of the object file. (`.init_array`
    // and friends must stay PROGBITS even when empty.)
    if global.section.is_none() && global.relocs.is_empty() && global.relocs32.is_empty() && bytes.iter().all(|b| *b == 0) {
        desc.define_zeroinit(bytes.len());
    } else {
        desc.define(bytes.into_boxed_slice());
    }
    if let Some(align) = global.align {
        desc.set_align(align);
    }
    if let Some(section) = &global.section {
        desc.set_custom_section(section);
    }
    for (off, reloc) in &global.relocs {
        match reloc {
            mir::DataReloc::Global { target, addend } => {
                let gv = clif.declare_data_in_data(data_ids[target], &mut desc);
                desc.write_data_addr(*off as u32, gv, *addend);
            }
            mir::DataReloc::Func { target } => {
                let fr = clif.declare_func_in_data(func_ids[target], &mut desc);
                desc.write_function_addr(*off as u32, fr);
            }
            mir::DataReloc::FuncExtern { target } => {
                let fr = clif.declare_func_in_data(extern_ids[target], &mut desc);
                desc.write_function_addr(*off as u32, fr);
            }
        }
    }
    // 32-bit patches stay out of the descriptor: Cranelift would emit a
    // pointer-width relocation and clobber the neighbouring bytes. The
    // object driver adds a 4-byte relocation over the zeroed image.
    for (off, reloc) in &global.relocs32 {
        let (target, addend) = match reloc {
            mir::DataReloc::Global { target, addend } => (NarrowTarget::Data(data_ids[target]), *addend),
            mir::DataReloc::Func { target } => (NarrowTarget::Func(func_ids[target]), 0),
            mir::DataReloc::FuncExtern { target } => (NarrowTarget::Func(extern_ids[target]), 0),
        };
        narrow.push(NarrowReloc {
            container: id,
            offset: *off,
            target,
            addend,
        });
    }
    clif.define_data(id, &desc).expect("cannot define data");
}

fn write_const(bytes: &mut [u8], off: usize, c: &mir::Const) {
    match c {
        mir::Const::Int { bits, value } => {
            let len = ((*bits).max(8) / 8) as usize;
            bytes[off..off + len].copy_from_slice(&value.to_le_bytes()[..len]);
        }
        mir::Const::Float { kind, bits } => {
            let len = match kind {
                mir::FloatKind::F32 => 4,
                mir::FloatKind::F64 => 8,
                mir::FloatKind::F80 | mir::FloatKind::F128 => 16,
            };
            bytes[off..off + len].copy_from_slice(&bits.to_le_bytes()[..len]);
        }
        mir::Const::Null => {}
        mir::Const::Undef => {}
        // Zeroinit: the image starts zeroed.
        mir::Const::Zero => {}
        mir::Const::Bytes(content) => {
            let end = off + content.len();
            if end > bytes.len() {
                panic!("bytes constant overruns its image");
            }
            bytes[off..end].copy_from_slice(content);
        }
    }
}

struct Body<'m> {
    func: &'m mir::Function,
    target: Target,
    layout: SigLayout,
    /// Every non-constant MIR value, resolved: scalars to Clif values,
    /// aggregates to the address of their stack slot.
    values: HashMap<mir::Value, ClifValue>,
    /// Constants already materialized in each Clif block. Constants
    /// have no defining block, so a single Clif definition would not
    /// dominate uses in sibling blocks; rebuilding per block keeps
    /// every definition and its uses in the same block.
    consts: HashMap<(ClifBlock, mir::Value), ClifValue>,
    blocks: HashMap<mir::Block, ClifBlock>,
    sret: Option<ClifValue>,
    /// The libc `memcpy` import, present when this function contains a
    /// copy large enough to call it.
    memcpy: Option<FuncRef>,
    /// Variadic prelude state, present exactly in variadic definitions.
    va: Option<VaCtx>,
    /// Where the Clif builder is positioned, to avoid redundant
    /// switches (switching away from an unfilled block panics).
    current: Option<ClifBlock>,
    /// Stack slot per `Alloca` result value, for resolving variable
    /// homes once the frame layout is known.
    slots: HashMap<mir::Value, StackSlot>,
    /// Debug sink while lowering with `-g`: interns instruction source
    /// locations and receives the function record after `define_function`.
    debug: Option<&'m mut glcc_dwarf::LoweredDebug>,
}

fn lower_function<M: Module>(
    id: mir::Func,
    mir_func: &mir::Function,
    target: Target,
    layouts: &HashMap<mir::Func, SigLayout>,
    extern_layouts: &HashMap<mir::Extern, SigLayout>,
    func_ids: &HashMap<mir::Func, FuncId>,
    extern_ids: &HashMap<mir::Extern, FuncId>,
    data_ids: &HashMap<mir::Global, cranelift_module::DataId>,
    tls_data: &std::collections::HashSet<mir::Global>,
    rt: &Rt,
    memcpy_id: Option<FuncId>,
    clif: &mut M,
    wrappers: &mut Wrappers,
    debug: Option<&mut glcc_dwarf::LoweredDebug>,
    unwind: Option<&mut Vec<WinUnwindEntry>>,
) {
    let layout = layouts[&id].clone();
    let mut ctx = clif.make_context();
    ctx.func.signature = abi::clif_signature(&layout, target.call_conv());
    let memcpy = memcpy_id.map(|id| clif.declare_func_in_func(id, &mut ctx.func));
    let mut func_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

    let mut body = Body {
        func: mir_func,
        target,
        layout,
        values: HashMap::new(),
        consts: HashMap::new(),
        blocks: HashMap::new(),
        sret: None,
        memcpy,
        va: None,
        current: None,
        slots: HashMap::new(),
        debug,
    };
    for (block_id, _) in mir_func.blocks.iter_enumerated() {
        body.blocks.insert(block_id, builder.create_block());
    }
    let entry = body.blocks[&mir_func.entry];
    builder.append_block_params_for_function_params(entry);
    switch_to(&mut body, &mut builder, entry);
    map_entry_params(&mut body, &mut builder, mir_func, entry);
    // Non-entry block params become Clif block params up front so
    // forward branches transcribe without fixups.
    for (block_id, block) in mir_func.blocks.iter_enumerated() {
        if block_id == mir_func.entry {
            continue;
        }
        let clif_block = body.blocks[&block_id];
        for param in &block.params {
            let mir_ty = &mir_func.values[*param].ty;
            let param_ty = clif_ty(mir_ty).unwrap_or(types::I64);
            let clif_param = builder.append_block_param(clif_block, param_ty);
            body.values.insert(*param, clif_param);
        }
    }
    if body.layout.variadic {
        let va = emit_vararg_prelude(&mut body, &mut builder);
        body.va = Some(va);
    }
    for block_id in block_fill_order(mir_func) {
        let target_block = body.blocks[&block_id];
        switch_to(&mut body, &mut builder, target_block);
        lower_block(
            &mut body,
            &mut builder,
            clif,
            func_ids,
            extern_ids,
            data_ids,
            tls_data,
            rt,
            &layouts,
            &extern_layouts,
            block_id,
            wrappers,
        );
    }
    builder.seal_all_blocks();
    let config = clif.target_config();
    builder.finalize(config);
    let clif_id = func_ids[&id];
    if let Err(e) = clif.define_function(clif_id, &mut ctx) {
        eprintln!(
            "FAILED {}:
{}",
            mir_func.name,
            ctx.func.display()
        );
        panic!("cannot define function: {e:?}");
    }
    if let Some(debug) = body.debug.take() {
        if let Some(code) = ctx.compiled_code() {
            crate::debug::collect_function_debug(debug, clif_id, mir_func, &body.slots, code, data_ids, tls_data);
        }
    }
    if let Some(unwind) = unwind {
        if let Some(code) = ctx.compiled_code() {
            let info = code
                .create_unwind_info(clif.isa())
                .unwrap_or_else(|e| panic!("cannot create unwind info for {}: {e:?}", mir_func.name));
            if let Some(cranelift_codegen::isa::unwind::UnwindInfo::WindowsX64(info)) = info {
                let mut xdata = vec![0u8; info.emit_size()];
                info.emit(&mut xdata);
                unwind.push(WinUnwindEntry {
                    func: clif_id,
                    size: code.buffer.data().len() as u32,
                    xdata,
                });
            }
        }
    }
}

/// Order for filling blocks: reverse postorder over the reachable CFG,
/// then unreachable blocks in arena order. A definition dominates its
/// uses, and dominators precede in reverse postorder, so every operand
/// exists as a Clif value by the time its use is lowered — no matter
/// how passes ordered the arena.
fn block_fill_order(func: &mir::Function) -> Vec<mir::Block> {
    let reachable = func.reachable_blocks();
    let mut post = Vec::new();
    let mut seen = vec![false; func.blocks.len()];
    let mut work = vec![(func.entry, false)];
    seen[func.entry.index()] = true;
    while let Some((block, leaving)) = work.pop() {
        if leaving {
            post.push(block);
            continue;
        }
        work.push((block, true));
        if let Some(term) = &func.blocks[block].term {
            for succ in term.successors().iter().rev() {
                if reachable[succ.index()] && !seen[succ.index()] {
                    seen[succ.index()] = true;
                    work.push((*succ, false));
                }
            }
        }
    }
    post.reverse();
    post.extend((0..func.blocks.len()).map(mir::Block::from_usize).filter(|block| !reachable[block.index()]));
    post
}

/// Map entry block params to MIR values: scalars directly, aggregates
/// reassembled into fresh slots from their pieces, chunk groups stored
/// whole, and `StructArgument` addresses copied to locals.
fn map_entry_params(body: &mut Body, builder: &mut FunctionBuilder, mir_func: &mir::Function, entry: ClifBlock) {
    let params = builder.block_params(entry).to_vec();
    // Per-owner piece collection: (MIR type, [(offset, ClifValue)]).
    // `BTreeMap` keeps owner iteration in param order so stack-slot
    // creation and store emission are deterministic across runs
    // (`HashMap` iteration would permute them via its random seed).
    let mut pieces: BTreeMap<usize, (mir::Ty, Vec<(u64, ClifValue)>)> = BTreeMap::new();
    let mut addrs: BTreeMap<usize, ClifValue> = BTreeMap::new();
    let mut slot_idx = 0;
    for slot in body.layout.slots.clone() {
        match slot.kind {
            abi::SlotKind::Sret => {
                body.sret = Some(params[slot_idx]);
                slot_idx += 1;
            }
            abi::SlotKind::Pad => {
                slot_idx += 1;
            }
            abi::SlotKind::Value { owner, offset } => {
                let v = params[slot_idx];
                slot_idx += 1;
                let mir_ty = param_mir_ty(mir_func, owner);
                if !mir_ty.is_aggregate() {
                    // Scalars may still span slots (128-bit pairs);
                    // collect them for joining below.
                    pieces.entry(owner).or_insert_with(|| (mir_ty, Vec::new())).1.push((offset, v));
                    continue;
                }
                pieces.entry(owner).or_insert_with(|| (mir_ty, Vec::new())).1.push((offset, v));
            }
            abi::SlotKind::Chunk { owner, index } => {
                let v = params[slot_idx];
                slot_idx += 1;
                let mir_ty = param_mir_ty(mir_func, owner);
                pieces.entry(owner).or_insert_with(|| (mir_ty, Vec::new())).1.push((index * 8, v));
            }
            abi::SlotKind::StackAddr { owner } | abi::SlotKind::RefAddr { owner } => {
                let v = params[slot_idx];
                slot_idx += 1;
                addrs.insert(owner, v);
            }
        }
    }
    for (owner, (mir_ty, parts)) in &pieces {
        let Some(mir_param) = mir_func.blocks[mir_func.entry].params.get(*owner).copied() else {
            continue;
        };
        if !mir_ty.is_aggregate() {
            if parts.len() == 1 {
                // Narrow scalars arrive in a wider register whose upper
                // bits the caller leaves unspecified on every target
                // (the cover type is register-width, e.g. a `char` in
                // w0 on AAPCS64). Besides dropping caller residue, the
                // reduce keeps the -O0 entry spill (a declaration-sized
                // slot) from receiving an over-wide store.
                let v = narrow_to_declared(builder, parts[0].1, mir_ty);
                body.values.insert(mir_param, v);
            } else {
                // Scalar spanning slots (128-bit pairs) rejoins here.
                let mut sorted = parts.clone();
                sorted.sort_by_key(|(off, _)| *off);
                let joined = join_values(builder, &sorted, mir_ty);
                body.values.insert(mir_param, joined);
            }
            continue;
        }
        let slot = stack_slot(builder, mir_ty);
        let addr = builder.ins().stack_addr(types::I64, slot, 0);
        for (off, v) in parts {
            // Chunk stores cover whole 8-byte granules; piece stores
            // land at exact field offsets. The address is always an
            // integer; only the stored value carries its own width.
            builder.ins().stack_store(types::I64, *v, slot, *off as i32);
        }
        body.values.insert(mir_param, addr);
    }
    for (owner, addr) in &addrs {
        let Some(mir_param) = mir_func.blocks[mir_func.entry].params.get(*owner).copied() else {
            continue;
        };
        let mir_ty = param_mir_ty(mir_func, *owner);
        if mir_ty.is_aggregate() {
            let slot = stack_slot(builder, &mir_ty);
            let dst = builder.ins().stack_addr(types::I64, slot, 0);
            emit_copy(body, builder, dst, *addr, mir_ty.size_bytes());
            body.values.insert(mir_param, dst);
        } else {
            // Scalar behind an address (x86-64 `long double`): load
            // the bits; values stay uniformly bitwise downstream.
            let ty = clif_ty(&mir_ty).expect("scalar address needs a Clif type");
            let v = builder.ins().load(ty, memflags::notrap(), *addr, Offset32::new(0));
            body.values.insert(mir_param, v);
        }
    }
    // Zero-sized aggregate parameters cross as nothing and so have no
    // slot: bind them to a dummy address of their own, since MIR uses
    // (entry spills, address math) expect them defined.
    for (i, param) in mir_func.blocks[mir_func.entry].params.iter().enumerate() {
        if body.values.contains_key(param) {
            continue;
        }
        let mir_ty = param_mir_ty(mir_func, i);
        if mir_ty.is_aggregate() && mir_ty.size_bytes() == 0 {
            let slot = stack_slot(builder, &mir_ty);
            let addr = builder.ins().stack_addr(types::I64, slot, 0);
            body.values.insert(*param, addr);
        }
    }
}

/// MIR type of the `owner`-th signature parameter.
fn param_mir_ty(mir_func: &mir::Function, owner: usize) -> mir::Ty {
    mir_func.sig.params[owner].clone()
}

/// Emit the variadic prelude for a definition: spill register-passed
/// parameters and compute the overflow base into [`VaCtx`].
fn emit_vararg_prelude(body: &mut Body, builder: &mut FunctionBuilder) -> VaCtx {
    let params = builder.block_params(body.blocks[&body.func.entry]).to_vec();
    let paired: Vec<(ClifValue, abi::Slot)> = body.layout.slots.clone().into_iter().zip(params).map(|(slot, v)| (v, slot)).collect();
    abi::emit_prelude(body.target, builder, &paired, &body.layout)
}

fn lower_block<M: Module>(
    body: &mut Body,
    builder: &mut FunctionBuilder,
    clif: &mut M,
    func_ids: &HashMap<mir::Func, FuncId>,
    extern_ids: &HashMap<mir::Extern, FuncId>,
    data_ids: &HashMap<mir::Global, cranelift_module::DataId>,
    tls_data: &std::collections::HashSet<mir::Global>,
    rt: &Rt,
    layouts: &HashMap<mir::Func, SigLayout>,
    extern_layouts: &HashMap<mir::Extern, SigLayout>,
    id: mir::Block,
    wrappers: &mut Wrappers,
) {
    let block = &body.func.blocks[id];
    let insts = block.insts.clone();
    for inst in insts {
        emit_inst(body, builder, clif, func_ids, extern_ids, data_ids, tls_data, rt, layouts, extern_layouts, inst, wrappers);
    }
    let Some(term) = block.term.clone() else {
        panic!("unterminated MIR block");
    };
    emit_term(body, builder, term);
}

fn operand(body: &mut Body, builder: &mut FunctionBuilder, mir_value: mir::Value) -> ClifValue {
    if let Some(resolved) = body.values.get(&mir_value) {
        return *resolved;
    }
    let data = &body.func.values[mir_value];
    let mir::Def::Const(c) = &data.def else {
        panic!("MIR value used before definition: {:?} of type {:?} in {}", data.def, data.ty, body.func.name);
    };
    let c = c.clone();
    let ty = data.ty.clone();
    let block = body.current.expect("operand outside a block");
    if let Some(&cached) = body.consts.get(&(block, mir_value)) {
        return cached;
    }
    let lowered = materialize_const(builder, &c, &ty);
    body.consts.insert((block, mir_value), lowered);
    lowered
}

/// Operand value used as a machine address: a 32-bit pointer widens to
/// the target's 64-bit address width (sign-extended, MSVC's default for
/// `__ptr32`). Ordinary addresses pass through untouched.
fn addr_operand(body: &mut Body, builder: &mut FunctionBuilder, v: mir::Value) -> ClifValue {
    let val = operand(body, builder, v);
    // Only a 32-bit address widens; other address operands (a SysV
    // `va_list` tag) have no scalar Clif type and pass through.
    let narrow = match &body.func.values[v].ty {
        mir::Ty::Ptr { bits, .. } => *bits <= 32,
        mir::Ty::Int { bits } => (17..=32).contains(bits),
        _ => false,
    };
    if narrow { builder.ins().sextend(types::I64, val) } else { val }
}

/// One volatile scalar load. C requires every volatile access to touch
/// memory; this Cranelift (0.136) has no volatile memory flag, and its
/// alias analysis happily forwards a plain load from a dominating store
/// (or replaces it with an equal earlier load). Atomic load/store are
/// opaque to that analysis — it treats them as fences — so they are what
/// volatile accesses lower to. The ordering is stronger than C's
/// volatile asks for, which is a legal implementation of it.
///
/// 16-byte scalars (F80/F128 carriers, I128) have no atomic lowering on
/// every supported target, so those keep a plain access fenced on both
/// sides: no store reaches across the fence to feed the load and no
/// later load reaches back over it.
fn volatile_load(builder: &mut FunctionBuilder, ty: Type, addr: ClifValue) -> ClifValue {
    if let Some(int_ty) = atomic_ty(ty) {
        let raw = builder.ins().atomic_load(int_ty, MemFlagsData::new(), addr);
        if int_ty == ty { raw } else { builder.ins().bitcast(ty, MemFlagsData::new(), raw) }
    } else {
        builder.ins().fence();
        let value = builder.ins().load(ty, MemFlagsData::new(), addr, Offset32::new(0));
        builder.ins().fence();
        value
    }
}

/// The store counterpart of [`volatile_load`]: atomic for integer-like
/// scalars up to 64 bits (32/64-bit floats bitcast through an integer),
/// fenced plain store otherwise.
fn volatile_store(builder: &mut FunctionBuilder, ty: Type, val: ClifValue, addr: ClifValue) {
    if let Some(int_ty) = atomic_ty(ty) {
        let raw = if int_ty == ty { val } else { builder.ins().bitcast(int_ty, MemFlagsData::new(), val) };
        builder.ins().atomic_store(MemFlagsData::new(), raw, addr);
    } else {
        builder.ins().fence();
        builder.ins().store(MemFlagsData::new(), val, addr, Offset32::new(0));
        builder.ins().fence();
    }
}

/// The integer type an atomic Clif op can carry for a volatile scalar:
/// integers and pointers pass through, 32/64-bit floats go through an
/// integer of the same size. 128-bit scalars return `None` (no atomic
/// lowering is available on every supported target).
fn atomic_ty(ty: Type) -> Option<Type> {
    match ty {
        types::I8 | types::I16 | types::I32 | types::I64 => Some(ty),
        types::F32 => Some(types::I32),
        types::F64 => Some(types::I64),
        _ => None,
    }
}

/// A compile-time constant as a Clif value. Aggregate-typed values have
/// no constant form besides `undef`, which materializes as a fresh,
/// uninitialized slot whose address is returned.
fn materialize_const(builder: &mut FunctionBuilder, c: &mir::Const, ty: &mir::Ty) -> ClifValue {
    if ty.is_aggregate() {
        let slot = stack_slot(builder, ty);
        let addr = builder.ins().stack_addr(types::I64, slot, 0);
        match c {
            mir::Const::Bytes(content) => store_const_bytes(builder, addr, content),
            // Fresh slots hold garbage: zeroinit must clear explicitly.
            _ => zero_range(builder, addr, ty.size_bytes()),
        }
        return addr;
    }
    match c {
        mir::Const::Zero | mir::Const::Bytes(_) => {
            panic!("zero/bytes constants need an aggregate type")
        }
        mir::Const::Int { bits, value } => {
            let clif_ty = clif_ty(ty).expect("int constant needs a scalar type");
            if *bits > 64 {
                let raw = *value as u128;
                let lo = builder.ins().iconst(types::I64, (raw as u64) as i64);
                let hi = builder.ins().iconst(types::I64, (raw >> 64) as u64 as i64);
                builder.ins().iconcat(lo, hi)
            } else {
                builder.ins().iconst(clif_ty, *value as i64)
            }
        }
        mir::Const::Float { kind, bits } => match kind {
            mir::FloatKind::F32 => builder.ins().f32const(f32::from_bits(*bits as u32)),
            mir::FloatKind::F64 => builder.ins().f64const(f64::from_bits(*bits as u64)),
            mir::FloatKind::F80 | mir::FloatKind::F128 => {
                let data: ConstantData = Ieee128::with_bits(*bits).into();
                let handle = builder.func.dfg.constants.insert(data);
                builder.ins().f128const(handle)
            }
        },
        // Null is the zero address; undef lowers to zero. A producer
        // must not rely on undef bits either way.
        mir::Const::Null | mir::Const::Undef => {
            let clif_ty = clif_ty(ty).expect("null/undef needs a scalar type");
            zero_value(builder, clif_ty)
        }
    }
}

/// Zero bits of a scalar type. `iconst` only covers 8..64-bit
/// integers, so float and 128-bit forms need their own constants.
fn zero_value(builder: &mut FunctionBuilder, ty: Type) -> ClifValue {
    match ty {
        types::F32 => builder.ins().f32const(0.0),
        types::F64 => builder.ins().f64const(0.0),
        types::F128 => {
            let data: ConstantData = Ieee128::with_bits(0).into();
            let handle = builder.func.dfg.constants.insert(data);
            builder.ins().f128const(handle)
        }
        types::I128 => {
            let lo = builder.ins().iconst(types::I64, 0);
            let hi = builder.ins().iconst(types::I64, 0);
            builder.ins().iconcat(lo, hi)
        }
        _ => builder.ins().iconst(ty, 0),
    }
}

fn stack_slot(builder: &mut FunctionBuilder, ty: &mir::Ty) -> StackSlot {
    // Zero-sized objects (empty structs, `char x[0]`) still need a
    // distinct address, so they get a byte; Cranelift rejects 0-byte
    // slots.
    let size = ty.size_bytes().max(1);
    if size > u32::MAX as u64 {
        panic!("cannot materialize stack slot of {size} bytes");
    }
    builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, size as u32, 4))
}

/// Store a static byte image at an address, chunked like `memcpy`.
fn store_const_bytes(builder: &mut FunctionBuilder, dst: ClifValue, content: &[u8]) {
    let mut off = 0usize;
    for (chunk_ty, chunk) in [(types::I64, 8), (types::I32, 4), (types::I16, 2), (types::I8, 1)] {
        while off + chunk <= content.len() {
            let mut buf = [0u8; 8];
            buf[..chunk].copy_from_slice(&content[off..off + chunk]);
            let v = builder.ins().iconst(chunk_ty, i64::from_le_bytes(buf));
            builder.ins().store(memflags::trusted(), v, dst, Offset32::new(off as i32));
            off += chunk;
        }
    }
}

/// Clear a range with 8-byte splats plus byte remainder.
fn zero_range(builder: &mut FunctionBuilder, dst: ClifValue, len: u64) {
    let splat = builder.ins().iconst(types::I64, 0);
    let words = len / 8;
    for i in 0..words {
        builder.ins().store(memflags::trusted(), splat, dst, Offset32::new((i * 8) as i32));
    }
    let zero_byte = builder.ins().iconst(types::I8, 0);
    for i in (words * 8)..len {
        builder.ins().store(memflags::trusted(), zero_byte, dst, Offset32::new(i as i32));
    }
}
/// Byte copy as unrolled loads/stores. Lengths are static (struct and
/// array sizes), so chunking is fixed at lowering time.
pub(crate) fn emit_memcpy_pub(builder: &mut FunctionBuilder, dst: ClifValue, src: ClifValue, len: u64) {
    emit_memcpy(builder, dst, src, len)
}

/// Copies at least this large call libc `memcpy` instead of unrolling
/// one load/store pair per 16 bytes. Smaller copies keep the inline
/// sequence to avoid call overhead on common small aggregates.
const LIBC_COPY_THRESHOLD: u64 = 256;

fn emit_memcpy(builder: &mut FunctionBuilder, dst: ClifValue, src: ClifValue, len: u64) {
    let mut off = 0u64;
    for (chunk_ty, chunk) in [(types::F128, 16), (types::I64, 8), (types::I32, 4), (types::I16, 2), (types::I8, 1)] {
        while off + chunk <= len {
            let v = builder.ins().load(chunk_ty, memflags::notrap(), src, Offset32::new(off as i32));
            builder.ins().store(memflags::notrap(), v, dst, Offset32::new(off as i32));
            off += chunk;
        }
    }
}

fn emit_copy(body: &Body, builder: &mut FunctionBuilder, dst: ClifValue, src: ClifValue, len: u64) {
    if len >= LIBC_COPY_THRESHOLD {
        if let Some(memcpy) = body.memcpy {
            let len = builder.ins().iconst(types::I64, len as i64);
            builder.ins().call(memcpy, &[dst, src, len]);
            return;
        }
    }
    emit_memcpy(builder, dst, src, len);
}

/// Whether a MIR value is an x86-64 80-bit extended float: arithmetic
/// routes through the runtime helpers, everything else through
/// `fstp`/`fucomip` helpers or generic moves. (`F128` rides the same
/// carriers: its arithmetic goes through the quad helpers — vendored
/// soft-float on x86-64, trampolines on quad targets — and every other
/// `F128` op inlines through `quad.rs`).
fn is_f80(body: &Body, v: mir::Value) -> bool {
    matches!(body.func.values[v].ty, mir::Ty::Float(mir::FloatKind::F80))
}

/// Whether a MIR value is an IEEE-quad float: AArch64/RISC-V `long
/// double` and `__float128` everywhere.
fn is_f128(body: &Body, v: mir::Value) -> bool {
    matches!(body.func.values[v].ty, mir::Ty::Float(mir::FloatKind::F128))
}

/// Spill an `F80` scalar (an `F128` carrier value) to a fresh 16-byte
/// slot and return its address for the out-pointer convention.
fn f80_addr(body: &mut Body, builder: &mut FunctionBuilder, v: mir::Value) -> ClifValue {
    let x = operand(body, builder, v);
    let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
    builder.ins().stack_store(types::I64, x, slot, 0);
    builder.ins().stack_addr(types::I64, slot, 0)
}

/// Load the `F128` carrier back from a 16-byte helper out-slot.
fn f80_result(builder: &mut FunctionBuilder, out: ClifValue) -> ClifValue {
    builder.ins().load(types::F128, memflags::trusted(), out, Offset32::new(0))
}

/// One `F80` binary operation through its helper: both sides cross as
/// addresses, the result comes back through an out-pointer temp as an
/// `F128` carrier. Rounds once, like the conversions. Also serves the
/// quad trampolines, which share the out-pointer convention exactly.
fn f80_binop<M: Module>(body: &mut Body, builder: &mut FunctionBuilder, clif: &mut M, helper: FuncId, a: mir::Value, b: mir::Value) -> ClifValue {
    let a_addr = f80_addr(body, builder, a);
    let b_addr = f80_addr(body, builder, b);
    let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
    let out = builder.ins().stack_addr(types::I64, slot, 0);
    let callee = clif.declare_func_in_func(helper, builder.func);
    builder.ins().call(callee, &[out, a_addr, b_addr]);
    f80_result(builder, out)
}

/// One x86-64 `__float128` binary operation through the vendored runtime: both
/// carriers pass in XMM registers and the result comes back the same
/// way, so unlike [`f80_binop`] no memory round-trip is needed.
fn f128_direct_binop<M: Module>(body: &mut Body, builder: &mut FunctionBuilder, clif: &mut M, helper: FuncId, a: mir::Value, b: mir::Value) -> ClifValue {
    let x = operand(body, builder, a);
    let y = operand(body, builder, b);
    let callee = clif.declare_func_in_func(helper, builder.func);
    let inst = builder.ins().call(callee, &[x, y]);
    builder.inst_results(inst)[0]
}

/// Widen an `F80` value to IEEE-quad through the vendored runtime
/// (`__extendxftf2`, exact like the `fld` widenings): the source
/// crosses by value on the stack, the quad result returns in XMM as
/// an `F128` carrier directly.
fn f80_extend<M: Module>(body: &mut Body, builder: &mut FunctionBuilder, clif: &mut M, helper: FuncId, a: mir::Value) -> ClifValue {
    let a_addr = f80_addr(body, builder, a);
    let callee = clif.declare_func_in_func(helper, builder.func);
    let inst = builder.ins().call(callee, &[a_addr]);
    builder.inst_results(inst)[0]
}

/// Narrow a quad carrier to `F80` through the vendored runtime (`__glcc_trunctfxf2`,
/// single rounding like the `fstp` narrowings): the carrier rides
/// XMM, the extended result arrives in st(0) and is captured exactly
/// like a foreign `long double` return.
fn quad_trunc<M: Module>(body: &mut Body, builder: &mut FunctionBuilder, clif: &mut M, rt: &Rt, helper: FuncId, a: mir::Value) -> ClifValue {
    let x = operand(body, builder, a);
    let callee = clif.declare_func_in_func(helper, builder.func);
    builder.ins().call(callee, &[x]);
    foreign_f80_result(builder, clif, rt)
}

/// One `F80` comparison through its helper, narrowed to the `Bool`
/// (`I8`) result the `Fcmp` instruction carries.
fn f80_cmp<M: Module>(body: &mut Body, builder: &mut FunctionBuilder, clif: &mut M, helper: FuncId, a: mir::Value, b: mir::Value) -> ClifValue {
    let a_addr = f80_addr(body, builder, a);
    let b_addr = f80_addr(body, builder, b);
    let callee = clif.declare_func_in_func(helper, builder.func);
    let inst = builder.ins().call(callee, &[a_addr, b_addr]);
    let r = builder.inst_results(inst)[0];
    builder.ins().ireduce(types::I8, r)
}

/// Width in bits of an integer MIR type.
fn mir_int_bits(ty: &mir::Ty) -> u32 {
    match ty {
        mir::Ty::Bool => 1,
        mir::Ty::Int { bits } => *bits,
        _ => panic!("not an integer MIR type"),
    }
}

fn emit_inst<M: Module>(
    body: &mut Body,
    builder: &mut FunctionBuilder,
    clif: &mut M,
    func_ids: &HashMap<mir::Func, FuncId>,
    extern_ids: &HashMap<mir::Extern, FuncId>,
    data_ids: &HashMap<mir::Global, cranelift_module::DataId>,
    tls_data: &std::collections::HashSet<mir::Global>,
    rt: &Rt,
    layouts: &HashMap<mir::Func, SigLayout>,
    extern_layouts: &HashMap<mir::Extern, SigLayout>,
    inst: mir::Inst,
    wrappers: &mut Wrappers,
) {
    let data = body.func.insts[inst].clone();
    if let Some(debug) = body.debug.as_deref_mut() {
        builder.set_srcloc(SourceLoc::new(debug.intern_location(data.loc)));
    }
    let result_ty = data.result.map(|v| body.func.values[v].ty.clone());
    let lowered: Option<ClifValue> = match data.kind {
        mir::InstKind::Add(a, b) => {
            let x = operand(body, builder, a);
            let y = operand(body, builder, b);
            Some(builder.ins().iadd(x, y))
        }
        mir::InstKind::Sub(a, b) => {
            let x = operand(body, builder, a);
            let y = operand(body, builder, b);
            Some(builder.ins().isub(x, y))
        }
        mir::InstKind::Mul(a, b) => {
            let x = operand(body, builder, a);
            let y = operand(body, builder, b);
            Some(builder.ins().imul(x, y))
        }
        mir::InstKind::Udiv(a, b) => {
            let x = operand(body, builder, a);
            let y = operand(body, builder, b);
            Some(builder.ins().udiv(x, y))
        }
        mir::InstKind::Sdiv(a, b) => {
            let x = operand(body, builder, a);
            let y = operand(body, builder, b);
            if body.target.arch == Arch::X86_64 {
                // C's `/` must fault with SIGFPE on a zero divisor, but
                // Cranelift's `sdiv` traps with `ud2` (SIGILL) to keep
                // the hardware fault reserved for overflow. Dividing the
                // absolute values keeps the fault in hardware; the sign
                // is restored afterwards. `INT_MIN / -1` wraps here
                // instead of faulting, but that quotient is undefined
                // behavior anyway.
                let ty = builder.func.dfg.value_type(x);
                let sign = builder.ins().iconst(ty, i64::from(ty.bits()) - 1);
                let sa = builder.ins().sshr(x, sign);
                let sb = builder.ins().sshr(y, sign);
                let xa = builder.ins().bxor(x, sa);
                let ax = builder.ins().isub(xa, sa);
                let yb = builder.ins().bxor(y, sb);
                let ay = builder.ins().isub(yb, sb);
                let q = builder.ins().udiv(ax, ay);
                let s = builder.ins().bxor(sa, sb);
                let qs = builder.ins().bxor(q, s);
                Some(builder.ins().isub(qs, s))
            } else {
                Some(builder.ins().sdiv(x, y))
            }
        }
        mir::InstKind::Urem(a, b) => {
            let x = operand(body, builder, a);
            let y = operand(body, builder, b);
            Some(builder.ins().urem(x, y))
        }
        mir::InstKind::Srem(a, b) => {
            let x = operand(body, builder, a);
            let y = operand(body, builder, b);
            Some(builder.ins().srem(x, y))
        }
        mir::InstKind::And(a, b) => {
            let x = operand(body, builder, a);
            let y = operand(body, builder, b);
            Some(builder.ins().band(x, y))
        }
        mir::InstKind::Or(a, b) => {
            let x = operand(body, builder, a);
            let y = operand(body, builder, b);
            Some(builder.ins().bor(x, y))
        }
        mir::InstKind::Xor(a, b) => {
            let x = operand(body, builder, a);
            let y = operand(body, builder, b);
            Some(builder.ins().bxor(x, y))
        }
        mir::InstKind::Shl(a, b) => {
            let x = operand(body, builder, a);
            let y = operand(body, builder, b);
            Some(builder.ins().ishl(x, y))
        }
        mir::InstKind::LShr(a, b) => {
            let x = operand(body, builder, a);
            let y = operand(body, builder, b);
            Some(builder.ins().ushr(x, y))
        }
        mir::InstKind::AShr(a, b) => {
            let x = operand(body, builder, a);
            let y = operand(body, builder, b);
            Some(builder.ins().sshr(x, y))
        }
        mir::InstKind::Neg(a) => {
            let x = operand(body, builder, a);
            Some(builder.ins().ineg(x))
        }
        mir::InstKind::Not(a) => {
            let x = operand(body, builder, a);
            Some(builder.ins().bnot(x))
        }
        mir::InstKind::FAdd(a, b) => {
            if is_f80(body, a) {
                let f80 = rt.f80.as_ref().expect("f80 arithmetic needs x86-64");
                Some(f80_binop(body, builder, clif, f80.add, a, b))
            } else if is_f128(body, a) && body.target.arch != Arch::X86_64 {
                // Quad arithmetic through the toolchain trampolines,
                // which share the out-pointer convention.
                let quad = rt.quad.as_ref().expect("quad arithmetic needs trampolines");
                Some(f80_binop(body, builder, clif, quad.add, a, b))
            } else if is_f128(body, a) {
                // x86-64 `__float128` arithmetic through the vendored runtime (see
                // `QuadDirectRt`): only this case reaches here, since
                // quad targets route through trampolines above.
                let quad = rt.quad_direct.as_ref().expect("quad arithmetic needs the soft-float runtime");
                Some(f128_direct_binop(body, builder, clif, quad.add, a, b))
            } else {
                let x = operand(body, builder, a);
                let y = operand(body, builder, b);
                Some(builder.ins().fadd(x, y))
            }
        }
        mir::InstKind::FSub(a, b) => {
            if is_f80(body, a) {
                let f80 = rt.f80.as_ref().expect("f80 arithmetic needs x86-64");
                Some(f80_binop(body, builder, clif, f80.sub, a, b))
            } else if is_f128(body, a) && body.target.arch != Arch::X86_64 {
                // Quad arithmetic through the toolchain trampolines,
                // which share the out-pointer convention.
                let quad = rt.quad.as_ref().expect("quad arithmetic needs trampolines");
                Some(f80_binop(body, builder, clif, quad.sub, a, b))
            } else if is_f128(body, a) {
                // x86-64 `__float128` arithmetic through the vendored runtime (see
                // `QuadDirectRt`): only this case reaches here, since
                // quad targets route through trampolines above.
                let quad = rt.quad_direct.as_ref().expect("quad arithmetic needs the soft-float runtime");
                Some(f128_direct_binop(body, builder, clif, quad.sub, a, b))
            } else {
                let x = operand(body, builder, a);
                let y = operand(body, builder, b);
                Some(builder.ins().fsub(x, y))
            }
        }
        mir::InstKind::FMul(a, b) => {
            if is_f80(body, a) {
                let f80 = rt.f80.as_ref().expect("f80 arithmetic needs x86-64");
                Some(f80_binop(body, builder, clif, f80.mul, a, b))
            } else if is_f128(body, a) && body.target.arch != Arch::X86_64 {
                // Quad arithmetic through the toolchain trampolines,
                // which share the out-pointer convention.
                let quad = rt.quad.as_ref().expect("quad arithmetic needs trampolines");
                Some(f80_binop(body, builder, clif, quad.mul, a, b))
            } else if is_f128(body, a) {
                // x86-64 `__float128` arithmetic through the vendored runtime (see
                // `QuadDirectRt`): only this case reaches here, since
                // quad targets route through trampolines above.
                let quad = rt.quad_direct.as_ref().expect("quad arithmetic needs the soft-float runtime");
                Some(f128_direct_binop(body, builder, clif, quad.mul, a, b))
            } else {
                let x = operand(body, builder, a);
                let y = operand(body, builder, b);
                Some(builder.ins().fmul(x, y))
            }
        }
        mir::InstKind::FDiv(a, b) => {
            if is_f80(body, a) {
                let f80 = rt.f80.as_ref().expect("f80 arithmetic needs x86-64");
                Some(f80_binop(body, builder, clif, f80.div, a, b))
            } else if is_f128(body, a) && body.target.arch != Arch::X86_64 {
                // Quad arithmetic through the toolchain trampolines,
                // which share the out-pointer convention.
                let quad = rt.quad.as_ref().expect("quad arithmetic needs trampolines");
                Some(f80_binop(body, builder, clif, quad.div, a, b))
            } else if is_f128(body, a) {
                // x86-64 `__float128` arithmetic through the vendored runtime (see
                // `QuadDirectRt`): only this case reaches here, since
                // quad targets route through trampolines above.
                let quad = rt.quad_direct.as_ref().expect("quad arithmetic needs the soft-float runtime");
                Some(f128_direct_binop(body, builder, clif, quad.div, a, b))
            } else {
                let x = operand(body, builder, a);
                let y = operand(body, builder, b);
                Some(builder.ins().fdiv(x, y))
            }
        }
        mir::InstKind::FNeg(a) => {
            if is_f80(body, a) {
                // Through the helper's fchs: an F128 fneg would flip
                // the quad sign at bit 127 instead of bit 79.
                let f80 = rt.f80.as_ref().expect("f80 negation needs x86-64");
                let a_addr = f80_addr(body, builder, a);
                let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
                let out = builder.ins().stack_addr(types::I64, slot, 0);
                let callee = clif.declare_func_in_func(f80.neg, builder.func);
                builder.ins().call(callee, &[out, a_addr]);
                Some(f80_result(builder, out))
            } else if is_f128(body, a) {
                // Bit-127 flip through integer ops: no Clif fneg on
                // F128 anywhere (see `quad.rs`).
                let x = operand(body, builder, a);
                Some(crate::quad::quad_neg(builder, x))
            } else {
                let x = operand(body, builder, a);
                Some(builder.ins().fneg(x))
            }
        }
        mir::InstKind::Ceil(a) => {
            if is_f80(body, a) {
                // Through frndint with the control pinned round-up: an
                // F128 ceil would read quad fields, not x87 ones.
                let f80 = rt.f80.as_ref().expect("f80 ceil needs x86-64");
                let a_addr = f80_addr(body, builder, a);
                let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
                let out = builder.ins().stack_addr(types::I64, slot, 0);
                let callee = clif.declare_func_in_func(f80.ceil, builder.func);
                builder.ins().call(callee, &[out, a_addr]);
                Some(f80_result(builder, out))
            } else if is_f128(body, a) {
                // Truncate-and-adjust through integer ops (see
                // `quad.rs`): no Clif ceil on F128 anywhere.
                let x = operand(body, builder, a);
                Some(crate::quad::quad_ceil(builder, x))
            } else {
                let x = operand(body, builder, a);
                Some(builder.ins().ceil(x))
            }
        }
        mir::InstKind::Floor(a) => {
            if is_f80(body, a) {
                // Through frndint pinned round-down (see above).
                let f80 = rt.f80.as_ref().expect("f80 floor needs x86-64");
                let a_addr = f80_addr(body, builder, a);
                let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
                let out = builder.ins().stack_addr(types::I64, slot, 0);
                let callee = clif.declare_func_in_func(f80.floor, builder.func);
                builder.ins().call(callee, &[out, a_addr]);
                Some(f80_result(builder, out))
            } else if is_f128(body, a) {
                // Integer truncate-and-adjust (see `quad.rs`).
                let x = operand(body, builder, a);
                Some(crate::quad::quad_floor(builder, x))
            } else {
                let x = operand(body, builder, a);
                Some(builder.ins().floor(x))
            }
        }
        mir::InstKind::Sqrt(a) => {
            if is_f80(body, a) {
                // Through fsqrt: an F128 sqrt would read quad fields,
                // not x87 ones.
                let f80 = rt.f80.as_ref().expect("f80 sqrt needs x86-64");
                let a_addr = f80_addr(body, builder, a);
                let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
                let out = builder.ins().stack_addr(types::I64, slot, 0);
                let callee = clif.declare_func_in_func(f80.sqrt, builder.func);
                builder.ins().call(callee, &[out, a_addr]);
                Some(f80_result(builder, out))
            } else if is_f128(body, a) {
                if body.target.arch == Arch::X86_64 {
                    todo!("f128 sqrt lowering on x86-64");
                }
                // Through the toolchain `sqrtl` trampoline, like the
                // legacy backend's complex magnitudes.
                let quad = rt.quad.as_ref().expect("quad sqrt needs trampolines");
                let a_addr = f80_addr(body, builder, a);
                let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
                let out = builder.ins().stack_addr(types::I64, slot, 0);
                let callee = clif.declare_func_in_func(quad.sqrt, builder.func);
                builder.ins().call(callee, &[out, a_addr]);
                Some(f80_result(builder, out))
            } else {
                let x = operand(body, builder, a);
                Some(builder.ins().sqrt(x))
            }
        }
        mir::InstKind::Clz(a) => {
            let x = operand(body, builder, a);
            Some(builder.ins().clz(x))
        }
        mir::InstKind::Ctz(a) => {
            let x = operand(body, builder, a);
            Some(builder.ins().ctz(x))
        }
        mir::InstKind::Popcnt(a) => {
            let x = operand(body, builder, a);
            Some(builder.ins().popcnt(x))
        }
        mir::InstKind::Bswap(a) => {
            let x = operand(body, builder, a);
            Some(builder.ins().bswap(x))
        }
        mir::InstKind::Icmp { pred, lhs, rhs } => {
            let x = operand(body, builder, lhs);
            let y = operand(body, builder, rhs);
            let cc = match pred {
                mir::IcmpPred::Eq => IntCC::Equal,
                mir::IcmpPred::Ne => IntCC::NotEqual,
                mir::IcmpPred::Ult => IntCC::UnsignedLessThan,
                mir::IcmpPred::Ule => IntCC::UnsignedLessThanOrEqual,
                mir::IcmpPred::Ugt => IntCC::UnsignedGreaterThan,
                mir::IcmpPred::Uge => IntCC::UnsignedGreaterThanOrEqual,
                mir::IcmpPred::Slt => IntCC::SignedLessThan,
                mir::IcmpPred::Sle => IntCC::SignedLessThanOrEqual,
                mir::IcmpPred::Sgt => IntCC::SignedGreaterThan,
                mir::IcmpPred::Sge => IntCC::SignedGreaterThanOrEqual,
            };
            Some(builder.ins().icmp(cc, x, y))
        }
        mir::InstKind::Fcmp { pred, lhs, rhs } => {
            if is_f80(body, lhs) {
                // Ordered comparisons through the fucomip helpers: an
                // F128 fcmp would read quad fields, not x87 ones.
                // Unordered/ordered reuse the `ne` helper on each side
                // against itself (`ne` is true exactly for NaN, so no
                // new asm or x87 bit-decoding is needed).
                if matches!(pred, mir::FcmpPred::Uno | mir::FcmpPred::Ord) {
                    let f80 = rt.f80.as_ref().expect("f80 comparison needs x86-64");
                    let ne = f80.ne;
                    let a_nan = f80_cmp(body, builder, clif, ne, lhs, lhs);
                    let b_nan = f80_cmp(body, builder, clif, ne, rhs, rhs);
                    let unordered = builder.ins().bor(a_nan, b_nan);
                    if matches!(pred, mir::FcmpPred::Uno) {
                        Some(unordered)
                    } else {
                        let one = builder.ins().iconst(types::I8, 1);
                        Some(builder.ins().bxor(unordered, one))
                    }
                } else {
                    let f80 = rt.f80.as_ref().expect("f80 comparison needs x86-64");
                    let helper = match pred {
                        mir::FcmpPred::Eq => f80.eq,
                        mir::FcmpPred::Ne => f80.ne,
                        mir::FcmpPred::Lt => f80.lt,
                        mir::FcmpPred::Le => f80.le,
                        mir::FcmpPred::Gt => f80.gt,
                        mir::FcmpPred::Ge => f80.ge,
                        mir::FcmpPred::Uno | mir::FcmpPred::Ord => unreachable!(),
                    };
                    Some(f80_cmp(body, builder, clif, helper, lhs, rhs))
                }
            } else if is_f128(body, lhs) {
                // Ordered comparisons as integer sequences over the
                // halves (see `quad.rs`): no Clif fcmp on F128
                // anywhere, on any target.
                let qpred = match pred {
                    mir::FcmpPred::Eq => crate::quad::QuadCmp::Eq,
                    mir::FcmpPred::Ne => crate::quad::QuadCmp::Ne,
                    mir::FcmpPred::Lt => crate::quad::QuadCmp::Lt,
                    mir::FcmpPred::Le => crate::quad::QuadCmp::Le,
                    mir::FcmpPred::Gt => crate::quad::QuadCmp::Gt,
                    mir::FcmpPred::Ge => crate::quad::QuadCmp::Ge,
                    mir::FcmpPred::Uno => crate::quad::QuadCmp::Uno,
                    mir::FcmpPred::Ord => crate::quad::QuadCmp::Ord,
                };
                let x = operand(body, builder, lhs);
                let y = operand(body, builder, rhs);
                let xa = crate::quad::quad_spill(builder, x);
                let (a_lo, a_hi) = crate::quad::quad_load_halves(builder, xa);
                let xb = crate::quad::quad_spill(builder, y);
                let (b_lo, b_hi) = crate::quad::quad_load_halves(builder, xb);
                let r = crate::quad::quad_cmp(builder, qpred, a_lo, a_hi, b_lo, b_hi);
                Some(builder.ins().ireduce(types::I8, r))
            } else {
                let x = operand(body, builder, lhs);
                let y = operand(body, builder, rhs);
                let cc = match pred {
                    mir::FcmpPred::Eq => FloatCC::Equal,
                    mir::FcmpPred::Ne => FloatCC::NotEqual,
                    mir::FcmpPred::Lt => FloatCC::LessThan,
                    mir::FcmpPred::Le => FloatCC::LessThanOrEqual,
                    mir::FcmpPred::Gt => FloatCC::GreaterThan,
                    mir::FcmpPred::Ge => FloatCC::GreaterThanOrEqual,
                    mir::FcmpPred::Uno => FloatCC::Unordered,
                    mir::FcmpPred::Ord => FloatCC::Ordered,
                };
                Some(builder.ins().fcmp(cc, x, y))
            }
        }
        mir::InstKind::Trunc(a) => {
            let dst = result_clif(&result_ty);
            let lowered = operand(body, builder, a);
            Some(builder.ins().ireduce(dst, lowered))
        }
        mir::InstKind::Zext(a) => {
            let dst = result_clif(&result_ty);
            let lowered = operand(body, builder, a);
            Some(builder.ins().uextend(dst, lowered))
        }
        mir::InstKind::Sext(a) => {
            let dst = result_clif(&result_ty);
            let lowered = operand(body, builder, a);
            Some(builder.ins().sextend(dst, lowered))
        }
        mir::InstKind::Bitcast(a) => {
            let dst = result_clif(&result_ty);
            let lowered = operand(body, builder, a);
            let from = builder.func.dfg.value_type(lowered);
            if from.bytes() == dst.bytes() {
                Some(builder.ins().bitcast(dst, MemFlagsData::new(), lowered))
            } else if from.bytes() < dst.bytes() {
                // A narrow pointer widens back to the machine address
                // width, sign-extended like MSVC's default for `__ptr32`.
                Some(builder.ins().sextend(dst, lowered))
            } else {
                Some(builder.ins().ireduce(dst, lowered))
            }
        }
        mir::InstKind::SItoF(a) => {
            let dst_ty = result_ty.clone().expect("sitof needs a result type");
            if matches!(dst_ty, mir::Ty::Float(mir::FloatKind::F80)) {
                // Exact widening through fild: sub-word sources widen
                // value-preservingly to the helper width first.
                let f80 = rt.f80.as_ref().expect("f80 conversion needs x86-64");
                let src_ty = body.func.values[a].ty.clone();
                let bits = mir_int_bits(&src_ty);
                if bits > 64 {
                    // Through the vendored runtime (`__glcc_floattixf`),
                    // captured like a foreign return.
                    let x = operand(body, builder, a);
                    let (lo, hi) = builder.ins().isplit(x);
                    let callee = clif.declare_func_in_func(f80.from_i128, builder.func);
                    builder.ins().call(callee, &[lo, hi]);
                    Some(foreign_f80_result(builder, clif, rt))
                } else {
                    let x = operand(body, builder, a);
                    let src_clif = clif_ty(&src_ty).expect("int source has a Clif type");
                    let (helper, wide) = if bits <= 32 {
                        let w = if src_clif.bytes() < 4 { builder.ins().sextend(types::I32, x) } else { x };
                        (f80.from_i32, w)
                    } else {
                        (f80.from_i64, x)
                    };
                    let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
                    let out = builder.ins().stack_addr(types::I64, slot, 0);
                    let callee = clif.declare_func_in_func(helper, builder.func);
                    builder.ins().call(callee, &[out, wide]);
                    Some(f80_result(builder, out))
                }
            } else if matches!(dst_ty, mir::Ty::Float(mir::FloatKind::F128)) {
                // Integer ops (see `quad.rs`): sub-word sources widen
                // value-preservingly to 64 bits first, 128-bit sources
                // take the rounding path. Works on every target,
                // including x86-64 `__float128`.
                let src_ty = body.func.values[a].ty.clone();
                let bits = mir_int_bits(&src_ty);
                let x = operand(body, builder, a);
                if bits > 64 {
                    let (lo, hi) = crate::quad::quad_from_int128(builder, x, false);
                    Some(crate::quad::quad_store_carrier(builder, lo, hi))
                } else {
                    let src_clif = clif_ty(&src_ty).expect("int source has a Clif type");
                    let w32 = if src_clif.bytes() < 4 { builder.ins().sextend(types::I32, x) } else { x };
                    let wide = if bits <= 32 { builder.ins().sextend(types::I64, w32) } else { w32 };
                    let (lo, hi) = crate::quad::quad_from_int(builder, wide, false);
                    Some(crate::quad::quad_store_carrier(builder, lo, hi))
                }
            } else {
                let dst = result_clif(&result_ty);
                let lowered = operand(body, builder, a);
                Some(builder.ins().fcvt_from_sint(dst, lowered))
            }
        }
        mir::InstKind::UItoF(a) => {
            let dst_ty = result_ty.clone().expect("uitof needs a result type");
            if matches!(dst_ty, mir::Ty::Float(mir::FloatKind::F80)) {
                // Exact like above; unsigned sources zero-extend.
                let f80 = rt.f80.as_ref().expect("f80 conversion needs x86-64");
                let src_ty = body.func.values[a].ty.clone();
                let bits = mir_int_bits(&src_ty);
                if bits > 64 {
                    // Through the vendored runtime (`__glcc_floatuntixf`),
                    // captured like a foreign return.
                    let x = operand(body, builder, a);
                    let (lo, hi) = builder.ins().isplit(x);
                    let callee = clif.declare_func_in_func(f80.from_u128, builder.func);
                    builder.ins().call(callee, &[lo, hi]);
                    Some(foreign_f80_result(builder, clif, rt))
                } else {
                    let x = operand(body, builder, a);
                    let src_clif = clif_ty(&src_ty).expect("int source has a Clif type");
                    let (helper, wide) = if bits <= 32 {
                        let w = if src_clif.bytes() < 4 { builder.ins().uextend(types::I32, x) } else { x };
                        (f80.from_u32, w)
                    } else {
                        (f80.from_u64, x)
                    };
                    let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
                    let out = builder.ins().stack_addr(types::I64, slot, 0);
                    let callee = clif.declare_func_in_func(helper, builder.func);
                    builder.ins().call(callee, &[out, wide]);
                    Some(f80_result(builder, out))
                }
            } else if matches!(dst_ty, mir::Ty::Float(mir::FloatKind::F128)) {
                // Integer ops (see `quad.rs`): unsigned sources
                // zero-extend, 128-bit sources take the rounding path.
                let src_ty = body.func.values[a].ty.clone();
                let bits = mir_int_bits(&src_ty);
                let x = operand(body, builder, a);
                if bits > 64 {
                    let (lo, hi) = crate::quad::quad_from_int128(builder, x, true);
                    Some(crate::quad::quad_store_carrier(builder, lo, hi))
                } else {
                    let src_clif = clif_ty(&src_ty).expect("int source has a Clif type");
                    let w32 = if src_clif.bytes() < 4 { builder.ins().uextend(types::I32, x) } else { x };
                    let wide = if bits <= 32 { builder.ins().uextend(types::I64, w32) } else { w32 };
                    let (lo, hi) = crate::quad::quad_from_int(builder, wide, true);
                    Some(crate::quad::quad_store_carrier(builder, lo, hi))
                }
            } else {
                let dst = result_clif(&result_ty);
                let lowered = operand(body, builder, a);
                Some(builder.ins().fcvt_from_uint(dst, lowered))
            }
        }
        mir::InstKind::SFtoI(a) => {
            if is_f80(body, a) {
                // Truncation toward zero through fistp (which pins the
                // control word itself): sub-word destinations convert
                // at 32 bits, then narrow.
                let f80 = rt.f80.as_ref().expect("f80 conversion needs x86-64");
                let dst_ty = result_ty.clone().expect("sftoi needs a result type");
                let dst_bits = mir_int_bits(&dst_ty);
                if dst_bits > 64 {
                    // Through the vendored runtime (`__glcc_fixxfti`),
                    // matching clang exactly (including out-of-range).
                    let a_addr = f80_addr(body, builder, a);
                    let callee = clif.declare_func_in_func(f80.to_i128, builder.func);
                    let inst = builder.ins().call(callee, &[a_addr]);
                    let res = builder.inst_results(inst).to_vec();
                    Some(builder.ins().iconcat(res[0], res[1]))
                } else {
                    let a_addr = f80_addr(body, builder, a);
                    let (helper, from) = if dst_bits <= 32 { (f80.to_i32, types::I32) } else { (f80.to_i64, types::I64) };
                    let callee = clif.declare_func_in_func(helper, builder.func);
                    let inst = builder.ins().call(callee, &[a_addr]);
                    let r = builder.inst_results(inst)[0];
                    let dst = result_clif(&result_ty);
                    Some(resize_int(builder, r, from, dst))
                }
            } else if is_f128(body, a) {
                // Saturating truncation (see `quad.rs`): sub-word
                // destinations convert at 32 bits, then narrow;
                // 128-bit destinations convert wide directly.
                let dst_ty = result_ty.clone().expect("sftoi needs a result type");
                let dst_bits = mir_int_bits(&dst_ty);
                let x = operand(body, builder, a);
                let xa = crate::quad::quad_spill(builder, x);
                let (lo, hi) = crate::quad::quad_load_halves(builder, xa);
                if dst_bits > 64 {
                    Some(crate::quad::quad_to_s128(builder, lo, hi))
                } else {
                    let r = if dst_bits <= 32 {
                        crate::quad::quad_to_s32(builder, lo, hi)
                    } else {
                        crate::quad::quad_to_s64(builder, lo, hi)
                    };
                    let dst = result_clif(&result_ty);
                    let from = if dst_bits <= 32 { types::I32 } else { types::I64 };
                    Some(resize_int(builder, r, from, dst))
                }
            } else {
                let dst = result_clif(&result_ty);
                let lowered = operand(body, builder, a);
                // x64 converts at 32/64 bits only: sub-word
                // destinations go through I32 and narrow (the value is
                // in range or already saturated).
                let wide = if dst.bits() < 32 { types::I32 } else { dst };
                let converted = builder.ins().fcvt_to_sint(wide, lowered);
                Some(resize_int(builder, converted, wide, dst))
            }
        }
        mir::InstKind::UFtoI(a) => {
            if is_f80(body, a) {
                // Through the 64-bit unsigned conversion, then narrow
                // (gcc-observed: 32-bit results truncate the 64-bit
                // outcome).
                let f80 = rt.f80.as_ref().expect("f80 conversion needs x86-64");
                let dst_ty = result_ty.clone().expect("uftoi needs a result type");
                let dst_bits = mir_int_bits(&dst_ty);
                if dst_bits > 64 {
                    // Through the vendored runtime (`__glcc_fixunsxfti`),
                    // matching clang exactly (including out-of-range).
                    let a_addr = f80_addr(body, builder, a);
                    let callee = clif.declare_func_in_func(f80.to_u128, builder.func);
                    let inst = builder.ins().call(callee, &[a_addr]);
                    let res = builder.inst_results(inst).to_vec();
                    Some(builder.ins().iconcat(res[0], res[1]))
                } else {
                    let a_addr = f80_addr(body, builder, a);
                    let callee = clif.declare_func_in_func(f80.to_u64, builder.func);
                    let inst = builder.ins().call(callee, &[a_addr]);
                    let r = builder.inst_results(inst)[0];
                    let dst = result_clif(&result_ty);
                    Some(resize_int(builder, r, types::I64, dst))
                }
            } else if is_f128(body, a) {
                // Saturating truncation through 32/64/128 bits (see
                // `quad.rs`), then narrow like the platform.
                let dst_ty = result_ty.clone().expect("uftoi needs a result type");
                let dst_bits = mir_int_bits(&dst_ty);
                let x = operand(body, builder, a);
                let xa = crate::quad::quad_spill(builder, x);
                let (lo, hi) = crate::quad::quad_load_halves(builder, xa);
                if dst_bits > 64 {
                    Some(crate::quad::quad_to_u128(builder, lo, hi))
                } else {
                    let r = if dst_bits <= 32 {
                        crate::quad::quad_to_u32(builder, lo, hi)
                    } else {
                        crate::quad::quad_to_u64(builder, lo, hi)
                    };
                    let dst = result_clif(&result_ty);
                    let from = if dst_bits <= 32 { types::I32 } else { types::I64 };
                    Some(resize_int(builder, r, from, dst))
                }
            } else {
                let dst = result_clif(&result_ty);
                let lowered = operand(body, builder, a);
                let wide = if dst.bits() < 32 { types::I32 } else { dst };
                let converted = builder.ins().fcvt_to_uint(wide, lowered);
                Some(resize_int(builder, converted, wide, dst))
            }
        }
        mir::InstKind::Fptrunc(a) => {
            let src_ty = body.func.values[a].ty.clone();
            let dst_ty = result_ty.clone().expect("fptrunc needs a result type");
            if matches!(src_ty, mir::Ty::Float(mir::FloatKind::F80)) {
                // Single rounding through fstp under the live control.
                let f80 = rt.f80.as_ref().expect("f80 narrowing needs x86-64");
                if matches!(dst_ty, mir::Ty::Float(mir::FloatKind::F128)) {
                    // Exact widening to quad through the vendored runtime (x87 has
                    // no quad op): source by value, carrier back.
                    Some(f80_extend(body, builder, clif, f80.extend, a))
                } else {
                    let helper = match dst_ty {
                        mir::Ty::Float(mir::FloatKind::F32) => f80.to_f32,
                        mir::Ty::Float(mir::FloatKind::F64) => f80.to_f64,
                        _ => panic!("long double narrowing to non-float"),
                    };
                    let a_addr = f80_addr(body, builder, a);
                    let callee = clif.declare_func_in_func(helper, builder.func);
                    let inst = builder.ins().call(callee, &[a_addr]);
                    Some(builder.inst_results(inst)[0])
                }
            } else if matches!(src_ty, mir::Ty::Float(mir::FloatKind::F128)) {
                if matches!(dst_ty, mir::Ty::Float(mir::FloatKind::F80)) {
                    // Narrowing to x87 through the vendored runtime, captured like a
                    // foreign return (st(0) has no Clif name).
                    let f80 = rt.f80.as_ref().expect("quad narrowing needs x86-64");
                    Some(quad_trunc(body, builder, clif, rt, f80.trunc, a))
                } else {
                    // Single round-half-even through integer ops (see
                    // `quad.rs`); works on every target.
                    let x = operand(body, builder, a);
                    let xa = crate::quad::quad_spill(builder, x);
                    let (lo, hi) = crate::quad::quad_load_halves(builder, xa);
                    let bits = match dst_ty {
                        mir::Ty::Float(mir::FloatKind::F32) => crate::quad::quad_to_f32(builder, lo, hi),
                        mir::Ty::Float(mir::FloatKind::F64) => crate::quad::quad_to_f64(builder, lo, hi),
                        _ => panic!("quad narrowing to non-float"),
                    };
                    let dst = result_clif(&result_ty);
                    Some(builder.ins().bitcast(dst, MemFlagsData::new(), bits))
                }
            } else {
                let dst = result_clif(&result_ty);
                let lowered = operand(body, builder, a);
                Some(builder.ins().fdemote(dst, lowered))
            }
        }
        mir::InstKind::Fpext(a) => {
            let dst_ty = result_ty.clone().expect("fpext needs a result type");
            if matches!(dst_ty, mir::Ty::Float(mir::FloatKind::F80)) {
                if matches!(body.func.values[a].ty, mir::Ty::Float(mir::FloatKind::F128)) {
                    // Narrowing from quad through the vendored runtime (see above).
                    let f80 = rt.f80.as_ref().expect("quad widening needs x86-64");
                    Some(quad_trunc(body, builder, clif, rt, f80.trunc, a))
                } else {
                    // Exact widening through fld.
                    let f80 = rt.f80.as_ref().expect("f80 widening needs x86-64");
                    let src_ty = body.func.values[a].ty.clone();
                    let helper = match src_ty {
                        mir::Ty::Float(mir::FloatKind::F32) => f80.from_f32,
                        mir::Ty::Float(mir::FloatKind::F64) => f80.from_f64,
                        _ => panic!("widening to long double from non-float"),
                    };
                    let x = operand(body, builder, a);
                    let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
                    let out = builder.ins().stack_addr(types::I64, slot, 0);
                    let callee = clif.declare_func_in_func(helper, builder.func);
                    builder.ins().call(callee, &[out, x]);
                    Some(f80_result(builder, out))
                }
            } else if matches!(dst_ty, mir::Ty::Float(mir::FloatKind::F128)) {
                if matches!(body.func.values[a].ty, mir::Ty::Float(mir::FloatKind::F80)) {
                    // Exact widening from x87 through the vendored runtime.
                    let f80 = rt.f80.as_ref().expect("f80 widening needs x86-64");
                    Some(f80_extend(body, builder, clif, f80.extend, a))
                } else {
                    // Exact widening through integer ops (see `quad.rs`);
                    // works on every target, including x86-64 `__float128`.
                    let src_ty = body.func.values[a].ty.clone();
                    let x = operand(body, builder, a);
                    let (lo, hi) = match src_ty {
                        mir::Ty::Float(mir::FloatKind::F32) => {
                            let b = builder.ins().bitcast(types::I32, MemFlagsData::new(), x);
                            crate::quad::quad_from_f32(builder, b)
                        }
                        mir::Ty::Float(mir::FloatKind::F64) => {
                            let b = builder.ins().bitcast(types::I64, MemFlagsData::new(), x);
                            crate::quad::quad_from_f64(builder, b)
                        }
                        _ => panic!("widening to quad from non-float"),
                    };
                    Some(crate::quad::quad_store_carrier(builder, lo, hi))
                }
            } else {
                let dst = result_clif(&result_ty);
                let lowered = operand(body, builder, a);
                Some(builder.ins().fpromote(dst, lowered))
            }
        }
        mir::InstKind::PtrToInt(a) => {
            let dst = result_clif(&result_ty);
            let x = operand(body, builder, a);
            let from = builder.func.dfg.value_type(x);
            Some(if from == dst {
                x
            } else if from.bytes() == dst.bytes() {
                builder.ins().bitcast(dst, MemFlagsData::new(), x)
            } else if from.bytes() < dst.bytes() {
                // A narrow pointer sign-extends into an integer, its
                // default `__ptr32` semantics.
                builder.ins().sextend(dst, x)
            } else {
                builder.ins().ireduce(dst, x)
            })
        }
        mir::InstKind::IntToPtr(a) => {
            let dst = result_clif(&result_ty);
            let x = operand(body, builder, a);
            let from = builder.func.dfg.value_type(x);
            Some(if from == dst {
                x
            } else if from.bytes() == dst.bytes() {
                builder.ins().bitcast(dst, MemFlagsData::new(), x)
            } else if from.bytes() < dst.bytes() {
                builder.ins().sextend(dst, x)
            } else {
                builder.ins().ireduce(dst, x)
            })
        }
        mir::InstKind::Select { cond, then_val, else_val } => {
            let c = operand(body, builder, cond);
            let x = operand(body, builder, then_val);
            let y = operand(body, builder, else_val);
            Some(builder.ins().select(c, x, y))
        }
        mir::InstKind::Alloca { pointee, count } => {
            let size = pointee.size_bytes().checked_mul(count).expect("alloca size overflow");
            let slot = stack_slot(
                builder,
                &mir::Ty::Array {
                    elem: Box::new(mir::Ty::int(8)),
                    len: size,
                },
            );
            if let Some(result) = data.result {
                body.slots.insert(result, slot);
            }
            Some(builder.ins().stack_addr(types::I64, slot, 0))
        }
        mir::InstKind::Load { addr, flags } => {
            // This Cranelift has no volatile memory flag: a plain Clif
            // load is fair game for alias analysis (RLE, forwarding).
            // Volatile MIR accesses lower to atomic ops instead, which
            // the analysis never rewrites; see `volatile_load`.
            let x = addr_operand(body, builder, addr);
            let Some(mir_ty) = result_ty.clone() else {
                panic!("load needs a result type");
            };
            if mir_ty.is_aggregate() {
                if flags.volatile {
                    builder.ins().fence();
                }
                let slot = stack_slot(builder, &mir_ty);
                let dst = builder.ins().stack_addr(types::I64, slot, 0);
                if flags.volatile {
                    emit_memcpy(builder, dst, x, mir_ty.size_bytes());
                } else {
                    emit_copy(body, builder, dst, x, mir_ty.size_bytes());
                }
                if flags.volatile {
                    builder.ins().fence();
                }
                Some(dst)
            } else {
                let ty = clif_ty(&mir_ty).unwrap_or_else(|| panic!("load of non-scalar type {mir_ty:?}"));
                if flags.volatile {
                    Some(volatile_load(builder, ty, x))
                } else {
                    Some(builder.ins().load(ty, memflags::notrap(), x, Offset32::new(0)))
                }
            }
        }
        mir::InstKind::Store { addr, val, flags } => {
            let dst = addr_operand(body, builder, addr);
            let src = operand(body, builder, val);
            let mir_ty = body.func.values[val].ty.clone();
            if mir_ty.is_aggregate() {
                if flags.volatile {
                    builder.ins().fence();
                    emit_memcpy(builder, dst, src, mir_ty.size_bytes());
                    builder.ins().fence();
                } else {
                    emit_copy(body, builder, dst, src, mir_ty.size_bytes());
                }
            } else {
                let ty = clif_ty(&mir_ty).unwrap_or_else(|| panic!("store of non-scalar type {mir_ty:?}"));
                if flags.volatile {
                    volatile_store(builder, ty, src, dst);
                } else {
                    builder.ins().store(memflags::notrap(), src, dst, Offset32::new(0));
                }
            }
            None
        }
        mir::InstKind::GetElementPtr { base, indices } => {
            let base_addr = addr_operand(body, builder, base);
            let mir::Ty::Ptr { pointee, .. } = body.func.values[base].ty.clone() else {
                panic!("gep base is not a pointer");
            };
            let addr = gep_addr(body, builder, base_addr, &pointee, &indices);
            // A `__ptr32` result keeps its narrow width in the value.
            let want = result_clif(&result_ty);
            let got = builder.func.dfg.value_type(addr);
            Some(if got == want { addr } else { builder.ins().ireduce(want, addr) })
        }
        mir::InstKind::ExtractValue { agg, index } => {
            let addr = operand(body, builder, agg);
            let mir_ty = body.func.values[agg].ty.clone();
            let (off, elem) = field_offset(&mir_ty, index);
            match clif_ty(&elem) {
                Some(ty) => Some(builder.ins().load(ty, memflags::notrap(), addr, Offset32::new(off as i32))),
                // Nested aggregates address into the base slot.
                None => Some(if off == 0 { addr } else { builder.ins().iadd_imm_u(addr, off as i64) }),
            }
        }
        mir::InstKind::InsertValue { agg, elem, index } => {
            let base = operand(body, builder, agg);
            let val = operand(body, builder, elem);
            let mir_ty = body.func.values[agg].ty.clone();
            let (off, elem_ty) = field_offset(&mir_ty, index);
            let slot = stack_slot(builder, &mir_ty);
            let dst = builder.ins().stack_addr(types::I64, slot, 0);
            if !is_undef(body, agg) {
                emit_copy(body, builder, dst, base, mir_ty.size_bytes());
            }
            if elem_ty.is_aggregate() {
                emit_memcpy_offset(body, builder, dst, off, val, elem_ty.size_bytes());
            } else {
                builder.ins().store(memflags::notrap(), val, dst, Offset32::new(off as i32));
            }
            Some(dst)
        }
        mir::InstKind::Memcpy { dst, src, len } => {
            let d = addr_operand(body, builder, dst);
            let s = addr_operand(body, builder, src);
            emit_copy(body, builder, d, s, len);
            None
        }
        mir::InstKind::Memset { dst, byte, len } => {
            let d = addr_operand(body, builder, dst);
            // Splat the byte across 8-byte stores, then mop up the tail.
            let splat = (byte as u64).wrapping_mul(0x0101010101010101);
            let wide = builder.ins().iconst(types::I64, splat as i64);
            let words = len / 8;
            for i in 0..words {
                builder.ins().store(memflags::notrap(), wide, d, Offset32::new((i * 8) as i32));
            }
            let fill = builder.ins().iconst(types::I8, byte as i64);
            for i in (words * 8)..len {
                builder.ins().store(memflags::notrap(), fill, d, Offset32::new(i as i32));
            }
            None
        }
        mir::InstKind::GlobalAddr { global } => {
            let id = data_ids[&global];
            let gv = clif.declare_data_in_func(id, builder.func);
            // Thread-local storage addresses through the TLS
            // descriptor (`elf_gd` model, like the foreign-variadic
            // trampoline); plain globals use the absolute symbol
            // value.
            Some(if tls_data.contains(&global) {
                builder.ins().tls_value(types::I64, gv)
            } else {
                builder.ins().symbol_value(types::I64, gv)
            })
        }
        mir::InstKind::FuncAddr { func } => {
            let id = func_ids[&func];
            let funcref = clif.declare_func_in_func(id, builder.func);
            Some(builder.ins().func_addr(types::I64, funcref))
        }
        mir::InstKind::ExternAddr { target } => {
            let id = extern_ids[&target];
            let funcref = clif.declare_func_in_func(id, builder.func);
            Some(builder.ins().func_addr(types::I64, funcref))
        }
        mir::InstKind::Call { callee, args } => {
            let callee_layout = layouts[&callee].clone();
            emit_call(
                body,
                builder,
                clif,
                func_ids,
                extern_ids,
                data_ids,
                rt,
                layouts,
                extern_layouts,
                CallTarget::Internal(callee),
                &callee_layout,
                &args,
                &result_ty,
                data.result,
            )
        }
        mir::InstKind::CallExtern { target, args } => {
            let layout = extern_layouts[&target].clone();
            emit_call(
                body,
                builder,
                clif,
                func_ids,
                extern_ids,
                data_ids,
                rt,
                layouts,
                extern_layouts,
                CallTarget::External(target),
                &layout,
                &args,
                &result_ty,
                data.result,
            )
        }
        // The coroutine pass rewrites every `await` to a poll loop
        // before Clif lowering; one here means the pass was skipped.
        mir::InstKind::Await { .. } => panic!("`await` instruction survived coroutine lowering"),
        mir::InstKind::CallIndirect { callee, args, sig } => {
            let layout = abi::layout_sig(body.target, &sig.params, &sig.ret, sig.variadic);
            let addr = addr_operand(body, builder, callee);
            emit_call(
                body,
                builder,
                clif,
                func_ids,
                extern_ids,
                data_ids,
                rt,
                layouts,
                extern_layouts,
                CallTarget::Indirect { addr, variadic: sig.variadic },
                &layout,
                &args,
                &result_ty,
                data.result,
            )
        }
        mir::InstKind::VaStart { ap } => {
            let ap_addr = addr_operand(body, builder, ap);
            let va = body.va.expect("va_start outside a variadic function");
            abi::emit_va_start(body.target, builder, ap_addr, &va, &body.layout);
            None
        }
        mir::InstKind::VaArg { ap } => {
            let ap_addr = addr_operand(body, builder, ap);
            let Some(mir_ty) = result_ty.clone() else {
                panic!("va_arg needs a result type");
            };
            // A `va_list` parameter read in a non-variadic function has
            // no prelude of its own: x86-64 and AAPCS64 keep all cursor
            // state in the object itself, so those lower against a
            // dummy context (untouched on these targets). RISC-V folds
            // the caller's spill layout into the walk, which no object
            // carries.
            let (ctx, layout) = match body.va {
                Some(va) => (va, body.layout.clone()),
                None => {
                    if body.target.arch == Arch::Riscv64 {
                        panic!("va_arg outside a variadic function on RISC-V");
                    }
                    (
                        abi::VaCtx {
                            save_base: ap_addr,
                            overflow: ap_addr,
                        },
                        body.layout.clone(),
                    )
                }
            };
            match abi::emit_va_arg(body.target, builder, ap_addr, &ctx, &layout, &mir_ty) {
                abi::VaArgValue::Scalar(v) => Some(v),
                abi::VaArgValue::Addr(a) => Some(a),
            }
        }
        mir::InstKind::VaEnd { .. } => None,
        mir::InstKind::VaCopy { dst, src } => {
            let d = addr_operand(body, builder, dst);
            let s = addr_operand(body, builder, src);
            abi::emit_va_copy(body.target, builder, d, s);
            None
        }
        mir::InstKind::AtomicLoad { addr } => {
            let ty = result_clif(&result_ty);
            let a = addr_operand(body, builder, addr);
            Some(builder.ins().atomic_load(ty, MemFlagsData::new(), a))
        }
        mir::InstKind::AtomicStore { addr, val } => {
            let a = addr_operand(body, builder, addr);
            let v = operand(body, builder, val);
            builder.ins().atomic_store(MemFlagsData::new(), v, a);
            None
        }
        mir::InstKind::AtomicRmw { op, addr, val } => {
            let ty = result_clif(&result_ty);
            let a = addr_operand(body, builder, addr);
            let v = operand(body, builder, val);
            let rmw = match op {
                mir::AtomicRmwOp::Add => AtomicRmwOp::Add,
                mir::AtomicRmwOp::Sub => AtomicRmwOp::Sub,
                mir::AtomicRmwOp::And => AtomicRmwOp::And,
                mir::AtomicRmwOp::Nand => AtomicRmwOp::Nand,
                mir::AtomicRmwOp::Or => AtomicRmwOp::Or,
                mir::AtomicRmwOp::Xor => AtomicRmwOp::Xor,
                mir::AtomicRmwOp::Xchg => AtomicRmwOp::Xchg,
                mir::AtomicRmwOp::Umin => AtomicRmwOp::Umin,
                mir::AtomicRmwOp::Umax => AtomicRmwOp::Umax,
                mir::AtomicRmwOp::Smin => AtomicRmwOp::Smin,
                mir::AtomicRmwOp::Smax => AtomicRmwOp::Smax,
            };
            Some(builder.ins().atomic_rmw(ty, MemFlagsData::new(), rmw, a, v))
        }
        mir::InstKind::AtomicCas { addr, expected, replacement } => {
            let _ty = result_clif(&result_ty);
            let a = addr_operand(body, builder, addr);
            let e = operand(body, builder, expected);
            let r = operand(body, builder, replacement);
            Some(builder.ins().atomic_cas(MemFlagsData::new(), a, e, r))
        }
        mir::InstKind::AtomicFence => {
            builder.ins().fence();
            None
        }
        mir::InstKind::InlineAsm { block } => emit_inline_asm(body, builder, clif, &block, wrappers),
        mir::InstKind::GetFramePointer => Some(builder.ins().get_frame_pointer(result_clif(&result_ty))),
    };
    match (data.result, lowered) {
        (Some(mir_value), Some(clif_value)) => {
            body.values.insert(mir_value, clif_value);
        }
        (None, None) => {}
        (Some(_), None) => panic!("instruction dropped its result value"),
        (None, Some(_)) => panic!("resultless instruction produced a value"),
    }
}

/// Lower one inline-asm block: fill the operand area, call the generated
/// wrapper, and copy the register outputs back out. The wrapper text
/// goes into `wrappers` and is assembled into the TU's `.s`; see
/// [`crate::asm`]. An `asm goto` block's result is the label selector
/// (0 = fallthrough), which the MIR terminator dispatches on.
fn emit_inline_asm<M: Module>(body: &mut Body, builder: &mut FunctionBuilder, clif: &mut M, block: &mir::AsmBlock, wrappers: &mut Wrappers) -> Option<ClifValue> {
    let site = crate::asm::build(block, body.func, body.target);
    let is_goto = !block.labels.is_empty();
    let sig = crate::asm::wrapper_sig(body.target, is_goto);
    let wrapper = clif
        .declare_function(&site.name, Linkage::Import, &sig)
        .unwrap_or_else(|_| panic!("cannot declare {}", site.name));
    wrappers.add(&site.name, &site.text);
    let wrapper = clif.declare_func_in_func(wrapper, builder.func);
    let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, site.area_size as u32, 4));
    let area = builder.ins().stack_addr(types::I64, slot, 0);
    let area_flags = memflags::trusted();
    let user_flags = memflags::notrap();
    let outputs = block.outputs as usize;
    for (i, op) in block.operands.iter().enumerate() {
        if i < outputs {
            let Some(slot) = site.slots[i] else { continue };
            if slot.addr {
                // A memory output's address crosses in the slot.
                let addr = addr_operand(body, builder, op.value);
                builder.ins().store(area_flags, addr, area, Offset32::new(slot.offset as i32));
            } else if op.constraint.contains('+') {
                // `+` outputs start from the lvalue's current value.
                let ty = clif_ty(&op.ty).expect("asm register operand needs a scalar type");
                let addr = addr_operand(body, builder, op.value);
                let value = builder.ins().load(ty, user_flags, addr, Offset32::new(0));
                builder.ins().store(area_flags, value, area, Offset32::new(slot.offset as i32));
            }
            continue;
        }
        if site.slots[i].is_none() && op.tie.is_none() {
            continue;
        }
        match op.tie {
            Some(tie) => {
                let target = tie as usize;
                if site.slots[target].is_some_and(|slot| slot.addr) {
                    // Tied to a memory output: the input value must be in
                    // the output's location before the template runs.
                    // Aggregates are already copied there by MIR lowering;
                    // scalars store here.
                    if clif_ty(&op.ty).is_some() {
                        let value = operand(body, builder, op.value);
                        let addr = operand(body, builder, block.operands[target].value);
                        builder.ins().store(user_flags, value, addr, Offset32::new(0));
                    }
                } else {
                    let value = operand(body, builder, op.value);
                    let slot = site.slots[target].expect("tied output has a slot");
                    builder.ins().store(area_flags, value, area, Offset32::new(slot.offset as i32));
                }
            }
            None => {
                let value = operand(body, builder, op.value);
                let slot = site.slots[i].expect("asm input has a slot");
                builder.ins().store(area_flags, value, area, Offset32::new(slot.offset as i32));
            }
        }
    }
    let call = builder.ins().call(wrapper, &[area]);
    let selector = is_goto.then(|| builder.inst_results(call)[0]);
    for (i, op) in block.operands.iter().enumerate().take(outputs) {
        let Some(slot) = site.slots[i] else { continue };
        if slot.addr {
            continue;
        }
        let ty = clif_ty(&op.ty).expect("asm register operand needs a scalar type");
        let value = builder.ins().load(ty, area_flags, area, Offset32::new(slot.offset as i32));
        let addr = operand(body, builder, op.value);
        builder.ins().store(user_flags, value, addr, Offset32::new(0));
    }
    selector
}

/// Clif type of an instruction result, which must be scalar. Aggregates
/// never reach here: they resolve to addresses along dedicated paths.
fn result_clif(ty: &Option<mir::Ty>) -> Type {
    match ty {
        Some(mir_ty) => clif_ty(mir_ty).expect("result needs a scalar Clif type"),
        None => panic!("instruction needs a result type"),
    }
}

fn is_undef(body: &Body, mir_value: mir::Value) -> bool {
    matches!(&body.func.values[mir_value].def, mir::Def::Const(mir::Const::Undef))
}

/// Zero- or sign-agnostic width change between integer Clif types.
/// Addresses are plain bit patterns, so extension is always zero fill.
fn resize_int(builder: &mut FunctionBuilder, v: ClifValue, from: Type, to: Type) -> ClifValue {
    if from == to {
        return v;
    }
    if from.bytes() < to.bytes() {
        builder.ins().uextend(to, v)
    } else {
        builder.ins().ireduce(to, v)
    }
}

/// Narrow an ABI cover-width value back to its declared width, the
/// inverse of `Return`'s widening. Narrow scalars cross at register
/// width (`int_cover`: W-register covers on AAPCS64/RISC-V, exact
/// widths on x86-64), while bodies carry declared widths so casts and
/// spills lower directly. Non-integer types are passed through: their
/// covers already match.
fn narrow_to_declared(builder: &mut FunctionBuilder, v: ClifValue, mir_ty: &mir::Ty) -> ClifValue {
    if !matches!(mir_ty, mir::Ty::Bool | mir::Ty::Int { .. }) {
        return v;
    }
    let Some(decl) = clif_ty(mir_ty) else {
        return v;
    };
    let got = builder.func.dfg.value_type(v);
    if got == decl {
        return v;
    }
    resize_int(builder, v, got, decl)
}

fn emit_memcpy_offset(body: &Body, builder: &mut FunctionBuilder, dst_base: ClifValue, dst_off: u64, src: ClifValue, len: u64) {
    let dst = if dst_off == 0 { dst_base } else { builder.ins().iadd_imm_u(dst_base, dst_off as i64) };
    emit_copy(body, builder, dst, src, len);
}

/// Lower a GEP to address arithmetic: constant steps fold to one
/// displacement, dynamic array indices scale and accumulate.
fn gep_addr(body: &mut Body, builder: &mut FunctionBuilder, base: ClifValue, pointee: &mir::Ty, indices: &[mir::GepIndex]) -> ClifValue {
    let mut ty = pointee.clone();
    let mut fixed: i64 = 0;
    let mut dynamic: Vec<ClifValue> = Vec::new();
    for index in indices {
        match index {
            mir::GepIndex::Field(i) => {
                let mir::Ty::Struct { fields, .. } = &ty else {
                    panic!("field step into non-struct");
                };
                let offsets = mir::struct_field_offsets(&ty);
                let slot = *i as usize;
                if slot >= fields.len() {
                    panic!("field step out of range");
                }
                fixed += offsets[slot] as i64;
                ty = fields[slot].clone();
            }
            mir::GepIndex::ConstIndex(k) => match &ty {
                mir::Ty::Array { elem, .. } => {
                    fixed += (*k * elem.size_bytes()) as i64;
                    ty = elem.as_ref().clone();
                }
                // Element step over a pointed-to scalar: `p++` scales by
                // the pointee size, not by bytes.
                mir::Ty::Int { .. } | mir::Ty::Float(_) | mir::Ty::Bool | mir::Ty::Ptr { .. } => {
                    fixed += (*k * ty.size_bytes()) as i64;
                }
                _ => panic!("constant step into aggregate"),
            },
            mir::GepIndex::Index(v) => {
                let (scale, next) = match &ty {
                    mir::Ty::Array { elem, .. } => (elem.size_bytes(), elem.as_ref().clone()),
                    // Pointer arithmetic over a single element.
                    mir::Ty::Int { .. } | mir::Ty::Float(_) | mir::Ty::Bool | mir::Ty::Ptr { .. } => (ty.size_bytes(), ty.clone()),
                    _ => panic!("dynamic step into aggregate"),
                };
                let idx = operand(body, builder, *v);
                // C indices are signed (`p[-1]` walks backwards), so
                // narrow values sign-extend like GCC's `movslq`. (MIR
                // integers carry no signedness; an unsigned index above
                // 2^31 would mis-step, but such indices don't occur
                // while negative ones are common.)
                let wide = match clif_ty(&body.func.values[*v].ty) {
                    Some(t) if t.bytes() < 8 => builder.ins().sextend(types::I64, idx),
                    Some(t) if t.bytes() == 8 => idx,
                    _ => panic!("dynamic gep index must fit in 64 bits"),
                };
                let scaled = if scale == 1 { wide } else { builder.ins().imul_imm_u(wide, scale as i64) };
                dynamic.push(scaled);
                ty = next;
            }
        }
    }
    let mut addr = base;
    if fixed != 0 {
        addr = builder.ins().iadd_imm_u(addr, fixed);
    }
    for scaled in dynamic {
        addr = builder.ins().iadd(addr, scaled);
    }
    addr
}

/// Byte offset and element type of one aggregate position: struct field
/// or array element by index.
fn field_offset(ty: &mir::Ty, index: u32) -> (u64, mir::Ty) {
    match ty {
        mir::Ty::Struct { fields, .. } => {
            let offsets = mir::struct_field_offsets(ty);
            let slot = index as usize;
            if slot >= fields.len() {
                panic!("extract step out of range");
            }
            (offsets[slot], fields[slot].clone())
        }
        mir::Ty::Array { elem, .. } => (u64::from(index) * elem.size_bytes(), elem.as_ref().clone()),
        _ => panic!("extract step into scalar"),
    }
}

fn emit_term(body: &mut Body, builder: &mut FunctionBuilder, term: mir::Terminator) {
    match term {
        mir::Terminator::Br { dest, args } => {
            let target = body.blocks[&dest];
            let lowered = lower_branch_args(body, builder, dest, &args);
            builder.ins().jump(target, &lowered);
        }
        mir::Terminator::CondBr {
            cond,
            then_block,
            else_block,
            then_args,
            else_args,
        } => {
            let test = cond_value(body, builder, cond);
            let then_target = body.blocks[&then_block];
            let else_target = body.blocks[&else_block];
            let then_lowered = lower_branch_args(body, builder, then_block, &then_args);
            let else_lowered = lower_branch_args(body, builder, else_block, &else_args);
            builder.ins().brif(test, then_target, &then_lowered, else_target, &else_lowered);
        }
        mir::Terminator::Switch { scrutinee, cases, default } => {
            let scrut = operand(body, builder, scrutinee);
            let mir_ty = body.func.values[scrutinee].ty.clone();
            let width = match clif_ty(&mir_ty) {
                Some(t) if t.is_int() && t.bytes() <= 8 => t,
                _ => panic!("switch scrutinee must fit in 64 bits"),
            };
            let default_target = body.blocks[&default];
            if body.func.blocks[default].params.len() != 0 {
                panic!("switch target takes block params");
            }
            if cases.is_empty() {
                builder.ins().jump(default_target, &[]);
                return;
            }
            for (_, dest) in &cases {
                if body.func.blocks[*dest].params.len() != 0 {
                    panic!("switch target takes block params");
                }
            }
            // Non-negative cases map directly onto Cranelift's switch
            // helper, which uses jump tables for dense runs and a binary
            // search over sparse ones. Negative or high-bit cases keep
            // the compare chain below.
            let max_case = match width.bits() {
                64 => i64::MAX as i128,
                bits => (1i128 << (bits - 1)) - 1,
            };
            if cases.iter().all(|(value, _)| *value >= 0 && *value <= max_case) {
                let base = cases.iter().map(|(value, _)| *value).min().expect("non-empty cases");
                let mut switch = Switch::new();
                for (value, dest) in &cases {
                    switch.set_entry((*value - base) as u128, body.blocks[dest]);
                }
                let index = if base == 0 {
                    scrut
                } else {
                    let delta = builder.ins().iconst(width, base as i64);
                    builder.ins().isub(scrut, delta)
                };
                switch.emit(builder, index, default_target);
                // The helper creates and switches to its own dispatch
                // blocks, so the next source block must force a switch.
                body.current = None;
                return;
            }
            // If-else chain: one test block per case, falling through to
            // the next test and finally the default.
            let mut tests = Vec::new();
            for _ in 0..cases.len() {
                tests.push(builder.create_block());
            }
            builder.ins().jump(tests[0], &[]);
            switch_to(&mut *body, &mut *builder, tests[0]);
            for (i, (value, dest)) in cases.into_iter().enumerate() {
                switch_to(&mut *body, &mut *builder, tests[i]);
                let target = body.blocks[&dest];
                let probe = builder.ins().iconst(width, value as i64);
                let same = builder.ins().icmp(IntCC::Equal, scrut, probe);
                let otherwise = if i + 1 < tests.len() { tests[i + 1] } else { default_target };
                builder.ins().brif(same, target, &[], otherwise, &[]);
            }
        }
        mir::Terminator::Return { val } => {
            // Clone the piece list first: loading the value below
            // reborrows the body. Multi-piece returns (aggregates and
            // 128-bit scalar pairs) need piece handling either way.
            let ret_pieces = match &body.layout.ret {
                abi::RetLoc::Regs(pieces) if val.is_some() && (body.func.sig.ret.is_aggregate() || pieces.len() > 1) => Some(pieces.clone()),
                _ => None,
            };
            match &body.layout.ret {
                abi::RetLoc::Sret => {
                    let slot = body.sret.expect("aggregate return needs an out-pointer");
                    let addr = operand(body, builder, val.expect("aggregate return needs a value"));
                    emit_copy(body, builder, slot, addr, body.func.sig.ret.size_bytes());
                    builder.ins().return_(&[]);
                }
                abi::RetLoc::Regs(_) if ret_pieces.is_some() => {
                    let pieces = ret_pieces.unwrap();
                    // Multi-piece scalars split one value; aggregates
                    // load each piece from the slot.
                    if pieces.len() > 1 && !body.func.sig.ret.is_aggregate() {
                        let v = operand(body, builder, val.unwrap());
                        let from = builder.func.dfg.value_type(v);
                        let outs = split_value(builder, v, from, pieces.len());
                        builder.ins().return_(&outs);
                    } else {
                        let addr = operand(body, builder, val.unwrap());
                        let mut outs = Vec::with_capacity(pieces.len());
                        for (off, ty) in pieces {
                            outs.push(builder.ins().load(ty, memflags::notrap(), addr, Offset32::new(off as i32)));
                        }
                        builder.ins().return_(&outs);
                    }
                }
                _ => {
                    // A zero-sized aggregate return crosses as nothing;
                    // its value is an address that is simply dropped.
                    if matches!(body.layout.ret, abi::RetLoc::Void) {
                        builder.ins().return_(&[]);
                    } else if let Some(v) = val {
                        let x = operand(body, builder, v);
                        // Non-x86 sub-int returns use W-register
                        // slots: widen the produced value to the slot
                        // width.
                        let from = builder.func.dfg.value_type(x);
                        let x = match &body.layout.ret {
                            abi::RetLoc::Regs(pieces) if pieces.len() == 1 && !from.is_float() && !pieces[0].1.is_float() => resize_int(builder, x, from, pieces[0].1),
                            _ => x,
                        };
                        builder.ins().return_(&[x]);
                    } else {
                        builder.ins().return_(&[]);
                    }
                }
            }
        }
        mir::Terminator::Unreachable => {
            builder.ins().trap(TrapCode::user(1).expect("nonzero trap code"));
        }
    }
}

fn lower_branch_args(body: &mut Body, builder: &mut FunctionBuilder, dest: mir::Block, args: &[mir::Value]) -> Vec<BlockArg> {
    let params = &body.func.blocks[dest].params;
    if params.len() != args.len() {
        panic!("branch arity mismatch");
    }
    args.iter().map(|a| BlockArg::Value(operand(body, builder, *a))).collect()
}

/// Any integer condition as an I8 test: nonzero is true.
fn cond_value(body: &mut Body, builder: &mut FunctionBuilder, cond: mir::Value) -> ClifValue {
    let x = operand(body, builder, cond);
    match &body.func.values[cond].ty {
        mir::Ty::Bool | mir::Ty::Int { bits: 1 } => x,
        mir::Ty::Int { .. } => builder.ins().icmp_imm_u(IntCC::NotEqual, x, 0),
        _ => panic!("branch condition must be an integer"),
    }
}

// These tests execute generated code in-process, so they run with the
// host's own ABI (`Target::native()`): SystemV on Unix, Win64 on
// Windows.
#[cfg(test)]
mod tests {
    use cranelift_jit::{JITBuilder, JITModule};

    use glcc_mir as mir;

    use super::lower_module;

    fn param(mir_func: &mut mir::Function, block: mir::Block, ty: mir::Ty, index: u32) -> mir::Value {
        let v = mir_func.values.push(mir::ValueData {
            ty,
            def: mir::Def::Param(block, index),
        });
        mir_func.blocks[block].params.push(v);
        v
    }

    fn iconst(mir_func: &mut mir::Function, bits: u32, value: i128) -> mir::Value {
        mir_func.push_const(mir::Const::Int { bits, value })
    }

    fn emit(mir_func: &mut mir::Function, block: mir::Block, kind: mir::InstKind, ty: mir::Ty) -> mir::Value {
        let result = mir_func.values.push(mir::ValueData {
            ty,
            def: mir::Def::Inst(mir::Inst::from_usize(mir_func.insts.len())),
        });
        let inst = mir_func.insts.push(mir::InstData {
            block,
            kind,
            result: Some(result),
            loc: None,
        });
        mir_func.values[result].def = mir::Def::Inst(inst);
        mir_func.blocks[block].insts.push(inst);
        result
    }

    fn emit_unit(mir_func: &mut mir::Function, block: mir::Block, kind: mir::InstKind) {
        let inst = mir_func.insts.push(mir::InstData {
            block,
            kind,
            result: None,
            loc: None,
        });
        mir_func.blocks[block].insts.push(inst);
    }

    fn ret(mir_func: &mut mir::Function, block: mir::Block, val: Option<mir::Value>) {
        mir_func.blocks[block].term = Some(mir::Terminator::Return { val });
    }

    fn jit_module(mir_module: &mir::Module) -> (JITModule, std::collections::HashMap<mir::Func, cranelift_module::FuncId>) {
        jit_module_with(mir_module, &[])
    }

    /// [`jit_module`] plus extra runtime symbols (soft-float trampolines,
    /// getret shims) resolved to their addresses before lowering.
    fn jit_module_with(mir_module: &mir::Module, symbols: &[(&'static str, *const u8)]) -> (JITModule, std::collections::HashMap<mir::Func, cranelift_module::FuncId>) {
        let mut builder = JITBuilder::new(cranelift_module::default_libcall_names()).expect("jit builder");
        for (name, addr) in symbols {
            builder.symbol(*name, *addr);
        }
        let mut jit = JITModule::new(builder);
        let ids = lower_module(mir_module, &mut jit, crate::abi::Target::native());
        jit.finalize_definitions().expect("jit finalize");
        (jit, ids)
    }

    fn func_addr(jit: &JITModule, ids: &std::collections::HashMap<mir::Func, cranelift_module::FuncId>, index: usize) -> *const u8 {
        let addr = jit.get_finalized_function(ids[&mir::Func::from_usize(index)]);
        assert!(!addr.is_null(), "missing jit function {index}");
        addr
    }

    fn run_i32(jit: &JITModule, ids: &std::collections::HashMap<mir::Func, cranelift_module::FuncId>, index: usize, arg: i32) -> i32 {
        unsafe {
            let func: extern "C" fn(i32) -> i32 = std::mem::transmute(func_addr(jit, ids, index));
            func(arg)
        }
    }

    fn run_i32_0(jit: &JITModule, ids: &std::collections::HashMap<mir::Func, cranelift_module::FuncId>, index: usize) -> i32 {
        unsafe {
            let func: extern "C" fn() -> i32 = std::mem::transmute(func_addr(jit, ids, index));
            func()
        }
    }

    fn run_i32_2(jit: &JITModule, ids: &std::collections::HashMap<mir::Func, cranelift_module::FuncId>, index: usize, a: i32, b: i32) -> i32 {
        unsafe {
            let func: extern "C" fn(i32, i32) -> i32 = std::mem::transmute(func_addr(jit, ids, index));
            func(a, b)
        }
    }

    fn run_f64(jit: &JITModule, ids: &std::collections::HashMap<mir::Func, cranelift_module::FuncId>, index: usize, arg: f64) -> f64 {
        unsafe {
            let func: extern "C" fn(f64) -> f64 = std::mem::transmute(func_addr(jit, ids, index));
            func(arg)
        }
    }

    fn run_u32(jit: &JITModule, ids: &std::collections::HashMap<mir::Func, cranelift_module::FuncId>, index: usize, arg: u32) -> u32 {
        unsafe {
            let func: extern "C" fn(u32) -> u32 = std::mem::transmute(func_addr(jit, ids, index));
            func(arg)
        }
    }

    fn i32_sig(params: usize) -> mir::Signature {
        mir::Signature::new(vec![mir::Ty::int(32); params], mir::Ty::int(32))
    }

    #[test]
    fn scalar_arithmetic_executes() {
        // f(x) = x * 2 + 1.
        let mut module = mir::Module::new();
        let (mut f, entry) = mir::Function::new("f", i32_sig(1));
        let x = param(&mut f, entry, mir::Ty::int(32), 0);
        let two = iconst(&mut f, 32, 2);
        let one = iconst(&mut f, 32, 1);
        let doubled = emit(&mut f, entry, mir::InstKind::Mul(x, two), mir::Ty::int(32));
        let result = emit(&mut f, entry, mir::InstKind::Add(doubled, one), mir::Ty::int(32));
        ret(&mut f, entry, Some(result));
        module.push_func(f);
        let (jit, ids) = jit_module(&module);
        assert_eq!(run_i32(&jit, &ids, 0, 20), 41);
    }

    #[test]
    fn sqrt_executes() {
        // sqrt(x) on f32/f64 lowers to the Clif sqrt instruction.
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![mir::Ty::Float(mir::FloatKind::F64)], mir::Ty::Float(mir::FloatKind::F64));
        let (mut f, entry) = mir::Function::new("dsqrt", sig);
        let x = param(&mut f, entry, mir::Ty::Float(mir::FloatKind::F64), 0);
        let r = emit(&mut f, entry, mir::InstKind::Sqrt(x), mir::Ty::Float(mir::FloatKind::F64));
        ret(&mut f, entry, Some(r));
        module.push_func(f);
        let (jit, ids) = jit_module(&module);
        assert_eq!(run_f64(&jit, &ids, 0, 25.0), 5.0);
        assert_eq!(run_f64(&jit, &ids, 0, 2.0), 2.0f64.sqrt());
    }

    #[test]
    fn icmp_and_select_execute() {
        // max(a, b) via signed compare and select.
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![mir::Ty::int(32), mir::Ty::int(32)], mir::Ty::int(32));
        let (mut f, entry) = mir::Function::new("max", sig);
        let a = param(&mut f, entry, mir::Ty::int(32), 0);
        let b = param(&mut f, entry, mir::Ty::int(32), 1);
        let gt = emit(
            &mut f,
            entry,
            mir::InstKind::Icmp {
                pred: mir::IcmpPred::Sgt,
                lhs: a,
                rhs: b,
            },
            mir::Ty::Bool,
        );
        let result = emit(
            &mut f,
            entry,
            mir::InstKind::Select {
                cond: gt,
                then_val: a,
                else_val: b,
            },
            mir::Ty::int(32),
        );
        ret(&mut f, entry, Some(result));
        module.push_func(f);
        let (jit, ids) = jit_module(&module);
        assert_eq!(run_i32_2(&jit, &ids, 0, 3, 7), 7);
        assert_eq!(run_i32_2(&jit, &ids, 0, 9, 2), 9);
    }

    #[test]
    fn struct_gep_store_load_executes() {
        // Point { x, y } in a slot; return x + y.
        let mut module = mir::Module::new();
        let (mut f, entry) = mir::Function::new("point", mir::Signature::new(vec![], mir::Ty::int(32)));
        let point_ty = mir::Ty::Struct {
            fields: vec![mir::Ty::int(32), mir::Ty::int(32)],
            offsets: vec![0, 4],
            size: 8,
        };
        let slot = emit(
            &mut f,
            entry,
            mir::InstKind::Alloca {
                pointee: point_ty.clone(),
                count: 1,
            },
            mir::Ty::ptr(point_ty.clone()),
        );
        let six = iconst(&mut f, 32, 6);
        let seven = iconst(&mut f, 32, 7);
        for (field, val) in [(0u32, six), (1u32, seven)] {
            let addr = emit(
                &mut f,
                entry,
                mir::InstKind::GetElementPtr {
                    base: slot,
                    indices: [mir::GepIndex::Field(field)].into_iter().collect(),
                },
                mir::Ty::ptr(mir::Ty::int(32)),
            );
            emit_unit(
                &mut f,
                entry,
                mir::InstKind::Store {
                    addr,
                    val,
                    flags: mir::MemFlags::plain(),
                },
            );
        }
        let mut loaded = Vec::new();
        for field in [0u32, 1u32] {
            let addr = emit(
                &mut f,
                entry,
                mir::InstKind::GetElementPtr {
                    base: slot,
                    indices: [mir::GepIndex::Field(field)].into_iter().collect(),
                },
                mir::Ty::ptr(mir::Ty::int(32)),
            );
            loaded.push(emit(
                &mut f,
                entry,
                mir::InstKind::Load {
                    addr,
                    flags: mir::MemFlags::plain(),
                },
                mir::Ty::int(32),
            ));
        }
        let sum = emit(&mut f, entry, mir::InstKind::Add(loaded[0], loaded[1]), mir::Ty::int(32));
        ret(&mut f, entry, Some(sum));
        module.push_func(f);
        let (jit, ids) = jit_module(&module);
        assert_eq!(run_i32_0(&jit, &ids, 0), 13);
    }

    #[test]
    fn call_between_functions_executes() {
        let mut module = mir::Module::new();
        let (mut double, dentry) = mir::Function::new("double", i32_sig(1));
        let x = param(&mut double, dentry, mir::Ty::int(32), 0);
        let two = iconst(&mut double, 32, 2);
        let result = emit(&mut double, dentry, mir::InstKind::Mul(x, two), mir::Ty::int(32));
        ret(&mut double, dentry, Some(result));
        let callee = module.push_func(double);

        let (mut caller, centry) = mir::Function::new("caller", i32_sig(1));
        let arg = param(&mut caller, centry, mir::Ty::int(32), 0);
        let call_ty = mir::Ty::int(32);
        let call_result = mir::Function::new("tmp", i32_sig(0));
        let _ = call_result;
        let call_val = emit(
            &mut caller,
            centry,
            mir::InstKind::Call {
                callee,
                args: [arg].into_iter().collect(),
            },
            call_ty,
        );
        ret(&mut caller, centry, Some(call_val));
        module.push_func(caller);
        let (jit, ids) = jit_module(&module);
        assert_eq!(run_i32(&jit, &ids, 1, 21), 42);
    }

    #[test]
    fn global_load_executes() {
        let mut module = mir::Module::new();
        let g = module.push_global("g", mir::Ty::int(32), Some(mir::Const::Int { bits: 32, value: 5 }));
        let (mut f, entry) = mir::Function::new("f", mir::Signature::new(vec![], mir::Ty::int(32)));
        let addr = emit(&mut f, entry, mir::InstKind::GlobalAddr { global: g }, mir::Ty::ptr(mir::Ty::int(32)));
        let loaded = emit(
            &mut f,
            entry,
            mir::InstKind::Load {
                addr,
                flags: mir::MemFlags::plain(),
            },
            mir::Ty::int(32),
        );
        let one = iconst(&mut f, 32, 1);
        let result = emit(&mut f, entry, mir::InstKind::Add(loaded, one), mir::Ty::int(32));
        ret(&mut f, entry, Some(result));
        module.push_func(f);
        let (jit, ids) = jit_module(&module);
        assert_eq!(run_i32_0(&jit, &ids, 0), 6);
    }

    #[test]
    fn switch_executes() {
        // f(x): 0 -> 10, 1 -> 20, otherwise 30.
        let mut module = mir::Module::new();
        let (mut f, entry) = mir::Function::new("f", i32_sig(1));
        let x = param(&mut f, entry, mir::Ty::int(32), 0);
        let b0 = f.push_block(vec![]);
        let b1 = f.push_block(vec![]);
        let bdef = f.push_block(vec![]);
        f.blocks[entry].term = Some(mir::Terminator::Switch {
            scrutinee: x,
            cases: vec![(0, b0), (1, b1)],
            default: bdef,
        });
        for (block, val) in [(b0, 10), (b1, 20), (bdef, 30)] {
            let c = iconst(&mut f, 32, val);
            f.blocks[block].term = Some(mir::Terminator::Return { val: Some(c) });
        }
        module.push_func(f);
        let (jit, ids) = jit_module(&module);
        assert_eq!(run_i32(&jit, &ids, 0, 0), 10);
        assert_eq!(run_i32(&jit, &ids, 0, 1), 20);
        assert_eq!(run_i32(&jit, &ids, 0, 99), 30);
    }

    #[test]
    fn large_copies_call_libc_memcpy() {
        use cranelift_codegen::ir::Opcode;

        fn opcodes(len: u64) -> Vec<Opcode> {
            let mut module = mir::Module::new();
            let sig = mir::Signature::new(vec![mir::Ty::ptr(mir::Ty::Void), mir::Ty::ptr(mir::Ty::Void)], mir::Ty::Void);
            let (mut f, entry) = mir::Function::new("copy", sig);
            let dst = param(&mut f, entry, mir::Ty::ptr(mir::Ty::Void), 0);
            let src = param(&mut f, entry, mir::Ty::ptr(mir::Ty::Void), 1);
            emit_unit(&mut f, entry, mir::InstKind::Memcpy { dst, src, len });
            ret(&mut f, entry, None);
            module.push_func(f);

            let mut sink = RecordingModule::new();
            lower_module(&module, &mut sink, crate::abi::Target::native());
            let func = &sink.funcs[0];
            func.layout
                .blocks()
                .flat_map(|block| func.layout.block_insts(block))
                .map(|inst| func.dfg.insts[inst].opcode())
                .collect()
        }

        let large = opcodes(1024);
        assert!(large.contains(&Opcode::Call), "{large:?}");
        assert!(!large.contains(&Opcode::Load), "{large:?}");

        let small = opcodes(64);
        assert!(!small.contains(&Opcode::Call), "{small:?}");
        assert!(small.contains(&Opcode::Load), "{small:?}");
    }

    #[test]
    fn volatile_aggregate_copies_stay_inline() {
        use cranelift_codegen::ir::Opcode;

        let mut module = mir::Module::new();
        let ty = mir::Ty::Array {
            elem: Box::new(mir::Ty::int(8)),
            len: 1024,
        };
        let sig = mir::Signature::new(vec![mir::Ty::ptr(mir::Ty::Void), mir::Ty::ptr(mir::Ty::Void)], mir::Ty::Void);
        let (mut f, entry) = mir::Function::new("copy", sig);
        let dst = param(&mut f, entry, mir::Ty::ptr(mir::Ty::Void), 0);
        let src = param(&mut f, entry, mir::Ty::ptr(mir::Ty::Void), 1);
        let value = emit(
            &mut f,
            entry,
            mir::InstKind::Load {
                addr: src,
                flags: mir::MemFlags::vol(),
            },
            ty,
        );
        emit_unit(
            &mut f,
            entry,
            mir::InstKind::Store {
                addr: dst,
                val: value,
                flags: mir::MemFlags::vol(),
            },
        );
        ret(&mut f, entry, None);
        module.push_func(f);

        let mut sink = RecordingModule::new();
        lower_module(&module, &mut sink, crate::abi::Target::native());
        let func = &sink.funcs[0];
        let opcodes: Vec<_> = func
            .layout
            .blocks()
            .flat_map(|block| func.layout.block_insts(block))
            .map(|inst| func.dfg.insts[inst].opcode())
            .collect();
        assert!(!opcodes.contains(&Opcode::Call), "{opcodes:?}");
        assert!(opcodes.contains(&Opcode::Load), "{opcodes:?}");
    }

    #[test]
    fn non_variadic_extern_calls_are_direct() {
        use cranelift_codegen::ir::Opcode;

        let mut module = mir::Module::new();
        let ext = module.push_extern("ext", mir::Signature::new(vec![mir::Ty::int(32)], mir::Ty::int(32)));
        let (mut f, entry) = mir::Function::new("f", i32_sig(1));
        let arg = param(&mut f, entry, mir::Ty::int(32), 0);
        let value = emit(
            &mut f,
            entry,
            mir::InstKind::CallExtern {
                target: ext,
                args: [arg].into_iter().collect(),
            },
            mir::Ty::int(32),
        );
        ret(&mut f, entry, Some(value));
        module.push_func(f);

        let mut sink = RecordingModule::new();
        lower_module(&module, &mut sink, crate::abi::Target::native());
        let func = &sink.funcs[0];
        let opcodes: Vec<_> = func
            .layout
            .blocks()
            .flat_map(|block| func.layout.block_insts(block))
            .map(|inst| func.dfg.insts[inst].opcode())
            .collect();
        assert!(opcodes.contains(&Opcode::Call), "{opcodes:?}");
        assert!(!opcodes.contains(&Opcode::FuncAddr), "{opcodes:?}");
    }

    #[test]
    fn dense_switches_lower_to_a_jump_table() {
        use cranelift_codegen::ir::Opcode;

        let mut module = mir::Module::new();
        let (mut f, entry) = mir::Function::new("f", i32_sig(1));
        let x = param(&mut f, entry, mir::Ty::int(32), 0);
        let mut cases = Vec::new();
        let default = f.push_block(vec![]);
        for value in 0..4 {
            let block = f.push_block(vec![]);
            cases.push((value, block));
            let c = iconst(&mut f, 32, 10 + value as i128);
            f.blocks[block].term = Some(mir::Terminator::Return { val: Some(c) });
        }
        f.blocks[entry].term = Some(mir::Terminator::Switch { scrutinee: x, cases, default });
        let fallback = iconst(&mut f, 32, 0);
        f.blocks[default].term = Some(mir::Terminator::Return { val: Some(fallback) });
        module.push_func(f);

        let mut sink = RecordingModule::new();
        lower_module(&module, &mut sink, crate::abi::Target::native());
        let func = &sink.funcs[0];
        let opcodes: Vec<_> = func
            .layout
            .blocks()
            .flat_map(|block| func.layout.block_insts(block))
            .map(|inst| func.dfg.insts[inst].opcode())
            .collect();
        assert!(opcodes.contains(&Opcode::BrTable), "{opcodes:?}");
    }

    #[test]
    fn condbr_loop_executes() {
        // Sum 1..=n with an explicit loop over block params.
        let mut module = mir::Module::new();
        let (mut f, entry) = mir::Function::new("f", i32_sig(1));
        let n = param(&mut f, entry, mir::Ty::int(32), 0);
        let header = f.push_block(vec![mir::Ty::int(32), mir::Ty::int(32)]);
        let body = f.push_block(vec![]);
        let exit = f.push_block(vec![mir::Ty::int(32)]);
        let zero = iconst(&mut f, 32, 0);
        let one_entry = iconst(&mut f, 32, 1);
        f.blocks[entry].term = Some(mir::Terminator::Br {
            dest: header,
            args: [one_entry, zero].into_iter().collect(),
        });
        let i = f.blocks[header].params[0];
        let acc = f.blocks[header].params[1];
        let done = emit(
            &mut f,
            header,
            mir::InstKind::Icmp {
                pred: mir::IcmpPred::Sgt,
                lhs: i,
                rhs: n,
            },
            mir::Ty::Bool,
        );
        f.blocks[header].term = Some(mir::Terminator::CondBr {
            cond: done,
            then_block: exit,
            else_block: body,
            then_args: [acc].into_iter().collect(),
            else_args: Default::default(),
        });
        let one = iconst(&mut f, 32, 1);
        let acc_next = emit(&mut f, body, mir::InstKind::Add(acc, i), mir::Ty::int(32));
        let i_next = emit(&mut f, body, mir::InstKind::Add(i, one), mir::Ty::int(32));
        f.blocks[body].term = Some(mir::Terminator::Br {
            dest: header,
            args: [i_next, acc_next].into_iter().collect(),
        });
        f.blocks[exit].term = Some(mir::Terminator::Return {
            val: Some(f.blocks[exit].params[0]),
        });
        module.push_func(f);
        let (jit, ids) = jit_module(&module);
        assert_eq!(run_i32(&jit, &ids, 0, 10), 55);
    }

    #[test]
    fn shared_constants_materialize_per_block() {
        // One constant used in sibling branches: a single definition
        // in either branch would not dominate the other's use, so
        // every block rebuilds its own copy.
        let mut module = mir::Module::new();
        let (mut f, entry) = mir::Function::new("f", i32_sig(1));
        let p = param(&mut f, entry, mir::Ty::int(32), 0);
        let then_block = f.push_block(vec![]);
        let else_block = f.push_block(vec![]);
        let merge = f.push_block(vec![mir::Ty::int(32)]);
        let zero = iconst(&mut f, 32, 0);
        let nonzero = emit(
            &mut f,
            entry,
            mir::InstKind::Icmp {
                pred: mir::IcmpPred::Ne,
                lhs: p,
                rhs: zero,
            },
            mir::Ty::Bool,
        );
        f.blocks[entry].term = Some(mir::Terminator::CondBr {
            cond: nonzero,
            then_block,
            else_block,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        let seven = iconst(&mut f, 32, 7);
        f.blocks[then_block].term = Some(mir::Terminator::Br {
            dest: merge,
            args: [seven].into_iter().collect(),
        });
        f.blocks[else_block].term = Some(mir::Terminator::Br {
            dest: merge,
            args: [seven].into_iter().collect(),
        });
        f.blocks[merge].term = Some(mir::Terminator::Return {
            val: Some(f.blocks[merge].params[0]),
        });
        module.push_func(f);
        let (jit, ids) = jit_module(&module);
        assert_eq!(run_i32(&jit, &ids, 0, 1), 7);
        assert_eq!(run_i32(&jit, &ids, 0, 0), 7);
    }

    #[test]
    fn definitions_may_live_in_later_indexed_dominators() {
        // `late` loads a value `mid` defines, and `mid` has the higher
        // arena index: filling in arena order would use the value
        // before its definition.
        let mut module = mir::Module::new();
        let (mut f, entry) = mir::Function::new("f", i32_sig(0));
        let late = f.push_block(vec![]);
        let mid = f.push_block(vec![]);
        f.blocks[entry].term = Some(mir::Terminator::Br {
            dest: mid,
            args: Default::default(),
        });
        let two = iconst(&mut f, 32, 2);
        let forty = iconst(&mut f, 32, 40);
        let value = emit(&mut f, mid, mir::InstKind::Add(two, forty), mir::Ty::int(32));
        f.blocks[mid].term = Some(mir::Terminator::Br {
            dest: late,
            args: Default::default(),
        });
        f.blocks[late].term = Some(mir::Terminator::Return { val: Some(value) });
        module.push_func(f);
        let (jit, ids) = jit_module(&module);
        assert_eq!(run_i32_0(&jit, &ids, 0), 42);
    }

    #[test]
    fn undef_lowers_to_zero_for_floats_and_wide_ints() {
        // Undef has no bits, and `iconst` covers neither f64 nor i128,
        // so both need their own zero forms.
        let mut module = mir::Module::new();
        let (mut f, entry) = mir::Function::new("returns_undef", mir::Signature::new(vec![], mir::Ty::Float(mir::FloatKind::F64)));
        let undef = f.values.push(mir::ValueData {
            ty: mir::Ty::Float(mir::FloatKind::F64),
            def: mir::Def::Const(mir::Const::Undef),
        });
        ret(&mut f, entry, Some(undef));
        module.push_func(f);

        let (mut g, entry) = mir::Function::new("wide_undef", i32_sig(0));
        let wide_undef = g.values.push(mir::ValueData {
            ty: mir::Ty::int(128),
            def: mir::Def::Const(mir::Const::Undef),
        });
        let one = iconst(&mut g, 128, 1);
        let sum = emit(&mut g, entry, mir::InstKind::Add(wide_undef, one), mir::Ty::int(128));
        let low = emit(&mut g, entry, mir::InstKind::Trunc(sum), mir::Ty::int(32));
        ret(&mut g, entry, Some(low));
        module.push_func(g);

        let (jit, ids) = jit_module(&module);
        let addr = func_addr(&jit, &ids, 0);
        let returns_undef: extern "C" fn() -> f64 = unsafe { std::mem::transmute(addr) };
        assert_eq!(returns_undef(), 0.0);
        assert_eq!(run_i32_0(&jit, &ids, 1), 1);
    }

    #[test]
    fn float_converts_execute() {
        // f(x: i32) = sftoi(sitof(x) + 0.5).
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![mir::Ty::int(32)], mir::Ty::int(32));
        let (mut f, entry) = mir::Function::new("f", sig);
        let x = param(&mut f, entry, mir::Ty::int(32), 0);
        let as_f = emit(&mut f, entry, mir::InstKind::SItoF(x), mir::Ty::Float(mir::FloatKind::F64));
        let half = f.push_const(mir::Const::Float {
            kind: mir::FloatKind::F64,
            bits: 0x3FE0000000000000,
        });
        let bumped = emit(&mut f, entry, mir::InstKind::FAdd(as_f, half), mir::Ty::Float(mir::FloatKind::F64));
        let back = emit(&mut f, entry, mir::InstKind::SFtoI(bumped), mir::Ty::int(32));
        ret(&mut f, entry, Some(back));
        module.push_func(f);
        let (jit, ids) = jit_module(&module);
        assert_eq!(run_i32(&jit, &ids, 0, 3), 3);

        // Unsigned roundtrip preserves large values a signed path would not.
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![mir::Ty::int(32)], mir::Ty::int(32));
        let (mut f, entry) = mir::Function::new("u", sig);
        let x = param(&mut f, entry, mir::Ty::int(32), 0);
        let as_f = emit(&mut f, entry, mir::InstKind::UItoF(x), mir::Ty::Float(mir::FloatKind::F64));
        let back = emit(&mut f, entry, mir::InstKind::UFtoI(as_f), mir::Ty::int(32));
        ret(&mut f, entry, Some(back));
        module.push_func(f);
        let (jit, ids) = jit_module(&module);
        assert_eq!(run_u32(&jit, &ids, 0, 4000000000), 4000000000);
    }

    #[test]
    fn float_converts_to_narrow_ints() {
        // Sub-word destinations convert at 32 bits and narrow: x64 has
        // no 16-bit float conversion.
        let mut module = mir::Module::new();
        let (mut f, entry) = mir::Function::new("to_u16", i32_sig(0));
        let v = f.push_const(mir::Const::Float {
            kind: mir::FloatKind::F64,
            bits: u128::from(5.25f64.to_bits()),
        });
        let conv = emit(&mut f, entry, mir::InstKind::UFtoI(v), mir::Ty::int(16));
        let wide = emit(&mut f, entry, mir::InstKind::Zext(conv), mir::Ty::int(32));
        ret(&mut f, entry, Some(wide));
        module.push_func(f);

        let (mut g, entry) = mir::Function::new("to_i16", i32_sig(0));
        let v = g.push_const(mir::Const::Float {
            kind: mir::FloatKind::F64,
            bits: u128::from((-7.9f64).to_bits()),
        });
        let conv = emit(&mut g, entry, mir::InstKind::SFtoI(v), mir::Ty::int(16));
        let wide = emit(&mut g, entry, mir::InstKind::Sext(conv), mir::Ty::int(32));
        ret(&mut g, entry, Some(wide));
        module.push_func(g);

        let (jit, ids) = jit_module(&module);
        assert_eq!(run_i32_0(&jit, &ids, 0), 5);
        assert_eq!(run_i32_0(&jit, &ids, 1), -7);
    }

    #[test]
    fn f64_arithmetic_executes() {
        let sig = mir::Signature::new(vec![mir::Ty::Float(mir::FloatKind::F64)], mir::Ty::Float(mir::FloatKind::F64));
        let mut module = mir::Module::new();
        let (mut f, entry) = mir::Function::new("f", sig);
        let x = param(&mut f, entry, mir::Ty::Float(mir::FloatKind::F64), 0);
        let two = f.push_const(mir::Const::Float {
            kind: mir::FloatKind::F64,
            bits: 0x4000000000000000,
        });
        let result = emit(&mut f, entry, mir::InstKind::FMul(x, two), mir::Ty::Float(mir::FloatKind::F64));
        ret(&mut f, entry, Some(result));
        module.push_func(f);
        let (jit, ids) = jit_module(&module);
        assert_eq!(run_f64(&jit, &ids, 0, 2.5), 5.0);
    }

    /// A foreign `long double` callee returns in st(0), not xmm0: the
    /// call declares no Clif return and fetches through getret. Uses
    /// the host libm `fabsl` (address only; the declared prototype is
    /// deliberately wrong and never called from Rust) and a
    /// one-instruction x87 shim for getret (Rust never touches st(0),
    /// so it survives from the foreign return intact).
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    #[test]
    fn foreign_f80_return_comes_from_st0() {
        #[link(name = "m")]
        unsafe extern "C" {
            fn fabsl(x: f64) -> f64;
        }
        // Pops the live st(0) into memory: the exact getret contract.
        unsafe extern "C" fn getret_shim(out: *mut u8) {
            unsafe {
                core::arch::asm!(
                    "fstpt (%rdi)",
                    in("rdi") out,
                    options(nostack, preserves_flags, att_syntax),
                );
            }
        }
        let f80 = mir::Ty::Float(mir::FloatKind::F80);
        let mut module = mir::Module::new();
        let ext = module.push_extern("fabsl", mir::Signature::new(vec![f80.clone()], f80.clone()));
        let (mut run, entry) = mir::Function::new("run", mir::Signature::new(vec![mir::Ty::int(64)], mir::Ty::Void));
        let out = param(&mut run, entry, mir::Ty::int(64), 0);
        // -2.5L: sign + exp 0x4000 over mantissa 0xA000....
        let arg = run.push_const(mir::Const::Float {
            kind: mir::FloatKind::F80,
            bits: 0xC000_A000_0000_0000_0000u128,
        });
        let got = emit(
            &mut run,
            entry,
            mir::InstKind::CallExtern {
                target: ext,
                args: [arg].into_iter().collect(),
            },
            f80,
        );
        emit_unit(
            &mut run,
            entry,
            mir::InstKind::Store {
                addr: out,
                val: got,
                flags: mir::MemFlags::plain(),
            },
        );
        ret(&mut run, entry, None);
        module.push_func(run);
        let (jit, ids) = jit_module_with(
            &module,
            &[
                ("fabsl", fabsl as *const () as usize as *const u8),
                ("__glccrt_f80_getret", getret_shim as *const () as usize as *const u8),
            ],
        );
        let addr = func_addr(&jit, &ids, 0);
        let mut buf = [0u8; 16];
        unsafe {
            let run_fn: extern "C" fn(*mut u8) = std::mem::transmute(addr);
            run_fn(buf.as_mut_ptr());
        }
        // +2.5L low 10 bytes; the carrier's pad is indeterminate.
        assert_eq!(&buf[..10], &[0, 0, 0, 0, 0, 0, 0, 0xA0, 0x00, 0x40], "fabsl(-2.5L) retrieved from st(0)");
    }

    /// Quad arithmetic lowers to the toolchain trampolines on
    /// AArch64/RISC-V (same out-pointer convention as the x87
    /// helpers): object emission succeeds and references the symbols.
    #[test]
    fn f128_arith_calls_quad_trampolines() {
        use cranelift_codegen::settings::Configurable;
        let cases: &[(fn(mir::Value, mir::Value) -> mir::InstKind, &str)] = &[
            (mir::InstKind::FAdd, "__glcc_quad_add"),
            (mir::InstKind::FSub, "__glcc_quad_sub"),
            (mir::InstKind::FMul, "__glcc_quad_mul"),
            (mir::InstKind::FDiv, "__glcc_quad_div"),
        ];
        for triple in ["aarch64-unknown-linux-gnu", "riscv64gc-unknown-linux-gnu"] {
            for (op, helper) in cases {
                let f128 = mir::Ty::Float(mir::FloatKind::F128);
                let mut module = mir::Module::new();
                let sig = mir::Signature::new(vec![f128.clone(), f128.clone()], f128.clone());
                let (mut f, entry) = mir::Function::new("f", sig);
                let a = param(&mut f, entry, f128.clone(), 0);
                let b = param(&mut f, entry, f128.clone(), 1);
                let sum = emit(&mut f, entry, op(a, b), f128.clone());
                ret(&mut f, entry, Some(sum));
                module.push_func(f);
                let mut flag_builder = cranelift_codegen::settings::builder();
                flag_builder.set("opt_level", "none").expect("flag");
                flag_builder.set("is_pic", "true").expect("flag");
                let flags = cranelift_codegen::settings::Flags::new(flag_builder);
                let isa = cranelift_codegen::isa::lookup(triple.parse().expect("triple"))
                    .expect("isa")
                    .finish(flags)
                    .expect("isa flags");
                let obj_builder = cranelift_object::ObjectBuilder::new(isa, "quadtest", cranelift_module::default_libcall_names()).expect("object builder");
                let mut obj = cranelift_object::ObjectModule::new(obj_builder);
                let target = crate::abi::Target::from_triple(triple).expect("target");
                lower_module(&module, &mut obj, target);
                let bytes = obj.finish().emit().expect("object emit");
                assert!(bytes.windows(helper.len()).any(|w| w == helper.as_bytes()), "{triple} object references {helper}");
            }
        }
    }

    /// x86-64 `__float128` arithmetic calls the vendored runtime directly (no
    /// adapter: the carrier already rides XMM registers): object
    /// emission succeeds and references the symbols.
    #[test]
    fn f128_arith_calls_libgcc_on_x86_64() {
        use cranelift_codegen::settings::Configurable;
        let cases: &[(fn(mir::Value, mir::Value) -> mir::InstKind, &str)] = &[
            (mir::InstKind::FAdd, "__glcc_addtf3"),
            (mir::InstKind::FSub, "__glcc_subtf3"),
            (mir::InstKind::FMul, "__glcc_multf3"),
            (mir::InstKind::FDiv, "__glcc_divtf3"),
        ];
        for (op, helper) in cases {
            let f128 = mir::Ty::Float(mir::FloatKind::F128);
            let mut module = mir::Module::new();
            let sig = mir::Signature::new(vec![f128.clone(), f128.clone()], f128.clone());
            let (mut f, entry) = mir::Function::new("f", sig);
            let a = param(&mut f, entry, f128.clone(), 0);
            let b = param(&mut f, entry, f128.clone(), 1);
            let sum = emit(&mut f, entry, op(a, b), f128.clone());
            ret(&mut f, entry, Some(sum));
            module.push_func(f);
            let mut flag_builder = cranelift_codegen::settings::builder();
            flag_builder.set("opt_level", "none").expect("flag");
            flag_builder.set("is_pic", "true").expect("flag");
            let flags = cranelift_codegen::settings::Flags::new(flag_builder);
            let isa = cranelift_codegen::isa::lookup("x86_64-unknown-linux-gnu".parse().expect("triple"))
                .expect("isa")
                .finish(flags)
                .expect("isa flags");
            let obj_builder = cranelift_object::ObjectBuilder::new(isa, "quadtest", cranelift_module::default_libcall_names()).expect("object builder");
            let mut obj = cranelift_object::ObjectModule::new(obj_builder);
            let target = crate::abi::Target::from_triple("x86_64-unknown-linux-gnu").expect("target");
            lower_module(&module, &mut obj, target);
            let bytes = obj.finish().emit().expect("object emit");
            assert!(bytes.windows(helper.len()).any(|w| w == helper.as_bytes()), "x86_64 object references {helper}");
        }
    }

    /// Entry-param stack slots are assigned in param order:
    /// `map_entry_params` collects per-owner pieces in a `BTreeMap`
    /// so iteration (which creates stack slots and emits stores)
    /// cannot permute incoming-register spills via a hash random
    /// seed. Lowering the same module repeatedly must give identical
    /// object bytes.
    #[test]
    fn entry_params_lower_deterministically() {
        use cranelift_codegen::settings::Configurable;
        let pair = || mir::Ty::Struct {
            fields: vec![mir::Ty::int(64), mir::Ty::int(64)],
            offsets: vec![0, 8],
            size: 16,
        };
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![pair(), pair(), pair(), pair()], mir::Ty::Void);
        let (mut f, entry) = mir::Function::new("f", sig);
        for i in 0..4 {
            param(&mut f, entry, pair(), i);
        }
        ret(&mut f, entry, None);
        module.push_func(f);
        let mut outputs = Vec::new();
        for _ in 0..8 {
            let mut flag_builder = cranelift_codegen::settings::builder();
            flag_builder.set("opt_level", "none").expect("flag");
            flag_builder.set("is_pic", "true").expect("flag");
            let flags = cranelift_codegen::settings::Flags::new(flag_builder);
            let isa = cranelift_codegen::isa::lookup("x86_64-unknown-linux-gnu".parse().expect("triple"))
                .expect("isa")
                .finish(flags)
                .expect("isa flags");
            let obj_builder = cranelift_object::ObjectBuilder::new(isa, "determtest", cranelift_module::default_libcall_names()).expect("object builder");
            let mut obj = cranelift_object::ObjectModule::new(obj_builder);
            let target = crate::abi::Target::from_triple("x86_64-unknown-linux-gnu").expect("target");
            lower_module(&module, &mut obj, target);
            outputs.push(obj.finish().emit().expect("object emit"));
        }
        for other in &outputs[1..] {
            assert_eq!(&outputs[0], other, "same MIR module lowered to different object bytes");
        }
    }

    /// Build `run(out, a, b)`: load an `F128` at each address, one
    /// binary op, store the `F128` result at `out`.
    fn qbinop_module(op: fn(mir::Value, mir::Value) -> mir::InstKind) -> mir::Module {
        let f128 = mir::Ty::Float(mir::FloatKind::F128);
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
        let (mut f, entry) = mir::Function::new("run", sig);
        let out = param(&mut f, entry, mir::Ty::int(64), 0);
        let a_ptr = param(&mut f, entry, mir::Ty::int(64), 1);
        let b_ptr = param(&mut f, entry, mir::Ty::int(64), 2);
        let a = emit(
            &mut f,
            entry,
            mir::InstKind::Load {
                addr: a_ptr,
                flags: mir::MemFlags::plain(),
            },
            f128.clone(),
        );
        let b = emit(
            &mut f,
            entry,
            mir::InstKind::Load {
                addr: b_ptr,
                flags: mir::MemFlags::plain(),
            },
            f128.clone(),
        );
        let r = emit(&mut f, entry, op(a, b), f128);
        emit_unit(
            &mut f,
            entry,
            mir::InstKind::Store {
                addr: out,
                val: r,
                flags: mir::MemFlags::plain(),
            },
        );
        ret(&mut f, entry, None);
        module.push_func(f);
        module
    }

    /// x86-64 `__float128` arithmetic executes through the vendored runtime: exact
    /// small cases, probed through the f64 oracles on both sides.
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    #[test]
    fn f128_arith_executes_x86_64() {
        // Addresses only, borrowed from the system toolchain for the
        // JIT linker (the vendored `rt/softfloat` copies are
        // bitwise-faithful; never called from Rust, and the
        // prototypes are deliberately integer-shaped).
        #[link(name = "gcc_s")]
        unsafe extern "C" {
            fn __addtf3(a: u128, b: u128) -> u128;
            fn __subtf3(a: u128, b: u128) -> u128;
            fn __multf3(a: u128, b: u128) -> u128;
            fn __divtf3(a: u128, b: u128) -> u128;
        }
        // (op, helper symbol, a as f64, b as f64, expected as f64).
        let cases: &[(fn(mir::Value, mir::Value) -> mir::InstKind, &str, f64, f64, f64)] = &[
            (mir::InstKind::FAdd, "__glcc_addtf3", 1.5, 2.5, 4.0),
            (mir::InstKind::FSub, "__glcc_subtf3", 1.0, 3.0, -2.0),
            (mir::InstKind::FMul, "__glcc_multf3", 2.0, 1.5, 3.0),
            (mir::InstKind::FDiv, "__glcc_divtf3", 7.0, 2.0, 3.5),
        ];
        for (op, helper, fa, fb, want) in cases {
            let module = qbinop_module(*op);
            let (jit, ids) = jit_module_with(
                &module,
                &[
                    ("__glcc_addtf3", __addtf3 as *const () as usize as *const u8),
                    ("__glcc_subtf3", __subtf3 as *const () as usize as *const u8),
                    ("__glcc_multf3", __multf3 as *const () as usize as *const u8),
                    ("__glcc_divtf3", __divtf3 as *const () as usize as *const u8),
                ],
            );
            let (alo, ahi) = crate::quad::ref_f64_to_quad(fa.to_bits());
            let (blo, bhi) = crate::quad::ref_f64_to_quad(fb.to_bits());
            let abuf = qbytes(alo, ahi);
            let bbuf = qbytes(blo, bhi);
            let mut out = [0u8; 16];
            unsafe {
                let func: extern "C" fn(*mut u8, *const u8, *const u8) = std::mem::transmute(func_addr(&jit, &ids, 0));
                func(out.as_mut_ptr(), abuf.as_ptr(), bbuf.as_ptr());
            }
            assert_eq!(qhalves(&out), crate::quad::ref_f64_to_quad(want.to_bits()), "{helper} of {fa} and {fb}");
        }
    }

    /// `long double` <-> `__float128` widening through the vendored runtime
    /// (`__extendxftf2`, exact): x87 bit patterns become their exact
    /// quad halves, NaN stays NaN.
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    #[test]
    fn f80_extend_executes() {
        // Address only, borrowed from the system toolchain (the
        // vendored copy is bitwise-faithful); deliberately
        // integer-shaped, never called from Rust.
        #[link(name = "gcc_s")]
        unsafe extern "C" {
            fn __extendxftf2(a: u128, b: u128) -> u128;
        }
        // (F80 bits, expected quad lo/hi or `None` for NaN).
        let cases: &[(u128, Option<(u64, u64)>)] = &[
            (0x3fff_8000_0000_0000_0000, Some((0, 0x3fff_0000_0000_0000))),                          // 1.0
            (0xbfff_c000_0000_0000_0000, Some(crate::quad::ref_f64_to_quad(0xbff8_0000_0000_0000))), // -1.5
            (0x8000_0000_0000_0000_0000, Some((0, 0x8000_0000_0000_0000))),                          // -0.0
            (0x7fff_8000_0000_0000_0000, Some((0, 0x7fff_0000_0000_0000))),                          // +inf
            (0x0000_0000_0000_0000_0001, Some((0x0002_0000_0000_0000, 0))),                          // min denormal -> 2^-16445
            (0x7fff_c000_0000_0000_0000, None),                                                      // qnan
        ];
        for (bits, want) in cases {
            let _f80 = mir::Ty::Float(mir::FloatKind::F80);
            let f128 = mir::Ty::Float(mir::FloatKind::F128);
            let mut module = mir::Module::new();
            let sig = mir::Signature::new(vec![mir::Ty::int(64)], mir::Ty::Void);
            let (mut f, entry) = mir::Function::new("run", sig);
            let out = param(&mut f, entry, mir::Ty::int(64), 0);
            let c = f.push_const(mir::Const::Float {
                kind: mir::FloatKind::F80,
                bits: *bits,
            });
            let q = emit(&mut f, entry, mir::InstKind::Fpext(c), f128);
            emit_unit(
                &mut f,
                entry,
                mir::InstKind::Store {
                    addr: out,
                    val: q,
                    flags: mir::MemFlags::plain(),
                },
            );
            ret(&mut f, entry, None);
            module.push_func(f);
            let (jit, ids) = jit_module_with(&module, &[("__glcc_extendxftf2", __extendxftf2 as *const () as usize as *const u8)]);
            let mut buf = [0u8; 16];
            unsafe {
                let func: extern "C" fn(*mut u8) = std::mem::transmute(func_addr(&jit, &ids, 0));
                func(buf.as_mut_ptr());
            }
            match want {
                Some((elo, ehi)) => assert_eq!(qhalves(&buf), (*elo, *ehi), "extend of {bits:#x}"),
                None => {
                    let (lo, hi) = qhalves(&buf);
                    assert!(crate::quad::ref_quad_is_nan(lo, hi), "extend of {bits:#x} stays NaN");
                }
            }
        }
    }

    /// Quad -> `long double` narrowing through the vendored runtime
    /// (`__trunctfxf2`, single rounding): exact values land on their
    /// x87 patterns (low 10 bytes; pad indeterminate), wide quads
    /// saturate to infinity, NaN stays NaN.
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    #[test]
    fn f80_trunc_executes() {
        // Address only, borrowed from the system toolchain like above.
        #[link(name = "gcc_s")]
        unsafe extern "C" {
            fn __trunctfxf2(a: u128) -> u128;
        }
        // Pops the live st(0): the exact getret contract (see above).
        unsafe extern "C" fn trunc_getret_shim(out: *mut u8) {
            unsafe {
                core::arch::asm!(
                    "fstpt (%rdi)",
                    in("rdi") out,
                    options(nostack, preserves_flags, att_syntax),
                );
            }
        }
        // (quad lo/hi, expected low 10 x87 bytes or `None` for NaN).
        let cases: &[((u64, u64), Option<[u8; 10]>)] = &[
            ((0, 0x3fff_0000_0000_0000), Some([0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0x3f])), // 1.0
            (crate::quad::ref_f64_to_quad(0xbff8_0000_0000_0000), Some([0, 0, 0, 0, 0, 0, 0, 0xc0, 0xff, 0xbf])), // -1.5
            ((0, 0x3bcd_0000_0000_0000), Some([0, 0, 0, 0, 0, 0, 0, 0x80, 0xcd, 0x3b])), // 2^-1074, exact in x87
            ((0xffff_ffff_ffff_ffff, 0x7ffe_ffff_ffff_ffff), Some([0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0x7f])), // quad max overflows to +inf
            ((0, 0x7fff_8000_0000_0000), None),                                          // qnan
        ];
        for ((ilo, ihi), want) in cases {
            let f128 = mir::Ty::Float(mir::FloatKind::F128);
            let f80 = mir::Ty::Float(mir::FloatKind::F80);
            let mut module = mir::Module::new();
            let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
            let (mut f, entry) = mir::Function::new("run", sig);
            let out = param(&mut f, entry, mir::Ty::int(64), 0);
            let inp = param(&mut f, entry, mir::Ty::int(64), 1);
            let v = emit(
                &mut f,
                entry,
                mir::InstKind::Load {
                    addr: inp,
                    flags: mir::MemFlags::plain(),
                },
                f128.clone(),
            );
            let n = emit(&mut f, entry, mir::InstKind::Fptrunc(v), f80.clone());
            emit_unit(
                &mut f,
                entry,
                mir::InstKind::Store {
                    addr: out,
                    val: n,
                    flags: mir::MemFlags::plain(),
                },
            );
            ret(&mut f, entry, None);
            module.push_func(f);
            let (jit, ids) = jit_module_with(
                &module,
                &[
                    ("__glcc_trunctfxf2", __trunctfxf2 as *const () as usize as *const u8),
                    ("__glccrt_f80_getret", trunc_getret_shim as *const () as usize as *const u8),
                ],
            );
            let inp = qbytes(*ilo, *ihi);
            let mut buf = [0u8; 16];
            unsafe {
                let func: extern "C" fn(*mut u8, *const u8) = std::mem::transmute(func_addr(&jit, &ids, 0));
                func(buf.as_mut_ptr(), inp.as_ptr());
            }
            match want {
                Some(bytes) => assert_eq!(&buf[..10], bytes, "trunc of {ilo:#x}/{ihi:#x}"),
                None => {
                    let exp = u16::from_le_bytes([buf[8], buf[9]]) & 0x7fff;
                    let mut mantissa = [0u8; 8];
                    mantissa.copy_from_slice(&buf[..8]);
                    assert_eq!(exp, 0x7fff, "trunc stays infinite-exponent");
                    assert_ne!(u64::from_le_bytes(mantissa), 0x8000_0000_0000_0000, "trunc stays NaN, not infinity");
                }
            }
        }
    }

    /// int128 -> `long double` through the vendored runtime
    /// (`__glcc_floattixf`/`__glcc_floatuntixf`, single rounding):
    /// exact values land on their x87 patterns (low 10 bytes; pad
    /// indeterminate), `u128::MAX` rounds up to 2^128.
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    #[test]
    fn f80_from_int128_executes() {
        // Addresses only, borrowed from the system toolchain like
        // above; deliberately integer-shaped, never called from Rust.
        #[link(name = "gcc_s")]
        unsafe extern "C" {
            fn __floattixf(a: u128, b: u128) -> u128;
            fn __floatuntixf(a: u128, b: u128) -> u128;
        }
        // Pops the live st(0): the exact getret contract (see above).
        unsafe extern "C" fn from_getret_shim(out: *mut u8) {
            unsafe {
                core::arch::asm!(
                    "fstpt (%rdi)",
                    in("rdi") out,
                    options(nostack, preserves_flags, att_syntax),
                );
            }
        }
        // (value, unsigned, expected low 10 x87 bytes).
        let cases: &[(u128, bool, [u8; 10])] = &[
            (1, false, [0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0x3f]),
            (1, true, [0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0x3f]),
            (u128::MAX, false, [0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0xbf]), // -1.0
            ((1 << 64) - 1, true, [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x3e, 0x40]),
            (1 << 64, true, [0, 0, 0, 0, 0, 0, 0, 0x80, 0x3f, 0x40]),
            (i64::MIN as u128, false, [0, 0, 0, 0, 0, 0, 0, 0x80, 0x3e, 0xc0]), // -2^63
            ((1 << 113) - 1, false, [0, 0, 0, 0, 0, 0, 0, 0x80, 0x70, 0x40]),   // rounds up to 2^113
            (1 << 113, false, [0, 0, 0, 0, 0, 0, 0, 0x80, 0x70, 0x40]),
            (u128::MAX, true, [0, 0, 0, 0, 0, 0, 0, 0x80, 0x7f, 0x40]),          // 2^128
            (i128::MIN as u128, false, [0, 0, 0, 0, 0, 0, 0, 0x80, 0x7e, 0xc0]), // -2^127
            (i128::MAX as u128, false, [0, 0, 0, 0, 0, 0, 0, 0x80, 0x7e, 0x40]), // 2^127
        ];
        for (v, unsigned, want) in cases {
            let f80 = mir::Ty::Float(mir::FloatKind::F80);
            let mut module = mir::Module::new();
            let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(128)], mir::Ty::Void);
            let (mut f, entry) = mir::Function::new("run", sig);
            let out = param(&mut f, entry, mir::Ty::int(64), 0);
            let x = param(&mut f, entry, mir::Ty::int(128), 1);
            let kind = if *unsigned { mir::InstKind::UItoF(x) } else { mir::InstKind::SItoF(x) };
            let r = emit(&mut f, entry, kind, f80);
            emit_unit(
                &mut f,
                entry,
                mir::InstKind::Store {
                    addr: out,
                    val: r,
                    flags: mir::MemFlags::plain(),
                },
            );
            ret(&mut f, entry, None);
            module.push_func(f);
            let (jit, ids) = jit_module_with(
                &module,
                &[
                    ("__glcc_floattixf", __floattixf as *const () as usize as *const u8),
                    ("__glcc_floatuntixf", __floatuntixf as *const () as usize as *const u8),
                    ("__glccrt_f80_getret", from_getret_shim as *const () as usize as *const u8),
                ],
            );
            let mut buf = [0u8; 16];
            unsafe {
                let func: extern "C" fn(*mut u8, u64, u64) = std::mem::transmute(func_addr(&jit, &ids, 0));
                func(buf.as_mut_ptr(), *v as u64, (*v >> 64) as u64);
            }
            assert_eq!(&buf[..10], want, "from_int128({v:#x}, unsigned={unsigned})");
        }
    }

    /// `long double` -> int128 through the vendored runtime
    /// (`__glcc_fixxfti`/`__glcc_fixunsxfti`), matching the toolchain
    /// exactly (vectors probed against it, including out-of-range).
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    #[test]
    fn f80_to_int128_executes() {
        // Addresses only, borrowed from the system toolchain like above.
        #[link(name = "gcc_s")]
        unsafe extern "C" {
            fn __fixxfti(a: u128) -> u128;
            fn __fixunsxfti(a: u128) -> u128;
        }
        // (F80 bits, signed expectation, unsigned expectation).
        // NaN (last row) asserts structurally below.
        let cases: &[(u128, u128, u128)] = &[
            (0x3fff_8000_0000_0000_0000, 1, 1),
            (0xbfff_c000_0000_0000_0000, u128::MAX, 0),       // -1.5
            (0x3ffe_8000_0000_0000_0000, 0, 0),               // 0.5
            (0x7fff_8000_0000_0000_0000, 0, 0),               // +inf
            (0xffff_8000_0000_0000_0000, 0, 0),               // -inf
            (0x0000_0000_0000_0000_0000, 0, 0),               // +0.0
            (0x407e_8000_0000_0000_0000, 1 << 127, 1 << 127), // +2^127
            (0xc07e_8000_0000_0000_0000, 1 << 127, 0),        // -2^127
        ];
        for unsigned in [false, true] {
            for (bits, s, u) in cases {
                let f80 = mir::Ty::Float(mir::FloatKind::F80);
                let mut module = mir::Module::new();
                let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
                let (mut f, entry) = mir::Function::new("run", sig);
                let out = param(&mut f, entry, mir::Ty::int(64), 0);
                let inp = param(&mut f, entry, mir::Ty::int(64), 1);
                let v = emit(
                    &mut f,
                    entry,
                    mir::InstKind::Load {
                        addr: inp,
                        flags: mir::MemFlags::plain(),
                    },
                    f80.clone(),
                );
                let kind = if unsigned { mir::InstKind::UFtoI(v) } else { mir::InstKind::SFtoI(v) };
                let r = emit(&mut f, entry, kind, mir::Ty::int(128));
                emit_unit(
                    &mut f,
                    entry,
                    mir::InstKind::Store {
                        addr: out,
                        val: r,
                        flags: mir::MemFlags::plain(),
                    },
                );
                ret(&mut f, entry, None);
                module.push_func(f);
                let (jit, ids) = jit_module_with(
                    &module,
                    &[
                        ("__glcc_fixxfti", __fixxfti as *const () as usize as *const u8),
                        ("__glcc_fixunsxfti", __fixunsxfti as *const () as usize as *const u8),
                    ],
                );
                let mut lo_b = [0u8; 8];
                let mut hi_b = [0u8; 8];
                lo_b.copy_from_slice(&bits.to_le_bytes()[..8]);
                hi_b.copy_from_slice(&bits.to_le_bytes()[8..]);
                let mut inp = [0u8; 16];
                inp[..8].copy_from_slice(&lo_b);
                inp[8..].copy_from_slice(&hi_b);
                let mut buf = [0u8; 16];
                unsafe {
                    let func: extern "C" fn(*mut u8, *const u8) = std::mem::transmute(func_addr(&jit, &ids, 0));
                    func(buf.as_mut_ptr(), inp.as_ptr());
                }
                let want = if unsigned { *u } else { *s };
                assert_eq!(u128::from_le_bytes(buf), want, "to_int128({bits:#x}, unsigned={unsigned})");
            }
        }
        // NaN converts structurally (exact payload is toolchain detail).
        for unsigned in [false, true] {
            let bits = 0x7fff_c000_0000_0000_0000u128;
            let f80 = mir::Ty::Float(mir::FloatKind::F80);
            let mut module = mir::Module::new();
            let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
            let (mut f, entry) = mir::Function::new("run", sig);
            let out = param(&mut f, entry, mir::Ty::int(64), 0);
            let inp = param(&mut f, entry, mir::Ty::int(64), 1);
            let v = emit(
                &mut f,
                entry,
                mir::InstKind::Load {
                    addr: inp,
                    flags: mir::MemFlags::plain(),
                },
                f80.clone(),
            );
            let kind = if unsigned { mir::InstKind::UFtoI(v) } else { mir::InstKind::SFtoI(v) };
            let r = emit(&mut f, entry, kind, mir::Ty::int(128));
            emit_unit(
                &mut f,
                entry,
                mir::InstKind::Store {
                    addr: out,
                    val: r,
                    flags: mir::MemFlags::plain(),
                },
            );
            ret(&mut f, entry, None);
            module.push_func(f);
            let (jit, ids) = jit_module_with(
                &module,
                &[
                    ("__glcc_fixxfti", __fixxfti as *const () as usize as *const u8),
                    ("__glcc_fixunsxfti", __fixunsxfti as *const () as usize as *const u8),
                ],
            );
            let inp = bits.to_le_bytes();
            let mut buf = [0u8; 16];
            unsafe {
                let func: extern "C" fn(*mut u8, *const u8) = std::mem::transmute(func_addr(&jit, &ids, 0));
                func(buf.as_mut_ptr(), inp.as_ptr());
            }
            assert_eq!(
                u128::from_le_bytes(buf),
                0x8000_0000_0000_0000_8000_0000_0000_0000u128,
                "to_int128(nan, unsigned={unsigned})"
            );
        }
    }

    /// Lower one `F128` op for a cross target without executing:
    /// catches ISLE gaps (like `scalar_size` for stray `F128`s) that
    /// pass x86-64-only episodically.
    fn cross_lower(op: &str, triple: &str, build: &dyn Fn(&mut mir::Function, mir::Block) -> mir::Value) {
        use cranelift_codegen::settings::Configurable;
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![], mir::Ty::int(64));
        let (mut f, entry) = mir::Function::new("f", sig);
        let v = build(&mut f, entry);
        f.sig.ret = f.values[v].ty.clone();
        ret(&mut f, entry, Some(v));
        module.push_func(f);
        let mut flag_builder = cranelift_codegen::settings::builder();
        flag_builder.set("opt_level", "none").expect("flag");
        flag_builder.set("is_pic", "true").expect("flag");
        let flags = cranelift_codegen::settings::Flags::new(flag_builder);
        let isa = cranelift_codegen::isa::lookup(triple.parse().expect("triple"))
            .expect("isa")
            .finish(flags)
            .expect("isa flags");
        let obj_builder = cranelift_object::ObjectBuilder::new(isa, "quadtest", cranelift_module::default_libcall_names()).expect("object builder");
        let mut obj = cranelift_object::ObjectModule::new(obj_builder);
        let target = crate::abi::Target::from_triple(triple).expect("target");
        lower_module(&module, &mut obj, target);
        obj.finish().emit().unwrap_or_else(|_| panic!("{op} lowers on {triple}"));
    }

    /// `FNeg`/`Fcmp`/conversions on `F128` inline as integer IR, so
    /// they lower on every target (only arithmetic needs trampolines).
    #[test]
    fn f128_nonarith_lowers_cross() {
        use mir::FcmpPred::*;
        let one = 0x3fff_0000_0000_0000u64;
        for triple in ["aarch64-unknown-linux-gnu", "riscv64gc-unknown-linux-gnu"] {
            // Negation of 1.0.
            cross_lower("fneg", triple, &|f, entry| {
                let c = f.push_const(mir::Const::Float {
                    kind: mir::FloatKind::F128,
                    bits: (one as u128) << 64,
                });
                let n = emit(f, entry, mir::InstKind::FNeg(c), mir::Ty::Float(mir::FloatKind::F128));
                emit(f, entry, mir::InstKind::SFtoI(n), mir::Ty::int(32))
            });
            // Each comparison against itself.
            for pred in [Eq, Ne, Lt, Le, Gt, Ge, Uno, Ord] {
                cross_lower("fcmp", triple, &|f, entry| {
                    let c = f.push_const(mir::Const::Float {
                        kind: mir::FloatKind::F128,
                        bits: (one as u128) << 64,
                    });
                    let t = emit(f, entry, mir::InstKind::Fcmp { pred, lhs: c, rhs: c }, mir::Ty::Bool);
                    emit(f, entry, mir::InstKind::Zext(t), mir::Ty::int(32))
                });
            }
            // int -> quad -> int roundtrip.
            cross_lower("int-roundtrip", triple, &|f, entry| {
                let seven = iconst(f, 64, 7);
                let q = emit(f, entry, mir::InstKind::SItoF(seven), mir::Ty::Float(mir::FloatKind::F128));
                emit(f, entry, mir::InstKind::SFtoI(q), mir::Ty::int(64))
            });
            // 128-bit int -> quad -> int roundtrip (rounding path,
            // multi-piece scalar calling convention included).
            cross_lower("int128-roundtrip", triple, &|f, entry| {
                let seven = iconst(f, 128, 7);
                let q = emit(f, entry, mir::InstKind::SItoF(seven), mir::Ty::Float(mir::FloatKind::F128));
                emit(f, entry, mir::InstKind::SFtoI(q), mir::Ty::int(128))
            });
            // ceil/floor roundtrip through ints.
            for op in [mir::InstKind::Ceil as fn(mir::Value) -> mir::InstKind, mir::InstKind::Floor] {
                cross_lower("round", triple, &|f, entry| {
                    let c = f.push_const(mir::Const::Float {
                        kind: mir::FloatKind::F128,
                        bits: (one as u128) << 64,
                    });
                    let r = emit(f, entry, op(c), mir::Ty::Float(mir::FloatKind::F128));
                    emit(f, entry, mir::InstKind::SFtoI(r), mir::Ty::int(64))
                });
            }
            // double -> quad -> double roundtrip (bit-exact through ints).
            cross_lower("float-roundtrip", triple, &|f, entry| {
                let half = f.push_const(mir::Const::Float {
                    kind: mir::FloatKind::F64,
                    bits: 0x3FE0_0000_0000_0000,
                });
                let q = emit(f, entry, mir::InstKind::Fpext(half), mir::Ty::Float(mir::FloatKind::F128));
                let back = emit(f, entry, mir::InstKind::Fptrunc(q), mir::Ty::Float(mir::FloatKind::F64));
                emit(f, entry, mir::InstKind::Bitcast(back), mir::Ty::int(64))
            });
        }
    }

    /// `F80` unordered/ordered lower on x86-64 through the `ne`
    /// self-comparison (no new asm): object emission succeeds with
    /// the helper imports left unresolved.
    #[test]
    fn f80_unordered_lowers_x86_64() {
        use mir::FcmpPred::*;
        let one_bits = ((0x3fffu128) << 64) | 0x8000_0000_0000_0000u128;
        for pred in [Uno, Ord] {
            cross_lower("f80-fcmp", "x86_64-unknown-linux-gnu", &|f, entry| {
                let c = f.push_const(mir::Const::Float {
                    kind: mir::FloatKind::F80,
                    bits: one_bits,
                });
                let t = emit(f, entry, mir::InstKind::Fcmp { pred, lhs: c, rhs: c }, mir::Ty::Bool);
                emit(f, entry, mir::InstKind::Zext(t), mir::Ty::int(32))
            });
        }
    }

    /// `F80` ceil/floor lower on x86-64 through the frndint helpers:
    /// object emission succeeds with the imports left unresolved.
    #[test]
    fn f80_round_lowers_x86_64() {
        let one_bits = ((0x3fffu128) << 64) | 0x8000_0000_0000_0000u128;
        for op in [mir::InstKind::Ceil as fn(mir::Value) -> mir::InstKind, mir::InstKind::Floor] {
            cross_lower("f80-round", "x86_64-unknown-linux-gnu", &|f, entry| {
                let c = f.push_const(mir::Const::Float {
                    kind: mir::FloatKind::F80,
                    bits: one_bits,
                });
                let r = emit(f, entry, op(c), mir::Ty::Float(mir::FloatKind::F80));
                emit(f, entry, mir::InstKind::SFtoI(r), mir::Ty::int(64))
            });
        }
    }

    /// Quad differential tests: builders against the `quad::ref_*`
    /// oracles through memory. `F128` values cross into Rust as
    /// 16-byte buffers (the carrier never names an `f128` in a Rust
    /// signature); every builder here is integer IR plus moves, so
    /// x86-64 JIT executes them exactly.
    fn qbytes(lo: u64, hi: u64) -> [u8; 16] {
        let mut b = [0u8; 16];
        b[..8].copy_from_slice(&lo.to_le_bytes());
        b[8..].copy_from_slice(&hi.to_le_bytes());
        b
    }

    fn qhalves(buf: &[u8]) -> (u64, u64) {
        (u64::from_le_bytes(buf[..8].try_into().unwrap()), u64::from_le_bytes(buf[8..16].try_into().unwrap()))
    }

    fn run_q1(jit: &JITModule, ids: &std::collections::HashMap<mir::Func, cranelift_module::FuncId>, index: usize, out: *mut u8, inp: *const u8) {
        unsafe {
            let func: extern "C" fn(*mut u8, *const u8) = std::mem::transmute(func_addr(jit, ids, index));
            func(out, inp);
        }
    }

    /// Build `run(out, inp)`: load an `F128` at `inp`, apply one
    /// unary op, store the `F128` result at `out`.
    fn qunary_module(op: fn(mir::Value) -> mir::InstKind) -> mir::Module {
        let f128 = mir::Ty::Float(mir::FloatKind::F128);
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
        let (mut f, entry) = mir::Function::new("run", sig);
        let out = param(&mut f, entry, mir::Ty::int(64), 0);
        let inp = param(&mut f, entry, mir::Ty::int(64), 1);
        let v = emit(
            &mut f,
            entry,
            mir::InstKind::Load {
                addr: inp,
                flags: mir::MemFlags::plain(),
            },
            f128.clone(),
        );
        let r = emit(&mut f, entry, op(v), f128.clone());
        emit_unit(
            &mut f,
            entry,
            mir::InstKind::Store {
                addr: out,
                val: r,
                flags: mir::MemFlags::plain(),
            },
        );
        ret(&mut f, entry, None);
        module.push_func(f);
        module
    }

    #[test]
    fn quad_neg_executes() {
        let module = qunary_module(mir::InstKind::FNeg);
        let (jit, ids) = jit_module(&module);
        // (input lo/hi, expected lo/hi): sign flip, payload kept.
        let cases: &[((u64, u64), (u64, u64))] = &[
            ((0, 0x3fff_8000_0000_0000), (0, 0xbfff_8000_0000_0000)),                                         // 1.5
            ((0, 0x8000_0000_0000_0000), (0, 0x0000_0000_0000_0000)),                                         // -0 -> +0
            ((0, 0x0000_0000_0000_0000), (0, 0x8000_0000_0000_0000)),                                         // +0 -> -0
            ((0x1234_5678_9abc_def0, 0x7fff_badc_0ffe_e0d0), (0x1234_5678_9abc_def0, 0xffff_badc_0ffe_e0d0)), // nan payload kept
            ((0, 0x7fff_0000_0000_0000), (0, 0xffff_0000_0000_0000)),                                         // inf
        ];
        for ((ilo, ihi), (elo, ehi)) in cases {
            let inp = qbytes(*ilo, *ihi);
            let mut out = [0u8; 16];
            run_q1(&jit, &ids, 0, out.as_mut_ptr(), inp.as_ptr());
            assert_eq!(qhalves(&out), (*elo, *ehi), "neg of {ilo:#x}/{ihi:#x}");
        }
    }

    #[test]
    fn quad_round_executes() {
        let q = crate::quad::ref_f64_to_quad;
        let inputs: &[(u64, u64)] = &[
            (0, 0x0000_0000_0000_0000),                     // +0
            (0, 0x8000_0000_0000_0000),                     // -0
            q(0x3ff0_0000_0000_0000),                       // 1.0
            q(0xbff0_0000_0000_0000),                       // -1.0
            q(0x3ff8_0000_0000_0000),                       // 1.5
            q(0xbff8_0000_0000_0000),                       // -1.5
            q(0x3fe0_0000_0000_0000),                       // 0.5
            q(0xbfe0_0000_0000_0000),                       // -0.5
            q(0x40c8_0000_0000_0000),                       // 10000.75
            q(0xc0c8_0000_0000_0000),                       // -10000.75
            (1, 0x0000_0000_0000_0000),                     // tiny denormal
            (1, 0x8000_0000_0000_0000),                     // tiny negative denormal
            (0, 0x403e_0000_0000_0000),                     // 2^63 integral
            (0, 0x3bcd_0000_0000_0000),                     // 2^-1074 (f64 min-denormal widened)
            (0, 0xbbcd_0000_0000_0000),                     // -2^-1074
            q(0x7ff0_0000_0000_0000),                       // +inf
            q(0xfff0_0000_0000_0000),                       // -inf
            q(0x7ff8_0000_0000_0000),                       // qnan
            (0xa000_0000_0000_0000, 0x3ffb_9999_9999_9999), // 0.1-ish
        ];
        for (op, oracle) in [
            (
                mir::InstKind::Ceil as fn(mir::Value) -> mir::InstKind,
                crate::quad::ref_quad_ceil as fn(u64, u64) -> (u64, u64),
            ),
            (
                mir::InstKind::Floor as fn(mir::Value) -> mir::InstKind,
                crate::quad::ref_quad_floor as fn(u64, u64) -> (u64, u64),
            ),
        ] {
            let module = qunary_module(op);
            let (jit, ids) = jit_module(&module);
            for (ilo, ihi) in inputs {
                let inp = qbytes(*ilo, *ihi);
                let mut out = [0u8; 16];
                run_q1(&jit, &ids, 0, out.as_mut_ptr(), inp.as_ptr());
                assert_eq!(qhalves(&out), oracle(*ilo, *ihi), "round of {ilo:#x}/{ihi:#x}");
            }
        }
    }

    /// Build `run(out, a, b)`: load an `F128` at each address, one
    /// `Fcmp`, store the `I8` result at `out`.
    fn qcmp_module(pred: mir::FcmpPred) -> mir::Module {
        let f128 = mir::Ty::Float(mir::FloatKind::F128);
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
        let (mut f, entry) = mir::Function::new("run", sig);
        let out = param(&mut f, entry, mir::Ty::int(64), 0);
        let a_ptr = param(&mut f, entry, mir::Ty::int(64), 1);
        let b_ptr = param(&mut f, entry, mir::Ty::int(64), 2);
        let a = emit(
            &mut f,
            entry,
            mir::InstKind::Load {
                addr: a_ptr,
                flags: mir::MemFlags::plain(),
            },
            f128.clone(),
        );
        let b = emit(
            &mut f,
            entry,
            mir::InstKind::Load {
                addr: b_ptr,
                flags: mir::MemFlags::plain(),
            },
            f128.clone(),
        );
        let t = emit(&mut f, entry, mir::InstKind::Fcmp { pred, lhs: a, rhs: b }, mir::Ty::Bool);
        emit_unit(
            &mut f,
            entry,
            mir::InstKind::Store {
                addr: out,
                val: t,
                flags: mir::MemFlags::plain(),
            },
        );
        ret(&mut f, entry, None);
        module.push_func(f);
        module
    }

    #[test]
    fn quad_cmp_executes() {
        use mir::FcmpPred::*;
        let one = crate::quad::ref_f64_to_quad(0x3ff0_0000_0000_0000);
        let two = crate::quad::ref_f64_to_quad(0x4000_0000_0000_0000);
        let negtwo = crate::quad::ref_f64_to_quad(0xc000_0000_0000_0000);
        let pzero = (0u64, 0u64);
        let nzero = (0u64, 0x8000_0000_0000_0000u64);
        let inf = crate::quad::ref_f64_to_quad(0x7ff0_0000_0000_0000);
        let nan = crate::quad::ref_f64_to_quad(0x7ff8_0000_0000_0000);
        let pairs: &[((u64, u64), (u64, u64))] = &[(one, one), (one, two), (negtwo, one), (pzero, nzero), (nan, one), (one, inf), (negtwo, nzero)];
        for (pred, check) in [
            (Eq, crate::quad::ref_quad_eq as fn(u64, u64, u64, u64) -> bool),
            (Ne, (|a, b, c, d| !crate::quad::ref_quad_eq(a, b, c, d)) as fn(u64, u64, u64, u64) -> bool),
            (Lt, crate::quad::ref_quad_lt as fn(u64, u64, u64, u64) -> bool),
            (
                Le,
                (|a, b, c, d| crate::quad::ref_quad_lt(a, b, c, d) || crate::quad::ref_quad_eq(a, b, c, d)) as fn(u64, u64, u64, u64) -> bool,
            ),
            (Gt, (|a, b, c, d| crate::quad::ref_quad_lt(c, d, a, b)) as fn(u64, u64, u64, u64) -> bool),
            (
                Ge,
                (|a, b, c, d| crate::quad::ref_quad_lt(c, d, a, b) || crate::quad::ref_quad_eq(a, b, c, d)) as fn(u64, u64, u64, u64) -> bool,
            ),
            (
                Uno,
                (|a, b, c, d| crate::quad::ref_quad_is_nan(a, b) || crate::quad::ref_quad_is_nan(c, d)) as fn(u64, u64, u64, u64) -> bool,
            ),
            (
                Ord,
                (|a, b, c, d| !(crate::quad::ref_quad_is_nan(a, b) || crate::quad::ref_quad_is_nan(c, d))) as fn(u64, u64, u64, u64) -> bool,
            ),
        ] {
            let module = qcmp_module(pred);
            let (jit, ids) = jit_module(&module);
            for ((alo, ahi), (blo, bhi)) in pairs {
                let abuf = qbytes(*alo, *ahi);
                let bbuf = qbytes(*blo, *bhi);
                let mut out = [0u8; 1];
                unsafe {
                    let func: extern "C" fn(*mut u8, *const u8, *const u8) = std::mem::transmute(func_addr(&jit, &ids, 0));
                    func(out.as_mut_ptr(), abuf.as_ptr(), bbuf.as_ptr());
                }
                assert_eq!(out[0], u8::from(check(*alo, *ahi, *blo, *bhi)), "cmp {pred:?} of {alo:#x}/{ahi:#x} vs {blo:#x}/{bhi:#x}");
            }
        }
    }

    /// Build `run(out, x)`: convert one `I64` to `F128` (signed or
    /// unsigned), store the halves at `out`.
    fn qfrom_int_module(unsigned: bool) -> mir::Module {
        let f128 = mir::Ty::Float(mir::FloatKind::F128);
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
        let (mut f, entry) = mir::Function::new("run", sig);
        let out = param(&mut f, entry, mir::Ty::int(64), 0);
        let x = param(&mut f, entry, mir::Ty::int(64), 1);
        let kind = if unsigned { mir::InstKind::UItoF(x) } else { mir::InstKind::SItoF(x) };
        let r = emit(&mut f, entry, kind, f128);
        emit_unit(
            &mut f,
            entry,
            mir::InstKind::Store {
                addr: out,
                val: r,
                flags: mir::MemFlags::plain(),
            },
        );
        ret(&mut f, entry, None);
        module.push_func(f);
        module
    }

    #[test]
    fn quad_from_int_executes() {
        for unsigned in [false, true] {
            let module = qfrom_int_module(unsigned);
            let (jit, ids) = jit_module(&module);
            for v in [0u64, 1, 2, 123456789, 0x7fff_ffff_ffff_ffff, 0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff] {
                let mut out = [0u8; 16];
                unsafe {
                    let func: extern "C" fn(*mut u8, u64) = std::mem::transmute(func_addr(&jit, &ids, 0));
                    func(out.as_mut_ptr(), v);
                }
                assert_eq!(qhalves(&out), crate::quad::ref_int_to_quad(v, unsigned), "from_int({v:#x}, unsigned={unsigned})");
            }
        }
    }

    /// Build `run(out, inp)`: load an `F128` at `inp`, convert to
    /// `I64` (signed or unsigned), store at `out`.
    fn qto_int_module(unsigned: bool) -> mir::Module {
        let f128 = mir::Ty::Float(mir::FloatKind::F128);
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
        let (mut f, entry) = mir::Function::new("run", sig);
        let out = param(&mut f, entry, mir::Ty::int(64), 0);
        let inp = param(&mut f, entry, mir::Ty::int(64), 1);
        let v = emit(
            &mut f,
            entry,
            mir::InstKind::Load {
                addr: inp,
                flags: mir::MemFlags::plain(),
            },
            f128,
        );
        let kind = if unsigned { mir::InstKind::UFtoI(v) } else { mir::InstKind::SFtoI(v) };
        let r = emit(&mut f, entry, kind, mir::Ty::int(64));
        emit_unit(
            &mut f,
            entry,
            mir::InstKind::Store {
                addr: out,
                val: r,
                flags: mir::MemFlags::plain(),
            },
        );
        ret(&mut f, entry, None);
        module.push_func(f);
        module
    }

    #[test]
    fn quad_to_int_executes() {
        // (lo, hi) inputs spanning exact, fractional, saturation, and
        // NaN/Inf edges; expectations from the oracles.
        let quads: &[(u64, u64)] = &[
            (0x0000_0000_0000_0000, 0x3fff_0000_0000_0000), // 1.0
            (0x0000_0000_0000_0000, 0xbfff_0000_0000_0000), // -1.0
            (0x0000_0000_0000_0000, 0x403e_0000_0000_0000), // 2^63
            (0xffff_ffff_ffff_ffff, 0x403e_ffff_ffff_ffff), // 2^64-1
            (0x0000_0000_0000_0000, 0x403f_0000_0000_0000), // 2^64
            (0x0000_0000_0000_0000, 0x7fff_0000_0000_0000), // +inf
            (0x0000_0000_0000_0000, 0xffff_0000_0000_0000), // -inf
            (0x0000_0000_0000_0001, 0x7fff_8000_0000_0000), // nan
            (0x8000_0000_0000_0000, 0x4005_c000_0000_0000), // 123.456-ish fractional
            (0x0000_0000_0000_0000, 0x0000_0000_0000_0000), // +0
        ];
        for unsigned in [false, true] {
            let module = qto_int_module(unsigned);
            let (jit, ids) = jit_module(&module);
            for (lo, hi) in quads {
                let inp = qbytes(*lo, *hi);
                let mut out = [0u8; 8];
                run_q1(&jit, &ids, 0, out.as_mut_ptr(), inp.as_ptr());
                let got = u64::from_le_bytes(out);
                let want = if unsigned {
                    crate::quad::ref_quad_to_u64(*lo, *hi)
                } else {
                    crate::quad::ref_quad_to_s64(*lo, *hi)
                };
                assert_eq!(got, want, "to_int({lo:#x}/{hi:#x}, unsigned={unsigned})");
            }
        }
    }

    /// Build `run(out, x)`: convert one `I128` to `F128` (signed or
    /// unsigned), store the halves at `out`. The 128-bit argument
    /// arrives as a register pair under the hood.
    fn qfrom_int128_module(unsigned: bool) -> mir::Module {
        let f128 = mir::Ty::Float(mir::FloatKind::F128);
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(128)], mir::Ty::Void);
        let (mut f, entry) = mir::Function::new("run", sig);
        let out = param(&mut f, entry, mir::Ty::int(64), 0);
        let x = param(&mut f, entry, mir::Ty::int(128), 1);
        let kind = if unsigned { mir::InstKind::UItoF(x) } else { mir::InstKind::SItoF(x) };
        let r = emit(&mut f, entry, kind, f128);
        emit_unit(
            &mut f,
            entry,
            mir::InstKind::Store {
                addr: out,
                val: r,
                flags: mir::MemFlags::plain(),
            },
        );
        ret(&mut f, entry, None);
        module.push_func(f);
        module
    }

    #[test]
    fn quad_from_int128_executes() {
        // (value, unsigned): exact, halfway, saturation-adjacent.
        let cases: &[(u128, bool)] = &[
            (0, false),
            (0, true),
            (1, false),
            (1, true),
            ((1 << 64) - 1, false),
            ((1 << 64) - 1, true),
            (1 << 64, true),
            (u64::MAX as u128, true),
            (i64::MAX as u128, false),
            (i64::MIN as u128, false),
            ((1 << 113) - 1, false),
            (1 << 113, false),
            ((1 << 113) + 1, false),
            ((1 << 113) + 3, false),
            (u128::MAX, false),
            (u128::MAX, true),
            (u128::MAX - 1, true),
            (i128::MAX as u128, false),
            (i128::MIN as u128, false),
            ((1u128 << 113) + 1, true),
            (0, true),
        ];
        for unsigned in [false, true] {
            let module = qfrom_int128_module(unsigned);
            let (jit, ids) = jit_module(&module);
            for (v, _u) in cases.iter().filter(|(_, u)| *u == unsigned) {
                let (lo, hi) = ((*v as u64), ((*v >> 64) as u64));
                let mut out = [0u8; 16];
                unsafe {
                    let func: extern "C" fn(*mut u8, u64, u64) = std::mem::transmute(func_addr(&jit, &ids, 0));
                    func(out.as_mut_ptr(), lo, hi);
                }
                assert_eq!(qhalves(&out), crate::quad::ref_int_to_quad128(*v, unsigned), "from_int128({v:#x}, unsigned={unsigned})");
            }
        }
    }

    /// Build `run(out, inp)`: load an `F128` at `inp`, convert to
    /// `I128` (signed or unsigned), store the 16 bytes at `out`.
    fn qto_int128_module(unsigned: bool) -> mir::Module {
        let f128 = mir::Ty::Float(mir::FloatKind::F128);
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
        let (mut f, entry) = mir::Function::new("run", sig);
        let out = param(&mut f, entry, mir::Ty::int(64), 0);
        let inp = param(&mut f, entry, mir::Ty::int(64), 1);
        let v = emit(
            &mut f,
            entry,
            mir::InstKind::Load {
                addr: inp,
                flags: mir::MemFlags::plain(),
            },
            f128,
        );
        let kind = if unsigned { mir::InstKind::UFtoI(v) } else { mir::InstKind::SFtoI(v) };
        let r = emit(&mut f, entry, kind, mir::Ty::int(128));
        emit_unit(
            &mut f,
            entry,
            mir::InstKind::Store {
                addr: out,
                val: r,
                flags: mir::MemFlags::plain(),
            },
        );
        ret(&mut f, entry, None);
        module.push_func(f);
        module
    }

    #[test]
    fn quad_to_int128_executes() {
        // (lo, hi) inputs spanning exact, fractional, saturation, and
        // NaN/Inf edges; expectations from the oracles.
        let quads: &[(u64, u64)] = &[
            (0x0000_0000_0000_0000, 0x3fff_0000_0000_0000), // 1.0
            (0x0000_0000_0000_0000, 0xbfff_0000_0000_0000), // -1.0
            (0x0000_0000_0000_0000, 0x3ffe_0000_0000_0000), // 0.5
            (0x0000_0000_0000_0000, 0x403e_0000_0000_0000), // 2^63
            (0x0000_0000_0000_0000, 0x403f_0000_0000_0000), // 2^64
            (0x0000_0000_0000_0000, 0x4063_0000_0000_0000), // 2^100
            (0x0000_0000_0000_0000, 0x407d_0000_0000_0000), // 2^126
            (0x0000_0000_0000_0000, 0x407e_0000_0000_0000), // 2^127
            (0x0000_0000_0000_0000, 0xc07e_0000_0000_0000), // -2^127
            (0x0000_0000_0000_0000, 0x407f_0000_0000_0000), // 2^128
            (0x0000_0000_0000_0000, 0x7fff_0000_0000_0000), // +inf
            (0x0000_0000_0000_0000, 0xffff_0000_0000_0000), // -inf
            (0x0000_0000_0000_0001, 0x7fff_8000_0000_0000), // nan
            (0x8000_0000_0000_0000, 0x4005_c000_0000_0000), // 123.456-ish fractional
            (0x0000_0000_0000_0000, 0x0000_0000_0000_0000), // +0
        ];
        for unsigned in [false, true] {
            let module = qto_int128_module(unsigned);
            let (jit, ids) = jit_module(&module);
            for (lo, hi) in quads {
                let inp = qbytes(*lo, *hi);
                let mut out = [0u8; 16];
                run_q1(&jit, &ids, 0, out.as_mut_ptr(), inp.as_ptr());
                let got = u128::from_le_bytes(out);
                let want = if unsigned {
                    crate::quad::ref_quad_to_u128(*lo, *hi)
                } else {
                    crate::quad::ref_quad_to_s128(*lo, *hi)
                };
                assert_eq!(got, want, "to_int128({lo:#x}/{hi:#x}, unsigned={unsigned})");
            }
        }
    }

    /// Build `run(out, x)`: widen one `F32`/`F64` value to `F128`,
    /// store the halves at `out`.
    fn qwiden_module(bits: u32) -> mir::Module {
        let f128 = mir::Ty::Float(mir::FloatKind::F128);
        let (pty, fty) = if bits == 32 {
            (mir::Ty::int(32), mir::Ty::Float(mir::FloatKind::F32))
        } else {
            (mir::Ty::int(64), mir::Ty::Float(mir::FloatKind::F64))
        };
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![mir::Ty::int(64), pty.clone()], mir::Ty::Void);
        let (mut f, entry) = mir::Function::new("run", sig);
        let out = param(&mut f, entry, mir::Ty::int(64), 0);
        let x = param(&mut f, entry, pty, 1);
        let as_f = emit(&mut f, entry, mir::InstKind::Bitcast(x), fty);
        let r = emit(&mut f, entry, mir::InstKind::Fpext(as_f), f128);
        emit_unit(
            &mut f,
            entry,
            mir::InstKind::Store {
                addr: out,
                val: r,
                flags: mir::MemFlags::plain(),
            },
        );
        ret(&mut f, entry, None);
        module.push_func(f);
        module
    }

    #[test]
    fn quad_widen_executes() {
        for v in [
            0x0000_0000u32,
            0x8000_0000,
            0x3fc0_0000,
            0x3dcc_cccd,
            0x0080_0000,
            0x0000_0001,
            0x7f80_0000,
            0x7fc0_0000,
            0x7f80_0001,
        ] {
            let module = qwiden_module(32);
            let (jit, ids) = jit_module(&module);
            let mut out = [0u8; 16];
            unsafe {
                let func: extern "C" fn(*mut u8, u32) = std::mem::transmute(func_addr(&jit, &ids, 0));
                func(out.as_mut_ptr(), v);
            }
            assert_eq!(qhalves(&out), crate::quad::ref_f32_to_quad(v), "widen f32 {v:#x}");
        }
        for v in [
            0u64,
            0x8000_0000_0000_0000,
            0x3ff8_0000_0000_0000,
            0x3fb9_9999_9999_999a,
            0x0000_0000_0000_0001,
            0x0010_0000_0000_0000,
            0x7ff0_0000_0000_0000,
            0x7ff8_0000_0000_0000,
            0x7ff0_0000_0000_0001,
        ] {
            let module = qwiden_module(64);
            let (jit, ids) = jit_module(&module);
            let mut out = [0u8; 16];
            unsafe {
                let func: extern "C" fn(*mut u8, u64) = std::mem::transmute(func_addr(&jit, &ids, 0));
                func(out.as_mut_ptr(), v);
            }
            assert_eq!(qhalves(&out), crate::quad::ref_f64_to_quad(v), "widen f64 {v:#x}");
        }
    }

    /// Build `run(out, inp)`: load an `F128` at `inp`, narrow to
    /// `F32`/`F64`, store the bits at `out`.
    fn qnarrow_module(bits: u32) -> mir::Module {
        let f128 = mir::Ty::Float(mir::FloatKind::F128);
        let (fty, sty) = if bits == 32 {
            (mir::Ty::Float(mir::FloatKind::F32), mir::Ty::int(32))
        } else {
            (mir::Ty::Float(mir::FloatKind::F64), mir::Ty::int(64))
        };
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
        let (mut f, entry) = mir::Function::new("run", sig);
        let out = param(&mut f, entry, mir::Ty::int(64), 0);
        let inp = param(&mut f, entry, mir::Ty::int(64), 1);
        let v = emit(
            &mut f,
            entry,
            mir::InstKind::Load {
                addr: inp,
                flags: mir::MemFlags::plain(),
            },
            f128,
        );
        let n = emit(&mut f, entry, mir::InstKind::Fptrunc(v), fty);
        let b = emit(&mut f, entry, mir::InstKind::Bitcast(n), sty);
        emit_unit(
            &mut f,
            entry,
            mir::InstKind::Store {
                addr: out,
                val: b,
                flags: mir::MemFlags::plain(),
            },
        );
        ret(&mut f, entry, None);
        module.push_func(f);
        module
    }

    #[test]
    fn quad_narrow_executes() {
        let quads: &[(u64, u64)] = &[
            (0x0000_0000_0000_0000, 0x3fff_0000_0000_0000),       // 1.0
            (0xa000_0000_0000_0000, 0x3ffb_9999_9999_9999),       // 0.1-ish
            (0xffff_ffff_ffff_ffff, 0x7ffe_ffff_ffff_ffff),       // quad max -> inf
            (0x0000_0000_0000_0000, 0x0001_0000_0000_0000),       // tiny -> 0
            (0x0000_0000_0000_0001, 0x7fff_8000_0000_0000),       // nan
            (0x1234_5678_9abc_def0, 0x7fff_badc_0ffe_e0d0),       // nan payload
            (1u64 << 59, 0x3fff_0000_0000_0000),                  // halfway, rounds down
            ((1u64 << 59) | (1u64 << 52), 0x3fff_0000_0000_0000), // halfway + sticky, rounds up
            (0x0000_0000_0000_0000, 0x3bcd_0000_0000_0000),       // 2^-1074 -> f64 min-denormal
            (0x0000_0000_0000_0001, 0x3bcd_0000_0000_0000),       // 2^-1074 + 2^-1186 -> f64 min-denormal
            (0xffff_ffff_ffff_ffff, 0x3bcd_0000_0000_0000),       // just below 2^-1073 -> f64 min-denormal
        ];
        let module32 = qnarrow_module(32);
        let (jit32, ids32) = jit_module(&module32);
        let module64 = qnarrow_module(64);
        let (jit64, ids64) = jit_module(&module64);
        for (lo, hi) in quads {
            let inp = qbytes(*lo, *hi);
            let mut out32 = [0u8; 4];
            run_q1(&jit32, &ids32, 0, out32.as_mut_ptr(), inp.as_ptr());
            assert_eq!(u32::from_le_bytes(out32), crate::quad::ref_quad_to_f32(*lo, *hi), "narrow f32 of {lo:#x}/{hi:#x}");
            let mut out64 = [0u8; 8];
            run_q1(&jit64, &ids64, 0, out64.as_mut_ptr(), inp.as_ptr());
            assert_eq!(u64::from_le_bytes(out64), crate::quad::ref_quad_to_f64(*lo, *hi), "narrow f64 of {lo:#x}/{hi:#x}");
        }
    }

    /// Lower one MIR module to x86-64 ELF object bytes for structural
    /// inspection (linkage, sections, alignment) the JIT cannot show.
    fn emit_object(mir_module: &mir::Module) -> Vec<u8> {
        use cranelift_codegen::settings::Configurable;
        let mut flag_builder = cranelift_codegen::settings::builder();
        flag_builder.set("opt_level", "none").expect("flag");
        flag_builder.set("is_pic", "true").expect("flag");
        flag_builder.set("tls_model", "elf_gd").expect("tls model");
        let flags = cranelift_codegen::settings::Flags::new(flag_builder);
        let isa = cranelift_codegen::isa::lookup("x86_64-unknown-linux-gnu".parse().expect("triple"))
            .expect("isa")
            .finish(flags)
            .expect("isa flags");
        let obj_builder = cranelift_object::ObjectBuilder::new(isa, "linktest", cranelift_module::default_libcall_names()).expect("object builder");
        let mut obj = cranelift_object::ObjectModule::new(obj_builder);
        let target = crate::abi::Target::from_triple("x86_64-unknown-linux-gnu").expect("target");
        lower_module(mir_module, &mut obj, target);
        obj.finish().emit().expect("object emit")
    }

    /// Kind, weakness, and address of one symbol in emitted bytes.
    fn symbol_info(object: &[u8], name: &str) -> (object::SymbolKind, bool, u64) {
        use object::read::{Object, ObjectSymbol};
        let file = object::File::parse(object).expect("parse object");
        for sym in file.symbols() {
            if sym.name().ok() == Some(name) {
                return (sym.kind(), sym.is_weak(), sym.address());
            }
        }
        panic!("no symbol {name}");
    }

    #[test]
    fn weak_definitions_emit_stb_weak() {
        let sig = mir::Signature::new(vec![], mir::Ty::int(32));
        let mut module = mir::Module::new();
        let (mut weak, wentry) = mir::Function::new("weak_fn", sig.clone());
        weak.weak = true;
        let five = iconst(&mut weak, 32, 5);
        ret(&mut weak, wentry, Some(five));
        module.push_func(weak);
        let (mut strong, sentry) = mir::Function::new("strong_fn", sig);
        let six = iconst(&mut strong, 32, 6);
        ret(&mut strong, sentry, Some(six));
        module.push_func(strong);

        let object = emit_object(&module);
        let (_, is_weak, _) = symbol_info(&object, "weak_fn");
        assert!(is_weak, "weak function emits STB_WEAK");
        let (_, is_weak, _) = symbol_info(&object, "strong_fn");
        assert!(!is_weak, "plain definition stays strong");
    }

    #[test]
    fn weak_imports_emit_stb_weak() {
        // `extern ... __attribute__((weak))` with no definition: the
        // undefined symbol is `STB_WEAK`, so the linker resolves it to
        // zero (tcctest.c's `weak_fpa`/`weak_fpc` declarations). Only
        // referenced imports are declared, so call both.
        let mut module = mir::Module::new();
        let weak = module.push_extern("weak_ext", mir::Signature::new(vec![], mir::Ty::int(32)));
        module.externs[weak].weak = true;
        let plain = module.push_extern("plain_ext", mir::Signature::new(vec![], mir::Ty::int(32)));
        let sig = mir::Signature::new(vec![], mir::Ty::int(32));
        let (mut caller, entry) = mir::Function::new("caller", sig);
        let w = emit(
            &mut caller,
            entry,
            mir::InstKind::CallExtern {
                target: weak,
                args: Default::default(),
            },
            mir::Ty::int(32),
        );
        let p = emit(
            &mut caller,
            entry,
            mir::InstKind::CallExtern {
                target: plain,
                args: Default::default(),
            },
            mir::Ty::int(32),
        );
        let sum = emit(&mut caller, entry, mir::InstKind::Add(w, p), mir::Ty::int(32));
        ret(&mut caller, entry, Some(sum));
        module.push_func(caller);
        let object = emit_object(&module);
        let (_, is_weak, _) = symbol_info(&object, "weak_ext");
        assert!(is_weak, "weak import emits STB_WEAK");
        let (_, is_weak, _) = symbol_info(&object, "plain_ext");
        assert!(!is_weak, "plain import stays strong");
    }

    #[test]
    fn tls_globals_emit_tls_symbols() {
        let mut module = mir::Module::new();
        let g = module.push_global("tls_counter", mir::Ty::int(32), Some(mir::Const::Int { bits: 32, value: 7 }));
        module.globals[g].tls = true;
        module.globals[g].align = Some(4);
        let sig = mir::Signature::new(vec![], mir::Ty::int(32));
        let (mut f, entry) = mir::Function::new("read_tls", sig);
        let addr = emit(&mut f, entry, mir::InstKind::GlobalAddr { global: g }, mir::Ty::ptr(mir::Ty::int(32)));
        let v = emit(
            &mut f,
            entry,
            mir::InstKind::Load {
                addr,
                flags: mir::MemFlags::plain(),
            },
            mir::Ty::int(32),
        );
        ret(&mut f, entry, Some(v));
        module.push_func(f);

        let object = emit_object(&module);
        let (kind, _, _) = symbol_info(&object, "tls_counter");
        assert_eq!(kind, object::SymbolKind::Tls, "STT_TLS symbol");
    }

    #[test]
    fn alignment_reaches_the_object() {
        let mut module = mir::Module::new();
        let g = module.push_global(
            "dbl",
            mir::Ty::Float(mir::FloatKind::F64),
            Some(mir::Const::Float {
                kind: mir::FloatKind::F64,
                bits: 1.5f64.to_bits() as u128,
            }),
        );
        module.globals[g].align = Some(8);
        let (mut f, entry) = mir::Function::new("f", mir::Signature::new(vec![], mir::Ty::Void));
        ret(&mut f, entry, None);
        module.push_func(f);

        let object = emit_object(&module);
        let (_, _, addr) = symbol_info(&object, "dbl");
        assert_eq!(addr % 8, 0, "double lands 8-aligned");
    }

    #[test]
    fn custom_section_defines_hook_array() {
        use object::read::{Object, ObjectSection};

        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![], mir::Ty::Void);
        let (mut hook, entry) = mir::Function::new("hook", sig);
        ret(&mut hook, entry, None);
        let hook_id = module.push_func(hook);

        let g = module.push_global(
            "__glcc_init_array",
            mir::Ty::Array {
                elem: Box::new(mir::Ty::ptr(mir::Ty::Void)),
                len: 1,
            },
            Some(mir::Const::Bytes(vec![0u8; 8])),
        );
        let global = &mut module.globals[g];
        global.local = true;
        global.align = Some(8);
        global.section = Some(".init_array".to_string());
        global.relocs = vec![(0, mir::DataReloc::Func { target: hook_id })];

        let object = emit_object(&module);
        let file = object::File::parse(&*object).expect("parse object");
        let section = file.section_by_name(".init_array").expect(".init_array section");
        assert_eq!(section.align(), 8);
        assert_eq!(section.data().expect("section data").len(), 8);
        assert!(section.relocations().next().is_some(), "hook address is a link-time reloc");
    }

    /// Volatile MIR accesses must be opaque to Cranelift's alias
    /// analysis, which happily forwards a plain load from a dominating
    /// store: they lower to atomic ops it treats as fences. 16-byte
    /// scalars have no portable atomic form and keep a plain access
    /// fenced on both sides instead.
    #[test]
    fn volatile_accesses_lower_to_atomics() {
        use cranelift_codegen::ir::types;
        use cranelift_codegen::ir::{Function, InstBuilder, Opcode, Type};
        use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

        fn access_opcodes(ty: Type) -> Vec<Opcode> {
            let mut func = Function::new();
            {
                let mut ctx = FunctionBuilderContext::new();
                let mut builder = FunctionBuilder::new(&mut func, &mut ctx);
                let entry = builder.create_block();
                builder.switch_to_block(entry);
                let addr = builder.append_block_param(entry, types::I64);
                let value = super::volatile_load(&mut builder, ty, addr);
                super::volatile_store(&mut builder, ty, value, addr);
                builder.ins().return_(&[]);
                builder.seal_all_blocks();
            }
            func.layout
                .blocks()
                .flat_map(|block| func.layout.block_insts(block))
                .map(|inst| func.dfg.insts[inst].opcode())
                .collect()
        }

        for ty in [types::I8, types::I16, types::I32, types::I64, types::F32, types::F64] {
            let opcodes = access_opcodes(ty);
            assert!(opcodes.contains(&Opcode::AtomicLoad), "{ty} load: {opcodes:?}");
            assert!(opcodes.contains(&Opcode::AtomicStore), "{ty} store: {opcodes:?}");
        }
        let opcodes = access_opcodes(types::F128);
        assert!(!opcodes.contains(&Opcode::AtomicLoad), "{opcodes:?}");
        assert!(!opcodes.contains(&Opcode::AtomicStore), "{opcodes:?}");
        assert!(opcodes.contains(&Opcode::Fence), "{opcodes:?}");
        assert!(opcodes.contains(&Opcode::Load) && opcodes.contains(&Opcode::Store), "{opcodes:?}");
    }

    /// A `Module` that compiles nothing: it keeps the finalized Clif of
    /// each defined function so tests can inspect what lowering
    /// emitted. Only enough of the trait for modules of plain functions.
    struct RecordingModule {
        isa: std::sync::Arc<dyn cranelift_codegen::isa::TargetIsa>,
        declarations: cranelift_module::ModuleDeclarations,
        funcs: Vec<cranelift_codegen::ir::Function>,
    }

    impl RecordingModule {
        fn new() -> Self {
            let flags = cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder());
            let isa = cranelift_native::builder().expect("native isa").finish(flags).expect("native isa");
            Self {
                isa,
                declarations: cranelift_module::ModuleDeclarations::default(),
                funcs: Vec::new(),
            }
        }
    }

    impl cranelift_module::Module for RecordingModule {
        fn isa(&self) -> &dyn cranelift_codegen::isa::TargetIsa {
            &*self.isa
        }

        fn declarations(&self) -> &cranelift_module::ModuleDeclarations {
            &self.declarations
        }

        fn declare_function(
            &mut self,
            name: &str,
            linkage: cranelift_module::Linkage,
            signature: &cranelift_codegen::ir::Signature,
        ) -> cranelift_module::ModuleResult<cranelift_module::FuncId> {
            self.declarations.declare_function(name, linkage, signature).map(|(id, _)| id)
        }

        fn declare_anonymous_function(&mut self, signature: &cranelift_codegen::ir::Signature) -> cranelift_module::ModuleResult<cranelift_module::FuncId> {
            self.declarations.declare_anonymous_function(signature)
        }

        fn declare_data(&mut self, name: &str, linkage: cranelift_module::Linkage, writable: bool, tls: bool) -> cranelift_module::ModuleResult<cranelift_module::DataId> {
            self.declarations.declare_data(name, linkage, writable, tls).map(|(id, _)| id)
        }

        fn declare_anonymous_data(&mut self, writable: bool, tls: bool) -> cranelift_module::ModuleResult<cranelift_module::DataId> {
            self.declarations.declare_anonymous_data(writable, tls)
        }

        fn define_function_with_control_plane(
            &mut self,
            _func: cranelift_module::FuncId,
            ctx: &mut cranelift_codegen::Context,
            _ctrl_plane: &mut cranelift_codegen::control::ControlPlane,
        ) -> cranelift_module::ModuleResult<()> {
            self.funcs.push(ctx.func.clone());
            Ok(())
        }

        fn define_function_bytes(
            &mut self,
            _func_id: cranelift_module::FuncId,
            _alignment: u64,
            _bytes: &[u8],
            _relocs: &[cranelift_module::ModuleReloc],
        ) -> cranelift_module::ModuleResult<()> {
            unreachable!("a recording module has no object bytes")
        }

        fn define_data(&mut self, _data_id: cranelift_module::DataId, _data: &cranelift_module::DataDescription) -> cranelift_module::ModuleResult<()> {
            unreachable!("a recording module defines no data")
        }
    }

    /// The wiring, not just the helpers: `emit_inst` must route a
    /// volatile MIR access through the atomic path (which Cranelift's
    /// alias analysis never forwards a store into) and leave plain
    /// accesses plain.
    #[test]
    fn volatile_mir_accesses_lower_to_atomic_clif() {
        use cranelift_codegen::ir::Opcode;

        fn lowered_opcodes(volatile: bool) -> Vec<Opcode> {
            let mut module = mir::Module::new();
            let sig = mir::Signature::new(vec![mir::Ty::ptr(mir::Ty::int(32))], mir::Ty::int(32));
            let (mut f, entry) = mir::Function::new("f", sig);
            let p = param(&mut f, entry, mir::Ty::ptr(mir::Ty::int(32)), 0);
            let flags = if volatile { mir::MemFlags::vol() } else { mir::MemFlags::plain() };
            let load = emit(&mut f, entry, mir::InstKind::Load { addr: p, flags }, mir::Ty::int(32));
            let seven = iconst(&mut f, 32, 7);
            emit_unit(&mut f, entry, mir::InstKind::Store { addr: p, val: seven, flags });
            ret(&mut f, entry, Some(load));
            module.push_func(f);

            let mut sink = RecordingModule::new();
            lower_module(&module, &mut sink, crate::abi::Target::native());
            let func = &sink.funcs[0];
            func.layout
                .blocks()
                .flat_map(|block| func.layout.block_insts(block))
                .map(|inst| func.dfg.insts[inst].opcode())
                .collect()
        }

        let volatile_ops = lowered_opcodes(true);
        assert!(volatile_ops.contains(&Opcode::AtomicLoad), "{volatile_ops:?}");
        assert!(volatile_ops.contains(&Opcode::AtomicStore), "{volatile_ops:?}");
        let plain_ops = lowered_opcodes(false);
        assert!(plain_ops.contains(&Opcode::Load), "{plain_ops:?}");
        assert!(plain_ops.contains(&Opcode::Store), "{plain_ops:?}");
        assert!(!plain_ops.contains(&Opcode::AtomicLoad), "{plain_ops:?}");
        assert!(!plain_ops.contains(&Opcode::AtomicStore), "{plain_ops:?}");
    }
}

/// Switch the builder only when the target differs: re-switching the
/// current block trips the builder's filled-check when setup code
/// already emitted there.
fn switch_to(body: &mut Body, builder: &mut FunctionBuilder, target: ClifBlock) {
    if body.current != Some(target) {
        builder.switch_to_block(target);
        body.current = Some(target);
    }
}

/// Which function a call targets: a known module function, a foreign
/// import, or a runtime code address for an indirect call.
#[derive(Clone, Copy)]
enum CallTarget {
    Internal(mir::Func),
    External(mir::Extern),
    Indirect { addr: ClifValue, variadic: bool },
}

/// Emit a call: map MIR arguments onto the callee's slots (named
/// prefix) plus caller-built tail slots (variadic actuals), then route
/// direct, indirect, or through the x86-64 `al` trampoline.
#[allow(clippy::too_many_arguments)]
fn emit_call<M: Module>(
    body: &mut Body,
    builder: &mut FunctionBuilder,
    clif: &mut M,
    func_ids: &HashMap<mir::Func, FuncId>,
    extern_ids: &HashMap<mir::Extern, FuncId>,
    _data_ids: &HashMap<mir::Global, cranelift_module::DataId>,
    rt: &Rt,
    layouts: &HashMap<mir::Func, SigLayout>,
    _extern_layouts: &HashMap<mir::Extern, SigLayout>,
    target: CallTarget,
    callee_layout: &SigLayout,
    args: &[mir::Value],
    result_ty: &Option<mir::Ty>,
    result_val: Option<mir::Value>,
) -> Option<ClifValue> {
    let (variadic, foreign) = match target {
        CallTarget::Internal(f) => (layouts[&f].variadic, false),
        CallTarget::External(_) => (callee_layout.variadic, true),
        // Indirect calls always go through the per-call signature;
        // variadic ones take the platform's indirect path below.
        CallTarget::Indirect { variadic, .. } => (variadic, true),
    };
    let _ = foreign;
    // A direct foreign call returning x86-64 `long double` leaves it
    // in st(0), not in an F128 register (SysV): declare no Clif
    // return and fetch it through the getret helper below.
    //
    // Limits, all documented rather than silently handled:
    // - Indirect calls assume the internal (xmm0-carrier) convention:
    //   a function pointer to a *foreign* long-double function reads
    //   xmm0 garbage. Address-taken internal functions (the common
    //   case, including callbacks returning other types) are exact.
    // - The reverse direction is unsupported: our address taken on a
    //   long-double-returning function and called from foreign code
    //   writes xmm0 while the caller reads st(0).
    // - A signature mismatch (MIR says F80, the symbol does not return
    //   long double) pops a stale st(0): an ABI mismatch like any
    //   other, loud only in its symptoms.
    let foreign_f80_ret = matches!(target, CallTarget::External(_)) && matches!(result_ty, Some(mir::Ty::Float(mir::FloatKind::F80)));
    let named = callee_layout.params.len();
    // Hidden out-pointer for aggregate returns comes first everywhere.
    let mut clif_args: Vec<ClifValue> = Vec::new();
    let mut sig_tys: Vec<(Type, ArgumentPurpose)> = Vec::new();
    let sret_temp = match &callee_layout.ret {
        abi::RetLoc::Sret => {
            let size = result_ty.as_ref().map(|t| t.size_bytes()).expect("sret needs a result type");
            let slot = stack_slot(
                builder,
                &mir::Ty::Array {
                    elem: Box::new(mir::Ty::int(8)),
                    len: size,
                },
            );
            let addr = builder.ins().stack_addr(types::I64, slot, 0);
            clif_args.push(addr);
            sig_tys.push((types::I64, ArgumentPurpose::StructReturn));
            Some(addr)
        }
        _ => None,
    };
    // Named prefix: fill the callee's slots positionally. Owners
    // spanning slots (128-bit scalar pairs) split once and reuse.
    // Variadic callees pad their signatures to the full register file
    // so the prelude spills register-passed varargs; callers skip those
    // pads and pass the tail actuals directly, or the pads land zeros
    // in exactly the registers the callee reads.
    let mut splits: HashMap<usize, Vec<ClifValue>> = HashMap::new();
    for slot in &callee_layout.slots {
        match slot.kind {
            abi::SlotKind::Sret => continue,
            abi::SlotKind::Pad if variadic => continue,
            abi::SlotKind::Pad => {
                // Pads need type-appropriate zeros (`iconst` on a
                // float type fails verification).
                let zero = if slot.ty.is_float() {
                    match slot.ty {
                        types::F32 => builder.ins().f32const(0.0),
                        _ => builder.ins().f64const(0.0),
                    }
                } else {
                    builder.ins().iconst(slot.ty, 0)
                };
                clif_args.push(zero);
                sig_tys.push((slot.ty, slot.purpose));
            }
            abi::SlotKind::Value { owner, offset } => {
                let arg = args[owner];
                let arg_ty = body.func.values[arg].ty.clone();
                let multi = !arg_ty.is_aggregate()
                    && callee_layout
                        .slots
                        .iter()
                        .filter(|s| matches!(s.kind, abi::SlotKind::Value { owner: o, .. } if o == owner))
                        .count()
                        > 1;
                if multi {
                    if !splits.contains_key(&owner) {
                        let v = operand(body, builder, arg);
                        let from = builder.func.dfg.value_type(v);
                        splits.insert(owner, split_value(builder, v, from, 2));
                    }
                    let idx = (offset / 8) as usize;
                    clif_args.push(splits[&owner][idx]);
                } else {
                    let v = call_piece(body, builder, arg, offset, slot.ty);
                    clif_args.push(v);
                }
                sig_tys.push((slot.ty, slot.purpose));
            }
            abi::SlotKind::StackAddr { owner } | abi::SlotKind::RefAddr { owner } => {
                let arg = args[owner];
                let arg_ty = body.func.values[arg].ty.clone();
                if arg_ty.is_aggregate() {
                    clif_args.push(operand(body, builder, arg));
                } else {
                    // Scalar behind a stack slot (x86-64 `long
                    // double`): spill to a temp first.
                    let v = operand(body, builder, arg);
                    let slot = stack_slot(builder, &arg_ty);
                    let addr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(memflags::trusted(), v, addr, Offset32::new(0));
                    clif_args.push(addr);
                }
                sig_tys.push((slot.ty, slot.purpose));
            }
            abi::SlotKind::Chunk { owner, index } => {
                let base = operand(body, builder, args[owner]);
                let v = builder.ins().load(types::I64, memflags::notrap(), base, Offset32::new((index * 8) as i32));
                clif_args.push(v);
                sig_tys.push((slot.ty, slot.purpose));
            }
        }
    }
    // Variadic tail: each actual classified with continuing counters.
    let mut arg_idx = named;
    if arg_idx < args.len() {
        emit_call_tail(body, builder, callee_layout, args, &mut arg_idx, &mut clif_args, &mut sig_tys);
    }
    let mut sig = Signature::new(body.target.call_conv());
    for (ty, purpose) in &sig_tys {
        sig.params.push(AbiParam {
            value_type: *ty,
            purpose: *purpose,
            extension: ArgumentExtension::None,
        });
    }
    match &callee_layout.ret {
        abi::RetLoc::Regs(pieces) => {
            // A foreign `long double` return arrives in st(0): no Clif
            // return to declare (fetched through getret at the call).
            if !foreign_f80_ret {
                for (_, ty) in pieces {
                    sig.returns.push(AbiParam::new(*ty));
                }
            }
        }
        _ => {}
    }
    // Direct calls when the declared signature matches the call-site
    // signature: non-variadic internal functions and non-variadic
    // externs (whose declarations use the same layout). Variadic calls
    // and foreign `long double` returns build a per-call signature and
    // go indirect.
    let indirect = variadic || foreign_f80_ret || matches!(target, CallTarget::Indirect { .. });
    if !indirect {
        let clif_id = match target {
            CallTarget::Internal(f) => func_ids[&f],
            CallTarget::External(e) => extern_ids[&e],
            CallTarget::Indirect { .. } => panic!("indirect call marked direct"),
        };
        let funcref: FuncRef = clif.declare_func_in_func(clif_id, builder.func);
        let inst = builder.ins().call(funcref, &clif_args);
        return map_call_results(body, builder, inst, &callee_layout.ret, sret_temp, result_val);
    }
    let addr = match target {
        CallTarget::Internal(f) => {
            let clif_id = func_ids[&f];
            let funcref = clif.declare_func_in_func(clif_id, builder.func);
            builder.ins().func_addr(types::I64, funcref)
        }
        CallTarget::External(e) => {
            let clif_id = extern_ids[&e];
            let funcref = clif.declare_func_in_func(clif_id, builder.func);
            builder.ins().func_addr(types::I64, funcref)
        }
        CallTarget::Indirect { addr, .. } => addr,
    };
    // x86-64 SysV foreign variadic calls need `%al` set: route through
    // the runtime trampoline with the callee's own signature. Win64 has
    // no such register and calls its variadics directly.
    if foreign && variadic && body.target.arch == Arch::X86_64 && body.target.os != Os::Windows {
        let rt = rt;
        let va_target = rt.va_target.expect("va_target not declared");
        let jmp_va = rt.jmp_va.expect("jmp_va not declared");
        let gv = clif.declare_data_in_func(va_target, builder.func);
        let tptr = builder.ins().tls_value(types::I64, gv);
        builder.ins().store(MemFlagsData::trusted(), addr, tptr, Offset32::new(0));
        let jump = clif.declare_func_in_func(jmp_va, builder.func);
        let jump_addr = builder.ins().func_addr(types::I64, jump);
        let sigref = builder.import_signature(sig);
        let inst = builder.ins().call_indirect(sigref, jump_addr, &clif_args);
        if foreign_f80_ret {
            return Some(foreign_f80_result(builder, clif, rt));
        }
        return map_call_results(body, builder, inst, &callee_layout.ret, sret_temp, result_val);
    }
    let sigref = builder.import_signature(sig);
    let inst = builder.ins().call_indirect(sigref, addr, &clif_args);
    if foreign_f80_ret {
        return Some(foreign_f80_result(builder, clif, rt));
    }
    map_call_results(body, builder, inst, &callee_layout.ret, sret_temp, result_val)
}

/// Fetch a foreign `long double` result from the live `st(0)` into a
/// fresh slot and return its carrier load. Emitted immediately after
/// the foreign call: nothing in Cranelift output touches x87, and a
/// call itself does not, so the value survives intact; `fstpt` pops
/// exactly the one value the callee pushed.
fn foreign_f80_result<M: Module>(builder: &mut FunctionBuilder, clif: &mut M, rt: &Rt) -> ClifValue {
    let f80 = rt.f80.as_ref().expect("foreign long double needs x86-64 helpers");
    let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
    let out = builder.ins().stack_addr(types::I64, slot, 0);
    let callee = clif.declare_func_in_func(f80.getret, builder.func);
    builder.ins().call(callee, &[out]);
    f80_result(builder, out)
}

/// Map call results back: scalars directly, register aggregates
/// reassembled into a fresh slot, sret through its temp slot.
fn map_call_results(
    body: &mut Body,
    builder: &mut FunctionBuilder,
    inst: cranelift_codegen::ir::Inst,
    ret: &abi::RetLoc,
    sret_temp: Option<ClifValue>,
    result_val: Option<mir::Value>,
) -> Option<ClifValue> {
    let results = builder.inst_results(inst).to_vec();
    match ret {
        abi::RetLoc::Void => {
            // A zero-sized aggregate result still exists as a MIR value
            // (its uses are 0-byte copies and address math); hand out a
            // fresh dummy address. Void calls carry no result.
            let size = result_val.map(|v| body.func.values[v].ty.size_bytes())?;
            if size != 0 {
                return None;
            }
            let slot = stack_slot(builder, &mir::Ty::int(8));
            Some(builder.ins().stack_addr(types::I64, slot, 0))
        }
        abi::RetLoc::Sret => sret_temp,
        abi::RetLoc::Regs(pieces) => {
            // Aggregates are addresses everywhere in MIR: a register
            // result reassembles into a fresh slot, even when it fits
            // one piece (a bare small-struct call result stored to a
            // local segfaulted before this). Only true scalars return
            // the value itself.
            let scalar = result_val.map(|v| !body.func.values[v].ty.is_aggregate()).unwrap_or(false);
            if pieces.len() == 1 && scalar {
                let v = results.into_iter().next()?;
                // The callee widened the result to its slot width
                // (`Return`); hand the body back a declared-width value,
                // like entry params get on arrival.
                let mir_val = result_val.expect("scalar result carries a MIR value");
                let mir_ty = body.func.values[mir_val].ty.clone();
                return Some(narrow_to_declared(builder, v, &mir_ty));
            }
            if scalar {
                // Multi-piece scalar (128-bit pair) rejoins to one value.
                assert_eq!(results.len(), 2);
                let wide = builder.ins().iconcat(results[0], results[1]);
                let dst = result_val
                    .map(|v| clif_ty(&body.func.values[v].ty))
                    .unwrap_or(None)
                    .expect("scalar result needs a Clif type");
                if dst == types::I128 {
                    return Some(wide);
                }
                return Some(builder.ins().bitcast(dst, MemFlagsData::new(), wide));
            }
            let size = pieces.iter().map(|(off, ty)| off + ty.bytes() as u64).max().unwrap_or(0);
            let slot = stack_slot(
                builder,
                &mir::Ty::Array {
                    elem: Box::new(mir::Ty::int(8)),
                    len: size,
                },
            );
            let addr = builder.ins().stack_addr(types::I64, slot, 0);
            for ((off, _ty), v) in pieces.iter().zip(results) {
                builder.ins().stack_store(types::I64, v, slot, *off as i32);
            }
            let _ = result_val;
            Some(addr)
        }
    }
}

/// One named argument piece for a call: scalars pass directly (the
/// checker made actual and formal widths agree); aggregate pieces load
/// from the argument slot.
fn call_piece(body: &mut Body, builder: &mut FunctionBuilder, arg: mir::Value, offset: u64, slot_ty: Type) -> ClifValue {
    let mir_ty = body.func.values[arg].ty.clone();
    if mir_ty.is_aggregate() {
        let base = operand(body, builder, arg);
        return builder.ins().load(slot_ty, memflags::notrap(), base, Offset32::new(offset as i32));
    }
    let v = operand(body, builder, arg);
    if builder.func.dfg.value_type(v) == slot_ty {
        return v;
    }
    // Width drift between actual and formal (unprototyped edges):
    // resize integers, bitcast floats of equal size.
    let from = builder.func.dfg.value_type(v);
    if !from.is_float() && !slot_ty.is_float() {
        return resize_int(builder, v, from, slot_ty);
    }
    if from.bytes() == slot_ty.bytes() {
        return builder.ins().bitcast(slot_ty, MemFlagsData::new(), v);
    }
    panic!("call argument type mismatch");
}

/// Variadic tail emission: each actual classified with counters
/// continuing past the named prefix. Scalars cross directly
/// (RISC-V unnamed floats as integer bits); aggregates follow fixed
/// rules, with one pad-and-chunk group per stack-bound aggregate.
#[allow(clippy::too_many_arguments)]
fn emit_call_tail(
    body: &mut Body,
    builder: &mut FunctionBuilder,
    callee_layout: &SigLayout,
    args: &[mir::Value],
    arg_idx: &mut usize,
    clif_args: &mut Vec<ClifValue>,
    sig_tys: &mut Vec<(Type, ArgumentPurpose)>,
) {
    let target = body.target;
    // Reseed counters from the named prefix both sides share.
    let mut counters = abi::Counters {
        gp: callee_layout.named_gp,
        fp: callee_layout.named_fp,
    };
    let mut padded = false;
    while *arg_idx < args.len() {
        let arg = args[*arg_idx];
        *arg_idx += 1;
        let mir_ty = body.func.values[arg].ty.clone();
        // C promotes sub-int and float varargs; the checker inserts
        // those conversions for C sources (documented precondition for
        // hand-built MIR: pre-promoted actuals).
        if unnamed_fp_in_int_regs(target, &mir_ty) {
            // Unnamed FP travels in integer registers on RISC-V: one
            // for float/double bits, two for quad. Win64 mirrors every
            // unnamed argument into its position's GP register, so the
            // bits ride there too (the XMM copy is never read back).
            let v = operand(body, builder, arg);
            if mir_ty.size_bytes() > 8 {
                // 16-byte values start at an even register (conformant
                // writers pad odd positions; our reader skips the same
                // way, and so does musl): pad the odd slot so positions
                // line up instead of shifting everything after by one.
                if counters.gp % 2 == 1 {
                    counters.gp += 1;
                    let zero = builder.ins().iconst(types::I64, 0);
                    clif_args.push(zero);
                    sig_tys.push((types::I64, ArgumentPurpose::Normal));
                }
                let wide = builder.ins().bitcast(types::I128, MemFlagsData::new(), v);
                let (lo, hi) = builder.ins().isplit(wide);
                counters.gp += 2;
                clif_args.push(lo);
                clif_args.push(hi);
                sig_tys.push((types::I64, ArgumentPurpose::Normal));
                sig_tys.push((types::I64, ArgumentPurpose::Normal));
                continue;
            }
            let bits = float_bits_to_int(builder, v, &mir_ty);
            counters.gp += 1;
            clif_args.push(bits);
            sig_tys.push((types::I64, ArgumentPurpose::Normal));
            continue;
        }
        let (even_pad, loc) = tail_classify(target, &mir_ty, &mut counters);
        if even_pad {
            // 16-aligned composite at an odd GP slot: consume the
            // skipped register with a dummy so the actual layout
            // matches the counters (the callee rounds its cursor the
            // same way).
            let zero = builder.ins().iconst(types::I64, 0);
            clif_args.push(zero);
            sig_tys.push((types::I64, ArgumentPurpose::Normal));
        }
        match loc {
            TailLoc::Scalar(ty) => {
                let mut v = operand(body, builder, arg);
                // Sub-int actuals widen to int for the register file.
                let from = builder.func.dfg.value_type(v);
                if !from.is_float() && from.bytes() < 4 && ty.bytes() == 4 {
                    v = builder.ins().uextend(ty, v);
                }
                clif_args.push(v);
                sig_tys.push((ty, ArgumentPurpose::Normal));
            }
            TailLoc::Pieces(pieces) => {
                push_pieces(target, body, builder, arg, &pieces, clif_args, sig_tys);
            }
            TailLoc::StackPieces(pieces) => {
                if !padded {
                    pad_to_full(target, builder, &mut counters.gp, &mut counters.fp, clif_args, sig_tys);
                    padded = true;
                }
                push_pieces(target, body, builder, arg, &pieces, clif_args, sig_tys);
            }
            TailLoc::Stack { bytes } => {
                if target.arch == Arch::X86_64 {
                    // Stack-bound scalars (x86-64 `long double`) spill
                    // to a temp: `StructArgument` takes the address,
                    // like the named path. Aggregates already are
                    // addresses.
                    let base = if mir_ty.is_aggregate() {
                        operand(body, builder, arg)
                    } else {
                        let v = operand(body, builder, arg);
                        let slot = stack_slot(builder, &mir_ty);
                        let addr = builder.ins().stack_addr(types::I64, slot, 0);
                        builder.ins().store(memflags::trusted(), v, addr, Offset32::new(0));
                        addr
                    };
                    clif_args.push(base);
                    sig_tys.push((types::I64, ArgumentPurpose::StructArgument(bytes as u32)));
                } else {
                    if !padded {
                        pad_to_full(target, builder, &mut counters.gp, &mut counters.fp, clif_args, sig_tys);
                        padded = true;
                    }
                    let base = operand(body, builder, arg);
                    for k in 0..bytes.div_ceil(8) {
                        let v = builder.ins().load(types::I64, memflags::notrap(), base, Offset32::new((k * 8) as i32));
                        clif_args.push(v);
                        sig_tys.push((types::I64, ArgumentPurpose::Normal));
                    }
                }
            }
            TailLoc::Pointer => {
                let base = operand(body, builder, arg);
                clif_args.push(base);
                sig_tys.push((types::I64, ArgumentPurpose::Normal));
            }
        }
    }
}

/// Unnamed floating-point values that cross as integer bits: RISC-V's
/// variadic convention, and Win64's GP mirror of every unnamed
/// argument position.
fn unnamed_fp_in_int_regs(target: Target, mir_ty: &mir::Ty) -> bool {
    matches!(mir_ty, mir::Ty::Float(_)) && (target.arch == Arch::Riscv64 || (target.arch == Arch::X86_64 && target.os == Os::Windows))
}

/// RISC-V unnamed FP values cross as integer bits.
fn riscv_fp_bits(builder: &mut FunctionBuilder, v: ClifValue, ty: Type) -> ClifValue {
    if ty == types::F64 {
        builder.ins().bitcast(types::I64, MemFlagsData::new(), v)
    } else {
        let narrow = builder.ins().bitcast(types::I32, MemFlagsData::new(), v);
        builder.ins().uextend(types::I64, narrow)
    }
}

/// Push one register slot per aggregate piece: load the piece from the
/// aggregate's memory image and, on RISC-V, send FP bits through the
/// integer registers.
fn push_pieces(
    target: Target,
    body: &mut Body,
    builder: &mut FunctionBuilder,
    arg: mir::Value,
    pieces: &[(u64, Type)],
    clif_args: &mut Vec<ClifValue>,
    sig_tys: &mut Vec<(Type, ArgumentPurpose)>,
) {
    for (off, ty) in pieces {
        let base = operand(body, builder, arg);
        let v = builder.ins().load(*ty, memflags::notrap(), base, Offset32::new(*off as i32));
        let (v, ty) = if target.arch == Arch::Riscv64 && ty.is_float() {
            (riscv_fp_bits(builder, v, *ty), types::I64)
        } else {
            (v, *ty)
        };
        clif_args.push(v);
        sig_tys.push((ty, ArgumentPurpose::Normal));
    }
}

/// Reinterpret float bits as an integer of the same width.
fn float_bits_to_int(builder: &mut FunctionBuilder, v: ClifValue, mir_ty: &mir::Ty) -> ClifValue {
    match mir_ty {
        mir::Ty::Float(mir::FloatKind::F32) => {
            let ext = builder.ins().fpromote(types::F64, v);
            builder.ins().bitcast(types::I64, MemFlagsData::new(), ext)
        }
        _ => builder.ins().bitcast(types::I64, MemFlagsData::new(), v),
    }
}

enum TailLoc {
    Scalar(Type),
    Pieces(Vec<(u64, Type)>),
    Stack {
        bytes: u64,
    },
    /// RISC-V stack-bound aggregate: the callee's walk reads one slot
    /// per piece, not the packed memory image, so the pieces go after
    /// the register padding instead of raw chunks.
    StackPieces(Vec<(u64, Type)>),
    /// By-reference aggregate: pass its address.
    Pointer,
}

/// Classify one variadic-tail actual with continuing counters. The
/// flag reports a 16-alignment even-skip, which the caller must
/// materialize as a dummy register argument.
fn tail_classify(target: Target, mir_ty: &mir::Ty, counters: &mut abi::Counters) -> (bool, TailLoc) {
    let (even_pad, loc) = abi::classify_next(target, mir_ty, counters);
    let tail = match loc {
        abi::ParamLoc::Regs(pieces) => {
            if pieces.len() == 1 && !mir_ty.is_aggregate() {
                TailLoc::Scalar(pieces[0].1)
            } else {
                TailLoc::Pieces(pieces)
            }
        }
        abi::ParamLoc::Stack { bytes } => match (target.arch, abi::aggregate_pieces(target, mir_ty)) {
            (Arch::Riscv64, Some(pieces)) if mir_ty.is_aggregate() => TailLoc::StackPieces(pieces),
            _ => TailLoc::Stack { bytes },
        },
        abi::ParamLoc::Pointer => TailLoc::Pointer,
    };
    (even_pad, tail)
}

/// Pad both register files to full with zero dummies.
fn pad_to_full(target: Target, builder: &mut FunctionBuilder, gp: &mut usize, fp: &mut usize, clif_args: &mut Vec<ClifValue>, sig_tys: &mut Vec<(Type, ArgumentPurpose)>) {
    while *gp < target.gp_regs() {
        *gp += 1;
        let zero = builder.ins().iconst(types::I64, 0);
        clif_args.push(zero);
        sig_tys.push((types::I64, ArgumentPurpose::Normal));
    }
    while *fp < target.fp_regs() {
        *fp += 1;
        let zero = builder.ins().f64const(0.0);
        clif_args.push(zero);
        sig_tys.push((types::F64, ArgumentPurpose::Normal));
    }
}

/// Rejoin split scalar pieces (128-bit pairs) into one value: concat
/// the integer lanes, then reinterpret to the value type.
fn join_values(builder: &mut FunctionBuilder, parts: &[(u64, ClifValue)], mir_ty: &mir::Ty) -> ClifValue {
    use cranelift_codegen::ir::InstBuilder;
    assert_eq!(parts.len(), 2, "only pairs split scalars");
    assert_eq!((parts[0].0, parts[1].0), (0, 8));
    let wide = builder.ins().iconcat(parts[0].1, parts[1].1);
    let dst = clif_ty(mir_ty).expect("scalar join needs a Clif type");
    if dst == types::I128 {
        return wide;
    }
    builder.ins().bitcast(dst, MemFlagsData::new(), wide)
}

/// Split one scalar value into integer lanes for multi-slot emission.
fn split_value(builder: &mut FunctionBuilder, v: ClifValue, from_ty: Type, count: usize) -> Vec<ClifValue> {
    use cranelift_codegen::ir::InstBuilder;
    assert_eq!(count, 2, "only pairs split scalars");
    let wide = if from_ty == types::I128 {
        v
    } else {
        builder.ins().bitcast(types::I128, MemFlagsData::new(), v)
    };
    let (lo, hi) = builder.ins().isplit(wide);
    vec![lo, hi]
}
