//! Legalize arbitrary-width integers into limb operations.
//!
//! [`Ty::Int`] is native only up to [`MAX_SCALAR_INT_BITS`] (the widest
//! scalar Cranelift lowers). A `_BitInt(4096)` or any other wider
//! integer becomes a [`Ty::WideInt`]: an aggregate whose representation
//! is a little-endian sequence of [`LIMB_BITS`]-bit limbs. This module
//! rewrites every integer operation over such a type into scalar
//! operations over its limbs, so the rest of MIR (and the backend)
//! never has to know how wide the original integer was.
//!
//! Values of a wide type always carry canonical limbs: unused bits in
//! the last limb are zero. Signedness lives on the operation, exactly
//! as it does for [`Ty::Int`].
//!
//! Loads, stores, allocas, GEPs, calls, and block arguments need no
//! rewriting: a [`Ty::WideInt`] is an aggregate, so existing aggregate
//! plumbing copies it through memory with the same size and layout the
//! original integer had.

use std::collections::HashMap;

use crate::ir::{
    Block, Const, Def, Func, Function, Inst, InstData, InstKind, IcmpPred, Module, Terminator, Value, ValueData,
};
use crate::pass::ModulePass;
use crate::ty::{Signature, Ty, LIMB_BITS, MAX_SCALAR_INT_BITS};

fn is_wide(bits: u32) -> bool {
    bits > MAX_SCALAR_INT_BITS
}

/// Rewrite every integer field of `ty` wider than the scalar limit into
/// a [`Ty::WideInt`]. Recurses through pointers, arrays, and structs so
/// nested wide integers keep their byte layout.
pub fn legalize_ty(ty: &Ty) -> Ty {
    match ty {
        Ty::Int { bits } if is_wide(*bits) => Ty::WideInt { bits: *bits },
        Ty::Ptr { pointee, bits } => Ty::Ptr {
            pointee: Box::new(legalize_ty(pointee)),
            bits: *bits,
        },
        Ty::Array { elem, len } => Ty::Array {
            elem: Box::new(legalize_ty(elem)),
            len: *len,
        },
        Ty::Struct { fields, offsets, size } => Ty::Struct {
            fields: fields.iter().map(legalize_ty).collect(),
            offsets: offsets.clone(),
            size: *size,
        },
        other => other.clone(),
    }
}

fn legalize_sig(sig: &Signature) -> Signature {
    Signature {
        params: sig.params.iter().map(legalize_ty).collect(),
        ret: legalize_ty(&sig.ret),
        variadic: sig.variadic,
    }
}

fn legalize_inst_types(kind: &mut InstKind) {
    match kind {
        InstKind::Alloca { pointee, .. } => *pointee = legalize_ty(pointee),
        InstKind::CallIndirect { sig, .. } => *sig = legalize_sig(sig),
        InstKind::InlineAsm { block } => {
            for operand in &mut block.operands {
                operand.ty = legalize_ty(&operand.ty);
            }
        }
        _ => {}
    }
}

/// Whether the module still contains a scalar integer that needs
/// legalization, anywhere.
fn module_needs_legalizing(module: &Module) -> bool {
    fn ty_needs(ty: &Ty) -> bool {
        match ty {
            Ty::Int { bits } => is_wide(*bits),
            Ty::Ptr { pointee, .. } => ty_needs(pointee),
            Ty::Array { elem, .. } => ty_needs(elem),
            Ty::Struct { fields, .. } => fields.iter().any(ty_needs),
            _ => false,
        }
    }
    module.globals.iter().any(|g| ty_needs(&g.ty))
        || module.externs.iter().any(|e| e.sig.params.iter().any(ty_needs) || ty_needs(&e.sig.ret))
        || module.funcs.iter().any(|f| {
            ty_needs(&f.sig.ret)
                || f.sig.params.iter().any(ty_needs)
                || f.values.iter().any(|v| ty_needs(&v.ty))
                || f.insts.iter().any(|inst| match &inst.kind {
                    InstKind::Alloca { pointee, .. } => ty_needs(pointee),
                    InstKind::CallIndirect { sig, .. } => ty_needs(&sig.ret) || sig.params.iter().any(ty_needs),
                    InstKind::InlineAsm { block } => block.operands.iter().any(|o| ty_needs(&o.ty)),
                    _ => false,
                })
        })
}

/// Rewrite all wide integer types in signatures, globals, and values.
/// Returns whether anything changed.
fn legalize_module_types(module: &mut Module) -> bool {
    let mut changed = false;
    for global in module.globals.iter_mut() {
        let ty = legalize_ty(&global.ty);
        changed |= ty != global.ty;
        global.ty = ty;
    }
    for external in module.externs.iter_mut() {
        let sig = legalize_sig(&external.sig);
        changed |= sig != external.sig;
        external.sig = sig;
    }
    for func in module.funcs.iter_mut() {
        let sig = legalize_sig(&func.sig);
        changed |= sig != func.sig;
        func.sig = sig;
        if let Some(ret) = func.coroutine.take() {
            func.coroutine = Some(legalize_ty(&ret));
        }
        for value in func.values.iter_mut() {
            let ty = legalize_ty(&value.ty);
            changed |= ty != value.ty;
            value.ty = ty;
        }
        for inst in func.insts.iter_mut() {
            legalize_inst_types(&mut inst.kind);
        }
        for block in func.blocks.iter_mut() {
            for param in &block.params {
                let ty = legalize_ty(&func.values[*param].ty);
                changed |= ty != func.values[*param].ty;
                func.values[*param].ty = ty;
            }
        }
    }
    changed
}

/// Which unsigned division helper a call site needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DivKind {
    Quotient,
    Remainder,
}

#[derive(Default)]
pub struct LegalizeWideInts {
    helpers: HashMap<(u32, DivKind), Func>,
}

impl LegalizeWideInts {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ModulePass for LegalizeWideInts {
    fn name(&self) -> &'static str {
        "legalize-wide-ints"
    }

    fn run(&mut self, module: &mut Module) -> bool {
        if !module_needs_legalizing(module) {
            return false;
        }
        legalize_module_types(module);
        // Helpers are synthesized before rewriting so call sites can
        // refer to them without borrowing two functions at once.
        let mut needed: Vec<(u32, DivKind)> = Vec::new();
        for func in module.funcs.iter() {
            for inst in func.insts.iter() {
                match &inst.kind {
                    InstKind::Udiv(a, _) | InstKind::Sdiv(a, _) if wide_of(func, *a) => {
                        push_unique(&mut needed, (wide_bits(func, *a), DivKind::Quotient));
                    }
                    InstKind::Urem(a, _) | InstKind::Srem(a, _) if wide_of(func, *a) => {
                        push_unique(&mut needed, (wide_bits(func, *a), DivKind::Remainder));
                    }
                    _ => {}
                }
            }
        }
        for (bits, kind) in needed {
            self.ensure_helper(module, bits, kind);
        }
        for index in 0..module.funcs.len() {
            let func = &mut module.funcs[Func::from_usize(index)];
            legalize_function(func, &self.helpers);
        }
        true
    }
}

fn push_unique(list: &mut Vec<(u32, DivKind)>, item: (u32, DivKind)) {
    if !list.contains(&item) {
        list.push(item);
    }
}

fn wide_of(func: &Function, value: Value) -> bool {
    matches!(func.values[value].ty, Ty::WideInt { .. })
}

fn wide_bits(func: &Function, value: Value) -> u32 {
    match func.values[value].ty {
        Ty::WideInt { bits } => bits,
        ref other => panic!("expected a wide integer operand, found {other:?}"),
    }
}

fn wide_ty(bits: u32) -> Ty {
    Ty::WideInt { bits }
}

fn emit_value(func: &mut Function, block: Block, kind: InstKind, ty: Ty) -> Value {
    let id = Inst::from_usize(func.insts.len());
    let value = func.values.push(ValueData {
        ty,
        def: Def::Inst(id),
    });
    let inst = func.insts.push(InstData {
        block,
        kind,
        result: Some(value),
        loc: None,
    });
    func.blocks[block].insts.push(inst);
    value
}

fn emit_unit(func: &mut Function, block: Block, kind: InstKind) -> Inst {
    let inst = func.insts.push(InstData {
        block,
        kind,
        result: None,
        loc: None,
    });
    func.blocks[block].insts.push(inst);
    inst
}

/// Emit `kind` and make it the definition of `value`, preserving the
/// value id used by earlier code.
fn emit_as(func: &mut Function, block: Block, kind: InstKind, ty: Ty, value: Value) -> Value {
    let id = Inst::from_usize(func.insts.len());
    let inst = func.insts.push(InstData {
        block,
        kind,
        result: Some(value),
        loc: None,
    });
    func.values[value] = ValueData {
        ty,
        def: Def::Inst(id),
    };
    func.blocks[block].insts.push(inst);
    value
}

fn typed_const(func: &mut Function, ty: Ty, c: Const) -> Value {
    func.values.push(ValueData { ty, def: Def::Const(c) })
}

fn bool_const(func: &mut Function, value: bool) -> Value {
    typed_const(
        func,
        Ty::Bool,
        Const::Int {
            bits: 1,
            value: i128::from(value),
        },
    )
}

fn i64_const(func: &mut Function, value: i64) -> Value {
    func.push_const(Const::Int {
        bits: LIMB_BITS,
        value: i128::from(value),
    })
}

fn zext_i64(func: &mut Function, block: Block, value: Value) -> Value {
    match func.values[value].ty {
        Ty::Int { bits } if bits < LIMB_BITS => emit_value(func, block, InstKind::Zext(value), Ty::int(LIMB_BITS)),
        Ty::Bool => emit_value(func, block, InstKind::Zext(value), Ty::int(LIMB_BITS)),
        _ => value,
    }
}

fn to_i64_count(func: &mut Function, block: Block, value: Value) -> Value {
    match func.values[value].ty.clone() {
        Ty::Bool => emit_value(func, block, InstKind::Zext(value), Ty::int(LIMB_BITS)),
        Ty::Int { bits } if bits < LIMB_BITS => {
            emit_value(func, block, InstKind::Zext(value), Ty::int(LIMB_BITS))
        }
        Ty::Int { bits } if bits == LIMB_BITS => value,
        Ty::Int { .. } => emit_value(func, block, InstKind::Trunc(value), Ty::int(LIMB_BITS)),
        Ty::WideInt { bits } => {
            let limbs = limbs_of(func, block, value);
            trunc_wide_to_scalar(func, block, &limbs, bits, LIMB_BITS)
        }
        _ => panic!("shift count is not an integer"),
    }
}

fn limb_ty_of(ty: &Ty, index: u32) -> Ty {
    ty.limb_ty(index)
}

fn trunc_limb(func: &mut Function, block: Block, value: Value, limb_ty: Ty) -> Value {
    if func.values[value].ty == limb_ty {
        return value;
    }
    emit_value(func, block, InstKind::Trunc(value), limb_ty)
}

/// Split a wide value into canonical 64-bit limbs. Constants are split
/// directly; anything else is read out of its aggregate slot by
/// `ExtractValue`.
fn limbs_of(func: &mut Function, block: Block, value: Value) -> Vec<Value> {
    let Ty::WideInt { bits } = func.values[value].ty else {
        panic!("limb split of a non-wide value {:?}", func.values[value].ty);
    };
    let count = bits.div_ceil(LIMB_BITS);
    if let Def::Const(c) = func.values[value].def.clone() {
        match c {
            Const::Int { value: n, .. } => {
                let raw = n as u128;
                return (0..count)
                    .map(|i| {
                        let limb = if i < 2 {
                            (raw >> (i * LIMB_BITS)) as u64
                        } else if n < 0 {
                            u64::MAX
                        } else {
                            0
                        };
                        i64_const(func, limb as i64)
                    })
                    .collect();
            }
            Const::Zero | Const::Undef => return (0..count).map(|_| i64_const(func, 0)).collect(),
            Const::Bytes(_) | Const::Null | Const::Float { .. } => {}
        }
    }
    let wide = wide_ty(bits);
    (0..count)
        .map(|i| {
            let limb = emit_value(
                func,
                block,
                InstKind::ExtractValue { agg: value, index: i },
                limb_ty_of(&wide, i),
            );
            zext_i64(func, block, limb)
        })
        .collect()
}

/// Assemble a wide value from canonical 64-bit limbs. The final insert
/// defines `result` when one is supplied.
fn build_wide(func: &mut Function, block: Block, bits: u32, limbs: &[Value], result: Option<Value>) -> Value {
    let ty = wide_ty(bits);
    assert_eq!(limbs.len() as u32, bits.div_ceil(LIMB_BITS), "limb count mismatch");
    let mut agg = typed_const(func, ty.clone(), Const::Undef);
    let last = limbs.len() - 1;
    for (index, &limb) in limbs.iter().enumerate() {
        let elem_ty = limb_ty_of(&ty, index as u32);
        let elem = trunc_limb(func, block, limb, elem_ty);
        let kind = InstKind::InsertValue {
            agg,
            elem,
            index: index as u32,
        };
        agg = if index == last {
            match result {
                Some(value) => emit_as(func, block, kind, ty.clone(), value),
                None => emit_value(func, block, kind, ty.clone()),
            }
        } else {
            emit_value(func, block, kind, ty.clone())
        };
    }
    agg
}

fn and_bool(func: &mut Function, block: Block, a: Value, b: Value) -> Value {
    emit_value(func, block, InstKind::And(a, b), Ty::Bool)
}

fn or_bool(func: &mut Function, block: Block, a: Value, b: Value) -> Value {
    emit_value(func, block, InstKind::Or(a, b), Ty::Bool)
}

fn bool_not(func: &mut Function, block: Block, a: Value) -> Value {
    let zero = bool_const(func, false);
    emit_value(
        func,
        block,
        InstKind::Icmp {
            pred: IcmpPred::Eq,
            lhs: a,
            rhs: zero,
        },
        Ty::Bool,
    )
}

fn icmp_i64(func: &mut Function, block: Block, pred: IcmpPred, a: Value, b: Value) -> Value {
    emit_value(func, block, InstKind::Icmp { pred, lhs: a, rhs: b }, Ty::Bool)
}

fn add_limbs(func: &mut Function, block: Block, a: &[Value], b: &[Value]) -> Vec<Value> {
    let mut out = Vec::with_capacity(a.len());
    let mut carry = bool_const(func, false);
    for (&x, &y) in a.iter().zip(b) {
        let t = emit_value(func, block, InstKind::Add(x, y), Ty::int(LIMB_BITS));
        let c1 = icmp_i64(func, block, IcmpPred::Ult, t, x);
        let carry64 = zext_i64(func, block, carry);
        let u = emit_value(func, block, InstKind::Add(t, carry64), Ty::int(LIMB_BITS));
        let c2 = icmp_i64(func, block, IcmpPred::Ult, u, carry64);
        carry = or_bool(func, block, c1, c2);
        out.push(u);
    }
    out
}

fn sub_limbs(func: &mut Function, block: Block, a: &[Value], b: &[Value]) -> Vec<Value> {
    let mut out = Vec::with_capacity(a.len());
    let mut borrow = bool_const(func, false);
    for (&x, &y) in a.iter().zip(b) {
        let t = emit_value(func, block, InstKind::Sub(x, y), Ty::int(LIMB_BITS));
        let b1 = icmp_i64(func, block, IcmpPred::Ult, x, y);
        let borrow64 = zext_i64(func, block, borrow);
        let u = emit_value(func, block, InstKind::Sub(t, borrow64), Ty::int(LIMB_BITS));
        let b2 = icmp_i64(func, block, IcmpPred::Ult, t, borrow64);
        borrow = or_bool(func, block, b1, b2);
        out.push(u);
    }
    out
}

fn bitwise_limbs(func: &mut Function, block: Block, a: &[Value], b: &[Value], op: fn(Value, Value) -> InstKind) -> Vec<Value> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| emit_value(func, block, op(x, y), Ty::int(LIMB_BITS)))
        .collect()
}

fn mul_limbs(func: &mut Function, block: Block, a: &[Value], b: &[Value]) -> Vec<Value> {
    let n = a.len();
    let mut out = Vec::with_capacity(n);
    let mut carry = i64_const(func, 0);
    for k in 0..n {
        let mut cur = i64_const(func, 0);
        let first = k.saturating_sub(n - 1);
        for i in first..=k {
            let j = k - i;
            let wide_ty = Ty::int(2 * LIMB_BITS);
            let x = emit_value(func, block, InstKind::Zext(a[i]), wide_ty.clone());
            let y = emit_value(func, block, InstKind::Zext(b[j]), wide_ty.clone());
            let product = emit_value(func, block, InstKind::Mul(x, y), wide_ty.clone());
            let cur_wide = emit_value(func, block, InstKind::Zext(cur), wide_ty.clone());
            let carry_wide = emit_value(func, block, InstKind::Zext(carry), wide_ty.clone());
            let sum = emit_value(func, block, InstKind::Add(cur_wide, product), wide_ty.clone());
            let sum = emit_value(func, block, InstKind::Add(sum, carry_wide), wide_ty.clone());
            cur = emit_value(func, block, InstKind::Trunc(sum), Ty::int(LIMB_BITS));
            let shift = func.push_const(Const::Int {
                bits: 2 * LIMB_BITS,
                value: i128::from(LIMB_BITS),
            });
            let high = emit_value(func, block, InstKind::LShr(sum, shift), wide_ty);
            carry = emit_value(func, block, InstKind::Trunc(high), Ty::int(LIMB_BITS));
        }
        out.push(cur);
    }
    out
}

fn negate_limbs(func: &mut Function, block: Block, a: &[Value]) -> Vec<Value> {
    let zeros: Vec<Value> = (0..a.len()).map(|_| i64_const(func, 0)).collect();
    sub_limbs(func, block, &zeros, a)
}

fn not_limbs(func: &mut Function, block: Block, a: &[Value]) -> Vec<Value> {
    a.iter()
        .map(|&x| {
            let ones = i64_const(func, -1);
            emit_value(func, block, InstKind::Xor(x, ones), Ty::int(LIMB_BITS))
        })
        .collect()
}

fn select_limbs(func: &mut Function, block: Block, cond: Value, a: &[Value], b: &[Value]) -> Vec<Value> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            emit_value(
                func,
                block,
                InstKind::Select {
                    cond,
                    then_val: x,
                    else_val: y,
                },
                Ty::int(LIMB_BITS),
            )
        })
        .collect()
}

fn select_i64(func: &mut Function, block: Block, cond: Value, a: Value, b: Value) -> Value {
    emit_value(
        func,
        block,
        InstKind::Select {
            cond,
            then_val: a,
            else_val: b,
        },
        Ty::int(LIMB_BITS),
    )
}

fn top_limb_bits(bits: u32) -> u32 {
    let rem = bits % LIMB_BITS;
    if rem == 0 { LIMB_BITS } else { rem }
}

fn sign_bit(func: &mut Function, block: Block, limbs: &[Value], bits: u32) -> Value {
    let top = *limbs.last().expect("wide value has limbs");
    let shift = i64_const(func, i64::from(top_limb_bits(bits) - 1));
    let shifted = emit_value(func, block, InstKind::LShr(top, shift), Ty::int(LIMB_BITS));
    let one = i64_const(func, 1);
    emit_value(func, block, InstKind::And(shifted, one), Ty::int(LIMB_BITS))
}

fn sign_mask(func: &mut Function, block: Block, limbs: &[Value], bits: u32) -> Value {
    let bit = sign_bit(func, block, limbs, bits);
    let zero = i64_const(func, 0);
    emit_value(func, block, InstKind::Sub(zero, bit), Ty::int(LIMB_BITS))
}

/// Unsigned comparison of canonical limbs. Returns `(lt, eq)`.
fn compare_limbs(func: &mut Function, block: Block, a: &[Value], b: &[Value]) -> (Value, Value) {
    let mut lt = bool_const(func, false);
    let mut eq = bool_const(func, true);
    for (&x, &y) in a.iter().zip(b).rev() {
        let x_lt = icmp_i64(func, block, IcmpPred::Ult, x, y);
        let x_eq = icmp_i64(func, block, IcmpPred::Eq, x, y);
        let lt_here = and_bool(func, block, eq, x_lt);
        lt = or_bool(func, block, lt, lt_here);
        eq = and_bool(func, block, eq, x_eq);
    }
    (lt, eq)
}

fn eval_icmp_limbs(func: &mut Function, block: Block, pred: IcmpPred, a: &[Value], b: &[Value], bits: u32) -> Value {
    let signed = matches!(
        pred,
        IcmpPred::Slt | IcmpPred::Sle | IcmpPred::Sgt | IcmpPred::Sge
    );
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    if signed {
        let flip = 1i64 << (top_limb_bits(bits) - 1);
        let flip = i64_const(func, flip);
        let last = a.len() - 1;
        a[last] = emit_value(func, block, InstKind::Xor(a[last], flip), Ty::int(LIMB_BITS));
        b[last] = emit_value(func, block, InstKind::Xor(b[last], flip), Ty::int(LIMB_BITS));
    }
    let (lt, eq) = compare_limbs(func, block, &a, &b);
    match pred {
        IcmpPred::Eq => eq,
        IcmpPred::Ne => bool_not(func, block, eq),
        IcmpPred::Ult | IcmpPred::Slt => lt,
        IcmpPred::Ule | IcmpPred::Sle => or_bool(func, block, lt, eq),
        IcmpPred::Ugt | IcmpPred::Sgt => {
            let le = or_bool(func, block, lt, eq);
            bool_not(func, block, le)
        }
        IcmpPred::Uge | IcmpPred::Sge => bool_not(func, block, lt),
    }
}

fn shift_limbs(
    func: &mut Function,
    block: Block,
    a: &[Value],
    count: Value,
    bits: u32,
    kind: ShiftKind,
) -> Vec<Value> {
    let n = a.len();
    let count = to_i64_count(func, block, count);
    let six = i64_const(func, i64::from(LIMB_BITS.trailing_zeros()));
    let limb_shift = emit_value(func, block, InstKind::LShr(count, six), Ty::int(LIMB_BITS));
    let mask = i64_const(func, i64::from(LIMB_BITS - 1));
    let bit_shift = emit_value(func, block, InstKind::And(count, mask), Ty::int(LIMB_BITS));
    let bit_shift_wide = emit_value(func, block, InstKind::Zext(bit_shift), Ty::int(2 * LIMB_BITS));
    let ext = |func: &mut Function, index: usize, sign: Option<Value>| -> Value {
        if index < n {
            a[index]
        } else {
            match sign {
                Some(mask) => mask,
                None => i64_const(func, 0),
            }
        }
    };
    let sign = matches!(kind, ShiftKind::AShr).then(|| sign_mask(func, block, a, bits));
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut value = if matches!(kind, ShiftKind::AShr) {
            sign.expect("ashr has a sign mask")
        } else {
            i64_const(func, 0)
        };
        // A left shift pulls source limb `i - limb_shift`; a right shift
        // pulls `i + limb_shift`, so the two scan opposite directions.
        let sources: Box<dyn Iterator<Item = usize>> = match kind {
            ShiftKind::Shl => Box::new((0..=i).rev()),
            ShiftKind::LShr | ShiftKind::AShr => Box::new(i..n),
        };
        for j in sources {
            let source = match kind {
                ShiftKind::Shl => {
                    let lo = if j == 0 { i64_const(func, 0) } else { ext(func, j - 1, None) };
                    let hi = ext(func, j, None);
                    pair_shift(func, block, lo, hi, bit_shift_wide, true)
                }
                ShiftKind::LShr | ShiftKind::AShr => {
                    let lo = ext(func, j, sign);
                    let hi = ext(func, j + 1, sign);
                    pair_shift(func, block, lo, hi, bit_shift_wide, false)
                }
            };
            let distance = j.abs_diff(i) as i64;
            let distance = i64_const(func, distance);
            let matches = icmp_i64(func, block, IcmpPred::Eq, limb_shift, distance);
            value = select_i64(func, block, matches, source, value);
        }
        out.push(value);
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ShiftKind {
    Shl,
    LShr,
    AShr,
}

/// Shift the 128-bit pair `lo | hi << 64` by `0..63` and return the
/// selected 64-bit half.
fn pair_shift(func: &mut Function, block: Block, lo: Value, hi: Value, shift: Value, left: bool) -> Value {
    let wide_ty = Ty::int(2 * LIMB_BITS);
    let lo = emit_value(func, block, InstKind::Zext(lo), wide_ty.clone());
    let hi = emit_value(func, block, InstKind::Zext(hi), wide_ty.clone());
    let sixty_four = func.push_const(Const::Int {
        bits: 2 * LIMB_BITS,
        value: i128::from(LIMB_BITS),
    });
    let hi = emit_value(func, block, InstKind::Shl(hi, sixty_four), wide_ty.clone());
    let pair = emit_value(func, block, InstKind::Or(lo, hi), wide_ty.clone());
    let shifted = if left {
        emit_value(func, block, InstKind::Shl(pair, shift), wide_ty.clone())
    } else {
        emit_value(func, block, InstKind::LShr(pair, shift), wide_ty.clone())
    };
    let sixty_four = func.push_const(Const::Int {
        bits: 2 * LIMB_BITS,
        value: i128::from(LIMB_BITS),
    });
    let selected = if left {
        emit_value(func, block, InstKind::LShr(shifted, sixty_four), wide_ty)
    } else {
        shifted
    };
    emit_value(func, block, InstKind::Trunc(selected), Ty::int(LIMB_BITS))
}

/// Extend a scalar or wide integer into canonical 64-bit limbs, filling
/// with the sign bit for `sext` and zero otherwise.
fn extend_to_limbs(func: &mut Function, block: Block, value: Value, target_bits: u32, sext: bool) -> Vec<Value> {
    let target_count = target_bits.div_ceil(LIMB_BITS);
    let source_ty = func.values[value].ty.clone();
    let (mut limbs, source_bits) = match source_ty {
        Ty::WideInt { bits } => (limbs_of(func, block, value), bits),
        Ty::Bool => {
            let v = zext_i64(func, block, value);
            (vec![v], 1)
        }
        Ty::Int { bits } if bits <= LIMB_BITS => {
            let v = if bits < LIMB_BITS {
                let kind = if sext { InstKind::Sext(value) } else { InstKind::Zext(value) };
                emit_value(func, block, kind, Ty::int(LIMB_BITS))
            } else {
                value
            };
            (vec![v], bits)
        }
        Ty::Int { bits } if bits <= MAX_SCALAR_INT_BITS => {
            let lo = emit_value(func, block, InstKind::Trunc(value), Ty::int(LIMB_BITS));
            let sixty_four = func.push_const(Const::Int { bits, value: i128::from(LIMB_BITS) });
            let high = emit_value(func, block, InstKind::LShr(value, sixty_four), Ty::int(bits));
            let hi = emit_value(func, block, InstKind::Trunc(high), Ty::int(LIMB_BITS));
            (vec![lo, hi], bits)
        }
        other => panic!("cannot extend {other:?} to a wide integer"),
    };
    let fill = if sext {
        sign_mask(func, block, &limbs, source_bits)
    } else {
        i64_const(func, 0)
    };
    while limbs.len() < target_count as usize {
        limbs.push(fill);
    }
    limbs.truncate(target_count as usize);
    limbs
}

/// Low `target_bits` of a wide value, as a scalar.
fn trunc_wide_to_scalar(func: &mut Function, block: Block, limbs: &[Value], _source_bits: u32, target_bits: u32) -> Value {
    assert!(target_bits <= MAX_SCALAR_INT_BITS);
    if target_bits <= LIMB_BITS {
        return trunc_limb(func, block, limbs[0], Ty::int(target_bits));
    }
    let wide_ty = Ty::int(MAX_SCALAR_INT_BITS);
    let lo = emit_value(func, block, InstKind::Zext(limbs[0]), wide_ty.clone());
    let sixty_four = func.push_const(Const::Int {
        bits: MAX_SCALAR_INT_BITS,
        value: i128::from(LIMB_BITS),
    });
    let hi = emit_value(func, block, InstKind::Zext(limbs[1]), wide_ty.clone());
    let hi = emit_value(func, block, InstKind::Shl(hi, sixty_four), wide_ty.clone());
    let joined = emit_value(func, block, InstKind::Or(lo, hi), wide_ty);
    if target_bits == MAX_SCALAR_INT_BITS {
        joined
    } else {
        emit_value(func, block, InstKind::Trunc(joined), Ty::int(target_bits))
    }
}

fn udiv_limbs(func: &mut Function, block: Block, a: &[Value], b: &[Value], bits: u32, kind: DivKind, helpers: &HashMap<(u32, DivKind), Func>) -> Vec<Value> {
    let helper = helpers[&(bits, kind)];
    let a = build_wide(func, block, bits, a, None);
    let b = build_wide(func, block, bits, b, None);
    let out = emit_value(func, block, InstKind::Call { callee: helper, args: [a, b].into_iter().collect() }, wide_ty(bits));
    limbs_of(func, block, out)
}

fn sdiv_limbs(func: &mut Function, block: Block, a: &[Value], b: &[Value], bits: u32, remainder: bool, kind: DivKind, helpers: &HashMap<(u32, DivKind), Func>) -> Vec<Value> {
    let a_neg = sign_bit(func, block, a, bits);
    let b_neg = sign_bit(func, block, b, bits);
    let zero = i64_const(func, 0);
    let a_neg_bool = icmp_i64(func, block, IcmpPred::Ne, a_neg, zero);
    let b_neg_bool = icmp_i64(func, block, IcmpPred::Ne, b_neg, zero);
    let neg_a = negate_limbs(func, block, a);
    let neg_b = negate_limbs(func, block, b);
    let ua = select_limbs(func, block, a_neg_bool, &neg_a, a);
    let ub = select_limbs(func, block, b_neg_bool, &neg_b, b);
    let quotient = udiv_limbs(func, block, &ua, &ub, bits, kind, helpers);
    let negate_result = if remainder {
        a_neg_bool
    } else {
        emit_value(
            func,
            block,
            InstKind::Xor(a_neg_bool, b_neg_bool),
            Ty::Bool,
        )
    };
    let neg_quotient = negate_limbs(func, block, &quotient);
    select_limbs(func, block, negate_result, &neg_quotient, &quotient)
}

fn float_from_limbs(func: &mut Function, block: Block, limbs: &[Value], bits: u32, result_ty: Ty, signed: bool) -> Value {
    let f64_ty = Ty::Float(crate::ty::FloatKind::F64);
    // A negative value is accumulated through its magnitude, so the
    // limbs never have to be interpreted modulo 2^bits in float space.
    let (limbs, negative) = if signed {
        let sign = sign_bit(func, block, limbs, bits);
        let zero = i64_const(func, 0);
        let negative = icmp_i64(func, block, IcmpPred::Ne, sign, zero);
        let negated = negate_limbs(func, block, limbs);
        (select_limbs(func, block, negative, &negated, limbs), Some(negative))
    } else {
        (limbs.to_vec(), None)
    };
    let mut acc = typed_const(func, f64_ty.clone(), Const::Float { kind: crate::ty::FloatKind::F64, bits: 0 });
    let two64 = typed_const(
        func,
        f64_ty.clone(),
        Const::Float {
            kind: crate::ty::FloatKind::F64,
            bits: (2f64.powi(64)).to_bits() as u128,
        },
    );
    for &limb in limbs.iter().rev() {
        acc = emit_value(func, block, InstKind::FMul(acc, two64), f64_ty.clone());
        let piece = emit_value(func, block, InstKind::UItoF(limb), f64_ty.clone());
        acc = emit_value(func, block, InstKind::FAdd(acc, piece), f64_ty.clone());
    }
    if let Some(negative) = negative {
        let negated = emit_value(func, block, InstKind::FNeg(acc), f64_ty.clone());
        acc = emit_value(
            func,
            block,
            InstKind::Select {
                cond: negative,
                then_val: negated,
                else_val: acc,
            },
            f64_ty.clone(),
        );
    }
    let Ty::Float(kind) = result_ty else {
        panic!("integer-to-float result is not a float");
    };
    match kind {
        crate::ty::FloatKind::F64 => acc,
        crate::ty::FloatKind::F32 => emit_value(func, block, InstKind::Fptrunc(acc), result_ty),
        crate::ty::FloatKind::F80 | crate::ty::FloatKind::F128 => {
            emit_value(func, block, InstKind::Fpext(acc), result_ty)
        }
    }
}

/// Truncate a float to a wide integer's limbs.
///
/// The backend lowers `fcvt_to_sint`/`fcvt_to_uint` only at 32 and 64
/// bits, so an `f32`/`f64` source is split into its integer mantissa
/// and binary exponent here and the mantissa is shifted into place with
/// the wide integer shift. 80- and 128-bit floats already have a
/// working scalar-to-`i128` path, which then extends to the target
/// width.
fn limbs_from_float(func: &mut Function, block: Block, value: Value, target_bits: u32, signed: bool) -> Vec<Value> {
    let kind = match func.values[value].ty {
        Ty::Float(kind) => kind,
        ref other => panic!("float-to-integer source is {other:?}"),
    };
    let (frac_bits, exp_mask, bias, sign_shift, raw) = match kind {
        crate::ty::FloatKind::F32 => {
            let raw = emit_value(func, block, InstKind::Bitcast(value), Ty::int(32));
            let wide = emit_value(func, block, InstKind::Zext(raw), Ty::int(LIMB_BITS));
            (23, 0xff, 127 + 23, 31, wide)
        }
        crate::ty::FloatKind::F64 => {
            let raw = emit_value(func, block, InstKind::Bitcast(value), Ty::int(LIMB_BITS));
            (52, 0x7ff, 1023 + 52, 63, raw)
        }
        crate::ty::FloatKind::F80 | crate::ty::FloatKind::F128 => {
            let scalar_ty = Ty::int(MAX_SCALAR_INT_BITS);
            let kind = if signed {
                InstKind::SFtoI(value)
            } else {
                InstKind::UFtoI(value)
            };
            let scalar = emit_value(func, block, kind, scalar_ty);
            return extend_to_limbs(func, block, scalar, target_bits, signed);
        }
    };

    let zero = i64_const(func, 0);
    let sign = {
        let shift = i64_const(func, sign_shift);
        let shifted = emit_value(func, block, InstKind::LShr(raw, shift), Ty::int(LIMB_BITS));
        let one = i64_const(func, 1);
        emit_value(func, block, InstKind::And(shifted, one), Ty::int(LIMB_BITS))
    };
    let negative = icmp_i64(func, block, IcmpPred::Ne, sign, zero);
    let exponent = {
        let shift = i64_const(func, frac_bits);
        let shifted = emit_value(func, block, InstKind::LShr(raw, shift), Ty::int(LIMB_BITS));
        let mask = i64_const(func, exp_mask);
        emit_value(func, block, InstKind::And(shifted, mask), Ty::int(LIMB_BITS))
    };
    let fraction = {
        let mask = i64_const(func, (1 << frac_bits) - 1);
        emit_value(func, block, InstKind::And(raw, mask), Ty::int(LIMB_BITS))
    };
    // The implicit leading one is absent for zero and subnormals, whose
    // integer part is zero anyway.
    let normal = icmp_i64(func, block, IcmpPred::Ne, exponent, zero);
    let mantissa = {
        let implicit = i64_const(func, 1 << frac_bits);
        let with_implicit = emit_value(func, block, InstKind::Or(fraction, implicit), Ty::int(LIMB_BITS));
        select_i64(func, block, normal, with_implicit, fraction)
    };
    // value = mantissa * 2^(exponent - bias), truncated toward zero.
    let bias = i64_const(func, bias);
    let exponent = emit_value(func, block, InstKind::Sub(exponent, bias), Ty::int(LIMB_BITS));
    let shift_right = icmp_i64(func, block, IcmpPred::Slt, exponent, zero);
    let distance = emit_value(func, block, InstKind::Neg(exponent), Ty::int(LIMB_BITS));
    let amount = select_i64(func, block, shift_right, distance, exponent);

    let count = target_bits.div_ceil(LIMB_BITS) as usize;
    let mut mantissa_limbs = vec![mantissa];
    mantissa_limbs.resize(count, zero);
    let left = shift_limbs(func, block, &mantissa_limbs, amount, target_bits, ShiftKind::Shl);
    let right = shift_limbs(func, block, &mantissa_limbs, amount, target_bits, ShiftKind::LShr);
    let magnitude = select_limbs(func, block, shift_right, &right, &left);
    let negated = negate_limbs(func, block, &magnitude);
    select_limbs(func, block, negative, &negated, &magnitude)
}

fn legalize_function(func: &mut Function, helpers: &HashMap<(u32, DivKind), Func>) {
    let blocks = func.blocks.len();
    for index in 0..blocks {
        let block = Block::from_usize(index);
        let pending = std::mem::take(&mut func.blocks[block].insts);
        for inst in pending {
            let data = func.insts[inst].clone();
            let Some(result) = data.result else {
                func.blocks[block].insts.push(inst);
                continue;
            };
            let result_ty = func.values[result].ty.clone();
            let is_wide_result = matches!(result_ty, Ty::WideInt { .. });
            let operands = data.kind.operands();
            let has_wide_operand = operands.iter().any(|v| matches!(func.values[*v].ty, Ty::WideInt { .. }));
            if (!is_wide_result && !has_wide_operand) || !needs_limb_rewrite(&data.kind) {
                func.blocks[block].insts.push(inst);
                continue;
            }
            func.insts[inst].result = None;
            let first_new = func.insts.len();
            match &data.kind {
                InstKind::Add(a, b) => {
                    let bits = wide_result_bits(func, result, &result_ty);
                    let x = limbs_of(func, block, *a);
                    let y = limbs_of(func, block, *b);
                    let limbs = add_limbs(func, block, &x, &y);
                    build_wide(func, block, bits, &limbs, Some(result));
                }
                InstKind::Sub(a, b) => {
                    let bits = wide_result_bits(func, result, &result_ty);
                    let x = limbs_of(func, block, *a);
                    let y = limbs_of(func, block, *b);
                    let limbs = sub_limbs(func, block, &x, &y);
                    build_wide(func, block, bits, &limbs, Some(result));
                }
                InstKind::Mul(a, b) => {
                    let bits = wide_result_bits(func, result, &result_ty);
                    let x = limbs_of(func, block, *a);
                    let y = limbs_of(func, block, *b);
                    let limbs = mul_limbs(func, block, &x, &y);
                    build_wide(func, block, bits, &limbs, Some(result));
                }
                InstKind::And(a, b) => {
                    let bits = wide_result_bits(func, result, &result_ty);
                    let x = limbs_of(func, block, *a);
                    let y = limbs_of(func, block, *b);
                    let limbs = bitwise_limbs(func, block, &x, &y, InstKind::And);
                    build_wide(func, block, bits, &limbs, Some(result));
                }
                InstKind::Or(a, b) => {
                    let bits = wide_result_bits(func, result, &result_ty);
                    let x = limbs_of(func, block, *a);
                    let y = limbs_of(func, block, *b);
                    let limbs = bitwise_limbs(func, block, &x, &y, InstKind::Or);
                    build_wide(func, block, bits, &limbs, Some(result));
                }
                InstKind::Xor(a, b) => {
                    let bits = wide_result_bits(func, result, &result_ty);
                    let x = limbs_of(func, block, *a);
                    let y = limbs_of(func, block, *b);
                    let limbs = bitwise_limbs(func, block, &x, &y, InstKind::Xor);
                    build_wide(func, block, bits, &limbs, Some(result));
                }
                InstKind::Neg(a) => {
                    let bits = wide_result_bits(func, result, &result_ty);
                    let x = limbs_of(func, block, *a);
                    let limbs = negate_limbs(func, block, &x);
                    build_wide(func, block, bits, &limbs, Some(result));
                }
                InstKind::Not(a) => {
                    let bits = wide_result_bits(func, result, &result_ty);
                    let x = limbs_of(func, block, *a);
                    let limbs = not_limbs(func, block, &x);
                    build_wide(func, block, bits, &limbs, Some(result));
                }
                InstKind::Shl(a, b) | InstKind::LShr(a, b) | InstKind::AShr(a, b) => {
                    let bits = wide_bits(func, *a);
                    let kind = match data.kind {
                        InstKind::Shl(..) => ShiftKind::Shl,
                        InstKind::LShr(..) => ShiftKind::LShr,
                        _ => ShiftKind::AShr,
                    };
                    let x = limbs_of(func, block, *a);
                    let limbs = shift_limbs(func, block, &x, *b, bits, kind);
                    build_wide(func, block, bits, &limbs, Some(result));
                }
                InstKind::Icmp { pred, lhs, rhs } => {
                    let bits = wide_bits(func, *lhs);
                    let a = limbs_of(func, block, *lhs);
                    let b = limbs_of(func, block, *rhs);
                    let cmp = eval_icmp_limbs(func, block, *pred, &a, &b, bits);
                    emit_as(func, block, InstKind::Bitcast(cmp), result_ty.clone(), result);
                }
                InstKind::Select {
                    cond,
                    then_val,
                    else_val,
                } => {
                    let bits = wide_bits(func, *then_val);
                    let a = limbs_of(func, block, *then_val);
                    let b = limbs_of(func, block, *else_val);
                    let limbs = select_limbs(func, block, *cond, &a, &b);
                    build_wide(func, block, bits, &limbs, Some(result));
                }
                InstKind::Trunc(a) => {
                    let Ty::WideInt { bits } = func.values[*a].ty else {
                        panic!("wide trunc of a non-wide value");
                    };
                    let limbs = limbs_of(func, block, *a);
                    match result_ty {
                        Ty::WideInt { bits: target } => {
                            let count = target.div_ceil(LIMB_BITS);
                            let mut out = limbs;
                            out.truncate(count as usize);
                            while out.len() < count as usize {
                                out.push(i64_const(func, 0));
                            }
                            build_wide(func, block, target, &out, Some(result));
                        }
                        Ty::Int { bits: target } => {
                            let value = trunc_wide_to_scalar(func, block, &limbs, bits, target);
                            emit_as(func, block, InstKind::Bitcast(value), result_ty.clone(), result);
                        }
                        other => panic!("wide trunc to {other:?}"),
                    }
                }
                InstKind::Zext(a) | InstKind::Sext(a) => {
                    let sext = matches!(data.kind, InstKind::Sext(_));
                    let bits = wide_result_bits(func, result, &result_ty);
                    let limbs = extend_to_limbs(func, block, *a, bits, sext);
                    build_wide(func, block, bits, &limbs, Some(result));
                }
                InstKind::Bitcast(a) => {
                    let bits = wide_result_bits(func, result, &result_ty);
                    let limbs = limbs_of(func, block, *a);
                    build_wide(func, block, bits, &limbs, Some(result));
                }
                InstKind::Udiv(a, b) => {
                    let bits = wide_bits(func, *a);
                    let x = limbs_of(func, block, *a);
                    let y = limbs_of(func, block, *b);
                    let limbs = udiv_limbs(func, block, &x, &y, bits, DivKind::Quotient, helpers);
                    build_wide(func, block, bits, &limbs, Some(result));
                }
                InstKind::Urem(a, b) => {
                    let bits = wide_bits(func, *a);
                    let x = limbs_of(func, block, *a);
                    let y = limbs_of(func, block, *b);
                    let limbs = udiv_limbs(func, block, &x, &y, bits, DivKind::Remainder, helpers);
                    build_wide(func, block, bits, &limbs, Some(result));
                }
                InstKind::Sdiv(a, b) => {
                    let bits = wide_bits(func, *a);
                    let x = limbs_of(func, block, *a);
                    let y = limbs_of(func, block, *b);
                    let limbs = sdiv_limbs(func, block, &x, &y, bits, false, DivKind::Quotient, helpers);
                    build_wide(func, block, bits, &limbs, Some(result));
                }
                InstKind::Srem(a, b) => {
                    let bits = wide_bits(func, *a);
                    let x = limbs_of(func, block, *a);
                    let y = limbs_of(func, block, *b);
                    let limbs = sdiv_limbs(func, block, &x, &y, bits, true, DivKind::Remainder, helpers);
                    build_wide(func, block, bits, &limbs, Some(result));
                }
                InstKind::SItoF(a) | InstKind::UItoF(a) => {
                    let bits = wide_bits(func, *a);
                    let limbs = limbs_of(func, block, *a);
                    let signed = matches!(data.kind, InstKind::SItoF(_));
                    let value = float_from_limbs(func, block, &limbs, bits, result_ty.clone(), signed);
                    emit_as(func, block, InstKind::Bitcast(value), result_ty.clone(), result);
                }
                InstKind::SFtoI(a) | InstKind::UFtoI(a) => {
                    let bits = wide_result_bits(func, result, &result_ty);
                    let signed = matches!(data.kind, InstKind::SFtoI(_));
                    let limbs = limbs_from_float(func, block, *a, bits, signed);
                    build_wide(func, block, bits, &limbs, Some(result));
                }
                other => panic!(
                    "wide integer legalization does not support {other:?} in {}",
                    func.name
                ),
            }
            let loc = data.loc;
            for id in first_new..func.insts.len() {
                if func.insts[id].loc.is_none() {
                    func.insts[id].loc = loc;
                }
            }
        }
    }
}

/// Instructions whose wide operands cannot flow through as aggregate
/// values: they compute a value in registers and must be rewritten
/// limb by limb. Everything else (loads, stores, calls, GEPs, aggregate
/// inserts/extracts, ...) already moves wide integers through memory.
fn needs_limb_rewrite(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::Add(..)
            | InstKind::Sub(..)
            | InstKind::Mul(..)
            | InstKind::Udiv(..)
            | InstKind::Sdiv(..)
            | InstKind::Urem(..)
            | InstKind::Srem(..)
            | InstKind::And(..)
            | InstKind::Or(..)
            | InstKind::Xor(..)
            | InstKind::Shl(..)
            | InstKind::LShr(..)
            | InstKind::AShr(..)
            | InstKind::Neg(..)
            | InstKind::Not(..)
            | InstKind::Icmp { .. }
            | InstKind::Trunc(..)
            | InstKind::Zext(..)
            | InstKind::Sext(..)
            | InstKind::Bitcast(..)
            | InstKind::SItoF(..)
            | InstKind::UItoF(..)
            | InstKind::SFtoI(..)
            | InstKind::UFtoI(..)
            | InstKind::Select { .. }
    )
}

fn wide_result_bits(func: &Function, result: Value, ty: &Ty) -> u32 {
    match ty {
        Ty::WideInt { bits } => *bits,
        other => panic!(
            "wide operation has non-wide result {} of type {other:?}",
            func.values[result].ty.size_bytes()
        ),
    }
}

impl LegalizeWideInts {
    fn ensure_helper(&mut self, module: &mut Module, bits: u32, kind: DivKind) {
        if self.helpers.contains_key(&(bits, kind)) {
            return;
        }
        let helper = build_div_helper(bits, kind);
        let id = module.push_func(helper);
        self.helpers.insert((bits, kind), id);
    }
}

fn build_div_helper(bits: u32, kind: DivKind) -> Function {
    let name = match kind {
        DivKind::Quotient => format!("__glcc_wide_udiv${bits}"),
        DivKind::Remainder => format!("__glcc_wide_urem${bits}"),
    };
    let wide = wide_ty(bits);
    let (mut func, entry) = Function::new(&name, Signature::new(vec![wide.clone(), wide.clone()], wide.clone()));
    func.local = true;
    let a = add_param(&mut func, entry, wide.clone(), 0);
    let b = add_param(&mut func, entry, wide.clone(), 1);
    let n = bits.div_ceil(LIMB_BITS);

    let a_arr = alloca_limbs(&mut func, entry, n);
    let b_arr = alloca_limbs(&mut func, entry, n);
    let q_arr = alloca_limbs(&mut func, entry, n);
    let r_arr = alloca_limbs(&mut func, entry, n);
    let a_limbs = limbs_of(&mut func, entry, a);
    let b_limbs = limbs_of(&mut func, entry, b);
    store_limbs(&mut func, entry, a_arr, &a_limbs, n);
    store_limbs(&mut func, entry, b_arr, &b_limbs, n);
    let zero = i64_const(&mut func, 0);
    for i in 0..n {
        store_limb(&mut func, entry, q_arr, i, zero);
        store_limb(&mut func, entry, r_arr, i, zero);
    }

    let loop_block = func.push_block(vec![Ty::int(LIMB_BITS)]);
    let body = func.push_block(vec![Ty::int(LIMB_BITS)]);
    let done = func.push_block(vec![]);
    let i0 = func.push_const(Const::Int {
        bits: LIMB_BITS,
        value: i128::from(bits - 1),
    });
    func.blocks[entry].term = Some(Terminator::Br {
        dest: loop_block,
        args: [i0].into_iter().collect(),
    });

    let i = func.blocks[loop_block].params[0];
    let zero64 = i64_const(&mut func, 0);
    let nonneg = icmp_i64(&mut func, loop_block, IcmpPred::Sge, i, zero64);
    func.blocks[loop_block].term = Some(Terminator::CondBr {
        cond: nonneg,
        then_block: body,
        else_block: done,
        then_args: [i].into_iter().collect(),
        else_args: Default::default(),
    });

    // r = (r << 1) | bit(a, i). The shifted-out bit of the top limb is
    // tracked separately: the remainder can transiently need one bit
    // more than the integer's precision.
    let last = n - 1;
    let top_width = top_limb_bits(bits);
    let mut carry = i64_const(&mut func, 0);
    let mut carry_out = bool_const(&mut func, false);
    for index in 0..n {
        let w = load_limb(&mut func, body, r_arr, index);
        let one = i64_const(&mut func, 1);
        let left = emit_value(&mut func, body, InstKind::Shl(w, one), Ty::int(LIMB_BITS));
        let next = emit_value(&mut func, body, InstKind::Or(left, carry), Ty::int(LIMB_BITS));
        if index == last && top_width < LIMB_BITS {
            let shift = i64_const(&mut func, i64::from(top_width - 1));
            let out = emit_value(&mut func, body, InstKind::LShr(next, shift), Ty::int(LIMB_BITS));
            let mask = i64_const(&mut func, 1);
            let out = emit_value(&mut func, body, InstKind::And(out, mask), Ty::int(LIMB_BITS));
            let out_zero = i64_const(&mut func, 0);
            carry_out = icmp_i64(&mut func, body, IcmpPred::Ne, out, out_zero);
        } else if index == last {
            carry_out = icmp_i64(&mut func, body, IcmpPred::Slt, next, zero64);
        }
        let next = mask_last_limb(&mut func, body, next, bits, index);
        store_limb(&mut func, body, r_arr, index, next);
        let rem = i64_const(&mut func, 63);
        carry = emit_value(&mut func, body, InstKind::LShr(w, rem), Ty::int(LIMB_BITS));
    }
    // r |= (a >> i) & 1, with `i` a runtime bit index.
    let six = i64_const(&mut func, i64::from(LIMB_BITS.trailing_zeros()));
    let word = emit_value(&mut func, body, InstKind::LShr(i, six), Ty::int(LIMB_BITS));
    let bit_mask = i64_const(&mut func, i64::from(LIMB_BITS - 1));
    let bit = emit_value(&mut func, body, InstKind::And(i, bit_mask), Ty::int(LIMB_BITS));
    let a_word = load_limb_dynamic(&mut func, body, a_arr, word);
    let a_bit = emit_value(&mut func, body, InstKind::LShr(a_word, bit), Ty::int(LIMB_BITS));
    let one = i64_const(&mut func, 1);
    let a_bit = emit_value(&mut func, body, InstKind::And(a_bit, one), Ty::int(LIMB_BITS));
    let r0 = load_limb(&mut func, body, r_arr, 0);
    let r0 = emit_value(&mut func, body, InstKind::Or(r0, a_bit), Ty::int(LIMB_BITS));
    store_limb(&mut func, body, r_arr, 0, r0);

    // The one extra bit decides the comparison: when the shift carried
    // out of the top, r is certainly at least b.
    let r_limbs: Vec<Value> = (0..n).map(|index| load_limb(&mut func, body, r_arr, index)).collect();
    let b_limbs: Vec<Value> = (0..n).map(|index| load_limb(&mut func, body, b_arr, index)).collect();
    let (lt, _) = compare_limbs(&mut func, body, &r_limbs, &b_limbs);
    let lt_or_carry = or_bool(&mut func, body, lt, carry_out);
    let ge = bool_not(&mut func, body, lt_or_carry);

    // r -= b when ge, with a borrow chain; q gets the current bit.
    let mut borrow = bool_const(&mut func, false);
    for index in 0..n {
        let rw = load_limb(&mut func, body, r_arr, index);
        let bw = load_limb(&mut func, body, b_arr, index);
        let t = emit_value(&mut func, body, InstKind::Sub(rw, bw), Ty::int(LIMB_BITS));
        let b1 = icmp_i64(&mut func, body, IcmpPred::Ult, rw, bw);
        let borrow64 = zext_i64(&mut func, body, borrow);
        let u = emit_value(&mut func, body, InstKind::Sub(t, borrow64), Ty::int(LIMB_BITS));
        let b2 = icmp_i64(&mut func, body, IcmpPred::Ult, t, borrow64);
        borrow = or_bool(&mut func, body, b1, b2);
        let u = mask_last_limb(&mut func, body, u, bits, index);
        let next = select_i64(&mut func, body, ge, u, rw);
        store_limb(&mut func, body, r_arr, index, next);
    }
    let q_word = load_limb_dynamic(&mut func, body, q_arr, word);
    let one_shifted = emit_value(&mut func, body, InstKind::Shl(one, bit), Ty::int(LIMB_BITS));
    let q_zero = i64_const(&mut func, 0);
    let set = select_i64(&mut func, body, ge, one_shifted, q_zero);
    let q_word = emit_value(&mut func, body, InstKind::Or(q_word, set), Ty::int(LIMB_BITS));
    store_limb_dynamic(&mut func, body, q_arr, word, q_word);

    let one = i64_const(&mut func, 1);
    let dec = emit_value(&mut func, body, InstKind::Sub(i, one), Ty::int(LIMB_BITS));
    func.blocks[body].term = Some(Terminator::Br {
        dest: loop_block,
        args: [dec].into_iter().collect(),
    });

    let result_limbs = (0..n)
        .map(|index| match kind {
            DivKind::Quotient => load_limb(&mut func, done, q_arr, index),
            DivKind::Remainder => load_limb(&mut func, done, r_arr, index),
        })
        .collect::<Vec<_>>();
    let result = build_wide(&mut func, done, bits, &result_limbs, None);
    func.blocks[done].term = Some(Terminator::Return { val: Some(result) });
    func
}

fn add_param(func: &mut Function, block: Block, ty: Ty, index: u32) -> Value {
    let value = func.values.push(ValueData {
        ty,
        def: Def::Param(block, index),
    });
    func.blocks[block].params.push(value);
    value
}

fn alloca_limbs(func: &mut Function, block: Block, count: u32) -> Value {
    emit_value(
        func,
        block,
        InstKind::Alloca {
            pointee: Ty::int(LIMB_BITS),
            count: u64::from(count),
        },
        Ty::ptr(Ty::int(LIMB_BITS)),
    )
}

fn limb_ptr(func: &mut Function, block: Block, base: Value, index: u32) -> Value {
    emit_value(
        func,
        block,
        InstKind::GetElementPtr {
            base,
            indices: [crate::ty::GepIndex::ConstIndex(u64::from(index))].into_iter().collect(),
        },
        Ty::ptr(Ty::int(LIMB_BITS)),
    )
}

fn limb_ptr_dynamic(func: &mut Function, block: Block, base: Value, index: Value) -> Value {
    emit_value(
        func,
        block,
        InstKind::GetElementPtr {
            base,
            indices: [crate::ty::GepIndex::Index(index)].into_iter().collect(),
        },
        Ty::ptr(Ty::int(LIMB_BITS)),
    )
}

fn load_limb(func: &mut Function, block: Block, base: Value, index: u32) -> Value {
    let addr = limb_ptr(func, block, base, index);
    emit_value(
        func,
        block,
        InstKind::Load {
            addr,
            flags: crate::ir::MemFlags::plain(),
        },
        Ty::int(LIMB_BITS),
    )
}

fn load_limb_dynamic(func: &mut Function, block: Block, base: Value, index: Value) -> Value {
    let addr = limb_ptr_dynamic(func, block, base, index);
    emit_value(
        func,
        block,
        InstKind::Load {
            addr,
            flags: crate::ir::MemFlags::plain(),
        },
        Ty::int(LIMB_BITS),
    )
}

fn store_limb(func: &mut Function, block: Block, base: Value, index: u32, value: Value) {
    let addr = limb_ptr(func, block, base, index);
    emit_unit(
        func,
        block,
        InstKind::Store {
            addr,
            val: value,
            flags: crate::ir::MemFlags::plain(),
        },
    );
}

fn store_limb_dynamic(func: &mut Function, block: Block, base: Value, index: Value, value: Value) {
    let addr = limb_ptr_dynamic(func, block, base, index);
    emit_unit(
        func,
        block,
        InstKind::Store {
            addr,
            val: value,
            flags: crate::ir::MemFlags::plain(),
        },
    );
}

fn store_limbs(func: &mut Function, block: Block, base: Value, limbs: &[Value], count: u32) {
    for index in 0..count {
        store_limb(func, block, base, index, limbs[index as usize]);
    }
}

fn mask_last_limb(func: &mut Function, block: Block, value: Value, bits: u32, index: u32) -> Value {
    let last = bits.div_ceil(LIMB_BITS) - 1;
    if index != last {
        return value;
    }
    let width = top_limb_bits(bits);
    if width == LIMB_BITS {
        return value;
    }
    let mask = i64_const(func, ((1u128 << width) - 1) as u64 as i64);
    emit_value(func, block, InstKind::And(value, mask), Ty::int(LIMB_BITS))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Module;
    use crate::ty::Signature;

    fn param(func: &mut Function, block: Block, ty: Ty, index: u32) -> Value {
        let value = func.values.push(ValueData {
            ty,
            def: Def::Param(block, index),
        });
        func.blocks[block].params.push(value);
        value
    }

    fn emit(func: &mut Function, block: Block, kind: InstKind, ty: Ty) -> Value {
        let result = func.values.push(ValueData {
            ty,
            def: Def::Inst(Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(InstData {
            block,
            kind,
            result: Some(result),
            loc: None,
        });
        func.values[result].def = Def::Inst(inst);
        func.blocks[block].insts.push(inst);
        result
    }

    fn ty_has_legalizable_int(ty: &Ty) -> bool {
        match ty {
            Ty::Int { bits } => *bits > MAX_SCALAR_INT_BITS,
            Ty::Ptr { pointee, .. } => ty_has_legalizable_int(pointee),
            Ty::Array { elem, .. } => ty_has_legalizable_int(elem),
            Ty::Struct { fields, .. } => fields.iter().any(ty_has_legalizable_int),
            _ => false,
        }
    }

    fn module_with_binop(divide: bool) -> Module {
        let mut module = Module::new();
        let sig = Signature::new(vec![Ty::int(256), Ty::int(256)], Ty::int(256));
        let (mut func, entry) = Function::new("binop", sig);
        let a = param(&mut func, entry, Ty::int(256), 0);
        let b = param(&mut func, entry, Ty::int(256), 1);
        let kind = if divide { InstKind::Udiv(a, b) } else { InstKind::Add(a, b) };
        let value = emit(&mut func, entry, kind, Ty::int(256));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(value) });
        module.push_func(func);
        module
    }

    #[test]
    fn rewrites_wide_addition_into_limbs() {
        let mut module = module_with_binop(false);
        assert!(LegalizeWideInts::new().run(&mut module));
        assert!(crate::verify::verify_module(&module).is_empty());

        let func = &module.funcs[0];
        assert_eq!(func.sig.ret, Ty::WideInt { bits: 256 });
        assert!(func.values.iter().all(|v| !ty_has_legalizable_int(&v.ty)));
        assert!(func.insts.iter().any(|i| matches!(&i.kind, InstKind::InsertValue { .. })));
        assert!(func.insts.iter().any(|i| matches!(&i.kind, InstKind::Icmp { pred: IcmpPred::Ult, .. })));
        assert!(!func.insts.iter().any(|i| matches!(&i.kind, InstKind::Add(..)) && i.result.is_some_and(|r| func.values[r].ty == Ty::int(256))));
    }

    #[test]
    fn synthesizes_one_division_helper_per_operation() {
        let mut module = module_with_binop(true);
        assert!(LegalizeWideInts::new().run(&mut module));
        assert!(crate::verify::verify_module(&module).is_empty());
        assert!(module.funcs.iter().any(|f| f.name == "__glcc_wide_udiv$256"));
        assert!(!module.funcs.iter().any(|f| f.name == "__glcc_wide_urem$256"));
    }

    #[test]
    fn optimize_at_o0_still_legalizes() {
        let mut module = module_with_binop(false);
        let report = crate::optimize::optimize(&mut module, crate::optimize::OptLevel::None);
        assert!(report.iter().any(|(name, changed)| *name == "legalize-wide-ints" && *changed));
        assert!(crate::verify::verify_module(&module).is_empty());
        assert!(module.funcs[0].values.iter().all(|v| !ty_has_legalizable_int(&v.ty)));
    }
}
