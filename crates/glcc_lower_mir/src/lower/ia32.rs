use super::*;

impl<'s, 'i> Lowerer<'s, 'i> {
    /// Lower a GCC x86 `__builtin_ia32_*` call. Pure MIR where the
    /// instruction set allows (`bsr` off `clz`, the rotates off
    /// shifts and masks, carry/borrow off comparisons, the count
    /// builtins off `clz`/`ctz`); inline asm where MIR has no opcode
    /// (the SSE4.2 CRC-32C step, `rdtsc`/`rdtscp`/`rdpmc`, BMI2
    /// `pdep`/`pext`, the CET shadow-stack and cache-management
    /// instructions). 64-bit results read the whole `rax`, which the
    /// hardware zero-extends when a template writes `eax`/`edx`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lower_ia32(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        op: &fe::check::Ia32Op,
        args: &[fe::check::CheckedExpr],
        fe_ty: &fe::parse::Ty,
    ) -> mir::Value {
        match op {
            // `sfence` orders stores; MIR's fence is a full barrier,
            // strictly stronger.
            fe::check::Ia32Op::Sfence => {
                self.emit_unit(func, frame, mir::InstKind::AtomicFence);
                self.iconst(func, 32, 0)
            }
            fe::check::Ia32Op::NoOperand { mnemonic } => {
                self.asm(func, frame, mnemonic, 0, vec![], true);
                self.iconst(func, 32, 0)
            }
            // `bsr(x)` = top_bit_index - clz(x); a zero operand is
            // undefined on x86, and `clz(0) == width` makes this
            // yield `-1`.
            fe::check::Ia32Op::Bsr32 | fe::check::Ia32Op::Bsr64 => {
                let x = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let operand_mir = lower_ty(&args[0].ty, self.sema, self.interner);
                let bits = int_bits(&operand_mir);
                let leading = self.emit(func, frame, mir::InstKind::Clz(x), operand_mir);
                let top = self.iconst(func, bits, i128::from(bits - 1));
                let r = self.emit(func, frame, mir::InstKind::Sub(top, leading), mir::Ty::int(bits));
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // Rotate with the count masked to the width; the second
            // mask folds the `c == 0` case (then `bits - c` is
            // `bits`, which would shift the whole width away).
            fe::check::Ia32Op::Rol8 | fe::check::Ia32Op::Rol16 | fe::check::Ia32Op::Ror8 | fe::check::Ia32Op::Ror16 => {
                let bits = match op {
                    fe::check::Ia32Op::Rol8 | fe::check::Ia32Op::Ror8 => 8,
                    _ => 16,
                };
                let operand_mir = mir::Ty::int(bits);
                let mask = self.iconst(func, bits, i128::from(bits - 1));
                let x = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let c = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let c = self.convert(func, frame, c, &args[1].ty, &args[0].ty);
                let c = self.emit(func, frame, mir::InstKind::And(c, mask), operand_mir.clone());
                let width_const = self.iconst(func, bits, i128::from(bits));
                let complement = self.emit(func, frame, mir::InstKind::Sub(width_const, c), operand_mir.clone());
                let complement = self.emit(func, frame, mir::InstKind::And(complement, mask), operand_mir.clone());
                // Rotate left moves the high bits down (`x >> bits-c`
                // supplies them); rotate right swaps the two shifts'
                // roles.
                let (down_amt, up_amt) = match op {
                    fe::check::Ia32Op::Ror8 | fe::check::Ia32Op::Ror16 => (c, complement),
                    _ => (complement, c),
                };
                let down = self.emit(func, frame, mir::InstKind::LShr(x, down_amt), operand_mir.clone());
                let up = self.emit(func, frame, mir::InstKind::Shl(x, up_amt), operand_mir.clone());
                let r = self.emit(func, frame, mir::InstKind::Or(down, up), operand_mir);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // The carry chain behind `_addcarry_u32/64`: the sum
            // overflows exactly when `a + b` wraps, or fills the
            // width and `cin` is set.
            fe::check::Ia32Op::AddCarry { width } => {
                let cin8 = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[2]);
                let out_ptr = self.expr_value(module, func, frame, funcs, externs, globals, &args[3]);
                let bits = u32::from(*width) * 8;
                let operand_mir = mir::Ty::int(bits);
                let zero = self.iconst(func, 8, 0);
                let cin_bool = self.emit(
                    func,
                    frame,
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Ne,
                        lhs: cin8,
                        rhs: zero,
                    },
                    mir::Ty::Bool,
                );
                let cin = self.emit(func, frame, mir::InstKind::Zext(cin8), operand_mir.clone());
                let ab = self.emit(func, frame, mir::InstKind::Add(a, b), operand_mir.clone());
                let wrapped = self.emit(
                    func,
                    frame,
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Ult,
                        lhs: ab,
                        rhs: a,
                    },
                    mir::Ty::Bool,
                );
                let full = self.iconst(func, bits, (1i128 << bits) - 1);
                let at_full = self.emit(
                    func,
                    frame,
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Eq,
                        lhs: ab,
                        rhs: full,
                    },
                    mir::Ty::Bool,
                );
                let at_full = self.emit(func, frame, mir::InstKind::And(at_full, cin_bool), mir::Ty::Bool);
                let carry = self.emit(func, frame, mir::InstKind::Or(wrapped, at_full), mir::Ty::Bool);
                let out = self.emit(func, frame, mir::InstKind::Add(ab, cin), operand_mir);
                self.store(func, frame, out_ptr, out, false);
                self.convert(func, frame, carry, &fe::parse::Ty::Bool, fe_ty)
            }
            // `_subborrow_u32/64`: `a - b - cin` borrows exactly when
            // `b + cin > a` without wrapping — `b > a`, or `b == a`
            // and `cin`.
            fe::check::Ia32Op::Sbb { width } => {
                let cin8 = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[2]);
                let out_ptr = self.expr_value(module, func, frame, funcs, externs, globals, &args[3]);
                let operand_mir = mir::Ty::int(u32::from(*width) * 8);
                let zero = self.iconst(func, 8, 0);
                let cin_bool = self.emit(
                    func,
                    frame,
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Ne,
                        lhs: cin8,
                        rhs: zero,
                    },
                    mir::Ty::Bool,
                );
                let cin = self.emit(func, frame, mir::InstKind::Zext(cin8), operand_mir.clone());
                let out = self.emit(func, frame, mir::InstKind::Sub(a, b), operand_mir.clone());
                let out = self.emit(func, frame, mir::InstKind::Sub(out, cin), operand_mir.clone());
                self.store(func, frame, out_ptr, out, false);
                let gt = self.emit(
                    func,
                    frame,
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Ugt,
                        lhs: b,
                        rhs: a,
                    },
                    mir::Ty::Bool,
                );
                let eq = self.emit(
                    func,
                    frame,
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Eq,
                        lhs: b,
                        rhs: a,
                    },
                    mir::Ty::Bool,
                );
                let with_cin = self.emit(func, frame, mir::InstKind::And(eq, cin_bool), mir::Ty::Bool);
                let bo = self.emit(func, frame, mir::InstKind::Or(gt, with_cin), mir::Ty::Bool);
                self.convert(func, frame, bo, &fe::parse::Ty::Bool, fe_ty)
            }
            // `tzcnt`/`lzcnt` are `ctz`/`clz` with the zero case
            // defined as the width (BMI1's contract; the plain count
            // leaves it undefined).
            fe::check::Ia32Op::Tzcnt { width } | fe::check::Ia32Op::Lzcnt { width } => {
                let bits = u32::from(*width) * 8;
                let x = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let operand_mir = mir::Ty::int(bits);
                let zero = self.iconst(func, bits, 0);
                let is_zero = self.emit(
                    func,
                    frame,
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Eq,
                        lhs: x,
                        rhs: zero,
                    },
                    mir::Ty::Bool,
                );
                let count = self.emit(
                    func,
                    frame,
                    if matches!(op, fe::check::Ia32Op::Lzcnt { .. }) {
                        mir::InstKind::Clz(x)
                    } else {
                        mir::InstKind::Ctz(x)
                    },
                    operand_mir.clone(),
                );
                let width_case = self.iconst(func, bits, i128::from(bits));
                let r = self.emit(
                    func,
                    frame,
                    mir::InstKind::Select {
                        cond: is_zero,
                        then_val: width_case,
                        else_val: count,
                    },
                    operand_mir,
                );
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // `bextr(x, y)`: extract `y[15:8]` bits from bit
            // `y[7:0]`, zero-filled above. The start clamps to the
            // width, the length to what remains; a clamped-away case
            // leaves an empty mask (the garbage shift is `And`ed off).
            fe::check::Ia32Op::Bextr { width } => {
                let bits = u32::from(*width) * 8;
                let operand_mir = mir::Ty::int(bits);
                let x = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let y = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let byte_mask = self.iconst(func, bits, 0xff);
                let start = self.emit(func, frame, mir::InstKind::And(y, byte_mask), operand_mir.clone());
                let eight = self.iconst(func, bits, 8);
                let len = self.emit(func, frame, mir::InstKind::LShr(y, eight), operand_mir.clone());
                let len = self.emit(func, frame, mir::InstKind::And(len, byte_mask), operand_mir.clone());
                let width_const = self.iconst(func, bits, i128::from(bits));
                let start = self.select_min(func, frame, start, width_const, operand_mir.clone());
                let rem = self.iconst(func, bits, i128::from(bits));
                let rem = self.emit(func, frame, mir::InstKind::Sub(rem, start), operand_mir.clone());
                let len = self.select_min(func, frame, len, rem, operand_mir.clone());
                let one = self.iconst(func, bits, 1);
                let mask = self.emit(func, frame, mir::InstKind::Shl(one, len), operand_mir.clone());
                let one = self.iconst(func, bits, 1);
                let mask = self.emit(func, frame, mir::InstKind::Sub(mask, one), operand_mir.clone());
                let shifted = self.emit(func, frame, mir::InstKind::LShr(x, start), operand_mir.clone());
                let r = self.emit(func, frame, mir::InstKind::And(shifted, mask), operand_mir);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // `bzhi(x, y)`: zero from bit `y & 0xff` up. Below the
            // width the mask is `(1 << idx) - 1` (clamped so the
            // shift stays defined); at or beyond it, nothing zeroes.
            fe::check::Ia32Op::Bzhi { width } => {
                let bits = u32::from(*width) * 8;
                let operand_mir = mir::Ty::int(bits);
                let x = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let y = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let byte_mask = self.iconst(func, bits, 0xff);
                let idx = self.emit(func, frame, mir::InstKind::And(y, byte_mask), operand_mir.clone());
                let top = self.iconst(func, bits, i128::from(bits - 1));
                let clamped = self.select_min(func, frame, idx, top, operand_mir.clone());
                let one = self.iconst(func, bits, 1);
                let low_mask = self.emit(func, frame, mir::InstKind::Shl(one, clamped), operand_mir.clone());
                let one = self.iconst(func, bits, 1);
                let low_mask = self.emit(func, frame, mir::InstKind::Sub(low_mask, one), operand_mir.clone());
                let width_const = self.iconst(func, bits, i128::from(bits));
                let below = self.emit(
                    func,
                    frame,
                    mir::InstKind::Icmp {
                        pred: mir::IcmpPred::Ult,
                        lhs: idx,
                        rhs: width_const,
                    },
                    mir::Ty::Bool,
                );
                let ones = self.iconst(func, bits, (1i128 << bits) - 1);
                let mask = self.emit(
                    func,
                    frame,
                    mir::InstKind::Select {
                        cond: below,
                        then_val: low_mask,
                        else_val: ones,
                    },
                    operand_mir.clone(),
                );
                let r = self.emit(func, frame, mir::InstKind::And(x, mask), operand_mir);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // `pdep`/`pext` are single BMI2 instructions.
            fe::check::Ia32Op::Pdep { width } | fe::check::Ia32Op::Pext { width } => {
                let bits = u32::from(*width) * 8;
                let x = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let mask = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let operand_mir = lower_ty(&args[0].ty, self.sema, self.interner);
                let mnemonic = if matches!(op, fe::check::Ia32Op::Pdep { .. }) { "pdep" } else { "pext" };
                let slot = self.alloca(func, frame, operand_mir.clone());
                self.asm(
                    func,
                    frame,
                    &format!("{mnemonic} %2, %1, %0"),
                    1,
                    vec![
                        mir::AsmOperand {
                            value: slot,
                            constraint: "=r".into(),
                            ty: operand_mir.clone(),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: x,
                            constraint: "r".into(),
                            ty: operand_mir.clone(),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: mask,
                            constraint: "r".into(),
                            ty: operand_mir,
                            tie: None,
                            name: None,
                        },
                    ],
                    false,
                );
                let r = self.load(func, frame, slot, mir::Ty::int(bits), false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            fe::check::Ia32Op::Rdtsc => {
                let out = self.asm_result_in_rax(func, frame, "rdtsc", vec![], vec![]);
                self.convert(func, frame, out, &fe::parse::Ty::UnsignedLongLong, fe_ty)
            }
            fe::check::Ia32Op::Rdtscp => {
                let ptr = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let aux_slot = self.alloca(func, frame, mir::Ty::int(32));
                let out = self.asm_result_in_rax(
                    func,
                    frame,
                    "rdtscp",
                    vec![mir::AsmOperand {
                        value: aux_slot,
                        constraint: "=c".into(),
                        ty: mir::Ty::int(32),
                        tie: None,
                        name: None,
                    }],
                    vec![],
                );
                let aux = self.load(func, frame, aux_slot, mir::Ty::int(32), false);
                self.store(func, frame, ptr, aux, false);
                self.convert(func, frame, out, &fe::parse::Ty::UnsignedLongLong, fe_ty)
            }
            fe::check::Ia32Op::Rdpmc => {
                let counter = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let out = self.asm_result_in_rax(
                    func,
                    frame,
                    "rdpmc",
                    vec![],
                    vec![mir::AsmOperand {
                        value: counter,
                        constraint: "c".into(),
                        ty: lower_ty(&args[0].ty, self.sema, self.interner),
                        tie: None,
                        name: None,
                    }],
                );
                self.convert(func, frame, out, &fe::parse::Ty::UnsignedLongLong, fe_ty)
            }
            fe::check::Ia32Op::Xgetbv => {
                let xcr = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let out = self.asm_result_in_rax(
                    func,
                    frame,
                    "xgetbv",
                    vec![],
                    vec![mir::AsmOperand {
                        value: xcr,
                        constraint: "c".into(),
                        ty: lower_ty(&args[0].ty, self.sema, self.interner),
                        tie: None,
                        name: None,
                    }],
                );
                self.convert(func, frame, out, &fe::parse::Ty::UnsignedLongLong, fe_ty)
            }
            // `crc32 %1, %0` accumulates the operand into the checksum
            // register; the `+r` output round-trips through a slot
            // like the asm statement lowering.
            fe::check::Ia32Op::Crc32 { width } => {
                let crc_bits = if *width == 8 { 64 } else { 32 };
                let crc_slot = self.alloca(func, frame, mir::Ty::int(crc_bits));
                let crc = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                self.store(func, frame, crc_slot, crc, false);
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let mnemonic = match width {
                    1 => "crc32b",
                    2 => "crc32w",
                    4 => "crc32l",
                    _ => "crc32q",
                };
                self.asm(
                    func,
                    frame,
                    &format!("{mnemonic} %1, %0"),
                    1,
                    vec![
                        mir::AsmOperand {
                            value: crc_slot,
                            constraint: "+r".into(),
                            ty: mir::Ty::int(crc_bits),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: v,
                            constraint: "r".into(),
                            ty: lower_ty(&args[1].ty, self.sema, self.interner),
                            tie: Some(0),
                            name: None,
                        },
                    ],
                    false,
                );
                let crc = self.load(func, frame, crc_slot, mir::Ty::int(crc_bits), false);
                self.convert(func, frame, crc, &fe::parse::Ty::UnsignedInt, fe_ty)
            }
            // `clzero`: the cache line at the address in `rax`.
            fe::check::Ia32Op::Clzero => {
                let ptr = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                self.asm(
                    func,
                    frame,
                    "clzero",
                    0,
                    vec![mir::AsmOperand {
                        value: ptr,
                        constraint: "a".into(),
                        ty: mir::Ty::ptr(mir::Ty::Void),
                        tie: None,
                        name: None,
                    }],
                    true,
                );
                self.iconst(func, 32, 0)
            }
            // `movdiri`: the value goes to the memory operand.
            fe::check::Ia32Op::DirectStore { width } => {
                let ptr = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                self.asm(
                    func,
                    frame,
                    "movdiri %1, %0",
                    0,
                    vec![
                        mir::AsmOperand {
                            value: ptr,
                            constraint: "m".into(),
                            ty: mir::Ty::ptr(mir::Ty::Void),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: v,
                            constraint: "r".into(),
                            ty: mir::Ty::int(u32::from(*width) * 8),
                            tie: None,
                            name: None,
                        },
                    ],
                    true,
                );
                self.iconst(func, 32, 0)
            }
            fe::check::Ia32Op::Movdir64b => {
                let dst = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let src = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                self.asm(
                    func,
                    frame,
                    "movdir64b %1, %0",
                    0,
                    vec![
                        mir::AsmOperand {
                            value: dst,
                            constraint: "m".into(),
                            ty: mir::Ty::ptr(mir::Ty::Void),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: src,
                            constraint: "r".into(),
                            ty: mir::Ty::ptr(mir::Ty::Void),
                            tie: None,
                            name: None,
                        },
                    ],
                    true,
                );
                self.iconst(func, 32, 0)
            }
            // The extended-state save/restore family passes the state
            // bitmap's low and high halves in `eax`/`edx`.
            fe::check::Ia32Op::XSave { mnemonic } => {
                let ptr = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let mask = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let mask_lo = self.half(func, frame, mask, false);
                let mask_hi = self.half(func, frame, mask, true);
                self.asm(
                    func,
                    frame,
                    &format!("{mnemonic} (%0)"),
                    0,
                    vec![
                        mir::AsmOperand {
                            value: ptr,
                            constraint: "m".into(),
                            ty: mir::Ty::ptr(mir::Ty::Void),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: mask_lo,
                            constraint: "a".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: mask_hi,
                            constraint: "d".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                    ],
                    true,
                );
                self.iconst(func, 32, 0)
            }
            fe::check::Ia32Op::XSetBv => {
                let xcr = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let v_lo = self.half(func, frame, v, false);
                let v_hi = self.half(func, frame, v, true);
                self.asm(
                    func,
                    frame,
                    "xsetbv",
                    0,
                    vec![
                        mir::AsmOperand {
                            value: xcr,
                            constraint: "c".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: v_lo,
                            constraint: "a".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: v_hi,
                            constraint: "d".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                    ],
                    true,
                );
                self.iconst(func, 32, 0)
            }
            // `xbegin` starts an RTM transaction; the fallback target
            // is the next instruction, so both the started (`eax` =
            // `-1`) and aborted paths continue here with the status in
            // `eax`.
            fe::check::Ia32Op::Xbegin => {
                let out = self.asm_result_in_rax(func, frame, "xbegin 1f\n1:", vec![], vec![]);
                self.convert(func, frame, out, &fe::parse::Ty::UnsignedInt, fe_ty)
            }
            // `testui` reports the user-interrupt flag through `eax`.
            fe::check::Ia32Op::Testui => {
                let out = self.asm_result_in_rax(func, frame, "testui", vec![], vec![]);
                self.convert(func, frame, out, &fe::parse::Ty::UnsignedChar, fe_ty)
            }
            fe::check::Ia32Op::Senduipi => {
                let target = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                self.asm(
                    func,
                    frame,
                    "senduipi %0",
                    0,
                    vec![mir::AsmOperand {
                        value: target,
                        constraint: "r".into(),
                        ty: mir::Ty::int(64),
                        tie: None,
                        name: None,
                    }],
                    true,
                );
                self.iconst(func, 32, 0)
            }
            fe::check::Ia32Op::Xtest => {
                let out = self.asm_result_in_rax(func, frame, "xtest", vec![], vec![]);
                self.convert(func, frame, out, &fe::parse::Ty::Int, fe_ty)
            }
            fe::check::Ia32Op::Rdpid => {
                let out = self.asm_result_in_rax(func, frame, "rdpid", vec![], vec![]);
                self.convert(func, frame, out, &fe::parse::Ty::UnsignedInt, fe_ty)
            }
            // The FSGSBASE accesses move the segment base through one
            // register operand.
            fe::check::Ia32Op::FsBase { write, width } => {
                let bits = u32::from(*width) * 8;
                let mnemonic = match (write, *width) {
                    (false, 4) => "rdfsbase",
                    (false, _) => "rdfsbaseq",
                    (true, 4) => "wrfsbase",
                    (true, _) => "wrfsbaseq",
                };
                if !*write {
                    let slot = self.alloca(func, frame, mir::Ty::int(bits));
                    self.asm(
                        func,
                        frame,
                        &format!("{mnemonic} %0"),
                        1,
                        vec![mir::AsmOperand {
                            value: slot,
                            constraint: "=r".into(),
                            ty: mir::Ty::int(bits),
                            tie: None,
                            name: None,
                        }],
                        true,
                    );
                    let out = self.load(func, frame, slot, mir::Ty::int(bits), false);
                    return self.convert(func, frame, out, &self.args_ty_of_width(*width), fe_ty);
                }
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                self.asm(
                    func,
                    frame,
                    &format!("{mnemonic} %0"),
                    0,
                    vec![mir::AsmOperand {
                        value: v,
                        constraint: "r".into(),
                        ty: mir::Ty::int(bits),
                        tie: None,
                        name: None,
                    }],
                    true,
                );
                self.iconst(func, 32, 0)
            }
            fe::check::Ia32Op::Hreset => {
                let select = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                self.asm(
                    func,
                    frame,
                    "hreset %0",
                    0,
                    vec![mir::AsmOperand {
                        value: select,
                        constraint: "e".into(),
                        ty: mir::Ty::int(32),
                        tie: None,
                        name: None,
                    }],
                    true,
                );
                self.iconst(func, 32, 0)
            }
            // `urdmsr` puts the MSR value in `edx:eax`; the two halves
            // are reassembled at full width.
            fe::check::Ia32Op::URdMsr => {
                let msr = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let lo_slot = self.alloca(func, frame, mir::Ty::int(32));
                let hi_slot = self.alloca(func, frame, mir::Ty::int(32));
                self.asm(
                    func,
                    frame,
                    "urdmsr",
                    2,
                    vec![
                        mir::AsmOperand {
                            value: lo_slot,
                            constraint: "=a".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: hi_slot,
                            constraint: "=d".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: msr,
                            constraint: "c".into(),
                            ty: mir::Ty::int(64),
                            tie: None,
                            name: None,
                        },
                    ],
                    true,
                );
                let lo = self.load(func, frame, lo_slot, mir::Ty::int(32), false);
                let hi = self.load(func, frame, hi_slot, mir::Ty::int(32), false);
                let lo = self.emit(func, frame, mir::InstKind::Zext(lo), mir::Ty::int(64));
                let hi = self.emit(func, frame, mir::InstKind::Zext(hi), mir::Ty::int(64));
                let shift = self.iconst(func, 64, 32);
                let hi = self.emit(func, frame, mir::InstKind::Shl(hi, shift), mir::Ty::int(64));
                let r = self.emit(func, frame, mir::InstKind::Or(lo, hi), mir::Ty::int(64));
                self.convert(func, frame, r, &fe::parse::Ty::UnsignedLongLong, fe_ty)
            }
            fe::check::Ia32Op::UWrMsr => {
                let msr = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let value = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let value_lo = self.half(func, frame, value, false);
                let value_hi = self.half(func, frame, value, true);
                self.asm(
                    func,
                    frame,
                    "uwrmsr",
                    0,
                    vec![
                        mir::AsmOperand {
                            value: value_lo,
                            constraint: "a".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: value_hi,
                            constraint: "d".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: msr,
                            constraint: "c".into(),
                            ty: mir::Ty::int(64),
                            tie: None,
                            name: None,
                        },
                    ],
                    true,
                );
                self.iconst(func, 32, 0)
            }
            fe::check::Ia32Op::PtWrite { width } => {
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                self.asm(
                    func,
                    frame,
                    "ptwrite %0",
                    0,
                    vec![mir::AsmOperand {
                        value: v,
                        constraint: "r".into(),
                        ty: mir::Ty::int(u32::from(*width) * 8),
                        tie: None,
                        name: None,
                    }],
                    true,
                );
                self.iconst(func, 32, 0)
            }
            // The MMX pairs stage both operands into `mm` registers
            // through 32-bit moves (the encoder has no 64-bit GPR↔mm
            // move): halves in via `%k`, halves out through `eax`/`edx`.
            // `emms` would leave the FPU state clean but callers issue
            // it themselves.
            fe::check::Ia32Op::MmxOp { mnemonic } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let (a_lo, a_hi) = (self.half(func, frame, a, false), self.half(func, frame, a, true));
                let (b_lo, b_hi) = (self.half(func, frame, b, false), self.half(func, frame, b, true));
                let lo_slot = self.alloca(func, frame, mir::Ty::int(32));
                let hi_slot = self.alloca(func, frame, mir::Ty::int(32));
                self.asm_with_clobbers(
                    func,
                    frame,
                    &format!("movd %k2, %%mm0\n\tmovd %k3, %%mm1\n\tpunpckldq %%mm1, %%mm0\n\tmovd %k4, %%mm2\n\tmovd %k5, %%mm3\n\tpunpckldq %%mm3, %%mm2\n\t{mnemonic} %%mm2, %%mm0\n\tmovd %%mm0, %k0\n\tpunpckhdq %%mm0, %%mm0\n\tmovd %%mm0, %k1"),
                    2,
                    vec![
                        mir::AsmOperand {
                            value: lo_slot,
                            constraint: "=a".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: hi_slot,
                            constraint: "=d".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: a_lo,
                            constraint: "r".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: a_hi,
                            constraint: "r".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: b_lo,
                            constraint: "r".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: b_hi,
                            constraint: "r".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                    ],
                    &[],
                    true,
                );
                let lo = self.load(func, frame, lo_slot, mir::Ty::int(32), false);
                let hi = self.load(func, frame, hi_slot, mir::Ty::int(32), false);
                let out = self.join_halves(func, frame, lo, hi);
                self.convert(func, frame, out, &args[0].ty, fe_ty)
            }
            // The count forms shift exactly like the binary pairs; the
            // immediate forms take the count as an `"e"` operand. The
            // vector halves stage in and out through 32-bit moves, as
            // in [`Ia32Op::MmxOp`].
            fe::check::Ia32Op::MmxShift { mnemonic, imm } => {
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let count = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let (v_lo, v_hi) = (self.half(func, frame, v, false), self.half(func, frame, v, true));
                let (count_lo, count_hi) = (self.half(func, frame, count, false), self.half(func, frame, count, true));
                let lo_slot = self.alloca(func, frame, mir::Ty::int(32));
                let hi_slot = self.alloca(func, frame, mir::Ty::int(32));
                let stage_in = "movd %k2, %%mm0\n\tmovd %k3, %%mm1\n\tpunpckldq %%mm1, %%mm0";
                let stage_out = "movd %%mm0, %k0\n\tpunpckhdq %%mm0, %%mm0\n\tmovd %%mm0, %k1";
                let (template, count_operands) = if *imm {
                    (format!("{stage_in}\n\t{mnemonic} %4, %%mm0\n\t{stage_out}"), vec![mir::AsmOperand {
                        value: count,
                        constraint: "e".into(),
                        ty: mir::Ty::int(64),
                        tie: None,
                        name: None,
                    }])
                } else {
                    (
                        format!("{stage_in}\n\tmovd %k4, %%mm2\n\tmovd %k5, %%mm3\n\tpunpckldq %%mm3, %%mm2\n\t{mnemonic} %%mm2, %%mm0\n\t{stage_out}"),
                        vec![
                            mir::AsmOperand {
                                value: count_lo,
                                constraint: "r".into(),
                                ty: mir::Ty::int(32),
                                tie: None,
                                name: None,
                            },
                            mir::AsmOperand {
                                value: count_hi,
                                constraint: "r".into(),
                                ty: mir::Ty::int(32),
                                tie: None,
                                name: None,
                            },
                        ],
                    )
                };
                self.asm_with_clobbers(
                    func,
                    frame,
                    &template,
                    2,
                    vec![
                        mir::AsmOperand {
                            value: lo_slot,
                            constraint: "=a".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: hi_slot,
                            constraint: "=d".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: v_lo,
                            constraint: "r".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: v_hi,
                            constraint: "r".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                    ]
                    .into_iter()
                    .chain(count_operands)
                    .collect(),
                    &[],
                    true,
                );
                let lo = self.load(func, frame, lo_slot, mir::Ty::int(32), false);
                let hi = self.load(func, frame, hi_slot, mir::Ty::int(32), false);
                let out = self.join_halves(func, frame, lo, hi);
                self.convert(func, frame, out, &args[0].ty, fe_ty)
            }
            // Lane composition over the 8-byte integer: each lane
            // masked to its width, widened, and shifted into place.
            fe::check::Ia32Op::VecInit { lanes } => {
                let lane_bits = u32::from(*lanes) * 8;
                let mut r = None;
                for (i, arg) in args.iter().enumerate() {
                    let lane = self.expr_value(module, func, frame, funcs, externs, globals, arg);
                    let mask = self.iconst(func, 64, (1i128 << lane_bits) - 1);
                    let lane = self.emit(func, frame, mir::InstKind::And(lane, mask), mir::Ty::int(64));
                    if i > 0 {
                        let shift = self.iconst(func, 64, i128::from(lane_bits) * i as i128);
                        let lane = self.emit(func, frame, mir::InstKind::Shl(lane, shift), mir::Ty::int(64));
                        let prev = r.take().unwrap();
                        r = Some(self.emit(func, frame, mir::InstKind::Or(prev, lane), mir::Ty::int(64)));
                    } else {
                        r = Some(lane);
                    }
                }
                let r = r.unwrap();
                self.convert(func, frame, r, &fe::parse::Ty::LongLong, fe_ty)
            }
            // `[a, b]`: stage `a` low, `b` high with the doubleword
            // interleave.
            fe::check::Ia32Op::VecInitV2Si => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                // {a, b} as dwords: pack the 64-bit image (the stub's
                // movd/punpckldq pair, without the asm) and bitcast.
                let out = self.join_halves(func, frame, a, b);
                self.convert(func, frame, out, &fe::parse::Ty::LongLong, fe_ty)
            }
            // Lane `idx` of an 8-byte vector: the low or high dword.
            fe::check::Ia32Op::VecExtV2Si => {
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                match fe::eval::eval_const(&args[1], self.sema, self.interner) {
                    Some(fe::eval::Value::Int(0)) => {
                        let r = self.emit(func, frame, mir::InstKind::Trunc(v), mir::Ty::int(32));
                        self.convert(func, frame, r, &fe::parse::Ty::Int, fe_ty)
                    }
                    Some(fe::eval::Value::Int(1)) => {
                        let shift = self.iconst(func, 64, 32);
                        let hi = self.emit(func, frame, mir::InstKind::LShr(v, shift), mir::Ty::int(64));
                        let r = self.emit(func, frame, mir::InstKind::Trunc(hi), mir::Ty::int(32));
                        self.convert(func, frame, r, &fe::parse::Ty::Int, fe_ty)
                    }
                    _ => panic!("vec_ext_v2si index must be a constant 0 or 1"),
                }
            }
            // SSE1 scalar tails: lane 0 computes, lanes 1-3 pass
            // through from the first operand.
            fe::check::Ia32Op::SsBin { op } => {
                use fe::check::SsOp;
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let f32 = mir::Ty::Float(mir::FloatKind::F32);
                let a0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: 0 }, f32.clone());
                let b0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: b, lane: 0 }, f32.clone());
                let r0 = match op {
                    SsOp::Add => self.emit(func, frame, mir::InstKind::FAdd(a0, b0), f32.clone()),
                    SsOp::Sub => self.emit(func, frame, mir::InstKind::FSub(a0, b0), f32.clone()),
                    SsOp::Mul => self.emit(func, frame, mir::InstKind::FMul(a0, b0), f32.clone()),
                    SsOp::Div => self.emit(func, frame, mir::InstKind::FDiv(a0, b0), f32.clone()),
                    // minss/maxss return the second operand when the
                    // first is NaN; the ordered compare falls through
                    // to `b` exactly there.
                    SsOp::Min | SsOp::Max => {
                        let pred = if matches!(op, SsOp::Min) { mir::FcmpPred::Lt } else { mir::FcmpPred::Gt };
                        let test = self.emit(func, frame, mir::InstKind::Fcmp { pred, lhs: a0, rhs: b0 }, mir::Ty::Bool);
                        self.emit(func, frame, mir::InstKind::Select { cond: test, then_val: a0, else_val: b0 }, f32)
                    }
                };
                let r = self.vec_replace_lane0(func, frame, a, r0, fe_ty);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            fe::check::Ia32Op::SqrtPs => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let f32 = mir::Ty::Float(mir::FloatKind::F32);
                let slot = self.alloca(func, frame, lower_ty(fe_ty, self.sema, self.interner));
                for lane in 0..4u8 {
                    let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane }, f32.clone());
                    let s = self.emit(func, frame, mir::InstKind::Sqrt(x), f32.clone());
                    let idx = self.iconst(func, 64, i128::from(lane));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, 4, f32.clone()).0;
                    self.store(func, frame, addr, s, false);
                }
                self.load(func, frame, slot, lower_ty(fe_ty, self.sema, self.interner), false)
            }
            fe::check::Ia32Op::SqrtSs => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let f32 = mir::Ty::Float(mir::FloatKind::F32);
                let a0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: 0 }, f32.clone());
                let r0 = self.emit(func, frame, mir::InstKind::Sqrt(a0), f32);
                let r = self.vec_replace_lane0(func, frame, a, r0, fe_ty);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            fe::check::Ia32Op::RcpSs { rsqrt } => {
                // The approximate instructions have no MIR opcode: one
                // asm stub applies the instruction to the vector
                // registers directly (lane semantics come from the
                // instruction: lane 0 for the `ss` forms).
                let mnemonic = if *rsqrt { "rsqrtss" } else { "rcpss" };
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let vec_ty = lower_ty(&args[0].ty, self.sema, self.interner);
                let out_slot = self.alloca(func, frame, vec_ty.clone());
                self.asm_with_clobbers(
                    func,
                    frame,
                    &format!("{mnemonic} %1, %0"),
                    1,
                    vec![
                        mir::AsmOperand {
                            value: out_slot,
                            constraint: "=x".into(),
                            ty: vec_ty.clone(),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: a,
                            constraint: "x".into(),
                            ty: vec_ty.clone(),
                            tie: None,
                            name: None,
                        },
                    ],
                    &[],
                    true,
                );
                let r = self.load(func, frame, out_slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // SSE1 packed arithmetic and min/max: shape-generic over
            // `__v4sf`.
            fe::check::Ia32Op::PsBin { op } => {
                use fe::check::SsOp;
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let f32 = mir::Ty::Float(mir::FloatKind::F32);
                let v4f = mir::Ty::Vector {
                    elem: Box::new(f32.clone()),
                    lanes: 4,
                };
                let kind = match op {
                    SsOp::Add => mir::InstKind::FAdd(a, b),
                    SsOp::Sub => mir::InstKind::FSub(a, b),
                    SsOp::Mul => mir::InstKind::FMul(a, b),
                    SsOp::Div => mir::InstKind::FDiv(a, b),
                    // minps/maxps return the second operand when the
                    // first is NaN; the ordered compare falls through
                    // to `b` exactly there.
                    SsOp::Min | SsOp::Max => {
                        let pred = if matches!(op, SsOp::Min) { mir::FcmpPred::Lt } else { mir::FcmpPred::Gt };
                        let i32x4 = mir::Ty::Vector {
                            elem: Box::new(mir::Ty::int(32)),
                            lanes: 4,
                        };
                        let test = self.emit(func, frame, mir::InstKind::Fcmp { pred, lhs: a, rhs: b }, i32x4);
                        return self.emit(func, frame, mir::InstKind::Select { cond: test, then_val: a, else_val: b }, v4f);
                    }
                };
                self.emit(func, frame, kind, v4f)
            }
            // SSE1 bitwise pairs: integer ops over the 128 bits, the
            // float lane type bitcast away and back (the integer image
            // follows the lane width: `i64x2` for `__v2df`).
            fe::check::Ia32Op::PsBitwise { op } => {
                use fe::check::PsBitOp;
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let shape = self.vector_shape(&args[0].ty);
                let elem_bits = (shape.bytes * 8 / u64::from(shape.lanes)) as u32;
                let int_ty = mir::Ty::Vector {
                    elem: Box::new(mir::Ty::int(elem_bits)),
                    lanes: shape.lanes,
                };
                let ai = self.emit(func, frame, mir::InstKind::Bitcast(a), int_ty.clone());
                let bi = self.emit(func, frame, mir::InstKind::Bitcast(b), int_ty.clone());
                let kind = match op {
                    PsBitOp::And => mir::InstKind::And(ai, bi),
                    PsBitOp::AndN => {
                        let not_a = self.emit(func, frame, mir::InstKind::Not(ai), int_ty.clone());
                        mir::InstKind::And(not_a, bi)
                    }
                    PsBitOp::Or => mir::InstKind::Or(ai, bi),
                    PsBitOp::Xor => mir::InstKind::Xor(ai, bi),
                };
                let r = self.emit(func, frame, kind, int_ty);
                self.emit(func, frame, mir::InstKind::Bitcast(r), lower_ty(fe_ty, self.sema, self.interner))
            }
            // The approximate packed reciprocals: one asm stub applies
            // the instruction to the vector registers directly (all
            // lanes).
            fe::check::Ia32Op::RcpPs { rsqrt } => {
                let mnemonic = if *rsqrt { "rsqrtps" } else { "rcpps" };
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let vec_ty = lower_ty(&args[0].ty, self.sema, self.interner);
                let out_slot = self.alloca(func, frame, vec_ty.clone());
                self.asm_with_clobbers(
                    func,
                    frame,
                    &format!("{mnemonic} %1, %0"),
                    1,
                    vec![
                        mir::AsmOperand {
                            value: out_slot,
                            constraint: "=x".into(),
                            ty: vec_ty.clone(),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: a,
                            constraint: "x".into(),
                            ty: vec_ty.clone(),
                            tie: None,
                            name: None,
                        },
                    ],
                    &[],
                    true,
                );
                let r = self.load(func, frame, out_slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // The SSE compare family: a full-width mask (`ps`) or lane
            // 0 over the first operand's remaining lanes (`ss`).
            fe::check::Ia32Op::Cmp { ps, pred } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let i32x4 = mir::Ty::Vector {
                    elem: Box::new(mir::Ty::int(32)),
                    lanes: 4,
                };
                let mut fcmp = |pred: mir::FcmpPred| self.emit(func, frame, mir::InstKind::Fcmp { pred, lhs: a, rhs: b }, i32x4.clone());
                let mask = match pred {
                    fe::check::CmpPred::Eq => fcmp(mir::FcmpPred::Eq),
                    fe::check::CmpPred::Lt => fcmp(mir::FcmpPred::Lt),
                    fe::check::CmpPred::Le => fcmp(mir::FcmpPred::Le),
                    fe::check::CmpPred::Gt => fcmp(mir::FcmpPred::Gt),
                    fe::check::CmpPred::Ge => fcmp(mir::FcmpPred::Ge),
                    fe::check::CmpPred::Neq => fcmp(mir::FcmpPred::Ne),
                    // The `n*` forms are the complement of their base:
                    // unordered-or-the-swapped-compare, ORed over the
                    // lane masks.
                    fe::check::CmpPred::Nlt => {
                        let ge = fcmp(mir::FcmpPred::Ge);
                        let uno = fcmp(mir::FcmpPred::Uno);
                        self.emit(func, frame, mir::InstKind::Or(ge, uno), i32x4.clone())
                    }
                    fe::check::CmpPred::Nle => {
                        let gt = fcmp(mir::FcmpPred::Gt);
                        let uno = fcmp(mir::FcmpPred::Uno);
                        self.emit(func, frame, mir::InstKind::Or(gt, uno), i32x4.clone())
                    }
                    fe::check::CmpPred::Ngt => {
                        let le = fcmp(mir::FcmpPred::Le);
                        let uno = fcmp(mir::FcmpPred::Uno);
                        self.emit(func, frame, mir::InstKind::Or(le, uno), i32x4.clone())
                    }
                    fe::check::CmpPred::Nge => {
                        let lt = fcmp(mir::FcmpPred::Lt);
                        let uno = fcmp(mir::FcmpPred::Uno);
                        self.emit(func, frame, mir::InstKind::Or(lt, uno), i32x4.clone())
                    }
                    fe::check::CmpPred::Ord => fcmp(mir::FcmpPred::Ord),
                    fe::check::CmpPred::Unord => fcmp(mir::FcmpPred::Uno),
                };
                if *ps {
                    let v4si = fe::parse::Ty::Vector {
                        elem: Box::new(fe::parse::Ty::Int),
                        len: 4,
                    };
                    return self.convert(func, frame, mask, &v4si, fe_ty);
                }
                let m0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: mask, lane: 0 }, mir::Ty::int(32));
                let r0 = self.emit(func, frame, mir::InstKind::Bitcast(m0), mir::Ty::Float(mir::FloatKind::F32));
                let r = self.vec_replace_lane0(func, frame, a, r0, fe_ty);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // The COMISS flag tests on lane 0, as 0/1. The instruction
            // sets ZF=CF=PF=1 when unordered, so: eq/lt/le are their
            // ordered form OR unordered (ZF/CF read), gt/ge are the
            // ordered forms alone (they read the complement flags),
            // and neq is `!ZF` — an *ordered* not-equal.
            fe::check::Ia32Op::Comi { pred, .. } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let f32 = mir::Ty::Float(mir::FloatKind::F32);
                let a0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: 0 }, f32.clone());
                let b0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: b, lane: 0 }, f32);
                let mut fcmp = |pred: mir::FcmpPred| self.emit(func, frame, mir::InstKind::Fcmp { pred, lhs: a0, rhs: b0 }, mir::Ty::Bool);
                let test = match pred {
                    fe::check::ComiPred::Eq => {
                        let eq = fcmp(mir::FcmpPred::Eq);
                        let uno = fcmp(mir::FcmpPred::Uno);
                        self.emit(func, frame, mir::InstKind::Or(eq, uno), mir::Ty::Bool)
                    }
                    fe::check::ComiPred::Lt => {
                        let lt = fcmp(mir::FcmpPred::Lt);
                        let uno = fcmp(mir::FcmpPred::Uno);
                        self.emit(func, frame, mir::InstKind::Or(lt, uno), mir::Ty::Bool)
                    }
                    fe::check::ComiPred::Le => {
                        let le = fcmp(mir::FcmpPred::Le);
                        let uno = fcmp(mir::FcmpPred::Uno);
                        self.emit(func, frame, mir::InstKind::Or(le, uno), mir::Ty::Bool)
                    }
                    fe::check::ComiPred::Gt => fcmp(mir::FcmpPred::Gt),
                    fe::check::ComiPred::Ge => fcmp(mir::FcmpPred::Ge),
                    fe::check::ComiPred::Neq => {
                        let ne = fcmp(mir::FcmpPred::Ne);
                        let ord = fcmp(mir::FcmpPred::Ord);
                        self.emit(func, frame, mir::InstKind::And(ne, ord), mir::Ty::Bool)
                    }
                };
                self.emit(func, frame, mir::InstKind::Zext(test), mir::Ty::int(32))
            }
            fe::check::Ia32Op::MovSs => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let f32 = mir::Ty::Float(mir::FloatKind::F32);
                let b0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: b, lane: 0 }, f32);
                let r = self.vec_replace_lane0(func, frame, a, b0, fe_ty);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // An integer into lane 0, upper lanes preserved: stage the
            // vector through `%xmm0`, convert in place, store back.
            fe::check::Ia32Op::CvtSi2Ss { width } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let vec_ty = lower_ty(&args[0].ty, self.sema, self.interner);
                let a_slot = self.alloca(func, frame, vec_ty.clone());
                self.store(func, frame, a_slot, a, false);
                let out_slot = self.alloca(func, frame, mir::Ty::int(128));
                self.asm_with_clobbers(
                    func,
                    frame,
                    "movups %2, %%xmm0\n\tcvtsi2ss %1, %%xmm0\n\tmovups %%xmm0, %0",
                    1,
                    vec![
                        mir::AsmOperand {
                            value: out_slot,
                            constraint: "=m".into(),
                            ty: mir::Ty::int(128),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: v,
                            constraint: "r".into(),
                            ty: mir::Ty::int((*width * 8) as u32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: a_slot,
                            constraint: "m".into(),
                            ty: vec_ty,
                            tie: None,
                            name: None,
                        },
                    ],
                    &["xmm0"],
                    true,
                );
                let r = self.load(func, frame, out_slot, lower_ty(&args[0].ty, self.sema, self.interner), false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // Lane 0 to an integer. Truncating forms ride MIR's
            // float-to-int; the round-to-nearest forms need the
            // instruction (`cvtss2si` honors the rounding mode MIR
            // cannot express).
            fe::check::Ia32Op::CvtSs2Si { width, truncate } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let bits = (*width * 8) as u32;
                let int_ty = mir::Ty::int(bits);
                if *truncate {
                    let f32 = mir::Ty::Float(mir::FloatKind::F32);
                    let a0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: 0 }, f32);
                    let r = self.emit(func, frame, mir::InstKind::SFtoI(a0), int_ty);
                    return self.convert(func, frame, r, &fe::parse::Ty::LongLong, fe_ty);
                }
                let reg = if *width == 4 { "eax" } else { "rax" };
                let a_slot = self.alloca(func, frame, lower_ty(&args[0].ty, self.sema, self.interner));
                self.store(func, frame, a_slot, a, false);
                let out = self.asm_result_in_rax(
                    func,
                    frame,
                    &format!("cvtss2si %1, %%{reg}"),
                    vec![],
                    vec![mir::AsmOperand {
                        value: a_slot,
                        constraint: "m".into(),
                        ty: mir::Ty::int(128),
                        tie: None,
                        name: None,
                    }],
                );
                // `eax` writes zero-extend into the rax slot; a 64-bit
                // result already fills it.
                self.convert(func, frame, out, &fe::parse::Ty::LongLong, fe_ty)
            }
            // Lanes 0-1 to the two 32-bit lanes of an 8-byte vector.
            fe::check::Ia32Op::CvtPs2Pi { truncate } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let f32 = mir::Ty::Float(mir::FloatKind::F32);
                let mmx_vec = fe::parse::Ty::Vector {
                    elem: Box::new(fe::parse::Ty::Int),
                    len: 2,
                };
                let image_ty = mir::Ty::int(64);
                if *truncate {
                    let x0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: 0 }, f32.clone());
                    let x1 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: 1 }, f32);
                    let i0 = self.emit(func, frame, mir::InstKind::SFtoI(x0), mir::Ty::int(32));
                    let i1 = self.emit(func, frame, mir::InstKind::SFtoI(x1), mir::Ty::int(32));
                    let wide0 = self.emit(func, frame, mir::InstKind::Zext(i0), image_ty.clone());
                    let shift = self.iconst(func, 64, 32);
                    let wide1 = self.emit(func, frame, mir::InstKind::Zext(i1), image_ty.clone());
                    let hi = self.emit(func, frame, mir::InstKind::Shl(wide1, shift), image_ty.clone());
                    let r = self.emit(func, frame, mir::InstKind::Or(wide0, hi), image_ty);
                    return self.convert(func, frame, r, &fe::parse::Ty::LongLong, fe_ty);
                }
                let a_slot = self.alloca(func, frame, lower_ty(&args[0].ty, self.sema, self.interner));
                self.store(func, frame, a_slot, a, false);
                let lo_slot = self.alloca(func, frame, mir::Ty::int(32));
                let hi_slot = self.alloca(func, frame, mir::Ty::int(32));
                self.asm_with_clobbers(
                    func,
                    frame,
                    "movups %2, %%xmm0\n\tcvtps2pi %%xmm0, %%mm0\n\tmovd %%mm0, %k0\n\tpunpckhdq %%mm0, %%mm0\n\tmovd %%mm0, %k1",
                    2,
                    vec![
                        mir::AsmOperand {
                            value: lo_slot,
                            constraint: "=a".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: hi_slot,
                            constraint: "=d".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: a_slot,
                            constraint: "m".into(),
                            ty: mir::Ty::int(128),
                            tie: None,
                            name: None,
                        },
                    ],
                    &["xmm0"],
                    true,
                );
                let lo = self.load(func, frame, lo_slot, mir::Ty::int(32), false);
                let hi = self.load(func, frame, hi_slot, mir::Ty::int(32), false);
                let out = self.join_halves(func, frame, lo, hi);
                self.convert(func, frame, out, &fe::parse::Ty::LongLong, &mmx_vec)
            }
            // `__m128` from an MMX pair of ints (lanes 0-1 float) over
            // the first operand's lanes 2-3.
            fe::check::Ia32Op::CvtPi2Ps => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let m = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let f32 = mir::Ty::Float(mir::FloatKind::F32);
                let i32t = mir::Ty::int(32);
                let i0 = self.emit(func, frame, mir::InstKind::Trunc(m), i32t.clone());
                let shift = self.iconst(func, 64, 32);
                let hi = self.emit(func, frame, mir::InstKind::LShr(m, shift), mir::Ty::int(64));
                let i1 = self.emit(func, frame, mir::InstKind::Trunc(hi), i32t);
                let f0 = self.emit(func, frame, mir::InstKind::SItoF(i0), f32.clone());
                let f1 = self.emit(func, frame, mir::InstKind::SItoF(i1), f32.clone());
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                self.store(func, frame, slot, a, false);
                for (lane, val) in [(0u8, f0), (1, f1)] {
                    let idx = self.iconst(func, 64, i128::from(lane));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, 4, f32.clone()).0;
                    self.store(func, frame, addr, val, false);
                }
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // Low half from the first operand, high half from the
            // second.
            fe::check::Ia32Op::MovLhPs => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let f32 = mir::Ty::Float(mir::FloatKind::F32);
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                for lane in 0..4u8 {
                    let src = if lane < 2 { a } else { b };
                    let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: src, lane: lane % 2 }, f32.clone());
                    let idx = self.iconst(func, 64, i128::from(lane));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, 4, f32.clone()).0;
                    self.store(func, frame, addr, x, false);
                }
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // High half of the second operand into the low half, the
            // first operand's high half through.
            fe::check::Ia32Op::MovHlPs => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let f32 = mir::Ty::Float(mir::FloatKind::F32);
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                for lane in 0..4u8 {
                    let src = if lane < 2 { b } else { a };
                    let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: src, lane: lane + 2 }, f32.clone());
                    let idx = self.iconst(func, 64, i128::from(lane));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, 4, f32.clone()).0;
                    self.store(func, frame, addr, x, false);
                }
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // The float interleave: `unpcklps` pairs each operand's low
            // half (a0 b0 a1 b1), `unpckhps` the high (a2 b2 a3 b3).
            fe::check::Ia32Op::UnpckPs { high } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let f32 = mir::Ty::Float(mir::FloatKind::F32);
                let base = if *high { 2 } else { 0 };
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                for (lane, (src, src_lane)) in [(a, base), (b, base), (a, base + 1), (b, base + 1)].iter().enumerate() {
                    let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: *src, lane: *src_lane }, f32.clone());
                    let idx = self.iconst(func, 64, lane as i128);
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, 4, f32.clone()).0;
                    self.store(func, frame, addr, x, false);
                }
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // Two floats from memory into one half, the other half
            // through from the first operand.
            fe::check::Ia32Op::LoadHalfPs { high } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let p = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let f32 = mir::Ty::Float(mir::FloatKind::F32);
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                self.store(func, frame, slot, a, false);
                for k in 0..2u8 {
                    let lane = if *high { k + 2 } else { k };
                    let x = self.load(func, frame, p, f32.clone(), false);
                    let _ = x;
                    let idx = self.iconst(func, 64, i128::from(k));
                    let elem_addr = self.scaled_index_addr(func, frame, p, idx, &fe::parse::Ty::LongLong, 4, f32.clone()).0;
                    let val = self.load(func, frame, elem_addr, f32.clone(), false);
                    let dst = self.iconst(func, 64, i128::from(lane));
                    let addr = self.scaled_index_addr(func, frame, slot, dst, &fe::parse::Ty::LongLong, 4, f32.clone()).0;
                    self.store(func, frame, addr, val, false);
                }
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // One half of the vector to memory.
            fe::check::Ia32Op::StoreHalfPs { high } => {
                let p = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let f32 = mir::Ty::Float(mir::FloatKind::F32);
                for k in 0..2u8 {
                    let lane = if *high { k + 2 } else { k };
                    let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane }, f32.clone());
                    let idx = self.iconst(func, 64, i128::from(k));
                    let elem_addr = self.scaled_index_addr(func, frame, p, idx, &fe::parse::Ty::LongLong, 4, f32.clone()).0;
                    self.store(func, frame, elem_addr, x, false);
                }
                let undef = func.values.push(mir::ValueData {
                    ty: mir::Ty::Void,
                    def: mir::Def::Const(mir::Const::Undef),
                });
                undef
            }
            // `shufps`: each result lane is a two-bit selector into
            // `a` (low half) or `b` (high half).
            fe::check::Ia32Op::ShufPs => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let imm = match fe::eval::eval_const(&args[2], self.sema, self.interner) {
                    Some(fe::eval::Value::Int(n)) => n as u64,
                    _ => panic!("shufps selector must be a constant"),
                };
                let f32 = mir::Ty::Float(mir::FloatKind::F32);
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                for lane in 0..4u8 {
                    let sel = ((imm >> (lane * 2)) & 3) as u8;
                    let (src, src_lane) = if lane < 2 { (a, sel) } else { (b, sel) };
                    let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: src, lane: src_lane }, f32.clone());
                    let idx = self.iconst(func, 64, i128::from(lane));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, 4, f32.clone()).0;
                    self.store(func, frame, addr, x, false);
                }
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // SSE3 `addsub`: even lanes subtract, odd lanes add.
            fe::check::Ia32Op::AddSub { ps } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let ft = packed_float_ty(*ps);
                let lanes = if *ps { 4 } else { 2 };
                let elem_size = if *ps { 4 } else { 8 };
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                for lane in 0..lanes as u8 {
                    let ax = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane }, ft.clone());
                    let bx = self.emit(func, frame, mir::InstKind::ExtractLane { vec: b, lane }, ft.clone());
                    let r = if lane % 2 == 0 {
                        self.emit(func, frame, mir::InstKind::FSub(ax, bx), ft.clone())
                    } else {
                        self.emit(func, frame, mir::InstKind::FAdd(ax, bx), ft.clone())
                    };
                    let idx = self.iconst(func, 64, i128::from(lane));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, elem_size, ft.clone()).0;
                    self.store(func, frame, addr, r, false);
                }
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // The horizontal pairs: each operand's adjacent lane
            // pairs collapse into the result's own half.
            fe::check::Ia32Op::HBin { add, ps } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let ft = packed_float_ty(*ps);
                let groups = if *ps { 2 } else { 1 };
                let elem_size = if *ps { 4 } else { 8 };
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                for lane in 0..(groups * 2) as u8 {
                    let src = if lane < groups { a } else { b };
                    let lo = (lane % groups) * 2;
                    let x0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: src, lane: lo }, ft.clone());
                    let x1 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: src, lane: lo + 1 }, ft.clone());
                    let r = if *add {
                        self.emit(func, frame, mir::InstKind::FAdd(x0, x1), ft.clone())
                    } else {
                        self.emit(func, frame, mir::InstKind::FSub(x0, x1), ft.clone())
                    };
                    let idx = self.iconst(func, 64, i128::from(lane));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, elem_size, ft.clone()).0;
                    self.store(func, frame, addr, r, false);
                }
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // `movshdup`/`movsldup`: the high or low lane of each
            // adjacent pair duplicated.
            fe::check::Ia32Op::MovDup { high } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let ft = mir::Ty::Float(mir::FloatKind::F32);
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                for lane in 0..4u8 {
                    let src_lane = (lane / 2) * 2 + u8::from(*high);
                    let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: src_lane }, ft.clone());
                    let idx = self.iconst(func, 64, i128::from(lane));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, 4, ft.clone()).0;
                    self.store(func, frame, addr, x, false);
                }
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // `lddqu`: an unaligned load; MIR has no alignment hint,
            // so the ordinary load serves.
            fe::check::Ia32Op::Lddqu => {
                let p = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                self.load(func, frame, p, lower_ty(fe_ty, self.sema, self.interner), false)
            }
            // The sign bits of the four lanes packed into an int.
            fe::check::Ia32Op::MovmskPs => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let vec_ty = lower_ty(&args[0].ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty);
                self.store(func, frame, slot, a, false);
                let out = self.asm_result_in_rax_with_clobbers(
                    func,
                    frame,
                    "movups %1, %%xmm0\n\tmovmskps %%xmm0, %%eax",
                    vec![],
                    vec![mir::AsmOperand {
                        value: slot,
                        constraint: "m".into(),
                        ty: mir::Ty::int(128),
                        tie: None,
                        name: None,
                    }],
                    &["xmm0"],
                );
                self.convert(func, frame, out, &fe::parse::Ty::LongLong, fe_ty)
            }
            // The SSE control-status word.
            fe::check::Ia32Op::Ldmxcsr => {
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                self.asm(
                    func,
                    frame,
                    "ldmxcsr %0",
                    0,
                    vec![mir::AsmOperand {
                        value: v,
                        constraint: "r".into(),
                        ty: mir::Ty::int(32),
                        tie: None,
                        name: None,
                    }],
                    true,
                );
                let undef = func.values.push(mir::ValueData {
                    ty: mir::Ty::Void,
                    def: mir::Def::Const(mir::Const::Undef),
                });
                undef
            }
            fe::check::Ia32Op::Stmxcsr => {
                let slot = self.alloca(func, frame, mir::Ty::int(64));
                self.asm(
                    func,
                    frame,
                    "stmxcsr %0",
                    0,
                    vec![mir::AsmOperand {
                        value: slot,
                        constraint: "=m".into(),
                        ty: mir::Ty::int(32),
                        tie: None,
                        name: None,
                    }],
                    true,
                );
                let r = self.load(func, frame, slot, mir::Ty::int(32), false);
                self.convert(func, frame, r, &fe::parse::Ty::LongLong, fe_ty)
            }
            // One word lane by constant index (`pextrw`), or a word
            // written into one lane (`pinsrw`).
            fe::check::Ia32Op::Pextrw => {
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let imm = match fe::eval::eval_const(&args[1], self.sema, self.interner) {
                    Some(fe::eval::Value::Int(n)) => (n as u64 & 3) as u32,
                    _ => panic!("pextrw selector must be a constant"),
                };
                let r = self.emit(func, frame, mir::InstKind::ExtractLane { vec: v, lane: imm as u8 }, mir::Ty::int(16));
                self.convert(func, frame, r, &fe::parse::Ty::LongLong, fe_ty)
            }
            fe::check::Ia32Op::Pinsrw => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let imm = match fe::eval::eval_const(&args[2], self.sema, self.interner) {
                    Some(fe::eval::Value::Int(n)) => (n as u64 & 3) as u32,
                    _ => panic!("pinsrw selector must be a constant"),
                };
                let vec_ty = lower_ty(&args[0].ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                self.store(func, frame, slot, a, false);
                let idx = self.iconst(func, 64, i128::from(imm));
                let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, 2, mir::Ty::int(16)).0;
                let word = self.convert(func, frame, v, &args[1].ty, &fe::parse::Ty::LongLong);
                let word = self.emit(func, frame, mir::InstKind::Trunc(word), mir::Ty::int(16));
                self.store(func, frame, addr, word, false);
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // The byte sign mask (`pmovmskb`), one bit per lane. The
            // vector stages into `mm0` through 32-bit moves, as in
            // [`Ia32Op::MmxOp`].
            fe::check::Ia32Op::Pmovmskb => {
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let (v_lo, v_hi) = (self.half(func, frame, v, false), self.half(func, frame, v, true));
                let out = self.asm_result_in_rax(
                    func,
                    frame,
                    "movd %k1, %%mm0\n\tmovd %k2, %%mm1\n\tpunpckldq %%mm1, %%mm0\n\tpmovmskb %%mm0, %%eax",
                    vec![],
                    vec![
                        mir::AsmOperand {
                            value: v_lo,
                            constraint: "r".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: v_hi,
                            constraint: "r".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                    ],
                );
                self.convert(func, frame, out, &fe::parse::Ty::LongLong, fe_ty)
            }
            // The MMX word shuffle with a two-bit-per-lane selector.
            fe::check::Ia32Op::Pshufw => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let imm = match fe::eval::eval_const(&args[1], self.sema, self.interner) {
                    Some(fe::eval::Value::Int(n)) => n as u64,
                    _ => panic!("pshufw selector must be a constant"),
                };
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                for lane in 0..4u8 {
                    let sel = ((imm >> (lane * 2)) & 3) as u8;
                    let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: sel }, mir::Ty::int(16));
                    let idx = self.iconst(func, 64, i128::from(lane));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, 2, mir::Ty::int(16)).0;
                    self.store(func, frame, addr, x, false);
                }
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // ---- SSE2: the double-precision mirror of the SSE1 set ----
            // `*sd`: lane 0 of `__v2df` computes; `*pd`: both lanes.
            fe::check::Ia32Op::SdBin { op } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let f64 = mir::Ty::Float(mir::FloatKind::F64);
                let a0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: 0 }, f64.clone());
                let b0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: b, lane: 0 }, f64.clone());
                let r0 = self.float_lane_bin(func, frame, *op, a0, b0, &f64);
                let r = self.vec_replace_lane_gen(func, frame, a, 0, r0, fe_ty);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            fe::check::Ia32Op::PdBin { op } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let f64 = mir::Ty::Float(mir::FloatKind::F64);
                let v2f = mir::Ty::Vector { elem: Box::new(f64.clone()), lanes: 2 };
                if matches!(op, fe::check::SsOp::Min | fe::check::SsOp::Max) {
                    let pred = if matches!(op, fe::check::SsOp::Min) { mir::FcmpPred::Lt } else { mir::FcmpPred::Gt };
                    let i64x2 = mir::Ty::Vector { elem: Box::new(mir::Ty::int(64)), lanes: 2 };
                    let test = self.emit(func, frame, mir::InstKind::Fcmp { pred, lhs: a, rhs: b }, i64x2);
                    return self.emit(func, frame, mir::InstKind::Select { cond: test, then_val: a, else_val: b }, v2f);
                }
                let kind = match op {
                    fe::check::SsOp::Add => mir::InstKind::FAdd(a, b),
                    fe::check::SsOp::Sub => mir::InstKind::FSub(a, b),
                    fe::check::SsOp::Mul => mir::InstKind::FMul(a, b),
                    fe::check::SsOp::Div => mir::InstKind::FDiv(a, b),
                    _ => unreachable!("min/max handled above"),
                };
                self.emit(func, frame, kind, v2f)
            }
            fe::check::Ia32Op::CmpPd { ps, pred } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let i64x2 = mir::Ty::Vector { elem: Box::new(mir::Ty::int(64)), lanes: 2 };
                let mut fcmp = |pred: mir::FcmpPred| self.emit(func, frame, mir::InstKind::Fcmp { pred, lhs: a, rhs: b }, i64x2.clone());
                let mask = self.float_mask(func, frame, *pred, a, b, &i64x2);
                if *ps {
                    let v2si_mask = fe::parse::Ty::Vector { elem: Box::new(fe::parse::Ty::LongLong), len: 2 };
                    return self.convert(func, frame, mask, &v2si_mask, fe_ty);
                }
                let m0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: mask, lane: 0 }, mir::Ty::int(64));
                let r0 = self.emit(func, frame, mir::InstKind::Bitcast(m0), mir::Ty::Float(mir::FloatKind::F64));
                let r = self.vec_replace_lane_gen(func, frame, a, 0, r0, fe_ty);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            fe::check::Ia32Op::SqrtPd => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let f64 = mir::Ty::Float(mir::FloatKind::F64);
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                for lane in 0..2u8 {
                    let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane }, f64.clone());
                    let sq = self.emit(func, frame, mir::InstKind::Sqrt(x), f64.clone());
                    let idx = self.iconst(func, 64, i128::from(lane));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, 8, f64.clone()).0;
                    self.store(func, frame, addr, sq, false);
                }
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            fe::check::Ia32Op::SqrtSd => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let f64 = mir::Ty::Float(mir::FloatKind::F64);
                let a0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: 0 }, f64.clone());
                let r0 = self.emit(func, frame, mir::InstKind::Sqrt(a0), f64);
                let r = self.vec_replace_lane_gen(func, frame, a, 0, r0, fe_ty);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // `shufpd`: one bit per half lane; result lane 0 from `a`,
            // lane 1 from `a` or `b` by the low bit, lane 2 from `b`,
            // lane 3 by the second bit.
            fe::check::Ia32Op::ShufPd => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let imm = match fe::eval::eval_const(&args[2], self.sema, self.interner) {
                    Some(fe::eval::Value::Int(n)) => n as u64,
                    _ => panic!("shufpd selector must be a constant"),
                };
                let f64 = mir::Ty::Float(mir::FloatKind::F64);
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                let picks = [(a, (imm & 1) as u8), (a, ((imm >> 1) & 1) as u8), (b, (imm & 1) as u8), (b, ((imm >> 1) & 1) as u8)];
                for (lane, (src, src_lane)) in picks.iter().enumerate() {
                    let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: *src, lane: *src_lane }, f64.clone());
                    let idx = self.iconst(func, 64, lane as i128);
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, 8, f64.clone()).0;
                    self.store(func, frame, addr, x, false);
                }
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            fe::check::Ia32Op::UnpckPd { high } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let f64 = mir::Ty::Float(mir::FloatKind::F64);
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                // {a0 b0 a0 b0} low, {a1 b1 a1 b1} high.
                let base = if *high { 1 } else { 0 };
                let picks = [(a, base), (b, base), (a, base), (b, base)];
                for (lane, (src, src_lane)) in picks.iter().enumerate() {
                    let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: *src, lane: *src_lane }, f64.clone());
                    let idx = self.iconst(func, 64, lane as i128);
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, 8, f64.clone()).0;
                    self.store(func, frame, addr, x, false);
                }
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            fe::check::Ia32Op::LoadHalfPd { high } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let p = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let f64 = mir::Ty::Float(mir::FloatKind::F64);
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                self.store(func, frame, slot, a, false);
                let val = self.load(func, frame, p, f64.clone(), false);
                let lane = if *high { 1 } else { 0 };
                let dst = self.iconst(func, 64, i128::from(lane));
                let addr = self.scaled_index_addr(func, frame, slot, dst, &fe::parse::Ty::LongLong, 8, f64.clone()).0;
                self.store(func, frame, addr, val, false);
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            fe::check::Ia32Op::StoreHalfPd { high } => {
                let p = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let f64 = mir::Ty::Float(mir::FloatKind::F64);
                let lane = if *high { 1 } else { 0 };
                let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane }, f64);
                self.store(func, frame, p, x, false);
                let undef = func.values.push(mir::ValueData {
                    ty: mir::Ty::Void,
                    def: mir::Def::Const(mir::Const::Undef),
                });
                undef
            }
            fe::check::Ia32Op::MovSd => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let f64 = mir::Ty::Float(mir::FloatKind::F64);
                let b0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: b, lane: 0 }, f64);
                let r = self.vec_replace_lane_gen(func, frame, a, 0, b0, fe_ty);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // `cvtsd2ss`: lane 0 double to float over `__v4sf`.
            fe::check::Ia32Op::CvtSd2Ss => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let f64 = mir::Ty::Float(mir::FloatKind::F64);
                let b0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: b, lane: 0 }, f64);
                let narrow = self.emit(func, frame, mir::InstKind::Fptrunc(b0), mir::Ty::Float(mir::FloatKind::F32));
                let r = self.vec_replace_lane0(func, frame, a, narrow, fe_ty);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // `cvtss2sd`: lanes 0-1 widened from `__v4sf` lanes 0-1.
            fe::check::Ia32Op::CvtSs2Sd => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let f32 = mir::Ty::Float(mir::FloatKind::F32);
                let f64 = mir::Ty::Float(mir::FloatKind::F64);
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                self.store(func, frame, slot, a, false);
                for k in 0..2u8 {
                    let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: b, lane: k }, f32.clone());
                    let wide = self.emit(func, frame, mir::InstKind::Fpext(x), f64.clone());
                    let idx = self.iconst(func, 64, i128::from(k));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, 8, f64.clone()).0;
                    self.store(func, frame, addr, wide, false);
                }
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            fe::check::Ia32Op::CvtSi2Sd { width } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let f64 = mir::Ty::Float(mir::FloatKind::F64);
                let wide = self.emit(func, frame, mir::InstKind::Sext(v), mir::Ty::int(64));
                let r0 = self.emit(func, frame, mir::InstKind::SItoF(wide), f64);
                let r = self.vec_replace_lane_gen(func, frame, a, 0, r0, fe_ty);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            fe::check::Ia32Op::CvtSd2Si { width, truncate } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let f64 = mir::Ty::Float(mir::FloatKind::F64);
                let bits = (*width * 8) as u32;
                let a0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: 0 }, f64);
                if *truncate {
                    let r = self.emit(func, frame, mir::InstKind::SFtoI(a0), mir::Ty::int(bits));
                    return self.convert(func, frame, r, &fe::parse::Ty::LongLong, fe_ty);
                }
                let reg = if *width == 4 { "eax" } else { "rax" };
                let a_slot = self.alloca(func, frame, lower_ty(&args[0].ty, self.sema, self.interner));
                self.store(func, frame, a_slot, a, false);
                let out = self.asm_result_in_rax(
                    func,
                    frame,
                    &format!("cvtsd2si %1, %%{reg}"),
                    vec![],
                    vec![mir::AsmOperand {
                        value: a_slot,
                        constraint: "m".into(),
                        ty: mir::Ty::int(128),
                        tie: None,
                        name: None,
                    }],
                );
                self.convert(func, frame, out, &fe::parse::Ty::LongLong, fe_ty)
            }
            fe::check::Ia32Op::CvtPd2Ps => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let f32 = mir::Ty::Float(mir::FloatKind::F32);
                let f64 = mir::Ty::Float(mir::FloatKind::F64);
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                let zero = self.expr_zero(func, frame, &vec_ty);
                self.store(func, frame, slot, zero, false);
                for k in 0..2u8 {
                    let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: k }, f64.clone());
                    let narrow = self.emit(func, frame, mir::InstKind::Fptrunc(x), f32.clone());
                    let idx = self.iconst(func, 64, i128::from(k));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, 4, f32.clone()).0;
                    self.store(func, frame, addr, narrow, false);
                }
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            fe::check::Ia32Op::CvtPs2Pd => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let f32 = mir::Ty::Float(mir::FloatKind::F32);
                let f64 = mir::Ty::Float(mir::FloatKind::F64);
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                for k in 0..2u8 {
                    let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: k }, f32.clone());
                    let wide = self.emit(func, frame, mir::InstKind::Fpext(x), f64.clone());
                    let idx = self.iconst(func, 64, i128::from(k));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, 8, f64.clone()).0;
                    self.store(func, frame, addr, wide, false);
                }
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // `cvtpd2dq`: the two doubles to the LOW half (truncate
            // form zeroes the rest; the rounding form leaves lanes
            // 2-3 zero per the intrinsic spec too).
            fe::check::Ia32Op::CvtPd2Dq { truncate } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let f64 = mir::Ty::Float(mir::FloatKind::F64);
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                let zero = self.expr_zero(func, frame, &vec_ty);
                self.store(func, frame, slot, zero, false);
                for k in 0..2u8 {
                    let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: k }, f64.clone());
                    let i = if *truncate {
                        self.emit(func, frame, mir::InstKind::SFtoI(x), mir::Ty::int(32))
                    } else {
                        return self.round_cvt_packed(func, frame, "cvtpd2dq", &args[0].ty, a, fe_ty);
                    };
                    let idx = self.iconst(func, 64, i128::from(k));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, 4, mir::Ty::int(32)).0;
                    self.store(func, frame, addr, i, false);
                }
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            fe::check::Ia32Op::CvtPd2Pi { truncate } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let f64 = mir::Ty::Float(mir::FloatKind::F64);
                let mmx_vec = fe::parse::Ty::Vector { elem: Box::new(fe::parse::Ty::Int), len: 2 };
                let image_ty = mir::Ty::int(64);
                if *truncate {
                    let x0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: 0 }, f64.clone());
                    let x1 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: 1 }, f64);
                    let i0 = self.emit(func, frame, mir::InstKind::SFtoI(x0), mir::Ty::int(32));
                    let i1 = self.emit(func, frame, mir::InstKind::SFtoI(x1), mir::Ty::int(32));
                    let wide0 = self.emit(func, frame, mir::InstKind::Zext(i0), image_ty.clone());
                    let shift = self.iconst(func, 64, 32);
                    let wide1 = self.emit(func, frame, mir::InstKind::Zext(i1), image_ty.clone());
                    let hi = self.emit(func, frame, mir::InstKind::Shl(wide1, shift), image_ty.clone());
                    let r = self.emit(func, frame, mir::InstKind::Or(wide0, hi), image_ty);
                    return self.convert(func, frame, r, &fe::parse::Ty::LongLong, &mmx_vec);
                }
                let a_slot = self.alloca(func, frame, lower_ty(&args[0].ty, self.sema, self.interner));
                self.store(func, frame, a_slot, a, false);
                let lo_slot = self.alloca(func, frame, mir::Ty::int(32));
                let hi_slot = self.alloca(func, frame, mir::Ty::int(32));
                self.asm_with_clobbers(
                    func,
                    frame,
                    "movupd %1, %%xmm0\n\tcvtpd2pi %%xmm0, %%mm0\n\tmovd %%mm0, %k0\n\tpunpckhdq %%mm0, %%mm0\n\tmovd %%mm0, %k1",
                    2,
                    vec![
                        mir::AsmOperand {
                            value: lo_slot,
                            constraint: "=a".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: hi_slot,
                            constraint: "=d".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: a_slot,
                            constraint: "m".into(),
                            ty: mir::Ty::int(128),
                            tie: None,
                            name: None,
                        },
                    ],
                    &["xmm0"],
                    true,
                );
                let lo = self.load(func, frame, lo_slot, mir::Ty::int(32), false);
                let hi = self.load(func, frame, hi_slot, mir::Ty::int(32), false);
                let out = self.join_halves(func, frame, lo, hi);
                self.convert(func, frame, out, &fe::parse::Ty::LongLong, &mmx_vec)
            }
            // `cvtpi2pd`: the two integer lanes of the 8-byte vector
            // convert to the two doubles.
            fe::check::Ia32Op::CvtPi2Pd => {
                let m = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let f64 = mir::Ty::Float(mir::FloatKind::F64);
                let i32t = mir::Ty::int(32);
                let i0 = self.emit(func, frame, mir::InstKind::Trunc(m), i32t.clone());
                let shift = self.iconst(func, 64, 32);
                let hi = self.emit(func, frame, mir::InstKind::LShr(m, shift), mir::Ty::int(64));
                let i1 = self.emit(func, frame, mir::InstKind::Trunc(hi), i32t);
                let wide0 = self.emit(func, frame, mir::InstKind::Sext(i0), mir::Ty::int(64));
                let wide1 = self.emit(func, frame, mir::InstKind::Sext(i1), mir::Ty::int(64));
                let f0 = self.emit(func, frame, mir::InstKind::SItoF(wide0), f64.clone());
                let f1 = self.emit(func, frame, mir::InstKind::SItoF(wide1), f64.clone());
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                for (lane, val) in [(0u8, f0), (1, f1)] {
                    let idx = self.iconst(func, 64, i128::from(lane));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, 8, f64.clone()).0;
                    self.store(func, frame, addr, val, false);
                }
                self.load(func, frame, slot, vec_ty, false)
            }
            // `cvtdq2ps`/`cvtdq2pd`: signed int lanes to float.
            fe::check::Ia32Op::CvtInt2Float { to } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let (out_lanes, out_elem) = match to {
                    fe::check::CvtfTy::Ps => (4u8, mir::Ty::Float(mir::FloatKind::F32)),
                    fe::check::CvtfTy::Pd => (2u8, mir::Ty::Float(mir::FloatKind::F64)),
                };
                let i32t = mir::Ty::int(32);
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty.clone());
                for lane in 0..out_lanes {
                    let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane }, i32t.clone());
                    let wide = self.emit(func, frame, mir::InstKind::Sext(x), mir::Ty::int(64));
                    let f = self.emit(func, frame, mir::InstKind::SItoF(wide), out_elem.clone());
                    let bytes = out_elem.size_bytes();
                    let idx = self.iconst(func, 64, i128::from(lane));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, bytes, out_elem.clone()).0;
                    self.store(func, frame, addr, f, false);
                }
                let r = self.load(func, frame, slot, vec_ty, false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // `cvtps2dq`/`cvttps2dq`/`cvtpd2dq`: float lanes to int.
            // The truncating forms ride MIR's float-to-int; the
            // round-to-nearest forms go through the instruction via
            // asm (`cvtps2dq`/`cvtpd2dq` honor the rounding mode).
            fe::check::Ia32Op::CvtFloat2Int { from, truncate } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                if *truncate {
                    let (in_lanes, from_elem) = match from {
                        fe::check::CvtfTy::Ps => (4u8, mir::Ty::Float(mir::FloatKind::F32)),
                        fe::check::CvtfTy::Pd => (2u8, mir::Ty::Float(mir::FloatKind::F64)),
                    };
                    let i32t = mir::Ty::int(32);
                    let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                    let slot = self.alloca(func, frame, vec_ty.clone());
                    for lane in 0..in_lanes {
                        let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane }, from_elem.clone());
                        let i = self.emit(func, frame, mir::InstKind::SFtoI(x), i32t.clone());
                        let idx = self.iconst(func, 64, i128::from(lane));
                        let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, 4, i32t.clone()).0;
                        self.store(func, frame, addr, i, false);
                    }
                    let r = self.load(func, frame, slot, vec_ty, false);
                    return self.convert(func, frame, r, &args[0].ty, fe_ty);
                }
                let mnemonic = match from {
                    fe::check::CvtfTy::Ps => "cvtps2dq",
                    fe::check::CvtfTy::Pd => "cvtpd2dq",
                };
                let a_slot = self.alloca(func, frame, lower_ty(&args[0].ty, self.sema, self.interner));
                self.store(func, frame, a_slot, a, false);
                let out_slot = self.alloca(func, frame, mir::Ty::int(128));
                self.asm_with_clobbers(
                    func,
                    frame,
                    &format!("movups %1, %%xmm0\n\t{mnemonic} %%xmm0, %%xmm0\n\tmovups %%xmm0, %0"),
                    1,
                    vec![
                        mir::AsmOperand {
                            value: out_slot,
                            constraint: "=m".into(),
                            ty: mir::Ty::int(128),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: a_slot,
                            constraint: "m".into(),
                            ty: mir::Ty::int(128),
                            tie: None,
                            name: None,
                        },
                    ],
                    &["xmm0"],
                    true,
                );
                let r = self.load(func, frame, out_slot, lower_ty(fe_ty, self.sema, self.interner), false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // ---- SSE2: the 128-bit integer lanes ----
            // Plain lane arithmetic: MIR's shape-generic integer ops
            // over the first-class vectors; `pandn128` inverts the
            // first operand per lane.
            fe::check::Ia32Op::Vec128Bin { op } => {
                use fe::check::Vec128BinOp;
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                if matches!(op, Vec128BinOp::AndN) {
                    let r = self.vec128_bin_map(func, frame, a, b, false, fe_ty, &mut |func, frame, x, y| {
                        let ones = self.iconst(func, 64, -1);
                        let not_x = self.emit(func, frame, mir::InstKind::Xor(x, ones), mir::Ty::int(64));
                        self.emit(func, frame, mir::InstKind::And(not_x, y), mir::Ty::int(64))
                    });
                    return self.convert(func, frame, r, &args[0].ty, fe_ty);
                }
                let kind = match op {
                    Vec128BinOp::Add => mir::InstKind::Add(a, b),
                    Vec128BinOp::Sub => mir::InstKind::Sub(a, b),
                    Vec128BinOp::And => mir::InstKind::And(a, b),
                    Vec128BinOp::Or => mir::InstKind::Or(a, b),
                    Vec128BinOp::Xor => mir::InstKind::Xor(a, b),
                    Vec128BinOp::AndN => unreachable!("handled above"),
                };
                let r = self.emit(func, frame, kind, vec_ty);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // Saturating add/sub: lanes compute in 64 bits and clamp
            // to the narrower lane's range before truncating back.
            fe::check::Ia32Op::Vec128Sat { add, unsigned } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let shape = self.vector_shape(&args[0].ty);
                let bits = (shape.bytes * 8 / u64::from(shape.lanes)) as u32;
                let (add, unsigned) = (*add, *unsigned);
                let r = self.vec128_bin_map(func, frame, a, b, !unsigned, fe_ty, &mut |func, frame, x, y| {
                    let kind = if add { mir::InstKind::Add(x, y) } else { mir::InstKind::Sub(x, y) };
                    let v = self.emit(func, frame, kind, mir::Ty::int(64));
                    self.lane_clamp(func, frame, v, bits, !unsigned)
                });
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // Lane shifts: per lane in 64 bits, like the rest of the
            // integer lanes. (Hardware zeroes lanes shifted past their
            // width and takes the vector-count form's count from its
            // low 64 bits; the counts ride per lane unmasked here.)
            fe::check::Ia32Op::Vec128Shift { dir, imm } => {
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let count = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let (vec_ty, elem, lanes) = self.vec128_result(fe_ty);
                let (v, v_slot, v_class) = self.vec_stage(func, frame, v, fe_ty);
                let (count, c_slot, c_class) = self.vec_stage(func, frame, count, &args[1].ty);
                let signed = matches!(dir, fe::check::Vec128ShiftDir::Ar);
                let slot = self.alloca(func, frame, vec_ty);
                for lane in 0..lanes {
                    let x = self.vec_lane(func, frame, v, v_slot, v_class, lane, &elem);
                    let x = self.lane_widen(func, frame, x, signed);
                    let c = if *imm {
                        self.convert(func, frame, count, &args[1].ty, &fe::parse::Ty::LongLong)
                    } else {
                        self.vec_lane(func, frame, count, c_slot, c_class, lane, &elem)
                    };
                    let kind = match dir {
                        fe::check::Vec128ShiftDir::Ll => mir::InstKind::Shl(x, c),
                        fe::check::Vec128ShiftDir::Lr => mir::InstKind::LShr(x, c),
                        fe::check::Vec128ShiftDir::Ar => mir::InstKind::AShr(x, c),
                    };
                    let r = self.emit(func, frame, kind, mir::Ty::int(64));
                    self.vec128_store_lane(func, frame, slot, lane, r, &elem);
                }
                self.load(func, frame, slot, lower_ty(fe_ty, self.sema, self.interner), false)
            }
            // Lane compare masks: MIR's vector Icmp yields the 0/-1
            // mask directly.
            fe::check::Ia32Op::Vec128Cmp { eq } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let pred = if *eq { mir::IcmpPred::Eq } else { mir::IcmpPred::Sgt };
                let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
                let mask = self.emit(func, frame, mir::InstKind::Icmp { pred, lhs: a, rhs: b }, vec_ty);
                self.convert(func, frame, mask, &args[0].ty, fe_ty)
            }
            fe::check::Ia32Op::Vec128MinMax { min, signed } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let (min, signed) = (*min, *signed);
                let r = self.vec128_bin_map(func, frame, a, b, signed, fe_ty, &mut |func, frame, x, y| {
                    let pred = match (min, signed) {
                        (true, true) => mir::IcmpPred::Slt,
                        (true, false) => mir::IcmpPred::Ult,
                        (false, true) => mir::IcmpPred::Sgt,
                        (false, false) => mir::IcmpPred::Ugt,
                    };
                    let test = self.emit(func, frame, mir::InstKind::Icmp { pred, lhs: x, rhs: y }, mir::Ty::Bool);
                    self.emit(func, frame, mir::InstKind::Select { cond: test, then_val: x, else_val: y }, mir::Ty::int(64))
                });
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // Unsigned rounding average: `(a + b + 1) >> 1` per lane.
            fe::check::Ia32Op::Vec128Avg => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let r = self.vec128_bin_map(func, frame, a, b, false, fe_ty, &mut |func, frame, x, y| {
                    let one = self.iconst(func, 64, 1);
                    let sum = self.emit(func, frame, mir::InstKind::Add(x, y), mir::Ty::int(64));
                    let sum = self.emit(func, frame, mir::InstKind::Add(sum, one), mir::Ty::int(64));
                    let sh = self.iconst(func, 64, 1);
                    self.emit(func, frame, mir::InstKind::LShr(sum, sh), mir::Ty::int(64))
                });
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // Packs: the operands' lanes clamp to the narrower range
            // and truncate into the result's low and high halves.
            fe::check::Ia32Op::Vec128Pack { kind } => {
                use fe::check::Vec128PackKind;
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let in_elem = self.vector_shape(&args[0].ty).elem_mir;
                let (vec_ty, elem, lanes) = self.vec128_result(fe_ty);
                let (bits, signed) = match kind {
                    Vec128PackKind::SignedByte => (8, true),
                    Vec128PackKind::SignedWord => (16, true),
                    Vec128PackKind::UnsignedByte => (8, false),
                };
                let slot = self.alloca(func, frame, vec_ty);
                let half = lanes / 2;
                for (src, base) in [(a, 0), (b, half)] {
                    for i in 0..half {
                        let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: src, lane: i as u8 }, in_elem.clone());
                        let x = self.lane_widen(func, frame, x, true);
                        let x = self.lane_clamp(func, frame, x, bits, signed);
                        self.vec128_store_lane(func, frame, slot, base + i, x, &elem);
                    }
                }
                let r = self.load(func, frame, slot, lower_ty(fe_ty, self.sema, self.interner), false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // `pmulhw`/`pmulhuw`: the high half of each 16-bit
            // product.
            fe::check::Ia32Op::Vec128Mul { high } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let high = *high;
                let r = self.vec128_bin_map(func, frame, a, b, high, fe_ty, &mut |func, frame, x, y| {
                    let p = self.emit(func, frame, mir::InstKind::Mul(x, y), mir::Ty::int(64));
                    let sh = self.iconst(func, 64, 16);
                    let kind = if high { mir::InstKind::AShr(p, sh) } else { mir::InstKind::LShr(p, sh) };
                    self.emit(func, frame, kind, mir::Ty::int(64))
                });
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // `pmaddwd128`: each pair's two 16-bit products sum into a
            // 32-bit lane.
            fe::check::Ia32Op::Vec128Madd => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let (vec_ty, elem, lanes) = self.vec128_result(fe_ty);
                let word = mir::Ty::int(16);
                let slot = self.alloca(func, frame, vec_ty);
                for i in 0..lanes {
                    let a0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: (2 * i) as u8 }, word.clone());
                    let a1 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: (2 * i + 1) as u8 }, word.clone());
                    let b0 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: b, lane: (2 * i) as u8 }, word.clone());
                    let b1 = self.emit(func, frame, mir::InstKind::ExtractLane { vec: b, lane: (2 * i + 1) as u8 }, word.clone());
                    let a0 = self.lane_widen(func, frame, a0, true);
                    let a1 = self.lane_widen(func, frame, a1, true);
                    let b0 = self.lane_widen(func, frame, b0, true);
                    let b1 = self.lane_widen(func, frame, b1, true);
                    let p0 = self.emit(func, frame, mir::InstKind::Mul(a0, b0), mir::Ty::int(64));
                    let p1 = self.emit(func, frame, mir::InstKind::Mul(a1, b1), mir::Ty::int(64));
                    let s = self.emit(func, frame, mir::InstKind::Add(p0, p1), mir::Ty::int(64));
                    self.vec128_store_lane(func, frame, slot, i, s, &elem);
                }
                let r = self.load(func, frame, slot, lower_ty(fe_ty, self.sema, self.interner), false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // `pmuludq128`: the even dwords (lanes 0 and 2) multiply
            // into the two 64-bit result lanes.
            fe::check::Ia32Op::Vec128MullDq => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let (vec_ty, elem, lanes) = self.vec128_result(fe_ty);
                let dword = mir::Ty::int(32);
                let slot = self.alloca(func, frame, vec_ty);
                for i in 0..lanes {
                    let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: (2 * i) as u8 }, dword.clone());
                    let y = self.emit(func, frame, mir::InstKind::ExtractLane { vec: b, lane: (2 * i) as u8 }, dword.clone());
                    let x = self.lane_widen(func, frame, x, false);
                    let y = self.lane_widen(func, frame, y, false);
                    let p = self.emit(func, frame, mir::InstKind::Mul(x, y), mir::Ty::int(64));
                    let idx = self.iconst(func, 64, i128::from(i));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, elem.size_bytes(), elem.clone()).0;
                    self.store(func, frame, addr, p, false);
                }
                let r = self.load(func, frame, slot, lower_ty(fe_ty, self.sema, self.interner), false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // `psadbw128`: the absolute byte differences of each
            // 8-byte block sum into the block's low 16 bits (the sum
            // stays under 2^16, so the upper bytes are zero).
            fe::check::Ia32Op::Vec128Sad => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let (vec_ty, elem, lanes) = self.vec128_result(fe_ty);
                let byte = mir::Ty::int(8);
                let slot = self.alloca(func, frame, vec_ty);
                for g in 0..lanes as u8 {
                    let mut sum = self.iconst(func, 64, 0);
                    for i in 0..8u8 {
                        let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: 8 * g + i }, byte.clone());
                        let y = self.emit(func, frame, mir::InstKind::ExtractLane { vec: b, lane: 8 * g + i }, byte.clone());
                        let x = self.lane_widen(func, frame, x, false);
                        let y = self.lane_widen(func, frame, y, false);
                        let ge = self.emit(func, frame, mir::InstKind::Icmp { pred: mir::IcmpPred::Uge, lhs: x, rhs: y }, mir::Ty::Bool);
                        let d1 = self.emit(func, frame, mir::InstKind::Sub(x, y), mir::Ty::int(64));
                        let d2 = self.emit(func, frame, mir::InstKind::Sub(y, x), mir::Ty::int(64));
                        let d = self.emit(func, frame, mir::InstKind::Select { cond: ge, then_val: d1, else_val: d2 }, mir::Ty::int(64));
                        sum = self.emit(func, frame, mir::InstKind::Add(sum, d), mir::Ty::int(64));
                    }
                    let idx = self.iconst(func, 64, i128::from(g));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, elem.size_bytes(), elem.clone()).0;
                    self.store(func, frame, addr, sum, false);
                }
                let r = self.load(func, frame, slot, lower_ty(fe_ty, self.sema, self.interner), false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // `punpck{h,l}*`: the interleave of one operand's half
            // with the other's.
            fe::check::Ia32Op::Vec128Unpck { high } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let (vec_ty, elem, lanes) = self.vec128_result(fe_ty);
                let slot = self.alloca(func, frame, vec_ty);
                let half = lanes / 2;
                let base = if *high { half } else { 0 };
                for i in 0..half {
                    let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: (base + i) as u8 }, elem.clone());
                    let idx = self.iconst(func, 64, i128::from(2 * i));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, elem.size_bytes(), elem.clone()).0;
                    self.store(func, frame, addr, x, false);
                    let y = self.emit(func, frame, mir::InstKind::ExtractLane { vec: b, lane: (base + i) as u8 }, elem.clone());
                    let idx = self.iconst(func, 64, i128::from(2 * i + 1));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, elem.size_bytes(), elem.clone()).0;
                    self.store(func, frame, addr, y, false);
                }
                let r = self.load(func, frame, slot, lower_ty(fe_ty, self.sema, self.interner), false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // `pshufd`/`pshuflw`/`pshufhw`: the two-bit selectors
            // index the dwords or the low/high words; the untouched
            // half passes through.
            fe::check::Ia32Op::Vec128Shuf { mode } => {
                use fe::check::Vec128ShufMode;
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let imm = match fe::eval::eval_const(&args[1], self.sema, self.interner) {
                    Some(fe::eval::Value::Int(n)) => n as u64,
                    _ => panic!("pshuf selector must be a constant"),
                };
                let (vec_ty, elem, lanes) = self.vec128_result(fe_ty);
                let slot = self.alloca(func, frame, vec_ty);
                let src_lane = |lane: u32| -> u8 {
                    match mode {
                        Vec128ShufMode::D => ((imm >> (lane * 2)) & 3) as u8,
                        Vec128ShufMode::LowW => {
                            if lane < 4 {
                                ((imm >> (lane * 2)) & 3) as u8
                            } else {
                                lane as u8
                            }
                        }
                        Vec128ShufMode::HighW => {
                            if lane < 4 {
                                lane as u8
                            } else {
                                4 + ((imm >> ((lane - 4) * 2)) & 3) as u8
                            }
                        }
                    }
                };
                for lane in 0..lanes {
                    let x = self.emit(func, frame, mir::InstKind::ExtractLane { vec: a, lane: src_lane(lane) }, elem.clone());
                    let idx = self.iconst(func, 64, i128::from(lane));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, elem.size_bytes(), elem.clone()).0;
                    self.store(func, frame, addr, x, false);
                }
                let r = self.load(func, frame, slot, lower_ty(fe_ty, self.sema, self.interner), false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // One dword lane by constant index.
            fe::check::Ia32Op::VecExtV4Si => {
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let imm = match fe::eval::eval_const(&args[1], self.sema, self.interner) {
                    Some(fe::eval::Value::Int(n)) => (n as u64 & 3) as u8,
                    _ => panic!("vec_ext_v4si index must be a constant"),
                };
                let r = self.emit(func, frame, mir::InstKind::ExtractLane { vec: v, lane: imm }, mir::Ty::int(32));
                self.convert(func, frame, r, &fe::parse::Ty::Int, fe_ty)
            }
            // Clang `__builtin_elementwise_sqrt`: the square root of
            // every lane (or of the scalar).
            fe::check::Ia32Op::Vec128Sqrt => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                if fe::check::is_vector(&args[0].ty) {
                    let (vec_ty, elem, lanes) = self.vec128_result(fe_ty);
                    let (a, a_slot, a_class) = self.vec_stage(func, frame, a, fe_ty);
                    let slot = self.alloca(func, frame, vec_ty);
                    for lane in 0..lanes {
                        let x = self.vec_lane(func, frame, a, a_slot, a_class, lane, &elem);
                        let sq = self.emit(func, frame, mir::InstKind::Sqrt(x), elem.clone());
                        let idx = self.iconst(func, 64, i128::from(lane));
                        let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, elem.size_bytes(), elem.clone()).0;
                        self.store(func, frame, addr, sq, false);
                    }
                    self.load(func, frame, slot, lower_ty(fe_ty, self.sema, self.interner), false)
                } else {
                    let elem = lower_ty(fe_ty, self.sema, self.interner);
                    self.emit(func, frame, mir::InstKind::Sqrt(a), elem)
                }
            }
            // Clang `__builtin_convertvector`: the element-wise
            // conversion to the target vector type. Source lanes read
            // at the source element's width, results store at the
            // destination's.
            fe::check::Ia32Op::ConvertVector => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let src_shape = self.vector_shape(&args[0].ty);
                let dst_elem_fe = self.vector_shape(fe_ty).elem_fe;
                let (vec_ty, elem, lanes) = self.vec128_result(fe_ty);
                let (a, a_slot, a_class) = self.vec_stage(func, frame, a, &args[0].ty);
                let slot = self.alloca(func, frame, vec_ty);
                for lane in 0..lanes {
                    let x = self.vec_lane(func, frame, a, a_slot, a_class, lane, &src_shape.elem_mir);
                    let c = self.convert(func, frame, x, &src_shape.elem_fe, &dst_elem_fe);
                    let idx = self.iconst(func, 64, i128::from(lane));
                    let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, elem.size_bytes(), elem.clone()).0;
                    self.store(func, frame, addr, c, false);
                }
                self.load(func, frame, slot, lower_ty(fe_ty, self.sema, self.interner), false)
            }
            // Clang `__builtin_nontemporal_store`: an ordinary store,
            // the hint dropped.
            fe::check::Ia32Op::NtStore => {
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let p = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let slot = self.alloca(func, frame, lower_ty(&args[0].ty, self.sema, self.interner));
                self.store(func, frame, slot, v, false);
                let img = self.load(func, frame, slot, lower_ty(&args[0].ty, self.sema, self.interner), false);
                self.store(func, frame, p, img, false);
                let undef = func.values.push(mir::ValueData {
                    ty: mir::Ty::Void,
                    def: mir::Def::Const(mir::Const::Undef),
                });
                undef
            }
            // An undefined vector.
            fe::check::Ia32Op::Undef => func.values.push(mir::ValueData {
                ty: lower_ty(fe_ty, self.sema, self.interner),
                def: mir::Def::Const(mir::Const::Undef),
            }),
            // The whole 128-bit register shifted by whole bytes,
            // zero-filled (the `slli`/`srli_si128` semantics).
            fe::check::Ia32Op::VecByteShift { left } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let imm = match fe::eval::eval_const(&args[1], self.sema, self.interner) {
                    Some(fe::eval::Value::Int(n)) => (n as u64).min(16),
                    _ => panic!("byte shift count must be a constant"),
                };
                let (vec_ty, elem, lanes) = self.vec128_result(fe_ty);
                let (a, a_slot, a_class) = self.vec_stage(func, frame, a, fe_ty);
                let slot = self.alloca(func, frame, vec_ty);
                for lane in 0..lanes {
                    let src = if *left {
                        lane.checked_sub(imm as u32)
                    } else {
                        lane.checked_add(imm as u32).filter(|s| *s < lanes)
                    };
                    let val = match src {
                        Some(s) => {
                            let x = self.vec_lane(func, frame, a, a_slot, a_class, s, &elem);
                            self.lane_widen(func, frame, x, false)
                        }
                        None => self.iconst(func, 64, 0),
                    };
                    self.vec128_store_lane(func, frame, slot, lane, val, &elem);
                }
                self.load(func, frame, slot, lower_ty(fe_ty, self.sema, self.interner), false)
            }
            // The 16 byte sign bits.
            fe::check::Ia32Op::Pmovmskb128 => {
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let vec_ty = lower_ty(&args[0].ty, self.sema, self.interner);
                let slot = self.alloca(func, frame, vec_ty);
                self.store(func, frame, slot, v, false);
                let out = self.asm_result_in_rax_with_clobbers(
                    func,
                    frame,
                    "movdqu %1, %%xmm0\n\tpmovmskb %%xmm0, %%eax",
                    vec![],
                    vec![mir::AsmOperand {
                        value: slot,
                        constraint: "m".into(),
                        ty: mir::Ty::int(128),
                        tie: None,
                        name: None,
                    }],
                    &["xmm0"],
                );
                self.convert(func, frame, out, &fe::parse::Ty::LongLong, fe_ty)
            }
            // `movq`: the low qword passes through, the high qword
            // zeroes.
            fe::check::Ia32Op::Movq128 => {
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let (vec_ty, elem, _) = self.vec128_result(fe_ty);
                let in_slot = self.alloca(func, frame, lower_ty(&args[0].ty, self.sema, self.interner));
                self.store(func, frame, in_slot, v, false);
                let lo = self.load(func, frame, in_slot, mir::Ty::int(64), false);
                let out_slot = self.alloca(func, frame, vec_ty);
                let zero = self.iconst(func, 128, 0);
                self.store(func, frame, out_slot, zero, false);
                let idx = self.iconst(func, 64, 0);
                let addr = self.scaled_index_addr(func, frame, out_slot, idx, &fe::parse::Ty::LongLong, elem.size_bytes(), elem).0;
                self.store(func, frame, addr, lo, false);
                let r = self.load(func, frame, out_slot, lower_ty(fe_ty, self.sema, self.interner), false);
                self.convert(func, frame, r, &args[0].ty, fe_ty)
            }
            // Non-temporal stores: ordinary MIR stores, the hint
            // dropped.
            fe::check::Ia32Op::MovNt { bytes } => {
                let p = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let bits = u32::from(*bytes) * 8;
                let slot = self.alloca(func, frame, lower_ty(&args[1].ty, self.sema, self.interner));
                self.store(func, frame, slot, v, false);
                let img = self.load(func, frame, slot, mir::Ty::int(bits), false);
                self.store(func, frame, p, img, false);
                let undef = func.values.push(mir::ValueData {
                    ty: mir::Ty::Void,
                    def: mir::Def::Const(mir::Const::Undef),
                });
                undef
            }
            // `maskmovdqu`/`maskmovq`: the mask-selected bytes of the
            // first operand go to `[rdi]`; the instruction's AT&T form
            // takes the mask first, then the data.
            fe::check::Ia32Op::MaskMov => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let m = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let p = self.expr_value(module, func, frame, funcs, externs, globals, &args[2]);
                let xmm = lower_ty(&args[0].ty, self.sema, self.interner).size_bytes() == 16;
                let (mov, insn, vec_ty) = if xmm {
                    ("movdqu", "maskmovdqu", mir::Ty::int(128))
                } else {
                    ("movq", "maskmovq", mir::Ty::int(64))
                };
                let mask_ty = vec_ty.clone();
                let a_slot = self.alloca(func, frame, lower_ty(&args[0].ty, self.sema, self.interner));
                self.store(func, frame, a_slot, a, false);
                let m_slot = self.alloca(func, frame, lower_ty(&args[1].ty, self.sema, self.interner));
                self.store(func, frame, m_slot, m, false);
                self.asm_with_clobbers(
                    func,
                    frame,
                    &format!("{mov} %0, %%xmm0\n\t{mov} %1, %%xmm1\n\tmovq %2, %%rdi\n\t{insn} %%xmm1, %%xmm0"),
                    0,
                    vec![
                        mir::AsmOperand {
                            value: a_slot,
                            constraint: "m".into(),
                            ty: vec_ty,
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: m_slot,
                            constraint: "m".into(),
                            ty: mask_ty,
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: p,
                            constraint: "r".into(),
                            ty: mir::Ty::ptr(mir::Ty::Void),
                            tie: None,
                            name: None,
                        },
                    ],
                    &["xmm0", "xmm1"],
                    true,
                );
                let undef = func.values.push(mir::ValueData {
                    ty: mir::Ty::Void,
                    def: mir::Def::Const(mir::Const::Undef),
                });
                undef
            }
            fe::check::Ia32Op::UMonitor => {
                let ptr = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                self.asm(
                    func,
                    frame,
                    "umonitor %0",
                    0,
                    vec![mir::AsmOperand {
                        value: ptr,
                        constraint: "r".into(),
                        ty: mir::Ty::ptr(mir::Ty::Void),
                        tie: None,
                        name: None,
                    }],
                    true,
                );
                self.iconst(func, 32, 0)
            }
            // `umwait`/`tpause` take the control word in `ecx` and the
            // deadline in `edx:eax`, reporting the wakeup in `eax`.
            fe::check::Ia32Op::UWait { tpause } => {
                let control = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let deadline = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let deadline_lo = self.half(func, frame, deadline, false);
                let deadline_hi = self.half(func, frame, deadline, true);
                let slot = self.alloca(func, frame, mir::Ty::int(32));
                self.asm(
                    func,
                    frame,
                    if *tpause { "tpause" } else { "umwait" },
                    1,
                    vec![
                        mir::AsmOperand {
                            value: slot,
                            constraint: "=a".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: control,
                            constraint: "c".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: deadline_lo,
                            constraint: "a".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: deadline_hi,
                            constraint: "d".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                    ],
                    true,
                );
                let out = self.load(func, frame, slot, mir::Ty::int(32), false);
                self.convert(func, frame, out, &fe::parse::Ty::UnsignedInt, fe_ty)
            }
            // EFLAGS round-trips through the stack: `pushf` spills
            // the flags, `popf` restores them.
            fe::check::Ia32Op::ReadFlags { width } => {
                let (bits, push, pop) = if *width == 4 {
                    (32, "pushfd", "popl %0")
                } else {
                    (64, "pushfq", "popq %0")
                };
                let slot = self.alloca(func, frame, mir::Ty::int(bits));
                self.asm(
                    func,
                    frame,
                    &format!("{push}\n\t{pop}"),
                    1,
                    vec![mir::AsmOperand {
                        value: slot,
                        constraint: "=r".into(),
                        ty: mir::Ty::int(bits),
                        tie: None,
                        name: None,
                    }],
                    true,
                );
                let out = self.load(func, frame, slot, mir::Ty::int(bits), false);
                self.convert(func, frame, out, &self.args_ty_of_width(*width), fe_ty)
            }
            fe::check::Ia32Op::WriteFlags { width } => {
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let (push, pop) = if *width == 4 {
                    ("pushl %0", "popfd")
                } else {
                    ("pushq %0", "popfq")
                };
                self.asm(
                    func,
                    frame,
                    &format!("{push}\n\t{pop}"),
                    0,
                    vec![mir::AsmOperand {
                        value: v,
                        constraint: "r".into(),
                        ty: mir::Ty::int(u32::from(*width) * 8),
                        tie: None,
                        name: None,
                    }],
                    true,
                );
                self.iconst(func, 32, 0)
            }
            // The CET shadow-stack builtins: each is one instruction.
            fe::check::Ia32Op::RdSsp { width } => {
                let bits = u32::from(*width) * 8;
                let slot = self.alloca(func, frame, mir::Ty::int(bits));
                let mnemonic = if *width == 4 { "rdsspd" } else { "rdsspq" };
                self.asm(
                    func,
                    frame,
                    mnemonic,
                    1,
                    vec![mir::AsmOperand {
                        value: slot,
                        constraint: "=a".into(),
                        ty: mir::Ty::int(bits),
                        tie: None,
                        name: None,
                    }],
                    true,
                );
                let out = self.load(func, frame, slot, mir::Ty::int(bits), false);
                self.convert(func, frame, out, &self.args_ty_of_width(*width), fe_ty)
            }
            // SSP increments take an immediate operand (the number of
            // slots), like GCC's `"e"` constraint.
            fe::check::Ia32Op::IncSsp { width } => {
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let mnemonic = if *width == 4 { "incsspd" } else { "incsspq" };
                self.asm(
                    func,
                    frame,
                    &format!("{mnemonic} (%0)"),
                    0,
                    vec![mir::AsmOperand {
                        value: v,
                        constraint: "e".into(),
                        ty: mir::Ty::int(u32::from(*width) * 8),
                        tie: None,
                        name: None,
                    }],
                    true,
                );
                self.iconst(func, 32, 0)
            }
            // The shadow-stack token and opaque-store instructions
            // take a memory operand.
            fe::check::Ia32Op::RstorSsp | fe::check::Ia32Op::ClrSsbsy => {
                let ptr = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let mnemonic = if matches!(op, fe::check::Ia32Op::RstorSsp) { "rstorssp" } else { "clrssbsy" };
                self.asm(
                    func,
                    frame,
                    &format!("{mnemonic} (%0)"),
                    0,
                    vec![mir::AsmOperand {
                        value: ptr,
                        constraint: "m".into(),
                        ty: mir::Ty::ptr(mir::Ty::Void),
                        tie: None,
                        name: None,
                    }],
                    true,
                );
                self.iconst(func, 32, 0)
            }
            fe::check::Ia32Op::Wrss { uss, width } => {
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let ptr = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let mnemonic = match (uss, *width) {
                    (false, 4) => "wrssd",
                    (false, _) => "wrssq",
                    (true, 4) => "wrussd",
                    (true, _) => "wrussq",
                };
                self.asm(
                    func,
                    frame,
                    &format!("{mnemonic} %0, (%1)"),
                    0,
                    vec![
                        mir::AsmOperand {
                            value: v,
                            constraint: "r".into(),
                            ty: mir::Ty::int(u32::from(*width) * 8),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: ptr,
                            constraint: "m".into(),
                            ty: mir::Ty::ptr(mir::Ty::Void),
                            tie: None,
                            name: None,
                        },
                    ],
                    true,
                );
                self.iconst(func, 32, 0)
            }
            // Cache-line hints and the FXSAVE/FXRSTOR pair: one
            // memory operand, no result.
            fe::check::Ia32Op::CacheOp { mnemonic } => {
                let ptr = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                self.asm(
                    func,
                    frame,
                    &format!("{mnemonic} (%0)"),
                    0,
                    vec![mir::AsmOperand {
                        value: ptr,
                        constraint: "m".into(),
                        ty: mir::Ty::ptr(mir::Ty::Void),
                        tie: None,
                        name: None,
                    }],
                    true,
                );
                self.iconst(func, 32, 0)
            }
            // `enqcmd`/`enqcmds` enqueue the 64-byte store from the
            // register operand to the memory operand and report
            // acceptance in ZF, which the instruction leaves in `eax`.
            fe::check::Ia32Op::EnqCmd { ordered } => {
                let dst = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let src = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let out = self.asm_result_in_rax(
                    func,
                    frame,
                    if *ordered { "enqcmds (%1), %2" } else { "enqcmd (%1), %2" },
                    vec![],
                    vec![
                        mir::AsmOperand {
                            value: dst,
                            constraint: "m".into(),
                            ty: mir::Ty::ptr(mir::Ty::Void),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: src,
                            constraint: "r".into(),
                            ty: mir::Ty::ptr(mir::Ty::Void),
                            tie: None,
                            name: None,
                        },
                    ],
                );
                self.convert(func, frame, out, &fe::parse::Ty::Int, fe_ty)
            }
            fe::check::Ia32Op::Llwpcb => {
                let ptr = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                self.asm(
                    func,
                    frame,
                    "llwpcb %0",
                    0,
                    vec![mir::AsmOperand {
                        value: ptr,
                        constraint: "r".into(),
                        ty: mir::Ty::ptr(mir::Ty::Void),
                        tie: None,
                        name: None,
                    }],
                    true,
                );
                self.iconst(func, 32, 0)
            }
            fe::check::Ia32Op::Slwpcb => {
                let out = self.asm_result_in_rax(func, frame, "slwpcb", vec![], vec![]);
                self.convert(func, frame, out, &fe::parse::Ty::UnsignedLongLong, fe_ty)
            }
            // The monitor/wait pairs pass the address in `eax`, the
            // extension word in `ecx`, and the hints in `edx`; the
            // `x` forms take an extra timer in `ebx`.
            fe::check::Ia32Op::Monitor { extended } => {
                let addr = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let extensions = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let hints = self.expr_value(module, func, frame, funcs, externs, globals, &args[2]);
                self.asm(
                    func,
                    frame,
                    if *extended { "monitorx" } else { "monitor" },
                    0,
                    vec![
                        mir::AsmOperand {
                            value: addr,
                            constraint: "a".into(),
                            ty: mir::Ty::ptr(mir::Ty::Void),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: extensions,
                            constraint: "c".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: hints,
                            constraint: "d".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                    ],
                    true,
                );
                self.iconst(func, 32, 0)
            }
            fe::check::Ia32Op::Mwait { extended } => {
                let extensions = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let hints = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let mut operands = vec![
                    mir::AsmOperand {
                        value: extensions,
                        constraint: "c".into(),
                        ty: mir::Ty::int(32),
                        tie: None,
                        name: None,
                    },
                    mir::AsmOperand {
                        value: hints,
                        constraint: "a".into(),
                        ty: mir::Ty::int(32),
                        tie: None,
                        name: None,
                    },
                ];
                if *extended {
                    let timer = self.expr_value(module, func, frame, funcs, externs, globals, &args[2]);
                    operands.push(mir::AsmOperand {
                        value: timer,
                        constraint: "b".into(),
                        ty: mir::Ty::int(32),
                        tie: None,
                        name: None,
                    });
                }
                self.asm(func, frame, if *extended { "mwaitx" } else { "mwait" }, 0, operands, true);
                self.iconst(func, 32, 0)
            }
            // The protection-key register is read and written with
            // `ecx` = `edx` = 0 by instruction contract.
            fe::check::Ia32Op::Rdpkru => {
                let zero = self.iconst(func, 32, 0);
                let edx_zero = self.iconst(func, 32, 0);
                let out = self.asm_result_in_rax(
                    func,
                    frame,
                    "rdpkru",
                    vec![],
                    vec![
                        mir::AsmOperand {
                            value: zero,
                            constraint: "c".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: edx_zero,
                            constraint: "d".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                    ],
                );
                self.convert(func, frame, out, &fe::parse::Ty::UnsignedInt, fe_ty)
            }
            // `rdseed`/`rdrand` write the value to the operand and the
            // validity flag to CF; `setc %al` moves the flag into the
            // `=a` output's low byte, which is all the caller sees.
            fe::check::Ia32Op::RdStep { rand, width } => {
                let bits = u32::from(*width) * 8;
                let ptr = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let val_slot = self.alloca(func, frame, mir::Ty::int(bits));
                let ok_slot = self.alloca(func, frame, mir::Ty::int(64));
                self.asm(
                    func,
                    frame,
                    &format!("{} %1\n\tsetc %%al", if *rand { "rdrand" } else { "rdseed" }),
                    2,
                    vec![
                        mir::AsmOperand {
                            value: ok_slot,
                            constraint: "=a".into(),
                            ty: mir::Ty::int(64),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: val_slot,
                            constraint: "=r".into(),
                            ty: mir::Ty::int(bits),
                            tie: None,
                            name: None,
                        },
                    ],
                    true,
                );
                let value = self.load(func, frame, val_slot, mir::Ty::int(bits), false);
                self.store(func, frame, ptr, value, false);
                let ok = self.load(func, frame, ok_slot, mir::Ty::int(64), false);
                self.convert(func, frame, ok, &fe::parse::Ty::UnsignedLongLong, fe_ty)
            }
            fe::check::Ia32Op::Wrpkru => {
                let key = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let zero = self.iconst(func, 32, 0);
                let edx_zero = self.iconst(func, 32, 0);
                self.asm(
                    func,
                    frame,
                    "wrpkru",
                    0,
                    vec![
                        mir::AsmOperand {
                            value: key,
                            constraint: "a".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: zero,
                            constraint: "c".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: edx_zero,
                            constraint: "d".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                    ],
                    true,
                );
                self.iconst(func, 32, 0)
            }
        }
    }

    /// Emit one inline-asm block with `outputs` output operands
    /// followed by input operands.
    fn asm(&self, func: &mut mir::Function, frame: &mut Frame, template: &str, outputs: u32, operands: Vec<mir::AsmOperand>, volatile: bool) {
        self.asm_with_clobbers(func, frame, template, outputs, operands, &[], volatile);
    }

    /// [`Self::asm`] with a clobber list: registers named here are
    /// reserved with the register allocator, so a literal `%%xmmN`
    /// scratch in the template cannot hold a live value.
    fn asm_with_clobbers(&self, func: &mut mir::Function, frame: &mut Frame, template: &str, outputs: u32, operands: Vec<mir::AsmOperand>, clobbers: &[&str], volatile: bool) {
        self.emit_unit(func, frame, mir::InstKind::InlineAsm {
            block: Box::new(mir::AsmBlock {
                template: template.into(),
                operands,
                outputs,
                clobbers: clobbers.iter().map(|c| c.to_string()).collect(),
                labels: vec![],
                is_volatile: volatile,
                is_basic: false,
            }),
        });
    }

    /// Run an asm template whose result lands in `rax` (the `=a`
    /// output operand, then `extra_outputs`, then `inputs`) and load
    /// it back: the `rdtsc` family writes `edx:eax`, and writing
    /// `eax` zero-extends `rax` on x86-64, so one 64-bit output
    /// carries the full result.
    /// One lane-wise float operation for the scalar-tail builtins.
    fn float_lane_bin(&self, func: &mut mir::Function, frame: &mut Frame, op: fe::check::SsOp, a: mir::Value, b: mir::Value, elem: &mir::Ty) -> mir::Value {
        use fe::check::SsOp;
        match op {
            SsOp::Add => self.emit(func, frame, mir::InstKind::FAdd(a, b), elem.clone()),
            SsOp::Sub => self.emit(func, frame, mir::InstKind::FSub(a, b), elem.clone()),
            SsOp::Mul => self.emit(func, frame, mir::InstKind::FMul(a, b), elem.clone()),
            SsOp::Div => self.emit(func, frame, mir::InstKind::FDiv(a, b), elem.clone()),
            // minss/maxss (and their SD spellings) return the second
            // operand when the first is NaN.
            SsOp::Min | SsOp::Max => {
                let pred = if matches!(op, SsOp::Min) { mir::FcmpPred::Lt } else { mir::FcmpPred::Gt };
                let test = self.emit(func, frame, mir::InstKind::Fcmp { pred, lhs: a, rhs: b }, mir::Ty::Bool);
                self.emit(func, frame, mir::InstKind::Select { cond: test, then_val: a, else_val: b }, elem.clone())
            }
        }
    }

    /// Build the lane mask of an SSE compare predicate (the `n*` forms
    /// OR in the unordered mask).
    fn float_mask(&self, func: &mut mir::Function, frame: &mut Frame, pred: fe::check::CmpPred, a: mir::Value, b: mir::Value, mask_ty: &mir::Ty) -> mir::Value {
        use fe::check::CmpPred;
        let fcmp = |pred: mir::FcmpPred, func: &mut mir::Function, frame: &mut Frame| {
            self.emit(func, frame, mir::InstKind::Fcmp { pred, lhs: a, rhs: b }, mask_ty.clone())
        };
        match pred {
            CmpPred::Eq => fcmp(mir::FcmpPred::Eq, func, frame),
            CmpPred::Lt => fcmp(mir::FcmpPred::Lt, func, frame),
            CmpPred::Le => fcmp(mir::FcmpPred::Le, func, frame),
            CmpPred::Gt => fcmp(mir::FcmpPred::Gt, func, frame),
            CmpPred::Ge => fcmp(mir::FcmpPred::Ge, func, frame),
            CmpPred::Neq => fcmp(mir::FcmpPred::Ne, func, frame),
            CmpPred::Nlt => {
                let ge = fcmp(mir::FcmpPred::Ge, func, frame);
                let uno = fcmp(mir::FcmpPred::Uno, func, frame);
                self.emit(func, frame, mir::InstKind::Or(ge, uno), mask_ty.clone())
            }
            CmpPred::Nle => {
                let gt = fcmp(mir::FcmpPred::Gt, func, frame);
                let uno = fcmp(mir::FcmpPred::Uno, func, frame);
                self.emit(func, frame, mir::InstKind::Or(gt, uno), mask_ty.clone())
            }
            CmpPred::Ngt => {
                let le = fcmp(mir::FcmpPred::Le, func, frame);
                let uno = fcmp(mir::FcmpPred::Uno, func, frame);
                self.emit(func, frame, mir::InstKind::Or(le, uno), mask_ty.clone())
            }
            CmpPred::Nge => {
                let lt = fcmp(mir::FcmpPred::Lt, func, frame);
                let uno = fcmp(mir::FcmpPred::Uno, func, frame);
                self.emit(func, frame, mir::InstKind::Or(lt, uno), mask_ty.clone())
            }
            CmpPred::Ord => fcmp(mir::FcmpPred::Ord, func, frame),
            CmpPred::Unord => fcmp(mir::FcmpPred::Uno, func, frame),
        }
    }

    /// An all-zero vector value of `ty` (first-class vector only).
    fn expr_zero(&self, func: &mut mir::Function, frame: &mut Frame, ty: &mir::Ty) -> mir::Value {
        let elem = match ty {
            mir::Ty::Vector { elem, .. } => (**elem).clone(),
            _ => unreachable!("vector zero"),
        };
        let zero = if matches!(elem, mir::Ty::Float(_)) {
            func.push_const(mir::Const::Float {
                kind: float_kind(&elem),
                bits: 0,
            })
        } else {
            self.iconst(func, int_bits(&elem), 0)
        };
        self.emit(func, frame, mir::InstKind::Splat(zero), ty.clone())
    }

    /// Round-to-nearest packed float→int through the instruction (MIR
    /// has no rounding conversion): one asm stub per site.
    fn round_cvt_packed(&self, func: &mut mir::Function, frame: &mut Frame, mnemonic: &str, in_fe_ty: &fe::parse::Ty, a: mir::Value, fe_ty: &fe::parse::Ty) -> mir::Value {
        let a_slot = self.alloca(func, frame, lower_ty(in_fe_ty, self.sema, self.interner));
        self.store(func, frame, a_slot, a, false);
        let out_slot = self.alloca(func, frame, mir::Ty::int(128));
        self.asm_with_clobbers(
            func,
            frame,
            &format!("movups %1, %%xmm0\n\t{mnemonic} %%xmm0, %%xmm0\n\tmovups %%xmm0, %0"),
            1,
            vec![
                mir::AsmOperand {
                    value: out_slot,
                    constraint: "=m".into(),
                    ty: mir::Ty::int(128),
                    tie: None,
                    name: None,
                },
                mir::AsmOperand {
                    value: a_slot,
                    constraint: "m".into(),
                    ty: mir::Ty::int(128),
                    tie: None,
                    name: None,
                },
            ],
            &["xmm0"],
            true,
        );
        let r = self.load(func, frame, out_slot, lower_ty(fe_ty, self.sema, self.interner), false);
        self.convert(func, frame, r, fe_ty, fe_ty)
    }

    /// A 16-byte vector with one lane replaced, by immediate index.
    fn vec_replace_lane_gen(&self, func: &mut mir::Function, frame: &mut Frame, v: mir::Value, lane: u8, new: mir::Value, fe_ty: &fe::parse::Ty) -> mir::Value {
        let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
        let (_, lanes) = vec_ty.as_vector().expect("vector replace lane");
        let elem_size = vec_ty.size_bytes() / lanes as u64;
        let elem = match &vec_ty {
            mir::Ty::Vector { elem, .. } => (**elem).clone(),
            _ => unreachable!("vector replace lane"),
        };
        let slot = self.alloca(func, frame, vec_ty.clone());
        self.store(func, frame, slot, v, false);
        let idx = self.iconst(func, 64, i128::from(lane));
        let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, elem_size, elem).0;
        self.store(func, frame, addr, new, false);
        self.load(func, frame, slot, vec_ty, false)
    }

    /// A 16-byte vector with lane 0 replaced: copy through a scratch
    /// slot, store the new lane over offset 0, reload.
    fn vec_replace_lane0(&self, func: &mut mir::Function, frame: &mut Frame, v: mir::Value, lane0: mir::Value, fe_ty: &fe::parse::Ty) -> mir::Value {
        let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
        let elem_size = vec_ty.size_bytes() / vec_ty.as_vector().map(|(_, lanes)| lanes).unwrap_or(1) as u64;
        let slot = self.alloca(func, frame, vec_ty.clone());
        self.store(func, frame, slot, v, false);
        let zero = self.iconst(func, 64, 0);
        let addr = self.scaled_index_addr(func, frame, slot, zero, &fe::parse::Ty::LongLong, elem_size, mir::Ty::int(32)).0;
        self.store(func, frame, addr, lane0, false);
        self.load(func, frame, slot, vec_ty, false)
    }

    fn asm_result_in_rax(&self, func: &mut mir::Function, frame: &mut Frame, template: &str, extra_outputs: Vec<mir::AsmOperand>, inputs: Vec<mir::AsmOperand>) -> mir::Value {
        self.asm_result_in_rax_with_clobbers(func, frame, template, extra_outputs, inputs, &[])
    }

    /// [`Self::asm_result_in_rax`] with a clobber list (see
    /// [`Self::asm_with_clobbers`]).
    fn asm_result_in_rax_with_clobbers(&self, func: &mut mir::Function, frame: &mut Frame, template: &str, extra_outputs: Vec<mir::AsmOperand>, inputs: Vec<mir::AsmOperand>, clobbers: &[&str]) -> mir::Value {
        let slot = self.alloca(func, frame, mir::Ty::int(64));
        let outputs = 1 + extra_outputs.len() as u32;
        let mut operands = vec![mir::AsmOperand {
            value: slot,
            constraint: "=a".into(),
            ty: mir::Ty::int(64),
            tie: None,
            name: None,
        }];
        operands.extend(extra_outputs);
        operands.extend(inputs);
        self.asm_with_clobbers(func, frame, template, outputs, operands, clobbers, true);
        self.load(func, frame, slot, mir::Ty::int(64), false)
    }

    /// One 32-bit half of a 64-bit MIR value: low when `high` is
    /// false, the `LShr`-by-32 half otherwise.
    fn half(&self, func: &mut mir::Function, frame: &mut Frame, v: mir::Value, high: bool) -> mir::Value {
        if !high {
            return self.emit(func, frame, mir::InstKind::Trunc(v), mir::Ty::int(32));
        }
        let shift = self.iconst(func, 64, 32);
        let hi = self.emit(func, frame, mir::InstKind::LShr(v, shift), mir::Ty::int(64));
        self.emit(func, frame, mir::InstKind::Trunc(hi), mir::Ty::int(32))
    }

    /// Reassemble two 32-bit halves into a 64-bit MIR value (the
    /// inverse of [`Self::half`]).
    fn join_halves(&self, func: &mut mir::Function, frame: &mut Frame, lo: mir::Value, hi: mir::Value) -> mir::Value {
        let image = mir::Ty::int(64);
        let wide_lo = self.emit(func, frame, mir::InstKind::Zext(lo), image.clone());
        let wide_hi = self.emit(func, frame, mir::InstKind::Zext(hi), image.clone());
        let shift = self.iconst(func, 64, 32);
        let shifted = self.emit(func, frame, mir::InstKind::Shl(wide_hi, shift), image.clone());
        self.emit(func, frame, mir::InstKind::Or(wide_lo, shifted), image)
    }

    /// The FE type matching a u32/u64 builtin width: the `eflags` and
    /// SSP reads convert to their declared result type.
    fn args_ty_of_width(&self, width: u8) -> fe::parse::Ty {
        if width == 4 {
            fe::parse::Ty::UnsignedInt
        } else {
            fe::parse::Ty::UnsignedLongLong
        }
    }

    /// `min(a, b)` over unsigned MIR integers.
    fn select_min(&self, func: &mut mir::Function, frame: &mut Frame, a: mir::Value, b: mir::Value, ty: mir::Ty) -> mir::Value {
        let cond = self.emit(
            func,
            frame,
            mir::InstKind::Icmp {
                pred: mir::IcmpPred::Ult,
                lhs: a,
                rhs: b,
            },
            mir::Ty::Bool,
        );
        self.emit(
            func,
            frame,
            mir::InstKind::Select {
                cond,
                then_val: a,
                else_val: b,
            },
            ty,
        )
    }

    // ---- SSE2 128-bit integer lanes ----

    /// Widen a lane scalar to 64 bits. Vectors are unsigned bit
    /// containers, so the signedness is the operation's, not the
    /// lane's.
    fn lane_widen(&self, func: &mut mir::Function, frame: &mut Frame, v: mir::Value, signed: bool) -> mir::Value {
        let kind = if signed { mir::InstKind::Sext(v) } else { mir::InstKind::Zext(v) };
        self.emit(func, frame, kind, mir::Ty::int(64))
    }

    /// Store one widened lane of a 16-byte vector, truncating back to
    /// the element width first.
    fn vec128_store_lane(&self, func: &mut mir::Function, frame: &mut Frame, slot: mir::Value, lane: u32, v: mir::Value, elem: &mir::Ty) {
        let v = self.emit(func, frame, mir::InstKind::Trunc(v), elem.clone());
        let idx = self.iconst(func, 64, i128::from(lane));
        let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, elem.size_bytes(), elem.clone()).0;
        self.store(func, frame, addr, v, false);
    }

    /// The element MIR type and lane count of a vector result (any GNU
    /// shape; 8-byte shapes bit-pun to an integer image).
    fn vec128_result(&self, fe_ty: &fe::parse::Ty) -> (mir::Ty, mir::Ty, u32) {
        let shape = self.vector_shape(fe_ty);
        (lower_ty(fe_ty, self.sema, self.interner), shape.elem_mir, shape.lanes)
    }

    /// A vector operand staged through a scratch slot: the value, the
    /// slot, and whether it is a first-class MIR vector (lane reads
    /// take `ExtractLane`) or a bit-punned image (byte-indexed loads).
    fn vec_stage(&self, func: &mut mir::Function, frame: &mut Frame, v: mir::Value, fe_ty: &fe::parse::Ty) -> (mir::Value, mir::Value, bool) {
        let vec_ty = lower_ty(fe_ty, self.sema, self.interner);
        let first_class = vec_ty.as_vector().is_some();
        let slot = self.alloca(func, frame, vec_ty);
        self.store(func, frame, slot, v, false);
        (v, slot, first_class)
    }

    /// Lane `lane` of a staged vector operand.
    fn vec_lane(&self, func: &mut mir::Function, frame: &mut Frame, v: mir::Value, slot: mir::Value, first_class: bool, lane: u32, elem: &mir::Ty) -> mir::Value {
        if first_class {
            return self.emit(func, frame, mir::InstKind::ExtractLane { vec: v, lane: lane as u8 }, elem.clone());
        }
        let idx = self.iconst(func, 64, i128::from(lane));
        let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, elem.size_bytes(), elem.clone()).0;
        self.load(func, frame, addr, elem.clone(), false)
    }

    /// Per-lane map over an integer vector: lanes widen to 64 bits for
    /// the closure, results reassemble through a scratch slot.
    fn vec128_map(&self, func: &mut mir::Function, frame: &mut Frame, v: mir::Value, signed: bool, fe_ty: &fe::parse::Ty, f: &mut dyn FnMut(&mut mir::Function, &mut Frame, mir::Value, u32) -> mir::Value) -> mir::Value {
        let (vec_ty, elem, lanes) = self.vec128_result(fe_ty);
        let (v, v_slot, v_class) = self.vec_stage(func, frame, v, fe_ty);
        let slot = self.alloca(func, frame, vec_ty);
        for lane in 0..lanes {
            let x = self.vec_lane(func, frame, v, v_slot, v_class, lane, &elem);
            let x = self.lane_widen(func, frame, x, signed);
            let r = f(func, frame, x, lane);
            self.vec128_store_lane(func, frame, slot, lane, r, &elem);
        }
        self.load(func, frame, slot, lower_ty(fe_ty, self.sema, self.interner), false)
    }

    /// Per-lane binary map over two integer vectors of the result's
    /// shape.
    #[allow(clippy::too_many_arguments)]
    fn vec128_bin_map(&self, func: &mut mir::Function, frame: &mut Frame, a: mir::Value, b: mir::Value, signed: bool, fe_ty: &fe::parse::Ty, f: &mut dyn FnMut(&mut mir::Function, &mut Frame, mir::Value, mir::Value) -> mir::Value) -> mir::Value {
        let (vec_ty, elem, lanes) = self.vec128_result(fe_ty);
        let (a, a_slot, a_class) = self.vec_stage(func, frame, a, fe_ty);
        let (b, b_slot, b_class) = self.vec_stage(func, frame, b, fe_ty);
        let slot = self.alloca(func, frame, vec_ty);
        for lane in 0..lanes {
            let x = self.vec_lane(func, frame, a, a_slot, a_class, lane, &elem);
            let y = self.vec_lane(func, frame, b, b_slot, b_class, lane, &elem);
            let x = self.lane_widen(func, frame, x, signed);
            let y = self.lane_widen(func, frame, y, signed);
            let r = f(func, frame, x, y);
            self.vec128_store_lane(func, frame, slot, lane, r, &elem);
        }
        self.load(func, frame, slot, lower_ty(fe_ty, self.sema, self.interner), false)
    }

    /// Clamp a widened lane to the signed or unsigned `bits`-wide
    /// range (the saturating pack/add/sub semantics).
    fn lane_clamp(&self, func: &mut mir::Function, frame: &mut Frame, v: mir::Value, bits: u32, signed: bool) -> mir::Value {
        let (min, max) = if signed {
            (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
        } else {
            (0, (1i128 << bits) - 1)
        };
        let lo = self.iconst(func, 64, min);
        let hi = self.iconst(func, 64, max);
        let below = self.emit(func, frame, mir::InstKind::Icmp { pred: mir::IcmpPred::Slt, lhs: v, rhs: lo }, mir::Ty::Bool);
        let v = self.emit(func, frame, mir::InstKind::Select { cond: below, then_val: lo, else_val: v }, mir::Ty::int(64));
        let above = self.emit(func, frame, mir::InstKind::Icmp { pred: mir::IcmpPred::Sgt, lhs: v, rhs: hi }, mir::Ty::Bool);
        self.emit(func, frame, mir::InstKind::Select { cond: above, then_val: hi, else_val: v }, mir::Ty::int(64))
    }
}

/// The MIR float lane type of an SSE packed shape: `ps` spellings
/// carry `f32` lanes, `pd` spellings `f64`.
fn packed_float_ty(ps: bool) -> mir::Ty {
    if ps {
        mir::Ty::Float(mir::FloatKind::F32)
    } else {
        mir::Ty::Float(mir::FloatKind::F64)
    }
}
