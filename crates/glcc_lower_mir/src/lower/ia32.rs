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
            // The MMX pairs stage both operands into fixed `mm`
            // registers through general ones; `emms` would leave the
            // FPU state clean but callers issue it themselves.
            fe::check::Ia32Op::MmxOp { mnemonic } => {
                let a = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let b = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let out = self.asm_result_in_rax(
                    func,
                    frame,
                    &format!("movq %1, %mm0\n\tmovq %2, %mm1\n\t{mnemonic} %mm1, %mm0\n\tmovq %mm0, %0"),
                    vec![],
                    vec![
                        mir::AsmOperand {
                            value: a,
                            constraint: "r".into(),
                            ty: mir::Ty::int(64),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: b,
                            constraint: "r".into(),
                            ty: mir::Ty::int(64),
                            tie: None,
                            name: None,
                        },
                    ],
                );
                self.convert(func, frame, out, &args[0].ty, fe_ty)
            }
            // The count forms shift exactly like the binary pairs;
            // the immediate forms take the count as an `"e"` operand.
            fe::check::Ia32Op::MmxShift { mnemonic, imm } => {
                let v = self.expr_value(module, func, frame, funcs, externs, globals, &args[0]);
                let count = self.expr_value(module, func, frame, funcs, externs, globals, &args[1]);
                let (count_constraint, count_op) = if *imm { ("e", count) } else { ("r", count) };
                let template = if *imm {
                    format!("movq %1, %mm0\n\t{mnemonic} %2, %mm0\n\tmovq %mm0, %0")
                } else {
                    format!("movq %1, %mm0\n\tmovq %2, %mm1\n\t{mnemonic} %mm1, %mm0\n\tmovq %mm0, %0")
                };
                let out = self.asm_result_in_rax(
                    func,
                    frame,
                    &template,
                    vec![],
                    vec![
                        mir::AsmOperand {
                            value: v,
                            constraint: "r".into(),
                            ty: mir::Ty::int(64),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: count_op,
                            constraint: count_constraint.into(),
                            ty: mir::Ty::int(64),
                            tie: None,
                            name: None,
                        },
                    ],
                );
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
                let out = self.asm_result_in_rax(
                    func,
                    frame,
                    "movd %1, %mm0\n\tmovd %2, %mm1\n\tpunpckldq %mm1, %mm0\n\tmovq %mm0, %0",
                    vec![],
                    vec![
                        mir::AsmOperand {
                            value: a,
                            constraint: "r".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                        mir::AsmOperand {
                            value: b,
                            constraint: "r".into(),
                            ty: mir::Ty::int(32),
                            tie: None,
                            name: None,
                        },
                    ],
                );
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
                    &format!("{} %1\n\tsetc %al", if *rand { "rdrand" } else { "rdseed" }),
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
        self.emit_unit(func, frame, mir::InstKind::InlineAsm {
            block: Box::new(mir::AsmBlock {
                template: template.into(),
                operands,
                outputs,
                clobbers: vec![],
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
    fn asm_result_in_rax(&self, func: &mut mir::Function, frame: &mut Frame, template: &str, extra_outputs: Vec<mir::AsmOperand>, inputs: Vec<mir::AsmOperand>) -> mir::Value {
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
        self.asm(func, frame, template, outputs, operands, true);
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
}
