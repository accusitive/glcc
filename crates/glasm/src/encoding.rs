//! The x86-64 encoding engine (reference: nasm's `asm/assemble.c`,
//! reduced to the mainstream integer subset).

use crate::expr::{EvalCtx, RValue};
use crate::insn::{Insn, Operand, Size};
use crate::regs::{Reg, RegClass};

pub struct FixupReq {
    /// Offset of the field within the encoded instruction bytes.
    pub at: usize,
    pub size: u8,
    pub val: RValue,
    pub pcrel: bool,
}

pub struct Encoded {
    pub bytes: Vec<u8>,
    pub fixups: Vec<FixupReq>,
}

#[derive(Default, Clone, Copy)]
struct Rex {
    w: bool,
    r: bool,
    x: bool,
    b: bool,
    /// Emit 0x40 even with no bits set (needed to name spl/bpl/sil/dil).
    force: bool,
}

fn rex_byte(rex: &Rex) -> Option<u8> {
    if rex.w || rex.r || rex.x || rex.b || rex.force {
        Some(0x40 | (rex.w as u8) << 3 | (rex.r as u8) << 2 | (rex.x as u8) << 1 | rex.b as u8)
    } else {
        None
    }
}

struct Enc {
    bytes: Vec<u8>,
    fixups: Vec<FixupReq>,
    /// Where the REX prefix belongs (right before the first opcode byte).
    rex_at: usize,
    rex: Rex,
}

impl Enc {
    fn new() -> Enc {
        Enc { bytes: Vec::new(), fixups: Vec::new(), rex_at: usize::MAX, rex: Rex::default() }
    }

    /// Marks where the first opcode byte lands so the REX prefix can be
    /// spliced in ahead of it.
    fn mark(&mut self) {
        if self.rex_at == usize::MAX {
            self.rex_at = self.bytes.len();
        }
    }

    fn byte(&mut self, b: u8) {
        self.mark();
        self.bytes.push(b);
    }

    fn op0f(&mut self, b: u8) {
        self.byte(0x0f);
        self.bytes.push(b);
    }

    fn osz(&mut self) {
        self.mark();
        self.bytes.push(0x66);
    }

    fn emit_rex(&mut self) {
        if let Some(b) = rex_byte(&self.rex) {
            self.bytes.insert(self.rex_at, b);
            for f in &mut self.fixups {
                f.at += 1;
            }
        }
    }

    fn field_fixup(&mut self, val: RValue, size: Size, pcrel: bool) {
        self.fixups.push(FixupReq { at: self.bytes.len(), size, val, pcrel });
        self.bytes.extend(std::iter::repeat_n(0, size as usize));
    }

    fn field(&mut self, val: RValue, size: Size) {
        match val.as_const() {
            Some(v) => self.imm_const(v, size),
            None => self.field_fixup(val, size, false),
        }
    }

    fn imm_const(&mut self, v: i64, size: Size) {
        for i in 0..size as u16 {
            self.bytes.push((v >> (i * 8)) as u8);
        }
    }
}

// ---------------------------------------------------------------------

fn cc_code(name: &str) -> Option<u8> {
    Some(match name {
        "o" => 0,
        "no" => 1,
        "b" | "c" | "nae" => 2,
        "ae" | "nb" | "nc" => 3,
        "e" | "z" => 4,
        "ne" | "nz" => 5,
        "be" | "na" => 6,
        "a" | "nbe" => 7,
        "s" => 8,
        "ns" => 9,
        "p" | "pe" => 10,
        "np" | "po" => 11,
        "l" | "nge" => 12,
        "ge" | "nl" => 13,
        "le" | "ng" => 14,
        "g" | "nle" => 15,
        _ => return None,
    })
}

fn digit_of(alu: &str) -> Option<u8> {
    Some(match alu {
        "add" => 0,
        "or" => 1,
        "adc" => 2,
        "sbb" => 3,
        "and" => 4,
        "sub" => 5,
        "xor" => 6,
        "cmp" => 7,
        _ => return None,
    })
}

fn is_acc(reg: &Reg) -> bool {
    reg.is_gpr() && reg.code == 0
}

/// Evaluated memory operand.
struct MemN {
    size: Option<Size>,
    base: Option<Reg>,
    index: Option<Reg>,
    scale: u8,
    disp: RValue,
}

impl MemN {
    fn is_rip_rel(&self) -> bool {
        self.base.map_or(false, |b| b.class == RegClass::Rip) && self.index.is_none()
    }
}

fn check_mem(m: &MemN) -> Result<(), String> {
    for r in [&m.base, &m.index].into_iter().flatten() {
        if r.class == RegClass::Rip {
            continue;
        }
        if r.size != 8 {
            return Err(
                "invalid effective address: address sizes in 64-bit mode are 64-bit".into()
            );
        }
    }
    if m.is_rip_rel() && m.index.is_some() {
        return Err("cannot scale rip".into());
    }
    Ok(())
}

fn apply_size(enc: &mut Enc, size: Size) {
    match size {
        2 => enc.osz(),
        8 => enc.rex.w = true,
        _ => {}
    }
}

/// Byte register that needs the REX prefix present to be named.
fn needs_rex_for(r: &Reg) -> bool {
    r.needs_rex()
}

/// Emits ModRM/SIB/displacement for `mem` with `reg` in the reg field.
fn emit_modrm_mem(enc: &mut Enc, reg: u8, mem: &MemN) -> Result<(), String> {
    enc.rex.r = reg >= 8;
    let reg = reg & 7;
    if mem.is_rip_rel() {
        // ModRM 00/101 + disp32 relative to the end of the instruction.
        enc.byte(0x05 | reg << 3);
        enc.field_fixup(mem.disp.clone(), 4, true);
        return Ok(());
    }
    if mem.base.is_none() && mem.index.is_none() {
        // Absolute disp32 address: ModRM 00/101, no SIB.
        enc.byte(0x05 | reg << 3);
        enc.field(mem.disp.clone(), 4);
        return Ok(());
    }
    let need_sib = mem.index.is_some()
        || mem.base.map_or(false, |b| b.is_gpr() && b.code & 7 == 4);
    let base_code5 = mem.base.map_or(false, |b| b.is_gpr() && b.code & 7 == 5);
    let (mode, disp_size) = match mem.disp.as_const() {
        Some(0) if mem.base.is_some() && !base_code5 => (0u8, 0u8),
        Some(v) if (-128..=127).contains(&v) => (1, 1),
        _ => (2, 4),
    };
    if need_sib {
        enc.rex.x = mem.index.map_or(false, |r| r.code >= 8);
        enc.rex.b = mem.base.map_or(false, |b| b.code >= 8);
        enc.byte(mode << 6 | reg << 3 | 0x04);
        let idx = mem.index.map_or(4u8, |r| r.code & 7);
        let (base_bits, sib_disp_size) = match mem.base {
            // SIB base=101 with mod=00 means "no base": emit disp32.
            None => (5u8, 4u8),
            Some(b) => (b.code & 7, disp_size),
        };
        enc.byte((mem.scale.trailing_zeros() as u8) << 6 | idx << 3 | base_bits);
        if sib_disp_size > 0 {
            enc.field(mem.disp.clone(), sib_disp_size);
        }
    } else {
        let base = mem.base.unwrap();
        enc.rex.b = base.code >= 8 || needs_rex_for(&base);
        enc.byte(mode << 6 | reg << 3 | (base.code & 7));
        if disp_size > 0 {
            enc.field(mem.disp.clone(), disp_size);
        }
    }
    Ok(())
}

fn norm_mem(op: &Operand, ctx: &EvalCtx) -> Result<MemN, String> {
    let Operand::Mem { size, base, index, scale, disp } = op else {
        unreachable!()
    };
    let m = MemN {
        size: *size,
        base: *base,
        index: *index,
        scale: *scale,
        disp: disp.eval(ctx)?,
    };
    check_mem(&m)?;
    Ok(m)
}

fn reg_of(op: &Operand) -> Option<Reg> {
    match op {
        Operand::Reg(r) => Some(*r),
        _ => None,
    }
}

/// The "spec" of an ALU-style operation: every form the group shares.
struct AluSpec {
    /// r/m8, r8 and r8, r/m8 (test reuses 84/85 symmetrically).
    rr8: u8,
    /// r/m, r (16/32/64).
    rrm: u8,
    /// r, r/m (16/32/64).
    mrm: u8,
    acc8: u8,
    accz: u8,
    imm8_op: u8,
    imm_op: u8,
    digit: u8,
}

fn alu_spec(name: &str) -> Option<AluSpec> {
    let digit = digit_of(name)?;
    let b = digit * 8;
    Some(AluSpec {
        rr8: b,
        rrm: b + 1,
        mrm: b + 3,
        acc8: b + 4,
        accz: b + 5,
        imm8_op: 0x80,
        imm_op: 0x81,
        digit,
    })
}

fn test_spec() -> AluSpec {
    AluSpec { rr8: 0x84, rrm: 0x85, mrm: 0x84, acc8: 0xa8, accz: 0xa9, imm8_op: 0xf6, imm_op: 0xf7, digit: 0 }
}

fn enc_alu(enc: &mut Enc, insn: &Insn, ctx: &EvalCtx, spec: &AluSpec) -> Result<(), String> {
    let a = insn
        .operands
        .first()
        .ok_or_else(|| format!("`{}' requires operands", insn.mnemonic))?;
    let b = insn
        .operands
        .get(1)
        .ok_or_else(|| format!("`{}' requires more operands", insn.mnemonic))?;
    if insn.operands.len() > 2 {
        return Err(format!("too many operands for `{}'", insn.mnemonic));
    }

    if let (Some(d), Some(s)) = (reg_of(a), reg_of(b)) {
        if d.size != s.size {
            return Err("mismatched operand sizes".into());
        }
        enc.rex.r = s.code >= 8 || needs_rex_for(&s);
        enc.rex.b = d.code >= 8 || needs_rex_for(&d);
        if d.size == 1 {
            enc.byte(spec.rr8);
        } else {
            apply_size(enc, d.size);
            enc.byte(spec.rrm);
        }
        enc.byte(0xc0 | (s.code & 7) << 3 | (d.code & 7));
        return Ok(());
    }

    // Short accumulator-immediate forms; for widths with an imm8 form
    // (0x83) that is shorter still, fall through to the generic path.
    if let (Some(d), Operand::Imm(e)) = (reg_of(a), b) {
        if is_acc(&d) {
            let v = e.eval(ctx)?;
            if d.size == 1 {
                enc.byte(spec.acc8);
                enc.field(v, 1);
                return Ok(());
            }
            let fits8 = matches!(v.as_const(), Some(c) if (-128..=127).contains(&c));
            if !fits8 {
                apply_size(enc, d.size);
                enc.byte(spec.accz);
                enc.field(v, d.size.min(4));
                return Ok(());
            }
            // Falls through: reg dst + imm via 0x83.
        }
    }

    if let (Operand::Mem { .. }, Some(s)) = (a, reg_of(b)) {
        let mem = norm_mem(a, ctx)?;
        let size = mem.size.unwrap_or(s.size);
        if size != s.size {
            return Err("mismatched operand sizes".into());
        }
        apply_size(enc, size);
        enc.byte(if size == 1 { spec.rr8 } else { spec.rrm });
        enc.rex.r = s.code >= 8 || needs_rex_for(&s);
        emit_modrm_mem(enc, s.code, &mem)?;
        return Ok(());
    }

    if let (Some(d), Operand::Mem { .. }) = (reg_of(a), b) {
        let mem = norm_mem(b, ctx)?;
        let size = mem.size.unwrap_or(d.size);
        if size != d.size {
            return Err("mismatched operand sizes".into());
        }
        apply_size(enc, size);
        enc.byte(if size == 1 { spec.rr8 } else { spec.mrm });
        emit_modrm_mem(enc, d.code, &mem)?;
        return Ok(());
    }

    let v = match b {
        Operand::Imm(e) => e.eval(ctx)?,
        _ => return Err("invalid operand combination".into()),
    };
    let (size, has_reg): (Size, Option<Reg>) = match a {
        Operand::Reg(r) => (r.size, Some(*r)),
        Operand::Mem { .. } => {
            let mem = norm_mem(a, ctx)?;
            (mem.size.ok_or("operation size not specified")?, None)
        }
        _ => return Err("invalid operand combination".into()),
    };
    let emit_rm = |enc: &mut Enc| -> Result<(), String> {
        match has_reg {
            Some(r) => {
                enc.rex.b = r.code >= 8 || needs_rex_for(&r);
                enc.byte(0xc0 | spec.digit << 3 | (r.code & 7));
                Ok(())
            }
            None => {
                let mem = norm_mem(a, ctx)?;
                emit_modrm_mem(enc, spec.digit, &mem)
            }
        }
    };
    if size == 1 {
        enc.byte(spec.imm8_op);
        emit_rm(enc)?;
        enc.field(v, 1);
    } else {
        apply_size(enc, size);
        let fits8 = matches!(v.as_const(), Some(c) if (-128..=127).contains(&c));
        if fits8 {
            enc.byte(0x83);
            emit_rm(enc)?;
            enc.imm_const(v.as_const().unwrap(), 1);
        } else {
            enc.byte(spec.imm_op);
            emit_rm(enc)?;
            enc.field(v, size.min(4));
        }
    }
    Ok(())
}

fn branch_disp(target: &RValue, ctx: &EvalCtx, len: i64) -> Option<i64> {
    let t = match target {
        RValue::Const(c) => *c,
        RValue::SectVal { sect, off } if *sect == ctx.cur_sect => *off,
        _ => return None,
    };
    Some(t - (ctx.cur_off + len))
}

/// Emits a rel field of `size` bytes toward `target`. The displacement
/// is relative to the end of the field, i.e. the end of the instruction.
fn emit_rel(enc: &mut Enc, target: RValue, ctx: &EvalCtx, size: Size) {
    let field_end = ctx.cur_off + enc.bytes.len() as i64 + size as i64;
    match target {
        RValue::Const(c) => enc.imm_const(c.wrapping_sub(field_end), size),
        RValue::SectVal { sect, off } if sect == ctx.cur_sect => {
            enc.imm_const(off.wrapping_sub(field_end), size)
        }
        v => enc.field_fixup(v, size, true),
    }
}

pub fn encode(insn: &Insn, ctx: &EvalCtx, force_long_branch: bool) -> Result<Encoded, String> {
    let mut enc = Enc::new();
    encode_into(&mut enc, insn, ctx, force_long_branch)?;
    enc.emit_rex();
    Ok(Encoded { bytes: enc.bytes, fixups: enc.fixups })
}

fn encode_into(enc: &mut Enc, insn: &Insn, ctx: &EvalCtx, force_long: bool) -> Result<(), String> {
    let m = insn.mnemonic.as_str();
    let ops = &insn.operands;

    if m == "jmp" || m.strip_prefix('j').and_then(cc_code).is_some() {
        let cc = if m == "jmp" { 0u8 } else { cc_code(&m[1..]).unwrap() };
        let Some(Operand::Imm(e)) = ops.first() else {
            return Err(format!("invalid operand for `{m}'"));
        };
        let target = e.eval(ctx)?;
        let short_fits = branch_disp(&target, ctx, 2).map_or(true, |d| (-128..=127).contains(&d));
        let use_short = !force_long && short_fits;
        if use_short {
            if m == "jmp" {
                enc.byte(0xeb);
            } else {
                enc.byte(0x70 + cc);
            }
            emit_rel(enc, target, ctx, 1);
        } else {
            if m == "jmp" {
                enc.byte(0xe9);
            } else {
                enc.op0f(0x80 + cc);
            }
            emit_rel(enc, target, ctx, 4);
        }
        return Ok(());
    }

    match m {
        "add" | "or" | "adc" | "sbb" | "and" | "sub" | "xor" | "cmp" => {
            enc_alu(enc, insn, ctx, &alu_spec(m).unwrap())
        }
        "test" => enc_alu(enc, insn, ctx, &test_spec()),
        "mov" => enc_mov(enc, insn, ctx),
        "lea" => {
            let d = reg_of(ops.first().ok_or("lea requires operands")?)
                .ok_or("lea requires a register destination")?;
            let Operand::Mem { .. } = ops.get(1).ok_or("lea requires a memory source")? else {
                return Err("lea requires a memory source".into());
            };
            let mem = norm_mem(&ops[1], ctx)?;
            apply_size(enc, d.size);
            enc.byte(0x8d);
            emit_modrm_mem(enc, d.code, &mem)
        }
        "movzx" | "movsx" | "movsxd" => enc_movx(enc, insn, ctx),
        "inc" | "dec" => {
            let digit = if m == "inc" { 0 } else { 1 };
            match ops.first() {
                Some(Operand::Reg(r)) => {
                    apply_size(enc, r.size);
                    enc.byte(if r.size == 1 { 0xfe } else { 0xff });
                    enc.rex.b = r.code >= 8 || needs_rex_for(r);
                    enc.byte(0xc0 | digit << 3 | (r.code & 7));
                    Ok(())
                }
                Some(Operand::Mem { .. }) => {
                    let mem = norm_mem(&ops[0], ctx)?;
                    let size = mem.size.ok_or("operation size not specified")?;
                    apply_size(enc, size);
                    enc.byte(if size == 1 { 0xfe } else { 0xff });
                    emit_modrm_mem(enc, digit, &mem)
                }
                _ => Err(format!("invalid operand for `{m}'")),
            }
        }
        "push" => {
            match ops.first() {
                Some(Operand::Reg(r)) if r.is_gpr() => {
                    if r.size == 2 {
                        enc.osz();
                    }
                    enc.rex.b = r.code >= 8;
                    enc.byte(0x50 | (r.code & 7));
                    Ok(())
                }
                Some(Operand::Imm(e)) => {
                    let v = e.eval(ctx)?;
                    match v.as_const() {
                        Some(c) if (-128..=127).contains(&c) => {
                            enc.byte(0x6a);
                            enc.imm_const(c, 1);
                            Ok(())
                        }
                        _ => {
                            enc.byte(0x68);
                            enc.field(v, 4);
                            Ok(())
                        }
                    }
                }
                Some(Operand::Mem { .. }) => {
                    let mem = norm_mem(&ops[0], ctx)?;
                    let size = mem.size.ok_or("operation size not specified")?;
                    if size == 2 {
                        enc.osz();
                    } else if size != 8 {
                        return Err("unsupported push operand size".into());
                    }
                    enc.byte(0xff);
                    emit_modrm_mem(enc, 6, &mem)
                }
                _ => Err("invalid operand for `push'".into()),
            }
        }
        "pop" => match ops.first() {
            Some(Operand::Reg(r)) if r.is_gpr() => {
                if r.size == 2 {
                    enc.osz();
                }
                enc.rex.b = r.code >= 8;
                enc.byte(0x58 | (r.code & 7));
                Ok(())
            }
            Some(Operand::Mem { .. }) => {
                let mem = norm_mem(&ops[0], ctx)?;
                let size = mem.size.ok_or("operation size not specified")?;
                if size == 2 {
                    enc.osz();
                } else if size != 8 {
                    return Err("unsupported pop operand size".into());
                }
                enc.byte(0x8f);
                emit_modrm_mem(enc, 0, &mem)
            }
            _ => Err("invalid operand for `pop'".into()),
        },
        "call" => match ops.first() {
            Some(Operand::Imm(e)) => {
                let target = e.eval(ctx)?;
                enc.byte(0xe8);
                emit_rel(enc, target, ctx, 4);
                Ok(())
            }
            Some(Operand::Reg(r)) if r.is_64() => {
                enc.rex.b = r.code >= 8;
                enc.byte(0xff);
                enc.byte(0xd0 | (r.code & 7));
                Ok(())
            }
            Some(Operand::Mem { .. }) => {
                let mem = norm_mem(&ops[0], ctx)?;
                enc.byte(0xff);
                emit_modrm_mem(enc, 2, &mem)
            }
            _ => Err("invalid operand for `call'".into()),
        },
        "ret" => {
            match ops.first() {
                None => {
                    enc.byte(0xc3);
                    Ok(())
                }
                Some(Operand::Imm(e)) => {
                    let v = e.eval(ctx)?;
                    enc.byte(0xc2);
                    enc.field(v, 2);
                    Ok(())
                }
                _ => Err("invalid operand for `ret'".into()),
            }
        }
        "leave" => {
            enc.byte(0xc9);
            Ok(())
        }
        "nop" => {
            enc.byte(0x90);
            Ok(())
        }
        "syscall" => {
            enc.op0f(0x05);
            Ok(())
        }
        "int3" => {
            enc.byte(0xcc);
            Ok(())
        }
        "ud2" => {
            enc.op0f(0x0b);
            Ok(())
        }
        "int" => {
            let Operand::Imm(e) = ops.first().ok_or("int requires an immediate")? else {
                return Err("int requires an immediate".into());
            };
            let v = e.eval(ctx)?;
            enc.byte(0xcd);
            enc.field(v, 1);
            Ok(())
        }
        "cdq" | "cqo" => {
            if m == "cqo" {
                enc.rex.w = true;
            }
            enc.byte(0x99);
            Ok(())
        }
        "cwde" | "cdqe" => {
            if m == "cdqe" {
                enc.rex.w = true;
            }
            enc.byte(0x98);
            Ok(())
        }
        "neg" | "not" => group_f7(enc, ops, ctx, if m == "neg" { 3 } else { 2 }),
        "mul" | "div" | "idiv" => group_f7(
            enc,
            ops,
            ctx,
            match m {
                "mul" => 4,
                "div" => 6,
                _ => 7,
            },
        ),
        "imul" => enc_imul(enc, insn, ctx),
        "shl" | "sal" | "shr" | "sar" | "rol" | "ror" | "rcl" | "rcr" => enc_shift(enc, insn, ctx),
        "movabs" => {
            let d = reg_of(ops.first().ok_or("movabs requires operands")?)
                .ok_or("movabs destination must be a register")?;
            if d.size != 8 {
                return Err("movabs requires a 64-bit register".into());
            }
            let Operand::Imm(e) = ops.get(1).ok_or("movabs requires two operands")? else {
                return Err("movabs requires an immediate source".into());
            };
            let v = e.eval(ctx)?;
            enc.rex.w = true;
            enc.rex.b = d.code >= 8;
            enc.byte(0xb8 | (d.code & 7));
            enc.field(v, 8);
            Ok(())
        }
        _ if m.starts_with("set") => {
            let cc = cc_code(&m[3..]).ok_or_else(|| format!("unknown instruction `{m}'"))?;
            let dst = ops.first().ok_or("setcc requires an operand")?;
            match dst {
                Operand::Reg(r) if r.size == 1 => {
                    enc.rex.b = r.code >= 8 || needs_rex_for(r);
                    enc.op0f(0x90 + cc);
                    enc.byte(0xc0 | (r.code & 7));
                    Ok(())
                }
                Operand::Mem { .. } => {
                    let mem = norm_mem(dst, ctx)?;
                    enc.op0f(0x90 + cc);
                    emit_modrm_mem(enc, 0, &mem)
                }
                _ => Err("setcc requires a byte register or memory operand".into()),
            }
        }
        _ => Err(format!("unknown or unsupported instruction `{m}'")),
    }
}

fn group_f7(enc: &mut Enc, ops: &[Operand], ctx: &EvalCtx, digit: u8) -> Result<(), String> {
    let dst = ops.first().ok_or("instruction requires an operand")?;
    match dst {
        Operand::Reg(r) => {
            apply_size(enc, r.size);
            enc.byte(if r.size == 1 { 0xf6 } else { 0xf7 });
            enc.rex.b = r.code >= 8 || needs_rex_for(r);
            enc.byte(0xc0 | digit << 3 | (r.code & 7));
            Ok(())
        }
        Operand::Mem { .. } => {
            let mem = norm_mem(dst, ctx)?;
            let size = mem.size.ok_or("operation size not specified")?;
            apply_size(enc, size);
            enc.byte(if size == 1 { 0xf6 } else { 0xf7 });
            emit_modrm_mem(enc, digit, &mem)
        }
        _ => Err("invalid operand".into()),
    }
}

fn enc_imul(enc: &mut Enc, insn: &Insn, ctx: &EvalCtx) -> Result<(), String> {
    let ops = &insn.operands;
    if ops.len() == 1 {
        return group_f7(enc, ops, ctx, 5);
    }
    let d = reg_of(ops.first().ok_or("imul requires operands")?)
        .ok_or("imul destination must be a register")?;
    if ops.len() == 3 {
        let Operand::Imm(e) = &ops[2] else {
            return Err("third imul operand must be an immediate".into());
        };
        let v = e.eval(ctx)?;
        let (src_reg, src_mem) = match &ops[1] {
            Operand::Reg(r) => (Some(*r), None),
            Operand::Mem { .. } => (None, Some(norm_mem(&ops[1], ctx)?)),
            _ => return Err("invalid imul operands".into()),
        };
        apply_size(enc, d.size);
        let fits8 = matches!(v.as_const(), Some(c) if (-128..=127).contains(&c));
        enc.byte(if fits8 { 0x6b } else { 0x69 });
        enc.rex.r = d.code >= 8;
        match (src_reg, src_mem) {
            (Some(r), _) => {
                if d.size != r.size {
                    return Err("mismatched operand sizes".into());
                }
                enc.rex.b = r.code >= 8;
                enc.byte(0xc0 | (d.code & 7) << 3 | (r.code & 7));
            }
            (_, Some(mem)) => {
                emit_modrm_mem(enc, d.code, &mem)?;
            }
            _ => unreachable!(),
        }
        enc.field(v, 1.min(if fits8 { 1 } else { 4 }));
        return Ok(());
    }
    match ops.get(1) {
        Some(Operand::Reg(s)) => {
            if d.size != s.size {
                return Err("mismatched operand sizes".into());
            }
            apply_size(enc, d.size);
            enc.op0f(0xaf);
            enc.rex.r = d.code >= 8;
            enc.rex.b = s.code >= 8;
            enc.byte(0xc0 | (d.code & 7) << 3 | (s.code & 7));
            Ok(())
        }
        Some(Operand::Mem { .. }) => {
            let mem = norm_mem(&ops[1], ctx)?;
            let size = mem.size.unwrap_or(d.size);
            if size != d.size {
                return Err("mismatched operand sizes".into());
            }
            apply_size(enc, d.size);
            enc.op0f(0xaf);
            emit_modrm_mem(enc, d.code, &mem)
        }
        _ => Err("invalid operands for `imul'".into()),
    }
}

fn enc_shift(enc: &mut Enc, insn: &Insn, ctx: &EvalCtx) -> Result<(), String> {
    let ops = &insn.operands;
    let digit: u8 = match insn.mnemonic.as_str() {
        "rol" => 0,
        "ror" => 1,
        "rcl" => 2,
        "rcr" => 3,
        "shl" | "sal" => 4,
        "shr" => 5,
        "sar" => 7,
        _ => unreachable!(),
    };
    let dst = ops.first().ok_or("shift requires operands")?;
    let (size, by_reg): (Size, Option<Reg>) = match dst {
        Operand::Reg(r) => (r.size, None),
        Operand::Mem { .. } => {
            let mem = norm_mem(dst, ctx)?;
            (mem.size.ok_or("operation size not specified")?, None)
        }
        _ => return Err("invalid shift operand".into()),
    };
    let opbase = match ops.get(1) {
        Some(Operand::Imm(e)) => {
            let v = e.eval(ctx)?;
            match v.as_const() {
                Some(1) => {
                    if size == 1 {
                        0xd0
                    } else {
                        0xd1
                    }
                }
                Some(c) if (0..=255).contains(&c) => {
                    if size == 1 {
                        0xc0
                    } else {
                        0xc1
                    }
                }
                _ => return Err("shift count must be a constant 0..255".into()),
            }
        }
        Some(Operand::Reg(r)) if r.is_gpr() && r.size == 1 && r.code == 1 => {
            if size == 1 {
                0xd2
            } else {
                0xd3
            }
        }
        _ => return Err("invalid shift count".into()),
    };
    let _ = by_reg;
    apply_size(enc, size);
    match dst {
        Operand::Reg(r) => {
            enc.rex.b = r.code >= 8 || needs_rex_for(r);
            enc.byte(opbase);
            enc.byte(0xc0 | digit << 3 | (r.code & 7));
        }
        _ => {
            let mem = norm_mem(dst, ctx)?;
            enc.byte(opbase);
            emit_modrm_mem(enc, digit, &mem)?;
        }
    }
    if opbase == 0xc0 || opbase == 0xc1 {
        let Some(Operand::Imm(e)) = ops.get(1) else {
            return Err("shift count missing".into());
        };
        let v = e.eval(ctx)?;
        enc.field(v, 1);
    }
    Ok(())
}

fn enc_movx(enc: &mut Enc, insn: &Insn, ctx: &EvalCtx) -> Result<(), String> {
    let ops = &insn.operands;
    let m = insn.mnemonic.as_str();
    let d = reg_of(ops.first().ok_or("requires operands")?)
        .ok_or("destination must be a register")?;
    if d.size == 1 {
        return Err("destination must be at least 16 bits".into());
    }
    let src_size: Size = match ops.get(1) {
        Some(Operand::Reg(r)) => r.size,
        Some(Operand::Mem { size: Some(s), .. }) => *s,
        Some(Operand::Mem { size: None, .. }) => {
            return Err("operation size not specified".into())
        }
        _ => return Err("invalid operands".into()),
    };
    let opbyte = match (m, src_size) {
        ("movzx", 1) => 0xb6,
        ("movzx", 2) => 0xb7,
        ("movsx", 1) => 0xbe,
        ("movsx", 2) => 0xbf,
        ("movsxd", 4) => 0x63,
        ("movsxd", _) => return Err("movsxd requires a 32-bit source".into()),
        _ => return Err(format!("invalid source size for `{m}'")),
    };
    if m == "movsxd" {
        if d.size != 8 {
            return Err("movsxd requires a 64-bit destination".into());
        }
        enc.rex.w = true;
    } else {
        apply_size(enc, d.size);
    }
    enc.rex.r = d.code >= 8;
    match ops.get(1) {
        Some(Operand::Reg(r)) => {
            if opbyte == 0x63 {
                enc.byte(opbyte);
            } else {
                enc.op0f(opbyte);
            }
            enc.rex.b = r.code >= 8 || needs_rex_for(r);
            enc.byte(0xc0 | (d.code & 7) << 3 | (r.code & 7));
            Ok(())
        }
        _ => {
            let mem = norm_mem(&ops[1], ctx)?;
            if opbyte == 0x63 {
                enc.byte(opbyte);
            } else {
                enc.op0f(opbyte);
            }
            emit_modrm_mem(enc, d.code, &mem)
        }
    }
}

fn enc_mov(enc: &mut Enc, insn: &Insn, ctx: &EvalCtx) -> Result<(), String> {
    let a = insn.operands.first().ok_or("mov requires operands")?;
    let b = insn.operands.get(1).ok_or("mov requires two operands")?;
    if insn.operands.len() > 2 {
        return Err("too many operands for `mov'".into());
    }

    // mov reg, imm with nasm's narrowing: a 64-bit register loaded with
    // a u32-range constant uses the zero-extending B8 imm32 form, a
    // negative i32-range constant the sign-extending C7 /0 form.
    if let (Operand::Reg(r), Operand::Imm(e)) = (a, b) {
        let v = e.eval(ctx)?;
        let size = r.size;
        if size == 8 {
            enc.rex.b = r.code >= 8;
            match v.as_const() {
                Some(c) if (0..=0xffff_ffff).contains(&c) => {
                    enc.byte(0xb8 | (r.code & 7));
                    enc.imm_const(c, 4);
                }
                Some(c) if (-0x8000_0000..=-1).contains(&c) => {
                    enc.rex.w = true;
                    enc.byte(0xc7);
                    enc.byte(0xc0 | (r.code & 7));
                    enc.imm_const(c, 4);
                }
                _ => {
                    enc.rex.w = true;
                    enc.byte(0xb8 | (r.code & 7));
                    enc.field(v, 8);
                }
            }
            return Ok(());
        }
        if size == 2 {
            enc.osz();
        }
        enc.rex.b = r.code >= 8 || needs_rex_for(r);
        enc.byte(if size == 1 { 0xb0 } else { 0xb8 } | (r.code & 7));
        enc.field(v, size.min(4));
        return Ok(());
    }

    if let (Some(d), Some(s)) = (reg_of(a), reg_of(b)) {
        if d.size != s.size {
            return Err("mismatched operand sizes".into());
        }
        if d.size == 1 {
            enc.rex.r = s.code >= 8 || needs_rex_for(&s);
            enc.rex.b = d.code >= 8 || needs_rex_for(&d);
            enc.byte(0x88);
        } else {
            apply_size(enc, d.size);
            enc.rex.r = s.code >= 8;
            enc.rex.b = d.code >= 8;
            enc.byte(0x89);
        }
        enc.byte(0xc0 | (s.code & 7) << 3 | (d.code & 7));
        return Ok(());
    }

    if let (Operand::Mem { .. }, Some(s)) = (a, reg_of(b)) {
        let mem = norm_mem(a, ctx)?;
        let size = mem.size.unwrap_or(s.size);
        if size != s.size {
            return Err("mismatched operand sizes".into());
        }
        apply_size(enc, size);
        enc.byte(if size == 1 { 0x88 } else { 0x89 });
        enc.rex.r = s.code >= 8 || needs_rex_for(&s);
        emit_modrm_mem(enc, s.code, &mem)?;
        return Ok(());
    }

    if let (Some(d), Operand::Mem { .. }) = (reg_of(a), b) {
        let mem = norm_mem(b, ctx)?;
        let size = mem.size.unwrap_or(d.size);
        if size != d.size {
            return Err("mismatched operand sizes".into());
        }
        apply_size(enc, size);
        enc.byte(if size == 1 { 0x8a } else { 0x8b });
        emit_modrm_mem(enc, d.code, &mem)?;
        return Ok(());
    }

    if let (Operand::Mem { .. }, Operand::Imm(e)) = (a, b) {
        let mem = norm_mem(a, ctx)?;
        let size = mem.size.ok_or("operation size not specified")?;
        apply_size(enc, size);
        enc.byte(if size == 1 { 0xc6 } else { 0xc7 });
        emit_modrm_mem(enc, 0, &mem)?;
        enc.field(e.eval(ctx)?, if size == 1 { 1 } else { size.min(4) });
        return Ok(());
    }
    Err("invalid operand combination for `mov'".into())
}
