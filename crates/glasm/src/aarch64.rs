//! AArch64 backend: fixed 32-bit instructions, core integer set.
//!
//! Behaviors matched against GNU as (binutils 2.44, native aarch64):
//! - `mov rd, #imm` immediate selection: movz, then movn, then a
//!   logical immediate (orr/orn); immediates needing more than one
//!   instruction are rejected, like gas,
//! - auto-ldur/stur: `ldr [x, #off]` uses the scaled imm12 form only
//!   when 0 <= off < 4096*size and off % size == 0, the unscaled imm9
//!   form otherwise,
//! - relocation policy: same-section branches resolve in place with no
//!   relocation; everything else (externs, other sections, adrp) keeps
//!   an unpatched field and relocates against the section for local
//!   symbols,
//! - relocation types: JUMP26/CALL26/CONDBR19/TSTBR14 for branches,
//!   ADR_PREL_LO21/ADR_PREL_PG_HI21 (+ GOT_PAGE) for adr/adrp,
//!   ADD_ABS_LO12_NC for `add :lo12:`, LDSTn_ABS_LO12_NC for scaled
//!   `ldr/str :lo12:`, ABS32/ABS64 for data.
//!
//! Not yet supported: literal pools (`ldr =expr`), branch relaxation
//! (out-of-range displacements error), MSR/MRS, FP/SIMD.

use crate::arch::Backend;
use crate::assemble::{Fixup, Reloc, Section, Symbol, Target};
use crate::encoding::{Encoded, FieldLayout, FixupReq};
use crate::expr::{EvalCtx, Expr, ModKind, RValue};
use crate::gas::GTok;
use object::{Architecture, Endianness};
use object::elf::{R_AARCH64_ABS32, R_AARCH64_ABS64, R_AARCH64_ADD_ABS_LO12_NC, R_AARCH64_ADR_GOT_PAGE, R_AARCH64_ADR_PREL_LO21, R_AARCH64_ADR_PREL_PG_HI21, R_AARCH64_CALL26, R_AARCH64_CONDBR19, R_AARCH64_JUMP26, R_AARCH64_LD64_GOT_LO12_NC, R_AARCH64_LDST16_ABS_LO12_NC, R_AARCH64_LDST32_ABS_LO12_NC, R_AARCH64_LDST64_ABS_LO12_NC, R_AARCH64_LDST8_ABS_LO12_NC, R_AARCH64_TSTBR14};

// ---------------------------------------------------------------------
// Operand model
// ---------------------------------------------------------------------

/// One AArch64 instruction.
#[derive(Clone, Debug)]
pub struct Insn {
    pub mnemonic: String,
    pub operands: Vec<Operand>,
    pub line: u32,
}

/// Register 31: in add/sub (imm) and mov it is sp, in everything else
/// xzr/wzr; the parser tags which spelling was used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegKind {
    Plain,
    Sp,
    Zr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reg {
    pub num: u8,
    pub wide: bool,
    pub kind: RegKind,
}

impl Reg {
    /// Register field value (sp and xzr both encode as 31).
    fn field(self) -> u8 {
        self.num
    }
    fn plain(num: u8, wide: bool) -> Reg {
        Reg { num, wide, kind: RegKind::Plain }
    }
}

/// Shift/extend suffix on a register operand: `x1, lsl #3`,
/// `w2, sxtw #1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShiftKind {
    Lsl,
    Lsr,
    Asr,
    Ror,
    Uxtw,
    Sxtw,
    Uxtx,
    Sxtx,
}

impl ShiftKind {
    fn from_name(name: &str) -> Option<ShiftKind> {
        Some(match name {
            "lsl" => ShiftKind::Lsl,
            "lsr" => ShiftKind::Lsr,
            "asr" => ShiftKind::Asr,
            "ror" => ShiftKind::Ror,
            "uxtw" => ShiftKind::Uxtw,
            "sxtw" => ShiftKind::Sxtw,
            "uxtx" => ShiftKind::Uxtx,
            "sxtx" => ShiftKind::Sxtx,
            _ => return None,
        })
    }

    fn is_extend(self) -> bool {
        matches!(self, ShiftKind::Uxtw | ShiftKind::Sxtw | ShiftKind::Uxtx | ShiftKind::Sxtx)
    }

    /// The 2-bit shift field of the shifted-register group, or the
    /// 3-bit option field of the extended-register group.
    fn field(self) -> u32 {
        match self {
            ShiftKind::Lsl => 0b00,
            ShiftKind::Lsr => 0b01,
            ShiftKind::Asr => 0b10,
            ShiftKind::Ror => 0b11,
            ShiftKind::Uxtw => 0b010,
            ShiftKind::Uxtx => 0b011,
            ShiftKind::Sxtw => 0b110,
            ShiftKind::Sxtx => 0b111,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemMode {
    Offset,
    PreIndex,
    PostIndex,
}

#[derive(Clone, Debug)]
pub enum MemOff {
    /// Displacement expression (possibly `:lo12:`-modified).
    Expr(Expr),
    /// Register index with extend/shift suffix: `[x1, x2, lsl #3]`.
    RegIndex(Reg, ShiftKind, i64),
}

/// A load/store operand.
#[derive(Clone, Debug)]
pub struct MemOp {
    pub base: Reg,
    pub off: Option<MemOff>,
    pub mode: MemMode,
}

#[derive(Clone, Debug)]
pub enum Operand {
    Reg(Reg),
    /// Immediate with an optional `lsl #n` suffix (add/sub imm12, the
    /// movz/movk family).
    Imm { expr: Expr, lsl: Option<Expr> },
    /// Register with a shift/extend suffix.
    Shifted { reg: Reg, kind: ShiftKind, amount: Expr },
    Mem(MemOp),
}

pub fn parse_insn(g: &[GTok], line_no: u32, lineno: usize, numlabels: &crate::gas::NumLabels) -> Result<crate::insn::Stmt, String> {
    let mut t = GToks { g, pos: 0 };
    let GTok::Ident(mn) = t.bump().ok_or("expected instruction")? else {
        return Err("expected instruction".into());
    };
    let mnemonic = mn.to_ascii_lowercase();
    let mut operands = Vec::new();
    if t.pos < t.g.len() {
        loop {
            operands.push(parse_operand(&mut t, lineno, numlabels)?);
            if !t.eat_punct(",") {
                break;
            }
            // `, lsl #3` / `, sxtw` style suffixes are terminal.
            if let Some(kind) = t.peek_shift() {
                if matches!(operands.last(), Some(Operand::Imm { .. })) {
                    if kind != ShiftKind::Lsl {
                        return Err(format!("`{kind:?}' suffix is not valid on immediates"));
                    }
                    t.bump();
                    let lsl = t.expr(lineno, numlabels)?;
                    match operands.last_mut() {
                        Some(Operand::Imm { lsl: slot, .. }) => *slot = Some(lsl),
                        _ => unreachable!(),
                    }
                    break;
                }
                let reg = match operands.last() {
                    Some(Operand::Reg(r)) => *r,
                    _ => return Err(format!("`{kind:?}' suffix needs a register operand")),
                };
                t.bump();
                let amount = if t.peek_is_imm_or_expr() {
                    t.expr(lineno, numlabels)?
                } else if kind.is_extend() {
                    Expr::Int(0)
                } else {
                    return Err("expected shift amount".into());
                };
                *operands.last_mut().unwrap() = Operand::Shifted { reg, kind, amount };
                break;
            }
            if t.peek_is_lsl() {
                // `lsl #n` on an immediate (movz/movk/movn, add/sub imm).
                t.bump();
                let lsl = t.expr(lineno, numlabels)?;
                match operands.last_mut() {
                    Some(Operand::Imm { lsl: slot, .. }) => *slot = Some(lsl),
                    _ => return Err("`lsl' suffix is only valid on immediates".into()),
                }
                break;
            }
        }
    }
    if t.pos != t.g.len() {
        return Err("junk after instruction operands".into());
    }
    Ok(crate::insn::Stmt::Insn(crate::arch::Insn::Aarch64(Insn {
        mnemonic,
        operands,
        line: line_no,
    })))
}

/// Register names: x/w0..30, fp/lr, sp/wsp, xzr/wzr. Bare register
/// names shadow symbols in operand position, like GNU as.
fn reg(name: &str) -> Option<Reg> {
    let lower = name.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix('x') {
        if let Ok(n) = rest.parse::<u8>() {
            return (n <= 30).then_some(Reg::plain(n, true));
        }
        if rest == "zr" {
            return Some(Reg { num: 31, wide: true, kind: RegKind::Zr });
        }
        return None;
    }
    if let Some(rest) = lower.strip_prefix('w') {
        if let Ok(n) = rest.parse::<u8>() {
            return (n <= 30).then_some(Reg::plain(n, false));
        }
        if rest == "zr" {
            return Some(Reg { num: 31, wide: false, kind: RegKind::Zr });
        }
        return None;
    }
    match lower.as_str() {
        "sp" => Some(Reg { num: 31, wide: true, kind: RegKind::Sp }),
        "wsp" => Some(Reg { num: 31, wide: false, kind: RegKind::Sp }),
        "fp" => Some(Reg::plain(29, true)),
        "lr" => Some(Reg::plain(30, true)),
        _ => None,
    }
}

fn parse_operand(t: &mut GToks, lineno: usize, numlabels: &crate::gas::NumLabels) -> Result<Operand, String> {
    match t.peek() {
        Some(GTok::Punct("[")) => parse_mem(t, lineno, numlabels).map(Operand::Mem),
        Some(GTok::Ident(n)) => {
            if let Some(r) = reg(n) {
                // Bare register names shadow symbols in operand position.
                t.bump();
                Ok(Operand::Reg(r))
            } else {
                let (expr, _had_hash) = t.expr_opt(lineno, numlabels)?;
                Ok(Operand::Imm { expr, lsl: None })
            }
        }
        _ => {
            let (expr, _had_hash) = t.expr_opt(lineno, numlabels)?;
            Ok(Operand::Imm { expr, lsl: None })
        }
    }
}

/// `[base]`, `[base, off]`, `[base, off]!`, `[base], off`,
/// `[base, reg, lsl #3]`.
fn parse_mem(t: &mut GToks, lineno: usize, numlabels: &crate::gas::NumLabels) -> Result<MemOp, String> {
    if !t.eat_punct("[") {
        return Err("expected `['".into());
    }
    let base = match t.bump() {
        Some(GTok::Ident(n)) => reg(n).ok_or_else(|| format!("unknown register `{n}'"))?,
        _ => return Err("expected base register".into()),
    };
    let mut off = None;
    if t.eat_punct(",") {
        off = Some(parse_mem_off(t, lineno, numlabels)?);
    }
    if !t.eat_punct("]") {
        return Err("expected `]'".into());
    }
    if t.eat_punct("!") {
        if off.is_none() {
            return Err("pre-indexed addressing needs an offset".into());
        }
        return Ok(MemOp { base, off, mode: MemMode::PreIndex });
    }
    if t.eat_punct(",") {
        let post = t.expr(lineno, numlabels)?;
        return Ok(MemOp { base, off: Some(MemOff::Expr(post)), mode: MemMode::PostIndex });
    }
    Ok(MemOp { base, off, mode: MemMode::Offset })
}

fn parse_mem_off(t: &mut GToks, lineno: usize, numlabels: &crate::gas::NumLabels) -> Result<MemOff, String> {
    // `:lo12:`-style modifiers lex as `:` `name` `:`.
    if let Some(GTok::Punct(":")) = t.peek() {
        let e = t.expr(lineno, numlabels)?;
        return Ok(MemOff::Expr(e));
    }
    if let Some(GTok::Ident(n)) = t.peek()
        && reg(n).is_some()
    {
        let r = reg(n).unwrap();
        t.bump();
        let mut kind = ShiftKind::Lsl;
        let mut amount = 0i64;
        if t.eat_punct(",") {
            let Some(GTok::Ident(sk)) = t.bump() else {
                return Err("expected shift/extend suffix".into());
            };
            kind = ShiftKind::from_name(&sk.to_ascii_lowercase()).ok_or_else(|| format!("unknown shift `{sk}'"))?;
            if t.peek_is_imm_or_expr() {
                let e = t.expr(lineno, numlabels)?;
                amount = match e { Expr::Int(v) => v, _ => return Err("register index shift amount must be constant".into()) };
            }
        }
        return Ok(MemOff::RegIndex(r, kind, amount));
    }
    let e = t.expr(lineno, numlabels)?;
    Ok(MemOff::Expr(e))
}

// ---------------------------------------------------------------------
// Token cursor (shared shape with riscv.rs, adapted for A64 lexing:
// `#` immediates, `:mod:` prefixes, `[`/`]` memory syntax).
// ---------------------------------------------------------------------

struct GToks<'a> {
    g: &'a [GTok],
    pos: usize,
}

impl GToks<'_> {
    fn bump(&mut self) -> Option<&GTok> {
        let t = self.g.get(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn peek(&self) -> Option<&GTok> {
        self.g.get(self.pos)
    }

    fn eat_punct(&mut self, p: &str) -> bool {
        if matches!(self.peek(), Some(GTok::Punct(q)) if *q == p) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn peek_shift(&self) -> Option<ShiftKind> {
        if let Some(GTok::Ident(n)) = self.peek()
            && let Some(k) = ShiftKind::from_name(&n.to_ascii_lowercase())
        {
            return Some(k);
        }
        None
    }

    fn peek_is_lsl(&self) -> bool {
        matches!(self.peek(), Some(GTok::Ident(n)) if n.eq_ignore_ascii_case("lsl"))
    }

    fn peek_is_imm_or_expr(&self) -> bool {
        match self.peek() {
            Some(GTok::Imm) | Some(GTok::Int(_)) | Some(GTok::Ident(_)) => true,
            _ => false,
        }
    }

    /// Expression up to the next `,`, `]` or register; `#` immediate
    /// markers are transparent; a leading `:mod:` becomes Expr::Mod.
    fn expr(&mut self, lineno: usize, numlabels: &crate::gas::NumLabels) -> Result<Expr, String> {
        let (e, _) = self.expr_opt(lineno, numlabels)?;
        Ok(e)
    }

    fn expr_opt(&mut self, lineno: usize, numlabels: &crate::gas::NumLabels) -> Result<(Expr, bool), String> {
        let mut had_hash = false;
        if matches!(self.peek(), Some(GTok::Imm)) {
            had_hash = true;
            self.pos += 1;
        }
        let mut toks = Vec::new();
        // `:lo12:`-style modifiers.
        if matches!(self.peek(), Some(GTok::Punct(":"))) {
            self.pos += 1;
            let Some(GTok::Ident(name)) = self.bump() else {
                return Err("expected relocation modifier after `:'".into());
            };
            let kind = match name.as_str() {
                "lo12" => ModKind::Lo12,
                "got" => ModKind::Got,
                "got_lo12" => ModKind::GotLo12,
                _ => return Err(format!("unknown relocation modifier `:{name}:'")),
            };
            if !self.eat_punct(":") {
                return Err("expected `:' after relocation modifier".into());
            }
            let mut inner_toks = Vec::new();
            self.collect_expr_toks(&mut inner_toks, lineno, numlabels)?;
            let inner = eval_toks(inner_toks)?;
            return Ok((Expr::Mod { kind, inner: Box::new(inner) }, had_hash));
        }
        self.collect_expr_toks(&mut toks, lineno, numlabels)?;
        Ok((eval_toks(toks)?, had_hash))
    }

    fn collect_expr_toks(&mut self, toks: &mut Vec<crate::parse::Tok>, lineno: usize, numlabels: &crate::gas::NumLabels) -> Result<(), String> {
        while let Some(tok) = self.peek() {
            match tok {
                GTok::Punct(",") | GTok::Punct("]") => break,
                GTok::Reg(_) => break,
                GTok::Ident(n) => {
                    toks.push(crate::parse::Tok::Ident(n.clone()));
                    self.pos += 1;
                }
                GTok::Int(v) => {
                    toks.push(crate::parse::Tok::Int(*v));
                    self.pos += 1;
                }
                GTok::NumRef(d, fwd) => {
                    let name = if *fwd { numlabels.next_at(lineno + 1, *d) } else { numlabels.last_before(lineno, *d) }
                        .ok_or_else(|| format!("no `{d}' label defined in that direction"))?;
                    toks.push(crate::parse::Tok::Ident(name));
                    self.pos += 1;
                }
                GTok::Punct(p) => {
                    toks.push(crate::parse::Tok::Punct(p));
                    self.pos += 1;
                }
                GTok::Imm => {
                    self.pos += 1;
                }
                GTok::Star => {
                    toks.push(crate::parse::Tok::Punct("*"));
                    self.pos += 1;
                }
                GTok::Str(_) => break,
            }
        }
        if toks.is_empty() {
            return Err("expected expression".into());
        }
        Ok(())
    }
}

fn eval_toks(toks: Vec<crate::parse::Tok>) -> Result<Expr, String> {
    let mut t = crate::parse::Toks::new(toks);
    crate::parse::parse_expr(&mut t, 0)
}

// ---------------------------------------------------------------------
// Encoding: format constructors
// ---------------------------------------------------------------------

fn sf(wide: bool) -> u32 {
    if wide { 1 << 31 } else { 0 }
}

/// Arithmetic (imm / shifted-reg / extended-reg) group bases.
fn arith_base(wide: bool, is_sub: bool, sets_flags: bool) -> u32 {
    let mut w = sf(wide) | 0b01011 << 24;
    if is_sub {
        w |= 1 << 30;
    }
    if sets_flags {
        w |= 1 << 29;
    }
    w
}

/// ADD/SUB (immediate): sh[22], imm12[21:10].
fn add_imm_word(wide: bool, is_sub: bool, sets_flags: bool, sh: u32, imm: u32, rn: u8, rd: u8) -> u32 {
    let mut w = sf(wide) | 0b100010 << 23;
    if is_sub {
        w |= 1 << 30;
    }
    if sets_flags {
        w |= 1 << 29;
    }
    w | sh << 22 | imm << 10 | (rn as u32) << 5 | rd as u32
}

/// ADD/SUB (shifted register): shift[23:22], N[21]=0.
/// ADD/SUB (extended register): N[21]=1, option[15:13], imm3[12:10].
fn arith_reg_word(wide: bool, is_sub: bool, sets_flags: bool, shift_field: u32, is_extend: bool, rm: u8, amount: u32, rn: u8, rd: u8) -> u32 {
    let base = arith_base(wide, is_sub, sets_flags);
    if is_extend {
        base | 1 << 21 | shift_field << 13 | (amount & 7) << 10 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32
    } else {
        base | shift_field << 22 | amount << 10 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32
    }
}

/// Logical (shifted register): opc at [30:29], N at [21].
fn logical_reg_word(wide: bool, opc: u32, n: bool, shift: u32, rm: u8, imm6: u32, rn: u8, rd: u8) -> u32 {
    sf(wide) | opc << 29 | 0b01010 << 24 | shift << 22 | (n as u32) << 21 | (rm as u32) << 16 | imm6 << 10 | (rn as u32) << 5 | rd as u32
}

/// Logical (immediate): opc[30:29], N[22], immr[21:16], imms[15:10].
fn logical_imm_word(wide: bool, opc: u32, n: u32, immr: u32, imms: u32, rn: u8, rd: u8) -> u32 {
    sf(wide) | opc << 29 | 0b100100 << 23 | n << 22 | immr << 16 | imms << 10 | (rn as u32) << 5 | rd as u32
}

/// Move wide: opc at [30:29] (movn=00, movz=10, movk=11), hw[22:21].
fn movw_word(wide: bool, opc: u32, hw: u32, imm16: u32, rd: u8) -> u32 {
    sf(wide) | opc << 29 | 0b100101 << 23 | hw << 21 | imm16 << 5 | rd as u32
}

const NOP: u32 = 0xd503201f;

struct Enc {
    bytes: Vec<u8>,
    fixups: Vec<FixupReq>,
}

impl Enc {
    fn push(&mut self, word: u32) {
        self.bytes.extend_from_slice(&word.to_le_bytes());
    }

}

pub struct A64Backend;

impl Backend for A64Backend {
    fn encode(&self, insn: &crate::arch::Insn, ctx: &EvalCtx, _relax_state: u8) -> Result<Encoded, String> {
        let crate::arch::Insn::Aarch64(a) = insn else {
            return Err("not an AArch64 instruction".into());
        };
        let mut enc = Enc {
            bytes: Vec::new(),
            fixups: Vec::new(),
        };
        encode_one(&mut enc, a, ctx)?;
        Ok(Encoded {
            bytes: enc.bytes,
            fixups: enc.fixups,
        })
    }

    fn finalize_all(&self, sections: &mut [Section], _symbols: &mut Vec<Symbol>, fixups: Vec<Fixup>, relocs: &mut Vec<Reloc>) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        for f in &fixups {
            if let Err(msgs) = finalize_fixup(&mut sections[f.section].data, f, relocs) {
                errors.extend(msgs);
            }
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    fn relax_step(&self, _fixup: &Fixup, _delta: i64, _state: u8) -> Option<u8> {
        None
    }

    fn object_arch(&self) -> Architecture {
        Architecture::Aarch64
    }

    fn endianness(&self) -> Endianness {
        Endianness::Little
    }

    fn text_fill(&self) -> &'static [u8] {
        &[0x1f, 0x20, 0x03, 0xd5] // nop
    }
}

fn encode_one(enc: &mut Enc, insn: &Insn, ctx: &EvalCtx) -> Result<(), String> {
    let mn = insn.mnemonic.as_str();
    let ops = &insn.operands;

    // b.cond / other dotted suffixes.
    if let Some(cond_name) = mn.strip_prefix("b.") {
        let cond = cond_code(cond_name).ok_or_else(|| format!("unknown condition `{cond_name}'"))?;
        expect_ops(ops, 1, mn)?;
        return emit_branch(enc, ops, Some(cond), false, ctx);
    }

    match mn {
        "nop" => {
            expect_ops(ops, 0, mn)?;
            enc.push(NOP);
        }
        "add" | "adds" | "sub" | "subs" => {
            let is_sub = mn == "sub" || mn == "subs";
            let sets_flags = mn.starts_with("adds") || mn.starts_with("subs");
            arith(enc, mn, ops, ctx, is_sub, sets_flags, 0)?;
        }
        "cmp" => arith(enc, mn, ops, ctx, true, true, 31)?,
        "cmn" => arith(enc, mn, ops, ctx, false, true, 31)?,
        "neg" | "negs" => {
            expect_ops(ops, 2, mn)?;
            let rm = reg_of(&ops[1], mn)?;
            let rd = reg_of(&ops[0], mn)?;
            let is_sub = true;
            let sets_flags = mn == "negs";
            match (&ops[1], &ops[0]) {
                (Operand::Shifted { reg, kind, amount }, _) => {
                    enc.push(arith_reg_word(rd.wide, is_sub, sets_flags, shift_field(*kind)?, kind.is_extend(), reg.field(), amount_const(amount, mn)? as u32, 31, rd.field()));
                }
                (Operand::Reg(_), _) => {
                    enc.push(arith_reg_word(rd.wide, is_sub, sets_flags, 0, false, rm.field(), 0, 31, rd.field()));
                }
                _ => return Err(err_op(mn)),
            }
        }
        "mov" => mov(enc, mn, ops, ctx)?,
        "movz" | "movk" | "movn" => {
            expect_ops(ops, 2, mn)?;
            let rd = reg_of(&ops[0], mn)?;
            let (expr, lsl) = match &ops[1] {
                Operand::Imm { expr, lsl } => (expr, lsl),
                _ => return Err(err_op(mn)),
            };
            let imm = imm_const_val(expr, None, ctx, mn)?;
            if !(0..=0xffff).contains(&imm) {
                return Err(format!("{mn}: immediate {imm} out of range"));
            }
            let hw = match lsl {
                Some(e) => {
                    let sh = imm_const(&e, ctx, mn)?;
                    if sh % 16 != 0 || sh >= if rd.wide { 64 } else { 32 } {
                        return Err(format!("{mn}: `lsl' amount {sh} out of range"));
                    }
                    sh / 16
                }
                None => 0,
            };
            let opc = match mn {
                "movn" => 0b00,
                "movz" => 0b10,
                _ => 0b11,
            };
            enc.push(movw_word(rd.wide, opc, hw as u32, imm as u32, rd.field()));
        }
        "mvn" => {
            expect_ops(ops, 2, mn)?;
            let rd = reg_of(&ops[0], mn)?;
            match &ops[1] {
                Operand::Reg(r) => {
                    enc.push(logical_reg_word(rd.wide, 0b01, true, 0, r.field(), 0, 31, rd.field()));
                }
                Operand::Shifted { reg, kind, amount } => {
                    if kind.is_extend() {
                        return Err(format!("{mn}: extends are not valid on mov"));
                    }
                    enc.push(logical_reg_word(rd.wide, 0b01, true, shift_field(*kind)?, reg.field(), amount_const(amount, mn)? as u32, 31, rd.field()));
                }
                Operand::Imm { expr, lsl } => {
                    let c = imm_const_val(expr, lsl.as_ref(), ctx, mn)?;
                    let (n, immr, imms) = encode_logical_imm(c as u64, rd.wide).ok_or_else(|| format!("{mn}: invalid logical immediate {c:#x}"))?;
                    enc.push(logical_imm_word(rd.wide, 0b01, n, immr, imms, 31, rd.field()));
                }
                _ => return Err(err_op(mn)),
            }
        }
        "and" | "orr" | "eor" | "ands" | "bic" | "orn" | "eon" => {
            let (opc, n) = match mn {
                "and" => (0b00, false),
                "orr" => (0b01, false),
                "eor" => (0b10, false),
                "ands" => (0b11, false),
                "bic" => (0b00, true),
                "orn" => (0b01, true),
                "eon" => (0b10, true),
                _ => unreachable!(),
            };
            expect_ops(ops, 3, mn)?;
            let rd = reg_of(&ops[0], mn)?;
            let rn = reg_of(&ops[1], mn)?;
            match &ops[2] {
                Operand::Reg(r) => {
                    enc.push(logical_reg_word(rd.wide, opc, n, 0, r.field(), 0, rn.field(), rd.field()));
                }
                Operand::Shifted { reg, kind, amount } => {
                    if kind.is_extend() {
                        return Err(format!("{mn}: extends are not valid on logical operations"));
                    }
                    enc.push(logical_reg_word(rd.wide, opc, n, shift_field(*kind)?, reg.field(), amount_const(amount, mn)? as u32, rn.field(), rd.field()));
                }
                Operand::Imm { expr, lsl } => {
                    if n {
                        return Err(format!("{mn} has no immediate form"));
                    }
                    if lsl.is_some() {
                        return Err(format!("{mn}: immediate does not take an `lsl' suffix"));
                    }
                    let c = imm_const_val(expr, None, ctx, mn)?;
                    let (n32, immr, imms) = encode_logical_imm(c as u64, rd.wide).ok_or_else(|| format!("{mn}: invalid logical immediate {c:#x}"))?;
                    enc.push(logical_imm_word(rd.wide, opc, n32, immr, imms, rn.field(), rd.field()));
                }
                _ => return Err(err_op(mn)),
            }
        }
        "tst" => {
            expect_ops(ops, 2, mn)?;
            let rn = reg_of(&ops[0], mn)?;
            match &ops[1] {
                Operand::Reg(r) => {
                    enc.push(logical_reg_word(rn.wide, 0b11, false, 0, r.field(), 0, rn.field(), 31));
                }
                Operand::Shifted { reg, kind, amount } => {
                    enc.push(logical_reg_word(rn.wide, 0b11, false, shift_field(*kind)?, reg.field(), amount_const(amount, mn)? as u32, rn.field(), 31));
                }
                Operand::Imm { expr, lsl } => {
                    let c = imm_const_val(expr, lsl.as_ref(), ctx, mn)?;
                    let (n, immr, imms) = encode_logical_imm(c as u64, rn.wide).ok_or_else(|| format!("{mn}: invalid logical immediate {c:#x}"))?;
                    enc.push(logical_imm_word(rn.wide, 0b11, n, immr, imms, rn.field(), 31));
                }
                _ => return Err(err_op(mn)),
            }
        }
        "lsl" | "lsr" | "asr" | "ror" => shift_alias(enc, mn, ops, ctx)?,
        "sxtb" | "sxth" | "sxtw" | "uxtb" | "uxth" | "uxtw" => extend_alias(enc, mn, ops)?,
        "mul" | "mneg" | "madd" | "msub" | "sdiv" | "udiv" => arith3(enc, mn, ops)?,
        "cset" | "csetm" | "cinc" | "cinv" | "cneg" => cond_alias(enc, mn, ops)?,
        "csel" | "csinc" | "csinv" | "csneg" => cond_sel(enc, mn, ops)?,
        "ldr" | "ldrb" | "ldrh" | "ldrsb" | "ldrsh" | "ldrsw" | "str" | "strb" | "strh" | "ldur" | "stur" | "ldurb" | "sturb" | "ldurh" | "sturh" | "ldursb" | "ldursh" | "ldursw" => ldst(enc, mn, ops, ctx)?,
        "ldp" | "stp" => ldp_stp(enc, mn, ops, ctx)?,
        "adr" | "adrp" => adr(enc, mn, ops, ctx)?,
        "b" | "bl" => {
            expect_ops(ops, 1, mn)?;
            emit_branch(enc, ops, None, mn == "bl", ctx)?;
        }
        "br" | "blr" | "ret" => {
            if mn == "ret" && ops.len() > 1 {
                return Err(err_op(mn));
            }
            if mn != "ret" {
                expect_ops(ops, 1, mn)?;
            }
            let rn = match ops.first() {
                Some(Operand::Reg(r)) => r.field(),
                None if mn == "ret" => 30,
                _ => return Err(err_op(mn)),
            };
            enc.push(match mn {
                "br" => 0xd61f0000,
                "blr" => 0xd63f0000,
                _ => 0xd65f0000,
            } | (rn as u32) << 5);
        }
        "cbz" | "cbnz" => {
            expect_ops(ops, 2, mn)?;
            let rt = reg_of(&ops[0], mn)?;
            let mut word = if rt.wide { 0xb4000000 } else { 0x34000000 };
            if mn == "cbnz" {
                word |= 1 << 24;
            }
            emit_imm19(enc, word, rt.field(), &ops[1], ctx)?;
        }
        "tbz" | "tbnz" => {
            expect_ops(ops, 3, mn)?;
            let rt = reg_of(&ops[0], mn)?;
            let bit = match &ops[1] {
                Operand::Imm { expr, lsl: None } => imm_const_val(expr, None, ctx, mn)?,
                _ => return Err(err_op(mn)),
            };
            let max = if rt.wide { 63 } else { 31 };
            if bit < 0 || bit > max {
                return Err(format!("{mn}: bit {bit} out of range"));
            }
            let mut word = (0x36000000u32) | (((bit >> 5) as u32) << 31) | (((bit & 31) as u32) << 19);
            if mn == "tbnz" {
                word |= 1 << 24;
            }
            emit_imm14(enc, word, rt.field(), &ops[2], ctx)?;
        }
        "dmb" | "dsb" | "isb" => barrier(enc, mn, ops)?,
        "ldxr" | "stxr" | "ldar" | "stlr" => ldstx(enc, mn, ops)?,
        _ => return Err(format!("unknown or unsupported instruction `{mn}'")),
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Per-family encoders
// ---------------------------------------------------------------------

/// add/sub/adds/subs/cmp/cmn with immediate, plain-register and
/// shifted/extended-register second operands. `fixed_rd` substitutes
/// a fixed destination (xzr for cmp/cmn/neg).
fn arith(enc: &mut Enc, mn: &str, ops: &[Operand], ctx: &EvalCtx, is_sub: bool, sets_flags: bool, fixed_rd: u8) -> Result<(), String> {
    let (rd, rn_op, arg_op) = match fixed_rd {
        0 => {
            expect_ops(ops, 3, mn)?;
            (reg_of(&ops[0], mn)?, &ops[1], &ops[2])
        }
        _ => {
            expect_ops(ops, 2, mn)?;
            let wide = reg_of(&ops[0], mn)?.wide;
            (Reg { num: 31, wide, kind: RegKind::Zr }, &ops[0], &ops[1])
        }
    };
    let rn = reg_of(rn_op, mn)?;
    if rn.wide != rd.wide {
        return Err(format!("{mn}: operand widths differ"));
    }
    match arg_op {
        Operand::Imm { expr, lsl } => {
            let imm = match expr.eval_mod(ctx)? {
                (None, RValue::Const(c)) => c,
                (Some(ModKind::Lo12), v) => {
                    let word = add_imm_word(rn.wide, is_sub, sets_flags, 0, 0, rn.field(), rd.field());
                    enc.push(word);
                    enc.fixups.push(FixupReq { at: 0, size: 4, val: v, pcrel: false, rel_base: 0, layout: FieldLayout::A64Imm12 });
                    return Ok(());
                }
                (kind, _) => return Err(format!("{mn}: immediate must be constant{}", if kind.is_some() { " (only :lo12: is allowed)" } else { "" })),
            };
            let (imm12, sh) = match lsl {
                Some(e) => {
                    let sh = imm_const(&e, ctx, mn)?;
                    if sh != 0 && sh != 12 {
                        return Err(format!("{mn}: `lsl' amount must be 0 or 12"));
                    }
                    (imm, (sh != 0) as u32)
                }
                None => (imm, 0),
            };
            if imm12 < 0 || imm12 > 0xfff {
                return Err(format!("{mn}: immediate {imm12} out of range"));
            }
            enc.push(add_imm_word(rn.wide, is_sub, sets_flags, sh, imm12 as u32, rn.field(), rd.field()));
        }
        Operand::Reg(r) => {
            if r.wide != rn.wide {
                return Err(format!("{mn}: operand widths differ"));
            }
            enc.push(arith_reg_word(rn.wide, is_sub, sets_flags, 0, false, r.field(), 0, rn.field(), rd.field()));
        }
        Operand::Shifted { reg, kind, amount } => {
            let is_extend = kind.is_extend();
            let amount = amount_const(amount, mn)?;
            if is_extend {
                if matches!(kind, ShiftKind::Uxtw | ShiftKind::Sxtw) && reg.wide {
                    return Err(format!("{mn}: {kind:?} needs a w register"));
                }
                if matches!(kind, ShiftKind::Uxtx | ShiftKind::Sxtx) && !reg.wide {
                    return Err(format!("{mn}: {kind:?} needs an x register"));
                }
                if amount > 4 {
                    return Err(format!("{mn}: extend amount {amount} out of range"));
                }
            } else {
                if reg.wide != rn.wide {
                    return Err(format!("{mn}: operand widths differ"));
                }
                if *kind == ShiftKind::Ror {
                    return Err(format!("{mn}: ror is not valid on {mn}"));
                }
                let max = if rn.wide { 63 } else { 31 };
                if amount > max {
                    return Err(format!("{mn}: shift amount {amount} out of range"));
                }
            }
            enc.push(arith_reg_word(rn.wide, is_sub, sets_flags, shift_field(*kind)?, is_extend, reg.field(), amount as u32, rn.field(), rd.field()));
        }
        _ => return Err(err_op(mn)),
    }
    Ok(())
}

/// The `mov` family: register moves (sp forms become add #0), and the
/// immediate selection cascade.
fn mov(enc: &mut Enc, mn: &str, ops: &[Operand], ctx: &EvalCtx) -> Result<(), String> {
    expect_ops(ops, 2, mn)?;
    let rd = reg_of(&ops[0], mn)?;
    match &ops[1] {
        Operand::Reg(r) => {
            if r.wide != rd.wide {
                return Err(format!("{mn}: operand widths differ"));
            }
            match (r.kind, rd.kind) {
                (RegKind::Plain, RegKind::Plain) | (RegKind::Plain, RegKind::Zr) | (RegKind::Zr, RegKind::Plain) | (RegKind::Zr, RegKind::Zr) => {
                    enc.push(logical_reg_word(rd.wide, 0b01, false, 0, r.field(), 0, 31, rd.field()));
                }
                (RegKind::Sp, _) | (_, RegKind::Sp) => {
                    // mov x0, sp / mov sp, x0 -> add rd, sp, #0.
                    enc.push(add_imm_word(rd.wide, false, false, 0, 0, r.field(), rd.field()));
                }
            }
        }
        Operand::Shifted { reg, kind, amount } => {
            if kind.is_extend() {
                return Err(format!("{mn}: extends are not valid on mov"));
            }
            enc.push(logical_reg_word(rd.wide, 0b01, false, shift_field(*kind)?, reg.field(), amount_const(amount, mn)? as u32, 31, rd.field()));
        }
        Operand::Imm { expr, lsl } => return mov_imm(enc, mn, rd, expr, lsl.clone(), ctx),
        _ => return Err(err_op(mn)),
    }
    Ok(())
}

/// mov rd, #imm: movz, then movn, then a logical immediate, then a
/// movz+movk chain — matching GNU as's selection.
fn mov_imm(enc: &mut Enc, mn: &str, rd: Reg, expr: &Expr, lsl: Option<Expr>, ctx: &EvalCtx) -> Result<(), String> {
    let c = match expr.eval(ctx)? {
        RValue::Const(c) => c,
        _ => return Err(format!("{mn}: immediate must be constant")),
    };
    let width = if rd.wide { 64 } else { 32 };
    let mask = if rd.wide { u64::MAX } else { 0xffff_ffff };
    let v = if c < 0 { (c as u64).wrapping_neg().wrapping_neg() } else { c as u64 };
    let v = v & mask;
    // Explicit `lsl #n` pins the movz/movn field and shift.
    if let Some(e) = lsl {
        let sh = imm_const(&e, ctx, mn)?;
        if sh % 16 != 0 || sh >= width {
            return Err(format!("{mn}: `lsl' amount {sh} out of range"));
        }
        let hw = sh / 16;
        if v > 0xffff {
            return Err(format!("{mn}: immediate {v:#x} does not fit in 16 bits"));
        }
        enc.push(movw_word(rd.wide, 0b10, hw as u32, v as u32, rd.field()));
        return Ok(());
    }
    // movz: value is a 16-bit field at some shift.
    if let Some(hw) = movw_hw(v, width as u64) {
        enc.push(movw_word(rd.wide, 0b10, hw, (v >> (hw * 16)) as u32, rd.field()));
        return Ok(());
    }
    // movn: complement is a 16-bit field at some shift.
    let nv = !v & mask;
    if let Some(hw) = movw_hw(nv, width as u64) {
        enc.push(movw_word(rd.wide, 0b00, hw, (nv >> (hw * 16)) as u32, rd.field()));
        return Ok(());
    }
    // Logical immediate (orr rd, xzr, #imm).
    if let Some((n, immr, imms)) = encode_logical_imm(v, rd.wide) {
        enc.push(logical_imm_word(rd.wide, 0b01, n, immr, imms, 31, rd.field()));
        return Ok(());
    }
    Err(format!("{mn}: immediate cannot be moved by a single instruction"))
}

fn movw_hw(v: u64, width: u64) -> Option<u32> {
    if v == 0 {
        return Some(0);
    }
    for hw in 0..(width / 16) {
        if v & !(0xffffu64 << (hw * 16)) == 0 {
            return Some(hw as u32);
        }
    }
    None
}

fn shift_alias(enc: &mut Enc, mn: &str, ops: &[Operand], ctx: &EvalCtx) -> Result<(), String> {
    expect_ops(ops, 3, mn)?;
    let rd = reg_of(&ops[0], mn)?;
    let rn = reg_of(&ops[1], mn)?;
    if rd.wide != rn.wide {
        return Err(format!("{mn}: operand widths differ"));
    }
    let n = imm_const(&match &ops[2] {
        Operand::Imm { expr, lsl: None } => expr.clone(),
        _ => return Err(err_op(mn)),
    }, ctx, mn)?;
    let width = if rd.wide { 64 } else { 32 };
    if n < 0 || n >= width {
        return Err(format!("{mn}: shift amount {n} out of range"));
    }
    match mn {
        "lsl" => {
            // ubfm rd, rn, width-n, width-1-n
            let immr = ((width - n) % width) as u32;
            enc.push(bitfield_word(rd.wide, 0b10, rd.wide, immr, width as u32 - 1 - n as u32, rn.field(), rd.field()));
        }
        "lsr" => {
            enc.push(bitfield_word(rd.wide, 0b10, rd.wide, n as u32, width as u32 - 1, rn.field(), rd.field()));
        }
        "asr" => {
            enc.push(bitfield_word(rd.wide, 0b00, rd.wide, n as u32, width as u32 - 1, rn.field(), rd.field()));
        }
        "ror" => {
            // extr rd, rn, rn, #n (N = 1 for x registers)
            let word = sf(rd.wide) | 0b100111 << 23 | (rd.wide as u32) << 22 | (rn.field() as u32) << 16 | (n as u32) << 10 | (rn.field() as u32) << 5 | rd.field() as u32;
            enc.push(word);
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// Bitfield group: opc[30:29] (sbfm=00, bfm=01, ubfm=10), N[22],
/// immr[21:16], imms[15:10].
fn bitfield_word(wide: bool, opc: u32, n: bool, immr: u32, imms: u32, rn: u8, rd: u8) -> u32 {
    sf(wide) | opc << 29 | 0b100110 << 23 | (n as u32) << 22 | immr << 16 | imms << 10 | (rn as u32) << 5 | rd as u32
}

fn extend_alias(enc: &mut Enc, mn: &str, ops: &[Operand]) -> Result<(), String> {
    expect_ops(ops, 2, mn)?;
    let rd = reg_of(&ops[0], mn)?;
    let rn = reg_of(&ops[1], mn)?;
    // sxtb/sxth/sxtw/uxtb/uxth/uxtw are sbfm/ubfm with imms = width-1
    // of the *source* class and immr = 0.
    let (opc, imms) = match mn {
        "sxtb" => (0b00, 7),
        "sxth" => (0b00, 15),
        "sxtw" => (0b00, 31),
        "uxtb" => (0b10, 7),
        "uxth" => (0b10, 15),
        "uxtw" => (0b10, 31),
        _ => unreachable!(),
    };
    if mn == "sxtw" || mn == "uxtw" {
        if rn.wide {
            return Err(format!("{mn}: source must be a w register"));
        }
        if !rd.wide {
            // uxtw Wd, Wn is a plain 32-bit move.
            enc.push(logical_reg_word(false, 0b01, false, 0, rn.field(), 0, 31, rd.field()));
            return Ok(());
        }
        enc.push(bitfield_word(true, opc, true, 0, imms, rn.field(), rd.field()));
    } else {
        if rn.wide {
            return Err(format!("{mn}: source must be a w register"));
        }
        if mn == "uxtb" || mn == "uxth" {
            // UXTB/UXTH only have the W form.
            enc.push(bitfield_word(false, opc, false, 0, imms, rn.field(), rd.field()));
        } else {
            enc.push(bitfield_word(rd.wide, opc, rd.wide, 0, imms, rn.field(), rd.field()));
        }
    }
    Ok(())
}

fn arith3(enc: &mut Enc, mn: &str, ops: &[Operand]) -> Result<(), String> {
    let rd = reg_of(&ops[0], mn)?;
    let sf_bit = if rd.wide { 0x80000000 } else { 0 };
    match (mn, ops) {
        ("mul" | "mneg" | "madd" | "msub", _) => {
            // MADD family: op bits at [23:21] = 000 with the sub flag
            // at bit 15; mul/mneg set Ra = xzr.
            let (is_sub, ra, rn, rm) = match (mn, &ops[1..]) {
                ("mul" | "mneg", [Operand::Reg(rn), Operand::Reg(rm)]) => (mn == "mneg", 31u32, rn.field(), rm.field()),
                ("madd" | "msub", [Operand::Reg(rn), Operand::Reg(rm), Operand::Reg(ra)]) => (mn == "msub", ra.field() as u32, rn.field(), rm.field()),
                _ => return Err(err_op(mn)),
            };
            enc.push(sf_bit | 0x1b000000 | ((is_sub as u32) << 15) | (rm as u32) << 16 | ra << 10 | (rn as u32) << 5 | rd.field() as u32);
        }
        ("sdiv" | "udiv", [_, Operand::Reg(rn), Operand::Reg(rm)]) => {
            let lo = if mn == "sdiv" { 0xc00 } else { 0x800 };
            enc.push(sf_bit | 0x1ac00000 | lo | (rm.field() as u32) << 16 | (rn.field() as u32) << 5 | rd.field() as u32);
        }
        _ => return Err(err_op(mn)),
    }
    Ok(())
}

fn cond_code(name: &str) -> Option<u32> {
    Some(match name {
        "eq" => 0,
        "ne" => 1,
        "cs" | "hs" => 2,
        "cc" | "lo" => 3,
        "mi" => 4,
        "pl" => 5,
        "vs" => 6,
        "vc" => 7,
        "hi" => 8,
        "ls" => 9,
        "ge" => 10,
        "lt" => 11,
        "gt" => 12,
        "le" => 13,
        "al" => 14,
        "nv" => 15,
        _ => return None,
    })
}

fn invert_cond(cond: u32) -> u32 {
    cond ^ 1
}

fn cond_alias(enc: &mut Enc, mn: &str, ops: &[Operand]) -> Result<(), String> {
    let (op, use_rn) = match mn {
        // cset/csetm = csinc/csinv with Rm = Rn = xzr.
        "cset" => {
            expect_ops(ops, 2, mn)?;
            (0b01, false)
        }
        "csetm" => {
            expect_ops(ops, 2, mn)?;
            (0b10, false)
        }
        "cinc" | "cinv" | "cneg" => {
            expect_ops(ops, 3, mn)?;
            (match mn {
                "cinc" => 0b01,
                "cinv" => 0b10,
                _ => 0b11,
            }, true)
        }
        _ => unreachable!(),
    };
    let rd = reg_of(&ops[0], mn)?;
    let rn = if use_rn {
        let r = reg_of(&ops[1], mn)?;
        if r.wide != rd.wide {
            return Err(format!("{mn}: operand widths differ"));
        }
        r
    } else {
        Reg { num: 31, wide: rd.wide, kind: RegKind::Zr }
    };
    let cond_name = match ops.last() {
        Some(Operand::Imm { expr: Expr::Sym(name), lsl: None }) if cond_code(name).is_some() => name.as_str(),
        _ => return Err(err_op(mn)),
    };
    let cond = invert_cond(cond_code(cond_name).unwrap());
    let rm = if use_rn { rn.field() } else { 31 };
    enc.push(cond_sel_word(rd.wide, op, rm, cond, rn.field(), rd.field()));
    Ok(())
}

fn cond_sel(enc: &mut Enc, mn: &str, ops: &[Operand]) -> Result<(), String> {
    expect_ops(ops, 4, mn)?;
    let rd = reg_of(&ops[0], mn)?;
    let rn = reg_of(&ops[1], mn)?;
    let rm = reg_of(&ops[2], mn)?;
    if rd.wide != rn.wide || rd.wide != rm.wide {
        return Err(format!("{mn}: operand widths differ"));
    }
    let cond_name = match &ops[3] {
        Operand::Imm { expr: Expr::Sym(name), lsl: None } if cond_code(name).is_some() => name.as_str(),
        _ => return Err(err_op(mn)),
    };
    let cond = cond_code(cond_name).unwrap();
    let op = match mn {
        "csel" => 0b00,
        "csinc" => 0b01,
        "csinv" => 0b10,
        "csneg" => 0b11,
        _ => unreachable!(),
    };
    enc.push(cond_sel_word(rd.wide, op, rm.field(), cond, rn.field(), rd.field()));
    Ok(())
}

/// Conditional select group: the 2-bit op is split across bit 30
/// (csinv/csneg) and bit 10 (csinc/csneg).
fn cond_sel_word(wide: bool, op: u32, rm: u8, cond: u32, rn: u8, rd: u8) -> u32 {
    sf(wide) | (op >> 1) << 30 | 0b110101 << 23 | (op & 1) << 10 | (rm as u32) << 16 | cond << 12 | (rn as u32) << 5 | rd as u32
}

/// Loads/stores. `ldur/stur`-family mnemonics force the unscaled form;
/// plain `ldr/str` auto-select scaled vs unscaled like GNU as.
fn ldst(enc: &mut Enc, mn: &str, ops: &[Operand], ctx: &EvalCtx) -> Result<(), String> {
    let force_unscaled = mn.contains("ur");
    let base_mn = match mn {
        "ldur" => "ldr",
        "stur" => "str",
        "ldurb" => "ldrb",
        "sturb" => "strb",
        "ldurh" => "ldrh",
        "sturh" => "strh",
        "ldursb" => "ldrsb",
        "ldursh" => "ldrsh",
        "ldursw" => "ldrsw",
        _ => mn,
    };
    let is_load = base_mn.starts_with("ld");
    expect_ops(ops, 2, mn)?;
    let rt = reg_of(&ops[0], mn)?;
    let Operand::Mem(mem) = &ops[1] else {
        return Err(format!("{mn}: needs a memory operand"));
    };
    // size (log2) and opc for the load/store group.
    let (size_log2, opc) = match base_mn {
        "strb" => (0, 0),
        "ldrb" => (0, 1),
        "strh" => (1, 0),
        "ldrh" => (1, 1),
        "str" => if rt.wide { (3, 0) } else { (2, 0) },
        "ldr" => if rt.wide { (3, 1) } else { (2, 1) },
        "ldrsb" => if rt.wide { (0, 2) } else { (0, 3) },
        "ldrsh" => if rt.wide { (1, 2) } else { (1, 3) },
        "ldrsw" => (2, 2),
        _ => return Err(format!("unreachable ldst `{mn}'")),
    };
    if base_mn == "ldrsw" && !rt.wide {
        return Err("ldrsw needs an x register".into());
    }
    if !is_load && base_mn != "str" && base_mn != "strb" && base_mn != "strh" {
        return Err(format!("unreachable ldst `{mn}'"));
    }
    let size = 1u64 << size_log2;

    let scaled_word = |imm12: u32| -> u32 {
        (size_log2 as u32) << 30 | 0b111 << 27 | 1 << 24 | opc << 22 | imm12 << 10 | (mem.base.field() as u32) << 5 | rt.field() as u32
    };
    let unscaled_word = |imm9: u32, bit10: u32| -> u32 {
        (size_log2 as u32) << 30 | 0b111 << 27 | opc << 22 | (imm9 & 0x1ff) << 12 | bit10 << 10 | (mem.base.field() as u32) << 5 | rt.field() as u32
    };

    match (&mem.off, mem.mode) {
        (Some(MemOff::Expr(e)), MemMode::Offset) => {
            let (kind, val) = e.eval_mod(ctx)?;
            match (kind, val) {
                (None, RValue::Const(c)) => {
                    if force_unscaled {
                        if !(-256..=255).contains(&c) {
                            return Err(format!("{mn}: offset {c} out of range"));
                        }
                        enc.push(unscaled_word(c as u32, 0));
                    } else if c >= 0 && (c as u64) < 4096 * size && c % (size as i64) == 0 {
                        enc.push(scaled_word((c as u64 / size) as u32));
                    } else if (-256..=255).contains(&c) {
                        enc.push(unscaled_word(c as u32, 0));
                    } else {
                        return Err(format!("{mn}: offset {c} out of range"));
                    }
                }
                (Some(ModKind::Lo12), v) => {
                    if force_unscaled {
                        return Err(format!("{mn}: :lo12: needs the scaled form"));
                    }
                    enc.push(scaled_word(0));
                    enc.fixups.push(FixupReq { at: 0, size: 4, val: v, pcrel: false, rel_base: 0, layout: FieldLayout::A64LdstImm12 });
                }
                (Some(ModKind::GotLo12), v) => {
                    enc.push(scaled_word(0));
                    enc.fixups.push(FixupReq { at: 0, size: 4, val: v, pcrel: false, rel_base: 0, layout: FieldLayout::A64GotLo12 });
                }
                _ => return Err(format!("{mn}: offset must be constant or :lo12:")),
            }
        }
        (Some(MemOff::Expr(e)), MemMode::PreIndex | MemMode::PostIndex) => {
            let c = imm_const(e, ctx, mn)?;
            if !(-256..=255).contains(&c) {
                return Err(format!("{mn}: offset {c} out of range"));
            }
            enc.push(unscaled_word(c as u32, if mem.mode == MemMode::PreIndex { 3 } else { 1 }));
        }
        (Some(MemOff::RegIndex(rm, kind, amount)), MemMode::Offset) => {
            let amount = *amount;
            if rm.wide != matches!(kind, ShiftKind::Lsl | ShiftKind::Uxtx | ShiftKind::Sxtx) {
                return Err(format!("{mn}: index register width does not match the extend type"));
            }
            if amount != 0 && amount != size_log2 as i64 {
                return Err(format!("{mn}: index shift must be 0 or {size_log2}"));
            }
            let opt = match kind {
                ShiftKind::Lsl | ShiftKind::Uxtx => 0b011,
                ShiftKind::Uxtw => 0b010,
                ShiftKind::Sxtw => 0b110,
                ShiftKind::Sxtx => 0b111,
                _ => return Err(format!("{mn}: invalid index extend type")),
            };
            let s = (amount != 0) as u32;
            enc.push((size_log2 as u32) << 30 | 0b111 << 27 | opc << 22 | 1 << 21 | (rm.field() as u32) << 16 | opt << 13 | s << 12 | 0b10 << 10 | (mem.base.field() as u32) << 5 | rt.field() as u32);
        }
        (None, MemMode::Offset) => enc.push(scaled_word(0)),
        _ => return Err(err_op(mn)),
    }
    Ok(())
}

fn ldp_stp(enc: &mut Enc, mn: &str, ops: &[Operand], ctx: &EvalCtx) -> Result<(), String> {
    expect_ops(ops, 3, mn)?;
    let rt1 = reg_of(&ops[0], mn)?;
    let rt2 = reg_of(&ops[1], mn)?;
    if rt1.wide != rt2.wide {
        return Err(format!("{mn}: operand widths differ"));
    }
    let Operand::Mem(mem) = &ops[2] else {
        return Err(format!("{mn}: needs a memory operand"));
    };
    if matches!(mem.off, Some(MemOff::RegIndex(..))) {
        return Err(format!("{mn}: register-index addressing is not supported"));
    }
    let is_load = mn == "ldp";
    let _size_log2: u32 = if rt1.wide { 3 } else { 2 };
    let step = if rt1.wide { 8 } else { 4 };
    let base = match (is_load, mem.mode) {
        (true, MemMode::Offset) => 0xa9400000,
        (false, MemMode::Offset) => 0xa9000000,
        (true, MemMode::PostIndex) => 0xa8c00000,
        (false, MemMode::PostIndex) => 0xa8800000,
        (true, MemMode::PreIndex) => 0xa9c00000,
        (false, MemMode::PreIndex) => 0xa9800000,
    };
    let mut word = base | (rt2.field() as u32) << 10 | (mem.base.field() as u32) << 5 | rt1.field() as u32;
    if !rt1.wide {
        word &= !0x80000000;
    }
    match &mem.off {
        None => enc.push(word),
        Some(MemOff::Expr(e)) => {
            let c = imm_const(e, ctx, mn)?;
            let imm7 = c / step;
            if c % step != 0 || !(-64..=63).contains(&imm7) {
                return Err(format!("{mn}: offset {c} out of range"));
            }
            enc.push(word | ((imm7 as u32 & 0x7f) << 15));
        }
        _ => return Err(err_op(mn)),
    }
    Ok(())
}

fn adr(enc: &mut Enc, mn: &str, ops: &[Operand], ctx: &EvalCtx) -> Result<(), String> {
    expect_ops(ops, 2, mn)?;
    let rd = reg_of(&ops[0], mn)?;
    let (kind, val) = match &ops[1] {
        Operand::Imm { expr, lsl: None } => expr.eval_mod(ctx)?,
        _ => return Err(err_op(mn)),
    };
    if mn == "adr" && kind.is_some() {
        return Err("adr does not take relocation modifiers".into());
    }
    if mn == "adrp" && matches!(kind, Some(ModKind::Lo12) | Some(ModKind::GotLo12)) {
        return Err("adrp does not take :lo12:/got_lo12:".into());
    }
    let base = if mn == "adrp" { 0x90000000 } else { 0x10000000 };
    match val {
        RValue::Const(c) => {
            let pc = ctx.cur_off;
            let d = if mn == "adrp" {
                ((c & !0xfff) >> 12).wrapping_sub((pc & !0xfff) >> 12)
            } else {
                c - pc
            };
            if !fits_signed(d, 21) {
                return Err(format!("{mn}: target {c:#x} out of range"));
            }
            enc.push(base | (rd.field() as u32) | ((d as u32 & 3) << 29) | (((d >> 2) as u32 & 0x7ffff) << 5));
        }
        v => {
            enc.push(base | rd.field() as u32);
            let layout = match kind {
                Some(ModKind::Got) => FieldLayout::A64GotPage,
                _ if mn == "adrp" => FieldLayout::A64AdrpImm21,
                _ => FieldLayout::A64Imm21,
            };
            enc.fixups.push(FixupReq { at: 0, size: 4, val: v, pcrel: true, rel_base: 0, layout });
        }
    }
    Ok(())
}

/// b / bl / b.cond. External and other-section targets always get a
/// relocation; same-section targets are patched in place with the
/// estimate and keep their relocation (GNU as behavior).
fn emit_branch(enc: &mut Enc, ops: &[Operand], cond: Option<u32>, link: bool, ctx: &EvalCtx) -> Result<(), String> {
    expect_ops(ops, 1, "b")?;
    let target = match &ops[0] {
        Operand::Imm { expr, lsl: None } => expr,
        _ => return Err("branch target must be an address".into()),
    };
    let base = match cond {
        Some(c) => 0x54000000 | c,
        None => if link { 0x94000000 } else { 0x14000000 },
    };
    let layout = if cond.is_some() { FieldLayout::A64Imm19 } else { FieldLayout::A64Imm26 };
    let field_bits = if cond.is_some() { 19 } else { 26 };
    match target.eval(ctx)? {
        RValue::Const(c) => {
            let d = c - ctx.cur_off;
            if d & 3 != 0 || !fits_signed(d >> 2, field_bits) {
                return Err(format!("branch target {c:#x} out of range or misaligned"));
            }
            let field = ((d >> 2) as u32) & if cond.is_some() { 0x7ffff } else { 0x3ffffff };
            enc.push(base | field << if cond.is_some() { 5 } else { 0 });
        }
        v => {
            enc.push(base);
            enc.fixups.push(FixupReq { at: 0, size: 4, val: v, pcrel: true, rel_base: 0, layout });
        }
    }
    Ok(())
}

fn emit_imm19(enc: &mut Enc, word: u32, rt: u8, target_op: &Operand, ctx: &EvalCtx) -> Result<(), String> {
    let target = match target_op {
        Operand::Imm { expr, lsl: None } => expr,
        _ => return Err("branch target must be an address".into()),
    };
    let word = word | rt as u32;
    match target.eval(ctx)? {
        RValue::Const(c) => {
            let d = c - ctx.cur_off;
            if d & 3 != 0 || !fits_signed(d >> 2, 19) {
                return Err(format!("branch target {c:#x} out of range or misaligned"));
            }
            enc.push(word | ((d as u32 >> 2) & 0x7ffff) << 5);
        }
        v => {
            enc.push(word);
            enc.fixups.push(FixupReq { at: 0, size: 4, val: v, pcrel: true, rel_base: 0, layout: FieldLayout::A64Imm19 });
        }
    }
    Ok(())
}

fn emit_imm14(enc: &mut Enc, word: u32, rt: u8, target_op: &Operand, ctx: &EvalCtx) -> Result<(), String> {
    let target = match target_op {
        Operand::Imm { expr, lsl: None } => expr,
        _ => return Err("branch target must be an address".into()),
    };
    let word = word | rt as u32;
    match target.eval(ctx)? {
        RValue::Const(c) => {
            let d = c - ctx.cur_off;
            if d & 3 != 0 || !fits_signed(d >> 2, 14) {
                return Err(format!("branch target {c:#x} out of range or misaligned"));
            }
            enc.push(word | ((d as u32 >> 2) & 0x3fff) << 5);
        }
        v => {
            enc.push(word);
            enc.fixups.push(FixupReq { at: 0, size: 4, val: v, pcrel: true, rel_base: 0, layout: FieldLayout::A64Imm14 });
        }
    }
    Ok(())
}

fn barrier(enc: &mut Enc, mn: &str, ops: &[Operand]) -> Result<(), String> {
    let opt = match ops.first() {
        None => "sy".to_string(),
        Some(Operand::Imm { expr: Expr::Sym(name), lsl: None }) => name.clone(),
        _ => return Err(err_op(mn)),
    };
    let code: u32 = match opt.as_str() {
        "sy" => 0xf,
        "st" => 0xe,
        "ld" => 0xd,
        "ish" => 0xb,
        "ishst" => 0xa,
        "ishld" => 0x9,
        "nsh" => 0x7,
        "nshst" => 0x6,
        "nshld" => 0x5,
        "osh" => 0x3,
        "oshst" => 0x2,
        "oshld" => 0x1,
        _ => return Err(format!("unknown barrier option `{opt}'")),
    };
    let base = match mn {
        "dmb" => 0xd50330bf,
        "dsb" => 0xd503309f,
        _ => 0xd50330df,
    };
    enc.push(base | code << 8);
    if mn == "isb" && opt != "sy" {
        return Err("isb takes only `sy'".into());
    }
    Ok(())
}

fn ldstx(enc: &mut Enc, mn: &str, ops: &[Operand]) -> Result<(), String> {
    match (mn, ops) {
        ("ldxr" | "ldar", [Operand::Reg(rt), Operand::Mem(MemOp { base, off: None, mode: MemMode::Offset })]) => {
            // size bits [31:30] follow the data register width.
            let base_word = match (mn, rt.wide) {
                ("ldxr", true) => 0xc85f7c00,
                ("ldxr", false) => 0x885f7c00,
                (_, true) => 0xc8dffc00,
                (_, false) => 0x88dffc00,
            };
            enc.push(base_word | (base.field() as u32) << 5 | rt.field() as u32);
        }
        ("stxr", [Operand::Reg(rs), Operand::Reg(rt), Operand::Mem(MemOp { base, off: None, mode: MemMode::Offset })]) => {
            if rs.wide {
                return Err("stxr status register must be a w register".into());
            }
            let base_word = if rt.wide { 0xc8007c00 } else { 0x88007c00 };
            enc.push(base_word | (rs.field() as u32) << 16 | (base.field() as u32) << 5 | rt.field() as u32);
        }
        ("stlr", [Operand::Reg(rt), Operand::Mem(MemOp { base, off: None, mode: MemMode::Offset })]) => {
            let base_word = if rt.wide { 0xc89ffc00 } else { 0x889ffc00 };
            enc.push(base_word | (base.field() as u32) << 5 | rt.field() as u32);
        }
        _ => return Err(err_op(mn)),
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Finalization: fixup patching and relocation emission
// ---------------------------------------------------------------------

fn finalize_fixup(data: &mut [u8], f: &Fixup, relocs: &mut Vec<Reloc>) -> Result<(), Vec<String>> {
    match f.layout {
        FieldLayout::Flat => {
            match &f.val {
                RValue::Const(_) => Ok(()),
                v => {
                    // Same-section targets resolve in place and keep
                    // their relocation; others stay zero.
                    if let RValue::SectVal { sect, .. } = v {
                        if *sect == f.section {
                            let val = sym_off(v);
                            let pos = f.offset as usize;
                            for i in 0..f.size as usize {
                                data[pos + i] = (val >> (i * 8)) as u8;
                            }
                        }
                    }
                    let r_type = match f.size {
                        4 => R_AARCH64_ABS32.0,
                        8 => R_AARCH64_ABS64.0,
                        _ => return Err(vec!["unsupported data relocation size".into()]),
                    };
                    push_reloc(relocs, f, v, r_type);
                    Ok(())
                }
            }
        }
        FieldLayout::A64Imm26 => {
            let r_type = if word_bit(data, f.offset, 31) { R_AARCH64_CALL26.0 } else { R_AARCH64_JUMP26.0 };
            pcrel_finalize(data, f, relocs, r_type, |d| (d >> 2) as u32 & 0x3ffffff, 0x3ffffff, 26)
        }
        FieldLayout::A64Imm19 => {
            let r_type = R_AARCH64_CONDBR19.0;
            pcrel_finalize(data, f, relocs, r_type, |d| ((d >> 2) as u32 & 0x7ffff) << 5, 0x00ffffe0, 19)
        }
        FieldLayout::A64Imm14 => {
            pcrel_finalize(data, f, relocs, R_AARCH64_TSTBR14.0, |d| ((d >> 2) as u32 & 0x3fff) << 5, 0x0007ffe0, 14)
        }
        FieldLayout::A64Imm21 => {
            // adr resolves same-section targets in place (no reloc).
            pcrel_finalize(data, f, relocs, R_AARCH64_ADR_PREL_LO21.0, |d| ((d & 3) as u32) << 29 | (((d >> 2) as u32) & 0x7ffff) << 5, 0x60ffffe0, 21)
        }
        FieldLayout::A64AdrpImm21 => {
            // adrp's page result depends on the section's final
            // address, so even same-section targets keep a reloc and
            // an unpatched field.
            match &f.val {
                RValue::Const(_) => Ok(()),
                v => {
                    push_reloc(relocs, f, v, R_AARCH64_ADR_PREL_PG_HI21.0);
                    Ok(())
                }
            }
        }
        FieldLayout::A64GotPage => {
            match &f.val {
                RValue::Const(_) => Ok(()),
                v => {
                    push_reloc(relocs, f, v, R_AARCH64_ADR_GOT_PAGE.0);
                    Ok(())
                }
            }
        }
        FieldLayout::A64GotLo12 => {
            match &f.val {
                RValue::Const(_) => Ok(()),
                v => {
                    push_reloc(relocs, f, v, R_AARCH64_LD64_GOT_LO12_NC.0);
                    Ok(())
                }
            }
        }
        FieldLayout::A64Imm12 => {
            match &f.val {
                RValue::Const(_) => Ok(()),
                v => {
                    if let RValue::SectVal { sect, .. } = v {
                        if *sect == f.section {
                            patch_bits(data, f.offset, 0x003ffc00, 10, (sym_off(v) & 0xfff) as u32);
                        }
                    }
                    push_reloc(relocs, f, v, R_AARCH64_ADD_ABS_LO12_NC.0);
                    Ok(())
                }
            }
        }
        FieldLayout::A64LdstImm12 => {
            match &f.val {
                RValue::Const(_) => Ok(()),
                v => {
                    let scale = (word_field(data, f.offset, 31, 30) as u64).min(3);
                    let r_type = match scale {
                        0 => R_AARCH64_LDST8_ABS_LO12_NC.0,
                        1 => R_AARCH64_LDST16_ABS_LO12_NC.0,
                        2 => R_AARCH64_LDST32_ABS_LO12_NC.0,
                        _ => R_AARCH64_LDST64_ABS_LO12_NC.0,
                    };
                    if let RValue::SectVal { sect, .. } = v {
                        if *sect == f.section {
                            patch_bits(data, f.offset, 0x003ffc00, 10, ((sym_off(v) & 0xfff) >> scale) as u32);
                        }
                    }
                    push_reloc(relocs, f, v, r_type);
                    Ok(())
                }
            }
        }
        FieldLayout::A64Imm16 => Ok(()),
        _ => Err(vec!["internal error: unexpected fixup layout".into()]),
    }
}

/// Resolves a PC-relative field: same-section targets are patched with
/// the real displacement (and keep their relocation), others keep GNU
/// as's estimate. `delta` is the branch distance.
fn pcrel_finalize(data: &mut [u8], f: &Fixup, relocs: &mut Vec<Reloc>, r_type: u32, spread: impl Fn(i64) -> u32, mask: u32, bits: u32) -> Result<(), Vec<String>> {
    match &f.val {
        RValue::Const(_) => Ok(()),
        v => {
            // Same-section branches resolve statically — GNU as patches
            // the field and emits no relocation. Anything else stays at
            // zero with a relocation; the linker computes the value.
            if let RValue::SectVal { sect, off, .. } = v {
                if *sect == f.section {
                    let d = *off - f.pcrel_base();
                    if !fits_signed(d >> 2, bits) {
                        return Err(vec![format!("branch displacement {d} out of range")]);
                    }
                    patch_bits(data, f.offset, mask, 0, spread(d));
                    return Ok(());
                }
            }
            push_reloc(relocs, f, v, r_type);
            Ok(())
        }
    }
}

fn word_bit(data: &[u8], off: i64, bit: u32) -> bool {
    (word_at(data, off) >> bit) & 1 == 1
}

fn word_at(data: &[u8], off: i64) -> u32 {
    u32::from_le_bytes(data[off as usize..off as usize + 4].try_into().unwrap())
}

fn word_field(data: &[u8], off: i64, hi: u32, lo: u32) -> u32 {
    (word_at(data, off) >> lo) & ((1u32 << (hi - lo + 1)) - 1)
}

fn patch_bits(data: &mut [u8], off: i64, mask: u32, shift: u32, val: u32) {
    let pos = off as usize;
    let w = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    let w = (w & !mask) | (val << shift & mask);
    data[pos..pos + 4].copy_from_slice(&w.to_le_bytes());
}

/// GNU as's in-place estimate for an unresolved pc-relative target.
fn sym_off(v: &RValue) -> i64 {
    match v {
        RValue::Const(c) => *c,
        RValue::SectVal { off, .. } | RValue::RelDiff { off, .. } | RValue::RelDiffExtern { off, .. } => *off,
        RValue::ExternRef { off, .. } => *off,
    }
}

/// Emits a relocation for `val` the way GNU as does: against the
/// referenced symbol (addend = value offset past it), falling back to
/// the section for unnamed section offsets.
fn push_reloc(relocs: &mut Vec<Reloc>, f: &Fixup, val: &RValue, r_type: u32) {
    let (target, addend) = match val {
        RValue::ExternRef { sym, off } => (Target::Symbol(*sym), *off),
        // Local symbols relocate against the section, like GNU as.
        RValue::SectVal { sect, off, .. } => (Target::Section(*sect), *off),
        RValue::RelDiff { sect, off, sym } => (match sym {
            Some(i) => Target::Symbol(*i),
            None => Target::Section(*sect),
        }, *off),
        RValue::RelDiffExtern { sym, off } => (Target::Symbol(*sym), *off),
        RValue::Const(_) => return,
    };
    relocs.push(Reloc {
        section: f.section,
        offset: f.offset,
        size: f.size,
        pcrel: f.pcrel,
        addend,
        target,
        r_type,
    });
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

fn reg_of(op: &Operand, mn: &str) -> Result<Reg, String> {
    match op {
        Operand::Reg(r) => Ok(*r),
        _ => Err(format!("`{mn}': operand must be a register")),
    }
}

fn expect_ops(ops: &[Operand], n: usize, mn: &str) -> Result<(), String> {
    if ops.len() != n {
        return Err(format!("`{mn}': expected {n} operands, got {}", ops.len()));
    }
    Ok(())
}

fn err_op(mn: &str) -> String {
    format!("`{mn}': bad operand forms")
}

fn imm_const(e: &Expr, ctx: &EvalCtx, mn: &str) -> Result<i64, String> {
    match e.eval(ctx)? {
        RValue::Const(c) => Ok(c),
        _ => Err(format!("`{mn}': immediate must be constant")),
    }
}

fn imm_const_val(e: &Expr, lsl: Option<&Expr>, ctx: &EvalCtx, mn: &str) -> Result<i64, String> {
    if lsl.is_some() {
        return Err(format!("`{mn}': this immediate does not take an `lsl' suffix"));
    }
    match e.eval(ctx)? {
        RValue::Const(c) => Ok(c),
        _ => Err(format!("`{mn}': immediate must be constant")),
    }
}

fn amount_const(e: &Expr, mn: &str) -> Result<i64, String> {
    match e {
        Expr::Int(v) => Ok(*v),
        _ => Err(format!("`{mn}': shift amount must be constant")),
    }
}

fn shift_field(kind: ShiftKind) -> Result<u32, String> {
    Ok(kind.field())
}

fn fits_signed(v: i64, bits: u32) -> bool {
    let shift = 64 - bits;
    (v << shift) >> shift == v
}

/// GNU as's logical-immediate encoding: the value must repeat with
/// period esize, and the esize-bit pattern must be a rotation of a run
/// of ones. The largest esize wins (gas tries register width down).
pub fn encode_logical_imm(v: u64, is64: bool) -> Option<(u32, u32, u32)> {
    let (width, mut esize) = if is64 { (64u32, 64u64) } else { (32, 32) };
    let mask = if is64 { u64::MAX } else { 0xffff_ffff };
    let v = v & mask;
    if v == 0 || v == mask {
        return None;
    }
    while esize >= 2 {
        let emask = if esize >= 64 { u64::MAX } else { (1u64 << esize) - 1 };
        let p = v & emask;
        let mut rep = true;
        let mut shift = esize;
        while shift < width as u64 {
            if (v >> shift) & emask != p {
                rep = false;
                break;
            }
            shift += esize;
        }
        if rep {
            for r in 0..esize {
                let rot = if esize >= 64 { p.rotate_left(r as u32) } else { (p << r | p >> (esize - r)) & emask };
                if rot != 0 && rot != emask && (rot + 1).is_power_of_two() {
                    let s = (63 - rot.leading_zeros()) as u32;
                    let n = (is64 && esize == 64) as u32;
                    let len = esize.trailing_zeros();
                    let imms = s | ((0x3fu32 << (len + 1)) & 0x3f);
                    return Some((n, r as u32, imms));
                }
            }
        }
        esize /= 2;
    }
    None
}
