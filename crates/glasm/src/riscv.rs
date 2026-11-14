//! RISC-V backend (RV32/RV64, IMA + C + Zicsr + Zifencei): fixed 32-bit
//! instructions in R/I/S/B/U/J formats plus the compressed C extension,
//! with %hi/%lo/%pcrel_hi/%pcrel_lo relocation modifiers.
//!
//! Behaviors matched against GNU as (binutils 2.45):
//! - automatic compression of user-written instructions (addi -> c.addi),
//! - the `li` load_const recursion (gas/config/tc-riscv.c) with its
//!   compressed forms (c.li/c.lui/c.addiw/c.slli),
//! - `la`/load-address expansion (auipc + %pcrel_lo pair),
//! - branch relaxation (compressed -> 4-byte -> inverted + `jal zero`),
//! - relocation tables (HI20/PCREL_HI20/LO12_I/PCREL_LO12_I/CALL_PLT
//!   carry a trailing R_RISCV_RELAX; the lo fixup references an internal
//!   symbol placed at the auipc).

use crate::arch::Backend;
use crate::assemble::{Fixup, Reloc, Section, Symbol, Target};
use crate::encoding::{Encoded, FieldLayout, FixupReq};
use crate::expr::{EvalCtx, Expr, ModKind, RValue};
use crate::gas::GTok;
use object::{Architecture, Endianness};
use std::collections::HashMap;

// ---------------------------------------------------------------------
// Registers
// ---------------------------------------------------------------------

/// Integer register names (numeric + ABI). FP registers are not
/// supported yet.
const GPR_NAMES: &[&str] = &[
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
    "t3", "t4", "t5", "t6",
];

pub fn gpr(name: &str) -> Option<u8> {
    let lower = name.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix('x') {
        return rest.parse::<u8>().ok().filter(|n| *n < 32);
    }
    if lower == "fp" {
        return Some(8);
    }
    GPR_NAMES.iter().position(|n| *n == lower).map(|i| i as u8)
}

// ---------------------------------------------------------------------
// Statement model
// ---------------------------------------------------------------------

/// One RISC-V instruction: mnemonic plus already-parsed operands.
#[derive(Clone, Debug)]
pub struct Insn {
    pub mnemonic: String,
    pub operands: Vec<Operand>,
    pub line: u32,
}

/// A RISC-V operand: bare register names (`a0`, `t2`, `x31`), immediates
/// (possibly carrying %hi/%lo-style modifiers in the expression), or
/// `disp(reg)` memory forms. CSR names parse as symbols and are
/// interpreted by the encoder by operand position.
#[derive(Clone, Debug)]
pub enum Operand {
    Reg(u8),
    Imm(Expr),
    /// `disp(reg)`; the bare `(reg)` form has `disp: Expr::Int(0)`.
    Mem {
        base: u8,
        disp: Expr,
    },
}

impl Operand {
    fn as_expr(&self) -> Option<&Expr> {
        match self {
            Operand::Imm(e) => Some(e),
            Operand::Mem { disp, .. } => Some(disp),
            Operand::Reg(_) => None,
        }
    }
}

pub fn parse_insn(g: &[crate::gas::GTok], line_no: u32, lineno: usize, numlabels: &crate::gas::NumLabels) -> Result<crate::insn::Stmt, String> {
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
        }
    }
    if t.pos != t.g.len() {
        return Err("junk after instruction operands".into());
    }
    Ok(crate::insn::Stmt::Insn(crate::arch::Insn::Riscv(Insn {
        mnemonic,
        operands,
        line: line_no,
    })))
}

fn parse_operand(t: &mut GToks, lineno: usize, numlabels: &crate::gas::NumLabels) -> Result<Operand, String> {
    match t.peek() {
        Some(GTok::Reg(name)) => {
            // %hi(sym)-style modifiers lex as register tokens.
            if matches!(name.as_str(), "hi" | "lo" | "pcrel_hi" | "pcrel_lo") && t.g.get(t.pos + 1) == Some(&GTok::Punct("(")) {
                let name = name.clone();
                t.bump();
                t.bump(); // consume the `('
                let disp = parse_modifier(&name, t, lineno, numlabels)?;
                if t.eat_punct("(") {
                    let base = parse_reg_operand(t)?;
                    if !t.eat_punct(")") {
                        return Err("expected `)'".into());
                    }
                    return Ok(Operand::Mem { base, disp });
                }
                return Ok(Operand::Imm(disp));
            }
            let name = name.clone();
            t.bump();
            gpr(&name).map(Operand::Reg).ok_or_else(|| format!("unknown register `%{name}'"))
        }
        Some(GTok::Punct("(")) => {
            // Bare (reg) memory operand (atomics).
            t.bump();
            let base = parse_reg_operand(t)?;
            if !t.eat_punct(")") {
                return Err("expected `)'".into());
            }
            Ok(Operand::Mem { base, disp: Expr::Int(0) })
        }
        Some(GTok::Ident(name)) if gpr(name).is_some() => {
            // Bare register name (a0, t2, x31...). Register names shadow
            // symbols in operand position, like GNU as.
            let reg = gpr(name).unwrap();
            t.bump();
            Ok(Operand::Reg(reg))
        }
        _ => {
            let disp = t.expr(lineno, numlabels)?;
            if t.eat_punct("(") {
                let base = parse_reg_operand(t)?;
                if !t.eat_punct(")") {
                    return Err("expected `)'".into());
                }
                Ok(Operand::Mem { base, disp })
            } else {
                Ok(Operand::Imm(disp))
            }
        }
    }
}

fn parse_reg_operand(t: &mut GToks) -> Result<u8, String> {
    match t.bump() {
        Some(GTok::Reg(name)) => gpr(name).ok_or_else(|| format!("unknown register `%{name}'")),
        Some(GTok::Ident(name)) => gpr(name).ok_or_else(|| format!("unknown register `{name}'")),
        _ => Err("expected register".into()),
    }
}

/// `%hi(sym)` — the `%name` was lexed as a Reg token and the `(` already
/// consumed by the caller; the inner expression and `)` follow.
fn parse_modifier(name: &str, t: &mut GToks, lineno: usize, numlabels: &crate::gas::NumLabels) -> Result<Expr, String> {
    let kind = match name {
        "hi" => ModKind::Hi,
        "lo" => ModKind::Lo,
        "pcrel_hi" => ModKind::PcrelHi,
        "pcrel_lo" => ModKind::PcrelLo,
        _ => return Err(format!("unknown relocation modifier `%{name}'")),
    };
    let inner = t.expr(lineno, numlabels)?;
    if !t.eat_punct(")") {
        return Err("expected `)' after relocation modifier".into());
    }
    Ok(Expr::Mod { kind, inner: Box::new(inner) })
}

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

    /// Expression up to the next `(`, `)` or `,` (GAS syntax: parens are
    /// register/memory syntax, not grouping).
    fn expr(&mut self, lineno: usize, numlabels: &crate::gas::NumLabels) -> Result<Expr, String> {
        let mut toks = Vec::new();
        while let Some(tok) = self.peek() {
            match tok {
                GTok::Punct("(") | GTok::Punct(")") | GTok::Punct(",") | GTok::Reg(_) => break,
                GTok::BigInt(_) => break,
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
                GTok::Imm | GTok::Star | GTok::Str(_) => break,
            }
        }
        if toks.is_empty() {
            return Err("expected expression".into());
        }
        let mut t = crate::parse::Toks::new(toks);
        crate::parse::parse_expr(&mut t, 0)
    }
}

// ---------------------------------------------------------------------
// Encoding: format constructors
// ---------------------------------------------------------------------

const OPC_OP_IMM: u32 = 0x13;
const OPC_OP: u32 = 0x33;
const OPC_LOAD: u32 = 0x03;
const OPC_STORE: u32 = 0x23;
const OPC_BRANCH: u32 = 0x63;
const OPC_LUI: u32 = 0x37;
const OPC_AUIPC: u32 = 0x17;
const OPC_JAL: u32 = 0x6f;
const OPC_JALR: u32 = 0x67;
const OPC_SYSTEM: u32 = 0x73;
const OPC_MISC_MEM: u32 = 0x0f;
const OPC_AMO: u32 = 0x2f;
const OPC_C0: u32 = 0;
const OPC_C1: u32 = 1;
const OPC_C2: u32 = 2;

fn enc_r(funct7: u32, rs2: u8, rs1: u8, funct3: u32, rd: u8, opc: u32) -> u32 {
    (funct7 << 25) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | opc
}

fn enc_i(imm12: i64, rs1: u8, funct3: u32, rd: u8, opc: u32) -> u32 {
    (((imm12 as u32) & 0xfff) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | opc
}

fn enc_s(imm12: i64, rs2: u8, rs1: u8, funct3: u32, opc: u32) -> u32 {
    let imm = imm12 as u32;
    (((imm >> 5) & 0x7f) << 25) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((imm & 0x1f) << 7) | opc
}

fn enc_b(imm13: i64, rs2: u8, rs1: u8, funct3: u32, opc: u32) -> u32 {
    let imm = imm13 as u32;
    (((imm >> 12) & 1) << 31)
        | (((imm >> 5) & 0x3f) << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | (((imm >> 1) & 0xf) << 8)
        | (((imm >> 11) & 1) << 7)
        | opc
}

fn enc_u(imm20: i64, rd: u8, opc: u32) -> u32 {
    (((imm20 as u32) & 0xfffff) << 12) | ((rd as u32) << 7) | opc
}

fn enc_j(imm21: i64, rd: u8, opc: u32) -> u32 {
    let imm = imm21 as u32;
    (((imm >> 20) & 1) << 31) | (((imm >> 1) & 0x3ff) << 21) | (((imm >> 11) & 1) << 20) | (((imm >> 12) & 0xff) << 12) | ((rd as u32) << 7) | opc
}

// RVC formats.

/// CI with a plain 6-bit signed immediate.
fn enc_ci(funct3: u32, rd_rs1: u8, imm6: i64) -> u32 {
    let imm = imm6 as u32;
    (funct3 << 13) | (((imm as u32) >> 5 & 1) << 12) | ((rd_rs1 as u32) << 7) | ((imm as u32 & 0x1f) << 2) | OPC_C1
}

/// CI on opcode C2 with a pre-spread field.
fn enc_ci2(funct3: u32, rd_rs1: u8, field: u32) -> u32 {
    (funct3 << 13) | ((rd_rs1 as u32) << 7) | (field & 0x1fff) | OPC_C2
}

fn enc_css(funct3: u32, rs2: u8, field: u32) -> u32 {
    (funct3 << 13) | ((rs2 as u32) << 2) | (field & 0x1fe0) | OPC_C2
}

fn enc_ciw(funct3: u32, rd_p: u8, field: u32) -> u32 {
    (funct3 << 13) | ((rd_p as u32) << 2) | (field & 0x1fe0) | OPC_C0
}

fn enc_cl(funct3: u32, rd_p: u8, field: u32, rs1_p: u8) -> u32 {
    (funct3 << 13) | ((rs1_p as u32) << 7) | (field & 0x1c60) | ((rd_p as u32) << 2) | OPC_C0
}

fn enc_cs(funct3: u32, rs2_p: u8, field: u32, rs1_p: u8) -> u32 {
    (funct3 << 13) | ((rs1_p as u32) << 7) | (field & 0x1c60) | ((rs2_p as u32) << 2) | OPC_C0
}

fn enc_cb(funct3: u32, rs1_p: u8, imm9: i64) -> u32 {
    let imm = imm9 as u32;
    (funct3 << 13)
        | (((imm >> 8) & 1) << 12)
        | (((imm >> 3) & 3) << 10)
        | ((rs1_p as u32) << 7)
        | (((imm >> 6) & 3) << 5)
        | (((imm >> 1) & 3) << 3)
        | (((imm >> 5) & 1) << 2)
        | OPC_C1
}

fn enc_cj(funct3: u32, imm12: i64) -> u32 {
    let imm = imm12 as u32;
    (funct3 << 13)
        | (((imm >> 11) & 1) << 12)
        | (((imm >> 4) & 1) << 11)
        | (((imm >> 8) & 3) << 9)
        | (((imm >> 10) & 1) << 8)
        | (((imm >> 6) & 1) << 7)
        | (((imm >> 7) & 1) << 6)
        | (((imm >> 1) & 7) << 3)
        | (((imm >> 5) & 1) << 2)
        | OPC_C1
}

/// CR: c.jr (funct4 8, bit12=0), c.jalr (9), c.mv (8, rs2), c.add (9).
fn enc_cr(funct4: u32, bit12: bool, rd: u8, rs2: u8) -> u32 {
    (funct4 << 12) | ((bit12 as u32) << 12) | ((rd as u32) << 7) | ((rs2 as u32) << 2) | OPC_C2
}

/// Compressed-register set for the `*p` forms: x8-x15.
fn rvc_reg(n: u8) -> Option<u8> {
    match n {
        8..=15 => Some(n - 8),
        _ => None,
    }
}

/// Compressed-register set for c.mv/c.add/c.jr: x1-x15.
fn is_rvc_wide(n: u8) -> bool {
    (1..=15).contains(&n)
}

// ---------------------------------------------------------------------
// Encoding: driver
// ---------------------------------------------------------------------

/// Relocation type numbers (ELF R_RISCV_*).
const R_RISCV_BRANCH: u32 = 16;
const R_RISCV_JAL: u32 = 17;
const R_RISCV_RVC_BRANCH: u32 = 44;
const R_RISCV_RVC_JAL: u32 = 45;
const R_RISCV_CALL_PLT: u32 = 19;
const R_RISCV_PCREL_HI20: u32 = 23;
const R_RISCV_PCREL_LO12_I: u32 = 24;
const R_RISCV_PCREL_LO12_S: u32 = 25;
const R_RISCV_HI20: u32 = 26;
const R_RISCV_LO12_I: u32 = 27;
const R_RISCV_LO12_S: u32 = 28;
const R_RISCV_RELAX: u32 = 51;

/// Sign-extend the low `bits` of `v`.
fn sext(v: i64, bits: u32) -> i64 {
    let shift = 64 - bits;
    (v << shift) >> shift
}

/// gas RISCV_CONST_HIGH_PART: hi such that hi + sext12(lo) == v.
fn high_part(v: i64) -> i64 {
    (v + 0x800) & !0xfff
}

fn fits_signed(v: i64, bits: u32) -> bool {
    sext(v, bits) == v
}

struct Enc {
    bytes: Vec<u8>,
    fixups: Vec<FixupReq>,
}

impl Enc {
    fn push(&mut self, word: u32) {
        self.bytes.extend_from_slice(&word.to_le_bytes());
    }

    fn push_c(&mut self, word: u32) {
        self.bytes.extend_from_slice(&word.to_le_bytes()[..2]);
    }
}

/// Evaluated immediate: constant, or a fixup to resolve later.
enum Imm {
    Const(i64),
    Fix(RValue),
    /// `%pcrel_lo(label)`: the label's address marks the auipc whose
    /// HI20 fixup the finalize pass pairs with.
    PcrelLo(RValue),
}

pub struct RiscvBackend {
    pub arch: crate::arch::Arch,
}

impl RiscvBackend {
    fn rv64(&self) -> bool {
        self.arch == crate::arch::Arch::Riscv64
    }
}

impl Backend for RiscvBackend {
    fn encode(&self, insn: &crate::arch::Insn, ctx: &EvalCtx, relax_state: u8) -> Result<Encoded, String> {
        let crate::arch::Insn::Riscv(rv) = insn else {
            return Err("not a RISC-V instruction".into());
        };
        let mut enc = Enc {
            bytes: Vec::new(),
            fixups: Vec::new(),
        };
        encode_one(&mut enc, rv, ctx, relax_state, self.rv64())?;
        Ok(Encoded {
            bytes: enc.bytes,
            fixups: enc.fixups,
        })
    }

    fn finalize_all(&self, sections: &mut [Section], symbols: &mut Vec<Symbol>, fixups: Vec<Fixup>, relocs: &mut Vec<Reloc>) -> Result<(), Vec<String>> {
        // Pair %pcrel_lo fixups with the PCREL_HI20 fixup at the auipc.
        let mut hi_targets: HashMap<(usize, i64), RValue> = HashMap::new();
        for f in &fixups {
            if f.layout == FieldLayout::RvPcrelHi20 {
                let start = f.offset - f.at as i64;
                hi_targets.insert((f.section, start), f.val.clone());
            }
        }

        // RELAX markers reference a nameless undefined symbol, like
        // GNU as's null-symbol relocs.
        let relax_sym = symbols.len();
        symbols.push(Symbol {
            name: String::new(),
            defined: false,
            seen: true,
            extern_: false,
            global: false,
            weak: false,
            sect: None,
            value: 0,
        });

        let mut errors = Vec::new();
        for f in &fixups {
            let data = &mut sections[f.section].data;
            if let Err(msgs) = finalize_fixup(data, f, &hi_targets, symbols, relocs, relax_sym) {
                errors.extend(msgs);
            }
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    fn relax_step(&self, fixup: &Fixup, delta: i64, state: u8) -> Option<u8> {
        if !fixup.pcrel {
            return None;
        }
        match (fixup.layout, fixup.size) {
            // c.beqz/c.bnez/c.j: GNU as relaxes when the displacement
            // leaves (val + reach/2) < reach, i.e. the half-window.
            (FieldLayout::RvBranch, 2) => {
                if !(-128..=127).contains(&delta) && state < 1 {
                    Some(1)
                } else {
                    None
                }
            }
            (FieldLayout::RvBranch, _) => {
                if !(-4096..=4094).contains(&delta) && state < 2 {
                    Some(2)
                } else {
                    None
                }
            }
            (FieldLayout::RvJal, 2) => {
                if !(-1024..=1022).contains(&delta) && state < 1 {
                    Some(1)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn object_arch(&self) -> Architecture {
        match self.arch {
            crate::arch::Arch::Riscv32 => Architecture::Riscv32,
            _ => Architecture::Riscv64,
        }
    }

    fn endianness(&self) -> Endianness {
        Endianness::Little
    }

    fn text_fill(&self) -> &'static [u8] {
        &[0]
    }
}

fn encode_one(enc: &mut Enc, insn: &Insn, ctx: &EvalCtx, relax_state: u8, rv64: bool) -> Result<(), String> {
    let mn = insn.mnemonic.as_str();
    let ops = &insn.operands;

    // Atomic ordering suffixes: lr.w.aqrl / amoadd.d.rl.
    let (mn, aq, rl) = split_ordering(mn);

    match mn {
        // -------------------------------------------------------------
        // Pseudo-instructions and multi-instruction expansions.
        // -------------------------------------------------------------
        "nop" => {
            expect_ops(ops, 0, mn)?;
            emit_addi(enc, 0, 0, Imm::Const(0), ctx)?;
        }
        "ret" => {
            expect_ops(ops, 0, mn)?;
            emit_jalr(enc, 0, 1, Imm::Const(0), ctx)?;
        }
        "mv" => {
            let (rd, rs) = two_regs(ops, mn)?;
            emit_addi(enc, rd, rs, Imm::Const(0), ctx)?;
        }
        "not" => {
            let (rd, rs) = two_regs(ops, mn)?;
            enc.push(enc_i(-1, rs, 0x4, rd, OPC_OP_IMM));
        }
        "neg" => {
            let (rd, rs) = two_regs(ops, mn)?;
            enc.push(enc_r(0x20, rs, 0, 0x0, rd, OPC_OP));
        }
        "negw" => {
            let (rd, rs) = two_regs(ops, mn)?;
            enc.push(enc_r(0x20, rs, 0, 0x0, rd, OPC_OP + 8));
        }
        "seqz" => {
            let (rd, rs) = two_regs(ops, mn)?;
            enc.push(enc_r(0, 1, rs, 0x3, rd, OPC_OP_IMM));
        }
        "snez" => {
            let (rd, rs) = two_regs(ops, mn)?;
            enc.push(enc_r(0, rs, 0, 0x3, rd, OPC_OP));
        }
        "sltz" => {
            let (rd, rs) = two_regs(ops, mn)?;
            enc.push(enc_r(0, 0, rs, 0x2, rd, OPC_OP));
        }
        "sgtz" => {
            let (rd, rs) = two_regs(ops, mn)?;
            enc.push(enc_r(0, rs, 0, 0x2, rd, OPC_OP));
        }
        "li" => {
            let rd = one_reg(ops, mn)?;
            let imm = match &ops[1] {
                Operand::Imm(e) => imm_val(e, ctx)?,
                _ => return Err(err_op(mn)),
            };
            let c = imm_const(&imm, mn)?;
            // Values in addi's range are plain `addi rd, zero, imm`
            // (compressing to c.li); larger ones go through load_const.
            if fits_signed(c, 12) {
                emit_addi(enc, rd, 0, Imm::Const(c), ctx)?;
            } else {
                load_const(enc, rd, c, ctx, rv64)?;
            }
        }
        "la" | "lla" => {
            let rd = one_reg(ops, mn)?;
            let val = match &ops[1] {
                Operand::Imm(e) => e.eval(ctx)?,
                _ => return Err(err_op(mn)),
            };
            let here = ctx.cur_off;
            emit_auipc(enc, rd, val, true, ctx)?;
            emit_pcrel_lo(enc, rd, rd, here, ctx.cur_sect, false)?;
        }
        "j" => {
            expect_ops(ops, 1, mn)?;
            let target = branch_target(&ops[0].as_expr().unwrap(), ctx)?;
            emit_jal(enc, 0, target, ctx, relax_state)?;
        }
        "jr" => {
            expect_ops(ops, 1, mn)?;
            let Operand::Reg(rs) = &ops[0] else { return Err(err_op(mn)) };
            emit_jalr(enc, 0, *rs, Imm::Const(0), ctx)?;
        }
        "jalr" => {
            // jalr rs | jalr rd, rs | jalr rd, imm(rs) | jalr rd, rs, imm
            match ops.as_slice() {
                [Operand::Reg(rs)] => emit_jalr(enc, 1, *rs, Imm::Const(0), ctx)?,
                [Operand::Reg(rd), Operand::Reg(rs)] => emit_jalr(enc, *rd, *rs, Imm::Const(0), ctx)?,
                [Operand::Reg(rd), Operand::Mem { base, disp }] => {
                    let imm = imm_val(&disp.clone(), ctx)?;
                    emit_jalr(enc, *rd, *base, imm, ctx)?
                }
                [Operand::Reg(rd), Operand::Reg(rs), Operand::Imm(e)] => {
                    let imm = imm_val(e, ctx)?;
                    emit_jalr(enc, *rd, *rs, imm, ctx)?
                }
                _ => return Err(err_op(mn)),
            }
        }
        "call" => {
            expect_ops(ops, 1, mn)?;
            let val = match &ops[0] {
                Operand::Imm(e) => e.eval(ctx)?,
                _ => return Err(err_op(mn)),
            };
            // auipc+jalr covered by one R_RISCV_CALL_PLT; no separate
            // PCREL_HI20 fixup (GNU as emits none here).
            enc.push(enc_u(0, 1, OPC_AUIPC));
            emit_call_jalr(enc, 1, 1, val)?;
        }
        "tail" => {
            expect_ops(ops, 1, mn)?;
            let val = match &ops[0] {
                Operand::Imm(e) => e.eval(ctx)?,
                _ => return Err(err_op(mn)),
            };
            enc.push(enc_u(0, 6, OPC_AUIPC));
            emit_call_jalr(enc, 0, 6, val)?;
        }
        "beqz" | "bnez" | "blez" | "bgez" | "bltz" | "bgtz" => {
            let rs = one_reg(ops, mn)?;
            let target = branch_target(&ops[1].as_expr().unwrap(), ctx)?;
            // beqz/bnez/bltz/bgez compare rs against x0 (`bge rs, x0`
            // and friends); blez/bgtz flip the operands.
            let funct3 = match mn {
                "beqz" => 0,
                "bnez" => 1,
                "blez" | "bgez" => 5,
                _ => 4,
            };
            let (rs1, rs2) = if mn == "blez" || mn == "bgtz" { (0, rs) } else { (rs, 0) };
            emit_branch(enc, funct3, rs1, rs2, target, ctx, relax_state)?;
        }
        "bgt" | "ble" | "bgtu" | "bleu" => {
            expect_ops(ops, 3, mn)?;
            let (Operand::Reg(rs2), Operand::Reg(rs1)) = (&ops[0], &ops[1]) else {
                return Err(err_op(mn));
            };
            let (rs2, rs1) = (*rs2, *rs1);
            let target = branch_target(&ops[2].as_expr().unwrap(), ctx)?;
            let funct3 = match mn {
                "bgt" => 4,
                "ble" => 5,
                "bgtu" => 6,
                _ => 7,
            };
            emit_branch(enc, funct3, rs1, rs2, target, ctx, relax_state)?;
        }

        // -------------------------------------------------------------
        // Loads and stores.
        // -------------------------------------------------------------
        "lw" | "lh" | "lb" | "lwu" | "lhu" | "lbu" | "ld" => {
            let funct3 = load_funct3(mn, rv64)?;
            let rd = rd_of(ops)?;
            emit_load(enc, rd, &ops[1], funct3, ctx)?;
        }
        "sw" | "sh" | "sb" | "sd" => {
            let funct3 = store_funct3(mn)?;
            emit_store(enc, ops, funct3, ctx)?;
        }

        // -------------------------------------------------------------
        // Real instructions.
        // -------------------------------------------------------------
        "add" | "sub" | "sll" | "slt" | "sltu" | "xor" | "srl" | "sra" | "or" | "and" | "addw" | "subw" | "sllw" | "srlw" | "sraw" => {
            let (rd, rs1, rs2) = three_regs(ops, mn)?;
            let (f7, f3, opc) = op_fields(mn)?;
            enc.push(enc_r(f7, rs2, rs1, f3, rd, opc));
            compress_rtype(enc, mn, rd, rs1, rs2);
        }
        "addi" | "slti" | "sltiu" | "ori" | "andi" => {
            let (rd, rs1) = two_regs(ops, mn)?;
            let imm = match &ops[2] {
                Operand::Imm(e) => imm_val(e, ctx)?,
                _ => return Err(err_op(mn)),
            };
            match mn {
                "addi" => emit_addi(enc, rd, rs1, imm, ctx)?,
                "slti" => enc.push(enc_i(imm_const(&imm, mn)?, rs1, 0x2, rd, OPC_OP_IMM)),
                "sltiu" => enc.push(enc_i(imm_const(&imm, mn)?, rs1, 0x3, rd, OPC_OP_IMM)),
                "ori" => enc.push(enc_i(imm_const(&imm, mn)?, rs1, 0x6, rd, OPC_OP_IMM)),
                "andi" => emit_andi(enc, rd, rs1, imm, ctx)?,
                _ => unreachable!(),
            }
        }
        "addiw" => {
            let (rd, rs1) = two_regs(ops, mn)?;
            let imm = match &ops[2] {
                Operand::Imm(e) => imm_val(e, ctx)?,
                _ => return Err(err_op(mn)),
            };
            match &imm {
                Imm::Const(c) if rs1 == 0 && rd != 0 && fits_signed(*c, 6) => {
                    enc.push_c(enc_ci(1, rd, *c)); // c.addiw == c.li slot? no
                }
                Imm::PcrelLo(_) => return Err("%pcrel_lo is not an addiw immediate".into()),
                _ => emit_i(enc, rd, rs1, imm, 0x0, OPC_OP_IMM + 8)?,
            }
        }
        "xori" => {
            let (rd, rs1) = two_regs(ops, mn)?;
            let imm = match &ops[2] {
                Operand::Imm(e) => imm_val(e, ctx)?,
                _ => return Err(err_op(mn)),
            };
            emit_i(enc, rd, rs1, imm, 0x4, OPC_OP_IMM)?;
        }
        "slli" | "srli" | "srai" | "slliw" | "srliw" | "sraiw" => {
            let (rd, rs1) = two_regs(ops, mn)?;
            let shamt = match &ops[2] {
                Operand::Imm(Expr::Int(v)) => *v,
                _ => return Err(format!("{mn}: shift amount must be constant")),
            };
            let (f7, f3, opc) = shift_fields(mn, rv64)?;
            if shamt < 0 || shamt >= 64 || (mn.ends_with('w') && shamt >= 32) {
                return Err(format!("{mn}: shift amount {shamt} out of range"));
            }
            enc.push(enc_r(f7, shamt as u8, rs1, f3, rd, opc));
            // c.slli / c.srli / c.srai compression (rd/rs1 == same reg).
            if rd == rs1 && rd != 0 && shamt > 0 && shamt < 32 {
                match mn {
                    "slli" => {
                        let c = ((shamt as u32) << 2) | ((rd as u32) << 7) | OPC_C2;
                        replace_with_compressed(enc, c);
                    }
                    "srli" | "srai" => {
                        if let (Some(_), Some(rp)) = (rvc_reg(rd), rvc_reg(rs1)) {
                            let funct3 = 4u32;
                            let bit12 = (mn == "srai") as u32;
                            let c = (funct3 << 13) | (bit12 << 12) | ((rp as u32) << 7) | (((shamt as u32) & 0x1f) << 2) | OPC_C1;
                            replace_with_compressed(enc, c);
                        }
                    }
                    _ => {}
                }
            }
        }
        "lui" => {
            expect_ops(ops, 2, mn)?;
            let rd = rd_of(ops)?;
            let Operand::Imm(e) = &ops[1] else { return Err(err_op(mn)) };
            let (kind, val) = e.eval_mod(ctx)?;
            emit_lui(enc, rd, val, matches!(kind, Some(ModKind::PcrelHi)))?;
        }
        "auipc" => {
            expect_ops(ops, 2, mn)?;
            let rd = rd_of(ops)?;
            let Operand::Imm(e) = &ops[1] else { return Err(err_op(mn)) };
            let (kind, val) = e.eval_mod(ctx)?;
            emit_auipc(enc, rd, val, kind == Some(ModKind::PcrelHi) || kind.is_none(), ctx)?;
        }
        "jal" => match ops.as_slice() {
            [Operand::Imm(e)] => {
                let target = branch_target(e, ctx)?;
                emit_jal(enc, 1, target, ctx, relax_state)?;
            }
            [Operand::Reg(rd), Operand::Imm(e)] => {
                let target = branch_target(e, ctx)?;
                emit_jal(enc, *rd, target, ctx, relax_state)?;
            }
            _ => return Err(err_op(mn)),
        },
        "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu" => {
            let (rs1, rs2) = two_regs(ops, mn)?;
            let target = branch_target(&ops[2].as_expr().unwrap(), ctx)?;
            let funct3 = match mn {
                "beq" => 0,
                "bne" => 1,
                "blt" => 4,
                "bge" => 5,
                "bltu" => 6,
                _ => 7,
            };
            emit_branch(enc, funct3, rs1, rs2, target, ctx, relax_state)?;
        }
        "fence" => {
            let (pred, succ) = match ops.as_slice() {
                [] => (0xfu32, 0xfu32),
                [Operand::Imm(p), Operand::Imm(s)] => (fence_mask(p)?, fence_mask(s)?),
                _ => return Err("fence operands must be constant masks".into()),
            };
            enc.push((pred << 24) | (succ << 20) | OPC_MISC_MEM);
        }
        "fence.i" => {
            enc.push(0x0000_100f);
        }
        "ecall" => {
            expect_ops(ops, 0, mn)?;
            enc.push(enc_r(0, 0, 0, 0, 0, OPC_SYSTEM));
        }
        "ebreak" => {
            expect_ops(ops, 0, mn)?;
            enc.push_c(0x9002); // c.ebreak
        }
        "csrrw" | "csrrs" | "csrrc" | "csrrwi" | "csrrsi" | "csrrci" | "csrr" | "csrw" | "csrs" | "csrc" | "csrwi" | "csrsi" | "csrcsi" => {
            encode_csr(enc, mn, ops)?;
        }
        _ => {
            let amo = amo_fields(mn);
            match amo {
                Some((funct5, width)) => encode_amo(enc, funct5, width, aq, rl, ops.as_slice(), mn)?,
                None => return Err(format!("unknown or unsupported instruction `{mn}'")),
            }
        }
    }
    Ok(())
}

/// c.mv/c.add/c.sub/c.xor/c.or/c.and/c.subw/c.addw compression for the
/// 3-register forms.
fn compress_rtype(enc: &mut Enc, mn: &str, rd: u8, rs1: u8, rs2: u8) {
    if !is_rvc_wide(rd) || rd == 0 {
        return;
    }
    let fields = ((rd as u32) << 7) | ((rs2 as u32) << 2);
    let rdp = ((rd as u32) & 7) << 7;
    let rs2p = ((rs2 as u32) & 7) << 2;
    let word = match mn {
        "add" if rs1 == 0 && rs2 != 0 => Some(0x8002 | fields), // c.mv
        "add" if rs1 == rd => Some(0x9002 | fields),            // c.add
        "sub" if rs1 == rd && is_rvc_wide(rs2) && rs2 != 0 => Some(0x8c01 | rdp | rs2p),
        "xor" if rs1 == rd && is_rvc_wide(rs2) && rs2 != 0 => Some(0x8c21 | rdp | rs2p),
        "or" if rs1 == rd && is_rvc_wide(rs2) && rs2 != 0 => Some(0x8c41 | rdp | rs2p),
        "and" if rs1 == rd && is_rvc_wide(rs2) && rs2 != 0 => Some(0x8c61 | rdp | rs2p),
        "subw" if rs1 == rd && is_rvc_wide(rs2) && rs2 != 0 => Some(0x9c01 | rdp | rs2p),
        "addw" if rs1 == rd && is_rvc_wide(rs2) && rs2 != 0 => Some(0x9c21 | rdp | rs2p),
        _ => None,
    };
    if let Some(w) = word {
        replace_with_compressed(enc, w);
    }
}

/// Replaces the trailing 4-byte word with a 2-byte compressed form.
fn replace_with_compressed(enc: &mut Enc, word: u32) {
    let n = enc.bytes.len();
    if n < 4 {
        return;
    }
    enc.bytes.truncate(n - 4);
    enc.push_c(word);
}

fn split_ordering(mn: &str) -> (&str, bool, bool) {
    let mut aq = false;
    let mut rl = false;
    let mut rest = mn;
    for suffix in [".aqrl", ".aq", ".rl"] {
        if let Some(r) = rest.strip_suffix(suffix) {
            rest = r;
            match suffix {
                ".aqrl" => {
                    aq = true;
                    rl = true;
                }
                ".aq" => aq = true,
                _ => rl = true,
            }
            break;
        }
    }
    (rest, aq, rl)
}

fn amo_fields(mn: &str) -> Option<(u32, u32)> {
    let (base, width) = if let Some(b) = mn.strip_suffix(".d") {
        (b, 3u32)
    } else {
        (mn.strip_suffix(".w").unwrap_or(mn), 2u32)
    };
    let funct5 = match base {
        "lr" => 0x02,
        "sc" => 0x03,
        "amoswap" => 0x01,
        "amoadd" => 0x00,
        "amoxor" => 0x04,
        "amoand" => 0x0c,
        "amoor" => 0x08,
        "amomin" => 0x10,
        "amomax" => 0x14,
        "amominu" => 0x18,
        "amomaxu" => 0x1c,
        _ => return None,
    };
    Some((funct5, width))
}

fn encode_amo(enc: &mut Enc, funct5: u32, width: u32, aq: bool, rl: bool, ops: &[Operand], mn: &str) -> Result<(), String> {
    let opc = OPC_AMO;
    let f7 = (funct5 << 2) | ((aq as u32) << 1) | rl as u32;
    match ops {
        [Operand::Reg(rd), Operand::Mem { base, .. }] if funct5 == 0x02 => {
            enc.push(enc_r(f7, 0, *base, width, *rd, opc));
        }
        [Operand::Reg(rd), Operand::Reg(rs2), Operand::Mem { base, .. }] => {
            enc.push(enc_r(f7, *rs2, *base, width, *rd, opc));
        }
        _ => return Err(format!("`{mn}': bad operand forms")),
    }
    Ok(())
}

fn op_fields(mn: &str) -> Result<(u32, u32, u32), String> {
    Ok(match mn {
        "add" => (0x00, 0x0, OPC_OP),
        "sub" => (0x20, 0x0, OPC_OP),
        "sll" => (0x00, 0x1, OPC_OP),
        "slt" => (0x00, 0x2, OPC_OP),
        "sltu" => (0x00, 0x3, OPC_OP),
        "xor" => (0x00, 0x4, OPC_OP),
        "srl" => (0x00, 0x5, OPC_OP),
        "sra" => (0x20, 0x5, OPC_OP),
        "or" => (0x00, 0x6, OPC_OP),
        "and" => (0x00, 0x7, OPC_OP),
        "addw" => (0x00, 0x0, OPC_OP + 8),
        "subw" => (0x20, 0x0, OPC_OP + 8),
        "sllw" => (0x00, 0x1, OPC_OP + 8),
        "srlw" => (0x00, 0x5, OPC_OP + 8),
        "sraw" => (0x20, 0x5, OPC_OP + 8),
        _ => return Err("unreachable op_fields".into()),
    })
}

fn shift_fields(mn: &str, rv64: bool) -> Result<(u32, u32, u32), String> {
    Ok(match mn {
        "slli" => (0x00, 0x1, OPC_OP_IMM),
        "srli" => (0x00, 0x5, OPC_OP_IMM),
        "srai" => (0x20, 0x5, OPC_OP_IMM),
        "slliw" if rv64 => (0x00, 0x1, OPC_OP_IMM + 8),
        "srliw" if rv64 => (0x00, 0x5, OPC_OP_IMM + 8),
        "sraiw" if rv64 => (0x20, 0x5, OPC_OP_IMM + 8),
        _ => return Err(format!("{mn} requires RV64")),
    })
}

fn load_funct3(mn: &str, rv64: bool) -> Result<u32, String> {
    Ok(match mn {
        "lb" => 0,
        "lh" => 1,
        "lw" => 2,
        "ld" if rv64 => 3,
        "lbu" => 4,
        "lhu" => 5,
        "lwu" if rv64 => 6,
        _ => return Err(format!("{mn} not available on this ISA")),
    })
}

fn store_funct3(mn: &str) -> Result<u32, String> {
    Ok(match mn {
        "sb" => 0,
        "sh" => 1,
        "sw" => 2,
        "sd" => 3,
        _ => return Err("unreachable store_funct3".into()),
    })
}

/// CSR instructions, including the canonical aliases:
/// csrr rd, csr / csrw csr, rd / csrwi csr, uimm / ...
/// `fence` mask letters: i=8, o=4, r=2, w=1; plain integers too.
fn fence_mask(e: &Expr) -> Result<u32, String> {
    match e {
        Expr::Int(v) => Ok(*v as u32),
        Expr::Sym(name) => {
            let mut mask = 0u32;
            for c in name.bytes() {
                let bit = match c {
                    b'i' => 8,
                    b'o' => 4,
                    b'r' => 2,
                    b'w' => 1,
                    _ => return Err(format!("bad fence mask `{name}'")),
                };
                mask |= bit;
            }
            if mask == 0 {
                return Err(format!("bad fence mask `{name}'"));
            }
            Ok(mask)
        }
        _ => Err("fence operands must be constant masks".into()),
    }
}

fn encode_csr(enc: &mut Enc, mn: &str, ops: &[Operand]) -> Result<(), String> {
    // Normalize aliases to (base, rd_arg_pos, rs_arg_pos, imm_form).
    let base_rd_rs: (&str, Option<usize>, Option<usize>) = match mn {
        "csrr" => ("csrrs", Some(0), None),
        "csrw" => ("csrrw", None, Some(1)),
        "csrs" => ("csrrs", None, Some(1)),
        "csrc" => ("csrrc", None, Some(1)),
        "csrwi" => ("csrrwi", None, Some(1)),
        "csrsi" => ("csrrsi", None, Some(1)),
        "csrcsi" => ("csrrci", None, Some(1)),
        _ => (mn, Some(0), Some(2)),
    };
    let (base, rd, rs) = base_rd_rs;
    let (f3, imm_form) = match base {
        "csrrw" => (1u32, false),
        "csrrs" => (2, false),
        "csrrc" => (3, false),
        "csrrwi" => (5, true),
        "csrrsi" => (6, true),
        "csrrci" => (7, true),
        _ => return Err("unreachable csr base".into()),
    };
    let csr = match (rd, rs) {
        (Some(_), _) => csr_num(&ops[1])?,
        (None, Some(_)) => csr_num(&ops[0])?,
        _ => unreachable!(),
    };
    let word = match (rd, rs) {
        (Some(_), Some(_)) => match &ops[2] {
            Operand::Reg(r) if !imm_form => enc_r(0, 0, *r, f3, rd_of(ops)?, OPC_SYSTEM),
            Operand::Imm(Expr::Int(z)) if imm_form => {
                if !(0..=31).contains(z) {
                    return Err(format!("{mn}: zimm {z} out of range"));
                }
                enc_r(0, 0, *z as u8, f3, rd_of(ops)?, OPC_SYSTEM)
            }
            _ => return Err(err_op(base)),
        },
        (Some(_), None) => enc_r(0, 0, 0, f3, rd_of(ops)?, OPC_SYSTEM),
        (None, Some(_)) => match &ops[1] {
            Operand::Reg(r) if !imm_form => enc_r(0, 0, *r, f3, 0, OPC_SYSTEM),
            Operand::Imm(Expr::Int(z)) if imm_form => {
                if !(0..=31).contains(z) {
                    return Err(format!("{mn}: zimm {z} out of range"));
                }
                enc_r(0, 0, *z as u8, f3, 0, OPC_SYSTEM)
            }
            _ => return Err(err_op(base)),
        },
        _ => unreachable!(),
    };
    enc.push((word & !(0xfff << 20)) | ((csr & 0xfff) << 20));
    Ok(())
}

// ---------------------------------------------------------------------
// Operand helpers
// ---------------------------------------------------------------------

fn expect_ops_min(ops: &[Operand], n: usize, mn: &str) -> Result<(), String> {
    if ops.len() < n {
        return Err(format!("`{mn}': too few operands"));
    }
    Ok(())
}

fn expect_ops(ops: &[Operand], n: usize, mn: &str) -> Result<(), String> {
    if ops.len() != n {
        return Err(format!("`{mn}': expected {n} operands, got {}", ops.len()));
    }
    Ok(())
}

fn one_reg(ops: &[Operand], mn: &str) -> Result<u8, String> {
    expect_ops_min(ops, 2, mn)?;
    match &ops[0] {
        Operand::Reg(r) => Ok(*r),
        _ => Err(format!("`{mn}': first operand must be a register")),
    }
}

fn two_regs(ops: &[Operand], mn: &str) -> Result<(u8, u8), String> {
    expect_ops_min(ops, 2, mn)?;
    match (&ops[0], &ops[1]) {
        (Operand::Reg(a), Operand::Reg(b)) => Ok((*a, *b)),
        _ => Err(format!("`{mn}': operands must be registers")),
    }
}

fn three_regs(ops: &[Operand], mn: &str) -> Result<(u8, u8, u8), String> {
    expect_ops(ops, 3, mn)?;
    match (&ops[0], &ops[1], &ops[2]) {
        (Operand::Reg(a), Operand::Reg(b), Operand::Reg(c)) => Ok((*a, *b, *c)),
        _ => Err(format!("`{mn}': operands must be registers")),
    }
}

fn rd_of(ops: &[Operand]) -> Result<u8, String> {
    match &ops[0] {
        Operand::Reg(r) => Ok(*r),
        _ => Err("first operand must be a register".into()),
    }
}

fn err_op(mn: &str) -> String {
    format!("`{mn}': bad operand forms")
}

fn imm_val(e: &Expr, ctx: &EvalCtx) -> Result<Imm, String> {
    let (kind, val) = e.eval_mod(ctx)?;
    Ok(match (kind, val) {
        (None, RValue::Const(c)) => Imm::Const(c),
        (Some(ModKind::PcrelLo), val) => Imm::PcrelLo(val),
        (_kind, val) => Imm::Fix(val),
    })
}

fn imm_const(imm: &Imm, mn: &str) -> Result<i64, String> {
    match imm {
        Imm::Const(c) => Ok(*c),
        Imm::Fix(_) | Imm::PcrelLo(_) => Err(format!("`{mn}': immediate must be constant")),
    }
}

fn branch_target(e: &Expr, ctx: &EvalCtx) -> Result<Imm, String> {
    match e.eval(ctx)? {
        RValue::Const(c) => Ok(Imm::Const(c)),
        v => Ok(Imm::Fix(v)),
    }
}

/// CSR operand: name or number.
fn csr_num(op: &Operand) -> Result<u32, String> {
    match op {
        Operand::Imm(Expr::Int(v)) => {
            if !(0..=0xfff).contains(v) {
                return Err(format!("CSR number {v} out of range"));
            }
            Ok(*v as u32)
        }
        Operand::Imm(Expr::Sym(name)) => csr_by_name(name).ok_or_else(|| format!("unknown CSR `{name}'")),
        _ => Err("expected CSR name or number".into()),
    }
}

/// Common CSR names (subset of riscv-opc.h's DECLARE_CSR list).
fn csr_by_name(name: &str) -> Option<u32> {
    Some(match name {
        "fflags" => 0x001,
        "frm" => 0x002,
        "fcsr" => 0x003,
        "cycle" => 0xc00,
        "time" => 0xc01,
        "instret" => 0xc02,
        "cycleh" => 0xc80,
        "timeh" => 0xc81,
        "instreth" => 0xc82,
        "sstatus" => 0x100,
        "sie" => 0x104,
        "stvec" => 0x105,
        "sscratch" => 0x140,
        "sepc" => 0x141,
        "scause" => 0x142,
        "stval" => 0x143,
        "sip" => 0x144,
        "senvcfg" => 0x10a,
        "satp" => 0x180,
        "mstatus" => 0x300,
        "misa" => 0x301,
        "medeleg" => 0x302,
        "mideleg" => 0x303,
        "mie" => 0x304,
        "mtvec" => 0x305,
        "mcounteren" => 0x306,
        "mscratch" => 0x340,
        "mepc" => 0x341,
        "mcause" => 0x342,
        "mtval" => 0x343,
        "mip" => 0x344,
        "menvcfg" => 0x30a,
        "mcycle" => 0xb00,
        "minstret" => 0xb02,
        "mcountinhibit" => 0x320,
        "tselect" => 0x7a0,
        "tdata1" => 0x7a1,
        "tdata2" => 0x7a2,
        "tdata3" => 0x7a3,
        "dcsr" => 0x7b0,
        "dpc" => 0x7b1,
        "dscratch0" => 0x7b2,
        "dscratch1" => 0x7b3,
        "mhartid" => 0xf14,
        _ => return None,
    })
}

// ---------------------------------------------------------------------
// Emitters (with automatic compression)
// ---------------------------------------------------------------------

/// addi rd, rs, imm with c.li / c.mv / c.addi / c.addi16sp / c.addi4spn
/// / c.nop compression.
fn emit_addi(enc: &mut Enc, rd: u8, rs1: u8, imm: Imm, _ctx: &EvalCtx) -> Result<(), String> {
    if let Imm::Const(c) = &imm {
        if rs1 == 0 && rd != 0 && fits_signed(*c, 6) {
            enc.push_c(enc_ci(2, rd, *c)); // c.li
            return Ok(());
        }
        if *c == 0 && rd != 0 && rs1 != 0 && is_rvc_wide(rs1) {
            enc.push_c(enc_cr(8, false, rd, rs1)); // c.mv
            return Ok(());
        }
        if rd == rs1 && rd != 0 && *c != 0 && fits_signed(*c, 6) {
            enc.push_c(enc_ci(0, rd, *c)); // c.addi
            return Ok(());
        }
        if rd == 0 && rs1 == 0 && *c == 0 {
            enc.push_c(0x0001); // c.nop
            return Ok(());
        }
        if rd == rs1 && rd == 2 && *c != 0 && *c % 16 == 0 && *c >= -512 && *c <= 496 {
            // c.addi16sp
            let imm = *c as u32;
            let field = (((imm >> 9) & 1) << 12) | (((imm >> 4) & 1) << 11) | (((imm >> 6) & 1) << 10) | (((imm >> 7) & 3) << 8) | (((imm >> 5) & 1) << 7);
            enc.push_c(enc_ci2(3, 2, field));
            return Ok(());
        }
        if rs1 == 2 && rd != 0 && rd != 2 && *c > 0 && *c < 256 && *c % 4 == 0 {
            if let Some(rp) = rvc_reg(rd) {
                // c.addi4spn
                let imm = *c as u32;
                let field = (((imm >> 4) & 3) << 11) | (((imm >> 6) & 0xf) << 7) | (((imm >> 2) & 1) << 6) | (((imm >> 3) & 1) << 5);
                enc.push_c(enc_ciw(0, rp, field));
                return Ok(());
            }
        }
    }
    emit_i(enc, rd, rs1, imm, 0x0, OPC_OP_IMM)
}

/// andi with c.andi compression (CB format, *p regs).
fn emit_andi(enc: &mut Enc, rd: u8, rs1: u8, imm: Imm, _ctx: &EvalCtx) -> Result<(), String> {
    if let Imm::Const(c) = &imm {
        if rd == rs1 && *c != 0 && fits_signed(*c, 6) {
            if let Some(rp) = rvc_reg(rd) {
                // c.andi: CI with the andi funct2 tag at [11:10].
                let imm = *c as u32 & 0x3f;
                enc.push_c((4 << 13) | (0b10 << 10) | (((imm >> 5) & 1) << 12) | ((rp as u32) << 7) | ((imm & 0x1f) << 2) | OPC_C1);
                return Ok(());
            }
        }
    }
    emit_i(enc, rd, rs1, imm, 0x7, OPC_OP_IMM)
}

/// Generic I-type emit with a possible %lo fixup in the immediate.
fn emit_i(enc: &mut Enc, rd: u8, rs1: u8, imm: Imm, funct3: u32, opc: u32) -> Result<(), String> {
    match imm {
        Imm::Const(c) => {
            if !fits_signed(c, 12) {
                return Err(format!("immediate {c} does not fit in 12 bits"));
            }
            enc.push(enc_i(c, rs1, funct3, rd, opc));
        }
        Imm::PcrelLo(val) => {
            let RValue::SectVal { .. } = &val else {
                return Err("%pcrel_lo label must reference an auipc".into());
            };
            let word = enc_i(0, rs1, funct3, rd, opc);
            enc.bytes.extend_from_slice(&word.to_le_bytes());
            enc.fixups.push(FixupReq {
                at: enc.bytes.len() - 4,
                size: 4,
                val,
                pcrel: false,
                rel_base: 0,
                layout: FieldLayout::RvPcrelLo12I,
            });
        }
        Imm::Fix(val) => {
            let word = enc_i(0, rs1, funct3, rd, opc);
            enc.bytes.extend_from_slice(&word.to_le_bytes());
            enc.fixups.push(FixupReq {
                at: enc.bytes.len() - 4,
                size: 4,
                val,
                pcrel: false,
                rel_base: 0,
                layout: FieldLayout::RvLo12I,
            });
        }
    }
    Ok(())
}

fn emit_lui(enc: &mut Enc, rd: u8, val: RValue, pcrel: bool) -> Result<(), String> {
    match val {
        RValue::Const(c) => {
            // `lui`'s immediate is the 20-bit field itself (the loaded
            // value is sext20(c) << 12); c.lui covers sext-6 fields.
            let field = sext(c & 0xfffff, 20);
            let z = field & 0x3f;
            if rd != 0 && rd != 2 && z != 0 && sext(z, 6) == field {
                enc.push_c(enc_ci(3, rd, z));
            } else {
                enc.push(enc_u(field & 0xfffff, rd, OPC_LUI));
            }
        }
        v => {
            let word = enc_u(0, rd, OPC_LUI);
            enc.bytes.extend_from_slice(&word.to_le_bytes());
            enc.fixups.push(FixupReq {
                at: enc.bytes.len() - 4,
                size: 4,
                val: v,
                pcrel,
                rel_base: 0,
                layout: FieldLayout::RvHi20,
            });
        }
    }
    Ok(())
}

fn emit_auipc(enc: &mut Enc, rd: u8, val: RValue, _pcrel: bool, _ctx: &EvalCtx) -> Result<(), String> {
    match val {
        RValue::Const(c) => {
            enc.push(enc_u(c >> 12, rd, OPC_AUIPC));
        }
        v => {
            let word = enc_u(0, rd, OPC_AUIPC);
            enc.bytes.extend_from_slice(&word.to_le_bytes());
            enc.fixups.push(FixupReq {
                at: enc.bytes.len() - 4,
                size: 4,
                val: v,
                pcrel: true,
                rel_base: 0,
                layout: FieldLayout::RvPcrelHi20,
            });
        }
    }
    Ok(())
}

/// The `addi rd, rd, %pcrel_lo(label)` half of a pcrel pair. The fixup's
/// value marks the auipc address; finalize pairs it with the HI20 fixup
/// there and emits R_RISCV_PCREL_LO12_I against an internal symbol.
fn emit_pcrel_lo(enc: &mut Enc, rd: u8, rs1: u8, auipc_off: i64, sect: usize, store: bool) -> Result<(), String> {
    let word = enc_i(0, rs1, 0x0, rd, OPC_OP_IMM);
    enc.bytes.extend_from_slice(&word.to_le_bytes());
    enc.fixups.push(FixupReq {
        at: enc.bytes.len() - 4,
        size: 4,
        val: RValue::SectVal { sect, off: auipc_off, sym: None },
        pcrel: false,
        rel_base: 0,
        layout: if store { FieldLayout::RvPcrelLo12S } else { FieldLayout::RvPcrelLo12I },
    });
    Ok(())
}

/// call/tail's jalr half: R_RISCV_CALL_PLT covers both words.
fn emit_call_jalr(enc: &mut Enc, rd: u8, rs1: u8, val: RValue) -> Result<(), String> {
    let word = enc_i(0, rs1, 0x0, rd, OPC_JALR);
    enc.bytes.extend_from_slice(&word.to_le_bytes());
    enc.fixups.push(FixupReq {
        at: enc.bytes.len() - 8,
        size: 8,
        val,
        pcrel: false,
        rel_base: 0,
        layout: FieldLayout::RvCall,
    });
    Ok(())
}

fn emit_jalr(enc: &mut Enc, rd: u8, rs1: u8, imm: Imm, _ctx: &EvalCtx) -> Result<(), String> {
    match &imm {
        Imm::Const(0) if rd == 0 && is_rvc_wide(rs1) => {
            enc.push_c(enc_cr(8, false, rs1, 0)); // c.jr rs
            return Ok(());
        }
        Imm::Const(0) if rd == 1 && is_rvc_wide(rs1) => {
            enc.push_c(enc_cr(9, false, rs1, 0)); // c.jalr rs
            return Ok(());
        }
        Imm::PcrelLo(_) => return Err("%pcrel_lo is not a jalr immediate".into()),
        _ => {}
    }
    emit_i(enc, rd, rs1, imm, 0x0, OPC_JALR)
}

fn emit_jal(enc: &mut Enc, rd: u8, target: Imm, ctx: &EvalCtx, relax_state: u8) -> Result<(), String> {
    // j -> c.j when rd == 0 and the displacement fits the CJ window.
    if rd == 0 && relax_state == 0 {
        match &target {
            Imm::Const(t) => {
                let d = *t - ctx.cur_off;
                if (d & 1) == 0 && fits_signed(d, 12) {
                    enc.push_c(enc_cj(5, d));
                    return Ok(());
                }
            }
            Imm::Fix(_) => {
                let word = enc_cj(5, 0);
                enc.bytes.extend_from_slice(&word.to_le_bytes()[..2]);
                enc.fixups.push(FixupReq {
                    at: 0,
                    size: 2,
                    val: branch_val(&target),
                    pcrel: true,
                    rel_base: 0,
                    layout: FieldLayout::RvJal,
                });
                return Ok(());
            }
            Imm::PcrelLo(_) => {}
        }
    }
    match target {
        Imm::Const(t) => {
            let d = t - ctx.cur_off;
            if (d & 1) != 0 || !fits_signed(d, 21) {
                return Err(format!("jal target {t} out of range or misaligned"));
            }
            enc.push(enc_j(d, rd, OPC_JAL));
        }
        Imm::Fix(val) => {
            let word = enc_j(0, rd, OPC_JAL);
            enc.bytes.extend_from_slice(&word.to_le_bytes());
            enc.fixups.push(FixupReq {
                at: 0,
                size: 4,
                val,
                pcrel: true,
                rel_base: 0,
                layout: FieldLayout::RvJal,
            });
        }
        Imm::PcrelLo(_) => return Err("%pcrel_lo is not a jump target".into()),
    }
    Ok(())
}

/// Branch emission with compression and relaxation:
/// - state 0 + beqz/bnez shape + compressed reg + in range: c.beqz/c.bnez,
/// - state < 2: B-type with a fixup (estimate patched, reloc emitted),
/// - state 2: inverted branch over `jal zero, target`.
fn emit_branch(enc: &mut Enc, funct3: u32, rs1: u8, rs2: u8, target: Imm, ctx: &EvalCtx, relax_state: u8) -> Result<(), String> {
    // Compressed forms exist only for beqz/bnez.
    if relax_state == 0 && funct3 <= 1 && rs2 == 0 {
        let c_funct = if funct3 == 0 { 6 } else { 7 };
        if let Some(rp) = rvc_reg(rs1) {
            match &target {
                Imm::Const(t) => {
                    let d = *t - ctx.cur_off;
                    if (d & 1) == 0 && fits_signed(d, 9) {
                        enc.push_c(enc_cb(c_funct, rp, d));
                        return Ok(());
                    }
                }
                Imm::Fix(_) => {
                    let word = enc_cb(c_funct, rp, 0);
                    enc.bytes.extend_from_slice(&word.to_le_bytes()[..2]);
                    enc.fixups.push(FixupReq {
                        at: 0,
                        size: 2,
                        val: branch_val(&target),
                        pcrel: true,
                        rel_base: 0,
                        layout: FieldLayout::RvBranch,
                    });
                    return Ok(());
                }
                Imm::PcrelLo(_) => {}
            }
        }
    }
    if relax_state == 2 {
        // Far sequence. A branch with a compressed form inverts it over
        // a 4-byte `jal zero, target` (6 bytes total, like GNU as);
        // otherwise the inverted B-type opens the 8-byte form.
        let head: usize = if funct3 <= 1 && rs2 == 0 && rvc_reg(rs1).is_some() {
            let rp = rvc_reg(rs1).unwrap();
            let c_funct = if funct3 == 0 { 7 } else { 6 };
            enc.push_c(enc_cb(c_funct, rp, 6));
            2
        } else {
            enc.push(enc_b(8, rs2, rs1, funct3 ^ 1, OPC_BRANCH));
            4
        };
        match target {
            Imm::Const(t) => {
                let d = t - (ctx.cur_off + head as i64 + 4);
                if (d & 1) != 0 || !fits_signed(d, 21) {
                    return Err(format!("branch target {t} out of range"));
                }
                enc.push(enc_j(d, 0, OPC_JAL));
            }
            Imm::Fix(val) => {
                let word = enc_j(0, 0, OPC_JAL);
                enc.bytes.extend_from_slice(&word.to_le_bytes());
                enc.fixups.push(FixupReq {
                    at: head,
                    size: 4,
                    val,
                    pcrel: true,
                    rel_base: head as u32,
                    layout: FieldLayout::RvJal,
                });
            }
            Imm::PcrelLo(_) => return Err("%pcrel_lo is not a branch target".into()),
        }
        return Ok(());
    }
    match target {
        Imm::Const(t) => {
            let d = t - ctx.cur_off;
            if (d & 1) != 0 || !fits_signed(d, 13) {
                return Err(format!("branch target {t} out of range or misaligned"));
            }
            enc.push(enc_b(d, rs2, rs1, funct3, OPC_BRANCH));
        }
        Imm::Fix(val) => {
            let word = enc_b(0, rs2, rs1, funct3, OPC_BRANCH);
            enc.bytes.extend_from_slice(&word.to_le_bytes());
            enc.fixups.push(FixupReq {
                at: 0,
                size: 4,
                val,
                pcrel: true,
                rel_base: 0,
                layout: FieldLayout::RvBranch,
            });
        }
        Imm::PcrelLo(_) => return Err("%pcrel_lo is not a branch target".into()),
    }
    Ok(())
}

fn branch_val(target: &Imm) -> RValue {
    match target {
        Imm::Fix(v) => v.clone(),
        Imm::Const(c) => RValue::Const(*c),
        Imm::PcrelLo(_) => unreachable!("checked by the encoders"),
    }
}

/// Loads: `lw rd, imm(rs)` direct (with c.* compression); `lw rd, sym`
/// expands to auipc rd, %pcrel_hi(sym); lw rd, %pcrel_lo(here)(rd).
fn emit_load(enc: &mut Enc, rd: u8, op: &Operand, funct3: u32, ctx: &EvalCtx) -> Result<(), String> {
    match op {
        Operand::Mem { base, disp } => {
            match disp.eval_mod(ctx)? {
                (None, RValue::Const(c)) => {
                    if !fits_signed(c, 12) {
                        return Err(format!("load offset {c} out of range"));
                    }
                    enc.push(enc_i(c, *base, funct3, rd, OPC_LOAD));
                    compress_load(enc, rd, *base, c, funct3);
                }
                (Some(ModKind::Lo), v) => {
                    let word = enc_i(0, *base, funct3, rd, OPC_LOAD);
                    enc.bytes.extend_from_slice(&word.to_le_bytes());
                    enc.fixups.push(FixupReq {
                        at: 0,
                        size: 4,
                        val: v,
                        pcrel: false,
                        rel_base: 0,
                        layout: FieldLayout::RvLo12I,
                    });
                }
                (Some(ModKind::PcrelLo), v) => {
                    let RValue::SectVal { .. } = &v else {
                        return Err("%pcrel_lo label must reference an auipc".into());
                    };
                    let word = enc_i(0, *base, funct3, rd, OPC_LOAD);
                    enc.bytes.extend_from_slice(&word.to_le_bytes());
                    enc.fixups.push(FixupReq {
                        at: 0,
                        size: 4,
                        val: v,
                        pcrel: false,
                        rel_base: 0,
                        layout: FieldLayout::RvPcrelLo12I,
                    });
                }
                _ => return Err("load offset must be constant or %lo".into()),
            }
            Ok(())
        }
        Operand::Imm(e) => {
            // `lw rd, symbol` address-formation pseudo.
            let val = e.eval(ctx)?;
            let here = ctx.cur_off;
            emit_auipc(enc, rd, val, true, ctx)?;
            let word = enc_i(0, rd, funct3, rd, OPC_LOAD);
            enc.bytes.extend_from_slice(&word.to_le_bytes());
            enc.fixups.push(FixupReq {
                at: 4,
                size: 4,
                val: RValue::SectVal {
                    sect: ctx.cur_sect,
                    off: here,
                    sym: None,
                },
                pcrel: false,
                rel_base: 0,
                layout: FieldLayout::RvPcrelLo12I,
            });
            Ok(())
        }
        Operand::Reg(_) => Err("load needs a memory operand".into()),
    }
}

/// c.lwsp/c.ldsp/c.lw/c.ld compression for plain loads.
fn compress_load(enc: &mut Enc, rd: u8, base: u8, disp: i64, funct3: u32) {
    if rd == 0 {
        return;
    }
    match (funct3, base) {
        (2, 2) => {
            // c.lwsp: uimm[5]@[12], uimm[4:2]@[6:4], uimm[7:6]@[3:2].
            if disp >= 0 && (disp & 3) == 0 && disp < 256 {
                let u = disp as u32;
                let field = (((u >> 5) & 1) << 12) | (((u >> 2) & 7) << 4) | (((u >> 6) & 3) << 2);
                replace_with_compressed(enc, enc_ci2(2, rd, field));
            }
        }
        (3, 2) => {
            // c.ldsp: uimm[5]@[12], uimm[4:3]@[6:5], uimm[8:6]@[4:2].
            if disp >= 0 && (disp & 7) == 0 && disp < 512 {
                let u = disp as u32;
                let field = (((u >> 5) & 1) << 12) | (((u >> 3) & 3) << 5) | (((u >> 6) & 7) << 2);
                replace_with_compressed(enc, enc_ci2(3, rd, field));
            }
        }
        (2, 8..=15) => {
            // c.lw: uimm[5:3]@[12:10], uimm[2]@[6], uimm[6]@[5].
            if let (Some(rdp), Some(rs1p)) = (rvc_reg(rd), rvc_reg(base)) {
                if disp >= 0 && (disp & 3) == 0 && disp < 128 {
                    let u = disp as u32;
                    let field = (((u >> 3) & 7) << 10) | (((u >> 2) & 1) << 6) | (((u >> 6) & 1) << 5);
                    replace_with_compressed(enc, enc_cl(2, rdp, field, rs1p));
                }
            }
        }
        (3, 8..=15) => {
            // c.ld: uimm[5:3]@[12:10], uimm[7:6]@[6:5].
            if let (Some(rdp), Some(rs1p)) = (rvc_reg(rd), rvc_reg(base)) {
                if disp >= 0 && (disp & 7) == 0 && disp < 256 {
                    let u = disp as u32;
                    let field = (((u >> 3) & 7) << 10) | (((u >> 6) & 3) << 6);
                    replace_with_compressed(enc, enc_cl(3, rdp, field, rs1p));
                }
            }
        }
        _ => {}
    }
}

/// Stores: `sw rs2, imm(rs1)`; `sw rs2, sym, rt` expands via auipc rt.
fn emit_store(enc: &mut Enc, ops: &[Operand], funct3: u32, ctx: &EvalCtx) -> Result<(), String> {
    let Operand::Reg(rs2) = ops[0] else {
        return Err("store: first operand must be a register".into());
    };
    match &ops[1] {
        Operand::Mem { base, disp } => {
            match disp.eval_mod(ctx)? {
                (None, RValue::Const(c)) => {
                    if !fits_signed(c, 12) {
                        return Err(format!("store offset {c} out of range"));
                    }
                    enc.push(enc_s(c, rs2, *base, funct3, OPC_STORE));
                    compress_store(enc, rs2, *base, c, funct3);
                }
                (Some(ModKind::Lo), v) => {
                    let word = enc_s(0, rs2, *base, funct3, OPC_STORE);
                    enc.bytes.extend_from_slice(&word.to_le_bytes());
                    enc.fixups.push(FixupReq {
                        at: 0,
                        size: 4,
                        val: v,
                        pcrel: false,
                        rel_base: 0,
                        layout: FieldLayout::RvLo12S,
                    });
                }
                (Some(ModKind::PcrelLo), v) => {
                    let RValue::SectVal { .. } = &v else {
                        return Err("%pcrel_lo label must reference an auipc".into());
                    };
                    let word = enc_s(0, rs2, *base, funct3, OPC_STORE);
                    enc.bytes.extend_from_slice(&word.to_le_bytes());
                    enc.fixups.push(FixupReq {
                        at: 0,
                        size: 4,
                        val: v,
                        pcrel: false,
                        rel_base: 0,
                        layout: FieldLayout::RvPcrelLo12S,
                    });
                }
                _ => return Err("store offset must be constant or %lo".into()),
            }
            Ok(())
        }
        Operand::Imm(e) => {
            // `sw rs2, symbol[, rt]`: auipc rt, %pcrel_hi; sw rs2, %pcrel_lo(rt).
            let rt = match ops.get(2) {
                Some(Operand::Reg(r)) => *r,
                _ => {
                    if rs2 != 6 && rs2 != 0 {
                        6 // t1
                    } else {
                        5 // t0, keeps rs2 usable as the store source
                    }
                }
            };
            let val = e.eval(ctx)?;
            let here = ctx.cur_off;
            emit_auipc(enc, rt, val, true, ctx)?;
            let word = enc_s(0, rs2, rt, funct3, OPC_STORE);
            enc.bytes.extend_from_slice(&word.to_le_bytes());
            enc.fixups.push(FixupReq {
                at: 4,
                size: 4,
                val: RValue::SectVal {
                    sect: ctx.cur_sect,
                    off: here,
                    sym: None,
                },
                pcrel: false,
                rel_base: 0,
                layout: FieldLayout::RvPcrelLo12S,
            });
            Ok(())
        }
        Operand::Reg(_) => Err("store needs a memory operand".into()),
    }
}

/// c.swsp/c.sdsp/c.sw/c.sd compression.
fn compress_store(enc: &mut Enc, rs2: u8, base: u8, disp: i64, funct3: u32) {
    match (funct3, base) {
        (2, 2) => {
            // c.swsp: uimm[5:2]@[12:9], uimm[7:6]@[8:7].
            if disp >= 0 && (disp & 3) == 0 && disp < 256 {
                let u = disp as u32;
                let field = (((u >> 2) & 0xf) << 9) | (((u >> 6) & 3) << 7);
                replace_with_compressed(enc, enc_css(6, rs2, field));
            }
        }
        (3, 2) => {
            // c.sdsp: uimm[5:3]@[12:10], uimm[8:6]@[9:7].
            if disp >= 0 && (disp & 7) == 0 && disp < 512 {
                let u = disp as u32;
                let field = (((u >> 3) & 7) << 10) | (((u >> 6) & 7) << 7);
                replace_with_compressed(enc, enc_css(7, rs2, field));
            }
        }
        (2, 8..=15) => {
            // c.sw: same field layout as c.lw.
            if let (Some(rsp), Some(rs1p)) = (rvc_reg(rs2), rvc_reg(base)) {
                if disp >= 0 && (disp & 3) == 0 && disp < 128 {
                    let u = disp as u32;
                    let field = (((u >> 3) & 7) << 10) | (((u >> 2) & 1) << 6) | (((u >> 6) & 1) << 5);
                    replace_with_compressed(enc, enc_cs(6, rsp, field, rs1p));
                }
            }
        }
        (3, 8..=15) => {
            // c.sd: same field layout as c.ld.
            if let (Some(rsp), Some(rs1p)) = (rvc_reg(rs2), rvc_reg(base)) {
                if disp >= 0 && (disp & 7) == 0 && disp < 256 {
                    let u = disp as u32;
                    let field = (((u >> 3) & 7) << 10) | (((u >> 6) & 3) << 6);
                    replace_with_compressed(enc, enc_cs(7, rsp, field, rs1p));
                }
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------
// li: the load_const recursion (gas/config/tc-riscv.c)
// ---------------------------------------------------------------------

fn load_const(enc: &mut Enc, reg: u8, value: i64, ctx: &EvalCtx, rv64: bool) -> Result<(), String> {
    // GNU as's load_const: split off the sign-extended low 12 bits, then
    // either lui+addiw the rest or recurse through slli chunks. RV32
    // constants truncate to 32 bits first.
    let value = if rv64 { value } else { value as i32 as i64 };
    let lower = sext(value, 12);
    let upper = value - lower;

    if rv64 && !is_sext_32bit(value) {
        let mut shift = 12u32;
        while (upper >> shift) & 1 == 0 {
            shift += 1;
        }
        let up = upper >> shift;
        load_const(enc, reg, up, ctx, rv64)?;
        enc.push(enc_r(0, (shift & 0x3f) as u8, reg, 0x1, reg, OPC_OP_IMM));
        if reg != 0 {
            let c = (((shift as u32) >> 5) << 12) | ((rd_field(reg)) << 7) | (((shift as u32) & 0x1f) << 2) | OPC_C2;
            replace_with_compressed(enc, c);
        }
        if lower != 0 {
            emit_addi(enc, reg, reg, Imm::Const(lower), ctx)?;
        }
    } else {
        if upper != 0 {
            emit_lui(enc, reg, RValue::Const((upper >> 12) & 0xfffff), false)?;
        }
        if lower != 0 || upper == 0 {
            let rs1 = if upper != 0 { reg } else { 0 };
            if rv64 {
                if rs1 == 0 {
                    // GAS keeps `addiw reg, zero, imm` uncompressed.
                    emit_i(enc, reg, 0, Imm::Const(lower), 0x0, OPC_OP_IMM + 8)?;
                } else if reg != 0 && fits_signed(lower, 6) {
                    enc.push_c(enc_ci(1, reg, lower)); // c.addiw
                } else {
                    emit_i(enc, reg, rs1, Imm::Const(lower), 0x0, OPC_OP_IMM + 8)?;
                }
            } else if rs1 == 0 {
                // GAS keeps load_const's x0-form addi uncompressed.
                emit_i(enc, reg, 0, Imm::Const(lower), 0x0, OPC_OP_IMM)?;
            } else {
                emit_addi(enc, reg, rs1, Imm::Const(lower), ctx)?;
            }
        }
    }
    Ok(())
}

fn rd_field(reg: u8) -> u32 {
    (reg as u32) & 0x1f
}

fn is_sext_32bit(v: i64) -> bool {
    (v & !0x7fff_ffff) == 0 || (v & !0x7fff_ffff) == !0x7fff_ffff
}

// ---------------------------------------------------------------------
// Finalization: fixup patching and relocation emission
// ---------------------------------------------------------------------

fn finalize_fixup(
    data: &mut [u8],
    f: &Fixup,
    hi_targets: &HashMap<(usize, i64), RValue>,
    symbols: &mut Vec<Symbol>,
    relocs: &mut Vec<Reloc>,
    relax_sym: usize,
) -> Result<(), Vec<String>> {
    match f.layout {
        FieldLayout::RvCall => {
            // call/tail: R_RISCV_CALL_PLT over both words + RELAX, with
            // GNU as's hi20 estimate pre-patched into the auipc.
            match &f.val {
                RValue::Const(_) => Err(vec!["call/tail target must be a symbol".into()]),
                v => {
                    let delta = sym_off(v) - f.offset;
                    let hi = ((delta + 0x800) >> 12) & 0xfffff;
                    patch_word(data, f.offset as usize, |w| {
                        let rd = ((w >> 7) & 0x1f) as u8;
                        enc_u(hi, rd, OPC_AUIPC)
                    });
                    let addend = push_reloc(relocs, symbols, f, v, R_RISCV_CALL_PLT);
                    push_relax(relocs, f, relax_sym, addend);
                    Ok(())
                }
            }
        }
        FieldLayout::RvBranch => {
            let width_bits = if f.size == 2 { 9 } else { 13 };
            match resolve_delta(f) {
                Some(d) if (d & 1) == 0 && fits_signed(d, width_bits) => {
                    patch_branch(data, f, d)?;
                    // GNU as keeps the relocation with the estimate
                    // pre-patched in the word.
                    if !matches!(f.val, RValue::Const(_)) {
                        let r_type = if f.size == 2 { R_RISCV_RVC_BRANCH } else { R_RISCV_BRANCH };
                        push_reloc(relocs, symbols, f, &f.val, r_type);
                    }
                    Ok(())
                }
                Some(d) => Err(vec![format!("branch displacement {d} out of range")]),
                None => {
                    // Unknown target (extern / other section): patch in
                    // GNU as's estimate, leave the relocation behind.
                    let est = est_delta(f);
                    patch_branch(data, f, est)?;
                    let r_type = if f.size == 2 { R_RISCV_RVC_BRANCH } else { R_RISCV_BRANCH };
                    push_reloc(relocs, symbols, f, &f.val, r_type);
                    Ok(())
                }
            }
        }
        FieldLayout::RvJal => {
            let width_bits = if f.size == 2 { 12 } else { 21 };
            match resolve_delta(f) {
                Some(d) if (d & 1) == 0 && fits_signed(d, width_bits) => {
                    if f.size == 2 {
                        let w = enc_cj(5, d);
                        data[f.offset as usize..f.offset as usize + 2].copy_from_slice(&w.to_le_bytes()[..2]);
                    } else {
                        patch_word(data, f.offset as usize, |w| {
                            let rd = ((w >> 7) & 0x1f) as u8;
                            enc_j(d, rd, OPC_JAL)
                        });
                    }
                    if !matches!(f.val, RValue::Const(_)) {
                        let r_type = if f.size == 2 { R_RISCV_RVC_JAL } else { R_RISCV_JAL };
                        push_reloc(relocs, symbols, f, &f.val, r_type);
                    }
                    Ok(())
                }
                Some(d) => Err(vec![format!("jal displacement {d} out of range")]),
                None => {
                    let est = est_delta(f);
                    if f.size == 2 {
                        let w = enc_cj(5, est);
                        data[f.offset as usize..f.offset as usize + 2].copy_from_slice(&w.to_le_bytes()[..2]);
                    } else {
                        patch_word(data, f.offset as usize, |w| {
                            let rd = ((w >> 7) & 0x1f) as u8;
                            enc_j(est, rd, OPC_JAL)
                        });
                    }
                    let r_type = if f.size == 2 { R_RISCV_RVC_JAL } else { R_RISCV_JAL };
                    push_reloc(relocs, symbols, f, &f.val, r_type);
                    Ok(())
                }
            }
        }
        FieldLayout::RvPcrelHi20 | FieldLayout::RvHi20 => match &f.val {
            RValue::Const(c) => {
                let imm20 = ((high_part(*c) >> 12) & 0xfffff) as u32;
                patch_word(data, f.offset as usize, |w| {
                    let rd = (w >> 7) & 0x1f;
                    let opc = w & 0x7f;
                    enc_u(imm20 as i64, rd as u8, opc)
                });
                Ok(())
            }
            v => {
                let rtype = if f.layout == FieldLayout::RvPcrelHi20 { R_RISCV_PCREL_HI20 } else { R_RISCV_HI20 };
                let addend = push_reloc(relocs, symbols, f, v, rtype);
                push_relax(relocs, f, relax_sym, addend);
                Ok(())
            }
        },
        FieldLayout::RvLo12I | FieldLayout::RvLo12S => finalize_lo12(data, f, symbols, relocs, relax_sym),
        FieldLayout::RvPcrelLo12I | FieldLayout::RvPcrelLo12S => {
            let RValue::SectVal { sect, off, sym } = &f.val else {
                return Err(vec!["internal error: bad pcrel_lo marker".into()]);
            };
            let Some(hi_val) = hi_targets.get(&(*sect, *off)) else {
                return Err(vec!["internal error: missing %pcrel_hi pair".into()]);
            };
            // lo = sign-adjusted low 12 bits of (hi_target - auipc_pc);
            // unresolved (extern / other-section) targets keep GNU as's
            // zero estimate.
            let lo = match hi_val {
                RValue::SectVal { sect: hs, .. } if *hs == f.section => {
                    let delta = sym_off(hi_val) - off;
                    ((delta + 0x800) & 0xfff) - 0x800
                }
                _ => 0,
            };
            let is_i = f.layout == FieldLayout::RvPcrelLo12I;
            patch_lo12(data, f, lo, is_i)?;
            // R_RISCV_PCREL_LO12_I/S against the label the operand
            // named; `la`-style internal pairs have no such symbol and
            // get one placed at the auipc (GNU as's `.L0 `).
            let target = match sym {
                Some(i) => *i,
                None => intern_at(symbols, *sect, *off),
            };
            relocs.push(Reloc {
                section: f.section,
                offset: f.offset,
                size: f.size,
                pcrel: f.pcrel,
                addend: 0,
                target: Target::Symbol(target),
                r_type: if is_i { R_RISCV_PCREL_LO12_I } else { R_RISCV_PCREL_LO12_S },
            });
            push_relax(relocs, f, relax_sym, 0);
            Ok(())
        }
        _ => Err(vec!["internal error: unexpected fixup layout".into()]),
    }
}

fn finalize_lo12(data: &mut [u8], f: &Fixup, symbols: &mut Vec<Symbol>, relocs: &mut Vec<Reloc>, relax_sym: usize) -> Result<(), Vec<String>> {
    let is_i = f.layout == FieldLayout::RvLo12I;
    match &f.val {
        RValue::Const(c) => {
            patch_lo12(data, f, *c & 0xfff, is_i)?;
            Ok(())
        }
        v => {
            let rtype = if is_i { R_RISCV_LO12_I } else { R_RISCV_LO12_S };
            // GNU as pre-patches the field with its estimate.
            let est = sym_off(v) & 0xfff;
            patch_lo12(data, f, est, is_i)?;
            let addend = push_reloc(relocs, symbols, f, v, rtype);
            push_relax(relocs, f, relax_sym, addend);
            Ok(())
        }
    }
}

fn patch_lo12(data: &mut [u8], f: &Fixup, imm: i64, is_i: bool) -> Result<(), Vec<String>> {
    let pos = f.offset as usize;
    if is_i {
        patch_word(data, pos, |w| (w & !(0xfff << 20)) | (((imm as u32) & 0xfff) << 20));
    } else {
        patch_word(data, pos, |w| {
            let funct3 = (w >> 12) & 0x7;
            let rs2 = ((w >> 20) & 0x1f) as u8;
            let rs1 = ((w >> 15) & 0x1f) as u8;
            enc_s(imm, rs2, rs1, funct3, OPC_STORE)
        });
    }
    Ok(())
}

fn patch_branch(data: &mut [u8], f: &Fixup, d: i64) -> Result<(), Vec<String>> {
    let pos = f.offset as usize;
    if f.size == 2 {
        let (funct, rs1p) = {
            let lo = u16::from_le_bytes([data[pos], data[pos + 1]]);
            ((lo >> 13) & 0x7, ((lo >> 7) & 0x7) as u8)
        };
        let w = enc_cb(funct as u32, rs1p, d);
        data[pos..pos + 2].copy_from_slice(&w.to_le_bytes()[..2]);
    } else {
        patch_word(data, pos, |w| {
            let funct3 = (w >> 12) & 0x7;
            let rs2 = ((w >> 20) & 0x1f) as u8;
            let rs1 = ((w >> 15) & 0x1f) as u8;
            enc_b(d, rs2, rs1, funct3, OPC_BRANCH)
        });
    }
    Ok(())
}

fn patch_word(data: &mut [u8], at: usize, f: impl FnOnce(u32) -> u32) {
    let w = u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
    data[at..at + 4].copy_from_slice(&f(w).to_le_bytes());
}

/// The displacement for pcrel fixups when it can be resolved statically.
fn resolve_delta(f: &Fixup) -> Option<i64> {
    if !f.pcrel {
        return None;
    }
    match &f.val {
        RValue::Const(c) => Some(*c),
        RValue::SectVal { sect, off, .. } if *sect == f.section => Some(off - f.pcrel_base()),
        _ => None,
    }
}

/// GNU as's in-place estimate for an unresolved pc-relative target:
/// the symbol value (0 for externs) relative to the field's base.
fn est_delta(f: &Fixup) -> i64 {
    match &f.val {
        RValue::ExternRef { off, .. } | RValue::SectVal { off, .. } | RValue::RelDiff { off, .. } | RValue::RelDiffExtern { off, .. } => off - f.pcrel_base(),
        RValue::Const(_) => 0,
    }
}

fn sym_off(v: &RValue) -> i64 {
    match v {
        RValue::Const(c) => *c,
        RValue::SectVal { off, .. } | RValue::RelDiff { off, .. } | RValue::RelDiffExtern { off, .. } => *off,
        RValue::ExternRef { off, .. } => *off,
    }
}

/// Interns a backend-synthesized local symbol at (sect, off), reusing
/// one if it already exists.
fn intern_at(symbols: &mut Vec<Symbol>, sect: usize, off: i64) -> usize {
    if let Some(i) = symbols.iter().position(|s| s.name == ".L0 " && s.sect == Some(sect) && s.value == off) {
        return i;
    }
    symbols.push(Symbol {
        name: ".L0 ".into(),
        defined: true,
        seen: true,
        extern_: false,
        global: false,
        weak: false,
        sect: Some(sect),
        value: off,
    });
    symbols.len() - 1
}

/// Emits a relocation for `val` the way GNU as does: against the
/// referenced symbol (addend = value offset past it), falling back to
/// the section for unnamed section offsets. Returns the addend so
/// companion RELAX markers can carry it.
fn push_reloc(relocs: &mut Vec<Reloc>, symbols: &[Symbol], f: &Fixup, val: &RValue, r_type: u32) -> i64 {
    let (target, addend) = match val {
        RValue::ExternRef { sym, off } => {
            let base = symbols.get(*sym).map(|s| s.value).unwrap_or(0);
            (Target::Symbol(*sym), *off - base)
        }
        RValue::SectVal { off, sym: Some(i), .. } => {
            let base = symbols.get(*i).map(|s| s.value).unwrap_or(0);
            (Target::Symbol(*i), *off - base)
        }
        RValue::SectVal { sect, off, sym: None } => (Target::Section(*sect), *off),
        RValue::RelDiff { sect, off, sym } => (match sym {
            Some(i) => Target::Symbol(*i),
            None => Target::Section(*sect),
        }, *off),
        RValue::RelDiffExtern { sym, off } => (Target::Symbol(*sym), *off),
        RValue::Const(_) => return 0,
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
    addend
}

fn push_relax(relocs: &mut Vec<Reloc>, f: &Fixup, relax_sym: usize, addend: i64) {
    relocs.push(Reloc {
        section: f.section,
        offset: f.offset,
        size: 0,
        pcrel: false,
        addend,
        target: Target::Symbol(relax_sym),
        r_type: R_RISCV_RELAX,
    });
}
