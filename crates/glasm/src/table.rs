//! Table-driven instruction matching and encoding, ported from nasm's
//! `asm/assemble.c` (find_match/matches/jmp_match/calcsize/gencode and
//! process_ea) over the generated instruction tables in `insns_gen`.

use crate::encoding::{Encoded, FixupReq};
use crate::expr::{EvalCtx, Expr, RValue};
use crate::insn::{eaf, op, Insn, Operand};
use crate::insns_gen::{self, if_, Template};
use std::cell::Cell;

// REX bits (nasm.h).
#[allow(non_upper_case_globals)]
const REX_B: u32 = 0x01;
const REX_X: u32 = 0x02;
const REX_R: u32 = 0x04;
const REX_W: u32 = 0x08;
const REX_L: u32 = 0x20;
const REX_P: u32 = 0x40;
const REX_H: u32 = 0x80;
const REX_V: u32 = 0x0100;
const REX_NH: u32 = 0x0200;
const REX_EV: u32 = 0x0400;
const REX_B1: u32 = 0x1000;
const REX_X1: u32 = 0x2000;
const REX_R1: u32 = 0x4000;
const REX_NW: u32 = 0x8000;
const REX_BV: u32 = 0x10000;
const REX_XV: u32 = 0x20000;
const REX_RV: u32 = 0x40000;
const REX_MASK: u32 = 0x4f;
#[allow(non_upper_case_globals)]
const REX_rB: u32 = REX_B | REX_B1 | REX_BV | REX_H | REX_P;
#[allow(non_upper_case_globals)]
const REX_rX: u32 = REX_X | REX_X1 | REX_XV | REX_H | REX_P;
#[allow(non_upper_case_globals)]
const REX_rR: u32 = REX_R | REX_R1 | REX_RV | REX_H | REX_P;

// iflag masks (bit indices from insns_gen::if_).
const IF_TSMASK: u32 = (1 << 18) - (1 << 10); // IF_SB .. IF_SZ
const IF_SMASK: u32 = (1 << 23) - (1 << 10); // IF_SB .. IF_SX
const IF_ARMASK: u32 = (1 << 10) - (1 << 5); // IF_AR0 .. IF_AR4
const IF_SMMASK: u32 = (1 << 5) - 1; // IF_SM0 .. IF_SM4
const IFM_OSIZE: u32 = 1 << if_::OSIZE;
const IFM_ASIZE: u32 = 1 << if_::ASIZE;
/// CPU-level field mask: everything up to IF_ANY.
const CPU_LEVEL_MASK: u32 = (1 << (if_::ANY & 31)) - 1;
/// x86-64 baseline CPU level (CPU field).
const CPU_LEVEL_X86_64: u32 = 1 << (if_::X86_64 & 31);

fn itemp_has(t: &Template, bit: u32) -> bool {
    t.f[(bit >> 5) as usize] & (1 << (bit & 31)) != 0
}

fn itemp_smask(t: &Template) -> u32 {
    t.f[0] & IF_SMASK
}

fn itemp_arx(t: &Template) -> u32 {
    (t.f[0] & IF_ARMASK) >> if_::AR0
}

fn itemp_smx(t: &Template) -> u32 {
    (t.f[0] & IF_SMMASK) >> if_::SM0
}

fn mode_to_op(bits: u8) -> u64 {
    (bits as u64) << (32 - 3)
}

// Prefix states (subset of nasm's `enum prefixes`).
const P_NONE: i32 = 0;
#[allow(dead_code)]
const P_A16: i32 = 1;
#[allow(dead_code)]
const P_A32: i32 = 2;
const P_A64: i32 = 3;
const P_ASP: i32 = 4;
const P_LOCK: i32 = 5;
const P_O16: i32 = 6;
const P_O32: i32 = 7;
const P_O64: i32 = 8;
const P_OSP: i32 = 9;
const P_REP: i32 = 10;
const P_REPE: i32 = 11;
const P_REPNE: i32 = 12;
const P_REPNZ: i32 = 13;
const P_REPZ: i32 = 14;
const P_WAIT: i32 = 15;
const P_XACQUIRE: i32 = 16;
const P_XRELEASE: i32 = 17;
const P_BND: i32 = 18;
const P_NOBND: i32 = 19;
const P_REX: i32 = 20;
const P_REX2: i32 = 21;
const P_EVEX: i32 = 22;
const P_VEX: i32 = 23;
const P_VEX3: i32 = 24;
const P_VEX2: i32 = 25;
const P_NF: i32 = 26;
const P_ZU: i32 = 27;

const PFE_NULL: i32 = -1;
const PFE_ERR: i32 = -2;
#[allow(dead_code)]
const PFE_WHAT: i32 = -3;
#[allow(dead_code)]
const PFE_MULTI: i32 = -4;

fn prefix_byte(pfx: i32, bits: u8) -> i32 {
    match pfx {
        P_WAIT => 0x9b,
        P_LOCK => 0xf0,
        P_REPNE | P_REPNZ | P_XACQUIRE | P_BND => 0xf2,
        P_REPE | P_REPZ | P_REP | P_XRELEASE => 0xf3,
        P_A16 => {
            if bits == 64 {
                PFE_ERR
            } else if bits == 32 {
                0x67
            } else {
                PFE_NULL
            }
        }
        P_A32 => {
            if bits != 32 {
                0x67
            } else {
                PFE_NULL
            }
        }
        P_ASP => 0x67,
        P_O16 => {
            if bits != 16 {
                0x66
            } else {
                PFE_NULL
            }
        }
        P_O32 => {
            if bits == 16 {
                0x66
            } else {
                PFE_NULL
            }
        }
        P_O64 => {
            if bits == 64 {
                PFE_NULL
            } else {
                PFE_ERR
            }
        }
        P_OSP => 0x66,
        P_REX | P_VEX | P_EVEX | P_VEX3 | P_VEX2 | P_REX2 => PFE_MULTI,
        P_NF | P_NOBND | P_ZU | P_NONE => PFE_NULL,
        _ => PFE_WHAT,
    }
}

/// An evaluated operand value: absolute, section-relative or external.
#[derive(Clone, Copy, Debug, Default)]
pub struct Val {
    pub value: i64,
    /// Section index the value is relative to; None means absolute.
    pub segment: Option<usize>,
    /// Symbol index for extern references.
    pub extern_sym: Option<usize>,
    /// A forward reference that wasn't resolvable this pass (nasm's
    /// OPFLAG_UNKNOWN).
    pub unknown: bool,
    /// Plain absolute constant.
    pub known: bool,
}

impl Val {
    fn from_rvalue(rv: &RValue) -> Val {
        match rv {
            RValue::Const(v) => Val { value: *v, known: true, ..Default::default() },
            RValue::SectVal { sect, off } => {
                Val { value: *off, segment: Some(*sect), ..Default::default() }
            }
            RValue::ExternRef { sym, off } => {
                Val { value: *off, extern_sym: Some(*sym), ..Default::default() }
            }
        }
    }

    fn to_rvalue(&self) -> RValue {
        if let Some(sym) = self.extern_sym {
            RValue::ExternRef { sym, off: self.value }
        } else if let Some(sect) = self.segment {
            RValue::SectVal { sect, off: self.value }
        } else {
            RValue::Const(self.value)
        }
    }
}

/// The evaluated form of an `Operand` — nasm's `struct operand`.
#[derive(Clone, Debug)]
struct NOperand {
    ty: u64,
    basereg: i32,
    indexreg: i32,
    scale: u8,
    eaflags: u32,
    /// Immediate or displacement value (nasm's single `offset` field).
    off: Val,
}

#[derive(Default, Clone)]
struct Ea {
    modrm: u8,
    sib: u8,
    sib_present: bool,
    bytes: u32,
    disp8: i8,
    disp8_ok: bool,
    rip: bool,
    rex: u32,
}

/// nasm's `struct insn` subset.
struct InsState {
    bits: u8,
    op_size: u8,
    addr_size: u8,
    rex: u32,
    /// Computed prefix states, indexed by prefix slot.
    prefixes: [i32; 9],
    /// Prefix requirements from the encoding (SSE-style extensions).
    need_pfx: [i32; 9],
    ea: Ea,
    ops: Vec<NOperand>,
    loc_sect: usize,
    loc_off: i64,
    /// Legacy map number from the 0355-0357 bytecodes (0 = none).
    vex_cm: u8,
}

// Prefix slots, in emit order.
const PPS_WAIT: usize = 0;
#[allow(dead_code)]
const PPS_SEG: usize = 1;
const PPS_ASIZE: usize = 2;
const PPS_LOCK: usize = 3;
const PPS_OSIZE: usize = 4;
const PPS_REP: usize = 5;
#[allow(dead_code)]
const PPS_REX: usize = 6;
#[allow(dead_code)]
const PPS_NF: usize = 7;
#[allow(dead_code)]
const PPS_ZU: usize = 8;

// match_result ranking: errors low, MOK_* above all; a higher rank wins.
const MERR_INVALOP: i32 = 0;
const MERR_OPSIZEINVAL: i32 = 1;
const MERR_OPSIZEMISSING: i32 = 2;
#[allow(dead_code)]
const MERR_OPSIZEMISMATCH: i32 = 3;
const MERR_BADCPU: i32 = 12;
const MERR_BADMODE: i32 = 13;
#[allow(dead_code)]
const MERR_ENCMISMATCH: i32 = 15;
const MERR_REGSETSIZE: i32 = 18;
const MERR_REGSET: i32 = 19;
#[allow(dead_code)]
const MERR_WRONGIMM: i32 = 20;
const MOK_FIRST: i32 = 21;
const MOK_GOOD: i32 = 30;

fn regflag(o: &NOperand) -> u64 {
    insns_gen::REGS[o.basereg as usize].flags
}

fn regval(o: &NOperand) -> i32 {
    if o.ty & op::IMMEDIATE != 0 {
        return o.off.value as i32;
    }
    insns_gen::REGS[o.basereg as usize].val
}

fn is_register(id: i32) -> bool {
    id >= 0
}

fn rexflags(val: i32, flags: u64, mask: u32) -> u32 {
    if val < 0 || flags & op::REGISTER == 0 {
        return 0;
    }
    let mut rex = 0u32;
    if val & 8 != 0 {
        rex |= REX_B | REX_X | REX_R;
    }
    if val & 16 != 0 {
        rex |= REX_B1 | REX_X1 | REX_R1;
    }
    if flags & op::REG_CLASS_VECTOR != 0 {
        rex |= REX_BV | REX_XV | REX_RV;
    } else if flags & (op::REG_GPR | op::BITS8) == (op::REG_GPR | op::BITS8) {
        if op::REG_HIGH & !flags == 0 {
            // AH, CH, DH, BH: REX forbidden.
            rex |= REX_H;
        } else if val >= 4 {
            // SPL, BPL, SIL, DIL, or extended: prefix required.
            rex |= REX_P;
        }
    }
    rex & mask
}

fn op_rexflags(o: &NOperand, mask: u32) -> u32 {
    if !is_register(o.basereg) {
        return 0;
    }
    rexflags(regval(o), regflag(o), mask)
}

/// Evaluates one operand, computing opflags the way nasm's parser does.
fn eval_operand(o: &Operand, ctx: &EvalCtx) -> Result<NOperand, String> {
    let unknown = Cell::new(false);
    let mut sub = EvalCtx {
        cur_sect: ctx.cur_sect,
        cur_off: ctx.cur_off,
        syms: ctx.syms,
        optimistic: ctx.optimistic,
        unknown_hit: Some(&unknown),
    };
    let eval_expr = |e: &Expr, ctx: &mut EvalCtx| -> Result<Val, String> {
        unknown.set(false);
        let rv = e.eval(ctx)?;
        let mut v = Val::from_rvalue(&rv);
        v.unknown = unknown.get();
        Ok(v)
    };

    // Memory operands carry the GEN_OPTYPE(3) bit (op::MEMORY = 8|REGMEM);
    // registers are REGMEM|REGISTER without it.
    if o.imm.is_none() && o.disp.is_none() {
        // Register operand: no offset value involved.
        return Ok(NOperand {
            ty: o.ty,
            basereg: o.basereg,
            indexreg: o.indexreg,
            scale: o.scale,
            eaflags: o.eaflags,
            off: Val::default(),
        });
    }
    let is_mem = o.disp.is_some() && o.ty & (op::MEMORY & !op::REGMEM) != 0;
    if is_mem {
        let d = o.disp.as_ref().ok_or("memory operand without displacement")?;
        let off = eval_expr(d, &mut sub)?;
        let mut ty = o.ty;
        if off.known {
            ty |= op::IMM_KNOWN;
        }
        Ok(NOperand {
            ty,
            basereg: o.basereg,
            indexreg: o.indexreg,
            scale: o.scale,
            eaflags: o.eaflags,
            off,
        })
    } else {
        let imm = o.imm.as_ref().ok_or("expected immediate operand")?;
        let v = eval_expr(imm, &mut sub)?;
        let mut ty = o.ty | op::IMM_NORMAL;
        if v.known {
            // Immediate subclass flags, as nasm's parser computes them.
            ty |= op::IMM_KNOWN;
            if v.value == 1 {
                ty |= op::UNITY;
            }
            if (-128..=127).contains(&v.value) {
                ty |= op::SBYTEWORD | op::SBYTEDWORD;
            }
            if (-0x8000_0000..=0x7fff_ffff).contains(&v.value) {
                ty |= op::SDWORD;
            }
            if (0..=0xffff_ffff).contains(&v.value) {
                ty |= op::UDWORD;
            }
        }
        Ok(NOperand {
            ty,
            basereg: o.basereg,
            indexreg: o.indexreg,
            scale: o.scale,
            eaflags: o.eaflags,
            off: v,
        })
    }
}

// ---------------------------------------------------------------------
// Effective address computation (port of process_ea)
// ---------------------------------------------------------------------

fn process_ea(
    input: &mut NOperand,
    rfield: i32,
    rflags: u64,
    ins: &mut InsState,
) -> Result<(), String> {
    ins.ea = Ea::default();
    let ea = &mut ins.ea;
    ea.rex |= rexflags(rfield, rflags, REX_rR);

    if input.ty & op::REGISTER != 0 {
        // Direct register operand.
        if !is_register(input.basereg) {
            return Err("invalid register operand".into());
        }
        if regflag(input) & op::REG_EA != op::REG_EA {
            return Err("invalid register in r/m position".into());
        }
        ea.rex |= op_rexflags(input, REX_rB);
        ea.sib_present = false;
        ea.bytes = 0;
        ea.modrm = (3 << 6) | ((rfield as u8 & 7) << 3) | (regval(input) as u8 & 7);
        return Ok(());
    }

    if input.basereg == -1 && (input.indexreg == -1 || input.scale == 0) {
        // Pure offset: RIP-relative or absolute.
        if ins.bits == 64
            && input.ty & op::IMMEDIATE != 0
            && input.eaflags & eaf::ABS == 0
            && input.eaflags & eaf::REL != 0
        {
            input.ty |= op::IP_REL;
        }
        if input.eaflags & eaf::SIB != 0 || (ins.bits == 64 && input.ty & op::IP_REL == 0) {
            // SIB-form absolute address (modrm rm=4, sib base=101).
            ea.sib_present = true;
            ea.sib = (0 << 6) | (4 << 3) | 5;
            ea.bytes = 4;
            ea.modrm = (0 << 6) | ((rfield as u8 & 7) << 3) | 4;
            ea.rip = false;
        } else {
            ea.sib_present = false;
            ea.bytes = 4;
            ea.modrm = (0 << 6) | ((rfield as u8 & 7) << 3) | 5;
            ea.rip = ins.bits == 64;
        }
        return Ok(());
    }

    // Indirection.
    let mut i = input.indexreg;
    let s = input.scale;
    if s == 0 {
        i = -1;
    }
    let known = !input.off.unknown && input.off.segment.is_none();
    let o = input.off.value;

    let (mut it, mut ix) = if is_register(i) {
        (insns_gen::REGS[i as usize].val, insns_gen::REGS[i as usize].flags)
    } else {
        (-1, 0u64)
    };
    let (mut bt, mut bx) = if is_register(input.basereg) {
        (
            insns_gen::REGS[input.basereg as usize].val,
            insns_gen::REGS[input.basereg as usize].flags,
        )
    } else {
        (-1, 0u64)
    };

    // Long mode: only 64-bit registers may take part.
    if (it != -1 && ix & op::BITS64 == 0) || (bt != -1 && bx & op::BITS64 == 0) {
        return Err("invalid effective address: 64-bit addressing only".into());
    }

    if bt == -1 && s == 1 && it != -1 {
        // Make single reg base.
        bt = it;
        bx = ix;
        it = -1;
        ix = 0;
    }
    if it != -1 && (it & 7) == 4 {
        if s == 1 {
            // Swap ESP into base if scale is 1.
            std::mem::swap(&mut it, &mut bt);
            std::mem::swap(&mut ix, &mut bx);
        } else {
            return Err("invalid effective address: ESP cannot be scaled index".into());
        }
    }
    if it != -1 && !matches!(s, 1 | 2 | 4 | 8) {
        return Err("invalid effective address".into());
    }

    if it == -1 && bt != -1 && (bt & 7) != 4 {
        // No SIB needed.
        let rm = bt & 7;
        let mod_ = memory_mod(input.eaflags, o, known, rm != 5, ea);
        ea.sib_present = false;
        ea.bytes = if mod_ == 2 { 4 } else { mod_ as u32 };
        ea.modrm = (mod_ << 6) | ((rfield as u8 & 7) << 3) | rm as u8;
    } else {
        let (scale, index) = if it == -1 {
            (0u8, 4u8)
        } else {
            let sc = match s {
                1 => 0u8,
                2 => 1,
                4 => 2,
                8 => 3,
                _ => unreachable!(),
            };
            (sc, (it & 7) as u8)
        };
        let (base, mod_) = if bt == -1 {
            (5u8, 0u8)
        } else {
            let base = (bt & 7) as u8;
            let mod_ = memory_mod(input.eaflags, o, known, base != 5, ea);
            (base, mod_)
        };
        ea.sib_present = true;
        ea.bytes = if bt == -1 || mod_ == 2 { 4 } else { mod_ as u32 };
        ea.modrm = (mod_ << 6) | ((rfield as u8 & 7) << 3) | 4;
        ea.sib = (scale << 6) | (index << 3) | base;
    }
    ea.rex |= rexflags(it, ix, REX_rX);
    ea.rex |= rexflags(bt, bx, REX_rB);
    Ok(())
}

/// Selects the ModRM mod for a memory operand with displacement and
/// records the disp8 bookkeeping (nasm's `memory_mod`).
fn memory_mod(eaflags: u32, o: i64, known: bool, zerook: bool, ea: &mut Ea) -> u8 {
    ea.disp8 = o as i8;
    ea.disp8_ok = known && (ea.disp8 as i64) == o;
    if eaflags & eaf::WORDOFFS != 0 {
        return 2;
    }
    if eaflags & eaf::BYTEOFFS != 0 {
        return 1;
    }
    if !known {
        return 2;
    }
    if o == 0 && zerook {
        return 0;
    }
    if ea.disp8_ok {
        1
    } else {
        2
    }
}

// ---------------------------------------------------------------------
// calcsize / gencode: the bytecode interpreter
// ---------------------------------------------------------------------

struct Encoder {
    bytes: Vec<u8>,
    fixups: Vec<FixupReq>,
    /// Total instruction length from calcsize; needed for rel8 checks.
    inslen: i64,
    rex_done: bool,
}

impl Encoder {
    fn raw(&mut self, b: u8) {
        self.bytes.push(b);
    }
    fn field(&mut self, val: &Val, size: u8, pcrel: bool) {
        let tail = (self.inslen - self.bytes.len() as i64 - size as i64).max(0) as u32;
        self.fixups.push(FixupReq {
            at: self.bytes.len(),
            size,
            val: val.to_rvalue(),
            pcrel,
            tail,
        });
        self.bytes.extend(std::iter::repeat_n(0, size as usize));
    }
    fn bytes_field(&mut self, v: i64, size: u8) {
        for i in 0..size as u16 {
            self.bytes.push((v >> (i * 8)) as u8);
        }
    }
}

fn out_imm(enc: &mut Encoder, o: &NOperand, size: u8) {
    if o.off.known && o.off.segment.is_none() && o.off.extern_sym.is_none() {
        enc.bytes_field(o.off.value, size);
    } else {
        enc.field(&o.off, size, false);
    }
}

/// Relative field: the fixup machinery patches same-section references
/// statically and emits PC-relative relocations otherwise.
fn out_rel(enc: &mut Encoder, o: &NOperand, size: u8) {
    enc.field(&o.off, size, true);
}

fn emit_rex(enc: &mut Encoder, ins: &InsState) {
    if enc.rex_done {
        return;
    }
    enc.rex_done = true;
    if ins.rex & (REX_V | REX_EV) != 0 {
        return; // Handled elsewhere (unsupported here).
    }
    let rex = ins.rex & (REX_MASK | REX_L);
    if rex == (REX_L | REX_R) {
        enc.raw(0xf0);
    } else if rex & REX_P != 0 {
        enc.raw(rex as u8);
    } else if rex != 0 {
        return; // Not reachable for supported encodings.
    }
    if ins.vex_cm != 0 {
        enc.raw(0x0f);
        if ins.vex_cm > 1 {
            // Map 2 = 0F 38, map 3 = 0F 3A.
            enc.raw(0x34 + (ins.vex_cm << 1));
        }
    }
}

/// One interpreter pass; `enc == None` sizes (calcsize), `Some` emits
/// (gencode). Returns the instruction length. Bytecode values are the
/// octal codes from `x86/bytecode.txt`.
fn walk(
    ins: &mut InsState,
    temp: &Template,
    mut enc: Option<&mut Encoder>,
) -> Result<i64, String> {
    let codes = &insns_gen::BYTECODES[temp.code_off as usize..];
    let mut length: i64 = 0;
    let mut pos = 0usize;
    let mut opex = 0u8;

    while pos < codes.len() && codes[pos] != 0 {
        let c = codes[pos];
        pos += 1;
        let op1 = ((c & 3) as usize) + (((opex & 1) as usize) << 2);
        let op2 = (((c >> 3) & 3) as usize) + (((opex & 2) as usize) << 1);
        opex = 0;

        macro_rules! opof {
            ($n:expr) => {
                match ins.ops.get($n) {
                    Some(o) => o,
                    None => return Err(format!("missing operand {} for encoding", $n)),
                }
            };
        }

        match c {
            // \1..\4: literal bytes.
            1..=4 => {
                let n = c as usize;
                if let Some(e) = &mut enc {
                    emit_rex(e, ins);
                    for k in 0..n {
                        e.raw(codes[pos + k]);
                    }
                }
                pos += n;
                length += n as i64;
            }
            // \5..\7: operand-number escape.
            5..=7 => {
                opex = c;
            }
            // \10..\13: byte + register value.
            8..=11 => {
                let o = opof!(op1);
                ins.rex |= op_rexflags(o, REX_rB);
                if let Some(e) = &mut enc {
                    emit_rex(e, ins);
                    let rv = regval(o) as u8;
                    e.raw(codes[pos].wrapping_add(rv & 7));
                }
                pos += 1;
                length += 1;
            }
            // \14..\17: MIB index register.
            12..=15 => {
                if itemp_has(temp, if_::MIB) {
                    return Err("MIB encodings are not supported".into());
                }
            }
            // \20..\23: ib; \24..\27: ib,u.
            16..=23 => {
                length += 1;
                if let Some(e) = &mut enc {
                    let o = opof!(op1);
                    out_imm(e, o, 1);
                }
            }
            // \30..\33: iw.
            24..=27 => {
                length += 2;
                if let Some(e) = &mut enc {
                    let o = opof!(op1);
                    out_imm(e, o, 2);
                }
            }
            // \34..\37: iw/id by operand size.
            28..=31 => {
                let o = opof!(op1);
                let size = if o.ty & (op::BITS16 | op::BITS32 | op::BITS64) != 0 {
                    if o.ty & op::BITS16 != 0 {
                        2
                    } else {
                        4
                    }
                } else if ins.op_size == 16 {
                    2
                } else {
                    4
                };
                length += size as i64;
                if let Some(e) = &mut enc {
                    out_imm(e, o, size);
                }
            }
            // \40..\43: id.
            32..=35 => {
                length += 4;
                if let Some(e) = &mut enc {
                    let o = opof!(op1);
                    out_imm(e, o, 4);
                }
            }
            // \44..\47: address-size immediate.
            36..=39 => {
                let size = ins.addr_size >> 3;
                length += size as i64;
                if let Some(e) = &mut enc {
                    let o = opof!(op1);
                    out_imm(e, o, size);
                }
            }
            // \50..\53: rel8.
            40..=43 => {
                length += 1;
                if let Some(e) = &mut enc {
                    let o = opof!(op1);
                    if o.off.known && o.off.segment == Some(ins.loc_sect) {
                        let delta = o.off.value - (ins.loc_off + e.inslen);
                        if !(-128..=127).contains(&delta) {
                            return Err("short jump is out of range".into());
                        }
                    }
                    out_rel(e, o, 1);
                }
            }
            // \54..\57: iq.
            44..=47 => {
                length += 8;
                if let Some(e) = &mut enc {
                    let o = opof!(op1);
                    out_imm(e, o, 8);
                }
            }
            // \60..\63: rel16.
            48..=51 => {
                length += 2;
                if let Some(e) = &mut enc {
                    let o = opof!(op1);
                    out_rel(e, o, 2);
                }
            }
            // \64..\67: rel by operand size.
            52..=55 => {
                let o = opof!(op1);
                let size = if o.ty & (op::BITS16 | op::BITS32 | op::BITS64) != 0 {
                    if o.ty & op::BITS16 != 0 {
                        2
                    } else {
                        4
                    }
                } else if ins.op_size == 16 {
                    2
                } else {
                    4
                };
                length += size as i64;
                if let Some(e) = &mut enc {
                    out_rel(e, o, size);
                }
            }
            // \70..\73: rel32.
            56..=59 => {
                length += 4;
                if let Some(e) = &mut enc {
                    let o = opof!(op1);
                    out_rel(e, o, 4);
                }
            }
            // \74..\77: far segment:offset.
            60..=63 => {
                return Err("far calls/jumps are not supported".into());
            }
            // \171: /r with reg field from the operand in the next byte.
            0o171 => {
                let cb = codes[pos];
                pos += 1;
                let op2 = (op2 & !3) | (((cb >> 3) & 3) as usize);
                let o = opof!(op2);
                ins.rex |= op_rexflags(o, REX_rR);
                length += 1;
                if let Some(e) = &mut enc {
                    emit_rex(e, ins);
                    let r = regval(o) as u8;
                    e.raw((cb & !0o70) | ((r & 7) << 3));
                }
            }
            // \172, \173: /is4 with immediate operand.
            0o172 | 0o173 => {
                pos += 1;
                length += 1;
            }
            // \174..\177: /is4 with register in bits 7..4.
            0o174..=0o177 => {
                length += 1;
                if let Some(e) = &mut enc {
                    let o = opof!(op1);
                    let r = regval(o) as u8;
                    e.raw((r << 4) | ((r & 0x10) >> 1));
                }
            }
            // \240..\243, \250: EVEX.
            0xa0..=0xa3 | 0xa8 => {
                return Err("EVEX-encoded instructions are not supported".into());
            }
            // \254..\257: id,s; \264..\267: id,u.
            0xac..=0xaf | 0xb4..=0xb7 => {
                length += 4;
                if let Some(e) = &mut enc {
                    let o = opof!(op1);
                    out_imm(e, o, 4);
                }
            }
            // \260..\263, \270: VEX.
            0xb0..=0xb3 | 0xb8 => {
                return Err("VEX-encoded instructions are not supported".into());
            }
            // \271..\273: HLE markers.
            0xb9..=0xbb => {}
            // \274..\277: ib,s.
            0xbc..=0xbf => {
                length += 1;
                if let Some(e) = &mut enc {
                    let o = opof!(op1);
                    if o.off.known {
                        e.bytes_field(o.off.value, 1);
                    } else {
                        out_imm(e, o, 1);
                    }
                }
            }
            // \300..\303: hint-NOP byte.
            0xc0..=0xc3 => {
                length += 1;
                if let Some(e) = &mut enc {
                    let o = opof!(op1);
                    emit_rex(e, ins);
                    if o.off.known {
                        e.bytes_field(o.off.value, 1);
                    } else {
                        out_imm(e, o, 1);
                    }
                }
            }
            // \304..\307: ib XOR modifier.
            0xc4..=0xc7 => {
                length += 1;
                let modifier = codes[pos + 1];
                pos += 2;
                if let Some(e) = &mut enc {
                    let o = opof!(op1);
                    emit_rex(e, ins);
                    if o.off.known {
                        e.raw((o.off.value as u8) ^ modifier);
                    } else {
                        return Err("immediate must be constant here".into());
                    }
                }
            }
            // \310: a16 — invalid in 64-bit mode.
            0xc8 => {
                return Err("16-bit addressing is not supported".into());
            }
            // \311: a32.
            0xc9 => {
                ins.addr_size = 32;
            }
            // \312: disassembler-only.
            0xca => {}
            // \313: a64.
            0xcb => {
                ins.addr_size = 64;
            }
            // \314..\317: disassembler-only.
            0xcc..=0xcf => {}
            // \320: o16; \321: o32.
            0xd0 => {
                ins.op_size = 16;
                if ins.prefixes[PPS_OSIZE] != P_OSP {
                    ins.prefixes[PPS_OSIZE] = P_O16;
                }
            }
            0xd1 => {
                if ins.rex & REX_NW != 0 {
                    return Err("instruction not valid with 32-bit operand size".into());
                }
                ins.op_size = 32;
                if ins.prefixes[PPS_OSIZE] != P_OSP {
                    ins.prefixes[PPS_OSIZE] = P_O32;
                }
            }
            // \322: odf — no-op.
            0xd2 => {}
            // \323: o64nw (nw + o64); \324: o64.
            0xd3 | 0xd4 => {
                if c == 0xd3 {
                    ins.rex |= REX_NW;
                }
                ins.op_size = 64;
                if ins.prefixes[PPS_OSIZE] == P_OSP {
                    if ins.rex & REX_NW == 0 {
                        ins.rex |= REX_W;
                    }
                } else {
                    ins.prefixes[PPS_OSIZE] = P_O64;
                }
            }
            // \325: nohi.
            0xd5 => {
                ins.rex |= REX_NH;
            }
            // \326: nof3 — no-op.
            0xd6 => {}
            // \327: nw.
            0xd7 => {
                if ins.bits == 64 {
                    ins.rex |= REX_NW;
                    if ins.op_size == 32 {
                        ins.op_size = 64;
                    }
                }
            }
            // \330: osz — default or user-specified operand size.
            0xd8 => match ins.prefixes[PPS_OSIZE] {
                P_NONE => match ins.op_size {
                    16 => ins.prefixes[PPS_OSIZE] = P_O16,
                    32 => ins.prefixes[PPS_OSIZE] = P_O32,
                    64 => ins.prefixes[PPS_OSIZE] = P_O64,
                    _ => unreachable!(),
                },
                P_OSP => {
                    ins.prefixes[PPS_OSIZE] = if ins.bits == 16 { P_O32 } else { P_O16 };
                }
                _ => {}
            },
            // \331..\333: norep / f2 / f3.
            0xd9..=0xdb => {
                ins.need_pfx[PPS_REP] = match c {
                    0xd9 => PFE_NULL,
                    0xda => 0xf2,
                    _ => 0xf3,
                };
            }
            // \334: rex.l (ignored in 64-bit mode); \335-\337: no-ops.
            0xdc | 0xdd | 0xde | 0xdf => {
                if c == 0xdc {
                    ins.rex |= REX_L;
                }
            }
            // \340: resb — handled by directives; \343: osd (unused).
            0xe0 | 0xe3 => {
                return Err("RESx pseudo-instructions are handled by directives".into());
            }
            // \341: wait.
            0xe1 => {
                if ins.prefixes[PPS_WAIT] == P_NONE {
                    ins.prefixes[PPS_WAIT] = P_WAIT;
                }
            }
            // \342: osm — o-size matching the assembly mode (64 here).
            0xe2 => {
                ins.op_size = 64;
                if ins.prefixes[PPS_OSIZE] != P_OSP {
                    ins.prefixes[PPS_OSIZE] = P_O64;
                }
            }
            // \344..\347: REX[2].B/X/R/W as opcode extension.
            0xe4..=0xe7 => {
                let bit = match c & 3 {
                    0 => REX_B,
                    1 => REX_X,
                    2 => REX_R,
                    _ => REX_W,
                };
                ins.rex |= REX_P | bit;
            }
            // \350..\351: REX2.
            0xe8 | 0xe9 => {
                return Err("REX2-encoded instructions are not supported".into());
            }
            // \355..\357: legacy map number.
            0xed..=0xef => {
                ins.vex_cm = c & 3;
            }
            // \360: np; \361: 66 + np.
            0xf0 => {
                ins.need_pfx[PPS_REP] = PFE_NULL;
                ins.need_pfx[PPS_OSIZE] = PFE_NULL;
            }
            0xf1 => {
                ins.need_pfx[PPS_REP] = PFE_NULL;
                ins.need_pfx[PPS_OSIZE] = 0x66;
            }
            // \364/\365: !osp / !asp.
            0xf4 => {
                ins.need_pfx[PPS_OSIZE] = PFE_NULL;
            }
            0xf5 => {
                ins.need_pfx[PPS_ASIZE] = PFE_NULL;
            }
            // \366: osp (66 as opcode extension); \367: asp (67).
            0xf6 => {
                ins.need_pfx[PPS_OSIZE] = 0x66;
            }
            0xf7 => {
                ins.need_pfx[PPS_ASIZE] = 0x67;
            }
            // \370/\371: jcc8 / jmp8 markers.
            0xf8 | 0xf9 => {
                if c == 0xf9 && ins.prefixes[PPS_REP] == P_BND {
                    ins.prefixes[PPS_REP] = P_NONE;
                }
            }
            // \373: jlen.
            0xfb => {
                length += 1;
                if let Some(e) = &mut enc {
                    e.raw(if ins.bits == 16 { 3 } else { 5 });
                }
            }
            // \374..\376: VSIB.
            0xfc..=0xfe => {
                return Err("VSIB memory operands are not supported".into());
            }
            // \1ab (0100..0177) and \2ab (0200..0237): ModRM forms.
            c if (0o100..=0o177).contains(&c) || (0o200..=0o237).contains(&c) => {
                let mut o = match ins.ops.get(op2) {
                    Some(o) => o.clone(),
                    None => return Err(format!("missing operand {} for EA", op2)),
                };
                let (rfield, rflags) = if c <= 0o177 {
                    let opx = opof!(op1);
                    (regval(opx), regflag(opx))
                } else {
                    ((c & 7) as i32, 0)
                };
                if itemp_has(temp, if_::MIB) {
                    return Err("MIB encodings are not supported".into());
                }
                if itemp_has(temp, if_::SIB) {
                    o.eaflags |= eaf::SIB;
                }
                process_ea(&mut o, rfield, rflags, ins)?;
                ins.rex |= ins.ea.rex;
                length += 1
                    + i64::from(ins.ea.sib_present)
                    + ins.ea.bytes as i64;
                if let Some(e) = &mut enc {
                    emit_rex(e, ins);
                    e.raw(ins.ea.modrm);
                    if ins.ea.sib_present {
                        e.raw(ins.ea.sib);
                    }
                    let opy = opof!(op2);
                    match ins.ea.bytes {
                        0 => {}
                        1 => {
                            e.raw(ins.ea.disp8 as u8);
                        }
                        n => {
                            if ins.ea.rip {
                                out_rel(e, opy, n as u8);
                            } else {
                                out_imm(e, opy, n as u8);
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(format!(
                    "unsupported bytecode \\{:o} in instruction encoding",
                    c
                ));
            }
        }
    }

    if ins.rex & REX_NH != 0 {
        if ins.rex & REX_H != 0 {
            return Err("instruction cannot use high registers".into());
        }
        ins.rex &= !REX_P; // Don't force REX due to the high-reg flag
    }

    if ins.prefixes[PPS_OSIZE] == P_O64 && ins.rex & REX_NW == 0 {
        ins.rex |= REX_W;
    }

    // A REX prefix is required whenever any REX bit is set.
    if ins.rex & REX_MASK != 0 {
        if ins.rex & REX_H != 0 {
            return Err("cannot use high byte register in this instruction".into());
        }
        ins.rex &= !REX_L;
        ins.rex |= REX_P;
        length += 1;
    }

    Ok(length)
}

fn emit_prefixes(bytes: &mut Vec<u8>, ins: &InsState) {
    let mut buf = [0u8; 9];
    let mut n = 0;
    for j in 0..9 {
        let pfx = ins.prefixes[j];
        let r = ins.need_pfx[j];
        let mut c = prefix_byte(pfx, ins.bits);
        if r != 0 && r != c && !((r <= 0 || r == PFE_NULL) && c <= 0) {
            match pfx {
                P_NONE | P_O16 | P_O32 | P_O64 | P_A16 | P_A32 | P_A64 => c = r,
                _ => {}
            }
        }
        if c >= 0 {
            buf[n] = c as u8;
            n += 1;
        }
    }
    bytes.extend_from_slice(&buf[..n]);
}

// ---------------------------------------------------------------------
// Matching (port of matches/jmp_match/find_match)
// ---------------------------------------------------------------------

impl InsState {
    fn clone_state(&self) -> InsState {
        InsState {
            bits: self.bits,
            op_size: self.op_size,
            addr_size: self.addr_size,
            rex: 0,
            prefixes: self.prefixes,
            need_pfx: [0; 9],
            ea: Ea::default(),
            ops: self.ops.clone(),
            loc_sect: self.loc_sect,
            loc_off: self.loc_off,
            vex_cm: 0,
        }
    }
}

fn calcsize_speculative(ins: &InsState, temp: &Template) -> Option<i64> {
    let mut tmp = ins.clone_state();
    walk(&mut tmp, temp, None).ok()
}

fn jmp_match(ins: &InsState, temp: &Template) -> i32 {
    let op0 = &ins.ops[0];
    if op0.ty & op::STRICT != 0 {
        return MERR_INVALOP;
    }
    if op0.off.unknown {
        // Be optimistic in pass 1.
        return MOK_GOOD;
    }
    if op0.off.segment != Some(ins.loc_sect) {
        // Cross-segment jump.
        return MERR_INVALOP;
    }
    let mut delta = op0.off.value - ins.loc_off;
    if !(-128 + 2..=127 + 15).contains(&delta) {
        return MERR_INVALOP;
    } else if !(-128 + 15..=127 + 2).contains(&delta) {
        // Borderline: calculate the exact instruction size.
        let Some(isize) = calcsize_speculative(ins, temp) else {
            return MERR_INVALOP;
        };
        delta -= isize;
        if delta != delta as i8 as i64 {
            return MERR_INVALOP;
        }
    }
    MOK_GOOD
}

fn matches(temp: &Template, ins: &InsState) -> i32 {
    let oprs = ins.ops.len();
    if temp.operands as usize != oprs {
        return MERR_INVALOP;
    }
    // Decorated templates are marked at table-generation time.
    if temp.deco.iter().any(|&d| d != 0) {
        return MERR_INVALOP;
    }

    let jsize_mask = op::NEAR | op::FAR | op::SHORT | op::ABS;
    let msize_mask = op::SIZE_MASK & !jsize_mask;

    let mut isize_ = [0u64; 5];
    let mut itype = [0u64; 5];
    for i in 0..oprs {
        let o = &ins.ops[i];
        let ttype = temp.opd[i];
        isize_[i] = o.ty & msize_mask;
        itype[i] = o.ty - isize_[i];
        if o.ty & !ttype & (op::COLON | op::TO) != 0 {
            return MERR_INVALOP;
        }
    }

    let mut op_size = ins.op_size;
    if itemp_has(temp, if_::NWSIZE) && op_size == 32 {
        op_size = 64;
    }
    let opsize = mode_to_op(op_size);

    let if_anysize = itemp_has(temp, if_::ANYSIZE);
    let if_sx = itemp_has(temp, if_::SX);

    for i in 0..oprs {
        let ttype = temp.opd[i];

        if ttype & (op::NEAR | op::SHORT) != 0 {
            if isize_[i] == op::BITS8 {
                itype[i] |= op::SHORT;
                isize_[i] = 0;
            }
            if itype[i] & op::SHORT != 0 {
                itype[i] &= !op::NEAR;
            }
            if itype[i] & (op::FAR | op::SHORT) == 0 {
                itype[i] |= ttype & op::NEAR;
            }
            if ttype & (op::NEAR | op::SHORT) == (op::NEAR | op::SHORT)
                && itype[i] & (op::FAR | op::ABS) == 0
            {
                itype[i] |= op::NEAR | op::SHORT;
            }
        }

        // Coerce explicitly sized immediates into extending patterns.
        if itype[i] & ttype & op::IMMEDIATE != 0 {
            if isize_[i] == op::BITS8 && ttype & op::BYTEEXTMASK != 0 {
                isize_[i] = ttype & msize_mask;
                itype[i] |= op::BYTEEXTMASK;
            } else if isize_[i] == op::BITS32 && ttype & op::DWORDEXTMASK != 0 {
                isize_[i] = ttype & msize_mask;
            }
            if !itemp_has(temp, if_::SDWORD) {
                itype[i] |= op::SDWORD;
            }
        }

        if ttype & !itype[i] & !(msize_mask | op::REGSET_MASK) != 0 {
            return MERR_INVALOP;
        }
        if !ttype & itype[i] & op::REGSET_MASK != 0 {
            return if ttype & op::REGSET_MASK != 0 {
                MERR_REGSETSIZE
            } else {
                MERR_REGSET
            };
        }
    }

    let arflag = itemp_smask(temp);
    let mut arsize: u64 = ((arflag & IF_TSMASK) as u64) << (32 - 10);
    if arflag & IFM_OSIZE != 0 {
        arsize = opsize;
    } else if arflag & IFM_ASIZE != 0 {
        arsize = mode_to_op(ins.addr_size);
    }
    let armask = itemp_arx(temp);
    let smmask = itemp_smx(temp);

    if !if_sx {
        for i in 0..oprs {
            let bit = 1usize << i;
            if isize_[i] == 0 {
                if itype[i] & op::REGISTER != 0 && (temp.opd[i] & msize_mask) != 0 {
                    isize_[i] = temp.opd[i] & msize_mask;
                } else if armask & bit as u32 != 0 {
                    isize_[i] = arsize;
                }
            }
        }
    }

    let mut smsize = msize_mask;
    let mut sm_missing = false;
    if smmask != 0 {
        let mut nosizemask = 0u32;
        for i in 0..oprs {
            let bit = 1u32 << i;
            if isize_[i] != 0 {
                if smmask & bit != 0 {
                    smsize &= isize_[i];
                }
            } else if temp.opd[i] & msize_mask != 0 && itype[i] & op::REGISTER == 0 {
                nosizemask |= bit;
            }
        }
        sm_missing = smmask & !nosizemask == 0;
    }

    for i in 0..oprs {
        let tsize = temp.opd[i] & msize_mask;
        let type_ = itype[i];
        let has_ar = (armask >> i) & 1 != 0;
        let has_sm = (smmask >> i) & 1 != 0;
        let is_reg = type_ & op::REGISTER != 0;

        if tsize != 0 {
            if isize_[i] != tsize {
                if has_ar {
                    if if_anysize {
                        // accepted regardless
                    } else if if_sx && !is_reg {
                        return MERR_OPSIZEINVAL;
                    }
                }
                if isize_[i] != 0 {
                    return if is_reg { MERR_INVALOP } else { MERR_OPSIZEINVAL };
                }
            }
            if has_sm {
                smsize &= tsize;
            }
        } else if if_sx && has_ar && isize_[i] != 0 {
            return MERR_OPSIZEINVAL;
        }

        if op::REG_HIGH & !type_ == 0 && is_register(ins.ops[i].basereg) {
            // High registers cannot combine with REX, which is always
            // available in 64-bit mode.
            return MERR_INVALOP;
        }
    }

    if smmask != 0 && (smsize == 0 || !smsize.is_power_of_two() || (if_sx && sm_missing)) {
        return MERR_OPSIZEMISSING;
    }

    // CPU level check (baseline x86-64).
    if temp.f[6] & CPU_LEVEL_MASK > CPU_LEVEL_X86_64 {
        return MERR_BADCPU;
    }

    // Long-mode flags.
    if itemp_has(temp, if_::NOLONG) {
        return MERR_BADMODE;
    }

    if itemp_has(temp, if_::JMP_RELAX) {
        return jmp_match(ins, temp);
    }

    MOK_GOOD
}

fn find_match(ins: &InsState, opcode: usize) -> Option<&'static Template> {
    let list: &'static [Template] = insns_gen::INSTRUCTIONS[opcode];
    let mut best: Option<(usize, i32)> = None;
    for (i, temp) in list.iter().enumerate() {
        let m = matches(temp, ins);
        if m > best.map_or(-1, |(_, b)| b) {
            best = Some((i, m));
            if m == MOK_GOOD {
                break;
            }
        }
    }
    let (i, m) = best?;
    if m >= MOK_FIRST {
        Some(&list[i])
    } else {
        None
    }
}

// ---------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------

/// Encode one instruction via the nasm tables.
pub fn encode(insn: &Insn, ctx: &EvalCtx) -> Result<Encoded, String> {
    let mut ops = Vec::with_capacity(insn.operands.len());
    for o in &insn.operands {
        ops.push(eval_operand(o, ctx)?);
    }
    let mut prefixes = [P_NONE; 9];
    for p in &insn.prefixes {
        match p.as_str() {
            "lock" => prefixes[PPS_LOCK] = P_LOCK,
            "rep" => prefixes[PPS_REP] = P_REP,
            "repe" | "repz" => prefixes[PPS_REP] = P_REPE,
            "repne" | "repnz" => prefixes[PPS_REP] = P_REPNE,
            "bnd" => prefixes[PPS_REP] = P_BND,
            "xacquire" => prefixes[PPS_REP] = P_XACQUIRE,
            "xrelease" => prefixes[PPS_REP] = P_XRELEASE,
            _ => return Err(format!("prefix `{p}' is not supported")),
        }
    }
    let ins = InsState {
        bits: 64,
        op_size: 32,
        addr_size: 64,
        rex: 0,
        prefixes,
        need_pfx: [0; 9],
        ea: Ea::default(),
        ops,
        loc_sect: ctx.cur_sect,
        loc_off: ctx.cur_off,
        vex_cm: 0,
    };

    let Some(temp) = find_match(&ins, insn.opcode) else {
        return Err(format!(
            "no valid encoding for `{}` with these operands",
            insns_gen::MNEMONICS[insn.opcode]
        ));
    };

    // Size the instruction first (computes rex, prefixes, EA).
    let mut sized = ins.clone_state();
    let inslen = walk(&mut sized, temp, None)?;

    // Emit.
    let mut enc = Encoder { bytes: Vec::new(), fixups: Vec::new(), inslen, rex_done: false };
    emit_prefixes(&mut enc.bytes, &sized);
    walk(&mut sized, temp, Some(&mut enc))?;

    Ok(Encoded { bytes: enc.bytes, fixups: enc.fixups })
}

