//! AArch64 backend: fixed 32-bit instructions, core integer set.

use crate::arch::Backend;
use crate::assemble::{Fixup, Reloc};
use crate::expr::EvalCtx;
use object::{Architecture, Endianness};

/// One AArch64 instruction.
#[derive(Clone, Debug)]
pub struct Insn {
    pub mnemonic: String,
    pub operands: Vec<Operand>,
    pub line: u32,
}

/// AArch64 operands: x/w registers, sp/xzr, immediates, and the
/// load/store addressing modes.
#[derive(Clone, Debug)]
pub enum Operand {
    /// General-purpose register; `wide` selects x (64-bit) vs w (32-bit).
    Reg {
        num: u8,
        wide: bool,
    },
    /// sp / wsp (encodes as register 31).
    Sp {
        wide: bool,
    },
    /// xzr / wzr (encodes as register 31).
    Zr {
        wide: bool,
    },
    Imm(crate::expr::Expr),
}

pub fn parse_insn(_g: &[crate::gas::GTok], _line_no: u32, _lineno: usize, _numlabels: &crate::gas::NumLabels) -> Result<crate::insn::Stmt, String> {
    Err("aarch64: parsing not yet implemented".into())
}

pub struct A64Backend;

impl Backend for A64Backend {
    fn encode(&self, insn: &crate::arch::Insn, _ctx: &EvalCtx, _relax_state: u8) -> Result<crate::encoding::Encoded, String> {
        match insn {
            crate::arch::Insn::Aarch64(_) => Err("aarch64: encoding not yet implemented".into()),
            _ => Err("not an AArch64 instruction".into()),
        }
    }

    fn finalize_all(
        &self,
        _sections: &mut [crate::assemble::Section],
        _symbols: &mut Vec<crate::assemble::Symbol>,
        _fixups: Vec<Fixup>,
        _relocs: &mut Vec<Reloc>,
    ) -> Result<(), Vec<String>> {
        Err(vec!["aarch64: fixups not yet implemented".into()])
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

    fn text_fill(&self) -> u8 {
        0
    }
}
