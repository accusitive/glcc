//! Instruction-level AST shared by the syntax frontends.

use crate::expr::Expr;
use crate::regs::Reg;

/// Operand size in bytes, as requested by `byte`/`dword`/etc. prefixes.
pub type Size = u8;

#[derive(Clone, Debug)]
pub enum Operand {
    Imm(Expr),
    Reg(Reg),
    Mem {
        size: Option<Size>,
        base: Option<Reg>,
        index: Option<Reg>,
        scale: u8,
        disp: Expr,
    },
}

impl Operand {
    pub fn is_reg(&self) -> bool {
        matches!(self, Operand::Reg(_))
    }
}

#[derive(Clone, Debug)]
pub struct Insn {
    pub mnemonic: String,
    pub operands: Vec<Operand>,
    pub line: u32,
}

/// Data items in `db`/`dw`/... lists.
#[derive(Clone, Debug)]
pub enum DataItem {
    /// A string literal contributes its bytes verbatim.
    Str(Vec<u8>),
    /// An expression laid out in the directive's element size.
    Expr(Expr),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SectionFlags {
    Text,
    Data,
    Bss,
    Rodata,
    Custom { write: bool, exec: bool, nobits: bool, align: Option<u64> },
}

#[derive(Clone, Debug)]
pub struct SectionSpec {
    pub name: String,
    pub flags: SectionFlags,
    pub align: Option<u64>,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Label(String, u32),
    Data { size: Size, items: Vec<DataItem>, line: u32 },
    Reserve { size: Size, count: Expr, line: u32 },
    Align(Expr, u32),
    Insn(Insn),
    Global(Vec<String>, u32),
    Extern(Vec<String>, u32),
    Section(SectionSpec),
    /// Default operand size (`bits 16/32/64`); only 64 is supported.
    Bits(u8, u32),
    /// `times N <inner>`; N must eventually be constant.
    Times { count: Expr, inner: Box<Stmt>, line: u32 },
}

impl Stmt {
    pub fn line(&self) -> u32 {
        match self {
            Stmt::Label(_, l)
            | Stmt::Data { line: l, .. }
            | Stmt::Reserve { line: l, .. }
            | Stmt::Align(_, l)
            | Stmt::Global(_, l)
            | Stmt::Extern(_, l)
            | Stmt::Bits(_, l)
            | Stmt::Times { line: l, .. } => *l,
            Stmt::Insn(i) => i.line,
            Stmt::Section(_) => 0,
        }
    }
}
