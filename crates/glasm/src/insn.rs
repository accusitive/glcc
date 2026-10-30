//! Instruction-level AST shared by the syntax frontends, modeled on
//! nasm's `struct operand` (opflags-driven) so the generated instruction
//! tables match against it directly.

use crate::expr::Expr;

/// opflags_t constants (values from nasm's `include/opflags.h`).
pub mod op {
    pub const REGISTER: u64 = 1;
    pub const IMMEDIATE: u64 = 2;
    pub const REGMEM: u64 = 4;
    pub const MEMORY: u64 = 8 | REGMEM;

    pub const BITS8: u64 = 1 << 32;
    pub const BITS16: u64 = 1 << 33;
    pub const BITS32: u64 = 1 << 34;
    pub const BITS64: u64 = 1 << 35;
    pub const BITS128: u64 = 1 << 37;
    pub const BITS256: u64 = 1 << 38;
    pub const BITS512: u64 = 1 << 39;
    pub const SIZE_MASK: u64 = 0xffff << 32;

    pub const FAR: u64 = 1 << 40;
    pub const NEAR: u64 = 1 << 41;
    pub const SHORT: u64 = 1 << 42;
    pub const ABS: u64 = 1 << 43;

    pub const TO: u64 = 1 << 4;
    pub const COLON: u64 = 1 << 5;
    pub const STRICT: u64 = 1 << 6;

    pub const REG_CLASS_GPR: u64 = 1 << (7 + 1);
    pub const REG_CLASS_SREG: u64 = 1 << (7 + 2);
    pub const REG_CLASS_FPUREG: u64 = 1 << (7 + 3);
    pub const REG_CLASS_RM_MMX: u64 = 1 << (7 + 4);
    pub const REG_CLASS_RM_XMM: u64 = 1 << (7 + 5);
    pub const REG_CLASS_RM_YMM: u64 = 1 << (7 + 6);
    pub const REG_CLASS_RM_ZMM: u64 = 1 << (7 + 7);
    pub const REG_CLASS_OPMASK: u64 = 1 << (7 + 8);
    pub const REG_CLASS_BND: u64 = 1 << (7 + 9);
    pub const REG_CLASS_RM_TMM: u64 = 1 << (7 + 10);
    pub const REG_CLASS_VECTOR: u64 =
        REG_CLASS_RM_XMM | REG_CLASS_RM_YMM | REG_CLASS_RM_ZMM | REG_CLASS_RM_TMM;

    pub const RN_L16: u64 = 1 << (10 + 18);
    pub const RN_ZERO: u64 = (1 << (11 + 18)) | RN_L16;
    pub const RN_NZERO: u64 = (1 << (12 + 18)) | RN_L16;

    pub const REG_EA: u64 = REGMEM | REGISTER;
    pub const RM_GPR: u64 = REG_CLASS_GPR | REGMEM;
    pub const REG_GPR: u64 = REG_CLASS_GPR | REG_EA;
    pub const FPUREG: u64 = REG_CLASS_FPUREG | REGISTER;
    pub const RM_MMX: u64 = REG_CLASS_RM_MMX | REGMEM;
    pub const MMXREG: u64 = REG_CLASS_RM_MMX | REG_EA;
    pub const RM_XMM: u64 = REG_CLASS_RM_XMM | REGMEM;
    pub const XMMREG: u64 = REG_CLASS_RM_XMM | REG_EA;
    pub const RM_YMM: u64 = REG_CLASS_RM_YMM | REGMEM;
    pub const YMMREG: u64 = REG_CLASS_RM_YMM | REG_EA;
    pub const RM_ZMM: u64 = REG_CLASS_RM_ZMM | REGMEM;
    pub const ZMMREG: u64 = REG_CLASS_RM_ZMM | REG_EA;
    pub const RM_OPMASK: u64 = REG_CLASS_OPMASK | REGMEM;
    pub const OPMASKREG: u64 = REG_CLASS_OPMASK | REG_EA;

    pub const REG8: u64 = REG_GPR | BITS8;
    pub const REG_HIGH: u64 = REG8 | RN_NZERO | (1 << (18 + 3));

    pub const MEM_OFFS: u64 = (1 << (18 + 0)) | MEMORY;
    pub const IP_REL: u64 = (1 << (18 + 1)) | MEMORY;
    pub const RM_L16: u64 = RN_L16 | REGMEM;
    pub const RM_ZERO: u64 = RN_ZERO | REGMEM;
    pub const RM_NZERO: u64 = RN_NZERO | REGMEM;
    /// All r/m classes a memory operand can match (nasm MEMORY_ANY).
    pub const MEMORY_ANY: u64 =
        MEMORY | RM_GPR | RM_MMX | RM_XMM | RM_YMM | RM_ZMM | RM_L16 | RM_ZERO
            | RM_NZERO;
    /// RM_SEL: valid segment selector class (added to unsized r/m).
    pub const RM_SEL: u64 = (1 << (18 + 13)) | REGMEM;

    pub const IMM_NORMAL: u64 = (1 << (18 + 0)) | IMMEDIATE;
    pub const UNITY: u64 = (1 << (18 + 1)) | IMMEDIATE;
    pub const SBYTEWORD: u64 = (1 << (18 + 2)) | IMMEDIATE;
    pub const SBYTEDWORD: u64 = (1 << (18 + 3)) | IMMEDIATE;
    pub const SDWORD: u64 = (1 << (18 + 4)) | IMMEDIATE;
    pub const UDWORD: u64 = (1 << (18 + 5)) | IMMEDIATE;
    pub const FOURBITS: u64 = (1 << (18 + 6)) | IMMEDIATE;
    pub const IMM_KNOWN: u64 = (1 << (18 + 7)) | IMMEDIATE;
    pub const BYTEEXTMASK: u64 = SBYTEWORD | SBYTEDWORD;
    pub const DWORDEXTMASK: u64 = SDWORD | UDWORD;

    pub const REGSET_MASK: u64 = 0x1f << 48;
}

/// eaflags constants (nasm `enum ea_flags`).
pub mod eaf {
    pub const BYTEOFFS: u32 = 1;
    pub const WORDOFFS: u32 = 2;
    pub const TIMESTWO: u32 = 4;
    pub const REL: u32 = 8;
    pub const ABS: u32 = 16;
    pub const MIB: u32 = 32;
    pub const SIB: u32 = 64;
}

/// One instruction operand, the moral equivalent of nasm's
/// `struct operand`. Expressions are evaluated before matching.
#[derive(Clone, Debug)]
pub struct Operand {
    /// opflags_t: operand class/size/subclass.
    pub ty: u64,
    /// Index into `insns_gen::REGS`, or -1.
    pub basereg: i32,
    pub indexreg: i32,
    pub scale: u8,
    /// Segment override register index, or -1.
    pub seg: i32,
    /// Memory displacement expression.
    pub disp: Option<Expr>,
    /// Immediate expression.
    pub imm: Option<Expr>,
    pub eaflags: u32,
}

impl Operand {
    pub fn imm(expr: Expr) -> Operand {
        Operand {
            ty: op::IMMEDIATE,
            basereg: -1,
            indexreg: -1,
            scale: 0,
            seg: -1,
            disp: None,
            imm: Some(expr),
            eaflags: 0,
        }
    }

    pub fn reg(id: i32, flags: u64) -> Operand {
        Operand {
            ty: flags,
            basereg: id,
            indexreg: -1,
            scale: 0,
            seg: -1,
            disp: None,
            imm: None,
            eaflags: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Insn {
    /// Index into `insns_gen::MNEMONICS`.
    pub opcode: usize,
    /// Instruction prefixes (lock/rep/...), lowercased.
    pub prefixes: Vec<String>,
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
    Data { size: u8, items: Vec<DataItem>, line: u32 },
    Reserve { size: u8, count: Expr, line: u32 },
    Align(Expr, u32),
    Insn(Insn),
    Global(Vec<String>, u32),
    /// GAS `.weak`: global binding with weak precedence at link time.
    Weak(Vec<String>, u32),
    Extern(Vec<String>, u32),
    Section(SectionSpec),
    /// Default operand size (`bits 16/32/64`); only 64 is supported.
    Bits(u8, u32),
    /// `times N <inner>`; N must eventually be constant.
    Times { count: Expr, inner: Box<Stmt>, line: u32 },
    /// GAS `.comm`/`.lcomm`: tentative definition carved out of .bss.
    Comm { name: String, size: Expr, align: Option<u64>, global: bool, line: u32 },
}

impl Stmt {
    pub fn line(&self) -> u32 {
        match self {
            Stmt::Label(_, l)
            | Stmt::Data { line: l, .. }
            | Stmt::Reserve { line: l, .. }
            | Stmt::Align(_, l)
            | Stmt::Global(_, l)
            | Stmt::Weak(_, l)
            | Stmt::Extern(_, l)
            | Stmt::Bits(_, l)
            | Stmt::Times { line: l, .. }
            | Stmt::Comm { line: l, .. } => *l,
            Stmt::Insn(i) => i.line,
            Stmt::Section(_) => 0,
        }
    }
}
