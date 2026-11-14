//! Architecture dispatch: the statement pipeline (parsing, multi-pass
//! loop, symbol/section state) is shared, while operand syntax, encoding
//! and relocation layout are per-architecture `Backend`s.

use crate::encoding::Encoded;
use crate::expr::EvalCtx;
use object::{Architecture, Endianness};

/// Target architectures. x86_64 is the nasm-port backend; RISC-V and
/// AArch64 are purpose-built fixed-width encoders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    X86_64,
    Riscv32,
    Riscv64,
    Aarch64,
}

impl Arch {
    pub fn from_str(name: &str) -> Option<Arch> {
        match name {
            "x86_64" | "x86-64" | "amd64" => Some(Arch::X86_64),
            "riscv32" | "riscv" => Some(Arch::Riscv32),
            "riscv64" => Some(Arch::Riscv64),
            "aarch64" | "arm64" => Some(Arch::Aarch64),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Arch::X86_64 => "x86_64",
            Arch::Riscv32 => "riscv32",
            Arch::Riscv64 => "riscv64",
            Arch::Aarch64 => "aarch64",
        }
    }

    pub(crate) fn backend(&self) -> Box<dyn Backend> {
        match self {
            Arch::X86_64 => Box::new(crate::x86::X86Backend),
            Arch::Riscv32 | Arch::Riscv64 => Box::new(crate::riscv::RiscvBackend { arch: *self }),
            Arch::Aarch64 => Box::new(crate::aarch64::A64Backend),
        }
    }
}

/// One instruction, whichever architecture it belongs to.
#[derive(Clone, Debug)]
pub enum Insn {
    X86(crate::insn::X86Insn),
    Riscv(crate::riscv::Insn),
    Aarch64(crate::aarch64::Insn),
}

impl Insn {
    pub fn line(&self) -> u32 {
        match self {
            Insn::X86(i) => i.line,
            Insn::Riscv(i) => i.line,
            Insn::Aarch64(i) => i.line,
        }
    }
}

/// Per-architecture encoder + finalizer. Everything else (sections,
/// symbols, expressions, the pass loop) is arch-neutral.
pub(crate) trait Backend {
    /// Encode one instruction into its 4..N bytes plus fixups.
    /// `relax_state` selects the form after branch relaxation (0 is the
    /// most optimistic; higher states are progressively longer).
    fn encode(&self, insn: &Insn, ctx: &EvalCtx, relax_state: u8) -> Result<Encoded, String>;

    /// Patch all resolved fixups into section data, recording
    /// relocations. Backends see the full list so paired relocations
    /// (RISC-V %pcrel_lo references its HI20 fixup) can be resolved;
    /// `symbols` may be grown for synthesized relocation targets.
    fn finalize_all(
        &self,
        sections: &mut [crate::assemble::Section],
        symbols: &mut Vec<crate::assemble::Symbol>,
        fixups: Vec<crate::assemble::Fixup>,
        relocs: &mut Vec<crate::assemble::Reloc>,
    ) -> Result<(), Vec<String>>;

    /// The next relaxation state for this fixup when the resolved
    /// displacement `delta` outgrew the current form, or None.
    fn relax_step(&self, fixup: &crate::assemble::Fixup, delta: i64, state: u8) -> Option<u8>;

    fn object_arch(&self) -> Architecture;
    fn endianness(&self) -> Endianness;
    /// Fill pattern for padding in executable sections (a NOP).
    fn text_fill(&self) -> &'static [u8];
}

/// Parse a GAS-syntax instruction line for this arch; x86 goes through
/// the AT&T translator, the other arches have native operand syntaxes.
pub fn parse_gas_insn(arch: Arch, g: &[crate::gas::GTok], line_no: u32, lineno: usize, numlabels: &crate::gas::NumLabels, aliases: &crate::gas::RegAliases) -> Result<crate::insn::Stmt, String> {
    match arch {
        Arch::X86_64 => crate::gas::parse_x86_insn(g, line_no, lineno, numlabels, aliases),
        Arch::Riscv32 | Arch::Riscv64 => crate::riscv::parse_insn(g, line_no, lineno, numlabels),
        Arch::Aarch64 => crate::aarch64::parse_insn(g, line_no, lineno, numlabels),
    }
}
