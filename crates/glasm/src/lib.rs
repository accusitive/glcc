//! glasm: a native Rust assembler for glcc, ported from nasm.
//!
//! Pipeline: `parse`/`gas` turn source into statements, `assemble` runs
//! the multi-pass encoding loop over them (via the per-architecture
//! `Backend` from `arch`), and `elf` serializes the resulting module.

pub mod aarch64;
pub mod arch;
pub mod assemble;
pub mod elf;
pub mod encoding;
pub mod expr;
pub mod gas;
pub use glasm_data::insns_gen;
pub mod insn;
pub mod parse;
pub mod regs;
pub mod riscv;
pub mod table;
pub mod x86;

pub use arch::Arch;
pub use assemble::{AsmError, Module};
pub use insn::Stmt;

/// Assembles nasm Intel-syntax source (x86_64 only) into an ELF64 object.
pub fn assemble_to_elf64(src: &str) -> Result<Vec<u8>, Vec<AsmError>> {
    let stmts = parse::parse_module(src).map_err(|errs| errs.into_iter().map(|e| AsmError { line: e.line, msg: e.msg }).collect::<Vec<_>>())?;
    let module = assemble::assemble(&stmts, Arch::X86_64)?;
    elf::write_module_elf(&module, Arch::X86_64).map_err(|e| vec![AsmError { line: 0, msg: e }])
}
