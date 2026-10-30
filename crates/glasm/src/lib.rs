//! glasm: a native Rust assembler for glcc, ported from nasm.
//!
//! Pipeline: `parse` turns source into statements, `assemble` runs the
//! multi-pass encoding loop over them, and `elf` (or a future COFF
//! backend) serializes the resulting module.

pub mod assemble;
pub mod elf;
pub mod encoding;
pub mod expr;
pub mod gas;
pub use glasm_data::insns_gen;
pub mod insn;
pub mod parse;
pub mod regs;
pub mod table;

pub use assemble::{AsmError, Module};
pub use insn::Stmt;

/// Assembles source text into an ELF64 relocatable object.
pub fn assemble_to_elf64(src: &str) -> Result<Vec<u8>, Vec<AsmError>> {
    let stmts = parse::parse_module(src).map_err(|errs| {
        errs.into_iter()
            .map(|e| AsmError { line: e.line, msg: e.msg })
            .collect::<Vec<_>>()
    })?;
    let module = assemble::assemble(&stmts)?;
    elf::write_elf64(&module).map_err(|e| vec![AsmError { line: 0, msg: e }])
}
