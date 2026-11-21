//! The prism shader backend.
//!
//! Prism is a C-subset shader language (see `docs/prism.md`): the
//! frontend checks the prism profile, lowers to MIR like any other C,
//! and this crate turns the optimized MIR into shader programs. All
//! targets are produced from one shared shader IR, PrismIR — an
//! SSA, width-typed, DX9-flavored IR with structured control flow:
//!
//! ```text
//! MIR ──► PrismIR ──┬── .isa  text (the IR's own canonical form)
//!                    ├── SPIR-V binary
//!                    ├── RSX (PS3) program text
//!                    └── Xenos (Xbox 360) microcode text
//! ```
//!
//! Physical registers, write-mask splitting, register banking, and
//! target-specific packing (Xenos scalar/vector co-issue) are
//! per-target lowering concerns; PrismIR only carries typed virtual
//! registers and op widths. Instruction scheduling runs once, here,
//! before any target consumes the IR.
//!
//! The pipeline ([`emit`]): the shader interface comes from the
//! validated [`Sema`](glcc_frontend::sema::Sema) ([`interface`]); the
//! MIR module is prepared ([`prep`]: inline everything, promote all
//! locals to SSA); the structurizer and op lowering build PrismIR
//! ([`build`]); the program is checked ([`validate`]); the selected
//! target renders it. Phase C ships the ISA text target; SPIR-V, RSX,
//! and Xenos follow.


#[cfg(test)]
mod tests;

pub mod build;
pub mod interface;
pub mod ir;
pub mod isa;
pub mod prep;
pub mod validate;

use glcc_frontend::lexer::Interner;
use glcc_frontend::sema::Sema;

/// The shader output formats, selected by the driver's `--emit-*`
/// flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderTarget {
    /// The PrismIR canonical text form (`--emit-isa`).
    Isa,
    /// SPIR-V binary (`--emit-spirv`).
    Spirv,
    /// RSX (PS3) program text (`--emit-rsx`).
    Rsx,
    /// Xenos (Xbox 360) microcode text (`--emit-xenos`).
    Xenos,
}

impl ShaderTarget {
    /// The driver flag that selects this target.
    pub fn flag(self) -> &'static str {
        match self {
            ShaderTarget::Isa => "--emit-isa",
            ShaderTarget::Spirv => "--emit-spirv",
            ShaderTarget::Rsx => "--emit-rsx",
            ShaderTarget::Xenos => "--emit-xenos",
        }
    }
}

/// Compile one optimized MIR module into `target`'s artifact.
/// Errors are driver-renderable messages (e.g. "out of temporary
/// registers"); frontend diagnostics were already reported by the
/// time this runs.
pub fn emit(module: &glcc_mir::Module, sema: &Sema, interner: &Interner, target: ShaderTarget) -> Result<Vec<u8>, String> {
    match target {
        ShaderTarget::Isa => {
            let interface = interface::build(sema, interner)?;
            let prepared = prep::prepare(module)?;
            let func = prepared
                .funcs
                .iter()
                .find(|f| f.name == interface.entry)
                .ok_or_else(|| format!("prism: entry point '{}' has no definition", interface.entry))?;
            let program = build::build(&interface, &prepared, func)?;
            let text = isa::print(&program);
            if let Err(errors) = validate::validate(&program) {
                if std::env::var_os("GLCC_PRISM_DEBUG").is_some() {
                    eprintln!("{text}");
                }
                return Err(errors);
            }
            Ok(text.into_bytes())
        }
        ShaderTarget::Spirv => Err("SPIR-V emission is not implemented yet".into()),
        ShaderTarget::Rsx => Err("RSX emission is not implemented yet".into()),
        ShaderTarget::Xenos => Err("Xenos emission is not implemented yet".into()),
    }
}
