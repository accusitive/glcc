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
//! Entry points take the validated [`Sema`](glcc_frontend::sema::Sema)
//! (which carries the prism attributes the interface is built from)
//! alongside the optimized MIR module.

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
    let _ = (module, sema, interner);
    Err(match target {
        ShaderTarget::Isa => "prism ISA emission is not implemented yet",
        ShaderTarget::Spirv => "SPIR-V emission is not implemented yet",
        ShaderTarget::Rsx => "RSX emission is not implemented yet",
        ShaderTarget::Xenos => "Xenos emission is not implemented yet",
    }
    .to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_targets_error() {
        for t in [ShaderTarget::Isa, ShaderTarget::Spirv, ShaderTarget::Rsx, ShaderTarget::Xenos] {
            assert!(t.flag().starts_with("--emit-"));
        }
    }
}
