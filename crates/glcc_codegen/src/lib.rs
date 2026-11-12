//! glcc's custom machine-code backend: MIR → machine IR → (RA,
//! scheduling, frame) → encoded objects. Architecturally: the pipeline
//! (see `pipeline()` below) is arch-neutral and pass-by-pass, targets
//! describe themselves with data (opcode/encoding/pattern tables) plus
//! a small `TargetIsa` impl, and register allocation runs on regalloc2
//! through a thin adapter. Targets that are not register machines
//! (stack VMs) skip the RA pass and encode their ISel output directly.

use isa::TargetIsa;

pub mod aarch64;
pub mod frame;
pub mod isa;
pub mod machine;
pub mod object;
pub mod printer;
pub mod ra;
pub mod sched;
pub mod sel;
pub mod x64;

/// Machine targets the native backend supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    X86_64,
    Aarch64,
}

impl Arch {
    pub fn isa(self) -> &'static dyn TargetIsa {
        match self {
            Arch::X86_64 => &x64::X64_ISA,
            Arch::Aarch64 => &aarch64::A64_ISA,
        }
    }
}

pub struct CodegenOutput {
    pub object: Vec<u8>,
    pub asm: String,
}

/// Full pipeline over one MIR module.
pub fn emit_module(mir: &glcc_mir::Module, arch: Arch) -> Result<CodegenOutput, String> {
    let isa = arch.isa();
    let emitted = object::emit(mir, isa)?;
    Ok(CodegenOutput { object: emitted.object, asm: emitted.asm })
}
