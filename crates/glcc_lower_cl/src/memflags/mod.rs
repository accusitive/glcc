//! Cranelift memory flags used by the lowering.

use cranelift_codegen::ir::MemFlagsData;

/// Compiler-owned storage, known to be naturally aligned and accessible.
pub(crate) const fn trusted() -> MemFlagsData {
    MemFlagsData::trusted()
}

/// C-visible memory: never assume alignment (packed structs), but an
/// access that is optimized away cannot fault.
pub(crate) const fn notrap() -> MemFlagsData {
    MemFlagsData::new().with_notrap()
}

#[cfg(test)]
mod tests;

