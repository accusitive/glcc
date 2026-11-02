//! ELF pre-`main` constructors (`.init_array`).
//!
//! [`glcc_mir::ir::Module::init_funcs`] names the functions that must
//! run before `main`; this module appends the object-level table to a
//! finished [`ObjectProduct`], the way the driver's `winunwind` module
//! appends `.pdata`/`.xdata`. Each entry is one 8-byte function pointer
//! patched with an `R_X86_64_64` relocation against the function
//! symbol. The section lists the MIR constructors in order, and the
//! linker concatenates per-TU `.init_array` sections in link order, so
//! the final run order matches source order across the program.
//!
//! Windows: not implemented. MSVC has no `.init_array`; the equivalent
//! story (`.CRT$XCU`) needs COFF grouping sections carrying absolute
//! pointer *images* rather than relocations, which would also want a
//! `.CRT` package (or `/INCLUDE` magic) to pull the section in — worth
//! doing only when a Windows C++ target is actually needed.

use cranelift_module::FuncId;
use cranelift_object::ObjectProduct;
use cranelift_object::object::{BinaryFormat, RelocationEncoding, RelocationFlags, RelocationKind, SectionKind};
use cranelift_object::object::write::{Object, Relocation};

/// LP64: one pointer-width slot per constructor.
const PTR_SIZE: u64 = 8;

/// Append `.init_array` with one relocated function pointer per id, in
/// slice order. ELF only: any other object format is an error the
/// caller reports as a warning (same convention as the winunwind path).
pub fn emit_init_array(product: &mut ObjectProduct, init_funcs: &[FuncId]) -> Result<(), String> {
    if init_funcs.is_empty() {
        return Ok(());
    }
    if product.object.format() != BinaryFormat::Elf {
        return Err(format!("no .init_array emission for {:?} objects", product.object.format()));
    }
    let symbols: Vec<_> = init_funcs.iter().map(|id| product.function_symbol(*id)).collect();
    let object: &mut Object = &mut product.object;
    // `SectionKind::Data` maps to SHF_ALLOC | SHF_WRITE on ELF; the
    // default linker script collects `.init_array` by name.
    let section = object.add_section(Vec::new(), b".init_array".to_vec(), SectionKind::Data);
    let data = vec![0u8; init_funcs.len() as usize * PTR_SIZE as usize];
    object.append_section_data(section, &data, PTR_SIZE);
    for (i, symbol) in symbols.into_iter().enumerate() {
        object
            .add_relocation(
                section,
                Relocation {
                    offset: i as u64 * PTR_SIZE,
                    symbol,
                    addend: 0,
                    flags: RelocationFlags::Generic {
                        kind: RelocationKind::Absolute,
                        encoding: RelocationEncoding::Generic,
                        size: 64,
                    },
                },
            )
            .map_err(|e| format!("cannot add .init_array relocation: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;

