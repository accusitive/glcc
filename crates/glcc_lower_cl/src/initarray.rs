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
mod tests {
    use super::*;
    use crate::{Target, lower_module};
    use cranelift_codegen::settings::Configurable;
    use glcc_mir as mir;
    use object::{Object, ObjectSection, ObjectSymbol, SectionKind as ReadSectionKind, elf};
    use object::read::elf::SectionHeader as _;

    fn build_product(module: &mir::Module) -> (cranelift_object::ObjectProduct, std::collections::HashMap<mir::Func, FuncId>) {
        let mut flag_builder = cranelift_codegen::settings::builder();
        flag_builder.set("opt_level", "none").expect("flag");
        flag_builder.set("is_pic", "true").expect("flag");
        let flags = cranelift_codegen::settings::Flags::new(flag_builder);
        let isa = cranelift_codegen::isa::lookup("x86_64-unknown-linux-gnu".parse().expect("triple"))
            .expect("isa")
            .finish(flags)
            .expect("isa flags");
        let obj_builder = cranelift_object::ObjectBuilder::new(isa, "inittest", cranelift_module::default_libcall_names()).expect("object builder");
        let mut obj = cranelift_object::ObjectModule::new(obj_builder);
        let target = Target::from_triple("x86_64-unknown-linux-gnu").expect("target");
        let ids = lower_module(module, &mut obj, target);
        (obj.finish(), ids)
    }

    /// A void `() -> void` function that just returns.
    fn ctor(module: &mut mir::Module, name: &str) -> mir::Func {
        let (mut func, entry) = mir::Function::new(name, mir::Signature::new(vec![], mir::Ty::Void));
        func.blocks[entry].term = Some(mir::Terminator::Return { val: None });
        let id = module.push_func(func);
        module.init_funcs.push(id);
        id
    }

    #[test]
    fn emits_relocated_entries_in_order() {
        let mut module = mir::Module::new();
        let first = ctor(&mut module, "__sti_first");
        let second = ctor(&mut module, "__sti_second");
        let (mut product, ids) = build_product(&module);

        assert!(emit_init_array(&mut product, &[ids[&first], ids[&second]]).is_ok());
        let bytes = product.emit().expect("object emit");

        let file = object::File::parse(&bytes[..]).expect("elf parse");
        let section = file.section_by_name(".init_array").expect(".init_array section");
        assert_eq!(section.kind(), ReadSectionKind::Data, "alloc + write, not mergeable");
        let data = section.data().expect("section data");
        assert_eq!(data.len(), 16, "one 8-byte slot per constructor");

        let relocs: Vec<_> = section.relocations().collect();
        assert_eq!(relocs.len(), 2, "one pointer relocation per slot");
        for (index, (offset, reloc)) in relocs.iter().enumerate() {
            assert_eq!(*offset, index as u64 * 8, "entry {index} sits at its own slot");
            assert_eq!(reloc.kind(), object::RelocationKind::Absolute);
            assert_eq!(reloc.size(), 64, "R_X86_64_64");
            let name = file
                .symbol_by_index(match reloc.target() {
                    object::RelocationTarget::Symbol(index) => index,
                    other => panic!("reloc target is a symbol, got {other:?}"),
                })
                .expect("reloc target symbol")
                .name()
                .expect("symbol name");
            let expected = if index == 0 { "__sti_first" } else { "__sti_second" };
            assert_eq!(name, expected, "entry {index} relocates against the constructor");
        }
    }

    #[test]
    fn section_header_is_alloc_write_8_byte_aligned_progbits() {
        let mut module = mir::Module::new();
        let only = ctor(&mut module, "__sti_only");
        let (mut product, ids) = build_product(&module);
        assert!(emit_init_array(&mut product, &[ids[&only]]).is_ok());
        let bytes = product.emit().expect("object emit");

        let file = object::read::elf::ElfFile64::<object::Endianness>::parse(&bytes[..]).expect("elf parse");
        let endian = file.endianness();
        let section = file.sections().find(|s| s.name().expect("name") == ".init_array").expect(".init_array section");
        let header = section.elf_section_header();
        assert_eq!(header.sh_addralign(endian), 8);
        assert_eq!(header.sh_type(endian), elf::SHT_PROGBITS);
        let flags = header.sh_flags(endian);
        assert_eq!(flags & (elf::SHF_ALLOC | elf::SHF_WRITE), elf::SHF_ALLOC | elf::SHF_WRITE);
        assert_eq!(flags & elf::SHF_MERGE, elf::SectionFlags(0));
    }

    #[test]
    fn empty_list_emits_nothing() {
        let mut module = mir::Module::new();
        ctor(&mut module, "unused");
        let (mut product, _) = build_product(&module);
        assert!(emit_init_array(&mut product, &[]).is_ok());
        let bytes = product.emit().expect("object emit");
        let file = object::File::parse(&bytes[..]).expect("elf parse");
        assert!(file.section_by_name(".init_array").is_none(), "no .init_array without constructors");
    }
}
