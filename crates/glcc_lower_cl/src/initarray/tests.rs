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
