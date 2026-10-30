//! ELF64 relocatable output (reference: nasm's `output/outelf.c`), built
//! on the `object` crate's writer the same way cranelift-object is.

use crate::assemble::{Module, Target};
use crate::insn::SectionFlags;
use object::elf::{R_X86_64_32S, R_X86_64_64, R_X86_64_PC32};
use object::write::{Object, Relocation, Symbol, SymbolFlags, SymbolSection};
use object::write::SectionId;
use object::{
    Architecture, BinaryFormat, Endianness, RelocationFlags, SectionKind, SymbolKind, SymbolScope,
};

pub fn write_elf64(module: &Module) -> Result<Vec<u8>, String> {
    let mut obj =
        Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);

    let mut section_ids: Vec<SectionId> = Vec::with_capacity(module.sections.len());
    for s in &module.sections {
        let kind = match s.flags {
            SectionFlags::Text | SectionFlags::Custom { exec: true, .. } => SectionKind::Text,
            SectionFlags::Data | SectionFlags::Custom { nobits: false, .. } => SectionKind::Data,
            SectionFlags::Bss | SectionFlags::Custom { nobits: true, .. } => {
                SectionKind::UninitializedData
            }
            SectionFlags::Rodata => SectionKind::ReadOnlyData,
        };
        let id = obj.add_section(Vec::new(), s.name.as_bytes().to_vec(), kind);
        section_ids.push(id);
    }

    let mut symbol_ids = vec![None; module.symbols.len()];
    for (i, sym) in module.symbols.iter().enumerate() {
        if !(sym.global || sym.extern_) {
            // Local symbols are resolved statically or via section-relative
            // relocations; they carry no linking semantics of their own.
            continue;
        }
        let scope = if sym.extern_ {
            SymbolScope::Unknown
        } else {
            SymbolScope::Linkage
        };
        let section = match (sym.extern_, sym.sect) {
            (true, _) | (_, None) => SymbolSection::Undefined,
            (false, Some(si)) => SymbolSection::Section(section_ids[si]),
        };
        let kind = match sym.sect.map(|si| &module.sections[si].flags) {
            Some(SectionFlags::Text) => SymbolKind::Text,
            Some(_) => SymbolKind::Data,
            None => SymbolKind::Unknown,
        };
        let id = obj.add_symbol(Symbol {
            name: sym.name.as_bytes().to_vec(),
            value: sym.value as u64,
            size: 0,
            kind,
            scope,
            weak: false,
            section,
            flags: SymbolFlags::None,
        });
        symbol_ids[i] = Some(id);
    }

    for r in &module.relocs {
        let sym = match r.target {
            Target::Section(si) => obj.section_symbol(section_ids[si]),
            Target::Symbol(i) => symbol_ids[i]
                .ok_or_else(|| format!("relocation against non-linking symbol {i}"))?,
        };
        let r_type = match (r.pcrel, r.size) {
            (true, 4) => R_X86_64_PC32,
            (false, 8) => R_X86_64_64,
            (false, 4) => R_X86_64_32S,
            _ => return Err(format!("unsupported relocation (pcrel={}, size={})", r.pcrel, r.size)),
        };
        obj.add_relocation(
            section_ids[r.section],
            Relocation {
                offset: r.offset as u64,
                symbol: sym,
                addend: r.addend,
                flags: RelocationFlags::Elf { r_type },
            },
        )
        .map_err(|e| format!("bad relocation: {e}"))?;
    }

    for (s, id) in module.sections.iter().zip(&section_ids) {
        let sid = obj.section_mut(*id);
        if matches!(s.flags, SectionFlags::Bss | SectionFlags::Custom { nobits: true, .. }) {
            sid.append_bss(s.data.len() as u64, s.align.max(1));
        } else {
            sid.set_data(s.data.clone(), s.align.max(1));
        }
    }

    let mut out = Vec::new();
    obj.emit(&mut out).map_err(|e| format!("ELF emission failed: {e}"))?;
    Ok(out)
}
