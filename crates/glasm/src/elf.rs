//! ELF64 relocatable output (reference: nasm's `output/outelf.c`), built
//! on the `object` crate's writer the same way cranelift-object is.

use crate::arch::Backend;
use crate::assemble::{Module, Target};
use crate::insn::SectionFlags;
use object::elf::{STB_GLOBAL, STB_WEAK, STV_DEFAULT};
use object::write::SectionId;
use object::write::{Object, Relocation, Symbol, SymbolFlags, SymbolSection};
use object::{Architecture, BinaryFormat, RelocationFlags, SectionKind, SymbolKind, SymbolScope};

/// Public entry: serialize a finished module for `arch`.
pub fn write_module_elf(module: &Module, arch: crate::arch::Arch) -> Result<Vec<u8>, String> {
    let backend = arch.backend();
    write_elf(module, backend.as_ref())
}

pub(crate) fn write_elf(module: &Module, backend: &dyn Backend) -> Result<Vec<u8>, String> {
    let mut obj = Object::new(BinaryFormat::Elf, backend.object_arch(), backend.endianness());

    let mut section_ids: Vec<SectionId> = Vec::with_capacity(module.sections.len());
    for s in &module.sections {
        let kind = match s.flags {
            SectionFlags::Text | SectionFlags::Custom { exec: true, .. } => SectionKind::Text,
            SectionFlags::Data | SectionFlags::Custom { nobits: false, .. } => SectionKind::Data,
            SectionFlags::Bss | SectionFlags::Custom { nobits: true, .. } => SectionKind::UninitializedData,
            SectionFlags::Rodata => SectionKind::ReadOnlyData,
        };
        let id = obj.add_section(Vec::new(), s.name.as_bytes().to_vec(), kind);
        section_ids.push(id);
    }

    // Local symbols are emitted only when a relocation references them
    // (RISC-V %pcrel_lo pairs target internal auipc labels).
    let mut needed_locals: Vec<usize> = Vec::new();
    for r in &module.relocs {
        if let Target::Symbol(i) = r.target {
            let sym = &module.symbols[i];
            if !(sym.global || sym.extern_) {
                needed_locals.push(i);
            }
        }
    }

    let mut symbol_ids = vec![None; module.symbols.len()];
    for (i, sym) in module.symbols.iter().enumerate() {
        if !(sym.global || sym.extern_) {
            if !needed_locals.contains(&i) {
                continue;
            }
            // Local relocation target: internal label (STB_LOCAL).
        }
        let is_local = !(sym.global || sym.extern_);
        let scope = if sym.extern_ {
            SymbolScope::Unknown
        } else if is_local {
            SymbolScope::Compilation
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
        // Explicit flags: the object crate maps Scope::Linkage to
        // STV_HIDDEN (its Cranelift heritage), but GNU as emits
        // STV_DEFAULT for plain globals, and NOTYPE types unless a
        // .type directive says otherwise.
        let bind = if is_local {
            object::elf::STB_LOCAL
        } else if sym.weak {
            STB_WEAK
        } else {
            STB_GLOBAL
        };
        let id = obj.add_symbol(Symbol {
            name: sym.name.as_bytes().to_vec(),
            value: sym.value as u64,
            size: 0,
            kind,
            scope,
            weak: sym.weak,
            section,
            flags: SymbolFlags::Elf {
                st_info: bind | object::elf::STT_NOTYPE,
                st_other: STV_DEFAULT.into(),
            },
        });
        symbol_ids[i] = Some(id);
    }

    // Marker relocations (R_RISCV_RELAX) reference an absolute symbol;
    // the object crate cannot address the null symbol, so one shared
    // ABS symbol stands in for them.
    let abs_symbol = obj.add_symbol(Symbol {
        name: Vec::new(),
        value: 0,
        size: 0,
        kind: SymbolKind::Label,
        scope: SymbolScope::Compilation,
        weak: false,
        section: SymbolSection::Absolute,
        flags: SymbolFlags::None,
    });

    for r in &module.relocs {
        let sym = match r.target {
            Target::Section(si) => obj.section_symbol(section_ids[si]),
            Target::Symbol(i) => symbol_ids[i].ok_or_else(|| format!("relocation against non-linking symbol {i}"))?,
            Target::Abs => abs_symbol,
        };
        let r_type = match r.r_type {
            0 => {
                // x86: derived from shape (pcrel, size); the other
                // backends set r_type explicitly during finalize.
                if backend.object_arch() != Architecture::X86_64 {
                    return Err(format!("relocation missing r_type (pcrel={}, size={})", r.pcrel, r.size));
                }
                match (r.pcrel, r.size) {
                    (true, 4) => object::elf::R_X86_64_PC32.0,
                    (false, 8) => object::elf::R_X86_64_64.0,
                    (false, 4) => object::elf::R_X86_64_32S.0,
                    _ => return Err(format!("unsupported relocation (pcrel={}, size={})", r.pcrel, r.size)),
                }
            }
            t => t,
        };
        obj.add_relocation(
            section_ids[r.section],
            Relocation {
                offset: r.offset as u64,
                symbol: sym,
                addend: r.addend,
                flags: RelocationFlags::Elf {
                    r_type: object::elf::RelocationType(r_type),
                },
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
