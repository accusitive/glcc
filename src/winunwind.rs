//! Windows x64 unwind tables (`.pdata`/`.xdata`).
//!
//! Cranelift computes a `UNWIND_INFO` record for every function on
//! Windows, but `cranelift-object` has no COFF unwind path (its
//! `unwind.rs` silently drops Windows infos), so the driver appends
//! the two sections itself: `.pdata` holds one 12-byte
//! `RUNTIME_FUNCTION` per function (begin/end/unwind RVAs, all
//! `IMAGE_REL_AMD64_ADDR32NB`), `.xdata` the serialized records.
//! Windows runtime unwinding (`RtlVirtualUnwind`, SEH dispatch, C++
//! exceptions, debuggers, `CaptureStackBackTrace`, crash reporting)
//! walks the table; without it stack walking stops at the first glcc
//! frame.

use cranelift_object::ObjectProduct;
use cranelift_object::object::write::{
    Object, Relocation, SectionId, SectionKind, Symbol, SymbolFlags, SymbolId, SymbolKind, SymbolScope, SymbolSection,
};
use cranelift_object::object::{RelocationEncoding, RelocationFlags, RelocationKind};
use glcc_lower_cl::WinUnwindEntry;

/// Both sections are read-only data; records are naturally 4-byte
/// aligned.
const SECTION_KIND: SectionKind = SectionKind::ReadOnlyData;

/// Append `.pdata`/`.xdata` for `entries` (definition order, which is
/// ascending `.text` order) to a finished object. No-op on modules
/// without captured records (non-Windows targets).
pub fn emit(product: &mut ObjectProduct, entries: &[WinUnwindEntry]) -> Result<(), String> {
    if entries.is_empty() {
        return Ok(());
    }
    let func_syms: Vec<SymbolId> = entries.iter().map(|e| product.function_symbol(e.func)).collect();
    let object = &mut product.object;
    let xdata = object.add_section(Vec::new(), b".xdata".to_vec(), SECTION_KIND);
    let pdata = object.add_section(Vec::new(), b".pdata".to_vec(), SECTION_KIND);

    let mut anchors = Vec::with_capacity(entries.len());
    let mut xdata_data = Vec::new();
    for (i, entry) in entries.iter().enumerate() {
        anchors.push(object.add_symbol(Symbol {
            name: format!(".xdata${i}").into_bytes(),
            value: xdata_data.len() as u64,
            size: entry.xdata.len() as u64,
            kind: SymbolKind::Data,
            scope: SymbolScope::Compilation,
            weak: false,
            section: SymbolSection::Section(xdata),
            flags: SymbolFlags::None,
        }));
        xdata_data.extend_from_slice(&entry.xdata);
    }
    let pdata_data = vec![0u8; entries.len() * 12];
    object.append_section_data(xdata, &xdata_data, 4);
    object.append_section_data(pdata, &pdata_data, 4);

    for (i, entry) in entries.iter().enumerate() {
        let row = i as u64 * 12;
        let relocs = [
            (row, func_syms[i], 0),
            (row + 4, func_syms[i], entry.size as i64),
            (row + 8, anchors[i], 0),
        ];
        for (offset, symbol, addend) in relocs {
            add_rva(object, pdata, offset, symbol, addend)?;
        }
    }
    Ok(())
}

/// Relocate one 32-bit field against a symbol's image-relative
/// address (`IMAGE_REL_AMD64_ADDR32NB`). The COFF writer stores the
/// implicit addend in the field itself.
fn add_rva(object: &mut Object, section: SectionId, offset: u64, symbol: SymbolId, addend: i64) -> Result<(), String> {
    object
        .add_relocation(
            section,
            Relocation {
                offset,
                symbol,
                addend,
                flags: RelocationFlags::Generic {
                    kind: RelocationKind::ImageOffset,
                    encoding: RelocationEncoding::Generic,
                    size: 32,
                },
            },
        )
        .map_err(|e| format!("cannot add .pdata relocation: {e}"))
}
