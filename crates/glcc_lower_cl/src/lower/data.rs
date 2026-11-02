use std::collections::HashMap;

use cranelift_module::{DataDescription, FuncId, Module};

use glcc_mir as mir;

use super::*;

pub(crate) fn declare_global<M: Module>(global: &mir::GlobalData, clif: &mut M) -> cranelift_module::DataId {
    // Imports name storage defined elsewhere: declare without an image.
    if global.external {
        return clif
            .declare_data(&global.name, import_linkage(global.weak), true, global.tls)
            .unwrap_or_else(|_| panic!("cannot declare extern data {}", global.name));
    }
    let linkage = definition_linkage(global.local, global.weak);
    clif.declare_data(&global.name, linkage, global.writable, global.tls)
        .unwrap_or_else(|_| panic!("cannot declare data {}", global.name))
}

/// Define one declared global: fold the constant image, patch address
/// relocations over it, then apply the declared alignment (every C
/// object's natural `_Alignof`, so `double` never lands in a
/// byte-aligned slot) and any custom section (`.init_array`).
pub(crate) fn define_global<M: Module>(
    global: &mir::GlobalData,
    id: cranelift_module::DataId,
    data_ids: &HashMap<mir::Global, cranelift_module::DataId>,
    func_ids: &HashMap<mir::Func, FuncId>,
    extern_ids: &HashMap<mir::Extern, FuncId>,
    clif: &mut M,
    narrow: &mut Vec<NarrowReloc>,
) {
    if global.external {
        return;
    }
    // Zero-sized objects (empty structs, `T x[0]` members) still need a
    // symbol; one byte of storage keeps Cranelift's data machinery
    // happy without observable effect.
    let size = global.ty.size_bytes().max(1);
    let mut bytes = vec![0u8; size as usize];
    if let Some(init) = &global.init {
        write_const(&mut bytes, 0, init);
    }
    let mut desc = DataDescription::new();
    // An all-zero object with no custom section is uninitialized storage:
    // defining it as such keeps it out of the object file. (`.init_array`
    // and friends must stay PROGBITS even when empty.) Read-only objects
    // never take the zero-init path: `.bss` is writable, so an all-zero
    // literal like `""` would otherwise stay mutable.
    if global.writable && global.section.is_none() && global.relocs.is_empty() && global.relocs32.is_empty() && bytes.iter().all(|b| *b == 0) {
        desc.define_zeroinit(bytes.len());
    } else {
        desc.define(bytes.into_boxed_slice());
    }
    if let Some(align) = global.align {
        desc.set_align(align);
    }
    if let Some(section) = &global.section {
        desc.set_custom_section(section);
    }
    for (off, reloc) in &global.relocs {
        match reloc {
            mir::DataReloc::Global { target, addend } => {
                let gv = clif.declare_data_in_data(data_ids[target], &mut desc);
                desc.write_data_addr(*off as u32, gv, *addend);
            }
            mir::DataReloc::Func { target } => {
                let fr = clif.declare_func_in_data(func_ids[target], &mut desc);
                desc.write_function_addr(*off as u32, fr);
            }
            mir::DataReloc::FuncExtern { target } => {
                let fr = clif.declare_func_in_data(extern_ids[target], &mut desc);
                desc.write_function_addr(*off as u32, fr);
            }
        }
    }
    // 32-bit patches stay out of the descriptor: Cranelift would emit a
    // pointer-width relocation and clobber the neighbouring bytes. The
    // object driver adds a 4-byte relocation over the zeroed image.
    for (off, reloc) in &global.relocs32 {
        let (target, addend) = match reloc {
            mir::DataReloc::Global { target, addend } => (NarrowTarget::Data(data_ids[target]), *addend),
            mir::DataReloc::Func { target } => (NarrowTarget::Func(func_ids[target]), 0),
            mir::DataReloc::FuncExtern { target } => (NarrowTarget::Func(extern_ids[target]), 0),
        };
        narrow.push(NarrowReloc {
            container: id,
            offset: *off,
            target,
            addend,
        });
    }
    clif.define_data(id, &desc).expect("cannot define data");
}

pub(crate) fn write_const(bytes: &mut [u8], off: usize, c: &mir::Const) {
    match c {
        mir::Const::Int { bits, value } => {
            let len = ((*bits).max(8) / 8) as usize;
            // Constants are i128 even for wider MIR types: sign-extend
            // the stored value over the full byte width.
            let image = value.to_le_bytes();
            let head = len.min(image.len());
            bytes[off..off + head].copy_from_slice(&image[..head]);
            let fill = if *value < 0 { 0xff } else { 0 };
            bytes[off + head..off + len].fill(fill);
        }
        mir::Const::Float { kind, bits } => {
            let len = match kind {
                mir::FloatKind::F32 => 4,
                mir::FloatKind::F64 => 8,
                mir::FloatKind::F80 | mir::FloatKind::F128 => 16,
            };
            bytes[off..off + len].copy_from_slice(&bits.to_le_bytes()[..len]);
        }
        mir::Const::Null => {}
        mir::Const::Undef => {}
        // Zeroinit: the image starts zeroed.
        mir::Const::Zero => {}
        mir::Const::Bytes(content) => {
            let end = off + content.len();
            if end > bytes.len() {
                panic!("bytes constant overruns its image");
            }
            bytes[off..end].copy_from_slice(content);
        }
    }
}

