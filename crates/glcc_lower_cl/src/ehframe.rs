//! Itanium zero-cost EH unwind metadata: per-function collection during
//! lowering and `.eh_frame` + `.gcc_except_table` emission onto the
//! finished object (the P2 piece of `docs/eh-plan.md`; proven by the
//! `eh_spike` example).
//!
//! The driver suppresses cranelift-object's own `.eh_frame`
//! (`ObjectBuilder::unwind_info(false)`) for modules with EH functions and
//! calls [`emit_unwind`] on the finished [`ObjectProduct`]. We then write
//! the whole section ourselves: a plain CIE for non-EH functions and a
//! personality CIE (`zPLR`, `__gxx_personality_v0`) for EH ones, FDEs from
//! Cranelift's `SystemVUnwindInfo` (the same CFI cranelift-object would
//! have written), and per-EH-function `.gcc_except_table` LSDAs whose
//! call-site records come from the compiled `call_sites` and whose type
//! table mirrors the landing pads' clause order.

use cranelift_codegen::gimli::constants::{self, DW_EH_PE_absptr, DW_EH_PE_sdata4, DW_EH_PE_sdata8,
    DW_EH_PE_udata2, DW_EH_PE_udata4, DW_EH_PE_udata8};
use cranelift_codegen::gimli::write::{
    Address, CallFrameInstruction, CommonInformationEntry, EhFrame, EndianVec,
    FrameDescriptionEntry, FrameTable, RelocateWriter, Relocation, RelocationTarget,
};
use cranelift_codegen::gimli::{Encoding, Format, Register, RunTimeEndian};
use cranelift_codegen::isa::unwind::{systemv, UnwindInfo};
use cranelift_codegen::isa::TargetIsa;
use cranelift_codegen::{Context, FinalizedMachExceptionHandler};
use cranelift_module::FuncId;
use glcc_mir as mir;
use object::write::{
    Object, Relocation as ObjRelocation, StandardSection, Symbol as ObjSymbol, SymbolSection,
};
use object::{
    RelocationEncoding, RelocationFlags, RelocationKind, SectionKind, SymbolFlags, SymbolKind,
    SymbolScope,
};

/// EH metadata collected for one lowered function.
pub struct FuncEh {
    /// The function's Clif id (its text symbol).
    pub func: FuncId,
    /// The function has a personality: it needs an LSDA and the
    /// personality CIE.
    pub eh: bool,
    /// Cranelift's SystemV CFI for the function (`None` on Windows; the
    /// driver's `winunwind` path covers those objects).
    pub sysv: Option<systemv::UnwindInfo>,
    /// `(return-address offset, landing-pad offset)` per invoke call
    /// site, ascending. Empty for non-EH functions.
    pub sites: Vec<(u32, u32)>,
    /// Type table in clause order: typeinfo global names.
    pub type_table: Vec<String>,
}

/// Collect one function's EH metadata right after it compiled. Cheap for
/// non-EH functions (one unwind-info pass, no call-site scan).
pub(crate) fn collect_func_eh(
    mir_func: &mir::Function,
    globals: &index_vec::IndexVec<mir::Global, mir::GlobalData>,
    clif_id: FuncId,
    ctx: &Context,
    isa: &dyn TargetIsa,
) -> FuncEh {
    let eh = mir_func.personality.is_some();
    // Recording/mock modules never compile; they never emit unwind
    // metadata either, so skip collection there.
    let Some(code) = ctx.compiled_code() else {
        return FuncEh { func: clif_id, eh: false, sysv: None, sites: Vec::new(), type_table: Vec::new() };
    };
    let sysv = match code.create_unwind_info(isa).expect("unwind info") {
        Some(UnwindInfo::SystemV(info)) => Some(info),
        Some(_) => None,
        None => None,
    };
    let sites = if eh {
        code.buffer
            .call_sites()
            .filter_map(|cs| {
                let lpad = cs.exception_handlers.iter().find_map(|h| match h {
                    FinalizedMachExceptionHandler::Default(offset) => Some(*offset),
                    _ => None,
                })?;
                Some((cs.ret_addr, lpad))
            })
            .collect()
    } else {
        Vec::new()
    };
    let type_table = if eh {
        mir_func
            .blocks
            .iter()
            .flat_map(|b| b.lpad.iter().flatten())
            .filter_map(|clause| match clause {
                mir::LpadClause::Catch(global) => Some(globals[*global].name.clone()),
                _ => None,
            })
            .collect()
    } else {
        Vec::new()
    };
    FuncEh { func: clif_id, eh, sysv, sites, type_table }
}

fn write_uleb(b: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            b.push(byte);
            return;
        }
        b.push(byte | 0x80);
    }
}

fn write_sleb(b: &mut Vec<u8>, mut v: i64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        let sign_bit = byte & 0x40 != 0;
        if (v == 0 && !sign_bit) || (v == -1 && sign_bit) {
            b.push(byte);
            return;
        }
        b.push(byte | 0x80);
    }
}

/// `.gcc_except_table` LSDA: one catch clause set, call-site ranges from
/// the compiled `ret_addr`s (the conservative `[prev, this)` encoding; the
/// personality looks up `ret_addr - 1`), landing pad from the compiled
/// handler record, type table in clause order. Returns the bytes and the
/// offset of the TType slot region start... plus, per entry, the offset of
/// each 4-byte slot (for the `_ZTI*` relocations).
fn build_lsda(sites: &[(u32, u32)], entries: usize) -> (Vec<u8>, Vec<usize>) {
    let mut b = Vec::new();
    b.push(0xff); // LPStart encoding: DW_EH_PE_omit (LPStart = function start)
    b.push(0x1b); // TType encoding: DW_EH_PE_pcrel|sdata4
    let ttype_off_pos = b.len();
    b.push(0); // uleb128 TType offset, patched below
    b.push(0x01); // call-site encoding: DW_EH_PE_uleb128 (plain offsets)

    let mut cs = Vec::new();
    let mut prev = 0;
    for &(ret, lpad) in sites {
        write_uleb(&mut cs, prev as u64);
        write_uleb(&mut cs, (ret - prev) as u64);
        write_uleb(&mut cs, lpad as u64);
        write_uleb(&mut cs, 1); // action: filter 1
        prev = ret;
    }
    write_uleb(&mut b, cs.len() as u64);
    b.extend_from_slice(&cs);

    // Action table: filter 1 (type table slot 1), no next action.
    write_sleb(&mut b, 1);
    write_sleb(&mut b, 0);

    // Type table, written backwards: the personality reads entry N
    // (clause N-1, filter N) at (base - N*4), so the slots go in reverse
    // clause order, ending at the TType base, and the uleb points just
    // past the last slot.
    let mut slot_offsets = vec![0usize; entries];
    for clause in (0..entries).rev() {
        slot_offsets[clause] = b.len();
        b.extend_from_slice(&[0; 4]);
    }
    let ttype_off = b.len() - (ttype_off_pos + 1);
    b[ttype_off_pos] = u8::try_from(ttype_off).expect("TType offset must fit one uleb byte");
    (b, slot_offsets)
}

fn undefined_symbol(obj: &mut Object<'static>, name: &str) -> object::write::SymbolId {
    obj.add_symbol(ObjSymbol {
        name: name.as_bytes().to_vec(),
        value: 0,
        size: 0,
        kind: SymbolKind::Data,
        scope: SymbolScope::Unknown,
        weak: false,
        section: SymbolSection::Undefined,
        flags: SymbolFlags::None,
    })
}

/// (code alignment, DWARF return-address register, DWARF stack-pointer
/// register, CFA offset relative to SP, return-address offset from CFA) per
/// ISA, mirroring each backend's `create_cie`. x86-64 saves the return
/// address at `CFA-8` on every call; aarch64 keeps it in `x30` (the FDE's
/// prologue rule covers saving it).
fn cie_params(isa: &dyn TargetIsa) -> (u8, Register, Register, i32, Option<i32>) {
    match isa.triple().architecture {
        target_lexicon::Architecture::Aarch64(_) => (4, Register(30), Register(31), 0, None),
        target_lexicon::Architecture::X86_64 => (1, Register(16), Register(7), 8, Some(-8)),
        arch => panic!("no SystemV CIE parameters for {arch}"),
    }
}

fn eh_pe_flags(pe: Option<constants::DwEhPe>, size: u8) -> (RelocationKind, u8) {
    let Some(pe) = pe else {
        return (RelocationKind::Absolute, size * 8);
    };
    let kind = if pe.application() == constants::DW_EH_PE_pcrel {
        RelocationKind::Relative
    } else if pe.application() == DW_EH_PE_absptr {
        RelocationKind::Absolute
    } else {
        panic!("unsupported eh_frame pointer application {:?}", pe.application())
    };
    let bit_size = if pe.format() == DW_EH_PE_absptr {
        size * 8
    } else if pe.format() == DW_EH_PE_udata4 || pe.format() == DW_EH_PE_sdata4 {
        32
    } else if pe.format() == DW_EH_PE_udata8 || pe.format() == DW_EH_PE_sdata8 {
        64
    } else if pe.format() == DW_EH_PE_udata2 {
        16
    } else {
        panic!("unsupported eh_frame pointer format {:?}", pe.format())
    };
    (kind, bit_size)
}

struct FrameSection {
    writer: EndianVec<RunTimeEndian>,
    relocations: Vec<Relocation>,
}

impl RelocateWriter for FrameSection {
    type Writer = EndianVec<RunTimeEndian>;

    fn writer(&self) -> &Self::Writer {
        &self.writer
    }

    fn writer_mut(&mut self) -> &mut Self::Writer {
        &mut self.writer
    }

    fn relocate(&mut self, relocation: Relocation) {
        self.relocations.push(relocation);
    }
}

/// Emit `.eh_frame` (plain + personality CIEs, FDEs for every function)
/// and `.gcc_except_table` (one LSDA per EH function) onto `product`.
/// Requires that cranelift-object's own `.eh_frame` was suppressed
/// (`ObjectBuilder::unwind_info(false)`) and that no function was skipped
/// at lowering time. `typeinfo` resolves a typeinfo global name to its
/// object symbol (declared by lowering as a data extern); unknown names
/// become undefined symbols so the linker resolves them from libstdc++.
pub fn emit_unwind(
    product: &mut cranelift_object::ObjectProduct,
    funcs: &[FuncEh],
    global_names: &[String],
    isa: &dyn TargetIsa,
) -> Result<(), String> {
    let func_symbols: std::collections::HashMap<FuncId, object::write::SymbolId> =
        funcs.iter().map(|f| (f.func, product.function_symbol(f.func))).collect();
    let obj = &mut product.object;
    if obj.format() != object::BinaryFormat::Elf {
        return Err(format!("no .eh_frame emission for {:?} objects", obj.format()));
    }
    if funcs.iter().any(|f| f.eh && f.sites.is_empty()) {
        return Err("EH function without invoke call sites".to_string());
    }
    let endian = match isa.triple().endianness().unwrap() {
        target_lexicon::Endianness::Little => RunTimeEndian::Little,
        target_lexicon::Endianness::Big => RunTimeEndian::Big,
    };

    // .gcc_except_table: one LSDA per EH function.
    let lsda_id = obj.add_section(Vec::new(), b".gcc_except_table".to_vec(), SectionKind::ReadOnlyData);
    struct Lsda {
        sym: object::write::SymbolId,
        /// (slot offset, typeinfo name) per type table entry.
        slots: Vec<(u64, String)>,
    }
    let mut lsdas: Vec<Lsda> = Vec::new();
    for f in funcs.iter().filter(|f| f.eh) {
        let (bytes, slot_offsets) = build_lsda(&f.sites, f.type_table.len());
        let off = obj.append_section_data(lsda_id, &bytes, 4);
        let sym = obj.add_symbol(ObjSymbol {
            name: format!("_GCC_except_table{}", lsdas.len()).into_bytes(),
            value: off,
            size: bytes.len() as u64,
            kind: SymbolKind::Data,
            scope: SymbolScope::Compilation,
            weak: false,
            section: SymbolSection::Section(lsda_id),
            flags: SymbolFlags::None,
        });
        let slots = slot_offsets
            .into_iter()
            .zip(f.type_table.iter())
            .map(|(o, name)| (off + o as u64, name.clone()))
            .collect();
        lsdas.push(Lsda { sym, slots });
    }

    // .eh_frame: plain CIE for non-EH functions, personality CIE for EH ones.
    let eh_frame_id = obj.section_id(StandardSection::EhFrame);
    let mut frame_table = FrameTable::default();
    // Index space for gimli relocations: gimli addresses reference these
    // indices, resolved to object SymbolIds after serialization.
    let mut symbols: Vec<object::write::SymbolId> = Vec::new();

    let personality_idx = symbols.len();
    symbols.push(undefined_symbol(obj, "__gxx_personality_v0"));

    let (code_align, ra_reg, sp_reg, cfa_sp_off, ra_offset) = cie_params(isa);
    let encoding = Encoding { address_size: 8, format: Format::Dwarf32, version: 1 };

    let plain_cie = {
        let mut cie = CommonInformationEntry::new(encoding, code_align, -8, ra_reg);
        cie.add_instruction(CallFrameInstruction::Cfa(sp_reg, cfa_sp_off));
        if let Some(offset) = ra_offset {
            cie.add_instruction(CallFrameInstruction::Offset(ra_reg, offset));
        }
        cie
    };
    let eh_cie = {
        let mut cie = CommonInformationEntry::new(encoding, code_align, -8, ra_reg);
        cie.add_instruction(CallFrameInstruction::Cfa(sp_reg, cfa_sp_off));
        if let Some(offset) = ra_offset {
            cie.add_instruction(CallFrameInstruction::Offset(ra_reg, offset));
        }
        cie.personality = Some((
            constants::DW_EH_PE_absptr,
            Address::Symbol { symbol: personality_idx, addend: 0 },
        ));
        cie.lsda_encoding = Some(constants::DW_EH_PE_pcrel | constants::DW_EH_PE_sdata4);
        cie.fde_address_encoding = constants::DW_EH_PE_pcrel | constants::DW_EH_PE_sdata4;
        cie
    };
    let plain_cie_id = frame_table.add_cie(plain_cie);
    let eh_cie_id = frame_table.add_cie(eh_cie);

    let mut lsda_iter = lsdas.iter();
    for f in funcs {
        let Some(sysv) = &f.sysv else {
            return Err(format!("function f{} has no SystemV unwind info", f.func.as_u32()));
        };
        let func_idx = symbols.len();
        symbols.push(func_symbols[&f.func]);
        let mut fde: FrameDescriptionEntry =
            sysv.to_fde(Address::Symbol { symbol: func_idx, addend: 0 });
        if f.eh {
            let lsda = lsda_iter.next().ok_or("ran out of LSDAs")?;
            let lsda_idx = symbols.len();
            symbols.push(lsda.sym);
            fde.lsda = Some(Address::Symbol { symbol: lsda_idx, addend: 0 });
            frame_table.add_fde(eh_cie_id, fde);
        } else {
            frame_table.add_fde(plain_cie_id, fde);
        }
    }

    let frame_section = FrameSection { writer: EndianVec::new(endian), relocations: Vec::new() };
    let frame_section = {
        let mut eh_frame = EhFrame(frame_section);
        frame_table.write_eh_frame(&mut eh_frame).map_err(|e| format!("write .eh_frame: {e}"))?;
        eh_frame.0
    };
    obj.append_section_data(eh_frame_id, &frame_section.writer.into_vec(), 8);
    for reloc in &frame_section.relocations {
        let symbol = match reloc.target {
            RelocationTarget::Symbol(idx) => symbols[idx],
            other => return Err(format!("unexpected .eh_frame relocation target {other:?}")),
        };
        let (kind, size) = eh_pe_flags(reloc.eh_pe, reloc.size);
        obj.add_relocation(
            eh_frame_id,
            ObjRelocation {
                offset: reloc.offset as u64,
                symbol,
                addend: reloc.addend,
                flags: RelocationFlags::Generic {
                    kind,
                    encoding: RelocationEncoding::Generic,
                    size,
                },
            },
        )
        .map_err(|e| format!(".eh_frame relocation: {e}"))?;
    }

    // LSDA type-table slots: pcrel sdata4 against each typeinfo symbol.
    // Declared typeinfos resolve to their object symbols; anything else
    // stays undefined for the linker (libstdc++).
    for lsda in &lsdas {
        for (offset, name) in &lsda.slots {
            let symbol = obj
                .symbol_id(name.as_bytes())
                .unwrap_or_else(|| undefined_symbol(obj, name));
            obj.add_relocation(
                lsda_id,
                ObjRelocation {
                    offset: *offset,
                    symbol,
                    addend: 0,
                    flags: RelocationFlags::Generic {
                        kind: RelocationKind::Relative,
                        encoding: RelocationEncoding::Generic,
                        size: 32,
                    },
                },
            )
            .map_err(|e| format!(".gcc_except_table relocation: {e}"))?;
        }
    }
    Ok(())
}
