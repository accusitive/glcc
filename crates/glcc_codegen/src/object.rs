//! Object emission: machine functions → bytes → a `glasm::assemble::Module`
//! → ELF. Intra-function branches and calls to strong local definitions
//! are patched directly; everything else becomes an ELF relocation with
//! an explicit type.

use glasm::assemble::{Module as AsmModule, Reloc, Section, Symbol, Target as RelocTarget};
use glasm::insn::SectionFlags;
use glasm::elf::write_module_elf;

use crate::frame;
use crate::isa::{EncOut, RelocReq, TargetIsa};
use crate::machine::{MFunction, RelocKind, SymTarget};
use crate::ra;
use crate::sel;

use glcc_mir::{Const, DataReloc};

type NamedSyms = std::collections::HashMap<&'static str, usize>;

pub struct Emitted {
    pub object: Vec<u8>,
    pub asm: String,
}

/// Encoded bytes of one function plus its pending link-time fields.
/// All offsets are function-relative until finalization.
struct TextBuilder {
    text: Vec<u8>,
    /// Byte offset of each block's first instruction.
    block_offsets: Vec<u32>,
    /// (field offset, target block).
    block_refs: Vec<(u32, u32)>,
    /// (field offset, request) from instruction encoding.
    relocs: Vec<(u32, RelocReq)>,
    /// Relocations against real symbols, resolved during finalization.
    out_relocs: Vec<Reloc>,
}

impl TextBuilder {
    fn new() -> Self {
        TextBuilder { text: Vec::new(), block_offsets: Vec::new(), block_refs: Vec::new(), relocs: Vec::new(), out_relocs: Vec::new() }
    }
}

pub fn emit(mir: &glcc_mir::Module, isa: &'static dyn TargetIsa) -> Result<Emitted, String> {
    let mut asm = AsmModule {
        sections: vec![Section { name: ".text".into(), flags: SectionFlags::Text, align: 16, data: Vec::new() }],
        symbols: Vec::new(),
        relocs: Vec::new(),
    };
    let mut printer = crate::printer::Printer::new();

    // Symbol indices, created lazily as targets appear.
    let mut func_syms: Vec<Option<usize>> = vec![None; mir.funcs.len()];
    let mut global_syms: Vec<Option<usize>> = vec![None; mir.globals.len()];
    let mut extern_syms: Vec<Option<usize>> = vec![None; mir.externs.len()];
    let mut func_text_offset: Vec<Option<u32>> = vec![None; mir.funcs.len()];
    let mut named_syms: NamedSyms = NamedSyms::new();

    // Pre-create every symbol row so functions can reference data
    // defined later in the module.
    for (fi, f) in mir.funcs.iter_enumerated() {
        sym_index(&mut asm, &mut func_syms, fi.index(), true, &f.name, f.local, f.weak, false);
    }
    for (gi, g) in mir.globals.iter_enumerated() {
        sym_index(&mut asm, &mut global_syms, gi.index(), true, &g.name, g.local, g.weak, false);
    }
    for (ei, e) in mir.externs.iter_enumerated() {
        sym_index(&mut asm, &mut extern_syms, ei.index(), false, &e.name, false, e.weak, true);
    }

    // GLCC_NATIVE_CENSUS=1: diagnostic mode — lower every function even
    // when unsupported features panic, emit an int3 stub for the
    // failures, and tally the panic messages. Used to survey which
    // feature gaps a large input hits.
    let census = std::env::var_os("GLCC_NATIVE_CENSUS").is_some();
    let mut census_tally: Vec<(String, u32)> = Vec::new();

    for (fi, f) in mir.funcs.iter_enumerated() {
        let lowered = if census {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lower_function(isa, mir, f)));
            std::panic::set_hook(prev);
            match r {
                Ok(mf) => Some(mf),
                Err(p) => {
                    let msg = p
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "unknown panic".into());
                    if let Some(e) = census_tally.iter_mut().find(|(m, _)| *m == msg) {
                        e.1 += 1;
                    } else {
                        census_tally.push((msg, 1));
                    }
                    None
                }
            }
        } else {
            Some(lower_function(isa, mir, f))
        };
        let Some(mf) = lowered else {
            let sym = func_syms[fi.index()].expect("function symbol");
            let start = asm.sections[0].data.len() as u32;
            func_text_offset[fi.index()] = Some(start);
            asm.sections[0].data.extend_from_slice(&[0xCC]); // int3 stub
            asm.symbols[sym].defined = true;
            asm.symbols[sym].sect = Some(0);
            asm.symbols[sym].value = start as i64;
            continue;
        };
        printer.function(&mf, isa);
        let sym = func_syms[fi.index()].expect("function symbol");
        let start = asm.sections[0].data.len() as u32;
        func_text_offset[fi.index()] = Some(start);
        let mut tb = TextBuilder::new();
        encode_function(isa, &mf, &mut tb);
        resolve_function_refs(
            isa,
            &mut tb,
            &mut asm,
            start,
            &func_text_offset,
            &func_syms,
            &global_syms,
            &extern_syms,
            &mut named_syms,
            mir,
            f.weak,
        );
        asm.sections[0].data.extend_from_slice(&tb.text);
        asm.relocs.append(&mut tb.out_relocs);
        asm.symbols[sym].defined = true;
        asm.symbols[sym].sect = Some(0);
        asm.symbols[sym].value = start as i64;
    }

    let (data_reloc64, data_reloc32) = isa.elf_data_relocs();
    for (gi, g) in mir.globals.iter_enumerated() {
        if g.external {
            // Declarations only: no storage, an undefined symbol.
            sym_index(&mut asm, &mut global_syms, gi.index(), false, &g.name, g.local, g.weak, true);
            continue;
        }
        emit_global(&mut asm, &func_syms, &mut global_syms, &extern_syms, gi.index(), g, data_reloc64, data_reloc32)?;
    }

    emit_init_array(&mut asm, mir, &func_syms, data_reloc64);

    if census {
        let lowered = mir.funcs.len() as u32 - census_tally.iter().map(|(_, n)| *n).sum::<u32>();
        eprintln!("=== GLCC_NATIVE_CENSUS: {lowered} of {} functions lowered; panic tally:", mir.funcs.len());
        for (msg, n) in &census_tally {
            eprintln!("  [{n:5}x] {msg}");
        }
    }

    let object = write_module_elf(&asm, isa.asm_arch())?;
    Ok(Emitted { object, asm: printer.finish() })
}

fn lower_function(isa: &'static dyn TargetIsa, mir: &glcc_mir::Module, f: &glcc_mir::Function) -> MFunction {
    let trace = std::env::var_os("GLCC_NATIVE_TRACE").is_some();
    if trace {
        eprintln!("TRACE: select {}", f.name);
    }
    let mut mf = sel::select_function(isa, mir, f);
    if trace {
        eprintln!("TRACE: ra {}", f.name);
    }
    if std::env::var_os("GLCC_NATIVE_DUMP").is_some() {
        let mut p = crate::printer::Printer::new();
        p.function(&mf, isa);
        eprintln!("=== before RA ==={}", p.finish());
    }
    if isa.needs_ra() {
        ra::allocate(&mut mf, isa);
    }
    if trace {
        eprintln!("TRACE: frame {}", f.name);
    }
    frame::expand(&mut mf, isa);
    if trace {
        eprintln!("TRACE: done {}", f.name);
    }
    if std::env::var_os("GLCC_NATIVE_DUMP").is_some() {
        let mut p = crate::printer::Printer::new();
        p.function(&mf, isa);
        eprintln!("=== after RA ==={}", p.finish());
    }
    mf
}

fn encode_function(isa: &'static dyn TargetIsa, f: &MFunction, tb: &mut TextBuilder) {
    for block in &f.blocks {
        tb.block_offsets.push(tb.text.len() as u32);
        for inst in &block.insts {
            let inst_start = tb.text.len() as u32;
            let mut out = EncOut::new();
            isa.encode_inst(f, inst, &mut out);
            tb.text.extend_from_slice(&out.bytes);
            for (at, target) in out.block_refs {
                tb.block_refs.push((inst_start + at as u32, target));
            }
            for req in out.relocs {
                tb.relocs.push((inst_start + req.at as u32, req));
            }
        }
    }
}

/// Patch intra-function branch fields in the function's own bytes and
/// queue relocations for everything else.
#[allow(clippy::too_many_arguments)]
fn resolve_function_refs(
    isa: &'static dyn TargetIsa,
    tb: &mut TextBuilder,
    asm: &mut AsmModule,
    func_start: u32,
    func_text_offset: &[Option<u32>],
    func_syms: &[Option<usize>],
    global_syms: &[Option<usize>],
    extern_syms: &[Option<usize>],
    named_syms: &mut NamedSyms,
    mir: &glcc_mir::Module,
    func_weak: bool,
) {
    for (field_off, target) in &tb.block_refs {
        let target_off = tb.block_offsets[*target as usize];
        let value = target_off as i64 - *field_off as i64 - 4;
        isa.patch_branch_field(&mut tb.text, *field_off, value);
    }
    for (field_off, req) in &tb.relocs {
        let sym = match &req.sym.target {
            SymTarget::Named(name) => *named_syms.entry(name).or_insert_with(|| {
                asm.symbols.push(Symbol {
                    name: name.to_string(),
                    defined: false,
                    seen: true,
                    extern_: true,
                    global: true,
                    weak: false,
                    sect: None,
                    value: 0,
                });
                asm.symbols.len() - 1
            }),
            SymTarget::Func(f) => {
                // Direct call to a strong definition in this object;
                // weak definitions stay preemptible for interposition.
                // Only branch-style fields (rel32, imm26) patch in
                // place: address-taking fields (aarch64's adrp/add
                // pair) resolve through their own relocations below.
                let branch_field = matches!(req.sym.reloc, RelocKind::PcRel32 | RelocKind::Plt32 | RelocKind::Call26);
                let defined = !mir.funcs[*f].weak && !func_weak && func_text_offset[f.index()].is_some();
                if defined && branch_field {
                    let value = func_text_offset[f.index()].unwrap() as i64 - func_start as i64 - *field_off as i64 - 4;
                    isa.patch_branch_field(&mut tb.text, *field_off, value);
                    continue;
                }
                func_syms[f.index()].expect("referenced function symbol")
            }
            SymTarget::Global(g) => global_syms[g.index()].expect("referenced global symbol"),
            SymTarget::Extern(e) => extern_syms[e.index()].expect("referenced extern symbol"),
        };
        let (r_type, pcrel, size) = isa.elf_reloc(req.sym.reloc);
        tb.out_relocs.push(Reloc {
            section: 0,
            offset: (func_start + *field_off) as i64,
            size,
            pcrel,
            addend: req.addend,
            target: RelocTarget::Symbol(sym),
            r_type,
        });
    }
}

fn sym_index(
    asm: &mut AsmModule,
    table: &mut [Option<usize>],
    index: usize,
    defined: bool,
    name: &str,
    local: bool,
    weak: bool,
    extern_: bool,
) -> usize {
    if let Some(s) = table[index] {
        return s;
    }
    asm.symbols.push(Symbol {
        name: name.to_string(),
        defined,
        seen: true,
        extern_,
        global: !local,
        weak,
        sect: None,
        value: 0,
    });
    table[index] = Some(asm.symbols.len() - 1);
    asm.symbols.len() - 1
}

fn emit_global(
    asm: &mut AsmModule,
    func_syms: &[Option<usize>],
    table: &mut [Option<usize>],
    extern_syms: &[Option<usize>],
    index: usize,
    g: &glcc_mir::GlobalData,
    r_type64: u32,
    r_type32: u32,
) -> Result<(), String> {
    let size = g.ty.size_bytes() as usize;
    let (section, data) = match &g.init {
        None => (ensure_section(asm, ".bss", SectionFlags::Bss), vec![0u8; size]),
        Some(Const::Zero) => (section_for_global(asm, g), vec![0u8; size]),
        Some(Const::Bytes(b)) => (section_for_global(asm, g), b.clone()),
        Some(c) => (section_for_global(asm, g), const_bytes(c, size)),
    };
    let start = asm.sections[section].data.len() as u64;
    asm.sections[section].data.extend_from_slice(&data);

    let sym = sym_index(asm, table, index, true, &g.name, g.local, g.weak, false);
    asm.symbols[sym].defined = true;
    asm.symbols[sym].sect = Some(section);
    asm.symbols[sym].value = start as i64;

    let relocs = g
        .relocs
        .iter()
        .map(|(off, r)| (*off, r, 8u8, r_type64))
        .chain(g.relocs32.iter().map(|(off, r)| (*off, r, 4u8, r_type32)));
    for (off, reloc, rsize, r_type) in relocs {
        let (target, addend) = match reloc {
            DataReloc::Global { target, addend } => (RelocTarget::Symbol(table[target.index()].expect("reloc global symbol")), *addend),
            DataReloc::Func { target } => (RelocTarget::Symbol(func_syms[target.index()].expect("reloc func symbol")), 0),
            DataReloc::FuncExtern { target } => (RelocTarget::Symbol(extern_syms[target.index()].expect("reloc extern symbol")), 0),
            DataReloc::Deferred { name, .. } => {
                return Err(format!("glcc native backend: deferred data relocation {name} is not yet supported"));
            }
        };
        asm.relocs.push(Reloc {
            section,
            offset: (start as i64) + off as i64,
            size: rsize,
            pcrel: false,
            addend,
            target,
            r_type,
        });
    }
    Ok(())
}

fn section_for_global(asm: &mut AsmModule, g: &glcc_mir::GlobalData) -> usize {
    if g.writable {
        ensure_section(asm, ".data", SectionFlags::Data)
    } else {
        ensure_section(asm, ".rodata", SectionFlags::Rodata)
    }
}

fn ensure_section(asm: &mut AsmModule, name: &str, flags: SectionFlags) -> usize {
    if let Some(i) = asm.sections.iter().position(|s| s.name == name) {
        return i;
    }
    asm.sections.push(Section { name: name.to_string(), flags, align: 1, data: Vec::new() });
    asm.sections.len() - 1
}

/// Scalar/aggregate constant image bytes; aggregate-level relocations
/// ride separately in `GlobalData::relocs` over a zeroed image.
fn const_bytes(c: &Const, size: usize) -> Vec<u8> {
    let mut out = vec![0u8; size];
    match c {
        Const::Int { value, .. } => {
            let bytes = value.to_le_bytes();
            let n = bytes.len().min(size);
            out[..n].copy_from_slice(&bytes[..n]);
        }
        Const::Float { bits, .. } => {
            let bytes = bits.to_le_bytes();
            let n = bytes.len().min(size);
            out[..n].copy_from_slice(&bytes[..n]);
        }
        Const::Null | Const::Undef | Const::Zero | Const::Bytes(_) => {}
    }
    out
}

fn emit_init_array(asm: &mut AsmModule, mir: &glcc_mir::Module, func_syms: &[Option<usize>], r_type: u32) {
    if mir.init_funcs.is_empty() {
        return;
    }
    let section = ensure_section(asm, ".init_array", SectionFlags::Data);
    asm.sections[section].align = 8;
    for f in &mir.init_funcs {
        let at = asm.sections[section].data.len() as i64;
        asm.sections[section].data.extend_from_slice(&[0u8; 8]);
        asm.relocs.push(Reloc {
            section,
            offset: at,
            size: 8,
            pcrel: false,
            addend: 0,
            target: RelocTarget::Symbol(func_syms[f.index()].expect("init func symbol")),
            r_type,
        });
    }
}
