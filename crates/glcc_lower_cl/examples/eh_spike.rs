//! S0 spike from docs/eh-plan.md: prove Itanium zero-cost unwinding through
//! Cranelift-compiled frames end to end, without MIR.
//!
//! Hand-writes two functions in CLIF, compiles them with cranelift-object
//! (with its own `.eh_frame` suppressed), and appends the unwind metadata
//! ourselves — the design this spike exists to validate:
//!
//! - a CIE carrying the `__gxx_personality_v0` augmentation (gimli's write
//!   API supports `personality`/`lsda_encoding` natively, so no byte surgery
//!   on Cranelift's section is needed),
//! - FDEs built from `CompiledCode::create_unwind_info` (the same CFI
//!   cranelift-object would have emitted),
//! - a hand-encoded `.gcc_except_table` LSDA derived from the compiled
//!   `call_sites` (call-site ranges + landing-pad offsets) and the clause
//!   list (type table: `int`, i.e. `_ZTIi`).
//!
//! The object is linked against a C++ main (thrower in a libstdc++ frame)
//! and executed; the assertions at the bottom are the spike's acceptance
//! criteria. Run: `cargo run -p glcc_lower_cl --example eh_spike`

use cranelift_codegen::ir::{
    types, AbiParam, BlockArg, BlockCall, ExceptionTableData, ExceptionTableItem, Function,
    InstBuilder, Signature, UserFuncName,
};
use cranelift_codegen::isa::unwind::UnwindInfo;
use cranelift_codegen::isa::OwnedTargetIsa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::{Context, FinalizedMachExceptionHandler};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use gimli::constants::{self, DW_EH_PE_absptr, DW_EH_PE_sdata4, DW_EH_PE_sdata8, DW_EH_PE_udata2,
    DW_EH_PE_udata4, DW_EH_PE_udata8};
use gimli::write::{
    Address, CallFrameInstruction, CommonInformationEntry, EhFrame, EndianVec,
    FrameDescriptionEntry, FrameTable, RelocateWriter, Relocation, RelocationTarget,
};
use gimli::{Encoding, Format, Register, RunTimeEndian};
use object::write::{
    Object, Relocation as ObjRelocation, StandardSection, Symbol as ObjSymbol, SymbolSection,
};
use object::{
    RelocationEncoding, RelocationFlags, RelocationKind, SectionKind, SymbolFlags, SymbolKind,
    SymbolScope,
};
use std::fs;
use std::process::Command;

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

/// (code alignment, DWARF return-address register, DWARF stack-pointer
/// register, CFA offset relative to SP, return-address offset from CFA) for
/// the host ISA, mirroring each backend's `create_cie`.
fn cie_params(triple: &target_lexicon::Triple) -> (u8, Register, Register, i32, Option<i32>) {
    use target_lexicon::Architecture;
    match triple.architecture {
        Architecture::Aarch64(_) => (4, Register(30), Register(31), 0, None),
        Architecture::X86_64 => (1, Register(16), Register(7), 8, Some(-8)),
        arch => panic!("unsupported spike architecture {arch}"),
    }
}

fn sig(
    isa: &OwnedTargetIsa,
    params: &[cranelift_codegen::ir::Type],
    rets: &[cranelift_codegen::ir::Type],
) -> Signature {
    Signature {
        params: params.iter().map(|t| AbiParam::new(*t)).collect(),
        returns: rets.iter().map(|t| AbiParam::new(*t)).collect(),
        call_conv: isa.default_call_conv(),
    }
}

/// `catcher(x) -> i32`: `try_call thrower(x)` with a landing pad that returns
/// the exception selector.
fn build_catcher(module: &mut ObjectModule, isa: &OwnedTargetIsa, thrower_id: FuncId) -> Function {
    let thrower_sig = sig(isa, &[types::I32], &[types::I32]);
    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        sig(isa, &[types::I32], &[types::I32]),
    );
    let thrower = module.declare_func_in_func(thrower_id, &mut func);
    let thrower_sigref = func.import_signature(thrower_sig);

    let payload =
        isa.default_call_conv().exception_payload_types(isa.pointer_type()).to_vec();
    let mut fb_ctx = FunctionBuilderContext::new();
    let mut fb = FunctionBuilder::new(&mut func, &mut fb_ctx);

    let b0 = fb.create_block();
    let b_norm = fb.create_block();
    let b_lpad = fb.create_block();
    for ty in &payload {
        fb.append_block_param(b_lpad, *ty);
    }
    // Payload types are two word-sized values (pointer, selector).
    let sel = fb.func.dfg.block_params(b_lpad)[1];
    fb.switch_to_block(b0);
    let x = fb.append_block_param(b0, types::I32);

    let normal = BlockCall::new(b_norm, [BlockArg::TryCallRet(0)], &mut fb.func.dfg.value_lists);
    let unwind = BlockCall::new(
        b_lpad,
        payload.iter().enumerate().map(|(i, _)| BlockArg::TryCallExn(i as u32)),
        &mut fb.func.dfg.value_lists,
    );
    let table = fb.func.dfg.exception_tables.push(ExceptionTableData::new(
        thrower_sigref,
        normal,
        [ExceptionTableItem::Default(unwind)],
    ));
    fb.ins().try_call(thrower, &[x], table);

    fb.switch_to_block(b_norm);
    let v = fb.append_block_param(b_norm, types::I32);
    fb.ins().return_(&[v]);

    fb.switch_to_block(b_lpad);
    let sel32 = if payload[1] == types::I64 {
        fb.ins().ireduce(types::I32, sel)
    } else {
        sel
    };
    fb.ins().return_(&[sel32]);

    fb.seal_all_blocks();
    fb.finalize(isa.frontend_config());
    func
}

/// `passthru(x) -> i32`: plain call; its frame must be unwindable *through*
/// (thrower throws below it; main catches above it).
fn build_passthru(module: &mut ObjectModule, isa: &OwnedTargetIsa, thrower_id: FuncId) -> Function {
    let thrower_sig = sig(isa, &[types::I32], &[types::I32]);
    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 1),
        sig(isa, &[types::I32], &[types::I32]),
    );
    let thrower = module.declare_func_in_func(thrower_id, &mut func);
    let _ = func.import_signature(thrower_sig);

    let mut fb_ctx = FunctionBuilderContext::new();
    let mut fb = FunctionBuilder::new(&mut func, &mut fb_ctx);
    let b0 = fb.create_block();
    fb.switch_to_block(b0);
    let x = fb.append_block_param(b0, types::I32);
    let call = fb.ins().call(thrower, &[x]);
    let v = fb.func.dfg.inst_results(call)[0];
    fb.ins().return_(&[v]);
    fb.seal_all_blocks();
    fb.finalize(isa.frontend_config());
    func
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

/// `.gcc_except_table` LSDA: one catch clause (`int`), call-site ranges from
/// the compiled `ret_addr`s (the conservative `[prev, this)` encoding),
/// landing pad from the compiled handler record. Returns the bytes and the
/// offset of the TType slot within them (for the `_ZTIi` relocation).
fn build_lsda(ret_addrs: &[u32], lpad: u32) -> (Vec<u8>, usize) {
    let mut b = Vec::new();
    b.push(0xff); // LPStart encoding: DW_EH_PE_omit (LPStart = function start)
    b.push(0x1b); // TType encoding: DW_EH_PE_pcrel|sdata4
    let ttype_off_pos = b.len();
    b.push(0); // uleb128 TType offset, patched below
    b.push(0x01); // call-site encoding: DW_EH_PE_uleb128 (plain offsets)

    let mut cs = Vec::new();
    let mut prev = 0;
    for &ret in ret_addrs {
        write_uleb(&mut cs, prev as u64);
        write_uleb(&mut cs, (ret - prev) as u64);
        write_uleb(&mut cs, lpad as u64);
        write_uleb(&mut cs, 1); // action
        prev = ret;
    }
    write_uleb(&mut b, cs.len() as u64);
    b.extend_from_slice(&cs);

    // Action table: filter 1 (type table slot 1), no next action.
    write_sleb(&mut b, 1);
    write_sleb(&mut b, 0);

    // Type table, written backwards: the slot for the highest filter first,
    // ending at the TType base. The personality reads entry N at
    // (base - N*size), so the uleb must point just past the last slot.
    // One entry: _ZTIi (patched by relocation).
    let ttype_slot_pos = b.len();
    b.extend_from_slice(&[0; 4]);
    let ttype_off = ttype_slot_pos + 4 - (ttype_off_pos + 1);
    b[ttype_off_pos] = u8::try_from(ttype_off).expect("TType offset must fit one uleb byte");
    (b, ttype_slot_pos)
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

fn main() {
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    let flags = settings::Flags::new(flag_builder);
    let isa = cranelift_native::builder()
        .expect("host CPU features")
        .finish(flags)
        .expect("host target");
    let call_conv = isa.default_call_conv();
    assert!(
        call_conv.supports_exceptions(),
        "host call conv {call_conv} does not support exceptions"
    );
    let payload_len = call_conv.exception_payload_types(isa.pointer_type()).len();
    assert_eq!(payload_len, 2, "expected two exception payload values");

    let mut builder = ObjectBuilder::new(isa.clone(), "eh_spike".to_string(), default_libcall_names())
        .unwrap();
    builder.unwind_info(false); // we emit .eh_frame ourselves below
    let mut module = ObjectModule::new(builder);

    let thrower_sig = sig(&isa, &[types::I32], &[types::I32]);
    let thrower_id =
        module.declare_function("thrower", Linkage::Import, &thrower_sig).unwrap();

    struct CompiledFunc {
        id: FuncId,
        eh: bool,
        ctx: Context,
        /// (return-address offset, landing-pad offset) per call site.
        sites: Vec<(u32, u32)>,
    }
    let mut compiled: Vec<CompiledFunc> = Vec::new();
    for (name, func, eh) in [
        ("catcher", build_catcher(&mut module, &isa, thrower_id), true),
        ("passthru", build_passthru(&mut module, &isa, thrower_id), false),
    ] {
        let id =
            module.declare_function(name, Linkage::Export, &func.signature).unwrap();
        let mut ctx = Context::for_function(func);
        module.define_function(id, &mut ctx).unwrap();
        let sites = if eh {
            ctx.compiled_code()
                .unwrap()
                .buffer
                .call_sites()
                .map(|cs| {
                    let lpad = cs
                        .exception_handlers
                        .iter()
                        .find_map(|h| match h {
                            FinalizedMachExceptionHandler::Default(off) => Some(*off),
                            _ => None,
                        })
                        .expect("invoke call site without a Default handler");
                    (cs.ret_addr, lpad)
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        if eh {
            assert!(!sites.is_empty(), "{name}: EH function without call sites");
        }
        compiled.push(CompiledFunc { id, eh, ctx, sites });
    }

    let product = module.finish();
    let func_syms: Vec<object::write::SymbolId> =
        compiled.iter().map(|f| product.function_symbol(f.id)).collect();
    let mut obj = product.object;
    assert_eq!(
        obj.format(),
        object::BinaryFormat::Elf,
        "spike assumes ELF unwind metadata"
    );
    let endian = match isa.triple().endianness().unwrap() {
        target_lexicon::Endianness::Little => RunTimeEndian::Little,
        target_lexicon::Endianness::Big => RunTimeEndian::Big,
    };

    // .gcc_except_table: one LSDA per EH function.
    let lsda_id = obj.add_section(Vec::new(), b".gcc_except_table".to_vec(), SectionKind::ReadOnlyData);
    struct Lsda {
        sym: object::write::SymbolId,
        ttype_slot_offset: u64,
    }
    let mut lsdas: Vec<Lsda> = Vec::new();
    for f in &compiled {
        if !f.eh {
            continue;
        }
        let rets: Vec<u32> = f.sites.iter().map(|(r, _)| *r).collect();
        let (bytes, ttype_slot_pos) = build_lsda(&rets, f.sites[0].1);
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
        lsdas.push(Lsda { sym, ttype_slot_offset: off + ttype_slot_pos as u64 });
    }
    assert_eq!(lsdas.len(), 1, "spike LSDA relocation assumes a single LSDA");

    // .eh_frame: plain CIE for non-EH functions, personality CIE for EH ones.
    let eh_frame_id = obj.section_id(StandardSection::EhFrame);
    let mut frame_table = FrameTable::default();
    // Index space for gimli relocations: gimli addresses reference these
    // indices, resolved to object SymbolIds after serialization.
    let mut symbols: Vec<object::write::SymbolId> = Vec::new();

    let personality_idx = symbols.len() as u32;
    symbols.push(undefined_symbol(&mut obj, "__gxx_personality_v0"));

    let (code_align, ra_reg, sp_reg, cfa_sp_off, ra_offset) = cie_params(&isa.triple());
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
            Address::Symbol { symbol: personality_idx as usize, addend: 0 },
        ));
        cie.lsda_encoding = Some(constants::DW_EH_PE_pcrel | constants::DW_EH_PE_sdata4);
        cie.fde_address_encoding = constants::DW_EH_PE_pcrel | constants::DW_EH_PE_sdata4;
        cie
    };
    let plain_cie_id = frame_table.add_cie(plain_cie);
    let eh_cie_id = frame_table.add_cie(eh_cie);

    for (i, f) in compiled.iter().enumerate() {
        let UnwindInfo::SystemV(sysv) = f
            .ctx
            .compiled_code()
            .unwrap()
            .create_unwind_info(&*isa)
            .unwrap()
            .expect("no unwind info for SystemV host")
        else {
            panic!("expected SystemV unwind info");
        };
        let func_idx = symbols.len() as u32;
        symbols.push(func_syms[i]);
        let mut fde: FrameDescriptionEntry =
            sysv.to_fde(Address::Symbol { symbol: func_idx as usize, addend: 0 });
        if f.eh {
            let lsda_idx = symbols.len() as u32;
            symbols.push(lsdas[i].sym);
            fde.lsda = Some(Address::Symbol { symbol: lsda_idx as usize, addend: 0 });
            frame_table.add_fde(eh_cie_id, fde);
        } else {
            frame_table.add_fde(plain_cie_id, fde);
        }
    }

    let frame_section = FrameSection { writer: EndianVec::new(endian), relocations: Vec::new() };
    let frame_section = {
        let mut eh_frame = EhFrame(frame_section);
        frame_table.write_eh_frame(&mut eh_frame).expect("write .eh_frame");
        eh_frame.0
    };
    obj.append_section_data(eh_frame_id, &frame_section.writer.into_vec(), 8);
    for reloc in &frame_section.relocations {
        let symbol = match reloc.target {
            RelocationTarget::Symbol(idx) => symbols[idx as usize],
            other => panic!("unexpected .eh_frame relocation target {other:?}"),
        };
        let (kind, bit_size) = eh_pe_flags(reloc.eh_pe, reloc.size);
        obj.add_relocation(
            eh_frame_id,
            ObjRelocation {
                offset: reloc.offset as u64,
                symbol,
                addend: reloc.addend,
                flags: RelocationFlags::Generic {
                    kind,
                    encoding: RelocationEncoding::Generic,
                    size: bit_size,
                },
            },
        )
        .unwrap();
    }

    // LSDA type-table slot: pcrel sdata4 against _ZTIi (copy-relocated by the
    // non-PIE link).
    let ztii = undefined_symbol(&mut obj, "_ZTIi");
    obj.add_relocation(
        lsda_id,
        ObjRelocation {
            offset: lsdas[0].ttype_slot_offset,
            symbol: ztii,
            addend: 0,
            flags: RelocationFlags::Generic {
                kind: RelocationKind::Relative,
                encoding: RelocationEncoding::Generic,
                size: 32,
            },
        },
    )
    .unwrap();

    let mut bytes = Vec::new();
    obj.emit(&mut bytes).unwrap();

    // Link with a throwing C++ main and run it.
    let out_dir = std::env::temp_dir().join("glcc-eh-spike");
    fs::create_dir_all(&out_dir).unwrap();
    fs::write(out_dir.join("spike.o"), &bytes).unwrap();
    fs::write(out_dir.join("main.cpp"), MAIN_CPP).unwrap();
    let run = |cmd: &mut Command| {
        let out = cmd.output().expect("spawn");
        if !out.status.success() {
            panic!("{cmd:?} failed:\n{}", String::from_utf8_lossy(&out.stderr));
        }
        out
    };
    run(Command::new("g++")
        .arg("-no-pie")
        .arg("main.cpp")
        .arg("spike.o")
        .arg("-o")
        .arg("spike")
        .current_dir(&out_dir));
    let out = run(Command::new("./spike").current_dir(&out_dir));
    let stdout = String::from_utf8_lossy(&out.stdout);
    println!("{stdout}");
    for expected in ["catcher normal: 0\n", "catcher catch: 1\n", "passthru catch: 42\n"] {
        assert!(
            stdout.contains(expected),
            "missing {expected:?} in spike output:\n{stdout}"
        );
    }
    println!("eh_spike: all checks passed");
}

const MAIN_CPP: &str = r#"
#include <cstdio>

extern "C" int catcher(int x);
extern "C" int passthru(int x);

extern "C" int thrower(int x) {
    if (x) throw 42;
    return x;
}

int main() {
    std::printf("catcher normal: %d\n", catcher(0));
    std::printf("catcher catch: %d\n", catcher(1));
    try {
        passthru(1);
        std::printf("passthru: NOT CAUGHT\n");
    } catch (int e) {
        std::printf("passthru catch: %d\n", e);
    }
    return 0;
}
"#;
