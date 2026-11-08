//! x86-64 backend: encoding via the nasm-port tables, flat fixup
//! patching, and rel8 branch relaxation.

use crate::arch::{Backend, Insn};
use crate::assemble::{Fixup, Reloc, Target};
use crate::expr::RValue;
use object::{Architecture, Endianness};

pub struct X86Backend;

impl Backend for X86Backend {
    fn encode(&self, insn: &Insn, ctx: &crate::expr::EvalCtx, _relax_state: u8) -> Result<crate::encoding::Encoded, String> {
        match insn {
            Insn::X86(i) => crate::table::encode(i, ctx),
            _ => Err("not an x86 instruction".into()),
        }
    }

    fn finalize_all(
        &self,
        sections: &mut [crate::assemble::Section],
        _symbols: &mut Vec<crate::assemble::Symbol>,
        fixups: Vec<Fixup>,
        relocs: &mut Vec<Reloc>,
    ) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        for f in &fixups {
            let data = &mut sections[f.section].data;
            if let Err(msgs) = self.finalize_one(data, f, relocs) {
                errors.extend(msgs);
            }
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    fn relax_step(&self, fixup: &Fixup, delta: i64, state: u8) -> Option<u8> {
        if fixup.pcrel && fixup.size == 1 && !(-128..=127).contains(&delta) && state < 1 {
            Some(1)
        } else {
            None
        }
    }

    fn object_arch(&self) -> Architecture {
        Architecture::X86_64
    }

    fn endianness(&self) -> Endianness {
        Endianness::Little
    }

    fn text_fill(&self) -> u8 {
        0x90
    }
}

impl X86Backend {
    fn finalize_one(&self, data: &mut [u8], f: &Fixup, relocs: &mut Vec<Reloc>) -> Result<(), Vec<String>> {
        let pos = f.offset as usize;
        match &f.val {
            RValue::Const(v) => {
                if !fits_signed(*v, f.size) && !fits_unsigned(*v, f.size) {
                    return Err(vec![format!("value {v} does not fit in {} bytes", f.size)]);
                }
                patch(data, pos, *v, f.size);
                Ok(())
            }
            RValue::SectVal { sect: target, off, .. } => {
                if f.pcrel && *target == f.section {
                    let d = off - f.pcrel_base();
                    if f.size == 1 && !(-128..=127).contains(&d) {
                        // Relaxation should have handled this; leave the
                        // placeholder bytes alone.
                        return Ok(());
                    }
                    patch(data, pos, d, f.size);
                    Ok(())
                } else {
                    if f.size == 1 {
                        return Err(vec!["byte-sized relocation is not possible".into()]);
                    }
                    relocs.push(Reloc {
                        section: f.section,
                        offset: f.offset,
                        size: f.size,
                        pcrel: f.pcrel,
                        addend: f.reloc_addend(*off),
                        target: Target::Section(*target),
                        r_type: 0,
                    });
                    Ok(())
                }
            }
            RValue::ExternRef { sym, off } => {
                if f.size < 4 {
                    return Err(vec!["relocation does not fit in field".into()]);
                }
                relocs.push(Reloc {
                    section: f.section,
                    offset: f.offset,
                    size: f.size,
                    pcrel: f.pcrel,
                    addend: f.reloc_addend(*off),
                    target: Target::Symbol(*sym),
                    r_type: 0,
                });
                Ok(())
            }
        }
    }
}

fn fits_signed(v: i64, size: u8) -> bool {
    match size {
        1 => (-128..=127).contains(&v),
        2 => (-0x8000..=0x7fff).contains(&v),
        4 => (-0x8000_0000..=0x7fff_ffff).contains(&v),
        _ => true,
    }
}

fn fits_unsigned(v: i64, size: u8) -> bool {
    match size {
        1 => (0..=255).contains(&v),
        2 => (0..=0xffff).contains(&v),
        4 => (0..=0xffff_ffff).contains(&v),
        _ => true,
    }
}

pub fn patch(data: &mut [u8], at: usize, v: i64, size: u8) {
    for i in 0..size as usize {
        data[at + i] = (v >> (i * 8)) as u8;
    }
}
