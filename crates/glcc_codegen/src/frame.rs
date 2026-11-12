//! Frame layout and ABI frame expansion, run post-allocation when the
//! frame size and callee-saved set are known. The register allocator
//! never sees these instructions: they are emitted with physical
//! registers and fixed stack-pointer adjustments only.

use crate::isa::TargetIsa;
use crate::machine::{MFunction, PReg, Reg};

/// Assign offsets to frame slots (user slots, then the allocator's
/// spill slots, which `ra.rs` already appended) and fix the total frame
/// size. A slot's `offset` is its top edge below the frame pointer: it
/// spans `[rbp - offset, rbp - offset + size)`, so successive slots
/// stack downward and never overlap. The target's encoder turns
/// offsets into `-offset` displacements.
pub fn layout(f: &mut MFunction) {
    // Slots start below the callee-saved registers the prologue pushes
    // directly under the frame pointer.
    let mut cursor: u32 = 8 + 8 * f.frame.saved_callee;
    for slot in &mut f.frame.slots {
        let base = align_up(cursor, slot.align.max(1));
        slot.offset = base + slot.size;
        cursor = base + slot.size;
    }
    f.frame.frame_size = align_up(cursor, 16);
}

/// Frame-index bases become frame-pointer-relative memory operands
/// (address = rbp - offset + disp).
fn resolve_frame_refs(f: &mut MFunction, isa: &'static dyn TargetIsa) {
    let fp = isa.reginfo().frame_ptr;
    let slots = &f.frame.slots;
    for block in &mut f.blocks {
        for inst in &mut block.insts {
            for op in inst.ops.iter_mut() {
                if let crate::machine::Operand::Mem(m) = op {
                    if let Some(Reg::FI(slot)) = m.base {
                        m.base = Some(Reg::P(fp));
                        m.disp -= slots[slot as usize].offset as i64;
                    }
                }
            }
        }
    }
}

/// Insert the target's prologue and epilogues.
pub fn expand(f: &mut MFunction, isa: &'static dyn TargetIsa) {
    f.frame.saved_callee = used_callee_saved(f, isa);
    layout(f);
    resolve_frame_refs(f, isa);
    isa.abi().emit_prologue(f);
    isa.abi().insert_epilogues(f);
}

fn align_up(v: u32, align: u32) -> u32 {
    v.div_ceil(align) * align
}

/// How many callee-saved registers this (post-RA) function references:
/// the prologue pushes exactly those under the frame pointer, and the
/// slot layout must start below them.
pub fn used_callee_saved(f: &MFunction, isa: &'static dyn TargetIsa) -> u32 {
    let reginfo = isa.reginfo();
    let mut used = std::collections::HashSet::<(u8, u8)>::new();
    for b in &f.blocks {
        for inst in &b.insts {
            if let Some(blob) = &inst.asm {
                for &p in &blob.clobbers {
                    if reginfo.info(p).callee_saved {
                        used.insert((p.hw, p.class as u8));
                    }
                }
            }
            for op in &inst.ops {
                match op {
                    crate::machine::Operand::Reg(Reg::P(p)) | crate::machine::Operand::FixedReg(Reg::P(p), _) => {
                        if reginfo.info(*p).callee_saved {
                            used.insert((p.hw, p.class as u8));
                        }
                    }
                    crate::machine::Operand::Mem(m) => {
                        for r in [m.base, m.index] {
                            if let Some(Reg::P(p)) = r {
                                if reginfo.info(p).callee_saved {
                                    used.insert((p.hw, p.class as u8));
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    used.len() as u32
}
