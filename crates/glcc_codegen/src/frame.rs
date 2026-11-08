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
    let mut cursor: u32 = 8;
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
    layout(f);
    resolve_frame_refs(f, isa);
    isa.abi().emit_prologue(f);
    isa.abi().insert_epilogues(f);
}

fn align_up(v: u32, align: u32) -> u32 {
    v.div_ceil(align) * align
}
