//! Shared encoder output types: the bytes of one instruction and the
//! fixups the finalizer must resolve into them.

use crate::expr::RValue;

/// How a resolved fixup value lands in the encoded instruction. x86
/// displacements/immediates are flat little-endian fields; fixed-width
/// ISAs spread fields across the instruction word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldLayout {
    /// Plain little-endian field, `size` bytes wide.
    Flat,
    // RISC-V.
    /// B-type branch: imm[12] bit 31, imm[10:5] bits 30..25,
    /// imm[4:1] bits 11..8, imm[11] bit 7.
    RvBranch,
    /// J-type jal: imm[20] bit 31, imm[10:1] bits 30..21,
    /// imm[11] bit 20, imm[19:12] bits 19..12.
    RvJal,
    /// Upper 20 bits (imm[31:12]) at bits 31..12 (lui/auipc/%hi).
    RvHi20,
    /// I-type immediate: imm[11:0] at bits 11..0 (addi/jalr/loads).
    RvLo12I,
    /// S-type store immediate spread like B-type without the sign bit:
    /// imm[11:5] bits 31..25, imm[4:0] bits 11..7.
    RvLo12S,
    /// U-type with PCREL_HI20 semantics (auipc).
    RvPcrelHi20,
    /// %pcrel_lo on an I-type immediate; pairs with the PCREL_HI20
    /// fixup at the marked auipc address.
    RvPcrelLo12I,
    /// %pcrel_lo on an S-type immediate.
    RvPcrelLo12S,
    /// R_RISCV_CALL_PLT covering both words of auipc+jalr.
    RvCall,
    // AArch64.
    /// imm26 at bits 25..0 (b/bl), signed. b vs bl (JUMP26 vs CALL26)
    /// and the other sub-forms are recovered from the instruction word
    /// during finalization.
    A64Imm26,
    /// imm19 at bits 23..5 (b.cond/cbz/cbnz), signed.
    A64Imm19,
    /// imm14 at bits 18..5 (tbz/tbnz), signed.
    A64Imm14,
    /// imm21 spread for adr: imm[1:0] bits 30..29 (immlo),
    /// imm[20:2] bits 23..5 (immhi).
    A64Imm21,
    /// Like A64Imm21 but the value is a page delta: the field holds
    /// ((S+A) >> 12) - (PC >> 12) (adrp).
    A64AdrpImm21,
    /// imm12 at bits 21..10 (add/sub imm with :lo12:).
    A64Imm12,
    /// 12-bit unsigned load/store offset, scaled by access size before
    /// the value lands at bits 21..10 (:lo12: loads/stores).
    A64LdstImm12,
    /// imm16 at bits 20..5 (movz/movk/movn), with an explicit shift
    /// already encoded in the word.
    A64Imm16,
    /// Like A64AdrpImm21 but relocating against the GOT entry
    /// (`adrp x, :got:sym`).
    A64GotPage,
    /// Scaled load/store field relocating against a GOT entry
    /// (`ldr x, [x, :got_lo12:sym]`).
    A64GotLo12,
}

pub struct FixupReq {
    /// Offset of the field within the encoded instruction bytes.
    pub at: usize,
    pub size: u8,
    pub val: RValue,
    pub pcrel: bool,
    /// PC-relative base, measured from the start of the instruction
    /// (x86: end of the instruction; RISC-V branches: its own start).
    pub rel_base: u32,
    pub layout: FieldLayout,
}

impl FixupReq {
    pub fn flat(at: usize, size: u8, val: RValue, pcrel: bool, rel_base: u32) -> FixupReq {
        FixupReq {
            at,
            size,
            val,
            pcrel,
            rel_base,
            layout: FieldLayout::Flat,
        }
    }
}

pub struct Encoded {
    pub bytes: Vec<u8>,
    pub fixups: Vec<FixupReq>,
}
