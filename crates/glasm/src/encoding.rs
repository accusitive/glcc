//! Instruction encoding. The hand-written encoder from milestone 1 was
//! replaced by the table-driven port of nasm's assembler; this module
//! now only defines the shared output types.

use crate::expr::RValue;

#[derive(Debug)]
pub struct FixupReq {
    /// Offset of the field within the encoded instruction bytes.
    pub at: usize,
    pub size: u8,
    pub val: RValue,
    pub pcrel: bool,
}

pub struct Encoded {
    pub bytes: Vec<u8>,
    pub fixups: Vec<FixupReq>,
}

pub fn encode(
    insn: &crate::insn::Insn,
    ctx: &crate::expr::EvalCtx,
    _force_long: bool,
) -> Result<Encoded, String> {
    crate::table::encode(insn, ctx)
}
