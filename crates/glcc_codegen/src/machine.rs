//! Machine-level IR: the shape every target lowers MIR into and encodes
//! from. The IR is deliberately small — one instruction record with an
//! opcode id plus an operand list — because everything downstream
//! (register allocation, scheduling, encoding) is driven by per-target
//! tables keyed on the opcode id, not by IR variants.
//!
//! Register values use the regalloc2 vocabulary: virtual registers
//! ([`VReg`]) before allocation, physical registers ([`PReg`]) after.
//! Exotic targets that do not allocate registers (stack machines, VMs)
//! still use `VReg`s as local slots and skip the RA pass; nothing in the
//! pipeline below selection assumes physical registers exist.

use glcc_mir::{Extern, Func, Global};
use smallvec::SmallVec;

/// Identifies a machine instruction within a target's opcode table. The
/// framework owns a reserved range for its pseudo instructions; target
/// opcodes start above it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct OpId(pub u32);

impl OpId {
    /// Framework pseudo-instructions, shared by every target. These
    /// never encode directly; targets expand them against their own
    /// register and frame models.
    pub const MOVE: OpId = OpId(0);
    pub const SPILL: OpId = OpId(1);
    pub const RELOAD: OpId = OpId(2);
    /// Return from the function; its register operand (when present)
    /// is constrained to the ABI return register.
    pub const RET: OpId = OpId(3);
    /// Unconditional jump, used by the framework's critical-edge
    /// splitting; targets encode it as a plain jump.
    pub const JMP: OpId = OpId(4);
    /// Trap (unreachable); ends a block like [`OpId::RET`].
    pub const UNREACHABLE: OpId = OpId(5);
    /// Inline asm block; the template and operand protocol live in
    /// [`MInst::asm`]. Targets render it post-RA with the assigned
    /// physical registers and splice the assembled bytes into `.text`.
    pub const ASM: OpId = OpId(6);

    pub fn is_pseudo(self) -> bool {
        self.0 < 16
    }
}

/// Register class of a value. Targets with more classes extend the
/// per-target register tables, not this enum: it names only what the
/// framework itself must distinguish.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RegClass {
    Gpr,
    Vec,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VReg(pub u32);

/// A physical register. `hw` is the target's hardware encoding number
/// (x86-64: the ModRM `reg` field value, 0..16), not a dense index.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PReg {
    pub hw: u8,
    pub class: RegClass,
}

impl PReg {
    pub const fn gpr(hw: u8) -> Self {
        PReg { hw, class: RegClass::Gpr }
    }
    pub const fn vec(hw: u8) -> Self {
        PReg { hw, class: RegClass::Vec }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reg {
    /// Virtual register (pre-allocation).
    V(VReg),
    /// Physical register (post-allocation, or ABI-fixed).
    P(PReg),
    /// A frame slot (alloca); resolved to a frame-pointer-relative
    /// address post-RA. Only meaningful as a memory base.
    FI(u32),
    /// A register-allocator spill slot (post-allocation).
    Spill(u32),
}

impl Reg {
    pub fn vreg(self) -> Option<VReg> {
        match self {
            Reg::V(v) => Some(v),
            _ => None,
        }
    }

    pub fn preg(self) -> Option<PReg> {
        match self {
            Reg::P(p) => Some(p),
            _ => None,
        }
    }
}

/// Operand width in bits, carried per instruction rather than per
/// opcode variant so one table row covers a whole width family. Ignored
/// by widthless opcodes (`call`, `ret`, branches).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Width {
    W8,
    W16,
    W32,
    W64,
}

impl Width {
    pub fn bytes(self) -> u32 {
        match self {
            Width::W8 => 1,
            Width::W16 => 2,
            Width::W32 => 4,
            Width::W64 => 8,
        }
    }
}

/// What a symbol reference points at. Resolved to an object-file symbol
/// at emission.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SymTarget {
    Func(Func),
    Extern(Extern),
    Global(Global),
    /// A named symbol outside the module's tables (runtime libcalls
    /// like `memcpy`); resolved to an extern at emission.
    Named(&'static str),
}

/// Relocation shape of a symbol field. Targets translate these to
/// format-specific types at emission.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RelocKind {
    Abs64,
    /// 32-bit absolute field (`.long sym - .` without pcrel... plain
    /// `R_X86_64_32` data references).
    Abs32,
    /// 32-bit RIP-relative field (x86-64 `lea`, direct calls want
    /// [`RelocKind::Plt32`]).
    PcRel32,
    /// 32-bit RIP-relative call/branch to a possibly-interposed symbol.
    Plt32,
    /// AArch64 `adrp` field: page-of(S+A) minus page-of(field), written
    /// into the split immhi:immlo bits.
    PagePc,
    /// AArch64 low-12 addend: (S+A) & 0xFFF, ORed into the instruction's
    /// imm12 field (the `add` of an adrp/add pair).
    Lo12,
    /// AArch64 26-bit PC-relative call/branch field (b/bl), in units of
    /// instructions.
    Call26,
    /// AArch64 GOT-indirect pair: page-of(GOT entry) for the `adrp`,
    /// then a 64-bit load of the entry.
    GotPage,
    /// AArch64 GOT-indirect pair: low 12 bits of the GOT entry address
    /// in the load's offset field.
    GotLo12,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SymOperand {
    pub target: SymTarget,
    pub reloc: RelocKind,
    pub addend: i64,
}

#[derive(Clone, PartialEq, Debug)]
pub struct MemOperand {
    /// `None` base means RIP-relative on x86-64 and PC-relative base on
    /// other ISAs; `sym` then names the referenced symbol.
    pub base: Option<Reg>,
    pub index: Option<Reg>,
    /// log2 of the stride; an index with scale 0 is meaningless and
    /// unused.
    pub scale: u8,
    pub disp: i64,
    pub sym: Option<SymOperand>,
}

#[derive(Clone, PartialEq, Debug)]
pub enum Operand {
    Reg(Reg),
    Imm(i64),
    FrameIndex(u32),
    Mem(MemOperand),
    Sym(SymOperand),
    /// A virtual register pinned to a physical register for this
    /// instruction (call arguments/results): the allocator inserts the
    /// moves, resolving overlaps between fixed targets.
    FixedReg(Reg, PReg),
}

impl Operand {
    pub fn reg(self) -> Option<Reg> {
        match self {
            Operand::Reg(r) => Some(r),
            _ => None,
        }
    }

    pub fn imm(self) -> Option<i64> {
        match self {
            Operand::Imm(i) => Some(i),
            _ => None,
        }
    }

    /// The virtual register of a register-ish operand.
    pub fn vreg(self) -> Option<VReg> {
        match self {
            Operand::Reg(Reg::V(v)) | Operand::FixedReg(Reg::V(v), _) => Some(v),
            _ => None,
        }
    }
}

/// Control-flow payload of a block-ending instruction. Operands carry
/// the condition/inputs the encoding needs; this carries where control
/// goes and the per-edge arguments (vregs of the successor's block
/// params).
#[derive(Clone, Debug)]
pub struct BranchInfo {
    pub targets: SmallVec<[BlockId; 2]>,
    pub args: Vec<SmallVec<[VReg; 4]>>,
}

#[derive(Clone, Debug)]
pub struct MInst {
    pub op: OpId,
    pub width: Width,
    pub ops: SmallVec<[Operand; 4]>,
    /// For call instructions: the ABI register of each register-class
    /// argument, in operand order. The argument moves themselves are
    /// inserted post-RA (see `ra::allocate`).
    pub call_args: Option<SmallVec<[PReg; 8]>>,
    /// For [`OpId::ASM`]: the template and its operand protocol.
    pub asm: Option<Box<AsmBlob>>,
    pub branch: Option<BranchInfo>,
}

/// One inline-asm operand slot, in GCC numbering order (`%0`, `%1`, ...).
/// RA operands for the slot live in `MInst.ops` at the same index for
/// Reg/Mem/Sel slots (Imm slots carry no RA operand).
#[derive(Clone, Debug)]
pub struct AsmSlot {
    pub kind: AsmSlotKind,
    pub width: Width,
    /// Reads the operand register before the template (inputs, ties,
    /// `+` outputs).
    pub is_use: bool,
    /// Written by the template (outputs, the goto selector).
    pub is_def: bool,
    /// An output written before the inputs are consumed (`&`); the
    /// allocator must not overlap it with an input register.
    pub early: bool,
    /// A named-constraint pin (`"a"` is rax).
    pub pin: Option<PReg>,
    /// A tied input: no RA operand of its own; it renders the target
    /// slot's register, whose def carries the `Reuse`/`Fixed` tie.
    pub rides: Option<usize>,
    /// The `[name]` spelling, for `%[name]` template references.
    pub name: Option<String>,
}

#[derive(Clone, Copy, Debug)]
pub enum AsmSlotKind {
    /// A register operand rendered as the width-named AT&T register
    /// (`%al`, `%ax`, `%eax`, `%rax` per `width`/`modifier`).
    Reg,
    /// A memory operand: the RA operand holds the address; rendered
    /// as `(%reg)`.
    Mem,
    /// An immediate rendered with target syntax (`$N` on x86-64).
    Imm(i64),
    /// The `asm goto` selector, referenced via `%l[name]` stubs.
    Sel,
}

/// A relocation the assembled asm fragment needs: the glasm symbol
/// name, resolved against the module's tables at selection time.
#[derive(Clone, Debug)]
pub struct AsmSymReloc {
    /// Byte offset within the assembled fragment.
    pub offset: u32,
    pub size: u8,
    pub pcrel: bool,
    pub addend: i64,
    /// `Some((section, offset))`: targets a fragment-local label at
    /// that position; the caller synthesizes a local symbol.
    pub local: Option<(String, i64)>,
    pub sym: String,
}

/// A non-`.text` section an inline-asm block wrote, with its symbol
/// relocations already resolved to module targets.
#[derive(Clone, Debug)]
pub struct AsmSideSection {
    pub name: String,
    pub write: bool,
    pub exec: bool,
    pub nobits: bool,
    pub data: Vec<u8>,
    pub relocs: Vec<(AsmSymReloc, SymOperand)>,
}

pub struct AsmBlob {
    /// The decoded template (frontend escapes already processed).
    pub template: String,
    pub slots: Vec<AsmSlot>,
    /// Registers the template may clobber beyond its operand defs:
    /// named clobbers plus literal register mentions detected in the
    /// template. Canonical (width-max) pregs.
    pub clobbers: Vec<PReg>,
    /// `asm goto` label names, in selector order.
    pub labels: Vec<String>,
    /// Per-site number substituted for `%=`.
    pub unique: u32,
    pub is_basic: bool,
    pub is_goto: bool,
    /// Symbol relocations the fragment needs, resolved to module
    /// targets (empty when the template references no symbols).
    pub sym_relocs: Vec<(AsmSymReloc, SymOperand)>,
    /// Side sections the fragment's `.pushsection` wrote.
    pub side_sections: Vec<AsmSideSection>,
    /// The enclosing function's name: fragment-local label relocations
    /// retarget to it (the fragment's text lands inside its body).
    pub func_name: String,
}

impl Clone for AsmBlob {
    fn clone(&self) -> AsmBlob {
        AsmBlob {
            template: self.template.clone(),
            slots: self.slots.clone(),
            clobbers: self.clobbers.clone(),
            labels: self.labels.clone(),
            unique: self.unique,
            is_basic: self.is_basic,
            is_goto: self.is_goto,
            sym_relocs: self.sym_relocs.clone(),
            side_sections: self.side_sections.clone(),
            func_name: self.func_name.clone(),
        }
    }
}

impl std::fmt::Debug for AsmBlob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsmBlob").field("template", &self.template).field("func", &self.func_name).finish()
    }
}

impl AsmBlob {
    /// How many RA operand entries a slot contributes to `MInst.ops`:
    /// a read-modify-write slot (use+def) has two — the value read
    /// before the template and the output it defines (tied by `Reuse`
    /// or a shared pin); other Reg/Mem/Sel slots have one; Imm and
    /// tied-input slots have none.
    pub fn ops_of(&self, slot: &AsmSlot) -> usize {
        match slot.kind {
            AsmSlotKind::Imm(_) => 0,
            _ if slot.rides.is_some() => 0,
            _ if slot.is_use && slot.is_def => 2,
            _ => 1,
        }
    }
}

impl MInst {
    pub fn new(op: OpId, width: Width, ops: impl Into<SmallVec<[Operand; 4]>>) -> Self {
        MInst { op, width, ops: ops.into(), branch: None, call_args: None, asm: None }
    }
}

pub type BlockId = u32;

#[derive(Debug, Default)]
pub struct MBlock {
    /// Value edge inputs, in order. Every edge into the block passes the
    /// same count; the register allocator resolves the moves.
    pub params: Vec<VReg>,
    pub insts: Vec<MInst>,
    pub succs: SmallVec<[BlockId; 2]>,
}

/// One stack slot of the frame. Allocation-order indices are
/// [`FrameIndex`]s referenced from operands; `offset` (filled by frame
/// layout) is the slot's distance below the frame pointer.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub size: u32,
    pub align: u32,
    pub offset: u32,
}

#[derive(Debug, Default)]
pub struct FrameInfo {
    pub slots: Vec<Slot>,
    /// Filled post-RA: total bytes to subtract from the stack pointer.
    pub frame_size: u32,
    /// Index in `slots` where the allocator's spill slots begin (they
    /// are appended after the user's allocas).
    pub spill_start: u32,
    /// Callee-saved registers the prologue pushes (set by the ABI).
    pub saved_callee: u32,
}

/// A machine function: blocks of instructions with virtual registers.
/// One block per MIR block, in MIR order; branches carry their targets
/// as operands and block params are resolved by the allocator.
#[derive(Debug)]
pub struct MFunction {
    pub name: String,
    pub local: bool,
    pub weak: bool,
    pub blocks: Vec<MBlock>,
    pub frame: FrameInfo,
    /// Total virtual registers allocated for this function.
    pub num_vregs: u32,
    /// Register class of each vreg (parallel to the vreg numbering).
    pub vreg_class: Vec<RegClass>,
}
