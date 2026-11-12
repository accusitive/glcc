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
    pub branch: Option<BranchInfo>,
}

impl MInst {
    pub fn new(op: OpId, width: Width, ops: impl Into<SmallVec<[Operand; 4]>>) -> Self {
        MInst { op, width, ops: ops.into(), branch: None, call_args: None }
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
