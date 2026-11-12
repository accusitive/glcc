//! Target description: the tables every target fills in and the trait
//! the framework drives. Everything that varies between ISAs lives in
//! data here where possible; code lives behind small trait methods only
//! where data cannot express it.

use crate::machine::{MFunction, OpId, PReg, RegClass};

/// Static description of every physical register the target names.
/// The position in `pregs` is the allocator's preg index; `hw` carries
/// the hardware encoding number.
pub struct RegInfo {
    pub pregs: Vec<PRegInfo>,
    /// Allocatable registers by class, most-preferred first. Feeds the
    /// allocator's `MachineEnv`.
    pub preferred: [Vec<PReg>; 2],
    pub non_preferred: [Vec<PReg>; 2],
    /// Dedicated scratch per class, excluded from the allocatable sets
    /// (regalloc2 uses it to resolve cyclic moves).
    pub scratch: [Option<PReg>; 3],
    pub stack_ptr: PReg,
    pub frame_ptr: PReg,
}

pub struct PRegInfo {
    pub preg: PReg,
    pub name: &'static str,
    pub allocatable: bool,
    pub callee_saved: bool,
}

impl RegInfo {
    pub fn preg_index(&self, preg: PReg) -> usize {
        self.pregs.iter().position(|p| p.preg == preg).expect("preg in RegInfo")
    }

    pub fn info(&self, preg: PReg) -> &PRegInfo {
        &self.pregs[self.preg_index(preg)]
    }
}

/// One operand slot of a machine instruction: what the framework (the
/// allocator, the scheduler, the printer) needs to know about it.
/// `def` operands come first in `slots`; encoders index into the
/// instruction's operand list positionally.
#[derive(Clone, Copy, Debug)]
pub struct SlotSpec {
    pub kind: SlotKind,
    pub def: bool,
    pub constraint: Constraint,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SlotKind {
    Reg(RegClass),
    Imm,
    /// A memory reference (`MemOperand`) used in place of a register.
    Mem,
    FrameIndex,
    Sym,
}

#[derive(Clone, Copy, Debug)]
pub enum Constraint {
    Any,
    /// Must be a register.
    Reg,
    Fixed(PReg),
    /// Def only: must share a register with the use in slot `n` of the
    /// same instruction (x86 two-operand forms).
    Reuse(u8),
    /// Immediate must fit in the inclusive range; encoders may pick a
    /// shorter form within it.
    ImmRange(i64, i64),
}

/// Per-opcode facts beyond the operand slots.
#[derive(Clone, Copy, Default)]
pub struct OpFlags {
    /// Three-operand in the IR but two-operand in the encoding: slot 0
    /// (the def) carries a `Constraint::Reuse` pointing at slot 1.
    pub two_addr: bool,
    /// Ends its block and must be the last instruction (branches,
    /// returns). The allocator treats branch operands specially.
    pub branch: bool,
    /// Clobbers all caller-saved registers (`clobbers` lists them).
    pub call: bool,
    pub clobbers: &'static [PReg],
}

/// The target's opcode table. Index = `OpId.0`; target opcodes occupy
/// the range above the framework's reserved pseudo range.
pub struct OpcodeTable {
    pub ops: Vec<OpSpec>,
}

pub struct OpSpec {
    pub name: &'static str,
    pub slots: Vec<SlotSpec>,
    pub flags: OpFlags,
}

impl OpcodeTable {
    pub fn spec(&self, op: OpId) -> &OpSpec {
        &self.ops[op.0 as usize]
    }
}

/// Encoding output: bytes for one instruction plus pending relocation
/// fields and intra-function branch references. Offsets are relative to
/// the start of the instruction; the object layer turns requests into
/// direct patches (targets within this object) or ELF relocations.
pub struct EncOut {
    pub bytes: Vec<u8>,
    pub relocs: Vec<RelocReq>,
    /// (byte offset of the rel32 field, target block) for branches.
    pub block_refs: Vec<(u8, BlockId)>,
}

impl EncOut {
    pub fn new() -> Self {
        EncOut { bytes: Vec::new(), relocs: Vec::new(), block_refs: Vec::new() }
    }
}

impl Default for EncOut {
    fn default() -> Self {
        Self::new()
    }
}

/// A field that needs a symbol address at link time (or at emission
/// time, for intra-object targets).
#[derive(Clone, Debug)]
pub struct RelocReq {
    /// Byte offset of the field within the instruction.
    pub at: u8,
    pub size: u8,
    pub pcrel: bool,
    /// Link-time addend of the relocation (`-4` for a rel32 field, so
    /// the resolved value measures from the field's end).
    pub addend: i64,
    pub sym: crate::machine::SymOperand,
}

pub type BlockId = u32;

/// Items the target's `select_custom` hook can be asked to lower.
pub enum SelItem {
    Inst(glcc_mir::Inst),
    /// The block's terminator.
    Term(glcc_mir::Block),
    /// Function entry: lay out ABI parameters.
    Entry,
}

/// One target. The framework runs the pipeline through this trait; a
/// new backend (including a stack machine or VM) implements exactly
/// these methods.
pub trait TargetIsa: Sync {
    fn name(&self) -> &'static str;
    fn reginfo(&self) -> &'static RegInfo;
    fn opcodes(&self) -> &'static OpcodeTable;
    /// Ordered first-match-wins selection patterns for single
    /// MIR instructions.
    fn patterns(&self) -> &'static crate::sel::PatternTable;
    fn abi(&self) -> &'static dyn AbiImpl;

    /// Escape hatch for shapes the pattern table cannot express
    /// (calls, control flow, complex address computation). Returns
    /// false to let the pattern table try.
    fn select_custom(&self, sel: &mut crate::sel::FuncSel, mir: &glcc_mir::Function, item: SelItem) -> bool;

    /// Opcode that materializes an integer immediate into a register.
    /// The framework's constant handling emits through it; `value`
    /// lets targets pick a narrower form when possible.
    fn const_move_op(&self, value: i64) -> OpId;

    /// Move a GPR's low bits into a vector register (x86-64 `movq`):
    /// float constants materialize their bit pattern in a GPR first,
    /// since SSE has no immediate form.
    fn gpr_to_vec_op(&self) -> OpId;

    /// Opcode that computes an address into a register from a memory
    /// operand (x86-64: `lea`).
    fn lea_op(&self) -> OpId;

    /// Encode one post-RA instruction. `preds`/context come from the
    /// object layer: block label offsets are resolved there.
    fn encode_inst(&self, f: &MFunction, inst: &crate::machine::MInst, out: &mut EncOut);

    /// Patch an intra-function branch/call field after the block
    /// offsets are known. `delta` is `target_off - field_off - 4`: the
    /// value a rel32 field at `at` measures from its own end (targets
    /// recording a whole-instruction field — aarch64 imm26 — divide by
    /// 4 and shift into place).
    fn patch_branch_field(&self, text: &mut [u8], at: u32, delta: i64) {
        let at = at as usize;
        text[at..at + 4].copy_from_slice(&(delta as i32).to_le_bytes());
    }

    /// ELF relocation parameters for a symbol field of `kind`:
    /// (r_type, pcrel, field size in bytes).
    fn elf_reloc(&self, kind: crate::machine::RelocKind) -> (u32, bool, u8);

    /// ELF relocation types for data images: (64-bit pointer, 32-bit
    /// pointer).
    fn elf_data_relocs(&self) -> (u32, u32);

    /// The glasm backend that writes this target's ELF.
    fn asm_arch(&self) -> glasm::Arch;

    /// Whether this ISA runs register allocation. Stack machines and
    /// VMs return false: their ISel output is already final.
    fn needs_ra(&self) -> bool {
        true
    }

    /// Scheduling model: per-opcode latency/throughput, trivial
    /// defaults allowed. The scheduler pass reads it; an empty model
    /// keeps program order.
    fn sched_model(&self) -> &'static crate::sched::SchedModel {
        &crate::sched::SchedModel::TRIVIAL
    }
}

/// Calling convention: how a MIR signature maps onto registers, stack
/// and frame layout. Split out of [`TargetIsa`] so calling conventions
/// (SysV vs Win64) can vary within one ISA later.
pub trait AbiImpl: Sync {
    /// Where the return value goes. Memory means the caller passes a
    /// hidden out-pointer in the first argument register.
    fn ret_loc(&self, ty: &glcc_mir::Ty) -> RetLoc;
    /// Where a parameter of type `ty` goes, given how many integer-
    /// and floating-point-class argument registers earlier parameters
    /// already consumed (SysV tracks the two budgets independently).
    fn param_loc(&self, gp_used: usize, fp_used: usize, ty: &glcc_mir::Ty) -> ParamLoc;
    /// Registers the calling convention gives the callee (integer
    /// class, in argument order).
    fn arg_regs(&self) -> &'static [PReg];
    /// Floating-point argument registers, in argument order.
    fn vec_arg_regs(&self) -> &'static [PReg];
    fn ret_reg(&self) -> PReg;
    /// Floating-point return register.
    fn vec_ret_reg(&self) -> PReg;
    /// Registers a call may destroy (caller-saved, all classes).
    fn caller_saved(&self) -> &'static [PReg];

    /// Insert the function prologue into the entry block (post-RA:
    /// physical registers and the final frame size are known).
    fn emit_prologue(&self, f: &mut crate::machine::MFunction);
    /// Insert an epilogue before every return instruction.
    fn insert_epilogues(&self, f: &mut crate::machine::MFunction);
}

#[derive(Clone, Copy, Debug)]
pub enum RetLoc {
    Void,
    Reg(PReg),
    /// Small by-value aggregate returned in up to four piece registers
    /// (x86-64 rax/rdx or xmm0/xmm1; AAPCS64 HFAs up to four V regs).
    Regs([PReg; 4], u8),
    /// Returned through a hidden pointer; the pointer comes back in the
    /// first argument register.
    Memory,
}

#[derive(Clone, Copy, Debug)]
pub enum ParamLoc {
    Reg(PReg),
    /// By-value aggregate pieces in argument order (class per piece
    /// from the classification; up to four — AAPCS64 HFA upper bound).
    Regs([PReg; 4], u8),
    Stack,
}
