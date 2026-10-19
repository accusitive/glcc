//! C calling-convention classification for MIR signatures.
//!
//! The frontend is responsible for lowering C types to calling
//! convention specific Clif types plus purposes (Cranelift operates at
//! LLVM's level here and will not do it for us). Every side of every
//! call derives locations from one [`layout_sig`] walk, so callers and
//! callees cannot disagree about which register or stack slot holds an
//! argument.
//!
//! Per-target rules, all verified against clang:
//!
//! - x86-64 SysV: each eightbyte votes INTEGER or SSE (INTEGER wins
//!   ties); one eightbyte crosses in one register (`I64`/`F64`), two in
//!   two registers in eightbyte order, anything else travels on the
//!   stack by value (`StructArgument`). Exhaustion sends the whole
//!   aggregate to the stack (no reg/stack split). `long double` is
//!   always stack class; `__float128` rides alone in `xmm0`.
//! - AAPCS64: homogeneous FP aggregates (up to four same-width members,
//!   16-byte alignment defeats) cross in V registers at member width;
//!   other composites up to 16 bytes cross in X registers; 16-aligned
//!   composites round the X counter to even; larger composites cross by
//!   reference. Register exhaustion sends the whole aggregate to the
//!   stack through padding-forced overflow.
//! - RISC-V LP64D: composites with up to two FP members cross field by
//!   field (`fa` for FP members, `a` for the rest); more FP members
//!   demote the whole aggregate to integer registers; larger than 16
//!   bytes crosses by reference; 16-aligned aggregates round the GPR
//!   counter to even. Unnamed FP arguments travel in integer registers.
//!
//! - Win64: 1/2/4/8-byte aggregates cross as an integer of that width,
//!   every other aggregate by reference; returns mirror that (RAX for
//!   small values, a hidden out-pointer otherwise). GP and XMM files
//!   are both indexed by argument position, and unnamed arguments are
//!   mirrored into their position's GP register, so a variadic callee
//!   reads one contiguous 8-byte-slot array: the caller's home area
//!   (spilled in the prelude) followed by the caller's stack slots.
//!
//! Anything that cannot cross in registers crosses on the stack: on
//! x86-64 through `StructArgument` (Cranelift copies the bytes), on
//! AArch64/RISC-V through padding-forced overflow (trailing `I64`
//! chunks after dummy register consumers).

use cranelift_codegen::ir::InstBuilder;
use cranelift_codegen::ir::types::{self, Type};
use cranelift_codegen::ir::{AbiParam, ArgumentExtension, ArgumentPurpose, Signature};
use cranelift_codegen::isa::CallConv;

use glcc_mir as mir;

use crate::memflags;

/// One of the three supported compilation targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    X86_64,
    Aarch64,
    Riscv64,
}

/// Operating system / ABI family. Selects the calling convention on
/// x86-64 (SysV vs Microsoft) and object format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Os {
    #[default]
    Linux,
    /// Windows x86-64 (Microsoft x64 convention, COFF objects).
    Windows,
}

/// Calling-convention target: architecture plus the triple needed to
/// build an ISA. Independent of the frontend's `Target` so this crate
/// stays usable without it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub arch: Arch,
    pub os: Os,
}

impl Target {
    /// Native target of this build: the host's own ABI, which is what
    /// the in-process JIT tests must speak. The driver derives its
    /// target from the frontend instead of calling this.
    pub fn native() -> Target {
        Target {
            arch: match std::env::consts::ARCH {
                "x86_64" => Arch::X86_64,
                "aarch64" => Arch::Aarch64,
                "riscv64" => Arch::Riscv64,
                other => panic!("unsupported host architecture {other}"),
            },
            os: if cfg!(windows) { Os::Windows } else { Os::Linux },
        }
    }

    /// x86-64 Windows target.
    pub const WINDOWS_X86_64: Target = Target {
        arch: Arch::X86_64,
        os: Os::Windows,
    };

    pub fn from_triple(triple: &str) -> Option<Target> {
        let arch = if triple.starts_with("x86_64") {
            Arch::X86_64
        } else if triple.starts_with("aarch64") {
            Arch::Aarch64
        } else if triple.starts_with("riscv64") {
            Arch::Riscv64
        } else {
            return None;
        };
        let os = if triple.contains("windows") || triple.contains("msvc") || triple.contains("mingw") {
            Os::Windows
        } else {
            Os::Linux
        };
        Some(Target { arch, os })
    }

    pub fn triple(&self) -> &'static str {
        match (self.arch, self.os) {
            (Arch::X86_64, Os::Windows) => "x86_64-pc-windows-msvc",
            (Arch::X86_64, Os::Linux) => "x86_64-unknown-linux-gnu",
            (Arch::Aarch64, _) => "aarch64-unknown-linux-gnu",
            (Arch::Riscv64, _) => "riscv64gc-unknown-linux-gnu",
        }
    }

    /// Calling convention: SystemV everywhere except x86-64 Windows,
    /// which uses the Microsoft x64 convention.
    pub fn call_conv(&self) -> CallConv {
        match (self.arch, self.os) {
            (Arch::X86_64, Os::Windows) => CallConv::WindowsFastcall,
            _ => CallConv::SystemV,
        }
    }

    /// General-purpose argument registers available to fixed and
    /// variadic calls.
    pub fn gp_regs(&self) -> usize {
        match (self.arch, self.os) {
            (Arch::X86_64, Os::Windows) => 4,
            (Arch::X86_64, Os::Linux) => 6,
            (Arch::Aarch64 | Arch::Riscv64, _) => 8,
        }
    }

    /// Floating-point argument registers.
    pub fn fp_regs(&self) -> usize {
        match (self.arch, self.os) {
            (Arch::X86_64, Os::Windows) => 4,
            _ => 8,
        }
    }

    /// `va_list` object size: x86-64 SysV's four-field tag, AAPCS64's
    /// five-field tag, RISC-V's walking pointer, and Win64's plain
    /// `char *` walk over the caller's home area.
    pub fn va_list_size(&self) -> u64 {
        match (self.arch, self.os) {
            (Arch::X86_64, Os::Windows) => 8,
            (Arch::X86_64, Os::Linux) => 24,
            (Arch::Aarch64, _) => 32,
            (Arch::Riscv64, _) => 8,
        }
    }

    /// Register save area size for variadic definitions: x86-64's
    /// 6×8 plus 8×16, AAPCS64's 8×8 plus 8×16, RISC-V's 8×8 (FP
    /// varargs travel in integer registers, so no FP spills). Win64
    /// needs no save area of its own: the caller's home area is the
    /// save area (see `win64_prelude`).
    pub fn va_save_size(&self) -> u64 {
        match (self.arch, self.os) {
            (Arch::X86_64, Os::Windows) => 0,
            (Arch::X86_64, Os::Linux) => 6 * 8 + 8 * 16,
            (Arch::Aarch64, _) => 8 * 8 + 8 * 16,
            (Arch::Riscv64, _) => 8 * 8,
        }
    }
}

/// Where one parameter lives: register pieces by value (each with its
/// byte offset in the aggregate), or stack-passed bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamLoc {
    /// Pieces in registers, in order: `(byte offset, Clif type)`.
    /// Empty means skipped (zero-size type).
    Regs(Vec<(u64, Type)>),
    /// Passed on the stack by value (`bytes` is 8-aligned).
    Stack { bytes: u64 },
    /// Passed by reference: one address in a GP register. The callee
    /// copies into a local slot; the caller passes its slot address.
    Pointer,
}

/// Where a return value lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetLoc {
    Void,
    /// Pieces in return registers, in order.
    Regs(Vec<(u64, Type)>),
    /// Hidden caller-provided out-pointer as first parameter.
    Sret,
}

/// Planned locations for one function signature: one entry per MIR
/// parameter plus the return, the expanded Clif slots both sides
/// declare, and variadic cursor state for the `va_list` prelude.
#[derive(Debug, Clone)]
pub struct SigLayout {
    pub target: Target,
    pub params: Vec<ParamLoc>,
    pub ret: RetLoc,
    pub variadic: bool,
    /// Expanded Clif parameter slots, in order: values, pads, chunks,
    /// and the hidden sret slot first when present.
    pub slots: Vec<Slot>,
    /// Expanded Clif return types.
    pub returns: Vec<Type>,
    /// Named GP-consuming parameters (sret included), uncapped.
    pub named_gp: usize,
    /// Named FP-consuming parameters, uncapped.
    pub named_fp: usize,
    /// Named stack-passed bytes (for overflow computation).
    pub named_stack_bytes: u64,
}

/// One Clif signature slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slot {
    pub ty: Type,
    pub purpose: ArgumentPurpose,
    pub kind: SlotKind,
}

/// What a slot carries: a real value (register piece at `offset` of
/// MIR parameter `owner`), a dummy register consumer, a stack-bound
/// aggregate chunk (`owner`, chunk `index`, byte offset `index * 8`),
/// or the hidden out-pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotKind {
    Value {
        owner: usize,
        offset: u64,
    },
    Pad,
    Chunk {
        owner: usize,
        index: u64,
    },
    /// x86-64 `StructArgument` address: the whole aggregate travels
    /// behind this pointer (Cranelift copies the bytes).
    StackAddr {
        owner: usize,
    },
    /// By-reference address: pass the aggregate slot address through.
    RefAddr {
        owner: usize,
    },
    Sret,
}

/// Running register consumption for one signature walk. Public so
/// call-tail emission can continue the callee's counting exactly.
#[derive(Debug, Clone, Default)]
pub struct Counters {
    pub gp: usize,
    pub fp: usize,
}

/// Classify one parameter with continuing counters (call-tail use).
/// Returns whether a 16-alignment even-skip consumed a GP register
/// first: the layout walk inserts a pad slot for it, and call tails
/// must materialize the same dummy argument so the actual register
/// assignment matches the counters.
pub fn classify_next(target: Target, ty: &mir::Ty, counters: &mut Counters) -> (bool, ParamLoc) {
    let skip = even_slot_skip(target, ty) && counters.gp % 2 == 1;
    // AAPCS64 has no register left to skip once the file is exhausted
    // and keeps its stack layout packed; RISC-V pads odd writer
    // positions in both regions, matching the unnamed-quad rule.
    let pad = skip && (target.arch == Arch::Riscv64 || counters.gp < target.gp_regs());
    if pad {
        counters.gp += 1;
    }
    (pad, classify_param(target, ty, counters, true))
}

/// Whether the 16-byte-alignment even-slot rule applies to `ty`: an
/// AAPCS64/RISC-V composite of at most 16 bytes whose alignment is 16
/// starts at an even X/GP slot.
fn even_slot_skip(target: Target, ty: &mir::Ty) -> bool {
    matches!(target.arch, Arch::Aarch64 | Arch::Riscv64)
        && matches!(ty, mir::Ty::Struct { .. } | mir::Ty::Array { .. })
        && (1..=16).contains(&ty.size_bytes())
        && mir_align(ty) == 16
}

/// Register pieces of one aggregate, independent of availability: used
/// by call tails that place stack-bound pieces one slot each.
pub fn aggregate_pieces(target: Target, ty: &mir::Ty) -> Option<Vec<(u64, Type)>> {
    match classify_param(target, ty, &mut Counters::default(), true) {
        ParamLoc::Regs(pieces) => Some(pieces),
        _ => None,
    }
}

/// Classify a whole signature and expand it to Clif slots in one
/// walk, so counting happens once and callers and callees agree
/// without communicating.
pub fn layout_sig(target: Target, params: &[mir::Ty], ret: &mir::Ty, variadic: bool) -> SigLayout {
    let ret_loc = classify_ret(target, ret);
    let mut counters = Counters::default();
    let mut slots = Vec::new();
    let mut returns = Vec::new();
    // x86-64 and RISC-V pass the hidden out-pointer in the first GP
    // register, shifting everything; AAPCS64 uses dedicated x8.
    if matches!(ret_loc, RetLoc::Sret) {
        slots.push(Slot {
            ty: types::I64,
            purpose: ArgumentPurpose::StructReturn,
            kind: SlotKind::Sret,
        });
        if target.arch != Arch::Aarch64 {
            counters.gp += 1;
        }
    } else if let RetLoc::Regs(pieces) = &ret_loc {
        for (_, ty) in pieces {
            returns.push(*ty);
        }
    }
    let mut layout = SigLayout {
        target,
        params: Vec::with_capacity(params.len()),
        ret: ret_loc,
        variadic,
        slots: Vec::new(),
        returns,
        named_gp: counters.gp,
        named_fp: 0,
        named_stack_bytes: 0,
    };
    for (i, param) in params.iter().enumerate() {
        // 16-aligned small composites round the X/GP counter to even on
        // AAPCS64 and RISC-V, consuming the skipped register with a pad.
        if even_slot_skip(target, param) && counters.gp % 2 == 1 {
            counters.gp += 1;
            slots.push(Slot {
                ty: types::I64,
                purpose: ArgumentPurpose::Normal,
                kind: SlotKind::Pad,
            });
        }
        let loc = classify_param(target, param, &mut counters, false);
        match &loc {
            ParamLoc::Pointer => {
                slots.push(Slot {
                    ty: types::I64,
                    purpose: ArgumentPurpose::Normal,
                    kind: SlotKind::RefAddr { owner: i },
                });
            }
            ParamLoc::Regs(pieces) => {
                for (off, ty) in pieces {
                    slots.push(Slot {
                        ty: *ty,
                        purpose: ArgumentPurpose::Normal,
                        kind: SlotKind::Value { owner: i, offset: *off },
                    });
                }
            }
            ParamLoc::Stack { bytes } => {
                if target.arch == Arch::X86_64 {
                    slots.push(Slot {
                        ty: types::I64,
                        purpose: ArgumentPurpose::StructArgument(*bytes as u32),
                        kind: SlotKind::StackAddr { owner: i },
                    });
                } else if !variadic {
                    // Fixed signatures declare stack payload as I64
                    // chunks; variadic callees read it through the
                    // overflow cursor instead of declaring it.
                    let chunks = bytes.div_ceil(8);
                    for k in 0..chunks {
                        slots.push(Slot {
                            ty: types::I64,
                            purpose: ArgumentPurpose::Normal,
                            kind: SlotKind::Chunk { owner: i, index: k },
                        });
                    }
                }
            }
        }
        if let ParamLoc::Stack { bytes } = loc {
            layout.named_stack_bytes += bytes;
        }
        layout.params.push(loc);
    }
    // Snapshot named consumption before variadic padding: pads are not
    // named, so they never enter the va_start cursors.
    layout.named_gp = counters.gp;
    layout.named_fp = counters.fp;
    if variadic {
        if target.os == Os::Windows {
            // Win64 padding to four positions: GP and XMM share the
            // position index, and every unnamed argument (fixed or
            // variadic) is mirrored into its position's GP register by
            // the caller, so I64 pads at the unnamed positions capture
            // all register-passed varargs for the prelude spill. No FP
            // file filling: the GP pads already hold FP bits.
            let positions = slots.iter().filter(|s| !matches!(s.purpose, ArgumentPurpose::StructArgument(_))).count();
            for _ in positions..target.gp_regs() {
                slots.push(Slot {
                    ty: types::I64,
                    purpose: ArgumentPurpose::Normal,
                    kind: SlotKind::Pad,
                });
            }
        } else {
            // Synthetic padding to the full register file: each padding
            // block param aliases the next caller vararg register, and the
            // prelude spill captures register-passed varargs through them.
            while counters.gp < target.gp_regs() {
                counters.gp += 1;
                slots.push(Slot {
                    ty: types::I64,
                    purpose: ArgumentPurpose::Normal,
                    kind: SlotKind::Pad,
                });
            }
            while counters.fp < target.fp_regs() {
                counters.fp += 1;
                slots.push(Slot {
                    // 128-bit pads: the next caller vararg register may
                    // hold a 128-bit actual (long double, `__float128`),
                    // which a 64-bit pad would truncate at the ABI
                    // boundary. Narrower actuals still round-trip: the
                    // stride reserves 16 bytes and readers use their own
                    // width.
                    ty: types::F128,
                    purpose: ArgumentPurpose::Normal,
                    kind: SlotKind::Pad,
                });
            }
        }
    }
    layout.slots = slots;
    layout
}

fn classify_ret(target: Target, ret: &mir::Ty) -> RetLoc {
    match classify_value(target, ret, &mut Counters::default(), true, false) {
        ValueClass::Skip => RetLoc::Void,
        ValueClass::Regs(pieces) => RetLoc::Regs(pieces),
        ValueClass::Stack { .. } | ValueClass::Pointer => RetLoc::Sret,
    }
}

fn classify_param(target: Target, ty: &mir::Ty, counters: &mut Counters, variadic: bool) -> ParamLoc {
    match classify_value(target, ty, counters, false, variadic) {
        ValueClass::Skip => ParamLoc::Regs(vec![]),
        ValueClass::Regs(pieces) => ParamLoc::Regs(pieces),
        ValueClass::Stack { bytes } => ParamLoc::Stack { bytes },
        // By-reference aggregates cross as a plain address.
        ValueClass::Pointer => {
            counters.gp += 1;
            ParamLoc::Pointer
        }
    }
}

/// Classification of one value's crossing, before register-availability
/// is considered.
enum ValueClass {
    Skip,
    Regs(Vec<(u64, Type)>),
    Stack {
        bytes: u64,
    },
    /// Crosses by reference (plain pointer, consuming one GP register).
    Pointer,
}

fn classify_value(target: Target, ty: &mir::Ty, counters: &mut Counters, is_ret: bool, variadic: bool) -> ValueClass {
    match ty {
        mir::Ty::Void => ValueClass::Skip,
        mir::Ty::Bool => {
            counters.gp += 1;
            ValueClass::Regs(vec![(0, int_cover(target, 1))])
        }
        mir::Ty::Int { bits } => {
            if *bits > 64 {
                counters.gp += 2;
                ValueClass::Regs(vec![(0, types::I64), (8, types::I64)])
            } else {
                counters.gp += 1;
                ValueClass::Regs(vec![(0, int_cover(target, *bits))])
            }
        }
        mir::Ty::Ptr { bits, .. } => {
            counters.gp += 1;
            ValueClass::Regs(vec![(0, int_cover(target, *bits))])
        }
        mir::Ty::BuiltinVaList { .. } => {
            counters.gp += 1;
            ValueClass::Regs(vec![(0, types::I64)])
        }
        mir::Ty::Float(kind) => match kind {
            mir::FloatKind::F32 => {
                counters.fp += 1;
                ValueClass::Regs(vec![(0, types::F32)])
            }
            mir::FloatKind::F64 => {
                counters.fp += 1;
                ValueClass::Regs(vec![(0, types::F64)])
            }
            mir::FloatKind::F128 => match target.arch {
                Arch::X86_64 | Arch::Aarch64 => {
                    counters.fp += 1;
                    ValueClass::Regs(vec![(0, types::F128)])
                }
                // RISC-V passes quad in integer registers.
                Arch::Riscv64 => {
                    counters.gp += 2;
                    ValueClass::Regs(vec![(0, types::I64), (8, types::I64)])
                }
            },
            // x87 extended arrives on the stack as a parameter, but a
            // return has no stack home (C would use st(0)), so our own
            // returns carry it as an F128 value.
            mir::FloatKind::F80 if !is_ret => ValueClass::Stack { bytes: 16 },
            mir::FloatKind::F80 => {
                counters.fp += 1;
                ValueClass::Regs(vec![(0, types::F128)])
            }
        },
        mir::Ty::Array { .. } | mir::Ty::Struct { .. } => classify_aggregate(target, ty, counters, variadic),
    }
}

/// Integer cover type: exact small widths on x86-64 (backend parity),
/// W-reg widths elsewhere.
fn int_cover(target: Target, bits: u32) -> Type {
    match target.arch {
        Arch::X86_64 => match bits {
            0..=8 => types::I8,
            9..=16 => types::I16,
            17..=32 => types::I32,
            _ => types::I64,
        },
        Arch::Aarch64 | Arch::Riscv64 => match bits {
            0..=32 => types::I32,
            _ => types::I64,
        },
    }
}

/// Natural alignment mirror of `mir` layout (which keeps this private).
fn mir_align(ty: &mir::Ty) -> u64 {
    match ty {
        mir::Ty::Void | mir::Ty::Bool => 1,
        mir::Ty::Int { bits } => ((*bits).max(8) / 8).min(16) as u64,
        mir::Ty::Float(k) => match k {
            mir::FloatKind::F32 => 4,
            mir::FloatKind::F64 => 8,
            // Quad and x87 extended both align to 16 bytes.
            mir::FloatKind::F128 | mir::FloatKind::F80 => 16,
        },
        mir::Ty::Ptr { bits, .. } => ((*bits as u64).max(8) / 8).min(16),
        mir::Ty::BuiltinVaList { align, .. } => *align,
        mir::Ty::Array { elem, .. } => mir_align(elem),
        mir::Ty::Struct { fields, .. } => fields.iter().map(mir_align).max().unwrap_or(1),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeafClass {
    Int,
    Sse,
    Quad,
    /// Long double, void, or anything else exotic: poisons the whole
    /// aggregate to stack class.
    Memory,
}

fn aggregate_leaves(ty: &mir::Ty, base: u64, out: &mut Vec<(u64, u64, LeafClass)>) {
    match ty {
        mir::Ty::Bool | mir::Ty::Int { .. } | mir::Ty::Ptr { .. } => out.push((base, ty.size_bytes(), LeafClass::Int)),
        mir::Ty::Float(mir::FloatKind::F32) | mir::Ty::Float(mir::FloatKind::F64) => out.push((base, ty.size_bytes(), LeafClass::Sse)),
        mir::Ty::Float(mir::FloatKind::F128) => out.push((base, 16, LeafClass::Quad)),
        mir::Ty::Float(mir::FloatKind::F80) | mir::Ty::Void | mir::Ty::BuiltinVaList { .. } => {
            out.push((base, 1, LeafClass::Memory));
        }
        mir::Ty::Array { elem, len } => {
            let stride = elem.size_bytes();
            for i in 0..*len {
                aggregate_leaves(elem, base + i * stride, out);
            }
        }
        mir::Ty::Struct { fields, offsets, .. } => {
            for (field, off) in fields.iter().zip(offsets) {
                aggregate_leaves(field, base + off, out);
            }
        }
    }
}

/// Classify one aggregate value: register pieces, stack bytes, or a
/// by-reference pointer. Consumes registers from `counters` on success.
fn classify_aggregate(target: Target, ty: &mir::Ty, counters: &mut Counters, variadic: bool) -> ValueClass {
    let size = ty.size_bytes();
    if size == 0 {
        return ValueClass::Skip;
    }
    match target.arch {
        Arch::X86_64 if target.os == Os::Windows => win64_aggregate(size, counters),
        Arch::X86_64 => x86_aggregate(ty, size, counters),
        Arch::Aarch64 => arm_aggregate(ty, size, counters),
        Arch::Riscv64 => riscv_aggregate(ty, size, counters, variadic),
    }
}

/// Win64: an aggregate of 1, 2, 4, or 8 bytes crosses as if it were an
/// integer of that width (never split across a register file); every
/// other size crosses by reference, with the copy owned by the caller.
/// The by-reference address consumes one GP register.
fn win64_aggregate(size: u64, counters: &mut Counters) -> ValueClass {
    let cover = match size {
        1 => types::I8,
        2 => types::I16,
        4 => types::I32,
        8 => types::I64,
        _ => return ValueClass::Pointer,
    };
    counters.gp += 1;
    ValueClass::Regs(vec![(0, cover)])
}

/// x86-64 SysV: each eightbyte votes INTEGER or SSE (INTEGER wins
/// ties); one eightbyte crosses in one register, two in two registers
/// in eightbyte order, anything else travels on the stack. Register
/// exhaustion sends the whole aggregate to the stack.
fn x86_aggregate(ty: &mir::Ty, size: u64, counters: &mut Counters) -> ValueClass {
    // A lone quad rides in one XMM register, never split.
    if size == 16 && is_single_quad(ty) {
        if counters.fp + 1 > 8 {
            return ValueClass::Stack { bytes: 16 };
        }
        counters.fp += 1;
        return ValueClass::Regs(vec![(0, types::F128)]);
    }
    let mut leaves = Vec::new();
    aggregate_leaves(ty, 0, &mut leaves);
    let n = size.div_ceil(8) as usize;
    if n > 2 {
        return ValueClass::Stack { bytes: size.next_multiple_of(8) };
    }
    let mut classes = vec![None; n];
    for (off, len, class) in &leaves {
        if *class == LeafClass::Memory || *class == LeafClass::Quad {
            return ValueClass::Stack { bytes: size.next_multiple_of(8) };
        }
        for i in 0..n {
            let start = (i * 8) as u64;
            if *off < start + 8 && off + len > start {
                let slot = &mut classes[i];
                // INTEGER wins ties inside one eightbyte.
                if *class == LeafClass::Int || slot.is_none() {
                    *slot = Some(*class);
                }
            }
        }
    }
    let mut pieces = Vec::new();
    let mut need_gp = 0;
    let mut need_fp = 0;
    for (i, class) in classes.iter().enumerate() {
        let significant = leaves
            .iter()
            .filter(|(off, len, _)| *off < ((i + 1) * 8) as u64 && off + len > (i * 8) as u64)
            .map(|(off, len, _)| (off + len).saturating_sub((i * 8) as u64).min(8))
            .max()
            .unwrap_or(8) as u32;
        match class {
            // Uninhabited eightbyte cannot occur in a sized aggregate.
            None => return ValueClass::Stack { bytes: size.next_multiple_of(8) },
            Some(LeafClass::Int) => {
                need_gp += 1;
                pieces.push(((i * 8) as u64, if significant <= 4 { types::I32 } else { types::I64 }));
            }
            Some(LeafClass::Sse) => {
                need_fp += 1;
                pieces.push(((i * 8) as u64, if significant <= 4 { types::F32 } else { types::F64 }));
            }
            Some(_) => {
                return ValueClass::Stack { bytes: size.next_multiple_of(8) };
            }
        }
    }
    if counters.gp + need_gp > 6 || counters.fp + need_fp > 8 {
        return ValueClass::Stack { bytes: size.next_multiple_of(8) };
    }
    counters.gp += need_gp;
    counters.fp += need_fp;
    ValueClass::Regs(pieces)
}

/// True when the type is exactly one 16-byte quad value.
fn is_single_quad(ty: &mir::Ty) -> bool {
    let mut leaves = Vec::new();
    aggregate_leaves(ty, 0, &mut leaves);
    matches!(leaves.as_slice(), [(0, 16, LeafClass::Quad)])
}

/// AAPCS64: homogeneous FP aggregates (up to four same-width members)
/// cross in V registers at member width; other composites up to 16
/// bytes cross in X registers; 16-aligned composites round the X
/// counter to even; larger composites cross by reference.
fn arm_aggregate(ty: &mir::Ty, size: u64, counters: &mut Counters) -> ValueClass {
    if size > 16 {
        return ValueClass::Pointer;
    }
    if let Some(pieces) = hfa_pieces(ty, size) {
        let need = pieces.len();
        if counters.fp + need > 8 {
            return arm_stack(size, counters);
        }
        counters.fp += need;
        return ValueClass::Regs(pieces);
    }
    let chunks = size.div_ceil(8) as usize;
    if counters.gp + chunks > 8 {
        return arm_stack(size, counters);
    }
    counters.gp += chunks;
    ValueClass::Regs((0..chunks).map(|i| ((i * 8) as u64, types::I64)).collect())
}

/// Round register consumption for stack-bound aggregates. The padding
/// that forces the overflow is emitted at expansion time; here we only
/// record the bytes for overflow computation.
fn arm_stack(size: u64, _counters: &Counters) -> ValueClass {
    ValueClass::Stack { bytes: size.next_multiple_of(8) }
}

/// HFA member pieces at member width and offsets, or `None`. Alignment
/// 16 defeats HFA; so do mixed widths, gaps, and more than four.
fn hfa_pieces(ty: &mir::Ty, size: u64) -> Option<Vec<(u64, Type)>> {
    if mir_align(ty) == 16 {
        return None;
    }
    let mut leaves = Vec::new();
    aggregate_leaves(ty, 0, &mut leaves);
    if leaves.is_empty() || leaves.len() > 4 {
        return None;
    }
    let width = match leaves[0].2 {
        LeafClass::Sse => match leaves[0].1 {
            4 => types::F32,
            8 => types::F64,
            _ => return None,
        },
        _ => return None,
    };
    let mut off = 0u64;
    let mut pieces = Vec::new();
    for (leaf_off, len, class) in &leaves {
        if *class != LeafClass::Sse || *len != leaves[0].1 || *leaf_off != off {
            return None;
        }
        let member_ty = if *len == 4 { types::F32 } else { types::F64 };
        if member_ty != width {
            return None;
        }
        pieces.push((off, member_ty));
        off += len;
    }
    if off != size {
        return None;
    }
    Some(pieces)
}

/// RISC-V LP64D: composites with up to two FP members cross field by
/// field (`fa` for FP members, `a` for the rest); more FP members
/// demote the whole aggregate to integer registers; larger than 16
/// bytes crosses by reference; 16-aligned aggregates round GPRs even.
/// Unnamed (variadic) arguments send every piece, FP members included,
/// through the integer registers.
fn riscv_aggregate(ty: &mir::Ty, size: u64, counters: &mut Counters, variadic: bool) -> ValueClass {
    if size > 16 {
        return ValueClass::Pointer;
    }
    let mut leaves = Vec::new();
    aggregate_leaves(ty, 0, &mut leaves);
    let fp_count = leaves.iter().filter(|(_, _, c)| *c == LeafClass::Sse).count();
    if fp_count > 2 {
        let chunks = size.div_ceil(8) as usize;
        if counters.gp + chunks > 8 {
            return ValueClass::Stack { bytes: size.next_multiple_of(8) };
        }
        counters.gp += chunks;
        return ValueClass::Regs((0..chunks).map(|i| ((i * 8) as u64, types::I64)).collect());
    }
    // Field-by-field assignment in declaration order: integer runs
    // become 8-byte granules emitted first (a later FP store wins any
    // overlap), FP leaves take their own `fa` registers after.
    let mut pieces = Vec::new();
    let mut need_gp = 0;
    let mut need_fp = 0;
    let mut int_spans: Vec<(u64, u64)> = Vec::new();
    {
        let mut cur: Option<(u64, u64)> = None;
        let mut sorted = leaves.clone();
        sorted.sort_by_key(|(off, _, _)| *off);
        for (off, len, class) in &sorted {
            // Quad leaves ride integer registers on RISC-V.
            if *class == LeafClass::Sse {
                if let Some((s, e)) = cur.take() {
                    int_spans.push((s, e));
                }
            } else if *class == LeafClass::Int || *class == LeafClass::Quad {
                cur = Some(match cur {
                    Some((s, e)) => (s.min(*off), e.max(off + len)),
                    None => (*off, off + len),
                });
            } else {
                return ValueClass::Stack { bytes: size.next_multiple_of(8) };
            }
        }
        if let Some((s, e)) = cur {
            int_spans.push((s, e));
        }
    }
    for (start, end) in int_spans {
        let base = start / 8 * 8;
        let mut off = base;
        while off < end {
            need_gp += 1;
            pieces.push((off, types::I64));
            off += 8;
        }
    }
    for (off, len, class) in leaves.iter() {
        if *class == LeafClass::Sse {
            let member_ty = if *len == 4 { types::F32 } else { types::F64 };
            need_fp += 1;
            pieces.push((*off, member_ty));
        }
    }
    let gp_pieces = if variadic { need_gp + need_fp } else { need_gp };
    if counters.gp + gp_pieces > 8 || (!variadic && counters.fp + need_fp > 8) {
        return ValueClass::Stack { bytes: size.next_multiple_of(8) };
    }
    counters.gp += gp_pieces;
    if !variadic {
        counters.fp += need_fp;
    }
    ValueClass::Regs(pieces)
}

/// Build a Clif signature from expanded slots.
pub fn clif_signature(layout: &SigLayout, call_conv: CallConv) -> Signature {
    let mut sig = Signature::new(call_conv);
    for slot in &layout.slots {
        sig.params.push(AbiParam {
            value_type: slot.ty,
            purpose: slot.purpose,
            extension: ArgumentExtension::None,
        });
    }
    for ret in &layout.returns {
        sig.returns.push(AbiParam::new(*ret));
    }
    sig
}

/// Runtime state for one variadic definition: the register save area
/// plus the overflow base. Created once in the prelude, consumed by
/// `VaStart` and RISC-V `VaArg`.
#[derive(Clone, Copy)]
pub struct VaCtx {
    pub save_base: cranelift_codegen::ir::Value,
    pub overflow: cranelift_codegen::ir::Value,
}

/// Emit the variadic prelude: spill register-passed parameters into
/// one contiguous save area and compute the stack-overflow base.
/// `params` arrives in slot order with the entry block's values.
pub fn emit_prelude(target: Target, builder: &mut cranelift_frontend::FunctionBuilder, params: &[(cranelift_codegen::ir::Value, Slot)], layout: &SigLayout) -> VaCtx {
    use cranelift_codegen::ir::InstBuilder;
    if target.os == Os::Windows {
        return win64_prelude(target, builder, params, layout);
    }
    let save = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
        target.va_save_size() as u32,
        4,
    ));
    let save_base = builder.ins().stack_addr(types::I64, save, 0);
    let mut gp: i32 = 0;
    let mut fp: i32 = 0;
    let gp_limit = target.gp_regs() as i32;
    let fp_limit = target.fp_regs() as i32;
    for (value, slot) in params {
        // Chunks and sret are already where they belong; only
        // register values (including pads, which alias vararg
        // registers) spill.
        if !matches!(slot.kind, SlotKind::Value { .. } | SlotKind::Pad) {
            continue;
        }
        if slot.ty.is_float() {
            // RISC-V never consults FP regs for varargs (unnamed FP
            // travels in integer registers), so nothing spills there.
            if target.arch == Arch::Riscv64 {
                continue;
            }
            if fp >= fp_limit {
                continue;
            }
            let stride: i32 = match target.arch {
                Arch::X86_64 => 16,
                _ => 16,
            };
            let base: i32 = match target.arch {
                Arch::X86_64 => 48,
                _ => 64,
            };
            // Spill register-passed parameters: pads alias the next
            // caller vararg register, so their declared type must be
            // wide enough for any actual (128-bit long doubles arrive
            // in 64-bit pad slots otherwise truncated at the boundary).
            // The stride reserves 16 bytes per FP slot either way.
            builder.ins().stack_store(types::I64, *value, save, base + fp * stride);
            fp += 1;
        } else {
            if gp >= gp_limit {
                continue;
            }
            builder.ins().stack_store(types::I64, *value, save, gp * 8);
            gp += 1;
        }
    }
    let skip = layout.named_gp.saturating_sub(target.gp_regs()) as i64 * 8 + layout.named_fp.saturating_sub(target.fp_regs()) as i64 * 8 + layout.named_stack_bytes as i64;
    let frame = builder.ins().get_frame_pointer(types::I64);
    let overflow = builder.ins().iadd_imm_u(frame, 16 + skip);
    VaCtx { save_base, overflow }
}

/// Win64 variadic prelude: spill the padding params (one per unnamed
/// register position) into the caller's home area, which is where
/// their GP registers are homed by the ABI. `home + 8 * named` is then
/// the first unnamed argument slot: register-argument slots run to
/// `home + 32`, and the caller's stack arguments continue seamlessly
/// from there, so the `char *` walk is one contiguous slot array
/// exactly as MSVC's `va_start`/`va_arg` expect.
fn win64_prelude(target: Target, builder: &mut cranelift_frontend::FunctionBuilder, params: &[(cranelift_codegen::ir::Value, Slot)], layout: &SigLayout) -> VaCtx {
    use cranelift_codegen::ir::InstBuilder;
    use cranelift_codegen::ir::immediates::Offset32;
    // Frame-pointer layout (x86-64): [fp] saved fp, [fp+8] return
    // address, [fp+16] the first home-area byte (entry rsp + 8).
    let frame = builder.ins().get_frame_pointer(types::I64);
    let home = builder.ins().iadd_imm_u(frame, 16);
    let named = layout
        .slots
        .iter()
        .take_while(|s| !matches!(s.kind, SlotKind::Pad))
        .filter(|s| !matches!(s.purpose, ArgumentPurpose::StructArgument(_)))
        .count();
    let mut pos: i64 = 0;
    for (value, slot) in params {
        // `StructArgument` params carry no position (they claim stack
        // bytes only) and do not occur on Win64 outside hand-built MIR.
        if matches!(slot.purpose, ArgumentPurpose::StructArgument(_)) {
            continue;
        }
        if matches!(slot.kind, SlotKind::Pad) && pos < target.gp_regs() as i64 {
            builder.ins().store(memflags::trusted(), *value, home, Offset32::new((pos * 8) as i32));
        }
        pos += 1;
    }
    let start = builder.ins().iadd_imm_u(home, (named * 8) as i64);
    VaCtx { save_base: home, overflow: start }
}

/// x86-64 `gp_offset`/`fp_offset` seeds: past the named register args.
fn x86_starts(layout: &SigLayout) -> (i32, i32) {
    (8 * layout.named_gp.min(6) as i32, 48 + 16 * layout.named_fp.min(8) as i32)
}

/// Emit `VaStart`: fill the tag at `*ap` from prelude state.
pub fn emit_va_start(target: Target, builder: &mut cranelift_frontend::FunctionBuilder, ap: cranelift_codegen::ir::Value, ctx: &VaCtx, layout: &SigLayout) {
    use cranelift_codegen::ir::Value as ClifValue;
    use cranelift_codegen::ir::immediates::Offset32;
    let store32 = |builder: &mut cranelift_frontend::FunctionBuilder, v: ClifValue, off: i32| {
        builder.ins().store(memflags::trusted(), v, ap, Offset32::new(off));
    };
    let storeptr = |builder: &mut cranelift_frontend::FunctionBuilder, v: ClifValue, off: i32| {
        builder.ins().store(memflags::trusted(), v, ap, Offset32::new(off));
    };
    match target.arch {
        Arch::X86_64 if target.os == Os::Windows => {
            // Win64's `va_list` is the walking pointer itself: seed it
            // with the first unnamed-argument slot from the prelude.
            storeptr(builder, ctx.overflow, 0);
        }
        Arch::X86_64 => {
            let (gp_start, fp_start) = x86_starts(layout);
            let gp = builder.ins().iconst(types::I32, gp_start as i64);
            let fp = builder.ins().iconst(types::I32, fp_start as i64);
            store32(builder, gp, 0);
            store32(builder, fp, 4);
            storeptr(builder, ctx.overflow, 8);
            storeptr(builder, ctx.save_base, 16);
        }
        Arch::Aarch64 => {
            let gr_top = builder.ins().iadd_imm_u(ctx.save_base, 64);
            let vr_top = builder.ins().iadd_imm_u(ctx.save_base, 192);
            let gr_offs = builder.ins().iconst(types::I32, (layout.named_gp as i32 - 8) as i64 * 8);
            let vr_offs = builder.ins().iconst(types::I32, (layout.named_fp as i32 - 8) as i64 * 16);
            storeptr(builder, ctx.overflow, 0);
            storeptr(builder, gr_top, 8);
            storeptr(builder, vr_top, 16);
            store32(builder, gr_offs, 24);
            store32(builder, vr_offs, 28);
        }
        Arch::Riscv64 => {
            let addr = if layout.named_gp >= 8 {
                ctx.overflow
            } else {
                builder.ins().iadd_imm_u(ctx.save_base, layout.named_gp as i64 * 8)
            };
            storeptr(builder, addr, 0);
        }
    }
}

/// A lowered `va_arg` result: a scalar value or an aggregate slot.
pub enum VaArgValue {
    Scalar(cranelift_codegen::ir::Value),
    Addr(cranelift_codegen::ir::Value),
}

/// Emit `va_arg` for a requested MIR type: scalars read through one
/// register/stack path, aggregates piece by piece into a fresh slot.
pub fn emit_va_arg(
    target: Target,
    builder: &mut cranelift_frontend::FunctionBuilder,
    ap: cranelift_codegen::ir::Value,
    ctx: &VaCtx,
    layout: &SigLayout,
    req_ty: &mir::Ty,
) -> VaArgValue {
    use cranelift_codegen::ir::InstBuilder;
    // Fresh classification for kind only; register fit never matters
    // for reads because every path selects register-vs-stack itself.
    let mut counters = Counters::default();
    match classify_value(target, req_ty, &mut counters, false, true) {
        // A zero-size aggregate crosses nothing (the caller side skips
        // it too): hand back an empty slot so the copy is a no-op.
        ValueClass::Skip if req_ty.is_aggregate() => {
            let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 0, 4));
            VaArgValue::Addr(builder.ins().stack_addr(types::I64, slot, 0))
        }
        ValueClass::Skip => panic!("va_arg of void type"),
        ValueClass::Regs(pieces) if pieces.len() == 1 && !is_aggregate(req_ty) => VaArgValue::Scalar(va_scalar(target, builder, ap, ctx, layout, pieces[0].1)),
        ValueClass::Regs(pieces) => {
            // AAPCS64 and RISC-V start 16-byte-aligned composites at
            // an even slot; RISC-V also starts 16-byte scalars (quads)
            // there. Callers materialize the same skip as a dummy
            // register (see `classify_next`).
            let quad_scalar = target.arch == Arch::Riscv64 && req_ty.size_bytes() == 16 && !is_aggregate(req_ty);
            if even_slot_skip(target, req_ty) || quad_scalar {
                match target.arch {
                    Arch::Aarch64 => arm_gp_align16(builder, ap),
                    Arch::Riscv64 => riscv_gp_align16(builder, ap, ctx, layout),
                    Arch::X86_64 => {}
                }
            }
            // RISC-V's walk is linear, so a multi-piece aggregate that
            // the caller sent whole to the stack must skip the spill
            // padding instead of reading it as pieces.
            if target.arch == Arch::Riscv64 && is_aggregate(req_ty) && pieces.len() > 1 {
                riscv_va_whole(builder, ap, ctx, layout, pieces.len());
            }
            // Multi-piece aggregates follow the whole-argument rule:
            // one runtime fit check sends every piece to the stack
            // together, mirroring the caller-side decision
            // (`x86_aggregate` / `arm_aggregate`). RISC-V's walking
            // cursor takes its own path below.
            if is_aggregate(req_ty) && pieces.len() > 1 {
                let size = req_ty.size_bytes();
                match target.arch {
                    Arch::X86_64 => {
                        return VaArgValue::Addr(va_aggregate_whole(builder, ap, &pieces, size, RegCursor::x86_gp(), RegCursor::x86_fp(), 8));
                    }
                    Arch::Aarch64 => {
                        return VaArgValue::Addr(va_aggregate_whole(builder, ap, &pieces, size, RegCursor::arm_gp(), RegCursor::arm_fp(), 0));
                    }
                    Arch::Riscv64 => {}
                }
            }
            let size = req_ty.size_bytes();
            let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                size as u32,
                4,
            ));
            let addr = builder.ins().stack_addr(types::I64, slot, 0);
            for (off, ty) in &pieces {
                let v = va_piece(target, builder, ap, ctx, layout, *ty);
                builder.ins().stack_store(types::I64, v, slot, *off as i32);
            }
            if is_aggregate(req_ty) {
                VaArgValue::Addr(addr)
            } else {
                // Multi-piece scalars (128-bit pairs) reassemble to a
                // value: downstream treats only aggregates as
                // addresses, so returning the slot address would spill
                // the address bytes as the value.
                use cranelift_codegen::ir::immediates::Offset32;
                let ty = crate::ty::clif_ty(req_ty).expect("scalar va_arg needs a Clif type");
                VaArgValue::Scalar(builder.ins().load(ty, memflags::trusted(), addr, Offset32::new(0)))
            }
        }
        // Big aggregates (x86) come from the overflow area whole;
        // scalar stack values (x87 `long double`) load as bits.
        ValueClass::Stack { bytes } => {
            // Win64 has no stack-class vararg type from C (long double
            // is a plain double), so this row is SysV-only.
            assert_ne!(target.os, Os::Windows, "no stack-class varargs on Win64");
            assert_eq!(target.arch, Arch::X86_64);
            let (overflow, bumped) = x86_overflow_next(builder, ap, bytes);
            let _ = bumped;
            if req_ty.is_aggregate() {
                let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                    cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                    bytes as u32,
                    4,
                ));
                let dst = builder.ins().stack_addr(types::I64, slot, 0);
                crate::lower::emit_memcpy_pub(builder, dst, overflow, bytes);
                VaArgValue::Addr(dst)
            } else {
                use cranelift_codegen::ir::immediates::Offset32;
                let ty = crate::ty::clif_ty(req_ty).expect("scalar va_arg needs a Clif type");
                VaArgValue::Scalar(builder.ins().load(ty, memflags::notrap(), overflow, Offset32::new(0)))
            }
        }
        // By-reference aggregates: the pointer itself crosses a GP path.
        ValueClass::Pointer => {
            let ptr = va_scalar(target, builder, ap, ctx, layout, types::I64);
            let size = req_ty.size_bytes();
            let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                size as u32,
                4,
            ));
            let dst = builder.ins().stack_addr(types::I64, slot, 0);
            crate::lower::emit_memcpy_pub(builder, dst, ptr, size);
            VaArgValue::Addr(dst)
        }
    }
}

fn is_aggregate(ty: &mir::Ty) -> bool {
    ty.is_aggregate()
}

/// One scalar `va_arg` read of Clif type `ty` through the class path.
fn va_scalar(
    target: Target,
    builder: &mut cranelift_frontend::FunctionBuilder,
    ap: cranelift_codegen::ir::Value,
    ctx: &VaCtx,
    layout: &SigLayout,
    ty: types::Type,
) -> cranelift_codegen::ir::Value {
    use cranelift_codegen::ir::InstBuilder;
    match target.arch {
        Arch::X86_64 if target.os == Os::Windows => win64_va_arg(builder, ap, ty),
        Arch::X86_64 => {
            if ty.is_float() {
                x86_reg_arg(builder, ap, 4, 176, 16, ty)
            } else {
                let step: u32 = 8;
                let _ = step;
                x86_reg_arg(builder, ap, 0, 48, 8, ty)
            }
        }
        Arch::Aarch64 => {
            if ty.is_float() {
                arm_reg_arg(builder, ap, 28, 16, 16, ty)
            } else {
                arm_reg_arg(builder, ap, 24, 8, 8, ty)
            }
        }
        // Unnamed values of every class travel the GP walk on RISC-V.
        Arch::Riscv64 => {
            if ty.bytes() > 8 {
                // 16-byte scalars read as two consecutive slots.
                let lo = riscv_gp_arg(builder, ap, ctx, layout);
                let hi = riscv_gp_arg(builder, ap, ctx, layout);
                let wide = builder.ins().iconcat(lo, hi);
                if ty.is_float() { builder.ins().bitcast(ty, single_flag(), wide) } else { wide }
            } else {
                let bits = riscv_gp_arg(builder, ap, ctx, layout);
                if ty.is_float() {
                    if ty == types::F64 {
                        builder.ins().bitcast(types::F64, single_flag(), bits)
                    } else {
                        let narrow = builder.ins().ireduce(types::I32, bits);
                        builder.ins().bitcast(types::F32, single_flag(), narrow)
                    }
                } else if ty.bytes() < 8 {
                    builder.ins().ireduce(ty, bits)
                } else {
                    bits
                }
            }
        }
    }
}

/// One Win64 `va_arg` read: the `va_list` is a `char *` walking the
/// contiguous 8-byte argument slots (home area, then the caller's
/// stack), so every value loads its natural width from the current
/// slot and the cursor advances one slot. MSVC takes no alignment
/// step here, and no type on this target needs one: `long double` is
/// a plain double and larger aggregates cross by reference.
fn win64_va_arg(builder: &mut cranelift_frontend::FunctionBuilder, ap: cranelift_codegen::ir::Value, ty: types::Type) -> cranelift_codegen::ir::Value {
    use cranelift_codegen::ir::InstBuilder;
    use cranelift_codegen::ir::immediates::Offset32;
    let ptr = builder.ins().load(types::I64, memflags::trusted(), ap, Offset32::new(0));
    let value = builder.ins().load(ty, memflags::notrap(), ptr, Offset32::new(0));
    let next = builder.ins().iadd_imm_u(ptr, 8);
    builder.ins().store(memflags::trusted(), next, ap, Offset32::new(0));
    value
}

/// A piece read for aggregate reassembly: like `va_scalar` but always
/// yielding raw bits for slot stores... scalars return typed values;
/// pieces store through their own width, so this shares the path.
fn va_piece(
    target: Target,
    builder: &mut cranelift_frontend::FunctionBuilder,
    ap: cranelift_codegen::ir::Value,
    ctx: &VaCtx,
    layout: &SigLayout,
    ty: types::Type,
) -> cranelift_codegen::ir::Value {
    // RISC-V FP pieces arrive as integer bits; reinterpret them.
    if target.arch == Arch::Riscv64 && ty.is_float() {
        let bits = riscv_gp_arg(builder, ap, ctx, layout);
        return if ty == types::F64 {
            builder.ins().bitcast(types::F64, single_flag(), bits)
        } else {
            let narrow = builder.ins().ireduce(types::I32, bits);
            builder.ins().bitcast(types::F32, single_flag(), narrow)
        };
    }
    va_scalar(target, builder, ap, ctx, layout, ty)
}

fn single_flag() -> cranelift_codegen::ir::MemFlagsData {
    cranelift_codegen::ir::MemFlagsData::new()
}

/// x86-64 register-or-stack read: the backend's exact select shape.
fn x86_reg_arg(
    builder: &mut cranelift_frontend::FunctionBuilder,
    ap: cranelift_codegen::ir::Value,
    off_field: i32,
    limit: i32,
    step: i32,
    ty: types::Type,
) -> cranelift_codegen::ir::Value {
    use cranelift_codegen::ir::InstBuilder;
    use cranelift_codegen::ir::condcodes::IntCC;
    use cranelift_codegen::ir::immediates::Offset32;
    let cur = builder.ins().load(types::I32, memflags::trusted(), ap, Offset32::new(off_field));
    let reg_save = builder.ins().load(types::I64, memflags::trusted(), ap, Offset32::new(16));
    let overflow = builder.ins().load(types::I64, memflags::trusted(), ap, Offset32::new(8));
    let bound = builder.ins().iconst(types::I32, limit as i64);
    let in_reg = builder.ins().icmp(IntCC::UnsignedLessThan, cur, bound);
    let cur_wide = builder.ins().uextend(types::I64, cur);
    let reg_addr = builder.ins().iadd(reg_save, cur_wide);
    let addr = builder.ins().select(in_reg, reg_addr, overflow);
    let value = builder.ins().load(ty, memflags::notrap(), addr, Offset32::new(0));
    let bumped = builder.ins().iadd_imm_u(cur, step as i64);
    let new_off = builder.ins().select(in_reg, bumped, cur);
    builder.ins().store(memflags::trusted(), new_off, ap, Offset32::new(off_field));
    let stack_step = (ty.bytes().max(8)) as i64;
    let bumped_overflow = builder.ins().iadd_imm_u(overflow, stack_step);
    let new_overflow = builder.ins().select(in_reg, overflow, bumped_overflow);
    builder.ins().store(memflags::trusted(), new_overflow, ap, Offset32::new(8));
    value
}

/// One register file's `va_list` cursor: base and counter fields, the
/// per-piece stride, and the fit bound. x86-64 counts unsigned byte
/// offsets up to a limit; AAPCS64 counts signed offsets down toward
/// zero from a negative start.
#[derive(Clone, Copy)]
struct RegCursor {
    base_field: i32,
    cursor_field: i32,
    stride: i64,
    limit: i64,
    signed: bool,
}

impl RegCursor {
    fn x86_gp() -> Self {
        Self {
            base_field: 16,
            cursor_field: 0,
            stride: 8,
            limit: 48,
            signed: false,
        }
    }
    fn x86_fp() -> Self {
        Self {
            base_field: 16,
            cursor_field: 4,
            stride: 16,
            limit: 176,
            signed: false,
        }
    }
    fn arm_gp() -> Self {
        Self {
            base_field: 8,
            cursor_field: 24,
            stride: 8,
            limit: 0,
            signed: true,
        }
    }
    fn arm_fp() -> Self {
        Self {
            base_field: 16,
            cursor_field: 28,
            stride: 16,
            limit: 0,
            signed: true,
        }
    }
}

/// Whole-argument read of a multi-piece aggregate: every piece comes
/// from the register files only when all of them fit, so one runtime
/// check gates the piece addresses and cursor updates together. x86-64
/// calls it for SysV's whole-argument stack rule, AAPCS64 for the same
/// rule with descending cursors. Stack-bound pieces read at their
/// offsets in the argument's memory image.
fn va_aggregate_whole(
    builder: &mut cranelift_frontend::FunctionBuilder,
    ap: cranelift_codegen::ir::Value,
    pieces: &[(u64, types::Type)],
    size: u64,
    gp: RegCursor,
    fp: RegCursor,
    stack_field: i32,
) -> cranelift_codegen::ir::Value {
    use cranelift_codegen::ir::InstBuilder;
    use cranelift_codegen::ir::condcodes::IntCC;
    use cranelift_codegen::ir::immediates::Offset32;
    let gp_cur = builder.ins().load(types::I32, memflags::trusted(), ap, Offset32::new(gp.cursor_field));
    let fp_cur = builder.ins().load(types::I32, memflags::trusted(), ap, Offset32::new(fp.cursor_field));
    let gp_base = builder.ins().load(types::I64, memflags::trusted(), ap, Offset32::new(gp.base_field));
    let fp_base = builder.ins().load(types::I64, memflags::trusted(), ap, Offset32::new(fp.base_field));
    let stack = builder.ins().load(types::I64, memflags::trusted(), ap, Offset32::new(stack_field));
    let gp_bytes = pieces.iter().filter(|(_, ty)| !ty.is_float()).count() as i64 * gp.stride;
    let fp_bytes = pieces.iter().filter(|(_, ty)| ty.is_float()).count() as i64 * fp.stride;
    let gp_end = builder.ins().iadd_imm_u(gp_cur, gp_bytes);
    let fp_end = builder.ins().iadd_imm_u(fp_cur, fp_bytes);
    let (gp_ok, fp_ok) = if gp.signed {
        (
            builder.ins().icmp_imm_s(IntCC::SignedLessThanOrEqual, gp_end, gp.limit),
            builder.ins().icmp_imm_s(IntCC::SignedLessThanOrEqual, fp_end, fp.limit),
        )
    } else {
        (
            builder.ins().icmp_imm_u(IntCC::UnsignedLessThanOrEqual, gp_end, gp.limit),
            builder.ins().icmp_imm_u(IntCC::UnsignedLessThanOrEqual, fp_end, fp.limit),
        )
    };
    let in_regs = builder.ins().band(gp_ok, fp_ok);
    let store_end = pieces.iter().map(|(off, ty)| *off + ty.bytes() as u64).max().unwrap_or(size);
    let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
        size.max(store_end) as u32,
        4,
    ));
    let mut gp_at = 0i64;
    let mut fp_at = 0i64;
    for (off, ty) in pieces {
        let (cur, cfg, base, piece_at) = if ty.is_float() {
            let at = fp_at;
            fp_at += fp.stride;
            (fp_cur, fp, fp_base, at)
        } else {
            let at = gp_at;
            gp_at += gp.stride;
            (gp_cur, gp, gp_base, at)
        };
        let wide = if cfg.signed {
            builder.ins().sextend(types::I64, cur)
        } else {
            builder.ins().uextend(types::I64, cur)
        };
        let reg_base = builder.ins().iadd(base, wide);
        let reg_addr = builder.ins().iadd_imm_u(reg_base, piece_at);
        let stack_addr = builder.ins().iadd_imm_u(stack, *off as i64);
        let addr = builder.ins().select(in_regs, reg_addr, stack_addr);
        let v = builder.ins().load(*ty, memflags::trusted(), addr, Offset32::new(0));
        builder.ins().stack_store(types::I64, v, slot, *off as i32);
    }
    let bumped_gp = builder.ins().iadd_imm_u(gp_cur, gp_bytes);
    let new_gp = builder.ins().select(in_regs, bumped_gp, gp_cur);
    builder.ins().store(memflags::trusted(), new_gp, ap, Offset32::new(gp.cursor_field));
    let bumped_fp = builder.ins().iadd_imm_u(fp_cur, fp_bytes);
    let new_fp = builder.ins().select(in_regs, bumped_fp, fp_cur);
    builder.ins().store(memflags::trusted(), new_fp, ap, Offset32::new(fp.cursor_field));
    let bumped_stack = builder.ins().iadd_imm_u(stack, size.next_multiple_of(8) as i64);
    let new_stack = builder.ins().select(in_regs, stack, bumped_stack);
    builder.ins().store(memflags::trusted(), new_stack, ap, Offset32::new(stack_field));
    builder.ins().stack_addr(types::I64, slot, 0)
}

/// x86-64 overflow read for whole aggregates: bump by `bytes`.
fn x86_overflow_next(
    builder: &mut cranelift_frontend::FunctionBuilder,
    ap: cranelift_codegen::ir::Value,
    bytes: u64,
) -> (cranelift_codegen::ir::Value, cranelift_codegen::ir::Value) {
    use cranelift_codegen::ir::InstBuilder;
    use cranelift_codegen::ir::immediates::Offset32;
    let overflow = builder.ins().load(types::I64, memflags::trusted(), ap, Offset32::new(8));
    let bumped = builder.ins().iadd_imm_u(overflow, bytes as i64);
    builder.ins().store(memflags::trusted(), bumped, ap, Offset32::new(8));
    (overflow, bumped)
}

/// AAPCS64 register-or-stack read through one cursor pair.
fn arm_reg_arg(
    builder: &mut cranelift_frontend::FunctionBuilder,
    ap: cranelift_codegen::ir::Value,
    off_field: i32,
    top_field: i32,
    step: i32,
    ty: types::Type,
) -> cranelift_codegen::ir::Value {
    use cranelift_codegen::ir::InstBuilder;
    use cranelift_codegen::ir::condcodes::IntCC;
    use cranelift_codegen::ir::immediates::Offset32;
    let offs = builder.ins().load(types::I32, memflags::trusted(), ap, Offset32::new(off_field));
    let top = builder.ins().load(types::I64, memflags::trusted(), ap, Offset32::new(top_field));
    let stack = builder.ins().load(types::I64, memflags::trusted(), ap, Offset32::new(0));
    let zero = builder.ins().iconst(types::I32, 0);
    let in_reg = builder.ins().icmp(IntCC::SignedLessThan, offs, zero);
    let offs_wide = builder.ins().sextend(types::I64, offs);
    let reg_addr = builder.ins().iadd(top, offs_wide);
    let addr = builder.ins().select(in_reg, reg_addr, stack);
    let value = builder.ins().load(ty, memflags::notrap(), addr, Offset32::new(0));
    let bumped_offs = builder.ins().iadd_imm_u(offs, step as i64);
    let new_offs = builder.ins().select(in_reg, bumped_offs, offs);
    builder.ins().store(memflags::trusted(), new_offs, ap, Offset32::new(off_field));
    let bumped_stack = builder.ins().iadd_imm_u(stack, 8);
    let new_stack = builder.ins().select(in_reg, stack, bumped_stack);
    builder.ins().store(memflags::trusted(), new_stack, ap, Offset32::new(0));
    value
}

/// Advance an AAPCS64 `va_list` past the 16-byte-alignment skip: a
/// composite with 16-byte alignment starts at an even X register, so
/// round the GP cursor toward zero to an even slot. Only live cursors
/// move; an exhausted one is left alone (the caller skips a register
/// only when it still has one).
fn arm_gp_align16(builder: &mut cranelift_frontend::FunctionBuilder, ap: cranelift_codegen::ir::Value) {
    use cranelift_codegen::ir::InstBuilder;
    use cranelift_codegen::ir::condcodes::IntCC;
    use cranelift_codegen::ir::immediates::Offset32;
    let offs = builder.ins().load(types::I32, memflags::trusted(), ap, Offset32::new(24));
    let zero = builder.ins().iconst(types::I32, 0);
    let in_regs = builder.ins().icmp(IntCC::SignedLessThan, offs, zero);
    // Negative cursors round up toward zero: (offs + 15) & ~15 turns
    // -56 (x1) into -48 (x2) and leaves -48 alone.
    let rounded = builder.ins().iadd_imm_u(offs, 15);
    let mask = builder.ins().iconst(types::I32, -16);
    let rounded = builder.ins().band(rounded, mask);
    let new_offs = builder.ins().select(in_regs, rounded, offs);
    builder.ins().store(memflags::trusted(), new_offs, ap, Offset32::new(24));
}

/// RISC-V overflow-area base: named arguments that exhausted the
/// register files sit at the front of the incoming stack, so the
/// anonymous stack area starts past them (same formula as the prelude
/// and the redirect in [`riscv_gp_arg`]).
fn riscv_overflow_base(builder: &mut cranelift_frontend::FunctionBuilder, layout: &SigLayout) -> cranelift_codegen::ir::Value {
    use cranelift_codegen::ir::InstBuilder;
    use cranelift_codegen::ir::types;
    let skip = layout.named_gp.saturating_sub(8) as i64 * 8 + layout.named_fp.saturating_sub(8) as i64 * 8 + layout.named_stack_bytes as i64;
    let frame = builder.ins().get_frame_pointer(types::I64);
    builder.ins().iadd_imm_u(frame, 16 + skip)
}

/// Redirect a RISC-V walking `va_list` to the overflow area when the
/// next multi-piece argument does not fit wholly in the remaining
/// spill slots: the caller keeps the pieces together and pads the
/// registers, so the walk must jump the same way instead of reading
/// spill padding as argument data.
fn riscv_va_whole(builder: &mut cranelift_frontend::FunctionBuilder, ap: cranelift_codegen::ir::Value, ctx: &VaCtx, layout: &SigLayout, slots: usize) {
    use cranelift_codegen::ir::InstBuilder;
    use cranelift_codegen::ir::condcodes::IntCC;
    use cranelift_codegen::ir::immediates::Offset32;
    use cranelift_codegen::ir::types;
    let ptr = builder.ins().load(types::I64, memflags::trusted(), ap, Offset32::new(0));
    let spill_end = builder.ins().iadd_imm_u(ctx.save_base, 64);
    let in_spill = builder.ins().icmp(IntCC::UnsignedLessThan, ptr, spill_end);
    let remaining = builder.ins().isub(spill_end, ptr);
    let short = builder.ins().icmp_imm_s(IntCC::SignedLessThan, remaining, slots as i64 * 8);
    let redirect = builder.ins().band(in_spill, short);
    let overflow_base = riscv_overflow_base(builder, layout);
    let new_ptr = builder.ins().select(redirect, overflow_base, ptr);
    builder.ins().store(memflags::trusted(), new_ptr, ap, Offset32::new(0));
}

/// Advance a RISC-V walking `va_list` past alignment padding: 16-byte
/// reads start at an even 8-byte slot (conformant writers pad odd
/// positions, and our writer does too). Region-aware, since the walk
/// spans two regions: spill slots count from `save_base`, overflow
/// slots from the recomputed overflow base (same formula as the
/// redirect in [`riscv_gp_arg`], so both agree on parity: each region
/// starts at an even writer position).
fn riscv_gp_align16(builder: &mut cranelift_frontend::FunctionBuilder, ap: cranelift_codegen::ir::Value, ctx: &VaCtx, layout: &SigLayout) {
    use cranelift_codegen::ir::InstBuilder;
    use cranelift_codegen::ir::condcodes::IntCC;
    use cranelift_codegen::ir::immediates::Offset32;
    use cranelift_codegen::ir::types;
    let ptr = builder.ins().load(types::I64, memflags::trusted(), ap, Offset32::new(0));
    let spill_end = builder.ins().iadd_imm_u(ctx.save_base, 64);
    let overflow_base = riscv_overflow_base(builder, layout);
    let in_spill = builder.ins().icmp(IntCC::UnsignedLessThan, ptr, spill_end);
    let base = builder.ins().select(in_spill, ctx.save_base, overflow_base);
    // Slot index within the region: the offset is always a multiple
    // of 8, so bit 3 is the odd/even test.
    let off = builder.ins().isub(ptr, base);
    let eight = builder.ins().iconst(types::I64, 8);
    let low = builder.ins().band(off, eight);
    let zero = builder.ins().iconst(types::I64, 0);
    let is_odd = builder.ins().icmp(IntCC::NotEqual, low, zero);
    let bumped = builder.ins().iadd_imm_u(ptr, 8);
    let aligned = builder.ins().select(is_odd, bumped, ptr);
    builder.ins().store(memflags::trusted(), aligned, ap, Offset32::new(0));
}

/// RISC-V walking-pointer read of one 8-byte slot: spill slots
/// first, then the stack the pointer is redirected to at the spill
/// boundary.
fn riscv_gp_arg(builder: &mut cranelift_frontend::FunctionBuilder, ap: cranelift_codegen::ir::Value, ctx: &VaCtx, layout: &SigLayout) -> cranelift_codegen::ir::Value {
    use cranelift_codegen::ir::InstBuilder;
    use cranelift_codegen::ir::condcodes::IntCC;
    use cranelift_codegen::ir::immediates::Offset32;
    let ptr = builder.ins().load(types::I64, memflags::trusted(), ap, Offset32::new(0));
    let spill_end = builder.ins().iadd_imm_u(ctx.save_base, 64);
    let value = builder.ins().load(types::I64, memflags::trusted(), ptr, Offset32::new(0));
    let next = builder.ins().iadd_imm_u(ptr, 8);
    // At the spill boundary, redirect onto the overflow area once.
    let skip = layout.named_gp.saturating_sub(8) as i64 * 8 + layout.named_fp.saturating_sub(8) as i64 * 8 + layout.named_stack_bytes as i64;
    let frame = builder.ins().get_frame_pointer(types::I64);
    let overflow_base = builder.ins().iadd_imm_u(frame, 16 + skip);
    let at_end = builder.ins().icmp(IntCC::Equal, next, spill_end);
    let redirected = builder.ins().select(at_end, overflow_base, next);
    builder.ins().store(memflags::trusted(), redirected, ap, Offset32::new(0));
    value
}

/// Emit `va_copy`: whole-tag copy. `va_end` emits nothing on any target.
pub fn emit_va_copy(target: Target, builder: &mut cranelift_frontend::FunctionBuilder, dst: cranelift_codegen::ir::Value, src: cranelift_codegen::ir::Value) {
    crate::lower::emit_memcpy_pub(builder, dst, src, target.va_list_size());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn struct_ty(fields: Vec<mir::Ty>) -> mir::Ty {
        let offsets = mir::dense_field_offsets(&fields);
        let size = fields.iter().map(|f| f.size_bytes()).sum();
        mir::Ty::Struct { fields, offsets, size }
    }

    fn i32_ty() -> mir::Ty {
        mir::Ty::int(32)
    }

    fn i64_ty() -> mir::Ty {
        mir::Ty::int(64)
    }

    fn f32_ty() -> mir::Ty {
        mir::Ty::Float(mir::FloatKind::F32)
    }

    fn f64_ty() -> mir::Ty {
        mir::Ty::Float(mir::FloatKind::F64)
    }

    fn pieces_of(loc: &ParamLoc) -> Vec<(u64, Type)> {
        match loc {
            ParamLoc::Regs(pieces) => pieces.clone(),
            _ => panic!("expected registers"),
        }
    }

    fn param0(target: Target, ty: &mir::Ty) -> ParamLoc {
        let layout = layout_sig(target, std::slice::from_ref(ty), &mir::Ty::Void, false);
        assert_eq!(layout.params.len(), 1);
        layout.params.into_iter().next().unwrap()
    }

    const X86: Target = Target {
        arch: Arch::X86_64,
        os: Os::Linux,
    };
    const ARM: Target = Target {
        arch: Arch::Aarch64,
        os: Os::Linux,
    };
    const RV: Target = Target {
        arch: Arch::Riscv64,
        os: Os::Linux,
    };
    const WIN64: Target = Target::WINDOWS_X86_64;

    // Oracle shapes t2..t10 (8/16/12/32B structs).
    fn p8() -> mir::Ty {
        struct_ty(vec![i32_ty(), i32_ty()])
    }
    fn q16() -> mir::Ty {
        struct_ty(vec![i64_ty(), i64_ty()])
    }
    fn r16() -> mir::Ty {
        struct_ty(vec![f64_ty(), f64_ty()])
    }
    fn m16() -> mir::Ty {
        struct_ty(vec![f64_ty(), i32_ty()])
    }
    fn s8() -> mir::Ty {
        struct_ty(vec![f32_ty(), f32_ty()])
    }
    fn t16() -> mir::Ty {
        struct_ty(vec![f32_ty(), f32_ty(), f32_ty(), f32_ty()])
    }
    fn u8() -> mir::Ty {
        struct_ty(vec![i32_ty(), f32_ty()])
    }
    fn big32() -> mir::Ty {
        struct_ty(vec![i64_ty(), i64_ty(), i64_ty(), i64_ty()])
    }
    fn twelve() -> mir::Ty {
        struct_ty(vec![i32_ty(), i32_ty(), i32_ty()])
    }

    #[test]
    fn x86_splits_per_eightbyte() {
        use types::*;
        assert_eq!(pieces_of(&param0(X86, &p8())), vec![(0, I64)]);
        assert_eq!(pieces_of(&param0(X86, &q16())), vec![(0, I64), (8, I64)]);
        assert_eq!(pieces_of(&param0(X86, &r16())), vec![(0, F64), (8, F64)]);
        // Mixed split: SSE eightbyte first, INTEGER second.
        assert_eq!(pieces_of(&param0(X86, &m16())), vec![(0, F64), (8, I32)]);
        // Single SSE eightbyte carries both floats.
        assert_eq!(pieces_of(&param0(X86, &s8())), vec![(0, F64)]);
        assert_eq!(pieces_of(&param0(X86, &t16())), vec![(0, F64), (8, F64)]);
        // INTEGER wins ties inside one eightbyte.
        assert_eq!(pieces_of(&param0(X86, &u8())), vec![(0, I64)]);
        assert_eq!(pieces_of(&param0(X86, &twelve())), vec![(0, I64), (8, I32)]);
        assert!(matches!(param0(X86, &big32()), ParamLoc::Stack { .. }));
        assert!(matches!(param0(X86, &mir::Ty::Float(mir::FloatKind::F80)), ParamLoc::Stack { .. }));
        // Quad rides alone.
        assert_eq!(pieces_of(&param0(X86, &mir::Ty::Float(mir::FloatKind::F128))), vec![(0, F128)]);
    }

    #[test]
    fn arm_hfa_or_x_or_pointer() {
        use types::*;
        assert_eq!(pieces_of(&param0(ARM, &p8())), vec![(0, I64)]);
        assert_eq!(pieces_of(&param0(ARM, &q16())), vec![(0, I64), (8, I64)]);
        // HFA at member width.
        assert_eq!(pieces_of(&param0(ARM, &r16())), vec![(0, F64), (8, F64)]);
        assert_eq!(pieces_of(&param0(ARM, &s8())), vec![(0, F32), (4, F32)]);
        assert_eq!(pieces_of(&param0(ARM, &t16())), vec![(0, F32), (4, F32), (8, F32), (12, F32)]);
        // Mixed composites go wholly in X registers.
        assert_eq!(pieces_of(&param0(ARM, &m16())), vec![(0, I64), (8, I64)]);
        assert_eq!(pieces_of(&param0(ARM, &u8())), vec![(0, I64)]);
        assert_eq!(pieces_of(&param0(ARM, &twelve())), vec![(0, I64), (8, I64)]);
        // >16B crosses by reference (one GP register).
        assert!(matches!(param0(ARM, &big32()), ParamLoc::Pointer));
        assert_eq!(pieces_of(&param0(ARM, &mir::Ty::Float(mir::FloatKind::F128))), vec![(0, F128)]);
    }

    #[test]
    fn riscv_field_split() {
        use types::*;
        assert_eq!(pieces_of(&param0(RV, &p8())), vec![(0, I64)]);
        assert_eq!(pieces_of(&param0(RV, &q16())), vec![(0, I64), (8, I64)]);
        assert_eq!(pieces_of(&param0(RV, &r16())), vec![(0, F64), (8, F64)]);
        // Split per field across register files.
        assert_eq!(pieces_of(&param0(RV, &m16())), vec![(8, I64), (0, F64)]);
        assert_eq!(pieces_of(&param0(RV, &s8())), vec![(0, F32), (4, F32)]);
        assert_eq!(pieces_of(&param0(RV, &u8())), vec![(0, I64), (4, F32)]);
        // More than two FP members demote to integer registers.
        assert_eq!(pieces_of(&param0(RV, &t16())), vec![(0, I64), (8, I64)]);
        assert_eq!(pieces_of(&param0(RV, &twelve())), vec![(0, I64), (8, I64)]);
        // Quad in integer registers.
        assert_eq!(pieces_of(&param0(RV, &mir::Ty::Float(mir::FloatKind::F128))), vec![(0, I64), (8, I64)]);
        // >16B crosses by reference.
        assert!(matches!(param0(RV, &big32()), ParamLoc::Pointer));
    }

    #[test]
    fn returns_mirror_params_and_sret() {
        // Small struct returns ride registers.
        for target in [X86, ARM, RV] {
            let layout = layout_sig(target, &[], &p8(), false);
            assert!(matches!(layout.ret, RetLoc::Regs(_)));
            assert!(layout.slots.iter().all(|s| !matches!(s.kind, SlotKind::Sret)));
        }
        // Large struct returns take the hidden out-pointer.
        for target in [X86, ARM, RV] {
            let layout = layout_sig(target, &[], &big32(), false);
            assert!(matches!(layout.ret, RetLoc::Sret));
            assert!(layout.slots.iter().any(|s| matches!(s.kind, SlotKind::Sret)));
        }
    }

    #[test]
    fn variadic_layout_cursors() {
        // x86-64 named (int): gp starts past one register.
        let layout = layout_sig(X86, &[i32_ty()], &i32_ty(), true);
        assert!(layout.variadic);
        assert_eq!(layout.named_gp, 1);
        assert_eq!(layout.named_fp, 0);
        // Padding fills both files in declarations (FP pads are
        // 128-bit so 128-bit actuals survive; see `layout_sig`).
        let gp_pads = layout.slots.iter().filter(|s| matches!(s.kind, SlotKind::Pad) && s.ty == types::I64).count();
        let fp_pads = layout.slots.iter().filter(|s| matches!(s.kind, SlotKind::Pad) && s.ty == types::F128).count();
        assert_eq!(gp_pads, 5);
        assert_eq!(fp_pads, 8);
    }

    #[test]
    fn win64_scalars_use_gp_and_xmm_files() {
        use types::*;
        assert_eq!(pieces_of(&param0(WIN64, &i32_ty())), vec![(0, I32)]);
        assert_eq!(pieces_of(&param0(WIN64, &i64_ty())), vec![(0, I64)]);
        assert_eq!(pieces_of(&param0(WIN64, &f32_ty())), vec![(0, F32)]);
        assert_eq!(pieces_of(&param0(WIN64, &f64_ty())), vec![(0, F64)]);
        assert_eq!(pieces_of(&param0(WIN64, &mir::Ty::ptr(mir::Ty::Void))), vec![(0, I64)]);
        assert_eq!(pieces_of(&param0(WIN64, &mir::Ty::Bool)), vec![(0, I8)]);
    }

    #[test]
    fn win64_aggregates_by_cover_or_reference() {
        use types::*;
        // 1, 2, 4, 8 bytes cross as an integer of that width.
        let s1 = struct_ty(vec![mir::Ty::int(8)]);
        let s2 = struct_ty(vec![mir::Ty::int(16)]);
        let s4 = struct_ty(vec![i32_ty()]);
        assert_eq!(pieces_of(&param0(WIN64, &s1)), vec![(0, I8)]);
        assert_eq!(pieces_of(&param0(WIN64, &s2)), vec![(0, I16)]);
        assert_eq!(pieces_of(&param0(WIN64, &s4)), vec![(0, I32)]);
        assert_eq!(pieces_of(&param0(WIN64, &p8())), vec![(0, I64)]);
        // Every other size crosses by reference, one GP register.
        let s3 = struct_ty(vec![mir::Ty::int(8), mir::Ty::int(16)]);
        let s5 = struct_ty(vec![mir::Ty::int(32), mir::Ty::int(8)]);
        for ty in [s3, s5, twelve(), q16(), r16(), big32()] {
            assert!(matches!(param0(WIN64, &ty), ParamLoc::Pointer), "{ty:?}");
        }
    }

    #[test]
    fn win64_returns_and_sret() {
        // Small aggregates come back in RAX.
        let layout = layout_sig(WIN64, &[], &p8(), false);
        assert_eq!(layout.ret, RetLoc::Regs(vec![(0, types::I64)]));
        let layout = layout_sig(WIN64, &[], &f64_ty(), false);
        assert_eq!(layout.ret, RetLoc::Regs(vec![(0, types::F64)]));
        // Anything else takes the hidden out-pointer.
        let layout = layout_sig(WIN64, &[], &twelve(), false);
        assert_eq!(layout.ret, RetLoc::Sret);
        assert!(layout.slots.iter().any(|s| matches!(s.kind, SlotKind::Sret)));
    }

    #[test]
    fn win64_target_shape() {
        assert_eq!(WIN64.call_conv(), CallConv::WindowsFastcall);
        assert_eq!(X86.call_conv(), CallConv::SystemV);
        assert_eq!(WIN64.triple(), "x86_64-pc-windows-msvc");
        assert_eq!(WIN64.gp_regs(), 4);
        assert_eq!(WIN64.fp_regs(), 4);
        assert_eq!(WIN64.va_list_size(), 8);
        assert_eq!(WIN64.va_save_size(), 0);
        assert_eq!(Target::from_triple("x86_64-pc-windows-msvc"), Some(WIN64));
        assert_eq!(Target::from_triple("x86_64-unknown-linux-gnu"), Some(X86));
        assert_eq!(Target::from_triple("mips64-unknown-linux-gnu"), None);
    }

    #[test]
    fn win64_variadic_pads_fill_positions() {
        // One named int: the three remaining positions take I64 pads
        // (never F128: FP varargs are read from their GP mirror).
        let layout = layout_sig(WIN64, &[i32_ty()], &i32_ty(), true);
        assert!(layout.variadic);
        let pads: Vec<Type> = layout.slots.iter().filter(|s| matches!(s.kind, SlotKind::Pad)).map(|s| s.ty).collect();
        assert_eq!(pads, vec![types::I64, types::I64, types::I64]);
        // FP named args consume positions like any other: mixed
        // (double, int) leaves two pads.
        let layout = layout_sig(WIN64, &[f64_ty(), i32_ty()], &i32_ty(), true);
        let pads = layout.slots.iter().filter(|s| matches!(s.kind, SlotKind::Pad)).count();
        assert_eq!(pads, 2);
        // Four named positions exhaust the register file: no pads.
        let layout = layout_sig(WIN64, &[i32_ty(), i32_ty(), i32_ty(), i32_ty()], &i32_ty(), true);
        assert!(layout.slots.iter().all(|s| !matches!(s.kind, SlotKind::Pad)));
        // The hidden sret pointer occupies position 0 like MSVC's.
        let layout = layout_sig(WIN64, &[i32_ty()], &big32(), true);
        let pads = layout.slots.iter().filter(|s| matches!(s.kind, SlotKind::Pad)).count();
        assert_eq!(pads, 2);
    }
}
