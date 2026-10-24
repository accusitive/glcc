//! Inline-asm wrapper generation (Design A in `docs/inline-asm.md`).
//!
//! Cranelift has no inline asm, so each [`mir::InstKind::InlineAsm`]
//! block becomes a call to a small out-of-line function emitted into
//! the per-TU assembler text. The caller passes the address of a stack
//! area holding every operand's value (or address, for memory
//! operands); the wrapper loads the operand registers, runs the
//! template with `%N` substituted, stores the outputs back into the
//! area, and returns. Only the call reaches Cranelift, so the template
//! never touches the register allocator.
//!
//! Operand registers are allocated statically, per site: outputs first
//! (they are the more constrained), then inputs; ties share; named
//! constraints pin; clobbered registers are excluded. One callee-saved
//! register the block never mentions addresses the area across the
//! template, so templates are free to clobber everything else.

use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, Signature};

use glcc_mir as mir;

use crate::abi::{Arch, Target};

/// Caller-side plan for one operand.
#[derive(Debug, Clone, Copy)]
pub struct Slot {
    pub offset: u64,
    /// The slot holds an address (a memory operand), not a value.
    pub addr: bool,
}

/// A generated wrapper plus the caller-side operand plan.
pub struct Site {
    pub name: String,
    pub text: String,
    /// One entry per operand in GCC order; `None` for immediates and
    /// for inputs tied to another operand's slot.
    pub slots: Vec<Option<Slot>>,
    pub area_size: u64,
}

/// How an operand reaches the template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Reg,
    Mem,
    Imm,
}

/// One register with every spelling a template modifier can ask for;
/// empty strings are widths the register does not have.
struct RegInfo {
    q: &'static str,
    b: &'static str,
    w: &'static str,
    k: &'static str,
    h: &'static str,
    callee_saved: bool,
}

struct Table {
    /// The target these letters and syntax belong to.
    arch: Arch,
    /// Register-name sigil: `%` on x86-64, empty elsewhere.
    sigil: &'static str,
    /// Register holding the area address at wrapper entry.
    arg: &'static str,
    /// Register a wrapper returns its `asm goto` selector in.
    ret: &'static str,
    /// Registers operands may be allocated from, in order.
    pool: &'static [&'static str],
    /// Registers with a high-byte form (`"Q"`); empty elsewhere.
    high_pool: &'static [&'static str],
    /// Area-register candidates, callee-saved and saved by the wrapper.
    reserved: &'static [&'static str],
    regs: &'static [RegInfo],
}

const X86: Table = Table {
    arch: Arch::X86_64,
    sigil: "%",
    arg: "rdi",
    ret: "rax",
    pool: &["rax", "rcx", "rdx", "rsi", "r8", "r9", "r10", "r11"],
    high_pool: &["rax", "rbx", "rcx", "rdx"],
    reserved: &["rbx", "r12", "r13", "r14", "r15"],
    regs: &[
        RegInfo {
            q: "rax",
            b: "al",
            w: "ax",
            k: "eax",
            h: "ah",
            callee_saved: false,
        },
        RegInfo {
            q: "rcx",
            b: "cl",
            w: "cx",
            k: "ecx",
            h: "ch",
            callee_saved: false,
        },
        RegInfo {
            q: "rdx",
            b: "dl",
            w: "dx",
            k: "edx",
            h: "dh",
            callee_saved: false,
        },
        RegInfo {
            q: "rbx",
            b: "bl",
            w: "bx",
            k: "ebx",
            h: "bh",
            callee_saved: true,
        },
        RegInfo {
            q: "rsi",
            b: "sil",
            w: "si",
            k: "esi",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "rdi",
            b: "dil",
            w: "di",
            k: "edi",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "rbp",
            b: "bpl",
            w: "bp",
            k: "ebp",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "rsp",
            b: "spl",
            w: "sp",
            k: "esp",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "r8",
            b: "r8b",
            w: "r8w",
            k: "r8d",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "r9",
            b: "r9b",
            w: "r9w",
            k: "r9d",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "r10",
            b: "r10b",
            w: "r10w",
            k: "r10d",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "r11",
            b: "r11b",
            w: "r11w",
            k: "r11d",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "r12",
            b: "r12b",
            w: "r12w",
            k: "r12d",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "r13",
            b: "r13b",
            w: "r13w",
            k: "r13d",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "r14",
            b: "r14b",
            w: "r14w",
            k: "r14d",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "r15",
            b: "r15b",
            w: "r15w",
            k: "r15d",
            h: "",
            callee_saved: true,
        },
    ],
};

const AARCH64: Table = Table {
    arch: Arch::Aarch64,
    sigil: "",
    arg: "x0",
    ret: "x0",
    pool: &["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17"],
    reserved: &["x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"],
    high_pool: &[],
    regs: &[
        RegInfo {
            q: "x0",
            b: "",
            w: "w0",
            k: "w0",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "x1",
            b: "",
            w: "w1",
            k: "w1",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "x2",
            b: "",
            w: "w2",
            k: "w2",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "x3",
            b: "",
            w: "w3",
            k: "w3",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "x4",
            b: "",
            w: "w4",
            k: "w4",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "x5",
            b: "",
            w: "w5",
            k: "w5",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "x6",
            b: "",
            w: "w6",
            k: "w6",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "x7",
            b: "",
            w: "w7",
            k: "w7",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "x8",
            b: "",
            w: "w8",
            k: "w8",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "x9",
            b: "",
            w: "w9",
            k: "w9",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "x10",
            b: "",
            w: "w10",
            k: "w10",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "x11",
            b: "",
            w: "w11",
            k: "w11",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "x12",
            b: "",
            w: "w12",
            k: "w12",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "x13",
            b: "",
            w: "w13",
            k: "w13",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "x14",
            b: "",
            w: "w14",
            k: "w14",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "x15",
            b: "",
            w: "w15",
            k: "w15",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "x16",
            b: "",
            w: "w16",
            k: "w16",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "x17",
            b: "",
            w: "w17",
            k: "w17",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "x18",
            b: "",
            w: "w18",
            k: "w18",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "x19",
            b: "",
            w: "w19",
            k: "w19",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "x20",
            b: "",
            w: "w20",
            k: "w20",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "x21",
            b: "",
            w: "w21",
            k: "w21",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "x22",
            b: "",
            w: "w22",
            k: "w22",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "x23",
            b: "",
            w: "w23",
            k: "w23",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "x24",
            b: "",
            w: "w24",
            k: "w24",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "x25",
            b: "",
            w: "w25",
            k: "w25",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "x26",
            b: "",
            w: "w26",
            k: "w26",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "x27",
            b: "",
            w: "w27",
            k: "w27",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "x28",
            b: "",
            w: "w28",
            k: "w28",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "x29",
            b: "",
            w: "w29",
            k: "w29",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "x30",
            b: "",
            w: "w30",
            k: "w30",
            h: "",
            callee_saved: false,
        },
    ],
};

const RISCV64: Table = Table {
    arch: Arch::Riscv64,
    sigil: "",
    arg: "a0",
    ret: "a0",
    pool: &["a1", "a2", "a3", "a4", "a5", "a6", "a7", "t0", "t1", "t2", "t3", "t4", "t5", "t6"],
    reserved: &["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11"],
    high_pool: &[],
    regs: &[
        RegInfo {
            q: "zero",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "ra",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "sp",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "gp",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "tp",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "t0",
            b: "",
            w: "t0",
            k: "t0",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "t1",
            b: "",
            w: "t1",
            k: "t1",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "t2",
            b: "",
            w: "t2",
            k: "t2",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "s0",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "s1",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "a0",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "a1",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "a2",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "a3",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "a4",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "a5",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "a6",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "a7",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "s2",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "s3",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "s4",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "s5",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "s6",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "s7",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "s8",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "s9",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "s10",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "s11",
            b: "",
            w: "",
            k: "",
            h: "",
            callee_saved: true,
        },
        RegInfo {
            q: "t3",
            b: "",
            w: "t3",
            k: "t3",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "t4",
            b: "",
            w: "t4",
            k: "t4",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "t5",
            b: "",
            w: "t5",
            k: "t5",
            h: "",
            callee_saved: false,
        },
        RegInfo {
            q: "t6",
            b: "",
            w: "t6",
            k: "t6",
            h: "",
            callee_saved: false,
        },
    ],
};

fn table(arch: Arch) -> &'static Table {
    match arch {
        Arch::X86_64 => &X86,
        Arch::Aarch64 => &AARCH64,
        Arch::Riscv64 => &RISCV64,
    }
}

fn find_reg(table: &Table, name: &str) -> Option<&'static RegInfo> {
    table.regs.iter().find(|info| info.q == name)
}

/// The Clif signature of a wrapper: `(area: i64) -> ()`, or
/// `(area: i64) -> i64` for an `asm goto` block, whose result is the
/// label selector (0 = fallthrough).
pub fn wrapper_sig(target: Target, is_goto: bool) -> Signature {
    let mut sig = Signature::new(target.call_conv());
    sig.params.push(AbiParam::new(types::I64));
    if is_goto {
        sig.returns.push(AbiParam::new(types::I64));
    }
    sig
}

/// One operand as the wrapper sees it.
struct Operand {
    kind: Kind,
    /// Pinned register for a named constraint.
    named: Option<&'static str>,
    /// Register allocated to the operand (canonical name), or the
    /// address register for a memory operand.
    reg: Option<&'static str>,
    /// Tied input: index of the output it shares a location with.
    tie: Option<u32>,
    /// Operand size in bytes.
    size: u64,
    /// Area slot; `None` for immediates and tied inputs.
    slot: Option<Slot>,
    /// Immediate value for `"i"`/`"n"` operands.
    imm: Option<i128>,
    /// Constraint has `+`: the output is also read.
    inout: bool,
    /// Output with a tied input: the register starts from the input
    /// value even without `+`.
    tied_input: bool,
    /// `"Q"`: allocate only a register with a high-byte form.
    high: bool,
    /// `[name]` spelling.
    name: Option<String>,
}

/// Parse and classify one operand's constraint. Returns the kind, a
/// pinned register, a tie index, and the `+` flag.
struct Constraint {
    kind: Kind,
    named: Option<&'static str>,
    tie: Option<u32>,
    inout: bool,
    /// `"Q"`: only a register with a high-byte form.
    high: bool,
    /// The constraint also names a register class (`"Ir"`, `"rm"`):
    /// an operand whose value is not a constant can ride it.
    reg_alt: bool,
    /// The constraint also names a memory class (`"im"`).
    mem_alt: bool,
}

/// What one constraint letter selects on a target. The frontend keeps
/// the same table in `glcc_frontend::check::constraint_letter` (it
/// decides there whether an input is an address or a value); keep the
/// two in step.
#[derive(Clone, Copy)]
enum LetterClass {
    /// An allocatable register, possibly pinned to one (`"a"` is rax).
    Reg(Option<&'static str>),
    /// A register with a high-byte form (x86-64 `"Q"`).
    HighByte,
    /// A memory operand: its address reaches the template in a scratch
    /// register.
    Mem,
    /// An immediate integer constant, substituted textually.
    Imm,
    /// A letter this target has but this backend cannot honor; the
    /// string names what it selects.
    Unsupported(&'static str),
}

/// Class of one constraint letter on `arch`, or `None` when the
/// machine has no such letter. Machine letters differ per
/// architecture: `"Q"` is a high-byte register on x86-64, a
/// base-register memory operand on AArch64, and no letter on RISC-V.
fn letter_class(arch: Arch, letter: &str) -> Option<LetterClass> {
    use Arch::*;
    let class = match (arch, letter) {
        // Registers.
        (_, "r" | "q" | "g" | "X") => LetterClass::Reg(None),
        (X86_64, "Q") => LetterClass::HighByte,
        (X86_64, "a" | "A") => LetterClass::Reg(Some("rax")),
        (X86_64, "b") => LetterClass::Reg(Some("rbx")),
        (X86_64, "c") => LetterClass::Reg(Some("rcx")),
        (X86_64, "d") => LetterClass::Reg(Some("rdx")),
        (X86_64, "S") => LetterClass::Reg(Some("rsi")),
        (X86_64, "D") => LetterClass::Reg(Some("rdi")),
        // AArch64 `Uci`/`Ucj` (registers 8-11 and 12-15, for SVE ZA
        // indexing): any register does for plain uses.
        (Aarch64, "Uci" | "Ucj") => LetterClass::Reg(None),
        // Memory.
        (_, "m" | "o") => LetterClass::Mem,
        (Aarch64, "Q") => LetterClass::Mem,
        (Aarch64, _) if letter.starts_with('U') => LetterClass::Mem,
        (Riscv64, "A") => LetterClass::Mem,
        // Immediates.
        (_, "i" | "n" | "s" | "I" | "J" | "K" | "L" | "M" | "N") => LetterClass::Imm,
        (X86_64 | Aarch64, "Z") => LetterClass::Imm,
        (X86_64, "e") => LetterClass::Imm,
        // Letters these targets have that this backend cannot honor.
        (Aarch64 | Riscv64, "S") => LetterClass::Unsupported("a symbolic address"),
        (Aarch64, "k") => LetterClass::Unsupported("the stack pointer"),
        (Aarch64, "w" | "x" | "y") => LetterClass::Unsupported("a floating-point or SIMD register"),
        (Aarch64, "Upl" | "Upa") => LetterClass::Unsupported("an SVE predicate register"),
        (_, "f") => LetterClass::Unsupported("a floating-point register"),
        _ => return None,
    };
    Some(class)
}

fn parse_constraint(table: &Table, constraint: &str) -> Constraint {
    let mut rest = constraint;
    let mut inout = false;
    loop {
        if let Some(r) = rest.strip_prefix('=') {
            rest = r;
        } else if let Some(r) = rest.strip_prefix('+') {
            inout = true;
            rest = r;
        } else if let Some(r) = rest.strip_prefix('&') {
            rest = r;
        } else {
            break;
        }
    }
    if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
        return Constraint {
            kind: Kind::Reg,
            named: None,
            tie: rest.parse().ok(),
            inout,
            high: false,
            reg_alt: true,
            mem_alt: false,
        };
    }
    let mut named = None;
    let mut kind = None;
    let mut high = false;
    let mut reg_alt = false;
    let mut mem_alt = false;
    let mut rest = rest;
    while let Some(first) = rest.chars().next() {
        let len = if first == 'U' {
            // `U` plus its lowercase suffix (`Ump`, `Ush`, ...).
            1 + rest[1..].find(|c: char| !c.is_ascii_lowercase()).unwrap_or(rest.len() - 1)
        } else {
            first.len_utf8()
        };
        let (letter, tail) = rest.split_at(len);
        rest = tail;
        let (k, n) = match letter_class(table.arch, letter) {
            Some(LetterClass::Reg(named)) => {
                reg_alt = true;
                (Some(Kind::Reg), named)
            }
            Some(LetterClass::HighByte) => {
                high = true;
                reg_alt = true;
                (Some(Kind::Reg), None)
            }
            Some(LetterClass::Mem) => {
                mem_alt = true;
                (Some(Kind::Mem), None)
            }
            Some(LetterClass::Imm) => (Some(Kind::Imm), None),
            Some(LetterClass::Unsupported(what)) => {
                panic!("asm constraint '{letter}' selects {what}, which is not supported")
            }
            None => panic!("unsupported asm constraint '{letter}'"),
        };
        if let Some(n) = n {
            if find_reg(table, n).is_none() {
                panic!("asm constraint '{letter}' is not valid on this target");
            }
            named = Some(n);
        }
        // A named register letter is the most constrained alternative;
        // register classes beat memory, memory beats immediates.
        if k == Some(Kind::Reg) && kind != Some(Kind::Imm) {
            kind = k;
        } else if k.is_some() && kind.is_none() {
            kind = k;
        }
    }
    if named.is_some() {
        kind = Some(Kind::Reg);
    }
    Constraint {
        kind: kind.unwrap_or_else(|| panic!("empty asm constraint")),
        named,
        tie: None,
        inout,
        high,
        reg_alt,
        mem_alt,
    }
}

/// Size in bytes of an operand's type.
fn ty_size(ty: &mir::Ty) -> u64 {
    match ty {
        mir::Ty::Bool => 1,
        mir::Ty::Int { bits } => u64::from(bits.div_ceil(8)).max(1),
        mir::Ty::Float(mir::FloatKind::F32) => 4,
        mir::Ty::Float(mir::FloatKind::F64) => 8,
        mir::Ty::Float(_) => 16,
        mir::Ty::Ptr { .. } => ty.size_bytes(),
        mir::Ty::Array { .. } | mir::Ty::Struct { .. } => ty.size_bytes(),
        mir::Ty::BuiltinVaList { size, .. } => *size,
        mir::Ty::Void => panic!("void asm operand"),
    }
}

/// Integer value of an immediate operand's MIR value.
fn const_int(func: &mir::Function, value: mir::Value) -> i128 {
    match &func.values[value].def {
        mir::Def::Const(mir::Const::Int { value, .. }) => *value,
        _ => panic!("asm operand constraint requires an integer constant"),
    }
}

/// Whether the MIR value is an integer constant, i.e. usable as an
/// immediate operand.
fn is_int_const(func: &mir::Function, value: mir::Value) -> bool {
    matches!(&func.values[value].def, mir::Def::Const(mir::Const::Int { .. }))
}

/// Clobber names that name a register this target can allocate or save;
/// everything else (`"memory"`, `"cc"`, vector registers, ...) needs no
/// wrapper action: the call ABI already covers caller-saved registers
/// and memory.
fn parse_clobbers(table: &Table, clobbers: &[String]) -> Vec<&'static str> {
    let mut out = Vec::new();
    for clobber in clobbers {
        let name = clobber.strip_prefix('%').unwrap_or(clobber);
        if let Some(info) = find_reg(table, name) {
            if info.q == "rsp" || info.q == "sp" {
                panic!("clobbering the stack pointer is not supported");
            }
            if !out.contains(&info.q) {
                out.push(info.q);
            }
        }
    }
    out
}

/// Whether `template` mentions `reg` in any of its spellings.
fn template_mentions(table: &Table, template: &str, reg: &RegInfo) -> bool {
    for name in [reg.q, reg.k, reg.w, reg.b, reg.h] {
        if name.is_empty() {
            continue;
        }
        if table.sigil == "%" {
            if template.contains(&format!("%{name}")) {
                return true;
            }
        } else {
            let bytes = template.as_bytes();
            let needle = name.as_bytes();
            let ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
            let mut from = 0;
            while let Some(pos) = template[from..].find(name) {
                let at = from + pos;
                let before_ok = at == 0 || !ident(bytes[at - 1]);
                let end = at + needle.len();
                let after_ok = end >= bytes.len() || !ident(bytes[end]);
                if before_ok && after_ok {
                    return true;
                }
                from = end;
            }
        }
    }
    false
}

/// The whole plan for one asm site.
pub fn build(block: &mir::AsmBlock, func: &mir::Function, target: Target) -> Site {
    let table = table(target.arch);
    let arch = target.arch;
    let clobbers = parse_clobbers(table, &block.clobbers);
    let outputs = block.outputs as usize;

    let mut operands: Vec<Operand> = Vec::with_capacity(block.operands.len());
    for (i, operand) in block.operands.iter().enumerate() {
        let constraint = parse_constraint(table, &operand.constraint);
        if i < outputs && constraint.tie.is_some() {
            panic!("asm output cannot be a matching constraint");
        }
        if i < outputs && constraint.kind == Kind::Imm {
            panic!("asm output cannot be an immediate");
        }
        let size = ty_size(&operand.ty);
        // An operand with several alternatives (`"Ir"`: immediate or
        // register) takes the immediate only when the value is a
        // constant, matching GCC's constraint matching; otherwise a
        // register (or memory) alternative carries it.
        let mut kind = constraint.kind;
        let imm = match kind {
            Kind::Imm if is_int_const(func, operand.value) => Some(const_int(func, operand.value)),
            Kind::Imm => {
                if constraint.reg_alt {
                    kind = Kind::Reg;
                } else if constraint.mem_alt {
                    kind = Kind::Mem;
                } else {
                    panic!("asm operand constraint requires an integer constant");
                }
                None
            }
            _ => None,
        };
        operands.push(Operand {
            kind,
            named: constraint.named,
            reg: None,
            tie: constraint.tie,
            size,
            slot: None,
            imm,
            inout: constraint.inout,
            tied_input: false,
            high: constraint.high,
            name: operand.name.clone(),
        });
        if kind == Kind::Reg && size > 8 {
            panic!("asm register operand is wider than a register");
        }
    }

    // One callee-saved register the block never names carries the area
    // address across the template.
    let mut reserved = None;
    'candidates: for candidate in table.reserved {
        if clobbers.contains(candidate) {
            continue;
        }
        let info = find_reg(table, candidate).expect("reserved candidate is a register");
        if template_mentions(table, &block.template, info) {
            continue;
        }
        for operand in &operands {
            if operand.kind == Kind::Reg && operand.named == Some(*candidate) {
                continue 'candidates;
            }
        }
        reserved = Some(*candidate);
        break;
    }
    let reserved = reserved.unwrap_or_else(|| panic!("asm block names every callee-saved register; no register is free to address the operand area"));

    // Static allocation: outputs first, then inputs. Ties copy the
    // target's register, named constraints pin, and the pool avoids
    // everything already taken or clobbered. Callee-saved registers the
    // wrapper writes are saved across the template.
    let mut saves: Vec<&'static str> = clobbers.iter().filter(|reg| find_reg(table, reg).is_some_and(|info| info.callee_saved)).copied().collect();
    let mut used: Vec<&'static str> = Vec::new();
    for i in 0..operands.len() {
        if operands[i].kind == Kind::Imm {
            continue;
        }
        if let Some(tie) = operands[i].tie {
            let target = tie as usize;
            if target >= operands.len() || target >= outputs {
                panic!("asm operand number out of range");
            }
            // A tied input shares the output's location, register or
            // memory alike.
            operands[i].kind = operands[target].kind;
            operands[i].reg = operands[target].reg;
            operands[target].tied_input = true;
            continue;
        }
        if let Some(named) = operands[i].named {
            if clobbers.contains(&named) {
                panic!("asm constraint names the clobbered register {named}");
            }
            // Several operands may pin one register (the `cpuid` idiom
            // pairs an `"=a"` output with an `"a"` input): inputs load
            // before the template and outputs store after, so sharing
            // is well defined.
            operands[i].reg = Some(named);
            if !used.contains(&named) {
                used.push(named);
                if find_reg(table, named).is_some_and(|info| info.callee_saved) {
                    saves.push(named);
                }
            }
            continue;
        }
        let pool = if operands[i].high {
            if table.high_pool.is_empty() {
                panic!("asm constraint 'Q' is not valid on this target");
            }
            table.high_pool
        } else {
            table.pool
        };
        let Some(reg) = pool.iter().find(|reg| !used.contains(reg) && !clobbers.contains(reg) && **reg != reserved) else {
            panic!("asm block needs more registers than the target provides");
        };
        operands[i].reg = Some(reg);
        used.push(reg);
    }

    // Area layout: one 8-byte slot per operand that crosses the call as
    // a value or an address; tied inputs share their target's slot.
    let mut offset = 0u64;
    for i in 0..operands.len() {
        let needs_slot = operands[i].tie.is_none() && operands[i].kind != Kind::Imm;
        if !needs_slot {
            continue;
        }
        operands[i].slot = Some(Slot {
            offset,
            addr: operands[i].kind == Kind::Mem,
        });
        offset += 8;
    }
    let area_size = (offset.div_ceil(16) * 16).max(16);

    let (name, hash) = wrapper_name(block, &operands, arch);
    let unique = (hash & 0x7fff_ffff).to_string();
    // `asm goto` returns a selector: 0 for the fallthrough, `i + 1` for a
    // jump to `labels[i]`. The stubs that set it run after the template
    // but before the output stores, so it needs a register no operand
    // uses (and the return move lands after the stores).
    let selector = if block.labels.is_empty() {
        None
    } else {
        Some(
            table
                .pool
                .iter()
                .find(|reg| !used.contains(reg) && **reg != reserved)
                .copied()
                .unwrap_or_else(|| panic!("asm block needs more registers than the target provides")),
        )
    };
    let text = wrapper_text(&name, block, table, arch, &operands, reserved, &saves, &unique, selector);
    Site {
        name,
        text,
        slots: operands.iter().map(|operand| operand.slot).collect(),
        area_size,
    }
}

/// Content-addressed wrapper symbol: identical blocks share one wrapper
/// (and are emitted once), so the hash must cover everything the text
/// depends on.
fn wrapper_name(block: &mir::AsmBlock, operands: &[Operand], arch: Arch) -> (String, u64) {
    let mut hash = 0xcbf29ce484222325u64;
    let mut eat = |bytes: &[u8]| {
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    };
    eat(match arch {
        Arch::X86_64 => b"x86_64",
        Arch::Aarch64 => b"aarch64",
        Arch::Riscv64 => b"riscv64",
    });
    eat(block.template.as_bytes());
    eat(&[block.outputs as u8]);
    for clobber in &block.clobbers {
        eat(clobber.as_bytes());
        eat(&[0]);
    }
    // The label list fixes the selector numbering and the stub text, so
    // two blocks that differ only in label order must not share a name.
    for label in &block.labels {
        eat(label.as_bytes());
        eat(&[0]);
    }
    for operand in operands {
        eat(&[match operand.kind {
            Kind::Reg => b'R',
            Kind::Mem => b'M',
            Kind::Imm => b'I',
        }]);
        eat(&[operand.inout as u8]);
        eat(&(operand.size as u16).to_le_bytes());
        if let Some(tie) = operand.tie {
            eat(&tie.to_le_bytes());
        }
        if let Some(imm) = operand.imm {
            eat(&imm.to_le_bytes());
        }
    }
    (format!("__glcc_asm_{hash:016x}"), hash)
}

fn wrapper_text(
    name: &str,
    block: &mir::AsmBlock,
    table: &Table,
    arch: Arch,
    operands: &[Operand],
    reserved: &str,
    saves: &[&str],
    unique: &str,
    selector: Option<&str>,
) -> String {
    let outputs = block.outputs as usize;
    let mut text = String::new();
    // The wrapper is a function: pin it to `.text` regardless of the
    // section the module text ended in, so a template's own `.text`
    // cannot leave `.size` with a cross-section difference.
    text.push_str(&format!(".text\n.weak {name}\n.type {name}, @function\n{name}:\n"));
    text.push_str(&format!("\t{}\n", push(arch, reserved)));
    for reg in saves {
        text.push_str(&format!("\t{}\n", push(arch, reg)));
    }
    // x86-64: the call leaves rsp ≡ 8 (mod 16); an odd number of 8-byte
    // pushes restores 16-byte alignment for templates that call.
    let pushes = 1 + saves.len();
    let pad = arch == Arch::X86_64 && pushes.is_multiple_of(2);
    if pad {
        text.push_str("\tsub $8, %rsp\n");
    }
    text.push_str(&format!("\t{}\n", move_area(arch, table.arg, reserved)));
    for (i, operand) in operands.iter().enumerate() {
        let is_output = i < outputs;
        if operand.tie.is_some() {
            // The tied input rides the target's register.
            continue;
        }
        match operand.kind {
            Kind::Reg if !is_output || operand.inout || operand.tied_input => {
                let info = find_reg(table, operand.reg.expect("allocated")).expect("known register");
                let offset = operand.slot.expect("slot").offset;
                text.push_str(&format!("\t{}\n", load_into(arch, info, reserved, offset, operand.size)));
            }
            Kind::Mem => {
                let info = find_reg(table, operand.reg.expect("allocated")).expect("known register");
                let offset = operand.slot.expect("slot").offset;
                text.push_str(&format!("\t{}\n", load_into(arch, info, reserved, offset, 8)));
            }
            _ => {}
        }
    }
    if block.is_basic {
        // Basic asm passes through verbatim: `%` is literal (`%eax`),
        // never an operand reference.
        text.push_str(&block.template);
    } else {
        text.push_str(&substitute(&block.template, operands, table, arch, unique, &block.labels));
    }
    if !text.ends_with('\n') {
        text.push('\n');
    }
    let merge = format!(".Lasm{unique}_merge");
    if let Some(sel) = selector {
        // The template falls through with selector 0.
        text.push_str(&format!("\t{}\n", load_imm(arch, sel, 0)));
        text.push_str(&format!("{merge}:\n"));
    }
    for operand in operands.iter().take(outputs) {
        if operand.kind != Kind::Reg || operand.tie.is_some() {
            continue;
        }
        let info = find_reg(table, operand.reg.expect("allocated")).expect("known register");
        let offset = operand.slot.expect("slot").offset;
        text.push_str(&format!("\t{}\n", store_from(arch, info, reserved, offset, operand.size)));
    }
    if let Some(sel) = selector {
        // After the stores: they may read the return register.
        text.push_str(&format!("\t{}\n", move_reg(arch, table.ret, sel)));
    }
    if pad {
        text.push_str("\tadd $8, %rsp\n");
    }
    for reg in saves.iter().rev() {
        text.push_str(&format!("\t{}\n", pop(arch, reg)));
    }
    text.push_str(&format!("\t{}\n", pop(arch, reserved)));
    text.push_str("\tret\n");
    if let Some(sel) = selector {
        // One stub per `asm goto` label, numbered as the caller's
        // dispatch expects: `labels[i]` is selector `i + 1`.
        for i in 0..block.labels.len() {
            text.push_str(&format!(".Lasm{unique}_{i}:\n"));
            text.push_str(&format!("\t{}\n", load_imm(arch, sel, (i + 1) as i128)));
            text.push_str(&format!("\t{}\n", jump(arch, &merge)));
        }
    }
    text.push_str(&format!(".size {name}, .-{name}\n"));
    text
}

/// A stack reference to the area, in the target's addressing syntax.
fn area_ref(arch: Arch, base: &str, offset: u64) -> String {
    match arch {
        Arch::X86_64 => format!("{offset}(%{base})"),
        Arch::Aarch64 => format!("[{base}, #{offset}]"),
        Arch::Riscv64 => format!("{offset}({base})"),
    }
}

/// The name of `info` at the operand's own width: x86-64 uses the exact
/// sub-register, aarch64 uses `wN` up to 4 bytes and `xN` at 8.
fn mode_name(info: &RegInfo, size: u64, arch: Arch) -> &'static str {
    match arch {
        Arch::X86_64 => match size {
            1 => info.b,
            2 => info.w,
            4 => info.k,
            _ => info.q,
        },
        Arch::Aarch64 => {
            if size == 8 {
                info.q
            } else {
                info.k
            }
        }
        Arch::Riscv64 => info.q,
    }
}

fn sigil(table: &Table, name: &str) -> String {
    format!("{}{name}", table.sigil)
}

fn load_into(arch: Arch, info: &RegInfo, base: &str, offset: u64, size: u64) -> String {
    let dst = mode_name(info, size, arch);
    let mem = area_ref(arch, base, offset);
    match arch {
        Arch::X86_64 => {
            let op = match size {
                1 => "movb",
                2 => "movw",
                4 => "movl",
                _ => "movq",
            };
            format!("{op} {mem}, %{dst}")
        }
        Arch::Aarch64 => {
            let op = match size {
                1 => "ldrb",
                2 => "ldrh",
                _ => "ldr",
            };
            format!("{op} {dst}, {mem}")
        }
        Arch::Riscv64 => {
            let op = match size {
                1 => "lbu",
                2 => "lhu",
                4 => "lw",
                _ => "ld",
            };
            format!("{op} {dst}, {mem}")
        }
    }
}

fn store_from(arch: Arch, info: &RegInfo, base: &str, offset: u64, size: u64) -> String {
    let src = mode_name(info, size, arch);
    let mem = area_ref(arch, base, offset);
    match arch {
        Arch::X86_64 => {
            let op = match size {
                1 => "movb",
                2 => "movw",
                4 => "movl",
                _ => "movq",
            };
            format!("{op} %{src}, {mem}")
        }
        Arch::Aarch64 => {
            let op = match size {
                1 => "strb",
                2 => "strh",
                _ => "str",
            };
            format!("{op} {src}, {mem}")
        }
        Arch::Riscv64 => {
            let op = match size {
                1 => "sb",
                2 => "sh",
                4 => "sw",
                _ => "sd",
            };
            format!("{op} {src}, {mem}")
        }
    }
}

fn push(arch: Arch, reg: &str) -> String {
    match arch {
        Arch::X86_64 => format!("push %{reg}"),
        Arch::Aarch64 => format!("str {reg}, [sp, #-16]!"),
        Arch::Riscv64 => format!("addi sp, sp, -16\n\tsd {reg}, 0(sp)"),
    }
}

fn pop(arch: Arch, reg: &str) -> String {
    match arch {
        Arch::X86_64 => format!("pop %{reg}"),
        Arch::Aarch64 => format!("ldr {reg}, [sp], #16"),
        Arch::Riscv64 => format!("ld {reg}, 0(sp)\n\taddi sp, sp, 16"),
    }
}

fn move_area(arch: Arch, arg: &str, reserved: &str) -> String {
    match arch {
        Arch::X86_64 => format!("movq %{arg}, %{reserved}"),
        Arch::Aarch64 => format!("mov {reserved}, {arg}"),
        Arch::Riscv64 => format!("mv {reserved}, {arg}"),
    }
}

fn move_reg(arch: Arch, dst: &str, src: &str) -> String {
    match arch {
        Arch::X86_64 => format!("movq %{src}, %{dst}"),
        Arch::Aarch64 => format!("mov {dst}, {src}"),
        Arch::Riscv64 => format!("mv {dst}, {src}"),
    }
}

fn load_imm(arch: Arch, reg: &str, value: i128) -> String {
    match arch {
        Arch::X86_64 => format!("movq ${value}, %{reg}"),
        Arch::Aarch64 => format!("mov {reg}, #{value}"),
        Arch::Riscv64 => format!("li {reg}, {value}"),
    }
}

fn jump(arch: Arch, label: &str) -> String {
    match arch {
        Arch::X86_64 => format!("jmp {label}"),
        Arch::Aarch64 => format!("b {label}"),
        Arch::Riscv64 => format!("j {label}"),
    }
}

/// Expand the template: `%%`, `%N`, `%cN`/`%nN`, `%bN`/`%wN`/`%kN`/`%qN`/`%hN`,
/// and `%[name]`.
fn substitute(template: &str, operands: &[Operand], table: &Table, arch: Arch, unique: &str, labels: &[String]) -> String {
    let mut out = String::with_capacity(template.len() + 16);
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.peek() {
            Some('%') => {
                chars.next();
                out.push('%');
            }
            Some('=') => {
                chars.next();
                out.push_str(unique);
            }
            Some('l') => {
                // `%l[name]`: the wrapper-local label the stub for
                // `labels[name]` is defined at.
                chars.next();
                if chars.next() != Some('[') {
                    panic!("expected '[' after %l in asm template");
                }
                let mut name = String::new();
                for c in chars.by_ref() {
                    if c == ']' {
                        break;
                    }
                    name.push(c);
                }
                let Some(index) = labels.iter().position(|label| *label == name) else {
                    panic!("asm template references unknown label [{name}]");
                };
                out.push_str(&format!(".Lasm{unique}_{index}"));
            }
            Some('[') => {
                chars.next();
                let mut name = String::new();
                for c in chars.by_ref() {
                    if c == ']' {
                        break;
                    }
                    name.push(c);
                }
                let Some(index) = operands.iter().position(|op| op.name.as_deref() == Some(&name)) else {
                    panic!("asm template references unknown operand name [{name}]");
                };
                out.push_str(&expand(&operands[index], None, table, arch));
            }
            _ => {
                let modifier = match chars.peek() {
                    Some(c) if c.is_ascii_alphabetic() => {
                        let c = *c;
                        chars.next();
                        Some(c)
                    }
                    _ => None,
                };
                let mut digits = String::new();
                while let Some(c) = chars.peek() {
                    if c.is_ascii_digit() {
                        digits.push(*c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                let Some(index) = digits.parse::<usize>().ok() else {
                    panic!("bad `%` in asm template");
                };
                let Some(operand) = operands.get(index) else {
                    panic!("asm template references operand {index} which does not exist");
                };
                out.push_str(&expand(operand, modifier, table, arch));
            }
        }
    }
    out
}

fn expand(operand: &Operand, modifier: Option<char>, table: &Table, arch: Arch) -> String {
    match operand.kind {
        Kind::Imm => {
            let value = operand.imm.expect("immediate operand without a value");
            // GCC prints `%N` with the target's immediate syntax (`$5`,
            // `#5`, `5`) and `%cN`/`%nN` without it.
            let prefix = match arch {
                Arch::X86_64 => "$",
                Arch::Aarch64 => "#",
                Arch::Riscv64 => "",
            };
            match modifier {
                Some('n') => format!("{}", -value),
                Some('c') => format!("{value}"),
                Some('b' | 'w' | 'k' | 'q') | None => format!("{prefix}{value}"),
                Some(other) => panic!("unsupported asm modifier %{other} on an immediate"),
            }
        }
        Kind::Mem => {
            let reg = operand.reg.expect("memory operand without an address register");
            match modifier {
                // RISC-V's assembler spells a memory operand
                // `offset(reg)`; AArch64's brackets it.
                None => match arch {
                    Arch::Aarch64 => format!("[{}]", sigil(table, reg)),
                    _ => format!("({})", sigil(table, reg)),
                },
                Some(other) => panic!("unsupported asm modifier %{other} on a memory operand"),
            }
        }
        Kind::Reg => {
            let reg = operand.reg.expect("register operand without a register");
            let info = find_reg(table, reg).expect("allocated register is known");
            let name = match modifier {
                None => mode_name(info, operand.size, arch),
                Some('b') => info.b,
                Some('w') => info.w,
                Some('k') => info.k,
                Some('q') => info.q,
                Some('h') => info.h,
                Some('c') => panic!("unsupported asm modifier %c on a register operand"),
                Some(other) => panic!("unsupported asm modifier %{other}"),
            };
            if name.is_empty() {
                panic!("register {reg} has no such sub-register");
            }
            sigil(table, name)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glcc_mir::{Const, Function, Signature, Ty};

    fn target(arch: Arch) -> Target {
        Target { arch, os: crate::abi::Os::Linux }
    }

    /// Build a site over `constraints` (GCC order) with all-i64 operands
    /// whose constant values are their operand index, and `outputs`
    /// leading outputs.
    fn build_site(arch: Arch, template: &str, outputs: u32, constraints: &[(&str, Option<u32>)], clobbers: &[&str]) -> Site {
        build_site_with_labels(arch, template, outputs, constraints, clobbers, &[])
    }

    /// [`build_site`] with `asm goto` labels, so the wrapper returns a
    /// selector.
    fn build_goto_site(arch: Arch, template: &str, outputs: u32, constraints: &[(&str, Option<u32>)], labels: &[&str]) -> Site {
        build_site_with_labels(arch, template, outputs, constraints, &[], labels)
    }

    fn build_site_with_labels(arch: Arch, template: &str, outputs: u32, constraints: &[(&str, Option<u32>)], clobbers: &[&str], labels: &[&str]) -> Site {
        let (mut func, _) = Function::new("f", Signature::new(Vec::new(), Ty::Void));
        let operands = constraints
            .iter()
            .enumerate()
            .map(|(i, (constraint, tie))| {
                let value = func.push_const(Const::Int { bits: 64, value: i as i128 });
                mir::AsmOperand {
                    value,
                    constraint: constraint.to_string(),
                    ty: Ty::int(32),
                    tie: *tie,
                    name: None,
                }
            })
            .collect();
        let block = mir::AsmBlock {
            template: template.to_string(),
            operands,
            outputs,
            clobbers: clobbers.iter().map(|c| c.to_string()).collect(),
            labels: labels.iter().map(|l| l.to_string()).collect(),
            is_volatile: false,
            is_basic: false,
        };
        build(&block, &func, target(arch))
    }

    /// [`build_site`] for a basic asm template (no colon): no operands,
    /// the template passes through verbatim.
    fn build_basic_site(arch: Arch, template: &str) -> Site {
        let (func, _) = Function::new("f", Signature::new(Vec::new(), Ty::Void));
        let block = mir::AsmBlock {
            template: template.to_string(),
            operands: Vec::new(),
            outputs: 0,
            clobbers: Vec::new(),
            labels: Vec::new(),
            is_volatile: false,
            is_basic: true,
        };
        build(&block, &func, target(arch))
    }

    #[test]
    fn x86_goto_wrapper_returns_a_selector() {
        // Operand 0 lands in rax, so the selector takes the next free
        // pool register and moves to the return register after the
        // stores.
        let site = build_goto_site(Arch::X86_64, "movl %1, %0; testl %0, %0; jnz %l[lab]", 1, &[("=r", None), ("r", None)], &["lab"]);
        assert!(site.text.contains("movl %ecx, %eax"), "{}", site.text);
        assert!(site.text.contains("jnz .Lasm"), "{}", site.text);
        assert!(site.text.contains("movq $0, %rdx"), "{}", site.text);
        assert!(site.text.contains("movq $1, %rdx"), "{}", site.text);
        assert!(site.text.contains("movq %rdx, %rax"), "{}", site.text);
        let store = site.text.find("movl %eax, 0(%rbx)").expect("output store");
        let ret = site.text.find("movq %rdx, %rax").expect("selector move");
        assert!(store < ret, "stores precede the selector move:\n{}", site.text);
        // The stub label definition follows the `ret`, reached only by
        // template jumps.
        let ret_pos = site.text.find("\tret\n").expect("ret");
        let stub_pos = site.text.find("_0:\n").expect("stub label");
        assert!(stub_pos > ret_pos, "stubs follow the epilogue:\n{}", site.text);
    }

    #[test]
    fn goto_wrapper_name_covers_labels() {
        let plain = build_site(Arch::X86_64, "nop", 0, &[], &[]);
        let with = build_goto_site(Arch::X86_64, "nop", 0, &[], &["lab"]);
        let swapped = build_goto_site(Arch::X86_64, "nop", 0, &[], &["other"]);
        assert_ne!(plain.name, with.name);
        assert_ne!(with.name, swapped.name);
        assert!(with.text.contains("$0,"), "{}", with.text);
        assert!(with.text.contains("$1,"), "{}", with.text);
        assert!(!plain.text.contains("$1,"), "{}", plain.text);
    }

    #[test]
    fn aarch64_and_riscv_goto_stubs_jump_with_the_local_mnemonic() {
        let arm = build_goto_site(Arch::Aarch64, "nop", 0, &[], &["lab"]);
        assert!(arm.text.contains("b .Lasm"), "{}", arm.text);
        assert!(arm.text.contains("mov x1, #1"), "{}", arm.text);
        assert!(arm.text.contains("mov x0, x1"), "{}", arm.text);
        let rv = build_goto_site(Arch::Riscv64, "nop", 0, &[], &["lab"]);
        assert!(rv.text.contains("j .Lasm"), "{}", rv.text);
        assert!(rv.text.contains("li a1, 1"), "{}", rv.text);
        assert!(rv.text.contains("mv a0, a1"), "{}", rv.text);
    }

    #[test]
    fn x86_immediate_class_constraints() {
        let site = build_site(Arch::X86_64, "shll %1, %0", 1, &[("+r", None), ("I", None)], &[]);
        assert!(site.text.contains("shll $1, %eax"), "{}", site.text);
        // `"A"` pins rax on x86-64 (printed at the operand's own width).
        let site = build_site(Arch::X86_64, "movq %1, %0", 1, &[("=A", None), ("r", None)], &[]);
        assert!(site.text.contains("movq %ecx, %eax"), "{}", site.text);
    }

    #[test]
    fn immediate_operands_carry_the_target_prefix() {
        // `%N` prints GCC's immediate syntax; `%cN` and `%nN` do not.
        let x86 = build_site(Arch::X86_64, "movl %1, %0", 1, &[("=r", None), ("i", None)], &[]);
        assert!(x86.text.contains("movl $1, %eax"), "{}", x86.text);
        let x86 = build_site(Arch::X86_64, "movl $%c1, %0", 1, &[("=r", None), ("i", None)], &[]);
        assert!(x86.text.contains("movl $1, %eax"), "{}", x86.text);
        let x86 = build_site(Arch::X86_64, "addl $%n1, %0", 1, &[("+r", None), ("i", None)], &[]);
        assert!(x86.text.contains("addl $-1, %eax"), "{}", x86.text);
        let arm = build_site(Arch::Aarch64, "mov %w0, %1", 1, &[("=r", None), ("i", None)], &[]);
        assert!(arm.text.contains("mov w1, #1"), "{}", arm.text);
        let rv = build_site(Arch::Riscv64, "li %0, %1", 1, &[("=r", None), ("i", None)], &[]);
        assert!(rv.text.contains("li a1, 1"), "{}", rv.text);
    }

    #[test]
    fn basic_templates_pass_through_verbatim() {
        // No colon: `%eax` is literal, not an operand reference, and
        // `%%` is not needed (GCC's basic/extended split; tcctest.c's
        // `asm_dot_test` templates).
        let site = build_basic_site(Arch::X86_64, "mov %eax, %ecx");
        assert!(site.text.contains("mov %eax, %ecx"), "{}", site.text);
        // A wrapper is a function: its symbol is pinned to `.text`, so
        // a template's own `.text` cannot leave `.size name, .-name`
        // spanning sections.
        let site = build_basic_site(Arch::X86_64, ".text; nop");
        assert!(site.text.starts_with(".text\n"), "{}", site.text);
        assert!(site.text.contains(".size __glcc_asm_"), "{}", site.text);
    }

    #[test]
    fn x86_register_wrapper() {
        let site = build_site(Arch::X86_64, "movl %1, %0", 1, &[("=r", None), ("r", None)], &[]);
        assert!(site.name.starts_with("__glcc_asm_"), "{}", site.name);
        assert!(site.text.contains("movl %ecx, %eax"), "{}", site.text);
        assert!(site.text.contains("movl 8(%rbx), %ecx"), "{}", site.text);
        assert_eq!(site.area_size, 16);
        assert_eq!(site.slots[0].map(|s| s.offset), Some(0));
        assert_eq!(site.slots[1].map(|s| s.offset), Some(8));
    }

    #[test]
    fn percent_equals_expands_to_a_unique_local_label() {
        let label_of = |site: &Site| {
            let jne = site.text.split("jne ").nth(1).expect("jne in the template");
            let label = jne.split_whitespace().next().expect("label after jne");
            assert!(label.ends_with('b'), "{label}");
            assert!(label[..label.len() - 1].chars().all(|c| c.is_ascii_digit()));
            label[..label.len() - 1].to_string()
        };
        let one = build_site(Arch::X86_64, "%=: decq %0; jne %=b", 1, &[("+r", None)], &[]);
        let label = label_of(&one);
        assert!(one.text.contains(&format!("{label}: decq")), "{}", one.text);
        let two = build_site(Arch::X86_64, "%=: decq %0; jne %=b", 1, &[("+r", None), ("r", None)], &[]);
        assert_ne!(label, label_of(&two));
    }

    #[test]
    fn x86_tie_shares_a_slot_and_register() {
        let site = build_site(Arch::X86_64, "addl $1, %0", 1, &[("=r", None), ("0", Some(0))], &[]);
        assert!(site.slots[1].is_none(), "tied input has no slot of its own");
        assert!(site.text.contains("addl $1, %eax"), "{}", site.text);
        assert!(site.text.contains("movl 0(%rbx), %eax"), "{}", site.text);
        assert!(site.text.contains("movl %eax, 0(%rbx)"), "{}", site.text);
    }

    #[test]
    fn x86_memory_operand_uses_its_own_register() {
        let site = build_site(Arch::X86_64, "addl $1, %0", 1, &[("+m", None)], &[]);
        assert!(site.slots[0].is_some_and(|slot| slot.addr));
        assert!(site.text.contains("movq 0(%rbx), %rax"), "{}", site.text);
        assert!(site.text.contains("addl $1, (%rax)"), "{}", site.text);
    }

    #[test]
    fn memory_operands_use_the_target_syntax() {
        // RISC-V's assembler spells a memory operand `offset(reg)`,
        // AArch64's brackets it.
        let site = build_site(Arch::Riscv64, "lw %0, %1", 1, &[("=r", None), ("m", None)], &[]);
        assert!(site.text.contains("lw a1, (a2)"), "{}", site.text);
        let site = build_site(Arch::Aarch64, "ldr %w0, %1", 1, &[("=r", None), ("m", None)], &[]);
        assert!(site.text.contains("ldr w1, [x2]"), "{}", site.text);
    }

    #[test]
    fn aarch64_memory_class_constraints() {
        // `"Q"` is a base-register memory operand on AArch64 (a
        // high-byte register on x86-64), and the `U..` qualifiers
        // (`"Ump"`: a load/store-pair address) are memory too.
        let site = build_site(Arch::Aarch64, "ldr %w0, %1", 1, &[("=r", None), ("Q", None)], &[]);
        assert!(site.slots[1].is_some_and(|slot| slot.addr));
        assert!(site.text.contains("ldr w1, [x2]"), "{}", site.text);
        let site = build_site(Arch::Aarch64, "stp %0, %1, %2", 0, &[("r", None), ("r", None), ("Ump", None)], &[]);
        assert!(site.text.contains("stp w1, w2, [x3]"), "{}", site.text);
    }

    #[test]
    fn riscv_address_constraint_is_a_memory_operand() {
        // `"A"`: an address held in a general-purpose register.
        let site = build_site(Arch::Riscv64, "amoadd.w %0, %2, %1", 1, &[("=r", None), ("+A", None), ("r", None)], &[]);
        assert!(site.slots[1].is_some_and(|slot| slot.addr));
        assert!(site.text.contains("amoadd.w a1, a3, (a2)"), "{}", site.text);
    }

    #[test]
    #[should_panic(expected = "asm constraint 'S' selects a symbolic address")]
    fn aarch64_symbolic_address_constraint_is_rejected() {
        build_site(Arch::Aarch64, "adrp %0, %1", 1, &[("=r", None), ("S", None)], &[]);
    }

    #[test]
    #[should_panic(expected = "unsupported asm constraint 'Q'")]
    fn riscv_high_byte_constraint_is_rejected() {
        build_site(Arch::Riscv64, "lw %0, %1", 1, &[("=r", None), ("Q", None)], &[]);
    }

    #[test]
    fn x86_immediate_needs_no_slot() {
        let site = build_site(Arch::X86_64, "movl $%c1, %0", 1, &[("=r", None), ("i", None)], &[]);
        assert!(site.slots[1].is_none());
        assert!(site.text.contains("movl $1, %eax"), "{}", site.text);
    }

    #[test]
    fn immediate_or_register_alternatives() {
        // `"Ir"`: the immediate alternative applies only to a constant
        // value; anything else rides the register.
        let site = build_site(Arch::Aarch64, "add %0, %0, %1", 1, &[("+r", None), ("Ir", None)], &[]);
        assert!(site.slots[1].is_none(), "{}", site.text);
        assert!(site.text.contains("add w1, w1, #1"), "{}", site.text);
        let (mut func, _) = Function::new("f", Signature::new(Vec::new(), Ty::Void));
        let block = func.push_block(vec![Ty::int(32)]);
        let param = func.blocks[block].params[0];
        let out = func.push_const(Const::Int { bits: 64, value: 0 });
        let operands = vec![
            mir::AsmOperand {
                value: out,
                constraint: "+r".to_string(),
                ty: Ty::int(32),
                tie: None,
                name: None,
            },
            mir::AsmOperand {
                value: param,
                constraint: "Ir".to_string(),
                ty: Ty::int(32),
                tie: None,
                name: None,
            },
        ];
        let block = mir::AsmBlock {
            template: "add %0, %0, %1".to_string(),
            operands,
            outputs: 1,
            clobbers: Vec::new(),
            labels: Vec::new(),
            is_volatile: false,
            is_basic: false,
        };
        let site = build(&block, &func, target(Arch::Aarch64));
        assert!(site.slots[1].is_some(), "{}", site.text);
        assert!(site.text.contains("add w1, w1, w2"), "{}", site.text);
    }

    #[test]
    fn x86_reserved_register_avoids_named_and_clobbered() {
        // rbx is written by the template: the area register moves to r12.
        let site = build_site(Arch::X86_64, "movq %%rbx, %%rax", 1, &[("=r", None)], &[]);
        assert!(site.text.contains("push %r12"), "{}", site.text);
        assert!(site.text.contains("movq %rdi, %r12"), "{}", site.text);
        assert!(site.text.contains("movq %rbx, %rax"), "{}", site.text);
        // rbx pinned by a named constraint and r12 clobbered: r13 carries.
        let site = build_site(Arch::X86_64, "movq %%rbx, %%rax", 1, &[("=b", None)], &["r12"]);
        assert!(site.text.contains("push %r13"), "{}", site.text);
        assert!(site.text.contains("movq %rdi, %r13"), "{}", site.text);
        assert!(site.text.contains("push %r12"), "{}", site.text);
        assert!(site.text.contains("pop %r12"), "{}", site.text);
    }

    #[test]
    fn x86_mode_sized_register_names() {
        let (mut func, _) = Function::new("f", Signature::new(Vec::new(), Ty::Void));
        let value = func.push_const(Const::Int { bits: 32, value: 7 });
        let operand = |ty: &Ty| mir::AsmOperand {
            value,
            constraint: "=r".to_string(),
            ty: ty.clone(),
            tie: None,
            name: None,
        };
        for (ty, name) in [(Ty::Bool, "%al"), (Ty::int(16), "%ax"), (Ty::int(32), "%eax"), (Ty::int(64), "%rax")] {
            let block = mir::AsmBlock {
                template: "movl %0, %0".to_string(),
                operands: vec![operand(&ty)],
                outputs: 1,
                clobbers: Vec::new(),
                labels: Vec::new(),
                is_volatile: false,
                is_basic: false,
            };
            let site = build(&block, &func, target(Arch::X86_64));
            assert!(site.text.contains(&format!("movl {name}, {name}")), "{ty:?}: {}", site.text);
        }
    }

    #[test]
    fn x86_subregister_modifiers() {
        let site = build_site(Arch::X86_64, "movb %b0, %h0", 1, &[("=Q", None)], &[]);
        assert!(site.text.contains("movb %al, %ah"), "{}", site.text);
    }

    #[test]
    fn aarch64_wrapper_shape() {
        let site = build_site(Arch::Aarch64, "add %w0, %w0, #1", 1, &[("+r", None)], &[]);
        assert!(site.text.contains("mov x19, x0"), "{}", site.text);
        assert!(site.text.contains("ldr w1, [x19, #0]"), "{}", site.text);
        assert!(site.text.contains("str w1, [x19, #0]"), "{}", site.text);
        assert!(site.text.contains("str x19, [sp, #-16]!"), "{}", site.text);
    }

    #[test]
    fn riscv64_wrapper_shape() {
        let site = build_site(Arch::Riscv64, "addi %0, %0, 1", 1, &[("+r", None)], &[]);
        assert!(site.text.contains("mv s1, a0"), "{}", site.text);
        assert!(site.text.contains("lw a1, 0(s1)"), "{}", site.text);
        assert!(site.text.contains("sw a1, 0(s1)"), "{}", site.text);
    }

    #[test]
    fn identical_blocks_share_a_symbol() {
        let a = build_site(Arch::X86_64, "nop", 0, &[], &[]);
        let b = build_site(Arch::X86_64, "nop", 0, &[], &[]);
        assert_eq!(a.name, b.name);
        let c = build_site(Arch::X86_64, "nop", 0, &[], &["cc"]);
        assert_ne!(a.name, c.name);
    }
}
