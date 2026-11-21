//! PrismIR: the shared shader IR all prism targets consume.
//!
//! An SSA, width-typed, DX9-flavored IR with structured control flow.
//! Virtual registers carry a type (`float`/`i32` scalars, 2/3/4-lane
//! vectors, predicates); operations are lane-wise ALU forms with
//! DX9-style source operands — swizzled, negatable, and reading
//! uniforms/inputs/consts inline — so `dp4 t0.x, c0, v0` is one
//! instruction. Physical registers, write masks, register banking, and
//! target-specific packing are per-target lowering concerns; targets
//! see typed virtual registers and op widths only.
//!
//! Control flow is a region tree ([`Stmt`]): `if/else`, a `loop`
//! (conditionless; exit is `break`, iteration is `continue`), and
//! `ret`. SSA joins live in the region statements themselves: an
//! [`Stmt::If`] carries its join registers (one value per arm), a
//! [`Stmt::Loop`] carries its per-iteration registers (the init value
//! plus one from every `continue`), and a loop's exit registers gather
//! one value per `break`. There are no standalone phi instructions —
//! the region boundaries are the phis, in statement order.
//!
//! Every prism program is a single entry point (helpers are inlined
//! into it before [`crate::build`]); inputs, outputs, uniforms, and
//! samplers are declared interface resources ([`crate::interface`]),
//! read inline through [`Src::Input`] / [`Src::Uniform`] and written
//! once by [`Stmt::Exit`].

use std::format;
use std::vec;
use std::vec::Vec;

use smallvec::SmallVec;

/// A virtual register. One arena per program; every register is
/// defined exactly once (by the statement holding it, or as a region
/// join).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VReg(pub u32);

impl core::fmt::Display for VReg {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One scalar kind. Integer width rides along; prism has no 64-bit
/// integers (the profile rejects them). Boolean values are `Int { bits: 32 }`
/// holding 0/1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scalar {
    F32,
    Int { bits: u8 },
}

impl Scalar {
    pub fn i32() -> Scalar {
        Scalar::Int { bits: 32 }
    }
}

/// Register type: scalars, the prism vector shapes, or a predicate
/// (produced by `setp`, consumed by `if`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegTy {
    Scalar(Scalar),
    Vector { elem: Scalar, lanes: u8 },
    Pred,
}

impl RegTy {
    /// Lane count: 1 for scalars and predicates.
    pub fn lanes(self) -> u8 {
        match self {
            RegTy::Scalar(_) | RegTy::Pred => 1,
            RegTy::Vector { lanes, .. } => lanes,
        }
    }

    pub fn is_f32(self) -> bool {
        matches!(self, RegTy::Scalar(Scalar::F32) | RegTy::Vector { elem: Scalar::F32, .. })
    }
}

/// One source operand. Swizzles select lanes of the source (lane
/// indices are within the source register's own width; repeats are
/// allowed, which is how scalar splats are spelled). `negate` is the
/// DX9 unary minus. Uniforms carry their c-register offset within the
/// object ([`crate::interface`]), optionally indexed relative — the
/// DX9 `aL` addressing form.
#[derive(Debug, Clone, PartialEq)]
pub enum Src {
    VReg {
        v: VReg,
        swizzle: Option<Swizzle>,
        negate: bool,
    },
    ConstF32(f32),
    ConstI32(i32),
    /// An undefined value (an interface output never written, a local
    /// read before any store): era targets read whatever is in the
    /// register; SPIR-V emits `OpUndef`.
    Undef,
    Input {
        input: u32,
        swizzle: Option<Swizzle>,
        negate: bool,
    },
    Uniform {
        object: u32,
        reg: u8,
        swizzle: Option<Swizzle>,
        negate: bool,
        rel: Option<VReg>,
    },
}

/// A src swizzle: 1–4 lane indices into the source register.
pub type Swizzle = SmallVec<[u8; 4]>;

/// Comparison code, shared by value compares (`cmp`) and predicates
/// (`setp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpCode {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl CmpCode {
    /// The code testing the opposite condition (NaN semantics are the
    /// ordered family throughout; era ISAs have no unordered forms).
    pub fn invert(self) -> CmpCode {
        match self {
            CmpCode::Lt => CmpCode::Ge,
            CmpCode::Le => CmpCode::Gt,
            CmpCode::Gt => CmpCode::Le,
            CmpCode::Ge => CmpCode::Lt,
            CmpCode::Eq => CmpCode::Ne,
            CmpCode::Ne => CmpCode::Eq,
        }
    }

    pub fn mnemonic(self) -> &'static str {
        match self {
            CmpCode::Lt => "lt",
            CmpCode::Le => "le",
            CmpCode::Gt => "gt",
            CmpCode::Ge => "ge",
            CmpCode::Eq => "eq",
            CmpCode::Ne => "ne",
        }
    }
}

/// ALU operation. Arithmetic/bitwise forms are lane-wise and typed by
/// their operands (`add` on float lanes is the float add, on i32 the
/// integer add); `dp3`/`dp4` reduce a vector to a scalar. Integer
/// division and shifts keep their signedness on the operation, like
/// MIR — register types do not carry sign.
#[derive(Debug, Clone, PartialEq)]
pub enum Alu {
    Mov,
    Add,
    Sub,
    Mul,
    /// Fused multiply-add: `dst = src0 * src1 + src2`.
    Mad,
    /// Integer division/remainder. Float division is `rcp` + `mul`
    /// (era ISAs have no divide).
    IDiv { unsigned: bool },
    IRem { unsigned: bool },
    Rcp,
    Rsq,
    Sqrt,
    Floor,
    Ceil,
    /// Fractional part (`frc`).
    Frac,
    Abs,
    Neg,
    Min,
    Max,
    /// Dot product of two float vectors (3 or 4 lanes) into a scalar.
    Dp3,
    Dp4,
    /// Value comparison: yields `0/1` per lane (`-1/0` mask when the
    /// operands are vectors, matching C vector semantics).
    Cmp(CmpCode),
    /// Integer width changes between the ≤32-bit widths prism allows.
    Zext,
    Sext,
    Trunc,
    /// Float↔integer conversions (C cast semantics: truncation toward
    /// zero, out-of-range undefined).
    IToF { unsigned: bool },
    FToI { unsigned: bool },
    /// Same-width reinterpretation (union punning).
    Bitcast,
    And,
    Or,
    Xor,
    Not,
    Shl,
    Shr { arithmetic: bool },
    /// `dst = src0 ? src1 : src2` (`src0` is an i32 0/1 scalar).
    Sel,
    /// Lane permutation over up to two same-element sources, MIR
    /// `ShuffleVector` semantics: lane `i` takes `lanes[i]` of the
    /// concatenated operands; a scalar second source acts as one lane
    /// (out-of-range indices select it).
    Shuffle { lanes: SmallVec<[u8; 8]> },
}

/// One ALU instruction: `dst = kind(srcs)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Op {
    pub dst: VReg,
    pub kind: Alu,
    pub srcs: SmallVec<[Src; 3]>,
}

/// Texture sample: `dst = sample(sampler, coord)`. Samples are pure —
/// shaders cannot write textures — so scheduling hoists them like any
/// other value.
#[derive(Debug, Clone, PartialEq)]
pub struct Tex {
    pub dst: VReg,
    pub sampler: u32,
    pub coord: Src,
    pub lod: Lod,
}

/// Sample LOD, per the plan: implicit (fragment pipeline derivative
/// LOD) or an explicit level (vertex texture fetch).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lod {
    Implicit,
    Explicit(f32),
}

/// Predicate definition: `dst = code(a, b)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Setp {
    pub dst: VReg,
    pub code: CmpCode,
    pub a: Src,
    pub b: Src,
}

/// One structured statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Op(Op),
    Tex(Tex),
    Setp(Setp),
    /// Two-sided branch. `results` lists the join registers when both
    /// arms flow to one continuation: `(join, then_value, else_value)`
    /// — the join register is the SSA phi, defined here. Empty when at
    /// least one arm leaves the region (break/continue/ret) and the
    /// continuation belongs to an enclosing region.
    If {
        pred: VReg,
        then: Body,
        els: Body,
        results: Vec<(VReg, VReg, VReg)>,
    },
    /// Conditionless loop: exit is `break`, the next iteration is
    /// `continue`. `carried` lists the per-iteration registers as
    /// `(register, init value)` — the register is the SSA phi defined
    /// here, read throughout the body; every `continue` supplies one
    /// value per carried register, in order. `results` are the exit
    /// registers, one value per `break` (in body order).
    Loop {
        carried: Vec<(VReg, VReg)>,
        body: Body,
        results: Vec<VReg>,
    },
    /// Leave the enclosing loop; values for the loop's exit registers,
    /// in order.
    Break(SmallVec<[VReg; 4]>),
    /// Next iteration; values for the loop's carried registers, in
    /// order.
    Continue(SmallVec<[VReg; 4]>),
    /// Leave the entry point; one value per declared output, in
    /// interface order.
    Exit(SmallVec<[VReg; 8]>),
}

/// A statement sequence (one region's body).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Body {
    pub stmts: Vec<Stmt>,
}

impl Body {
    pub fn new() -> Body {
        Body { stmts: Vec::new() }
    }
}

/// Shader stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Fragment,
}

impl Stage {
    pub fn mnemonic(self) -> &'static str {
        match self {
            Stage::Vertex => "vertex",
            Stage::Fragment => "fragment",
        }
    }
}

/// One compiled shader: the interface plus the structured body over
/// one virtual-register arena.
#[derive(Debug, Clone)]
pub struct Program {
    pub name: String,
    pub stage: Stage,
    pub interface: crate::interface::Interface,
    /// Every virtual register, in definition order. Predicates share
    /// this arena.
    pub regs: Vec<RegTy>,
    pub body: Body,
}

impl Program {
    pub fn reg_ty(&self, v: VReg) -> RegTy {
        self.regs[v.0 as usize].clone()
    }

    pub fn push_reg(&mut self, ty: RegTy) -> VReg {
        let v = VReg(self.regs.len() as u32);
        self.regs.push(ty);
        v
    }

    /// Record the type of a register parsed from text, where
    /// definitions may appear in any order (ids are absolute).
    pub fn set_reg_ty(&mut self, v: VReg, ty: RegTy) -> Result<(), String> {
        if (v.0 as usize) < self.regs.len() {
            self.regs[v.0 as usize] = ty;
            return Ok(());
        }
        if v.0 as usize >= 1 << 24 {
            return Err(format!("%{}: register id out of range", v.0));
        }
        self.regs.resize(v.0 as usize + 1, RegTy::Scalar(Scalar::F32));
        self.regs[v.0 as usize] = ty;
        Ok(())
    }
}
