use index_vec::IndexVec;
use smallvec::SmallVec;

use crate::debug::{DebugType, File, FunctionDebug, GlobalDebug, Location};
use crate::ty::{FloatKind, GepIndex, Signature, Ty};

index_vec::define_index_type! {
    pub struct Value = u32;
    MAX_INDEX = i32::MAX as usize;
}

index_vec::define_index_type! {
    pub struct Inst = u32;
    MAX_INDEX = i32::MAX as usize;
}

index_vec::define_index_type! {
    pub struct Block = u32;
    MAX_INDEX = i32::MAX as usize;
}

index_vec::define_index_type! {
    pub struct Func = u32;
    MAX_INDEX = i32::MAX as usize;
}

index_vec::define_index_type! {
    pub struct Global = u32;
    MAX_INDEX = i32::MAX as usize;
}

index_vec::define_index_type! {
    pub struct Extern = u32;
    MAX_INDEX = i32::MAX as usize;
}

/// Compile-time known value. Integers carry their width so folding stays
/// exact, and `value` is always the bit pattern sign-extended to `i128`
/// (see [`Const::canonical`]); floats are raw bits with their kind.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Const {
    Int {
        bits: u32,
        value: i128,
    },
    Float {
        kind: FloatKind,
        bits: u128,
    },
    Null,
    Undef,
    /// All-zero image of the value's type. Only meaningful where a type
    /// is known (global initializers); `ty()` cannot recover it.
    Zero,
    /// Raw byte image, interpreted under the value's type (string
    /// literals, folded aggregate initializers). Same typing caveat.
    Bytes(Vec<u8>),
}

impl Const {
    /// Sign-extend an integer constant to its declared width so `value`
    /// denotes exactly the bit pattern's value: folds that shift or
    /// divide by sign would otherwise read a wide literal like
    /// `0x8000_0000_0000_0000` as a positive `i128`. `Bool` (one bit)
    /// keeps the 0/1 convention instead of the sign interpretation.
    pub fn canonical(self) -> Const {
        match self {
            Const::Int { bits, value } if bits > 1 && bits < 128 => {
                let truncated = value as u128 & ((1u128 << bits) - 1);
                let shift = 128 - bits;
                Const::Int {
                    bits,
                    value: ((truncated << shift) as i128) >> shift,
                }
            }
            other => other,
        }
    }

    pub fn ty(&self) -> Ty {
        match self {
            Const::Int { bits, .. } => Ty::int(*bits),
            Const::Float { kind, .. } => Ty::Float(*kind),
            Const::Null => Ty::ptr(Ty::Void),
            Const::Undef => Ty::Void,
            Const::Zero | Const::Bytes(_) => {
                panic!("zero/bytes constants need an explicit type")
            }
        }
    }
}

/// How a value came to exist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Def {
    Inst(Inst),
    Param(Block, u32),
    Const(Const),
}

/// One SSA value with its type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueData {
    pub ty: Ty,
    pub def: Def,
}

/// Debug-only provenance of a value a pass materialized in place of
/// another: what the replaced definition looked like and where it was.
#[derive(Debug, Clone)]
pub struct ValueOrigin {
    /// Which pass materialized the value ("sccp", "const-fold", ...).
    pub pass: &'static str,
    /// How the replaced definition rendered at fold time, e.g. "add %3 i32(1)".
    pub text: String,
    /// Source location of the replaced definition, if it had one.
    pub loc: Option<crate::debug::Location>,
}

/// Integer comparison predicates. Signedness is explicit per op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IcmpPred {
    Eq,
    Ne,
    Ult,
    Ule,
    Ugt,
    Uge,
    Slt,
    Sle,
    Sgt,
    Sge,
}

/// Float comparison predicates, including unordered (NaN-aware) forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FcmpPred {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Uno,
    Ord,
}

/// Memory-access flags. Extensible record so future properties
/// (ordering, non-temporal hints) ride the same path as today's
/// single flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemFlags {
    /// C `volatile` access: passes must not eliminate, fold, forward
    /// through, or promote it away. Clif itself has no volatile memory
    /// flag (as of the pinned Cranelift), so lowering emits atomic
    /// memory ops for these to keep Cranelift's optimizer from eliding
    /// them.
    pub volatile: bool,
}

impl MemFlags {
    pub fn plain() -> Self {
        MemFlags { volatile: false }
    }

    pub fn vol() -> Self {
        MemFlags { volatile: true }
    }
}

/// One operand of an inline-asm block, in GCC numbering: outputs
/// first, then inputs.
#[derive(Debug, Clone)]
pub struct AsmOperand {
    /// Inputs: the value to place in the operand's register or slot.
    /// Outputs: the address the result is stored to.
    pub value: Value,
    /// Raw constraint text (`"="`/`"+"`/`"&"` prefix plus letters or a
    /// tie digit).
    pub constraint: String,
    /// Operand type: register load/store width and slot sizing follow
    /// it.
    pub ty: Ty,
    /// Input tied to this output index (`"0"`): the two share one
    /// location.
    pub tie: Option<u32>,
    /// `[name]` spelling, for `%[name]` template references.
    pub name: Option<String>,
}

/// One inline-asm block: template plus operands, clobbers, and
/// volatility. Lowering calls a generated wrapper
/// (`glcc_lower_cl::asm`), so the template never reaches the register
/// allocator.
#[derive(Debug, Clone)]
pub struct AsmBlock {
    pub template: String,
    /// GCC numbering: `operands[..outputs]` are outputs.
    pub operands: Vec<AsmOperand>,
    pub outputs: u32,
    pub clobbers: Vec<String>,
    /// `asm goto` label names, in selector order: the wrapper returns
    /// `0` for the fallthrough and `index + 1` when the template jumps
    /// to `labels[index]`.
    pub labels: Vec<String>,
    pub is_volatile: bool,
    /// Basic asm (no colon in the statement): the template passes
    /// through verbatim, with `%` literal and no operand substitution.
    pub is_basic: bool,
}

/// Non-terminator instruction. Each produces at most one value; memory and
/// calls carry their side effects implicitly.
#[derive(Debug, Clone)]
pub enum InstKind {
    Add(Value, Value),
    Sub(Value, Value),
    Mul(Value, Value),
    Udiv(Value, Value),
    Sdiv(Value, Value),
    Urem(Value, Value),
    Srem(Value, Value),
    And(Value, Value),
    Or(Value, Value),
    Xor(Value, Value),
    Shl(Value, Value),
    LShr(Value, Value),
    AShr(Value, Value),
    Neg(Value),
    Not(Value),
    FAdd(Value, Value),
    FSub(Value, Value),
    FMul(Value, Value),
    FDiv(Value, Value),
    FNeg(Value),
    Icmp {
        pred: IcmpPred,
        lhs: Value,
        rhs: Value,
    },
    Fcmp {
        pred: FcmpPred,
        lhs: Value,
        rhs: Value,
    },
    Trunc(Value),
    Zext(Value),
    Sext(Value),
    Bitcast(Value),
    SItoF(Value),
    UItoF(Value),
    SFtoI(Value),
    UFtoI(Value),
    Fptrunc(Value),
    Fpext(Value),
    PtrToInt(Value),
    IntToPtr(Value),
    Select {
        cond: Value,
        then_val: Value,
        else_val: Value,
    },
    Alloca {
        pointee: Ty,
        count: u64,
    },
    Load {
        addr: Value,
        flags: MemFlags,
    },
    Store {
        addr: Value,
        val: Value,
        flags: MemFlags,
    },
    GetElementPtr {
        base: Value,
        indices: SmallVec<[GepIndex; 4]>,
    },
    ExtractValue {
        agg: Value,
        index: u32,
    },
    InsertValue {
        agg: Value,
        elem: Value,
        index: u32,
    },
    Memcpy {
        dst: Value,
        src: Value,
        len: u64,
    },
    /// Fill `len` bytes at `dst` with `byte`. Unrolled at lowering like
    /// `Memcpy`; covers zero-initialization without a libc call.
    Memset {
        dst: Value,
        byte: u8,
        len: u64,
    },
    /// Address of a module function. Lowers to `func_addr`.
    FuncAddr {
        func: Func,
    },
    /// Address of an imported function. Lowers to `func_addr` on the import.
    ExternAddr {
        target: Extern,
    },
    /// Initialize a `va_list` from the current function's fixed
    /// arguments. The register save area and overflow base come from
    /// the function's own signature, so this needs no operands beyond
    /// the list address.
    VaStart {
        ap: Value,
    },
    /// Read the next variadic argument of the result type from a
    /// `va_list`, advancing it. Aggregates arrive in a fresh slot.
    VaArg {
        ap: Value,
    },
    /// Tear down a `va_list`. No code on any current target.
    VaEnd {
        ap: Value,
    },
    /// Copy one `va_list` object onto another.
    VaCopy {
        dst: Value,
        src: Value,
    },
    /// Address of a module global. Lowers to `symbol_value`.
    GlobalAddr {
        global: Global,
    },
    Call {
        callee: Func,
        args: SmallVec<[Value; 4]>,
    },
    /// Indirect call through a code address. The callee signature
    /// rides along for call lowering (per-call Clif signature, like
    /// foreign calls); variadic callees take the platform's indirect
    /// path there.
    CallIndirect {
        callee: Value,
        args: SmallVec<[Value; 4]>,
        sig: Signature,
    },
    /// Call to a function defined outside this module (C library,
    /// runtime helpers). Lowered to an imported symbol; variadic
    /// foreign calls take the platform's indirect path.
    CallExtern {
        target: Extern,
        args: SmallVec<[Value; 4]>,
    },
    /// `__builtin_await(fut)` (glcc async extension): suspend the
    /// enclosing coroutine until the future handle in `fut` completes,
    /// producing its result. Valid only inside a function whose
    /// `coroutine` flag is set; the coroutine pass rewrites every
    /// occurrence (poll loop, frame state save/restore) before codegen,
    /// so it must never reach `glcc_lower_cl`.
    Await {
        fut: Value,
    },
    /// Atomic load of a scalar. Orders are ignored (sequentially
    /// consistent, like the legacy backend).
    AtomicLoad {
        addr: Value,
    },
    /// Atomic store of a scalar value.
    AtomicStore {
        addr: Value,
        val: Value,
    },
    /// Atomic read-modify-write: returns the previous value.
    AtomicRmw {
        op: AtomicRmwOp,
        addr: Value,
        val: Value,
    },
    /// Atomic compare-and-swap: stores `replacement` iff `*addr` equals
    /// `expected`, and returns the previous value (compare it against
    /// `expected` for success).
    AtomicCas {
        addr: Value,
        expected: Value,
        replacement: Value,
    },
    /// Full memory fence.
    AtomicFence,
    /// Inline assembler block. Operand values are the block's inputs
    /// and its outputs' addresses; the instruction is side-effecting
    /// and produces no value (outputs land in memory). Lowered as a
    /// call to a generated out-of-line wrapper, so the template never
    /// reaches the register allocator.
    InlineAsm {
        block: Box<AsmBlock>,
    },
    /// Address held in the frame pointer register (`__builtin_frame_address`
    /// level 0). Requires the `preserve_frame_pointers` setting.
    GetFramePointer,
    /// Round a float up (`__builtin_ceil`).
    Ceil(Value),
    /// Round a float down (`__builtin_floor`).
    Floor(Value),
    /// Square root (`cabs` magnitudes lower through this).
    Sqrt(Value),
    /// Count leading zero bits at the operand width (`__builtin_clz*`).
    Clz(Value),
    /// Count trailing zero bits (`__builtin_ctz*`).
    Ctz(Value),
    /// Count set bits (`__builtin_popcount*`).
    Popcnt(Value),
    /// Reverse bytes (`__builtin_bswap*`).
    Bswap(Value),
}

/// Atomic read-modify-write operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicRmwOp {
    Add,
    Sub,
    And,
    Nand,
    Or,
    Xor,
    Xchg,
    Umin,
    Umax,
    Smin,
    Smax,
}

impl InstKind {
    /// Call `f` for every operand without allocating, for hot use-counting.
    pub fn for_each_operand(&self, f: &mut impl FnMut(Value)) {
        match self {
            InstKind::Add(a, b)
            | InstKind::Sub(a, b)
            | InstKind::Mul(a, b)
            | InstKind::Udiv(a, b)
            | InstKind::Sdiv(a, b)
            | InstKind::Urem(a, b)
            | InstKind::Srem(a, b)
            | InstKind::And(a, b)
            | InstKind::Or(a, b)
            | InstKind::Xor(a, b)
            | InstKind::Shl(a, b)
            | InstKind::LShr(a, b)
            | InstKind::AShr(a, b)
            | InstKind::FAdd(a, b)
            | InstKind::FSub(a, b)
            | InstKind::FMul(a, b)
            | InstKind::FDiv(a, b) => {
                f(*a);
                f(*b);
            }
            InstKind::Neg(a)
            | InstKind::Not(a)
            | InstKind::FNeg(a)
            | InstKind::Trunc(a)
            | InstKind::Zext(a)
            | InstKind::Sext(a)
            | InstKind::Bitcast(a)
            | InstKind::SItoF(a)
            | InstKind::UItoF(a)
            | InstKind::SFtoI(a)
            | InstKind::UFtoI(a)
            | InstKind::Fptrunc(a)
            | InstKind::Fpext(a)
            | InstKind::PtrToInt(a)
            | InstKind::IntToPtr(a)
            | InstKind::Ceil(a)
            | InstKind::Floor(a)
            | InstKind::Sqrt(a)
            | InstKind::Clz(a)
            | InstKind::Ctz(a)
            | InstKind::Popcnt(a)
            | InstKind::Bswap(a) => f(*a),
            InstKind::Icmp { lhs, rhs, .. } | InstKind::Fcmp { lhs, rhs, .. } => {
                f(*lhs);
                f(*rhs);
            }
            InstKind::Select { cond, then_val, else_val } => {
                f(*cond);
                f(*then_val);
                f(*else_val);
            }
            InstKind::Alloca { .. } => {}
            InstKind::Load { addr, .. } => f(*addr),
            InstKind::Store { addr, val, .. } => {
                f(*addr);
                f(*val);
            }
            InstKind::GetElementPtr { base, indices } => {
                f(*base);
                for idx in indices.iter() {
                    if let GepIndex::Index(v) = idx {
                        f(*v);
                    }
                }
            }
            InstKind::ExtractValue { agg, .. } => f(*agg),
            InstKind::InsertValue { agg, elem, .. } => {
                f(*agg);
                f(*elem);
            }
            InstKind::Memcpy { dst, src, .. } => {
                f(*dst);
                f(*src);
            }
            InstKind::Memset { dst, .. } => f(*dst),
            InstKind::FuncAddr { .. } | InstKind::ExternAddr { .. } => {}
            InstKind::VaStart { ap } | InstKind::VaArg { ap } | InstKind::VaEnd { ap } => f(*ap),
            InstKind::VaCopy { dst, src } => {
                f(*dst);
                f(*src);
            }
            InstKind::GlobalAddr { .. } => {}
            InstKind::Call { args, .. } | InstKind::CallExtern { args, .. } => {
                for &a in args.iter() {
                    f(a);
                }
            }
            InstKind::Await { fut } => f(*fut),
            InstKind::CallIndirect { callee, args, .. } => {
                f(*callee);
                for &a in args.iter() {
                    f(a);
                }
            }
            InstKind::AtomicLoad { addr } => f(*addr),
            InstKind::AtomicStore { addr, val } => {
                f(*addr);
                f(*val);
            }
            InstKind::AtomicRmw { addr, val, .. } => {
                f(*addr);
                f(*val);
            }
            InstKind::AtomicCas { addr, expected, replacement } => {
                f(*addr);
                f(*expected);
                f(*replacement);
            }
            InstKind::InlineAsm { block } => {
                for operand in &block.operands {
                    f(operand.value);
                }
            }
            InstKind::AtomicFence | InstKind::GetFramePointer => {}
        }
    }

    /// Operands read by this instruction, for use-counting and display.
    pub fn operands(&self) -> SmallVec<[Value; 4]> {
        let mut out = SmallVec::new();
        self.for_each_operand(&mut |v| out.push(v));
        out
    }

    /// Pure when removing it cannot change observable behavior: no memory
    /// access, no calls, no potentially trapping arithmetic (division,
    /// float-to-int conversion). The `Va*` state machine walks its argument
    /// list on every operation, so it is not pure even though it touches no
    /// user memory.
    pub fn is_pure(&self) -> bool {
        matches!(
            self,
            InstKind::Add(..)
                | InstKind::Sub(..)
                | InstKind::Mul(..)
                | InstKind::And(..)
                | InstKind::Or(..)
                | InstKind::Xor(..)
                | InstKind::Shl(..)
                | InstKind::LShr(..)
                | InstKind::AShr(..)
                | InstKind::Neg(_)
                | InstKind::Not(_)
                | InstKind::FAdd(..)
                | InstKind::FSub(..)
                | InstKind::FMul(..)
                | InstKind::FDiv(..)
                | InstKind::FNeg(_)
                | InstKind::Icmp { .. }
                | InstKind::Fcmp { .. }
                | InstKind::Trunc(_)
                | InstKind::Zext(_)
                | InstKind::Sext(_)
                | InstKind::Bitcast(_)
                | InstKind::SItoF(_)
                | InstKind::UItoF(_)
                | InstKind::Fptrunc(_)
                | InstKind::Fpext(_)
                | InstKind::PtrToInt(_)
                | InstKind::IntToPtr(_)
                | InstKind::Ceil(_)
                | InstKind::Floor(_)
                | InstKind::Sqrt(_)
                | InstKind::Clz(_)
                | InstKind::Ctz(_)
                | InstKind::Popcnt(_)
                | InstKind::Bswap(_)
                | InstKind::Select { .. }
                | InstKind::GetFramePointer
                | InstKind::GlobalAddr { .. }
                | InstKind::FuncAddr { .. }
                | InstKind::ExternAddr { .. }
                | InstKind::GetElementPtr { .. }
                | InstKind::ExtractValue { .. }
                | InstKind::InsertValue { .. }
        )
    }
}

/// Block terminator. Branches carry arguments for the target block params,
/// mirroring Cranelift so lowering is a direct transcription.
#[derive(Debug, Clone)]
pub enum Terminator {
    Br {
        dest: Block,
        args: SmallVec<[Value; 4]>,
    },
    CondBr {
        cond: Value,
        then_block: Block,
        else_block: Block,
        then_args: SmallVec<[Value; 4]>,
        else_args: SmallVec<[Value; 4]>,
    },
    Switch {
        scrutinee: Value,
        cases: Vec<(i128, Block)>,
        default: Block,
    },
    Return {
        val: Option<Value>,
    },
    Unreachable,
}

impl Terminator {
    pub fn successors(&self) -> SmallVec<[Block; 2]> {
        match self {
            Terminator::Br { dest, .. } => [*dest].into_iter().collect(),
            Terminator::CondBr { then_block, else_block, .. } => [*then_block, *else_block].into_iter().collect(),
            Terminator::Switch { cases, default, .. } => cases.iter().map(|(_, b)| *b).chain([*default]).collect(),
            Terminator::Return { .. } | Terminator::Unreachable => SmallVec::new(),
        }
    }

    pub fn operands(&self) -> SmallVec<[Value; 4]> {
        let mut out = SmallVec::new();
        self.for_each_operand(&mut |v| out.push(v));
        out
    }

    /// Call `f` for every operand without allocating.
    pub fn for_each_operand(&self, f: &mut impl FnMut(Value)) {
        match self {
            Terminator::Br { args, .. } => {
                for &a in args.iter() {
                    f(a);
                }
            }
            Terminator::CondBr { cond, then_args, else_args, .. } => {
                f(*cond);
                for &a in then_args.iter() {
                    f(a);
                }
                for &a in else_args.iter() {
                    f(a);
                }
            }
            Terminator::Switch { scrutinee, .. } => f(*scrutinee),
            Terminator::Return { val } => {
                if let Some(v) = val {
                    f(*v);
                }
            }
            Terminator::Unreachable => {}
        }
    }
}

#[derive(Debug, Clone)]
pub struct InstData {
    pub block: Block,
    pub kind: InstKind,
    pub result: Option<Value>,
    /// Source position the instruction was lowered from, when known.
    /// Passes that clone instructions carry it along for free; passes
    /// that drop instructions drop their locations with them.
    pub loc: Option<Location>,
}

#[derive(Debug, Clone, Default)]
pub struct BlockData {
    pub params: Vec<Value>,
    pub insts: Vec<Inst>,
    pub term: Option<Terminator>,
}

/// One function with its own value/inst/block arenas.
/// `local` is internal linkage (`static`): emitted `Local` so
/// definitions never collide across translation units.
/// `weak` is a weak definition (`__attribute__((weak))`, C++ vague
/// linkage from `extern inline`): emitted `Preemptible` so a strong
/// definition in another TU wins and duplicates never error.
/// `inline` is the C `inline` hint from the frontend: the `inline`
/// pass uses a larger budget for marked callees and may inline
/// multi-block bodies that heuristic inlining refuses.
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub sig: Signature,
    pub local: bool,
    pub weak: bool,
    pub inline: bool,
    /// `__attribute__((noinline))`: the inline pass never inlines this
    /// function at its call sites.
    pub noinline: bool,
    /// Contains a call site that may return twice (`setjmp`): `longjmp`
    /// re-enters the function there with the registers of the first
    /// return, so slot promotion must leave locals in memory.
    pub returns_twice: bool,
    /// The definition carries `__attribute__((async))` (glcc
    /// extension): its body is a stackless coroutine producing a
    /// result of this type (`Void` for `_Future(void)`). The public
    /// symbol starts the coroutine (allocating the boxed frame,
    /// returning the `_Future(T)` handle); a synthesized
    /// `<name>$resume` holds the body. `LowerCoroutines` consumes the
    /// flag (and every `Await`) and clears it — the emitted functions
    /// are plain.
    pub coroutine: Option<Ty>,
    /// Debug record: declaration, parameters, variables, and lexical
    /// blocks. `Some` whenever the function was lowered from source
    /// with debug collection enabled.
    pub debug: Option<FunctionDebug>,
    pub values: IndexVec<Value, ValueData>,
    pub insts: IndexVec<Inst, InstData>,
    pub blocks: IndexVec<Block, BlockData>,
    pub entry: Block,
    /// Debug-only: provenance of constants a pass materialized, keyed by
    /// the fresh value. Empty outside debug-oriented dumps.
    pub origins: std::collections::HashMap<Value, ValueOrigin>,
    /// Memory behavior of each module extern, indexed by [`Extern`].
    /// [`PassManager`](crate::pass::PassManager) fills this before
    /// function passes; a missing entry means [`ExternEffect::clobber`].
    pub(crate) extern_effects: Vec<ExternEffect>,
}

impl Function {
    pub fn new(name: &str, sig: Signature) -> (Self, Block) {
        let mut func = Function {
            name: name.to_string(),
            sig,
            local: false,
            weak: false,
            inline: false,
            noinline: false,
            returns_twice: false,
            coroutine: None,
            debug: None,
            values: IndexVec::new(),
            insts: IndexVec::new(),
            blocks: IndexVec::new(),
            entry: Block::from_usize(0),
            origins: std::collections::HashMap::new(),
            extern_effects: Vec::new(),
        };
        let entry = func.push_block(vec![]);
        func.entry = entry;
        (func, entry)
    }

    pub(crate) fn set_extern_effects(&mut self, effects: &[ExternEffect]) {
        self.extern_effects.clear();
        self.extern_effects.extend_from_slice(effects);
    }

    pub(crate) fn extern_effect(&self, target: Extern) -> ExternEffect {
        self.extern_effects.get(target.index()).copied().unwrap_or_default()
    }

    pub fn push_block(&mut self, param_tys: Vec<Ty>) -> Block {
        let block = self.blocks.push(BlockData::default());
        for (i, ty) in param_tys.into_iter().enumerate() {
            let v = self.values.push(ValueData {
                ty,
                def: Def::Param(block, i as u32),
            });
            self.blocks[block].params.push(v);
        }
        block
    }

    pub fn push_const(&mut self, c: Const) -> Value {
        let c = c.canonical();
        let ty = c.ty();
        self.values.push(ValueData { ty, def: Def::Const(c) })
    }

    pub fn const_value(&self, v: Value) -> Option<&Const> {
        match &self.values[v].def {
            Def::Const(c) => Some(c),
            _ => None,
        }
    }

    pub fn use_count(&self, v: Value) -> usize {
        let mut n = 0;
        for inst in self.insts.iter() {
            inst.kind.for_each_operand(&mut |o| {
                if o == v {
                    n += 1;
                }
            });
        }
        for block in self.blocks.iter() {
            if let Some(term) = &block.term {
                term.for_each_operand(&mut |o| {
                    if o == v {
                        n += 1;
                    }
                });
            }
        }
        n
    }

    /// Count every use once, for passes that query many values.
    pub fn count_all_uses(&self) -> std::collections::HashMap<Value, usize> {
        let mut counts = std::collections::HashMap::new();
        for inst in self.insts.iter() {
            inst.kind.for_each_operand(&mut |o| {
                *counts.entry(o).or_insert(0) += 1;
            });
        }
        for block in self.blocks.iter() {
            if let Some(term) = &block.term {
                term.for_each_operand(&mut |o| {
                    *counts.entry(o).or_insert(0) += 1;
                });
            }
        }
        counts
    }

    /// Count uses from instructions and terminators still present in
    /// blocks, indexed by value. Arena-orphaned instructions (unlinked
    /// by an earlier pass) do not count: they cannot execute, and
    /// letting them keep their operands alive would defeat dead-code
    /// elimination after any pass that removes instructions from blocks.
    pub fn count_resident_uses(&self) -> Vec<u32> {
        let mut counts = vec![0u32; self.values.len()];
        for block in self.blocks.iter() {
            for inst in &block.insts {
                self.insts[*inst].kind.for_each_operand(&mut |o| counts[o.index()] += 1);
            }
            if let Some(term) = &block.term {
                term.for_each_operand(&mut |o| counts[o.index()] += 1);
            }
        }
        counts
    }

    /// Which arena entries are still listed in a block, indexed by
    /// instruction. Cheaper than a `HashSet<Inst>` for membership tests.
    pub fn resident_mask(&self) -> Vec<bool> {
        let mut mask = vec![false; self.insts.len()];
        for block in self.blocks.iter() {
            for &inst in &block.insts {
                mask[inst.index()] = true;
            }
        }
        mask
    }

    /// Replace every operand occurrence of `old` with `new`.
    ///
    /// Every arena entry is visited because some passes temporarily
    /// detach a block's instruction list while rewriting it; rebuilding
    /// an `InstKind` is avoided unless `old` is actually an operand.
    pub fn replace_uses(&mut self, old: Value, new: Value) {
        if old == new {
            return;
        }
        for inst in self.insts.iter_mut() {
            let mut uses_old = false;
            inst.kind.for_each_operand(&mut |v| {
                if v == old {
                    uses_old = true;
                }
            });
            if uses_old {
                inst.kind = remap_inst(&inst.kind, |v| if v == old { new } else { v });
            }
        }
        for block in self.blocks.iter_mut() {
            if let Some(term) = block.term.take() {
                block.term = Some(remap_term(&term, |v| if v == old { new } else { v }));
            }
        }
    }

    /// Rewrite every operand through `map` in one sweep over the live
    /// blocks and their terminators; values absent from the map stay
    /// unchanged. The map is applied once, so chains (`a -> b`,
    /// `b -> c`) must be collapsed by the caller.
    ///
    /// Callers hold all block lists attached; orphaned arena entries
    /// cannot be used and are left alone, which keeps the sweep
    /// proportional to live code.
    pub fn remap_operands(&mut self, map: &std::collections::HashMap<Value, Value>) {
        if map.is_empty() {
            return;
        }
        let remap = |v: Value| map.get(&v).copied().unwrap_or(v);
        let insts = &mut self.insts;
        for block in self.blocks.iter_mut() {
            for &id in &block.insts {
                let inst = &mut insts[id];
                let mut hit = false;
                inst.kind.for_each_operand(&mut |v| hit |= map.contains_key(&v));
                if hit {
                    inst.kind = remap_inst(&inst.kind, remap);
                }
            }
            if let Some(term) = block.term.take() {
                let mut hit = false;
                term.for_each_operand(&mut |v| hit |= map.contains_key(&v));
                block.term = Some(if hit { remap_term(&term, remap) } else { term });
            }
        }
    }

    /// Reachable blocks from the entry following terminator edges.
    pub fn reachable_blocks(&self) -> Vec<bool> {
        let mut seen = vec![false; self.blocks.len()];
        if self.blocks.is_empty() {
            return seen;
        }
        let mut stack = vec![self.entry];
        seen[self.entry.index()] = true;
        while let Some(b) = stack.pop() {
            if let Some(term) = &self.blocks[b].term {
                for s in term.successors() {
                    if s.index() < seen.len() && !seen[s.index()] {
                        seen[s.index()] = true;
                        stack.push(s);
                    }
                }
            }
        }
        seen
    }
}

pub(crate) fn remap_inst(kind: &InstKind, swap: impl Fn(Value) -> Value) -> InstKind {
    match kind {
        InstKind::Add(a, b) => InstKind::Add(swap(*a), swap(*b)),
        InstKind::Sub(a, b) => InstKind::Sub(swap(*a), swap(*b)),
        InstKind::Mul(a, b) => InstKind::Mul(swap(*a), swap(*b)),
        InstKind::Udiv(a, b) => InstKind::Udiv(swap(*a), swap(*b)),
        InstKind::Sdiv(a, b) => InstKind::Sdiv(swap(*a), swap(*b)),
        InstKind::Urem(a, b) => InstKind::Urem(swap(*a), swap(*b)),
        InstKind::Srem(a, b) => InstKind::Srem(swap(*a), swap(*b)),
        InstKind::And(a, b) => InstKind::And(swap(*a), swap(*b)),
        InstKind::Or(a, b) => InstKind::Or(swap(*a), swap(*b)),
        InstKind::Xor(a, b) => InstKind::Xor(swap(*a), swap(*b)),
        InstKind::Shl(a, b) => InstKind::Shl(swap(*a), swap(*b)),
        InstKind::LShr(a, b) => InstKind::LShr(swap(*a), swap(*b)),
        InstKind::AShr(a, b) => InstKind::AShr(swap(*a), swap(*b)),
        InstKind::Neg(a) => InstKind::Neg(swap(*a)),
        InstKind::Not(a) => InstKind::Not(swap(*a)),
        InstKind::FAdd(a, b) => InstKind::FAdd(swap(*a), swap(*b)),
        InstKind::FSub(a, b) => InstKind::FSub(swap(*a), swap(*b)),
        InstKind::FMul(a, b) => InstKind::FMul(swap(*a), swap(*b)),
        InstKind::FDiv(a, b) => InstKind::FDiv(swap(*a), swap(*b)),
        InstKind::FNeg(a) => InstKind::FNeg(swap(*a)),
        InstKind::Ceil(a) => InstKind::Ceil(swap(*a)),
        InstKind::Floor(a) => InstKind::Floor(swap(*a)),
        InstKind::Sqrt(a) => InstKind::Sqrt(swap(*a)),
        InstKind::Clz(a) => InstKind::Clz(swap(*a)),
        InstKind::Ctz(a) => InstKind::Ctz(swap(*a)),
        InstKind::Popcnt(a) => InstKind::Popcnt(swap(*a)),
        InstKind::Bswap(a) => InstKind::Bswap(swap(*a)),
        InstKind::Icmp { pred, lhs, rhs } => InstKind::Icmp {
            pred: *pred,
            lhs: swap(*lhs),
            rhs: swap(*rhs),
        },
        InstKind::Fcmp { pred, lhs, rhs } => InstKind::Fcmp {
            pred: *pred,
            lhs: swap(*lhs),
            rhs: swap(*rhs),
        },
        InstKind::Trunc(a) => InstKind::Trunc(swap(*a)),
        InstKind::Zext(a) => InstKind::Zext(swap(*a)),
        InstKind::Sext(a) => InstKind::Sext(swap(*a)),
        InstKind::Bitcast(a) => InstKind::Bitcast(swap(*a)),
        InstKind::SItoF(a) => InstKind::SItoF(swap(*a)),
        InstKind::UItoF(a) => InstKind::UItoF(swap(*a)),
        InstKind::SFtoI(a) => InstKind::SFtoI(swap(*a)),
        InstKind::UFtoI(a) => InstKind::UFtoI(swap(*a)),
        InstKind::Fptrunc(a) => InstKind::Fptrunc(swap(*a)),
        InstKind::Fpext(a) => InstKind::Fpext(swap(*a)),
        InstKind::PtrToInt(a) => InstKind::PtrToInt(swap(*a)),
        InstKind::IntToPtr(a) => InstKind::IntToPtr(swap(*a)),
        InstKind::Select { cond, then_val, else_val } => InstKind::Select {
            cond: swap(*cond),
            then_val: swap(*then_val),
            else_val: swap(*else_val),
        },
        InstKind::Alloca { pointee, count } => InstKind::Alloca {
            pointee: pointee.clone(),
            count: *count,
        },
        InstKind::Load { addr, flags } => InstKind::Load { addr: swap(*addr), flags: *flags },
        InstKind::Store { addr, val, flags } => InstKind::Store {
            addr: swap(*addr),
            val: swap(*val),
            flags: *flags,
        },
        InstKind::GetElementPtr { base, indices } => {
            let mut next = SmallVec::new();
            for idx in indices.iter() {
                match idx {
                    GepIndex::Index(v) => next.push(GepIndex::Index(swap(*v))),
                    other => next.push(other.clone()),
                }
            }
            InstKind::GetElementPtr { base: swap(*base), indices: next }
        }
        InstKind::ExtractValue { agg, index } => InstKind::ExtractValue { agg: swap(*agg), index: *index },
        InstKind::InsertValue { agg, elem, index } => InstKind::InsertValue {
            agg: swap(*agg),
            elem: swap(*elem),
            index: *index,
        },
        InstKind::Memcpy { dst, src, len } => InstKind::Memcpy {
            dst: swap(*dst),
            src: swap(*src),
            len: *len,
        },
        InstKind::GlobalAddr { global } => InstKind::GlobalAddr { global: *global },
        InstKind::Memset { dst, byte, len } => InstKind::Memset {
            dst: swap(*dst),
            byte: *byte,
            len: *len,
        },
        InstKind::FuncAddr { func } => InstKind::FuncAddr { func: *func },
        InstKind::ExternAddr { target } => InstKind::ExternAddr { target: *target },
        InstKind::VaStart { ap } => InstKind::VaStart { ap: swap(*ap) },
        InstKind::VaArg { ap } => InstKind::VaArg { ap: swap(*ap) },
        InstKind::VaEnd { ap } => InstKind::VaEnd { ap: swap(*ap) },
        InstKind::VaCopy { dst, src } => InstKind::VaCopy { dst: swap(*dst), src: swap(*src) },
        InstKind::Call { callee, args } => InstKind::Call {
            callee: *callee,
            args: args.iter().map(|&a| swap(a)).collect(),
        },
        InstKind::CallIndirect { callee, args, sig } => InstKind::CallIndirect {
            callee: swap(*callee),
            args: args.iter().map(|&a| swap(a)).collect(),
            sig: sig.clone(),
        },
        InstKind::CallExtern { target, args } => InstKind::CallExtern {
            target: *target,
            args: args.iter().map(|&a| swap(a)).collect(),
        },
        InstKind::Await { fut } => InstKind::Await { fut: swap(*fut) },
        InstKind::AtomicLoad { addr } => InstKind::AtomicLoad { addr: swap(*addr) },
        InstKind::InlineAsm { block } => {
            let mut next = block.clone();
            for operand in &mut next.operands {
                operand.value = swap(operand.value);
            }
            InstKind::InlineAsm { block: next }
        }
        InstKind::AtomicStore { addr, val } => InstKind::AtomicStore {
            addr: swap(*addr),
            val: swap(*val),
        },
        InstKind::AtomicRmw { op, addr, val } => InstKind::AtomicRmw {
            op: *op,
            addr: swap(*addr),
            val: swap(*val),
        },
        InstKind::AtomicCas { addr, expected, replacement } => InstKind::AtomicCas {
            addr: swap(*addr),
            expected: swap(*expected),
            replacement: swap(*replacement),
        },
        InstKind::AtomicFence => InstKind::AtomicFence,
        InstKind::GetFramePointer => InstKind::GetFramePointer,
    }
}

pub(crate) fn remap_term(term: &Terminator, swap: impl Fn(Value) -> Value) -> Terminator {
    match term {
        Terminator::Br { dest, args } => Terminator::Br {
            dest: *dest,
            args: args.iter().map(|&a| swap(a)).collect(),
        },
        Terminator::CondBr {
            cond,
            then_block,
            else_block,
            then_args,
            else_args,
        } => Terminator::CondBr {
            cond: swap(*cond),
            then_block: *then_block,
            else_block: *else_block,
            then_args: then_args.iter().map(|&a| swap(a)).collect(),
            else_args: else_args.iter().map(|&a| swap(a)).collect(),
        },
        Terminator::Switch { scrutinee, cases, default } => Terminator::Switch {
            scrutinee: swap(*scrutinee),
            cases: cases.clone(),
            default: *default,
        },
        Terminator::Return { val } => Terminator::Return { val: val.map(swap) },
        Terminator::Unreachable => Terminator::Unreachable,
    }
}

/// A relocation inside a global byte image: the address of module
/// storage patched at link time over the zeroed image.
#[derive(Debug, Clone)]
pub enum DataReloc {
    /// Address of a module global (defined or imported) plus a byte
    /// addend (`&arr[i]`, `&s.member`, `&x + k`).
    Global { target: Global, addend: i64 },
    /// Address of a defined module function.
    Func { target: Func },
    /// Address of an imported function.
    FuncExtern { target: Extern },
}

/// Module-level global with an optional constant initializer.
/// `external` names storage defined elsewhere (a pure `extern`
/// declaration): declared as an import with no image, the data twin
/// of a prototype-only `Extern` function. `local` is internal
/// linkage (`static`, block-statics, and anonymous objects): emitted
/// `Local` so definitions never collide across translation units.
/// `weak` is a weak definition (ELF `STB_WEAK`), emitted
/// `Preemptible` like a weak function. `tls` is thread-local storage
/// (`_Thread_local`/`__thread`): declared as a TLS symbol and
/// addressed through `tls_value`, never `symbol_value`. `align` is
/// the object's required alignment (`type_align`), the `set_align`
/// every definition needs. `section` overrides the default section
/// for linker-collected data (`.init_array`/`.fini_array`).
/// `writable` is false for objects the program must never modify
/// (string literals): the backend places them in read-only storage.
#[derive(Debug, Clone)]
pub struct GlobalData {
    pub name: String,
    pub ty: Ty,
    pub init: Option<Const>,
    pub external: bool,
    pub local: bool,
    pub weak: bool,
    pub tls: bool,
    pub writable: bool,
    pub align: Option<u64>,
    pub section: Option<String>,
    /// Link-time patches over the image (`Const::Bytes` only), at
    /// ordinary pointer width.
    pub relocs: Vec<(u64, DataReloc)>,
    /// Link-time patches that write only 32 bits: a `__ptr32` pointer
    /// field whose initializer is an address. Kept apart from
    /// `relocs` because the backend's default data relocation is
    /// pointer-width and cannot express a 4-byte patch.
    pub relocs32: Vec<(u64, DataReloc)>,
    /// Debug record for module-level objects; `None` for synthesized
    /// storage (string literals, compound literals, import shims).
    pub debug: Option<GlobalDebug>,
}

/// How a call to an external function may touch program-visible memory.
///
/// The default is deliberately the most conservative answer: an import
/// may read and write any memory the program can name. Frontends that
/// know more about a runtime helper can opt it into narrower behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternEffect {
    /// May read program-visible memory.
    pub reads: bool,
    /// May write program-visible memory.
    pub writes: bool,
    /// A pointer result is a fresh allocation that cannot alias any
    /// object that was live at the call.
    pub allocates: bool,
}

impl ExternEffect {
    /// An ordinary external call: it may read or write anything.
    pub const fn clobber() -> Self {
        Self {
            reads: true,
            writes: true,
            allocates: false,
        }
    }

    /// The call has no effect on program-visible memory. It may still
    /// have other side effects, so it is not necessarily removable.
    pub const fn neutral() -> Self {
        Self {
            reads: false,
            writes: false,
            allocates: false,
        }
    }

    /// The call may read program-visible memory but never writes it.
    pub const fn read_only() -> Self {
        Self {
            reads: true,
            writes: false,
            allocates: false,
        }
    }

    /// The call is an allocator: it may inspect existing memory (for
    /// example, a collector tracing roots) but it does not modify it.
    /// A pointer result is fresh and does not alias an existing object.
    pub const fn allocator() -> Self {
        Self {
            reads: true,
            writes: false,
            allocates: true,
        }
    }
}

impl Default for ExternEffect {
    fn default() -> Self {
        Self::clobber()
    }
}

/// A function defined outside this module: name plus signature for
/// import. Variadic imports take the platform foreign-variadic path.
#[derive(Debug, Clone)]
pub struct ExternData {
    pub name: String,
    pub sig: crate::ty::Signature,
    /// Weak declaration with no definition in this module: the object
    /// marks the undefined symbol `STB_WEAK`, so the linker resolves it
    /// to zero instead of reporting it (GCC's `extern ... __attribute__((weak))`).
    pub weak: bool,
    /// Memory behavior an optimizer may assume at call sites.
    pub effect: ExternEffect,
}

/// Whole compilation unit: functions plus globals plus externs.
#[derive(Debug, Clone, Default)]
pub struct Module {
    pub funcs: IndexVec<Func, Function>,
    pub globals: IndexVec<Global, GlobalData>,
    pub externs: IndexVec<Extern, ExternData>,
    /// Source file table; [`Location::file`] indexes it.
    pub files: Vec<File>,
    /// Shared debug type table; [`crate::debug::DebugTypeId`] indexes it.
    pub debug_types: Vec<DebugType>,
}

impl Module {
    pub fn new() -> Self {
        Module::default()
    }

    pub fn push_func(&mut self, func: Function) -> Func {
        self.funcs.push(func)
    }

    pub fn push_global(&mut self, name: &str, ty: Ty, init: Option<Const>) -> Global {
        self.globals.push(GlobalData {
            name: name.to_string(),
            ty,
            init,
            external: false,
            local: false,
            weak: false,
            tls: false,
            writable: true,
            align: None,
            section: None,
            relocs: Vec::new(),
            relocs32: Vec::new(),
            debug: None,
        })
    }

    /// Declare storage defined elsewhere: no image is ever attached.
    pub fn push_data_extern(&mut self, name: &str, ty: Ty) -> Global {
        self.globals.push(GlobalData {
            name: name.to_string(),
            ty,
            init: None,
            external: true,
            local: false,
            weak: false,
            tls: false,
            writable: true,
            align: None,
            section: None,
            relocs: Vec::new(),
            relocs32: Vec::new(),
            debug: None,
        })
    }

    pub fn push_extern(&mut self, name: &str, sig: crate::ty::Signature) -> Extern {
        self.externs.push(ExternData {
            name: name.to_string(),
            sig,
            weak: false,
            effect: ExternEffect::clobber(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_to_int_conversions_are_not_pure() {
        let value = Value::from_usize(0);
        // Cranelift's f32/f64-to-int conversions trap on NaN and
        // out-of-range inputs, so they cannot be removed or speculated.
        assert!(!InstKind::SFtoI(value).is_pure());
        assert!(!InstKind::UFtoI(value).is_pure());
        assert!(InstKind::SItoF(value).is_pure());
        assert!(InstKind::UItoF(value).is_pure());
    }
}
