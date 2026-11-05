//! Checking for GCC's x86 `__builtin_ia32_*` builtins. Only the
//! spellings with real lowering exist here; each checks inline into an
//! [`Ia32Op`] node (no declared prototype), like the `__sync` family.
//! The lowering lives in `glcc_lower_mir`'s `lower/ia32.rs`: pure MIR
//! sequences where possible (`bsrsi` off `clz`, the rotates off
//! shifts), inline asm for the instructions MIR has no opcode for
//! (`rdtsc`/`rdtscp`/`rdpmc`, the SSE4.2 CRC-32C step).

use super::*;

/// Which lane-0 scalar operation an [`Ia32Op::SsBin`] computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

/// Which bitwise operation an [`Ia32Op::PsBitwise`] computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PsBitOp {
    And,
    AndN,
    Or,
    Xor,
}

/// The SSE compare predicates, in the instruction's own ordering (the
/// immediate form's 0-7) plus the spelled `n*` swaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpPred {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
    Neq,
    Nlt,
    Nle,
    Ngt,
    Nge,
    Ord,
    Unord,
}

/// Which packed float shape an int↔float conversion targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CvtfTy {
    Ps,
    Pd,
}

/// Which lane operation a [`Ia32Op::Vec128Bin`] computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vec128BinOp {
    Add,
    Sub,
    And,
    AndN,
    Or,
    Xor,
}

/// The direction of a 128-bit integer lane shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vec128ShiftDir {
    Ll,
    Lr,
    Ar,
}

/// The saturation range a [`Ia32Op::Vec128Pack`] clamps each lane to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vec128PackKind {
    SignedByte,
    SignedWord,
    UnsignedByte,
}

/// Which subvector a [`Ia32Op::Vec128Shuf`] permutes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vec128ShufMode {
    D,
    LowW,
    HighW,
}

/// The COMISS flag test a [`Ia32Op::Comi`] returns: which of ZF/CF
/// (and their complement) the spelling reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComiPred {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
    Neq,
}

/// Which x86 builtin an `Ia32` node computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ia32Op {
    /// `int __builtin_ia32_bsrsi(int)`: bit-scan reverse — the index
    /// of the highest set bit. Undefined for a zero operand (lowered
    /// to `-1`).
    Bsr32,
    /// `long long __builtin_ia32_bsrdi(long long)`.
    Bsr64,
    /// `unsigned __builtin_ia32_crc32{qi,hi,si,di}(C crc, V v)`:
    /// one SSE4.2 CRC-32C (polynomial `0x11EDC6F41`) step over a
    /// 1/2/4/8-byte value.
    Crc32 { width: u8 },
    /// `unsigned char __builtin_ia32_rolqi(unsigned char, int)`:
    /// rotate left, the count masked to the width.
    Rol8,
    /// `unsigned short __builtin_ia32_rolhi(unsigned short, int)`.
    Rol16,
    /// `unsigned char __builtin_ia32_rorqi(unsigned char, int)`.
    Ror8,
    /// `unsigned short __builtin_ia32_rorhi(unsigned short, int)`.
    Ror16,
    /// `unsigned char __builtin_ia32_addcarryx_u{32,64}(unsigned char
    /// cin, T a, T b, T *out)`: the ADX carry chain — `*out = a + b +
    /// cin`, returning the carry out.
    AddCarry { width: u8 },
    /// `unsigned char __builtin_ia32_sbb_u{32,64}(unsigned char cin,
    /// T a, T b, T *out)`: the borrow chain — `*out = a - b - cin`,
    /// returning the borrow out.
    Sbb { width: u8 },
    /// `T __builtin_ia32_tzcnt_u{16,32,64}(T)`: count trailing zeros,
    /// defined as the width for zero (the BMI1 `tzcnt` contract,
    /// unlike plain `ctz`).
    Tzcnt { width: u8 },
    /// `T __builtin_ia32_bextr_u{32,64}(T x, T y)`: extract
    /// `y[15:8]` bits starting at `y[7:0]`, zero-filled above.
    Bextr { width: u8 },
    /// `T __builtin_ia32_bzhi_{si,di}(T x, T y)`: zero every bit at
    /// or above index `y & 0xff`.
    Bzhi { width: u8 },
    /// `T __builtin_ia32_pdep_{si,di}(T x, T mask)`: scatter the low
    /// bits of `x` into the positions `mask` sets.
    Pdep { width: u8 },
    /// `T __builtin_ia32_pext_{si,di}(T x, T mask)`: gather the bits
    /// of `x` at the positions `mask` sets down to the low end.
    Pext { width: u8 },
    /// `unsigned long long __builtin_ia32_rdtsc(void)`: the 64-bit
    /// time-stamp counter.
    Rdtsc,
    /// `unsigned long long __builtin_ia32_rdtscp(unsigned int *)`:
    /// also stores the TSC aux value through the pointer.
    Rdtscp,
    /// `unsigned long long __builtin_ia32_rdpmc(unsigned)`: a
    /// performance-counter read.
    Rdpmc,
    /// `unsigned long long __builtin_ia32_xgetbv(unsigned)`: an XCR
    /// register read.
    Xgetbv,
    /// `void __builtin_ia32_sfence(void)`: a store fence. Lowered to
    /// MIR's full fence, which is stricter but preserves every store
    /// order sfence guarantees.
    Sfence,
    /// `unsigned {int,long long} __builtin_ia32_readeflags_u{32,64}(void)`.
    ReadFlags { width: u8 },
    /// `void __builtin_ia32_writeeflags_u{32,64}(unsigned ...)`.
    WriteFlags { width: u8 },
    /// CET shadow-stack builtins (`cetintrin.h`): SSP reads and
    /// increments, token save/restore, and the WRSS/WRUS opaque
    /// stores. `Wrss` covers `wrss{d,q}` and `wruss{d,q}`.
    RdSsp { width: u8 },
    IncSsp { width: u8 },
    RstorSsp,
    Wrss { uss: bool, width: u8 },
    ClrSsbsy,
    /// Cache-line hints over one memory operand: `cldemote`,
    /// `clflushopt`, `clwb`, `clflush`.
    CacheOp { mnemonic: &'static str },
    /// `T __builtin_ia32_lzcnt_u{16,32,64}(T)`: leading zeros, the
    /// width for zero (the BMI1 `lzcnt` contract).
    Lzcnt { width: u8 },
    /// `void __builtin_ia32_clzero(void *)`: cache-line zeroing, the
    /// address in `rax`.
    Clzero,
    /// `void __builtin_ia32_directstoreu_u{32,64}(void *, T)`:
    /// non-temporal direct store (`movdiri`).
    DirectStore { width: u8 },
    /// `void __builtin_ia32_movdir64b(void *, const void *)`: the
    /// 64-byte direct store.
    Movdir64b,
    /// `void __builtin_ia32_x{save,rstor,saveopt,savec,saves}{,64}(
    /// void *, long long)`: extended processor state save/restore,
    /// the bitmap's low and high halves in `eax`/`edx`.
    XSave { mnemonic: &'static str },
    /// `unsigned __builtin_ia32_xbegin(void)`: start an RTM
    /// transaction, the abort status (or `-1`) in `eax`.
    Xbegin,
    /// `unsigned char __builtin_ia32_testui(void)`: the UINTR user
    /// interrupt flag, read through `eax`'s low bit.
    Testui,
    /// `void __builtin_ia32_senduipi(unsigned long long)`: send a
    /// user interrupt to the target's UPID.
    Senduipi,
    /// `unsigned {int,long long} __builtin_ia32_rd{fs,gs}base{32,64}(void)`
    /// and the `wr` forms: the FSGSBASE segment-base accesses.
    FsBase { write: bool, width: u8 },
    /// `unsigned int __builtin_ia32_rdpid(void)`.
    Rdpid,
    /// `int __builtin_ia32_xtest(void)`: whether in an RTM/HTM
    /// transaction.
    Xtest,
    /// `void __builtin_ia32_hreset(unsigned)`: the reset select, an
    /// immediate operand.
    Hreset,
    /// `unsigned long long __builtin_ia32_urdmsr(unsigned long long)`
    /// / `void uwrmsr(msr, value)`: the user-mode MSR accesses, the
    /// value in `edx:eax`.
    URdMsr,
    UWrMsr,
    /// `void __builtin_ia32_ptwrite{32,64}(unsigned ...)`: an Intel
    /// Processor Trace payload write.
    PtWrite { width: u8 },
    /// The MMX integer-SIMD pairs (`padd*`, `psub*`, the packed
    /// compares/interleaves/packs, ...): `__m64` operands in
    /// `mm0`/`mm1`, staged through general registers. The frontend
    /// sees `__m64` as an 8-byte integer (`vector_size` is not
    /// modeled).
    MmxOp { mnemonic: &'static str },
    /// The MMX shifts: `psllw` takes the count in `mm1`, the
    /// `psllwi` spellings an immediate.
    MmxShift { mnemonic: &'static str, imm: bool },
    /// `__builtin_ia32_vec_init_v{4hi,8qi}`: lane composition, done
    /// in pure MIR over the 8-byte integer the frontend sees.
    VecInit { lanes: u8 },
    /// `__builtin_ia32_vec_init_v2si(int, int)`: the two 32-bit lanes
    /// of an 8-byte vector.
    VecInitV2Si,
    /// `__builtin_ia32_vec_ext_v2si(T, int)`: one 32-bit lane by
    /// constant index.
    VecExtV2Si,
    /// The no-operand instructions (`pause`, `serialize`, `xend`,
    /// `clui`/`stui`, `wbnoinvd`, the CET land trackers).
    NoOperand { mnemonic: &'static str },
    /// `void __builtin_ia32_xsetbv(unsigned, long long)`: write an
    /// XCR register.
    XSetBv,
    /// `void __builtin_ia32_umonitor(void *)`: the wait-package
    /// monitor arm.
    UMonitor,
    /// `unsigned int __builtin_ia32_{umwait,tpause}(unsigned,
    /// unsigned long long)`: wait for a monitor or deadline; returns
    /// the wakeup reason in `eax`.
    UWait { tpause: bool },
    /// `int __builtin_ia32_enqcmd{s}(void *, const void *)`: enqueue
    /// a 64-byte store, reporting acceptance.
    EnqCmd { ordered: bool },
    /// The SSE scalar-tail arithmetic pairs (`addss`, `subss`,
    /// `mulss`, `divss`, `minss`, `maxss`): lane 0 of `__v4sf`
    /// computes, lanes 1-3 pass through from the first operand. The
    /// lowering is pure MIR over extracted lanes; `min`/`max` are the
    /// select forms (a NaN first operand yields the second operand,
    /// matching `minss`/`maxss`).
    SsBin { op: SsOp },
    /// `__builtin_ia32_sqrtps(__v4sf)`: square root of every lane.
    SqrtPs,
    /// `__builtin_ia32_sqrtss(__v4sf)`: square root of lane 0.
    SqrtSs,
    /// `__builtin_ia32_rcpss(__v4sf)` / `rsqrtss`: the approximate
    /// reciprocal / reciprocal square root of lane 0, via inline asm
    /// over a memory operand.
    RcpSs { rsqrt: bool },
    /// The packed SSE1 arithmetic and min/max pairs (`addps` ..
    /// `maxps`): every lane computes, shape-generically over `__v4sf`.
    PsBin { op: SsOp },
    /// The packed bitwise pairs (`andps`, `andnps`, `orps`, `xorps`):
    /// bitwise over the 128 bits regardless of the float lanes.
    PsBitwise { op: PsBitOp },
    /// `__builtin_ia32_rcpps(__v4sf)` / `rsqrtps`: the approximate
    /// reciprocal / reciprocal square root of every lane, via inline
    /// asm over a memory operand.
    RcpPs { rsqrt: bool },
    /// The SSE compare family (`cmpeqss` .. `cmpunordps`, plus the
    /// raw `cmpss`/`cmpps` immediate forms): `ps` compares every lane
    /// and the mask is the result; `ss` compares lane 0 and passes the
    /// first operand's lanes 1-3 through.
    Cmp { ps: bool, pred: CmpPred },
    /// `int __builtin_ia32_(u)comi{eq,lt,le,gt,ge,neq}(__v4sf,
    /// __v4sf)`: the COMISS flag tests on lane 0, returned as 0/1.
    /// `comi` and `ucomi` differ only in QNaN signaling; the result
    /// bits are the same, so one lowering serves both.
    Comi { pred: ComiPred, unordered: bool },
    /// `__builtin_ia32_movss(__v4sf, __v4sf)`: lane 0 of the second
    /// operand over lane 0 of the first.
    MovSs,
    /// The MMX word/byte extras SSE1 adds (`pmaxsw`, `pminub`,
    /// `pavgb`, `psadbw`, `pextrw`, `pinsrw`, `pmovmskb`, `pshufw`):
    /// the pairs ride [`Ia32Op::MmxOp`]; the immediate and flag forms
    /// get their own ops.
    Pextrw,
    Pinsrw,
    Pmovmskb,
    Pshufw,
    /// `void __builtin_ia32_ldmxcsr(unsigned)` /
    /// `unsigned stmxcsr(void)`: the SSE control-status word.
    Ldmxcsr,
    Stmxcsr,
    /// `int __builtin_ia32_movmskps(__v4sf)`: the four sign bits.
    MovmskPs,
    /// `__builtin_ia32_shufps(__v4sf, __v4sf, int)`: the two-bit
    /// per-lane selector, `a` for the low half, `b` for the high.
    ShufPs,
    /// The double-precision mirror of the SSE1 set: everything above
    /// with `__v2df` lanes (SSE2's `*sd`/`*pd` spellings).
    SdBin { op: SsOp },
    PdBin { op: SsOp },
    CmpPd { ps: bool, pred: CmpPred },
    SqrtPd,
    SqrtSd,
    ShufPd,
    UnpckPd { high: bool },
    LoadHalfPd { high: bool },
    StoreHalfPd { high: bool },
    MovSd,
    CvtSd2Ss,
    CvtSs2Sd,
    CvtSi2Sd { width: u8 },
    CvtSd2Si { width: u8, truncate: bool },
    CvtPd2Ps,
    CvtPs2Pd,
    CvtPd2Dq { truncate: bool },
    CvtPd2Pi { truncate: bool },
    CvtPi2Pd,
    /// The packed int↔float conversions (`cvtdq2ps`, `cvtps2dq`, ...).
    CvtInt2Float { to: CvtfTy },
    CvtFloat2Int { from: CvtfTy, truncate: bool },
    /// `__m128 __builtin_ia32_cvtsi2ss(__v4sf, int)` /
    /// `cvtsi642ss(__v4sf, long long)`: an integer into lane 0, the
    /// first operand's lanes 1-3 through.
    CvtSi2Ss { width: u8 },
    /// `int/long long __builtin_ia32_(cvt|cvtt)ss2si{,64}(__v4sf)`:
    /// lane 0 to an integer, rounding to nearest (`cvt`) or
    /// truncating (`cvtt`). Rounding goes through the instruction via
    /// inline asm (MIR's float→int is truncating only).
    CvtSs2Si { width: u8, truncate: bool },
    /// `__m64 __builtin_ia32_(cvt|cvtt)ps2pi(__v4sf)`: lanes 0-1 to
    /// the two 32-bit lanes of an 8-byte vector, rounding (`cvt`) or
    /// truncating (`cvtt`).
    CvtPs2Pi { truncate: bool },
    /// `__m128 __builtin_ia32_cvtpi2ps(__v4sf, __v2si)`: the two
    /// integer lanes converted to float into lanes 0-1, the first
    /// operand's lanes 2-3 through.
    CvtPi2Ps,
    /// `__builtin_ia32_movlhps(__v4sf, __v4sf)`: the first operand's
    /// lanes into the low half, the second's into the high half.
    MovLhPs,
    /// `movhlps`: the second operand's high half into the low half,
    /// the first operand's high half stays in the high half.
    MovHlPs,
    /// `unpcklps`/`unpckhps`: the interleave of one operand's low or
    /// high pair with the other's.
    UnpckPs { high: bool },
    /// `loadhps`/`loadlps`: the two floats behind the pointer into the
    /// high or low half, the first operand's other half through.
    LoadHalfPs { high: bool },
    /// `storehps`/`storelps`: the high or low half to the pointer.
    StoreHalfPs { high: bool },
    /// LWP control: `__builtin_ia32_llwpcb(void *)` sets the
    /// asynchronous-event PCB, `slwpcb()` returns it.
    Llwpcb,
    Slwpcb,
    /// MWAIT-family monitor/wait (`monitor`/`monitorx`,
    /// `mwait`/`mwaitx`): address in `eax`, extensions `ecx`, hints
    /// `edx`, with `mwaitx` adding a timer in `ebx`.
    Monitor { extended: bool },
    Mwait { extended: bool },
    /// `unsigned __builtin_ia32_rdpkru(void)` /
    /// `void wrpkru(unsigned)`: the protection-key register, `ecx`
    /// and `edx` required to be zero.
    Rdpkru,
    Wrpkru,
    /// `int __builtin_ia32_{rdseed,rdrand}_{hi,si,di}_step(T *)`:
    /// store a hardware random value through the pointer, returning
    /// whether the value is valid (the instruction's carry flag).
    RdStep { rand: bool, width: u8 },
    /// The SSE2 128-bit integer lane pairs (`padd*128`-class): every
    /// lane computes over first-class 16-byte vectors.
    Vec128Bin { op: Vec128BinOp },
    /// `padds*`/`paddus*`/`psubs*`/`psubus*128`: lane-wise add or sub
    /// saturating at the narrower lane's signed or unsigned range.
    Vec128Sat { add: bool, unsigned: bool },
    /// `psll*`/`psrl*`/`psra*128` and the `i` immediate forms.
    Vec128Shift { dir: Vec128ShiftDir, imm: bool },
    /// `pcmpeq*128`/`pcmpgt*128`: lane masks of 0/`-1`.
    Vec128Cmp { eq: bool },
    /// `pmax*`/`pmin*128`.
    Vec128MinMax { min: bool, signed: bool },
    /// `pavgb128`/`pavgw128`: the unsigned rounding average.
    Vec128Avg,
    /// `packsswb128`/`packssdw128`/`packuswb128`: the operands' lanes
    /// saturate and truncate into the narrow result's halves.
    Vec128Pack { kind: Vec128PackKind },
    /// `pmulhw128`/`pmulhuw128`: the high half of each lane product.
    Vec128Mul { high: bool },
    /// `pmaddwd128`: each pair's two 16-bit products summed into a
    /// 32-bit result lane.
    Vec128Madd,
    /// `pmuludq128`: the even dwords multiplied into 64-bit lanes.
    Vec128MullDq,
    /// `psadbw128`: sums of absolute byte differences, one per
    /// 8-byte block, in the block's low 16 bits.
    Vec128Sad,
    /// `punpck{h,l}*128`: the lane interleave of the operands'
    /// halves.
    Vec128Unpck { high: bool },
    /// `pshufd`/`pshuflw`/`pshufhw`: the two-bit-per-lane selector
    /// over the dwords or the low/high words.
    Vec128Shuf { mode: Vec128ShufMode },
    /// `__builtin_ia32_vec_ext_v4si(__v4si, int)`: one dword lane by
    /// constant index.
    VecExtV4Si,
    /// Clang `__builtin_elementwise_sqrt(x)`: the square root of
    /// every lane (or of the scalar).
    Vec128Sqrt,
    /// Clang `__builtin_convertvector(v, T)`: the element-wise
    /// conversion to the target vector type.
    ConvertVector,
    /// Clang `__builtin_nontemporal_store(v, p)`: a non-temporal
    /// store, an ordinary store in our lowering.
    NtStore,
    /// `__builtin_ia32_undef128()`: an undefined 128-bit vector.
    Undef,
    /// Clang `psl{,r}dqi128_byteshift`: the whole 128-bit register
    /// shifted by whole bytes, zero-filled.
    VecByteShift { left: bool },
    /// `__builtin_ia32_pmovmskb128(__v16qi)`: the 16 byte sign bits.
    Pmovmskb128,
    /// `__builtin_ia32_movq128(__v2di)`: the low qword through, the
    /// high qword zeroed.
    Movq128,
    /// The `movnt*` non-temporal stores: an ordinary store in our
    /// lowering, the non-temporal hint dropped.
    MovNt { bytes: u8 },
    /// `maskmovq`/`maskmovdqu`: the mask-selected bytes of the first
    /// operand stored to `[rdi]` (the third argument).
    MaskMov,
}

/// The pointer parameters of the `sbb`/`addcarryx` builtins.
fn ptr_to(pointee: Ty) -> Ty {
    Ty::Pointer {
        qualifiers: TypeQualifiers::empty(),
        size: PtrSize::Default,
        pointee: Box::new(pointee),
    }
}

/// `__v4sf`, the four-lane `float` vector the SSE1 builtins spell.
fn v4sf() -> Ty {
    Ty::Vector {
        elem: Box::new(Ty::Float),
        len: 4,
    }
}

/// The SSE2 shapes: `__v2df` and the 128-bit integer spellings.
fn v2df() -> Ty {
    Ty::Vector {
        elem: Box::new(Ty::Double),
        len: 2,
    }
}

fn vec(elem: Ty, len: u32) -> Ty {
    Ty::Vector {
        elem: Box::new(elem),
        len,
    }
}

fn v16qi() -> Ty {
    vec(Ty::Char, 8 * 2)
}

fn v8hi() -> Ty {
    vec(Ty::Short, 4 * 2)
}

fn v4si() -> Ty {
    vec(Ty::Int, 2 * 2)
}

fn v2di() -> Ty {
    vec(Ty::LongLong, 2)
}

fn v2si() -> Ty {
    vec(Ty::Int, 2)
}

fn v4hi() -> Ty {
    vec(Ty::Short, 4)
}

fn v8qi() -> Ty {
    vec(Ty::Char, 8)
}

/// The `const void *` parameters (`movdir64b`'s source, the monitor
/// address): headers pass qualified pointers, and `void *` would warn.
/// Pointee qualifiers live in `Ty::Qualified`, not on the pointer.
fn const_ptr_to(pointee: Ty) -> Ty {
    Ty::Pointer {
        qualifiers: TypeQualifiers::empty(),
        size: PtrSize::Default,
        pointee: Box::new(Ty::Qualified {
            quals: TypeQualifiers::CONST,
            inner: Box::new(pointee),
        }),
    }
}

impl<'a> Sema<'a> {
    /// A GCC x86 `__builtin_ia32_*` call, or `None` when `name` is not
    /// one of the supported spellings.
    pub(super) fn check_ia32(&self, name: &str, args: &[Expr<'a>], interner: &Interner, span: Span) -> Option<Result<CheckedExpr<'a>, CompileError>> {
        let rest = name.strip_prefix("__builtin_ia32_")?;
        // The RAO-INT memory operators are atomic RMWs, so they check
        // into the same `Atomic` nodes as the `__atomic` builtins.
        if let Some(op) = match rest {
            "aadd32" | "aadd64" => Some(RmwOp::Add),
            "aand32" | "aand64" => Some(RmwOp::And),
            "aor32" | "aor64" => Some(RmwOp::Or),
            "axor32" | "axor64" => Some(RmwOp::Xor),
            _ => None,
        } {
            if args.len() != 2 {
                return Some(Err(CompileError::new(format!("`{name}` takes exactly 2 arguments"), span)));
            }
            let checked = self.check_expr(&args[0], interner).and_then(|ptr| {
                let ptr = self.decay_checked(ptr);
                let Ty::Pointer { pointee, .. } = ptr.ty.clone() else {
                    return Err(CompileError::new(format!("`{name}` needs a pointer argument"), ptr.span));
                };
                self.check_expr(&args[1], interner).and_then(|v| self.convert_for_assignment(&pointee, v, interner)).map(|value| (ptr, value))
            });
            return Some(checked.map(|(ptr, value)| CheckedExpr {
                ty: Ty::Void,
                span,
                kind: CheckedExprKind::Atomic {
                    op: AtomicOp::Rmw {
                        op,
                        fetch: true,
                        sized: None,
                    },
                    args: self.alloc_slice([ptr, value]),
                },
            }));
        }
        let (op, params, ret): (Ia32Op, &[Ty], Ty) = match rest {
            "bsrsi" => (Ia32Op::Bsr32, &[Ty::Int], Ty::Int),
            "bsrdi" => (Ia32Op::Bsr64, &[Ty::LongLong], Ty::LongLong),
            "crc32qi" => (Ia32Op::Crc32 { width: 1 }, &[Ty::UnsignedInt, Ty::UnsignedChar], Ty::UnsignedInt),
            "crc32hi" => (Ia32Op::Crc32 { width: 2 }, &[Ty::UnsignedInt, Ty::UnsignedShort], Ty::UnsignedInt),
            "crc32si" => (Ia32Op::Crc32 { width: 4 }, &[Ty::UnsignedInt, Ty::UnsignedInt], Ty::UnsignedInt),
            "crc32di" => (Ia32Op::Crc32 { width: 8 }, &[Ty::UnsignedLongLong, Ty::UnsignedLongLong], Ty::UnsignedLongLong),
            "rolqi" => (Ia32Op::Rol8, &[Ty::UnsignedChar, Ty::Int], Ty::UnsignedChar),
            "rolhi" => (Ia32Op::Rol16, &[Ty::UnsignedShort, Ty::Int], Ty::UnsignedShort),
            "rorqi" => (Ia32Op::Ror8, &[Ty::UnsignedChar, Ty::Int], Ty::UnsignedChar),
            "rorhi" => (Ia32Op::Ror16, &[Ty::UnsignedShort, Ty::Int], Ty::UnsignedShort),
            "addcarryx_u32" => (Ia32Op::AddCarry { width: 4 }, &[Ty::UnsignedChar, Ty::UnsignedInt, Ty::UnsignedInt, ptr_to(Ty::UnsignedInt)], Ty::Bool),
            "addcarryx_u64" => (Ia32Op::AddCarry { width: 8 }, &[Ty::UnsignedChar, Ty::UnsignedLongLong, Ty::UnsignedLongLong, ptr_to(Ty::UnsignedLongLong)], Ty::Bool),
            "sbb_u32" => (Ia32Op::Sbb { width: 4 }, &[Ty::UnsignedChar, Ty::UnsignedInt, Ty::UnsignedInt, ptr_to(Ty::UnsignedInt)], Ty::Bool),
            "sbb_u64" => (Ia32Op::Sbb { width: 8 }, &[Ty::UnsignedChar, Ty::UnsignedLongLong, Ty::UnsignedLongLong, ptr_to(Ty::UnsignedLongLong)], Ty::Bool),
            "tzcnt_u16" => (Ia32Op::Tzcnt { width: 2 }, &[Ty::UnsignedShort], Ty::UnsignedShort),
            "tzcnt_u32" => (Ia32Op::Tzcnt { width: 4 }, &[Ty::UnsignedInt], Ty::UnsignedInt),
            "tzcnt_u64" => (Ia32Op::Tzcnt { width: 8 }, &[Ty::UnsignedLongLong], Ty::UnsignedLongLong),
            "bextr_u32" => (Ia32Op::Bextr { width: 4 }, &[Ty::UnsignedInt, Ty::UnsignedInt], Ty::UnsignedInt),
            "bextr_u64" => (Ia32Op::Bextr { width: 8 }, &[Ty::UnsignedLongLong, Ty::UnsignedLongLong], Ty::UnsignedLongLong),
            "bzhi_si" => (Ia32Op::Bzhi { width: 4 }, &[Ty::UnsignedInt, Ty::UnsignedInt], Ty::UnsignedInt),
            "bzhi_di" => (Ia32Op::Bzhi { width: 8 }, &[Ty::UnsignedLongLong, Ty::UnsignedLongLong], Ty::UnsignedLongLong),
            "pdep_si" => (Ia32Op::Pdep { width: 4 }, &[Ty::UnsignedInt, Ty::UnsignedInt], Ty::UnsignedInt),
            "pdep_di" => (Ia32Op::Pdep { width: 8 }, &[Ty::UnsignedLongLong, Ty::UnsignedLongLong], Ty::UnsignedLongLong),
            "pext_si" => (Ia32Op::Pext { width: 4 }, &[Ty::UnsignedInt, Ty::UnsignedInt], Ty::UnsignedInt),
            "pext_di" => (Ia32Op::Pext { width: 8 }, &[Ty::UnsignedLongLong, Ty::UnsignedLongLong], Ty::UnsignedLongLong),
            "rdtsc" => (Ia32Op::Rdtsc, &[], Ty::UnsignedLongLong),
            "rdtscp" => (Ia32Op::Rdtscp, &[ptr_to(Ty::UnsignedInt)], Ty::UnsignedLongLong),
            "rdpmc" => (Ia32Op::Rdpmc, &[Ty::UnsignedInt], Ty::UnsignedLongLong),
            "xgetbv" => (Ia32Op::Xgetbv, &[Ty::UnsignedInt], Ty::UnsignedLongLong),
            "sfence" => (Ia32Op::Sfence, &[], Ty::Void),
            "pause" => (Ia32Op::NoOperand { mnemonic: "pause" }, &[], Ty::Void),
            "readeflags_u32" => (Ia32Op::ReadFlags { width: 4 }, &[], Ty::UnsignedInt),
            "readeflags_u64" => (Ia32Op::ReadFlags { width: 8 }, &[], Ty::UnsignedLongLong),
            "writeeflags_u32" => (Ia32Op::WriteFlags { width: 4 }, &[Ty::UnsignedInt], Ty::Void),
            "writeeflags_u64" => (Ia32Op::WriteFlags { width: 8 }, &[Ty::UnsignedLongLong], Ty::Void),
            "rdsspd" => (Ia32Op::RdSsp { width: 4 }, &[], Ty::UnsignedInt),
            "rdsspq" => (Ia32Op::RdSsp { width: 8 }, &[], Ty::UnsignedLongLong),
            "incsspd" => (Ia32Op::IncSsp { width: 4 }, &[Ty::UnsignedInt], Ty::Void),
            "incsspq" => (Ia32Op::IncSsp { width: 8 }, &[Ty::UnsignedLongLong], Ty::Void),
            "saveprevssp" => (Ia32Op::NoOperand { mnemonic: "saveprevssp" }, &[], Ty::Void),
            "rstorssp" => (Ia32Op::RstorSsp, &[ptr_to(Ty::Void)], Ty::Void),
            "wrssd" => (Ia32Op::Wrss { uss: false, width: 4 }, &[Ty::UnsignedInt, ptr_to(Ty::Void)], Ty::Void),
            "wrssq" => (Ia32Op::Wrss { uss: false, width: 8 }, &[Ty::UnsignedLongLong, ptr_to(Ty::Void)], Ty::Void),
            "wrussd" => (Ia32Op::Wrss { uss: true, width: 4 }, &[Ty::UnsignedInt, ptr_to(Ty::Void)], Ty::Void),
            "wrussq" => (Ia32Op::Wrss { uss: true, width: 8 }, &[Ty::UnsignedLongLong, ptr_to(Ty::Void)], Ty::Void),
            "setssbsy" => (Ia32Op::NoOperand { mnemonic: "setssbsy" }, &[], Ty::Void),
            "clrssbsy" => (Ia32Op::ClrSsbsy, &[ptr_to(Ty::Void)], Ty::Void),
            "cldemote" => (Ia32Op::CacheOp { mnemonic: "cldemote" }, &[ptr_to(Ty::Void)], Ty::Void),
            "clflushopt" => (Ia32Op::CacheOp { mnemonic: "clflushopt" }, &[ptr_to(Ty::Void)], Ty::Void),
            "clwb" => (Ia32Op::CacheOp { mnemonic: "clwb" }, &[ptr_to(Ty::Void)], Ty::Void),
            "clflush" => (Ia32Op::CacheOp { mnemonic: "clflush" }, &[const_ptr_to(Ty::Void)], Ty::Void),
            "fxsave" | "fxsave64" => (Ia32Op::CacheOp { mnemonic: "fxsave" }, &[ptr_to(Ty::Void)], Ty::Void),
            "fxrstor" | "fxrstor64" => (Ia32Op::CacheOp { mnemonic: "fxrstor" }, &[ptr_to(Ty::Void)], Ty::Void),
            "lzcnt_u16" => (Ia32Op::Lzcnt { width: 2 }, &[Ty::UnsignedShort], Ty::UnsignedShort),
            "lzcnt_u32" => (Ia32Op::Lzcnt { width: 4 }, &[Ty::UnsignedInt], Ty::UnsignedInt),
            "lzcnt_u64" => (Ia32Op::Lzcnt { width: 8 }, &[Ty::UnsignedLongLong], Ty::UnsignedLongLong),
            "clzero" => (Ia32Op::Clzero, &[ptr_to(Ty::Void)], Ty::Void),
            "directstoreu_u32" => (Ia32Op::DirectStore { width: 4 }, &[ptr_to(Ty::Void), Ty::UnsignedInt], Ty::Void),
            "directstoreu_u64" => (Ia32Op::DirectStore { width: 8 }, &[ptr_to(Ty::Void), Ty::UnsignedLongLong], Ty::Void),
            "movdir64b" => (Ia32Op::Movdir64b, &[ptr_to(Ty::Void), const_ptr_to(Ty::Void)], Ty::Void),
            "serialize" => (Ia32Op::NoOperand { mnemonic: "serialize" }, &[], Ty::Void),
            "xsave" | "xsave64" => (Ia32Op::XSave { mnemonic: "xsave" }, &[ptr_to(Ty::Void), Ty::LongLong], Ty::Void),
            "xrstor" | "xrstor64" | "xrstors" | "xrstors64" => (Ia32Op::XSave { mnemonic: "xrstors" }, &[ptr_to(Ty::Void), Ty::LongLong], Ty::Void),
            "xsaveopt" | "xsaveopt64" => (Ia32Op::XSave { mnemonic: "xsaveopt" }, &[ptr_to(Ty::Void), Ty::LongLong], Ty::Void),
            "xsavec" | "xsavec64" => (Ia32Op::XSave { mnemonic: "xsavec" }, &[ptr_to(Ty::Void), Ty::LongLong], Ty::Void),
            "xsaves" | "xsaves64" => (Ia32Op::XSave { mnemonic: "xsaves" }, &[ptr_to(Ty::Void), Ty::LongLong], Ty::Void),
            "xbegin" => (Ia32Op::Xbegin, &[], Ty::UnsignedInt),
            "xend" => (Ia32Op::NoOperand { mnemonic: "xend" }, &[], Ty::Void),
            "testui" => (Ia32Op::Testui, &[], Ty::UnsignedChar),
            "clui" => (Ia32Op::NoOperand { mnemonic: "clui" }, &[], Ty::Void),
            "stui" => (Ia32Op::NoOperand { mnemonic: "stui" }, &[], Ty::Void),
            "senduipi" => (Ia32Op::Senduipi, &[Ty::UnsignedLongLong], Ty::Void),
            "wbnoinvd" => (Ia32Op::NoOperand { mnemonic: "wbnoinvd" }, &[], Ty::Void),
            "xresldtrk" => (Ia32Op::NoOperand { mnemonic: "xresldtrk" }, &[], Ty::Void),
            "xtest" => (Ia32Op::Xtest, &[], Ty::Int),
            "hreset" => (Ia32Op::Hreset, &[Ty::UnsignedInt], Ty::Void),
            "wbinvd" => (Ia32Op::NoOperand { mnemonic: "wbinvd" }, &[], Ty::Void),
            "emms" => (Ia32Op::NoOperand { mnemonic: "emms" }, &[], Ty::Void),
            "rdfsbase32" => (Ia32Op::FsBase { write: false, width: 4 }, &[], Ty::UnsignedInt),
            "rdfsbase64" => (Ia32Op::FsBase { write: false, width: 8 }, &[], Ty::UnsignedLongLong),
            "rdgsbase32" => (Ia32Op::FsBase { write: false, width: 4 }, &[], Ty::UnsignedInt),
            "rdgsbase64" => (Ia32Op::FsBase { write: false, width: 8 }, &[], Ty::UnsignedLongLong),
            "wrfsbase32" => (Ia32Op::FsBase { write: true, width: 4 }, &[Ty::UnsignedInt], Ty::Void),
            "wrfsbase64" => (Ia32Op::FsBase { write: true, width: 8 }, &[Ty::UnsignedLongLong], Ty::Void),
            "wrgsbase32" => (Ia32Op::FsBase { write: true, width: 4 }, &[Ty::UnsignedInt], Ty::Void),
            "wrgsbase64" => (Ia32Op::FsBase { write: true, width: 8 }, &[Ty::UnsignedLongLong], Ty::Void),
            "rdpid" => (Ia32Op::Rdpid, &[], Ty::UnsignedInt),
            "ptwrite32" => (Ia32Op::PtWrite { width: 4 }, &[Ty::UnsignedInt], Ty::Void),
            "ptwrite64" => (Ia32Op::PtWrite { width: 8 }, &[Ty::UnsignedLongLong], Ty::Void),
            "pmaxsw" => (Ia32Op::MmxOp { mnemonic: "pmaxsw" }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "pmaxub" => (Ia32Op::MmxOp { mnemonic: "pmaxub" }, &[Ty::Vector { elem: Box::new(Ty::Char), len: 8 }, Ty::Vector { elem: Box::new(Ty::Char), len: 8 }], Ty::Vector { elem: Box::new(Ty::Char), len: 8 }),
            "pminsw" => (Ia32Op::MmxOp { mnemonic: "pminsw" }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "pminub" => (Ia32Op::MmxOp { mnemonic: "pminub" }, &[Ty::Vector { elem: Box::new(Ty::Char), len: 8 }, Ty::Vector { elem: Box::new(Ty::Char), len: 8 }], Ty::Vector { elem: Box::new(Ty::Char), len: 8 }),
            "pavgb" => (Ia32Op::MmxOp { mnemonic: "pavgb" }, &[Ty::Vector { elem: Box::new(Ty::Char), len: 8 }, Ty::Vector { elem: Box::new(Ty::Char), len: 8 }], Ty::Vector { elem: Box::new(Ty::Char), len: 8 }),
            "pavgw" => (Ia32Op::MmxOp { mnemonic: "pavgw" }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "psadbw" => (Ia32Op::MmxOp { mnemonic: "psadbw" }, &[Ty::Vector { elem: Box::new(Ty::Char), len: 8 }, Ty::Vector { elem: Box::new(Ty::Char), len: 8 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "pextrw" => (Ia32Op::Pextrw, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Int], Ty::Int),
            "pinsrw" => (Ia32Op::Pinsrw, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Int, Ty::Int], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "pmovmskb" => (Ia32Op::Pmovmskb, &[Ty::Vector { elem: Box::new(Ty::Char), len: 8 }], Ty::Int),
            "pshufw" => (Ia32Op::Pshufw, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Int], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "ldmxcsr" => (Ia32Op::Ldmxcsr, &[Ty::UnsignedInt], Ty::Void),
            "stmxcsr" => (Ia32Op::Stmxcsr, &[], Ty::UnsignedInt),
            "movmskps" => (Ia32Op::MovmskPs, &[v4sf()], Ty::Int),
            "shufps" => (Ia32Op::ShufPs, &[v4sf(), v4sf(), Ty::Int], v4sf()),
            "paddb" => (Ia32Op::MmxOp { mnemonic: "paddb" }, &[Ty::Vector { elem: Box::new(Ty::Char), len: 8 }, Ty::Vector { elem: Box::new(Ty::Char), len: 8 }], Ty::Vector { elem: Box::new(Ty::Char), len: 8 }),
            "paddw" => (Ia32Op::MmxOp { mnemonic: "paddw" }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "paddd" => (Ia32Op::MmxOp { mnemonic: "paddd" }, &[Ty::Vector { elem: Box::new(Ty::Int), len: 2 }, Ty::Vector { elem: Box::new(Ty::Int), len: 2 }], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            "paddq" => (Ia32Op::MmxOp { mnemonic: "paddq" }, &[Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }, Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }], Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }),
            "paddsb" => (Ia32Op::MmxOp { mnemonic: "paddsb" }, &[Ty::Vector { elem: Box::new(Ty::Char), len: 8 }, Ty::Vector { elem: Box::new(Ty::Char), len: 8 }], Ty::Vector { elem: Box::new(Ty::Char), len: 8 }),
            "paddsw" => (Ia32Op::MmxOp { mnemonic: "paddsw" }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "paddusb" => (Ia32Op::MmxOp { mnemonic: "paddusb" }, &[Ty::Vector { elem: Box::new(Ty::Char), len: 8 }, Ty::Vector { elem: Box::new(Ty::Char), len: 8 }], Ty::Vector { elem: Box::new(Ty::Char), len: 8 }),
            "paddusw" => (Ia32Op::MmxOp { mnemonic: "paddusw" }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "psubb" => (Ia32Op::MmxOp { mnemonic: "psubb" }, &[Ty::Vector { elem: Box::new(Ty::Char), len: 8 }, Ty::Vector { elem: Box::new(Ty::Char), len: 8 }], Ty::Vector { elem: Box::new(Ty::Char), len: 8 }),
            "psubw" => (Ia32Op::MmxOp { mnemonic: "psubw" }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "psubd" => (Ia32Op::MmxOp { mnemonic: "psubd" }, &[Ty::Vector { elem: Box::new(Ty::Int), len: 2 }, Ty::Vector { elem: Box::new(Ty::Int), len: 2 }], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            "psubq" => (Ia32Op::MmxOp { mnemonic: "psubq" }, &[Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }, Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }], Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }),
            "psubsb" => (Ia32Op::MmxOp { mnemonic: "psubsb" }, &[Ty::Vector { elem: Box::new(Ty::Char), len: 8 }, Ty::Vector { elem: Box::new(Ty::Char), len: 8 }], Ty::Vector { elem: Box::new(Ty::Char), len: 8 }),
            "psubsw" => (Ia32Op::MmxOp { mnemonic: "psubsw" }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "psubusb" => (Ia32Op::MmxOp { mnemonic: "psubusb" }, &[Ty::Vector { elem: Box::new(Ty::Char), len: 8 }, Ty::Vector { elem: Box::new(Ty::Char), len: 8 }], Ty::Vector { elem: Box::new(Ty::Char), len: 8 }),
            "psubusw" => (Ia32Op::MmxOp { mnemonic: "psubusw" }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "pand" => (Ia32Op::MmxOp { mnemonic: "pand" }, &[Ty::Vector { elem: Box::new(Ty::Int), len: 2 }, Ty::Vector { elem: Box::new(Ty::Int), len: 2 }], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            "pandn" => (Ia32Op::MmxOp { mnemonic: "pandn" }, &[Ty::Vector { elem: Box::new(Ty::Int), len: 2 }, Ty::Vector { elem: Box::new(Ty::Int), len: 2 }], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            "por" => (Ia32Op::MmxOp { mnemonic: "por" }, &[Ty::Vector { elem: Box::new(Ty::Int), len: 2 }, Ty::Vector { elem: Box::new(Ty::Int), len: 2 }], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            "pxor" => (Ia32Op::MmxOp { mnemonic: "pxor" }, &[Ty::Vector { elem: Box::new(Ty::Int), len: 2 }, Ty::Vector { elem: Box::new(Ty::Int), len: 2 }], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            "pcmpeqb" => (Ia32Op::MmxOp { mnemonic: "pcmpeqb" }, &[Ty::Vector { elem: Box::new(Ty::Char), len: 8 }, Ty::Vector { elem: Box::new(Ty::Char), len: 8 }], Ty::Vector { elem: Box::new(Ty::Char), len: 8 }),
            "pcmpeqw" => (Ia32Op::MmxOp { mnemonic: "pcmpeqw" }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "pcmpeqd" => (Ia32Op::MmxOp { mnemonic: "pcmpeqd" }, &[Ty::Vector { elem: Box::new(Ty::Int), len: 2 }, Ty::Vector { elem: Box::new(Ty::Int), len: 2 }], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            "pcmpgtb" => (Ia32Op::MmxOp { mnemonic: "pcmpgtb" }, &[Ty::Vector { elem: Box::new(Ty::Char), len: 8 }, Ty::Vector { elem: Box::new(Ty::Char), len: 8 }], Ty::Vector { elem: Box::new(Ty::Char), len: 8 }),
            "pcmpgtw" => (Ia32Op::MmxOp { mnemonic: "pcmpgtw" }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "pcmpgtd" => (Ia32Op::MmxOp { mnemonic: "pcmpgtd" }, &[Ty::Vector { elem: Box::new(Ty::Int), len: 2 }, Ty::Vector { elem: Box::new(Ty::Int), len: 2 }], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            "pmulhw" => (Ia32Op::MmxOp { mnemonic: "pmulhw" }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "pmullw" => (Ia32Op::MmxOp { mnemonic: "pmullw" }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "pmaddwd" => (Ia32Op::MmxOp { mnemonic: "pmaddwd" }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "punpckhbw" => (Ia32Op::MmxOp { mnemonic: "punpckhbw" }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "punpckhwd" => (Ia32Op::MmxOp { mnemonic: "punpckhwd" }, &[Ty::Vector { elem: Box::new(Ty::Int), len: 2 }, Ty::Vector { elem: Box::new(Ty::Int), len: 2 }], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            "punpckhdq" => (Ia32Op::MmxOp { mnemonic: "punpckhdq" }, &[Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }, Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }], Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }),
            "punpcklbw" => (Ia32Op::MmxOp { mnemonic: "punpcklbw" }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "punpcklwd" => (Ia32Op::MmxOp { mnemonic: "punpcklwd" }, &[Ty::Vector { elem: Box::new(Ty::Int), len: 2 }, Ty::Vector { elem: Box::new(Ty::Int), len: 2 }], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            "punpckldq" => (Ia32Op::MmxOp { mnemonic: "punpckldq" }, &[Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }, Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }], Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }),
            "packsswb" => (Ia32Op::MmxOp { mnemonic: "packsswb" }, &[Ty::Vector { elem: Box::new(Ty::Char), len: 8 }, Ty::Vector { elem: Box::new(Ty::Char), len: 8 }], Ty::Vector { elem: Box::new(Ty::Char), len: 8 }),
            "packssdw" => (Ia32Op::MmxOp { mnemonic: "packssdw" }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "packuswb" => (Ia32Op::MmxOp { mnemonic: "packuswb" }, &[Ty::Vector { elem: Box::new(Ty::Char), len: 8 }, Ty::Vector { elem: Box::new(Ty::Char), len: 8 }], Ty::Vector { elem: Box::new(Ty::Char), len: 8 }),
            "psllw" => (Ia32Op::MmxShift { mnemonic: "psllw", imm: false }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "pslld" => (Ia32Op::MmxShift { mnemonic: "pslld", imm: false }, &[Ty::Vector { elem: Box::new(Ty::Int), len: 2 }, Ty::Vector { elem: Box::new(Ty::Int), len: 2 }], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            "psllq" => (Ia32Op::MmxShift { mnemonic: "psllq", imm: false }, &[Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }, Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }], Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }),
            "psrlw" => (Ia32Op::MmxShift { mnemonic: "psrlw", imm: false }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "psrld" => (Ia32Op::MmxShift { mnemonic: "psrld", imm: false }, &[Ty::Vector { elem: Box::new(Ty::Int), len: 2 }, Ty::Vector { elem: Box::new(Ty::Int), len: 2 }], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            "psrlq" => (Ia32Op::MmxShift { mnemonic: "psrlq", imm: false }, &[Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }, Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }], Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }),
            "psraw" => (Ia32Op::MmxShift { mnemonic: "psraw", imm: false }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Vector { elem: Box::new(Ty::Short), len: 4 }], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "psrad" => (Ia32Op::MmxShift { mnemonic: "psrad", imm: false }, &[Ty::Vector { elem: Box::new(Ty::Int), len: 2 }, Ty::Vector { elem: Box::new(Ty::Int), len: 2 }], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            "psllwi" => (Ia32Op::MmxShift { mnemonic: "psllw", imm: true }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Int], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "pslldi" => (Ia32Op::MmxShift { mnemonic: "pslld", imm: true }, &[Ty::Vector { elem: Box::new(Ty::Int), len: 2 }, Ty::Int], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            "psllqi" => (Ia32Op::MmxShift { mnemonic: "psllq", imm: true }, &[Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }, Ty::Int], Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }),
            "psrlwi" => (Ia32Op::MmxShift { mnemonic: "psrlw", imm: true }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Int], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "psrldi" => (Ia32Op::MmxShift { mnemonic: "psrld", imm: true }, &[Ty::Vector { elem: Box::new(Ty::Int), len: 2 }, Ty::Int], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            "psrlqi" => (Ia32Op::MmxShift { mnemonic: "psrlq", imm: true }, &[Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }, Ty::Int], Ty::Vector { elem: Box::new(Ty::LongLong), len: 1 }),
            "psrawi" => (Ia32Op::MmxShift { mnemonic: "psraw", imm: true }, &[Ty::Vector { elem: Box::new(Ty::Short), len: 4 }, Ty::Int], Ty::Vector { elem: Box::new(Ty::Short), len: 4 }),
            "psradi" => (Ia32Op::MmxShift { mnemonic: "psrad", imm: true }, &[Ty::Vector { elem: Box::new(Ty::Int), len: 2 }, Ty::Int], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
"vec_init_v2si" => (Ia32Op::VecInitV2Si, &[Ty::Int, Ty::Int], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            // SSE1 scalar tails and square roots over `__v4sf`.
            "addss" => (Ia32Op::SsBin { op: SsOp::Add }, &[v4sf(), v4sf()], v4sf()),
            "subss" => (Ia32Op::SsBin { op: SsOp::Sub }, &[v4sf(), v4sf()], v4sf()),
            "mulss" => (Ia32Op::SsBin { op: SsOp::Mul }, &[v4sf(), v4sf()], v4sf()),
            "divss" => (Ia32Op::SsBin { op: SsOp::Div }, &[v4sf(), v4sf()], v4sf()),
            "minss" => (Ia32Op::SsBin { op: SsOp::Min }, &[v4sf(), v4sf()], v4sf()),
            "maxss" => (Ia32Op::SsBin { op: SsOp::Max }, &[v4sf(), v4sf()], v4sf()),
            "sqrtps" => (Ia32Op::SqrtPs, &[v4sf()], v4sf()),
            "sqrtss" => (Ia32Op::SqrtSs, &[v4sf()], v4sf()),
            "rcpss" => (Ia32Op::RcpSs { rsqrt: false }, &[v4sf()], v4sf()),
            "rsqrtss" => (Ia32Op::RcpSs { rsqrt: true }, &[v4sf()], v4sf()),
            "addps" => (Ia32Op::PsBin { op: SsOp::Add }, &[v4sf(), v4sf()], v4sf()),
            "subps" => (Ia32Op::PsBin { op: SsOp::Sub }, &[v4sf(), v4sf()], v4sf()),
            "mulps" => (Ia32Op::PsBin { op: SsOp::Mul }, &[v4sf(), v4sf()], v4sf()),
            "divps" => (Ia32Op::PsBin { op: SsOp::Div }, &[v4sf(), v4sf()], v4sf()),
            "minps" => (Ia32Op::PsBin { op: SsOp::Min }, &[v4sf(), v4sf()], v4sf()),
            "maxps" => (Ia32Op::PsBin { op: SsOp::Max }, &[v4sf(), v4sf()], v4sf()),
            "andps" => (Ia32Op::PsBitwise { op: PsBitOp::And }, &[v4sf(), v4sf()], v4sf()),
            "andnps" => (Ia32Op::PsBitwise { op: PsBitOp::AndN }, &[v4sf(), v4sf()], v4sf()),
            "orps" => (Ia32Op::PsBitwise { op: PsBitOp::Or }, &[v4sf(), v4sf()], v4sf()),
            "xorps" => (Ia32Op::PsBitwise { op: PsBitOp::Xor }, &[v4sf(), v4sf()], v4sf()),
            "rcpps" => (Ia32Op::RcpPs { rsqrt: false }, &[v4sf()], v4sf()),
            "rsqrtps" => (Ia32Op::RcpPs { rsqrt: true }, &[v4sf()], v4sf()),
            "cmpeqss" => (Ia32Op::Cmp { ps: false, pred: CmpPred::Eq }, &[v4sf(), v4sf()], v4sf()),
            "cmpltss" => (Ia32Op::Cmp { ps: false, pred: CmpPred::Lt }, &[v4sf(), v4sf()], v4sf()),
            "cmpless" => (Ia32Op::Cmp { ps: false, pred: CmpPred::Le }, &[v4sf(), v4sf()], v4sf()),
            "cmpgess" => (Ia32Op::Cmp { ps: false, pred: CmpPred::Ge }, &[v4sf(), v4sf()], v4sf()),
            "cmpgtss" => (Ia32Op::Cmp { ps: false, pred: CmpPred::Gt }, &[v4sf(), v4sf()], v4sf()),
            "cmpneqss" => (Ia32Op::Cmp { ps: false, pred: CmpPred::Neq }, &[v4sf(), v4sf()], v4sf()),
            "cmpnltss" => (Ia32Op::Cmp { ps: false, pred: CmpPred::Nlt }, &[v4sf(), v4sf()], v4sf()),
            "cmpnless" => (Ia32Op::Cmp { ps: false, pred: CmpPred::Nle }, &[v4sf(), v4sf()], v4sf()),
            "cmpngtss" => (Ia32Op::Cmp { ps: false, pred: CmpPred::Ngt }, &[v4sf(), v4sf()], v4sf()),
            "cmpngess" => (Ia32Op::Cmp { ps: false, pred: CmpPred::Nge }, &[v4sf(), v4sf()], v4sf()),
            "cmpordss" => (Ia32Op::Cmp { ps: false, pred: CmpPred::Ord }, &[v4sf(), v4sf()], v4sf()),
            "cmpunordss" => (Ia32Op::Cmp { ps: false, pred: CmpPred::Unord }, &[v4sf(), v4sf()], v4sf()),
            "cmpeqps" => (Ia32Op::Cmp { ps: true, pred: CmpPred::Eq }, &[v4sf(), v4sf()], v4sf()),
            "cmpltps" => (Ia32Op::Cmp { ps: true, pred: CmpPred::Lt }, &[v4sf(), v4sf()], v4sf()),
            "cmpleps" => (Ia32Op::Cmp { ps: true, pred: CmpPred::Le }, &[v4sf(), v4sf()], v4sf()),
            "cmpgeps" => (Ia32Op::Cmp { ps: true, pred: CmpPred::Ge }, &[v4sf(), v4sf()], v4sf()),
            "cmpgtps" => (Ia32Op::Cmp { ps: true, pred: CmpPred::Gt }, &[v4sf(), v4sf()], v4sf()),
            "cmpneqps" => (Ia32Op::Cmp { ps: true, pred: CmpPred::Neq }, &[v4sf(), v4sf()], v4sf()),
            "cmpnltps" => (Ia32Op::Cmp { ps: true, pred: CmpPred::Nlt }, &[v4sf(), v4sf()], v4sf()),
            "cmpnleps" => (Ia32Op::Cmp { ps: true, pred: CmpPred::Nle }, &[v4sf(), v4sf()], v4sf()),
            "cmpngtps" => (Ia32Op::Cmp { ps: true, pred: CmpPred::Ngt }, &[v4sf(), v4sf()], v4sf()),
            "cmpngeps" => (Ia32Op::Cmp { ps: true, pred: CmpPred::Nge }, &[v4sf(), v4sf()], v4sf()),
            "cmpordps" => (Ia32Op::Cmp { ps: true, pred: CmpPred::Ord }, &[v4sf(), v4sf()], v4sf()),
            "cmpunordps" => (Ia32Op::Cmp { ps: true, pred: CmpPred::Unord }, &[v4sf(), v4sf()], v4sf()),
            "comieq" | "ucomieq" => (Ia32Op::Comi { pred: ComiPred::Eq, unordered: name.starts_with("ucomi") }, &[v4sf(), v4sf()], Ty::Int),
            "comilt" | "ucomilt" => (Ia32Op::Comi { pred: ComiPred::Lt, unordered: name.starts_with("ucomi") }, &[v4sf(), v4sf()], Ty::Int),
            "comile" | "ucomile" => (Ia32Op::Comi { pred: ComiPred::Le, unordered: name.starts_with("ucomi") }, &[v4sf(), v4sf()], Ty::Int),
            "comigt" | "ucomigt" => (Ia32Op::Comi { pred: ComiPred::Gt, unordered: name.starts_with("ucomi") }, &[v4sf(), v4sf()], Ty::Int),
            "comige" | "ucomige" => (Ia32Op::Comi { pred: ComiPred::Ge, unordered: name.starts_with("ucomi") }, &[v4sf(), v4sf()], Ty::Int),
            "comineq" | "ucomineq" => (Ia32Op::Comi { pred: ComiPred::Neq, unordered: name.starts_with("ucomi") }, &[v4sf(), v4sf()], Ty::Int),
            "movss" => (Ia32Op::MovSs, &[v4sf(), v4sf()], v4sf()),
            // ---- SSE2: the double-precision mirror of the SSE1 set ----
            "addsd" => (Ia32Op::SdBin { op: SsOp::Add }, &[v2df(), v2df()], v2df()),
            "subsd" => (Ia32Op::SdBin { op: SsOp::Sub }, &[v2df(), v2df()], v2df()),
            "mulsd" => (Ia32Op::SdBin { op: SsOp::Mul }, &[v2df(), v2df()], v2df()),
            "divsd" => (Ia32Op::SdBin { op: SsOp::Div }, &[v2df(), v2df()], v2df()),
            "minsd" => (Ia32Op::SdBin { op: SsOp::Min }, &[v2df(), v2df()], v2df()),
            "maxsd" => (Ia32Op::SdBin { op: SsOp::Max }, &[v2df(), v2df()], v2df()),
            "addpd" => (Ia32Op::PdBin { op: SsOp::Add }, &[v2df(), v2df()], v2df()),
            "subpd" => (Ia32Op::PdBin { op: SsOp::Sub }, &[v2df(), v2df()], v2df()),
            "mulpd" => (Ia32Op::PdBin { op: SsOp::Mul }, &[v2df(), v2df()], v2df()),
            "divpd" => (Ia32Op::PdBin { op: SsOp::Div }, &[v2df(), v2df()], v2df()),
            "minpd" => (Ia32Op::PdBin { op: SsOp::Min }, &[v2df(), v2df()], v2df()),
            "maxpd" => (Ia32Op::PdBin { op: SsOp::Max }, &[v2df(), v2df()], v2df()),
            "andpd" => (Ia32Op::PsBitwise { op: PsBitOp::And }, &[v2df(), v2df()], v2df()),
            "andnpd" => (Ia32Op::PsBitwise { op: PsBitOp::AndN }, &[v2df(), v2df()], v2df()),
            "orpd" => (Ia32Op::PsBitwise { op: PsBitOp::Or }, &[v2df(), v2df()], v2df()),
            "xorpd" => (Ia32Op::PsBitwise { op: PsBitOp::Xor }, &[v2df(), v2df()], v2df()),
            "sqrtpd" => (Ia32Op::SqrtPd, &[v2df()], v2df()),
            "sqrtsd" => (Ia32Op::SqrtSd, &[v2df()], v2df()),
            "cmpeqpd" => (Ia32Op::CmpPd { ps: true, pred: CmpPred::Eq }, &[v2df(), v2df()], v2df()),
            "cmpltpd" => (Ia32Op::CmpPd { ps: true, pred: CmpPred::Lt }, &[v2df(), v2df()], v2df()),
            "cmplepd" => (Ia32Op::CmpPd { ps: true, pred: CmpPred::Le }, &[v2df(), v2df()], v2df()),
            "cmpgtpd" => (Ia32Op::CmpPd { ps: true, pred: CmpPred::Gt }, &[v2df(), v2df()], v2df()),
            "cmpgepd" => (Ia32Op::CmpPd { ps: true, pred: CmpPred::Ge }, &[v2df(), v2df()], v2df()),
            "cmpneqpd" => (Ia32Op::CmpPd { ps: true, pred: CmpPred::Neq }, &[v2df(), v2df()], v2df()),
            "cmpnltpd" => (Ia32Op::CmpPd { ps: true, pred: CmpPred::Nlt }, &[v2df(), v2df()], v2df()),
            "cmpnlepd" => (Ia32Op::CmpPd { ps: true, pred: CmpPred::Nle }, &[v2df(), v2df()], v2df()),
            "cmpngtpd" => (Ia32Op::CmpPd { ps: true, pred: CmpPred::Ngt }, &[v2df(), v2df()], v2df()),
            "cmpngepd" => (Ia32Op::CmpPd { ps: true, pred: CmpPred::Nge }, &[v2df(), v2df()], v2df()),
            "cmpordpd" => (Ia32Op::CmpPd { ps: true, pred: CmpPred::Ord }, &[v2df(), v2df()], v2df()),
            "cmpunordpd" => (Ia32Op::CmpPd { ps: true, pred: CmpPred::Unord }, &[v2df(), v2df()], v2df()),
            "cmpeqsd" => (Ia32Op::CmpPd { ps: false, pred: CmpPred::Eq }, &[v2df(), v2df()], v2df()),
            "cmpltsd" => (Ia32Op::CmpPd { ps: false, pred: CmpPred::Lt }, &[v2df(), v2df()], v2df()),
            "cmplesd" => (Ia32Op::CmpPd { ps: false, pred: CmpPred::Le }, &[v2df(), v2df()], v2df()),
            "cmpgtsd" => (Ia32Op::CmpPd { ps: false, pred: CmpPred::Gt }, &[v2df(), v2df()], v2df()),
            "cmpgesd" => (Ia32Op::CmpPd { ps: false, pred: CmpPred::Ge }, &[v2df(), v2df()], v2df()),
            "cmpneqsd" => (Ia32Op::CmpPd { ps: false, pred: CmpPred::Neq }, &[v2df(), v2df()], v2df()),
            "cmpnltsd" => (Ia32Op::CmpPd { ps: false, pred: CmpPred::Nlt }, &[v2df(), v2df()], v2df()),
            "cmpnlesd" => (Ia32Op::CmpPd { ps: false, pred: CmpPred::Nle }, &[v2df(), v2df()], v2df()),
            "cmpngtsd" => (Ia32Op::CmpPd { ps: false, pred: CmpPred::Ngt }, &[v2df(), v2df()], v2df()),
            "cmpngesd" => (Ia32Op::CmpPd { ps: false, pred: CmpPred::Nge }, &[v2df(), v2df()], v2df()),
            "cmpordsd" => (Ia32Op::CmpPd { ps: false, pred: CmpPred::Ord }, &[v2df(), v2df()], v2df()),
            "cmpunordsd" => (Ia32Op::CmpPd { ps: false, pred: CmpPred::Unord }, &[v2df(), v2df()], v2df()),
            "comisdeq" | "ucomisdeq" => (Ia32Op::Comi { pred: ComiPred::Eq, unordered: name.starts_with("ucomi") }, &[v2df(), v2df()], Ty::Int),
            "comisdlt" | "ucomisdlt" => (Ia32Op::Comi { pred: ComiPred::Lt, unordered: name.starts_with("ucomi") }, &[v2df(), v2df()], Ty::Int),
            "comisdle" | "ucomisdle" => (Ia32Op::Comi { pred: ComiPred::Le, unordered: name.starts_with("ucomi") }, &[v2df(), v2df()], Ty::Int),
            "comisdgt" | "ucomisdgt" => (Ia32Op::Comi { pred: ComiPred::Gt, unordered: name.starts_with("ucomi") }, &[v2df(), v2df()], Ty::Int),
            "comisdge" | "ucomisdge" => (Ia32Op::Comi { pred: ComiPred::Ge, unordered: name.starts_with("ucomi") }, &[v2df(), v2df()], Ty::Int),
            "comisdneq" | "ucomisdneq" => (Ia32Op::Comi { pred: ComiPred::Neq, unordered: name.starts_with("ucomi") }, &[v2df(), v2df()], Ty::Int),
            "shufpd" => (Ia32Op::ShufPd, &[v2df(), v2df(), Ty::Int], v2df()),
            "unpckhpd" => (Ia32Op::UnpckPd { high: true }, &[v2df(), v2df()], v2df()),
            "unpcklpd" => (Ia32Op::UnpckPd { high: false }, &[v2df(), v2df()], v2df()),
            "loadhpd" => (Ia32Op::LoadHalfPd { high: true }, &[v2df(), const_ptr_to(Ty::Double)], v2df()),
            "loadlpd" => (Ia32Op::LoadHalfPd { high: false }, &[v2df(), const_ptr_to(Ty::Double)], v2df()),
            "storehpd" => (Ia32Op::StoreHalfPd { high: true }, &[ptr_to(Ty::Double), v2df()], Ty::Void),
            "storelpd" => (Ia32Op::StoreHalfPd { high: false }, &[ptr_to(Ty::Double), v2df()], Ty::Void),
            "movsd" => (Ia32Op::MovSd, &[v2df(), v2df()], v2df()),
            "movmskpd" => (Ia32Op::MovmskPs, &[v2df()], Ty::Int),
            "cvtsd2ss" => (Ia32Op::CvtSd2Ss, &[v4sf(), v2df()], v4sf()),
            "cvtss2sd" => (Ia32Op::CvtSs2Sd, &[v2df(), v4sf()], v2df()),
            "cvtsi2sd" => (Ia32Op::CvtSi2Sd { width: 4 }, &[v2df(), Ty::Int], v2df()),
            "cvtsi642sd" => (Ia32Op::CvtSi2Sd { width: 8 }, &[v2df(), Ty::LongLong], v2df()),
            "cvtsd2si" => (Ia32Op::CvtSd2Si { width: 4, truncate: false }, &[v2df()], Ty::Int),
            "cvtsd2si64" => (Ia32Op::CvtSd2Si { width: 8, truncate: false }, &[v2df()], Ty::LongLong),
            "cvttsd2si" => (Ia32Op::CvtSd2Si { width: 4, truncate: true }, &[v2df()], Ty::Int),
            "cvttsd2si64" => (Ia32Op::CvtSd2Si { width: 8, truncate: true }, &[v2df()], Ty::LongLong),
            "cvtpd2ps" => (Ia32Op::CvtPd2Ps, &[v2df()], v4sf()),
            "cvtps2pd" => (Ia32Op::CvtPs2Pd, &[v4sf()], v2df()),
            "cvtpd2dq" => (Ia32Op::CvtPd2Dq { truncate: false }, &[v2df()], v4si()),
            "cvttpd2dq" => (Ia32Op::CvtPd2Dq { truncate: true }, &[v2df()], v4si()),
            "cvtpd2pi" => (Ia32Op::CvtPd2Pi { truncate: false }, &[v2df()], v2si()),
            "cvttpd2pi" => (Ia32Op::CvtPd2Pi { truncate: true }, &[v2df()], v2si()),
            "cvtpi2pd" => (Ia32Op::CvtPi2Pd, &[v2si()], v2df()),
            "cvtdq2ps" => (Ia32Op::CvtInt2Float { to: CvtfTy::Ps }, &[v4si()], v4sf()),
            "cvtdq2pd" => (Ia32Op::CvtInt2Float { to: CvtfTy::Pd }, &[v4si()], v2df()),
            "cvtps2dq" => (Ia32Op::CvtFloat2Int { from: CvtfTy::Ps, truncate: false }, &[v4sf()], v4si()),
            "cvttps2dq" => (Ia32Op::CvtFloat2Int { from: CvtfTy::Ps, truncate: true }, &[v4sf()], v4si()),
            
            "cvtsi2ss" => (Ia32Op::CvtSi2Ss { width: 4 }, &[v4sf(), Ty::Int], v4sf()),
            "cvtsi642ss" => (Ia32Op::CvtSi2Ss { width: 8 }, &[v4sf(), Ty::LongLong], v4sf()),
            "cvtss2si" => (Ia32Op::CvtSs2Si { width: 4, truncate: false }, &[v4sf()], Ty::Int),
            "cvtss2si64" => (Ia32Op::CvtSs2Si { width: 8, truncate: false }, &[v4sf()], Ty::LongLong),
            "cvttss2si" => (Ia32Op::CvtSs2Si { width: 4, truncate: true }, &[v4sf()], Ty::Int),
            "cvttss2si64" => (Ia32Op::CvtSs2Si { width: 8, truncate: true }, &[v4sf()], Ty::LongLong),
            "cvtps2pi" => (Ia32Op::CvtPs2Pi { truncate: false }, &[v4sf()], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            "cvttps2pi" => (Ia32Op::CvtPs2Pi { truncate: true }, &[v4sf()], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            "cvtpi2ps" => (Ia32Op::CvtPi2Ps, &[v4sf(), Ty::Vector { elem: Box::new(Ty::Int), len: 2 }], v4sf()),
            "movlhps" => (Ia32Op::MovLhPs, &[v4sf(), v4sf()], v4sf()),
            "movhlps" => (Ia32Op::MovHlPs, &[v4sf(), v4sf()], v4sf()),
            "unpcklps" => (Ia32Op::UnpckPs { high: false }, &[v4sf(), v4sf()], v4sf()),
            "unpckhps" => (Ia32Op::UnpckPs { high: true }, &[v4sf(), v4sf()], v4sf()),
            "loadhps" => (Ia32Op::LoadHalfPs { high: true }, &[v4sf(), const_ptr_to(Ty::Vector { elem: Box::new(Ty::Float), len: 2 })], v4sf()),
            "loadlps" => (Ia32Op::LoadHalfPs { high: false }, &[v4sf(), const_ptr_to(Ty::Vector { elem: Box::new(Ty::Float), len: 2 })], v4sf()),
            "storehps" => (Ia32Op::StoreHalfPs { high: true }, &[ptr_to(Ty::Vector { elem: Box::new(Ty::Float), len: 2 }), v4sf()], Ty::Void),
            "storelps" => (Ia32Op::StoreHalfPs { high: false }, &[ptr_to(Ty::Vector { elem: Box::new(Ty::Float), len: 2 }), v4sf()], Ty::Void),
            "cmpps" | "cmpss" => {
                // The raw immediate forms: the predicate is argument 2,
                // encoded in the instruction's own order.
                let ps = name == "cmpps";
                let Ok(checked) = self.check_expr(&args[2], interner) else {
                    return Some(Err(CompileError::new("`cmpss`/`cmpps` predicate must be a constant 0-7", span)));
                };
                let pred = match crate::eval::eval_const(&checked, self, interner) {
                    Some(crate::eval::Value::Int(0)) => CmpPred::Eq,
                    Some(crate::eval::Value::Int(1)) => CmpPred::Lt,
                    Some(crate::eval::Value::Int(2)) => CmpPred::Le,
                    Some(crate::eval::Value::Int(3)) => CmpPred::Unord,
                    Some(crate::eval::Value::Int(4)) => CmpPred::Neq,
                    Some(crate::eval::Value::Int(5)) => CmpPred::Nlt,
                    Some(crate::eval::Value::Int(6)) => CmpPred::Nle,
                    Some(crate::eval::Value::Int(7)) => CmpPred::Ord,
                    _ => return Some(Err(CompileError::new("`cmpss`/`cmpps` predicate must be a constant 0-7", span))),
                };
                (Ia32Op::Cmp { ps, pred }, &[v4sf(), v4sf()], v4sf())
            }
            "vec_init_v4hi" => (Ia32Op::VecInit { lanes: 2 }, &[Ty::Int, Ty::Int, Ty::Int, Ty::Int], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            "vec_init_v8qi" => (Ia32Op::VecInit { lanes: 1 }, &[Ty::Int, Ty::Int, Ty::Int, Ty::Int, Ty::Int, Ty::Int, Ty::Int, Ty::Int], Ty::Vector { elem: Box::new(Ty::Int), len: 2 }),
            "vec_ext_v2si" => (Ia32Op::VecExtV2Si, &[Ty::LongLong, Ty::Int], Ty::Int),
            // ---- SSE2: the 128-bit integer lanes ----
            "lfence" => (Ia32Op::NoOperand { mnemonic: "lfence" }, &[], Ty::Void),
            "mfence" => (Ia32Op::NoOperand { mnemonic: "mfence" }, &[], Ty::Void),
            "pandn128" => (Ia32Op::Vec128Bin { op: Vec128BinOp::AndN }, &[v2di(), v2di()], v2di()),
            "paddsb128" => (Ia32Op::Vec128Sat { add: true, unsigned: false }, &[v16qi(), v16qi()], v16qi()),
            "paddsw128" => (Ia32Op::Vec128Sat { add: true, unsigned: false }, &[v8hi(), v8hi()], v8hi()),
            "paddusb128" => (Ia32Op::Vec128Sat { add: true, unsigned: true }, &[v16qi(), v16qi()], v16qi()),
            "paddusw128" => (Ia32Op::Vec128Sat { add: true, unsigned: true }, &[v8hi(), v8hi()], v8hi()),
            "psubsb128" => (Ia32Op::Vec128Sat { add: false, unsigned: false }, &[v16qi(), v16qi()], v16qi()),
            "psubsw128" => (Ia32Op::Vec128Sat { add: false, unsigned: false }, &[v8hi(), v8hi()], v8hi()),
            "psubusb128" => (Ia32Op::Vec128Sat { add: false, unsigned: true }, &[v16qi(), v16qi()], v16qi()),
            "psubusw128" => (Ia32Op::Vec128Sat { add: false, unsigned: true }, &[v8hi(), v8hi()], v8hi()),
            "psllw128" => (Ia32Op::Vec128Shift { dir: Vec128ShiftDir::Ll, imm: false }, &[v8hi(), v8hi()], v8hi()),
            "psllwi128" => (Ia32Op::Vec128Shift { dir: Vec128ShiftDir::Ll, imm: true }, &[v8hi(), Ty::Int], v8hi()),
            "pslld128" => (Ia32Op::Vec128Shift { dir: Vec128ShiftDir::Ll, imm: false }, &[v4si(), v4si()], v4si()),
            "pslldi128" => (Ia32Op::Vec128Shift { dir: Vec128ShiftDir::Ll, imm: true }, &[v4si(), Ty::Int], v4si()),
            "psllq128" => (Ia32Op::Vec128Shift { dir: Vec128ShiftDir::Ll, imm: false }, &[v2di(), v2di()], v2di()),
            "psllqi128" => (Ia32Op::Vec128Shift { dir: Vec128ShiftDir::Ll, imm: true }, &[v2di(), Ty::Int], v2di()),
            "psrlw128" => (Ia32Op::Vec128Shift { dir: Vec128ShiftDir::Lr, imm: false }, &[v8hi(), v8hi()], v8hi()),
            "psrlwi128" => (Ia32Op::Vec128Shift { dir: Vec128ShiftDir::Lr, imm: true }, &[v8hi(), Ty::Int], v8hi()),
            "psrld128" => (Ia32Op::Vec128Shift { dir: Vec128ShiftDir::Lr, imm: false }, &[v4si(), v4si()], v4si()),
            "psrldi128" => (Ia32Op::Vec128Shift { dir: Vec128ShiftDir::Lr, imm: true }, &[v4si(), Ty::Int], v4si()),
            "psrlq128" => (Ia32Op::Vec128Shift { dir: Vec128ShiftDir::Lr, imm: false }, &[v2di(), v2di()], v2di()),
            "psrlqi128" => (Ia32Op::Vec128Shift { dir: Vec128ShiftDir::Lr, imm: true }, &[v2di(), Ty::Int], v2di()),
            "psraw128" => (Ia32Op::Vec128Shift { dir: Vec128ShiftDir::Ar, imm: false }, &[v8hi(), v8hi()], v8hi()),
            "psrawi128" => (Ia32Op::Vec128Shift { dir: Vec128ShiftDir::Ar, imm: true }, &[v8hi(), Ty::Int], v8hi()),
            "psrad128" => (Ia32Op::Vec128Shift { dir: Vec128ShiftDir::Ar, imm: false }, &[v4si(), v4si()], v4si()),
            "psradi128" => (Ia32Op::Vec128Shift { dir: Vec128ShiftDir::Ar, imm: true }, &[v4si(), Ty::Int], v4si()),
            "pcmpeqb128" => (Ia32Op::Vec128Cmp { eq: true }, &[v16qi(), v16qi()], v16qi()),
            "pcmpeqw128" => (Ia32Op::Vec128Cmp { eq: true }, &[v8hi(), v8hi()], v8hi()),
            "pcmpeqd128" => (Ia32Op::Vec128Cmp { eq: true }, &[v4si(), v4si()], v4si()),
            "pcmpgtb128" => (Ia32Op::Vec128Cmp { eq: false }, &[v16qi(), v16qi()], v16qi()),
            "pcmpgtw128" => (Ia32Op::Vec128Cmp { eq: false }, &[v8hi(), v8hi()], v8hi()),
            "pcmpgtd128" => (Ia32Op::Vec128Cmp { eq: false }, &[v4si(), v4si()], v4si()),
            "pmaxsw128" => (Ia32Op::Vec128MinMax { min: false, signed: true }, &[v8hi(), v8hi()], v8hi()),
            "pmaxub128" => (Ia32Op::Vec128MinMax { min: false, signed: false }, &[v16qi(), v16qi()], v16qi()),
            "pminsw128" => (Ia32Op::Vec128MinMax { min: true, signed: true }, &[v8hi(), v8hi()], v8hi()),
            "pminub128" => (Ia32Op::Vec128MinMax { min: true, signed: false }, &[v16qi(), v16qi()], v16qi()),
            "pavgb128" => (Ia32Op::Vec128Avg, &[v16qi(), v16qi()], v16qi()),
            "pavgw128" => (Ia32Op::Vec128Avg, &[v8hi(), v8hi()], v8hi()),
            "packsswb128" => (Ia32Op::Vec128Pack { kind: Vec128PackKind::SignedByte }, &[v8hi(), v8hi()], v16qi()),
            "packssdw128" => (Ia32Op::Vec128Pack { kind: Vec128PackKind::SignedWord }, &[v4si(), v4si()], v8hi()),
            "packuswb128" => (Ia32Op::Vec128Pack { kind: Vec128PackKind::UnsignedByte }, &[v8hi(), v8hi()], v16qi()),
            "pmulhw128" => (Ia32Op::Vec128Mul { high: true }, &[v8hi(), v8hi()], v8hi()),
            "pmulhuw128" => (Ia32Op::Vec128Mul { high: false }, &[v8hi(), v8hi()], v8hi()),
            "pmaddwd128" => (Ia32Op::Vec128Madd, &[v8hi(), v8hi()], v4si()),
            "pmuludq128" => (Ia32Op::Vec128MullDq, &[v4si(), v4si()], v2di()),
            "psadbw128" => (Ia32Op::Vec128Sad, &[v16qi(), v16qi()], v2di()),
            "punpckhbw128" => (Ia32Op::Vec128Unpck { high: true }, &[v16qi(), v16qi()], v16qi()),
            "punpckhwd128" => (Ia32Op::Vec128Unpck { high: true }, &[v8hi(), v8hi()], v8hi()),
            "punpckhdq128" => (Ia32Op::Vec128Unpck { high: true }, &[v4si(), v4si()], v4si()),
            "punpckhqdq128" => (Ia32Op::Vec128Unpck { high: true }, &[v2di(), v2di()], v2di()),
            "punpcklbw128" => (Ia32Op::Vec128Unpck { high: false }, &[v16qi(), v16qi()], v16qi()),
            "punpcklwd128" => (Ia32Op::Vec128Unpck { high: false }, &[v8hi(), v8hi()], v8hi()),
            "punpckldq128" => (Ia32Op::Vec128Unpck { high: false }, &[v4si(), v4si()], v4si()),
            "punpcklqdq128" => (Ia32Op::Vec128Unpck { high: false }, &[v2di(), v2di()], v2di()),
            "pshufd" => (Ia32Op::Vec128Shuf { mode: Vec128ShufMode::D }, &[v4si(), Ty::Int], v4si()),
            "pshuflw" => (Ia32Op::Vec128Shuf { mode: Vec128ShufMode::LowW }, &[v8hi(), Ty::Int], v8hi()),
            "pshufhw" => (Ia32Op::Vec128Shuf { mode: Vec128ShufMode::HighW }, &[v8hi(), Ty::Int], v8hi()),
            "vec_ext_v4si" => (Ia32Op::VecExtV4Si, &[v4si(), Ty::Int], Ty::Int),
            "pmovmskb128" => (Ia32Op::Pmovmskb128, &[v16qi()], Ty::Int),
            "movq128" => (Ia32Op::Movq128, &[v2di()], v2di()),
            "movnti" => (Ia32Op::MovNt { bytes: 4 }, &[ptr_to(Ty::Int), Ty::Int], Ty::Void),
            "movnti64" => (Ia32Op::MovNt { bytes: 8 }, &[ptr_to(Ty::LongLong), Ty::LongLong], Ty::Void),
            "movntq" => (Ia32Op::MovNt { bytes: 8 }, &[ptr_to(Ty::UnsignedLongLong), Ty::UnsignedLongLong], Ty::Void),
            "movntdq" => (Ia32Op::MovNt { bytes: 16 }, &[ptr_to(v2di()), v2di()], Ty::Void),
            "movntpd" => (Ia32Op::MovNt { bytes: 16 }, &[ptr_to(Ty::Double), v2df()], Ty::Void),
            "movntps" => (Ia32Op::MovNt { bytes: 16 }, &[ptr_to(Ty::Float), v4sf()], Ty::Void),
            "maskmovdqu" => (Ia32Op::MaskMov, &[v16qi(), v16qi(), ptr_to(Ty::Char)], Ty::Void),
            "maskmovq" => (Ia32Op::MaskMov, &[v8qi(), v8qi(), ptr_to(Ty::Char)], Ty::Void),
            "pmulhuw" => (Ia32Op::MmxOp { mnemonic: "pmulhuw" }, &[v4hi(), v4hi()], v4hi()),
            "pmuludq" => (Ia32Op::MmxOp { mnemonic: "pmuludq" }, &[v2si(), v2si()], vec(Ty::LongLong, 1)),
            "undef128" => (Ia32Op::Undef, &[], v4sf()),
            "pslldqi128_byteshift" => (Ia32Op::VecByteShift { left: true }, &[v16qi(), Ty::Int], v16qi()),
            "psrldqi128_byteshift" => (Ia32Op::VecByteShift { left: false }, &[v16qi(), Ty::Int], v16qi()),
            "urdmsr" => (Ia32Op::URdMsr, &[Ty::UnsignedLongLong], Ty::UnsignedLongLong),
            "uwrmsr" => (Ia32Op::UWrMsr, &[Ty::UnsignedLongLong, Ty::UnsignedLongLong], Ty::Void),
            "xsusldtrk" => (Ia32Op::NoOperand { mnemonic: "xsusldtrk" }, &[], Ty::Void),
            "xsetbv" => (Ia32Op::XSetBv, &[Ty::UnsignedInt, Ty::LongLong], Ty::Void),
            "umonitor" => (Ia32Op::UMonitor, &[ptr_to(Ty::Void)], Ty::Void),
            "umwait" => (Ia32Op::UWait { tpause: false }, &[Ty::UnsignedInt, Ty::UnsignedLongLong], Ty::UnsignedInt),
            "tpause" => (Ia32Op::UWait { tpause: true }, &[Ty::UnsignedInt, Ty::UnsignedLongLong], Ty::UnsignedInt),
            "enqcmd" => (Ia32Op::EnqCmd { ordered: false }, &[ptr_to(Ty::Void), const_ptr_to(Ty::Void)], Ty::Int),
            "enqcmds" => (Ia32Op::EnqCmd { ordered: true }, &[ptr_to(Ty::Void), const_ptr_to(Ty::Void)], Ty::Int),
            "llwpcb" => (Ia32Op::Llwpcb, &[ptr_to(Ty::Void)], Ty::Void),
            "slwpcb" => (Ia32Op::Slwpcb, &[], ptr_to(Ty::Void)),
            "monitor" => (Ia32Op::Monitor { extended: false }, &[const_ptr_to(Ty::Void), Ty::UnsignedInt, Ty::UnsignedInt], Ty::Void),
            "monitorx" => (Ia32Op::Monitor { extended: true }, &[const_ptr_to(Ty::Void), Ty::UnsignedInt, Ty::UnsignedInt], Ty::Void),
            "mwait" => (Ia32Op::Mwait { extended: false }, &[Ty::UnsignedInt, Ty::UnsignedInt], Ty::Void),
            "mwaitx" => (Ia32Op::Mwait { extended: true }, &[Ty::UnsignedInt, Ty::UnsignedInt, Ty::UnsignedInt], Ty::Void),
            "rdpkru" => (Ia32Op::Rdpkru, &[], Ty::UnsignedInt),
            "wrpkru" => (Ia32Op::Wrpkru, &[Ty::UnsignedInt], Ty::Void),
            "rdseed_hi_step" => (Ia32Op::RdStep { rand: false, width: 2 }, &[ptr_to(Ty::UnsignedShort)], Ty::Int),
            "rdseed_si_step" => (Ia32Op::RdStep { rand: false, width: 4 }, &[ptr_to(Ty::UnsignedInt)], Ty::Int),
            "rdseed_di_step" => (Ia32Op::RdStep { rand: false, width: 8 }, &[ptr_to(Ty::UnsignedLongLong)], Ty::Int),
            "rdrand16_step" => (Ia32Op::RdStep { rand: true, width: 2 }, &[ptr_to(Ty::UnsignedShort)], Ty::Int),
            "rdrand32_step" => (Ia32Op::RdStep { rand: true, width: 4 }, &[ptr_to(Ty::UnsignedInt)], Ty::Int),
            "rdrand64_step" => (Ia32Op::RdStep { rand: true, width: 8 }, &[ptr_to(Ty::UnsignedLongLong)], Ty::Int),
            _ => return None,
        };
        if args.len() != params.len() {
            return Some(Err(CompileError::new(
                format!("`{name}` takes exactly {} argument{}", params.len(), if params.len() == 1 { "" } else { "s" }),
                span,
            )));
        }
        let checked = args
            .iter()
            .zip(params)
            .map(|(arg, param)| self.check_expr(arg, interner).and_then(|arg| self.convert_ia32_arg(param, arg, interner)))
            .collect::<Result<Vec<_>, _>>();
        Some(checked.map(|args| CheckedExpr {
            ty: ret,
            span,
            kind: CheckedExprKind::Ia32 {
                op,
                args: self.alloc_slice(args),
            },
        }))
    }

    /// Convert one builtin argument. Beyond the ordinary assignment
    /// rules, GNU vector operands retype to the table's spelling when
    /// the sizes agree: the intrinsics pass `__m64`/`__m128`-class
    /// types interchangeably (`__v2si` against `__m64`, `__m128i`
    /// against `__m128`), and the lowering moves the bits whole — the
    /// MMX shapes lower as integers of the vector's size anyway.
    fn convert_ia32_arg(&self, param: &Ty, arg: CheckedExpr<'a>, interner: &Interner) -> Result<CheckedExpr<'a>, CompileError> {
        let same_size_vector_pair = |a: &Ty, b: &Ty| {
            (is_vector(a) || is_vector(b)) && !is_complex(a) && !is_complex(b) && self.type_size(a, interner) == self.type_size(b, interner)
        };
        let vector_retype = same_size_vector_pair(param, &arg.ty);
        match self.convert_for_assignment(param, arg.clone(), interner) {
            Ok(converted) => Ok(converted),
            Err(e) => {
                if vector_retype {
                    let CheckedExpr { span, kind, .. } = arg;
                    return Ok(CheckedExpr {
                        ty: param.clone(),
                        span,
                        kind,
                    });
                }
                Err(e)
            }
        }
    }
}
