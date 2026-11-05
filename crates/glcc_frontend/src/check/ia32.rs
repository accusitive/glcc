//! Checking for GCC's x86 `__builtin_ia32_*` builtins. Only the
//! spellings with real lowering exist here; each checks inline into an
//! [`Ia32Op`] node (no declared prototype), like the `__sync` family.
//! The lowering lives in `glcc_lower_mir`'s `lower/ia32.rs`: pure MIR
//! sequences where possible (`bsrsi` off `clz`, the rotates off
//! shifts), inline asm for the instructions MIR has no opcode for
//! (`rdtsc`/`rdtscp`/`rdpmc`, the SSE4.2 CRC-32C step).

use super::*;

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
}

/// The pointer parameters of the `sbb`/`addcarryx` builtins.
fn ptr_to(pointee: Ty) -> Ty {
    Ty::Pointer {
        qualifiers: TypeQualifiers::empty(),
        size: PtrSize::Default,
        pointee: Box::new(pointee),
    }
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
            "clflush" => (Ia32Op::CacheOp { mnemonic: "clflush" }, &[ptr_to(Ty::Void)], Ty::Void),
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
            "paddb" => (Ia32Op::MmxOp { mnemonic: "paddb" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "paddw" => (Ia32Op::MmxOp { mnemonic: "paddw" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "paddd" => (Ia32Op::MmxOp { mnemonic: "paddd" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "paddq" => (Ia32Op::MmxOp { mnemonic: "paddq" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "paddsb" => (Ia32Op::MmxOp { mnemonic: "paddsb" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "paddsw" => (Ia32Op::MmxOp { mnemonic: "paddsw" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "paddusb" => (Ia32Op::MmxOp { mnemonic: "paddusb" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "paddusw" => (Ia32Op::MmxOp { mnemonic: "paddusw" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "psubb" => (Ia32Op::MmxOp { mnemonic: "psubb" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "psubw" => (Ia32Op::MmxOp { mnemonic: "psubw" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "psubd" => (Ia32Op::MmxOp { mnemonic: "psubd" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "psubq" => (Ia32Op::MmxOp { mnemonic: "psubq" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "psubsb" => (Ia32Op::MmxOp { mnemonic: "psubsb" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "psubsw" => (Ia32Op::MmxOp { mnemonic: "psubsw" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "psubusb" => (Ia32Op::MmxOp { mnemonic: "psubusb" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "psubusw" => (Ia32Op::MmxOp { mnemonic: "psubusw" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "pand" => (Ia32Op::MmxOp { mnemonic: "pand" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "pandn" => (Ia32Op::MmxOp { mnemonic: "pandn" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "por" => (Ia32Op::MmxOp { mnemonic: "por" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "pxor" => (Ia32Op::MmxOp { mnemonic: "pxor" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "pcmpeqb" => (Ia32Op::MmxOp { mnemonic: "pcmpeqb" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "pcmpeqw" => (Ia32Op::MmxOp { mnemonic: "pcmpeqw" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "pcmpeqd" => (Ia32Op::MmxOp { mnemonic: "pcmpeqd" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "pcmpgtb" => (Ia32Op::MmxOp { mnemonic: "pcmpgtb" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "pcmpgtw" => (Ia32Op::MmxOp { mnemonic: "pcmpgtw" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "pcmpgtd" => (Ia32Op::MmxOp { mnemonic: "pcmpgtd" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "pmulhw" => (Ia32Op::MmxOp { mnemonic: "pmulhw" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "pmullw" => (Ia32Op::MmxOp { mnemonic: "pmullw" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "pmaddwd" => (Ia32Op::MmxOp { mnemonic: "pmaddwd" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "punpckhbw" => (Ia32Op::MmxOp { mnemonic: "punpckhbw" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "punpckhwd" => (Ia32Op::MmxOp { mnemonic: "punpckhwd" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "punpckhdq" => (Ia32Op::MmxOp { mnemonic: "punpckhdq" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "punpcklbw" => (Ia32Op::MmxOp { mnemonic: "punpcklbw" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "punpcklwd" => (Ia32Op::MmxOp { mnemonic: "punpcklwd" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "punpckldq" => (Ia32Op::MmxOp { mnemonic: "punpckldq" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "packsswb" => (Ia32Op::MmxOp { mnemonic: "packsswb" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "packssdw" => (Ia32Op::MmxOp { mnemonic: "packssdw" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "packuswb" => (Ia32Op::MmxOp { mnemonic: "packuswb" }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "psllw" => (Ia32Op::MmxShift { mnemonic: "psllw", imm: false }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "pslld" => (Ia32Op::MmxShift { mnemonic: "pslld", imm: false }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "psllq" => (Ia32Op::MmxShift { mnemonic: "psllq", imm: false }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "psrlw" => (Ia32Op::MmxShift { mnemonic: "psrlw", imm: false }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "psrld" => (Ia32Op::MmxShift { mnemonic: "psrld", imm: false }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "psrlq" => (Ia32Op::MmxShift { mnemonic: "psrlq", imm: false }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "psraw" => (Ia32Op::MmxShift { mnemonic: "psraw", imm: false }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "psrad" => (Ia32Op::MmxShift { mnemonic: "psrad", imm: false }, &[Ty::LongLong, Ty::LongLong], Ty::LongLong),
            "psllwi" => (Ia32Op::MmxShift { mnemonic: "psllw", imm: true }, &[Ty::LongLong, Ty::Int], Ty::LongLong),
            "pslldi" => (Ia32Op::MmxShift { mnemonic: "pslld", imm: true }, &[Ty::LongLong, Ty::Int], Ty::LongLong),
            "psllqi" => (Ia32Op::MmxShift { mnemonic: "psllq", imm: true }, &[Ty::LongLong, Ty::Int], Ty::LongLong),
            "psrlwi" => (Ia32Op::MmxShift { mnemonic: "psrlw", imm: true }, &[Ty::LongLong, Ty::Int], Ty::LongLong),
            "psrldi" => (Ia32Op::MmxShift { mnemonic: "psrld", imm: true }, &[Ty::LongLong, Ty::Int], Ty::LongLong),
            "psrlqi" => (Ia32Op::MmxShift { mnemonic: "psrlq", imm: true }, &[Ty::LongLong, Ty::Int], Ty::LongLong),
            "psrawi" => (Ia32Op::MmxShift { mnemonic: "psraw", imm: true }, &[Ty::LongLong, Ty::Int], Ty::LongLong),
            "psradi" => (Ia32Op::MmxShift { mnemonic: "psrad", imm: true }, &[Ty::LongLong, Ty::Int], Ty::LongLong),
                        "vec_init_v2si" => (Ia32Op::VecInitV2Si, &[Ty::Int, Ty::Int], Ty::Int),
            "vec_init_v4hi" => (Ia32Op::VecInit { lanes: 2 }, &[Ty::Int, Ty::Int, Ty::Int, Ty::Int], Ty::LongLong),
            "vec_init_v8qi" => (Ia32Op::VecInit { lanes: 1 }, &[Ty::Int, Ty::Int, Ty::Int, Ty::Int, Ty::Int, Ty::Int, Ty::Int, Ty::Int], Ty::LongLong),
            "vec_ext_v2si" => (Ia32Op::VecExtV2Si, &[Ty::LongLong, Ty::Int], Ty::Int),
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
            .map(|(arg, param)| self.check_expr(arg, interner).and_then(|arg| self.convert_for_assignment(param, arg, interner)))
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
}
