//! Target machine description: sizes, alignments, and related facts.
//!
//! `Ty` is target-independent; every byte size, alignment, and
//! representation choice lives here instead, so retargeting means
//! describing a new [`Target`], not hunting assumptions through the
//! frontend. `Sema` carries one (`Sema::target`, LP64 by default — see
//! `Sema::new`), and layout/sizeof queries read it.

/// Machine description for layout and `sizeof`.
///
/// All sizes and alignments are in bytes. The associated constants
/// describe the five targets glcc supports: [`Target::X86_64`]
/// (x86-64 System V), [`Target::AARCH64`] (AArch64 AAPCS64),
/// [`Target::RISCV64`] (RISC-V psABI, LP64D),
/// [`Target::WINDOWS_X86_64`] (Windows LLP64, MSVC flavor), and
/// [`Target::WINDOWS_X86_64_GNU`] (Windows LLP64, mingw-w64 flavor).
/// The three Linux
/// targets agree on integer and pointer sizes; they differ in
/// plain-`char` signedness (signed on x86-64, unsigned on
/// AArch64/RISC-V per clang), in `long double` representation
/// (x86 80-bit extended vs IEEE quad), and in `va_list` shape
/// (24 vs 32 vs 8 bytes). Windows differs in data model (`long` is
/// 4 bytes), `long double` (`double`), `wchar_t` (16-bit), and
/// `va_list` (8-byte pointer); the two Windows flavors differ only
/// in the toolchain around the object files (see [`Env`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    /// Which instruction set / ABI family this describes.
    pub arch: Arch,
    /// Operating system / ABI family. Independent of `arch` because the
    /// same instruction set can carry different data models (LP64 vs
    /// LLP64) and calling conventions.
    pub os: Os,
    /// Toolchain flavor of a Windows target: whose CRT and link driver
    /// the objects are meant for. The code-visible ABI is identical
    /// (same calling convention, same object format), so this is only
    /// consulted by the driver (link command, predefines, include
    /// paths) and by the triple spellings; Linux targets carry a
    /// placeholder that nothing reads.
    pub env: Env,
    /// How `long double` values behave: x86 80-bit extended (`F80`),
    /// IEEE-754 quad precision (`F128`, AArch64 and RISC-V), or plain
    /// `double` (`F64`, Windows). Size and alignment are 16/16 in the
    /// first two cases and 8/8 for `F64`; only the representation (and
    /// therefore constant folding and codegen) differs.
    pub long_double: LongDoubleKind,
    /// Size and alignment of `__builtin_va_list` (24 on x86-64
    /// SysV, 32 on AArch64, a single pointer on RISC-V and Windows).
    pub va_list_size: u64,
    pub va_list_align: u64,
    pub bool_size: u64,
    pub bool_align: u64,
    pub char_size: u64,
    pub char_align: u64,
    /// Whether plain `char` is signed (true on x86-64; C leaves it
    /// implementation-defined).
    pub char_signed: bool,
    /// Whether `wchar_t` is signed (`int` on x86-64 and RISC-V;
    /// `unsigned int` on AArch64; `unsigned short` on Windows, per
    /// `__WCHAR_TYPE__`).
    pub wchar_signed: bool,
    /// Width of `wchar_t` in bytes: 4 everywhere but Windows, whose
    /// `unsigned short` is 16-bit. Declarations get the type from the
    /// header typedef; wide literals (`L"..."`, `L'x'`) read it here.
    pub wchar_size: u64,
    pub short_size: u64,
    pub short_align: u64,
    pub int_size: u64,
    pub int_align: u64,
    pub long_size: u64,
    pub long_align: u64,
    pub long_long_size: u64,
    pub long_long_align: u64,
    /// GCC's 128-bit integers (`__int128`, used by kernel headers for
    /// `__s128`/`__u128`): 16 bytes, 16-aligned on x86-64 SysV.
    pub int128_size: u64,
    pub int128_align: u64,
    pub float_size: u64,
    pub float_align: u64,
    pub double_size: u64,
    pub double_align: u64,
    pub long_double_size: u64,
    pub long_double_align: u64,
    pub pointer_size: u64,
    pub pointer_align: u64,
    /// Enums are lowered like `int` (compatible with it for
    /// conversions); layout gives them `int` size and alignment.
    pub enum_size: u64,
    pub enum_align: u64,
    /// The alignment a bare `__attribute__((aligned))` requests (GCC's
    /// maximum useful alignment: 16 on x86-64 System V).
    pub max_useful_align: u64,
    /// Byte order, deciding bit-field allocation order within a storage
    /// unit (layout allocates from the low bit on little-endian).
    pub little_endian: bool,
}

/// Instruction set family. Data model and ABI are decided by `arch`
/// together with [`Os`] — x86-64 carries both LP64 Linux and LLP64
/// Windows variants — and the backend maps each combination to its
/// Cranelift ISA/ABI lowering. Every supported target is
/// little-endian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    X86_64,
    Aarch64,
    Riscv64,
}

/// Operating system / ABI family. Selects the data model where it
/// differs from LP64 (`long`, `wchar_t`, `size_t`) and the calling
/// convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    Linux,
    /// Windows x86-64 (LLP64, Microsoft ABI).
    Windows,
}

/// Windows toolchain flavor: Microsoft's CRT (`link.exe`, `_MSC_VER`
/// predefines) or the mingw-w64 GNU toolchain (`x86_64-w64-mingw32-gcc`
/// link driver, `__MINGW64__` predefines). Both flavors run the same
/// Microsoft x64 calling convention over COFF objects; only the
/// surrounding toolchain — predefines, system headers, crt files,
/// default libraries, runtime object — differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Env {
    /// Microsoft Visual C++ toolchain (the native flavor on Windows
    /// hosts, and the placeholder on Linux targets).
    Msvc,
    /// mingw-w64 GNU toolchain (`x86_64-w64-mingw32` triples).
    Gnu,
}

/// Representation of C `long double`: x86 80-bit extended precision
/// (`F80`), IEEE-754 binary128 quad precision (`F128`, used on
/// AArch64 and RISC-V where `long double` is an alias for
/// `__float128`), or the same as `double` (`F64`, on Windows).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LongDoubleKind {
    F80,
    F128,
    F64,
}

/// `wchar_t` as the string-literal and character-constant machinery
/// needs it: width in bytes plus signedness, from the target's
/// `__WCHAR_TYPE__` (`unsigned short` on Windows, `int`/`unsigned int`
/// elsewhere).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WChar {
    pub size: u64,
    pub signed: bool,
}

impl WChar {
    /// The `Ty` one wide element gets (the element type of `L"..."`
    /// and the type of `L'x'`).
    pub fn ty(self) -> crate::parse::Ty {
        match (self.size, self.signed) {
            (2, true) => crate::parse::Ty::Short,
            (2, false) => crate::parse::Ty::UnsignedShort,
            (_, true) => crate::parse::Ty::Int,
            (_, false) => crate::parse::Ty::UnsignedInt,
        }
    }
}

impl Target {
    /// x86-64 System V (LP64): signed plain `char`, 80-bit extended
    /// `long double`, 24-byte SysV `__va_list_tag`.
    pub const X86_64: Target = Target {
        arch: Arch::X86_64,
        os: Os::Linux,
        env: Env::Msvc,
        long_double: LongDoubleKind::F80,
        va_list_size: 24,
        // System V only requires 8, but the backend spills 16-byte SSE
        // slots through the save area and sizes the local `va_list`
        // slot at 16; keep the historical over-alignment.
        va_list_align: 16,
        bool_size: 1,
        bool_align: 1,
        char_size: 1,
        char_align: 1,
        char_signed: true,
        wchar_signed: true,
        wchar_size: 4,
        short_size: 2,
        short_align: 2,
        int_size: 4,
        int_align: 4,
        long_size: 8,
        long_align: 8,
        long_long_size: 8,
        long_long_align: 8,
        int128_size: 16,
        int128_align: 16,
        float_size: 4,
        float_align: 4,
        double_size: 8,
        double_align: 8,
        long_double_size: 16,
        long_double_align: 16,
        pointer_size: 8,
        pointer_align: 8,
        enum_size: 4,
        enum_align: 4,
        max_useful_align: 16,
        little_endian: true,
    };

    /// Previous name for [`Target::X86_64`], kept for existing callers.
    pub const LP64: Target = Target::X86_64;

    /// AArch64 Linux (AAPCS64, LP64): unsigned plain `char` (verified
    /// via `clang --target=aarch64-linux-gnu`: `(char)-1` is 255),
    /// IEEE-quad `long double` (`__LDBL_MANT_DIG__` 113), 32-byte
    /// `struct __va_list`.
    pub const AARCH64: Target = Target {
        arch: Arch::Aarch64,
        os: Os::Linux,
        env: Env::Msvc,
        long_double: LongDoubleKind::F128,
        va_list_size: 32,
        va_list_align: 8,
        bool_size: 1,
        bool_align: 1,
        char_size: 1,
        char_align: 1,
        char_signed: false,
        wchar_signed: false,
        wchar_size: 4,
        short_size: 2,
        short_align: 2,
        int_size: 4,
        int_align: 4,
        long_size: 8,
        long_align: 8,
        long_long_size: 8,
        long_long_align: 8,
        int128_size: 16,
        int128_align: 16,
        float_size: 4,
        float_align: 4,
        double_size: 8,
        double_align: 8,
        long_double_size: 16,
        long_double_align: 16,
        pointer_size: 8,
        pointer_align: 8,
        enum_size: 4,
        enum_align: 4,
        max_useful_align: 16,
        little_endian: true,
    };

    /// RISC-V 64 Linux (psABI, LP64D): unsigned plain `char`,
    /// IEEE-quad `long double`, pointer-sized `va_list` (a single
    /// `void *`; `sizeof(__builtin_va_list)` is 8 under clang).
    pub const RISCV64: Target = Target {
        arch: Arch::Riscv64,
        os: Os::Linux,
        env: Env::Msvc,
        long_double: LongDoubleKind::F128,
        va_list_size: 8,
        va_list_align: 8,
        bool_size: 1,
        bool_align: 1,
        char_size: 1,
        char_align: 1,
        char_signed: false,
        wchar_signed: true,
        wchar_size: 4,
        short_size: 2,
        short_align: 2,
        int_size: 4,
        int_align: 4,
        long_size: 8,
        long_align: 8,
        long_long_size: 8,
        long_long_align: 8,
        int128_size: 16,
        int128_align: 16,
        float_size: 4,
        float_align: 4,
        double_size: 8,
        double_align: 8,
        long_double_size: 16,
        long_double_align: 16,
        pointer_size: 8,
        pointer_align: 8,
        enum_size: 4,
        enum_align: 4,
        max_useful_align: 16,
        little_endian: true,
    };

    /// x86-64 Windows (LLP64, Microsoft x64 ABI): signed plain
    /// `char`, 32-bit `long`, `long double` == `double`, 16-bit
    /// unsigned `wchar_t`, `size_t` == `unsigned long long`, and a
    /// pointer-sized (8-byte) `char *` `__builtin_va_list`.
    ///
    /// Verified with `clang --target=x86_64-pc-windows-msvc -dM`:
    /// `__SIZEOF_LONG__` 4, `__SIZEOF_LONG_DOUBLE__` 8,
    /// `__SIZEOF_WCHAR_T__` 2, `__WCHAR_TYPE__` `unsigned short`.
    pub const WINDOWS_X86_64: Target = Target {
        arch: Arch::X86_64,
        os: Os::Windows,
        env: Env::Msvc,
        long_double: LongDoubleKind::F64,
        va_list_size: 8,
        va_list_align: 8,
        bool_size: 1,
        bool_align: 1,
        char_size: 1,
        char_align: 1,
        char_signed: true,
        wchar_signed: false,
        wchar_size: 2,
        short_size: 2,
        short_align: 2,
        int_size: 4,
        int_align: 4,
        long_size: 4,
        long_align: 4,
        long_long_size: 8,
        long_long_align: 8,
        int128_size: 16,
        int128_align: 16,
        float_size: 4,
        float_align: 4,
        double_size: 8,
        double_align: 8,
        long_double_size: 8,
        long_double_align: 8,
        pointer_size: 8,
        pointer_align: 8,
        enum_size: 4,
        enum_align: 4,
        max_useful_align: 16,
        little_endian: true,
    };

    /// x86-64 Windows, mingw-w64 flavor: the same data model and ABI as
    /// [`Target::WINDOWS_X86_64`] (the two flavors are indistinguishable
    /// at the C level); only [`Target::env`] differs, steering the
    /// predefines, include paths, and link command.
    pub const WINDOWS_X86_64_GNU: Target = Target {
        env: Env::Gnu,
        ..Target::WINDOWS_X86_64
    };

    /// Native target of this build (host architecture and OS): a
    /// Windows host compiles the Windows data model and ABI by default.
    pub fn native() -> Target {
        #[cfg(target_arch = "aarch64")]
        return Target::AARCH64;
        #[cfg(target_arch = "riscv64")]
        return Target::RISCV64;
        #[cfg(all(target_arch = "x86_64", windows))]
        return Target::WINDOWS_X86_64;
        #[cfg(all(target_arch = "x86_64", not(windows)))]
        return Target::X86_64;
        // wasm32 (the in-browser compiler): the page always selects an
        // explicit codegen target, and x86_64 is its default.
        #[cfg(not(any(target_arch = "aarch64", target_arch = "riscv64", target_arch = "x86_64")))]
        return Target::X86_64;
    }

    /// Short name used by `--target`: `x86_64`, `aarch64`, `riscv64`
    /// (plus `x86_64-windows` and `x86_64-windows-gnu`).
    pub fn name(&self) -> &'static str {
        match (self.arch, self.os, self.env) {
            (Arch::X86_64, Os::Windows, Env::Gnu) => "x86_64-windows-gnu",
            (Arch::X86_64, Os::Windows, Env::Msvc) => "x86_64-windows",
            (Arch::X86_64, Os::Linux, _) => "x86_64",
            (Arch::Aarch64, _, _) => "aarch64",
            (Arch::Riscv64, _, _) => "riscv64",
        }
    }

    /// Triple forwarded to `clang --target=`, the system link driver,
    /// and cross tool names. The mingw flavor uses binutils' canonical
    /// `x86_64-w64-mingw32` spelling: it names the installed cross
    /// tools (`x86_64-w64-mingw32-gcc`) and is what clang's MinGW
    /// driver expects when finding the sysroot.
    pub fn triple(&self) -> &'static str {
        match (self.arch, self.os, self.env) {
            (Arch::X86_64, Os::Windows, Env::Gnu) => "x86_64-w64-mingw32",
            (Arch::X86_64, Os::Windows, Env::Msvc) => "x86_64-pc-windows-msvc",
            (Arch::X86_64, Os::Linux, _) => "x86_64-linux-gnu",
            (Arch::Aarch64, _, _) => "aarch64-linux-gnu",
            (Arch::Riscv64, _, _) => "riscv64-linux-gnu",
        }
    }

    /// Triple for Cranelift's ISA lookup. Same as [`Target::triple`]
    /// except the mingw flavor: target-lexicon (Cranelift's triple
    /// parser) knows `x86_64-pc-windows-gnu` but not the binutils
    /// `w64-mingw32` spelling, and the code-visible result is identical
    /// (x86-64 ISA over COFF; the calling convention is chosen
    /// separately).
    pub fn isa_triple(&self) -> &'static str {
        match (self.arch, self.os, self.env) {
            (Arch::X86_64, Os::Windows, Env::Gnu) => "x86_64-pc-windows-gnu",
            _ => self.triple(),
        }
    }

    /// Parse a `--target` value: a short name (`x86_64`, `aarch64`,
    /// `riscv64`, plus `x64`/`arm64` aliases) or a full triple
    /// starting with one (`aarch64-linux-gnu`, `riscv64-linux-gnueabihf`
    /// style spellings all map to the same `Target`). A triple naming
    /// Windows selects the LLP64 Windows target, with the flavor from
    /// its environment: `...-mingw32`/`...-gnu` is mingw-w64
    /// ([`Target::WINDOWS_X86_64_GNU`]), everything else
    /// (`...-msvc`, plain `x86_64-windows`) is MSVC
    /// ([`Target::WINDOWS_X86_64`]).
    pub fn from_name(name: &str) -> Option<Target> {
        let lower = name.to_ascii_lowercase();
        let os = if lower.contains("windows") || lower.contains("msvc") || lower.contains("mingw") {
            Os::Windows
        } else {
            Os::Linux
        };
        let env = if os == Os::Windows && (lower.contains("mingw") || lower.contains("gnu")) {
            Env::Gnu
        } else {
            Env::Msvc
        };
        let prefix = lower.split('-').next().unwrap_or(&lower);
        let arch = match prefix {
            "x86_64" | "x64" | "amd64" => Arch::X86_64,
            "aarch64" | "arm64" => Arch::Aarch64,
            "riscv64" => Arch::Riscv64,
            _ => return None,
        };
        Some(match (arch, os, env) {
            (Arch::X86_64, Os::Windows, Env::Gnu) => Target::WINDOWS_X86_64_GNU,
            (Arch::X86_64, Os::Windows, Env::Msvc) => Target::WINDOWS_X86_64,
            (Arch::X86_64, Os::Linux, _) => Target::X86_64,
            (Arch::Aarch64, _, _) => Target::AARCH64,
            (Arch::Riscv64, _, _) => Target::RISCV64,
        })
    }

    /// The `size_t` type: `unsigned long` on LP64, `unsigned long
    /// long` on Windows (LLP64). Both are 8 bytes; the distinction
    /// matters for type identity and `printf` checking.
    pub fn size_type(&self) -> crate::parse::Ty {
        match self.os {
            Os::Windows => crate::parse::Ty::UnsignedLongLong,
            Os::Linux => crate::parse::Ty::UnsignedLong,
        }
    }

    /// The `ptrdiff_t` type: `long` on LP64, `long long` on Windows
    /// (LLP64, `__PTRDIFF_TYPE__`). Both are 8 bytes.
    pub fn ptrdiff_type(&self) -> crate::parse::Ty {
        match self.os {
            Os::Windows => crate::parse::Ty::LongLong,
            Os::Linux => crate::parse::Ty::Long,
        }
    }

    /// `wchar_t` width and signedness (see [`WChar`]).
    pub fn wchar(&self) -> WChar {
        WChar {
            size: self.wchar_size,
            signed: self.wchar_signed,
        }
    }

    /// Whether this is the host target (no `--target=` forwarding needed).
    pub fn is_native(&self) -> bool {
        *self == Target::native()
    }
}

#[cfg(test)]
mod tests;

