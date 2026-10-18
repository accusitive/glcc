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
/// describe the four targets glcc supports: [`Target::X86_64`]
/// (x86-64 System V), [`Target::AARCH64`] (AArch64 AAPCS64),
/// [`Target::RISCV64`] (RISC-V psABI, LP64D), and
/// [`Target::WINDOWS_X86_64`] (Windows LLP64). The three Linux
/// targets agree on integer and pointer sizes; they differ in
/// plain-`char` signedness (signed on x86-64, unsigned on
/// AArch64/RISC-V per clang), in `long double` representation
/// (x86 80-bit extended vs IEEE quad), and in `va_list` shape
/// (24 vs 32 vs 8 bytes). Windows differs in data model (`long` is
/// 4 bytes), `long double` (`double`), `wchar_t` (16-bit), and
/// `va_list` (8-byte pointer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    /// Which instruction set / ABI family this describes.
    pub arch: Arch,
    /// Operating system / ABI family. Independent of `arch` because the
    /// same instruction set can carry different data models (LP64 vs
    /// LLP64) and calling conventions.
    pub os: Os,
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
    /// (plus `x86_64-windows`).
    pub fn name(&self) -> &'static str {
        match (self.arch, self.os) {
            (Arch::X86_64, Os::Windows) => "x86_64-windows",
            (Arch::X86_64, Os::Linux) => "x86_64",
            (Arch::Aarch64, _) => "aarch64",
            (Arch::Riscv64, _) => "riscv64",
        }
    }

    /// Triple forwarded to `clang --target=` and the system link
    /// driver, and to Cranelift's ISA lookup.
    pub fn triple(&self) -> &'static str {
        match (self.arch, self.os) {
            (Arch::X86_64, Os::Windows) => "x86_64-pc-windows-msvc",
            (Arch::X86_64, Os::Linux) => "x86_64-linux-gnu",
            (Arch::Aarch64, _) => "aarch64-linux-gnu",
            (Arch::Riscv64, _) => "riscv64-linux-gnu",
        }
    }

    /// Parse a `--target` value: a short name (`x86_64`, `aarch64`,
    /// `riscv64`, plus `x64`/`arm64` aliases) or a full triple
    /// starting with one (`aarch64-linux-gnu`, `riscv64-linux-gnueabihf`
    /// style spellings all map to the same `Target`). A triple naming
    /// Windows (`x86_64-pc-windows-msvc`, `...-windows-gnu`) selects
    /// the LLP64 [`Target::WINDOWS_X86_64`].
    pub fn from_name(name: &str) -> Option<Target> {
        let lower = name.to_ascii_lowercase();
        let os = if lower.contains("windows") || lower.contains("msvc") || lower.contains("mingw") {
            Os::Windows
        } else {
            Os::Linux
        };
        let prefix = lower.split('-').next().unwrap_or(&lower);
        let arch = match prefix {
            "x86_64" | "x64" | "amd64" => Arch::X86_64,
            "aarch64" | "arm64" => Arch::Aarch64,
            "riscv64" => Arch::Riscv64,
            _ => return None,
        };
        Some(match (arch, os) {
            (Arch::X86_64, Os::Windows) => Target::WINDOWS_X86_64,
            (Arch::X86_64, Os::Linux) => Target::X86_64,
            (Arch::Aarch64, _) => Target::AARCH64,
            (Arch::Riscv64, _) => Target::RISCV64,
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
mod tests {
    use super::*;

    #[test]
    fn short_names_and_triples_parse() {
        assert_eq!(Target::from_name("x86_64"), Some(Target::X86_64));
        assert_eq!(Target::from_name("aarch64"), Some(Target::AARCH64));
        assert_eq!(Target::from_name("riscv64"), Some(Target::RISCV64));
        assert_eq!(Target::from_name("x86_64-linux-gnu"), Some(Target::X86_64));
        assert_eq!(Target::from_name("aarch64-linux-gnu"), Some(Target::AARCH64));
        assert_eq!(Target::from_name("riscv64-linux-gnu"), Some(Target::RISCV64));
        assert_eq!(Target::from_name("arm64"), Some(Target::AARCH64));
        assert_eq!(Target::from_name("amd64"), Some(Target::X86_64));
        assert_eq!(Target::from_name("mips"), None);
        assert_eq!(Target::from_name(""), None);
    }

    #[test]
    fn lp64_sizes_agree_across_targets() {
        // Verified against `clang -E -dM` (`__SIZEOF_*__`) for all three
        // triples: every integer and pointer size matches.
        for t in [Target::X86_64, Target::AARCH64, Target::RISCV64] {
            assert_eq!((t.int_size, t.int_align), (4, 4));
            assert_eq!((t.long_size, t.long_align), (8, 8));
            assert_eq!((t.pointer_size, t.pointer_align), (8, 8));
            assert_eq!((t.long_double_size, t.long_double_align), (16, 16));
            assert!(t.little_endian);
        }
    }

    #[test]
    fn char_signedness_matches_clang() {
        // `(char)-1 < 0` holds only on x86-64; both other targets
        // define `__CHAR_UNSIGNED__`.
        assert!(Target::X86_64.char_signed);
        assert!(!Target::AARCH64.char_signed);
        assert!(!Target::RISCV64.char_signed);
    }

    #[test]
    fn long_double_kinds_match_clang() {
        // `__LDBL_MANT_DIG__` is 64 (extended) on x86-64, 113 (quad)
        // on the other two.
        assert_eq!(Target::X86_64.long_double, LongDoubleKind::F80);
        assert_eq!(Target::AARCH64.long_double, LongDoubleKind::F128);
        assert_eq!(Target::RISCV64.long_double, LongDoubleKind::F128);
    }

    #[test]
    fn va_list_shapes_match_clang() {
        // `sizeof(__builtin_va_list)` under clang for each triple.
        assert_eq!(Target::X86_64.va_list_size, 24);
        assert_eq!(Target::AARCH64.va_list_size, 32);
        assert_eq!(Target::RISCV64.va_list_size, 8);
    }

    #[test]
    fn triples_are_linux_gnu() {
        assert_eq!(Target::X86_64.triple(), "x86_64-linux-gnu");
        assert_eq!(Target::AARCH64.triple(), "aarch64-linux-gnu");
        assert_eq!(Target::RISCV64.triple(), "riscv64-linux-gnu");
    }

    #[test]
    fn windows_target_is_llp64() {
        // Verified with `clang --target=x86_64-pc-windows-msvc -dM`.
        let t = Target::WINDOWS_X86_64;
        assert_eq!((t.int_size, t.int_align), (4, 4));
        assert_eq!((t.long_size, t.long_align), (4, 4));
        assert_eq!((t.long_long_size, t.long_long_align), (8, 8));
        assert_eq!((t.pointer_size, t.pointer_align), (8, 8));
        assert_eq!((t.long_double_size, t.long_double_align), (8, 8));
        assert_eq!(t.long_double, LongDoubleKind::F64);
        assert!(t.char_signed);
        assert_eq!((t.wchar_size, t.wchar_signed), (2, false));
        assert_eq!(t.wchar().ty(), crate::parse::Ty::UnsignedShort);
        assert_eq!((t.va_list_size, t.va_list_align), (8, 8));
        assert!(t.little_endian);
    }

    #[test]
    fn wchar_shapes_match_clang() {
        // `__SIZEOF_WCHAR_T__`/`__WCHAR_TYPE__` per triple.
        assert_eq!(Target::X86_64.wchar(), WChar { size: 4, signed: true });
        assert_eq!(Target::AARCH64.wchar(), WChar { size: 4, signed: false });
        assert_eq!(Target::RISCV64.wchar(), WChar { size: 4, signed: true });
        assert_eq!(Target::X86_64.wchar().ty(), crate::parse::Ty::Int);
        assert_eq!(Target::AARCH64.wchar().ty(), crate::parse::Ty::UnsignedInt);
    }

    #[test]
    fn windows_target_name_triple_and_size_type() {
        let t = Target::WINDOWS_X86_64;
        assert_eq!(t.name(), "x86_64-windows");
        assert_eq!(t.triple(), "x86_64-pc-windows-msvc");
        assert_eq!(t.size_type(), crate::parse::Ty::UnsignedLongLong);
        assert_eq!(Target::X86_64.size_type(), crate::parse::Ty::UnsignedLong);
        for name in ["x86_64-windows", "x86_64-pc-windows-msvc", "x86_64-windows-gnu", "AMD64-MSVC"] {
            assert_eq!(Target::from_name(name), Some(t), "{name}");
        }
    }

    #[test]
    fn native_matches_the_host() {
        #[cfg(windows)]
        assert_eq!(Target::native(), Target::WINDOWS_X86_64);
        #[cfg(not(windows))]
        assert_ne!(Target::native(), Target::WINDOWS_X86_64);
    }
}
