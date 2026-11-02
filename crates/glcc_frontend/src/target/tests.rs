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
