//! Predefined macros, generated from the target description.
//!
//! glcc defines its own macro set rather than replaying a clang/gcc
//! dump: everything here derives from [`Target`] (sizes, signedness,
//! ABI) and the `-std` flag, so a new target or dialect needs no
//! snapshot. The set covers the GNU/clang macro families real-world
//! headers select on (`__SIZE_TYPE__`-style type macros, the
//! `__INTn_*__` families with `_C_SUFFIX_`/`_WIDTH_`/`FMT*` members,
//! float limits, atomics, OS/arch markers) plus a GNU version bump for
//! library feature detection — see `predefines`.

use glcc_frontend::lang::Std;
use glcc_frontend::target::{Arch, LongDoubleKind, Os, Target};

/// The predefined macro table for `target` under `std_flag`
/// (`-std=...`/`-ansi`), in definition order.
pub fn predefines(target: &Target, std_flag: Option<&str>) -> Vec<(String, String)> {
    let mut d = Vec::new();
    stdc(&mut d, std_flag);
    identity(&mut d);
    gnu_compat(&mut d);
    integer_model(&mut d, target);
    integer_families(&mut d, target);
    wchar_wint(&mut d, target);
    floats(&mut d, target);
    atomics(&mut d);
    arch(&mut d, target);
    os(&mut d, target);
    d
}

fn def(d: &mut Vec<(String, String)>, name: &str, value: impl Into<String>) {
    d.push((name.to_string(), value.into()));
}

/// `__STDC__` core, with `__STDC_VERSION__` from the dialect flag
/// (C89 has no version macro).
fn stdc(d: &mut Vec<(String, String)>, std_flag: Option<&str>) {
    def(d, "__STDC__", "1");
    def(d, "__STDC_HOSTED__", "1");
    let text = std_flag.map(|flag| if flag == "-ansi" { "c89" } else { flag.strip_prefix("-std=").unwrap_or(flag) });
    let version = match Std::parse(text.unwrap_or("c17")) {
        Some(Std::C89 | Std::Gnu89) => None,
        Some(Std::C99 | Std::Gnu99) => Some("199901L"),
        Some(Std::C11 | Std::Gnu11) => Some("201112L"),
        Some(Std::C17 | Std::Gnu17) | None => Some("201710L"),
        Some(Std::C23 | Std::Gnu23) => Some("202311L"),
        Some(Std::C2y | Std::Gnu2y) => Some("202400L"),
    };
    if let Some(version) = version {
        def(d, "__STDC_VERSION__", version);
    }
    def(d, "__STDC_UTF_16__", "1");
    def(d, "__STDC_UTF_32__", "1");
    def(d, "__STDC_EMBED_NOT_FOUND__", "0");
    def(d, "__STDC_EMBED_FOUND__", "1");
    def(d, "__STDC_EMBED_EMPTY__", "2");
}

/// glcc's own identity plus the feature-detection version libraries
/// key on. GNU 4.8 sits above the 4.3/4.8 `__builtin_bswap` thresholds
/// many libraries use and below the 5.4 overflow builtins glcc lowers
/// conservatively; `-D` can override.
fn identity(d: &mut Vec<(String, String)>) {
    def(d, "__glcc__", "1");
    def(d, "__VERSION__", concat!("\"glcc ", env!("CARGO_PKG_VERSION"), "\""));
    def(d, "__GNUC__", "4");
    def(d, "__GNUC_MINOR__", "8");
    def(d, "__GNUC_PATCHLEVEL__", "0");
    def(d, "__GNUC_STDC_INLINE__", "1");
    def(d, "__GXX_ABI_VERSION", "1002");
}

fn gnu_compat(d: &mut Vec<(String, String)>) {
    def(d, "__NO_INLINE__", "1");
    def(d, "__PRAGMA_REDEFINE_EXTNAME", "1");
    def(d, "__REGISTER_PREFIX__", "");
    def(d, "__USER_LABEL_PREFIX__", "");
    def(d, "__FINITE_MATH_ONLY__", "0");
    def(d, "__BIGGEST_ALIGNMENT__", "16");
}

/// Sizes, limits, and type spellings that follow the data model
/// (`long` 8 on LP64 Linux, 4 on LLP64 Windows).
fn integer_model(d: &mut Vec<(String, String)>, t: &Target) {
    let lp64 = t.long_size == 8;
    let long_t = if lp64 { "long int" } else { "long long int" };
    let ulong_t = if lp64 { "long unsigned int" } else { "long long unsigned int" };
    let long_max = if lp64 { "9223372036854775807L" } else { "2147483647L" };
    let ulong_max = if lp64 { "18446744073709551615UL" } else { "18446744073709551615ULL" };

    def(d, "__CHAR_BIT__", "8");
    def(d, "__SCHAR_MAX__", "127");
    def(d, "__SHRT_MAX__", "32767");
    def(d, "__INT_MAX__", "2147483647");
    def(d, "__LONG_MAX__", long_max);
    def(d, "__LONG_LONG_MAX__", "9223372036854775807LL");
    def(d, "__SIZE_MAX__", ulong_max);
    def(d, "__PTRDIFF_MAX__", long_max);
    def(d, "__INTMAX_MAX__", long_max);
    def(d, "__UINTMAX_MAX__", ulong_max);
    def(d, "__INTPTR_MAX__", long_max);
    def(d, "__UINTPTR_MAX__", ulong_max);

    def(d, "__SHRT_WIDTH__", "16");
    def(d, "__INT_WIDTH__", "32");
    def(d, "__LONG_WIDTH__", (t.long_size * 8).to_string());
    def(d, "__LLONG_WIDTH__", "64");
    def(d, "__SIZE_WIDTH__", "64");
    def(d, "__PTRDIFF_WIDTH__", "64");
    def(d, "__INTMAX_WIDTH__", "64");
    def(d, "__UINTMAX_WIDTH__", "64");
    def(d, "__INTPTR_WIDTH__", "64");
    def(d, "__UINTPTR_WIDTH__", "64");
    def(d, "__POINTER_WIDTH__", "64");
    def(d, "__SIG_ATOMIC_WIDTH__", "32");

    def(d, "__SIZEOF_SHORT__", "2");
    def(d, "__SIZEOF_INT__", "4");
    def(d, "__SIZEOF_LONG__", t.long_size.to_string());
    def(d, "__SIZEOF_LONG_LONG__", "8");
    def(d, "__SIZEOF_POINTER__", "8");
    def(d, "__SIZEOF_FLOAT__", "4");
    def(d, "__SIZEOF_DOUBLE__", "8");
    def(d, "__SIZEOF_LONG_DOUBLE__", t.long_double_size.to_string());
    def(d, "__SIZEOF_SIZE_T__", "8");
    def(d, "__SIZEOF_PTRDIFF_T__", "8");
    def(d, "__SIZEOF_WCHAR_T__", t.wchar_size.to_string());
    def(d, "__SIZEOF_WINT_T__", "4");
    def(d, "__SIZEOF_INT128__", "16");
    def(d, "__BITINT_MAXWIDTH__", "128");
    def(d, "__BOOL_WIDTH__", "8");

    def(d, "__ORDER_LITTLE_ENDIAN__", "1234");
    def(d, "__ORDER_BIG_ENDIAN__", "4321");
    def(d, "__ORDER_PDP_ENDIAN__", "3412");
    def(d, "__BYTE_ORDER__", "__ORDER_LITTLE_ENDIAN__");
    def(d, "__LITTLE_ENDIAN__", "1234");

    def(d, "__SIZE_TYPE__", ulong_t);
    def(d, "__PTRDIFF_TYPE__", long_t);
    def(d, "__INTMAX_TYPE__", long_t);
    def(d, "__UINTMAX_TYPE__", ulong_t);
    def(d, "__INTPTR_TYPE__", long_t);
    def(d, "__UINTPTR_TYPE__", ulong_t);
    def(d, "__SIG_ATOMIC_TYPE__", "int");
    def(d, "__CHAR16_TYPE__", "unsigned short");
    def(d, "__CHAR32_TYPE__", "unsigned int");
    def(d, "__WINT_TYPE__", "unsigned int");
    def(d, "__WINT_UNSIGNED__", "1");
    def(d, "__WCHAR_TYPE__", wchar_type(t));
}

fn wchar_type(t: &Target) -> &'static str {
    match (t.wchar_size, t.wchar_signed) {
        (2, false) => "unsigned short",
        (_, false) => "unsigned int",
        (_, true) => "int",
    }
}

/// The `__INTn_*__`/`__UINTn_*__` families (TYPE, MAX, C_SUFFIX,
/// WIDTH, FMTd/FMTi/FMTo/FMTu/FMTx/FMTX) for the plain, least, and
/// fast widths, plus the function-like `__INTn_C(c)` stdint helpers.
fn integer_families(d: &mut Vec<(String, String)>, t: &Target) {
    let lp64 = t.long_size == 8;
    let l = if lp64 { "L" } else { "LL" };
    let ul = if lp64 { "UL" } else { "ULL" };
    let fmt64 = if lp64 { "l" } else { "ll" };

    // (width bits, signed type, unsigned type, C suffix, unsigned C
    // suffix, printf length modifier)
    let widths: [(u32, &str, &str, &str, &str, &str); 4] = [
        (8, "signed char", "unsigned char", "", "", "hh"),
        (16, "short int", "short unsigned int", "", "", "h"),
        (32, "int", "unsigned int", "", "U", ""),
        (64, if lp64 { "long int" } else { "long long int" }, if lp64 { "long unsigned int" } else { "long long unsigned int" }, l, ul, fmt64),
    ];
    let max_signed: [(u32, &str); 4] = [
        (8, "127"),
        (16, "32767"),
        (32, "2147483647"),
        (64, if lp64 { "9223372036854775807L" } else { "9223372036854775807LL" }),
    ];
    let max_unsigned: [(u32, &str); 4] = [
        (8, "255"),
        (16, "65535"),
        (32, "4294967295U"),
        (64, if lp64 { "18446744073709551615UL" } else { "18446744073709551615ULL" }),
    ];

    for (bits, signed_t, unsigned_t, csuf, ucsuf, fmt) in widths {
        let n = bits.to_string();
        def(d, &format!("__INT{n}_TYPE__"), signed_t);
        def(d, &format!("__UINT{n}_TYPE__"), unsigned_t);
        def(d, &format!("__INT{n}_MAX__"), max_signed.iter().find(|(b, _)| *b == bits).unwrap().1);
        def(d, &format!("__UINT{n}_MAX__"), max_unsigned.iter().find(|(b, _)| *b == bits).unwrap().1);
        def(d, &format!("__INT{n}_C_SUFFIX__"), csuf);
        def(d, &format!("__UINT{n}_C_SUFFIX__"), ucsuf);
        def(d, &format!("__INT{n}_WIDTH__"), &n);
        def(d, &format!("__UINT{n}_WIDTH__"), &n);
        def(d, &format!("__INT{n}_FMTd__"), &format!("\"{fmt}d\""));
        def(d, &format!("__INT{n}_FMTi__"), &format!("\"{fmt}i\""));
        def(d, &format!("__UINT{n}_FMTo__"), &format!("\"{fmt}o\""));
        def(d, &format!("__UINT{n}_FMTu__"), &format!("\"{fmt}u\""));
        def(d, &format!("__UINT{n}_FMTx__"), &format!("\"{fmt}x\""));
        def(d, &format!("__UINT{n}_FMTX__"), &format!("\"{fmt}X\""));
        let paste = |suf: &str| if suf.is_empty() { "c".to_string() } else { format!("c##{suf}") };
        def(d, &format!("__INT{n}_C(c)"), paste(csuf));
        def(d, &format!("__UINT{n}_C(c)"), paste(ucsuf));
    }

    // leastN tracks the plain width; fast8 is byte-sized, fast16..64
    // are pointer-sized on these LP64/LLP64 targets.
    for (bits, signed_t, unsigned_t, _csuf, _ucsuf, fmt) in widths {
        let n = bits.to_string();
        def(d, &format!("__INT_LEAST{n}_TYPE__"), signed_t);
        def(d, &format!("__UINT_LEAST{n}_TYPE__"), unsigned_t);
        def(d, &format!("__INT_LEAST{n}_MAX__"), max_signed.iter().find(|(b, _)| *b == bits).unwrap().1);
        def(d, &format!("__UINT_LEAST{n}_MAX__"), max_unsigned.iter().find(|(b, _)| *b == bits).unwrap().1);
        def(d, &format!("__INT_LEAST{n}_WIDTH__"), &n);
        def(d, &format!("__INT_LEAST{n}_FMTd__"), &format!("\"{fmt}d\""));
        def(d, &format!("__INT_LEAST{n}_FMTi__"), &format!("\"{fmt}i\""));
        def(d, &format!("__UINT_LEAST{n}_FMTo__"), &format!("\"{fmt}o\""));
        def(d, &format!("__UINT_LEAST{n}_FMTu__"), &format!("\"{fmt}u\""));
        def(d, &format!("__UINT_LEAST{n}_FMTx__"), &format!("\"{fmt}x\""));
        def(d, &format!("__UINT_LEAST{n}_FMTX__"), &format!("\"{fmt}X\""));
    }
    let fast: [(u32, &str, &str, &str, &str); 4] = [
        (8, "signed char", "unsigned char", "", "hh"),
        (16, if lp64 { "long int" } else { "long long int" }, if lp64 { "long unsigned int" } else { "long long unsigned int" }, l, fmt64),
        (32, if lp64 { "long int" } else { "long long int" }, if lp64 { "long unsigned int" } else { "long long unsigned int" }, l, fmt64),
        (64, if lp64 { "long int" } else { "long long int" }, if lp64 { "long unsigned int" } else { "long long unsigned int" }, l, fmt64),
    ];
    for (bits, signed_t, unsigned_t, _csuf, fmt) in fast {
        let n = bits.to_string();
        def(d, &format!("__INT_FAST{n}_TYPE__"), signed_t);
        def(d, &format!("__UINT_FAST{n}_TYPE__"), unsigned_t);
        def(d, &format!("__INT_FAST{n}_MAX__"), max_signed.iter().find(|(b, _)| *b == bits).unwrap().1);
        def(d, &format!("__UINT_FAST{n}_MAX__"), max_unsigned.iter().find(|(b, _)| *b == bits).unwrap().1);
        def(d, &format!("__INT_FAST{n}_WIDTH__"), (if bits == 8 { 8 } else { 64 }).to_string());
        def(d, &format!("__INT_FAST{n}_FMTd__"), &format!("\"{fmt}d\""));
        def(d, &format!("__INT_FAST{n}_FMTi__"), &format!("\"{fmt}i\""));
        def(d, &format!("__UINT_FAST{n}_FMTo__"), &format!("\"{fmt}o\""));
        def(d, &format!("__UINT_FAST{n}_FMTu__"), &format!("\"{fmt}u\""));
        def(d, &format!("__UINT_FAST{n}_FMTx__"), &format!("\"{fmt}x\""));
        def(d, &format!("__UINT_FAST{n}_FMTX__"), &format!("\"{fmt}X\""));
    }

    def(d, "__INTMAX_C_SUFFIX__", l);
    def(d, "__UINTMAX_C_SUFFIX__", ul);
    def(d, "__INTMAX_C(c)", format!("c##{l}"));
    def(d, "__UINTMAX_C(c)", format!("c##{ul}"));
    def(d, "__INTMAX_FMTd__", &format!("\"{fmt64}d\""));
    def(d, "__INTMAX_FMTi__", &format!("\"{fmt64}i\""));
    def(d, "__UINTMAX_FMTo__", &format!("\"{fmt64}o\""));
    def(d, "__UINTMAX_FMTu__", &format!("\"{fmt64}u\""));
    def(d, "__UINTMAX_FMTx__", &format!("\"{fmt64}x\""));
    def(d, "__UINTMAX_FMTX__", &format!("\"{fmt64}X\""));
    def(d, "__INTPTR_FMTd__", &format!("\"{fmt64}d\""));
    def(d, "__INTPTR_FMTi__", &format!("\"{fmt64}i\""));
    def(d, "__UINTPTR_FMTo__", &format!("\"{fmt64}o\""));
    def(d, "__UINTPTR_FMTu__", &format!("\"{fmt64}u\""));
    def(d, "__UINTPTR_FMTx__", &format!("\"{fmt64}x\""));
    def(d, "__UINTPTR_FMTX__", &format!("\"{fmt64}X\""));
    def(d, "__SIZE_FMTo__", &format!("\"{fmt64}o\""));
    def(d, "__SIZE_FMTu__", &format!("\"{fmt64}u\""));
    def(d, "__SIZE_FMTx__", &format!("\"{fmt64}x\""));
    def(d, "__SIZE_FMTX__", &format!("\"{fmt64}X\""));
    def(d, "__PTRDIFF_FMTd__", &format!("\"{fmt64}d\""));
    def(d, "__PTRDIFF_FMTi__", &format!("\"{fmt64}i\""));
}

fn wchar_wint(d: &mut Vec<(String, String)>, t: &Target) {
    match (t.wchar_size, t.wchar_signed) {
        (2, false) => {
            def(d, "__WCHAR_MAX__", "65535");
            def(d, "__WCHAR_MIN__", "0");
            def(d, "__WCHAR_UNSIGNED__", "1");
        }
        (_, false) => {
            def(d, "__WCHAR_MAX__", "4294967295U");
            def(d, "__WCHAR_MIN__", "0U");
            def(d, "__WCHAR_UNSIGNED__", "1");
        }
        (_, true) => {
            def(d, "__WCHAR_MAX__", "2147483647");
            def(d, "__WCHAR_MIN__", "(-2147483647 - 1)");
        }
    }
    def(d, "__WCHAR_WIDTH__", (t.wchar_size * 8).to_string());
    def(d, "__WINT_MAX__", "4294967295U");
    def(d, "__WINT_MIN__", "0U");
    def(d, "__WINT_WIDTH__", "32");
    def(d, "__SIG_ATOMIC_MAX__", "2147483647");
    def(d, "__SIG_ATOMIC_MIN__", "(-2147483647 - 1)");
    if !t.char_signed {
        def(d, "__CHAR_UNSIGNED__", "1");
    }
}

/// Float limit families; `long double` follows the target's
/// representation kind.
fn floats(d: &mut Vec<(String, String)>, t: &Target) {
    def(d, "__FLT_RADIX__", "2");
    float_family(
        d,
        "FLT",
        (24, 6, 9, -125, 128, -37, 38),
        ("3.40282347e+38F", "1.17549435e-38F", "1.19209290e-7F", "1.40129846e-45F"),
    );
    float_family(
        d,
        "DBL",
        (53, 15, 17, -1021, 1024, -307, 308),
        ("1.7976931348623157e+308", "2.2250738585072014e-308", "2.2204460492503131e-16", "4.9406564584124654e-324"),
    );
    match t.long_double {
        LongDoubleKind::F80 => float_family(
            d,
            "LDBL",
            (64, 18, 21, -16381, 16384, -4931, 4932),
            ("1.18973149535723176502e+4932L", "3.36210314311209350626e-4932L", "1.08420217248550443401e-19L", "3.64519953188247460253e-4951L"),
        ),
        LongDoubleKind::F128 => float_family(
            d,
            "LDBL",
            (113, 33, 36, -16381, 16384, -4931, 4932),
            (
                "1.18973149535723176502e+4932L",
                "3.36210314311209350626e-4932L",
                "1.92592994438723585305597794258492732e-34L",
                "6.47517511943802511092443895822764655e-4966L",
            ),
        ),
        LongDoubleKind::F64 => float_family(
            d,
            "LDBL",
            (53, 15, 17, -1021, 1024, -307, 308),
            ("1.7976931348623157e+308L", "2.2250738585072014e-308L", "2.2204460492503131e-16L", "4.9406564584124654e-324L"),
        ),
    }
    def(d, "__DECIMAL_DIG__", "__LDBL_DECIMAL_DIG__");
}

type FloatExponents = (u32, u32, u32, i32, i32, i32, i32);

/// `__P_*__` for mantissa/digits/exponents (MANT_DIG, DIG,
/// DECIMAL_DIG, MIN_EXP, MAX_EXP, MIN_10_EXP, MAX_10_EXP) and the
/// limit constants (MAX, MIN, EPSILON, DENORM_MIN; NORM_MAX aliases
/// MAX; the HAS_* predicates are all true).
fn float_family(d: &mut Vec<(String, String)>, p: &str, (mant, dig, dec, min_exp, max_exp, min10, max10): FloatExponents, (max, min, eps, denorm): (&str, &str, &str, &str)) {
    let pre = format!("__{p}_");
    def(d, &format!("{pre}MANT_DIG__"), mant.to_string());
    def(d, &format!("{pre}DIG__"), dig.to_string());
    def(d, &format!("{pre}DECIMAL_DIG__"), dec.to_string());
    def(d, &format!("{pre}MIN_EXP__"), format!("({min_exp})"));
    def(d, &format!("{pre}MAX_EXP__"), max_exp.to_string());
    def(d, &format!("{pre}MIN_10_EXP__"), format!("({min10})"));
    def(d, &format!("{pre}MAX_10_EXP__"), max10.to_string());
    def(d, &format!("{pre}MAX__"), max);
    def(d, &format!("{pre}MIN__"), min);
    def(d, &format!("{pre}EPSILON__"), eps);
    def(d, &format!("{pre}DENORM_MIN__"), denorm);
    def(d, &format!("{pre}NORM_MAX__"), max);
    def(d, &format!("{pre}HAS_DENORM__"), "1");
    def(d, &format!("{pre}HAS_INFINITY__"), "1");
    def(d, &format!("{pre}HAS_QUIET_NAN__"), "1");
}

fn atomics(d: &mut Vec<(String, String)>) {
    def(d, "__ATOMIC_RELAXED", "0");
    def(d, "__ATOMIC_CONSUME", "1");
    def(d, "__ATOMIC_ACQUIRE", "2");
    def(d, "__ATOMIC_RELEASE", "3");
    def(d, "__ATOMIC_ACQ_REL", "4");
    def(d, "__ATOMIC_SEQ_CST", "5");
    for ty in ["BOOL", "CHAR", "CHAR16_T", "CHAR32_T", "WCHAR_T", "SHORT", "INT", "LONG", "LLONG", "POINTER"] {
        def(d, &format!("__GCC_ATOMIC_{ty}_LOCK_FREE"), "2");
    }
    def(d, "__GCC_ATOMIC_TEST_AND_SET_TRUEVAL", "1");
    for n in ["1", "2", "4", "8"] {
        def(d, &format!("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_{n}"), "1");
    }
    def(d, "__GCC_CONSTRUCTIVE_SIZE", "64");
    def(d, "__GCC_DESTRUCTIVE_SIZE", "64");
}

fn arch(d: &mut Vec<(String, String)>, t: &Target) {
    match t.arch {
        Arch::X86_64 => {
            def(d, "__x86_64", "1");
            def(d, "__x86_64__", "1");
            def(d, "__amd64", "1");
            def(d, "__amd64__", "1");
            def(d, "__MMX__", "1");
            def(d, "__SSE__", "1");
            def(d, "__SSE2__", "1");
            def(d, "__SSE_MATH__", "1");
            def(d, "__SSE2_MATH__", "1");
            def(d, "__FXSR__", "1");
            def(d, "__SEG_FS", "1");
            def(d, "__SEG_GS", "1");
            if t.os == Os::Windows {
                def(d, "_M_X64", "100");
                def(d, "_M_AMD64", "100");
            }
        }
        Arch::Aarch64 => {
            def(d, "__aarch64__", "1");
            def(d, "__arm64__", "1");
        }
        Arch::Riscv64 => {
            def(d, "__riscv", "1");
            def(d, "__riscv_xlen", "64");
            def(d, "__riscv_flen", "64");
            def(d, "__riscv_fdiv", "1");
        }
    }
}

fn os(d: &mut Vec<(String, String)>, t: &Target) {
    match t.os {
        Os::Linux => {
            def(d, "__linux", "1");
            def(d, "__linux__", "1");
            def(d, "linux", "1");
            def(d, "unix", "1");
            def(d, "__unix", "1");
            def(d, "__unix__", "1");
            def(d, "__gnu_linux__", "1");
            def(d, "__ELF__", "1");
            def(d, "__LP64__", "1");
            def(d, "_LP64", "1");
            def(d, "__PIC__", "2");
            def(d, "__pic__", "2");
        }
        Os::Windows => {
            def(d, "_WIN32", "1");
            def(d, "_WIN64", "1");
            def(d, "_MSC_VER", "1934");
            def(d, "_MSC_FULL_VER", "193431937");
            def(d, "_MSC_BUILD", "1");
            def(d, "_MSC_EXTENSIONS", "1");
            def(d, "_INTEGRAL_MAX_BITS", "64");
            def(d, "_MSVC_EXECUTION_CHARACTER_SET", "65001");
            def(d, "__STDC_NO_THREADS__", "1");
        }
    }
}
