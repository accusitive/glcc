//! Language dialect options: which C standard version the source
//! targets and which compiler-family extensions the parser accepts.
//!
//! The driver builds a [`LangOpts`] from `-std=`/`-ansi` and the
//! `-f{gnu,ms,clang}-extensions` flags and stores it on `Sema`; parse
//! sites gate versioned and family-spelled constructs through it. The
//! default is fully permissive (C17 with every extension family on), so
//! callers that only construct a `Sema` unchanged keep the historical
//! behavior; the glcc-native async extension stays gated by
//! `Sema::async_enabled` (`-fglcc-async`).

/// C standard the source is written against. Versioned-only constructs
/// (`_Static_assert` needs C11, bare `typeof`/`thread_local` need C23)
/// read the level via [`Std::level`]; `gnu` variants carry the same
/// level — the GNU *family* gate is [`LangOpts::gnu`], chosen by
/// explicit flags rather than implied by the standard's spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Std {
    C89,
    C99,
    C11,
    C17,
    /// C23 (`-std=c23`; the pre-standard spelling `c2x` parses too).
    C23,
    /// C2y draft (`-std=c2y`); a superset of C23.
    C2y,
    Gnu89,
    Gnu99,
    Gnu11,
    Gnu17,
    Gnu23,
    Gnu2y,
}

impl Std {
    /// Standard level ignoring gnu-ness: 0 = C89, 5 = C2y.
    pub fn level(self) -> u8 {
        match self {
            Std::C89 | Std::Gnu89 => 0,
            Std::C99 | Std::Gnu99 => 1,
            Std::C11 | Std::Gnu11 => 2,
            Std::C17 | Std::Gnu17 => 3,
            Std::C23 | Std::Gnu23 => 4,
            Std::C2y | Std::Gnu2y => 5,
        }
    }

    pub fn is_gnu(self) -> bool {
        matches!(self, Std::Gnu89 | Std::Gnu99 | Std::Gnu11 | Std::Gnu17 | Std::Gnu23 | Std::Gnu2y)
    }

    /// A `-std=` value without its prefix (the ISO spellings match
    /// gcc's list; the gnu-vs-c prefix decides the variant).
    pub fn parse(text: &str) -> Option<Std> {
        let (gnu, base) = match text.strip_prefix("gnu") {
            Some(base) => (true, base),
            None => (false, text),
        };
        // Plain spellings carry the leading `c`; ISO spellings do not.
        let base = base.strip_prefix('c').unwrap_or(base);
        let level = match base {
            "89" | "90" | "iso9899:1990" | "iso9899:199409" => 0,
            "99" | "9x" | "09" | "iso9899:1999" => 1,
            "11" | "1x" | "iso9899:2011" => 2,
            "17" | "18" | "iso9899:2017" | "iso9899:2018" => 3,
            "2x" | "23" | "iso9899:2024" => 4,
            "2y" => 5,
            _ => return None,
        };
        Some(match (gnu, level) {
            (false, 0) => Std::C89,
            (false, 1) => Std::C99,
            (false, 2) => Std::C11,
            (false, 3) => Std::C17,
            (false, 4) => Std::C23,
            (false, _) => Std::C2y,
            (true, 0) => Std::Gnu89,
            (true, 1) => Std::Gnu99,
            (true, 2) => Std::Gnu11,
            (true, 3) => Std::Gnu17,
            (true, 4) => Std::Gnu23,
            (true, _) => Std::Gnu2y,
        })
    }
}

/// Parser-visible language options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LangOpts {
    pub std: Std,
    /// GNU-family spellings: `__asm__`, `__attribute__`, `__int128`,
    /// `__typeof__`, `__thread`, `__extension__`, file-scope/bodies
    /// `asm`, and friends.
    pub gnu: bool,
    /// MSVC-family spellings: `__declspec`, calling conventions
    /// (`__cdecl`, `__stdcall`, ...), `__intN`, `__pragma`,
    /// `__forceinline`, pointer-size attributes (`__ptr32`/`__ptr64`).
    pub ms: bool,
    /// clang-family spellings: `_Float32`/`_Float64` and friends.
    pub clang: bool,
    /// `-pedantic-errors`: make standard constraint violations fatal.
    /// Currently used for the C89 implicit `int` and implicit function
    /// declarations that C99 removed.
    pub strict: bool,
}

impl Default for LangOpts {
    /// Fully permissive: C17 with every extension family on, matching
    /// the behavior before gating existed.
    fn default() -> Self {
        Self {
            std: Std::Gnu17,
            gnu: true,
            ms: true,
            clang: true,
            strict: false,
        }
    }
}

impl LangOpts {
    /// Whether the selected standard includes the construct: the
    /// selected level reaches `min`.
    pub fn standard_includes(self, min: Std) -> bool {
        self.std.level() >= min.level()
    }

    /// Whether C99's removal of implicit `int` and implicit function
    /// declarations is enforced. C89 still permits both, even in
    /// pedantic mode.
    pub fn rejects_implicit(self) -> bool {
        self.strict && self.standard_includes(Std::C99)
    }
}

/// Which gate a keyword spelling reaches the parser through.
/// Classification table in [`spelling_family`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fam {
    Gnu,
    Ms,
    Clang,
    /// Standard level required (no matter which extension family).
    C11,
    C23,
    /// GNU (or C23 for the standardized spelling).
    GnuOrC23,
}

impl Fam {
    pub fn allowed(self, lang: LangOpts) -> bool {
        match self {
            Fam::Gnu => lang.gnu,
            Fam::Ms => lang.ms,
            Fam::Clang => lang.clang,
            Fam::C11 => lang.standard_includes(Std::C11),
            Fam::C23 => lang.standard_includes(Std::C23),
            Fam::GnuOrC23 => lang.gnu || lang.std.level() >= Std::C23.level(),
        }
    }
}

/// Family of a keyword spelling. Everything unlisted is GNU-family
/// (`peek_is_gnu` only sees spellings the frontend treats as keywords).
pub fn spelling_family(word: &str) -> Fam {
    match word {
        "_Thread_local" => Fam::C11,
        // Bare `thread_local` and `typeof` are C23 spellings; glcc kept
        // accepting them liberally, and unchanged defaults (GNU on)
        // keep doing so.
        "thread_local" | "typeof" => Fam::GnuOrC23,
        "__int8" | "__int16" | "__int32" | "__int64" | "__declspec" | "_declspec" | "__pragma" | "__cdecl" | "__stdcall" | "__fastcall" | "__thiscall" | "__vectorcall"
        | "__clrcall" | "__forceinline" | "__ptr32" | "__ptr64" | "__unaligned" => Fam::Ms,
        "_Float32" | "_Float64" => Fam::Clang,
        _ => Fam::Gnu,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn std_spellings_parse() {
        for (text, want) in [
            ("c89", Std::C89),
            ("c90", Std::C89),
            ("iso9899:1990", Std::C89),
            ("iso9899:199409", Std::C89),
            ("c99", Std::C99),
            ("c9x", Std::C99),
            ("c11", Std::C11),
            ("c1x", Std::C11),
            ("c17", Std::C17),
            ("c18", Std::C17),
            ("c2x", Std::C23),
            ("c23", Std::C23),
            ("iso9899:2024", Std::C23),
            ("c2y", Std::C2y),
            ("gnu89", Std::Gnu89),
            ("gnu17", Std::Gnu17),
            ("gnu2x", Std::Gnu23),
            ("gnu2y", Std::Gnu2y),
        ] {
            assert_eq!(Std::parse(text), Some(want), "{text}");
        }
        assert_eq!(Std::parse("c1y"), None);
        assert_eq!(Std::parse(""), None);
        assert_eq!(Std::parse("iso9899:1991"), None);
    }

    #[test]
    fn levels_and_gnu_flags_agree() {
        assert_eq!(Std::C89.level(), 0);
        assert_eq!(Std::Gnu17.level(), Std::C17.level());
        assert_eq!(Std::C23.level(), 4);
        assert_eq!(Std::C2y.level(), 5);
        assert!(Std::Gnu11.is_gnu());
        assert!(Std::Gnu2y.is_gnu());
        assert!(!Std::C11.is_gnu());
        assert!(LangOpts::default().standard_includes(Std::C11));
        assert!(
            !LangOpts {
                std: Std::C89,
                ..LangOpts::default()
            }
            .standard_includes(Std::C11)
        );
    }
}
