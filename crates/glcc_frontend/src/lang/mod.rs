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

    /// The canonical `-std=` spelling for this dialect.
    pub fn flag(self) -> &'static str {
        match self {
            Std::C89 => "-std=c89",
            Std::C99 => "-std=c99",
            Std::C11 => "-std=c11",
            Std::C17 => "-std=c17",
            Std::C23 => "-std=c23",
            Std::C2y => "-std=c2y",
            Std::Gnu89 => "-std=gnu89",
            Std::Gnu99 => "-std=gnu99",
            Std::Gnu11 => "-std=gnu11",
            Std::Gnu17 => "-std=gnu17",
            Std::Gnu23 => "-std=gnu23",
            Std::Gnu2y => "-std=gnu2y",
        }
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

#[cfg(test)]
mod tests;

