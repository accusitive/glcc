//! Compiler extension spellings, grouped by the compiler whose syntax
//! they come from.
//!
//! Extensions are orthogonal to the host OS: the parser accepts MSVC
//! spellings (`__declspec`, `__cdecl`) on Linux targets whenever
//! [`LangOpts::ms`] is on, and GNU spellings on Windows. Each
//! submodule owns one family's keyword table and predicates; the
//! parser gates every spelling through [`spelling_family`] and asks
//! the family modules for family-specific facts.

mod clang;
mod gnu;
mod msvc;

pub use clang::{FLOAT_SPELLINGS, is_float_spelling};
pub use gnu::{QUALIFIER_SPELLINGS, is_inline_spelling, qualifier_of};
pub use msvc::{PTR_SIZE_SPELLINGS, is_inline_spelling as is_ms_inline_spelling, KEYWORDS as MS_KEYWORDS};

use crate::lang::{LangOpts, Std};

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
    /// GNU (or C2y for reserved spellings accepted there).
    GnuOrC2y,
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
            Fam::GnuOrC2y => lang.gnu || lang.std.level() >= Std::C2y.level(),
        }
    }
}

/// Family of a keyword spelling. Everything unlisted is GNU-family
/// (`peek_ext` only sees spellings the frontend treats as keywords).
pub fn spelling_family(word: &str) -> Fam {
    if msvc::is_keyword(word) {
        Fam::Ms
    } else if clang::is_float_spelling(word) {
        Fam::Clang
    } else {
        match word {
            "_Thread_local" => Fam::C11,
            // Bare `thread_local` and `typeof` are C23 spellings; glcc
            // kept accepting them liberally, and unchanged defaults (GNU
            // on) keep doing so.
            "thread_local" | "typeof" => Fam::GnuOrC23,
            // GNU's `__auto_type` predates C23 `auto`; glcc accepts the
            // reserved spelling in C2y too.
            "__auto_type" => Fam::GnuOrC2y,
            _ => Fam::Gnu,
        }
    }
}
