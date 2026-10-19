//! fern frontend: lexer, parser, a kind checker over type expressions,
//! Hindley-Milner inference with class obligations over higher-kinded
//! constructors, and lowering to `glcc_mir`.
//!
//! Pipeline: `lexer` -> `parser` (surface AST) -> `defs` (kinds and item
//! collection) -> `check` (HM inference + class obligations) -> HIR ->
//! `lower` (monomorphizing MIR construction). `compile` runs it all.
//! fern is pure: no mutation, no loops, no pointers.

pub mod ast;
pub mod check;
pub mod defs;
pub mod hir;
pub mod infer;
pub mod lexer;
pub mod lower;
pub mod parser;
pub mod prelude;
#[cfg(feature = "c-rt")]
pub mod runtime;
pub mod solve;
pub mod span;
pub mod ty;

pub use span::{Diag, KResult, SourceMap, Span};

/// Spans whose `lo` is at or beyond this offset come from the built-in
/// prelude; later passes use this to recognize built-ins.
pub const PRELUDE_TAG: u32 = 1 << 30;

pub type MirModule = glcc_mir::ir::Module;

/// Compile fern source to MIR: prelude items first, then `src`.
pub fn compile(src: &str) -> KResult<MirModule> {
    let mut tokens = lexer::lex(prelude::SOURCE)?;
    // The prelude's `Eof` is not the program's `Eof`.
    tokens.pop();
    for token in &mut tokens {
        token.span = Span::new((token.span.lo + PRELUDE_TAG) as usize, (token.span.hi + PRELUDE_TAG) as usize);
    }
    tokens.extend(lexer::lex(src)?);
    let program = parser::parse(tokens)?;
    let mut defs = defs::collect(&program)?;
    let hir = check::check(&mut defs, &program)?;
    lower::lower(&defs, &hir)
}

/// Render a diagnostic for a driver. Prelude spans resolve against the
/// prelude text, so a caret in a user file is never off the end.
pub fn render_diag(src: &str, diag: &Diag) -> String {
    if diag.span.lo >= PRELUDE_TAG {
        let lo = (diag.span.lo - PRELUDE_TAG) as usize;
        let hi = diag.span.hi.saturating_sub(PRELUDE_TAG) as usize;
        let shifted = Diag::new(diag.msg.clone(), Span::new(lo, hi));
        return SourceMap::new(prelude::SOURCE).render(prelude::SOURCE, &shifted).replacen(" --> ", " --> prelude:", 1);
    }
    SourceMap::new(src).render(src, diag)
}
