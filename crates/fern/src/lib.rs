//! fern frontend: lexer, parser, a kind checker over type expressions,
//! Hindley-Milner inference with class obligations over higher-kinded
//! constructors, and lowering to `glcc_mir`.
//!
//! Pipeline: `lexer` -> `parser` (surface AST) -> `defs` (kinds and item
//! collection) -> `check` (HM inference + class obligations) -> HIR ->
//! `lower` (monomorphizing MIR construction).
//!
//! Front-ends go through [`session::Session`], which owns the source
//! files and memoizes parse results. [`compile`] is a one-shot wrapper
//! over a session. fern is pure: no mutation, no loops, no pointers in
//! the surface language; the shell layer is syntax plus prelude calls.

pub mod ast;
pub mod check;
pub mod defs;
pub mod hir;
pub mod infer;
pub mod intern;
pub mod lexer;
pub mod lower;
pub mod parser;
pub mod prelude;
#[cfg(feature = "c-rt")]
pub mod runtime;
pub mod session;
pub mod solve;
pub mod span;
pub mod ty;

pub use session::{Analysis, ParseResult, Session};
pub use span::{Diag, FileId, KResult, SourceMap, Span};

pub type MirModule = glcc_mir::ir::Module;

/// Compile fern source with the built-in prelude to MIR.
pub fn compile(src: &str) -> KResult<MirModule> {
    session::compile_str(src)
}

/// Render a diagnostic for a driver. Prelude diagnostics resolve against
/// the prelude text; everything else against `src`.
pub fn render_diag(src: &str, diag: &Diag) -> String {
    let source = if diag.span.file == FileId::PRELUDE {
        prelude::SOURCE
    } else {
        src
    };
    let rendered = SourceMap::new(source).render(source, diag);
    if diag.span.file == FileId::PRELUDE {
        rendered.replacen(" --> ", " --> prelude:", 1)
    } else {
        rendered
    }
}
