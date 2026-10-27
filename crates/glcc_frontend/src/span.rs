//! Source spans for diagnostics.
//!
//! Every AST struct carries a `span: Span` field (queried uniformly via
//! [`HasSpan`]); semantic keys that live in `Sema` (`Ty`, `DeclId`,
//! `StructId`) stay spanless by design, with spans held by their owners
//! (`Binding`, `StructDef`). Enum nodes (`DirectBase`, `TypeSpecifier`,
//! …) are covered by the span of the parent struct that contains them.
//!
//! [`CompileError`] is the structured diagnostic payload: a message plus
//! one primary and any number of secondary labeled spans, an optional
//! rust-style [`DiagCode`], and `help:` suggestions, tagged with a
//! [`Severity`]. The parser renders errors through codespan-reporting
//! (see `Parser::emit`), accumulates them instead of aborting, and
//! exposes the whole set via `Parser::diagnostics`; a machine-readable
//! (JSON) form drives editor integrations (`Parser::diagnostics_json`).

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::ops::Range;

use string_interner::symbol::SymbolU32;

use crate::diag::{DiagCode, help_for_message};
use crate::lexer::Location;

/// A source range, start-inclusive to end-exclusive, in the lexer's
/// line/column coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub file: Option<SymbolU32>,
    pub start_ln: u64,
    pub start_col: u64,
    pub end_ln: u64,
    pub end_col: u64,
}

impl Span {
    /// Placeholder for synthesized nodes with no source location
    /// (used in tests and for compiler-generated entities).
    pub fn dummy() -> Self {
        Self {
            file: None,
            start_ln: 0,
            start_col: 0,
            end_ln: 0,
            end_col: 0,
        }
    }

    /// A zero-width span at one token position.
    pub fn point(loc: Location) -> Self {
        Self {
            file: Some(loc.file),
            start_ln: loc.ln,
            start_col: loc.col,
            end_ln: loc.ln,
            end_col: loc.col,
        }
    }

    /// The range from `start` (inclusive) to `end` (exclusive).
    pub fn new(start: Location, end: Location) -> Self {
        Self {
            file: Some(start.file),
            start_ln: start.ln,
            start_col: start.col,
            end_ln: end.ln,
            end_col: end.col,
        }
    }

    pub fn is_dummy(&self) -> bool {
        self.file.is_none()
    }

    /// Compact `start-end` rendering without filename resolution, for
    /// contexts with no interner at hand (the parser's `span_str`
    /// renders the full form).
    pub fn short(&self) -> String {
        if self.is_dummy() {
            return "unknown location".to_string();
        }
        format!("{}:{}-{}:{}", self.start_ln, self.start_col, self.end_ln, self.end_col)
    }

    /// The smallest span covering both inputs. Dummy inputs are
    /// absorbed, so merging a real span with an unset one keeps the
    /// real span.
    pub fn merge(self, other: Span) -> Span {
        if self.is_dummy() {
            return other;
        }
        if other.is_dummy() {
            return self;
        }
        let file = self.file.or(other.file);
        let (start_ln, start_col) = if (self.start_ln, self.start_col) <= (other.start_ln, other.start_col) {
            (self.start_ln, self.start_col)
        } else {
            (other.start_ln, other.start_col)
        };
        let (end_ln, end_col) = if (self.end_ln, self.end_col) >= (other.end_ln, other.end_col) {
            (self.end_ln, self.end_col)
        } else {
            (other.end_ln, other.end_col)
        };
        Span {
            file,
            start_ln,
            start_col,
            end_ln,
            end_col,
        }
    }
}

impl Default for Span {
    fn default() -> Self {
        Span::dummy()
    }
}

/// Uniform access to the `span` field carried by every AST struct.
pub trait HasSpan {
    fn span(&self) -> Span;
}

impl<T: HasSpan> HasSpan for &T {
    fn span(&self) -> Span {
        (**self).span()
    }
}

// Whether the current thread already emitted a rendered diagnostic for
// the in-flight failure. The parser sets this in `fail`/`unwrap_diag`
// before panicking; `main` consults it to decide whether the panic
// payload still needs printing (lexer failures emit nothing, so their
// message would otherwise be lost).
//
// `thread_local!` needs `std`; the `no_std` fallback is a plain process-
// global flag, which is equivalent on the single-threaded targets
// `no_std` builds serve.
#[cfg(feature = "std")]
thread_local! {
    static DIAGNOSTIC_EMITTED: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}
#[cfg(not(feature = "std"))]
static DIAGNOSTIC_EMITTED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Record that a rendered diagnostic was emitted for the failure that
/// follows.
pub fn mark_diagnostic_emitted() {
    #[cfg(feature = "std")]
    DIAGNOSTIC_EMITTED.with(|flag| flag.set(true));
    #[cfg(not(feature = "std"))]
    DIAGNOSTIC_EMITTED.store(true, core::sync::atomic::Ordering::Relaxed);
}

/// Take (and clear) the emitted flag.
pub fn take_diagnostic_emitted() -> bool {
    #[cfg(feature = "std")]
    return DIAGNOSTIC_EMITTED.with(|flag| flag.replace(false));
    // Load-then-store rather than `swap`: read-modify-write atomics
    // are missing on some `no_std` targets (Cortex-M0), while plain
    // bool loads and stores exist everywhere `AtomicBool` does.
    #[cfg(not(feature = "std"))]
    {
        let v = DIAGNOSTIC_EMITTED.load(core::sync::atomic::Ordering::Relaxed);
        DIAGNOSTIC_EMITTED.store(false, core::sync::atomic::Ordering::Relaxed);
        v
    }
}

/// The severity of a [`CompileError`]. Warnings share the payload shape
/// (message + labeled spans + notes) so both render through the same
/// codespan path and serialize through the same machine-readable form;
/// only the severity tag differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

impl Severity {
    /// Lowercase spelling used by the machine-readable (JSON) form and
    /// accepted by Monaco's marker severities.
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
        }
    }
}

/// One labeled span inside a [`CompileError`]: the range plus the text
/// rendered beside the underline (`""` for a bare underline).
#[derive(Debug, Clone)]
pub struct SpanLabel {
    pub span: Span,
    pub message: String,
}

/// A structured compile error: a message, one primary labeled span, and
/// any number of secondary labels (e.g. "previously declared here" for
/// redeclaration conflicts), plus an optional stable code and
/// rust-style `help:` suggestions.
#[derive(Debug, Clone)]
pub struct CompileError {
    pub severity: Severity,
    /// Stable code (`E0301`) shared by every site reporting the same
    /// semantic error; `None` for diagnostics without a code.
    pub code: Option<DiagCode>,
    pub message: String,
    pub primary: SpanLabel,
    pub secondary: Vec<SpanLabel>,
    pub notes: Vec<String>,
    /// Actionable suggestions, rendered after the notes as
    /// `help: <text>` and serialized separately for editors.
    pub helps: Vec<String>,
}

impl CompileError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self::at(Severity::Error, message, span)
    }

    /// A non-fatal diagnostic: rendered and serialized like an error,
    /// but parsing/checking continues.
    pub fn warning(message: impl Into<String>, span: Span) -> Self {
        Self::at(Severity::Warning, message, span)
    }

    fn at(severity: Severity, message: impl Into<String>, span: Span) -> Self {
        let message = message.into();
        let code = match severity {
            Severity::Error => DiagCode::for_message(&message),
            Severity::Warning => None,
        };
        let helps = help_for_message(&message).map(String::from).into_iter().collect();
        Self {
            severity,
            code,
            message,
            primary: SpanLabel { span, message: String::new() },
            secondary: Vec::new(),
            notes: Vec::new(),
            helps,
        }
    }

    /// Override or supply the diagnostic code: needed for formatted
    /// messages, which [`DiagCode::for_message`] cannot match.
    pub fn with_code(mut self, code: DiagCode) -> Self {
        self.code = Some(code);
        self
    }

    /// Attach a `help:` suggestion.
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.helps.push(help.into());
        self
    }

    pub fn with_secondary(mut self, span: Span, message: impl Into<String>) -> Self {
        self.secondary.push(SpanLabel { span, message: message.into() });
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Whether this diagnostic is fatal (blocks codegen); warnings do not.
    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }
}

/// Convert a [`Span`]'s line/column range to byte offsets into `source`.
/// Out-of-range lines/columns are clamped, and an inverted range
/// collapses to an empty one, so rendering can never fail: diagnostics
/// must not panic.
pub fn byte_range(source: &str, span: Span) -> Range<usize> {
    fn offset_of(source: &str, ln: u64, col: u64) -> usize {
        let mut off = 0usize;
        let mut cur_ln = 1u64;
        for line in source.split_inclusive('\n') {
            if cur_ln == ln {
                // `col` counts characters (see the lexer); convert to bytes,
                // clamped to the visible line (excluding the terminator).
                let body = line.strip_suffix("\r\n").or_else(|| line.strip_suffix('\n'));
                let body = body.unwrap_or(line);
                let bytes: usize = body.chars().take(col as usize).map(|c| c.len_utf8()).sum();
                return off + bytes.min(body.len());
            }
            off += line.len();
            cur_ln += 1;
        }
        source.len()
    }
    let start = offset_of(source, span.start_ln, span.start_col);
    let mut end = offset_of(source, span.end_ln, span.end_col);
    if end < start {
        end = start;
    }
    start..end.min(source.len())
}
