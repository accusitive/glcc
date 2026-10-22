use codespan_reporting::diagnostic::{Diagnostic, Label, Severity};
use codespan_reporting::files::{self as csfiles, Error as FilesError, Files};
#[cfg(feature = "std")]
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};
use codespan_reporting::term::{self, Config};

use hashbrown::HashMap;

use alloc::{
    borrow::ToOwned,
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::{cell::{Cell, RefCell}, fmt, mem, ops::Range};

use string_interner::symbol::SymbolU32;

use crate::{
    check::{CheckedExpr, CheckedExprKind, is_integer},
    lang::Std,
    lexer::{IntLiteral, IntSuffix, Interner, Keyword, Located, Location, Punctuator, Radix, StringLiteral, StringPrefix, Token, TokenStream},
    sema::{Binding, BindingKind, DeclId, EnumId, Enumerator, Linkage, Member, Sema, StructId},
    span::{CompileError, HasSpan, Span, byte_range},
};

/// The preprocessed TU viewed as many files sharing one buffer.
///
/// The lexer tags every token with its real file (from `# linemarker`
/// lines) while line/column numbers stay buffer-relative, so all
/// diagnostics resolve byte offsets in the single preprocessed source.
/// codespan, however, prints one filename per `FileId`: giving every
/// distinct file its own id with the real name makes the header name
/// the original file while the snippet still comes from the shared
/// buffer.
///
/// `line_number` maps back through the linemarkers so the header shows
/// the original file's line (e.g. `tcc.h:49`), not the TU-buffer line.
/// The snippet itself is still the preprocessed text at that TU offset.
///
/// Efficiency: one `Box<str>` plus one `line_starts` table and one
/// `orig_lines` table, no matter how many headers the TU pulls in.
/// `add` only pushes a name.
#[derive(Debug, Clone)]
pub(crate) struct SourceFiles {
    names: Vec<String>,
    source: Box<str>,
    line_starts: Vec<usize>,
    /// Original line number for each TU-buffer line, from linemarkers:
    /// `# 26 "tcc.h"` means the *next* line is 26, with following
    /// content lines counting up until the next marker.
    orig_lines: Vec<usize>,
}

impl SourceFiles {
    fn new(source: &str) -> Self {
        let source: Box<str> = source.into();
        let line_starts: Vec<usize> = csfiles::line_starts(&source).collect();
        let orig_lines = original_lines(&source, &line_starts);
        Self {
            names: Vec::new(),
            source,
            line_starts,
            orig_lines,
        }
    }

    fn add(&mut self, name: String) -> usize {
        if let Some(id) = self.names.iter().position(|n| *n == name) {
            return id;
        }
        let id = self.names.len();
        self.names.push(name);
        id
    }

    fn source_text(&self) -> &str {
        &self.source
    }

    fn line_start(&self, line_index: usize) -> Result<usize, FilesError> {
        use core::cmp::Ordering;
        match line_index.cmp(&self.line_starts.len()) {
            Ordering::Less => Ok(self.line_starts[line_index]),
            Ordering::Equal => Ok(self.source.len()),
            Ordering::Greater => Err(FilesError::LineTooLarge {
                given: line_index,
                max: self.line_starts.len().saturating_sub(1),
            }),
        }
    }
}

impl<'a> Files<'a> for SourceFiles {
    type FileId = usize;
    type Name = &'a str;
    type Source = &'a str;

    fn name(&'a self, id: usize) -> Result<&'a str, FilesError> {
        self.names.get(id).map(String::as_str).ok_or(FilesError::FileMissing)
    }

    fn source(&'a self, id: usize) -> Result<&'a str, FilesError> {
        self.names.get(id).ok_or(FilesError::FileMissing)?;
        Ok(&self.source)
    }

    fn line_index(&'a self, id: usize, byte_index: usize) -> Result<usize, FilesError> {
        self.names.get(id).ok_or(FilesError::FileMissing)?;
        Ok(self.line_starts.binary_search(&byte_index).unwrap_or_else(|next| next - 1))
    }

    fn line_range(&'a self, id: usize, line_index: usize) -> Result<Range<usize>, FilesError> {
        self.names.get(id).ok_or(FilesError::FileMissing)?;
        Ok(self.line_start(line_index)?..self.line_start(line_index + 1)?)
    }

    fn line_number(&'a self, id: usize, line_index: usize) -> Result<usize, FilesError> {
        self.names.get(id).ok_or(FilesError::FileMissing)?;
        Ok(self.orig_lines.get(line_index).copied().unwrap_or(line_index + 1))
    }
}

/// A `# 26 "file"` linemarker (plus the bare `# 26` form): the lexer
/// only treats `#` at column 0 followed by a number as a marker, so
/// mirror that here and ignore `#pragma` etc.
fn parse_linemarker(line: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    if bytes.first() != Some(&b'#') {
        return None;
    }
    let mut i = 1;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
        i += 1;
    }
    let start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if start == i {
        return None;
    }
    line[start..i].parse().ok()
}

fn original_lines(source: &str, line_starts: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(line_starts.len());
    let mut pending: usize = 1;
    for (idx, &start) in line_starts.iter().enumerate() {
        let end = if idx + 1 < line_starts.len() { line_starts[idx + 1] } else { source.len() };
        let line = &source[start..end];
        if let Some(n) = parse_linemarker(line) {
            // Markers occupy TU lines but aren't original source lines:
            // the *next* content line is `n`, so latch without counting.
            pending = n;
            out.push(n);
        } else {
            out.push(pending);
            pending = pending.saturating_add(1);
        }
    }
    out
}

/// Error budget for one parse. Recovery is useful up to a point; past
/// this many errors the source is almost certainly not just missing one
/// construct, and continuing only produces cascades (sqlite's shell.c
/// used to print thousands of lines for one unsupported attribute).
pub(crate) const MAX_ERRORS: usize = 10;

pub struct Parser<'a> {
    pub(crate) tokens: TokenStream,
    pub(crate) files: SourceFiles,
    pub(crate) sema: &'a mut Sema,
    /// The driver's interner, borrowed like `sema`: spellings resolve
    /// and intern through this during parsing, and the driver keeps
    /// owning it afterwards.
    pub(crate) intern: &'a mut Interner,
    pub(crate) this: usize,
    /// Per-file codespan handles: `Span::file` (an interner symbol set
    /// from Linemarkers by the lexer) maps to a `SourceFiles` id with
    /// the real filename. All ids share the one preprocessed buffer
    /// (see `SourceFiles`), so this is just a name table built once in
    /// `new`.
    pub(crate) file_ids: HashMap<SymbolU32, usize>,
    /// Start of the first token (or a synthesized position for empty
    /// input): fallback for spans at EOF with nothing consumed.
    pub(crate) bof: Location,
    /// The enclosing function definition's return type and name, while
    /// its body parses (`None` at file scope): statement expressions
    /// check `return` against the type, and `__func__` (and friends)
    /// expand to the name.
    pub(crate) function_return: Option<Ty>,
    pub(crate) function_name: Option<SymbolU32>,
    /// True while the body of an `__attribute__((naked))` definition
    /// parses: its asm statements are whole function bodies, so
    /// non-empty templates are expected there.
    pub(crate) naked_function: bool,
    /// Nesting depth inside function prototype scopes
    /// (`parse_function_suffix` pushes one): `[*]` bounds are only
    /// legal here.
    pub(crate) prototype_depth: usize,
    /// Nesting depth inside struct/union bodies: array bounds here
    /// belong to members (checked as members, not as file-scope
    /// declarations).
    pub(crate) struct_depth: usize,
    /// `goto`s from initializers parsed inside the current function
    /// body: their statement expressions can jump to body labels, so
    /// they resolve with the body (see `check_function_body`) rather
    /// than at insert time, when later labels are not yet known.
    pub(crate) pending_gotos: Vec<(SymbolU32, Span)>,
    /// Open block scopes of `__label__` declarations (innermost last):
    /// each maps a declared spelling to a per-declaration unique
    /// symbol. GNU local labels shadow outer ones of the same spelling,
    /// so uses resolve through this stack before falling back to the
    /// plain function-scoped label.
    pub(crate) label_scopes: Vec<HashMap<SymbolU32, SymbolU32>>,
    /// Counter for the unique spellings local labels intern under
    /// (spellings cannot collide: identifiers have no dots).
    pub(crate) label_gen: u64,
    /// Structured diagnostics collected during lexing, parsing, and
    /// checking, in source order. Recovery records **one** error per
    /// construct: `failed` latches until the enclosing loop
    /// resynchronizes, so a single bad declaration doesn't cascade.
    pub(crate) diagnostics: RefCell<Vec<CompileError>>,
    /// Set once an error was recorded for the construct currently being
    /// parsed; suppresses further errors until the next resync point.
    pub(crate) failed: Cell<bool>,
    /// Errors reported through [`Parser::report`]. Lexer diagnostics
    /// were already recovered from and do not count against the budget.
    pub(crate) error_count: Cell<usize>,
    /// Set once [`MAX_ERRORS`] errors were reported: parsing stops at the
    /// next construct boundary so cascades cannot flood the output.
    pub(crate) cancelled: Cell<bool>,
    /// Render diagnostics to stderr as they are recorded. Machine
    /// consumers (the driver's `--error-format=json`, the explorer's
    /// JSON) turn this off and render [`Parser::diagnostics`] themselves.
    pub emit_human: bool,
}

/// A poison value produced when a parse/check step fails. Recovery uses
/// these to keep the surrounding construct structurally valid while the
/// enclosing loop resynchronizes; compilation stops before codegen once
/// any error diagnostic exists, so poison values never reach the
/// backend.
pub trait Recover: Sized {
    fn recover() -> Self;
}

impl Recover for () {
    fn recover() -> Self {}
}

impl Recover for String {
    fn recover() -> Self {
        String::new()
    }
}

impl Recover for u64 {
    fn recover() -> Self {
        0
    }
}

impl<T> Recover for Vec<T> {
    fn recover() -> Self {
        Vec::new()
    }
}

impl<T> Recover for Option<T> {
    fn recover() -> Self {
        None
    }
}

impl Recover for DeclId {
    fn recover() -> Self {
        DeclId(u32::MAX)
    }
}

impl Recover for StructId {
    fn recover() -> Self {
        StructId(u32::MAX)
    }
}

impl Recover for EnumId {
    fn recover() -> Self {
        EnumId(u32::MAX)
    }
}

impl Recover for Ty {
    fn recover() -> Self {
        Ty::Void
    }
}

impl Recover for TypeSpecifier {
    fn recover() -> Self {
        TypeSpecifier::Void
    }
}

impl Recover for Expr {
    fn recover() -> Self {
        Expr::SizeofTy {
            ty: Ty::Void,
            span: Span::dummy(),
        }
    }
}

impl Recover for Stmt {
    fn recover() -> Self {
        Stmt::Empty { span: Span::dummy() }
    }
}

impl Recover for Initializer {
    fn recover() -> Self {
        Initializer::List {
            items: Vec::new(),
            span: Span::dummy(),
        }
    }
}

impl Recover for CheckedExpr {
    fn recover() -> Self {
        CheckedExpr {
            ty: Ty::Void,
            span: Span::dummy(),
            kind: CheckedExprKind::Str(Vec::new()),
        }
    }
}

impl Recover for Compound {
    fn recover() -> Self {
        Compound {
            span: Span::dummy(),
            items: Vec::new(),
        }
    }
}

impl Recover for TranslationUnit {
    fn recover() -> Self {
        TranslationUnit {
            external_declarations: Vec::new(),
            span: Span::dummy(),
        }
    }
}

impl Recover for DeclarationSpecifiers {
    fn recover() -> Self {
        DeclarationSpecifiers::default()
    }
}

impl Recover for GnuAttributes {
    fn recover() -> Self {
        GnuAttributes::default()
    }
}

impl Recover for ArrayDeclarator {
    fn recover() -> Self {
        ArrayDeclarator::default()
    }
}

impl Recover for FunctionDeclarator {
    fn recover() -> Self {
        FunctionDeclarator::default()
    }
}

impl Recover for Param {
    fn recover() -> Self {
        Param {
            name: None,
            ty: Ty::Void,
            body_ty: None,
            vla_bound: None,
            span: Span::dummy(),
            star: false,
            proto: None,
        }
    }
}

impl Recover for (Vec<Enumerator>, Ty) {
    fn recover() -> Self {
        (Vec::new(), Ty::Void)
    }
}

macro_rules! pat_storage_class_specifier {
    () => {
        Keyword::Typedef | Keyword::Extern | Keyword::Static | Keyword::Auto | Keyword::Register
    };
}
macro_rules! pat_struct_or_union {
    () => {
        Keyword::Struct | Keyword::Union
    };
}
macro_rules! pat_type_qualifier {
    () => {
        Keyword::Const | Keyword::Restrict | Keyword::Volatile
    };
}
macro_rules! pat_basic_type_specifier {
    () => {
        Keyword::Void
            | Keyword::Char
            | Keyword::Short
            | Keyword::Int
            | Keyword::Long
            | Keyword::Float
            | Keyword::Double
            | Keyword::Signed
            | Keyword::Unsigned
            | Keyword::_Bool
            | Keyword::_Complex
    };
}

impl<'a> Parser<'a> {
    pub fn new(source: &str, tokens: TokenStream, sema: &'a mut Sema, interner: &'a mut Interner) -> Self {
        let diagnostics: Vec<CompileError> = tokens.diagnostics().to_vec();
        let bof = tokens.peek().map(|t| t.loc).unwrap_or_else(|| Location {
            ln: 1,
            col: 0,
            file: interner.get_or_intern("<__glcc_source>"),
        });

        // One `SourceFiles` id per distinct linemarker file, all sharing
        // the preprocessed buffer: codespan prints the real filename in
        // the header while offsets still resolve in the one TU.
        // `this` (the bof file) is registered first so the fallback for
        // unseen spans is stable.
        let mut files = SourceFiles::new(source);
        let mut file_ids: HashMap<SymbolU32, usize> = HashMap::new();
        let mut register = |sym: SymbolU32| {
            if file_ids.contains_key(&sym) {
                return;
            }
            let name = interner.resolve(sym).unwrap_or("<__glcc_source>").to_string();
            let id = files.add(name);
            file_ids.insert(sym, id);
        };
        register(bof.file);
        for tok in tokens.iter() {
            register(tok.loc.file);
            register(tok.end.file);
        }
        let this = file_ids[&bof.file];
        // Share the linemarker mapping with the backend for DWARF: spans
        // stay buffer-relative, so `-g` maps through `orig_lines`.
        sema.orig_lines = files.orig_lines.clone();

        Self {
            tokens,
            files,
            sema,
            intern: interner,
            this,
            file_ids,
            bof,
            function_return: None,
            function_name: None,
            naked_function: false,
            prototype_depth: 0,
            struct_depth: 0,
            pending_gotos: Vec::new(),
            label_scopes: Vec::new(),
            label_gen: 0,
            diagnostics: RefCell::new(diagnostics),
            failed: Cell::new(false),
            error_count: Cell::new(0),
            cancelled: Cell::new(false),
            emit_human: true,
        }
    }
    /// Start of the upcoming token, if any.
    fn peek_loc(&self) -> Option<Location> {
        self.tokens.peek().map(|t| t.loc)
    }
    /// End of the most recently consumed token, if any.
    fn prev_end(&self) -> Option<Location> {
        self.tokens.prev().map(|t| t.end)
    }
    /// Position where the next node starts: the upcoming token's start,
    /// else the end of what was consumed, else beginning of file.
    fn mark(&self) -> Location {
        self.peek_loc().or_else(|| self.prev_end()).unwrap_or(self.bof)
    }
    /// Span of the upcoming token, or a point at the end of consumed
    /// input (or beginning of file when empty).
    fn peek_span(&self) -> Span {
        if let Some(t) = self.tokens.peek() {
            return t.span();
        }
        match self.prev_end() {
            Some(end) => Span::point(end),
            None => Span::point(self.bof),
        }
    }
    /// Span from a `mark()` to the end of the most recently consumed
    /// token. Empty when nothing was consumed since the mark (in which
    /// case it is a point at the mark, never an inverted range).
    fn span_since(&self, start: Location) -> Span {
        match self.prev_end() {
            Some(end) if end.file == start.file && (end.ln, end.col) >= (start.ln, start.col) => Span::new(start, end),
            Some(end) if end.file != start.file => Span::new(start, end),
            _ => Span::point(start),
        }
    }
    /// The codespan file id for a span, falling back to `this` (the
    /// bof file) for dummy spans and for synthesized files that were
    /// unseen when `SourceFiles` was built.
    fn file_id(&self, span: Span) -> usize {
        span.file.and_then(|f| self.file_ids.get(&f).copied()).unwrap_or(self.this)
    }
    /// Build the codespan [`Diagnostic`] for a [`CompileError`].
    /// Spans resolve to byte ranges in the single shared preprocessed
    /// buffer; each label uses its own span's file id so the header
    /// names the real file. Dummy spans are skipped, and spans from
    /// files unseen at `new` time (synthesized) fall back to `this`,
    /// so rendering never fails.
    pub(crate) fn diagnostic(&self, err: &CompileError) -> Diagnostic<usize> {
        let source = self.files.source_text();
        let mut labels = Vec::new();
        if !err.primary.span.is_dummy() {
            labels.push(Label::primary(self.file_id(err.primary.span), byte_range(source, err.primary.span)).with_message(err.primary.message.clone()));
        }
        for secondary in &err.secondary {
            if secondary.span.is_dummy() {
                continue;
            }
            labels.push(Label::secondary(self.file_id(secondary.span), byte_range(source, secondary.span)).with_message(secondary.message.clone()));
        }
        let diag = Diagnostic::error().with_message(err.message.clone()).with_labels(labels).with_notes(err.notes.clone());
        match err.severity {
            crate::span::Severity::Error => diag,
            crate::span::Severity::Warning => Diagnostic {
                severity: Severity::Warning,
                ..diag
            },
        }
    }
    /// Emit to standard error, with colors when attached to a terminal.
    /// `std`-only: `no_std` builds skip rendering (callers still receive
    /// the diagnostic through [`Parser::diagnostics`]).
    #[cfg(feature = "std")]
    pub(crate) fn emit(&self, err: &CompileError) {
        let diag = self.diagnostic(err);
        let mut writer = StandardStream::stderr(ColorChoice::Auto);
        // Rendering is infallible for a valid diagnostic; ignore I/O
        // errors so a broken pipe cannot mask the real one.
        let _ = term::emit_to_write_style(&mut writer, &Config::default(), &self.files, &diag);
    }
    /// `no_std` fallback: no stderr to write to, so render into a
    /// scratch string and drop it. This keeps the diagnostic builder
    /// compile-tested without `std`; callers still receive the
    /// diagnostic through [`Parser::diagnostics`].
    #[cfg(not(feature = "std"))]
    pub(crate) fn emit(&self, err: &CompileError) {
        let _ = term::emit_into_string(&Config::default(), &self.files, &self.diagnostic(err));
    }
    /// Render into a plain string: the test-friendly entry point for
    /// asserting on diagnostic output.
    #[cfg(test)]
    pub(crate) fn emit_to_string(&self, err: &CompileError) -> String {
        term::emit_into_string(&Config::default(), &self.files, &self.diagnostic(err)).unwrap_or_default()
    }
    /// Record the first error in the construct currently being parsed.
    /// Later errors are suppressed until the enclosing loop calls
    /// [`Parser::resync`], which keeps one mistake from cascading into a
    /// wall of follow-on diagnostics. Once [`MAX_ERRORS`] errors have
    /// been reported, parsing is cancelled outright: the final diagnostic
    /// says so, and the enclosing loops stop at their next boundary.
    pub(crate) fn report(&self, err: CompileError) {
        if self.failed.get() || self.cancelled.get() {
            return;
        }
        self.failed.set(true);
        let span = err.primary.span;
        if self.emit_human {
            self.emit(&err);
            // A rendered error was printed: a later panic (e.g. MIR
            // verification) should not also print an internal-error
            // banner on top of it.
            crate::span::mark_diagnostic_emitted();
        }
        self.diagnostics.borrow_mut().push(err);
        let errors = self.error_count.get() + 1;
        self.error_count.set(errors);
        if errors >= MAX_ERRORS {
            self.cancel(span);
        }
    }
    /// Stop parsing after [`MAX_ERRORS`] errors. The token cursor is left
    /// untouched; loops check [`Parser::cancelled`] at their next
    /// boundary, so the construct that hit the limit can still unwind
    /// without further diagnostics.
    fn cancel(&self, span: Span) {
        self.cancelled.set(true);
        let stop = CompileError::new("too many errors emitted, stopping now", span);
        if self.emit_human {
            self.emit(&stop);
        }
        self.diagnostics.borrow_mut().push(stop);
    }
    /// Record a warning: rendered and serialized like an error, but it
    /// does not latch or stop parsing.
    pub(crate) fn report_warning(&self, err: CompileError) {
        if self.emit_human {
            self.emit(&err);
        }
        self.diagnostics.borrow_mut().push(err);
    }
    /// Unwrap a checker result: on error, record it and return a poison
    /// value so the surrounding construct stays structurally valid.
    fn unwrap_diag<T: Recover>(&self, result: Result<T, CompileError>) -> T {
        match result {
            Ok(value) => value,
            Err(err) => {
                self.report(err);
                T::recover()
            }
        }
    }
    /// Unwrap a checker result, yielding `None` after recording an error:
    /// used where a poison id must never be constructed (it would index
    /// `Sema`'s arenas).
    fn unwrap_diag_opt<T>(&self, result: Result<T, CompileError>) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(err) => {
                self.report(err);
                None
            }
        }
    }
    /// Semantic checks for parsed expressions. When the construct already
    /// failed to parse, its AST holds poison nodes; checking them would
    /// produce follow-on diagnostics (or inspect meaningless types) for a
    /// mistake the user already saw reported. The latch is cleared at the
    /// enclosing construct's resynchronization point.
    fn checked_expr(&self, expr: &Expr) -> CheckedExpr {
        if self.failed() {
            return CheckedExpr::recover();
        }
        self.unwrap_diag(self.sema.check_expr(expr, &self.intern))
    }
    fn checked_condition(&self, expr: &Expr) -> CheckedExpr {
        if self.failed() {
            return CheckedExpr::recover();
        }
        self.unwrap_diag(self.sema.check_condition(expr, &self.intern))
    }
    fn checked_integer_expr(&self, expr: &Expr, what: &str) -> CheckedExpr {
        if self.failed() {
            return CheckedExpr::recover();
        }
        self.unwrap_diag(self.sema.check_integer_expr(expr, &self.intern, what))
    }
    fn checked_asm_output(&self, expr: &Expr) -> CheckedExpr {
        if self.failed() {
            return CheckedExpr::recover();
        }
        self.unwrap_diag(self.sema.check_asm_output(expr, &self.intern))
    }
    fn checked_return(&self, ret: &Ty, expr: &Expr) -> CheckedExpr {
        if self.failed() {
            return CheckedExpr::recover();
        }
        self.unwrap_diag(self.sema.check_return(ret, expr, &self.intern))
    }
    fn checked_initializer(&self, target: &Ty, init: &Initializer) -> Initializer {
        if self.failed() {
            return Initializer::recover();
        }
        self.unwrap_diag(self.sema.check_initializer(target, init, &self.intern))
    }
    /// Record a located error and return a poison value for the
    /// enclosing construct.
    fn fail<T: Recover>(&self, span: Span, msg: impl fmt::Display) -> T {
        self.report(CompileError::new(msg.to_string(), span));
        T::recover()
    }
    /// Emit a non-fatal diagnostic (a warning): rendered like an error
    /// but parsing continues. GCC-equivalent warnings with no error
    /// counterpart use this (an overlong string initializer); constraint
    /// violations still use `fail`.
    fn warn(&self, span: Span, msg: impl fmt::Display) {
        self.report_warning(CompileError::warning(msg.to_string(), span));
    }
    /// Render warnings collected during checking (see `Sema::warnings`)
    /// and clear them. Called after each declaration and at end of
    /// translation unit, so every checked initializer is covered no
    /// matter which entry point parsed it.
    fn emit_warnings(&mut self) {
        let warnings: Vec<CompileError> = self.sema.warnings.borrow_mut().drain(..).collect();
        for mut warning in warnings {
            warning.severity = crate::span::Severity::Warning;
            self.report_warning(warning);
        }
    }
    /// All diagnostics collected so far (lexing, parsing, and checking).
    pub fn diagnostics(&self) -> Vec<CompileError> {
        self.diagnostics.borrow().clone()
    }
    /// Take the collected diagnostics, leaving the sink empty.
    pub fn take_diagnostics(&self) -> Vec<CompileError> {
        mem::take(&mut *self.diagnostics.borrow_mut())
    }
    /// Machine-readable (JSON) rendering of the collected diagnostics for
    /// editor integration (Monaco and friends): 1-based lines and columns
    /// plus byte offsets into the preprocessed translation-unit buffer.
    pub fn diagnostics_json(&self) -> String {
        let mut out = String::from("{\"diagnostics\":[");
        for (i, err) in self.diagnostics.borrow().iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str("{\"severity\":\"");
            out.push_str(err.severity.as_str());
            out.push_str("\",\"message\":\"");
            push_escaped(&mut out, &err.message);
            out.push('"');
            out.push_str(",\"primary\":");
            self.push_span_json(&mut out, &err.primary);
            out.push_str(",\"labels\":[");
            for (j, label) in err.secondary.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                self.push_span_json(&mut out, label);
            }
            out.push_str("],\"notes\":[");
            for (j, note) in err.notes.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                out.push('"');
                push_escaped(&mut out, note);
                out.push('"');
            }
            out.push_str("]}");
        }
        out.push_str("]}");
        out
    }
    fn push_span_json(&self, out: &mut String, label: &crate::span::SpanLabel) {
        out.push_str("{\"message\":\"");
        push_escaped(out, &label.message);
        out.push('"');
        let span = label.span;
        if span.is_dummy() {
            out.push_str(",\"file\":null,\"line\":null,\"column\":null,\"endLine\":null,\"endColumn\":null,\"start\":null,\"end\":null}");
            return;
        }
        let file_id = self.file_id(span);
        out.push_str(",\"file\":");
        match self.files.name(file_id) {
            Ok(name) => {
                out.push('"');
                push_escaped(out, name);
                out.push('"');
            }
            Err(_) => out.push_str("null"),
        }
        let line = self
            .files
            .line_number(file_id, span.start_ln.saturating_sub(1) as usize)
            .unwrap_or(span.start_ln as usize);
        let end_line = self
            .files
            .line_number(file_id, span.end_ln.saturating_sub(1) as usize)
            .unwrap_or(span.end_ln as usize);
        let byte = byte_range(self.files.source_text(), span);
        out.push_str(&format!(
            ",\"line\":{line},\"column\":{},\"endLine\":{end_line},\"endColumn\":{},\"start\":{},\"end\":{}",
            span.start_col + 1,
            span.end_col + 1,
            byte.start,
            byte.end
        ));
        out.push('}');
    }
    /// Whether any collected diagnostic is an error (warnings don't
    /// count): codegen must not run when this is true.
    pub fn has_errors(&self) -> bool {
        self.diagnostics.borrow().iter().any(CompileError::is_error)
    }
    /// Whether the current construct already produced an error.
    pub(crate) fn failed(&self) -> bool {
        self.failed.get()
    }
    /// Clear the per-construct error latch after resynchronizing.
    pub(crate) fn clear_failure(&self) {
        self.failed.set(false);
    }
    /// Whether parsing was cancelled after hitting [`MAX_ERRORS`].
    pub(crate) fn cancelled(&self) -> bool {
        self.cancelled.get()
    }
    /// Skip to the next declaration boundary after an error: consume
    /// through the next `;` at brace depth zero. A closing `}` at depth
    /// zero is consumed too, so file-scope garbage can never spin the
    /// translation-unit loop.
    pub(crate) fn resync_declaration(&mut self) {
        // A construct that already consumed its `;` (or closed its `}`)
        // ends at a declaration boundary: resyncing would skip the next,
        // unrelated declaration.
        if matches!(
            self.tokens.prev().map(|t| &t.val),
            Some(Token::Punctuator(Punctuator::Semi | Punctuator::RBrace))
        ) {
            return;
        }
        let mut depth = 0i32;
        while let Some(tok) = self.tokens.advance() {
            match tok.val {
                Token::Punctuator(Punctuator::LParen | Punctuator::LBracket) => depth += 1,
                Token::Punctuator(Punctuator::RParen | Punctuator::RBracket) => depth = (depth - 1).max(0),
                Token::Punctuator(Punctuator::Semi) if depth == 0 => break,
                Token::Punctuator(Punctuator::RBrace) if depth == 0 => break,
                _ => {}
            }
        }
    }
    /// Skip to the next statement boundary: consume through the next
    /// `;` at bracket depth zero, or stop *before* a closing `}` so the
    /// enclosing compound-statement loop can see it.
    pub(crate) fn resync_statement(&mut self) {
        // A statement that already consumed its `;` (or a block that
        // closed) is at a boundary: resyncing would skip the next one.
        if matches!(
            self.tokens.prev().map(|t| &t.val),
            Some(Token::Punctuator(Punctuator::Semi | Punctuator::LBrace | Punctuator::RBrace))
        ) {
            return;
        }
        let mut depth = 0i32;
        loop {
            let Some(tok) = self.tokens.peek() else { return };
            match tok.val {
                Token::Punctuator(Punctuator::RParen | Punctuator::RBracket) => {
                    self.tokens.advance();
                    depth = (depth - 1).max(0);
                }
                Token::Punctuator(Punctuator::RBrace) if depth == 0 => return,
                Token::Punctuator(Punctuator::LBrace | Punctuator::LParen | Punctuator::LBracket) => {
                    self.tokens.advance();
                    depth += 1;
                }
                Token::Punctuator(Punctuator::Semi) if depth == 0 => {
                    self.tokens.advance();
                    return;
                }
                _ => {
                    self.tokens.advance();
                }
            }
        }
    }
    pub fn parse_translation_unit(&mut self) -> TranslationUnit {
        let start = self.mark();
        let mut external_declarations = vec![];
        while !self.cancelled() && self.tokens.peek().is_some() {
            if self.failed() {
                self.resync_declaration();
                self.clear_failure();
                continue;
            }
            let before = self.tokens.position();
            // C99 `_Pragma ( string-literal )`: the native preprocessor
            // lowers it to a `#pragma` line, so only raw-source callers
            // reach the operator here. Skip it like `#pragma`.
            if self.peek_pragma_operator() {
                self.skip_pragma_operator();
            } else if self.peek_msvc_operator("__pragma") {
                self.skip_msvc_operator();
            // Top-level assembler (`__asm__(".symver ...");` in system
            // headers): collected into the per-TU assembler text. Only
            // basic asm is valid at file scope (GCC), so no operands.
            } else if self.peek_is_asm_keyword() {
                let asm_start = self.mark();
                self.tokens.advance();
                while self.tokens.peek_val().is_some_and(|t| self.asm_qualifier(t).is_some()) {
                    self.tokens.advance();
                }
                if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LParen))) {
                    self.tokens.advance();
                    let template = self.parse_asm_string("template");
                    if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Colon))) {
                        self.fail::<()>(self.peek_span(), "file-scope asm cannot have operands");
                    } else {
                        self.expect_punct(Punctuator::RParen);
                        let span = self.span_since(asm_start);
                        self.sema.toplevel_asm.push((template, span));
                    }
                }
                self.expect_punct(Punctuator::Semi);
            } else {
                external_declarations.extend(self.parse_external_declaration());
            }
            if self.cancelled() {
                break;
            }
            if self.failed() {
                self.resync_declaration();
                self.clear_failure();
                if self.tokens.position() == before {
                    // The failing construct ended at a boundary (or
                    // resync was a no-op): still guarantee progress.
                    self.tokens.advance();
                }
            } else if self.tokens.position() == before {
                // Recovery guarantee: never spin on a construct that
                // consumed nothing.
                self.tokens.advance();
            }
        }
        // Tentative definitions become real definitions at end of TU
        // (C17 §6.9.2p2): `int x;` with no initializer elsewhere emits
        // a zero-initialized definition.
        for decl in self.sema.decls.iter_mut() {
            if decl.tentative {
                decl.tentative = false;
                decl.defined = true;
            }
        }
        self.emit_warnings();
        let span = self.span_since(start);
        TranslationUnit { external_declarations, span }
    }
    // A declaration "with several declarators" flattens: each declarator
    // yields its own `ExternalDeclaration`, so `int a, b;` is
    // indistinguishable from `int a; int b;`. Specifier-only tag
    // declarations (`struct S;`) yield nothing. A function declarator
    // followed by a body dispatches to `parse_function_definition` and
    // yields one `Func`.
    pub fn parse_external_declaration(&mut self) -> Vec<ExternalDeclaration> {
        // Stray `;` at file scope (including the one in `int f() {};`):
        // consume and yield nothing.
        if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Semi))) {
            self.tokens.advance();
            return vec![];
        }
        // `_Static_assert` declares nothing (`sys/stat.h` uses it in
        // inline functions at block scope and at file scope).
        if self.peek_is_static_assert() {
            let _ = self.parse_static_assert();
            return vec![];
        }
        // Prefix annotations (`__attribute__((deprecated))` on its own
        // line before a declaration, as in <X11/Xlib.h>). Captured, not
        // skipped: a leading `__attribute__((weak))` belongs to the
        // declaration that follows.
        let prefix = self.parse_gnu_annotations();
        let decl_start = self.mark();
        let specifiers = if self.can_start_declaration() {
            let mut specifiers = self.parse_declaration_specifiers();
            specifiers.attrs.merge(prefix);
            specifiers
        } else if let Some(Token::Identifier(sym)) = self.tokens.peek_val() {
            let sym = *sym;
            if self.is_typedef_name(sym) || self.is_gnu_type_keyword(sym) {
                let span = self.peek_span();
                return self.fail(span, format!("expected external declaration, got {:?}", self.tokens.peek().map(|x| &x.val)));
            }
            // C89 implicit `int` (`main () { ... }`, `f (x, y) { ... }`):
            // a bare identifier that is neither a typedef name nor a
            // GNU type keyword opens an `int` declaration. The rest of
            // the pipeline (old-style declarators defaulting missing
            // parameter types to `int`, tentative definitions) already
            // handles the shape; only the missing specifier is filled
            // in here.
            let span = self.peek_span();
            self.warn(span, "missing type specifier; C89 implicit `int` assumed");
            DeclarationSpecifiers {
                type_specifier: vec![TypeSpecifier::Int],
                span,
                ..Default::default()
            }
        } else {
            let span = self.peek_span();
            return self.fail(span, format!("expected external declaration, got {:?}", self.tokens.peek().map(|x| &x.val)));
        };
        if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Semi))) {
            self.tokens.advance();
            // No declarators: only a tag may have been declared —
            // `struct S;` or `struct S { ... };`. The tag itself was
            // already handled while parsing the specifiers; any other
            // specifier-only declaration declares nothing (C17 §6.7).
            // An enum specifier always declares its enumerators (even
            // anonymous: `enum { A };`), so it is allowed too.
            let span = self.span_since(decl_start);
            match specifiers.type_specifier.as_slice() {
                [TypeSpecifier::Struct(id)] => {
                    if self.sema.structs.get(id.0 as usize).map_or(true, |s| s.tag.is_none()) {
                        return self.fail(span, "declaration does not declare anything");
                    }
                }
                [TypeSpecifier::Enum(_)] => {}
                _ => {
                    return self.fail(span, "declaration does not declare anything");
                }
            }
            return vec![];
        }
        self.check_specifiers_file_scope(&specifiers);
        // First declarator decides: a `{` after it — possibly past an
        // old-style declaration-list — makes this a function definition
        // (C17 §6.9.1), which takes exactly one declarator and no `;`.
        let first = self.parse_declarator();
        let suffix = self.parse_gnu_annotations();
        let ty = self.spec_to_ty(&specifiers, &first);
        if matches!(ty, Ty::Func { .. }) && self.is_function_definition_ahead(&first) {
            let id = self.parse_function_definition(&specifiers, first, ty, &suffix);
            if self.failed() {
                return vec![];
            }
            return vec![ExternalDeclaration::Func(id)];
        }
        // Otherwise an init-declarator-list (`int a = 1, *b, f(void);`):
        // the first declarator is already parsed (it was needed for the
        // definition check above); the rest is shared bookkeeping.
        let ids = self.parse_init_declarator_list(&specifiers, first, suffix);
        self.emit_warnings();
        ids.into_iter()
            .filter(|id| id.0 != u32::MAX && (id.0 as usize) < self.sema.decls.len())
            .map(|id| match self.sema.decls[id.0 as usize].kind {
                BindingKind::Func => ExternalDeclaration::Func(id),
                _ => ExternalDeclaration::Var(id),
            })
            .collect()
    }
    // Function definition (C17 §6.9.1): `declaration-specifiers
    // declarator declaration-list_opt compound-statement`. The specifiers,
    // declarator, and lowered type arrive already parsed (see
    // `parse_external_declaration`); this validates the combination,
    // parses an old-style declaration-list in a prototype scope when one
    // is present, records the definition, declares the parameters in a
    // fresh function scope, and parses the body — returning the defined
    // function's `DeclId` with the body stored in `Sema::bodies`.
    // Like `parse_declaration`, it consumes exactly one declared entity.
    pub(crate) fn parse_function_definition(&mut self, specifiers: &DeclarationSpecifiers, mut declarator: Declarator, mut ty: Ty, attrs: &GnuAttributes) -> DeclId {
        let def_span = specifiers.span().merge(declarator.span());
        let Some(symbol) = declarator.name() else {
            return self.fail(def_span, "declaration does not declare anything");
        };
        self.check_function_specifiers(specifiers, &ty);
        if specifiers.is_typedef() {
            return self.fail(def_span, "function definition declared `typedef`");
        }
        if specifiers.thread_local {
            return self.fail(def_span, "thread-local storage on function");
        }
        // Old-style `int f(a) int a; { ... }`: rewrite to a modern
        // prototype. The identifier-list names are in the declarator;
        // the declaration-list supplies their types (missing entries
        // default to `int`). The rewritten declarator becomes
        // prototyped, so the rest of the pipeline (type merging,
        // body-scope declaration, calls, backend signatures) sees a
        // normal function.
        if self.declarator_is_old_style(&declarator) {
            let old_ids: Vec<(SymbolU32, Span)> = Self::defined_old_style(&declarator).map(|f| f.idents.clone()).unwrap_or_default();
            for i in 0..old_ids.len() {
                for j in (i + 1)..old_ids.len() {
                    if old_ids[i].0 == old_ids[j].0 {
                        let text = self.intern.resolve(old_ids[i].0).unwrap_or("?");
                        return self.fail_secondary(old_ids[j].1, format!("redefinition of parameter '{text}'"), old_ids[i].1, "previous definition here");
                    }
                }
            }
            self.sema.push_scope();
            let decl_count = self.sema.decls.len();
            let mut kr_types: HashMap<SymbolU32, (Ty, Span, DeclId)> = HashMap::new();
            while self.can_start_declaration() {
                if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LBrace))) {
                    break;
                }
                self.parse_kr_param_declaration(&old_ids, &mut kr_types);
            }
            let mut kr_params = Vec::with_capacity(old_ids.len());
            for (sym, ident_span) in &old_ids {
                match kr_types.remove(sym) {
                    Some((param_ty, decl_span, proto)) => {
                        // K&R ABI: the signature carries the promoted
                        // type (an unprototyped caller passes promoted
                        // values), while the body's parameter keeps the
                        // declared type — `f(c) unsigned char c` reads
                        // the low byte, so `f(-1)` sees 255 like gcc.
                        kr_params.push(Param {
                            name: Some(*sym),
                            ty: Self::kr_promote_ty(param_ty.clone()),
                            body_ty: Some(param_ty),
                            vla_bound: None,
                            span: decl_span,
                            star: false,
                            // The declaration-list scope is truncated
                            // below, so VLA bounds naming this
                            // temporary remap to the body parameter
                            // like prototyped ones do.
                            proto: Some(proto),
                        });
                    }
                    None => {
                        kr_params.push(Param {
                            name: Some(*sym),
                            ty: Ty::Int,
                            body_ty: None,
                            vla_bound: None,
                            span: *ident_span,
                            star: false,
                            proto: None,
                        });
                    }
                }
            }
            self.sema.pop_scope();
            self.sema.decls.truncate(decl_count);
            if let Some(f) = Self::defined_old_style_mut(&mut declarator) {
                f.params = kr_params;
                f.is_old_style = false;
                f.has_params = true;
                f.variadic = false;
                f.idents.clear();
            }
            ty = self.spec_to_ty(specifiers, &declarator);
        }
        let linkage = self.compute_linkage(symbol, specifiers, BindingKind::Func);
        // The return type threads into body parsing so `return` values
        // check (and convert) against it. Dispatch guarantees a function
        // type here.
        let ret = match &ty {
            Ty::Func { ret, .. } => (**ret).clone(),
            _ => return self.fail(def_span, "function definition without function type"),
        };
        let is_async = attrs.is_async || specifiers.attrs.is_async;
        if is_async {
            if matches!(&ty, Ty::Func { variadic: true, .. }) {
                return self.fail(def_span, "async function cannot be variadic");
            }
            if self.intern.resolve(symbol).is_some_and(|name| name == "main") {
                return self.fail(def_span, "async function `main` is not supported; write a synchronous `main` and call `glcc_block_on`");
            }
            // The handle type must exist before any call site checks, so
            // a call typed `_Future(ret)` always finds this struct.
            self.sema.future_handle(ret.clone());
        }
        // Context for statement expressions (`return` checking) and
        // `__func__` (name expansion) in the body below; restored
        // afterwards.
        self.function_return = Some(ret.clone());
        self.function_name = Some(symbol);
        let naked = attrs.naked || specifiers.attrs.naked;
        if is_async && naked {
            return self.fail(def_span, "async function cannot be naked");
        }
        let section = attrs.section.or(specifiers.attrs.section);
        if section.is_some() && !naked {
            self.warn(def_span, "attribute `section` on a function is ignored");
        }
        let declared = self.sema.declare(
            symbol,
            Binding {
                name: Some(symbol),
                ty,
                kind: BindingKind::Func,
                linkage,
                block_static: false,
                tls: false,
                defined: true,
                tentative: false,
                constructor: attrs.constructor,
                destructor: attrs.destructor,
                // Non-static `inline` definitions are weak under the C++
                // driver (vague linkage for EDG-lowered COMDAT inline
                // functions, which lose their `extern` on the
                // definition); see `Sema::weak_extern_inline`.
                weak: attrs.weak || specifiers.attrs.weak || (self.sema.weak_extern_inline && specifiers.is_inline() && !specifiers.has_storage(StorageClass::Static)),
                alias: None,
                naked,
                is_async,
                inline: specifiers.is_inline(),
                noinline: attrs.noinline || specifiers.attrs.noinline,
                align: attrs.aligned.or(specifiers.attrs.aligned),
                section,
                asm_label: attrs.asm_label.or(specifiers.attrs.asm_label),
                span: def_span,
            },
        );
        let id = self.unwrap_diag(declared);
        if self.failed() {
            return id;
        }
        // The merged binding knows about a `naked` prototype even when
        // this definition's attributes omit it.
        self.naked_function = self.sema.decls[id.0 as usize].naked;
        // Function scope: parameters live here (C17 §6.2.1p4), so body
        // blocks nest inside it and `int f(int x) { int x; }` shadows
        // rather than conflicts. Old-style identifier-lists carry no
        // types, so only prototyped parameters declare.
        self.sema.push_scope();
        let mut params = Vec::new();
        let mut remap = HashMap::new();
        let mut entry_bounds: Vec<(u32, Span)> = Vec::new();
        if let Some(suffix) = Self::defined_params(&declarator) {
            for param in &suffix.params {
                // `[*]` is only for prototypes, never for the parameter
                // list of a definition (verified against gcc).
                if param.star {
                    return self.fail(param.span, "`[*]` outside function prototype scope");
                }
                if let Some(bound) = param.vla_bound {
                    entry_bounds.push((bound, param.span));
                }
                let Some(name) = param.name else { continue };
                let declared = self.sema.declare(
                    name,
                    Binding {
                        name: Some(name),
                        ty: param.body_ty.clone().unwrap_or_else(|| param.ty.clone()),
                        kind: BindingKind::Var,
                        linkage: Linkage::NoLinkage,
                        block_static: false,
                        tls: false,
                        defined: true,
                        tentative: false,
                        constructor: None,
                        destructor: None,
                        weak: false,
                        alias: None,
                        naked: false,
                        noinline: false,
                        is_async: false,
                        section: None,
                        asm_label: None,
                        inline: false,
                        align: None,
                        span: param.span,
                    },
                );
                let new = self.unwrap_diag(declared);
                if self.failed() {
                    break;
                }
                // VLA bounds in later parameters were checked against
                // the prototype-scope binding; point them at the
                // body's binding instead, or the backend would lower a
                // dead `DeclId` (e.g. `void f(int n, int a[m][n])`).
                // Collected first and applied in one simultaneous pass
                // below: sequential rewrites cascade when ids overlap
                // (K&R parameters reuse truncated temporaries).
                if let Some(proto) = param.proto {
                    remap.insert(proto, new);
                }
                params.push(new);
            }
        }
        if !remap.is_empty() {
            self.sema.remap_vla_idents(&remap);
        }
        let outer_async = self.sema.checking_async_fn.replace(is_async);
        let block = {
            // Pending initializer gotos belong to this body alone;
            // drop any left by an earlier aborted parse on a reused
            // parser (tests catch panics and keep going).
            self.pending_gotos.clear();
            let mut block = self.parse_compound_statement(&ret);
            // VLA parameter bounds evaluate on entry (C17 §6.7.6.2p6):
            // the parameter decayed to a pointer, but `int a[n++]`
            // still steps `n` once per call, in declaration order.
            if !entry_bounds.is_empty() {
                let mut items = Vec::with_capacity(block.items.len() + entry_bounds.len());
                for (id, span) in &entry_bounds {
                    let expr = self.sema.vla_bounds[*id as usize].clone();
                    items.push(BlockItem::Stmt(Stmt::Expr { expr, span: *span }));
                }
                items.append(&mut block.items);
                block.items = items;
            }
            self.check_function_body(&block);
            block
        };
        self.sema.checking_async_fn.set(outer_async);
        if self.naked_function {
            self.check_naked_body(&block);
        }
        self.sema.pop_scope();
        // Statement expressions and `__func__` outside any body see no
        // enclosing function again afterwards.
        self.function_return = None;
        self.function_name = None;
        self.naked_function = false;
        self.emit_warnings();
        let previous = self.sema.bodies.insert(
            id,
            FuncBody {
                span: def_span.merge(block.span),
                params,
                block,
            },
        );
        debug_assert!(previous.is_none(), "function body recorded twice");
        id
    }
    /// A naked function's body is emitted verbatim as assembler, so it
    /// may hold only basic asm statements (no operands) and empty
    /// statements; anything else would need a compiler prologue.
    fn check_naked_body(&self, block: &Compound) {
        for item in &block.items {
            let ok = match item {
                BlockItem::Stmt(Stmt::Asm(asm)) => asm.outputs.is_empty() && asm.inputs.is_empty() && asm.clobbers.is_empty() && asm.labels.is_empty(),
                BlockItem::Stmt(Stmt::Empty { .. }) => true,
                _ => false,
            };
            if !ok {
                return self.fail(item.span(), "naked function body may only contain basic asm statements");
            }
        }
    }
    /// The parameter list of the defined function: the innermost
    /// prototyped function suffix on the path to the declared name.
    /// For `(*f(p))(void)` the definition's parameters are `(p)` (the
    /// outer `(void)` belongs to the pointed-to type), while
    /// `int (f)(int)` still finds the `(int)`. Old-style
    /// identifier-lists carry no parameter types, so this is `None`
    /// for them.
    fn defined_params(declarator: &Declarator) -> Option<&FunctionDeclarator> {
        let mut d = declarator;
        let mut found = None;
        loop {
            for suffix in &d.direct.suffixes {
                if let DirectSuffix::Function(f) = suffix {
                    if !f.is_old_style {
                        found = Some(f);
                    }
                }
            }
            match &d.direct.base {
                DirectBase::Paren(inner) => d = inner,
                _ => return found,
            }
        }
    }
    /// The identifier-list of the defined function: the innermost
    /// old-style suffix on the path to the declared name, mirroring
    /// `defined_params`. After the K&R rewrite runs the suffix becomes
    /// prototyped, so this is `None` for rewritten definitions.
    fn defined_old_style(declarator: &Declarator) -> Option<&FunctionDeclarator> {
        let mut d = declarator;
        let mut found = None;
        loop {
            for suffix in &d.direct.suffixes {
                if let DirectSuffix::Function(f) = suffix {
                    if f.is_old_style {
                        found = Some(f);
                    }
                }
            }
            match &d.direct.base {
                DirectBase::Paren(inner) => d = inner,
                _ => return found,
            }
        }
    }
    fn defined_old_style_mut(declarator: &mut Declarator) -> Option<&mut FunctionDeclarator> {
        let mut d = declarator;
        let mut found: Option<*mut FunctionDeclarator> = None;
        loop {
            for suffix in &mut d.direct.suffixes {
                if let DirectSuffix::Function(f) = suffix {
                    if f.is_old_style {
                        found = Some(f as *mut FunctionDeclarator);
                    }
                }
            }
            match &mut d.direct.base {
                DirectBase::Paren(inner) => d = inner,
                _ => return found.map(|p| unsafe { &mut *p }),
            }
        }
    }
    /// Default promotions for a K&R parameter type: narrow integers to
    /// `int` and `float` to `double`. Array and function declarators
    /// already decayed to pointers before this runs. Outer qualifiers
    /// are preserved (`const char` promotes to `const int`). The
    /// promoted type is the ABI's; the body keeps the declared one
    /// (`Param::body_ty`) and converts on entry.
    fn kr_promote_ty(ty: Ty) -> Ty {
        match ty {
            Ty::Qualified { quals, inner } => {
                let promoted = Self::kr_promote_ty(*inner);
                Ty::qualified(quals, promoted)
            }
            Ty::Bool | Ty::Char | Ty::SignedChar | Ty::UnsignedChar | Ty::Short | Ty::UnsignedShort => Ty::Int,
            Ty::Float => Ty::Double,
            other => other,
        }
    }
    /// One declaration of a K&R declaration-list: `specifiers
    /// declarator-list `;``. Declarators must name parameters from the
    /// identifier-list; their types are adjusted like modern parameters
    /// (function to pointer, array to pointer) and recorded in `out`.
    /// Each parameter also declares in the temporary scope so later
    /// bounds (`int a[n]`) resolve `n`. Initializers, non-`register`
    /// storage, and `void` parameters are rejected.
    fn parse_kr_param_declaration(&mut self, old_ids: &[(SymbolU32, Span)], out: &mut HashMap<SymbolU32, (Ty, Span, DeclId)>) {
        let decl_start = self.mark();
        self.skip_gnu_suffixes();
        let specs = self.parse_declaration_specifiers();
        if specs.class.iter().any(|sc| !matches!(sc, StorageClass::Register)) {
            let span = self.span_since(decl_start);
            return self.fail(span, "storage class specifier in parameter");
        }
        if specs.thread_local {
            let span = self.span_since(decl_start);
            return self.fail(span, "thread-local storage in parameter");
        }
        if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Semi))) {
            self.tokens.advance();
            let span = self.span_since(decl_start);
            match specs.type_specifier.as_slice() {
                [TypeSpecifier::Struct(id)] => {
                    if self.sema.structs.get(id.0 as usize).map_or(true, |s| s.tag.is_none()) {
                        return self.fail(span, "declaration does not declare anything");
                    }
                }
                [TypeSpecifier::Enum(_)] => {}
                _ => {
                    return self.fail(span, "declaration does not declare anything");
                }
            }
            return;
        }
        loop {
            let declarator = self.parse_declarator();
            self.skip_gnu_suffixes();
            let decl_span = specs.span().merge(declarator.span());
            let Some(name) = declarator.name() else {
                return self.fail(decl_span, "declaration does not declare anything");
            };
            if !old_ids.iter().any(|(id, _)| *id == name) {
                let text = self.intern.resolve(name).unwrap_or("?");
                return self.fail(decl_span, format!("declaration for parameter '{text}' but no such parameter"));
            }
            if let Some((_, prev, _)) = out.get(&name) {
                let text = self.intern.resolve(name).unwrap_or("?");
                return self.fail_secondary(decl_span, format!("redefinition of parameter '{text}'"), *prev, "previous definition here");
            }
            let mut ty = self.spec_to_ty(&specs, &declarator);
            let star = ty_has_bare_star(&ty);
            if star {
                return self.fail(decl_span, "`[*]` outside function prototype scope");
            }
            if matches!(ty, Ty::Func { .. }) {
                ty = Ty::Pointer {
                    qualifiers: Vec::new(),
                    size: PtrSize::Default,
                    pointee: Box::new(ty),
                };
            }
            if let Ty::Array { elem, quals, .. } = ty {
                ty = Ty::Pointer { qualifiers: quals, size: PtrSize::Default, pointee: elem };
            }
            if ty.unqualified() == &Ty::Void {
                return self.fail(decl_span, "declaration of `void` object");
            }
            if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Assign))) {
                return self.fail(decl_span, "initializer in parameter declaration");
            }
            let declared = self.sema.declare(
                name,
                Binding {
                    name: Some(name),
                    ty: ty.clone(),
                    kind: BindingKind::Var,
                    linkage: Linkage::NoLinkage,
                    block_static: false,
                    tls: false,
                    defined: false,
                    tentative: false,
                    constructor: None,
                    destructor: None,
                    weak: false,
                    alias: None,
                    naked: false,
                    noinline: false,
                    is_async: false,
                    section: None,
                    asm_label: None,
                    inline: false,
                    align: None,
                    span: decl_span,
                },
            );
            let id = self.unwrap_diag(declared);
            if self.failed() {
                return;
            }
            out.insert(name, (ty, decl_span, id));
            match self.tokens.peek_val() {
                Some(Token::Punctuator(Punctuator::Comma)) => {
                    self.tokens.advance();
                    continue;
                }
                Some(Token::Punctuator(Punctuator::Semi)) => {
                    self.tokens.advance();
                    break;
                }
                _ => {
                    let span = self.peek_span();
                    return self.fail(span, "expected `,` or `;` in parameter declaration");
                }
            }
        }
    }
    /// Compound statement `{ block-item-list_opt }`: pushes a block scope,
    /// parses declarations for real and nested blocks recursively, and
    /// returns the items with the braces' span. `ret` is the enclosing
    /// function's return type, threaded to `return` checking.
    pub fn parse_compound_statement(&mut self, ret: &Ty) -> Compound {
        let start = self.mark();
        self.expect_punct(Punctuator::LBrace);
        self.sema.push_scope();
        self.label_scopes.push(HashMap::new());
        let mut items = Vec::new();
        let mut unterminated = false;
        loop {
            if self.cancelled() {
                break;
            }
            // `_Pragma("...")` may separate block items, or sit last
            // before the close brace; skip it like `#pragma`.
            if self.peek_pragma_operator() {
                self.skip_pragma_operator();
                continue;
            }
            match self.tokens.peek_val() {
                Some(Token::Punctuator(Punctuator::RBrace)) => break,
                None => {
                    unterminated = true;
                    break;
                }
                _ => {
                    let before = self.tokens.position();
                    items.push(self.parse_block_item(ret));
                    if self.cancelled() {
                        break;
                    }
                    if self.failed() {
                        self.resync_statement();
                        self.clear_failure();
                        if self.tokens.position() == before {
                            self.tokens.advance();
                        }
                    } else if self.tokens.position() == before {
                        self.tokens.advance();
                    }
                }
            }
        }
        if !self.cancelled() {
            if unterminated {
                self.fail::<()>(self.span_since(start), "unterminated compound statement");
            } else {
                self.expect_punct(Punctuator::RBrace);
            }
        }
        self.sema.pop_scope();
        self.label_scopes.pop();
        Compound {
            span: self.span_since(start),
            items,
        }
    }
    /// Whether the next tokens are the C99 `_Pragma ( string-literal )`
    /// operator (C17 §6.10.9). The native preprocessor lowers the
    /// operator to a `#pragma` line before the parser runs, so only
    /// raw-source callers reach one; it is skipped wherever a declaration
    /// or statement may start, like `#pragma` itself.
    fn peek_pragma_operator(&self) -> bool {
        matches!(self.tokens.peek_val(), Some(Token::Identifier(sym)) if self.intern.resolve(*sym) == Some("_Pragma"))
    }

    /// Consume a `_Pragma ( string-literal )` operator, diagnosing a
    /// malformed operand. The pragma's effect is ignored, matching the
    /// `#pragma` treatment (including `pack`).
    fn skip_pragma_operator(&mut self) {
        let start = self.mark();
        self.tokens.advance();
        if !self.eat_punct(Punctuator::LParen) {
            return self.fail(self.span_since(start), "`_Pragma` requires a parenthesized string literal");
        }
        if !matches!(self.tokens.peek_val(), Some(Token::StringLiteral(_))) {
            return self.fail(self.peek_span(), "`_Pragma` requires a string literal");
        }
        self.tokens.advance();
        self.expect_punct(Punctuator::RParen);
    }

    /// Whether the next tokens are an MSVC annotation operator:
    /// `__pragma(...)` (the inline `#pragma` spelling, `corecrt.h`
    /// writes `__pragma(pack(push, 8))`) or `__declspec(...)`.
    fn peek_msvc_operator(&self, word: &str) -> bool {
        self.peek_is_gnu(word) && matches!(self.tokens.peek_nth(1).map(|t| &t.val), Some(Token::Punctuator(Punctuator::LParen)))
    }

    /// Consume an MSVC annotation operator with balanced parentheses.
    /// The contents are annotation text; the frontend ignores it (the
    /// same treatment `#pragma` gets — including `pack`, so `#pragma
    /// pack` layouts stay unsupported either way).
    fn skip_msvc_operator(&mut self) {
        self.tokens.advance();
        self.tokens.advance();
        let mut depth = 1usize;
        while depth > 0 {
            match self.tokens.peek_val() {
                Some(Token::Punctuator(Punctuator::LParen)) => depth += 1,
                Some(Token::Punctuator(Punctuator::RParen)) => depth -= 1,
                Some(_) => {}
                None => break,
            }
            self.tokens.advance();
        }
    }

    /// One block item: a declaration if one can start here (including a
    /// lone `struct S;`, which declares nothing), otherwise a statement.
    /// GNU markers prefixing either are skipped first.
    fn parse_block_item(&mut self, ret: &Ty) -> BlockItem {
        // `__asm__(...)` starts a statement here, not a declaration
        // annotation, so `__asm__` labels are not skipped.
        self.skip_gnu_markers();
        if self.peek_msvc_operator("__pragma") {
            self.skip_msvc_operator();
        }
        if self.peek_is_static_assert() {
            return BlockItem::Stmt(self.parse_static_assert());
        }
        if self.can_start_declaration() {
            let start = self.mark();
            let ids = self.parse_declaration();
            BlockItem::Decl {
                ids,
                span: self.span_since(start),
            }
        } else {
            BlockItem::Stmt(self.parse_statement(ret))
        }
    }
    /// Whether the next token spells `asm`, `__asm__`, or `__asm` —
    /// the GNU asm family (gated by `-fno-gnu-extensions`).
    fn peek_is_asm_keyword(&self) -> bool {
        match self.tokens.peek_val() {
            Some(Token::Identifier(sym)) if matches!(self.intern.resolve(*sym), Some("asm" | "__asm__" | "__asm")) => self.sema.lang.gnu,
            _ => false,
        }
    }

    /// Classify a qualifier between `asm` and its `(`: `Some((volatile,
    /// goto))` for `volatile`/`inline`/`goto` and their GNU spellings.
    fn asm_qualifier(&self, token: &Token) -> Option<(bool, bool)> {
        match token {
            Token::Keyword(Keyword::Volatile) => Some((true, false)),
            Token::Keyword(Keyword::Goto) => Some((false, true)),
            Token::Keyword(Keyword::Inline) => Some((false, false)),
            Token::Identifier(sym) => match self.intern.resolve(*sym) {
                Some("volatile" | "__volatile__") => Some((true, false)),
                Some("goto" | "__goto__") => Some((false, true)),
                Some("inline" | "__inline__") => Some((false, false)),
                _ => None,
            },
            _ => None,
        }
    }

    /// Consume a punctuator when it is next.
    fn eat_punct(&mut self, punct: Punctuator) -> bool {
        if matches!(self.tokens.peek_val(), Some(Token::Punctuator(p)) if *p == punct) {
            self.tokens.advance();
            true
        } else {
            false
        }
    }

    /// Whether an `asm` statement starts here, looking through
    /// `__extension__` markers: an `asm` spelling followed by qualifiers
    /// and `(`.
    fn asm_statement_ahead(&self) -> bool {
        let mut scan = 0;
        loop {
            match self.tokens.peek_nth(scan).map(|t| &t.val) {
                Some(Token::Identifier(sym)) if self.intern.resolve(*sym) == Some("__extension__") => {
                    scan += 1;
                }
                Some(Token::Identifier(sym)) if matches!(self.intern.resolve(*sym), Some("asm" | "__asm__" | "__asm")) => {
                    scan += 1;
                    break;
                }
                _ => return false,
            }
        }
        loop {
            match self.tokens.peek_nth(scan).map(|t| &t.val) {
                Some(token) if self.asm_qualifier(token).is_some() => scan += 1,
                Some(Token::Punctuator(Punctuator::LParen)) => return true,
                _ => return false,
            }
        }
    }

    /// An `asm` statement starting at `start`, or `None` when the `asm`
    /// spelling does not begin one (so it parses as an identifier).
    /// Boxed so the recursive statement path never holds a whole `Stmt`
    /// temporary.
    ///
    /// Grammar: `asm [qualifiers] ( template [: outputs [: inputs
    /// [: clobbers [: goto-labels]]]] )`. Only the empty template runs
    /// so far; a non-empty one is rejected until the backend can emit
    /// it.
    fn try_parse_asm_statement(&mut self, start: Location) -> Option<Box<Stmt>> {
        while self.peek_is_gnu("__extension__") {
            self.tokens.advance();
        }
        if !self.asm_statement_ahead() {
            return None;
        }
        self.tokens.advance();
        let mut is_volatile = false;
        let mut is_goto = false;
        while let Some((qual_volatile, qual_goto)) = self.tokens.peek_val().and_then(|t| self.asm_qualifier(t)) {
            is_volatile |= qual_volatile;
            is_goto |= qual_goto;
            self.tokens.advance();
        }
        self.expect_punct(Punctuator::LParen);
        let template = self.parse_asm_string("template");
        let mut outputs = Vec::new();
        let mut inputs = Vec::new();
        let mut clobbers = Vec::new();
        let mut labels = Vec::new();
        let mut saw_labels = false;
        let saw_colon = self.eat_punct(Punctuator::Colon);
        if saw_colon {
            outputs = self.parse_asm_operands(true);
            if self.eat_punct(Punctuator::Colon) {
                inputs = self.parse_asm_operands(false);
                if self.eat_punct(Punctuator::Colon) {
                    clobbers = self.parse_asm_string_list();
                    if self.eat_punct(Punctuator::Colon) {
                        labels = self.parse_asm_label_list();
                        saw_labels = true;
                    }
                }
            }
        }
        self.expect_punct(Punctuator::RParen);
        self.expect_punct(Punctuator::Semi);
        for input in &inputs {
            if let Some(tie) = input.tie() {
                if tie >= outputs.len() {
                    return self.fail(input.span, "asm operand number out of range");
                }
            }
        }
        for output in &outputs {
            if !output.constraint.starts_with('=') && !output.constraint.starts_with('+') {
                return self.fail(output.span, "output operand constraint lacks '='");
            }
        }
        let span = self.span_since(start);
        if is_goto && (!saw_labels || labels.is_empty()) {
            return self.fail(span, "asm goto needs at least one label");
        }
        if !is_goto && saw_labels {
            return self.fail(span, "asm label list requires asm goto");
        }
        let listed: Vec<&str> = labels.iter().filter_map(|label| self.intern.resolve(*label)).collect();
        for name in asm_label_refs(&template) {
            if !listed.contains(&name.as_str()) {
                return self.fail(span, format!("undefined named operand '{name}'"));
            }
        }
        Some(Box::new(Stmt::Asm(AsmStmt {
            template,
            outputs,
            inputs,
            clobbers,
            labels,
            is_volatile,
            is_goto,
            is_basic: !saw_colon,
            span,
        })))
    }

    /// Whether a constraint's only alternative is memory: it names a
    /// memory class and no register class, so the operand must be an
    /// addressable object. (A `"rm"` operand may take the register
    /// alternative instead.)
    fn memory_only_constraint(&self, constraint: &str) -> bool {
        crate::check::memory_only_constraint(constraint, self.sema.target.arch)
    }
    /// One colon section of an `asm` statement: `[name] "constraint"
    /// (expr)` operands, comma-separated; an empty section is allowed.
    fn parse_asm_operands(&mut self, outputs: bool) -> Vec<AsmOperand> {
        let mut operands = Vec::new();
        loop {
            if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Colon | Punctuator::RParen))) {
                break;
            }
            let start = self.mark();
            let name = self.parse_asm_operand_name();
            let constraint = self.parse_asm_string("constraint");
            self.expect_punct(Punctuator::LParen);
            let expr = self.parse_expression();
            let expr = if outputs {
                self.checked_asm_output(&expr)
            } else {
                self.checked_expr(&expr)
            };
            self.expect_punct(Punctuator::RParen);
            let span = self.span_since(start);
            if self.memory_only_constraint(&constraint) {
                if crate::check::designates_bitfield(&expr) {
                    return self.fail(span, "cannot take address of bit-field");
                }
                if !outputs && !crate::check::is_lvalue(&expr, &self.sema.decls) {
                    return self.fail(span, "memory input is not directly addressable");
                }
            }
            operands.push(AsmOperand { name, constraint, expr, span });
            if !self.eat_punct(Punctuator::Comma) {
                break;
            }
        }
        operands
    }

    /// The optional `[name]` of an operand.
    fn parse_asm_operand_name(&mut self) -> Option<SymbolU32> {
        if !matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LBracket))) {
            return None;
        }
        self.tokens.advance();
        let name = match self.tokens.advance().map(|located| located.val) {
            Some(Token::Identifier(sym)) => sym,
            other => {
                let span = self.peek_span();
                return self.fail(span, format!("expected asm operand name, got {other:?}"));
            }
        };
        self.expect_punct(Punctuator::RBracket);
        Some(name)
    }

    /// The clobber section: string literals, comma-separated.
    fn parse_asm_string_list(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        loop {
            if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Colon | Punctuator::RParen))) {
                break;
            }
            out.push(self.parse_asm_string("clobber"));
            if !self.eat_punct(Punctuator::Comma) {
                break;
            }
        }
        out
    }

    /// The `asm goto` label section: label names, comma-separated.
    fn parse_asm_label_list(&mut self) -> Vec<SymbolU32> {
        let mut out = Vec::new();
        loop {
            if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::RParen))) {
                break;
            }
            match self.tokens.advance().map(|located| located.val) {
                Some(Token::Identifier(sym)) => out.push(self.resolve_label(sym)),
                other => {
                    let span = self.peek_span();
                    return self.fail(span, format!("expected label name in asm goto, got {other:?}"));
                }
            }
            if !self.eat_punct(Punctuator::Comma) {
                break;
            }
        }
        out
    }

    /// The symbol a label use binds to: the innermost enclosing
    /// `__label__` declaration of that spelling, else the plain
    /// function-scoped label of standard C.
    fn resolve_label(&self, name: SymbolU32) -> SymbolU32 {
        for scope in self.label_scopes.iter().rev() {
            if let Some(&local) = scope.get(&name) {
                return local;
            }
        }
        name
    }

    /// A fresh symbol for one block-local label declaration. Ids live
    /// and lower by symbol, so shadowing needs distinct ones; the
    /// dotted spelling cannot collide with real identifiers.
    fn fresh_label(&mut self, name: SymbolU32) -> SymbolU32 {
        self.label_gen += 1;
        let spelling = format!("{}.{}", self.intern.resolve(name).unwrap_or("?"), self.label_gen);
        self.intern.get_or_intern(&spelling)
    }

    /// A decoded string literal inside an `asm` statement (template,
    /// constraint, or clobber).
    fn parse_asm_string(&mut self, what: &str) -> String {
        let span = self.peek_span();
        let mut parts = Vec::new();
        while let Some(Token::StringLiteral(lit)) = self.tokens.peek_val() {
            parts.push(*lit);
            self.tokens.advance();
        }
        if parts.is_empty() {
            return self.fail(span, format!("expected string literal for asm {what}"));
        }
        match crate::eval::string_bytes(&parts, &self.intern) {
            Some(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            None => self.fail(span, format!("invalid string literal for asm {what}")),
        }
    }

    /// One statement (C17 §6.8): compounds nest, labels carry their
    /// sub-statement, and anything else is an expression statement.
    /// A bare identifier followed by `:` is a label — the `:` cannot
    /// start any expression continuation, so no ternary can be
    /// misread (`a ? b : c` never has `:` second).
    ///
    /// Every expression position checks inline and stores the checked
    /// tree; `ret` carries the enclosing return type for `return`.
    fn parse_statement(&mut self, ret: &Ty) -> Stmt {
        let start = self.mark();
        // `asm` statements are not annotations: check for one before
        // `skip_gnu_suffixes` can swallow `__asm__(...)` as a label.
        if let Some(stmt) = self.try_parse_asm_statement(start) {
            return *stmt;
        }
        // Annotations prefixing the statement (`__attribute__`), or a
        // lone annotation statement (`__attribute__((fallthrough));`
        // in `switch`es): skipped, leaving `;` for the Empty arm.
        self.skip_gnu_suffixes();
        match self.tokens.peek_val() {
            // GNU `__label__ name[, name]*;` declares block-local
            // labels: each binds to a unique symbol in the enclosing
            // block, so an inner label shadows an outer one of the same
            // spelling (GCC extension; plain C labels stay
            // function-scoped).
            Some(Token::Identifier(sym)) if self.intern.resolve(*sym) == Some("__label__") => {
                self.tokens.advance();
                loop {
                    let name = match self.tokens.advance().map(|located| located.val) {
                        Some(Token::Identifier(name)) => name,
                        other => {
                            let span = self.peek_span();
                            return self.fail(span, format!("expected label name, got {other:?}"));
                        }
                    };
                    let local = self.fresh_label(name);
                    if let Some(scope) = self.label_scopes.last_mut() {
                        scope.entry(name).or_insert(local);
                    }
                    if !matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Comma))) {
                        break;
                    }
                    self.tokens.advance();
                }
                self.expect_punct(Punctuator::Semi);
                Stmt::Empty { span: self.span_since(start) }
            }
            Some(Token::Punctuator(Punctuator::LBrace)) => Stmt::Compound(self.parse_compound_statement(ret)),
            Some(Token::Punctuator(Punctuator::Semi)) => {
                self.tokens.advance();
                Stmt::Empty { span: self.span_since(start) }
            }
            Some(Token::Keyword(Keyword::If)) => {
                self.tokens.advance();
                self.expect_punct(Punctuator::LParen);
                let cond = self.parse_expression();
                let cond = self.checked_condition(&cond);
                self.expect_punct(Punctuator::RParen);
                let mut then = self.parse_statement(ret);
                let mut els = if matches!(self.tokens.peek_val(), Some(Token::Keyword(Keyword::Else))) {
                    self.tokens.advance();
                    Some(Box::new(self.parse_statement(ret)))
                } else {
                    None
                };
                fold_constant_if(&cond, &mut then, &mut els, &self.sema, &self.intern);
                Stmt::If {
                    cond,
                    then: Box::new(then),
                    els,
                    span: self.span_since(start),
                }
            }
            Some(Token::Keyword(Keyword::While)) => {
                self.tokens.advance();
                self.expect_punct(Punctuator::LParen);
                let cond = self.parse_expression();
                let cond = self.checked_condition(&cond);
                self.expect_punct(Punctuator::RParen);
                let body = self.parse_statement(ret);
                Stmt::While {
                    cond,
                    body: Box::new(body),
                    span: self.span_since(start),
                }
            }
            Some(Token::Keyword(Keyword::Do)) => {
                self.tokens.advance();
                let body = self.parse_statement(ret);
                self.expect_keyword(Keyword::While);
                self.expect_punct(Punctuator::LParen);
                let cond = self.parse_expression();
                let cond = self.checked_condition(&cond);
                self.expect_punct(Punctuator::RParen);
                self.expect_punct(Punctuator::Semi);
                Stmt::DoWhile {
                    body: Box::new(body),
                    cond,
                    span: self.span_since(start),
                }
            }
            Some(Token::Keyword(Keyword::For)) => {
                self.tokens.advance();
                self.expect_punct(Punctuator::LParen);
                // The initializer lives in its own scope with the rest
                // of the loop (C99 onwards).
                self.sema.push_scope();
                let init = if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Semi))) {
                    self.tokens.advance();
                    ForInit::Empty
                } else if self.can_start_declaration() {
                    ForInit::Decl(self.parse_declaration())
                } else {
                    let expr = self.parse_expression();
                    let expr = self.checked_expr(&expr);
                    self.expect_punct(Punctuator::Semi);
                    ForInit::Expr(expr)
                };
                let cond = if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Semi))) {
                    self.tokens.advance();
                    None
                } else {
                    let expr = self.parse_expression();
                    let expr = self.checked_condition(&expr);
                    self.expect_punct(Punctuator::Semi);
                    Some(expr)
                };
                let step = if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::RParen))) {
                    None
                } else {
                    let expr = self.parse_expression();
                    let expr = self.checked_expr(&expr);
                    Some(expr)
                };
                self.expect_punct(Punctuator::RParen);
                let body = self.parse_statement(ret);
                self.sema.pop_scope();
                Stmt::For {
                    init,
                    cond,
                    step,
                    body: Box::new(body),
                    span: self.span_since(start),
                }
            }
            Some(Token::Keyword(Keyword::Switch)) => {
                self.tokens.advance();
                self.expect_punct(Punctuator::LParen);
                let cond = self.parse_expression();
                let cond = self.checked_integer_expr(&cond, "switch controlling expression");
                self.expect_punct(Punctuator::RParen);
                let body = self.parse_statement(ret);
                self.check_switch_cases(&cond, &body);
                Stmt::Switch {
                    cond,
                    body: Box::new(body),
                    span: self.span_since(start),
                }
            }
            Some(Token::Keyword(Keyword::Return)) => {
                self.tokens.advance();
                let expr = if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Semi))) {
                    None
                } else {
                    let expr = self.parse_expression();
                    let expr = self.checked_return(ret, &expr);
                    Some(expr)
                };
                self.expect_punct(Punctuator::Semi);
                Stmt::Return {
                    expr,
                    span: self.span_since(start),
                }
            }
            Some(Token::Keyword(Keyword::Break)) => {
                self.tokens.advance();
                self.expect_punct(Punctuator::Semi);
                Stmt::Break { span: self.span_since(start) }
            }
            Some(Token::Keyword(Keyword::Continue)) => {
                self.tokens.advance();
                self.expect_punct(Punctuator::Semi);
                Stmt::Continue { span: self.span_since(start) }
            }
            Some(Token::Keyword(Keyword::Goto)) => {
                self.tokens.advance();
                if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Star))) {
                    // GNU computed goto: the operand is any pointer
                    // expression holding a label address.
                    self.tokens.advance();
                    let expr = self.parse_expression();
                    let expr = self.checked_expr(&expr);
                    self.expect_punct(Punctuator::Semi);
                    return Stmt::GotoComputed {
                        expr,
                        span: self.span_since(start),
                    };
                }
                let label = match self.tokens.advance().map(|located| located.val) {
                    Some(Token::Identifier(name)) => self.resolve_label(name),
                    other => {
                        let span = self.peek_span();
                        return self.fail(span, format!("expected label name, got {other:?}"));
                    }
                };
                self.expect_punct(Punctuator::Semi);
                Stmt::Goto {
                    label,
                    span: self.span_since(start),
                }
            }
            Some(Token::Keyword(Keyword::Case)) => {
                self.tokens.advance();
                // A case label is a conditional expression — no
                // assignment, no comma (C17 §6.8.1p1) — and must be an
                // integer constant expression (checked by folding, like
                // enumerator values).
                let expr = self.parse_conditional_expression();
                let checked = self.checked_expr(&expr);
                if !matches!(crate::eval::eval_const(&checked, self.sema, &self.intern), Some(crate::eval::Value::Int(_))) {
                    return self.fail(checked.span, "case label is not an integer constant expression");
                }
                // GNU `case lo ... hi:`: both bounds are integer
                // constant expressions and the label covers the range.
                let hi = if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Ellipsis))) {
                    self.tokens.advance();
                    let expr = self.parse_conditional_expression();
                    let checked = self.checked_expr(&expr);
                    if !matches!(crate::eval::eval_const(&checked, self.sema, &self.intern), Some(crate::eval::Value::Int(_))) {
                        return self.fail(checked.span, "case range bound is not an integer constant expression");
                    }
                    Some(checked)
                } else {
                    None
                };
                self.expect_punct(Punctuator::Colon);
                let stmt = self.parse_statement(ret);
                Stmt::Case {
                    expr: Some(checked),
                    hi,
                    stmt: Box::new(stmt),
                    span: self.span_since(start),
                }
            }
            Some(Token::Keyword(Keyword::Default)) => {
                self.tokens.advance();
                self.expect_punct(Punctuator::Colon);
                let stmt = self.parse_statement(ret);
                Stmt::Case {
                    expr: None,
                    hi: None,
                    stmt: Box::new(stmt),
                    span: self.span_since(start),
                }
            }
            Some(Token::Identifier(_)) if matches!(self.tokens.peek_nth(1).map(|located| &located.val), Some(Token::Punctuator(Punctuator::Colon))) => {
                let name = match self.tokens.advance().map(|located| located.val) {
                    Some(Token::Identifier(name)) => self.resolve_label(name),
                    _ => unreachable!("label identifier peeked"),
                };
                self.expect_punct(Punctuator::Colon);
                // Ids run in definition order, one per spelling (the
                // first definition wins), so `&&label` and computed
                // gotos agree on them without extra context.
                let next = self.sema.label_ids.len() as u64;
                self.sema.label_ids.entry(name).or_insert(next);
                let stmt = self.parse_statement(ret);
                Stmt::Label {
                    name,
                    stmt: Box::new(stmt),
                    span: self.span_since(start),
                }
            }
            None => {
                let span = self.peek_span();
                return self.fail(span, "unterminated statement");
            }
            _ => {
                let expr = self.parse_expression();
                let expr = self.checked_expr(&expr);
                self.expect_punct(Punctuator::Semi);
                Stmt::Expr {
                    expr,
                    span: self.span_since(start),
                }
            }
        }
    }
    /// Abort like `fail` but with a secondary label pointing at the
    /// previous declaration (duplicate `default`, duplicate `case`).
    fn fail_secondary<T: Recover>(&self, span: Span, msg: impl fmt::Display, prev: Span, prev_msg: &str) -> T {
        self.report(CompileError::new(msg.to_string(), span).with_secondary(prev, prev_msg));
        T::recover()
    }
    /// Duplicate `case` values (and a second `default`) in one `switch`
    /// are constraint violations (C17 §6.8.4.2p3): walk the body,
    /// stopping at nested `switch`es — their labels belong to them, and
    /// each gets its own check when parsed. Labels compare after
    /// conversion to the promoted controlling type (like GCC), so
    /// `case -1:` duplicates `case 4294967295:` under `unsigned`.
    fn check_switch_cases(&self, cond: &CheckedExpr, body: &Stmt) {
        let promoted = crate::eval::promoted(&cond.ty);
        let mut values: Vec<(i128, Span)> = Vec::new();
        let mut default: Option<Span> = None;
        self.collect_switch_cases(&promoted, body, &mut values, &mut default);
    }
    /// One step of `check_switch_cases`: record `case`/`default` labels,
    /// recursing through compounds, labels, selections, and loops
    /// (Duff's device puts labels inside loops), but never into a
    /// nested `switch`.
    fn collect_switch_cases(&self, promoted: &Ty, stmt: &Stmt, values: &mut Vec<(i128, Span)>, default: &mut Option<Span>) {
        match stmt {
            Stmt::Switch { .. } => {}
            Stmt::Case { expr, hi, stmt, span } => {
                match expr {
                    None => {
                        if let Some(prev) = default.replace(*span) {
                            return self.fail_secondary(*span, "duplicate `default` in `switch`", prev, "previous `default` here");
                        }
                    }
                    Some(label) => {
                        // Labels already folded to `Int` by the `case`
                        // arm; compare post-conversion values. A range
                        // contributes both endpoints (its interior is
                        // not compared, like GCC's overlap warning).
                        for label in core::iter::once(label).chain(hi.as_ref()) {
                            if let Some(v) = crate::eval::const_as(promoted, label, self.sema, &self.intern) {
                                if let Some((_, prev)) = values.iter().find(|(u, _)| *u == v) {
                                    return self.fail_secondary(*span, "duplicate `case` value in `switch`", *prev, "previous `case` here");
                                }
                                values.push((v, *span));
                            }
                        }
                    }
                }
                self.collect_switch_cases(promoted, stmt, values, default);
            }
            Stmt::Compound(compound) => {
                for item in &compound.items {
                    if let BlockItem::Stmt(inner) = item {
                        self.collect_switch_cases(promoted, inner, values, default);
                    }
                }
            }
            Stmt::If { then, els, .. } => {
                self.collect_switch_cases(promoted, then, values, default);
                if let Some(els) = els {
                    self.collect_switch_cases(promoted, els, values, default);
                }
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
                self.collect_switch_cases(promoted, body, values, default);
            }
            Stmt::For { body, .. } => {
                self.collect_switch_cases(promoted, body, values, default);
            }
            Stmt::Label { stmt, .. } => {
                self.collect_switch_cases(promoted, stmt, values, default);
            }
            Stmt::Empty { .. }
            | Stmt::Expr { .. }
            | Stmt::Asm(_)
            | Stmt::Return { .. }
            | Stmt::Break { .. }
            | Stmt::Continue { .. }
            | Stmt::Goto { .. }
            | Stmt::GotoComputed { .. } => {}
        }
    }
    /// Validate control flow in a parsed function body: labels are
    /// unique, every `goto` resolves to a label in the same function,
    /// `break`/`continue` sit in a loop (or, for `break`, a `switch`),
    /// and `case`/`default` sit in a `switch` (all C17 §6.8). Pending
    /// gotos from the body's initializers (collected at insert time,
    /// since their statement expressions can jump to body labels)
    /// resolve here too.
    fn check_function_body(&mut self, block: &Compound) {
        let mut labels: Vec<(SymbolU32, Span)> = Vec::new();
        let mut gotos = mem::take(&mut self.pending_gotos);
        let mut fv = FlowVla::new();
        for item in &block.items {
            match item {
                BlockItem::Decl { ids, .. } => {
                    self.record_vla_decls(ids, 0, &mut fv);
                    for id in ids {
                        self.check_flow_decl_init(*id, &mut gotos, 0, &mut fv);
                    }
                }
                BlockItem::Stmt(stmt) => {
                    self.check_flow_stmt(stmt, FlowCtx::NONE, &mut labels, &mut gotos, 0, &mut fv);
                }
            }
        }
        for (name, span) in &gotos {
            if !labels.iter().any(|(defined, _)| defined == name) {
                let text = self.intern.resolve(*name).unwrap_or("?");
                return self.fail(*span, format!("label '{text}' used but not defined"));
            }
        }
        for (name, seq, span) in mem::take(&mut fv.gotos) {
            let Some((scope, label_seq)) = fv.labels.get(&name).copied() else {
                continue;
            };
            self.check_vla_entry(seq, scope, label_seq, span, &fv);
        }
    }

    /// Whether `ty` names a VLA object (an array type that is variably
    /// modified): the declarations `check_vla_entry` tracks.
    fn is_vla_object_ty(&self, ty: &Ty) -> bool {
        matches!(ty.unqualified(), Ty::Array { .. }) && self.ty_is_variably_modified(ty)
    }

    /// Record the VLA objects among `ids` in flow scope `scope`.
    fn record_vla_decls(&self, ids: &[DeclId], scope: usize, fv: &mut FlowVla) {
        for id in ids {
            let Some(binding) = self.sema.decls.get(id.0 as usize) else {
                continue;
            };
            if binding.kind == BindingKind::Var && self.is_vla_object_ty(&binding.ty) {
                fv.record_vla(scope);
            }
        }
    }

    /// Flow-walk one declaration's initializer at the declaration's
    /// scope: only statement expressions can hold control flow, and
    /// gotos escaping them resolve outward (see `check_flow_expr`).
    fn check_flow_decl_init(&self, id: DeclId, gotos: &mut Vec<(SymbolU32, Span)>, scope: usize, fv: &mut FlowVla) {
        if let Some(init) = self.sema.inits.get(&id).cloned() {
            self.check_flow_init(&init, gotos, scope, fv);
        }
    }

    /// Whether a jump elaborated at `goto_seq` may land at
    /// (`label_scope`, `label_seq`): no VLA object may be in scope
    /// there past an unelaborated declaration — the jump would enter
    /// its scope (C11 §6.8.6.1p1, "jump into scope of identifier with
    /// variably modified type").
    fn check_vla_entry(&self, goto_seq: u64, label_scope: usize, label_seq: u64, span: Span, fv: &FlowVla) {
        for (vscope, vseq) in &fv.vlas {
            if *vseq < label_seq && goto_seq < *vseq && fv.encloses(*vscope, label_scope) {
                return self.fail(span, "jump into scope of identifier with variably modified type");
            }
        }
    }
    /// One step of `check_function_body`: enforce label uniqueness and
    /// `break`/`continue`/`case` placement, recursing with the
    /// enclosing context (loops and switches nest; labels do not reset
    /// the context).
    fn check_flow_stmt(&self, stmt: &Stmt, ctx: FlowCtx, labels: &mut Vec<(SymbolU32, Span)>, gotos: &mut Vec<(SymbolU32, Span)>, scope: usize, fv: &mut FlowVla) {
        match stmt {
            Stmt::Compound(compound) => {
                let inner = fv.alloc_scope(scope);
                for item in &compound.items {
                    match item {
                        BlockItem::Decl { ids, .. } => {
                            self.record_vla_decls(ids, inner, fv);
                            for id in ids {
                                self.check_flow_decl_init(*id, gotos, inner, fv);
                            }
                        }
                        BlockItem::Stmt(inner_stmt) => {
                            self.check_flow_stmt(inner_stmt, ctx, labels, gotos, inner, fv);
                        }
                    }
                }
            }
            Stmt::Empty { .. } => {}
            Stmt::Expr { expr, .. } => self.check_flow_expr(expr, labels, gotos, scope, fv),
            Stmt::Return { expr, .. } => {
                if let Some(expr) = expr {
                    self.check_flow_expr(expr, labels, gotos, scope, fv);
                }
            }
            Stmt::If { cond, then, els, .. } => {
                self.check_flow_expr(cond, labels, gotos, scope, fv);
                self.check_flow_stmt(then, ctx, labels, gotos, scope, fv);
                if let Some(els) = els {
                    self.check_flow_stmt(els, ctx, labels, gotos, scope, fv);
                }
            }
            Stmt::While { cond, body, .. } | Stmt::DoWhile { body, cond, .. } => {
                self.check_flow_expr(cond, labels, gotos, scope, fv);
                self.check_flow_stmt(body, ctx.in_loop(), labels, gotos, scope, fv);
            }
            Stmt::For { init, cond, step, body, .. } => {
                // The `for` initializer scope covers the whole
                // statement: VLA objects declared there live across
                // iterations.
                let fscope = fv.alloc_scope(scope);
                match init {
                    ForInit::Decl(ids) => {
                        self.record_vla_decls(ids, fscope, fv);
                        for id in ids {
                            self.check_flow_decl_init(*id, gotos, fscope, fv);
                        }
                    }
                    ForInit::Expr(expr) => self.check_flow_expr(expr, labels, gotos, fscope, fv),
                    ForInit::Empty => {}
                }
                if let Some(cond) = cond {
                    self.check_flow_expr(cond, labels, gotos, fscope, fv);
                }
                if let Some(step) = step {
                    self.check_flow_expr(step, labels, gotos, fscope, fv);
                }
                self.check_flow_stmt(body, ctx.in_loop(), labels, gotos, fscope, fv);
            }
            Stmt::Switch { cond, body, .. } => {
                self.check_flow_expr(cond, labels, gotos, scope, fv);
                // The dispatch jumps over anything declared before a
                // case: model each case as a goto from here.
                let dispatch = fv.bump();
                fv.switches.push((dispatch, Vec::new()));
                self.check_flow_stmt(body, ctx.in_switch(), labels, gotos, scope, fv);
                let (_, cases) = fv.switches.pop().expect("switch pushed by this arm");
                for (case_scope, case_seq, case_span) in cases {
                    self.check_vla_entry(dispatch, case_scope, case_seq, case_span, fv);
                }
            }
            Stmt::Break { span } => {
                if !ctx.in_loop && !ctx.in_switch {
                    return self.fail(*span, "break statement not within loop or switch");
                }
            }
            Stmt::Continue { span } => {
                if !ctx.in_loop {
                    return self.fail(*span, "continue statement not within a loop");
                }
            }
            Stmt::Goto { label, span } => {
                gotos.push((*label, *span));
                fv.record_goto(*label, *span);
            }
            Stmt::GotoComputed { expr, .. } => self.check_flow_expr(expr, labels, gotos, scope, fv),
            Stmt::Asm(asm) => {
                for op in asm.outputs.iter().chain(&asm.inputs) {
                    self.check_flow_expr(&op.expr, labels, gotos, scope, fv);
                }
                for label in &asm.labels {
                    gotos.push((*label, asm.span));
                    fv.record_goto(*label, asm.span);
                }
            }
            Stmt::Label { name, stmt, span } => {
                if let Some((_, prev)) = labels.iter().find(|(defined, _)| defined == name) {
                    let text = self.intern.resolve(*name).unwrap_or("?");
                    return self.fail_secondary(*span, format!("duplicate label '{text}'"), *prev, "previous definition here");
                }
                labels.push((*name, *span));
                fv.record_label(*name, scope);
                self.check_flow_stmt(stmt, ctx, labels, gotos, scope, fv);
            }
            Stmt::Case { stmt, span, .. } => {
                if !ctx.in_switch {
                    return self.fail(*span, "case label not within a switch statement");
                }
                // A case is a dispatch target like a label.
                let seq = fv.bump();
                if let Some((_, cases)) = fv.switches.last_mut() {
                    cases.push((scope, seq, *span));
                }
                self.check_flow_stmt(stmt, ctx, labels, gotos, scope, fv);
            }
        }
    }
    /// One initializer's control flow: only statement expressions can
    /// hold any, with their own label scope (jumping out resolves in
    /// the caller's `gotos`, like GCC; jumping in resolves to nothing
    /// and fails as undefined).
    fn check_flow_init(&self, init: &Initializer, gotos: &mut Vec<(SymbolU32, Span)>, scope: usize, fv: &mut FlowVla) {
        match init {
            Initializer::Expr(expr) => self.check_flow_expr(expr, &mut Vec::new(), gotos, scope, fv),
            Initializer::List { items, .. } => {
                for item in items {
                    self.check_flow_init(&item.init, gotos, scope, fv);
                }
            }
        }
    }
    /// Control flow inside an expression: only statement expressions
    /// hold statements, each with a fresh label scope.
    fn check_flow_expr(&self, expr: &CheckedExpr, labels: &mut Vec<(SymbolU32, Span)>, gotos: &mut Vec<(SymbolU32, Span)>, scope: usize, fv: &mut FlowVla) {
        match &expr.kind {
            CheckedExprKind::Ident(_)
            | CheckedExprKind::Int(_)
            | CheckedExprKind::Float(_)
            | CheckedExprKind::Char(_)
            | CheckedExprKind::Str(_)
            | CheckedExprKind::ConstantP(_)
            | CheckedExprKind::ClassifyType(_)
            | CheckedExprKind::Offsetof { .. }
            | CheckedExprKind::SizeofTy(_)
            | CheckedExprKind::AlignofTy(_) => {}
            // Taking a label's address is a use of it, checked exactly
            // like a `goto`: the label must exist in this function.
            CheckedExprKind::LabelAddr(name) => gotos.push((*name, expr.span)),
            CheckedExprKind::Unary(_, operand)
            | CheckedExprKind::Await(operand)
            | CheckedExprKind::Cast { expr: operand, .. }
            | CheckedExprKind::Convert { expr: operand, .. }
            | CheckedExprKind::SizeofExpr(operand)
            | CheckedExprKind::AlignofExpr(operand)
            | CheckedExprKind::VaArg { ap: operand, .. } => {
                self.check_flow_expr(operand, labels, gotos, scope, fv);
            }
            CheckedExprKind::Binary(_, lhs, rhs) | CheckedExprKind::Assign(_, lhs, rhs) | CheckedExprKind::Subscript { base: lhs, index: rhs } => {
                self.check_flow_expr(lhs, labels, gotos, scope, fv);
                self.check_flow_expr(rhs, labels, gotos, scope, fv);
            }
            CheckedExprKind::CompoundAssign { lhs, rhs, .. } => {
                self.check_flow_expr(lhs, labels, gotos, scope, fv);
                self.check_flow_expr(rhs, labels, gotos, scope, fv);
            }
            CheckedExprKind::Ternary { cond, then, els } => {
                self.check_flow_expr(cond, labels, gotos, scope, fv);
                self.check_flow_expr(then, labels, gotos, scope, fv);
                self.check_flow_expr(els, labels, gotos, scope, fv);
            }
            CheckedExprKind::Elvis { cond, els } => {
                self.check_flow_expr(cond, labels, gotos, scope, fv);
                self.check_flow_expr(els, labels, gotos, scope, fv);
            }
            CheckedExprKind::Call { func, args } => {
                self.check_flow_expr(func, labels, gotos, scope, fv);
                for arg in args {
                    self.check_flow_expr(arg, labels, gotos, scope, fv);
                }
            }
            CheckedExprKind::Atomic { args, .. } => {
                for arg in args {
                    self.check_flow_expr(arg, labels, gotos, scope, fv);
                }
            }
            CheckedExprKind::Overflow { lhs, rhs, res, .. } => {
                self.check_flow_expr(lhs, labels, gotos, scope, fv);
                self.check_flow_expr(rhs, labels, gotos, scope, fv);
                self.check_flow_expr(res, labels, gotos, scope, fv);
            }
            CheckedExprKind::Alloca { size } => {
                self.check_flow_expr(size, labels, gotos, scope, fv);
            }
            CheckedExprKind::FrameAddress { .. } => {}
            CheckedExprKind::Clz { arg, fallback, .. } => {
                self.check_flow_expr(arg, labels, gotos, scope, fv);
                if let Some(fb) = fallback {
                    self.check_flow_expr(fb, labels, gotos, scope, fv);
                }
            }
            CheckedExprKind::FpClass { arg, .. } => {
                self.check_flow_expr(arg, labels, gotos, scope, fv);
            }
            CheckedExprKind::FpClassify { codes, arg } => {
                for code in codes {
                    self.check_flow_expr(code, labels, gotos, scope, fv);
                }
                self.check_flow_expr(arg, labels, gotos, scope, fv);
            }
            CheckedExprKind::Member { base, .. } => {
                self.check_flow_expr(base, labels, gotos, scope, fv);
            }
            CheckedExprKind::Generic { ctrl, assocs, .. } => {
                self.check_flow_expr(ctrl, labels, gotos, scope, fv);
                for (_, expr) in assocs {
                    self.check_flow_expr(expr, labels, gotos, scope, fv);
                }
            }
            CheckedExprKind::CompoundLiteral { init, .. } => {
                self.check_flow_init(init, gotos, scope, fv);
            }
            CheckedExprKind::StmtExpr { items } => {
                // Each statement expression is its own label scope:
                // labels defined inside never leak out (jumping in
                // resolves to nothing and fails as undefined, like
                // GCC), while gotos fall through to the enclosing
                // scope (jumping out is allowed). Declarations inside
                // are visited through their initializers, which parse
                // (and flow-check) like any other.
                let mut inner_labels = Vec::new();
                // Scratch: every goto pushed here is also recorded on
                // `sub` (`record_goto` accompanies each push), so the
                // filtered walk below decides what escapes.
                let mut scratch = Vec::new();
                let mut sub = FlowVla::new();
                for item in items {
                    match item {
                        BlockItem::Decl { ids, .. } => {
                            self.record_vla_decls(ids, 0, &mut sub);
                            for id in ids {
                                self.check_flow_decl_init(*id, &mut scratch, 0, &mut sub);
                            }
                        }
                        BlockItem::Stmt(stmt) => {
                            self.check_flow_stmt(stmt, FlowCtx::NONE, &mut inner_labels, &mut scratch, 0, &mut sub);
                        }
                    }
                }
                // Gotos that resolve to an inner label stop here (their
                // VLA entry checks run as the inner walk ends); the
                // rest escape to the enclosing scope.
                for (name, _, span) in mem::take(&mut sub.gotos) {
                    if let Some((scope, label_seq)) = sub.labels.get(&name).copied() {
                        let seq = sub.bump();
                        self.check_vla_entry(seq, scope, label_seq, span, &sub);
                    } else {
                        gotos.push((name, span));
                        fv.record_goto(name, span);
                    }
                }
            }
        }
    }
    pub fn can_start_declaration(&mut self) -> bool {
        // `__extension__` opens GNU-annotated declarations from system
        // headers (`__extension__ typedef ...`).
        if self.peek_is_gnu("__extension__") {
            return true;
        }
        // `_Alignas`/`alignas` may open a declaration (`_Alignas(16) int x`).
        if self.peek_alignas() {
            return true;
        }
        // Thread-local specifiers open declarations too
        // (`__thread int x;`, `_Thread_local int x;`).
        if self.peek_is_gnu("_Thread_local") || self.peek_is_gnu("__thread") || self.peek_is_gnu("thread_local") {
            return true;
        }
        match self.tokens.peek().map(|located| &located.val) {
            Some(Token::Keyword(
                pat_storage_class_specifier!()
                | pat_basic_type_specifier!()
                | pat_type_qualifier!()
                | pat_struct_or_union!()
                | Keyword::Enum
                | Keyword::Inline
                | Keyword::_Imaginary
                | Keyword::_Atomic
                | Keyword::_Noreturn,
            )) => true,
            // A typedef name may open a declaration (`Arg a;`), as may
            // the alternate `inline` spellings (`__inline__ int f();`
            // and `__forceinline int f();`) and the GNU qualifier
            // spellings (`__const char *`).
            Some(Token::Identifier(sym)) => {
                // `t :` in a block is a label named like a typedef
                // (C17 §6.8.1): the colon of a member bit-field only
                // follows a struct member core declarator (C17
                // §6.7.2.1), so a declaration cannot open at this shape.
                if matches!(self.tokens.peek_nth(1).map(|located| &located.val), Some(Token::Punctuator(Punctuator::Colon))) {
                    return false;
                }
                self.is_typedef_name(*sym)
                    || self.is_gnu_type_keyword(*sym)
                    || self.peek_c23_keyword("typeof_unqual")
                    || self.peek_c23_keyword("bool")
                    || self.peek_alternate_inline()
                    || self.peek_gnu_qualifier().is_some()
                    // MSVC declaration prefixes (`__declspec(noreturn)
                    // void f(...);`, `__cdecl int f(void);`).
                    || self.peek_msvc_operator("__declspec")
                    || self.peek_msvc_operator("_declspec")
                    || ["__cdecl", "__stdcall", "__fastcall", "__thiscall", "__vectorcall", "__clrcall"]
                        .into_iter()
                        .any(|kw| self.peek_is_gnu(kw))
                    // `_Future(T)` opens a declaration like a typedef name does.
                    || self.is_future_specifier(*sym)
            }
            _ => false,
        }
    }
    // Where an identifier could be read either as a typedef name or as
    // the name being declared, C takes it as the typedef
    // (C17 §6.7.6.3p11) — so this decides both whether a declaration
    // can start here and where a parameter-type-list starts instead of
    // an old-style identifier-list.
    fn is_typedef_name(&self, sym: SymbolU32) -> bool {
        self.sema.lookup_symbol(sym).is_some_and(|id| self.sema.decls[id.0 as usize].kind == BindingKind::Typedef)
    }
    /// Whether `sym` spells the `_Future(T)` specifier of the async
    /// extension (see `parse_future_specifier`). It reads like an
    /// identifier, so every "declaration or expression?" decision has
    /// to know about it; the specifier itself reports a missing
    /// `-fglcc-async`, which beats a generic parse error.
    fn is_future_specifier(&self, sym: SymbolU32) -> bool {
        self.intern.resolve(sym) == Some("_Future")
    }
    /// Whether the upcoming token is the identifier `word` — a spelled
    /// keyword whose family the LangOpts gate chooses (`__extension__`,
    /// `__attribute__`, `__int64`, `typeof`, ...). Resolving never
    /// interns, so probing leaves the interner alone.
    fn peek_is_gnu(&self, word: &str) -> bool {
        match self.tokens.peek_val() {
            Some(Token::Identifier(sym)) => self.intern.resolve(*sym) == Some(word) && crate::lang::spelling_family(word).allowed(self.sema.lang),
            _ => false,
        }
    }
    /// A bare C23 keyword. The lexer has no dialect, so spellings that
    /// became keywords in C23 stay identifiers here and are gated at the
    /// parse site.
    fn peek_c23_keyword(&self, word: &str) -> bool {
        match self.tokens.peek_val() {
            Some(Token::Identifier(sym)) => self.is_c23_keyword(*sym, word),
            _ => false,
        }
    }
    /// Whether `sym` spells the C23 keyword `word` under the selected
    /// dialect. Lookahead callers pass a symbol they already hold.
    fn is_c23_keyword(&self, sym: SymbolU32, word: &str) -> bool {
        self.sema.lang.standard_includes(Std::C23) && self.intern.resolve(sym) == Some(word)
    }
    /// A GNU alternate qualifier spelling (`__restrict__`, `__const`,
    /// ...), accepted everywhere the standard spelling is. System
    /// headers use these pervasively (`void *memcpy(void *__restrict,
    /// ...)`). Gated by `-fno-gnu-extensions`.
    fn peek_gnu_qualifier(&self) -> Option<TypeQualifier> {
        if !self.sema.lang.gnu {
            return None;
        }
        match self.tokens.peek_val() {
            Some(Token::Identifier(sym)) => match self.intern.resolve(*sym) {
                Some("__const__") | Some("__const") => Some(TypeQualifier::Const),
                Some("__restrict__") | Some("__restrict") => Some(TypeQualifier::Restrict),
                Some("__volatile__") | Some("__volatile") => Some(TypeQualifier::Volatile),
                _ => None,
            },
            _ => None,
        }
    }
    /// An alternate `inline` spelling: the GNU ones (`__inline__`,
    /// `__inline`, used for `static __inline` definitions in system
    /// headers) and MSVC's `__forceinline`.
    fn peek_alternate_inline(&self) -> bool {
        self.peek_is_gnu("__inline__") || self.peek_is_gnu("__inline") || self.peek_is_gnu("__forceinline")
    }
    /// MSVC's pointer-size attributes, which follow the `*` they modify.
    fn peek_ms_ptr_size(&self) -> Option<PtrSize> {
        if self.peek_is_gnu("__ptr32") {
            Some(PtrSize::Ptr32)
        } else if self.peek_is_gnu("__ptr64") {
            Some(PtrSize::Ptr64)
        } else {
            None
        }
    }
    /// MSVC's `__unaligned` type qualifier (`UNALIGNED` in the Windows
    /// headers). It only promises the object need not be naturally
    /// aligned; our codegen already emits every access unaligned-safe,
    /// so it is accepted in MS mode and erased like `_Atomic`.
    fn peek_ms_unaligned(&self) -> bool {
        self.peek_is_gnu("__unaligned")
    }
    /// GNU builtin type keywords behave as specifiers, never as
    /// old-style identifier-list entries (`f(__float128)` prototypes).
    /// This includes GCC's predefined typedef spellings
    /// (`__uint128_t`, used bare by sqlite with no declaration). Each
    /// spelling gates through its LangOpts family.
    fn is_gnu_type_keyword(&self, sym: SymbolU32) -> bool {
        let word = match self.intern.resolve(sym) {
            Some(word)
                if matches!(
                    word,
                    "__float128"
                        | "__builtin_va_list"
                        | "__extension__"
                        | "__signed__"
                        | "__int128"
                        | "__uint128_t"
                        | "__int128_t"
                        | "__int8"
                        | "__int16"
                        | "__int32"
                        | "__int64"
                        | "_Float32"
                        | "_Float64"
                        | "__complex__"
                        | "__complex"
                        | "typeof"
                        | "__typeof__"
                        | "__typeof"
                ) =>
            {
                word
            }
            _ => return false,
        };
        crate::lang::spelling_family(word).allowed(self.sema.lang)
    }
    /// GNU `typeof ( expression )` / `typeof ( type-name )`: the type of
    /// the operand exactly as written (arrays and functions do not
    /// decay), usable wherever a type specifier is. The contents parse
    /// as a type name when they start like one — a typedef name counts,
    /// mirroring the cast disambiguation — and as an expression checked
    /// here otherwise.
    fn parse_typeof_specifier(&mut self, unqual: bool) -> Ty {
        self.tokens.advance();
        self.skip_gnu_suffixes();
        self.expect_punct(Punctuator::LParen);
        let ty = if self.token_starts_type_name(self.tokens.peek_val()) {
            self.parse_type_name()
        } else {
            let expr = self.parse_expression();
            self.checked_expr(&expr).ty
        };
        self.expect_punct(Punctuator::RParen);
        if unqual { ty.unqualified_owned() } else { ty }
    }

    /// Implicit GCC builtins, none declared anywhere: the `<stdarg.h>`
    /// expansions (`__builtin_va_start`, `__builtin_va_end`), sqlite's
    /// `__sync_synchronize()`, `__builtin_clzll(m)`,
    /// `__builtin_inff()`, `__builtin_nanf(tag)`, and
    /// `__builtin_floor`/`__builtin_ceil(x)`. When a call's callee
    /// names one, declare its prototype on first use so the later
    /// check resolves like any declared function. (`va_arg` takes a
    /// *type* operand, which calls cannot check, and is handled
    /// separately in postfix parsing.)
    fn declare_builtin_if_called(&mut self, func: &Expr) {
        let Expr::Ident { name, .. } = func else {
            return;
        };
        let (ret, params, variadic) = match self.intern.resolve(*name) {
            Some("__builtin_va_start") => (Ty::Void, vec![Ty::BuiltinVaList], true),
            Some("__builtin_va_end") => (Ty::Void, vec![Ty::BuiltinVaList], false),
            Some("__builtin_va_copy") => (Ty::Void, vec![Ty::BuiltinVaList, Ty::BuiltinVaList], false),
            Some("__sync_synchronize") => (Ty::Void, vec![], false),
            Some("__builtin_clz") => (Ty::Int, vec![Ty::UnsignedInt], false),
            Some("__builtin_clzl") => (Ty::Int, vec![Ty::UnsignedLong], false),
            Some("__builtin_clzll") => (Ty::Int, vec![Ty::UnsignedLongLong], false),
            Some("__builtin_ctz") => (Ty::Int, vec![Ty::UnsignedInt], false),
            Some("__builtin_ctzl") => (Ty::Int, vec![Ty::UnsignedLong], false),
            Some("__builtin_ctzll") => (Ty::Int, vec![Ty::UnsignedLongLong], false),
            Some("__builtin_popcount") => (Ty::Int, vec![Ty::UnsignedInt], false),
            Some("__builtin_popcountl") => (Ty::Int, vec![Ty::UnsignedLong], false),
            Some("__builtin_popcountll") => (Ty::Int, vec![Ty::UnsignedLongLong], false),
            Some("__builtin_clrsb") => (Ty::Int, vec![Ty::Int], false),
            Some("__builtin_clrsbl") => (Ty::Int, vec![Ty::Long], false),
            Some("__builtin_clrsbll") => (Ty::Int, vec![Ty::LongLong], false),
            Some("__builtin_parity") => (Ty::Int, vec![Ty::UnsignedInt], false),
            Some("__builtin_parityl") => (Ty::Int, vec![Ty::UnsignedLong], false),
            Some("__builtin_parityll") => (Ty::Int, vec![Ty::UnsignedLongLong], false),
            Some("__builtin_copysign") => (Ty::Double, vec![Ty::Double, Ty::Double], false),
            Some("__builtin_copysignf") => (Ty::Float, vec![Ty::Float, Ty::Float], false),
            Some("__builtin_copysignl") => (Ty::LongDouble, vec![Ty::LongDouble, Ty::LongDouble], false),
            Some("__builtin_inff") => (Ty::Float, vec![], false),
            Some("__builtin_inf") => (Ty::Double, vec![], false),
            Some("__builtin_infl") => (Ty::LongDouble, vec![], false),
            Some("__builtin_huge_valf") => (Ty::Float, vec![], false),
            Some("__builtin_huge_val") => (Ty::Double, vec![], false),
            Some("__builtin_huge_vall") => (Ty::LongDouble, vec![], false),
            Some("__builtin_nanf") => (
                Ty::Float,
                vec![Ty::Pointer {
                    qualifiers: vec![],
                    size: PtrSize::Default,
                    pointee: Box::new(Ty::Char),
                }],
                false,
            ),
            Some("__builtin_nan" | "__builtin_nanl") => (
                if self.intern.resolve(*name) == Some("__builtin_nanl") {
                    Ty::LongDouble
                } else {
                    Ty::Double
                },
                vec![Ty::Pointer {
                    qualifiers: vec![],
                    size: PtrSize::Default,
                    pointee: Box::new(Ty::Char),
                }],
                false,
            ),
            Some("__builtin_floor" | "__builtin_ceil") => (Ty::Double, vec![Ty::Double], false),
            Some("__builtin_floorf" | "__builtin_ceilf") => (Ty::Float, vec![Ty::Float], false),
            Some("__builtin_floorl" | "__builtin_ceill") => (Ty::LongDouble, vec![Ty::LongDouble], false),
            Some("__builtin_abs") => (Ty::Int, vec![Ty::Int], false),
            Some("__builtin_labs") => (Ty::Long, vec![Ty::Long], false),
            Some("__builtin_llabs") => (Ty::LongLong, vec![Ty::LongLong], false),
            Some("__builtin_fabs") => (Ty::Double, vec![Ty::Double], false),
            Some("__builtin_fabsf") => (Ty::Float, vec![Ty::Float], false),
            Some("__builtin_fabsl") => (Ty::LongDouble, vec![Ty::LongDouble], false),
            Some("__builtin_trap") => (Ty::Int, vec![], false),
            Some("__builtin_bswap16") => (Ty::UnsignedShort, vec![Ty::UnsignedShort], false),
            Some("__builtin_bswap32") => (Ty::UnsignedInt, vec![Ty::UnsignedInt], false),
            Some("__builtin_bswap64") => (Ty::UnsignedLongLong, vec![Ty::UnsignedLongLong], false),
            // Complex conjugates: `__builtin_conjf` over
            // `float _Complex`, `__builtin_conj` over
            // `double _Complex`, `__builtin_conjl` over
            // `long double _Complex`.
            Some("__builtin_conjf") => (Ty::Complex(Box::new(Ty::Float)), vec![Ty::Complex(Box::new(Ty::Float))], false),
            Some("__builtin_conj") => (Ty::Complex(Box::new(Ty::Double)), vec![Ty::Complex(Box::new(Ty::Double))], false),
            Some("__builtin_conjl") => (Ty::Complex(Box::new(Ty::LongDouble)), vec![Ty::Complex(Box::new(Ty::LongDouble))], false),
            // Complex accessors (`__builtin_creal/cimag/cabs` and
            // `f`/`l` variants): the real part, the imaginary part,
            // and the magnitude of a complex number.
            Some("__builtin_crealf") => (Ty::Float, vec![Ty::Complex(Box::new(Ty::Float))], false),
            Some("__builtin_creal") => (Ty::Double, vec![Ty::Complex(Box::new(Ty::Double))], false),
            Some("__builtin_creall") => (Ty::LongDouble, vec![Ty::Complex(Box::new(Ty::LongDouble))], false),
            Some("__builtin_cimagf") => (Ty::Float, vec![Ty::Complex(Box::new(Ty::Float))], false),
            Some("__builtin_cimag") => (Ty::Double, vec![Ty::Complex(Box::new(Ty::Double))], false),
            Some("__builtin_cimagl") => (Ty::LongDouble, vec![Ty::Complex(Box::new(Ty::LongDouble))], false),
            Some("__builtin_cabsf") => (Ty::Float, vec![Ty::Complex(Box::new(Ty::Float))], false),
            Some("__builtin_cabs") => (Ty::Double, vec![Ty::Complex(Box::new(Ty::Double))], false),
            Some("__builtin_cabsl") => (Ty::LongDouble, vec![Ty::Complex(Box::new(Ty::LongDouble))], false),
            _ => return,
        };
        if self.sema.lookup_symbol(*name).is_some() {
            return;
        }
        let span = func.span();
        let declared = self.sema.declare_global(
            *name,
            Binding {
                name: Some(*name),
                ty: Ty::Func {
                    ret: Box::new(ret),
                    params,
                    variadic,
                    prototyped: true,
                },
                kind: BindingKind::Func,
                linkage: Linkage::External,
                block_static: false,
                tls: false,
                defined: false,
                tentative: false,
                constructor: None,
                destructor: None,
                weak: false,
                alias: None,
                naked: false,
                noinline: false,
                is_async: false,
                section: None,
                asm_label: None,
                inline: false,
                align: None,
                span,
            },
        );
        self.unwrap_diag(declared);
    }
    /// `__builtin_X` spellings backed by the glcc runtime's assembly
    /// (`rt/glccrt.c`): the call rewrites to the runtime symbol, so
    /// the implicit-declaration path declares it with the real
    /// prototype (`libc_prototype`). The frame-address builtins need
    /// no rewrite — checking folds them inline.
    const RUNTIME_BUILTIN_NAMES: &[(&str, &str)] = &[("__builtin_setjmp", "__glccrt_setjmp"), ("__builtin_longjmp", "__glccrt_longjmp")];
    /// C89 implicit function declaration (§3.3.2.2): a call `f(...)`
    /// with no visible declaration of `f` implicitly declares
    /// `extern int f()` (unprototyped, so arguments undergo the
    /// default promotions) — except known C library names, which get
    /// their real prototype (see `libc_prototype`): declaring `malloc`
    /// as returning `int` would truncate pointers on LP64, where even
    /// GCC's implicit declaration is unusable. Old torture tests call
    /// `abort()`/`exit()` with no header in scope, and the suite
    /// builds with `-Wno-implicit-function-declaration`, so accept
    /// with a warning instead of erroring. Only bare-identifier
    /// callees trigger this (never member access or indirect calls);
    /// non-call uses of undeclared names still error in checking. A
    /// later real declaration merges via the usual composite-type
    /// rules. Compiler builtins (`__builtin_*`, `__atomic_*`) are
    /// excluded: those either have explicit support above (or inline
    /// in checking) or should keep the clear "undeclared" error until
    /// a real lowering exists — an implicit declaration would only
    /// move the failure to a link error.
    fn declare_implicit_function_if_called(&mut self, func: &Expr) {
        let Expr::Ident { name, .. } = func else {
            return;
        };
        if self.sema.lookup_symbol(*name).is_some() {
            return;
        }
        let Some(text) = self.intern.resolve(*name) else {
            return;
        };
        if text.starts_with("__builtin_") || text.starts_with("__atomic_") || text.starts_with("__c11_atomic_") {
            return;
        }
        let span = func.span();
        // Known C library names declare with their real prototype;
        // everything else is C89 `extern int f()`. Runtime entry
        // points are compiler-generated (rewritten builtins), so they
        // declare silently.
        let (ret, params, variadic, prototyped, assumption) = match Self::libc_prototype(text, &self.sema.target.size_type()) {
            Some((ret, params, variadic)) => (ret, params, variadic, true, "C library prototype assumed"),
            None => (Ty::Int, vec![], false, false, "C89 `extern int` assumed"),
        };
        if !text.starts_with("__glccrt_") {
            self.warn(span, format!("implicit declaration of function '{text}'; {assumption}"));
        }
        let declared = self.sema.declare_global(
            *name,
            Binding {
                name: Some(*name),
                ty: Ty::Func {
                    ret: Box::new(ret),
                    params,
                    variadic,
                    prototyped,
                },
                kind: BindingKind::Func,
                linkage: Linkage::External,
                block_static: false,
                tls: false,
                defined: false,
                tentative: false,
                constructor: None,
                destructor: None,
                weak: false,
                alias: None,
                naked: false,
                noinline: false,
                is_async: false,
                section: None,
                asm_label: None,
                inline: false,
                align: None,
                span,
            },
        );
        self.unwrap_diag(declared);
    }
    /// The real C library prototype for a known libc name (target
    /// `size_t`; `FILE *` is opaque `void *`; `va_list` is the
    /// builtin): return type, parameter types, and variadicity — the
    /// string/memory/allocation/format functions plus the standard
    /// `<math.h>` family (`math_prototype`). `None` for anything else
    /// (plain C89 implicit `int`). This table is also the allowlist for
    /// rewriting `__builtin_X` calls to `X` (GCC's builtins for C
    /// library functions have the same contract as the functions).
    /// Parameter pointers stay unqualified, like the existing builtin
    /// table above — callers convert.
    fn libc_prototype(name: &str, size_t: &Ty) -> Option<(Ty, Vec<Ty>, bool)> {
        let cstr = || ptr(Ty::Char);
        let cvoid = || ptr(Ty::Void);
        let (ret, params, variadic) = match name {
            "abort" => (Ty::Void, vec![], false),
            // glcc runtime entry points (see `RUNTIME_BUILTIN_NAMES`):
            // the builtin buffers are plain `void *` so arrays of any
            // object pointer convert without a diagnostic.
            "__glccrt_setjmp" => (Ty::Int, vec![cvoid()], false),
            "__glccrt_longjmp" => (Ty::Void, vec![cvoid(), Ty::Int], false),
            "abs" | "ffs" => (Ty::Int, vec![Ty::Int], false),
            "ffsl" => (Ty::Int, vec![Ty::Long], false),
            "ffsll" => (Ty::Int, vec![Ty::LongLong], false),
            "labs" => (Ty::Long, vec![Ty::Long], false),
            "llabs" => (Ty::LongLong, vec![Ty::LongLong], false),
            "strlen" => (size_t.clone(), vec![cstr()], false),
            "strspn" | "strcspn" => (size_t.clone(), vec![cstr(), cstr()], false),
            "strcpy" | "strcat" | "strstr" | "strpbrk" => (cstr(), vec![cstr(), cstr()], false),
            "strncpy" | "strncat" => (cstr(), vec![cstr(), cstr(), size_t.clone()], false),
            "strcmp" | "strncmp" => (Ty::Int, vec![cstr(), cstr()], false),
            "strchr" | "strrchr" => (cstr(), vec![cstr(), Ty::Int], false),
            "memchr" => (cvoid(), vec![cvoid(), Ty::Int, size_t.clone()], false),
            "memset" => (cvoid(), vec![cvoid(), Ty::Int, size_t.clone()], false),
            "memcpy" | "memmove" => (cvoid(), vec![cvoid(), cvoid(), size_t.clone()], false),
            "memcmp" => (Ty::Int, vec![cvoid(), cvoid(), size_t.clone()], false),
            "malloc" => (cvoid(), vec![size_t.clone()], false),
            "calloc" => (cvoid(), vec![size_t.clone(), size_t.clone()], false),
            "realloc" => (cvoid(), vec![cvoid(), size_t.clone()], false),
            "free" => (Ty::Void, vec![cvoid()], false),
            "printf" | "vprintf" => (Ty::Int, vec![cstr()], true),
            "fprintf" | "vfprintf" => (Ty::Int, vec![cvoid(), cstr()], true),
            "sprintf" | "vsprintf" => (Ty::Int, vec![cstr(), cstr()], true),
            "snprintf" | "vsnprintf" => (Ty::Int, vec![cstr(), size_t.clone(), cstr()], true),
            "puts" => (Ty::Int, vec![cstr()], false),
            "fputs" => (Ty::Int, vec![cstr(), cvoid()], false),
            "putchar" => (Ty::Int, vec![Ty::Int], false),
            _ => Self::math_prototype(name)?,
        };
        // `v*` forms take a `va_list`, not `...`.
        let (params, variadic) = match name {
            "vprintf" => (vec![cstr(), Ty::BuiltinVaList], false),
            "vfprintf" => (vec![cvoid(), cstr(), Ty::BuiltinVaList], false),
            "vsprintf" => (vec![cstr(), cstr(), Ty::BuiltinVaList], false),
            "vsnprintf" => (vec![cstr(), size_t.clone(), cstr(), Ty::BuiltinVaList], false),
            _ => (params, variadic),
        };
        Some((ret, params, variadic))
    }
    /// Prototype of a standard `<math.h>` function: an `f` / `l` suffix
    /// selects the `float` / `long double` form (all listed functions
    /// have both), the bare name is `double`. Most are `T(T)`; the
    /// shapes that differ — extra pointer or integer parameters,
    /// integer results, `sincos` — are listed explicitly. `lgamma_r`'s
    /// float/long double forms put the width before the `_r`. `fabs`,
    /// `floor` and `ceil` calls expand to MIR instructions instead of
    /// calls, but the prototypes are the same.
    fn math_prototype(name: &str) -> Option<(Ty, Vec<Ty>, bool)> {
        /// Base names that have the usual `f`/`l` width forms.
        const BASES: &[&str] = &[
            "acos",
            "acosh",
            "asin",
            "asinh",
            "atan",
            "atan2",
            "atanh",
            "cbrt",
            "ceil",
            "copysign",
            "cos",
            "cosh",
            "drem",
            "erf",
            "erfc",
            "exp",
            "exp2",
            "exp10",
            "expm1",
            "fabs",
            "fdim",
            "floor",
            "fma",
            "fmax",
            "fmin",
            "fmod",
            "frexp",
            "gamma",
            "hypot",
            "ilogb",
            "j0",
            "j1",
            "jn",
            "ldexp",
            "lgamma",
            "llrint",
            "llround",
            "log",
            "log10",
            "log1p",
            "log2",
            "logb",
            "lrint",
            "lround",
            "modf",
            "nearbyint",
            "nextafter",
            "nexttoward",
            "pow",
            "pow10",
            "remainder",
            "remquo",
            "rint",
            "round",
            "roundeven",
            "scalb",
            "scalbln",
            "scalbn",
            "significand",
            "sin",
            "sincos",
            "sinh",
            "sqrt",
            "tan",
            "tanh",
            "tgamma",
            "trunc",
            "y0",
            "y1",
            "yn",
        ];
        for (stem, ty) in [("lgamma_r", Ty::Double), ("lgammaf_r", Ty::Float), ("lgammal_r", Ty::LongDouble)] {
            if name == stem {
                return Some((ty.clone(), vec![ty, ptr(Ty::Int)], false));
            }
        }
        let (base, ty) = match name.strip_suffix('f') {
            Some(base) if BASES.contains(&base) => (base, Ty::Float),
            _ => match name.strip_suffix('l') {
                Some(base) if BASES.contains(&base) => (base, Ty::LongDouble),
                _ => (name, Ty::Double),
            },
        };
        if !BASES.contains(&base) {
            return None;
        }
        let one = |t: Ty| (t.clone(), vec![t], false);
        Some(match base {
            "atan2" | "copysign" | "drem" | "fdim" | "fmax" | "fmin" | "fmod" | "hypot" | "nextafter" | "pow" | "remainder" | "scalb" => (ty.clone(), vec![ty.clone(), ty], false),
            "fma" => (ty.clone(), vec![ty.clone(), ty.clone(), ty], false),
            "frexp" => (ty.clone(), vec![ty.clone(), ptr(Ty::Int)], false),
            "modf" => (ty.clone(), vec![ty.clone(), ptr(ty.clone())], false),
            "remquo" => (ty.clone(), vec![ty.clone(), ty.clone(), ptr(Ty::Int)], false),
            "ldexp" | "scalbn" => (ty.clone(), vec![ty.clone(), Ty::Int], false),
            "scalbln" => (ty.clone(), vec![ty.clone(), Ty::Long], false),
            "nexttoward" => (ty.clone(), vec![ty.clone(), Ty::LongDouble], false),
            "ilogb" => (Ty::Int, vec![ty], false),
            "lrint" | "lround" => (Ty::Long, vec![ty], false),
            "llrint" | "llround" => (Ty::LongLong, vec![ty], false),
            "sincos" => (Ty::Void, vec![ty.clone(), ptr(ty.clone()), ptr(ty)], false),
            "jn" | "yn" => (ty.clone(), vec![Ty::Int, ty], false),
            _ => one(ty),
        })
    }
    // Parse one declaration, returning every id it introduces: a
    // declarator list flattens into the same result as the equivalent
    // discrete declarations — `int a, b;` ≡ `int a; int b;`. A
    // specifier-only tag declaration (`struct S;`, `struct S { ... };`)
    // returns an empty list; the tag lives in `Sema`, not here.
    //
    // File scope (no pushed scopes) follows C17 §6.9.2: `int x;` and
    // `static int x;` are tentative, `extern int x;` a pure
    // declaration, `int x = 1;` (even `extern int x = 1;`) a
    // definition. Block scope has no tentative definitions: `int x;`
    // defines. Initializers parse into `Sema::inits`. Function bodies
    // never appear here — `int f() {` panics, use
    // `parse_external_declaration`.
    pub fn parse_declaration(&mut self) -> Vec<DeclId> {
        let decl_start = self.mark();
        // Prefix annotations (see `parse_external_declaration`).
        self.skip_gnu_suffixes();
        let specifiers = self.parse_declaration_specifiers();
        if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Semi))) {
            self.tokens.advance();
            // No declarators: only a tag may have been declared —
            // `struct S;` or `struct S { ... };`. The tag itself was
            // already handled while parsing the specifiers; any other
            // specifier-only declaration declares nothing (C17 §6.7).
            // An enum specifier always declares its enumerators (even
            // anonymous: `enum { A };`), so it is allowed too.
            let span = self.span_since(decl_start);
            match specifiers.type_specifier.as_slice() {
                [TypeSpecifier::Struct(id)] => {
                    if self.sema.structs.get(id.0 as usize).map_or(true, |s| s.tag.is_none()) {
                        return self.fail(span, "declaration does not declare anything");
                    }
                }
                [TypeSpecifier::Enum(_)] => {}
                _ => {
                    return self.fail(span, "declaration does not declare anything");
                }
            }
            return vec![];
        }
        let file_scope = self.sema.is_file_scope();
        if file_scope {
            self.check_specifiers_file_scope(&specifiers);
        } else {
            self.check_specifiers_block_scope(&specifiers);
        }
        // Declarator-list: the first declarator is parsed here so the
        // specifier-only `;` case above can be told apart; the shared
        // helper handles the rest exactly as `parse_external_declaration`
        // does for non-definitions.
        let first = self.parse_declarator();
        let ids = self.parse_init_declarator_list(&specifiers, first, GnuAttributes::default());
        self.emit_warnings();
        ids
    }
    // Init-declarator-list (`int a = 1, *b, f(void);`): `first` arrives
    // already parsed — every caller needs it for lookahead
    // (`parse_external_declaration` for the definition check,
    // `parse_declaration` for the specifier-only `;` check) — and the
    // rest follow after commas. Every declarator repeats the same
    // specifiers and is declared as if on its own (`int a, *b;` ≡
    // `int a; int *b;`), returning one `DeclId` per declarator.
    // Each declarator declares *before* its initializer parses, so
    // the name scopes over its own initializer (C17 §6.2.1p4) and
    // over later declarators' (`int a = 1, b = a;` sees `a`).
    // Initializers parse into `Sema::inits`.
    pub(crate) fn parse_init_declarator_list(&mut self, specifiers: &DeclarationSpecifiers, first: Declarator, first_suffix: GnuAttributes) -> Vec<DeclId> {
        let mut ids = Vec::new();
        let mut next = Some((first, first_suffix));
        let mut is_first = true;
        loop {
            if self.cancelled() {
                break;
            }
            let (declarator, mut suffix) = next.take().unwrap_or_else(|| (self.parse_declarator(), GnuAttributes::default()));
            // Suffix attributes (`__attribute__((constructor))` on a
            // function declarator) ride into the binding below. The
            // first declarator's were already consumed by the caller
            // (which passes them in); the capture here is a no-op for
            // it and live for the rest.
            suffix.merge(self.parse_gnu_annotations());
            let ty = self.spec_to_ty(specifiers, &declarator);
            // A `{` after a function declarator opens a definition, which
            // takes exactly one declarator (C17 §6.9.1p2). Leading
            // definitions are dispatched by the caller, so one reaching
            // here is misplaced.
            if matches!(ty, Ty::Func { .. }) && matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LBrace))) {
                let span = self.peek_span();
                if is_first {
                    return self.fail(span, "function definition in declaration");
                }
                return self.fail(span, "function definition with multiple declarators");
            }
            let has_init = matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Assign)));
            if has_init && declarator.name().is_none() {
                let span = declarator.span();
                return self.fail(span, "declaration does not declare anything");
            }
            let id = self.declare_one(specifiers, &declarator, ty, has_init, &suffix);
            if self.failed() {
                if has_init {
                    let _ = self.parse_initializer();
                }
                break;
            }
            if has_init {
                let init = self.parse_initializer();
                let ty = self.sema.decls[id.0 as usize].ty.clone();
                let checked = self.checked_initializer(&ty, &init);
                // An array of unknown size takes its size from the
                // initializer (see `complete_array_bound`).
                self.complete_array_bound(id, &checked);
                // The checked initializer (implicit conversions
                // inserted) is what later phases consume.
                self.sema.inits.insert(id, checked);
                // Control flow inside the initializer (only statement
                // expressions can hold any): labels stay local, while
                // gotos resolve with the enclosing body — except at
                // file scope, where statement expressions cannot
                // appear, so there is nothing to visit. (The clone
                // dodges a borrow of `self` against the map below.)
                if self.function_return.is_some() {
                    if let Some(init) = self.sema.inits.get(&id).cloned() {
                        let mut pending = mem::take(&mut self.pending_gotos);
                        self.check_flow_init(&init, &mut pending, 0, &mut FlowVla::new());
                        self.pending_gotos = pending;
                    }
                }
            }
            ids.push(id);
            if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Comma))) {
                self.tokens.advance();
                is_first = false;
                continue;
            }
            break;
        }
        // No body allowed for plain declarations.
        if !self.cancelled() {
            if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LBrace))) {
                let span = self.peek_span();
                return self.fail(span, "expected `;`");
            }
            self.expect_punct(Punctuator::Semi);
        }
        ids
    }
    /// Complete an array of unknown size from its initializer (C17
    /// §6.7.9p22): `int a[] = {1, 2}` becomes `int a[2]`, and
    /// `char s[] = "ab"` becomes `char s[3]` (NUL included). Only the
    /// outermost dimension completes, and only when every designator
    /// folds (otherwise the size is unknowable and the type stays
    /// incomplete). This is what lets dwm's static assertion
    /// (`char x[(sizeof(tags)/sizeof(tags[0])) > 31 ? -1 : 1];`)
    /// fold: `tags` is `const char *tags[]` with an initializer.
    fn complete_array_bound(&mut self, id: DeclId, init: &Initializer) {
        let ty = self.sema.decls[id.0 as usize].ty.clone();
        let Ty::Array { elem, quals, bound } = ty else {
            return;
        };
        if bound.is_some() {
            return;
        }
        let count: u64 = match init {
            Initializer::Expr(expr) => match &expr.kind {
                CheckedExprKind::Str(parts) => match crate::eval::string_length(parts, &self.intern) {
                    Some(len) => len.saturating_add(1),
                    None => return,
                },
                _ => return,
            },
            Initializer::List { items, .. } => {
                // Mirror `check_array_init`'s tracking: bare items fill
                // in order, `[i]` selects; anything untrackable leaves
                // the type incomplete.
                let mut next: u64 = 0;
                let mut max: u64 = 0;
                for item in items {
                    let mut pos = next;
                    let mut designated = false;
                    for designator in &item.designators {
                        match designator {
                            Designator::Index { index, .. } if !designated => {
                                designated = true;
                                match crate::eval::eval_const(index, self.sema, &self.intern) {
                                    Some(crate::eval::Value::Int(n)) if n >= 0 => {
                                        pos = u64::try_from(n).unwrap_or(u64::MAX);
                                    }
                                    _ => return,
                                }
                            }
                            Designator::Index { .. } => {}
                            Designator::Field { .. } | Designator::AnonField { .. } => return,
                        }
                    }
                    max = max.max(pos.saturating_add(1));
                    next = pos.saturating_add(1);
                }
                max
            }
        };
        self.sema.decls[id.0 as usize].ty = Ty::Array {
            elem,
            quals,
            bound: Some(ArrayBound::Fixed(count)),
        };
    }
    /// Declare one `specifiers` + `declarator` pair from a declaration
    /// (never a function definition — those go through
    /// `parse_function_definition`): validates the combination (`void`
    /// objects, `inline` on objects, function initializers, old-style
    /// declarators, ...), computes linkage (C17 §6.2.2), and merges via
    /// `Sema::declare`. `has_init` (`= ...` follows) makes an object a
    /// definition; file-scope `extern`-less declarations without one
    /// are tentative. The initializer itself parses (and checks)
    /// afterwards, in `parse_init_declarator_list`, with the name
    /// already in scope.
    fn declare_one(&mut self, specifiers: &DeclarationSpecifiers, declarator: &Declarator, ty: Ty, has_init: bool, suffix: &GnuAttributes) -> DeclId {
        let span = specifiers.span().merge(declarator.span());
        let Some(symbol) = declarator.name() else {
            return self.fail(span, "declaration does not declare anything");
        };
        let kind = if specifiers.is_typedef() {
            BindingKind::Typedef
        } else if matches!(ty, Ty::Func { .. }) {
            BindingKind::Func
        } else {
            BindingKind::Var
        };
        self.check_function_specifiers(specifiers, &ty);
        if kind == BindingKind::Typedef && has_init {
            return self.fail(span, "`typedef` with initializer");
        }
        // A `typedef` is a declaration like any other: variably
        // modified at file scope it violates the same rule (C17
        // §6.7.6.2p2), while block-scope VM typedefs are legal.
        if kind == BindingKind::Typedef && self.sema.is_file_scope() && self.ty_is_variably_modified(&ty) {
            return self.fail(span, "variably modified type at file scope");
        }
        if kind == BindingKind::Func && has_init {
            return self.fail(span, "function with initializer");
        }
        // An identifier-list (`f(a, b)`) is only allowed in a function
        // *definition* (clang: "a parameter list without types is only
        // allowed in a function definition"), which never reaches here.
        if kind == BindingKind::Func && self.declarator_is_old_style(declarator) {
            return self.fail(span, "parameter list without types is only allowed in a function definition");
        }
        if kind == BindingKind::Var && ty.unqualified() == &Ty::Void {
            return self.fail(span, "declaration of `void` object");
        }
        // Constructor/destructor attributes only mean anything on
        // functions (like GCC, warn and ignore elsewhere).
        if kind != BindingKind::Func && (suffix.constructor.is_some() || suffix.destructor.is_some()) {
            self.warn(span, "constructor/destructor attribute only applies to functions");
        }
        let file_scope = self.sema.is_file_scope();
        let linkage = self.compute_linkage(symbol, specifiers, kind);
        // Thread-local storage validates like a storage class: only
        // objects, never functions/typedefs or `auto`/`register`.
        if specifiers.thread_local {
            if kind != BindingKind::Var {
                return self.fail(span, "thread-local storage on non-object");
            }
            if specifiers.has_storage(StorageClass::Auto) || specifiers.has_storage(StorageClass::Register) || specifiers.is_typedef() {
                return self.fail(span, "thread-local with `auto`/`register`/`typedef`");
            }
        }
        // A block-scope `static` object has static storage duration but
        // still no linkage, so `linkage` alone cannot tell it from an
        // automatic local; record the distinction for the backend. A
        // block-scope thread-local without `extern` likewise has
        // static (thread) storage even without `static`.
        let block_static = kind == BindingKind::Var && !file_scope && (specifiers.has_storage(StorageClass::Static) || (specifiers.thread_local && !specifiers.is_extern()));
        // A variably modified object must have no linkage: an `extern`
        // VLA names a file-scope object that can never be variably
        // modified (GCC: "object with variably modified type must have
        // no linkage"). Of those with no linkage, only one whose own
        // type is a VLA also needs automatic storage (C17 §6.7.6.2p2):
        // a `static int (*p)[n]` points to a VLA but is a fixed-size
        // pointer, which GCC accepts.
        if kind == BindingKind::Var && self.ty_is_variably_modified(&ty) {
            if linkage != Linkage::NoLinkage {
                return self.fail(span, "variably modified object with linkage");
            }
            if (block_static || specifiers.thread_local) && self.sema.is_vla_type(&ty, &self.intern) {
                return self.fail(span, "variably modified object with static storage duration");
            }
        }
        let (defined, tentative) = match kind {
            BindingKind::Typedef => (false, false),
            BindingKind::Func => (false, false),
            // Enumerators never reach here: they declare directly in
            // `parse_enumerator_list`, already complete.
            BindingKind::EnumConst { .. } => (true, false),
            BindingKind::Var => {
                if has_init {
                    (true, false)
                } else if file_scope {
                    if specifiers.is_extern() { (false, false) } else { (false, true) }
                } else {
                    // Block scope: no tentative definitions; even
                    // `static int x;` without `=` defines (zero-init).
                    // `extern` stays a pure declaration.
                    if specifiers.is_extern() { (false, false) } else { (true, false) }
                }
            }
        };
        // `__attribute__((alias("name")))`: the declaration denotes the
        // named entity (GCC requires it declared in this TU), so resolve
        // the spelling to its declaration. The target's definition may
        // come later; the merge in `declare` keeps one `DeclId` for
        // both spellings.
        let alias = match suffix.alias {
            None => None,
            Some(target) => {
                let Some(target_id) = self.sema.lookup_symbol(target) else {
                    let name = self.intern.resolve(symbol).unwrap_or("?").to_string();
                    let target_name = self.intern.resolve(target).unwrap_or("?").to_string();
                    return self.fail(span, format!("'{name}' aliased to undefined symbol '{target_name}'"));
                };
                let target_binding = &self.sema.decls[target_id.0 as usize];
                if target_binding.kind != kind {
                    return self.fail(span, "alias target is a different kind of entity");
                }
                if !crate::sema::types_compatible(&target_binding.ty, &ty) {
                    return self.fail(span, "alias target has an incompatible type");
                }
                Some(target_id)
            }
        };
        if alias.is_some() && has_init {
            return self.fail(span, "alias attribute on a definition with an initializer");
        }
        let naked = kind == BindingKind::Func && (suffix.naked || specifiers.attrs.naked);
        let is_async = kind == BindingKind::Func && (suffix.is_async || specifiers.attrs.is_async);
        if kind != BindingKind::Func && (suffix.is_async || specifiers.attrs.is_async) {
            self.warn(span, "attribute `async` only applies to functions");
        }
        if is_async {
            if matches!(&ty, Ty::Func { variadic: true, .. }) {
                return self.fail(span, "async function cannot be variadic");
            }
            // The handle type must exist before any call site checks, so
            // a call typed `_Future(ret)` always finds this struct.
            if let Ty::Func { ret, .. } = &ty {
                self.sema.future_handle((**ret).clone());
            }
        }
        let section = suffix.section.or(specifiers.attrs.section);
        if section.is_some() && kind == BindingKind::Func && !naked {
            self.warn(span, "attribute `section` on a function is ignored");
        }
        let declared = self.sema.declare(
            symbol,
            Binding {
                name: Some(symbol),
                ty,
                kind,
                linkage,
                block_static,
                tls: specifiers.thread_local && kind == BindingKind::Var,
                defined,
                tentative,
                // Constructor/destructor ride the binding so a later
                // definition merges them (first wins in `declare`):
                // the extern prototype usually carries the attribute.
                constructor: if kind == BindingKind::Func { suffix.constructor } else { None },
                destructor: if kind == BindingKind::Func { suffix.destructor } else { None },
                weak: suffix.weak || specifiers.attrs.weak,
                alias,
                naked,
                is_async,
                inline: kind == BindingKind::Func && specifiers.is_inline(),
                noinline: kind == BindingKind::Func && (suffix.noinline || specifiers.attrs.noinline),
                align: suffix.aligned.or(specifiers.attrs.aligned),
                section,
                asm_label: suffix.asm_label.or(specifiers.attrs.asm_label),
                span,
            },
        );
        let id = self.unwrap_diag(declared);
        id
    }
    /// Linkage for one declaration (C17 §6.2.2): `typedef` never has
    /// any; `static` at file scope is internal (a block-scope `static`
    /// object is local, but a `static` function keeps internal
    /// linkage); `extern` — and a function declarator without storage,
    /// which acts as `extern` — inherits visible linkage or is
    /// external; otherwise file-scope entities are external and
    /// block-scope objects local.
    fn compute_linkage(&self, symbol: SymbolU32, specifiers: &DeclarationSpecifiers, kind: BindingKind) -> Linkage {
        if kind == BindingKind::Typedef {
            return Linkage::NoLinkage;
        }
        let file_scope = self.sema.is_file_scope();
        if specifiers.has_storage(StorageClass::Static) {
            if file_scope || kind == BindingKind::Func {
                return Linkage::Internal;
            }
            return Linkage::NoLinkage;
        }
        if specifiers.is_extern() || (kind == BindingKind::Func && !specifiers.has_storage_class() && !specifiers.is_typedef()) {
            if let Some(prior) = self.sema.visible_linkage(symbol) {
                return prior;
            }
            return Linkage::External;
        }
        if specifiers.has_storage(StorageClass::Auto) || specifiers.has_storage(StorageClass::Register) {
            return Linkage::NoLinkage;
        }
        if file_scope { Linkage::External } else { Linkage::NoLinkage }
    }
    /// At most one storage-class specifier (C17 §6.7.1p2), and no
    /// `auto`/`register` at file scope (C17 §6.9p2).
    fn check_specifiers_file_scope(&self, specifiers: &DeclarationSpecifiers) {
        if specifiers.class.len() > 1 {
            return self.fail(specifiers.span, "multiple storage class specifiers");
        }
        if specifiers.has_storage(StorageClass::Auto) || specifiers.has_storage(StorageClass::Register) {
            return self.fail(specifiers.span, "`auto`/`register` at file scope");
        }
    }
    fn check_specifiers_block_scope(&self, specifiers: &DeclarationSpecifiers) {
        if specifiers.class.len() > 1 {
            return self.fail(specifiers.span, "multiple storage class specifiers");
        }
    }
    /// `inline` and `_Noreturn` only declare functions (C17 §6.7.4p2).
    fn check_function_specifiers(&self, specifiers: &DeclarationSpecifiers, ty: &Ty) {
        if matches!(ty, Ty::Func { .. }) {
            return;
        }
        if specifiers.is_inline() {
            return self.fail(specifiers.span, "`inline` on non-function declaration");
        }
        if specifiers.is_noreturn() {
            return self.fail(specifiers.span, "`_Noreturn` on non-function declaration");
        }
    }
    /// Whether a function declarator is followed by a body: `{`
    /// directly (new-style), or — for an old-style identifier-list —
    /// a declaration-list then `{` (detected as "a declaration can
    /// start here"). Anything else (`;`, `,`, `=`) is a prototype
    /// declaration; the list itself is parsed by the caller.
    fn is_function_definition_ahead(&mut self, declarator: &Declarator) -> bool {
        if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LBrace))) {
            return true;
        }
        if !self.declarator_is_old_style(declarator) {
            return false;
        }
        self.can_start_declaration()
    }
    /// Whether the declarator's outermost function suffix is an
    /// old-style identifier-list (`f(a, b)`).
    fn declarator_is_old_style(&self, declarator: &Declarator) -> bool {
        let mut d = declarator;
        loop {
            for suf in d.direct.suffixes.iter() {
                if let DirectSuffix::Function(f) = suf {
                    if f.is_old_style {
                        return true;
                    }
                }
            }
            match &d.direct.base {
                DirectBase::Paren(inner) => d = inner,
                _ => return false,
            }
        }
    }
    /// An initializer with the `=` already peeked (not consumed): either
    /// a single assignment expression or a brace-enclosed list (C17
    /// §6.7.9). Leaves the trailing `,`/`;`/`}` unconsumed.
    fn parse_initializer(&mut self) -> Initializer {
        match self.tokens.advance().map(|x| x.val) {
            Some(Token::Punctuator(Punctuator::Assign)) => {}
            other => {
                let span = self.peek_span();
                return self.fail(span, format!("expected `=`, got {other:?}"));
            }
        }
        if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LBrace))) {
            self.parse_brace_list()
        } else {
            let expr = self.parse_assignment_expression();
            Initializer::Expr(self.checked_expr(&expr))
        }
    }

    /// `{ initializer-list [,] }` with the `{` unconsumed: items are
    /// `[designation] initializer` separated by `,`, and a trailing
    /// comma before `}` is allowed.
    fn parse_brace_list(&mut self) -> Initializer {
        let start = self.mark();
        self.expect_punct(Punctuator::LBrace);
        let mut items = Vec::new();
        loop {
            if self.cancelled() {
                break;
            }
            match self.tokens.peek_val() {
                Some(Token::Punctuator(Punctuator::RBrace)) => break,
                None => {
                    let span = self.span_since(start);
                    return self.fail(span, "unterminated initializer");
                }
                _ => items.extend(self.parse_init_item()),
            }
            match self.tokens.peek_val() {
                Some(Token::Punctuator(Punctuator::Comma)) => {
                    self.tokens.advance();
                    continue;
                }
                Some(Token::Punctuator(Punctuator::RBrace)) => break,
                None => {
                    let span = self.span_since(start);
                    return self.fail(span, "unterminated initializer");
                }
                _ => {
                    let span = self.peek_span();
                    return self.fail(span, "expected `,` or `}` in initializer");
                }
            }
        }
        if !self.cancelled() {
            self.expect_punct(Punctuator::RBrace);
        }
        Initializer::List {
            items,
            span: self.span_since(start),
        }
    }

    /// One brace-list element: any number of `.field`/`[index]`
    /// designators (or GCC's obsolete `field:` spelling), `=` when
    /// designators are present, then the value — which may itself be a
    /// brace list (`.a = {1, 2}` or bare `{1}`). A GNU range
    /// designator (`[1 ... 5] = x`) yields one item per position.
    fn parse_init_item(&mut self) -> Vec<InitItem> {
        let start = self.mark();
        let mut designators = Vec::new();
        let mut range: Option<(u64, u64, Span)> = None;
        loop {
            match self.tokens.peek_val() {
                Some(Token::Punctuator(Punctuator::Dot)) => {
                    let dot_start = self.mark();
                    self.tokens.advance();
                    let name = match self.tokens.advance().map(|located| located.val) {
                        Some(Token::Identifier(name)) => name,
                        other => {
                            let span = self.peek_span();
                            return self.fail(span, format!("expected field name, got {other:?}"));
                        }
                    };
                    designators.push(Designator::Field {
                        name,
                        span: self.span_since(dot_start),
                    });
                }
                // Obsolete GNU designator (`{ field: value }`): a bare
                // identifier immediately followed by `:`.
                Some(Token::Identifier(name)) if matches!(self.tokens.peek_nth(1).map(|located| &located.val), Some(Token::Punctuator(Punctuator::Colon))) => {
                    let name_start = self.mark();
                    let name = *name;
                    self.tokens.advance();
                    designators.push(Designator::Field {
                        name,
                        span: self.span_since(name_start),
                    });
                    // The `:` stands in for the `=` below.
                    self.tokens.advance();
                    let init = if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LBrace))) {
                        self.parse_brace_list()
                    } else {
                        let expr = self.parse_assignment_expression();
                        Initializer::Expr(self.checked_expr(&expr))
                    };
                    return vec![InitItem {
                        designators,
                        init: Box::new(init),
                        span: self.span_since(start),
                    }];
                }
                Some(Token::Punctuator(Punctuator::LBracket)) => {
                    let open = self.mark();
                    self.tokens.advance();
                    let index = self.parse_assignment_expression();
                    let index = self.checked_expr(&index);
                    if !is_integer(&index.ty) {
                        return self.fail(index.span, "array designator must have integer type");
                    }
                    // GNU range designator `[first ... last] = value`:
                    // it ends the designation, and the item expands to
                    // one per position once the value is parsed.
                    if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Ellipsis))) {
                        self.tokens.advance();
                        let last = self.parse_assignment_expression();
                        let last = self.checked_expr(&last);
                        if !is_integer(&last.ty) {
                            return self.fail(last.span, "array designator must have integer type");
                        }
                        let span = self.span_since(open);
                        self.expect_punct(Punctuator::RBracket);
                        let first = self.fold_designator(&index);
                        let end = self.fold_designator(&last);
                        if end < first {
                            return self.fail(span, "empty range designator");
                        }
                        range = Some((first, end, span));
                        break;
                    }
                    self.expect_punct(Punctuator::RBracket);
                    designators.push(Designator::Index {
                        index,
                        span: self.span_since(open),
                    });
                }
                _ => break,
            }
        }
        if !designators.is_empty() || range.is_some() {
            self.expect_punct(Punctuator::Assign);
        }
        let init = if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LBrace))) {
            self.parse_brace_list()
        } else {
            let expr = self.parse_assignment_expression();
            Initializer::Expr(self.checked_expr(&expr))
        };
        let item = InitItem {
            designators,
            init: Box::new(init),
            span: self.span_since(start),
        };
        let Some((first, last, span)) = range else {
            return vec![item];
        };
        // GNU range designator: the same value for every position, each
        // as a plain constant index the backend folds.
        let mut items = Vec::with_capacity((last - first + 1).try_into().unwrap_or(0));
        for pos in first..=last {
            let spelling = self.intern.get_or_intern(pos.to_string());
            let mut it = item.clone();
            it.designators.push(Designator::Index {
                index: CheckedExpr {
                    ty: Ty::Int,
                    span,
                    kind: CheckedExprKind::Int(IntLiteral {
                        spelling,
                        radix: Radix::Dec,
                        suffix: IntSuffix::None,
                        imag: false,
                    }),
                },
                span,
            });
            items.push(it);
        }
        items
    }

    /// Fold a range designator bound: GNU requires an integer constant.
    fn fold_designator(&mut self, expr: &CheckedExpr) -> u64 {
        match crate::eval::eval_const(expr, self.sema, &self.intern) {
            Some(crate::eval::Value::Int(n)) if n >= 0 => u64::try_from(n).unwrap_or(u64::MAX),
            _ => self.fail(expr.span, "range designator bounds must be constant"),
        }
    }
    pub fn parse_declarator(&mut self) -> Declarator {
        let start = self.mark();
        // Annotations between the specifiers and the declarator
        // (`static int __attribute__((noinline)) f(void);` in sqlite):
        // every declarator position funnels through here.
        self.skip_gnu_suffixes();
        let pointer = self.parse_pointer();
        let direct = self.parse_direct_declarator();
        let span = self.span_since(start);
        Declarator { pointer, direct, span }
    }
    pub fn parse_pointer(&mut self) -> Vec<Pointer> {
        let mut pointer = Vec::new();
        loop {
            // Attributes may sit between pointer levels or between the
            // pointer list and the direct declarator
            // (`int * __attribute__((noinline)) f(void);`).
            self.skip_gnu_suffixes();
            match self.tokens.peek().map(|x| &x.val) {
                Some(Token::Punctuator(Punctuator::Star)) => {
                    let start = self.mark();
                    self.tokens.advance();
                    let mut qualifiers = Vec::new();
                    let mut size = PtrSize::Default;
                    loop {
                        if let Some(s) = self.peek_ms_ptr_size() {
                            size = s;
                            self.tokens.advance();
                            continue;
                        }
                        let before = qualifiers.len();
                        qualifiers.extend(self.parse_type_qualifier_list());
                        if qualifiers.len() == before {
                            break;
                        }
                    }
                    // On a 64-bit target `__ptr64` names the ordinary
                    // pointer type (MSVC and clang both treat it so),
                    // so normalize it away instead of minting a
                    // distinct-but-identical type.
                    if size == PtrSize::Ptr64 && self.sema.target.pointer_size == 8 {
                        size = PtrSize::Default;
                    }
                    let span = self.span_since(start);
                    pointer.push(Pointer { qualifiers, size, span });
                }
                _ => break,
            }
        }
        pointer
    }
    /// Whether `(` at the cursor opens a parameter list rather than a
    /// grouped declarator: the contents to the matching `)` hold a `,`
    /// or `...` at depth 0, are empty, or contain a declaration
    /// specifier at depth 0 (a type keyword, `struct`/`union`/`enum`,
    /// qualifier, storage class, `inline`, a typedef name, or a GNU
    /// type keyword). A grouped declarator holds no specifiers, so
    /// `(*fp)` and `(x)` keep grouping — while `(void)`, `(int)`,
    /// `(int, char)`, `(T)`, and `(int *)` can only be parameter lists
    /// (verified against gcc: abstract function declarators in
    /// type-names and parameters).
    fn paren_is_parameter_list(&self) -> bool {
        let mut depth = 0u32;
        let mut n = 0usize;
        let mut content = 0u32;
        let mut comma_or_ellipsis = false;
        let mut first_is_specifier = false;
        loop {
            let tok = self.tokens.peek_nth(n).map(|t| &t.val);
            match tok {
                None => return false, // unterminated: grouping reports it
                Some(Token::Punctuator(Punctuator::LParen)) => {
                    // A nested `(` is grouped content, so `((*f)())`
                    // stays a parenthesized declarator rather than
                    // reading as an empty parameter list.
                    if depth == 1 {
                        content += 1;
                    }
                    depth += 1;
                }
                Some(Token::Punctuator(Punctuator::RParen)) => {
                    depth -= 1;
                    if depth == 0 {
                        return comma_or_ellipsis || content == 0 || first_is_specifier;
                    }
                }
                _ if depth != 1 => {}
                _ => {
                    match tok {
                        Some(Token::Punctuator(Punctuator::Comma)) | Some(Token::Punctuator(Punctuator::Ellipsis)) => {
                            comma_or_ellipsis = true;
                        }
                        Some(Token::Keyword(
                            pat_storage_class_specifier!()
                            | pat_basic_type_specifier!()
                            | pat_type_qualifier!()
                            | pat_struct_or_union!()
                            | Keyword::Enum
                            | Keyword::Inline
                            | Keyword::_Imaginary
                            | Keyword::_Atomic
                            | Keyword::_Noreturn,
                        )) if content == 0 => {
                            first_is_specifier = true;
                        }
                        Some(Token::Identifier(sym)) if content == 0 && (self.is_typedef_name(*sym) || self.is_gnu_type_keyword(*sym)) => {
                            first_is_specifier = true;
                        }
                        _ => {}
                    }
                    content += 1;
                }
            }
            n += 1;
        }
    }
    pub fn parse_direct_declarator(&mut self) -> DirectDeclarator {
        let start = self.mark();
        let base = match self.tokens.peek().map(|x| &x.val) {
            Some(Token::Identifier(name)) => {
                let name = *name;
                self.tokens.advance();
                DirectBase::Ident(name)
            }
            Some(Token::Punctuator(Punctuator::LParen)) if self.paren_is_parameter_list() => {
                // Abstract function declarator (`int (void)`,
                // `int (int, char)`, `int (T)` in type-names and
                // parameters): no base name — the suffix loop below
                // parses the parameter list.
                DirectBase::Abstract
            }
            Some(Token::Punctuator(Punctuator::LParen)) => {
                self.tokens.advance();
                let inner = self.parse_declarator();
                self.expect_punct(Punctuator::RParen);
                DirectBase::Paren(Box::new(inner))
            }
            _ => DirectBase::Abstract,
        };
        let mut suffixes = Vec::new();
        loop {
            match self.tokens.peek().map(|x| &x.val) {
                Some(Token::Punctuator(Punctuator::LBracket)) => {
                    let open = self.tokens.peek().map(|t| t.span()).unwrap_or(Span::dummy());
                    self.tokens.advance();
                    let mut suffix = self.parse_array_suffix();
                    suffix.span = open.merge(suffix.span);
                    suffixes.push(DirectSuffix::Array(suffix));
                }
                Some(Token::Punctuator(Punctuator::LParen)) => {
                    let open = self.tokens.peek().map(|t| t.span()).unwrap_or(Span::dummy());
                    self.tokens.advance();
                    let mut suffix = self.parse_function_suffix();
                    suffix.span = open.merge(suffix.span);
                    suffixes.push(DirectSuffix::Function(suffix));
                }
                _ => break,
            }
        }
        let span = self.span_since(start);
        DirectDeclarator { base, suffixes, span }
    }
    /// Whether a type is variably modified (see
    /// `Sema::is_variably_modified`): a parser-side shorthand threading
    /// the interner.
    fn ty_is_variably_modified(&self, ty: &Ty) -> bool {
        self.sema.is_variably_modified(ty, &self.intern)
    }
    pub fn parse_array_suffix(&mut self) -> ArrayDeclarator {
        // Entry is just past the consumed `[` (the caller merges its span
        // in), so mark here and let the caller extend the range.
        let start = self.mark();
        let mut suffix = ArrayDeclarator::default();
        suffix.qualifiers = self.parse_type_qualifier_list();
        // `[*]`: a `*` closed immediately by `]` marks an unspecified
        // variable-length bound (C17 §6.7.6.2p2). Any other `*` (e.g. the
        // `*p` in `[ *p ]`) starts an assignment expression instead.
        match (self.tokens.peek().map(|x| &x.val), self.tokens.peek_nth(1).map(|x| &x.val)) {
            (Some(Token::Punctuator(Punctuator::Star)), Some(Token::Punctuator(Punctuator::RBracket))) => {
                self.tokens.advance();
                self.tokens.advance();
                // `[*]` lives only in prototype scope (C17 §6.7.6p3).
                if self.prototype_depth == 0 {
                    let span = self.span_since(start);
                    return self.fail(span, "`[*]` outside function prototype scope");
                }
                suffix.bound = Some(ArrayBound::Star);
                suffix.span = self.span_since(start);
                return suffix;
            }
            _ => {}
        }
        if matches!(self.tokens.peek().map(|x| &x.val), Some(Token::Keyword(Keyword::Static))) {
            self.tokens.advance();
            suffix.is_static = true;
            suffix.qualifiers.extend(self.parse_type_qualifier_list());
        }
        match self.tokens.peek().map(|x| &x.val) {
            Some(Token::Punctuator(Punctuator::RBracket)) => {
                self.tokens.advance();
            }
            // The bound is an assignment expression (C17 §6.7.6p1:
            // `[ assignment-expression ]`); commas never appear here.
            // Bounds check like any expression and must be integer. A
            // folded negative bound is rejected (C17 §6.7.6.2p1); a
            // bound that does not fold is variable-length: legal in
            // prototypes and block scope, but never at file scope
            // (C17 §6.7.6.2p2) — members are checked separately (see
            // `parse_struct_declaration_list`). Zero stays accepted as
            // a GNU extension.
            _ => {
                let bound = self.parse_assignment_expression();
                let bound = self.checked_expr(&bound);
                if !is_integer(&bound.ty) {
                    return self.fail(bound.span, "array bound must have integer type");
                }
                // Fold now, while the expression and its span are both
                // in hand; afterwards only the count survives. A bound
                // that does not fold is variable-length: its checked
                // expression is retained in `Sema::vla_bounds` for
                // runtime evaluation (allocation, `sizeof`, strides).
                suffix.bound = Some(match crate::eval::eval_const(&bound, self.sema, &self.intern) {
                    Some(crate::eval::Value::Int(n)) if n < 0 => return self.fail(bound.span, "negative array bound"),
                    Some(crate::eval::Value::Int(n)) => ArrayBound::Fixed(u64::try_from(n).unwrap_or(u64::MAX)),
                    _ => {
                        if self.struct_depth == 0 && self.sema.is_file_scope() {
                            return self.fail(bound.span, "variably modified type at file scope");
                        }
                        let id = self.sema.vla_bounds.len() as u32;
                        self.sema.vla_bounds.push(bound);
                        ArrayBound::Vla(id)
                    }
                });
                self.expect_punct(Punctuator::RBracket);
            }
        }
        suffix.span = self.span_since(start);
        suffix
    }
    // The `)`-ward suffix of a function declarator, reached with the
    // opening `(` already consumed: an empty list (`f()`, which supplies
    // no information about the parameters, C17 §6.7.6.3p14), an
    // old-style identifier-list (`f(a, b)`), or a parameter-type-list
    // (`f(int, ...)`).
    //
    // A prototype scope is pushed for the list so tags declared inside
    // (`int f(struct S { int x; } s);`) do not outlive the declarator;
    // the scope pops before the function itself is declared, so the
    // function lands in the enclosing scope.
    pub fn parse_function_suffix(&mut self) -> FunctionDeclarator {
        // Entry is just past the consumed `(` (the caller merges its span
        // in), so mark here and let the caller extend the range.
        let start = self.mark();
        self.sema.push_scope();
        self.prototype_depth += 1;
        let mut result = self.parse_function_suffix_inner();
        self.prototype_depth -= 1;
        self.sema.pop_scope();
        result.span = self.span_since(start);
        result
    }
    fn parse_function_suffix_inner(&mut self) -> FunctionDeclarator {
        let mut suffix = FunctionDeclarator::default();
        if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::RParen))) {
            self.tokens.advance();
            return suffix;
        }
        // identifier-list: bare names, no specifiers. Their types come
        // from the declaration list of a function *definition*, so
        // the names are recorded in `idents` for `parse_function_definition`
        // to rewrite into a prototype; `is_old_style` still marks the
        // declarator unprototyped until that rewrite runs.
        if matches!(
            self.tokens.peek_val(),
            Some(Token::Identifier(sym))
                if !self.is_typedef_name(*sym)
                    && !self.is_gnu_type_keyword(*sym)
                    && !self.is_future_specifier(*sym)
        ) && self.peek_gnu_qualifier().is_none()
        {
            let mut idents = Vec::new();
            loop {
                match self.tokens.peek().copied() {
                    Some(Located {
                        val: Token::Identifier(sym),
                        loc,
                        end,
                    }) => {
                        self.tokens.advance();
                        idents.push((sym, Span::new(loc, end)));
                    }
                    other => {
                        let span = self.peek_span();
                        return self.fail(span, format!("expected identifier in identifier list, got {other:?}"));
                    }
                }
                if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Comma))) {
                    self.tokens.advance();
                    continue;
                }
                break;
            }
            self.expect_punct(Punctuator::RParen);
            suffix.has_params = true;
            suffix.is_old_style = true;
            suffix.idents = idents;
            return suffix;
        }
        // parameter-type-list: parameter-list [, ...] (C17 §6.7.6.3p1).
        suffix.has_params = true;
        let mut params = Vec::new();
        loop {
            if self.cancelled() {
                break;
            }
            if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Ellipsis))) {
                // The grammar lets the ellipsis follow only a
                // parameter-list, but a bare `f(...)` is a GNU
                // extension (clang warns "ISO C requires a named
                // parameter before '...'"): accept with a warning.
                // libc and EDG output declare these.
                if params.is_empty() {
                    // self.warn(self.peek_span(), "ISO C requires a named parameter before `...`");
                }
                self.tokens.advance();
                suffix.variadic = true;
                break;
            }
            let mut param = self.parse_parameter_declaration();
            // Function prototype scope (C17 §6.2.1p4): a parameter name
            // is visible to later parameters' bounds, so
            // `void f(int n, int a[n])` resolves `n`.
            let proto = if let Some(name) = param.name {
                let declared = self.sema.declare(
                    name,
                    Binding {
                        name: Some(name),
                        ty: param.ty.clone(),
                        kind: BindingKind::Var,
                        linkage: Linkage::NoLinkage,
                        block_static: false,
                        tls: false,
                        defined: false,
                        tentative: false,
                        constructor: None,
                        destructor: None,
                        weak: false,
                        alias: None,
                        naked: false,
                        noinline: false,
                        is_async: false,
                        section: None,
                        asm_label: None,
                        inline: false,
                        align: None,
                        span: param.span,
                    },
                );
                self.unwrap_diag_opt(declared)
            } else {
                None
            };
            param.proto = proto;
            params.push(param);
            if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Comma))) {
                self.tokens.advance();
                continue;
            }
            break;
        }
        // An unnamed `void` as the only item means "no parameters"
        // (C17 §6.7.6.3p10); anywhere else — named, accompanied, or
        // followed by `...` — it is an error (gcc: "'void' must be the
        // only parameter and unnamed").
        self.expect_punct(Punctuator::RParen);
        if params.iter().any(|param| param.ty.unqualified() == &Ty::Void) {
            if params.len() != 1 || params[0].name.is_some() || suffix.variadic {
                let span = params.iter().map(|p| p.span).reduce(|a, b| a.merge(b)).unwrap_or_else(|| self.peek_span());
                return self.fail(span, "`void` must be the only parameter and unnamed");
            }
            params.clear();
        }
        suffix.params = params;
        suffix
    }
    // parameter-declaration: declaration-specifiers [declarator]
    // (C17 §6.7.6.3p1); both the declarator and the specifiers beyond
    // the type may be absent, so `f(int a)`, `f(int *)`, and `f(int)`
    // all parse here.
    pub fn parse_parameter_declaration(&mut self) -> Param {
        let start = self.mark();
        self.skip_gnu_suffixes();
        let specs = self.parse_declaration_specifiers();
        // `register` is the only storage class a parameter may carry
        // (C17 §6.7.6.3p2).
        if specs.class.iter().any(|sc| !matches!(sc, StorageClass::Register)) {
            let span = self.span_since(start);
            return self.fail(span, "storage class specifier in parameter");
        }
        if specs.thread_local {
            let span = self.span_since(start);
            return self.fail(span, "thread-local storage in parameter");
        }
        let declarator = self.parse_declarator();
        let name = declarator.name();
        self.skip_gnu_suffixes();
        let mut ty = self.spec_to_ty(&specs, &declarator);
        // Record a `[*]` bound before decay erases it (see `Param::star`).
        let star = ty_has_bare_star(&ty);
        // Likewise the variable-length bound: the parameter decays to a
        // pointer, but the bound still evaluates on entry to a
        // definition's body (C17 §6.7.6.2p6).
        let vla_bound = match &ty {
            Ty::Array {
                bound: Some(ArrayBound::Vla(id)), ..
            } => Some(*id),
            _ => None,
        };
        // A parameter declared as a function adjusts to a pointer to
        // function (C17 §6.7.6.3p8).
        if matches!(ty, Ty::Func { .. }) {
            ty = Ty::Pointer {
                qualifiers: Vec::new(),
                size: PtrSize::Default,
                pointee: Box::new(ty),
            };
        }
        // A parameter declared as an array adjusts to a qualified
        // pointer to its element; bracket qualifiers (`[const 3]`)
        // qualify the pointer (C17 §6.7.6.3p7).
        if let Ty::Array { elem, quals, .. } = ty {
            ty = Ty::Pointer { qualifiers: quals, size: PtrSize::Default, pointee: elem };
        }
        let span = self.span_since(start);
        Param {
            name,
            ty,
            body_ty: None,
            vla_bound,
            span,
            star,
            proto: None,
        }
    }
    pub fn parse_type_qualifier_list(&mut self) -> Vec<TypeQualifier> {
        let mut qualifiers = Vec::new();
        loop {
            // `__unaligned` modifies pointer types too; its effect is
            // erased (see `peek_ms_unaligned`).
            if self.peek_ms_unaligned() {
                self.tokens.advance();
                continue;
            }
            if let Some(q) = self.peek_gnu_qualifier() {
                qualifiers.push(q);
                self.tokens.advance();
                continue;
            }
            match self.tokens.peek().map(|x| &x.val) {
                Some(Token::Keyword(tq @ pat_type_qualifier!())) => {
                    qualifiers.push(match tq {
                        Keyword::Const => TypeQualifier::Const,
                        Keyword::Restrict => TypeQualifier::Restrict,
                        Keyword::Volatile => TypeQualifier::Volatile,
                        _ => unreachable!(),
                    });
                    self.tokens.advance();
                }
                // `_Atomic` qualifies pointers/arrays too; its effect is
                // erased (see `TypeSpecifier::Atomic`).
                Some(Token::Keyword(Keyword::_Atomic)) => {
                    self.tokens.advance();
                }
                _ => break,
            }
        }
        qualifiers
    }
    fn expect_punct(&mut self, expected: Punctuator) {
        // Underline the token that was actually wrong. Peeking instead
        // would name the one *after* it, since `advance` has moved on.
        match self.tokens.advance().map(|t| (t.span(), t.val)) {
            Some((_, Token::Punctuator(found))) if found == expected => {}
            Some((span, found)) => self.fail(span, format!("expected `{}`, got {:?}", expected.as_str(), found)),
            None => {
                let span = self.peek_span();
                self.fail(span, format!("expected `{}`, got end of input", expected.as_str()))
            }
        }
    }
    fn expect_keyword(&mut self, expected: Keyword) {
        match self.tokens.advance().map(|t| (t.span(), t.val)) {
            Some((_, Token::Keyword(found))) if found == expected => {}
            Some((span, found)) => self.fail(span, format!("expected `{}`, got {:?}", expected.as_str(), found)),
            None => {
                let span = self.peek_span();
                self.fail(span, format!("expected `{}`, got end of input", expected.as_str()))
            }
        }
    }
    /// `_Future(T)` (glcc extension): the boxed handle an async call
    /// returns. Handles are interned by result type (see
    /// `Sema::future_handle`), so every `_Future(T)` mention is one
    /// nominal struct. Without `-fglcc-async` this still diagnoses
    /// (rather than pre-processing away) so the spelling never silently
    /// means something else.
    fn parse_future_specifier(&mut self) -> TypeSpecifier {
        let start = self.mark();
        self.tokens.advance();
        if !self.sema.async_enabled {
            return self.fail(self.span_since(start), "`_Future` requires the glcc async extension (-fglcc-async)");
        }
        self.expect_punct(Punctuator::LParen);
        let ty = self.parse_type_name();
        self.expect_punct(Punctuator::RParen);
        let handle = self.sema.future_handle(ty);
        TypeSpecifier::Builtin(handle)
    }
    pub fn parse_declaration_specifiers(&mut self) -> DeclarationSpecifiers {
        let start = self.mark();
        let mut ds = DeclarationSpecifiers::default();
        loop {
            // Annotations interleave specifiers freely in GNU C
            // (`static __attribute__((aligned(128))) union ...`):
            // declaration-level effects (`weak`) fold into the
            // specifiers; asm labels and `__extension__` markers
            // still just skip.
            ds.attrs.merge(self.parse_gnu_annotations());
            // `_Alignas`/`alignas` alignment specifiers (C11 §6.7.5)
            // interleave with the other specifiers; the effect is a
            // minimum alignment, shared with GNU `aligned`.
            if self.peek_alignas() {
                self.parse_alignas(&mut ds.attrs);
                continue;
            }
            // Thread-local storage (`_Thread_local`, `__thread`,
            // `thread_local`): like a storage class, combinable with
            // `static`/`extern`.
            if self.peek_is_gnu("_Thread_local") || self.peek_is_gnu("__thread") || self.peek_is_gnu("thread_local") {
                ds.thread_local = true;
                self.tokens.advance();
                continue;
            }
            if let Some(q) = self.peek_gnu_qualifier() {
                ds.type_qualifier.push(q);
                self.tokens.advance();
                continue;
            }
            if self.peek_alternate_inline() {
                ds.function_specifier += 1;
                self.tokens.advance();
                continue;
            }
            // C11 `_Noreturn` function specifier; the attribute is a
            // hint this compiler does not use, so it is only counted.
            if matches!(self.tokens.peek_val(), Some(Token::Keyword(Keyword::_Noreturn))) {
                ds.noreturn = true;
                self.tokens.advance();
                continue;
            }
            // C11 `_Atomic`: either the type specifier `_Atomic(type-name)`
            // or the qualifier `_Atomic type`. Atomic types have the
            // representation of their base and no atomic operations are
            // modeled, so both spellings erase to the operand type.
            if matches!(self.tokens.peek_val(), Some(Token::Keyword(Keyword::_Atomic))) {
                self.tokens.advance();
                if self.eat_punct(Punctuator::LParen) {
                    let ty = self.parse_type_name();
                    self.expect_punct(Punctuator::RParen);
                    ds.type_specifier.push(TypeSpecifier::Atomic(ty));
                }
                continue;
            }
            // The builtin `va_list` and `__float128` types: opaque
            // builtins, each a whole type by itself. `__signed__` is
            // the GNU spelling of `signed` (<asm-generic/int-ll64.h>
            // writes `typedef __signed__ char __s8;`).
            if self.peek_c23_keyword("typeof_unqual") {
                let ty = self.parse_typeof_specifier(true);
                ds.type_specifier.push(TypeSpecifier::Builtin(ty));
                continue;
            }
            if self.peek_is_gnu("typeof") || self.peek_is_gnu("__typeof__") || self.peek_is_gnu("__typeof") {
                let ty = self.parse_typeof_specifier(false);
                ds.type_specifier.push(TypeSpecifier::Builtin(ty));
                continue;
            }
            if self.peek_is_gnu("__builtin_va_list") {
                self.tokens.advance();
                // On Windows the builtin `va_list` *is* `char *` (the
                // MSVC ABI, and how clang models it there), so the
                // spelling normalizes to a plain pointer: the
                // `typedef __builtin_va_list va_list;` of clang's
                // <stdarg.h> then merges with `typedef char* va_list;`
                // from <vadefs.h> instead of conflicting.
                let ty = if self.sema.target.os == crate::target::Os::Windows {
                    Ty::Pointer {
                        qualifiers: Vec::new(),
                        size: PtrSize::Default,
                        pointee: Box::new(Ty::Char),
                    }
                } else {
                    Ty::BuiltinVaList
                };
                ds.type_specifier.push(TypeSpecifier::Builtin(ty));
                continue;
            }
            if self.peek_is_gnu("__float128") {
                self.tokens.advance();
                ds.type_specifier.push(TypeSpecifier::Builtin(Ty::Float128));
                continue;
            }
            if self.peek_is_gnu("__signed__") {
                self.tokens.advance();
                ds.type_specifier.push(TypeSpecifier::Signed);
                continue;
            }
            // MSVC calling-convention keywords (`vadefs.h`:
            // `void __cdecl __va_start(va_list*, ...);`). Win64 has a
            // single convention, so they are accepted and ignored.
            if ["__cdecl", "__stdcall", "__fastcall", "__thiscall", "__vectorcall", "__clrcall"]
                .into_iter()
                .any(|kw| self.peek_is_gnu(kw))
            {
                self.tokens.advance();
                continue;
            }
            // MSVC `__declspec(...)` annotations (`vcruntime.h`:
            // `__declspec(noreturn) void __cdecl __report_gsfailure(...)`).
            // Ignored like other attributes: `dllimport` resolves from
            // the static CRT libraries at link time.
            if self.peek_msvc_operator("__declspec") || self.peek_msvc_operator("_declspec") {
                self.skip_msvc_operator();
                continue;
            }
            if self.peek_is_gnu("__int128") {
                self.tokens.advance();
                ds.type_specifier.push(TypeSpecifier::Int128);
                continue;
            }
            // MSVC's fixed-width spellings (`vcruntime.h`:
            // `typedef unsigned __int64 uintptr_t;`), composable with
            // `signed`/`unsigned` like the ISO keywords they alias.
            if self.peek_is_gnu("__int8") {
                self.tokens.advance();
                ds.type_specifier.push(TypeSpecifier::Char);
                continue;
            }
            if self.peek_is_gnu("__int16") {
                self.tokens.advance();
                ds.type_specifier.push(TypeSpecifier::Short);
                continue;
            }
            if self.peek_is_gnu("__int32") {
                self.tokens.advance();
                ds.type_specifier.push(TypeSpecifier::Int);
                continue;
            }
            if self.peek_is_gnu("__int64") {
                self.tokens.advance();
                ds.type_specifier.push(TypeSpecifier::Long);
                ds.type_specifier.push(TypeSpecifier::Long);
                continue;
            }
            // GCC's predefined 128-bit typedefs need no declaration.
            if self.peek_is_gnu("__uint128_t") {
                self.tokens.advance();
                ds.type_specifier.push(TypeSpecifier::Builtin(Ty::UnsignedInt128));
                continue;
            }
            if self.peek_is_gnu("__int128_t") {
                self.tokens.advance();
                ds.type_specifier.push(TypeSpecifier::Builtin(Ty::Int128));
                continue;
            }
            // GNU complex spellings (`__complex__`, `__complex`):
            // synonyms of `_Complex`, combinable with any real type
            // (`__complex__ double`, `__complex__ int`).
            if self.peek_is_gnu("__complex__") || self.peek_is_gnu("__complex") {
                self.tokens.advance();
                ds.type_specifier.push(TypeSpecifier::_Complex);
                continue;
            }
            // ISO C23 `_Float32`/`_Float64`: ABI-identical to
            // `float`/`double` (binary32/binary64), so alias the
            // existing types (glibc declares `strtof32` et al. with
            // these under `_GNU_SOURCE`). Only in specifier-first
            // position: glibc also ships `typedef float _Float32;`
            // (for pre-C23 compilers), where the name is the
            // declarator — like `int T;` below, a trailing ident
            // declares instead of specifying.
            if ds.type_specifier.is_empty() && self.peek_is_gnu("_Float32") {
                self.tokens.advance();
                ds.type_specifier.push(TypeSpecifier::Builtin(Ty::Float));
                continue;
            }
            if ds.type_specifier.is_empty() && self.peek_is_gnu("_Float64") {
                self.tokens.advance();
                ds.type_specifier.push(TypeSpecifier::Builtin(Ty::Double));
                continue;
            }
            match self.tokens.peek().map(|x| &x.val) {
                // C99 §6.7.2 lists `_Imaginary` among the type specifiers,
                // but imaginary types are a conditional feature this
                // compiler does not implement (GCC rejects the keyword
                // outright too). Diagnose it here so it never falls
                // through to the implicit-`int` or expression paths.
                Some(Token::Keyword(Keyword::_Imaginary)) => {
                    return self.fail(self.peek_span(), "`_Imaginary` is not supported");
                }
                Some(Token::Keyword(sc @ pat_storage_class_specifier!())) => {
                    ds.class.push(match sc {
                        Keyword::Typedef => StorageClass::Typedef,
                        Keyword::Extern => StorageClass::Extern,
                        Keyword::Static => StorageClass::Static,
                        Keyword::Auto => StorageClass::Auto,
                        Keyword::Register => StorageClass::Register,
                        _ => unreachable!(),
                    });

                    self.tokens.advance();
                }
                Some(Token::Keyword(ts @ pat_basic_type_specifier!())) => {
                    ds.type_specifier.push(match ts {
                        Keyword::Void => TypeSpecifier::Void,
                        Keyword::Char => TypeSpecifier::Char,
                        Keyword::Short => TypeSpecifier::Short,
                        Keyword::Int => TypeSpecifier::Int,
                        Keyword::Long => TypeSpecifier::Long,
                        Keyword::Float => TypeSpecifier::Float,
                        Keyword::Double => TypeSpecifier::Double,
                        Keyword::Signed => TypeSpecifier::Signed,
                        Keyword::Unsigned => TypeSpecifier::Unsigned,
                        Keyword::_Bool => TypeSpecifier::_Bool,
                        Keyword::_Complex => TypeSpecifier::_Complex,
                        _ => unreachable!(),
                    });

                    self.tokens.advance();
                }
                Some(Token::Keyword(sou @ pat_struct_or_union!())) => {
                    let is_union = matches!(sou, Keyword::Union);
                    ds.type_specifier.push(self.parse_struct_specifier(is_union));
                }
                Some(Token::Keyword(Keyword::Enum)) => {
                    ds.type_specifier.push(self.parse_enum_specifier());
                }
                // `__unaligned` rides among the declaration specifiers
                // (`WCHAR UNALIGNED *`) but affects nothing here.
                Some(Token::Identifier(_)) if self.peek_ms_unaligned() => {
                    self.tokens.advance();
                }
                // A typedef name acts as the type specifier itself
                // (`Arg a;`), but only while no type specifier has been
                // seen: the `T` in `int T;` is the declarator, not a
                // second specifier.
                Some(Token::Identifier(sym)) if ds.type_specifier.is_empty() => {
                    let sym = *sym;
                    if self.sema.lang.standard_includes(Std::C23) && self.intern.resolve(sym) == Some("bool") {
                        self.tokens.advance();
                        ds.type_specifier.push(TypeSpecifier::_Bool);
                        continue;
                    }
                    if self.is_future_specifier(sym) {
                        ds.type_specifier.push(self.parse_future_specifier());
                        continue;
                    }
                    // Anything else that is neither a typedef name nor
                    // a GNU type keyword opens the declarator under C89
                    // implicit `int` (`static inline fu (x) { ... }`,
                    // `f (a, b)`), so leave the token for the declarator
                    // parser and fill in the missing specifier.
                    if !self.is_typedef_name(sym) {
                        let span = self.peek_span();
                        self.warn(span, "missing type specifier; C89 implicit `int` assumed");
                        ds.type_specifier.push(TypeSpecifier::Int);
                        break;
                    }
                    self.tokens.advance();
                    let id = self.sema.lookup_symbol(sym).expect("typedef name has a binding");
                    let ty = self.sema.decls[id.0 as usize].ty.clone();
                    ds.type_specifier.push(TypeSpecifier::TypedefName(ty));
                }
                Some(Token::Keyword(tq @ pat_type_qualifier!())) => {
                    ds.type_qualifier.push(match tq {
                        Keyword::Const => TypeQualifier::Const,
                        Keyword::Restrict => TypeQualifier::Restrict,
                        Keyword::Volatile => TypeQualifier::Volatile,
                        _ => unreachable!(),
                    });
                    self.tokens.advance();
                }
                Some(Token::Keyword(Keyword::Inline)) => {
                    ds.function_specifier += 1;
                    self.tokens.advance();
                }
                _ => break,
            }
        }
        ds.span = self.span_since(start);
        ds
    }

    // struct/union [identifier] ["{" struct-declaration-list "}"]
    // (C17 §6.7.2.1). A body makes it a definition: the tag enters the
    // current scope *before* the body parses, so the struct can contain
    // pointers to itself, and is completed afterwards. Without a body it
    // only references — or implicitly forward-declares — the tag.
    pub fn parse_struct_specifier(&mut self, is_union: bool) -> TypeSpecifier {
        let kw_span = self.tokens.peek().map(|t| t.span()).unwrap_or(Span::dummy());
        self.tokens.advance();
        // Prefix attributes (`struct __attribute__((packed)) S {...}`):
        // decoded effects apply to the definition below.
        let mut attrs = self.parse_gnu_annotations();
        let tag = if let Some(Located { val: Token::Identifier(s), .. }) = self.tokens.peek().copied() {
            let tag_span = self.tokens.peek().map(|t| t.span()).unwrap_or(Span::dummy());
            self.tokens.advance();
            Some((s, tag_span))
        } else {
            None
        };
        attrs.merge(self.parse_gnu_annotations());
        if !matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LBrace))) {
            // Anonymous types must carry a body to describe (C17 §6.7.2.3).
            let Some((tag, tag_span)) = tag else {
                let span = self.peek_span();
                return self.fail(span, "anonymous struct/union without a body");
            };
            let span = kw_span.merge(tag_span);
            let referred = self.sema.reference_tag(tag, is_union, span);
            return match self.unwrap_diag_opt(referred) {
                Some(id) => TypeSpecifier::Struct(id),
                None => TypeSpecifier::recover(),
            };
        }
        let id = match tag {
            Some((tag, tag_span)) => {
                let defined = self.sema.define_tag(tag, is_union, kw_span.merge(tag_span));
                match self.unwrap_diag_opt(defined) {
                    Some(id) => id,
                    None => return TypeSpecifier::recover(),
                }
            }
            None => self.sema.new_anonymous_struct(is_union, kw_span),
        };
        attrs.merge(self.parse_gnu_annotations());
        self.expect_punct(Punctuator::LBrace);
        let members = self.parse_struct_declaration_list();
        if !self.cancelled() {
            self.expect_punct(Punctuator::RBrace);
        }
        // Trailing attributes (`struct S {...} __attribute__((packed))`):
        // verified against GCC, these apply to the type even when a
        // declarator follows (only attributes *after* the declarator
        // belong to the declaration, which layout never sees).
        attrs.merge(self.parse_gnu_annotations());
        self.check_struct_members_complete(is_union, &members);
        let completed = self.sema.complete_tag(id, members);
        self.unwrap_diag(completed);
        // Attributes on a mere forward declaration carry nothing (verified
        // against GCC); only a definition takes effects. They land
        // before any layout query, but clear the memo anyway.
        {
            let def = &mut self.sema.structs[id.0 as usize];
            if attrs.packed {
                def.packed = true;
            }
            if let Some(n) = attrs.aligned {
                def.align_override = Some(def.align_override.map_or(n, |m| m.max(n)));
            }
        }
        self.sema.invalidate_layout(id);
        TypeSpecifier::Struct(id)
    }

    /// Every named member of a struct/union definition must have complete
    /// type (C17 §6.7.2.1p2); the definition is still open here, which is
    /// what catches self-embedding by value. A trailing bound-less array
    /// in a struct is the flexible array member exception (C17 §6.7.2.1p3).
    fn check_struct_members_complete(&self, is_union: bool, members: &[Member]) {
        let last = members.len().saturating_sub(1);
        for (i, member) in members.iter().enumerate() {
            let Some(name) = member.name else { continue };
            if !is_union && i == last && matches!(member.ty.unqualified(), Ty::Array { bound: None, .. }) {
                continue;
            }
            if self.sema.is_complete(&member.ty) {
                continue;
            }
            let kind = if is_union { "union" } else { "struct" };
            let spelling = self.intern.resolve(name).unwrap_or("?");
            return self.fail(member.span, format!("{kind} member '{spelling}' has incomplete type"));
        }
    }

    /// Skip one balanced parenthesized group with the `(` still
    /// unconsumed: the contents of a GNU `__attribute__((...))` or
    /// `__asm__("...")`. Contents parse as raw balanced tokens (no
    /// attribute semantics are modeled).
    fn skip_balanced_parens(&mut self) {
        debug_assert!(matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LParen))));
        let mut depth = 0u32;
        loop {
            match self.tokens.advance().map(|x| x.val) {
                Some(Token::Punctuator(Punctuator::LParen)) => depth += 1,
                Some(Token::Punctuator(Punctuator::RParen)) => {
                    depth -= 1;
                    if depth == 0 {
                        return;
                    }
                }
                None => {
                    let span = self.peek_span();
                    return self.fail(span, "unterminated `__attribute__`/`__asm__`");
                }
                _ => {}
            }
        }
    }

    /// Skip GNU `__attribute__((...))` annotations, `__asm__("...")`
    /// groups (symbol renames after declarators, assembly statements
    /// in bodies, top-level assembler — all parsed as balanced groups),
    /// and bare `__extension__` markers. Attributes parse through the
    /// validating list parser (so malformed ones error instead of
    /// slipping past a balanced skip); effects are discarded here —
    /// only struct/union and member positions apply them, since only
    /// layout consumes any (`packed`, `aligned`). Called at every
    /// position one may appear: declaration prefixes and suffixes,
    /// declarators, members, parameters, and statements.
    fn skip_gnu_suffixes(&mut self) {
        // Effects discarded: positions that use them (`parse_struct_specifier`,
        // `parse_struct_declaration_list`) call `parse_gnu_annotations` directly.
        self.parse_gnu_annotations();
    }

    /// Skip GNU markers at a position where `__asm__` starts a statement
    /// rather than naming a symbol (block items): attributes and
    /// `__extension__` only.
    fn skip_gnu_markers(&mut self) {
        self.parse_gnu_annotations_impl(false);
    }

    /// Parse GNU annotations into decoded effects: `__attribute__`
    /// lists fold `packed`/`aligned` (unknown attributes validate
    /// structurally and are ignored), `__asm__` labels record the
    /// assembler symbol name, and bare `__extension__` markers advance.
    fn parse_gnu_annotations(&mut self) -> GnuAttributes {
        self.parse_gnu_annotations_impl(true)
    }

    fn parse_gnu_annotations_impl(&mut self, asm_labels: bool) -> GnuAttributes {
        let mut out = GnuAttributes::default();
        loop {
            // Both spellings are accepted: `__attribute` appears in the
            // wild next to `__attribute__` (GCC and clang take both).
            if (self.peek_is_gnu("__attribute__") || self.peek_is_gnu("__attribute"))
                && matches!(self.tokens.peek_nth(1).map(|t| &t.val), Some(Token::Punctuator(Punctuator::LParen)))
            {
                out.merge(self.parse_gnu_attribute_list());
            } else if asm_labels && self.peek_is_asm_keyword() && matches!(self.tokens.peek_nth(1).map(|t| &t.val), Some(Token::Punctuator(Punctuator::LParen))) {
                self.tokens.advance();
                self.expect_punct(Punctuator::LParen);
                let label = self.parse_asm_string("label");
                self.expect_punct(Punctuator::RParen);
                out.asm_label = out.asm_label.or(Some(self.intern.get_or_intern(&label)));
            } else if self.peek_is_gnu("__extension__") {
                self.tokens.advance();
            } else if ["__cdecl", "__stdcall", "__fastcall", "__thiscall", "__vectorcall", "__clrcall"]
                .into_iter()
                .any(|kw| self.peek_is_gnu(kw))
            {
                // MSVC calling conventions sit between the specifiers
                // and the declared name (`unsigned __int64* __cdecl f(void)`).
                self.tokens.advance();
            } else if self.peek_msvc_operator("__declspec") || self.peek_msvc_operator("_declspec") || self.peek_msvc_operator("__pragma") {
                self.skip_msvc_operator();
            } else {
                break;
            }
        }
        out
    }

    /// Whether the next token is a `_Static_assert`/`static_assert`
    /// keyword (`_Static_assert` needs C11, the bare C23 spelling needs
    /// C23).
    fn peek_is_static_assert(&self) -> bool {
        match self.tokens.peek_val() {
            Some(Token::Identifier(sym)) => match self.intern.resolve(*sym) {
                Some("_Static_assert") => self.sema.lang.standard_includes(Std::C11),
                Some("static_assert") => self.sema.lang.standard_includes(Std::C23),
                _ => false,
            },
            _ => false,
        }
    }

    /// `_Static_assert ( constant-expression , string-literal ) ;`
    /// (C11 §6.7.10): checked here and discarded — it declares nothing.
    /// A false or non-integer-constant condition is an error; the
    /// message only feeds the diagnostic. The C23 single-argument form
    /// parses too, liberally.
    fn parse_static_assert(&mut self) -> Stmt {
        let start = self.mark();
        self.tokens.advance();
        self.expect_punct(Punctuator::LParen);
        let arg = self.parse_assignment_expression();
        let checked = self.checked_expr(&arg);
        let value = crate::eval::eval_const(&checked, self.sema, &self.intern);
        let msg = if self.eat_punct(Punctuator::Comma) {
            let Some(Token::StringLiteral(lit)) = self.tokens.peek_val() else {
                return self.fail(self.peek_span(), "`_Static_assert` needs a string literal message");
            };
            let lit = *lit;
            self.tokens.advance();
            self.intern.resolve(lit.spelling).unwrap_or("").to_string()
        } else {
            String::new()
        };
        self.expect_punct(Punctuator::RParen);
        let span = self.span_since(start);
        self.expect_punct(Punctuator::Semi);
        match value {
            Some(crate::eval::Value::Int(0)) => {
                if msg.is_empty() {
                    self.fail(span, "static assertion failed")
                } else {
                    self.fail(span, format!("static assertion failed: {msg}"))
                }
            }
            Some(crate::eval::Value::Int(_)) => Stmt::Empty { span },
            _ => self.fail(span, "`_Static_assert` needs an integer constant expression"),
        }
    }

    /// Whether the next token is an `_Alignas`/`alignas` specifier
    /// (C11 spelling / C23 bare spelling, gated per standard).
    fn peek_alignas(&self) -> bool {
        match self.tokens.peek_val() {
            Some(Token::Identifier(sym)) => match self.intern.resolve(*sym) {
                Some("_Alignas") => self.sema.lang.standard_includes(Std::C11),
                Some("alignas") => self.sema.lang.gnu || self.sema.lang.standard_includes(Std::C23),
                _ => false,
            },
            _ => false,
        }
    }

    /// `_Alignas ( type-name | constant-expression )` (C11 §6.7.5):
    /// folds into `attrs.aligned` as a minimum, like GNU `aligned`.
    /// `_Alignas(0)` is ignored (C11 §6.7.5p6); other constants must be
    /// powers of two, and the type-name form uses that type's
    /// alignment.
    fn parse_alignas(&mut self, attrs: &mut GnuAttributes) {
        let kw_span = self.peek_span();
        self.tokens.advance();
        self.expect_punct(Punctuator::LParen);
        let n = if self.token_starts_type_name(self.tokens.peek().map(|t| &t.val)) {
            let ty = self.parse_type_name();
            match self.sema.type_align(&ty, &self.intern) {
                Some(align) => align,
                None => return self.fail(kw_span, "alignment of incomplete type requested"),
            }
        } else {
            let arg = self.parse_assignment_expression();
            let checked = self.checked_expr(&arg);
            match crate::eval::eval_const(&checked, self.sema, &self.intern) {
                Some(crate::eval::Value::Int(0)) => 0,
                Some(crate::eval::Value::Int(n)) if n > 0 => u64::try_from(n).unwrap_or(u64::MAX),
                _ => return self.fail(kw_span, "`_Alignas` needs an integer constant expression or a type name"),
            }
        };
        self.expect_punct(Punctuator::RParen);
        if n == 0 {
            return;
        }
        if !n.is_power_of_two() {
            return self.fail(kw_span, format!("requested alignment '{n}' is not a power of 2"));
        }
        attrs.aligned = Some(attrs.aligned.map_or(n, |m: u64| m.max(n)));
    }

    /// Record `async` on `out` when the extension is enabled; without
    /// `-fglcc-async` the attribute warns and is ignored, so code never
    /// silently compiles as something other than what it asked for.
    fn set_async_attr(&self, out: &mut GnuAttributes, span: Span) {
        if self.sema.async_enabled {
            out.is_async = true;
        } else {
            self.warn(span, "attribute `async` ignored (the glcc async extension needs -fglcc-async)");
        }
    }

    /// Parse one `__attribute__((name, name(args), ...))` with the
    /// keyword unconsumed: the double parens wrap a comma-separated
    /// list of names with optional argument groups. `packed` and
    /// `aligned(n)` decode into effects; every other attribute (the
    /// `nothrow`/`format`/`mode`/... in system headers) validates
    /// structurally and is ignored — the frontend has no diagnostics
    /// or codegen to consume it. A malformed list errors.
    fn parse_gnu_attribute_list(&mut self) -> GnuAttributes {
        self.tokens.advance();
        self.expect_punct(Punctuator::LParen);
        // The standard double parens wrap the list (`((packed))`); a
        // single paren also parses, liberally.
        let double = matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LParen)));
        if double {
            self.tokens.advance();
        }
        let mut out = GnuAttributes::default();
        loop {
            if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::RParen))) {
                // `__attribute__(())`: empty list, accepted like GCC.
                self.tokens.advance();
                break;
            }
            // The name: an identifier in practice; a keyword also
            // parses (robustness — no name is ever looked up).
            let name_start = self.mark();
            let sym = match self.tokens.advance().map(|located| located.val) {
                Some(Token::Identifier(sym)) => Some(sym),
                Some(Token::Keyword(_)) => None,
                other => {
                    let span = self.peek_span();
                    return self.fail(span, format!("expected attribute name, got {other:?}"));
                }
            };
            let name_span = self.span_since(name_start);
            // Normalized spelling: `__aligned__` ≡ `aligned`.
            let normalized = match sym {
                Some(sym) => self.intern.resolve(sym).unwrap_or("").trim_matches('_').to_string(),
                None => String::new(),
            };
            if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LParen))) {
                if normalized == "weak" || normalized == "weakref" {
                    // `weak` takes no meaningful arguments for
                    // codegen (`weak("alias")` names an alias target
                    // we do not implement); skip whatever follows.
                    self.skip_balanced_parens();
                    out.weak = true;
                } else if normalized == "naked" {
                    self.skip_balanced_parens();
                    out.naked = true;
                } else if normalized == "noinline" {
                    self.skip_balanced_parens();
                    out.noinline = true;
                } else if normalized == "async" {
                    self.skip_balanced_parens();
                    self.set_async_attr(&mut out, name_span);
                } else if normalized == "alias" {
                    // `alias("name")`: the target spelling is resolved
                    // to a declaration when this declaration registers.
                    self.tokens.advance();
                    let body = match self.tokens.advance().map(|located| located.val) {
                        Some(Token::StringLiteral(lit)) => self.string_literal_body(lit),
                        other => {
                            let span = self.peek_span();
                            return self.fail(span, format!("attribute `alias` needs a string literal, got {other:?}"));
                        }
                    };
                    let target = self.intern.get_or_intern(&body);
                    out.alias = out.alias.or(Some(target));
                    self.expect_punct(Punctuator::RParen);
                } else if normalized == "section" {
                    // `section("name")`: the object lands in that
                    // section (bare-metal vector tables, `.noinit`, ...).
                    self.tokens.advance();
                    let body = match self.tokens.advance().map(|located| located.val) {
                        Some(Token::StringLiteral(lit)) => self.string_literal_body(lit),
                        other => {
                            let span = self.peek_span();
                            return self.fail(span, format!("attribute `section` needs a string literal, got {other:?}"));
                        }
                    };
                    let name = self.intern.get_or_intern(&body);
                    out.section = out.section.or(Some(name));
                    self.expect_punct(Punctuator::RParen);
                } else if normalized == "constructor" || normalized == "destructor" {
                    // Optional startup priority (`constructor(101)`,
                    // bare attributes are 65535 like GCC; reserved
                    // 0-100 range accepted but not specially ordered).
                    self.tokens.advance();
                    let priority = if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::RParen))) {
                        65535
                    } else {
                        let arg = self.parse_assignment_expression();
                        let checked = self.checked_expr(&arg);
                        match crate::eval::eval_const(&checked, self.sema, &self.intern) {
                            Some(crate::eval::Value::Int(n)) if n >= 0 => u64::try_from(n).unwrap_or(u64::MAX),
                            _ => {
                                return self.fail(name_span, format!("attribute `{normalized}` needs a non-negative integer constant"));
                            }
                        }
                    };
                    self.expect_punct(Punctuator::RParen);
                    if normalized == "constructor" {
                        out.constructor = out.constructor.or(Some(priority));
                    } else {
                        out.destructor = out.destructor.or(Some(priority));
                    }
                } else if normalized == "aligned" {
                    self.tokens.advance();
                    let arg = self.parse_assignment_expression();
                    let checked = self.checked_expr(&arg);
                    match crate::eval::eval_const(&checked, self.sema, &self.intern) {
                        Some(crate::eval::Value::Int(n)) if n > 0 => {
                            let n = u64::try_from(n).unwrap_or(u64::MAX);
                            if n.is_power_of_two() {
                                out.aligned = Some(out.aligned.map_or(n, |m: u64| m.max(n)));
                            } else {
                                return self.fail(name_span, format!("requested alignment '{n}' is not a power of 2"));
                            }
                        }
                        Some(crate::eval::Value::Int(0)) => {
                            // GCC warns and ignores `aligned(0)`.
                            self.warn(name_span, "requested alignment '0' is ignored");
                        }
                        _ => {
                            return self.fail(name_span, "attribute `aligned` needs an integer constant");
                        }
                    }
                    self.expect_punct(Punctuator::RParen);
                } else {
                    // Unknown attributes never evaluate their arguments
                    // (`format(printf, 1, 2)` names formats, not values).
                    self.skip_balanced_parens();
                }
            } else if normalized == "packed" {
                out.packed = true;
            } else if normalized == "weak" || normalized == "weakref" {
                out.weak = true;
            } else if normalized == "naked" {
                out.naked = true;
            } else if normalized == "noinline" {
                out.noinline = true;
            } else if normalized == "async" {
                self.set_async_attr(&mut out, name_span);
            } else if normalized == "aligned" {
                // Bare `aligned`: GCC's maximum useful alignment for
                // the target (16 on x86-64).
                let max = self.sema.target.max_useful_align;
                out.aligned = Some(out.aligned.map_or(max, |m: u64| m.max(max)));
            } else if normalized == "constructor" {
                // Bare `constructor`: default priority 65535.
                out.constructor = out.constructor.or(Some(65535));
            } else if normalized == "destructor" {
                out.destructor = out.destructor.or(Some(65535));
            }
            if normalized == "interrupt" {
                // The prologue/epilogue rewrite (save every register,
                // return with `iret`/`eret`/`mret`) cannot be expressed
                // through a compiler-generated body.
                self.warn(name_span, "attribute `interrupt` is not supported; write the handler as a naked function");
            }
            if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Comma))) {
                self.tokens.advance();
                continue;
            }
            break;
        }
        self.expect_punct(Punctuator::RParen);
        if double {
            self.expect_punct(Punctuator::RParen);
        }
        out
    }

    /// Body of a string literal token, escapes left raw: attributes
    /// like `alias("name")` take a plain spelling, not a value.
    fn string_literal_body(&self, lit: StringLiteral) -> String {
        let spelling = self.intern.resolve(lit.spelling).unwrap_or("");
        spelling.split_once('"').and_then(|(_, rest)| rest.strip_suffix('"')).unwrap_or(spelling).to_string()
    }

    // enum [identifier] [{ enumerator-list [,] }] (C17 §6.7.2.2). A body
    // makes it a definition: each enumerator declares an `int` constant
    // (C17 §6.4.4.3) in the current scope from its declaration point on,
    // so later enumerators may use earlier ones (`B = A + 1`). Without
    // a body it only references — or implicitly forward-declares — the
    // tag, mirroring struct specifiers.
    pub fn parse_enum_specifier(&mut self) -> TypeSpecifier {
        let kw_span = self.tokens.peek().map(|t| t.span()).unwrap_or(Span::dummy());
        self.tokens.advance();
        self.skip_gnu_suffixes();
        let tag = if let Some(Located { val: Token::Identifier(s), .. }) = self.tokens.peek().copied() {
            let tag_span = self.tokens.peek().map(|t| t.span()).unwrap_or(Span::dummy());
            self.tokens.advance();
            Some((s, tag_span))
        } else {
            None
        };
        self.skip_gnu_suffixes();
        if !matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LBrace))) {
            // Anonymous types must carry a body to describe (C17 §6.7.2.3).
            let Some((tag, tag_span)) = tag else {
                let span = self.peek_span();
                return self.fail(span, "anonymous enum without a body");
            };
            let span = kw_span.merge(tag_span);
            let referred = self.sema.reference_enum_tag(tag, span);
            return match self.unwrap_diag_opt(referred) {
                Some(id) => TypeSpecifier::Enum(id),
                None => TypeSpecifier::recover(),
            };
        }
        let id = match tag {
            Some((tag, tag_span)) => {
                let defined = self.sema.define_enum_tag(tag, kw_span.merge(tag_span));
                match self.unwrap_diag_opt(defined) {
                    Some(id) => id,
                    None => return TypeSpecifier::recover(),
                }
            }
            None => self.sema.new_anonymous_enum(kw_span),
        };
        self.expect_punct(Punctuator::LBrace);
        let (enumerators, repr) = self.parse_enumerator_list();
        if !self.cancelled() {
            self.expect_punct(Punctuator::RBrace);
        }
        self.skip_gnu_suffixes();
        let completed = self.sema.complete_enum_tag(id, enumerators, repr);
        self.unwrap_diag(completed);
        TypeSpecifier::Enum(id)
    }

    // `A, B = 3, C, ...` with the `{` consumed and `}` unconsumed: each
    // name declares a constant in the current scope, and an optional
    // `= assignment-expression` checks and folds to the enumerator's
    // value (later enumerators may use earlier ones: `B = A + 1`).
    // Anything but an integer constant expression — floats,
    // variables, calls — fails here. A trailing comma is allowed.
    //
    // Values follow the C23 §6.7.2.2p4 ladder, not just `int`: each
    // constant takes the smallest of `int`, `unsigned int`, `long`,
    // `unsigned long` holding it (so `DW_CIE_ID_32` is unsigned), and
    // the enum's representation is the largest of those. Anything
    // outside `unsigned long` fails.
    fn parse_enumerator_list(&mut self) -> (Vec<Enumerator>, Ty) {
        let mut enumerators = Vec::new();
        let mut previous: Option<i128> = None;
        let mut repr = Ty::Int;
        let mut all_nonnegative = true;
        loop {
            if self.cancelled() {
                break;
            }
            if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::RBrace))) {
                break;
            }
            let start = self.mark();
            let name = match self.tokens.peek().copied() {
                Some(Located { val: Token::Identifier(name), .. }) => {
                    self.tokens.advance();
                    name
                }
                other => {
                    let span = self.peek_span();
                    return self.fail(span, format!("expected enumerator name, got {other:?}"));
                }
            };
            let value = if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Assign))) {
                self.tokens.advance();
                let expr = self.parse_assignment_expression();
                Some(self.checked_expr(&expr))
            } else {
                None
            };
            let span = self.span_since(start);
            // Fold the value: an explicit value must be an integer
            // constant expression. Previous enumerators resolve through
            // their bindings, so each value is known when the next
            // parses.
            let const_value: i128 = match &value {
                Some(expr) => match crate::eval::eval_const(expr, self.sema, &self.intern) {
                    Some(crate::eval::Value::Int(v)) => v,
                    _ => {
                        return self.fail(span, "enumerator value is not an integer constant expression");
                    }
                },
                None => previous.map(|p| p.wrapping_add(1)).unwrap_or(0),
            };
            let const_ty = match Self::enum_ladder_ty(const_value) {
                Some(ty) => ty,
                None => {
                    return self.fail(span, "enumerator value is not representable");
                }
            };
            // The enum's representation is the largest ladder rung any
            // value needed (plain `int` for ordinary enums), with the
            // unsigned rung when no value is negative — GCC's rule for
            // C enums, and what makes an enum bit-field read unsigned.
            if Self::enum_ladder_rank(&const_ty) > Self::enum_ladder_rank(&repr) {
                repr = const_ty.clone();
            }
            all_nonnegative &= const_value >= 0;
            previous = Some(const_value);
            let declared = self.sema.declare(
                name,
                Binding {
                    name: Some(name),
                    ty: const_ty,
                    kind: BindingKind::EnumConst { value: const_value },
                    linkage: Linkage::NoLinkage,
                    block_static: false,
                    tls: false,
                    defined: true,
                    tentative: false,
                    constructor: None,
                    destructor: None,
                    weak: false,
                    alias: None,
                    naked: false,
                    noinline: false,
                    is_async: false,
                    section: None,
                    asm_label: None,
                    inline: false,
                    align: None,
                    span,
                },
            );
            self.unwrap_diag(declared);
            enumerators.push(Enumerator { name, value, span });
            if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Comma))) {
                self.tokens.advance();
                continue;
            }
            break;
        }
        if all_nonnegative {
            repr = match repr {
                Ty::Int => Ty::UnsignedInt,
                Ty::Long => Ty::UnsignedLong,
                other => other,
            };
        }
        (enumerators, repr)
    }

    // Smallest C integer type holding enumerator value `v` (C23
    // §6.7.2.2p4 ladder: `int`, `unsigned int`, `long`,
    // `unsigned long`), or `None` past `unsigned long`.
    fn enum_ladder_ty(v: i128) -> Option<Ty> {
        if v >= i32::MIN as i128 && v <= i32::MAX as i128 {
            Some(Ty::Int)
        } else if v >= 0 && v <= u32::MAX as i128 {
            Some(Ty::UnsignedInt)
        } else if v >= i64::MIN as i128 && v <= i64::MAX as i128 {
            Some(Ty::Long)
        } else if v >= 0 && v <= u64::MAX as i128 {
            Some(Ty::UnsignedLong)
        } else {
            None
        }
    }

    // Ladder order for picking an enum's representation: the largest
    // rung any value needed.
    fn enum_ladder_rank(ty: &Ty) -> u8 {
        match ty {
            Ty::Int => 0,
            Ty::UnsignedInt => 1,
            Ty::Long => 2,
            Ty::UnsignedLong => 3,
            _ => 0,
        }
    }

    // struct-declaration-list: any number of `specifier-qualifier-list
    // struct-declarator-list ";"` declarations before the closing `}`
    // (C17 §6.7.2.1p3). Storage classes and function members are not
    // allowed here.
    pub fn parse_struct_declaration_list(&mut self) -> Vec<Member> {
        let mut members = Vec::new();
        self.struct_depth += 1;
        while !self.cancelled() {
            let Some(tok) = self.tokens.peek_val() else { break };
            if matches!(tok, Token::Punctuator(Punctuator::RBrace)) {
                break;
            }
            // A lone `;` is an empty member declaration (GNU, C23):
            // it declares nothing.
            if matches!(tok, Token::Punctuator(Punctuator::Semi)) {
                self.tokens.advance();
                continue;
            }
            // Prefix annotations on members; `packed`/`aligned` here
            // apply to every declarator in this specifier list, while
            // suffix annotations bind one declarator each.
            let prefix_attrs = self.parse_gnu_annotations();
            let specifiers = self.parse_declaration_specifiers();
            if specifiers.has_storage_class() || specifiers.thread_local {
                let span = specifiers.span;
                return self.fail(span, "storage class specifier in struct member");
            }
            loop {
                let declarator = self.parse_declarator();
                // Suffix annotations: `int x __attribute__((aligned(8)))`
                // and bit-field widths share this position.
                let mut member_attrs = prefix_attrs.clone();
                member_attrs.merge(specifiers.attrs.clone());
                member_attrs.merge(self.parse_gnu_annotations());
                // A bit-field width (`int x : 3;`, `unsigned : 0;`): the
                // width checks as an integer-typed expression but is not
                // constant-evaluated yet (like array bounds).
                let bitfield = if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Colon))) {
                    self.tokens.advance();
                    let width = self.parse_assignment_expression();
                    Some(self.checked_integer_expr(&width, "bit-field width"))
                } else {
                    None
                };
                // Attributes may also follow a bit-field's width:
                // `unsigned x : 3 __attribute__((packed))`.
                member_attrs.merge(self.parse_gnu_annotations());
                let member_span = specifiers.span().merge(declarator.span());
                let ty = self.spec_to_ty(&specifiers, &declarator);
                if bitfield.is_some() && !is_integer(&ty) {
                    return self.fail(member_span, "bit-field has non-integer type");
                }
                // Widths fold here: they must be non-negative integer
                // constants within the member's type, and a named width
                // must be nonzero (layout re-derives them from the
                // stored expressions).
                if let Some(width) = &bitfield {
                    let folded = crate::eval::eval_const(width, self.sema, &self.intern);
                    match folded {
                        Some(crate::eval::Value::Int(v)) => {
                            if v < 0 {
                                return self.fail(member_span, "negative bit-field width");
                            }
                            let bits = crate::check::int_type_size(&ty, &self.sema.target).map(|s| s as u64 * 8).unwrap_or(0);
                            if v as u64 > bits {
                                return self.fail(member_span, "bit-field width exceeds its type");
                            }
                            if declarator.name().is_some() && v == 0 {
                                return self.fail(member_span, "zero width for bit-field");
                            }
                        }
                        _ => {
                            return self.fail(member_span, "bit-field width is not an integer constant expression");
                        }
                    }
                }
                let name = match declarator.name() {
                    Some(name) => Some(name),
                    None if bitfield.is_some() => None,
                    None => {
                        // Anonymous struct/union member (C11 §6.7.2.1p13 —
                        // the `__extension__ union {...};` shape in
                        // <bits/sigcontext.h>): kept as one nameless
                        // member, so its members overlap (unions) and
                        // keep their padding (structs) in layout. Access
                        // promotes through it (see `check_member`).
                        // Anything else without a declarator (a bare
                        // tag, a named definition, an enum) declares no
                        // member — accepted, like gcc's "declaration
                        // does not declare anything" warning.
                        let anonymous = matches!(
                            specifiers.type_specifier.as_slice(),
                            [TypeSpecifier::Struct(id)] if {
                                let def = &self.sema.structs[id.0 as usize];
                                def.tag.is_none() && def.complete
                            }
                        );
                        if !anonymous {
                            break;
                        }
                        None
                    }
                };
                // `int f();` lowers to a function type, which no member may
                // have; `int (*f)();` (pointer *to* function) lowers to a
                // pointer and stays legal.
                if matches!(ty, Ty::Func { .. }) {
                    return self.fail(member_span, "function member");
                }
                // Members are never variably modified (C17 §6.7.6.2p2):
                // neither `[*]` nor a runtime bound, at any depth
                // outside a prototype (`int (*f)(int[n])` is a plain
                // pointer and stays legal).
                if self.ty_is_variably_modified(&ty) {
                    return self.fail(member_span, "variably modified struct member");
                }
                members.push(Member {
                    name,
                    ty,
                    bitfield,
                    packed: member_attrs.packed,
                    align_override: member_attrs.aligned,
                    span: member_span,
                });
                if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Comma))) {
                    self.tokens.advance();
                    continue;
                }
                break;
            }
            self.expect_punct(Punctuator::Semi);
        }
        self.struct_depth -= 1;
        members
    }

    // ---- expressions (C17 §6.5) ----
    //
    // Precedence climbing: `parse_expression` is the comma level used
    // where commas cannot separate anything else (statement, condition,
    // `return`, `for` clauses); `parse_assignment_expression` and below
    // serve positions where a comma would separate instead (call
    // arguments, array bounds, initializers, `case` labels).

    /// Full expression, comma operator included.
    pub fn parse_expression(&mut self) -> Expr {
        let mut lhs = self.parse_assignment_expression();
        while matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Comma))) {
            self.tokens.advance();
            let rhs = self.parse_assignment_expression();
            let span = lhs.span().merge(rhs.span());
            lhs = Expr::Binary {
                op: BinaryOp::Comma,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            };
        }
        lhs
    }

    /// Assignment level (right-associative). Lvalues are not checked
    /// here — any expression may appear left of `=`; validity is later
    /// semantic work.
    fn parse_assignment_expression(&mut self) -> Expr {
        let start = self.mark();
        let lhs = self.parse_conditional_expression();
        let Some(op) = self.peek_assign_op() else {
            return lhs;
        };
        self.tokens.advance();
        let rhs = self.parse_assignment_expression();
        Expr::Assign {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            span: self.span_since(start),
        }
    }

    /// `cond ? then : els`: the middle operand allows commas, the third
    /// associates right (`a ? b : c ? d : e` groups as `a ? b : (c?d:e)`).
    fn parse_conditional_expression(&mut self) -> Expr {
        let start = self.mark();
        let cond = self.parse_binary_expression(0);
        if !matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Question))) {
            return cond;
        }
        self.tokens.advance();
        // GNU `x ?: y`: no then-arm — the condition's own value is the
        // result when nonzero (evaluated once, unlike `x ? x : y`).
        if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Colon))) {
            self.tokens.advance();
            let els = self.parse_conditional_expression();
            return Expr::Elvis {
                cond: Box::new(cond),
                els: Box::new(els),
                span: self.span_since(start),
            };
        }
        let then = self.parse_expression();
        self.expect_punct(Punctuator::Colon);
        let els = self.parse_conditional_expression();
        Expr::Ternary {
            cond: Box::new(cond),
            then: Box::new(then),
            els: Box::new(els),
            span: self.span_since(start),
        }
    }

    /// Left-associative binary levels, `min_prec` and up. Binding powers
    /// follow C17 §6.5 (tighter binds higher): `||` 10 through `*` 100.
    fn parse_binary_expression(&mut self, min_prec: u8) -> Expr {
        let start = self.mark();
        let mut lhs = self.parse_cast_expression();
        loop {
            let Some(prec) = self.peek_binary_prec() else {
                break;
            };
            if prec < min_prec {
                break;
            }
            let op = self.peek_binary_op().expect("precedence implies operator");
            self.tokens.advance();
            let rhs = self.parse_binary_expression(prec + 1);
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span: self.span_since(start),
            };
        }
        lhs
    }

    /// Binding power of the upcoming binary operator, if any. Single
    /// tokens decide: the lexer already split `==` from `=` (and `<<=`
    /// from `<<`), so no multi-token lookahead is needed.
    fn peek_binary_prec(&self) -> Option<u8> {
        match self.tokens.peek_val() {
            Some(Token::Punctuator(p)) => binary_binding_power(*p),
            _ => None,
        }
    }

    /// The binary operator upcoming; call only when `peek_binary_prec`
    /// returned `Some`.
    fn peek_binary_op(&self) -> Option<BinaryOp> {
        match self.tokens.peek_val() {
            Some(Token::Punctuator(p)) => Some(match p {
                Punctuator::Star => BinaryOp::Mul,
                Punctuator::Slash => BinaryOp::Div,
                Punctuator::Percent => BinaryOp::Rem,
                Punctuator::Plus => BinaryOp::Add,
                Punctuator::Minus => BinaryOp::Sub,
                Punctuator::LessLess => BinaryOp::Shl,
                Punctuator::GreaterGreater => BinaryOp::Shr,
                Punctuator::Less => BinaryOp::Lt,
                Punctuator::LessEq => BinaryOp::Le,
                Punctuator::Greater => BinaryOp::Gt,
                Punctuator::GreaterEq => BinaryOp::Ge,
                Punctuator::EqEq => BinaryOp::Eq,
                Punctuator::BangEq => BinaryOp::Ne,
                Punctuator::Amp => BinaryOp::BitAnd,
                Punctuator::Caret => BinaryOp::BitXor,
                Punctuator::Pipe => BinaryOp::BitOr,
                Punctuator::AmpAmp => BinaryOp::And,
                Punctuator::PipePipe => BinaryOp::Or,
                _ => return None,
            }),
            _ => None,
        }
    }

    /// The upcoming assignment operator (`=` plus the ten compound
    /// forms), if any.
    fn peek_assign_op(&self) -> Option<AssignOp> {
        match self.tokens.peek_val() {
            Some(Token::Punctuator(p)) => Some(match p {
                Punctuator::Assign => AssignOp::Assign,
                Punctuator::PlusEq => AssignOp::Add,
                Punctuator::MinusEq => AssignOp::Sub,
                Punctuator::StarEq => AssignOp::Mul,
                Punctuator::SlashEq => AssignOp::Div,
                Punctuator::PercentEq => AssignOp::Rem,
                Punctuator::LessLessEq => AssignOp::Shl,
                Punctuator::GreaterGreaterEq => AssignOp::Shr,
                Punctuator::AmpEq => AssignOp::BitAnd,
                Punctuator::CaretEq => AssignOp::BitXor,
                Punctuator::PipeEq => AssignOp::BitOr,
                _ => return None,
            }),
            _ => None,
        }
    }

    /// Cast or unary: `( type-name ) cast-expr` when the parenthesized
    /// lead is a declaration specifier (see `paren_starts_type_name`),
    /// else a unary expression. Chained casts associate right.
    ///
    /// A `( type-name ) { ... }` lead is a compound literal (C17
    /// §6.5.2.5), which is a postfix expression: it parses below so
    /// member access, subscripting, and calls attach to the literal
    /// itself (`(int[]){1, 2}[0]`).
    fn parse_cast_expression(&mut self) -> Expr {
        if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LParen))) && self.paren_starts_compound_literal() {
            return self.parse_unary_expression();
        }
        if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LParen))) && self.paren_starts_type_name() {
            let start = self.mark();
            self.tokens.advance();
            let ty = self.parse_type_name();
            self.expect_punct(Punctuator::RParen);
            let expr = self.parse_cast_expression();
            return Expr::Cast {
                ty,
                expr: Box::new(expr),
                span: self.span_since(start),
            };
        }
        self.parse_unary_expression()
    }

    /// Whether `( ...` opens a compound literal: a type-name lead
    /// whose matching `)` is followed by `{` (C17 §6.5.2.5). Scans
    /// past nested parens (function-pointer declarators carry
    /// parameter lists), so `(int (*)(int)) {0}` still counts while
    /// `(x)[0]` never does.
    fn paren_starts_compound_literal(&self) -> bool {
        if !matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LParen))) {
            return false;
        }
        if !self.paren_starts_type_name() {
            return false;
        }
        let mut depth = 0u32;
        let mut n = 0usize;
        loop {
            match self.tokens.peek_nth(n).map(|located| &located.val) {
                Some(Token::Punctuator(Punctuator::LParen)) => depth += 1,
                Some(Token::Punctuator(Punctuator::RParen)) => {
                    depth -= 1;
                    if depth == 0 {
                        return matches!(self.tokens.peek_nth(n + 1).map(|located| &located.val), Some(Token::Punctuator(Punctuator::LBrace)));
                    }
                }
                None => return false,
                _ => {}
            }
            n += 1;
        }
    }

    /// `( type-name ) { initializer-list [,] }` with `(` unconsumed:
    /// the unnamed object of C17 §6.5.2.5. Only reached on a
    /// `paren_starts_compound_literal` lead (see
    /// `parse_postfix_expression`); the array bound of `[]`
    /// completes later, at check time.
    fn parse_compound_literal(&mut self) -> Expr {
        let start = self.mark();
        self.tokens.advance();
        let ty = self.parse_type_name();
        self.expect_punct(Punctuator::RParen);
        // The function-type constraint reports here, where the type is
        // known: the initializer check below would only say "invalid
        // initializer for type" (GCC: "compound literal has function
        // type"; the check-time arm keeps the same error as a backstop).
        if matches!(ty.unqualified(), Ty::Func { .. }) {
            return self.fail(self.span_since(start), "compound literal has function type");
        }
        let brace = self.parse_brace_list();
        let init = self.checked_initializer(&ty, &brace);
        Expr::CompoundLiteral {
            ty,
            init: Box::new(init),
            span: self.span_since(start),
        }
    }

    /// Whether `( ...` opens a type-name: the token after `(` starts a
    /// declaration specifier. A typedef name counts as one (C17
    /// §6.7.6.3p11 takes an ambiguous identifier as the typedef), so
    /// `(T)+x` casts while `(x)+y` groups — matching the disambiguation
    /// used for parameter lists.
    fn paren_starts_type_name(&self) -> bool {
        // GNU `__extension__` markers and `__attribute__((...))` groups
        // after `(` belong to whatever follows: `(__extension__ int)x`
        // and `(__attribute__((unused)) int)x` cast, while
        // `(__extension__ 1.0iF)` groups (the `I` macro in
        // <complex.h> needs the latter).
        let mut n = 1;
        loop {
            match self.tokens.peek_nth(n).map(|located| &located.val) {
                Some(Token::Identifier(sym)) if self.intern.resolve(*sym) == Some("__extension__") => n += 1,
                Some(Token::Identifier(sym))
                    if matches!(self.intern.resolve(*sym), Some("__attribute__" | "__attribute"))
                        && matches!(self.tokens.peek_nth(n + 1).map(|t| &t.val), Some(Token::Punctuator(Punctuator::LParen))) =>
                {
                    match self.balanced_paren_end(n + 1) {
                        Some(end) => n = end,
                        None => return false,
                    }
                }
                tok => return self.token_starts_type_name(tok),
            }
        }
    }

    /// Index just past the balanced `(`...`)` group starting at offset
    /// `n`, or `None` if it never closes.
    fn balanced_paren_end(&self, n: usize) -> Option<usize> {
        let mut depth = 0u32;
        let mut i = n;
        loop {
            match self.tokens.peek_nth(i).map(|located| &located.val) {
                Some(Token::Punctuator(Punctuator::LParen)) => depth += 1,
                Some(Token::Punctuator(Punctuator::RParen)) => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i + 1);
                    }
                }
                None => return None,
                _ => {}
            }
            i += 1;
        }
    }

    /// Whether one token starts a declaration specifier: a basic type
    /// keyword, or an identifier naming a typedef or a GNU builtin
    /// type. Shared by the cast/group disambiguation above.
    fn token_starts_type_name(&self, tok: Option<&Token>) -> bool {
        match tok {
            Some(Token::Keyword(
                Keyword::Void
                | Keyword::Char
                | Keyword::Short
                | Keyword::Int
                | Keyword::Long
                | Keyword::Float
                | Keyword::Double
                | Keyword::Signed
                | Keyword::Unsigned
                | Keyword::_Bool
                | Keyword::_Complex
                | Keyword::_Imaginary
                | Keyword::_Atomic
                | Keyword::Struct
                | Keyword::Union
                | Keyword::Enum
                | Keyword::Const
                | Keyword::Restrict
                | Keyword::Volatile,
            )) => true,
            Some(Token::Identifier(sym)) => {
                self.is_typedef_name(*sym)
                    || self.is_gnu_type_keyword(*sym)
                    || self.is_future_specifier(*sym)
                    || self.is_c23_keyword(*sym, "typeof_unqual")
                    || self.is_c23_keyword(*sym, "bool")
            }
            _ => false,
        }
    }

    /// A type-name: specifier-qualifier-list plus an (optionally abstract)
    /// declarator, as in casts and `sizeof`. Storage classes and `inline`
    /// are rejected, as is a named outer declarator (parameter names
    /// inside are fine — they never enter scope).
    fn parse_type_name(&mut self) -> Ty {
        let specs = self.parse_declaration_specifiers();
        if specs.has_storage_class() {
            return self.fail(specs.span, "storage class specifier in type name");
        }
        if specs.is_inline() {
            return self.fail(specs.span, "`inline` in type name");
        }
        if specs.is_noreturn() {
            return self.fail(specs.span, "`_Noreturn` in type name");
        }
        let declarator = self.parse_declarator();
        if declarator.name().is_some() {
            return self.fail(declarator.span(), "declarator in type name");
        }
        self.spec_to_ty(&specs, &declarator)
    }

    /// Unary operators (right-associative) and `sizeof` in both forms.
    /// A `sizeof ( ... )` parenthesized lead takes the type branch under
    /// the same rule as casts, except when it is a compound literal
    /// (`sizeof (type){...}`), which parses as a unary operand.
    /// GNU `__alignof__`/`__alignof` and C11 `_Alignof` parse identically
    /// (a type-name or a unary operand), yielding the alignment.
    fn parse_unary_expression(&mut self) -> Expr {
        let start = self.mark();
        // GNU label address (`&&label`): only a prefix form, so `a &&
        // b` still parses as logical and (the `&&` there is consumed
        // by the binary loop, never seen here).
        if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::AmpAmp))) {
            if let Some(Token::Identifier(label)) = self.tokens.peek_nth(1).map(|located| &located.val) {
                let label = *label;
                self.tokens.advance();
                self.tokens.advance();
                return Expr::LabelAddr {
                    label: self.resolve_label(label),
                    span: self.span_since(start),
                };
            }
        }
        let is_alignof = match self.tokens.peek_val() {
            Some(Token::Identifier(sym)) => match self.intern.resolve(*sym) {
                Some("__alignof__" | "__alignof" | "_Alignof") => true,
                Some("alignof") => self.sema.lang.standard_includes(Std::C23),
                _ => false,
            },
            _ => false,
        };
        if is_alignof {
            self.tokens.advance();
            if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LParen))) && self.paren_starts_type_name() && !self.paren_starts_compound_literal() {
                self.tokens.advance();
                let ty = self.parse_type_name();
                self.expect_punct(Punctuator::RParen);
                return Expr::AlignofTy { ty, span: self.span_since(start) };
            }
            let operand = self.parse_unary_expression();
            return Expr::AlignofExpr {
                expr: Box::new(operand),
                span: self.span_since(start),
            };
        }
        let op = match self.tokens.peek_val() {
            Some(Token::Punctuator(Punctuator::PlusPlus)) => UnaryOp::PreInc,
            Some(Token::Punctuator(Punctuator::MinusMinus)) => UnaryOp::PreDec,
            Some(Token::Punctuator(Punctuator::Plus)) => UnaryOp::Pos,
            Some(Token::Punctuator(Punctuator::Minus)) => UnaryOp::Neg,
            Some(Token::Punctuator(Punctuator::Bang)) => UnaryOp::Not,
            Some(Token::Punctuator(Punctuator::Tilde)) => UnaryOp::BitNot,
            Some(Token::Punctuator(Punctuator::Star)) => UnaryOp::Deref,
            Some(Token::Punctuator(Punctuator::Amp)) => UnaryOp::AddrOf,
            Some(Token::Identifier(sym)) if matches!(self.intern.resolve(*sym), Some("__real__" | "__real")) => UnaryOp::Real,
            Some(Token::Identifier(sym)) if matches!(self.intern.resolve(*sym), Some("__imag__" | "__imag")) => UnaryOp::Imag,
            Some(Token::Keyword(Keyword::Sizeof)) => {
                self.tokens.advance();
                // `sizeof (type){...}` is `sizeof` of a compound
                // literal, not a parenthesized type-name: `(type){`
                // takes the operand branch and parses as a postfix
                // expression (C17 §6.5.2.5).
                if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LParen))) && self.paren_starts_type_name() && !self.paren_starts_compound_literal() {
                    self.tokens.advance();
                    let ty = self.parse_type_name();
                    self.expect_punct(Punctuator::RParen);
                    return Expr::SizeofTy { ty, span: self.span_since(start) };
                }
                let operand = self.parse_unary_expression();
                return Expr::SizeofExpr {
                    expr: Box::new(operand),
                    span: self.span_since(start),
                };
            }
            _ => return self.parse_postfix_expression(),
        };
        self.tokens.advance();
        let operand = self.parse_cast_expression();
        Expr::Unary {
            op,
            operand: Box::new(operand),
            span: self.span_since(start),
        }
    }

    /// Postfix suffixes, applied left to right: `f(args)`, `a[i]`,
    /// `s.m`, `p->m`, `i++`, `i--`.
    fn parse_postfix_expression(&mut self) -> Expr {
        let start = self.mark();
        // A compound literal opens the postfix chain (C17 §6.5.2.5p1:
        // it *is* a postfix expression), so suffixes attach to the
        // object itself. Anything else starts at the primary.
        let mut expr = if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LParen))) && self.paren_starts_compound_literal() {
            self.parse_compound_literal()
        } else {
            self.parse_primary_expression()
        };
        loop {
            match self.tokens.peek_val() {
                Some(Token::Punctuator(Punctuator::LBracket)) => {
                    self.tokens.advance();
                    let index = self.parse_expression();
                    self.expect_punct(Punctuator::RBracket);
                    expr = Expr::Subscript {
                        base: Box::new(expr),
                        index: Box::new(index),
                        span: self.span_since(start),
                    };
                }
                Some(Token::Punctuator(Punctuator::LParen)) => {
                    self.tokens.advance();
                    // `va_arg(ap, TYPE)` spells a call but takes a
                    // *type* second operand (C17 §7.16.1.1): parse it
                    // as such instead of an expression.
                    let is_va_arg = matches!(&expr, Expr::Ident { name, .. } if self.intern.resolve(*name) == Some("__builtin_va_arg"));
                    if is_va_arg {
                        let ap = self.parse_assignment_expression();
                        self.expect_punct(Punctuator::Comma);
                        let ty = self.parse_type_name();
                        self.expect_punct(Punctuator::RParen);
                        expr = Expr::VaArg {
                            ap: Box::new(ap),
                            ty,
                            span: self.span_since(start),
                        };
                        continue;
                    }
                    // `__builtin_offsetof(TYPE, .a.b[i])` likewise takes
                    // a type plus a member designator (GCC's `offsetof`).
                    let is_offsetof = matches!(&expr, Expr::Ident { name, .. } if self.intern.resolve(*name) == Some("__builtin_offsetof"));
                    if is_offsetof {
                        let ty = self.parse_type_name();
                        self.expect_punct(Punctuator::Comma);
                        let designators = self.parse_offset_designators();
                        self.expect_punct(Punctuator::RParen);
                        expr = Expr::Offsetof {
                            ty,
                            designators,
                            span: self.span_since(start),
                        };
                        continue;
                    }
                    // `__builtin_types_compatible_p(TYPE, TYPE)` takes
                    // two *type* operands and folds to `int` right here
                    // (a constant expression, usable as an array bound).
                    let is_types_compatible = matches!(&expr, Expr::Ident { name, .. } if self.intern.resolve(*name) == Some("__builtin_types_compatible_p"));
                    if is_types_compatible {
                        let a = self.parse_type_name();
                        self.expect_punct(Punctuator::Comma);
                        let b = self.parse_type_name();
                        self.expect_punct(Punctuator::RParen);
                        let spelling = self.intern.get_or_intern(if crate::sema::types_compatible_p(&a, &b) { "1" } else { "0" });
                        expr = Expr::Int {
                            lit: crate::lexer::IntLiteral {
                                spelling,
                                radix: crate::lexer::Radix::Dec,
                                suffix: crate::lexer::IntSuffix::None,
                                imag: false,
                            },
                            span: self.span_since(start),
                        };
                        continue;
                    }
                    let mut args = Vec::new();
                    if !matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::RParen))) {
                        loop {
                            args.push(self.parse_assignment_expression());
                            if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Comma))) {
                                self.tokens.advance();
                                continue;
                            }
                            break;
                        }
                    }
                    self.expect_punct(Punctuator::RParen);
                    // `__builtin_expect(x, c)` is `x` (branch prediction
                    // only): drop the call, keep the value. The other
                    // arguments are still evaluated, so they become
                    // comma operands: `(c, x)`.
                    if matches!(&expr, Expr::Ident { name, .. } if self.intern.resolve(*name) == Some("__builtin_expect")) {
                        if !args.is_empty() {
                            let mut value = args.remove(0);
                            for side in args.drain(..) {
                                let span = side.span().merge(value.span());
                                value = Expr::Binary {
                                    op: BinaryOp::Comma,
                                    lhs: Box::new(side),
                                    rhs: Box::new(value),
                                    span,
                                };
                            }
                            expr = value;
                            continue;
                        }
                    }
                    // `__builtin_unreachable()` is abnormal termination
                    // if ever reached: `abort()`. A trap would do as
                    // well, but a call needs no block-terminator
                    // surgery and the implicit-declaration path (or the
                    // real header) applies unchanged.
                    if matches!(&expr, Expr::Ident { name, .. } if self.intern.resolve(*name) == Some("__builtin_unreachable")) {
                        let span = expr.span();
                        let abort = self.intern.get_or_intern("abort");
                        expr = Expr::Ident { name: abort, span };
                    }
                    // `__builtin_X()` for X a known C library function
                    // is exactly `X()` (same contract; only compile-time
                    // folding is lost): rewrite so the normal
                    // implicit-declaration path applies (or the real
                    // header declaration, when included). The prototype
                    // table is the allowlist — it is also what keeps the
                    // implicit declaration from truncating the result.
                    // Runtime-backed builtins rewrite to their runtime
                    // symbol the same way. Anything else keeps its callee
                    // below (and its clear "undeclared" error until a
                    // real lowering exists).
                    if let Expr::Ident { name, .. } = &expr {
                        if let Some(text) = self.intern.resolve(*name) {
                            let target = text
                                .strip_prefix("__builtin_")
                                .filter(|libc| Self::libc_prototype(libc, &self.sema.target.size_type()).is_some())
                                .map(str::to_owned)
                                .or_else(|| {
                                    Self::RUNTIME_BUILTIN_NAMES
                                        .iter()
                                        .find(|(builtin, _)| *builtin == text)
                                        .map(|(_, symbol)| (*symbol).to_owned())
                                });
                            if let Some(target) = target {
                                let span = expr.span();
                                let id = self.intern.get_or_intern(&target);
                                expr = Expr::Ident { name: id, span };
                            }
                        }
                    }
                    // Implicit GCC builtins behind <stdarg.h> (`va_start`
                    // expands to `__builtin_va_start`, ...): declare on
                    // first call so the later check resolves. Anything
                    // else undeclared gets a C89 implicit declaration.
                    self.declare_builtin_if_called(&expr);
                    self.declare_implicit_function_if_called(&expr);
                    expr = Expr::Call {
                        func: Box::new(expr),
                        args,
                        span: self.span_since(start),
                    };
                }
                Some(Token::Punctuator(Punctuator::Dot | Punctuator::Arrow)) => {
                    let arrow = matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Arrow)));
                    self.tokens.advance();
                    let field = match self.tokens.advance().map(|located| located.val) {
                        Some(Token::Identifier(name)) => name,
                        other => {
                            let span = self.peek_span();
                            return self.fail(span, format!("expected member name, got {other:?}"));
                        }
                    };
                    expr = Expr::Member {
                        base: Box::new(expr),
                        field,
                        arrow,
                        span: self.span_since(start),
                    };
                }
                Some(Token::Punctuator(Punctuator::PlusPlus)) => {
                    self.tokens.advance();
                    expr = Expr::Unary {
                        op: UnaryOp::PostInc,
                        operand: Box::new(expr),
                        span: self.span_since(start),
                    };
                }
                Some(Token::Punctuator(Punctuator::MinusMinus)) => {
                    self.tokens.advance();
                    expr = Expr::Unary {
                        op: UnaryOp::PostDec,
                        operand: Box::new(expr),
                        span: self.span_since(start),
                    };
                }
                _ => break,
            }
        }
        expr
    }

    /// `_Generic (ctrl, assoc, ...)` with `_Generic` unconsumed: the
    /// controlling expression plus a comma-separated association list
    /// of `type-name : assignment-expression` and at most one
    /// `default : assignment-expression` (C11 §6.5.1.1).
    fn parse_generic_selection(&mut self) -> Expr {
        let start = self.mark();
        self.tokens.advance();
        self.expect_punct(Punctuator::LParen);
        let ctrl = self.parse_assignment_expression();
        self.expect_punct(Punctuator::Comma);
        let mut assocs = Vec::new();
        loop {
            let assoc_start = self.mark();
            let ty = if matches!(self.tokens.peek_val(), Some(Token::Keyword(Keyword::Default))) {
                self.tokens.advance();
                None
            } else {
                Some(self.parse_type_name())
            };
            self.expect_punct(Punctuator::Colon);
            let expr = self.parse_assignment_expression();
            assocs.push(GenericAssoc {
                ty,
                expr: Box::new(expr),
                span: self.span_since(assoc_start),
            });
            if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Comma))) {
                self.tokens.advance();
                continue;
            }
            break;
        }
        self.expect_punct(Punctuator::RParen);
        Expr::Generic {
            ctrl: Box::new(ctrl),
            assocs,
            span: self.span_since(start),
        }
    }
    /// Member designators for `__builtin_offsetof(TYPE, ...)`: a field
    /// first (with an optional leading dot — both `offsetof(S, x)` and
    /// `offsetof(S, .x)` occur), then `.field`/`[index]` steps
    /// (mirroring initializer designators, including inline checking
    /// of indices).
    fn parse_offset_designators(&mut self) -> Vec<Designator> {
        let mut designators = Vec::new();
        if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::Dot))) {
            self.tokens.advance();
        }
        let first_start = self.mark();
        let first = match self.tokens.advance().map(|located| located.val) {
            Some(Token::Identifier(name)) => name,
            other => {
                let span = self.peek_span();
                return self.fail(span, format!("expected field name, got {other:?}"));
            }
        };
        designators.push(Designator::Field {
            name: first,
            span: self.span_since(first_start),
        });
        loop {
            match self.tokens.peek_val() {
                Some(Token::Punctuator(Punctuator::Dot)) => {
                    let dot_start = self.mark();
                    self.tokens.advance();
                    let name = match self.tokens.advance().map(|located| located.val) {
                        Some(Token::Identifier(name)) => name,
                        other => {
                            let span = self.peek_span();
                            return self.fail(span, format!("expected field name, got {other:?}"));
                        }
                    };
                    designators.push(Designator::Field {
                        name,
                        span: self.span_since(dot_start),
                    });
                }
                Some(Token::Punctuator(Punctuator::LBracket)) => {
                    let open = self.mark();
                    self.tokens.advance();
                    let index = self.parse_assignment_expression();
                    let index = self.checked_expr(&index);
                    self.expect_punct(Punctuator::RBracket);
                    designators.push(Designator::Index {
                        index,
                        span: self.span_since(open),
                    });
                }
                _ => break,
            }
        }
        if designators.is_empty() {
            let span = self.peek_span();
            return self.fail(span, "expected member designator in `offsetof`");
        }
        designators
    }
    /// A GNU statement expression `({ block-items })` with `(` still
    /// unconsumed: like a compound statement (own scope, real
    /// declarations, checked inline) whose value is the last
    /// expression statement, if any (`void` otherwise). `return`
    /// inside checks against the enclosing function. Outside a
    /// function body there is no meaning for one (like GCC, which
    /// rejects braced groups outside functions).
    fn parse_statement_expression(&mut self) -> Expr {
        let start = self.mark();
        if self.function_return.is_none() {
            return self.fail(self.peek_span(), "braced-group within expression allowed only inside a function");
        }
        self.expect_punct(Punctuator::LParen);
        self.expect_punct(Punctuator::LBrace);
        self.sema.push_scope();
        self.label_scopes.push(HashMap::new());
        let ret = self.function_return.clone().unwrap_or(Ty::Void);
        let mut items = Vec::new();
        loop {
            if self.peek_pragma_operator() {
                self.skip_pragma_operator();
                continue;
            }
            match self.tokens.peek_val() {
                Some(Token::Punctuator(Punctuator::RBrace)) => break,
                None => {
                    let span = self.span_since(start);
                    return self.fail(span, "unterminated statement expression");
                }
                _ => items.push(self.parse_block_item(&ret)),
            }
        }
        self.expect_punct(Punctuator::RBrace);
        self.sema.pop_scope();
        self.label_scopes.pop();
        self.expect_punct(Punctuator::RParen);
        Expr::StmtExpr {
            items,
            span: self.span_since(start),
        }
    }
    /// Primary: identifiers, constants, adjacent string literals
    /// (concatenated, C17 §6.4.5p4), `( expression )`, `_Generic`
    /// selections, GNU statement expressions `({ ... })`, and the
    /// function-name identifiers (`__func__` and friends, expanding to
    /// the enclosing function's name). `( expression )` yields the
    /// inner expression untouched, so grouping parens fall outside
    /// every span.
    fn parse_primary_expression(&mut self) -> Expr {
        let start = self.mark();
        // `__extension__` prefixes any expression in GNU C (a no-op
        // here, like everywhere else it is skipped).
        if self.peek_is_gnu("__extension__") {
            self.tokens.advance();
        }
        // `_Generic` lexes as an identifier, so match its spelling
        // before the identifier arm below.
        if matches!(self.tokens.peek_val(), Some(Token::Identifier(sym)) if self.intern.resolve(*sym) == Some("_Generic")) {
            return self.parse_generic_selection();
        }
        // C23 `true`/`false` have type `bool`; spelling them as a cast
        // keeps `sizeof(true) == 1` and `_Generic` selecting `bool`.
        let bool_value = if self.peek_c23_keyword("true") {
            Some(true)
        } else if self.peek_c23_keyword("false") {
            Some(false)
        } else {
            None
        };
        if let Some(value) = bool_value {
            self.tokens.advance();
            let lit = IntLiteral {
                spelling: self.intern.get_or_intern(if value { "1" } else { "0" }),
                radix: Radix::Dec,
                suffix: IntSuffix::None,
                imag: false,
            };
            let span = self.span_since(start);
            return Expr::Cast {
                ty: Ty::Bool,
                expr: Box::new(Expr::Int { lit, span }),
                span,
            };
        }
        // C23 `nullptr` is a null pointer constant of type `nullptr_t`.
        // Modeling it as `(void *)0` gives it pointer size and makes
        // `<stddef.h>`'s `typeof(nullptr) nullptr_t` work; a distinct
        // `nullptr_t` is only needed where `_Generic` observes it.
        if self.peek_c23_keyword("nullptr") {
            self.tokens.advance();
            let lit = IntLiteral {
                spelling: self.intern.get_or_intern("0"),
                radix: Radix::Dec,
                suffix: IntSuffix::None,
                imag: false,
            };
            let span = self.span_since(start);
            let ty = Ty::Pointer {
                qualifiers: Vec::new(),
                size: PtrSize::Default,
                pointee: Box::new(Ty::Void),
            };
            return Expr::Cast {
                ty,
                expr: Box::new(Expr::Int { lit, span }),
                span,
            };
        }
        // `__func__`, `__FUNCTION__`, and (in C) `__PRETTY_FUNCTION__`
        // expand to the enclosing function's name as a string literal;
        // outside a function they stay identifiers (and fail lookup,
        // like GCC rejects them there).
        if let Some(Token::Identifier(sym)) = self.tokens.peek_val() {
            let predefined = matches!(self.intern.resolve(*sym), Some("__func__" | "__FUNCTION__" | "__PRETTY_FUNCTION__"));
            if predefined {
                if let Some(func) = self.function_name {
                    let text = self.intern.resolve(func).unwrap_or("?").to_string();
                    let spelling = self.intern.get_or_intern(format!("\"{text}\""));
                    self.tokens.advance();
                    return Expr::Str {
                        parts: vec![StringLiteral {
                            spelling,
                            prefix: StringPrefix::None,
                        }],
                        span: self.span_since(start),
                    };
                }
            }
        }
        // A statement expression opens `({` — unambiguous, since
        // neither casts, calls, nor groupings start that way.
        if matches!(self.tokens.peek_val(), Some(Token::Punctuator(Punctuator::LParen)))
            && matches!(self.tokens.peek_nth(1).map(|t| &t.val), Some(Token::Punctuator(Punctuator::LBrace)))
        {
            return self.parse_statement_expression();
        }
        match self.tokens.peek_val() {
            Some(Token::Identifier(_)) => {
                let (name, span) = match self.tokens.advance().map(|located| located.val) {
                    Some(Token::Identifier(name)) => (name, self.tokens.prev().map(|t| t.span()).unwrap_or(Span::dummy())),
                    _ => unreachable!("identifier peeked"),
                };
                Expr::Ident { name, span }
            }
            Some(Token::Constant(_)) => {
                let (constant, span) = match self.tokens.advance().map(|located| located.val) {
                    Some(Token::Constant(constant)) => (constant, self.tokens.prev().map(|t| t.span()).unwrap_or(Span::dummy())),
                    _ => unreachable!("constant peeked"),
                };
                match constant {
                    crate::lexer::Constant::Int(lit) => Expr::Int { lit, span },
                    crate::lexer::Constant::Float(lit) => Expr::Float { lit, span },
                    crate::lexer::Constant::Char(lit) => Expr::Char { lit, span },
                }
            }
            Some(Token::StringLiteral(_)) => {
                let start = self.mark();
                let mut parts = Vec::new();
                while let Some(Token::StringLiteral(lit)) = self.tokens.peek_val() {
                    parts.push(*lit);
                    self.tokens.advance();
                }
                Expr::Str {
                    parts,
                    span: self.span_since(start),
                }
            }
            Some(Token::Punctuator(Punctuator::LParen)) => {
                self.tokens.advance();
                let expr = self.parse_expression();
                self.expect_punct(Punctuator::RParen);
                expr
            }
            _ => {
                let span = self.peek_span();
                return self.fail(span, format!("expected expression, got {:?}", self.tokens.peek().map(|located| &located.val)));
            }
        }
    }

    pub fn spec_to_ty(&self, specs: &DeclarationSpecifiers, decl: &Declarator) -> Ty {
        if self.failed() {
            return Ty::recover();
        }
        let span = specs.span().merge(decl.span());
        let base = self.unwrap_diag(resolve_type_specifiers(&specs.type_specifier, span));
        // Base qualifiers (`const int`) wrap the base type; declarator
        // operators build outside them, so `const int *p` is a pointer
        // to const int. `restrict` in the specifiers must land on a
        // pointer (C17 §6.7.3.1p2): `restrict int *p` qualifies the
        // pointee and is rejected, while `typedef int *P; restrict P p`
        // merges into the pointer and stays legal.
        let mut ty = Ty::qualified(specs.type_qualifier.clone(), base);
        let mut d = decl;
        loop {
            for ptr in &d.pointer {
                ty = Ty::Pointer {
                    qualifiers: ptr.qualifiers.clone(),
                    size: ptr.size,
                    pointee: Box::new(ty),
                };
            }
            for suf in d.direct.suffixes.iter().rev() {
                match suf {
                    DirectSuffix::Array(a) => {
                        // An array element shall not be a function type
                        // (C17 §6.7.6.1p1).
                        if matches!(ty, Ty::Func { .. }) {
                            return self.fail(span, "array of functions");
                        }
                        ty = Ty::Array {
                            elem: Box::new(ty),
                            quals: a.qualifiers.clone(),
                            // Bounds arrive already folded to a count
                            // (`Fixed`) or a VLA slot (`Vla`, indexing
                            // `Sema::vla_bounds`); no expression — and no
                            // source position beyond that indirection —
                            // reaches into `Ty`.
                            bound: a.bound,
                        };
                    }
                    DirectSuffix::Function(f) => {
                        // A function declarator shall not specify a
                        // function or array return type (C17 §6.7.6.3p1).
                        if matches!(ty, Ty::Func { .. }) {
                            return self.fail(span, "function returning function");
                        }
                        if matches!(ty, Ty::Array { .. }) {
                            return self.fail(span, "function returning array");
                        }
                        ty = Ty::Func {
                            ret: Box::new(ty),
                            params: f.params.iter().map(|p| p.ty.clone()).collect(),
                            variadic: f.variadic,
                            prototyped: f.has_params && !f.is_old_style,
                        };
                    }
                }
            }
            match &d.direct.base {
                DirectBase::Ident(_) | DirectBase::Abstract => {
                    if specs.type_qualifier.iter().any(|q| matches!(q, TypeQualifier::Restrict))
                        && !matches!(&ty, Ty::Pointer { qualifiers, .. } if qualifiers.contains(&TypeQualifier::Restrict))
                    {
                        return self.fail(span, "invalid use of 'restrict'");
                    }
                    return ty;
                }
                DirectBase::Paren(inner) => d = inner,
            }
        }
    }
}
#[derive(Debug)]
pub struct TranslationUnit {
    pub external_declarations: Vec<ExternalDeclaration>,
    pub span: Span,
}

/// What a statement sits in, for `break`/`continue`/`case` placement.
/// Loops and switches track separately: a `case` binds the nearest
/// enclosing `switch` even through a loop (Duff's device), while a
/// `break` binds whichever (loop or `switch`) is nearest. Labels never
/// change either context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FlowCtx {
    in_loop: bool,
    in_switch: bool,
}

impl FlowCtx {
    const NONE: FlowCtx = FlowCtx { in_loop: false, in_switch: false };
    fn in_loop(self) -> FlowCtx {
        FlowCtx {
            in_loop: true,
            in_switch: self.in_switch,
        }
    }
    fn in_switch(self) -> FlowCtx {
        FlowCtx {
            in_loop: self.in_loop,
            in_switch: true,
        }
    }
}

/// Jump-into-VLA-scope tracking for one label scope (a function body
/// or a GNU statement expression): block scopes, VLA objects, labels,
/// gotos, and `switch` dispatches in walk order, checked at the end
/// (see `check_function_body`). Jumping from outside a variably
/// modified object's scope to inside it — past its declaration — is a
/// constraint violation (C11 §6.8.6.1p1); without the diagnostic the
/// backend would free storage that never allocated.
struct FlowVla {
    /// Parent scope per scope id; scope 0 is the walk root.
    parents: Vec<Option<usize>>,
    /// VLA object declarations: (scope, sequence) in walk order.
    vlas: Vec<(usize, u64)>,
    /// Labels by name: (scope, sequence). Duplicates are rejected
    /// during the walk, like the existing `labels` list.
    labels: HashMap<SymbolU32, (usize, u64)>,
    /// Gotos: target name and sequence, resolved at the end.
    gotos: Vec<(SymbolU32, u64, Span)>,
    /// Per-`switch` dispatch: sequence at the dispatch plus each
    /// `case`/`default` position (scope, sequence, span), checked when
    /// the switch walk completes (the dispatch jumps over anything
    /// declared before a case).
    switches: Vec<(u64, Vec<(usize, u64, Span)>)>,
    seq: u64,
}

impl FlowVla {
    fn new() -> FlowVla {
        FlowVla {
            parents: vec![None],
            vlas: Vec::new(),
            labels: HashMap::new(),
            gotos: Vec::new(),
            switches: Vec::new(),
            seq: 0,
        }
    }

    /// A nested block scope (`Compound`, `for` scope, statement
    /// expression items walk in their own tracker).
    fn alloc_scope(&mut self, parent: usize) -> usize {
        let id = self.parents.len();
        self.parents.push(Some(parent));
        id
    }

    fn bump(&mut self) -> u64 {
        let seq = self.seq;
        self.seq += 1;
        seq
    }

    fn record_vla(&mut self, scope: usize) {
        let seq = self.bump();
        self.vlas.push((scope, seq));
    }

    fn record_label(&mut self, name: SymbolU32, scope: usize) {
        let seq = self.bump();
        self.labels.insert(name, (scope, seq));
    }

    fn record_goto(&mut self, name: SymbolU32, span: Span) {
        let seq = self.bump();
        self.gotos.push((name, seq, span));
    }

    /// Whether scope `anc` is `desc` or an ancestor of it.
    fn encloses(&self, anc: usize, desc: usize) -> bool {
        let mut cur = Some(desc);
        while let Some(id) = cur {
            if id == anc {
                return true;
            }
            cur = self.parents[id];
        }
        false
    }
}

/// Decoded GNU attributes: the only effects the frontend models
/// (`packed`, which drops member alignment to 1, and `aligned(n)`, a
/// minimum alignment). Every other attribute validates structurally
/// in `Parser::parse_gnu_attribute_list` and is ignored — the frontend
/// has no diagnostics or codegen to consume it.
#[derive(Debug, Default, Clone)]
pub(crate) struct GnuAttributes {
    pub packed: bool,
    pub aligned: Option<u64>,
    /// `__attribute__((noinline))`: hard opt-out of call-site inlining.
    pub noinline: bool,
    /// `__attribute__((constructor[(prio)]))`: startup hook priority
    /// (`None` = absent; bare attributes are 65535, like GCC).
    pub constructor: Option<u64>,
    /// `__attribute__((destructor[(prio)]))`: shutdown hook priority.
    pub destructor: Option<u64>,
    /// `__attribute__((weak))`: emit as a weak symbol (ELF `STB_WEAK`),
    /// so duplicate definitions across TUs and archives link instead
    /// of erroring. EDG marks every COMDAT-group entity this way.
    pub weak: bool,
    /// `__attribute__((alias("name")))`: the declared entity is the
    /// named one's storage; the spelling is resolved to the target's
    /// declaration when the declaration is registered.
    pub alias: Option<SymbolU32>,
    /// `__attribute__((naked))`: the function has no compiler-generated
    /// prologue/epilogue; its body must be basic asm statements, which
    /// the backend emits as a whole function.
    pub naked: bool,
    /// `__attribute__((async))` (glcc extension): the function compiles
    /// as a stackless coroutine that returns a `_Future(T)` handle.
    pub is_async: bool,
    /// `__attribute__((section("name")))`: place the object (or a naked
    /// function) in that section; the linker script decides where the
    /// section lands.
    pub section: Option<SymbolU32>,
    /// `__asm__("name")` on a declarator: the emitted assembler symbol.
    /// The C spelling stays the declaration's lookup key. EDG labels
    /// renamed C++ entities this way, e.g. the C library overloads it
    /// emits as `_Z6strchrPci(...) __asm__("strchr")`.
    pub asm_label: Option<SymbolU32>,
}

impl GnuAttributes {
    fn merge(&mut self, other: GnuAttributes) {
        self.packed |= other.packed;
        self.aligned = match (self.aligned, other.aligned) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        self.constructor = self.constructor.or(other.constructor);
        self.destructor = self.destructor.or(other.destructor);
        self.weak |= other.weak;
        self.alias = self.alias.or(other.alias);
        self.naked |= other.naked;
        self.noinline |= other.noinline;
        self.is_async |= other.is_async;
        self.section = self.section.or(other.section);
        self.asm_label = self.asm_label.or(other.asm_label);
    }
}

impl HasSpan for TranslationUnit {
    fn span(&self) -> Span {
        self.span
    }
}
#[derive(Debug)]
pub enum ExternalDeclaration {
    Var(DeclId),
    Func(DeclId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]

pub enum StorageClass {
    Typedef,
    Extern,
    Static,
    Auto,
    Register,
}
#[derive(Debug)]

pub enum TypeSpecifier {
    Void,
    Char,
    Short,
    Int,
    Long,
    Float,
    Double,
    Signed,
    Unsigned,
    _Bool,
    _Complex,
    /// GCC's `__int128` keyword: combines only with a sign to `Int128`
    /// / `UnsignedInt128` (see `resolve_type_specifiers`).
    Int128,
    // A struct/union specifier, identified by its tag: the members and
    // completeness live behind the id in `Sema::structs`.
    Struct(StructId),
    // An enum specifier, identified by its tag: the enumerators and
    // completeness live behind the id in `Sema::enums`.
    Enum(EnumId),
    // A builtin type keyword (`__builtin_va_list`, `__float128`):
    // a whole type by itself, carrying the `Ty` it names.
    Builtin(Ty),
    // C11 `_Atomic(type-name)`: atomic types have the size, alignment,
    // and representation of their unqualified base and this compiler
    // models no atomic operations, so the qualifier is erased at
    // resolution and only the operand type survives.
    Atomic(Ty),
    // A typedef name standing in as the type specifier, carrying the type
    // it aliases.
    TypedefName(Ty),
}

/// `%l[name]` references in an asm template, skipping `%%` escapes: the
/// names must appear in the statement's `asm goto` label list.
fn asm_label_refs(template: &str) -> Vec<String> {
    let mut refs = Vec::new();
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            continue;
        }
        if chars.peek() == Some(&'%') {
            chars.next();
            continue;
        }
        if chars.peek() != Some(&'l') {
            continue;
        }
        chars.next();
        if chars.next() != Some('[') {
            continue;
        }
        let mut name = String::new();
        for c in chars.by_ref() {
            if c == ']' {
                break;
            }
            name.push(c);
        }
        refs.push(name);
    }
    refs
}

/// Append `s` to `out` with JSON string escaping (`"`/`\`/control
/// characters); non-ASCII passes through as UTF-8.
fn push_escaped(out: &mut String, s: &str) {
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
}

/// Pointer to `pointee`, unqualified: the implicit-declaration
/// prototypes name parameter pointers without qualifiers and let
/// argument conversion add them.
fn ptr(pointee: Ty) -> Ty {
    Ty::Pointer {
        qualifiers: vec![],
        size: PtrSize::Default,
        pointee: Box::new(pointee),
    }
}

/// Whether a `[*]` bound appears anywhere in a type without crossing a
/// function type. A `[*]` is only legal in prototype scope (verified
/// against gcc): nested prototypes are their own scope, so the search
/// stops at `Ty::Func` — `int f(int g(int [*])) {}` is fine — while
/// anything else (`int a[*]`, `int a[2][*]`, `int (*p)[*]`) in a
/// definition's own parameters is rejected.
fn ty_has_bare_star(ty: &Ty) -> bool {
    match ty {
        Ty::Array {
            bound: Some(ArrayBound::Star), ..
        } => true,
        Ty::Array { elem, .. } => ty_has_bare_star(elem),
        Ty::Pointer { pointee, .. } => ty_has_bare_star(pointee),
        Ty::Func { .. } => false,
        _ => false,
    }
}

// Binding power of each binary operator for precedence climbing (tighter
// binds higher): `||` 10 through `*` 100, mirroring C17 §6.5. Comma is
// handled separately in `parse_expression`.
fn binary_binding_power(punct: Punctuator) -> Option<u8> {
    Some(match punct {
        Punctuator::PipePipe => 10,
        Punctuator::AmpAmp => 20,
        Punctuator::Pipe => 30,
        Punctuator::Caret => 40,
        Punctuator::Amp => 50,
        Punctuator::EqEq | Punctuator::BangEq => 60,
        Punctuator::Less | Punctuator::LessEq | Punctuator::Greater | Punctuator::GreaterEq => 70,
        Punctuator::LessLess | Punctuator::GreaterGreater => 80,
        Punctuator::Plus | Punctuator::Minus => 90,
        Punctuator::Star | Punctuator::Slash | Punctuator::Percent => 100,
        _ => return None,
    })
}

// Combine one declaration's type specifiers into a single type (C17 §6.7.2).
// Order-independent: `unsigned long int` and `int unsigned long` agree.
// Anything outside the standard combinations fails with the declaration's
// span attached.
pub fn resolve_type_specifiers(specs: &[TypeSpecifier], span: Span) -> Result<Ty, CompileError> {
    // Local shorthand: every complaint carries the specifiers' span.
    macro_rules! bail {
        ($($arg:tt)*) => {
            return Err(CompileError::new(format!($($arg)*), span))
        };
    }
    if specs.is_empty() {
        bail!("no type specifier");
    }
    // A struct/union or typedef-name specifier is a whole type by itself:
    // it neither combines with the primitive specifiers counted below nor
    // with another such specifier (C17 §6.7.2p1).
    let mut tag = None;
    let mut etag = None;
    let mut builtin = None;
    let mut alias = None;
    let mut atomic = None;
    let (mut n_signed, mut n_unsigned, mut n_short, mut n_long) = (0u8, 0u8, 0u8, 0u8);
    let (mut n_char, mut n_int, mut n_float, mut n_double) = (0u8, 0u8, 0u8, 0u8);
    let (mut n_bool, mut n_complex, mut n_void) = (0u8, 0u8, 0u8);
    let mut n_int128 = 0u8;
    for s in specs {
        match s {
            TypeSpecifier::Struct(id) => {
                if tag.is_some() {
                    bail!("duplicate struct/union specifier");
                }
                tag = Some(*id);
            }
            TypeSpecifier::Enum(id) => {
                if etag.is_some() {
                    bail!("duplicate enum specifier");
                }
                etag = Some(*id);
            }
            TypeSpecifier::Builtin(ty) => builtin = Some(ty),
            TypeSpecifier::TypedefName(ty) => alias = Some(ty),
            TypeSpecifier::Atomic(ty) => atomic = Some(ty),
            TypeSpecifier::Signed => n_signed += 1,
            TypeSpecifier::Unsigned => n_unsigned += 1,
            TypeSpecifier::Short => n_short += 1,
            TypeSpecifier::Long => n_long += 1,
            TypeSpecifier::Char => n_char += 1,
            TypeSpecifier::Int => n_int += 1,
            TypeSpecifier::Float => n_float += 1,
            TypeSpecifier::Double => n_double += 1,
            TypeSpecifier::_Bool => n_bool += 1,
            TypeSpecifier::_Complex => n_complex += 1,
            TypeSpecifier::Void => n_void += 1,
            TypeSpecifier::Int128 => n_int128 += 1,
        }
    }
    if let Some(id) = tag {
        if specs.len() != 1 {
            bail!("`struct`/`union` with other type specifiers");
        }
        return Ok(Ty::Struct(id));
    }
    if let Some(id) = etag {
        if specs.len() != 1 {
            bail!("`enum` with other type specifiers");
        }
        return Ok(Ty::Enum(id));
    }
    if let Some(ty) = builtin {
        if specs.len() != 1 {
            bail!("builtin type with other type specifiers");
        }
        return Ok(ty.clone());
    }
    if let Some(ty) = alias {
        if specs.len() != 1 {
            bail!("typedef name with other type specifiers");
        }
        return Ok(ty.clone());
    }
    if let Some(ty) = atomic {
        if specs.len() != 1 {
            bail!("`_Atomic` with other type specifiers");
        }
        return Ok(ty.clone());
    }
    if n_signed == 1 && n_unsigned == 1 {
        bail!("both `signed` and `unsigned` in declaration specifiers");
    }
    if n_signed > 1 || n_unsigned > 1 {
        bail!("duplicate sign specifier");
    }
    if n_short > 1 {
        bail!("duplicate `short`");
    }
    if n_long > 2 {
        bail!("too many `long` specifiers");
    }
    if n_char > 1 || n_int > 1 || n_float > 1 || n_double > 1 || n_bool > 1 || n_void > 1 {
        bail!("duplicate type specifier");
    }
    if n_complex > 1 {
        bail!("duplicate `_Complex`");
    }
    // GCC's 128-bit integers combine only with a sign (`__int128`,
    // `signed __int128`, `unsigned __int128`).
    if n_int128 > 0 {
        if n_int128 > 1 || n_short > 0 || n_long > 0 || n_char > 0 || n_int > 0 || n_float > 0 || n_double > 0 || n_bool > 0 || n_complex > 0 || n_void > 0 {
            bail!("`__int128` with other type specifiers");
        }
        return Ok(if n_unsigned == 1 { Ty::UnsignedInt128 } else { Ty::Int128 });
    }
    let unsigned = n_unsigned == 1;
    if n_void == 1 {
        if specs.len() != 1 {
            bail!("`void` with other type specifiers");
        }
        return Ok(Ty::Void);
    }
    if n_bool == 1 {
        if specs.len() != 1 {
            bail!("`_Bool` with other type specifiers");
        }
        if n_complex == 1 {
            bail!("both `complex` and `_Bool` in declaration specifiers");
        }
        return Ok(Ty::Bool);
    }
    if n_char == 1 {
        if n_short == 1 || n_long > 0 || n_int == 1 || n_float == 1 || n_double == 1 {
            bail!("`char` with other type specifiers");
        }
        let base = if n_signed == 1 {
            Ty::SignedChar
        } else if unsigned {
            Ty::UnsignedChar
        } else {
            Ty::Char
        };
        return Ok(if n_complex == 1 { Ty::Complex(Box::new(base)) } else { base });
    }
    if n_float == 1 {
        if n_signed == 1 || unsigned || n_short == 1 || n_long > 0 || n_int == 1 || n_double == 1 {
            bail!("`float` with other type specifiers");
        }
        return Ok(if n_complex == 1 { Ty::Complex(Box::new(Ty::Float)) } else { Ty::Float });
    }
    if n_double == 1 {
        if n_signed == 1 || unsigned || n_short == 1 || n_int == 1 || n_long > 1 {
            bail!("invalid `double` combination");
        }
        if n_complex == 1 {
            return Ok(Ty::Complex(Box::new(if n_long == 1 { Ty::LongDouble } else { Ty::Double })));
        }
        return Ok(if n_long == 1 { Ty::LongDouble } else { Ty::Double });
    }
    // Leftover: `[signed|unsigned] [short|long[long]] [int]`, each
    // optionally complex (bare `_Complex` is `double _Complex`).
    let base = match (n_short == 1, n_long) {
        (false, 0) => {
            if n_int == 0 && n_signed == 0 && !unsigned {
                if n_complex == 1 {
                    return Ok(Ty::Complex(Box::new(Ty::Double)));
                }
                bail!("no type specifier");
            }
            if unsigned { Ty::UnsignedInt } else { Ty::Int }
        }
        (true, 0) => {
            if unsigned {
                Ty::UnsignedShort
            } else {
                Ty::Short
            }
        }
        (false, 1) => {
            if unsigned {
                Ty::UnsignedLong
            } else {
                Ty::Long
            }
        }
        (false, 2) => {
            if unsigned {
                Ty::UnsignedLongLong
            } else {
                Ty::LongLong
            }
        }
        _ => bail!("`short` with `long`"),
    };
    Ok(if n_complex == 1 { Ty::Complex(Box::new(base)) } else { base })
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeQualifier {
    Const,
    Restrict,
    Volatile,
}
/// MSVC's pointer-size attributes (`__ptr32`/`__ptr64`). They select the
/// width of the pointer itself, not of its pointee.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PtrSize {
    #[default]
    Default,
    Ptr32,
    Ptr64,
}

impl PtrSize {
    /// Width in bits on a target whose ordinary pointers are `default` bits.
    pub fn bits(self, default: u32) -> u32 {
        match self {
            PtrSize::Default => default,
            PtrSize::Ptr32 => 32,
            PtrSize::Ptr64 => 64,
        }
    }
}
#[derive(Debug, Default)]
pub struct DeclarationSpecifiers {
    class: Vec<StorageClass>,
    pub(crate) type_specifier: Vec<TypeSpecifier>,
    type_qualifier: Vec<TypeQualifier>,
    function_specifier: usize, // only option is Inline
    /// C11 `_Noreturn` function specifier: like `inline`, only a hint
    /// and only valid on functions, so it is tracked separately and
    /// otherwise ignored.
    noreturn: bool,
    /// Thread-local storage (`_Thread_local`, `__thread`,
    /// `thread_local`): combines with `static`/`extern` like GCC.
    pub(crate) thread_local: bool,
    /// GNU attributes interleaved with the specifiers
    /// (`__attribute__((weak)) int x;`): declaration-level effects
    /// (`weak` here) apply to every declarator in the list.
    pub(crate) attrs: GnuAttributes,
    pub span: Span,
}

impl HasSpan for DeclarationSpecifiers {
    fn span(&self) -> Span {
        self.span
    }
}
impl DeclarationSpecifiers {
    pub fn is_extern(&self) -> bool {
        self.class.contains(&StorageClass::Extern)
    }
    pub fn is_typedef(&self) -> bool {
        self.class.contains(&StorageClass::Typedef)
    }
    /// Whether an `inline` function specifier was present (`inline`,
    /// `__inline__`, `__inline`, `__forceinline` all count; duplicates
    /// stack in `function_specifier`).
    pub fn is_inline(&self) -> bool {
        self.function_specifier > 0
    }
    /// Whether a `_Noreturn` function specifier was present.
    pub fn is_noreturn(&self) -> bool {
        self.noreturn
    }
    pub fn has_storage(&self, sc: StorageClass) -> bool {
        self.class.contains(&sc)
    }
    pub fn has_storage_class(&self) -> bool {
        !self.class.is_empty()
    }
}

#[derive(Debug, Default)]
pub struct Declarator {
    pub(crate) pointer: Vec<Pointer>,
    pub(crate) direct: DirectDeclarator,
    pub span: Span,
}
impl Declarator {
    pub fn name(&self) -> Option<SymbolU32> {
        match &self.direct.base {
            DirectBase::Ident(symbol_u32) => Some(*symbol_u32),
            DirectBase::Paren(declarator) => declarator.name(),
            DirectBase::Abstract => None,
        }
    }
}

impl HasSpan for Declarator {
    fn span(&self) -> Span {
        self.span
    }
}
#[derive(Debug, Default)]
pub struct Pointer {
    pub(crate) qualifiers: Vec<TypeQualifier>,
    pub(crate) size: PtrSize,
    pub span: Span,
}

impl HasSpan for Pointer {
    fn span(&self) -> Span {
        self.span
    }
}
#[derive(Debug, Default)]
pub struct DirectDeclarator {
    pub(crate) base: DirectBase,
    pub(crate) suffixes: Vec<DirectSuffix>,
    pub span: Span,
}

impl HasSpan for DirectDeclarator {
    fn span(&self) -> Span {
        self.span
    }
}
#[derive(Debug, Default)]
pub enum DirectBase {
    Ident(SymbolU32),
    Paren(Box<Declarator>),
    #[default]
    Abstract,
}
#[derive(Debug)]
pub enum DirectSuffix {
    Array(ArrayDeclarator),
    Function(FunctionDeclarator),
}
#[derive(Debug, Default)]
pub struct ArrayDeclarator {
    pub(crate) qualifiers: Vec<TypeQualifier>,
    pub(crate) is_static: bool,
    /// `None` for `[]`, `Star` for `[*]`, or the bound expression.
    pub(crate) bound: Option<ArrayBound>,
    pub span: Span,
}

/// An array bound.
///
/// Constant bounds fold to a count at the point the parser checks
/// them, so fixed arrays carry no expression. That keeps the common
/// `Ty::Array` free of source positions, which in turn keeps `Ty`'s
/// derived `PartialEq` honest: two arrays are compatible when their
/// bounds *are* equal, not when their bound expressions happen to
/// have identical spelling.
///
/// `[]` and `[*]` leave `bound` as `None` / `Star` here; `[]` in a
/// definition completes from its initializer later
/// (`Parser::complete_array_bound`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayBound {
    /// `[*]` (C17 §6.7.6.2p2): variable length, extent not deduced.
    Star,
    /// A bound that folded to a constant element count.
    Fixed(u64),
    /// A bound that did not fold: a VLA. The payload indexes the
    /// checked bound expression in `Sema::vla_bounds`, evaluated at
    /// runtime wherever the type's size is needed (allocation,
    /// `sizeof`, strides). Compatibility ignores the payload (see
    /// `sema::bounds_compatible`): two VLAs always match.
    Vla(u32),
}

impl HasSpan for ArrayDeclarator {
    fn span(&self) -> Span {
        self.span
    }
}
#[derive(Debug, Default)]
pub struct FunctionDeclarator {
    // Whether any list was parsed: `f()` has none, so — unlike
    // `f(void)` or an identifier-list — it supplies no parameter
    // information (C17 §6.7.6.3p14).
    pub(crate) has_params: bool,
    // The parsed parameter-type-list. Empty for `f()`, for `f(void)`,
    // and for identifier-lists — old-style parameter types only a
    // function definition's declaration list supplies.
    pub(crate) params: Vec<Param>,
    pub(crate) variadic: bool,
    // True for an old-style identifier-list (`f(a, b)`): like `f()`
    // it leaves the function unprototyped, even though `has_params`
    // is set.
    pub(crate) is_old_style: bool,
    // The identifier-list names in order (`f(a, b)`): filled only for
    // old-style declarators, consumed by `parse_function_definition`
    // to rewrite K&R definitions into prototyped functions.
    pub(crate) idents: Vec<(SymbolU32, Span)>,
    pub span: Span,
}

impl HasSpan for FunctionDeclarator {
    fn span(&self) -> Span {
        self.span
    }
}

/// One parsed parameter: its spelling (absent for abstract declarators,
/// as in `f(int *)`) and its type after the adjustments of
/// C17 §6.7.6.3p7/p8. `Ty::Func` keeps only the types, so `int f(int)`
/// and `int f(int x)` stay the same type.
#[derive(Debug)]
pub struct Param {
    pub(crate) name: Option<SymbolU32>,
    pub(crate) ty: Ty,
    /// The parameter's type in the body when it differs from the
    /// signature's (K&R definitions): the ABI passes the promoted
    /// value, the body sees the declared type and the prologue
    /// converts. `None` means `ty` is both.
    pub(crate) body_ty: Option<Ty>,
    /// A variable-length bound (`int a[n]`), kept past the decay to a
    /// pointer: the bound evaluates once on entry to the definition's
    /// body (C17 §6.7.6.2p6), so `int a[i++]` steps `i` per call.
    pub(crate) vla_bound: Option<u32>,
    pub span: Span,
    /// Whether the parameter was declared with a `[*]` bound. Recorded
    /// before array-to-pointer decay erases it: `[*]` is legal in
    /// prototypes but not in a definition's own parameter list, and
    /// `parse_function_definition` checks this flag.
    pub(crate) star: bool,
    /// The prototype-scope binding for a named parameter, if declared
    /// there. A definition re-declares its parameters in function
    /// scope, so VLA bounds checked against this binding are remapped
    /// to the body's binding (see `parse_function_definition`).
    pub(crate) proto: Option<DeclId>,
}

impl HasSpan for Param {
    fn span(&self) -> Span {
        self.span
    }
}

// ---- expressions ----
//
// Full C17 expression grammar (§6.5) with spans throughout: precedence
// climbing for binaries, right-associative assignment and ternary,
// postfix/call/subscript/member, casts and `sizeof` (both forms).
// Comma is the lowest-precedence binary operator, so `parse_expression`
// covers statement/condition positions while `parse_assignment_expression`
// (and below) serve positions where commas separate instead.

/// A parsed expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Ident {
        name: SymbolU32,
        span: Span,
    },
    Int {
        lit: crate::lexer::IntLiteral,
        span: Span,
    },
    /// GNU `x ?: y` (elvis): the condition's own value is the result
    /// when nonzero, otherwise `y`. Evaluates the condition once.
    Elvis {
        cond: Box<Expr>,
        els: Box<Expr>,
        span: Span,
    },
    Float {
        lit: crate::lexer::FloatLiteral,
        span: Span,
    },
    Char {
        lit: crate::lexer::CharLiteral,
        span: Span,
    },
    /// Adjacent string literals, concatenated per C17 §6.4.5p4.
    Str {
        parts: Vec<crate::lexer::StringLiteral>,
        span: Span,
    },
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
        span: Span,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Span,
    },
    Assign {
        op: AssignOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Span,
    },
    Ternary {
        cond: Box<Expr>,
        then: Box<Expr>,
        els: Box<Expr>,
        span: Span,
    },
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
        span: Span,
    },
    Subscript {
        base: Box<Expr>,
        index: Box<Expr>,
        span: Span,
    },
    Member {
        base: Box<Expr>,
        field: SymbolU32,
        arrow: bool,
        span: Span,
    },
    Cast {
        ty: Ty,
        expr: Box<Expr>,
        span: Span,
    },
    /// A C11 generic selection: the controlling expression plus
    /// associations (`type-name: expr`, at most one `default: expr`
    /// with `ty: None`), resolved by the checker.
    Generic {
        ctrl: Box<Expr>,
        assocs: Vec<GenericAssoc>,
        span: Span,
    },
    /// `va_arg(ap, TYPE)`: like a call, but the second operand is a
    /// type-name, so it parses separately (C17 §7.16.1.1).
    VaArg {
        ap: Box<Expr>,
        ty: Ty,
        span: Span,
    },
    /// A GNU statement expression `({ ... })`: checked block items
    /// whose value is the last expression statement, if any.
    StmtExpr {
        items: Vec<BlockItem>,
        span: Span,
    },
    /// `__builtin_offsetof(TYPE, .a.b[i])` (GCC's `offsetof`): the
    /// designators resolve against the type at check time.
    Offsetof {
        ty: Ty,
        designators: Vec<Designator>,
        span: Span,
    },
    CompoundLiteral {
        ty: Ty,
        init: Box<Initializer>,
        span: Span,
    },
    SizeofExpr {
        expr: Box<Expr>,
        span: Span,
    },
    SizeofTy {
        ty: Ty,
        span: Span,
    },
    /// GNU `__alignof__` / C11 `_Alignof`, in expression and type
    /// form: the operand's required alignment, as a `size_t` constant.
    AlignofExpr {
        expr: Box<Expr>,
        span: Span,
    },
    AlignofTy {
        ty: Ty,
        span: Span,
    },
    /// GNU label address `&&label`: a `void *` value usable with
    /// computed `goto *expr` (the address-of-label companion).
    LabelAddr {
        label: SymbolU32,
        span: Span,
    },
}

impl HasSpan for Expr {
    fn span(&self) -> Span {
        match self {
            Expr::Ident { span, .. }
            | Expr::Int { span, .. }
            | Expr::Float { span, .. }
            | Expr::Char { span, .. }
            | Expr::Str { span, .. }
            | Expr::Unary { span, .. }
            | Expr::Binary { span, .. }
            | Expr::Assign { span, .. }
            | Expr::Ternary { span, .. }
            | Expr::Elvis { span, .. }
            | Expr::Call { span, .. }
            | Expr::Subscript { span, .. }
            | Expr::Member { span, .. }
            | Expr::Cast { span, .. }
            | Expr::Generic { span, .. }
            | Expr::Offsetof { span, .. }
            | Expr::StmtExpr { span, .. }
            | Expr::VaArg { span, .. }
            | Expr::CompoundLiteral { span, .. }
            | Expr::SizeofExpr { span, .. }
            | Expr::AlignofExpr { span, .. }
            | Expr::AlignofTy { span, .. }
            | Expr::SizeofTy { span, .. }
            | Expr::LabelAddr { span, .. } => *span,
        }
    }
}

/// Prefix and postfix unary operators. Postfix `++`/`--` live here too:
/// they apply in `parse_postfix_expression` but share the `Unary` node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Pos,
    Neg,
    Not,
    BitNot,
    Deref,
    AddrOf,
    PreInc,
    PreDec,
    PostInc,
    PostDec,
    /// GNU `__real__ x` / `__imag__ x`: the real/imaginary part of a
    /// complex operand, as a modifiable lvalue when the operand is one.
    Real,
    Imag,
}

/// Binary operators, including `Comma` at the lowest precedence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Mul,
    Div,
    Rem,
    Add,
    Sub,
    Shl,
    Shr,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    BitAnd,
    BitXor,
    BitOr,
    And,
    Or,
    Comma,
}

/// Assignment operators: plain `=` plus the ten compound forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignOp {
    Assign,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Shl,
    Shr,
    BitAnd,
    BitXor,
    BitOr,
}

/// An initializer (`= ...`, C17 §6.7.9): a single assignment expression
/// (checked against the object's type) or a brace-enclosed (possibly
/// designated, possibly nested) list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Initializer {
    Expr(CheckedExpr),
    List { items: Vec<InitItem>, span: Span },
}

impl HasSpan for Initializer {
    fn span(&self) -> Span {
        match self {
            Initializer::Expr(expr) => expr.span(),
            Initializer::List { span, .. } => *span,
        }
    }
}

/// One element of a brace-enclosed initializer list: optional
/// designators (`.field`, `[index]`) plus the value, which may itself
/// be a nested list (`.a = {1, 2}`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitItem {
    pub designators: Vec<Designator>,
    pub init: Box<Initializer>,
    pub span: Span,
}

impl HasSpan for InitItem {
    fn span(&self) -> Span {
        self.span
    }
}

/// A single `.field` or `[index]` designator (C17 §6.7.9p1). Index
/// expressions check like any other expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Designator {
    Field {
        name: SymbolU32,
        span: Span,
    },
    /// A hop through an anonymous member: the checker rewrites
    /// designations that name a field inside one (`.b` for
    /// `union { struct { int a, b; }; }`) so the backend can address
    /// members by index without searching.
    AnonField {
        index: u32,
        span: Span,
    },
    Index {
        index: CheckedExpr,
        span: Span,
    },
}

impl HasSpan for Designator {
    fn span(&self) -> Span {
        match self {
            Designator::Field { span, .. } | Designator::AnonField { span, .. } | Designator::Index { span, .. } => *span,
        }
    }
}

/// One `_Generic` association: `type-name : expr`, or `default : expr`
/// (`ty: None`). Checked by `Sema::check_generic`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenericAssoc {
    pub ty: Option<Ty>,
    pub expr: Box<Expr>,
    pub span: Span,
}

impl HasSpan for GenericAssoc {
    fn span(&self) -> Span {
        self.span
    }
}

// ---- function bodies ----
//
// Bodies parse for scope, declaration, and expression/statement
// effects: parameters and block items declare for real, nested `{ ... }`
// blocks nest for real, and statements parse into `Stmt`. The parsed
// body is stored in `Sema::bodies` under the function's `DeclId`;
// initializers live in `Sema::inits` under their object's `DeclId`.

/// A parsed function body: the parameter bindings declared in the
/// function's scope plus the outermost compound statement.
#[derive(Debug, Clone)]
pub struct FuncBody {
    pub span: Span,
    pub params: Vec<DeclId>,
    pub block: Compound,
}

impl HasSpan for FuncBody {
    fn span(&self) -> Span {
        self.span
    }
}

/// `{ block-item-list_opt }`: declarations and statements with their own
/// block scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compound {
    pub span: Span,
    pub items: Vec<BlockItem>,
}

impl HasSpan for Compound {
    fn span(&self) -> Span {
        self.span
    }
}

/// One thing inside `{ ... }`: either a declaration (parsed for real) or
/// a statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockItem {
    Decl { ids: Vec<DeclId>, span: Span },
    Stmt(Stmt),
}

/// The value-producing expression of a GNU statement expression: the
/// last item when it is an expression statement, unwrapping label
/// statements (`({ l: x; })` yields `x`, like GCC; the labels still
/// define their jump targets). `None` leaves the statement expression
/// `void`.
pub fn stmt_expr_value(items: &[BlockItem]) -> Option<&CheckedExpr> {
    let mut stmt = match items.last()? {
        BlockItem::Stmt(stmt) => stmt,
        BlockItem::Decl { .. } => return None,
    };
    loop {
        match stmt {
            Stmt::Expr { expr, .. } => return Some(expr),
            Stmt::Label { stmt: inner, .. } => stmt = inner,
            _ => return None,
        }
    }
}

/// One operand of an `asm` statement: `[name] "constraint" (expr)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsmOperand {
    pub name: Option<SymbolU32>,
    pub constraint: String,
    pub expr: CheckedExpr,
    pub span: Span,
}

impl AsmOperand {
    /// The constraint without its `=`/`+`/`&` direction prefix.
    pub fn constraint_body(&self) -> &str {
        self.constraint.trim_start_matches(['=', '+', '&'])
    }

    /// `"0"`-`"9"` matching constraint: this input shares a location
    /// with output `n`.
    pub fn tie(&self) -> Option<usize> {
        let body = self.constraint_body();
        match body.as_bytes() {
            [digit @ b'0'..=b'9'] => Some((digit - b'0') as usize),
            _ => None,
        }
    }

    /// A read-write operand (`"+r"`): the value enters and leaves.
    pub fn is_inout(&self) -> bool {
        self.constraint.starts_with('+')
    }

    /// An early-clobber operand (`"&"`): it may be written before every
    /// input has been read.
    pub fn is_early(&self) -> bool {
        self.constraint.contains('&')
    }
}

/// A parsed `asm`/`__asm__` statement
/// (`asm [volatile|inline|goto] ( template [: outputs [: inputs [: clobbers
/// [: goto-labels]]]] )`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsmStmt {
    pub template: String,
    pub outputs: Vec<AsmOperand>,
    pub inputs: Vec<AsmOperand>,
    pub clobbers: Vec<String>,
    pub labels: Vec<SymbolU32>,
    pub is_volatile: bool,
    pub is_goto: bool,
    /// Basic asm: the statement has no colon, so the template is not
    /// parsed for operand references and `%` is literal (`%eax`). GCC
    /// splits the two forms at the first colon, which this mirrors;
    /// `asm volatile ("...")` with no colon is still basic.
    pub is_basic: bool,
    pub span: Span,
}

/// Statements (C17 §6.8). Labels carry their sub-statement so nesting
/// stays structural; `case` holds `None` for `default`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    Compound(Compound),
    Empty {
        span: Span,
    },
    Expr {
        expr: CheckedExpr,
        span: Span,
    },
    Asm(AsmStmt),
    Return {
        expr: Option<CheckedExpr>,
        span: Span,
    },
    If {
        cond: CheckedExpr,
        then: Box<Stmt>,
        els: Option<Box<Stmt>>,
        span: Span,
    },
    While {
        cond: CheckedExpr,
        body: Box<Stmt>,
        span: Span,
    },
    DoWhile {
        body: Box<Stmt>,
        cond: CheckedExpr,
        span: Span,
    },
    For {
        init: ForInit,
        cond: Option<CheckedExpr>,
        step: Option<CheckedExpr>,
        body: Box<Stmt>,
        span: Span,
    },
    Switch {
        cond: CheckedExpr,
        body: Box<Stmt>,
        span: Span,
    },
    Break {
        span: Span,
    },
    Continue {
        span: Span,
    },
    Goto {
        label: SymbolU32,
        span: Span,
    },
    /// GNU computed goto (`goto *expr`): the expression yields a label
    /// address (`&&label`) to dispatch on.
    GotoComputed {
        expr: CheckedExpr,
        span: Span,
    },
    Label {
        name: SymbolU32,
        stmt: Box<Stmt>,
        span: Span,
    },
    Case {
        expr: Option<CheckedExpr>,
        /// GNU case range `case lo ... hi:`: the label matches every
        /// value from `lo` through `hi` (an empty range when
        /// `hi < lo`, like GCC).
        hi: Option<CheckedExpr>,
        stmt: Box<Stmt>,
        span: Span,
    },
}

impl HasSpan for Stmt {
    fn span(&self) -> Span {
        match self {
            Stmt::Compound(compound) => compound.span,
            Stmt::Asm(asm) => asm.span,
            Stmt::Empty { span }
            | Stmt::Expr { span, .. }
            | Stmt::Return { span, .. }
            | Stmt::If { span, .. }
            | Stmt::While { span, .. }
            | Stmt::DoWhile { span, .. }
            | Stmt::For { span, .. }
            | Stmt::Switch { span, .. }
            | Stmt::Break { span }
            | Stmt::Continue { span }
            | Stmt::Goto { span, .. }
            | Stmt::GotoComputed { span, .. }
            | Stmt::Label { span, .. }
            | Stmt::Case { span, .. } => *span,
        }
    }
}

/// Whether a statement holds a label — named or `case`. Both can be
/// jumped to from outside, so code under a constant condition must
/// survive when it contains one. An explicit stack keeps the walk off
/// the call stack (statements nest as deeply as the parser allows).
fn stmt_has_label(stmt: &Stmt) -> bool {
    let mut stack = vec![stmt];
    while let Some(stmt) = stack.pop() {
        match stmt {
            Stmt::Label { .. } | Stmt::Case { .. } => return true,
            Stmt::Compound(compound) => stack.extend(compound.items.iter().filter_map(|item| match item {
                BlockItem::Stmt(inner) => Some(inner),
                BlockItem::Decl { .. } => None,
            })),
            Stmt::If { then, els, .. } => {
                stack.push(then);
                if let Some(els) = els {
                    stack.push(els);
                }
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::For { body, .. } | Stmt::Switch { body, .. } => stack.push(body),
            _ => {}
        }
    }
    false
}

/// Fold `if` on a constant condition in place: the untaken branch is
/// dropped unless it holds a label, which `goto` (or a `switch`) can
/// jump to. This lives in its own function, taking only references, so
/// `parse_statement`'s frame stays the size it was — it sits on the
/// stack through the whole nested parse.
fn fold_constant_if(cond: &CheckedExpr, then: &mut Stmt, els: &mut Option<Box<Stmt>>, sema: &Sema, interner: &Interner) {
    match crate::eval::eval_const(cond, sema, interner) {
        // The else arm is never taken.
        Some(crate::eval::Value::Int(n)) if n != 0 => {
            if els.as_ref().is_some_and(|e| !stmt_has_label(e)) {
                *els = None;
            }
        }
        // The then arm is never taken; the else stays where it is.
        Some(crate::eval::Value::Int(_)) => {
            if !stmt_has_label(then) {
                let span = then.span();
                *then = Stmt::Empty { span };
            }
        }
        _ => {}
    }
}

/// A `for` initializer: a declaration (`for (int i = 0; ...)`), an
/// expression, or nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForInit {
    Decl(Vec<DeclId>),
    Expr(CheckedExpr),
    Empty,
}

impl HasSpan for BlockItem {
    fn span(&self) -> Span {
        match self {
            BlockItem::Decl { span, .. } => *span,
            BlockItem::Stmt(stmt) => stmt.span(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    Void,
    Bool,
    Char,
    SignedChar,
    UnsignedChar,
    Short,
    UnsignedShort,
    Int,
    UnsignedInt,
    Long,
    UnsignedLong,
    LongLong,
    UnsignedLongLong,
    /// GCC's 128-bit integers: rank above `long long`, 16 bytes.
    Int128,
    UnsignedInt128,
    Float,
    Double,
    LongDouble,
    /// A complex number (`_Complex`, GNU `__complex__`) over any real
    /// component type: `float`/`double`/`long double` in standard C,
    /// any integer type as a GNU extension (`__complex__ int`).
    /// Twice the component's size, same alignment.
    Complex(Box<Ty>),
    Pointer {
        qualifiers: Vec<TypeQualifier>,
        size: PtrSize,
        pointee: Box<Ty>,
    },
    Func {
        ret: Box<Ty>,
        // Parameter types only — names stay on the declarator, and
        // parameters declared as arrays or functions carry their
        // adjusted type (C17 §6.7.6.3p7/p8). `f()` and old-style
        // identifier-lists are unprototyped (`prototyped: false`,
        // C17 §6.7.6.3p14); `f(void)` is prototyped with an empty
        // list. An unprototyped declaration is compatible with any
        // prototype, and the composite takes the prototyped side —
        // so `int f(); int f(int);` merges (see `Sema::declare`).
        params: Vec<Ty>,
        variadic: bool,
        prototyped: bool,
    },
    Array {
        elem: Box<Ty>,
        // Qualifiers from inside the brackets (`[const 3]`); like
        // `Pointer`, they participate in compatibility.
        quals: Vec<TypeQualifier>,
        // `None` for `[]`, `Star` for `[*]`, else the bound expression.
        // Bounds compare structurally: `a[3]` and a later `a[3]` merge,
        // but `a[1+1]` against `a[2]` conflicts (no constant folding
        // yet — a documented limitation).
        bound: Option<ArrayBound>,
    },
    // A struct or union, by nominal tag id: members and completeness live
    // in `Sema::structs`, so a forward declaration and the later
    // definition share one `Ty`.
    Struct(StructId),
    // An enum, by nominal tag id: the enumerator list and completeness
    // live behind the id in `Sema::enums`. The enum type itself behaves
    // like `int` here (compatible with it for conversions, rank and
    // size); layout owns the exact representation choice eventually.
    Enum(EnumId),
    // GCC's builtin `va_list` type (`__builtin_va_list`, the type behind
    // `va_list` in <stdarg.h>/<stdio.h>): opaque here — it names
    // parameter and object types, but no arithmetic or conversions apply.
    BuiltinVaList,
    // GCC's quad-precision floating type (`__float128`, via <bits/floatn.h>):
    // opaque until layout/codegen models it — like `BuiltinVaList`, it
    // names types but takes part in no conversions or arithmetic yet.
    Float128,
    /// A qualified type: base qualifiers (`const int`, in sorted order)
    /// around any other type. Only `spec_to_ty` builds these (see
    /// `Ty::qualified` for the normalizations); every predicate looks
    /// through via `unqualified`, and conversions compare qualifier
    /// sets instead of erasing them.
    Qualified {
        quals: Vec<TypeQualifier>,
        inner: Box<Ty>,
    },
}

impl Ty {
    /// Wrap a type in qualifiers, normalizing so equal spellings compare
    /// equal: qualifier sets sort and dedupe, empty sets vanish, and
    /// qualifiers distribute onto what they qualify in C — into array
    /// elements (`typedef int A[3]; const A a;` is an array of const
    /// int), into a pointer's own set (`typedef int *P; const P p;` is
    /// a const pointer), and off function types entirely (a qualifier
    /// on a function type is meaningless, silently like GCC).
    pub fn qualified(mut quals: Vec<TypeQualifier>, inner: Ty) -> Ty {
        quals.sort_by_key(|q| *q as u8);
        quals.dedup();
        if quals.is_empty() {
            return inner;
        }
        match inner {
            Ty::Array { elem, quals: aq, bound } => Ty::Array {
                elem: Box::new(Ty::qualified(quals, *elem)),
                quals: aq,
                bound,
            },
            Ty::Pointer { mut qualifiers, size, pointee } => {
                qualifiers.extend(quals);
                qualifiers.sort_by_key(|q| *q as u8);
                qualifiers.dedup();
                Ty::Pointer { qualifiers, size, pointee }
            }
            Ty::Func { .. } => inner,
            Ty::Qualified {
                quals: mut inner_q,
                inner: inner_i,
            } => {
                inner_q.extend(quals);
                Ty::qualified(inner_q, *inner_i)
            }
            _ => Ty::Qualified { quals, inner: Box::new(inner) },
        }
    }

    /// Strip qualifiers, for predicates over the unqualified type
    /// (ranks, sizes, alignments, shapes).
    pub fn unqualified(&self) -> &Ty {
        match self {
            Ty::Qualified { inner, .. } => inner.unqualified(),
            ty => ty,
        }
    }

    /// The unqualified version of this type, as C23 `typeof_unqual`
    /// spells it: top-level qualifiers go, while qualifiers below a
    /// pointer stay (`const int *const` becomes `const int *`).
    pub fn unqualified_owned(&self) -> Ty {
        match self {
            Ty::Qualified { inner, .. } => inner.unqualified_owned(),
            Ty::Pointer { size, pointee, .. } => Ty::Pointer {
                qualifiers: Vec::new(),
                size: *size,
                pointee: pointee.clone(),
            },
            Ty::Array { elem, bound, .. } => Ty::Array {
                elem: Box::new(elem.unqualified_owned()),
                quals: Vec::new(),
                bound: bound.clone(),
            },
            ty => ty.clone(),
        }
    }
    /// Whether the type itself is `const`-qualified (looking only at
    /// its own qualifier set, not through pointers or arrays).
    pub fn is_const(&self) -> bool {
        matches!(self, Ty::Qualified { quals, .. } if quals.contains(&TypeQualifier::Const))
    }

    /// Whether an integer type is unsigned: `_Bool` and the `unsigned`
    /// variants. Plain `char` follows `char_signed` (false on
    /// AArch64/RISC-V; implementation-defined per C17 §6.2.5p15).
    /// Non-integer types (floats, pointers, void, aggregates,
    /// functions, enums) report false — enums behave like `int` here.
    /// Mirrors `check::is_signed_int_in`.
    pub fn is_unsigned_in(&self, char_signed: bool) -> bool {
        match self.unqualified() {
            Ty::Bool | Ty::UnsignedChar | Ty::UnsignedShort | Ty::UnsignedInt | Ty::UnsignedLong | Ty::UnsignedLongLong | Ty::UnsignedInt128 => true,
            Ty::Char => !char_signed,
            _ => false,
        }
    }

    /// Whether an integer type is unsigned under the LP64 default
    /// (signed plain `char`). Prefer [`Ty::is_unsigned_in`] with the
    /// compilation target wherever a `Sema` is in reach.
    pub fn is_unsigned(&self) -> bool {
        self.is_unsigned_in(crate::target::Target::LP64.char_signed)
    }
}
