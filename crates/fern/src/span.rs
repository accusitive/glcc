//! Byte spans and diagnostics for the fern frontend.

/// Identifies a source file within a [`crate::session::Session`].
///
/// File ids are indices into the session's file table. `UNSPECIFIED` is
/// used by callers that lex a string without a session (unit tests, the
/// prelude builder); `PRELUDE` is reserved for the built-in prelude,
/// which every session inserts first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FileId(pub u32);

impl FileId {
    pub const UNSPECIFIED: FileId = FileId(u32::MAX);
    pub const PRELUDE: FileId = FileId(0);

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub file: FileId,
    pub lo: u32,
    pub hi: u32,
}

impl Span {
    /// A span with no owning file. Used for synthesized nodes and for
    /// callers that lex without a session.
    pub fn new(lo: usize, hi: usize) -> Span {
        Span {
            file: FileId::UNSPECIFIED,
            lo: lo as u32,
            hi: hi as u32,
        }
    }

    pub fn in_file(file: FileId, lo: usize, hi: usize) -> Span {
        Span {
            file,
            lo: lo as u32,
            hi: hi as u32,
        }
    }

    /// Smallest span covering both. The receiver's file wins; spans from
    /// different files are never combined by the parser.
    pub fn to(self, other: Span) -> Span {
        debug_assert!(
            self.file == other.file
                || self.file == FileId::UNSPECIFIED
                || other.file == FileId::UNSPECIFIED,
            "combined spans from different files"
        );
        Span {
            file: if self.file == FileId::UNSPECIFIED {
                other.file
            } else {
                self.file
            },
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    pub fn text<'s>(&self, src: &'s str) -> &'s str {
        &src[self.lo as usize..self.hi as usize]
    }
}

/// A located error.
#[derive(Debug, Clone)]
pub struct Diag {
    pub msg: String,
    pub span: Span,
    /// True when the error is caused by input that stops mid-construct
    /// (end of file inside a block/quote, a trailing operator). The REPL
    /// uses this to decide whether to read more.
    pub incomplete: bool,
}

impl Diag {
    pub fn new(msg: impl Into<String>, span: Span) -> Diag {
        Diag {
            msg: msg.into(),
            span,
            incomplete: false,
        }
    }

    pub fn incomplete(msg: impl Into<String>, span: Span) -> Diag {
        Diag {
            msg: msg.into(),
            span,
            incomplete: true,
        }
    }
}

pub type KResult<T> = Result<T, Diag>;

/// Line index for rendering diagnostics.
pub struct SourceMap {
    line_starts: Vec<u32>,
}

impl SourceMap {
    pub fn new(src: &str) -> SourceMap {
        let mut line_starts = vec![0u32];
        for (i, b) in src.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i as u32 + 1);
            }
        }
        SourceMap { line_starts }
    }

    /// 1-based (line, column) of a byte offset.
    pub fn line_col(&self, off: u32) -> (usize, usize) {
        let line = match self.line_starts.binary_search(&off) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let start = self.line_starts[line] as usize;
        (line + 1, off as usize - start + 1)
    }

    /// Render a diagnostic as `error: msg\n --> line:col\n  |\nN | source\n  | ^`.
    pub fn render(&self, src: &str, diag: &Diag) -> String {
        let (line, col) = self.line_col(diag.span.lo);
        let line_text = src.lines().nth(line - 1).unwrap_or("");
        let width = (diag.span.hi.saturating_sub(diag.span.lo).max(1) as usize).min(line_text.len().saturating_sub(col.saturating_sub(1)).max(1));
        let caret = format!("{}{}", " ".repeat(col - 1), "^".repeat(width));
        format!("error: {}\n --> {}:{}\n  |\n{} | {}\n  | {}", diag.msg, line, col, line, line_text, caret)
    }
}

/// Render a diagnostic against a set of sources indexed by file id.
pub fn render_in(files: &[&str], diag: &Diag) -> String {
    let file = diag.span.file.index();
    let src = files.get(file).copied().unwrap_or("");
    let map = SourceMap::new(src);
    let rendered = map.render(src, diag);
    if diag.span.file == FileId::PRELUDE {
        rendered.replacen(" --> ", " --> prelude:", 1)
    } else {
        rendered
    }
}
