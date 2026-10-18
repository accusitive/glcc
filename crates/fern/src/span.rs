//! Byte spans and diagnostics for the fern frontend.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub lo: u32,
    pub hi: u32,
}

impl Span {
    pub fn new(lo: usize, hi: usize) -> Span {
        Span {
            lo: lo as u32,
            hi: hi as u32,
        }
    }

    /// Smallest span covering both.
    pub fn to(self, other: Span) -> Span {
        Span {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    pub fn text<'s>(&self, src: &'s str) -> &'s str {
        &src[self.lo as usize..self.hi as usize]
    }
}

/// A located error. The frontend fails fast: the first diagnostic ends a
/// compile with `Err`; the driver renders it.
#[derive(Debug, Clone)]
pub struct Diag {
    pub msg: String,
    pub span: Span,
}

impl Diag {
    pub fn new(msg: impl Into<String>, span: Span) -> Diag {
        Diag {
            msg: msg.into(),
            span,
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
        let width = (diag.span.hi.saturating_sub(diag.span.lo).max(1) as usize).min(
            line_text
                .len()
                .saturating_sub(col.saturating_sub(1))
                .max(1),
        );
        let caret = format!("{}{}", " ".repeat(col - 1), "^".repeat(width));
        format!(
            "error: {}\n --> {}:{}\n  |\n{} | {}\n  | {}",
            diag.msg, line, col, line, line_text, caret
        )
    }
}
