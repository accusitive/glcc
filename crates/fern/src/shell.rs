//! Shell command scanning and word parsing.
//!
//! A `$` command is delimited by the surrounding fern expression: a
//! newline, `&&`, `||`, `|>`, or a closing bracket ends it, while a
//! single `|` separates pipeline stages. Quotes and `${...}` / `$(...)`
//! holes are scanned here; interpolation expressions are handed back as
//! source text and parsed by the fern parser with their original spans.

use crate::span::{Diag, FileId, KResult, Span};

#[derive(Debug, Clone)]
pub struct RawCommand {
    pub stages: Vec<RawStage>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct RawStage {
    pub words: Vec<RawWord>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct RawWord {
    pub parts: Vec<RawPart>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum RawPart {
    Lit(String),
    /// The source between `${` and `}`, plus its exact file span.
    Interp {
        src: String,
        span: Span,
    },
    Capture(Box<RawCommand>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// Stop before a fern-level terminator.
    TopLevel,
    /// Stop after the matching `)` and consume it.
    RParen,
}

fn span(file: FileId, lo: usize, hi: usize) -> Span {
    Span::in_file(file, lo, hi)
}

fn char_len(src: &str, at: usize) -> usize {
    src[at..].chars().next().map(char::len_utf8).unwrap_or(1)
}

/// Find the end of a shell command starting just after `$` or `$(`.
///
/// For [`Stop::TopLevel`] the returned index is the first byte of the
/// terminating fern token (or the end of input). For [`Stop::RParen`]
/// it is the index just past the matching `)`.
pub fn command_end(file: FileId, src: &str, start: usize, stop: Stop) -> KResult<usize> {
    let bytes = src.as_bytes();
    let mut i = start;
    let mut word_start = true;
    let mut pending_pipe = false;
    while i < bytes.len() {
        let b = bytes[i];
        match b {
            b'\'' => {
                i = single_quote_end(file, src, i)?;
                word_start = false;
                pending_pipe = false;
            }
            b'"' => {
                i = double_quote_end(file, src, i)?;
                word_start = false;
                pending_pipe = false;
            }
            b'\\' => {
                if i + 1 >= bytes.len() {
                    return Err(Diag::incomplete("unterminated escape in a shell command", span(file, i, i + 1)));
                }
                i += if bytes[i + 1] == b'\n' { 2 } else { 1 + char_len(src, i + 1) };
                word_start = false;
                pending_pipe = false;
            }
            b'$' if i + 1 < bytes.len() && bytes[i + 1] == b'(' => {
                i = command_end(file, src, i + 2, Stop::RParen)?;
                word_start = false;
                pending_pipe = false;
            }
            b'$' if i + 1 < bytes.len() && bytes[i + 1] == b'{' => {
                i = brace_end(file, src, i + 2)?;
                word_start = false;
                pending_pipe = false;
            }
            b'|' => {
                if i + 1 < bytes.len() && matches!(bytes[i + 1], b'|' | b'>') {
                    break;
                }
                i += 1;
                word_start = true;
                pending_pipe = true;
            }
            b'&' if i + 1 < bytes.len() && bytes[i + 1] == b'&' => break,
            b';' | b',' | b']' | b'}' => break,
            b')' => {
                return match stop {
                    Stop::RParen => Ok(i + 1),
                    Stop::TopLevel => Ok(i),
                };
            }
            b'\n' => {
                if stop == Stop::TopLevel {
                    if !pending_pipe {
                        break;
                    }
                    pending_pipe = false;
                    i += 1;
                    word_start = true;
                    continue;
                }
                i += 1;
                word_start = true;
            }
            b' ' | b'\t' | b'\r' => {
                i += 1;
                word_start = true;
            }
            _ => {
                if word_start && stop == Stop::TopLevel && keyword_here(src, i) {
                    break;
                }
                i += char_len(src, i);
                word_start = false;
                pending_pipe = false;
            }
        }
    }
    if stop == Stop::RParen {
        return Err(Diag::incomplete("unterminated `$(` shell capture", span(file, start.saturating_sub(2), src.len())));
    }
    Ok(i)
}

/// `then` and `else` end a command that sits in a fern conditional.
/// Other item/expression keywords cannot follow a command without a
/// delimiter, so they are left as ordinary words.
fn keyword_here(src: &str, at: usize) -> bool {
    let rest = &src[at..];
    ["then", "else"].iter().any(|kw| match rest.strip_prefix(kw) {
        Some(after) => !after.as_bytes().first().is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_'),
        None => false,
    })
}

fn single_quote_end(file: FileId, src: &str, start: usize) -> KResult<usize> {
    let bytes = src.as_bytes();
    let mut i = start + 1;
    while i < bytes.len() {
        if bytes[i] == b'\'' {
            return Ok(i + 1);
        }
        i += char_len(src, i);
    }
    Err(Diag::incomplete("unterminated single quote in a shell command", span(file, start, src.len())))
}

fn double_quote_end(file: FileId, src: &str, start: usize) -> KResult<usize> {
    let bytes = src.as_bytes();
    let mut i = start + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => return Ok(i + 1),
            b'\\' => {
                if i + 1 >= bytes.len() {
                    return Err(Diag::incomplete("unterminated escape in a shell command", span(file, i, i + 1)));
                }
                i += if bytes[i + 1] == b'\n' { 2 } else { 1 + char_len(src, i + 1) };
            }
            b'$' if i + 1 < bytes.len() && bytes[i + 1] == b'(' => {
                i = command_end(file, src, i + 2, Stop::RParen)?;
            }
            b'$' if i + 1 < bytes.len() && bytes[i + 1] == b'{' => {
                i = brace_end(file, src, i + 2)?;
            }
            _ => i += char_len(src, i),
        }
    }
    Err(Diag::incomplete("unterminated double quote in a shell command", span(file, start, src.len())))
}

/// Find the `}` matching a `${`, returning the position just past it.
/// Fern strings and comments inside the expression are skipped so their
/// braces do not count.
fn brace_end(file: FileId, src: &str, start: usize) -> KResult<usize> {
    let bytes = src.as_bytes();
    let mut i = start;
    let mut depth = 1usize;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => i = double_quote_end(file, src, i)?,
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'/' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'-' if i + 1 < bytes.len() && bytes[i + 1] == b'-' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'*' => {
                i = block_comment_end(file, src, i)?;
            }
            b'$' if i + 1 < bytes.len() && bytes[i + 1] == b'(' => {
                i = command_end(file, src, i + 2, Stop::RParen)?;
            }
            b'{' => {
                depth += 1;
                i += 1;
            }
            b'}' => {
                depth -= 1;
                i += 1;
                if depth == 0 {
                    return Ok(i);
                }
            }
            _ => i += char_len(src, i),
        }
    }
    Err(Diag::incomplete("unterminated `${` interpolation", span(file, start.saturating_sub(2), src.len())))
}

fn block_comment_end(file: FileId, src: &str, start: usize) -> KResult<usize> {
    let bytes = src.as_bytes();
    let mut i = start + 2;
    let mut depth = 1usize;
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'/' {
            depth -= 1;
            i += 2;
            if depth == 0 {
                return Ok(i);
            }
        } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            depth += 1;
            i += 2;
        } else {
            i += char_len(src, i);
        }
    }
    Err(Diag::incomplete("unterminated block comment in an interpolation", span(file, start, src.len())))
}

/// Parse the words of a command whose text has already been cut at the
/// surrounding fern expression. `base` is the file offset of `src[0]`.
pub fn parse_raw(file: FileId, src: &str, base: u32) -> KResult<RawCommand> {
    let mut parser = WordParser {
        file,
        src,
        bytes: src.as_bytes(),
        pos: 0,
        base,
    };
    let stages = parser.stages()?;
    if stages.is_empty() {
        return Err(Diag::new("expected a command", span(file, base as usize, base as usize + src.len())));
    }
    Ok(RawCommand {
        stages,
        span: span(file, base as usize, base as usize + src.len()),
    })
}

struct WordParser<'s> {
    file: FileId,
    src: &'s str,
    bytes: &'s [u8],
    pos: usize,
    base: u32,
}

impl WordParser<'_> {
    fn span(&self, lo: usize, hi: usize) -> Span {
        let base = self.base as usize;
        span(self.file, base + lo, base + hi)
    }

    fn peek(&self) -> u8 {
        *self.bytes.get(self.pos).unwrap_or(&0)
    }

    fn skip_ws(&mut self) {
        loop {
            match self.peek() {
                b' ' | b'\t' | b'\r' | b'\n' => self.pos += 1,
                b'\\' if self.bytes.get(self.pos + 1) == Some(&b'\n') => self.pos += 2,
                _ => break,
            }
        }
    }

    fn stages(&mut self) -> KResult<Vec<RawStage>> {
        let mut stages = Vec::new();
        loop {
            self.skip_ws();
            if self.pos >= self.bytes.len() {
                break;
            }
            let start = self.pos;
            let mut words = Vec::new();
            loop {
                self.skip_ws();
                if self.pos >= self.bytes.len() {
                    break;
                }
                if self.peek() == b'|' {
                    let pipe = self.pos;
                    self.pos += 1;
                    self.skip_ws();
                    if self.pos >= self.bytes.len() {
                        return Err(Diag::incomplete("expected a command after `|`", self.span(pipe, pipe + 1)));
                    }
                    break;
                }
                words.push(self.word()?);
            }
            if words.is_empty() {
                return Err(Diag::new("expected a command word", self.span(start, self.pos)));
            }
            let end = words.last().map(|w| w.span.hi as usize - self.base as usize).unwrap_or(start);
            stages.push(RawStage {
                words,
                span: self.span(start, end),
            });
        }
        Ok(stages)
    }

    fn word(&mut self) -> KResult<RawWord> {
        let start = self.pos;
        let mut parts = Vec::new();
        while self.pos < self.bytes.len() {
            match self.peek() {
                b' ' | b'\t' | b'\r' | b'\n' | b'|' => break,
                b'\'' => parts.push(RawPart::Lit(self.single_quoted()?)),
                b'"' => self.double_quoted(&mut parts)?,
                b'\\' => {
                    self.pos += 1;
                    if self.pos >= self.bytes.len() {
                        return Err(Diag::incomplete("unterminated escape in a shell word", self.span(start, self.pos)));
                    }
                    if self.peek() == b'\n' {
                        self.pos += 1;
                        continue;
                    }
                    let len = char_len(self.src, self.pos);
                    parts.push(RawPart::Lit(self.src[self.pos..self.pos + len].to_string()));
                    self.pos += len;
                }
                b'$' => {
                    let next = self.bytes.get(self.pos + 1).copied();
                    match next {
                        Some(b'(') => {
                            let end = command_end(self.file, self.src, self.pos + 2, Stop::RParen)?;
                            let inner = &self.src[self.pos + 2..end - 1];
                            let cmd = parse_raw(self.file, inner, self.base + (self.pos + 2) as u32)?;
                            parts.push(RawPart::Capture(Box::new(cmd)));
                            self.pos = end;
                        }
                        Some(b'{') => {
                            let end = brace_end(self.file, self.src, self.pos + 2)?;
                            let inner = &self.src[self.pos + 2..end - 1];
                            parts.push(RawPart::Interp {
                                src: inner.to_string(),
                                span: self.span(self.pos + 2, end - 1),
                            });
                            self.pos = end;
                        }
                        _ => {
                            return Err(Diag::new("`$` in a shell word must start `${expr}` or `$(command)`", self.span(self.pos, self.pos + 1)));
                        }
                    }
                }
                _ => {
                    let run = self.pos;
                    while self.pos < self.bytes.len() {
                        match self.peek() {
                            b' ' | b'\t' | b'\r' | b'\n' | b'|' | b'\'' | b'"' | b'\\' | b'$' => break,
                            _ => self.pos += char_len(self.src, self.pos),
                        }
                    }
                    parts.push(RawPart::Lit(self.src[run..self.pos].to_string()));
                }
            }
        }
        if parts.is_empty() {
            return Err(Diag::new("expected a command word", self.span(start, start + 1)));
        }
        Ok(RawWord {
            parts,
            span: self.span(start, self.pos),
        })
    }

    fn single_quoted(&mut self) -> KResult<String> {
        let start = self.pos;
        let end = single_quote_end(self.file, self.src, start)?;
        self.pos = end;
        Ok(self.src[start + 1..end - 1].to_string())
    }

    fn double_quoted(&mut self, out: &mut Vec<RawPart>) -> KResult<()> {
        self.pos += 1;
        let mut lit = String::new();
        loop {
            if self.pos >= self.bytes.len() {
                return Err(Diag::incomplete(
                    "unterminated double quote in a shell word",
                    self.span(self.pos.saturating_sub(1), self.pos),
                ));
            }
            match self.peek() {
                b'"' => {
                    self.pos += 1;
                    if !lit.is_empty() || out.is_empty() {
                        out.push(RawPart::Lit(std::mem::take(&mut lit)));
                    }
                    return Ok(());
                }
                b'\\' => {
                    self.pos += 1;
                    if self.pos >= self.bytes.len() {
                        return Err(Diag::incomplete("unterminated escape in a shell word", self.span(self.pos.saturating_sub(1), self.pos)));
                    }
                    if self.peek() == b'\n' {
                        self.pos += 1;
                        continue;
                    }
                    let len = char_len(self.src, self.pos);
                    let escaped = &self.src[self.pos..self.pos + len];
                    if matches!(escaped, "\"" | "\\" | "$" | "`") {
                        lit.push_str(escaped);
                    } else {
                        lit.push('\\');
                        lit.push_str(escaped);
                    }
                    self.pos += len;
                }
                b'$' => {
                    let next = self.bytes.get(self.pos + 1).copied();
                    match next {
                        Some(b'(') | Some(b'{') => {
                            if !lit.is_empty() {
                                out.push(RawPart::Lit(std::mem::take(&mut lit)));
                            }
                            if next == Some(b'(') {
                                let end = command_end(self.file, self.src, self.pos + 2, Stop::RParen)?;
                                let inner = &self.src[self.pos + 2..end - 1];
                                let cmd = parse_raw(self.file, inner, self.base + (self.pos + 2) as u32)?;
                                out.push(RawPart::Capture(Box::new(cmd)));
                                self.pos = end;
                            } else {
                                let end = brace_end(self.file, self.src, self.pos + 2)?;
                                let inner = &self.src[self.pos + 2..end - 1];
                                out.push(RawPart::Interp {
                                    src: inner.to_string(),
                                    span: self.span(self.pos + 2, end - 1),
                                });
                                self.pos = end;
                            }
                        }
                        _ => {
                            return Err(Diag::new("`$` in a shell word must start `${expr}` or `$(command)`", self.span(self.pos, self.pos + 1)));
                        }
                    }
                }
                _ => {
                    let len = char_len(self.src, self.pos);
                    lit.push_str(&self.src[self.pos..self.pos + len]);
                    self.pos += len;
                }
            }
        }
    }
}
