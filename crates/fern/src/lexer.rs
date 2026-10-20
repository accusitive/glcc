//! Hand-written fern lexer. Every token carries its byte span.
//!
//! fern has a small token set: construction keywords, literals, and a
//! few operators. Names are split into `Lower` (values, type variables)
//! and `Upper` (type and constructor names) — the ML convention.

use crate::span::{Diag, FileId, KResult, Span};

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    // keywords
    Data,
    Class,
    Instance,
    Extern,
    Fun,
    Let,
    If,
    Then,
    Else,
    Match,
    Case,
    When,
    True,
    False,
    // names and literals
    Lower(String),
    Upper(String),
    /// `#name`: a builtin namespace reference.
    Builtin(String),
    Int(i64),
    Float(f64),
    Str(String),
    /// `$ command` or `$(command)`: `src` is the command text, `capture`
    /// selects `$()`.
    Shell { src: String, capture: bool },
    // punctuation
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semi,
    Colon,
    ColonColon, // ::
    Arrow,      // ->
    FatArrow,   // =>
    Eq,         // =
    Plus,
    Minus,
    Star,
    Slash,
    Lt,
    Le,
    Gt,
    Ge,
    EqEq,
    Ne,
    AmpAmp,
    PipePipe,
    PipeGt, // |> value pipe
    Pipe,   // | (data decl separator)
    Underscore,
    Eof,
}

impl Tok {
    /// Human-readable name for parser diagnostics.
    pub fn describe(&self) -> String {
        match self {
            Tok::Lower(s) => format!("name `{s}`"),
            Tok::Upper(s) => format!("name `{s}`"),
            Tok::Builtin(s) => format!("builtin `#{s}`"),
            Tok::Int(v) => format!("integer `{v}`"),
            Tok::Float(v) => format!("float `{v}`"),
            Tok::Str(_) => "a string literal".to_string(),
            Tok::Shell { .. } => "a shell command".to_string(),
            Tok::Eof => "end of file".to_string(),
            other => format!("`{}`", other.spelling()),
        }
    }

    fn spelling(&self) -> &'static str {
        match self {
            Tok::Data => "data",
            Tok::Class => "class",
            Tok::Instance => "instance",
            Tok::Extern => "extern",
            Tok::Fun => "fun",
            Tok::Let => "let",
            Tok::If => "if",
            Tok::Then => "then",
            Tok::Else => "else",
            Tok::Match => "match",
            Tok::Case => "case",
            Tok::When => "when",
            Tok::True => "true",
            Tok::False => "false",
            Tok::LParen => "(",
            Tok::RParen => ")",
            Tok::LBrace => "{",
            Tok::RBrace => "}",
            Tok::LBracket => "[",
            Tok::RBracket => "]",
            Tok::Comma => ",",
            Tok::Semi => ";",
            Tok::Colon => ":",
            Tok::ColonColon => "::",
            Tok::Arrow => "->",
            Tok::FatArrow => "=>",
            Tok::Eq => "=",
            Tok::Plus => "+",
            Tok::Minus => "-",
            Tok::Star => "*",
            Tok::Slash => "/",
            Tok::Lt => "<",
            Tok::Le => "<=",
            Tok::Gt => ">",
            Tok::Ge => ">=",
            Tok::EqEq => "==",
            Tok::Ne => "!=",
            Tok::AmpAmp => "&&",
            Tok::PipePipe => "||",
            Tok::PipeGt => "|>",
            Tok::Pipe => "|",
            Tok::Underscore => "_",
            Tok::Str(_) => "string",
            Tok::Shell { .. } => "shell",
            _ => "?",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    pub tok: Tok,
    pub span: Span,
}

const IDENT_REST: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";

pub fn lex(src: &str) -> KResult<Vec<Token>> {
    lex_in(FileId::UNSPECIFIED, src)
}

/// Lex `src`, stamping every token and diagnostic with `file`.
pub fn lex_in(file: FileId, src: &str) -> KResult<Vec<Token>> {
    Lexer::new(file, src).lex_all()
}

/// Lex a source fragment whose first byte is at `base` in its file. Used
/// for `${expr}` holes so their spans stay in the enclosing file.
pub fn lex_fragment(file: FileId, src: &str, base: u32) -> KResult<Vec<Token>> {
    Lexer::new_at(file, src, base).lex_all()
}

struct Lexer<'s> {
    file: FileId,
    src: &'s str,
    bytes: &'s [u8],
    pos: usize,
    base: u32,
}

impl<'s> Lexer<'s> {
    fn new(file: FileId, src: &'s str) -> Lexer<'s> {
        Lexer::new_at(file, src, 0)
    }

    fn new_at(file: FileId, src: &'s str, base: u32) -> Lexer<'s> {
        Lexer {
            file,
            src,
            bytes: src.as_bytes(),
            pos: 0,
            base,
        }
    }

    fn span(&self, lo: usize, hi: usize) -> Span {
        let base = self.base as usize;
        Span::in_file(self.file, base + lo, base + hi)
    }

    fn err(&self, msg: impl Into<String>, lo: usize, hi: usize) -> Diag {
        Diag::new(msg, self.span(lo, hi))
    }

    fn peek(&self) -> u8 {
        *self.bytes.get(self.pos).unwrap_or(&0)
    }

    fn peek2(&self) -> u8 {
        *self.bytes.get(self.pos + 1).unwrap_or(&0)
    }

    fn bump(&mut self) -> u8 {
        let b = self.peek();
        self.pos += 1;
        b
    }

    fn skip_trivia(&mut self) -> KResult<()> {
        loop {
            match self.peek() {
                b' ' | b'\t' | b'\r' | b'\n' => {
                    self.pos += 1;
                }
                b'-' if self.peek2() == b'-' => self.line_comment(),
                b'/' if self.peek2() == b'/' => self.line_comment(),
                b'/' if self.peek2() == b'*' => self.block_comment()?,
                _ => break,
            }
        }
        Ok(())
    }

    fn line_comment(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
            self.pos += 1;
        }
    }

    fn block_comment(&mut self) -> KResult<()> {
        let lo = self.pos;
        self.pos += 2;
        let mut depth = 1;
        loop {
            if self.pos + 1 >= self.bytes.len() {
                return Err(Diag::incomplete(
                    "unterminated block comment",
                    self.span(lo, self.pos),
                ));
            }
            if self.peek() == b'*' && self.peek2() == b'/' {
                depth -= 1;
                self.pos += 2;
                if depth == 0 {
                    break;
                }
            } else if self.peek() == b'/' && self.peek2() == b'*' {
                depth += 1;
                self.pos += 2;
            } else {
                self.pos += 1;
            }
        }
        Ok(())
    }

    fn push(&mut self, tok: Tok, lo: usize, tokens: &mut Vec<Token>) {
        tokens.push(Token {
            tok,
            span: self.span(lo, self.pos),
        });
    }

    fn ident(&mut self, lo: usize, tokens: &mut Vec<Token>) -> KResult<()> {
        let mut end = self.pos;
        while end < self.bytes.len() && IDENT_REST.contains(&self.bytes[end]) {
            end += 1;
        }
        self.pos = end;
        let word = self.src[lo..end].to_string();
        if word == "_" {
            self.push(Tok::Underscore, lo, tokens);
            return Ok(());
        }
        let tok = match word.as_str() {
            "data" => Tok::Data,
            "class" => Tok::Class,
            "instance" => Tok::Instance,
            "extern" => Tok::Extern,
            "fun" => Tok::Fun,
            "let" => Tok::Let,
            "if" => Tok::If,
            "then" => Tok::Then,
            "else" => Tok::Else,
            "match" => Tok::Match,
            "case" => Tok::Case,
            "when" => Tok::When,
            "true" => Tok::True,
            "false" => Tok::False,
            _ => {
                if word.as_bytes()[0].is_ascii_uppercase() {
                    Tok::Upper(word)
                } else {
                    Tok::Lower(word)
                }
            }
        };
        self.push(tok, lo, tokens);
        Ok(())
    }

    fn builtin(&mut self, lo: usize, tokens: &mut Vec<Token>) -> KResult<()> {
        self.pos += 1;
        let start = self.pos;
        let mut end = start;
        while end < self.bytes.len() && IDENT_REST.contains(&self.bytes[end]) {
            end += 1;
        }
        if end == start || !self.bytes[start].is_ascii_lowercase() {
            return Err(self.err(
                "`#` must be followed by a lowercase builtin name",
                lo,
                end.max(start + 1),
            ));
        }
        self.pos = end;
        let name = self.src[start..end].to_string();
        self.push(Tok::Builtin(name), lo, tokens);
        Ok(())
    }

    fn dollar(&mut self, lo: usize, tokens: &mut Vec<Token>) -> KResult<()> {
        use crate::shell::{Stop, command_end};

        if self.peek2() == b'{' {
            return Err(self.err(
                "`${...}` is an interpolation inside a shell word; start a command with `$`",
                lo,
                lo + 2,
            ));
        }
        if self.peek2() == b'(' {
            let start = self.pos + 2;
            let end = command_end(self.file, self.src, start, Stop::RParen)?;
            let src = self.src[start..end - 1].to_string();
            if src.trim().is_empty() {
                return Err(Diag::new("expected a command inside `$(`", self.span(lo, end)));
            }
            self.pos = end;
            self.push(Tok::Shell { src, capture: true }, lo, tokens);
            return Ok(());
        }
        let start = self.pos + 1;
        let end = command_end(self.file, self.src, start, Stop::TopLevel)?;
        let src = self.src[start..end].to_string();
        if src.trim().is_empty() {
            return Err(Diag::new("expected a command after `$`", self.span(lo, end)));
        }
        self.pos = end;
        self.push(Tok::Shell { src, capture: false }, lo, tokens);
        Ok(())
    }

    fn number(&mut self, lo: usize, tokens: &mut Vec<Token>) -> KResult<()> {
        let mut end = self.pos;
        while end < self.bytes.len() && self.bytes[end].is_ascii_digit() {
            end += 1;
        }
        let mut is_float = false;
        if end + 1 < self.bytes.len() && self.bytes[end] == b'.' && self.bytes[end + 1].is_ascii_digit() {
            is_float = true;
            end += 2;
            while end < self.bytes.len() && self.bytes[end].is_ascii_digit() {
                end += 1;
            }
        }
        let text = &self.src[lo..end];
        self.pos = end;
        if is_float {
            let value = text.parse::<f64>().map_err(|_| self.err(format!("float literal `{text}` is out of range"), lo, end))?;
            self.push(Tok::Float(value), lo, tokens);
        } else {
            let value = text.parse::<i64>().map_err(|_| self.err(format!("integer literal `{text}` is out of range"), lo, end))?;
            self.push(Tok::Int(value), lo, tokens);
        }
        Ok(())
    }

    fn punct(&mut self, lo: usize, tokens: &mut Vec<Token>) -> KResult<()> {
        let start = self.pos;
        let b = self.bump();
        let two = |this: &Lexer| this.peek();
        match b {
            b'(' => {
                tokens.push(Token {
                    tok: Tok::LParen,
                    span: self.span(lo, start + 1),
                });
                Ok(())
            }
            b')' => {
                tokens.push(Token {
                    tok: Tok::RParen,
                    span: self.span(lo, start + 1),
                });
                Ok(())
            }
            b'{' => {
                tokens.push(Token {
                    tok: Tok::LBrace,
                    span: self.span(lo, start + 1),
                });
                Ok(())
            }
            b'}' => {
                tokens.push(Token {
                    tok: Tok::RBrace,
                    span: self.span(lo, start + 1),
                });
                Ok(())
            }
            b'[' => {
                tokens.push(Token {
                    tok: Tok::LBracket,
                    span: self.span(lo, start + 1),
                });
                Ok(())
            }
            b']' => {
                tokens.push(Token {
                    tok: Tok::RBracket,
                    span: self.span(lo, start + 1),
                });
                Ok(())
            }
            b',' => {
                tokens.push(Token {
                    tok: Tok::Comma,
                    span: self.span(lo, start + 1),
                });
                Ok(())
            }
            b';' => {
                tokens.push(Token {
                    tok: Tok::Semi,
                    span: self.span(lo, start + 1),
                });
                Ok(())
            }
            b'*' => {
                tokens.push(Token {
                    tok: Tok::Star,
                    span: self.span(lo, start + 1),
                });
                Ok(())
            }
            b'+' => {
                tokens.push(Token {
                    tok: Tok::Plus,
                    span: self.span(lo, start + 1),
                });
                Ok(())
            }
            b'/' => {
                tokens.push(Token {
                    tok: Tok::Slash,
                    span: self.span(lo, start + 1),
                });
                Ok(())
            }
            b'-' => {
                if two(self) == b'>' {
                    self.bump();
                    tokens.push(Token {
                        tok: Tok::Arrow,
                        span: self.span(lo, start + 2),
                    });
                } else {
                    tokens.push(Token {
                        tok: Tok::Minus,
                        span: self.span(lo, start + 1),
                    });
                }
                Ok(())
            }
            b'|' => {
                if two(self) == b'|' {
                    self.bump();
                    tokens.push(Token {
                        tok: Tok::PipePipe,
                        span: self.span(lo, start + 2),
                    });
                } else if two(self) == b'>' {
                    self.bump();
                    tokens.push(Token {
                        tok: Tok::PipeGt,
                        span: self.span(lo, start + 2),
                    });
                } else {
                    tokens.push(Token {
                        tok: Tok::Pipe,
                        span: self.span(lo, start + 1),
                    });
                }
                Ok(())
            }
            b'=' => {
                if two(self) == b'=' {
                    self.bump();
                    tokens.push(Token {
                        tok: Tok::EqEq,
                        span: self.span(lo, start + 2),
                    });
                } else if two(self) == b'>' {
                    self.bump();
                    tokens.push(Token {
                        tok: Tok::FatArrow,
                        span: self.span(lo, start + 2),
                    });
                } else {
                    tokens.push(Token {
                        tok: Tok::Eq,
                        span: self.span(lo, start + 1),
                    });
                }
                Ok(())
            }
            b'!' => {
                if two(self) == b'=' {
                    self.bump();
                    tokens.push(Token {
                        tok: Tok::Ne,
                        span: self.span(lo, start + 2),
                    });
                    Ok(())
                } else {
                    Err(self.err("`!` is not valid fern; use `Eq::eq`", lo, self.pos))
                }
            }
            b'<' => {
                if two(self) == b'=' {
                    self.bump();
                    tokens.push(Token {
                        tok: Tok::Le,
                        span: self.span(lo, start + 2),
                    });
                } else {
                    tokens.push(Token {
                        tok: Tok::Lt,
                        span: self.span(lo, start + 1),
                    });
                }
                Ok(())
            }
            b'>' => {
                if two(self) == b'=' {
                    self.bump();
                    tokens.push(Token {
                        tok: Tok::Ge,
                        span: self.span(lo, start + 2),
                    });
                } else {
                    tokens.push(Token {
                        tok: Tok::Gt,
                        span: self.span(lo, start + 1),
                    });
                }
                Ok(())
            }
            b':' => {
                if two(self) == b':' {
                    self.bump();
                    tokens.push(Token {
                        tok: Tok::ColonColon,
                        span: self.span(lo, start + 2),
                    });
                } else {
                    tokens.push(Token {
                        tok: Tok::Colon,
                        span: self.span(lo, start + 1),
                    });
                }
                Ok(())
            }
            b'&' => {
                if two(self) == b'&' {
                    self.bump();
                    tokens.push(Token {
                        tok: Tok::AmpAmp,
                        span: self.span(lo, start + 2),
                    });
                    Ok(())
                } else {
                    Err(self.err("single `&` is not valid fern", lo, self.pos))
                }
            }
            _ => Err(self.err("unexpected character in source text", lo, self.pos.max(lo + 1))),
        }
    }

    fn next(&mut self, tokens: &mut Vec<Token>) -> KResult<()> {
        self.skip_trivia()?;
        let lo = self.pos;
        let b = self.peek();
        if b == 0 {
            self.push(Tok::Eof, lo, tokens);
            return Ok(());
        }
        if b == b'-' && self.peek2() == b'-' {
            // `--` starts a comment even mid-expression (`a - -b` must be
            // written `a - (-b)`).
            self.line_comment();
            return self.skip_trivia().map_err(|e| e);
        }
        if b.is_ascii_digit() {
            return self.number(lo, tokens);
        }
        if b.is_ascii_alphabetic() || b == b'_' {
            return self.ident(lo, tokens);
        }
        if b == b'"' {
            return self.string(lo, tokens);
        }
        if b == b'#' {
            return self.builtin(lo, tokens);
        }
        if b == b'$' {
            return self.dollar(lo, tokens);
        }
        self.punct(lo, tokens)
    }

    /// A double-quoted literal. Newlines survive verbatim; the escapes
    /// are the small portable set, with `\0` reserved for NUL.
    fn string(&mut self, lo: usize, tokens: &mut Vec<Token>) -> KResult<()> {
        self.pos += 1;
        let mut out = String::new();
        loop {
            if self.pos >= self.bytes.len() {
                return Err(Diag::incomplete("unterminated string literal", self.span(lo, self.pos)));
            }
            let c = self.src[self.pos..].chars().next().expect("char boundary");
            self.pos += c.len_utf8();
            match c {
                '"' => {
                    self.push(Tok::Str(out), lo, tokens);
                    return Ok(());
                }
                '\\' => {
                    if self.pos >= self.bytes.len() {
                        return Err(Diag::incomplete("unterminated string literal", self.span(lo, self.pos)));
                    }
                    let escape = self.src[self.pos..].chars().next().expect("char boundary");
                    self.pos += escape.len_utf8();
                    match escape {
                        'n' => out.push('\n'),
                        't' => out.push('\t'),
                        'r' => out.push('\r'),
                        '0' => out.push('\0'),
                        '\\' => out.push('\\'),
                        '"' => out.push('"'),
                        other => return Err(self.err(format!("unknown string escape `\\{other}`"), lo, self.pos)),
                    }
                }
                other => out.push(other),
            }
        }
    }

    fn lex_all(mut self) -> KResult<Vec<Token>> {
        let mut tokens = Vec::new();
        loop {
            self.next(&mut tokens)?;
            if matches!(tokens.last().map(|t| &t.tok), Some(Tok::Eof)) {
                break;
            }
        }
        tokens.pop().unwrap();
        tokens.push(Token {
            tok: Tok::Eof,
            span: self.span(self.pos, self.pos),
        });
        Ok(tokens)
    }
}
