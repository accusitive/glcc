use alloc::{format, string::String, vec::Vec};
use core::str::Chars;

use peekmore::{PeekMore, PeekMoreIterator};
use string_interner::{backend::BufferBackend, symbol::SymbolU32};

use crate::span::{CompileError, HasSpan, Span};

pub type Interner = string_interner::StringInterner<BufferBackend<SymbolU32>>;

#[derive(Debug, Clone, Copy)]
pub struct Location {
    pub ln: u64,
    pub col: u64,
    pub file: SymbolU32,
}
#[derive(Debug, Clone, Copy)]
pub struct Located<T> {
    /// Start position (inclusive) of the token.
    pub loc: Location,
    /// End position (exclusive) of the token.
    pub end: Location,
    pub val: T,
}

impl<T> Located<T> {
    pub fn span(&self) -> Span {
        Span::new(self.loc, self.end)
    }
}

impl<T> HasSpan for Located<T> {
    fn span(&self) -> Span {
        self.span()
    }
}

/// Iterator over logical chars: backslash-newline splices are deleted
/// (translation phase 2). Each item carries the lines and raw bytes that
/// vanished immediately before it, so the consumer can apply the position
/// effects when the item is actually consumed — peeking past a splice
/// must not move the reported location.
pub(crate) struct PrelexerIter<'a> {
    inner: PeekMoreIterator<Chars<'a>>,
}

impl<'a> PrelexerIter<'a> {
    fn new(source: Chars<'a>) -> Self {
        Self { inner: source.peekmore() }
    }

    /// Length of the run a backslash escapes: blanks plus the newline
    /// (`\r\n`/`\n\r` count as one), or `None` when the backslash does
    /// not splice. clang `getEscapedNewLineSize` tolerates blanks
    /// (`-Wbackslash-newline-escape`).
    fn splice_len(&mut self) -> Option<usize> {
        let mut i = 0;
        loop {
            let c = *self.inner.peek_nth(i)?;
            match c {
                '\n' | '\r' => {
                    let other = if c == '\n' { '\r' } else { '\n' };
                    let paired = self.inner.peek_nth(i + 1).is_some_and(|n| *n == other);
                    return Some(i + 1 + usize::from(paired));
                }
                ' ' | '\t' | '\x0b' | '\x0c' => i += 1,
                _ => return None,
            }
        }
    }
}

impl Iterator for PrelexerIter<'_> {
    /// `(char, lines, raw)`: the logical char plus the number of lines
    /// and raw bytes of the splice run deleted immediately before it.
    type Item = (char, u64, u64);

    fn next(&mut self) -> Option<(char, u64, u64)> {
        let mut lines = 0;
        let mut raw = 0;
        let mut c = self.inner.next()?;
        while c == '\\' {
            let Some(len) = self.splice_len() else { break };
            for _ in 0..len {
                self.inner.next();
            }
            lines += 1;
            raw += len as u64 + 1;
            c = self.inner.next()?;
        }
        Some((c, lines, raw))
    }
}

/// Logical char stream with arbitrary lookahead: the prelexer deletes
/// splices, the peekmore iterator buffers the result for peeking.
type Prelexer<'a> = PeekMoreIterator<PrelexerIter<'a>>;

/// Eager lexer: borrows the char stream and the driver's interner,
/// produces all tokens up front via [`Lexer::lex_all`]. Spellings live
/// in the interner the driver owns (and later threads into the
/// backend).
pub struct Lexer<'a> {
    chars: Prelexer<'a>,
    loc: Location,
    intern: &'a mut Interner,
    /// Lexical errors recovered while lexing. `lex_all` carries them on
    /// the returned [`TokenStream`]; they never abort lexing, so a bad
    /// literal produces a token plus a diagnostic instead of a panic.
    diagnostics: Vec<CompileError>,
    /// Physical offset just past the last consumed char, and the offset
    /// of that char itself. Splices count: `pp_next` sees the spliced
    /// stream, but clang's spelling offsets live in the raw buffer.
    raw: u64,
    char_off: u64,
}

/// One preprocessing token for `glcc_pp` (over glcc's interner): raw
/// spellings, whitespace preserved, numbers
/// and strings deliberately unvalidated. A later pass converts and
/// validates pp-numbers into proper literals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PpToken {
    /// Identifier or keyword (PP does not distinguish them).
    Ident(SymbolU32),
    /// Preprocessing number, raw spelling (`1U`, `0x1p-3`, `1e+1`).
    PpNumber(SymbolU32),
    /// String literal, raw spelling including prefix and quotes.
    Str(SymbolU32),
    /// Character literal, raw spelling including quotes.
    Char(SymbolU32),
    Punct(Punctuator),
    Ws(PpWs),
    /// `#` as the first non-whitespace character of a line.
    Directive,
    /// Anything else (stray backslash, `@`, backtick, ...): totality
    /// for the lexer; downstream stages report precise errors.
    Other(char),
}

/// Whitespace that survives preprocessing lexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PpWs {
    Space,
    Tab,
    Newline,
}

/// A preprocessing token plus its 1-based logical line and byte column.
#[derive(Debug, Clone, Copy)]
pub struct PpTok {
    pub tok: PpToken,
    pub line: u64,
    pub col: u64,
    /// clang `Token::LeadingSpace` (whitespace or a comment precedes).
    pub space: bool,
    /// clang `Token::StartOfLine` (only whitespace/comments since the
    /// last newline).
    pub sol: bool,
    /// Byte offset of the token's first char in the physical file.
    pub off: u64,
    /// Physical byte length, splices and escapes included.
    pub len: u64,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str, interner: &'a mut Interner) -> Self {
        let file = interner.get_or_intern("<__glcc_source>");
        Self {
            chars: PrelexerIter::new(source.chars()).peekmore(),
            loc: Location { ln: 1, col: 0, file },
            intern: interner,
            diagnostics: Vec::new(),
            raw: 0,
            char_off: 0,
        }
    }

    /// Record a recovered lexical error without aborting the lex.
    fn error(&mut self, start: Location, end: Location, msg: impl Into<String>) {
        self.diagnostics.push(CompileError::new(msg, Span::new(start, end)));
    }

    /// Lex the entire input into a [`TokenStream`]. The driver passes
    /// the stream on to [`Parser::new`](crate::parse::Parser::new).
    pub fn lex_all(mut self) -> TokenStream {
        let mut tokens = Vec::new();
        while let Some(tok) = self.next_token() {
            tokens.push(tok);
        }
        TokenStream::with_diagnostics(tokens, self.diagnostics)
    }

    /// Next spliced char for PP lexing, folding vanished (spliced-away)
    /// lines into `line` so physical numbering never drifts. `raw`
    /// tracks the byte offset in the physical buffer (splices included)
    /// and `char_off` the offset of the char just consumed.
    fn pp_next(&mut self, line: &mut u64) -> Option<char> {
        let (c, lines, raw) = self.chars.next()?;
        *line += lines;
        self.raw += raw;
        self.char_off = self.raw;
        self.raw += c.len_utf8() as u64;
        Some(c)
    }

    /// Lex the entire input as preprocessing tokens for `glcc_pp`.
    /// Unlike [`Lexer::lex_all`], whitespace is preserved, numbers and
    /// strings are kept raw (pp-numbers, no validation), comments become
    /// a single space, and lexing is total: stray characters surface as
    /// [`PpToken::Other`] for downstream diagnostics instead of panics.
    pub fn lex_pp_all(&mut self) -> Vec<PpTok> {
        let mut out = Vec::new();
        // Logical line/col over the spliced stream; `\`-newline splices
        // vanish (translation phase 2), every other `\n` is a Newline.
        let mut line: u64 = 1;
        let mut col: u64 = 0;
        // clang `StartOfLine` (nothing but whitespace/comments since the
        // last newline: only whitespace may precede a `Directive` hash)
        // and `LeadingSpace`, both for the *next* token.
        let mut sol = true;
        let mut space = false;
        // The pending flags captured for the token being emitted; `push!`
        // reads them (macro bodies resolve locals at the definition site).
        let mut tok_sol;
        let mut tok_space;
        // Emit one non-whitespace token with the captured lexical flags
        // and its physical span; the caller clears the pending flags.
        macro_rules! push {
            ($tok:expr, $line:expr, $col:expr, $off:expr) => {{
                out.push(PpTok {
                    tok: $tok,
                    line: $line,
                    col: $col,
                    space: tok_space,
                    sol: tok_sol,
                    off: $off,
                    len: self.raw - $off,
                });
            }};
        }
        while let Some(c) = self.pp_next(&mut line) {
            let tok_off = self.char_off;
            let tok_line = line;
            tok_space = space;
            tok_sol = sol;
            space = false;
            sol = false;
            match c {
                '\n' => {
                    push!(PpToken::Ws(PpWs::Newline), line, col, tok_off);
                    line += 1;
                    col = 0;
                    sol = true;
                }
                '\r' => {
                    // `\r\n` is one newline; a lone `\r` also ends the line.
                    if self.chars.peek().is_some_and(|(c, _, _)| *c == '\n') {
                        let _ = self.pp_next(&mut line);
                    }
                    push!(PpToken::Ws(PpWs::Newline), line, col, tok_off);
                    line += 1;
                    col = 0;
                    sol = true;
                }
                ' ' | '\t' => {
                    let start_col = col;
                    let mut spaces = 0u32;
                    let mut tabs = 0u32;
                    if c == ' ' {
                        spaces += 1;
                    } else {
                        tabs += 1;
                    }
                    col += 1;
                    while let Some((nc, _, _)) = self.chars.peek().copied() {
                        if nc == ' ' {
                            spaces += 1;
                            col += 1;
                            let _ = self.pp_next(&mut line);
                        } else if nc == '\t' {
                            tabs += 1;
                            col += 1;
                            let _ = self.pp_next(&mut line);
                        } else {
                            break;
                        }
                    }
                    // One token per run; tabs win when mixed (rendering is
                    // whitespace-insensitive outside literals).
                    let ws = if tabs > 0 && spaces == 0 { PpWs::Tab } else { PpWs::Space };
                    out.push(PpTok {
                        tok: PpToken::Ws(ws),
                        line,
                        col: start_col,
                        space: false,
                        sol: false,
                        off: tok_off,
                        len: self.raw - tok_off,
                    });
                    space = true;
                    sol = tok_sol;
                }
                '/' if self.chars.peek().is_some_and(|(c, _, _)| *c == '/') => {
                    // Line comment: skip to (not past) the newline, which
                    // ends the line for the next token.
                    while let Some(nc) = self.pp_next(&mut line) {
                        col += 1;
                        if nc == '\n' {
                            out.push(PpTok {
                                tok: PpToken::Ws(PpWs::Newline),
                                line,
                                col: 0,
                                space: false,
                                sol: false,
                                off: tok_off,
                                len: self.raw - tok_off,
                            });
                            line += 1;
                            col = 0;
                            sol = true;
                            space = false;
                            break;
                        }
                    }
                }
                '/' if self.chars.peek().is_some_and(|(c, _, _)| *c == '*') => {
                    // Block comment: newlines preserved for line counts,
                    // plus one space for the token that follows, mirroring
                    // phase-3 comment removal (clang sets `LeadingSpace`).
                    let start_col = col;
                    col += 2;
                    let _ = self.pp_next(&mut line);
                    let mut newlines = 0u32;
                    loop {
                        match self.pp_next(&mut line) {
                            None => break,
                            Some('*') if self.chars.peek().is_some_and(|(c, _, _)| *c == '/') => {
                                let _ = self.pp_next(&mut line);
                                col += 2;
                                break;
                            }
                            Some('\n') => {
                                newlines += 1;
                                line += 1;
                                col = 0;
                            }
                            Some(nc) => {
                                col += nc.len_utf8() as u64;
                            }
                        }
                    }
                    // Newlines first (original positions), then the space,
                    // mirroring phase-3 comment removal.
                    for _ in 0..newlines {
                        out.push(PpTok {
                            tok: PpToken::Ws(PpWs::Newline),
                            line,
                            col: 0,
                            space: false,
                            sol: false,
                            off: tok_off,
                            len: self.raw - tok_off,
                        });
                    }
                    out.push(PpTok {
                        tok: PpToken::Ws(PpWs::Space),
                        line,
                        col: start_col,
                        space: false,
                        sol: false,
                        off: tok_off,
                        len: self.raw - tok_off,
                    });
                    space = true;
                    sol = tok_sol;
                }
                '#' => {
                    // `##` wins over the directive marker: a paste result
                    // relexed at "line start" is still an operator, not a
                    // directive introducer.
                    if self.chars.peek().is_some_and(|(c, _, _)| *c == '#') {
                        let _ = self.pp_next(&mut line);
                        push!(PpToken::Punct(Punctuator::HashHash), tok_line, col, tok_off);
                    } else if tok_sol {
                        push!(PpToken::Directive, tok_line, col, tok_off);
                    } else {
                        push!(PpToken::Punct(Punctuator::Hash), tok_line, col, tok_off);
                    }
                    col += 1;
                }
                '"' => {
                    let start_col = col;
                    let raw = self.lex_pp_quoted('"', &mut col, &mut line);
                    let sym = self.intern.get_or_intern(raw);
                    push!(PpToken::Str(sym), tok_line, start_col, tok_off);
                }
                '\'' => {
                    let start_col = col;
                    let raw = self.lex_pp_quoted('\'', &mut col, &mut line);
                    let sym = self.intern.get_or_intern(raw);
                    push!(PpToken::Char(sym), tok_line, start_col, tok_off);
                }
                c if c.is_alphabetic() || c == '_' || c == '$' => {
                    let start_col = col;
                    let mut word = String::from(c);
                    col += c.len_utf8() as u64;
                    while let Some((nc, _, _)) = self.chars.peek().copied() {
                        if nc.is_alphanumeric() || nc == '_' || nc == '$' {
                            word.push(nc);
                            col += nc.len_utf8() as u64;
                            let _ = self.pp_next(&mut line);
                        } else {
                            break;
                        }
                    }
                    // String/char prefixes (`u8".."`, `L'a'`): the quote
                    // joins the literal; anything else is an identifier.
                    if matches!(word.as_str(), "u8" | "u" | "U" | "L") {
                        if let Some((q, _, _)) = self.chars.peek().copied() {
                            if q == '"' || q == '\'' {
                                let _ = self.pp_next(&mut line);
                                let raw = self.lex_pp_quoted(q, &mut col, &mut line);
                                let sym = self.intern.get_or_intern(format!("{word}{raw}"));
                                let tok = if q == '"' { PpToken::Str(sym) } else { PpToken::Char(sym) };
                                push!(tok, tok_line, start_col, tok_off);
                                continue;
                            }
                        }
                    }
                    let sym = self.intern.get_or_intern(word);
                    push!(PpToken::Ident(sym), tok_line, start_col, tok_off);
                }
                c if c.is_ascii_digit() || (c == '.' && self.chars.peek().is_some_and(|(nc, _, _)| nc.is_ascii_digit())) => {
                    let start_col = col;
                    let raw = self.lex_pp_number(c, &mut col, &mut line);
                    let sym = self.intern.get_or_intern(raw);
                    push!(PpToken::PpNumber(sym), tok_line, start_col, tok_off);
                }
                _ => {
                    if let Some((punct, len)) = self.lex_pp_punct(c, &mut line) {
                        push!(PpToken::Punct(punct), tok_line, col, tok_off);
                        col += len as u64;
                    } else {
                        push!(PpToken::Other(c), tok_line, col, tok_off);
                        col += c.len_utf8() as u64;
                    }
                }
            }
        }
        out
    }

    /// Raw quoted literal starting at the already-seen quote `q`:
    /// consumes through the closing quote (backslash escapes honored) or
    /// to end-of-line/EOF, whichever comes first. Returns the raw text
    /// *including* both quotes (or the unterminated tail).
    fn lex_pp_quoted(&mut self, q: char, col: &mut u64, line: &mut u64) -> String {
        let mut raw = String::from(q);
        *col += 1;
        loop {
            match self.pp_next(line) {
                None => break,
                Some('\n') | Some('\r') => break,
                Some('\\') => {
                    raw.push('\\');
                    *col += 1;
                    // Keep the escaped char raw (even a quote or newline).
                    if let Some(e) = self.pp_next(line) {
                        raw.push(e);
                        *col += e.len_utf8() as u64;
                    } else {
                        break;
                    }
                }
                Some(ch) => {
                    raw.push(ch);
                    *col += ch.len_utf8() as u64;
                    if ch == q {
                        break;
                    }
                }
            }
        }
        raw
    }

    /// Raw preprocessing number starting with `first` (a digit, or `.`
    /// before a digit): consumes `[0-9a-zA-Z_.']*` plus one exponent sign
    /// after `e`/`E`/`p`/`P`.
    fn lex_pp_number(&mut self, first: char, col: &mut u64, line: &mut u64) -> String {
        let mut raw = String::from(first);
        *col += first.len_utf8() as u64;
        let mut exp_sign = false;
        while let Some((nc, _, _)) = self.chars.peek().copied() {
            if nc.is_ascii_alphanumeric() || nc == '.' || nc == '_' || nc == '\'' {
                // A sign continues the pp-number only right after an
                // `e`/`E`/`p`/`P` (C11 6.4.8), so the flag follows the
                // character just consumed: `0xE1+` splits, `0xE+` joins.
                exp_sign = matches!(nc, 'e' | 'E' | 'p' | 'P');
                raw.push(nc);
                *col += nc.len_utf8() as u64;
                let _ = self.pp_next(line);
            } else if (nc == '+' || nc == '-') && exp_sign {
                exp_sign = false;
                raw.push(nc);
                *col += 1;
                let _ = self.pp_next(line);
            } else {
                break;
            }
        }
        raw
    }

    /// Longest-match C punctuator starting with the already-seen `first`,
    /// or `None` when `first` starts no punctuator.
    fn lex_pp_punct(&mut self, first: char, line: &mut u64) -> Option<(Punctuator, usize)> {
        let p0 = self.chars.peek().copied().map(|(c, _, _)| c);
        let p1 = self.chars.peek_nth(1).copied().map(|(c, _, _)| c);
        let found = match (first, p0, p1) {
            ('.', Some('.'), Some('.')) => Some((Punctuator::Ellipsis, 3)),
            ('<', Some('<'), Some('=')) => Some((Punctuator::LessLessEq, 3)),
            ('>', Some('>'), Some('=')) => Some((Punctuator::GreaterGreaterEq, 3)),
            ('-', Some('>'), _) => Some((Punctuator::Arrow, 2)),
            ('+', Some('+'), _) => Some((Punctuator::PlusPlus, 2)),
            ('-', Some('-'), _) => Some((Punctuator::MinusMinus, 2)),
            ('<', Some('<'), _) => Some((Punctuator::LessLess, 2)),
            ('>', Some('>'), _) => Some((Punctuator::GreaterGreater, 2)),
            ('<', Some('='), _) => Some((Punctuator::LessEq, 2)),
            ('>', Some('='), _) => Some((Punctuator::GreaterEq, 2)),
            ('=', Some('='), _) => Some((Punctuator::EqEq, 2)),
            ('!', Some('='), _) => Some((Punctuator::BangEq, 2)),
            ('&', Some('&'), _) => Some((Punctuator::AmpAmp, 2)),
            ('|', Some('|'), _) => Some((Punctuator::PipePipe, 2)),
            ('+', Some('='), _) => Some((Punctuator::PlusEq, 2)),
            ('-', Some('='), _) => Some((Punctuator::MinusEq, 2)),
            ('*', Some('='), _) => Some((Punctuator::StarEq, 2)),
            ('/', Some('='), _) => Some((Punctuator::SlashEq, 2)),
            ('%', Some('='), _) => Some((Punctuator::PercentEq, 2)),
            ('&', Some('='), _) => Some((Punctuator::AmpEq, 2)),
            ('^', Some('='), _) => Some((Punctuator::CaretEq, 2)),
            ('|', Some('='), _) => Some((Punctuator::PipeEq, 2)),
            ('[', _, _) => Some((Punctuator::LBracket, 1)),
            (']', _, _) => Some((Punctuator::RBracket, 1)),
            ('(', _, _) => Some((Punctuator::LParen, 1)),
            (')', _, _) => Some((Punctuator::RParen, 1)),
            ('{', _, _) => Some((Punctuator::LBrace, 1)),
            ('}', _, _) => Some((Punctuator::RBrace, 1)),
            ('.', _, _) => Some((Punctuator::Dot, 1)),
            ('&', _, _) => Some((Punctuator::Amp, 1)),
            ('*', _, _) => Some((Punctuator::Star, 1)),
            ('+', _, _) => Some((Punctuator::Plus, 1)),
            ('-', _, _) => Some((Punctuator::Minus, 1)),
            ('~', _, _) => Some((Punctuator::Tilde, 1)),
            ('!', _, _) => Some((Punctuator::Bang, 1)),
            ('/', _, _) => Some((Punctuator::Slash, 1)),
            ('%', _, _) => Some((Punctuator::Percent, 1)),
            ('<', _, _) => Some((Punctuator::Less, 1)),
            ('>', _, _) => Some((Punctuator::Greater, 1)),
            ('^', _, _) => Some((Punctuator::Caret, 1)),
            ('|', _, _) => Some((Punctuator::Pipe, 1)),
            ('?', _, _) => Some((Punctuator::Question, 1)),
            (':', _, _) => Some((Punctuator::Colon, 1)),
            (';', _, _) => Some((Punctuator::Semi, 1)),
            ('=', _, _) => Some((Punctuator::Assign, 1)),
            (',', _, _) => Some((Punctuator::Comma, 1)),
            _ => None,
        };
        if let Some((punct, len)) = found {
            // `first` is already consumed; eat the rest.
            for _ in 1..len {
                let _ = self.pp_next(line);
            }
            Some((punct, len))
        } else {
            None
        }
    }
}

/// Owning buffer of eagerly-lexed tokens with peek/advance cursor.
#[derive(Debug)]
pub struct TokenStream {
    tokens: Vec<Located<Token>>,
    pos: usize,
    /// Lexical errors recovered during lexing, in source order. The
    /// parser adopts them so callers see them alongside parse/check
    /// diagnostics.
    diagnostics: Vec<CompileError>,
}

impl TokenStream {
    pub fn new(tokens: Vec<Located<Token>>) -> Self {
        Self {
            tokens,
            pos: 0,
            diagnostics: Vec::new(),
        }
    }

    pub fn with_diagnostics(tokens: Vec<Located<Token>>, diagnostics: Vec<CompileError>) -> Self {
        Self {
            tokens,
            pos: 0,
            diagnostics,
        }
    }

    /// Lexical errors that lexing recovered from (unterminated literals,
    /// malformed UCNs, stray characters).
    pub fn diagnostics(&self) -> &[CompileError] {
        &self.diagnostics
    }

    /// Peek the next token without consuming.
    pub fn peek(&self) -> Option<&Located<Token>> {
        self.peek_nth(0)
    }
    pub fn peek_val(&self) -> Option<&Token> {
        self.peek_nth(0).map(|v| &v.val)
    }
    /// Peek the nth upcoming token without consuming (`n = 0` is next).
    pub fn peek_nth(&self, n: usize) -> Option<&Located<Token>> {
        self.tokens.get(self.pos + n)
    }

    /// Consume and return the next token.
    pub fn advance(&mut self) -> Option<&Located<Token>> {
        let i = self.pos;
        if i >= self.tokens.len() {
            return None;
        }
        self.pos += 1;
        self.tokens.get(i)
    }

    /// The most recently consumed token, if any.
    pub fn prev(&self) -> Option<&Located<Token>> {
        if self.pos == 0 { None } else { self.tokens.get(self.pos - 1) }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    /// All lexed tokens, including already-consumed ones: used by
    /// [`Parser::new`](crate::parse::Parser::new) to pre-register the
    /// per-file diagnostic names (linemarkers give each token its own
    /// file).
    pub fn iter(&self) -> core::slice::Iter<'_, Located<Token>> {
        self.tokens.iter()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Token {
    Keyword(Keyword),
    Identifier(SymbolU32),
    Constant(Constant),
    StringLiteral(StringLiteral),
    Punctuator(Punctuator),
}
#[derive(Debug, Clone, Copy)]
pub enum Constant {
    Int(IntLiteral),
    Float(FloatLiteral),
    Char(CharLiteral),
}
/// Base of a constant's digit part.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Radix {
    Oct,
    Dec,
    Hex,
    Bin,
}
/// `u`/`U` and/or `l`/`L`/`ll`/`LL` suffix of an integer constant, plus
/// MSVC's fixed-width `__intN` suffixes (`i8`..`i64`, `ui8`..`ui64`).
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntSuffix {
    None,
    U,
    L,
    UL,
    LL,
    ULL,
    I8,
    I16,
    I32,
    I64,
    UI8,
    UI16,
    UI32,
    UI64,
}
/// `f`/`F` (`float`) or `l`/`L` (`long double`) suffix of a float constant;
/// no suffix means `double`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatSuffix {
    None,
    F,
    L,
}
/// An unparsed integer constant: the raw spelling plus the syntactic
/// properties from which its type and value are derived later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntLiteral {
    pub spelling: SymbolU32,
    pub radix: Radix,
    pub suffix: IntSuffix,
    /// GNU imaginary suffix (`1i`, `1j`): the value is the imaginary
    /// part of a complex number, not a real integer. The spelling keeps
    /// only the real-suffix part so value parsing is unchanged.
    pub imag: bool,
}
/// Encoding prefix of a character constant (C17 §6.4.4.4): `L'a'`, `u'a'`,
/// `U'a'`, or plain `'a'`. (`u8` is only a string-literal prefix, so it is
/// not a valid char prefix.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharPrefix {
    None,
    L,
    SmallU,
    BigU,
}

impl CharPrefix {
    pub fn as_str(self) -> &'static str {
        match self {
            CharPrefix::None => "",
            CharPrefix::L => "L",
            CharPrefix::SmallU => "u",
            CharPrefix::BigU => "U",
        }
    }
}

/// An unparsed character constant: the raw spelling (prefix + quotes +
/// raw body, escapes unprocessed) plus its encoding prefix, like
/// [`IntLiteral`] and [`FloatLiteral`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharLiteral {
    pub spelling: SymbolU32,
    pub prefix: CharPrefix,
}
/// Encoding prefix of a string literal (C17 §6.4.5): `u8"..."`, `u"..."`,
/// `U"..."`, `L"..."`, or plain `"..."`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringPrefix {
    None,
    L,
    SmallU,
    BigU,
    U8,
}

impl StringPrefix {
    pub fn as_str(self) -> &'static str {
        match self {
            StringPrefix::None => "",
            StringPrefix::L => "L",
            StringPrefix::SmallU => "u",
            StringPrefix::BigU => "U",
            StringPrefix::U8 => "u8",
        }
    }
}

/// An unparsed string literal: the raw spelling (prefix + quotes + raw
/// body, escapes unprocessed) plus its encoding prefix, mirroring
/// `CharLiteral`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StringLiteral {
    pub spelling: SymbolU32,
    pub prefix: StringPrefix,
}
/// An unparsed floating constant: the raw spelling plus its base and suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FloatLiteral {
    pub spelling: SymbolU32,
    pub radix: Radix,
    pub suffix: FloatSuffix,
    /// GNU imaginary suffix (`1.0i`, `1.0fi`, `1.0if`): the value is the
    /// imaginary part of a complex number. The spelling keeps only the
    /// real-suffix part so value parsing is unchanged.
    pub imag: bool,
}
impl Constant {
    pub fn spelling(&self) -> SymbolU32 {
        match self {
            Constant::Int(i) => i.spelling,
            Constant::Float(f) => f.spelling,
            Constant::Char(c) => c.spelling,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keyword {
    Auto,
    Break,
    Case,
    Char,
    Const,
    Continue,
    Default,
    Do,
    Double,
    Else,
    Enum,
    Extern,
    Float,
    For,
    Goto,
    If,
    Inline,
    Int,
    Long,
    Register,
    Restrict,
    Return,
    Short,
    Signed,
    Sizeof,
    Static,
    Struct,
    Switch,
    Typedef,
    Union,
    Unsigned,
    Void,
    Volatile,
    While,
    _Bool,
    _Complex,
    _Imaginary,
    _Atomic,
    _Noreturn,
}
impl Keyword {
    pub fn as_str(&self) -> &'static str {
        match self {
            Keyword::Auto => "auto",
            Keyword::Break => "break",
            Keyword::Case => "case",
            Keyword::Char => "char",
            Keyword::Const => "const",
            Keyword::Continue => "continue",
            Keyword::Default => "default",
            Keyword::Do => "do",
            Keyword::Double => "double",
            Keyword::Else => "else",
            Keyword::Enum => "enum",
            Keyword::Extern => "extern",
            Keyword::Float => "float",
            Keyword::For => "for",
            Keyword::Goto => "goto",
            Keyword::If => "if",
            Keyword::Inline => "inline",
            Keyword::Int => "int",
            Keyword::Long => "long",
            Keyword::Register => "register",
            Keyword::Restrict => "restrict",
            Keyword::Return => "return",
            Keyword::Short => "short",
            Keyword::Signed => "signed",
            Keyword::Sizeof => "sizeof",
            Keyword::Static => "static",
            Keyword::Struct => "struct",
            Keyword::Switch => "switch",
            Keyword::Typedef => "typedef",
            Keyword::Union => "union",
            Keyword::Unsigned => "unsigned",
            Keyword::Void => "void",
            Keyword::Volatile => "volatile",
            Keyword::While => "while",
            Keyword::_Bool => "_Bool",
            Keyword::_Complex => "_Complex",
            Keyword::_Imaginary => "_Imaginary",
            Keyword::_Atomic => "_Atomic",
            Keyword::_Noreturn => "_Noreturn",
        }
    }
}
/// Every punctuator from C17 §6.4.6, excluding digraphs / alternative
/// spellings (C17 §6.4.6p3) which we deliberately do not support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Punctuator {
    // single-character
    LBracket, // [
    RBracket, // ]
    LParen,   // (
    RParen,   // )
    LBrace,   // {
    RBrace,   // }
    Dot,      // .
    Amp,      // &
    Star,     // *
    Plus,     // +
    Minus,    // -
    Tilde,    // ~
    Bang,     // !
    Slash,    // /
    Percent,  // %
    Less,     // <
    Greater,  // >
    Caret,    // ^
    Pipe,     // |
    Question, // ?
    Colon,    // :
    Semi,     // ;
    Assign,   // =
    Comma,    // ,
    Hash,     // #

    // two-character
    Arrow,          // ->
    PlusPlus,       // ++
    MinusMinus,     // --
    LessLess,       // <<
    GreaterGreater, // >>
    LessEq,         // <=
    GreaterEq,      // >=
    EqEq,           // ==
    BangEq,         // !=
    AmpAmp,         // &&
    PipePipe,       // ||
    PlusEq,         // +=
    MinusEq,        // -=
    StarEq,         // *=
    SlashEq,        // /=
    PercentEq,      // %=
    AmpEq,          // &=
    CaretEq,        // ^=
    PipeEq,         // |=
    HashHash,       // ##

    // three-character
    Ellipsis,         // ...
    LessLessEq,       // <<=
    GreaterGreaterEq, // >>=
}

impl Punctuator {
    pub fn as_str(self) -> &'static str {
        match self {
            Punctuator::LBracket => "[",
            Punctuator::RBracket => "]",
            Punctuator::LParen => "(",
            Punctuator::RParen => ")",
            Punctuator::LBrace => "{",
            Punctuator::RBrace => "}",
            Punctuator::Dot => ".",
            Punctuator::Amp => "&",
            Punctuator::Star => "*",
            Punctuator::Plus => "+",
            Punctuator::Minus => "-",
            Punctuator::Tilde => "~",
            Punctuator::Bang => "!",
            Punctuator::Slash => "/",
            Punctuator::Percent => "%",
            Punctuator::Less => "<",
            Punctuator::Greater => ">",
            Punctuator::Caret => "^",
            Punctuator::Pipe => "|",
            Punctuator::Question => "?",
            Punctuator::Colon => ":",
            Punctuator::Semi => ";",
            Punctuator::Assign => "=",
            Punctuator::Comma => ",",
            Punctuator::Hash => "#",
            Punctuator::Arrow => "->",
            Punctuator::PlusPlus => "++",
            Punctuator::MinusMinus => "--",
            Punctuator::LessLess => "<<",
            Punctuator::GreaterGreater => ">>",
            Punctuator::LessEq => "<=",
            Punctuator::GreaterEq => ">=",
            Punctuator::EqEq => "==",
            Punctuator::BangEq => "!=",
            Punctuator::AmpAmp => "&&",
            Punctuator::PipePipe => "||",
            Punctuator::PlusEq => "+=",
            Punctuator::MinusEq => "-=",
            Punctuator::StarEq => "*=",
            Punctuator::SlashEq => "/=",
            Punctuator::PercentEq => "%=",
            Punctuator::AmpEq => "&=",
            Punctuator::CaretEq => "^=",
            Punctuator::PipeEq => "|=",
            Punctuator::HashHash => "##",
            Punctuator::Ellipsis => "...",
            Punctuator::LessLessEq => "<<=",
            Punctuator::GreaterGreaterEq => ">>=",
        }
    }
}

impl<'a> Lexer<'a> {
    /// Peek the nth upcoming logical char without consuming.
    /// n = 0 is the next char, n = 1 the one after, etc.
    fn peek_char(&mut self, n: usize) -> Option<char> {
        self.chars.peek_nth(n).map(|(c, _, _)| *c)
    }

    /// Consume the next logical char.
    fn bump(&mut self) -> Option<char> {
        self.bump_at().map(|(c, _)| c)
    }

    /// Consume the next logical char, returning it and the position it
    /// starts at. The position effects of any line splices (translation
    /// phase 2) preceding it are applied first, so tokens starting right
    /// after a splice get the correct start location.
    fn bump_at(&mut self) -> Option<(char, Location)> {
        let (c, lines, _) = self.chars.next()?;
        if lines > 0 {
            self.loc.ln += lines;
            self.loc.col = 0;
        }
        let pos = self.loc;
        self.loc.col += 1;
        Some((c, pos))
    }

    fn lex_punctuator(&mut self) -> Located<Token> {
        let (punct, len) = match (self.peek_char(0).expect("lex_punctuator at EOF"), self.peek_char(1), self.peek_char(2)) {
            // three-character
            ('.', Some('.'), Some('.')) => (Punctuator::Ellipsis, 3),
            ('<', Some('<'), Some('=')) => (Punctuator::LessLessEq, 3),
            ('>', Some('>'), Some('=')) => (Punctuator::GreaterGreaterEq, 3),

            // two-character
            ('-', Some('>'), _) => (Punctuator::Arrow, 2),
            ('+', Some('+'), _) => (Punctuator::PlusPlus, 2),
            ('-', Some('-'), _) => (Punctuator::MinusMinus, 2),
            ('<', Some('<'), _) => (Punctuator::LessLess, 2),
            ('>', Some('>'), _) => (Punctuator::GreaterGreater, 2),
            ('<', Some('='), _) => (Punctuator::LessEq, 2),
            ('>', Some('='), _) => (Punctuator::GreaterEq, 2),
            ('=', Some('='), _) => (Punctuator::EqEq, 2),
            ('!', Some('='), _) => (Punctuator::BangEq, 2),
            ('&', Some('&'), _) => (Punctuator::AmpAmp, 2),
            ('|', Some('|'), _) => (Punctuator::PipePipe, 2),
            ('+', Some('='), _) => (Punctuator::PlusEq, 2),
            ('-', Some('='), _) => (Punctuator::MinusEq, 2),
            ('*', Some('='), _) => (Punctuator::StarEq, 2),
            ('/', Some('='), _) => (Punctuator::SlashEq, 2),
            ('%', Some('='), _) => (Punctuator::PercentEq, 2),
            ('&', Some('='), _) => (Punctuator::AmpEq, 2),
            ('^', Some('='), _) => (Punctuator::CaretEq, 2),
            ('|', Some('='), _) => (Punctuator::PipeEq, 2),
            ('#', Some('#'), _) => (Punctuator::HashHash, 2),

            // single-character
            ('[', _, _) => (Punctuator::LBracket, 1),
            (']', _, _) => (Punctuator::RBracket, 1),
            ('(', _, _) => (Punctuator::LParen, 1),
            (')', _, _) => (Punctuator::RParen, 1),
            ('{', _, _) => (Punctuator::LBrace, 1),
            ('}', _, _) => (Punctuator::RBrace, 1),
            ('.', _, _) => (Punctuator::Dot, 1),
            ('&', _, _) => (Punctuator::Amp, 1),
            ('*', _, _) => (Punctuator::Star, 1),
            ('+', _, _) => (Punctuator::Plus, 1),
            ('-', _, _) => (Punctuator::Minus, 1),
            ('~', _, _) => (Punctuator::Tilde, 1),
            ('!', _, _) => (Punctuator::Bang, 1),
            ('/', _, _) => (Punctuator::Slash, 1),
            ('%', _, _) => (Punctuator::Percent, 1),
            ('<', _, _) => (Punctuator::Less, 1),
            ('>', _, _) => (Punctuator::Greater, 1),
            ('^', _, _) => (Punctuator::Caret, 1),
            ('|', _, _) => (Punctuator::Pipe, 1),
            ('?', _, _) => (Punctuator::Question, 1),
            (':', _, _) => (Punctuator::Colon, 1),
            (';', _, _) => (Punctuator::Semi, 1),
            ('=', _, _) => (Punctuator::Assign, 1),
            (',', _, _) => (Punctuator::Comma, 1),
            ('#', _, _) => (Punctuator::Hash, 1),

            (c, _, _) => unreachable!("lex_punctuator called on non-punctuator {c:?}"),
        };

        let (_, start) = self.bump_at().expect("lex_punctuator at EOF");
        for _ in 1..len {
            self.bump();
        }
        Located {
            loc: start,
            end: self.loc,
            val: Token::Punctuator(punct),
        }
    }

    /// Consume the next char into `buf`.
    fn bump_into(&mut self, buf: &mut String) {
        buf.push(self.bump().expect("bump_into at EOF"));
    }

    /// Consume the `u`/`U` and following hexadecimal digits of a
    /// universal character name (C17 §6.4.3), appending the raw text to
    /// `buf` and returning the named character. The leading backslash is
    /// already consumed (identifier) or appended (literal). Malformed,
    /// surrogate, and out-of-range names are lexer errors like the other
    /// panics here (translation phase 1 would have diagnosed them before
    /// tokenizing).
    fn lex_ucn_into(&mut self, buf: &mut String) -> char {
        let start = self.loc;
        let small = match self.bump() {
            Some('u') => true,
            Some('U') => false,
            Some(other) => {
                self.error(start, self.loc, format!("invalid universal character name: `\\{other}` does not start one"));
                return char::REPLACEMENT_CHARACTER;
            }
            None => {
                self.error(start, self.loc, "invalid universal character name at end of input");
                return char::REPLACEMENT_CHARACTER;
            }
        };
        buf.push(if small { 'u' } else { 'U' });
        let digits = if small { 4 } else { 8 };
        let mut value = 0u32;
        for _ in 0..digits {
            let Some(d) = self.peek_char(0).and_then(|c| c.to_digit(16)) else {
                self.error(start, self.loc, format!("invalid universal character name: expected {digits} hexadecimal digits"));
                return char::REPLACEMENT_CHARACTER;
            };
            buf.push(self.bump().unwrap());
            value = value * 16 + d;
        }
        let Some(c) = char::from_u32(value) else {
            self.error(start, self.loc, format!("universal character name \\U{value:08X} is not a valid character"));
            return char::REPLACEMENT_CHARACTER;
        };
        c
    }

    /// Consume a universal character name whose backslash is the current
    /// character and return the named character.
    fn lex_ucn(&mut self) -> char {
        self.bump();
        let mut raw = String::new();
        self.lex_ucn_into(&mut raw)
    }

    /// Lex an identifier (or keyword, per `Keyword::try_from`) whose
    /// first character `first` was already consumed at `start`.
    /// Identifier-continue characters include further UCNs
    /// (C17 §6.4.2.1).
    fn lex_identifier(&mut self, first: char, start: Location) -> Located<Token> {
        let mut identifier = String::from(first);
        loop {
            if self.peek_char(0) == Some('\\') {
                let c = self.lex_ucn();
                if !(c.is_alphanumeric() || c == '_' || c == '$') {
                    self.error(start, self.loc, format!("universal character name `{c}` is not valid in an identifier"));
                    identifier.push(c);
                    continue;
                }
                identifier.push(c);
                continue;
            }
            match self.peek_char(0) {
                Some(c) if c.is_alphanumeric() || c == '_' || c == '$' => identifier.push(self.bump().unwrap()),
                _ => break,
            }
        }
        Located {
            loc: start,
            end: self.loc,
            val: match Keyword::try_from(identifier.as_str()) {
                Ok(kw) => Token::Keyword(kw),
                Err(_) => Token::Identifier(self.intern.get_or_intern(identifier)),
            },
        }
    }

    /// Consume a run of digits (hex digits if `hex`) into `buf`.
    fn lex_digits(&mut self, buf: &mut String, hex: bool) {
        self.lex_digit_run(buf, |d| if hex { d.is_ascii_hexdigit() } else { d.is_ascii_digit() });
    }

    /// Digits with C23 digit separators. A `'` is consumed (and left out
    /// of the spelling) only between two digits, so `1'000` is one
    /// constant while `1'` still starts a character constant.
    fn lex_digit_run(&mut self, buf: &mut String, is_digit: impl Fn(char) -> bool) {
        let mut after_sep = false;
        loop {
            let Some(c) = self.peek_char(0) else { break };
            if is_digit(c) {
                self.bump_into(buf);
                after_sep = false;
            } else if c == '\''
                && !after_sep
                && buf.chars().last().is_some_and(&is_digit)
                && matches!(self.peek_char(1), Some(next) if is_digit(next))
            {
                self.bump();
                after_sep = true;
            } else {
                break;
            }
        }
    }

    /// Consume an exponent — `e`/`E` for decimal floats, `p`/`P` for hex
    /// floats — with optional sign and decimal digits. Returns whether one
    /// was present.
    fn lex_exponent(&mut self, buf: &mut String, lo: char, hi: char) -> bool {
        if !matches!(self.peek_char(0), Some(c) if c == lo || c == hi) {
            return false;
        }
        self.bump_into(buf);
        if matches!(self.peek_char(0), Some('+') | Some('-')) {
            self.bump_into(buf);
        }
        self.lex_digits(buf, false);
        true
    }

    /// Consume a run of one or two same-case `l`/`L` into `buf`.
    fn lex_l_run(&mut self, buf: &mut String) -> usize {
        let Some(l) = self.peek_char(0).filter(|c| matches!(*c, 'l' | 'L')) else {
            return 0;
        };
        self.bump_into(buf);
        if self.peek_char(0) == Some(l) {
            self.bump_into(buf);
            return 2;
        }
        1
    }

    /// Consume a `u`/`U` into `buf` if present.
    fn lex_u(&mut self, buf: &mut String) -> bool {
        if matches!(self.peek_char(0), Some('u') | Some('U')) {
            self.bump_into(buf);
            true
        } else {
            false
        }
    }

    /// MSVC's `__intN` literal suffixes: `i8`/`i16`/`i32`/`i64` and the
    /// unsigned `ui8`..`ui64` forms, which select that fixed-width type
    /// regardless of the constant's value (MSVC's <stdint.h> spells
    /// `INT64_MIN` as `-9223372036854775807i64`). Case-insensitive.
    /// Not consumed when the text is not one of these (`1i` stays the
    /// GNU imaginary suffix).
    fn lex_ms_int_suffix(&mut self, buf: &mut String) -> Option<IntSuffix> {
        let i_at = match self.peek_char(0) {
            Some('u') | Some('U') => 1,
            Some('i') | Some('I') => 0,
            _ => return None,
        };
        if !matches!(self.peek_char(i_at), Some('i') | Some('I')) {
            return None;
        }
        let mut width = 0u32;
        let mut digits = 0;
        let mut len = i_at + 1;
        while digits < 3 {
            let Some(digit) = self.peek_char(len).and_then(|c| c.to_digit(10)) else { break };
            if digits == 0 && digit == 0 {
                return None;
            }
            width = width * 10 + digit;
            digits += 1;
            len += 1;
        }
        if matches!(self.peek_char(len), Some(c) if c.is_ascii_digit()) {
            return None;
        }
        let suffix = match (i_at == 1, width) {
            (false, 8) => IntSuffix::I8,
            (false, 16) => IntSuffix::I16,
            (false, 32) => IntSuffix::I32,
            (false, 64) => IntSuffix::I64,
            (true, 8) => IntSuffix::UI8,
            (true, 16) => IntSuffix::UI16,
            (true, 32) => IntSuffix::UI32,
            (true, 64) => IntSuffix::UI64,
            _ => return None,
        };
        for _ in 0..len {
            self.bump_into(buf);
        }
        Some(suffix)
    }

    fn lex_int_suffix(&mut self, buf: &mut String) -> IntSuffix {
        if let Some(suffix) = self.lex_ms_int_suffix(buf) {
            return suffix;
        }
        if self.lex_u(buf) {
            match self.lex_l_run(buf) {
                0 => IntSuffix::U,
                1 => IntSuffix::UL,
                _ => IntSuffix::ULL,
            }
        } else {
            match self.lex_l_run(buf) {
                0 => IntSuffix::None,
                1 if self.lex_u(buf) => IntSuffix::UL,
                1 => IntSuffix::L,
                _ if self.lex_u(buf) => IntSuffix::ULL,
                _ => IntSuffix::LL,
            }
        }
    }

    fn lex_float_suffix(&mut self, buf: &mut String) -> FloatSuffix {
        match self.peek_char(0) {
            Some('f') | Some('F') => {
                self.bump_into(buf);
                FloatSuffix::F
            }
            Some('l') | Some('L') => {
                self.bump_into(buf);
                FloatSuffix::L
            }
            _ => FloatSuffix::None,
        }
    }

    /// Consume a GNU imaginary suffix (`i`/`I`/`j`/`J`) when present,
    /// without recording it in `buf` (the spelling keeps only the
    /// real-suffix part so value parsing is unchanged). Either order
    /// with the real suffix is accepted (`1.0fi`, `1.0if`, `2.2iL`),
    /// so callers retry the real suffix after this.
    fn lex_imaginary_suffix(&mut self) -> bool {
        if matches!(self.peek_char(0), Some('i') | Some('I') | Some('j') | Some('J')) {
            self.bump();
            true
        } else {
            false
        }
    }

    /// Lex an integer (C17 §6.4.4.1) or float (C17 §6.4.4.2) constant.
    /// The caller guarantees the next char starts one: a digit, or a `.`
    /// followed by a digit.
    fn lex_constant(&mut self) -> Located<Token> {
        let (first, start) = self.bump_at().expect("lex_constant at char");
        let mut buf = String::from(first);

        if first == '.' {
            // fractional-only decimal float: `.5`, `.5e-3f`, ...
            self.lex_digits(&mut buf, false);
            self.lex_exponent(&mut buf, 'e', 'E');
            let mut suffix = self.lex_float_suffix(&mut buf);
            let imag = self.lex_imaginary_suffix();
            if imag && suffix == FloatSuffix::None {
                suffix = self.lex_float_suffix(&mut buf);
            }
            return Located {
                loc: start,
                end: self.loc,
                val: Token::Constant(Constant::Float(FloatLiteral {
                    spelling: self.intern.get_or_intern(buf),
                    radix: Radix::Dec,
                    suffix,
                    imag,
                })),
            };
        }

        // C23 binary integer (`0b1010`); no binary floating form.
        if first == '0' && matches!(self.peek_char(0), Some('b') | Some('B')) {
            self.bump_into(&mut buf);
            self.lex_digit_run(&mut buf, |d| matches!(d, '0' | '1'));
            let mut suffix = self.lex_int_suffix(&mut buf);
            let imag = self.lex_imaginary_suffix();
            if imag && suffix == IntSuffix::None {
                suffix = self.lex_int_suffix(&mut buf);
            }
            return Located {
                loc: start,
                end: self.loc,
                val: Token::Constant(Constant::Int(IntLiteral {
                    spelling: self.intern.get_or_intern(buf),
                    radix: Radix::Bin,
                    suffix,
                    imag,
                })),
            };
        }

        // Hex integer or hex float.
        if first == '0' && matches!(self.peek_char(0), Some('x') | Some('X')) {
            self.bump_into(&mut buf);
            self.lex_digits(&mut buf, true);
            // Fractional part; `0x1...` must lex as int `0x1` + `...`, and a
            // trailing `.` with nothing hex after it stays with the punctuator.
            let has_frac = self.peek_char(0) == Some('.')
                && !matches!(self.peek_char(1), Some('.') | None)
                && (matches!(self.peek_char(1), Some('p') | Some('P')) || matches!(self.peek_char(1), Some(d) if d.is_ascii_hexdigit()));
            if has_frac {
                self.bump_into(&mut buf);
                self.lex_digits(&mut buf, true);
            }
            let has_exp = self.lex_exponent(&mut buf, 'p', 'P');
            let is_float = has_frac || has_exp;
            if is_float {
                let mut suffix = self.lex_float_suffix(&mut buf);
                let imag = self.lex_imaginary_suffix();
                if imag && suffix == FloatSuffix::None {
                    suffix = self.lex_float_suffix(&mut buf);
                }
                return Located {
                    loc: start,
                    end: self.loc,
                    val: Token::Constant(Constant::Float(FloatLiteral {
                        spelling: self.intern.get_or_intern(buf),
                        radix: Radix::Hex,
                        suffix,
                        imag,
                    })),
                };
            }
            let mut suffix = self.lex_int_suffix(&mut buf);
            let imag = self.lex_imaginary_suffix();
            if imag && suffix == IntSuffix::None {
                suffix = self.lex_int_suffix(&mut buf);
            }
            return Located {
                loc: start,
                end: self.loc,
                val: Token::Constant(Constant::Int(IntLiteral {
                    spelling: self.intern.get_or_intern(buf),
                    radix: Radix::Hex,
                    suffix,
                    imag,
                })),
            };
        }

        self.lex_digits(&mut buf, false);
        let mut is_float = false;
        // `1.` is a float, but `1..2` must keep the dots for the punctuator arm.
        if self.peek_char(0) == Some('.') && !matches!(self.peek_char(1), Some('.')) {
            self.bump_into(&mut buf);
            self.lex_digits(&mut buf, false);
            is_float = true;
        }
        if self.lex_exponent(&mut buf, 'e', 'E') {
            is_float = true;
        }
        if is_float {
            let mut suffix = self.lex_float_suffix(&mut buf);
            let imag = self.lex_imaginary_suffix();
            if imag && suffix == FloatSuffix::None {
                suffix = self.lex_float_suffix(&mut buf);
            }
            Located {
                loc: start,
                end: self.loc,
                val: Token::Constant(Constant::Float(FloatLiteral {
                    spelling: self.intern.get_or_intern(buf),
                    radix: Radix::Dec,
                    suffix,
                    imag,
                })),
            }
        } else {
            let mut suffix = self.lex_int_suffix(&mut buf);
            let imag = self.lex_imaginary_suffix();
            if imag && suffix == IntSuffix::None {
                suffix = self.lex_int_suffix(&mut buf);
            }
            let radix = if buf.starts_with('0') { Radix::Oct } else { Radix::Dec };
            Located {
                loc: start,
                end: self.loc,
                val: Token::Constant(Constant::Int(IntLiteral {
                    spelling: self.intern.get_or_intern(buf),
                    radix,
                    suffix,
                    imag,
                })),
            }
        }
    }

    /// Lex a character constant (C17 §6.4.4.4): `'a'`, `'\n'`, `'ab'`,
    /// `L'a'`, `u'a'`, `U'a'`. The caller guarantees the next char starts
    /// one: `'` for a plain constant, or `L`/`u`/`U` immediately followed
    /// by `'` for a prefixed one.
    fn lex_char_constant(&mut self) -> Located<Token> {
        let (first, start) = self.bump_at().expect("lex_char_constant at EOF");
        let (prefix, mut buf) = match first {
            '\'' => (CharPrefix::None, String::from("'")),
            'L' => (CharPrefix::L, String::from("L")),
            'u' => (CharPrefix::SmallU, String::from("u")),
            'U' => (CharPrefix::BigU, String::from("U")),
            c => unreachable!("lex_char_constant called on {c:?}"),
        };
        if prefix != CharPrefix::None {
            assert_eq!(self.bump(), Some('\''), "lex_char_constant: prefix without opening quote");
            buf.push('\'');
        }
        loop {
            match self.peek_char(0) {
                None => {
                    self.error(start, self.loc, format!("unterminated character constant in {buf:?}"));
                    break;
                }
                Some('\'') => {
                    self.bump_into(&mut buf);
                    break;
                }
                // Keep escapes raw (`'\n'` stays backslash + `n`);
                // consuming both keeps `'\''` from ending the constant
                // early (the string-literal lexer below does the same for
                // `\"`). UCNs stay in the spelling but are validated now.
                Some('\\') => {
                    self.bump_into(&mut buf);
                    match self.peek_char(0) {
                        Some('u' | 'U') => {
                            self.lex_ucn_into(&mut buf);
                        }
                        Some(_) => self.bump_into(&mut buf),
                        None => {
                            self.error(start, self.loc, format!("unterminated character constant in {buf:?}"));
                            break;
                        }
                    }
                }
                Some(_) => self.bump_into(&mut buf),
            }
        }
        Located {
            loc: start,
            end: self.loc,
            val: Token::Constant(Constant::Char(CharLiteral {
                spelling: self.intern.get_or_intern(buf),
                prefix,
            })),
        }
    }

    /// Lex a string literal (C17 §6.4.5): `"..."`, `L"..."`, `u"..."`,
    /// `U"..."`, `u8"..."`. The caller guarantees the next chars start
    /// one: `"` for a plain literal, `L`/`u`/`U` immediately followed by
    /// `"`, or `u8` immediately followed by `"` for a prefixed one.
    fn lex_string_literal(&mut self) -> Located<Token> {
        let (first, start) = self.bump_at().expect("lex_string_literal at EOF");
        let (prefix, mut buf) = match first {
            '"' => (StringPrefix::None, String::from("\"")),
            'L' => (StringPrefix::L, String::from("L")),
            'U' => (StringPrefix::BigU, String::from("U")),
            'u' => {
                let mut buf = String::from("u");
                if self.peek_char(0) == Some('8') {
                    self.bump_into(&mut buf);
                    (StringPrefix::U8, buf)
                } else {
                    (StringPrefix::SmallU, buf)
                }
            }
            c => unreachable!("lex_string_literal called on {c:?}"),
        };
        if prefix != StringPrefix::None {
            assert_eq!(self.bump(), Some('"'), "lex_string_literal: prefix without opening quote");
            buf.push('"');
        }
        loop {
            match self.peek_char(0) {
                None => {
                    self.error(start, self.loc, format!("unterminated string literal in {buf:?}"));
                    break;
                }
                Some('"') => {
                    self.bump_into(&mut buf);
                    break;
                }
                // s-char excludes newline (C17 §6.4.5p1); a raw newline
                // here means the literal was never closed on its line.
                // (Backslash-newline splices are already deleted in
                // translation phase 2, so they never reach this arm.)
                Some('\n') => {
                    self.error(start, self.loc, format!("unterminated string literal in {buf:?}"));
                    break;
                }
                // Keep escapes raw (`"\n"` stays backslash + `n`);
                // consuming both keeps `\"` from ending the literal early.
                // UCNs stay in the spelling but are validated now.
                Some('\\') => {
                    self.bump_into(&mut buf);
                    match self.peek_char(0) {
                        Some('u' | 'U') => {
                            self.lex_ucn_into(&mut buf);
                        }
                        Some(_) => self.bump_into(&mut buf),
                        None => {
                            self.error(start, self.loc, format!("unterminated string literal in {buf:?}"));
                            break;
                        }
                    }
                }
                Some(_) => self.bump_into(&mut buf),
            }
        }
        Located {
            loc: start,
            end: self.loc,
            val: Token::StringLiteral(StringLiteral {
                spelling: self.intern.get_or_intern(buf),
                prefix,
            }),
        }
    }

    fn next_token(&mut self) -> Option<Located<Token>> {
        loop {
            let (ch, _, _) = *self.chars.peek()?;
            match ch {
                '#' if self.loc.col == 0 => {
                    self.bump();
                    // A linemarker (`# 12 "file" flags`) carries a line
                    // number; anything else on a `#` line is a directive
                    // the preprocessor passed through (`#pragma ...`
                    // survives `clang -E` in system headers) and is
                    // skipped to end of line.
                    let mut i = 0;
                    while self.peek_char(i).is_some_and(|c| c == ' ' || c == '\t') {
                        i += 1;
                    }
                    if !self.peek_char(i).is_some_and(|c| c.is_ascii_digit()) {
                        while self.peek_char(0).is_some_and(|c| c != '\n') {
                            self.bump();
                        }
                        continue;
                    }
                    while self.peek_char(0).is_some_and(|c| c == ' ' || c == '\t') {
                        self.bump();
                    }
                    enum LineMarkerStage {
                        LineNumber,
                        FilePath,
                        Flags,
                    }
                    let mut stage = LineMarkerStage::LineNumber;

                    loop {
                        if self.peek_char(0) == Some('\n') {
                            assert_eq!(self.bump(), Some('\n'));
                            self.loc.ln += 1;
                            self.loc.col = 0;
                            break;
                        }
                        match stage {
                            LineMarkerStage::LineNumber => {
                                let mut number = String::new();
                                while self.peek_char(0).is_some_and(|c| c.is_ascii_digit()) {
                                    number.push(self.bump().unwrap());
                                }
                                // No file field: the marker carries
                                // only a line number.
                                if self.peek_char(0) == Some('\n') {
                                    assert_eq!(self.bump(), Some('\n'));
                                    self.loc.ln += 1;
                                    self.loc.col = 0;
                                    break;
                                }
                                assert_eq!(self.bump(), Some(' '));
                                stage = LineMarkerStage::FilePath;
                            }
                            LineMarkerStage::FilePath => {
                                let mut filepath = String::new();
                                assert_eq!(self.bump(), Some('"'));
                                while self.peek_char(0) != Some('"') {
                                    filepath.push(self.bump().unwrap());
                                }
                                assert_eq!(self.bump(), Some('"'));
                                self.loc.file = self.intern.get_or_intern(filepath);
                                stage = LineMarkerStage::Flags
                            }
                            // The preprocessor has already applied the
                            // flags; skip to end of line.
                            LineMarkerStage::Flags => {
                                while self.peek_char(0).is_some_and(|c| c != '\n') {
                                    self.bump();
                                }
                            }
                        }
                    }
                    continue;
                }
                '\n' => {
                    self.bump();
                    self.loc.col = 0;
                    self.loc.ln += 1;
                    continue;
                }
                ' ' | '\x0C' | '\r' | '\t' => {
                    self.bump();
                    continue;
                }
                // Translation phase 3: comments are whitespace. Literal
                // arms consume their own text, so `"//"` never gets here.
                '/' if self.peek_char(1) == Some('/') => {
                    // Skip to (not past) the newline.
                    self.bump();
                    self.bump();
                    while self.peek_char(0).is_some_and(|c| c != '\n') {
                        self.bump();
                    }
                    continue;
                }
                '/' if self.peek_char(1) == Some('*') => {
                    let start = self.loc;
                    self.bump();
                    self.bump();
                    loop {
                        match self.peek_char(0) {
                            None => {
                                self.error(start, self.loc, "unterminated block comment");
                                break;
                            }
                            Some('*') if self.peek_char(1) == Some('/') => {
                                self.bump();
                                self.bump();
                                break;
                            }
                            // Newlines still advance the reported line.
                            Some('\n') => {
                                self.bump();
                                self.loc.col = 0;
                                self.loc.ln += 1;
                            }
                            Some(_) => {
                                self.bump();
                            }
                        }
                    }
                    continue;
                }
                // C17 §6.4.4.4: character constants, plain or L/u/U-prefixed.
                // Must come before the identifier arm so `L'a'` etc. don't
                // lex as an identifier.
                '\'' => return Some(self.lex_char_constant()),
                'L' | 'u' | 'U' if self.peek_char(1) == Some('\'') => {
                    return Some(self.lex_char_constant());
                }
                // C17 §6.4.5: string literals, plain or L/u/U/u8-prefixed.
                // Must come before the identifier arm so `L"..."` etc. don't
                // lex as an identifier.
                '"' => return Some(self.lex_string_literal()),
                'L' | 'U' if self.peek_char(1) == Some('"') => {
                    return Some(self.lex_string_literal());
                }
                'u' if self.peek_char(1) == Some('"') || (self.peek_char(1) == Some('8') && self.peek_char(2) == Some('"')) => {
                    return Some(self.lex_string_literal());
                }
                // GNU `$` in identifiers (EDG/linker-generated names
                // like `vec$I...`): accepted like GCC.
                c if c.is_alphabetic() || c == '_' || c == '$' => {
                    let (c, start) = self.bump_at()?;
                    return Some(self.lex_identifier(c, start));
                }
                // A universal character name (C17 §6.4.2.1) can also open
                // an identifier (`int \u00e9;`).
                '\\' => {
                    let start = self.loc;
                    let c = self.lex_ucn();
                    if !(c.is_alphabetic() || c == '_' || c == '$') {
                        self.error(start, self.loc, format!("universal character name `{c}` is not valid at the start of an identifier"));
                    }
                    return Some(self.lex_identifier(c, start));
                }
                // C17 §6.4.4.1/§6.4.4.2: integer and float constants
                c if c.is_ascii_digit() => return Some(self.lex_constant()),
                // fractional-only float: `.5`, `.5e-3f`, ...
                '.' if matches!(self.peek_char(1), Some(d) if d.is_ascii_digit()) => {
                    return Some(self.lex_constant());
                }
                '[' | ']' | '(' | ')' | '{' | '}' | '.' | '&' | '*' | '+' | '-' | '~' | '!' | '/' | '%' | '<' | '>' | '^' | '|' | '?' | ':' | ';' | '=' | ',' | '#' => {
                    return Some(self.lex_punctuator());
                }
                c => {
                    let start = self.loc;
                    self.bump();
                    self.error(start, self.loc, format!("unexpected character {c:?} at line {}, column {}", start.ln, start.col));
                }
            }
        }
    }
}

impl<'a> TryFrom<&'a str> for Keyword {
    type Error = ();

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        Ok(match value {
            "auto" => Keyword::Auto,
            "break" => Keyword::Break,
            "case" => Keyword::Case,
            "char" => Keyword::Char,
            "const" => Keyword::Const,
            "continue" => Keyword::Continue,
            "default" => Keyword::Default,
            "do" => Keyword::Do,
            "double" => Keyword::Double,
            "else" => Keyword::Else,
            "enum" => Keyword::Enum,
            "extern" => Keyword::Extern,
            "float" => Keyword::Float,
            "for" => Keyword::For,
            "goto" => Keyword::Goto,
            "if" => Keyword::If,
            "inline" => Keyword::Inline,
            "int" => Keyword::Int,
            "long" => Keyword::Long,
            "register" => Keyword::Register,
            "restrict" => Keyword::Restrict,
            "return" => Keyword::Return,
            "short" => Keyword::Short,
            "signed" => Keyword::Signed,
            "sizeof" => Keyword::Sizeof,
            "static" => Keyword::Static,
            "struct" => Keyword::Struct,
            "switch" => Keyword::Switch,
            "typedef" => Keyword::Typedef,
            "union" => Keyword::Union,
            "unsigned" => Keyword::Unsigned,
            "void" => Keyword::Void,
            "volatile" => Keyword::Volatile,
            "while" => Keyword::While,
            "_Bool" => Keyword::_Bool,
            "_Complex" => Keyword::_Complex,
            "_Imaginary" => Keyword::_Imaginary,
            "_Atomic" => Keyword::_Atomic,
            "_Noreturn" => Keyword::_Noreturn,
            _ => return Err(()),
        })
    }
}
