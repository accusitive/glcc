//! Directive grouping: parse `#`-lines into
//! [`ControlLine`]s and nest `#if`/`#ifdef`/`#ifndef` bodies with their
//! `#elif`/`#else`/`#endif` oppositions. Malformed groups are precise
//! [`PpError`]s, never panics.

use glcc_frontend::lexer::{Interner, PpToken, PpWs, Punctuator};

use crate::expand::Tok;
use crate::{PpError, err};

#[derive(Debug, Clone)]
pub enum ControlLine {
    IfDef {
        macro_name: String,
    },
    IfNDef {
        macro_name: String,
    },
    If {
        condition: Vec<Tok>,
    },
    Elif {
        condition: Vec<Tok>,
    },
    ElifDef {
        macro_name: String,
    },
    ElifNDef {
        macro_name: String,
    },
    Else,
    EndIf,
    DefineFunction {
        macro_name: String,
        parameters: Vec<Tok>,
        replacement: Vec<Tok>,
    },
    DefineObject {
        macro_name: String,
        replacement: Vec<Tok>,
    },
    Undefine(String),
    Include {
        /// Raw header-name text when already in `"..."`/`<...>` form, or
        /// the raw token sequence needing macro expansion first.
        header: Vec<Tok>,
        verbatim: bool,
        next: bool,
    },
    Error(Vec<Tok>),
    Warning(Vec<Tok>),
    /// `#embed`: the raw header/parameter token sequence.
    Embed(Vec<Tok>),
    PragmaOnce,
    /// Any other `#pragma`: the tokens after the directive keyword
    /// (whitespace dropped, `space` flags kept) and the directive line.
    PragmaOther {
        tokens: Vec<Tok>,
        line: u64,
    },
    Line(Vec<Tok>),
    Empty,
}

#[derive(Debug, Clone)]
pub enum TokenTree {
    Directive(ControlLine),
    IfDef {
        macro_name: String,
        body: Vec<TokenTree>,
        opposition: Box<TokenTree>,
    },
    IfNDef {
        macro_name: String,
        body: Vec<TokenTree>,
        opposition: Box<TokenTree>,
    },
    If {
        condition: Vec<Tok>,
        body: Vec<TokenTree>,
        opposition: Box<TokenTree>,
    },
    Elif {
        condition: Vec<Tok>,
        body: Vec<TokenTree>,
        opposition: Box<TokenTree>,
    },
    ElifDef {
        macro_name: String,
        body: Vec<TokenTree>,
        opposition: Box<TokenTree>,
    },
    ElifNDef {
        macro_name: String,
        body: Vec<TokenTree>,
        opposition: Box<TokenTree>,
    },
    Else {
        body: Vec<TokenTree>,
    },
    Endif,
    Token(Tok),
}

/// Group one file's tokens into trees. Stops at (consuming nothing
/// past) end-of-input; unterminated groups are errors.
pub struct IntoTokenTree {
    pub tokens: Vec<Tok>,
    pub pos: usize,
    pub expecting_opposition: bool,
}

impl IntoTokenTree {
    pub fn new(tokens: Vec<Tok>) -> Self {
        Self {
            tokens,
            pos: 0,
            expecting_opposition: false,
        }
    }

    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Tok> {
        let t = self.tokens.get(self.pos).cloned()?;
        self.pos += 1;
        Some(t)
    }

    /// Collect a directive line: assumes the `Directive` token was just
    /// consumed; gathers tokens up to (not including) the newline.
    fn directive_rest(&mut self) -> Vec<Tok> {
        let mut out = Vec::new();
        while let Some(t) = self.peek() {
            if matches!(t.tok, PpToken::Ws(PpWs::Newline)) {
                break;
            }
            out.push(self.next().unwrap());
        }
        out
    }

    pub fn next_tree(&mut self, file: &str, interner: &Interner) -> Result<Option<TokenTree>, PpError> {
        let Some(tok) = self.next() else {
            return Ok(None);
        };
        if !matches!(tok.tok, PpToken::Directive) {
            return Ok(Some(TokenTree::Token(tok)));
        }
        let rest = self.directive_rest();
        let control = parse_control_line(&rest, file, tok.line, interner)?;
        match control {
            ControlLine::DefineObject { .. }
            | ControlLine::DefineFunction { .. }
            | ControlLine::Error(..)
            | ControlLine::Warning(..)
            | ControlLine::Embed(..)
            | ControlLine::Undefine(..)
            | ControlLine::Include { .. }
            | ControlLine::PragmaOnce
            | ControlLine::PragmaOther { .. }
            | ControlLine::Line(..) => Ok(Some(TokenTree::Directive(control))),
            ControlLine::If { condition } => {
                let body = self.until_closer(file, interner)?;
                let opposition = Box::new(self.maybe_opposition(file, interner)?);
                Ok(Some(TokenTree::If { condition, body, opposition }))
            }
            ControlLine::IfDef { macro_name } => {
                let body = self.until_closer(file, interner)?;
                let opposition = Box::new(self.maybe_opposition(file, interner)?);
                Ok(Some(TokenTree::IfDef { macro_name, body, opposition }))
            }
            ControlLine::IfNDef { macro_name } => {
                let body = self.until_closer(file, interner)?;
                let opposition = Box::new(self.maybe_opposition(file, interner)?);
                Ok(Some(TokenTree::IfNDef { macro_name, body, opposition }))
            }
            ControlLine::Elif { .. } | ControlLine::ElifDef { .. } | ControlLine::ElifNDef { .. } | ControlLine::Else if !self.expecting_opposition => {
                Err(err(format!("{file}:{}: #elif/#else without #if", tok.line)))
            }
            ControlLine::Elif { condition } => {
                let body = self.until_closer(file, interner)?;
                let opposition = Box::new(self.maybe_opposition(file, interner)?);
                Ok(Some(TokenTree::Elif { condition, body, opposition }))
            }
            ControlLine::ElifDef { macro_name } => {
                let body = self.until_closer(file, interner)?;
                let opposition = Box::new(self.maybe_opposition(file, interner)?);
                Ok(Some(TokenTree::ElifDef { macro_name, body, opposition }))
            }
            ControlLine::ElifNDef { macro_name } => {
                let body = self.until_closer(file, interner)?;
                let opposition = Box::new(self.maybe_opposition(file, interner)?);
                Ok(Some(TokenTree::ElifNDef { macro_name, body, opposition }))
            }
            ControlLine::Else => {
                let body = self.until_closer(file, interner)?;
                let _ = self.maybe_opposition(file, interner)?;
                Ok(Some(TokenTree::Else { body }))
            }
            ControlLine::Empty => self.next_tree(file, interner),
            ControlLine::EndIf => Err(err(format!("{file}:{}: #endif without #if", tok.line))),
        }
    }

    fn until_closer(&mut self, file: &str, interner: &Interner) -> Result<Vec<TokenTree>, PpError> {
        let mut body = Vec::new();
        loop {
            let Some(tok) = self.peek().cloned() else {
                return Err(err(format!("{file}: unterminated conditional")));
            };
            if matches!(tok.tok, PpToken::Directive) {
                // Peek at the directive kind without consuming.
                let save = self.pos;
                self.next();
                let rest = self.directive_rest();
                self.pos = save;
                match directive_kind(&rest, interner) {
                    Kind::EndIf | Kind::Elif | Kind::Else => return Ok(body),
                    Kind::Empty => {
                        self.next_tree(file, interner)?;
                        continue;
                    }
                    Kind::Other => {}
                }
            }
            body.push(self.next_tree(file, interner)?.ok_or_else(|| err(format!("{file}: unterminated conditional")))?);
        }
    }

    fn maybe_opposition(&mut self, file: &str, interner: &Interner) -> Result<TokenTree, PpError> {
        let Some(tok) = self.peek().cloned() else {
            return Err(err(format!("{file}: unterminated conditional")));
        };
        if !matches!(tok.tok, PpToken::Directive) {
            return Err(err(format!("{file}: unterminated conditional")));
        }
        self.next();
        let rest = self.directive_rest();
        let control = parse_control_line(&rest, file, tok.line, interner)?;
        match control {
            ControlLine::Elif { condition } => {
                self.expecting_opposition = true;
                let body = self.until_closer(file, interner)?;
                let opposition = Box::new(self.maybe_opposition(file, interner)?);
                self.expecting_opposition = false;
                Ok(TokenTree::Elif { condition, body, opposition })
            }
            ControlLine::ElifDef { macro_name } => {
                self.expecting_opposition = true;
                let body = self.until_closer(file, interner)?;
                let opposition = Box::new(self.maybe_opposition(file, interner)?);
                self.expecting_opposition = false;
                Ok(TokenTree::ElifDef { macro_name, body, opposition })
            }
            ControlLine::ElifNDef { macro_name } => {
                self.expecting_opposition = true;
                let body = self.until_closer(file, interner)?;
                let opposition = Box::new(self.maybe_opposition(file, interner)?);
                self.expecting_opposition = false;
                Ok(TokenTree::ElifNDef { macro_name, body, opposition })
            }
            ControlLine::Else => {
                self.expecting_opposition = true;
                let body = self.until_closer(file, interner)?;
                let _ = self.maybe_opposition(file, interner)?;
                self.expecting_opposition = false;
                Ok(TokenTree::Else { body })
            }
            ControlLine::Empty => self.maybe_opposition(file, interner),
            ControlLine::EndIf => Ok(TokenTree::Endif),
            _ => Err(err(format!("{file}:{}: expected #elif, #else or #endif", tok.line))),
        }
    }
}

enum Kind {
    EndIf,
    Elif,
    Else,
    Empty,
    Other,
}

/// Collect every tree of one file's token stream up front.
pub fn collect_trees(toks: Vec<Tok>, file: &str, interner: &Interner) -> Result<Vec<TokenTree>, PpError> {
    let mut grouper = IntoTokenTree::new(toks);
    let mut out = Vec::new();
    while let Some(t) = grouper.next_tree(file, interner)? {
        out.push(t);
    }
    Ok(out)
}

/// Classify a directive's first token for nesting decisions.
fn directive_kind(rest: &[Tok], interner: &Interner) -> Kind {
    let mut it = rest.iter().filter(|t| !matches!(t.tok, PpToken::Ws(PpWs::Space | PpWs::Tab)));
    let Some(first) = it.next() else {
        return Kind::Empty;
    };
    let PpToken::Ident(sym) = first.tok else {
        return Kind::Other;
    };
    match interner.resolve(sym).unwrap_or("") {
        "endif" => Kind::EndIf,
        "elif" | "elifdef" | "elifndef" => Kind::Elif,
        "else" => Kind::Else,
        _ => Kind::Other,
    }
}

fn is_ws(t: &Tok) -> bool {
    matches!(t.tok, PpToken::Ws(PpWs::Space | PpWs::Tab))
}

fn ident_name<'a>(t: &Tok, interner: &'a Interner) -> Option<&'a str> {
    match t.tok {
        PpToken::Ident(sym) => Some(interner.resolve(sym).unwrap_or("")),
        _ => None,
    }
}

/// Parse directive tokens (after the `#`) into a [`ControlLine`].
pub fn parse_control_line(rest: &[Tok], file: &str, line: u64, interner: &Interner) -> Result<ControlLine, PpError> {
    let loc = format!("{file}:{line}");
    let toks: Vec<&Tok> = rest.iter().filter(|t| !is_ws(t)).collect();
    let Some(first) = toks.first() else {
        return Ok(ControlLine::Empty);
    };
    // A leading number is a GNU linemarker (`# 12 "file" flags`, also
    // the bare `# 12` form); feeding the same operands through the
    // `#line` path reuses its line/file remapping.
    if matches!(first.tok, PpToken::PpNumber(_)) {
        return Ok(ControlLine::Line(rest.to_vec()));
    }
    let Some(name) = ident_name(first, interner) else {
        return Err(err(format!("{loc}: malformed directive (expected a name)")));
    };
    // Raw tail after the directive name (for faithful bodies): skip the
    // name token itself in `rest`.
    let name_pos = rest.iter().position(|t| matches!(t.tok, PpToken::Ident(_))).unwrap_or(0);
    let tail = &rest[name_pos + 1..];
    let tail_args: Vec<&Tok> = tail.iter().filter(|t| !is_ws(t)).collect();
    match name {
        "ifdef" => {
            let [m] = tail_args.as_slice() else {
                return Err(err(format!("{loc}: malformed #ifdef (expected one macro name)")));
            };
            let Some(n) = ident_name(m, interner) else {
                return Err(err(format!("{loc}: malformed #ifdef (expected one macro name)")));
            };
            Ok(ControlLine::IfDef { macro_name: n.to_string() })
        }
        "ifndef" => {
            let [m] = tail_args.as_slice() else {
                return Err(err(format!("{loc}: malformed #ifndef (expected one macro name)")));
            };
            let Some(n) = ident_name(m, interner) else {
                return Err(err(format!("{loc}: malformed #ifndef (expected one macro name)")));
            };
            Ok(ControlLine::IfNDef { macro_name: n.to_string() })
        }
        "if" => Ok(ControlLine::If { condition: tail.to_vec() }),
        "elif" => Ok(ControlLine::Elif { condition: tail.to_vec() }),
        "elifdef" | "elifndef" => {
            let [m] = tail_args.as_slice() else {
                return Err(err(format!("{loc}: malformed #{name} (expected one macro name)")));
            };
            let Some(n) = ident_name(m, interner) else {
                return Err(err(format!("{loc}: malformed #{name} (expected one macro name)")));
            };
            if name == "elifdef" {
                Ok(ControlLine::ElifDef { macro_name: n.to_string() })
            } else {
                Ok(ControlLine::ElifNDef { macro_name: n.to_string() })
            }
        }
        "else" => {
            if !tail_args.is_empty() {
                return Err(err(format!("{loc}: junk after #else")));
            }
            Ok(ControlLine::Else)
        }
        "endif" => {
            if !tail_args.is_empty() {
                return Err(err(format!("{loc}: junk after #endif")));
            }
            Ok(ControlLine::EndIf)
        }
        "define" => parse_define(&tail_args, tail, &loc, interner),
        "undef" => {
            let [m] = tail_args.as_slice() else {
                return Err(err(format!("{loc}: malformed #undef")));
            };
            let Some(n) = ident_name(m, interner) else {
                return Err(err(format!("{loc}: malformed #undef")));
            };
            Ok(ControlLine::Undefine(n.to_string()))
        }
        "error" => Ok(ControlLine::Error(tail.to_vec())),
        "warning" => Ok(ControlLine::Warning(tail.to_vec())),
        "embed" => Ok(ControlLine::Embed(tail.to_vec())),
        "pragma" => {
            if tail_args.len() == 1 {
                if let Some(n) = ident_name(tail_args[0], interner) {
                    if n == "once" {
                        return Ok(ControlLine::PragmaOnce);
                    }
                }
            }
            Ok(ControlLine::PragmaOther {
                tokens: tail_args.iter().map(|t| (*t).clone()).collect(),
                line,
            })
        }
        "line" => Ok(ControlLine::Line(tail.to_vec())),
        "include" | "include_next" => parse_include(tail, &loc, interner, name == "include_next"),
        _ => Err(err(format!("{loc}: unknown directive '#{name}'"))),
    }
}

/// `#define NAME ...` / `#define NAME(params) ...`: adjacency of `(` to
/// the name decides function-like. `args` are the filtered tokens after
/// the name; `tail` is the raw slice (for adjacency). Leading and
/// trailing whitespace of bodies is insignificant (pp-tokens).
fn parse_define(args: &[&Tok], tail: &[Tok], loc: &str, interner: &Interner) -> Result<ControlLine, PpError> {
    let Some(first) = args.first() else {
        return Err(err(format!("{loc}: malformed #define")));
    };
    let Some(macro_name) = ident_name(first, interner) else {
        return Err(err(format!("{loc}: malformed #define")));
    };
    let macro_name = macro_name.to_string();
    // The macro name is the first identifier of `tail`; a `(` token
    // *immediately* after it (no whitespace token between) makes it
    // function-like.
    let name_idx = tail.iter().position(|t| matches!(t.tok, PpToken::Ident(_))).unwrap_or(usize::MAX);
    if name_idx != usize::MAX {
        if let Some(PpToken::Punct(Punctuator::LParen)) = tail.get(name_idx + 1).map(|t| &t.tok) {
            // Function-like: collect to the matching `)`.
            let mut depth = 0u32;
            let mut j = name_idx + 1;
            while j < tail.len() {
                match tail[j].tok {
                    PpToken::Punct(Punctuator::LParen) => depth += 1,
                    PpToken::Punct(Punctuator::RParen) => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            if j >= tail.len() {
                return Err(err(format!("{loc}: malformed #define (unclosed parameter list)")));
            }
            let parameters = tail[name_idx + 2..j].to_vec();
            let replacement = trim_ws_tokens(&tail[j + 1..]);
            return Ok(ControlLine::DefineFunction {
                macro_name,
                parameters,
                replacement,
            });
        }
    }
    Ok(ControlLine::DefineObject {
        macro_name,
        replacement: trim_ws_tokens(&tail[name_idx.saturating_add(1)..]),
    })
}

/// Strip leading/trailing `Space`/`Tab` tokens (newlines never occur
/// inside a directive line).
fn trim_ws_tokens(toks: &[Tok]) -> Vec<Tok> {
    let mut start = 0;
    let mut end = toks.len();
    while start < end && is_ws(&toks[start]) {
        start += 1;
    }
    while end > start && is_ws(&toks[end - 1]) {
        end -= 1;
    }
    toks[start..end].to_vec()
}

/// `#include` / `#include_next` over the raw tail: already a header
/// name in quotes/brackets (used verbatim, never expanded), or an
/// arbitrary token sequence needing macro expansion first.
fn parse_include(tail: &[Tok], loc: &str, interner: &Interner, next: bool) -> Result<ControlLine, PpError> {
    // Skip leading whitespace for the form check.
    let mut k = 0;
    while k < tail.len() && is_ws(&tail[k]) {
        k += 1;
    }
    let verbatim = match tail.get(k).map(|t| &t.tok) {
        Some(PpToken::Str(_)) | Some(PpToken::Punct(Punctuator::Less)) => true,
        _ => false,
    };
    if verbatim {
        // Parse `"..."` / `<...>` directly (C17 6.4.7: never expanded).
        let header = tail[k..].to_vec();
        return Ok(ControlLine::Include { header, verbatim: true, next });
    }
    let _ = (interner, loc);
    Ok(ControlLine::Include {
        header: tail.to_vec(),
        verbatim: false,
        next,
    })
}
