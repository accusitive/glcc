//! Argument substitution into macro bodies:
//! parameters take prescanned arguments, except beside `#` (stringify)
//! and `##` (paste), which use raw arguments. `__VA_ARGS__` stands for
//! the extras; GNU `, ## __VA_ARGS__` drops a trailing comma with an
//! empty pack.

use std::collections::HashMap;

use glcc_frontend::lexer::{Interner, Lexer, PpToken, PpWs, Punctuator};

use crate::expand::{Tok, tok_is_ws};

pub type ArgMap = HashMap<String, MacroArgument>;

/// One call argument: raw tokens plus the prescanned (expanded) form.
#[derive(Debug, Clone)]
pub struct MacroArgument {
    pub unexpanded: Vec<Tok>,
    pub expanded: Vec<Tok>,
}

pub struct ArgumentSubstitution<'a> {
    pub replacements: Vec<Tok>,
    pub pos: usize,
    pub map: ArgMap,
    /// The variadic pack split per call piece (used for `__VA_ARGS__`
    /// joining and raw operand text).
    pub extras: Vec<MacroArgument>,
    pub is_variadic: bool,
    /// GNU `name...`: the pack goes through `map` under this name instead
    /// of the implicit `__VA_ARGS__` (clang `MacroInfo::isGNUVarargs`).
    pub named_variadic: Option<String>,
    /// The invocation elided the variadic argument entirely (clang
    /// `MacroArgs::isVarargsElidedUse`), as opposed to passing it empty.
    pub variadic_elided: bool,
    /// Function-like replacement lists process `#`/`##`; in object-like
    /// ones they are ordinary tokens.
    pub function_like: bool,
    pub interner: &'a mut Interner,
    /// Tokens accumulated for the paste in progress; a `##` was consumed
    /// and the right operand is still awaited (`glue_open`). Operands are
    /// whole token runs, so multi-token arguments keep their own flags.
    pub glue_chain: Vec<Tok>,
    pub glue_open: bool,
    /// Strict C99/C11 mode and the clang parameter count (`...` counts as
    /// one): the `, ## __VA_ARGS__` comma removal is disabled in strict
    /// C99 for macros with fewer than two parameters.
    pub strict_c99: bool,
    pub num_params: usize,
    /// clang `NextTokGetsSpace`: set by a body token with leading space
    /// (unless it follows `##`), then consumed by the next emitted token.
    /// An empty argument substitution leaves it pending.
    pub next_space: bool,
    pub prev_hashhash: bool,
    pub seen_body_token: bool,
    pub stringify_next: bool,
    /// Raw tokens collected for the `#` operand (clang stringifies the
    /// token list, not the text).
    pub stringify_toks: Vec<Tok>,
}

impl<'a> ArgumentSubstitution<'a> {
    /// Substitute the whole replacement list, returning output tokens
    /// plus any still-pending `NextTokGetsSpace` from the end of the list
    /// (clang hands that to the macro's eof token, which propagates it to
    /// the next token of the stream).
    pub fn run(mut self) -> Result<(Vec<Tok>, bool), crate::PpError> {
        let mut out = Vec::new();
        while let Some(chunk) = self.next_chunk()? {
            out.extend(chunk);
        }
        if self.glue_open {
            return Err(crate::err("macro replacement ends with '##'".to_string()));
        }
        if self.stringify_next {
            return Err(crate::err("macro replacement ends with '#'".to_string()));
        }
        Ok((out, self.next_space))
    }

    fn peek_non_ws(&self) -> Option<&Tok> {
        let mut i = self.pos;
        while i < self.replacements.len() {
            if !tok_is_ws(&self.replacements[i]) {
                return Some(&self.replacements[i]);
            }
            i += 1;
        }
        None
    }

    fn eat_hashhash(&mut self) {
        while self.pos < self.replacements.len() && tok_is_ws(&self.replacements[self.pos]) {
            self.pos += 1;
        }
        if matches!(self.replacements.get(self.pos).map(|t| &t.tok), Some(PpToken::Punct(Punctuator::HashHash))) {
            self.pos += 1;
            // The `##` was consumed: the operand after it must not set
            // `NextTokGetsSpace` from its own leading space
            // (`TokenLexer.cpp` line 271).
            self.prev_hashhash = true;
        }
    }

    fn next_chunk(&mut self) -> Result<Option<Vec<Tok>>, crate::PpError> {
        if self.pos >= self.replacements.len() {
            return Ok(None);
        }
        let tok = self.replacements[self.pos].clone();
        // A `#` operand: clang's expansion loop skips the parameter token
        // itself, so its leading space never reaches `NextTokGetsSpace`;
        // the stringified token keeps the space the `#` collected.
        if self.stringify_next && !tok_is_ws(&tok) {
            self.pos += 1;
            let name = match &tok.tok {
                PpToken::Ident(sym) => self.interner.resolve(*sym).unwrap_or("").to_string(),
                _ => String::new(),
            };
            let is_param = self.map.contains_key(&name) || self.is_va_param(&tok);
            if !is_param {
                return Err(crate::err(format!("'#' must be followed by a macro parameter, not '{name}'")));
            }
            if self.is_va_param(&tok) {
                let raw = self.va_raw_tokens(&tok);
                self.stringify_toks.extend(raw);
            } else if let Some(arg) = self.map.get(&name) {
                self.stringify_toks.extend(arg.unexpanded.iter().cloned());
            }
            let lit = stringify_tokens(&self.stringify_toks, self.interner);
            self.stringify_toks.clear();
            self.stringify_next = false;
            let space = std::mem::take(&mut self.next_space);
            return Ok(Some(vec![Tok {
                tok: PpToken::Str(self.interner.get_or_intern(lit)),
                line: tok.line,
                file: tok.file,
                expanded: true,
                lined: false,
                space,
                sol: false,
                col: tok.col,
                end_line: tok.end_line,
                spell: None,
                noexpand: false,
            }]));
        }
        // clang `ExpandFunctionArguments`: a body token's leading space
        // makes the next emitted token get a space (`NextTokGetsSpace`).
        if !tok_is_ws(&tok) {
            if self.seen_body_token && !self.prev_hashhash && tok.space {
                self.next_space = true;
            }
            self.seen_body_token = true;
            self.prev_hashhash = matches!(tok.tok, PpToken::Punct(Punctuator::HashHash));
        }
        // Whitespace adjacent to `##` vanishes: pasting ignores it.
        if tok_is_ws(&tok) {
            // Whitespace between `#` and its parameter is dropped too (its
            // own space never reaches `NextTokGetsSpace`).
            if self.stringify_next {
                self.pos += 1;
                return Ok(Some(Vec::new()));
            }
            let mut i = self.pos + 1;
            while i < self.replacements.len() && tok_is_ws(&self.replacements[i]) {
                i += 1;
            }
            if matches!(self.replacements.get(i).map(|t| &t.tok), Some(PpToken::Punct(Punctuator::HashHash))) {
                self.pos += 1;
                return Ok(Some(Vec::new()));
            }
        }
        self.pos += 1;
        // Right operand of an open paste chain.
        if self.glue_open {
            // Whitespace between `##` and the operand is insignificant.
            if tok_is_ws(&tok) {
                return Ok(Some(Vec::new()));
            }
            if matches!(tok.tok, PpToken::Punct(Punctuator::HashHash)) {
                return Ok(Some(Vec::new()));
            }
            // GNU `, ## __VA_ARGS__` with a non-empty pack is not a real
            // paste: the `##` disappears, the comma stays, and the pack
            // tokens follow with their own flags (clang `VaArgsPseudoPaste`).
            let pseudo = self.is_va_param(&tok) && matches!(self.glue_chain.last().map(|t| &t.tok), Some(PpToken::Punct(Punctuator::Comma)));
            if pseudo && !self.va_empty() {
                let out = self.va_raw_tokens(&tok);
                self.next_space = false;
                return self.finish_chain(out);
            }
            let operand = self.operand_tokens(&tok);
            if !operand.is_empty() {
                let mut rest = operand.into_iter();
                let rhs = rest.next().unwrap();
                match self.glue_chain.pop() {
                    Some(lhs) => {
                        let merged = self.paste_pair(&lhs, &rhs);
                        self.glue_chain.extend(merged);
                    }
                    None => {
                        // An empty left operand skipped the `##`; the
                        // right one is emitted as-is.
                        let mut t = rhs;
                        t.space = std::mem::take(&mut self.next_space);
                        t.sol = false;
                        self.glue_chain.push(t);
                    }
                }
                self.glue_chain.extend(rest);
                if matches!(self.peek_non_ws().map(|t| &t.tok), Some(PpToken::Punct(Punctuator::HashHash))) {
                    self.eat_hashhash();
                    return Ok(Some(Vec::new()));
                }
                return self.finish_chain(Vec::new());
            }
            // Placemarker: the `##` vanishes. An elided variadic pack also
            // removes the comma before it, unless strict C99 leaves single
            // named parameters alone (clang `MaybeRemoveCommaBeforeVaArgs`).
            if pseudo && self.variadic_elided && !(self.strict_c99 && self.num_params < 2) {
                if matches!(self.glue_chain.last().map(|t| &t.tok), Some(PpToken::Punct(Punctuator::Comma))) {
                    self.glue_chain.pop();
                }
                self.next_space = false;
            }
            if matches!(self.peek_non_ws().map(|t| &t.tok), Some(PpToken::Punct(Punctuator::HashHash))) {
                self.eat_hashhash();
                return Ok(Some(Vec::new()));
            }
            return self.finish_chain(Vec::new());
        }
        // Continuation of a `#` stringify is handled above, before the
        // body-token bookkeeping.
        match &tok.tok {
            PpToken::Punct(Punctuator::Hash) if self.function_like => {
                self.stringify_next = true;
                Ok(Some(Vec::new()))
            }
            PpToken::Punct(Punctuator::HashHash) => Err(crate::err("'##' cannot appear at the start of a macro replacement")),
            _ if matches!(self.peek_non_ws().map(|t| &t.tok), Some(PpToken::Punct(Punctuator::HashHash))) => {
                // `X ##`: start a paste with X's raw tokens as the left
                // operand (`##` is consumed; its own spacing never lands).
                let mut lhs = self.operand_tokens(&tok);
                self.apply_subst_flags(&mut lhs);
                self.glue_chain = lhs;
                self.glue_open = true;
                self.eat_hashhash();
                Ok(Some(Vec::new()))
            }
            PpToken::Ident(sym) => {
                let name = self.interner.resolve(*sym).unwrap_or("").to_string();
                if self.is_variadic && name == "__VA_OPT__" {
                    return Err(crate::err("'__VA_OPT__' is not supported by native pp yet"));
                }
                if self.is_variadic && self.named_variadic.is_none() && name == "__VA_ARGS__" {
                    let mut joined: Vec<Tok> = Vec::new();
                    for (i, extra) in self.extras.iter().enumerate() {
                        if i > 0 {
                            joined.push(pack_comma(&tok));
                        }
                        joined.extend(extra.expanded.clone());
                    }
                    self.apply_subst_flags(&mut joined);
                    Ok(Some(joined))
                } else if let Some(arg) = self.map.get(&name) {
                    let mut toks = arg.expanded.clone();
                    self.apply_subst_flags(&mut toks);
                    Ok(Some(toks))
                } else {
                    Ok(Some(vec![self.plain(tok)]))
                }
            }
            _ => Ok(Some(vec![self.plain(tok)])),
        }
    }

    /// clang `ExpandFunctionArguments`: the first token substituted from
    /// an argument takes the pending `NextTokGetsSpace` and never
    /// `StartOfLine`; empty arguments leave the flag pending.
    fn apply_subst_flags(&mut self, toks: &mut [Tok]) {
        if let Some(first) = toks.iter_mut().find(|t| !tok_is_ws(t)) {
            first.space = std::mem::take(&mut self.next_space);
            first.sol = false;
        }
    }

    /// A plain replacement-list token: consumes the pending space flag.
    fn plain(&mut self, mut tok: Tok) -> Tok {
        if std::mem::take(&mut self.next_space) {
            tok.space = true;
        }
        tok
    }

    /// Is this identifier the variadic parameter (`__VA_ARGS__` or the
    /// GNU named spelling)?
    fn is_va_param(&self, tok: &Tok) -> bool {
        let PpToken::Ident(sym) = tok.tok else {
            return false;
        };
        let name = self.interner.resolve(sym).unwrap_or("");
        match &self.named_variadic {
            Some(n) => name == n,
            None => self.is_variadic && name == "__VA_ARGS__",
        }
    }

    /// Whether the variadic pack is empty (clang `getArgLength == 0`).
    fn va_empty(&self) -> bool {
        self.extras.iter().all(|a| a.unexpanded.is_empty())
    }

    /// The variadic pack's raw tokens, for paste operands.
    fn va_raw_tokens(&self, site: &Tok) -> Vec<Tok> {
        if let Some(name) = &self.named_variadic {
            return self.map.get(name).map(|a| a.unexpanded.clone()).unwrap_or_default();
        }
        let mut out = Vec::new();
        for (i, extra) in self.extras.iter().enumerate() {
            if i > 0 {
                out.push(pack_comma(site));
            }
            out.extend(extra.unexpanded.iter().cloned());
        }
        out
    }

    /// The raw token run for one paste operand: a parameter's unexpanded
    /// argument, the joined variadic pack, or the token itself.
    fn operand_tokens(&self, tok: &Tok) -> Vec<Tok> {
        if let PpToken::Ident(sym) = &tok.tok {
            let name = self.interner.resolve(*sym).unwrap_or("");
            if let Some(arg) = self.map.get(name) {
                return arg.unexpanded.clone();
            }
            if self.is_va_param(tok) {
                return self.va_raw_tokens(tok);
            }
        }
        vec![tok.clone()]
    }

    /// One `##` step: concatenate the two spellings and re-lex, the
    /// result inheriting the left token's flags (clang `pasteTokens`).
    fn paste_pair(&mut self, lhs: &Tok, rhs: &Tok) -> Vec<Tok> {
        let text = format!("{}{}", render_tok(lhs, self.interner), render_tok(rhs, self.interner));
        self.relex_glued(&text, lhs)
    }

    /// Emit a finished paste chain: the accumulated tokens plus `extra`.
    /// An all-placemarker chain leaves the pending space for the next
    /// token; real output resolves it.
    fn finish_chain(&mut self, extra: Vec<Tok>) -> Result<Option<Vec<Tok>>, crate::PpError> {
        self.glue_open = false;
        let mut out = std::mem::take(&mut self.glue_chain);
        out.extend(extra);
        if !out.is_empty() {
            self.next_space = false;
        }
        Ok(Some(out))
    }

    /// Re-lex pasted text into tokens (the spelling decides the token
    /// kinds).
    fn relex_glued(&mut self, text: &str, site: &Tok) -> Vec<Tok> {
        if text.trim().is_empty() {
            return Vec::new();
        }
        let (line, file, col) = (site.line, site.file, site.col);
        let interner = &mut *self.interner;
        let mut lexer = Lexer::new(text, interner);
        let mut out: Vec<Tok> = lexer
            .lex_pp_all()
            .into_iter()
            .filter(|t| !matches!(t.tok, PpToken::Ws(_)))
            .map(|t| Tok {
                tok: t.tok,
                line,
                file,
                expanded: true,
                lined: false,
                space: t.space,
                sol: false,
                col,
                end_line: line,
                spell: None,
                noexpand: false,
            })
            .collect();
        // clang keeps the LHS token's flags on the paste result.
        if let Some(first) = out.first_mut() {
            first.space = site.space;
            first.sol = site.sol;
        }
        out
    }
}

/// A synthesized comma joining variadic pack pieces.
fn pack_comma(site: &Tok) -> Tok {
    Tok {
        tok: PpToken::Punct(Punctuator::Comma),
        line: site.line,
        file: site.file,
        expanded: true,
        lined: false,
        space: false,
        sol: false,
        col: site.col,
        end_line: site.end_line,
        spell: None,
        noexpand: false,
    }
}

/// Render tokens to their raw concatenated spellings.
pub fn render_raw(toks: &[Tok], interner: &Interner) -> String {
    let mut s = String::new();
    for t in toks {
        s.push_str(&render_tok(t, interner));
    }
    s
}

pub fn render_tok(t: &Tok, interner: &Interner) -> String {
    match t.tok {
        PpToken::Ident(sym) | PpToken::PpNumber(sym) | PpToken::Str(sym) | PpToken::Char(sym) => interner.resolve(sym).unwrap_or("").to_string(),
        PpToken::Punct(p) => p.as_str().to_string(),
        PpToken::Ws(PpWs::Space) => " ".to_string(),
        PpToken::Ws(PpWs::Tab) => "\t".to_string(),
        PpToken::Ws(PpWs::Newline) => "\n".to_string(),
        PpToken::Directive => "#".to_string(),
        PpToken::Other(c) => c.to_string(),
    }
}

/// `#param` rendering, clang `MacroArgs::StringifyArgument`: place one
/// space where a token was separated by whitespace (or started a line),
/// escape string/char literals with `Lexer::Stringify` (backslashes and
/// double quotes gain a backslash), and keep everything else verbatim.
fn stringify_tokens(toks: &[Tok], interner: &Interner) -> String {
    let mut s = String::from("\"");
    let mut first = true;
    for t in toks {
        // Whitespace is not a token in clang's argument list; separation
        // lives in the following token's flags.
        if matches!(t.tok, PpToken::Ws(_)) {
            continue;
        }
        if !first && (t.space || t.sol) {
            s.push(' ');
        }
        first = false;
        let spell = render_tok(t, interner);
        if matches!(t.tok, PpToken::Str(_) | PpToken::Char(_)) {
            escape_literal(&spell, &mut s);
        } else {
            s.push_str(&spell);
        }
    }
    s.push('"');
    s
}

/// clang `StringifyImpl`: escape `\` and `"`; rewrite newlines as `\n`.
fn escape_literal(spell: &str, s: &mut String) {
    let chars: Vec<char> = spell.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\\' || c == '"' {
            s.push('\\');
            s.push(c);
        } else if c == '\n' || c == '\r' {
            s.push_str("\\n");
            if i + 1 < chars.len() && (chars[i + 1] == '\n' || chars[i + 1] == '\r') && chars[i + 1] != c {
                i += 1;
            }
        } else {
            s.push(c);
        }
        i += 1;
    }
}
