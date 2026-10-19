//! Token-stream macro expansion: object- and
//! function-like calls with prescan, `#`/`##` substitution,
//! blue-paint recursion guard, `defined` handling in conditions, and
//! `#if` evaluation through `glcc_frontend` (`parse` + `check` +
//! `eval_const`).
//!
//! Deviations from a by-the-book expansion, all deliberate: diagnostics are [`PpError`]
//! (never panics/`todo!()`); `#if` folds via glcc's own consteval (no
//! duplicated evaluator); `#include` recurses through the driver's file
//! layer (markers, `#pragma once`, search paths); clang `__has_*`
//! builtins and `__LINE__`/`__FILE__` expand here; paint is a threaded
//! name set, so no `Begin/FinishExpandingMacro` queue markers and no
//! `BlueIdentifier` kind are needed.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use glcc_frontend::lexer::{Interner, Lexer, PpToken, PpWs, Punctuator};
use string_interner::symbol::SymbolU32;

use crate::macros::parse_parameters;
use crate::prepend::PrependingPeekableIterator;
use crate::substitute::{ArgMap, ArgumentSubstitution, MacroArgument, render_raw};
use crate::tree::{ControlLine, TokenTree};
use crate::{PpError, err};

/// One working token: a preprocessing token plus the source line
/// (restamped to the use site through expansions), the definition-site
/// file display (never restamped, for `__FILE__`), whether it came from
/// a macro expansion (rendering drops expansion newlines and pads like
/// `clang -E` at expansion edges), and whether `line` is already
/// effective (relocated tokens skip normalization, so `#line` bases
/// never apply twice).
#[derive(Debug, Clone)]
pub struct Tok {
    pub tok: PpToken,
    pub line: u64,
    pub file: SymbolU32,
    pub expanded: bool,
    pub lined: bool,
    /// clang `Token::LeadingSpace`: whitespace or a comment precedes.
    pub space: bool,
    /// clang `Token::StartOfLine`: nothing but whitespace/comments since
    /// the last newline. Expansion output carries the macro name's flag.
    pub sol: bool,
    /// 1-based column of the token's expansion site.
    pub col: u64,
    /// Line where the token's expansion range ends (clang
    /// `SourceManager::getExpansionRange`): the closing `)` of a
    /// function-like invocation, else the token's own line. `__LINE__`
    /// reads it.
    pub end_line: u64,
    /// Physical spelling `(region, byte offset, byte length)` for clang's
    /// `AvoidConcat` adjacency shortcut; `None` when synthesized.
    pub spell: Option<(SymbolU32, u64, u64)>,
    /// clang `Token::DisableExpand`: already-blue-painted output token.
    pub noexpand: bool,
}

impl Tok {
    /// A source token: adopts the lexer's lexical flags and spelling span.
    pub fn from_pp(t: &glcc_frontend::lexer::PpTok, file: SymbolU32, expanded: bool, lined: bool) -> Self {
        Tok {
            tok: t.tok,
            line: t.line,
            file,
            expanded,
            lined,
            space: t.space,
            sol: t.sol,
            col: t.col + 1,
            end_line: t.line,
            spell: Some((file, t.off, t.len)),
            noexpand: false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum MacroDef {
    Object(Vec<Tok>),
    Func {
        params: Vec<String>,
        variadic: bool,
        named_variadic: bool,
        body: Vec<Tok>,
    },
}

pub fn tok_is_ws(t: &Tok) -> bool {
    matches!(t.tok, PpToken::Ws(_))
}

/// C17 keywords: everything else identifier-like in an `#if` is an
/// undefined macro meaning `0`.
fn is_keyword(word: &str) -> bool {
    matches!(
        word,
        "auto"
            | "break"
            | "case"
            | "char"
            | "const"
            | "continue"
            | "default"
            | "do"
            | "double"
            | "else"
            | "enum"
            | "extern"
            | "float"
            | "for"
            | "goto"
            | "if"
            | "inline"
            | "int"
            | "long"
            | "register"
            | "restrict"
            | "return"
            | "short"
            | "signed"
            | "sizeof"
            | "static"
            | "struct"
            | "switch"
            | "typedef"
            | "union"
            | "unsigned"
            | "void"
            | "volatile"
            | "while"
            | "_Alignas"
            | "_Alignof"
            | "_Atomic"
            | "_Bool"
            | "_Complex"
            | "_Generic"
            | "_Imaginary"
            | "_Noreturn"
            | "_Static_assert"
            | "_Thread_local"
    )
}

fn is_has_builtin_name(name: &str) -> bool {
    matches!(
        name,
        "__has_include"
            | "__has_include_next"
            | "__has_feature"
            | "__has_extension"
            | "__has_attribute"
            | "__has_builtin"
            | "__has_warning"
            | "__has_cpp_attribute"
            | "__has_c_attribute"
            | "__has_declspec_attribute"
            | "__building_module"
    )
}

fn is_location_name(name: &str) -> bool {
    matches!(
        name,
        "__LINE__" | "__FILE__" | "__DATE__" | "__TIME__" | "__TIMESTAMP__" | "__COUNTER__" | "__BASE_FILE__" | "__INCLUDE_LEVEL__"
    )
}

/// clang `TokenConcatenation` info bits.
const ACI_CUSTOM: u32 = 1;
const ACI_CUSTOM_FIRSTCHAR: u32 = 2;
const ACI_AVOID_EQUAL: u32 = 4;

/// The slicing of clang's token kinds that `AvoidConcat` cares about (C
/// mode: literals and C++-only punctuators carry no info).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConcatKind {
    Ident,
    Number,
    Period,
    Amp,
    Plus,
    Minus,
    Slash,
    Less,
    Greater,
    Pipe,
    Percent,
    Colon,
    Hash,
    Arrow,
    Star,
    Exclaim,
    Caret,
    Equal,
    LessLess,
    GreaterGreater,
    Other,
}

fn concat_kind(t: &Tok) -> ConcatKind {
    match t.tok {
        PpToken::Ident(_) => ConcatKind::Ident,
        PpToken::PpNumber(_) => ConcatKind::Number,
        PpToken::Directive => ConcatKind::Hash,
        PpToken::Punct(p) => match p {
            Punctuator::Dot => ConcatKind::Period,
            Punctuator::Amp => ConcatKind::Amp,
            Punctuator::Plus => ConcatKind::Plus,
            Punctuator::Minus => ConcatKind::Minus,
            Punctuator::Slash => ConcatKind::Slash,
            Punctuator::Less => ConcatKind::Less,
            Punctuator::Greater => ConcatKind::Greater,
            Punctuator::Pipe => ConcatKind::Pipe,
            Punctuator::Percent => ConcatKind::Percent,
            Punctuator::Colon => ConcatKind::Colon,
            Punctuator::Hash => ConcatKind::Hash,
            Punctuator::Arrow => ConcatKind::Arrow,
            Punctuator::Star => ConcatKind::Star,
            Punctuator::Bang => ConcatKind::Exclaim,
            Punctuator::Caret => ConcatKind::Caret,
            Punctuator::Assign => ConcatKind::Equal,
            Punctuator::LessLess => ConcatKind::LessLess,
            Punctuator::GreaterGreater => ConcatKind::GreaterGreater,
            _ => ConcatKind::Other,
        },
        _ => ConcatKind::Other,
    }
}

fn concat_info(kind: ConcatKind) -> u32 {
    match kind {
        ConcatKind::Ident => ACI_CUSTOM,
        ConcatKind::Number | ConcatKind::Period | ConcatKind::Colon | ConcatKind::Hash | ConcatKind::Arrow => ACI_CUSTOM_FIRSTCHAR,
        ConcatKind::Amp | ConcatKind::Plus | ConcatKind::Minus | ConcatKind::Slash | ConcatKind::Less | ConcatKind::Greater | ConcatKind::Pipe | ConcatKind::Percent => {
            ACI_CUSTOM_FIRSTCHAR | ACI_AVOID_EQUAL
        }
        ConcatKind::Star | ConcatKind::Exclaim | ConcatKind::Caret | ConcatKind::Equal | ConcatKind::LessLess | ConcatKind::GreaterGreater => ACI_AVOID_EQUAL,
        ConcatKind::Other => 0,
    }
}

/// Human rendering of one token for diagnostics.
fn show_tok(t: &Tok, interner: &Interner) -> String {
    match t.tok {
        PpToken::Ident(sym) | PpToken::PpNumber(sym) | PpToken::Str(sym) | PpToken::Char(sym) => interner.resolve(sym).unwrap_or("?").to_string(),
        PpToken::Punct(p) => p.as_str().to_string(),
        PpToken::Ws(_) => " ".to_string(),
        PpToken::Directive => "#".to_string(),
        PpToken::Other(c) => c.to_string(),
    }
}

pub struct Expander<'a, 'b> {
    pub interner: &'a mut Interner,
    pub trees: PrependingPeekableIterator<std::vec::IntoIter<TokenTree>>,
    pub output: Vec<Tok>,
    /// Shared so argument prescans and `#if` conditions can borrow the
    /// table instead of deep-cloning it (thousands of definitions make
    /// cloning per macro argument quadratic); directives mutate through
    /// `Rc::make_mut`, which no live borrower can see.
    pub macros: Rc<HashMap<SymbolU32, MacroDef>>,
    pub expanding: HashSet<SymbolU32>,
    /// Inside an `#if`/`#elif` condition: `defined` and `__has_*` apply.
    pub respect_defined: bool,
    /// Current file: display symbol, filesystem path, and where it was
    /// found (for `#include_next`).
    pub file_display: SymbolU32,
    pub file_path: std::path::PathBuf,
    pub file_found: crate::FoundKind,
    /// Main input file's display, for `__BASE_FILE__` (includes share it).
    pub base_display: SymbolU32,
    /// `__COUNTER__`'s next value.
    pub counter: u32,
    /// Dynamic builtins the source `#undef`ed: they stop expanding and
    /// report undefined to `defined`/`#ifdef`, like clang.
    pub undef_builtins: HashSet<String>,
    /// Lazily computed local `__DATE__`/`__TIME__`.
    pub date_time: Option<(String, String)>,
    /// `__TIMESTAMP__` per current file.
    pub timestamps: HashMap<std::path::PathBuf, String>,
    /// `#pragma push_macro` stacks, innermost last.
    pub pushed_macros: HashMap<String, Vec<Option<MacroDef>>>,
    /// `#line` adjustment added to token lines (0 normally) plus an
    /// optional display rename.
    pub line_base: i64,
    pub display_override: Option<SymbolU32>,
    /// Target + `-std` for `#if` consteval.
    pub target: Option<glcc_frontend::target::Target>,
    pub std_flag: Option<String>,
    /// Recursion depth guard (pathological self-feeding macros).
    pub depth: u32,
    /// Driver file layer (search paths, markers, `#pragma once` set).
    pub files: &'b mut crate::Ctx,
    /// Output rendering state: clang's boundary pad + linemarker resync
    /// apply as tokens stream out (markers and content interleave).
    pub mark_n: i64,
    pub mark_emit: usize,
    pub fresh: bool,
    /// Previous emitted token: (from expansion, is pp-number, last char).
    /// Padding fires iff exactly one side of a boundary comes from macro
    /// expansion and the token shapes match (`11 +1` but `mem0.mutex`).
    pub prev: Option<(bool, bool, char)>,
    /// Render output tokens inline to the file layer (main runs) or
    /// buffer them (argument/condition sub-runs).
    pub emit_inline: bool,
    /// Active expansion regions: `(macro, queue length below the region)`.
    /// Pulling past a region re-enables its macro, as clang pops the
    /// macro's `TokenLexer` when it is exhausted.
    pub frames: Vec<(SymbolU32, usize)>,
    /// clang-faithful `-E` rendering state (used when `linemarkers`):
    /// current output line and whether tokens/directives were emitted on
    /// it, the last two printed tokens (for `AvoidConcat`), and a
    /// deferred start-of-line from a skipped `#pragma` eod.
    pub r_cur_line: u64,
    pub r_tokens_on_line: bool,
    pub r_directive_on_line: bool,
    pub r_pending_sol: bool,
    pub r_prev: Option<Tok>,
    pub r_prevprev: Option<Tok>,
    /// An empty expansion propagates its name's `StartOfLine`/`LeadingSpace`
    /// to the next token of the stream (clang
    /// `PropagateLineStartLeadingSpaceInfo`).
    pub carry: Option<(bool, bool)>,
}

impl<'a, 'b> Expander<'a, 'b> {
    pub fn display(&self) -> String {
        let sym = self.display_override.unwrap_or(self.file_display);
        self.interner.resolve(sym).unwrap_or("").to_string()
    }

    pub fn loc(&self, line: u64) -> String {
        format!("{}:{line}", self.display())
    }

    /// Effective (display) line: callers pass raw token lines exactly
    /// once (normalized at `expand_tok` entry), so this is used only
    /// for freshly computed numbers.
    pub fn eff_raw(&self, line: u64) -> u64 {
        (line as i64 + self.line_base).max(1) as u64
    }

    pub fn spelling(&self, sym: SymbolU32) -> String {
        self.interner.resolve(sym).unwrap_or("").to_string()
    }

    /// Strict C99-family mode (`-std=c99`/`c11`/... without GNU
    /// extensions), where `, ## __VA_ARGS__` keeps its comma in macros
    /// with fewer than two named parameters.
    fn strict_c99(&self) -> bool {
        let Some(name) = self.std_flag.as_deref().and_then(|s| s.strip_prefix("-std=")) else {
            return false;
        };
        name.starts_with('c') && !name.starts_with("c++") && !matches!(name, "c89" | "c90")
    }

    pub fn expand(&mut self) -> Result<(), PpError> {
        while let Some(tt) = self.trees.next() {
            self.expand_tree(tt)?;
        }
        Ok(())
    }

    /// Emit one output token with linemarker resyncs and clang's
    /// boundary pad. Newlines from macro expansions vanish (like
    /// `clang -E`); source newlines flush the line.
    ///
    /// Padding rule (probed from `clang -E` over 16 cases): at a boundary
    /// where exactly one side comes from macro expansion, pad iff the
    /// previous token is a pp-number and the next starts with `+`/`-`/`.`,
    /// or the adjoining characters are `(+, -, +, -)`. Identifiers never
    /// trigger it (`mem0.mutex`, `foo+1` stay tight).
    fn emit_out(&mut self, t: &Tok) -> Result<(), PpError> {
        if self.files.opts.linemarkers {
            self.render_clang(t);
            return Ok(());
        }
        if matches!(t.tok, PpToken::Ws(PpWs::Newline)) {
            if t.expanded {
                return Ok(());
            }
            self.files.out.push('\n');
            self.mark_emit += 1;
            self.fresh = true;
            self.prev = None;
            return Ok(());
        }
        if self.fresh {
            let orig = t.line as i64;
            if self.files.opts.linemarkers && orig != self.mark_n + self.mark_emit as i64 {
                let disp = self.interner.resolve(t.file).unwrap_or("").to_string();
                self.files.out.push_str(&crate::marker(orig, &disp));
                self.mark_n = orig;
                self.mark_emit = 0;
            }
            self.fresh = false;
        }
        let chunk = render_tok_text(t, self.interner);
        if chunk.is_empty() {
            return Ok(());
        }
        let first = chunk.chars().next().unwrap();
        if let Some((pe, pp, a)) = self.prev {
            let edge = t.expanded != pe;
            if edge && ((pp && (first == '+' || first == '-' || first == '.')) || ((a == '+' || a == '-') && (first == '+' || first == '-'))) {
                self.files.out.push(' ');
            }
        }
        self.files.out.push_str(&chunk);
        self.prev = Some((t.expanded, matches!(t.tok, PpToken::PpNumber(_)), chunk.chars().next_back().unwrap()));
        Ok(())
    }

    /// clang `-E` rendering (`PrintPreprocessedOutput.cpp`): move to each
    /// token's line, insert clang's `AvoidConcat` whitespace, and indent
    /// the first token of a line to its expansion column.
    fn render_clang(&mut self, t: &Tok) {
        if matches!(t.tok, PpToken::Ws(_)) {
            return;
        }
        let chunk = render_tok_text(t, self.interner);
        if chunk.is_empty() {
            return;
        }
        let sol = t.sol || self.r_pending_sol;
        self.handle_ws_before(t, false, !sol);
        self.r_pending_sol = false;
        self.files.out.push_str(&chunk);
        self.r_tokens_on_line = true;
        self.r_prevprev = self.r_prev.take();
        self.r_prev = Some(t.clone());
    }

    fn handle_ws_before(&mut self, t: &Tok, require_space: bool, require_same_line: bool) {
        // EmittedDirectiveOnThisLine takes priority over RequireSameLine.
        if (!require_same_line || self.r_directive_on_line) && self.move_to_line_tok(t, self.r_directive_on_line) {
            // Indent the first token on a line to its source column, with
            // clang's empty-macro-argument and `#`-at-column-1 quirks.
            let mut col = t.col;
            if col == 1 && t.space {
                col = 2;
            }
            if col <= 1 && matches!(t.tok, PpToken::Directive | PpToken::Punct(Punctuator::Hash)) {
                self.files.out.push(' ');
            }
            while col > 1 {
                self.files.out.push(' ');
                col -= 1;
            }
        } else if require_space || t.space || ((self.r_tokens_on_line || self.r_directive_on_line) && self.avoid_concat(self.r_prevprev.as_ref(), self.r_prev.as_ref(), t)) {
            self.files.out.push(' ');
        }
    }

    fn move_to_line_tok(&mut self, t: &Tok, require_start_of_line: bool) -> bool {
        // A token at the start of the file needs its column adjusted even
        // when no newline is written.
        let first_in_file = t.sol && t.line == 1;
        self.move_to_line(t.line, require_start_of_line) || first_in_file
    }

    /// clang `MoveToLine`: get to `target` with literal newlines when it is
    /// within reach, else with a fresh linemarker. Returns whether a new
    /// line (or marker) was started.
    fn move_to_line(&mut self, target: u64, require_start_of_line: bool) -> bool {
        let mut started = false;
        if (require_start_of_line && self.r_tokens_on_line) || self.r_directive_on_line {
            self.files.out.push('\n');
            started = true;
            self.r_cur_line += 1;
            self.r_tokens_on_line = false;
            self.r_directive_on_line = false;
        }
        if target == self.r_cur_line {
            // Already there.
        } else if !started && target == self.r_cur_line + 1 {
            self.files.out.push('\n');
            started = true;
        } else if target > self.r_cur_line && target - self.r_cur_line <= 8 {
            for _ in self.r_cur_line..target {
                self.files.out.push('\n');
            }
            started = true;
        } else {
            self.write_line_info(target);
            started = true;
        }
        if started {
            self.r_tokens_on_line = false;
            self.r_directive_on_line = false;
        }
        self.r_cur_line = target;
        started
    }

    /// clang `WriteLineInfo`: a marker naming the *current* display file.
    fn write_line_info(&mut self, line: u64) {
        self.start_new_line_if_needed();
        let disp = self.display();
        self.files.out.push_str(&crate::marker(line as i64, &disp));
    }

    /// clang `WriteLineInfo` with a file-change suffix (` 1` enter, ` 2`
    /// exit): used by the include enter/exit markers.
    fn write_line_info_flag(&mut self, line: u64, display: &str, flag: u8) {
        self.start_new_line_if_needed();
        self.files.out.push_str(&crate::marker_flag(line as i64, display, flag));
    }

    fn start_new_line_if_needed(&mut self) {
        if self.r_tokens_on_line || self.r_directive_on_line {
            self.files.out.push('\n');
            self.r_tokens_on_line = false;
            self.r_directive_on_line = false;
        }
    }

    /// clang `TokenConcatenation::AvoidConcat`: whether printing `cur`
    /// right after `prev` would lex as a single token.
    fn avoid_concat(&self, prevprev: Option<&Tok>, prev: Option<&Tok>, cur: &Tok) -> bool {
        let Some(prev) = prev else {
            return false;
        };
        // Directly adjacent in one buffer: nothing to keep apart.
        if let (Some((preg, poff, plen)), Some((creg, coff, _))) = (prev.spell, cur.spell) {
            if preg == creg && poff + plen == coff {
                return false;
            }
        }
        let kind = concat_kind(prev);
        let mut info = concat_info(kind);
        if info == 0 {
            return false;
        }
        if info & ACI_AVOID_EQUAL != 0 {
            if matches!(cur.tok, PpToken::Punct(Punctuator::Assign) | PpToken::Punct(Punctuator::EqEq)) {
                return true;
            }
            info &= !ACI_AVOID_EQUAL;
        }
        if info == 0 {
            return false;
        }
        let first = if info & ACI_CUSTOM != 0 {
            None
        } else {
            render_tok_text(cur, self.interner).chars().next()
        };
        let first_is = |c: char| first == Some(c);
        match kind {
            ConcatKind::Ident => {
                if matches!(cur.tok, PpToken::PpNumber(_)) {
                    return !first_is('.');
                }
                if matches!(cur.tok, PpToken::Ident(_)) {
                    return true;
                }
                if !matches!(cur.tok, PpToken::Str(_) | PpToken::Char(_)) {
                    return false;
                }
                // `L`, `u`, `U` followed by a string/char literal paste.
                matches!(render_tok_text(prev, self.interner).as_str(), "L" | "u" | "U")
            }
            ConcatKind::Number => first.map_or(false, |c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '+' || c == '-' || c == '$'),
            ConcatKind::Period => (first_is('.') && prevprev.map_or(false, |p| concat_kind(p) == ConcatKind::Period)) || first.map_or(false, |c| c.is_ascii_digit()),
            ConcatKind::Amp => first_is('&'),
            ConcatKind::Plus => first_is('+'),
            ConcatKind::Minus => first_is('-') || first_is('>'),
            ConcatKind::Slash => first_is('*') || first_is('/'),
            ConcatKind::Less => first_is('<') || first_is(':') || first_is('%'),
            ConcatKind::Greater => first_is('>'),
            ConcatKind::Pipe => first_is('|'),
            ConcatKind::Percent => first_is('>') || first_is(':'),
            ConcatKind::Colon => first_is('>'),
            ConcatKind::Hash => first_is('#') || first_is('@') || first_is('%'),
            // `->*`, `<=>` and the rest are C++ only, so no avoidance.
            ConcatKind::Arrow
            | ConcatKind::Star
            | ConcatKind::Exclaim
            | ConcatKind::Caret
            | ConcatKind::Equal
            | ConcatKind::LessLess
            | ConcatKind::GreaterGreater
            | ConcatKind::Other => false,
        }
    }

    fn expand_tree(&mut self, tt: TokenTree) -> Result<(), PpError> {
        match tt {
            TokenTree::Token(t) => {
                if self.emit_inline {
                    let out = self.expand_tok(t)?;
                    for o in &out {
                        self.emit_out(o)?;
                    }
                } else {
                    let out = self.expand_tok(t)?;
                    self.output.extend(out);
                }
            }
            TokenTree::Directive(control) => self.handle_control(control)?,
            TokenTree::IfDef { macro_name, body, opposition } => {
                let hit = self.is_defined_name(&macro_name);
                let trees = if hit { body } else { vec![*opposition] };
                self.trees.prepend_extend(trees.into_iter());
            }
            TokenTree::IfNDef { macro_name, body, opposition } => {
                let hit = self.is_defined_name(&macro_name);
                let trees = if hit { vec![*opposition] } else { body };
                self.trees.prepend_extend(trees.into_iter());
            }
            TokenTree::If { condition, body, opposition } => {
                let take = self.eval_condition(&condition)?;
                let trees = if take { body } else { vec![*opposition] };
                self.trees.prepend_extend(trees.into_iter());
            }
            TokenTree::Elif { condition, body, opposition } => {
                let take = self.eval_condition(&condition)?;
                let trees = if take { body } else { vec![*opposition] };
                self.trees.prepend_extend(trees.into_iter());
            }
            TokenTree::Else { body } => {
                self.trees.prepend_extend(body.into_iter());
            }
            TokenTree::Endif => {}
        }
        Ok(())
    }

    /// Fresh tokens carry raw source lines; the first expander to see
    /// one rewrites `line`/`end_line` to display numbering. Relocated
    /// and synthesized tokens already carry effective lines.
    fn eff_tok(&self, mut t: Tok) -> Tok {
        if !t.lined {
            t.line = self.eff_raw(t.line);
            t.end_line = self.eff_raw(t.end_line);
            t.lined = true;
        }
        t
    }

    /// Expand one token to output tokens, handling identifiers and
    /// leaf passthrough; everything downstream uses display lines.
    fn expand_tok(&mut self, t: Tok) -> Result<Vec<Tok>, PpError> {
        let mut t = self.eff_tok(t);
        // Whitespace is not a preprocessor token: it never consumes a
        // pending propagation.
        if matches!(t.tok, PpToken::Ws(_)) {
            return Ok(vec![t]);
        }
        // C11 6.10.6 `_Pragma`: handled where the operator is lexed (clang
        // `Preprocessor::Lex`), so `_Pragma` produced by macro expansion
        // runs its directive too.
        if let PpToken::Ident(sym) = t.tok {
            if self.spelling(sym) == "_Pragma" {
                if let Some(out) = self.expand_pragma_operator(&t)? {
                    return Ok(out);
                }
            }
        }
        // A pending empty-macro propagation reaches the next token: a
        // `TokenLexer` token ORs the flags in, while a fresh file token
        // takes its own whitespace/newline flags from the source layout
        // (clang `PropagateLineStartLeadingSpaceInfo` + the lexers).
        if let Some((csol, cspace)) = self.carry.take() {
            if t.expanded {
                t.sol |= csol;
                t.space |= cspace;
            } else {
                let tight = !t.space && !t.sol;
                t.sol |= csol;
                if tight {
                    t.space = cspace;
                }
            }
        }
        let PpToken::Ident(sym) = t.tok else {
            return Ok(vec![t]);
        };
        let name = self.spelling(sym);
        if self.respect_defined && name == "defined" {
            return self.expand_defined(&t);
        }
        if self.respect_defined && is_has_builtin_name(&name) {
            return self.expand_has_builtin(&t, &name);
        }
        // Dynamic builtins expand unless the source `#undef`ed them or a
        // macro of the same name shadows them (clang's `MacroInfo`).
        if is_location_name(&name) && !self.macros.contains_key(&sym) && !self.undef_builtins.contains(&name) {
            return Ok(vec![self.expand_dynamic(&t, &name)?]);
        }
        // C99 6.10.3.4p2: a macro that is being expanded (or was painted
        // by an earlier scan) never expands again; clang sets
        // `DisableExpand` on the token so later rescans leave it alone.
        if t.noexpand || self.expanding.contains(&sym) {
            if self.macros.contains_key(&sym) {
                t.noexpand = true;
            }
            return Ok(vec![t]);
        }
        match self.macros.get(&sym).cloned() {
            None => Ok(vec![t]),
            Some(MacroDef::Object(body)) => self.splice(sym, &body, &t),
            Some(MacroDef::Func {
                params,
                variadic,
                named_variadic,
                body,
            }) => self.expand_call(&t, &name, &params, variadic, named_variadic, &body),
        }
    }

    /// clang's dynamic builtin macros (`__LINE__`, `__FILE__`, ...):
    /// each expands in place, keeping the token flags.
    fn expand_dynamic(&mut self, t: &Tok, name: &str) -> Result<Tok, PpError> {
        let tok = match name {
            "__LINE__" => PpToken::PpNumber(self.interner.get_or_intern(t.end_line.to_string())),
            "__FILE__" => {
                let lit = quoted_display(&self.display());
                PpToken::Str(self.interner.get_or_intern(lit))
            }
            "__BASE_FILE__" => {
                let disp = self.interner.resolve(self.base_display).unwrap_or("").to_string();
                let lit = quoted_display(&disp);
                PpToken::Str(self.interner.get_or_intern(lit))
            }
            "__COUNTER__" => {
                let n = self.counter;
                self.counter += 1;
                PpToken::PpNumber(self.interner.get_or_intern(n.to_string()))
            }
            "__INCLUDE_LEVEL__" => {
                let level = self.files.depth.to_string();
                PpToken::PpNumber(self.interner.get_or_intern(level))
            }
            "__DATE__" | "__TIME__" => {
                let (date, time) = self.date_time_pair();
                let lit = quoted_display(if name == "__DATE__" { &date } else { &time });
                PpToken::Str(self.interner.get_or_intern(lit))
            }
            "__TIMESTAMP__" => {
                let ts = self.file_timestamp();
                let lit = quoted_display(&ts);
                PpToken::Str(self.interner.get_or_intern(lit))
            }
            _ => return Err(err(format!("{}: macro '{name}' is not supported by native pp yet", self.loc(t.line)))),
        };
        Ok(Tok {
            tok,
            line: t.line,
            file: t.file,
            expanded: true,
            lined: true,
            space: t.space,
            sol: t.sol,
            col: t.col,
            end_line: t.end_line,
            spell: None,
            noexpand: false,
        })
    }

    /// Local `__DATE__`/`__TIME__`, fixed at first use for the run.
    fn date_time_pair(&mut self) -> (String, String) {
        if self.date_time.is_none() {
            self.date_time = Some(crate::timefmt::date_time(std::time::SystemTime::now()));
        }
        self.date_time.clone().unwrap_or_default()
    }

    /// `__TIMESTAMP__` of the current file, cached per path.
    fn file_timestamp(&mut self) -> String {
        if let Some(v) = self.timestamps.get(&self.file_path) {
            return v.clone();
        }
        let v = crate::timefmt::file_timestamp(&self.file_path);
        self.timestamps.insert(self.file_path.clone(), v.clone());
        v
    }

    /// Splice a replacement list for rescan: substitute `#`/`##`
    /// (clang's `TokenLexer` handles both for every macro), restamp
    /// locations to the use site (clang expansion locs), and drain
    /// exactly that region, with blue paint held throughout. Every
    /// body token counts as expansion output, so `emit_out` pads where
    /// the replacement meets untouched source text.
    fn splice(&mut self, sym: SymbolU32, body: &[Tok], name: &Tok) -> Result<Vec<Tok>, PpError> {
        self.expanding.insert(sym);
        self.bump_depth(name.line)?;
        let (substituted, pending) = {
            let sub = ArgumentSubstitution {
                replacements: body.to_vec(),
                pos: 0,
                map: ArgMap::new(),
                extras: Vec::new(),
                is_variadic: false,
                named_variadic: None,
                variadic_elided: false,
                function_like: false,
                interner: &mut *self.interner,
                glue_chain: Vec::new(),
                glue_open: false,
                strict_c99: false,
                num_params: 0,
                next_space: false,
                prev_hashhash: false,
                seen_body_token: false,
                stringify_next: false,
                stringify_toks: Vec::new(),
            };
            sub.run()?
        };
        let mut out = Vec::new();
        for (i, s) in substituted.into_iter().enumerate() {
            let mut tok = s;
            tok.line = name.line;
            tok.col = name.col;
            tok.end_line = name.end_line;
            tok.lined = true;
            tok.expanded = true;
            // clang `TokenLexer::Lex`: the first emitted token takes the
            // macro name's `StartOfLine`/`LeadingSpace`.
            if i == 0 {
                tok.sol = name.sol;
                tok.space = name.space;
            }
            out.push(TokenTree::Token(tok));
        }
        self.drain_expansion(out, name, Some(sym), pending)
    }

    /// Drain one expansion's prepared tokens (paint already held) and
    /// re-enable the macro afterwards. An expansion that produces no
    /// output (or ends with an unconsumed pending space) propagates the
    /// macro's eof-token flags to the next token of this stream.
    fn drain_expansion(&mut self, trees: Vec<TokenTree>, name: &Tok, sym: Option<SymbolU32>, pending: bool) -> Result<Vec<Tok>, PpError> {
        let mut out = Vec::new();
        if !trees.is_empty() {
            let stop_at = self.trees.remaining();
            if let Some(sym) = sym {
                self.frames.push((sym, stop_at));
            }
            self.trees.prepend_extend(trees.into_iter());
            self.drain_into(stop_at, &mut out)?;
        }
        // clang `TokenLexer::Lex` at eof: `StartOfLine = AtStartOfLine`
        // (the macro name's, unless a token already consumed it) and
        // `LeadingSpace = HasLeadingSpace || NextTokGetsSpace`; a nested
        // empty expansion's propagation composes first.
        let (csol, cspace) = self.carry.take().unwrap_or(if out.is_empty() { (name.sol, name.space) } else { (false, false) });
        if csol || cspace || pending {
            self.carry = Some((csol, cspace || pending));
        }
        if let Some(sym) = sym {
            if matches!(self.frames.last(), Some(&(top, _)) if top == sym) {
                self.frames.pop();
            }
            self.expanding.remove(&sym);
        }
        self.depth -= 1;
        Ok(out)
    }

    /// clang pops an exhausted macro's `TokenLexer` (re-enabling the macro)
    /// as soon as the next token is requested. `frames` mirrors that:
    /// pulling past the innermost tracked region end re-enables its macro.
    fn sync_frames(&mut self) {
        while let Some(&(sym, stop_at)) = self.frames.last() {
            if self.trees.remaining() <= stop_at {
                self.frames.pop();
                self.expanding.remove(&sym);
            } else {
                break;
            }
        }
    }

    /// Expand exactly the prepended region (plus anything it prepends
    /// in turn) into `out`. Anything else in the stream is untouched.
    fn drain_into(&mut self, stop_at: usize, out: &mut Vec<Tok>) -> Result<(), PpError> {
        while self.trees.remaining() > stop_at {
            self.sync_frames();
            match self.trees.next() {
                None => break,
                Some(TokenTree::Token(t)) => {
                    let mut v = self.expand_tok(t)?;
                    out.append(&mut v);
                }
                Some(_) => {
                    return Err(err(format!("{}: internal error: unexpected tree in macro expansion", self.display())));
                }
            }
        }
        Ok(())
    }

    fn bump_depth(&mut self, line: u64) -> Result<(), PpError> {
        self.depth += 1;
        if self.depth > 256 {
            return Err(err(format!("{}: macro expansion too deep", self.loc(line))));
        }
        Ok(())
    }

    /// `defined X` / `defined(X)` inside conditions: consumes the
    /// operand from the tree queue.
    fn expand_defined(&mut self, t: &Tok) -> Result<Vec<Tok>, PpError> {
        let loc = self.loc(t.line);
        // Skip whitespace (never newlines here in practice, but harmless).
        let mut operand = None;
        while let Some(tree) = self.trees.next() {
            match tree {
                TokenTree::Token(t) if matches!(t.tok, PpToken::Ws(_)) => continue,
                other => {
                    operand = Some(other);
                    break;
                }
            }
        }
        let one = self.interner.get_or_intern("1".to_string());
        let zero = self.interner.get_or_intern("0".to_string());
        let num = |hit: bool| Tok {
            tok: PpToken::PpNumber(if hit { one } else { zero }),
            line: t.line,
            file: t.file,
            expanded: true,
            lined: true,
            space: t.space,
            sol: t.sol,
            col: t.col,
            end_line: t.end_line,
            spell: None,
            noexpand: false,
        };
        match operand {
            Some(TokenTree::Token(o)) if matches!(o.tok, PpToken::Ident(_)) => {
                let name = self.name_of(&o);
                let hit = self.is_defined_name(&name);
                Ok(vec![num(hit)])
            }
            Some(TokenTree::Token(o)) if matches!(o.tok, PpToken::Punct(Punctuator::LParen)) => {
                // Collect to the matching `)`; exactly one ident inside.
                let mut inner = Vec::new();
                let mut depth = 0u32;
                let mut closed = false;
                while let Some(tree) = self.trees.next() {
                    match tree {
                        TokenTree::Token(x) if matches!(x.tok, PpToken::Punct(Punctuator::LParen)) => {
                            depth += 1;
                            inner.push(x);
                        }
                        TokenTree::Token(x) if matches!(x.tok, PpToken::Punct(Punctuator::RParen)) => {
                            if depth == 0 {
                                closed = true;
                                break;
                            }
                            depth -= 1;
                            inner.push(x);
                        }
                        TokenTree::Token(x) => inner.push(x),
                        _ => {
                            return Err(err(format!("{loc}: 'defined(' expects an identifier, not tree content")));
                        }
                    }
                }
                if !closed {
                    return Err(err(format!("{loc}: 'defined(' without ')'")));
                }
                let words: Vec<Tok> = inner.into_iter().filter(|x| !matches!(x.tok, PpToken::Ws(_))).collect();
                let [only] = words.as_slice() else {
                    return Err(err(format!("{loc}: 'defined(...)' takes exactly one identifier")));
                };
                let PpToken::Ident(_) = only.tok else {
                    return Err(err(format!("{loc}: 'defined(...)' takes exactly one identifier")));
                };
                let name = self.name_of(only);
                let hit = self.is_defined_name(&name);
                Ok(vec![num(hit)])
            }
            _ => Err(err(format!("{loc}: 'defined' must be followed by an identifier, optionally parenthesized"))),
        }
    }

    fn name_of(&self, t: &Tok) -> String {
        match t.tok {
            PpToken::Ident(sym) => self.spelling(sym),
            _ => String::new(),
        }
    }

    fn is_defined_name(&mut self, name: &str) -> bool {
        let sym = self.interner.get_or_intern(name.to_string());
        self.macros.contains_key(&sym) || is_has_builtin_name(name) || (is_location_name(name) && !self.undef_builtins.contains(name))
    }

    /// Clang `__has_*` inside conditions: `__has_include[_next]` consults
    /// header search, features use the probed tables, attributes and
    /// builtins are claimed.
    fn expand_has_builtin(&mut self, t: &Tok, name: &str) -> Result<Vec<Tok>, PpError> {
        let loc = self.loc(t.line);
        // Collect the parenthesized argument from the tree queue.
        let mut arg_toks = Vec::new();
        let mut depth = 0u32;
        let mut opened = false;
        let mut closed = false;
        while let Some(tree) = self.trees.next() {
            match tree {
                TokenTree::Token(x) if matches!(x.tok, PpToken::Ws(_)) && !opened => continue,
                TokenTree::Token(x) if matches!(x.tok, PpToken::Punct(Punctuator::LParen)) && !opened => {
                    opened = true;
                }
                _ if !opened => {
                    return Err(err(format!("{loc}: '{name}' must be followed by '(...)'")));
                }
                TokenTree::Token(x) if matches!(x.tok, PpToken::Punct(Punctuator::LParen)) => {
                    depth += 1;
                    arg_toks.push(x);
                }
                TokenTree::Token(x) if matches!(x.tok, PpToken::Punct(Punctuator::RParen)) => {
                    if depth == 0 {
                        closed = true;
                        break;
                    }
                    depth -= 1;
                    arg_toks.push(x);
                }
                TokenTree::Token(x) => arg_toks.push(x),
                _ => return Err(err(format!("{loc}: '{name}' takes parenthesized arguments"))),
            }
        }
        if !opened || !closed {
            return Err(err(format!("{loc}: '{name}' must be followed by '(...)'")));
        }
        let value = match name {
            "__has_include" | "__has_include_next" => self.eval_has_include(&arg_toks, name == "__has_include_next", &loc)?,
            "__has_feature" | "__has_extension" => {
                let arg = render_raw(&arg_toks, self.interner);
                crate::has_feature(name, arg.trim(), self.std_flag.as_deref())
            }
            // Never building modules in a native preprocessing run.
            "__building_module" => false,
            _ => true,
        };
        let sym = self.interner.get_or_intern(if value { "1" } else { "0" }.to_string());
        Ok(vec![Tok {
            tok: PpToken::PpNumber(sym),
            line: t.line,
            file: t.file,
            expanded: true,
            lined: true,
            space: t.space,
            sol: t.sol,
            col: t.col,
            end_line: t.end_line,
            spell: None,
            noexpand: false,
        }])
    }

    fn eval_has_include(&mut self, arg_toks: &[Tok], next: bool, loc: &str) -> Result<bool, PpError> {
        // Expand any macros in the argument first (already-expanded
        // stream, but be thorough), then expect `"..."` / `<...>`.
        let mut expanded = Vec::new();
        for a in arg_toks {
            expanded.extend(self.expand_tok(a.clone())?);
        }
        let flat: Vec<&Tok> = expanded.iter().filter(|t| !matches!(t.tok, PpToken::Ws(_))).collect();
        let (header, quoted) = match flat.as_slice() {
            [t] if matches!(t.tok, PpToken::Str(_)) => {
                let raw = render_raw(std::slice::from_ref(t), self.interner);
                let body = raw.strip_prefix('"').and_then(|s| s.strip_suffix('"')).unwrap_or(&raw);
                (body.to_string(), true)
            }
            [l, rest @ ..] if matches!(l.tok, PpToken::Punct(Punctuator::Less)) => {
                // To the matching `>`.
                let mut depth = 0u32;
                let mut end = None;
                for (i, x) in rest.iter().enumerate() {
                    match x.tok {
                        PpToken::Punct(Punctuator::Less) => depth += 1,
                        PpToken::Punct(Punctuator::Greater) if depth == 0 => {
                            end = Some(i);
                            break;
                        }
                        PpToken::Punct(Punctuator::Greater) => depth -= 1,
                        _ => {}
                    }
                }
                let Some(end) = end else {
                    return Ok(false);
                };
                (render_raw(&rest[..end].iter().map(|t| (*t).clone()).collect::<Vec<_>>(), self.interner), false)
            }
            _ => return Ok(false),
        };
        let dir = self.file_path.parent().map(std::path::Path::to_path_buf).unwrap_or_else(|| std::path::PathBuf::from("."));
        let after = next.then_some(self.file_found);
        Ok(self.files.resolve_header(&header, quoted, after, &dir, loc).is_ok())
    }

    /// Expand one function-like call: collect balanced args, prescan,
    /// substitute, rescan.
    fn expand_call(&mut self, t: &Tok, name: &str, params: &[String], variadic: bool, named_variadic: bool, body: &[Tok]) -> Result<Vec<Tok>, PpError> {
        // The `(` may follow on later lines; skip whitespace (never a
        // directive: that aborts the call and leaves it literal).
        let mut depth = 0u32;
        let mut arg_toks: Vec<Tok> = Vec::new();
        let mut closed = false;
        let mut rparen: Option<Tok> = None;
        let mut saw_open = false;
        // Whitespace skipped while looking for the `(`: restored
        // verbatim when this turns out not to be a call.
        let mut skipped: Vec<TokenTree> = Vec::new();
        loop {
            self.sync_frames();
            let Some(tree) = self.trees.next() else {
                break;
            };
            match tree {
                TokenTree::Token(x) if !saw_open && matches!(x.tok, PpToken::Ws(_)) => {
                    skipped.push(TokenTree::Token(x));
                    continue;
                }
                TokenTree::Token(x) if !saw_open && matches!(x.tok, PpToken::Punct(Punctuator::LParen)) => {
                    saw_open = true;
                }
                _ if !saw_open => {
                    // Not a call: put everything back, emit the name.
                    skipped.push(tree);
                    self.trees.prepend_extend(skipped.into_iter());
                    return Ok(vec![t.clone()]);
                }
                TokenTree::Token(x) if matches!(x.tok, PpToken::Directive) => {
                    // A directive ends the would-be call: leave literal
                    // (put the directive back for normal handling).
                    skipped.push(TokenTree::Token(x));
                    self.trees.prepend_extend(skipped.into_iter());
                    return Ok(vec![t.clone()]);
                }
                TokenTree::Token(x) if matches!(x.tok, PpToken::Punct(Punctuator::LParen)) => {
                    depth += 1;
                    arg_toks.push(x);
                }
                TokenTree::Token(x) if matches!(x.tok, PpToken::Punct(Punctuator::RParen)) => {
                    if depth == 0 {
                        closed = true;
                        rparen = Some(self.eff_tok(x));
                        break;
                    }
                    depth -= 1;
                    arg_toks.push(x);
                }
                TokenTree::Token(mut x) => {
                    // clang `readMacroCallArgumentList`: an identifier
                    // naming a macro that is disabled right now (its
                    // expansion is in progress) is painted
                    // `DisableExpand`, so re-enabling while this list is
                    // read does not let it expand later (C99 6.10.3.4p2,
                    // PR3927).
                    if let PpToken::Ident(sym) = x.tok {
                        if self.macros.contains_key(&sym) && self.expanding.contains(&sym) {
                            x.noexpand = true;
                        }
                    }
                    arg_toks.push(x);
                }
                other => {
                    // clang `readMacroCallArgumentList` processes
                    // directives found while reading an argument list
                    // (PR4438/PR5163): splice conditionals and run
                    // control lines, then keep collecting.
                    self.expand_tree(other)?;
                }
            }
        }
        if !saw_open {
            self.trees.prepend_extend(skipped.into_iter());
            return Ok(vec![t.clone()]);
        }
        if !closed {
            // Unbalanced: put the consumed tokens back, leave literal.
            let mut back = skipped;
            back.extend(arg_toks.into_iter().map(TokenTree::Token));
            self.trees.prepend_extend(back.into_iter());
            return Ok(vec![t.clone()]);
        }
        // Call arguments split on top-level commas; only parens nest.
        let mut pieces = crate::macros::split_call_args(&arg_toks);
        // `F()` is an empty argument list: zero args for a parameterless
        // macro, one empty argument otherwise (matching clang, including
        // `G( )`).
        let empty_list = pieces.len() == 1 && pieces[0].iter().all(|x| matches!(x.tok, PpToken::Ws(_)));
        if empty_list {
            if !variadic && params.is_empty() {
                pieces.clear();
            } else {
                pieces = vec![&[] as &[Tok]];
            }
        }
        // The named parameters take the leading pieces; a variadic macro
        // keeps the rest (including interior commas) as one pack
        // (clang `ReadMacroCallArgumentList`).
        let fixed = if named_variadic { params.len() - 1 } else { params.len() };
        let actuals = if empty_list { 0 } else { pieces.len() };
        if !variadic {
            if pieces.len() != params.len() {
                return Err(err(format!("{}: macro '{name}' needs {} arguments, got {}", self.loc(t.line), params.len(), pieces.len())));
            }
        } else if actuals < fixed && !(empty_list && fixed <= 1) {
            // clang allows a missing variadic pack, not missing named
            // parameters.
            return Err(err(format!(
                "{}: macro '{name}' needs at least {} arguments, got {}",
                self.loc(t.line),
                fixed,
                pieces.len()
            )));
        }
        // The pack was elided entirely (rather than passed empty):
        // GNU `, ## __VA_ARGS__` swallows the comma only then.
        let variadic_elided = variadic && actuals <= fixed;
        // Prescan without the invoking macro painted (C17 6.10.3.1):
        // each argument expands as its own stream, so nested calls
        // expand inside-out.
        let mut map: ArgMap = HashMap::new();
        for (i, p) in params.iter().enumerate().take(fixed) {
            let raw: Vec<Tok> = trim_tokens(pieces[i]).into_iter().cloned().collect();
            let cooked = self.expand_arg(raw.clone())?;
            map.insert(
                p.clone(),
                MacroArgument {
                    unexpanded: raw,
                    expanded: cooked,
                },
            );
        }
        let mut extras = Vec::new();
        for piece in pieces.iter().skip(fixed) {
            let raw: Vec<Tok> = trim_tokens(piece).into_iter().cloned().collect();
            let cooked = self.expand_arg(raw.clone())?;
            extras.push(MacroArgument {
                unexpanded: raw,
                expanded: cooked,
            });
        }
        if named_variadic && !params.is_empty() {
            // GNU `name...`: the pack substitutes as the comma-joined
            // pieces (clang keeps them as one token run).
            let mut raw = Vec::new();
            let mut cooked = Vec::new();
            for (i, extra) in extras.iter().enumerate() {
                if i > 0 {
                    let sep = Tok {
                        tok: PpToken::Punct(Punctuator::Comma),
                        line: t.line,
                        file: t.file,
                        expanded: true,
                        lined: true,
                        space: false,
                        sol: false,
                        col: t.col,
                        end_line: t.end_line,
                        spell: None,
                        noexpand: false,
                    };
                    raw.push(sep.clone());
                    cooked.push(sep);
                }
                raw.extend(extra.unexpanded.iter().cloned());
                cooked.extend(extra.expanded.iter().cloned());
            }
            map.insert(
                params[fixed].clone(),
                MacroArgument {
                    unexpanded: raw,
                    expanded: cooked,
                },
            );
        }
        let PpToken::Ident(sym) = t.tok else {
            return Err(err(format!("{}: internal error: call site is not an identifier", self.loc(t.line))));
        };
        self.expanding.insert(sym);
        self.bump_depth(t.line)?;
        let strict_c99 = self.strict_c99();
        let num_params = params.len() + usize::from(variadic && !named_variadic);
        let (substituted, pending) = {
            let (body_owned, map_owned, extras_owned) = (body.to_vec(), map, extras);
            let sub = ArgumentSubstitution {
                replacements: body_owned,
                pos: 0,
                map: map_owned,
                extras: extras_owned,
                is_variadic: variadic,
                named_variadic: if named_variadic { params.last().cloned() } else { None },
                variadic_elided,
                function_like: true,
                interner: &mut *self.interner,
                glue_chain: Vec::new(),
                glue_open: false,
                strict_c99,
                num_params,
                next_space: false,
                prev_hashhash: false,
                seen_body_token: false,
                stringify_next: false,
                stringify_toks: Vec::new(),
            };
            sub.run()?
        };
        let mut out = Vec::new();
        // clang `HandleMacroExpandedIdentifier`: the expansion range ends
        // at the `)` for a function-like macro, and `__LINE__` resolves
        // through the range's end.
        let end_line = rparen.map_or(t.end_line, |r| r.end_line);
        for (i, s) in substituted.into_iter().enumerate() {
            let mut tok = s;
            tok.line = t.line;
            tok.col = t.col;
            tok.end_line = end_line;
            tok.lined = true;
            tok.expanded = true;
            // clang `TokenLexer::Lex`: the first emitted token takes the
            // macro name's `StartOfLine`/`LeadingSpace`.
            if i == 0 {
                tok.sol = t.sol;
                tok.space = t.space;
            }
            out.push(TokenTree::Token(tok));
        }
        self.drain_expansion(out, t, Some(sym), pending)
    }

    /// Prescan one argument as its own stream with shared macros and
    /// paint. Output tokens are marked expanded
    /// (interior newlines drop at render, like `clang -E`).
    fn expand_arg(&mut self, arg: Vec<Tok>) -> Result<Vec<Tok>, PpError> {
        let mut sub = Expander {
            interner: &mut *self.interner,
            trees: PrependingPeekableIterator::new(arg.into_iter().map(TokenTree::Token).collect::<Vec<_>>().into_iter()),
            output: Vec::new(),
            macros: Rc::clone(&self.macros),
            expanding: self.expanding.clone(),
            respect_defined: self.respect_defined,
            file_display: self.file_display,
            file_path: self.file_path.clone(),
            file_found: self.file_found,
            base_display: self.base_display,
            counter: self.counter,
            undef_builtins: self.undef_builtins.clone(),
            date_time: self.date_time.clone(),
            timestamps: HashMap::new(),
            pushed_macros: HashMap::new(),
            line_base: self.line_base,
            display_override: self.display_override,
            target: self.target,
            std_flag: self.std_flag.clone(),
            depth: self.depth,
            files: &mut *self.files,
            mark_n: 1,
            mark_emit: 0,
            fresh: true,
            prev: None,
            emit_inline: false,
            frames: self.frames.clone(),
            r_cur_line: 1,
            r_tokens_on_line: false,
            r_directive_on_line: false,
            r_pending_sol: false,
            r_prev: None,
            r_prevprev: None,
            carry: None,
        };
        sub.expand()?;
        self.counter = sub.counter;
        if self.date_time.is_none() {
            self.date_time = sub.date_time.clone();
        }
        let mut out = std::mem::take(&mut sub.output);
        drop(sub);
        for t in &mut out {
            t.expanded = true;
        }
        Ok(out)
    }

    /// Evaluate an `#if`/`#elif` condition: expand with a sub-expander
    /// (`respect_defined`, shared macros), then fold through
    /// `glcc_frontend` consteval. Unfoldable garbage selects `#else`.
    fn eval_condition(&mut self, condition: &[Tok]) -> Result<bool, PpError> {
        if std::env::var_os("GLCC_PP_DEBUG").is_some() {
            let raw = condition.iter().map(|t| show_tok(t, self.interner)).collect::<Vec<_>>().join("");
            eprintln!("DBG raw condition in {}: {}", self.display(), raw.trim());
        }
        let mut sub = Expander {
            interner: &mut *self.interner,
            trees: PrependingPeekableIterator::new(condition.iter().cloned().map(TokenTree::Token).collect::<Vec<_>>().into_iter()),
            output: Vec::new(),
            macros: Rc::clone(&self.macros),
            expanding: HashSet::new(),
            respect_defined: true,
            file_display: self.file_display,
            file_path: self.file_path.clone(),
            file_found: self.file_found,
            base_display: self.base_display,
            counter: self.counter,
            undef_builtins: self.undef_builtins.clone(),
            date_time: self.date_time.clone(),
            timestamps: HashMap::new(),
            pushed_macros: HashMap::new(),
            line_base: self.line_base,
            display_override: self.display_override,
            target: self.target,
            std_flag: self.std_flag.clone(),
            depth: 0,
            files: &mut *self.files,
            mark_n: 1,
            mark_emit: 0,
            fresh: true,
            prev: None,
            emit_inline: false,
            frames: Vec::new(),
            r_cur_line: 1,
            r_tokens_on_line: false,
            r_directive_on_line: false,
            r_pending_sol: false,
            r_prev: None,
            r_prevprev: None,
            carry: None,
        };
        sub.expand()?;
        self.counter = sub.counter;
        if self.date_time.is_none() {
            self.date_time = sub.date_time.clone();
        }
        let output = std::mem::take(&mut sub.output);
        drop(sub);
        let text = Self::render_condition(&output, self.interner);
        let verdict = eval_frontend_text(&text, self.target);
        if std::env::var_os("GLCC_PP_DEBUG").is_some() {
            eprintln!("DBG verdict [{}] = {} in {}", text.trim(), verdict, self.display());
        }
        Ok(verdict)
    }

    /// Render expanded condition tokens to text: leftover identifiers
    /// count as `0` (C17 6.10.1p4); keywords and literals verbatim.
    fn render_condition(toks: &[Tok], interner: &Interner) -> String {
        let mut s = String::new();
        for t in toks {
            match t.tok {
                PpToken::Ident(sym) => {
                    let w = interner.resolve(sym).unwrap_or("");
                    if is_keyword(w) {
                        s.push_str(w);
                    } else {
                        s.push('0');
                    }
                }
                PpToken::PpNumber(sym) | PpToken::Str(sym) | PpToken::Char(sym) => {
                    s.push_str(interner.resolve(sym).unwrap_or(""));
                    s.push(' ');
                }
                PpToken::Punct(p) => {
                    s.push_str(p.as_str());
                    s.push(' ');
                }
                PpToken::Ws(_) => s.push(' '),
                PpToken::Directive => s.push('#'),
                PpToken::Other(c) => s.push(c),
            }
        }
        s
    }

    fn handle_control(&mut self, control: ControlLine) -> Result<(), PpError> {
        match control {
            ControlLine::DefineObject { macro_name, replacement } => {
                let sym = self.interner.get_or_intern(macro_name);
                Rc::make_mut(&mut self.macros).insert(sym, MacroDef::Object(replacement));
            }
            ControlLine::DefineFunction {
                macro_name,
                parameters,
                replacement,
            } => {
                let loc = self.loc(parameters.first().map(|t| t.line).unwrap_or(0));
                let parsed = parse_parameters(&parameters, &loc, &mut *self.interner)?;
                let sym = self.interner.get_or_intern(macro_name);
                Rc::make_mut(&mut self.macros).insert(
                    sym,
                    MacroDef::Func {
                        params: parsed.parameters,
                        variadic: parsed.variadic,
                        named_variadic: parsed.named_variadic,
                        body: replacement,
                    },
                );
            }
            ControlLine::Undefine(name) => {
                let sym = self.interner.get_or_intern(name.clone());
                Rc::make_mut(&mut self.macros).remove(&sym);
                if is_location_name(&name) {
                    self.undef_builtins.insert(name);
                }
            }
            ControlLine::Error(message) => {
                let text = message.iter().map(|t| show_tok(t, self.interner)).collect::<Vec<_>>().join("");
                let line = message.first().map(|t| self.eff_raw(t.line)).unwrap_or(0);
                return Err(err(format!("{}: #error{}", self.loc(line), text.trim())));
            }
            ControlLine::Warning => {}
            ControlLine::PragmaOnce => {
                self.files.record_once(&self.file_path);
            }
            ControlLine::PragmaOther { tokens, line } => {
                let line = self.eff_raw(line);
                let tokens: Vec<Tok> = tokens
                    .into_iter()
                    .map(|mut t| {
                        t.line = self.eff_raw(t.line);
                        t.lined = true;
                        t
                    })
                    .collect();
                let (leftovers, eod) = self.handle_pragma_line(&tokens, line)?;
                for t in leftovers {
                    self.emit_out(&t)?;
                }
                if let Some((line, col)) = eod {
                    self.emit_eod_artifact(line, col);
                }
            }
            ControlLine::Line(tail) => {
                self.apply_line(&tail)?;
            }
            ControlLine::Include { header, verbatim, next, .. } => {
                self.handle_include(&header, verbatim, next)?;
            }
            ControlLine::Empty | ControlLine::EndIf => {}
            ControlLine::IfDef { .. } | ControlLine::IfNDef { .. } | ControlLine::If { .. } | ControlLine::Elif { .. } | ControlLine::Else => {
                // Built by the tree layer into If-nodes; never observed here.
            }
        }
        Ok(())
    }

    /// Reproduce clang's `-E` handling of one `#pragma` line: the
    /// callbacks and unknown-pragma handlers installed by
    /// `PrintPreprocessedOutput.cpp`. Returns tokens the handler left in
    /// the stream (printed like ordinary source) and the directive's
    /// end-of-line location when the handler did not consume it.
    fn handle_pragma_line(&mut self, toks: &[Tok], line: u64) -> Result<(Vec<Tok>, Option<(u64, u64)>), PpError> {
        let mut leftovers = Vec::new();
        let mut eod = None;
        let echo = self.emit_inline && self.files.opts.linemarkers;
        let Some(first) = toks.first().cloned() else {
            // Bare `#pragma`: the root null handler gets the eod.
            if echo {
                self.echo_pragma("#pragma", line, &[]);
            }
            return Ok((leftovers, eod));
        };
        match self.name_of(&first).as_str() {
            "GCC" | "clang" => {
                let ns = self.name_of(&first);
                let prefix = if ns == "GCC" { "#pragma GCC" } else { "#pragma clang" };
                let rest = &toks[1..];
                match rest.first().map(|t| self.name_of(t)) {
                    None => {
                        if echo {
                            self.echo_pragma(prefix, line, &[]);
                        }
                    }
                    Some(op) => match op.as_str() {
                        "diagnostic" => {
                            if echo {
                                self.echo_pragma_diagnostic(&ns, rest, line);
                            }
                        }
                        "assume_nonnull" if ns == "clang" => {
                            if echo {
                                self.echo_assume_nonnull(rest, line);
                            }
                        }
                        "__debug" if ns == "clang" => {
                            if let Some(cmd_tok) = rest.get(1) {
                                let cmd = self.name_of(cmd_tok);
                                if !cmd.is_empty() {
                                    if echo {
                                        self.echo_line(line, &format!("#pragma clang __debug {cmd}"));
                                    }
                                    // `captured` reads the eod itself; the
                                    // other commands leave the rest of the
                                    // line (and its eod) to the stream.
                                    if cmd != "captured" {
                                        leftovers.extend(rest[2..].iter().cloned());
                                        if self.files.opts.linemarkers {
                                            eod = Some((line, self.pragma_eod_col(toks)));
                                        }
                                    }
                                }
                            }
                        }
                        // Handlers with no printing callback.
                        "poison" | "system_header" | "dependency" | "__set_pp_state" => {}
                        _ => {
                            if echo {
                                self.echo_pragma(prefix, line, rest);
                            }
                        }
                    },
                }
            }
            "omp" => {
                if echo {
                    // `#pragma omp` tokens are macro-expanded.
                    let mut expanded = Vec::new();
                    for t in &toks[1..] {
                        expanded.extend(self.expand_tok(t.clone())?);
                    }
                    let flat: Vec<Tok> = expanded.into_iter().filter(|t| !matches!(t.tok, PpToken::Ws(_))).collect();
                    self.echo_pragma("#pragma omp", line, &flat);
                }
            }
            // Consumed silently: clang's handlers have no printing
            // callback, unlike the unknown-pragma fallback below.
            "push_macro" => self.pragma_push_macro(toks),
            "pop_macro" => self.pragma_pop_macro(toks),
            _ => {
                if echo {
                    self.echo_pragma("#pragma", line, toks);
                }
            }
        }
        Ok((leftovers, eod))
    }

    /// `#pragma push_macro("name")`: save the current definition (or the
    /// undefined state) for a later `pop_macro`.
    fn pragma_push_macro(&mut self, toks: &[Tok]) {
        let Some(name) = self.pragma_macro_name(toks) else {
            return;
        };
        let sym = self.interner.get_or_intern(name.clone());
        let def = self.macros.get(&sym).cloned();
        self.pushed_macros.entry(name).or_default().push(def);
    }

    /// `#pragma pop_macro("name")`: restore the definition pushed
    /// before, removing the macro if it was undefined then.
    fn pragma_pop_macro(&mut self, toks: &[Tok]) {
        let Some(name) = self.pragma_macro_name(toks) else {
            return;
        };
        let Some(stack) = self.pushed_macros.get_mut(&name) else {
            return;
        };
        let Some(def) = stack.pop() else {
            return;
        };
        let sym = self.interner.get_or_intern(name);
        match def {
            Some(d) => {
                Rc::make_mut(&mut self.macros).insert(sym, d);
            }
            None => {
                Rc::make_mut(&mut self.macros).remove(&sym);
            }
        }
    }

    /// The `("name")` operand of `push_macro`/`pop_macro`: a string
    /// literal or bare identifier, never macro-expanded.
    fn pragma_macro_name(&self, toks: &[Tok]) -> Option<String> {
        let flat: Vec<&Tok> = toks.iter().filter(|t| !matches!(t.tok, PpToken::Ws(_))).collect();
        let [_, open, operand, close, ..] = flat.as_slice() else {
            return None;
        };
        if !matches!(open.tok, PpToken::Punct(Punctuator::LParen)) || !matches!(close.tok, PpToken::Punct(Punctuator::RParen)) {
            return None;
        }
        match &operand.tok {
            PpToken::Str(sym) => Some(unescape_string_body(&self.spelling(*sym))),
            PpToken::Ident(sym) => Some(self.spelling(*sym)),
            _ => None,
        }
    }

    /// clang `UnknownPragmaHandler`: move to the directive's line, write
    /// the namespace prefix, then every remaining token separated by the
    /// usual single-space / `AvoidConcat` rules.
    fn echo_pragma(&mut self, prefix: &str, line: u64, toks: &[Tok]) {
        let target = toks.first().map(|t| t.line).unwrap_or(line);
        self.move_to_line(target, true);
        self.files.out.push_str(prefix);
        self.r_tokens_on_line = true;
        let mut first = true;
        for t in toks {
            self.handle_ws_before(t, first, true);
            first = false;
            self.files.out.push_str(&render_tok_text(t, self.interner));
            self.r_tokens_on_line = true;
            self.r_prevprev = self.r_prev.take();
            self.r_prev = Some(t.clone());
        }
        self.r_directive_on_line = true;
    }

    /// A directive callback that prints one normalized line
    /// (`PragmaDebug`, `PragmaDiagnostic*`, `PragmaAssumeNonNullBegin`,
    /// `PragmaMessage`, ...).
    fn echo_line(&mut self, line: u64, text: &str) {
        self.move_to_line(line, true);
        self.files.out.push_str(text);
        self.r_directive_on_line = true;
    }

    /// clang `MoveToLine` with an invalid location (`PragmaAssumeNonNullEnd`
    /// resets the active region): the target collapses to the current
    /// line, which still ends the line in progress and re-syncs with a
    /// marker when the line number moved backwards.
    fn move_to_line_invalid(&mut self) {
        let target = self.r_cur_line;
        self.move_to_line(target, true);
    }

    /// `#pragma GCC|clang diagnostic ...` (`PragmaDiagnostic*`): the
    /// callback rebuilds the directive from the parsed severity and
    /// option name.
    fn echo_pragma_diagnostic(&mut self, ns: &str, rest: &[Tok], line: u64) {
        // `rest` starts at the `diagnostic` token.
        let Some(op) = rest.get(1) else {
            return;
        };
        let op_name = self.name_of(op);
        if matches!(op_name.as_str(), "push" | "pop") {
            if rest.len() == 2 {
                self.echo_line(line, &format!("#pragma {ns} diagnostic {op_name}"));
            }
            return;
        }
        if !matches!(op_name.as_str(), "ignored" | "warning" | "error" | "fatal") {
            return;
        }
        let Some(opt) = rest.get(2) else {
            return;
        };
        let Some(body) = self.string_literal_body(opt) else {
            return;
        };
        if !(body.starts_with("-W") || body.starts_with("-R")) {
            return;
        }
        self.echo_line(line, &format!("#pragma {ns} diagnostic {op_name} \"{body}\""));
    }

    /// `#pragma clang assume_nonnull begin|end` (`PragmaAssumeNonNull*`).
    fn echo_assume_nonnull(&mut self, rest: &[Tok], line: u64) {
        match rest.get(1).map(|t| self.name_of(t)).as_deref() {
            Some("begin") => self.echo_line(line, "#pragma clang assume_nonnull begin"),
            Some("end") => {
                self.move_to_line_invalid();
                self.files.out.push_str("#pragma clang assume_nonnull end");
                self.r_directive_on_line = true;
            }
            _ => {}
        }
    }

    /// The print loop's `tok::eod` case: the end-of-directive token prints
    /// nothing but still moves to its line (a marker when the directive's
    /// line is already behind) and indents to its column.
    fn emit_eod_artifact(&mut self, line: u64, col: u64) {
        let eod = Tok {
            tok: PpToken::Ws(PpWs::Space),
            line,
            file: self.file_display,
            expanded: false,
            lined: true,
            space: false,
            sol: false,
            col,
            end_line: line,
            spell: None,
            noexpand: false,
        };
        let sol = self.r_pending_sol;
        self.handle_ws_before(&eod, false, !sol);
        self.r_pending_sol = true;
    }

    /// Column just past the last token of a directive: where the lexer
    /// puts the end-of-directive token.
    fn pragma_eod_col(&self, toks: &[Tok]) -> u64 {
        toks.last().map(|t| t.col + render_tok_text(t, self.interner).chars().count() as u64).unwrap_or(0)
    }

    /// Decoded body of a string-literal token. Only the `\"`/`\\` escapes
    /// `_Pragma` destringizing and diagnostic rewriting undo are handled;
    /// other escapes rarely appear in directives.
    fn string_literal_body(&self, t: &Tok) -> Option<String> {
        let PpToken::Str(sym) = t.tok else {
            return None;
        };
        Some(unescape_string_body(&self.spelling(sym)))
    }

    /// C11 6.10.6 `_Pragma`: consume the `( "..." )` operand, destringize
    /// and lex the body, then run it like a `#pragma` line whose tokens
    /// sit at the invocation site. Returns `None`, with the scanned trees
    /// restored, when the operand shape does not match.
    fn expand_pragma_operator(&mut self, site: &Tok) -> Result<Option<Vec<Tok>>, PpError> {
        if !self.emit_inline {
            return Ok(None);
        }
        let mut scanned: Vec<TokenTree> = Vec::new();
        let mut step = 0u8;
        let mut string: Option<SymbolU32> = None;
        let mut matched = false;
        loop {
            self.sync_frames();
            let Some(tt) = self.trees.next() else {
                break;
            };
            let TokenTree::Token(x) = &tt else {
                scanned.push(tt);
                break;
            };
            if matches!(x.tok, PpToken::Ws(_)) {
                scanned.push(tt);
                continue;
            }
            match (step, &x.tok) {
                (0, PpToken::Punct(Punctuator::LParen)) => step = 1,
                (1, PpToken::Str(sym)) => {
                    string = Some(*sym);
                    step = 2;
                }
                (2, PpToken::Punct(Punctuator::RParen)) => {
                    scanned.push(tt);
                    matched = true;
                    break;
                }
                _ => {
                    scanned.push(tt);
                    break;
                }
            }
            scanned.push(tt);
        }
        let Some(sym) = string.filter(|_| matched) else {
            self.trees.prepend_extend(scanned.into_iter());
            return Ok(None);
        };
        let text = unescape_string_body(&self.spelling(sym));
        let toks = self.lex_pragma_text(&text, site);
        let (leftovers, eod) = self.handle_pragma_line(&toks, site.line)?;
        if leftovers.is_empty() {
            if let Some((line, col)) = eod {
                self.emit_eod_artifact(line, col);
            }
        } else {
            self.trees.prepend_extend(leftovers.into_iter().map(TokenTree::Token));
        }
        Ok(Some(Vec::new()))
    }

    /// Lex a destringized `_Pragma` body at the invocation site (clang
    /// `Lexer::Create_PragmaLexer`).
    fn lex_pragma_text(&mut self, text: &str, site: &Tok) -> Vec<Tok> {
        let (line, file, col) = (site.line, site.file, site.col);
        let interner = &mut *self.interner;
        Lexer::new(text, interner)
            .lex_pp_all()
            .into_iter()
            .filter(|t| !matches!(t.tok, PpToken::Ws(_)))
            .map(|t| Tok {
                tok: t.tok,
                line,
                file,
                expanded: false,
                lined: true,
                space: t.space,
                sol: false,
                col,
                end_line: line,
                spell: None,
                noexpand: false,
            })
            .collect()
    }

    /// `#line n` / `#line n "file"`: the *next* source line reports as
    /// `n` (optionally under another display name). Operands are
    /// macro-expanded first (C11 6.10.4).
    fn apply_line(&mut self, tail: &[Tok]) -> Result<(), PpError> {
        let loc = self.loc(tail.first().map(|t| self.eff_raw(t.line)).unwrap_or(0));
        let mut expanded = Vec::new();
        for t in tail {
            expanded.extend(self.expand_tok(t.clone())?);
        }
        let flat: Vec<&Tok> = expanded.iter().filter(|t| !matches!(t.tok, PpToken::Ws(_))).collect();
        let Some(num_tok) = flat.first() else {
            return Err(err(format!("{loc}: malformed #line")));
        };
        let PpToken::PpNumber(sym) = num_tok.tok else {
            return Err(err(format!("{loc}: malformed #line")));
        };
        let n: i64 = self.spelling(sym).parse().map_err(|_| err(format!("{loc}: malformed #line")))?;
        if let Some(file_tok) = flat.get(1) {
            let PpToken::Str(sym) = file_tok.tok else {
                return Err(err(format!("{loc}: malformed #line")));
            };
            let raw = self.spelling(sym);
            let body = raw.strip_prefix('"').and_then(|s| s.strip_suffix('"')).unwrap_or(&raw);
            self.display_override = Some(self.interner.get_or_intern(unescape_header(body)));
        }
        // GNU linemarkers may carry integer flags after the file name
        // (1 = enter, 2 = return, 3 = system, 4 = extern C). They
        // describe an include stack this preprocessor does not rebuild
        // from preprocessed input, so drop them.
        for flag in flat.iter().skip(2) {
            if !matches!(flag.tok, PpToken::PpNumber(_)) {
                return Err(err(format!("{loc}: malformed #line")));
            }
        }
        // `num_tok.line` carries an effective line; back out the current
        // base so the *next* raw line reports `n`.
        let raw = num_tok.line as i64 - self.line_base;
        self.line_base = n - raw - 1;
        Ok(())
    }

    /// `#include` / `#include_next`: expand non-verbatim headers, resolve,
    /// and recurse with enter/return markers.
    fn handle_include(&mut self, header: &[Tok], verbatim: bool, next: bool) -> Result<(), PpError> {
        let site_line = header.first().map(|t| self.eff_raw(t.line)).unwrap_or(0);
        let loc = self.loc(site_line);
        let (name, is_quoted) = if verbatim {
            self.parse_header_name(header, &loc)?
        } else {
            // Expand first; the result must take header-name form.
            let mut expanded = Vec::new();
            for h in header {
                expanded.extend(self.expand_tok(h.clone())?);
            }
            let flat: Vec<Tok> = expanded.into_iter().filter(|t| !matches!(t.tok, PpToken::Ws(_))).collect();
            self.parse_header_tokens(&flat, &loc)?
        };
        let dir = self.file_path.parent().map(std::path::Path::to_path_buf).unwrap_or_else(|| std::path::PathBuf::from("."));
        let after = next.then_some(self.file_found);
        let (target, found) = self.files.resolve_header(&name, is_quoted, after, &dir, &loc)?;
        if self.include_file(&target, found, site_line)? && self.files.opts.linemarkers {
            // clang `FileChanged(ExitFile)`: back to the line after the
            // `#include`, named by a ` 2` marker.
            let disp = self.display();
            let next_line = site_line + 1;
            self.r_cur_line = next_line;
            self.write_line_info_flag(next_line, &disp, 2);
            self.mark_n = next_line as i64 - 1;
            self.mark_emit = 0;
        }
        Ok(())
    }

    /// Parse an already-verbatim header (`"..."` / `<...>`) to
    /// (name, quoted).
    fn parse_header_name(&self, header: &[Tok], loc: &str) -> Result<(String, bool), PpError> {
        let flat: Vec<&Tok> = header.iter().filter(|t| !matches!(t.tok, PpToken::Ws(_))).collect();
        match flat.as_slice() {
            [t] => match t.tok {
                PpToken::Str(sym) => {
                    let raw = self.spelling(sym);
                    let body = raw.strip_prefix('"').and_then(|s| s.strip_suffix('"')).unwrap_or(&raw);
                    Ok((unescape_header(body), true))
                }
                _ => Err(err(format!("{loc}: malformed #include (expected \"...\" or <...>)"))),
            },
            [l, rest @ ..] if matches!(l.tok, PpToken::Punct(Punctuator::Less)) => {
                let mut depth = 0u32;
                let mut end = None;
                for (i, x) in rest.iter().enumerate() {
                    match x.tok {
                        PpToken::Punct(Punctuator::Less) => depth += 1,
                        PpToken::Punct(Punctuator::Greater) if depth == 0 => {
                            end = Some(i);
                            break;
                        }
                        PpToken::Punct(Punctuator::Greater) => depth -= 1,
                        _ => {}
                    }
                }
                let Some(end) = end else {
                    return Err(err(format!("{loc}: malformed #include (missing '>')")));
                };
                if rest[end + 1..].iter().any(|x| !matches!(x.tok, PpToken::Ws(_))) {
                    return Err(err(format!("{loc}: malformed #include (trailing tokens)")));
                }
                let mut name = String::new();
                for x in &rest[..end] {
                    name.push_str(&show_tok(x, self.interner));
                }
                Ok((name, false))
            }
            _ => Err(err(format!("{loc}: malformed #include (expected \"...\" or <...>)"))),
        }
    }

    /// Parse an expanded (non-verbatim) header the same way.
    fn parse_header_tokens(&self, flat: &[Tok], loc: &str) -> Result<(String, bool), PpError> {
        self.parse_header_name(flat, loc)
    }

    /// Read, lex, group, and expand an included file with an enter
    /// marker. Returns whether anything was emitted (a `#pragma once`
    /// skip emits nothing). Shares the macro table and paint set.
    fn include_file(&mut self, path: &std::path::Path, found: crate::FoundKind, site_line: u64) -> Result<bool, PpError> {
        if self.files.depth >= 32 {
            return Err(err(format!("{}: #include nested too deeply", path.display())));
        }
        let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        if self.files.once.contains(&canon) {
            return Ok(false);
        }
        let text = std::fs::read_to_string(path).map_err(|_| err(format!("{}: cannot read '{}'", self.display(), path.display())))?;
        let display = self.interner.get_or_intern(path.display().to_string());
        if self.files.opts.linemarkers {
            // clang `FileChanged(EnterFile)`: skip ahead to the `#include`
            // line (newlines when close, a marker otherwise), then name
            // the new file and jump to its line 1.
            self.move_to_line(site_line, false);
            self.write_line_info_flag(1, &path.display().to_string(), 1);
            self.r_cur_line = 1;
        }
        self.mark_n = 1;
        self.mark_emit = 0;
        // Save the file frame; the child shares macros and paint.
        // Name the included file for diagnostics raised while lexing,
        // grouping, or expanding it.
        let save_display = std::mem::replace(&mut self.file_display, display);
        let save_path = std::mem::replace(&mut self.file_path, path.to_path_buf());
        let save_found = std::mem::replace(&mut self.file_found, found);
        let save_base = self.line_base;
        let save_override = self.display_override.take();
        self.line_base = 0;
        self.files.depth += 1;
        let r = self.run_tokens(&text, display);
        self.files.depth -= 1;
        self.file_display = save_display;
        self.file_path = save_path;
        self.file_found = save_found;
        self.line_base = save_base;
        self.display_override = save_override;
        r.map(|()| true)
    }

    /// Lex + group + expand a source text in the current file frame,
    /// stopping where the prepended trees run out (the parent stream
    /// resumes after).
    fn run_tokens(&mut self, text: &str, display: SymbolU32) -> Result<(), PpError> {
        use glcc_frontend::lexer::Lexer;
        let mut lexer = Lexer::new(text, self.interner);
        let toks: Vec<Tok> = lexer.lex_pp_all().into_iter().map(|t| Tok::from_pp(&t, display, false, false)).collect();
        let disp = self.display();
        let trees = crate::tree::collect_trees(toks, &disp, &mut *self.interner)?;
        let stop_at = self.trees.remaining();
        self.trees.prepend_extend(trees.into_iter());
        while self.trees.remaining() > stop_at {
            match self.trees.next() {
                None => break,
                Some(tt) => self.expand_tree(tt)?,
            }
        }
        Ok(())
    }
}

/// Trim leading/trailing whitespace tokens (for call arguments).
fn trim_tokens(tokens: &[Tok]) -> &[Tok] {
    let mut start = 0;
    let mut end = tokens.len();
    while start < end && matches!(tokens[start].tok, PpToken::Ws(_)) {
        start += 1;
    }
    while end > start && matches!(tokens[end - 1].tok, PpToken::Ws(_)) {
        end -= 1;
    }
    &tokens[start..end]
}

/// One `#if` attempt through the real frontend: parse a single
/// expression, check, and constant-fold. `None` means "not foldable
/// here" (callers take `#else`); panics are caught.
fn eval_frontend_text(text: &str, target: Option<glcc_frontend::target::Target>) -> bool {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        use glcc_frontend::eval::{eval_const, truthy};
        use glcc_frontend::lexer::{Interner, Lexer};
        use glcc_frontend::parse::Parser;
        use glcc_frontend::sema::Sema;
        let target = target.unwrap_or_else(glcc_frontend::target::Target::native);
        let mut interner = Interner::new();
        let mut sema = Sema::with_target(target);
        let tokens = Lexer::new(text, &mut interner).lex_all();
        let mut parser = Parser::new(text, tokens, &mut sema, &mut interner);
        let parsed = parser.parse_expression();
        let checked = sema.check_expr(&parsed, &interner).ok()?;
        eval_const(&checked, &sema, &interner).map(|v| truthy(&v))
    }));
    outcome.unwrap_or(None).unwrap_or(false)
}

/// Body of a string literal with its quotes stripped: `\"` becomes `"`
/// and `\\` becomes `\` (C11 6.10.6 destringizing); other escapes are
/// kept verbatim.
fn unescape_string_body(raw: &str) -> String {
    let body = raw.strip_prefix('"').and_then(|s| s.strip_suffix('"')).unwrap_or(raw);
    let mut out = String::new();
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn render_tok_text(t: &Tok, interner: &Interner) -> String {
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

fn quoted_display(display: &str) -> String {
    let mut s = String::with_capacity(display.len() + 2);
    s.push('"');
    for c in display.chars() {
        if c == '\\' {
            s.push_str("\\\\");
        } else if c == '"' {
            s.push_str("\\\"");
        } else {
            s.push(c);
        }
    }
    s.push('"');
    s
}

fn unescape_header(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            if let Some(e) = it.next() {
                out.push(e);
            }
        } else {
            out.push(c);
        }
    }
    out
}
