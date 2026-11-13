//! GAS/AT&T syntax frontend (the `.s` dialect GNU `as` and glcc's own
//! generated asm use), translated onto the shared statement model.
//! x86_64 uses AT&T operand syntax (`%reg`/`$imm`/`disp(base,index,
//! scale)`); RISC-V and AArch64 have their own operand syntaxes with the
//! same lexer and directive handling.

use crate::arch::Arch;
use crate::assemble::{self, AsmError};
use crate::expr::Expr;
use crate::insn::{DataItem, Operand, Stmt, X86Insn, eaf, op};
use crate::parse::{self, ParseError, Tok, Toks};
use crate::regs;
use std::collections::HashMap;

/// Lexing quirks per architecture: comment/immediate/register markers.
#[derive(Clone, Copy, PartialEq)]
pub enum LexMode {
    /// x86: `#` and `;` comments, `$` immediates, `%` registers.
    X86,
    /// RISC-V: `#` comments, `%hi(...)`-style modifiers (lexed as
    /// register tokens by the shared lexer).
    Riscv,
    /// AArch64: `//` comments, `#` immediates.
    A64,
}

/// Assembles GAS-syntax source (any arch) into an ELF64 object.
pub fn assemble_to_elf(src: &str, arch: Arch) -> Result<Vec<u8>, Vec<AsmError>> {
    let stmts = parse_module(src, arch).map_err(|errs| errs.into_iter().map(|e| AsmError { line: e.line, msg: e.msg }).collect::<Vec<_>>())?;
    let module = assemble::assemble(&stmts, arch)?;
    crate::elf::write_module_elf(&module, arch).map_err(|e| vec![AsmError { line: 0, msg: e }])
}

pub fn parse_module(src: &str, arch: Arch) -> Result<Vec<Stmt>, Vec<ParseError>> {
    let src = strip_block_comments(src);
    let raw: Vec<&str> = src.lines().collect();
    let mode = match arch {
        Arch::X86_64 => LexMode::X86,
        Arch::Riscv32 | Arch::Riscv64 => LexMode::Riscv,
        Arch::Aarch64 => LexMode::A64,
    };
    // Macros, conditionals, and `.rept`/`.irp` expand textually before
    // any statement parsing: bodies repeat per invocation and `.if`
    // sees the substituted arguments (GAS expands before assembling).
    let mut pp = Preproc::new(mode);
    let expanded = match pp.run(&raw) {
        Ok(lines) => lines,
        Err(errors) => return Err(errors),
    };
    #[allow(unused_mut)]
    let mut expanded = if std::env::var_os("GLCC_NO_PREPROC").is_some() {
        raw.iter().enumerate().map(|(i, s)| (s.to_string(), (i + 1) as u32)).collect::<Vec<_>>()
    } else {
        expanded
    };
    let texts: Vec<&str> = expanded.iter().map(|(text, _)| text.as_str()).collect();
    let numlabels = NumLabels::scan(&texts);
    let mut stmts = Vec::new();
    let mut errors = Vec::new();
    for (lineno, (text, src_line)) in expanded.iter().enumerate() {
        // GAS `;` separates statements on a line.
        for part in split_statements(text) {
            match parse_line(&part, *src_line, lineno, &numlabels, arch) {
                Ok(ss) => stmts.extend(ss),
                Err(msg) => errors.push(ParseError { line: *src_line, msg }),
            }
        }
    }
    if errors.is_empty() {
        Ok(stmts)
    } else {
        if std::env::var_os("GLCC_DUMP_FAILED_ASM").is_some() {
            let _ = std::fs::write("/tmp/glcc-pp-out.s", expanded.iter().map(|(text, line)| format!("{text} !line {line}\n")).collect::<String>());
        }
        Err(errors)
    }
}

// ---------------------------------------------------------------------
// Macro + conditional preprocessor (.macro/.rept/.irp/.if)
// ---------------------------------------------------------------------

/// One `.macro` definition: formal parameters (with optional defaults,
/// raw text) and the raw body lines.
#[derive(Clone)]
struct GasMacro {
    params: Vec<(String, Option<String>)>,
    body: Vec<String>,
}

/// An expanded line: text plus the 1-based source line it came from
/// (for diagnostics and numeric-label reporting).
type PPLine = (String, u32);

enum PPStop {
    /// `.exitm`: stop the enclosing macro body / `.rept` loop.
    Exit,
    Fatal(u32, String),
}

type PPResult<T> = Result<T, PPStop>;

fn fatal(line: u32, msg: impl Into<String>) -> PPStop {
    PPStop::Fatal(line, msg.into())
}

struct Preproc {
    mode: LexMode,
    macros: HashMap<String, GasMacro>,
    /// Values of `.set`/`sym =` definitions, for `.if` evaluation.
    syms: HashMap<String, i64>,
    counter: u64,
    /// A bare instruction prefix (`lock`/`rep` alone before its
    /// instruction — `lock; addl ...` or `lock` on its own line).
    pending_prefix: Option<String>,
}

impl Preproc {
    fn new(mode: LexMode) -> Preproc {
        Preproc {
            mode,
            macros: HashMap::new(),
            syms: HashMap::new(),
            counter: 0,
            pending_prefix: None,
        }
    }

    fn run(&mut self, raw: &[&str]) -> Result<Vec<PPLine>, Vec<ParseError>> {
        // Join backslash-continuations first: a continued line behaves
        // as one logical line (GAS), and macro machinery expects it.
        let mut joined: Vec<(String, u32)> = Vec::with_capacity(raw.len());
        let mut pending: Option<(String, u32)> = None;
        for (i, text) in raw.iter().enumerate() {
            let line = (i + 1) as u32;
            let text = text.trim_end();
            let (base, base_line) = match pending.take() {
                Some((b, l)) => (b, l),
                None => (String::new(), line),
            };
            if let Some(stripped) = text.strip_suffix('\\') {
                pending = Some((format!("{base}{} ", stripped), base_line));
            } else {
                joined.push((format!("{base}{text}"), base_line));
            }
        }
        if let Some((b, l)) = pending {
            joined.push((b, l));
        }
        let lines = joined;
        match self.process(lines, 0) {
            Ok(out) => Ok(out),
            Err(PPStop::Exit) => Ok(Vec::new()),
            Err(PPStop::Fatal(line, msg)) => {
                if std::env::var_os("GLCC_DUMP_FAILED_ASM").is_some()
                    && let Ok(existing) = std::fs::read_to_string("/tmp/glcc-failed.s")
                {
                    let _ = std::fs::write("/tmp/glcc-pp-out.s", format!("{msg}
--- at line {line} of:
{existing}"));
                }
                Err(vec![ParseError { line, msg }])
            }
        }
    }

    fn process(&mut self, lines: Vec<PPLine>, depth: usize) -> PPResult<Vec<PPLine>> {
        if depth > 64 {
            let (_, line) = lines.first().cloned().unwrap_or_default();
            return Err(fatal(line, "macro/conditional nesting too deep"));
        }
        let mut out = Vec::new();
        // Conditional frames: (active, ever-taken). `active` already
        // accounts for outer frames, so a false outer frame pins
        // everything inside off.
        let mut conds: Vec<(bool, bool)> = Vec::new();
        let mut i = 0;
        while i < lines.len() {
            let (text, line) = lines[i].clone();
            let code = strip_line_comment(&text, self.mode);
            let active = conds.iter().all(|(a, _)| *a);
            let word = first_word(&code);
            if !active {
                match word.as_deref() {
                    Some("if" | "ifdef" | "ifndef" | "ifb" | "ifnb" | "ifc" | "ifnc") => conds.push((false, true)),
                    Some("else") => {
                        let parent_active = conds[..conds.len().saturating_sub(1)].iter().all(|(a, _)| *a);
                        if let Some(frame) = conds.last_mut()
                            && parent_active
                            && !frame.1 {
                                *frame = (true, true);
                            }
                    }
                    Some("elseif" | "elif") => {
                        let parent_active = conds[..conds.len().saturating_sub(1)].iter().all(|(a, _)| *a);
                        if let Some(frame) = conds.last_mut()
                            && parent_active
                            && !frame.1 {
                                // Inside a false outer frame the
                                // condition cannot be evaluated here;
                                // stay eligible, not taken.
                                *frame = (parent_active && !frame.1, frame.1);
                            }
                    }
                    Some("endif") => {
                        conds.pop();
                    }
                    // Skip whole definitions and blocks as units so a
                    // stray `.endm`/`.endr` inside them cannot unbalance
                    // the nesting.
                    Some("macro") => {
                        i = self.skip_block(&lines, i, ".macro", ".endm");
                        continue;
                    }
                    Some("rept" | "irp" | "irpc") => {
                        i = self.skip_block(&lines, i, "", ".endr");
                        continue;
                    }
                    _ => {}
                }
                i += 1;
                continue;
            }
            match word.as_deref() {
                Some(name) if matches!(name, "if" | "ifdef" | "ifndef" | "ifb" | "ifnb" | "ifc" | "ifnc") => {
                    let taken = self.eval_cond(&name, code.trim()[name.len() + 1..].trim(), line)?;
                    conds.push((taken, taken));
                    i += 1;
                }
                Some("else") => {
                    if let Some(frame) = conds.last_mut() {
                        // Outer frames already fold into `active` above:
                        // a frame is only reached while every outer one
                        // is on, so reactivation depends on `taken`.
                        let taken = frame.1;
                        *frame = (!taken, true);
                    }
                    i += 1;
                }
                Some(name @ ("elseif" | "elif")) => {
                    if let Some(frame) = conds.last_mut() {
                        if frame.1 {
                            *frame = (false, true);
                        } else {
                            let taken = self.eval_cond("if", code.trim()[name.len() + 1..].trim(), line)?;
                            *frame = (taken, taken);
                        }
                    }
                    i += 1;
                }
                Some("endif") => {
                    conds.pop();
                    i += 1;
                }
                Some("exitm") => return Err(PPStop::Exit),
                Some("macro") => {
                    let (name, params, body, end) = self.collect_macro(&lines, i, line)?;
                    self.macros.insert(name, GasMacro { params, body });
                    i = end;
                }
                Some("endm") => {
                    // A stray `.endm` outside a definition: ignore, like
                    // an unmatched directive would at least not crash.
                    i += 1;
                }
                Some("rept") => {
                    let count = self.eval_expr(code.trim()[".rept".len()..].trim(), line)?;
                    let (body, end) = collect_block(&lines, i + 1, "", ".endr", self.mode)?;
                    let mut reps = Vec::new();
                    for _ in 0..count.max(0) {
                        let rep: Vec<PPLine> = body.iter().map(|(text, _)| (text.clone(), line)).collect();
                        match self.process(rep, depth + 1) {
                            Ok(lines) => reps.extend(lines),
                            Err(PPStop::Exit) => break,
                            Err(e) => return Err(e),
                        }
                    }
                    out.extend(reps);
                    i = end;
                }
                Some("irp" | "irpc") => {
                    let header = code.trim()[word.unwrap().len() + 1..].trim();
                    let (sym, items) = self.parse_irp(header, line)?;
                    let (body, end) = collect_block(&lines, i + 1, "", ".endr", self.mode)?;
                    let mut reps = Vec::new();
                    for item in items {
                        let rep: Vec<PPLine> = body
                            .iter()
                            .map(|(text, _)| (substitute(text, &[(sym.clone(), item.clone())], self.counter), line))
                            .collect();
                        match self.process(rep, depth + 1) {
                            Ok(lines) => reps.extend(lines),
                            Err(PPStop::Exit) => break,
                            Err(e) => return Err(e),
                        }
                    }
                    out.extend(reps);
                    i = end;
                }
                Some("endr") => {
                    i += 1;
                }
                _ => {
                    // GAS splits statements on top-level `;` before
                    // anything else: a macro call followed by more
                    // statements must not swallow them as arguments.
                    let stmts = split_statements(&code);
                    if stmts.len() > 1 {
                        let rest: Vec<PPLine> = stmts[1..].iter().map(|s| (s.to_string(), line)).collect();
                        let mut first_stmts: Vec<PPLine> = vec![(stmts[0].to_string(), line)];
                        first_stmts.extend(rest);
                        let expanded = self.process(first_stmts, depth)?;
                        out.extend(expanded);
                        i += 1;
                        continue;
                    }
                    let (labels, rest) = split_leading_labels(&code);
                    let rest = rest.trim_start();
                    // A bare prefix statement (`lock; addl` splits into
                    // `lock` + `addl ...`) defers to the next
                    // instruction, even across directives.
                    if !rest.is_empty() && is_bare_prefix(rest) {
                        self.pending_prefix = Some(rest.to_string());
                        i += 1;
                        continue;
                    }
                    // The invocation name ends at whitespace or an
                    // argument-opening paren (`MACRO(a, b)`).
                    let first_len = rest
                        .char_indices()
                        .take_while(|(_, c)| c.is_ascii_alphanumeric() || *c == '_' || *c == '$' || *c == '.')
                        .map(|(k, _)| k)
                        .last()
                        .map(|k| k + 1)
                        .unwrap_or(0);
                    let first = &rest[..first_len];
                    if let Some(def) = self.macros.get(first).cloned() {
                        for label in labels {
                            out.push((format!("{label}:"), line));
                        }
                        // `MACRO(a, b)` parenthesizes the arguments;
                        // `MACRO a, b` leaves them bare.
                        let tail = rest[first_len..].trim();
                        let arg_text = match paren_group(tail) {
                            Some((inside, _)) => inside.to_string(),
                            None => tail.to_string(),
                        };
                        let expanded = self.expand(&def, &arg_text, line, depth)?;
                        out.extend(expanded);
                        i += 1;
                        continue;
                    }
                    if std::env::var_os("GLCC_AS_TRACE").is_some() {
                        eprintln!("AS[{line}]: {code}");
                    }
                    if let Some((name, value)) = assignment(&code) {
                        self.record_set(name, value, line)?;
                    }
                    // Attach a deferred prefix to this statement.
                    let text = if let Some(prefix) = self.pending_prefix.take()
                        && !rest.starts_with('.')
                        && self.macros.get(rest.split([' ', '\t', '(']).next().unwrap_or("")).is_none()
                    {
                        format!("{prefix} {text}")
                    } else {
                        text
                    };
                    if labels.is_empty() {
                        // Pass the untouched line through: the statement
                        // parser handles its own comments and labels.
                        out.push((text, line));
                    } else {
                        // Labels were split off for macro matching; emit
                        // them ahead of the (label-less) remainder.
                        for label in labels {
                            out.push((format!("{label}:"), line));
                        }
                        if !rest.is_empty() {
                            out.push((rest.to_string(), line));
                        }
                    }
                    i += 1;
                }
            }
        }
        Ok(out)
    }

    /// Skip to the line after the matching closer, counting nested
    /// openers. `open` is empty when any `.rept`/`.irp`/`.irpc` opens.
    fn skip_block(&self, lines: &[PPLine], start: usize, open: &str, close: &str) -> usize {
        let mut depth = 1;
        let mut i = start + 1;
        while i < lines.len() {
            let word = first_word(&strip_line_comment(&lines[i].0, self.mode)).unwrap_or_default();
            let is_open = if open.is_empty() {
                matches!(word.as_str(), "rept" | "irp" | "irpc")
            } else {
                word == open.trim_start_matches('.')
            };
            if is_open {
                depth += 1;
            } else if word == close.trim_start_matches('.') {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            i += 1;
        }
        lines.len()
    }

    /// Collect a `.macro` definition: parameters from the header, body
    /// through the matching `.endm`.
    fn collect_macro(&mut self, lines: &[PPLine], start: usize, line: u32) -> PPResult<(String, Vec<(String, Option<String>)>, Vec<String>, usize)> {
        let header = strip_line_comment(&lines[start].0, self.mode);
        let header = header.trim()[".macro".len()..].trim();
        let mut parts = header.splitn(2, [' ', '\t']);
        let name = parts.next().unwrap_or("").trim().to_string();
        if name.is_empty() {
            return Err(fatal(line, "`.macro' needs a name"));
        }
        let mut params = Vec::new();
        if let Some(rest) = parts.next() {
            for param in split_args(rest) {
                if param.is_empty() {
                    continue;
                }
                // `:req`/`:vararg`/`:green` qualifiers qualify the name
                // (`type:req` binds `	ype`); split at the colon.
                let param = param.split(':').next().unwrap_or(&param).trim().to_string();
                match param.split_once('=') {
                    Some((name, default)) => params.push((name.trim().to_string(), Some(default.trim().to_string()))),
                    None => params.push((param, None)),
                }
            }
        }
        let (body, end) = collect_block(lines, start + 1, ".macro", ".endm", self.mode)?;
        let body: Vec<String> = body.into_iter().map(|(text, _)| text).collect();
        Ok((name, params, body, end))
    }

    fn expand(&mut self, def: &GasMacro, arg_text: &str, line: u32, depth: usize) -> PPResult<Vec<PPLine>> {
        self.counter += 1;
        // GNU as strips one layer of enclosing double quotes from
        // macro arguments (`ALTERNATIVE "jmp x", ""` substitutes
        // unquoted); directive arguments keep their quotes.
        let args: Vec<String> = split_args(arg_text).into_iter().map(|a| unquote(&a)).collect();
        // Bind keyword arguments first, then positionals to the rest.
        let mut values: Vec<Option<String>> = vec![None; def.params.len()];
        let mut positional = Vec::new();
        for arg in &args {
            if let Some((name, value)) = arg.split_once('=') {
                if let Some(slot) = def.params.iter().position(|(p, _)| p == name.trim()) {
                    values[slot] = Some(value.trim().to_string());
                    continue;
                }
            }
            positional.push(arg.clone());
        }
        for (slot, arg) in positional.into_iter().enumerate() {
            if slot < values.len() {
                values[slot] = Some(arg);
            } else {
                return Err(fatal(line, format!("too many macro arguments ({} given, {} expected)", args.len(), def.params.len())));
            }
        }
        let bindings: Vec<(String, String)> = def
            .params
            .iter()
            .zip(values)
            .filter_map(|((name, default), value)| Some((name.clone(), value.or_else(|| default.clone())?)))
            .collect();
        let body: Vec<PPLine> = def
            .body
            .iter()
            .map(|text| (substitute(text, &bindings, self.counter), line))
            .collect();
        self.process(body, depth + 1)
    }

    fn parse_irp(&self, header: &str, line: u32) -> PPResult<(String, Vec<String>)> {
        let mut parts = split_args(header).into_iter();
        let sym = parts.next().unwrap_or_default();
        if sym.is_empty() {
            return Err(fatal(line, "`.irp' needs a symbol name"));
        }
        Ok((sym, parts.collect()))
    }

    /// Evaluate a `.if`-family condition. `.ifc`/`.ifnc` compare two
    /// (unquoted) strings; everything else is a numeric expression.
    fn eval_cond(&mut self, name: &str, rest: &str, line: u32) -> PPResult<bool> {
        let result = match name {
            "ifc" => {
                let mut parts = split_args(rest).into_iter();
                let (a, b) = (parts.next().unwrap_or_default(), parts.next().unwrap_or_default());
                unquote(&a) == unquote(&b)
            }
            "ifnc" => {
                let mut parts = split_args(rest).into_iter();
                let (a, b) = (parts.next().unwrap_or_default(), parts.next().unwrap_or_default());
                unquote(&a) != unquote(&b)
            }
            "ifdef" => {
                let sym = rest.trim();
                !sym.is_empty() && self.syms.contains_key(sym)
            }
            "ifndef" => {
                let sym = rest.trim();
                sym.is_empty() || !self.syms.contains_key(sym)
            }
            "ifb" => split_args(rest).into_iter().next().map(|a| a.trim().is_empty()).unwrap_or(true),
            "ifnb" => !split_args(rest).into_iter().next().map(|a| a.trim().is_empty()).unwrap_or(true),
            _ => self.eval_expr(rest, line)? != 0,
        };
        Ok(result)
    }

    /// Evaluate a constant expression over the preprocessor's own
    /// `.set` symbols (register names count as their GAS numbers).
    fn eval_expr(&self, text: &str, line: u32) -> PPResult<i64> {
        if text.is_empty() {
            return Err(fatal(line, "expected an expression"));
        }
        let g = lex_line(text, self.mode).map_err(|msg| fatal(line, msg))?;
        let mut t = GToks { g: &g, pos: 0 };
        let expr = t.expr(line as usize, &NumLabels::empty()).map_err(|msg| fatal(line, msg))?;
        let syms = PPSyms(&self.syms);
        let ctx = crate::expr::EvalCtx {
            cur_sect: 0,
            cur_off: 0,
            syms: &syms,
            optimistic: false,
            unknown_hit: None,
        };
        match expr.eval(&ctx).map_err(|msg| fatal(line, msg))? {
            crate::expr::RValue::Const(v) => Ok(v),
            _ => Err(fatal(line, "conditional expression is not constant")),
        }
    }

    fn record_set(&mut self, name: &str, value: &str, line: u32) -> PPResult<()> {
        if let Ok(v) = self.eval_expr(value, line) {
            self.syms.insert(name.to_string(), v);
        }
        Ok(())
    }
}

/// Symbol table adapter for `.if` evaluation: the preprocessor's
/// `.set` values, plus x86 register names at their GAS numbers (`.if
/// \base == %rsp` compares register codes).
struct PPSyms<'a>(&'a HashMap<String, i64>);

impl crate::expr::SymLookup for PPSyms<'_> {
    fn lookup(&self, name: &str) -> Option<crate::expr::SymbolVal> {
        if let Some(v) = self.0.get(name) {
            return Some(crate::expr::SymbolVal {
                index: 0,
                defined: true,
                seen: true,
                extern_: false,
                sect: None,
                value: *v,
            });
        }
        let reg = name.strip_prefix('%')?;
        Some(crate::expr::SymbolVal {
            index: 0,
            defined: true,
            seen: true,
            extern_: false,
            sect: None,
            value: gas_reg_number(reg)?,
        })
    }
}

/// The numbers GAS compares registers by (`%rsp == 4` in a macro
/// argument check).
fn gas_reg_number(name: &str) -> Option<i64> {
    let name = name.strip_suffix(['d', 'w', 'b']).filter(|_| name.len() > 3).unwrap_or(name);
    let n = match name {
        "rax" => 0,
        "rcx" => 1,
        "rdx" => 2,
        "rbx" => 3,
        "rsp" => 4,
        "rbp" => 5,
        "rsi" => 6,
        "rdi" => 7,
        other => other.strip_prefix('r')?.parse::<i64>().ok()?,
    };
    Some(n)
}

/// Remove a line's `;` (and `#` in x86 mode) comment tail, keeping
/// quoted strings intact byte-for-byte.
fn strip_line_comment(line: &str, mode: LexMode) -> String {
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'"' || c == b'\'' {
            let quote = c;
            let start = i;
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i += 2;
                    continue;
                }
                if bytes[i] == quote {
                    i += 1;
                    break;
                }
                i += 1;
            }
            out.push_str(&line[start..i.min(line.len())]);
            continue;
        }
        if c == b';' {
            break;
        }
        if c == b'#' && mode != LexMode::A64 {
            break;
        }
        if mode == LexMode::A64 && c == b'/' && bytes.get(i + 1) == Some(&b'/') {
            break;
        }
        let ch = line[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// The first directive word of a line (without the leading dot), when
/// the line opens with one.
fn first_word(code: &str) -> Option<String> {
    let trimmed = code.trim_start();
    if !trimmed.starts_with('.') {
        return None;
    }
    let end = trimmed.find(|c: char| c.is_whitespace()).unwrap_or(trimmed.len());
    Some(trimmed[1..end].trim_end().to_string())
}

/// Split macro arguments on commas and spaces outside quotes and
/// bracketing punctuation (GAS separates macro arguments on either).
fn split_args(text: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut cur = String::new();
    let mut depth = 0u32;
    let mut in_str = false;
    let mut quote = ' ';
    for c in text.chars() {
        if in_str {
            cur.push(c);
            if c == quote {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' | '\'' => {
                in_str = true;
                quote = c;
                cur.push(c);
            }
            '(' | '[' | '<' => {
                depth += 1;
                cur.push(c);
            }
            ')' | ']' | '>' => {
                depth = depth.saturating_sub(1);
                cur.push(c);
            }
            ',' if depth == 0 => {
                args.push(cur.trim().to_string());
                cur = String::new();
            }
            c if depth == 0 && (c == ' ' || c == '\t') => {
                if !cur.trim().is_empty() {
                    args.push(cur.trim().to_string());
                    cur = String::new();
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        args.push(cur.trim().to_string());
    }
    args
}

/// A parenthesized group at the start of `text`: the contents and the
/// offset just past the closing paren.
fn paren_group(text: &str) -> Option<(&str, usize)> {
    if !text.starts_with('(') {
        return None;
    }
    let bytes = text.as_bytes();
    let mut depth = 0u32;
    let mut i = 0;
    let mut in_str = false;
    let mut quote = ' ';
    while i < bytes.len() {
        let c = bytes[i] as char;
        if in_str {
            if c == quote {
                in_str = false;
            }
        } else {
            match c {
                '"' | '\'' => {
                    in_str = true;
                    quote = c;
                }
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some((&text[1..i], i + 1));
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    None
}

/// Strip one layer of surrounding quotes (`.ifc` operands).
fn unquote(text: &str) -> String {
    let text = text.trim();
    if text.len() >= 2 && (text.starts_with('"') && text.ends_with('"') || text.starts_with('\'') && text.ends_with('\'')) {
        text[1..text.len() - 1].to_string()
    } else {
        text.to_string()
    }
}

/// Expand `\param`, `\@` (invocation counter), and `\()` in a macro
/// body line. Longest parameter name wins, and a name must end at a
/// non-identifier boundary.
fn substitute(text: &str, bindings: &[(String, String)], counter: u64) -> String {
    if !text.contains('\\') {
        return text.to_string();
    }
    let mut names: Vec<&(String, String)> = bindings.iter().collect();
    names.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'\\' {
            out.push(text[i..].chars().next().unwrap());
            i += text[i..].chars().next().unwrap().len_utf8();
            continue;
        }
        if i + 2 < bytes.len() && bytes[i + 1] == b'(' && bytes[i + 2] == b')' {
            i += 3;
            continue;
        }
        if i + 1 < bytes.len() && bytes[i + 1] == b'@' {
            out.push_str(&counter.to_string());
            i += 2;
            continue;
        }
        let rest = &text[i + 1..];
        let name_len = rest
            .char_indices()
            .take_while(|(_, c)| c.is_ascii_alphanumeric() || *c == '_')
            .map(|(k, _)| k)
            .last()
            .map(|k| k + 1)
            .unwrap_or(0);
        let candidate = &rest[..name_len];
        if let Some((_, value)) = names.iter().find(|(name, _)| name.as_str() == candidate) {
            out.push_str(value);
            i += 1 + name_len;
        } else {
            out.push('\\');
            i += 1;
        }
    }
    out
}

/// Split leading `label:` pairs off a line, textually (idents and
/// numeric labels; `%gs:`-style prefixes never match).
fn split_leading_labels(code: &str) -> (Vec<String>, &str) {
    let mut labels = Vec::new();
    let mut rest = code;
    loop {
        let trimmed = rest.trim_start();
        let end = trimmed
            .char_indices()
            .take_while(|(_, c)| c.is_ascii_alphanumeric() || *c == '_' || *c == '.' || *c == '$')
            .map(|(k, _)| k)
            .last()
            .map(|k| k + 1)
            .unwrap_or(0);
        if end == 0 {
            break;
        }
        let name = &trimmed[..end];
        if !name.bytes().all(|c| c.is_ascii_digit()) && !name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_' || c == '.') {
            break;
        }
        let after = trimmed[end..].trim_start();
        if !after.starts_with(':') {
            break;
        }
        labels.push(name.to_string());
        rest = &after[1..];
    }
    (labels, rest)
}

/// Whether the statement is a lone instruction prefix.
fn is_bare_prefix(stmt: &str) -> bool {
    matches!(stmt.trim_end_matches([';', ' ', '\t']), "lock" | "rep" | "repe" | "repz" | "repne" | "repnz" | "bnd" | "xacquire" | "xrelease")
}

/// `sym = expr` assignment (the `.set` alias), recognized textually
/// after label stripping.
fn assignment(code: &str) -> Option<(&str, &str)> {
    let (labels, rest) = split_leading_labels(code);
    let _ = labels;
    let rest = rest.trim_start();
    let end = rest
        .char_indices()
        .take_while(|(_, c)| c.is_ascii_alphanumeric() || *c == '_' || *c == '.' || *c == '$')
        .map(|(k, _)| k)
        .last()
        .map(|k| k + 1)
        .unwrap_or(0);
    if end == 0 {
        return None;
    }
    let name = &rest[..end];
    let after = rest[end..].trim_start();
    let value = after.strip_prefix('=')?.trim_start();
    if value.is_empty() {
        return None;
    }
    Some((name, value))
}

/// Collect the lines between `open`'s matching `close`, counting
/// nested opens. `open` empty means any `.rept`/`.irp`/`.irpc`.
fn collect_block(lines: &[PPLine], start: usize, open: &str, close: &str, mode: LexMode) -> PPResult<(Vec<PPLine>, usize)> {
    let mut depth = 1u32;
    let mut body = Vec::new();
    let mut i = start;
    while i < lines.len() {
        let word = first_word(&strip_line_comment(&lines[i].0, mode)).unwrap_or_default();
        let is_open = if open.is_empty() {
            matches!(word.as_str(), "rept" | "irp" | "irpc")
        } else {
            word == open.trim_start_matches('.')
        };
        if is_open {
            depth += 1;
        } else if word == close.trim_start_matches('.') {
            depth -= 1;
            if depth == 0 {
                return Ok((body, i + 1));
            }
        }
        body.push(lines[i].clone());
        i += 1;
    }
    let (_, line) = lines.last().cloned().unwrap_or_default();
    Err(fatal(line, format!("missing `.{}`", close.trim_start_matches('.'))))
}

// ---------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------

/// GAS accepts C-style `/* ... */` block comments, which span lines and
/// so cannot be handled by the per-line lexer. Strip them before
/// splitting into lines, keeping newlines so line numbers survive.
/// Quoted strings are left untouched; `//` is only a comment in AArch64
/// mode (on x86 it is division) and stays with the line lexer.
fn strip_block_comments(src: &str) -> String {
    if !src.contains("/*") {
        return src.to_string();
    }
    let mut out = String::with_capacity(src.len());
    let mut chars = src.chars().peekable();
    #[derive(PartialEq)]
    enum S {
        Code,
        Block,
        Str(char),
    }
    let mut state = S::Code;
    while let Some(c) = chars.next() {
        match state {
            S::Code => {
                if c == '/' && chars.peek() == Some(&'*') {
                    chars.next();
                    state = S::Block;
                    out.push(' ');
                } else if c == '"' || c == '\'' {
                    state = S::Str(c);
                    out.push(c);
                } else {
                    out.push(c);
                }
            }
            S::Block => {
                if c == '*' && chars.peek() == Some(&'/') {
                    chars.next();
                    state = S::Code;
                } else if c == '\n' {
                    out.push('\n');
                }
            }
            S::Str(q) => {
                out.push(c);
                if c == '\\' && chars.peek().is_some() {
                    out.push(chars.next().unwrap());
                } else if c == q {
                    state = S::Code;
                }
            }
        }
    }
    out
}

#[derive(Clone, Debug, PartialEq)]
pub enum GTok {
    Ident(String),
    Int(i64),
    /// Numeric local label reference: `1b` / `1f`.
    NumRef(u32, bool),
    Str(Vec<u8>),
    Reg(String),
    Punct(&'static str),
    /// `$` immediate marker.
    Imm,
    Star,
}

fn lex_line(line: &str, mode: LexMode) -> Result<Vec<GTok>, String> {
    let b = line.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        match c {
            b';' => break, // statement separator (already split, but be safe)
            b'#' => match mode {
                LexMode::A64 => {
                    out.push(GTok::Imm);
                    i += 1;
                }
                _ => break, // comment
            },
            b'/' if mode == LexMode::A64 && b.get(i + 1) == Some(&b'/') => break,
            b'$' => {
                if mode != LexMode::X86 {
                    return Err(format!("unexpected character `$'"));
                }
                out.push(GTok::Imm);
                i += 1;
            }
            b'%' => {
                if mode == LexMode::A64 {
                    return Err(format!("unexpected character `%'"));
                }
                i += 1;
                let start = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                if start == i {
                    return Err("expected register name after `%'".into());
                }
                let mut name = line[start..i].to_ascii_lowercase();
                // %st(N) / %st — x87 stack registers.
                if name == "st" {
                    let save = i;
                    if i < b.len() && b[i] == b'(' {
                        let close = line[i..].find(')').ok_or("unclosed %st(")?;
                        let n = &line[i + 1..i + close];
                        name = format!("st{n}");
                        i += close + 1;
                    } else {
                        i = save;
                        name = "st0".into();
                    }
                }
                out.push(GTok::Reg(name));
            }
            b'"' | b'\'' => {
                let (bytes, len) = lex_string(line, i)?;
                out.push(GTok::Str(bytes));
                i += len;
            }
            b'0'..=b'9' => {
                let start = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'.') {
                    i += 1;
                }
                let text = &line[start..i];
                // `1b`/`1f` (and `42b`/`42f`) numeric local label
                // references; GNU as allows any number of digits.
                if let Some(rest) = text.strip_suffix(['b', 'f']) {
                    if !rest.is_empty() && rest.bytes().all(|c| c.is_ascii_digit()) {
                        if let Ok(digit) = rest.parse::<u32>() {
                            out.push(GTok::NumRef(digit, text.ends_with('f')));
                            continue;
                        }
                    }
                }
                let v = gas_parse_number(text).ok_or_else(|| format!("invalid number `{text}'"))?;
                out.push(GTok::Int(v));
            }
            b'(' => {
                out.push(GTok::Punct("("));
                i += 1;
            }
            b')' => {
                out.push(GTok::Punct(")"));
                i += 1;
            }
            b'[' | b']' if mode == LexMode::A64 => {
                out.push(GTok::Punct(if c == b'[' { "[" } else { "]" }));
                i += 1;
            }
            b',' => {
                out.push(GTok::Punct(","));
                i += 1;
            }
            b'*' => {
                out.push(GTok::Star);
                i += 1;
            }
            b'+' | b'-' | b'/' | b'|' | b'&' | b'^' | b'~' | b'!' | b'<' | b'>' | b':' | b'=' => {
                // Two-character operators (`==`, `!=`, `<=`, `>=`, `&&`,
                // `||`) matter for `.if`/`.skip` expressions.
                let two: Option<&'static str> = match c {
                    b'=' if b.get(i + 1) == Some(&b'=') => Some("=="),
                    b'!' if b.get(i + 1) == Some(&b'=') => Some("!="),
                    b'<' if b.get(i + 1) == Some(&b'=') => Some("<="),
                    b'>' if b.get(i + 1) == Some(&b'=') => Some(">="),
                    b'&' if b.get(i + 1) == Some(&b'&') => Some("&&"),
                    b'|' if b.get(i + 1) == Some(&b'|') => Some("||"),
                    _ => None,
                };
                if let Some(p) = two {
                    out.push(GTok::Punct(p));
                    i += 2;
                    continue;
                }
                if c == b'<' && i + 1 < b.len() && b[i + 1] == b'<' {
                    out.push(GTok::Punct("<<"));
                    i += 2;
                    continue;
                }
                if c == b'>' && i + 1 < b.len() && b[i + 1] == b'>' {
                    out.push(GTok::Punct(">>"));
                    i += 2;
                    continue;
                }
                if c == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
                    // GAS treats `//` as an operator; keep it simple.
                    out.push(GTok::Punct("/"));
                    i += 2;
                    continue;
                }
                let p: &'static str = match c {
                    b'+' => "+",
                    b'-' => "-",
                    b'/' => "/",
                    b'|' => "|",
                    b'&' => "&",
                    b'^' => "^",
                    b'~' => "~",
                    b'!' => "!",
                    b'<' => "<",
                    b'>' => ">",
                    b':' => ":",
                    b'=' => "=",
                    _ => unreachable!(),
                };
                out.push(GTok::Punct(p));
                i += 1;
            }
            _ if is_ident_start(c) || c == b'@' => {
                let start = i;
                i += 1;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'.') {
                    i += 1;
                }
                out.push(GTok::Ident(line[start..i].to_string()));
            }
            _ => return Err(format!("unexpected character `{}'", c as char)),
        }
    }
    Ok(out)
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_' || c == b'.'
}

fn gas_parse_number(text: &str) -> Option<i64> {
    let (radix, digits) = if let Some(h) = text.strip_prefix("0x").or(text.strip_prefix("0X")) {
        (16, h)
    } else if let Some(h) = text.strip_prefix("0b").or(text.strip_prefix("0B")) {
        (2, h)
    } else if let Some(h) = text.strip_prefix("0o").or(text.strip_prefix("0O")) {
        (8, h)
    } else if text.len() > 1 && text.starts_with('0') {
        (8, &text[1..])
    } else {
        (10, text)
    };
    // Wrap like GNU as: 0xfffffffffffff7ff is -2049.
    u64::from_str_radix(digits, radix).ok().map(|v| v as i64)
}

/// Lexes a quoted string starting at `at`, handling GAS backslash
/// escapes. Returns the bytes and total length consumed.
fn lex_string(line: &str, at: usize) -> Result<(Vec<u8>, usize), String> {
    let b = line.as_bytes();
    let quote = b[at];
    let mut i = at + 1;
    let mut out = Vec::new();
    while i < b.len() && b[i] != quote {
        if b[i] == b'\\' && i + 1 < b.len() {
            i += 1;
            let c = b[i];
            match c {
                b'n' => out.push(b'\n'),
                b't' => out.push(b'\t'),
                b'r' => out.push(b'\r'),
                b'0' => out.push(0),
                b'\\' => out.push(b'\\'),
                b'"' => out.push(b'"'),
                b'\'' => out.push(b'\''),
                b'a' => out.push(7),
                b'b' => out.push(8),
                b'f' => out.push(12),
                b'v' => out.push(11),
                b'x' => {
                    let mut v = 0u32;
                    let mut n = 0;
                    while n < 2 && i + 1 + n < b.len() && (b[i + 1 + n] as char).is_ascii_hexdigit() {
                        v = v * 16 + (b[i + 1 + n] as char).to_digit(16).unwrap();
                        n += 1;
                    }
                    if n == 0 {
                        return Err("invalid \\x escape".into());
                    }
                    out.push(v as u8);
                    i += n;
                }
                other => out.push(other),
            }
            i += 1;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    if i >= b.len() {
        return Err("unterminated string".into());
    }
    Ok((out, i - at + 1))
}

// ---------------------------------------------------------------------
// Numeric local labels (1:, 1b/1f)
// ---------------------------------------------------------------------

fn numeric_label_name(digit: u32, seq: usize) -> String {
    // GNU as's internal numeric-label names: instance counts are
    // 1-based per digit, and relocations against them use this name.
    format!(".L{digit}\u{2}{}", seq + 1)
}

/// Maps (line index, direction) to the numeric label's unique name.
pub struct NumLabels {
    /// Per line: names of `N:` definitions made on that line.
    defs: Vec<Vec<(u32, String)>>,
}

impl NumLabels {
    fn empty() -> NumLabels {
        NumLabels { defs: Vec::new() }
    }

    fn scan(lines: &[&str]) -> NumLabels {
        let mut defs = Vec::with_capacity(lines.len());
        for line in lines {
            let mut here = Vec::new();
            // Labels only occur before the statement; scan leading
            // `digits+:` groups (multi-digit numbers included).
            let mut rest = line.trim_start();
            loop {
                let digits_len = rest.bytes().take_while(|c| c.is_ascii_digit()).count();
                if digits_len == 0 || rest.as_bytes().get(digits_len) != Some(&b':') {
                    break;
                }
                let digit: u32 = rest[..digits_len].parse().unwrap();
                here.push((digit, String::new()));
                rest = rest[digits_len + 1..].trim_start();
            }
            defs.push(here);
        }
        // Assign unique names in document order.
        let mut seqs: Vec<(u32, usize)> = Vec::new();
        for here in &mut defs {
            for (digit, name) in here {
                let seq = match seqs.iter_mut().find(|(d, _)| d == digit) {
                    Some((_, seq)) => {
                        let value = *seq;
                        *seq += 1;
                        value
                    }
                    None => {
                        seqs.push((*digit, 1));
                        0
                    }
                };
                *name = numeric_label_name(*digit, seq);
            }
        }
        NumLabels { defs }
    }

    pub(crate) fn def_on_line(&self, line: usize, digit: u32) -> Option<&str> {
        self.defs.get(line)?.iter().find(|(d, _)| *d == digit).map(|(_, n)| n.as_str())
    }

    /// Most recent definition strictly before `line`.
    pub(crate) fn last_before(&self, line: usize, digit: u32) -> Option<String> {
        for (l, defs) in self.defs.iter().enumerate().rev() {
            if l >= line {
                continue;
            }
            if let Some((_, n)) = defs.iter().find(|(d, _)| *d == digit) {
                return Some(n.clone());
            }
        }
        None
    }

    /// First definition at or after `line`.
    pub(crate) fn next_at(&self, line: usize, digit: u32) -> Option<String> {
        for (_l, defs) in self.defs.iter().enumerate().skip(line) {
            if let Some((_, n)) = defs.iter().find(|(d, _)| *d == digit) {
                return Some(n.clone());
            }
        }
        None
    }
}

// ---------------------------------------------------------------------
// Statement parsing
// ---------------------------------------------------------------------

const IGNORED_DIRECTIVES: &[&str] = &[
    "file",
    "ident",
    "size",
    "type",
    "local",
    "extern",
    "end",
    "appfile",
    "appline",
    "line",
    "loc",
    "eject",
    "psize",
    "noformat",
    "nolist",
    "list",
    "title",
    "sbttl",
    "version",
    "warn",
    "err",
    "print",
    "pushsection_nop",
    "stabn",
    "stabs",
    "stabd",
];

/// `.word` is 2 bytes on x86 and 4 bytes on the fixed-width ISAs, like
/// GNU as.
fn data_directives(arch: Arch) -> &'static [(&'static str, u8)] {
    match arch {
        Arch::X86_64 => &[("byte", 1), ("half", 2), ("word", 2), ("short", 2), ("value", 2), ("2byte", 2), ("long", 4), ("int", 4), ("4byte", 4), ("quad", 8), ("8byte", 8)],
        // GNU as calls the 2-byte directive `.hword` on AArch64 and
        // `.half` on RISC-V.
        Arch::Aarch64 => &[("byte", 1), ("hword", 2), ("half", 2), ("short", 2), ("value", 2), ("word", 4), ("long", 4), ("int", 4), ("quad", 8), ("xword", 8)],
        _ => &[("byte", 1), ("half", 2), ("hword", 2), ("short", 2), ("value", 2), ("word", 4), ("long", 4), ("int", 4), ("quad", 8), ("xword", 8)],
    }
}

/// Splits on `;` outside string literals.
fn split_statements(line: &str) -> Vec<&str> {
    let b = line.as_bytes();
    let mut parts = Vec::new();
    let mut start = 0;
    let mut i = 0;
    // Parens group macro-call arguments: a `;` inside them (from a
    // substituted argument that itself carried statements) stays put.
    let mut depth = 0u32;
    while i < b.len() {
        match b[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth = depth.saturating_sub(1),
            b'"' | b'\'' => {
                i += 1;
                while i < b.len() && b[i] != b'"' && b[i] != b'\'' {
                    if b[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
            }
            b';' if depth == 0 => {
                parts.push(&line[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    parts.push(&line[start..]);
    parts
}

fn parse_line(line: &str, line_no: u32, lineno: usize, numlabels: &NumLabels, arch: Arch) -> Result<Vec<Stmt>, String> {
    let mode = match arch {
        Arch::X86_64 => LexMode::X86,
        Arch::Riscv32 | Arch::Riscv64 => LexMode::Riscv,
        Arch::Aarch64 => LexMode::A64,
    };
    let g = lex_line(line, mode)?;
    let mut i = 0;
    let mut out = Vec::new();

    // Labels: `name:` and numeric `N:`.
    loop {
        match g.get(i) {
            Some(GTok::Ident(name)) if g.get(i + 1) == Some(&GTok::Punct(":")) => {
                out.push(Stmt::Label(name.clone(), line_no));
                i += 2;
            }
            Some(GTok::Int(digit)) if *digit >= 0 && u32::try_from(*digit).is_ok() && g.get(i + 1) == Some(&GTok::Punct(":")) => {
                let name = numlabels.def_on_line(lineno, *digit as u32).ok_or("internal: numeric label not scanned")?.to_string();
                out.push(Stmt::Label(name, line_no));
                i += 2;
            }
            _ => break,
        }
    }

    if i >= g.len() {
        return Ok(out);
    }

    // `symbol = expression` assignment (the `.set` alias).
    if let (Some(GTok::Ident(name)), Some(GTok::Punct("="))) = (g.get(i), g.get(i + 1)) {
        let mut t = GToks { g: &g, pos: i + 2 };
        let value = t.expr(lineno, numlabels)?;
        out.push(Stmt::Set { name: name.clone(), value, line: line_no });
        return Ok(out);
    }

    match &g[i] {
        GTok::Ident(word) if word.starts_with('.') => {
            out.push(parse_directive(&g[i..], line_no, lineno, numlabels, arch)?);
        }
        _ => {
            out.push(crate::arch::parse_gas_insn(arch, &g[i..], line_no, lineno, numlabels)?);
        }
    }
    Ok(out)
}

fn directive_name(word: &str) -> &str {
    word.trim_start_matches('.')
}

fn parse_directive(g: &[GTok], line_no: u32, lineno: usize, numlabels: &NumLabels, arch: Arch) -> Result<Stmt, String> {
    let GTok::Ident(word) = &g[0] else {
        return Err("expected directive".into());
    };
    let name = directive_name(word);
    let mut t = GToks { g, pos: 1 };
    match name {
        "text" => Ok(section_stmt(".text", SectionKind::Text, None)),
        "data" => Ok(section_stmt(".data", SectionKind::Data, None)),
        "bss" => Ok(section_stmt(".bss", SectionKind::Bss, None)),
        "rodata" => Ok(section_stmt(".rodata", SectionKind::Rodata, None)),
        // .section and .pushsection share the spec grammar; pushsection
        // additionally saves the section active before it for
        // .popsection/.previous (assemble.rs keeps the stack).
        "section" => Ok(Stmt::Section(parse_section(&mut t)?)),
        "pushsection" => Ok(Stmt::PushSection(parse_section(&mut t)?)),
        "previous" => Ok(Stmt::Previous(line_no)),
        "popsection" => Ok(Stmt::PopSection(line_no)),
        "set" => {
            let Some(GTok::Ident(name)) = t.bump().cloned() else {
                return Err("expected symbol name after .set".into());
            };
            if !t.eat_punct(",") {
                return Err("expected `,' after .set symbol".into());
            }
            let value = t.expr(lineno, numlabels)?;
            Ok(Stmt::Set { name, value, line: line_no })
        }
        "globl" | "global" | "weak" | "extern" => {
            let mut names = Vec::new();
            while let Some(GTok::Ident(n)) = t.bump() {
                names.push(n.clone());
                if t.eat_punct(",") {
                    continue;
                }
                break;
            }
            if name == "weak" {
                Ok(Stmt::Weak(names, line_no))
            } else if name == "extern" {
                Ok(Stmt::Extern(names, line_no))
            } else {
                Ok(Stmt::Global(names, line_no))
            }
        }
        "comm" | "lcomm" => {
            let global = name == "comm";
            let sym = t
                .bump()
                .and_then(|x| match x {
                    GTok::Ident(n) => Some(n.clone()),
                    _ => None,
                })
                .ok_or("expected symbol for .comm")?;
            if !t.eat_punct(",") {
                return Err("expected `,' after .comm symbol".into());
            }
            let size = t.expr(lineno, numlabels)?;
            let align = if t.eat_punct(",") {
                match t.bump() {
                    Some(GTok::Int(v)) => Some((*v).max(1) as u64),
                    _ => return Err("expected alignment for .comm".into()),
                }
            } else {
                None
            };
            Ok(Stmt::Comm {
                name: sym,
                size,
                align,
                global,
                line: line_no,
            })
        }
        "align" => {
            // On x86 ELF, .align is in bytes. The argument is a full
            // expression (`.align (1 << 6)` shapes in kernel headers).
            let v = t.expr(lineno, numlabels)?;
            Ok(Stmt::Align(v, line_no))
        }
        "p2align" | "balign" => {
            let v = t.expr(lineno, numlabels)?;
            let bytes = if name == "p2align" {
                // Exponent form: evaluate then shift (values are small
                // alignment powers in practice).
                match v {
                    Expr::Int(k) => Expr::Int(1i64 << k.clamp(0, 40)),
                    _ => return Err("p2align needs a constant exponent".into()),
                }
            } else {
                v
            };
            Ok(Stmt::Align(bytes, line_no))
        }
        "ascii" => {
            let items = t.strings()?;
            Ok(Stmt::Data { size: 1, items, line: line_no })
        }
        "string" | "asciz" => {
            let mut items = t.strings()?;
            if let Some(DataItem::Str(bytes)) = items.last_mut() {
                bytes.push(0);
            }
            Ok(Stmt::Data { size: 1, items, line: line_no })
        }
        d if data_directives(arch).iter().any(|(n, _)| *n == d) => {
            let size = data_directives(arch).iter().find(|(n, _)| *n == d).unwrap().1;
            let mut items = Vec::new();
            loop {
                items.push(DataItem::Expr(t.expr(lineno, numlabels)?));
                if !t.eat_punct(",") {
                    break;
                }
            }
            Ok(Stmt::Data { size, items, line: line_no })
        }
        "fill" => {
            // `.fill repeat[, size[, value]]`: repeats of a data item.
            let count = t.expr(lineno, numlabels)?;
            let size = if t.eat_punct(",") {
                match t.bump() {
                    Some(GTok::Int(v)) if *v == 1 || *v == 2 || *v == 4 || *v == 8 => *v as u8,
                    _ => return Err("expected 1/2/4/8 size for .fill".into()),
                }
            } else {
                1
            };
            let value = if t.eat_punct(",") { t.expr(lineno, numlabels)? } else { Expr::Int(0) };
            Ok(Stmt::Times {
                count,
                inner: Box::new(Stmt::Data {
                    size,
                    items: vec![DataItem::Expr(value)],
                    line: line_no,
                }),
                line: line_no,
            })
        }
        "zero" | "skip" | "space" => {
            // The count is a full expression (`.skip -(a - b) * n, 0x90`
            // padding guards in kernel asm); a non-positive count skips
            // nothing (assemble clamps it).
            let count = t.expr(lineno, numlabels)?;
            let fill = if t.eat_punct(",") {
                match t.bump() {
                    Some(GTok::Int(v)) => *v as i64,
                    _ => return Err("expected fill value".into()),
                }
            } else {
                0
            };
            Ok(Stmt::Times {
                count,
                inner: Box::new(Stmt::Data {
                    size: 1,
                    items: vec![DataItem::Expr(Expr::Int(fill))],
                    line: line_no,
                }),
                line: line_no,
            })
        }
        d if IGNORED_DIRECTIVES.contains(&d) => Ok(Stmt::Bits(64, line_no)),
        d if d.starts_with("cfi_") => Ok(Stmt::Bits(64, line_no)),
        d if d.starts_with("stab") => Ok(Stmt::Bits(64, line_no)),
        _ => Err(format!("unsupported assembler directive `.{name}'")),
    }
}

#[derive(Clone, Copy)]
enum SectionKind {
    Text,
    Data,
    Bss,
    Rodata,
}

fn section_stmt(name: &str, kind: SectionKind, align: Option<u64>) -> Stmt {
    let flags = match kind {
        SectionKind::Text => crate::insn::SectionFlags::Text,
        SectionKind::Data => crate::insn::SectionFlags::Data,
        SectionKind::Bss => crate::insn::SectionFlags::Bss,
        SectionKind::Rodata => crate::insn::SectionFlags::Rodata,
    };
    Stmt::Section(crate::insn::SectionSpec {
        name: name.to_string(),
        flags,
        align,
    })
}

fn parse_section(t: &mut GToks) -> Result<crate::insn::SectionSpec, String> {
    // The name is usually a bare ident, but the kernel's EXPORT_SYMBOL
    // expansion quotes it (`.section ".export_symbol","a"`); a quoted
    // name is the literal string, no escapes in practice.
    let name = match t.bump().cloned() {
        Some(GTok::Ident(name)) => name,
        Some(GTok::Str(bytes)) => String::from_utf8_lossy(&bytes).into_owned(),
        _ => return Err("expected section name".into()),
    };
    // Attributes: `.section name, "flags", @type`.
    let mut write = false;
    let mut exec = false;
    let mut nobits = false;
    if t.eat_punct(",") {
        if let Some(GTok::Str(flags)) = t.bump() {
            for c in flags {
                match c {
                    b'w' | b'a' => write = true,
                    b'x' => exec = true,
                    b'M' | b'S' | b'G' | b'T' => {}
                    _ => {}
                }
            }
        }
        if t.eat_punct(",") {
            if let Some(GTok::Ident(ty)) = t.bump() {
                if ty.contains("nobits") || ty.contains("NOBITS") {
                    nobits = true;
                }
            }
        }
    }
    let base = name.split('.').nth(1).unwrap_or("");
    if name.starts_with(".bss") {
        nobits = true;
        write = true;
    } else if name.starts_with(".text") {
        exec = true;
    } else if name.starts_with(".rodata") {
        // read-only data
    } else if name.starts_with(".data") {
        write = true;
    } else if base.is_empty() {
        write = true;
    }
    let align = None;
    Ok(crate::insn::SectionSpec {
        name: name.clone(),
        flags: crate::insn::SectionFlags::Custom { write, exec, nobits, align },
        align: None,
    })
}

/// Wrapper over GAS tokens for operand/directive parsing.
struct GToks<'a> {
    g: &'a [GTok],
    pos: usize,
}

impl GToks<'_> {
    fn bump(&mut self) -> Option<&GTok> {
        let t = self.g.get(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn eat_punct(&mut self, p: &str) -> bool {
        if matches!(self.g.get(self.pos), Some(GTok::Punct(q)) if *q == p) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn strings(&mut self) -> Result<Vec<DataItem>, String> {
        let mut out = Vec::new();
        loop {
            match self.bump() {
                Some(GTok::Str(bytes)) => out.push(DataItem::Str(bytes.clone())),
                _ => return Err("expected string literal".into()),
            }
            if !self.eat_punct(",") {
                break;
            }
        }
        Ok(out)
    }

    /// Parses an expression in GAS syntax into the shared Expr AST.
    fn expr(&mut self, lineno: usize, numlabels: &NumLabels) -> Result<Expr, String> {
        self.expr_ex(lineno, numlabels, true)
    }

    /// `parens` off keeps `(` out of the expression (memory operands:
    /// `disp(base,index,scale)`), on treats parens as grouping
    /// (directive arguments).
    fn expr_ex(&mut self, lineno: usize, numlabels: &NumLabels, parens: bool) -> Result<Expr, String> {
        let mut toks = Vec::new();
        let depth_of = |g: &[GTok], start: usize| -> usize {
            // Consume tokens that belong to an expression: idents, ints,
            // numeric label refs, and arithmetic punctuation.
            // Parentheses are memory-operand syntax, not grouping:
            // displacement expressions run until one or a comma.
            let mut end = start;
            while end < g.len() {
                match &g[end] {
                    GTok::Ident(_)
                    | GTok::Int(_)
                    | GTok::NumRef(..)
                    // `*` lexes as Star; mid-expression it is
                    // multiplication (the leading indirect position is
                    // consumed before expr runs).
                    | GTok::Star
                    | GTok::Punct("+")
                    | GTok::Punct("-")
                    | GTok::Punct("*")
                    | GTok::Punct("/")
                    | GTok::Punct("|")
                    | GTok::Punct("&")
                    | GTok::Punct("^")
                    | GTok::Punct("~")
                    | GTok::Punct("!")
                    | GTok::Punct("<")
                    | GTok::Punct("<=")
                    | GTok::Punct(">")
                    | GTok::Punct(">=")
                    | GTok::Punct("==")
                    | GTok::Punct("!=")
                    | GTok::Punct("=")
                    | GTok::Punct("&&")
                    | GTok::Punct("||")
                    | GTok::Punct("<<")
                    | GTok::Punct(">>") => end += 1,
                    // Parens are memory-operand syntax in displacement
                    // expressions (`disp(base)`); in directive
                    // expressions they group. The guard binds the whole
                    // arm, so this pair must stay on its own.
                    GTok::Punct("(") | GTok::Punct(")") if parens => end += 1,
                    _ => break,
                }
            }
            end
        };
        let end = depth_of(self.g, self.pos);
        let slice = &self.g[self.pos..end];
        self.pos = end;
        let mut sub = 0usize;
        let mut paren = 0usize;
        for t in slice {
            match t {
                GTok::Punct("(") => {
                    paren += 1;
                    toks.push(Tok::Punct("("));
                }
                GTok::Punct(")") => {
                    paren = paren.saturating_sub(1);
                    toks.push(Tok::Punct(")"));
                }
                GTok::Star => toks.push(Tok::Punct("*")),
                GTok::Punct(p) => toks.push(Tok::Punct(p)),
                // GAS's `.` (current location) maps onto the nasm-side `$`.
                GTok::Ident(n) if n == "." => toks.push(Tok::Punct("$")),
                // Register operands appear in `.if` comparisons
                // (`.if \base == %rsp`); they resolve to their GAS
                // numbers in the preprocessor's symbol lookup.
                GTok::Reg(name) => toks.push(Tok::Ident(format!("%{name}"))),
                GTok::Ident(n) => toks.push(Tok::Ident(n.clone())),
                GTok::Int(v) => toks.push(Tok::Int(*v)),
                GTok::NumRef(d, fwd) => {
                    let name = if *fwd { numlabels.next_at(lineno + 1, *d) } else { numlabels.last_before(lineno, *d) }
                        .ok_or_else(|| format!("no `{d}' label defined in that direction"))?;
                    toks.push(Tok::Ident(name));
                    let _ = &mut sub;
                }
                _ => break,
            }
        }
        if toks.is_empty() {
            return Ok(Expr::Int(0));
        }
        let mut t = Toks::new(toks);
        parse::parse_expr(&mut t, 0)
    }
}

// ---------------------------------------------------------------------
// Instruction translation (AT&T -> internal Intel-shaped operands)
// ---------------------------------------------------------------------

struct GasInsn {
    /// nasm mnemonic.
    intel: &'static str,
    /// Size bits (op::BITS*) applied to memory operands.
    mem_size: Option<u64>,
    /// Extra size applied to the first (source) operand of movzx/movsx.
    src_size: Option<u64>,
    /// x87 p-forms keep GAS operand order.
    keep_order: bool,
}

fn translate_mnemonic(mn: &str) -> Option<GasInsn> {
    fn insn(intel: &'static str, mem_size: Option<u64>, src_size: Option<u64>, keep_order: bool) -> Option<GasInsn> {
        Some(GasInsn {
            intel,
            mem_size,
            src_size,
            keep_order,
        })
    }
    // Two-part forms: movzXY / movsXY (movzbl, movswq, ...). Single
    // suffixes (movsb/movsw/movsl/movsq string ops) fall through.
    if let Some(rest) = mn.strip_prefix("movz") {
        if rest.len() >= 2 {
            let (src, dst) = parse_two_suffixes(rest);
            return insn("movzx", Some(dst), Some(src), false);
        }
    }
    if let Some(rest) = mn.strip_prefix("movs") {
        if matches!(rest.as_bytes().first(), Some(b'b') | Some(b'w') | Some(b'l') | Some(b'q')) && rest.len() >= 2 {
            let (src, dst) = parse_two_suffixes(rest);
            return insn("movsx", Some(dst), Some(src), false);
        }
    }
    const SPECIAL: &[(&str, &str)] = &[("movslq", "movsxd"), ("cltq", "cdqe"), ("cltd", "cdq"), ("cqto", "cqo"), ("cwtl", "cwde"), ("cbtw", "cbw")];
    for (gas, intel) in SPECIAL {
        if mn == *gas {
            return insn(intel, None, None, false);
        }
    }
    // Integer forms take precedence over same-named SSE mnemonics
    // (movq the SSE move vs movq the integer alias).
    if let Some((base, size)) = strip_size_suffix(mn) {
        if crate::parse::opcode_index(base).is_some() {
            let keep = false;
            return insn(leak(base), Some(size), None, keep);
        }
    }
    if crate::parse::opcode_index(mn).is_some() {
        return insn(leak(mn), None, None, false);
    }
    None
}

fn leak(mn: &str) -> &'static str {
    Box::leak(mn.to_string().into_boxed_str())
}

/// movz/movs two-suffix decode: returns (src bits, dst bits).
fn parse_two_suffixes(rest: &str) -> (u64, u64) {
    let bits = |c: char| match c {
        'b' => op::BITS8,
        'w' => op::BITS16,
        'l' => op::BITS32,
        _ => op::BITS64,
    };
    let mut it = rest.chars();
    let src = it.next().map(bits).unwrap_or(op::BITS8);
    let dst = it.next().map(bits).unwrap_or(op::BITS64);
    (src, dst)
}

fn strip_size_suffix(mn: &str) -> Option<(&str, u64)> {
    let (base, size) = match mn.as_bytes().last()? {
        b'b' => (&mn[..mn.len() - 1], op::BITS8),
        b'w' => (&mn[..mn.len() - 1], op::BITS16),
        b'l' => (&mn[..mn.len() - 1], op::BITS32),
        b'q' => (&mn[..mn.len() - 1], op::BITS64),
        _ => return None,
    };
    Some((base, size))
}

fn is_branch_mnemonic(intel: &str) -> bool {
    if intel == "jmp" || intel == "call" {
        return true;
    }
    // Conditional branches: jcc (jne, jb, ...) but not jmpq-style names.
    intel.len() <= 3 && intel.starts_with('j') && crate::parse::opcode_index(intel).is_some()
}

pub fn parse_x86_insn(g: &[GTok], line_no: u32, lineno: usize, numlabels: &NumLabels) -> Result<Stmt, String> {
    // Split off instruction prefixes (lock/rep/...).
    let mut idx = 0;
    let mut prefixes = Vec::new();
    while let Some(GTok::Ident(word)) = g.get(idx) {
        match word.as_str() {
            "lock" | "rep" | "repe" | "repz" | "repne" | "repnz" | "bnd" | "xacquire" | "xrelease" => {
                prefixes.push(word.clone());
                idx += 1;
            }
            _ => break,
        }
    }
    let GTok::Ident(mn) = g.get(idx).ok_or("expected instruction after prefix")? else {
        return Err(format!("expected instruction, got {:?}", g.get(idx)));
    };
    let mn = mn.to_ascii_lowercase();
    let Some(gas_insn) = translate_mnemonic(&mn) else {
        return Err(format!("unknown or unsupported instruction `{mn}'"));
    };
    let GasInsn {
        intel,
        mem_size,
        src_size,
        keep_order,
    } = gas_insn;
    let opcode = crate::parse::opcode_index(intel).ok_or_else(|| format!("unknown instruction `{intel}'"))?;
    let branch = is_branch_mnemonic(intel);

    let mut t = GToks { g: &g[idx + 1..], pos: 0 };
    let mut gas_operands: Vec<Operand> = Vec::new();
    if t.pos < t.g.len() {
        loop {
            // The lexer emits `*` as GTok::Star, not Punct.
            let indirect = if matches!(t.g.get(t.pos), Some(GTok::Star)) {
                t.bump();
                true
            } else {
                t.eat_punct("*")
            };
            let operand = if matches!(t.g.get(t.pos), Some(GTok::Imm)) {
                // $immediate: size-agnostic (the suffix sizes memory).
                t.bump();
                Operand::imm(t.expr(lineno, numlabels)?)
            } else if matches!(t.g.get(t.pos), Some(GTok::Reg(_))) {
                let Some(GTok::Reg(name)) = t.bump() else { unreachable!() };
                let r = regs::lookup(name).ok_or_else(|| format!("unknown register `%{name}'"))?;
                Operand::reg(r as i32, regs::flags(r))
            } else {
                // Memory: `expr(base,index,scale)` or a bare displacement.
                let had_paren = matches!((t.g.get(t.pos), find_paren(t.g, t.pos)), (_, Some(_)));
                let disp = t.expr_ex(lineno, numlabels, false)?;
                let mut base: Option<i32> = None;
                let mut index: Option<i32> = None;
                let mut scale: u8 = 1;
                let mut rel = false;
                if had_paren {
                    if !t.eat_punct("(") {
                        return Err("expected `('".into());
                    }
                    if matches!(t.g.get(t.pos), Some(GTok::Reg(_))) {
                        let Some(GTok::Reg(bname)) = t.bump() else { unreachable!() };
                        if bname == regs::RIP {
                            rel = true;
                        } else {
                            let r = regs::lookup(bname).ok_or_else(|| format!("unknown register `%{bname}'"))?;
                            base = Some(r as i32);
                        }
                    }
                    if t.eat_punct(",") {
                        if matches!(t.g.get(t.pos), Some(GTok::Reg(_))) {
                            let Some(GTok::Reg(iname)) = t.bump() else { unreachable!() };
                            let r = regs::lookup(iname).ok_or_else(|| format!("unknown register `%{iname}'"))?;
                            index = Some(r as i32);
                        }
                        if t.eat_punct(",") {
                            match t.bump() {
                                Some(GTok::Int(k)) => {
                                    if !matches!(k, 1 | 2 | 4 | 8) {
                                        return Err(format!("invalid scale {k}"));
                                    }
                                    scale = *k as u8;
                                }
                                _ => return Err("expected scale".into()),
                            }
                        }
                    }
                    if !t.eat_punct(")") {
                        return Err("expected `)'".into());
                    }
                }

                // Jump targets are immediates; everything else is memory.
                // Bare displacement on a data instruction is rip-relative,
                // matching GAS in 64-bit mode; `*sym` is absolute.
                if branch && !indirect && base.is_none() && index.is_none() && !rel {
                    gas_operands.push(Operand::imm(disp));
                    if !t.eat_punct(",") {
                        break;
                    }
                    continue;
                }

                // All memory can match any r/m template (nasm mref_set_optype);
                // `*` only distinguishes branch targets from labels.
                let mut ty = op::MEMORY_ANY;
                let bits = src_size.or(mem_size);
                if let Some(b) = bits {
                    ty = ty & !op::SIZE_MASK | b;
                }
                let rel = rel || (base.is_none() && index.is_none() && !indirect && !rel && !had_paren && false);
                let is_rip_rel = rel || (base.is_none() && index.is_none() && !indirect);
                let eaflags = if is_rip_rel && !indirect { eaf::REL } else { 0 };
                let mut operand = Operand {
                    ty,
                    basereg: -1,
                    indexreg: -1,
                    scale,
                    seg: -1,
                    disp: Some(disp),
                    imm: None,
                    eaflags,
                };
                if let Some(b) = base {
                    operand.ty &= regs::flags(b as usize) | !op::RN_L16;
                    operand.basereg = b;
                }
                if let Some(i) = index {
                    operand.ty &= regs::flags(i as usize) | !op::RN_L16;
                    operand.indexreg = i;
                }
                if operand.ty & op::SIZE_MASK == 0 {
                    operand.ty |= op::RM_SEL;
                }
                operand
            };
            gas_operands.push(operand);
            if !t.eat_punct(",") {
                break;
            }
        }
    }

    let mut operands = gas_operands;
    if !keep_order {
        operands.reverse();
    }

    Ok(Stmt::Insn(crate::arch::Insn::X86(X86Insn {
        opcode,
        prefixes,
        operands,
        line: line_no,
    })))
}

/// Whether a `(` follows at or after `pos` before the operand ends.
fn find_paren(g: &[GTok], pos: usize) -> Option<usize> {
    for (i, t) in g.iter().enumerate().skip(pos) {
        match t {
            GTok::Punct("(") => return Some(i),
            GTok::Punct(",") => return None,
            _ => {}
        }
    }
    None
}
