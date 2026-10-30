//! GAS/AT&T syntax frontend (the `.s` dialect GNU `as` and glcc's own
//! generated asm use), translated onto the same statement model as the
//! Intel frontend: `src, dst` operand order, `%reg`/`$imm`/
//! `disp(base,index,scale)` operand syntax, and the directives compilers
//! actually emit.

use crate::assemble::{self, AsmError};
use crate::expr::Expr;
use crate::insn::{eaf, op, DataItem, Insn, Operand, Stmt};
use crate::parse::{self, ParseError, Tok, Toks};
use crate::regs;

pub fn assemble_to_elf64(src: &str) -> Result<Vec<u8>, Vec<AsmError>> {
    let stmts = parse_module(src).map_err(|errs| {
        errs.into_iter()
            .map(|e| AsmError { line: e.line, msg: e.msg })
            .collect::<Vec<_>>()
    })?;
    let module = assemble::assemble(&stmts)?;
    crate::elf::write_elf64(&module).map_err(|e| vec![AsmError { line: 0, msg: e }])
}

// ---------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum GTok {
    Ident(String),
    Int(i64),
    /// Numeric local label reference: `1b` / `1f`.
    NumRef(u8, bool),
    Str(Vec<u8>),
    Reg(String),
    Punct(&'static str),
    /// `$` immediate marker.
    Imm,
    Star,
}

fn lex_line(line: &str) -> Result<Vec<GTok>, String> {
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
            b'#' | b';' => break, // comment
            b'$' => {
                out.push(GTok::Imm);
                i += 1;
            }
            b'%' => {
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
                // `1b`/`1f` numeric local label references.
                if let Some(rest) = text.strip_suffix(['b', 'f']) {
                    if let Ok(digit) = rest.parse::<u8>() {
                        if digit < 10 && rest.len() == 1 {
                            out.push(GTok::NumRef(digit, text.ends_with('f')));
                            continue;
                        }
                    }
                }
                let v = gas_parse_number(text)
                    .ok_or_else(|| format!("invalid number `{text}'"))?;
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
            b',' => {
                out.push(GTok::Punct(","));
                i += 1;
            }
            b'*' => {
                out.push(GTok::Star);
                i += 1;
            }
            b'+' | b'-' | b'/' | b'|' | b'&' | b'^' | b'~' | b'!' | b'<' | b'>'
            | b':' | b'=' => {
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
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'.')
                {
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
    i64::from_str_radix(digits, radix).ok()
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
                    while n < 2
                        && i + 1 + n < b.len()
                        && (b[i + 1 + n] as char).is_ascii_hexdigit()
                    {
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

fn numeric_label_name(digit: u8, seq: usize) -> String {
    format!("..glasm{digit}_{seq}")
}

/// Maps (line index, direction) to the numeric label's unique name.
struct NumLabels {
    /// Per line: names of `N:` definitions made on that line.
    defs: Vec<Vec<(u8, String)>>,
}

impl NumLabels {
    fn scan(lines: &[&str]) -> NumLabels {
        let mut defs = Vec::with_capacity(lines.len());
        for line in lines {
            let mut here = Vec::new();
            // Labels only occur before the statement; scan leading
            // `digit+:` groups.
            let mut rest = line.trim_start();
            while let Some(digit) = rest.chars().next().and_then(|c| c.to_digit(10)) {
                let Ok(digit) = u8::try_from(digit) else { break };
                if let Some(after) = rest[1..].strip_prefix(':') {
                    here.push((digit, String::new()));
                    rest = after.trim_start();
                } else {
                    break;
                }
            }
            defs.push(here);
        }
        // Assign unique names in document order.
        let mut seqs = [0usize; 10];
        for here in &mut defs {
            for (digit, name) in here {
                *name = numeric_label_name(*digit, seqs[*digit as usize]);
                seqs[*digit as usize] += 1;
            }
        }
        NumLabels { defs }
    }

    fn def_on_line(&self, line: usize, digit: u8) -> Option<&str> {
        self.defs
            .get(line)?
            .iter()
            .find(|(d, _)| *d == digit)
            .map(|(_, n)| n.as_str())
    }

    /// Most recent definition strictly before `line`.
    fn last_before(&self, line: usize, digit: u8) -> Option<String> {
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
    fn next_at(&self, line: usize, digit: u8) -> Option<String> {
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
    "file", "ident", "size", "type", "local", "extern", "previous", "end", "appfile",
    "appline", "line", "loc", "eject", "psize", "noformat", "nolist", "list", "title", "sbttl",
    "version", "warn", "err", "print", "popsection", "pushsection_nop", "stabn", "stabs", "stabd",
];

const DATA_DIRECTIVES: &[(&str, u8)] = &[
    ("byte", 1),
    ("word", 2),
    ("short", 2),
    ("value", 2),
    ("long", 4),
    ("int", 4),
    ("quad", 8),
];

pub fn parse_module(src: &str) -> Result<Vec<Stmt>, Vec<ParseError>> {
    let lines: Vec<&str> = src.lines().collect();
    let numlabels = NumLabels::scan(&lines);
    let mut stmts = Vec::new();
    let mut errors = Vec::new();
    for (lineno, line) in lines.iter().enumerate() {
        // GAS `;` separates statements on a line.
        for part in split_statements(line) {
            let line_no = (lineno + 1) as u32;
            match parse_line(&part, line_no, lineno, &numlabels) {
                Ok(ss) => stmts.extend(ss),
                Err(msg) => errors.push(ParseError { line: line_no, msg }),
            }
        }
    }
    if errors.is_empty() {
        Ok(stmts)
    } else {
        Err(errors)
    }
}

/// Splits on `;` outside string literals.
fn split_statements(line: &str) -> Vec<&str> {
    let b = line.as_bytes();
    let mut parts = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'"' | b'\'' => {
                i += 1;
                while i < b.len() && b[i] != b'"' && b[i] != b'\'' {
                    if b[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
            }
            b';' => {
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

fn parse_line(
    line: &str,
    line_no: u32,
    lineno: usize,
    numlabels: &NumLabels,
) -> Result<Vec<Stmt>, String> {
    let g = lex_line(line)?;
    let mut i = 0;
    let mut out = Vec::new();

    // Labels: `name:` and numeric `N:`.
    loop {
        match g.get(i) {
            Some(GTok::Ident(name)) if g.get(i + 1) == Some(&GTok::Punct(":")) => {
                out.push(Stmt::Label(name.clone(), line_no));
                i += 2;
            }
            Some(GTok::Int(digit))
                if *digit < 10 && g.get(i + 1) == Some(&GTok::Punct(":")) =>
            {
                let name = numlabels
                    .def_on_line(lineno, *digit as u8)
                    .ok_or("internal: numeric label not scanned")?
                    .to_string();
                out.push(Stmt::Label(name, line_no));
                i += 2;
            }
            _ => break,
        }
    }

    if i >= g.len() {
        return Ok(out);
    }

    match &g[i] {
        GTok::Ident(word) if word.starts_with('.') => {
            out.push(parse_directive(&g[i..], line_no, lineno, numlabels)?);
        }
        _ => {
            out.push(parse_instruction(&g[i..], line_no, lineno, numlabels)?);
        }
    }
    Ok(out)
}

fn directive_name(word: &str) -> &str {
    word.trim_start_matches('.')
}

fn parse_directive(
    g: &[GTok],
    line_no: u32,
    lineno: usize,
    numlabels: &NumLabels,
) -> Result<Stmt, String> {
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
        "section" => parse_section(&mut t),
        "globl" | "global" | "weak" => {
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
            Ok(Stmt::Comm { name: sym, size, align, global, line: line_no })
        }
        "align" => {
            let Some(GTok::Int(v)) = t.bump() else {
                return Err("expected alignment for .align".into());
            };
            let v = *v;
            // On x86 ELF, .align is in bytes.
            Ok(Stmt::Align(Expr::Int(v), line_no))
        }
        "p2align" | "balign" => {
            let Some(GTok::Int(v)) = t.bump() else {
                return Err("expected alignment for .p2align".into());
            };
            let bytes = if name == "p2align" { 1i64 << (*v).min(40) } else { *v };
            Ok(Stmt::Align(Expr::Int(bytes), line_no))
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
        d if DATA_DIRECTIVES.iter().any(|(n, _)| *n == d) => {
            let size = DATA_DIRECTIVES.iter().find(|(n, _)| *n == d).unwrap().1;
            let mut items = Vec::new();
            loop {
                items.push(DataItem::Expr(t.expr(lineno, numlabels)?));
                if !t.eat_punct(",") {
                    break;
                }
            }
            Ok(Stmt::Data { size, items, line: line_no })
        }
        "zero" | "skip" | "space" => {
            let Some(GTok::Int(n)) = t.bump() else {
                return Err("expected count".into());
            };
            let n = *n;
            let fill = if t.eat_punct(",") {
                match t.bump() {
                    Some(GTok::Int(v)) => *v as u8,
                    _ => return Err("expected fill value".into()),
                }
            } else {
                0
            };
            Ok(Stmt::Times {
                count: Expr::Int(n),
                inner: Box::new(Stmt::Data {
                    size: 1,
                    items: vec![DataItem::Expr(Expr::Int(fill as i64))],
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
    Stmt::Section(crate::insn::SectionSpec { name: name.to_string(), flags, align })
}

fn parse_section(t: &mut GToks) -> Result<Stmt, String> {
    let Some(GTok::Ident(name)) = t.bump().cloned() else {
        return Err("expected section name".into());
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
    Ok(Stmt::Section(crate::insn::SectionSpec {
        name: name.clone(),
        flags: crate::insn::SectionFlags::Custom { write, exec, nobits, align },
        align: None,
    }))
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
                    | GTok::Punct("+")
                    | GTok::Punct("-")
                    | GTok::Punct("*")
                    | GTok::Punct("/")
                    | GTok::Punct("|")
                    | GTok::Punct("&")
                    | GTok::Punct("^")
                    | GTok::Punct("~")
                    | GTok::Punct("!")
                    | GTok::Punct("<<")
                    | GTok::Punct(">>") => end += 1,
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
                GTok::Punct(p) => toks.push(Tok::Punct(p)),
                GTok::Ident(n) => toks.push(Tok::Ident(n.clone())),
                GTok::Int(v) => toks.push(Tok::Int(*v)),
                GTok::NumRef(d, fwd) => {
                    let name = if *fwd {
                        numlabels.next_at(lineno + 1, *d)
                    } else {
                        numlabels.last_before(lineno, *d)
                    }
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
        Some(GasInsn { intel, mem_size, src_size, keep_order })
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
        if matches!(rest.as_bytes().first(), Some(b'b') | Some(b'w') | Some(b'l') | Some(b'q'))
            && rest.len() >= 2
        {
            let (src, dst) = parse_two_suffixes(rest);
            return insn("movsx", Some(dst), Some(src), false);
        }
    }
    const SPECIAL: &[(&str, &str)] = &[
        ("movslq", "movsxd"),
        ("cltq", "cdqe"),
        ("cltd", "cdq"),
        ("cqto", "cqo"),
        ("cwtl", "cwde"),
        ("cbtw", "cbw"),
    ];
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
    intel.len() <= 3
        && intel.starts_with('j')
        && crate::parse::opcode_index(intel).is_some()
}

fn parse_instruction(
    g: &[GTok],
    line_no: u32,
    lineno: usize,
    numlabels: &NumLabels,
) -> Result<Stmt, String> {
    // Split off instruction prefixes (lock/rep/...).
    let mut idx = 0;
    let mut prefixes = Vec::new();
    while let Some(GTok::Ident(word)) = g.get(idx) {
        match word.as_str() {
            "lock" | "rep" | "repe" | "repz" | "repne" | "repnz" | "bnd" | "xacquire"
            | "xrelease" => {
                prefixes.push(word.clone());
                idx += 1;
            }
            _ => break,
        }
    }
    let GTok::Ident(mn) = g.get(idx).ok_or("expected instruction after prefix")? else {
        unreachable!()
    };
    let mn = mn.to_ascii_lowercase();
    let Some(gas_insn) = translate_mnemonic(&mn) else {
        return Err(format!("unknown or unsupported instruction `{mn}'"));
    };
    let GasInsn { intel, mem_size, src_size, keep_order } = gas_insn;
    let opcode = crate::parse::opcode_index(intel)
        .ok_or_else(|| format!("unknown instruction `{intel}'"))?;
    let branch = is_branch_mnemonic(intel);

    let mut t = GToks { g: &g[idx + 1..], pos: 0 };
    let mut gas_operands: Vec<Operand> = Vec::new();
    if t.pos < t.g.len() {
        loop {
            let indirect = t.eat_punct("*");
            let operand = if matches!(t.g.get(t.pos), Some(GTok::Imm)) {
                // $immediate: size-agnostic (the suffix sizes memory).
                t.bump();
                Operand::imm(t.expr(lineno, numlabels)?)
            } else if matches!(t.g.get(t.pos), Some(GTok::Reg(_))) {
                let Some(GTok::Reg(name)) = t.bump() else { unreachable!() };
                let r = regs::lookup(name)
                    .ok_or_else(|| format!("unknown register `%{name}'"))?;
                Operand::reg(r as i32, regs::flags(r))
            } else {
                // Memory: `expr(base,index,scale)` or a bare displacement.
                let had_paren = matches!(
                    (t.g.get(t.pos), find_paren(t.g, t.pos)),
                    (_, Some(_))
                );
                let disp = t.expr(lineno, numlabels)?;
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
                            let r = regs::lookup(bname)
                                .ok_or_else(|| format!("unknown register `%{bname}'"))?;
                            base = Some(r as i32);
                        }
                    }
                    if t.eat_punct(",") {
                        if matches!(t.g.get(t.pos), Some(GTok::Reg(_))) {
                            let Some(GTok::Reg(iname)) = t.bump() else { unreachable!() };
                            let r = regs::lookup(iname)
                                .ok_or_else(|| format!("unknown register `%{iname}'"))?;
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

                let mut ty = op::MEMORY_ANY;
                if indirect {
                    ty = op::MEMORY;
                }
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

    Ok(Stmt::Insn(Insn { opcode, prefixes, operands, line: line_no }))
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
