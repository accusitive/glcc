//! Tokenizer and the nasm Intel-syntax frontend (reference: nasm's
//! `asm/parser.c` and `asm/stdscan.c`, reduced to the subset glasm
//! supports).

use crate::expr::Expr;
use crate::insn::*;
use crate::insn::op;
use crate::regs;

#[derive(Debug)]
pub struct ParseError {
    pub line: u32,
    pub msg: String,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Tok {
    Ident(String),
    Int(i64),
    Str(Vec<u8>),
    Punct(&'static str),
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_' || b == b'.' || b == b'?' || b == b'@'
}

fn is_ident_cont(b: u8) -> bool {
    is_ident_start(b) || b.is_ascii_digit() || b == b'$' || b == b'#'
}

struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Lexer<'a> {
        Lexer { src: src.as_bytes(), pos: 0 }
    }

    fn next(&mut self) -> Option<Tok> {
        while self.pos < self.src.len() && (self.src[self.pos] as char).is_whitespace() {
            self.pos += 1;
        }
        if self.pos >= self.src.len() {
            return None;
        }
        let b = self.src[self.pos];
        let rest = &self.src[self.pos..];
        match b {
            b';' | b'#' => None, // rest of line is a comment
            b'"' | b'\'' => {
                let quote = b;
                let mut bytes = Vec::new();
                let mut i = 1;
                while i < rest.len() && rest[i] != quote {
                    bytes.push(rest[i]);
                    i += 1;
                }
                self.pos += i + 1.min(rest.len() - i);
                Some(Tok::Str(bytes))
            }
            b'`' => {
                let mut bytes = Vec::new();
                let mut i = 1;
                while i < rest.len() && rest[i] != b'`' {
                    if rest[i] == b'\\' && i + 1 < rest.len() {
                        i += 1;
                        bytes.push(match rest[i] {
                            b'n' => b'\n',
                            b't' => b'\t',
                            b'r' => b'\r',
                            b'0' => 0,
                            b'\\' => b'\\',
                            b'`' => b'`',
                            other => other,
                        });
                    } else {
                        bytes.push(rest[i]);
                    }
                    i += 1;
                }
                self.pos += i + 1.min(rest.len() - i);
                Some(Tok::Str(bytes))
            }
            b'0'..=b'9' => Some(self.lex_number(rest)),
            b'$' => {
                if rest.starts_with(b"$$") {
                    self.pos += 2;
                    Some(Tok::Punct("$$"))
                } else {
                    self.pos += 1;
                    Some(Tok::Punct("$"))
                }
            }
            _ if is_ident_start(b) => {
                let mut i = 0;
                while i < rest.len() && is_ident_cont(rest[i]) {
                    i += 1;
                }
                let s = std::str::from_utf8(&rest[..i]).unwrap().to_string();
                self.pos += i;
                Some(Tok::Ident(s))
            }
            _ => {
                for p in ["<<", ">>", "//", "%%"] {
                    if rest.starts_with(p.as_bytes()) {
                        self.pos += 2;
                        return Some(Tok::Punct(p));
                    }
                }
                let c = b as char;
                let p: &'static str = match c {
                    '+' => "+",
                    '-' => "-",
                    '*' => "*",
                    '/' => "/",
                    ',' => ",",
                    '[' => "[",
                    ']' => "]",
                    ':' => ":",
                    '(' => "(",
                    ')' => ")",
                    '|' => "|",
                    '&' => "&",
                    '^' => "^",
                    '~' => "~",
                    '!' => "!",
                    '=' => "=",
                    _ => return None,
                };
                self.pos += 1;
                Some(Tok::Punct(p))
            }
        }
    }

    fn lex_number(&mut self, rest: &[u8]) -> Tok {
        let (mut radix, digits_start) = if rest.starts_with(b"0x") || rest.starts_with(b"0X") {
            (16u32, 2)
        } else if rest.starts_with(b"0b") || rest.starts_with(b"0B") {
            (2, 2)
        } else if rest.starts_with(b"0o") || rest.starts_with(b"0O") {
            (8, 2)
        } else {
            (10, 0)
        };
        let mut i = digits_start;
        while i < rest.len()
            && ((rest[i] as char).is_ascii_alphanumeric() || rest[i] == b'_')
        {
            i += 1;
        }
        let mut text: String = std::str::from_utf8(&rest[digits_start..i])
            .unwrap()
            .replace('_', "");
        if radix == 10 && !text.is_empty() {
            match text.as_bytes()[text.len() - 1] {
                b'h' | b'H' => {
                    radix = 16;
                    text.pop();
                }
                b'b' | b'B' => {
                    radix = 2;
                    text.pop();
                }
                b'o' | b'O' | b'q' | b'Q' => {
                    radix = 8;
                    text.pop();
                }
                _ => {}
            }
        }
        self.pos += i;
        let val = i64::from_str_radix(&text, radix).unwrap_or_else(|_| {
            panic!("bad numeric literal `{text}'");
        });
        Tok::Int(val)
    }
}

// ---------------------------------------------------------------------
// Expression parsing
// ---------------------------------------------------------------------

pub(crate) struct Toks {
    toks: Vec<Tok>,
    pos: usize,
}

impl Toks {
    pub(crate) fn new(toks: Vec<Tok>) -> Toks {
        Toks { toks, pos: 0 }
    }

    pub(crate) fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    pub(crate) fn bump(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    pub(crate) fn eat_punct(&mut self, p: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Punct(q)) if *q == p) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
}

pub(crate) fn parse_expr(t: &mut Toks, min_prec: u8) -> Result<Expr, String> {
    let mut lhs = parse_unary(t)?;
    loop {
        let (op, prec) = match t.peek() {
            Some(Tok::Punct("|")) => ("|", 1),
            Some(Tok::Punct("^")) => ("^", 2),
            Some(Tok::Punct("&")) => ("&", 3),
            Some(Tok::Punct("<<")) => ("<<", 4),
            Some(Tok::Punct(">>")) => (">>", 4),
            Some(Tok::Punct("+")) => ("+", 5),
            Some(Tok::Punct("-")) => ("-", 5),
            Some(Tok::Punct("*")) => ("*", 6),
            Some(Tok::Punct("/")) | Some(Tok::Punct("//")) | Some(Tok::Punct("%%")) => ("/", 6),
            _ => break,
        };
        if prec < min_prec {
            break;
        }
        t.bump();
        let rhs = parse_expr(t, prec + 1)?;
        lhs = match op {
            "+" => Expr::Add(Box::new(lhs), Box::new(rhs)),
            "-" => Expr::Sub(Box::new(lhs), Box::new(rhs)),
            "*" => Expr::Mul(Box::new(lhs), Box::new(rhs)),
            "/" => Expr::Div(Box::new(lhs), Box::new(rhs)),
            "<<" => Expr::Shl(Box::new(lhs), Box::new(rhs)),
            ">>" => Expr::Shr(Box::new(lhs), Box::new(rhs)),
            "&" => Expr::And(Box::new(lhs), Box::new(rhs)),
            "|" => Expr::Or(Box::new(lhs), Box::new(rhs)),
            "^" => Expr::Xor(Box::new(lhs), Box::new(rhs)),
            _ => unreachable!(),
        };
    }
    Ok(lhs)
}

fn parse_unary(t: &mut Toks) -> Result<Expr, String> {
    if t.eat_punct("-") {
        return Ok(Expr::Neg(Box::new(parse_unary(t)?)));
    }
    if t.eat_punct("+") {
        return parse_unary(t);
    }
    if t.eat_punct("~") {
        return Ok(Expr::BitNot(Box::new(parse_unary(t)?)));
    }
    if t.eat_punct("(") {
        let e = parse_expr(t, 0)?;
        if !t.eat_punct(")") {
            return Err("expected `)'".into());
        }
        return Ok(e);
    }
    match t.bump() {
        Some(Tok::Int(i)) => Ok(Expr::Int(i)),
        Some(Tok::Punct("$")) => Ok(Expr::Here),
        Some(Tok::Punct("$$")) => Ok(Expr::SectStart),
        Some(Tok::Ident(name)) => Ok(Expr::Sym(name)),
        _ => Err("expected expression".into()),
    }
}

// ---------------------------------------------------------------------
// Statement parsing
// ---------------------------------------------------------------------

const DATA_DIRS: [&str; 8] = ["db", "dw", "dd", "dq", "resb", "resw", "resd", "resq"];
const OTHER_DIRS: [&str; 6] = ["section", "segment", "global", "extern", "alignb", "bits"];

fn is_directive(name: &str) -> bool {
    matches!(name, "align" | "times") || DATA_DIRS.contains(&name) || OTHER_DIRS.contains(&name)
}

/// A bare identifier at line start is a label unless it names an
/// instruction (nasm gives mnemonics priority over colon-less labels).
fn is_mnemonic(name: &str) -> bool {
    crate::parse::opcode_index(name).is_some()
}

pub fn parse_module(src: &str) -> Result<Vec<Stmt>, Vec<ParseError>> {
    let mut stmts = Vec::new();
    let mut errors = Vec::new();
    let mut last_global: Option<String> = None;
    for (lineno, line) in src.lines().enumerate() {
        let line_no = (lineno + 1) as u32;
        match parse_line(line, line_no, &mut last_global) {
            Ok(ss) => stmts.extend(ss),
            Err(msg) => errors.push(ParseError { line: line_no, msg }),
        }
    }
    if errors.is_empty() {
        Ok(stmts)
    } else {
        Err(errors)
    }
}

fn parse_line(
    line: &str,
    line_no: u32,
    last_global: &mut Option<String>,
) -> Result<Vec<Stmt>, String> {
    let mut lx = Lexer::new(line);
    let mut toks = Vec::new();
    while let Some(t) = lx.next() {
        toks.push(t);
    }
    let mut t = Toks { toks, pos: 0 };
    if t.peek().is_none() {
        return Ok(Vec::new());
    }

    // `name:` always labels; a bare identifier labels when nothing follows
    // or when what follows is a directive (nasm's `foo db 1` form).
    let mut label: Option<String> = None;
    if let Some(Tok::Ident(name)) = t.peek().cloned() {
        let next = t.toks.get(t.pos + 1).cloned();
        let mnemonic = is_mnemonic(&name.to_ascii_lowercase());
        let is_label = match &next {
            None => !mnemonic,
            Some(Tok::Punct(":")) => true,
            Some(Tok::Punct("[")) => !mnemonic,
            Some(Tok::Ident(n)) => !mnemonic && is_directive(&n.to_ascii_lowercase()),
            _ => false,
        };
        if is_label {
            t.bump();
            if next == Some(Tok::Punct(":")) {
                t.bump();
            }
            // nasm local labels (`.foo`) attach to the last non-local label.
            let local = name.starts_with('.');
            let name = match name.strip_prefix('.') {
                Some(local_name) => {
                    let parent = last_global.as_ref().ok_or_else(|| {
                        format!("local label `.{local_name}' before any non-local label")
                    })?;
                    format!("{parent}.{local_name}")
                }
                None => name,
            };
            if !local {
                *last_global = Some(name.clone());
            }
            label = Some(name);
        }
    }

    if t.pos >= t.toks.len() {
        return Ok(label.map_or_else(Vec::new, |n| vec![Stmt::Label(n, line_no)]));
    }

    let mut stmt = parse_stmt(&mut t, line_no)?;
    if t.pos < t.toks.len() {
        return Err(format!("junk at end of statement: {:?}", t.toks[t.pos]));
    }
    if let Some(parent) = &*last_global {
        expand_stmt_locals(&mut stmt, parent);
    }
    let mut out = Vec::new();
    if let Some(name) = label {
        out.push(Stmt::Label(name, line_no));
    }
    out.push(stmt);
    Ok(out)
}

/// Rewrites local label references (`.foo`) inside a statement to their
/// fully qualified form (`parent.foo`), matching nasm's scoping.
fn expand_stmt_locals(stmt: &mut Stmt, parent: &str) {
    fn expand_expr(e: &mut Expr, parent: &str) {
        match e {
            Expr::Sym(s) if s.starts_with('.') && !s.starts_with("..") => {
                *s = format!("{parent}{s}");
            }
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Shl(a, b)
            | Expr::Shr(a, b)
            | Expr::And(a, b)
            | Expr::Or(a, b)
            | Expr::Xor(a, b) => {
                expand_expr(a, parent);
                expand_expr(b, parent);
            }
            Expr::Neg(a) | Expr::BitNot(a) => expand_expr(a, parent),
            _ => {}
        }
    }
    fn expand_operand(op: &mut Operand, parent: &str) {
        if let Some(e) = &mut op.imm {
            expand_expr(e, parent);
        }
        if let Some(d) = &mut op.disp {
            expand_expr(d, parent);
        }
    }
    match stmt {
        Stmt::Insn(insn) => {
            for op in &mut insn.operands {
                expand_operand(op, parent);
            }
        }
        Stmt::Data { items, .. } => {
            for item in items {
                if let DataItem::Expr(e) = item {
                    expand_expr(e, parent);
                }
            }
        }
        Stmt::Reserve { count, .. } | Stmt::Align(count, _) => {
            expand_expr(count, parent);
        }
        Stmt::Times { count, inner, .. } => {
            expand_expr(count, parent);
            expand_stmt_locals(inner, parent);
        }
        _ => {}
    }
}

fn parse_stmt(t: &mut Toks, line_no: u32) -> Result<Stmt, String> {
    if t.eat_punct("[") {
        let inner = parse_directive(t, line_no)?;
        if !t.eat_punct("]") {
            return Err("expected `]'".into());
        }
        return Ok(inner);
    }
    let Some(Tok::Ident(word)) = t.peek().cloned() else {
        return Err("expected directive or instruction".into());
    };
    let lw = word.to_ascii_lowercase();
    if is_directive(&lw) {
        t.bump();
        return parse_directive_named(&lw, t, line_no);
    }
    parse_instruction(t, line_no)
}

fn section_flags(name: &str) -> Option<SectionFlags> {
    match name {
        ".text" => Some(SectionFlags::Text),
        ".data" => Some(SectionFlags::Data),
        ".bss" => Some(SectionFlags::Bss),
        ".rodata" => Some(SectionFlags::Rodata),
        _ => None,
    }
}

fn parse_directive_named(word: &str, t: &mut Toks, line_no: u32) -> Result<Stmt, String> {
    match word {
        "section" | "segment" => {
            let name = match t.bump() {
                Some(Tok::Ident(n)) => n,
                _ => return Err("expected section name".into()),
            };
            let mut flags = section_flags(&name).unwrap_or(SectionFlags::Custom {
                write: false,
                exec: false,
                nobits: false,
                align: None,
            });
            let mut align = None;
            while t.pos < t.toks.len() {
                match t.bump() {
                    Some(Tok::Ident(n)) if n.eq_ignore_ascii_case("write") => {
                        if let SectionFlags::Custom { write, .. } = &mut flags {
                            *write = true;
                        }
                    }
                    Some(Tok::Ident(n)) if n.eq_ignore_ascii_case("exec") => {
                        if let SectionFlags::Custom { exec, .. } = &mut flags {
                            *exec = true;
                        }
                    }
                    Some(Tok::Ident(n))
                        if n.eq_ignore_ascii_case("nobits") || n.eq_ignore_ascii_case("nobytes") =>
                    {
                        if let SectionFlags::Custom { nobits, .. } = &mut flags {
                            *nobits = true;
                        }
                    }
                    Some(Tok::Ident(n)) if n.eq_ignore_ascii_case("align") => {
                        t.eat_punct("=");
                        match t.bump() {
                            Some(Tok::Int(v)) => align = Some(v as u64),
                            _ => return Err("expected integer for align".into()),
                        }
                    }
                    Some(Tok::Punct("=")) => {}
                    _ => return Err("bad section attribute".into()),
                }
            }
            Ok(Stmt::Section(SectionSpec { name, flags, align }))
        }
        "global" | "extern" => {
            let mut names = Vec::new();
            loop {
                match t.bump() {
                    Some(Tok::Ident(n)) => names.push(n),
                    _ => return Err(format!("expected symbol name after `{word}'")),
                }
                if !t.eat_punct(",") {
                    break;
                }
            }
            Ok(if word == "global" {
                Stmt::Global(names, line_no)
            } else {
                Stmt::Extern(names, line_no)
            })
        }
        "bits" => match t.bump() {
            Some(Tok::Int(n)) => Ok(Stmt::Bits(n as u8, line_no)),
            _ => Err("expected integer for bits".into()),
        },
        "align" | "alignb" => {
            let e = parse_expr(t, 0)?;
            Ok(Stmt::Align(e, line_no))
        }
        "times" => {
            let count = parse_expr(t, 0)?;
            let inner = parse_stmt(t, line_no)?;
            Ok(Stmt::Times { count, inner: Box::new(inner), line: line_no })
        }
        d if DATA_DIRS.contains(&d) => {
            let (size, reserve) = match d {
                "db" => (1, false),
                "dw" => (2, false),
                "dd" => (4, false),
                "dq" => (8, false),
                "resb" => (1, true),
                "resw" => (2, true),
                "resd" => (4, true),
                "resq" => (8, true),
                _ => unreachable!(),
            };
            if reserve {
                let count = parse_expr(t, 0)?;
                return Ok(Stmt::Reserve { size, count, line: line_no });
            }
            let mut items = Vec::new();
            loop {
                match t.peek() {
                    Some(Tok::Str(bytes)) => {
                        items.push(DataItem::Str(bytes.clone()));
                        t.bump();
                    }
                    _ => items.push(DataItem::Expr(parse_expr(t, 0)?)),
                }
                if !t.eat_punct(",") {
                    break;
                }
            }
            Ok(Stmt::Data { size, items, line: line_no })
        }
        _ => Err(format!("unknown directive `{word}'")),
    }
}

fn parse_directive(t: &mut Toks, line_no: u32) -> Result<Stmt, String> {
    let Some(Tok::Ident(word)) = t.peek().cloned() else {
        return Err("expected directive inside [ ]".into());
    };
    let lw = word.to_ascii_lowercase();
    if !is_directive(&lw) {
        return Err(format!("unknown directive `{word}'"));
    }
    t.bump();
    parse_directive_named(&lw, t, line_no)
}

// ---------------------------------------------------------------------
// Instruction parsing
// ---------------------------------------------------------------------

const SIZE_PREFIXES: [(&str, u64); 7] = [
    ("byte", op::BITS8),
    ("word", op::BITS16),
    ("dword", op::BITS32),
    ("qword", op::BITS64),
    ("oword", op::BITS128),
    ("yword", op::BITS256),
    ("zword", op::BITS512),
];

pub fn opcode_index(mnemonic: &str) -> Option<usize> {
    crate::insns_gen::MNEMONICS.iter().position(|m| *m == mnemonic)
}

const PREFIXES: [&str; 11] = [
    "lock", "rep", "repe", "repz", "repne", "repnz", "bnd", "notrack", "xacquire", "xrelease",
    "hint_nop",
];

fn parse_instruction(t: &mut Toks, line_no: u32) -> Result<Stmt, String> {
    let mut prefixes = Vec::new();
    let mn = loop {
        let Some(Tok::Ident(word)) = t.bump() else {
            return Err("expected instruction".into());
        };
        let lower = word.to_ascii_lowercase();
        if PREFIXES.contains(&lower.as_str()) {
            prefixes.push(lower);
        } else {
            break lower;
        }
    };
    let Some(opcode) = opcode_index(&mn) else {
        return Err(format!("unknown or unsupported instruction `{mn}'"));
    };
    let mut operands = Vec::new();
    if t.pos < t.toks.len() {
        loop {
            operands.push(parse_operand(t)?);
            if !t.eat_punct(",") {
                break;
            }
        }
    }
    Ok(Stmt::Insn(Insn { opcode, prefixes, operands, line: line_no }))
}

fn parse_operand(t: &mut Toks) -> Result<Operand, String> {
    // Branch distance keywords carry no encoding here; jump matching
    // picks short or long forms itself.
    if matches!(t.peek(), Some(Tok::Ident(n)) if n.eq_ignore_ascii_case("short")
        || n.eq_ignore_ascii_case("near"))
    {
        t.bump();
    }
    let mut size_bits: Option<u64> = None;
    if let Some(Tok::Ident(n)) = t.peek() {
        let lower = n.to_ascii_lowercase();
        if let Some((_, bits)) = SIZE_PREFIXES.iter().find(|(p, _)| *p == lower) {
            size_bits = Some(*bits);
            t.bump();
        }
    }
    if t.eat_punct("[") {
        let mem = parse_mem_body(t)?;
        if !t.eat_punct("]") {
            return Err("expected `]'".into());
        }
        return Ok(mem_operand(mem, size_bits));
    }
    if let Some(bits) = size_bits {
        let e = parse_expr(t, 0)?;
        let mut o = Operand::imm(e);
        o.ty |= bits;
        return Ok(o);
    }
    if let Some(Tok::Ident(n)) = t.peek() {
        if let Some(r) = regs::lookup(&n.to_ascii_lowercase()) {
            t.bump();
            return Ok(Operand::reg(r as i32, regs::flags(r)));
        }
    }
    Ok(Operand::imm(parse_expr(t, 0)?))
}

/// Parses the inside of `[...]`: registers, scaled indexes, and a
/// displacement expression, combined with +/-.
struct MemParts {
    base: Option<i32>,
    index: Option<i32>,
    scale: u8,
    disp: Expr,
    rel: bool,
}

fn mem_operand(mem: MemParts, size_bits: Option<u64>) -> Operand {
    // nasm's mref_set_optype: a memory operand claims all r/m classes,
    // then narrows by the registers involved.
    let mut ty = op::MEMORY_ANY;
    if let Some(bits) = size_bits {
        ty |= bits;
    }
    // `[rel expr]` (or a bare rip base) is rip-relative: no base
    // register, the flag tells process_ea to encode rip-relative.
    let (basereg, eaflags) = if mem.rel {
        (-1, crate::insn::eaf::REL)
    } else {
        (mem.base.unwrap_or(-1), 0)
    };
    let mut operand = Operand {
        ty,
        basereg,
        indexreg: mem.index.unwrap_or(-1),
        scale: mem.scale,
        seg: -1,
        disp: Some(mem.disp),
        imm: None,
        eaflags,
    };
    if basereg != -1 {
        operand.ty &= regs::flags(basereg as usize) | !op::RN_L16;
    } else if operand.indexreg == -1 {
        operand.ty |= if eaflags & crate::insn::eaf::REL != 0 {
            op::IP_REL
        } else {
            op::MEM_OFFS
        };
    }
    if operand.indexreg != -1 {
        operand.ty &= regs::flags(operand.indexreg as usize) | !op::RN_L16;
    }
    if operand.ty & op::SIZE_MASK == 0 {
        operand.ty |= op::RM_SEL;
    }
    operand
}

fn parse_mem_body(t: &mut Toks) -> Result<MemParts, String> {
    let mut base: Option<i32> = None;
    let mut index: Option<i32> = None;
    let mut scale: u8 = 1;
    let mut rel = false;
    let mut terms: Vec<(bool, Expr)> = Vec::new();
    // `[rel expr]` is rip-relative.
    if matches!(t.peek(), Some(Tok::Ident(n)) if n.eq_ignore_ascii_case("rel"))
        && matches!(t.toks.get(t.pos + 1), Some(Tok::Ident(_)))
    {
        t.bump();
        rel = true;
    }
    // `[abs expr]` forces absolute addressing.
    if matches!(t.peek(), Some(Tok::Ident(n)) if n.eq_ignore_ascii_case("abs"))
        && matches!(t.toks.get(t.pos + 1), Some(Tok::Ident(_)))
    {
        t.bump();
    }
    loop {
        let mut neg = false;
        while t.eat_punct("-") {
            neg = !neg;
        }
        if !neg {
            t.eat_punct("+");
        }
        // Terms may be scaled (`4*rbx`), so parse multiplicatively but
        // stop before the +/- separator.
        let term = parse_expr(t, 6)?;
        match &term {
            Expr::Sym(name) => {
                if name == regs::RIP {
                    if base.is_none() {
                        rel = true;
                    } else {
                        return Err("cannot scale rip".into());
                    }
                } else if let Some(r) = regs::lookup(name) {
                    if regs::flags(r) & op::BITS64 != 0 {
                        if base.is_none() {
                            base = Some(r as i32);
                        } else if index.is_none() {
                            index = Some(r as i32);
                        } else {
                            return Err("too many registers in address".into());
                        }
                    } else {
                        return Err(format!("register `{name}' cannot be used as base"));
                    }
                } else {
                    terms.push((neg, term));
                }
            }
            Expr::Mul(a, b) => {
                let scaled = if let (Some(rn), Some(k)) = (a.as_sym(), b.as_int()) {
                    Some((rn.to_string(), *k))
                } else if let (Some(rn), Some(k)) = (b.as_sym(), a.as_int()) {
                    Some((rn.to_string(), *k))
                } else {
                    None
                };
                match scaled {
                    Some((rn, k)) => {
                        let r = regs::lookup(&rn)
                            .ok_or_else(|| format!("unknown register `{rn}'"))?;
                        if !matches!(k, 1 | 2 | 4 | 8) {
                            return Err(format!("invalid scale {k}"));
                        }
                        if base.is_none() && k == 1 {
                            base = Some(r as i32);
                        } else if index.is_none() {
                            index = Some(r as i32);
                            scale = k as u8;
                        } else {
                            return Err("too many registers in address".into());
                        }
                    }
                    None => terms.push((neg, term)),
                }
            }
            _ => terms.push((neg, term)),
        }
        if !matches!(t.peek(), Some(Tok::Punct("+")) | Some(Tok::Punct("-"))) {
            break;
        }
    }
    let mut disp = Expr::Int(0);
    for (neg, e) in terms {
        disp = if neg {
            Expr::Sub(Box::new(disp), Box::new(e))
        } else {
            Expr::Add(Box::new(disp), Box::new(e))
        };
    }
    Ok(MemParts { base, index, scale, disp, rel })
}

impl Expr {
    fn as_sym(&self) -> Option<&str> {
        match self {
            Expr::Sym(s) => Some(s),
            _ => None,
        }
    }

    fn as_int(&self) -> Option<&i64> {
        match self {
            Expr::Int(i) => Some(i),
            _ => None,
        }
    }
}
