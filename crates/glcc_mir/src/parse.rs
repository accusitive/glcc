//! MIR text parser (format v1, grammar in `docs/edg-mir-backend.md`).
//!
//! Line-oriented, explicitly typed, and designed for extension: a small
//! lexer turns each line into words and punctuation, a cursor walks it,
//! and per-line dispatch handles module items, labels, and instructions.
//! Later revisions (va/atomic/inline-asm ops) extend the instruction
//! dispatch and the module-item set without touching the core.

use std::collections::{HashMap, HashSet};
use std::fmt;

use smallvec::SmallVec;

use crate::builder::ModuleBuilder;
use crate::ir::{Block, Const, DataReloc, Def, Extern, FcmpPred, Func, Global, Inst, InstData, InstKind, IcmpPred, InvokeCallee, LpadClause, MemFlags, Module, Terminator, Value, ValueData};
use crate::ty::{FloatKind, GepIndex, Signature, Ty};

/// Parse failure: the 1-based source line it was detected on plus a
/// plain message. Never panics on malformed input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirParseError {
    pub line: u32,
    pub message: String,
}

impl fmt::Display for MirParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for MirParseError {}

pub fn parse_module(text: &str) -> Result<Module, MirParseError> {
    let mut lines = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("");
        if line.trim().is_empty() {
            continue;
        }
        lines.push((i as u32 + 1, lex(line)));
    }

    let mut builder = ModuleBuilder::new();
    let mut names = Names::default();
    let mut func_order: Vec<(Func, u32)> = Vec::new();
    let mut func_by_line: HashMap<u32, Func> = HashMap::new();

    for (line_no, toks) in &lines {
        let Tok::Word(word) = &toks[0] else {
            return Err(MirParseError { line: *line_no, message: "expected a directive or instruction".into() });
        };
        if !word.starts_with('.') {
            continue;
        }
        let mut c = Cursor { toks, pos: 1, line: *line_no, depth: 0 };
        match word.as_str() {
            ".func" => {
                let id = parse_func_decl(&mut builder, &mut names, &mut c)?;
                func_order.push((id, *line_no));
                func_by_line.insert(*line_no, id);
            }
            ".extern" => parse_extern_decl(&mut builder, &mut names, &mut c)?,
            ".dataextern" => parse_dataextern_decl(&mut builder, &mut names, &mut c)?,
            ".global" => parse_global_decl(&mut builder, &mut names, &mut c)?,
            ".init" => parse_init_decl(&mut builder, &mut names, &mut c)?,
            // Body directives and instructions are handled in pass two.
            ".params" => {}
            other => return Err(c.err(format!("unknown directive `{other}`"))),
        }
    }

    let mut fctx: Option<FnCtx> = None;
    for (line_no, toks) in &lines {
        let Tok::Word(first) = &toks[0] else {
            return Err(MirParseError { line: *line_no, message: "expected a directive or instruction".into() });
        };
        match first.as_str() {
            ".func" => {
                if let Some(ctx) = fctx.take() {
                    close_function(&mut builder.module, ctx)?;
                }
                let id = func_by_line[line_no];
                fctx = Some(FnCtx::new(&builder.module, id, *line_no));
            }
            ".extern" | ".dataextern" | ".global" | ".init" => {}
            ".params" => {
                let Some(ctx) = fctx.as_mut() else {
                    return Err(MirParseError { line: *line_no, message: ".params outside of a function".into() });
                };
                let mut c = Cursor { toks: &toks[1..], pos: 0, line: *line_no, depth: 0 };
                parse_params(&mut builder.module, ctx, &mut c)?;
            }
            _ => {
                let Some(ctx) = fctx.as_mut() else {
                    return Err(MirParseError { line: *line_no, message: format!("`{first}` outside of a function") });
                };
                let is_label_line = parse_label(first).is_some()
                    && matches!(toks.get(1), Some(Tok::Punct(':')) | Some(Tok::Punct('(')));
                if is_label_line {
                    let mut c = Cursor { toks: &toks[1..], pos: 0, line: *line_no, depth: 0 };
                    parse_label_line(&mut builder.module, ctx, first, &names, &mut c)?;
                } else {
                    let mut c = Cursor { toks, pos: 0, line: *line_no, depth: 0 };
                    parse_inst(&mut builder.module, ctx, &names, &mut c)?;
                }
            }
        }
    }
    if let Some(ctx) = fctx.take() {
        close_function(&mut builder.module, ctx)?;
    }
    resolve_data_relocs(&mut builder.module, &names)?;

    for (id, line) in &func_order {
        let errors = crate::verify::verify_func(&builder.module.funcs[*id]);
        if let Some(message) = errors.into_iter().next() {
            return Err(MirParseError { line: *line, message });
        }
    }
    Ok(builder.finish())
}

/// Deferred relocations (forward references in global initializers)
/// become ordinary Func/FuncExtern/Global relocations once every
/// declaration in the file is known.
fn resolve_data_relocs(module: &mut Module, names: &Names) -> Result<(), MirParseError> {
    let mut resolve = |reloc: &mut DataReloc| -> Result<(), (u32, String)> {
        let DataReloc::Deferred { name, addend, line } = reloc else {
            return Ok(());
        };
        let (name, addend, line) = (std::mem::take(name), *addend, *line);
        if let Some(&target) = names.funcs.get(name.as_str()) {
            *reloc = DataReloc::Func { target };
        } else if let Some(&target) = names.externs.get(name.as_str()) {
            *reloc = DataReloc::FuncExtern { target };
        } else if let Some(&target) = names.globals.get(name.as_str()) {
            *reloc = DataReloc::Global { target, addend };
        } else {
            return Err((line, format!("unknown relocation symbol @{name}")));
        }
        Ok(())
    };
    for global in module.globals.iter_mut() {
        for (_, reloc) in global.relocs.iter_mut() {
            resolve(reloc).map_err(|(line, message)| MirParseError { line, message })?;
        }
        for (_, reloc) in global.relocs32.iter_mut() {
            resolve(reloc).map_err(|(line, message)| MirParseError { line, message })?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    Word(String),
    Punct(char),
}

const PUNCT: &str = "(){}[]<>,:=";

fn lex(line: &str) -> Vec<Tok> {
    let mut toks = Vec::new();
    let mut chars = line.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }
        if PUNCT.contains(c) {
            chars.next();
            toks.push(Tok::Punct(c));
            continue;
        }
        let mut word = String::new();
        while let Some(&c) = chars.peek() {
            if c.is_whitespace() || PUNCT.contains(c) {
                if c == '>' && word == "-" {
                    word.push('>');
                    chars.next();
                }
                break;
            }
            word.push(c);
            chars.next();
        }
        toks.push(Tok::Word(word));
    }
    toks
}

struct Cursor<'t> {
    toks: &'t [Tok],
    pos: usize,
    line: u32,
    depth: u32,
}

impl<'t> Cursor<'t> {
    fn err(&self, message: impl Into<String>) -> MirParseError {
        MirParseError { line: self.line, message: message.into() }
    }

    fn done(&self) -> bool {
        self.pos >= self.toks.len()
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<Tok> {
        let tok = self.toks.get(self.pos).cloned();
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    fn peek_word(&self) -> Option<&str> {
        match self.peek() {
            Some(Tok::Word(w)) => Some(w),
            _ => None,
        }
    }

    fn word(&mut self) -> Result<String, MirParseError> {
        match self.next() {
            Some(Tok::Word(w)) => Ok(w),
            Some(Tok::Punct(p)) => Err(self.err(format!("expected a token, found `{p}`"))),
            None => Err(self.err("unexpected end of line")),
        }
    }

    fn eat_word(&mut self, word: &str) -> bool {
        if self.peek_word() == Some(word) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect_word(&mut self, word: &str) -> Result<(), MirParseError> {
        if self.eat_word(word) {
            Ok(())
        } else {
            Err(self.err(format!("expected `{word}`")))
        }
    }

    fn eat_punct(&mut self, p: char) -> bool {
        if self.peek() == Some(&Tok::Punct(p)) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect_punct(&mut self, p: char) -> Result<(), MirParseError> {
        if self.eat_punct(p) {
            Ok(())
        } else {
            Err(self.err(format!("expected `{p}`")))
        }
    }
}

fn expect_done(c: &Cursor) -> Result<(), MirParseError> {
    match c.peek() {
        Some(Tok::Word(w)) => Err(c.err(format!("unexpected trailing input starting at `{w}`"))),
        Some(Tok::Punct(p)) => Err(c.err(format!("unexpected trailing input starting at `{p}`"))),
        None => Ok(()),
    }
}

fn symbol(word: &str) -> Option<&str> {
    word.strip_prefix('@').filter(|s| !s.is_empty())
}

fn value_num(word: &str) -> Option<u32> {
    let rest = word.strip_prefix('%')?;
    if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

fn parse_label(word: &str) -> Option<u32> {
    let rest = word.strip_prefix('b')?;
    if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

fn is_digits(word: &str) -> bool {
    !word.is_empty() && word.bytes().all(|b| b.is_ascii_digit())
}

fn parse_i128_lit(word: &str) -> Option<i128> {
    let (negative, rest) = match word.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, word),
    };
    let value = match rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
        Some(hex) if is_hex(hex) => i128::from_str_radix(hex, 16).ok()?,
        Some(_) => return None,
        None if is_digits(rest) => rest.parse().ok()?,
        None => return None,
    };
    Some(if negative { -value } else { value })
}

fn parse_u64_lit(word: &str) -> Option<u64> {
    let value = parse_i128_lit(word)?;
    u64::try_from(value).ok()
}

fn is_hex(word: &str) -> bool {
    !word.is_empty() && word.bytes().all(|b| b.is_ascii_hexdigit())
}

fn parse_hex_bytes(word: &str) -> Result<Vec<u8>, String> {
    if word.len() % 2 != 0 {
        return Err("bytes literal must have an even number of hex digits".into());
    }
    if !is_hex(word) {
        return Err("bytes literal must be hexadecimal".into());
    }
    (0..word.len() / 2)
        .map(|i| u8::from_str_radix(&word[i * 2..i * 2 + 2], 16))
        .collect::<Result<Vec<u8>, _>>()
        .map_err(|_| "invalid hex byte".into())
}

/// `f64:0x3ff0...` at the cursor: `word` must name a float kind and be
/// followed by `:` and a hex bit pattern. Returns `None` when `word` is
/// not a float kind (the caller falls through to other interpretations).
fn parse_float_const(c: &mut Cursor, word: &str) -> Result<Option<(FloatKind, u128)>, MirParseError> {
    let kind = match word {
        "f32" => FloatKind::F32,
        "f64" => FloatKind::F64,
        "f80" => FloatKind::F80,
        "f128" => FloatKind::F128,
        _ => return Ok(None),
    };
    if c.peek() != Some(&Tok::Punct(':')) {
        return Ok(None);
    }
    c.expect_punct(':')?;
    let hex = c.word()?;
    let hex = hex.strip_prefix("0x").unwrap_or(&hex);
    let max_digits = match kind {
        FloatKind::F32 => 8,
        FloatKind::F64 => 16,
        FloatKind::F80 | FloatKind::F128 => 32,
    };
    if hex.len() > max_digits || !is_hex(&hex) {
        return Err(c.err(format!("bad {word} bit pattern `{hex}`")));
    }
    let bits = u128::from_str_radix(&hex, 16).map_err(|_| c.err(format!("bad {word} bit pattern `{hex}`")))?;
    Ok(Some((kind, bits)))
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

fn int_ty_from_word(word: &str) -> Option<Ty> {
    let rest = word.strip_prefix('i')?;
    if !is_digits(rest) {
        return None;
    }
    let bits: u32 = rest.parse().ok()?;
    if bits == 0 {
        return None;
    }
    Some(if bits > crate::ty::MAX_SCALAR_INT_BITS { Ty::WideInt { bits } } else { Ty::Int { bits } })
}

fn parse_ty(c: &mut Cursor) -> Result<Ty, MirParseError> {
    c.depth += 1;
    if c.depth > 100 {
        return Err(c.err("type nesting too deep"));
    }
    let ty = parse_ty_inner(c);
    c.depth -= 1;
    ty
}

fn parse_ty_inner(c: &mut Cursor) -> Result<Ty, MirParseError> {
    if c.eat_punct('[') {
        let len_word = c.word()?;
        let len = parse_u64_lit(&len_word).ok_or_else(|| c.err(format!("bad array length `{len_word}`")))?;
        c.expect_word("x")?;
        let elem = parse_ty(c)?;
        c.expect_punct(']')?;
        return Ok(Ty::Array { elem: Box::new(elem), len });
    }
    let word = c.word()?;
    parse_ty_word(c, &word)
}

fn parse_ty_word(c: &mut Cursor, word: &str) -> Result<Ty, MirParseError> {
    match word {
        "void" => return Ok(Ty::Void),
        "bool" => return Ok(Ty::Bool),
        "f32" => return Ok(Ty::Float(FloatKind::F32)),
        "f64" => return Ok(Ty::Float(FloatKind::F64)),
        "f80" => return Ok(Ty::Float(FloatKind::F80)),
        "f128" => return Ok(Ty::Float(FloatKind::F128)),
        "ptr" => {
            if c.eat_punct('(') {
                let pointee = parse_ty(c)?;
                c.expect_punct(')')?;
                return Ok(Ty::ptr(pointee));
            }
            c.expect_punct('<')?;
            let digits = match c.peek() {
                Some(Tok::Word(w)) if is_digits(w) => w.clone(),
                _ => String::new(),
            };
            if !digits.is_empty() {
                if !c.peek_ahead_is(1, '>') {
                    return Err(c.err("ptr needs a pointee: ptr<T> or ptr<bits>(T)"));
                }
                c.next();
                c.expect_punct('>')?;
                c.expect_punct('(')?;
                let pointee = parse_ty(c)?;
                c.expect_punct(')')?;
                let bits: u32 = digits.parse().map_err(|_| c.err(format!("bad pointer width `{digits}`")))?;
                return Ok(Ty::ptr_bits(pointee, bits));
            }
            let pointee = parse_ty(c)?;
            c.expect_punct('>')?;
            return Ok(Ty::ptr(pointee));
        }
        "struct" => {
            c.expect_punct('<')?;
            let size_word = c.word()?;
            let size = parse_u64_lit(&size_word).ok_or_else(|| c.err(format!("bad struct size `{size_word}`")))?;
            c.expect_punct('>')?;
            c.expect_punct('{')?;
            let mut fields = Vec::new();
            let mut offsets = Vec::new();
            loop {
                if c.eat_punct('}') {
                    break;
                }
                let off_word = c.word()?;
                let off = parse_u64_lit(&off_word).ok_or_else(|| c.err(format!("bad field offset `{off_word}`")))?;
                c.expect_punct(':')?;
                fields.push(parse_ty(c)?);
                offsets.push(off);
                if !c.eat_punct(',') {
                    c.expect_punct('}')?;
                    break;
                }
            }
            return Ok(Ty::Struct { fields, offsets, size });
        }
        _ => {}
    }
    if let Some(ty) = vector_ty_from_word(word) {
        return Ok(ty);
    }
    if let Some(ty) = int_ty_from_word(word) {
        return Ok(ty);
    }
    Err(c.err(format!("unknown type `{word}`")))
}

impl<'t> Cursor<'t> {
    fn peek_ahead_is(&self, offset: usize, p: char) -> bool {
        self.toks.get(self.pos + offset) == Some(&Tok::Punct(p))
    }
}

/// `v<lanes>x<lanety>`: a SIMD vector type as displayed by the printer.
fn vector_ty_from_word(word: &str) -> Option<Ty> {
    let rest = word.strip_prefix('v')?;
    let (lanes, lane) = rest.split_once('x')?;
    let lanes: u32 = lanes.parse().ok()?;
    let elem = match lane {
        "f32" => Ty::Float(FloatKind::F32),
        "f64" => Ty::Float(FloatKind::F64),
        other => int_ty_from_word(other)?,
    };
    Some(Ty::vector(elem, lanes))
}

fn parse_sig(c: &mut Cursor) -> Result<Signature, MirParseError> {
    c.expect_word("sig")?;
    c.expect_punct('=')?;
    c.expect_punct('(')?;
    let mut params = Vec::new();
    let mut variadic = false;
    loop {
        if c.eat_punct(')') {
            break;
        }
        if c.eat_word("...") {
            variadic = true;
            c.expect_punct(')')?;
            break;
        }
        params.push(parse_ty(c)?);
        if !c.eat_punct(',') {
            c.expect_punct(')')?;
            break;
        }
    }
    c.expect_word("->")?;
    let ret = parse_ty(c)?;
    Ok(Signature { params, ret, variadic })
}

// ---------------------------------------------------------------------------
// Module items
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Names {
    funcs: HashMap<String, Func>,
    externs: HashMap<String, Extern>,
    globals: HashMap<String, Global>,
}

fn parse_func_decl(builder: &mut ModuleBuilder, names: &mut Names, c: &mut Cursor) -> Result<Func, MirParseError> {
    let word = c.word()?;
    let name = symbol(&word).ok_or_else(|| c.err(format!("expected @name after .func, found `{word}`")))?;
    let mut local = false;
    let mut weak = false;
    let mut inline = false;
    let mut force_inline = false;
    let mut personality: Option<Extern> = None;
    loop {
        let flag = c.peek_word().map(str::to_string);
        match flag.as_deref() {
            Some("local") => {
                c.next();
                local = true;
            }
            Some("weak") => {
                c.next();
                weak = true;
            }
            Some("inline") => {
                c.next();
                inline = true;
            }
            Some("force_inline") => {
                c.next();
                inline = true;
                force_inline = true;
            }
            Some("sig") => break,
            Some(other) => return Err(c.err(format!("unexpected `.func` flag `{other}`"))),
            None => return Err(c.err(".func is missing sig=")),
        }
    }
    let sig = parse_sig(c)?;
    if c.eat_word("personality") {
        c.expect_punct('=')?;
        let word = c.word()?;
        let pname = symbol(&word)
            .ok_or_else(|| c.err(format!("expected @name after personality=, found `{word}`")))?;
        personality = Some(
            *names
                .externs
                .get(pname)
                .ok_or_else(|| c.err(format!("unknown personality @{pname} (missing .extern)")))?,
        );
    }
    expect_done(c)?;
    if names.funcs.contains_key(name) {
        return Err(c.err(format!("duplicate function @{name}")));
    }
    let (id, _) = builder.declare(name, sig);
    let func = builder.func_mut(id);
    func.local = local;
    func.weak = weak;
    func.inline = inline;
    func.force_inline = force_inline;
    func.personality = personality;
    names.funcs.insert(name.to_string(), id);
    Ok(id)
}

fn parse_extern_decl(builder: &mut ModuleBuilder, names: &mut Names, c: &mut Cursor) -> Result<(), MirParseError> {
    let word = c.word()?;
    let name = symbol(&word).ok_or_else(|| c.err(format!("expected @name after .extern, found `{word}`")))?;
    let mut sig: Option<Signature> = None;
    let mut weak = false;
    while !c.done() {
        let item = c.peek_word().map(str::to_string);
        match item.as_deref() {
            Some("sig") => sig = Some(parse_sig(c)?),
            Some("weak") => {
                c.next();
                weak = true;
            }
            Some(other) => return Err(c.err(format!("unexpected `.extern` item `{other}`"))),
            None => return Err(c.err(format!("unexpected token after .extern @{}", name))),
        }
    }
    let sig = sig.ok_or_else(|| c.err(".extern is missing sig="))?;
    expect_done(c)?;
    if names.externs.contains_key(name) {
        return Err(c.err(format!("duplicate extern @{name}")));
    }
    let id = builder.module.push_extern(name, sig);
    builder.module.externs[id].weak = weak;
    names.externs.insert(name.to_string(), id);
    Ok(())
}

/// `.init @name`: register a function defined in this file as a
/// pre-`main` constructor. The target must already be declared
/// (single-pass resolution, like reloc symbols); a function may be
/// registered at most once. List order is the emission order.
fn parse_init_decl(builder: &mut ModuleBuilder, names: &mut Names, c: &mut Cursor) -> Result<(), MirParseError> {
    let word = c.word()?;
    let name = symbol(&word).ok_or_else(|| c.err(format!("expected @name after .init, found `{word}`")))?;
    expect_done(c)?;
    let Some(&func) = names.funcs.get(name) else {
        if names.externs.contains_key(name) {
            return Err(c.err(format!(".init target @{name} is an .extern; it must be a function defined in this file")));
        }
        return Err(c.err(format!("unknown .init target @{name} (missing .func)")));
    };
    let sig = &builder.module.funcs[func].sig;
    if !sig.params.is_empty() || !sig.ret.is_void() {
        return Err(c.err(format!(".init target @{name} must take no parameters and return void")));
    }
    if builder.module.init_funcs.contains(&func) {
        return Err(c.err(format!("duplicate .init entry @{name}")));
    }
    builder.module.init_funcs.push(func);
    Ok(())
}

fn parse_dataextern_decl(builder: &mut ModuleBuilder, names: &mut Names, c: &mut Cursor) -> Result<(), MirParseError> {
    let word = c.word()?;
    let name = symbol(&word).ok_or_else(|| c.err(format!("expected @name after .dataextern, found `{word}`")))?;
    c.expect_word("ty")?;
    c.expect_punct('=')?;
    let ty = parse_ty(c)?;
    expect_done(c)?;
    if names.globals.contains_key(name) {
        return Err(c.err(format!("duplicate data symbol @{name}")));
    }
    let id = builder.module.push_data_extern(name, ty);
    names.globals.insert(name.to_string(), id);
    Ok(())
}

fn parse_global_decl(builder: &mut ModuleBuilder, names: &mut Names, c: &mut Cursor) -> Result<(), MirParseError> {
    let word = c.word()?;
    let name = symbol(&word).ok_or_else(|| c.err(format!("expected @name after .global, found `{word}`")))?;
    let mut local = false;
    let mut weak = false;
    let mut tls = false;
    let mut writable = true;
    let mut align: Option<u64> = None;
    let mut ty: Option<Ty> = None;
    let mut init: Option<Option<Const>> = None;
    let mut relocs: Vec<(u64, DataReloc)> = Vec::new();
    while !c.done() {
        let item = c.peek_word().map(str::to_string);
        match item.as_deref() {
            Some("local") => {
                c.next();
                local = true;
            }
            Some("weak") => {
                c.next();
                weak = true;
            }
            Some("tls") => {
                c.next();
                tls = true;
            }
            Some("const") => {
                c.next();
                writable = false;
            }
            Some("align") => {
                c.next();
                c.expect_punct('=')?;
                let word = c.word()?;
                align = Some(parse_u64_lit(&word).ok_or_else(|| c.err(format!("bad alignment `{word}`")))?);
            }
            Some("ty") => {
                c.next();
                c.expect_punct('=')?;
                ty = Some(parse_ty(c)?);
            }
            Some("init") => {
                c.next();
                c.expect_punct('=')?;
                let ty = ty.as_ref().ok_or_else(|| c.err(".global init= needs ty= first"))?;
                init = Some(parse_global_init(c, ty, names, &mut relocs)?);
            }
            Some(other) => return Err(c.err(format!("unexpected `.global` item `{other}`"))),
            None => return Err(c.err(format!("unexpected token after .global @{}", name))),
        }
    }
    let ty = ty.ok_or_else(|| c.err(format!(".global @{name} is missing ty=")))?;
    let init = init.ok_or_else(|| c.err(format!(".global @{name} is missing init=")))?;
    expect_done(c)?;
    if names.globals.contains_key(name) {
        return Err(c.err(format!("duplicate data symbol @{name}")));
    }
    let id = builder.module.push_global(name, ty, init);
    let global = &mut builder.module.globals[id];
    global.local = local;
    global.weak = weak;
    global.tls = tls;
    global.writable = writable;
    global.align = align;
    global.relocs = relocs;
    names.globals.insert(name.to_string(), id);
    Ok(())
}

fn parse_global_init(c: &mut Cursor, ty: &Ty, names: &Names, relocs: &mut Vec<(u64, DataReloc)>) -> Result<Option<Const>, MirParseError> {
    if c.eat_word("undef") {
        return Ok(None);
    }
    if c.eat_word("zeroinit") {
        return Ok(Some(Const::Zero));
    }
    if c.eat_punct('[') {
        let size = ty.size_bytes();
        if size > 1 << 32 {
            return Err(c.err("aggregate initializer exceeds 4 GiB"));
        }
        let mut image = vec![0u8; size as usize];
        parse_aggregate_init(c, ty, &mut image, names, relocs)?;
        return Ok(Some(Const::Bytes(image)));
    }
    Ok(Some(parse_scalar_init(c, ty, names, relocs, 0)?))
}

/// `reloc @sym` or `reloc @sym+<offset>` (no whitespace before the
/// addend, per the emitter). The symbol kind picks the relocation
/// flavor: .func → Func, .extern → FuncExtern, .global/.dataextern →
/// Global with the addend.
fn parse_reloc(names: &Names, c: &mut Cursor) -> Result<DataReloc, MirParseError> {
    let word = c.word().map_err(|_| c.err("reloc needs a @symbol"))?;
    let body = symbol(&word).ok_or_else(|| c.err(format!("expected @symbol after reloc, found `{word}`")))?;
    let (name, addend) = match body.split_once('+') {
        Some((name, off)) => {
            let offset = parse_u64_lit(off).ok_or_else(|| c.err(format!("bad relocation addend `+{off}`")))?;
            let addend = i64::try_from(offset).map_err(|_| c.err(format!("relocation addend `+{off}` out of range")))?;
            (name, addend)
        }
        None => (body, 0i64),
    };
    if let Some(&target) = names.funcs.get(name) {
        return Ok(DataReloc::Func { target });
    }
    if let Some(&target) = names.externs.get(name) {
        return Ok(DataReloc::FuncExtern { target });
    }
    if let Some(&target) = names.globals.get(name) {
        return Ok(DataReloc::Global { target, addend });
    }
    // Not declared yet: a forward reference (the emitter may order a
    // global's users before its definition). Deferred until the whole
    // file has been parsed.
    Ok(DataReloc::Deferred { name: name.to_string(), addend, line: c.line })
}

/// Relocations may only patch pointer-width slots (`long` covers C's
/// `(long)&obj` spelling), matching where glcc_lower_mir emits them.
fn check_reloc_type(c: &Cursor, ty: &Ty) -> Result<(), MirParseError> {
    match ty {
        Ty::Ptr { .. } | Ty::Int { bits: 64 } => Ok(()),
        _ => Err(c.err(format!("reloc initializer for non-pointer type {ty:?}"))),
    }
}

fn parse_scalar_init(c: &mut Cursor, ty: &Ty, names: &Names, relocs: &mut Vec<(u64, DataReloc)>, off: u64) -> Result<Const, MirParseError> {
    let word = c.word()?;
    if word == "reloc" {
        let reloc = parse_reloc(names, c)?;
        check_reloc_type(c, ty)?;
        let size = ty.size_bytes() as usize;
        relocs.push((off, reloc));
        return Ok(Const::Bytes(vec![0u8; size]));
    }
    if word == "null" {
        if !matches!(ty, Ty::Ptr { .. }) {
            return Err(c.err("null initializer for a non-pointer"));
        }
        return Ok(Const::Null);
    }
    if word == "bytes" {
        c.expect_punct('<')?;
        let hex = c.word()?;
        c.expect_punct('>')?;
        return Ok(Const::Bytes(parse_hex_bytes(&hex).map_err(|e| c.err(e))?));
    }
    if let Some((kind, bits)) = parse_float_const(c, &word)? {
        match ty {
            Ty::Float(k) if *k == kind => {}
            _ => return Err(c.err(format!("float initializer kind does not match type {ty:?}"))),
        }
        return Ok(Const::Float { kind, bits });
    }
    if let Some(value) = parse_i128_lit(&word) {
        let bits = match ty {
            Ty::Int { bits } | Ty::WideInt { bits } => *bits,
            Ty::Bool => 1,
            Ty::Ptr { bits, .. } => *bits,
            _ => return Err(c.err(format!("integer initializer for non-scalar type {ty:?}"))),
        };
        return Ok(Const::Int { bits, value }.canonical());
    }
    Err(c.err(format!("expected an initializer constant, found `{word}`")))
}

/// Fill a dense aggregate image from a `[c0, c1, ...]` list; elements
/// land at the type's own offsets (struct) or strides (array), C's
/// remainder rule covers omitted elements with zero bytes.
fn parse_aggregate_init(c: &mut Cursor, ty: &Ty, image: &mut [u8], names: &Names, relocs: &mut Vec<(u64, DataReloc)>) -> Result<(), MirParseError> {
    c.depth += 1;
    if c.depth > 100 {
        return Err(c.err("initializer nesting too deep"));
    }
    let result = parse_aggregate_init_inner(c, ty, image, names, relocs);
    c.depth -= 1;
    result
}

fn parse_aggregate_init_inner(c: &mut Cursor, ty: &Ty, image: &mut [u8], names: &Names, relocs: &mut Vec<(u64, DataReloc)>) -> Result<(), MirParseError> {
    let mut index = 0u64;
    loop {
        if c.eat_punct(']') {
            return Ok(());
        }
        let (off, elem) = match ty {
            Ty::Array { elem, len } => {
                if index >= *len {
                    return Err(c.err("more initializers than array elements"));
                }
                (index * elem.size_bytes(), elem.as_ref().clone())
            }
            Ty::Struct { fields, offsets, .. } => {
                if index as usize >= fields.len() {
                    return Err(c.err("more initializers than struct fields"));
                }
                (offsets[index as usize], fields[index as usize].clone())
            }
            _ => return Err(c.err("aggregate initializer for a non-aggregate type")),
        };
        write_init_element(c, &elem, image, off, names, relocs)?;
        index += 1;
        if !c.eat_punct(',') {
            c.expect_punct(']')?;
            return Ok(());
        }
    }
}

fn write_init_element(c: &mut Cursor, ty: &Ty, image: &mut [u8], off: u64, names: &Names, relocs: &mut Vec<(u64, DataReloc)>) -> Result<(), MirParseError> {
    if c.eat_punct('[') {
        return parse_aggregate_init(c, ty, image, names, relocs);
    }
    if c.eat_word("zeroinit") || c.eat_word("undef") || c.eat_word("null") {
        return Ok(());
    }
    if c.eat_word("reloc") {
        let reloc = parse_reloc(names, c)?;
        check_reloc_type(c, ty)?;
        relocs.push((off, reloc));
        return Ok(());
    }
    let size = ty.size_bytes() as usize;
    let end = off as usize + size;
    if end > image.len() {
        return Err(c.err("initializer does not fit the global's type"));
    }
    let word = c.word()?;
    if word == "bytes" {
        c.expect_punct('<')?;
        let hex = c.word()?;
        c.expect_punct('>')?;
        let bytes = parse_hex_bytes(&hex).map_err(|e| c.err(e))?;
        if bytes.len() != size {
            return Err(c.err("bytes literal length does not match the element size"));
        }
        image[off as usize..end].copy_from_slice(&bytes);
        return Ok(());
    }
    if let Some((kind, bits)) = parse_float_const(c, &word)? {
        match ty {
            Ty::Float(k) if *k == kind => {}
            _ => return Err(c.err(format!("float initializer kind does not match type {ty:?}"))),
        }
        image[off as usize..end].copy_from_slice(&bits.to_le_bytes()[..size]);
        return Ok(());
    }
    if let Some(value) = parse_i128_lit(&word) {
        image[off as usize..end].copy_from_slice(&(value as u128).to_le_bytes()[..size]);
        return Ok(());
    }
    Err(c.err(format!("expected an initializer constant, found `{word}`")))
}

// ---------------------------------------------------------------------------
// Function bodies
// ---------------------------------------------------------------------------

struct FnCtx {
    func: Func,
    /// Every `%n` mentioned in the function, bound to its arena value.
    values: HashMap<u32, Value>,
    /// The `%n` that have a definition (param or instruction result).
    defined: HashSet<u32>,
    /// First-use line of every `%n`, for use-without-definition errors.
    use_lines: HashMap<u32, u32>,
    /// Arena values created by a forward reference and not yet defined.
    unresolved: HashSet<Value>,
    /// Block params typed from a pending branch argument, awaiting the
    /// argument's definition to learn their type.
    ty_deps: HashMap<Value, Vec<(Block, u32)>>,
    /// Label number to arena block.
    blocks: HashMap<u32, Block>,
    /// Labels that appeared as a label line.
    labeled: HashSet<u32>,
    block_line: HashMap<Block, u32>,
    cur: Option<Block>,
    last_inst_line: Option<u32>,
}

impl FnCtx {
    fn new(module: &Module, func: Func, line: u32) -> Self {
        let entry = module.funcs[func].entry;
        let mut block_line = HashMap::new();
        block_line.insert(entry, line);
        FnCtx {
            func,
            values: HashMap::new(),
            defined: HashSet::new(),
            use_lines: HashMap::new(),
            unresolved: HashSet::new(),
            ty_deps: HashMap::new(),
            blocks: HashMap::from([(0, entry)]),
            labeled: HashSet::new(),
            block_line,
            cur: Some(entry),
            last_inst_line: None,
        }
    }
}

fn close_function(module: &mut Module, ctx: FnCtx) -> Result<(), MirParseError> {
    let entry = module.funcs[ctx.func].entry;
    for (&label, &b) in &ctx.blocks {
        if module.funcs[ctx.func].blocks[b].term.is_none() {
            let line = if Some(b) == ctx.cur { ctx.last_inst_line.unwrap_or(ctx.block_line[&b]) } else { ctx.block_line[&b] };
            return Err(MirParseError { line, message: format!("block b{label} has no terminator") });
        }
        if b != entry && !ctx.labeled.contains(&label) {
            return Err(MirParseError { line: ctx.block_line[&b], message: format!("block b{label} is referenced but never labeled") });
        }
    }
    let mut unbound: Vec<(u32, u32)> = ctx
        .values
        .iter()
        .filter(|(n, _)| !ctx.defined.contains(n))
        .map(|(n, _)| (*n, ctx.use_lines[n]))
        .collect();
    unbound.sort_unstable();
    if let Some((n, line)) = unbound.first() {
        return Err(MirParseError { line: *line, message: format!("%{n} is used but never defined") });
    }
    Ok(())
}

fn value_ref(module: &mut Module, ctx: &mut FnCtx, n: u32, line: u32) -> Result<Value, MirParseError> {
    ctx.use_lines.entry(n).or_insert(line);
    if let Some(&v) = ctx.values.get(&n) {
        return Ok(v);
    }
    let v = module.funcs[ctx.func].values.push(ValueData { ty: Ty::Void, def: Def::Const(Const::Null) });
    ctx.values.insert(n, v);
    ctx.unresolved.insert(v);
    Ok(v)
}

fn define_value(module: &mut Module, ctx: &mut FnCtx, n: u32, ty: Ty, def: Def, line: u32) -> Result<Value, MirParseError> {
    if ctx.defined.contains(&n) {
        return Err(MirParseError { line, message: format!("double definition of %{n}") });
    }
    let v = match ctx.values.get(&n) {
        Some(&v) => v,
        None => {
            let v = module.funcs[ctx.func].values.push(ValueData { ty: Ty::Void, def: Def::Const(Const::Null) });
            ctx.values.insert(n, v);
            v
        }
    };
    module.funcs[ctx.func].values[v] = ValueData { ty: ty.clone(), def };
    ctx.defined.insert(n);
    ctx.unresolved.remove(&v);
    if let Some(deps) = ctx.ty_deps.remove(&v) {
        for (b, i) in deps {
            let param = module.funcs[ctx.func].blocks[b].params[i as usize];
            module.funcs[ctx.func].values[param].ty = ty.clone();
        }
    }
    Ok(v)
}

/// The arena block for a label, materializing it (with the branch's
/// argument types as param types) on first reference. Arity is checked
/// once the params exist; an argument whose type is still pending skips
/// the type check and defers the param's type via `ty_deps`.
fn block_ref(module: &mut Module, ctx: &mut FnCtx, label: u32, args: &[Value], line: u32) -> Result<Block, MirParseError> {
    if let Some(&b) = ctx.blocks.get(&label) {
        let func = &mut module.funcs[ctx.func];
        if args.len() != func.blocks[b].params.len() {
            return Err(MirParseError {
                line,
                message: format!("branch to b{label} passes {} arguments but the block takes {}", args.len(), func.blocks[b].params.len()),
            });
        }
        for (i, &a) in args.iter().enumerate() {
            let arg_ty = &func.values[a].ty;
            let param_ty = &func.values[func.blocks[b].params[i]].ty;
            if !matches!(arg_ty, Ty::Void) && arg_ty != param_ty {
                let arg_ty = arg_ty.clone();
                let param_ty = param_ty.clone();
                return Err(MirParseError {
                    line,
                    message: format!("branch argument {} to b{label} has type {arg_ty:?} but the block param is {param_ty:?}", i + 1),
                });
            }
        }
        return Ok(b);
    }
    let func = &mut module.funcs[ctx.func];
    let tys: Vec<Ty> = args.iter().map(|&a| func.values[a].ty.clone()).collect();
    let b = func.push_block(tys);
    for (i, &a) in args.iter().enumerate() {
        if ctx.unresolved.contains(&a) {
            ctx.ty_deps.entry(a).or_default().push((b, i as u32));
        }
    }
    ctx.blocks.insert(label, b);
    ctx.block_line.insert(b, line);
    Ok(b)
}

fn parse_label_line(module: &mut Module, ctx: &mut FnCtx, word: &str, names: &Names, c: &mut Cursor) -> Result<(), MirParseError> {
    let line = c.line;
    let label = parse_label(word).expect("label checked by caller");
    let mut decls: Vec<(u32, Ty)> = Vec::new();
    let mut declares_params = false;
    if c.eat_punct('(') {
        declares_params = true;
        if !c.eat_punct(')') {
            loop {
                let param = c.word()?;
                let n = value_num(&param).ok_or_else(|| c.err(format!("expected %n block parameter, found `{param}`")))?;
                c.expect_punct(':')?;
                let ty = parse_ty(c)?;
                decls.push((n, ty));
                if !c.eat_punct(',') {
                    c.expect_punct(')')?;
                    break;
                }
            }
        }
    }
    let mut lpad: Option<SmallVec<[LpadClause; 4]>> = None;
    if c.eat_word(".lpad") {
        c.expect_punct('[')?;
        let mut clauses = SmallVec::new();
        loop {
            if c.eat_punct(']') {
                break;
            }
            let word = c.word()?;
            match word.as_str() {
                "catchall" => clauses.push(LpadClause::CatchAll),
                "cleanup" => clauses.push(LpadClause::Cleanup),
                "catch" => {
                    let word = c.word()?;
                    let gname = symbol(&word)
                        .ok_or_else(|| c.err(format!("expected @name after catch, found `{word}`")))?;
                    let global = *names
                        .globals
                        .get(gname)
                        .ok_or_else(|| c.err(format!("catch of unknown typeinfo @{gname} (missing .dataextern)")))?;
                    clauses.push(LpadClause::Catch(global));
                }
                other => return Err(c.err(format!("unknown .lpad clause `{other}`"))),
            }
            if !c.eat_punct(',') {
                c.expect_punct(']')?;
                break;
            }
        }
        lpad = Some(clauses);
    }
    c.expect_punct(':')?;
    expect_done(c)?;

    if ctx.labeled.contains(&label) {
        return Err(c.err(format!("duplicate label b{label}")));
    }
    if let Some(cur) = ctx.cur {
        if ctx.blocks.get(&label) != Some(&cur) && module.funcs[ctx.func].blocks[cur].term.is_none() {
            let at = ctx.last_inst_line.unwrap_or(ctx.block_line[&cur]);
            return Err(MirParseError { line: at, message: format!("block b{label} starts while the previous block has no terminator") });
        }
    }
    let b = match ctx.blocks.get(&label) {
        Some(&b) => b,
        None => {
            let tys: Vec<Ty> = decls.iter().map(|(_, ty)| ty.clone()).collect();
            let b = module.funcs[ctx.func].push_block(tys);
            ctx.blocks.insert(label, b);
            ctx.block_line.insert(b, line);
            b
        }
    };
    if let Some(clauses) = lpad {
        module.funcs[ctx.func].blocks[b].lpad = Some(clauses);
    }
    let nparams = module.funcs[ctx.func].blocks[b].params.len();
    if declares_params && nparams != decls.len() {
        return Err(c.err(format!("block b{label} takes {nparams} branch arguments but its label declares {}", decls.len())));
    }
    for (i, (n, ty)) in decls.into_iter().enumerate() {
        let existing = module.funcs[ctx.func].values[module.funcs[ctx.func].blocks[b].params[i]].ty.clone();
        if !matches!(existing, Ty::Void) && existing != ty {
            return Err(c.err(format!("block b{label} param {} is {existing:?} per an earlier branch argument but declared {ty:?}", i + 1)));
        }
        let v = define_value(module, ctx, n, ty, Def::Param(b, i as u32), line)?;
        module.funcs[ctx.func].blocks[b].params[i] = v;
    }
    ctx.labeled.insert(label);
    ctx.cur = Some(b);
    ctx.last_inst_line = None;
    Ok(())
}

fn parse_params(module: &mut Module, ctx: &mut FnCtx, c: &mut Cursor) -> Result<(), MirParseError> {
    let line = c.line;
    let entry = module.funcs[ctx.func].entry;
    if ctx.cur != Some(entry) {
        return Err(c.err(".params must precede any other block"));
    }
    let mut index = module.funcs[ctx.func].blocks[entry].params.len() as u32;
    while !c.done() {
        let word = c.word()?;
        let n = value_num(&word).ok_or_else(|| c.err(format!("expected a %n parameter, found `{word}`")))?;
        c.expect_punct(':')?;
        let ty = parse_ty(c)?;
        let v = define_value(module, ctx, n, ty, Def::Param(entry, index), line)?;
        module.funcs[ctx.func].blocks[entry].params.push(v);
        index += 1;
        if !c.eat_punct(',') {
            break;
        }
    }
    expect_done(c)
}

// ---------------------------------------------------------------------------
// Instructions
// ---------------------------------------------------------------------------

/// An operand as written: a value reference or a constant awaiting its
/// context type.
enum RawOperand {
    Val(Value),
    Int(i128),
    Float { kind: FloatKind, bits: u128 },
    Null,
    Undef,
    Zero,
    Bytes(Vec<u8>),
}

fn parse_operand(module: &mut Module, ctx: &mut FnCtx, c: &mut Cursor) -> Result<RawOperand, MirParseError> {
    let word = c.word()?;
    if let Some(n) = value_num(&word) {
        return Ok(RawOperand::Val(value_ref(module, ctx, n, c.line)?));
    }
    match word.as_str() {
        "null" => return Ok(RawOperand::Null),
        "undef" => return Ok(RawOperand::Undef),
        "zeroinit" => return Ok(RawOperand::Zero),
        "bytes" => {
            c.expect_punct('<')?;
            let hex = c.word()?;
            c.expect_punct('>')?;
            return Ok(RawOperand::Bytes(parse_hex_bytes(&hex).map_err(|e| c.err(e))?));
        }
        _ => {}
    }
    if let Some((kind, bits)) = parse_float_const(c, &word)? {
        return Ok(RawOperand::Float { kind, bits });
    }
    if let Some(value) = parse_i128_lit(&word) {
        return Ok(RawOperand::Int(value));
    }
    Err(c.err(format!("expected an operand, found `{word}`")))
}

fn materialize(module: &mut Module, ctx: &FnCtx, raw: RawOperand, expected: Option<&Ty>, line: u32) -> Result<Value, MirParseError> {
    let err = |message: String| MirParseError { line, message };
    let func = &mut module.funcs[ctx.func];
    Ok(match raw {
        RawOperand::Val(v) => v,
        RawOperand::Int(value) => {
            let ty = expected.ok_or_else(|| err("integer constant has no type from context".into()))?;
            let bits = match ty {
                Ty::Int { bits } | Ty::WideInt { bits } => *bits,
                Ty::Bool => {
                    if value != 0 && value != 1 {
                        return Err(err("boolean constant must be 0 or 1".to_string()));
                    }
                    1
                }
                _ => return Err(err(format!("integer constant used where {ty:?} is expected"))),
            };
            let konst = Const::Int { bits, value }.canonical();
            func.values.push(ValueData { ty: ty.clone(), def: Def::Const(konst) })
        }
        RawOperand::Float { kind, bits } => {
            match expected {
                Some(Ty::Float(k)) if *k == kind => {}
                Some(other) => return Err(err(format!("float constant used where {other:?} is expected"))),
                None => {}
            }
            func.values.push(ValueData { ty: Ty::Float(kind), def: Def::Const(Const::Float { kind, bits }) })
        }
        RawOperand::Null => {
            let ty = match expected {
                Some(ty @ Ty::Ptr { .. }) => ty.clone(),
                Some(other) => return Err(err(format!("null used where {other:?} is expected"))),
                None => Ty::ptr(Ty::Void),
            };
            func.values.push(ValueData { ty, def: Def::Const(Const::Null) })
        }
        RawOperand::Undef => {
            let ty = expected.ok_or_else(|| err("undef operand has no type from context".into()))?;
            func.values.push(ValueData { ty: ty.clone(), def: Def::Const(Const::Undef) })
        }
        RawOperand::Zero => {
            let ty = expected.ok_or_else(|| err("zeroinit operand has no type from context".into()))?;
            func.values.push(ValueData { ty: ty.clone(), def: Def::Const(Const::Zero) })
        }
        RawOperand::Bytes(bytes) => {
            let ty = expected.ok_or_else(|| err("bytes operand has no type from context".into()))?;
            func.values.push(ValueData { ty: ty.clone(), def: Def::Const(Const::Bytes(bytes)) })
        }
    })
}

fn raw_val_ty(module: &Module, ctx: &FnCtx, raw: &RawOperand) -> Option<Ty> {
    match raw {
        RawOperand::Val(v) => Some(module.funcs[ctx.func].values[*v].ty.clone()),
        _ => None,
    }
}

fn need_val_operand(module: &mut Module, ctx: &mut FnCtx, c: &mut Cursor, what: &str) -> Result<Value, MirParseError> {
    match parse_operand(module, ctx, c)? {
        RawOperand::Val(v) => Ok(v),
        _ => Err(c.err(format!("{what} must be a value reference"))),
    }
}

fn pointee_ty(module: &Module, ctx: &FnCtx, addr: Value) -> Option<Ty> {
    match &module.funcs[ctx.func].values[addr].ty {
        Ty::Ptr { pointee, .. } => Some(pointee.as_ref().clone()),
        _ => None,
    }
}

fn elem_ty_of(module: &Module, ctx: &FnCtx, agg: Value, index: u32) -> Option<Ty> {
    match &module.funcs[ctx.func].values[agg].ty {
        Ty::Struct { fields, .. } => fields.get(index as usize).cloned(),
        Ty::Array { elem, .. } => Some(elem.as_ref().clone()),
        _ => None,
    }
}

fn need_result(result: Option<(u32, Ty)>, op: &str, line: u32) -> Result<(u32, Ty), MirParseError> {
    result.ok_or_else(|| MirParseError { line, message: format!("{op} needs a result: %n:ty = {op} ...") })
}

fn no_result(result: &Option<(u32, Ty)>, op: &str, line: u32) -> Result<(), MirParseError> {
    if result.is_some() {
        Err(MirParseError { line, message: format!("{op} does not produce a value") })
    } else {
        Ok(())
    }
}

fn push_inst(module: &mut Module, ctx: &FnCtx, block: Block, kind: InstKind, result: Option<Value>) -> Inst {
    let func = &mut module.funcs[ctx.func];
    let id = Inst::from_usize(func.insts.len());
    func.insts.push(InstData { block, kind, result, loc: None });
    func.blocks[block].insts.push(id);
    id
}

fn emit_value(module: &mut Module, ctx: &mut FnCtx, block: Block, kind: InstKind, n: u32, ty: Ty, line: u32) -> Result<Value, MirParseError> {
    let v = define_value(module, ctx, n, ty, Def::Const(Const::Null), line)?;
    let inst = push_inst(module, ctx, block, kind, Some(v));
    module.funcs[ctx.func].values[v].def = Def::Inst(inst);
    Ok(v)
}

fn binop_op(op: &str) -> Option<fn(Value, Value) -> InstKind> {
    Some(match op {
        "add" => InstKind::Add,
        "sub" => InstKind::Sub,
        "mul" => InstKind::Mul,
        "udiv" => InstKind::Udiv,
        "sdiv" => InstKind::Sdiv,
        "urem" => InstKind::Urem,
        "srem" => InstKind::Srem,
        "and" => InstKind::And,
        "or" => InstKind::Or,
        "xor" => InstKind::Xor,
        "shl" => InstKind::Shl,
        "lshr" => InstKind::LShr,
        "ashr" => InstKind::AShr,
        "fadd" => InstKind::FAdd,
        "fsub" => InstKind::FSub,
        "fmul" => InstKind::FMul,
        "fdiv" => InstKind::FDiv,
        _ => return None,
    })
}

fn unary_op(op: &str) -> Option<fn(Value) -> InstKind> {
    Some(match op {
        "neg" => InstKind::Neg,
        "not" => InstKind::Not,
        "clz" => InstKind::Clz,
        "ctz" => InstKind::Ctz,
        "popcnt" => InstKind::Popcnt,
        "bswap" => InstKind::Bswap,
        "sqrt" => InstKind::Sqrt,
        "ceil" => InstKind::Ceil,
        "floor" => InstKind::Floor,
        "fneg" => InstKind::FNeg,
        _ => return None,
    })
}

fn cast_op(op: &str) -> Option<fn(Value) -> InstKind> {
    Some(match op {
        "trunc" => InstKind::Trunc,
        "zext" => InstKind::Zext,
        "sext" => InstKind::Sext,
        "bitcast" => InstKind::Bitcast,
        "sitof" => InstKind::SItoF,
        "uitof" => InstKind::UItoF,
        "sftoi" => InstKind::SFtoI,
        "uftoi" => InstKind::UFtoI,
        "fptrunc" => InstKind::Fptrunc,
        "fpext" => InstKind::Fpext,
        "ptrtoint" => InstKind::PtrToInt,
        "inttoptr" => InstKind::IntToPtr,
        _ => return None,
    })
}

fn icmp_pred(word: &str) -> Option<IcmpPred> {
    Some(match word {
        "eq" => IcmpPred::Eq,
        "ne" => IcmpPred::Ne,
        "ult" => IcmpPred::Ult,
        "ule" => IcmpPred::Ule,
        "ugt" => IcmpPred::Ugt,
        "uge" => IcmpPred::Uge,
        "slt" => IcmpPred::Slt,
        "sle" => IcmpPred::Sle,
        "sgt" => IcmpPred::Sgt,
        "sge" => IcmpPred::Sge,
        _ => return None,
    })
}

fn fcmp_pred(word: &str) -> Option<FcmpPred> {
    Some(match word {
        "eq" => FcmpPred::Eq,
        "ne" => FcmpPred::Ne,
        "lt" => FcmpPred::Lt,
        "le" => FcmpPred::Le,
        "gt" => FcmpPred::Gt,
        "ge" => FcmpPred::Ge,
        "ord" => FcmpPred::Ord,
        "uno" => FcmpPred::Uno,
        _ => return None,
    })
}

fn parse_result_annotation(c: &mut Cursor) -> Result<Option<(u32, Ty)>, MirParseError> {
    let is_value = matches!(c.peek(), Some(Tok::Word(w)) if w.starts_with('%'));
    if !is_value {
        return Ok(None);
    }
    let word = c.word()?;
    let n = value_num(&word).ok_or_else(|| c.err(format!("bad result binding `{word}`")))?;
    c.expect_punct(':')?;
    let ty = parse_ty(c)?;
    c.expect_punct('=')?;
    Ok(Some((n, ty)))
}

/// `b<N>` optionally followed by `(%a, %b)`; the arguments are plain
/// value references and their count must match the target's params.
fn parse_branch_target(module: &mut Module, ctx: &mut FnCtx, c: &mut Cursor) -> Result<(Block, SmallVec<[Value; 4]>), MirParseError> {
    let word = c.word()?;
    let label = parse_label(&word).ok_or_else(|| c.err(format!("expected a block label, found `{word}`")))?;
    let mut args = SmallVec::new();
    if c.eat_punct('(') && !c.eat_punct(')') {
        loop {
            let word = c.word()?;
            let n = value_num(&word).ok_or_else(|| c.err(format!("branch argument `{word}` must be a value reference")))?;
            args.push(value_ref(module, ctx, n, c.line)?);
            if !c.eat_punct(',') {
                c.expect_punct(')')?;
                break;
            }
        }
    }
    let block = block_ref(module, ctx, label, &args, c.line)?;
    Ok((block, args))
}

/// Pre-create a referenced block's params for an edge whose arguments
/// are not plain value references (invoke results, lpad payload). A
/// fresh block gets placeholder params; one that already has params
/// (from its label or another edge) must match the count.
fn reserve_edge_params(module: &mut Module, ctx: &FnCtx, block: Block, tys: &[Ty], line: u32) -> Result<(), MirParseError> {
    let func = &mut module.funcs[ctx.func];
    let have = func.blocks[block].params.len();
    if have == 0 {
        for (i, ty) in tys.iter().enumerate() {
            let v = func.values.push(ValueData { ty: ty.clone(), def: Def::Param(block, i as u32) });
            func.blocks[block].params.push(v);
        }
        return Ok(());
    }
    if have != tys.len() {
        return Err(MirParseError {
            line,
            message: format!("block b{} takes {have} params but this edge supplies {}", block.index(), tys.len()),
        });
    }
    Ok(())
}

fn parse_term(module: &mut Module, ctx: &mut FnCtx, names: &Names, c: &mut Cursor, op: &str) -> Result<Terminator, MirParseError> {
    match op {
        "br" => {
            let (dest, args) = parse_branch_target(module, ctx, c)?;
            Ok(Terminator::Br { dest, args })
        }
        "brif" => {
            let cond = need_val_operand(module, ctx, c, "brif condition")?;
            let (then_block, then_args) = parse_branch_target(module, ctx, c)?;
            let (else_block, else_args) = parse_branch_target(module, ctx, c)?;
            Ok(Terminator::CondBr { cond, then_block, else_block, then_args, else_args })
        }
        "switch" => {
            let scrutinee = need_val_operand(module, ctx, c, "switch scrutinee")?;
            c.expect_punct('[')?;
            let mut cases: Vec<(i128, Block)> = Vec::new();
            loop {
                if c.eat_punct(']') {
                    break;
                }
                let word = c.word()?;
                let value = parse_i128_lit(&word).ok_or_else(|| c.err(format!("bad switch case `{word}`")))?;
                c.expect_punct(':')?;
                let word = c.word()?;
                let label = parse_label(&word).ok_or_else(|| c.err(format!("expected a block label, found `{word}`")))?;
                let block = block_ref(module, ctx, label, &[], c.line)?;
                cases.push((value, block));
                if !c.eat_punct(',') {
                    c.expect_punct(']')?;
                    break;
                }
            }
            c.expect_word("default")?;
            let word = c.word()?;
            let label = parse_label(&word).ok_or_else(|| c.err(format!("expected a block label, found `{word}`")))?;
            let default = block_ref(module, ctx, label, &[], c.line)?;
            Ok(Terminator::Switch { scrutinee, cases, default })
        }
        "ret" => {
            let ret = module.funcs[ctx.func].sig.ret.clone();
            if c.done() {
                if !ret.is_void() {
                    return Err(c.err("missing return value"));
                }
                Ok(Terminator::Return { val: None })
            } else {
                if ret.is_void() {
                    return Err(c.err("void function cannot return a value"));
                }
                let raw = parse_operand(module, ctx, c)?;
                let val = materialize(module, ctx, raw, Some(&ret), c.line)?;
                Ok(Terminator::Return { val: Some(val) })
            }
        }
        "resume" => {
            let exn = need_val_operand(module, ctx, c, "resume exception")?;
            c.expect_punct(',')?;
            let sel = need_val_operand(module, ctx, c, "resume selector")?;
            expect_done(c)?;
            Ok(Terminator::Resume { exn, sel })
        }
        "invoke" | "invoke.ext" | "invoke.ind" => {
            let callee = match op {
                "invoke" => {
                    let word = c.word()?;
                    let name = symbol(&word).ok_or_else(|| c.err(format!("expected @name, found `{word}`")))?;
                    let f = *names.funcs.get(name).ok_or_else(|| c.err(format!("invoke of unknown function @{name} (missing .func)")))?;
                    InvokeCallee::Func(f)
                }
                "invoke.ext" => {
                    let word = c.word()?;
                    let name = symbol(&word).ok_or_else(|| c.err(format!("expected @name, found `{word}`")))?;
                    let e = *names.externs.get(name).ok_or_else(|| c.err(format!("invoke of unknown extern @{name} (missing .extern)")))?;
                    InvokeCallee::Extern(e)
                }
                _ => InvokeCallee::Indirect(need_val_operand(module, ctx, c, "invoke.ind callee")?),
            };
            let sig = parse_sig(c)?;
            if !c.done() {
                c.expect_punct(',')?;
            }
            let args = parse_call_args(module, ctx, c)?;
            c.expect_word("to")?;
            let (normal, _normal_args) = parse_branch_target(module, ctx, c)?;
            // The normal edge carries the call's results as the first
            // block params; reserve them now so the label line (or a
            // later branch) sees the arity.
            let result_tys: Vec<Ty> = if sig.ret.is_void() { Vec::new() } else { vec![sig.ret.clone()] };
            reserve_edge_params(module, ctx, normal, &result_tys, c.line)?;
            c.expect_word("unwind")?;
            let word = c.word()?;
            let label = parse_label(&word).ok_or_else(|| c.err(format!("expected a block label after unwind, found `{word}`")))?;
            let unwind = block_ref(module, ctx, label, &[], c.line)?;
            // Landing-pad payload params are declared on the label line
            // (their types are target-dependent); reserve two pending
            // slots (`Ty::Void` is the established placeholder).
            reserve_edge_params(module, ctx, unwind, &[Ty::Void, Ty::Void], c.line)?;
            expect_done(c)?;
            Ok(Terminator::Invoke { callee, args, sig, normal, normal_args: Default::default(), unwind })
        }
        "unreachable" => Ok(Terminator::Unreachable),
        _ => unreachable!("terminator op checked by caller"),
    }
}

fn parse_call_args(module: &mut Module, ctx: &mut FnCtx, c: &mut Cursor) -> Result<SmallVec<[Value; 4]>, MirParseError> {
    let mut args = SmallVec::new();
    while !c.done() {
        let arg_ty = parse_ty(c)?;
        let raw = parse_operand(module, ctx, c)?;
        args.push(materialize(module, ctx, raw, Some(&arg_ty), c.line)?);
        if !c.eat_punct(',') {
            break;
        }
    }
    Ok(args)
}

fn parse_inst(module: &mut Module, ctx: &mut FnCtx, names: &Names, c: &mut Cursor) -> Result<(), MirParseError> {
    let line = c.line;
    let result = parse_result_annotation(c)?;
    let op = c.word()?;
    let cur = ctx.cur.expect("functions always start inside the entry block");

    ctx.last_inst_line = Some(line);

    if matches!(op.as_str(), "br" | "brif" | "switch" | "ret" | "unreachable" | "resume" | "invoke" | "invoke.ext" | "invoke.ind") {
        no_result(&result, &op, line)?;
        let term = parse_term(module, ctx, names, c, &op)?;
        let func = &mut module.funcs[ctx.func];
        if func.blocks[cur].term.is_some() {
            return Err(c.err("block already has a terminator"));
        }
        func.blocks[cur].term = Some(term);
        expect_done(c)?;
        return Ok(());
    }
    if module.funcs[ctx.func].blocks[cur].term.is_some() {
        return Err(c.err("instruction after the block terminator"));
    }

    if let Some(mk) = binop_op(&op) {
        let (n, ty) = need_result(result, &op, line)?;
        let a = parse_operand(module, ctx, c)?;
        c.expect_punct(',')?;
        let b = parse_operand(module, ctx, c)?;
        let av = materialize(module, ctx, a, Some(&ty), line)?;
        let bv = materialize(module, ctx, b, Some(&ty), line)?;
        emit_value(module, ctx, cur, mk(av, bv), n, ty, line)?;
        expect_done(c)?;
        return Ok(());
    }
    if let Some(mk) = unary_op(&op) {
        let (n, ty) = need_result(result, &op, line)?;
        let raw = parse_operand(module, ctx, c)?;
        let v = materialize(module, ctx, raw, Some(&ty), line)?;
        emit_value(module, ctx, cur, mk(v), n, ty, line)?;
        expect_done(c)?;
        return Ok(());
    }
    if let Some(rest) = op.strip_prefix("icmp.") {
        let pred = icmp_pred(rest).ok_or_else(|| c.err(format!("unknown icmp predicate `{rest}`")))?;
        let (n, ty) = need_result(result, &op, line)?;
        let a = parse_operand(module, ctx, c)?;
        c.expect_punct(',')?;
        let b = parse_operand(module, ctx, c)?;
        let exp_a = raw_val_ty(module, ctx, &b);
        let exp_b = raw_val_ty(module, ctx, &a);
        let av = materialize(module, ctx, a, exp_a.as_ref(), line)?;
        let bv = materialize(module, ctx, b, exp_b.as_ref(), line)?;
        emit_value(module, ctx, cur, InstKind::Icmp { pred, lhs: av, rhs: bv }, n, ty, line)?;
        expect_done(c)?;
        return Ok(());
    }
    if let Some(rest) = op.strip_prefix("fcmp.") {
        let pred = fcmp_pred(rest).ok_or_else(|| c.err(format!("unknown fcmp predicate `{rest}`")))?;
        let (n, ty) = need_result(result, &op, line)?;
        let a = parse_operand(module, ctx, c)?;
        c.expect_punct(',')?;
        let b = parse_operand(module, ctx, c)?;
        let exp_a = raw_val_ty(module, ctx, &b);
        let exp_b = raw_val_ty(module, ctx, &a);
        let av = materialize(module, ctx, a, exp_a.as_ref(), line)?;
        let bv = materialize(module, ctx, b, exp_b.as_ref(), line)?;
        emit_value(module, ctx, cur, InstKind::Fcmp { pred, lhs: av, rhs: bv }, n, ty, line)?;
        expect_done(c)?;
        return Ok(());
    }
    if let Some(mk) = cast_op(&op) {
        let (n, ty) = need_result(result, &op, line)?;
        let v = need_val_operand(module, ctx, c, "cast operand")?;
        emit_value(module, ctx, cur, mk(v), n, ty, line)?;
        expect_done(c)?;
        return Ok(());
    }

    match op.as_str() {
        "select" => {
            let (n, ty) = need_result(result, &op, line)?;
            let cond = need_val_operand(module, ctx, c, "select condition")?;
            c.expect_punct(',')?;
            let a = parse_operand(module, ctx, c)?;
            c.expect_punct(',')?;
            let b = parse_operand(module, ctx, c)?;
            let av = materialize(module, ctx, a, Some(&ty), line)?;
            let bv = materialize(module, ctx, b, Some(&ty), line)?;
            emit_value(module, ctx, cur, InstKind::Select { cond, then_val: av, else_val: bv }, n, ty, line)?;
        }
        "alloca" => {
            let (n, ty) = need_result(result, &op, line)?;
            let pointee = parse_ty(c)?;
            let count = if c.eat_word("x") {
                c.expect_punct('<')?;
                let word = c.word()?;
                let count = parse_u64_lit(&word).ok_or_else(|| c.err(format!("bad alloca count `{word}`")))?;
                c.expect_punct('>')?;
                count
            } else {
                1
            };
            emit_value(module, ctx, cur, InstKind::Alloca { pointee, count }, n, ty, line)?;
        }
        "load" => {
            let (n, ty) = need_result(result, &op, line)?;
            let mut flags = MemFlags::plain();
            if c.eat_word("volatile") {
                flags.volatile = true;
            }
            let addr = need_val_operand(module, ctx, c, "load address")?;
            emit_value(module, ctx, cur, InstKind::Load { addr, flags }, n, ty, line)?;
        }
        "store" => {
            no_result(&result, &op, line)?;
            let mut flags = MemFlags::plain();
            if c.eat_word("volatile") {
                flags.volatile = true;
            }
            let addr = need_val_operand(module, ctx, c, "store address")?;
            c.expect_punct(',')?;
            let expected = pointee_ty(module, ctx, addr);
            let raw = parse_operand(module, ctx, c)?;
            let val = materialize(module, ctx, raw, expected.as_ref(), line)?;
            push_inst(module, ctx, cur, InstKind::Store { addr, val, flags }, None);
        }
        "gep" => {
            let (n, ty) = need_result(result, &op, line)?;
            let base = need_val_operand(module, ctx, c, "gep base")?;
            let mut indices: Vec<GepIndex> = Vec::new();
            while !c.done() {
                c.eat_punct(',');
                if c.eat_word("field") {
                    let word = c.word()?;
                    let index = parse_u64_lit(&word).ok_or_else(|| c.err(format!("bad field index `{word}`")))?;
                    let index = u32::try_from(index).map_err(|_| c.err(format!("field index `{word}` out of range")))?;
                    indices.push(GepIndex::Field(index));
                } else if c.eat_word("index") {
                    let v = need_val_operand(module, ctx, c, "gep index")?;
                    indices.push(GepIndex::Index(v));
                } else {
                    let word = c.word()?;
                    let index = parse_u64_lit(&word).ok_or_else(|| c.err(format!("bad GEP step `{word}`")))?;
                    indices.push(GepIndex::ConstIndex(index));
                }
            }
            if indices.is_empty() {
                return Err(c.err("gep needs at least one step"));
            }
            emit_value(module, ctx, cur, InstKind::GetElementPtr { base, indices: indices.into() }, n, ty, line)?;
        }
        "extract" => {
            let (n, ty) = need_result(result, &op, line)?;
            let agg = need_val_operand(module, ctx, c, "extract aggregate")?;
            c.expect_punct(',')?;
            let word = c.word()?;
            let index = parse_u64_lit(&word).ok_or_else(|| c.err(format!("bad extract index `{word}`")))?;
            let index = u32::try_from(index).map_err(|_| c.err(format!("extract index `{word}` out of range")))?;
            emit_value(module, ctx, cur, InstKind::ExtractValue { agg, index }, n, ty, line)?;
        }
        "insert" => {
            let (n, ty) = need_result(result, &op, line)?;
            let agg = need_val_operand(module, ctx, c, "insert aggregate")?;
            c.expect_punct(',')?;
            let raw = parse_operand(module, ctx, c)?;
            c.expect_punct(',')?;
            let word = c.word()?;
            let index = parse_u64_lit(&word).ok_or_else(|| c.err(format!("bad insert index `{word}`")))?;
            let index = u32::try_from(index).map_err(|_| c.err(format!("insert index `{word}` out of range")))?;
            let expected = elem_ty_of(module, ctx, agg, index);
            let elem = materialize(module, ctx, raw, expected.as_ref(), line)?;
            emit_value(module, ctx, cur, InstKind::InsertValue { agg, elem, index }, n, ty, line)?;
        }
        "splat" => {
            let (n, ty) = need_result(result, &op, line)?;
            let raw = parse_operand(module, ctx, c)?;
            let expected = ty.as_vector().map(|(elem, _)| elem.clone());
            let v = materialize(module, ctx, raw, expected.as_ref(), line)?;
            emit_value(module, ctx, cur, InstKind::Splat(v), n, ty, line)?;
        }
        "extractlane" => {
            let (n, ty) = need_result(result, &op, line)?;
            let vec = need_val_operand(module, ctx, c, "extractlane vector")?;
            c.expect_punct(',')?;
            let word = c.word()?;
            let lane = parse_u64_lit(&word).ok_or_else(|| c.err(format!("bad extractlane index `{word}`")))?;
            let lane = u8::try_from(lane).map_err(|_| c.err(format!("extractlane index `{word}` out of range")))?;
            emit_value(module, ctx, cur, InstKind::ExtractLane { vec, lane }, n, ty, line)?;
        }
        "memcpy" => {
            no_result(&result, &op, line)?;
            let dst = need_val_operand(module, ctx, c, "memcpy destination")?;
            c.expect_punct(',')?;
            let src = need_val_operand(module, ctx, c, "memcpy source")?;
            c.expect_punct(',')?;
            c.expect_word("len")?;
            c.expect_punct('=')?;
            let word = c.word()?;
            let len = parse_u64_lit(&word).ok_or_else(|| c.err(format!("bad memcpy length `{word}`")))?;
            push_inst(module, ctx, cur, InstKind::Memcpy { dst, src, len }, None);
        }
        "memset" => {
            no_result(&result, &op, line)?;
            let dst = need_val_operand(module, ctx, c, "memset destination")?;
            c.expect_punct(',')?;
            c.expect_word("byte")?;
            c.expect_punct('=')?;
            let word = c.word()?;
            let byte = parse_u64_lit(&word).ok_or_else(|| c.err(format!("bad memset byte `{word}`")))?;
            if byte > 0xff {
                return Err(c.err(format!("memset byte `{word}` out of range")));
            }
            c.expect_punct(',')?;
            c.expect_word("len")?;
            c.expect_punct('=')?;
            let word = c.word()?;
            let len = parse_u64_lit(&word).ok_or_else(|| c.err(format!("bad memset length `{word}`")))?;
            push_inst(module, ctx, cur, InstKind::Memset { dst, byte: byte as u8, len }, None);
        }
        "globaladdr" => {
            let (n, ty) = need_result(result, &op, line)?;
            let word = c.word()?;
            let name = symbol(&word).ok_or_else(|| c.err(format!("expected @name, found `{word}`")))?;
            let global = *names.globals.get(name).ok_or_else(|| c.err(format!("unknown global @{name}")))?;
            emit_value(module, ctx, cur, InstKind::GlobalAddr { global }, n, ty, line)?;
        }
        "funcaddr" => {
            let (n, ty) = need_result(result, &op, line)?;
            let word = c.word()?;
            let name = symbol(&word).ok_or_else(|| c.err(format!("expected @name, found `{word}`")))?;
            let func = *names.funcs.get(name).ok_or_else(|| c.err(format!("unknown function @{name} (missing .func)")))?;
            emit_value(module, ctx, cur, InstKind::FuncAddr { func }, n, ty, line)?;
        }
        "externaddr" => {
            let (n, ty) = need_result(result, &op, line)?;
            let word = c.word()?;
            let name = symbol(&word).ok_or_else(|| c.err(format!("expected @name, found `{word}`")))?;
            let target = *names.externs.get(name).ok_or_else(|| c.err(format!("unknown extern @{name}")))?;
            emit_value(module, ctx, cur, InstKind::ExternAddr { target }, n, ty, line)?;
        }
        "call" | "call.ext" | "call.ind" => {
            enum Callee {
                Func(Func),
                Ext(crate::ir::Extern),
                Ind(Value, Signature),
            }
            let (callee, ret) = match op.as_str() {
                "call" => {
                    let word = c.word()?;
                    let name = symbol(&word).ok_or_else(|| c.err(format!("expected @name, found `{word}`")))?;
                    let callee = *names.funcs.get(name).ok_or_else(|| c.err(format!("call to unknown function @{name} (missing .func)")))?;
                    let ret = module.funcs[callee].sig.ret.clone();
                    (Callee::Func(callee), ret)
                }
                "call.ext" => {
                    let word = c.word()?;
                    let name = symbol(&word).ok_or_else(|| c.err(format!("expected @name, found `{word}`")))?;
                    let target = *names.externs.get(name).ok_or_else(|| c.err(format!("call to unknown extern @{name} (missing .extern)")))?;
                    let ret = module.externs[target].sig.ret.clone();
                    (Callee::Ext(target), ret)
                }
                _ => {
                    let sig = parse_sig(c)?;
                    let fnptr = need_val_operand(module, ctx, c, "call.ind callee")?;
                    if !c.done() {
                        c.expect_punct(',')?;
                    }
                    let ret = sig.ret.clone();
                    (Callee::Ind(fnptr, sig), ret)
                }
            };
            let args = parse_call_args(module, ctx, c)?;
            let kind = match callee {
                Callee::Func(callee) => InstKind::Call { callee, args },
                Callee::Ext(target) => InstKind::CallExtern { target, args },
                Callee::Ind(callee, sig) => InstKind::CallIndirect { callee, args, sig },
            };
            if let Some((n, ty)) = result {
                if ret.is_void() {
                    return Err(c.err(format!("{op} to a void function cannot produce a value")));
                }
                emit_value(module, ctx, cur, kind, n, ty, line)?;
            } else {
                push_inst(module, ctx, cur, kind, None);
            }
        }
        "eh.typeid" => {
            let (n, ty) = need_result(result, &op, line)?;
            let word = c.word()?;
            let gname = symbol(&word).ok_or_else(|| c.err(format!("expected @name after eh.typeid, found `{word}`")))?;
            let typeinfo = *names
                .globals
                .get(gname)
                .ok_or_else(|| c.err(format!("eh.typeid of unknown typeinfo @{gname} (missing .dataextern)")))?;
            emit_value(module, ctx, cur, InstKind::EhTypeId { typeinfo }, n, ty, line)?;
        }
        _ => return Err(c.err(format!("unknown instruction `{op}`"))),
    }
    expect_done(c)?;
    Ok(())
}
