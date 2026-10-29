//! Generates `glasm_data/src/insns_gen.rs` from nasm's pre-generated tables
//! (`x86/insnsa.c`, `x86/insnsb.c`, `x86/regflags.c`, `x86/regvals.c`,
//! `x86/insnsi.h`, `x86/iflag.c`, `include/opflags.h`,
//! `x86/iflaggen.h`). Run nasm's `./configure && make` in the `nasm/`
//! submodule first, then `cargo run -p glasm --bin gen-insns`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn main() {
    let root = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("nasm")
        });
    let out = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("glasm_data/src/insns_gen.rs");
    let src = Tables::read(&root);
    std::fs::write(&out, emit(&src)).expect("write insns_gen.rs");
    println!(
        "wrote {} ({} bytecodes, {} templates, {} registers, {} mnemonics)",
        out.display(),
        src.bytecodes.len(),
        src.templates.iter().map(|t| t.len()).sum::<usize>(),
        src.regs.len(),
        src.mnemonics.len()
    );
}

struct RegInfo {
    name: String,
    flags: u64,
    val: i32,
}

struct Template {
    mnemonic: String,
    operands: u8,
    opd: [u64; 5],
    deco: [u64; 5],
    code_off: u32,
    f: [u32; 7],
    iflag_idx: usize,
}

struct Tables {
    bytecodes: Vec<u8>,
    templates: Vec<Vec<Template>>, // per mnemonic
    mnemonics: Vec<String>,
    regs: Vec<RegInfo>,
}

impl Tables {
    fn read(root: &Path) -> Tables {
        let defines = Defines::load(root);
        let bytecodes = read_bytecodes(root);
        let regs = read_regs(root, &defines);
        let (templates, mnemonics) = read_insnsa(root, &defines);

        Tables { bytecodes, templates, mnemonics, regs }
    }
}

// ---------------------------------------------------------------------
// C constant-expression evaluation over the opflags headers
// ---------------------------------------------------------------------

struct Defines {
    map: HashMap<String, String>,
}

impl Defines {
    fn load(root: &Path) -> Defines {
        let mut map = HashMap::new();
        for hdr in ["include/opflags.h", "include/nasm.h", "include/tables.h"] {
            let path = root.join(hdr);
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            for line in text.lines() {
                let Some(rest) = line.trim().strip_prefix("#define ") else {
                    continue;
                };
                let Some((name, value)) = rest.split_once(char::is_whitespace) else {
                    continue;
                };
                // Function-like macros are handled as builtins or skipped.
                if name.ends_with('(') {
                    continue;
                }
                map.entry(name.to_string())
                    .or_insert_with(|| strip_comments(value).trim().to_string());
            }
        }
        Defines { map }
    }

    fn get(&self, name: &str) -> Option<String> {
        self.map.get(name).cloned()
    }
}

/// Evaluates a C constant expression built from the opflags macros.
fn eval(expr: &str, d: &Defines) -> Result<u64, String> {
    let toks = tokenize(&strip_comments(expr));
    let mut pos = 0;
    let v = eval_ternary(&toks, &mut pos, d)?;
    if pos != toks.len() {
        return Err(format!("trailing tokens in `{expr}'"));
    }
    Ok(v)
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Id(String),
    Num(u64),
    Op(&'static str),
    LParen,
    RParen,
    Comma,
}

fn tokenize(s: &str) -> Vec<Tok> {
    let b: Vec<char> = s.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric()) {
                i += 1;
            }
            let text: String = b[start..i].iter().collect();
            let v = if let Some(h) = text.strip_prefix("0x").or(text.strip_prefix("0X")) {
                u64::from_str_radix(h, 16)
            } else if text.len() > 1 && text.starts_with('0') {
                u64::from_str_radix(&text[1..], 8)
            } else {
                text.parse::<u64>()
            };
            if let Ok(v) = v {
                toks.push(Tok::Num(v));
            } else {
                toks.push(Tok::Id(text)); // e.g. UINT64_C handled as call
            }
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == '_') {
                i += 1;
            }
            toks.push(Tok::Id(b[start..i].iter().collect()));
            continue;
        }
        let two = format!("{}{}", c, b.get(i + 1).map(|x| *x).unwrap_or(' '));
        let (op, len): (&'static str, usize) = match two.as_str() {
            "<< " => ("<<", 2),
            ">> " => (">>", 2),
            "==" => ("==", 2),
            "!=" => ("!=", 2),
            "&&" => ("&&", 2),
            "||" => ("||", 2),
            _ => {
                let one: &'static str = match c {
                    '|' => "|",
                    '&' => "&",
                    '^' => "^",
                    '+' => "+",
                    '-' => "-",
                    '*' => "*",
                    '~' => "~",
                    '/' => "/",
                    '<' => "<",
                    '>' => ">",
                    '?' => "?",
                    ':' => ":",
                    _ => {
                        match c {
                            '(' => toks.push(Tok::LParen),
                            ')' => toks.push(Tok::RParen),
                            ',' => toks.push(Tok::Comma),
                            _ => {}
                        }
                        i += 1;
                        continue;
                    }
                };
                (one, 1)
            }
        };
        toks.push(Tok::Op(op));
        i += len;
    }
    toks
}

fn eval_ternary(t: &[Tok], pos: &mut usize, d: &Defines) -> Result<u64, String> {
    let cond = eval_binary(t, pos, d, 0)?;
    if matches!(t.get(*pos), Some(Tok::Op("?"))) {
        *pos += 1;
        let a = eval_ternary(t, pos, d)?;
        if !matches!(t.get(*pos), Some(Tok::Op(":"))) {
            return Err("expected `:' in conditional".into());
        }
        *pos += 1;
        let b = eval_ternary(t, pos, d)?;
        return Ok(if cond != 0 { a } else { b });
    }
    Ok(cond)
}

fn prec_of(op: &str) -> Option<u8> {
    Some(match op {
        "|" => 1,
        "^" => 2,
        "&" => 3,
        "==" | "!=" | "<" | ">" => 4,
        "<<" | ">>" => 5,
        "+" | "-" => 6,
        "*" | "/" => 7,
        _ => return None,
    })
}

fn eval_binary(t: &[Tok], pos: &mut usize, d: &Defines, min_prec: u8) -> Result<u64, String> {
    let mut lhs = eval_unary(t, pos, d)?;
    loop {
        let Some(Tok::Op(op)) = t.get(*pos) else { break };
        let Some(prec) = prec_of(op) else { break };
        if prec < min_prec {
            break;
        }
        let op = *op;
        *pos += 1;
        let rhs = eval_binary(t, pos, d, prec + 1)?;
        lhs = match op {
            "|" => lhs | rhs,
            "^" => lhs ^ rhs,
            "&" => lhs & rhs,
            "==" => (lhs == rhs) as u64,
            "!=" => (lhs != rhs) as u64,
            "<" => (lhs < rhs) as u64,
            ">" => (lhs > rhs) as u64,
            "<<" => lhs << rhs,
            ">>" => lhs >> rhs,
            "+" => lhs.wrapping_add(rhs),
            "-" => lhs.wrapping_sub(rhs),
            "*" => lhs.wrapping_mul(rhs),
            "/" => lhs / rhs,
            _ => unreachable!(),
        };
    }
    Ok(lhs)
}

fn eval_unary(t: &[Tok], pos: &mut usize, d: &Defines) -> Result<u64, String> {
    match t.get(*pos) {
        Some(Tok::Op("-")) => {
            *pos += 1;
            Ok((eval_unary(t, pos, d)? as i64).wrapping_neg() as u64)
        }
        Some(Tok::Op("~")) => {
            *pos += 1;
            Ok(!eval_unary(t, pos, d)?)
        }
        Some(Tok::LParen) => {
            *pos += 1;
            let v = eval_ternary(t, pos, d)?;
            if !matches!(t.get(*pos), Some(Tok::RParen)) {
                return Err("expected `)'".into());
            }
            *pos += 1;
            Ok(v)
        }
        Some(Tok::Num(v)) => {
            *pos += 1;
            Ok(*v)
        }
        Some(Tok::Id(name)) => {
            *pos += 1;
            if matches!(t.get(*pos), Some(Tok::LParen)) {
                *pos += 1;
                let mut args = Vec::new();
                if !matches!(t.get(*pos), Some(Tok::RParen)) {
                    loop {
                        args.push(eval_ternary(t, pos, d)?);
                        if matches!(t.get(*pos), Some(Tok::Comma)) {
                            *pos += 1;
                        } else {
                            break;
                        }
                    }
                }
                if !matches!(t.get(*pos), Some(Tok::RParen)) {
                    return Err("expected `)'".into());
                }
                *pos += 1;
                return macro_call(name, &args, d);
            }
            match d.get(name) {
                Some(v) => eval(&v, d),
                None => Err(format!("unknown identifier `{name}'")),
            }
        }
        other => Err(format!("unexpected token {other:?}")),
    }
}

// ---------------------------------------------------------------------
// Table readers
// ---------------------------------------------------------------------

fn strip_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_comment = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if in_comment {
            if c == '*' && chars.peek() == Some(&'/') {
                chars.next();
                in_comment = false;
            }
            continue;
        }
        if c == '/' && chars.peek() == Some(&'*') {
            chars.next();
            in_comment = true;
            continue;
        }
        if c == '/' && chars.peek() == Some(&'/') {
            while let Some(c) = chars.next() {
                if c == '\n' {
                    out.push('\n');
                    break;
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

fn read_bytecodes(root: &Path) -> Vec<u8> {
    let text = strip_comments(
        &std::fs::read_to_string(root.join("x86/insnsb.c")).expect("x86/insnsb.c"),
    );
    let start = text.find('{').expect("bytecodes {");
    let end = text.rfind('}').expect("bytecodes }");
    let body = &text[start + 1..end];
    let mut out = Vec::new();
    for tok in body.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        let v = if let Some(h) = tok.strip_prefix("0x").or(tok.strip_prefix("0X")) {
            u32::from_str_radix(h, 16)
        } else if tok.len() > 1 && tok.starts_with('0') {
            u32::from_str_radix(&tok[1..], 8)
        } else {
            tok.parse::<u32>()
        };
        out.push(v.expect("bytecode value") as u8);
    }
    out
}

fn read_iflags(root: &Path) -> Vec<[u32; 7]> {
    let text = std::fs::read_to_string(root.join("x86/iflag.c")).expect("x86/iflag.c");
    let mut out = Vec::new();
    for line in text.lines() {
        let Some(line) = line.trim().strip_prefix("{{") else {
            continue;
        };
        let Some(end) = line.find("}}") else {
            eprintln!("no }} for line: {line}");
            continue;
        };
        let fields: Vec<u32> = line[..end]
            .split(',')
            .filter_map(|f| {
                let f = f.trim().trim_start_matches("UINT32_C(").trim_end_matches(')');
                let f = f.trim();
                if let Some(h) = f.strip_prefix("0x").or(f.strip_prefix("0X")) {
                    u32::from_str_radix(h, 16).ok()
                } else {
                    f.parse::<u32>().ok()
                }
            })
            .collect();
        if fields.len() == 7 {
            out.push(fields.try_into().unwrap());
        }
    }
    out
}

/// Bit index constants from iflaggen.h (IF_<name> = index).
fn read_if_bits(root: &Path) -> HashMap<String, u32> {
    let text = std::fs::read_to_string(root.join("x86/iflaggen.h")).expect("iflaggen.h");
    let mut out = HashMap::new();
    for line in text.lines() {
        let Some(rest) = line.trim().strip_prefix("#define IF_") else {
            continue;
        };
        let mut parts = rest.split_whitespace();
        let name = parts.next().unwrap_or("");
        let value = parts.next().unwrap_or("");
        if let Ok(v) = value.parse::<u32>() {
            out.insert(format!("IF_{name}"), v);
        }
    }
    out
}

fn read_regs(root: &Path, d: &Defines) -> Vec<RegInfo> {
    let vals_text =
        std::fs::read_to_string(root.join("x86/regvals.c")).expect("x86/regvals.c");
    let flags_text =
        std::fs::read_to_string(root.join("x86/regflags.c")).expect("x86/regflags.c");

    let mut vals: Vec<(String, i32)> = Vec::new();
    let mut lines = vals_text.lines().peekable();
    while let Some(line) = lines.next() {
        let Some(idx) = line.find("/*") else { continue };
        let Some(end) = line[idx..].find("*/") else { continue };
        let name = line[idx + 2..idx + end].trim().to_string();
        // The value is on this line or the next.
        let code = line[..idx].trim();
        let code = if code.is_empty() {
            lines.next().map(|l| l.trim()).unwrap_or("")
        } else {
            code
        };
        let Some(num) = code.trim_end_matches(',').parse::<i32>().ok() else {
            continue;
        };
        vals.push((name, num));
    }

    let mut flags: Vec<(String, u64)> = Vec::new();
    for line in flags_text.lines() {
        let Some(idx) = line.find("/*") else { continue };
        let Some(end) = line[idx..].find("*/") else { continue };
        let name = line[idx + 2..idx + end].trim().to_string();
        let Some(expr) = line[idx + end + 2..].trim().strip_suffix(',') else {
            continue;
        };
        let Ok(v) = eval(expr, d) else {
            continue;
        };
        flags.push((name, v));
    }

    assert_eq!(vals.len(), flags.len(), "regvals/regflags mismatch");
    vals.into_iter()
        .zip(flags)
        .map(|((vname, val), (fname, fl))| {
            assert_eq!(vname, fname, "register order mismatch: {vname} vs {fname}");
            RegInfo { name: vname, flags: fl, val }
        })
        .collect()
}

fn read_insnsa(root: &Path, d: &Defines) -> (Vec<Vec<Template>>, Vec<String>) {
    let text = strip_comments(
        &std::fs::read_to_string(root.join("x86/insnsa.c")).expect("x86/insnsa.c"),
    );

    // Opcode enum from insnsi.h gives mnemonic names in nasm_instructions
    // order; nasm_instructions entries carry the same via comments, but the
    // instrux_<NAME> groups are keyed by name anyway.
    let enum_text =
        std::fs::read_to_string(root.join("x86/insnsi.h")).expect("x86/insnsi.h");
    let mut mnemonics: Vec<String> = Vec::new();
    for line in enum_text.lines() {
        let Some(rest) = line.trim().strip_prefix("I_") else {
            continue;
        };
        let Some(name) = rest.strip_suffix(',') else { continue };
        mnemonics.push(name.to_ascii_lowercase());
    }

    // Parse every instrux_<NAME> group.
    let mut groups: HashMap<String, Vec<Template>> = HashMap::new();
    let mut rest = text.as_str();
    while let Some(pos) = rest.find("static const struct itemplate instrux_") {
        rest = &rest[pos..];
        let name_start = rest.find("instrux_").unwrap() + 8;
        let name_end = rest[name_start..]
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap()
            + name_start;
        let name = &rest[name_start..name_end];
        let body_start = rest.find('{').unwrap();
        let body_end = rest[body_start..].find("\n};").map(|p| body_start + p);
        let Some(body_end) = body_end else { break };
        let body = &rest[body_start..body_end];

        let mut list = Vec::new();
        for line in body.lines() {
            let entry = line.trim();
            if !entry.starts_with("{I_") {
                continue;
            }
            let entry = &entry[1..];
            let Some(first_comma) = entry.find(',') else { continue };
            let _opcode_name = entry[..first_comma].trim();
            let parts: Vec<&str> = entry.splitn(4, ',').collect();
            if parts.len() < 4 {
                continue;
            }
            let operands: u8 = parts[1].trim().parse().expect("operand count");
            // opd list
            let opd_start = entry.find('{').unwrap() + 1;
            let opd_end = entry[opd_start..].find('}').unwrap() + opd_start;
            let opd_str = &entry[opd_start..opd_end];
            let mut opd = [0u64; 5];
            for (i, p) in opd_str.split(',').enumerate() {
                let p = p.trim();
                if p.is_empty() || p == "0" {
                    continue;
                }
                opd[i] = eval(p, d).unwrap_or_else(|e| panic!("opd `{p}': {e}"));
            }
            // deco: the token after the opd closing brace
            let after = entry[opd_end + 1..].trim_start_matches(',');
            let after = after.trim_start();
            let deco_end = after
                .find(|c: char| c == ',')
                .unwrap_or(after.len());
            let deco_str = after[..deco_end].trim();
            let mut deco = [0u64; 5];
            if deco_str != "NO_DECORATOR" && !deco_str.is_empty() {
                // Decorated templates are recorded with a nonzero marker;
                // the assembler rejects them.
                deco[0] = 1;
            }
            // code offset: nasm_bytecodes+N
            let Some(code_pos) = entry.find("nasm_bytecodes+") else {
                continue;
            };
            let code_str = &entry[code_pos + 15..];
            let code_end = code_str
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(code_str.len());
            let code_off: u32 = code_str[..code_end].parse().expect("bytecode offset");
            // iflag index: the number after the code offset
            let after_code = &code_str[code_end..];
            let nums: Vec<&str> = after_code.split(',').collect();
            let iflag_idx: usize = nums[1].trim().parse().expect("iflag idx");
            let f = iflag_idx; // resolved below

            list.push(Template {
                mnemonic: name.to_ascii_lowercase(),
                operands,
                opd,
                deco,
                code_off,
                f: [0; 7],
                iflag_idx: f,
            });
        }
        groups.insert(name.to_string(), list);
        rest = &rest[body_end..];
    }

    // Resolve flag fields per template via iflag_idx.
    let iflags = read_iflags(root);
    let mut templates: Vec<Vec<Template>> = Vec::with_capacity(mnemonics.len());
    for m in &mnemonics {
        let key: String = m.to_ascii_uppercase();
        let mut list = groups.remove(&key).unwrap_or_default();
        for t in &mut list {
            t.f = iflags[t.iflag_idx];
        }
        templates.push(list);
    }

    (templates, mnemonics)
}

// ---------------------------------------------------------------------
// Emitter
// ---------------------------------------------------------------------

fn emit(src: &Tables) -> String {
    let mut s = String::with_capacity(1 << 20);
    s.push_str("//! Generated from nasm's pre-generated tables by `gen-insns`.\n");
    s.push_str("//! Do not edit; regenerate with `cargo run -p glasm --bin gen-insns`.\n\n");
    s.push_str("#![allow(clippy::all)]\n\n");

    s.push_str("pub static BYTECODES: [u8; ");
    s.push_str(&src.bytecodes.len().to_string());
    s.push_str("] = [\n");
    for (i, b) in src.bytecodes.iter().enumerate() {
        if i % 24 == 0 {
            s.push_str("    ");
        }
        s.push_str(&format!("{b:3},"));
        if i % 24 == 23 {
            s.push('\n');
        }
    }
    s.push_str("\n];\n\n");

    // Instruction flags (bit indices) needed by the interpreter.
    let if_bits = read_if_bits(Path::new(&nasm_root_for_iflag()));
    s.push_str("pub mod if_ {\n");
    let mut names: Vec<_> = if_bits.iter().collect();
    names.sort_by_key(|(_, v)| *v);
    for (name, bit) in names {
        let mut short = name.trim_start_matches("IF_").to_string();
        if short.starts_with(|c: char| c.is_ascii_digit()) {
            short = format!("N{short}");
        }
        s.push_str(&format!("    pub const {short}: u32 = {bit};\n"));
    }
    s.push_str("}\n\n");

    s.push_str("pub struct Template {\n    pub opcode: u16,\n    pub operands: u8,\n    pub opd: [u64; 5],\n    pub deco: [u64; 5],\n    pub code_off: u32,\n    pub f: [u32; 7],\n}\n\n");

    s.push_str("pub static MNEMONICS: &[&str] = &[\n");
    for m in &src.mnemonics {
        s.push_str(&format!("    {m:?},\n"));
    }
    s.push_str("];\n\n");

    s.push_str("pub static INSTRUCTIONS: &[&[Template]] = &[\n");
    for (i, list) in src.templates.iter().enumerate() {
        s.push_str(&format!("    // {}: {}\n", src.mnemonics[i], list.len()));
        s.push_str("    &[\n");
        for t in list {
            let opd: Vec<String> = t.opd.iter().map(|v| format!("{v:#x}")).collect();
            let deco: Vec<String> = t.deco.iter().map(|v| format!("{v:#x}")).collect();
            s.push_str(&format!(
                "        Template {{ opcode: {}, operands: {}, opd: [{}], deco: [{}], code_off: {}, f: [{:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}] }},\n",
                t.mnemonic_idx(src),
                t.operands,
                opd.join(", "),
                deco.join(", "),
                t.code_off,
                t.f[0],
                t.f[1],
                t.f[2],
                t.f[3],
                t.f[4],
                t.f[5],
                t.f[6]
            ));
        }
        s.push_str("    ],\n");
    }
    s.push_str("];\n\n");

    s.push_str("pub struct RegInfo {\n    pub name: &'static str,\n    pub flags: u64,\n    pub val: i32,\n}\n\n");
    s.push_str("pub static REGS: &[RegInfo] = &[\n");
    for r in &src.regs {
        s.push_str(&format!(
            "    RegInfo {{ name: {:?}, flags: {:#x}, val: {} }},\n",
            r.name, r.flags, r.val
        ));
    }
    s.push_str("];\n");
    s
}

fn nasm_root_for_iflag() -> String {
    // gen-insns runs from the crate dir in tests; env-based lookup mirrors main().
    std::env::var("NASM_SRC").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{manifest}/../../nasm")
    })
}

impl Template {
    fn mnemonic_idx(&self, src: &Tables) -> u16 {
        src.mnemonics
            .iter()
            .position(|m| *m == self.mnemonic)
            .expect("mnemonic known") as u16
    }
}

/// Function-like opflags macros the tables use.
fn macro_call(name: &str, args: &[u64], d: &Defines) -> Result<u64, String> {
    let a = |i: usize| args.get(i).copied().ok_or_else(|| format!("{name}: missing arg"));
    match name {
        // RN_FLAGS(n): nasm's register-number flags.
        "RN_FLAGS" => {
            let n = a(0)?;
            let l16 = 1u64 << (10 + 18);
            let mut v = if n == 0 {
                l16 | 1u64 << (11 + 18)
            } else {
                l16 | 1u64 << (12 + 18)
            };
            if n < 16 {
                v |= l16;
            }
            Ok(v)
        }
        "GEN_OPTYPE" => Ok(1u64 << (a(0)?)),
        "GEN_MODIFIER" => Ok(1u64 << (4 + a(0)?)),
        "GEN_REG_CLASS" => Ok(1u64 << (7 + a(0)?)),
        "GEN_SUBCLASS" => Ok(1u64 << (18 + a(0)?)),
        "GEN_SIZE" => Ok(1u64 << (32 + a(0)?)),
        "GEN_REGSET" => Ok(1u64 << (48 + a(0)?)),
        "OP_GENBIT" => Ok(1u64 << (a(1)? + a(0)?)),
        "OP_GENMASK" => Ok(((1u64 << a(0)?) - 1) << a(1)?),
        "UINT64_C" | "UINT32_C" | "UINT16_C" | "UINT8_C" => a(0),
        _ => {
            let _ = d;
            Err(format!("unsupported macro call `{name}'"))
        }
    }
}

