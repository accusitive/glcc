//! The assembly driver: section/symbol state and the multi-pass loop
//! (reference: nasm's `asm/nasm.c` assembly loop and `asm/labels.c`).

use crate::arch::{Backend, Insn};
use crate::encoding::FieldLayout;
use crate::expr::{EvalCtx, RValue, SymLookup, SymbolVal};
use crate::insn::*;
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Debug)]
pub struct AsmError {
    pub line: u32,
    pub msg: String,
}

impl std::fmt::Display for AsmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.msg)
    }
}

pub struct Section {
    pub name: String,
    pub flags: SectionFlags,
    pub align: u64,
    /// For bss sections this is zero-filled bookkeeping; the writer uses
    /// its length as the section size.
    pub data: Vec<u8>,
}

pub struct Symbol {
    pub name: String,
    pub defined: bool,
    /// Defined in any earlier pass; values from previous passes are
    /// valid estimates for forward references.
    pub seen: bool,
    pub extern_: bool,
    pub global: bool,
    /// GAS `.weak`: weak binding in the emitted object.
    pub weak: bool,
    pub sect: Option<usize>,
    pub value: i64,
}

/// Where a finalized field points.
pub enum Target {
    Section(usize),
    Symbol(usize),
    /// The null symbol (R_RISCV_RELAX and other marker relocs).
    Abs,
}

pub struct Reloc {
    pub section: usize,
    pub offset: i64,
    pub size: u8,
    pub pcrel: bool,
    pub addend: i64,
    pub target: Target,
    /// ELF relocation type constant for the target architecture.
    pub r_type: u32,
}

pub struct Module {
    pub sections: Vec<Section>,
    pub symbols: Vec<Symbol>,
    pub relocs: Vec<Reloc>,
}

/// Read-only symbol view handed to the expression evaluator.
struct SymTab<'a> {
    syms: &'a [Symbol],
    undefined: &'a RefCell<Vec<String>>,
}

impl SymLookup for SymTab<'_> {
    fn lookup(&self, name: &str) -> Option<SymbolVal> {
        let found = self.syms.iter().position(|s| s.name == name).map(|index| {
            let s = &self.syms[index];
            SymbolVal {
                index,
                defined: s.defined,
                extern_: s.extern_,
                sect: s.sect,
                value: s.value,
                seen: s.seen,
            }
        });
        if found.is_none() {
            self.undefined.borrow_mut().push(name.to_string());
        }
        found
    }
}

pub(crate) struct Fixup {
    pub section: usize,
    /// Absolute section offset of the field to patch.
    pub offset: i64,
    /// Offset of the field within its instruction.
    pub at: u32,
    pub size: u8,
    pub val: RValue,
    pub pcrel: bool,
    /// PC-relative base distance from the instruction start (x86: the
    /// whole instruction; RISC-V branches: zero).
    pub rel_base: u32,
    pub layout: FieldLayout,
    /// Which instruction (by occurrence index) produced this, for
    /// branch-length relaxation.
    pub insn: Option<usize>,
}

impl Fixup {
    /// Absolute section offset a PC-relative displacement is measured
    /// from.
    pub(crate) fn pcrel_base(&self) -> i64 {
        self.offset - self.at as i64 + self.rel_base as i64
    }

    /// Addend for an ELF relocation of this field against `off`. For
    /// PC-relative fields the base is the end of the field (x86's old
    /// `off - size - tail`, i.e. `off - rel_base + at`).
    pub(crate) fn reloc_addend(&self, off: i64) -> i64 {
        if self.pcrel { off - self.rel_base as i64 + self.at as i64 } else { off }
    }
}

const MAX_PASSES: usize = 32;

pub fn assemble(stmts: &[Stmt], arch: crate::arch::Arch) -> Result<Module, Vec<AsmError>> {
    let backend = arch.backend();
    let backend = backend.as_ref();
    let mut sections: Vec<Section> = Vec::new();
    let mut symbols: Vec<Symbol> = Vec::new();
    ensure_section(&mut sections, ".text", SectionFlags::Text);

    // Collect symbols before the first pass so forward references resolve.
    for stmt in stmts {
        match stmt {
            Stmt::Global(names, _line) => {
                for n in names {
                    let sym = intern(&mut symbols, n);
                    symbols[sym].global = true;
                }
            }
            Stmt::Weak(names, _line) => {
                for n in names {
                    let sym = intern(&mut symbols, n);
                    let s = &mut symbols[sym];
                    s.global = true;
                    s.weak = true;
                }
            }
            Stmt::Extern(names, line) => {
                for n in names {
                    let sym = intern(&mut symbols, n);
                    let s = &mut symbols[sym];
                    if s.defined {
                        return Err(vec![AsmError {
                            line: *line,
                            msg: format!("symbol `{n}' both defined and extern"),
                        }]);
                    }
                    s.extern_ = true;
                }
            }
            _ => {}
        }
    }

    // Relaxation state per instruction (0 = most optimistic form; the
    // backend defines the meaning of higher states).
    let mut force_long: HashMap<usize, u8> = HashMap::new();
    for _attempt in 0..MAX_PASSES {
        let undefined = RefCell::new(Vec::new());
        let prev_lens: Vec<i64> = sections.iter().map(|s| s.data.len() as i64).collect();
        let prev_syms: Vec<i64> = symbols.iter().map(|s| s.value).collect();
        for s in &mut sections {
            s.data.clear();
        }
        for s in &mut symbols {
            // Values/sections persist across passes as forward-reference
            // estimates; only definedness resets each pass.
            s.defined = false;
        }
        let (fixups, errors) = run_pass(&mut sections, &mut symbols, stmts, &force_long, &undefined, backend);
        let changed = sections.iter().map(|s| s.data.len() as i64).collect::<Vec<_>>() != prev_lens || symbols.iter().map(|s| s.value).collect::<Vec<_>>() != prev_syms;

        // Relax branches whose displacement outgrew their field, based on
        // real section-relative targets. Externs and other sections have
        // unknown reach; GNU as assumes the far forms for them.
        let mut relaxed = false;
        for f in &fixups {
            if !f.pcrel {
                continue;
            }
            let d = match f.val {
                RValue::SectVal { sect, off, .. } if sect == f.section => Some(off - f.pcrel_base()),
                RValue::SectVal { .. } | RValue::ExternRef { .. } => Some(i64::MAX),
                RValue::Const(_) => None,
            };
            if let Some(d) = d {
                let insn = f.insn.unwrap_or(usize::MAX);
                let current = force_long.get(&insn).copied().unwrap_or(0);
                if let Some(next) = backend.relax_step(f, d, current)
                    && next > current
                {
                    force_long.insert(insn, next);
                    relaxed = true;
                }
            }
        }

        if !changed && !relaxed {
            if !errors.is_empty() {
                return Err(errors);
            }
            let mut unresolved = undefined.borrow().clone();
            unresolved.retain(|n| symbols.iter().find(|s| &s.name == n).map_or(true, |s| !s.defined && !s.extern_));
            if let Some(name) = unresolved.first() {
                return Err(vec![AsmError {
                    line: 0,
                    msg: format!("undefined symbol `{name}'"),
                }]);
            }
            let mut relocs = Vec::new();
            backend
                .finalize_all(sections.as_mut_slice(), &mut symbols, fixups, &mut relocs)
                .map_err(|msgs| msgs.into_iter().map(|m| AsmError { line: 0, msg: m }).collect::<Vec<_>>())?;
            return Ok(Module { sections, symbols, relocs });
        }
        if !errors.is_empty() && !changed {
            return Err(errors);
        }
    }
    Err(vec![AsmError {
        line: 0,
        msg: "assembly did not converge".into(),
    }])
}

fn intern(symbols: &mut Vec<Symbol>, name: &str) -> usize {
    if let Some(i) = symbols.iter().position(|s| s.name == name) {
        return i;
    }
    symbols.push(Symbol {
        name: name.to_string(),
        defined: false,
        seen: false,
        extern_: false,
        global: false,
        weak: false,
        sect: None,
        value: 0,
    });
    symbols.len() - 1
}

fn ensure_section(sections: &mut Vec<Section>, name: &str, flags: SectionFlags) -> usize {
    if let Some(i) = sections.iter().position(|s| s.name == name) {
        return i;
    }
    sections.push(Section {
        name: name.to_string(),
        flags,
        align: 1,
        data: Vec::new(),
    });
    sections.len() - 1
}

type PassResult = (Vec<Fixup>, Vec<AsmError>);

fn run_pass(
    sections: &mut Vec<Section>,
    symbols: &mut Vec<Symbol>,
    stmts: &[Stmt],
    force_long: &HashMap<usize, u8>,
    undefined: &RefCell<Vec<String>>,
    backend: &dyn Backend,
) -> PassResult {
    let mut fixups = Vec::new();
    let mut errors = Vec::new();
    let mut cur_sect = 0usize;
    let mut cur_off = 0i64;
    let mut insn_index = 0usize;
    for stmt in stmts {
        match stmt {
            Stmt::Label(name, line) => {
                let i = intern(symbols, name);
                let s = &mut symbols[i];
                if s.defined || s.extern_ {
                    errors.push(AsmError {
                        line: *line,
                        msg: format!("symbol `{name}' redefined"),
                    });
                }
                s.defined = true;
                s.seen = true;
                s.extern_ = false;
                s.sect = Some(cur_sect);
                s.value = cur_off;
            }
            Stmt::Global(..) | Stmt::Weak(..) | Stmt::Extern(..) => {}
            Stmt::Section(spec) => {
                cur_sect = ensure_section(sections, &spec.name, spec.flags.clone());
                if let Some(a) = spec.align {
                    sections[cur_sect].align = a.max(sections[cur_sect].align);
                }
                cur_off = sections[cur_sect].data.len() as i64;
            }
            Stmt::Bits(bits, line) => {
                if *bits != 64 {
                    errors.push(AsmError {
                        line: *line,
                        msg: format!("bits {bits} is not supported; only 64"),
                    });
                }
            }
            Stmt::Align(e, line) => {
                let tab = SymTab { syms: symbols, undefined };
                let ctx = EvalCtx {
                    cur_sect,
                    cur_off,
                    syms: &tab,
                    optimistic: true,
                    unknown_hit: None,
                };
                match e.eval(&ctx) {
                    Ok(RValue::Const(a)) if a > 0 && (a & (a - 1)) == 0 => {
                        let pad = pad_to(cur_off, a);
                        emit_fill(sections, cur_sect, &mut cur_off, pad, backend.text_fill());
                    }
                    Ok(_) => errors.push(AsmError {
                        line: *line,
                        msg: "align must be a power of two".into(),
                    }),
                    Err(m) => errors.push(AsmError { line: *line, msg: m }),
                }
            }
            Stmt::Data { size, items, line } => {
                let tab = SymTab { syms: symbols, undefined };
                let ctx = EvalCtx {
                    cur_sect,
                    cur_off,
                    syms: &tab,
                    optimistic: true,
                    unknown_hit: None,
                };
                let mut vals: Vec<(Option<RValue>, u8, Vec<u8>)> = Vec::new();
                for item in items {
                    match item {
                        DataItem::Str(bytes) => vals.push((None, *size, bytes.clone())),
                        DataItem::Expr(e) => match e.eval(&ctx) {
                            Ok(v) => vals.push((Some(v), *size, Vec::new())),
                            Err(m) => errors.push(AsmError { line: *line, msg: m }),
                        },
                    }
                }
                drop(ctx);
                for (v, size, bytes) in vals {
                    match v {
                        None => {
                            let data = &mut sections[cur_sect].data;
                            for b in bytes {
                                data.push(b);
                                *(&mut cur_off) += 1;
                            }
                        }
                        Some(v) => emit_field(sections, cur_sect, &mut cur_off, &mut fixups, v, size),
                    }
                }
            }
            Stmt::Reserve { size, count, line } => {
                let tab = SymTab { syms: symbols, undefined };
                let ctx = EvalCtx {
                    cur_sect,
                    cur_off,
                    syms: &tab,
                    optimistic: true,
                    unknown_hit: None,
                };
                let evaluated = count.eval(&ctx);
                drop(ctx);
                match evaluated {
                    Ok(RValue::Const(n)) if n >= 0 => {
                        let nobits = matches!(sections[cur_sect].flags, SectionFlags::Bss) || matches!(sections[cur_sect].flags, SectionFlags::Custom { nobits: true, .. });
                        if !nobits {
                            errors.push(AsmError {
                                line: *line,
                                msg: "attempt to reserve space in an initialized section".into(),
                            });
                        } else {
                            // data.len() doubles as the bss size.
                            for _ in 0..n * *size as i64 {
                                sections[cur_sect].data.push(0);
                                cur_off += 1;
                            }
                        }
                    }
                    Ok(_) => errors.push(AsmError {
                        line: *line,
                        msg: "reserve count must be constant".into(),
                    }),
                    Err(m) => errors.push(AsmError { line: *line, msg: m }),
                }
            }
            Stmt::Times { count, inner, line } => {
                let tab = SymTab { syms: symbols, undefined };
                let ctx = EvalCtx {
                    cur_sect,
                    cur_off,
                    syms: &tab,
                    optimistic: true,
                    unknown_hit: None,
                };
                let count = match count.eval(&ctx) {
                    Ok(RValue::Const(n)) if n >= 0 => n,
                    Ok(_) => {
                        errors.push(AsmError {
                            line: *line,
                            msg: "times count must be constant".into(),
                        });
                        0
                    }
                    Err(m) => {
                        errors.push(AsmError { line: *line, msg: m });
                        0
                    }
                };
                // Inner data items are evaluated once here and repeated.
                let mut vals: Vec<(Option<RValue>, u8, Vec<u8>)> = Vec::new();
                if let Stmt::Data { size, items, .. } = inner.as_ref() {
                    for item in items {
                        match item {
                            DataItem::Str(bytes) => vals.push((None, *size, bytes.clone())),
                            DataItem::Expr(e) => match e.eval(&ctx) {
                                Ok(v) => vals.push((Some(v), *size, Vec::new())),
                                Err(m) => errors.push(AsmError { line: *line, msg: m }),
                            },
                        }
                    }
                }
                drop(ctx);
                for _ in 0..count {
                    match inner.as_ref() {
                        Stmt::Data { .. } => {
                            for (v, size, bytes) in &vals {
                                match v {
                                    None => {
                                        for b in bytes {
                                            sections[cur_sect].data.push(*b);
                                            cur_off += 1;
                                        }
                                    }
                                    Some(v) => emit_field(sections, cur_sect, &mut cur_off, &mut fixups, *v, *size),
                                }
                            }
                        }
                        Stmt::Reserve { size, .. } => {
                            for _ in 0..*size as i64 {
                                sections[cur_sect].data.push(0);
                                cur_off += 1;
                            }
                        }
                        Stmt::Insn(insn) => {
                            emit_insn(
                                backend,
                                sections,
                                &mut cur_off,
                                cur_sect,
                                &mut fixups,
                                &mut errors,
                                insn,
                                symbols,
                                undefined,
                                0,
                                Some(insn_index),
                            );
                            insn_index += 1;
                        }
                        _ => errors.push(AsmError {
                            line: *line,
                            msg: "times only supports data, reserve and instructions".into(),
                        }),
                    }
                }
            }
            Stmt::Comm { name, size, align, global, line } => {
                let tab = SymTab { syms: symbols, undefined };
                let ctx = EvalCtx {
                    cur_sect,
                    cur_off,
                    syms: &tab,
                    optimistic: true,
                    unknown_hit: None,
                };
                let default_align = |n: i64| (1..=63).find(|p| (1i64 << p) >= n).map_or(16, |p| (1i64 << p).min(16));
                let (size, align) = match (size.eval(&ctx), align) {
                    (Ok(RValue::Const(n)), a) if n > 0 => (n, a.unwrap_or(default_align(n) as u64) as i64),
                    (Ok(_), _) => {
                        errors.push(AsmError {
                            line: *line,
                            msg: format!("common symbol `{name}' needs a positive size"),
                        });
                        (0, 1)
                    }
                    (Err(m), _) => {
                        errors.push(AsmError { line: *line, msg: m });
                        (0, 1)
                    }
                };
                drop(ctx);
                // Carve the block out of .bss regardless of the current
                // section.
                let bss = ensure_section(sections, ".bss", SectionFlags::Bss);
                let bss_off = sections[bss].data.len() as i64;
                let pad = pad_to(bss_off, align as i64);
                for _ in 0..pad {
                    sections[bss].data.push(0);
                }
                let i = intern(symbols, name);
                let sym = &mut symbols[i];
                if sym.defined || sym.extern_ {
                    errors.push(AsmError {
                        line: *line,
                        msg: format!("symbol `{name}' redefined"),
                    });
                }
                sym.defined = true;
                sym.seen = true;
                sym.global = *global;
                sym.sect = Some(bss);
                sym.value = sections[bss].data.len() as i64;
                for _ in 0..size {
                    sections[bss].data.push(0);
                }
            }
            Stmt::Insn(insn) => {
                emit_insn(
                    backend,
                    sections,
                    &mut cur_off,
                    cur_sect,
                    &mut fixups,
                    &mut errors,
                    insn,
                    symbols,
                    undefined,
                    force_long.get(&insn_index).copied().unwrap_or(0),
                    Some(insn_index),
                );
                insn_index += 1;
            }
        }
    }

    (fixups, errors)
}

fn pad_to(off: i64, align: i64) -> i64 {
    (-off).rem_euclid(align)
}

fn emit_fill(sections: &mut [Section], sect: usize, cur_off: &mut i64, mut pad: i64, pattern: &[u8]) {
    let pattern: &[u8] = match sections[sect].flags {
        SectionFlags::Text | SectionFlags::Custom { exec: true, .. } => pattern,
        _ => &[0],
    };
    let mut i = 0;
    while pad > 0 {
        sections[sect].data.push(pattern[i % pattern.len()]);
        *cur_off += 1;
        pad -= 1;
        i += 1;
    }
}

/// Emits one data field of `size` bytes, recording a fixup when the value
/// is not a plain constant.
fn emit_field(sections: &mut [Section], sect: usize, cur_off: &mut i64, fixups: &mut Vec<Fixup>, val: RValue, size: u8) {
    let data = &mut sections[sect].data;
    match &val {
        RValue::Const(v) => {
            for i in 0..size as u16 {
                data.push((v >> (i * 8)) as u8);
            }
        }
        v => {
            fixups.push(Fixup {
                section: sect,
                offset: *cur_off,
                at: 0,
                size,
                val: v.clone(),
                pcrel: false,
                rel_base: 0,
                layout: FieldLayout::Flat,
                insn: None,
            });
            for _ in 0..size {
                data.push(0);
            }
        }
    }
    *cur_off += size as i64;
}

#[allow(clippy::too_many_arguments)]
fn emit_insn(
    backend: &dyn Backend,
    sections: &mut [Section],
    cur_off: &mut i64,
    cur_sect: usize,
    fixups: &mut Vec<Fixup>,
    errors: &mut Vec<AsmError>,
    insn: &Insn,
    symbols: &[Symbol],
    undefined: &RefCell<Vec<String>>,
    relax_state: u8,
    insn_index: Option<usize>,
) {
    let tab = SymTab { syms: symbols, undefined };
    let ctx = EvalCtx {
        cur_sect,
        cur_off: *cur_off,
        syms: &tab,
        optimistic: true,
        unknown_hit: None,
    };
    let encoded = backend.encode(insn, &ctx, relax_state);
    drop(ctx);
    match encoded {
        Ok(encoded) => {
            let start = *cur_off;
            for f in encoded.fixups {
                fixups.push(Fixup {
                    section: cur_sect,
                    offset: start + f.at as i64,
                    at: f.at as u32,
                    size: f.size,
                    val: f.val,
                    pcrel: f.pcrel,
                    rel_base: f.rel_base,
                    layout: f.layout,
                    insn: insn_index,
                });
            }
            for b in encoded.bytes {
                sections[cur_sect].data.push(b);
                *cur_off += 1;
            }
        }
        Err(m) => errors.push(AsmError { line: insn.line(), msg: m }),
    }
}
