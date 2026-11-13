//! Inline-asm planning for the native backend: constraint parsing,
//! clobber collection, and template rendering, all table-driven per
//! architecture. x86-64 is implemented; aarch64/riscv64 grow parallel
//! `plan_*`/`render_*` functions when their backends take inline asm.
//!
//! Blocks become one `OpId::ASM` machine instruction whose operands are
//! register-allocated (uses @Early, defs @Late, named constraints
//! pinned); the template is rendered post-RA with the assigned physical
//! registers and assembled by glasm, so the bytes land directly in
//! `.text` — true in-place asm, no wrapper call.

use crate::machine::{AsmSlot, AsmSlotKind, PReg, Width};
use glasm::Arch as GArch;
use glcc_mir as mir;
use std::sync::atomic::{AtomicU32, Ordering};

static ASM_SITES: AtomicU32 = AtomicU32::new(0);

/// A module-wide unique number per asm site, substituted for `%=` (each
/// block is assembled as its own glasm module, so blob-local labels are
/// already distinct; the counter only keeps the rendered text readable).
pub fn next_site() -> u32 {
    ASM_SITES.fetch_add(1, Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// x86-64 register tables
// ---------------------------------------------------------------------------

const GPR_Q: [&str; 16] = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"];
const GPR_K: [&str; 16] = ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d"];
const GPR_W: [&str; 16] = ["ax", "cx", "dx", "bx", "sp", "bp", "si", "di", "r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w", "r15w"];
const GPR_B: [&str; 16] = ["al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil", "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b"];
const GPR_H: [&str; 16] = ["ah", "ch", "dh", "bh", "", "", "", "", "", "", "", "", "", "", "", ""];
const XMM_NAMES: [&str; 16] = ["xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"];

/// The full AT&T name of a GPR at a template modifier (`b`/`w`/`k`/`q`/`h`)
/// or, with no modifier, at the operand's own width.
fn gpr_name(hw: u8, modifier: Option<char>, width: Width) -> &'static str {
    let name = match modifier {
        None => match width {
            Width::W8 => GPR_B[hw as usize],
            Width::W16 => GPR_W[hw as usize],
            Width::W32 => GPR_K[hw as usize],
            Width::W64 => GPR_Q[hw as usize],
        },
        Some('b') => GPR_B[hw as usize],
        Some('w') => GPR_W[hw as usize],
        Some('k') => GPR_K[hw as usize],
        Some('q') => GPR_Q[hw as usize],
        Some('h') => GPR_H[hw as usize],
        Some(other) => panic!("unsupported asm modifier %{other} on a register operand"),
    };
    if name.is_empty() {
        panic!("glcc native backend: register {} has no high-byte form", GPR_Q[hw as usize]);
    }
    name
}

/// Canonical register for any spelling a clobber list or template can
/// use (any GPR width, `xmmN`).
fn reg_by_name(name: &str) -> Option<PReg> {
    if let Some(hw) = GPR_SPELLINGS.iter().find(|(spelling, _)| *spelling == name).map(|(_, hw)| *hw) {
        return Some(PReg::gpr(hw));
    }
    if let Some(rest) = name.strip_prefix("xmm") {
        return rest.parse::<u8>().ok().filter(|&n| n < 16).map(PReg::vec);
    }
    None
}

/// Every spelling a template or clobber list can use for a GPR, with
/// the hardware encoding it names.
const GPR_SPELLINGS: [(&str, u8); 68] = {
    let mut out: [(&str, u8); 68] = [("", 0); 68];
    let mut n = 0;
    let mut hw = 0;
    while hw < 16 {
        let spellings = [GPR_Q[hw], GPR_K[hw], GPR_W[hw], GPR_B[hw], GPR_H[hw]];
        let mut si = 0;
        while si < 5 {
            let name = spellings[si];
            if !name.is_empty() {
                out[n] = (name, hw as u8);
                n += 1;
            }
            si += 1;
        }
        hw += 1;
    }
    out
};

// ---------------------------------------------------------------------------
// Constraint parsing
// ---------------------------------------------------------------------------

/// How an operand reaches the template.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Reg,
    Mem,
    Imm,
}

struct Constraint {
    kind: Kind,
    /// Pinned GPR (`"a"` is rax), by hardware encoding.
    named: Option<u8>,
    /// The output this input shares a location with.
    tie: Option<u32>,
    inout: bool,
    /// The output is written before the inputs are consumed (`&`).
    early: bool,
    reg_alt: bool,
    mem_alt: bool,
    /// The operand must be in a/b/c/d ("Q" / gcc's rh set).
    abcd: bool,
}

fn parse_constraint(constraint: &str) -> Constraint {
    parse_constraint_checked(constraint).unwrap_or_else(|reason| panic!("glcc native backend: {reason}"))
}

fn parse_constraint_checked(constraint: &str) -> Result<Constraint, String> {
    let mut rest = constraint;
    let mut inout = false;
    let mut early = false;
    loop {
        if let Some(r) = rest.strip_prefix('=') {
            rest = r;
        } else if let Some(r) = rest.strip_prefix('+') {
            inout = true;
            rest = r;
        } else if let Some(r) = rest.strip_prefix('&') {
            early = true;
            rest = r;
        } else {
            break;
        }
    }
    if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
        return Ok(Constraint { kind: Kind::Reg, named: None, tie: rest.parse().ok(), inout, early, reg_alt: true, mem_alt: false, abcd: false });
    }
    let mut named = None;
    let mut kind = None;
    let mut reg_alt = false;
    let mut mem_alt = false;
    let mut abcd = false;
    while let Some(first) = rest.chars().next() {
        let (letter, tail) = rest.split_at(first.len_utf8());
        rest = tail;
        let (k, n) = match letter {
            "r" | "q" | "g" | "X" => {
                reg_alt = true;
                (Some(Kind::Reg), None)
            }
            // `"A"` nominally selects the rdx:rax pair; like the wrapper
            // path, only rax is carried.
            "a" | "A" => {
                reg_alt = true;
                (Some(Kind::Reg), Some(0))
            }
            "b" => {
                reg_alt = true;
                (Some(Kind::Reg), Some(3))
            }
            "c" => {
                reg_alt = true;
                (Some(Kind::Reg), Some(1))
            }
            "d" => {
                reg_alt = true;
                (Some(Kind::Reg), Some(2))
            }
            "S" => {
                reg_alt = true;
                (Some(Kind::Reg), Some(6))
            }
            "D" => {
                reg_alt = true;
                (Some(Kind::Reg), Some(7))
            }
            // "Q" is any high-byte-addressable register: a, b, c, d.
            "Q" => {
                reg_alt = true;
                abcd = true;
                (Some(Kind::Reg), None)
            }
            "x" | "v" => return Err("vector register operands are not supported by the native backend yet".to_string()),
            "m" | "o" | "V" => {
                mem_alt = true;
                (Some(Kind::Mem), None)
            }
            "i" | "n" | "s" | "I" | "J" | "K" | "L" | "M" | "N" | "Z" | "e" => (Some(Kind::Imm), None),
            "f" | "t" | "u" | "y" => {
                return Err(format!("asm constraint '{letter}' selects a floating-point register, which is not supported by the native backend"))
            }
            _ => return Err(format!("unsupported asm constraint '{letter}'")),
        };
        if let Some(n) = n {
            named = Some(n);
        }
        // Register classes beat memory, memory beats immediates.
        if k == Some(Kind::Reg) && kind != Some(Kind::Imm) {
            kind = k;
        } else if k.is_some() && kind.is_none() {
            kind = k;
        }
    }
    if named.is_some() {
        kind = Some(Kind::Reg);
    }
    Ok(Constraint { kind: kind.ok_or_else(|| "empty asm constraint".to_string())?, named, tie: None, inout, early, reg_alt, mem_alt, abcd })
}

/// Size in bytes of an operand's type.
fn ty_size(ty: &mir::Ty) -> u64 {
    match ty {
        mir::Ty::Bool => 1,
        mir::Ty::Int { bits } => u64::from(bits.div_ceil(8)).max(1),
        mir::Ty::WideInt { bits } => u64::from(bits.div_ceil(8)).max(1),
        mir::Ty::Float(mir::FloatKind::F32) => 4,
        mir::Ty::Float(mir::FloatKind::F64) => 8,
        mir::Ty::Float(_) => 16,
        mir::Ty::Ptr { .. } => 8,
        mir::Ty::Array { .. } | mir::Ty::Struct { .. } | mir::Ty::Vector { .. } => ty.size_bytes(),
        mir::Ty::BuiltinVaList { size, .. } => *size,
        mir::Ty::Void => panic!("glcc native backend: void asm operand"),
    }
}

fn width_of(size: u64) -> Width {
    match size {
        1 => Width::W8,
        2 => Width::W16,
        4 => Width::W32,
        _ => Width::W64,
    }
}

fn const_int(func: &mir::Function, value: mir::Value) -> i64 {
    match &func.values[value].def {
        mir::Def::Const(mir::Const::Int { value, .. }) => i64::try_from(*value)
            .unwrap_or_else(|_| panic!("glcc native backend: asm immediate operand does not fit in 64 bits")),
        mir::Def::Const(mir::Const::Null) | mir::Def::Const(mir::Const::Undef) => 0,
        _ => panic!("glcc native backend: asm operand constraint requires an integer constant"),
    }
}

fn is_int_const(func: &mir::Function, value: mir::Value) -> bool {
    matches!(
        &func.values[value].def,
        mir::Def::Const(mir::Const::Int { .. } | mir::Const::Null | mir::Const::Undef)
    )
}

/// Canonical pregs a clobber string names; `None` for `memory`/`cc` and
/// other non-register names. Stack/frame pointer clobbers panic.
fn parse_clobber(clobber: &str) -> Option<PReg> {
    let name = clobber.strip_prefix('%').unwrap_or(clobber);
    match reg_by_name(name) {
        Some(preg) if preg.class == crate::machine::RegClass::Gpr && matches!(preg.hw, 4 | 5) => {
            panic!("glcc native backend: clobbering the {} is not supported", GPR_Q[preg.hw as usize])
        }
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Template register detection
// ---------------------------------------------------------------------------

/// Whether `text` mentions `name` as a standalone token.
fn mentions_token(text: &str, name: &str) -> bool {
    let bytes = text.as_bytes();
    let ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut from = 0;
    while let Some(pos) = text[from..].find(name) {
        let at = from + pos;
        let end = at + name.len();
        let before_ok = at == 0 || !ident(bytes[at - 1]);
        let after_ok = end >= bytes.len() || !ident(bytes[end]);
        if before_ok && after_ok {
            return true;
        }
        from = end;
    }
    false
}

/// Registers the template mentions literally (`%%rax`, `%%xmm3`, `%%al`).
/// The scan works over a copy with `%`-escapes blanked, so `%%` opens a
/// literal token while `%N`/`%[name]`/`%l[name]` are operand references.
fn mentioned_regs(template: &str) -> Vec<&'static str> {
    let mut literal = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            literal.push(c);
            continue;
        }
        match chars.peek() {
            Some('%') => {
                chars.next();
            }
            Some('[') => {
                for c in chars.by_ref() {
                    if c == ']' {
                        break;
                    }
                }
            }
            Some('l') => {
                chars.next();
                if chars.peek() == Some(&'[') {
                    for c in chars.by_ref() {
                        if c == ']' {
                            break;
                        }
                    }
                }
            }
            _ => {
                while chars.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
                    chars.next();
                }
                while chars.peek().is_some_and(|c| c.is_ascii_digit()) {
                    chars.next();
                }
            }
        }
        literal.push(' ');
    }
    let mut found = Vec::new();
    for (spelling, _) in GPR_SPELLINGS {
        if mentions_token(&literal, spelling) && !found.contains(&spelling) {
            found.push(spelling);
        }
    }
    for name in XMM_NAMES {
        if mentions_token(&literal, name) {
            found.push(name);
        }
    }
    found
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

/// Operands the template reads through the high-byte modifier (`%hN`).
/// Named operands can't carry a modifier in the render grammar, so the
/// scan only needs the `%h<digit>` form.
fn hi_byte_refs(template: &str) -> Vec<usize> {
    let b = template.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'%' {
            i += 1;
            continue;
        }
        if i + 1 < b.len() && b[i + 1] == b'%' {
            i += 2;
            continue;
        }
        let mut j = i + 1;
        let modified = j < b.len() && b[j].is_ascii_alphabetic();
        if modified {
            j += 1;
        }
        let start = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if modified && b[i + 1] == b'h' && j > start {
            out.push(template[start..j].parse().unwrap());
        }
        i = j.max(i + 1);
    }
    out
}

/// Operand references (`%N`, `%[name]`) in one template line, resolved
/// through the slot names.
fn operand_refs_in(line: &str, names: &[Option<String>]) -> Vec<usize> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < b.len() {
        if b[i] != b'%' {
            i += 1;
            continue;
        }
        if b[i + 1] == b'%' {
            i += 2;
            continue;
        }
        let mut j = i + 1;
        if b[j] == b'[' {
            let start = j + 1;
            while j < b.len() && b[j] != b']' {
                j += 1;
            }
            let name = &line[start..j.min(b.len())];
            j += 1;
            let mut digits = String::new();
            while j < b.len() && b[j].is_ascii_digit() {
                digits.push(b[j] as char);
                j += 1;
            }
            if !digits.is_empty() {
                out.push(digits.parse().unwrap());
            } else if let Some(idx) = names.iter().position(|n| n.as_deref() == Some(name)) {
                out.push(idx);
            }
            i = j;
            continue;
        }
        if b[j].is_ascii_alphabetic() {
            j += 1;
        }
        let start = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j > start {
            out.push(line[start..j].parse().unwrap());
        }
        i = j.max(i + 1);
    }
    out
}

/// The selection-side plan for one asm block.
pub struct AsmPlan {
    pub slots: Vec<AsmSlot>,
    /// Registers written by the template beyond its operand defs: named
    /// clobbers plus literal register mentions. Never includes rsp/rbp.
    pub clobbers: Vec<PReg>,
}

/// Plan one asm block for the x86-64 native backend. Panics loudly on
/// shapes the backend cannot honor (stable messages; tests match on
/// them), mirroring the rest of the native backend.
pub fn plan_x86_64(block: &mir::AsmBlock, func: &mir::Function) -> AsmPlan {
    let outputs = block.outputs as usize;
    let mut slots: Vec<AsmSlot> = Vec::with_capacity(block.operands.len());
    let mut abcd_ops: Vec<usize> = Vec::new();

    for (i, operand) in block.operands.iter().enumerate() {
        let constraint = parse_constraint(&operand.constraint);
        let is_output = i < outputs;
        if is_output && constraint.tie.is_some() {
            panic!("glcc native backend: asm output cannot be a matching constraint");
        }
        if is_output && constraint.kind == Kind::Imm {
            panic!("glcc native backend: asm output cannot be an immediate");
        }
        // A tied input rides its target output's slot: the output's
        // read side carries the input value (selection wires that up),
        // the def carries the tie, and this slot renders the target's
        // register so GCC numbering keeps working.
        if !is_output {
            if let Some(tie) = constraint.tie {
                let target = tie as usize;
                if target >= outputs {
                    panic!("glcc native backend: asm operand number out of range");
                }
                let target_slot = &slots[target];
                slots.push(AsmSlot {
                    kind: target_slot.kind,
                    width: target_slot.width,
                    is_use: false,
                    is_def: false,
                    early: false,
                    pin: target_slot.pin,
                    rides: Some(target),
                    name: operand.name.clone(),
                });
                continue;
            }
        }
        let size = ty_size(&operand.ty);
        // An operand with several alternatives (`"Ir"`) takes the
        // immediate only when the value is a constant, matching GCC's
        // constraint matching; otherwise a register or memory
        // alternative carries it.
        let mut kind = constraint.kind;
        let mut imm = None;
        match kind {
            Kind::Imm if is_int_const(func, operand.value) => imm = Some(const_int(func, operand.value)),
            Kind::Imm => {
                if constraint.reg_alt {
                    kind = Kind::Reg;
                } else if constraint.mem_alt {
                    kind = Kind::Mem;
                } else {
                    panic!("glcc native backend: asm operand constraint requires an integer constant");
                }
            }
            _ => {}
        }
        if kind == Kind::Reg {
            if size > 8 {
                panic!("glcc native backend: asm register operand is wider than a register");
            }
            if matches!(operand.ty, mir::Ty::Array { .. } | mir::Ty::Struct { .. } | mir::Ty::Vector { .. } | mir::Ty::Float(_)) {
                panic!("glcc native backend: aggregate and floating-point register asm operands are not supported yet");
            }
        }
        // Memory slots are always address uses: the template reads and
        // writes the location in place, so even a `+m` output carries
        // no register def. A register output's read side (its `+`
        // current value or a tied input's value) is carried by the
        // slot's use operand; the def ties to it.
        let tied_by_input = is_output && block.operands.iter().skip(outputs).any(|op| op.tie == Some(i as u32));
        let (is_use, is_def) = match kind {
            Kind::Mem => (true, false),
            _ => (!is_output || constraint.inout || tied_by_input, is_output),
        };
        if constraint.abcd && kind == Kind::Reg {
            abcd_ops.push(i);
        }
        slots.push(AsmSlot {
            kind: match kind {
                Kind::Reg => AsmSlotKind::Reg,
                Kind::Mem => AsmSlotKind::Mem,
                Kind::Imm => AsmSlotKind::Imm(imm.expect("immediate operand without a value")),
            },
            width: width_of(size),
            is_use,
            is_def,
            early: is_output && constraint.early,
            pin: constraint.named.map(PReg::gpr),
            rides: None,
            name: operand.name.clone(),
        });
    }

    let mut clobbers: Vec<PReg> = Vec::new();
    for clobber in &block.clobbers {
        if let Some(preg) = parse_clobber(clobber) {
            if !clobbers.contains(&preg) {
                clobbers.push(preg);
            }
        }
    }

    // Registers the template mentions literally are clobbered even when
    // undeclared — a template scratch register must not hold a live
    // value across the block. rsp/rbp are never allocatable so their
    // mentions need no clobber (reading them is legitimate).
    let pinned: Vec<PReg> = slots.iter().filter_map(|slot| slot.pin).collect();
    for name in mentioned_regs(&block.template) {
        let Some(preg) = reg_by_name(name) else { continue };
        if preg.class == crate::machine::RegClass::Gpr && matches!(preg.hw, 4 | 5) {
            continue;
        }
        if !clobbers.contains(&preg) && !pinned.contains(&preg) {
            clobbers.push(preg);
        }
    }

    // `%hN` and "Q" operands must land in a/b/c/d (gcc's rh set), and
    // a high-byte register cannot share an instruction with a REX
    // operand — so every operand referenced on a `%h` line joins the
    // same set. Regalloc has no class subsets, so pin each such slot to
    // a distinct register the rest of the plan leaves free.
    let mut pinned: Vec<PReg> = slots.iter().filter_map(|slot| slot.pin).collect();
    let mut used: Vec<u8> = pinned
        .iter()
        .map(|p| p.hw)
        .chain(clobbers.iter().map(|p| p.hw))
        .collect();
    let names: Vec<Option<String>> = slots.iter().map(|s| s.name.clone()).collect();
    let mut refs = abcd_ops;
    if !hi_byte_refs(&block.template).is_empty() {
        for line in block.template.lines() {
            if !line.contains("%h") {
                continue;
            }
            for idx in operand_refs_in(line, &names) {
                if idx < slots.len() && !refs.contains(&idx) {
                    refs.push(idx);
                }
            }
        }
    }
    for op_idx in refs {
        let target = slots[op_idx].rides.unwrap_or(op_idx);
        if slots[target].pin.is_some() {
            continue;
        }
        let Some(hw) = (0u8..4).find(|hw| !used.contains(hw)) else {
            panic!("glcc native backend: no free a/b/c/d register for the asm's high-byte operands");
        };
        let preg = PReg::gpr(hw);
        slots[target].pin = Some(preg);
        pinned.push(preg);
        used.push(hw);
    }

    // A declared clobber may not name a register an operand pins.
    for pin in &pinned {
        if clobbers.contains(pin) {
            panic!(
                "glcc native backend: asm constraint names the clobbered register {}",
                GPR_Q[pin.hw as usize]
            );
        }
    }

    if !block.labels.is_empty() {
        slots.push(AsmSlot { kind: AsmSlotKind::Sel, width: Width::W32, is_use: false, is_def: true, early: false, pin: None, rides: None, name: None });
    }

    AsmPlan { slots, clobbers }
}

// ---------------------------------------------------------------------------
// Template rendering and blob assembly
// ---------------------------------------------------------------------------

/// Render context: the physical register assigned to each RA operand of
/// the block (probe mode passes a stand-in for every slot).
pub struct RenderCx<'a> {
    pub slots: &'a [AsmSlot],
    /// Physical register of slot `i` (Reg/Mem/Sel slots only).
    pub preg_of: &'a dyn Fn(usize) -> PReg,
    pub labels: &'a [String],
    pub unique: u32,
}

/// Render the template for x86-64: `%%`, `%N`, `%cN`/`%nN`,
/// `%bN`/`%wN`/`%kN`/`%qN`/`%hN`, `%[name]`, `%l[name]`, and `%=`.
pub fn render_x86_64(template: &str, cx: &RenderCx) -> String {
    let cx_template_debug = template;
    let mut out = String::with_capacity(template.len() + 16);
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.peek() {
            Some('%') => {
                chars.next();
                out.push('%');
            }
            Some('=') => {
                chars.next();
                out.push_str(&cx.unique.to_string());
            }
            Some('l') => {
                chars.next();
                if chars.next() != Some('[') {
                    panic!("expected '[' after %l in asm template");
                }
                let mut name = String::new();
                for c in chars.by_ref() {
                    if c == ']' {
                        break;
                    }
                    name.push(c);
                }
                let Some(index) = cx.labels.iter().position(|label| *label == name) else {
                    panic!("glcc native backend: asm template references unknown label [{name}]");
                };
                out.push_str(&format!(".Lasm{}_{index}", cx.unique));
            }
            Some('[') => {
                chars.next();
                let mut name = String::new();
                for c in chars.by_ref() {
                    if c == ']' {
                        break;
                    }
                    name.push(c);
                }
                let Some(index) = cx.slots.iter().position(|slot| slot.name.as_deref() == Some(&name)) else {
                    panic!("glcc native backend: asm template references unknown operand name [{name}]");
                };
                out.push_str(&expand_x86_64(cx, index, None));
            }
            Some('%') => {
                // GCC escapes a literal percent sign as `%%`.
                chars.next();
                out.push('%');
            }
            _ => {
                let modifier = match chars.peek() {
                    Some(c) if c.is_ascii_alphabetic() => {
                        let c = *c;
                        chars.next();
                        Some(c)
                    }
                    _ => None,
                };
                // A modifier may apply to a named operand too
                // (`%c[size]` prints the constant without the `$`).
                if chars.peek() == Some(&'[') {
                    chars.next();
                    let mut name = String::new();
                    for c in chars.by_ref() {
                        if c == ']' {
                            break;
                        }
                        name.push(c);
                    }
                    let Some(index) = cx.slots.iter().position(|slot| slot.name.as_deref() == Some(&name)) else {
                        panic!("glcc native backend: asm template references unknown operand name [{name}]");
                    };
                    out.push_str(&expand_x86_64(cx, index, modifier));
                    continue;
                }
                let mut digits = String::new();
                while let Some(c) = chars.peek() {
                    if c.is_ascii_digit() {
                        digits.push(*c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                let Ok(index) = digits.parse::<usize>() else {
                    panic!("bad `%` sequence in asm template: %{}{modifier:?}
TEMPLATE:
{}", digits, cx_template_debug);
                };
                if index >= cx.slots.len() {
                    panic!("glcc native backend: asm template references operand {index} which does not exist");
                }
                out.push_str(&expand_x86_64(cx, index, modifier));
            }
        }
    }
    out
}

fn expand_x86_64(cx: &RenderCx, index: usize, modifier: Option<char>) -> String {
    let slot = &cx.slots[index];
    match &slot.kind {
        AsmSlotKind::Imm(value) => match modifier {
            // `%nN` prints the negated constant (xor-style templates).
            Some('n') => format!("{}", -value),
            Some('c') => format!("{value}"),
            Some('b' | 'w' | 'k' | 'q') | None => format!("${value}"),
            Some(other) => panic!("unsupported asm modifier %{other} on an immediate"),
        },
        AsmSlotKind::Mem => {
            let preg = (cx.preg_of)(index);
            match modifier {
                None => format!("(%{})", GPR_Q[preg.hw as usize]),
                Some(other) => panic!("unsupported asm modifier %{other} on a memory operand"),
            }
        }
        AsmSlotKind::Reg => format!("%{}", gpr_name((cx.preg_of)(index).hw, modifier, slot.width)),
        AsmSlotKind::Sel => panic!("glcc native backend: the asm goto selector is not a numbered operand"),
    }
}

/// A relocation the assembled fragment needs, carrying the glasm
/// symbol name for the caller to resolve.
#[derive(Clone, Debug)]
pub struct FragmentReloc {
    pub offset: u32,
    pub size: u8,
    pub pcrel: bool,
    pub addend: i64,
    /// `None`: the relocation targets the named symbol in `sym`.
    /// `Some((section, offset))`: it targets a fragment-local label
    /// defined at that position — the caller synthesizes a symbol.
    pub local: Option<(String, i64)>,
    pub sym: String,
}

/// A non-`.text` section the fragment's `.pushsection` wrote (the
/// kernel's ALTERNATIVE/ANNOTATE blocks emit metadata this way).
#[derive(Clone, Debug)]
pub struct FragmentSection {
    pub name: String,
    pub write: bool,
    pub exec: bool,
    pub nobits: bool,
    pub data: Vec<u8>,
    pub relocs: Vec<FragmentReloc>,
}

/// Assemble AT&T x86-64 text into `.text` bytes, its symbol
/// relocations, and any side sections the fragment created (the caller
/// resolves relocation names).
pub fn assemble_x86_64(text: &str) -> Result<(Vec<u8>, Vec<FragmentReloc>, Vec<FragmentSection>), String> {
    let stmts = glasm::gas::parse_module(text, GArch::X86_64)
        .map_err(|errs| errs.iter().map(|e| format!("line {}: {}", e.line, e.msg)).collect::<Vec<_>>().join("; "))?;
    let module = glasm::assemble::assemble_with(&stmts, GArch::X86_64, false)
        .map_err(|errs| errs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; "))?;
    let local_of = |r: &glasm::assemble::Reloc| -> Result<Option<(String, i64)>, String> {
        match &r.target {
            glasm::assemble::Target::Symbol(i) => {
                let s = &module.symbols[*i];
                if s.defined && !s.extern_
                    && let Some(sect) = s.sect {
                        return Ok(Some((module.sections[sect].name.clone(), s.value)));
                    }
                Ok(None)
            }
            _ => Ok(None),
        }
    };
    let name_of = |r: &glasm::assemble::Reloc| -> Result<String, String> {
        match &r.target {
            glasm::assemble::Target::Symbol(i) => Ok(module.symbols[*i].name.clone()),
            // A fragment-local target (a local label's section) gets an
            // unambiguous marker; the caller retargets it to the
            // enclosing function.
            glasm::assemble::Target::Section(s) => Ok(format!("<section {}>", module.sections[*s].name)),
            glasm::assemble::Target::Abs => Err("inline asm references the null symbol".to_string()),
        }
    };
    let mut relocs = Vec::new();
    let mut sides: Vec<FragmentSection> = Vec::new();
    for r in &module.relocs {
        let rel = FragmentReloc {
            offset: r.offset as u32,
            size: r.size,
            pcrel: r.pcrel,
            addend: r.addend,
            local: local_of(r)?,
            sym: name_of(r)?,
        };
        if r.section == 0 && module.sections[0].name == ".text" {
            relocs.push(rel);
        } else {
            let sect = &module.sections[r.section];
            match sides.iter_mut().find(|s| s.name == sect.name) {
                Some(side) => side.relocs.push(rel),
                None => sides.push(FragmentSection { name: sect.name.clone(), write: false, exec: false, nobits: false, data: Vec::new(), relocs: vec![rel] }),
            }
        }
    }
    let mut code = Vec::new();
    for section in &module.sections {
        if section.name == ".text" {
            code.extend_from_slice(&section.data);
            continue;
        }
        if section.data.is_empty() {
            continue;
        }
        let (write, exec, nobits) = match section.flags {
            glasm::insn::SectionFlags::Text => (false, true, false),
            glasm::insn::SectionFlags::Data => (true, false, false),
            glasm::insn::SectionFlags::Bss => (true, false, true),
            glasm::insn::SectionFlags::Rodata => (false, false, false),
            glasm::insn::SectionFlags::Custom { write, exec, nobits, .. } => (write, exec, nobits),
        };
        match sides.iter_mut().find(|s| s.name == section.name) {
            Some(side) => {
                side.data = section.data.clone();
                side.write = write;
                side.exec = exec;
                side.nobits = nobits;
            }
            None => sides.push(FragmentSection { name: section.name.clone(), write, exec, nobits, data: section.data.clone(), relocs: Vec::new() }),
        }
    }
    Ok((code, relocs, sides))
}

/// The complete assembled blob for one block: rendered template plus,
/// for `asm goto`, the per-label selector stubs (defined before the
/// template, so execution falls into no stub) and the fallthrough clear.
pub fn blob_x86_64(
    template: &str,
    slots: &[AsmSlot],
    labels: &[String],
    is_basic: bool,
    is_goto: bool,
    unique: u32,
    preg_of: &dyn Fn(usize) -> PReg,
) -> Result<(Vec<u8>, Vec<FragmentReloc>, Vec<FragmentSection>), String> {
    let cx = RenderCx { slots, preg_of, labels, unique };
    let mut text = String::new();
    if is_goto {
        let sel_slot = slots.iter().position(|slot| matches!(slot.kind, AsmSlotKind::Sel)).expect("goto block without a selector slot");
        let sel = GPR_K[(cx.preg_of)(sel_slot).hw as usize];
        // The stubs live before the template: control enters the blob
        // at its start, so a jump skips over them, and each stub writes
        // its selector number and rejoins after the fallthrough clear.
        text.push_str(&format!("\tjmp .Lasm{unique}_entry\n"));
        for i in 0..labels.len() {
            text.push_str(&format!(".Lasm{unique}_{i}:\n\tmovl ${}, %{sel}\n\tjmp .Lasm{unique}_merge\n", i + 1));
        }
        text.push_str(&format!(".Lasm{unique}_entry:\n"));
    }
    if is_basic {
        text.push_str(template);
    } else {
        text.push_str(&render_x86_64(template, &cx));
    }
    if !text.ends_with('\n') {
        text.push('\n');
    }
    if is_goto {
        let sel_slot = slots.iter().position(|slot| matches!(slot.kind, AsmSlotKind::Sel)).expect("goto block without a selector slot");
        let sel = GPR_K[(cx.preg_of)(sel_slot).hw as usize];
        text.push_str(&format!("\txorl %{sel}, %{sel}\n\t.Lasm{unique}_merge:\n"));
    }
    assemble_x86_64(&text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine::{AsmSlotKind, Width};
    use glcc_mir::{Function, Signature};

    fn func_with_const(value: i128) -> (Function, mir::Value) {
        let (mut f, _entry) = Function::new("t", Signature::new(vec![], mir::Ty::Void));
        let v = f.push_const(mir::Const::Int { value, bits: 64 });
        (f, v)
    }

    fn block(template: &str, operands: Vec<mir::AsmOperand>, outputs: u32, clobbers: &[&str]) -> mir::AsmBlock {
        mir::AsmBlock {
            template: template.to_string(),
            operands,
            outputs,
            clobbers: clobbers.iter().map(|c| c.to_string()).collect(),
            labels: vec![],
            is_volatile: false,
            is_basic: false,
        }
    }

    fn reg_operand(constraint: &str) -> mir::AsmOperand {
        mir::AsmOperand { value: mir::Value::from_usize(0), constraint: constraint.to_string(), ty: mir::Ty::Int { bits: 64 }, tie: None, name: None }
    }

    #[test]
    fn parses_register_mem_and_pinned_constraints() {
        let c = parse_constraint("=r");
        assert_eq!(c.kind, Kind::Reg);
        assert!(c.named.is_none());
        let c = parse_constraint("a");
        assert_eq!(c.named, Some(0));
        let c = parse_constraint("=D");
        assert_eq!(c.named, Some(7));
        let c = parse_constraint("+m");
        assert_eq!(c.kind, Kind::Mem);
        let c = parse_constraint("=rm");
        assert_eq!(c.kind, Kind::Reg, "register beats memory");
        assert!(c.mem_alt);
        let c = parse_constraint("0");
        assert_eq!(c.tie, Some(0));
    }

    #[test]
    #[should_panic = "asm constraint 't' selects a floating-point register"]
    fn rejects_x87_constraints() {
        parse_constraint("=t");
    }

    #[test]
    fn immediates_need_constants() {
        let (f, _) = func_with_const(42);
        let mut op = reg_operand("i");
        op.ty = mir::Ty::Int { bits: 32 };
        let plan = plan_x86_64(&block("", vec![op], 0, &[]), &f);
        assert!(matches!(plan.slots[0].kind, AsmSlotKind::Imm(42)));
    }

    #[test]
    fn tied_input_rides_its_output() {
        let (mut f, _) = Function::new("t", Signature::new(vec![], mir::Ty::Void));
        let val = f.push_const(mir::Const::Int { value: 7, bits: 64 });
        let out = reg_operand("=r");
        let mut tied = reg_operand("0");
        tied.tie = Some(0);
        tied.value = val;
        let plan = plan_x86_64(&block("", vec![out, tied], 1, &[]), &f);
        assert_eq!(plan.slots.len(), 2);
        assert!(plan.slots[0].is_def);
        assert_eq!(plan.slots[1].rides, Some(0), "tied input renders the output's register");
    }

    #[test]
    fn detects_literal_register_mentions() {
        let found = mentioned_regs("movl $0, %%eax; cpuid");
        assert!(found.contains(&"eax"));
        assert!(!mentioned_regs("%0").contains(&"rax"), "operand refs are not literal registers");
        assert!(!mentioned_regs("%[rax]").contains(&"rax"), "named operand refs are not literal registers");
        assert!(mentioned_regs("movb %%al, (%rdi)").contains(&"al"));
        assert!(mentioned_regs("pxor %%xmm3, %%xmm3").contains(&"xmm3"));
    }

    #[test]
    fn renders_and_assembles_a_blob() {
        // Two register operands at stand-in r11: %0 is the output.
        let slots = vec![
            AsmSlot { kind: AsmSlotKind::Reg, width: Width::W64, is_use: true, is_def: true, early: false, pin: None, rides: None, name: None },
            AsmSlot { kind: AsmSlotKind::Reg, width: Width::W64, is_use: false, is_def: false, early: false, pin: None, rides: None, name: None },
        ];
        let bytes = blob_x86_64("movq %1, %0\n\tincq %0", &slots, &[], false, false, 0, &|_| PReg::gpr(11)).expect("assembles");
        assert!(!bytes.is_empty());
    }

    #[test]
    fn goto_blob_clears_and_sets_the_selector() {
        let slots = vec![
            AsmSlot { kind: AsmSlotKind::Reg, width: Width::W64, is_use: false, is_def: false, early: false, pin: None, rides: None, name: None },
            AsmSlot { kind: AsmSlotKind::Sel, width: Width::W32, is_use: false, is_def: true, early: false, pin: None, rides: None, name: None },
        ];
        // The stub and merge labels resolve inside the blob; the stub
        // writes 1 and the fallthrough clears the selector.
        let bytes = blob_x86_64("jmp %l[done]", &slots, &["done".to_string()], false, true, 7, &|_| PReg::gpr(11)).expect("goto stubs assemble");
        assert!(!bytes.is_empty());
    }

    #[test]
    fn renders_immediate_modifiers() {
        let slots = vec![AsmSlot { kind: AsmSlotKind::Imm(-3), width: Width::W32, is_use: false, is_def: false, early: false, pin: None, rides: None, name: None }];
        let cx = RenderCx { slots: &slots, preg_of: &|_| PReg::gpr(11), labels: &[], unique: 4 };
        assert_eq!(render_x86_64("%0 %c0 %n0 %=", &cx), "$-3 -3 3 4");
    }

    #[test]
    fn parses_q_as_the_high_byte_class() {
        let c = parse_constraint("Q");
        assert_eq!(c.kind, Kind::Reg);
        assert!(c.abcd);
        assert!(c.named.is_none());
    }

    #[test]
    fn high_byte_operands_pin_to_a_b_c_d() {
        let (f, _) = Function::new("t", Signature::new(vec![], mir::Ty::Void));
        // Both operands of the %h line need a/b/c/d: the %h output and
        // the other operand of the same instruction (a REX register
        // cannot coexist with a high-byte register).
        let plan = plan_x86_64(&block("movb %1, %h0", vec![reg_operand("=r"), reg_operand("r")], 1, &[]), &f);
        let hw0 = plan.slots[0].pin.expect("%h output pins into a/b/c/d").hw;
        let hw1 = plan.slots[1].pin.expect("same-line operand pins into a/b/c/d too").hw;
        assert!(hw0 < 4 && hw1 < 4);
        assert_ne!(hw0, hw1, "the two operands get distinct a/b/c/d registers");
    }

    #[test]
    fn q_constraint_pins_outside_the_high_byte_lines() {
        let (f, _) = Function::new("t", Signature::new(vec![], mir::Ty::Void));
        // A "Q" operand on a %h-free line still needs a/b/c/d.
        let plan = plan_x86_64(&block("movb %1, %0", vec![reg_operand("=Q"), reg_operand("r")], 1, &[]), &f);
        assert_eq!(plan.slots[0].pin, Some(PReg::gpr(0)));
        assert_eq!(plan.slots[1].pin, None, "plain r operands stay free");
    }

    #[test]
    fn high_byte_modifier_rejects_r11() {
        let slots = vec![AsmSlot { kind: AsmSlotKind::Reg, width: Width::W8, is_use: false, is_def: false, early: false, pin: None, rides: None, name: None }];
        let cx = RenderCx { slots: &slots, preg_of: &|_| PReg::gpr(11), labels: &[], unique: 0 };
        let cx_ref = &cx;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| render_x86_64("%h0", &cx)));
        assert!(result.is_err(), "%h on r11 has no high-byte form");
    }
}
