//! Native lowering of inline-asm blocks onto the Cranelift
//! `inline_asm` instruction (the wasmtime fork's x86-64
//! implementation; see `cranelift/docs/inline-asm.md` in the fork).
//!
//! The fork's surface is narrower than GCC's, so [`plan`] validates a
//! block against it and either returns the translation for
//! `InlineAsmContents` or the reason the block must fall back to the
//! out-of-line wrapper in [`crate::asm`]. Constraints the fork does not
//! take yet (`"m"`, `"i"`, matching constraints, non-fixed inouts)
//! stay plain constraint strings here: when the fork learns them, the
//! corresponding `Err` arm below is all that changes.
//!
//! Two gaps the template side cannot paper over:
//!
//! - the fork always prints the full 64-bit register for `%N`, so a
//!   narrower operand needs a fixed-register constraint (`a`/`b`/`c`/
//!   `d`/`S`/`D`) whose mode-sized name can be substituted statically;
//! - template modifiers (`%b0`, `%w0`, ...) only work on fixed
//!   operands for the same reason.

use glcc_mir as mir;

use crate::abi::Arch;
use crate::asm::{find_reg, mode_name, table, try_parse_constraint, ty_size, Kind};

/// An asm block translated for the fork's `inline_asm` instruction.
#[derive(Debug)]
pub struct NativeAsm {
    /// Template after static substitution of fixed narrow operands.
    pub template: String,
    /// One fork-grammar constraint per operand, GCC order.
    pub constraints: Vec<String>,
    /// Clobbers: GPRs canonicalized to 64-bit names, `cc`/`memory`
    /// passed through; other names need no action, as in the wrapper.
    pub clobbers: Vec<String>,
    /// Per-operand kind, GCC order: how the emitter passes each value.
    pub operand_kinds: Vec<NativeKind>,
}

/// One operand's shape, as the native path will use it.
struct NativeOperand {
    width: u64,
    /// Canonical 64-bit name of the pinned register, if any.
    fixed: Option<&'static str>,
    kind: NativeKind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NativeKind {
    Reg,
    Mem,
    Imm,
}

/// Translate `block` for the native x86-64 path, or return why it must
/// use the out-of-line wrapper.
pub fn plan(block: &mir::AsmBlock, arch: Arch, func: &mir::Function) -> Result<NativeAsm, String> {
    debug_assert_eq!(arch, Arch::X86_64);
    let t = table(arch);
    if block.is_basic {
        return Err("basic asm templates are not native yet".into());
    }
    if !block.labels.is_empty() {
        return Err("asm goto needs template label support".into());
    }

    let mut clobbers = Vec::new();
    for clobber in &block.clobbers {
        let name = clobber.strip_prefix('%').unwrap_or(clobber);
        match name {
            "cc" | "memory" => {
                if !clobbers.iter().any(|c| c == name) {
                    clobbers.push(name.to_string());
                }
            }
            _ => {
                let Some(info) = find_reg(t, name) else { continue };
                if info.q == "rsp" || info.q == "rbp" {
                    return Err(format!("clobbering {} is not native yet", info.q));
                }
                if !clobbers.iter().any(|c| c == info.q) {
                    clobbers.push(info.q.to_string());
                }
            }
        }
    }

    let outputs = block.outputs as usize;
    let mut constraints = Vec::with_capacity(block.operands.len());
    let mut operands = Vec::with_capacity(block.operands.len());
    for (i, op) in block.operands.iter().enumerate() {
        let parsed = try_parse_constraint(t, &op.constraint)?;
        if parsed.tie.is_some() {
            return Err("matching constraints (\"0\") are not native yet".into());
        }
        match parsed.kind {
            Kind::Imm => {
                // The fork substitutes immediates from compile-time
                // constants only; a non-constant `"Ir"`-style alternative
                // stays on the wrapper, which can pick the register form.
                if !crate::asm::is_int_const(func, op.value) {
                    return Err(
                        "immediate operand value is not a compile-time constant".into(),
                    );
                }
            }
            Kind::Mem | Kind::Reg => {}
        }
        if parsed.high {
            return Err("the \"Q\" high-byte class is not native yet".into());
        }
        if i < outputs && parsed.kind != Kind::Reg {
            return Err("only register outputs are native yet".into());
        }
        if i < outputs && parsed.inout && parsed.named.is_none() {
            return Err("non-fixed inouts (\"+r\") are not native yet".into());
        }
        if op.ty.is_aggregate() && parsed.kind != Kind::Mem {
            return Err("aggregate operands are not native yet".into());
        }
        let width = ty_size(&op.ty);
        if width > 8 && parsed.kind == Kind::Reg {
            return Err("register operands wider than a register are not native yet".into());
        }
        constraints.push(fork_constraint(&op.constraint, &parsed)?);
        operands.push(NativeOperand {
            width,
            fixed: parsed.named,
            kind: match parsed.kind {
                Kind::Reg => NativeKind::Reg,
                Kind::Mem => NativeKind::Mem,
                Kind::Imm => NativeKind::Imm,
            },
        });
    }

    for clobber in &clobbers {
        if operands.iter().any(|op| op.fixed == Some(clobber.as_str())) {
            return Err(format!("asm constraint names the clobbered register {clobber}"));
        }
    }

    let template = translate_template(&block.template, &operands)?;
    Ok(NativeAsm {
        template,
        constraints,
        clobbers,
        operand_kinds: operands.iter().map(|op| op.kind).collect(),
    })
}

/// The fork-grammar form of a glcc constraint: the `=`/`+`/`&` prefixes
/// plus the register letter mapped onto the fork's set.
fn fork_constraint(constraint: &str, parsed: &crate::asm::Constraint) -> Result<String, String> {
    let mut rest = constraint;
    let mut prefixes = String::new();
    while let Some(c) = rest.chars().next() {
        if c == '=' || c == '+' || c == '&' {
            prefixes.push(c);
            rest = &rest[c.len_utf8()..];
        } else {
            break;
        }
    }
    let mut mapped = String::new();
    for letter in rest.chars() {
        let fork = match letter {
            'r' | 'q' | 'g' | 'X' => 'r',
            'a' | 'A' => 'a',
            'b' => 'b',
            'c' => 'c',
            'd' => 'd',
            'S' => 'S',
            'D' => 'D',
            'i' | 'n' => 'i',
            'm' | 'o' | 'V' => 'm',
            _ => {
                return Err(format!(
                    "the constraint letter '{letter}' is not native yet"
                ))
            }
        };
        mapped.push(fork);
    }
    debug_assert!(!mapped.is_empty() || parsed.tie.is_some());
    Ok(format!("{prefixes}{mapped}"))
}

/// Rewrite the template for the fork's expander: `%N` passes through
/// only for 8-byte operands (the fork prints the full 64-bit register);
/// narrower operands and every modifier need a fixed register, whose
/// mode-sized name is substituted statically.
fn translate_template(template: &str, operands: &[NativeOperand]) -> Result<String, String> {
    let mut out = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.peek() {
            Some('%') => {
                chars.next();
                out.push_str("%%");
            }
            Some('=') => return Err("'%=' unique labels are not native yet".into()),
            Some('[') => return Err("named operand references (%[name]) are not native yet".into()),
            Some('l') => return Err("template labels (%l[name]) are not native yet".into()),
            Some(c) if c.is_ascii_alphabetic() => {
                let modifier = *c;
                chars.next();
                let index = take_operand_index(&mut chars)?;
                let op = operand_at(operands, index)?;
                let fixed = op.fixed.ok_or_else(|| {
                    "the template modifier needs a fixed-register operand (a/b/c/d/S/D) on the native path".to_string()
                })?;
                let info = find_reg(table(Arch::X86_64), fixed).expect("fixed register is known");
                let name = match modifier {
                    'b' => info.b,
                    'w' => info.w,
                    'k' => info.k,
                    'q' => info.q,
                    'h' => info.h,
                    _ => return Err(format!("the template modifier %{modifier} is not native yet")),
                };
                if name.is_empty() {
                    return Err(format!("register {fixed} has no such sub-register"));
                }
                out.push_str(&format!("%{name}"));
            }
            Some(_) => {
                let index = take_operand_index(&mut chars)?;
                let op = operand_at(operands, index)?;
                if op.width == 8 || op.kind != NativeKind::Reg {
                    // Memory operands render as `(%reg)` from the 64-bit
                    // address register; immediates render from the
                    // constant; neither cares about the operand's width.
                    out.push_str(&format!("%{index}"));
                } else if let Some(fixed) = op.fixed {
                    let info = find_reg(table(Arch::X86_64), fixed).expect("fixed register is known");
                    out.push_str(&format!("%{}", mode_name(info, op.width, Arch::X86_64)));
                } else {
                    return Err(
                        "register operands narrower than 8 bytes need a fixed register (a/b/c/d/S/D) on the native path"
                            .into(),
                    );
                }
            }
            None => return Err("bad `%` in asm template".into()),
        }
    }
    Ok(out)
}

fn operand_at(operands: &[NativeOperand], index: usize) -> Result<&NativeOperand, String> {
    operands.get(index).ok_or_else(|| {
        format!("asm template references operand {index} which does not exist")
    })
}

fn take_operand_index(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Result<usize, String> {
    let mut digits = String::new();
    while let Some(c) = chars.peek() {
        if c.is_ascii_digit() {
            digits.push(*c);
            chars.next();
        } else {
            break;
        }
    }
    digits.parse::<usize>().map_err(|_| "bad `%` in asm template".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mir::Const;

    struct Block {
        template: String,
        outputs: u32,
        constraints: Vec<String>,
        clobbers: Vec<String>,
        labels: Vec<String>,
        is_basic: bool,
    }

    impl Block {
        fn new(template: &str, outputs: u32, constraints: &[&str]) -> Self {
            Self {
                template: template.to_string(),
                outputs,
                constraints: constraints.iter().map(|c| c.to_string()).collect(),
                clobbers: Vec::new(),
                labels: Vec::new(),
                is_basic: false,
            }
        }

        fn clobbers(mut self, clobbers: &[&str]) -> Self {
            self.clobbers = clobbers.iter().map(|c| c.to_string()).collect();
            self
        }

        fn mir(&self, tys: &[mir::Ty]) -> (mir::Function, mir::AsmBlock) {
            let (mut func, _) = mir::Function::new("f", mir::Signature::new(Vec::new(), mir::Ty::Void));
            let block = mir::AsmBlock {
                template: self.template.clone(),
                operands: self
                    .constraints
                    .iter()
                    .zip(tys.iter().cloned().chain(std::iter::repeat(mir::Ty::int(64))))
                    .map(|(constraint, ty)| mir::AsmOperand {
                        value: func.push_const(Const::Int { bits: 64, value: 0 }),
                        constraint: constraint.to_string(),
                        ty,
                        tie: None,
                        name: None,
                    })
                    .collect(),
                outputs: self.outputs,
                clobbers: self.clobbers.clone(),
                labels: self.labels.clone(),
                is_volatile: false,
                is_basic: self.is_basic,
            };
            (func, block)
        }

        fn plan(&self, tys: &[mir::Ty]) -> Result<NativeAsm, String> {
            let (func, block) = self.mir(tys);
            plan(&block, Arch::X86_64, &func)
        }
    }

    const I64: mir::Ty = mir::Ty::Int { bits: 64 };
    const I32: mir::Ty = mir::Ty::Int { bits: 32 };

    #[test]
    fn fixed_output_constraints_pass_through() {
        // rdtsc: fixed-register outputs the template never names.
        let native = Block::new("rdtsc", 2, &["=a", "=d"]).plan(&[I32, I32]).expect("native");
        assert_eq!(native.template, "rdtsc");
        assert_eq!(native.constraints, ["=a", "=d"]);
    }

    #[test]
    fn narrow_fixed_operands_substitute_mode_sized_names() {
        let native = Block::new("addl $1, %0", 1, &["=a"]).plan(&[I32]).expect("native");
        assert_eq!(native.template, "addl $1, %eax");
        let native = Block::new("movb %b0, %h0", 1, &["=Q"]).plan(&[I32]);
        // "Q" needs a high-byte register the fork cannot pick.
        assert!(native.unwrap_err().contains("\"Q\""));
        let native = Block::new("inb %w1, %b0", 1, &["=a", "d"]).plan(&[I32, I32]).expect("native");
        assert_eq!(native.template, "inb %dx, %al");
    }

    #[test]
    fn wide_register_operands_pass_through() {
        let native = Block::new("movq %1, %0\n\tnotq %0", 1, &["=r", "r"]).plan(&[I64, I64]).expect("native");
        assert_eq!(native.template, "movq %1, %0\n\tnotq %0");
        assert_eq!(native.constraints, ["=r", "r"]);
    }

    #[test]
    fn clobbers_are_canonicalized() {
        let native = Block::new("nop", 0, &[]).clobbers(&["memory", "cc", "%rbx"]).plan(&[]).expect("native");
        assert_eq!(native.clobbers, ["memory", "cc", "rbx"]);
        // Non-GPR names need no action, as in the wrapper (glcc's table
        // matches canonical 64-bit names only).
        let native = Block::new("nop", 0, &[]).clobbers(&["xmm0", "eax"]).plan(&[]).expect("native");
        assert!(native.clobbers.is_empty());
    }

    #[test]
    fn unsupported_shapes_fall_back() {
        assert!(Block::new("nop", 0, &[]).plan(&[]).is_ok());
        // Non-fixed inout.
        assert!(Block::new("addl %1, %0", 1, &["+r", "r"]).plan(&[I32, I32]).unwrap_err().contains("non-fixed inouts"));
        // Memory and immediate operands are native now; a non-constant
        // `"i"` value (with no register alternative to demote to) still
        // falls back. The test helper always pushes constants, so this
        // only checks the shape; the fallback triggers at plan time when
        // is_int_const fails.
        // Memory inputs are native; memory outputs still use the wrapper
        // (the fork's defs are registers only).
        let native = Block::new("movq %1, %0", 1, &["=r", "m"]).plan(&[I64, I64]).expect("m input is native");
        assert_eq!(native.constraints, ["=r", "m"]);
        assert!(Block::new("movq %1, %0", 1, &["=m", "r"]).plan(&[I64, I64]).unwrap_err().contains("only register outputs"));
        let native = Block::new("addq %1, %0", 1, &["=r", "i"]).plan(&[I64, I64]).expect("i is native");
        assert_eq!(native.constraints, ["=r", "i"]);
        // Matching constraint.
        assert!(Block::new("addl %1, %0", 1, &["=r", "0"]).plan(&[I64, I64]).unwrap_err().contains("matching"));
        // Narrow register operand without a fixed register.
        assert!(Block::new("addl %1, %0", 1, &["=r", "r"]).plan(&[I32, I32]).unwrap_err().contains("narrower than 8 bytes"));
        // asm goto and basic asm.
        let mut goto = Block::new("jnz %l[lab]", 0, &[]);
        goto.labels = vec!["lab".to_string()];
        assert!(goto.plan(&[]).unwrap_err().contains("asm goto"));
        let mut basic = Block::new("nop", 0, &[]);
        basic.is_basic = true;
        assert!(basic.plan(&[]).unwrap_err().contains("basic asm"));
        // Template features the fork cannot expand.
        assert!(Block::new("%=1:", 0, &[]).plan(&[]).unwrap_err().contains("%="));
        assert!(Block::new("movl %1, %[out]", 1, &["=r", "r"]).plan(&[I64, I64]).unwrap_err().contains("%[name]"));
        // A fixed register also named as a clobber.
        assert!(Block::new("nop", 1, &["=a"]).clobbers(&["rax"]).plan(&[I32]).unwrap_err().contains("clobbered register"));
    }
}
