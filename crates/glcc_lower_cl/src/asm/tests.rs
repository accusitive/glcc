use super::*;
use glcc_mir::{Const, Function, Signature, Ty};

fn target(arch: Arch) -> Target {
    Target { arch, os: crate::abi::Os::Linux }
}

/// Build a site over `constraints` (GCC order) with all-i64 operands
/// whose constant values are their operand index, and `outputs`
/// leading outputs.
fn build_site(arch: Arch, template: &str, outputs: u32, constraints: &[(&str, Option<u32>)], clobbers: &[&str]) -> Site {
    build_site_with_labels(arch, template, outputs, constraints, clobbers, &[])
}

/// [`build_site`] with `asm goto` labels, so the wrapper returns a
/// selector.
fn build_goto_site(arch: Arch, template: &str, outputs: u32, constraints: &[(&str, Option<u32>)], labels: &[&str]) -> Site {
    build_site_with_labels(arch, template, outputs, constraints, &[], labels)
}

fn build_site_with_labels(arch: Arch, template: &str, outputs: u32, constraints: &[(&str, Option<u32>)], clobbers: &[&str], labels: &[&str]) -> Site {
    let (mut func, _) = Function::new("f", Signature::new(Vec::new(), Ty::Void));
    let operands = constraints
        .iter()
        .enumerate()
        .map(|(i, (constraint, tie))| {
            let value = func.push_const(Const::Int { bits: 64, value: i as i128 });
            mir::AsmOperand {
                value,
                constraint: constraint.to_string(),
                ty: Ty::int(32),
                tie: *tie,
                name: None,
            }
        })
        .collect();
    let block = mir::AsmBlock {
        template: template.to_string(),
        operands,
        outputs,
        clobbers: clobbers.iter().map(|c| c.to_string()).collect(),
        labels: labels.iter().map(|l| l.to_string()).collect(),
        is_volatile: false,
        is_basic: false,
    };
    build(&block, &func, target(arch))
}

/// [`build_site`] for a basic asm template (no colon): no operands,
/// the template passes through verbatim.
fn build_basic_site(arch: Arch, template: &str) -> Site {
    let (func, _) = Function::new("f", Signature::new(Vec::new(), Ty::Void));
    let block = mir::AsmBlock {
        template: template.to_string(),
        operands: Vec::new(),
        outputs: 0,
        clobbers: Vec::new(),
        labels: Vec::new(),
        is_volatile: false,
        is_basic: true,
    };
    build(&block, &func, target(arch))
}

#[test]
fn x86_goto_wrapper_returns_a_selector() {
    // Operand 0 lands in rax, so the selector takes the next free
    // pool register and moves to the return register after the
    // stores.
    let site = build_goto_site(Arch::X86_64, "movl %1, %0; testl %0, %0; jnz %l[lab]", 1, &[("=r", None), ("r", None)], &["lab"]);
    assert!(site.text.contains("movl %ecx, %eax"), "{}", site.text);
    assert!(site.text.contains("jnz .Lasm"), "{}", site.text);
    assert!(site.text.contains("movq $0, %rdx"), "{}", site.text);
    assert!(site.text.contains("movq $1, %rdx"), "{}", site.text);
    assert!(site.text.contains("movq %rdx, %rax"), "{}", site.text);
    let store = site.text.find("movl %eax, 0(%rbx)").expect("output store");
    let ret = site.text.find("movq %rdx, %rax").expect("selector move");
    assert!(store < ret, "stores precede the selector move:\n{}", site.text);
    // The stub label definition follows the `ret`, reached only by
    // template jumps.
    let ret_pos = site.text.find("\tret\n").expect("ret");
    let stub_pos = site.text.find("_0:\n").expect("stub label");
    assert!(stub_pos > ret_pos, "stubs follow the epilogue:\n{}", site.text);
}

#[test]
fn goto_wrapper_name_covers_labels() {
    let plain = build_site(Arch::X86_64, "nop", 0, &[], &[]);
    let with = build_goto_site(Arch::X86_64, "nop", 0, &[], &["lab"]);
    let swapped = build_goto_site(Arch::X86_64, "nop", 0, &[], &["other"]);
    assert_ne!(plain.name, with.name);
    assert_ne!(with.name, swapped.name);
    assert!(with.text.contains("$0,"), "{}", with.text);
    assert!(with.text.contains("$1,"), "{}", with.text);
    assert!(!plain.text.contains("$1,"), "{}", plain.text);
}

#[test]
fn aarch64_and_riscv_goto_stubs_jump_with_the_local_mnemonic() {
    let arm = build_goto_site(Arch::Aarch64, "nop", 0, &[], &["lab"]);
    assert!(arm.text.contains("b .Lasm"), "{}", arm.text);
    assert!(arm.text.contains("mov x1, #1"), "{}", arm.text);
    assert!(arm.text.contains("mov x0, x1"), "{}", arm.text);
    let rv = build_goto_site(Arch::Riscv64, "nop", 0, &[], &["lab"]);
    assert!(rv.text.contains("j .Lasm"), "{}", rv.text);
    assert!(rv.text.contains("li a1, 1"), "{}", rv.text);
    assert!(rv.text.contains("mv a0, a1"), "{}", rv.text);
}

#[test]
fn x86_immediate_class_constraints() {
    let site = build_site(Arch::X86_64, "shll %1, %0", 1, &[("+r", None), ("I", None)], &[]);
    assert!(site.text.contains("shll $1, %eax"), "{}", site.text);
    // `"A"` pins rax on x86-64 (printed at the operand's own width).
    let site = build_site(Arch::X86_64, "movq %1, %0", 1, &[("=A", None), ("r", None)], &[]);
    assert!(site.text.contains("movq %ecx, %eax"), "{}", site.text);
}

#[test]
fn immediate_operands_carry_the_target_prefix() {
    // `%N` prints GCC's immediate syntax; `%cN` and `%nN` do not.
    let x86 = build_site(Arch::X86_64, "movl %1, %0", 1, &[("=r", None), ("i", None)], &[]);
    assert!(x86.text.contains("movl $1, %eax"), "{}", x86.text);
    let x86 = build_site(Arch::X86_64, "movl $%c1, %0", 1, &[("=r", None), ("i", None)], &[]);
    assert!(x86.text.contains("movl $1, %eax"), "{}", x86.text);
    let x86 = build_site(Arch::X86_64, "addl $%n1, %0", 1, &[("+r", None), ("i", None)], &[]);
    assert!(x86.text.contains("addl $-1, %eax"), "{}", x86.text);
    let arm = build_site(Arch::Aarch64, "mov %w0, %1", 1, &[("=r", None), ("i", None)], &[]);
    assert!(arm.text.contains("mov w1, #1"), "{}", arm.text);
    let rv = build_site(Arch::Riscv64, "li %0, %1", 1, &[("=r", None), ("i", None)], &[]);
    assert!(rv.text.contains("li a1, 1"), "{}", rv.text);
}

#[test]
fn basic_templates_pass_through_verbatim() {
    // No colon: `%eax` is literal, not an operand reference, and
    // `%%` is not needed (GCC's basic/extended split; tcctest.c's
    // `asm_dot_test` templates).
    let site = build_basic_site(Arch::X86_64, "mov %eax, %ecx");
    assert!(site.text.contains("mov %eax, %ecx"), "{}", site.text);
    // A wrapper is a function: its symbol is pinned to `.text`, so
    // a template's own `.text` cannot leave `.size name, .-name`
    // spanning sections.
    let site = build_basic_site(Arch::X86_64, ".text; nop");
    assert!(site.text.starts_with(".text\n"), "{}", site.text);
    assert!(site.text.contains(".size __glcc_asm_"), "{}", site.text);
}

#[test]
fn x86_register_wrapper() {
    let site = build_site(Arch::X86_64, "movl %1, %0", 1, &[("=r", None), ("r", None)], &[]);
    assert!(site.name.starts_with("__glcc_asm_"), "{}", site.name);
    assert!(site.text.contains("movl %ecx, %eax"), "{}", site.text);
    assert!(site.text.contains("movl 8(%rbx), %ecx"), "{}", site.text);
    assert_eq!(site.area_size, 16);
    assert_eq!(site.slots[0].map(|s| s.offset), Some(0));
    assert_eq!(site.slots[1].map(|s| s.offset), Some(8));
}

#[test]
fn percent_equals_expands_to_a_unique_local_label() {
    let label_of = |site: &Site| {
        let jne = site.text.split("jne ").nth(1).expect("jne in the template");
        let label = jne.split_whitespace().next().expect("label after jne");
        assert!(label.ends_with('b'), "{label}");
        assert!(label[..label.len() - 1].chars().all(|c| c.is_ascii_digit()));
        label[..label.len() - 1].to_string()
    };
    let one = build_site(Arch::X86_64, "%=: decq %0; jne %=b", 1, &[("+r", None)], &[]);
    let label = label_of(&one);
    assert!(one.text.contains(&format!("{label}: decq")), "{}", one.text);
    let two = build_site(Arch::X86_64, "%=: decq %0; jne %=b", 1, &[("+r", None), ("r", None)], &[]);
    assert_ne!(label, label_of(&two));
}

#[test]
fn x86_tie_shares_a_slot_and_register() {
    let site = build_site(Arch::X86_64, "addl $1, %0", 1, &[("=r", None), ("0", Some(0))], &[]);
    assert!(site.slots[1].is_none(), "tied input has no slot of its own");
    assert!(site.text.contains("addl $1, %eax"), "{}", site.text);
    assert!(site.text.contains("movl 0(%rbx), %eax"), "{}", site.text);
    assert!(site.text.contains("movl %eax, 0(%rbx)"), "{}", site.text);
}

#[test]
fn x86_memory_operand_uses_its_own_register() {
    let site = build_site(Arch::X86_64, "addl $1, %0", 1, &[("+m", None)], &[]);
    assert!(site.slots[0].is_some_and(|slot| slot.addr));
    assert!(site.text.contains("movq 0(%rbx), %rax"), "{}", site.text);
    assert!(site.text.contains("addl $1, (%rax)"), "{}", site.text);
}

#[test]
fn memory_operands_use_the_target_syntax() {
    // RISC-V's assembler spells a memory operand `offset(reg)`,
    // AArch64's brackets it.
    let site = build_site(Arch::Riscv64, "lw %0, %1", 1, &[("=r", None), ("m", None)], &[]);
    assert!(site.text.contains("lw a1, (a2)"), "{}", site.text);
    let site = build_site(Arch::Aarch64, "ldr %w0, %1", 1, &[("=r", None), ("m", None)], &[]);
    assert!(site.text.contains("ldr w1, [x2]"), "{}", site.text);
}

#[test]
fn aarch64_memory_class_constraints() {
    // `"Q"` is a base-register memory operand on AArch64 (a
    // high-byte register on x86-64), and the `U..` qualifiers
    // (`"Ump"`: a load/store-pair address) are memory too.
    let site = build_site(Arch::Aarch64, "ldr %w0, %1", 1, &[("=r", None), ("Q", None)], &[]);
    assert!(site.slots[1].is_some_and(|slot| slot.addr));
    assert!(site.text.contains("ldr w1, [x2]"), "{}", site.text);
    let site = build_site(Arch::Aarch64, "stp %0, %1, %2", 0, &[("r", None), ("r", None), ("Ump", None)], &[]);
    assert!(site.text.contains("stp w1, w2, [x3]"), "{}", site.text);
}

#[test]
fn riscv_address_constraint_is_a_memory_operand() {
    // `"A"`: an address held in a general-purpose register.
    let site = build_site(Arch::Riscv64, "amoadd.w %0, %2, %1", 1, &[("=r", None), ("+A", None), ("r", None)], &[]);
    assert!(site.slots[1].is_some_and(|slot| slot.addr));
    assert!(site.text.contains("amoadd.w a1, a3, (a2)"), "{}", site.text);
}

#[test]
#[should_panic(expected = "asm constraint 'S' selects a symbolic address")]
fn aarch64_symbolic_address_constraint_is_rejected() {
    build_site(Arch::Aarch64, "adrp %0, %1", 1, &[("=r", None), ("S", None)], &[]);
}

#[test]
#[should_panic(expected = "unsupported asm constraint 'Q'")]
fn riscv_high_byte_constraint_is_rejected() {
    build_site(Arch::Riscv64, "lw %0, %1", 1, &[("=r", None), ("Q", None)], &[]);
}

#[test]
fn x86_immediate_needs_no_slot() {
    let site = build_site(Arch::X86_64, "movl $%c1, %0", 1, &[("=r", None), ("i", None)], &[]);
    assert!(site.slots[1].is_none());
    assert!(site.text.contains("movl $1, %eax"), "{}", site.text);
}

#[test]
fn immediate_or_register_alternatives() {
    // `"Ir"`: the immediate alternative applies only to a constant
    // value; anything else rides the register.
    let site = build_site(Arch::Aarch64, "add %0, %0, %1", 1, &[("+r", None), ("Ir", None)], &[]);
    assert!(site.slots[1].is_none(), "{}", site.text);
    assert!(site.text.contains("add w1, w1, #1"), "{}", site.text);
    let (mut func, _) = Function::new("f", Signature::new(Vec::new(), Ty::Void));
    let block = func.push_block(vec![Ty::int(32)]);
    let param = func.blocks[block].params[0];
    let out = func.push_const(Const::Int { bits: 64, value: 0 });
    let operands = vec![
        mir::AsmOperand {
            value: out,
            constraint: "+r".to_string(),
            ty: Ty::int(32),
            tie: None,
            name: None,
        },
        mir::AsmOperand {
            value: param,
            constraint: "Ir".to_string(),
            ty: Ty::int(32),
            tie: None,
            name: None,
        },
    ];
    let block = mir::AsmBlock {
        template: "add %0, %0, %1".to_string(),
        operands,
        outputs: 1,
        clobbers: Vec::new(),
        labels: Vec::new(),
        is_volatile: false,
        is_basic: false,
    };
    let site = build(&block, &func, target(Arch::Aarch64));
    assert!(site.slots[1].is_some(), "{}", site.text);
    assert!(site.text.contains("add w1, w1, w2"), "{}", site.text);
}

#[test]
fn x86_reserved_register_avoids_named_and_clobbered() {
    // rbx is written by the template: the area register moves to r12.
    let site = build_site(Arch::X86_64, "movq %%rbx, %%rax", 1, &[("=r", None)], &[]);
    assert!(site.text.contains("push %r12"), "{}", site.text);
    assert!(site.text.contains("movq %rdi, %r12"), "{}", site.text);
    assert!(site.text.contains("movq %rbx, %rax"), "{}", site.text);
    // rbx pinned by a named constraint and r12 clobbered: r13 carries.
    let site = build_site(Arch::X86_64, "movq %%rbx, %%rax", 1, &[("=b", None)], &["r12"]);
    assert!(site.text.contains("push %r13"), "{}", site.text);
    assert!(site.text.contains("movq %rdi, %r13"), "{}", site.text);
    assert!(site.text.contains("push %r12"), "{}", site.text);
    assert!(site.text.contains("pop %r12"), "{}", site.text);
}

#[test]
fn x86_mode_sized_register_names() {
    let (mut func, _) = Function::new("f", Signature::new(Vec::new(), Ty::Void));
    let value = func.push_const(Const::Int { bits: 32, value: 7 });
    let operand = |ty: &Ty| mir::AsmOperand {
        value,
        constraint: "=r".to_string(),
        ty: ty.clone(),
        tie: None,
        name: None,
    };
    for (ty, name) in [(Ty::Bool, "%al"), (Ty::int(16), "%ax"), (Ty::int(32), "%eax"), (Ty::int(64), "%rax")] {
        let block = mir::AsmBlock {
            template: "movl %0, %0".to_string(),
            operands: vec![operand(&ty)],
            outputs: 1,
            clobbers: Vec::new(),
            labels: Vec::new(),
            is_volatile: false,
            is_basic: false,
        };
        let site = build(&block, &func, target(Arch::X86_64));
        assert!(site.text.contains(&format!("movl {name}, {name}")), "{ty:?}: {}", site.text);
    }
}

#[test]
fn x86_subregister_modifiers() {
    let site = build_site(Arch::X86_64, "movb %b0, %h0", 1, &[("=Q", None)], &[]);
    assert!(site.text.contains("movb %al, %ah"), "{}", site.text);
}

#[test]
fn aarch64_wrapper_shape() {
    let site = build_site(Arch::Aarch64, "add %w0, %w0, #1", 1, &[("+r", None)], &[]);
    assert!(site.text.contains("mov x19, x0"), "{}", site.text);
    assert!(site.text.contains("ldr w1, [x19, #0]"), "{}", site.text);
    assert!(site.text.contains("str w1, [x19, #0]"), "{}", site.text);
    assert!(site.text.contains("str x19, [sp, #-16]!"), "{}", site.text);
}

#[test]
fn riscv64_wrapper_shape() {
    let site = build_site(Arch::Riscv64, "addi %0, %0, 1", 1, &[("+r", None)], &[]);
    assert!(site.text.contains("mv s1, a0"), "{}", site.text);
    assert!(site.text.contains("lw a1, 0(s1)"), "{}", site.text);
    assert!(site.text.contains("sw a1, 0(s1)"), "{}", site.text);
}

#[test]
fn identical_blocks_share_a_symbol() {
    let a = build_site(Arch::X86_64, "nop", 0, &[], &[]);
    let b = build_site(Arch::X86_64, "nop", 0, &[], &[]);
    assert_eq!(a.name, b.name);
    let c = build_site(Arch::X86_64, "nop", 0, &[], &["cc"]);
    assert_ne!(a.name, c.name);
}
