//! GAS/AT&T frontend tests: semantic link-and-run checks against the
//! system assembler's behavior, plus the asm glcc's own driver emits.

use object::read::{Object, ObjectSymbol};
use std::process::Command;

fn assemble(src: &str, obj: &std::path::Path) -> Result<(), String> {
    let bytes = glasm::gas::assemble_to_elf(src, glasm::Arch::X86_64).map_err(|errs| errs.iter().map(|e| format!("line {}: {}", e.line, e.msg)).collect::<Vec<_>>().join("\n"))?;
    std::fs::write(obj, bytes).map_err(|e| e.to_string())
}

/// Runs `_start`-style source; returns the process exit code. The
/// programs exit through Linux syscalls and link as ELF, so this is
/// only meaningful on a Linux x86_64 host (another arch's `ld` cannot
/// link the objects, another OS has no `syscall` ABI at all).
#[cfg(target_os = "linux")]
fn run_exit(src: &str, name: &str) -> Option<i32> {
    if Command::new("ld").arg("--version").output().is_err() {
        return None;
    }
    let dir = std::env::temp_dir();
    let obj = dir.join(format!("{name}.o"));
    let exe = dir.join(name);
    assemble(src, &obj).expect("assemble");
    let linked = Command::new("ld").arg(&obj).arg("-o").arg(&exe).output().unwrap();
    assert!(linked.status.success(), "ld failed: {}", String::from_utf8_lossy(&linked.stderr));
    let code = Command::new(&exe).status().unwrap().code();
    std::fs::remove_file(&obj).ok();
    std::fs::remove_file(&exe).ok();
    code
}

// Links and runs a Linux ELF program; other hosts cannot.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn naked_function_asm() {
    // Exactly what glcc's driver writes for a naked function.
    let code = run_exit(
        r"
        .globl _start
_start:
        mov $60, %rax
        mov $7, %rdi
        syscall
        .size _start, .-_start
    ",
        "glasm-gas-naked",
    );
    assert_eq!(code, Some(7));
}

// Links and runs a Linux ELF program; other hosts cannot.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn rip_relative_data_and_comms() {
    // Buffer in .comm, byte store + load, arithmetic in .data.
    let code = run_exit(
        r"
        .text
        .globl _start
_start:
        movb $65, buffer(%rip)
        movzbl buffer(%rip), %edi
        movq sum(%rip), %rax
        addq %rdi, %rax
        movq %rax, sum(%rip)
        movq $60, %rax
        movq sum(%rip), %rdi
        syscall
        .comm buffer, 64, 8
        .data
sum:    .quad 100
    ",
        "glasm-gas-comm",
    );
    assert_eq!(code, Some(165));
}

// Links and runs a Linux ELF program; other hosts cannot.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn numeric_labels() {
    let code = run_exit(
        r"
        .globl _start
_start:
        movq $0, %rax
        jmp 1f
        movq $9, %rax
1:      movq $60, %rax
        movq $8, %rdi
        syscall
    ",
        "glasm-gas-numlab",
    );
    // The skipped movq must not run: exit 8, not 9.
    assert_eq!(code, Some(8));
}

// Links and runs a Linux ELF program; other hosts cannot.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn suffixes_and_special_mnemonics() {
    let code = run_exit(
        r"
        .globl _start
_start:
        movl $1, %eax
        addq $6, %rax
        movslq %eax, %rdx
        shlq $3, %rax
        cltq
        orq %rdx, %rax
        movq %rax, %rdi
        movq $60, %rax
        syscall
    ",
        "glasm-gas-suffix",
    );
    // (1+6)<<3 | 7 = 63
    assert_eq!(code, Some(63));
}

#[test]
fn glcc_generated_text() {
    // File-scope asm exactly as the driver's assemble_asm_text sees it.
    let stmts = glasm::gas::parse_module(
        r#"
        .globl marker
        .type marker, @function
marker: .quad 1234
        .size marker, .-marker
        .section .note.GNU-stack,"",@progbits
        .pushsection .text.extra,"ax",@progbits
extra:  ret
        .popsection
    "#,
        glasm::Arch::X86_64,
    )
    .expect("parse");
    assert!(stmts.iter().any(|s| matches!(s, glasm::Stmt::Label(n, _) if n == "marker")));
    // .pushsection must place subsequent code in its section.
    assert!(stmts.iter().any(|s| matches!(
        s,
        glasm::Stmt::PushSection(spec) if spec.name == ".text.extra"
    )));
}

#[test]
fn weak_symbols_reach_the_symbol_table() {
    // Regression: `.weak` was silently dropped, so glcc's out-of-line
    // inline-asm wrappers came out undefined and failed to link.
    let src = r#"
        .text
        .weak __glcc_asm_add
        .globl caller
__glcc_asm_add:
        ret
caller:
        call __glcc_asm_add
        ret
    "#;
    let bytes = glasm::gas::assemble_to_elf(src, glasm::Arch::X86_64).expect("assemble");
    let obj = object::File::parse(&*bytes).expect("parse object");
    let mut saw_weak = false;
    for sym in obj.symbols() {
        if sym.name().unwrap_or("") == "__glcc_asm_add" {
            assert!(sym.is_weak(), "wrapper must be weak");
            assert!(sym.is_definition(), "wrapper must be defined");
            saw_weak = true;
        }
    }
    assert!(saw_weak, "wrapper missing from symbol table");
}

#[test]
fn multi_digit_numeric_labels() {
    // GNU as allows numeric local labels of any digit count.
    let bytes = glasm::gas::assemble_to_elf("jmp 42f\n nop\n42: nop", glasm::Arch::X86_64).unwrap();
    let obj = object::File::parse(&*bytes).unwrap();
    use object::read::{Object, ObjectSection};
    let text = obj.sections().find(|s| s.name().unwrap() == ".text").unwrap();
    assert_eq!(text.data().unwrap(), &[0xeb, 0x01, 0x90, 0x90][..]);
}

// Links and runs a Linux ELF program; other hosts cannot.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn indirect_register_call_and_jump() {
    // `call *%reg` / `jmp *%reg` encode as FF /2 and FF /4 (with REX.B
    // for r8-r15); the register forms must not fall back to a rel32
    // branch to the next instruction.
    let code = run_exit(
        r"
        .globl _start
_start:
        leaq target(%rip), %rax
        movq $5, %rdi
        call *%rax
        movq %rax, %rbx
        leaq skip(%rip), %rax
        jmp *%rax
        movq $99, %rbx
skip:
        movq $6, %rdi
        call *fp(%rip)
        addq %rax, %rbx
        movq $7, %rdi
        leaq target(%rip), %r13
        call *%r13
        addq %rax, %rbx
        movq %rbx, %rdi
        movq $60, %rax
        syscall
target:
        leaq (%rdi,%rdi,2), %rax
        incq %rax
        ret
        .data
fp:     .quad target
    ",
        "glasm-gas-indirect",
    );
    // 5*3+1 + 6*3+1 + 7*3+1 = 16+19+22; the skipped movq must not fire.
    assert_eq!(code, Some(57));
}

#[test]
fn pushsection_popsection_restores_section() {
    // The kernel emits exception-table entries this way inside function
    // bodies; the two nops must end up back in .text.
    let bytes = glasm::gas::assemble_to_elf(
        r#"
        .text
        nop
        .pushsection .foo,"a"
        .long 1
        .popsection
        nop
    "#,
        glasm::Arch::X86_64,
    )
    .expect("assemble");
    let obj = object::File::parse(&*bytes).expect("parse object");
    use object::read::{Object, ObjectSection};
    let text = obj.section_by_name(".text").expect(".text");
    assert_eq!(text.data().unwrap(), &[0x90, 0x90][..]);
    let foo = obj.section_by_name(".foo").expect(".foo");
    assert_eq!(foo.data().unwrap(), &[1, 0, 0, 0][..]);
}

#[test]
fn previous_restores_last_section() {
    let bytes = glasm::gas::assemble_to_elf(
        r#"
        .text
        nop
        .section .foo,"a"
        .long 1
        .previous
        nop
    "#,
        glasm::Arch::X86_64,
    )
    .expect("assemble");
    let obj = object::File::parse(&*bytes).expect("parse object");
    use object::read::{Object, ObjectSection};
    let text = obj.section_by_name(".text").expect(".text");
    assert_eq!(text.data().unwrap(), &[0x90, 0x90][..]);
    let foo = obj.section_by_name(".foo").expect(".foo");
    assert_eq!(foo.data().unwrap(), &[1, 0, 0, 0][..]);
}

#[test]
fn nested_pushsections() {
    let bytes = glasm::gas::assemble_to_elf(
        r#"
        .text
        nop
        .pushsection .foo,"a"
        .long 1
        .pushsection .bar,"a"
        .long 2
        .popsection
        .long 3
        .popsection
        nop
    "#,
        glasm::Arch::X86_64,
    )
    .expect("assemble");
    let obj = object::File::parse(&*bytes).expect("parse object");
    use object::read::{Object, ObjectSection};
    let text = obj.section_by_name(".text").expect(".text");
    assert_eq!(text.data().unwrap(), &[0x90, 0x90][..]);
    let foo = obj.section_by_name(".foo").expect(".foo");
    assert_eq!(foo.data().unwrap(), &[1, 0, 0, 0, 3, 0, 0, 0][..]);
    let bar = obj.section_by_name(".bar").expect(".bar");
    assert_eq!(bar.data().unwrap(), &[2, 0, 0, 0][..]);
}

#[test]
fn set_directive() {
    let bytes = glasm::gas::assemble_to_elf(
        r#"
        .text
f:      nop
        .set sz, .-f
        .globl sz
        .set K, 42
        .quad K
        .set K, 43
        .quad K
    "#,
        glasm::Arch::X86_64,
    )
    .expect("assemble");
    let obj = object::File::parse(&*bytes).expect("parse object");
    use object::read::{Object, ObjectSection, ObjectSymbol};
    let sz = obj
        .symbols()
        .find(|s| s.name().unwrap_or("") == "sz")
        .expect("sz symbol");
    assert_eq!(sz.address(), 1);
    assert!(matches!(sz.section(), object::SymbolSection::Absolute), "sz must be absolute");
    let text = obj.section_by_name(".text").expect(".text");
    let mut want = vec![0x90u8];
    want.extend_from_slice(&42i64.to_le_bytes());
    want.extend_from_slice(&43i64.to_le_bytes());
    assert_eq!(text.data().unwrap(), &want[..]);
}

#[test]
fn quoted_section_name() {
    // EXPORT_SYMBOL emits `.section ".export_symbol","a"`.
    let bytes = glasm::gas::assemble_to_elf(
        r#"
        .section ".foo","a"
        .long 1
    "#,
        glasm::Arch::X86_64,
    )
    .expect("assemble");
    let obj = object::File::parse(&*bytes).expect("parse object");
    use object::read::{Object, ObjectSection};
    let foo = obj.section_by_name(".foo").expect(".foo");
    assert_eq!(foo.data().unwrap(), &[1, 0, 0, 0][..]);
}

#[test]
fn quoted_pushsection_name() {
    let bytes = glasm::gas::assemble_to_elf(
        r#"
        .text
        nop
        .pushsection ".foo","a"
        .long 1
        .popsection
        nop
    "#,
        glasm::Arch::X86_64,
    )
    .expect("assemble");
    let obj = object::File::parse(&*bytes).expect("parse object");
    use object::read::{Object, ObjectSection};
    let text = obj.section_by_name(".text").expect(".text");
    assert_eq!(text.data().unwrap(), &[0x90, 0x90][..]);
    let foo = obj.section_by_name(".foo").expect(".foo");
    assert_eq!(foo.data().unwrap(), &[1, 0, 0, 0][..]);
}

#[test]
fn star_multiplies_in_displacement() {
    // `1*8(%rdi)` is a displacement expression: `*` mid-expression is
    // multiplication, so it must encode like `8(%rdi)`.
    let scaled = glasm::gas::assemble_to_elf(".text\n movq %rax,1*8(%rdi)\n", glasm::Arch::X86_64).expect("assemble");
    let plain = glasm::gas::assemble_to_elf(".text\n movq %rax,8(%rdi)\n", glasm::Arch::X86_64).expect("assemble");
    assert_eq!(scaled, plain);
}

#[test]
fn leading_star_is_still_indirect() {
    // The leading indirect position (`call *%rax`) is consumed before
    // the displacement expression runs; it must stay FF /2 (mod=11).
    let bytes = glasm::gas::assemble_to_elf(".text\n call *%rax\n", glasm::Arch::X86_64).expect("assemble");
    let obj = object::File::parse(&*bytes).expect("parse object");
    use object::read::{Object, ObjectSection};
    let text = obj.section_by_name(".text").expect(".text");
    assert_eq!(text.data().unwrap(), &[0xff, 0xd0][..]);
}

#[test]
fn set_assignment_alias() {
    // `symbol = expression` is the .set alias GAS also accepts.
    let bytes = glasm::gas::assemble_to_elf(
        r#"
        .text
f:      nop
sz = . - f
        .globl sz
    "#,
        glasm::Arch::X86_64,
    )
    .expect("assemble");
    let obj = object::File::parse(&*bytes).expect("parse object");
    use object::read::{Object, ObjectSymbol};
    let sz = obj
        .symbols()
        .find(|s| s.name().unwrap_or("") == "sz")
        .expect("sz symbol");
    assert_eq!(sz.address(), 1);
}
