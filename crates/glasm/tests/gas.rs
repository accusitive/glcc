//! GAS/AT&T frontend tests: semantic link-and-run checks against the
//! system assembler's behavior, plus the asm glcc's own driver emits.

use object::read::{Object, ObjectSymbol};
use std::process::Command;

fn assemble(src: &str, obj: &std::path::Path) -> Result<(), String> {
    let bytes = glasm::gas::assemble_to_elf64(src).map_err(|errs| {
        errs.iter()
            .map(|e| format!("line {}: {}", e.line, e.msg))
            .collect::<Vec<_>>()
            .join("\n")
    })?;
    std::fs::write(obj, bytes).map_err(|e| e.to_string())
}

/// Runs `_start`-style source; returns the process exit code.
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
    "#,
    )
    .expect("parse");
    assert!(stmts.iter().any(|s| matches!(s, glasm::Stmt::Label(n, _) if n == "marker")));
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
    let bytes = glasm::gas::assemble_to_elf64(src).expect("assemble");
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
