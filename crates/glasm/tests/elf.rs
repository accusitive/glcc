//! Object-file level checks: sections, symbols, relocations, and a full
//! link-and-run through the system linker.

use glasm::parse::parse_module;
use glasm::{assemble, elf};
use object::read::{Object, ObjectSection, ObjectSymbol};

fn build(src: &str) -> Vec<u8> {
    let stmts = parse_module(src).expect("parse");
    let module = assemble::assemble(&stmts).expect("assemble");
    elf::write_elf64(&module).expect("elf")
}

#[test]
fn object_layout() {
    let bytes = build(
        r"
        global start
        extern helper

        section .text
start:  call helper
        mov rax, offset_tbl
        lea rsi, [rel msg]
        ret

        section .data
msg:    db 'x'
offset_tbl: dq msg

        section .bss
buf:    resq 4
    ",
    );
    let obj = object::File::parse(&*bytes).expect("parse object");
    assert_eq!(obj.architecture(), object::Architecture::X86_64);

    let names: Vec<_> = obj
        .sections()
        .map(|s| s.name().ok().unwrap_or_default().to_string())
        .collect();
    assert!(names.iter().any(|n| n == ".text"));
    assert!(names.iter().any(|n| n == ".data"));
    assert!(names.iter().any(|n| n == ".bss"));

    let symbols: Vec<_> = obj
        .symbols()
        .map(|s| (s.name().unwrap_or("").to_string(), s.is_definition(), s.is_global()))
        .collect();
    let start = symbols.iter().find(|(n, ..)| n == "start").expect("start symbol");
    assert!(start.1 && start.2);
    let helper = symbols.iter().find(|(n, ..)| n == "helper").expect("helper symbol");
    assert!(!helper.1 && helper.2);

    let text = obj.section_by_name(".text").expect(".text");
    let relocs: Vec<_> = text.relocations().map(|(_, r)| r).collect();
    // call helper (PC32), mov rax, offset_tbl (64), lea [rel msg] (PC32)
    assert_eq!(relocs.len(), 3);
}

#[test]
fn bss_has_size_but_no_bits() {
    let bytes = build(
        r"
        section .bss
buf:    resq 4
    ",
    );
    let obj = object::File::parse(&*bytes).expect("parse object");
    let bss = obj.section_by_name(".bss").expect(".bss");
    assert_eq!(bss.size(), 32);
    assert_eq!(bss.kind(), object::SectionKind::UninitializedData);
}

/// Assembles, links and runs a program; the exit code is the verdict.
/// Skipped quietly when no system linker is available.
#[test]
fn link_and_run() {
    // glasm emits x86-64 ELF: skip hosts without an ELF-capable native
    // toolchain (Windows ld links PE, non-x86_64 ld links other arches).
    if cfg!(windows) || cfg!(not(target_arch = "x86_64")) {
        return;
    }
    if std::process::Command::new("ld").arg("--version").output().is_err() {
        return;
    }
    let bytes = build(
        r"
        global _start
        section .text
_start: mov rax, 60
        mov rdi, 42
        syscall
    ",
    );
    let dir = std::env::temp_dir();
    let opath = dir.join(format!("glasm-link-{}.o", std::process::id()));
    let epath = dir.join(format!("glasm-link-{}", std::process::id()));
    std::fs::write(&opath, bytes).expect("write object");
    let linked = std::process::Command::new("ld")
        .arg(&opath)
        .arg("-o")
        .arg(&epath)
        .output()
        .expect("run ld");
    assert!(linked.status.success(), "ld failed: {}", String::from_utf8_lossy(&linked.stderr));
    let run = std::process::Command::new(&epath).status().expect("run program");
    std::fs::remove_file(&opath).ok();
    std::fs::remove_file(&epath).ok();
    assert_eq!(run.code(), Some(42));
}
