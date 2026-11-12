//! Shared differential-test helper: assembles each case with both GNU
//! as and glasm; .text bytes and relocation tables must match exactly.
//! Cases are skipped quietly when the reference assembler is missing.

use object::read::{Object, ObjectSection, ObjectSymbol};
use std::process::Command;

pub fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok()
}

struct RefObj {
    text: Vec<u8>,
    relocs: Vec<(u64, u32, i64, String)>,
}

fn read_obj(bytes: &[u8]) -> RefObj {
    let obj = object::File::parse(bytes).expect("parse object");
    let text = obj.section_by_name(".text").expect(".text").data().expect("text data").to_vec();
    let mut relocs = Vec::new();
    if let Some(sec) = obj.section_by_name(".text") {
        for (off, r) in sec.relocations() {
            let target = match r.target() {
                object::RelocationTarget::Symbol(i) => obj.symbol_by_index(i).map(|s| s.name().unwrap_or("").to_string()).unwrap_or_default(),
                object::RelocationTarget::Section(i) => obj.section_by_index(i).map(|s| s.name().unwrap_or("").to_string()).unwrap_or_default(),
                _ => String::new(),
            };
            let kind = match r.flags() {
                object::RelocationFlags::Elf { r_type: t } => t.0,
                _ => 0,
            };
            relocs.push((off, kind, r.addend(), target));
        }
    }
    RefObj { text, relocs }
}

pub fn check_case(arch: glasm::Arch, as_name: &str, march: &str, name: &str, src: &str) {
    let dir = std::env::temp_dir();
    let id = format!("{}-{}", name.replace(['/', ' ', '.'], "_"), std::process::id());
    let s_path = dir.join(format!("{id}.s"));
    std::fs::write(&s_path, src).expect("write source");

    let ref_path = dir.join(format!("{id}-ref.o"));
    let mut cmd = Command::new(as_name);
    if !march.is_empty() {
        cmd.arg(format!("-march={march}"));
    }
    let out = cmd.args(["-o"]).arg(&ref_path).arg(&s_path).output().expect("run reference assembler");
    assert!(out.status.success(), "GNU as failed on {name}: {}", String::from_utf8_lossy(&out.stderr));
    let ref_obj = read_obj(&std::fs::read(&ref_path).unwrap());

    let ours = glasm::gas::assemble_to_elf(src, arch).unwrap_or_else(|errs| panic!("glasm failed on {name}: {errs:?}"));
    let our_obj = read_obj(&ours);

    if ref_obj.text != our_obj.text {
        eprintln!("case: {name}");
        eprintln!("source:\n{src}");
        eprintln!("  ref: {}", hex(&ref_obj.text));
        eprintln!("  ours: {}", hex(&our_obj.text));
        panic!(".text bytes differ for case `{name}'");
    }
    if ref_obj.relocs != our_obj.relocs {
        eprintln!("case: {name}");
        eprintln!("source:\n{src}");
        eprintln!("  ref relocs: {:?}", ref_obj.relocs);
        eprintln!("  our relocs: {:?}", our_obj.relocs);
        panic!("relocations differ for case `{name}'");
    }
    std::fs::remove_file(&s_path).ok();
    std::fs::remove_file(&ref_path).ok();
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Runs every case, collecting failures so all mismatches are reported.
pub fn run(arch: glasm::Arch, as_name: &str, march: &str, cases: &[(&str, &str)]) {
    if !have(as_name) {
        return;
    }
    let mut failures = Vec::new();
    for (name, src) in cases {
        if std::panic::catch_unwind(|| check_case(arch, as_name, march, name, src)).is_err() {
            failures.push(*name);
        }
    }
    if !failures.is_empty() {
        panic!("differential cases failed: {failures:?}");
    }
}
