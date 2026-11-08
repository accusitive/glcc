//! glasm CLI, modeled on nasm's: `glasm [-f elf64] [-o out.o] in.asm`.

use std::process::ExitCode;

fn usage() -> ! {
    eprintln!("usage: glasm [-f elf64] [-o outfile] infile");
    std::process::exit(2);
}

fn main() -> ExitCode {
    let mut format = String::from("elf64");
    let mut arch = glasm::Arch::X86_64;
    let mut output: Option<String> = None;
    let mut input: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-f" => {
                format = args.next().unwrap_or_else(|| usage());
            }
            "-a" | "--arch" => {
                let name = args.next().unwrap_or_else(|| usage());
                match glasm::Arch::from_str(&name) {
                    Some(a) => arch = a,
                    None => {
                        eprintln!("glasm: unknown architecture `{name}'");
                        usage();
                    }
                }
            }
            "-o" => {
                output = Some(args.next().unwrap_or_else(|| usage()));
            }
            "-h" | "--help" => {
                println!("usage: glasm [-f elf64] [-a arch] [-o outfile] infile");
                println!("arch: x86_64 (default) | riscv32 | riscv64 | aarch64");
                return ExitCode::SUCCESS;
            }
            a if a.starts_with('-') => {
                eprintln!("glasm: unknown option `{a}'");
                usage();
            }
            a => input = Some(a.to_string()),
        }
    }
    if format != "elf64" {
        eprintln!("glasm: unsupported output format `{format}'");
        return ExitCode::FAILURE;
    }
    let Some(input) = input else { usage() };
    let output = output.unwrap_or_else(|| {
        let p = std::path::Path::new(&input);
        p.with_extension("o").to_string_lossy().into_owned()
    });

    let src = match std::fs::read_to_string(&input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("glasm: cannot read `{input}': {e}");
            return ExitCode::FAILURE;
        }
    };
    // Intel syntax is x86-only; the GAS frontend covers every arch.
    let result = if arch == glasm::Arch::X86_64 {
        glasm::assemble_to_elf64(&src)
    } else {
        glasm::gas::assemble_to_elf(&src, arch)
    };
    match result {
        Ok(bytes) => {
            if let Err(e) = std::fs::write(&output, bytes) {
                eprintln!("glasm: cannot write `{output}': {e}");
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Err(errors) => {
            for e in &errors {
                eprintln!("{input}:{e}");
            }
            ExitCode::FAILURE
        }
    }
}
