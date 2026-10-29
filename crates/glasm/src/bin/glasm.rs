//! glasm CLI, modeled on nasm's: `glasm [-f elf64] [-o out.o] in.asm`.

use std::process::ExitCode;

fn usage() -> ! {
    eprintln!("usage: glasm [-f elf64] [-o outfile] infile");
    std::process::exit(2);
}

fn main() -> ExitCode {
    let mut format = String::from("elf64");
    let mut output: Option<String> = None;
    let mut input: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-f" => {
                format = args.next().unwrap_or_else(|| usage());
            }
            "-o" => {
                output = Some(args.next().unwrap_or_else(|| usage()));
            }
            "-h" | "--help" => {
                println!("usage: glasm [-f elf64] [-o outfile] infile");
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
    match glasm::assemble_to_elf64(&src) {
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
