//! fernc: the fern driver.
//!
//! Fern's own command driver, deliberately separate from `glcc`'s. It
//! owns argument parsing, the MIR pass pipeline, and object emission;
//! the `fern` crate stops at MIR.

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use glcc_mir::pass::PassManager;
use glcc_mir::passes;

const USAGE: &str = "\
fernc — compile fern to an object file or a MIR dump

usage: fernc [options] <file.fern>

options:
  -o <path>        output path (default: input with its extension replaced by `.o`)
  --emit=<kind>    `object` (default), `mir`, `full-mir`, or `check`
  --emit-mir       print the MIR to stdout (or `-o`) and stop
  --emit-full-mir  like `--emit-mir`, but located instructions get a
                   `; file:line:column` source position suffix
  -O0              no optimization (default)
  -O1, -O2         run the MIR optimization pipeline
  --debug          print the MIR before and after passes
  -h, --help       show this help
";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Emit {
    Object,
    Mir,
    FullMir,
    Check,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum OptLevel {
    #[default]
    None,
    Speed,
}

impl OptLevel {
    fn flag(self) -> &'static str {
        match self {
            OptLevel::None => "none",
            OptLevel::Speed => "speed",
        }
    }

    fn optimize(self) -> bool {
        self == OptLevel::Speed
    }
}

struct Options {
    input: PathBuf,
    output: Option<PathBuf>,
    emit: Emit,
    opt: OptLevel,
    debug: bool,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = match parse_args(&args) {
        Ok(Some(opts)) => opts,
        Ok(None) => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(msg) => {
            eprintln!("fernc: {msg}");
            eprintln!("try `fernc --help`");
            return ExitCode::FAILURE;
        }
    };
    match run(&opts) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("fernc: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn parse_args(args: &[String]) -> Result<Option<Options>, String> {
    let mut input = None;
    let mut output = None;
    let mut emit = Emit::Object;
    let mut opt = OptLevel::None;
    let mut debug = false;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "-o" => {
                i += 1;
                let path = args.get(i).ok_or("`-o` needs a path")?;
                output = Some(PathBuf::from(path));
            }
            "--debug" => debug = true,
            "--emit=mir" => emit = Emit::Mir,
            "--emit=full-mir" => emit = Emit::FullMir,
            "--emit=object" => emit = Emit::Object,
            "--emit=check" => emit = Emit::Check,
            "--emit-mir" => emit = Emit::Mir,
            "--emit-full-mir" => emit = Emit::FullMir,
            "-O0" => opt = OptLevel::None,
            "-O1" | "-O2" | "-O3" | "-O" => opt = OptLevel::Speed,
            other if other.starts_with("--emit=") => {
                return Err(format!("unknown emit kind `{}`", &other["--emit=".len()..]));
            }
            other if other.starts_with('-') && other.len() > 1 => {
                return Err(format!("unknown option `{other}`"));
            }
            other => {
                if input.is_some() {
                    return Err(format!("unexpected extra input `{other}`"));
                }
                input = Some(PathBuf::from(other));
            }
        }
        i += 1;
    }

    let input = input.ok_or("no input file")?;
    Ok(Some(Options {
        input,
        output,
        emit,
        opt,
        debug,
    }))
}

fn run(opts: &Options) -> Result<(), String> {
    let src = fs::read_to_string(&opts.input)
        .map_err(|e| format!("cannot read `{}`: {e}", opts.input.display()))?;
    let mut module = fern::compile(&src).map_err(|diag| {
        format!("{}:{}", opts.input.display(), fern::render_diag(&src, &diag))
    })?;

    if opts.debug {
        eprintln!(
            "--- mir (frontend output) ---\n{}",
            glcc_mir::display::dump_module(&module)
        );
    }
    if opts.opt.optimize() {
        // Same pipeline the C driver runs: inline, promote slots, fold
        // and simplify instructions, simplify the CFG, and sweep dead
        // code to a fixpoint.
        let mut pm = PassManager::new()
            .add_module_pass(passes::Inline::small())
            .add_func_pass(passes::Mem2Reg)
            .add_func_pass(passes::ConstFold)
            .add_func_pass(passes::InstCombine)
            .add_func_pass(passes::ConstMerge)
            .add_func_pass(passes::Cse)
            .add_func_pass(passes::SimplifyCfg)
            .add_func_pass(passes::DeadCodeElim);
        pm.run(&mut module);
    }
    if opts.debug {
        eprintln!(
            "--- mir (after passes) ---\n{}",
            glcc_mir::display::dump_module(&module)
        );
    }

    let errors = glcc_mir::verify::verify_module(&module);
    if !errors.is_empty() {
        for error in &errors {
            eprintln!("fernc: mir error: {error}");
        }
        return Err("mir verification failed".to_string());
    }

    match opts.emit {
        Emit::Check => Ok(()),
        Emit::Mir => write_dump(opts, glcc_mir::display::dump_module(&module)),
        Emit::FullMir => write_dump(opts, glcc_mir::display::dump_module_full(&module)),
        Emit::Object => {
            let bytes = emit_object(&module, opts.opt)?;
            let path = output_path(opts);
            fs::write(&path, &bytes)
                .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
            Ok(())
        }
    }
}

/// Write a MIR dump to `-o`, or to stdout when no output path was given.
fn write_dump(opts: &Options, text: String) -> Result<(), String> {
    match &opts.output {
        Some(path) => fs::write(path, text.as_bytes())
            .map_err(|e| format!("cannot write `{}`: {e}", path.display())),
        None => {
            print!("{text}");
            Ok(())
        }
    }
}

fn output_path(opts: &Options) -> PathBuf {
    if let Some(path) = &opts.output {
        return path.clone();
    }
    let stem = opts.input.file_stem().unwrap_or_else(|| "out".as_ref());
    let mut path = PathBuf::from(stem);
    path.set_extension("o");
    path
}

fn emit_object(module: &glcc_mir::ir::Module, opt: OptLevel) -> Result<Vec<u8>, String> {
    use cranelift_codegen::settings::Configurable;

    let mut flag_builder = cranelift_codegen::settings::builder();
    flag_builder
        .set("opt_level", opt.flag())
        .map_err(|e| e.to_string())?;
    flag_builder.set("is_pic", "true").map_err(|e| e.to_string())?;
    flag_builder
        .set("preserve_frame_pointers", "true")
        .map_err(|e| e.to_string())?;
    let flags = cranelift_codegen::settings::Flags::new(flag_builder);
    let isa = cranelift_native::builder()
        .map_err(|e| e.to_string())?
        .finish(flags)
        .map_err(|e| e.to_string())?;
    let builder = cranelift_object::ObjectBuilder::new(
        isa,
        "fern",
        cranelift_module::default_libcall_names(),
    )
    .map_err(|e| e.to_string())?;
    let mut object = cranelift_object::ObjectModule::new(builder);
    // The runtime comes first: the program's externs for `fern_enter`,
    // `fern_pin`, `fern_leave`, and `fern_alloc` then bind to the
    // runtime's definitions (cranelift-module merges same-name
    // declarations).
    let runtime = fern::runtime::module();
    glcc_lower_cl::lower_module(&runtime, &mut object, glcc_lower_cl::Target::native());
    glcc_lower_cl::lower_module(module, &mut object, glcc_lower_cl::Target::native());
    object
        .finish()
        .emit()
        .map_err(|e| format!("object emission failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Options {
        let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
        parse_args(&args)
            .expect("parses")
            .expect("not a help request")
    }

    #[test]
    fn emit_flags_pick_the_dump_and_the_last_one_wins() {
        assert_eq!(parse(&["a.fern"]).emit, Emit::Object);
        assert_eq!(parse(&["--emit=mir", "a.fern"]).emit, Emit::Mir);
        assert_eq!(parse(&["--emit=full-mir", "a.fern"]).emit, Emit::FullMir);
        assert_eq!(parse(&["--emit-mir", "a.fern"]).emit, Emit::Mir);
        assert_eq!(parse(&["--emit-full-mir", "a.fern"]).emit, Emit::FullMir);
        assert_eq!(parse(&["--emit-full-mir", "--emit-mir", "a.fern"]).emit, Emit::Mir);
        assert_eq!(parse(&["--emit-mir", "--emit-full-mir", "a.fern"]).emit, Emit::FullMir);
        assert_eq!(parse(&["--emit-full-mir", "--emit=object", "a.fern"]).emit, Emit::Object);
    }

    #[test]
    fn unknown_emit_kind_is_an_error() {
        let args = vec!["--emit=nope".to_string(), "a.fern".to_string()];
        let err = match parse_args(&args) {
            Err(err) => err,
            Ok(_) => panic!("unknown emit kind must fail"),
        };
        assert!(err.contains("nope"), "{err}");
    }
}
