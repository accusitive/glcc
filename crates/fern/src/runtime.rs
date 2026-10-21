//! The fern runtime, compiled from C.
//!
//! `rt/gc.c` is the collector plus the entry points generated code
//! calls: `fern_enter`, `fern_pin`, `fern_leave`, and `fern_alloc` (the
//! GC ABI declared in `lower::gc`), and the coroutine frame hooks plus
//! blocking driver used by async lowering. It is compiled here
//! with glcc's own frontend so that a driver can lower it into the *same*
//! object file -- or JIT image -- as the program: cranelift-module merges
//! same-name declarations, so the program's externs bind to the runtime's
//! definitions.
//!
//! Programs also use the C library's `write` and `abort`; the JIT driver
//! must supply those symbols, the object driver links libc.

use glcc_frontend::lexer::{Interner, Lexer};
use glcc_frontend::parse::Parser;
use glcc_frontend::sema::Sema;
use glcc_frontend::target::Target;
use glcc_mir as mir;

/// The runtime source: plain C in the subset the frontend accepts.
pub const SOURCE: &str = include_str!("../rt/gc.c");

/// The runtime as an optimized, verified MIR module for the native target.
/// The runtime is compiler-internal, so it always gets the standard
/// pipeline regardless of the program's optimization level.
pub fn module() -> mir::Module {
    let module = std::panic::catch_unwind(std::panic::AssertUnwindSafe(compile)).expect("fern runtime: the C frontend rejected rt/gc.c (see the diagnostic above)");
    let errors = mir::verify::verify_module(&module);
    assert!(errors.is_empty(), "fern runtime: MIR verification failed: {errors:?}");
    module
}

/// The name the runtime source is compiled under: it only appears in
/// frontend diagnostics.
const NAME: &str = "fern-rt/gc.c";

fn compile() -> mir::Module {
    // The frontend takes preprocessed C, exactly like the driver feeds it.
    // The runtime is self-contained (its own macros, no includes), so the
    // default options plus macro expansion suffice; linemarkers are
    // suppressed because nothing maps runtime positions back to source.
    let mut opts = glcc_pp::cpp_flags_to_opts(&[], NAME);
    opts.linemarkers = false;
    opts.target = Some(Target::native());
    // The runtime selects its OS layer with `_WIN32`, but the embedded
    // preprocessor is not seeded with the host's builtin macros here.
    #[cfg(windows)]
    opts.defines.push(("_WIN32".to_string(), Some("1".to_string())));
    let src = glcc_pp::preprocess(SOURCE, NAME, &opts).expect("fern runtime: preprocessing rt/gc.c failed");
    let mut interner = Interner::new();
    let mut sema = Sema::with_target(Target::native());
    let tokens = Lexer::new(&src, &mut interner).lex_all();
    let mut parser = Parser::new(&src, tokens, &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    drop(parser);
    let mut module = glcc_lower_mir::lower_translation_unit(&sema, &interner, &tu);
    glcc_mir::optimize(&mut module, glcc_mir::OptLevel::Speed);
    module
}
