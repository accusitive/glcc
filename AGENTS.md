# AGENTS.md — working conventions for glcc
## Comments
- Do not leave "chain of thought" comments, do not leave superfluous comments on code that is clear. when possible, rewrite code to be self documenting via clearer control flow, variable names, etc instead of leaving a comment. For truly complex scenarios where a comment is needed, be as clear/concise as possible while being technically accurate. Assume a human will be the main audience of the comemnt.
## Tests
- Tests that current fail but should work eventually are allowed and encouraged. Testing is not just for regressions, but also helps immensesly for building features.
## Verify constantly

- Run `cargo build` after every edit and run the tests pertinent to the
  change before finishing (the owning package, e.g. `cargo test -p
  glcc_frontend` or `cargo test -p fern --test lower`). Small steps,
  always green. (Past mishaps came from batching too many edits between
  compiles.) CI runs the full workspace suite, so there is no need to
  run it locally.

  Building `glcpp` (and with it EDG) is opt-in via `--features cxx`:
  `cargo build -p glcpp --features cxx` (or `cargo build --workspace
  --features cxx`). Plain workspace builds skip it.

## Error handling

- Parser/sema errors flow as `Result<T, CompileError>` internally, then
  emit via codespan (`Parser::emit` / `unwrap_diag`) and panic with the
  plain message. Keep panic-message prefixes stable: `#[should_panic]`
  tests match on substrings.
- Every AST struct carries a `span: Span` field, queried via `HasSpan`.
  Semantic keys (`Ty`, `DeclId`, `StructId`) stay spanless; their owners
  (`Binding`, `StructDef`) hold the spans. Enum nodes are covered by
  their parent struct's span.

## Architecture

- `parse.rs` produces owned AST; `Sema` owns entities (`decls`,
  `structs`, `bodies`, `inits`) keyed by ID. New cross-cutting passes
  (like `check.rs`) go in as `impl Sema` in their own module.
- `Ty` is a pure semantic type: no source positions inside (bounds are
  span-erased on lowering). Target assumptions (LP64, signed `char`)
  must be marked with comments where used.
- full #![no_std] support is not necessarily mandatory at this time, but we should not build features that explicitly require std
## Boundaries

- Never run broad filesystem searches (no `find /`, no registry-wide
  greps). Stay inside the repo and `~/.cargo` for crate sources.
- In general, searches for compiler-related code should be contained to `crates`, we have a lot of c++ submodules that slows down rg
- `examples/` holds runnable demos, not tests: keep `make` and `make
  clean` working in every example (see `examples/README.md`).
- Be VERY careful and considerate when using git stashes. when possible avoid it entirely.
