# AGENTS.md — working conventions & orientation guide for glcc

glcc is a C compiler written in Rust ("George Lewis's C Compiler"), with an
experimental C++ front end (glcpp, backed by a vendored EDG) and an
experimental ML-family language (fern) built on the same IR. Targets:
x86-64 Linux (SysV), x86-64 Windows (GNU/MinGW + MSVC), aarch64 Linux,
riscv64 Linux. Backend is a forked Cranelift
(`accusitive/wasmtime`, branch `inline-asm-prototype`), plus an optional
in-repo "native" machine-code backend. Not production quality — heavily
tested against external suites instead.

## Part 1 — Working conventions (unchanged)

### Comments
- Do not leave "chain of thought" comments, do not leave superfluous comments on code that is clear. when possible, rewrite code to be self documenting via clearer control flow, variable names, etc instead of leaving a comment. For truly complex scenarios where a comment is needed, be as clear/concise as possible while being technically accurate. Assume a human will be the main audience of the comemnt.

### Tests
- Tests that current fail but should work eventually are allowed and encouraged. Testing is not just for regressions, but also helps immensesly for building features.

### Verify constantly

- Run `cargo build` after every edit and run the tests pertinent to the
  change before finishing (the owning package, e.g. `cargo test -p
  glcc_frontend` or `cargo test -p fern --test lower`). Small steps,
  always green. (Past mishaps came from batching too many edits between
  compiles.) CI runs the full workspace suite, so there is no need to
  run it locally.

  Building the C++ front end (crate `glcc_cxx`, binary `glcpp`) and
  with it EDG is opt-in via `--features cxx`:
  `cargo build -p glcc_cxx --features cxx` (or `cargo build --workspace
  --features cxx`). Plain workspace builds skip it.

### Error handling

- Parser/sema errors flow as `Result<T, CompileError>` internally, then
  emit via codespan (`Parser::emit` / `unwrap_diag`) and panic with the
  plain message. Keep panic-message prefixes stable: `#[should_panic]`
  tests match on substrings.
- Every AST struct carries a `span: Span` field, queried via `HasSpan`.
  Semantic keys (`Ty`, `DeclId`, `StructId`) stay spanless; their owners
  (`Binding`, `StructDef`) hold the spans. Enum nodes are covered by
  their parent struct's span.

### Architecture

- `parse.rs` produces owned AST; `Sema` owns entities (`decls`,
  `structs`, `bodies`, `inits`) keyed by ID. New cross-cutting passes
  (like `check.rs`) go in as `impl Sema` in their own module.
- `Ty` is a pure semantic type: no source positions inside (bounds are
  span-erased on lowering). Target assumptions (LP64, signed `char`)
  must be marked with comments where used.
- full #![no_std] support is not necessarily mandatory at this time, but we should not build features that explicitly require std

### Boundaries

- Never run broad filesystem searches (no `find /`, no registry-wide
  greps). Stay inside the repo and `~/.cargo` for crate sources.
- In general, searches for compiler-related code should be contained to `crates`, we have a lot of c++ submodules that slows down rg
- `examples/` holds runnable demos, not tests: keep `make` and `make
  clean` working in every example (see `examples/README.md`).
- Be VERY careful and considerate when using git stashes. when possible avoid it entirely.

## Part 2 — Orientation guide

### 2.1 Repo layout at a glance

```
Cargo.toml          workspace root; package `glcc` is the driver binary
build.rs            stages runtime (glccrt) + headers into ~/.local/lib/glcc
src/                the glcc driver (src/driver.rs is the heart, ~4.3k lines)
crates/             all pipeline crates (see 2.3)
tests/              external suite runners (c-testsuite, C++/EDG, sqlite, torture)
examples/           runnable demos with Makefiles (NOT cargo tests)
docs/               design docs (async ABI, EH plan, inline asm, EDG backend, ...)
rt/                 C runtime source (glccrt.c, glcc_async.c/h, softfloat/)
include/            glcc-owned std headers (stdarg.h, stddef.h, tgmath.h, ...)
stdlib/             the FERN standard library (not C!)
tools/              shell test runners + build scripts (ctestsuite.bash, ...)
edg/                git submodule: EDG C++ front end (for glcpp)
nasm/               git submodule: reference source for the glasm tables
```

Submodule notes: `edg/` must be initialized to build glcpp
(`git submodule update --init edg`). `nasm/` is only needed to regenerate
the assembler tables (`tools/build-glasm-data.bash`); the generated
`crates/glasm_data/src/insns_gen.rs` is committed so normal builds never
need nasm.

### 2.2 The compilation pipeline

```
source.c
  │  1. Preprocess: glcc_pp (native, in-process, default)
  │     or `clang -E` (--pp=clang / GLCC_PP). Output: text with `# <n> "file"`
  │     linemarkers that the lexer maps back to original files.
  ▼
glcc_frontend: Lexer → Parser (parser+checker FUSED)
  │     AST allocated in a bumpalo arena; declarations checked during parse;
  │     errors accumulate in Parser::diagnostics (cap 10).
  │     Result: Sema { decls: Vec<Binding>, structs: Vec<StructDef>,
  │                    enums, bodies: HashMap<DeclId, FuncBody>,
  │                    inits: HashMap<DeclId, Initializer>, ... }
  │     Expressions live as checked `CheckedExpr` trees (explicit Convert
  │     nodes for every implicit conversion).
  ▼
glcc_lower_mir: CheckedExpr/Sema → glcc_mir (SSA IR with aggregate types)
  │     -O0 discipline: every param/local in a memory slot; Mem2Reg promotes
  │     later. Then glcc_mir::optimize(module, level) + verify_module.
  ▼
┌─ BACKEND A (default, `-fbackend=cranelift`):
│    glcc_lower_cl: MIR → Cranelift CLIF → object (cranelift-object)
│      + own .eh_frame emission (ehframe.rs), .init_array (initarray.rs),
│      quad-precision (F80/F128) software support (quad/),
│      inline asm via the wasmtime fork's `inline_asm` instruction
│      (asm_native.rs) or out-of-line wrapper functions (asm/mod.rs).
│    glcc_dwarf: LoweredDebug → DWARF (.debug_*) via gimli, appended to
│      the cranelift-object product.
│
└─ BACKEND B (`-fbackend=native`, x86-64 Linux SysV only):
     glcc_codegen: MIR → machine IR (machine.rs) → instruction selection
       (sel/, declarative Pattern tables) → regalloc2 register allocation
       (ra.rs) → frame layout (frame.rs) → x86-64 encoding (x64/)
       → ELF via glasm. No Cranelift involved; F80/F128 unsupported here.
       Inline asm lowers in place (asm/ planner + OpId::ASM pseudo-inst,
       template rendered post-RA and assembled by glasm; see
       docs/inline-asm.md).
  ▼
driver (src/driver.rs): object bytes + per-TU asm text (file-scope/naked
  asm, assembled in-process by glasm with -fintegrated-as) + Win64
  .pdata/.xdata (src/winunwind.rs) + __ptr32 narrow reloc patching
  → link via host cc / mingw gcc / MSVC link.exe, appending glccrt.
```

Key IR convention: in MIR, aggregates are first-class (`Ty::Struct` with
layout baked in). In the Cranelift backend, every aggregate-typed value
lowers to the *address* of a stack slot; aggregate returns use a hidden
caller out-pointer.

### 2.3 Crate map

**Front end**

| Crate | What it is |
|---|---|
| `glcc_frontend` | The core: lexer, parser, sema, const eval, layout, `Target`, `LangOpts`, diagnostics. `no_std`-capable. Modules: `lexer.rs`, `span.rs`, `parse.rs` (~9.4k lines), `sema.rs`, `check/` (mod, atomic, sync, gnu, ia32, msvc, prototypes), `eval.rs`, `layout.rs`, `lang/`, `target/`, `ext/`, `diag/`, `dump.rs`. |
| `glcc_pp` | Native C preprocessor. `preprocess(source, filename, opts)`; expansion in `expand.rs` (blue paint rescan), `substitute.rs`, `tree.rs`; clang-matching predefines (`predefine.rs`) and include paths computed per-target. Corpus-tested against `clang -E`. |
| `glcc_cxx` | The `glcpp` binary (C++ flavor). Tiny main.rs; with `--features cxx` it installs `EdgOps` (FFI into EDG) into the root driver and runs `driver::run(args, Flavor::Cpp)`. |
| `edg-sys` | Raw FFI (`glcpp_edg_frontend*`) + build.rs that compiles the EDG front end into a static lib and stages its C++ runtime (`libC.a`). Only active under the `cxx` feature. `bridge/mir_gen_be.cpp` walks EDG IL and emits MIR text directly. |

**Middle / backends**

| Crate | What it is |
|---|---|
| `glcc_mir` | The middle IR: SSA with aggregate types, `FunctionBuilder`, `PassManager`, the pass suite in `passes/` (Mem2Reg, Sroa, Sccp, InstCombine, Inline, LoopUnroll/Vec, Slp, Licm, ...), coroutine lowering (`LowerCoroutines`), text parse/print (round-trip), verifier, debug records. |
| `glcc_lower_mir` | C → MIR lowering over `Sema`. `lower_translation_unit`; per-topic modules (expr, addr, data, init, atomic, complex, fp, vector, vla). Unimplemented features `todo!()` loudly. |
| `glcc_lower_cl` | MIR → Cranelift (production backend). `lower_module*` entry points; `abi/mod.rs` is the target/ABI brain (Arch/Os/Env, SysV vs WindowsFastcall, eightbyte classification, per-target va_list); `quad/` F128/F80 support; `asm/` + `asm_native.rs` inline asm; `ehframe.rs`; `initarray.rs`. Targets: x86-64 Linux/Win, aarch64, riscv64. |
| `glcc_codegen` | The native backend (Cranelift-free): MIR → machine IR → sel → regalloc2 → frame → encode → ELF. Targets: x86-64 SysV and aarch64 Linux (AAPCS64; integer + F32/F64, `src/aarch64/`). Adding an x86-64 instruction = three table rows in `x64/tables.rs` (`Op`, `OPCODES`/`ENCODINGS`, `PATTERNS`); aarch64 is hook-driven (`aarch64/mod.rs`) with the opcode/slot table in `aarch64/tables.rs`. |
| `glcc_dwarf` | DWARF emission from `LoweredDebug` records (produced by glcc_lower_cl) via gimli onto the cranelift-object product. |
| `glasm` + `glasm_data` | Native assembler ported from nasm (x86-64 full; hand-written RV32/64 and AArch64 core-integer encoders), GAS/AT&T frontend (`gas.rs`), ELF output via the `object` crate. `glasm_data` holds the generated ~26k-line instruction table (committed) so it recompiles rarely. The RISC-V and AArch64 backends are differential-tested against GNU as (`tests/riscv_diff.rs`, `tests/aarch64_diff.rs`); the aarch64 backend drives `-fintegrated-as` on native aarch64 builds. |
| `glcc_toolchain` | Host tool discovery: `find_clang()`, MSVC/Windows SDK layout (cfg(windows) gated). |

**Consumers / side projects**

| Crate | What it is |
|---|---|
| `glcc` (root) | Driver binary. Hand-rolled argv classifier (NOT clap — link order matters), actions: link/-c/-E/-M/-fsyntax-only/--emit-mir. Flags: `--target=`, `-std=`, `-O`, `-g`, `-fbackend=cranelift|native`, `-fintegrated-as`, `-fglcc-async`, `--pp=`, `--error-format=json`, `-###`. C++ path via `EdgOps`. Links via host cc/mingw/MSVC + glccrt. 55 unit tests at bottom of driver.rs. |
| `glcc_explorer` | Godbolt-style inspector: C source → MIR + asm panels. CLI + HTTP server (feature `server`) + wasm32 cdylib (`tools/build-wasm.sh`) running the full pipeline in the browser. |
| `abi_probe` | ABI conformance prober ("executable specification" of the C ABI): compiles cases with glcc, harnesses with cc/zig, runs under qemu-user, diffs stdout. `cargo run -p abi_probe`. |
| `fern` | An ML-family language (HM inference + type classes, monomorphizing lowering to glcc_mir). Shows MIR is language-agnostic. |
| `fernc` / `fern_exec` / `fern_repl` | fern's batch compiler / persistent JIT session / REPL. |

Dependency flow:

```
glcc_toolchain ─► glcc_pp ─► glcc_frontend ─► glcc_lower_mir ─► glcc_mir
                                                    │             │
                        ┌───────────────────────────┴─────────────┤
                        ▼                                         ▼
                 glcc_lower_cl ──► Cranelift ──► glcc_dwarf   glcc_codegen ──► glasm
                        │
              fern (frontend) ─► glcc_mir (same pipeline as C)
```

### 2.4 Key types & invariants to internalize

- **`Ty` (frontend)** vs **`mir::Ty`**: frontend `Ty` is target-independent
  and spanless (nominally keyed: `Struct(StructId)`, `Enum(EnumId)`); sizes
  come from `Sema::target`. `mir::Ty::Struct` is structural — C layout baked
  in at lowering time (`glcc_lower_mir/src/ty.rs`).
- **IDs**: `DeclId(u32)` indexes `sema.decls`; `StructId`/`EnumId` index
  `sema.structs`/`enums`. Forward-declared and completed `struct S` share
  one `StructId`. `Sema::bodies`/`inits` are keyed by `DeclId` and hold the
  *checked* forms.
- **Spans** are lexer line/col (NOT byte offsets); `byte_range`/`Parser::span_str`
  render them. `Sema::orig_lines` maps linemarker lines for DWARF.
- **Error flow**: `Result<T, CompileError>` → codespan emit → panic with
  stable message prefix (`#[should_panic]` tests depend on the substrings).
- **Targets**: five frontend constants (`Target::X86_64`, `AARCH64`,
  `RISCV64`, `WINDOWS_X86_64`, `WINDOWS_X86_64_GNU`); the Cranelift backend
  has its own `Target {Arch, Os, Env}` in `glcc_lower_cl/src/abi/mod.rs`
  and the native backend has `glcc_codegen`'s `Arch` — keep these in sync
  when adding a target.
- **Floats**: F32/F64 everywhere; `long double` is F80 on x86-64, F128 on
  aarch64/riscv64. The Cranelift backend carries F80/F128 as Clif `F128`
  with software support (`glcc_lower_cl/src/quad/` + `rt/softfloat/` +
  x87 asm in `rt/glccrt.c`); the native backend panics on them (on
  aarch64 that means `long double`/F128 — the next milestone routes it
  through `__*tf3` libcalls).
- **Runtime**: `rt/glccrt.c` (x87 f80, variadic trampolines, setjmp,
  quad trampolines) is compiled by root `build.rs` into
  `~/.local/lib/glcc/glccrt.o` (and `glccrt-win64.lib` / `glccrt-mingw.a`
  on Windows hosts). The driver appends it unless `-nostdlib`. Headers in
  `include/` are staged next to it and shadow toolchain headers.
- **Inline asm**: two paths in glcc_lower_cl — blocks fitting the fork's
  `inline_asm` Clif instruction (x86-64, `asm_native.rs`) vs out-of-line
  wrapper functions emitted as assembler text (`asm/mod.rs`). The
  wasmtime fork (`accusitive/wasmtime`, branch `inline-asm-prototype`)
  exists specifically for this instruction.

### 2.5 How to build & test

```bash
cargo build                          # everything except EDG/C++
cargo build --workspace --features cxx   # + glcpp/EDG (needs edg/ submodule)
cargo test -p glcc_frontend          # per-crate tests live INSIDE each crate
cargo test --workspace
cargo run -p abi_probe               # ABI conformance matrix (needs zig, qemu-*)
./install.sh                         # stage runtime + headers (+ EDG libC.a)
```

External suites (shell-driven, not cargo):

- `tools/ctestsuite.bash` — c-testsuite (879 tests in `tests/single-exec/`),
  native x86-64 plus aarch64/riscv64 cross via `tools/zig-cc` + qemu-user.
  Current known failures tracked in `ctestsuite-failures.txt`.
- `tests/cpp/run.sh` — EDG C++ end-to-end (~40 programs, output diffed
  against g++ when available; `xfail.txt`, XPASS detection).
- `tools/torture.bash` / `tests/run-llvm-cxx.sh` — gcc-c-torture / LLVM
  C++ suites (need the LLVM test-suite submodule in `tests/llvm-test-suite/`).
- `tests/sqlite/` — build SQLite amalgamation with glcc and run veryquick.
- `tools/pp-diff.bash` — differential preprocessor test vs `clang -E`.

Testing conventions:

- Frontend/backend unit tests are `#[cfg(test)]` modules inside crate
  sources (e.g. `glcc_frontend/src/tests/mod.rs` is ~11k lines);
  `tests/backend_entry.rs` pins the backend-facing API.
- `#[should_panic]` matches on message substrings — keep prefixes stable.
- Failing-but-should-pass tests are welcome (encode desired behavior).

### 2.6 Working patterns for common tasks

- **New C feature in the frontend**: usually touches
  `glcc_frontend/src/parse.rs` (AST + declaration parsing),
  `check/mod.rs` (or one of its family submodules for builtins),
  `eval.rs` (const folding), then `glcc_lower_mir` (lowering) and
  `glcc_lower_cl`/`glcc_codegen` if it needs codegen support. Update
  `docs/c-standard-audit.md` if it changes conformance.
- **New builtin**: add to the right `check/` family submodule
  (`gnu.rs`, `ia32.rs`, `msvc.rs`, `atomic.rs`), declare a prototype in
  `check/prototypes.rs`, lower in `glcc_lower_mir` (see `builtin.rs`,
  `fp.rs`, `ia32.rs`).
- **New aarch64 feature in the native backend**: `crates/glcc_codegen/src/aarch64/` — opcode/slot rows in `tables.rs`, encoding in `enc.rs` (fixed 32-bit words; golden tests pin them to GNU as output), AAPCS64 in `abi.rs`, selection in `mod.rs`. Narrow (sub-32-bit) values keep garbage upper halves; signed uses canonicalize with sxtb/uxtb at the use site.
- **New x86-64 instruction in the native backend**: three table rows in
  `crates/glcc_codegen/src/x64/tables.rs` (`Op` enum + `OPCODES`/slots +
  `ENCODINGS` recipe); pattern selection in `PATTERNS` or `select_custom`
  in `x64/mod.rs`.
- **New MIR pass**: `glcc_mir/src/passes/`, register in
  `glcc_mir/src/optimize/mod.rs`'s `optimize()` pipeline; implement
  `FunctionPass`/`ModulePass`.
- **New target**: frontend `Target` constant (`glcc_frontend/src/target/`),
  `glcc_lower_cl::abi` (`Arch`/`Os`/`Env`, call conv, va_list),
  glasm backend if the integrated assembler should work, runtime support
  in `rt/glccrt.c`, driver link logic in `src/driver.rs`.
- **Wasm**: only `glcc_explorer` builds for wasm32 (`tools/build-wasm.sh`);
  the driver has no wasm codegen target. `[profile.wasm]` in the root
  Cargo.toml (panic=abort, small binary).

### 2.7 Design docs worth reading before related work

- `docs/async-abi.md` — the `-fglcc-async` stackless coroutine ABI
  (`__attribute__((async))`, `_Future(T)`, `__builtin_await`; lowering in
  `glcc_mir/src/coroutine/`, runtime in `rt/glcc_async.c`).
- `docs/edg-mir-backend.md` — how EDG IL lowers to MIR text
  (`edg-sys/bridge/mir_gen_be.cpp`), replacing the generated-C round trip.
- `docs/eh-plan.md` — Itanium zero-cost EH design
  (implemented in `glcc_lower_cl/src/ehframe.rs`).
- `docs/inline-asm*.md` — inline asm design, implementation plan, and a
  catalogue of what the gcc-c-torture asm tests need.
- `docs/c-standard-audit.md` — ISO C conformance status (C99–C2y).
- `vector-size-plan.md`, `xmm-native-asm-plan.md` (repo root) — current
  work plans (GNU vector extensions; xmm constraints in native inline asm).
- `examples/README.md` — the demo Makefile conventions.
