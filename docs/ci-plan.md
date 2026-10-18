# CI plan for glcc

Status: implemented. The v1 core lives in `.github/workflows/ci.yml`
(three test jobs: Linux x86-64, Linux aarch64, Windows x86-64); this
document is the rationale and reference for what is and is not in CI.

Revision (2026-10-18, after the first public push): `fern` and `fernc`
are experiments, not part of the compiler contract, so the CI test
command excludes them. The first run showed why: `fern`'s JIT tests fail
on aarch64 inside `cranelift-jit`, and `fernc` panics lowering for Win64
(details under "Portability audit").

Revision (2026-10-18, second run): with the exclusion in place,
`linux-x86_64` (33 s warm) and `windows-x86_64` (2m12s) are green;
`linux-aarch64` failed exactly one test — the driver test `target_flag`
assumed `--target=aarch64` is a cross build, which is false on an
aarch64 host, and now derives its cross target from the host (audit
aarch64-6). The job stopped there, so most of the tree still has to
prove itself on arm.

Revision (2026-10-18, third run): the driver-test fix held —
`linux-x86_64` (41 s) and `windows-x86_64` (1m30s) green, and
`linux-aarch64` ran through `glcc_frontend` (568 passing) into the
`glcc_lower_mir` execution tests, where it failed two: narrow call
results kept their ABI cover width instead of the declared width
(audit aarch64-7, a real codegen bug, fixed) and one test assumed
signed `char` (audit aarch64-8, fixed).

Owner's goals, quoted:

> - Test the C frontend/compiler only — never C++ / EDG (no
>   `--features cxx`, no edg submodule; plain `cargo build`/`cargo test
>   --workspace` already skip EDG, verified).
> - Platforms: x86_64 Linux, x86_64 Windows, aarch64 Linux.
> - "Nothing egregious since I want to be fair to the others on the
>   free CI infra" — modest resource use; the repo will be public on
>   GitHub.
> - Windows regression coverage matters most (owner develops on Linux).

## Goals and non-goals

Goals

- Keep `cargo test --workspace --exclude fern --exclude fernc --locked`
  green on the three platforms. This covers the C frontend, MIR,
  backend, driver, and preprocessor (965 tests). `fern`/`fernc` are
  experiments and sit this one out — see "Deliberate exclusions".
- Cheap, boring, low-maintenance CI: three jobs, no submodules, no
  containers, no artifact uploads, cache only what pays for itself.
- Make Windows behaviour observable early, because the owner cannot
  iterate on it locally.

Non-goals

- **C++ / EDG.** No `--features cxx`, no `--all-features` (which would
  enable `cxx` via `crates/glcpp/Cargo.toml`), no `edg` submodule, no
  `glcpp` binary. The workspace already skips EDG by design:
  `edg-sys/build.rs:73-81` returns immediately unless `CARGO_FEATURE_CXX`
  is set, and `glcpp`'s binary is `required-features = ["cxx"]`
  (`crates/glcpp/Cargo.toml:6-14`).
- **Submodule suites.** `tools/torture.bash` needs the 2+ GB
  `tests/llvm-test-suite` checkout; `tools/ctestsuite.bash` needs
  `tests/c-testsuite`. Both are bash-driven and out of scope for now.
- **The full `abi_probe` matrix** (needs `zig` and `qemu-{aarch64,riscv64}`
  and a POSIX workdir; see the audit). One x86-64 case is cheap enough
  to run optionally — see below.
- **Release packaging / `cargo install` / wasm builds / examples.**
  `examples/*` are demos (`make`), not tests; the wasm module
  (`tools/build-wasm.sh`) is not part of the compiler contract.
- No coverage, no benchmarks, no macOS, no riscv64 runner (no free
  hosted riscv64 runner exists).

## What `cargo test --workspace` actually covers

Counted with `grep -c '#\[test\]'` (matches the verified 1069 passing
tests exactly; doc-tests add nothing material). CI runs these minus
`fern` and `fernc` (104 tests), so 965:

| crate | tests | kind |
|---|---|---|
| `glcc_frontend` | 568 | pure lexer/parser/sema/layout/eval; no processes, no filesystem, no platform cfg |
| `glcc_lower_mir` | 105 | frontend → MIR, plus in-process JIT execution with `Target::native()` (`crates/glcc_lower_mir/src/lower.rs:8930-8963`) |
| `glcc_mir` | 90 | pure IR/passes |
| `glcc_lower_cl` | 83 | lowering + JIT execution; x86-64 execution tests are gated `#[cfg(all(target_arch = "x86_64", target_os = "linux"))]` (e.g. `crates/glcc_lower_cl/src/lower.rs:3132,3361,3406`) |
| `glcc_pp` | 59 | pure preprocessor + tempdir round-trips via `std::env::temp_dir()` |
| `glcc` (root `src/driver.rs`) | 43 | driver arg parsing, link-command assembly, in-process object emission; two asserts are `#[cfg(not(windows))]` (`.eh_frame`) |
| `glcc_explorer` | 16 | in-process compile; disassembly assertions are skipped when no `objdump`/`llvm-objdump` is on PATH (`crates/glcc_explorer/src/lib.rs:561-569`) |
| `glcc_toolchain` | 1 | pure version-ordering test |
| `glcc_dwarf`, `abi_probe`, `glcpp` | 0 | — |
| `fern` | 94 | experiment, excluded from CI (JIT tests fail on aarch64; see audit) |
| `fernc` | 10 | experiment, excluded from CI (Win64 lowering panics; see audit) |

Host-runtime notes that matter for CI:

- **No test invokes the `glcc` binary.** The only `CARGO_BIN_EXE_*` use
  is `fernc` testing itself (excluded from CI). Driver tests call
  `compile_source` / `link_command` in-process.
- **`build.rs` staging is a build side effect, not a test dependency.**
  On Linux it stages `~/.local/lib/glcc/glccrt.o` with `cc` (+`ld -r`
  on x86-64); on Windows it warns and skips unless clang, MSVC
  `lib.exe`, and a previously-built `glcc` binary exist
  (`build.rs:91-111`, `build.rs:163-178`). Tests pass with the runtime
  missing.
- **`abi_probe` is not wired into `cargo test`.** It is a standalone
  binary (`crates/abi_probe/Cargo.toml`, bin `abi-probe`) that must be
  run explicitly.

External tools actually required by `cargo test --workspace`:

| tool | Linux | Windows | needed by |
|---|---|---|---|
| `cc` | yes (image has it) | no (nothing left needs it) | the Linux driver smoke step; `build.rs` staging (build side effect, not tests) |
| `clang` | no (native pp default) | no | only `GLCC_PP`/seeding paths, unused in CI |
| `as`, `objdump` | present on images | not needed | optional test branches degrade gracefully |
| MSVC `link.exe`/`lib.exe` | n/a | not needed for tests | only real links and `build.rs` staging |

## Recommended workflow

File: `.github/workflows/ci.yml`. Triggers: pushes to `master` (the
default branch), all pull requests, and manual dispatch. `concurrency`
cancels superseded runs per ref, so force-pushing a PR costs nothing.

### Job matrix

Test command in every row: `cargo test --workspace --exclude fern
--exclude fernc --locked` (the two experiments are excluded, see below).

| job | runner label | what runs | timeout | cache |
|---|---|---|---|---|
| `test (linux-x86_64)` | `ubuntu-24.04` | test command; optional driver smoke | 20 min | rust-cache, save only on master |
| `test (linux-aarch64)` | `ubuntu-24.04-arm` | test command | 20 min | rust-cache, save only on master |
| `test (windows-x86_64)` | `windows-2025` | `ilammy/msvc-dev-cmd` + test command; optional smoke | 30 min | rust-cache (registry-only fallback) |
| `lint` (optional, off by default) | `ubuntu-24.04` | `cargo fmt --all --check`, `cargo clippy --workspace --all-targets` | 10 min | rust-cache |
| `abi-probe` (optional) | `ubuntu-24.04` | `cargo run -p abi_probe -- --target=x86_64` | 15 min | rust-cache |

Runner-label facts, verified October 2026 (docs.github.com hosted-runners
reference and `actions/runner-images` README):

- Standard hosted runners are **free and unlimited on public
  repositories**, including `ubuntu-24.04-arm` (arm64 standard runners
  are free for public repos since their 2025 GA; private-repo arm64 only
  became available in Jan 2026).
- `windows-latest`/`windows-2025` currently resolve to the Windows Server
  2025 image with Visual Studio 2026; it ships **LLVM 20.1.8**, VSWhere
  3.1.7, Windows SDK 10.0.26100, MSVC x64 tools (`VSWhere`, `lib.exe`,
  `link.exe`), and Rust 1.99. A `windows-2022` label also exists. Pin
  `windows-2025` so an image migration (`windows-latest` has already
  moved VS versions twice in 2026) cannot silently change the job.
- `ubuntu-latest` is currently Ubuntu 24.04; pin `ubuntu-24.04` for the
  same reason.

### YAML sketch

```yaml
name: CI

on:
  push:
    branches: [master, main]
  pull_request:
  workflow_dispatch:

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

permissions:
  contents: read

env:
  CARGO_TERM_COLOR: always
  CARGO_INCREMENTAL: "0"          # smaller target dirs for caching

jobs:
  test:
    name: test (${{ matrix.name }})
    strategy:
      fail-fast: false            # a Windows failure must not cancel Linux
      matrix:
        include:
          - { name: linux-x86_64,  os: ubuntu-24.04,     timeout: 20 }
          - { name: linux-aarch64, os: ubuntu-24.04-arm, timeout: 20 }
          - { name: windows-x86_64, os: windows-2025,    timeout: 30 }
    runs-on: ${{ matrix.os }}
    timeout-minutes: ${{ matrix.timeout }}
    steps:
      # submodules: false is the default. Spelled out as intent: the
      # edg submodule is never needed, and the test-suite submodules
      # pull 2+ GB.
      - uses: actions/checkout@v7
        with:
          submodules: false
      - uses: dtolnay/rust-toolchain@stable   # edition 2024 needs >= 1.85
      - name: Set up MSVC (link.exe/lib.exe on PATH, LIB set)
        if: runner.os == 'Windows'
        uses: ilammy/msvc-dev-cmd@v1
      - uses: Swatinem/rust-cache@v2
        with:
          key: ${{ matrix.name }}
          save-if: ${{ github.ref == 'refs/heads/master' || github.ref == 'refs/heads/main' }}
      # fern/fernc are experiments; their aarch64 JIT and Win64 paths
      # are unfinished, so they stay out of the gate.
      - name: Build and test
        run: cargo test --workspace --exclude fern --exclude fernc --locked
      # Optional smoke test, after tests pass: compile and run a C program
      # through the driver. Linux: build.rs staged glccrt.o, cc is present.
      - name: C driver smoke test
        if: runner.os != 'Windows'
        run: |
          printf '#include <stdio.h>\nint main(void) { puts("smoke-ok"); return 0; }\n' > "$RUNNER_TEMP/smoke.c"
          cargo run -q --locked -- "$RUNNER_TEMP/smoke.c" -o "$RUNNER_TEMP/smoke"
          "$RUNNER_TEMP/smoke"
      # Windows smoke: exercises link.exe + CRT end to end. Enable once
      # it is proven green (see staged rollout); keep out of v1.
      # - name: C driver smoke test
      #   if: runner.os == 'Windows'
      #   run: |
      #     Set-Content $env:RUNNER_TEMP\smoke.c "#include <stdio.h>`nint main(void) { puts(`"smoke-ok`"); return 0; }"
      #     cargo run -q --locked -- $env:RUNNER_TEMP\smoke.c -o $env:RUNNER_TEMP\smoke.exe
      #     & $env:RUNNER_TEMP\smoke.exe; if ($LASTEXITCODE -ne 0) { exit 1 }
```

Notes on specific choices:

- `cargo test --workspace --exclude fern --exclude fernc --locked`
  builds everything it needs; a separate `cargo build` step would only
  repeat work. `--locked` refuses lockfile churn (`Cargo.lock` is
  committed).
- `--exclude fern --exclude fernc` keeps the experiments out of the
  gate. It also drops their *compile* coverage in CI (nothing else
  depends on them); a local `cargo test --workspace` still runs them.
- **Never** use `--all-features` here: it enables `cxx` and with it the
  EDG build. The plain workspace selection is the contract.
- `ilammy/msvc-dev-cmd@v1` is defensive rather than strictly required:
  `crates/glcc_toolchain` discovers MSVC itself via `vswhere` (see
  audit), and `src/driver.rs:112-127` prefers `LIB` when set, else the
  discovered paths. The action matters for (a) `build.rs` Win64 runtime
  staging and (b) any future Windows link/run test.
- Pinning actions to full commit SHAs (`@<sha> # v6` style) is
  recommended for a public repo but left to the owner.
- Do not set `GLCC`, `GLCC_PP`, `GLCC_CC`, `GLCC_LD`, `GLCC_AS`,
  `GLCC_RT` in CI; they change driver behaviour and would invalidate
  test assumptions.

### Per-platform details

**Linux x86-64 (`ubuntu-24.04`)**

```sh
cargo test --workspace --exclude fern --exclude fernc --locked
```

No env vars needed. `cc` (gcc), `as`, `objdump`, `clang` are all on the
image. `build.rs` stages `$HOME/.local/lib/glcc/glccrt.o` (HOME=`/home/runner`),
so the optional smoke test works immediately.

**Linux aarch64 (`ubuntu-24.04-arm`)**

```sh
cargo test --workspace --exclude fern --exclude fernc --locked
```

Identical to x86-64. `build.rs`'s ELF path only needs `cc` on non-x86-64
(the `ld -r` soft-float combine is x86-64-only, `build.rs:60-70`). The
remaining JIT tests (`glcc_lower_mir`; `glcc_lower_cl`'s x86-64-only
execution tests are compiled out by their cfg gates) execute native
AArch64 code — this job is what exercises that backend. `fern`'s JIT
suite is excluded (it fails here; see audit). Nothing extra to install.

**Windows x86-64 (`windows-2025`)**

```powershell
# steps, in order
# 1. actions/checkout@v7 with submodules: false (default)
# 2. dtolnay/rust-toolchain@stable
# 3. ilammy/msvc-dev-cmd@v1
cargo test --workspace --exclude fern --exclude fernc --locked
```

What Windows needs and does not need, verified against the code:

- `cc` does not exist on the image. Nothing left in the gate needs it:
  the tests that used it (`fernc` link+run) are excluded.
- `clang` is not needed by tests. It *is* on the image (LLVM 20.1.8)
  for the future runtime-staging step.
- `link.exe`/`lib.exe` are not needed by tests. They are needed to
  link any produced `.exe`, and `build.rs` uses `lib.exe` to stage
  `glccrt-win64.lib`.
- Windows runtime staging is a two-pass affair on a clean checkout:
  the first build script run sees no `glcc.exe` yet and warns
  (`build.rs:100-103`). If the staged runtime is wanted, after the
  first `cargo build`, force a build-script rerun and build again:

  ```powershell
  cargo build --workspace --locked
  (Get-Item rt/glccrt.c).LastWriteTime = Get-Date
  cargo build --workspace --locked   # now finds target\debug\glcc.exe
  ```

  This is optional; `cargo test` does not depend on it.

**`actions/checkout` and submodules**

`submodules: false` is already the default; checkout leaves `edg/`,
`tests/c-testsuite/`, and `tests/llvm-test-suite/` as empty directories.
Verified: `cargo test --workspace` passes with no submodules initialized.
Do not use `submodules: true` or `recursive`.

### aarch64 approach: native runner first, qemu fallback if needed

Primary: `ubuntu-24.04-arm` (native, free for public repos). This runs
real AArch64 codegen and real AArch64 JIT tests — which is the point.

Fallback if the repo were private or arm runners were unavailable:
cross-compile + run under qemu-user on an x86-64 runner. Sketch:

```sh
sudo apt-get install -y gcc-aarch64-linux-gnu libc6-dev-arm64-cross qemu-user
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -L /usr/aarch64-linux-gnu" \
cargo test --workspace --exclude fern --exclude fernc --locked --target aarch64-unknown-linux-gnu
```

Caveats, from the audit:

- Test binaries are AArch64 and run under qemu, so the in-process JIT
  tests exercise the AArch64 backend correctly (`cfg!(target_arch)`
  decides at compile time). Good.
- The `fernc` tests spawn the *host* `cc` and would fail to link
  AArch64 objects; masking `cc` with a wrapper pointing at
  `aarch64-linux-gnu-gcc` fixes them — but `fernc` is excluded from CI
  anyway, so the gate needs no wrapper.
- `build.rs` would stage a host-x86-64 `glccrt.o`; harmless for tests,
  wrong for e2e.
- Emulation makes the suite slower and adds a class of flakes the native
  runner does not have. Treat this strictly as a fallback.

### Caching

Recommendation: `Swatinem/rust-cache@v2` everywhere, with per-matrix
`key: ${{ matrix.name }}` and `save-if` limited to `master` pushes.

- It caches `~/.cargo/registry`, `~/.cargo/git`, and `target/`, and
  prunes the things that bloat (incremental dirs, stale fingerprints,
  old versions). That is strictly better than hand-rolled
  `actions/cache` entries.
- `CARGO_INCREMENTAL=0` keeps the cached target smaller and avoids
  incremental-cache churn; CI gains nothing from incremental.
- Per-platform keys prevent Linux artifacts from poisoning Windows
  (`$CARGO_TARGET_DIR` artifacts are host-specific) and the
  arm/x86 split is mandatory.
- `save-if: refs/heads/master` means PR runs restore but do not write;
  that keeps the 10 GB repo cache budget from filling with one entry
  per PR and is the "fair to others" choice.

What **not** to cache:

- **The Windows `target/` tree if it grows large.** The local debug
  tree here is 3.6 GB (including wasm/flycheck artifacts) and a clean CI
  debug tree with cranelift is still ~1.5-2.5 GB; one naïve
  `actions/cache` entry for it would dominate the whole repo cache
  budget and spend more time uploading than it saves. Use rust-cache
  (prunes), and if the Windows cache entry exceeds ~1.5 GB in the
  action summary, switch that job to `cache-targets: false` (registry
  only) and eat the recompile.
- `~/.local/lib/glcc/glccrt.o` (staging): tiny, host-specific,
  regenerated by `build.rs` in seconds.
- Submodule checkouts: never fetched at all.
- Test artifacts: nothing is uploaded.

### Deliberate exclusions

| excluded | why |
|---|---|
| `fern`, `fernc` | experiments outside the compiler; their aarch64 JIT and Win64 lowering paths are unfinished (first-run findings in the audit). Local `cargo test --workspace` still runs them |
| `--features cxx`, EDG, `glcpp` binary | owner decision; plain workspace commands already skip them |
| `tests/llvm-test-suite` + `tools/torture.bash` | 2+ GB submodule; long bash-driven run; not needed to keep the C compiler honest day-to-day |
| `tests/c-testsuite` + `tools/ctestsuite.bash` | submodule + bash + zig/qemu cross targets; good as a *later* nightly, not a per-PR job |
| `abi_probe` full matrix | needs zig + `qemu-aarch64`/`qemu-riscv64` |
| `examples/*` | demos with their own `make`; the C++/SDL triangle example would drag in C++ toolchains |
| wasm build, `cargo install`, docs publish | not part of the test contract for this plan |
| macOS, riscv64 | not requested; no free riscv64 hosted runner |
| `cargo fmt`/`clippy` as *gating* | owner didn't ask; repo may not be clean today. Optional separate job, non-blocking at first |

## Portability audit: what would bite on Windows / aarch64

### Windows

1. **Fern JIT tests are compiled out.** `crates/fern/tests/jit.rs:9` is
   `#![cfg(not(windows))]` with the comment that Win64 convention is not
   implemented for that path. Moot for CI: the crate is excluded as an
   experiment. The `glcc_lower_cl`/`glcc_lower_mir` JIT tests
   (Win64-aware, `crates/glcc_lower_mir/src/lower.rs:8943-8945`) still
   run.
2. **`fernc` cannot lower for Win64 at all** (observed on the first CI
   run). `-p fernc --test driver`: 3 of 8 tests fail; `fernc` panics at
   `crates/glcc_lower_cl/src/lower.rs:142` with `cannot declare extern
   fern_enter`. Excluded from CI while it is an experiment — do not
   mistake the exclusion for Win64 support.
3. **Win64 runtime staging cannot complete on a clean first build.**
   `build.rs:91-111` warns and returns unless clang, `lib.exe`, and a
   previously built `glcc` exist; `find_glcc` (`build.rs:163-178`) only
   sees `$GLCC` or `target/<profile>/glcc.exe`. First CI run warns
   (`cargo:warning=...`), which is expected and harmless for
   `cargo test`; see the two-pass recipe if staging is wanted.
4. **MSVC discovery leans on `vswhere`.**
   `crates/glcc_toolchain/src/lib.rs:201-230` runs
   `vswhere -latest -products *`; the hard-coded fallback list
   (`:221-227`) only enumerates VS 2019/2022 directories, while
   windows-2025 ships VS 2026 under `...\Microsoft Visual Studio\18\...`.
   VSWhere 3.1.7 *is* installed, so this works today — but it is the
   fragile link to watch if image contents change.
5. **Runtime-path test is synthetic on Windows.**
   `src/driver.rs:3511-3521` asserts `/usr/lib/glcc/...` exactly. It
   passes on Windows only because `PathBuf::join` and `PathBuf::from`
   produce the same (drive-relative) path; it does not verify Windows
   precedence (the real code also checks `%USERPROFILE%`,
   `src/driver.rs:238-248`). Cosmetic, but don't treat it as Windows
   coverage.
6. **Archive path needs `ar` outside the MSVC branch.**
   `src/driver.rs:2355-2381`: `msvc_link(job)` routes to `lib.exe`;
   otherwise `ar` is spawned. Windows CI never hits the non-MSVC branch
   (`GLCC_CC` unset), but a future Windows test with `GLCC_CC` set
   would need GNU `ar`.
7. **No test links or runs a Windows `.exe` at all.** Everything the
   driver does with `link.exe` is unit-tested as argument vectors
   (`src/driver.rs:3547-3575`); the first real link will be the optional
   smoke test.

Nothing found in the suite that hard-fails Windows on inspection:
no `#[ignore]`, no `symlink`, no `Command::new("sh")`, no `/home`, no
unix-permission usage, `std::os::unix` appears once with a Windows twin
(`src/driver.rs:535-548`), `timefmt` has a non-unix fallback
(`crates/glcc_pp/src/timefmt.rs:48-64`), and the frontend's
`native_matches_the_host` test is Windows-conditional
(`crates/glcc_frontend/src/target.rs:520-525`).

### aarch64

1. **`abi_probe` cannot run cross-arch.**
   `crates/abi_probe/src/main.rs:16` hardcodes
   `/tmp/opencode/abi_probe`; `:498-520` needs `cc` for x86-64 and
   `zig cc -target ...-musl` for aarch64/riscv64; `:56-57` always runs
   those under `qemu-aarch64`/`qemu-riscv64`; and the x86-64 case builds
   its ISA with `cranelift_native` (`:562`), so it is only meaningful on
   an x86-64 host. On an arm runner, `--target=aarch64` still needs zig
   and qemu-user installed. Keep it out of the matrix (or run only the
   x86-64 case on x86-64).
2. **x86-only execution tests gate correctly off.** e.g.
   `crates/glcc_lower_cl/src/lower.rs:3132` — so arm loses those, but
   they cannot fail to compile.
3. **`glcc_lower_mir` tests have no arch cfg and execute native code.**
   They are written against `Target::native()` and long-double kind
   selection is dynamic (`crates/glcc_lower_mir/src/lower.rs:9608-9613`).
   The third run exercised them on arm: 103 of 105 passed; the two
   failures are audit aarch64-7 (real, fixed) and aarch64-8 (test
   assumption, fixed).
4. **`fern` JIT tests fail on aarch64** (observed on the first CI run):
   `-p fern --test jit` — 17 of 24 fail, all panicking inside
   `cranelift-jit 0.136.2` (`compiled_blob.rs:324`,
   `TryFromIntError(NegOverflow)`, while applying an AArch64 ADRP
   relocation). Upstream JIT issue, not a glcc codegen regression; the
   crate is excluded from CI as an experiment.
5. Bash tooling (`tools/*.bash`, `tests/*/run.sh`) is excluded anyway.
6. **`target_flag` assumed aarch64 is a cross target** (second run,
   fixed). The test asserted `--target=aarch64` forwards
   `--target=aarch64-linux-gnu`, but `Job::target_args()`
   (`src/driver.rs:504-514`) forwards nothing when the target is
   native — and the arm runner *is* native aarch64. It now picks a
   target that is cross relative to the host. Any test that hardcodes
   "arch X is cross" breaks on an X host.
7. **Narrow call results kept their cover width** (third run, fixed).
   `int_cover` (`crates/glcc_lower_cl/src/abi.rs:534-549`) widens
   sub-32-bit scalars to W-register width on AAPCS64, and
   `map_call_results` handed that I32 back as a value whose MIR type is
   `unsigned char`; the next widening cast then emitted `uextend.i32`
   on an i32, which the Cranelift verifier rejects
   (`bool_to_byte_convert_is_a_no_op`). Entries, arguments and returns
   already normalized at their borders; results now do too
   (`narrow_to_declared`, `crates/glcc_lower_cl/src/lower.rs`). x86-64
   is unaffected: its covers are exact.
8. **`switch_promotion` assumed signed `char`** (third run, fixed):
   plain `char` is unsigned on aarch64 by ABI, and glcc models that
   (`Target::AARCH64.char_signed == false`, `char_signedness_matches_clang`),
   so `char c = -1` is 255 and matching `case 255` is the correct
   result. The test now spells `signed char` for the sign-extension
   assertion and checks the plain-`char` case against
   `Target::native().char_signed`.
9. **The arm link line never carried `-lm`** (fourth run, fixed).
   The aarch64/riscv64 quad trampolines in `rt/glccrt.c:650-731`
   branch to the toolchain's `sqrtl`, so linking the staged runtime
   needs libm on those targets. The smoke test (a plain `puts`) failed
   with `undefined reference to 'sqrtl'`; x86-64's runtime is a
   soft-float `ld -r` combine that references no libm symbol, which is
   why its smoke had always passed. `quad_runtime_needs_libm`
   (`src/driver.rs`) now appends `-lm` right after the runtime on
   aarch64/riscv64, leaving x86-64 link lines byte-identical.

### Both

- `abi_probe`'s XFAIL list has only two entries (`union_mixed`,
  `union_fp`; `crates/abi_probe/src/main.rs:384,398`), so the ABI story
  is essentially green across x86_64/aarch64/riscv64 — evidence that
  the aarch64 backend is in good shape.

## Risks, unknowns, staged rollout

Staged rollout:

- **Week 1 — Linux x86-64 only.** Merge the workflow with one matrix
  row. Confirm green (965 CI tests; 1069 with fern/fernc locally),
  record cold/warm timings and cache sizes. Cheap to revert, zero risk
  to other platforms.
- **Week 2 — add aarch64.** Native `ubuntu-24.04-arm`. Expect green
  (see audit); if a test class fails, either gate it upstream properly
  or run the arm job `continue-on-error: true` while fixing. Do not
  leave it yellow forever.
- **Week 3 — add Windows.** Run `windows-2025` with
  `continue-on-error: true` for the first PRs, collect failures, fix,
  then flip to gating (`fail-fast: false` stays). `fern`/`fernc` are
  out of the gate, so anything red is a real finding.
- **Later (optional):** fmt/clippy job; `abi_probe --target=x86_64`
  job; the driver smoke on all three; two-pass Windows runtime staging +
  a quad/setjmp smoke to cover `glccrt-win64.lib`.

Implemented note: all three jobs landed together (the workflow above).
First run (2026-10-18): `linux-x86_64` green; `linux-aarch64` and
`windows-x86_64` failed in `fern`/`fernc` (see audit) before reaching
the rest of the tree — cargo stops at the first failing test target.
`fern`/`fernc` are now excluded. Second run: `linux-x86_64` green
(33 s), `windows-x86_64` fully green (2m12s), `linux-aarch64` red on a
host-dependent driver test (audit aarch64-6). Third run: x86-64 and
Windows green again; arm got through `glcc_frontend` and
`glcc_lower_cl` and failed two `glcc_lower_mir` execution tests
(audit aarch64-7/-8), both fixed since. Fourth run: x86-64 (36 s) and
Windows (1m17s) green; arm's *whole* test suite passed — `cargo test`
walked every target to the end — and only the `C driver smoke test`
failed: the staged runtime's quad trampolines branch to libm's
`sqrtl`, and the link line never carried `-lm` (audit aarch64-9),
fixed since. If Windows goes red, the cheap dial is
`continue-on-error: true` on that matrix row; do not leave it yellow
forever.

How to validate Windows assumptions cheaply *before* merging:

- Push a branch with the workflow to the public repo and open a PR:
  Actions runs on PRs without merging, and `workflow_dispatch` lets the
  owner re-run after tweaks. This is the cheapest real answer.
- `nektos/act` cannot emulate the Windows runner; don't rely on it.
- Read the job log for the `cargo:warning=...` lines from `build.rs` to
  confirm which staging path was taken.

Unknowns / risks:

- **Windows verified green** (second run, experiments excluded): the
  whole workspace ran and passed on `windows-2025` in 2m12s. The smoke
  test (link + run) is still off; see follow-ups.
- **aarch64 suite verified** (fourth run): every test target passed on
  arm; the only failure left was the driver smoke link, audit
  aarch64-9, fixed. The next run should close out the platform.
- **Runner image churn.** windows-latest moved to VS 2026 during 2026;
  arm labels are still young. Pin explicit labels (`windows-2025`,
  `ubuntu-24.04`, `ubuntu-24.04-arm`) and revisit annually.
- **Rust version drift.** No `rust-toolchain.toml`; the plan installs
  `stable` (runner images already carry 1.99). If reproducibility
  matters, pin `dtolnay/rust-toolchain@1.85` (edition 2024 MSRV) or add
  a `rust-toolchain.toml`. Watch out for new lints breaking builds.
- **Cache value is unmeasured.** Estimates below; adjust after week 1.
- **Public-repo abuse surface**: `pull_request` from forks with secrets
  — none are used, `permissions: contents: read` covers it.

Rough estimates (to be measured):

| job | cold | warm (cache hit) | cache size |
|---|---|---|---|
| linux x86-64 | 5-8 min | 1.5-3 min | ~0.5-1 GB |
| linux aarch64 | 5-8 min | 1.5-3 min | ~0.5-1 GB |
| windows x86-64 | 8-12 min | 3-5 min | ~1-2 GB (watch this) |

Measured runs (2026-10-18): `linux-x86_64` 1m24s cold, then 33 s, 41 s
and 36 s warm; `windows-x86_64` 2m12s, 1m30s and 1m17s; `linux-aarch64`
51 s and 53 s to its early failures, then 1m5s with the full suite
green and only the smoke link failing.

Cost: **$0** while the repo is public — standard GitHub-hosted runners,
including `ubuntu-24.04-arm`, are free for public repositories. For a
private fork the same run is roughly $0.008/min (Linux x64), $0.005/min
(arm64), $0.016/min (Windows) as of the 2026 pricing; three jobs stay
well under an hour of wall time per push either way. Cache storage
counts against the 10 GB per-repo budget; the recommendations above keep
it to a few hundred MB per platform and it expires after 7 idle days.

## Follow-ups this plan creates (not done here)

1. (done) `.github/workflows/ci.yml` exists, built from the sketch above.
2. (done) `fern`/`fernc` are excluded from the CI test run: experiments
   whose aarch64 JIT and Win64 lowering paths are unfinished (see
   audit). Revisit only if they stop being experiments; a local
   `cargo test --workspace` still covers them on x86-64 Linux.
3. If compile coverage for the experiments in CI is wanted without
   running their tests, add a `cargo build -p fern -p fernc` step.
4. If Windows link/run coverage is wanted, stage
   `glccrt-win64.lib` in CI (two-pass recipe) and promote the smoke
   test to gating.
5. Optional: pin `rust-toolchain.toml`; add a fmt/clippy job once the
   tree is clean under the pinned toolchain.
