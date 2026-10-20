# fern examples

Each example is a `.fern` program compiled with `fernc` into an object
and linked against a small C driver by `glcc`, the repository's own C
compiler:

- `showcase/` — sum types, type classes, higher-order functions;
  `answer()` returns 137.
- `brainfuck/` — a Brainfuck interpreter built from immutable lists.
  Programs are parsed from Brainfuck source strings, `demo()` prints what
  each one emits, and `./build/brainfuck hello.bf` runs a program loaded
  from disk. It imports `std::foldable` and `std::list`, and `answer()`
  returns a rolling checksum of everything the built-in programs emitted.
- `modules/` — a two-file program: `main.fern` imports `math.fern`,
  uses a qualified generic function and constructor, and `answer()`
  returns 42.

The Makefile runner lives in `tools/fernmake`, not here; once
bootstrapped, it can drive every example Makefile without GNU make.

On Linux and macOS, run `make && make run` in any example directory.
On Windows, build the tools with `cargo build -p glcc -p fernc`, build
`tools/fernmake/build/fernmake.exe` once (see
`tools/fernmake/README.md`), then use the PowerShell recipes:

```powershell
..\..\..\tools\fernmake\build\fernmake.exe Makefile.win
..\..\..\tools\fernmake\build\fernmake.exe Makefile.win run
```

The platform Makefiles share their rules through `common.mk`. The POSIX
ones look for `fernc` and `glcc` in the repo's build tree first
(`cargo build -p glcc -p fernc`), then on `PATH`; `Makefile.win`
defaults to `target\debug` and accepts `FERNC`/`GLCC` overrides. The
brainfuck driver also accepts a source file: `./build/brainfuck hello.bf`.
