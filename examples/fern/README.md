# fern examples

Each example is a `.fern` program compiled with `fernc` into an object
and linked against a small C driver:

- `showcase/` — sum types, type classes, higher-order functions;
  `answer()` returns 137.
- `brainfuck/` — a Brainfuck interpreter built from immutable lists.
  Programs are parsed from Brainfuck source strings, `demo()` prints what
  each one emits, and `./build/brainfuck hello.bf` runs a program loaded
  from disk. It imports `std::foldable`, and `answer()` returns a rolling
  checksum of everything the built-in programs emitted.
- `modules/` — a two-file program: `main.fern` imports `math.fern`,
  uses a qualified generic function and constructor, and `answer()`
  returns 42.
- `make/` — a small Makefile runner written in fern. It parses
  variables, rules, recipes, and `.PHONY`, then builds prerequisites and
  runs shell commands. `make run` invokes it on `demo.mk`.

`make && make run` in any directory. `fernc` is looked up in the
repo's build tree first (`cargo build -p fernc`), then on `PATH`. The
brainfuck driver also accepts a source file: `./build/brainfuck hello.bf`.
