# fern examples

Each example is a `.fern` program compiled with `fernc` into an object
and linked against a small C driver that calls `answer()`:

- `showcase/` — sum types, type classes, higher-order functions;
  `answer()` returns 137.
- `brainfuck/` — a Brainfuck interpreter built from immutable lists;
  it imports `std::foldable`, and `answer()` returns a rolling checksum of
  everything it emitted.
- `modules/` — a two-file program: `main.fern` imports `math.fern`,
  uses a qualified generic function and constructor, and `answer()`
  returns 42.

`make && make run` in any directory. `fernc` is looked up in the
repo's build tree first (`cargo build -p fernc`), then on `PATH`.
