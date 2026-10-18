# fern examples

Each example is a `.fern` program compiled with `fernc` into an object
and linked against a small C driver that calls `answer()`:

- `showcase/` — sum types, type classes, higher-order functions;
  `answer()` returns 137.
- `brainfuck/` — a Brainfuck interpreter built from immutable lists;
  `answer()` returns a rolling checksum of everything it emitted.

`make && make run` in either directory. `fernc` is looked up in the
repo's build tree first (`cargo build -p fernc`), then on `PATH`.
