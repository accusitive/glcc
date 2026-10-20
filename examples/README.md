# Examples

Runnable demos, one directory per example. These are not unit tests and
are not part of `cargo test`: each builds with `make`, cleans with
`make clean`, and leaves everything in its gitignored `build/`.

| directory | what it shows |
|---|---|
| `hello/` | a C hello world compiled with glcc |
| `inline-asm/` | glcc's inline asm: register operands, fixed registers, memory operands, `asm goto` |
| `triangle/` | a C++/SDL2 OpenGL triangle compiled with glcpp |
| `fern/` | fern programs compiled with fernc and linked into a C driver |
| `async/` | the `-fglcc-async` extension: a basic demo, twin infinite loops, and a multithreaded executor |

Each Makefile prefers the tool from the repo's build tree
(`cargo build`, `cargo build -p fernc`) and falls back to `glcc`,
`glcpp`, or `fernc` on `PATH`.

When GNU make is not installed, run `./bootstrap.sh` in
`tools/fernmake`, then use
`tools/fernmake/build/fernmake path/to/Makefile [target]` in place of
`make` or `make <target>`.
