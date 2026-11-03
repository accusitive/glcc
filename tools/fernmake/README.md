# fernmake

A small Makefile runner written in fern. It is the drop-in build entry
point for the examples when GNU make is not available. The tool lives
under `tools/` so it is not an example.

Build it with the `build.sh` / `build.ps1` scripts; there is no
Makefile here, so the Makefile runner does not need GNU make to build
itself. The scripts rebuild everything on every invocation.

```bash
cd tools/fernmake
./build.sh                            # builds build/fernmake
./build.sh run                        # builds, then runs the demo
./build.sh install                    # installs to /usr/local/bin
./build.sh install PREFIX=$HOME/.local  # or under another prefix
./build.sh clean                      # removes build/
```

`DESTDIR` stages the install into a packaging root, as usual. The
scripts look for `fernc` and `glcc` in the repo's `target/debug` and
`target/release` trees first (`cargo build -p glcc -p fernc`), then on
`PATH`; `FERNC`/`GLCC` environment variables override both.

Once built, use the binary for every example Makefile:

```bash
./build/fernmake demo.mk              # named makefile, default target
./build/fernmake demo.mk clean        # named target
(cd ../../examples/fern/showcase && ../../../tools/fernmake/build/fernmake)
```

With no arguments, fernmake builds `Makefile` in the current directory;
the example directories all have one.

On Windows, use `build.ps1`:

```powershell
.\build.ps1            # builds build\fernmake.exe
.\build.ps1 run        # builds, then runs the demo
.\build.ps1 clean      # removes build\
```

`demo.win.mk` uses PowerShell as its recipe shell; the POSIX `demo.mk`
uses `sh`.

Supported subset:

- comments, blank lines, and tab-indented recipes
- variables with `=`, `:=`, and `?=`, expanded as `$(NAME)`;
  `:=` is treated like `=` (lazy expansion), and `?=` still lets
  environment variables override the default
- `include` of another makefile, resolved relative to the including file
- rules with prerequisites and multiple targets
- `.PHONY`
- `$@`, `$<`, and `$^` in recipes
- nested `$(...)` expansion and the `firstword`, `wildcard`, `dir`,
  `shell`, `realpath`, and `abspath` functions
- backslash-newline continuations in variable definitions and recipes
- `MAKEFILE_LIST`
- up-to-date checks using file existence and modification times
- `@` at the start of a recipe to suppress echoing the command
- recipes through `$(SHELL)` and `.SHELLFLAGS` (default `sh -c`)

`wildcard`, `realpath`, `abspath`, and `$(shell ...)` still invoke `sh`;
the Windows makefiles avoid them. Pattern rules, conditionals,
`patsubst`, `foreach`, and recursive `$(MAKE)` are not implemented.
