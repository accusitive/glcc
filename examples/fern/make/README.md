# fernmake

A small Makefile runner written in fern. It is the drop-in build entry
point for the examples when GNU make is not available.

With GNU make in the repo:

```bash
make
make run                              # runs the demo
```

Without GNU make, bootstrap it once with `fernc` and `glcc`, then use
the resulting binary for every example Makefile:

```bash
FERNC=../../../target/debug/fernc \
GLCC=../../../target/debug/glcc \
./bootstrap.sh

./build/fernmake demo.mk              # default target
./build/fernmake demo.mk clean        # named target
./build/fernmake path/to/Makefile all
```

On Windows, build the runner with `fernc` and `glcc` directly, then use
`Makefile.win`:

```powershell
New-Item -ItemType Directory -Force -Path build
..\..\..\target\debug\fernc.exe -O1 make.fern -o build\fernmaker.obj
..\..\..\target\debug\glcc.exe driver.c build\fernmaker.obj -o build\fernmake.exe
.\build\fernmake.exe Makefile.win
.\build\fernmake.exe Makefile.win run
```

`Makefile.win` and `demo.win.mk` use PowerShell as their recipe shell;
the POSIX `Makefile` and `demo.mk` use `sh`.

`./build/fernmake Makefile` rebuilds fernmake from its own Makefile, so
the example is self-hosting after the initial bootstrap. The platform
Makefiles share their rules through `../common.mk`.

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
