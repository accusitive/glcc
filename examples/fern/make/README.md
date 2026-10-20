# fernmake

A small Makefile runner written in fern. It is the drop-in build entry
point for the examples when GNU make is not available.

With GNU make in the repo:

```bash
make
make run                              # runs the demo
```

Without GNU make, bootstrap it once with `fernc` and a C compiler, then
use the resulting binary for every example Makefile:

```bash
FERNC=../../../target/debug/fernc CC=cc ./bootstrap.sh
./build/fernmake demo.mk              # default target
./build/fernmake demo.mk clean        # named target
./build/fernmake path/to/Makefile all
```

`./build/fernmake Makefile` rebuilds fernmake from its own Makefile, so
the example is self-hosting after the initial bootstrap.

Supported subset:

- comments, blank lines, and tab-indented recipes
- variables with `=`, `:=`, and `?=`, expanded as `$(NAME)`;
  `:=` is treated like `=` (lazy expansion), and `?=` still lets
  environment variables override the default
- `CC` defaults to `cc`, and can be overridden through the environment
- rules with prerequisites and multiple targets
- `.PHONY`
- `$@`, `$<`, and `$^` in recipes
- nested `$(...)` expansion and the `firstword`, `wildcard`, `dir`,
  `shell`, `realpath`, and `abspath` functions
- backslash-newline continuations in variable definitions and recipes
- `MAKEFILE_LIST`
- up-to-date checks using file existence and modification times
- `@` at the start of a recipe to suppress echoing the command

Recipes are passed to `sh -c`, so pipes, redirection, and other shell
syntax work. Pattern rules, `include`, conditionals, `patsubst`,
`foreach`, and recursive `$(MAKE)` are not implemented.
