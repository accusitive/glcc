# fernmake

A small Makefile runner written in fern.

```bash
make
make run                              # runs the demo
./build/fernmake demo.mk              # default target
./build/fernmake demo.mk clean        # named target
./build/fernmake path/to/Makefile all
```

Supported subset:

- comments, blank lines, and tab-indented recipes
- variables with `=`, `:=`, and `?=`, expanded as `$(NAME)`;
  `:=` is treated like `=` (lazy expansion)
- rules with prerequisites and multiple targets
- `.PHONY`
- `$@`, `$<`, and `$^` in recipes
- up-to-date checks using file existence and modification times
- `@` at the start of a recipe to suppress echoing the command

Recipes are passed to `sh -c`, so pipes, redirection, and other shell
syntax work. Make functions such as `$(wildcard ...)` and line
continuations are not implemented.
