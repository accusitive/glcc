# fern standard library

The standard library is ordinary fern source, exposed through the `std`
namespace. `import std::foldable` resolves to `stdlib/std/foldable.fern`
in a development checkout.

Module search order:

1. the entry file's directory
2. directories in `FERN_PATH` (colon-separated on Unix)
3. the standard-library root from `FERN_STD`
4. `stdlib/std` in a development checkout
5. `/usr/share/fern/std`
6. `$HOME/.local/share/fern/std`

`FERN_STD` should point directly at the directory containing modules such
as `foldable.fern`, not at its parent. `FERN_PATH` entries are ordinary
module roots, so `import std::foldable` looks for
`<root>/std/foldable.fern`.
