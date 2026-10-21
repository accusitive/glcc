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

## Modules

| module | contents |
|---|---|
| `std::foldable` | the `Foldable` class and instances for `List` and `Maybe` |
| `std::result` | `Result a e`, `Ok`, `Err`, mapping, unwrapping, and conversions |
| `std::option` | helpers for the prelude `Maybe` type, including `to_result` |
| `std::list` | list construction, folds, mapping, filtering, zipping, ranges |
| `std::iter` | lazy `Iterator`/`Step`, ranges, repeats, mapping, filtering, `take`/`drop`, `collect` |
| `std::string` | trimming, splitting, joining, replacing, repeating, indexing |
| `std::path` | `dirname`, `basename`, `extension`, `stem`, `join`, and components |
| `std::io` | futures for sleeping, reading/writing files, and reading stdin |

`Result a e` is `Ok(a) | Err(e)`. `Iterator` values are lazy: `range`,
`repeat`, `map`, `filter`, and the other combinators produce another
iterator, while `collect` forces one into a `List`.

`std::io` operations are futures. Await them from an `async fun`:

```fern
import std::io

async fun copy(from : String, to : String) : Unit = {
    let text = await io.read_file(from);
    await io.write_file(to, text)
}
```

`import` binds the module name; `use` brings an item into scope. Use the
module qualification or an alias when two modules provide the same name:

```fern
import std::list
import std::iter

fun evens(n : Int) : List Int =
    list.filter(x -> x / 2 * 2 == x, list.range(0, n))

fun first_evens(n : Int) : List Int =
    iter.collect(iter.take(n, iter.filter(x -> x / 2 * 2 == x, iter.range(0, 100))))
```

For convenience, `use` can import individual functions or types:

```fern
import std::result
use std::result::Result
use std::result::Ok
use std::result::Err

fun ratio(ok : Bool) : Result Int String =
    if ok then Ok(1) else Err("not ok")
```

`use std::list::map as list_map` is useful when importing `std::iter::map`
or another `map` in the same file.
