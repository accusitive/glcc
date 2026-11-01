# Lua 5.1 backend (experimental)

`crates/glcc_lower_lua` lowers MIR to a single textual Lua 5.1 script.
It is a fun portability target, not a replacement for `glcc_lower_cl`:
there is no native ABI, and C memory is emulated.

```sh
glcc --emit-lua -O2 hello.c -o hello.lua
lua5.1 hello.lua
```

The emitted script defines `main` and calls it at the end, forwarding
`argc`/`argv` when the signature takes them.

## Model

- **Control flow** becomes a `pc` state machine (`while true do if pc == ...`).
  Lua 5.1 has no `goto`, but arbitrary branches are still direct. Blocks
  unreachable from the entry block are not emitted.
- **Functions** are installed into an `FN` table (`FN[7] = function(...)`), not
  declared as chunk-level `local function`s. A translation unit can therefore
  contain more than Lua's 200-local functions.
- **SSA values** live in a per-call table, so a function is not limited by
  Lua's 200 live locals.
- **Memory** is a sparse table of bytes. Pointers are ordinary Lua numbers.
  Globals, string literals, and aggregate constants get static addresses;
  stack frames are carved out of a bump-allocated address range.
- **Aggregates** use the same convention as the Cranelift backend: a value
  is the address of its storage. Loads and inserts copy into frame slots.
  Aggregate returns use a hidden out-pointer argument.
- **Integers** are Lua doubles normalized to their MIR width. Values up to
  2^53 are exact; wider 64-bit values lose low bits.
- **Floats** are Lua doubles. `f32` operations are rounded back to single
  precision, and memory loads/stores encode/decode IEEE bytes manually.
- **Externs** are mapped onto a small Lua runtime: `puts`, `printf`,
  `malloc`, `memcpy`, `strlen`, `exit`, and friends. `getchar` uses the
  host's line-oriented `read` when it exists (draining one byte at a time)
  and falls back to `io.read(1)`. Unknown externs fail at run time with a
  clear message.
- **stdio files** (`fopen`, `fclose`, `fread`, `fwrite`, `fgets`, `fputs`,
  `fgetc`, `fputc`, `fseek`, `ftell`, `feof`, `remove`, `rename`, ...) use
  a `FILE *` as a small integer handle. Files open through `fs.open` under
  ComputerCraft and `io.open` on stock Lua; `stdin`, `stdout`, and `stderr`
  are preinitialized to handles 0, 1, and 2.
- **Variadic arguments** are captured as a Lua table on entry to a variadic
  function. A `va_list` object stores an id for an immutable cursor into
  that table, so `va_copy` snapshots the position and advancing one reader
  never moves another. Scalars are stored by value; aggregates are stored
  as their address, which stays live in the caller's frame.

## Known limits

- No inline assembly, `f80`/`f128`, or C++ exceptions.
- 64-bit integer arithmetic and 8-byte loads beyond 2^53 are approximate.
- `printf` supports the common conversions but ignores width/precision.
- stdio covers files and character/line input; `scanf`/`fscanf` and
  `tmpfile` are not mapped yet.
- Address space is linear and unbounded; there is no real stack guard.
