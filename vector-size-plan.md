# Plan: GNU vector types (`vector_size`) — unblocking SSE headers on mingw

Status: plan only, not implemented. Written 2026-11-05; the builtin work it
builds on is committed (`ab4643a` check/ modularization + `__sync`,
`d770c19` `__builtin_ia32_*`).

## Context

Compiling `windows.h` (and SDL's `SDL_cpuinfo.h`) for
`x86_64-windows-w64-mingw` now fails in the **SSE headers**
(`xmmintrin.h` and friends): `(__m128){ 0.0f, ... }` parse-errors as
"typedef name used as a value". Everything scalar/system below that —
winnt.h, intrin-impl.h, bmi/adx/cet/cache/msr/rtm/uintr, and the whole
MMX set — already compiles (see `check/ia32.rs` / `lower/ia32.rs`).

`__m128` **is** a GNU vector type:
`typedef float __m128 __attribute__((__vector_size__(16), __may_alias__))`.
The frontend currently ignores `vector_size`, so `__m128` degrades to
`float` and the compound literal becomes nonsense. GCC's vector
extension is the same feature, so one implementation serves both.

**The backends are already done.** MIR has first-class vectors —
`Ty::Vector { elem, lanes }` (crates/glcc_mir/src/ty/mod.rs:66,
128-bit shapes only) — produced today by SLP/loopvec ("produced by the
vectorizer, never by a frontend" is the comment to delete). The
arithmetic insts are shape-generic: SLP emits plain
`Add`/`FAdd`/`FMul`/... with Vector-typed values (see
crates/glcc_mir/src/passes/slp/mod.rs:473), so no new vector opcodes
are needed for element-wise math. CLIF lowering maps the six shapes
(crates/glcc_lower_cl/src/ty/mod.rs:32: i8x16, i16x8, i32x4, i64x2,
f32x4, f64x2). `Splat` and `ExtractLane` exist for broadcast/extract.

**The one backend gap is explicit**: crates/glcc_lower_cl/src/abi/mod.rs:574
panics with "vector values do not cross function signatures".

## Working conventions

`cargo build` after every edit; test the owning package per change
(`cargo test -p glcc_frontend`, `-p glcc_lower_mir`, `-p glcc_lower_cl`);
small steps, always green. No workspace-wide `cargo fmt`. Statements
flow as `Result<T, CompileError>`; every AST `Ty` is spanless; keep
panic-message prefixes stable for `#[should_panic]`.

## Phase 1 — type plumbing (parse + check)

1. **`Ty::Vector`** in the FE `Ty` enum (crates/glcc_frontend/src/parse.rs:9150,
   alongside `Complex`): `Vector { elem: Box<Ty>, len: u32 }`.
   Mirror `Complex` everywhere a `Ty` variant is matched exhaustively
   (the compiler will enumerate: `unqualified`/`unqualified_owned`,
   `types_compatible`, `size_type`-style queries in layout.rs,
   dump.rs, sema.rs remapping, eval.rs, check.rs conversions).
   Sizes: `len * elem_size`; alignment: **the vector's total size up to
   16** (GCC gives `vector_size(8)` 8-byte and `vector_size(16)` 16-byte
   alignment — this is load-bearing for SSE ABI classification).
2. **`vector_size(N)` attribute**: add a `vector_size: Option<u32>` (or
   `(u32, Span)`) field to `GnuAttributes` (crates/glcc_frontend/src/parse.rs:7753)
   and a `GnuAttrFlags` bit if the flags enum fits; apply it where the
   attribute list merges into the declarator's type (the same place
   `aligned` applies). Reject non-arithmetic element types and sizes
   that are not a power of two ≤ 16 bytes with the element's rank
   (GCC's rule: size must be a multiple of the element size). Also
   accept `__may_alias__` by ignoring it (it's an aliasing-analysis
   hint; note that in a comment).
3. **No new nodes for value semantics yet** — vectors flow through the
   existing `CheckedExprKind` paths as an ordinary first-class type
   (like `Complex` does).

Verify: parse/check tests in `crates/glcc_frontend/src/tests/mod.rs`
(`check_with` pattern) for `typedef int v4 __attribute__((vector_size(16)));`,
sizeof == 16, alignof == 16, and a `vector_size(8)` variant at 8/8.

## Phase 2 — expression semantics (GCC's vector extension)

Per GCC's "Vector Extensions" docs:

1. **Compound literals**: `(T){...}` for a vector typedef. The
   cast-vs-compound-literal gate is `paren_starts_compound_literal`
   (crates/glcc_frontend/src/parse.rs:6718); the
   "typedef name used as a value" error comes from the cast path
   consuming `(__m128)` then tripping on `{`. With Phase 1's real type
   the existing literal machinery likely works if the type name
   resolves — test first, extend the disambiguation only if needed.
2. **Element-wise operators**: `+ - * /` and `& | ^` on two same-shaped
   vectors; one operand may be a scalar (GCC splats it, and requires it
   to be a *vector-size-wide* integer for mixed vector/scalar bitops —
   simplest: accept scalars and splat). Comparisons yield the same
   shape with 0/−1 lanes. These check into the existing
   `Binary`/`Assign` node kinds; the lowerer's binary lowering already
   works shape-generic because MIR ops are shape-generic.
3. **Subscript**: `v[i]` reads (rvalue only, like GCC) → an
   `ExtractLane`-style node; `v[i] = x` — GCC rejects subscript
   *writes* on vector types unless per-lane lvalues are modeled; follow
   GCC: error "subscripted value is neither array nor pointer" is wrong
   here, GCC allows reads only. (If a header needs writes, revisit.)
4. **Conversions**: no implicit vector↔vector conversions except
   same-shape; vector↔scalar only via explicit splat semantics above.
   `__builtin_shufflevector` can be deferred — mingw headers use
   `_mm_shuffle_ps`-style builtins, not shufflevector (grep to confirm
   before skipping).
5. **`__builtin_shufflevector`, `__builtin_convertvector`**: optional;
   add as `check/gnu.rs` rows + lowering only if a header references
   them.

Verify: JIT exec tests in `crates/glcc_lower_mir/src/lower/tests.rs`
(`run_main` pattern) — element-wise add/mul on `int` and `float`
vectors, comparison masks, subscript reads, compound literals, and
vector locals copied by assignment.

## Phase 3 — ABI (the backend gap)

Teach `glcc_lower_cl::abi` (crates/glcc_lower_cl/src/abi/mod.rs, the
panic is at :574) to classify vectors:

- **SysV**: `__m128`-shaped (16 bytes, SSE class) → passed in
  `xmm0`–`xmm7`, returned in `xmm0`; `vector_size(8)` (MMX class) →
  passed like INTEGER eightbytes (the x86-64 psABI passes MMX class in
  integer registers). Follow how the file classifies aggregates today
  (eightbyte walk) — a 16-byte vector is one SSE eightbyte.
- **Win64**: vectors ≤ 8 bytes by value in integer regs; `__m128` in
  `xmm` per the vector-passing convention (Win64 passes `__m128`
  by pointer on old ABI rules — check which flavor mingw gcc uses:
  modern mingw-w64 passes `__m128` in xmm when `__AVX__`-era ABI is
  assumed, otherwise by reference. **Verify against mingw gcc's actual
  behavior** (`x86_64-w64-mingw32-gcc -S`) before implementing; the
  safe default mirroring gcc's default is by-pointer for `__m128`).
- `mir_align` (crates/glcc_lower_cl/src/abi or ty) already handles
  vector alignment via elem (:609).
- Stack slots/variadic: `__m128` varargs on Win64 are passed by
  pointer — same code path as aggregates.

Verify: `-p glcc_lower_cl` tests if any cover ABI; end-to-end
`run_main` with a function taking/returning `__m128`-shaped vectors,
plus cross-TU linking via the mingw smoke path.

## Phase 4 — the header-driven intrinsic rows

With types in place, the SSE1/SSE2 intrinsics xmmintrin/emmintrin
*check* are each one table row in `check/ia32.rs` + one lowering arm:

- Prefer real MIR: `_mm_add_ps`/`_mm_sub_ps`/`_mm_mul_ps`/`_mm_div_ps`
  → `FAdd`/... on `v4f32` (shape-generic, zero new code);
  `_mm_and_si128`/`or`/`xor` → `And`/`Or`/`Xor` on `i64x2`-shaped;
  `_mm_set1_ps` → `Splat`; `_mm_cvtsi128_si32`/`_mm_cvtsi32_si128` →
  `ExtractLane`/lane insert or `Bitcast` through an i32x4 slot.
- Everything else (shuffles, `_mm_sqrt_ps` needs `Sqrt`-at-vector —
  check whether MIR's `Sqrt` is shape-generic) → inline-asm rows on
  `xmm` registers exactly like the MMX pairs in `lower/ia32.rs`
  (literal `%xmm0`/`%xmm1` staging over gpr `"r"` operands, or pinned
  `"x"`-class if the asm wrapper grows one).
- Iterate the same way the scalar ia32 work did: compile the mingw
  smoke test (`windows.h` + `SDL.h`), collect
  "unsupported builtin \`__builtin_ia32_X\`" errors, add rows until
  clean. Keep `glcc_pp`'s `has_builtin` closed table in step
  (crates/glcc_pp/src/expand.rs — the `ia32_builtin` helper).

## Risks / notes

- **Vector constants in static initializers**: `_mm_set1_ps` results
  and compound literals at file scope lower through `lower/data.rs` —
  may need vector global-data support (16-byte constants). Defer if
  headers only use them in local inline functions (likely).
- **MIR shapes are 128-bit only** (`Ty::vector` panics otherwise):
  `__m256`/`__m512` stay unsupported — acceptable; mingw's
  `*avx*.h` headers are `#ifdef __AVX__`-gated and glcc does not
  define those macros.
- **`__m64`**: already compiles as an 8-byte integer because
  `vector_size` was ignored; after Phase 1 it becomes a real
  `Ty::Vector` — the MMX asm rows in `lower/ia32.rs` take
  `Ty::LongLong` params and will need their signatures widened to the
  new vector type (or the FE keeps `__m64` usable either way). Grep
  `mmintrin` expectations again after Phase 1.
- **FE `Vector` must not panic the vectorizer**: SLP/loopvec assume
  they are the only producers of `Ty::Vector`; a quick audit of
  `passes/slp` and `passes/loopvec` assumptions (e.g. is_mir_scalar
  checks) is warranted where frontend vectors could appear in their
  input.
- `fern`/`glcc_cxx` should be untouched; CI runs the full workspace.

## Suggested commit sequence

1. `feat(fe): GNU vector types` — Phase 1 (+ Phase 2's compound
   literals if green together).
2. `feat(fe): vector expression semantics` — Phase 2.
3. `feat(lower): vector ABI` — Phase 3.
4. `feat(fe): SSE intrinsics` — Phase 4, possibly several commits as
   the header set clears (one per header family, like the ia32 work).

End-to-end acceptance: the mingw smoke test
(`glcc --target=x86_64-windows-w64-mingw -I/usr/x86_64-w64-mingw32/include/
-fms-extension tri.c`) gets past xmmintrin/emmintrin, and the user's
triangle example compiles.
