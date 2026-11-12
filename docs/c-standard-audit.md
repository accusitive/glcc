# ISO C conformance audit (C99 through C2y)

Audit date: 2026-10-22 (tables re-verified 2026-11-12; see "Post-fix
status" and the current-status notes in each section)

Compiler: `target/debug/glcc` 0.1.0 (Cranelift backend), built from
`025098c` plus uncommitted MIR/lowering changes in
`crates/glcc_lower_mir/src/lower.rs`, `crates/glcc_mir/src/optimize.rs`,
`crates/glcc_mir/src/passes/mod.rs` and `crates/glcc_mir/src/passes/unroll.rs`.

System headers: clang 22.1.8 resource headers (`/usr/lib/clang/22/include`)
and glibc headers under `/usr/include`. Host: x86-64 Linux.

The frontend and preprocessor were unmodified during the audit, so the
language and preprocessor findings do not depend on the dirty MIR changes.
Codegen observations (the aggregate-`_Atomic` panic) can be affected by them.

## Verdict at a glance

| Standard | Probes for new features | Accepted in matching mode | Verdict |
| --- | ---: | ---: | --- |
| C99 | 22 | 21 (95%) | Core language essentially complete. Required digraphs and a working `<tgmath.h>` are missing; trigraphs and optional `_Imaginary` are also absent. |
| C11 | 15 | 14 (93%) | Language features present, with one placement bug (`_Alignas`) and one codegen hole (aggregate `_Atomic`). |
| C17 | 2 | 2 (100%) | Adds no new language features; inherits every C11 gap. |
| C23 | 45 | 10 (22%) | Only about one fifth of probes are accepted, and several of those are GNU extensions or false positives. With correct semantics the figure is closer to 10–15%. |
| C2y | 9 | 0 (0%) | `-std=c2y` is rejected outright. A few C2y-adjacent features exist as GNU extensions. |

"Accepted" means the probe compiles with `-fsyntax-only` in the mode
named by the standard. It does not always mean the construct has the
standard semantics; the false positives are listed below.

**Bottom line:** glcc is a solid C99/C11 compiler with a few notable
holes, and a C23 compiler in name only. C17 comes for free because it adds
no language features. C2y is not selectable at all.

## Post-fix status (2026-10-22)

The table and findings above are the pre-fix audit snapshot. The fixes in
the conformance plan have since landed:

* C2y is a selectable dialect (`-std=c2y`) with `__STDC_VERSION__`
  `202400L`.
* C23 declarations, literals, attributes, `_BitInt`, `constexpr`, `auto`,
  `nullptr`, fixed enums, and the C23 headers are implemented.
* C23 preprocessor features (`#warning`, `#elifdef`/`#elifndef`,
  `__VA_OPT__`, `#embed`, truthful `__has_*`) are implemented.
* Digraphs and trigraphs work; `<tgmath.h>` is glcc-owned and works in
  C99 and later.
* C2y `_Countof`, selection declarations, named loops, `_Generic` type
  operands, delimited escapes, bit-precise enums, expression
  `static_assert`, `0o` octal literals, and complex `++`/`--` are
  implemented.
* GNU `__auto_type` is accepted in GNU modes and in C2y.
* `-pedantic-errors` rejects implicit `int` and implicit function
  declarations in C99 and later; the permissive default keeps the old
  warning-only behavior.
* `crates/glcc_frontend/src/tests/conformance.rs` carries table-driven
  language probes for C99 through C2y.

Re-verified 2026-11-12 against `crates/glcc_frontend` plus this session's
change: `nullptr_t` and `char8_t` are now C23 type keywords in their own
right (`nullptr_t` names `Ty::Nullptr`; `char8_t` names `unsigned char`,
matching the type `u8"..."` strings already carry), usable in
declarations, casts, `sizeof`/`_Alignof`/`_Generic`, and still plain
identifiers (typedef-able) in C17 and earlier.

Known remaining gaps are deliberate or narrow: N3658 duplicate-label
reuse is not in the C2y working draft yet; `_Imaginary` is optional Annex
G and stays diagnosed as unsupported; `[[gnu::packed]]` does not pack;
`deprecated` fires on identifier uses; `maybe_unused` is not enforced;
bare `(thread_local T){...}` is accepted more liberally than GCC; TLS
compound literals are not JIT-tested; and long-double complex foreign
returns remain a documented MIR/CL ABI gap.

## Method

* **Feature matrix.** 93 small probes (`/tmp/opencode/cstd-audit/run.sh`)
  were compiled with `-std=c89`, `c99`, `c11`, `c17`, `c23`, and `c2y`.
  The generated table is `/tmp/opencode/cstd-audit/results.tsv`.
* **Semantic/runtime checks.** Selected probes were compiled and run to
  check values, `sizeof`, alignment, `_Generic` selection and overflow
  behaviour (`sem.sh`, `sem2.sh`, `t/`).
* **Header audit.** Every standard header was included on its own under
  each mode, then the important ones were exercised in real code.
* **Preprocessor probes.** Directive and `__has_*` behaviour was checked
  directly (`pp/`, `pp2/`).
* **Existing suites.** `c-testsuite` passes 220/220 on x86-64. The GCC
  torture baseline in `docs/inline-asm-plan.md` is 1441/1514; the
  dirty-tree run was not used as a conformance signal.

## C99

### Working

All 21 non-trigraph probes pass, including:

* `//` comments, `long long`, `inline`, `restrict`
* variable-length arrays, flexible array members
* designated initializers, compound literals
* hexadecimal floating constants
* variadic macros and empty macro arguments
* `_Pragma`, `__func__`, universal character names
* trailing comma in enumerators
* declarations mixed with statements, `for`-declarations
* `static` array parameters, repeated qualifiers
* `_Complex` and `<complex.h>`

Runtime checks confirmed VLA sizing, designated initialization,
compound-literal evaluation, `long long` arithmetic and complex
multiplication.

### Gaps

**Digraphs and trigraphs: fixed.** Both now lex and preprocess
(`int a<:1:> = { 1 };`, `%:include <stdio.h>`, `??(` all verified
2026-11-12).

**`<tgmath.h>`: fixed.** glcc ships its own `_Generic`-based
`include/tgmath.h` that works in C99 and later.

**`_Imaginary` is rejected.**

```c
_Imaginary float x;      /* error: `_Imaginary` is not supported */
```

`_Complex` works, so this only affects the optional imaginary types.
Annex G is optional; glcc deliberately keeps the clear diagnostic rather
than implementing a partial imaginary-type library.

**`<tgmath.h>` does not work in any mode.** The clang header depends on
`__attribute__((__overloadable__))`, which glcc parses as a conflicting
redeclaration:

```c
#include <tgmath.h>
/* error: conflicting redeclaration at tgmath.h:36 */
```

## C11

### Working

`_Static_assert` (file and block scope), `_Noreturn`, `_Thread_local`,
`_Alignof`, `_Generic` (including qualified controlling expressions),
anonymous structs/unions, `_Bool` bit-fields, `<stdatomic.h>` for scalar
atomics, `<stdalign.h>`, `<stdnoreturn.h>`, `<uchar.h>`,
`char16_t`/`char32_t`, and `<threads.h>` (header).

Runtime checks confirmed `_Generic` selection, `_Thread_local` storage,
`_Static_assert`, and atomic load/store/add.

### Gaps

**Leading `_Alignas`: fixed.** `_Alignas(16) int x;` and
`_Alignas(16) int y;` at block scope are accepted (`can_start_declaration`
recognises `_Alignas`/`alignas`).

**Loading an atomic aggregate panics in codegen.** Scalar atomics are fine;
`atomic_load` on an `_Atomic struct` crashes:

```c
#include <stdatomic.h>
struct S { int a; };
_Atomic struct S s;
int main(void){ struct S x = atomic_load(&s); return x.a; }
```

```
panicked at crates/glcc_lower_cl/src/lower.rs:2570:
result needs a scalar Clif type
```

Direct member access (`s.a`) and copy assignment happen to work; the
generic `atomic_load` path does not. This is a real C11 conformance hole,
not just a missed diagnostic.

## C17

C17 (C18) is C11 plus editorial fixes; the only observable language
difference is `__STDC_VERSION__ == 201710L`, which is correct. Both C17
probes pass, but C17 inherits all C11 defects above.

## C23

Raw probe result: 10 of 45 accepted. Several accepted probes are not real
C23 support:

* `auto x = 1;` compiles, but `auto` is treated as a storage class and the
  missing type becomes C89 implicit `int`. `auto x = 2.5;` gives an `int`
  with `sizeof == 4`, not a `double`.
* `typeof` works because it is a GNU extension.
* `thread_local` works as an alias for `_Thread_local`.
* `{}` empty initializers work as a GNU extension.
* `#warning` is accepted but silently ignored.
* `__has_c_attribute(x)` is always true, including for nonexistent
  attributes.
* `#line` and `__has_include` genuinely work but are not C23-specific.

### Keywords and syntax

Statuses re-verified 2026-11-12.

| Feature | Status |
| --- | --- |
| `bool`, `true`, `false` as keywords | Works |
| `nullptr` | Works (`Ty::Nullptr`) |
| `nullptr_t` | Works as a type keyword (names `Ty::Nullptr`); plain identifier in C17 and earlier |
| `constexpr` objects | Works |
| `auto` type inference | Works (deduced from the initializer) |
| `typeof` | Works (GNU extension) |
| `typeof_unqual` | Works |
| `static_assert` spelling | Works (file and block scope) |
| `alignas` spelling | Works, including as the leading specifier |
| `alignof` spelling | Works |
| `thread_local` spelling | Works |
| `_BitInt(N)`, `wb` suffix | Works |
| `[[attribute]]` syntax | Works |
| `enum E : type` | Works |
| `char8_t` | Works as a C23 type keyword (maps to `unsigned char`, the type `u8"..."` already carries); `<uchar.h>` typedef spelling still works pre-C23 |
| `_Decimal32/64/128` | Missing (optional decimal floating types) |
| `unreachable()` | Works via `<stddef.h>` / `__builtin_unreachable` |

### Literals and tokens

| Feature | Status |
| --- | --- |
| Binary literals `0b1010` | Works |
| Digit separators `1'000'000` | Works |
| `u8'c'` character constants | Works (typed `unsigned char`) |
| `u8"..."` type `char8_t[]` | Works (typed `unsigned char`, which is what `char8_t` names here) |
| `wb` bit-precise suffix | Works |

### Declarations and statements

* Labels before declarations work: `L: int x = 1;` is accepted.
* Compound-literal storage classes work: `((static int[]){1,2})[0]`.
* Enum fixed underlying types work: `enum E : unsigned char { A, B };`.
* Attributes on enumerators (`A [[deprecated]]`) are accepted.
* `static_assert`/`alignas`/`alignof` are keywords in C23 mode.

### Preprocessor

| Feature | Status |
| --- | --- |
| `#embed` | Works (verified: embeds file bytes) |
| `__VA_OPT__` | Works |
| `#elifdef`, `#elifndef` | Work |
| `#warning` | Emits a real warning diagnostic |
| `__has_c_attribute` | Truthful (false for nonexistent attributes) |
| `__has_builtin` | Truthful |
| `__has_attribute` | Truthful |
| `__has_embed` | Implemented |
| `__has_include` | Works |
| `#line` | Works |

### Headers

Headers that parse on their own in C23: `assert.h`, `complex.h`,
`ctype.h`, `errno.h`, `fenv.h`, `float.h`, `inttypes.h`, `iso646.h`,
`limits.h`, `locale.h`, `math.h`, `setjmp.h`, `signal.h`, `stdalign.h`,
`stdarg.h`, `stdbool.h`, `stdint.h`, `stdio.h`, `stdlib.h`,
`stdckdint.h`, `stdnoreturn.h`, `string.h`, `threads.h`, `time.h`,
`uchar.h`, `wchar.h`, `wctype.h`.

Broken in C23 mode (as of 2026-10-22): all fixed. Re-verified 2026-11-12:
`<stddef.h>`, `<stdatomic.h>`, `<stdbit.h>`, `<stdarg.h>` (variadic
definitions compile and run end-to-end) and `<tgmath.h>` all parse and
work in C23 mode.

`<stdckdint.h>` works, including overflow detection (signed overflow
wraps and reports `true`). `<uchar.h>` provides `char8_t`, and the
`u8"..."` literal now has the type `char8_t` names (`unsigned char`).

## C2y

`-std=c2y` is now a recognised dialect (see "Post-fix status"): `_Countof`,
selection declarations, named loops, `_Generic` type operands, delimited
escapes, bit-precise enums, expression `static_assert`, `0o` octal
literals, and complex `++`/`--` are implemented behind the C2y gate.

Historical audit findings (2026-10-22, when `-std=c2y` was rejected):

```
glcc: error: invalid argument to '-std=' (unrecognized standard 'c2y')
```

`Std` stops at `C23` (`crates/glcc_frontend/src/lang.rs:18`). There is no
C2y feature gate, so no C2y probe can pass in C2y mode.

Some C2y-adjacent features are available as GNU extensions in the older
modes:

* case ranges (`case 1 ... 5`)
* `__COUNTER__`
* complex literal suffix (`2.0i`)
* `__STDC_VERSION__ >= 202311L` (not a C2y feature, just the C23 value)

`_Countof` is not implemented despite the probe appearing to pass: the
frontend treats `_Countof(a)` as an implicit function call returning
`int`, so the probe only exercises the permissive implicit-declaration
path.

Absent:

* `if (int x = expr)` declarations
* named loops (`break outer`)
* type operands in `_Generic`
* delimited escape sequences
* implicit `constexpr` `const int` semantics
* bit-precise enums
* expression `static_assert`

## Cross-cutting findings

### Dialect gating is permissive

`-std=c89` accepts C99 and C11 constructs; `-std=c99` and `-std=c23`
accept C89 implicit `int` and implicit function declarations (with a
warning). There is no `-pedantic-errors` equivalent that makes a
constraint violation fatal. For example:

```c
int main(void){ return foo(); }   /* -std=c23: warning only, exit 0 */
```

This means the compiler cannot currently be used as a conformance checker
even where the feature is implemented.

### `__STDC_VERSION__` is correct

| Mode | Value |
| --- | --- |
| `c89` | undefined |
| `c99` | `199901L` |
| `c11` | `201112L` |
| `c17` | `201710L` |
| `c23` | `202311L` |

`__STRICT_ANSI__` is defined for `-std=c*`, but GNU extensions remain
enabled, so it does not isolate strict ISO behaviour.

Because `-std=c23` advertises `202311L`, system headers legitimately
assume C23 keywords and builtins. That is why missing features break
headers (`bool` in `<stdbit.h>`, `nullptr` in `<stddef.h>`,
`__builtin_c23_va_start` in `<stdarg.h>`) instead of merely being
unavailable in user code.

### Codegen hole

The only runtime crash found was the aggregate `_Atomic` load described
under C11. The panic is in the Cranelift lowering path, not the frontend.

## Highest-impact work

Items 1–9 and 11–12 of the original list are done (see "Post-fix
status" and the section updates above). Still open, ranked:

1. **Fix aggregate `_Atomic` lowering.** Loads of `_Atomic struct` must
   go through the address path rather than asking for a scalar Clif
   result. This is the last known C11 conformance hole.
2. **`_Decimal32/64/128`** — optional in C23; large effort, low value.
3. **Attribute fidelity.** `[[gnu::packed]]` does not pack;
   `deprecated` fires on identifier uses; `maybe_unused` is not
   enforced.

## Reproduction

The audit-time scripts are in `/tmp/opencode/cstd-audit` (not checked
into the repository):

```sh
cd /tmp/opencode/cstd-audit
./run.sh                 # 93-probe matrix -> results.tsv
./sem.sh                 # runtime checks
./sem2.sh                # additional runtime checks
```

Single-probe example:

```sh
target/debug/glcc -std=c23 -fsyntax-only probe.c
```

Header check:

```sh
printf '#include <%s>\nint probe;\n' stddef.h > /tmp/h.c
target/debug/glcc -std=c23 -fsyntax-only /tmp/h.c
```

## Caveats

* The audit is language- and header-focused. It does not claim runtime
  conformance for every library function (for example Annex F environment
  behaviour, `<fenv.h>` rounding modes, or `<threads.h>` execution).
* System clang headers are used for `<stddef.h>`, `<stdarg.h>`,
  `<stdatomic.h>`, `<stdbit.h>`, `<tgmath.h>` and others. Some failures
  are therefore a mismatch between glcc's feature set and clang's header
  feature detection, but they still reflect real missing language
  features (notably `nullptr`, `bool`, and `__builtin_c23_va_start`).
* C2y is still evolving; the feature list above reflects the current
  draft vocabulary, not a published standard.
* Codegen results may change with the uncommitted MIR/lowering work in
  the tree at audit time; frontend and preprocessor findings are
  independent of it.
