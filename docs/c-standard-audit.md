# ISO C conformance audit (C99 through C2y)

Audit date: 2026-10-22

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
  operands, delimited escapes, bit-precise enums, and expression
  `static_assert` are implemented.
* `-pedantic-errors` rejects implicit `int` and implicit function
  declarations in C99 and later; the permissive default keeps the old
  warning-only behavior.
* `crates/glcc_frontend/src/tests/conformance.rs` carries table-driven
  language probes for C99 through C2y.

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

**Digraphs and trigraphs are absent.** The lexer explicitly excludes
digraphs from its punctuator set (`crates/glcc_frontend/src/lexer.rs:893`),
and no trigraph translation happens before lexing:

```c
int a<:1:> = { 1 };      /* error: expected `;`, got Punctuator(Less) */
int a??(1??) = { 1 };    /* error */
```

Digraphs are required by C99 and C23; trigraphs are required through
C17 and removed in C23.

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

**`_Alignas` only works after a type specifier.** It fails when it opens
a declaration, at both file and block scope:

```c
_Alignas(16) int x;              /* error: no type specifier */
int f(void){ _Alignas(16) int y; /* error */ }
```

These forms work:

```c
int _Alignas(16) x;
struct S { int _Alignas(8) c; };
int f(void){ int _Alignas(16) y; }
```

Root cause: `can_start_declaration` (`crates/glcc_frontend/src/parse.rs:2663`)
does not recognise `_Alignas`/`alignas`, so the leading spelling is read as
an implicit-`int` identifier. `parse_declaration_specifiers` already handles
`_Alignas` correctly once it is entered.

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

| Feature | Status |
| --- | --- |
| `bool`, `true`, `false` as keywords | Missing (only `<stdbool.h>` macros) |
| `nullptr`, `nullptr_t` | Missing |
| `constexpr` objects | Missing |
| `auto` type inference | Wrong (implicit `int`) |
| `typeof` | Works (GNU extension) |
| `typeof_unqual` | Missing |
| `static_assert` spelling | Missing (only `_Static_assert`) |
| `alignas` spelling | Missing as keyword (the C23 header defines no macro); `_Alignas` is partial |
| `alignof` spelling | Missing as keyword; `_Alignof` works |
| `thread_local` spelling | Works |
| `_BitInt(N)`, `wb` suffix | Missing |
| `[[attribute]]` syntax | Missing |
| `enum E : type` | Missing |
| `char8_t` | Works through `<uchar.h>` |
| `_Decimal32/64/128` | Missing (optional decimal floating types) |
| `unreachable()` | Not reachable: `<stddef.h>` fails in C23 mode |

Reproductions:

```c
bool b = true;                    /* error: expected `;`, got Identifier */
nullptr_t p = nullptr;            /* error: use of undeclared identifier 'nullptr' */
constexpr int n = 3;              /* error: expected `;`, got Keyword(Int) */
typeof_unqual(int) x;             /* error */
_Static_assert(1);                /* works */
static_assert(1);                 /* block scope: implicit function call; file scope: parse error */
[[nodiscard]] int f(void);        /* error: expected external declaration, got '[' */
enum E : unsigned char { A, B };  /* error: declaration does not declare anything */
_BitInt(17) x;                    /* error: no type specifier */
```

`static_assert` is especially dangerous at block scope: without
`<assert.h>`, `static_assert(0, "boom");` passes `-fsyntax-only` as an
implicit call to an undeclared function. A full compile then panics in
codegen (mismatched argument count) rather than reporting a failed
assertion. With `<assert.h>` it does fail correctly, because the header
maps `static_assert` to `_Static_assert`.

### Literals and tokens

| Feature | Status |
| --- | --- |
| Binary literals `0b1010` | Missing |
| Digit separators `1'000'000` | Missing |
| `u8'c'` character constants | Missing |
| `u8"..."` type `char8_t[]` | Wrong: it is `char[]` |
| `wb` bit-precise suffix | Missing |

`u8"x"[0]` selects `char` in `_Generic`, not `char8_t` (which is
`unsigned char` in C23).

### Declarations and statements

* Labels before declarations are rejected:
  `L: int x = 1;` fails with `expected expression`.
* Compound-literal storage classes are rejected:
  `((static int[]){1,2})[0]` fails.
* Enum fixed underlying types are rejected; `enum E : short { ... }`
  produces `declaration does not declare anything`.
* Attributes on enumerators (`A [[deprecated]]`) are rejected.
* `static_assert`/`alignas`/`alignof` are not keywords.

### Preprocessor

| Feature | Status |
| --- | --- |
| `#embed` | Unknown directive |
| `__VA_OPT__` | Explicitly unsupported error |
| `#elifdef`, `#elifndef` | Unknown directive |
| `#warning` | Silently ignored (`ControlLine::Warning => {}` in `glcc_pp/src/expand.rs:1514`) |
| `__has_c_attribute` | Always true, even for nonexistent attributes |
| `__has_builtin` | Always true |
| `__has_attribute` | Always true |
| `__has_embed` | Undefined, so evaluates to 0 |
| `__has_include` | Works |
| `#line` | Works |

The `__has_*` defaults come from `expand.rs:1044` (`_ => true`). This is
not merely cosmetic. For example, `<stdbit.h>` currently fails on `bool`
first, but once `bool` works it would select the `__builtin_stdc_*`
implementations because `__has_builtin` reports them as available.

### Headers

Headers that parse on their own in C23: `assert.h`, `complex.h`,
`ctype.h`, `errno.h`, `fenv.h`, `float.h`, `inttypes.h`, `iso646.h`,
`limits.h`, `locale.h`, `math.h`, `setjmp.h`, `signal.h`, `stdalign.h`,
`stdarg.h`, `stdbool.h`, `stdint.h`, `stdio.h`, `stdlib.h`,
`stdckdint.h`, `stdnoreturn.h`, `string.h`, `threads.h`, `time.h`,
`uchar.h`, `wchar.h`, `wctype.h`.

Broken in C23 mode:

* **`<stddef.h>`** — uses `nullptr` in `__stddef_nullptr_t.h:26`, so the
  header cannot be included. This also blocks `unreachable()`, `nullptr_t`
  and anything that includes it.
* **`<stdatomic.h>`** — includes `<stddef.h>`, so it fails on `nullptr`.
  It works in C11/C17.
* **`<stdbit.h>`** — uses `bool` at `stdbit.h:622`; `bool` is not a
  keyword, so parsing fails.
* **`<stdarg.h>`** — the header itself parses, but `va_start` expands to
  `__builtin_c23_va_start`, which is undeclared. Any variadic function
  definition fails.
* **`<tgmath.h>`** — fails in every mode (see C99).

`<stdckdint.h>` works, including overflow detection (signed overflow
wraps and reports `true`). `<uchar.h>` provides `char8_t`, but the
`u8"..."` literal still has type `char[]`.

## C2y

`-std=c2y` is not a recognised standard:

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

Ranked by conformance gained per unit of effort:

1. **Accept `-std=c2y`.** Add the variant to `Std`, give it a level and
   `__STDC_VERSION__`, and gate the C2y features behind it. Even an
   empty gate is better than rejecting the flag.
2. **Fix leading `_Alignas`.** Add `peek_alignas()` to
   `can_start_declaration`. This is a small parser change that repairs
   C11 and C23 alignment declarations, including `<stdalign.h>` users.
3. **Add the C23 keyword set.** `bool`, `true`, `false`, `nullptr`,
   `constexpr`, `static_assert`, `alignas`, `alignof`, `typeof_unqual`.
   `bool` unlocks `<stdbit.h>`; `<stdatomic.h>` additionally needs
   `nullptr` for `<stddef.h>`.
4. **Implement C23 `auto` inference.** Reject the current implicit-`int`
   interpretation; derive the type from the initializer.
5. **Implement `[[attributes]]` and make `__has_c_attribute` truthful.**
   Start with the seven standard attributes and namespace handling.
6. **Fix the C23 headers.** `__builtin_c23_va_start` for `<stdarg.h>`,
   `nullptr` for `<stddef.h>`, and `bool` for `<stdatomic.h>`/`<stdbit.h>`.
7. **Add C23 literals.** Binary constants, digit separators, `u8'c'`,
   and the correct `u8"..."` type.
8. **Add the C23 preprocessor features.** `#elifdef`/`#elifndef`,
   `#embed`, `__VA_OPT__`, and a real `#warning` diagnostic.
9. **Add `_BitInt(N)` and fixed enum underlying types.**
10. **Fix aggregate `_Atomic` lowering.** Loads of `_Atomic struct` must
    go through the address path rather than asking for a scalar Clif
    result.
11. **Add C99 digraphs (and trigraphs for pre-C23 modes).**
12. **Repair `<tgmath.h>`.** Either support `__attribute__((overloadable))`
    or provide a glcc-specific `tgmath.h` built on `_Generic`.

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
