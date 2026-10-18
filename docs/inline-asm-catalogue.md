# Inline asm in the gcc-c-torture failures — catalogue and feasibility

65 of the failing `execute/` tests contain `asm`/`__asm__`/`__asm`. This is a
catalogue of what that asm actually does, plus what a glcc implementation
would have to provide. The short version: **64 of the 65 templates are
empty** (`""`); the tests use asm purely as a semantic annotation on the
optimizer, not as machine code. The rustc_codegen_cranelift "hoist to a
generated `.s`" machinery is not needed for this set.

> Status (2026-10-16): the implementation went further than this catalogue
> requires — real templates run through generated out-of-line wrappers and
> `asm goto` works (`docs/inline-asm-plan.md`, `docs/inline-asm.md`). The
> survey below is kept as the per-test analysis it is; where it says
> "reject non-empty templates / `asm goto`", that describes the P0 scope
> only.

## Inventory

Every asm statement across all 65 tests:

- **64 of 65 tests use only empty templates (`""`).** The single exception is
  990413-2 (`fpatan`/`fsqrt`).
- Constraint alphabet actually used: `r`, `g`, `m`, `mr`, `+r`, `+g`, `+m`,
  `=r`, `=g`, `=m`, ties `"0"`, clobbers `"memory"` (and `"st(1)"` in the x87
  test).
- **No** `asm goto`, no named-register constraints (`"eax"` etc.), no
  immediates, no labels, no multi-instruction templates.
- One occurrence is not a statement at all: `register void *ptr asm ("rbx");`
  in pr51447.

Bucket sizes (one bucket per test, most demanding shape wins):

| bucket | tests |
|---|---|
| A. pure barriers | 27 |
| B. operand barriers (inputs) | 14 |
| C. value pinning / ties | 18 |
| D. memory operands | 4 |
| E. real instructions | 1 |
| F. register variable | 1 |

## Buckets by purpose

### A. Pure optimization barriers — 27 tests

`asm [volatile] ("")`, optionally with a `"memory"` clobber. Purpose: stop the
optimizer from folding/merging/eliminating the surrounding code. Nothing is
computed; operands are absent.

20050203-1, 20061031-1, 20071211-1, 20080122-1, bitfld-5, pr44852, pr46309,
pr47925, pr49218, pr51581-1, pr51581-2, pr51877, pr56205, pr56982,
pr57344-1..4, pr58277-1, pr58419, pr63641, pr69320-2, pr69691, pr78438,
pr79354, pr79737-2, pr84478.

### B. Operand barriers (inputs only) — 14 tests

Empty template with `"r"`/`"g"`/`"m"`/`"mr"` inputs: the operands must be
evaluated and treated as opaque so the optimizer cannot constant-fold the
surrounding expressions.

20001009-2 (`"r"(c)`), 960312-1 (`"r"(t2),"r"(t3)`), pr40022, pr51933,
pr53645, pr53645-2, pr56866, pr65648, pr68328, pr78726, pr80421, pr81588,
pr82954, pr85756.

### C. Value pinning / opaque identity — 18 tests

Read-write (`+r`, `+g`) or tied (`"=r"(x) : "0"(y)`) outputs. Semantically the
asm is an identity (or a tied copy, e.g. `i = (int)x`), but the value must
appear to come from an opaque operation: it may not be folded, CSE'd, or
evaluated twice.

20020107-1, 20030222-1, 20071220-1, 20071220-2, 990130-1 (`+r(*bar())` —
deref must happen exactly once), pr38533 (314 sites), pr41239, pr43385,
pr43560, pr45695, pr49279, pr49390, pr52286, pr65053-1, pr65053-2, pr65956,
pr84524, pr88904.

### D. Memory operands — 4 tests

`"=m"`/`"+m"` outputs: the asm may read/write the named object; the compiler
must assume the memory is clobbered.

20061220-1 (`=m`, `m`, `mr`), pr40657 (`"=m"(x)` untied), pr85156
(`"+m"(x), "+m"(y)`), stkalign (`"=g"(x), "+m"(s) : "0"(&x)`).

### E. Real instructions — 1 test

990413-2: x87 `fpatan`/`fsqrt` with `"=t"`/`"u"`/`"st(1)"` constraints,
testing the x87 register stack. The only case in the whole set that needs
actual machine code.

### F. Not an asm statement — 1 test

pr51447: `register void *ptr asm ("rbx");` — a global register variable,
requiring the value to live in a fixed register across a nonlocal goto. (Also
blocked by nested functions.)

## What a no-code implementation would unlock

I modelled the semantics of an empty-template asm (inputs evaluated, `+`
identity, `"0"` ties as assignments, untied outputs undefined, clobbers
ignored) by rewriting every statement and running the suite with glcc as it
is today:

- **57 / 65 pass** with the asm semantics alone.
- The 8 exceptions:

| test | why it still fails |
|---|---|
| 990413-2 | tie model gives the wrong value; needs real x87 `fpatan` (bucket E) |
| bitfld-5 | **independent bug**: `(s.b-8)+8` on a 40-bit bitfield is computed in 64-bit instead of 40-bit precision |
| pr85156 | **independent bug**: `__builtin_expect`'s second argument (`z++`) is not evaluated |
| 20061220-1, pr51447 | nested functions (out of scope) |
| pr53645, pr53645-2 | GCC vector extensions (out of scope) |
| pr49218 | i128 `udiv` (Cranelift gap) |

So asm support alone would take the torture score from 1384 to ~1441, and the
two independent bug fixes would bring 59 of the 65 within reach. The
remaining 6 need features already scoped out.

## Design implications

### What the implementation actually needs

1. Parse `asm`/`__asm__`/`__asm` statements with optional `volatile` and the
   four colon-separated lists (template, outputs, inputs, clobbers). Reject
   `asm goto` and non-empty templates for now with a clear error.
2. Check the operand expressions; outputs must be lvalues.
3. Lower with these semantics:
   - inputs: evaluate (side effects happen);
   - `"+X"(v)`: identity, but opaque;
   - `"=X"(v) : "0"(w)`: `v = (converted) w`;
   - `"=X"(v)` untied: value undefined (anything is conforming);
   - `"m"`/`"=m"`/`"+m"`: evaluate the address, mark the object clobbered;
   - `volatile`, and any asm without outputs, must not be removed;
   - a `"memory"` clobber (and, for honesty, every asm) is an optimization
     barrier.
4. Do **not** emit machine code.

Options for step 3's representation:

- **Desugar in the C→MIR lowerer** to the copies above plus a barrier. The
  barrier can be a call to a tiny runtime stub (`ret`) — calls are already
  opaque to every MIR pass — or a dedicated MIR op.
- **A dedicated MIR `InlineAsm` instruction** that passes treat like a call
  (no folding, no DCE when volatile, memory clobbered) and lowering emits
  nothing. Most faithful; costs the usual plumbing through verify/display/
  passes.
- **Pure desugar with no barrier.** Empirically enough for the current
  pipeline (which performs no aggressive motion), but unfaithful for user
  code that relies on `asm("" ::: "memory")` as a barrier.

Recommended: the dedicated op (or desugar + stub call), so the barrier
semantics are real rather than accidental.

### When the `.s`-hoisting idea *would* be needed

For non-empty templates, `asm goto`, stack manipulation, or register
variables. If it is ever built, the mechanical side is easy for glcc (it
already produces objects and drives the linker; a generated `.s` assembled
with `cc -c` slots straight into the link). The hard part is the operand
protocol: constraints must be translated into locations inside the generated
wrapper (a generated prologue moves ABI arguments into the registers
substituted for `%0`, `%1`; `"m"`/`"g"` pass addresses; ties are copies;
clobbers constrain scratch-register choice). That is a small per-ISA
constraint translator, not impossible, but it is real work — and this
catalogue shows it would buy exactly one test.

The "naked function + globals + jump back" variant is not needed for any of
this: a normal call already passes inputs and outputs, and the "jump back" is
just `ret`. Globals only help for many-output asm or `asm goto`; the latter
needs the tail-jump trick (pop the return address, then jump to a label
address passed in), which is where the approach genuinely gets messy.

## Two independent bugs found while cataloguing

- **bitfld-5**: for `struct s { unsigned long long a:2, b:40, c:22; }`,
  `((unsigned long long)(s.b - 8)) + 8` must be computed with 40-bit
  precision (`2 - 8` wraps at 2^40, giving `0x10000000002`), but glcc
  computes it in 64-bit and gets `2`. GCC's bitfield arithmetic happens at
  the field's declared precision.
- **pr85156**: `__builtin_expect(cond, z++)` must evaluate `z++`; glcc drops
  it. Reproduced standalone: gcc returns 11, glcc returns 10.

## Appendix: per-test detail

Bucket key: A barrier · B inputs · C pinning/tie · D memory · E real asm ·
F register variable. "model" = result with modelled empty-template semantics.

| test | bucket | asm shape | purpose | model |
|---|---|---|---|---|
| 20001009-2 | B | `""::"r"(c)` | keep sign-extended `c` in a register across the loop | PASS |
| 20020107-1 | C | `"=r"(tmp) : "0"(tmp)` | keep pointer opaque so `foo(buf)-1-buf` isn't folded | PASS |
| 20030222-1 | C | `"=r"(i) : "0"(x)` | low half of `long long x` via tied register | PASS |
| 20050203-1 | A | `asm("")` | keep noinline `bar` a real call | PASS |
| 20061031-1 | A | `asm volatile("")` ×2 | scheduling barriers around a call and a load | PASS |
| 20061220-1 | D | `=m/m/mr/r/tied` ×5 | every operand class on a nested-function capture | COMPILE (nested) |
| 20071211-1 | A | `"memory"` | barrier before `++sv.f2` bitfield wrap | PASS |
| 20071220-1 | C | `"+r"(lab)` | hide label address from the optimizer | PASS |
| 20071220-2 | C | `"+r"(lab)` | same, without computed goto | PASS |
| 20080122-1 | A | `asm("")` ×5 | barriers around byte-swap stores | PASS |
| 960312-1 | B | `"r"(t2),"r"(t3)` | keep values live across struct stores | PASS |
| 990130-1 | C | `"+r"(*bar())` | deref once, in a register | PASS |
| 990413-2 | E | `fpatan`, `fsqrt` | x87 register-stack semantics | FAIL (needs code) |
| bitfld-5 | A | `asm("")` ×2 | keep bitfield expression runtime-evaluated | FAIL (bitfield bug) |
| pr38533 | C | `"=r"(f) : "0"(0)` ×314 | combine-pass stress | PASS |
| pr40022 | B | `"g"(x)` + memory | alias-sensitive barrier | PASS |
| pr40657 | D | `"=m"(x)` | force a framed call | PASS |
| pr41239 | C | `+r`/`r` + memory | vararg/register pressure | PASS |
| pr43385 | C | `"+r"(z)` | force args through registers | PASS |
| pr43560 | C | `"=r"(p) : "0"(&s)` | hide a constant address | PASS |
| pr44852 | A | `asm("")` | stop loop optimization | PASS |
| pr45695 | C | tied `=r`, `+r` | dead-value/return stress | PASS |
| pr46309 | A | `"memory"` | full barrier in `main` | PASS |
| pr47925 | A | `asm volatile("")` | keep `bar` a real call | PASS |
| pr49218 | A | `asm("")` | keep float→int loop from folding | COMPILE (i128) |
| pr49279 | C | `"=r"(r) : "0"(p)` + memory | opaque `restrict` alias test | PASS |
| pr49390 | C | `"+r"(c) : "r"(&a)` | hide null from folding | PASS |
| pr51447 | F | `register ... asm("rbx")` | global register variable + nonlocal goto | COMPILE (nested) |
| pr51581-1 | A | `asm("")` | barrier in init loop | PASS |
| pr51581-2 | A | `asm("")` | barrier in init loop | PASS |
| pr51877 | A | `"memory"` | struct return/copy barrier | PASS |
| pr51933 | B | `"g"(...)` ×4 | keep global addresses live | PASS |
| pr52286 | C | `"=r"(a) : "0"(0)` | hide constant from folding | PASS |
| pr53645 | B | `"r"(&ur)` + memory | reload barrier around vector division | COMPILE (vector) |
| pr53645-2 | B | `"r"(&sr)` + memory | same, short vectors | COMPILE (vector) |
| pr56205 | A | `"memory"` ×3 | keep va_list save-area materialization | PASS |
| pr56866 | B | `"g"(...)` + memory | keep rotate-loop inputs live | PASS |
| pr56982 | A | `"memory"` | barrier around setjmp/longjmp | PASS |
| pr57344-1..4 | A | `"memory"` | barriers around packed bitfield reads | PASS |
| pr58277-1 | A | `asm("")` | keep `baz` non-removable | PASS |
| pr58419 | A | `asm volatile("")` | keep `dummy` a real call | PASS |
| pr63641 | A | `"memory"` | barrier before range tests | PASS |
| pr65053-1 | C | `"+g"(u)` | make loop variable appear modified | PASS |
| pr65053-2 | C | `"+g"(x)`, `"g"(u)` | same, two shapes | PASS |
| pr65648 | B | `"r"(x),"r"(y)` + memory | address-of-parameter scenario | PASS |
| pr65956 | C | `"+g"(x)` + memory | keep pointer opaque across struct passing | PASS |
| pr68328 | B | `"g"(x),"g"(y)` + memory | keep `bar` non-pure | PASS |
| pr69320-2 | A | `"memory"` | barrier between two compares | PASS |
| pr69691 | A | `asm("")` | keep loop body alive | PASS |
| pr78438 | A | `"memory"` | barrier before noinline call | PASS |
| pr78726 | B | `"g"(&b),"g"(&c)` + memory | keep global addresses live | PASS |
| pr79354 | A | `"memory"` ×3 | block cross-call optimization | PASS |
| pr79737-2 | A | `"memory"` | block store merging | PASS |
| pr80421 | B | `"r"(t)` + memory | keep first vararg in a register | PASS |
| pr81588 | B | `"g"(x)` + memory | block signedness folding | PASS |
| pr82954 | B | `"g"(p),"g"(q)` + memory | block constant propagation | PASS |
| pr84478 | A | `"memory"` | barrier before string-pool copy | PASS |
| pr84524 | C | `"+r"(v)` | defeat if-conversion in CRC loop | PASS |
| pr85156 | D | `"+m"(x),"+m"(y)` | force may-write assumption | FAIL (`__builtin_expect`) |
| pr85756 | B | `"r"(a)` + memory | expose codegen bug (unreachable at runtime) | PASS |
| pr88904 | C | `"+g"(x/y/z)` | exercise struct-return codegen | PASS |
| stkalign | D | `"=g"(x),"+m"(s):"0"(&x)` | frame alignment vs 64-aligned static | PASS |
