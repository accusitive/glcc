# Kernel-build blockers — fix writeups

From the Linux-6.12.30 `tinyconfig` compile-fest with
`CC=glcc -fbackend=native -fintegrated-as` (see `tools/kernel.bash`; tree in
`~/.cache/glcc-kernel/linux-6.12.30`, build log `build.log` there). Each
section is self-contained: symptom, root cause, fix sketch, verification.
All line references are to this tree as of the fix wave that produced ~30
kernel objects (`lib/sort.o`, `lib/list_sort.o`, `kernel/range.o`, …).

Rebuild glcc first (`cargo build --release`); the shim
(`~/.cache/glcc-kernel/cc-glcc.sh`) uses the release binary, and
`tools/kernel.bash build` reruns the build with `-k` so one pass maps many
failures.

---

## 1. glasm: `.popsection` / `.previous` are no-ops (silent wrong-section hazard)

**Symptom.** Not an error — a correctness bug. `.pushsection __ex_table,"a" ;
.long ... ; .popsection` leaves the current section as `__ex_table`, so any
code emitted after it lands in the wrong section. The kernel's `.S` files do
exactly this *inside* function bodies (exception-table entries in
`arch/x86/lib/clear_user_64.S` et al). `IGNORED_DIRECTIVES` in
`crates/glasm/src/gas.rs:429` lists both `previous` and `popsection`, and the
`.pushsection` arm (`gas.rs:560`) documents "no section stack" as the design.

**Fix.** Implement a real section stack:
- `Stmt::Section` already switches sections; add `Stmt::PushSection` /
  `Stmt::PopSection` (or a flag on the existing `Section` variant) carrying
  the spec, and have the assembler (`crates/glasm/src/assemble.rs`) maintain
  a stack: push the *previous* section on push, restore on pop.
- `.previous` restores the section active before the last `.section`
  (GAS: a stack of the same shape; `EXPORT_SYMBOL` sections rely on it).
- Track the current section as state in the module/section builder; the
  no-stack shortcut means it is likely just "last section wins" today.

**Verify.**
```bash
printf '.text\nnop\n.pushsection .foo,"a"\n.long 1\n.popsection\nnop\n' > /tmp/sec.s
glcc -fintegrated-as -c /tmp/sec.s -o /tmp/sec.o && objdump -h /tmp/sec.o
# .text must contain TWO nops (2 bytes); .foo 4 bytes of .long
```
Then: the `__ex_table`-heavy files (`AS arch/x86/lib/clear_user_64.o` in the
kernel log) should stop mis-assembling.

## 2. glasm: `.set` directive (symbol assignment)

**Symptom.** `unsupported assembler directive '.set'` on
`.set .L__sym_size_clear_page_rep, .-clear_page_rep` — the SYM_FUNC_END size
idiom every kernel function emits. Dispatch ends at the `_ =>` arm,
`crates/glasm/src/gas.rs:670`.

**Fix sketch.**
- Add `Stmt::Set { name: String, value: Expr, line: u32 }`.
- Grammar: `.set symbol, expression` and the `symbol = expression` alias;
  expression may reference `.` (see below) and symbols.
- In `assemble.rs`, define/override the symbol at the current location with
  the expression's evaluated value (absolute symbol if the expression is
  constant; a section-relative symbol if it involves `.` — `.L__sym_size_x`
  values are `.-x`, i.e. relative to the same section). Check how the ELF
  writer (`elf.rs`) handles absolute vs section-relative symbols; a local
  ABS symbol is the fallback that is still correct for `.size` consumers.
- Cheap tier first: `Expr::Here` already exists (`expr.rs:27`, spelled `$`);
  confirm the lexer maps GAS's `.` to `Expr::Here` inside expressions — if
  `.` lexes as an ident or is rejected, add it.
- Note: `.size`/`.type` are currently in `IGNORED_DIRECTIVES`, so ignoring
  `.set` of a `.L__sym_size_*`-named symbol would also unblock the kernel;
  the real implementation is preferred since `.set` symbols can feed
  instructions.

**Verify.**
```bash
printf '.text\nf:\nnop\n.set sz, .-f\n.set K, 42\n.quad K\n' > /tmp/set.s
glcc -fintegrated-as -c /tmp/set.s -o /tmp/set.o && nm /tmp/set.o
# expect `sz` (or no error at minimum) and 42 in the data
```

## 3. glasm: quoted section names

**Symptom.** `expected section name` on `.section ".export_symbol","a"` —
the kernel's `EXPORT_SYMBOL` expansion quotes the name. `parse_section`
(`crates/glasm/src/gas.rs:695`) accepts only `GTok::Ident`.

**Fix.** Accept `GTok::Str` too: strip the quotes (they are literal names,
no escapes in practice) and use the body. Same arm covers `.pushsection`
(`gas.rs:560`). The flags string parsing (`"a"`, `"ax"`, `"awx"`, `@progbits`
third arg) already exists right below — keep its behavior.

**Verify.**
```bash
printf '.section ".foo","a"\n.long 1\n' > /tmp/qsec.s
glcc -fintegrated-as -c /tmp/qsec.s -o /tmp/qsec.o && objdump -h /tmp/qsec.o
# expect a section named .foo
```

## 4. glasm: `*` (multiply) in displacement expressions

**Symptom.** `expected '('` on `movq %rax,1*8(%rdi)` — unrolled string ops
in `arch/x86/lib/*.S`. The lexer emits `*` as `GTok::Star` (it is the
indirect-call marker), and the expression-token scan `depth_of`
(`crates/glasm/src/gas.rs:783`) only accepts `GTok::Punct("*")`, so the
displacement stops after `1`; the memory-operand path then finds `Star`
where it wants `(`.

**Fix.** Two spots, both in `gas.rs`:
- `depth_of` (line ~789): also accept `GTok::Star` as an expression token.
- the conversion loop a bit below (`GTok::Punct(p) => Tok::Punct(p) …`):
  map `GTok::Star` to `Tok::Punct("*")` so the expr parser sees a multiply.
Mid-expression `*` is always multiplication; the *leading* indirect position
(`call *%rax`, `jmp *sym`) is consumed before `expr` runs (see the
`indirect` handling at `gas.rs:~977`), so this is safe — add a regression
test for `call *%rax` anyway.

**Verify.**
```bash
printf '.text\nf:\nmovq %%rax,1*8(%%rdi)\nmovl $4096/8,%%ecx\nret\n' > /tmp/disp.s
glcc -fintegrated-as -c /tmp/disp.s -o /tmp/disp.o && objdump -d /tmp/disp.o
# expect mov 0x8(%rdi),%rax and mov $0x200,%ecx
```

## 5. pp: `#` comment lines in assembler-with-cpp mode

**Symptom.** `unknown directive '#Setup'` on `arch/x86/realmode/rm/*.S` —
x86 asm uses `# Setup stack` comments; the pp (`crates/glcc_pp/src/tree.rs:418`)
errors on any unrecognized `#word`. Since the driver now preprocesses `.S`
itself (driver.rs, the `.S` branch added under `-fintegrated-as`), every x86
`#` comment is fatal. The kernel's `#APP`/`#NO_APP` markers have the same
shape.

**Fix.** Add an assembler mode to `PpOptions` (e.g. `assembler: bool`, set by
the driver's `.S` preprocessing path). In `tree.rs`'s directive dispatch:
when in assembler mode and the token after `#` is neither a known directive
nor a `# <number>` linemarker, treat the rest of the line as a comment and
pass it through verbatim to the output (GAS lexes `#` as a comment on x86).
Do NOT passthrough glcc's own `# <n> "file"` linemarkers — they are
regenerated by the pp; leave the linemarker machinery as-is.

**Verify.**
```bash
printf '#include <stdio.h>\n# a comment\nint main(){return 0;}\n' > /tmp/hash.c
glcc -E /tmp/hash.c | tail -2   # needs the assembler mode wired through the driver
# or unit-test in glcc_pp: pp() with the new option, assert the comment survives
```
Kernel check: `AS arch/x86/realmode/rm/header.o` stops failing at `#Setup`.

## 6. frontend: `offsetof(T, m[i])` with a *runtime* index (GCC extension)

**Symptom.** ``offsetof index is not an integer constant expression`` (101
errors, the biggest C-side cluster) on the kernel's
`container_of`-style macros: `__builtin_offsetof(struct latch_tree_node,
node[idx])` where `idx` is a function parameter
(`include/linux/rbtree_latch.h:72`). GCC permits a non-constant array index
in `offsetof` and lowers it to `offsetof(T, m) + i * sizeof(elem)` as a
runtime expression.

**Root cause.** `crates/glcc_frontend/src/check/mod.rs:3693` — the
`Designator::Index` arm of the offsetof walker demands
`eval_const(index)` succeed and errors otherwise.

**Fix sketch.**
- Keep the ICE requirement for contexts that *need* a constant (a
  `_Static_assert` operand, an array size, a case label): when `offsetof`
  with a runtime index reaches a const-eval context it will fail there
  naturally — verify that path gives a sane message rather than a panic.
- Make offsetof potentially non-constant: the checked offsetof currently
  folds to a constant at check time (see how the result is stored in the
  `CheckedExpr` — likely a `Const`/`OffsetOf` kind with a folded `u64`).
  For the runtime-index case, keep the constant prefix (the walk up to the
  designator) and store the trailing `base + i * elem_size` as a real
  expression: `(u64)&((T*)0)->m` style lowering — i.e. lower to
  `Const(prefix) + Convert(index, u64) * Const(elem_size)`. The lower-mir
  side then needs no new node if you reuse plain arithmetic.
- The `.checked_add`/`checked_mul` overflow checks at `mod.rs:3697+` only
  apply to the constant path; the runtime path needs none.

**Verify.**
```c
struct s { int a[4]; int b; };
unsigned long f(unsigned long i) { return __builtin_offsetof(struct s, a[i]); }
// GCC: f(2) == 8; must compile at -O0 and -O2, and run correctly via JIT/obj
_Static_assert(__builtin_offsetof(struct s, a[2]) == 8, "ice path");
```
Kernel check: `CC mm/*` / `CC lib/find_bit.o`-style files stop failing at
rbtree_latch.h / find.h.

## 7. Driver/shim debt (the kbuild shim works around these today)

Not kernel-blockers per se (the shim in `tools/kernel.bash` handles them),
but each is a small driver fix that lets the shim shrink:

- **stdin input** (`glcc ... -`): kbuild's `cc-version.sh`/`as-version.sh`
  probes pipe snippets to the compiler. The driver's input classifier
  (`src/driver.rs`, `FileKind`/`Lang` classification) should accept `-` as
  "read stdin, language from `-x` or default C", stage it to a temp file,
  and proceed. Shim workaround: splice stdin into a temp file.
- **`-S` assembly output**: offset-generation rules
  (`scripts/mod/devicetable-offsets.s`, `arch/x86/kernel/asm-offsets.s`)
  invoke `$(CC) -S`. The native backend could emit glasm text — glasm has a
  text *frontend* but no encoder→text path, so either print glasm-source
  from a new codegen dump mode or accept `-S` only for `-x
  assembler-with-cpp` inputs by forwarding pp output. Shim workaround:
  delegate `-S` to host gcc (valid: same LP64 SysV ABI).
- **GCC version masquerade**: glcc predefines `__GNUC__ 4 8 0`
  (`crates/glcc_pp/src/predefine.rs`); the kernel requires ≥5.1 and kbuild
  feature probes key off it. The shim passes
  `-D__GNUC__=8 -D__GNUC_MINOR__=2 -D__GNUC_PATCHLEVEL__=0`. Bumping the
  predefined version in-tree would deviate from the clang-matching
  predefine table (pp-diff corpus tests) — decide deliberately, or teach
  the driver a `-fgcc-compat-version=N` knob.

Also note for anyone continuing: kbuild flag probes all pass via the
driver's warn-and-continue on unknown flags, so `-mno-red-zone`,
`-mcmodel=kernel`, `-fno-omit-frame-pointer` etc. are silently *ignored* —
fine for compiling, but any runtime experiment with this kernel needs them
revisited.

Sections 1–6 above are fixed (2026-11-12). The next C-side wave, by error
count in the post-fix build log, is the two below. The glasm macro family
(`.macro`/`.endm`/`.if`/`.endif`/`.else`/`.rept`/`.irp`/`.fill`/`.code16/32/64`,
~700 error lines, blocks `arch/x86/entry/entry.S`) is deliberately deferred —
it is a much bigger chunk of assembler work.

## 8. frontend: `packed` on enums (252 errors)

**Symptom.** `_Static_assert(sizeof(enum rw_hint) == 1)` fails everywhere
`include/linux/rw_hint.h` is included. The kernel declares

```c
enum rw_hint { WRITE_LIFE_NOT_SET = 0, …, WRITE_LIFE_EXTREME } __packed;
```

GCC's packed-enum extension shrinks the representation to the smallest
integer type that holds every enumerator — here one byte.

**Root cause.** Two gaps, both cheap:
- `parse_enum` (`crates/glcc_frontend/src/parse.rs:6200`) ends with
  `self.skip_gnu_suffixes()` (parse.rs:6216 → parse.rs:5589), which *parses
  and discards* trailing `__attribute__((packed))`. The struct path right
  next door (`parse.rs:5509`–5523, "Trailing attributes") shows the shape:
  collect `GnuAttributes`, set `def.packed = true`.
- `EnumDef` (`crates/glcc_frontend/src/sema.rs:52`) has no `packed` field at
  all, and the representation ladder in `parse_enumerator_list`
  (parse.rs:6293–6408: starts at `int`, widens via `enum_ladder_rank`,
  flips to unsigned when all values are non-negative) has no packed tier —
  it can never go *below* `int`.

**Fix sketch.**
- Add `pub packed: bool` to `EnumDef` (default false), threaded through
  `complete_enum_tag` like `fixed`. Packed-ness is fixed at completion,
  same as `repr` — forward declarations share the `EnumId`, so nothing
  else changes.
- In `parse_enum`, replace `skip_gnu_suffixes()` with
  `parse_gnu_annotations()` and read `attrs.is_packed()` (same helper the
  struct path uses), passing it into `complete_enum_tag`.
- Extend the ladder: when packed, *after* the existing widen/unsigned logic,
  shrink to the first fitting type of `unsigned char`/`signed char` →
  `unsigned short`/`short` → `unsigned int`/`int` → … using the existing
  `enum_value_fits` (parse.rs:6243) over the completed min/max. GCC picks
  unsigned when every value is non-negative — differential-test the sign
  choice against the host `gcc` rather than guessing (a one-line
  `enum { A } __attribute__((packed));` + `sizeof`). A C23 `enum E : T`
  fixed underlying type wins over packed (GCC warns/errors in some
  versions; pin whatever gcc does).
- The ripple is mostly free: layout, `sizeof`, conversions, and promotions
  all key off `repr` already (check/mod.rs:679–703 promotes enums to their
  repr unless it is `int` — a `uchar`-repr enum still promotes to `int` in
  arithmetic, exactly like GCC; `enum_repr_pair` at check/mod.rs:1189 keys
  off repr too). Watch `complete_enum_tag`'s re-declaration compatibility
  check so a later redecl without `__packed` still matches.

**Verify.**
```c
enum hint { A = 0, B, C, D, E } __attribute__((packed));
_Static_assert(sizeof(enum hint) == 1, "packed enum");
int f(void) { return B + 1; }   // arithmetic still promotes to int
```
compile at -O0 and -O2, run, and diff `sizeof` against the host gcc. Kernel
check: the `sizeof(enum rw_hint) == 1` cluster (252 errors) disappears.

## 9. frontend: `__builtin_bswap16/32/64` don't const-fold (26 errors)

**Symptom.** `error: case label is not an integer constant expression` on
the kernel's ethertype switch (`include/linux/netdevice.h:4876`):

```c
case (( __be16)(__u16)__builtin_bswap16((__u16)((0x0800)))):
```

**Root cause.** bswap goes through the generic builtin-prototype path
(`check/prototypes.rs:73`–75) and is checked as an ordinary call, so it
lands in `CheckedExprKind::Call` — which `eval_const` rejects wholesale
(`eval.rs:1052`). The case-label check (`parse.rs:2581`) folds the label and
fails. Runtime lowering is fine already (`ScalarBuiltin::Bswap`,
`glcc_lower_mir/src/lower/builtin.rs:51`); only const-eval is missing.

**Fix sketch.** Fold at check time, no new node kind: in the builtin-check
path, intercept `__builtin_bswap16/32/64` before the generic-call fallback.
If the argument folds via `eval_const`, emit a constant node with the byte
order reversed at the argument's width, typed as the prototype's result type
(`UnsignedShort`/`UnsignedInt`/`UnsignedLongLong`); otherwise fall back to
the existing call so the runtime path is untouched. Care points:
- The fold happens *after* the usual implicit conversions, at the
  prototype's parameter type — `bswap16(0x0800u)` swaps a `u16`, i.e.
  `0x0008`, regardless of the source literal's width.
- Keep the existing should-panic substrings intact; this only *adds* folds,
  never new errors. (The same treatment would cover the clz family later if
  the kernel needs it, but bswap is the only cluster in the log.)
- Mirroring precedent: `__builtin_constant_p` folds to
  `CheckedExprKind::ConstantP` (`eval.rs:985`) and `__builtin_clzg` gets its
  own `CheckedExprKind::Clz` (`check/gnu.rs:123`). A dedicated
  `CheckedExprKind::Bswap` + `eval_const` arm is the more uniform shape if
  folding at check time fights the AST; try check-time folding first since
  it touches one site.

**Verify.**
```c
_Static_assert(__builtin_bswap32(0x12345678u) == 0x78563412u, "bswap");
unsigned short g(unsigned short x) { return __builtin_bswap16(x); }
int main(void) { return g(0x0102) == 0x0201 ? 0 : 1; }
```
must compile at -O0 and -O2 and run correctly (the non-constant path still
goes through `ScalarBuiltin::Bswap`). Kernel check: the netdevice.h
case-label cluster (26 errors) disappears.
