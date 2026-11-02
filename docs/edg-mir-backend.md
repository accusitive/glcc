# EDG IL → glcc_mir backend (`mir_gen_be`)

Architecture for the C++ path: EDG is the C++ front end; glcc_mir is the
backend. This replaces the "EDG → generated C → re-parse" round trip.

```
foo.cpp ──EDG cpfe (DO_IL_LOWERING=1)──> lowered C-like IL
        ──mir_gen_be (bridge/)──> foo.mir (text, grammar below)
        ──glcc_mir::parse──> mir::Module ──optimize/lower_cl──> foo.o
```

## Where each piece lives

| Piece | Location |
|---|---|
| MIR text emitter (C++) | `crates/edg-sys/bridge/mir_gen_be.cpp` |
| Backend dispatcher (`back_end`) | `crates/edg-sys/bridge/mir_gen_be.cpp` |
| Old C-gen backend | `edg/src/c_gen_be.c`, compiled with `-Dback_end=c_gen_be_back_end` |
| MIR text parser | `crates/glcc_mir/src/parse.rs` |
| Driver splice | `src/driver.rs` `compile_edg_input` / `compile_source` |

Build: `cargo build -p glcpp --features cxx`. Run:
`GLCPP_EDG_BACKEND=cgen glcpp ...` selects the legacy generated-C path for
A/B debugging; default is the MIR path.

## Why the IL is C-like

cpfe runs with `DO_IL_LOWERING=1`: name mangling, vtables, ctor/dtor
generation, copy ops, and pointer-to-member lowering all happen in the
front end before `back_end()` runs. The backend sees a C-shaped program:
`a_routine` definitions, `a_variable` with static storage, statements,
and expressions. Template instantiation is complete before the backend
runs (front end writes the `.ti` file; the backend does not touch it).
External names are already mangled in `source_corresp.name` (or use
`get_mangled_function_name` / `get_mangled_variable_name`,
`edg/src/lower_name.h:69-78`).

Backend entry point contract (EDG calls exactly these):

- `void back_end(void)` — called from `cfe.c:170` after `fe_wrapup`, with
  the whole IL resident in memory, reachable from the global `il_header`
  (`IL_SHOULD_BE_WRITTEN_TO_FILE` is FALSE in-process). Output file path
  arrives via the existing `--gen_c_file_name=` option.
- `void mir_gen_be_cleanup(void)` — called from `fe_cleanup`
  (`fe_wrapup.c:~903`) when `MAKE_FRONT_END_CALLABLE` is set.

Walk model: copy the skeleton of `generate_C_output_file`
(`c_gen_be.c:11710`) — sweep `il_header.primary_scope` (types, variables
without inits, routines as declarations, variables with inits, routine
definitions) and dispatch statements/expressions with `switch` on
`->kind`, mirroring `dump_statement` (`c_gen_be.c:9681`) and `dump_expr`
(`c_gen_be.c:5760`, operands left-to-right). Memory-region rule: entries
in a function-scope region may only be referenced from that region —
emit everything eagerly, do not retain IL pointers across functions.

`-fno-exceptions` only: the front end is built `-fno-exceptions`; any IL
construct requiring EH (throw/try, landing pads) is not expected. If one
appears, `mir_gen_be` must fail with a clear diagnostic, never emit
wrong code.

## MIR text format (v1)

One logical construct per line. `#` starts a comment (to end of line).
Whitespace-separated tokens. Names after `@` are symbol names (mangled
C++ names match `[A-Za-z0-9_$._]+`, no quoting needed in v1). Value
references are `%n` (decimal, unique per function, allocation order
irrelevant — parser binds them lazily). Block labels are `b<n>`.

### Types

```
void  bool  i<bits>  f32|f64|f80|f128  ptr<T>  ptr<bits>(T)
[N x T]                          array of N elements of T
struct<size>{off0:T0, off1:T1,}  explicit layout (offsets in bytes)
```

`struct<24>{0:i64, 8:ptr(i8)}` → `Ty::Struct{fields, offsets, size}`.

`i<bits>` with `bits > 128` parses as `Ty::WideInt{bits}`. Signatures may be
C-variadic: `sig=(ptr(i8), ..., i32)` (a `...` token before the closing
paren sets `variadic`).

### Values

| Syntax | Meaning |
|---|---|
| `%n` | SSA value or block param |
| `123`, `-42` | integer constant (fits i128; typed by context) |
| `f64:0x3ff0000000000000` | float constant, raw bit pattern |
| `null`, `undef`, `zeroinit` | Const::Null / Undef / Zero |
| `bytes<aabbcc>` | Const::Bytes, hex, even length |

### Module items

```
.func @name [local] [weak] [inline] sig=([T1, T2]) -> RT
.extern @name sig=([T1]) -> RT [weak]        # undefined function
.dataextern @name ty=<T>                     # extern data declaration
.global @name [local] [weak] [tls] [const] [align=N] ty=<T> init=<init>
.init @name                                  # run @name before main (.init_array)
```

`<init>`: `undef` (no init) | a scalar const | `zeroinit` |
`bytes<hex>` | `[c0, c1, ...]` aggregate of consts (array or struct,
dense layout). Pointer/reloc initializers are **not** in v1 — the
emitter must reject them with a diagnostic.

### Function body

First line after `.func` declares params; the entry block is `b0`:

```
.func @add sig=(i64, i64) -> i64
.params %0:i64, %1:i64
b0:
  %2:i64 = add %0, %1
  ret %2
```

`.params` allocates the entry block params (`ValueData{Def::Param(b0, i)}`
pushed to `blocks[b0].params` in order). A `.params` line is optional when
the function takes no parameters. Other blocks declare their params on the
label line (CLIF-style), because the branch args are pre-existing values
while the params are fresh SSA names:

```
b2(%7:i64, %8:ptr(i8)):
```

Every param name must be unique within the function (like every other
`%n`). A branch target is `b<N>` optionally followed by `(%a, %b)` —
plain value references, whose count must equal the target's param count:

```
br b2(%3, %1)
```

A branch may reference a block before its label line; the arity is then
checked against the label when it appears, and the label's param types
must match the argument types the earlier branch carried. Every block a
branch or switch references must eventually have a label line, and every
block must end in a terminator before the next label, module item, or end
of file. Blocks appear as `bN(...):` on their own line; instructions are
indented (any leading whitespace) — `%n:ty = <op> ...` or bare `<op> ...`.

### Instruction set (v1)

Arithmetic/logic (operands: `%n` or int const; result type from `%n:ty`):
`add sub mul udiv sdiv urem srem and or xor shl lshr ashr`.

Unary: `neg not clz ctz popcnt bswap sqrt ceil floor fneg <v>`.

Float binops: `fadd fsub fmul fdiv`.

Comparisons:
```
%n:bool = icmp.<eq ne ugt uge ult ule sgt sge slt sle> <a>, <b>
%n:bool = fcmp.<eq ne lt le gt ge ord uno> <a>, <b>
```

Casts (result ty from `%n:ty`): `trunc zext sext bitcast sitof uitof
sftoi uftoi fptrunc fpext ptrtoint inttoptr <v>`.

Select: `%n:ty = select <cond>, <a>, <b>`.

Memory:
```
%n:ptr(T) = alloca T x<N>          # N may be 1 and omitted
%n:ty = load [volatile] <addr>
store [volatile] <addr>, <val>
%n:ptr(T2) = gep <base> [field <i>,] [index %v,] [<const>,] ...
%n:ty = extract <agg>, <i>          # scalar field load from agg value
%n:agg = insert <agg>, <elem>, <i>
memcpy <dst>, <src>, len=<n>
memset <dst>, byte=<b>, len=<n>
```

GEP steps may be separated by commas or whitespace. A bare `<const>` step
is a non-negative element index.

Addresses:
```
%n:ptr(T) = globaladdr @name
%n:ptr(S) = funcaddr @name
%n:ptr(S) = externaddr @name
```

Calls (args each spelled `ty val`, comma-separated):
```
%n:RT = call @f ty1 %a, ty2 7        # internal (must have .func)
%n:RT = call.ext @g ty1 %a           # via .extern
%n:RT = call.ind sig=([T]) -> RT <fnptr>, ty1 %a
```

A call to a non-void function may be written bare (`call @f ...` with no
`%n:RT =`) to discard its result. A call with a result annotation must
target a non-void signature.

Terminators (end every block):
```
br b<N>(%a, %b)
brif <cond> b<then> b<else>(%x)
switch <v> [1: b1, 2: b2] default b3
ret [<v>]                # no value iff sig.ret is void
unreachable
```

Branch argument lists are optional (`br b1` for a param-less target);
switch targets take no arguments, so their blocks must have no params.

`va_start/va_arg/va_end/va_copy`, atomics, and inline asm are reserved
for later revisions; the emitter must not produce them in v1.

#### Format amendments (from the parser implementation, 2026-10)

Recorded against the first draft of this section, which the parser
(`crates/glcc_mir/src/parse.rs`) defines authoritatively:

- Block params are declared on the **label line** (`b2(%7:i64):`),
  CLIF-style. The draft's "params come from the first branch" left the
  params nameless, so no `%n` could reference them in the block body.
  Branches still determine arity (checked against the label if it
  follows the first reference) and a label's param types must agree
  with the argument types an earlier branch carried.
- Labels with no params are spelled `bN:` (no empty parens needed,
  though `bN():` is accepted).
- Signatures may declare C variadics with a `...` token
  (`sig=(ptr(i8), ..., i32)`).
- `i<bits>` above 128 bits is `Ty::WideInt`.
- `ptr(T)` is an accepted spelling of `ptr<T>` (both 64-bit), matching
  what `glcc_mir::display` prints; `ptr<bits>(T)` stays explicit-width.
- Calls to non-void functions may be written bare to discard the
  result; a result annotation on a void call is an error.
- GEP steps accept commas or plain whitespace as separators.
- Every referenced block must have a label line; a block materialized
  only by a branch reference and never labeled is a parse error.

#### Format amendments (from the EDG emitter implementation, 2026-10)

Deviations the EDG-side emitter (`crates/edg-sys/bridge/mir_gen_be.cpp`)
needed against the first draft; the parser must accept all of these for
the MIR path to consume EDG output:

- `sig=(...)` carries a **flat parenthesized list** — no inner `[...]`
  brackets around the parameter types (`sig=(i32, ptr(i8), ...) -> i32`).
  The draft's `sig=([T1, T2])` collides with array-type syntax.
- **Relocation initializers**: a pointer-valued static initializer is
  spelled `reloc @sym`, optionally followed by a byte offset with no
  whitespace (`reloc @.str.1`, `reloc @_ZTV1A+16`). Needed for any
  global holding an address (`const char *g = "x"`, vtables later).
  Accepted in `.global init=` (scalar position) and inside aggregate
  element lists. Parser semantics (implemented, `glcc_mir::parse`):
  the symbol resolves through the module tables — `.func` →
  `DataReloc::Func`, `.extern` → `DataReloc::FuncExtern`,
  `.global`/`.dataextern` → `DataReloc::Global { addend }` with the
  `+<n>` suffix as addend (default 0). The reloc lands in
  `GlobalData.relocs` at the element's byte offset (0 for a scalar
  init) and the patched image bytes are a zeroed `Const::Bytes` of the
  slot's size, exactly what `glcc_lower_mir::define_global`'s image
  path produces. Relocs are only valid in pointer-width slots
  (`ptr(...)` or `i64`, the latter covering C's `(long)&obj`); anywhere
  else is a parse error. Unknown reloc symbols are parse errors.
- gep steps are written **without surrounding brackets**:
  `gep %base field 0`, `gep %base index %v`, `gep %base 12`.
- There are **no constant instructions** — `%n:ty = 3` is not valid.
  Constants appear only as inline instruction operands (typed by
  context per the parser's `materialize` rules), so:
  - control flow folds constants at emission: `brif`/`switch` with a
    constant condition/scrutinee become direct `br`s and never reach
    the text (the parser requires value operands there);
  - casts of constants fold at emission;
  - aggregate results of `?:`/`&&`/`||` flow through an `alloca`
    slot + `store`/`memcpy` + `load` instead of block parameters.
- Address-typed MIR values (the results of `alloca`, `globaladdr`,
  `gep`, `funcaddr`, `externaddr`, and glvalue evaluation) carry a
  `ptr(T)` type, matching `glcc_lower_mir::lower_ty`: pointers to
  structs/arrays of structs erase their pointee to `void`
  (`ptr(void)`), so self-referential structs cannot recurse the inline
  type syntax. Navigation into erased pointees uses explicit byte
  offsets (`ptrtoint`/`add`/`inttoptr`); navigation into kept pointees
  uses `gep`.
- Unnamed static-storage temporaries from the IL (guard objects for
  `static` locals, "troublesome aggregate constant" holders) are
  emitted as `.global @.temp.<n> ty=<T> init=zeroinit` lines; the same
  IL variable always maps to the same `@.temp.<n>`.
- String literals are emitted as `.global @.str.<n> [const] ty=[len x
  i8] init=bytes<...>` hoisted ahead of the bodies that reference
  them; `globaladdr` resolves by name, so module-item order only
  matters for the parser's single-pass name resolution (the emitter
  hoists everything a body references).

#### Format amendments (dynamic initializers / `.init_array`, 2026-10)

C++ dynamic initializers (EDG's `__sti__` routines) are ordinary
`.func` definitions; the new `.init` directive wires them into the
ELF `.init_array` startup table. Grammar, authoritative in
`glcc_mir::parse`:

```
.init @name
```

- `@name` must be a function **defined in this file** with `.func`
  declared on an earlier line (single-pass resolution, like `reloc`
  symbols). An `.extern` target or an unknown name is a parse error;
  a `.init` line placed before its `.func` is therefore also a parse
  error. The recommended emitter placement is immediately after the
  `__sti__` function's body.
- The target's signature must be `sig=() -> void` (constructors take
  no arguments); anything else is a parse error, and
  `glcc_mir::verify` re-checks the same invariant on the built module.
- Registering the same function twice is a parse error ("duplicate
  .init entry"). There is no `[once]` qualifier: the runtime calls
  each `.init_array` entry exactly once per program/library load, and
  function-local statics guard themselves in the body.
- List order is emission order: `Module.init_funcs` keeps the `.init`
  lines in file order, the object layer writes one 8-byte slot per
  entry in that order, and the linker concatenates per-TU
  `.init_array` sections in link order, so the final run order across
  the program is source order.
- MIR representation: `glcc_mir::ir::Module::init_funcs: Vec<Func>`.
  Optimizer contract: `DeadModuleItems` treats every `init_funcs`
  entry as a root (a constructor is typically `local` with no MIR
  caller) and remaps the entries when it renumbers functions. Nothing
  else in the pass pipeline needs to know about them.
- Object emission: `glcc_lower_cl::emit_init_array(product:
  &mut cranelift_object::ObjectProduct, init_funcs:
  &[cranelift_module::FuncId]) -> Result<(), String>` appends the
  `.init_array` section (PROGBITS, `SHF_ALLOC | SHF_WRITE`, 8-byte
  aligned, no merge) to the finished `ObjectProduct`, one zeroed
  8-byte slot per id patched with an `R_X86_64_64` absolute
  relocation against the function symbol (`product.function_symbol`).
  The driver maps `Module.init_funcs` through the `mir::Func ->
  FuncId` table from lowering (`lower_module_with_debug` returns it;
  `lower_module_collecting_with_ids` is the non-debug variant) and
  calls the helper right after `obj.finish()`, before narrow-reloc
  patchup and DWARF emission. Empty `init_funcs` emits no section;
  non-ELF products are an `Err` (reported as a warning, like the
  winunwind path). Windows has no story yet: `.CRT$XCU` needs COFF
  grouping sections with absolute pointer images instead of
  relocations.
- This mirrors the C path's existing convention
  (`glcc_lower_mir::emit_hook_array` emits a `__glcc_init_array`
  global with a custom `.init_array` section name); the `.init`
  route produces a proper section from the object writer instead of
  reusing a data global, and both may coexist in one program.

#### Format amendments (class support, 2026-10-31)

Observed while bringing ctors/dtors/references/virtual dispatch
through; the parser already accepts all of these spellings, but the
ORDERING and conventions are now load-bearing:

- **Module-item order.** The parser resolves `reloc` in `.global
  init=` during pass one, so every symbol a data definition relocates
  to must be declared earlier in the file. Function bodies are parsed
  in pass two and may reference anything. The emitter therefore emits:
  `.extern` routine interfaces, string globals, all `.func` bodies,
  `.init` entries, `.dataextern` declarations, defined `.global`s in
  **dependency order** (topological over the variable-to-variable
  edges of their static initializers; a cycle, e.g. `void *p = &p`,
  is a hard emitter diagnostic), then the unnamed `@.temp.<n>`
  globals.
- **C++ references** spell as their underlying pointer everywhere
  (`int&` is `ptr(i32)`): after `DO_IL_LOWERING` a reference lvalue's
  value IS the address of the referent, so params/returns/bindings
  need no distinct handling.
- **Null pointer constants** are ck_integer constants carrying a
  pointer type (EDG's `make_zero_of_proper_type`); in static
  initializers they print as `null`, and runtime stores spell
  `store <addr>, null`.
- **Inline function definitions are weak** (`.func @name inline weak
  ...`), mirroring the ODR; definitions for the EH bookkeeping
  variables (`__curr_eh_stack_entry`, `__eh_curr_region`) and
  `__dso_handle` are emitted `local` so they do not collide with the
  runtime/crtbegin copies.
- **Known `__builtin_X` libc wrappers** (`__builtin_strlen`,
  `__builtin_memcpy`, ...) emit as calls/addresses of `X` itself;
  `__builtin_is_constant_evaluated` folds to 0, `__builtin_expect` to
  its first argument, `__builtin_unreachable` to the `unreachable`
  terminator.
- GEP **index steps are only emitted over scalar pointees** (the
  parser strides ConstIndex/Index by the pointee size for scalars and
  by the ELEMENT size for arrays -- wrong for `T(*)[N]` pointer
  arithmetic -- and rejects them over structs/void). Struct, array,
  void and routine pointees navigate with explicit byte arithmetic
  (`ptrtoint`/`add`/`inttoptr`) instead; likewise all pointer
  subtraction and negative-index subscripts scale by the element size
  explicitly.
- A local static variable's constant initializer lives in a
  `local-static-variable-init` IL entry (the variable itself sits in
  the file-scope region, so `init_kind` is `initk_function_local` and
  the `.global` starts zeroed); the emitter applies the constant with
  runtime stores at first reference in the body.

### Aggregate-value convention

Matches `glcc_lower_cl`'s lowering (see `crates/glcc_lower_cl/src/lib.rs`
header): an aggregate-typed MIR value is represented by the **address of
its storage** (a stack slot from `alloca`, or a global). Passing an
aggregate argument means passing that address value; aggregate results
arrive as caller-provided addresses per the lowering's hidden out-pointer
convention. Study `crates/glcc_lower_mir/src/lower.rs` call/return
handling and mirror it exactly — the parser does not reinterpret
anything; conformance lives in the emitter.

### Parser invariants (`glcc_mir::parse`)

- Output must pass `glcc_mir::verify_module`.
- Parse errors: `Result<Module, MirParseError>` with `line: u32` and a
  plain message; never panic on malformed input.
- Lazy value binding: `%n` may be referenced before the line that
  defines it (an instruction result later in the function, or a block
  param whose label follows the first branch to the block); double
  definition or never-defined use is an error.
- Constants are values: every literal allocates a value via
  `push_const` (see `ir.rs:803`). Integer constants are typed by
  context: the result type for binop/unary/select operands, the other
  operand's type for `icmp`/`fcmp`, the addressed pointee for `store`,
  the spelled argument type for calls, the return type for `ret`.
- `.func` bodies use `ModuleBuilder::declare` + manual entry params
  (`builder.rs:146`; entry block has no params until pushed — see the
  add example in `crates/glcc_mir/src/lib.rs` docs and
  `glcc_lower_mir/src/lower.rs:1614`).
- After parsing, the module must pass `verify_module`; the parser runs
  it itself and reports the first violation as a parse error.

## Slice roadmap

1. **v1 slice (this milestone)** — scalar ints/floats/pointers, structs
   and arrays (layout from IL), globals with constant inits, locals,
   arithmetic/comparison/casts, control flow (if/loops/switch/break/
   continue), direct+indirect+extern calls, string literals, function
   pointers. Acceptance: `glcpp tests/cpp/smoke.cpp -o smoke` links and
   runs correctly.
2. Next: class support end-to-end (ctors/dtors appear as ordinary
   routines in lowered IL), operator new/delete, references, mangling
   verification against g++ symbols, `-g` debug info, inline asm.
3. EH: `docs/eh-plan.md` — LLVM-shaped `invoke`/landing-pad model in MIR,
   lowered through Cranelift's `try_call` (no fork needed; the unwinder
   metadata is ours). Until it lands, EH constructs must diagnose.

## Host/arch support audit (2026-11-02)

`cargo build -p glcpp --features cxx` used to be x86-64-only in three
independent ways, all fixed in `crates/edg-sys/build.rs`:

- EDG's `linux-gcc` cmakedef hardcodes `USE_QUADMATH_LIBRARY=1`
  (libquadmath is x86-only) and `FP_LONG_DOUBLE_IS_80BIT_EXTENDED=1`.
  Non-x86 builds override the former to the software-emulation path
  (`APPROXIMATE_QUADMATH`) and the latter to `FP_LONG_DOUBLE_IS_BINARY128`
  (aarch64/riscv64 `long double` is IEEE binary128); the MIR emitter
  spells `fk_long_double` as `f128` under the same `#if`, and
  `glcc_lower_cl` already lowers f128 arithmetic on non-x86 targets.
  aarch64 also overrides the unsuffixed `TARG_JMP_BUF_*` defaults: the
  cmakedef's x86-64 values (25 elements) beat `defines_linux.h`'s ARM64
  auto-detection via `#ifndef` guards, and EDG's EH lowering would
  allocate a 200-byte setjmp buffer for glibc's 312-byte jmp_buf —
  stack corruption.
- EDG's host-defaults cmake detection only knows x86_64/amd64 Linux
  hosts, so `EDG_BASE` is now passed explicitly (the base itself is
  arch-independent).
- The cmake runtime target/path was hardcoded `linux_x86_64`.

### The EDG runtime on non-x86

cpfe's self-hosted runtime build emits generated C containing the GNU
`__float80`/`__float128` keywords (lib_src guards on the predefined
`__EDG_FLOAT{80,128}_ENABLING_POSSIBLE`, which must stay 1 so the ISO
`_Float128` keyword that glibc's own headers need exists — gcc on
aarch64/riscv64 has neither GNU spelling, and `-U`/`-D` of those
predefines is rejected). Instead, `build_runtime_with_host_cc` compiles
the same `edg/lib_src` sources with the host C++ compiler plus cpfe's
`__EDG_*` predefines (captured via `cpfe --list_macros`; keep in sync
with the EDG checkout). Skipped there: placement new/delete,
`std::exception`/`bad_alloc`/`type_info` (libstdc++ provides them),
`c99_complex.c` (needs the f80/128 keywords), and the
`std::uncaught_exception` definition in `eh_util.c` (return-type clash
with the libstdc++ header).

### Known gaps found by the audit

- **EH-shaped code miscompiles on aarch64.** Any construct EDG lowers
  with an internal try block (operator new expressions — e.g.
  `std::vector::push_back` — and throw/try) crashes at runtime on
  aarch64; plain `setjmp`/`longjmp` through glcpp works, so the bug is
  in the interplay of the EH bookkeeping code shape with aarch64
  lowering. This is the aarch64 face of the known EH gap (`exceptions`
  stays xfail for the missing unwinder on all targets).
- **Inherited-member access through a base-typed operand emits a wrong
  field index.** `place()`'s `eok_dot_field` case indexes the field by
  position in the class the IL types the operand with; for `d.b` where
  `b` is inherited, the IL types the operand with the *base* class
  while the value is the derived storage, so the gep step lands on the
  wrong member of the derived layout. Also, `static_cast<D*>(bp)` on an
  erased (`ptr(void)`) base followed by member selection emits a `gep
  field` step the MIR parser rejects ("bad gep step Field(1) on Void").
  Correct navigation needs byte offsets relative to the operand type
  (plus handling for aggregate-typed values in `byte_offset_addr`).
- **Unsupported constructs** (clean `mir_gen_be` diagnostics):
  bit-fields, virtual inheritance (expression kind 20 in the probed
  shape), non-char string literals (`L""`, `u8""`, `u""`, `U""`), VLA,
  designated/repeat aggregate initializers, integer constants wider
  than 64 bits.
- **`wchar_t` signedness diverges from native g++** on aarch64
  (cpfe predefines `__WCHAR_TYPE__ int` from the x86-oriented
  predefined-macro file, gcc uses `unsigned int`). Fixing it requires
  patching `$EDG_BASE/lib/predefined_macros.txt` (the driver pins
  `EDG_BASE` to the submodule base).

