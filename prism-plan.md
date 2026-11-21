# prism-plan.md — glcc_prism: a C-subset shader language on glcc

Status: **Phases A–C complete and green** (whole-workspace `cargo test` passes:
65 test binaries, 0 failures). This file is the working plan for the remaining
phases (D–H) and a precise record of what is already in the tree, so work can
continue without re-deriving decisions.

Related docs: `AGENTS.md` (repo conventions — read first), `docs/prism.md` (to be
written in Phase H), `include/prism.h` (the user surface).

## 1. What prism is

Prism is a C subset for shader programs, compiled through the existing glcc
pipeline (`glcc_frontend` → `glcc_lower_mir` → `glcc_mir`) into shader backends.
Targets, **all produced from one shared shader IR** (PrismIR):

```
MIR ──► PrismIR ──┬── .isa  text (the IR's own canonical form; goldens)
                  ├── SPIR-V binary        (--emit-spirv)
                  ├── RSX (PS3, nv40) text (--emit-rsx)
                  └── Xenos (Xbox 360) text(--emit-xenos)
```

Design principles (user-approved, do not relitigate):

1. **New backend crate** (`crates/glcc_prism`), not native-backend architectures:
   the CPU backends are ELF-terminating, regalloc2-coupled, stack-frame-coupled.
   PrismIR is SSA, width-typed (f32/i32 × 2/3/4 lanes), DX9-flavored (src
   swizzles + negate, dp4, texld), with a structured region tree.
2. **Physical registers, write-mask splitting, register banking (RSX R/H, Xenos
   vector+scalar co-issue) are per-target lowering concerns.** PrismIR carries
   typed virtual registers only. Op widths in the IR are what let the Xenos
   backend classify scalar vs vector co-issue slots and let SPIR-V use real
   vec2/vec3 types.
3. **Instruction scheduling is shared**, once, in `glcc_prism`, before any
   target consumes the IR (`SchedModel { alu: 1, texld: 8 }`, sample hoisting,
   live-pressure guard). Targets render the scheduled order; target-specific
   packing (Xenos co-issue pairing) layers on top locally.
4. **Minimal language surface**: compiler support = `[[prism::*]]` attributes,
   swizzles, `tex2D`, and the profile checker. Everything else lives in
   `#include <prism.h>` (macros over swizzles/calls).
5. Entry points return output structs; inputs are value params (structs).
   `[[prism::vertex]]`/`[[prism::fragment]]` mark entry points;
   `[[prism::semantic(NAME)]]` attaches hardware bindings;
   `[[prism::uniform]]` marks host-set uniforms; `sampler2D` globals are
   host-bound resources.
6. Uniform layout: ISA targets pack uniforms into c-registers (float4 = 1 reg,
   scalars packed 4-per-reg); SPIR-V uses std140. `interface.rs` computes both
   tables from one uniform walk.
7. F32-only, ≤32-bit ints (era fidelity; doubles/64-bit ints are profile
   errors — see the checker).

## 2. Current state — what Phases A + B already put in the tree

Everything below exists, builds, and is tested (whole workspace green). Do not
re-do; do not rename without updating the call sites listed.

### 2.1 Driver (`src/driver.rs`)

- `LangOpts.prism: bool` (`crates/glcc_frontend/src/lang/mod.rs`), set by
  `-fprism` / `-fno-prism` via `Job.prism` and `language()`.
- `Action::{EmitIsa, EmitSpirv, EmitRsx, EmitXenos}` + flags `--emit-isa`,
  `--emit-spirv`, `--emit-rsx`, `--emit-xenos` (last wins, like `--emit-mir`).
- `glcc_prism::ShaderTarget { Isa, Spirv, Rsx, Xenos }` (crate root lib.rs) and
  `glcc_prism::emit(&Module, &Sema, &Interner, ShaderTarget) -> Result<Vec<u8>, String>`
  — **the backend entry point; currently stubs erroring "not implemented yet"**.
  Phase C replaces the `Isa` arm; E/F/G the others.
- `parse_source(src, arena, flavor, target, async_enabled, lang, error_format)`
  — extracted from `lower_source_to_mir`, returns `(Sema, Interner, TranslationUnit)`;
  runs `parser.prism_validate(&tu)` when `lang.prism` (while the parser can still
  render spans). `lower_source_to_mir` now calls it.
- `emit_shader_source(...)` (prism pipeline: parse → validate → lower →
  `optimize_with(..., OptConfig { vectorize: false })` → verify → `glcc_prism::emit`)
  and `do_emit_shader(ctx, job, target)` + `shader_outputs(job, target)`
  (output naming: `-o` for a single input, else `<stem>.isa|spv|rsx|xenos` in cwd).
- `run()` rejects `-fprism` with Link/Compile actions; `--emit-*` without
  `-fprism` errors with a hint.
- `__GLCC_PRISM__=1` predefine: pushed into `job.cpp` (clang -E path) and into
  the native pp's `predefined` (native_pp_opts). prism.h `#error`s without it.
- `glcc_prism` is a workspace member and a non-optional dependency of the root
  `glcc` package (no cargo feature needed — pure Rust, no heavy deps).

### 2.2 Frontend (`crates/glcc_frontend`)

- Attributes: `parse::PrismStage { Vertex, Fragment }`, `parse::PrismAttributes
  { stage, semantic: Option<SymbolU32>, uniform: bool }`. `parse_std_attributes`
  takes `&mut PrismAttributes` and handles the `prism` namespace (payload
  parsing for `semantic(NAME)`; previously payloads were skipped).
  `GnuAttributes.prism` + merge; `Member.prism` field;
  `Sema.prism_attrs: HashMap<DeclId, PrismAttributes>` recorded where
  `std_attrs` is recorded (function definitions and plain declarations only).
- `[[...]]` parses at any `-std` when prism mode is on (`peek_std_attributes`).
- `apply_vector_size` accepts **12-byte** vectors under prism (two sites: the
  power-of-two check in the GNU attribute parser and the range check in
  `apply_vector_size`). Plain C behavior unchanged.
- `check/prism.rs` (new):
  - `Sema::prism_validate(&tu, interner) -> Vec<CompileError>` — whole-Sema
    profile walk: pointers anywhere (decl types, struct members, casts,
    address-of, deref, arrow, pointer subscript), 64-bit ints, double/long
    double/__float128/_Complex/va_list, VLA, tls, goto/computed goto/asm,
    variadics, recursion (call-graph cycle over `sema.bodies`), calls to
    undefined non-allowlisted functions (allowlist: `__builtin_` prefix +
    `PRISM_SCALAR_BUILTINS`), sampler rules (file scope only, no copy/pass/
    return/assign/initialize), entry points (≥1, defined, external linkage,
    struct in/outs whose scalar/vector leaves all carry semantics).
  - `Sema::is_prism_sampler_ty(&Ty, &Interner)` — matches the reserved tag
    `__prism_sampler2d`.
  - `parse_swizzle(&str) -> Option<Vec<u8>>` (xyzw + rgba aliases, 1–4 lanes).
  - `Sema::swizzle_node` (single-lane swizzle reads as the scalar element type!),
    `Sema::check_swizzle_store` (compound assign rejected; scalar rhs splats),
    `Sema::check_tex2d` (arg0 must name a file-scope sampler; coord must be
    float2), `Sema::callee_is_defined`.
- `check/mod.rs`: new `CheckedExprKind` variants —
  - `Swizzle { base, lanes: &[u8] }` (rvalue read),
  - `SwizzleStore { base, lanes, rhs }` (whole-vector read-shuffle-store),
  - `PrismSample { global: DeclId, coord }` (tex2D; NOT a Call node),
  - `Splat { expr }` (scalar → vector broadcast).
  Intercept points: prism-gated swizzle intercept in `check_member` (before
  `find_field`; vector base only, so struct fields named `x` still work);
  swizzle-store rewrite in `check_assign` (before the lvalue check); `tex2D`
  arm in the check_call builtin ladder (skipped when a user-defined `tex2D`
  exists). All four variants also have arms in `eval.rs` (never constant),
  `dump.rs`, `parse.rs` `check_flow_expr`, and `sema.rs` `remap_expr_ident`.
- `Parser::prism_validate(&tu)` renders the walk's diagnostics (no latch).

### 2.3 prism.h (`include/prism.h`)

- Typedefs: float2/3/4, int2/3/4, uint2/3/4 via `vector_size`; `float4x4` =
  struct of 4 `float4` columns; `sampler2D` = complete opaque struct
  `__prism_sampler2d { unsigned int __prism_opaque[2]; }`.
- Builtin prototypes: `tex2D`, scalar `min`/`max`/`sqrtf`, `splat2/3/4`,
  `min2/3/4`, `max2/3/4`, `abs2/3/4`, `mul4x4`.
- Macros (swizzles + calls only): `dot2/3/4`, `length2/3/4`, `normalize2/3/4`,
  `distance2/3/4`, `lerp2/3/4`, `clamp2/3/4`, `saturate_f`, and type-generic
  `dot`/`length`/`normalize`/`distance`/`saturate`/`clamp`/`vmin`/`vmax`/`mul`
  via `__builtin_choose_expr` + `__builtin_types_compatible_p`.
- **Load-bearing constraint**: `__builtin_choose_expr` checks only the selected
  branch (see `check_choose_expr`), *but compound literals are checked eagerly
  at parse time* (found via backtrace: `parse_postfix_expression` →
  `checked_expr` on the CompoundLiteral node). Therefore **no macro reachable
  from a dispatch may build a compound literal** — hence the per-width
  constructors are compiler builtins, not `(float2){...}` macros. Keep this
  invariant when extending prism.h (documented in the header comment).

### 2.4 MIR (`crates/glcc_mir`)

- New `InstKind` (all pure; wired into `for_each_operand`, `is_pure`,
  `remap_inst`, display, parse, verify):
  - `ShuffleVector { lhs: Value, rhs: Value, lanes: SmallVec<[u8; 8]> }` —
    result = `Ty::Vector{elem, lanes.len()}`; lhs must be a Vector; rhs may be
    a Vector (any lane count) **or a scalar** (acts as a 1-lane vector that
    every `≥ lhs.lanes` index selects — that is how `v.xy = 1.0f` is one
    instruction); index `< lhs.lanes` selects lhs, else `rhs` lane
    `i - lhs.lanes`. Text: `%3:v4xf32 = shufflevector %0, %1 [0,1,4]`
    (comma between operands, bracketed lane list). Verified lane bounds.
  - `SampleImage { image: Value, coord: Value }` — `image` is defined by a
    `GlobalAddr` of a resource global (this is why existing liveness/operand
    walkers need no changes). `is_pure() == true`, so CSE/LICM hoist samples
    (legal: shaders cannot write textures). Text: `sampleimage %i, %c`.
  - `Min { a, b }` / `Max { a, b }` — lane-wise float min/max (scalar or
    vector). Text: `min %a, %b`.
- `Ty::vector_shape(elem, lanes)` — f32/i32 × 2/3/4 (prism shapes). `Ty::vector`
  keeps the 128-bit invariant for the vectorizer. **Never call `Ty::vector` for
  prism shapes** (it panics on non-power-of-two lanes).
- Non-exhaustive InstKind matches that consume modules needed guards:
  `glcc_lower_cl/src/lower/inst.rs` panics if prism ops reach a CPU backend
  (correct: prism modules never route there).

### 2.5 Lowering (`crates/glcc_lower_mir`)

- `ty.rs` `lower_ty`: prism mode maps 8/12/16-byte f32/i32 vectors to
  first-class `Ty::vector_shape` (plain C keeps 16-byte-only first-class).
- `lower/vector.rs`: `VecShape.first_class` = the *vector's own* MIR type is a
  `Ty::Vector` (beware: keyed on the vector, not the element — this exact bug
  was caught by the existing 16-byte JIT tests). Dispatch is `!first_class`
  for the bit-pun image paths.
- `lower/expr.rs` new arms (before the `todo!` catch-all):
  - `Swizzle` — single lane → `ExtractLane`; multi → `ShuffleVector(v, v, lanes)`.
  - `SwizzleStore` — place base, load whole vector, mask = (dest lanes →
    `base_lanes + j`, others → `i`), shuffle with rhs, store back; value is
    the combined vector.
  - `PrismSample` — `GlobalAddr{globals[global]}` + `SampleImage`.
  - `Splat` — `InstKind::Splat`.
  - `ShuffleVector` arm has a first-class fast path (out-of-range mask lanes
    clamp to 0; the bit-pun slot path remains for plain-C shapes).
- `lower/builtin.rs`: `lower_prism_builtin(...)` (called from the Call lowering
  when `sema.lang.prism`; user-defined names fall through to normal calls):
  `splat2/3/4` → Splat; `min2..4`/`max2..4` → Min/Max on vectors; `abs2..4` →
  `Max(v, FNeg(v))`; `mul4x4` → per-column `ExtractValue` + dot trees
  (`ExtractLane`/`FMul`/`FAdd` — the dp4-fusion fuel) composed via splats +
  shuffle chain (`[0,5]` → v2, `[0,1,4]` → v3, `[0,1,2,4]` → v4).
  `ScalarBuiltin::{Min, Max, Sqrt}` map plain names `min`/`max`/`sqrt`/`sqrtf`
  under prism.

### 2.6 Tests currently in place

- `crates/glcc_frontend/src/tests/mod.rs`: `prism_errors`/`prism_ok`/
  `prism_rejects` helpers + `PRISM_INTERFACE`/`PRISM_SAMPLER` fixtures + 17
  tests (pointer/goto/recursion/extern/double/variadic/VLA/asm rejections,
  sampler abuse, entry-point presence + semantics, tex2D argument validation,
  swizzle range/member errors, swizzle-store scalar splat success, compound
  swizzle rejection, valid fragment shader, 12-byte vector prism-only).
- `crates/glcc_mir/tests/parse.rs`: 4 tests — new-op parse + verify,
  dump spellings, lane-bounds rejection, scalar-rhs rule.
- Driver e2e (manual, not committed as tests yet): `target/debug/glcc -fprism
  --emit-mir` on a full vertex+fragment shader shows `sampleimage`,
  swizzle stores, Min/Max, first-class shapes. `--emit-*` currently reaches the
  `glcc_prism` stubs (errors "not implemented yet").

## 3. Phase C — `crates/glcc_prism`: PrismIR + ISA text — COMPLETE

Modules: `ir.rs` (PrismIR data model), `interface.rs`, `prep.rs`, `build.rs`
(structurizer + op lowering + fusion), `isa.rs` (printer + parser),
`validate.rs`, `src/tests.rs` (9 tests incl. committed goldens under
`crates/glcc_prism/tests/goldens/`). Dev-dep on `glcc_lower_mir` for
end-to-end C-shader tests; the library deps stay `glcc_mir` + `glcc_frontend`.
`emit()`'s Isa arm builds interface → prep → PrismIR → validate → text;
Spirv/Rsx/Xenos still error "not implemented yet" (Phases E/F/G).

Implementation decisions (deltas from the sketches above, all tested):

- **PrismIR model**: `VReg` + `RegTy {Scalar(F32|Int{bits≤32}), Vector{elem,2..4}, Pred}`
  (predicates share the register arena). `Src` variants: VReg, ConstF32,
  ConstI32, Undef, Input, Uniform (absolute c-reg + optional relative
  index). Bools are i32 0/1; value compares (`cmp.<code>`) yield i32
  0/1 scalars and -1/0 masks for vectors, matching MIR. `FDiv` lowers
  rcp+mul. `Alu` = DX9 set {mov..dp4, cmp, zext/sext/trunc, itof/ftoi,
  bitcast, bitwise, shifts, sel, shuffle} + Mad.
- **SSA joins live in the region statements** (the sketch's parameterless
  regions cannot carry values): `If{pred, then, els, results: [(join,
  then_v, else_v)]}`, `Loop{carried: [(phi, init)], body, results}` —
  conditionless loops, exit = `break(args)`, iteration = `continue(args)`
  (continue args in carried order; exit regs gather one value per break,
  in body order). `Exit(outputs)` rets. No standalone phis.
- **Structurizer**: dominator back-edge loop detection (self-loops
  included) + **branch joins via stop-restricted forward reachability**
  (intersection of each arm's reach, earliest candidate that flows into
  the others). Pure post-dominators miss reentrant joins (a block shared
  by two nested branch levels); the reach form handles those and
  subsumes the pdom cases. Loops whose exit is shared with an enclosing
  join contribute their exit regs as the arm's fall-through value.
  `switch` lowers to a setp/if chain (fallthrough and grouped cases
  rejected with clear errors for now). Loops still holding allocas
  (a `do{}while` at function entry, where Mem2Reg refuses phis) are
  fixed by prep's entry shim below.
- **PVal shapes**: MIR values map to V / Src (inputs, uniform leaves,
  lane extracts — materialized as mov only when a register is required)
  / Agg (flattened interface aggregates; extract/insertvalue walk the
  tree) / Undef / Global (sampler+uniform addresses resolve at their
  use; any other use errors).
- **prep.rs**: shim entry (a fresh empty entry block) only when the
  entry has a back edge (do-while at entry) so Mem2Reg can place phis;
  force-inline everything (noinline cleared), LowerLibcMemory,
  GlobalConstProp, then Sroa/Mem2Reg/ConstFold/Sccp/InstCombine/Cse/
  SimplifyCfg/DeadCodeElim, then profile assertions (no Call/alloca/
  memory ops/atomics/EH survive) + verify. GLCC_PRISM_DEBUG dumps MIR.
- **Fusion** (build.rs): sink single-use movs into operand positions
  (shuffles excluded — lane indices assume source widths; phi positions
  keep registers); dot fusion: add-trees of single-use lane-product muls
  sharing two bases → `dp3`/`dp4` (roots first — reverse statement
  order); mad fusion: single-use mul feeding add/sub → `mad` (sub
  negates the addend; both-sides-mul shapes are left for the dot pass).
- **ISA grammar** (round-trip tested): `shader @name : stage`; dcl lines
  for inputs/outputs/uniforms (leaves as `c<reg>[.lane]=<ty>`)/samplers;
  statements `%r : ty = <op>`, `%r : pred = setp.<code>`, `%r : float4 =
  sample s<n>, <src>`, `break/continue/ret <regs>`; region lines declare
  their joins: `if %p -> %j1, %j2 {` and `loop -> %x {` — this makes the
  trailing `phi` statements unambiguous (a join-less nested if would
  otherwise steal the outer if's phis). Carried phis lead a loop body
  (`[%init, entry], [%v, continue.K]`); if joins follow the if
  (`[%v, then], [%v, else]`); loop exits follow its brace
  (`[%v, break.K]`). Swizzles spell as 1–4 lane chars (`.xyzw`), negate
  as `-`.
- **interface.rs**: inputs flatten one v-register per semantic leaf
  (multiple struct params continue the numbering); uniforms keep member
  paths and get c-reg packing (one global cursor, vectors whole regs,
  scalars 4-per-reg) + std140 sizes/offsets (each uniform its own UBO);
  samplers take slots in declaration order. Sampler detection filters to
  tentative/defined file-scope Vars (prototype parameters of `tex2D` are
  sampler-typed but neither).

### MIR groundwork Phase C required (in `glcc_mir`)

- `Mem2Reg` promotes **vector-typed** slots too (`promotable_ty`:
  scalar ∨ vector) — prism's float2/3/4 locals ride block params.
- `Sroa` splits **whole-aggregate loads/stores** into
  `ExtractValue`/leaf-store and leaf-load/`InsertValue` chains when the
  aggregate's field boundaries agree with the slot's other accesses
  (punned/sub-leaf shapes stay in memory). This is what lets interface
  structs flow through registers.
- `GlobalConstProp` folds a whole-aggregate load of a read-only global
  into a per-leaf constant reassembly (`rebuild_const_aggregate`) —
  SROA splitting the destination copy previously hid the field loads
  that used to fold.
- `DomInfo`, `DomInfo::compute/dominates`, and `reverse_postorder` are
  public (the structurizer reuses them); `display::inst_str` public.


### 3.1 `interface.rs`
`PrismInterface`, built from the validated `&Sema` + `&Interner`:
- stage(s) present, per entry function: inputs (params, flattened interface
  structs: leaf semantic + FE type) and outputs (return struct leaves);
- uniforms: file-scope globals with `[[prism::uniform]]`, with BOTH layout
  tables computed once — c-register packing (float4 = 1 c-reg; scalars packed
  4-per-reg; float4x4 = 4 c-regs) and std140;
- samplers: file-scope `is_prism_sampler_ty` globals, bindings auto-assigned
  in declaration order.
Entry functions must survive `DeadModuleItems` (they are non-static → external
→ kept; the checker rejects `static` entry points).

### 3.2 `ir.rs` — PrismIR
- `VReg(u32)` + `RegTy { F32, I32 } × width 1..=4` (width-1 = scalar slot; on
  Xenos these classify into the scalar ALU).
- Ops: `Op { dst: VReg, kind: Alu, srcs: SmallVec<[Src; 3]> }` with
  `Alu ∈ {mov, add, sub, mul, mad, dp3, dp4, rcp, rsq, min, max, slt, sge,
  frc, exp, log, shuffle}`; `Src { vreg | const f32 | uniform(reg, rel:
  Option<VReg>), swizzle: [u8; 4]-subset, negate }`; `Tex { dst, sampler,
  coord, lod: Implicit | Explicit(VReg) }`; `Cmp { pred, cmp: Le|Lt|Ge|Gt|Eq|Ne,
  a, b }` (predicates feed If/Loop; both DX9 `if_gt` fusion and ps
  `setp_comp`+`if_pred` map onto it).
- Regions: `Region::{Seq(Vec<Op>), If { pred, then, els }, Loop { body },
  Break, Continue, Call(FuncId), Exit}`. Regions are shared with SPIR-V
  (`OpSelectionMerge`/`OpLoopMerge`) and the ISA (`if/else/endif`, `rep/
  endrep`, `break`).
- Partial updates stay explicit shuffles (no masked dst writes — SSA stays
  SSA). Fusion peepholes live here or in `build.rs`: FMul+FAdd → `mad`;
  dot-shaped FAdd trees of lane extracts → `dp4`/`dp3` (mul4x4's output is
  exactly this shape).
- Textual form (`.isa`): define a small grammar, printer + parser, **round-trip
  tested** (unlike MIR's dump, which is diagnostic-only — do not copy that
  shortcoming). This is the golden-test format.

### 3.3 `build.rs` — MIR + interface → PrismIR
- Structurize the MIR CFG once: post-dominator tree (dominators over the
  reversed CFG — model on `glcc_mir::passes::DomInfo`, `passes/mod.rs:~2461`),
  back-edge loop detection, break/continue edges. The profile guarantees
  reducibility (no goto).
- Map MIR ops → PrismIR ops (widths from operand types), SSA vregs from MIR
  values; `ExtractLane`/`ShuffleVector` → PrismIR src-swizzles/shuffle op;
  `SampleImage` → `Tex`; `Min`/`Max` → min/max; float compare → `Cmp`.
- Run the fusion peepholes (mad, dp4/dp3).
- `validate.rs`: final-MIR profile assertions before building (no
  CallIndirect/InlineAsm/Atomic*/Va*/EH; `SampleImage.image` is a `GlobalAddr`
  of a resource global; every global the interface knows about matches).

### 3.4 `isa.rs` — the `.isa` printer (+ parser)
DX9-class virtual machine text. Registers are VIRTUAL in this form (targets do
their own RA); the text is the IR's canonical form and the golden format.
Sketch (final grammar is yours to finalize, but keep it round-trippable and
documented in `docs/prism-isa.md`):

```
; prism ISA — vertex entry vs_main
vs.main:
  dcl.input v0.position : float4     ; from interface (semantic → binding)
  dcl.output o0 : float4
  dcl.uniform c0..c3 : float4 mvp
  dcl.sampler s0 : sampler2D diffuse
  ...
  dp4  t0.x, c0, v0
  mov  o0, t0
  ret
```

Driver wiring: replace the `ShaderTarget::Isa` arm of `glcc_prism::emit` with
build + print; return text bytes. Golden tests: C source → committed `.isa`
output (in `crates/glcc_prism/tests/`), plus PrismIR parse/print round-trip
unit tests, plus structurizer unit tests (diamonds, loops with break/continue,
nested).

## 4. Phase D — shared instruction scheduling

`schedule.rs` in `glcc_prism`: per-`Seq` list scheduling on PrismIR.
- `SchedModel { alu: 1, texld: 8 }` cycles; DAG from SSA deps (`for_each_operand`
- style walk over PrismIR ops); `Tex` ops hoisted as early as legal, keeping
  relative order among themselves (dependent-read addressing); greedy fill of
  latency windows with ready ALU (critical-path priority); live-pressure guard
  against the temp budget (no spilling — the cap is hard, "out of temporary
  registers" is a driver-renderable error via `glcc_prism::emit`'s Err).
- Unit test proving ALU work lands between a texld and its first use; goldens
  updated.
- SPIR-V/Xenos/RSX all render the scheduled order.

## 5. Phase E — SPIR-V backend

`spirv.rs`: hand-rolled binary writer (word stream, ID allocator, type/const
cache) — no external deps, mirroring how glcc_dwarf hand-rolls DWARF.
- Header (magic 0x07230203), `OpCapability Shader`, `OpMemoryModel Logical
  GLSL450`; `OpEntryPoint Vertex|Fragment` with interface vars.
- Interface: in/out vars with `Location`/`BuiltIn` decorations (POSITION →
  BuiltIn Position etc.); uniforms → UBO (`Block`, `DescriptorSet`/`Binding`,
  std140 from interface.rs); samplers → `OpTypeSampledImage` + bindings.
- PrismIR regions → `OpSelectionMerge`/`OpLoopMerge`/`OpBranch*`; ops map 1:1
  (`Shuffle` → `OpVectorShuffle`, `Tex` → `OpImageSampleImplicitLod` (fragment)
  / `OpImageSampleExplicitLod` Lod=0 (vertex), `Min`/`Max` → `OpFMin`/`OpFMax`,
  mad → OpFMul+OpFAdd or fused if you add Fma; helpers → `OpFunctionCall`).
- vec2/vec3 map natively (the reason PrismIR is width-typed).
- Validation: internal structural self-check (word counts, ID bounds,
  type/const consistency) + optional `spirv-val` differential test with the
  have-tool skip pattern (`crates/glasm/tests/common/mod.rs`).

## 6. Phase F — RSX (PS3) backend

`rsx.rs`: PrismIR → nv40-family vertex (vp) and fragment (fp) program text,
syntax pinned to community-documented formats (envytools). Inline all calls at
lowering (nv40 fp has no subroutine calls). RA to R (fp32) banks — H (half)
deferred. Uniforms → C# constant registers via the c-reg table. `Tex` → TEXLD
(fragment) / explicit-LOD fetch (vertex). Region tree → IF/ELSE/ENDIF,
REP/ENDREP/BRK, predication. Output `.rsx` text + goldens; integration with
external RSX assembler tools behind have-tool skips when present.

## 7. Phase G — Xenos (Xbox 360) backend

`xenon.rs`: PrismIR → Xenos microcode assembly text. Width-1 float ops in the
scalar set (rcp/rsq/exp/log + scalar add/mul/mad) → scalar ALU slots; width ≥ 2
→ vector ALU; fetch ops (vertex fetch / texture fetch); constant registers
c#/cb#/cf#; region tree → Xenos branching/predication (p0). Co-issue pairing
on top of the shared schedule. Goldens; text format pinned to homebrew
microcode tooling conventions.

## 8. Phase H — polish

- `docs/prism.md` (language profile, attributes, prism.h reference, driver
  flags, target matrix, the compound-literal/eager-check constraint),
  `docs/prism-isa.md` (PrismIR text spec), per-target notes; AGENTS.md
  crate-map + orientation notes.
- `examples/prism/`: 2–3 shaders + Makefile per `examples/README.md`
  conventions (`make` emits `.isa`, `.spv`, `.rsx`, `.xenos`; `make clean`
  works).
- Optional stretch: glcc_explorer PrismIR panel; promote `dot` from macro to
  builtin if dp4 fusion proves fragile.

## 9. Verification expectations (per phase)

- `cargo build` after every edit (AGENTS.md); per-crate tests:
  `cargo test -p glcc_prism` (new), `-p glcc_frontend`, `-p glcc_mir`,
  `-p glcc_lower_mir`, full `cargo test --workspace` before declaring done.
- End-to-end: compile `examples/prism` shaders with each `--emit-*` flag; diff
  ISA against committed goldens; `spirv-val` when available; `--emit-mir` to
  inspect the pipeline mid-way.
- The driver binary is `target/debug/glcc`; headers stage via root `build.rs`
  into `~/.local/lib/glcc/include` (touch `build.rs` or rebuild the root
  package after editing `include/prism.h` if staging seems stale).
- Error messages surfaced by `glcc_prism::emit`'s `Err(String)` print as
  `glcc: error: <msg>`; frontend diagnostics render via the parser (prism
  validation runs inside `parse_source` while the parser is alive).

## 10. Gotchas learned the hard way (do not rediscover)

1. Compound literals are checked eagerly at parse time — they cannot ride
   inside `__builtin_choose_expr` dispatch branches (see §2.3).
2. `Ty::vector` enforces the 128-bit invariant; prism shapes must use
   `Ty::vector_shape`. A stray `Ty::vector(..., 2)` panics at runtime.
3. `VecShape::first_class` keys on the *vector's* MIR type, not the element's
   (breaking this breaks the existing 16-byte vector JIT tests — they caught
   it once already).
4. A file-scope `sampler2D x;` is a *tentative definition* (`defined == true`),
   so sampler-initializer checks must look at `sema.inits`, not `defined`.
5. Single-lane swizzle reads (`v.x`) are scalars in the FE; multi-lane are
   vectors. Swizzle stores use the MIR shuffle's scalar-rhs rule.
6. `dump_module` (MIR) is diagnostic-only and does not round-trip; the `.isa`
   printer must round-trip (it is the golden format).
7. prism modules never reach `glcc_lower_cl`/`glcc_codegen` — the CPU backends
   panic on prism ops by design (guards in `lower_cl/src/lower/inst.rs`).
8. AGENTS.md conventions apply: small steps, build after every edit, tests
   inside the owning crate, no broad filesystem searches, `#[should_panic]`
   substrings are API.
