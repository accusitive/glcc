# Handoff: native x86-64 backend — WIP state (2026-11-09)

Working from an **aarch64 host**: everything x86_64 runs cross via
qemu. The established loop (also see `tools/ctestsuite.bash`, which
does this automatically):

```bash
cargo build --release          # debug codegen is too slow for big TUs
GLCC_CC=x86_64-linux-gnu-gcc \
GLCC_RT=/tmp/native-test/rt/glccrt-x64.o \
  ./target/release/glcc -w --target=x86_64 -fbackend=native FILE.c -o out -lm
qemu-x86_64 -L /usr/x86_64-linux-gnu ./out
```

The x86_64 runtime object is built once via (see build.rs `stage_elf` /
`combine_softfloat` for the exact flags — CC_FLAGS + the 11 `__glcc_*`
renames, 12 TUs, combined with `x86_64-linux-gnu-ld -r`):

```bash
mkdir -p /tmp/native-test/rt && cd /tmp/native-test/rt
# compile rt/glccrt.c + rt/softfloat/{addtf3,subtf3,multf3,divtf3,
# extendxftf2,trunctfxf2,floattixf,floatuntixf,fixxfti,fixunsxfti,
# clzti2,glcc-fe}.c with:
x86_64-linux-gnu-gcc -c -Os -fPIC -ftls-model=initial-exec \
  -fno-asynchronous-unwind-tables -fno-stack-protector \
  -I<repo>/rt/softfloat -D__addtf3=__glcc_addtf3 [...] <src> -o <tu>.o
x86_64-linux-gnu-ld -r glccrt-main.o <softfloat objs> -o glccrt-x64.o
```

## State

- c-testsuite x86_64: **cranelift 220/220, native 219/220** (runner
  handles the cross setup on non-x86_64 hosts).
- Native backend gap: 00204 va_arg of a float-containing struct
  (`glcc_codegen/src/x64/mod.rs` `VaArg` panics; SysV passes these in
  xmm via the register save area).
- **SQLite amalgamation compiles+links with BOTH backends** under the
  flags `-DSQLITE_THREADSAFE=0 -DSQLITE_DISABLE_INTRINSIC`
  (the second flag avoids `__uint128_t`, unsupported in the native
  backend). Sources in `/tmp/sqlite-dl/sqlite-amalgamation-3540100`
  (re-download: `sqlite-amalgamation-3540100.zip`, tests/sqlite/ has
  the URL; scripts in /tmp/sqlite-dl/*.sh).
- cranelift sqlite3 works under qemu (`select 1+2` → 3, insert/select
  roundtrip OK).
- **native sqlite3 crashes in yy_reduce (SQL parsing)** — this is the
  open bug, see below.

## The open bug (next agent starts here)

**2026-11-10 update (x86_64 host, no qemu needed): three real
miscompiles found and fixed — commit 5b64d00, pushed.**

1. **Struct-arg pin clobber** (`entry()` in `x64/mod.rs`): 16-byte
   by-value struct args were stored from their ABI registers
   directly; a sibling entry pin whose vreg the allocator placed in
   that same register executed first (`Reg::P` operands are
   invisible to regalloc2). sqlite3Parser's `yyminor` Token arrived
   as `{z=yymajor, n}`. Fixed by pinning each eightbyte piece with a
   MOVE that joins the entry parallel-move run, resolved by
   `resolve_entry_pins` post-RA. Reduced repro shape: callee with
   `(ptr, int, struct{ptr,i64}, ptr)` params.
2. **Missing 0x66 prefix** (`enc.rs`): W16 register/register and
   reg/imm forms encoded as 32-bit ops — `emit_mem` had the prefix,
   `emit_rm` did not. A `mov $0,%ecx`'s imm32 then swallowed the
   following `cmp`'s bytes (`c7 c1 00 00 39 c8`), so `nCol != 0`
   guards tested garbage flags. Now `emit_rm` emits 66 for W16
   (covers Rm/Rm2/GrpI/Grp0/ShiftI/Cmov), plus PlusR/PlusR2/PlusRI
   and MOVE's three integer arms; imul W16 takes imm16.
3. **Variable shifts never in CL** (`tables.rs`): shl/shr/sar `.cl`
   specs used `alu_rr()` whose count slot is an ordinary register;
   the encoder hardcodes `%cl` and nothing moved the count there →
   shift-by-0 loops. Fixed with `alu_cl()` = `Constraint::Fixed(RCX)`
   on the count slot (same mechanism div uses for RAX/RDX).

After the fixes (native -O0 build, ~2 min compile): `--version` OK,
`SELECT` with arithmetic/string functions OK, ctestsuite native still
219/220 (run-fail 0, 00204 compile-fail unchanged).

**Remaining bug**: `CREATE TABLE t(a,b);` fails — usually
`Parse error near line 1: unrecognized token: "#"` (error message
itself assembled from garbage memory), sometimes SIGSEGV in the
`sqlite3VdbeExec` region (huge mislabeled function; frames at
`+0x144900` / caller `+0x1d303c` in the 2026-11-10 core), reading
`[ptr+0x14]` through an unmapped pointer loaded from a deep spill
slot (`-0xef8(%rbp)`). Nondeterministic: never crashes under gdb
(layout shift), so capture cores via systemd-coredump (`coredumpctl
dump <pid> -o file`) and post-mortem with `gdb BINARY CORE`.

Earlier manifestation (pre-shift-fix), same area: during the
internal sqlite_schema bootstrap parse
(`CREATE TABLE x(type text,name text,...)` — visible at
sqlite3RunParser entry, call 2), `sqlite3ColumnIndex` resolved
"type" against sqlite_master with `aCol=NULL, nCol=0` because an
`nCol != 0` guard passed on garbage — that guard is now correct.

**Debugging techniques that worked on this host**:
- Plain gdb (ASLR off): break at absolute `0x555555554000+offset`
  (nm gives true VAs for PIE), `x/…i $rip`, walk `[rbp]/[rbp+8]`
  manually — the rbp chain is intact even when gdb's backtrace
  is confused.
- Hardware watchpoints on stack spill slots (e.g.
  `watch *(long*)(yy_reduce_rbp-0x4c0)`) to find who clobbers a
  value; on `sEngine.yytos` to rule the parser state in/out.
- Differential: `gcc -O0 -g` build of the same amalgamation as
  ground truth; `--emit-mir` and `GLCC_NATIVE_DUMP=1 glcc -c`
  for before/after-RA machine code.
- nm/gdb local-symbol names in this binary are shifted (e.g.
  "sqlite3TableColumnToStorage+8744" is really other statics);
  identify functions by their call targets, not names.
- Reduced repro `repro.c`-style tests (struct arg shapes, shifts,
  16-bit compares) run in <1 s — bisect at the backend level.

## Older notes (pre-2026-11-10, aarch64 VM era)

`sqlite3-native` (native backend, -O0 build) segfaulted executing
any SQL. Fault context: PC inside `yy_reduce`, huge switch chain
suspected. Verified NOT the switch itself (`/tmp/native-test/
bigswitch.c` matched gcc). The real cause turned out to be the
struct-arg pin clobber above — yy_reduce is just where the corrupt
tokens were consumed.

## Notes

- 2026-11-10 host: native x86_64 (no qemu); 9 commits pushed to
  origin (push works on this machine). Runtime staged at
  `~/.local/lib/glcc/glccrt.o` by cargo build (includes the
  `__glcc_*` softfloat renames) — no manual GLCC_RT needed.
- VM drops connections and kills processes — run long compiles
  `setsid nohup ... < /dev/null &` with logs under /tmp, poll the log
  file instead of waiting.
- OOM: the native backend peaked 27 GB RSS on sqlite3_open_v2 (-O2)
  once; ulimit -v 12000000 in the /tmp/sqlite-dl scripts guards this.
- Suite scripts: `CFLAGS_EXTRA="-fbackend=native" ./tools/ctestsuite.bash
  --targets=x86_64` (GLCC env var picks the binary; it defaults to
  target/debug/glcc — **rebuild debug after edits or results mislead**).
- 9 commits ahead of origin/master; push needs credentials this VM
  doesn't have — `git push origin master` from the usual machine.
- AGENTS.md has an uncommitted local edit by the user — leave it.
