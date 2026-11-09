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

`sqlite3-native` (native backend, -O0 build) runs `--version` fine but
segfaults executing any SQL. Fault context (qemu -d cpu): PC inside
`yy_reduce` (`mov -0x4c0(%rbp),%r8` at ~0x682039 in the -O0 binary),
rbp/rsp looked mapped — the dump was ambiguous. yy_reduce is a giant
lemon-parser function: huge switch (comparison chain via
`term_hook`'s `Terminator::Switch` + hundreds of critical-edge splits)
+ a very large frame. Verified NOT the switch itself (small/medium
switch repro matches gcc exactly, `/tmp/native-test/bigswitch.c`).

Debugging leads that went nowhere: qemu `-d in_asm/cpu` (CR2 never
filled), gdb via `-g` stub (interrupted by VM issues). Next ideas:

1. `gdb-multiarch` + `qemu-x86_64 -g 1234`, `set architecture
   i386:x86-64`, catch the real fault + backtrace.
2. Try `-O2` native build (was compiling when session died —
   `/tmp/sqlite-dl/logs/o2.log`) — may not crash, narrows the pass.
3. Extract `yy_reduce` + minimal driver into a standalone TU and
   bisect: frame size? switch chain? critical-edge forwarding?
4. Suspect areas: `frame::expand` with huge frames, switch
   fallthrough wiring, or the parallel-move runs in a function with
   thousands of calls.

## Notes

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
