#!/usr/bin/env bash
# End-to-end inline-asm tests for the native backend (`-fbackend=native
# -fintegrated-as`): each $t.c must print exactly what its $t.expected
# holds (generated from gcc). Cross-compiling/linking for x86-64 on
# another host goes through x86_64-linux-gnu-gcc + qemu-user, matching
# tools/ctestsuite.bash.
#
#   tests/native_asm/run.sh            run all tests
#   tests/native_asm/run.sh -v         also list passes
#   GLCC=... CFLAGS_EXTRA="-O2" tests/native_asm/run.sh
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GLCC="${GLCC:-$ROOT/target/debug/glcc}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if ! [[ -x "$GLCC" ]]; then
  echo "glcc binary not found at $GLCC (build with cargo build)" >&2
  exit 1
fi

x86_64="x86_64"
if [[ "$(uname -m)" == x86_64 ]]; then
  qemu=""
  qemu_flags=()
else
  qemu="qemu-$x86_64-static"
  command -v "$qemu" >/dev/null 2>&1 || qemu="qemu-$x86_64"
  if ! command -v "$qemu" >/dev/null 2>&1; then
    echo "skip: $qemu not found" >&2
    exit 0
  fi
  [[ "$qemu" == *-static ]] || qemu_flags=(-L "/usr/$x86_64-linux-gnu")
fi

# The runtime is the one build.rs staged (matches the host, but the
# driver compiles what it needs for the target; on a non-x86_64 host the
# cross gcc builds it).
GLCC_RT="$TMP/glccrt-x86_64.o"
if [[ "$(uname -m)" == x86_64 ]]; then
  cp ~/.local/lib/glcc/glccrt.o "$GLCC_RT" 2>/dev/null || GLCC_RT=""
else
  x86_64-linux-gnu-gcc -c "$ROOT/rt/glccrt.c" -o "$GLCC_RT" || exit 1
fi

VERBOSE=""
[[ "${1:-}" == "-v" ]] && VERBOSE=1

pass=0
fail=0
for src in "$ROOT"/tests/native_asm/*.c; do
  base="$(basename "$src" .c)"
  expected="$ROOT/tests/native_asm/$base.expected"
  bin="$TMP/$base"
  log="$TMP/$base.log"
  cc_env=()
  [[ -n "$GLCC_RT" ]] && cc_env=("GLCC_CC=${CROSS_CC:-x86_64-linux-gnu-gcc}" "GLCC_RT=$GLCC_RT")
  [[ "$(uname -m)" == x86_64 ]] && cc_env=()

  # shellcheck disable=SC2086
  status=0; env "${cc_env[@]}" timeout 60 "$GLCC" -w ${CFLAGS_EXTRA:-} --target=$x86_64 -fbackend=native -fintegrated-as "$src" -lm -o "$bin" >"$log" 2>&1
  if ((status != 0)); then
    echo "COMPILE_FAIL $base  log: $log"
    fail=$((fail + 1))
    continue
  fi
  if [[ -n "$qemu" ]]; then
    status=0; timeout 60 "$qemu" "${qemu_flags[@]}" "$bin" >"$log.run" 2>&1
  else
    status=0; timeout 60 "$bin" >"$log.run" 2>&1
  fi
  if ((status != 0)); then
    echo "RUN_FAIL $base  log: $log.run"
    fail=$((fail + 1))
    continue
  fi
  if ! diff -u "$expected" "$log.run" >"$log.diff" 2>&1; then
    echo "OUTPUT_FAIL $base  diff: $log.diff"
    fail=$((fail + 1))
    continue
  fi
  [[ -n "$VERBOSE" ]] && echo "PASS $base"
  pass=$((pass + 1))
done

echo "$pass passed, $fail failed"
((fail == 0))
