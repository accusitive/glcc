#!/usr/bin/env bash
# ctestsuite.bash — run the c-testsuite single-exec tests through glcc.
#
# Pass criterion (the suite's runner contract, see
# tests/c-testsuite/README.md): the compiled program exits 0 and its
# stdout+stderr matches the $t.c.expected file.
#
# Layout in tests/c-testsuite/tests/single-exec:
#   *.c            single-file tests (main is the entry point)
#   *.c.expected   expected stdout+stderr
#   *.c.tags       suite metadata (portability/arch tags; unused here)
#
# glcc is invoked as `glcc -w $CFLAGS_EXTRA <t>.c -lm -o <t>`: -lm is
# linked because the suite's math tests call libm, which gcc's
# reference runner only escapes through its -O2 constant folding.
#
# Targets: x86_64 runs natively; aarch64 and riscv64 cross-compile with
# `--target=`, link statically through tools/zig-cc (zig's musl libc),
# and run under qemu-$t-static. Cross
# targets need zig, clang (compiles rt/glccrt.c for the target) and qemu-user; a
# target whose tooling is missing is skipped with a message.
#
# Usage:
#   ./tools/ctestsuite.bash [-j N] [--compile-timeout S] [--timeout S] [--no-build]
#                           [--targets=LIST] [--keep-tmp] [-v] [pattern ...]
#
#   pattern filters by basename substring (e.g. ./tools/ctestsuite.bash 0003)
#   --targets=LIST selects targets from x86_64,aarch64,riscv64
#                  (default: all three)
#
# Env overrides: GLCC (default target/debug/glcc), CFLAGS_EXTRA,
# ZIG_CC (default tools/zig-cc).

set -u
shopt -s nullglob

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="$ROOT/tests/c-testsuite/tests/single-exec"
GLCC="${GLCC:-$ROOT/target/debug/glcc}"
ZIG_CC="${ZIG_CC:-$ROOT/tools/zig-cc}"

JOBS="$(nproc 2>/dev/null || echo 8)"
CTIMEOUT=30
RTIMEOUT=5
BUILD=1
KEEP=""
VERBOSE=""
TARGETS_OPT="x86_64,aarch64,riscv64"
PATTERNS=()

while (($#)); do
  case "$1" in
    -j|--jobs) JOBS="$2"; shift 2;;
    -j*) JOBS="${1#-j}"; shift;;
    --compile-timeout) CTIMEOUT="$2"; shift 2;;
    --timeout) RTIMEOUT="$2"; shift 2;;
    --no-build) BUILD=0; shift;;
    --targets=*) TARGETS_OPT="${1#--targets=}"; shift;;
    --keep-tmp) KEEP=1; shift;;
    -v|--verbose) VERBOSE=1; shift;;
    -h|--help) sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# //;s/^#//'; exit 0;;
    -*) echo "unknown option: $1 (see --help)" >&2; exit 2;;
    *) PATTERNS+=("$1"); shift;;
  esac
done

if ((BUILD)); then
  echo "building glcc..."
  (cd "$ROOT" && cargo build 2>&1 | tail -2) || exit 1
fi
[[ -x "$GLCC" ]] || { echo "error: glcc binary not found: $GLCC" >&2; exit 1; }
command -v timeout >/dev/null || { echo "error: 'timeout' not found" >&2; exit 1; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/glcc-ctestsuite-XXXXXX")"
BIN="$TMP/bin"; LOG="$TMP/log"; RES="$TMP/res"
mkdir -p "$BIN" "$LOG" "$RES"
if [[ -z "$KEEP" ]]; then
  trap 'rm -rf "$TMP"' EXIT
else
  echo "tmpdir: $TMP"
fi

CANDIDATES="$TMP/candidates.txt"
for f in "$SUITE"/*.c; do printf '%s\n' "$f"; done > "$CANDIDATES"

# Pattern filter: basename must contain one of the patterns.
if ((${#PATTERNS[@]})); then
  FILTERED="$TMP/filtered.txt"
  : > "$FILTERED"
  while IFS= read -r src; do
    base="$(basename "$src")"
    for p in "${PATTERNS[@]}"; do
      if [[ "$base" == *"$p"* ]]; then printf '%s\n' "$src" >> "$FILTERED"; break; fi
    done
  done < "$CANDIDATES"
  mv "$FILTERED" "$CANDIDATES"
fi

TOTAL=$(wc -l < "$CANDIDATES")
((TOTAL)) || { echo "no tests selected"; exit 0; }

# Target selection. The native target always runs; a cross target needs
# its qemu, clang to compile the runtime, and the zig link driver.
ACTIVE_TARGETS=()
for t in ${TARGETS_OPT//,/ }; do
  case "$t" in
    x86_64)
      ACTIVE_TARGETS+=("$t")
      ;;
    aarch64|riscv64)
      qemu="qemu-$t-static"
      command -v "$qemu" >/dev/null 2>&1 || qemu="qemu-$t"
      if ! command -v "$qemu" >/dev/null 2>&1; then
        echo "skip $t: $qemu not found"
        continue
      fi
      if ! command -v clang >/dev/null 2>&1; then
        echo "skip $t: clang not found (compiles rt/glccrt.c)"
        continue
      fi
      if ! command -v zig >/dev/null 2>&1 || [[ ! -x "$ZIG_CC" ]]; then
        echo "skip $t: zig or $ZIG_CC not found"
        continue
      fi
      if ! clang --target="$t-linux-gnu" -c "$ROOT/rt/glccrt.c" -o "$TMP/glccrt-$t.o" 2>"$TMP/glccrt-$t.err"; then
        echo "skip $t: cannot compile the runtime:"
        sed -n '1,3p' "$TMP/glccrt-$t.err"
        continue
      fi
      ACTIVE_TARGETS+=("$t")
      ;;
    *)
      echo "unknown target: $t (expected x86_64, aarch64, riscv64)" >&2
      exit 2
      ;;
  esac
done
((${#ACTIVE_TARGETS[@]})) || { echo "no targets selected"; exit 0; }

echo "running $TOTAL tests on ${ACTIVE_TARGETS[*]} (jobs=$JOBS, compile-timeout=${CTIMEOUT}s, run-timeout=${RTIMEOUT}s)"

run_one() {
  local t="$1" src="$2"
  local base name bin log res expected label
  base="$(basename "$src")"
  name="${base%.c}"
  bin="$BIN/$t-$name"
  log="$LOG/$t-$name.log"
  res="$RES/$t-$name.result"
  expected="$src.expected"
  label="$t/$base"
  local flag="" extra="" qemu="" cc_env=()
  local cc_status run_status

  if [[ "$t" != x86_64 ]]; then
    qemu="qemu-$t-static"
    command -v "$qemu" >/dev/null 2>&1 || qemu="qemu-$t"
    flag="--target=$t"
    extra="-static"
    cc_env=("GLCC_CC=$ZIG_CC" "GLCC_RT=$TMP/glccrt-$t.o")
  fi

  # shellcheck disable=SC2086
  env "${cc_env[@]}" timeout "$CTIMEOUT" "$GLCC" -w $CFLAGS_EXTRA $flag "$src" -lm -o "$bin" $extra >"$log" 2>&1
  cc_status=$?
  if ((cc_status == 124)); then
    echo "COMPILE_TIMEOUT $label" > "$res"
    echo "COMPILE_TIMEOUT $label"
    return
  elif ((cc_status != 0)); then
    echo "COMPILE_FAIL $label (cc=$cc_status)" > "$res"
    echo "COMPILE_FAIL $label (cc=$cc_status)  log: $log"
    return
  fi

  if [[ -n "$qemu" ]]; then
    timeout "$RTIMEOUT" "$qemu" "$bin" >"$log.run" 2>&1
  else
    timeout "$RTIMEOUT" "$bin" >"$log.run" 2>&1
  fi
  run_status=$?
  if ((run_status == 124)); then
    echo "RUN_TIMEOUT $label" > "$res"
    echo "RUN_TIMEOUT $label"
    return
  elif ((run_status != 0)); then
    echo "RUN_FAIL $label (exit=$run_status)" > "$res"
    echo "RUN_FAIL $label (exit=$run_status)  log: $log"
    return
  fi

  if ! diff -u "$expected" "$log.run" >"$log.diff" 2>&1; then
    echo "OUTPUT_FAIL $label" > "$res"
    echo "OUTPUT_FAIL $label  log: $log  diff: $log.diff"
    return
  fi

  echo "PASS $label" > "$res"
  [[ -n "$VERBOSE" ]] && echo "PASS $label"
}
export -f run_one
export BIN LOG RES GLCC CTIMEOUT RTIMEOUT VERBOSE ZIG_CC TMP CFLAGS_EXTRA="${CFLAGS_EXTRA:-}"

JOBS_FILE="$TMP/jobs.txt"
: > "$JOBS_FILE"
for t in "${ACTIVE_TARGETS[@]}"; do
  while IFS= read -r src; do printf '%s %s\n' "$t" "$src" >> "$JOBS_FILE"; done < "$CANDIDATES"
done
xargs -P "$JOBS" -n 2 bash -c 'run_one "$@"' _ < "$JOBS_FILE"

# Aggregate, one line per target.
FAIL_LIST="$TMP/failures.txt"; : > "$FAIL_LIST"
grand_fail=0
echo "----------------------------------------"
for t in "${ACTIVE_TARGETS[@]}"; do
  pass=0; cfail=0; ctimeout=0; rfail=0; rtimeout=0; ofail=0
  for r in "$RES/$t-"*.result; do
    status=$(cut -d' ' -f1 "$r")
    case "$status" in
      PASS) pass=$((pass+1));;
      COMPILE_FAIL) cfail=$((cfail+1)); cat "$r" >> "$FAIL_LIST";;
      COMPILE_TIMEOUT) ctimeout=$((ctimeout+1)); cat "$r" >> "$FAIL_LIST";;
      RUN_FAIL) rfail=$((rfail+1)); cat "$r" >> "$FAIL_LIST";;
      RUN_TIMEOUT) rtimeout=$((rtimeout+1)); cat "$r" >> "$FAIL_LIST";;
      OUTPUT_FAIL) ofail=$((ofail+1)); cat "$r" >> "$FAIL_LIST";;
    esac
  done
  printf '%s: total: %d  pass: %d  compile-fail: %d  compile-timeout: %d  run-fail: %d  run-timeout: %d  output-fail: %d\n' \
    "$t" "$TOTAL" "$pass" "$cfail" "$ctimeout" "$rfail" "$rtimeout" "$ofail"
  grand_fail=$((grand_fail + cfail + ctimeout + rfail + rtimeout + ofail))
done
if [[ -s "$FAIL_LIST" ]]; then
  echo "--- failures ---"
  sort "$FAIL_LIST"
  cp "$FAIL_LIST" "$ROOT/ctestsuite-failures.txt"
  echo "(full list also saved to $ROOT/ctestsuite-failures.txt)"
fi
if [[ -n "$KEEP" ]]; then
  echo "logs kept in $LOG"
fi

(( grand_fail )) && exit 1 || exit 0
