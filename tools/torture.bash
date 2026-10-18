#!/usr/bin/env bash
# torture.bash — run the gcc-c-torture execute tests through glcc.
#
# Pass criterion (matches the DejaGNU `run` default): the compiled
# program's exit status is 0. Anything else — compile failure,
# nonzero exit, signal death, timeout — is a failure.
#
# Layout in tests/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute:
#   *.c            single-file tests:  glcc -w foo.c -o foo && ./foo
#   ieee/*.c       same, single-file (FP edge cases; .x TCL flags ignored)
#   builtins/*.c  two/three-file: foo.c + foo-lib.c + lib/main.c (main_test
#                  harness; the .c file alone has no main). .x TCL ignored.
#
# Usage:
#   ./tools/torture.bash [-j N] [--compile-timeout S] [--timeout S] [--no-build]
#                        [--keep-tmp] [--all | --ieee | --builtins] [-v] [pattern ...]
#
#   pattern filters by basename substring
#   (e.g. ./tools/torture.bash pr49768 20000112). Default suite is the
#   top-level execute/*.c; --ieee / --builtins add those suites, --all
#   adds both. -v prints PASS lines too (default prints only failures).
#
# Env overrides: GLCC (default target/debug/glcc), CFLAGS_EXTRA.

set -u
shopt -s nullglob

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="$ROOT/tests/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute"
GLCC="${GLCC:-$ROOT/target/debug/glcc}"

JOBS="$(nproc 2>/dev/null || echo 8)"
CTIMEOUT=30
RTIMEOUT=5
BUILD=1
KEEP=""
VERBOSE=""
WANT_TOP=1
WANT_IEEE=0
WANT_BUILTINS=0
PATTERNS=()

while (($#)); do
  case "$1" in
    -j|--jobs) JOBS="$2"; shift 2;;
    -j*) JOBS="${1#-j}"; shift;;
    --compile-timeout) CTIMEOUT="$2"; shift 2;;
    --timeout) RTIMEOUT="$2"; shift 2;;
    --no-build) BUILD=0; shift;;
    --keep-tmp) KEEP=1; shift;;
    -v|--verbose) VERBOSE=1; shift;;
    --all) WANT_IEEE=1; WANT_BUILTINS=1; shift;;
    --ieee) WANT_IEEE=1; WANT_TOP=0; shift;;
    --builtins) WANT_BUILTINS=1; WANT_TOP=0; shift;;
    --top) WANT_TOP=1; shift;;
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

TMP="$(mktemp -d "${TMPDIR:-/tmp}/glcc-torture-XXXXXX")"
BIN="$TMP/bin"; LOG="$TMP/log"; RES="$TMP/res"
mkdir -p "$BIN" "$LOG" "$RES"
if [[ -z "$KEEP" ]]; then
  trap 'rm -rf "$TMP"' EXIT
else
  echo "tmpdir: $TMP"
fi

# Collect candidate tests as "suite|src" lines.
CANDIDATES="$TMP/candidates.txt"
: > "$CANDIDATES"
if ((WANT_TOP)); then
  for f in "$SUITE"/*.c; do printf 'top|%s\n' "$f"; done >> "$CANDIDATES"
fi
if ((WANT_IEEE)); then
  for f in "$SUITE"/ieee/*.c; do printf 'ieee|%s\n' "$f"; done >> "$CANDIDATES"
fi
if ((WANT_BUILTINS)); then
  for f in "$SUITE"/builtins/*.c; do
    case "$f" in *-lib.c) continue;; esac
    printf 'builtins|%s\n' "$f"
  done >> "$CANDIDATES"
fi

# Pattern filter: basename must contain one of the patterns.
if ((${#PATTERNS[@]})); then
  FILTERED="$TMP/filtered.txt"
  : > "$FILTERED"
  while IFS='|' read -r suite src; do
    base="$(basename "$src")"
    for p in "${PATTERNS[@]}"; do
      if [[ "$base" == *"$p"* ]]; then printf '%s|%s\n' "$suite" "$src" >> "$FILTERED"; break; fi
    done
  done < "$CANDIDATES"
  mv "$FILTERED" "$CANDIDATES"
fi

TOTAL=$(wc -l < "$CANDIDATES")
((TOTAL)) || { echo "no tests selected"; exit 0; }
echo "running $TOTAL torture tests (jobs=$JOBS, compile-timeout=${CTIMEOUT}s, run-timeout=${RTIMEOUT}s)"

run_one() {
  local suite="$1" src="$2"
  local base name bin log res
  base="$(basename "$src")"
  name="${base%.c}"
  bin="$BIN/$suite-$name"
  log="$LOG/$suite-$name.log"
  res="$RES/$suite-$name.result"
  local cc_status run_status extra=()

  if [[ "$suite" == builtins ]]; then
    local lib="${src%.c}-lib.c" main="$SUITE/builtins/lib/main.c"
    extra=("$lib" "$main")
    [[ -f "$lib" ]] || { echo "COMPILE_FAIL $suite/$base (no $lib)" > "$res"; echo "COMPILE_FAIL $suite/$base (missing $lib)" ; return; }
  fi

  # shellcheck disable=SC2086
  timeout "$CTIMEOUT" "$GLCC" -w $CFLAGS_EXTRA "$src" "${extra[@]}" -o "$bin" >"$log" 2>&1
  cc_status=$?
  if ((cc_status == 124)); then
    echo "COMPILE_TIMEOUT $suite/$base" > "$res"
    echo "COMPILE_TIMEOUT $suite/$base"
    return
  elif ((cc_status != 0)); then
    echo "COMPILE_FAIL $suite/$base (cc=$cc_status)" > "$res"
    echo "COMPILE_FAIL $suite/$base (cc=$cc_status)  log: $log"
    return
  fi

  timeout "$RTIMEOUT" "$bin" >"$log.run" 2>&1
  run_status=$?
  if ((run_status == 0)); then
    echo "PASS $suite/$base" > "$res"
    [[ -n "$VERBOSE" ]] && echo "PASS $suite/$base"
  elif ((run_status == 124)); then
    echo "RUN_TIMEOUT $suite/$base" > "$res"
    echo "RUN_TIMEOUT $suite/$base"
  else
    echo "RUN_FAIL $suite/$base (exit=$run_status)" > "$res"
    echo "RUN_FAIL $suite/$base (exit=$run_status)"
  fi
}
export -f run_one
export BIN LOG RES SUITE GLCC CTIMEOUT RTIMEOUT VERBOSE CFLAGS_EXTRA="${CFLAGS_EXTRA:-}"

cut -d'|' -f1- --output-delimiter=' ' "$CANDIDATES" \
  | xargs -P "$JOBS" -n 2 bash -c 'run_one "$@"' _

# Aggregate.
pass=0; cfail=0; ctimeout=0; rfail=0; rtimeout=0
FAIL_LIST="$TMP/failures.txt"; : > "$FAIL_LIST"
for r in "$RES"/*.result; do
  status=$(cut -d' ' -f1 "$r")
  case "$status" in
    PASS) pass=$((pass+1));;
    COMPILE_FAIL) cfail=$((cfail+1)); cat "$r" >> "$FAIL_LIST";;
    COMPILE_TIMEOUT) ctimeout=$((ctimeout+1)); cat "$r" >> "$FAIL_LIST";;
    RUN_FAIL) rfail=$((rfail+1)); cat "$r" >> "$FAIL_LIST";;
    RUN_TIMEOUT) rtimeout=$((rtimeout+1)); cat "$r" >> "$FAIL_LIST";;
  esac
done

echo "----------------------------------------"
echo "total: $TOTAL  pass: $pass  compile-fail: $cfail  compile-timeout: $ctimeout  run-fail: $rfail  run-timeout: $rtimeout"
if [[ -s "$FAIL_LIST" ]]; then
  echo "--- failures ---"
  sort "$FAIL_LIST"
  cp "$FAIL_LIST" "$ROOT/torture-failures.txt"
  echo "(full list also saved to $ROOT/torture-failures.txt)"
fi
if [[ -z "$KEEP" ]]; then
  :
else
  echo "logs kept in $LOG"
fi

(( cfail + ctimeout + rfail + rtimeout )) && exit 1 || exit 0
