#!/usr/bin/env bash
# pp-diff.bash — differential-test glcc's native preprocessor against clang.
#
# The corpus is the local LLVM checkout's clang/test/Preprocessor (no
# test files are copied into glcc, so pulling ../llvm-project refreshes
# the corpus). For each test whose RUN lines stay within the supported
# flag subset, `clang -E` and `glcc --pp=native -E` process the same
# input, and their stdout is compared byte-for-byte after dropping
# `# <n> ...` linemarker lines from both sides.
#
# Usage:
#   ./tools/pp-diff.bash [--llvm=DIR] [--corpus=DIR] [--no-build] [-v] [pattern ...]
#   ./tools/pp-diff.bash --vendor=DIR [--llvm=DIR] [--corpus=DIR] [pattern ...]
#
#   pattern filters by basename substring (e.g. ./tools/pp-diff.bash macro_paste)
#   -v prints a unified diff for each failing input
#   --vendor=DIR exports inputs, raw `clang -E` goldens, and a
#     corpus.tsv manifest under DIR (consumed by
#     crates/glcc_pp/tests/clang_corpus.rs); no glcc build needed. Runs
#     whose output depends on host predefines or system headers are
#     skipped, and `# <n> ...` lines are stripped at comparison time.
#
# Supported RUN flags: -E, -std=, -D, -U, -I, -include, -undef, -o.
# A RUN line with any other flag (-P, -f*, -W*, -target, ...) or a lit
# path substitution (%S, %t, ...) is skipped, as is a file whose RUN
# lines are all unsupported; widen the whitelist in parse_run as
# glcc_pp grows.
#
# Env overrides: GLCC (default target/debug/glcc), CLANG (default clang),
# LLVM (default ../llvm-project).

set -u
shopt -s nullglob

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GLCC="${GLCC:-$ROOT/target/debug/glcc}"
CLANG="${CLANG:-clang}"
LLVM="${LLVM:-$ROOT/../llvm-project}"
CORPUS=""
VENDOR=""
BUILD=1
VERBOSE=""
PATTERNS=()

while (($#)); do
  case "$1" in
    --llvm=*) LLVM="${1#--llvm=}"; shift;;
    --corpus=*) CORPUS="${1#--corpus=}"; shift;;
    --vendor=*) VENDOR="${1#--vendor=}"; shift;;
    --no-build) BUILD=0; shift;;
    -v|--verbose) VERBOSE=1; shift;;
    -h|--help) sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# //;s/^#//'; exit 0;;
    -*) echo "unknown option: $1 (see --help)" >&2; exit 2;;
    *) PATTERNS+=("$1"); shift;;
  esac
done

if [[ -z "$VENDOR" ]]; then
  if ((BUILD)); then
    echo "building glcc..."
    (cd "$ROOT" && cargo build 2>&1 | tail -2) || exit 1
  fi
  [[ -x "$GLCC" ]] || { echo "error: glcc binary not found: $GLCC" >&2; exit 1; }
fi
command -v "$CLANG" >/dev/null || { echo "error: clang not found: $CLANG" >&2; exit 1; }

CORPUS="${CORPUS:-$LLVM/clang/test/Preprocessor}"
[[ -d "$CORPUS" ]] || { echo "error: corpus not found: $CORPUS (set --llvm/--corpus)" >&2; exit 1; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/glcc-pp-diff-XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

FILES=()
for f in "$CORPUS"/*.c; do
  base="$(basename "$f")"
  if ((${#PATTERNS[@]})); then
    hit=""
    for p in "${PATTERNS[@]}"; do [[ "$base" == *"$p"* ]] && hit=1 && break; done
    [[ -n "$hit" ]] || continue
  fi
  FILES+=("$f")
done

# Parse one `// RUN:` line into FLAGS (flag tokens; paired values stay
# separate), rejecting lit path substitutions and any flag outside the
# whitelist. Sets HAS_E when the line preprocesses.
FLAGS=()
HAS_E=""
parse_run() {
  local line="$1" tok pending=""
  FLAGS=()
  HAS_E=""
  line="${line#*RUN: }"
  line="${line%%|*}"
  case "$line" in *%S*|*%t*|*%T*|*%p*) return 1;; esac
  for tok in $line; do
    if [[ -n "$pending" ]]; then
      [[ "$pending" == "-o" ]] || FLAGS+=("$pending" "$tok")
      pending=""
      continue
    fi
    case "$tok" in
      %clang_cc1|%s) ;;
      -E) HAS_E=1;;
      -o) pending="-o";;
      -D|-U|-I|-isystem|-iquote|-include) pending="$tok";;
      -std=*|-D*|-U*|-I*|-include*|-undef) FLAGS+=("$tok");;
      *) return 1;;
    esac
  done
  [[ -z "$pending" ]] || return 1
  [[ -n "$HAS_E" ]]
}

if [[ -n "$VENDOR" ]]; then
  mkdir -p "$VENDOR/inputs" "$VENDOR/expected" "$VENDOR/predefines"
  : > "$VENDOR/corpus.tsv"
  STDS="$TMP/stds"
  : > "$STDS"
  # Sibling headers quoted by vendored tests travel along.
  for h in "$CORPUS"/*.h "$CORPUS"/*.inc "$CORPUS"/*.def; do
    cp "$h" "$VENDOR/inputs/"
  done
  if [[ -d "$CORPUS/Inputs" ]]; then
    mkdir -p "$VENDOR/inputs/Inputs"
    cp -r "$CORPUS/Inputs/." "$VENDOR/inputs/Inputs/"
  fi
  VFILES=0
  VRUNS=0
  for f in "${FILES[@]}"; do
    base="$(basename "$f")"
    # Golden would embed this machine's wall clock (__DATE__/__TIME__).
    [[ "$base" == init-datetime* ]] && continue
    n=0
    copied=""
    while IFS= read -r run; do
      parse_run "$run" || continue
      flags="${FLAGS[*]-}"
      # Goldens keep clang's linemarkers (the Rust runner strips them from
      # both sides, matching this script's live comparison). Default
      # predefines make clang jump the built-in region with a marker;
      # `-undef` would instead pad it with one blank per standard macro.
      (cd "$CORPUS" && "$CLANG" -E $flags "$base") > "$TMP/c.out" 2> "$TMP/c.err" || continue
      # Skip tests whose output depends on the host: predefines beyond the
      # standard set clang keeps under -undef, or system headers.
      (cd "$CORPUS" && "$CLANG" -E -P $flags "$base") > "$TMP/p1.out" 2>/dev/null || continue
      (cd "$CORPUS" && "$CLANG" -E -P -undef $flags "$base") > "$TMP/p2.out" 2>/dev/null || continue
      (cd "$CORPUS" && "$CLANG" -E -P -nostdinc $flags "$base") > "$TMP/p3.out" 2>/dev/null || continue
      cmp -s "$TMP/p1.out" "$TMP/p2.out" || continue
      cmp -s "$TMP/p1.out" "$TMP/p3.out" || continue
      if [[ -z "$copied" ]]; then cp "$f" "$VENDOR/inputs/$base"; copied=1; fi
      cp "$TMP/c.out" "$VENDOR/expected/$base.$n.txt"
      printf '%s\t%s\t%s\n' "$base" "$flags" "expected/$base.$n.txt" >> "$VENDOR/corpus.tsv"
      key=default
      for t in $flags; do [[ "$t" == -std=* ]] && key="${t#-std=}"; done
      printf '%s\n' "$key" >> "$STDS"
      n=$((n + 1))
    done < <(grep -E '^// RUN:' "$f")
    if [[ -n "$copied" ]]; then
      VFILES=$((VFILES + 1))
      VRUNS=$((VRUNS + n))
    fi
  done
  sort -u "$STDS" | while read -r key; do
    stdflag=""
    [[ "$key" == default ]] || stdflag="-std=$key"
    printf '' | "$CLANG" -E -undef $stdflag -dM -x c - | sort > "$VENDOR/predefines/$key.txt"
  done
  echo "vendored $VFILES files, $VRUNS runs -> $VENDOR"
  exit 0
fi

SAME=0; DIFF=0; ERR=0; SKIP=0; TOTAL=0
for f in "${FILES[@]}"; do
  TOTAL=$((TOTAL + 1))
  RUNS=()
  while IFS= read -r run; do
    if parse_run "$run"; then
      RUNS+=("${FLAGS[*]-}")
    fi
  done < <(grep -E '^// RUN:' "$f")
  ((${#RUNS[@]})) || { SKIP=$((SKIP + 1)); continue; }

  status=same
  for flags in "${RUNS[@]}"; do
    # shellcheck disable=SC2086
    "$CLANG" -E $flags -I "$CORPUS" "$f" > "$TMP/c.out" 2> "$TMP/c.err"; cret=$?
    # shellcheck disable=SC2086
    "$GLCC" --pp=native -E $flags -I "$CORPUS" "$f" > "$TMP/g.out" 2> "$TMP/g.err"; gret=$?
    if ((cret != 0 || gret != 0)); then
      status=err
      break
    fi
    grep -v '^# [0-9]' "$TMP/c.out" > "$TMP/c.norm"
    grep -v '^# [0-9]' "$TMP/g.out" > "$TMP/g.norm"
    if ! diff -q "$TMP/c.norm" "$TMP/g.norm" >/dev/null; then
      status=diff
      break
    fi
  done

  case "$status" in
    same) SAME=$((SAME + 1));;
    diff)
      DIFF=$((DIFF + 1))
      n=$(diff "$TMP/c.norm" "$TMP/g.norm" | grep -c '^[<>]' || true)
      echo "DIFF $(basename "$f")  ($n lines)"
      if [[ -n "$VERBOSE" ]]; then
        diff -u "$TMP/c.norm" "$TMP/g.norm" | sed -n '1,60p' | sed 's/^/  /'
      fi
      ;;
    err)
      ERR=$((ERR + 1))
      echo "ERR  $(basename "$f")  clang=$cret glcc=$gret"
      if [[ -n "$VERBOSE" ]]; then
        sed 's/^/  clang: /' "$TMP/c.err" | sed -n '1,5p'
        sed 's/^/  glcc:  /' "$TMP/g.err" | sed -n '1,5p'
      fi
      ;;
  esac
done

echo
echo "corpus: $CORPUS"
echo "$TOTAL files: $SAME same, $DIFF diff, $ERR err, $SKIP skipped"
((DIFF + ERR == 0))
