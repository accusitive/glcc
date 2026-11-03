#!/bin/sh
# Run the LLVM test suite's C++ programs through glcpp: compile, execute,
# compare stdout against the suite's .reference_output file.
#
# Scope: every directory listed in CXX_DIRS. A .cpp runs only if a
# matching .reference_output exists beside it. Expected failures live in
# xfail-cxx.txt (one test name per line, # comments allowed).
#
# Usage: tests/llvm-test-suite/run-cxx.sh [path-to-glcpp]
set -u
cd "$(dirname "$0")/llvm-test-suite"

GLCPP="${1:-../../target/debug/glcpp}"
TS=.
CXX_DIRS="SingleSource/Regression/C++ SingleSource/UnitTests/C++11 SingleSource/Benchmarks/Adobe-C++ SingleSource/Benchmarks/CoyoteBench"
XFAIL=../llvm-xfail-cxx.txt
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

is_xfail() {
  [ -f "$XFAIL" ] && grep -q "^$1\$" "$XFAIL"
}

ok=0; fail=0; xf=0; xp=0
fails_list=""

for dir in $CXX_DIRS; do
  for src in "$dir"/*.cpp; do
    [ -e "$src" ] || continue
    name="${src%.cpp}"
    ref="$name.reference_output"
    [ -e "$ref" ] || continue
    base=$(basename "$name")
    printf '%-40s' "$name"
    if is_xfail "$base"; then
      want=xfail
    else
      want=pass
    fi
    if ! timeout 120 "$GLCPP" -O2 "$src" -o "$tmp/a.out" 2>"$tmp/cc.err"; then
      if [ "$want" = xfail ]; then echo "xfail (compile)"; xf=$((xf+1));
      else echo "FAIL (compile)"; fail=$((fail+1)); fails_list="$fails_list $base"; fi
      continue
    fi
    timeout 60 "$tmp/a.out" >"$tmp/out" 2>"$tmp/run.err"; rc=$?
    if [ $rc -ne 0 ]; then
      if [ "$want" = xfail ]; then echo "xfail (run rc=$rc)"; xf=$((xf+1));
      else echo "FAIL (run rc=$rc)"; fail=$((fail+1)); fails_list="$fails_list $base"; fi
      continue
    fi
    # The upstream reference outputs end with an `exit NN` line recorded
    # by the suite's ancient RunSafely.sh -n harness; drop it before
    # comparing (the program itself never prints it).
    sed -e '${/^exit [0-9][0-9]*$/d; s/exit [0-9][0-9]*$//;}' "$ref" >"$tmp/ref"
    # Trailing-newline-insensitive compare: some program outputs are
    # unterminated, and the harness strip above keeps the reference's.
    if [ "$(cat "$tmp/ref")" = "$(cat "$tmp/out")" ]; then
      if [ "$want" = xfail ]; then echo "XPASS"; xp=$((xp+1));
      else echo "ok"; ok=$((ok+1)); fi
    else
      if [ "$want" = xfail ]; then echo "xfail (output)"; xf=$((xf+1));
      else echo "FAIL (output)"; fail=$((fail+1)); fails_list="$fails_list $base"; fi
    fi
  done
done

echo "----"
echo "ok=$ok fail=$fail xfail=$xf xpass=$xp"
if [ -n "$fails_list" ]; then echo "failing:$fails_list"; fi
[ "$fail" -eq 0 ] && [ "$xp" -eq 0 ]
