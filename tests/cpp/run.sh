#!/bin/sh
# Run the EDG -> MIR C++ end-to-end tests: every .cpp in this directory is
# compiled with the default glcpp pipeline (cpfe mir backend -> MIR text ->
# parse -> optimize -> Cranelift object -> link), executed, and its output
# compared against g++ when g++ is available.
#
# Usage: tests/cpp/run.sh [path-to-glcpp]
set -e
cd "$(dirname "$0")"

GLCPP="${1:-../../target/debug/glcpp}"
GXX="$(command -v g++ || true)"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fails=0
for src in *.cpp; do
  name="${src%.cpp}"
  printf '%-12s' "$name"
  if ! "$GLCPP" "$src" -o "$tmp/$name"; then
    echo "FAIL (compile)"
    fails=$((fails + 1))
    continue
  fi
  glcpp_out="$("$tmp/$name" 2>&1)"
  glcpp_rc=$?
  if [ -n "$GXX" ]; then
    "$GXX" -std=c++17 "$src" -o "$tmp/$name.gxx"
    gxx_out="$("$tmp/$name.gxx" 2>&1)"
    gxx_rc=$?
    if [ "$glcpp_out" = "$gxx_out" ] && [ "$glcpp_rc" = "$gxx_rc" ]; then
      echo "ok (matches g++)"
    else
      echo "FAIL (output/rc differs from g++)"
      fails=$((fails + 1))
    fi
  else
    echo "ok (rc=$glcpp_rc, no g++ to compare)"
  fi
done

if [ "$fails" -ne 0 ]; then
  echo "$fails test(s) failed"
  exit 1
fi
