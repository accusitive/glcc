#!/bin/sh
# Run the EDG -> MIR C++ end-to-end tests: every .cpp in this directory is
# compiled with the default glcpp pipeline (cpfe mir backend -> MIR text ->
# parse -> optimize -> Cranelift object -> link), executed, and its output
# compared against g++ when g++ is available.
#
# Programs listed in xfail.txt (one name per line, '#' starts a comment)
# are expected not to compile or run correctly yet: they report as
# "xfail (known)", or "XPASS (unexpected pass!)" if they start passing.
# The run exits nonzero only on FAIL or XPASS.
#
# Usage: tests/cpp/run.sh [path-to-glcpp]
set -e
cd "$(dirname "$0")"

GLCPP="${1:-../../target/debug/glcpp}"
GXX="$(command -v g++ || true)"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

xfail_list=$(sed -e 's/#.*//' -e 's/[[:space:]]//g' xfail.txt 2>/dev/null || true)

is_xfail() {
  for x in $xfail_list; do
    [ "$x" = "$1" ] && return 0
  done
  return 1
}

# The emitter is in flux and can loop on unsupported constructs; bound
# every compile and run so a hang shows up as a FAIL instead of freezing
# the runner.
limit() {
  if command -v timeout >/dev/null 2>&1; then
    timeout 60 "$@"
  else
    "$@"
  fi
}

fails=0
for src in *.cpp; do
  name="${src%.cpp}"
  printf '%-12s' "$name"

  passed=no
  reason=""
  if limit "$GLCPP" "$src" -o "$tmp/$name" 2>"$tmp/$name.err"; then
    glcpp_rc=0
    glcpp_out=$(limit "$tmp/$name" 2>&1) || glcpp_rc=$?
    if [ -n "$GXX" ]; then
      if limit "$GXX" -std=c++17 "$src" -o "$tmp/$name.gxx" 2>"$tmp/$name.gxx.err"; then
        gxx_rc=0
        gxx_out=$(limit "$tmp/$name.gxx" 2>&1) || gxx_rc=$?
        if [ "$glcpp_out" = "$gxx_out" ] && [ "$glcpp_rc" = "$gxx_rc" ]; then
          passed=yes
        else
          reason="FAIL (output/rc differs from g++)"
        fi
      else
        reason="FAIL (g++ reference compile failed)"
      fi
    else
      if [ "$glcpp_rc" -eq 0 ]; then
        passed=yes
      fi
      reason="FAIL (rc=$glcpp_rc, no g++ to compare)"
    fi
  else
    reason="FAIL (compile)"
  fi

  if [ "$passed" = yes ]; then
    if is_xfail "$name"; then
      echo "XPASS (unexpected pass!)"
      fails=$((fails + 1))
    elif [ -n "$GXX" ]; then
      echo "ok (matches g++)"
    else
      echo "ok (rc=0, no g++ to compare)"
    fi
  elif is_xfail "$name"; then
    echo "xfail (known)"
  else
    echo "$reason"
    fails=$((fails + 1))
    if [ -s "$tmp/$name.err" ]; then
      sed 's/^/    /' "$tmp/$name.err" | head -8
    fi
  fi
done

if [ "$fails" -ne 0 ]; then
  echo "$fails test(s) failed"
  exit 1
fi
