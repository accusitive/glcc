#!/bin/sh
# Build SQLite from source with glcc and run its in-house "veryquick"
# suite as a compiler regression check.
set -eu

cd "$(dirname "$0")"

VERSION=3540100
SRC="sqlite-src-$VERSION"
URL="https://sqlite.org/2026/sqlite-src-$VERSION.zip"

REPO_ROOT=$(cd ../.. && pwd)
GLCC=${GLCC:-"$REPO_ROOT/target/debug/glcc"}

if [ ! -x "$GLCC" ]; then
  echo "glcc not found at $GLCC (run 'cargo build' or set GLCC=...)" >&2
  exit 1
fi

TCLSH=$(command -v tclsh8.6 || command -v tclsh || true)
if [ -z "$TCLSH" ]; then
  echo "tclsh not found; SQLite's test suite requires Tcl" >&2
  exit 1
fi

if [ ! -f "$SRC/configure" ]; then
  echo "Downloading $URL"
  rm -rf "$SRC" "$SRC.zip"
  wget -q "$URL" -O "$SRC.zip"
  unzip -q "$SRC.zip"
  rm "$SRC.zip"
fi

cd "$SRC"

CC="$GLCC" ./configure --with-tclsh="$TCLSH"
LOG=veryquick.log
JOBS=$(nproc 2>/dev/null || echo 4)

# symlink.test leaves its scratch dirs x/y/z/w in testdir/. The later
# zipfile-25.0 test assumes nothing named "x" exists; fopen() on a
# directory succeeds on Linux, so the test silently skips its expected error.
if ! grep -q '^forcedelete x y z w$' test/symlink.test; then
  perl -0pi -e 's/(list \[file exists x\/test\.db-wal\] \[file exists w\/test\.db-wal\]\n\} \{1 0\}\n)/${1}forcedelete x y z w\n/' test/symlink.test
  if ! grep -q '^forcedelete x y z w$' test/symlink.test; then
    echo "error: failed to patch test/symlink.test (upstream layout changed)" >&2
    exit 1
  fi
fi

# backup5.test leaves test2.db..test8.db behind; shell1-3.21.7 then sees
# stale tables in test7.db and checks the wrong .schema output.
if ! grep -q '^forcedelete test2.db test3.db test4.db test5.db test6.db test7.db test8.db$' test/backup5.test; then
  perl -0pi -e 's/\nfinish_test\n/\nforcedelete test2.db test3.db test4.db test5.db test6.db test7.db test8.db\n\nfinish_test\n/' test/backup5.test
  if ! grep -q '^forcedelete test2.db ' test/backup5.test; then
    echo "error: failed to patch test/backup5.test (upstream layout changed)" >&2
    exit 1
  fi
fi

# Make does not notice that the compiler changed, so start from scratch.
make clean >/dev/null 2>&1 || true

echo "Building testfixture and testprogs with $GLCC"
make -j"$JOBS" CC="$GLCC" testfixture testprogs

# Leave no scratch state behind for the next run.
rm -rf testdir

echo "Running the veryquick suite"
make tcltest CC="$GLCC" 2>&1 | tee "$LOG" || true

if grep -q '^0 errors out of' "$LOG"; then
  grep '^0 errors out of' "$LOG"
  echo "SQLite veryquick suite passed."
else
  echo "SQLite veryquick suite FAILED:" >&2
  grep -E '^[0-9]+ errors out of|^!Failures on these tests:' "$LOG" >&2 || true
  exit 1
fi
