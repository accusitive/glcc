#!/bin/sh
# Build fernmake with fernc and glcc. No dependency tracking: every
# invocation rebuilds from scratch.
#
#   ./build.sh                build build/fernmake
#   ./build.sh run            build, then run the demo
#   ./build.sh install        install to /usr/local/bin
#   ./build.sh install PREFIX=$HOME/.local   DESTDIR stages a packaging root
#   ./build.sh clean          remove build/
set -eu

cd "$(dirname "$0")"

pick_tool() {
  for cand in "$@"; do
    if [ -x "$cand" ]; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}

if [ -z "${FERNC:-}" ]; then
  FERNC="$(pick_tool ../../target/debug/fernc ../../target/release/fernc || command -v fernc)"
fi
if [ -z "${GLCC:-}" ]; then
  GLCC="$(pick_tool ../../target/debug/glcc ../../target/release/glcc || command -v glcc)"
fi

case "${1:-build}" in
build)
  mkdir -p build
  "$FERNC" -O1 make.fern -o build/make.o
  "$GLCC" driver.c build/make.o -o build/fernmake
  ;;
run)
  "$0" build
  ./build/fernmake demo.mk run
  ;;
install)
  "$0" build
  PREFIX="${PREFIX:-/usr/local}"
  install -d "${DESTDIR:-}$PREFIX/bin"
  install -m 755 build/fernmake "${DESTDIR:-}$PREFIX/bin/fernmake"
  ;;
clean)
  rm -rf build
  ;;
*)
  echo "usage: $0 [build|run|install|clean]" >&2
  exit 1
  ;;
esac
