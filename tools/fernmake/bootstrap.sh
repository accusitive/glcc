#!/bin/sh
# Build fernmake without assuming GNU make is installed. Once this script
# has produced build/fernmake, that binary can rebuild itself from the
# tool's Makefile.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
fernc=${FERNC:-fernc}
glcc=${GLCC:-glcc}

mkdir -p "$here/build"
"$fernc" -O1 "$here/make.fern" -o "$here/build/make.o"
"$glcc" "$here/driver.c" "$here/build/make.o" -o "$here/build/fernmake"
