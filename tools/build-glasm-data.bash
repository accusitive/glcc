#!/usr/bin/env bash
# Regenerate glasm_data's instruction tables from the nasm submodule.
#
# nasm's tables are consumed in their pre-generated C form (insnsa.c,
# insnsb.c, regflags.c, regvals.c, iflag.c), so this first builds nasm
# in the submodule, then runs glasm's gen-insns tool over the result.
# Committing the generated insns_gen.rs keeps the workspace build free
# of any nasm dependency; only refreshing the data needs this script.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d nasm ]; then
    echo "build-glasm-data: nasm submodule missing; run:" >&2
    echo "  git submodule update --init nasm" >&2
    exit 1
fi

if [ ! -f nasm/nasm ]; then
    echo "build-glasm-data: building nasm from the submodule..." >&2
    (cd nasm && ./autogen.sh >/dev/null && ./configure >/dev/null && make -j"$(nproc)" >/dev/null)
fi

cargo run -q -p glasm --bin gen-insns
echo "build-glasm-data: regenerated crates/glasm_data/src/insns_gen.rs" >&2
