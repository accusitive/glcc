#!/usr/bin/env bash
# Build the in-browser glcc compiler module.
#
# The page (crates/glcc_explorer/src/page.html) runs the whole
# pipeline -- glcc_pp, frontend, MIR, Cranelift codegen for
# x86_64/aarch64/riscv64 -- inside this wasm module. Only disassembly
# stays on the server: the page POSTs the object bytes to /disasm.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build -p glcc_explorer --lib --target wasm32-unknown-unknown --profile wasm --no-default-features

wasm=target/wasm32-unknown-unknown/wasm/glcc_explorer.wasm
if ! [ -f "$wasm" ]; then
    echo "build-wasm.sh: expected $wasm to exist" >&2
    exit 1
fi

# The module is ~5 MB as built. `wasm-opt -Oz --enable-bulk-memory` can
# shrink it further, but it takes minutes on a module this size, so it
# is left as an explicit post-step rather than part of the build.
cp "$wasm" "crates/glcc_explorer/glcc-explorer.wasm"
ls -la "crates/glcc_explorer/glcc-explorer.wasm"
echo "smoke test: node crates/glcc_explorer/wasm-smoke.mjs"
