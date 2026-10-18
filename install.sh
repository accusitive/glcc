#!/usr/bin/env bash
# Stage the link-time runtimes under ~/.local/lib/glcc. The compilers
# themselves install with cargo:
#
#   cargo install --path . --locked                            # glcc
#   cargo install --path crates/glcpp --features cxx --locked  # optional C++
#
# glcc's build script already stages glccrt.o during those installs; this
# script rebuilds it with the freshest local glcc and copies EDG's libC.a
# beside it, so an installed glcpp links without GLCPP_LIB_DIR.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LIB="$HOME/.local/lib/glcc"
mkdir -p "$LIB"

if [ -z "${GLCC:-}" ]; then
  for cand in "$ROOT/target/release/glcc" "$ROOT/target/debug/glcc"; do
    if [ -x "$cand" ]; then
      GLCC="$cand"
      break
    fi
  done
fi
if [ -z "${GLCC:-}" ]; then
  GLCC="$(command -v glcc || true)"
fi
if [ -z "${GLCC:-}" ]; then
  echo "install.sh: glcc not found; run 'cargo install --path .' first, or set GLCC" >&2
  exit 1
fi

# One runtime source for every host: hand-written asm blocks behind
# target guards, so no per-architecture file juggling.
"$GLCC" -c -Os -fPIC -ftls-model=initial-exec \
  -fno-asynchronous-unwind-tables -fno-stack-protector \
  "$ROOT/rt/glccrt.c" -o "$LIB/glccrt.o"

if [ "$(uname -m)" = "x86_64" ]; then
  # Vendored soft-float for quad/int128 low-level ops (entries renamed
  # `__glcc_*`, see rt/softfloat/glcc-fe.c): one object per TU, combined
  # into glccrt.o behind the asm runtime. Flags are inline (not in a
  # variable) so any Bourne shell splits them correctly.
  SF_D="$LIB/softfloat-tmp"
  mkdir -p "$SF_D"
  for src in addtf3 subtf3 multf3 divtf3 extendxftf2 trunctfxf2 floattixf floatuntixf fixxfti fixunsxfti clzti2 glcc-fe; do
    cc -c -Os -fPIC -ftls-model=initial-exec \
      -fno-asynchronous-unwind-tables -fno-stack-protector \
      -I"$ROOT/rt/softfloat" \
      -D__addtf3=__glcc_addtf3 -D__subtf3=__glcc_subtf3 \
      -D__multf3=__glcc_multf3 -D__divtf3=__glcc_divtf3 \
      -D__extendxftf2=__glcc_extendxftf2 -D__trunctfxf2=__glcc_trunctfxf2 \
      -D__floattixf=__glcc_floattixf -D__floatuntixf=__glcc_floatuntixf \
      -D__fixxfti=__glcc_fixxfti -D__fixunsxfti=__glcc_fixunsxfti \
      -D__clzti2=__glcc_clzti2 \
      "$ROOT/rt/softfloat/$src.c" -o "$SF_D/$src.o"
  done
  ld -r "$LIB/glccrt.o" "$SF_D"/*.o -o "$LIB/glccrt-combined.o"
  mv "$LIB/glccrt-combined.o" "$LIB/glccrt.o"
  rm -rf "$SF_D"
fi

# EDG's C++ runtime, built into the checkout's target dir by edg-sys.
# (Cargo build dirs are `build/edg-sys-<hash>/out/` or `build/edg-sys/<hash>/out/`
# depending on the cargo version.)
LIBC_A=""
for profile in release debug; do
  [ -d "$ROOT/target/$profile" ] || continue
  LIBC_A="$(find "$ROOT/target/$profile" -path '*/build/edg-sys*' -name libC.a -print -quit 2>/dev/null)"
  [ -n "$LIBC_A" ] && break
done
if [ -n "$LIBC_A" ]; then
  cp "$LIBC_A" "$LIB/libC.a"
else
  echo "install.sh: libC.a not found under target/; build glcpp first:" >&2
  echo "  cargo install --path crates/glcpp --features cxx --locked" >&2
fi
