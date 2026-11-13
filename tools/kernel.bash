#!/usr/bin/env bash
# kernel.bash — compile the Linux kernel with glcc's native backend.
#
# An experiment, not a supported workflow: a minimal (tinyconfig) kernel
# is configured and built with CC=glcc to see how far the compiler gets.
# Failure is expected; the point is collecting the first real blockers.
#
# Usage:
#   ./tools/kernel.bash download [--version=X.Y.Z] [--dir=DIR]
#   ./tools/kernel.bash config   [--dir=DIR]
#   ./tools/kernel.bash build    [-j N] [--dir=DIR] [--v=1]
#   ./tools/kernel.bash all      (download + config + build)
#
# The kernel tree is kept outside the repo (default
# ~/.cache/glcc-kernel/linux-<version>); everything else needed for the
# build (host tools, .config) lives in that tree. kbuild host tools
# (kconfig, fixdep) always build with the host gcc; only kernel C files
# go through glcc. glcc is invoked through a generated shim
# (cc-glcc.sh next to the tree) that injects -fbackend=native and
# -fintegrated-as, since kbuild's CC= must be a single program.
#
# The build log lands in build.log in the tree; `build` prints the last
# error context and a tally of "error:" lines instead of the full log.

set -u

repo=$(cd "$(dirname "$0")/.." && pwd)
version=6.12.30
dir=
jobs=$(nproc)
v=

glcc=${GLCC:-$repo/target/release/glcc}
glcc=$(readlink -f "$glcc")
[[ -x $glcc ]] || { echo "glcc not built at $glcc (GLCC env overrides)" >&2; exit 1; }

cmd=$1
shift
while [[ $# -gt 0 ]]; do
	case $1 in
	--version=*) version=${1#*=} ;;
	--dir=*) dir=${1#*=} ;;
	-j*) jobs=${1#-j} ;;
	--v=*) v=${1#--v=} ;;
	*) echo "unknown arg: $1" >&2; exit 2 ;;
	esac
	shift
done

dir=${dir:-$HOME/.cache/glcc-kernel}
tree=$dir/linux-$version
baseurl=https://cdn.kernel.org/pub/linux/kernel/v6.x
shim=$dir/cc-glcc.sh

fetch() {
	mkdir -p "$dir"
	url=$baseurl/linux-$version.tar.xz
	echo "downloading $url"
	curl -fL --retry 3 -o "$dir/linux-$version.tar.xz" "$url"
	tar -C "$dir" -xf "$dir/linux-$version.tar.xz"
	rm "$dir/linux-$version.tar.xz"
	echo "extracted to $tree"
}

write_shim() {
	cat >"$shim" <<EOF
#!/bin/bash
# kbuild shim for glcc. Workarounds, all fixable in glcc itself:
# - -fbackend=native/-fintegrated-as forced (kbuild CC= takes one word)
# - stdin: kbuild probes pipe snippets to the compiler ("-" input),
#   which glcc does not support; splice into a temp file
# - version masquerade: glcc reports GCC 4.8.0, below the kernel's
#   minimum (5.1); claim GCC 8.2 so toolchain checks pass
# - -S (assembly output): not implemented in glcc; kbuild only uses it
#   for offset-generation tooling (asm-offsets.s, devicetable-offsets.s).
#   Those invocations are delegated to the host gcc, which shares the
#   LP64 SysV ABI, so the emitted offsets are valid for glcc too.
#   Flags the host gcc rejects are dropped on the way.
if [[ " \$* " == *" -S "* ]]; then
	args=()
	for a in "\$@"; do
		[[ \$a == --param=* ]] && continue
		args+=("\$a")
	done
	exec \${GLCC_HOSTCC:-cc} "\${args[@]}"
fi
args=()
for a in "\$@"; do
	if [[ \$a == - ]]; then
		tmp=\$(mktemp /tmp/glcc-kbuild-stdin.XXXXXX.c)
		cat > "\$tmp"
		args+=("\$tmp")
	else
		args+=("\$a")
	fi
done
exec $glcc -fbackend=native -fintegrated-as \\
	-D__GNUC__=8 -D__GNUC_MINOR__=2 -D__GNUC_PATCHLEVEL__=0 \\
	"\${args[@]}"
EOF
	chmod +x "$shim"
}

configure() {
	[[ -d $tree ]] || { echo "no tree at $tree; run download first" >&2; exit 1; }
	# allnoconfig (tinyconfig's base) turns 64BIT off, making this a
	# 32-bit kernel config; glcc is LP64-only, so force it back on.
	# olddefconfig answers any new symbols non-interactively.
	(cd "$tree" && make tinyconfig && ./scripts/config -e 64BIT && make olddefconfig)
	# tinyconfig alone is enough of a stress test; record the CC in
	# the log so the config provenance is visible.
	write_shim
	echo "configured $tree with tinyconfig, CC shim at $shim"
}

build() {
	[[ -d $tree ]] || { echo "no tree at $tree; run download first" >&2; exit 1; }
	[[ -f $tree/.config ]] || { echo "not configured; run config first" >&2; exit 1; }
	write_shim
	log=$tree/build.log
	# -k keeps going past the first failed file so one run maps out
	# many failures at once.
	# CLANG_FLAGS carries -fintegrated-as into kbuild's assembler
	# probes (scripts/as-version.sh / as-instr): glcc is an
	# integrated assembler (glasm), so the "integrated" probe path
	# is the truthful one to take.
	if ! (cd "$tree" && make -j "$jobs" -k V=$v CC="$shim" HOSTCC=gcc \
			CLANG_FLAGS=-fintegrated-as >"$log" 2>&1); then
		echo "BUILD FAILED (log: $log)"
		echo
		echo "first error context:"
		grep -n -m1 -B3 -A15 'error:' "$log" || tail -40 "$log"
	else
		echo "BUILD SUCCEEDED (surprisingly)"
	fi
	echo
	errors=$(grep -c 'error:' "$log" || true)
	echo "error lines: $errors, log size: $(du -h "$log" | cut -f1)"
}

case $cmd in
download) fetch ;;
config) configure ;;
build) build ;;
all)
	fetch
	configure
	build
	;;
*) echo "usage: $0 download|config|build|all [--version=V] [--dir=DIR] [-j N] [--v=1]" >&2; exit 2 ;;
esac
