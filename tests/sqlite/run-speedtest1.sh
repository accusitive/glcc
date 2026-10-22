#!/usr/bin/env bash
# Build SQLite's speedtest1 against the amalgamation with glcc and a
# reference compiler, run both, and compare per-test timings. The
# comparison is the point: a total tells you whether something is wrong,
# the per-test table tells you which MIR/codegen gap to work on.
#
# Usage:
#   tests/sqlite/run-speedtest1.sh [script option | speedtest1 options...]
#
#   tests/sqlite/run-speedtest1.sh --size 10 --testset main
#
# Script options:
#   --no-build       reuse the binaries in BUILD_DIR
#   --no-reference   run glcc only; skip the reference compiler
#   -h, --help       show this help
#
# Environment:
#   GLCC             compiler under test (default: target/release/glcc,
#                    falling back to target/debug/glcc)
#   REF_CC           reference compiler (default: cc, then gcc, then
#                    clang). REF_CC=none skips the reference build.
#   SPEEDTEST_SIZE   --size passed when the arguments don't (default: 100)
#   SPEEDTEST_DB     database file to use instead of an in-memory database
#   SPEEDTEST_CFLAGS extra flags appended to the common -O2 build
#   BUILD_DIR        scratch/build directory
#                    (default: ${TMPDIR:-/tmp}/glcc-speedtest1-$VERSION)
#   TOP              slowest tests to print (default: 20)
#   RUNS             complete runs per binary; the best total wins (default: 1)
#   SQLITE_VERSION   download version to look for (default: 3540100)
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)
version=${SQLITE_VERSION:-3540100}
top=${TOP:-20}
runs=${RUNS:-1}

die() {
    echo "run-speedtest1: $*" >&2
    exit 1
}

[[ $top =~ ^[1-9][0-9]*$ ]] || die "TOP must be a positive integer"
[[ $runs =~ ^[1-9][0-9]*$ ]] || die "RUNS must be a positive integer"

usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"
}

no_build=0
no_reference=0
passthrough=()
while [ $# -gt 0 ]; do
    case $1 in
        --no-build) no_build=1 ;;
        --no-reference) no_reference=1 ;;
        -h|--help)
            usage
            exit 0
            ;;
        --) shift; passthrough+=("$@"); break ;;
        *) passthrough+=("$1") ;;
    esac
    shift
done

if [ -n "${GLCC:-}" ]; then
    glcc=$GLCC
elif [ -x "$repo/target/release/glcc" ]; then
    glcc=$repo/target/release/glcc
elif [ -x "$repo/target/debug/glcc" ]; then
    glcc=$repo/target/debug/glcc
else
    die "glcc not found; run 'cargo build --release' or set GLCC=..."
fi

if [ "$no_reference" = 0 ] && [ "${REF_CC:-}" != none ]; then
    ref=${REF_CC:-}
    if [ -z "$ref" ]; then
        for candidate in cc gcc clang; do
            if command -v "$candidate" >/dev/null 2>&1; then
                ref=$candidate
                break
            fi
        done
    fi
    if [ -n "$ref" ] && ! command -v "$ref" >/dev/null 2>&1; then
        die "reference compiler '$ref' not found (set REF_CC=none to skip it)"
    fi
else
    ref=
fi

amal="$here/sqlite-amalgamation-$version"
src="$here/sqlite-src-$version"
speedtest1_c="$src/test/speedtest1.c"
if [ -f "$amal/sqlite3.c" ]; then
    sqlite_dir=$amal
elif [ -f "$src/sqlite3.c" ]; then
    sqlite_dir=$src
else
    die "sqlite3.c not found under $amal or $src; run tests/sqlite/download-sqlite.sh first"
fi
[ -f "$speedtest1_c" ] || die "$speedtest1_c not found; run tests/sqlite/download-sqlite.sh first"

build_dir=${BUILD_DIR:-${TMPDIR:-/tmp}/glcc-speedtest1-$version}
mkdir -p "$build_dir"
glcc_bin=$build_dir/speedtest1-glcc
ref_bin=$build_dir/speedtest1-ref
glcc_log=$build_dir/glcc.log
ref_log=$build_dir/ref.log
glcc_build_log=$build_dir/glcc-build.log
ref_build_log=$build_dir/ref-build.log

cflags=(-O2 "-I$sqlite_dir")
if [ -n "${SPEEDTEST_CFLAGS:-}" ]; then
    read -r -a extra_flags <<< "$SPEEDTEST_CFLAGS"
    cflags+=("${extra_flags[@]}")
fi
libs=(-lm -lpthread)
case $(uname -s) in
    Linux) libs+=(-ldl) ;;
esac

build() {
    local label=$1 compiler=$2 output=$3 log=$4
    local elapsed
    echo "building $label: $compiler"
    TIMEFORMAT='%3R'
    if ! { time "$compiler" "${cflags[@]}" -o "$output" "$speedtest1_c" "$sqlite_dir/sqlite3.c" "${libs[@]}" >"$log" 2>&1; } 2>"$log.time"; then
        echo "  compile failed; last lines of $log:" >&2
        tail -n 20 "$log" >&2
        exit 1
    fi
    elapsed=$(cat "$log.time")
    echo "  ${elapsed}s"
}

if [ "$no_build" = 0 ]; then
    build glcc "$glcc" "$glcc_bin" "$glcc_build_log"
    if [ -n "$ref" ]; then
        build "reference ($ref)" "$ref" "$ref_bin" "$ref_build_log"
    fi
else
    [ -x "$glcc_bin" ] || die "--no-build given but $glcc_bin is missing"
    if [ -n "$ref" ]; then
        [ -x "$ref_bin" ] || die "--no-build given but $ref_bin is missing"
    fi
fi

has_size=0
has_db=0
for ((i = 0; i < ${#passthrough[@]}; i++)); do
    arg=${passthrough[$i]}
    if [[ $arg != -* ]]; then
        has_db=1
        continue
    fi
    opt=${arg#-}
    opt=${opt#-}
    case $opt in
        size) has_size=1; i=$((i + 1)) ;;
        heap|lookaside|pcache) i=$((i + 2)) ;;
        cachesize|hard-heap-limit|journal|key|mmap|output|pagesize|repeat|reserve|script|soft-heap-limit|temp|testset|threads|vfs) i=$((i + 1)) ;;
    esac
done
if [ -n "${SPEEDTEST_DB:-}" ] && [ "$has_db" = 1 ]; then
    die "pass either a database path or SPEEDTEST_DB, not both"
fi

run_args=()
if [ "$has_size" = 0 ]; then
    run_args+=(--size "${SPEEDTEST_SIZE:-100}")
fi
if [ -n "${SPEEDTEST_DB:-}" ]; then
    run_args+=("${passthrough[@]}" "$SPEEDTEST_DB")
elif [ "$has_db" = 1 ]; then
    run_args+=("${passthrough[@]}")
else
    run_args+=(--memdb "${passthrough[@]}")
fi

# Run one binary, repeating the whole benchmark if RUNS > 1 and keeping
# the run with the lowest reported TOTAL. Writes that run's output to $log.
run_bench() {
    local label=$1 binary=$2 log=$3
    local best_run= total best_total= i
    for ((i = 1; i <= runs; i++)); do
        echo "running $label (${i}/${runs}): $binary ${run_args[*]}"
        if ! "$binary" "${run_args[@]}" >"$log.run$i" 2>&1; then
            echo "  speedtest1 failed; last lines:" >&2
            tail -n 20 "$log.run$i" >&2
            exit 1
        fi
        total=$(awk '
            { line = $0; sub(/^[ \t]+/, "", line) }
            line ~ /^TOTAL\./ {
                n = split(line, field, " ")
                value = field[n]
                sub(/s$/, "", value)
                print value
            }
        ' "$log.run$i")
        [ -n "$total" ] || die "$label produced no TOTAL line"
        if [ -z "$best_total" ] || awk -v a="$total" -v b="$best_total" 'BEGIN { exit !(a < b) }'; then
            best_total=$total
            best_run=$i
        fi
    done
    cp "$log.run$best_run" "$log"
    echo "  total ${best_total}s"
}

run_bench glcc "$glcc_bin" "$glcc_log"
if [ -n "$ref" ]; then
    run_bench "reference ($ref)" "$ref_bin" "$ref_log"
fi

echo
echo "per-test times (slowest glcc tests first), logs in $build_dir"
if [ -n "$ref" ]; then
    awk -v top="$top" -v glcc_log="$glcc_log" -v ref_log="$ref_log" '
        function parse(path, times, names, sets, nums, totals,    line, set, n, field, key, value, name, num) {
            set = "main"
            while ((getline line < path) > 0) {
                sub(/^[ \t]+/, "", line)
                if (line ~ /^Begin testset "/) {
                    set = line
                    sub(/^Begin testset "/, "", set)
                    sub(/".*$/, "", set)
                    continue
                }
                if (line ~ /^TOTAL\./) {
                    n = split(line, field, " ")
                    value = field[n]
                    sub(/s$/, "", value)
                    totals["total"] = value
                    continue
                }
                if (line !~ /^[0-9]+ - / || line !~ /[0-9.]+s$/) continue
                n = split(line, field, " ")
                value = field[n]
                sub(/s$/, "", value)
                num = line
                sub(/ - .*$/, "", num)
                name = line
                sub(/^[0-9]+ - /, "", name)
                sub(/\.\.\..*$/, "", name)
                key = set SUBSEP num
                times[key] = value
                names[key] = name
                sets[key] = set
                nums[key] = num
            }
            close(path)
        }
        BEGIN {
            parse(ref_log, ref_time, ref_name, ref_set, ref_num, ref_total)
            parse(glcc_log, glcc_time, glcc_name, glcc_set, glcc_num, glcc_total)
            total = 0
            for (key in glcc_time) {
                total++
                keys[total] = key
            }
            for (i = 1; i <= total; i++) {
                for (j = i + 1; j <= total; j++) {
                    if (glcc_time[keys[j]] + 0 > glcc_time[keys[i]] + 0) {
                        swap = keys[i]
                        keys[i] = keys[j]
                        keys[j] = swap
                    }
                }
            }
            printf "%8s %8s %7s  %s\n", "glcc", "ref", "ratio", "test"
            limit = top + 0 < total ? top + 0 : total
            for (i = 1; i <= limit; i++) {
                key = keys[i]
                if (key in ref_time) {
                    if (ref_time[key] + 0 > 0) {
                        ratio = glcc_time[key] / ref_time[key]
                        printf "%7.3fs %7.3fs %6.2fx  %-11s %4s %s\n", glcc_time[key], ref_time[key], ratio, glcc_set[key], glcc_num[key], glcc_name[key]
                    } else {
                        printf "%7.3fs %7.3fs %7s  %-11s %4s %s\n", glcc_time[key], ref_time[key], "-", glcc_set[key], glcc_num[key], glcc_name[key]
                    }
                } else {
                    printf "%7.3fs %8s %7s  %-11s %4s %s\n", glcc_time[key], "-", "-", glcc_set[key], glcc_num[key], glcc_name[key]
                }
            }
            printf "\n"
            if (ref_total["total"] + 0 > 0) {
                printf "%7.3fs %7.3fs %6.2fx  TOTAL\n", glcc_total["total"], ref_total["total"], glcc_total["total"] / ref_total["total"]
            } else {
                printf "%7.3fs %7.3fs %7s  TOTAL\n", glcc_total["total"], ref_total["total"], "-"
            }
        }
    '
else
    awk '
        BEGIN { set = "main" }
        {
            line = $0
            sub(/^[ \t]+/, "", line)
            if (line ~ /^Begin testset "/) {
                set = line
                sub(/^Begin testset "/, "", set)
                sub(/".*$/, "", set)
                next
            }
            if (line ~ /^TOTAL\./) {
                n = split(line, field, " ")
                total = field[n]
                sub(/s$/, "", total)
                next
            }
            if (line !~ /^[0-9]+ - / || line !~ /[0-9.]+s$/) next
            n = split(line, field, " ")
            value = field[n]
            sub(/s$/, "", value)
            num = line
            sub(/ - .*$/, "", num)
            name = line
            sub(/^[0-9]+ - /, "", name)
            sub(/\.\.\..*$/, "", name)
            printf "%7.3fs %-11s %4s %s\n", value, set, num, name
        }
    ' "$glcc_log" >"$build_dir/glcc.per-test"
    printf "%8s  %s\n" "glcc" "test"
    sort -k1,1gr "$build_dir/glcc.per-test" | sed -n "1,${top}p"
    echo
    awk '
        { line = $0; sub(/^[ \t]+/, "", line) }
        line ~ /^TOTAL\./ {
            n = split(line, field, " ")
            print "TOTAL " field[n]
        }
    ' "$glcc_log"
fi
