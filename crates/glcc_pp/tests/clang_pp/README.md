# clang preprocessor corpus

Differential-test corpus vendored from LLVM's `clang/test/Preprocessor`
(Apache-2.0 WITH LLVM-exception). `inputs/` holds the original test
sources, `expected/` holds raw `clang -E` output, and `corpus.tsv` maps
each input + RUN flags to its golden. The Rust runner strips `# <n> ...`
linemarker lines from both sides before comparing, so the goldens keep
everything clang prints; `predefines/<std>.txt` pins the standard macros
clang keeps under `-undef`, which the runner seeds into glcc so the
comparison is host-independent.

Regenerate with a local LLVM checkout:

    ./tools/pp-diff.bash --vendor=crates/glcc_pp/tests/clang_pp

Vendoring skips runs whose output depends on this machine: wall-clock
macros, non-standard predefines (probed with `-undef`), or system
headers (probed with `-nostdinc`).

The goldens are ground truth: fix `glcc_pp` to match them; do not edit
them by hand (regenerate instead when pulling a newer clang).
