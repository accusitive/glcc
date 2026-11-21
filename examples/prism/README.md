# Prism shader examples

Three C shaders compiled with the prism dialect: `glcc -fprism
--emit-isa` turns each into the shader IR's canonical ISA text under
`build/`. The ISA text is the artifact — there is no executable to
run (SPIR-V, RSX, and Xenos renderers are still to come) — so
`make show` builds everything and prints it.

| shader | what it shows |
|---|---|
| `vs.c` | vertex stage: MVP transform, Lambert term from a helper function, matrix and vector uniforms packing into c-registers |
| `frag.c` | fragment stage: tex2D over a loop-carried detail accumulation, `lerp4`, scalar uniform packing |
| `rings.c` | control flow: early return, a do-while with break and continue, `saturate_f`/`length2`/`lerp4` helpers |

`make` builds, `make show` prints, `make clean` removes `build/`.
