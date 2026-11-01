Two-TU link test: not driven by tests/cpp/run.sh (run.sh compiles each
.cpp as a single self-contained TU; its glob is flat *.cpp), but glcpp
does accept multiple inputs and links the per-TU objects. Verify by hand:

  g++ -std=c++17 lib.cpp main.cpp -o /tmp/multi && /tmp/multi
  glcpp lib.cpp main.cpp -o /tmp/multi && /tmp/multi

Expected output: "42 21 42" and exit 0.

Status 2026-10-31: g++ passes; glcpp fails at the MIR parse of main.cpp
with "unknown global @g_shared" -- the emitter references extern data
via `globaladdr @g_shared` but never emits a `.dataextern @g_shared`
declaration in the referencing TU (extern *functions* do get `.extern`
lines, so extern calls across TUs work). Once .dataextern emission
lands, this pair should be moved up into tests/cpp/ proper (e.g. as
glcpp-only runner support) or given a small dedicated runner.
