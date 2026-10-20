# fern Brainfuck with fernc: make, make run, make clean.
# fernc and glcc come from the repo's build tree.
PROGRAM = brainfuck
SOURCE = brainfuck.fern
OPT = -O2
FERNC ?= $(firstword $(wildcard ../../../target/debug/fernc ../../../target/debug/fernc.exe) fernc)
GLCC ?= $(firstword $(wildcard ../../../target/debug/glcc ../../../target/debug/glcc.exe) glcc)

include ../common.mk
