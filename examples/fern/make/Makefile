# fern Makefile runner with fernc: make, make run, make clean.
# fernc and glcc come from the repo's build tree.
PROGRAM = fernmake
SOURCE = make.fern
OPT = -O1
RUN_ARGS = demo.mk run
FERNC ?= $(firstword $(wildcard ../../../target/debug/fernc ../../../target/debug/fernc.exe) fernc)
GLCC ?= $(firstword $(wildcard ../../../target/debug/glcc ../../../target/debug/glcc.exe) glcc)

include ../common.mk
