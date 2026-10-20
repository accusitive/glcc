# fern modules with fernc: make, make run, make clean.
# fernc loads imports relative to main.fern and emits one object for now.
PROGRAM = modules
SOURCE = main.fern
EXTRA_DEPS = math.fern
OPT = -O1
FERNC ?= $(firstword $(wildcard ../../../target/debug/fernc ../../../target/debug/fernc.exe) fernc)
GLCC ?= $(firstword $(wildcard ../../../target/debug/glcc ../../../target/debug/glcc.exe) glcc)

include ../common.mk
