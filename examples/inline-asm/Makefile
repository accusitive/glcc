# Inline-asm demo with glcc: make, make run, make clean.
# Prefers the repo's build tree; falls back to an installed glcc.
GLCC ?= $(firstword $(wildcard ../../target/debug/glcc) glcc)

all: build/asm

build/asm: asm.c
	mkdir -p $(dir $@)
	$(GLCC) -O2 $< -o $@

run: build/asm
	./build/asm

clean:
	rm -rf build

.PHONY: all run clean
