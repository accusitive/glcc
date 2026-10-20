# fern Brainfuck with fernc: make, make run, make clean.
# fernc comes from the repo's build tree; `cargo build -p fernc` builds it.
FERNC ?= $(firstword $(wildcard ../../../target/debug/fernc) fernc)
GLCC ?= $(firstword $(wildcard ../../../target/debug/glcc) glcc)
all: build/brainfuck

build/brainfuck: build/brainfuck.o driver.c
	$(GLCC) driver.c build/brainfuck.o -o $@

build/brainfuck.o: brainfuck.fern
	mkdir -p $(dir $@)
	$(FERNC) -O2 $< -o $@

run: build/brainfuck
	./build/brainfuck

clean:
	rm -rf build

.PHONY: all run clean
