# fern Brainfuck with fernc: make, make run, make clean.
# fernc comes from the repo's build tree; `cargo build -p fernc` builds it.
FERNC ?= $(firstword $(wildcard ../../../target/debug/fernc) fernc)

all: build/brainfuck

build/brainfuck: build/brainfuck.o driver.c
	$(CC) driver.c build/brainfuck.o -o $@

build/brainfuck.o: brainfuck.fern
	mkdir -p $(dir $@)
	$(FERNC) -O1 $< -o $@

run: build/brainfuck
	./build/brainfuck

clean:
	rm -rf build

.PHONY: all run clean
