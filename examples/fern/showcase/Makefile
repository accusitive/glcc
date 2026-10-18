# fern showcase with fernc: make, make run, make clean.
# fernc comes from the repo's build tree; `cargo build -p fernc` builds it.
FERNC ?= $(firstword $(wildcard ../../../target/debug/fernc) fernc)

all: build/showcase

build/showcase: build/showcase.o driver.c
	$(CC) driver.c build/showcase.o -o $@

build/showcase.o: showcase.fern
	mkdir -p $(dir $@)
	$(FERNC) -O1 $< -o $@

run: build/showcase
	./build/showcase

clean:
	rm -rf build

.PHONY: all run clean
