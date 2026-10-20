# fern Makefile runner with fernc: make, make run, make clean.
# fernc comes from the repo's build tree; `cargo build -p fernc` builds it.
FERNC ?= $(firstword $(wildcard ../../../target/debug/fernc) fernc)

all: build/fernmake

build/fernmake: build/make.o driver.c
	$(CC) driver.c build/make.o -o $@

build/make.o: make.fern
	mkdir -p $(dir $@)
	$(FERNC) -O1 $< -o $@

run: build/fernmake
	./build/fernmake demo.mk run

clean:
	rm -rf build

.PHONY: all run clean
