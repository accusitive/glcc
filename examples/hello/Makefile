# Hello world with glcc: make, make run, make clean.
# Prefers the repo's build tree; falls back to an installed glcc.
GLCC ?= $(firstword $(wildcard ../../target/debug/glcc) glcc)

all: build/hello

build/hello: hello.c
	mkdir -p $(dir $@)
	$(GLCC) -O2 $< -o $@

run: build/hello
	./build/hello

clean:
	rm -rf build

.PHONY: all run clean
