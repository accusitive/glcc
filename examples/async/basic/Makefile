# Basic async/await demo over the glcc polling runtime: make, make run.
# Prefers the repo's build tree; falls back to an installed glcc.
GLCC ?= $(firstword $(wildcard ../../../target/debug/glcc) glcc)
RT := ../../../rt

all: build/demo

build/demo: demo.c $(RT)/glcc_async.c $(RT)/glcc_async.h
	mkdir -p $(dir $@)
	$(GLCC) -fglcc-async -O2 -I$(RT) demo.c $(RT)/glcc_async.c -o $@

run: build/demo
	./build/demo

clean:
	rm -rf build

.PHONY: all run clean
