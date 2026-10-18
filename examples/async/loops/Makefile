# Two infinite coroutine loops on the multithreaded executor: make, make run.
# Prefers the repo's build tree; falls back to an installed glcc.
GLCC ?= $(firstword $(wildcard ../../../target/debug/glcc) glcc)
RUNTIME := ../runtime

all: build/loops

build/loops: demo.c $(RUNTIME)/exec.c $(RUNTIME)/io.c $(RUNTIME)/exec.h $(RUNTIME)/io.h
	mkdir -p $(dir $@)
	$(GLCC) -fglcc-async -O2 -I$(RUNTIME) demo.c $(RUNTIME)/exec.c $(RUNTIME)/io.c -pthread -o $@

run: build/loops
	./build/loops

clean:
	rm -rf build

.PHONY: all run clean
