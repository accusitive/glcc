# Multithreaded executor demo: make, make run.
# Prefers the repo's build tree; falls back to an installed glcc.
GLCC ?= $(firstword $(wildcard ../../../target/debug/glcc) glcc)
RUNTIME := ../runtime

all: build/executor

build/executor: demo.c $(RUNTIME)/exec.c $(RUNTIME)/io.c $(RUNTIME)/exec.h $(RUNTIME)/io.h
	mkdir -p $(dir $@)
	$(GLCC) -fglcc-async -O2 -I$(RUNTIME) demo.c $(RUNTIME)/exec.c $(RUNTIME)/io.c -pthread -o $@

run: build/executor
	./build/executor

clean:
	rm -rf build

.PHONY: all run clean
