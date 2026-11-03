# Hello world with glcc: make, make run, make clean.
# Prefers the repo's build tree; falls back to an installed glcc.
GLCPP ?= $(firstword $(wildcard ../../target/debug/glcpp) glcpp)

all: build/hello_cxx

build/hello_cxx.o: hello.cc
	mkdir -p $(dir $@)
	$(GLCPP) -O3 --emit-mir $< -o $@.mir
	$(GLCPP) -O3 -c $< -o $@

build/hello_cxx: build/hello_cxx.o
	$(GLCPP) -O3 build/hello_cxx.o -o build/hello_cxx
run: build/hello_cxx
	./build/hello_cxx

clean:
	rm -rf build

.PHONY: all run clean
