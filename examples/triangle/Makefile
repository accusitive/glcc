# C++ OpenGL triangle with glcpp + SDL2: make, make run, make clean.
# Prefers the repo's build tree; falls back to an installed glcpp.
GLCPP ?= $(firstword $(wildcard ../../target/debug/glcpp) glcpp)
SDL2_CFLAGS := $(shell sdl2-config --cflags)
SDL2_LIBS := $(shell sdl2-config --libs)

all: build/triangle

build/triangle: tri.cc
	mkdir -p $(dir $@)
	$(GLCPP) -O2 $(SDL2_CFLAGS) $< $(SDL2_LIBS) -o $@

run: build/triangle
	./build/triangle

clean:
	rm -rf build

.PHONY: all run clean
