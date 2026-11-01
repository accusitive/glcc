# C -> portable Lua 5.1 script: make, make run, make clean.
# Prefers the repo's build tree; falls back to an installed glcc.
GLCC ?= $(firstword $(wildcard ../../target/debug/glcc) glcc)
LUA ?= $(firstword $(wildcard lua5.1) lua)

all: build/hello.lua

build/hello.lua: hello.c
	mkdir -p $(dir $@)
	$(GLCC) --emit-lua -O2 $< -o $@

run: build/hello.lua
	$(LUA) build/hello.lua

clean:
	rm -rf build

.PHONY: all run clean
