# fern modules with fernc: make, make run, make clean.
# fernc loads imports relative to main.fern and emits one object for now.
FERNC ?= $(firstword $(wildcard ../../../target/debug/fernc) fernc)

all: build/modules

build/modules: build/main.o driver.c
	$(CC) driver.c build/main.o -o $@

build/main.o: main.fern math.fern
	mkdir -p $(dir $@)
	$(FERNC) -O1 main.fern -o $@

run: build/modules
	./build/modules

clean:
	rm -rf build

.PHONY: all run clean
