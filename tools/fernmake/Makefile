# Build and run fernmake, the Makefile runner written in fern.
PROGRAM = fernmake
SOURCE = make.fern
OPT = -O1
RUN_ARGS = demo.mk run
FERNC ?= $(firstword $(wildcard ../../target/debug/fernc ../../target/debug/fernc.exe) fernc)
GLCC ?= $(firstword $(wildcard ../../target/debug/glcc ../../target/debug/glcc.exe) glcc)
PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
INSTALL ?= install

all: build/$(PROGRAM)

build/$(PROGRAM): build/$(PROGRAM).o driver.c
	$(GLCC) driver.c build/$(PROGRAM).o -o $@

build/$(PROGRAM).o: $(SOURCE)
	mkdir -p build
	$(FERNC) $(OPT) $< -o $@

run: build/$(PROGRAM)
	./build/$(PROGRAM) $(RUN_ARGS)

install: build/$(PROGRAM)
	$(INSTALL) -d $(DESTDIR)$(BINDIR)
	$(INSTALL) -m 755 build/$(PROGRAM) $(DESTDIR)$(BINDIR)/$(PROGRAM)

clean:
	rm -rf build

.PHONY: all run install clean
