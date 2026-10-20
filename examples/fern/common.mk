# Shared example build rules. Each platform Makefile sets PROGRAM, SOURCE,
# and OPT, then selects the platform's tool names and shell.
FERNC ?= fernc
GLCC ?= glcc
MKDIR ?= mkdir -p
REMOVE ?= rm -rf build
RUN ?= ./
EXE ?=
OBJ ?= .o
EXTRA_DEPS ?=
OPT ?= -O1
RUN_ARGS ?=

all: build/$(PROGRAM)$(EXE)

build/$(PROGRAM)$(EXE): build/$(PROGRAM)$(OBJ) driver.c
	$(GLCC) driver.c build/$(PROGRAM)$(OBJ) -o $@

build/$(PROGRAM)$(OBJ): $(SOURCE) $(EXTRA_DEPS)
	$(MKDIR) $(dir $@)
	$(FERNC) $(OPT) $< -o $@

run: build/$(PROGRAM)$(EXE)
	$(RUN)build/$(PROGRAM)$(EXE) $(RUN_ARGS)

clean:
	$(REMOVE)

.PHONY: all run clean
