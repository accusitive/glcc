# Demo makefile for the Windows build: run with
#   ./build/fernmake.exe demo.win.mk run
SHELL = powershell
.SHELLFLAGS = -NoProfile -Command
GLCC ?= ../../target/debug/glcc.exe

all: build/demo.exe

build/demo.exe: demo.c
	New-Item -ItemType Directory -Force -Path build
	$(GLCC) demo.c -o build/demo.exe

run: build/demo.exe
	& ./build/demo.exe

clean:
	if (Test-Path build) { Remove-Item -Recurse -Force build }

.PHONY: all run clean
