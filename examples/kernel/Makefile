# examples/kernel/Makefile — build and run the minimal riscv64 bare-metal kernel.
#
#   make          build kernel.elf (freestanding ELF for qemu-system-riscv64)
#   make run      boot it under qemu (`-bios none`, UART on stdio)
#   make glcc     build the compiler this uses (cargo build at the repo root)
#   make clean
#
# Works from the repo root (`make -C examples/kernel`) or from this directory.
# Overridable: GLCC, GLCC_CC, QEMU.

KERNEL_DIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
ROOT := $(abspath $(KERNEL_DIR)../..)

GLCC ?= $(ROOT)/target/debug/glcc
GLCC_CC ?= $(ROOT)/tools/zig-cc
QEMU ?= qemu-system-riscv64

# Freestanding: no libc, no startup files; tv.ld places the image in RAM
# at 0x80000000 with `_start` first.
GLCC_FLAGS := --target=riscv64 -nostdlib -nostartfiles -static \
              -Wl,-T,$(KERNEL_DIR)tv.ld

all: kernel.elf

kernel.elf: kernel.c tv.ld
	@command -v "$(GLCC)" >/dev/null || \
	  { echo "error: glcc not found ($(GLCC)); run 'make glcc' first" >&2; exit 1; }
	GLCC_CC="$(GLCC_CC)" "$(GLCC)" $(GLCC_FLAGS) kernel.c -o $@

run: kernel.elf
	$(QEMU) -M virt -bios none -kernel kernel.elf -nographic -monitor none

glcc:
	cd "$(ROOT)" && cargo build

clean:
	rm -f kernel.elf

.DELETE_ON_ERROR:
.PHONY: all run glcc clean
