# examples/kernel/Makefile — build and run the riscv64 bare-metal kernel.
#
#   make          build kernel.elf (freestanding ELF for qemu-system-riscv64)
#   make run      boot it under qemu (`-bios none`, UART on stdio)
#   make glcc     build the compiler this uses (cargo build at the repo root)
#   make clean
#
# Works from the repo root (`make -C examples/kernel`) or from this directory.
# Overridable: GLCC, GLCC_CC, QEMU, OBJCOPY.

KERNEL_DIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
ROOT := $(abspath $(KERNEL_DIR)../..)

GLCC ?= $(ROOT)/target/debug/glcc
GLCC_CC ?= $(ROOT)/tools/zig-cc
QEMU ?= qemu-system-riscv64
OBJCOPY ?= $(shell command -v llvm-objcopy 2>/dev/null || command -v objcopy 2>/dev/null)

BUILD := build
KERNEL_OBJ := $(BUILD)/kernel.o
USER_OBJ := $(BUILD)/user.o
USER_ELF := $(BUILD)/user.elf
USER_BIN := $(BUILD)/user.bin
USER_IMG_SRC := $(BUILD)/user_image.S
USER_IMG := $(BUILD)/user.img.o

TARGET_FLAGS := --target=riscv64 -ffreestanding
LINK_FLAGS := --target=riscv64 -nostdlib -nostartfiles -static

all: kernel.elf

$(KERNEL_OBJ): kernel.c
	@mkdir -p $(BUILD)
	GLCC_CC="$(GLCC_CC)" "$(GLCC)" $(TARGET_FLAGS) -c $< -o $@

$(USER_OBJ): user.c
	@mkdir -p $(BUILD)
	GLCC_CC="$(GLCC_CC)" "$(GLCC)" $(TARGET_FLAGS) -c $< -o $@

# The user program is a separate image: link it at 0x80400000, then turn
# it into a flat binary so its GOT stays inside its own U=1 page.
$(USER_ELF): $(USER_OBJ) user.ld
	GLCC_CC="$(GLCC_CC)" "$(GLCC)" $(LINK_FLAGS) -Wl,-T,$(KERNEL_DIR)user.ld $< -o $@

$(USER_BIN): $(USER_ELF)
	$(OBJCOPY) -O binary $< $@

# Emit the bytes into the assembly source rather than using .incbin: the
# assembler's cache keys on source text, but not on an included binary.
$(USER_IMG_SRC): $(USER_BIN)
	@{ \
	  echo '.section .user, "ax"'; \
	  od -An -v -tx1 $< | tr -s ' ' | sed -e 's/^ //' -e 's/ /, 0x/g' -e 's/^/\t.byte 0x/'; \
	} > $@

$(USER_IMG): $(USER_IMG_SRC)
	GLCC_CC="$(GLCC_CC)" "$(GLCC)" $(TARGET_FLAGS) -c $< -o $@

kernel.elf: $(KERNEL_OBJ) $(USER_IMG) tv.ld
	@command -v "$(GLCC)" >/dev/null || \
	  { echo "error: glcc not found ($(GLCC)); run 'make glcc' first" >&2; exit 1; }
	GLCC_CC="$(GLCC_CC)" "$(GLCC)" $(LINK_FLAGS) -Wl,-T,$(KERNEL_DIR)tv.ld \
	  $(KERNEL_OBJ) $(USER_IMG) -o $@

run: kernel.elf
	$(QEMU) -M virt -bios none -kernel kernel.elf -nographic -monitor none

glcc:
	cd "$(ROOT)" && cargo build

clean:
	rm -rf $(BUILD) kernel.elf

.DELETE_ON_ERROR:
.PHONY: all run glcc clean
