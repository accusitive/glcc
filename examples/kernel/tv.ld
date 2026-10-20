/* Minimal linker script for the riscv64 kernel.
 *
 * The image sits at the base of the virt machine's RAM. qemu starts at
 * the load address, not the ELF entry point, so `_start` must be first:
 * it lives in `.text.boot`, placed ahead of everything else.
 *
 * The flat user image (build/user.bin) is embedded in `.user` at
 * 0x80400000; the kernel page tables mark that region U=1. `.eh_frame`
 * is claimed explicitly: lld would otherwise place the synthesized
 * `.eh_frame_hdr` orphan ahead of `.text`.
 */
ENTRY(_start)
SECTIONS
{
    . = 0x80000000;
    .text : { *(.text.boot) *(.text*) }
    .rodata : { *(.rodata*) *(.eh_frame*) }
    .data : { *(.data*) }
    .bss : { *(.bss*) *(COMMON) }

    . = 0x80400000;
    .user : { *(.user) }
}
