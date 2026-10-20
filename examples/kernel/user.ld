/* Minimal linker script for the user-mode image.
 *
 * Everything lives in the kernel's U=1 code page at 0x80400000. The
 * entry trampoline is placed first so the kernel can jump to a fixed
 * address after loading the flat image. The GOT stays here too: user
 * code is compiled PIC, and it must never reach into the kernel's GOT.
 */
ENTRY(_ustart)
SECTIONS
{
    . = 0x80400000;
    .text : { *(.text._ustart) *(.text*) }
    .rodata : { *(.rodata*) *(.eh_frame*) *(.srodata*) }
    .data : { *(.data*) *(.sdata*) *(.got*) *(.got.plt*) }
    .bss : { *(.bss*) *(.sbss*) *(COMMON) }
}
