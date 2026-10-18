/* Minimal riscv64 kernel for QEMU's virt machine, built by glcc.
 *
 * Freestanding: no libc, no startup files. `-bios none` makes qemu load
 * the ELF and start at the load address (0x80000000) in machine mode,
 * so `_start` is placed first via the `.text.boot` section.
 *
 * Console: NS16550A UART at 0x10000000 (the virt machine's "screen";
 * riscv has no VGA text buffer).
 * Exit: SiFive test finisher at 0x100000 (0x5555 = poweroff).
 */

#define UART0 ((volatile unsigned char *)0x10000000UL)
#define TEST_FINISHER ((volatile unsigned int *)0x100000UL)

/* Boot stack; `_start` points sp at its top. */
char kernel_stack[16384] __attribute__((aligned(16)));

static void uart_putc(char c) {
    /* LSR bit 5: transmit holding register empty. */
    while (!(UART0[5] & 0x20)) {
    }
    UART0[0] = (unsigned char)c;
}

static void uart_puts(const char *s) {
    while (*s) {
        uart_putc(*s++);
    }
}

static void uart_putu(unsigned long v) {
    char buf[20];
    int i = 0;
    if (v == 0) {
        uart_putc('0');
        return;
    }
    while (v) {
        buf[i++] = (char)('0' + v % 10);
        v /= 10;
    }
    while (i) {
        uart_putc(buf[--i]);
    }
}

static void uart_puthex(unsigned long v) {
    char buf[16];
    int i = 0;
    if (v == 0) {
        uart_putc('0');
        return;
    }
    while (v) {
        unsigned long d = v & 0xf;
        buf[i++] = (char)(d < 10 ? '0' + d : 'a' + d - 10);
        v >>= 4;
    }
    while (i) {
        uart_putc(buf[--i]);
    }
}

void kmain(void) {
    unsigned long hartid;
    unsigned long misa;

    __asm__ volatile("csrr %0, mhartid" : "=r"(hartid));
    __asm__ volatile("csrr %0, misa" : "=r"(misa));

    uart_puts("glcc riscv64 kernel: hello from bare metal\n");
    uart_puts("hart ");
    uart_putu(hartid);
    uart_puts(", misa 0x");
    uart_puthex(misa);
    uart_puts("\n");
    uart_puts("inline asm csrr works; UART at 0x10000000\n");

    *TEST_FINISHER = 0x5555;
    for (;;) {
    }
}

__attribute__((naked, section(".text.boot"))) void _start(void) {
    __asm__ volatile(
        "la sp, kernel_stack + 16384\n"
        "call kmain\n"
        "j .\n");
}
