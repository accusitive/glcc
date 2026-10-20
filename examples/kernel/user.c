/* User-mode program for the riscv64 kernel example.
 *
 * Runs at 0x80400000 in U-mode with no privileges: it can only reach
 * its own pages and gets the kernel's attention through `ecall`.
 */

#define SYS_write 1
#define SYS_exit 2

/* C ABI in, Linux-style syscall ABI out: number in a7, args in a0-a2. */
__attribute__((naked)) long usyscall(long nr, long a, long b, long c) {
    __asm__ volatile(
        "mv a7, a0\n"
        "mv a0, a1\n"
        "mv a1, a2\n"
        "mv a2, a3\n"
        "ecall\n"
        "ret\n");
}

static long uwrite(const char *buf, unsigned long len) {
    return usyscall(SYS_write, 1, (long)buf, (long)len);
}

static const char msg[] = "hello from U-mode (unprivileged)\n";
static const char again[] = "write syscall returned to user space\n";

void ustart_c(void) {
    long n;

    n = uwrite(msg, sizeof(msg) - 1);
    if (n != (long)(sizeof(msg) - 1)) {
        usyscall(SYS_exit, 1, 0, 0);
    }
    uwrite(again, sizeof(again) - 1);

    usyscall(SYS_exit, 0, 0, 0);
    for (;;) {
    }
}

/* Fixed entry point: user.ld places this section first at 0x80400000. */
__attribute__((naked, section(".text._ustart"))) void _ustart(void) {
    __asm__ volatile(
        "call ustart_c\n"
        "j .\n");
}
