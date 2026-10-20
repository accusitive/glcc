/* Minimal riscv64 kernel for QEMU's virt machine, built by glcc.
 *
 * Freestanding: no libc, no startup files. `-bios none` makes qemu load
 * the ELF and start at the load address (0x80000000) in machine mode.
 *
 * Boot flow:
 *   _start (M-mode)       PMP + trap delegation, then mret into S-mode
 *   smain  (S-mode)       Sv39 page tables, trap vector, sret into U-mode
 *   _ustart (U-mode)      user.c; prints through the write syscall
 *
 * The kernel identity-maps its own RAM plus the devices it uses. User
 * code and stack live in separate 2 MiB leaf mappings with the U bit set,
 * so U-mode cannot touch kernel memory. Traps from U-mode land on a
 * dedicated stack via sscratch.
 *
 * Console: NS16550A UART at 0x10000000 (the virt machine's "screen";
 * riscv has no VGA text buffer).
 * Exit: SiFive test finisher at 0x100000 (0x5555 = poweroff).
 */

#define UART0 ((volatile unsigned char *)0x10000000UL)
#define TEST_FINISHER ((volatile unsigned int *)0x100000UL)

#define RAM_BASE 0x80000000UL
#define USER_CODE_BASE 0x80400000UL
#define USER_ENTRY USER_CODE_BASE
#define USER_STACK_BASE 0x80600000UL
#define USER_STACK_SIZE 16384UL

/* Sv39 page-table entry flags. */
#define PTE_V (1UL << 0)
#define PTE_R (1UL << 1)
#define PTE_W (1UL << 2)
#define PTE_X (1UL << 3)
#define PTE_U (1UL << 4)
#define PTE_A (1UL << 6)
#define PTE_D (1UL << 7)
#define PTE_LEAF (PTE_V | PTE_R | PTE_W | PTE_X | PTE_A | PTE_D)

/* sstatus bits. */
#define SSTATUS_SPIE (1UL << 5)
#define SSTATUS_SPP (1UL << 8)
#define SSTATUS_SUM (1UL << 18)

/* Syscall numbers shared with user.c. */
#define SYS_write 1
#define SYS_exit 2

#define USER_END (USER_STACK_BASE + USER_STACK_SIZE)

/* Boot stack; `_start` points sp at its top. */
char kernel_stack[16384] __attribute__((aligned(16)));

/* Trap stack; U-mode traps swap sp with sscratch to land here. */
static char trap_stack[8192] __attribute__((aligned(16)));

/* Sv39 root, plus the level-1 tables it points at. */
static unsigned long root_pt[512] __attribute__((aligned(4096)));
static unsigned long low_pt[512] __attribute__((aligned(4096)));
static unsigned long ram_pt[512] __attribute__((aligned(4096)));

struct trap_frame {
    unsigned long x[32];
};

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

static void poweroff(unsigned long code) {
    if (code == 0) {
        *TEST_FINISHER = 0x5555;
    } else {
        *TEST_FINISHER = (unsigned int)((code << 16) | 0x3333);
    }
    for (;;) {
    }
}

static void fail(const char *what, unsigned long a, unsigned long b) {
    uart_puts("kernel panic: ");
    uart_puts(what);
    uart_puts(" (0x");
    uart_puthex(a);
    uart_puts(", 0x");
    uart_puthex(b);
    uart_puts(")\n");
    poweroff(1);
}

static unsigned long pte_leaf(unsigned long pa, unsigned long flags) {
    return ((pa >> 12) << 10) | flags;
}

static unsigned long pte_table(unsigned long pa) {
    return ((pa >> 12) << 10) | PTE_V;
}

static void clear_table(unsigned long *t) {
    int i;
    for (i = 0; i < 512; i++) {
        t[i] = 0;
    }
}

/* Identity map: kernel RAM (U=0), UART/test finisher (U=0), and two
 * user 2 MiB leaves (U=1) for code/data and stack. */
static void setup_vm(void) {
    unsigned long kernel_flags = PTE_LEAF;
    unsigned long user_code_flags = PTE_V | PTE_R | PTE_X | PTE_U | PTE_A | PTE_D;
    unsigned long user_stack_flags = PTE_V | PTE_R | PTE_W | PTE_U | PTE_A | PTE_D;

    clear_table(root_pt);
    clear_table(low_pt);
    clear_table(ram_pt);

    /* Root index 0 covers 0x00000000, root index 2 covers 0x80000000. */
    root_pt[0] = pte_table((unsigned long)low_pt);
    root_pt[2] = pte_table((unsigned long)ram_pt);

    /* 2 MiB leaves: 0x0 (test finisher) and 0x10000000 (UART). */
    low_pt[0] = pte_leaf(0x0, kernel_flags);
    low_pt[0x10000000UL >> 21] = pte_leaf(0x10000000UL, kernel_flags);

    ram_pt[0] = pte_leaf(RAM_BASE, kernel_flags);
    ram_pt[1] = pte_leaf(RAM_BASE + 0x200000UL, kernel_flags);
    ram_pt[2] = pte_leaf(USER_CODE_BASE, user_code_flags);
    ram_pt[3] = pte_leaf(USER_STACK_BASE, user_stack_flags);
}

static void enable_vm(void) {
    unsigned long satp = (8UL << 60) | ((unsigned long)root_pt >> 12);
    __asm__ volatile("csrw satp, %0" : : "r"(satp));
    __asm__ volatile("sfence.vma zero, zero");
}

static long sys_write(unsigned long fd, const char *buf, unsigned long len) {
    unsigned long i;

    if (fd != 1 && fd != 2) {
        return -1;
    }
    if (len > USER_END - USER_CODE_BASE) {
        return -1;
    }
    if ((unsigned long)buf < USER_CODE_BASE || (unsigned long)buf > USER_END - len) {
        return -1;
    }
    for (i = 0; i < len; i++) {
        uart_putc(buf[i]);
    }
    return (long)len;
}

static void sys_exit(unsigned long code) {
    uart_puts("\nuser program exited with code ");
    uart_putu(code);
    uart_puts("\n");
    poweroff(code);
}

void trap_handler(struct trap_frame *tf) {
    unsigned long scause;
    unsigned long sepc;
    unsigned long stval;
    unsigned long sstatus;

    __asm__ volatile("csrr %0, scause" : "=r"(scause));
    __asm__ volatile("csrr %0, sepc" : "=r"(sepc));
    __asm__ volatile("csrr %0, stval" : "=r"(stval));
    __asm__ volatile("csrr %0, sstatus" : "=r"(sstatus));

    if (sstatus & SSTATUS_SPP) {
        fail("trap from S-mode", scause, sepc);
    }

    if (scause == 8) {
        unsigned long nr = tf->x[17]; /* a7 */
        unsigned long a0 = tf->x[10];
        unsigned long a1 = tf->x[11];
        unsigned long a2 = tf->x[12];

        if (nr == SYS_write) {
            tf->x[10] = (unsigned long)sys_write(a0, (const char *)a1, a2);
        } else if (nr == SYS_exit) {
            sys_exit(a0);
        } else {
            uart_puts("unknown syscall ");
            uart_putu(nr);
            uart_puts("\n");
            sys_exit(127);
        }
        __asm__ volatile("csrw sepc, %0" : : "r"(sepc + 4));
        return;
    }

    if (scause == 12 || scause == 13 || scause == 15) {
        uart_puts("user page fault: scause 0x");
        uart_puthex(scause);
        uart_puts(" sepc 0x");
        uart_puthex(sepc);
        uart_puts(" stval 0x");
        uart_puthex(stval);
        uart_puts("\n");
        sys_exit(139);
    }

    fail("unexpected trap", scause, sepc);
}

__attribute__((naked)) void trap_entry(void) {
    __asm__ volatile(
        /* sscratch holds the trap stack top while in U-mode. */
        "csrrw sp, sscratch, sp\n"
        "addi sp, sp, -256\n"
        "sd x1, 8(sp)\n"
        "sd x3, 24(sp)\n"
        "sd x4, 32(sp)\n"
        "sd x5, 40(sp)\n"
        "sd x6, 48(sp)\n"
        "sd x7, 56(sp)\n"
        "sd x8, 64(sp)\n"
        "sd x9, 72(sp)\n"
        "sd x10, 80(sp)\n"
        "sd x11, 88(sp)\n"
        "sd x12, 96(sp)\n"
        "sd x13, 104(sp)\n"
        "sd x14, 112(sp)\n"
        "sd x15, 120(sp)\n"
        "sd x16, 128(sp)\n"
        "sd x17, 136(sp)\n"
        "sd x18, 144(sp)\n"
        "sd x19, 152(sp)\n"
        "sd x20, 160(sp)\n"
        "sd x21, 168(sp)\n"
        "sd x22, 176(sp)\n"
        "sd x23, 184(sp)\n"
        "sd x24, 192(sp)\n"
        "sd x25, 200(sp)\n"
        "sd x26, 208(sp)\n"
        "sd x27, 216(sp)\n"
        "sd x28, 224(sp)\n"
        "sd x29, 232(sp)\n"
        "sd x30, 240(sp)\n"
        "sd x31, 248(sp)\n"
        "csrr t0, sscratch\n"
        "sd t0, 16(sp)\n"
        "mv a0, sp\n"
        "call trap_handler\n"
        /* Restore everything but sp/t0/t1, then switch stacks. */
        "ld x1, 8(sp)\n"
        "ld x3, 24(sp)\n"
        "ld x4, 32(sp)\n"
        "ld x7, 56(sp)\n"
        "ld x8, 64(sp)\n"
        "ld x9, 72(sp)\n"
        "ld x10, 80(sp)\n"
        "ld x11, 88(sp)\n"
        "ld x12, 96(sp)\n"
        "ld x13, 104(sp)\n"
        "ld x14, 112(sp)\n"
        "ld x15, 120(sp)\n"
        "ld x16, 128(sp)\n"
        "ld x17, 136(sp)\n"
        "ld x18, 144(sp)\n"
        "ld x19, 152(sp)\n"
        "ld x20, 160(sp)\n"
        "ld x21, 168(sp)\n"
        "ld x22, 176(sp)\n"
        "ld x23, 184(sp)\n"
        "ld x24, 192(sp)\n"
        "ld x25, 200(sp)\n"
        "ld x26, 208(sp)\n"
        "ld x27, 216(sp)\n"
        "ld x28, 224(sp)\n"
        "ld x29, 232(sp)\n"
        "ld x30, 240(sp)\n"
        "ld x31, 248(sp)\n"
        "mv t0, sp\n"
        "csrr t1, sscratch\n"
        "addi sp, t0, 256\n"
        "csrw sscratch, sp\n"
        "mv sp, t1\n"
        "ld t1, 48(t0)\n"
        "ld t0, 40(t0)\n"
        "sret\n");
}

__attribute__((naked)) void enter_user(unsigned long entry, unsigned long stack) {
    __asm__ volatile(
        "csrw sepc, a0\n"
        "mv sp, a1\n"
        "sret\n");
}

void smain(void) {
    unsigned long satp;
    unsigned long sstatus;

    __asm__ volatile("csrw stvec, %0" : : "r"((unsigned long)trap_entry));
    __asm__ volatile("csrw sscratch, %0" : : "r"((unsigned long)(trap_stack + sizeof(trap_stack))));

    uart_puts("glcc riscv64 kernel: hello from S-mode\n");

    setup_vm();
    enable_vm();
    __asm__ volatile("csrr %0, satp" : "=r"(satp));

    uart_puts("Sv39 paging enabled, satp 0x");
    uart_puthex(satp);
    uart_puts("\n");
    uart_puts("kernel 0x");
    uart_puthex(RAM_BASE);
    uart_puts("-0x");
    uart_puthex(RAM_BASE + 0x400000UL);
    uart_puts(" (U=0)\n");
    uart_puts("user code 0x");
    uart_puthex(USER_CODE_BASE);
    uart_puts("-0x");
    uart_puthex(USER_STACK_BASE);
    uart_puts(", stack 0x");
    uart_puthex(USER_STACK_BASE);
    uart_puts("-0x");
    uart_puthex(USER_END);
    uart_puts(" (U=1)\n");

    __asm__ volatile("csrr %0, sstatus" : "=r"(sstatus));
    sstatus |= SSTATUS_SUM | SSTATUS_SPIE;
    sstatus &= ~SSTATUS_SPP;
    __asm__ volatile("csrw sstatus, %0" : : "r"(sstatus));

    uart_puts("entering U-mode at 0x");
    uart_puthex(USER_ENTRY);
    uart_puts("\n");

    enter_user(USER_ENTRY, USER_END);
    poweroff(1);
}

__attribute__((naked, section(".text.boot"))) void _start(void) {
    __asm__ volatile(
        "la sp, kernel_stack + 16384\n"
        /* Allow S/U access to the whole address space. */
        "li t0, -1\n"
        "csrw pmpaddr0, t0\n"
        "li t0, 0x0f\n"
        "csrw pmpcfg0, t0\n"
        /* Delegate exceptions (ecall from U, faults) to S-mode. */
        "li t0, 0xffff\n"
        "csrw medeleg, t0\n"
        "csrw mideleg, zero\n"
        "la t0, smain\n"
        "csrw mepc, t0\n"
        /* mstatus.MPP = S (1), everything else off for now. */
        "li t0, 0x800\n"
        "csrw mstatus, t0\n"
        "csrw satp, zero\n"
        "mret\n"
        "j .\n");
}
