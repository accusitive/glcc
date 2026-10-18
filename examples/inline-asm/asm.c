/* Inline asm in glcc: make && make run.
 *
 * Cranelift has no inline asm, so glcc compiles each site to an
 * out-of-line wrapper that runs the template with statically allocated
 * registers (docs/inline-asm.md). Everything here is tier T0/T1:
 * operand templates, fixed registers, clobbers, memory operands, and
 * asm goto. Register variables and reading %rsp are out of scope.
 */
#include <stdio.h>
#include <string.h>

/* One tied output ("+r") and one input: addl %1, %0. */
static int asm_add(int a, int b) {
    __asm__("addl %1, %0" : "+r"(a) : "r"(b));
    return a;
}

/* An instruction with no C spelling. */
static unsigned long asm_byteswap(unsigned long v) {
    __asm__("bswap %0" : "+r"(v));
    return v;
}

/* An "m" output: the wrapper substitutes the operand's address. */
static void asm_store(int *p, int v) {
    __asm__ volatile("movl %1, %0" : "=m"(*p) : "r"(v));
}

/* The compiler barrier: nothing may move across this point. */
static int asm_barrier(int x) {
    __asm__ volatile("" : "+r"(x) : : "memory");
    return x;
}

/* asm goto: the template jumps to a C label. The wrapper turns the
 * branch into a selector and the caller dispatches on it. */
static int asm_is_odd(int x) {
    __asm__ goto("testl $1, %0; jnz %l[odd]" : : "r"(x) : "cc" : odd);
    return 0;
odd:
    return 1;
}

#if defined(__x86_64__)
/* Fixed-register outputs: rdtsc writes edx:eax. */
static unsigned long long read_tsc(void) {
    unsigned int lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)hi << 32) | lo;
}

/* A tied input pinning eax to leaf 0, with the other cpuid outputs. */
static void cpu_vendor(char out[13]) {
    unsigned int a = 0, b, c, d;
    __asm__ volatile("cpuid" : "+a"(a), "=b"(b), "=c"(c), "=d"(d));
    memcpy(out + 0, &b, 4);
    memcpy(out + 4, &d, 4);
    memcpy(out + 8, &c, 4);
    out[12] = '\0';
}
#endif

int main(void) {
    printf("add(20, 22)               = %d\n", asm_add(20, 22));
    printf("bswap(0x1122334455667788) = 0x%lx\n", asm_byteswap(0x1122334455667788UL));

    int v = 0;
    asm_store(&v, 7);
    printf("store 7 through memory    = %d\n", v);
    printf("barrier(5)                = %d\n", asm_barrier(5));
    printf("is_odd(7), is_odd(8)      = %d, %d\n", asm_is_odd(7), asm_is_odd(8));

#if defined(__x86_64__)
    char vendor[13];
    cpu_vendor(vendor);
    printf("cpu vendor                = %s\n", vendor);

    unsigned long long t0 = read_tsc();
    volatile int sink = 0;
    for (int i = 0; i < 1000000; i++)
        sink += i;
    unsigned long long t1 = read_tsc();
    printf("rdtsc over 1e6 adds       = %llu ticks\n", t1 - t0);
    (void)sink;
#else
    printf("(the rdtsc/cpuid part is x86-64 only)\n");
#endif
    return 0;
}
