#include <stdio.h>
static int target(int x) { return x * 3 + 1; }
int main(void) {
    void *fp = (void *)target;
    long arg = 14;
    long r;
    __asm__ volatile (
        "movq %2, %%rdi; call *%1"
        : "=a"(r)
        : "r"(fp), "r"(arg)
        : "rdx", "rsi", "rdi", "rcx", "r8", "r9", "r10", "r11", "memory");
    printf("%ld\n", r);
    return 0;
}
