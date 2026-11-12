#include <stdio.h>
static long f(long x) {
    long y = 0;
    // The template writes rbx and r15: the prologue must save the
    // caller's copies of whatever the function itself holds there.
    __asm__ volatile ("movq %1, %%rbx; addq $1, %%rbx; movq %%rbx, %0; movq $9, %%r15" : "=r"(y) : "r"(x) : "r15");
    return y;
}
int main(void) {
    long a = f(41);
    long b = f(99);
    printf("%ld %ld\n", a, b);
    return 0;
}
