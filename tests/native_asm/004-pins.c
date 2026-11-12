#include <stdio.h>
int main(void) {
    unsigned lo, hi;
    __asm__ volatile ("movl $0, %%eax; cpuid" : "=a"(lo), "=d"(hi) : : "ebx", "ecx", "memory");
    unsigned a1, d1, a2, d2;
    __asm__ volatile ("rdtsc" : "=a"(a1), "=d"(d1));
    __asm__ volatile ("rdtsc" : "=a"(a2), "=d"(d2));
    printf("%u %d\n", lo & 0xff, a2 >= a1);
    return 0;
}
