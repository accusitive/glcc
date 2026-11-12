#include <stdio.h>
int main(void) {
    unsigned long x = 0x00000001000000FF, hi;
    __asm__ ("movq %1, %0; shrq $32, %0" : "=r"(hi) : "r"(x));
    unsigned char lo = (unsigned char)x, up;
    __asm__ ("movb %b1, %b0" : "=a"(up) : "0"(lo));
    unsigned w = 0xAABBCCDD;
    __asm__ ("movl %k1, %k0" : "=a"(w) : "0"(w));
    printf("%lu %u %u\n", hi, up, w);
    return 0;
}
