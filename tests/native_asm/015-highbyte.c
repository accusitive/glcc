#include <stdio.h>
int main(void) {
    unsigned char x = 0x5A, via_q;
    __asm__ ("movb %1, %0" : "=Q"(via_q) : "r"(x));
    unsigned int v = 0xABCD1234;
    unsigned char hb;
    __asm__ ("movb %h1, %0" : "=r"(hb) : "q"(v));
    printf("%x %x\n", via_q, hb);
    return 0;
}
