#include <stdio.h>
int main(void) {
    long x = 41, y;
    __asm__ ("leaq 1(%1), %0" : "=r"(y) : "r"(x));
    int a = 3, b;
    __asm__ ("movl %1, %0" : "=r"(b) : "r"(a));
    unsigned char c = 200, d;
    __asm__ ("movb %1, %0" : "=r"(d) : "r"(c));
    printf("%ld %d %d\n", y, b, d);
    return 0;
}
