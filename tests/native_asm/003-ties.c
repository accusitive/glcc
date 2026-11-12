#include <stdio.h>
int main(void) {
    long a = 7, b;
    __asm__ ("leaq 3(%1), %0" : "=r"(b) : "0"(a));
    int x = 5, y;
    __asm__ ("imull $3, %0" : "+r"(x));
    unsigned u = 0x11223344, w;
    __asm__ ("bswapl %0" : "=r"(w) : "0"(u));
    printf("%ld %d %u\n", b, x, w);
    return 0;
}
