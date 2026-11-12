#include <stdio.h>
int main(void) {
    long x = 0, y = 0;
    __asm__ ("movq %1, %0" : "=r"(x) : "i"(42));
    __asm__ ("addq %1, %0" : "+r"(y) : "n"(-1));
    __asm__ ("addq %1, %0" : "+r"(y) : "i"(100));
    printf("%ld %ld\n", x, y);
    return 0;
}
