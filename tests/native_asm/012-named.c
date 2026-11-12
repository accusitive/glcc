#include <stdio.h>
int main(void) {
    long a = 30, b = 12, sum = 0;
    __asm__ ("leaq (%[x], %[y]), %[z]" : [z] "=r"(sum) : [x] "r"(a), [y] "r"(b));
    long v = 5;
    __asm__ ("addq %[n], %0" : "+m"(v) : [n] "i"(8));
    printf("%ld %ld\n", sum, v);
    return 0;
}
