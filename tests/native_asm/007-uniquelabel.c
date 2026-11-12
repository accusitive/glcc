#include <stdio.h>
static long spin(long n) {
    __asm__ volatile (
        "1: decq %0\n\t"
        "cmpq $0, %0\n\t"
        "jne 1b"
        : "+r"(n));
    return n;
}
int main(void) {
    // Two instantiations of the same template: %= keeps local labels unique.
    long a = spin(4);
    long b = spin(9);
    printf("%ld %ld\n", a, b);
    return 0;
}
