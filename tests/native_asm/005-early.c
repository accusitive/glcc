#include <stdio.h>
int main(void) {
    long q = 100, r;
    __asm__ ("divq %4" : "=a"(q), "=d"(r) : "0"(q), "1"(0L), "r"(7L));
    long q2 = -100, r2;
    __asm__ ("cqto; idivq %4" : "=a"(q2), "=d"(r2) : "0"(q2), "1"(q2), "r"(7L));
    printf("%ld %ld %ld %ld\n", q, r, q2, r2);
    return 0;
}
