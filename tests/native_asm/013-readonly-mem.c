#include <stdio.h>
int main(void) {
    static const char msg[] = "hi";
    char a = 0, b = 0;
    __asm__ volatile ("movb (%1), %b0" : "=a"(a) : "r"(msg), "m"(msg));
    __asm__ volatile ("movb 1(%1), %b0" : "=a"(b) : "r"(msg), "m"(msg));
    printf("%c%c\n", a, b);
    return 0;
}
