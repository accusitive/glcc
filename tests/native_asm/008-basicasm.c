#include <stdio.h>
int main(void) {
    __asm__ ("nop");
    __asm__ volatile ("nop; nop");
    int x = 1;
    __asm__ ("movl %0, %%eax" :: "r"(x) : "eax");
    printf("ok\n");
    return 0;
}
