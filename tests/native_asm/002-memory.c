#include <stdio.h>
int main(void) {
    long v = 10;
    __asm__ ("addq $5, %0" : "+m"(v));
    long w = 3;
    __asm__ ("movq %1, %0" : "=m"(w) : "r"(v));
    int arr[4] = {1, 2, 3, 4};
    __asm__ ("addl $10, %0" : "+m"(arr[2]));
    printf("%ld %ld %d\n", v, w, arr[2]);
    return 0;
}
