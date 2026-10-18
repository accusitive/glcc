/* C harness for showcase.fern: fernc emits `answer()`; print it. */
#include <stdio.h>

long answer(void);

int main(void) {
    long got = answer();
    printf("answer = %ld (expected 137)\n", got);
    return got == 137 ? 0 : 1;
}
