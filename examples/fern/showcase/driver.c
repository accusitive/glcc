/* C harness for showcase.fern: fernc emits `answer()`; print it. */
#include <stdio.h>

long long answer(void);

int main(void) {
    long long got = answer();
    printf("answer = %lld (expected 137)\n", got);
    return got == 137 ? 0 : 1;
}
