/* C harness for brainfuck.fern: `answer()` returns a rolling checksum
   of every byte the five test programs emitted. */
#include <stdio.h>

long answer(void);

int main(void) {
    long got = answer();
    printf("checksum = %ld (expected 49729618)\n", got);
    return got == 49729618 ? 0 : 1;
}
