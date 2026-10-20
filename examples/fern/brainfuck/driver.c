/* C harness for brainfuck.fern: `demo()` prints the five built-in
   programs, and `answer()` returns their rolling checksum. With a path
   argument, `run_file()` loads and runs a Brainfuck file instead. */
#include <stdio.h>

void demo(void);
long run_file(const char *path);
long answer(void);

int main(int argc, char **argv) {
    if (argc > 1) {
        return run_file(argv[1]) == 0 ? 0 : 1;
    }
    demo();
    long got = answer();
    printf("checksum = %ld (expected 49729618)\n", got);
    return got == 49729618 ? 0 : 1;
}
