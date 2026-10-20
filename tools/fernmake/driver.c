/* C harness for fernmake: pass a Makefile and optional target on the
   command line. The default Makefile is demo.mk in the current dir. */
#include <stdio.h>

long long run_makefile(const char *path, const char *target);

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "demo.mk";
    const char *target = argc > 2 ? argv[2] : "";
    return run_makefile(path, target) == 0 ? 0 : 1;
}
