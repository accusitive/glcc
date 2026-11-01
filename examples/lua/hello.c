/* C compiled to a portable Lua 5.1 script: make && make run. */
#include <stdio.h>

struct point {
    int x;
    int y;
};

static int sum(const int *values, int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        total += values[i];
    }
    return total;
}

int main(void) {
    struct point p = {6, 7};
    int values[] = {1, 2, 3, 4};
    printf("point=%d.%d sum=%d\n", p.x, p.y, sum(values, 4));
    return 0;
}
