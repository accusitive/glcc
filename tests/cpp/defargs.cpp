// Default arguments: trailing defaults, call sites that override some or
// none of them, and a default expression referencing a global.
#include <cstdio>

int g_scale = 10;

int add(int a, int b = 2, int c = 3) { return a + b * g_scale + c * 100; }

void bump(int *x, int by = 1) { *x += by; }

int main() {
  int v = 10;
  bump(&v);
  bump(&v, 5);
  printf("%d %d %d %d\n", add(1), add(1, 2), add(1, 2, 4), v);
  return (v == 16 && add(1) == 321 && add(1, 2) == 321 && add(1, 2, 4) == 421)
             ? 0
             : 1;
}
