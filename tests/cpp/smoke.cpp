// Smoke test for the EDG -> MIR back end (v1 slice): arithmetic,
// control flow, globals with constant initializers, extern calls,
// string literals, a struct passed by value and returned, and a
// function pointer call.

#include <cstdio>

struct Point {
  int x, y;
};

int g_counter = 42;
int g_table[4] = {1, 2, 3};
const char *g_name = "smoke";

Point make_point(int x, int y) {
  Point p;
  p.x = x;
  p.y = y;
  return p;
}

int sum_point(Point p) { return p.x + p.y; }

int apply_twice(int (*f)(int), int v) { return f(f(v)); }

int triple(int x) { return x * 3; }

int classify(int c) {
  switch (c) {
    case 0:
      return 100;
    case 1:
      return 200;
    case 2:
      return 300;
    default:
      return -1;
  }
}

int loops(int n) {
  int total = 0;
  for (int i = 0; i < n; i++) {
    if (i == 2) continue;
    total += i;
    if (total > 50) break;
  }
  int k = n;
  while (k > 0) {
    total += 1;
    k--;
  }
  do {
    total -= 1;
  } while (false);
  return total;
}

int main() {
  Point p = make_point(3, 4);
  int a = sum_point(p);
  int b = apply_twice(triple, 2);
  int c = classify(2);
  int d = loops(5);
  int e = g_counter + g_table[2];
  printf("%s: %d %d %d %d %d\n", g_name, a, b, c, d, e);
  return (a == 7 && b == 18 && c == 300 && d == 12 && e == 45) ? 0 : 1;
}
