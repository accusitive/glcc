// Function overloading: resolution across int / double / const char *
// overloads, plus overload by parameter count.
#include <cstdio>

const char *pick(int x) { return "int"; }
const char *pick(double x) { return "double"; }
const char *pick(const char *x) { return "string"; }
int pick(int a, int b) { return a * 100 + b; }

int main() {
  const char *s = "lit";
  int two = pick(3, 4);
  printf("%s %s %s %d\n", pick(1), pick(1.5), pick(s), two);
  const char *r1 = pick(1);
  const char *r2 = pick(1.5);
  return (two == 304 && r1[0] == 'i' && r2[0] == 'd') ? 0 : 1;
}
