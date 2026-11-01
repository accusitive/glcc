// constexpr functions and variables: constant folding at compile time,
// static_assert on a constexpr call, constexpr array bounds.
#include <cstdio>

constexpr int fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); }

constexpr int kFive = fact(5);
static_assert(fact(5) == 120, "constexpr fact works");
static_assert(kFive == 120, "constexpr variable works");

constexpr int fib(int n) {
  int a = 0, b = 1;
  for (int i = 0; i < n; i++) {
    int t = a + b;
    a = b;
    b = t;
  }
  return a;
}

int buffer[fact(4)];

int main() {
  constexpr int f7 = fact(7);
  printf("%d %d %d %zu\n", kFive, f7, fib(10), sizeof(buffer) / sizeof(int));
  return (kFive == 120 && f7 == 5040 && fib(10) == 55 &&
          sizeof(buffer) / sizeof(int) == 24)
             ? 0
             : 1;
}
