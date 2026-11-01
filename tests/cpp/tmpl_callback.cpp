// std::function-free callbacks: local functor objects, lambdas, and
// function pointers all flowing through the same template.
#include <cstdio>

template <class F>
int fold(F f, int init, const int *a, int n) {
  int acc = init;
  for (int i = 0; i < n; i++) acc = f(acc, a[i]);
  return acc;
}

struct Adder {
  int step;
  int operator()(int a, int b) const { return a + b + step; }
};

struct Maxer {
  int operator()(int a, int b) const { return a > b ? a : b; }
};

int main() {
  int a[5] = {4, 8, 15, 16, 23};
  Adder add{100};
  Maxer mx;
  auto mul = [](int x, int y) { return x * y; };

  int s = fold(add, 0, a, 5);
  int m = fold(mx, 0, a, 5);
  int p = fold(mul, 1, a, 3);

  printf("%d %d %d\n", s, m, p);
  return (s == 66 + 500 && m == 23 && p == 4 * 8 * 15) ? 0 : 1;
}
