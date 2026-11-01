// Lambdas: capture by value, capture by reference, mixed capture lists,
// and explicit captures.
#include <cstdio>

int main() {
  int base = 100;
  int counter = 0;

  auto byval = [base](int x) { return base + x; };
  auto byref = [&counter](int x) { counter += x; return counter; };
  auto mixed = [base, &counter](int x) {
    counter = counter + x;
    return base + counter;
  };
  auto explicit_cap = [base, &base_is_gone = base](int x) {
    base_is_gone += x;
    return base;
  };

  int a = byval(5);
  byref(3);
  int b = counter;
  int c = mixed(2);
  int d = explicit_cap(1);

  printf("%d %d %d %d %d\n", a, b, c, d, counter);
  // d reads the by-value copy of base (100); the init-capture reference
  // bumped the real base to 101 without touching counter.
  return (a == 105 && b == 3 && c == 105 && d == 100 && counter == 5) ? 0 : 1;
}
