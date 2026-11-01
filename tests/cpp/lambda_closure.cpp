// Closures stored in auto variables and passed to a template that calls
// them -- no std::function anywhere.
#include <cstdio>

template <class F>
int apply(F f, int x) { return f(x); }

template <class F, class G>
int compose(F f, G g, int x) { return f(g(x)); }

int call_ptr(int (*f)(int), int x) { return f(x); }

static int twice(int x) { return 2 * x; }

int main() {
  int n = 7;
  auto inc = [](int x) { return x + 1; };
  auto addn = [n](int x) { return x + n; };

  int a = inc(10);
  int b = addn(10);
  int c = apply(inc, 100);
  int d = compose(inc, addn, 1);
  int e = call_ptr(twice, 21);

  printf("%d %d %d %d %d\n", a, b, c, d, e);
  return (a == 11 && b == 17 && c == 101 && d == 9 && e == 42) ? 0 : 1;
}
