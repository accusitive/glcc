// auto and decltype: deduction from initializers, decltype of variables
// and expressions, decltype in trailing return type.
#include <cstdio>

int gi = 10;
double gd = 1.5;

auto ident(int x) -> decltype(x) { return x; }

int main() {
  auto a = gi;
  auto b = 2.5;
  auto c = 3u;
  decltype(a + 1L) big = 7;
  decltype(gd) d = 2.25;
  decltype(ident(0)) e = ident(9);

  printf("%d %.2f %u %lld %.2f %d\n", a, b, c, (long long)big, d, e);
  return (a == 10 && b > 2.4 && b < 2.6 && c == 3u && big == 7 && d == 2.25 &&
          e == 9)
             ? 0
             : 1;
}
