// long double (x87 80-bit, printf %Lf): arithmetic that stays exact and
// narrowing to double / int.
#include <cstdio>

int main() {
  long double a = 1.5L;
  long double b = a * 4.0L;        // 6.0
  long double c = b / 3.0L;        // 2.0
  long double d = c + 0.125L;      // 2.125
  double e = (double)d;
  int f = (int)(d * 8.0L);         // 17
  long double g = -d;

  printf("%.3Lf %.3Lf %.3Lf %.3Lf\n", b, c, d, g);
  printf("%.1f %d %d\n", e, f, (int)(d > 2.0L) + (int)(g < 0.0L));
  return (b == 6.0L && c == 2.0L && d == 2.125L && e == 2.125 && f == 17)
             ? 0
             : 1;
}
