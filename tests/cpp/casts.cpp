// Focused test: integer/float/pointer conversions, comparisons and
// boolean control flow, shift/division signedness.

#include <cstdio>

int main() {
  int i = -7;
  unsigned u = 4000000000u;
  long long big = i;
  short s = (short)i;
  char ch = 'A';
  double d = i;
  float f = 2.5f;
  double fd = (double)f;
  int truncated = (int)fd;
  bool flag = i != 0;
  int from_bool = flag ? 1 : 0;

  int div = i / 2;        // -3 (sdiv)
  unsigned udiv = u / 2u; // 2000000000 (udiv)
  int shift = -8 >> 1;    // -4 (ashr)
  unsigned ushift = u << 1;

  int arr[3] = {10, 20, 30};
  int *ptr = arr + 1;
  int diff = (int)(ptr - arr);
  long as_int = (long)ptr;

  printf("%d %lld %d %d %d %f %d %d %u %d %u %d %d %d\n", i, big, s, ch,
         from_bool, d, truncated, div, udiv, shift, ushift, diff, ptr[0],
         as_int != 0);

  int a = 5, b = 0;
  if (a > 3 && b == 0) a += 1;
  if (a > 100 || b) a += 100;
  int ternary = a > 5 ? 10 : 20;
  printf("%d %d\n", a, ternary);
  return (div == -3 && shift == -4 && diff == 1 && truncated == 2 &&
          ternary == 10 && a == 6)
             ? 0
             : 1;
}
