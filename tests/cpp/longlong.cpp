// long long / unsigned long long arithmetic: 64-bit multiply, division
// and remainder, shifts, hex constants, overflow wraparound.
#include <cstdio>

int main() {
  long long a = 123456789LL;
  long long b = a * a;               // 15241578750190521
  unsigned long long u = 18446744073709551615ULL;
  unsigned long long w = u + 1ULL;   // wraps to 0
  long long big = 1LL << 62;
  long long q = b / 1000000LL;
  long long r = b % 1000000LL;
  long long h = 0x7fffffffffffffffLL;
  long long neg = -b;

  printf("%lld %llu %lld %lld %lld %lld\n", b, w, big, q, r, h);
  printf("%lld %lld\n", neg / 1000LL, neg % 1000LL);
  return (b == 15241578750190521LL && w == 0 && big == 4611686018427387904LL &&
          h == 9223372036854775807LL && neg < 0)
             ? 0
             : 1;
}
