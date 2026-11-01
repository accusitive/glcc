// Signed/unsigned mixed comparisons: usual arithmetic conversions make
// -1 compare large against unsigned, unsigned wraparound, and
// unsigned char promotion differences.
#include <cstdio>

int main() {
  int i = -1;
  unsigned u = 1;
  int lt = (i < u);        // false: i converts to UINT_MAX
  int gt = (i > u);        // true
  unsigned wrapped = 0u - 1u;
  unsigned char uc = 200;
  int promoted = uc + uc;  // promoted to int: 400
  unsigned back = (unsigned)(i + u);

  printf("%d %d %u %d %u\n", lt, gt, wrapped, promoted, back);
  return (lt == 0 && gt == 1 && wrapped == 4294967295u && promoted == 400 &&
          back == 0)
             ? 0
             : 1;
}
