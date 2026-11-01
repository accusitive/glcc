#include <cstdio>
#include "lib.h"

int main() {
  Widget ws[2] = {{10}, {11}};
  int a = ws[0].id_times(2) + ws[1].id_times(2);
  int b = widget_sum(ws, 2);
  int c = helper(35);
  printf("%d %d %d\n", a, b, c);
  return (a == 42 && b == 21 && c == 42) ? 0 : 1;
}
