// 2-D array traversal: row pointers over int[2][3] must step by
// sizeof(int[3]) (12), so the loop end `m + 2` lands two rows in and a
// range-for over the rows visits every element exactly once.
#include <cstdio>

int main() {
  int m[2][3] = {{1, 2, 3}, {4, 5, 6}};

  int sum = 0;
  for (auto &row : m)
    for (int v : row) sum += v;

  int ptrsum = 0;
  for (int (*p)[3] = m; p != m + 2; ++p)
    for (int j = 0; j < 3; ++j) ptrsum += (*p)[j];

  int shifted = ((int (*)[3])&m[0][0])[1][2];

  printf("%d %d %d\n", sum, ptrsum, shifted);
  return (sum == 21 && ptrsum == 21 && shifted == 6) ? 0 : 1;
}
