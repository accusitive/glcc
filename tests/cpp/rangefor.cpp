// Range-for over arrays: flat, nested, and with an index counter on the
// side; also over a const array.
#include <cstdio>

int main() {
  int a[5] = {3, 1, 4, 1, 5};
  int total = 0;
  int idx = 0;
  for (int x : a) {
    total += x * idx;
    idx++;
  }

  int grid[2][3] = {{1, 2, 3}, {4, 5, 6}};
  int rowsum = 0;
  for (auto &row : grid) {
    for (int v : row) rowsum += v;
  }

  const char msg[4] = {'a', 'b', 'c', 0};
  int letters = 0;
  for (char c : msg) letters += (c == 0) ? 0 : 1;

  printf("%d %d %d %c%c%c\n", total, rowsum, letters, msg[0], msg[1], msg[2]);
  return (total == 0 * 3 + 1 * 1 + 2 * 4 + 3 * 1 + 4 * 5 && rowsum == 21 &&
          letters == 3)
             ? 0
             : 1;
}
