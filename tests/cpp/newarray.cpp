// new[] / delete[]: heap arrays of ints and of a POD struct, round-trip
// through assignment, plus scalar new/delete.
#include <cstdio>

struct Item {
  int key;
  int val;
};

int main() {
  int *a = new int[8];
  for (int i = 0; i < 8; i++) a[i] = i * i;
  int sum = 0;
  for (int i = 0; i < 8; i++) sum += a[i];
  delete[] a;

  Item *items = new Item[3];
  for (int i = 0; i < 3; i++) {
    items[i].key = i;
    items[i].val = 10 - i;
  }
  int kv = 0;
  for (int i = 0; i < 3; i++) kv += items[i].key * items[i].val;
  delete[] items;

  int *one = new int(41);
  *one += 1;
  int v = *one;
  delete one;

  printf("%d %d %d\n", sum, kv, v);
  return (sum == 140 && kv == 0 * 10 + 1 * 9 + 2 * 8 && v == 42) ? 0 : 1;
}
