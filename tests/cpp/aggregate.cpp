// Aggregate initialization without initializer_list: braced structs,
// nested aggregates, arrays of structs, and remaining-member zeroing.
#include <cstdio>

struct Point {
  int x, y;
};

struct Rect {
  Point lo, hi;
  int tag;
};

struct Data {
  int a[3];
  Point p;
};

int main() {
  Point p = {1, 2};
  Rect r = {{3, 4}, {5, 6}, 7};
  Rect zeroed = {8, 9};
  Data d = {{10, 20, 30}, {40, 50}};
  Point pts[3] = {{1, 1}, {2, 2}, {3, 3}};
  int arr[5] = {1, 2};

  int s = 0;
  for (int i = 0; i < 3; i++) s += pts[i].x * pts[i].y;

  printf("%d %d %d %d %d %d %d %d %d %d\n", p.x, p.y, r.lo.x, r.hi.y, r.tag,
         zeroed.lo.x, zeroed.hi.x, zeroed.tag, d.a[1], d.p.y);
  printf("%d %d %d\n", s, zeroed.lo.y, arr[3]);
  return (zeroed.hi.x == 0 && zeroed.tag == 0 && arr[3] == 0 && s == 14 &&
          d.p.x == 40)
             ? 0
             : 1;
}
