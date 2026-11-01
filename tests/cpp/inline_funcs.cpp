// inline functions: header-style definitions used from several call
// sites, an inline with a static local (one shared instance), and a
// template defined inline.
#include <cstdio>

inline int square(int x) { return x * x; }

inline int token() {
  static int n = 100;
  return n++;
}

template <class T>
inline T pick_max(T a, T b) { return a > b ? a : b; }

struct Pt {
  int x, y;
  inline int manhattan() const { return x > 0 ? x + y : -x + y; }
};

int main() {
  int a = square(7);
  int t1 = token();
  int t2 = token();
  Pt p;
  p.x = -3;
  p.y = 4;

  printf("%d %d %d %d %d %d\n", a, t1, t2, p.manhattan(), pick_max(2, 9),
         (int)pick_max(1.5, 0.5));
  return (a == 49 && t1 == 100 && t2 == 101 && p.manhattan() == 7) ? 0 : 1;
}
