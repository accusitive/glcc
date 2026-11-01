// Cast matrix: static_cast (numeric conversions, enum, upcast,
// downcast across a static hierarchy), const_cast, and
// reinterpret_cast through integer round trips.
#include <cstdio>

struct Base { int b; };
struct Mid : Base { int m; };
struct Leaf : Mid { int l; };

int main() {
  int i = static_cast<int>(3.9);
  double d = static_cast<double>(7) / 2;
  enum E { X = 5 };
  int e = static_cast<int>(X) + static_cast<int>('A');

  Leaf leaf;
  leaf.b = 1;
  leaf.m = 2;
  leaf.l = 3;
  Base *bp = static_cast<Base *>(&leaf);
  Leaf *back = static_cast<Leaf *>(bp);

  int mutable_src = 10;
  const int *cp = &mutable_src;
  int *mp = const_cast<int *>(cp);
  *mp += 5;

  long addr = reinterpret_cast<long>(&leaf);
  Leaf *again = reinterpret_cast<Leaf *>(addr);
  int *intview = reinterpret_cast<int *>(&leaf);  // first int member

  printf("%d %.1f %d %d %d %d %d %d\n", i, d, e, bp->b, back->l, *mp,
         (int)(again == &leaf), *intview);
  return (i == 3 && d == 3.5 && e == 70 && bp->b == 1 && back->l == 3 &&
          *mp == 15 && again == &leaf && *intview == 1)
             ? 0
             : 1;
}
