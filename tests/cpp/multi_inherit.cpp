// Multiple inheritance: two non-virtual bases, base-pointer adjustment,
// sizeof the derived layout, and calling members from both bases.
#include <cstdio>

struct A {
  int a;
  int get_a() { return a * 10; }
};

struct B {
  int b;
  int get_b() { return b * 100; }
};

struct C : A, B {
  int c;
  int total() { return a + b + c; }
};

int main() {
  C obj;
  obj.a = 1;
  obj.b = 2;
  obj.c = 3;

  A *pa = &obj;
  B *pb = &obj;
  C *pc = static_cast<C *>(pb);  // pointer adjustment back to C

  printf("%d %d %d %d %d %d %d\n", obj.get_a(), obj.get_b(), obj.total(),
         pa->a, pb->b, pc->c, (int)sizeof(C));
  return (pa->a == 1 && pb->b == 2 && pc->c == 3 && obj.total() == 6 &&
          sizeof(C) == 3 * (int)sizeof(int))
             ? 0
             : 1;
}
