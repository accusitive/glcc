// Static class members: out-of-line definition, in-class const int
// initializer, static member function, access through instance and scope.
#include <cstdio>

struct Counter {
  static int live;
  static const int limit = 4;
  static int bump() { return ++live; }
  int id;
};

int Counter::live = 0;

int main() {
  Counter::bump();
  Counter a;
  a.id = Counter::bump();
  Counter b;
  b.id = Counter::bump();

  printf("%d %d %d %d %d\n", Counter::live, a.id, b.id, Counter::limit,
         sizeof(Counter));
  return (Counter::live == 3 && a.id == 2 && b.id == 3 && Counter::limit == 4 &&
          sizeof(Counter) == sizeof(int))
             ? 0
             : 1;
}
