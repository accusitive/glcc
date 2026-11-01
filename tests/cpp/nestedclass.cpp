// Nested classes: inner struct inside a class, member functions of the
// inner class, an enum nested in a class, and static data on the outer.
#include <cstdio>

class Outer {
 public:
  enum State { IDLE, BUSY = 3 };
  struct Inner {
    int v;
    int doubled() const { return v * 2; }
  };
  Inner make(int v) {
    Inner i;
    i.v = v;
    return i;
  }
};

int main() {
  Outer o;
  Outer::Inner i = o.make(21);
  Outer::Inner j = {5};
  Outer::State s = Outer::BUSY;

  printf("%d %d %d %d\n", i.v, i.doubled(), j.doubled(), (int)s);
  return (i.v == 21 && i.doubled() == 42 && j.doubled() == 10 && s == 3 &&
          (int)Outer::IDLE == 0)
             ? 0
             : 1;
}
