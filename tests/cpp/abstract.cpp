// Abstract base class with pure virtual functions, override keyword,
// virtual dispatch through base pointers, and a virtual destructor.
#include <cstdio>

struct Shape {
  virtual int area() const = 0;
  virtual const char *name() const = 0;
  virtual ~Shape() {}
};

struct Square : Shape {
  int side;
  Square(int s) : side(s) {}
  int area() const override { return side * side; }
  const char *name() const override { return "square"; }
};

struct Rect : Shape {
  int w, h;
  Rect(int w, int h) : w(w), h(h) {}
  int area() const override { return w * h; }
  const char *name() const override { return "rect"; }
};

int main() {
  Square sq(4);
  Rect rc(3, 5);
  Shape *shapes[2] = {&sq, &rc};

  int total = 0;
  for (int i = 0; i < 2; i++) {
    total += shapes[i]->area();
    printf("%s:%d ", shapes[i]->name(), shapes[i]->area());
  }
  printf("total=%d\n", total);
  return (total == 16 + 15 && shapes[0]->area() == 16) ? 0 : 1;
}
