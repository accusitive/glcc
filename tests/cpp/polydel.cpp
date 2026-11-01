// Polymorphic delete and virtual dispatch through a base pointer:
// vtable slots, virtual destructor (D0/D1/D2 chain), operator new/delete,
// and a nontrivial derived member destroyed via the base interface.
#include <cstdio>

int dtor_log = 0;

struct Shape {
  virtual int area() { return 0; }
  virtual const char *name() { return "shape"; }
  virtual ~Shape() { dtor_log = dtor_log * 10 + 1; }
};

struct Square : Shape {
  int side;
  explicit Square(int s) : side(s) {}
  int area() override { return side * side; }
  const char *name() override { return "square"; }
  ~Square() override { dtor_log = dtor_log * 10 + 2; }
};

struct Box : Shape {
  int w, h;
  int *pad;
  Box(int a, int b) : w(a), h(b), pad(new int[3]) {
    pad[0] = w;
    pad[1] = h;
    pad[2] = 42;
  }
  int area() override { return w * h; }
  const char *name() override { return "box"; }
  ~Box() override {
    dtor_log = dtor_log * 10 + 3;
    delete[] pad;
  }
};

int main() {
  Shape *shapes[2];
  shapes[0] = new Square(5);
  shapes[1] = new Box(3, 4);

  printf("%s=%d %s=%d\n", shapes[0]->name(), shapes[0]->area(),
         shapes[1]->name(), shapes[1]->area());

  int pad_via_base = 0;
  if (shapes[1]->area() == 12) {
    Box *b = static_cast<Box *>(shapes[1]);
    pad_via_base = b->pad[2];
  }

  // Deleting through the base pointer must run the full derived dtor
  // chain and free the derived allocation.
  delete shapes[0];
  delete shapes[1];

  printf("pad=%d dtor_log=%d\n", pad_via_base, dtor_log);
  // Square: ~Square(2) then ~Shape(1) -> 21; Box: 213 then 1 -> 2131.
  return (pad_via_base == 42 && dtor_log == 2131) ? 0 : 1;
}
