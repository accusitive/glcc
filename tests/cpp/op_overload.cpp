// Operator overloading: member operator+= and free operator+ / operator==
// / unary operator- on a small value type.
#include <cstdio>

struct Vec2 {
  int x, y;
  Vec2 &operator+=(const Vec2 &o) {
    x += o.x;
    y += o.y;
    return *this;
  }
};

Vec2 operator+(const Vec2 &a, const Vec2 &b) {
  Vec2 r = {a.x + b.x, a.y + b.y};
  return r;
}

bool operator==(const Vec2 &a, const Vec2 &b) {
  return a.x == b.x && a.y == b.y;
}

Vec2 operator-(const Vec2 &a) {
  Vec2 r = {-a.x, -a.y};
  return r;
}

int main() {
  Vec2 a = {1, 2};
  Vec2 b = {10, 20};
  Vec2 c = a + b;
  Vec2 d = -c;
  a += b;
  printf("%d %d %d %d %d %d\n", c.x, c.y, d.x, d.y, a.x, a.y);
  return (c.x == 11 && c.y == 22 && d.x == -11 && d.y == -22 && a == c &&
          !(a == b))
             ? 0
             : 1;
}
