// Unscoped and scoped enums: implicit conversion of unscoped enumerators,
// explicit conversion for scoped ones, and enumerator values with custom
// initializers.
#include <cstdio>

enum Color { RED, GREEN = 5, BLUE };
enum class Fruit : unsigned char { Apple = 1, Banana = 200 };

int main() {
  int a = RED;
  int b = GREEN;
  int c = BLUE;
  Fruit f = Fruit::Banana;
  Color x = BLUE;
  int xi = (int)x;
  unsigned char fb = (unsigned char)f;
  printf("%d %d %d %d %d\n", a, b, c, (int)fb, xi);
  return (a == 0 && b == 5 && c == 6 && fb == 200 && xi == 6) ? 0 : 1;
}
