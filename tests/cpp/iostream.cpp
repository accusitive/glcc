// iostream: std::cout with strings, ints, doubles, hex, and std::endl.
#include <iostream>

int main() {
  int v = 255;
  double d = 2.5;
  std::cout << "start " << v << ' ' << d << std::endl;
  std::cout << std::hex << v << std::endl;
  std::cout << "sum=" << v + 1 << "\n";
  return (v == 255 && d == 2.5) ? 0 : 1;
}
