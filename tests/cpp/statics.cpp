// Focused test: globals (zero, partial array, struct), a static local
// with dynamic init (guard path), and indirect struct flow.

#include <cstdio>

struct Pair {
  int a;
  int b;
};

Pair g_pair = {11, 22};
int g_zeroed;
long g_partial[4] = {7, 8};

int counter_calls = 0;

int next_counter() {
  counter_calls++;
  return counter_calls * 10;
}

int statics() {
  static int constant_one = 5;
  static int dynamic_one = next_counter();
  constant_one++;
  return constant_one + dynamic_one;
}

Pair swap(Pair in) {
  Pair out;
  out.a = in.b;
  out.b = in.a;
  return out;
}

int main() {
  Pair flipped = swap(g_pair);
  statics();
  int second = statics();
  printf("%d %d %d %d %d %d\n", flipped.a, flipped.b, g_zeroed, g_partial[0],
         g_partial[3], second);
  return (flipped.a == 22 && flipped.b == 11 && second == 17) ? 0 : 1;
}
