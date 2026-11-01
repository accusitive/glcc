// Variadic templates: recursive pack expansion and sizeof...(Ts).
#include <cstdio>

int sum() { return 0; }

template <class T, class... Rest>
int sum(T head, Rest... rest) {
  return (int)head + sum(rest...);
}

template <class... Ts>
int count() { return (int)sizeof...(Ts); }

template <class... Ts>
int all_doubled(Ts... args) {
  int total = 0;
  int dummy[] = {(total += 2 * (int)args, 0)...};
  (void)dummy;
  return total;
}

int main() {
  int a = sum(1, 2, 3);
  int b = sum(10, 20);
  int c = sum();
  int d = count<int, double, char>();
  int e = all_doubled(1, 2, 3, 4);

  printf("%d %d %d %d %d\n", a, b, c, d, e);
  return (a == 6 && b == 30 && c == 0 && d == 3 && e == 20) ? 0 : 1;
}
