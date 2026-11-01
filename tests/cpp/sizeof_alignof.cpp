// sizeof / alignof edge cases: struct padding, empty struct (size 1),
// arrays, alignment of long double, and offset ordering.
#include <cstdio>

struct Empty {};
struct Padded {
  char c;
  int i;
  char c2;
};
struct Tight {
  int i;
  char c;
};
struct Aligned {
  long double ld;
  char c;
};

int main() {
  int arr[10];
  size_t sz_padded = sizeof(Padded);
  size_t sz_empty = sizeof(Empty);
  size_t al_ld = alignof(long double);
  size_t al_max = alignof(Aligned);

  printf("%zu %zu %zu %zu %zu %zu %zu\n", sizeof(int), sizeof(long),
         sizeof(void *), sz_padded, sz_empty, al_ld, al_max);
  printf("%zu %zu %zu\n", sizeof(arr), sizeof(Tight), alignof(Padded));
  return (sz_padded == 12 && sz_empty == 1 && al_ld >= 8 && al_max == 16 &&
          sizeof(arr) == 40 * 1 && sizeof(Tight) == 8)
             ? 0
             : 1;
}
