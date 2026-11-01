// References: binding to locals, pass-by-reference, reference return
// values, const references, and references into arrays.
#include <cstdio>

void swap(int &a, int &b) {
  int t = a;
  a = b;
  b = t;
}

int &pick(int *a, int i) { return a[i]; }

int main() {
  int x = 1, y = 2;
  swap(x, y);

  int arr[3] = {10, 20, 30};
  pick(arr, 1) = 99;

  const int &cref = x;
  int sum = 0;
  for (int i = 0; i < 3; i++) sum += arr[i];
  int v = cref;

  printf("%d %d %d %d\n", x, y, sum, v);
  return (x == 2 && y == 1 && arr[1] == 99 && sum == 10 + 99 + 30 && v == 2)
             ? 0
             : 1;
}
