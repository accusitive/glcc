// std::vector<int> basics: push_back growth, size, operator[],
// iteration, and element modification.
#include <cstdio>
#include <vector>

int main() {
  std::vector<int> v;
  for (int i = 1; i <= 6; i++) v.push_back(i * i);

  size_t n = v.size();
  int sum = 0;
  for (size_t i = 0; i < v.size(); i++) sum += v[i];

  int iter = 0;
  for (int x : v) iter += x;

  v[0] = 100;
  int first = v[0];
  int last = v.back();

  printf("%zu %d %d %d %d\n", n, sum, iter, first, last);
  return (n == 6 && sum == 1 + 4 + 9 + 16 + 25 + 36 && iter == sum &&
          first == 100 && last == 36)
             ? 0
             : 1;
}
