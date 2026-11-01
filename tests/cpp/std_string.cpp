// std::string basics: construction, append, size, subscript, c_str,
// comparison, find.
#include <cstdio>
#include <string>

int main() {
  std::string s = "hello";
  s += ", world";
  s.append("!");

  size_t n = s.size();
  char c0 = s[0];
  std::string copy = s;
  copy[0] = 'H';

  std::string tail = s.substr(7, 5);
  size_t pos = s.find("world");
  int cmp = (s == std::string("hello, world!"));

  printf("%zu %c %s %s %zu %d %zu\n", n, c0, copy.c_str(), tail.c_str(), pos,
         cmp, s.size());
  return (n == 13 && c0 == 'h' && copy[0] == 'H' && tail == std::string("world") &&
          pos == 7 && cmp == 1)
             ? 0
             : 1;
}
