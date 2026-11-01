// String literals: length via a hand-rolled strlen (std::strlen-style
// loop), pointer arithmetic over a literal, escapes, literal
// concatenation, and indexing.
#include <cstdio>

size_t my_strlen(const char *s) {
  const char *p = s;
  while (*p) p++;
  return (size_t)(p - s);
}

int count_char(const char *s, char c) {
  int n = 0;
  for (; *s; s++)
    if (*s == c) n++;
  return n;
}

int main() {
  const char *lit = "hello";
  size_t n = my_strlen(lit);
  const char concat[] = "foo" "bar";
  char back = lit[4];
  char escaped[] = "a\tb\nc\\d\"e";

  printf("%zu %zu %s %c %s %d %d %d %d\n", n, my_strlen(concat), concat, back,
         lit + 1, lit[0], (int)escaped[1], (int)escaped[3], count_char("aXbXc", 'X'));
  return (n == 5 && my_strlen(concat) == 6 && back == 'o' && lit[0] == 'h' &&
          escaped[1] == '\t' && escaped[3] == '\n' && count_char("aXbXc", 'X') == 2)
             ? 0
             : 1;
}
