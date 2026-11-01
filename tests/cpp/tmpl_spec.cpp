// Template specialization: full function-template specialization, full
// and partial class-template specialization, and member access on each.
#include <cstdio>

template <class T>
const char *kind(T) { return "generic"; }

template <>
const char *kind(int) { return "int-spec"; }

template <class T>
struct Box {
  T v;
  int size() { return (int)sizeof(T); }
};

template <>
struct Box<bool> {
  int bit;
  int size() { return 1; }
};

template <class T>
struct Box<T *> {
  T *p;
  int size() { return 8; }
};

int main() {
  Box<int> bi;
  bi.v = 42;
  Box<bool> bb;
  bb.bit = 1;
  int x = 7;
  Box<int *> bp;
  bp.p = &x;

  printf("%s %s %d %d %d %d\n", kind(1.5), kind(3), bi.v, bi.size(), bb.size(),
         bp.size());
  return (bi.v == 42 && bi.size() == 4 && bb.size() == 1 && bp.size() == 8 &&
          *bp.p == 7)
             ? 0
             : 1;
}
