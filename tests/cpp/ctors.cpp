// Special member functions: default ctor, parameterized ctor, copy ctor,
// destructor (RAII), and member initializer lists.
#include <cstdio>

int live_objects = 0;
int total_destroyed = 0;

struct Resource {
  int id;
  int *data;

  Resource() : id(0), data(nullptr) { live_objects++; }

  Resource(int i, int n) : id(i), data(new int[n]) {
    for (int k = 0; k < n; k++) data[k] = k + i;
    live_objects++;
  }

  Resource(const Resource &o) : id(o.id + 1000), data(nullptr) {
    live_objects++;
  }

  ~Resource() {
    delete[] data;
    live_objects--;
    total_destroyed++;
  }

  int first() const { return data ? data[0] : -1; }
};

int main() {
  {
    Resource a(5, 3);
    Resource b = a;  // copy ctor
    Resource c;      // default ctor
    printf("%d %d %d %d\n", a.first(), b.first(), c.first(), live_objects);
  }
  printf("%d %d\n", live_objects, total_destroyed);
  return (live_objects == 0) ? 0 : 1;
}
