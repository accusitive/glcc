// Global dynamic initialization and destruction: __sti__-style dynamic
// global init (ordered), a global with a nontrivial destructor (run via
// __cxa_atexit), and a function-local static with a dynamic initializer.
#include <cstdio>

int order = 0;
int init_seen[3] = {0, 0, 0};

int compute(int base) { return base * 10; }

int g_a = compute(1);
int g_b = compute(2) + g_a;

struct Tracked {
  int id;
  explicit Tracked(int i) : id(i) { init_seen[i] = ++order; }
  ~Tracked() { init_seen[id] = -init_seen[id]; }
};

Tracked g_t0(0);
Tracked g_t2(2);

int destroyed_late() {
  static Tracked local(1);
  return local.id;
}

int main() {
  // All dynamic initializers ran before main, in declaration order.
  if (g_a != 10 || g_b != 30) {
    printf("bad globals %d %d\n", g_a, g_b);
    return 1;
  }
  // g_t0's ctor ran before g_t2's.
  printf("init order: t0=%d t2=%d\n", init_seen[0], init_seen[2]);
  if (init_seen[0] != 1 || init_seen[2] != 2) return 1;
  // The function-local static initializes on first call.
  if (destroyed_late() != 1 || init_seen[1] != 3) return 1;
  printf("ok\n");
  return 0;
}
