// Dynamic global initialization end-to-end: the __sti__ routines the
// emitter produces must run before main via .init_array. Exercises
// std::string construction at load time plus a user ctor with logic.
#include <cstdio>
#include <string>

std::string g_name = std::string("dyn-") + std::to_string(42);

struct G {
  int v;
  G(int x) : v(x * 2) {}
};
G g_obj(21);

int main() {
  printf("%s %d\n", g_name.c_str(), g_obj.v);
  return 0;
}
