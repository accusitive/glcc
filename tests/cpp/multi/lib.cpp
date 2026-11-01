#include "lib.h"

int g_shared = 7;

int Widget::id_times(int k) const { return id * k; }

int widget_sum(const Widget *w, int n) {
  int s = 0;
  for (int i = 0; i < n; i++) s += w[i].id;
  return s;
}
