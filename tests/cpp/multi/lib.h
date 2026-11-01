#ifndef MULTI_LIB_H
#define MULTI_LIB_H

struct Widget {
  int id;
  int id_times(int k) const;
};

extern int g_shared;
inline int helper(int x) { return x + g_shared; }

int widget_sum(const Widget *w, int n);

#endif
