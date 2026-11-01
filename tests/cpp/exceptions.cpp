// Exceptions: try/throw/catch with int and const char * payloads,
// catch-all, rethrow, and ordering of handlers. Documents the current
// EH gap in the pipeline.
#include <cstdio>

int last_seen = 0;

void might_throw(int mode) {
  if (mode == 1) throw 42;
  if (mode == 2) throw "string exception";
  last_seen = mode;
}

int main() {
  int results[3] = {0, 0, 0};

  try {
    might_throw(0);
    results[0] = 1;
  } catch (int e) {
    results[0] = 100 + e;
  }

  try {
    might_throw(1);
    results[1] = 2;
  } catch (int e) {
    results[1] = 200 + e;
  } catch (const char *s) {
    results[1] = 300;
  }

  try {
    try {
      might_throw(2);
    } catch (int) {
      throw;  // rethrow: not an int, so it escapes to the outer catch
    }
  } catch (const char *s) {
    results[2] = 400;
  } catch (...) {
    results[2] = 500;
  }

  printf("%d %d %d %d\n", results[0], results[1], results[2], last_seen);
  return (results[0] == 1 && results[1] == 242 && results[2] == 400 &&
          last_seen == 0)
             ? 0
             : 1;
}
