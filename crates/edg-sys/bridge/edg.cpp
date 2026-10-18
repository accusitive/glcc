// edg.cpp -- small C ABI bridge that lets Rust drive EDG in-process.
//
// EDG runs its front end (`edg::edg_main`, macro'd as `EDG_MAIN`) compiled
// with MAKE_FRONT_END_CALLABLE=1; with the C-gen back end enabled it
// translates C++ to the generated C consumed by glcc, writes `<source>.ti`
// instantiation data next to $PWD, and returns a process-style exit code
// instead of calling exit().

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

extern "C++" int edg_main(int argc, char *argv[]);

namespace {
// Run f while fd 1 points at a tmpfile; then copy the buffered bytes out.
template <typename F> int run_capture_stdout(F &&f, char **out_buf, unsigned long long *out_len) {
  *out_buf = nullptr;
  *out_len = 0;
  fflush(stdout);
  fflush(stderr);
  int saved = dup(1);
  FILE *tmp = tmpfile();
  if (tmp != nullptr && saved >= 0) {
    dup2(fileno(tmp), 1);
  }
  int rc = f();
  fflush(stdout);
  if (tmp != nullptr && saved >= 0) {
    dup2(saved, 1);
    close(saved);
    fseek(tmp, 0, SEEK_END);
    long size = ftell(tmp);
    if (size < 0) size = 0;
    fseek(tmp, 0, SEEK_SET);
    char *buf = nullptr;
    if (size > 0) {
      buf = (char *)malloc((size_t)size);
      if (buf != nullptr && fread(buf, 1, (size_t)size, tmp) < 0) {
        free(buf);
        buf = nullptr;
      }
    }
    *out_buf = buf;
    *out_len = (unsigned long long)size;
    fclose(tmp);
  } else {
    if (tmp) fclose(tmp);
    if (saved >= 0) close(saved);
  }
  return rc;
}
} // namespace

extern "C" int glcpp_edg_frontend(int argc, char **argv) {
  return edg_main(argc, argv);
}

extern "C" int glcpp_edg_frontend_capture(int argc, char **argv, char **out_buf, unsigned long long *out_len) {
  return run_capture_stdout([&]() { return edg_main(argc, argv); }, out_buf, out_len);
}

extern "C" void glcpp_edg_free(void *p) {
  if (p) free(p);
}

