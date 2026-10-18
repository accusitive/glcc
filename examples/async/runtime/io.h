/* io.h — leaf futures provided by io.c (glcc-only: they return
   `_Future(T)` handles). */
#ifndef GLCC_IO_H
#define GLCC_IO_H

#include <stddef.h>
#include <sys/types.h>

#include "exec.h"

_Future(ssize_t) glcc_async_read(int fd, void *buf, size_t count);
_Future(ssize_t) glcc_async_write(int fd, const void *buf, size_t count);
_Future(ssize_t) glcc_async_write_all(int fd, const void *buf, size_t count);
_Future(int) glcc_async_accept(int fd, int flags);
_Future(void) glcc_async_sleep(unsigned long long delay_ns);
_Future(void) glcc_async_yield(void);
_Future(void) glcc_async_blocking_void(void (*fn)(void *), void *arg);

#endif /* GLCC_IO_H */
