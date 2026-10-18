/* io.c — the leaf futures: plain functions that return a `_Future(T)`
   handle whose frame is allocated here and whose poll tries the
   syscall, registering with the reactor and returning PENDING on
   EAGAIN. The result sits at frame offset 0, as the ABI requires. */
#define _GNU_SOURCE

#include "exec.h"

#include <errno.h>
#include <stdlib.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

static glcc_future make_future(void *frame, void *poll) {
    glcc_future handle;
    handle.frame = frame;
    handle.poll = poll;
    return handle;
}

#define GLCC_LEAF(T, frame, poll)                                        \
    (__extension__({                                                     \
        glcc_future glcc_raw_ = make_future((frame), (void *)(poll));    \
        _Future(T) glcc_handle_;                                         \
        memcpy(&glcc_handle_, &glcc_raw_, sizeof glcc_raw_);             \
        glcc_handle_;                                                    \
    }))

/* ---- read ---- */

struct read_state {
    ssize_t result;
    unsigned state;
    int fd;
    void *buf;
    size_t count;
};

static unsigned read_poll(void *frame, void *ctx) {
    struct read_state *st = frame;
    ssize_t n = read(st->fd, st->buf, st->count);
    if (n >= 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
        if (st->state) glcc_reactor_del(st->fd);
        st->result = n;
        return GLCC_READY;
    }
    if (glcc_reactor_wait(st->fd, EPOLLIN, ctx) != 0) {
        st->result = -1;
        return GLCC_READY;
    }
    st->state = 1;
    return GLCC_PENDING;
}

_Future(ssize_t) glcc_async_read(int fd, void *buf, size_t count) {
    struct read_state *st = __glcc_frame_alloc(sizeof *st, _Alignof(struct read_state));
    if (!st) abort();
    st->result = -1;
    st->state = 0;
    st->fd = fd;
    st->buf = buf;
    st->count = count;
    return GLCC_LEAF(ssize_t, st, read_poll);
}

/* ---- write ---- */

struct write_state {
    ssize_t result;
    unsigned state;
    int fd;
    const void *buf;
    size_t count;
    size_t written;
    int all;
};

static unsigned write_poll(void *frame, void *ctx) {
    struct write_state *st = frame;
    ssize_t n = write(st->fd, (const char *)st->buf + st->written, st->count - st->written);
    if (n >= 0) {
        st->written += (size_t)n;
        if (!st->all || st->written == st->count) {
            if (st->state) glcc_reactor_del(st->fd);
            st->result = st->all ? (ssize_t)st->written : n;
            return GLCC_READY;
        }
    } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
        if (st->state) glcc_reactor_del(st->fd);
        st->result = -1;
        return GLCC_READY;
    }
    if (glcc_reactor_wait(st->fd, EPOLLOUT, ctx) != 0) {
        st->result = -1;
        return GLCC_READY;
    }
    st->state = 1;
    return GLCC_PENDING;
}

static _Future(ssize_t) make_write(int fd, const void *buf, size_t count, int all) {
    struct write_state *st = __glcc_frame_alloc(sizeof *st, _Alignof(struct write_state));
    if (!st) abort();
    st->result = -1;
    st->state = 0;
    st->fd = fd;
    st->buf = buf;
    st->count = count;
    st->written = 0;
    st->all = all;
    return GLCC_LEAF(ssize_t, st, write_poll);
}

_Future(ssize_t) glcc_async_write(int fd, const void *buf, size_t count) {
    return make_write(fd, buf, count, 0);
}

_Future(ssize_t) glcc_async_write_all(int fd, const void *buf, size_t count) {
    return make_write(fd, buf, count, 1);
}

/* ---- accept ---- */

struct accept_state {
    int result;
    unsigned state;
    int fd;
    int flags;
};

/* glibc's accept4 takes a `__SOCKADDR_ARG` transparent union, which
   glcc does not model yet; build the union explicitly. */
static int sys_accept(int fd, int flags) {
    __SOCKADDR_ARG addr;
    memset(&addr, 0, sizeof addr);
    return accept4(fd, addr, 0, flags);
}

static unsigned accept_poll(void *frame, void *ctx) {
    struct accept_state *st = frame;
    int fd = sys_accept(st->fd, st->flags);
    if (fd >= 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
        if (st->state) glcc_reactor_del(st->fd);
        st->result = fd;
        return GLCC_READY;
    }
    if (glcc_reactor_wait(st->fd, EPOLLIN, ctx) != 0) {
        st->result = -1;
        return GLCC_READY;
    }
    st->state = 1;
    return GLCC_PENDING;
}

_Future(int) glcc_async_accept(int fd, int flags) {
    struct accept_state *st = __glcc_frame_alloc(sizeof *st, _Alignof(struct accept_state));
    if (!st) abort();
    st->result = -1;
    st->state = 0;
    st->fd = fd;
    st->flags = flags;
    return GLCC_LEAF(int, st, accept_poll);
}

/* ---- sleep ---- */

struct sleep_state {
    unsigned state;
    unsigned long long delay_ns;
    glcc_timer timer;
};

static unsigned sleep_poll(void *frame, void *ctx) {
    struct sleep_state *st = frame;
    if (st->state == 0) {
        st->state = 1;
        glcc_timer_start(&st->timer, ctx, st->delay_ns);
        return GLCC_PENDING;
    }
    return glcc_timer_fired(&st->timer) ? GLCC_READY : GLCC_PENDING;
}

_Future(void) glcc_async_sleep(unsigned long long delay_ns) {
    struct sleep_state *st = __glcc_frame_alloc(sizeof *st, _Alignof(struct sleep_state));
    if (!st) abort();
    st->state = 0;
    st->delay_ns = delay_ns;
    return GLCC_LEAF(void, st, sleep_poll);
}

/* ---- yield ---- */

static unsigned yield_poll(void *frame, void *ctx) {
    unsigned *state = frame;
    if (*state == 0) {
        *state = 1;
        glcc_exec_wake(ctx);
        return GLCC_PENDING;
    }
    return GLCC_READY;
}

_Future(void) glcc_async_yield(void) {
    unsigned *state = __glcc_frame_alloc(sizeof *state, _Alignof(unsigned));
    if (!state) abort();
    *state = 0;
    return GLCC_LEAF(void, state, yield_poll);
}

/* ---- blocking offload ---- */

struct blocking_state {
    unsigned state;
    void (*fn)(void *);
    void *arg;
    glcc_pool_job job;
};

static unsigned blocking_poll(void *frame, void *ctx) {
    struct blocking_state *st = frame;
    if (st->state == 0) {
        st->state = 1;
        st->job.fn = st->fn;
        st->job.arg = st->arg;
        st->job.ctx = ctx;
        glcc_pool_submit(&st->job);
        return GLCC_PENDING;
    }
    return st->job.done ? GLCC_READY : GLCC_PENDING;
}

_Future(void) glcc_async_blocking_void(void (*fn)(void *), void *arg) {
    struct blocking_state *st = __glcc_frame_alloc(sizeof *st, _Alignof(struct blocking_state));
    if (!st) abort();
    st->state = 0;
    st->fn = fn;
    st->arg = arg;
    return GLCC_LEAF(void, st, blocking_poll);
}
