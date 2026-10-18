/* glcc executor — a multi-threaded Linux prototype runtime for glcc
   futures, in the shape of a small tokio: worker threads poll tasks,
   an epoll reactor wakes them for IO, a timerfd drives sleeps, and a
   thread pool absorbs blocking calls. See README.md for the tour and
   docs/async-abi.md for the future ABI it is built on.

   Everything here uses only the ABI contract, not the compiler: the
   handle, the poll signature, and the frame hooks. */
#ifndef GLCC_EXEC_H
#define GLCC_EXEC_H

#include <stddef.h>
#include <string.h>

#include "../../../rt/glcc_async.h"

#ifdef GLCC_STANDALONE
/* So clang can compile the runtime for cross-checking: `_Future(T)`
   collapses to the raw handle. Coroutine syntax still needs glcc. */
#define _Future(T) glcc_future
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* A task result is copied through this many bytes. */
#define GLCC_TASK_RESULT_MAX 128

typedef struct glcc_task glcc_task;
typedef struct glcc_timer glcc_timer;
typedef struct glcc_pool_job glcc_pool_job;

/* ---- executor lifecycle: init is implicit, shutdown is terminal ---- */
void glcc_exec_init(void);
void glcc_exec_shutdown(void);

/* ---- tasks ---- */

/* Take over an already-started future, box it as a task, and schedule
   it on the executor. */
glcc_task *glcc_adopt(const void *handle);

/* Spawn an async call as an independent task: `glcc_spawn(f(x))`.
   Returns the task, which can be joined or awaited later. */
#define glcc_spawn(call)                                                 \
    (__extension__({                                                     \
        __typeof__(call) glcc_handle_ = (call);                          \
        (void)sizeof(struct {                                            \
            int glcc_spawn_wants_an_async_call : (sizeof glcc_handle_ == sizeof(glcc_future)) ? 1 : -1; \
        });                                                              \
        glcc_adopt(&glcc_handle_);                                       \
    }))

/* Block the calling (non-worker) thread until `task` completes, copy
   the result out, and free the task. `glcc_join(t, &r)`. */
void glcc_join_impl(glcc_task *task, void *out, unsigned long size);

#define glcc_join(task, out)                                             \
    (__extension__({                                                     \
        (void)sizeof(struct {                                            \
            int glcc_join_result_too_large : (sizeof *(out) <= GLCC_TASK_RESULT_MAX) ? 1 : -1; \
        });                                                              \
        glcc_join_impl((task), (out), sizeof *(out));                    \
    }))
#define glcc_join_void(task) glcc_join_impl((task), 0, 0)

/* Await a task from inside a coroutine: `__builtin_await(
   glcc_task_future(t, int))` suspends until `t` completes and yields
   its result. Single-awaiter; the task is freed with the handle. */
glcc_future glcc_join_future_(glcc_task *task, unsigned long size);

#define glcc_task_future(task, T)                                        \
    (__extension__({                                                     \
        (void)sizeof(struct {                                            \
            int glcc_task_result_too_large : (sizeof(T) <= GLCC_TASK_RESULT_MAX) ? 1 : -1; \
        });                                                              \
        glcc_future glcc_raw_ = glcc_join_future_((task), sizeof(T));    \
        _Future(T) glcc_handle_;                                         \
        memcpy(&glcc_handle_, &glcc_raw_, sizeof glcc_raw_);             \
        glcc_handle_;                                                    \
    }))

#define glcc_task_future_void(task)                                      \
    (__extension__({                                                     \
        glcc_future glcc_raw_ = glcc_join_future_((task), 0);            \
        _Future(void) glcc_handle_;                                      \
        memcpy(&glcc_handle_, &glcc_raw_, sizeof glcc_raw_);             \
        glcc_handle_;                                                    \
    }))

/* ---- leaf-future support ---- */

/* Wake whatever `ctx` names: a task's waker, or (for code shared with
   the busy-poll runtime) a plain glcc_ctx. */
void glcc_exec_wake(void *ctx);

/* Arm a one-shot readiness watch on `fd`; ADD or MOD as needed, so it
   is idempotent and safe to call on every retry. */
int glcc_reactor_wait(int fd, unsigned events, void *ctx);
void glcc_reactor_del(int fd);

unsigned long long glcc_now_ns(void);
void glcc_timer_start(glcc_timer *timer, void *ctx, unsigned long long delay_ns);
int glcc_timer_fired(const glcc_timer *timer);

void glcc_pool_submit(glcc_pool_job *job);

/* ---- structs embedded in leaf frames ---- */

struct glcc_timer {
    unsigned long long deadline; /* CLOCK_MONOTONIC ns */
    unsigned long long seq;
    void *ctx;
    volatile int fired;         /* set by the reactor before the wake */
    int heap_index;
    struct glcc_timer *fired_next;
};

struct glcc_pool_job {
    void (*fn)(void *);
    void *arg;
    void *ctx;
    volatile int done;          /* set by the pool thread before the wake */
    struct glcc_pool_job *next;
};

#ifdef __cplusplus
}
#endif

#endif /* GLCC_EXEC_H */
