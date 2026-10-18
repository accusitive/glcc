/* glcc async runtime: the polling driver over the future ABI the
   compiler emits for `__attribute__((async))` functions. See
   docs/async-abi.md for the ABI contract. */
#ifndef GLCC_ASYNC_H
#define GLCC_ASYNC_H

#ifdef __cplusplus
extern "C" {
#endif

/* Poll results: a coroutine reports GLCC_PENDING while the future it is
   awaiting has not finished, and GLCC_READY once its result (if any)
   sits at offset 0 of its frame. */
#define GLCC_PENDING 0u
#define GLCC_READY 1u

/* Scheduler context threaded through every poll. v1 drivers busy-poll,
   so `woken` only records wake-ups a waker already delivered; a
   scheduler with idle states can park on it instead of spinning. */
typedef struct glcc_ctx {
    int woken;
} glcc_ctx;

/* Type-erased view of a `_Future(T)` handle: field 0 is the boxed
   frame, field 1 the resume entry point (a `unsigned (*)(void *frame,
   glcc_ctx *ctx)` stored as a raw pointer). */
typedef struct glcc_future {
    void *frame;
    void *poll;
} glcc_future;

/* Frame hooks: the compiler emits calls to these at every coroutine
   start and completion. Both are weak here, so an embedder can link
   its own pool allocator instead. */
void *__glcc_frame_alloc(unsigned long size, unsigned long align);
void __glcc_frame_free(void *frame);

/* Record a wake-up from a waker. */
static inline void glcc_wake(glcc_ctx *ctx) { ctx->woken = 1; }

/* Poll `future` until it completes, copying `size` bytes of its result
   out through `out` (null with size 0 for void results), then free its
   frame. */
void glcc_block_on_impl(void *handle, void *out, unsigned long size);

/* Drive a `_Future(T)` to completion: `glcc_block_on(f, &result)`. */
#define glcc_block_on(future, out) \
    glcc_block_on_impl(&(future), (out), sizeof(*(out)))

/* The same for `_Future(void)`. */
#define glcc_block_on_void(future) glcc_block_on_impl(&(future), 0, 0)

#ifdef __cplusplus
}
#endif

#endif /* GLCC_ASYNC_H */
