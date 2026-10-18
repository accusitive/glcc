/* glcc async runtime: frame hooks and the v1 blocking driver. */
#include "glcc_async.h"

#include <stdlib.h>
#include <string.h>

__attribute__((weak)) void *__glcc_frame_alloc(unsigned long size, unsigned long align) {
    (void)align; /* malloc returns 16-byte-aligned memory; frames need no more */
    return malloc(size);
}

__attribute__((weak)) void __glcc_frame_free(void *frame) { free(frame); }

void glcc_block_on_impl(void *handle, void *out, unsigned long size) {
    glcc_future *future = (glcc_future *)handle;
    unsigned (*poll)(void *, void *) = (unsigned (*)(void *, void *))future->poll;
    glcc_ctx ctx;
    ctx.woken = 0;
    /* v1 has no idle state to park on: re-poll until the future reports
       ready. Steady-state embedders replace this with a scheduler that
       consumes `ctx.woken`. */
    while (poll(future->frame, &ctx) == GLCC_PENDING) {
        ctx.woken = 0;
    }
    if (out != 0 && size != 0) {
        memcpy(out, future->frame, size);
    }
    __glcc_frame_free(future->frame);
}
