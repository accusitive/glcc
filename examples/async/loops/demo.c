/* Two infinite coroutine loops, each printing its own accumulator —
 * and this file creates no threads. `glcc_spawn` hands both loops to
 * the executor in ../runtime, whose worker pool polls them on separate
 * OS threads; `worker=` is the pthread that ran that iteration, so the
 * two timelines interleave with different ids.
 *
 * make && make run.
 */
#define async __attribute__((async))
#define await(x) __builtin_await(x)
#ifndef GLCC
#define __builtin_await(x)
#endif

#define _GNU_SOURCE

#include <pthread.h>
#include <stdio.h>
#include <time.h>

#include "exec.h"
#include "io.h"

#define NS_PER_MS 1000000ULL

async void loop(char id, unsigned long long period_ns) {
    unsigned long acc = 0;
    for (;;) {
        await(glcc_async_sleep(period_ns));
        acc++;
        printf("%c acc=%-5lu worker=%lu\n", id, acc, (unsigned long)pthread_self());
        fflush(stdout);
    }
}

int main(void) {
    printf("spawning the same coroutine twice, as two infinite loops\n");
    printf("(no pthread_create here: glcc_spawn puts them on the executor)\n\n");

    glcc_spawn(loop('A', 80 * NS_PER_MS));
    glcc_spawn(loop('B', 125 * NS_PER_MS));

    /* The loops never finish; watch a one-second window, then tear down. */
    struct timespec window;
    window.tv_sec = 1;
    window.tv_nsec = 0;
    nanosleep(&window, 0);

    printf("\nwindow over while both accumulators were still advancing\n");
    glcc_exec_shutdown();
    return 0;
}
