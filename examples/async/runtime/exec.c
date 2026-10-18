/* exec.c — the executor core: task table, run queue, workers, epoll
   reactor, timer heap, blocking pool, and the frame hooks the compiler
   calls into. See exec.h. */
#define _GNU_SOURCE

#include "exec.h"

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/timerfd.h>
#include <time.h>
#include <unistd.h>

#define GLCC_WAKER_MAGIC 0x676c63635f776b72ULL /* "glcc_wkr" */

/* data.u64 in epoll events tags what fired; wakers are at least
   8-byte aligned so the low two bits are free. */
#define GLCC_TAG_MASK 3ULL
#define GLCC_TAG_WAKER 0ULL
#define GLCC_TAG_TIMER 1ULL
#define GLCC_TAG_SHUTDOWN 2ULL

enum task_state {
    T_READY,    /* on the run queue, or about to be re-queued by its worker */
    T_RUNNING,  /* a worker is inside the task's poll */
    T_PARKED,   /* suspended on a future that promised a wake */
    T_DONE,     /* poll returned READY; the result frame awaits pick-up */
    T_ZOMBIE,   /* freed by its owner; kept so stale wakers stay valid */
};

struct glcc_waker {
    unsigned long long magic;
    glcc_task *task;
};

union glcc_task_result {
    long double align;
    unsigned char raw[GLCC_TASK_RESULT_MAX];
};

struct glcc_task {
    union glcc_task_result result; /* offset 0: the awaiter's ABI contract */
    glcc_future inner;
    void *result_frame;
    unsigned long result_size;
    void *joiner;
    volatile int state;
    struct glcc_waker waker;
    pthread_cond_t done_cv;
    struct glcc_task *next;     /* run queue link */
    struct glcc_task *all_next; /* live-then-zombie table link */
};

/* ---- globals ---- */

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_work = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_pool_work = PTHREAD_COND_INITIALIZER;

static int g_started;
static int g_shutdown;
static int g_epfd = -1;
static int g_tfd = -1;
static int g_efd = -1;

static glcc_task *g_run_head;
static glcc_task *g_run_tail;
static glcc_task *g_tasks;   /* live entries (singly linked) */
static glcc_task *g_zombies; /* freed entries (tombstones, freed at shutdown) */

static glcc_pool_job *g_pool_head;
static glcc_pool_job *g_pool_tail;

static glcc_timer **g_heap;
static size_t g_heap_len;
static size_t g_heap_cap;
static unsigned long long g_timer_seq;

static pthread_t g_worker_tids[16];
static int g_nworkers;
static pthread_t g_reactor_tid;
static pthread_t g_pool_tids[8];
static int g_npool;

static pthread_once_t g_once = PTHREAD_ONCE_INIT;

/* ---- run queue and task table (g_lock held) ---- */

static void push_run_locked(glcc_task *t) {
    t->next = 0;
    if (g_run_tail) {
        g_run_tail->next = t;
    } else {
        g_run_head = t;
    }
    g_run_tail = t;
    pthread_cond_signal(&g_work);
}

static glcc_task *find_task_locked(const void *frame) {
    for (glcc_task *t = g_tasks; t; t = t->all_next) {
        if (t == frame) return t;
    }
    for (glcc_task *t = g_zombies; t; t = t->all_next) {
        if (t == frame) return t;
    }
    return 0;
}

/* ---- timer heap ---- */

static int timer_before(const glcc_timer *a, const glcc_timer *b) {
    if (a->deadline != b->deadline) return a->deadline < b->deadline;
    return a->seq < b->seq;
}

static void heap_swap(size_t i, size_t j) {
    glcc_timer *tmp = g_heap[i];
    g_heap[i] = g_heap[j];
    g_heap[j] = tmp;
    g_heap[i]->heap_index = (int)i;
    g_heap[j]->heap_index = (int)j;
}

static void heap_sift_up(size_t i) {
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (!timer_before(g_heap[i], g_heap[parent])) break;
        heap_swap(i, parent);
        i = parent;
    }
}

static void heap_sift_down(size_t i) {
    for (;;) {
        size_t left = 2 * i + 1;
        size_t right = left + 1;
        size_t best = i;
        if (left < g_heap_len && timer_before(g_heap[left], g_heap[best])) best = left;
        if (right < g_heap_len && timer_before(g_heap[right], g_heap[best])) best = right;
        if (best == i) break;
        heap_swap(i, best);
        i = best;
    }
}

static void heap_push(glcc_timer *t) {
    if (g_heap_len == g_heap_cap) {
        size_t cap = g_heap_cap ? g_heap_cap * 2 : 64;
        glcc_timer **grown = realloc(g_heap, cap * sizeof *g_heap);
        if (!grown) abort();
        g_heap = grown;
        g_heap_cap = cap;
    }
    g_heap[g_heap_len] = t;
    t->heap_index = (int)g_heap_len;
    g_heap_len++;
    heap_sift_up(g_heap_len - 1);
}

static glcc_timer *heap_pop(void) {
    glcc_timer *top = g_heap[0];
    g_heap_len--;
    if (g_heap_len > 0) {
        g_heap[0] = g_heap[g_heap_len];
        g_heap[0]->heap_index = 0;
        heap_sift_down(0);
    }
    top->heap_index = -1;
    return top;
}

/* Point the timerfd at the earliest deadline (or disarm when idle). */
static void timer_rearm_locked(void) {
    struct itimerspec its;
    memset(&its, 0, sizeof its);
    if (g_heap_len > 0) {
        unsigned long long now = glcc_now_ns();
        unsigned long long delay = g_heap[0]->deadline > now ? g_heap[0]->deadline - now : 1;
        its.it_value.tv_sec = (time_t)(delay / 1000000000ULL);
        its.it_value.tv_nsec = (long)(delay % 1000000000ULL);
    }
    timerfd_settime(g_tfd, 0, &its, 0);
}

/* ---- public clock / timer / reactor API ---- */

unsigned long long glcc_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ULL + (unsigned long long)ts.tv_nsec;
}

void glcc_timer_start(glcc_timer *timer, void *ctx, unsigned long long delay_ns) {
    glcc_exec_init();
    pthread_mutex_lock(&g_lock);
    timer->deadline = glcc_now_ns() + delay_ns;
    timer->seq = ++g_timer_seq;
    timer->ctx = ctx;
    timer->fired = 0;
    timer->fired_next = 0;
    heap_push(timer);
    timer_rearm_locked();
    pthread_mutex_unlock(&g_lock);
}

int glcc_timer_fired(const glcc_timer *timer) {
    return timer->fired;
}

static int reactor_ctl(int op, int fd, unsigned events, void *ctx) {
    struct epoll_event ev;
    memset(&ev, 0, sizeof ev);
    ev.events = events | EPOLLONESHOT;
    ev.data.u64 = (uint64_t)(uintptr_t)ctx | GLCC_TAG_WAKER;
    return epoll_ctl(g_epfd, op, fd, &ev);
}

int glcc_reactor_wait(int fd, unsigned events, void *ctx) {
    glcc_exec_init();
    int rc = reactor_ctl(EPOLL_CTL_ADD, fd, events, ctx);
    if (rc < 0 && errno == EEXIST) rc = reactor_ctl(EPOLL_CTL_MOD, fd, events, ctx);
    return rc;
}

void glcc_reactor_del(int fd) {
    if (g_epfd >= 0) epoll_ctl(g_epfd, EPOLL_CTL_DEL, fd, 0);
}

/* ---- waking ---- */

void glcc_exec_wake(void *ctx) {
    if (!ctx) return;
    struct glcc_waker *waker = ctx;
    if (waker->magic != GLCC_WAKER_MAGIC) {
        /* Not ours: a plain driver context from the busy-poll runtime. */
        ((glcc_ctx *)ctx)->woken = 1;
        return;
    }
    glcc_task *t = waker->task;
    pthread_mutex_lock(&g_lock);
    if (t->state == T_PARKED) {
        t->state = T_READY;
        push_run_locked(t);
    } else if (t->state == T_RUNNING) {
        /* The worker will re-queue it when its poll returns PENDING. */
        t->state = T_READY;
    }
    pthread_mutex_unlock(&g_lock);
}

/* ---- frame hooks: the compiler's allocation path ---- */

void *__glcc_frame_alloc(unsigned long size, unsigned long align) {
    if (size == 0) size = 1;
    if (align <= sizeof(void *)) return malloc(size);
    void *frame = 0;
    if (posix_memalign(&frame, align, size) != 0) abort();
    return frame;
}

void __glcc_frame_free(void *frame) {
    if (!frame) return;
    pthread_mutex_lock(&g_lock);
    glcc_task *t = find_task_locked(frame);
    if (t) {
        if (t->state != T_ZOMBIE) {
            void *result_frame = t->result_frame;
            t->result_frame = 0;
            t->state = T_ZOMBIE;
            glcc_task **link = &g_tasks;
            while (*link && *link != t) link = &(*link)->all_next;
            if (*link) {
                *link = t->all_next;
                t->all_next = g_zombies;
                g_zombies = t;
            }
            pthread_mutex_unlock(&g_lock);
            free(result_frame);
            return;
        }
        pthread_mutex_unlock(&g_lock);
        return;
    }
    pthread_mutex_unlock(&g_lock);
    free(frame);
}

/* ---- task lifecycle ---- */

static unsigned glcc_join_poll(void *frame, void *ctx);

glcc_task *glcc_adopt(const void *handle_bytes) {
    glcc_exec_init();
    glcc_task *t = calloc(1, sizeof *t);
    if (!t) abort();
    memcpy(&t->inner, handle_bytes, sizeof t->inner);
    t->waker.magic = GLCC_WAKER_MAGIC;
    t->waker.task = t;
    t->result_size = ~0UL;
    pthread_cond_init(&t->done_cv, 0);

    pthread_mutex_lock(&g_lock);
    t->state = T_READY;
    t->all_next = g_tasks;
    g_tasks = t;
    push_run_locked(t);
    pthread_mutex_unlock(&g_lock);
    return t;
}

/* Copy the finished task's result into its offset-0 area and hand the
   result frame back to the caller. Called with g_lock held; the frame
   is freed outside the lock by the caller. */
static void *take_result_locked(glcc_task *t) {
    void *result_frame = t->result_frame;
    t->result_frame = 0;
    if (result_frame) {
        unsigned long n = t->result_size;
        if (n > GLCC_TASK_RESULT_MAX) n = GLCC_TASK_RESULT_MAX;
        if (n) memcpy(t->result.raw, result_frame, n);
    }
    return result_frame;
}

glcc_future glcc_join_future_(glcc_task *task, unsigned long size) {
    glcc_exec_init();
    pthread_mutex_lock(&g_lock);
    task->result_size = size;
    pthread_mutex_unlock(&g_lock);
    glcc_future handle;
    handle.frame = task;
    handle.poll = (void *)glcc_join_poll;
    return handle;
}

static unsigned glcc_join_poll(void *frame, void *ctx) {
    glcc_task *t = frame;
    pthread_mutex_lock(&g_lock);
    if (t->state != T_DONE) {
        t->joiner = ctx;
        pthread_mutex_unlock(&g_lock);
        return GLCC_PENDING;
    }
    void *result_frame = take_result_locked(t);
    pthread_mutex_unlock(&g_lock);
    free(result_frame);
    return GLCC_READY;
}

void glcc_join_impl(glcc_task *task, void *out, unsigned long size) {
    pthread_mutex_lock(&g_lock);
    while (task->state != T_DONE && task->state != T_ZOMBIE) {
        pthread_cond_wait(&task->done_cv, &g_lock);
    }
    void *result_frame = 0;
    if (task->state == T_DONE) {
        result_frame = task->result_frame;
        task->result_frame = 0;
        if (result_frame) {
            unsigned long n = size < GLCC_TASK_RESULT_MAX ? size : GLCC_TASK_RESULT_MAX;
            memcpy(task->result.raw, result_frame, n);
        }
    }
    pthread_mutex_unlock(&g_lock);
    if (size && out) {
        unsigned long n = size < GLCC_TASK_RESULT_MAX ? size : GLCC_TASK_RESULT_MAX;
        memcpy(out, task->result.raw, n);
    }
    free(result_frame);
    __glcc_frame_free(task);
}

/* The rt/glcc_async.h driver entry point: run a future to completion
   on the executor and wait here for the result. */
void glcc_block_on_impl(void *handle, void *out, unsigned long size) {
    glcc_task *t = glcc_adopt(handle);
    glcc_join_impl(t, out, size);
}

/* ---- workers ---- */

static void poll_task(glcc_task *t) {
    unsigned (*poll)(void *, void *) = (unsigned (*)(void *, void *))t->inner.poll;
    unsigned result = poll(t->inner.frame, &t->waker);

    pthread_mutex_lock(&g_lock);
    if (result == GLCC_READY) {
        t->state = T_DONE;
        t->result_frame = t->inner.frame;
        void *joiner = t->joiner;
        t->joiner = 0;
        pthread_cond_broadcast(&t->done_cv);
        pthread_mutex_unlock(&g_lock);
        if (joiner) glcc_exec_wake(joiner);
    } else if (t->state == T_RUNNING) {
        t->state = T_PARKED;
        pthread_mutex_unlock(&g_lock);
    } else {
        /* Woken while polling: stay runnable, the queue link is owed. */
        t->state = T_READY;
        push_run_locked(t);
        pthread_mutex_unlock(&g_lock);
    }
}

static void *worker_main(void *unused) {
    (void)unused;
    for (;;) {
        pthread_mutex_lock(&g_lock);
        while (!g_run_head && !g_shutdown) pthread_cond_wait(&g_work, &g_lock);
        if (!g_run_head) {
            pthread_mutex_unlock(&g_lock);
            return 0;
        }
        glcc_task *t = g_run_head;
        g_run_head = t->next;
        if (!g_run_head) g_run_tail = 0;
        t->next = 0;
        t->state = T_RUNNING;
        pthread_mutex_unlock(&g_lock);
        poll_task(t);
    }
}

/* ---- reactor ---- */

static void process_expired_timers(void) {
    uint64_t expirations;
    ssize_t rd = read(g_tfd, &expirations, sizeof expirations);
    (void)rd;

    glcc_timer *fired = 0;
    pthread_mutex_lock(&g_lock);
    unsigned long long now = glcc_now_ns();
    while (g_heap_len > 0 && g_heap[0]->deadline <= now) {
        glcc_timer *timer = heap_pop();
        timer->fired = 1;
        timer->fired_next = fired;
        fired = timer;
    }
    timer_rearm_locked();
    pthread_mutex_unlock(&g_lock);

    while (fired) {
        glcc_timer *next = fired->fired_next;
        void *ctx = fired->ctx;
        glcc_exec_wake(ctx);
        fired = next;
    }
}

static void *reactor_main(void *unused) {
    (void)unused;
    struct epoll_event events[64];
    for (;;) {
        int n = epoll_wait(g_epfd, events, 64, -1);
        if (n < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        for (int i = 0; i < n; i++) {
            uint64_t data = events[i].data.u64;
            switch (data & GLCC_TAG_MASK) {
            case GLCC_TAG_TIMER:
                process_expired_timers();
                break;
            case GLCC_TAG_SHUTDOWN:
                return 0;
            default:
                glcc_exec_wake((void *)(uintptr_t)(data & ~GLCC_TAG_MASK));
                break;
            }
        }
    }
}

/* ---- blocking pool ---- */

void glcc_pool_submit(glcc_pool_job *job) {
    glcc_exec_init();
    job->done = 0;
    job->next = 0;
    pthread_mutex_lock(&g_lock);
    if (g_pool_tail) {
        g_pool_tail->next = job;
    } else {
        g_pool_head = job;
    }
    g_pool_tail = job;
    pthread_cond_signal(&g_pool_work);
    pthread_mutex_unlock(&g_lock);
}

static void *pool_main(void *unused) {
    (void)unused;
    for (;;) {
        pthread_mutex_lock(&g_lock);
        while (!g_pool_head && !g_shutdown) pthread_cond_wait(&g_pool_work, &g_lock);
        if (!g_pool_head) {
            pthread_mutex_unlock(&g_lock);
            return 0;
        }
        glcc_pool_job *job = g_pool_head;
        g_pool_head = job->next;
        if (!g_pool_head) g_pool_tail = 0;
        pthread_mutex_unlock(&g_lock);

        job->fn(job->arg);
        job->done = 1;
        glcc_exec_wake(job->ctx);
    }
}

/* ---- startup and teardown ---- */

static void exec_start(void) {
    g_epfd = epoll_create1(EPOLL_CLOEXEC);
    g_tfd = timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC);
    g_efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (g_epfd < 0 || g_tfd < 0 || g_efd < 0) {
        perror("glcc_exec_init");
        abort();
    }

    struct epoll_event ev;
    memset(&ev, 0, sizeof ev);
    ev.events = EPOLLIN;
    ev.data.u64 = GLCC_TAG_TIMER;
    epoll_ctl(g_epfd, EPOLL_CTL_ADD, g_tfd, &ev);
    memset(&ev, 0, sizeof ev);
    ev.events = EPOLLIN;
    ev.data.u64 = GLCC_TAG_SHUTDOWN;
    epoll_ctl(g_epfd, EPOLL_CTL_ADD, g_efd, &ev);

    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu < 2) ncpu = 2;
    if (ncpu > (long)(sizeof g_worker_tids / sizeof g_worker_tids[0])) ncpu = (long)(sizeof g_worker_tids / sizeof g_worker_tids[0]);
    for (long i = 0; i < ncpu; i++) {
        if (pthread_create(&g_worker_tids[g_nworkers], 0, worker_main, 0) == 0) g_nworkers++;
    }
    if (pthread_create(&g_reactor_tid, 0, reactor_main, 0) != 0) abort();
    for (int i = 0; i < 4; i++) {
        if (pthread_create(&g_pool_tids[g_npool], 0, pool_main, 0) == 0) g_npool++;
    }
    g_started = 1;
}

void glcc_exec_init(void) {
    pthread_once(&g_once, exec_start);
}

void glcc_exec_shutdown(void) {
    if (!g_started) return;

    pthread_mutex_lock(&g_lock);
    g_shutdown = 1;
    pthread_cond_broadcast(&g_work);
    pthread_cond_broadcast(&g_pool_work);
    pthread_mutex_unlock(&g_lock);

    uint64_t one = 1;
    ssize_t w = write(g_efd, &one, sizeof one);
    (void)w;

    for (int i = 0; i < g_nworkers; i++) pthread_join(g_worker_tids[i], 0);
    pthread_join(g_reactor_tid, 0);
    for (int i = 0; i < g_npool; i++) pthread_join(g_pool_tids[i], 0);

    glcc_task *t = g_tasks;
    while (t) {
        glcc_task *next = t->all_next;
        free(t->result_frame);
        free(t);
        t = next;
    }
    t = g_zombies;
    while (t) {
        glcc_task *next = t->all_next;
        free(t);
        t = next;
    }
    g_tasks = 0;
    g_zombies = 0;
    g_run_head = 0;
    g_run_tail = 0;
    g_pool_head = 0;
    g_pool_tail = 0;
    free(g_heap);
    g_heap = 0;
    g_heap_len = 0;
    g_heap_cap = 0;

    close(g_epfd);
    close(g_tfd);
    close(g_efd);
    g_epfd = g_tfd = g_efd = -1;
    g_started = 0;
}
