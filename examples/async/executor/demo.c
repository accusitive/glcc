/* demo.c — exercises the glcc executor prototype in examples/async/runtime/.
   Build and run: make && make run. */
#define _GNU_SOURCE

#include "exec.h"
#include "io.h"

#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static int checks;
static int failures;

#define CHECK(cond)                                                     \
    do {                                                                \
        checks++;                                                       \
        if (!(cond)) {                                                  \
            failures++;                                                 \
            printf("    FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
        }                                                               \
    } while (0)

static void run(const char *name, void (*test)(void)) {
    int before = failures;
    unsigned long long t0 = glcc_now_ns();
    test();
    unsigned long long ms = (glcc_now_ns() - t0) / 1000000ULL;
    printf("%-36s %6llu ms  %s\n", name, ms, failures == before ? "ok" : "FAILED");
}

/* glibc's bind/connect/getsockname take transparent unions, which glcc
   does not model yet; build the union explicitly. */
static int sys_bind(int fd, const struct sockaddr *sa, socklen_t len) {
    __CONST_SOCKADDR_ARG u;
    memset(&u, 0, sizeof u);
    u.__sockaddr__ = sa;
    return bind(fd, u, len);
}

static int sys_connect(int fd, const struct sockaddr *sa, socklen_t len) {
    __CONST_SOCKADDR_ARG u;
    memset(&u, 0, sizeof u);
    u.__sockaddr__ = sa;
    return connect(fd, u, len);
}

static int sys_getsockname(int fd, struct sockaddr *sa, socklen_t *len) {
    __SOCKADDR_ARG u;
    memset(&u, 0, sizeof u);
    u.__sockaddr__ = sa;
    return getsockname(fd, u, len);
}

/* ---- basics ---- */

__attribute__((async)) int add1(int x) { return x + 1; }

static void test_block_on(void) {
    int r = 0;
    _Future(int) f = add1(41);
    glcc_block_on(f, &r);
    CHECK(r == 42);
}

__attribute__((async)) int square(int x) { return x * x; }

static void test_spawn_join(void) {
    enum { N = 64 };
    glcc_task *tasks[N];
    for (int i = 0; i < N; i++) tasks[i] = glcc_spawn(square(i));
    for (int i = 0; i < N; i++) {
        int r = -1;
        glcc_join(tasks[i], &r);
        CHECK(r == i * i);
    }
}

/* ---- nested futures and thread diversity ---- */

__attribute__((async)) int leaf_compute(int x) { return x * 3; }
__attribute__((async)) int mid_level(int x) { return __builtin_await(leaf_compute(x)) + 1; }

__attribute__((async)) int top_level(int x) {
    int a = __builtin_await(mid_level(x));
    int b = __builtin_await(mid_level(a));
    return a * 100 + b;
}

static void test_nested_await(void) {
    int r = 0;
    _Future(int) f = top_level(7);
    glcc_block_on(f, &r);
    CHECK(r == 2200 + 67);
}

static pthread_mutex_t tid_mu = PTHREAD_MUTEX_INITIALIZER;
static unsigned long seen_tids[256];
static int nseen_tids;

__attribute__((async)) int stamp_thread(void) {
    __builtin_await(glcc_async_sleep(2 * 1000 * 1000));
    pthread_mutex_lock(&tid_mu);
    if (nseen_tids < (int)(sizeof seen_tids / sizeof seen_tids[0])) {
        seen_tids[nseen_tids++] = (unsigned long)pthread_self();
    }
    pthread_mutex_unlock(&tid_mu);
    return 0;
}

static void test_thread_diversity(void) {
    enum { N = 64 };
    glcc_task *tasks[N];
    nseen_tids = 0;
    for (int i = 0; i < N; i++) tasks[i] = glcc_spawn(stamp_thread());
    for (int i = 0; i < N; i++) glcc_join_void(tasks[i]);

    CHECK(nseen_tids == N);
    int distinct = 0;
    for (int i = 0; i < nseen_tids; i++) {
        int first = 1;
        for (int j = 0; j < i; j++) {
            if (seen_tids[j] == seen_tids[i]) first = 0;
        }
        distinct += first;
    }
    printf("    ran on %d distinct worker threads\n", distinct);
    CHECK(distinct >= 2);
}

/* ---- timers ---- */

static pthread_mutex_t order_mu = PTHREAD_MUTEX_INITIALIZER;
static int order[8];
static int norder;

__attribute__((async)) int sleeper(int idx, unsigned long long ns) {
    __builtin_await(glcc_async_sleep(ns));
    pthread_mutex_lock(&order_mu);
    order[norder++] = idx;
    pthread_mutex_unlock(&order_mu);
    return idx;
}

static void test_sleep_order(void) {
    static const unsigned long long ns[3] = {5, 30, 60};
    glcc_task *tasks[3];
    norder = 0;
    for (int i = 0; i < 3; i++) tasks[i] = glcc_spawn(sleeper(i, ns[i] * 1000 * 1000));
    for (int i = 0; i < 3; i++) glcc_join_void(tasks[i]);
    CHECK(norder == 3);
    CHECK(order[0] == 0 && order[1] == 1 && order[2] == 2);
}

static void test_parallel_sleeps(void) {
    glcc_task *a = glcc_spawn(sleeper(0, 80 * 1000 * 1000));
    glcc_task *b = glcc_spawn(sleeper(1, 80 * 1000 * 1000));
    unsigned long long t0 = glcc_now_ns();
    glcc_join_void(a);
    glcc_join_void(b);
    unsigned long long ms = (glcc_now_ns() - t0) / 1000000ULL;
    printf("    two 80 ms sleeps joined in %llu ms\n", ms);
    CHECK(ms < 140);
}

static void test_join_waits(void) {
    glcc_task *t = glcc_spawn(sleeper(0, 60 * 1000 * 1000));
    unsigned long long t0 = glcc_now_ns();
    glcc_join_void(t);
    unsigned long long ms = (glcc_now_ns() - t0) / 1000000ULL;
    printf("    join returned after %llu ms\n", ms);
    CHECK(ms >= 40);
}

static void test_many_timers(void) {
    enum { N = 128 };
    glcc_task *tasks[N];
    for (int i = 0; i < N; i++) {
        tasks[i] = glcc_spawn(sleeper(i % 8, 1000 * 1000 + (unsigned long long)i * 200000));
    }
    for (int i = 0; i < N; i++) glcc_join_void(tasks[i]);
}

/* ---- io ---- */

__attribute__((async)) int read_message(int fd) {
    char buf[32];
    _Future(ssize_t) rf = glcc_async_read(fd, buf, sizeof buf);
    ssize_t n = __builtin_await(rf);
    return n == 5 && memcmp(buf, "hello", 5) == 0;
}

__attribute__((async)) void write_later(int fd, unsigned long long delay) {
    __builtin_await(glcc_async_sleep(delay));
    _Future(ssize_t) wf = glcc_async_write_all(fd, "hello", 5);
    ssize_t n = __builtin_await(wf);
    (void)n;
}

static void test_socketpair(void) {
    int sv[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0, sv) == 0);

    glcc_task *writer = glcc_spawn(write_later(sv[1], 20 * 1000 * 1000));

    int r = 0;
    _Future(int) f = read_message(sv[0]);
    glcc_block_on(f, &r);
    CHECK(r == 1);

    glcc_join_void(writer);
    close(sv[0]);
    close(sv[1]);
}

static int make_listener(void) {
    int lfd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (lfd < 0) return -1;
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof sa);
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    sa.sin_port = 0;
    if (sys_bind(lfd, (const struct sockaddr *)&sa, sizeof sa) != 0) {
        close(lfd);
        return -1;
    }
    if (listen(lfd, 8) != 0) {
        close(lfd);
        return -1;
    }
    return lfd;
}

static struct sockaddr_in listener_peer(int lfd) {
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof sa);
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t slen = sizeof sa;
    sys_getsockname(lfd, (struct sockaddr *)&sa, &slen);
    return sa;
}

__attribute__((async)) int echo_once(int lfd) {
    _Future(int) af = glcc_async_accept(lfd, SOCK_NONBLOCK);
    int cfd = __builtin_await(af);
    if (cfd < 0) return -1;

    char buf[64];
    _Future(ssize_t) rf = glcc_async_read(cfd, buf, sizeof buf);
    ssize_t n = __builtin_await(rf);
    if (n <= 0) {
        close(cfd);
        return -2;
    }
    _Future(ssize_t) wf = glcc_async_write_all(cfd, buf, (size_t)n);
    ssize_t w = __builtin_await(wf);
    close(cfd);
    return w == n ? (int)n : -3;
}

static void test_echo_once(void) {
    int lfd = make_listener();
    CHECK(lfd >= 0);
    struct sockaddr_in peer = listener_peer(lfd);

    int client = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(client >= 0);
    CHECK(sys_connect(client, (const struct sockaddr *)&peer, sizeof peer) == 0);

    glcc_task *server = glcc_spawn(echo_once(lfd));

    const char *msg = "ping";
    CHECK(write(client, msg, 4) == 4);
    char reply[8];
    ssize_t n = read(client, reply, sizeof reply);
    CHECK(n == 4);
    if (n == 4) CHECK(memcmp(reply, msg, 4) == 0);

    int echoed = 0;
    glcc_join(server, &echoed);
    CHECK(echoed == 4);

    close(client);
    close(lfd);
}

/* ---- blocking offload ---- */

static void blocking_work(void *arg) {
    usleep(100 * 1000);
    *(int *)arg = 7;
}

__attribute__((async)) int offload(int *out) {
    __builtin_await(glcc_async_blocking_void(blocking_work, out));
    return *out;
}

static void test_blocking_offload(void) {
    int out = 0;
    _Future(int) f = offload(&out);
    glcc_block_on(f, &out);
    CHECK(out == 7);
}

__attribute__((async)) int quick_sleep(int *done) {
    __builtin_await(glcc_async_sleep(10 * 1000 * 1000));
    *done = 1;
    return 0;
}

static void test_blocking_keeps_workers_free(void) {
    int outs[4] = {0};
    glcc_task *busy[4];
    for (int i = 0; i < 4; i++) busy[i] = glcc_spawn(offload(&outs[i]));

    int done = 0;
    unsigned long long t0 = glcc_now_ns();
    _Future(int) f = quick_sleep(&done);
    int r = 0;
    glcc_block_on(f, &r);
    unsigned long long ms = (glcc_now_ns() - t0) / 1000000ULL;
    printf("    10 ms sleep with the pool busy took %llu ms\n", ms);
    CHECK(done == 1);
    CHECK(ms < 80);

    for (int i = 0; i < 4; i++) {
        glcc_join(busy[i], &outs[i]);
        CHECK(outs[i] == 7);
    }
}

/* ---- tasks from coroutines ---- */

__attribute__((async)) int child_delayed(int x, unsigned long long ns) {
    __builtin_await(glcc_async_sleep(ns));
    return x + 1;
}

__attribute__((async)) int parent_spawns(int x) {
    glcc_task *child = glcc_spawn(child_delayed(x, 10 * 1000 * 1000));
    _Future(int) cf = glcc_task_future(child, int);
    int r = __builtin_await(cf);
    return r * 10;
}

static void test_spawn_from_coroutine(void) {
    int r = 0;
    _Future(int) f = parent_spawns(1);
    glcc_block_on(f, &r);
    CHECK(r == 20);
}

__attribute__((async)) int parent_overlaps(void) {
    glcc_task *a = glcc_spawn(child_delayed(0, 70 * 1000 * 1000));
    glcc_task *b = glcc_spawn(child_delayed(0, 70 * 1000 * 1000));
    _Future(int) fa = glcc_task_future(a, int);
    _Future(int) fb = glcc_task_future(b, int);
    int ra = __builtin_await(fa);
    int rb = __builtin_await(fb);
    return ra + rb;
}

static void test_children_overlap(void) {
    unsigned long long t0 = glcc_now_ns();
    int r = 0;
    _Future(int) f = parent_overlaps();
    glcc_block_on(f, &r);
    unsigned long long ms = (glcc_now_ns() - t0) / 1000000ULL;
    printf("    two 70 ms children awaited in %llu ms\n", ms);
    CHECK(r == 2);
    CHECK(ms < 130);
}

__attribute__((async)) int spawn_many_children(int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        glcc_task *c = glcc_spawn(child_delayed(i, (unsigned long long)(i % 3) * 1000 * 1000));
        _Future(int) f = glcc_task_future(c, int);
        total += __builtin_await(f);
    }
    return total;
}

static void test_children_in_loop(void) {
    int r = 0;
    _Future(int) f = spawn_many_children(20);
    glcc_block_on(f, &r);
    CHECK(r == 20 * 21 / 2);
}

__attribute__((async)) void set_flag(int *p, unsigned long long ns) {
    __builtin_await(glcc_async_sleep(ns));
    *p = 1;
}

__attribute__((async)) int await_void_child(int *p) {
    glcc_task *t = glcc_spawn(set_flag(p, 10 * 1000 * 1000));
    __builtin_await(glcc_task_future_void(t));
    return *p;
}

static void test_void_task(void) {
    int flag = 0;
    int r = 0;
    _Future(int) f = await_void_child(&flag);
    glcc_block_on(f, &r);
    CHECK(r == 1);
}

__attribute__((async)) int detached_setter(int *p) {
    __builtin_await(glcc_async_sleep(5 * 1000 * 1000));
    *p = 1;
    return 0;
}

static void test_detached(void) {
    int flag = 0;
    glcc_spawn(detached_setter(&flag));
    int unused = 0;
    _Future(int) f = quick_sleep(&unused);
    glcc_block_on(f, &unused);
    CHECK(flag == 1);
}

/* ---- yield and stress ---- */

static void test_yield(void) {
    enum { N = 32 };
    glcc_task *tasks[N];
    for (int i = 0; i < N; i++) tasks[i] = glcc_spawn(square(3));
    for (int i = 0; i < N; i++) {
        int r = 0;
        glcc_join(tasks[i], &r);
        CHECK(r == 9);
    }

    int r = 0;
    _Future(int) f = parent_spawns(2);
    glcc_block_on(f, &r);
    CHECK(r == 30);

    /* an explicit yield: still completes */
    _Future(void) y = glcc_async_yield();
    glcc_block_on_void(y);
}

__attribute__((async)) int stress_task(int i) {
    __builtin_await(glcc_async_sleep((unsigned long long)(i % 4) * 1000 * 1000));
    int acc = 0;
    for (int k = 0; k < 3; k++) {
        acc += __builtin_await(child_delayed(i + k, 0));
    }
    return acc;
}

static void test_stress(void) {
    enum { N = 200 };
    glcc_task *tasks[N];
    for (int i = 0; i < N; i++) tasks[i] = glcc_spawn(stress_task(i));
    long sum = 0;
    for (int i = 0; i < N; i++) {
        int r = 0;
        glcc_join(tasks[i], &r);
        int expected = (i + 1) + (i + 2) + (i + 3);
        if (r != expected) failures++;
        checks++;
        sum += r;
    }
    printf("    %d tasks, checksum %ld\n", N, sum);
    CHECK(sum > 0);
}

int main(void) {
    printf("glcc executor prototype demo\n\n");
    run("block_on", test_block_on);
    run("spawn/join", test_spawn_join);
    run("nested await", test_nested_await);
    run("thread diversity", test_thread_diversity);
    run("sleep order", test_sleep_order);
    run("parallel sleeps", test_parallel_sleeps);
    run("join waits", test_join_waits);
    run("many timers", test_many_timers);
    run("socketpair read", test_socketpair);
    run("loopback echo", test_echo_once);
    run("blocking offload", test_blocking_offload);
    run("blocking keeps workers free", test_blocking_keeps_workers_free);
    run("spawn from coroutine", test_spawn_from_coroutine);
    run("children overlap", test_children_overlap);
    run("children in loop", test_children_in_loop);
    run("void task await", test_void_task);
    run("detached task", test_detached);
    run("yield", test_yield);
    run("stress", test_stress);

    glcc_exec_shutdown();
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures != 0;
}
