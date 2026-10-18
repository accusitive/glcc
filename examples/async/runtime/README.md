# glcc executor — a Linux tokio, in C, over the future ABI

This directory is a prototype runtime for `-fglcc-async` futures: worker
threads poll tasks, an epoll reactor wakes them on IO, a timerfd drives
sleeps, and a small thread pool absorbs blocking calls. It exists to show
that multithreaded async/await works end to end on the compiler ABI
before the real target — an all-async OS — has one of its own. Nothing
here is compiler-specific beyond the ABI contract in
[`docs/async-abi.md`](../../docs/async-abi.md): a handle, a poll
function, and the frame hooks.

```
cd examples/async/executor
make && make run
```

which runs:

```
glcc -fglcc-async -O2 -I../runtime -o build/executor demo.c \
    ../runtime/exec.c ../runtime/io.c -pthread
```

`exec.c` is the scheduler; `io.c` the leaf futures (`glcc_async_read`,
`glcc_async_write_all`, `glcc_async_accept`, `glcc_async_sleep`,
`glcc_async_yield`, `glcc_async_blocking_void`); `demo.c` drives all of
it. The runtime deliberately does not link `rt/glcc_async.c`: it defines
its own `glcc_block_on_impl` and frame hooks.

## API in one screen

```c
__attribute__((async)) int child(int x) {
    __builtin_await(glcc_async_sleep(10 * 1000 * 1000)); /* 10 ms */
    return x + 1;
}

__attribute__((async)) int parent(void) {
    glcc_task *t = glcc_spawn(child(41));        /* runs on any worker */
    _Future(int) f = glcc_task_future(t, int);   /* awaitable handle   */
    return __builtin_await(f);                   /* 42                 */
}

int main(void) {
    int r;
    _Future(int) f = parent();
    glcc_block_on(f, &r);      /* run on the executor, block main     */
    printf("%d\n", r);
    glcc_exec_shutdown();
}
```

- `glcc_spawn(call)` starts an async call as an independent task (like
  `tokio::spawn`). The call must return `_Future(T)`; anything else is a
  compile-time error.
- `glcc_task_future(t, T)` makes a task awaitable from inside a
  coroutine; awaiting it suspends the awaiter until the task finishes.
  The awaiter reads the task's result straight out of the task box, so
  `T` must match and must not exceed `GLCC_TASK_RESULT_MAX` (128 bytes).
- `glcc_join(t, &out)` blocks a *foreign* thread (main, a test) until
  the task is done. Don't join from a worker/pool thread: nothing
  preempts the task you're waiting on.
- `glcc_block_on(f, &r)` is the `rt/glcc_async.h` entry point: it
  adopts a future as a task and joins it, so top-level code runs on the
  executor like everything else.

## How it works

**Tasks.** A task is a box with room for the result at offset 0 (so the
compiler's await lowering can read it, exactly as the ABI describes),
the spawned handle, state, waker, joiner slot, and a condition variable
for `glcc_join`. Spawned tasks sit on a FIFO run queue; workers take
one, poll it, and either park it (PENDING) or finish it (READY). A wake
that lands while the task is running just flips it back to runnable,
and the worker re-queues it after the poll returns — the classic
run-state protocol.

**Waker.** Every task owns a `{magic, task*}` waker whose address is the
`ctx` passed into every poll of that task's tree. IO leaves hand that
pointer to the reactor; a wake marks the task runnable and signals the
workers. A `ctx` that fails the magic check is treated as the
busy-poll runtime's `glcc_ctx`, so leaves can be shared between both
drivers without crashing.

**Reactor.** One thread in `epoll_wait`. Every registration is
`EPOLLONESHOT` and carries the waker in `data.u64`; leaves arm with
`glcc_reactor_wait(fd, EPOLLIN/OUT, ctx)`, which is an ADD the first
time and a MOD afterwards — including the re-arm after an event, so a
spurious wake that finds the fd drained simply arms again. `EPOLLERR`
and `EPOLLHUP` need no registration: the leaf retries the syscall and
reports the real error.

**Timers.** A binary min-heap of `glcc_timer` nodes (embedded in leaf
frames), a single `timerfd` re-armed to the earliest deadline. On expiry
the reactor pops the due nodes under the lock, then wakes them. Sleeps
cost one syscall per deadline change, not one per sleeper.

**Blocking pool.** `glcc_async_blocking_void(fn, arg)` submits a job to
four pool threads; completion stores the result and wakes the task. A
thread blocked in `read()` or `usleep()` never occupies a scheduler
worker.

**Frame lifetime.** The compiler frees a child frame at the end of
every await, and a task box is freed exactly once: by the waiting
coroutine after `glcc_task_future`, or by `glcc_join`. But an epoll
event may already have been copied out of the kernel with the waker
address in it when that happens — and `EPOLL_CTL_DEL` cannot un-copy
it. So `__glcc_frame_free` does not actually free a task box: it moves
it to a zombie list and keeps the memory, making every stale wake a
checked no-op. Zombies are reclaimed at `glcc_exec_shutdown`. That is
the whole trick, and the reason the executor overrides the frame hooks.

## Compiler gaps this runtime works around

- glibc's `bind`/`connect`/`accept4`/`getsockname` take
  `__attribute__((transparent_union))` parameters; glcc does not model
  the attribute yet, so `io.c` and `demo.c` build the `__SOCKADDR_ARG`
  union explicitly before each call. Supporting transparent unions in
  the frontend would delete those wrappers.
- `_Static_assert` is not implemented; the spawn/await macros get
  compile-time checks from a negative-width bit-field instead.
- There is no `__builtin_ctx()` yet, so a *coroutine* cannot reach its
  own `ctx`; all IO lives in hand-written leaf futures like `io.c`.
  That builtin is the first compiler follow-up.
- Building a `_Future(T)` value in C means `memcpy`ing 16 bytes from a
  `glcc_future` — that is what `GLCC_LEAF` in `io.c` and
  `glcc_task_future` in `exec.h` do.

## v1 rules

- A task result is at most `GLCC_TASK_RESULT_MAX` bytes.
- A task/future is single-awaiter; awaiting twice or joining while an
  awaiter is parked is undefined.
- No cancellation; a suspended-forever task keeps its frame, and a
  detached task's result frame is only reclaimed at shutdown.
- `glcc_exec_shutdown` is terminal (call it at exit); init is implicit
  and idempotent.
- `GLCC_STANDALONE` lets clang compile `exec.c`/`io.c` for
  cross-checking; coroutine syntax still needs glcc.

## What the demo exercises

`demo.c` covers: block_on, spawn/join (64 tasks), nested awaits,
running on ≥2 distinct worker threads, timer ordering, parallel sleeps,
join latency, 128 concurrent timers, socketpair reads, a loopback echo
server, blocking offload, worker liveness while the pool is saturated,
spawning and awaiting tasks from coroutines (including a spawn-await
loop), overlapping children, void and detached tasks, yield, and a
200-task stress run with checksums. 328 checks in all; the whole suite
runs in well under a second.
