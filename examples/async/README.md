# Async examples

Three demos of the `-fglcc-async` extension, each with `make && make run`:

- `basic/` — the smallest complete demo: coroutines, `_Future(T)`
  handles, `__builtin_await`, and the busy-poll runtime in `rt/`,
  driven by `glcc_block_on`.
- `executor/` — a check-driver for the multithreaded runtime below:
  spawn/join, timers, sockets, the blocking pool, and a stress run.
- `loops/` — one coroutine spawned twice as two infinite loops, each
  printing its own accumulator from the worker thread that polled it:
  concurrency with no `pthread_create` in the demo.

`runtime/` is not an example. It is the implementation of a Linux
executor (worker threads, epoll reactor, timerfd sleeps, blocking pool)
that the `executor/` demo links against. See its README for the design,
and `docs/async-abi.md` for the compiler ABI it is built on.
