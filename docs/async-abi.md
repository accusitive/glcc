# glcc async ABI — coroutines, futures, and the waker hand-off

`-fglcc-async` turns C functions into stackless coroutines. A call to an
async function no longer runs the body: it allocates a frame, starts the
function as a *poll* state machine, and returns a `_Future(T)` handle. The
caller (or any other driver) polls the handle until it reports ready, then
reads the result out of the frame.

The feature is opt-in. Without the flag an `__attribute__((async))`
function is an ordinary function (the attribute warns and is ignored), and
`_Future`/`__builtin_await` are errors. Nothing in the pipeline changes
for existing C.

```
__attribute__((async)) int add1(int x) { return x + 1; }

_Future(int) f = add1(41);          /* allocates + starts, returns handle */
int r = __builtin_await(f);         /* suspend until f completes        */
```

## Surface

- `__attribute__((async))` on a function declaration/definition. Rejected
  on `main`, on variadic functions, and on `naked` functions.
- `_Future(T) a;` — the handle value. One nominal struct type per `T`,
  interned by the frontend. `T` may be `void`.
- `__builtin_await(e)` — only valid inside an async function; `e` must be
  a `_Future(T)`; yields `T`. Suspends the current coroutine.

`sizeof(_Future(T)) == 16` on LP64: two words.

## Handle layout

```c
typedef struct glcc_future {   /* the compiler's `_Future(T)` */
    void *frame;               /* the boxed frame (see below) */
    void *poll;                /* resume entry point          */
} glcc_future;
```

`poll` points at the resume function, cast to a raw code pointer; the
runtime calls it as:

```c
unsigned (*poll)(void *frame, void *ctx);
```

The handle is returned by value and lowered like any 16-byte struct
(two registers on x86-64). The frontend's nominal type has two unnamed
`void *` fields, so C code should treat it as opaque and use the runtime
helpers (`glcc_block_on`) rather than inspecting it.

## Frame layout

One heap allocation per future, size/align passed to the allocator. The
compiler lays out fields in this order, with target alignment:

| offset | contents |
| ------ | -------- |
| 0 | the coroutine's result (`T`), *omitted when `T` is `void`* |
| next | `i32` resume state (0 = start; `k` = at await `k`) |
| ... | one slot per entry parameter |
| ... | one slot per stack slot (`alloca`) whose address escapes a suspension |
| ... | one 16-byte handle slot per await site |
| ... | one slot per SSA value live across a suspension (spills) |

Result-at-offset-0 is the ABI contract: a driver that knows only the
handle and the result size can poll, `memcpy` the result out, and free
the frame without knowing anything else about the coroutine.

## Poll contract

- The resume function takes `(void *frame, void *ctx)` and returns
  `1` (READY) when the coroutine has finished — its result then sits at
  frame offset 0 — or `0` (PENDING) when it suspended on a child future.
- A driver polls while PENDING.
- On READY the poller (1) copies the result out if it wants it and
  (2) frees the frame via the frame hooks. *Owning* the future means
  owning the frame.
- Re-polling a READY future is undefined: the driver has freed the frame
  by then. `glcc_block_on` never does.
- `ctx` is passed through unchanged to every poll, including the polls of
  nested child futures. It is the single channel between wakers and the
  scheduler.

The compiler emits both returns itself: `Return 1` at the coroutine's
original return points (after storing the result at offset 0) and
`Return 0` at each suspension point.

## Start and resume

An async function `f` becomes two functions:

- `f` — the *start*. Same name, same parameters, but its result type is
  the handle. It allocates the frame, stores the parameters and state 0,
  fills the handle with the frame pointer and the resume address, and
  returns it. Call sites do not change: calling `f` *is* the spawn.
- `f$resume` — the *resume*. Local symbol, `inline`/`weak` cleared, no
  debug info. Its entry block switches on the frame's state: state 0
  runs the body from the top, state `k` re-enters at await `k`, which
  first re-polls the child future. Dispatch also re-materializes the
  frame-derived values (parameter loads, stack-slot addresses) so both
  entry paths see them.

An async function with no `await` still becomes the pair: the resume
runs the body to completion on the first poll.

## Await lowering

At `r = __builtin_await(h)` the coroutine:

1. stores the handle `h` into its frame's handle slot for this await and
   writes the resume state;
2. jumps to a wait block that loads `h.frame`/`h.poll` from the slot,
   calls `poll(h.frame, ctx)`, and branches on `== 1`;
3. PENDING returns 0 to its own poller;
4. READY loads the child's result (from the child frame, offset 0),
   frees the child frame, and continues with the instructions that
   followed the await.

## Frame hooks

```c
void *__glcc_frame_alloc(unsigned long size, unsigned long align);
void  __glcc_frame_free(void *frame);
```

Both are weak in `rt/glcc_async.c` (malloc/free). An embedder — an OS
kernel with per-task pools, an arena per scheduler — can link its own
definitions and the coroutines will use them. `alloc` must return memory
aligned to `align`; `glcc_block_on` frees the top-level frame, and each
coroutine frees the frames of the futures it awaits.

## Runtime and drivers

`rt/glcc_async.h` + `rt/glcc_async.c` provide the v1 blocking driver:

- `glcc_wake(ctx)` records a wake-up in `ctx->woken`; wakers (interrupt
  handlers, timer callbacks) that were handed the `ctx` use it.
- `glcc_block_on(future, &out)` / `glcc_block_on_void(future)` poll to
  completion, copy the result out, and free the frame. The first
  argument must be an lvalue (the macro takes its address).

The v1 driver busy-polls, so `woken` is only informative. Replacing
`glcc_block_on_impl` (or linking a different runtime entirely) is the
intended embedding path: it needs the handle layout, the poll contract,
and nothing else. A scheduler with an idle state parks until `woken` is
set rather than spinning. `examples/async/runtime/` is a worked example: a
multithreaded Linux runtime (worker threads, epoll reactor, timerfd
sleeps, blocking pool) built only on this ABI.

## v1 rules and caveats

- A future is single-owner and single-await: awaiting it twice, awaiting
  a handle while another driver polls it, or awaiting a completed handle
  is undefined (the frame is already freed).
- Dropping a handle without awaiting it leaks its frame; there is no
  cancellation and no `Drop`-style finalizer. A coroutine that is
  suspended forever also keeps any child frames it holds in slots.
- Awaiting is not a context switch by itself: `__builtin_await` returns
  to the poller, and whatever drives the poller decides when to poll
  again. A waker cannot resume a coroutine directly, only mark `ctx`.
- No `main` trampoline yet: write a synchronous `main` and block on the
  top-level future. No variadic, no `naked`, no async debug info.
- The awaiter trusts the child's result type: `_Future(T)` is nominal,
  so a handle punned through `void *` and awaited as the wrong `T` reads
  garbage at offset 0.

## Pipeline position

`glcc_mir::coroutine::LowerCoroutines` is a MIR module pass that runs
immediately after MIR lowering, before the optimizer and at every opt
level. It is inert unless a function carries the coroutine flag, which
only the frontend sets under `-fglcc-async`. At `-O1+` the start
functions inline into callers like any small function; resumes never do
(their two-parameter signature never matches a call site).
