/* Async/await in C with glcc: make && make run.
 *
 * `-fglcc-async` turns an `__attribute__((async))` function into a
 * stackless coroutine: calling it returns a `_Future(T)` handle, and
 * `__builtin_await` suspends until a future completes. `glcc_block_on`
 * drives the tree to completion on this thread. See docs/async-abi.md
 * for the contract, and examples/async/runtime/ for a multithreaded
 * scheduler built on it.
 */
#include <stdio.h>

#include <glcc_async.h>

static int failures;

static void report(const char *what, long got, long want) {
    if (got == want) {
        printf("  %-16s = %ld\n", what, got);
    } else {
        printf("  %-16s = %ld MISMATCH (want %ld)\n", what, got, want);
        failures++;
    }
}

/* A leaf coroutine: no awaits, so the first poll runs it to the end. */
__attribute__((async)) int add1(int x) { return x + 1; }

/* A named (first-class) future awaited later. */
__attribute__((async)) int twice(int x) {
    _Future(int) inner = add1(x);
    return __builtin_await(inner) * 2;
}

/* Values computed before a suspension and after it. */
__attribute__((async)) int sum_two(int x) {
    int a = x + 10;
    int b = __builtin_await(add1(x));
    int c = a + b + 100;
    int d = __builtin_await(add1(c));
    return d;
}

/* A local array whose address must live across a suspension. */
__attribute__((async)) int sum_array(int x) {
    int v[3];
    v[0] = x;
    v[1] = x + 1;
    v[2] = x + 2;
    int idx = __builtin_await(add1(1));
    return v[0] + v[1] + v[2] + idx;
}

/* A suspension inside a loop: the state machine resumes mid-iteration. */
__attribute__((async)) int loop_await(int n) {
    int total = 0;
    for (int i = 0; i < n; i++)
        total += __builtin_await(add1(i));
    return total;
}

/* A void result. */
static int touched;

__attribute__((async)) void bump(void) {
    int r = __builtin_await(add1(1));
    touched += r;
}

/* A future built in main and awaited inside another coroutine. */
__attribute__((async)) int drain(_Future(int) f) { return __builtin_await(f) + 1; }

int main(void) {
    int r;

    printf("awaiting futures from main:\n");

    _Future(int) f1 = add1(41);
    glcc_block_on(f1, &r);
    report("add1(41)", r, 42);

    _Future(int) f2 = twice(20);
    glcc_block_on(f2, &r);
    report("twice(20)", r, 42);

    _Future(int) f3 = sum_two(1);
    glcc_block_on(f3, &r);
    report("sum_two(1)", r, 114);

    _Future(int) f4 = sum_array(10);
    glcc_block_on(f4, &r);
    report("sum_array(10)", r, 35);

    _Future(int) f5 = loop_await(4);
    glcc_block_on(f5, &r);
    report("loop_await(4)", r, 10);

    _Future(void) f6 = bump();
    glcc_block_on_void(f6);
    report("bump()", touched, 2);

    printf("a future awaited inside another coroutine:\n");
    _Future(int) f7 = drain(add1(40));
    glcc_block_on(f7, &r);
    report("drain(add1(40))", r, 42);

    printf("\n%s\n", failures ? "FAILED" : "all checks passed");
    return failures != 0;
}
