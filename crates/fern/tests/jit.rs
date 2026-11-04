//! End-to-end fern tests: source -> MIR -> Cranelift -> JIT -> run.
//!
//! Also mirror the C driver's pass pipeline when optimizing, and lower the
//! C runtime into the same JIT module, so the program's `fern_alloc` and
//! frame calls bind to real code. Run with `cargo test -p fern`.

// These execute generated code in-process. The JIT harness binds POSIX
// symbols and drives tests through Unix shell utilities, so it stays off
// Windows even though the backend now lowers Win64.
//
// aarch64 hosts are excluded too: cranelift-jit only implements
// `is_pic=true` on x86_64, so on aarch64 an extern whose libc mapping
// lands outside ADRP's ±2GB range panics at relocation time — under
// ASLR that is per-run luck, and the suite fails nondeterministically.
#![cfg(not(any(windows, target_arch = "aarch64")))]

use std::collections::HashMap;

use cranelift_jit::{JITBuilder, JITModule};
use glcc_mir::ir::Module;

unsafe extern "C" {
    fn memcpy(dst: *mut core::ffi::c_void, src: *const core::ffi::c_void, n: usize) -> *mut core::ffi::c_void;
    fn memset(dst: *mut core::ffi::c_void, c: i32, n: usize) -> *mut core::ffi::c_void;
    fn write(fd: i32, buf: *const core::ffi::c_void, n: usize) -> isize;
    fn read(fd: i32, buf: *mut core::ffi::c_void, n: usize) -> isize;
    fn abort();
    fn malloc(size: usize) -> *mut core::ffi::c_void;
    fn realloc(p: *mut core::ffi::c_void, size: usize) -> *mut core::ffi::c_void;
    fn free(p: *mut core::ffi::c_void);
    fn fopen(path: *const core::ffi::c_char, mode: *const core::ffi::c_char) -> *mut core::ffi::c_void;
    fn fread(ptr: *mut core::ffi::c_void, size: usize, count: usize, stream: *mut core::ffi::c_void) -> usize;
    fn fwrite(ptr: *const core::ffi::c_void, size: usize, count: usize, stream: *mut core::ffi::c_void) -> usize;
    fn fclose(stream: *mut core::ffi::c_void) -> i32;
    fn usleep(usec: u32) -> i32;
    fn gettimeofday(tv: *mut core::ffi::c_void, tz: *mut core::ffi::c_void) -> i32;
    fn pthread_create(
        thread: *mut usize,
        attr: *const core::ffi::c_void,
        start: unsafe extern "C" fn(*mut core::ffi::c_void) -> *mut core::ffi::c_void,
        arg: *mut core::ffi::c_void,
    ) -> i32;
    fn pthread_detach(thread: usize) -> i32;
}

struct Jit {
    jit: JITModule,
    ids: HashMap<glcc_mir::ir::Func, cranelift_module::FuncId>,
    module: Module,
    runtime: Module,
    runtime_ids: HashMap<glcc_mir::ir::Func, cranelift_module::FuncId>,
}

fn build(src: &str) -> Jit {
    let module = fern::compile(src).unwrap_or_else(|d| panic!("compile failed: {}\nin {src}", d.msg));
    let errors = glcc_mir::verify::verify_module(&module);
    assert!(errors.is_empty(), "mir errors in {src}: {errors:?}");
    let mut builder = JITBuilder::with_flags(&glcc_lower_cl::codegen_flags(), cranelift_module::default_libcall_names()).expect("jit builder");
    builder.symbol("memcpy", memcpy as *const u8);
    builder.symbol("memset", memset as *const u8);
    builder.symbol("write", write as *const u8);
    builder.symbol("read", read as *const u8);
    builder.symbol("abort", abort as *const u8);
    builder.symbol("malloc", malloc as *const u8);
    builder.symbol("realloc", realloc as *const u8);
    builder.symbol("free", free as *const u8);
    builder.symbol("fopen", fopen as *const u8);
    builder.symbol("fread", fread as *const u8);
    builder.symbol("fwrite", fwrite as *const u8);
    builder.symbol("fclose", fclose as *const u8);
    builder.symbol("usleep", usleep as *const u8);
    builder.symbol("gettimeofday", gettimeofday as *const u8);
    builder.symbol("pthread_create", pthread_create as *const u8);
    builder.symbol("pthread_detach", pthread_detach as *const u8);
    let mut jit = JITModule::new(builder);
    // The runtime goes in first, so the program's externs for
    // `fern_enter`/`fern_pin`/`fern_leave`/`fern_alloc` bind to its
    // definitions.
    let runtime = fern::runtime::module();
    let runtime_ids = glcc_lower_cl::lower_module(&runtime, &mut jit, glcc_lower_cl::Target::native());
    let ids = glcc_lower_cl::lower_module(&module, &mut jit, glcc_lower_cl::Target::native());
    jit.finalize_definitions().expect("jit finalize");
    Jit {
        jit,
        ids,
        module,
        runtime,
        runtime_ids,
    }
}

impl Jit {
    fn func(&self, name: &str) -> cranelift_module::FuncId {
        self.ids
            .iter()
            .find(|(id, _)| self.module.funcs[**id].name == name)
            .map(|(_, fid)| *fid)
            .unwrap_or_else(|| panic!("no function named `{name}`"))
    }

    /// A code pointer to a runtime function, e.g. `fern_stats_live`.
    fn rt(&self, name: &str) -> *const u8 {
        let id = self
            .runtime_ids
            .iter()
            .find(|(id, _)| self.runtime.funcs[**id].name == name)
            .map(|(_, fid)| *fid)
            .unwrap_or_else(|| panic!("no runtime function named `{name}`"));
        self.jit.get_finalized_function(id)
    }

    fn i64(&mut self, name: &str, args: &[i64]) -> i64 {
        let addr = self.jit.get_finalized_function(self.func(name));
        unsafe {
            match args {
                [] => std::mem::transmute::<_, extern "C" fn() -> i64>(addr)(),
                [a] => std::mem::transmute::<_, extern "C" fn(i64) -> i64>(addr)(*a),
                [a, b] => std::mem::transmute::<_, extern "C" fn(i64, i64) -> i64>(addr)(*a, *b),
                _ => panic!("test helper supports at most two arguments"),
            }
        }
    }

    fn bool(&mut self, name: &str, args: &[i64]) -> bool {
        let addr = self.jit.get_finalized_function(self.func(name));
        unsafe {
            match args {
                [] => std::mem::transmute::<_, extern "C" fn() -> bool>(addr)(),
                [a] => std::mem::transmute::<_, extern "C" fn(i64) -> bool>(addr)(*a),
                _ => panic!("test helper supports at most one argument"),
            }
        }
    }

    fn f64(&mut self, name: &str, args: &[f64]) -> f64 {
        let addr = self.jit.get_finalized_function(self.func(name));
        unsafe {
            match args {
                [] => std::mem::transmute::<_, extern "C" fn() -> f64>(addr)(),
                [a] => std::mem::transmute::<_, extern "C" fn(f64) -> f64>(addr)(*a),
                _ => panic!("test helper supports at most one argument"),
            }
        }
    }

    fn string(&mut self, name: &str) -> String {
        let addr = self.jit.get_finalized_function(self.func(name));
        unsafe {
            let f: extern "C" fn() -> *const core::ffi::c_char = std::mem::transmute(addr);
            let p = f();
            assert!(!p.is_null(), "`{name}` returned a null string");
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

fn run_i64(src: &str) -> i64 {
    build(src).i64("main", &[])
}

fn run_bool(src: &str) -> bool {
    build(src).bool("main", &[])
}

fn run_f64(src: &str) -> f64 {
    build(src).f64("main", &[])
}

#[test]
fn async_and_await_run_to_completion() {
    assert_eq!(run_i64("async fn inc(x) : Int -> Int = x + 1\nfn main() :  -> Int = block_on_int(inc(41))"), 42);
    assert_eq!(
        run_i64(
            "async fn inc(x) : Int -> Int = x + 1\n\
             async fn twice(x) : Int -> Int = { let a = inc(x).await; let b = inc(a).await; b }\n\
             fn main() :  -> Int = block_on_int(twice(1))"
        ),
        3
    );
}

#[test]
fn async_frames_survive_collection() {
    let src = "\
        async fn sum(n) : Int -> Int = if n == 0 then 0 else { let xs = [n]; n + sum(n - 1).await }\n\
        fn main() :  -> Int = block_on_int(sum(200))";
    assert_eq!(run_i64(src), 200 * 201 / 2);
}

#[test]
fn arithmetic() {
    assert_eq!(run_i64("fn main() :  -> Int = 2 + 3 * 4"), 14);
    assert_eq!(run_i64("fn main() :  -> Int = (10 - 3) * 2"), 14);
    assert_eq!(run_i64("fn main() :  -> Int = -5 + 12"), 7);
    assert_eq!(run_i64("fn main() :  -> Int = 17 / 5"), 3);
    assert_eq!(run_i64("fn main() :  -> Int = -17 / 5"), -3);
}

#[test]
fn conditionals() {
    assert_eq!(run_i64("fn abs(x) : Int -> Int = if x < 0 then -x else x\nfn main() :  -> Int = abs(-9)"), 9);
    assert_eq!(run_i64("fn max(x, y) : (Int, Int) -> Int = if x > y then x else y\nfn main() :  -> Int = max(3, 8)"), 8);
    assert_eq!(run_i64("fn main() :  -> Int = if true then 1 else 2"), 1);
}

#[test]
fn short_circuit() {
    assert_eq!(run_i64("fn main() :  -> Int = if 1 < 2 && 2 < 3 then 1 else 0"), 1);
    assert_eq!(run_i64("fn main() :  -> Int = if 1 < 2 || 2 < 1 then 1 else 0"), 1);
    assert_eq!(run_i64("fn main() :  -> Int = if 3 < 2 && 1 < 2 then 1 else 0"), 0);
    assert_eq!(run_i64("fn main() :  -> Int = if 3 < 2 || 2 < 1 then 1 else 0"), 0);
}

#[test]
fn recursion() {
    assert_eq!(run_i64("fn fac(n) : Int -> Int = if n == 0 then 1 else n * fac(n - 1)\nfn main() :  -> Int = fac(6)"), 720);
    assert_eq!(
        run_i64("fn fib(n) : Int -> Int = if n < 2 then n else fib(n - 1) + fib(n - 2)\nfn main() :  -> Int = fib(12)"),
        144
    );
}

#[test]
fn tail_calls_do_not_grow_the_stack() {
    assert_eq!(
        run_i64(
            "fn sum(n, acc) : (Int, Int) -> Int = if n == 0 then acc else sum(n - 1, acc + n)\n\
             fn main() :  -> Int = sum(1000000, 0)"
        ),
        500000500000
    );
    let src = "\
        fn count(xs, acc) : (List<Int>, Int) -> Int = match xs {\n\
            case Nil => acc\n\
            case Cons(_, rest) => count(rest, acc + 1)\n\
        }\n\
        fn build(n, acc) : (Int, List<Int>) -> List<Int> = if n == 0 then acc else build(n - 1, Cons(n, acc))\n\
        fn done(n) : Int -> Bool = n == 0 || done(n - 1)\n\
        fn main() :  -> Int = if done(200000) then count(build(20000, Nil), 0) else 0";
    assert_eq!(run_i64(src), 20000);
}

#[test]
fn lists() {
    assert_eq!(
        run_i64(
            "fn sum(xs) : List<Int> -> Int = match xs { case Nil => 0 case Cons(x, xs) => x + sum(xs) }\n\
             fn main() :  -> Int = sum([1, 2, 3])"
        ),
        6
    );
    assert_eq!(
        run_i64(
            "fn len(xs) : for<A> List<A> -> Int = match xs { case Nil => 0 case Cons(_, xs) => 1 + len(xs) }\n\
             fn main() :  -> Int = len(Cons(1, Cons(2, Nil)))"
        ),
        2
    );
    assert_eq!(
        run_i64(
            "fn head(xs) : List<Int> -> Int = match xs { case Nil => -1 case Cons(x, _) => x }\n\
             fn main() :  -> Int = head(Cons(4, Nil)) + head(Nil)"
        ),
        3
    );
}

#[test]
fn heap_keeps_up_with_allocations() {
    let src = "\
        fn range(n) : Int -> List<Int> = if n == 0 then Nil else Cons(n, range(n - 1))\n\
        fn sum(xs) : List<Int> -> Int = match xs { case Nil => 0 case Cons(x, xs) => x + sum(xs) }\n\
        fn main() :  -> Int = sum(range(500))";
    assert_eq!(run_i64(src), 500 * 501 / 2);
}

#[test]
fn tuples_and_patterns() {
    assert_eq!(run_i64("fn fst(p) : for<A, B> (A, B) -> A = match p { case (a, _) => a }\nfn main() :  -> Int = fst((7, 9))"), 7);
    assert_eq!(
        run_i64(
            "fn fst(p) : for<A, B> (A, B) -> A = match p { case (a, _) => a }\n\
             fn swap(p) : for<A, B> (A, B) -> (B, A) = match p { case (a, b) => (b, a) }\n\
             fn main() :  -> Int = fst(swap((1, 2)))"
        ),
        2
    );
    assert_eq!(run_i64("fn main() :  -> Int = { let (a, b) = (3, 4); a * b }"), 12);
    assert_eq!(run_i64("fn main() :  -> Int = { let u = {}; 1 }"), 1);
}

#[test]
fn parameter_patterns() {
    assert_eq!(run_i64("fn uncurry((a, b)) : (Int, Int) -> Int = a * b\nfn main() :  -> Int = uncurry((6, 7))"), 42);
    assert_eq!(
        run_i64(
            "fn swap((a, b)) : (Int, Int) -> (Int, Int) = (b, a)\n\
             fn fst(p) : for<A, B> (A, B) -> A = match p { case (a, _) => a }\n\
             fn main() :  -> Int = fst(swap((1, 2)))"
        ),
        2
    );
}

#[test]
fn user_data_types() {
    let src = "\
        type Shape = | Circle(Int) | Rect(Int, Int) | Point\n\
        fn area(s) : Shape -> Int = match s {\n\
            case Circle(r) => r * r * 3\n\
            case Rect(w, h) => w * h\n\
            case Point => 0\n\
        }\n\
        fn main() :  -> Int = area(Circle(2)) + area(Rect(3, 4)) + area(Point)\n\
            + match Point { case Point => 1 }";
    assert_eq!(run_i64(src), 25);
}

#[test]
fn record_fields_and_reordering() {
    let src = "\
        type Point = { x: Int, y: Int }\n\
        type Pair<A> = { first: A, second: A }\n\
        fn sum(p) : Point -> Int = p.x + p.y\n\
        fn swap(p) : for<A> Pair<A> -> Pair<A> = Pair { first = p.second, second = p.first }\n\
        fn main() :  -> Int = sum(Point { y = 4, x = 1 }) + swap(Pair { first = 2, second = 3 }).first";
    assert_eq!(run_i64(src), 8);
}

#[test]
fn try_maybe_returns_the_residual() {
    let src = "\
        fn bump(m) : Maybe<Int> -> Maybe<Int> = { let x = m?; Just(x + 1) }\n\
        fn label(m) : Maybe<Int> -> Maybe<String> = { let _ = m?; Just(\"ok\") }\n\
        fn unwrap(m, fallback) : (Maybe<Int>, Int) -> Int = match m { case Nothing => fallback case Just(x) => x }\n\
        fn main() :  -> Int = unwrap(bump(Just(4)), 0) * 10\n\
            + unwrap(bump(Nothing), 7) + match label(Nothing) { case Nothing => 1 case Just(_) => 0 }";
    assert_eq!(run_i64(src), 58);
}

#[test]
fn try_returns_from_the_innermost_lambda() {
    let src = "\
        fn outer(m) : Maybe<Int> -> Maybe<Int> = {\n\
            let f = |unused| { let x = m?; Just(x + 10) };\n\
            match f(0) { case Nothing => Just(99) case Just(x) => Just(x) }\n\
        }\n\
        fn unwrap(m, fallback) : (Maybe<Int>, Int) -> Int = match m { case Nothing => fallback case Just(x) => x }\n\
        fn main() :  -> Int = unwrap(outer(Just(1)), 0) * 10 + unwrap(outer(Nothing), 0)";
    assert_eq!(run_i64(src), 209);
}

#[test]
fn boolean_not() {
    assert!(run_bool("fn main() :  -> Bool = not(false)"));
    assert!(!run_bool("fn main() :  -> Bool = not(1 < 2)"));
    assert!(run_bool("fn main() :  -> Bool = not(eq([1], [2]))"));
}

#[test]
fn self_reference_as_a_value() {
    // Both bodies present themselves as values while being lowered;
    // their thunks must not collide on a placeholder symbol. `a` is the
    // identity, `b` is constantly zero, so a mix-up changes the answer.
    let src = "\
        fn twice(f, x) : for<A> (A -> A, A) -> A = f(f(x))\n\
        fn a(n) : Int -> Int = if n == 0 then 0 else twice(a, n - 1) + 1\n\
        fn b(n) : Int -> Int = if n == 0 then 0 else twice(b, n - 1) * 2\n\
        fn main() :  -> Int = a(3) * 100 + b(999)";
    assert_eq!(run_i64(src), 300);
}

#[test]
fn literal_patterns_and_guards() {
    let sign = "fn sign(x) : Int -> Int = match x { case 0 => 0 case n when n < 0 => -1 case _ => 1 }\n";
    assert_eq!(run_i64(&format!("{sign}fn main() :  -> Int = sign(-4)")), -1);
    assert_eq!(run_i64(&format!("{sign}fn main() :  -> Int = sign(0)")), 0);
    assert_eq!(run_i64(&format!("{sign}fn main() :  -> Int = sign(9)")), 1);
    assert_eq!(
        run_i64("fn f(b) : Bool -> Int = match b { case true => 1 case false => 0 }\nfn main() :  -> Int = f(false)"),
        0
    );
}

#[test]
fn higher_order_functions() {
    assert_eq!(
        run_i64("fn apply(f, x) : for<A, B> (A -> B, A) -> B = f(x)\nfn adder(n) : Int -> Int -> Int = y -> y + n\nfn main() :  -> Int = apply(adder(3), 4)"),
        7
    );
    assert_eq!(
        run_i64("fn twice(f, x) : for<A> (A -> A, A) -> A = f(f(x))\nfn inc(x) : Int -> Int = x + 1\nfn main() :  -> Int = twice(inc, 10)"),
        12
    );
    assert_eq!(
        run_i64(
            "fn compose(f, g) : for<A, B, C> (B -> C, A -> B) -> A -> C = x -> f(g(x))\n\
             fn main() :  -> Int = { let h = compose(|y| y + 1, |y| y * 2); h(5) }"
        ),
        11
    );
}

#[test]
fn closures_escape_their_frame() {
    let src = "\
        fn make(n) : Int -> Int -> Int = { let f = |x| x + n; let g = |y| f(y) + 1; g }\n\
        fn main() :  -> Int = { let h = make(10); h(5) }";
    assert_eq!(run_i64(src), 16);
}

#[test]
fn boxing_closures_survive() {
    let src = "\
        fn make(n) : Int -> Int -> Int = { let f = |x| x + n; let g = |y| f(y) + 1; g }\n\
        fn apply(f, x) : for<A, B> (A -> B, A) -> B = f(x)\n\
        fn main() :  -> Int = apply(make(2), 3) + apply(make(4), 5)";
    assert_eq!(run_i64(src), 6 + 10);
}

#[test]
fn class_dispatch() {
    assert!(run_bool("fn main() :  -> Bool = eq([1, 2], [1, 2])"));
    assert!(!run_bool("fn main() :  -> Bool = eq([1, 2], [1, 3])"));
    assert!(run_bool("fn main() :  -> Bool = eq(Just([1]), Just([1]))"));
    assert!(!run_bool("fn main() :  -> Bool = eq(Nothing, Just(1))"));
    assert!(run_bool("fn main() :  -> Bool = eq(1.5, 1.5) && eq(true, true)"));
}

#[test]
fn hkt_map_over_list() {
    let src = "\
        class Functor<F> { map: for<A, B> (A -> B) -> F<A> -> F<B> }\n\
        instance Functor<List> {\n\
            map(f, xs) = match xs { case Nil => Nil case Cons(x, xs) => Cons(f(x), map(f, xs)) }\n\
        }\n\
        fn sum(xs) : List<Int> -> Int = match xs { case Nil => 0 case Cons(x, xs) => x + sum(xs) }\n\
        fn main() :  -> Int = sum(map(|x| x * 2, [1, 2, 3]))";
    assert_eq!(run_i64(src), 12);
}

#[test]
fn hkt_map_over_maybe() {
    let src = "\
        class Functor<F> { map: for<A, B> (A -> B) -> F<A> -> F<B> }\n\
        instance Functor<Maybe> {\n\
            map(f, m) = match m { case Nothing => Nothing case Just(x) => Just(f(x)) }\n\
        }\n\
        fn get(m) : Maybe<Int> -> Int = match m { case Nothing => -1 case Just(x) => x }\n\
        fn main() :  -> Int = get(map(|x| x + 1, Just(4))) + get(map(|x| x + 1, Nothing))";
    assert_eq!(run_i64(src), 5 - 1);
}

#[test]
fn multi_parameter_classes() {
    let src = "\
        class Convert<A, B> { convert: A -> B }\n\
        instance Convert<Int, Int> { convert(x) = x + 10 }\n\
        instance Convert<Float, Int> { convert(x) = 7 }\n\
        fn main() :  -> Int = convert(1) + convert(1.5)";
    assert_eq!(run_i64(src), 18);
}

#[test]
fn floats() {
    assert_eq!(run_f64("fn main() :  -> Float = 1.5 + 2.25"), 3.75);
    assert!(run_bool("fn main() :  -> Bool = 1.5 < 2.0"));
    assert!(!run_bool("fn main() :  -> Bool = 2.5 < 2.0"));
    assert!(!run_bool("fn main() :  -> Bool = 1.5 == 1.25"));
    assert_eq!(
        run_i64(
            "fn near(x) : Float -> Int = if x > 25.0 then 1 else 0\n\
             fn main() :  -> Int = near(3.0 * 3.0 + 4.0 * 4.0) + near(3.0 * 3.0 + 5.0 * 5.0)"
        ),
        1
    );
}

#[test]
fn top_level_values() {
    let src = "\
        let two = 2\n\
        let inc = |x| x + two\n\
        fn main() :  -> Int = inc(40)";
    assert_eq!(run_i64(src), 42);
}

// ------------------------------------------------------------- collector --

/// The runtime's allocation accounting: live payload bytes, heap bytes
/// touched, and collections so far.
fn stats(jit: &Jit) -> (i64, i64, i64) {
    unsafe {
        let live: extern "C" fn() -> i64 = std::mem::transmute(jit.rt("fern_stats_live"));
        let used: extern "C" fn() -> i64 = std::mem::transmute(jit.rt("fern_stats_used"));
        let collections: extern "C" fn() -> i64 = std::mem::transmute(jit.rt("fern_stats_collections"));
        (live(), used(), collections())
    }
}

fn collect_now(jit: &Jit) {
    unsafe {
        let collect: extern "C" fn() = std::mem::transmute(jit.rt("fern_collect"));
        collect();
    }
}

/// Drive the allocator directly: a million blocks with one live root must
/// recycle, keep the root's payload intact, and never grow past its
/// headroom.
#[test]
fn allocator_recycles_garbage() {
    let jit = build("fn main() :  -> Int = 0");
    type Alloc = unsafe extern "C" fn(i64) -> *mut u8;
    type Pin = unsafe extern "C" fn(i64, i64, *mut u8);
    type Enter = unsafe extern "C" fn() -> i64;
    type Leave = unsafe extern "C" fn(i64);
    let alloc: Alloc = unsafe { std::mem::transmute(jit.rt("fern_alloc")) };
    let pin: Pin = unsafe { std::mem::transmute(jit.rt("fern_pin")) };
    let enter: Enter = unsafe { std::mem::transmute(jit.rt("fern_enter")) };
    let leave: Leave = unsafe { std::mem::transmute(jit.rt("fern_leave")) };

    unsafe {
        let frame = enter();
        let mut keep = alloc(8);
        *(keep as *mut i64) = 0x1234_5678;
        for i in 0..1_000_000i64 {
            pin(frame, 0, keep);
            let block = alloc(48);
            *(block as *mut i64) = i;
            keep = block;
        }
        assert_eq!(*(keep as *mut i64), 999_999, "the live block was reused");
        collect_now(&jit);
        let (live, used, collections) = stats(&jit);
        assert!(live <= 4096, "live bytes {live} after collecting one root");
        assert!(used < 20 << 20, "heap grew to {used} bytes");
        assert!(collections > 4, "expected collections, saw {collections}");
        leave(frame);
    }
}

/// A linked chain with only its head pinned: a collection must follow the
/// payload pointers and keep every node.
#[test]
fn marking_follows_payloads() {
    let jit = build("fn main() :  -> Int = 0");
    type Alloc = unsafe extern "C" fn(i64) -> *mut u8;
    type Pin = unsafe extern "C" fn(i64, i64, *mut u8);
    type Enter = unsafe extern "C" fn() -> i64;
    type Leave = unsafe extern "C" fn(i64);
    let alloc: Alloc = unsafe { std::mem::transmute(jit.rt("fern_alloc")) };
    let pin: Pin = unsafe { std::mem::transmute(jit.rt("fern_pin")) };
    let enter: Enter = unsafe { std::mem::transmute(jit.rt("fern_enter")) };
    let leave: Leave = unsafe { std::mem::transmute(jit.rt("fern_leave")) };

    unsafe {
        let frame = enter();
        let mut head = alloc(16);
        *(head as *mut *mut u8) = std::ptr::null_mut();
        for _ in 0..100_000 {
            pin(frame, 0, head);
            let node = alloc(16);
            *(node as *mut *mut u8) = head;
            head = node;
        }
        collect_now(&jit);
        let mut len = 0i64;
        let mut at = head;
        while !at.is_null() {
            len += 1;
            at = *(at as *mut *mut u8);
        }
        assert_eq!(len, 100_001, "chain broken by collection");
        leave(frame);
    }
}

/// A program whose live list must survive collections triggered by the
/// garbage it allocates on the way: if a frame's root were missing, the
/// list would be swept out from under it.
#[test]
fn live_values_survive_collections() {
    let src = "\
        fn churn(n) : Int -> Int = if n == 0 then 0 else { let (a, _) = (n, n); a + churn(n - 1) }\n\
        fn keep(xs, n) : (List<Int>, Int) -> List<Int> = if n == 0 then xs else { let dead = churn(2000); keep(xs, n - 1) }\n\
        fn sum(xs) : List<Int> -> Int = match xs { case Nil => 0 case Cons(x, xs) => x + sum(xs) }\n\
        fn main() :  -> Int = sum(keep([1, 2, 3], 200))";
    let mut jit = build(src);
    assert_eq!(jit.i64("main", &[]), 6);
    let (live, used, collections) = stats(&jit);
    assert!(collections > 0, "the program never triggered a collection");
    assert!(used < 24 << 20, "heap grew to {used} bytes");
    assert!(live > 0, "nothing was live when main returned");
    // Every frame is left, so the program's garbage is all reclaimable.
    collect_now(&jit);
    let (live, _, _) = stats(&jit);
    assert!(live <= 4096, "live bytes {live} after the program returned");
}

#[test]
fn value_pipe_applies_functions() {
    let src = "fn inc(x) : Int -> Int = x + 1\nfn main() :  -> Int = ((5 |> inc |> inc) * 10) + (1 |> inc)";
    assert_eq!(run_i64(src), 72);
    let src = "fn inc(x) : Int -> Int = x + 1\nfn main() :  -> Bool = 1 |> inc == 2";
    assert!(build(src).bool("main", &[]));
    let src = "fn inc(x) : Int -> Int = x + 1\nfn main() :  -> Int = 10 - 2 |> inc";
    assert_eq!(run_i64(src), 9);
}

#[test]
fn hash_builtins_resolve_to_prelude_functions() {
    assert!(build("fn main() :  -> Bool = #env(\"HOME\") == getenv(\"HOME\")").bool("main", &[]));
    assert_eq!(build("fn main() :  -> String = #str_of_int(7)").string("main"), "7");
}

#[test]
fn strings_concatenate_and_compare() {
    assert_eq!(build("fn main() :  -> String = \"hello\"").string("main"), "hello");
    assert_eq!(build("fn main() :  -> String = concat(\"foo\", \"bar\")").string("main"), "foobar");
    assert_eq!(build("fn main() :  -> String = str_of_int(-1234)").string("main"), "-1234");
    assert!(build("fn main() :  -> Bool = \"abc\" == \"abc\"").bool("main", &[]));
    assert!(!build("fn main() :  -> Bool = \"abc\" == \"abd\"").bool("main", &[]));
    assert!(build("fn main() :  -> Bool = \"abc\" != \"abd\"").bool("main", &[]));
    assert!(build("fn main() :  -> Bool = Eq::eq([\"a\", \"b\"], [\"a\", \"b\"])").bool("main", &[]));
    assert!(!build("fn main() :  -> Bool = Eq::eq([\"a\", \"b\"], [\"a\", \"c\"])").bool("main", &[]));
}

#[test]
fn byte_strings_and_file_loading() {
    assert_eq!(build("fn main() :  -> String = str_of_byte(65)").string("main"), "A");
    let path = std::env::temp_dir().join(format!("fern-read-file-{}", std::process::id()));
    std::fs::write(&path, "hello\n").expect("write file");
    let src = format!("fn main() :  -> String = read_file(\"{}\")", path.display());
    assert_eq!(build(&src).string("main"), "hello\n");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn string_search_split_and_join() {
    assert_eq!(build("fn main() :  -> String = str_slice(\"abcdef\", 1, 3)").string("main"), "bcd");
    assert_eq!(build("fn main() :  -> String = str_join(\"|\", str_split(\",\", \"a,b,c\"))").string("main"), "a|b|c");
    assert_eq!(build("fn main() :  -> String = str_join(\",\", str_lines(\"a\\nb\\nc\"))").string("main"), "a,b,c");
    assert!(build("fn main() :  -> Bool = str_contains(\"bc\", \"abcd\")").bool("main", &[]));
    assert!(!build("fn main() :  -> Bool = str_contains(\"zz\", \"abcd\")").bool("main", &[]));
    assert_eq!(build("fn main() :  -> String = str_of_int(str_find(\"bc\", \"abcd\"))").string("main"), "1");
    assert!(build("fn main() :  -> Bool = ok(0)").bool("main", &[]));
    assert!(!build("fn main() :  -> Bool = ok(1)").bool("main", &[]));
}

#[test]
fn externs_work_as_values() {
    let src = "\
        fn apply(f, x, y) : for<A, B, C> (A -> B -> C, A, B) -> C = f(x, y)\n\
        fn main() :  -> Int = if apply(str_eq, \"same\", \"same\") then 1 else 0";
    assert_eq!(run_i64(src), 1);
}

#[test]
fn spawn_status_and_capture() {
    assert_eq!(run_i64("fn main() :  -> Int = spawn_status(\"true\", [])"), 0);
    assert_eq!(run_i64("fn main() :  -> Int = spawn_status(\"false\", [])"), 1);
    assert_eq!(build("fn main() :  -> String = spawn_capture(\"echo\", [\"hi\"])").string("main"), "hi\n");
    assert_eq!(run_i64("fn main() :  -> Int = if spawn(\"true\", []) then 1 else 0"), 1);
}

#[test]
fn shell_commands_and_captures() {
    assert_eq!(run_i64("fn main() :  -> Int = if $ true then 1 else 0"), 1);
    assert_eq!(run_i64("fn main() :  -> Int = if $ false then 1 else 0"), 0);
    assert_eq!(build("fn main() :  -> String = $(echo hi)").string("main"), "hi\n");
    assert_eq!(build("fn main() :  -> String = $(printf \"[%s]\" ${\"a b\"})").string("main"), "[a b]");
    assert_eq!(build("fn main() :  -> String = $(printf \"%s\" ${concat(\"x\", \"y\")})").string("main"), "xy");
    assert_eq!(build("fn main() :  -> String = $(echo $(echo nested))").string("main"), "nested\n\n");
    assert_eq!(build("fn main() :  -> String = $(printf \"%s\" 'a b')").string("main"), "a b");
    assert_eq!(build("fn main() :  -> String = $(echo \"a|b\")").string("main"), "a|b\n");
}

#[test]
fn shell_pipelines() {
    assert_eq!(run_i64("fn main() :  -> Int = if $ true | false then 1 else 0"), 0);
    assert_eq!(run_i64("fn main() :  -> Int = if $ false | true then 1 else 0"), 1);
    assert_eq!(build("fn main() :  -> String = $(echo hi | tr a-z A-Z)").string("main"), "HI\n");
    assert!(build("fn main() :  -> Bool = str_contains(\"3\", $(printf \"a\\nb\\nc\\n\" | wc -l))").bool("main", &[]));
}

#[test]
fn shell_builtins_and_bool_composition() {
    assert!(build("fn main() :  -> Bool = ($ #env HOME) == getenv(\"HOME\")").bool("main", &[]));
    assert!(build("fn main() :  -> Bool = #env(\"HOME\") == getenv(\"HOME\")").bool("main", &[]));
    assert!(build("fn main() :  -> Bool = $ true && $ true").bool("main", &[]));
    assert!(!build("fn main() :  -> Bool = $ true && $ false").bool("main", &[]));
    assert!(build("fn main() :  -> Bool = $ false || $ true").bool("main", &[]));
    assert_eq!(build("fn main() :  -> Int = $(printf hi) |> str_len").i64("main", &[]), 2);
    assert_eq!(run_i64("fn main() :  -> Int = if $ true && $ true then 1 else 0"), 1);
    assert_eq!(run_i64("fn main() :  -> Int = if $ false then 0 else 1"), 1);
    assert_eq!(run_i64("fn main() :  -> Int = if $ true | false |> not then 1 else 0"), 1);
    assert_eq!(run_i64("fn main() :  -> Int = { let ok = $ true; if ok then 1 else 0 }"), 1);
}

#[test]
fn shell_rejects_bare_env_vars_and_builtin_pipelines() {
    let err = fern::compile("fn main() :  -> Int = $(echo $HOME)").expect_err("rejects");
    assert!(err.msg.contains("${expr}"), "{:?}", err.msg);
    let err = fern::compile("fn main() :  -> Int = $ #env HOME | cat").expect_err("rejects");
    assert!(err.msg.contains("pipeline stage"), "{:?}", err.msg);
    let err = fern::compile("fn main() :  -> Int = $(echo ${1})").expect_err("rejects");
    assert!(err.msg.contains("expected `String`"), "{:?}", err.msg);
}
