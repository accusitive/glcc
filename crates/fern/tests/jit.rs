//! End-to-end fern tests: source -> MIR -> Cranelift -> JIT -> run.
//!
//! Also mirror the C driver's pass pipeline when optimizing, and lower the
//! C runtime into the same JIT module, so the program's `fern_alloc` and
//! frame calls bind to real code. Run with `cargo test -p fern`.

// These execute generated code in-process, so they require a host whose C
// ABI matches the SystemV convention the backend emits. Windows uses the
// Win64 convention (not implemented yet), so skip there until the Windows
// target lands.
#![cfg(not(windows))]

use std::collections::HashMap;

use cranelift_jit::{JITBuilder, JITModule};
use glcc_mir::ir::Module;

unsafe extern "C" {
    fn memcpy(dst: *mut core::ffi::c_void, src: *const core::ffi::c_void, n: usize) -> *mut core::ffi::c_void;
    fn memset(dst: *mut core::ffi::c_void, c: i32, n: usize) -> *mut core::ffi::c_void;
    fn write(fd: i32, buf: *const core::ffi::c_void, n: usize) -> isize;
    fn abort();
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
    let mut builder = JITBuilder::with_flags(
        &[("preserve_frame_pointers", "true")],
        cranelift_module::default_libcall_names(),
    )
    .expect("jit builder");
    builder.symbol("memcpy", memcpy as *const u8);
    builder.symbol("memset", memset as *const u8);
    builder.symbol("write", write as *const u8);
    builder.symbol("abort", abort as *const u8);
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
fn arithmetic() {
    assert_eq!(run_i64("fun main() = 2 + 3 * 4"), 14);
    assert_eq!(run_i64("fun main() = (10 - 3) * 2"), 14);
    assert_eq!(run_i64("fun main() = -5 + 12"), 7);
    assert_eq!(run_i64("fun main() = 17 / 5"), 3);
    assert_eq!(run_i64("fun main() = -17 / 5"), -3);
}

#[test]
fn conditionals() {
    assert_eq!(
        run_i64("fun abs(x) = if x < 0 then -x else x\nfun main() = abs(-9)"),
        9
    );
    assert_eq!(
        run_i64("fun max(x, y) = if x > y then x else y\nfun main() = max(3, 8)"),
        8
    );
    assert_eq!(run_i64("fun main() = if true then 1 else 2"), 1);
}

#[test]
fn short_circuit() {
    assert_eq!(run_i64("fun main() = if 1 < 2 && 2 < 3 then 1 else 0"), 1);
    assert_eq!(run_i64("fun main() = if 1 < 2 || 2 < 1 then 1 else 0"), 1);
    assert_eq!(run_i64("fun main() = if 3 < 2 && 1 < 2 then 1 else 0"), 0);
    assert_eq!(run_i64("fun main() = if 3 < 2 || 2 < 1 then 1 else 0"), 0);
}

#[test]
fn recursion() {
    assert_eq!(
        run_i64("fun fac(n) = if n == 0 then 1 else n * fac(n - 1)\nfun main() = fac(6)"),
        720
    );
    assert_eq!(
        run_i64(
            "fun fib(n) = if n < 2 then n else fib(n - 1) + fib(n - 2)\nfun main() = fib(12)"
        ),
        144
    );
}

#[test]
fn lists() {
    assert_eq!(
        run_i64(
            "fun sum(xs) = match xs { case Nil => 0 case Cons(x, xs) => x + sum(xs) }\n\
             fun main() = sum([1, 2, 3])"
        ),
        6
    );
    assert_eq!(
        run_i64(
            "fun len(xs) = match xs { case Nil => 0 case Cons(_, xs) => 1 + len(xs) }\n\
             fun main() = len(Cons(1, Cons(2, Nil)))"
        ),
        2
    );
    assert_eq!(
        run_i64(
            "fun head(xs) = match xs { case Nil => -1 case Cons(x, _) => x }\n\
             fun main() = head(Cons(4, Nil)) + head(Nil)"
        ),
        3
    );
}

#[test]
fn heap_keeps_up_with_allocations() {
    let src = "\
        fun range(n) = if n == 0 then Nil else Cons(n, range(n - 1))\n\
        fun sum(xs) = match xs { case Nil => 0 case Cons(x, xs) => x + sum(xs) }\n\
        fun main() = sum(range(500))";
    assert_eq!(run_i64(src), 500 * 501 / 2);
}

#[test]
fn tuples_and_patterns() {
    assert_eq!(
        run_i64("fun fst(p) = match p { case (a, _) => a }\nfun main() = fst((7, 9))"),
        7
    );
    assert_eq!(
        run_i64(
            "fun fst(p) = match p { case (a, _) => a }\n\
             fun swap(p) = match p { case (a, b) => (b, a) }\n\
             fun main() = fst(swap((1, 2)))"
        ),
        2
    );
    assert_eq!(
        run_i64("fun main() = { let (a, b) = (3, 4); a * b }"),
        12
    );
    assert_eq!(run_i64("fun main() = { let u = {}; 1 }"), 1);
}

#[test]
fn parameter_patterns() {
    assert_eq!(
        run_i64("fun uncurry((a, b)) = a * b\nfun main() = uncurry((6, 7))"),
        42
    );
    assert_eq!(
        run_i64(
            "fun swap((a, b)) = (b, a)\n\
             fun fst(p) = match p { case (a, _) => a }\n\
             fun main() = fst(swap((1, 2)))"
        ),
        2
    );
}

#[test]
fn user_data_types() {
    let src = "\
        data Shape = Circle(Int) | Rect(Int, Int) | Point\n\
        fun area(s) = match s {\n\
            case Circle(r) => r * r * 3\n\
            case Rect(w, h) => w * h\n\
            case Point => 0\n\
        }\n\
        fun main() = area(Circle(2)) + area(Rect(3, 4)) + area(Point)\n\
            + match Point { case Point => 1 }";
    assert_eq!(run_i64(src), 25);
}

#[test]
fn boolean_not() {
    assert!(run_bool("fun main() = not(false)"));
    assert!(!run_bool("fun main() = not(1 < 2)"));
    assert!(run_bool("fun main() = not(eq([1], [2]))"));
}

#[test]
fn self_reference_as_a_value() {
    // Both bodies present themselves as values while being lowered;
    // their thunks must not collide on a placeholder symbol. `a` is the
    // identity, `b` is constantly zero, so a mix-up changes the answer.
    let src = "\
        fun twice(f, x) = f(f(x))\n\
        fun a(n) = if n == 0 then 0 else twice(a, n - 1) + 1\n\
        fun b(n) = if n == 0 then 0 else twice(b, n - 1) * 2\n\
        fun main() = a(3) * 100 + b(999)";
    assert_eq!(run_i64(src), 300);
}

#[test]
fn literal_patterns_and_guards() {    let sign = "fun sign(x) = match x { case 0 => 0 case n when n < 0 => -1 case _ => 1 }\n";
    assert_eq!(run_i64(&format!("{sign}fun main() = sign(-4)")), -1);
    assert_eq!(run_i64(&format!("{sign}fun main() = sign(0)")), 0);
    assert_eq!(run_i64(&format!("{sign}fun main() = sign(9)")), 1);
    assert_eq!(
        run_i64("fun f(b) = match b { case true => 1 case false => 0 }\nfun main() = f(false)"),
        0
    );
}

#[test]
fn higher_order_functions() {
    assert_eq!(
        run_i64(
            "fun apply(f, x) = f(x)\nfun adder(n) = y -> y + n\nfun main() = apply(adder(3), 4)"
        ),
        7
    );
    assert_eq!(
        run_i64(
            "fun twice(f, x) = f(f(x))\nfun inc(x) = x + 1\nfun main() = twice(inc, 10)"
        ),
        12
    );
    assert_eq!(
        run_i64(
            "fun compose(f, g) = x -> f(g(x))\n\
             fun main() = { let h = compose(y -> y + 1, y -> y * 2); h(5) }"
        ),
        11
    );
}

#[test]
fn closures_escape_their_frame() {
    let src = "\
        fun make(n) = { let f = x -> x + n; let g = y -> f(y) + 1; g }\n\
        fun main() = { let h = make(10); h(5) }";
    assert_eq!(run_i64(src), 16);
}

#[test]
fn boxing_closures_survive() {
    let src = "\
        fun make(n) = { let f = x -> x + n; let g = y -> f(y) + 1; g }\n\
        fun apply(f, x) = f(x)\n\
        fun main() = apply(make(2), 3) + apply(make(4), 5)";
    assert_eq!(run_i64(src), 6 + 10);
}

#[test]
fn class_dispatch() {
    assert!(run_bool("fun main() = eq([1, 2], [1, 2])"));
    assert!(!run_bool("fun main() = eq([1, 2], [1, 3])"));
    assert!(run_bool("fun main() = eq(Just([1]), Just([1]))"));
    assert!(!run_bool("fun main() = eq(Nothing, Just(1))"));
    assert!(run_bool("fun main() = eq(1.5, 1.5) && eq(true, true)"));
}

#[test]
fn hkt_map_over_list() {
    let src = "\
        class Functor f { map : (a -> b) -> f a -> f b }\n\
        instance Functor List {\n\
            map(f, xs) = match xs { case Nil => Nil case Cons(x, xs) => Cons(f(x), map(f, xs)) }\n\
        }\n\
        fun sum(xs) = match xs { case Nil => 0 case Cons(x, xs) => x + sum(xs) }\n\
        fun main() = sum(map(x -> x * 2, [1, 2, 3]))";
    assert_eq!(run_i64(src), 12);
}

#[test]
fn hkt_map_over_maybe() {
    let src = "\
        class Functor f { map : (a -> b) -> f a -> f b }\n\
        instance Functor Maybe {\n\
            map(f, m) = match m { case Nothing => Nothing case Just(x) => Just(f(x)) }\n\
        }\n\
        fun get(m) = match m { case Nothing => -1 case Just(x) => x }\n\
        fun main() = get(map(x -> x + 1, Just(4))) + get(map(x -> x + 1, Nothing))";
    assert_eq!(run_i64(src), 5 - 1);
}

#[test]
fn multi_parameter_classes() {
    let src = "\
        class Convert a b { convert : a -> b }\n\
        instance Convert Int Int { convert(x) = x + 10 }\n\
        instance Convert Float Int { convert(x) = 7 }\n\
        fun main() = convert(1) + convert(1.5)";
    assert_eq!(run_i64(src), 18);
}

#[test]
fn floats() {
    assert_eq!(run_f64("fun main() = 1.5 + 2.25"), 3.75);
    assert!(run_bool("fun main() = 1.5 < 2.0"));
    assert!(!run_bool("fun main() = 2.5 < 2.0"));
    assert!(!run_bool("fun main() = 1.5 == 1.25"));
    assert_eq!(
        run_i64(
            "fun near(x) = if x > 25.0 then 1 else 0\n\
             fun main() = near(3.0 * 3.0 + 4.0 * 4.0) + near(3.0 * 3.0 + 5.0 * 5.0)"
        ),
        1
    );
}

#[test]
fn top_level_values() {
    let src = "\
        let two = 2\n\
        let inc = x -> x + two\n\
        fun main() = inc(40)";
    assert_eq!(run_i64(src), 42);
}

// ------------------------------------------------------------- collector --

/// The runtime's allocation accounting: live payload bytes, heap bytes
/// touched, and collections so far.
fn stats(jit: &Jit) -> (i64, i64, i64) {
    unsafe {
        let live: extern "C" fn() -> i64 = std::mem::transmute(jit.rt("fern_stats_live"));
        let used: extern "C" fn() -> i64 = std::mem::transmute(jit.rt("fern_stats_used"));
        let collections: extern "C" fn() -> i64 =
            std::mem::transmute(jit.rt("fern_stats_collections"));
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
    let jit = build("fun main() = 0");
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
    let jit = build("fun main() = 0");
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
        fun churn(n) = if n == 0 then 0 else { let (a, _) = (n, n); a + churn(n - 1) }\n\
        fun keep(xs, n) = if n == 0 then xs else { let dead = churn(2000); keep(xs, n - 1) }\n\
        fun sum(xs) = match xs { case Nil => 0 case Cons(x, xs) => x + sum(xs) }\n\
        fun main() = sum(keep([1, 2, 3], 200))";
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
