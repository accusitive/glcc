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
    let mut builder = JITBuilder::with_flags(&[("preserve_frame_pointers", "true")], cranelift_module::default_libcall_names()).expect("jit builder");
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
fn arithmetic() {
    assert_eq!(run_i64("fun main() : Int = 2 + 3 * 4"), 14);
    assert_eq!(run_i64("fun main() : Int = (10 - 3) * 2"), 14);
    assert_eq!(run_i64("fun main() : Int = -5 + 12"), 7);
    assert_eq!(run_i64("fun main() : Int = 17 / 5"), 3);
    assert_eq!(run_i64("fun main() : Int = -17 / 5"), -3);
}

#[test]
fn conditionals() {
    assert_eq!(run_i64("fun abs(x : Int) : Int = if x < 0 then -x else x\nfun main() : Int = abs(-9)"), 9);
    assert_eq!(run_i64("fun max(x : Int, y : Int) : Int = if x > y then x else y\nfun main() : Int = max(3, 8)"), 8);
    assert_eq!(run_i64("fun main() : Int = if true then 1 else 2"), 1);
}

#[test]
fn short_circuit() {
    assert_eq!(run_i64("fun main() : Int = if 1 < 2 && 2 < 3 then 1 else 0"), 1);
    assert_eq!(run_i64("fun main() : Int = if 1 < 2 || 2 < 1 then 1 else 0"), 1);
    assert_eq!(run_i64("fun main() : Int = if 3 < 2 && 1 < 2 then 1 else 0"), 0);
    assert_eq!(run_i64("fun main() : Int = if 3 < 2 || 2 < 1 then 1 else 0"), 0);
}

#[test]
fn recursion() {
    assert_eq!(run_i64("fun fac(n : Int) : Int = if n == 0 then 1 else n * fac(n - 1)\nfun main() : Int = fac(6)"), 720);
    assert_eq!(run_i64("fun fib(n : Int) : Int = if n < 2 then n else fib(n - 1) + fib(n - 2)\nfun main() : Int = fib(12)"), 144);
}

#[test]
fn lists() {
    assert_eq!(
        run_i64(
            "fun sum(xs : List Int) : Int = match xs { case Nil => 0 case Cons(x, xs) => x + sum(xs) }\n\
             fun main() : Int = sum([1, 2, 3])"
        ),
        6
    );
    assert_eq!(
        run_i64(
            "fun len(xs : List a) : Int = match xs { case Nil => 0 case Cons(_, xs) => 1 + len(xs) }\n\
             fun main() : Int = len(Cons(1, Cons(2, Nil)))"
        ),
        2
    );
    assert_eq!(
        run_i64(
            "fun head(xs : List Int) : Int = match xs { case Nil => -1 case Cons(x, _) => x }\n\
             fun main() : Int = head(Cons(4, Nil)) + head(Nil)"
        ),
        3
    );
}

#[test]
fn heap_keeps_up_with_allocations() {
    let src = "\
        fun range(n : Int) : List Int = if n == 0 then Nil else Cons(n, range(n - 1))\n\
        fun sum(xs : List Int) : Int = match xs { case Nil => 0 case Cons(x, xs) => x + sum(xs) }\n\
        fun main() : Int = sum(range(500))";
    assert_eq!(run_i64(src), 500 * 501 / 2);
}

#[test]
fn tuples_and_patterns() {
    assert_eq!(run_i64("fun fst(p : (a, b)) : a = match p { case (a, _) => a }\nfun main() : Int = fst((7, 9))"), 7);
    assert_eq!(
        run_i64(
            "fun fst(p : (a, b)) : a = match p { case (a, _) => a }\n\
             fun swap(p : (a, b)) : (b, a) = match p { case (a, b) => (b, a) }\n\
             fun main() : Int = fst(swap((1, 2)))"
        ),
        2
    );
    assert_eq!(run_i64("fun main() : Int = { let (a, b) = (3, 4); a * b }"), 12);
    assert_eq!(run_i64("fun main() : Int = { let u = {}; 1 }"), 1);
}

#[test]
fn parameter_patterns() {
    assert_eq!(run_i64("fun uncurry((a, b) : (Int, Int)) : Int = a * b\nfun main() : Int = uncurry((6, 7))"), 42);
    assert_eq!(
        run_i64(
            "fun swap((a, b) : (Int, Int)) : (Int, Int) = (b, a)\n\
             fun fst(p : (a, b)) : a = match p { case (a, _) => a }\n\
             fun main() : Int = fst(swap((1, 2)))"
        ),
        2
    );
}

#[test]
fn user_data_types() {
    let src = "\
        data Shape = Circle(Int) | Rect(Int, Int) | Point\n\
        fun area(s : Shape) : Int = match s {\n\
            case Circle(r) => r * r * 3\n\
            case Rect(w, h) => w * h\n\
            case Point => 0\n\
        }\n\
        fun main() : Int = area(Circle(2)) + area(Rect(3, 4)) + area(Point)\n\
            + match Point { case Point => 1 }";
    assert_eq!(run_i64(src), 25);
}

#[test]
fn boolean_not() {
    assert!(run_bool("fun main() : Bool = not(false)"));
    assert!(!run_bool("fun main() : Bool = not(1 < 2)"));
    assert!(run_bool("fun main() : Bool = not(eq([1], [2]))"));
}

#[test]
fn self_reference_as_a_value() {
    // Both bodies present themselves as values while being lowered;
    // their thunks must not collide on a placeholder symbol. `a` is the
    // identity, `b` is constantly zero, so a mix-up changes the answer.
    let src = "\
        fun twice(f : a -> a, x : a) : a = f(f(x))\n\
        fun a(n : Int) : Int = if n == 0 then 0 else twice(a, n - 1) + 1\n\
        fun b(n : Int) : Int = if n == 0 then 0 else twice(b, n - 1) * 2\n\
        fun main() : Int = a(3) * 100 + b(999)";
    assert_eq!(run_i64(src), 300);
}

#[test]
fn literal_patterns_and_guards() {
    let sign = "fun sign(x : Int) : Int = match x { case 0 => 0 case n when n < 0 => -1 case _ => 1 }\n";
    assert_eq!(run_i64(&format!("{sign}fun main() : Int = sign(-4)")), -1);
    assert_eq!(run_i64(&format!("{sign}fun main() : Int = sign(0)")), 0);
    assert_eq!(run_i64(&format!("{sign}fun main() : Int = sign(9)")), 1);
    assert_eq!(run_i64("fun f(b : Bool) : Int = match b { case true => 1 case false => 0 }\nfun main() : Int = f(false)"), 0);
}

#[test]
fn higher_order_functions() {
    assert_eq!(run_i64("fun apply(f : a -> b, x : a) : b = f(x)\nfun adder(n : Int) : Int -> Int = y -> y + n\nfun main() : Int = apply(adder(3), 4)"), 7);
    assert_eq!(run_i64("fun twice(f : a -> a, x : a) : a = f(f(x))\nfun inc(x : Int) : Int = x + 1\nfun main() : Int = twice(inc, 10)"), 12);
    assert_eq!(
        run_i64(
            "fun compose(f : b -> c, g : a -> b) : a -> c = x -> f(g(x))\n\
             fun main() : Int = { let h = compose(y -> y + 1, y -> y * 2); h(5) }"
        ),
        11
    );
}

#[test]
fn closures_escape_their_frame() {
    let src = "\
        fun make(n : Int) : Int -> Int = { let f = x -> x + n; let g = y -> f(y) + 1; g }\n\
        fun main() : Int = { let h = make(10); h(5) }";
    assert_eq!(run_i64(src), 16);
}

#[test]
fn boxing_closures_survive() {
    let src = "\
        fun make(n : Int) : Int -> Int = { let f = x -> x + n; let g = y -> f(y) + 1; g }\n\
        fun apply(f : a -> b, x : a) : b = f(x)\n\
        fun main() : Int = apply(make(2), 3) + apply(make(4), 5)";
    assert_eq!(run_i64(src), 6 + 10);
}

#[test]
fn class_dispatch() {
    assert!(run_bool("fun main() : Bool = eq([1, 2], [1, 2])"));
    assert!(!run_bool("fun main() : Bool = eq([1, 2], [1, 3])"));
    assert!(run_bool("fun main() : Bool = eq(Just([1]), Just([1]))"));
    assert!(!run_bool("fun main() : Bool = eq(Nothing, Just(1))"));
    assert!(run_bool("fun main() : Bool = eq(1.5, 1.5) && eq(true, true)"));
}

#[test]
fn hkt_map_over_list() {
    let src = "\
        class Functor f { map : (a -> b) -> f a -> f b }\n\
        instance Functor List {\n\
            map(f, xs) = match xs { case Nil => Nil case Cons(x, xs) => Cons(f(x), map(f, xs)) }\n\
        }\n\
        fun sum(xs : List Int) : Int = match xs { case Nil => 0 case Cons(x, xs) => x + sum(xs) }\n\
        fun main() : Int = sum(map(x -> x * 2, [1, 2, 3]))";
    assert_eq!(run_i64(src), 12);
}

#[test]
fn hkt_map_over_maybe() {
    let src = "\
        class Functor f { map : (a -> b) -> f a -> f b }\n\
        instance Functor Maybe {\n\
            map(f, m) = match m { case Nothing => Nothing case Just(x) => Just(f(x)) }\n\
        }\n\
        fun get(m : Maybe Int) : Int = match m { case Nothing => -1 case Just(x) => x }\n\
        fun main() : Int = get(map(x -> x + 1, Just(4))) + get(map(x -> x + 1, Nothing))";
    assert_eq!(run_i64(src), 5 - 1);
}

#[test]
fn multi_parameter_classes() {
    let src = "\
        class Convert a b { convert : a -> b }\n\
        instance Convert Int Int { convert(x) = x + 10 }\n\
        instance Convert Float Int { convert(x) = 7 }\n\
        fun main() : Int = convert(1) + convert(1.5)";
    assert_eq!(run_i64(src), 18);
}

#[test]
fn floats() {
    assert_eq!(run_f64("fun main() : Float = 1.5 + 2.25"), 3.75);
    assert!(run_bool("fun main() : Bool = 1.5 < 2.0"));
    assert!(!run_bool("fun main() : Bool = 2.5 < 2.0"));
    assert!(!run_bool("fun main() : Bool = 1.5 == 1.25"));
    assert_eq!(
        run_i64(
            "fun near(x : Float) : Int = if x > 25.0 then 1 else 0\n\
             fun main() : Int = near(3.0 * 3.0 + 4.0 * 4.0) + near(3.0 * 3.0 + 5.0 * 5.0)"
        ),
        1
    );
}

#[test]
fn top_level_values() {
    let src = "\
        let two = 2\n\
        let inc = x -> x + two\n\
        fun main() : Int = inc(40)";
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
    let jit = build("fun main() : Int = 0");
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
    let jit = build("fun main() : Int = 0");
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
        fun churn(n : Int) : Int = if n == 0 then 0 else { let (a, _) = (n, n); a + churn(n - 1) }\n\
        fun keep(xs : List Int, n : Int) : List Int = if n == 0 then xs else { let dead = churn(2000); keep(xs, n - 1) }\n\
        fun sum(xs : List Int) : Int = match xs { case Nil => 0 case Cons(x, xs) => x + sum(xs) }\n\
        fun main() : Int = sum(keep([1, 2, 3], 200))";
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
    let src = "fun inc(x : Int) : Int = x + 1\nfun main() : Int = ((5 |> inc |> inc) * 10) + (1 |> inc)";
    assert_eq!(run_i64(src), 72);
    let src = "fun inc(x : Int) : Int = x + 1\nfun main() : Bool = 1 |> inc == 2";
    assert!(build(src).bool("main", &[]));
    let src = "fun inc(x : Int) : Int = x + 1\nfun main() : Int = 10 - 2 |> inc";
    assert_eq!(run_i64(src), 9);
}

#[test]
fn hash_builtins_resolve_to_prelude_functions() {
    assert!(build("fun main() : Bool = #env(\"HOME\") == getenv(\"HOME\")").bool("main", &[]));
    assert_eq!(build("fun main() : String = #str_of_int(7)").string("main"), "7");
}

#[test]
fn strings_concatenate_and_compare() {
    assert_eq!(build("fun main() : String = \"hello\"").string("main"), "hello");
    assert_eq!(
        build("fun main() : String = concat(\"foo\", \"bar\")").string("main"),
        "foobar"
    );
    assert_eq!(
        build("fun main() : String = str_of_int(-1234)").string("main"),
        "-1234"
    );
    assert!(build("fun main() : Bool = \"abc\" == \"abc\"").bool("main", &[]));
    assert!(!build("fun main() : Bool = \"abc\" == \"abd\"").bool("main", &[]));
    assert!(build("fun main() : Bool = \"abc\" != \"abd\"").bool("main", &[]));
    assert!(build("fun main() : Bool = Eq::eq([\"a\", \"b\"], [\"a\", \"b\"])").bool("main", &[]));
    assert!(!build("fun main() : Bool = Eq::eq([\"a\", \"b\"], [\"a\", \"c\"])").bool("main", &[]));
}

#[test]
fn byte_strings_and_file_loading() {
    assert_eq!(build("fun main() : String = str_of_byte(65)").string("main"), "A");
    let path = std::env::temp_dir().join(format!("fern-read-file-{}", std::process::id()));
    std::fs::write(&path, "hello\n").expect("write file");
    let src = format!("fun main() : String = read_file(\"{}\")", path.display());
    assert_eq!(build(&src).string("main"), "hello\n");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn string_search_split_and_join() {
    assert_eq!(
        build("fun main() : String = str_slice(\"abcdef\", 1, 3)").string("main"),
        "bcd"
    );
    assert_eq!(
        build("fun main() : String = str_join(\"|\", str_split(\",\", \"a,b,c\"))").string("main"),
        "a|b|c"
    );
    assert_eq!(
        build("fun main() : String = str_join(\",\", str_lines(\"a\\nb\\nc\"))").string("main"),
        "a,b,c"
    );
    assert!(build("fun main() : Bool = str_contains(\"bc\", \"abcd\")").bool("main", &[]));
    assert!(!build("fun main() : Bool = str_contains(\"zz\", \"abcd\")").bool("main", &[]));
    assert_eq!(
        build("fun main() : String = str_of_int(str_find(\"bc\", \"abcd\"))").string("main"),
        "1"
    );
    assert!(build("fun main() : Bool = ok(0)").bool("main", &[]));
    assert!(!build("fun main() : Bool = ok(1)").bool("main", &[]));
}

#[test]
fn externs_work_as_values() {
    let src = "\
        fun apply(f : a -> b -> c, x : a, y : b) : c = f(x, y)\n\
        fun main() : Int = if apply(str_eq, \"same\", \"same\") then 1 else 0";
    assert_eq!(run_i64(src), 1);
}

#[test]
fn spawn_status_and_capture() {
    assert_eq!(run_i64("fun main() : Int = spawn_status(\"true\", [])"), 0);
    assert_eq!(run_i64("fun main() : Int = spawn_status(\"false\", [])"), 1);
    assert_eq!(
        build("fun main() : String = spawn_capture(\"echo\", [\"hi\"])").string("main"),
        "hi\n"
    );
    assert_eq!(
        run_i64("fun main() : Int = if spawn(\"true\", []) then 1 else 0"),
        1
    );
}

#[test]
fn shell_commands_and_captures() {
    assert_eq!(run_i64("fun main() : Int = if $ true then 1 else 0"), 1);
    assert_eq!(run_i64("fun main() : Int = if $ false then 1 else 0"), 0);
    assert_eq!(build("fun main() : String = $(echo hi)").string("main"), "hi\n");
    assert_eq!(
        build("fun main() : String = $(printf \"[%s]\" ${\"a b\"})").string("main"),
        "[a b]"
    );
    assert_eq!(
        build("fun main() : String = $(printf \"%s\" ${concat(\"x\", \"y\")})").string("main"),
        "xy"
    );
    assert_eq!(
        build("fun main() : String = $(echo $(echo nested))").string("main"),
        "nested\n\n"
    );
    assert_eq!(
        build("fun main() : String = $(printf \"%s\" 'a b')").string("main"),
        "a b"
    );
    assert_eq!(
        build("fun main() : String = $(echo \"a|b\")").string("main"),
        "a|b\n"
    );
}

#[test]
fn shell_pipelines() {
    assert_eq!(run_i64("fun main() : Int = if $ true | false then 1 else 0"), 0);
    assert_eq!(run_i64("fun main() : Int = if $ false | true then 1 else 0"), 1);
    assert_eq!(
        build("fun main() : String = $(echo hi | tr a-z A-Z)").string("main"),
        "HI\n"
    );
    assert!(build(
        "fun main() : Bool = str_contains(\"3\", $(printf \"a\\nb\\nc\\n\" | wc -l))"
    )
    .bool("main", &[]));
}

#[test]
fn shell_builtins_and_bool_composition() {
    assert!(build("fun main() : Bool = ($ #env HOME) == getenv(\"HOME\")").bool("main", &[]));
    assert!(build("fun main() : Bool = #env(\"HOME\") == getenv(\"HOME\")").bool("main", &[]));
    assert!(build("fun main() : Bool = $ true && $ true").bool("main", &[]));
    assert!(!build("fun main() : Bool = $ true && $ false").bool("main", &[]));
    assert!(build("fun main() : Bool = $ false || $ true").bool("main", &[]));
    assert_eq!(
        build("fun main() : Int = $(printf hi) |> str_len").i64("main", &[]),
        2
    );
    assert_eq!(run_i64("fun main() : Int = if $ true && $ true then 1 else 0"), 1);
    assert_eq!(run_i64("fun main() : Int = if $ false then 0 else 1"), 1);
    assert_eq!(
        run_i64("fun main() : Int = if $ true | false |> not then 1 else 0"),
        1
    );
    assert_eq!(
        run_i64("fun main() : Int = { let ok = $ true; if ok then 1 else 0 }"),
        1
    );
}

#[test]
fn shell_rejects_bare_env_vars_and_builtin_pipelines() {
    let err = fern::compile("fun main() : Int = $(echo $HOME)").expect_err("rejects");
    assert!(err.msg.contains("${expr}"), "{:?}", err.msg);
    let err = fern::compile("fun main() : Int = $ #env HOME | cat").expect_err("rejects");
    assert!(err.msg.contains("pipeline stage"), "{:?}", err.msg);
    let err = fern::compile("fun main() : Int = $(echo ${1})").expect_err("rejects");
    assert!(err.msg.contains("expected `String`"), "{:?}", err.msg);
}

