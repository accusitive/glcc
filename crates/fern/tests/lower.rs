//! Lowering tests: compile whole programs and check the built MIR with
//! the MIR verifier, plus a few shape assertions on the generated code.

use fern::compile;
use glcc_mir::ir::Module;

fn lower(src: &str) -> Module {
    let module = compile(src).unwrap_or_else(|e| panic!("compile failed: {}", e.msg));
    let errors = glcc_mir::verify::verify_module(&module);
    assert!(errors.is_empty(), "MIR verification failed:\n{}", errors.join("\n"));
    module
}

fn names(module: &Module) -> Vec<String> {
    module.funcs.iter().map(|f| f.name.clone()).collect()
}

fn has(module: &Module, name: &str) -> bool {
    names(module).iter().any(|n| n == name)
}

#[test]
fn empty_program() {
    let m = lower("");
    // The runtime's entry points are always declared as externs for the
    // driver to resolve against `rt/gc.c`.
    let externs: Vec<&str> = m.externs.iter().map(|e| e.name.as_str()).collect();
    for name in ["fern_enter", "fern_pin", "fern_leave", "fern_alloc"] {
        assert!(externs.contains(&name), "missing runtime extern `{name}`");
    }
}

#[test]
fn runtime_externs_carry_memory_effects() {
    let m = lower("");
    let effect = |name: &str| {
        m.externs
            .iter()
            .find(|data| data.name == name)
            .unwrap_or_else(|| panic!("missing runtime extern `{name}`"))
            .effect
    };
    assert_eq!(effect("fern_enter"), glcc_mir::ExternEffect::neutral());
    assert_eq!(effect("fern_pin"), glcc_mir::ExternEffect::neutral());
    assert_eq!(effect("fern_leave"), glcc_mir::ExternEffect::neutral());
    assert_eq!(effect("fern_alloc"), glcc_mir::ExternEffect::allocator());
    assert_eq!(effect("fern_str_eq"), glcc_mir::ExternEffect::read_only());
}

#[test]
fn identity_and_call() {
    let m = lower("fn id(x) : for<A> A -> A = x\nfn main() : -> Int = id(1)");
    assert!(has(&m, "id$$Int"));
    assert!(has(&m, "main"));
}

#[test]
fn arithmetic_and_branches() {
    lower("fn abs(x) : Int -> Int = if x < 0 then -x else x\nfn max(x, y) : (Int, Int) -> Int = if x > y then x else y");
    lower("fn f(x) : Int -> Int = x * 2 - 3 / 1\nfn g(x) : Float -> Bool = x < 1.5");
}

#[test]
fn short_circuit_keeps_narrowing() {
    lower("fn both(a, b) : (Bool, Bool) -> Bool = a && b\nfn either(a, b) : (Bool, Bool) -> Bool = a || b");
}

#[test]
fn recursion_over_lists() {
    let m = lower(
        "fn sum(xs) : List<Int> -> Int = match xs { case Nil => 0 case Cons(x, xs) => x + sum(xs) }\n\
         fn main() : -> Int = sum([1, 2, 3])",
    );
    assert!(has(&m, "sum"));
}

#[test]
fn patterns_and_tuples() {
    lower(
        "fn swap(p) : for<A, B> (A, B) -> (B, A) = match p { case (a, b) => (b, a) }\n\
         fn first(p) : for<A, B> (A, B) -> A = match p { case (a, _) => a }\n\
         fn zip_len(xs) : List<(Int, Int)> -> Int = match xs { case Nil => 0 case Cons((a, b), _) => a + b }\n\
         fn main() : -> Int = first(swap((1, 2))) + zip_len([(1, 2)])",
    );
    lower("fn unit() : -> Unit = {}\nfn k(x) : Int -> Int = { let (a, b) = (1, 2); a + b + x }");
}

#[test]
fn int_literal_patterns_and_guards() {
    lower(
        "fn sign(x) : Int -> Int = match x { case 0 => 0 case n when n < 0 => -1 case _ => 1 }\n\
         fn main() : -> Int = sign(0) + sign(1)",
    );
    lower("fn b(x) : Bool -> Int = match x { case true => 1 case false => 0 }");
}

#[test]
fn closures_and_higher_order() {
    let m = lower(
        "fn apply(f, x) : for<A, B> (A -> B, A) -> B = f(x)\n\
         fn adder(n) : Int -> Int -> Int = |x| x + n\n\
         fn compose(f, g) : for<A, B, C> (B -> C, A -> B) -> A -> C = |x| f(g(x))\n\
         fn main() : -> Int = apply(adder(3), 4) + apply(compose(|y| y * 2, |y| y + 1), 1)",
    );
    assert!(has(&m, "apply$$Int$Int"));
}

#[test]
fn escaping_closure() {
    lower("fn adder(n) : Int -> Int -> Int = |x| x + n\nfn main() : -> Int = { let f = adder(2); f(3) }");
}

#[test]
fn class_dispatch_over_lists() {
    let m = lower("fn main() : -> Bool = eq([1, 2], [1, 3])");
    assert!(names(&m).iter().any(|n| n.starts_with("List$Int::Eq::eq")));
    assert!(names(&m).iter().any(|n| n.starts_with("Int::Eq::eq")));
}

#[test]
fn custom_hkt_class() {
    let m = lower(
        "class Functor<F> { map: for<A, B> (A -> B) -> F<A> -> F<B> }\n\
         instance Functor<List> {\n\
             map(f, xs) = match xs { case Nil => Nil case Cons(x, xs) => Cons(f(x), map(f, xs)) }\n\
         }\n\
         fn main() : -> List<Int> = map(|x| x + 1, [1, 2])",
    );
    assert!(names(&m).iter().any(|n| n.starts_with("List::Functor::map$$Int$Int")));
}

#[test]
fn named_function_as_value() {
    let m = lower(
        "fn twice(f, x) : for<A> (A -> A, A) -> A = f(f(x))\n\
         fn inc(x) : Int -> Int = x + 1\n\
         fn main() : -> Int = twice(inc, 1)",
    );
    assert!(names(&m).iter().any(|n| n.contains("$thunk")));
}

#[test]
fn method_value_and_partial_use() {
    lower(
        "fn apply(f, x) : for<A, B> (A -> B, A) -> B = f(x)\n\
         fn main() : -> Int = apply(|y| y + 1, 1) + apply(|z| z + 1, 2)",
    );
}

#[test]
fn maybe_with_bounds() {
    let m = lower(
        "fn eq_maybe(mx, my) : (Maybe<Int>, Maybe<Int>) -> Bool = eq(mx, my)\n\
         fn main() : -> Bool = eq_maybe(Just(1), Nothing)",
    );
    assert!(names(&m).iter().any(|n| n.contains("Maybe")));
}

#[test]
fn tail_self_calls_lower_to_a_loop() {
    let m = lower(
        "fn count(n, acc) : (Int, Int) -> Int =\n\
         if n == 0 then acc else if n / 2 * 2 == n then count(n - 1, acc + 1) else count(n - 1, acc + 2)\n\
         fn main() : -> Int = count(1000000, 0)",
    );
    assert!(has(&m, "count"));
}

#[test]
fn async_lowers_to_start_and_resume() {
    let m = lower("async fn f(x) : Int -> Int = x + 1\nfn main() : -> Int = block_on_int(f(1))");
    assert!(has(&m, "f"));
    assert!(has(&m, "f$resume"));
    assert!(has(&m, "main"));
}

#[test]
fn await_lowers_to_getelementptr_and_result() {
    let m = lower(
        "async fn f(x) : Int -> Int = x\n\
         async fn g(x) : Int -> Int = f(x).await + 1\n\
         fn main() : -> Int = block_on_int(g(1))",
    );
    assert!(has(&m, "g$resume"));
    assert!(m.externs.iter().any(|e| e.name == "__glcc_frame_alloc"));
}

#[test]
fn polymorphic_instances_monomorphize_per_type() {
    let m = lower("fn main() : -> Bool = eq(1, 2) && eq([true], [false])");
    assert!(has(&m, "Int::Eq::eq"));
    assert!(has(&m, "Bool::Eq::eq"));
    assert!(names(&m).iter().any(|n| n.starts_with("List$Bool::Eq::eq")));
}
