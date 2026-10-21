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
fn identity_and_call() {
    let m = lower("fun id(x : a) : a = x\nfun main() : Int = id(1)");
    assert!(has(&m, "id$$Int"));
    assert!(has(&m, "main"));
}

#[test]
fn arithmetic_and_branches() {
    lower("fun abs(x : Int) : Int = if x < 0 then -x else x\nfun max(x : Int, y : Int) : Int = if x > y then x else y");
    lower("fun f(x : Int) : Int = x * 2 - 3 / 1\nfun g(x : Float) : Bool = x < 1.5");
}

#[test]
fn short_circuit_keeps_narrowing() {
    lower("fun both(a : Bool, b : Bool) : Bool = a && b\nfun either(a : Bool, b : Bool) : Bool = a || b");
}

#[test]
fn recursion_over_lists() {
    let m = lower(
        "fun sum(xs : List Int) : Int = match xs { case Nil => 0 case Cons(x, xs) => x + sum(xs) }\n\
         fun main() : Int = sum([1, 2, 3])",
    );
    assert!(has(&m, "sum"));
}

#[test]
fn patterns_and_tuples() {
    lower(
        "fun swap(p : (a, b)) : (b, a) = match p { case (a, b) => (b, a) }\n\
         fun first(p : (a, b)) : a = match p { case (a, _) => a }\n\
         fun zip_len(xs : List (Int, Int)) : Int = match xs { case Nil => 0 case Cons((a, b), _) => a + b }\n\
         fun main() : Int = first(swap((1, 2))) + zip_len([(1, 2)])",
    );
    lower("fun unit() : Unit = {}\nfun k(x : Int) : Int = { let (a, b) = (1, 2); a + b + x }");
}

#[test]
fn int_literal_patterns_and_guards() {
    lower(
        "fun sign(x : Int) : Int = match x { case 0 => 0 case n when n < 0 => -1 case _ => 1 }\n\
         fun main() : Int = sign(0) + sign(1)",
    );
    lower("fun b(x : Bool) : Int = match x { case true => 1 case false => 0 }");
}

#[test]
fn closures_and_higher_order() {
    let m = lower(
        "fun apply(f : a -> b, x : a) : b = f(x)\n\
         fun adder(n : Int) : Int -> Int = x -> x + n\n\
         fun compose(f : b -> c, g : a -> b) : a -> c = x -> f(g(x))\n\
         fun main() : Int = apply(adder(3), 4) + apply(compose(y -> y * 2, y -> y + 1), 1)",
    );
    assert!(has(&m, "apply$$Int$Int"));
}

#[test]
fn escaping_closure() {
    lower("fun adder(n : Int) : Int -> Int = x -> x + n\nfun main() : Int = { let f = adder(2); f(3) }");
}

#[test]
fn class_dispatch_over_lists() {
    let m = lower("fun main() : Bool = eq([1, 2], [1, 3])");
    assert!(names(&m).iter().any(|n| n.starts_with("List$Int::Eq::eq")));
    assert!(names(&m).iter().any(|n| n.starts_with("Int::Eq::eq")));
}

#[test]
fn custom_hkt_class() {
    let m = lower(
        "class Functor f { map : (a -> b) -> f a -> f b }\n\
         instance Functor List {\n\
             map(f, xs) = match xs { case Nil => Nil case Cons(x, xs) => Cons(f(x), map(f, xs)) }\n\
         }\n\
         fun main() : List Int = map(x -> x + 1, [1, 2])",
    );
    assert!(names(&m).iter().any(|n| n.starts_with("List::Functor::map$$Int$Int")));
}

#[test]
fn named_function_as_value() {
    let m = lower(
        "fun twice(f : a -> a, x : a) : a = f(f(x))\n\
         fun inc(x : Int) : Int = x + 1\n\
         fun main() : Int = twice(inc, 1)",
    );
    assert!(names(&m).iter().any(|n| n.contains("$thunk")));
}

#[test]
fn method_value_and_partial_use() {
    lower(
        "fun apply(f : a -> b, x : a) : b = f(x)\n\
         fun main() : Int = apply(y -> y + 1, 1) + apply(z -> z + 1, 2)",
    );
}

#[test]
fn maybe_with_bounds() {
    let m = lower(
        "fun eq_maybe(mx : Maybe Int, my : Maybe Int) : Bool = eq(mx, my)\n\
         fun main() : Bool = eq_maybe(Just(1), Nothing)",
    );
    assert!(names(&m).iter().any(|n| n.contains("Maybe")));
}

#[test]
fn async_lowers_to_start_and_resume() {
    let m = lower("async fun f(x : Int) : Int = x + 1\nfun main() : Int = block_on_int(f(1))");
    assert!(has(&m, "f"));
    assert!(has(&m, "f$resume"));
    assert!(has(&m, "main"));
}

#[test]
fn await_lowers_to_getelementptr_and_result() {
    let m = lower(
        "async fun f(x : Int) : Int = x\n\
         async fun g(x : Int) : Int = await f(x) + 1\n\
         fun main() : Int = block_on_int(g(1))",
    );
    assert!(has(&m, "g$resume"));
    assert!(m.externs.iter().any(|e| e.name == "__glcc_frame_alloc"));
}

#[test]
fn polymorphic_instances_monomorphize_per_type() {
    let m = lower("fun main() : Bool = eq(1, 2) && eq([true], [false])");
    assert!(has(&m, "Int::Eq::eq"));
    assert!(has(&m, "Bool::Eq::eq"));
    assert!(names(&m).iter().any(|n| n.starts_with("List$Bool::Eq::eq")));
}
