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
    let m = lower("fun id(x) = x\nfun main() = id(1)");
    assert!(has(&m, "id$$Int"));
    assert!(has(&m, "main"));
}

#[test]
fn arithmetic_and_branches() {
    lower("fun abs(x) = if x < 0 then -x else x\nfun max(x, y) = if x > y then x else y");
    lower("fun f(x) = x * 2 - 3 / 1\nfun g(x) = x < 1.5");
}

#[test]
fn short_circuit_keeps_narrowing() {
    lower("fun both(a, b) = a && b\nfun either(a, b) = a || b");
}

#[test]
fn recursion_over_lists() {
    let m = lower(
        "fun sum(xs) = match xs { case Nil => 0 case Cons(x, xs) => x + sum(xs) }\n\
         fun main() = sum([1, 2, 3])",
    );
    assert!(has(&m, "sum"));
}

#[test]
fn patterns_and_tuples() {
    lower(
        "fun swap(p) = match p { case (a, b) => (b, a) }\n\
         fun first(p) = match p { case (a, _) => a }\n\
         fun zip_len(xs) = match xs { case Nil => 0 case Cons((a, b), _) => a + b }\n\
         fun main() = first(swap((1, 2))) + zip_len([(1, 2)])",
    );
    lower("fun unit() = {}\nfun k(x) = { let (a, b) = (1, 2); a + b + x }");
}

#[test]
fn int_literal_patterns_and_guards() {
    lower(
        "fun sign(x) = match x { case 0 => 0 case n when n < 0 => -1 case _ => 1 }\n\
         fun main() = sign(0) + sign(1)",
    );
    lower("fun b(x) = match x { case true => 1 case false => 0 }");
}

#[test]
fn closures_and_higher_order() {
    let m = lower(
        "fun apply(f, x) = f(x)\n\
         fun adder(n) = x -> x + n\n\
         fun compose(f, g) = x -> f(g(x))\n\
         fun main() = apply(adder(3), 4) + apply(compose(y -> y * 2, y -> y + 1), 1)",
    );
    assert!(has(&m, "apply$$Int$Int"));
}

#[test]
fn escaping_closure() {
    lower("fun adder(n) = x -> x + n\nfun main() = { let f = adder(2); f(3) }");
}

#[test]
fn class_dispatch_over_lists() {
    let m = lower("fun main() = eq([1, 2], [1, 3])");
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
         fun main() = map(x -> x + 1, [1, 2])",
    );
    assert!(names(&m).iter().any(|n| n.starts_with("List::Functor::map$$Int$Int")));
}

#[test]
fn named_function_as_value() {
    let m = lower(
        "fun twice(f, x) = f(f(x))\n\
         fun inc(x) = x + 1\n\
         fun main() = twice(inc, 1)",
    );
    assert!(names(&m).iter().any(|n| n.contains("$thunk")));
}

#[test]
fn method_value_and_partial_use() {
    lower(
        "fun apply(f, x) = f(x)\n\
         fun main() = apply(y -> y + 1, 1) + apply(z -> z + 1, 2)",
    );
}

#[test]
fn maybe_with_bounds() {
    let m = lower(
        "fun eq_maybe(mx, my) = eq(mx, my)\n\
         fun main() = eq_maybe(Just(1), Nothing)",
    );
    assert!(names(&m).iter().any(|n| n.contains("Maybe")));
}

#[test]
fn polymorphic_instances_monomorphize_per_type() {
    let m = lower("fun main() = eq(1, 2) && eq([true], [false])");
    assert!(has(&m, "Int::Eq::eq"));
    assert!(has(&m, "Bool::Eq::eq"));
    assert!(names(&m).iter().any(|n| n.starts_with("List$Bool::Eq::eq")));
}
