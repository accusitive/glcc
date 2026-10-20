//! Collection tests: kinds, instance heads, and diagnostics.

use fern::defs::{self, Defs};
use fern::ty::Kind;
use fern::{lexer, parser};

fn collect(src: &str) -> Result<Defs, fern::Diag> {
    let tokens = lexer::lex(src)?;
    let program = parser::parse(tokens)?;
    defs::collect(&program)
}

fn collect_ok(src: &str) -> Defs {
    match collect(src) {
        Ok(defs) => defs,
        Err(diag) => panic!("collection failed: {}", diag.msg),
    }
}

fn collect_err(src: &str) -> fern::Diag {
    match collect(src) {
        Ok(_) => panic!("expected a collection error"),
        Err(diag) => diag,
    }
}

fn star() -> Kind {
    Kind::star()
}

fn arrow(a: Kind, b: Kind) -> Kind {
    Kind::arrow(a, b)
}

#[test]
fn infers_higher_kinded_parameters() {
    let defs = collect_ok("data Free f a = Pure(a) | Wrap(f a, Free f a)");
    assert_eq!(defs.data.len(), 1);
    let free = defs.kind_of_head("Free").expect("Free's kind");
    assert_eq!(*free, arrow(arrow(star(), star()), arrow(star(), star())));
}

#[test]
fn resolves_class_and_instance_heads() {
    let src = "\
class Functor f { map : (a -> b) -> f a -> f b }
data List a = Nil | Cons(a, List a)
instance Functor List {}
instance Functor (Free g) when Functor g {}
data Free f a = Pure(a) | Wrap(f a, Free f a)
";
    let defs = collect_ok(src);
    assert_eq!(defs.instances.len(), 2);

    let functor = defs.kind_of_head("Functor").expect("Functor");
    assert_eq!(*functor, arrow(arrow(star(), star()), star()));

    let second = &defs.instances[1];
    assert_eq!(second.generics.len(), 1);
    assert_eq!(second.generics[0].0, "g");
    assert_eq!(second.generics[0].1, arrow(star(), star()));
    assert_eq!(second.args.len(), 1);
    assert_eq!(second.args[0].to_string(), "Free g");
    assert_eq!(second.bounds.len(), 1);

    assert_eq!(defs.ctor_by_name("Cons"), Some((defs.data_ids["List"], 1)));
}

#[test]
fn rejects_nullary_instance_of_kinded_class() {
    let src = "\
data Wrapped = Wrapped(Int)
class Functor f { map : (a -> b) -> f a -> f b }
instance Functor Wrapped {}
";
    let err = collect_err(src);
    assert!(err.msg.contains("kinds do not match"), "{}", err.msg);
}

#[test]
fn rejects_wrong_arity_class_application() {
    let err = collect_err("class Clash a b { clash : a -> b -> a }\ninstance Clash Int {}");
    assert!(err.msg.contains("takes 2 type arguments"), "{}", err.msg);

    let err = collect_err("class Clash a b { clash : a -> b -> a }\ninstance Clash a b c {}");
    assert!(err.msg.contains("takes 2 type arguments"), "{}", err.msg);
}

#[test]
fn multi_parameter_class_takes_positional_arguments() {
    let src = "\
class Clash a b { clash : a -> b -> a }
instance Clash Int Bool {}
instance Clash (List a) Float {}
data List a = Nil | Cons(a, List a)
";
    let defs = collect_ok(src);
    assert_eq!(defs.instances[0].args[0].to_string(), "Int");
    assert_eq!(defs.instances[0].self_ty().to_string(), "Int");
    assert_eq!(defs.instances[1].args[0].to_string(), "List a");
    assert_eq!(defs.instances[1].args[1].to_string(), "Float");
}

#[test]
fn rejects_ambiguous_and_duplicate_instances() {
    let src = "class Sum a { add : a -> a -> a }\ninstance Sum a {}";
    let err = collect_err(src);
    assert!(err.msg.contains("bare type parameter"), "{}", err.msg);

    let src = "\
class Sum a { add : a -> a -> a }
instance Sum Int {}
instance Sum Int {}
";
    let err = collect_err(src);
    assert!(err.msg.contains("duplicate instance"), "{}", err.msg);
}

#[test]
fn collects_extern_signatures() {
    let defs = collect_ok("extern \"puts\" put : String -> Int\nextern twice : Int -> Int -> Int");
    let put = defs.extern_by_name("put").expect("put");
    assert_eq!(defs.externs[put.index()].symbol, "puts");
    assert_eq!(defs.externs[put.index()].params.len(), 1);
    assert_eq!(defs.externs[put.index()].params[0].to_string(), "String");
    assert_eq!(defs.externs[put.index()].ret.to_string(), "Int");

    let twice = defs.extern_by_name("twice").expect("twice");
    assert_eq!(defs.externs[twice.index()].symbol, "twice");
    assert_eq!(defs.externs[twice.index()].params.len(), 2);
    assert_eq!(defs.externs[twice.index()].ret.to_string(), "Int");
}

#[test]
fn externs_share_the_value_namespace() {
    let err = collect_err("fun f(x : Int) : Int = x\nextern f : Int -> Int");
    assert!(err.msg.contains("already defined"), "{}", err.msg);
    let err = collect_err("extern f : Int -> Int\nfun f(x : Int) : Int = x");
    assert!(err.msg.contains("already defined"), "{}", err.msg);
}

#[test]
fn reports_duplicate_definitions() {
    let err = collect_err("data Foo = A\ndata Foo = B");
    assert!(err.msg.contains("duplicate type name"), "{}", err.msg);

    let err = collect_err("fun f(x : Int) : Int = x\nlet f = 1");
    assert!(err.msg.contains("already defined"), "{}", err.msg);

    let err = collect_err("data A = Same\ndata B = Same");
    assert!(err.msg.contains("duplicate constructor"), "{}", err.msg);
}

#[test]
fn reports_unknown_types() {
    let err = collect_err("data Foo a = Foo(Missing a)");
    assert!(err.msg.contains("cannot find type `Missing`"), "{}", err.msg);
}

#[test]
fn fern_prefix_is_reserved_for_the_runtime() {
    for src in ["fun fern_alloc(x : Int) : Int = x", "let fern_two = 2"] {
        let diag = collect_err(src);
        assert!(diag.msg.contains("reserved"), "{src}: {}", diag.msg);
    }
}
