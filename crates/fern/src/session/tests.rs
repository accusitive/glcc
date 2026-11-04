use super::*;

fn unit(session: &Session, file: FileId) -> Vec<FileId> {
    vec![session.prelude(), file]
}

#[test]
fn parse_is_cached_until_the_revision_changes() {
    let mut session = Session::new();
    let file = session.add_file("t.fern", "fn main() : -> Int = 1");
    let first = session.parse(file);
    let second = session.parse(file);
    assert!(Rc::ptr_eq(&first, &second));

    session.set_text(file, "fn main() : -> Int = 2");
    let third = session.parse(file);
    assert!(!Rc::ptr_eq(&second, &third));
}

#[test]
fn unchanged_unit_does_not_recompile() {
    let mut session = Session::new();
    let file = session.add_file("t.fern", "fn main() : -> Int = 1");
    let files = unit(&session, file);

    session.compile(&files).expect("compiles");
    session.compile(&files).expect("compiles again");
    assert_eq!(session.unit_compiles.get(), 1);

    session.set_text(file, "fn main() : -> Int = 2");
    let files = unit(&session, file);
    session.compile(&files).expect("compiles after edit");
    assert_eq!(session.unit_compiles.get(), 2);
}

#[test]
fn analysis_is_reused_for_an_unchanged_unit() {
    let mut session = Session::new();
    let file = session.add_file("t.fern", "fn main() : -> Int = 1");
    let files = unit(&session, file);
    let first = session.analyze(&files).expect("checks");
    let second = session.analyze(&files).expect("checks again");
    assert!(Rc::ptr_eq(&first, &second));
    assert_eq!(session.unit_compiles.get(), 1);

    session.set_text(file, "fn main() : -> Int = 2");
    let third = session.analyze(&unit(&session, file)).expect("checks");
    assert!(!Rc::ptr_eq(&first, &third));
}

#[test]
fn session_surfaces_recovered_items_and_diags() {
    let mut session = Session::new();
    let file = session.add_file("t.fern", "fn broken( = 1\nfn ok() : -> Int = 2\n");
    let parsed = session.parse(file);
    assert!(!parsed.diags.is_empty());
    let names: Vec<&str> = parsed
        .program
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            crate::ast::ItemKind::Fn(fun) => Some(fun.name.as_str()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"ok"), "{names:?}");
}

#[test]
fn multiple_files_merge_in_order() {
    let mut session = Session::new();
    let a = session.add_file("a.fern", "fn twice(x) : Int -> Int = x * 2");
    let b = session.add_file("b.fern", "fn main() : -> Int = twice(21)");
    let files = [session.prelude(), a, b];
    let (program, diags, _) = session.program(&files);
    assert!(diags.is_empty(), "{diags:?}");
    let user_items = program.items.iter().filter(|item| item.span.file == a || item.span.file == b).count();
    assert_eq!(user_items, 2);
    let module = session.compile(&files).unwrap_or_else(|d| {
        panic!("compile failed: {}", d[0].msg);
    });
    assert!(module.funcs.iter().any(|f| f.name == "main"));
}

#[test]
fn appending_a_definition_rechecks_only_it() {
    let mut session = Session::new();
    let file = session.add_file("t.fern", "fn a() : -> Int = 1\nfn b() : -> Int = 2\n");
    let files = unit(&session, file);
    session.analyze(&files).expect("checks");

    session.set_text(file, "fn a() : -> Int = 1\nfn b() : -> Int = 2\nfn c() : -> Int = 3\n");
    let second = session.analyze(&files).expect("checks again");
    assert_eq!(session.unit_compiles.get(), 2);
    assert_eq!(second.checked.get(), 1, "only the appended item is checked");
}

#[test]
fn appended_functions_can_forward_reference_each_other() {
    let mut session = Session::new();
    let file = session.add_file("t.fern", "fn a() : -> Int = 1\n");
    let files = unit(&session, file);
    session.analyze(&files).expect("checks");

    session.set_text(file, "fn a() : -> Int = 1\nfn f() : -> Int = g()\nfn g() : -> Int = 2\n");
    let second = session.analyze(&files).expect("checks forward references");
    assert_eq!(second.checked.get(), 2, "both appended bodies are checked");
}

#[test]
fn replacing_a_temporary_expression_rechecks_only_it() {
    let mut session = Session::new();
    let file = session.add_file("t.fern", "fn a() : -> Int = 1\n");
    let files = unit(&session, file);
    session.analyze(&files).expect("checks");

    session.set_text(file, "fn a() : -> Int = 1\nfn __repl_0() : -> Int = (a())\n");
    let first = session.analyze(&files).expect("checks the expression");
    assert_eq!(first.checked.get(), 1);

    session.set_text(file, "fn a() : -> Int = 1\nfn __repl_1() : -> Int = (a() + 1)\n");
    let second = session.analyze(&files).expect("checks the next expression");
    assert_eq!(second.checked.get(), 1, "only the new expression is checked");
    assert!(!second.hir.bodies.iter().any(|body| body.name == "__repl_0"), "the old expression's body is dropped");
    assert!(second.hir.bodies.iter().any(|body| body.name == "__repl_1"));
}

#[test]
fn editing_the_last_item_rechecks_from_there() {
    let mut session = Session::new();
    let file = session.add_file("t.fern", "fn a() : -> Int = 1\nfn b() : -> Int = 2\n");
    let files = unit(&session, file);
    session.analyze(&files).expect("checks");

    session.set_text(file, "fn a() : -> Int = 1\nfn b() : -> Int = 20\n");
    let second = session.analyze(&files).expect("checks the edit");
    assert_eq!(second.checked.get(), 1, "only the edited item is checked");
}

#[test]
fn appending_lowers_only_the_new_body() {
    let mut session = Session::new();
    let file = session.add_file("t.fern", "fn a() : -> Int = 1\nfn b() : -> Int = 2\n");
    let files = unit(&session, file);
    session.compile_shared(&files).expect("compiles");
    assert!(session.lower_stats().full);

    session.set_text(file, "fn a() : -> Int = 1\nfn b() : -> Int = 2\nfn c() : -> Int = 3\n");
    let module = session.compile_shared(&files).expect("compiles again");
    let stats = session.lower_stats();
    assert!(!stats.full, "the new item resumes the previous lowering");
    assert_eq!(stats.new_roots, 1);
    assert_eq!(stats.bodies, 1, "only c is emitted");
    assert!(module.funcs.iter().any(|f| f.name == "c"));
}

#[test]
fn replacing_an_expression_keeps_earlier_bodies() {
    let mut session = Session::new();
    let file = session.add_file("t.fern", "fn a() : -> Int = 1\n");
    let files = unit(&session, file);
    session.compile_shared(&files).expect("compiles the definitions");

    session.set_text(file, "fn a() : -> Int = 1\nfn __repl_0() : -> Int = (a())\n");
    let module = session.compile_shared(&files).expect("compiles");
    assert!(module.funcs.iter().any(|f| f.name == "__repl_0"));
    assert!(!session.lower_stats().full);

    session.set_text(file, "fn a() : -> Int = 1\nfn __repl_1() : -> Int = (a() + 1)\n");
    let module = session.compile_shared(&files).expect("compiles again");
    assert!(!session.lower_stats().full, "the prefix is reused");
    assert_eq!(session.lower_stats().bodies, 1, "only the new expression is emitted");
    assert!(module.funcs.iter().any(|f| f.name == "__repl_1"));
    assert!(module.funcs.iter().any(|f| f.name == "a"), "a is still there");
}

#[test]
fn editing_a_definition_drops_the_stale_lowering() {
    let mut session = Session::new();
    let file = session.add_file("t.fern", "fn a() : -> Int = 1\nfn b() : -> Int = 2\n");
    let files = unit(&session, file);
    session.compile_shared(&files).expect("compiles");

    session.set_text(file, "fn a() : -> Int = 1\nfn b() : -> Int = 20\n");
    let module = session.compile_shared(&files).expect("recompiles");
    assert!(session.lower_stats().full, "a changed body forces a fresh module");
    assert_eq!(module.funcs.iter().filter(|f| f.name == "b").count(), 1);
}
