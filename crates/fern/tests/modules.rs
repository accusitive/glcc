use fern::ast::ModuleId;
use fern::span::FileId;

fn compile(root: &str, modules: &[(&[&str], &str)]) -> fern::MirModule {
    let mut session = fern::Session::new();
    let main = session.add_file("<main>", root);
    let mut files = vec![session.prelude(), main];
    for (path, source) in modules {
        let path: Vec<String> = path.iter().map(|s| s.to_string()).collect();
        files.push(session.add_module_file(format!("<{}>", path.join("::")), path, *source));
    }
    session.compile(&files).unwrap_or_else(|diags| {
        panic!("compile failed: {}", diags[0].msg);
    })
}

fn compile_err(root: &str, modules: &[(&[&str], &str)]) -> String {
    let mut session = fern::Session::new();
    let main = session.add_file("<main>", root);
    let mut files = vec![session.prelude(), main];
    for (path, source) in modules {
        let path: Vec<String> = path.iter().map(|s| s.to_string()).collect();
        files.push(session.add_module_file(format!("<{}>", path.join("::")), path, *source));
    }
    session.compile(&files).err().map(|diags| diags[0].msg.clone()).unwrap_or_else(|| panic!("compiled"))
}

#[test]
fn import_binds_a_module_alias() {
    let module = compile("import math\nfun main() : Int = math.double(21)", &[(&["math"], "fun double(x : Int) : Int = x * 2")]);
    assert!(module.funcs.iter().any(|f| f.name == "main"));
    assert!(module.funcs.iter().any(|f| f.name == "math$double"));
}

#[test]
fn module_generics_monomorphize_across_files() {
    let module = compile("import math\nfun main() : Int = math.id(41) + 1", &[(&["math"], "fun id(x : a) : a = x")]);
    assert!(module.funcs.iter().any(|f| f.name == "main"));
    assert!(module.funcs.iter().any(|f| f.name.starts_with("math$id$$")));
}

#[test]
fn use_brings_an_item_into_scope() {
    let module = compile(
        "import math\nuse math::double\nfun main() : Int = double(21)",
        &[(&["math"], "fun double(x : Int) : Int = x * 2")],
    );
    assert!(module.funcs.iter().any(|f| f.name == "main"));
}

#[test]
fn qualified_types_and_constructors_work() {
    let module = compile(
        "import math\nfun main() : Int = math.size(math.Leaf(1))",
        &[(
            &["math"],
            "data Tree a = Leaf(a) | Node(Tree a, Tree a)\n\
             fun size(t : Tree a) : Int = match t { case Leaf(_) => 1 case Node(l, r) => size(l) + size(r) }",
        )],
    );
    assert!(module.funcs.iter().any(|f| f.name == "main"));
}

#[test]
fn qualified_constructors_work_in_patterns() {
    let module = compile(
        "import math\n\
         fun main() : Int = match math.Leaf(7) { case math.Leaf(x) => x case math.Node(_, _) => 0 }",
        &[(&["math"], "data Tree a = Leaf(a) | Node(Tree a, Tree a)")],
    );
    assert!(module.funcs.iter().any(|f| f.name == "main"));
}

#[test]
fn used_types_and_constructors_lose_their_prefix() {
    let module = compile(
        "import math\nuse math::Tree\nuse math::Leaf\nuse math::size\n\
         fun main() : Int = size(Leaf(1))",
        &[(
            &["math"],
            "data Tree a = Leaf(a) | Node(Tree a, Tree a)\n\
             fun size(t : Tree a) : Int = match t { case Leaf(_) => 1 case Node(l, r) => size(l) + size(r) }",
        )],
    );
    assert!(module.funcs.iter().any(|f| f.name == "main"));
}

#[test]
fn modules_can_call_each_other() {
    let module = compile(
        "import a\nfun main() : Int = a.one()",
        &[(&["a"], "import b\nfun one() : Int = b.two() + 1"), (&["b"], "fun two() : Int = 2")],
    );
    assert!(module.funcs.iter().any(|f| f.name == "a$one"));
    assert!(module.funcs.iter().any(|f| f.name == "b$two"));
}

#[test]
fn import_cycles_are_allowed() {
    let module = compile(
        "import a\nfun main() : Bool = a.even(4)",
        &[
            (&["a"], "import b\nfun even(n : Int) : Bool = if n == 0 then true else b.odd(n - 1)"),
            (&["b"], "import a\nfun odd(n : Int) : Bool = if n == 0 then false else a.even(n - 1)"),
        ],
    );
    assert!(module.funcs.iter().any(|f| f.name == "a$even"));
    assert!(module.funcs.iter().any(|f| f.name == "b$odd"));
}

#[test]
fn a_used_class_can_be_selected_by_name() {
    let module = compile(
        "import math\nuse math::Describe\nfun main() : Int = Describe::describe(5)",
        &[(
            &["math"],
            "class Describe a { describe : a -> Int }\n\
             instance Describe Int { describe(x) = x }",
        )],
    );
    assert!(module.funcs.iter().any(|f| f.name == "main"));
}

#[test]
fn module_body_edits_recheck_only_the_edited_body() {
    let mut session = fern::Session::new();
    let root = session.add_file("<main>", "import a\nimport c\nfun main() : Int = c.value()");
    let a = session.add_module_file("<a>", vec!["a".into()], "fun helper() : Int = 1");
    let c = session.add_module_file("<c>", vec!["c".into()], "fun value() : Int = 2");
    let files = [session.prelude(), root, a, c];
    let first = session.analyze(&files).expect("first check");
    assert!(first.checked.get() > 3, "the first check is full");

    session.set_text(a, "fun helper() : Int = 10");
    let body = session.analyze(&files).expect("body check");
    assert_eq!(body.checked.get(), 1, "only a's body is rechecked");
    session.compile(&files).expect("lowering still sees the new body");

    session.set_text(a, "fun helper(x : Int) : Int = x");
    let signature = session.analyze(&files).expect("signature check");
    assert_eq!(signature.checked.get(), 2, "a's interface change rechecks a and its importer, not c");
}

#[test]
fn adding_an_instance_changes_the_module_interface() {
    let mut session = fern::Session::new();
    let root = session.add_file("<main>", "import a\nfun main() : Int = 1");
    let a = session.add_module_file("<a>", vec!["a".into()], "data T = T\nclass C x { c : x -> Int }\ninstance C T { c(x) = 1 }");
    let files = [session.prelude(), root, a];
    let first = session.analyze(&files).expect("first");
    let interface = |analysis: &fern::session::Analysis| analysis.module_graph.nodes.iter().find(|node| node.file == a).expect("module node").interface;
    let before = interface(&first);

    session.set_text(a, "data T = T\nclass C x { c : x -> Int }\ninstance C T { c(x) = 1 }\ninstance C Int { c(x) = 2 }");
    let second = session.analyze(&files).expect("second");
    assert_ne!(before, interface(&second));
    assert!(second.checked.get() > 3, "adding an item forces a full check");
}

#[test]
fn the_module_graph_groups_cycles_and_orders_dependencies_first() {
    let mut session = fern::Session::new();
    let root = session.add_file("<main>", "import a\nimport c\nfun main() : Int = a.get() + c.get()");
    let a = session.add_module_file("<a>", vec!["a".into()], "import b\nfun get() : Int = b.get()");
    let b = session.add_module_file("<b>", vec!["b".into()], "import a\nfun get() : Int = 1");
    let c = session.add_module_file("<c>", vec!["c".into()], "fun get() : Int = 2");
    let files = [session.prelude(), root, a, b, c];
    let graph = session.module_graph(&files);
    assert_eq!(graph.nodes.len(), 4);
    assert_eq!(graph.groups.len(), 3);
    assert_eq!(graph.groups[0].files, vec![a, b]);
    assert_eq!(graph.groups[1].files, vec![c]);
    assert_eq!(graph.groups[2].files, vec![root]);
}

#[test]
fn a_self_import_is_a_one_file_cycle() {
    let mut session = fern::Session::new();
    let a = session.add_module_file("<a>", vec!["a".into()], "import a\nfun get() : Int = 1");
    let graph = session.module_graph(&[session.prelude(), a]);
    assert_eq!(graph.groups.len(), 1);
    assert_eq!(graph.groups[0].files, vec![a]);
}

#[test]
fn interface_hashes_ignore_function_bodies_but_see_signatures() {
    let outer = |body: &str| {
        let mut session = fern::Session::new();
        let a = session.add_module_file("<a>", vec!["a".into()], body);
        let graph = session.module_graph(&[session.prelude(), a]);
        graph.nodes[0].interface
    };
    assert_eq!(outer("fun f(x : Int) : Int = x + 1"), outer("fun f(x : Int) : Int = x * 100"));
    assert_ne!(outer("fun f(x : Int) : Int = x + 1"), outer("fun f(x : Int) : Bool = true"));
}

#[test]
fn same_names_in_two_modules_stay_distinct() {
    let source = "data Tree a = Leaf(a)\n\
                  fun make() : Tree Int = Leaf(1)\n\
                  fun get(t : Tree Int) : Int = match t { case Leaf(x) => x }";
    let module = compile(
        "import a\nimport b\nfun main() : Int = a.get(a.make()) + b.get(b.make())",
        &[(&["a"], source), (&["b"], source)],
    );
    assert!(module.funcs.iter().any(|f| f.name == "a$make"));
    assert!(module.funcs.iter().any(|f| f.name == "b$make"));
}

#[test]
fn missing_modules_and_items_are_reported() {
    let message = compile_err("import nope\nfun main() : Int = 1", &[]);
    assert!(message.contains("unknown module `nope`"), "{message}");

    let message = compile_err("import math\nuse math::missing\nfun main() : Int = 1", &[(&["math"], "fun x() : Int = 1")]);
    assert!(message.contains("no item `missing`"), "{message}");

    let message = compile_err("import math\nfun main() : Int = math.missing()", &[(&["math"], "fun x() : Int = 1")]);
    assert!(message.contains("no matching `missing`"), "{message}");
}

#[test]
fn a_local_class_allows_a_foreign_instance_head() {
    compile(
        "import math\n\
         class Local a { method : a -> Int }\n\
         instance Local (math.Tree Int) { method(x) = 1 }",
        &[(&["math"], "data Tree a = Leaf(a)")],
    );
}

#[test]
fn foreign_or_orphan_instances_are_rejected() {
    let message = compile_err("import math\ninstance Eq (math.Tree Int) { eq(x, y) = true }", &[(&["math"], "data Tree a = Leaf(a)")]);
    assert!(message.contains("orphan"), "{message}");
}

#[test]
fn loading_a_project_finds_leaf_and_package_modules() {
    let dir = std::env::temp_dir().join(format!("fern-modules-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("geo")).expect("mkdir");
    std::fs::write(dir.join("main.fern"), "import math\nimport geo::point\nfun main() : Int = math.double(geo.point.x())").expect("write main");
    std::fs::write(dir.join("math.fern"), "fun double(x : Int) : Int = x * 2").expect("write math");
    std::fs::write(dir.join("geo/point.fern"), "fun x() : Int = 21").expect("write point");

    let mut session = fern::Session::new();
    let files = session.load_entry(&dir.join("main.fern")).unwrap_or_else(|diags| panic!("load failed: {}", diags[0].msg));
    let module = session.compile(&files).unwrap_or_else(|diags| panic!("compile failed: {}", diags[0].msg));
    assert!(module.funcs.iter().any(|f| f.name == "main"));
    assert!(module.funcs.iter().any(|f| f.name == "geo$point$x"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn module_ids_reach_the_program() {
    let mut session = fern::Session::new();
    let main = session.add_file("<main>", "import math\nfun main() : Int = math.x()");
    let math = session.add_module_file("<math>", vec!["math".into()], "fun x() : Int = 1");
    let (program, diags, _) = session.program(&[session.prelude(), main, math]);
    assert!(diags.is_empty(), "{diags:?}");
    let import = program.items.iter().find(|item| matches!(item.kind, fern::ast::ItemKind::Import(_))).expect("import");
    assert_eq!(import.module, ModuleId(1));
    let x = program
        .items
        .iter()
        .find(|item| matches!(&item.kind, fern::ast::ItemKind::Fun(f) if f.name == "x"))
        .expect("math item");
    assert_eq!(x.module, ModuleId(2));
    let _ = FileId::PRELUDE;
}

#[test]
fn the_bundled_std_provides_foldable() {
    let mut session = fern::Session::new();
    let foldable = session
        .load_import(vec!["std".into(), "foldable".into()])
        .unwrap_or_else(|diags| panic!("std load failed: {}", diags[0].msg));
    assert!(std::path::Path::new(session.path(foldable)).ends_with("stdlib/std/foldable.fern"));

    let main = session.add_file(
        "<main>",
        "import std::foldable\nfun main() : Int = fold((h, b) -> h + b, 0, [1, 2, 3])",
    );
    let module = session
        .compile(&[session.prelude(), main, foldable])
        .unwrap_or_else(|diags| panic!("compile failed: {}", diags[0].msg));
    assert!(module.funcs.iter().any(|f| f.name == "main"));
}

#[test]
fn module_search_paths_are_searched_before_the_stdlib() {
    let dir = std::env::temp_dir().join(format!("fern-modules-search-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("std")).expect("mkdir");
    std::fs::write(dir.join("std/foldable.fern"), "fun marker() : Int = 7").expect("write");

    let mut session = fern::Session::new();
    session.add_module_search_path(&dir);
    let foldable = session
        .load_import(vec!["std".into(), "foldable".into()])
        .unwrap_or_else(|diags| panic!("load failed: {}", diags[0].msg));
    assert!(session.text(foldable).contains("marker"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn std_roots_map_the_std_prefix_to_their_contents() {
    let dir = std::env::temp_dir().join(format!("fern-std-root-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    std::fs::write(dir.join("only_in_test.fern"), "fun marker() : Int = 9").expect("write");

    let mut session = fern::Session::new();
    session.add_std_search_path(&dir);
    let module = session
        .load_import(vec!["std".into(), "only_in_test".into()])
        .unwrap_or_else(|diags| panic!("load failed: {}", diags[0].msg));
    assert!(std::path::Path::new(session.path(module)).ends_with("only_in_test.fern"));
    assert!(session.text(module).contains("marker"));
    let _ = std::fs::remove_dir_all(&dir);
}
