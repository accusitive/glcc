//! External API smoke test: what the backend gets from owning the
//! driver state — interner, lexer output, and `Sema` — and driving a
//! parse: declarations, bodies, layout, name resolution, and the debug
//! dump. If this stops compiling, the backend-facing surface broke.

use glcc_frontend::{
    dump::dump_translation_unit,
    lexer::{Interner, Lexer},
    parse::{ExternalDeclaration, Parser},
    sema::{BindingKind, Sema},
};

#[test]
fn backend_owns_driver_state_and_parses() {
    let mut interner = Interner::new();
    let mut sema = Sema::new();
    let src = "struct P { int x; int y; }; int f(struct P *p) { return p->x + p->y; }";
    let tokens = Lexer::new(src, &mut interner).lex_all();
    let mut parser = Parser::new(src, tokens, &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    assert_eq!(tu.external_declarations.len(), 1);
    drop(parser);

    let [ExternalDeclaration::Func(id)] = tu.external_declarations.as_slice() else {
        panic!("expected one function");
    };
    let binding = &sema.decls[id.0 as usize];
    assert_eq!(binding.kind, BindingKind::Func);
    assert!(binding.defined);
    let body = sema.bodies.get(id).expect("body recorded");
    assert_eq!(body.params.len(), 1);
    assert!(!body.block.items.is_empty());

    // Layout and name resolution through the owned interner.
    let tag = interner.get_or_intern("P");
    let struct_id = sema.lookup_tag(tag).expect("tag declared");
    let layout = sema.struct_layout(struct_id, &interner).expect("layout known");
    assert_eq!((layout.size, layout.align), (8, 4));
    let f = interner.get_or_intern("f");
    assert_eq!(sema.lookup_symbol(f), Some(*id));

    // The debug dump renders through the same public surface.
    let dump = dump_translation_unit(&sema, &interner, &tu);
    assert!(dump.contains("(member "), "{dump}");
    assert!(dump.contains("x :int"), "{dump}");
}
