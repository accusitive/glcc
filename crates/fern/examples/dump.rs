//! Dump HIR bodies, MIR functions, and verifier errors for a program.
//!
//! Usage: `cargo run -p fern --example dump -- '<source>'`

use fern::{PRELUDE_TAG, Span};

fn main() {
    let src = std::env::args().nth(1).expect("usage: dump <source>");
    let mut tokens = fern::lexer::lex(fern::prelude::SOURCE).expect("lex");
    tokens.pop();
    for token in &mut tokens {
        token.span = Span::new((token.span.lo + PRELUDE_TAG) as usize, (token.span.hi + PRELUDE_TAG) as usize);
    }
    tokens.extend(fern::lexer::lex(&src).expect("lex"));
    let program = fern::parser::parse(tokens).expect("parse");
    let mut defs = fern::defs::collect(&program).expect("defs");
    let hir = fern::check::check(&mut defs, &program).expect("check");
    println!("== HIR ==");
    for (i, b) in hir.bodies.iter().enumerate() {
        println!(
            "#{i} {:?} name={} generics={:?} params={} locals={}",
            b.key,
            b.name,
            b.generics.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            b.params.len(),
            b.locals.len()
        );
    }
    println!("closures: {}", hir.closures.len());
    match fern::lower::lower(&defs, &hir) {
        Ok(module) => {
            println!("== MIR ==");
            for f in &module.funcs {
                println!("{} : {:?} -> {:?}", f.name, f.sig.params, f.sig.ret);
            }
            for e in glcc_mir::verify::verify_module(&module) {
                println!("verify: {e}");
            }
        }
        Err(e) => println!("lower error: {}", e.msg),
    }
}
