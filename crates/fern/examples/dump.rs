//! Dump HIR bodies, MIR functions, and verifier errors for a program.
//!
//! Usage: `cargo run -p fern --example dump -- '<source>'`

fn main() {
    let src = std::env::args().nth(1).expect("usage: dump <source>");
    let mut session = fern::Session::new();
    let file = session.add_file("<input>", src);
    let (program, diags, _) = session.program(&[session.prelude(), file]);
    for diag in &diags {
        eprintln!("{}", session.render(diag));
    }
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
