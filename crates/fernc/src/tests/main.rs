use super::*;

fn parse(args: &[&str]) -> Options {
    let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
    parse_args(&args).expect("parses").expect("not a help request")
}

#[test]
fn emit_flags_pick_the_dump_and_the_last_one_wins() {
    assert_eq!(parse(&["a.fern"]).emit, Emit::Object);
    assert_eq!(parse(&["--emit=mir", "a.fern"]).emit, Emit::Mir);
    assert_eq!(parse(&["--emit=full-mir", "a.fern"]).emit, Emit::FullMir);
    assert_eq!(parse(&["--emit-mir", "a.fern"]).emit, Emit::Mir);
    assert_eq!(parse(&["--emit-full-mir", "a.fern"]).emit, Emit::FullMir);
    assert_eq!(parse(&["--emit-full-mir", "--emit-mir", "a.fern"]).emit, Emit::Mir);
    assert_eq!(parse(&["--emit-mir", "--emit-full-mir", "a.fern"]).emit, Emit::FullMir);
    assert_eq!(parse(&["--emit-full-mir", "--emit=object", "a.fern"]).emit, Emit::Object);
}

#[test]
fn unknown_emit_kind_is_an_error() {
    let args = vec!["--emit=nope".to_string(), "a.fern".to_string()];
    let err = match parse_args(&args) {
        Err(err) => err,
        Ok(_) => panic!("unknown emit kind must fail"),
    };
    assert!(err.contains("nope"), "{err}");
}
