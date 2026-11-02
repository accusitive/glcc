use super::*;

#[test]
fn std_spellings_parse() {
    for (text, want) in [
        ("c89", Std::C89),
        ("c90", Std::C89),
        ("iso9899:1990", Std::C89),
        ("iso9899:199409", Std::C89),
        ("c99", Std::C99),
        ("c9x", Std::C99),
        ("c11", Std::C11),
        ("c1x", Std::C11),
        ("c17", Std::C17),
        ("c18", Std::C17),
        ("c2x", Std::C23),
        ("c23", Std::C23),
        ("iso9899:2024", Std::C23),
        ("c2y", Std::C2y),
        ("gnu89", Std::Gnu89),
        ("gnu17", Std::Gnu17),
        ("gnu2x", Std::Gnu23),
        ("gnu2y", Std::Gnu2y),
    ] {
        assert_eq!(Std::parse(text), Some(want), "{text}");
    }
    assert_eq!(Std::parse("c1y"), None);
    assert_eq!(Std::parse(""), None);
    assert_eq!(Std::parse("iso9899:1991"), None);
}

#[test]
fn std_flags_roundtrip_through_parse() {
    for std in [
        Std::C89,
        Std::C99,
        Std::C11,
        Std::C17,
        Std::C23,
        Std::C2y,
        Std::Gnu89,
        Std::Gnu99,
        Std::Gnu11,
        Std::Gnu17,
        Std::Gnu23,
        Std::Gnu2y,
    ] {
        let flag = std.flag();
        assert_eq!(Std::parse(flag.trim_start_matches("-std=")), Some(std), "{flag}");
    }
}

#[test]
fn levels_and_gnu_flags_agree() {
    assert_eq!(Std::C89.level(), 0);
    assert_eq!(Std::Gnu17.level(), Std::C17.level());
    assert_eq!(Std::C23.level(), 4);
    assert_eq!(Std::C2y.level(), 5);
    assert!(Std::Gnu11.is_gnu());
    assert!(Std::Gnu2y.is_gnu());
    assert!(!Std::C11.is_gnu());
    assert!(LangOpts::default().standard_includes(Std::C11));
    assert!(
        !LangOpts {
            std: Std::C89,
            ..LangOpts::default()
        }
        .standard_includes(Std::C11)
    );
}
