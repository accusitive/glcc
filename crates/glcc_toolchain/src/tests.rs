use super::*;

#[test]
fn versions_order_numerically_not_lexically() {
    assert!(parse_version("10.0.22621.0") > parse_version("10.0.19041.0"));
    assert!(parse_version("14.34.31933") > parse_version("14.9.31933"));
    assert!(parse_version("2022") > parse_version("2019"));
    assert_eq!(parse_version("not-a-version"), None);
    assert_eq!(parse_version(""), None);
}
