use super::*;

#[test]
fn utc_civil_conversion() {
    let p = utc_parts(0);
    assert_eq!((p.year, p.month, p.day, p.weekday), (1970, 1, 1, 4));
    let p = utc_parts(1_700_000_000);
    assert_eq!((p.year, p.month, p.day, p.weekday), (2023, 11, 14, 2));
    let p = utc_parts(951_782_400);
    assert_eq!((p.year, p.month, p.day, p.weekday), (2000, 2, 29, 2));
}

#[test]
fn date_time_shape() {
    let (d, t) = date_time(UNIX_EPOCH);
    assert_eq!(d.len(), 11);
    assert_eq!(t.len(), 8);
    assert_eq!(t.as_bytes()[2], b':');
}
