//! Local-time formatting for the dynamic `__DATE__`/`__TIME__`/
//! `__TIMESTAMP__` macros. clang formats these in local time; the
//! non-unix fallback stays UTC (there is no `localtime`).

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const WEEKDAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

/// Broken-down time: year, month (1-12), day, hour, minute, second,
/// weekday (0 = Sunday).
struct Parts {
    year: i64,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    weekday: u32,
}

/// `__DATE__` and `__TIME__` at `now`, in clang's formats:
/// `"Mmm dd yyyy"` (day space-padded) and `"hh:mm:ss"`.
pub fn date_time(now: SystemTime) -> (String, String) {
    let secs = now.duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let p = parts(secs);
    (format!("{} {:2} {}", MONTHS[p.month as usize - 1], p.day, p.year), format!("{:02}:{:02}:{:02}", p.hour, p.minute, p.second))
}

/// `__TIMESTAMP__` of `path`: ctime-style local time
/// (`"Ddd Mmm dd hh:mm:ss yyyy"`). A file whose mtime cannot be read
/// formats the epoch, like clang on a stream without a path.
pub fn file_timestamp(path: &Path) -> String {
    let secs = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let p = parts(secs);
    format!(
        "{} {} {:2} {:02}:{:02}:{:02} {}",
        WEEKDAYS[p.weekday as usize], MONTHS[p.month as usize - 1], p.day, p.hour, p.minute, p.second, p.year
    )
}

#[cfg(unix)]
fn parts(secs: i64) -> Parts {
    let t = secs as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    if unsafe { libc::localtime_r(&t, &mut tm) }.is_null() {
        return utc_parts(secs);
    }
    Parts {
        year: tm.tm_year as i64 + 1900,
        month: (tm.tm_mon + 1) as u32,
        day: tm.tm_mday as u32,
        hour: tm.tm_hour as u32,
        minute: tm.tm_min as u32,
        second: tm.tm_sec as u32,
        weekday: tm.tm_wday as u32,
    }
}

#[cfg(not(unix))]
fn parts(secs: i64) -> Parts {
    utc_parts(secs)
}

/// Civil date from seconds since the Unix epoch (Howard Hinnant's
/// `civil_from_days`), used when no local-time API is available.
fn utc_parts(secs: i64) -> Parts {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    Parts {
        year: yoe + era * 400 + i64::from(month <= 2),
        month: month as u32,
        day: (doy - (153 * mp + 2) / 5 + 1) as u32,
        hour: (rem / 3_600) as u32,
        minute: (rem / 60 % 60) as u32,
        second: (rem % 60) as u32,
        weekday: (days + 4).rem_euclid(7) as u32,
    }
}

#[cfg(test)]
mod tests {
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
}
