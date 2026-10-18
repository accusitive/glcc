//! Diagnostic rendering: user spans point into the file, while prelude
//! spans (tagged by [`fern::PRELUDE_TAG`]) point into the prelude.

use fern::{Diag, PRELUDE_TAG, Span};

#[test]
fn user_spans_render_against_the_source() {
    let src = "fun main() = nope\n";
    let text = fern::render_diag(src, &Diag::new("boom", Span::new(13, 17)));
    assert!(text.contains("error: boom"), "{text}");
    assert!(text.contains(" --> 1:14"), "{text}");
    assert!(text.contains("nope"), "{text}");
    assert!(text.contains("^^^^"), "{text}");
}

#[test]
fn prelude_spans_render_against_the_prelude() {
    let off = PRELUDE_TAG as usize;
    let text = fern::render_diag("", &Diag::new("boom", Span::new(off, off + 8)));
    assert!(text.contains("error: boom"), "{text}");
    assert!(text.contains(" --> prelude:1:"), "{text}");
    assert!(text.contains("data List a"), "{text}");
}

#[test]
fn wide_spans_do_not_run_off_the_line() {
    let src = "fun main() = 1\n";
    let text = fern::render_diag(src, &Diag::new("boom", Span::new(0, 999)));
    assert!(text.contains('^'), "{text}");
    assert!(!text.contains("^^^^^^^^^^^^^^^^"), "{text}");
}
