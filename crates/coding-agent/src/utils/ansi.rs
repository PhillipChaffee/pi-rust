//! ANSI escape stripping, upstream's `src/utils/ansi.ts`.
//!
//! Portions restated from the `strip-ansi` package (MIT, (c) Sindre
//! Sorhus), which upstream's file credits: the OSC/CSI grammar is the same
//! pattern the `regex` crate compiles, so the generated-compatibility
//! expectations port unchanged.
//!
//! Upstream's `TypeError` for non-string inputs has no counterpart — Rust's
//! `&str` cannot be anything else.

use regex::Regex;

/// The OSC and CSI sequence grammar, upstream's `ansiRegex` body.
///
/// Valid string terminator sequences are BEL, ESC backslash, and 0x9c; the
/// OSC branch matches non-greedily until the first terminator, the CSI
/// branch takes optional intermediates and parameters (`;` and `:`) before
/// the final byte.
const SEQUENCE_PATTERN: &str = concat!(
    r"(?:\x1B\][\s\S]*?(?:\x07|\x1B\x5C|\u{9C}))",
    "|",
    r"[\x1B\u{9B}][\[\]()#;?]*(?:\d{1,4}(?:[;:]\d{0,4})*)?[\dA-PR-TZcf-nq-uy=><~]"
);

fn sequence_regex() -> Regex {
    #[expect(
        clippy::expect_used,
        reason = "the grammar is a compile-time constant; a failure is a programming error, not a runtime condition"
    )]
    Regex::new(SEQUENCE_PATTERN).expect("static pattern")
}

/// Strip ANSI escape sequences from `value`.
///
/// The fast path returns the input untouched unless an ESC (7-bit) or CSI
/// (8-bit) introducer is present, upstream's pre-check.
#[must_use]
pub fn strip_ansi(value: &str) -> String {
    if !value.contains('\x1b') && !value.contains('\u{9b}') {
        return value.to_string();
    }
    sequence_regex().replace_all(value, "").into_owned()
}
