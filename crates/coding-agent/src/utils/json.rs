//! JSON comment and trailing-comma stripping, upstream's
//! `src/utils/json.ts`.
//!
//! Upstream expresses the two passes as chained `RegExp` replaces; the
//! port runs the same two grammars through the `regex` crate, string
//! literals matched first in both so their contents survive untouched.

use regex::Regex;

/// Strip `//` line comments and trailing commas from JSON, leaving string
/// literals untouched.
///
/// # Panics
/// Never in practice: the two grammars are compile-time constants the
/// suite compiles on every run, and the capture-group reads only index
/// groups the same constants define, so the `expect` guards only fire on a
/// programming error.
#[must_use]
pub fn strip_json_comments(input: &str) -> String {
    #[expect(
        clippy::expect_used,
        reason = "the grammar is a compile-time constant; a failure is a programming error, not a runtime condition"
    )]
    let strings_or_comments = Regex::new(r#""(?:\\.|[^"\\])*"|//[^\n]*"#).expect("static pattern");
    #[expect(
        clippy::expect_used,
        reason = "the grammar is a compile-time constant; a failure is a programming error, not a runtime condition"
    )]
    let strings_or_trailing_commas =
        Regex::new(r#""(?:\\.|[^"\\])*"|,(\s*[}\]])"#).expect("static pattern");
    let without_comments = strings_or_comments.replace_all(input, |capture: &regex::Captures| {
        #[expect(
            clippy::expect_used,
            reason = "group 0 always matches in a replace_all callback"
        )]
        let matched = capture.get(0).expect("whole match").as_str();
        if matched.starts_with('"') {
            matched.to_string()
        } else {
            String::new()
        }
    });
    strings_or_trailing_commas
        .replace_all(&without_comments, |capture: &regex::Captures| {
            #[expect(
                clippy::expect_used,
                reason = "group 0 always matches in a replace_all callback"
            )]
            let matched = capture.get(0).expect("whole match").as_str();
            if matched.starts_with('"') {
                matched.to_string()
            } else {
                #[expect(
                    clippy::expect_used,
                    reason = "the trailing-comma alternative is the only non-string arm and always carries group 1"
                )]
                let tail = capture
                    .get(1)
                    .expect("the trailing-comma alternative always carries the tail")
                    .as_str()
                    .to_string();
                tail
            }
        })
        .into_owned()
}
