//! The minimatch full-match subset the model scoping and file tools share,
//! upstream's `minimatch` npm dependency at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatement: the matcher compiles a glob into a `regex` pattern
//! with minimatch's default `noglobstar`-off semantics — `*` and `?` never
//! cross `/`, a lone `**` segment matches any number of segments (a
//! non-trailing globstar swallows its own separator, so `**/c` also matches
//! the bare `c`), `{a,b}` braces expand (nested braces included), `[...]`
//! classes carry ranges and `!` negation, and `nocase` lowercases both sides
//! before matching. Segments match dot-leading names — minimatch's `dot`
//! option stays off in the ported callers and the file tools need fd's
//! dot-blind `--hidden` behavior, so `*` matches `.env`. The extglob forms
//! (`?(...)`, `*(...)`, `+(...)`, `@(...)`, `!(...)`) are not part of the
//! subset the ported callers use and compile as literals; a pattern carrying
//! them fails to match rather than guessing.

use regex::Regex;

/// One expanded alternative of a brace expression.
fn expand_braces(pattern: &str) -> Vec<String> {
    // Find the first top-level `{`...`}` group (nesting-aware) and expand
    // the comma-separated alternatives recursively.
    let mut depth = 0usize;
    let mut open = None;
    for (index, ch) in pattern.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    open = Some(index);
                }
                depth += 1;
            }
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    if let Some(open) = open {
                        let prefix = &pattern[..open];
                        let body = &pattern[open + 1..index];
                        let suffix = &pattern[index + 1..];
                        let mut expanded = Vec::new();
                        for alternative in split_top_level_commas(body) {
                            for combination in
                                expand_braces(&format!("{prefix}{alternative}{suffix}"))
                            {
                                expanded.push(combination);
                            }
                        }
                        return expanded;
                    }
                    open = None;
                }
            }
            _ => {}
        }
    }
    vec![pattern.to_owned()]
}

/// Split on commas at the expression's own nesting depth.
fn split_top_level_commas(body: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (index, ch) in body.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(&body[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    parts.push(&body[start..]);
    parts
}

/// Compile one glob segment into a regex fragment, upstream's segment
/// grammar: `*` and `?` stop at `/`, `[...]` classes carry ranges and `!`
/// negation.
fn compile_segment(segment: &str) -> String {
    let mut regex = String::new();
    let mut chars = segment.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '*' => {
                regex.push_str("[^/]*");
            }
            '?' => {
                regex.push_str("[^/]");
            }
            '[' => {
                let mut class = String::from("[");
                let negated = chars.peek() == Some(&'!');
                if negated {
                    chars.next();
                    class.push('^');
                }
                let mut closed = false;
                let mut first = true;
                for class_ch in chars.by_ref() {
                    if class_ch == ']' && !first {
                        closed = true;
                        break;
                    }
                    first = false;
                    match class_ch {
                        '\\' | '^' | '$' | '.' | '|' | '+' | '(' | ')' => {
                            class.push('\\');
                            class.push(class_ch);
                        }
                        ']' if first => class.push_str("\\]"),
                        _ => class.push(class_ch),
                    }
                }
                if closed {
                    regex.push_str(&class);
                    regex.push(']');
                } else {
                    // An unterminated class is a literal `[` followed by the
                    // literal remainder, minimatch's behavior.
                    regex.push_str("\\[");
                    let Some(open) = segment.rfind('[') else {
                        continue;
                    };
                    regex.push_str(&regex::escape(&segment[open + 1..]));
                }
            }
            '\\' => {
                if let Some(next) = chars.next() {
                    regex.push_str(&regex::escape(&next.to_string()));
                }
            }
            _ => regex.push_str(&regex::escape(&ch.to_string())),
        }
    }
    regex
}

/// Compile a whole glob into an anchored regex, upstream's pattern
/// compilation: a lone `**` segment matches any number of segments. A
/// non-trailing globstar compiles to `(?:[^/]*/)*` — zero or more
/// segment-and-separator groups, the segment itself allowed empty so the
/// doubled-slash spelling matches too — so `**/c` matches the bare `c` the
/// way minimatch's globstar does; a trailing one keeps the separator
/// optional so `a/**` matches both `a` and `a/b`.
fn compile_glob(pattern: &str) -> Option<Regex> {
    let segments: Vec<&str> = pattern.split('/').collect();
    let mut regex = String::from("^(?:");
    for (index, segment) in segments.iter().enumerate() {
        let is_last = index + 1 == segments.len();
        if *segment == "**" {
            if is_last {
                if index == 0 {
                    regex.push_str(".*");
                } else {
                    regex.push_str("(?:/.*)?");
                }
            } else {
                regex.push_str("(?:[^/]*/)*");
            }
            continue;
        }
        regex.push_str(&compile_segment(segment));
        let next_is_trailing_globstar =
            !is_last && segments[index + 1] == "**" && index + 2 == segments.len();
        if !is_last && !next_is_trailing_globstar {
            regex.push('/');
        }
    }
    regex.push_str(")$");
    Regex::new(&regex).ok()
}

/// Whether the candidate matches the glob, upstream's
/// `minimatch(candidate, pattern, options)`.
#[must_use]
pub fn matches(candidate: &str, pattern: &str, nocase: bool) -> bool {
    let (candidate, pattern) = if nocase {
        (candidate.to_lowercase(), pattern.to_lowercase())
    } else {
        (candidate.to_owned(), pattern.to_owned())
    };
    expand_braces(&pattern)
        .iter()
        .any(|expanded| compile_glob(expanded).is_some_and(|regex| regex.is_match(&candidate)))
}

/// Whether the candidate matches any of the globs.
#[must_use]
pub fn matches_any(candidate: &str, patterns: &[String], nocase: bool) -> bool {
    patterns
        .iter()
        .any(|pattern| matches(candidate, pattern, nocase))
}

#[cfg(test)]
mod tests {
    use super::matches;

    #[test]
    fn star_does_not_cross_slashes() {
        assert!(matches("claude-sonnet-4", "*sonnet*", true));
        assert!(!matches("anthropic/claude-sonnet-4", "*sonnet*", true));
        assert!(matches("anthropic/claude-3", "anthropic/*", true));
        assert!(!matches("anthropic/a/b", "anthropic/*", true));
    }

    #[test]
    fn globstar_crosses_segments() {
        assert!(matches("anthropic/a/b", "anthropic/**", true));
        assert!(matches("anthropic", "anthropic/**", true));
        assert!(matches("a/b/c", "**/c", true));
        // A non-trailing globstar also matches zero segments, separator
        // included, so a bare name matches a `**/`-prefixed pattern.
        assert!(matches("root.txt", "**/*.txt", true));
        assert!(matches("c", "**/c", true));
        assert!(matches(
            "src/foo/bar/example.spec.ts",
            "src/**/*.spec.ts",
            true
        ));
        assert!(!matches(
            "some/parent/child/test.spec.ts",
            "src/**/*.spec.ts",
            true
        ));
    }

    #[test]
    fn segments_match_dot_leading_names() {
        // The ported callers need fd's dot-blind `--hidden` behavior, so a
        // `*` segment matches a dot-leading name (minimatch's `dot` option
        // would refuse; it stays off here).
        assert!(matches(".env", "*", false));
        assert!(matches(".secret/hidden.txt", "**/*.txt", false));
        assert!(matches("a/.b/c", "a/**/c", false));
    }

    #[test]
    fn braces_expand() {
        assert!(matches("openai/gpt", "{openai,anthropic}/*", true));
        assert!(matches("anthropic/claude", "{openai,anthropic}/*", true));
        assert!(!matches("google/gemini", "{openai,anthropic}/*", true));
    }

    #[test]
    fn classes_match_and_negate() {
        assert!(matches("model-a", "model-[abc]", false));
        assert!(!matches("model-d", "model-[abc]", false));
        assert!(matches("model-d", "model-[!abc]", false));
        assert!(matches("id-20240122", "id-[0-9]*", false));
        // An unterminated class is a literal bracket.
        assert!(!matches("id-x", "id-[abc", false));
        assert!(matches("id-[abc", "id-[abc", false));
    }

    #[test]
    fn question_mark_stops_at_slash() {
        assert!(matches("ab", "a?", false));
        assert!(!matches("a/b", "a?", false));
    }

    #[test]
    fn nocase_lowercases_both_sides() {
        assert!(matches("MiniMax-M2.7", "minimax*", true));
        assert!(!matches("MiniMax-M2.7", "minimax*", false));
    }
}

#[cfg(test)]
mod matches_any_tests {
    use super::matches_any;

    #[test]
    fn any_pattern_list_matches_on_the_first_hit() {
        assert!(matches_any(
            "openai/gpt",
            &["anthropic/*".to_owned(), "openai/*".to_owned()],
            false
        ));
        assert!(!matches_any(
            "google/gemini",
            &["anthropic/*".to_owned(), "openai/*".to_owned()],
            false
        ));
        assert!(matches_any("OPENAI/GPT", &["openai/*".to_owned()], true));
    }
}
