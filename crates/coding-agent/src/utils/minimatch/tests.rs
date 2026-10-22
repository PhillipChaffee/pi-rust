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
