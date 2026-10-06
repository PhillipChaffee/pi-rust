//! The git-URL suite, upstream's `test/git-ssh-url.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, over the hand-ported
//! `hosted-git-info` grammar, plus boundary vectors for the shorthand
//! forms the upstream suite does not cover.

#![expect(clippy::panic, reason = "tests assert by panicking")]

use pi_coding_agent::utils::git::GitSource;
use pi_coding_agent::utils::git::parse_git_url;

fn expect_source(source: &str, host: &str, path: &str, repo: &str) -> GitSource {
    let result = parse_git_url(source);
    let result = result.unwrap_or_else(|| panic!("{source:?} parses"));
    assert_eq!(result.host, host, "{source:?}");
    assert_eq!(result.path, path, "{source:?}");
    assert_eq!(result.repo, repo, "{source:?}");
    assert!(result.pinned == result.r#ref.is_some(), "{source:?}");
    result
}

// === protocol URLs (accepted without git: prefix) ===========================

#[test]
fn parses_https_url() {
    let result = expect_source(
        "https://github.com/user/repo",
        "github.com",
        "user/repo",
        "https://github.com/user/repo",
    );
    assert_eq!(result.r#ref, None);
    assert!(!result.pinned);
}

#[test]
fn parses_ssh_url() {
    expect_source(
        "ssh://git@github.com/user/repo",
        "github.com",
        "user/repo",
        "ssh://git@github.com/user/repo",
    );
}

#[test]
fn parses_protocol_url_with_ref() {
    let result = expect_source(
        "https://github.com/user/repo@v1.0.0",
        "github.com",
        "user/repo",
        "https://github.com/user/repo",
    );
    assert_eq!(result.r#ref.as_deref(), Some("v1.0.0"));
    assert!(result.pinned);
}

// === shorthand URLs (accepted only with git: prefix) ========================

#[test]
fn parses_git_at_host_path_with_git_prefix() {
    expect_source(
        "git:git@github.com:user/repo",
        "github.com",
        "user/repo",
        "git@github.com:user/repo",
    );
}

#[test]
fn parses_host_path_shorthand_with_git_prefix() {
    expect_source(
        "git:github.com/user/repo",
        "github.com",
        "user/repo",
        "https://github.com/user/repo",
    );
}

#[test]
fn parses_shorthand_with_ref_and_git_prefix() {
    let result = expect_source(
        "git:git@github.com:user/repo@v1.0.0",
        "github.com",
        "user/repo",
        "git@github.com:user/repo",
    );
    assert_eq!(result.r#ref.as_deref(), Some("v1.0.0"));
    assert!(result.pinned);
}

#[test]
fn rejects_unsafe_git_install_path_inputs() {
    for source in [
        "git:git@evil.example:../../victim/repo",
        "https://evil.example/..%2F..%2Fvictim/repo",
        "https://evil.example/..%2F..%2Fvictim/repo%",
        "git:git@evil.example:/absolute/repo",
        "git:git@evil.example:user\\repo/name",
        "git:git@evil.example:user/repo\0name",
    ] {
        assert_eq!(parse_git_url(source), None, "{source:?}");
    }
}

// === unsupported without git: prefix ========================================

#[test]
fn rejects_git_at_host_path_without_git_prefix() {
    assert_eq!(parse_git_url("git@github.com:user/repo"), None);
}

#[test]
fn rejects_host_path_shorthand_without_git_prefix() {
    assert_eq!(parse_git_url("github.com/user/repo"), None);
}

#[test]
fn rejects_user_repo_shorthand() {
    assert_eq!(parse_git_url("user/repo"), None);
}

// === boundary vectors over the hosted grammar ===============================

#[test]
fn parses_the_gitlab_and_bitbucket_domains() {
    expect_source(
        "https://gitlab.com/user/repo",
        "gitlab.com",
        "user/repo",
        "https://gitlab.com/user/repo",
    );
    expect_source(
        "https://bitbucket.org/user/repo.git",
        "bitbucket.org",
        "user/repo",
        "https://bitbucket.org/user/repo.git",
    );
}

#[test]
fn strips_the_git_suffix_from_hosted_projects() {
    expect_source(
        "https://github.com/user/repo.git",
        "github.com",
        "user/repo",
        "https://github.com/user/repo.git",
    );
}

#[test]
fn carries_the_fragment_committish_through_the_hosted_grammar() {
    // The ref-carrying URL keeps its fragment in `repo`: splitRef only
    // rebuilds the repo string for the `@`-ref form, upstream's
    // `{ repo: url }` passthrough.
    let result = expect_source(
        "https://github.com/user/repo#v1.0.0",
        "github.com",
        "user/repo",
        "https://github.com/user/repo#v1.0.0",
    );
    assert_eq!(result.r#ref.as_deref(), Some("v1.0.0"));
    assert!(result.pinned);
}

#[test]
fn generic_git_urls_keep_their_protocol_and_reject_bare_hosts() {
    // The generic (non-hosted) parser requires a dot in the host or
    // localhost.
    expect_source(
        "git:myhost.example/user/repo",
        "myhost.example",
        "user/repo",
        "https://myhost.example/user/repo",
    );
    assert_eq!(parse_git_url("git:nodot/user/repo"), None);
    // localhost is the one dot-less host the generic parser accepts.
    expect_source(
        "git:localhost/user/repo",
        "localhost",
        "user/repo",
        "https://localhost/user/repo",
    );
}

#[test]
fn git_prefix_accepts_the_shortcut_spelling() {
    // The `github:user/repo` shortcut rides the hosted shorthand grammar.
    // The repo string carries upstream's https-prefix rule verbatim — the
    // prefix lands on the shorthand itself, which stays a hostless URL;
    // only host and path normalize.
    expect_source(
        "git:github:user/repo",
        "github.com",
        "user/repo",
        "https://github:user/repo",
    );
}
