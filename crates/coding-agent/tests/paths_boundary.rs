//! Boundary vectors for the path resolver, upstream's `resolvePath` slice:
//! the POSIX-resolve table and the tilde expansion. The resolve expectations
//! are the outputs of upstream's Node `path.resolve` at the pin, run for
//! this table; the relative-input expectations compose off the process cwd,
//! which is the base `resolve_path` supplies upstream.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use pi_coding_agent::utils::paths::{expand_tilde, resolve_path};

fn process_cwd() -> String {
    std::env::current_dir()
        .expect("current directory resolves")
        .to_string_lossy()
        .into_owned()
}

fn process_home() -> String {
    std::env::home_dir()
        .expect("home directory resolves")
        .to_string_lossy()
        .into_owned()
}

#[test]
fn resolve_normalizes_absolute_inputs_like_node() {
    // outputs run from upstream's path.resolve at the pin
    for (input, expected) in [
        ("/Users/foo/bar", "/Users/foo/bar"),
        ("/tmp/a b", "/tmp/a b"),
        ("/", "/"),
        ("/a//b///c", "/a/b/c"),
        ("/a/b/", "/a/b"),
        ("//a", "/a"),
        ("///a", "/a"),
        ("/a/b/../c", "/a/c"),
        ("/../a", "/a"),
        ("/a/..", "/"),
        ("/a/b/../../c", "/c"),
        ("/../../a", "/a"),
        ("/.", "/"),
        ("/./", "/"),
        ("/..", "/"),
        ("/a/.../b", "/a/.../b"),
        ("/a/..b/c", "/a/..b/c"),
        ("/a/b/./.././c", "/a/c"),
        ("/a b/c", "/a b/c"),
    ] {
        assert_eq!(
            resolve_path(input, "/unused-base", "/home/u"),
            expected,
            "{input}"
        );
    }
}

#[test]
fn resolve_joins_relative_inputs_onto_the_process_cwd() {
    let cwd = process_cwd();

    for (input, expected) in [
        ("a/./b", format!("{cwd}/a/b")),
        ("", cwd.clone()),
        (".", cwd.clone()),
        (
            "a/../..",
            std::env::current_dir() // the lexical parent
                .expect("current directory resolves")
                .parent()
                .expect("cwd has a parent")
                .to_string_lossy()
                .into_owned(),
        ),
    ] {
        assert_eq!(resolve_path(input, &cwd, "/home/u"), expected, "{input}");
    }
}

#[test]
fn resolve_joins_relative_inputs_onto_the_base_dir() {
    for (input, base, expected) in [
        ("a", "/b/c", "/b/c/a"),
        ("..", "/b/c", "/b"),
        ("../x", "/b/c", "/b/x"),
        ("a", "/b/c/", "/b/c/a"),
        ("/a", "/b/c", "/a"),
        ("", "/b/c", "/b/c"),
        (".", "/b/c", "/b/c"),
        ("a/b/../c", "/base", "/base/a/c"),
    ] {
        assert_eq!(
            resolve_path(input, base, "/home/u"),
            expected,
            "{input} {base}"
        );
    }
}

#[test]
fn resolve_expands_a_tilde_input_before_resolving() {
    let home = process_home();

    // the expanded home is absolute, so the base never applies
    assert_eq!(resolve_path("~", "/unused-base", &home), home);
    assert_eq!(
        resolve_path("~/x", "/unused-base", &home),
        format!("{home}/x")
    );
    // a near-miss tilde passes through the expansion and joins the base
    assert_eq!(resolve_path("~x", "/b/c", &home), "/b/c/~x");
}

#[test]
fn expand_tilde_matches_upstreams_normalize_path_slice() {
    for (input, expected) in [
        ("~", "/home/u"),
        ("~/", "/home/u"),
        ("~//x", "/home/u/x"),
        ("~/../x", "/home/x"),
        ("~/..", "/home"),
        ("~/x/../y", "/home/u/y"),
        ("~x", "~x"),
        ("~~/x", "~~/x"),
        ("a~", "a~"),
        ("", ""),
        ("~/x", "/home/u/x"),
    ] {
        assert_eq!(expand_tilde(input, "/home/u"), expected, "{input}");
    }
}
