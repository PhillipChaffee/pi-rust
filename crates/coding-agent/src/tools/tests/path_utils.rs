//! Boundary tests for the tool path resolution, upstream's
//! `path-utils.ts` — the resolved path, the macOS variant ladder, and the
//! option set the resolvers run.

#![expect(
    clippy::unwrap_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use crate::tools::path_utils::{expand_path, resolve_read_path, resolve_to_cwd};

#[test]
fn resolve_to_cwd_joins_relative_paths() {
    let resolved = resolve_to_cwd("a/b.txt", "/base/dir").unwrap();
    assert_eq!(resolved, "/base/dir/a/b.txt");
    assert_eq!(resolve_to_cwd("/abs/x.txt", "/base").unwrap(), "/abs/x.txt");
    assert_eq!(resolve_to_cwd("x/../y", "/base").unwrap(), "/base/y");
}

#[test]
fn resolve_to_cwd_expands_tilde() {
    let resolved = resolve_to_cwd("~/notes.txt", "/base").unwrap();
    assert!(resolved.ends_with("/notes.txt"), "{resolved}");
    assert!(resolved.starts_with('/'));
}

#[test]
fn resolve_to_cwd_normalizes_unicode_spaces_and_strips_at() {
    // The narrow no-break space the macOS screenshot paths carry.
    let resolved = resolve_to_cwd("a\u{202F}b.txt", "/base").unwrap();
    assert_eq!(resolved, "/base/a b.txt");
    let stripped = resolve_to_cwd("@file.txt", "/base").unwrap();
    assert_eq!(stripped, "/base/file.txt");
}

#[test]
fn expand_path_normalizes_in_place() {
    // normalizePath is input-reshaping only: no dot collapse, no join.
    assert_eq!(expand_path("/a/./b//c.txt").unwrap(), "/a/./b//c.txt");
    // The dot collapse rides resolve's nodeResolvePath step.
    assert_eq!(
        resolve_to_cwd("a/./b//c.txt", "/base").unwrap(),
        "/base/a/b/c.txt"
    );
}

fn make_tree(dir: &std::path::Path) {
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    std::fs::write(dir.join("plain.txt"), "plain").unwrap();
    std::fs::write(dir.join("shot 10.30.45\u{202F}AM.txt"), "am-nnbsp").unwrap();
    std::fs::write(dir.join("cafe\u{301}.txt"), "nfd").unwrap();
    std::fs::write(dir.join("captur\u{2019}e.txt"), "curly").unwrap();
}

#[test]
fn read_path_falls_back_to_the_macos_variants() {
    let dir = tempfile::tempdir().unwrap();
    make_tree(dir.path());
    let base = dir.path().to_string_lossy().into_owned();

    // Exact hit first.
    assert_eq!(
        resolve_read_path("plain.txt", &base).unwrap(),
        format!("{base}/plain.txt")
    );
    // The AM/PM narrow no-break space variant (only the space before AM/PM
    // becomes narrow, the regex replacement's single-site reach).
    assert!(
        resolve_read_path("shot 10.30.45 AM.txt", &base)
            .unwrap()
            .ends_with("shot 10.30.45\u{202F}AM.txt")
    );
    // The NFD variant for composed user input. APFS is
    // normalization-insensitive, so the composed probe already finds the
    // decomposed file and the returned form is whichever the volume stores;
    // pin existence.
    {
        let resolved = resolve_read_path("caf\u{e9}.txt", &base).unwrap();
        assert!(std::path::Path::new(&resolved).exists(), "{resolved}");
    }
    // The curly-quote variant.
    assert!(
        resolve_read_path("captur'e.txt", &base)
            .unwrap()
            .ends_with("captur\u{2019}e.txt")
    );
    // A miss returns the resolved path unchanged.
    assert_eq!(
        resolve_read_path("missing.txt", &base).unwrap(),
        format!("{base}/missing.txt")
    );
}
