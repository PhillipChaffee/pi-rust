#![expect(
    clippy::expect_used,
    reason = "the unit tests pin changelog parsing; an unexpected result panics the test by design"
)]

use super::*;

#[test]
fn parses_entries_between_headers() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("CHANGELOG.md");
    std::fs::write(
        &path,
        "# Title\n\n## [1.2.3] - date\n\nFirst body.\n\n## [0.9.0]\n\nOlder.\ntrailing line\n",
    )
    .expect("write");
    let entries = parse_changelog(&path);
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].major, 1);
    assert_eq!(entries[0].content, "## [1.2.3] - date\n\nFirst body.");
    assert_eq!(entries[1].content, "## [0.9.0]\n\nOlder.\ntrailing line");
}

#[test]
fn resets_on_unparseable_headers_and_missing_files() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("CHANGELOG.md");
    std::fs::write(&path, "## [1.0.0]\n\nA\n\n## Nope\n\nB\n").expect("write");
    let entries = parse_changelog(&path);
    assert_eq!(entries.len(), 1);
    assert!(parse_changelog(&dir.path().join("absent.md")).is_empty());
}

#[test]
fn normalizes_legacy_repo_and_floating_refs() {
    assert_eq!(
        normalize_changelog_links(
            "[x](https://github.com/badlogic/pi-mono/blob/main/docs/a.md)",
            "1.2.3"
        ),
        "[x](https://github.com/earendil-works/pi/blob/v1.2.3/docs/a.md)"
    );
    assert_eq!(
        normalize_changelog_links(
            "[x](https://github.com/earendil-works/pi/tree/master/packages/coding-agent)",
            "1.2.3"
        ),
        "[x](https://github.com/earendil-works/pi/tree/v1.2.3/packages/coding-agent)"
    );
}

#[test]
fn routes_local_paths_to_blob_or_tree() {
    let out = normalize_changelog_links("[x](docs/a.md)", "1.2.3");
    assert_eq!(
        out,
        "[x](https://github.com/earendil-works/pi/blob/v1.2.3/packages/coding-agent/docs/a.md)"
    );
    let dir_out = normalize_changelog_links("[x](docs/)", "1.2.3");
    assert_eq!(
        dir_out,
        "[x](https://github.com/earendil-works/pi/tree/v1.2.3/packages/coding-agent/docs/)"
    );
    // A basename without a dot routes as a tree even without a slash.
    let pkg_out = normalize_changelog_links("[x](skills)", "1.2.3");
    assert_eq!(
        pkg_out,
        "[x](https://github.com/earendil-works/pi/tree/v1.2.3/packages/coding-agent/skills)"
    );
}

#[test]
fn leaves_anchors_urls_and_escaping_paths_untouched() {
    assert_eq!(
        normalize_changelog_links("[x](#section)", "1.2.3"),
        "[x](#section)"
    );
    assert_eq!(
        normalize_changelog_links("[x](https://example.com/a.md)", "1.2.3"),
        "[x](https://example.com/a.md)"
    );
    // A climb that folds back inside the repository rewrites; the fold
    // above the repository root is what stays untouched (upstream's
    // `resolveRepositoryPath` rejects only `joined.startsWith("../")`).
    assert_eq!(
        normalize_changelog_links("[x](../../outside.md)", "1.2.3"),
        "[x](https://github.com/earendil-works/pi/blob/v1.2.3/outside.md)"
    );
    assert_eq!(
        normalize_changelog_links("[x](../../../outside.md)", "1.2.3"),
        "[x](../../../outside.md)"
    );
    // The fragment and query ride the rewritten URL verbatim.
    assert_eq!(
        normalize_changelog_links("[x](docs/a.md?x=1#frag)", "1.2.3"),
        "[x](https://github.com/earendil-works/pi/blob/v1.2.3/packages/coding-agent/docs/a.md?x=1#frag)"
    );
}

#[test]
fn encodes_spaces_and_keeps_reserved_bytes() {
    // The encoding contract, ECMA-262 `encodeURI`: the space escapes,
    // the reserved bytes ride through.
    assert_eq!(encode_uri("docs/my file.md"), "docs/my%20file.md");
    // At the markdown level the inline-link regex's target group
    // excludes whitespace (upstream's `[^\s)]+`), so a space-bearing
    // link splits: the "docs/my" prefix rewrites (dotless basename →
    // tree route) and the " file.md)" suffix rides verbatim.
    assert_eq!(
        normalize_changelog_links("[x](docs/my file.md)", "1.2.3"),
        "[x](https://github.com/earendil-works/pi/tree/v1.2.3/packages/coding-agent/docs/my file.md)"
    );
}

#[test]
fn compares_and_filters_entries() {
    let one = ChangelogEntry {
        major: 1,
        minor: 2,
        patch: 3,
        content: String::new(),
    };
    let two = ChangelogEntry {
        major: 1,
        minor: 2,
        patch: 4,
        content: String::new(),
    };
    assert_eq!(compare_versions(&two, &one), std::cmp::Ordering::Greater);
    let entries = vec![one.clone(), two.clone()];
    assert_eq!(get_new_entries(&entries, "1.2.3"), vec![&two]);
    // `Number("v1")` and the missing components both land at zero, so
    // the fallback baseline (0,2,2) and (0,0,0) leave both entries newer.
    assert_eq!(get_new_entries(&entries, "v1.2.2"), vec![&one, &two]);
    assert_eq!(get_new_entries(&entries, ""), vec![&one, &two]);
    assert_eq!(normalize_tag("1.2.3"), "v1.2.3");
    assert_eq!(normalize_tag("v1.2.3"), "v1.2.3");
}
