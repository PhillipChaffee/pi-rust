//! Unit tests for the shared filesystem-scanning machinery: frontmatter
//! parsing edges (the `---` slicing contract, normalization, YAML
//! rejections), the path helpers, entry ordering, and kind resolution
//! through the nodejs environment.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use yaml_rust2::Yaml;

use crate::harness::env::nodejs::NodeExecutionEnv;
use crate::harness::fs_scan::{
    dirname_env_path, frontmatter_is_true, frontmatter_string, parse_frontmatter,
    relative_env_path, resolve_kind, sort_entries_by_name,
};
use crate::harness::types::{FileInfo, FileKind, FileSystem as _};

fn parsed(content: &str) -> (Yaml, String) {
    let parsed = parse_frontmatter(content).expect("parses");
    (parsed.frontmatter, parsed.body)
}

/// A file without a `---` opening carries no frontmatter; the normalized
/// text is the body.
#[test]
fn a_file_without_the_opening_delimiter_has_no_frontmatter() {
    let (frontmatter, body) = parsed("Hello\nWorld");
    assert_eq!(frontmatter, Yaml::Hash(yaml_rust2::yaml::Hash::new()));
    assert_eq!(body, "Hello\nWorld");
}

/// CRLF and lone-CR line endings normalize to LF in the body.
#[test]
fn line_endings_normalize_to_lf() {
    let (_, body) = parsed("---\nname: x\n---\r\nline one\r\nline two\r");
    assert_eq!(body, "line one\nline two");
}

/// A `---` opening with no closing delimiter carries no frontmatter.
#[test]
fn an_unclosed_delimiter_has_no_frontmatter() {
    let (frontmatter, body) = parsed("---\nname: x\nno closing delimiter");
    assert_eq!(frontmatter, Yaml::Hash(yaml_rust2::yaml::Hash::new()));
    assert_eq!(body, "---\nname: x\nno closing delimiter");
}

/// The closing delimiter is the first `\n---` after the opening, so a
/// longer dash run leaves its remainder in the body.
#[test]
fn a_longer_dash_run_leaves_its_remainder_in_the_body() {
    let (frontmatter, body) = parsed("---\nname: x\n----\nbody");
    assert_eq!(frontmatter_string(&frontmatter, "name"), Some("x"));
    assert_eq!(body, "-\nbody");
}

/// Empty YAML between the delimiters reads as an empty mapping.
#[test]
fn empty_frontmatter_reads_as_an_empty_mapping() {
    let (frontmatter, body) = parsed("---\n---\nbody");
    assert_eq!(frontmatter, Yaml::Hash(yaml_rust2::yaml::Hash::new()));
    assert_eq!(body, "body");
}

/// Non-object YAML reads as an empty mapping: property reads miss the way
/// JavaScript property reads miss on scalars and arrays.
#[test]
fn non_object_frontmatter_reads_as_an_empty_mapping() {
    for (frontmatter, _) in [
        parsed("---\n42\n---\nbody"),
        parsed("---\n[a, b]\n---\nbody"),
    ] {
        assert_eq!(frontmatter, Yaml::Hash(yaml_rust2::yaml::Hash::new()));
    }
}

/// Quoted and block values parse as strings.
#[test]
fn string_values_parse() {
    let (frontmatter, body) =
        parsed("---\nname: \"quoted name\"\ndescription: |\n  multi\n  line\n---\nbody");
    assert_eq!(
        frontmatter_string(&frontmatter, "name"),
        Some("quoted name")
    );
    assert_eq!(
        frontmatter_string(&frontmatter, "description"),
        Some("multi\nline\n")
    );
    assert_eq!(body, "body");
}

/// An unterminated flow sequence fails to parse, the input npm `yaml`
/// throws on.
#[test]
fn an_unterminated_flow_sequence_fails_to_parse() {
    let error = parse_frontmatter("---\ndescription: [invalid\n---\nbody")
        .expect_err("malformed YAML fails");
    // The message is yaml-rust2's scanner text, not npm's; it is asserted
    // loosely so a yaml-rust2 update cannot redden an unrelated port.
    assert!(
        error.0.contains("flow sequence"),
        "unexpected message: {error}"
    );
}

/// Duplicate top-level keys fail to parse, the input npm `yaml` throws on.
#[test]
fn duplicate_keys_fail_to_parse() {
    assert!(parse_frontmatter("---\na: b\na: c\n---\nbody").is_err());
}

/// Multiple documents (via `...` separators, which survive the slicing)
/// fail to parse, the input npm `yaml` throws on.
#[test]
fn multiple_documents_fail_to_parse() {
    assert!(parse_frontmatter("---\na: 1\n...\nb: 2\n---\nbody").is_err());
}

/// `disable-model-invocation` reads true only from the boolean `true`.
#[test]
fn the_disable_flag_reads_only_from_a_true_boolean() {
    let (frontmatter, _) = parsed("---\ndisable-model-invocation: true\n---\nbody");
    assert!(frontmatter_is_true(
        &frontmatter,
        "disable-model-invocation"
    ));
    let (frontmatter, _) = parsed("---\ndisable-model-invocation: \"true\"\n---\nbody");
    assert!(!frontmatter_is_true(
        &frontmatter,
        "disable-model-invocation"
    ));
}

/// The directory part keeps both separators in play, holds a Windows drive
/// root, and answers `/` when no separator exists.
#[test]
fn dirname_env_path_covers_the_separator_shapes() {
    assert_eq!(
        dirname_env_path("/skills/example/SKILL.md"),
        "/skills/example"
    );
    assert_eq!(
        dirname_env_path("/skills/example/SKILL.md/"),
        "/skills/example"
    );
    assert_eq!(dirname_env_path("C:\\skills\\SKILL.md"), "C:\\skills");
    assert_eq!(dirname_env_path("C:\\SKILL.md"), "C:\\");
    assert_eq!(dirname_env_path("SKILL.md"), "/");
    assert_eq!(dirname_env_path("/SKILL.md"), "/");
}

/// The relative path helper strips the root prefix, keeps outside paths'
/// leading separators dropped, and normalizes backslashes.
#[test]
fn relative_env_path_covers_the_root_shapes() {
    assert_eq!(relative_env_path("/root", "/root/a/b"), "a/b");
    assert_eq!(relative_env_path("/root", "/root"), "");
    assert_eq!(relative_env_path("/root", "/root/a/"), "a");
    assert_eq!(relative_env_path("/root", "/other/a"), "other/a");
    assert_eq!(relative_env_path("/root", "a/b"), "a/b");
    assert_eq!(relative_env_path("C:\\root", "C:\\root\\a"), "a");
}

/// Entries sort by name in byte order.
#[test]
fn entries_sort_by_name() {
    let mut entries = vec![
        file_info("root.md", FileKind::File),
        file_info("AGENTS.md", FileKind::File),
        file_info("CLAUDE.md", FileKind::File),
    ];
    sort_entries_by_name(&mut entries);
    let names: Vec<&str> = entries.iter().map(|entry| entry.name.as_str()).collect();
    assert_eq!(names, ["AGENTS.md", "CLAUDE.md", "root.md"]);
}

fn file_info(name: &str, kind: FileKind) -> FileInfo {
    FileInfo {
        name: name.to_owned(),
        path: format!("/{name}"),
        kind,
        size: 0,
        mtime_ms: 0,
    }
}

/// A symlinked skill file resolves through its canonical target; a broken
/// symlink answers `None` silently and a missing path answers `None`.
#[tokio::test]
async fn resolve_kind_walks_symlinks_and_drops_unresolvable_paths() {
    let root = tempfile::tempdir().expect("temp root");
    let env = NodeExecutionEnv::new(root.path().to_string_lossy().into_owned(), None, None);
    let context = crate::harness::context::background_context();
    std::fs::write(root.path().join("real.md"), "x").expect("write");
    std::os::unix::fs::symlink(root.path().join("real.md"), root.path().join("file-link"))
        .expect("symlink");
    std::os::unix::fs::symlink(root.path().join("missing"), root.path().join("broken-link"))
        .expect("symlink");

    let file_link = env
        .file_info("file-link", &context)
        .await
        .expect("file info");
    assert_eq!(
        resolve_kind(&env, &file_link, &mut Vec::new(), (), &context).await,
        Some(FileKind::File)
    );
    let broken = env
        .file_info("broken-link", &context)
        .await
        .expect("file info");
    let mut diagnostics = Vec::new();
    assert_eq!(
        resolve_kind(&env, &broken, &mut diagnostics, (), &context).await,
        None
    );
    // `not_found` on the canonical target stays silent.
    assert!(diagnostics.is_empty());
}
