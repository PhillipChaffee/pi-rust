//! The paths belt suite, upstream's `test/paths.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The `homedir()` reads port with an explicit home, the `fileURLToPath`
//! conversions restate over the `url` crate, and the win32-gated
//! "Windows file URL pathname" case rides the map's Windows exclusion
//! (recorded with the ticket).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::HashMap;

use pi_coding_agent::utils::paths::{
    PathInputOptions, canonicalize_path, format_path_relative_to_cwd_or_absolute,
    get_cwd_relative_path, is_local_path, normalize_path, normalize_windows_shell_path,
    resolve_path_with,
};

const HOME: &str = "/home/tester";

fn home_options() -> PathInputOptions {
    PathInputOptions::with_home(HOME)
}

fn with_env(pairs: &[(&str, String)]) -> pi_coding_agent::config::EnvLookup {
    let map: HashMap<String, String> = pairs
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect();
    Box::new(move |key: &str| map.get(key).cloned())
}

fn make_temp_dir(prefix: &str) -> std::path::PathBuf {
    let dir = tempfile::tempdir().expect("scratch dir");
    let path = dir.path().join(prefix);
    std::fs::create_dir_all(&path).expect("mkdir");
    // Leak the TempDir guard: the suite cleans up per case and the paths
    // outlive the assertion.
    std::mem::forget(dir);
    path
}

// === canonicalizePath =======================================================

#[test]
fn canonicalize_returns_the_real_path_for_a_regular_file() {
    let dir = make_temp_dir("pi-paths-canonical");
    let file = dir.join("file.txt");
    std::fs::write(&file, "hello").expect("write");
    assert_eq!(
        canonicalize_path(&file.to_string_lossy()),
        std::fs::canonicalize(&file)
            .expect("realpath")
            .to_string_lossy()
    );
}

#[test]
fn canonicalize_resolves_symlinks_to_their_targets() {
    let dir = make_temp_dir("pi-paths-link");
    let target = dir.join("target.txt");
    let link = dir.join("link.txt");
    std::fs::write(&target, "hello").expect("write");
    std::os::unix::fs::symlink(&target, &link).expect("symlink");
    assert_eq!(
        canonicalize_path(&link.to_string_lossy()),
        std::fs::canonicalize(&target)
            .expect("realpath")
            .to_string_lossy()
    );
}

#[test]
fn canonicalize_resolves_directory_symlinks() {
    let dir = make_temp_dir("pi-paths-dirlink");
    let target_dir = dir.join("target-dir");
    let link_dir = dir.join("link-dir");
    std::fs::create_dir_all(&target_dir).expect("mkdir");
    std::os::unix::fs::symlink(&target_dir, &link_dir).expect("symlink");
    assert_eq!(
        canonicalize_path(&link_dir.to_string_lossy()),
        std::fs::canonicalize(&target_dir)
            .expect("realpath")
            .to_string_lossy()
    );
}

#[test]
fn canonicalize_falls_back_to_the_raw_path_when_the_target_does_not_exist() {
    let dir = make_temp_dir("pi-paths-missing");
    let nonexistent = dir.join("no-such-file");
    assert_eq!(
        canonicalize_path(&nonexistent.to_string_lossy()),
        nonexistent.to_string_lossy()
    );
}

#[test]
fn canonicalize_falls_back_to_the_raw_path_for_a_dangling_symlink() {
    let dir = make_temp_dir("pi-paths-dangling");
    let target = dir.join("target.txt");
    let link = dir.join("link.txt");
    // Create a symlink whose target does not exist.
    std::os::unix::fs::symlink(&target, &link).expect("symlink");
    // realpath would fail, so canonicalize_path returns the link path.
    assert_eq!(
        canonicalize_path(&link.to_string_lossy()),
        link.to_string_lossy()
    );
}

// === getCwdRelativePath =====================================================

#[test]
fn cwd_relative_keeps_names_that_start_with_dots() {
    let cwd = std::env::temp_dir().join("pi-paths-cwd");
    let file = cwd.join("..config").join("AGENTS.md");
    assert_eq!(
        get_cwd_relative_path(&file.to_string_lossy(), &cwd.to_string_lossy()),
        Some("..config/AGENTS.md".to_string())
    );
}

#[test]
fn cwd_relative_rejects_parent_directory_traversals() {
    let cwd = std::env::temp_dir().join("pi-paths-cwd");
    let file = cwd.join("..").join("AGENTS.md");
    assert_eq!(
        get_cwd_relative_path(&file.to_string_lossy(), &cwd.to_string_lossy()),
        None
    );
}

// === resolvePath / normalizePath ============================================

#[test]
fn expands_only_home_tilde_shortcuts() {
    let cwd = std::env::temp_dir().join("pi-paths-cwd");
    let cwd = cwd.to_string_lossy().into_owned();
    assert_eq!(
        normalize_path("~", &home_options()).expect("normalize"),
        HOME
    );
    assert_eq!(
        normalize_path("~/file.txt", &home_options()).expect("normalize"),
        format!("{HOME}/file.txt")
    );
    assert_eq!(
        resolve_path_with("~draft.md", &cwd, &home_options()).expect("resolve"),
        format!("{cwd}/~draft.md")
    );
    assert_eq!(
        normalize_path("~draft.md", &home_options()).expect("normalize"),
        "~draft.md"
    );
}

#[test]
fn resolves_relative_paths_against_the_base_directory() {
    let cwd = std::env::temp_dir().join("pi-paths-cwd");
    let cwd = cwd.to_string_lossy().into_owned();
    assert_eq!(
        resolve_path_with("subdir/file.txt", &cwd, &home_options()).expect("resolve"),
        format!("{cwd}/subdir/file.txt")
    );
    let url = url::Url::from_file_path(&cwd)
        .expect("file url")
        .to_string();
    assert_eq!(
        resolve_path_with("subdir/file.txt", &url, &home_options()).expect("resolve"),
        format!("{cwd}/subdir/file.txt")
    );
}

#[test]
fn accepts_file_urls() {
    let dir = make_temp_dir("pi-paths-url");
    let file_path = dir.join("file with spaces.txt");
    let url = url::Url::from_file_path(&file_path)
        .expect("file url")
        .to_string();
    assert_eq!(
        resolve_path_with(&url, &dir.join("base").to_string_lossy(), &home_options())
            .expect("resolve"),
        file_path.to_string_lossy()
    );
}

#[test]
fn throws_for_invalid_file_urls() {
    let error = resolve_path_with("file:///%E0%A4%A", "/unused", &home_options());
    assert_eq!(
        error,
        Err(pi_coding_agent::utils::paths::PathNormalizeError)
    );
}

#[test]
fn preserves_posix_absolute_paths_with_literal_percent_sequences() {
    let dir = make_temp_dir("pi-paths-percent");
    for name in ["report%2026.md", "foo%2Fbar", "malformed%A.md"] {
        let file_path = dir.join(name);
        assert_eq!(
            resolve_path_with(
                &file_path.to_string_lossy(),
                &dir.join("base").to_string_lossy(),
                &home_options()
            )
            .expect("resolve"),
            file_path.to_string_lossy()
        );
    }
}

#[test]
fn file_urls_with_a_host_fail_the_conversion() {
    // node's fileURLToPath rejects any non-empty host on POSIX.
    assert_eq!(
        resolve_path_with("file://somehost/tmp/x", "/unused", &home_options()),
        Err(pi_coding_agent::utils::paths::PathNormalizeError)
    );
}

// === normalizeWindowsShellPath ==============================================

#[test]
fn converts_git_bash_msys_cygwin_and_wsl_drive_paths() {
    assert_eq!(
        normalize_windows_shell_path("/c/Users/example/project"),
        "C:\\Users\\example\\project"
    );
    assert_eq!(normalize_windows_shell_path("/cygdrive/d/work"), "D:\\work");
    assert_eq!(normalize_windows_shell_path("/mnt/e/source"), "E:\\source");
    assert_eq!(normalize_windows_shell_path("/c"), "C:\\");
}

#[test]
fn leaves_other_path_forms_unchanged() {
    for path in [
        "C:/Users/example",
        "C:\\Users\\example",
        "//server/share/file",
        "/c/Users\\example",
        "relative/file",
        "/tmp/file",
    ] {
        assert_eq!(normalize_windows_shell_path(path), path);
    }
}

// === isLocalPath ============================================================

#[test]
fn local_path_returns_true_for_bare_names() {
    assert!(is_local_path("my-package"));
}

#[test]
fn local_path_returns_true_for_relative_paths() {
    assert!(is_local_path("./foo"));
}

#[test]
fn local_path_returns_true_for_file_urls() {
    assert!(is_local_path("file:///tmp/foo"));
}

#[test]
fn local_path_returns_false_for_remote_protocols() {
    for source in ["npm:package", "git://repo", "https://example.com"] {
        assert!(!is_local_path(source));
    }
}

// === the belt additions the upstream file carries without a dedicated
// suite =====================================================================

#[test]
fn file_revision_stamps_the_stat_fields() {
    let dir = make_temp_dir("pi-paths-rev");
    let file = dir.join("file.txt");
    std::fs::write(&file, "hello").expect("write");
    let revision = pi_coding_agent::utils::paths::get_file_revision(&file.to_string_lossy())
        .expect("revision");
    let fields: Vec<&str> = revision.split(':').collect();
    assert_eq!(fields.len(), 5, "{revision}");
    // dev, ino, and size are numeric; the stat is stable across a re-read.
    assert!(
        fields
            .iter()
            .all(|field| field.chars().all(|ch| ch.is_ascii_digit()))
    );
    assert_eq!(
        pi_coding_agent::utils::paths::get_file_revision(&file.to_string_lossy()),
        Some(revision)
    );
    assert_eq!(
        pi_coding_agent::utils::paths::get_file_revision(&dir.join("no-such").to_string_lossy()),
        None
    );
}

#[test]
fn cloud_sync_tagging_runs_the_platform_helper_silently() {
    // The spawn's output is ignored and failures stay silent; the call must
    // not panic on any platform.
    let dir = make_temp_dir("pi-paths-xattr");
    let file = dir.join("ignored.txt");
    std::fs::write(&file, "hello").expect("write");
    pi_coding_agent::utils::paths::mark_path_ignored_by_cloud_sync(&file.to_string_lossy());
}

#[test]
fn unicode_space_options_normalize_the_variants() {
    let env = with_env(&[("HOME", HOME.to_string())]);
    let _guard = env;
    let options = PathInputOptions {
        normalize_unicode_spaces: true,
        ..home_options()
    };
    assert_eq!(
        normalize_path("file\u{00A0}name.txt", &options).expect("normalize"),
        "file name.txt"
    );
    assert_eq!(
        normalize_path("Screenshot\u{202F}AM.png", &options).expect("normalize"),
        "Screenshot AM.png"
    );
}

#[test]
fn trim_and_at_prefix_options_apply_in_order() {
    let options = PathInputOptions {
        trim: true,
        strip_at_prefix: true,
        ..home_options()
    };
    assert_eq!(
        normalize_path("  @file.txt  ", &options).expect("normalize"),
        "file.txt"
    );
    // The tilde form expands after the @ strips.
    assert_eq!(normalize_path("@~", &options).expect("normalize"), HOME);
    // Tilde expansion can be disabled.
    let no_expand = PathInputOptions {
        expand_tilde: false,
        ..home_options()
    };
    assert_eq!(
        normalize_path("~/file.txt", &no_expand).expect("normalize"),
        "~/file.txt"
    );
}

#[test]
fn format_relative_falls_back_when_the_input_cannot_normalize() {
    // A file:// URL that does not convert (non-empty host) cannot resolve;
    // the input passes through verbatim, upstream's catch.
    assert_eq!(
        format_path_relative_to_cwd_or_absolute("file://evilhost/x", "/tmp"),
        "file://evilhost/x"
    );
}
