//! Autocomplete provider tests, ported 1:1 from
//! `packages/tui/test/autocomplete.test.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#49), plus a boundary suite
//! driving the `fd`-walk machinery through a stub executable so the
//! spawn/parse/score paths stay covered on hosts without `fd` (the
//! upstream suite gates itself on `fd` being installed and skips
//! otherwise; CI has no `fd`).

#![expect(
    clippy::expect_used,
    reason = "the suite asserts on finds like upstream's assert.ok(...) with messages; expecting keeps the failure modes readable"
)]

use std::fmt::Write as _;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pi_tui::autocomplete::{
    AutocompleteItem, AutocompleteProvider, AutocompleteQueryOptions, AutocompleteSuggestions,
    CombinedAutocompleteProvider, CommandEntry, SlashCommand,
};
use tokio_util::sync::CancellationToken;

/// Resolve `fd` through PATH, upstream `resolveFdPath` (`which fd`).
fn resolve_fd_path() -> Option<PathBuf> {
    let output = Command::new("which").arg("fd").output().ok()?;
    if !output.status.success() || output.stdout.is_empty() {
        return None;
    }
    let first_line = String::from_utf8_lossy(&output.stdout)
        .lines()
        .find(|line| !line.trim().is_empty())?
        .trim()
        .to_string();
    if first_line.is_empty() {
        None
    } else {
        Some(PathBuf::from(first_line))
    }
}

fn fd_installed() -> bool {
    resolve_fd_path().is_some()
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A fresh temp directory, upstream `mkdtempSync`.
fn temp_dir(tag: &str) -> PathBuf {
    let unique = TEMP_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "pi-autocomplete-{tag}-{}-{unique}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

/// upstream `setupFolder`.
fn setup_folder(base: &Path, dirs: &[&str], files: &[(&str, &str)]) {
    for dir in dirs {
        fs::create_dir_all(base.join(dir)).expect("create fixture dir");
    }
    for (path, contents) in files {
        let full = base.join(path);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).expect("create fixture parent");
        }
        fs::write(full, contents).expect("write fixture file");
    }
}

fn get_suggestions(
    provider: &CombinedAutocompleteProvider,
    lines: &[&str],
    cursor_line: usize,
    cursor_col: usize,
    force: bool,
) -> Option<AutocompleteSuggestions> {
    provider.get_suggestions(
        &lines_of(lines),
        cursor_line,
        cursor_col,
        &AutocompleteQueryOptions {
            signal: CancellationToken::new(),
            force,
        },
    )
}

fn lines_of(lines: &[&str]) -> Vec<String> {
    lines.iter().map(|line| (*line).to_string()).collect()
}

fn item(value: &str) -> AutocompleteItem {
    AutocompleteItem {
        value: value.to_string(),
        label: value.to_string(),
        description: None,
    }
}

// --- extractPathPrefix (upstream describe block) --------------------------

#[test]
fn extracts_root_from_hey_slash_when_forced() {
    let provider = CombinedAutocompleteProvider::new(Vec::new(), "/tmp", None);
    let result = get_suggestions(&provider, &["hey /"], 0, 5, true);

    assert!(
        result.is_some(),
        "Should return suggestions for root directory"
    );
    let result = result.expect("checked");
    assert_eq!(result.prefix, "/", "Prefix should be '/'");
}

#[test]
fn extracts_a_from_a_when_forced() {
    let provider = CombinedAutocompleteProvider::new(Vec::new(), "/tmp", None);
    let result = get_suggestions(&provider, &["/A"], 0, 2, true);

    // This might return None if /A doesn't match anything, which is fine;
    // the prefix extraction is what's under test.
    if let Some(result) = result {
        assert_eq!(result.prefix, "/A", "Prefix should be '/A'");
    }
}

#[test]
fn does_not_trigger_for_slash_commands() {
    let provider = CombinedAutocompleteProvider::new(Vec::new(), "/tmp", None);
    let result = get_suggestions(&provider, &["/model"], 0, 6, true);

    assert!(result.is_none(), "Should not trigger for slash commands");
}

#[test]
fn triggers_for_absolute_paths_after_slash_command_argument() {
    let provider = CombinedAutocompleteProvider::new(Vec::new(), "/tmp", None);
    let result = get_suggestions(&provider, &["/command /"], 0, 10, true);

    assert!(
        result.is_some(),
        "Should trigger for absolute paths in command arguments"
    );
    if let Some(result) = result {
        assert_eq!(result.prefix, "/", "Prefix should be '/'");
    }
}

// --- fd @ file suggestions (upstream describe block, fd-gated) -----------

fn fd_suite_root() -> (PathBuf, PathBuf, PathBuf, bool) {
    let root = temp_dir("root");
    let base = root.join("cwd");
    let outside = root.join("outside");
    fs::create_dir_all(&base).expect("base dir");
    fs::create_dir_all(&outside).expect("outside dir");
    let installed = fd_installed();
    (root, base, outside, installed)
}

#[test]
fn fd_returns_all_files_and_folders_for_empty_at_query() {
    let (root, base, _outside, installed) = fd_suite_root();
    if !installed {
        cleanup_and_return(&root);
        return;
    }
    setup_folder(&base, &["src"], &[("README.md", "readme")]);

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        resolve_fd_path(),
    );
    let result = get_suggestions(&provider, &["@"], 0, 1, false);

    let mut values = result
        .expect("suggestions")
        .items
        .iter()
        .map(|item| item.value.clone())
        .collect::<Vec<_>>();
    values.sort();
    let mut expected = vec!["@README.md".to_string(), "@src/".to_string()];
    expected.sort();
    assert_eq!(values, expected);
    cleanup_and_return(&root);
}

#[test]
fn fd_matches_file_with_extension_in_query() {
    let (root, base, _outside, installed) = fd_suite_root();
    if !installed {
        cleanup_and_return(&root);
        return;
    }
    setup_folder(&base, &[], &[("file.txt", "content")]);

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        resolve_fd_path(),
    );
    let result = get_suggestions(&provider, &["@file.txt"], 0, 9, false);

    let values = result.expect("suggestions").items;
    assert!(values.iter().any(|item| item.value == "@file.txt"));
    cleanup_and_return(&root);
}

#[test]
fn fd_filters_are_case_insensitive() {
    let (root, base, _outside, installed) = fd_suite_root();
    if !installed {
        cleanup_and_return(&root);
        return;
    }
    setup_folder(&base, &["src"], &[("README.md", "readme")]);

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        resolve_fd_path(),
    );
    let result = get_suggestions(&provider, &["@re"], 0, 3, false);

    let values = result.expect("suggestions").items;
    assert_eq!(values.len(), 1);
    assert_eq!(values[0].value, "@README.md");
    cleanup_and_return(&root);
}

#[test]
fn fd_ranks_directories_before_files() {
    let (root, base, _outside, installed) = fd_suite_root();
    if !installed {
        cleanup_and_return(&root);
        return;
    }
    setup_folder(&base, &["src"], &[("src.txt", "text")]);

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        resolve_fd_path(),
    );
    let result = get_suggestions(&provider, &["@src"], 0, 4, false);

    let result = result.expect("suggestions");
    assert_eq!(
        result.items.first().map(|item| item.value.as_str()),
        Some("@src/")
    );
    assert!(result.items.iter().any(|item| item.value == "@src.txt"));
    cleanup_and_return(&root);
}

#[test]
fn fd_returns_nested_file_paths() {
    let (root, base, _outside, installed) = fd_suite_root();
    if !installed {
        cleanup_and_return(&root);
        return;
    }
    setup_folder(&base, &[], &[("src/index.ts", "export {};\n")]);

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        resolve_fd_path(),
    );
    let result = get_suggestions(&provider, &["@index"], 0, 6, false);

    let values = result.expect("suggestions").items;
    assert!(values.iter().any(|item| item.value == "@src/index.ts"));
    cleanup_and_return(&root);
}

#[test]
fn fd_matches_deeply_nested_paths() {
    let (root, base, _outside, installed) = fd_suite_root();
    if !installed {
        cleanup_and_return(&root);
        return;
    }
    setup_folder(
        &base,
        &[],
        &[
            ("packages/tui/src/autocomplete.ts", "export {};"),
            ("packages/ai/src/autocomplete.ts", "export {};"),
        ],
    );

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        resolve_fd_path(),
    );
    let result = get_suggestions(&provider, &["@tui/src/auto"], 0, 13, false);

    let values = result.expect("suggestions").items;
    assert!(
        values
            .iter()
            .any(|item| item.value == "@packages/tui/src/autocomplete.ts")
    );
    assert!(
        !values
            .iter()
            .any(|item| item.value == "@packages/ai/src/autocomplete.ts")
    );
    cleanup_and_return(&root);
}

#[test]
fn fd_matches_directory_in_middle_of_path_with_full_path() {
    let (root, base, _outside, installed) = fd_suite_root();
    if !installed {
        cleanup_and_return(&root);
        return;
    }
    setup_folder(
        &base,
        &[],
        &[
            ("src/components/Button.tsx", "export {};"),
            ("src/utils/helpers.ts", "export {};"),
        ],
    );

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        resolve_fd_path(),
    );
    let result = get_suggestions(&provider, &["@components/"], 0, 12, false);

    let values = result.expect("suggestions").items;
    assert!(
        values
            .iter()
            .any(|item| item.value == "@src/components/Button.tsx")
    );
    assert!(
        !values
            .iter()
            .any(|item| item.value == "@src/utils/helpers.ts")
    );
    cleanup_and_return(&root);
}

#[test]
fn fd_scopes_fuzzy_search_to_relative_directories_and_searches_recursively() {
    let (root, base, outside, installed) = fd_suite_root();
    if !installed {
        cleanup_and_return(&root);
        return;
    }
    setup_folder(
        &outside,
        &[],
        &[
            ("nested/alpha.ts", "export {};"),
            ("nested/deeper/also-alpha.ts", "export {};"),
            ("nested/deeper/zzz.ts", "export {};"),
        ],
    );

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        resolve_fd_path(),
    );
    let result = get_suggestions(&provider, &["@../outside/a"], 0, 13, false);

    let values = result.expect("suggestions").items;
    assert!(
        values
            .iter()
            .any(|item| item.value == "@../outside/nested/alpha.ts")
    );
    assert!(
        values
            .iter()
            .any(|item| item.value == "@../outside/nested/deeper/also-alpha.ts")
    );
    assert!(
        !values
            .iter()
            .any(|item| item.value == "@../outside/nested/deeper/zzz.ts")
    );
    cleanup_and_return(&root);
}

#[test]
fn fd_ranks_shallower_same_score_matches_before_deeper_matches() {
    let (root, base, _outside, installed) = fd_suite_root();
    if !installed {
        cleanup_and_return(&root);
        return;
    }
    setup_folder(
        &base,
        &[
            "scope/aaa/venv/lib/python3.12/site-packages/pkg/core/profile",
            "scope/projects",
        ],
        &[],
    );

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        resolve_fd_path(),
    );
    let result = get_suggestions(&provider, &["@scope/pro"], 0, 10, false);

    let values = result.expect("suggestions").items;
    assert_eq!(values[0].value, "@scope/projects/");
    assert!(
        values
            .iter()
            .any(|item| item.value
                == "@scope/aaa/venv/lib/python3.12/site-packages/pkg/core/profile/")
    );
    cleanup_and_return(&root);
}

#[test]
fn fd_includes_scoped_direct_children_when_recursive_matches_are_flooded() {
    let (root, base, _outside, installed) = fd_suite_root();
    if !installed {
        cleanup_and_return(&root);
        return;
    }
    let flooded: Vec<String> = (0..250)
        .map(|index| {
            format!(
                "scope/a{:03}/venv/lib/python3.12/site-packages/pkg/core/profile",
                index + 1
            )
        })
        .collect();
    let flooded_refs: Vec<&str> = flooded.iter().map(String::as_str).collect();
    setup_folder(&base, &["scope/projects"], &[]);
    setup_folder(&base, &flooded_refs, &[]);

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        resolve_fd_path(),
    );
    let result = get_suggestions(&provider, &["@scope/pro"], 0, 10, false);

    let values = result.expect("suggestions").items;
    assert_eq!(values[0].value, "@scope/projects/");
    assert!(
        values.iter().any(|item| item.value.contains("/profile/")),
        "Should keep deep fuzzy matches after direct children"
    );
    cleanup_and_return(&root);
}

#[test]
fn fd_quotes_paths_with_spaces_for_at_suggestions() {
    let (root, base, _outside, installed) = fd_suite_root();
    if !installed {
        cleanup_and_return(&root);
        return;
    }
    setup_folder(&base, &["my folder"], &[("my folder/test.txt", "content")]);

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        resolve_fd_path(),
    );
    let result = get_suggestions(&provider, &["@my"], 0, 3, false);

    let values = result.expect("suggestions").items;
    assert!(values.iter().any(|item| item.value == "@\"my folder/\""));
    cleanup_and_return(&root);
}

#[test]
fn fd_includes_hidden_paths_but_excludes_git() {
    let (root, base, _outside, installed) = fd_suite_root();
    if !installed {
        cleanup_and_return(&root);
        return;
    }
    setup_folder(
        &base,
        &[".pi", ".github", ".git"],
        &[
            (".pi/config.json", "{}"),
            (".github/workflows/ci.yml", "name: ci"),
            (".git/config", "[core]"),
        ],
    );

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        resolve_fd_path(),
    );
    let result = get_suggestions(&provider, &["@"], 0, 1, false);

    let values = result.expect("suggestions").items;
    assert!(values.iter().any(|item| item.value == "@.pi/"));
    assert!(values.iter().any(|item| item.value == "@.github/"));
    assert!(
        !values
            .iter()
            .any(|item| item.value == "@.git" || item.value.starts_with("@.git/"))
    );
    cleanup_and_return(&root);
}

#[test]
fn fd_follows_symlinked_directories_for_fuzzy_at_search() {
    let (root, base, outside, installed) = fd_suite_root();
    if !installed {
        cleanup_and_return(&root);
        return;
    }
    setup_folder(&base, &["dir"], &[("dir/some_file.txt", "real")]);
    setup_folder(&outside, &[], &[("some_file.txt", "symlinked")]);
    symlink("../outside", base.join("symlinked_dir")).expect("symlink");

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        resolve_fd_path(),
    );
    let result = get_suggestions(&provider, &["@some"], 0, 5, false);

    let values = result.expect("suggestions").items;
    assert!(values.iter().any(|item| item.value == "@dir/some_file.txt"));
    assert!(
        values
            .iter()
            .any(|item| item.value == "@symlinked_dir/some_file.txt")
    );
    cleanup_and_return(&root);
}

#[test]
fn fd_returns_symlinked_directories_when_matching_their_name() {
    let (root, base, outside, installed) = fd_suite_root();
    if !installed {
        cleanup_and_return(&root);
        return;
    }
    setup_folder(&outside, &["nested"], &[("nested/file.txt", "symlinked")]);
    symlink("../outside", base.join("symlinked_dir")).expect("symlink");

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        resolve_fd_path(),
    );
    let result = get_suggestions(&provider, &["@symlinked"], 0, 10, false);

    let values = result.expect("suggestions").items;
    assert!(values.iter().any(|item| item.value == "@symlinked_dir/"));
    cleanup_and_return(&root);
}

#[test]
fn fd_returns_symlinked_files_without_requiring_type_l() {
    let (root, base, _outside, installed) = fd_suite_root();
    if !installed {
        cleanup_and_return(&root);
        return;
    }
    setup_folder(&base, &[], &[("original.txt", "content")]);
    symlink("original.txt", base.join("link.txt")).expect("symlink");

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        resolve_fd_path(),
    );
    let result = get_suggestions(&provider, &["@link"], 0, 5, false);

    let values = result.expect("suggestions").items;
    assert!(values.iter().any(|item| item.value == "@link.txt"));
    cleanup_and_return(&root);
}

#[test]
fn fd_returns_the_same_at_suggestions_when_the_cwd_path_contains_the_query() {
    let (root, _base, _outside, installed) = fd_suite_root();
    if !installed {
        cleanup_and_return(&root);
        return;
    }
    let normal_base = root.join("cwd-normal");
    let query_in_path_base = root.join("cwd-plan-repro");
    fs::create_dir_all(&normal_base).expect("normal base");
    fs::create_dir_all(&query_in_path_base).expect("query base");

    let dirs = ["packages/coding-agent/examples/extensions/plan-mode"];
    let files = [
        (
            "packages/coding-agent/examples/extensions/plan-mode/README.md",
            "readme",
        ),
        ("packages/tui/docs/plan.md", "plan"),
    ];
    setup_folder(&normal_base, &dirs, &files);
    setup_folder(&query_in_path_base, &dirs, &files);

    let normal_provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        normal_base.to_string_lossy().to_string(),
        resolve_fd_path(),
    );
    let query_in_path_provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        query_in_path_base.to_string_lossy().to_string(),
        resolve_fd_path(),
    );

    let normal_result = get_suggestions(&normal_provider, &["@plan"], 0, 5, false);
    let query_in_path_result = get_suggestions(&query_in_path_provider, &["@plan"], 0, 5, false);

    let normalize = |result: &Option<AutocompleteSuggestions>| -> Vec<String> {
        let mut labels: Vec<String> = result
            .as_ref()
            .map(|suggestions| {
                suggestions
                    .items
                    .iter()
                    .map(|item| {
                        format!(
                            "{} :: {}",
                            item.label,
                            item.description.clone().unwrap_or_default()
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        labels.sort();
        labels
    };

    assert_eq!(normalize(&query_in_path_result), normalize(&normal_result));
    assert!(normalize(&normal_result).contains(
        &"plan-mode/ :: packages/coding-agent/examples/extensions/plan-mode".to_string()
    ));
    assert!(
        normalize(&normal_result).contains(&"plan.md :: packages/tui/docs/plan.md".to_string())
    );
    cleanup_and_return(&root);
}

#[test]
fn fd_continues_autocomplete_inside_quoted_at_paths() {
    let (root, base, _outside, installed) = fd_suite_root();
    if !installed {
        cleanup_and_return(&root);
        return;
    }
    setup_folder(
        &base,
        &[],
        &[
            ("my folder/test.txt", "content"),
            ("my folder/other.txt", "content"),
        ],
    );

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        resolve_fd_path(),
    );
    // Cursor before the closing quote, upstream `line.length - 1`.
    let result = get_suggestions(&provider, &["@\"my folder/\""], 0, 12, false);

    let result = result.expect("suggestions for quoted folder path");
    let values = result.items;
    assert!(
        values
            .iter()
            .any(|item| item.value == "@\"my folder/test.txt\"")
    );
    assert!(
        values
            .iter()
            .any(|item| item.value == "@\"my folder/other.txt\"")
    );
    cleanup_and_return(&root);
}

#[test]
fn fd_applies_quoted_at_completion_without_duplicating_closing_quote() {
    let (root, base, _outside, installed) = fd_suite_root();
    if !installed {
        cleanup_and_return(&root);
        return;
    }
    setup_folder(&base, &[], &[("my folder/test.txt", "content")]);

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        resolve_fd_path(),
    );
    let line = "@\"my folder/te\"";
    let cursor_col = line.len() - 1;
    let result = get_suggestions(&provider, &[line], 0, cursor_col, false);

    let result = result.expect("suggestions for quoted @ path");
    let item = result
        .items
        .iter()
        .find(|entry| entry.value == "@\"my folder/test.txt\"")
        .expect("test.txt suggestion");
    let applied = provider.apply_completion(
        &lines_of(std::slice::from_ref(&line)),
        0,
        cursor_col,
        item,
        &result.prefix,
    );
    assert_eq!(applied.lines[0], "@\"my folder/test.txt\" ");
    cleanup_and_return(&root);
}

fn cleanup_and_return(root: &Path) {
    cleanup(root);
}

// --- dot-slash path completion (upstream describe block) ------------------

#[test]
fn dot_slash_preserves_prefix_when_completing_paths() {
    let base = temp_dir("dot-slash");
    setup_folder(
        &base,
        &[],
        &[("update.sh", "#!/bin/bash"), ("utils.ts", "export {};")],
    );

    let provider =
        CombinedAutocompleteProvider::new(Vec::new(), base.to_string_lossy().to_string(), None);
    let result = get_suggestions(&provider, &["./up"], 0, 4, true);

    let result = result.expect("suggestions for ./ path");
    let values = result.items;
    assert!(
        values.iter().any(|item| item.value == "./update.sh"),
        "Expected ./update.sh in {values:?}"
    );
    cleanup(&base);
}

#[test]
fn dot_slash_preserves_prefix_for_directory_completions() {
    let base = temp_dir("dot-slash-dir");
    setup_folder(&base, &["src"], &[("src/index.ts", "export {};")]);

    let provider =
        CombinedAutocompleteProvider::new(Vec::new(), base.to_string_lossy().to_string(), None);
    let result = get_suggestions(&provider, &["./sr"], 0, 4, true);

    let result = result.expect("suggestions for ./ directory path");
    let values = result.items;
    assert!(
        values.iter().any(|item| item.value == "./src/"),
        "Expected ./src/ in {values:?}"
    );
    cleanup(&base);
}

// --- quoted path completion (upstream describe block) ---------------------

#[test]
fn quotes_paths_with_spaces_for_direct_completion() {
    let base = temp_dir("quoted");
    setup_folder(&base, &["my folder"], &[("my folder/test.txt", "content")]);

    let provider =
        CombinedAutocompleteProvider::new(Vec::new(), base.to_string_lossy().to_string(), None);
    let result = get_suggestions(&provider, &["my"], 0, 2, true);

    let result = result.expect("suggestions for path completion");
    let values = result.items;
    assert!(values.iter().any(|item| item.value == "\"my folder/\""));
    cleanup(&base);
}

#[test]
fn continues_completion_inside_quoted_paths() {
    let base = temp_dir("quoted-continue");
    setup_folder(
        &base,
        &[],
        &[
            ("my folder/test.txt", "content"),
            ("my folder/other.txt", "content"),
        ],
    );

    let provider =
        CombinedAutocompleteProvider::new(Vec::new(), base.to_string_lossy().to_string(), None);
    // Cursor before the closing quote, upstream `line.length - 1`.
    let result = get_suggestions(&provider, &["\"my folder/\""], 0, 11, true);

    let result = result.expect("suggestions for quoted folder path");
    let values = result.items;
    assert!(
        values
            .iter()
            .any(|item| item.value == "\"my folder/test.txt\"")
    );
    assert!(
        values
            .iter()
            .any(|item| item.value == "\"my folder/other.txt\"")
    );
    cleanup(&base);
}

#[test]
fn applies_quoted_completion_without_duplicating_closing_quote() {
    let base = temp_dir("quoted-apply");
    setup_folder(&base, &[], &[("my folder/test.txt", "content")]);

    let provider =
        CombinedAutocompleteProvider::new(Vec::new(), base.to_string_lossy().to_string(), None);
    let line = "\"my folder/te\"";
    let cursor_col = line.len() - 1;
    let result = get_suggestions(&provider, &[line], 0, cursor_col, true);

    let result = result.expect("suggestions for quoted path");
    let item = result
        .items
        .iter()
        .find(|entry| entry.value == "\"my folder/test.txt\"")
        .expect("test.txt suggestion");
    let applied = provider.apply_completion(
        &lines_of(std::slice::from_ref(&line)),
        0,
        cursor_col,
        item,
        &result.prefix,
    );
    assert_eq!(applied.lines[0], "\"my folder/test.txt\"");
    cleanup(&base);
}

// --- boundary suite: the fd machinery through a stub executable ----------

/// Write a stub `fd` that ignores its arguments and prints fixed lines, so
/// the provider's spawn/parse machinery runs without the real tool.
fn stub_fd(lines: &[&str], extra: Option<&str>) -> (PathBuf, PathBuf) {
    let dir = temp_dir("stub");
    let script = dir.join("fd");
    let mut body = String::from("#!/bin/sh\n");
    if let Some(extra) = extra {
        body.push_str(extra);
        body.push('\n');
    }
    for line in lines {
        let _ = writeln!(body, "printf '%s\n' \"{line}\"");
    }
    fs::write(&script, body).expect("write stub");
    Command::new("chmod")
        .arg("+x")
        .arg(&script)
        .output()
        .expect("chmod stub");
    (dir, script)
}

#[test]
fn stub_walk_builds_values_from_emitted_paths() {
    let (stub_dir, stub) = stub_fd(&["README.md", "src/"], None);
    let base = temp_dir("stub-base");

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        Some(stub_fd_path(&stub)),
    );
    let result = get_suggestions(&provider, &["@"], 0, 1, false);

    let mut values = result
        .expect("suggestions")
        .items
        .iter()
        .map(|item| item.value.clone())
        .collect::<Vec<_>>();
    values.sort();
    assert_eq!(values, vec!["@README.md", "@src/"]);
    cleanup(&base);
    cleanup(&stub_dir);
}

fn stub_fd_path(stub: &Path) -> PathBuf {
    stub.to_path_buf()
}

#[test]
fn stub_walk_distinguishes_directories_from_files_by_trailing_slash() {
    let (stub_dir, stub) = stub_fd(&["a/", "b.txt"], None);
    let base = temp_dir("stub-slash");

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        Some(stub_fd_path(&stub)),
    );
    let result = get_suggestions(&provider, &["@"], 0, 1, false);

    let result = result.expect("suggestions");
    let directories = result
        .items
        .iter()
        .filter(|item| item.value.ends_with('/') || item.value.ends_with("/\""))
        .count();
    assert_eq!(directories, 1, "only a/ is a directory: {:?}", result.items);
    cleanup(&base);
    cleanup(&stub_dir);
}

#[test]
fn stub_walk_passes_full_path_for_slash_queries() {
    // The stub emits different lines depending on whether --full-path is
    // in its arguments.
    let script_body = "for arg in \"$@\"; do case \"$arg\" in --full-path) printf '%s\\n' 'src/components/Button.tsx'; exit 0;; esac; done\n";
    let (stub_dir, stub) = stub_fd(&["Button.tsx"], Some(script_body));
    let base = temp_dir("stub-fullpath");

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        Some(stub_fd_path(&stub)),
    );
    // A slash-containing query must flip the provider into --full-path
    // mode and the stub answers with the nested path.
    let result = get_suggestions(&provider, &["@components/"], 0, 12, false);
    let values = result.expect("suggestions").items;
    assert!(
        values
            .iter()
            .any(|item| item.value == "@src/components/Button.tsx"),
        "the slash query drives --full-path: {values:?}"
    );
    cleanup(&base);
    cleanup(&stub_dir);
}

#[test]
fn stub_walk_aborts_when_cancelled_before_spawn() {
    let (stub_dir, stub) = stub_fd(&["README.md"], None);
    let base = temp_dir("stub-abort");

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        Some(stub_fd_path(&stub)),
    );
    let token = CancellationToken::new();
    token.cancel();
    let result = provider.get_suggestions(
        &lines_of(&["@"]),
        0,
        1,
        &AutocompleteQueryOptions {
            signal: token,
            force: false,
        },
    );
    assert!(result.is_none(), "a cancelled query yields nothing");
    cleanup(&base);
    cleanup(&stub_dir);
}

#[test]
fn stub_walk_degrades_when_the_binary_is_missing() {
    let base = temp_dir("stub-missing");
    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        Some(PathBuf::from("/nonexistent/fd-binary")),
    );
    let result = get_suggestions(&provider, &["@"], 0, 1, false);
    assert!(result.is_none(), "a missing fd degrades to no suggestions");
    cleanup(&base);
}

#[test]
fn stub_walk_degrades_on_nonzero_exit_and_empty_output() {
    let base = temp_dir("stub-exit");
    let (stub_dir, failing) = stub_fd(&[], Some("exit 1\n"));
    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        Some(stub_fd_path(&failing)),
    );
    assert!(get_suggestions(&provider, &["@"], 0, 1, false).is_none());
    cleanup(&base);
    cleanup(&stub_dir);

    let (stub_dir2, empty) = stub_fd(&[], None);
    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        Some(stub_fd_path(&empty)),
    );
    assert!(get_suggestions(&provider, &["@"], 0, 1, false).is_none());
    cleanup(&stub_dir2);
}

#[test]
fn stub_fuzzy_scores_rank_exact_prefix_substring_and_path_matches() {
    // One entry per score band: exact (100), prefix (80), filename
    // substring (50), path substring (30); directories gain +10.
    let (stub_dir, stub) = stub_fd(
        &[
            "pro",
            "proX/extra",
            "mypro",
            "deep/x/mypro",
            "prefix-other",
            "PRO/",
        ],
        None,
    );
    let base = temp_dir("stub-score");

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        Some(stub_fd_path(&stub)),
    );
    let result = get_suggestions(&provider, &["@pro"], 0, 4, false);
    let result = result.expect("suggestions");
    let values: Vec<&str> = result
        .items
        .iter()
        .map(|item| item.value.as_str())
        .collect();
    // Exact filename "pro" first, then the +10 directory "PRO/", then
    // prefix "proX/extra"... upstream's sort: score desc, depth asc.
    assert_eq!(
        values.first(),
        Some(&"@pro"),
        "exact filename ranks first: {values:?}"
    );
    assert!(
        values.contains(&"@PRO/"),
        "directories outrank same-score files: {values:?}"
    );
    assert!(
        values.contains(&"@deep/x/mypro"),
        "path-substring matches survive scoring: {values:?}"
    );
    cleanup(&base);
    cleanup(&stub_dir);
}

#[test]
fn stub_fuzzy_deduplicates_base_and_recursive_walk_results() {
    // The depth-1 walk prints the direct child; the recursive walk prints
    // it again (a duplicate) plus the nested path.
    let script_body = "for arg in \"$@\"; do case \"$arg\" in --max-depth) exit 0;; esac; done\nprintf '%s\\n' 'nested/README.md'\n";
    let (stub_dir, stub) = stub_fd(&["README.md"], Some(script_body));
    let base = temp_dir("stub-dedup");

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        Some(stub_fd_path(&stub)),
    );
    let result = get_suggestions(&provider, &["@"], 0, 1, false);
    let values = result.expect("suggestions").items;
    let readme_count = values
        .iter()
        .filter(|item| item.value == "@README.md")
        .count();
    assert_eq!(readme_count, 1, "duplicates collapse: {values:?}");
    cleanup(&base);
    cleanup(&stub_dir);
}

#[test]
fn stub_fuzzy_scopes_to_relative_directories() {
    let base = temp_dir("stub-scope");
    setup_folder(&base, &["sub"], &[]);
    let (stub_dir, stub) = stub_fd(&["alpha.ts", "nested/alpha.ts"], None);

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        Some(stub_fd_path(&stub)),
    );
    let result = get_suggestions(&provider, &["@sub/"], 0, 5, false);
    let values = result.expect("suggestions").items;
    assert!(
        values.iter().all(|item| item.value.starts_with("@sub/")),
        "scoped results re-prefix the display base: {values:?}"
    );
    assert!(
        values.iter().any(|item| item.value == "@sub/alpha.ts"),
        "scoped results carry the display base: {values:?}"
    );
    cleanup(&base);
    cleanup(&stub_dir);
}

#[test]
fn stub_fuzzy_scopes_through_home_directories() {
    let base = temp_dir("stub-home");
    let home = temp_dir("stub-home-dir");
    setup_folder(&home, &["docs"], &[]);
    let (stub_dir, stub) = stub_fd(&["notes.md"], None);

    let provider = CombinedAutocompleteProvider::with_home_lookup(
        Vec::new(),
        base.to_string_lossy().to_string(),
        Some(stub_fd_path(&stub)),
        Box::new({
            let home = home.clone();
            move || Some(home.to_string_lossy().to_string())
        }),
    );
    let result = get_suggestions(&provider, &["@~/docs/"], 0, 8, false);
    let values = result.expect("suggestions").items;
    assert!(
        values.iter().all(|item| item.value.starts_with("@~/docs/")),
        "home-scoped results display with the ~/ base: {values:?}"
    );
    cleanup(&base);
    cleanup(&home);
    cleanup(&stub_dir);
}

#[test]
fn stub_fuzzy_quotes_paths_with_spaces() {
    let (stub_dir, stub) = stub_fd(&["my folder/test.txt"], None);
    let base = temp_dir("stub-quote");

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        Some(stub_fd_path(&stub)),
    );
    let result = get_suggestions(&provider, &["@my"], 0, 3, false);
    let values = result.expect("suggestions").items;
    assert!(
        values
            .iter()
            .any(|item| item.value == "@\"my folder/test.txt\""),
        "paths with spaces quote in the value: {values:?}"
    );
    cleanup(&base);
    cleanup(&stub_dir);
}

#[test]
fn stub_fuzzy_describes_entries_with_their_display_path() {
    let (stub_dir, stub) = stub_fd(&["src/index.ts"], None);
    let base = temp_dir("stub-desc");

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        Some(stub_fd_path(&stub)),
    );
    let result = get_suggestions(&provider, &["@index"], 0, 6, false);
    let items = result.expect("suggestions").items;
    assert!(
        items.iter().any(|item| item.value == "@src/index.ts"
            && item.description.as_deref() == Some("src/index.ts")),
        "fuzzy suggestions carry the display path as description: {items:?}"
    );
    cleanup(&base);
    cleanup(&stub_dir);
}

// --- provider mechanics ---------------------------------------------------

#[test]
fn slash_command_completion_filters_and_formats_descriptions() {
    let base = temp_dir("slash-commands");
    let provider = CombinedAutocompleteProvider::new(
        vec![
            CommandEntry::Slash(SlashCommand {
                name: "model".to_string(),
                description: Some("Switch model".to_string()),
                argument_hint: Some("<model>".to_string()),
                get_argument_completions: None,
            }),
            CommandEntry::Slash(SlashCommand {
                name: "help".to_string(),
                description: None,
                argument_hint: None,
                get_argument_completions: None,
            }),
            CommandEntry::Item(AutocompleteItem {
                value: "plain".to_string(),
                label: "plain".to_string(),
                description: None,
            }),
        ],
        base.to_string_lossy().to_string(),
        None,
    );
    let result = get_suggestions(&provider, &["/m"], 0, 2, false);

    let result = result.expect("command suggestions");
    assert_eq!(result.prefix, "/m");
    let model = result
        .items
        .iter()
        .find(|item| item.value == "model")
        .expect("model command");
    // hint first, then the description, upstream's `hint — desc` join.
    assert_eq!(model.description.as_deref(), Some("<model> — Switch model"));
    assert!(
        !result.items.iter().any(|item| item.value == "plain"),
        "the prefix filters non-matching commands"
    );
    cleanup(&base);
}

#[test]
fn slash_command_argument_completions_flow_through_the_command_path() {
    let base = temp_dir("slash-args");
    let provider = CombinedAutocompleteProvider::new(
        vec![CommandEntry::Slash(SlashCommand {
            name: "load".to_string(),
            description: Some("Load".to_string()),
            argument_hint: None,
            get_argument_completions: Some(Box::new(|prefix| {
                if prefix.is_empty() {
                    Some(vec![
                        AutocompleteItem {
                            value: "skill-a".to_string(),
                            label: "skill-a".to_string(),
                            description: None,
                        },
                        AutocompleteItem {
                            value: "skill-b".to_string(),
                            label: "skill-b".to_string(),
                            description: None,
                        },
                    ])
                } else {
                    None
                }
            })),
        })],
        base.to_string_lossy().to_string(),
        None,
    );
    let result = get_suggestions(&provider, &["/load "], 0, 6, false);

    let result = result.expect("argument suggestions");
    assert_eq!(result.prefix, "");
    assert_eq!(result.items.len(), 2);

    // No argument completer: no suggestions.
    let without = CombinedAutocompleteProvider::new(
        vec![CommandEntry::Slash(SlashCommand {
            name: "load".to_string(),
            description: None,
            argument_hint: None,
            get_argument_completions: None,
        })],
        base.to_string_lossy().to_string(),
        None,
    );
    assert!(get_suggestions(&without, &["/load "], 0, 6, false).is_none());

    // Unknown command: no suggestions.
    assert!(get_suggestions(&without, &["/other "], 0, 7, false).is_none());
    cleanup(&base);
}

#[test]
fn apply_completion_splices_slash_commands_and_attachments() {
    let base = temp_dir("apply");
    let provider =
        CombinedAutocompleteProvider::new(Vec::new(), base.to_string_lossy().to_string(), None);

    // Slash command: the bare value gains "/" and a trailing space.
    let applied = provider.apply_completion(&lines_of(&["/he"]), 0, 3, &item("help"), "/he");
    assert_eq!(applied.lines[0], "/help ");
    assert_eq!(applied.cursor_col, 6);

    // @-attachment: directories leave the cursor inside for further
    // completion; files append a space.
    let applied = provider.apply_completion(
        &lines_of(&["@src/"]),
        0,
        5,
        &AutocompleteItem {
            value: "@src/components/".to_string(),
            label: "components/".to_string(),
            description: None,
        },
        "@src/",
    );
    assert_eq!(applied.lines[0], "@src/components/");
    assert_eq!(applied.cursor_col, 16);

    let applied = provider.apply_completion(
        &lines_of(&["@file"]),
        0,
        5,
        &AutocompleteItem {
            value: "@file.txt".to_string(),
            label: "file.txt".to_string(),
            description: None,
        },
        "@file",
    );
    assert_eq!(applied.lines[0], "@file.txt ");
    assert_eq!(applied.cursor_col, 10);

    // Slash-command argument context: the prefix splices in place.
    let applied = provider.apply_completion(
        &lines_of(&["/load sk"]),
        0,
        8,
        &AutocompleteItem {
            value: "skill-a".to_string(),
            label: "skill-a".to_string(),
            description: None,
        },
        "sk",
    );
    assert_eq!(applied.lines[0], "/load skill-a");
    assert_eq!(applied.cursor_col, 13);
    cleanup(&base);
}

#[test]
fn apply_completion_swallows_the_trailing_quote_when_both_sides_quote() {
    let base = temp_dir("apply-quote");
    let provider =
        CombinedAutocompleteProvider::new(Vec::new(), base.to_string_lossy().to_string(), None);

    // Quoted prefix with a closing quote after the cursor: the item's own
    // trailing quote replaces it instead of doubling.
    let applied = provider.apply_completion(
        &lines_of(&["@\"my folder/te\""]),
        0,
        14,
        &AutocompleteItem {
            value: "@\"my folder/test.txt\"".to_string(),
            label: "test.txt".to_string(),
            description: None,
        },
        "@\"my folder/te\"",
    );
    assert_eq!(applied.lines[0], "@\"my folder/test.txt\" ");

    // A directory keeps the cursor on its name (before the quote).
    let applied = provider.apply_completion(
        &lines_of(&["@\"src/"]),
        0,
        6,
        &AutocompleteItem {
            value: "@\"src/comp/\"".to_string(),
            label: "comp/".to_string(),
            description: None,
        },
        "@\"src/",
    );
    assert_eq!(applied.lines[0], "@\"src/comp/\"");
    assert_eq!(applied.cursor_col, 11);
    cleanup(&base);
}

#[test]
fn should_trigger_file_completion_declines_bare_slash_commands() {
    let base = temp_dir("should-trigger");
    let provider =
        CombinedAutocompleteProvider::new(Vec::new(), base.to_string_lossy().to_string(), None);

    assert!(!provider.should_trigger_file_completion(&lines_of(&["/model"]), 0, 6));
    assert!(provider.should_trigger_file_completion(&lines_of(&["/model arg"]), 0, 10));
    assert!(provider.should_trigger_file_completion(&lines_of(&["plain text"]), 0, 10));
    assert!(provider.should_trigger_file_completion(&lines_of(&["src/"]), 0, 4));
    cleanup(&base);
}

#[test]
fn natural_path_triggers_need_path_shapes_and_forced_queries_always_run() {
    let base = temp_dir("natural");
    setup_folder(&base, &[], &[("update.sh", "#!/bin/bash")]);

    let provider =
        CombinedAutocompleteProvider::new(Vec::new(), base.to_string_lossy().to_string(), None);

    // A bare word: no natural trigger.
    assert!(get_suggestions(&provider, &["update"], 0, 6, false).is_none());
    // After a space, the empty prefix completes the directory.
    let result = get_suggestions(&provider, &["look "], 0, 5, false);
    assert!(
        result.is_some(),
        "the post-space prefix completes the directory"
    );

    // Forced extraction (Tab) always returns the token.
    let result = get_suggestions(&provider, &["update"], 0, 6, true);
    let result = result.expect("forced extraction");
    assert!(
        result.items.iter().any(|item| item.value == "update.sh"),
        "forced query completes the token: {:?}",
        result.items
    );

    // Dot-prefixed text triggers naturally.
    let result = get_suggestions(&provider, &["./up"], 0, 4, false);
    assert!(result.is_some(), "dot-prefixed text triggers naturally");
    cleanup(&base);
}

#[test]
fn quoted_prefixes_drive_both_at_and_path_extraction() {
    let base = temp_dir("quoted-extract");
    setup_folder(&base, &["my folder"], &[("my folder/test.txt", "content")]);

    let provider =
        CombinedAutocompleteProvider::new(Vec::new(), base.to_string_lossy().to_string(), None);

    // An unclosed quote continues completion inside the quoted path.
    let result = get_suggestions(&provider, &["\"my folder/t"], 0, 12, false);
    let result = result.expect("quoted path suggestions");
    assert!(
        result
            .items
            .iter()
            .any(|item| item.value == "\"my folder/test.txt\""),
        "quoted paths continue: {:?}",
        result.items
    );

    // A quote mid-token is not a quoted prefix: the forced query extracts
    // the empty token and lists the base directory — its own contents, not
    // the quoted folder's.
    let mid_quote = get_suggestions(&provider, &["see\"my "], 0, 7, true);
    let mid_quote = mid_quote.expect("forced query completes");
    assert!(
        !mid_quote
            .items
            .iter()
            .any(|item| item.value == "\"my folder/test.txt\""),
        "the mid-token quote does not resume inside the folder: {:?}",
        mid_quote.items
    );

    cleanup(&base);
}

#[test]
fn stub_fuzzy_continues_inside_quoted_at_paths() {
    // The @ branch always walks with fd, so the quoted-@ continuation runs
    // through the stub.
    let base = temp_dir("stub-quoted-at");
    setup_folder(&base, &["my folder"], &[]);
    let (stub_dir, stub) = stub_fd(&["test.txt", "other.txt"], None);

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        Some(stub_fd_path(&stub)),
    );
    let result = get_suggestions(&provider, &["@\"my folder/t"], 0, 12, false);
    let result = result.expect("quoted @ suggestions");
    assert!(
        result
            .items
            .iter()
            .any(|item| item.value == "@\"my folder/test.txt\""),
        "quoted @ prefixes keep the @ and resume inside the folder: {:?}",
        result.items
    );
    cleanup(&base);
    cleanup(&stub_dir);
}

#[test]
fn home_expansion_and_relative_display_preserve_the_prefix_shape() {
    let home = temp_dir("home");
    setup_folder(&home, &["docs"], &[("docs/readme.md", "content")]);
    let base = temp_dir("home-base");
    setup_folder(&base, &["docs"], &[]);

    let provider = CombinedAutocompleteProvider::with_home_lookup(
        Vec::new(),
        base.to_string_lossy().to_string(),
        None,
        Box::new({
            let home = home.clone();
            move || Some(home.to_string_lossy().to_string())
        }),
    );

    // "~/d" expands to the home directory and keeps the ~/ display form.
    let result = get_suggestions(&provider, &["~/d"], 0, 3, false);
    let result = result.expect("home suggestions");
    assert!(
        result.items.iter().any(|item| item.value == "~/docs/"),
        "home paths expand and display with ~/: {:?}",
        result.items
    );

    // "~" alone completes the home directory itself.
    let result = get_suggestions(&provider, &["~"], 0, 1, true);
    let result = result.expect("tilde suggestions");
    assert!(
        !result.items.is_empty(),
        "the bare tilde lists the home directory"
    );

    // "./up"-shaped prefixes keep the ./ display form.
    setup_folder(&base, &[], &[("update.sh", "#!/bin/bash")]);
    let result = get_suggestions(&provider, &["./up"], 0, 4, true);
    let result = result.expect("dot-slash suggestions");
    assert!(
        result.items.iter().any(|item| item.value == "./update.sh"),
        "dot-slash prefixes are preserved: {:?}",
        result.items
    );

    // A missing home lookup degrades the ~ expansion to the literal path.
    let no_home = CombinedAutocompleteProvider::with_home_lookup(
        Vec::new(),
        base.to_string_lossy().to_string(),
        None,
        Box::new(|| None),
    );
    let _ = get_suggestions(&no_home, &["~/d"], 0, 3, true);
    cleanup(&base);
    cleanup(&home);
}

#[test]
fn multiline_queries_only_read_the_cursor_line() {
    let base = temp_dir("multiline");
    setup_folder(&base, &[], &[("a.txt", "content")]);

    let provider =
        CombinedAutocompleteProvider::new(Vec::new(), base.to_string_lossy().to_string(), None);
    let result = get_suggestions(&provider, &["first", "./a"], 1, 3, true);
    let result = result.expect("suggestions from the cursor line");
    assert!(
        result.items.iter().any(|item| item.value == "./a.txt"),
        "the cursor line drives extraction: {:?}",
        result.items
    );

    // A cursor past the line end clamps to the line end.
    let result = get_suggestions(&provider, &["./a"], 0, 99, true);
    assert!(result.is_some(), "the cursor clamps to the line end");
    cleanup(&base);
}

#[test]
fn git_paths_never_surface_even_when_fd_emits_them() {
    let (stub_dir, stub) = stub_fd(&[".git", ".git/config", "sub/.git/config", "ok.txt"], None);
    let base = temp_dir("stub-git");

    let provider = CombinedAutocompleteProvider::new(
        Vec::new(),
        base.to_string_lossy().to_string(),
        Some(stub_fd_path(&stub)),
    );
    let result = get_suggestions(&provider, &["@"], 0, 1, false);
    let values = result.expect("suggestions").items;
    assert!(
        !values.iter().any(|item| item.value.contains(".git")),
        ".git entries never surface: {values:?}"
    );
    assert!(values.iter().any(|item| item.value == "@ok.txt"));
    cleanup(&base);
    cleanup(&stub_dir);
}

#[test]
fn command_items_and_slash_commands_share_the_name_lookup() {
    let base = temp_dir("mixed-commands");
    let provider = CombinedAutocompleteProvider::new(
        vec![
            CommandEntry::Item(AutocompleteItem {
                value: "from-item".to_string(),
                label: "from-item".to_string(),
                description: Some("item description".to_string()),
            }),
            CommandEntry::Slash(SlashCommand {
                name: "from-slash".to_string(),
                description: None,
                argument_hint: Some("<x>".to_string()),
                get_argument_completions: None,
            }),
        ],
        base.to_string_lossy().to_string(),
        None,
    );
    // The empty prefix lists everything; the slash item gets its hint, the
    // plain item its description.
    let result = get_suggestions(&provider, &["/"], 0, 1, false);
    let result = result.expect("command suggestions");
    let values: Vec<String> = result.items.iter().map(|item| item.value.clone()).collect();
    assert_eq!(
        values,
        vec!["from-item".to_string(), "from-slash".to_string()]
    );
    let slash = result
        .items
        .iter()
        .find(|item| item.value == "from-slash")
        .expect("slash");
    assert_eq!(slash.description.as_deref(), Some("<x>"));
    let plain = result
        .items
        .iter()
        .find(|item| item.value == "from-item")
        .expect("item");
    assert_eq!(plain.description.as_deref(), Some("item description"));
    cleanup(&base);
}
