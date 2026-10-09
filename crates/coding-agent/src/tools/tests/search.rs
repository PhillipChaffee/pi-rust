//! The grep/find/ls blocks of upstream's `tools.test.ts`, the
//! `3302-find-path-glob` and `3303-find-nested-gitignore` regressions, and
//! the boundary tests binding the native glob/gitignore restatement.

#![expect(
    clippy::unwrap_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use serde_json::json;

use crate::extensions::types::{CwdContext, ToolDefinition};
use crate::tools::find::{create_find_tool_definition, relativize_find_result_path};
use crate::tools::grep::{GrepOperations, GrepToolOptions, create_grep_tool_definition};
use crate::tools::ls::create_ls_tool_definition;

use std::sync::Arc;

use pi_agent_core::harness::context::{background_context, with_cancel};

use super::file_tools::{first_text, run_definition};
use super::helpers::block_on;

fn find_lines(text: &str) -> Vec<String> {
    text.split('\n')
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('['))
        .map(str::to_owned)
        .collect()
}

fn make_tree(dir: &std::path::Path) {
    std::fs::create_dir_all(dir.join("some/parent/child")).unwrap();
    std::fs::create_dir_all(dir.join("src/foo/bar")).unwrap();
    std::fs::write(dir.join("some/parent/child/file.ext"), "").unwrap();
    std::fs::write(dir.join("some/parent/child/test.spec.ts"), "").unwrap();
    std::fs::write(dir.join("src/foo/bar/example.spec.ts"), "").unwrap();
}

// ---------------------------------------------------------------------------
// grep tool
// ---------------------------------------------------------------------------

#[test]
fn grep_includes_filename_when_searching_a_single_file() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("example.txt");
        std::fs::write(&test_file, "first line\nmatch line\nlast line").unwrap();
        let tool = create_grep_tool_definition("/", None);

        let result = run_definition(
            &tool,
            json!({ "pattern": "match", "path": test_file.to_string_lossy() }),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(
            text_contains(&result, "example.txt:2: match line"),
            "{}",
            text_of(&result)
        );
    });
}

fn text_of(result: &pi_agent_core::types::AgentToolResult) -> String {
    super::helpers::text_output(result)
}

fn text_contains(result: &pi_agent_core::types::AgentToolResult, needle: &str) -> bool {
    text_of(result).contains(needle)
}

#[test]
fn grep_respects_global_limit_and_includes_context_lines() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("context.txt");
        let content = [
            "before",
            "match one",
            "after",
            "middle",
            "match two",
            "after two",
        ]
        .join("\n");
        std::fs::write(&test_file, content).unwrap();
        let tool = create_grep_tool_definition("/", None);

        let result = run_definition(
            &tool,
            json!({ "pattern": "match", "path": test_file.to_string_lossy(), "limit": 1, "context": 1 }),
            None,
            None,
        )
        .await
        .unwrap();
        let output = text_of(&result);
        assert!(output.contains("context.txt-1- before"), "{output}");
        assert!(output.contains("context.txt:2: match one"), "{output}");
        assert!(output.contains("context.txt-3- after"), "{output}");
        assert!(
            output.contains("[1 matches limit reached. Use limit=2 for more, or refine pattern]"),
            "{output}"
        );
        // Ensure second match is not present
        assert!(!output.contains("match two"), "{output}");
    });
}

#[test]
fn grep_treats_flag_like_patterns_as_search_text() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("grep-injection-marker");
        let payload = dir.path().join("payload.sh");
        let test_file = dir.path().join("target.txt");
        std::fs::write(
            &payload,
            format!(
                "#!/bin/sh\necho executed > {}\ncat \"$1\"\n",
                marker.to_string_lossy()
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&payload, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        std::fs::write(&test_file, "target\n").unwrap();
        let tool = create_grep_tool_definition("/", None);

        let result = run_definition(
            &tool,
            json!({
                "pattern": format!("--pre={}", payload.to_string_lossy()),
                "path": dir.path().to_string_lossy()
            }),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(
            text_contains(&result, "No matches found"),
            "{}",
            text_of(&result)
        );
        assert!(!marker.exists(), "the payload executed");
    });
}

// ---------------------------------------------------------------------------
// find tool
// ---------------------------------------------------------------------------

#[test]
fn find_includes_hidden_files_that_are_not_gitignored() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let hidden_dir = dir.path().join(".secret");
        std::fs::create_dir(&hidden_dir).unwrap();
        std::fs::write(hidden_dir.join("hidden.txt"), "hidden").unwrap();
        std::fs::write(dir.path().join("visible.txt"), "visible").unwrap();
        let tool = create_find_tool_definition("/", None);

        let result = run_definition(
            &tool,
            json!({ "pattern": "**/*.txt", "path": dir.path().to_string_lossy() }),
            None,
            None,
        )
        .await
        .unwrap();
        let lines: Vec<String> = find_lines(&text_of(&result));
        assert!(lines.contains(&"visible.txt".to_owned()), "{lines:?}");
        assert!(
            lines.contains(&".secret/hidden.txt".to_owned()),
            "{lines:?}"
        );
    });
}

#[test]
fn find_respects_gitignore() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(dir.path().join("ignored.txt"), "ignored").unwrap();
        std::fs::write(dir.path().join("kept.txt"), "kept").unwrap();
        let tool = create_find_tool_definition("/", None);

        let result = run_definition(
            &tool,
            json!({ "pattern": "**/*.txt", "path": dir.path().to_string_lossy() }),
            None,
            None,
        )
        .await
        .unwrap();
        let output = text_of(&result);
        assert!(output.contains("kept.txt"), "{output}");
        assert!(!output.contains("ignored.txt"), "{output}");
    });
}

#[test]
fn find_surfaces_glob_parse_errors() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let tool = create_find_tool_definition("/", None);

        let error = run_definition(
            &tool,
            json!({ "pattern": "[", "path": dir.path().to_string_lossy() }),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(
            error
                .to_string()
                .to_lowercase()
                .contains("error parsing glob"),
            "{}",
            error
        );
    });
}

#[test]
fn find_treats_flag_like_patterns_as_search_text() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let tool = create_find_tool_definition("/", None);

        let result = run_definition(
            &tool,
            json!({ "pattern": "--help", "path": dir.path().to_string_lossy() }),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(
            text_contains(&result, "No files found matching pattern"),
            "{}",
            text_of(&result)
        );
    });
}

// ---------------------------------------------------------------------------
// ls tool
// ---------------------------------------------------------------------------

#[test]
fn ls_lists_dotfiles_and_directories() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(".hidden-file"), "secret").unwrap();
        std::fs::create_dir(dir.path().join(".hidden-dir")).unwrap();
        let tool = create_ls_tool_definition("/", None);

        let result = run_definition(
            &tool,
            json!({ "path": dir.path().to_string_lossy() }),
            None,
            None,
        )
        .await
        .unwrap();
        let output = text_of(&result);
        assert!(output.contains(".hidden-file"), "{output}");
        assert!(output.contains(".hidden-dir/"), "{output}");
    });
}

// ---------------------------------------------------------------------------
// tool cwd resolution: grep / find
// ---------------------------------------------------------------------------

#[test]
fn grep_uses_ctx_cwd_when_provided() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("ctx-cwd-grep.txt"), "match in ctx.cwd").unwrap();
        let tool = create_grep_tool_definition("/", None);
        let context = CwdContext {
            cwd: dir.path().to_string_lossy().into_owned(),
        };
        let result = run_definition(&tool, json!({ "pattern": "match" }), None, Some(&context))
            .await
            .unwrap();
        assert!(
            text_contains(&result, "ctx-cwd-grep.txt"),
            "{}",
            text_of(&result)
        );
    });
}

#[test]
fn find_uses_ctx_cwd_when_provided() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("ctx-cwd-find.txt"), "find me").unwrap();
        let tool = create_find_tool_definition("/", None);
        let context = CwdContext {
            cwd: dir.path().to_string_lossy().into_owned(),
        };
        let result = run_definition(
            &tool,
            json!({ "pattern": "ctx-cwd-find.txt" }),
            None,
            Some(&context),
        )
        .await
        .unwrap();
        assert!(
            text_contains(&result, "ctx-cwd-find.txt"),
            "{}",
            text_of(&result)
        );
    });
}

// ---------------------------------------------------------------------------
// Regression 3302: find returns no results for path-based glob patterns
// ---------------------------------------------------------------------------

async fn run_find_pattern(tool: &ToolDefinition, pattern: &str, root: &str) -> Vec<String> {
    let result = run_definition(
        tool,
        json!({ "pattern": pattern, "path": root }),
        None,
        None,
    )
    .await
    .unwrap();
    let text = first_text(&result);
    if text == "No files found matching pattern" {
        return Vec::new();
    }
    find_lines(&text)
}

#[test]
fn regression_3302_find_returns_path_glob_results() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        make_tree(dir.path());
        let root = dir.path().to_string_lossy().into_owned();
        let tool = create_find_tool_definition("/", None);

        // Basename pattern still matches (regression-safe).
        let mut files = run_find_pattern(&tool, "*.spec.ts", &root).await;
        files.sort();
        assert_eq!(
            files,
            vec![
                "some/parent/child/test.spec.ts".to_owned(),
                "src/foo/bar/example.spec.ts".to_owned()
            ]
        );

        // Directory-prefixed pattern with ** tail matches subtree.
        let files = run_find_pattern(&tool, "some/parent/child/**", &root).await;
        assert!(
            files.contains(&"some/parent/child/file.ext".to_owned()),
            "{files:?}"
        );
        assert!(
            files.contains(&"some/parent/child/test.spec.ts".to_owned()),
            "{files:?}"
        );

        // Leading ** wildcard with path segments matches.
        let files = run_find_pattern(&tool, "**/parent/child/*", &root).await;
        assert!(
            files.contains(&"some/parent/child/file.ext".to_owned()),
            "{files:?}"
        );
        assert!(
            files.contains(&"some/parent/child/test.spec.ts".to_owned()),
            "{files:?}"
        );

        // src/**/*.spec.ts matches the nested spec file.
        let files = run_find_pattern(&tool, "src/**/*.spec.ts", &root).await;
        assert_eq!(files, vec!["src/foo/bar/example.spec.ts".to_owned()]);
    });
}

// ---------------------------------------------------------------------------
// Regression 3303: nested .gitignore rules leak into sibling directories
// ---------------------------------------------------------------------------

#[test]
fn regression_3303_flat_sibling_gitignore_scoping() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("a")).unwrap();
        std::fs::create_dir(dir.path().join("b")).unwrap();
        std::fs::write(dir.path().join("a/.gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(dir.path().join("a/ignored.txt"), "").unwrap();
        std::fs::write(dir.path().join("a/kept.txt"), "").unwrap();
        std::fs::write(dir.path().join("b/ignored.txt"), "").unwrap();
        std::fs::write(dir.path().join("b/kept.txt"), "").unwrap();
        std::fs::write(dir.path().join("root.txt"), "").unwrap();
        let root = dir.path().to_string_lossy().into_owned();
        let tool = create_find_tool_definition("/", None);

        let mut files = run_find_pattern(&tool, "**/*.txt", &root).await;
        files.sort();
        assert_eq!(
            files,
            vec![
                "a/kept.txt".to_owned(),
                "b/ignored.txt".to_owned(),
                "b/kept.txt".to_owned(),
                "root.txt".to_owned(),
            ]
        );
    });
}

#[test]
fn regression_3303_deeply_nested_gitignore_scoping() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("a/deep")).unwrap();
        std::fs::create_dir(dir.path().join("b")).unwrap();
        std::fs::write(dir.path().join("a/.gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(dir.path().join("a/deep/.gitignore"), "secret.txt\n").unwrap();
        std::fs::write(dir.path().join("a/ignored.txt"), "").unwrap();
        std::fs::write(dir.path().join("a/kept.txt"), "").unwrap();
        std::fs::write(dir.path().join("a/deep/ignored.txt"), "").unwrap();
        std::fs::write(dir.path().join("a/deep/secret.txt"), "").unwrap();
        std::fs::write(dir.path().join("a/deep/kept.txt"), "").unwrap();
        std::fs::write(dir.path().join("b/ignored.txt"), "").unwrap();
        std::fs::write(dir.path().join("b/kept.txt"), "").unwrap();
        std::fs::write(dir.path().join("root.txt"), "").unwrap();
        let root = dir.path().to_string_lossy().into_owned();
        let tool = create_find_tool_definition("/", None);

        let mut files = run_find_pattern(&tool, "**/*.txt", &root).await;
        files.sort();
        assert_eq!(
            files,
            vec![
                "a/deep/kept.txt".to_owned(),
                "a/kept.txt".to_owned(),
                "b/ignored.txt".to_owned(),
                "b/kept.txt".to_owned(),
                "root.txt".to_owned(),
            ]
        );
    });
}

// ---------------------------------------------------------------------------
// Boundary tests: the restated matcher surfaces
// ---------------------------------------------------------------------------

#[test]
fn relativize_find_result_path_normalizes_to_posix() {
    assert_eq!(
        relativize_find_result_path("/root/sub/file.txt", "/root"),
        "sub/file.txt"
    );
    assert_eq!(relativize_find_result_path("file.txt", "/root"), "file.txt");
    assert_eq!(relativize_find_result_path("/root/dir/", "/root"), "dir/");
    assert_eq!(
        relativize_find_result_path("/elsewhere/x.txt", "/root"),
        "../elsewhere/x.txt"
    );
}

#[test]
fn find_inside_git_repos_keeps_the_default_git_requirement() {
    // The walk-up probe: a repo root pins require-git on, the outside case
    // rides the `--no-require-git` arm. Bound through the observable walk:
    // outside a repo, the .gitignore still applies (the fix's behavior).
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(dir.path().join("ignored.txt"), "").unwrap();
        std::fs::write(dir.path().join("kept.txt"), "").unwrap();
        let tool = create_find_tool_definition("/", None);
        let result = run_definition(
            &tool,
            json!({ "pattern": "**/*.txt", "path": dir.path().to_string_lossy() }),
            None,
            None,
        )
        .await
        .unwrap();
        let output = first_text(&result);
        assert!(output.contains("kept.txt"), "{output}");
        assert!(!output.contains("ignored.txt"), "{output}");
    });
}

#[test]
fn grep_applies_the_glob_filter_to_the_walked_files() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("keep.spec.ts"), "needle here").unwrap();
        std::fs::write(dir.path().join("skip.txt"), "needle here").unwrap();
        let tool = create_grep_tool_definition("/", None);
        let result = run_definition(
            &tool,
            json!({
                "pattern": "needle",
                "path": dir.path().to_string_lossy(),
                "glob": "**/*.spec.ts"
            }),
            None,
            None,
        )
        .await
        .unwrap();
        let output = text_of(&result);
        assert!(output.contains("keep.spec.ts"), "{output}");
        assert!(!output.contains("skip.txt"), "{output}");
    });
}

#[test]
fn grep_rejects_an_uncompilable_pattern() {
    block_on(async {
        let tool = create_grep_tool_definition("/", None);
        let error = run_definition(&tool, json!({ "pattern": "([" }), None, None)
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("Failed to run ripgrep:"),
            "{error}"
        );
    });
}

#[test]
fn grep_reports_a_missing_search_path() {
    block_on(async {
        let tool = create_grep_tool_definition("/", None);
        let error = run_definition(
            &tool,
            json!({ "pattern": "x", "path": "/pi-coding-agent-no-such-dir" }),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("Path not found"), "{error}");
    });
}

#[test]
fn grep_reports_a_missing_single_file_path() {
    block_on(async {
        let tool = create_grep_tool_definition("/", None);
        let error = run_definition(
            &tool,
            json!({ "pattern": "x", "path": "/pi-coding-agent-no-such-file.txt" }),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("Path not found"), "{error}");
    });
}

#[test]
fn find_reports_an_abort_before_the_walk() {
    block_on(async {
        let (_context, controller) = with_cancel(&background_context());
        controller.abort_without_reason();
        let tool = create_find_tool_definition("/", None);
        let error = run_definition(
            &tool,
            json!({ "pattern": "**/*.txt" }),
            Some(controller.signal()),
            None,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("Operation aborted"), "{error}");
    });
}

#[test]
fn find_flags_the_result_limit_and_wires_the_details() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        for index in 0..5 {
            std::fs::write(dir.path().join(format!("file-{index}.txt")), "").unwrap();
        }
        let tool = create_find_tool_definition("/", None);
        let result = run_definition(
            &tool,
            json!({ "pattern": "**/*.txt", "path": dir.path().to_string_lossy(), "limit": 2 }),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(text_of(&result).contains("results limit reached"));
        assert_eq!(result.details["resultLimitReached"], 2);
    });
}

#[test]
fn grep_flags_the_match_limit_and_wires_the_details() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        for index in 0..5 {
            std::fs::write(dir.path().join(format!("file-{index}.txt")), "needle\n").unwrap();
        }
        let tool = create_grep_tool_definition("/", None);
        let result = run_definition(
            &tool,
            json!({ "pattern": "needle", "path": dir.path().to_string_lossy(), "limit": 2 }),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(text_of(&result).contains("matches limit reached"));
        assert_eq!(result.details["matchLimitReached"], 2);
    });
}

#[test]
fn grep_reports_an_unreadable_file_with_context() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("gone.txt"), "needle\n").unwrap();
        let operations = GrepOperations {
            is_directory: Arc::new(|_path: String| Box::pin(async { Some(true) })),
            read_file: Arc::new(|_path: String| {
                Box::pin(async { Err("permission denied".to_owned()) })
            }),
        };
        let tool = create_grep_tool_definition(
            "/",
            Some(GrepToolOptions {
                operations: Some(operations),
            }),
        );
        let result = run_definition(
            &tool,
            json!({ "pattern": "needle", "path": dir.path().to_string_lossy(), "context": 1 }),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(text_of(&result).contains("(unable to read file)"));
    });
}

#[test]
fn find_rejects_an_uncompilable_glob() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("plain.txt"), "").unwrap();
        let tool = create_find_tool_definition("/", None);
        let error = run_definition(
            &tool,
            json!({ "pattern": "[unclosed", "path": dir.path().to_string_lossy() }),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("error parsing glob"), "{error}");
    });
}
