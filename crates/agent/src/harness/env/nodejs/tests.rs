//! The nodejs execution environment suite, ported from upstream
//! `test/harness/nodejs-env.test.ts`.
//!
//! Upstream's fake-`process.platform` tests (the legacy WSL stdin transport
//! and the win32 `taskkill` branches) have no Rust slot — the platform is
//! compile-time — so those ride the map's win32 ticket. The callback-error
//! test rides the same cut: Rust update callbacks cannot throw, so the
//! handler-error plumbing carries no equivalent trigger.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex};

use pi_ai::types::BoxedFuture;
use pi_chord::context::{Context, background_context, with_cancel};

use crate::harness::env::nodejs::{NodeExecutionEnv, SPILL_FILE_PREFIX};
use crate::harness::types::{
    ExecutionErrorCode, ShellOutputCaptureOptions, ShellOutputLimits, TextLineReader,
};
use crate::harness::types::{
    CreateDirOptions, ExecutionEnv, FileContent, FileErrorCode, FileKind, FileSystem,
    ReadTextLinesOptions, RemoveOptions, Shell, ShellExecOptions, ShellExecResult,
    ShellOutputUpdate, ShellOutputView, TempFileOptions,
};
use crate::harness::utils::output_capture::apply_shell_output_update;
use crate::harness::utils::shell_output::execute_shell_with_capture;

fn context() -> Context {
    background_context()
}

fn aborted_context() -> Context {
    let (context, controller) = with_cancel(&background_context());
    controller.abort("aborted");
    context
}

/// Collects the bounded output view through `onUpdate`, upstream's
/// `collectShellOutput`.
async fn collect_shell_output(
    env: &NodeExecutionEnv,
    command: &str,
    options: Option<ShellExecOptions>,
    context: &Context,
) -> (
    Result<ShellExecResult, crate::harness::types::ExecutionError>,
    Option<ShellOutputView>,
) {
    let collected: Arc<Mutex<Option<ShellOutputView>>> = Arc::new(Mutex::new(None));
    let reducer = Arc::clone(&collected);
    let mut options = options.unwrap_or_default();
    options.on_update = Some(Arc::new(move |update: &ShellOutputUpdate, _context| {
        let previous = reducer.lock().expect("output lock").take();
        *reducer.lock().expect("output lock") =
            Some(apply_shell_output_update(previous.as_ref(), update));
    }));
    let result = Shell::exec(env, command, Some(options), context).await;
    let output = collected.lock().expect("output lock").take();
    (result, output)
}

/// The reads, writes, listings, and removals round-trip through the
/// environment's cwd.
#[tokio::test]
async fn reads_writes_lists_and_removes_files_and_directories() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let context = context();
    assert_eq!(
        FileSystem::absolute_path(&env, "nested/child", &context)
            .await
            .expect("absolute path"),
        format!("{root}/nested/child")
    );
    assert_eq!(
        FileSystem::join_path(
            &env,
            &[root.clone(), "nested".to_owned(), "child".to_owned()],
            &context
        )
        .await
        .expect("join path"),
        format!("{root}/nested/child")
    );
    FileSystem::create_dir(&env, "nested/child", None, &context)
        .await
        .expect("create dir");
    FileSystem::write_file(&env, "nested/child/file.txt", "hel".into(), &context)
        .await
        .expect("write");
    FileSystem::append_file(&env, "nested/child/file.txt", "lo".into(), &context)
        .await
        .expect("append");
    assert_eq!(
        FileSystem::read_text_file(&env, "nested/child/file.txt", &context)
            .await
            .expect("read text"),
        "hello"
    );
    assert_eq!(
        FileSystem::read_text_lines(
            &env,
            "nested/child/file.txt",
            Some(ReadTextLinesOptions { max_lines: Some(1) }),
            &context
        )
        .await
        .expect("read lines"),
        vec!["hello"]
    );
    assert_eq!(
        FileSystem::read_binary_file(&env, "nested/child/file.txt", &context)
            .await
            .expect("read binary"),
        b"hello".to_vec()
    );

    let entries = FileSystem::list_dir(&env, "nested/child", &context)
        .await
        .expect("list dir");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].name, "file.txt");
    assert_eq!(entries[0].path, format!("{root}/nested/child/file.txt"));
    assert_eq!(entries[0].kind, FileKind::File);
    assert_eq!(entries[0].size, 5);

    assert!(
        FileSystem::exists(&env, "nested/child/file.txt", &context)
            .await
            .expect("exists")
    );
    FileSystem::remove(&env, "nested/child/file.txt", None, &context)
        .await
        .expect("remove");
    assert!(
        !FileSystem::exists(&env, "nested/child/file.txt", &context)
            .await
            .expect("exists after remove")
    );
}

fn env_at(root: &str) -> NodeExecutionEnv {
    NodeExecutionEnv::new(root.to_owned(), None, None)
}

/// Home-relative paths expand to the caller's home and `file://` URLs
/// resolve to their percent-decoded paths.
#[tokio::test]
async fn expands_home_relative_paths_and_file_urls() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let context = context();
    let home = std::env::var("HOME").expect("HOME is set");
    assert_eq!(
        FileSystem::absolute_path(&env, "~/pi-node-env-test", &context)
            .await
            .expect("home expansion"),
        format!("{home}/pi-node-env-test")
    );
    let file_path = format!("{root}/file with spaces.txt");
    assert_eq!(
        FileSystem::absolute_path(
            &env,
            &format!("file://{root}/file%20with%20spaces.txt"),
            &context
        )
        .await
        .expect("file url"),
        file_path
    );
}

/// File and directory metadata reports kinds without following symlinks.
#[tokio::test]
async fn file_info_reports_kinds_without_following_symlinks() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let context = context();
    FileSystem::create_dir(
        &env,
        "dir",
        Some(CreateDirOptions {
            recursive: Some(true),
        }),
        &context,
    )
    .await
    .expect("create dir");
    FileSystem::write_file(&env, "dir/file.txt", "hello".into(), &context)
        .await
        .expect("write file");
    std::os::unix::fs::symlink(format!("{root}/dir/file.txt"), format!("{root}/file-link"))
        .expect("symlink file");
    std::os::unix::fs::symlink(format!("{root}/dir"), format!("{root}/dir-link"))
        .expect("symlink dir");

    let info = FileSystem::file_info(&env, "dir", &context)
        .await
        .expect("dir info");
    assert_eq!(info.kind, FileKind::Directory);
    let info = FileSystem::file_info(&env, "dir/file.txt", &context)
        .await
        .expect("file info");
    assert_eq!(info.kind, FileKind::File);
    assert_eq!(info.size, 5);
    let info = FileSystem::file_info(&env, "file-link", &context)
        .await
        .expect("file link info");
    assert_eq!(info.kind, FileKind::Symlink);
    let info = FileSystem::file_info(&env, "dir-link", &context)
        .await
        .expect("dir link info");
    assert_eq!(info.kind, FileKind::Symlink);
    let canonical = FileSystem::canonical_path(&env, "file-link", &context)
        .await
        .expect("canonical");
    let expected = std::fs::canonicalize(format!("{root}/dir/file.txt"))
        .expect("realpath")
        .to_string_lossy()
        .into_owned();
    assert_eq!(canonical, expected);
}

/// `listDir` reports symlinks as symlinks without following them.
#[tokio::test]
async fn lists_symlinks_as_symlinks() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let context = context();
    FileSystem::write_file(&env, "target.txt", "hello".into(), &context)
        .await
        .expect("write");
    std::os::unix::fs::symlink(format!("{root}/target.txt"), format!("{root}/link.txt"))
        .expect("symlink");

    let entries = FileSystem::list_dir(&env, ".", &context)
        .await
        .expect("list");
    let mut named: Vec<(String, FileKind)> = entries
        .into_iter()
        .map(|entry| (entry.name, entry.kind))
        .collect();
    named.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        named,
        vec![
            ("link.txt".to_owned(), FileKind::Symlink),
            ("target.txt".to_owned(), FileKind::File),
        ]
    );
}

/// The line reader stops at the requested limit.
#[tokio::test]
async fn stops_reading_text_lines_at_the_requested_limit() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let context = context();
    FileSystem::write_file(&env, "file.txt", "one\ntwo\nthree".into(), &context)
        .await
        .expect("write");
    assert_eq!(
        FileSystem::read_text_lines(
            &env,
            "file.txt",
            Some(ReadTextLinesOptions { max_lines: Some(1) }),
            &context
        )
        .await
        .expect("read lines"),
        vec!["one"]
    );
}

/// Missing paths report `not_found` with the resolved path, and `exists`
/// answers `false` rather than erroring.
#[tokio::test]
async fn missing_paths_report_not_found_and_exists_false() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let context = context();
    let error = FileSystem::file_info(&env, "missing.txt", &context)
        .await
        .expect_err("missing path errors");
    assert_eq!(error.code, FileErrorCode::NotFound);
    assert_eq!(
        error.path.as_deref(),
        Some(format!("{root}/missing.txt").as_str())
    );
    assert!(
        !FileSystem::exists(&env, "missing.txt", &context)
            .await
            .expect("exists false")
    );
}

/// Listing a non-directory reports `not_directory`.
#[tokio::test]
async fn listing_a_non_directory_reports_not_directory() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let context = context();
    FileSystem::write_file(&env, "file.txt", "hello".into(), &context)
        .await
        .expect("write");
    let error = FileSystem::list_dir(&env, "file.txt", &context)
        .await
        .expect_err("list dir on a file errors");
    assert_eq!(error.code, FileErrorCode::NotDirectory);
}

/// Appends create missing parents and build content incrementally.
#[tokio::test]
async fn appends_to_new_files_and_creates_parent_directories() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let context = context();
    FileSystem::append_file(&env, "new/nested/file.txt", "a".into(), &context)
        .await
        .expect("append a");
    FileSystem::append_file(&env, "new/nested/file.txt", "b".into(), &context)
        .await
        .expect("append b");
    assert_eq!(
        FileSystem::read_text_file(&env, "new/nested/file.txt", &context)
            .await
            .expect("read"),
        "ab"
    );
}

/// Renames replace the destination atomically.
#[tokio::test]
async fn renames_atomically_and_replaces_the_destination() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let context = context();
    FileSystem::write_file(&env, "source.txt", "new".into(), &context)
        .await
        .expect("write source");
    FileSystem::write_file(&env, "destination.txt", "old".into(), &context)
        .await
        .expect("write destination");
    FileSystem::rename_file(&env, "source.txt", "destination.txt", &context)
        .await
        .expect("rename");
    assert!(
        !FileSystem::exists(&env, "source.txt", &context)
            .await
            .expect("source gone")
    );
    assert_eq!(
        FileSystem::read_text_file(&env, "destination.txt", &context)
            .await
            .expect("read destination"),
        "new"
    );
}

/// A missing rename source reports `not_found` with the source path and
/// leaves the destination untouched.
#[tokio::test]
async fn rename_reports_the_source_path_when_the_source_is_missing() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let context = context();
    FileSystem::write_file(&env, "destination.txt", "unchanged".into(), &context)
        .await
        .expect("write destination");
    let error = FileSystem::rename_file(&env, "missing-source.txt", "destination.txt", &context)
        .await
        .expect_err("missing source errors");
    assert_eq!(error.code, FileErrorCode::NotFound);
    assert_eq!(
        error.path.as_deref(),
        Some(format!("{root}/missing-source.txt").as_str())
    );
    assert_eq!(
        FileSystem::read_text_file(&env, "destination.txt", &context)
            .await
            .expect("read"),
        "unchanged"
    );
}

/// Temporary directories and files persist under the platform temp root.
#[tokio::test]
async fn creates_temporary_directories_and_files() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path().to_string_lossy().as_ref());
    let context = context();
    let temp_dir = FileSystem::create_temp_dir(&env, Some("node-env-test-"), &context)
        .await
        .expect("temp dir");
    assert!(std::fs::metadata(&temp_dir).is_ok());
    let temp_file = FileSystem::create_temp_file(
        &env,
        Some(TempFileOptions {
            prefix: Some("prefix-".to_owned()),
            suffix: Some(".txt".to_owned()),
        }),
        &context,
    )
    .await
    .expect("temp file");
    assert!(
        std::path::Path::new(&temp_file)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("txt"))
    );
    assert!(std::fs::metadata(&temp_file).is_ok());
}

/// `createDir` honors `recursive: false` and `remove` honors the
/// recursive/force options.
#[tokio::test]
async fn honors_create_dir_and_remove_options() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let context = context();
    let error = FileSystem::create_dir(
        &env,
        "missing/child",
        Some(CreateDirOptions {
            recursive: Some(false),
        }),
        &context,
    )
    .await
    .expect_err("non-recursive create of a deep path errors");
    assert_eq!(error.code, FileErrorCode::NotFound);

    FileSystem::write_file(&env, "dir/child/file.txt", "hello".into(), &context)
        .await
        .expect("write");
    let error = FileSystem::remove(
        &env,
        "dir",
        Some(RemoveOptions {
            recursive: Some(false),
            force: None,
        }),
        &context,
    )
    .await
    .expect_err("non-recursive dir removal errors");
    let _ = error;
    FileSystem::remove(
        &env,
        "dir",
        Some(RemoveOptions {
            recursive: Some(true),
            force: None,
        }),
        &context,
    )
    .await
    .expect("recursive removal");
    assert!(
        !FileSystem::exists(&env, "dir", &context)
            .await
            .expect("dir gone")
    );

    let error = FileSystem::remove(
        &env,
        "missing",
        Some(RemoveOptions {
            recursive: None,
            force: Some(false),
        }),
        &context,
    )
    .await
    .expect_err("missing removal without force errors");
    assert_eq!(error.code, FileErrorCode::NotFound);
    FileSystem::remove(
        &env,
        "missing",
        Some(RemoveOptions {
            recursive: None,
            force: Some(true),
        }),
        &context,
    )
    .await
    .expect("forced removal ignores the missing path");
}

/// Pre-aborted cancellable file operations report `aborted`.
#[tokio::test]
async fn pre_aborted_operations_report_aborted() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let context = context();
    FileSystem::write_file(&env, "file.txt", "hello".into(), &context)
        .await
        .expect("write");
    let aborted = aborted_context();

    let results = (
        FileSystem::read_text_file(&env, "file.txt", &aborted).await,
        FileSystem::read_text_lines(&env, "file.txt", None, &aborted).await,
        FileSystem::read_binary_file(&env, "file.txt", &aborted).await,
        FileSystem::write_file(&env, "other.txt", "hello".into(), &aborted).await,
        FileSystem::rename_file(&env, "file.txt", "renamed.txt", &aborted).await,
        FileSystem::list_dir(&env, ".", &aborted).await,
    );
    for error in [
        results.0.expect_err("read text aborted"),
        results.1.expect_err("read lines errors"),
        results.2.expect_err("read binary errors"),
        results.3.expect_err("write errors"),
        results.4.expect_err("rename errors"),
        results.5.expect_err("list dir errors"),
    ] {
        assert_eq!(error.code, FileErrorCode::Aborted);
    }
}

/// `cleanup` on an idle environment is best-effort and resolves.
#[tokio::test]
async fn cleanup_is_best_effort() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path().to_string_lossy().as_ref());
    Shell::cleanup(&env, &context()).await;
}

/// Commands execute in the cwd with the environment overrides applied.
#[tokio::test]
async fn executes_commands_in_cwd_with_env_overrides() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let mut env_overrides = BTreeMap::new();
    env_overrides.insert("NODE_ENV_TEST".to_owned(), "ok".to_owned());
    let (result, output) = collect_shell_output(
        &env,
        "printf '%s:%s' \"$PWD\" \"$NODE_ENV_TEST\"",
        Some(ShellExecOptions {
            env: Some(env_overrides),
            ..ShellExecOptions::default()
        }),
        &context(),
    )
    .await;
    let result = result.expect("exec ok");
    assert_eq!(result.exit_code, 0);
    let realpath = std::fs::canonicalize(&root)
        .expect("realpath")
        .to_string_lossy()
        .into_owned();
    assert_eq!(output.expect("output").text, format!("{realpath}:ok"));
}

/// Shell environment overrides: a missing override preserves the base
/// value, an empty override shadows it, and a string override replaces it.
#[tokio::test]
async fn applies_string_shell_environment_overrides() {
    let base = BTreeMap::from([
        (
            "PI_SESSION_FILE".to_owned(),
            "/stale/parent.jsonl".to_owned(),
        ),
        ("PI_CODING_AGENT".to_owned(), "true".to_owned()),
        (
            "PI_NODE_ENV_PRESERVED_TEST".to_owned(),
            "preserved".to_owned(),
        ),
    ]);
    for (overrides, expected_session_file) in [
        (None, "x:/stale/parent.jsonl"),
        (
            Some(BTreeMap::from([(
                "PI_SESSION_FILE".to_owned(),
                String::new(),
            )])),
            "x:",
        ),
        (
            Some(BTreeMap::from([(
                "PI_SESSION_FILE".to_owned(),
                "/sessions/current.jsonl".to_owned(),
            )])),
            "x:/sessions/current.jsonl",
        ),
    ] {
        let root = tempfile::tempdir().expect("temp root");
        let env = NodeExecutionEnv::new(
            root.path().to_string_lossy().into_owned(),
            None,
            Some(base.clone()),
        );
        let command = "printf '%s:%s|%s|%s' \"${PI_SESSION_FILE+x}\" \"${PI_SESSION_FILE-}\" \"$PI_CODING_AGENT\" \"$PI_NODE_ENV_PRESERVED_TEST\"";
        let (result, output) = collect_shell_output(
            &env,
            command,
            Some(ShellExecOptions {
                env: overrides,
                ..ShellExecOptions::default()
            }),
            &context(),
        )
        .await;
        result.expect("exec");
        assert_eq!(
            output.expect("output").text,
            format!("{expected_session_file}|true|preserved")
        );
    }
}

/// `inheritEnv: false` replaces the default shell environment wholesale.
#[tokio::test]
async fn can_replace_rather_than_inherit_the_default_shell_environment() {
    let root = tempfile::tempdir().expect("temp root");
    let env = NodeExecutionEnv::new(
        root.path().to_string_lossy().into_owned(),
        None,
        Some(BTreeMap::from([(
            "PI_NODE_ENV_CONFIGURED_TEST".to_owned(),
            "configured".to_owned(),
        )])),
    );
    let mut env_overrides = BTreeMap::new();
    env_overrides.insert(
        "PI_NODE_ENV_EXPLICIT_TEST".to_owned(),
        "explicit".to_owned(),
    );
    let (result, output) = collect_shell_output(
        &env,
        "printf '%s:%s:%s' \"${PI_NODE_ENV_INHERITED_TEST-}\" \"${PI_NODE_ENV_CONFIGURED_TEST-}\" \"${PI_NODE_ENV_EXPLICIT_TEST-}\"",
        Some(ShellExecOptions {
            env: Some(env_overrides),
            inherit_env: Some(false),
            ..ShellExecOptions::default()
        }),
        &context(),
    )
    .await;
    result.expect("exec");
    assert_eq!(output.expect("output").text, "::explicit");
}

/// `cleanup` kills active shells and their exec settles with a signal
/// exit.
#[tokio::test]
async fn cleanup_terminates_active_shell_processes() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let background = context();
    let mut execution = std::pin::pin!(Shell::exec(
        &env,
        "touch started; sleep 60",
        None,
        &background
    ));
    // The exec future is lazy; polling it alongside the probe loop starts
    // the shell the way upstream's eager promise does.
    let started = loop {
        tokio::select! {
            _ = &mut execution => panic!("the long-running shell should still be running"),
            () = tokio::time::sleep(std::time::Duration::from_millis(10)) => {
                if FileSystem::exists(&env, "started", &background).await.unwrap_or(false) {
                    break true;
                }
            }
        }
    };
    assert!(started, "the shell should have started");
    Shell::cleanup(&env, &background).await;
    let result = tokio::time::timeout(std::time::Duration::from_secs(3), execution)
        .await
        .expect("the killed exec settles")
        .expect("the killed exec still succeeds");
    assert!(result.exit_code != 0);
}

/// Stdout and stderr merge into one bounded view; the first update is a
/// full replace.
#[tokio::test]
async fn combines_stdout_and_stderr_into_one_bounded_view() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path().to_string_lossy().as_ref());
    let updates: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let collected: Arc<Mutex<Option<ShellOutputView>>> = Arc::new(Mutex::new(None));
    let updates_sink = Arc::clone(&updates);
    let reducer = Arc::clone(&collected);
    let options = ShellExecOptions {
        on_update: Some(Arc::new(move |update: &ShellOutputUpdate, _context| {
            updates_sink
                .lock()
                .expect("updates lock")
                .push(update_kind(update).to_owned());
            let previous = reducer.lock().expect("output lock").take();
            *reducer.lock().expect("output lock") =
                Some(apply_shell_output_update(previous.as_ref(), update));
        })),
        ..ShellExecOptions::default()
    };
    let result = Shell::exec(
        &env,
        "printf out; printf err >&2",
        Some(options),
        &context(),
    )
    .await
    .expect("exec");
    assert_eq!(result.exit_code, 0);
    let output = collected
        .lock()
        .expect("output lock")
        .take()
        .expect("output");
    assert!(output.text.contains("out"));
    assert!(output.text.contains("err"));
    assert_eq!(
        updates
            .lock()
            .expect("updates lock")
            .first()
            .map(String::as_str),
        Some("replace")
    );
}

/// The update kind discriminator, upstream's `update.kind`.
fn update_kind(update: &ShellOutputUpdate) -> &'static str {
    match update {
        ShellOutputUpdate::Replace { .. } => "replace",
        ShellOutputUpdate::Append { .. } => "append",
        _ => "slide",
    }
}

/// A missing working directory reports `spawn_error` before spawning.
#[tokio::test]
async fn reports_a_missing_working_directory_before_spawning() {
    let root = tempfile::tempdir().expect("temp root");
    let env = NodeExecutionEnv::new(
        root.path().join("missing").to_string_lossy().into_owned(),
        None,
        None,
    );
    let error = Shell::exec(&env, "printf ok", None, &context())
        .await
        .expect_err("missing cwd errors");
    assert_eq!(
        error.code,
        ExecutionErrorCode::SpawnError
    );
    assert!(error.message.contains("Working directory does not exist"));
}

/// Non-zero command exits stay successful execution results.
#[tokio::test]
async fn non_zero_command_exits_are_successful_results() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path().to_string_lossy().as_ref());
    let result = Shell::exec(&env, "exit 7", None, &context())
        .await
        .expect("exec");
    assert_eq!(result.exit_code, 7);
    assert_eq!(result.truncation.total_bytes, 0);
}

/// A process killed by a signal maps to the conventional 128 + signal
/// exit code.
#[tokio::test]
async fn signal_killed_processes_map_to_a_non_zero_exit_code() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path().to_string_lossy().as_ref());
    let result = Shell::exec(&env, "kill -9 $$", None, &context())
        .await
        .expect("exec");
    assert_eq!(result.exit_code, 128 + 9);
}

/// Commands exceeding the timeout report timeout errors.
#[tokio::test]
async fn commands_exceeding_the_timeout_report_timeout_errors() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path().to_string_lossy().as_ref());
    let error = Shell::exec(
        &env,
        "sleep 5",
        Some(ShellExecOptions {
            timeout: Some(0.01),
            ..ShellExecOptions::default()
        }),
        &context(),
    )
    .await
    .expect_err("timeout errors");
    assert_eq!(
        error.code,
        ExecutionErrorCode::Timeout
    );
}

/// A configured shell path that does not exist reports
/// `shell_unavailable`; a non-executable shell reports `spawn_error`.
#[tokio::test]
async fn shell_unavailable_and_spawn_errors() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let missing_shell_env =
        NodeExecutionEnv::new(root.clone(), Some(format!("{root}/missing-shell")), None);
    let error = Shell::exec(&missing_shell_env, "printf ok", None, &context())
        .await
        .expect_err("missing shell errors");
    assert_eq!(
        error.code,
        ExecutionErrorCode::ShellUnavailable
    );

    let shell_path = format!("{root}/not-executable-shell");
    let env = env_at(&root);
    FileSystem::write_file(
        &env,
        "not-executable-shell",
        "not executable".into(),
        &context(),
    )
    .await
    .expect("write shell");
    let spawn_error_env = NodeExecutionEnv::new(root, Some(shell_path), None);
    let error = Shell::exec(&spawn_error_env, "printf ok", None, &context())
        .await
        .expect_err("non-executable shell errors");
    assert_eq!(
        error.code,
        ExecutionErrorCode::SpawnError
    );
}

/// Aborted commands report the aborted error.
#[tokio::test]
async fn aborted_commands_report_the_aborted_error() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path().to_string_lossy().as_ref());
    let (context, controller) = with_cancel(&background_context());
    let execution = Shell::exec(&env, "sleep 5", None, &context);
    controller.abort("aborted");
    let error = execution.await.expect_err("aborted exec errors");
    assert_eq!(
        error.code,
        ExecutionErrorCode::Aborted
    );
}

/// No spill file is created while the bounded output stays within its
/// limits.
#[tokio::test]
async fn does_not_create_a_spill_before_bounded_output_crosses_its_limits() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path().to_string_lossy().as_ref());
    let result = Shell::exec(
        &env,
        "printf short",
        Some(ShellExecOptions {
            capture: Some(spill_capture(100, 10)),
            ..ShellExecOptions::default()
        }),
        &context(),
    )
    .await
    .expect("exec");
    assert!(result.spill_path.is_none());
}

fn spill_capture(
    max_bytes: u64,
    max_lines: u64,
) -> ShellOutputCaptureOptions {
    ShellOutputCaptureOptions {
        limits: ShellOutputLimits {
            max_bytes,
            max_lines,
            retain: Some(crate::harness::types::ShellOutputRetention::Tail),
        },
        spill: true,
    }
}

/// The spill preserves exact raw bytes while the bounded view decodes text.
#[tokio::test]
async fn spill_preserves_exact_raw_bytes() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path().to_string_lossy().as_ref());
    let result = Shell::exec(
        &env,
        "printf '\\x66\\x80\\x00\\x6f'",
        Some(ShellExecOptions {
            capture: Some(spill_capture(1, 10)),
            ..ShellExecOptions::default()
        }),
        &context(),
    )
    .await
    .expect("exec");
    let spill_path = result.spill_path.expect("spill path");
    let bytes = FileSystem::read_binary_file(&env, &spill_path, &context())
        .await
        .expect("read spill");
    assert_eq!(bytes, vec![0x66, 0x80, 0x00, 0x6f]);
}

/// A spill the environment fails to create fails the whole execution
/// rather than silently losing output, upstream's
/// `FailingSpillExecutionEnv`.
#[tokio::test]
async fn fails_rather_than_silently_losing_a_requested_spill() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = FailingSpillExecutionEnv::new(NodeExecutionEnv::new(root, None, None));
    let error = Shell::exec(
        &env,
        "printf 12345678901234567890",
        Some(ShellExecOptions {
            capture: Some(spill_capture(10, 10)),
            ..ShellExecOptions::default()
        }),
        &context(),
    )
    .await
    .expect_err("the failed spill errors");
    assert_eq!(
        error.code,
        ExecutionErrorCode::Unknown
    );
    assert!(
        error
            .message
            .contains("Failed to preserve complete shell output")
    );
}

/// The spill writer, upstream's `FailingSpillExecutionEnv`: the spill file
/// creation targets a missing directory, and every other capability
/// delegates.
struct FailingSpillExecutionEnv {
    inner: NodeExecutionEnv,
    pids: Mutex<HashSet<u32>>,
}

impl FailingSpillExecutionEnv {
    fn new(inner: NodeExecutionEnv) -> Self {
        Self {
            inner,
            pids: Mutex::new(HashSet::new()),
        }
    }
}

impl FileSystem for FailingSpillExecutionEnv {
    fn cwd(&self) -> &str {
        self.inner.cwd()
    }

    fn absolute_path<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, crate::harness::types::FileError>> {
        self.inner.absolute_path(path, context)
    }

    fn join_path<'a>(
        &'a self,
        parts: &'a [String],
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, crate::harness::types::FileError>> {
        self.inner.join_path(parts, context)
    }

    fn read_text_file<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, crate::harness::types::FileError>> {
        self.inner.read_text_file(path, context)
    }

    fn open_text_line_reader<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<
        'a,
        Result<Box<dyn TextLineReader>, crate::harness::types::FileError>,
    > {
        self.inner.open_text_line_reader(path, context)
    }

    fn read_text_lines<'a>(
        &'a self,
        path: &'a str,
        options: Option<ReadTextLinesOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Vec<String>, crate::harness::types::FileError>> {
        self.inner.read_text_lines(path, options, context)
    }

    fn read_binary_file<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Vec<u8>, crate::harness::types::FileError>> {
        self.inner.read_binary_file(path, context)
    }

    fn write_file<'a>(
        &'a self,
        path: &'a str,
        content: FileContent,
        ctx: &'a Context,
    ) -> BoxedFuture<'a, Result<(), crate::harness::types::FileError>> {
        self.inner.write_file(path, content, ctx)
    }

    fn append_file<'a>(
        &'a self,
        path: &'a str,
        content: FileContent,
        ctx: &'a Context,
    ) -> BoxedFuture<'a, Result<(), crate::harness::types::FileError>> {
        self.inner.append_file(path, content, ctx)
    }

    fn rename_file<'a>(
        &'a self,
        source_path: &'a str,
        destination_path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), crate::harness::types::FileError>> {
        self.inner
            .rename_file(source_path, destination_path, context)
    }

    fn file_info<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<crate::harness::types::FileInfo, crate::harness::types::FileError>>
    {
        self.inner.file_info(path, context)
    }

    fn list_dir<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<
        'a,
        Result<Vec<crate::harness::types::FileInfo>, crate::harness::types::FileError>,
    > {
        self.inner.list_dir(path, context)
    }

    fn canonical_path<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, crate::harness::types::FileError>> {
        self.inner.canonical_path(path, context)
    }

    fn exists<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<bool, crate::harness::types::FileError>> {
        self.inner.exists(path, context)
    }

    fn create_dir<'a>(
        &'a self,
        path: &'a str,
        options: Option<CreateDirOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), crate::harness::types::FileError>> {
        self.inner.create_dir(path, options, context)
    }

    fn remove<'a>(
        &'a self,
        path: &'a str,
        options: Option<RemoveOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), crate::harness::types::FileError>> {
        self.inner.remove(path, options, context)
    }

    fn create_temp_dir<'a>(
        &'a self,
        prefix: Option<&'a str>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, crate::harness::types::FileError>> {
        self.inner.create_temp_dir(prefix, context)
    }

    fn create_temp_file<'a>(
        &'a self,
        options: Option<TempFileOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, crate::harness::types::FileError>> {
        let spill = options
            .as_ref()
            .and_then(|options| options.prefix.as_deref())
            .is_some_and(|prefix| prefix == SPILL_FILE_PREFIX);
        if spill {
            return Box::pin(async move { Ok(format!("{}/missing/spill.log", self.inner.cwd())) });
        }
        self.inner.create_temp_file(options, context)
    }

    fn cleanup<'a>(&'a self, context: &'a Context) -> BoxedFuture<'a, ()> {
        FileSystem::cleanup(&self.inner, context)
    }
}

impl Shell for FailingSpillExecutionEnv {
    fn exec<'a>(
        &'a self,
        command: &'a str,
        options: Option<ShellExecOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<ShellExecResult, crate::harness::types::ExecutionError>> {
        Box::pin(super::exec_command(
            command.to_owned(),
            options,
            context.clone(),
            self.inner.cwd().to_owned(),
            None,
            None,
            &self.pids,
            self,
        ))
    }

    fn cleanup<'a>(&'a self, context: &'a Context) -> BoxedFuture<'a, ()> {
        Shell::cleanup(&self.inner, context)
    }
}

impl ExecutionEnv for FailingSpillExecutionEnv {}

/// A fast-exiting process's complete output still lands in the spill.
#[tokio::test]
async fn spill_preserves_complete_output_for_a_fast_exiting_process() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path().to_string_lossy().as_ref());
    let result = Shell::exec(
        &env,
        "head -c 500000 /dev/zero | tr '\\0' 'x'",
        Some(ShellExecOptions {
            capture: Some(spill_capture(10, 10)),
            ..ShellExecOptions::default()
        }),
        &context(),
    )
    .await
    .expect("exec");
    let spill_path = result.spill_path.expect("spill path");
    let text = FileSystem::read_text_file(&env, &spill_path, &context())
        .await
        .expect("read spill");
    assert_eq!(text.len(), 500_000);
}

/// Large captured output flows to a full output file through the
/// execution env, upstream's `executeShellWithCapture` integration.
#[tokio::test]
async fn captures_large_shell_output_to_a_full_output_file_through_the_execution_env() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path().to_string_lossy().as_ref());
    let result = execute_shell_with_capture(&env, "yes line | head -n 15000", None, &context())
        .await
        .expect("capture");
    assert!(result.truncated);
    let full_output_path = result.full_output_path.expect("full output path");
    let full_output = FileSystem::read_text_file(&env, &full_output_path, &context())
        .await
        .expect("read full output");
    assert!(full_output.split('\n').count() > 10_000);
    assert!(result.output.len() < full_output.len());
}

/// `cleanup` on an idle environment resolves; the win32-only stdin
/// transport and detached-grandchild tests ride the map's win32 ticket.
#[tokio::test]
async fn cleanup_on_an_idle_environment_resolves() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path().to_string_lossy().as_ref());
    Shell::cleanup(&env, &context()).await;
}

// --- Private helper surface: the tests module is a sibling of the
// environment module, so the resolution and probe helpers are reachable
// directly instead of through contrived filesystem states.

use super::{
    CommandTransport, file_url_to_path, find_bash_on_path, get_bash_shell_config,
    get_shell_config, get_shell_env, resolve_path, resolve_timeout_ms, run_command, temp_name,
};

#[test]
fn the_timeout_resolution_rejects_non_finite_negative_and_oversized_values() {
    assert_eq!(
        resolve_timeout_ms(None).expect("absent timeout"),
        None
    );
    assert_eq!(
        resolve_timeout_ms(Some(1.5)).expect("a positive timeout"),
        Some(1_500)
    );
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let error = resolve_timeout_ms(Some(invalid))
            .expect_err("an invalid timeout errors");
        assert_eq!(error.code, ExecutionErrorCode::Timeout);
    }
    let error = resolve_timeout_ms(Some(2_147_483.648))
        .expect_err("a timeout past the millisecond cap errors");
    assert_eq!(error.code, ExecutionErrorCode::Timeout);
}

#[test]
fn file_urls_decode_local_paths_and_keep_malformed_input() {
    assert_eq!(
        file_url_to_path("file:///home/a%20user/x.txt"),
        "/home/a user/x.txt"
    );
    assert_eq!(file_url_to_path("file://localhost/x.txt"), "/x.txt");
    assert_eq!(file_url_to_path("file://127.0.0.1/x.txt"), "/x.txt");
    assert_eq!(file_url_to_path("file://localhost"), "file://localhost");
    assert_eq!(file_url_to_path("file:///bad%2zx"), "/bad%2zx");
    assert_eq!(file_url_to_path("file:///short%2"), "/short%2");
    assert_eq!(file_url_to_path("plain.txt"), "plain.txt");
}

#[test]
fn resolve_path_expands_home_and_normalizes_segments() {
    assert_eq!(resolve_path("/work", "a/./b/../c"), "/work/a/c");
    assert_eq!(resolve_path("/work", "/abs/x"), "/abs/x");
    assert_eq!(resolve_path("/work", "./x"), "/work/x");
    assert_eq!(resolve_path("/work", "a//b"), "/work/a/b");
    if std::env::var("HOME").is_ok_and(|home| !home.is_empty()) {
        assert_eq!(resolve_path("/work", "~"), std::env::var("HOME").expect("home"));
        assert_eq!(
            resolve_path("/work", "~/child"),
            format!("{}/child", std::env::var("HOME").expect("home"))
        );
    }
}

#[test]
fn the_legacy_wsl_bash_maps_to_stdin_transport() {
    let config = get_bash_shell_config("C:\\Windows\\System32\\bash.exe".to_owned());
    assert!(matches!(
        config.command_transport,
        CommandTransport::Stdin
    ));
    assert_eq!(config.args, vec!["-s".to_owned()]);
    let config = get_bash_shell_config("/bin/bash".to_owned());
    assert!(matches!(
        config.command_transport,
        CommandTransport::Argv
    ));
    assert_eq!(config.args, vec!["-c".to_owned()]);
}

#[tokio::test]
async fn a_missing_custom_shell_reports_shell_unavailable() {
    let error = get_shell_config(Some("/definitely/missing/bash"))
        .await
        .expect_err("a missing custom shell errors");
    assert_eq!(error.code, ExecutionErrorCode::ShellUnavailable);
}

#[tokio::test]
async fn a_present_custom_shell_skips_discovery() {
    let config = get_shell_config(Some("/bin/bash"))
        .await
        .expect("the custom shell exists");
    assert_eq!(config.shell, "/bin/bash");
}

#[test]
fn the_shell_environment_composes_inheritance_and_overrides() {
    let mut base = BTreeMap::new();
    base.insert("BASE".to_owned(), "1".to_owned());
    let mut extra = BTreeMap::new();
    extra.insert("BASE".to_owned(), "2".to_owned());
    extra.insert("EXTRA".to_owned(), "3".to_owned());
    let composed = get_shell_env(Some(&base), Some(&extra), false);
    assert_eq!(composed["BASE"], "2");
    assert_eq!(composed["EXTRA"], "3");
    assert!(get_shell_env(Some(&base), None, false).is_empty());
    assert!(get_shell_env(None, None, true).contains_key("PATH"));
}

#[tokio::test]
async fn the_probe_runner_reports_spawn_failures_and_timeouts() {
    let (stdout, status) = run_command("/bin/echo", &["-n", "probe-ok"], 5_000).await;
    assert_eq!(stdout, "probe-ok");
    assert_eq!(status, Some(0));
    let (stdout, status) = run_command("/definitely/missing/probe", &[], 5_000).await;
    assert!(stdout.is_empty());
    assert_eq!(status, None);
    let (stdout, status) = run_command("/bin/sleep", &["30"], 20).await;
    assert!(stdout.is_empty());
    assert_eq!(status, None);
}

#[tokio::test]
async fn find_bash_on_path_resolves_on_this_machine() {
    let found = find_bash_on_path()
        .await
        .expect("bash exists on mac/linux CI machines");
    assert!(found.contains("bash"));
}

#[test]
fn temp_names_stay_unique_per_counter() {
    let first = temp_name(1);
    let second = temp_name(2);
    assert_ne!(first, second);
    assert!(first.split('-').count() >= 3);
}

// --- Abort guards and error mapping the operation suites do not reach.

#[tokio::test]
async fn pre_aborted_auxiliary_operations_report_aborted() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let context = context();
    FileSystem::write_file(&env, "file.txt", "hello".into(), &context)
        .await
        .expect("write");
    let aborted = aborted_context();
    let results = (
        FileSystem::append_file(&env, "file.txt", "x".into(), &aborted).await,
        FileSystem::file_info(&env, "file.txt", &aborted).await,
        FileSystem::create_temp_dir(&env, None, &aborted).await,
        FileSystem::create_temp_file(&env, None, &aborted).await,
        FileSystem::create_dir(&env, "dir", None, &aborted).await,
        FileSystem::remove(&env, "file.txt", None, &aborted).await,
    );
    for error in [
        results.0.expect_err("append aborted"),
        results.1.expect_err("file info aborted"),
        results.2.expect_err("temp dir aborted"),
        results.3.expect_err("temp file aborted"),
        results.4.expect_err("create dir aborted"),
        results.5.expect_err("remove aborted"),
    ] {
        assert_eq!(error.code, FileErrorCode::Aborted);
    }
}

#[tokio::test]
async fn file_errors_map_the_io_error_kinds() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let context = context();
    FileSystem::write_file(&env, "file.txt", "hello".into(), &context)
        .await
        .expect("write");

    // Reading through a file path names a non-directory parent.
    let error = FileSystem::read_text_file(&env, "file.txt/child", &context)
        .await
        .expect_err("a file used as a directory errors");
    assert_eq!(error.code, FileErrorCode::NotDirectory);

    // Reading a directory is an is-a-directory failure.
    let error = FileSystem::read_text_file(&env, ".", &context)
        .await
        .expect_err("a directory read errors");
    assert_eq!(error.code, FileErrorCode::IsDirectory);

    // A NUL byte in the path is an invalid input.
    let error = FileSystem::write_file(&env, "bad\0path", "x".into(), &context)
        .await
        .expect_err("a NUL path errors");
    assert_eq!(error.code, FileErrorCode::Invalid);

    // Removing a missing path is a not-found failure.
    let error = FileSystem::remove(&env, "missing.txt", None, &context)
        .await
        .expect_err("a missing removal errors");
    assert_eq!(error.code, FileErrorCode::NotFound);
}

#[tokio::test]
async fn file_info_rejects_unsupported_types() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let socket = std::path::Path::new(&root).join("sock");
    std::os::unix::net::UnixListener::bind(&socket).expect("socket");
    let error = FileSystem::file_info(&env, "sock", &context())
        .await
        .expect_err("a socket is an unsupported file type");
    assert_eq!(error.code, FileErrorCode::Invalid);
    assert_eq!(error.message, "Unsupported file type");
}

#[tokio::test]
async fn a_closed_text_line_reader_reports_invalid() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let context = context();
    FileSystem::write_file(&env, "file.txt", "one\ntwo".into(), &context)
        .await
        .expect("write");
    let mut reader = FileSystem::open_text_line_reader(&env, "file.txt", &context)
        .await
        .expect("open");
    reader.close(&context).await;
    let error = reader.read_line(&context).await.expect_err("closed");
    assert_eq!(error.code, FileErrorCode::Invalid);
    assert_eq!(error.message, "Text line reader is closed");
}

#[tokio::test]
async fn an_abort_during_a_streaming_read_reports_aborted() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let context = context();
    let body = "x".repeat(8 * 1024 * 1024);
    FileSystem::write_file(&env, "big.txt", body.into(), &context)
        .await
        .expect("write");
    let (abort_context, controller) = with_cancel(&background_context());
    let mut reader = FileSystem::open_text_line_reader(&env, "big.txt", &context)
        .await
        .expect("open");
    let (outcome, ()) = tokio::join!(
        TextLineReader::read_line(reader.as_mut(), &abort_context),
        async {
            tokio::time::sleep(std::time::Duration::from_micros(200)).await;
            controller.abort("aborted");
        }
    );
    let error = outcome.expect_err("the mid-read abort surfaces");
    assert_eq!(error.code, FileErrorCode::Aborted);
}

#[tokio::test]
async fn the_environment_renders_its_debug_view() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path().to_string_lossy().as_ref());
    let debug = format!("{env:?}");
    assert!(debug.contains("NodeExecutionEnv"), "{debug}");
    assert!(debug.contains(&root.path().to_string_lossy().to_string()));
}

#[tokio::test]
async fn an_exec_with_a_zero_capture_limit_fails_before_spawning() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path().to_string_lossy().as_ref());
    let options = ShellExecOptions {
        capture: Some(ShellOutputCaptureOptions {
            limits: ShellOutputLimits {
                max_bytes: 0,
                max_lines: 0,
                retain: None,
            },
            spill: false,
        }),
        ..ShellExecOptions::default()
    };
    let error = Shell::exec(&env, "echo hi", Some(options), &context())
        .await
        .expect_err("a zero capture limit refuses to start");
    assert_eq!(error.code, ExecutionErrorCode::Unknown);
}

#[tokio::test]
async fn exec_completes_before_its_timeout() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path().to_string_lossy().as_ref());
    let options = ShellExecOptions {
        timeout: Some(30.0),
        ..ShellExecOptions::default()
    };
    let (result, _) = collect_shell_output(&env, "echo ok", Some(options), &context()).await;
    let result = result.expect("the run settles inside its timeout");
    assert_eq!(result.exit_code, 0);
}

// --- The remaining error surfaces the suites above do not reach.

#[test]
fn remote_authority_file_urls_keep_their_original_text() {
    assert_eq!(
        file_url_to_path("file://evil.example/x.txt"),
        "file://evil.example/x.txt"
    );
}

#[tokio::test]
async fn a_directory_read_during_line_streaming_closes_the_reader() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path().to_string_lossy().as_ref());
    let context = context();
    let error = FileSystem::read_text_lines(&env, ".", None, &context)
        .await
        .expect_err("a directory line stream errors");
    assert_eq!(error.code, FileErrorCode::IsDirectory);
}

#[tokio::test]
async fn writes_through_a_file_parent_report_the_io_error() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let context = context();
    FileSystem::write_file(&env, "file.txt", "hello".into(), &context)
        .await
        .expect("write");
    let error = FileSystem::write_file(&env, "file.txt/child", "x".into(), &context)
        .await
        .expect_err("a file parent errors");
    assert!(error.path.as_deref().is_some_and(|path| path.contains("file.txt/child")), "{error:?}");
    let error = FileSystem::append_file(&env, "file.txt/child", "x".into(), &context)
        .await
        .expect_err("a file parent errors on append too");
    assert!(error.path.as_deref().is_some_and(|path| path.contains("file.txt/child")), "{error:?}");
}

#[tokio::test]
async fn a_temp_file_with_an_unwritable_prefix_reports_the_io_error() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path().to_string_lossy().as_ref());
    let context = context();
    let error = FileSystem::create_temp_file(
        &env,
        Some(TempFileOptions {
            prefix: Some("/definitely-missing-dir/".to_owned()),
            suffix: None,
        }),
        &context,
    )
    .await
    .expect_err("an unwritable prefix errors");
    assert_eq!(error.code, FileErrorCode::NotFound);
}

#[tokio::test]
async fn canonical_and_create_dir_error_surfaces() {
    let root = tempfile::tempdir().expect("temp root");
    let root = root.path().to_string_lossy().into_owned();
    let env = env_at(&root);
    let context = context();
    FileSystem::write_file(&env, "file.txt", "hello".into(), &context)
        .await
        .expect("write");
    let error = FileSystem::canonical_path(&env, "missing.txt", &context)
        .await
        .expect_err("a missing canonical path errors");
    assert_eq!(error.code, FileErrorCode::NotFound);
    let error = FileSystem::create_dir(&env, "file.txt", Some(CreateDirOptions { recursive: Some(false) }), &context)
        .await
        .expect_err("a non-recursive create over a file errors");
    assert!(error.code != FileErrorCode::Aborted, "{error:?}");
}
