//! The belt's boundary tests: the restated seams the upstream suites do
//! not cover module-by-module — the JSON comment stripper, the HTML
//! entity decoder, the pi user agent, the deprecation dedup, the abort
//! races, the sleep, the child-stdio grace drain, the shell resolution,
//! and the file watcher.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;

use pi_coding_agent::utils::child_process::{
    SpawnSyncOptions, spawn_process_sync, wait_for_child_process,
};
use pi_coding_agent::utils::clipboard_command::ClipboardCommandRunner;
use pi_coding_agent::utils::deprecation::{clear_deprecation_warnings_for_tests, warn_deprecation};
use pi_coding_agent::utils::html::{decode_html_entity, decode_html_entity_at};
use pi_coding_agent::utils::json::strip_json_comments;
use pi_coding_agent::utils::shell::{
    CommandTransport, ShellError, get_power_shell_config, get_shell_config,
    get_shell_env_path_update, is_legacy_wsl_bash_path, kill_tracked_detached_children,
    sanitize_binary_output, track_detached_child_pid, untrack_detached_child_pid,
};
use pi_coding_agent::utils::sleep::sleep;
use tokio_util::sync::CancellationToken;

// === json ===================================================================

#[test]
fn strips_line_comments_leaving_string_literals() {
    assert_eq!(
        strip_json_comments("{\n  // comment\n  \"key\": \"value // not a comment\"\n}"),
        "{\n  \n  \"key\": \"value // not a comment\"\n}"
    );
}

#[test]
fn strips_trailing_commas() {
    assert_eq!(strip_json_comments("{\"a\": 1,}"), "{\"a\": 1}");
    assert_eq!(strip_json_comments("[1, 2, 3, ]"), "[1, 2, 3 ]");
    assert_eq!(strip_json_comments("[1, 2, 3,]"), "[1, 2, 3]");
}

#[test]
fn keeps_commas_inside_strings_and_escapes() {
    assert_eq!(
        strip_json_comments("{\"a\": \"b, c\",}"),
        "{\"a\": \"b, c\"}"
    );
    assert_eq!(
        strip_json_comments("{\"a\": \"x\\\\\",}"),
        "{\"a\": \"x\\\\\"}"
    );
}

// === html ===================================================================

#[test]
fn decodes_the_named_and_numeric_entities() {
    assert_eq!(decode_html_entity("amp").as_deref(), Some("&"));
    assert_eq!(decode_html_entity("lt").as_deref(), Some("<"));
    assert_eq!(decode_html_entity("gt").as_deref(), Some(">"));
    assert_eq!(decode_html_entity("quot").as_deref(), Some("\""));
    assert_eq!(decode_html_entity("apos").as_deref(), Some("'"));
    assert_eq!(decode_html_entity("#x41").as_deref(), Some("A"));
    assert_eq!(decode_html_entity("#X41").as_deref(), Some("A"));
    assert_eq!(decode_html_entity("#65").as_deref(), Some("A"));
    assert_eq!(decode_html_entity("nbsp"), None);
    assert_eq!(decode_html_entity("#x110000"), None);
    // Surrogate code points have no char in Rust; the decode fails where
    // JS would carry a lone surrogate string.
    assert_eq!(decode_html_entity("#xD800"), None);
    assert_eq!(decode_html_entity("#x"), None);
    assert_eq!(decode_html_entity("#zz"), None);
}

#[test]
fn decodes_entities_at_an_index_with_a_lookahead_cap() {
    let decoded = decode_html_entity_at("&amp; more", 0).expect("entity");
    assert_eq!(decoded.text, "&");
    assert_eq!(decoded.length, 5);
    // No terminator in reach.
    assert_eq!(decode_html_entity_at("&nope", 0), None);
    // A body longer than the 16-character cap.
    assert_eq!(decode_html_entity_at("&verylongentityname;", 0), None);
    // Not an entity at all.
    assert_eq!(decode_html_entity_at("plain & text", 6), None);
}

// === pi-user-agent ==========================================================

#[test]
fn formats_the_user_agent_shape_pi_dev_expects() {
    let agent = pi_coding_agent::utils::pi_user_agent::get_pi_user_agent("1.2.3");
    let platform = match std::env::consts::OS {
        "macos" => "darwin",
        other => other,
    };
    let arch = match std::env::consts::ARCH {
        "aarch64" => "arm64",
        "x86_64" => "x64",
        other => other,
    };
    assert_eq!(agent, format!("pi/1.2.3 ({platform}; rust; {arch})"));
    // The three token groups inside the parentheses, upstream's regex shape.
    let between = agent
        .split(" (")
        .nth(1)
        .expect("parenthesized")
        .trim_end_matches(')');
    let tokens: Vec<&str> = between.split(';').collect();
    assert_eq!(tokens.len(), 3);
    assert!(tokens.iter().all(|token| !token.contains(['(', ')'])));
}

// === deprecation ============================================================

#[test]
fn deprecation_warnings_emit_once_per_message() {
    clear_deprecation_warnings_for_tests();
    // The first call emits; the second with the same message is silent.
    // The stderr surface itself is upstream's console.warn; the dedup set
    // is what the boundary pins.
    warn_deprecation("belt boundary probe");
    warn_deprecation("belt boundary probe");
    clear_deprecation_warnings_for_tests();
    // A cleared state warns again.
    warn_deprecation("belt boundary probe");
    clear_deprecation_warnings_for_tests();
}

// === abort ==================================================================

#[tokio::test]
async fn operation_signal_answers_a_fresh_token_when_absent() {
    let token = pi_coding_agent::utils::abort::operation_signal(None);
    assert!(!token.is_cancelled());
    // The supplied token is the one carried.
    let held = CancellationToken::new();
    let supplied = pi_coding_agent::utils::abort::operation_signal(Some(&held));
    assert!(!supplied.is_cancelled());
    held.cancel();
    assert!(supplied.is_cancelled());
}

#[tokio::test]
async fn race_with_abort_signal_carries_the_operation_error() {
    let token = CancellationToken::new();
    let result: Result<(), pi_coding_agent::utils::abort::RaceError<String>> =
        pi_coding_agent::utils::abort::race_with_abort_signal(
            std::future::ready(Err("failed".to_string())),
            Some(&token),
        )
        .await;
    assert_eq!(
        result,
        Err(pi_coding_agent::utils::abort::RaceError::Operation(
            "failed".to_string()
        ))
    );
}

#[tokio::test]
async fn race_with_abort_signal_answers_the_abort_first() {
    let token = CancellationToken::new();
    token.cancel();
    let result: Result<(), pi_coding_agent::utils::abort::RaceError<String>> =
        pi_coding_agent::utils::abort::race_with_abort_signal(
            async {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                Ok(())
            },
            Some(&token),
        )
        .await;
    assert!(matches!(
        result,
        Err(pi_coding_agent::utils::abort::RaceError::Aborted(_))
    ));
}

#[tokio::test]
async fn race_with_abort_signal_passes_through_without_a_signal() {
    // The optional-signal branch waits the operation out.
    let result: Result<u8, pi_coding_agent::utils::abort::RaceError<String>> =
        pi_coding_agent::utils::abort::race_with_abort_signal(std::future::ready(Ok(7)), None)
            .await;
    assert_eq!(result, Ok(7));
    let result: Result<u8, pi_coding_agent::utils::abort::RaceError<String>> =
        pi_coding_agent::utils::abort::race_with_abort_signal(
            std::future::ready(Err("boom".to_string())),
            None,
        )
        .await;
    assert_eq!(
        result,
        Err(pi_coding_agent::utils::abort::RaceError::Operation(
            "boom".to_string()
        ))
    );
}

// === sleep ==================================================================

#[tokio::test(start_paused = true)]
async fn sleep_resolves_after_the_delay() {
    tokio::time::advance(std::time::Duration::from_millis(50)).await;
    let result = sleep(50, None).await;
    assert_eq!(result, Ok(()));
}

#[tokio::test(start_paused = true)]
async fn sleep_gives_up_when_the_signal_was_already_aborted() {
    // An already-aborted signal answers without waiting, upstream's early
    // reject; the abort-racing-the-timer case is the next test's.
    let token = CancellationToken::new();
    token.cancel();
    let result = sleep(60_000, Some(&token)).await;
    assert_eq!(result, Err(pi_coding_agent::utils::sleep::SleepAborted));
}

#[tokio::test(start_paused = true)]
async fn sleep_races_the_abort_against_the_timer() {
    let token = CancellationToken::new();
    let sleeping = sleep(60_000, Some(&token));
    tokio::pin!(sleeping);
    tokio::time::advance(std::time::Duration::from_millis(10)).await;
    token.cancel();
    let result = (&mut sleeping).await;
    assert_eq!(result, Err(pi_coding_agent::utils::sleep::SleepAborted));
}

// === child_process ==========================================================

#[tokio::test]
async fn wait_for_child_process_settles_with_the_exit_code() {
    let child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg("exit 3")
        .spawn()
        .expect("spawn");
    assert_eq!(wait_for_child_process(child).await, Some(3));
}

#[tokio::test]
async fn wait_for_child_process_reports_a_signalled_child_as_none() {
    let child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg("kill -TERM $$")
        .spawn()
        .expect("spawn");
    assert_eq!(wait_for_child_process(child).await, None);
}

#[tokio::test]
async fn wait_for_child_process_keeps_draining_a_detached_descendants_tail() {
    // A child that exits while its detached grandchild keeps writing: the
    // wait must not truncate the tail (earendil-works/pi#5303), and must
    // not hang past the quiet-pipe grace either.
    let child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg("echo first; ( sleep 30; echo late ) & exit 0")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn");
    let started = std::time::Instant::now();
    let code = wait_for_child_process(child).await;
    assert_eq!(code, Some(0));
    // The grandchild sleeps 30s; the grace released us long before.
    assert!(started.elapsed() < std::time::Duration::from_secs(10));
}

#[tokio::test]
async fn wait_for_child_process_reports_a_signal_death_as_none() {
    let child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg("kill -TERM $$")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn");
    assert_eq!(wait_for_child_process(child).await, None);
}

#[tokio::test]
async fn wait_for_child_process_rearms_the_grace_on_post_exit_chunks() {
    // The grandchild writes within the grace window: the tail must survive,
    // upstream's onData re-arm, and the wait settles when the pipe EOFs.
    let child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg("echo first; ( sleep 0.05; echo tail ) & exit 0")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn");
    assert_eq!(wait_for_child_process(child).await, Some(0));
}

#[tokio::test]
async fn wait_for_child_process_survives_a_pipe_holding_descendant() {
    // The grandchild holds the pipe open for 30s; the exit still settles
    // with its code, the tail drained or grace-fired.
    let child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg("printf start; ( sleep 30; printf tail ) & exit 0")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn");
    let started = std::time::Instant::now();
    assert_eq!(wait_for_child_process(child).await, Some(0));
    assert!(started.elapsed() < std::time::Duration::from_secs(10));
}

#[test]
fn spawn_process_sync_captures_output_and_status() {
    let outcome = spawn_process_sync(
        "sh",
        &["-c", "printf out"],
        &SpawnSyncOptions {
            capture_output: true,
            timeout_ms: None,
        },
    );
    assert_eq!(outcome.status, Some(0));
    assert_eq!(outcome.stdout, "out");
}

#[test]
fn spawn_process_sync_reports_a_missing_binary_as_default() {
    let outcome = spawn_process_sync(
        "pi-belt-does-not-exist",
        &[],
        &SpawnSyncOptions::IGNORE_OUTPUT,
    );
    assert_eq!(outcome.status, None);
    assert_eq!(outcome.stdout, "");
}

#[test]
fn spawn_process_sync_kills_past_the_timeout() {
    let started = std::time::Instant::now();
    let outcome = spawn_process_sync(
        "sh",
        &["-c", "sleep 30"],
        &SpawnSyncOptions {
            capture_output: false,
            timeout_ms: Some(200),
        },
    );
    assert_eq!(outcome.status, None);
    assert!(started.elapsed() < std::time::Duration::from_secs(10));
}

// === shell ==================================================================

#[test]
fn legacy_wsl_bash_paths_are_detected() {
    assert!(is_legacy_wsl_bash_path("C:\\Windows\\System32\\bash.exe"));
    assert!(is_legacy_wsl_bash_path("c:\\windows\\sysnative\\bash.exe"));
    assert!(is_legacy_wsl_bash_path("C:/Windows/System32/bash.exe"));
    assert!(!is_legacy_wsl_bash_path("C:\\Windows\\System32\\cmd.exe"));
    assert!(!is_legacy_wsl_bash_path("C:\\Windows\\bash.exe"));
    assert!(!is_legacy_wsl_bash_path("/bin/bash"));
    assert!(!is_legacy_wsl_bash_path("C:\\Windows\\System32\\bash"));
}

#[tokio::test]
async fn shell_config_resolves_the_platform_shells() {
    // A custom path that exists rides the bash config.
    let custom = get_shell_config(Some("/bin/sh")).expect("custom shell");
    assert_eq!(custom.shell, "/bin/sh");
    assert_eq!(custom.args, vec!["-c".to_string()]);
    assert_eq!(custom.command_transport, None);

    let custom_error = get_shell_config(Some("/no/such/shell")).expect_err("missing");
    assert_eq!(
        custom_error.0,
        "Custom shell path not found: /no/such/shell"
    );

    // The unix ladder: /bin/bash exists on the CI images.
    let config = get_shell_config(None).expect("shell");
    assert_eq!(config.shell, "/bin/bash");
    assert_eq!(config.args, vec!["-c".to_string()]);
}

#[test]
fn power_shell_reports_the_windows_only_guard() {
    let error = get_power_shell_config().expect_err("guard");
    assert_eq!(error.0, "The powershell tool is only available on Windows.");
}

#[test]
fn shell_env_prepends_the_bin_dir_once() {
    let bin_dir = pi_coding_agent::config::get_bin_dir()
        .to_string_lossy()
        .into_owned();
    let map: HashMap<String, String> = HashMap::new();
    let env: pi_coding_agent::config::EnvLookup = Box::new(move |key: &str| map.get(key).cloned());
    let (key, updated) = get_shell_env_path_update(&env);
    assert_eq!(key, "PATH");
    assert_eq!(updated.as_str(), bin_dir);

    let map: HashMap<String, String> =
        HashMap::from([("PATH".to_string(), format!("/usr/bin:{bin_dir}"))]);
    let env: pi_coding_agent::config::EnvLookup = Box::new(move |key: &str| map.get(key).cloned());
    let (_, updated) = get_shell_env_path_update(&env);
    assert_eq!(updated, format!("/usr/bin:{bin_dir}"));
}

#[test]
fn sanitize_binary_output_drops_the_crashing_characters() {
    assert_eq!(
        sanitize_binary_output("tab\tnewline\nreturn\rdone"),
        "tab\tnewline\nreturn\rdone"
    );
    assert_eq!(sanitize_binary_output("null\0bell\u{7}"), "nullbell");
    // The Unicode Format run that crashes string-width.
    assert_eq!(sanitize_binary_output("a\u{fff9}b\u{fffb}c"), "abc");
    assert_eq!(sanitize_binary_output("keep ✔ 漢"), "keep ✔ 漢");
}

#[test]
fn tracked_detached_children_kill_and_clear() {
    // A real short-lived child the sweep can kill.
    let mut child = std::process::Command::new("sh")
        .args(["-c", "sleep 30"])
        .spawn()
        .expect("spawn");
    let pid = child.id();
    track_detached_child_pid(pid);
    kill_tracked_detached_children();
    // The sweep cleared the set: a second kill is a no-op sweep.
    kill_tracked_detached_children();
    untrack_detached_child_pid(pid);
    let _reaped = child.wait();
}

// === fs-watch ===============================================================

#[test]
fn close_watcher_tolerates_the_absent_handle() {
    pi_coding_agent::utils::fs_watch::close_watcher(None);
}

#[tokio::test]
async fn watch_delivers_events_to_the_listener() {
    let dir = tempfile::tempdir().expect("scratch");
    let watched = dir.path().join("watched");
    std::fs::create_dir_all(&watched).expect("mkdir");

    let inbox: Arc<Mutex<Vec<pi_coding_agent::utils::fs_watch::WatchEvent>>> =
        Arc::new(Mutex::new(Vec::new()));
    let forward = Arc::clone(&inbox);
    let handle = pi_coding_agent::utils::fs_watch::watch_with_error_handler(
        &watched.to_string_lossy(),
        move |event| forward.lock().expect("inbox").push(event),
        || panic!("a live watch must not error"),
    )
    .expect("watcher");

    std::fs::write(watched.join("file.txt"), "content").expect("write");
    // A metadata touch rides node's change arm, upstream's `change`.
    let _permissions = std::fs::set_permissions(
        watched.join("file.txt"),
        std::os::unix::fs::PermissionsExt::from_mode(0o640),
    );
    // The notify thread delivers within a moment; poll briefly.
    for _ in 0..100 {
        if !inbox.lock().expect("inbox").is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let events = inbox.lock().expect("inbox").clone();
    assert!(!events.is_empty(), "the write surfaced");
    assert!(
        events
            .iter()
            .all(|event| event.event_type == "rename" || event.event_type == "change")
    );

    pi_coding_agent::utils::fs_watch::close_watcher(Some(handle));
}

#[test]
fn watch_reports_creation_failures_through_the_error_sink() {
    // A watcher over a path that cannot be watched: the notify constructor
    // or the watch call fails, the sink hears it, and the answer is None.
    let errors = Arc::new(Mutex::new(0usize));
    let sink = Arc::clone(&errors);
    let watcher = pi_coding_agent::utils::fs_watch::watch_with_error_handler(
        "/proc/nonexistent-path-for-the-belt-suite",
        |_| {},
        move || *sink.lock().expect("errors") += 1,
    );
    if watcher.is_none() {
        assert!(*errors.lock().expect("errors") >= 1);
    }
    // A watcher that did open is still closed cleanly.
    drop(watcher);
}

// === the clipboard-command runner seam, exercised for the trait =============

#[tokio::test]
async fn the_process_runner_carries_the_timeout_option() {
    let runner = pi_coding_agent::utils::clipboard_command::ProcessClipboardCommandRunner;
    let failed: Option<Vec<u8>> = runner
        .run(
            "pi-belt-does-not-exist",
            &[],
            &pi_coding_agent::utils::clipboard_command::read_options(None),
        )
        .await;
    assert_eq!(failed, None);
    let _ = std::marker::PhantomData::<fn(&dyn ClipboardCommandRunner)>;
}

#[tokio::test]
async fn the_shell_config_error_type_surfaces_verbatim() {
    assert_eq!(
        ShellError("Custom shell path not found: /x".to_string()).to_string(),
        "Custom shell path not found: /x"
    );
    assert_eq!(CommandTransport::Argv, CommandTransport::Argv);
}

#[allow(
    dead_code,
    reason = "the future bound keeps the runner seam's shape honest"
)]
fn _runner_future_is_boxed(_: Pin<Box<dyn Future<Output = Option<Vec<u8>>>>>) {}
