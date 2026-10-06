//! The clipboard-command suite, upstream's `test/clipboard-command.test.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream drives real `node -e` children; the port drives real `sh -c`
//! children the same way, so the binary-output, timeout, and buffer-cap
//! contracts hold against the process machinery itself.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use pi_coding_agent::utils::clipboard_command::{ClipboardCommandOptions, run_clipboard_command};

#[tokio::test]
async fn preserves_binary_output_and_distinguishes_empty_success_from_failure() {
    let binary = run_clipboard_command(
        "sh",
        &["-c".to_string(), "printf '\\000\\377\\012'".to_string()],
        None,
    )
    .await;
    assert_eq!(binary, Some(vec![0, 255, 10]));

    let empty = run_clipboard_command("sh", &["-c".to_string(), String::new()], None).await;
    assert_eq!(empty, Some(Vec::new()));

    let failed = run_clipboard_command("sh", &["-c".to_string(), "exit 1".to_string()], None).await;
    assert_eq!(failed, None);

    let missing = run_clipboard_command("pi-clipboard-command-does-not-exist", &[], None).await;
    assert_eq!(missing, None);
}

#[tokio::test]
async fn sends_unicode_input_to_clipboard_writers() {
    let temp = tempfile::tempdir().expect("scratch");
    let received = temp.path().join("received");
    let script = format!("cat > {}", received.to_string_lossy());
    let written = run_clipboard_command(
        "sh",
        &["-c".to_string(), script],
        Some(&ClipboardCommandOptions {
            input: Some("café 日本語".to_string()),
            ..ClipboardCommandOptions::default()
        }),
    )
    .await;
    assert_eq!(written, Some(Vec::new()));
    let received = std::fs::read(&received).expect("writer received the input");
    assert_eq!(String::from_utf8_lossy(&received), "café 日本語");
}

#[tokio::test]
async fn times_out_without_blocking_the_event_loop() {
    // The paused clock drives the timeout: the suite must not wait out the
    // child's lifetime, and the run must return control promptly.
    let started = std::time::Instant::now();
    let timed_out = run_clipboard_command(
        "sh",
        &["-c".to_string(), "sleep 30".to_string()],
        Some(&ClipboardCommandOptions {
            timeout_ms: Some(200),
            ..ClipboardCommandOptions::default()
        }),
    )
    .await;
    assert_eq!(timed_out, None);
    assert!(
        started.elapsed() < std::time::Duration::from_millis(10_000),
        "the timeout released the caller, not the child's lifetime"
    );
}

#[tokio::test]
async fn rejects_output_above_the_buffer_limit() {
    let over_limit = run_clipboard_command(
        "sh",
        &["-c".to_string(), "yes | head -c 1024".to_string()],
        Some(&ClipboardCommandOptions {
            max_buffer_bytes: Some(16),
            ..ClipboardCommandOptions::default()
        }),
    )
    .await;
    assert_eq!(over_limit, None);
}
