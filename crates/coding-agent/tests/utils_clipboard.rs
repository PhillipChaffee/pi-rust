//! The clipboard suite, upstream's `test/clipboard.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The `vi.mock` seams restate to the crate's overrides: the native
//! clipboard, the OSC 52 stdout sink, the command runner, and the
//! injected environment/platform. The "waits for the native write before
//! emitting remote OSC 52" case restates to its end state: the native
//! write is synchronous in Rust, so the ordering upstream drives through a
//! pending promise holds by construction (recorded with the ticket).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use pi_coding_agent::config::EnvLookup;
use pi_coding_agent::utils::clipboard::{
    ClipboardError, NativeClipboard, Platform, copy_to_clipboard_with, read_clipboard_text_with,
    set_native_clipboard_disabled, set_native_clipboard_override, set_osc52_sink,
};
use pi_coding_agent::utils::clipboard_command::ClipboardCommandRunner;

struct MockClipboard {
    get_text_result: Mutex<Result<String, ClipboardError>>,
    get_image_result: Mutex<Result<Vec<u8>, ClipboardError>>,
    set_text_result: Mutex<bool>,
    set_text_calls: Mutex<Vec<String>>,
}

impl MockClipboard {
    const fn new() -> Self {
        Self {
            get_text_result: Mutex::new(Ok(String::new())),
            get_image_result: Mutex::new(Ok(Vec::new())),
            set_text_result: Mutex::new(true),
            set_text_calls: Mutex::new(Vec::new()),
        }
    }

    fn with_set_text(writable: bool) -> Self {
        Self {
            set_text_result: Mutex::new(writable),
            ..Self::new()
        }
    }
}

impl NativeClipboard for MockClipboard {
    fn get_text(&self) -> Result<String, ClipboardError> {
        self.get_text_result.lock().expect("mock").clone()
    }

    fn get_image(&self) -> Result<Vec<u8>, ClipboardError> {
        self.get_image_result.lock().expect("mock").clone()
    }

    fn set_text(&self, text: &str) -> bool {
        self.set_text_calls
            .lock()
            .expect("mock")
            .push(text.to_string());
        *self.set_text_result.lock().expect("mock")
    }
}

#[derive(Default)]
struct MockRunner {
    /// The scripted outcome per command name; an absent command falls to
    /// the default (upstream's `mockResolvedValue`).
    results: Mutex<HashMap<String, Option<Vec<u8>>>>,
    default_result: Mutex<Option<Vec<u8>>>,
    calls: Mutex<Vec<String>>,
}

impl MockRunner {
    fn resolve_all_to_empty() -> Self {
        let runner = Self::default();
        *runner.default_result.lock().expect("mock") = Some(Vec::new());
        runner
    }
}

impl ClipboardCommandRunner for MockRunner {
    fn run<'a>(
        &'a self,
        command: &'a str,
        _args: &'a [String],
        _options: &'a pi_coding_agent::utils::clipboard_command::ClipboardCommandOptions,
    ) -> Pin<Box<dyn Future<Output = Option<Vec<u8>>> + 'a>> {
        self.calls.lock().expect("mock").push(command.to_string());
        let result = self
            .results
            .lock()
            .expect("mock")
            .get(command)
            .cloned()
            .unwrap_or_else(|| self.default_result.lock().expect("mock").clone());
        Box::pin(std::future::ready(result))
    }
}

fn env_lookup(pairs: &[(&str, &str)]) -> EnvLookup {
    let map: HashMap<String, String> = pairs
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect();
    Box::new(move |key: &str| map.get(key).cloned())
}

fn empty_env() -> EnvLookup {
    Box::new(|_key| None)
}

fn install_osc_sink() -> Arc<Mutex<Vec<String>>> {
    let writes = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&writes);
    set_osc52_sink(Some(Arc::new(move |bytes: &[u8]| {
        let chunk = String::from_utf8_lossy(bytes).into_owned();
        if chunk.starts_with("\x1b]52;c;") {
            sink.lock().expect("sink").push(chunk);
        }
    })));
    writes
}

// === readClipboardText ======================================================

#[tokio::test]
async fn awaits_native_clipboard_text_and_catches_rejected_reads() {
    let clipboard = Arc::new(MockClipboard::new());
    *clipboard.get_text_result.lock().expect("mock") = Ok("clipboard text".to_string());
    set_native_clipboard_override(Some(clipboard.clone()));
    assert_eq!(
        read_clipboard_text_with(&empty_env(), Platform::Darwin, &MockRunner::default()).await,
        Some("clipboard text".to_string())
    );

    *clipboard.get_text_result.lock().expect("mock") =
        Err(ClipboardError("clipboard unavailable".to_string()));
    assert_eq!(
        read_clipboard_text_with(&empty_env(), Platform::Darwin, &MockRunner::default()).await,
        None
    );

    set_native_clipboard_override(None);
}

#[tokio::test]
async fn wayland_results_stop_fallback_before_the_native_clipboard() {
    // Regression test for #7248: empty Wayland content must not fall through to stale X11.
    for text in ["clipboard text", ""] {
        let runner = MockRunner::default();
        runner
            .results
            .lock()
            .expect("mock")
            .insert("wl-paste".to_string(), Some(text.as_bytes().to_vec()));
        let env = env_lookup(&[("WAYLAND_DISPLAY", "wayland-0"), ("DISPLAY", ":0")]);
        assert_eq!(
            read_clipboard_text_with(&env, Platform::Linux, &runner).await,
            if text.is_empty() {
                None
            } else {
                Some(text.to_string())
            }
        );
        assert_eq!(
            *runner.calls.lock().expect("mock"),
            vec!["wl-paste".to_string()]
        );
    }
    set_native_clipboard_override(None);
}

#[tokio::test]
async fn xclip_results_stop_fallback() {
    let runner = MockRunner::default();
    runner
        .results
        .lock()
        .expect("mock")
        .insert("xclip".to_string(), Some(b"clipboard text".to_vec()));
    let env = env_lookup(&[("DISPLAY", ":0")]);
    assert_eq!(
        read_clipboard_text_with(&env, Platform::Linux, &runner).await,
        Some("clipboard text".to_string())
    );
    assert_eq!(
        *runner.calls.lock().expect("mock"),
        vec!["xclip".to_string()]
    );
}

#[tokio::test]
async fn xsel_tries_after_xclip() {
    let runner = MockRunner::default();
    runner
        .results
        .lock()
        .expect("mock")
        .insert("xsel".to_string(), Some(Vec::new()));
    let env = env_lookup(&[("DISPLAY", ":0")]);
    assert_eq!(
        read_clipboard_text_with(&env, Platform::Linux, &runner).await,
        None
    );
    assert_eq!(
        *runner.calls.lock().expect("mock"),
        vec!["xclip".to_string(), "xsel".to_string()]
    );
}

#[tokio::test]
async fn termux_results_stop_fallback() {
    let runner = MockRunner::default();
    runner.results.lock().expect("mock").insert(
        "termux-clipboard-get".to_string(),
        Some(b"termux text".to_vec()),
    );
    let env = env_lookup(&[("TERMUX_VERSION", "0.119")]);
    assert_eq!(
        read_clipboard_text_with(&env, Platform::Linux, &runner).await,
        Some("termux text".to_string())
    );
    assert_eq!(
        *runner.calls.lock().expect("mock"),
        vec!["termux-clipboard-get".to_string()]
    );
}

#[tokio::test]
async fn uses_native_after_command_failures() {
    for text in ["native text", "", "sentinel"] {
        let clipboard = Arc::new(MockClipboard::new());
        *clipboard.get_text_result.lock().expect("mock") = if text == "sentinel" {
            // The empty-read null case, upstream's `|| null`.
            Ok(String::new())
        } else {
            Ok(text.to_string())
        };
        set_native_clipboard_override(Some(clipboard.clone()));
        let runner = MockRunner::default();
        let env = env_lookup(&[("WAYLAND_DISPLAY", "wayland-0"), ("DISPLAY", ":0")]);
        let expected = match text {
            "native text" => Some("native text".to_string()),
            // The empty read answers null.
            _ => None,
        };
        assert_eq!(
            read_clipboard_text_with(&env, Platform::Linux, &runner).await,
            expected
        );
        assert_eq!(
            *runner.calls.lock().expect("mock"),
            vec![
                "wl-paste".to_string(),
                "xclip".to_string(),
                "xsel".to_string()
            ]
        );
        set_native_clipboard_override(None);
    }
}

#[tokio::test]
async fn falls_back_to_x11_tools_when_wl_paste_is_unavailable() {
    let runner = MockRunner::default();
    runner
        .results
        .lock()
        .expect("mock")
        .insert("xclip".to_string(), Some(b"X11 text".to_vec()));
    let env = env_lookup(&[("WAYLAND_DISPLAY", "wayland-0"), ("DISPLAY", ":0")]);
    assert_eq!(
        read_clipboard_text_with(&env, Platform::Linux, &runner).await,
        Some("X11 text".to_string())
    );
    assert_eq!(
        *runner.calls.lock().expect("mock"),
        vec!["wl-paste".to_string(), "xclip".to_string()]
    );
}

// === copyToClipboard ========================================================

#[tokio::test]
async fn local_native_success_skips_osc_52_and_commands() {
    let writes = install_osc_sink();
    let clipboard = Arc::new(MockClipboard::new());
    set_native_clipboard_override(Some(clipboard.clone()));
    copy_to_clipboard_with(
        "hello",
        &empty_env(),
        Platform::Darwin,
        &MockRunner::default(),
    )
    .await
    .expect("copy");
    assert_eq!(
        *clipboard.set_text_calls.lock().expect("mock"),
        vec!["hello".to_string()]
    );
    assert!(writes.lock().expect("sink").is_empty());
    set_native_clipboard_override(None);
    set_osc52_sink(None);
}

#[tokio::test]
async fn linux_skips_the_native_writer() {
    let clipboard = Arc::new(MockClipboard::new());
    set_native_clipboard_override(Some(clipboard.clone()));
    let runner = MockRunner::default();
    runner
        .results
        .lock()
        .expect("mock")
        .insert("xclip".to_string(), Some(Vec::new()));
    let env = env_lookup(&[("DISPLAY", ":0")]);
    copy_to_clipboard_with("hello", &env, Platform::Linux, &runner)
        .await
        .expect("copy");
    assert!(clipboard.set_text_calls.lock().expect("mock").is_empty());
    set_native_clipboard_override(None);
}

#[tokio::test]
async fn emits_remote_osc_52_after_the_native_write_settles() {
    // Upstream drives this ordering through a pending promise; the sync
    // native write holds it by construction — the end state is one OSC 52
    // write and no command call.
    let writes = install_osc_sink();
    let clipboard = Arc::new(MockClipboard::new());
    set_native_clipboard_override(Some(clipboard.clone()));
    let env = env_lookup(&[("SSH_CONNECTION", "client server")]);
    copy_to_clipboard_with("hello", &env, Platform::Darwin, &MockRunner::default())
        .await
        .expect("copy");
    assert_eq!(writes.lock().expect("sink").len(), 1);
    set_native_clipboard_override(None);
    set_osc52_sink(None);
}

#[tokio::test]
async fn a_rejected_native_write_falls_back_to_pbcopy() {
    let writes = install_osc_sink();
    let clipboard = Arc::new(MockClipboard::new());
    *clipboard.set_text_result.lock().expect("mock") = false;
    set_native_clipboard_override(Some(clipboard.clone()));
    let runner = MockRunner::default();
    runner
        .results
        .lock()
        .expect("mock")
        .insert("pbcopy".to_string(), Some(Vec::new()));
    copy_to_clipboard_with("hello", &empty_env(), Platform::Darwin, &runner)
        .await
        .expect("copy");
    assert!(writes.lock().expect("sink").is_empty());
    set_native_clipboard_override(None);
    set_osc52_sink(None);
}

#[tokio::test]
async fn a_read_only_native_clipboard_uses_the_command_writer() {
    let clipboard = Arc::new(MockClipboard::with_set_text(false));
    set_native_clipboard_override(Some(clipboard.clone()));
    let runner = MockRunner::resolve_all_to_empty();
    copy_to_clipboard_with("hello", &empty_env(), Platform::Darwin, &runner)
        .await
        .expect("copy");
    assert_eq!(runner.calls.lock().expect("mock").len(), 1);
    set_native_clipboard_override(None);
}

#[tokio::test]
async fn tries_xclip_and_xsel_after_wl_copy_fails() {
    let writes = install_osc_sink();
    let runner = MockRunner::default();
    runner
        .results
        .lock()
        .expect("mock")
        .insert("xsel".to_string(), Some(Vec::new()));
    let env = env_lookup(&[("WAYLAND_DISPLAY", "wayland-0"), ("DISPLAY", ":0")]);
    copy_to_clipboard_with("hello", &env, Platform::Linux, &runner)
        .await
        .expect("copy");
    assert_eq!(
        *runner.calls.lock().expect("mock"),
        vec![
            "wl-copy".to_string(),
            "xclip".to_string(),
            "xsel".to_string()
        ]
    );
    assert!(writes.lock().expect("sink").is_empty());
    set_osc52_sink(None);
}

#[tokio::test]
async fn local_linux_failure_does_not_report_an_unverified_osc_52_write_as_success() {
    // Regression test for #9618.
    let writes = install_osc_sink();
    let runner = MockRunner::default();
    let env = env_lookup(&[("DISPLAY", ":0")]);
    let error = copy_to_clipboard_with("hello", &env, Platform::Linux, &runner)
        .await
        .expect_err("unavailable");
    assert_eq!(
        error.0,
        "Clipboard unavailable: install `xclip` or `xsel`, or check X11 access"
    );
    assert_eq!(
        *runner.calls.lock().expect("mock"),
        vec!["xclip".to_string(), "xsel".to_string()]
    );
    assert!(writes.lock().expect("sink").is_empty());
    set_osc52_sink(None);
}

#[tokio::test]
async fn reports_the_wayland_clipboard_tool_instead_of_the_x11_fallback() {
    let runner = MockRunner::default();
    let env = env_lookup(&[("WAYLAND_DISPLAY", "wayland-0"), ("DISPLAY", ":0")]);
    let error = copy_to_clipboard_with("hello", &env, Platform::Linux, &runner)
        .await
        .expect_err("unavailable");
    assert_eq!(
        error.0,
        "Clipboard unavailable: install `wl-clipboard` (`wl-copy`) or check Wayland access"
    );
    assert_eq!(
        *runner.calls.lock().expect("mock"),
        vec![
            "wl-copy".to_string(),
            "xclip".to_string(),
            "xsel".to_string()
        ]
    );
}

#[tokio::test]
async fn uses_osc_52_when_native_and_command_writes_fail_in_a_remote_session() {
    let writes = install_osc_sink();
    let clipboard = Arc::new(MockClipboard::new());
    *clipboard.set_text_result.lock().expect("mock") = false;
    set_native_clipboard_override(Some(clipboard));
    let runner = MockRunner::default();
    let env = env_lookup(&[("SSH_CONNECTION", "client server")]);
    copy_to_clipboard_with("hello", &env, Platform::Darwin, &runner)
        .await
        .expect("osc 52 carried it");
    assert_eq!(writes.lock().expect("sink").len(), 1);
    set_native_clipboard_override(None);
    set_osc52_sink(None);
}

#[tokio::test]
async fn does_not_emit_oversized_osc_52_payloads() {
    let writes = install_osc_sink();
    let clipboard = Arc::new(MockClipboard::new());
    *clipboard.set_text_result.lock().expect("mock") = false;
    set_native_clipboard_override(Some(clipboard));
    let runner = MockRunner::default();
    let env = env_lookup(&[("SSH_CONNECTION", "client server")]);
    let error = copy_to_clipboard_with(&"x".repeat(80_000), &env, Platform::Darwin, &runner)
        .await
        .expect_err("too large for osc 52");
    assert_eq!(error.0, "Clipboard unavailable");
    assert!(writes.lock().expect("sink").is_empty());
    set_native_clipboard_override(None);
    set_osc52_sink(None);
}

#[tokio::test]
async fn reports_the_platform_diagnosis_per_environment() {
    let runner = MockRunner::default();
    let termux = env_lookup(&[("TERMUX_VERSION", "0.119")]);
    let error = copy_to_clipboard_with("hello", &termux, Platform::Linux, &runner)
        .await
        .expect_err("termux diagnosis");
    assert_eq!(
        error.0,
        "Clipboard unavailable: install the Termux:API app and `termux-api` package"
    );
    let headless = env_lookup(&[]);
    let error = copy_to_clipboard_with("hello", &headless, Platform::Linux, &runner)
        .await
        .expect_err("headless diagnosis");
    assert_eq!(
        error.0,
        "Clipboard unavailable: no Wayland or X11 display detected"
    );
    // A write-less native surface and no command writer: the generic
    // diagnosis.
    set_native_clipboard_disabled();
    let win_error = copy_to_clipboard_with("hello", &empty_env(), Platform::Win32, &runner)
        .await
        .expect_err("windows diagnosis");
    assert_eq!(win_error.0, "Clipboard unavailable");
    set_native_clipboard_override(None);
}
