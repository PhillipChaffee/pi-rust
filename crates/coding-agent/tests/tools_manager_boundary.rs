//! Boundary tests for the tools manager, upstream's `tools-manager.ts`
//! (no upstream suite imports it; the offline, termux, and version-probe
//! paths bind here, and the network download paths gate on the runner's
//! tool inventory the same way the credential-gated suites gate).

#![expect(
    clippy::unwrap_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::sync::Arc;

use pi_ai::http::mock::{MockHttpClient, MockResponse};

use pi_coding_agent::utils::tools_manager::{
    ManagedTool, ToolStatus, ensure_tool_with, get_latest_version_with, get_tool_path,
    install_tool, is_offline_mode_enabled, is_offline_mode_enabled_with,
};

fn env_with(entries: &[(&str, &str)]) -> pi_coding_agent::config::EnvLookup {
    let map: std::collections::BTreeMap<String, String> = entries
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect();
    Box::new(move |key: &str| map.get(key).cloned())
}

#[test]
fn offline_mode_reads_the_injected_env() {
    assert!(!is_offline_mode_enabled_with(&env_with(&[])));
    assert!(is_offline_mode_enabled_with(&env_with(&[(
        "PI_OFFLINE",
        "1"
    )])));
    assert!(is_offline_mode_enabled_with(&env_with(&[(
        "PI_OFFLINE",
        "true"
    )])));
    assert!(is_offline_mode_enabled_with(&env_with(&[(
        "PI_OFFLINE",
        "Yes"
    )])));
    assert!(!is_offline_mode_enabled_with(&env_with(&[(
        "PI_OFFLINE",
        "0"
    )])));
    assert!(!is_offline_mode_enabled_with(&env_with(&[(
        "PI_OFFLINE",
        "off"
    )])));
}

#[test]
fn the_asset_names_carry_the_platform_matrix() {
    assert_eq!(
        ManagedTool::Fd.asset_name_for_test("10.3.0", "darwin", "x64"),
        Some("fd-v10.3.0-x86_64-apple-darwin.tar.gz".to_owned())
    );
    assert_eq!(
        ManagedTool::Fd.asset_name_for_test("10.3.0", "darwin", "arm64"),
        Some("fd-v10.3.0-aarch64-apple-darwin.tar.gz".to_owned())
    );
    assert_eq!(
        ManagedTool::Fd.asset_name_for_test("10.3.0", "linux", "arm64"),
        Some("fd-v10.3.0-aarch64-unknown-linux-musl.tar.gz".to_owned())
    );
    assert_eq!(
        ManagedTool::Rg.asset_name_for_test("14.1.1", "linux", "x64"),
        Some("ripgrep-14.1.1-x86_64-unknown-linux-musl.tar.gz".to_owned())
    );
    // Windows is out of scope for this effort (map ticket "Decide the Rust
    // stack"): the win32 arm is unreachable.
    assert_eq!(
        ManagedTool::Fd.asset_name_for_test("10.3.0", "win32", "x64"),
        None
    );
}

#[test]
fn the_tool_path_probe_is_self_consistent() {
    for tool in [ManagedTool::Fd, ManagedTool::Rg] {
        if let Some(path) = get_tool_path(tool) {
            // A managed-bin hit ends in the bin dir; a system hit is the
            // command name itself, which runs.
            let is_command_name = path == "fd"
                || path == "fdfind"
                || path == "rg"
                || path.ends_with("/fd")
                || path.ends_with("/rg")
                || path.ends_with("/fd.exe")
                || path.ends_with("/rg.exe");
            assert!(is_command_name, "{path}");
        }
    }
}

#[tokio::test]
async fn offline_mode_skips_the_download_with_a_warning() {
    // The PATH probe precedes the offline check upstream too; when the
    // runner has the tool, the resolution succeeds and the warning arm is
    // unreachable.
    if get_tool_path(ManagedTool::Fd).is_some() {
        return;
    }
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(MockHttpClient::new());
    let statuses = Arc::new(std::sync::Mutex::new(Vec::<ToolStatus>::new()));
    let sink = Arc::clone(&statuses);
    let env = env_with(&[("PI_OFFLINE", "1")]);
    let resolved = ensure_tool_with(
        &client,
        ManagedTool::Fd,
        Some(&move |status: &ToolStatus| {
            sink.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(status.clone());
        }),
        &env,
    )
    .await;
    assert!(resolved.is_none());
    let collected = statuses
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert!(matches!(
        collected.as_slice(),
        [ToolStatus::Warning(message)] if message.contains("fd not found. Offline mode enabled, skipping download.")
    ));
}

#[tokio::test]
async fn the_termux_environment_hints_the_package_manager() {
    // Same precedence: an installed rg resolves before the hint.
    if get_tool_path(ManagedTool::Rg).is_some() {
        return;
    }
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(MockHttpClient::new());
    let statuses = Arc::new(std::sync::Mutex::new(Vec::<ToolStatus>::new()));
    let sink = Arc::clone(&statuses);
    let env = env_with(&[("TERMUX_VERSION", "0.118")]);
    let resolved = ensure_tool_with(
        &client,
        ManagedTool::Rg,
        Some(&move |status: &ToolStatus| {
            sink.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(status.clone());
        }),
        &env,
    )
    .await;
    assert!(resolved.is_none());
    let collected = statuses
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert!(matches!(
        collected.as_slice(),
        [ToolStatus::Warning(message)] if message.contains("ripgrep not found. Install with: pkg install ripgrep")
    ));
}

#[tokio::test]
async fn the_version_probe_reads_the_redirect_tag() {
    // The no-redirect probe: the mock answers the release page with the
    // redirect the web endpoint sends, upstream's `getLatestVersion`.
    let client = MockHttpClient::new();
    client
        .on(|request| request.url.ends_with("/sharkdp/fd/releases/latest"))
        .respond(MockResponse::status(302).with_header(
            "Location",
            "https://github.com/sharkdp/fd/releases/tag/v10.3.0",
        ));
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(client);
    let version = get_latest_version_with(&client, "sharkdp/fd")
        .await
        .unwrap();
    assert_eq!(version, "10.3.0");
}

#[tokio::test]
async fn the_version_probe_rejects_a_response_without_a_redirect() {
    let client = MockHttpClient::new();
    client
        .on(|request| request.url.ends_with("/sharkdp/fd/releases/latest"))
        .respond(MockResponse::status(200).with_body("release page"));
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(client);
    let error = get_latest_version_with(&client, "sharkdp/fd")
        .await
        .unwrap_err();
    assert_eq!(
        error,
        "Failed to resolve latest sharkdp/fd release: HTTP 200 without redirect"
    );
}

#[tokio::test]
async fn the_version_probe_rejects_an_unexpected_redirect() {
    let client = MockHttpClient::new();
    client
        .on(|request| request.url.ends_with("/sharkdp/fd/releases/latest"))
        .respond(
            MockResponse::status(302).with_header("Location", "https://github.com/login?next=here"),
        );
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(client);
    let error = get_latest_version_with(&client, "sharkdp/fd")
        .await
        .unwrap_err();
    assert!(
        error.contains("Failed to resolve latest sharkdp/fd release: unexpected redirect to"),
        "{error}"
    );
}

#[test]
fn offline_mode_reads_the_process_environment() {
    let expected = std::env::var("PI_OFFLINE").is_ok_and(|value| {
        value == "1" || value.to_lowercase() == "true" || value.to_lowercase() == "yes"
    });
    assert_eq!(is_offline_mode_enabled(), expected);
}

/// Build a tar.gz holding one file, upstream's release archive shape.
fn write_tar_fixture(dir: &std::path::Path, member_dir: &str, name: &str) -> Vec<u8> {
    let payload = dir.join("payload").join(member_dir);
    std::fs::create_dir_all(&payload).unwrap();
    std::fs::write(payload.join(name), b"#!/bin/sh\n").unwrap();
    let archive = dir.join("fixture.tar.gz");
    let packed = std::process::Command::new("tar")
        .args(["czf"])
        .arg(&archive)
        .arg("-C")
        .arg(dir.join("payload"))
        .arg(member_dir)
        .status()
        .unwrap();
    assert!(packed.success(), "tar fixture build");
    std::fs::read(&archive).unwrap()
}

#[tokio::test]
async fn install_places_the_binary_and_sweeps_the_download_scratch() {
    let dir = tempfile::tempdir().unwrap();
    let archive_bytes = write_tar_fixture(dir.path(), "fd-v10.3.0", "fd");
    let mock = MockHttpClient::new();
    mock.on(|_request| true)
        .respond(MockResponse::status(200).with_body(archive_bytes));
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(mock);
    let tools_dir = dir.path().join("bin");
    let installed = install_tool(&client, ManagedTool::Fd, "10.3.0", &tools_dir)
        .await
        .unwrap();
    assert_eq!(
        installed,
        tools_dir.join("fd").to_string_lossy().into_owned()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(tools_dir.join("fd"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o755, "the binary installs executable");
    }
    // The download scratch is swept after the layout step: no archive or
    // extraction leftovers sit in the tools dir.
    let leftovers: Vec<_> = std::fs::read_dir(&tools_dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(leftovers, vec!["fd".to_owned()]);
}

#[tokio::test]
async fn an_archive_without_the_binary_reports_the_expected_layout() {
    let dir = tempfile::tempdir().unwrap();
    let archive_bytes = write_tar_fixture(dir.path(), "payload", "not-fd");
    let mock = MockHttpClient::new();
    mock.on(|_request| true)
        .respond(MockResponse::status(200).with_body(archive_bytes));
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(mock);
    let error = install_tool(&client, ManagedTool::Fd, "10.3.0", &dir.path().join("bin"))
        .await
        .unwrap_err();
    assert!(error.contains("Binary not found in archive"), "{error}");
}
