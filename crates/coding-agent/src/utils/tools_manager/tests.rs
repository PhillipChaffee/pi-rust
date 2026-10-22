#![expect(
    clippy::unwrap_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use super::*;
use pi_ai::http::client::HttpMethod;
use pi_ai::http::mock::{MockHttpClient, MockResponse};

#[test]
fn urldecode_expands_percent_escapes_and_keeps_the_rest() {
    assert_eq!(urldecode("v10.3.0"), "v10.3.0");
    assert_eq!(urldecode("10%2E3%2e0"), "10.3.0");
    assert_eq!(urldecode("a%20b%2Fc"), "a b/c");
    // A stray percent without two hex digits stays literal.
    assert_eq!(urldecode("100%zz"), "100%zz");
    assert_eq!(urldecode("trailing%2"), "trailing%2");
}

#[test]
fn spawn_failure_prefers_stderr_then_stdout_then_the_exit_status() {
    assert_eq!(format_spawn_failure(Some(2), "boom\n", "ignored\n"), "boom");
    assert_eq!(format_spawn_failure(Some(2), "", "  out  \n"), "out");
    assert_eq!(format_spawn_failure(Some(2), "", ""), "exit status 2");
    assert_eq!(format_spawn_failure(None, "", ""), "exit status unknown");
}

#[test]
fn command_presence_runs_the_command() {
    assert!(command_exists("echo"));
    assert!(!command_exists("pi-coding-agent-no-such-binary"));
}

#[test]
fn the_platform_pair_yields_a_supported_asset() {
    let (platform, architecture) = platform_and_arch();
    for tool in [ManagedTool::Fd, ManagedTool::Rg] {
        let asset = tool.asset_name_for_test("1.0.0", platform, architecture);
        assert!(asset.is_some(), "{platform}/{architecture}");
        assert!(asset.unwrap().ends_with(".tar.gz"));
    }
}

#[test]
fn offline_mode_reads_the_process_environment() {
    let expected = std::env::var("PI_OFFLINE").is_ok_and(|value| {
        value == "1" || value.to_lowercase() == "true" || value.to_lowercase() == "yes"
    });
    assert_eq!(is_offline_mode_enabled(), expected);
}

#[test]
fn extraction_commands_report_success_and_failure() {
    // Success: any zero-exit command.
    assert_eq!(run_extraction_command("echo", &["ok"]), None);
    // Failure: the message carries the command and the exit rendering.
    let failure = run_extraction_command("sh", &["-c", "exit 3"]).unwrap();
    assert_eq!(failure, "sh: exit status 3");
    // A nonexistent command spawns a failure, not a panic.
    assert!(run_extraction_command("pi-coding-agent-no-such-binary", &[]).is_some());
}

#[test]
fn tar_gz_archives_extract_and_failures_render_the_asset_name() {
    let dir = tempfile::tempdir().unwrap();
    let payload = dir.path().join("payload");
    std::fs::create_dir_all(&payload).unwrap();
    std::fs::write(payload.join("fd"), "#!/bin/sh\necho fd\n").unwrap();
    let archive = dir.path().join("fd.tar.gz");
    let packed = std::process::Command::new("tar")
        .args(["czf"])
        .arg(&archive)
        .arg("-C")
        .arg(&payload)
        .arg("fd")
        .status()
        .unwrap();
    assert!(packed.success(), "tar fixture build");

    let extract_dir = dir.path().join("out");
    std::fs::create_dir_all(&extract_dir).unwrap();
    extract_tar_gz_archive(
        &archive.to_string_lossy(),
        &extract_dir.to_string_lossy(),
        "fd-v10.3.0.tar.gz",
    )
    .unwrap();
    assert!(extract_dir.join("fd").exists());

    let corrupt = dir.path().join("corrupt.tar.gz");
    std::fs::write(&corrupt, b"not a tarball").unwrap();
    let error = extract_tar_gz_archive(
        &corrupt.to_string_lossy(),
        &dir.path().join("out2").to_string_lossy(),
        "fd-v10.3.0.tar.gz",
    )
    .unwrap_err();
    assert!(
        error.contains("Failed to extract fd-v10.3.0.tar.gz"),
        "{error}"
    );
}

#[test]
fn zip_archives_extract_and_the_failure_ladder_joins_both_messages() {
    let dir = tempfile::tempdir().unwrap();
    let payload = dir.path().join("payload");
    std::fs::create_dir_all(&payload).unwrap();
    std::fs::write(payload.join("rg"), "#!/bin/sh\necho rg\n").unwrap();
    let archive = dir.path().join("rg.zip");
    let packed = std::process::Command::new("zip")
        .args(["-q", "-j"])
        .arg(&archive)
        .arg(payload.join("rg"))
        .status()
        .unwrap();
    assert!(packed.success(), "zip fixture build");

    let extract_dir = dir.path().join("out");
    std::fs::create_dir_all(&extract_dir).unwrap();
    extract_zip_archive(
        &archive.to_string_lossy(),
        &extract_dir.to_string_lossy(),
        "ripgrep-14.1.1.zip",
    )
    .unwrap();
    assert!(extract_dir.join("rg").exists());

    let corrupt = dir.path().join("corrupt.zip");
    std::fs::write(&corrupt, b"not a zip").unwrap();
    let error = extract_zip_archive(
        &corrupt.to_string_lossy(),
        &dir.path().join("out2").to_string_lossy(),
        "ripgrep-14.1.1.zip",
    )
    .unwrap_err();
    assert!(error.contains("unzip"), "{error}");
    assert!(error.contains("tar"), "{error}");
}

#[test]
fn the_binary_search_walks_the_extracted_tree() {
    let dir = tempfile::tempdir().unwrap();
    let nested = dir.path().join("fd-v10.3.0").join("sub");
    std::fs::create_dir_all(&nested).unwrap();
    let binary = nested.join("fd");
    std::fs::write(&binary, b"#!/bin/sh\n").unwrap();
    assert_eq!(find_binary_recursively(dir.path(), "fd"), Some(binary));
    assert_eq!(find_binary_recursively(dir.path(), "rg"), None);
}

#[tokio::test]
async fn downloads_stream_to_the_destination_file() {
    let mock = MockHttpClient::new();
    mock.on(|_request| true)
        .respond(MockResponse::status(200).with_body("archive bytes"));
    let client: Arc<dyn HttpClient> = Arc::new(mock);
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("archive.tar.gz");
    download_file(&client, "https://example.test/archive.tar.gz", &dest)
        .await
        .unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), b"archive bytes");
}

#[tokio::test]
async fn download_failures_carry_the_status_and_url() {
    let mock = MockHttpClient::new();
    mock.on(|_request| true).respond(MockResponse::status(404));
    let client: Arc<dyn HttpClient> = Arc::new(mock);
    let dir = tempfile::tempdir().unwrap();
    let error = download_file(
        &client,
        "https://example.test/archive.tar.gz",
        &dir.path().join("out.bin"),
    )
    .await
    .unwrap_err();
    assert!(error.contains("Download failed with HTTP 404"), "{error}");
}

#[tokio::test]
async fn install_places_the_binary_and_makes_it_executable() {
    // The archive layout mirrors a real fd release: the binary nests
    // under the versioned directory the extracted-dir candidate names.
    let dir = tempfile::tempdir().unwrap();
    let payload = dir.path().join("payload").join("fd-v10.3.0");
    std::fs::create_dir_all(&payload).unwrap();
    std::fs::write(payload.join("fd"), b"#!/bin/sh\n").unwrap();
    let archive = dir.path().join("fd.tar.gz");
    let packed = std::process::Command::new("tar")
        .args(["czf"])
        .arg(&archive)
        .arg("-C")
        .arg(dir.path().join("payload"))
        .arg("fd-v10.3.0")
        .status()
        .unwrap();
    assert!(packed.success(), "tar fixture build");
    let archive_bytes = std::fs::read(&archive).unwrap();

    let mock = MockHttpClient::new();
    mock.on(|_request| true)
        .respond(MockResponse::status(200).with_body(archive_bytes));
    let client: Arc<dyn HttpClient> = Arc::new(mock);
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
    // The download scratch is swept after the layout step.
    let (platform, architecture) = platform_and_arch();
    let asset_name = ManagedTool::Fd
        .asset_name_for_test("10.3.0", platform, architecture)
        .unwrap();
    assert!(!tools_dir.join(&asset_name).exists());
}

#[tokio::test]
async fn an_archive_without_the_binary_reports_the_expected_layout() {
    let dir = tempfile::tempdir().unwrap();
    let payload = dir.path().join("payload");
    std::fs::create_dir_all(&payload).unwrap();
    std::fs::write(payload.join("not-fd"), b"").unwrap();
    let archive = dir.path().join("fd.tar.gz");
    let packed = std::process::Command::new("tar")
        .args(["czf"])
        .arg(&archive)
        .arg("-C")
        .arg(&payload)
        .arg("not-fd")
        .status()
        .unwrap();
    assert!(packed.success(), "tar fixture build");
    let archive_bytes = std::fs::read(&archive).unwrap();

    let mock = MockHttpClient::new();
    mock.on(|_request| true)
        .respond(MockResponse::status(200).with_body(archive_bytes));
    let client: Arc<dyn HttpClient> = Arc::new(mock);
    let error = install_tool(&client, ManagedTool::Fd, "10.3.0", &dir.path().join("bin"))
        .await
        .unwrap_err();
    assert!(error.contains("Binary not found in archive"), "{error}");
}

#[test]
fn the_probe_client_builds_once() {
    probe_client().unwrap();
}

#[tokio::test]
async fn the_probe_client_reads_redirects_without_following_them() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = vec![0u8; 4096];
        let mut read = 0usize;
        loop {
            let n = socket.read(&mut request[read..]).await.unwrap();
            read += n;
            if request[..read]
                .windows(4)
                .any(|window| window == b"\r\n\r\n")
            {
                break;
            }
        }
        socket
            .write_all(
                b"HTTP/1.1 302 Found\r\nLocation: https://github.com/sharkdp/fd/releases/tag/v10.3.0\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )
            .await
            .unwrap();
    });

    let client = NoRedirectProbeClient::new().unwrap();
    let response = client
        .execute(HttpRequest {
            method: HttpMethod::Get,
            url: format!("http://{addr}/sharkdp/fd/releases/latest"),
            headers: vec![],
            body: None,
            timeout_ms: None,
            signal: CancellationToken::new(),
        })
        .await
        .unwrap();
    assert_eq!(response.status, 302);
    let location = response
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("location"))
        .map(|(_, value)| value.clone())
        .unwrap();
    assert_eq!(
        location,
        "https://github.com/sharkdp/fd/releases/tag/v10.3.0"
    );
}

#[tokio::test]
async fn a_pre_cancelled_signal_aborts_the_probe_before_the_request() {
    let client = NoRedirectProbeClient::new().unwrap();
    let signal = CancellationToken::new();
    signal.cancel();
    let error = client
        .execute(HttpRequest {
            method: HttpMethod::Get,
            url: "https://example.test/releases/latest".to_owned(),
            headers: vec![],
            body: None,
            timeout_ms: None,
            signal,
        })
        .await
        .unwrap_err();
    assert!(matches!(error, HttpError::Aborted));
}

#[tokio::test]
async fn offline_mode_reports_the_skip_before_any_download() {
    // The PATH probe precedes the offline check upstream too; when the
    // runner has the tool, the resolution succeeds and the warning arm
    // is unreachable.
    if get_tool_path(ManagedTool::Fd).is_some() {
        return;
    }
    let client: Arc<dyn HttpClient> = Arc::new(MockHttpClient::new());
    let statuses = std::sync::Mutex::new(Vec::<ToolStatus>::new());
    let sink = &statuses;
    let env: crate::config::EnvLookup =
        Box::new(|key: &str| (key == "PI_OFFLINE").then(|| "1".to_owned()));
    let resolved = ensure_tool_with(
        &client,
        ManagedTool::Fd,
        Some(&|status: &ToolStatus| sink.lock().unwrap().push(status.clone())),
        &env,
    )
    .await;
    assert!(resolved.is_none());
    let collected = statuses.into_inner().unwrap();
    assert!(matches!(
        collected.as_slice(),
        [ToolStatus::Warning(message)] if message.contains("fd not found. Offline mode enabled, skipping download.")
    ));
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
                || path.ends_with("/rg");
            assert!(is_command_name, "{path}");
        }
    }
}

#[tokio::test]
async fn the_termux_environment_hints_the_package_manager() {
    // Same precedence as upstream: an installed rg resolves first.
    if get_tool_path(ManagedTool::Rg).is_some() {
        return;
    }
    let client: Arc<dyn HttpClient> = Arc::new(MockHttpClient::new());
    let statuses = std::sync::Mutex::new(Vec::<ToolStatus>::new());
    let sink = &statuses;
    let env: crate::config::EnvLookup =
        Box::new(|key: &str| (key == "TERMUX_VERSION").then(|| "0.118".to_owned()));
    let resolved = ensure_tool_with(
        &client,
        ManagedTool::Rg,
        Some(&|status: &ToolStatus| sink.lock().unwrap().push(status.clone())),
        &env,
    )
    .await;
    assert!(resolved.is_none());
    let collected = statuses.into_inner().unwrap();
    assert!(matches!(
        collected.as_slice(),
        [ToolStatus::Warning(message)] if message.contains("ripgrep not found. Install with: pkg install ripgrep")
    ));
}

#[tokio::test]
async fn the_version_probe_resolves_the_redirect_tag() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.ends_with("/sharkdp/fd/releases/latest"))
        .respond(MockResponse::status(302).with_header(
            "Location",
            "https://github.com/sharkdp/fd/releases/tag/v10.3.0",
        ));
    let client: Arc<dyn HttpClient> = Arc::new(mock);
    let version = get_latest_version_with(&client, "sharkdp/fd")
        .await
        .unwrap();
    assert_eq!(version, "10.3.0");
}

#[tokio::test]
async fn the_version_probe_keeps_tags_without_a_v_prefix() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.ends_with("/sharkdp/fd/releases/latest"))
        .respond(MockResponse::status(302).with_header(
            "Location",
            "https://github.com/sharkdp/fd/releases/tag/10.3.0",
        ));
    let client: Arc<dyn HttpClient> = Arc::new(mock);
    let version = get_latest_version_with(&client, "sharkdp/fd")
        .await
        .unwrap();
    assert_eq!(version, "10.3.0");
}

#[tokio::test]
async fn the_version_probe_rejects_a_response_without_a_redirect() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.ends_with("/sharkdp/fd/releases/latest"))
        .respond(MockResponse::status(200).with_body("release page"));
    let client: Arc<dyn HttpClient> = Arc::new(mock);
    let error = get_latest_version_with(&client, "sharkdp/fd")
        .await
        .unwrap_err();
    assert_eq!(
        error,
        "Failed to resolve latest sharkdp/fd release: HTTP 200 without redirect"
    );
}

#[tokio::test]
async fn the_version_probe_rejects_an_unexpected_redirect_target() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.ends_with("/sharkdp/fd/releases/latest"))
        .respond(
            MockResponse::status(302).with_header("Location", "https://github.com/login?next=here"),
        );
    let client: Arc<dyn HttpClient> = Arc::new(mock);
    let error = get_latest_version_with(&client, "sharkdp/fd")
        .await
        .unwrap_err();
    assert!(
        error.contains("Failed to resolve latest sharkdp/fd release: unexpected redirect to"),
        "{error}"
    );
}
