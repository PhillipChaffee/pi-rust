//! The tools manager, ported from upstream `src/utils/tools-manager.ts`.
//!
//! The manager carries the fd/rg binary resolution, the GitHub-release
//! download pipeline, and the status-reporting `ensureTool`. The grep/find
//! tools restate the binaries' semantics natively (the tools slice's
//! recorded decision), so the manager's consumers are the interactive
//! mode's prefetch (its ticket) and any remote-operations backend the tool
//! seams delegate to.
//!
//! The version probe rides its own no-redirect client, upstream's
//! `redirect: "manual"` fetch: the process-default client follows
//! redirects, so the probe reads the `Location` header the release page
//! answers with only when redirects are refused.

use std::path::PathBuf;
use std::sync::Arc;

use pi_ai::http::HttpClient;
use pi_ai::http::client::{BoxHttpFuture, HttpByteStream, HttpError, HttpRequest, HttpResponse};
use tokio_util::sync::CancellationToken;

use crate::config::{APP_NAME, get_bin_dir};
use crate::utils::child_process::{SpawnSyncOptions, spawn_process_sync};
use crate::utils::management_http::{FetchRetryOptions, fetch_with_retry};

/// The no-redirect probe client, upstream's `redirect: "manual"` fetch:
/// the process-default client follows redirects, so the version probe
/// refuses them to read the `Location` header the release page answers
/// with.
#[derive(Debug)]
struct NoRedirectProbeClient {
    client: reqwest::Client,
}

impl NoRedirectProbeClient {
    fn new() -> Result<Self, reqwest::Error> {
        Ok(Self {
            client: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
        })
    }
}

impl HttpClient for NoRedirectProbeClient {
    fn execute(&self, request: HttpRequest) -> BoxHttpFuture<Result<HttpResponse, HttpError>> {
        use futures_util::StreamExt as _;

        let client = self.client.clone();
        Box::pin(async move {
            if request.signal.is_cancelled() {
                return Err(HttpError::Aborted);
            }
            let mut builder = client.get(&request.url);
            for (name, value) in &request.headers {
                builder = builder.header(name, value);
            }
            if let Some(timeout_ms) = request.timeout_ms {
                builder = builder.timeout(std::time::Duration::from_millis(timeout_ms));
            }
            let response = tokio::select! {
                () = request.signal.cancelled() => return Err(HttpError::Aborted),
                response = builder.send() => response.map_err(|error| HttpError::Transport(error.to_string()))?,
            };
            let status = response.status().as_u16();
            let headers: Vec<(String, String)> = response
                .headers()
                .iter()
                .map(|(name, value)| {
                    (
                        name.as_str().to_owned(),
                        value.to_str().unwrap_or_default().to_owned(),
                    )
                })
                .collect();
            let stream = response
                .bytes_stream()
                .map(|chunk| chunk.map_err(|error| HttpError::Transport(error.to_string())));
            Ok(HttpResponse {
                status,
                headers,
                body: HttpByteStream::new(stream),
            })
        })
    }
}

/// The process probe client, shared like the process-default one.
fn probe_client() -> Result<Arc<dyn HttpClient>, String> {
    static CLIENT: std::sync::OnceLock<Result<Arc<dyn HttpClient>, String>> =
        std::sync::OnceLock::new();
    CLIENT
        .get_or_init(|| {
            fn probe_http_client(client: NoRedirectProbeClient) -> Arc<dyn HttpClient> {
                Arc::new(client)
            }
            NoRedirectProbeClient::new()
                .map(probe_http_client)
                .map_err(|error| error.to_string())
        })
        .clone()
}

/// The network timeout, upstream's `NETWORK_TIMEOUT_MS`.
const NETWORK_TIMEOUT_MS: u64 = 10_000;
/// The download timeout, upstream's `DOWNLOAD_TIMEOUT_MS`.
const DOWNLOAD_TIMEOUT_MS: u64 = 120_000;

/// The managed tools, upstream's `TOOLS` record keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ManagedTool {
    /// The `fd` file finder.
    Fd,
    /// The `rg` ripgrep searcher.
    Rg,
}

impl ManagedTool {
    /// The download asset name for the platform, upstream's
    /// `getAssetName` — the observable surface the download URL composes.
    #[must_use]
    pub fn asset_name_for_test(
        self,
        version: &str,
        platform: &str,
        architecture: &str,
    ) -> Option<String> {
        self.asset_name(version, platform, architecture)
    }

    /// The config, upstream's `TOOLS[tool]`.
    fn config(self) -> ToolConfig {
        match self {
            Self::Fd => ToolConfig {
                name: "fd",
                repo: "sharkdp/fd",
                binary_name: "fd",
                system_binary_names: vec!["fd", "fdfind"],
                tag_prefix: "v",
                pinned_darwin_x64_version: Some("10.3.0"),
            },
            Self::Rg => ToolConfig {
                name: "ripgrep",
                repo: "BurntSushi/ripgrep",
                binary_name: "rg",
                system_binary_names: vec!["rg"],
                tag_prefix: "",
                pinned_darwin_x64_version: None,
            },
        }
    }

    /// The download asset name for the platform, upstream's
    /// `getAssetName`. Windows is out of scope for this effort (map ticket
    /// "Decide the Rust stack"), so the win32 arm is unreachable.
    fn asset_name(self, version: &str, platform: &str, architecture: &str) -> Option<String> {
        let arch = if architecture == "arm64" {
            "aarch64"
        } else {
            "x86_64"
        };
        match (self, platform) {
            (Self::Fd, "darwin") => Some(format!("fd-v{version}-{arch}-apple-darwin.tar.gz")),
            (Self::Fd, "linux") => Some(format!("fd-v{version}-{arch}-unknown-linux-musl.tar.gz")),
            (Self::Rg, "darwin") => Some(format!("ripgrep-{version}-{arch}-apple-darwin.tar.gz")),
            (Self::Rg, "linux") => Some(format!(
                "ripgrep-{version}-{arch}-unknown-linux-musl.tar.gz"
            )),
            _ => None,
        }
    }
}

/// One managed tool's download config, upstream's `ToolConfig`.
struct ToolConfig {
    name: &'static str,
    repo: &'static str,
    binary_name: &'static str,
    system_binary_names: Vec<&'static str>,
    tag_prefix: &'static str,
    pinned_darwin_x64_version: Option<&'static str>,
}

/// Whether offline mode is enabled, upstream's `isOfflineModeEnabled`.
#[must_use]
pub fn is_offline_mode_enabled() -> bool {
    is_offline_mode_enabled_with(&crate::config::default_env_lookup())
}

/// The offline check over an injected environment, the belt's `_with`
/// seam: the plain getter reads the real environment.
#[must_use]
pub fn is_offline_mode_enabled_with(env: &crate::config::EnvLookup) -> bool {
    let Some(value) = env("PI_OFFLINE") else {
        return false;
    };
    value == "1" || value.to_lowercase() == "true" || value.to_lowercase() == "yes"
}

/// Whether a command exists in PATH by running it, upstream's
/// `commandExists`: the spawn succeeding (any exit code) is existence.
fn command_exists(cmd: &str) -> bool {
    let result = spawn_process_sync(
        cmd,
        &["--version"],
        &SpawnSyncOptions {
            capture_output: true,
            capture_stderr: false,
            timeout_ms: Some(5_000),
        },
    );
    result.status.is_some()
}

/// The platform pair, upstream's `platform()`/`arch()`.
const fn platform_and_arch() -> (&'static str, &'static str) {
    let platform = if cfg!(target_os = "macos") {
        "darwin"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else {
        "unknown"
    };
    let architecture = if cfg!(target_arch = "aarch64") {
        "arm64"
    } else {
        "x64"
    };
    (platform, architecture)
}

/// The path to a tool (system-wide or in the managed bin dir), upstream's
/// `getToolPath`.
#[must_use]
pub fn get_tool_path(tool: ManagedTool) -> Option<String> {
    let config = tool.config();

    // Check the managed bin dir first.
    let (platform, _) = platform_and_arch();
    let executable_ext = if platform == "win32" { ".exe" } else { "" };
    let local_path = get_bin_dir().join(format!("{}{executable_ext}", config.binary_name));
    if local_path.exists() {
        return Some(local_path.to_string_lossy().into_owned());
    }

    // Check system PATH — if found, return the command name (it's in PATH).
    for system_binary_name in config.system_binary_names {
        if command_exists(system_binary_name) {
            return Some((*system_binary_name).to_owned());
        }
    }

    None
}

/// Resolve the latest release version from the release page redirect,
/// upstream's `getLatestVersion`.
///
/// The api.github.com releases endpoint counts against the anonymous API
/// quota (60 requests/hour per IP), which is permanently exhausted behind
/// shared egress IPs; the web endpoint answers with a redirect to the
/// tagged release at no quota cost and lives on the same origin as the
/// binary download itself.
///
/// # Errors
/// The two resolution failures, upstream's throws: a non-redirect response
/// and an unexpected redirect target.
pub async fn get_latest_version(repo: &str) -> Result<String, String> {
    let probe_client = probe_client()?;
    get_latest_version_with(&probe_client, repo).await
}

/// The version resolution over an injected client, the testable core the
/// [`get_latest_version`] probe wraps.
///
/// # Errors
/// The two resolution failures, upstream's throws.
pub async fn get_latest_version_with(
    client: &Arc<dyn HttpClient>,
    repo: &str,
) -> Result<String, String> {
    let response = fetch_with_retry(
        client,
        &format!("https://github.com/{repo}/releases/latest"),
        vec![("User-Agent".to_owned(), format!("{APP_NAME}-coding-agent"))],
        CancellationToken::new(),
        FetchRetryOptions {
            timeout_ms: Some(NETWORK_TIMEOUT_MS),
            ..FetchRetryOptions::default()
        },
    )
    .await
    .map_err(|error| error.to_string())?;

    let location = (300..400).contains(&response.status).then(|| {
        response
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("location"))
            .map(|(_, value)| value.clone())
    });
    let Some(Some(location)) = location else {
        return Err(format!(
            "Failed to resolve latest {repo} release: HTTP {} without redirect",
            response.status
        ));
    };

    let tag = url::Url::parse(&location).ok().and_then(|parsed| {
        if !parsed.path().contains("/releases/tag/") {
            return None;
        }
        parsed.path().rsplit('/').next().map(str::to_owned)
    });
    let Some(tag) = tag.filter(|tag| !tag.is_empty()) else {
        return Err(format!(
            "Failed to resolve latest {repo} release: unexpected redirect to {location}"
        ));
    };
    let tag = urldecode(&tag);
    Ok(tag.strip_prefix('v').unwrap_or(&tag).to_owned())
}

/// The percent-decode, upstream's `decodeURIComponent`.
fn urldecode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'%'
            && index + 2 < bytes.len()
            && let Ok(byte) = u8::from_str_radix(&value[index + 1..index + 3], 16)
        {
            out.push(byte);
            index += 3;
        } else {
            out.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The extraction command's failure rendering, upstream's
/// `formatSpawnFailure`.
fn format_spawn_failure(status: Option<i32>, stderr: &str, stdout: &str) -> String {
    if !stderr.trim().is_empty() {
        return stderr.trim().to_owned();
    }
    if !stdout.trim().is_empty() {
        return stdout.trim().to_owned();
    }
    format!(
        "exit status {}",
        status.map_or_else(|| "unknown".to_owned(), |status| status.to_string())
    )
}

/// Run one extraction command, upstream's `runExtractionCommand`: `None`
/// on success, the combined failure message otherwise.
fn run_extraction_command(command: &str, args: &[&str]) -> Option<String> {
    let result = spawn_process_sync(
        command,
        args,
        &SpawnSyncOptions {
            capture_output: true,
            capture_stderr: false,
            timeout_ms: None,
        },
    );
    if result.status == Some(0) {
        return None;
    }
    Some(format!(
        "{command}: {}",
        format_spawn_failure(result.status, "", &result.stdout)
    ))
}

/// Extract a `.tar.gz` archive, upstream's `extractTarGzArchive`.
///
/// # Errors
/// The extraction failure, upstream's throw.
fn extract_tar_gz_archive(
    archive_path: &str,
    extract_dir: &str,
    asset_name: &str,
) -> Result<(), String> {
    if let Some(failure) = run_extraction_command("tar", &["xzf", archive_path, "-C", extract_dir])
    {
        return Err(format!("Failed to extract {asset_name}: {failure}"));
    }
    Ok(())
}

/// Extract a `.zip` archive, upstream's `extractZipArchive`. Windows is out
/// of scope for this effort (map ticket "Decide the Rust stack"), so the
/// win32 arms are unreachable and the unix unzip-then-tar ladder carries.
///
/// # Errors
/// The extraction failure, upstream's throw.
fn extract_zip_archive(
    archive_path: &str,
    extract_dir: &str,
    asset_name: &str,
) -> Result<(), String> {
    let mut failures: Vec<String> = Vec::new();
    if let Some(failure) = run_extraction_command("unzip", &["-q", archive_path, "-d", extract_dir])
    {
        failures.push(failure);
        if let Some(failure) =
            run_extraction_command("tar", &["xf", archive_path, "-C", extract_dir])
        {
            failures.push(failure);
        } else {
            return Ok(());
        }
    } else {
        return Ok(());
    }
    Err(format!(
        "Failed to extract {asset_name}: {}",
        failures.join("; ")
    ))
}

/// Find the binary in the extracted tree, upstream's
/// `findBinaryRecursively`.
fn find_binary_recursively(root_dir: &std::path::Path, binary_file_name: &str) -> Option<PathBuf> {
    let mut stack = vec![root_dir.to_path_buf()];
    while let Some(current_dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&current_dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let full_path = entry.path();
            if entry.file_type().is_ok_and(|file_type| file_type.is_file())
                && entry.file_name() == binary_file_name
            {
                return Some(full_path);
            }
            if entry.file_type().is_ok_and(|file_type| file_type.is_dir()) {
                stack.push(full_path);
            }
        }
    }
    None
}

/// Download a file from URL, upstream's `downloadFile`.
///
/// # Errors
/// The HTTP failure, upstream's throw.
async fn download_file(
    client: &Arc<dyn HttpClient>,
    url: &str,
    dest: &std::path::Path,
) -> Result<(), String> {
    let response = fetch_with_retry(
        client,
        url,
        Vec::new(),
        CancellationToken::new(),
        FetchRetryOptions {
            timeout_ms: Some(DOWNLOAD_TIMEOUT_MS),
            ..FetchRetryOptions::default()
        },
    )
    .await
    .map_err(|error| error.to_string())?;

    if !(200..300).contains(&response.status) {
        return Err(format!(
            "Download failed with HTTP {}: {url}",
            response.status
        ));
    }

    // The body streams to the file, upstream's Readable.fromWeb pipeline.
    let mut file = std::fs::File::create(dest).map_err(|error| error.to_string())?;
    let mut body = response.body;
    loop {
        match body.next_chunk().await {
            Ok(Some(chunk)) => {
                std::io::Write::write_all(&mut file, &chunk).map_err(|error| error.to_string())?;
            }
            Ok(None) => break,
            Err(error) => return Err(error.to_string()),
        }
    }
    std::io::Write::flush(&mut file).map_err(|error| error.to_string())?;
    Ok(())
}

/// Download and install a tool, upstream's `downloadTool`.
///
/// # Errors
/// The resolution, download, extraction, and layout failures, upstream's
/// throws.
async fn download_tool(client: &Arc<dyn HttpClient>, tool: ManagedTool) -> Result<String, String> {
    let config = tool.config();
    let (platform, architecture) = platform_and_arch();

    // fd is pinned on darwin/x64, so skip the version lookup there.
    let version = if tool == ManagedTool::Fd && platform == "darwin" && architecture == "x64" {
        config
            .pinned_darwin_x64_version
            .unwrap_or_default()
            .to_owned()
    } else {
        get_latest_version(config.repo).await?
    };

    install_tool(client, tool, &version, &get_bin_dir()).await
}

/// The download-layout half of `downloadTool` (the private download wrapper
/// in this file) over an injected version and tools directory, the seam the
/// download boundary tests drive.
///
/// The production call reads the pinned-or-probed version and the real bin
/// dir; the seam shape matches [`get_latest_version_with`]'s for the version
/// probe.
///
/// # Errors
/// The download, extraction, and layout failures, upstream's throws.
pub async fn install_tool(
    client: &Arc<dyn HttpClient>,
    tool: ManagedTool,
    version: &str,
    tools_dir: &std::path::Path,
) -> Result<String, String> {
    let config = tool.config();
    let (platform, architecture) = platform_and_arch();

    // Get asset name for this platform
    let Some(asset_name) = tool.asset_name(version, platform, architecture) else {
        return Err(format!("Unsupported platform: {platform}/{architecture}"));
    };

    // Create the tools directory
    std::fs::create_dir_all(tools_dir).map_err(|error| error.to_string())?;

    let download_url = format!(
        "https://github.com/{}/releases/download/{}{version}/{asset_name}",
        config.repo, config.tag_prefix
    );
    let archive_path = tools_dir.join(&asset_name);
    let binary_path = tools_dir.join(config.binary_name);

    // Download
    download_file(client, &download_url, &archive_path).await?;

    // Extract into a unique temp directory. fd and rg downloads can run
    // concurrently during startup, so sharing a fixed directory causes
    // races.
    let extract_dir = tools_dir.join(format!(
        "extract_tmp_{}_{}_{}",
        config.binary_name,
        std::process::id(),
        crate::tools::random_bytes_hex(8)
    ));
    std::fs::create_dir_all(&extract_dir).map_err(|error| error.to_string())?;

    #[expect(
        clippy::case_sensitive_file_extension_comparisons,
        reason = "the asset names are canonical lowercase, upstream's endsWith"
    )]
    let extraction = if asset_name.ends_with(".tar.gz") {
        extract_tar_gz_archive(
            &archive_path.to_string_lossy(),
            &extract_dir.to_string_lossy(),
            &asset_name,
        )
    } else if asset_name.ends_with(".zip") {
        extract_zip_archive(
            &archive_path.to_string_lossy(),
            &extract_dir.to_string_lossy(),
            &asset_name,
        )
    } else {
        Err(format!("Unsupported archive format: {asset_name}"))
    };

    let extraction = extraction.and_then(|()| {
        // Find the binary in extracted files. Some archives contain files
        // directly at root, others nest under a versioned subdirectory.
        let extracted_dir = extract_dir.join(asset_name.replace(".tar.gz", "").replace(".zip", ""));
        let candidates = [
            extracted_dir.join(config.binary_name),
            extract_dir.join(config.binary_name),
        ];
        let extracted_binary = candidates
            .into_iter()
            .find(|candidate| candidate.exists())
            .or_else(|| find_binary_recursively(&extract_dir, config.binary_name));
        match extracted_binary {
            Some(extracted_binary) => {
                std::fs::rename(&extracted_binary, &binary_path)
                    .map_err(|error| error.to_string())?;
                Ok(())
            }
            None => Err(format!(
                "Binary not found in archive: expected {} under {}",
                config.binary_name,
                extract_dir.to_string_lossy()
            )),
        }
    });

    // Cleanup
    let _removed = std::fs::remove_file(&archive_path);
    let _removed = std::fs::remove_dir_all(&extract_dir);

    extraction?;
    // Make executable (Unix only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&binary_path, std::fs::Permissions::from_mode(0o755))
            .map_err(|error| error.to_string())?;
    }
    Ok(binary_path.to_string_lossy().into_owned())
}

/// A status report, upstream's `ToolStatus`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolStatus {
    /// A progress note, upstream's `"info"`.
    Info(String),
    /// A recoverable problem, upstream's `"warning"`.
    Warning(String),
}

/// Ensure a tool is available, downloading if necessary, upstream's
/// `ensureTool`. Reports progress through `on_status`; status messages are
/// otherwise silent.
///
/// The `client` seam is the process default at the call sites; the
/// `env` seam reads `PI_OFFLINE`.
pub async fn ensure_tool(
    client: &Arc<dyn HttpClient>,
    tool: ManagedTool,
    on_status: Option<&dyn Fn(&ToolStatus)>,
) -> Option<String> {
    ensure_tool_with(
        client,
        tool,
        on_status,
        &crate::config::default_env_lookup(),
    )
    .await
}

/// The ensure over an injected environment, the belt's `_with` seam: the
/// plain getter reads the real environment.
pub async fn ensure_tool_with(
    client: &Arc<dyn HttpClient>,
    tool: ManagedTool,
    on_status: Option<&dyn Fn(&ToolStatus)>,
    env: &crate::config::EnvLookup,
) -> Option<String> {
    if let Some(existing_path) = get_tool_path(tool) {
        return Some(existing_path);
    }

    let config = tool.config();

    if is_offline_mode_enabled_with(env) {
        if let Some(on_status) = on_status {
            on_status(&ToolStatus::Warning(format!(
                "{} not found. Offline mode enabled, skipping download.",
                config.name
            )));
        }
        return None;
    }

    // On Android/Termux, Linux binaries don't work due to Bionic libc
    // incompatibility. Users must install via pkg.
    if cfg!(target_os = "android") || env("TERMUX_VERSION").is_some() {
        let package_name = match tool {
            ManagedTool::Fd => "fd",
            ManagedTool::Rg => "ripgrep",
        };
        if let Some(on_status) = on_status {
            on_status(&ToolStatus::Warning(format!(
                "{} not found. Install with: pkg install {package_name}",
                config.name
            )));
        }
        return None;
    }

    // Tool not found - download it
    if let Some(on_status) = on_status {
        on_status(&ToolStatus::Info(format!(
            "{} not found. Downloading...",
            config.name
        )));
    }

    match download_tool(client, tool).await {
        Ok(path) => {
            if let Some(on_status) = on_status {
                on_status(&ToolStatus::Info(format!(
                    "{} installed to {path}",
                    config.name
                )));
            }
            Some(path)
        }
        Err(error) => {
            // The cause chain: fetch failures surface as one composed
            // message here — the Rust error type carries the detail the
            // upstream loop walked `Error.cause` for, with the same
            // dedupe depth cap's shape.
            let messages = [error];
            if let Some(on_status) = on_status {
                on_status(&ToolStatus::Warning(format!(
                    "Failed to download {}: {}",
                    config.name,
                    messages.join(": ")
                )));
            }
            None
        }
    }
}

#[cfg(test)]
mod tests;
