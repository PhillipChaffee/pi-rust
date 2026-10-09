//! The package command CLI, upstream's `src/package-manager-cli.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements this module records:
//!
//! - The npm/pnpm/yarn/bun self-update paths drop with the npm channel
//!   (ADR 0007): the running surface detects the cargo install (config's
//!   redesigned [`crate::config::InstallMethod`]), and the
//!   installer-managed layout ports 1:1 with the release download restated
//!   to the tarball channel's artifact fetch and unpack —
//!   `${installerApiBase}/<version>/download` answers a tarball whose tree
//!   is the release directory, smoke-tested through `<release>/bin/pi
//!   --version` before activation.
//! - `process.exitCode` restates to the returned [`CommandExit`]: the
//!   caller (the CLI-grammar slice) owns the process exit.
//! - The chalk color vocabulary prints plain: the theme system slice owns
//!   color output, and the update note renders through pi-tui's `Markdown`
//!   with an identity theme where upstream styled per-element.
//! - The trust resolution reads the saved project trust and the
//!   `--approve`/`--no-approve` override; the interactive prompt and the
//!   `project_trust` extension handlers ride the interactive slices.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tokio_util::sync::CancellationToken;

use pi_tui::components::{ColorFn, Markdown as MarkdownComponent, MarkdownTheme};

use crate::config::{
    APP_NAME, EnvLookup, InstallMethod, PACKAGE_NAME, SelfUpdateCommand, SelfUpdatePackageTarget,
    VERSION, current_exe_path, detect_install_method_with, get_agent_dir_with,
    get_package_dir_with, get_self_update_command_with,
    get_self_update_unavailable_instruction_with,
};
use crate::file_lock::{LockError, acquire_once, lock_dir_for};
use crate::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
use crate::package_manager::{
    CommandRunner, ConfiguredPackage, DefaultPackageManager, MAX_TARBALL_BYTES,
    PackageManagerError, PackageManagerOptions, ResolvedPaths, SourceScope,
};
use crate::settings_manager::{
    FileSettingsStorage, SettingsManager, SettingsManagerCreateOptions, SettingsStorage,
};
use crate::trust_manager::ProjectTrustStore;
use crate::utils::paths::{canonicalize_path, get_cwd_relative_path};
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::version_check::{
    VersionCheckOptions, format_version_check_error, get_latest_pi_release_with,
    is_newer_package_version,
};

/// The package commands, upstream's `PackageCommand`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageCommand {
    /// `pi install`, upstream's `"install"`.
    Install,
    /// `pi remove` (alias `pi uninstall`), upstream's `"remove"`.
    Remove,
    /// `pi update`, upstream's `"update"`.
    Update,
    /// `pi list`, upstream's `"list"`.
    List,
}

impl PackageCommand {
    /// The spelling the grammar and help text use, upstream's union member.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Install => "install",
            Self::Remove => "remove",
            Self::Update => "update",
            Self::List => "list",
        }
    }
}

/// What `pi update` targets, upstream's `UpdateTarget`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpdateTarget {
    /// Both pi and the packages, upstream's `{ type: "all" }`.
    All,
    /// Pi only, upstream's `{ type: "self" }`.
    SelfUpdate,
    /// The packages, one by source when given, upstream's
    /// `{ type: "extensions", source? }`.
    Extensions {
        /// The one source to update, upstream's `source?`.
        source: Option<String>,
    },
    /// Model catalogs only, upstream's `{ type: "models" }`.
    Models,
}

/// The parsed command, upstream's `PackageCommandOptions`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the struct mirrors upstream's handlePackageCommand options object field-for-field; bundling the flags would break the 1:1 port"
)]
pub struct PackageCommandOptions {
    /// The command, upstream's `command`.
    pub command: PackageCommand,
    /// The positional source, upstream's `source?`.
    pub source: Option<String>,
    /// The update target, upstream's `updateTarget?`.
    pub update_target: Option<UpdateTarget>,
    /// Whether the default self-update note applies, upstream's
    /// `showExtensionsSkippedNote`.
    pub show_extensions_skipped_note: bool,
    /// The `-l` local flag, upstream's `local`.
    pub local: bool,
    /// The `--force` flag, upstream's `force`.
    pub force: bool,
    /// The `--approve`/`--no-approve` override, upstream's
    /// `projectTrustOverride?`.
    pub project_trust_override: Option<bool>,
    /// The help request, upstream's `help`.
    pub help: bool,
    /// The first unknown option, upstream's `invalidOption?`.
    pub invalid_option: Option<String>,
    /// The first unexpected argument, upstream's `invalidArgument?`.
    pub invalid_argument: Option<String>,
    /// The first option missing its value, upstream's `missingOptionValue?`.
    pub missing_option_value: Option<String>,
    /// The first conflicting-options message, upstream's
    /// `conflictingOptions?`.
    pub conflicting_options: Option<String>,
}

/// The command's exit code, upstream's `process.exitCode` writes: `None`
/// keeps the success code, `Some(n)` is the failure the dispatcher applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandExit(pub Option<i32>);

impl CommandExit {
    /// The success state, upstream's `exitCode` left undefined.
    pub const OK: Self = Self(None);
    /// The failure state, upstream's `exitCode = 1`.
    pub const FAIL: Self = Self(Some(1));
}

/// The installer API base override, upstream's `PI_INSTALLER_API_BASE`.
const INSTALLER_API_BASE_ENV: &str = "PI_INSTALLER_API_BASE";
/// The default release feed, upstream's `DEFAULT_INSTALLER_API_BASE`.
const DEFAULT_INSTALLER_API_BASE: &str = "https://pi.dev/api/installer/releases";
/// The managed-layout marker, upstream's `MANAGED_INSTALL_MARKER`.
const MANAGED_INSTALL_MARKER: &str = "managed-install.json";
/// The managed-layout env override, upstream's `PI_MANAGED_INSTALL_ROOT`.
const MANAGED_INSTALL_ROOT_ENV: &str = "PI_MANAGED_INSTALL_ROOT";

/// The active managed install root, upstream's
/// `getActiveManagedInstallRoot`: `PI_MANAGED_INSTALL_ROOT` set, the
/// running package dir inside its `releases` tree, and a valid
/// `managed-install.json` marker.
///
/// # Errors
/// A set root whose marker is missing or invalid, upstream's thrown
/// `Managed install marker is missing or invalid`.
fn get_active_managed_install_root(env: &EnvLookup) -> Result<Option<String>, PackageManagerError> {
    let Some(configured_root) = env(MANAGED_INSTALL_ROOT_ENV)
        .map(|root| root.trim().to_string())
        .filter(|root| !root.is_empty())
    else {
        return Ok(None);
    };

    let managed_root = PathBuf::from(&configured_root);
    let releases_dir = canonicalize_path(&managed_root.join("releases").to_string_lossy());
    // The launcher environment is inherited by child processes. Do not
    // classify a source checkout or another pi installation launched from
    // managed pi as managed.
    let package_dir = get_package_dir_with(env);
    if get_cwd_relative_path(
        &canonicalize_path(&package_dir.to_string_lossy()),
        &releases_dir,
    )
    .is_none()
    {
        return Ok(None);
    }

    let marker_path = managed_root.join(MANAGED_INSTALL_MARKER);
    let read_marker = || -> Result<serde_json::Value, PackageManagerError> {
        let content = std::fs::read_to_string(&marker_path).map_err(|_| {
            PackageManagerError(format!(
                "Managed install marker is missing or invalid: {}",
                marker_path.to_string_lossy()
            ))
        })?;
        serde_json::from_str(&content).map_err(|_| {
            PackageManagerError(format!(
                "Managed install marker is missing or invalid: {}",
                marker_path.to_string_lossy()
            ))
        })
    };
    let marker = read_marker()?;
    if marker.get("kind").and_then(serde_json::Value::as_str) != Some("pi-managed-install")
        || marker
            .get("schemaVersion")
            .and_then(serde_json::Value::as_i64)
            != Some(1)
        || marker.get("layout").and_then(serde_json::Value::as_str) != Some("releases-v1")
    {
        return Err(PackageManagerError(format!(
            "Managed install marker is missing or invalid: {}",
            marker_path.to_string_lossy()
        )));
    }

    Ok(Some(managed_root.to_string_lossy().into_owned()))
}

/// Fetch the installer release artifact, upstream's `fetchInstallerArtifact`
/// over the [`HttpClient`](pi_ai::http::HttpClient) seam.
///
/// # Errors
/// The transport failure or a non-ok status.
async fn fetch_installer_artifact(
    client: &Arc<dyn pi_ai::http::HttpClient>,
    url: &str,
    label: &str,
) -> Result<Vec<u8>, PackageManagerError> {
    let request = pi_ai::http::client::HttpRequest {
        method: pi_ai::http::client::HttpMethod::Get,
        url: url.to_string(),
        headers: vec![("User-Agent".to_string(), get_pi_user_agent(VERSION))],
        body: None,
        timeout_ms: None,
        signal: CancellationToken::new(),
    };
    let response = client
        .execute(request)
        .await
        .map_err(|error| PackageManagerError(error.to_string()))?;
    if response.status < 200 || response.status >= 300 {
        return Err(PackageManagerError(format!(
            "Could not download managed installer {label} from {url}: HTTP {}",
            response.status
        )));
    }
    let mut body = response.body;
    let mut bytes = Vec::new();
    while let Some(chunk) = body
        .next_chunk()
        .await
        .map_err(|error| PackageManagerError(error.to_string()))?
    {
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

/// Run the staged release's `--version` smoke test, upstream's
/// `verifyManagedRelease`.
///
/// # Errors
/// The spawn failure, a non-zero exit, or a version mismatch.
fn verify_managed_release(
    release_dir: &Path,
    expected_version: &str,
) -> Result<(), PackageManagerError> {
    let bin_path = release_dir.join("bin").join(APP_NAME);
    let arg_refs = ["--version"];
    let result = crate::utils::child_process::spawn_process_sync(
        &bin_path.to_string_lossy(),
        &arg_refs,
        &crate::utils::child_process::SpawnSyncOptions {
            capture_output: true,
            capture_stderr: true,
            timeout_ms: None,
        },
    );
    if result.status != Some(0) {
        let reason = if result.stderr.trim().is_empty() {
            result.status.map_or_else(
                || "unknown exit status".to_string(),
                |code| format!("exit code {code}"),
            )
        } else {
            result.stderr.trim().to_string()
        };
        return Err(PackageManagerError(format!(
            "Could not verify managed pi {expected_version}: {reason}"
        )));
    }
    let installed_version = result.stdout.trim();
    if installed_version != expected_version {
        return Err(PackageManagerError(format!(
            "Managed pi smoke test returned version {installed_version}; expected {expected_version}."
        )));
    }
    Ok(())
}

/// Point `current-version` at the release, upstream's `activateManagedRelease`:
/// the version writes to a temp file and renames atomically.
///
/// # Errors
/// The write or rename failure.
fn activate_managed_release(managed_root: &Path, version: &str) -> Result<(), PackageManagerError> {
    let current_path = managed_root.join("current-version");
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_millis())
        .unwrap_or_default();
    let temporary_path = managed_root.join(format!(
        "current-version.tmp.{}-{nanos}",
        std::process::id()
    ));
    let write = std::fs::write(&temporary_path, format!("{version}\n"));
    if let Err(error) = write {
        return Err(PackageManagerError(error.to_string()));
    }
    match std::fs::rename(&temporary_path, &current_path) {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = std::fs::remove_file(&temporary_path);
            Err(PackageManagerError(error.to_string()))
        }
    }
}

/// Sweep stale staging dirs, upstream's `cleanupManagedStaging`.
fn cleanup_managed_staging(managed_root: &Path) {
    let staging_root = managed_root.join("staging");
    let Ok(entries) = std::fs::read_dir(&staging_root) else {
        // The staging directory does not exist yet or is not writable.
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("update-") {
            let _ = std::fs::remove_dir_all(entry.path());
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Clear abandoned staging under the update lock, upstream's
/// `cleanupManagedInstall`: every failure swallows.
pub fn cleanup_managed_install() {
    cleanup_managed_install_with(&crate::config::default_env_lookup());
}

/// [`cleanup_managed_install`] over an injected environment, the test seam.
pub fn cleanup_managed_install_with(env: &EnvLookup) {
    let Ok(Some(managed_root)) = get_active_managed_install_root(env) else {
        return;
    };

    let lock_dir = lock_dir_for(&Path::new(&managed_root).join("update").to_string_lossy());
    let Ok(release_lock) = acquire_once(&lock_dir, Some(30_000)) else {
        // A live update owns the staging directory, or cleanup is
        // unavailable.
        return;
    };
    cleanup_managed_staging(Path::new(&managed_root));
    let _ = release_lock.release();
}

/// Update the managed installation to a version, upstream's
/// `runManagedSelfUpdate` with the npm-ci half restated to the tarball
/// channel: the release tarball downloads, unpacks into staging under the
/// installer-safety rules, smoke-tests, renames into the immutable release
/// directory, and activates.
///
/// # Errors
/// The version check, the lock's contention
/// (`Another managed pi update is already running.`), the download, the
/// unpack, the smoke test, or the activation.
async fn run_managed_self_update(
    client: &Arc<dyn pi_ai::http::HttpClient>,
    managed_root: &str,
    version: &str,
    env: &EnvLookup,
) -> Result<(), PackageManagerError> {
    if semver::Version::parse(version).is_err() {
        return Err(PackageManagerError(format!(
            "Invalid managed release version: {version}"
        )));
    }

    let managed_root_path = Path::new(managed_root);
    let lock_dir = lock_dir_for(&managed_root_path.join("update").to_string_lossy());
    let release_lock = acquire_once(&lock_dir, Some(30_000)).map_err(|error| match error {
        LockError::Locked => {
            PackageManagerError("Another managed pi update is already running.".to_string())
        }
        other => PackageManagerError(other.to_string()),
    })?;

    cleanup_managed_staging(managed_root_path);
    let installer_api_base =
        env_trimmed_default(env, INSTALLER_API_BASE_ENV, DEFAULT_INSTALLER_API_BASE)
            .trim_end_matches('/')
            .to_string();
    let release_url = format!("{installer_api_base}/{}", url_encode_component(version));
    let staging_root = managed_root_path.join("staging");
    let releases_root = managed_root_path.join("releases");
    std::fs::create_dir_all(&releases_root)
        .map_err(|error| PackageManagerError(error.to_string()))?;
    let release_dir = releases_root.join(version);
    if release_dir.exists() {
        verify_managed_release(&release_dir, version)?;
        activate_managed_release(managed_root_path, version)?;
        return Ok(());
    }

    std::fs::create_dir_all(&staging_root)
        .map_err(|error| PackageManagerError(error.to_string()))?;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_nanos())
        .unwrap_or_default();
    let stage_dir = staging_root.join(format!("update-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&stage_dir).map_err(|error| PackageManagerError(error.to_string()))?;

    let result: Result<(), PackageManagerError> = async {
        let artifact = fetch_installer_artifact(
            client,
            &format!("{release_url}/download"),
            "release artifact",
        )
        .await?;
        if artifact.len() as u64 > MAX_TARBALL_BYTES {
            return Err(PackageManagerError(format!(
                "Managed release artifact exceeds the {MAX_TARBALL_BYTES}-byte cap"
            )));
        }
        crate::package_manager::unpack_tarball(&artifact, &stage_dir)?;
        verify_managed_release(&stage_dir, version)?;
        std::fs::rename(&stage_dir, &release_dir)
            .map_err(|error| PackageManagerError(error.to_string()))?;
        activate_managed_release(managed_root_path, version)?;
        Ok(())
    }
    .await;
    // The stage survives only while the update runs; the rename moved it on
    // success, the removal clears it on failure, upstream's finally.
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&stage_dir);
    }
    // The lock guard must outlive the staging rename and activation; the
    // explicit drop marks the critical section's end.
    drop(release_lock);
    result
}

/// The upstream `encodeURIComponent` vocabulary for the release URL.
fn url_encode_component(value: &str) -> String {
    use std::fmt::Write as _;
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(byte as char);
        } else {
            let _ = write!(encoded, "%{byte:02X}");
        }
    }
    encoded
}

fn env_trimmed_default(env: &EnvLookup, name: &str, default: &str) -> String {
    env(name)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// The identity theme the note renders with, the restated
/// `SELF_UPDATE_NOTE_MARKDOWN_THEME` (the chalk styling rides the theme
/// system slice).
fn plain_markdown_theme() -> MarkdownTheme {
    let identity: ColorFn = Arc::new(ToString::to_string);
    MarkdownTheme {
        heading: Arc::clone(&identity),
        link: Arc::clone(&identity),
        link_url: Arc::clone(&identity),
        code: Arc::clone(&identity),
        code_block: Arc::clone(&identity),
        code_block_border: Arc::clone(&identity),
        quote: Arc::clone(&identity),
        quote_border: Arc::clone(&identity),
        hr: Arc::clone(&identity),
        list_bullet: Arc::clone(&identity),
        bold: Arc::clone(&identity),
        italic: Arc::clone(&identity),
        strikethrough: Arc::clone(&identity),
        underline: identity,
        highlight_code: None,
        code_block_indent: None,
    }
}

/// Print the release note, upstream's `printSelfUpdateNote`: the markdown
/// renders at the terminal width, and a render failure falls back to the
/// raw text.
fn print_self_update_note(out: &mut dyn Write, note: &str) {
    let trimmed_note = note.trim();
    if trimmed_note.is_empty() {
        return;
    }

    let _ = writeln!(out);
    let _ = writeln!(out, "Update note");
    // The terminal-width probe rides the TUI slice's stdout belt; the
    // 80-column fallback is upstream's own.
    let width = 80_usize;
    let theme = plain_markdown_theme();
    let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        use pi_tui::tui::Component;
        MarkdownComponent::new(trimmed_note, 0, 0, theme)
            .render(width)
            .iter()
            .map(|line| line.trim_end().to_string())
            .collect::<Vec<_>>()
            .join("\n")
    }))
    .unwrap_or_else(|_| trimmed_note.to_string());
    let _ = writeln!(out, "{rendered}");
    let _ = writeln!(out);
}

/// The self-update decision, upstream's `SelfUpdatePlan`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfUpdatePlan {
    /// The package the release publishes under, upstream's `packageName`.
    pub package_name: String,
    /// The spec the update installs, upstream's `installSpec`.
    pub install_spec: String,
    /// The latest version, upstream's `version`.
    pub version: String,
    /// Whether the update runs, upstream's `shouldRun`.
    pub should_run: bool,
    /// The release note, upstream's `note?`.
    pub note: Option<String>,
}

/// Plan the self-update, upstream's `getSelfUpdatePlan`.
///
/// # Errors
/// The version-check failure with the upstream message shapes.
pub async fn get_self_update_plan(
    client: &Arc<dyn pi_ai::http::HttpClient>,
    force: bool,
    env: &EnvLookup,
    out: &mut dyn Write,
) -> Result<SelfUpdatePlan, PackageManagerError> {
    let latest_release = get_latest_pi_release_with(
        client,
        CancellationToken::new(),
        VERSION,
        VersionCheckOptions {
            retry: true,
            ..VersionCheckOptions::default()
        },
        env,
    )
    .await
    .map_err(|error| {
        PackageManagerError(format!(
            "Could not determine latest {APP_NAME} version: {}",
            format_version_check_error(&error)
        ))
    })?;
    let Some(latest_release) = latest_release else {
        return Err(PackageManagerError(format!(
            "Could not determine latest {APP_NAME} version."
        )));
    };

    let package_name = latest_release
        .package_name
        .unwrap_or_else(|| PACKAGE_NAME.to_string());
    let install_spec = format!("{package_name}@{}", latest_release.version);
    if force
        || package_name != PACKAGE_NAME
        || is_newer_package_version(&latest_release.version, VERSION)
    {
        return Ok(SelfUpdatePlan {
            package_name,
            install_spec,
            version: latest_release.version,
            note: latest_release.note,
            should_run: true,
        });
    }

    let _ = writeln!(out, "{APP_NAME} is already up to date (v{VERSION})");
    Ok(SelfUpdatePlan {
        package_name,
        install_spec,
        version: latest_release.version,
        should_run: false,
        note: None,
    })
}

/// Run the update steps, upstream's `runSelfUpdate`: each step spawns with
/// inherited stdio and a failure surfaces with the step's display.
///
/// # Errors
/// A spawn failure, a signal, or a non-zero exit.
async fn run_self_update(
    command: &SelfUpdateCommand,
    out: &mut dyn Write,
) -> Result<(), PackageManagerError> {
    let _ = writeln!(out, "Updating {APP_NAME} with {}...", command.display);
    let steps: Vec<crate::config::SelfUpdateCommandStep> =
        command.steps.clone().unwrap_or_else(|| {
            vec![crate::config::SelfUpdateCommandStep {
                command: command.command.clone(),
                args: command.args.clone(),
                display: command.display.clone(),
            }]
        });
    for step in steps {
        let mut cmd = tokio::process::Command::new(&step.command);
        cmd.args(&step.args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit());
        let mut child = cmd
            .spawn()
            .map_err(|error| PackageManagerError(error.to_string()))?;
        let status = child
            .wait()
            .await
            .map_err(|error| PackageManagerError(error.to_string()))?;
        match status.code() {
            Some(0) => {}
            Some(code) => {
                return Err(PackageManagerError(format!(
                    "{} exited with code {code}",
                    step.display
                )));
            }
            None => {
                return Err(PackageManagerError(format!(
                    "{} terminated by signal {}",
                    step.display,
                    crate::package_manager::signal_name(status)
                )));
            }
        }
    }
    Ok(())
}

fn print_self_update_unavailable(
    err: &mut dyn Write,
    update_package_target: &SelfUpdatePackageTarget,
    env: &EnvLookup,
) {
    let _ = writeln!(
        err,
        "error: {APP_NAME} cannot self-update this installation."
    );
    let _ = writeln!(
        err,
        "{}",
        get_self_update_unavailable_instruction_with(
            PACKAGE_NAME,
            update_package_target,
            &current_exe_path(),
            env
        )
    );
    let entrypoint = std::env::args().nth(1);
    if let Some(entrypoint) = entrypoint {
        let _ = writeln!(err);
        let _ = writeln!(err, "Location of {APP_NAME} executable: {entrypoint}");
    }
}

fn print_self_update_fallback(err: &mut dyn Write, command: &SelfUpdateCommand) {
    let _ = writeln!(
        err,
        "If this keeps failing, run this command yourself: {}",
        command.display
    );
}

/// Parse a package command out of the arguments, upstream's
/// `parsePackageCommand`: `None` when the arguments open with some other
/// command.
#[expect(
    clippy::too_many_lines,
    reason = "parse_package_command restates upstream's parsePackageCommand grammar dispatch one-to-one; splitting it would scatter the option-precedence rules"
)]
#[must_use]
pub fn parse_package_command(args: &[String]) -> Option<PackageCommandOptions> {
    let raw_command = args.first()?;
    let command = match raw_command.as_str() {
        "uninstall" | "remove" => PackageCommand::Remove,
        "install" => PackageCommand::Install,
        "update" => PackageCommand::Update,
        "list" => PackageCommand::List,
        _ => return None,
    };

    let rest = &args[1..];
    let mut local = false;
    let mut force = false;
    let mut project_trust_override: Option<bool> = None;
    let mut help = false;
    let mut invalid_option: Option<String> = None;
    let mut invalid_argument: Option<String> = None;
    let mut missing_option_value: Option<String> = None;
    let mut conflicting_options: Option<String> = None;
    let mut source: Option<String> = None;
    let mut self_flag = false;
    let mut extensions_flag = false;
    let mut models_flag = false;
    let mut all_flag = false;
    let mut extension_flag_source: Option<String> = None;

    let mut index = 0;
    while index < rest.len() {
        let arg = rest[index].as_str();
        if arg == "-h" || arg == "--help" {
            help = true;
        } else if arg == "-l" || arg == "--local" {
            if command == PackageCommand::Install || command == PackageCommand::Remove {
                local = true;
            } else {
                invalid_option = invalid_option.or_else(|| Some(arg.to_string()));
            }
        } else if arg == "--self" {
            if command == PackageCommand::Update {
                self_flag = true;
            } else {
                invalid_option = invalid_option.or_else(|| Some(arg.to_string()));
            }
        } else if arg == "--extensions" {
            if command == PackageCommand::Update {
                extensions_flag = true;
            } else {
                invalid_option = invalid_option.or_else(|| Some(arg.to_string()));
            }
        } else if arg == "--models" {
            if command == PackageCommand::Update {
                models_flag = true;
            } else {
                invalid_option = invalid_option.or_else(|| Some(arg.to_string()));
            }
        } else if arg == "--all" {
            if command == PackageCommand::Update {
                all_flag = true;
            } else {
                invalid_option = invalid_option.or_else(|| Some(arg.to_string()));
            }
        } else if arg == "--approve" || arg == "-a" {
            project_trust_override = Some(true);
        } else if arg == "--no-approve" || arg == "-na" {
            project_trust_override = Some(false);
        } else if arg == "--force" {
            if command == PackageCommand::Update {
                force = true;
            } else {
                invalid_option = invalid_option.or_else(|| Some(arg.to_string()));
            }
        } else if arg == "--extension" {
            if command == PackageCommand::Update {
                let value = rest.get(index + 1).filter(|value| !value.starts_with('-'));
                match (value, extension_flag_source.is_some()) {
                    (None, _) => {
                        missing_option_value =
                            missing_option_value.or_else(|| Some(arg.to_string()));
                    }
                    (Some(_), true) => {
                        conflicting_options = conflicting_options
                            .or_else(|| Some("--extension can only be provided once".to_string()));
                        index += 1;
                    }
                    (Some(value), false) => {
                        extension_flag_source = Some(value.clone());
                        index += 1;
                    }
                }
            } else {
                invalid_option = invalid_option.or_else(|| Some(arg.to_string()));
            }
        } else if arg.starts_with('-') {
            invalid_option = invalid_option.or_else(|| Some(arg.to_string()));
        } else if source.is_none() {
            source = Some(arg.to_string());
        } else {
            invalid_argument = invalid_argument.or_else(|| Some(arg.to_string()));
        }
        index += 1;
    }

    let mut update_target: Option<UpdateTarget> = None;
    let mut show_extensions_skipped_note = false;
    if command == PackageCommand::Update {
        if all_flag
            && (self_flag || extensions_flag || models_flag || extension_flag_source.is_some())
        {
            conflicting_options = conflicting_options.or_else(|| {
                Some(
                    "--all cannot be combined with --self, --extensions, --models, or --extension"
                        .to_string(),
                )
            });
        }
        if all_flag && source.is_some() {
            conflicting_options = conflicting_options
                .or_else(|| Some("--all cannot be combined with a positional source".to_string()));
        }

        if models_flag {
            if self_flag || extensions_flag || all_flag || extension_flag_source.is_some() {
                conflicting_options = conflicting_options
                    .or_else(|| Some("--models cannot be combined with --self, --extensions, --all, or --extension".to_string()));
            }
            if source.is_some() {
                conflicting_options = conflicting_options.or_else(|| {
                    Some("--models cannot be combined with a positional source".to_string())
                });
            }
            update_target = Some(UpdateTarget::Models);
        } else if let Some(extension_source) = extension_flag_source {
            if self_flag || extensions_flag || all_flag {
                conflicting_options = conflicting_options.or_else(|| {
                    Some(
                        "--extension cannot be combined with --self, --extensions, or --all"
                            .to_string(),
                    )
                });
            }
            if source.is_some() {
                conflicting_options = conflicting_options.or_else(|| {
                    Some("--extension cannot be combined with a positional source".to_string())
                });
            }
            update_target = Some(UpdateTarget::Extensions {
                source: Some(extension_source),
            });
        } else if let Some(source) = source.clone() {
            let source_is_self = source == "self" || source == "pi";
            if source_is_self {
                update_target = Some(if extensions_flag {
                    UpdateTarget::All
                } else {
                    UpdateTarget::SelfUpdate
                });
            } else {
                if extensions_flag || self_flag || all_flag {
                    conflicting_options = conflicting_options
                        .or_else(|| Some("positional update targets cannot be combined with --self, --extensions, or --all".to_string()));
                }
                update_target = Some(UpdateTarget::Extensions {
                    source: Some(source),
                });
            }
        } else if all_flag || (self_flag && extensions_flag) {
            update_target = Some(UpdateTarget::All);
        } else if self_flag {
            update_target = Some(UpdateTarget::SelfUpdate);
        } else if extensions_flag {
            update_target = Some(UpdateTarget::Extensions { source: None });
        } else {
            update_target = Some(UpdateTarget::SelfUpdate);
            show_extensions_skipped_note = true;
        }
    }

    Some(PackageCommandOptions {
        command,
        source,
        update_target,
        show_extensions_skipped_note,
        local,
        force,
        project_trust_override,
        help,
        invalid_option,
        invalid_argument,
        missing_option_value,
        conflicting_options,
    })
}

const fn update_target_includes_self(target: &UpdateTarget) -> bool {
    matches!(target, UpdateTarget::All | UpdateTarget::SelfUpdate)
}

const fn update_target_includes_extensions(target: &UpdateTarget) -> bool {
    matches!(target, UpdateTarget::All | UpdateTarget::Extensions { .. })
}

/// Refresh the model catalogs, upstream's `refreshModelCatalogs`.
///
/// # Errors
/// The runtime's create failure, the refresh abort, or the per-provider
/// error surface.
async fn refresh_model_catalogs(
    agent_dir: &Path,
    out: &mut dyn Write,
) -> Result<(), PackageManagerError> {
    let token = CancellationToken::new();
    let timeout_token = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(15)).await;
        timeout_token.cancel();
    });
    let model_runtime = ModelRuntime::create(CreateModelRuntimeOptions {
        auth_path: Some(agent_dir.join("auth.json").to_string_lossy().into_owned()),
        models_path: Some(Some(
            agent_dir.join("models.json").to_string_lossy().into_owned(),
        )),
        allow_model_network: false,
        signal: Some(token.clone()),
        ..CreateModelRuntimeOptions::default()
    })
    .await
    .map_err(|error| PackageManagerError(error.to_string()))?;
    let result = model_runtime
        .refresh(pi_ai::models::ModelsRefreshOptions {
            allow_network: Some(true),
            force: Some(true),
            providers: None,
            signal: Some(token.clone()),
        })
        .await;
    if result.aborted {
        return Err(PackageManagerError(
            "Model catalog refresh timed out.".to_string(),
        ));
    }
    if !result.errors.is_empty() {
        let details = result
            .errors
            .iter()
            .map(|(provider, error)| format!("{provider}: {error}"))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(PackageManagerError(format!(
            "Could not refresh model catalogs: {details}"
        )));
    }
    let _ = writeln!(out, "Model catalogs refreshed");
    Ok(())
}

/// The runtime inputs the command handlers share, upstream's
/// `createCommandSettingsManager` plus the manager handles.
pub struct PackageCommandRuntime<S: SettingsStorage> {
    /// The settings manager the commands share, upstream's
    /// `settingsManager`.
    pub settings: Arc<Mutex<SettingsManager<S>>>,
    /// The agent dir, upstream's `getAgentDir()`.
    pub agent_dir: PathBuf,
    /// The cwd, upstream's `process.cwd()`.
    pub cwd: PathBuf,
    /// The environment factory, upstream's `process.env` reads: the lookup
    /// itself is a boxed closure, so sharing means rebuilding the box
    /// around the same data.
    pub env: EnvLookupProvider,
    /// The child-process runner the package manager rides; the default
    /// spawns real processes.
    pub command_runner: Option<Arc<dyn CommandRunner>>,
}

/// The shareable environment handle, the factory the runtimes carry.
pub type EnvLookupProvider = Arc<dyn Fn() -> EnvLookup + Send + Sync>;

/// The project-trust state a command runs with, upstream's
/// `resolveProjectTrusted` at its command-mode slice: the override when
/// given, else the saved trust decision, else the settings' default —
/// except `update`, which reads saved trust only (upstream's
/// `useSavedProjectTrustOnly` branch). The interactive prompt and the
/// `project_trust` extension handlers ride the interactive slices; an
/// unresolved trust reads untrusted.
fn resolve_saved_project_trust<S: SettingsStorage>(
    settings: &SettingsManager<S>,
    cwd: &str,
    agent_dir: &Path,
    project_trust_override: Option<bool>,
    use_saved_only: bool,
) -> bool {
    if let Some(trusted) = project_trust_override {
        return trusted;
    }
    if !crate::trust_manager::has_trust_requiring_project_resources(cwd) {
        // A project carrying nothing trust-requiring is trusted outright,
        // upstream's early return.
        return true;
    }
    let trust_store = ProjectTrustStore::new(&agent_dir.to_string_lossy());
    match trust_store.get(cwd).ok().flatten() {
        Some(trusted) => trusted,
        None if use_saved_only => false,
        None => match settings.get_default_project_trust() {
            crate::settings_manager::DefaultProjectTrust::Always => true,
            crate::settings_manager::DefaultProjectTrust::Ask
            | crate::settings_manager::DefaultProjectTrust::Never => false,
        },
    }
}

/// Handle a package command, upstream's `handlePackageCommand`.
///
/// `Ok((false, _))` when the arguments are not a package command,
/// `Ok((true, exit))` when consumed, with the exit code riding
/// [`CommandExit`].
///
/// # Errors
/// The handler's failures carry their message; the exit code is the
/// failure's `1`, upstream's `process.exitCode = 1` writes.
#[expect(
    clippy::too_many_lines,
    reason = "handle_package_command restates upstream's handlePackageCommand command dispatch one-to-one; splitting it would scatter the per-command flow"
)]
pub async fn handle_package_command<S: SettingsStorage + 'static>(
    args: &[String],
    runtime: PackageCommandRuntime<S>,
    client: &Arc<dyn pi_ai::http::HttpClient>,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<(bool, CommandExit), PackageManagerError> {
    let Some(options) = parse_package_command(args) else {
        return Ok((false, CommandExit::OK));
    };

    if options.help {
        print_package_command_help(options.command, out);
        return Ok((true, CommandExit::OK));
    }

    if let Some(invalid_option) = &options.invalid_option {
        let _ = writeln!(
            err,
            "Unknown option {invalid_option} for \"{}\".",
            options.command.as_str()
        );
        let _ = writeln!(
            err,
            "Use \"{APP_NAME} --help\" or \"{}\".",
            get_package_command_usage(options.command)
        );
        return Ok((true, CommandExit::FAIL));
    }

    if let Some(missing_option_value) = &options.missing_option_value {
        let _ = writeln!(err, "Missing value for {missing_option_value}.");
        let _ = writeln!(err, "Usage: {}", get_package_command_usage(options.command));
        return Ok((true, CommandExit::FAIL));
    }

    if let Some(invalid_argument) = &options.invalid_argument {
        let _ = writeln!(err, "Unexpected argument {invalid_argument}.");
        let _ = writeln!(err, "Usage: {}", get_package_command_usage(options.command));
        return Ok((true, CommandExit::FAIL));
    }

    if let Some(conflicting_options) = &options.conflicting_options {
        let _ = writeln!(err, "{conflicting_options}");
        let _ = writeln!(err, "Usage: {}", get_package_command_usage(options.command));
        return Ok((true, CommandExit::FAIL));
    }

    let source = options.source.clone();
    if (options.command == PackageCommand::Install || options.command == PackageCommand::Remove)
        && source.is_none()
    {
        let _ = writeln!(err, "Missing {} source.", options.command.as_str());
        let _ = writeln!(err, "Usage: {}", get_package_command_usage(options.command));
        return Ok((true, CommandExit::FAIL));
    }

    if options.command == PackageCommand::Update
        && options
            .update_target
            .as_ref()
            .is_some_and(|target| matches!(target, UpdateTarget::Models))
    {
        if let Err(error) = refresh_model_catalogs(&get_agent_dir_with(&(runtime.env)()), out).await
        {
            let _ = writeln!(err, "Error: {error}");
            return Ok((true, CommandExit::FAIL));
        }
        return Ok((true, CommandExit::OK));
    }

    let cwd = runtime.cwd.to_string_lossy().into_owned();
    let agent_dir = runtime.agent_dir.clone();
    let writes_project_package_config = (options.command == PackageCommand::Install
        || options.command == PackageCommand::Remove)
        && options.local;

    let settings = Arc::clone(&runtime.settings);
    {
        let saved_trust = {
            let manager = settings
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            resolve_saved_project_trust(
                &manager,
                &cwd,
                &agent_dir,
                options.project_trust_override,
                options.command == PackageCommand::Update,
            )
        };
        settings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .set_project_trusted(saved_trust);
    }
    {
        let mut manager = settings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !manager.is_project_trusted() && writes_project_package_config {
            let _ = writeln!(
                err,
                "Project is not trusted. Use --approve to modify local package config."
            );
            return Ok((true, CommandExit::FAIL));
        }
        for error in manager.drain_errors() {
            let _ = writeln!(
                err,
                "Warning (package command, {} settings): {}",
                if error.scope == crate::settings_manager::SettingsScope::Global {
                    "global"
                } else {
                    "project"
                },
                error.message
            );
        }
    }

    let package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: cwd.clone(),
        agent_dir: agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&settings),
        command_runner: runtime.command_runner.clone(),
        env: Some((runtime.env)()),
        http_client: Some(Arc::clone(client)),
    });

    // The dispatch's failures print the upstream `Error: <message>` shape
    // and fail the exit code; only the parse errors above return earlier.
    let dispatched: Result<(bool, CommandExit), PackageManagerError> = (async {
    match options.command {
        PackageCommand::Install => {
            // The missing-source guard above already returned for install.
            let Some(source) = source.as_deref() else {
                return Ok((true, CommandExit::FAIL));
            };
            package_manager.install_and_persist(source, options.local).await?;
            let _ = writeln!(out, "Installed {source}");
            Ok((true, CommandExit::OK))
        }
        PackageCommand::Remove => {
            // The missing-source guard above already returned for remove.
            let Some(source) = source.as_deref() else {
                return Ok((true, CommandExit::FAIL));
            };
            let removed = package_manager.remove_and_persist(source, options.local).await?;
            if !removed {
                let _ = writeln!(err, "No matching package found for {source}");
                return Ok((true, CommandExit::FAIL));
            }
            let _ = writeln!(out, "Removed {source}");
            Ok((true, CommandExit::OK))
        }
        PackageCommand::List => {
            let configured_packages = package_manager.list_configured_packages();
            let user_packages: Vec<&ConfiguredPackage> = configured_packages
                .iter()
                .filter(|pkg| pkg.scope == SourceScope::User)
                .collect();
            let project_packages: Vec<&ConfiguredPackage> = configured_packages
                .iter()
                .filter(|pkg| pkg.scope == SourceScope::Project)
                .collect();

            if configured_packages.is_empty() {
                let _ = writeln!(out, "No packages installed.");
                return Ok((true, CommandExit::OK));
            }
            let has_user_packages = configured_packages.iter().any(|pkg| pkg.scope == SourceScope::User);

            let format_package = |pkg: &ConfiguredPackage, out: &mut dyn Write| {
                let display = if pkg.filtered {
                    format!("{} (filtered)", pkg.source)
                } else {
                    pkg.source.clone()
                };
                let _ = writeln!(out, "  {display}");
                if let Some(installed_path) = &pkg.installed_path {
                    let _ = writeln!(out, "    {installed_path}");
                }
            };

            if !user_packages.is_empty() {
                let _ = writeln!(out, "User packages:");
                for pkg in user_packages {
                    format_package(pkg, out);
                }
            }

            if !project_packages.is_empty() {
                if has_user_packages {
                    let _ = writeln!(out);
                }
                let _ = writeln!(out, "Project packages:");
                for pkg in project_packages {
                    format_package(pkg, out);
                }
            }

            Ok((true, CommandExit::OK))
        }
        PackageCommand::Update => {
            let target = options.update_target.clone().unwrap_or(UpdateTarget::SelfUpdate);
            if options.show_extensions_skipped_note {
                let _ = writeln!(out, "Extensions are skipped. Run {APP_NAME} update --extensions to update extensions.");
            }
            if update_target_includes_extensions(&target) {
                let update_source = match &target {
                    UpdateTarget::Extensions { source } => source.clone(),
                    _ => None,
                };
                package_manager.update(update_source.as_deref()).await?;
                match update_source {
                    Some(update_source) => {
                        let _ = writeln!(out, "Updated {update_source}");
                    }
                    None => {
                        let _ = writeln!(out, "Updated packages");
                    }
                }
            }
            if update_target_includes_self(&target) {
                let managed_install_root = match get_active_managed_install_root(&(runtime.env)()) {
                    Ok(Some(managed_install_root)) => Some(managed_install_root),
                    Ok(None) => None,
                    Err(error) => {
                        let _ = writeln!(err, "Error: {error}");
                        return Ok((true, CommandExit::FAIL));
                    }
                };
                if managed_install_root.is_some() && options.force {
                    let _ = writeln!(
                        err,
                        "Managed {APP_NAME} installations do not support --force; rerun the installer to repair this installation."
                    );
                    return Ok((true, CommandExit::FAIL));
                }
                let self_update_plan = get_self_update_plan(client, options.force, &(runtime.env)(), out).await?;
                if !self_update_plan.should_run {
                    return Ok((true, CommandExit::OK));
                }
                if let Some(managed_install_root) = managed_install_root {
                    if let Some(note) = &self_update_plan.note {
                        print_self_update_note(out, note);
                    }
                    let _ = writeln!(out, "Updating managed {APP_NAME} installation...");
                    if let Err(error) =
                        run_managed_self_update(client, &managed_install_root, &self_update_plan.version, &(runtime.env)())
                            .await
                    {
                        let _ = writeln!(err, "Error: {error}");
                        return Ok((true, CommandExit::FAIL));
                    }
                    let _ = writeln!(out, "Updated {APP_NAME} from {VERSION} to {}", self_update_plan.version);
                    return Ok((true, CommandExit::OK));
                }

                let install_method = detect_install_method_with(&current_exe_path(), &(runtime.env)());
                let self_update_target = SelfUpdatePackageTarget::new(&self_update_plan.package_name, Some(&self_update_plan.install_spec));
                let self_update_command = get_self_update_command_with(PACKAGE_NAME, &self_update_target, &current_exe_path(), &(runtime.env)());
                let Some(self_update_command) = self_update_command else {
                    print_self_update_unavailable(err, &self_update_target, &(runtime.env)());
                    return Ok((true, CommandExit::FAIL));
                };
                if let Some(note) = &self_update_plan.note {
                    print_self_update_note(out, note);
                }
                if let Err(error) = run_self_update(&self_update_command, out).await {
                    let _ = writeln!(err, "Error: {error}");
                    if install_method == InstallMethod::Cargo {
                        print_cargo_self_update_hint(err);
                    }
                    print_self_update_fallback(err, &self_update_command);
                    return Ok((true, CommandExit::FAIL));
                }
                let _ = writeln!(out, "Updated {APP_NAME} from {VERSION} to {}", self_update_plan.version);
            }
            Ok((true, CommandExit::OK))
        }
    }
    })
    .await;
    match dispatched {
        Ok(consumed) => Ok(consumed),
        Err(error) => {
            let _ = writeln!(err, "Error: {error}");
            Ok((true, CommandExit::FAIL))
        }
    }
}

/// The cargo-specific self-update hint, upstream's
/// `printPnpmSelfUpdateMetadataHint` restated to the cargo channel.
fn print_cargo_self_update_hint(err: &mut dyn Write) {
    let _ = writeln!(
        err,
        "If cargo reports a missing crate version, the registry index may be stale."
    );
    let _ = writeln!(
        err,
        "Run `cargo update` in a scratch checkout and retry `{APP_NAME} update --self`."
    );
}

/// Handle the `config` command, upstream's `handleConfigCommand` up to the
/// selector.
///
/// `Ok((false, _))` when the arguments are not a config command. The
/// resource-configuration TUI rides the interactive slice; this ports
/// the grammar, the trust gate, and the resolved-path computation.
///
/// # Errors
/// The grammar failures and the trust gate.
pub async fn handle_config_command<S: SettingsStorage + 'static>(
    args: &[String],
    runtime_env: &EnvLookup,
    err: &mut dyn Write,
) -> Result<(bool, CommandExit), PackageManagerError> {
    let Some(command) = args.first() else {
        return Ok((false, CommandExit::OK));
    };
    if command != "config" {
        return Ok((false, CommandExit::OK));
    }
    let rest = &args[1..];

    if rest.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_config_command_help(err);
        return Ok((true, CommandExit::OK));
    }

    let mut local = false;
    let mut project_trust_override: Option<bool> = None;
    for arg in rest {
        if arg == "-l" || arg == "--local" {
            local = true;
        } else if arg == "-a" || arg == "--approve" {
            project_trust_override = Some(true);
        } else if arg == "-na" || arg == "--no-approve" {
            project_trust_override = Some(false);
        } else if arg.starts_with('-') {
            let _ = writeln!(err, "Unknown option {arg} for \"config\".");
            let _ = writeln!(
                err,
                "Use \"{APP_NAME} --help\" or \"{CONFIG_COMMAND_USAGE}\"."
            );
            return Ok((true, CommandExit::FAIL));
        } else {
            let _ = writeln!(err, "Unexpected argument {arg}.");
            let _ = writeln!(err, "Usage: {CONFIG_COMMAND_USAGE}");
            return Ok((true, CommandExit::FAIL));
        }
    }

    let cwd = crate::config::process_cwd();
    let agent_dir = get_agent_dir_with(runtime_env);
    let mut manager = SettingsManager::<FileSettingsStorage>::create(
        &cwd,
        &agent_dir.to_string_lossy(),
        SettingsManagerCreateOptions::default(),
    );
    let saved_trust =
        resolve_saved_project_trust(&manager, &cwd, &agent_dir, project_trust_override, false);
    manager.set_project_trusted(saved_trust);
    if local && !manager.is_project_trusted() {
        let _ = writeln!(
            err,
            "Project is not trusted. Use --approve to modify local resource config."
        );
        return Ok((true, CommandExit::FAIL));
    }
    let global_settings_manager = SettingsManager::<FileSettingsStorage>::create(
        &cwd,
        &agent_dir.to_string_lossy(),
        SettingsManagerCreateOptions::default(),
    );
    let global_package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: cwd.clone(),
        agent_dir: agent_dir.to_string_lossy().into_owned(),
        settings: Arc::new(Mutex::new(global_settings_manager)),
        command_runner: None,
        env: None,
        http_client: None,
    });
    let global_resolved_paths: ResolvedPaths = global_package_manager.resolve(None).await?;
    let project_resolved_paths = if manager.is_project_trusted() {
        DefaultPackageManager::new(PackageManagerOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.to_string_lossy().into_owned(),
            settings: Arc::new(Mutex::new(manager)),
            command_runner: None,
            env: None,
            http_client: None,
        })
        .resolve(None)
        .await?
    } else {
        global_resolved_paths
    };

    // The resource-configuration TUI rides the interactive slice; the plan
    // the selector would consume is the computed pair.
    let _ = project_resolved_paths;
    Ok((true, CommandExit::OK))
}

/// The config command usage, upstream's `CONFIG_COMMAND_USAGE`.
pub const CONFIG_COMMAND_USAGE: &str = "pi config [-l] [--approve|--no-approve]";

fn print_config_command_help(out: &mut dyn Write) {
    let _ = writeln!(
        out,
        "Usage:\n  {CONFIG_COMMAND_USAGE}\n\nOpen the resource configuration TUI to enable or disable package resources.\nWithout -l, starts in global settings (~/{}/agent/settings.json).\nPress Tab in the TUI to switch between global and project-local modes.\n\nOptions:\n  -l, --local       Edit project overrides ({}/settings.json)\n  -a, --approve     Trust project-local files for this command with -l\n  -na, --no-approve Ignore project-local files for this command with -l",
        crate::config::CONFIG_DIR_NAME,
        crate::config::CONFIG_DIR_NAME
    );
}

fn print_package_command_help(command: PackageCommand, out: &mut dyn Write) {
    let usage = get_package_command_usage(command);
    match command {
        PackageCommand::Install => {
            let _ = writeln!(
                out,
                "Usage:\n  {usage}\n\nInstall a package and add it to settings.\n\nOptions:\n  -l, --local       Install project-locally ({}/settings.json)\n  -a, --approve     Trust project-local files for this command\n  -na, --no-approve Ignore project-local files for this command\n\nExamples:\n  {APP_NAME} install crate:pi-llm-tools\n  {APP_NAME} install github:user/repo\n  {APP_NAME} install https://example.com/tool-1.0.0.tgz\n  {APP_NAME} install ./local/path",
                crate::config::CONFIG_DIR_NAME
            );
        }
        PackageCommand::Remove => {
            let _ = writeln!(
                out,
                "Usage:\n  {usage}\n\nRemove a package and its source from settings.\nAlias: {APP_NAME} uninstall <source> [-l]\n\nOptions:\n  -l, --local       Remove from project settings ({}/settings.json)\n  -a, --approve     Trust project-local files for this command\n  -na, --no-approve Ignore project-local files for this command\n\nExamples:\n  {APP_NAME} remove crate:pi-llm-tools\n  {APP_NAME} uninstall crate:pi-llm-tools",
                crate::config::CONFIG_DIR_NAME
            );
        }
        PackageCommand::Update => {
            let _ = writeln!(
                out,
                "Usage:\n  {usage}\n\nUpdate pi, installed packages, or model catalogs.\n\nOptions:\n  --self                  Update pi only (default when no target is given)\n  --extensions            Update installed packages only\n  --models                Refresh model catalogs only\n  --all                   Update pi and installed packages\n  --extension <source>    Update one package only\n  -a, --approve           Trust project-local files for this command\n  -na, --no-approve       Ignore project-local files for this command\n  --force                 Reinstall pi even if the current version is latest\n\nShort forms:\n  {APP_NAME} update                Update pi only\n  {APP_NAME} update --all          Update pi and all extensions\n  {APP_NAME} update --models       Refresh model catalogs only\n  {APP_NAME} update <source>       Update one package\n  {APP_NAME} update pi             Update pi only (self works as alias to pi)"
            );
        }
        PackageCommand::List => {
            let _ = writeln!(
                out,
                "Usage:\n  {usage}\n\nList installed packages from user and project settings.\n\nOptions:\n  -a, --approve      Trust project-local files for this command\n  -na, --no-approve  Ignore project-local files for this command"
            );
        }
    }
}

fn get_package_command_usage(command: PackageCommand) -> String {
    match command {
        PackageCommand::Install => {
            format!("{APP_NAME} install <source> [-l] [--approve|--no-approve]")
        }
        PackageCommand::Remove => {
            format!("{APP_NAME} remove <source> [-l] [--approve|--no-approve]")
        }
        PackageCommand::Update => format!(
            "{APP_NAME} update [source|self|pi] [--self|--extensions|--models|--all] [--extension <source>] [--approve|--no-approve] [--force]"
        ),
        PackageCommand::List => format!("{APP_NAME} list [--approve|--no-approve]"),
    }
}
impl<S: SettingsStorage> std::fmt::Debug for PackageCommandRuntime<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackageCommandRuntime")
            .field("agent_dir", &self.agent_dir)
            .field("cwd", &self.cwd)
            .finish_non_exhaustive()
    }
}
