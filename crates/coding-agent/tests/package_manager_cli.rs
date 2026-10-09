//! The package-command paths suite, upstream's
//! `test/package-command-paths.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements this suite records:
//!
//! - The `main([...])` dispatch cases restate to direct
//!   [`handle_package_command`] calls with the runtime injected — the full
//!   main dispatch lands with the CLI-grammar slice (#130's map ticket).
//! - The npm self-update cases drop with the npm channel (ADR 0007); the
//!   managed-install cases port over the tarball channel's release
//!   artifact (`${installerApiBase}/<version>/download` unpacks to the
//!   release directory, smoke-tested through `bin/pi --version`), and the
//!   non-managed rename/pnpm-hint cases drop with the cargo redesign.
//! - The config-selector cycling case rides the interactive slice's
//!   selector component (the map's interactive-shell ticket) and defers.
//! - The `project_trust` extension-handler cases defer with the
//!   extension-system slice (ADR 0007's spawn surface).
//! - `process.exitCode` assertions restate to the returned
//!   [`CommandExit`].

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use flate2::write::GzEncoder;
use serde_json::json;

use pi_coding_agent::config::EnvLookup;
use pi_coding_agent::config::VERSION;
use pi_coding_agent::package_manager_cli::{
    CommandExit, PackageCommandRuntime, handle_config_command, handle_package_command,
};
use pi_coding_agent::settings_manager::{
    FileSettingsStorage, SettingsManager, SettingsManagerCreateOptions,
};

#[expect(
    dead_code,
    reason = "the shared fixture module is compiled into every test binary and this suite consumes only the settings helpers"
)]
mod common;

/// The CLI rig: a temp tree with the agent dir, project dir, and the
/// file-backed settings manager the commands share.
struct CliRig {
    temp_dir: PathBuf,
    agent_dir: PathBuf,
    project_dir: PathBuf,
    env_entries: Vec<(String, String)>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    client: Option<Arc<dyn pi_ai::http::HttpClient>>,
}

impl CliRig {
    fn new() -> Self {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let root = temp_dir.path().to_path_buf();
        std::mem::forget(temp_dir);
        let agent_dir = root.join("agent");
        let project_dir = root.join("project");
        std::fs::create_dir_all(&agent_dir).expect("agent dir");
        std::fs::create_dir_all(&project_dir).expect("project dir");
        Self {
            temp_dir: root,
            agent_dir,
            project_dir,
            env_entries: Vec::new(),
            stdout: Vec::new(),
            stderr: Vec::new(),
            client: None,
        }
    }

    fn with_env(mut self, entries: &[(&str, &str)]) -> Self {
        self.env_entries.extend(
            entries
                .iter()
                .map(|(key, value)| ((*key).to_string(), (*value).to_string())),
        );
        self
    }

    fn settings(
        &self,
        project_trusted: Option<bool>,
    ) -> Arc<Mutex<SettingsManager<FileSettingsStorage>>> {
        Arc::new(Mutex::new(SettingsManager::<FileSettingsStorage>::create(
            &self.project_dir.to_string_lossy(),
            &self.agent_dir.to_string_lossy(),
            SettingsManagerCreateOptions { project_trusted },
        )))
    }

    fn runtime(
        &self,
        settings: Arc<Mutex<SettingsManager<FileSettingsStorage>>>,
    ) -> PackageCommandRuntime<FileSettingsStorage> {
        PackageCommandRuntime {
            settings,
            agent_dir: self.agent_dir.clone(),
            cwd: self.project_dir.clone(),
            env: Arc::new({
                let entries = self.env_entries.clone();
                move || env_with_entries(entries.clone())
            }),
            command_runner: None,
        }
    }

    fn args(command: &[&str]) -> Vec<String> {
        command.iter().map(ToString::to_string).collect()
    }

    fn write(&self, relative: &str, contents: &str) -> PathBuf {
        let path = self.temp_dir.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("parent dirs");
        }
        std::fs::write(&path, contents).expect("write");
        path
    }

    fn stdout_text(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }

    fn stderr_text(&self) -> String {
        String::from_utf8_lossy(&self.stderr).into_owned()
    }
}

fn env_with_entries(entries: Vec<(String, String)>) -> EnvLookup {
    Box::new(move |key| {
        entries
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.clone())
    })
}

fn write_release_tarball(path: &Path, version: &str, exit_code: i32) {
    let bin_script = format!(
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then\n  if [ {exit_code} -ne 0 ]; then exit {exit_code}; fi\n  echo {version}\nfi\n"
    );
    write_tarball_with_pi_script(path, &bin_script);
}

/// The gzipped release artifact whose `bin/pi` runs the given script, the
/// shared builder behind the smoke-test fixtures.
fn write_tarball_with_pi_script(path: &Path, bin_script: &str) {
    let tar_path = path.with_extension("tar");
    let tar_file = std::fs::File::create(&tar_path).expect("tar file");
    let mut builder = tar::Builder::new(tar_file);
    let mut header = tar::Header::new_gnu();
    header.set_size(bin_script.len() as u64);
    header.set_mode(0o755);
    header.set_cksum();
    header.set_entry_type(tar::EntryType::Regular);
    builder
        .append_data(&mut header, "bin/pi", bin_script.as_bytes())
        .expect("append bin/pi");
    builder.finish().expect("finish tar");
    drop(builder);

    let tar_bytes = std::fs::read(&tar_path).expect("tar bytes");
    let gz_path = path.to_path_buf();
    let gz_file = std::fs::File::create(&gz_path).expect("gz file");
    let mut encoder = GzEncoder::new(gz_file, flate2::Compression::default());
    Write::write_all(&mut encoder, &tar_bytes).expect("gzip write");
    encoder.finish().expect("gzip finish");
    let _ = std::fs::remove_file(&tar_path);
}

/// The managed install fixture, upstream's `prepareManagedInstall` minus
/// the npm fake: the managed root, its marker, the active release, and the
/// served tarball.
struct ManagedInstall {
    managed_root: PathBuf,
}

/// The managed root's directory layout plus the launcher environment, the
/// half of `prepareManagedInstall` the marker/download variants share.
fn stage_managed_layout(rig: &mut CliRig) -> PathBuf {
    let managed_root = rig.agent_dir.join("install");
    let active_release = managed_root.join("releases").join(VERSION);
    let self_package_dir = active_release.join("package");
    std::fs::create_dir_all(&self_package_dir).expect("release package dir");
    rig.write(
        &format!("agent/install/releases/{VERSION}/active.txt"),
        "active",
    );
    rig.write("agent/install/current-version", &format!("{VERSION}\n"));
    rig.write(
        "agent/install/managed-install.json",
        &format!(
            "{}\n",
            json!({"kind": "pi-managed-install", "schemaVersion": 1, "layout": "releases-v1"})
        ),
    );
    rig.env_entries.push((
        "PI_MANAGED_INSTALL_ROOT".to_string(),
        managed_root.to_string_lossy().into_owned(),
    ));
    rig.env_entries.push((
        "PI_INSTALLER_API_BASE".to_string(),
        "https://example.test/api/installer/releases".to_string(),
    ));
    rig.env_entries.push((
        "PI_PACKAGE_DIR".to_string(),
        self_package_dir.to_string_lossy().into_owned(),
    ));
    managed_root
}

/// An already-unpacked release directory whose `bin/pi --version` prints
/// the given version, the fixture for the activate-existing-release path.
fn stage_release_dir(managed_root: &Path, version: &str) {
    use std::os::unix::fs::PermissionsExt;

    let release_bin = managed_root.join("releases").join(version).join("bin");
    std::fs::create_dir_all(&release_bin).expect("release bin dir");
    let bin_script = format!("#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo {version}; fi\n");
    let bin_path = release_bin.join("pi");
    std::fs::write(&bin_path, bin_script).expect("release bin script");
    std::fs::set_permissions(&bin_path, std::fs::Permissions::from_mode(0o755))
        .expect("release bin mode");
}

fn prepare_managed_install(
    rig: &mut CliRig,
    target_version: &str,
    smoke_exit_code: i32,
) -> ManagedInstall {
    let managed_root = stage_managed_layout(rig);

    // The release artifact: a tarball whose `bin/pi --version` prints the
    // target version (or exits nonzero when the smoke test must fail).
    let artifact = rig.temp_dir.join("release-artifact.tar.gz");
    write_release_tarball(&artifact, target_version, smoke_exit_code);
    let artifact_bytes = std::fs::read(&artifact).expect("artifact bytes");

    let version = target_version.to_string();
    let served = Arc::new(Mutex::new(artifact_bytes));
    let mock = pi_ai::http::MockHttpClient::new();
    let route_version = version.clone();
    mock.on(move |request| {
        request
            .url
            .ends_with(&format!("/api/installer/releases/{route_version}/download"))
    })
    .respond_fn({
        let served = Arc::clone(&served);
        move |_request| {
            let bytes = served
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            Box::pin(async move {
                Ok(pi_ai::http::MockResponse::status(200)
                    .with_header("content-type", "application/gzip")
                    .with_body(bytes))
            })
        }
    });
    // The update plan's version check rides the same client, upstream's
    // mockManagedUpdate answering both endpoints.
    mock.on(|request| request.url == "https://pi.dev/api/latest-version")
        .respond(pi_ai::http::json_response(
            200,
            &json!({ "packageName": "pi-coding-agent", "version": version }),
        ));
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(mock);
    // The client rides the runtime through the caller; stash it on the rig
    // via the return contract instead.
    rig.client.replace(client);
    ManagedInstall { managed_root }
}

/// Backdate a path's mtime with the POSIX touch, the utimesSync fixture
/// stand-in (the workspace has no safe set-time API without a new dep).
fn crate_runner_touch(path: &Path) -> bool {
    let outcome = pi_coding_agent::utils::child_process::spawn_process_sync(
        "touch",
        &["-t", "197001010000", &path.to_string_lossy()],
        &pi_coding_agent::utils::child_process::SpawnSyncOptions::IGNORE_OUTPUT,
    );
    outcome.status == Some(0)
}

fn newer_patch_version() -> String {
    let parts: Vec<&str> = VERSION.split('.').collect();
    let patch: u32 = parts
        .get(2)
        .and_then(|patch| patch.parse().ok())
        .unwrap_or(0);
    format!(
        "{}.{}.{}",
        parts.first().copied().unwrap_or("0"),
        parts.get(1).copied().unwrap_or("0"),
        patch + 1
    )
}

async fn run(rig: &mut CliRig, args: &[String]) -> (bool, CommandExit) {
    let settings = rig.settings(None);
    let runtime = rig.runtime(settings);
    let client = rig
        .client
        .clone()
        .unwrap_or_else(|| Arc::new(pi_ai::http::MockHttpClient::new()));
    let mut out = std::mem::take(&mut rig.stdout);
    let mut err = std::mem::take(&mut rig.stderr);
    let result = handle_package_command(args, runtime, &client, &mut out, &mut err).await;
    rig.stdout = out;
    rig.stderr = err;
    result.expect("command handles")
}

/// The config-command runner: [`handle_config_command`] with the rig's
/// injected environment and an error capture.
async fn run_config(rig: &CliRig, args: &[&str]) -> (bool, CommandExit, String) {
    let env = env_with_entries(rig.env_entries.clone());
    let mut err = Vec::new();
    let args: Vec<String> = args.iter().map(ToString::to_string).collect();
    let (consumed, exit) = handle_config_command::<FileSettingsStorage>(&args, &env, &mut err)
        .await
        .expect("config command handles");
    (consumed, exit, String::from_utf8_lossy(&err).into_owned())
}

/// The `pi.dev/api/latest-version` route answering the given JSON body.
fn route_latest_version(mock: &pi_ai::http::MockHttpClient, body: &serde_json::Value) {
    mock.on(|request| request.url == "https://pi.dev/api/latest-version")
        .respond(pi_ai::http::json_response(200, body));
}

/// The installer download route serving the given bytes.
fn route_release_download(mock: &pi_ai::http::MockHttpClient, version: &str, bytes: Vec<u8>) {
    let version = version.to_string();
    let served = Arc::new(Mutex::new(bytes));
    mock.on(move |request| {
        request
            .url
            .ends_with(&format!("/api/installer/releases/{version}/download"))
    })
    .respond_fn(move |_request| {
        let served = Arc::clone(&served);
        Box::pin(async move {
            let bytes = served
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            Ok(pi_ai::http::MockResponse::status(200)
                .with_header("content-type", "application/gzip")
                .with_body(bytes))
        })
    });
}

/// The managed-update fixture with the latest-version and download routes
/// mocked, returning the staged managed root.
fn prepare_managed_update(
    rig: &mut CliRig,
    latest_body: &serde_json::Value,
    artifact: &Path,
) -> PathBuf {
    let managed_root = stage_managed_layout(rig);
    let version = latest_body
        .get("version")
        .and_then(serde_json::Value::as_str)
        .expect("artifact version")
        .to_string();
    let mock = pi_ai::http::MockHttpClient::new();
    route_latest_version(&mock, latest_body);
    let bytes = std::fs::read(artifact).expect("artifact bytes");
    route_release_download(&mock, &version, bytes);
    rig.client.replace(Arc::new(mock));
    managed_root
}

// =============================================================================
// Grammar and dispatch
// =============================================================================

#[tokio::test]
async fn shows_the_install_subcommand_help() {
    let mut rig = CliRig::new();
    let args = CliRig::args(&["install", "--help"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK);
    let stdout = rig.stdout_text();
    assert!(stdout.contains("Usage:"));
    assert!(stdout.contains("pi install <source> [-l]"));
    assert!(!rig.stderr_text().contains("Error"));
}

#[tokio::test]
async fn shows_a_friendly_error_for_unknown_install_options() {
    let mut rig = CliRig::new();
    let args = CliRig::args(&["install", "--unknown"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    let stderr = rig.stderr_text();
    assert!(stderr.contains("Unknown option --unknown for \"install\"."));
    assert!(
        stderr.contains(
            "Use \"pi --help\" or \"pi install <source> [-l] [--approve|--no-approve]\"."
        )
    );
}

#[tokio::test]
async fn shows_a_friendly_error_for_a_missing_install_source() {
    let mut rig = CliRig::new();
    let args = CliRig::args(&["install"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    let stderr = rig.stderr_text();
    assert!(stderr.contains("Missing install source."));
    assert!(stderr.contains("Usage: pi install <source> [-l]"));
    assert!(!stderr.contains("at "), "no stack noise rides the error");
}

#[tokio::test]
async fn rejects_update_models_combined_with_another_update_target() {
    let mut rig = CliRig::new();
    let args = CliRig::args(&["update", "--models", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(
        rig.stderr_text()
            .contains("--models cannot be combined with --self")
    );
}

// =============================================================================
// Persistence and trust gating
// =============================================================================

#[tokio::test]
async fn persists_global_relative_local_package_paths_relative_to_settings() {
    let mut rig = CliRig::new();
    let relative_pkg_dir = rig.project_dir.join("packages/local-package");
    std::fs::create_dir_all(&relative_pkg_dir).expect("pkg dir");

    let args = CliRig::args(&["install", "./packages/local-package"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK);

    let settings_path = rig.agent_dir.join("settings.json");
    let settings: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&settings_path).expect("settings"))
            .expect("json");
    let packages = settings
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .expect("packages");
    assert_eq!(packages.len(), 1);
    let stored = packages[0].as_str().expect("string entry");
    let resolved = std::fs::canonicalize(rig.agent_dir.join(stored)).expect("resolve");
    assert_eq!(
        resolved,
        std::fs::canonicalize(&relative_pkg_dir).expect("resolve")
    );
}

#[tokio::test]
async fn removes_local_packages_using_a_path_with_a_trailing_slash() {
    let mut rig = CliRig::new();
    let package_dir = rig.temp_dir.join("local-package");
    std::fs::create_dir_all(&package_dir).expect("pkg dir");

    let args = CliRig::args(&["install", &format!("{}/", package_dir.to_string_lossy())]);
    let (_, exit) = run(&mut rig, &args).await;
    assert_eq!(exit, CommandExit::OK);
    let settings_path = rig.agent_dir.join("settings.json");
    let installed: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&settings_path).expect("settings"))
            .expect("json");
    assert_eq!(
        installed
            .get("packages")
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len),
        1
    );

    let args = CliRig::args(&["remove", &format!("{}/", package_dir.to_string_lossy())]);
    let (_, exit) = run(&mut rig, &args).await;
    assert_eq!(exit, CommandExit::OK);
    let removed: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&settings_path).expect("settings"))
            .expect("json");
    assert_eq!(
        removed
            .get("packages")
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len),
        0
    );
}

#[tokio::test]
async fn blocks_local_package_changes_when_the_project_is_untrusted() {
    let mut rig = CliRig::new();
    std::fs::create_dir_all(rig.project_dir.join(".pi")).expect(".pi dir");
    rig.write("project/.pi/settings.json", "{}");

    let args = CliRig::args(&["install", "-l", "./local-package"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(
        rig.stderr_text()
            .contains("Project is not trusted. Use --approve to modify local package config.")
    );
}

#[tokio::test]
async fn allows_local_package_install_to_initialize_fresh_project_settings() {
    let mut rig = CliRig::new();
    let package_dir = rig.temp_dir.join("local-package");
    std::fs::create_dir_all(&package_dir).expect("pkg dir");

    let args = CliRig::args(&["install", "-l", package_dir.to_string_lossy().as_ref()]);
    let (_, exit) = run(&mut rig, &args).await;
    assert_eq!(exit, CommandExit::OK);

    let settings_path = rig.project_dir.join(".pi/settings.json");
    let settings: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&settings_path).expect("settings"))
            .expect("json");
    let packages = settings
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .expect("packages");
    assert_eq!(packages.len(), 1);
    let stored = packages[0].as_str().expect("string entry");
    let resolved =
        std::fs::canonicalize(rig.project_dir.join(".pi").join(stored)).expect("resolve");
    assert_eq!(
        resolved,
        std::fs::canonicalize(&package_dir).expect("resolve")
    );
}

#[tokio::test]
async fn skips_untrusted_project_package_settings_for_list() {
    let mut rig = CliRig::new();
    std::fs::create_dir_all(rig.project_dir.join(".pi")).expect(".pi dir");
    rig.write(
        "project/.pi/settings.json",
        &json!({ "packages": ["crate:@project/pkg"] }).to_string(),
    );

    let args = CliRig::args(&["list"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK);
    let stdout = rig.stdout_text();
    assert!(stdout.contains("No packages installed."));
    assert!(!stdout.contains("Project packages:"));
}

#[tokio::test]
async fn uses_remembered_project_trust_for_list() {
    let mut rig = CliRig::new();
    std::fs::create_dir_all(rig.project_dir.join(".pi")).expect(".pi dir");
    rig.write(
        "project/.pi/settings.json",
        &json!({ "packages": ["crate:@project/pkg"] }).to_string(),
    );
    let trust_store =
        pi_coding_agent::trust_manager::ProjectTrustStore::new(&rig.agent_dir.to_string_lossy());
    trust_store
        .set(&rig.project_dir.to_string_lossy(), Some(true))
        .expect("trust set");

    let args = CliRig::args(&["list"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK);
    let stdout = rig.stdout_text();
    assert!(stdout.contains("Project packages:"));
    assert!(stdout.contains("crate:@project/pkg"));
    assert!(!stdout.contains("No packages installed."));
}

#[tokio::test]
async fn overrides_remembered_trust_for_list_with_no_approve() {
    let mut rig = CliRig::new();
    std::fs::create_dir_all(rig.project_dir.join(".pi")).expect(".pi dir");
    rig.write(
        "project/.pi/settings.json",
        &json!({ "packages": ["crate:@project/pkg"] }).to_string(),
    );
    let trust_store =
        pi_coding_agent::trust_manager::ProjectTrustStore::new(&rig.agent_dir.to_string_lossy());
    trust_store
        .set(&rig.project_dir.to_string_lossy(), Some(true))
        .expect("trust set");

    let args = CliRig::args(&["list", "--no-approve"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK);
    let stdout = rig.stdout_text();
    assert!(stdout.contains("No packages installed."));
    assert!(!stdout.contains("Project packages:"));
}

#[tokio::test]
async fn approves_project_trust_for_list_with_approve() {
    let mut rig = CliRig::new();
    std::fs::create_dir_all(rig.project_dir.join(".pi")).expect(".pi dir");
    rig.write(
        "project/.pi/settings.json",
        &json!({ "packages": ["crate:@project/pkg"] }).to_string(),
    );

    let args = CliRig::args(&["list", "--approve"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK);
    let stdout = rig.stdout_text();
    assert!(stdout.contains("Project packages:"));
    assert!(stdout.contains("crate:@project/pkg"));
    assert!(!stdout.contains("No packages installed."));
}

#[tokio::test]
async fn uses_the_default_project_trust_for_list() {
    let mut rig = CliRig::new();
    std::fs::create_dir_all(rig.project_dir.join(".pi")).expect(".pi dir");
    rig.write(
        "agent/settings.json",
        &json!({ "defaultProjectTrust": "always" }).to_string(),
    );
    rig.write(
        "project/.pi/settings.json",
        &json!({ "packages": ["crate:@project/pkg"] }).to_string(),
    );

    let args = CliRig::args(&["list"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK);
    let stdout = rig.stdout_text();
    assert!(stdout.contains("Project packages:"));
    assert!(stdout.contains("crate:@project/pkg"));
    assert!(!stdout.contains("No packages installed."));
}

#[tokio::test]
async fn lets_the_trust_store_override_the_default_project_trust() {
    let mut rig = CliRig::new();
    std::fs::create_dir_all(rig.project_dir.join(".pi")).expect(".pi dir");
    rig.write(
        "agent/settings.json",
        &json!({ "defaultProjectTrust": "always" }).to_string(),
    );
    rig.write(
        "project/.pi/settings.json",
        &json!({ "packages": ["crate:@project/pkg"] }).to_string(),
    );
    let trust_store =
        pi_coding_agent::trust_manager::ProjectTrustStore::new(&rig.agent_dir.to_string_lossy());
    trust_store
        .set(&rig.project_dir.to_string_lossy(), Some(false))
        .expect("trust set");

    let args = CliRig::args(&["list"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK);
    let stdout = rig.stdout_text();
    assert!(stdout.contains("No packages installed."));
    assert!(!stdout.contains("Project packages:"));
}

#[tokio::test]
async fn suggests_the_configured_source_when_the_update_input_omits_the_prefix() {
    let mut rig = CliRig::new();
    rig.write(
        "agent/settings.json",
        &json!({ "packages": ["crate:pi-formatter"] }).to_string(),
    );

    let args = CliRig::args(&["update", "pi-formatter"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    let stderr = rig.stderr_text();
    assert!(stderr.contains("Did you mean crate:pi-formatter?"));
    let stdout = rig.stdout_text();
    assert!(!stdout.contains("Updated pi-formatter"));

    let settings: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(rig.agent_dir.join("settings.json")).expect("settings"),
    )
    .expect("json");
    let packages = settings
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .expect("packages");
    assert!(
        packages
            .iter()
            .any(|package| package.as_str() == Some("crate:pi-formatter"))
    );
}

// =============================================================================
// Self-update
// =============================================================================

#[tokio::test]
async fn allows_explicit_self_update_checks_when_automatic_version_checks_are_disabled() {
    let mut rig = CliRig::new().with_env(&[("PI_SKIP_VERSION_CHECK", "1")]);
    let mock = pi_ai::http::MockHttpClient::new();
    mock.on(|request| request.url == "https://pi.dev/api/latest-version")
        .respond(pi_ai::http::json_response(
            200,
            &json!({ "version": VERSION }),
        ));
    rig.client.replace(Arc::new(mock.clone()));

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK);
    let stdout = rig.stdout_text();
    assert!(stdout.contains(&format!("pi is already up to date (v{VERSION})")));
    assert!(!rig.stderr_text().contains("Error"));
}

#[tokio::test]
async fn updates_installer_managed_pi_through_a_staged_immutable_release() {
    let mut rig = CliRig::new();
    let target_version = newer_patch_version();
    let install = prepare_managed_install(&mut rig, &target_version, 0);
    // An abandoned stage and a stale lock dir, upstream's fixtures.
    let abandoned_stage = install.managed_root.join("staging/update-abandoned");
    std::fs::create_dir_all(&abandoned_stage).expect("abandoned stage");
    rig.write("agent/install/staging/update-abandoned/partial", "partial");
    let abandoned_lock = install.managed_root.join("update.lock");
    std::fs::create_dir_all(&abandoned_lock).expect("abandoned lock");
    // Backdate the lock to the epoch, upstream's utimesSync fixture — the
    // stale-window recovery reclaims it.
    let _ = crate_runner_touch(&abandoned_lock);

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK, "stderr: {}", rig.stderr_text());

    let current = std::fs::read_to_string(install.managed_root.join("current-version"))
        .expect("current version");
    assert_eq!(current, format!("{target_version}\n"));
    assert!(
        install
            .managed_root
            .join("releases")
            .join(&target_version)
            .exists()
    );
    assert!(
        install.managed_root.join("releases").join(VERSION).exists(),
        "the old release stays"
    );
    let staging: Vec<_> = std::fs::read_dir(install.managed_root.join("staging"))
        .expect("staging")
        .flatten()
        .collect();
    assert!(
        staging.is_empty(),
        "the staging sweep clears the abandoned stage"
    );
    let stdout = rig.stdout_text();
    assert!(stdout.contains(&format!("Updated pi from {VERSION} to {target_version}")));
    assert!(!rig.stderr_text().contains("Error"));
}

#[tokio::test]
async fn rejects_a_concurrent_managed_update() {
    let mut rig = CliRig::new();
    let target_version = newer_patch_version();
    let install = prepare_managed_install(&mut rig, &target_version, 0);
    // Hold the update lock, upstream's lockfile.lock fixture.
    let lock_dir = pi_coding_agent::file_lock::lock_dir_for(
        &install.managed_root.join("update").to_string_lossy(),
    );
    let held = pi_coding_agent::file_lock::acquire_once(&lock_dir, None).expect("hold the lock");

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    let current = std::fs::read_to_string(install.managed_root.join("current-version"))
        .expect("current version");
    assert_eq!(current, format!("{VERSION}\n"), "the active release stays");
    let stdout = rig.stdout_text();
    assert!(!stdout.contains("Updated pi from"));
    assert!(
        rig.stderr_text()
            .contains("Another managed pi update is already running.")
    );
    let _ = held.release();
}

#[tokio::test]
async fn rejects_forced_managed_reinstalls() {
    let mut rig = CliRig::new();
    let target_version = newer_patch_version();
    let install = prepare_managed_install(&mut rig, &target_version, 0);

    let args = CliRig::args(&["update", "--self", "--force"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(
        rig.stderr_text()
            .contains("Managed pi installations do not support --force")
    );
    let current = std::fs::read_to_string(install.managed_root.join("current-version"))
        .expect("current version");
    assert_eq!(current, format!("{VERSION}\n"));
}

#[tokio::test]
async fn keeps_the_managed_release_active_when_its_update_fails() {
    let mut rig = CliRig::new();
    let target_version = newer_patch_version();
    let install = prepare_managed_install(&mut rig, &target_version, 23);

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    let current = std::fs::read_to_string(install.managed_root.join("current-version"))
        .expect("current version");
    assert_eq!(current, format!("{VERSION}\n"));
    assert!(
        !install
            .managed_root
            .join("releases")
            .join(&target_version)
            .exists(),
        "the failed release never activates"
    );
    let staging: Vec<_> = std::fs::read_dir(install.managed_root.join("staging"))
        .expect("staging")
        .flatten()
        .collect();
    assert!(staging.is_empty());
    let stderr = rig.stderr_text();
    assert!(stderr.contains("exit code 23"), "{stderr}");
    assert!(!rig.stdout_text().contains("Updated pi from"));
}

#[tokio::test]
async fn retries_a_transient_self_update_version_check() {
    let mut rig = CliRig::new();
    let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = Arc::clone(&attempts);
    let mock = pi_ai::http::MockHttpClient::new();
    mock.on(|request| request.url == "https://pi.dev/api/latest-version")
        .respond_fn(move |_request| {
            let nth = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async move {
                if nth < 2 {
                    Err(pi_ai::http::HttpError::Transport(
                        "fetch failed".to_string(),
                    ))
                } else {
                    Ok(pi_ai::http::json_response(
                        200,
                        &json!({ "version": VERSION }),
                    ))
                }
            }
        });
    rig.client.replace(Arc::new(mock.clone()));

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK);
    assert_eq!(
        attempts.load(std::sync::atomic::Ordering::SeqCst),
        3,
        "two failures then the answer"
    );
    assert!(!rig.stderr_text().contains("Error"));
}

// =============================================================================
// Grammar and dispatch, the remaining arms
// =============================================================================

#[tokio::test]
async fn treats_an_unrecognized_first_argument_as_a_non_package_command() {
    let mut rig = CliRig::new();
    let args = CliRig::args(&["nonsense"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(!consumed);
    assert_eq!(exit, CommandExit::OK);
    assert!(rig.stdout_text().is_empty());
    assert!(rig.stderr_text().is_empty());
}

#[tokio::test]
async fn shows_a_friendly_error_for_unknown_update_options() {
    let mut rig = CliRig::new();
    let args = CliRig::args(&["update", "--unknown"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    let stderr = rig.stderr_text();
    assert!(stderr.contains("Unknown option --unknown for \"update\"."));
    assert!(stderr.contains("pi update [source|self|pi]"));
}

#[tokio::test]
async fn shows_a_friendly_error_for_unknown_list_options() {
    let mut rig = CliRig::new();
    let args = CliRig::args(&["list", "--unknown"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    let stderr = rig.stderr_text();
    assert!(stderr.contains("Unknown option --unknown for \"list\"."));
    assert!(stderr.contains("pi list [--approve|--no-approve]"));
}

#[tokio::test]
async fn shows_a_friendly_error_for_a_missing_remove_source() {
    let mut rig = CliRig::new();
    let args = CliRig::args(&["remove"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    let stderr = rig.stderr_text();
    assert!(stderr.contains("Missing remove source."));
    assert!(stderr.contains("pi remove <source>"));
}

#[tokio::test]
async fn rejects_a_second_positional_argument() {
    let mut rig = CliRig::new();
    let args = CliRig::args(&["install", "./first", "./second"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    let stderr = rig.stderr_text();
    assert!(stderr.contains("Unexpected argument ./second."));
    assert!(stderr.contains("pi install <source>"));
}

#[tokio::test]
async fn rejects_update_only_flags_on_the_other_commands() {
    let mut rig = CliRig::new();
    for (command, flag) in [
        ("install", "--self"),
        ("install", "--extensions"),
        ("install", "--models"),
        ("install", "--all"),
        ("install", "--force"),
        ("install", "--extension"),
        ("remove", "--self"),
        ("remove", "--all"),
        ("list", "--self"),
        ("update", "-l"),
    ] {
        let args = CliRig::args(&[command, flag]);
        let (consumed, exit) = run(&mut rig, &args).await;
        assert!(consumed, "{command} {flag}");
        assert_eq!(exit, CommandExit::FAIL, "{command} {flag}");
        assert!(
            rig.stderr_text()
                .contains(&format!("Unknown option {flag} for \"{command}\".")),
            "{command} {flag}: {}",
            rig.stderr_text()
        );
    }
}

#[tokio::test]
async fn rejects_a_missing_extension_value() {
    let mut rig = CliRig::new();
    let args = CliRig::args(&["update", "--extension"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    let stderr = rig.stderr_text();
    assert!(stderr.contains("Missing value for --extension."));
    assert!(stderr.contains("pi update [source|self|pi]"));
}

#[tokio::test]
async fn rejects_a_second_extension_option() {
    let mut rig = CliRig::new();
    let args = CliRig::args(&["update", "--extension", "a", "--extension", "b"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(
        rig.stderr_text()
            .contains("--extension can only be provided once")
    );
}

#[tokio::test]
async fn rejects_all_combined_with_other_update_targets() {
    let mut rig = CliRig::new();
    let args = CliRig::args(&["update", "--all", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(
        rig.stderr_text().contains(
            "--all cannot be combined with --self, --extensions, --models, or --extension"
        )
    );

    let args = CliRig::args(&["update", "--all", "crate:some-package"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(
        rig.stderr_text()
            .contains("--all cannot be combined with a positional source")
    );
}

#[tokio::test]
async fn rejects_models_combined_with_a_positional_source() {
    let mut rig = CliRig::new();
    let args = CliRig::args(&["update", "--models", "crate:some-package"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(
        rig.stderr_text()
            .contains("--models cannot be combined with a positional source")
    );
}

#[tokio::test]
async fn rejects_extension_combined_with_other_targets() {
    let mut rig = CliRig::new();
    let args = CliRig::args(&["update", "--extension", "a", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(
        rig.stderr_text()
            .contains("--extension cannot be combined with --self, --extensions, or --all")
    );

    let args = CliRig::args(&["update", "--extension", "a", "b"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(
        rig.stderr_text()
            .contains("--extension cannot be combined with a positional source")
    );
}

#[tokio::test]
async fn rejects_a_positional_source_combined_with_flags() {
    let mut rig = CliRig::new();
    let args = CliRig::args(&["update", "crate:some-package", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(rig.stderr_text().contains(
        "positional update targets cannot be combined with --self, --extensions, or --all"
    ));
}

// =============================================================================
// Update targets, the remaining grammar
// =============================================================================

#[tokio::test]
async fn updates_the_named_package_through_the_extension_option() {
    let mut rig = CliRig::new().with_env(&[("PI_OFFLINE", "1")]);
    rig.write(
        "agent/settings.json",
        &json!({ "packages": ["crate:pi-formatter"] }).to_string(),
    );

    let args = CliRig::args(&["update", "--extension", "crate:pi-formatter"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK);
    assert!(rig.stdout_text().contains("Updated crate:pi-formatter"));
}

#[tokio::test]
async fn maps_the_pi_alias_to_the_self_target() {
    let mut rig = CliRig::new().with_env(&[("PI_OFFLINE", "1")]);

    let args = CliRig::args(&["update", "pi"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(
        rig.stderr_text()
            .contains("Could not determine latest pi version.")
    );
    assert!(!rig.stdout_text().contains("Updated"));
}

#[tokio::test]
async fn combines_the_pi_alias_with_the_extensions_target() {
    let mut rig = CliRig::new().with_env(&[("PI_OFFLINE", "1")]);

    let args = CliRig::args(&["update", "pi", "--extensions"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(rig.stdout_text().contains("Updated packages"));
    assert!(
        rig.stderr_text()
            .contains("Could not determine latest pi version.")
    );
}

#[tokio::test]
async fn updates_the_extensions_target() {
    let mut rig = CliRig::new().with_env(&[("PI_OFFLINE", "1")]);

    let args = CliRig::args(&["update", "--extensions"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK);
    assert!(rig.stdout_text().contains("Updated packages"));
    assert!(!rig.stderr_text().contains("Error"));
}

#[tokio::test]
async fn maps_the_bare_update_to_self_and_notes_the_extensions_skip() {
    let mut rig = CliRig::new().with_env(&[("PI_OFFLINE", "1")]);

    let args = CliRig::args(&["update"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(
        rig.stdout_text()
            .contains("Extensions are skipped. Run pi update --extensions to update extensions.")
    );
    assert!(
        rig.stderr_text()
            .contains("Could not determine latest pi version.")
    );
}

#[tokio::test]
async fn updates_everything_with_all() {
    let mut rig = CliRig::new().with_env(&[("PI_OFFLINE", "1")]);

    let args = CliRig::args(&["update", "--all"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(rig.stdout_text().contains("Updated packages"));
    assert!(
        rig.stderr_text()
            .contains("Could not determine latest pi version.")
    );
}

#[tokio::test]
async fn reads_saved_trust_only_for_update_commands() {
    let mut rig = CliRig::new().with_env(&[("PI_OFFLINE", "1")]);
    std::fs::create_dir_all(rig.project_dir.join(".pi")).expect(".pi dir");
    rig.write(
        "project/.pi/settings.json",
        &json!({ "packages": ["crate:@project/pkg"] }).to_string(),
    );

    let args = CliRig::args(&["update", "--extensions"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK);
    assert!(rig.stdout_text().contains("Updated packages"));
}

// =============================================================================
// List rendering
// =============================================================================

#[tokio::test]
async fn lists_user_and_project_packages_with_filters_and_install_paths() {
    let mut rig = CliRig::new();
    let local_dir = rig.temp_dir.join("local-listed");
    std::fs::create_dir_all(&local_dir).expect("local dir");
    rig.write(
        "agent/settings.json",
        &json!({ "packages": [
            "crate:@user/pkg",
            json!({"source": "crate:@user/limited", "skills": ["one"]}),
            local_dir.to_string_lossy()
        ] })
        .to_string(),
    );
    std::fs::create_dir_all(rig.project_dir.join(".pi")).expect(".pi dir");
    rig.write(
        "project/.pi/settings.json",
        &json!({ "packages": ["crate:@project/pkg"] }).to_string(),
    );
    let trust_store =
        pi_coding_agent::trust_manager::ProjectTrustStore::new(&rig.agent_dir.to_string_lossy());
    trust_store
        .set(&rig.project_dir.to_string_lossy(), Some(true))
        .expect("trust set");

    let args = CliRig::args(&["list"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK);
    let stdout = rig.stdout_text();
    assert!(stdout.contains("User packages:"));
    assert!(stdout.contains("crate:@user/pkg"));
    assert!(stdout.contains("crate:@user/limited (filtered)"));
    assert!(stdout.contains(local_dir.to_string_lossy().as_ref()));
    assert!(stdout.contains("Project packages:"));
    assert!(stdout.contains("crate:@project/pkg"));
    assert!(!stdout.contains("No packages installed."));
}

#[tokio::test]
async fn surfaces_settings_load_warnings_for_the_package_commands() {
    let mut rig = CliRig::new();
    std::fs::create_dir_all(rig.project_dir.join(".pi")).expect(".pi dir");
    rig.write("agent/settings.json", "{oops");
    rig.write("project/.pi/settings.json", "{oops");

    let args = CliRig::args(&["list"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK);
    let stderr = rig.stderr_text();
    assert!(stderr.contains("Warning (package command, global settings):"));
    assert!(stderr.contains("Warning (package command, project settings):"));
}

#[tokio::test]
async fn reports_no_matching_package_for_remove() {
    let mut rig = CliRig::new();

    let args = CliRig::args(&["remove", "crate:@never/installed"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(
        rig.stderr_text()
            .contains("No matching package found for crate:@never/installed")
    );
}

// =============================================================================
// Managed root detection
// =============================================================================

#[tokio::test]
async fn rejects_a_managed_root_with_a_missing_marker() {
    let mut rig = CliRig::new();
    let managed_root = stage_managed_layout(&mut rig);
    std::fs::remove_file(managed_root.join("managed-install.json")).expect("drop marker");

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(
        rig.stderr_text()
            .contains("Managed install marker is missing or invalid")
    );
}

#[tokio::test]
async fn rejects_a_managed_root_with_an_invalid_marker() {
    let mut rig = CliRig::new();
    let _managed_root = stage_managed_layout(&mut rig);
    rig.write("agent/install/managed-install.json", "{oops");

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(
        rig.stderr_text()
            .contains("Managed install marker is missing or invalid")
    );
}

#[tokio::test]
async fn rejects_a_managed_root_with_a_wrong_marker_kind() {
    let mut rig = CliRig::new();
    let _managed_root = stage_managed_layout(&mut rig);
    rig.write(
        "agent/install/managed-install.json",
        &json!({"kind": "other", "schemaVersion": 1, "layout": "releases-v1"}).to_string(),
    );

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(
        rig.stderr_text()
            .contains("Managed install marker is missing or invalid")
    );
}

#[tokio::test]
async fn ignores_a_managed_root_when_the_package_dir_sits_outside_it() {
    let mut rig = CliRig::new();
    // The root rides the env, the package dir env stays unset: the running
    // deps dir sits outside the root's releases tree.
    rig.env_entries.push((
        "PI_MANAGED_INSTALL_ROOT".to_string(),
        rig.temp_dir.join("install").to_string_lossy().into_owned(),
    ));
    let target = newer_patch_version();
    let mock = pi_ai::http::MockHttpClient::new();
    route_latest_version(&mock, &json!({ "version": target }));
    rig.client.replace(Arc::new(mock));

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(
        rig.stderr_text()
            .contains("error: pi cannot self-update this installation.")
    );
}

#[tokio::test]
async fn reports_self_update_unavailable_for_an_unknown_install() {
    let mut rig = CliRig::new();
    let target = newer_patch_version();
    let mock = pi_ai::http::MockHttpClient::new();
    route_latest_version(&mock, &json!({ "version": target }));
    rig.client.replace(Arc::new(mock));

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    let stderr = rig.stderr_text();
    assert!(stderr.contains("error: pi cannot self-update this installation."));
    assert!(stderr.contains("Update pi-coding-agent@"));
    assert!(stderr.contains(
        "using the package manager, wrapper, or source checkout that provides this installation."
    ));
    assert!(!rig.stdout_text().contains("Updated pi from"));
}

// =============================================================================
// Managed update, the failure arms
// =============================================================================

#[tokio::test]
async fn rejects_a_managed_release_version_that_is_not_semver() {
    let mut rig = CliRig::new();
    let _managed_root = stage_managed_layout(&mut rig);
    let mock = pi_ai::http::MockHttpClient::new();
    route_latest_version(
        &mock,
        &json!({ "packageName": "pi-other", "version": "not.a.version" }),
    );
    rig.client.replace(Arc::new(mock));

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(
        rig.stdout_text()
            .contains("Updating managed pi installation...")
    );
    assert!(
        rig.stderr_text()
            .contains("Invalid managed release version: not.a.version")
    );
}

#[tokio::test]
async fn fails_the_managed_update_when_the_update_lock_cannot_be_created() {
    use std::os::unix::fs::PermissionsExt;

    let mut rig = CliRig::new();
    let managed_root = stage_managed_layout(&mut rig);
    let target = newer_patch_version();
    let mock = pi_ai::http::MockHttpClient::new();
    route_latest_version(&mock, &json!({ "version": target }));
    rig.client.replace(Arc::new(mock));
    // The read-only root rejects the update lock's directory creation.
    std::fs::set_permissions(&managed_root, std::fs::Permissions::from_mode(0o555))
        .expect("read-only root");

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(rig.stderr_text().contains("Permission denied"));
}

#[tokio::test]
async fn activates_an_existing_managed_release_without_downloading() {
    let mut rig = CliRig::new();
    let target = newer_patch_version();
    let install = prepare_managed_install(&mut rig, &target, 0);
    stage_release_dir(&install.managed_root, &target);

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK, "stderr: {}", rig.stderr_text());
    let current = std::fs::read_to_string(install.managed_root.join("current-version"))
        .expect("current version");
    assert_eq!(current, format!("{target}\n"));
    assert!(
        rig.stdout_text()
            .contains(&format!("Updated pi from {VERSION} to {target}"))
    );
}

#[tokio::test]
async fn fails_the_managed_update_when_the_smoke_test_reports_another_version() {
    let mut rig = CliRig::new();
    let target = newer_patch_version();
    let artifact = rig.temp_dir.join("wrong-version.tar.gz");
    write_release_tarball(&artifact, "9.9.9", 0);
    let managed_root = prepare_managed_update(
        &mut rig,
        &json!({ "packageName": "pi-coding-agent", "version": target }),
        &artifact,
    );

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(rig.stderr_text().contains(&format!(
        "Managed pi smoke test returned version 9.9.9; expected {target}."
    )));
    assert!(
        !managed_root.join("releases").join(&target).exists(),
        "the mismatched release never activates"
    );
}

#[tokio::test]
async fn fails_the_managed_update_when_the_smoke_test_is_killed() {
    let mut rig = CliRig::new();
    let target = newer_patch_version();
    let artifact = rig.temp_dir.join("killed-smoke.tar.gz");
    write_tarball_with_pi_script(&artifact, "#!/bin/sh\nkill -TERM $$\n");
    let _managed_root = prepare_managed_update(
        &mut rig,
        &json!({ "packageName": "pi-coding-agent", "version": target }),
        &artifact,
    );

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(rig.stderr_text().contains("unknown exit status"));
}

#[tokio::test]
async fn fails_the_managed_update_when_the_release_download_fails() {
    let mut rig = CliRig::new();
    let target = newer_patch_version();
    let managed_root = stage_managed_layout(&mut rig);
    let mock = pi_ai::http::MockHttpClient::new();
    route_latest_version(&mock, &json!({ "version": target.clone() }));
    let route_target = target.clone();
    mock.on(move |request| {
        request
            .url
            .ends_with(&format!("/api/installer/releases/{route_target}/download"))
    })
    .respond(pi_ai::http::MockResponse::status(404).with_body(Vec::new()));
    rig.client.replace(Arc::new(mock));

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(rig.stderr_text().contains("HTTP 404"));
    assert!(
        !managed_root.join("releases").join(&target).exists(),
        "the failed download never activates"
    );
}

#[tokio::test]
async fn prints_the_release_note_before_a_managed_update() {
    let mut rig = CliRig::new();
    let target = newer_patch_version();
    let artifact = rig.temp_dir.join("noted-release.tar.gz");
    write_release_tarball(&artifact, &target, 0);
    let _managed_root = prepare_managed_update(
        &mut rig,
        &json!({
            "packageName": "pi-coding-agent",
            "version": target,
            "note": "## Notes\n\nHello note"
        }),
        &artifact,
    );

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK, "stderr: {}", rig.stderr_text());
    let stdout = rig.stdout_text();
    assert!(stdout.contains("Update note"));
    assert!(stdout.contains("Hello note"));
    assert!(
        rig.stdout_text()
            .contains(&format!("Updated pi from {VERSION} to {target}"))
    );
}

#[tokio::test]
async fn skips_a_blank_release_note() {
    let mut rig = CliRig::new();
    let target = newer_patch_version();
    let artifact = rig.temp_dir.join("blank-note.tar.gz");
    write_release_tarball(&artifact, &target, 0);
    let _managed_root = prepare_managed_update(
        &mut rig,
        &json!({
            "packageName": "pi-coding-agent",
            "version": target,
            "note": "   "
        }),
        &artifact,
    );

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK, "stderr: {}", rig.stderr_text());
    assert!(
        rig.stdout_text()
            .contains(&format!("Updated pi from {VERSION} to {target}"))
    );
    assert!(!rig.stdout_text().contains("Update note"));
}

#[tokio::test]
async fn fails_the_managed_update_when_the_version_pointer_cannot_be_renamed() {
    let mut rig = CliRig::new();
    let target = newer_patch_version();
    let install = prepare_managed_install(&mut rig, &target, 0);
    // A directory on the pointer path rejects the atomic rename.
    std::fs::remove_file(install.managed_root.join("current-version")).expect("drop pointer");
    std::fs::create_dir(install.managed_root.join("current-version")).expect("pointer dir");

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(rig.stderr_text().contains("Error:"));
    assert!(
        install.managed_root.join("releases").join(&target).exists(),
        "the release renamed in before the activation failed"
    );
}

#[test]
fn sweeps_stale_staging_through_the_cleanup_helpers() {
    let mut rig = CliRig::new();
    let managed_root = stage_managed_layout(&mut rig);
    let env = env_with_entries(rig.env_entries.clone());
    // The sweep with no staging directory takes the read-dir miss arm.
    pi_coding_agent::package_manager_cli::cleanup_managed_install_with(&env);

    let staging_root = managed_root.join("staging");
    std::fs::create_dir_all(staging_root.join("update-stale")).expect("stale staging");
    std::fs::write(staging_root.join("update-stale/partial"), "partial").expect("partial");
    // A live update holds the lock; the cleanup skips its sweep.
    let lock_dir =
        pi_coding_agent::file_lock::lock_dir_for(&managed_root.join("update").to_string_lossy());
    let held = pi_coding_agent::file_lock::acquire_once(&lock_dir, None).expect("hold the lock");
    std::fs::create_dir_all(staging_root.join("update-held")).expect("held staging");
    pi_coding_agent::package_manager_cli::cleanup_managed_install_with(&env);
    let held_remaining: Vec<_> = std::fs::read_dir(&staging_root)
        .expect("staging")
        .flatten()
        .collect();
    assert_eq!(held_remaining.len(), 2, "the held lock skips the sweep");
    let _ = held.release();
    pi_coding_agent::package_manager_cli::cleanup_managed_install_with(&env);
    let remaining: Vec<_> = std::fs::read_dir(&staging_root)
        .expect("staging")
        .flatten()
        .collect();
    assert!(remaining.is_empty(), "the stale stage swept");

    // The default-environment helper reads the process env, which carries
    // no managed root here.
    pi_coding_agent::package_manager_cli::cleanup_managed_install();
}

// =============================================================================
// Model catalogs
// =============================================================================

#[tokio::test]
async fn refreshes_model_catalogs_with_update_models() {
    let mut rig = CliRig::new();

    let args = CliRig::args(&["update", "--models"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK);
    assert!(rig.stdout_text().contains("Model catalogs refreshed"));
    assert!(!rig.stderr_text().contains("Error"));
}

// =============================================================================
// Config command grammar
// =============================================================================

#[tokio::test]
async fn treats_other_arguments_as_a_non_config_command() {
    let mut rig = CliRig::new();
    rig.env_entries.push((
        "PI_CODING_AGENT_DIR".to_string(),
        rig.agent_dir.to_string_lossy().into_owned(),
    ));

    let (consumed, exit, err) = run_config(&rig, &["other"]).await;
    assert!(!consumed);
    assert_eq!(exit, CommandExit::OK);
    assert!(err.is_empty());

    let (consumed, exit, err) = run_config(&rig, &[]).await;
    assert!(!consumed);
    assert_eq!(exit, CommandExit::OK);
    assert!(err.is_empty());
}

#[tokio::test]
async fn prints_the_config_command_help() {
    let mut rig = CliRig::new();
    rig.env_entries.push((
        "PI_CODING_AGENT_DIR".to_string(),
        rig.agent_dir.to_string_lossy().into_owned(),
    ));

    let (consumed, exit, err) = run_config(&rig, &["config", "--help"]).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK);
    assert!(err.contains("Usage:"));
    assert!(err.contains("pi config [-l] [--approve|--no-approve]"));
    assert!(err.contains("Press Tab in the TUI"));
}

#[tokio::test]
async fn rejects_unknown_config_options() {
    let mut rig = CliRig::new();
    rig.env_entries.push((
        "PI_CODING_AGENT_DIR".to_string(),
        rig.agent_dir.to_string_lossy().into_owned(),
    ));

    let (consumed, exit, err) = run_config(&rig, &["config", "--nope"]).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(err.contains("Unknown option --nope for \"config\"."));
    assert!(err.contains("pi config [-l] [--approve|--no-approve]"));
}

#[tokio::test]
async fn rejects_unexpected_config_arguments() {
    let mut rig = CliRig::new();
    rig.env_entries.push((
        "PI_CODING_AGENT_DIR".to_string(),
        rig.agent_dir.to_string_lossy().into_owned(),
    ));

    let (consumed, exit, err) = run_config(&rig, &["config", "extra"]).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(err.contains("Unexpected argument extra."));
    assert!(err.contains("pi config [-l] [--approve|--no-approve]"));
}

#[tokio::test]
async fn blocks_local_config_when_the_override_declines_trust() {
    let mut rig = CliRig::new();
    rig.env_entries.push((
        "PI_CODING_AGENT_DIR".to_string(),
        rig.agent_dir.to_string_lossy().into_owned(),
    ));

    let (consumed, exit, err) = run_config(&rig, &["config", "-l", "--no-approve"]).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(err.contains("Project is not trusted. Use --approve to modify local resource config."));
}

#[tokio::test]
async fn resolves_paths_for_the_config_command_with_approve() {
    let mut rig = CliRig::new();
    rig.env_entries.push((
        "PI_CODING_AGENT_DIR".to_string(),
        rig.agent_dir.to_string_lossy().into_owned(),
    ));

    let (consumed, exit, err) = run_config(&rig, &["config", "--approve"]).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK, "stderr: {err}");
    assert!(err.is_empty());
}

#[tokio::test]
async fn resolves_paths_for_the_config_command_without_trust() {
    let mut rig = CliRig::new();
    rig.env_entries.push((
        "PI_CODING_AGENT_DIR".to_string(),
        rig.agent_dir.to_string_lossy().into_owned(),
    ));

    let (consumed, exit, err) = run_config(&rig, &["config", "--no-approve"]).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK, "stderr: {err}");
    assert!(err.is_empty());
}

// =============================================================================
// Shared runtime shape
// =============================================================================

#[test]
fn renders_the_runtime_debug_shape() {
    let rig = CliRig::new();
    let settings = rig.settings(None);
    let runtime = rig.runtime(settings);
    let rendered = format!("{runtime:?}");
    assert!(rendered.contains("PackageCommandRuntime"));
    assert!(rendered.contains("agent_dir"));
}

// =============================================================================
// Help texts, the remaining commands
// =============================================================================

#[tokio::test]
async fn prints_the_help_for_the_other_package_commands() {
    let mut rig = CliRig::new();
    for (command, fragment) in [
        ("remove", "Alias: pi uninstall <source> [-l]"),
        ("update", "--extension <source>"),
        ("list", "pi list [--approve|--no-approve]"),
    ] {
        let args = CliRig::args(&[command, "--help"]);
        let (consumed, exit) = run(&mut rig, &args).await;
        assert!(consumed, "{command}");
        assert_eq!(exit, CommandExit::OK, "{command}");
        assert!(rig.stdout_text().contains(fragment), "{command}");
        assert!(!rig.stderr_text().contains("Error"), "{command}");
    }
}

// =============================================================================
// Self-update plan failures
// =============================================================================

#[tokio::test]
async fn reports_a_failed_version_check_for_self_update() {
    let mut rig = CliRig::new();
    // No mock route: the version check exhausts its immediate retries.
    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(
        rig.stderr_text()
            .contains("Could not determine latest pi version:")
    );
}

// =============================================================================
// Managed update, transport and smoke failures
// =============================================================================

#[tokio::test]
async fn fails_the_managed_update_when_the_release_download_transport_fails() {
    let mut rig = CliRig::new();
    let target = newer_patch_version();
    let managed_root = stage_managed_layout(&mut rig);
    // The latest-version route answers; the download route is absent, so
    // the artifact fetch fails at transport.
    let mock = pi_ai::http::MockHttpClient::new();
    route_latest_version(&mock, &json!({ "version": target.clone() }));
    rig.client.replace(Arc::new(mock));

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(rig.stderr_text().contains("Error:"));
    assert!(
        !managed_root.join("releases").join(&target).exists(),
        "the failed download never activates"
    );
}

#[tokio::test]
async fn fails_the_managed_update_when_the_smoke_test_reports_an_error() {
    let mut rig = CliRig::new();
    let target = newer_patch_version();
    let artifact = rig.temp_dir.join("stderr-smoke.tar.gz");
    write_tarball_with_pi_script(&artifact, "#!/bin/sh\necho boom >&2\nexit 7\n");
    let _managed_root = prepare_managed_update(
        &mut rig,
        &json!({ "packageName": "pi-coding-agent", "version": target }),
        &artifact,
    );

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::FAIL);
    assert!(rig.stderr_text().contains("Could not verify managed pi"));
    assert!(rig.stderr_text().contains("boom"));
}

#[tokio::test]
async fn percent_encodes_the_release_url_version() {
    let mut rig = CliRig::new();
    // Build metadata carries `+`, the byte the release URL percent-encodes.
    let version = "1.2.3+meta.1";
    let artifact = rig.temp_dir.join("meta-release.tar.gz");
    write_release_tarball(&artifact, version, 0);
    let managed_root = stage_managed_layout(&mut rig);
    let mock = pi_ai::http::MockHttpClient::new();
    route_latest_version(&mock, &json!({ "version": version }));
    let encoded = "1.2.3%2Bmeta.1";
    let served_bytes = std::fs::read(&artifact).expect("artifact bytes");
    mock.on(move |request| {
        request
            .url
            .ends_with(&format!("/api/installer/releases/{encoded}/download"))
    })
    .respond_fn({
        let served = Arc::new(Mutex::new(served_bytes));
        move |_request| {
            let bytes = served
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            Box::pin(async move {
                Ok(pi_ai::http::MockResponse::status(200)
                    .with_header("content-type", "application/gzip")
                    .with_body(bytes))
            })
        }
    });
    rig.client.replace(Arc::new(mock));

    let args = CliRig::args(&["update", "--self"]);
    let (consumed, exit) = run(&mut rig, &args).await;
    assert!(consumed);
    assert_eq!(exit, CommandExit::OK, "stderr: {}", rig.stderr_text());
    let current =
        std::fs::read_to_string(managed_root.join("current-version")).expect("current version");
    assert_eq!(current, format!("{version}\n"));
}
