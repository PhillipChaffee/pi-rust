//! The package-manager suite, upstream's `test/package-manager.test.ts` and
//! `test/package-manager-ssh.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, plus the boundary tests for
//! the re-expressed channels.
//!
//! Porting restatements this suite records:
//!
//! - The npm `npmCommand` describe block drops with the npm channel (ADR
//!   0007); the update-check cases restate onto the `crate:` channel over
//!   the crates.io mock, the batch update onto the fake runner, and the
//!   legacy-global/npm-root lookups drop (no global npm root exists).
//! - The `.ts`/`.js` extension fixtures restate to executable placeholder
//!   scripts — the executability filter is the restated extension-entry
//!   rule — and the `index.ts` convention restates to an executable
//!   `index`.
//! - The temporary npm path test restates to the `crate:` channel's
//!   temporary directory; the `node_modules` layout drops.
//! - The `spawnCaptureCommand` ordering test restates to the real runner:
//!   a child writing after a delay must resolve only with the full output.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use pi_coding_agent::config::EnvLookup;
use pi_coding_agent::package_manager::{
    CommandRunOptions, CommandRunner, DefaultPackageManager, InstallReceipt, PackageManagerError,
    PackageManagerOptions, PackageSourceView, ParsedSource, ProgressAction, ProgressEvent,
    ProgressEventType, ResolvedResource, SourceScope, TarballSource,
};
use pi_coding_agent::settings_manager::{
    InMemorySettingsStorage, SettingsManager, SettingsManagerCreateOptions,
};

#[expect(
    dead_code,
    reason = "the shared fixture module is compiled into every test binary and this suite consumes only its settings helpers"
)]
mod common;

// =============================================================================
// The fake runner, upstream's vi.spyOn(runCommand) restated
// =============================================================================

/// The outcome a fake command produces.
#[derive(Debug, Clone, Default)]
pub struct FakeOutcome {
    /// The captured stdout for `run_capture`.
    pub stdout: String,
    /// The captured stderr for `run_capture`.
    pub stderr: String,
    /// The exit code; `None` is a success.
    pub code: Option<i32>,
    /// A spawn failure message, upstream's `child.on("error")`.
    pub spawn_error: Option<String>,
}

impl FakeOutcome {
    fn ok() -> Self {
        Self::default()
    }

    fn stdout(stdout: impl Into<String>) -> Self {
        Self {
            stdout: stdout.into(),
            ..Self::default()
        }
    }
}

type FakeBehavior = Box<
    dyn Fn(&str, &[String], &CommandRunOptions) -> Result<FakeOutcome, PackageManagerError>
        + Send
        + Sync,
>;

/// One recorded runner invocation: the command, its argv, and the call's cwd,
/// the shape the install and update assertions replay.
type RecordedCall = (String, Vec<String>, Option<String>);

/// The cargo-install fake, the `vi.spyOn(runCommand)` clone: the `--root`
/// argument names the staging root, the fake populates `bin/example` so the
/// atomic rename lands, and every other command succeeds.
fn fake_cargo_install() -> FakeBehavior {
    Box::new(
        |command: &str, args: &[String], _options: &CommandRunOptions| {
            if command == "cargo" && args.first().map(String::as_str) == Some("install") {
                let root_index = args
                    .iter()
                    .position(|arg| arg == "--root")
                    .expect("--root present");
                let stage_root = args.get(root_index + 1).cloned().unwrap_or_default();
                std::fs::create_dir_all(Path::new(&stage_root).join("bin")).expect("stage bin");
                std::fs::write(Path::new(&stage_root).join("bin/example"), "#!/bin/sh\n")
                    .expect("bin");
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    std::fs::set_permissions(
                        Path::new(&stage_root).join("bin/example"),
                        std::fs::Permissions::from_mode(0o755),
                    )
                    .expect("chmod");
                }
            }
            Ok(FakeOutcome::ok())
        },
    )
}

/// The recording command runner, upstream's `vi.spyOn(packageManager as any,
/// "runCommand")` double: every invocation records, the behavior closure
/// answers.
#[derive(Clone)]
struct FakeRunner {
    behavior: Arc<FakeBehavior>,
    ran: Arc<Mutex<Vec<RecordedCall>>>,
}

impl FakeRunner {
    fn new(behavior: FakeBehavior) -> Self {
        Self {
            behavior: Arc::new(behavior),
            ran: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn rejecting(message: &'static str) -> Self {
        Self::new(Box::new(move |_command, _args, _options| {
            Err(PackageManagerError(message.to_string()))
        }))
    }

    fn recorded(&self) -> Vec<RecordedCall> {
        self.ran
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl CommandRunner for FakeRunner {
    fn run(
        &self,
        command: &str,
        args: &[String],
        options: &CommandRunOptions,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<(), PackageManagerError>> + Send + '_>> {
        let outcome = (self.behavior)(command, args, options);
        let label = format!("{command} {}", args.join(" "));
        self.ran
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((command.to_string(), args.to_vec(), options.cwd.clone()));
        Box::pin(async move {
            let outcome = outcome?;
            match outcome.code {
                None | Some(0) => Ok(()),
                Some(code) => Err(PackageManagerError(format!(
                    "{label} failed with code {code}"
                ))),
            }
        })
    }

    fn run_capture(
        &self,
        command: &str,
        args: &[String],
        options: &CommandRunOptions,
        _timeout_ms: Option<u64>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<String, PackageManagerError>> + Send + '_>>
    {
        let outcome = (self.behavior)(command, args, options);
        let label = format!("{command} {}", args.join(" "));
        self.ran
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((command.to_string(), args.to_vec(), options.cwd.clone()));
        Box::pin(async move {
            let outcome = outcome?;
            if let Some(spawn_error) = outcome.spawn_error {
                return Err(PackageManagerError(spawn_error));
            }
            match outcome.code {
                None | Some(0) => Ok(outcome.stdout.trim().to_string()),
                Some(code) => {
                    let detail = if outcome.stderr.trim().is_empty() {
                        outcome.stdout
                    } else {
                        outcome.stderr
                    };
                    Err(PackageManagerError(format!(
                        "{label} failed with code {code}: {detail}"
                    )))
                }
            }
        })
    }

    fn run_sync(&self, command: &str, args: &[String]) -> Result<String, PackageManagerError> {
        let outcome = (self.behavior)(command, args, &CommandRunOptions::default())?;
        self.ran
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((command.to_string(), args.to_vec(), None));
        if let Some(spawn_error) = outcome.spawn_error {
            return Err(PackageManagerError(format!(
                "Failed to run {} {}: {spawn_error}",
                command,
                args.join(" ")
            )));
        }
        if outcome.code.is_some_and(|code| code != 0) {
            return Err(PackageManagerError(format!(
                "Failed to run {} {}: exit code {}",
                command,
                args.join(" "),
                outcome.code.expect("checked")
            )));
        }
        let output = if outcome.stdout.is_empty() {
            outcome.stderr
        } else {
            outcome.stdout
        };
        Ok(output.trim().to_string())
    }
}

// =============================================================================
// The rig
// =============================================================================

struct Rig {
    temp_dir: PathBuf,
    agent_dir: PathBuf,
    settings: Arc<Mutex<SettingsManager<InMemorySettingsStorage>>>,
    package_manager: DefaultPackageManager<InMemorySettingsStorage>,
    runner: Option<FakeRunner>,
}

impl Rig {
    fn new() -> Self {
        Self::with_env(&[])
    }

    fn with_env(entries: &[(&str, &str)]) -> Self {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let root = temp_dir.path().to_path_buf();
        let agent_dir = root.join("agent");
        std::fs::create_dir_all(&agent_dir).expect("agent dir");
        let settings = Arc::new(Mutex::new(SettingsManager::in_memory(
            &serde_json::Map::new(),
            SettingsManagerCreateOptions::default(),
        )));
        let package_manager = DefaultPackageManager::new(PackageManagerOptions {
            cwd: root.to_string_lossy().into_owned(),
            agent_dir: agent_dir.to_string_lossy().into_owned(),
            settings: Arc::clone(&settings),
            command_runner: None,
            env: Some(common::env_with(entries)),
            http_client: None,
        });
        std::mem::forget(temp_dir);
        Self {
            temp_dir: root,
            agent_dir,
            settings,
            package_manager,
            runner: None,
        }
    }

    fn with_runner(&mut self, runner: FakeRunner) -> &mut Self {
        self.runner = Some(runner.clone());
        self.package_manager = DefaultPackageManager::new(PackageManagerOptions {
            cwd: self.temp_dir.to_string_lossy().into_owned(),
            agent_dir: self.agent_dir.to_string_lossy().into_owned(),
            settings: Arc::clone(&self.settings),
            command_runner: Some(Arc::new(runner)),
            env: Some(common::env_with(&[])),
            http_client: None,
        });
        self
    }

    fn with_cwd(&mut self, cwd: &Path) -> &mut Self {
        self.package_manager = DefaultPackageManager::new(PackageManagerOptions {
            cwd: cwd.to_string_lossy().into_owned(),
            agent_dir: self.agent_dir.to_string_lossy().into_owned(),
            settings: Arc::clone(&self.settings),
            command_runner: None,
            env: Some(common::env_with(&[])),
            http_client: None,
        });
        self
    }

    fn set_packages(&self, packages: &Value) {
        self.settings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .set_packages(packages.as_array().map_or(&[], Vec::as_slice));
    }

    fn set_project_packages(&self, packages: &Value) {
        self.settings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .set_project_packages(packages.as_array().map_or(&[], Vec::as_slice))
            .expect("project packages write");
    }

    fn set_setting(&self, key: &str, value: Value) {
        let mut manager = self
            .settings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut settings = manager.get_global_settings();
        settings.insert(key.to_string(), value);
        manager.set_global_settings_map(&settings);
    }

    fn set_project_setting(&self, key: &str, value: Value) {
        let mut manager = self
            .settings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut settings = manager.get_project_settings();
        settings.insert(key.to_string(), value);
        manager
            .set_project_settings_map(&settings)
            .expect("project settings write");
    }

    fn global_packages(&self) -> Vec<Value> {
        self.settings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_packages()
    }

    fn write(&self, relative: &str, contents: &str) -> PathBuf {
        let path = self.temp_dir.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("parent dirs");
        }
        std::fs::write(&path, contents).expect("write");
        path
    }

    fn mkdir(&self, relative: &str) -> PathBuf {
        let path = self.temp_dir.join(relative);
        std::fs::create_dir_all(&path).expect("mkdir");
        path
    }

    /// A placeholder executable, the restated `.ts` fixture: a shell script
    /// with the execute bit set.
    fn write_executable(&self, relative: &str) -> PathBuf {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let path = self.write(relative, "#!/bin/sh\n");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
            path
        }
        #[cfg(not(unix))]
        {
            self.write(relative, "stub")
        }
    }
}

// The settings-manager helpers the rig's typed setters ride; the raw map
// accessors exist upstream as full-settings clones the manager mutates.
trait RawSettingsAccess {
    fn set_global_settings_map(&mut self, settings: &serde_json::Map<String, Value>);
    fn set_project_settings_map(
        &mut self,
        settings: &serde_json::Map<String, Value>,
    ) -> Result<(), String>;
}

impl RawSettingsAccess for SettingsManager<InMemorySettingsStorage> {
    fn set_global_settings_map(&mut self, settings: &serde_json::Map<String, Value>) {
        self.set_packages(
            settings
                .get("packages")
                .and_then(Value::as_array)
                .map_or(&[], Vec::as_slice),
        );
        for key in ["extensions", "skills", "prompts", "themes"] {
            let entries: Vec<String> = settings
                .get(key)
                .and_then(Value::as_array)
                .map(|array| {
                    array
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            match key {
                "extensions" => self.set_extension_paths(&entries),
                "skills" => self.set_skill_paths(&entries),
                "prompts" => self.set_prompt_template_paths(&entries),
                "themes" => self.set_theme_paths(&entries),
                _ => unreachable!("the loop enumerates these four keys"),
            }
        }
        if let Some(default_project_trust) = settings.get("defaultProjectTrust") {
            self.set_default_project_trust(match default_project_trust.as_str() {
                Some("always") => pi_coding_agent::settings_manager::DefaultProjectTrust::Always,
                Some("never") => pi_coding_agent::settings_manager::DefaultProjectTrust::Never,
                _ => pi_coding_agent::settings_manager::DefaultProjectTrust::Ask,
            });
        }
        if let Some(npm_command) = settings.get("npmCommand") {
            let command: Vec<String> = npm_command
                .as_array()
                .map(|array| {
                    array
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            self.set_npm_command(Some(&command));
        }
    }

    fn set_project_settings_map(
        &mut self,
        settings: &serde_json::Map<String, Value>,
    ) -> Result<(), String> {
        self.set_project_packages(
            settings
                .get("packages")
                .and_then(Value::as_array)
                .map_or(&[], Vec::as_slice),
        )?;
        for key in ["extensions", "skills", "prompts", "themes"] {
            let entries: Vec<String> = settings
                .get(key)
                .and_then(Value::as_array)
                .map(|array| {
                    array
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            match key {
                "extensions" => {
                    self.set_project_extension_paths(&entries)?;
                }
                "skills" => {
                    self.set_project_skill_paths(&entries)?;
                }
                "prompts" => {
                    self.set_project_prompt_template_paths(&entries)?;
                }
                "themes" => {
                    self.set_project_theme_paths(&entries)?;
                }
                _ => unreachable!("the loop enumerates these four keys"),
            }
        }
        Ok(())
    }
}

// =============================================================================
// resolve
// =============================================================================

/// Whether two paths name the same location once canonicalized — the macOS
/// `/tmp` → `/private/tmp` symlink folds away.
fn same_path(left: &str, right: &Path) -> bool {
    let left = std::fs::canonicalize(left)
        .map_or_else(|_| left.to_string(), |p| p.to_string_lossy().into_owned());
    let right = std::fs::canonicalize(right).map_or_else(
        |_| right.to_string_lossy().into_owned(),
        |p| p.to_string_lossy().into_owned(),
    );
    left == right
}

fn find_enabled(resources: &[ResolvedResource], suffix: &str) -> bool {
    resources
        .iter()
        .any(|resource| resource.path.replace('\\', "/").ends_with(suffix) && resource.enabled)
}

fn find_disabled(resources: &[ResolvedResource], suffix: &str) -> bool {
    resources
        .iter()
        .any(|resource| resource.path.replace('\\', "/").ends_with(suffix) && !resource.enabled)
}

fn find_with(resources: &[ResolvedResource], needle: &str) -> bool {
    resources
        .iter()
        .any(|resource| resource.path.replace('\\', "/").contains(needle) && resource.enabled)
}

#[tokio::test]
async fn resolve_returns_no_package_sourced_paths_without_sources() {
    let rig = Rig::new();
    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(result.extensions.is_empty());
    assert!(result.prompts.is_empty());
    assert!(result.themes.is_empty());
    assert!(
        result
            .skills
            .iter()
            .all(|resource| resource.metadata.source == "auto"
                && resource.metadata.origin.as_str() == "top-level")
    );
}

#[tokio::test]
async fn resolve_resolves_local_extension_paths_from_settings() {
    let rig = Rig::new();
    let ext_path = rig.write("agent/extensions/my-extension.ts-placeholder", "stub");
    let _ = ext_path;
    rig.write_executable("agent/extensions/my-extension");
    rig.set_setting("extensions", json!(["extensions/my-extension"]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(result.extensions.iter().any(|resource| {
        resource.path
            == rig
                .temp_dir
                .join("agent/extensions/my-extension")
                .to_string_lossy()
            && resource.enabled
    }));
}

#[tokio::test]
async fn resolve_resolves_skill_paths_from_settings() {
    let rig = Rig::new();
    let skill_file = rig.write(
        "agent/skills/my-skill/SKILL.md",
        "---\nname: test-skill\ndescription: A test skill\n---\nContent",
    );
    rig.set_setting("skills", json!(["skills"]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(
        result
            .skills
            .iter()
            .any(|resource| resource.path == skill_file.to_string_lossy() && resource.enabled)
    );
}

#[tokio::test]
async fn resolve_auto_discovers_root_markdown_skills() {
    let rig = Rig::new();
    let skill_file = rig.write(
        "agent/skills/single-file.md",
        "---\nname: single-file\ndescription: A root markdown skill\n---\nContent",
    );

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(
        result
            .skills
            .iter()
            .any(|resource| resource.path == skill_file.to_string_lossy() && resource.enabled)
    );
}

#[tokio::test]
async fn resolve_resolves_project_paths_relative_to_the_pi_dir() {
    let rig = Rig::new();
    let ext_path = rig.write_executable(".pi/extensions/project-ext");
    rig.set_project_setting("extensions", json!(["extensions/project-ext"]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(
        result
            .extensions
            .iter()
            .any(|resource| resource.path == ext_path.to_string_lossy() && resource.enabled)
    );
}

#[tokio::test]
async fn resolve_auto_discovers_user_prompts_with_overrides() {
    let rig = Rig::new();
    let prompt_path = rig.write("agent/prompts/auto.md", "Auto prompt");
    rig.set_setting("prompts", json!(["!prompts/auto.md"]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(
        result
            .prompts
            .iter()
            .any(|resource| resource.path == prompt_path.to_string_lossy() && !resource.enabled)
    );
}

#[tokio::test]
async fn resolve_resolves_symlinked_user_and_project_resources_once() {
    let rig = Rig::new();
    let shared_dir = rig.mkdir("shared-resources");
    for name in ["extensions", "skills", "prompts", "themes"] {
        std::fs::create_dir_all(shared_dir.join(name)).expect("shared dir");
    }
    rig.write_executable("shared-resources/extensions/shared");
    rig.write(
        "shared-resources/skills/shared-skill/SKILL.md",
        "---\nname: shared-skill\ndescription: Shared skill\n---\nContent",
    );
    rig.write("shared-resources/prompts/shared.md", "Shared prompt");
    rig.write(
        "shared-resources/themes/shared.json",
        r#"{"name":"shared-theme"}"#,
    );

    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(
            shared_dir.join("extensions"),
            rig.agent_dir.join("extensions"),
        )
        .expect("symlink");
        std::os::unix::fs::symlink(shared_dir.join("skills"), rig.agent_dir.join("skills"))
            .expect("symlink");
        std::os::unix::fs::symlink(shared_dir.join("prompts"), rig.agent_dir.join("prompts"))
            .expect("symlink");
        std::os::unix::fs::symlink(shared_dir.join("themes"), rig.agent_dir.join("themes"))
            .expect("symlink");
        std::fs::create_dir_all(rig.temp_dir.join(".pi")).expect(".pi dir");
        std::os::unix::fs::symlink(
            shared_dir.join("extensions"),
            rig.temp_dir.join(".pi/extensions"),
        )
        .expect("symlink");
        std::os::unix::fs::symlink(shared_dir.join("skills"), rig.temp_dir.join(".pi/skills"))
            .expect("symlink");
        std::os::unix::fs::symlink(shared_dir.join("prompts"), rig.temp_dir.join(".pi/prompts"))
            .expect("symlink");
        std::os::unix::fs::symlink(shared_dir.join("themes"), rig.temp_dir.join(".pi/themes"))
            .expect("symlink");
    }

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert_eq!(result.extensions.len(), 1);
    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.prompts.len(), 1);
    assert_eq!(result.themes.len(), 1);
    // Project auto-discovery outranks user auto-discovery, so the survivor
    // is project-scoped.
    assert_eq!(result.extensions[0].metadata.scope, SourceScope::Project);
    assert_eq!(result.skills[0].metadata.scope, SourceScope::Project);
    assert_eq!(result.prompts[0].metadata.scope, SourceScope::Project);
    assert_eq!(result.themes[0].metadata.scope, SourceScope::Project);
}

#[tokio::test]
async fn resolve_auto_discovers_project_prompts_with_overrides() {
    let rig = Rig::new();
    let prompt_path = rig.write(".pi/prompts/is.md", "Is prompt");
    rig.set_project_setting("prompts", json!(["!prompts/is.md"]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(
        result
            .prompts
            .iter()
            .any(|resource| resource.path == prompt_path.to_string_lossy() && !resource.enabled)
    );
}

#[tokio::test]
async fn resolve_resolves_a_directory_declaring_pi_extensions() {
    let rig = Rig::new();
    let pkg_dir = rig.mkdir("my-extensions-pkg");
    std::fs::create_dir_all(pkg_dir.join("extensions")).expect("extensions dir");
    rig.write_executable("my-extensions-pkg/extensions/clip");
    rig.write_executable("my-extensions-pkg/extensions/cost");
    rig.write_executable("my-extensions-pkg/extensions/helper");
    rig.write(
        "my-extensions-pkg/package.json",
        r#"{"name":"my-extensions-pkg","pi":{"extensions":["./extensions/clip","./extensions/cost"]}}"#,
    );
    rig.set_setting(
        "extensions",
        json!([rig
            .temp_dir
            .join("my-extensions-pkg")
            .to_string_lossy()
            .into_owned()]),
    );

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(result.extensions.iter().any(|resource| resource.path
        == pkg_dir.join("extensions/clip").to_string_lossy()
        && resource.enabled));
    assert!(result.extensions.iter().any(|resource| resource.path
        == pkg_dir.join("extensions/cost").to_string_lossy()
        && resource.enabled));
    assert!(!find_with(&result.extensions, "helper"));
}

// =============================================================================
// Auto-discovered skill metadata
// =============================================================================

#[tokio::test]
async fn uses_the_agent_dir_as_base_dir_for_user_pi_skills() {
    let rig = Rig::new();
    let skill_path = rig.write(
        "agent/skills/user-pi/SKILL.md",
        "---\nname: user-pi\ndescription: user pi\n---\n",
    );

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    let skill = result
        .skills
        .iter()
        .find(|resource| resource.path == skill_path.to_string_lossy())
        .expect("skill resolves");
    assert_eq!(skill.metadata.source, "auto");
    assert_eq!(skill.metadata.scope, SourceScope::User);
    assert_eq!(
        skill.metadata.base_dir.as_deref(),
        Some(rig.agent_dir.to_string_lossy().as_ref())
    );
}

#[tokio::test]
async fn uses_the_project_pi_dir_as_base_dir_for_project_pi_skills() {
    let rig = Rig::new();
    let project_base_dir = rig.temp_dir.join(".pi");
    let skill_path = rig.write(
        ".pi/skills/project-pi/SKILL.md",
        "---\nname: project-pi\ndescription: project pi\n---\n",
    );

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    let skill = result
        .skills
        .iter()
        .find(|resource| resource.path == skill_path.to_string_lossy())
        .expect("skill resolves");
    assert_eq!(skill.metadata.source, "auto");
    assert_eq!(skill.metadata.scope, SourceScope::Project);
    assert_eq!(
        skill.metadata.base_dir.as_deref(),
        Some(project_base_dir.to_string_lossy().as_ref())
    );
}

#[tokio::test]
async fn uses_the_agents_dir_as_base_dir_for_user_agents_skills() {
    let mut rig = Rig::new();
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: None,
        env: Some(home_env_lookup(&rig.temp_dir)),
        http_client: None,
    });
    let agents_base_dir = rig.temp_dir.join(".agents");
    let skill_path = rig.write(
        ".agents/skills/user-agents/SKILL.md",
        "---\nname: user-agents\ndescription: user agents\n---\n",
    );

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    let skill = result
        .skills
        .iter()
        .find(|resource| resource.path == skill_path.to_string_lossy())
        .expect("skill resolves");
    assert_eq!(skill.metadata.source, "auto");
    assert_eq!(skill.metadata.scope, SourceScope::User);
    assert_eq!(
        skill.metadata.base_dir.as_deref(),
        Some(agents_base_dir.to_string_lossy().as_ref())
    );
}

#[tokio::test]
async fn uses_each_project_agents_dir_as_base_dir() {
    let rig = Rig::new();
    let repo_root = rig.mkdir("repo");
    let nested_cwd = rig.mkdir("repo/packages/feature");
    std::fs::create_dir_all(repo_root.join(".git")).expect(".git dir");

    let repo_agents_base_dir = repo_root.join(".agents");
    let repo_skill = rig.write(
        "repo/.agents/skills/repo/SKILL.md",
        "---\nname: repo\ndescription: repo\n---\n",
    );
    let package_agents_base_dir = repo_root.join("packages/.agents");
    let package_skill = rig.write(
        "repo/packages/.agents/skills/package/SKILL.md",
        "---\nname: package\ndescription: package\n---\n",
    );

    let mut rig = rig;
    rig.with_cwd(&nested_cwd);
    let result = rig.package_manager.resolve(None).await.expect("resolve");
    let resolved_repo = result
        .skills
        .iter()
        .find(|resource| resource.path == repo_skill.to_string_lossy())
        .expect("repo skill");
    let resolved_package = result
        .skills
        .iter()
        .find(|resource| resource.path == package_skill.to_string_lossy())
        .expect("package skill");
    assert_eq!(resolved_repo.metadata.source, "auto");
    assert_eq!(resolved_repo.metadata.scope, SourceScope::Project);
    assert_eq!(
        resolved_repo.metadata.base_dir.as_deref(),
        Some(repo_agents_base_dir.to_string_lossy().as_ref())
    );
    assert_eq!(resolved_package.metadata.scope, SourceScope::Project);
    assert_eq!(
        resolved_package.metadata.base_dir.as_deref(),
        Some(package_agents_base_dir.to_string_lossy().as_ref())
    );
}
// =============================================================================
// .agents/skills auto-discovery
// =============================================================================

#[tokio::test]
async fn scans_agents_skills_from_the_cwd_up_to_the_git_root() {
    let rig = Rig::new();
    let repo_root = rig.mkdir("repo");
    let nested_cwd = rig.mkdir("repo/packages/feature");
    std::fs::create_dir_all(repo_root.join(".git")).expect(".git dir");

    let above_repo_skill = rig.write(
        ".agents/skills/above-repo/SKILL.md",
        "---\nname: above-repo\ndescription: above\n---\n",
    );
    let repo_root_skill = rig.write(
        "repo/.agents/skills/repo-root/SKILL.md",
        "---\nname: repo-root\ndescription: repo\n---\n",
    );
    let nested_skill = rig.write(
        "repo/packages/.agents/skills/nested/SKILL.md",
        "---\nname: nested\ndescription: nested\n---\n",
    );

    let mut rig = rig;
    rig.with_cwd(&nested_cwd);
    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(
        result
            .skills
            .iter()
            .any(|resource| resource.path == repo_root_skill.to_string_lossy() && resource.enabled)
    );
    assert!(
        result
            .skills
            .iter()
            .any(|resource| resource.path == nested_skill.to_string_lossy() && resource.enabled)
    );
    assert!(
        !result
            .skills
            .iter()
            .any(|resource| resource.path == above_repo_skill.to_string_lossy())
    );
}

#[tokio::test]
async fn scans_agents_skills_up_to_the_filesystem_root_outside_a_repo() {
    let rig = Rig::new();
    let _non_repo_root = rig.mkdir("non-repo");
    let nested_cwd = rig.mkdir("non-repo/a/b");

    let root_skill = rig.write(
        "non-repo/.agents/skills/root/SKILL.md",
        "---\nname: root\ndescription: root\n---\n",
    );
    let middle_skill = rig.write(
        "non-repo/a/.agents/skills/middle/SKILL.md",
        "---\nname: middle\ndescription: middle\n---\n",
    );

    let mut rig = rig;
    rig.with_cwd(&nested_cwd);
    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(
        result
            .skills
            .iter()
            .any(|resource| resource.path == root_skill.to_string_lossy() && resource.enabled)
    );
    assert!(
        result
            .skills
            .iter()
            .any(|resource| resource.path == middle_skill.to_string_lossy() && resource.enabled)
    );
}

#[tokio::test]
async fn ignores_root_markdown_in_agents_skills_but_discovers_nested_skills() {
    let rig = Rig::new();
    rig.write(
        ".agents/skills/root-file.md",
        "---\nname: root-file\ndescription: Root markdown file\n---\n",
    );
    let nested_skill = rig.write(
        ".agents/skills/nested-skill/SKILL.md",
        "---\nname: nested-skill\ndescription: Nested skill\n---\n",
    );
    let nested_markdown_skill = rig.write(
        ".agents/skills/third-party/child-skill.md",
        "---\nname: child-skill\ndescription: Nested markdown skill\n---\n",
    );
    let deeply_nested = rig.write(
        ".agents/skills/third-party/vendor/pack/deep-skill.md",
        "---\nname: deep-skill\ndescription: Deep markdown skill\n---\n",
    );

    let work_cwd = rig.mkdir("work");
    let mut rig = rig;
    rig.with_cwd(&work_cwd);
    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(!result.skills.iter().any(|resource| {
        resource.path
            == rig
                .temp_dir
                .join(".agents/skills/root-file.md")
                .to_string_lossy()
    }));
    assert!(
        result
            .skills
            .iter()
            .any(|resource| resource.path == nested_skill.to_string_lossy() && resource.enabled)
    );
    assert!(result.skills.iter().any(|resource| resource.path
        == nested_markdown_skill.to_string_lossy()
        && resource.enabled));
    assert!(
        result
            .skills
            .iter()
            .any(|resource| resource.path == deeply_nested.to_string_lossy() && resource.enabled)
    );
}

#[tokio::test]
async fn keeps_the_user_agents_skills_user_scoped_when_the_cwd_sits_under_home() {
    // The home seam rides the manager's env; the crate resolves `~/.agents`
    // through the injected HOME. The cwd nests under the temp home.
    let rig = Rig::new();
    let cwd = rig.mkdir("scratch/nested");
    let local_agent_dir = rig.mkdir(".pi/agent");

    let home_skill = rig.write(
        ".agents/skills/home-skill/SKILL.md",
        "---\nname: home-skill\ndescription: home\n---\n",
    );

    let mut rig = rig;
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: cwd.to_string_lossy().into_owned(),
        agent_dir: local_agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: None,
        env: Some(home_env_lookup(&rig.temp_dir)),
        http_client: None,
    });
    let result = rig.package_manager.resolve(None).await.expect("resolve");
    let matching: Vec<_> = result
        .skills
        .iter()
        .filter(|resource| resource.path == home_skill.to_string_lossy())
        .collect();
    assert_eq!(matching.len(), 1);
    assert!(matching[0].enabled);
    assert_eq!(matching[0].metadata.scope, SourceScope::User);
    assert_eq!(matching[0].metadata.source, "auto");
}

/// The env lookup with HOME pointed at the temp root, the restated
/// `process.env.HOME = tempDir` stub.
fn home_env_lookup(root: &Path) -> EnvLookup {
    let home = root.to_string_lossy().into_owned();
    Box::new(move |key: &str| {
        if key == "HOME" {
            return Some(home.clone());
        }
        None
    })
}

#[tokio::test]
async fn dedupes_user_skill_entries_when_the_agent_skills_dir_symlinks_the_agents_dir() {
    let rig = Rig::new();
    let agents_skills_dir = rig.mkdir(".agents/skills");
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(agents_skills_dir.clone(), rig.agent_dir.join("skills"))
            .expect("symlink");
    }
    let skill_path = rig.write(
        ".agents/skills/foo/SKILL.md",
        "---\nname: foo\ndescription: foo\n---\n",
    );

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert_eq!(
        result
            .skills
            .iter()
            .filter(|resource| resource.path.replace('\\', "/").ends_with("foo/SKILL.md"))
            .count(),
        1
    );
    let _ = skill_path;
}

// =============================================================================
// Ignore files
// =============================================================================

#[tokio::test]
async fn respects_a_gitignore_in_skill_directories() {
    let rig = Rig::new();
    rig.write("agent/skills/.gitignore", "venv\n__pycache__\n");
    rig.write(
        "agent/skills/good-skill/SKILL.md",
        "---\nname: good-skill\ndescription: Good\n---\nContent",
    );
    rig.write(
        "agent/skills/venv/bad-skill/SKILL.md",
        "---\nname: bad-skill\ndescription: Bad\n---\nContent",
    );
    rig.set_setting("skills", json!(["skills"]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_with(&result.skills, "good-skill"));
    assert!(!find_with(&result.skills, "venv"));
}

#[tokio::test]
async fn does_not_apply_the_parent_gitignore_to_pi_discovery() {
    let rig = Rig::new();
    rig.write(".gitignore", ".pi\n");
    let skill_path = rig.write(
        ".pi/skills/auto-skill/SKILL.md",
        "---\nname: auto-skill\ndescription: Auto\n---\nContent",
    );

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(
        result
            .skills
            .iter()
            .any(|resource| resource.path == skill_path.to_string_lossy() && resource.enabled)
    );
}

// =============================================================================
// resolveExtensionSources
// =============================================================================

#[tokio::test]
async fn resolves_extension_sources_from_local_paths() {
    let rig = Rig::new();
    let ext_path = rig.write_executable("ext");

    let result = rig
        .package_manager
        .resolve_extension_sources(&[ext_path.to_string_lossy().into_owned()], false, false)
        .await
        .expect("resolve");
    assert!(
        result
            .extensions
            .iter()
            .any(|resource| resource.path == ext_path.to_string_lossy() && resource.enabled)
    );
}

#[tokio::test]
async fn resolves_extension_sources_from_directories_with_a_pi_manifest() {
    let rig = Rig::new();
    let pkg_dir = rig.mkdir("my-package");
    rig.write_executable("my-package/src/index");
    rig.write(
        "my-package/skills/my-skill/SKILL.md",
        "---\nname: my-skill\ndescription: Test\n---\nContent",
    );
    rig.write(
        "my-package/package.json",
        r#"{"name":"my-package","pi":{"extensions":["./src/index"],"skills":["./skills"]}}"#,
    );

    let result = rig
        .package_manager
        .resolve_extension_sources(&[pkg_dir.to_string_lossy().into_owned()], false, false)
        .await
        .expect("resolve");
    let debug_paths: Vec<String> = result
        .extensions
        .iter()
        .map(|resource| resource.path.clone())
        .collect();
    assert!(
        result.extensions.iter().any(|resource| resource.path
            == pkg_dir.join("src/index").to_string_lossy()
            && resource.enabled),
        "extensions were {debug_paths:?}"
    );
    assert!(result.skills.iter().any(|resource| resource.path
        == pkg_dir.join("skills/my-skill/SKILL.md").to_string_lossy()
        && resource.enabled));
}

#[tokio::test]
async fn keeps_pi_manifest_entries_with_leading_tilde_package_relative() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("tilde-manifest-package");
    let _direct_extension_path = rig.write_executable("tilde-manifest-package/~extensions/main");
    let slash_extension_path =
        rig.write_executable("tilde-manifest-package/~extensions/../~/extensions-alt");
    let _ = slash_extension_path;
    let slash_extension_path = rig.write_executable("tilde-manifest-package/~extensions/alt");
    let _ = slash_extension_path;
    let direct_extension_path = rig.write_executable("tilde-manifest-package/~extensions/main");
    let direct_skill_path = rig.write(
        "tilde-manifest-package/~skills/direct-skill/SKILL.md",
        "---\nname: direct-skill\ndescription: Direct\n---\nContent",
    );
    let slash_skill_path = rig.write(
        "tilde-manifest-package/~skills/slash-skill/SKILL.md",
        "---\nname: slash-skill\ndescription: Slash\n---\nContent",
    );
    rig.write(
        "tilde-manifest-package/package.json",
        r#"{"name":"tilde-manifest-package","pi":{"extensions":["~extensions/main","~/extensions/main-alt"],"skills":["~skills","~/skills"]}}"#,
    );
    let _ = (direct_extension_path.clone(), direct_skill_path.clone());

    let result = rig
        .package_manager
        .resolve_extension_sources(&[pkg_dir.to_string_lossy().into_owned()], false, false)
        .await
        .expect("resolve");
    assert!(result.extensions.iter().any(|resource| resource.path
        == pkg_dir.join("~extensions/main").to_string_lossy()
        && resource.enabled));
    assert!(
        result.skills.iter().any(
            |resource| resource.path == direct_skill_path.to_string_lossy() && resource.enabled
        )
    );
    assert!(
        result.skills.iter().any(
            |resource| resource.path == slash_skill_path.to_string_lossy() && resource.enabled
        )
    );
}

#[tokio::test]
async fn resolves_extension_sources_from_auto_discovery_layouts() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("auto-pkg");
    rig.write_executable("auto-pkg/extensions/main");
    rig.write("auto-pkg/themes/dark.json", "{}");

    let result = rig
        .package_manager
        .resolve_extension_sources(&[pkg_dir.to_string_lossy().into_owned()], false, false)
        .await
        .expect("resolve");
    assert!(find_enabled(&result.extensions, "main") || find_with(&result.extensions, "main"));
    assert!(find_enabled(&result.themes, "dark.json"));
}

#[tokio::test]
async fn stops_recursing_when_a_package_skill_directory_contains_skill_md() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("skill-root-pkg");
    let root_skill = rig.write(
        "skill-root-pkg/skills/root-skill/SKILL.md",
        "---\nname: root-skill\ndescription: Root skill\n---\n",
    );
    let nested_skill = rig.write(
        "skill-root-pkg/skills/root-skill/nested-skill/SKILL.md",
        "---\nname: nested-skill\ndescription: Nested skill\n---\n",
    );

    let result = rig
        .package_manager
        .resolve_extension_sources(&[pkg_dir.to_string_lossy().into_owned()], false, false)
        .await
        .expect("resolve");
    assert!(
        result
            .skills
            .iter()
            .any(|resource| resource.path == root_skill.to_string_lossy() && resource.enabled)
    );
    assert!(
        !result
            .skills
            .iter()
            .any(|resource| resource.path == nested_skill.to_string_lossy())
    );
}

// =============================================================================
// Progress, command spawning, source parsing
// =============================================================================

#[tokio::test]
async fn progress_callback_stays_quiet_for_local_sources() {
    let rig = Rig::new();
    let events: Arc<Mutex<Vec<ProgressEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    let mut rig = rig;
    rig.package_manager
        .set_progress_callback(Some(Box::new(move |event| {
            sink.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(event.clone());
        })));
    let ext_path = rig.write_executable("ext");
    rig.package_manager
        .resolve_extension_sources(&[ext_path.to_string_lossy().into_owned()], false, false)
        .await
        .expect("resolve");
    assert!(
        events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    );
}

#[tokio::test]
async fn emits_progress_events_on_an_install_failure() {
    let mut rig = Rig::new();
    let events: Arc<Mutex<Vec<ProgressEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    rig.with_runner(FakeRunner::rejecting("simulated cargo install failure"));
    rig.package_manager
        .set_progress_callback(Some(Box::new(move |event| {
            sink.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(event.clone());
        })));

    let error = rig
        .package_manager
        .install("crate:nonexistent-package@1.0.0", false)
        .await
        .expect_err("install fails");
    assert_eq!(error.0, "simulated cargo install failure");

    let events = events
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert!(
        events
            .iter()
            .any(|event| event.event_type == ProgressEventType::Start
                && event.action == ProgressAction::Install)
    );
    assert!(
        events
            .iter()
            .any(|event| event.event_type == ProgressEventType::Error)
    );
}

#[tokio::test]
async fn installs_github_urls_without_a_prefix_through_the_git_channel() {
    let mut rig = Rig::new();
    let target_dir = rig.agent_dir.join("git/github.com/nonexistent/repo");
    let recorded_target = Arc::new(Mutex::new(None::<String>));
    let sink = Arc::clone(&recorded_target);
    rig.with_runner(FakeRunner::new(Box::new(move |command, args, _options| {
        if command == "git" && args.first().map(String::as_str) == Some("clone") {
            let target = args.get(2).cloned().unwrap_or_default();
            std::fs::create_dir_all(&target).expect("target dir");
            *sink
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(target);
        }
        Err(PackageManagerError("simulated git failure".to_string()))
    })));

    let error = rig
        .package_manager
        .install("https://github.com/nonexistent/repo", false)
        .await
        .expect_err("git install fails");
    assert_eq!(error.0, "simulated git failure");
    assert!(
        recorded_target
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_some()
    );
    let _ = target_dir;
}

#[tokio::test]
async fn parses_package_source_types() {
    let rig = Rig::new();
    let crate_pinned = rig
        .package_manager
        .parse_source("crate:pi-llm-tools@1.2.3")
        .expect("parses");
    match crate_pinned {
        ParsedSource::Crate(parsed) => {
            assert!(parsed.pinned);
            assert_eq!(parsed.name, "pi-llm-tools");
            assert_eq!(parsed.version.as_deref(), Some("1.2.3"));
        }
        _ => panic!("expected a crate source"),
    }
    let crate_floating = rig
        .package_manager
        .parse_source("crate:pi-llm-tools")
        .expect("parses");
    match crate_floating {
        ParsedSource::Crate(parsed) => {
            assert!(!parsed.pinned);
            assert_eq!(parsed.name, "pi-llm-tools");
        }
        _ => panic!("expected a crate source"),
    }
    for source in [
        "git:github.com/user/repo@v1",
        "github:user/repo@v1",
        "https://github.com/user/repo@v1",
        "git:git@github.com:user/repo@v1",
        "ssh://git@github.com/user/repo@v1",
    ] {
        assert!(
            matches!(
                rig.package_manager.parse_source(source).expect("parses"),
                ParsedSource::Git(_)
            ),
            "{source} parses as git"
        );
    }
    for source in [
        "/absolute/path/to/package",
        "./relative/path/to/package",
        "../relative/path/to/package",
    ] {
        assert!(
            matches!(
                rig.package_manager.parse_source(source).expect("parses"),
                ParsedSource::Local(_)
            ),
            "{source} parses as local"
        );
    }
    assert!(rig.package_manager.parse_source("npm:@scope/pkg").is_err());
}

#[tokio::test]
async fn never_parses_dot_relative_paths_as_git() {
    let rig = Rig::new();
    let dot_slash = rig
        .package_manager
        .parse_source("./packages/agent-timers")
        .expect("parses");
    match dot_slash {
        ParsedSource::Local(local) => assert_eq!(local.path, "./packages/agent-timers"),
        _ => panic!("expected local"),
    }
    let dot_dot = rig
        .package_manager
        .parse_source("../packages/agent-timers")
        .expect("parses");
    match dot_dot {
        ParsedSource::Local(local) => assert_eq!(local.path, "../packages/agent-timers"),
        _ => panic!("expected local"),
    }
}

#[tokio::test]
async fn rejects_paths_outside_the_git_install_roots() {
    let rig = Rig::new();
    // Upstream builds the traversal source as a raw object literal — the
    // parser's unsafe-part check would reject it before the gate runs.
    let traversal = pi_coding_agent::utils::git::GitSource {
        repo: "git@evil.example:../../victim/repo".to_string(),
        host: "evil.example".to_string(),
        path: "../../victim/repo".to_string(),
        r#ref: None,
        pinned: false,
    };
    for scope in [
        SourceScope::User,
        SourceScope::Project,
        SourceScope::Temporary,
    ] {
        let error = rig
            .package_manager
            .get_git_install_path(&traversal, scope)
            .expect_err("traversal rejects");
        assert!(
            error.0.contains("outside package install root"),
            "{scope:?}"
        );
    }
}

#[tokio::test]
async fn places_temporary_crate_packages_under_the_agent_temp_extension_folder() {
    let rig = Rig::new();
    let parsed = rig
        .package_manager
        .parse_source("crate:left-pad")
        .expect("parses");
    let ParsedSource::Crate(source) = parsed else {
        panic!("expected crate");
    };
    let install_path = rig
        .package_manager
        .get_installed_path(&format!("crate:{}", source.name), SourceScope::Temporary)
        .expect("path")
        .unwrap_or_else(|| {
            // The path only reports when it exists; compute the shape the
            // temporary dir contract pins.
            let temp_root =
                pi_coding_agent::package_manager::get_extension_temp_folder(&rig.agent_dir)
                    .expect("temp");
            temp_root
                .join("crate")
                .join("left-pad")
                .to_string_lossy()
                .into_owned()
        });
    assert!(install_path.replace('\\', "/").ends_with("left-pad"));
    let temp_root = rig.agent_dir.join("tmp/extensions");
    assert!(
        !install_path.starts_with(
            std::env::temp_dir()
                .join("pi-extensions")
                .to_string_lossy()
                .as_ref()
        )
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&temp_root)
            .expect("temp root")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o700);
    }
}

// =============================================================================
// Settings source normalization
// =============================================================================

#[tokio::test]
async fn stores_global_local_packages_relative_to_the_agent_settings_base() {
    let rig = Rig::new();
    let pkg_dir = rig.mkdir("packages/local-global-pkg");
    rig.write_executable("packages/local-global-pkg/extensions/index");

    let added = rig
        .package_manager
        .add_source_to_settings("./packages/local-global-pkg", false)
        .expect("added");
    assert!(added);

    let packages = rig.global_packages();
    assert_eq!(packages.len(), 1);
    let stored = packages[0].as_str().expect("string entry");
    let resolved = std::fs::canonicalize(rig.agent_dir.join(stored)).expect("resolve");
    assert_eq!(resolved, std::fs::canonicalize(&pkg_dir).expect("resolve"));
}

#[tokio::test]
async fn stores_project_local_packages_relative_to_the_pi_settings_base() {
    let rig = Rig::new();
    let project_pkg_dir = rig.mkdir("project-local-pkg");
    rig.write_executable("project-local-pkg/extensions/index");

    let added = rig
        .package_manager
        .add_source_to_settings("./project-local-pkg", true)
        .expect("added");
    assert!(added);

    let packages = rig.global_packages();
    let _ = packages;
    let project = rig
        .settings
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get_project_settings();
    let entries = project
        .get("packages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    assert_eq!(entries.len(), 1);
    let stored = entries[0].as_str().expect("string entry");
    let resolved = std::fs::canonicalize(rig.temp_dir.join(".pi").join(stored)).expect("resolve");
    assert_eq!(
        resolved,
        std::fs::canonicalize(&project_pkg_dir).expect("resolve")
    );
}

#[tokio::test]
async fn removes_local_package_entries_using_equivalent_path_forms() {
    let rig = Rig::new();
    rig.mkdir("remove-local-pkg");
    rig.write_executable("remove-local-pkg/extensions/index");

    rig.package_manager
        .add_source_to_settings("./remove-local-pkg", false)
        .expect("added");
    let removed = rig
        .package_manager
        .remove_source_from_settings(
            &format!(
                "{}/",
                rig.temp_dir.join("remove-local-pkg").to_string_lossy()
            ),
            false,
        )
        .expect("removed");
    assert!(removed);
    assert!(rig.global_packages().is_empty());
}

#[tokio::test]
async fn returns_false_when_adding_the_same_git_source_with_the_same_ref() {
    let rig = Rig::new();
    assert!(
        rig.package_manager
            .add_source_to_settings("git:github.com/user/repo@v1", false)
            .expect("added")
    );
    assert!(
        !rig.package_manager
            .add_source_to_settings("git:github.com/user/repo@v1", false)
            .expect("dedup")
    );
    assert_eq!(
        rig.global_packages(),
        vec![json!("git:github.com/user/repo@v1")]
    );
}

#[tokio::test]
async fn updates_the_ref_when_adding_the_same_git_source_with_a_different_ref() {
    let rig = Rig::new();
    rig.package_manager
        .add_source_to_settings("git:github.com/user/repo@v1", false)
        .expect("added");
    assert!(
        rig.package_manager
            .add_source_to_settings("git:github.com/user/repo@v2", false)
            .expect("updated")
    );
    assert_eq!(
        rig.global_packages(),
        vec![json!("git:github.com/user/repo@v2")]
    );
}

#[tokio::test]
async fn preserves_package_filters_when_replacing_a_source_ref() {
    let rig = Rig::new();
    rig.set_packages(&json!([
        {
            "source": "git:github.com/user/repo@v1",
            "extensions": ["extensions/main"],
            "skills": [],
            "prompts": ["prompts/review.md"],
            "themes": ["themes/dark.json"],
        }
    ]));

    assert!(
        rig.package_manager
            .add_source_to_settings("git:github.com/user/repo@v2", false)
            .expect("updated")
    );
    assert_eq!(
        rig.global_packages(),
        vec![json!({
            "source": "git:github.com/user/repo@v2",
            "extensions": ["extensions/main"],
            "skills": [],
            "prompts": ["prompts/review.md"],
            "themes": ["themes/dark.json"],
        })]
    );
}

// =============================================================================
// Pattern filtering in the top-level arrays
// =============================================================================

#[tokio::test]
async fn excludes_extensions_with_a_bang_pattern() {
    let rig = Rig::new();
    rig.write_executable("agent/extensions/keep");
    rig.write_executable("agent/extensions/remove");
    rig.set_setting("extensions", json!(["extensions", "!**/remove"]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_enabled(&result.extensions, "keep"));
    assert!(find_disabled(&result.extensions, "remove"));
}

#[tokio::test]
async fn filters_themes_with_glob_patterns() {
    let rig = Rig::new();
    rig.write("agent/themes/dark.json", "{}");
    rig.write("agent/themes/light.json", "{}");
    rig.write("agent/themes/funky.json", "{}");
    rig.set_setting("themes", json!(["themes", "!funky.json"]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_enabled(&result.themes, "dark.json"));
    assert!(find_enabled(&result.themes, "light.json"));
    assert!(find_disabled(&result.themes, "funky.json"));
}

#[tokio::test]
async fn filters_prompts_with_an_exclusion_pattern() {
    let rig = Rig::new();
    rig.write("agent/prompts/review.md", "Review code");
    rig.write("agent/prompts/explain.md", "Explain code");
    rig.set_setting("prompts", json!(["prompts", "!explain.md"]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_enabled(&result.prompts, "review.md"));
    assert!(find_disabled(&result.prompts, "explain.md"));
}

#[tokio::test]
async fn filters_skills_with_an_exclusion_pattern() {
    let rig = Rig::new();
    rig.write(
        "agent/skills/good-skill/SKILL.md",
        "---\nname: good-skill\ndescription: Good\n---\nContent",
    );
    rig.write(
        "agent/skills/bad-skill/SKILL.md",
        "---\nname: bad-skill\ndescription: Bad\n---\nContent",
    );
    rig.set_setting("skills", json!(["skills", "!**/bad-skill"]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_with(&result.skills, "good-skill"));
    assert!(result.skills.iter().any(|resource| {
        resource.path.replace('\\', "/").contains("bad-skill") && !resource.enabled
    }));
}

#[tokio::test]
async fn works_without_patterns() {
    let rig = Rig::new();
    let ext_path = rig.write_executable("agent/extensions/my-ext");
    rig.set_setting("extensions", json!(["extensions/my-ext"]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(
        result
            .extensions
            .iter()
            .any(|resource| resource.path == ext_path.to_string_lossy() && resource.enabled)
    );
}

// =============================================================================
// Pattern filtering in the pi manifest
// =============================================================================

#[tokio::test]
async fn supports_glob_patterns_in_manifest_extensions() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("manifest-pkg");
    rig.write_executable("manifest-pkg/extensions/local");
    rig.write_executable("manifest-pkg/node_modules/dep/extensions/remote");
    rig.write_executable("manifest-pkg/node_modules/dep/extensions/skip");
    rig.write(
        "manifest-pkg/package.json",
        r#"{"name":"manifest-pkg","pi":{"extensions":["extensions","node_modules/dep/extensions","!**/skip"]}}"#,
    );

    let result = rig
        .package_manager
        .resolve_extension_sources(&[pkg_dir.to_string_lossy().into_owned()], false, false)
        .await
        .expect("resolve");
    assert!(find_enabled(&result.extensions, "local"));
    assert!(find_enabled(&result.extensions, "remote"));
    assert!(!find_with(&result.extensions, "skip"));
}

#[tokio::test]
async fn supports_glob_patterns_in_manifest_skills() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("skill-manifest-pkg");
    rig.write(
        "skill-manifest-pkg/skills/good-skill/SKILL.md",
        "---\nname: good-skill\ndescription: Good\n---\nContent",
    );
    rig.write(
        "skill-manifest-pkg/skills/bad-skill/SKILL.md",
        "---\nname: bad-skill\ndescription: Bad\n---\nContent",
    );
    rig.write(
        "skill-manifest-pkg/package.json",
        r#"{"name":"skill-manifest-pkg","pi":{"skills":["skills","!**/bad-skill"]}}"#,
    );

    let result = rig
        .package_manager
        .resolve_extension_sources(&[pkg_dir.to_string_lossy().into_owned()], false, false)
        .await
        .expect("resolve");
    assert!(find_with(&result.skills, "good-skill"));
    assert!(
        !result
            .skills
            .iter()
            .any(|resource| resource.path.contains("bad-skill"))
    );
}

#[tokio::test]
async fn expands_positive_glob_manifest_entries_before_collecting_skills() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("skill-manifest-glob-pkg");
    rig.write(
        "skill-manifest-glob-pkg/plugins/pdf-to-markdown/skills/pdf-to-markdown/SKILL.md",
        "---\nname: pdf-to-markdown\ndescription: PDF to Markdown\n---\nContent",
    );
    rig.write(
        "skill-manifest-glob-pkg/plugins/nutrient-dws/skills/document-processor-api/SKILL.md",
        "---\nname: document-processor-api\ndescription: DWS\n---\nContent",
    );
    rig.write(
        "skill-manifest-glob-pkg/package.json",
        r#"{"name":"skill-manifest-glob-pkg","pi":{"skills":["./plugins/*/skills"]}}"#,
    );

    let result = rig
        .package_manager
        .resolve_extension_sources(&[pkg_dir.to_string_lossy().into_owned()], false, false)
        .await
        .expect("resolve");
    assert!(find_with(&result.skills, "pdf-to-markdown/SKILL.md"));
    assert!(find_with(&result.skills, "document-processor-api/SKILL.md"));
}

#[tokio::test]
async fn sorts_manifest_glob_matches_and_uses_exact_entries_for_dot_paths() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("manifest-glob-semantics-pkg");
    let _linked_source = rig.mkdir("manifest-glob-semantics-pkg/linked-plugin-source");
    rig.write_executable("manifest-glob-semantics-pkg/extension-files/z");
    rig.write_executable("manifest-glob-semantics-pkg/extension-files/a");
    std::fs::write(
        rig.temp_dir
            .join("manifest-glob-semantics-pkg/extension-files/.ignored"),
        "stub",
    )
    .expect("write");
    rig.write_executable("manifest-glob-semantics-pkg/extension-files/nested/.hidden");
    rig.write_executable("manifest-glob-semantics-pkg/extension-groups/group/index");
    rig.write(
        "manifest-glob-semantics-pkg/plugins/local/skills/local-skill/SKILL.md",
        "---\nname: local-skill\ndescription: Local\n---\n",
    );
    rig.write(
        "manifest-glob-semantics-pkg/linked-plugin-source/skills/linked-skill/SKILL.md",
        "---\nname: linked-skill\ndescription: Linked\n---\n",
    );
    #[cfg(unix)]
    std::os::unix::fs::symlink(
        pkg_dir.join("linked-plugin-source"),
        pkg_dir.join("plugins/linked"),
    )
    .expect("symlink");
    rig.write(
        "manifest-glob-semantics-pkg/package.json",
        r#"{"name":"manifest-glob-semantics-pkg","pi":{"extensions":["./extension-files/*","./extension-groups/*/"],"skills":["./plugins/*/skills","./plugins/linked/skills"]}}"#,
    );

    let result = rig
        .package_manager
        .resolve_extension_sources(&[pkg_dir.to_string_lossy().into_owned()], false, false)
        .await
        .expect("resolve");
    let relative: Vec<String> = result
        .extensions
        .iter()
        .map(|resource| {
            Path::new(&resource.path)
                .strip_prefix(&pkg_dir)
                .expect("under pkg")
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(
        relative,
        vec![
            "extension-files/a",
            "extension-files/z",
            "extension-groups/group/index",
        ]
    );
    assert!(find_with(&result.skills, "local-skill/SKILL.md"));
    assert!(find_with(&result.skills, "linked-skill/SKILL.md"));
}

// =============================================================================
// Pattern filtering in package filters
// =============================================================================

#[tokio::test]
async fn applies_user_filters_on_top_of_manifest_filters() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("layered-pkg");
    rig.write_executable("layered-pkg/extensions/foo");
    rig.write_executable("layered-pkg/extensions/bar");
    rig.write_executable("layered-pkg/extensions/baz");
    rig.write(
        "layered-pkg/package.json",
        r#"{"name":"layered-pkg","pi":{"extensions":["extensions","!**/baz"]}}"#,
    );
    rig.set_packages(&json!([
        {
            "source": pkg_dir.to_string_lossy().into_owned(),
            "extensions": ["!**/bar"],
            "skills": [],
            "prompts": [],
            "themes": [],
        }
    ]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_enabled(&result.extensions, "foo"));
    assert!(find_disabled(&result.extensions, "bar"));
    assert!(!find_with(&result.extensions, "baz"));
}

#[tokio::test]
async fn excludes_package_extensions_with_a_bang_pattern() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("pattern-pkg");
    rig.write_executable("pattern-pkg/extensions/foo");
    rig.write_executable("pattern-pkg/extensions/bar");
    rig.write_executable("pattern-pkg/extensions/baz");
    rig.set_packages(&json!([
        {
            "source": pkg_dir.to_string_lossy().into_owned(),
            "extensions": ["!**/baz"],
            "skills": [],
            "prompts": [],
            "themes": [],
        }
    ]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_enabled(&result.extensions, "foo"));
    assert!(find_enabled(&result.extensions, "bar"));
    assert!(find_disabled(&result.extensions, "baz"));
}

#[tokio::test]
async fn filters_package_themes() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("theme-pkg");
    rig.write("theme-pkg/themes/nice.json", "{}");
    rig.write("theme-pkg/themes/ugly.json", "{}");
    rig.set_packages(&json!([
        {
            "source": pkg_dir.to_string_lossy().into_owned(),
            "extensions": [],
            "skills": [],
            "prompts": [],
            "themes": ["!ugly.json"],
        }
    ]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_enabled(&result.themes, "nice.json"));
    assert!(find_disabled(&result.themes, "ugly.json"));
}

#[tokio::test]
async fn combines_include_and_exclude_patterns() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("combo-pkg");
    rig.write_executable("combo-pkg/extensions/alpha");
    rig.write_executable("combo-pkg/extensions/beta");
    rig.write_executable("combo-pkg/extensions/gamma");
    rig.set_packages(&json!([
        {
            "source": pkg_dir.to_string_lossy().into_owned(),
            "extensions": ["**/alpha", "**/beta", "!**/beta"],
            "skills": [],
            "prompts": [],
            "themes": [],
        }
    ]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_enabled(&result.extensions, "alpha"));
    assert!(find_disabled(&result.extensions, "beta"));
    assert!(find_disabled(&result.extensions, "gamma"));
}

#[tokio::test]
async fn works_with_direct_paths_without_patterns() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("direct-pkg");
    rig.write_executable("direct-pkg/extensions/one");
    rig.write_executable("direct-pkg/extensions/two");
    rig.set_packages(&json!([
        {
            "source": pkg_dir.to_string_lossy().into_owned(),
            "extensions": ["extensions/one"],
            "skills": [],
            "prompts": [],
            "themes": [],
        }
    ]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_enabled(&result.extensions, "one"));
    assert!(find_disabled(&result.extensions, "two"));
}

#[tokio::test]
async fn resolves_autoload_disabled_project_entries_as_deltas_over_global_packages() {
    let rig = Rig::new();
    let pkg_dir = rig.agent_dir.join("crates/pi-tools");
    std::fs::create_dir_all(pkg_dir.join("extensions")).expect("extensions dir");
    rig.write(
        "agent/crates/pi-tools/package.json",
        r#"{"name":"pi-tools","version":"1.0.0"}"#,
    );
    let _ = &pkg_dir;
    rig.write_executable("agent/crates/pi-tools/extensions/foo");
    rig.write_executable("agent/crates/pi-tools/extensions/bar");
    rig.set_packages(&json!(["crate:pi-tools"]));
    rig.set_project_packages(&json!([
        { "source": "crate:pi-tools", "autoload": false, "extensions": ["-extensions/foo"] },
    ]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    let debug: Vec<(String, bool, String)> = result
        .extensions
        .iter()
        .map(|resource| {
            (
                resource.path.clone(),
                resource.enabled,
                format!("{:?}", resource.metadata.scope),
            )
        })
        .collect();
    let foo = result
        .extensions
        .iter()
        .find(|resource| same_path(&resource.path, &pkg_dir.join("extensions/foo")))
        .unwrap_or_else(|| panic!("foo entry missing, extensions were {debug:?}"));
    let bar = result
        .extensions
        .iter()
        .find(|resource| same_path(&resource.path, &pkg_dir.join("extensions/bar")))
        .expect("bar entry");
    assert!(!foo.enabled);
    assert_eq!(foo.metadata.scope, SourceScope::Project);
    assert!(bar.enabled);
    assert_eq!(bar.metadata.scope, SourceScope::User);
}

#[tokio::test]
async fn resolves_autoload_disabled_entries_as_positive_only_without_a_global_package() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("positive-only-pkg");
    rig.write_executable("positive-only-pkg/extensions/foo");
    rig.write_executable("positive-only-pkg/extensions/bar");
    rig.write("positive-only-pkg/skills/foo/SKILL.md", "# Foo\n");
    rig.set_project_packages(&json!([
        {
            "source": "../positive-only-pkg",
            "autoload": false,
            "extensions": ["+extensions/foo"],
        },
    ]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    let paths: Vec<String> = result
        .extensions
        .iter()
        .map(|resource| resource.path.clone())
        .collect();
    assert_eq!(paths.len(), 1);
    assert!(same_path(&paths[0], &pkg_dir.join("extensions/foo")));
    assert!(result.skills.is_empty());
}

// =============================================================================
// Force-include and force-exclude patterns
// =============================================================================

#[tokio::test]
async fn force_includes_extensions_with_a_plus_pattern_after_exclusion() {
    let rig = Rig::new();
    rig.write_executable("agent/extensions/keep");
    rig.write_executable("agent/extensions/excluded");
    rig.write_executable("agent/extensions/force-back");
    rig.set_setting(
        "extensions",
        json!(["extensions", "!extensions/*", "+extensions/force-back"]),
    );

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_disabled(&result.extensions, "keep"));
    assert!(find_disabled(&result.extensions, "excluded"));
    assert!(find_enabled(&result.extensions, "force-back"));
}

#[tokio::test]
async fn force_include_overrides_exclude_in_package_filters() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("force-pkg");
    rig.write_executable("force-pkg/extensions/alpha");
    rig.write_executable("force-pkg/extensions/beta");
    rig.write_executable("force-pkg/extensions/gamma");
    rig.set_packages(&json!([
        {
            "source": pkg_dir.to_string_lossy().into_owned(),
            "extensions": ["!**/*", "+extensions/beta"],
            "skills": [],
            "prompts": [],
            "themes": [],
        }
    ]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_disabled(&result.extensions, "alpha"));
    assert!(find_enabled(&result.extensions, "beta"));
    assert!(find_disabled(&result.extensions, "gamma"));
}

#[tokio::test]
async fn force_includes_multiple_resources() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("multi-force-pkg");
    rig.write(
        "multi-force-pkg/skills/skill-a/SKILL.md",
        "---\nname: skill-a\ndescription: A\n---\nContent",
    );
    rig.write(
        "multi-force-pkg/skills/skill-b/SKILL.md",
        "---\nname: skill-b\ndescription: B\n---\nContent",
    );
    rig.write(
        "multi-force-pkg/skills/skill-c/SKILL.md",
        "---\nname: skill-c\ndescription: C\n---\nContent",
    );
    rig.set_packages(&json!([
        {
            "source": pkg_dir.to_string_lossy().into_owned(),
            "extensions": [],
            "skills": ["!**/*", "+skills/skill-a", "+skills/skill-c"],
            "prompts": [],
            "themes": [],
        }
    ]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_with(&result.skills, "skill-a/SKILL.md"));
    assert!(find_disabled(&result.skills, "skill-b/SKILL.md"));
    assert!(find_with(&result.skills, "skill-c/SKILL.md"));
}

#[tokio::test]
async fn force_includes_after_a_specific_exclusion() {
    let rig = Rig::new();
    rig.write_executable("agent/extensions/a");
    rig.write_executable("agent/extensions/b");
    rig.set_setting(
        "extensions",
        json!(["extensions", "!extensions/b", "+extensions/b"]),
    );

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_enabled(&result.extensions, "a"));
    assert!(find_enabled(&result.extensions, "b"));
}

#[tokio::test]
async fn handles_force_include_in_manifest_patterns() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("manifest-force-pkg");
    rig.write_executable("manifest-force-pkg/extensions/one");
    rig.write_executable("manifest-force-pkg/extensions/two");
    rig.write_executable("manifest-force-pkg/extensions/three");
    rig.write(
        "manifest-force-pkg/package.json",
        r#"{"name":"manifest-force-pkg","pi":{"extensions":["extensions","!**/two","+extensions/two"]}}"#,
    );

    let result = rig
        .package_manager
        .resolve_extension_sources(&[pkg_dir.to_string_lossy().into_owned()], false, false)
        .await
        .expect("resolve");
    assert!(find_enabled(&result.extensions, "one"));
    assert!(find_enabled(&result.extensions, "two"));
    assert!(find_enabled(&result.extensions, "three"));
}

#[tokio::test]
async fn force_includes_themes_and_prompts() {
    let rig = Rig::new();
    rig.write("agent/themes/dark.json", "{}");
    rig.write("agent/themes/light.json", "{}");
    rig.write("agent/themes/special.json", "{}");
    rig.set_setting(
        "themes",
        json!(["themes", "!themes/*.json", "+themes/special.json"]),
    );

    rig.write("agent/prompts/review.md", "Review");
    rig.write("agent/prompts/explain.md", "Explain");
    rig.write("agent/prompts/debug.md", "Debug");
    rig.set_setting(
        "prompts",
        json!(["prompts", "!prompts/*.md", "+prompts/debug.md"]),
    );

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_disabled(&result.themes, "dark.json"));
    assert!(find_disabled(&result.themes, "light.json"));
    assert!(find_enabled(&result.themes, "special.json"));
    assert!(find_disabled(&result.prompts, "review.md"));
    assert!(find_disabled(&result.prompts, "explain.md"));
    assert!(find_enabled(&result.prompts, "debug.md"));
}

#[tokio::test]
async fn force_excludes_top_level_resources() {
    let rig = Rig::new();
    rig.write_executable("agent/extensions/alpha");
    rig.write_executable("agent/extensions/beta");
    rig.set_setting(
        "extensions",
        json!(["extensions", "+extensions/alpha", "-extensions/alpha"]),
    );

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_disabled(&result.extensions, "alpha"));
    assert!(find_enabled(&result.extensions, "beta"));
}

#[tokio::test]
async fn force_excludes_in_package_filters() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("force-exclude-pkg");
    rig.write_executable("force-exclude-pkg/extensions/alpha");
    rig.write_executable("force-exclude-pkg/extensions/beta");
    rig.set_packages(&json!([
        {
            "source": pkg_dir.to_string_lossy().into_owned(),
            "extensions": ["extensions/*", "+extensions/alpha", "-extensions/alpha"],
            "skills": [],
            "prompts": [],
            "themes": [],
        }
    ]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_disabled(&result.extensions, "alpha"));
    assert!(find_enabled(&result.extensions, "beta"));
}

// =============================================================================
// Package deduplication
// =============================================================================

#[tokio::test]
async fn dedupes_the_same_local_package_in_global_and_project() {
    let rig = Rig::new();
    let pkg_dir = rig.mkdir("shared-pkg");
    rig.write_executable("shared-pkg/extensions/shared");

    rig.set_packages(&json!([pkg_dir.to_string_lossy().into_owned()]));
    rig.set_project_packages(&json!([pkg_dir.to_string_lossy().into_owned()]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    let shared: Vec<_> = result
        .extensions
        .iter()
        .filter(|resource| resource.path.contains("shared-pkg"))
        .collect();
    assert_eq!(shared.len(), 1);
    assert_eq!(shared[0].metadata.scope, SourceScope::Project);
}

#[tokio::test]
async fn keeps_different_packages() {
    let rig = Rig::new();
    let pkg1 = rig.mkdir("pkg1");
    let pkg2 = rig.mkdir("pkg2");
    rig.write_executable("pkg1/extensions/from-pkg1");
    rig.write_executable("pkg2/extensions/from-pkg2");

    rig.set_packages(&json!([pkg1.to_string_lossy().into_owned()]));
    rig.set_project_packages(&json!([pkg2.to_string_lossy().into_owned()]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(
        result
            .extensions
            .iter()
            .any(|resource| resource.path.contains("pkg1"))
    );
    assert!(
        result
            .extensions
            .iter()
            .any(|resource| resource.path.contains("pkg2"))
    );
}

fn identity_of(rig: &Rig, source: &str) -> String {
    // The identity is the settings match key for the user scope, the shape
    // the dedupe and suggestion surfaces share.
    match rig.package_manager.parse_source(source).expect("parses") {
        ParsedSource::Git(parsed) => format!("git:{}/{}", parsed.host, parsed.path),
        _ => panic!("expected git for {source}"),
    }
}

#[tokio::test]
async fn dedupes_ssh_and_https_urls_for_the_same_repo() {
    let rig = Rig::new();
    let https_identity = identity_of(&rig, "https://github.com/user/repo");
    let ssh_identity = identity_of(&rig, "git:git@github.com:user/repo");
    let ssh_protocol = identity_of(&rig, "ssh://git@github.com/user/repo");
    let git_dot = identity_of(&rig, "https://github.com/user/repo.git");
    assert_eq!(https_identity, "git:github.com/user/repo");
    assert_eq!(ssh_identity, https_identity);
    assert_eq!(ssh_protocol, https_identity);
    assert_eq!(git_dot, https_identity);
}

#[tokio::test]
async fn identity_ignores_the_ref() {
    let rig = Rig::new();
    assert_eq!(
        identity_of(&rig, "https://github.com/user/repo@v1.0.0"),
        identity_of(&rig, "git:git@github.com:user/repo@v1.0.0")
    );
    assert_eq!(
        identity_of(&rig, "https://github.com/user/repo@v1.0.0"),
        "git:github.com/user/repo"
    );
}

#[tokio::test]
async fn keeps_different_repos_separate() {
    let rig = Rig::new();
    assert_ne!(
        identity_of(&rig, "https://github.com/user/repo1"),
        identity_of(&rig, "git:git@github.com:user/repo2")
    );
}

// =============================================================================
// Multi-file extension discovery
// =============================================================================

#[tokio::test]
async fn loads_only_the_index_entry_from_subdirectories() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("multifile-pkg");
    rig.write_executable("multifile-pkg/extensions/subagent/index");
    // Helper modules carry no execute bit — the restated .ts rule.
    std::fs::write(
        rig.temp_dir
            .join("multifile-pkg/extensions/subagent/agents"),
        "stub",
    )
    .expect("write");
    rig.write_executable("multifile-pkg/extensions/standalone");

    let result = rig
        .package_manager
        .resolve_extension_sources(&[pkg_dir.to_string_lossy().into_owned()], false, false)
        .await
        .expect("resolve");
    assert!(find_with(&result.extensions, "subagent/index"));
    assert!(find_with(&result.extensions, "standalone"));
    assert!(!find_with(&result.extensions, "agents"));
}

#[tokio::test]
async fn respects_a_pi_manifest_in_subdirectories() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("manifest-subdir-pkg");
    rig.write_executable("manifest-subdir-pkg/extensions/custom/main");
    std::fs::write(
        rig.temp_dir
            .join("manifest-subdir-pkg/extensions/custom/utils"),
        "stub",
    )
    .expect("write");
    rig.write(
        "manifest-subdir-pkg/extensions/custom/package.json",
        r#"{"pi":{"extensions":["./main"]}}"#,
    );

    let result = rig
        .package_manager
        .resolve_extension_sources(&[pkg_dir.to_string_lossy().into_owned()], false, false)
        .await
        .expect("resolve");
    assert!(find_with(&result.extensions, "custom/main"));
    assert!(!find_with(&result.extensions, "utils"));
}

#[tokio::test]
async fn handles_mixed_top_level_files_and_subdirectories() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("mixed-pkg");
    rig.write_executable("mixed-pkg/extensions/simple");
    rig.write_executable("mixed-pkg/extensions/complex/index");
    std::fs::write(rig.temp_dir.join("mixed-pkg/extensions/complex/a"), "stub").expect("write");
    std::fs::write(rig.temp_dir.join("mixed-pkg/extensions/complex/b"), "stub").expect("write");

    let result = rig
        .package_manager
        .resolve_extension_sources(&[pkg_dir.to_string_lossy().into_owned()], false, false)
        .await
        .expect("resolve");
    assert!(find_with(&result.extensions, "simple"));
    assert!(find_with(&result.extensions, "complex/index"));
    assert!(!find_with(&result.extensions, "complex/a"));
    assert!(!find_with(&result.extensions, "complex/b"));
    assert_eq!(
        result
            .extensions
            .iter()
            .filter(|resource| resource.enabled)
            .count(),
        2
    );
}

#[tokio::test]
async fn skips_subdirectories_without_an_index_entry_or_manifest() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("no-entry-pkg");
    std::fs::create_dir_all(rig.temp_dir.join("no-entry-pkg/extensions/broken"))
        .expect("broken dir");
    std::fs::write(
        rig.temp_dir.join("no-entry-pkg/extensions/broken/helper"),
        "stub",
    )
    .expect("write");
    std::fs::write(
        rig.temp_dir.join("no-entry-pkg/extensions/broken/another"),
        "stub",
    )
    .expect("write");
    rig.write_executable("no-entry-pkg/extensions/valid");

    let result = rig
        .package_manager
        .resolve_extension_sources(&[pkg_dir.to_string_lossy().into_owned()], false, false)
        .await
        .expect("resolve");
    let debug: Vec<String> = result
        .extensions
        .iter()
        .map(|resource| resource.path.clone())
        .collect();
    assert!(
        find_with(&result.extensions, "valid"),
        "extensions were {debug:?}"
    );
    assert_eq!(
        result
            .extensions
            .iter()
            .filter(|resource| resource.enabled)
            .count(),
        1
    );
}

// =============================================================================
// The crate: channel's bin/ convention
// =============================================================================

#[tokio::test]
async fn resolves_a_local_directory_without_resources_as_one_extension() {
    // Upstream's resolveLocalExtensionSource: a directory with no manifest
    // and no convention directories is itself the extension entry.
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("crate-bin-pkg");
    rig.write_executable("crate-bin-pkg/bin/pi-llm-tools");
    rig.write_executable("crate-bin-pkg/bin/other-tool");

    let result = rig
        .package_manager
        .resolve_extension_sources(&[pkg_dir.to_string_lossy().into_owned()], false, false)
        .await
        .expect("resolve");
    assert_eq!(result.extensions.len(), 1);
    assert_eq!(
        result.extensions[0].path,
        pkg_dir.to_string_lossy().into_owned()
    );
    assert!(result.extensions[0].enabled);
}

#[tokio::test]
async fn collects_crate_channel_binaries_through_the_bin_convention() {
    // The `crate:` channel's installed layout: the receipt-carrying package
    // root holds only `bin/`, and the binaries are the extension entries.
    let rig = Rig::new();
    let pkg_dir = rig.agent_dir.join("crates/pi-llm-tools");
    std::fs::create_dir_all(pkg_dir.join("bin")).expect("bin dir");
    rig.write("agent/crates/pi-llm-tools/pi-package-install.json", r#"{"kind":"pi-package-install","schemaVersion":1,"channel":"crate","source":"crate:pi-llm-tools","resolvedVersion":"1.0.0","files":{}}"#);
    rig.write_executable("agent/crates/pi-llm-tools/bin/pi-llm-tools");
    rig.write_executable("agent/crates/pi-llm-tools/bin/other-tool");
    rig.set_packages(&json!(["crate:pi-llm-tools"]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_with(&result.extensions, "bin/pi-llm-tools"));
    assert!(find_with(&result.extensions, "bin/other-tool"));
    assert_eq!(
        result
            .extensions
            .iter()
            .filter(|resource| resource.enabled)
            .count(),
        2
    );
}

// =============================================================================
// Update checks over the crate: channel (the npm restatements)
// =============================================================================

#[tokio::test]
async fn updates_a_crate_source_whose_registry_version_is_newer() {
    let mut rig = Rig::new();
    let installed = rig.temp_dir.join(".pi/crates/example");
    std::fs::create_dir_all(&installed).expect("install dir");
    rig.write(
        ".pi/crates/example/pi-package-install.json",
        r#"{"kind":"pi-package-install","schemaVersion":1,"channel":"crate","source":"crate:example","resolvedVersion":"1.0.0","files":{}}"#,
    );
    rig.set_project_packages(&json!(["crate:example"]));

    let mock = pi_ai::http::MockHttpClient::new();
    mock.on(|request| request.url.contains("/api/v1/crates/example"))
        .respond(pi_ai::http::json_response(
            200,
            &json!({ "crate": { "max_version": "1.2.0" } }),
        ));
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(mock.clone());
    let runner = FakeRunner::new(fake_cargo_install());
    rig.runner = Some(runner.clone());
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: Some(Arc::new(runner)),
        env: Some(common::env_with(&[])),
        http_client: Some(client),
    });

    rig.package_manager
        .update(Some("crate:example"))
        .await
        .expect("update");
    let recorded = rig.package_manager_list_commands();
    assert_eq!(recorded.len(), 1, "the update installs once");
    let (command, args, _) = &recorded[0];
    assert_eq!(command, "cargo");
    assert!(args.contains(&"example".to_string()));
    // The unpinned update floats to latest, upstream's `name@latest` spec —
    // cargo's default when no --version rides.
    assert!(!args.contains(&"--version".to_string()));
    assert!(args.contains(&"--locked".to_string()));
}

impl Rig {
    fn package_manager_list_commands(&self) -> Vec<RecordedCall> {
        self.runner
            .as_ref()
            .map(FakeRunner::recorded)
            .unwrap_or_default()
    }
}

#[tokio::test]
async fn skips_a_crate_update_when_the_installed_version_matches_latest() {
    let mut rig = Rig::new();
    rig.write(
        ".pi/crates/example/pi-package-install.json",
        r#"{"kind":"pi-package-install","schemaVersion":1,"channel":"crate","source":"crate:example","resolvedVersion":"1.3.1","files":{}}"#,
    );
    std::fs::create_dir_all(rig.temp_dir.join(".pi/crates/example")).expect("install dir");
    rig.set_project_packages(&json!(["crate:example"]));

    let mock = pi_ai::http::MockHttpClient::new();
    mock.on(|request| request.url.contains("/api/v1/crates/example"))
        .respond(pi_ai::http::json_response(
            200,
            &json!({ "crate": { "max_version": "1.3.1" } }),
        ));
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(mock.clone());
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: Some(Arc::new(FakeRunner::rejecting("unexpected install"))),
        env: Some(common::env_with(&[])),
        http_client: Some(client),
    });

    rig.package_manager
        .update(Some("crate:example"))
        .await
        .expect("update");
    assert_eq!(
        rig.package_manager_list_commands().len(),
        0,
        "no install runs for a current package"
    );
}

#[tokio::test]
async fn suggests_the_crate_prefix_for_update_lookups() {
    let rig = Rig::new();
    rig.set_project_packages(&json!(["crate:example"]));

    let error = rig
        .package_manager
        .update(Some("example"))
        .await
        .expect_err("no match");
    assert_eq!(
        error.0,
        "No matching package found for example. Did you mean crate:example?"
    );
}

#[tokio::test]
async fn suggests_the_git_prefix_for_update_lookups() {
    let rig = Rig::new();
    rig.set_project_packages(&json!(["git:github.com/example/repo"]));

    let error = rig
        .package_manager
        .update(Some("github.com/example/repo"))
        .await
        .expect_err("no match");
    assert_eq!(
        error.0,
        "No matching package found for github.com/example/repo. Did you mean git:github.com/example/repo?"
    );
}

#[tokio::test]
async fn skips_installing_missing_sources_when_offline() {
    let rig = Rig::with_env(&[("PI_OFFLINE", "1")]);
    rig.set_project_packages(&json!([
        "crate:missing-package",
        "git:github.com/example/missing-repo"
    ]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    let all: Vec<&ResolvedResource> = result
        .extensions
        .iter()
        .chain(result.skills.iter())
        .chain(result.prompts.iter())
        .chain(result.themes.iter())
        .collect();
    assert!(
        !all.iter()
            .any(|resource| resource.metadata.origin.as_str() == "package")
    );
}

#[tokio::test]
async fn skips_refreshing_temporary_git_sources_when_offline() {
    let rig = Rig::with_env(&[("PI_OFFLINE", "1")]);
    let parsed = rig
        .package_manager
        .parse_source("git:github.com/example/repo")
        .expect("parses");
    let ParsedSource::Git(git_source) = parsed else {
        panic!("expected git");
    };
    let temp =
        pi_coding_agent::package_manager::get_extension_temp_folder(&rig.agent_dir).expect("temp");
    let root = temp.join(format!("git-{}", git_source.host));
    let hash = format!("{:x}", {
        use sha2::Digest as _;
        let mut hasher = sha2::Sha256::new();
        hasher.update(format!("git-{}-{}", git_source.host, git_source.path));
        hasher.finalize()
    })[..8]
        .to_string();
    let installed_path = root.join(&hash).join(&git_source.path);
    std::fs::create_dir_all(installed_path.join("extensions")).expect("temp install dir");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let index = installed_path.join("extensions/index");
        std::fs::write(&index, "#!/bin/sh\n").expect("write index");
        std::fs::set_permissions(&index, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    }

    let result = rig
        .package_manager
        .resolve_extension_sources(&["git:github.com/example/repo".to_string()], false, true)
        .await
        .expect("resolve");
    assert!(find_with(&result.extensions, "extensions/index"));
}

#[tokio::test]
async fn does_not_query_the_registry_during_resolve_for_installed_unpinned_crates() {
    let rig = Rig::with_env(&[("PI_OFFLINE", "1")]);
    rig.write(
        ".pi/crates/example/pi-package-install.json",
        r#"{"kind":"pi-package-install","schemaVersion":1,"channel":"crate","source":"crate:example","resolvedVersion":"1.0.0","files":{}}"#,
    );
    std::fs::create_dir_all(rig.temp_dir.join(".pi/crates/example/extensions"))
        .expect("install dir");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let index = rig.temp_dir.join(".pi/crates/example/extensions/index");
        std::fs::write(&index, "#!/bin/sh\n").expect("write index");
        std::fs::set_permissions(&index, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    }
    rig.set_project_packages(&json!(["crate:example"]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_with(&result.extensions, "extensions/index"));
}

#[tokio::test]
async fn reinstalls_a_pinned_crate_when_the_installed_version_mismatches() {
    let mut rig = Rig::new();
    let installed_dir = rig.temp_dir.join(".pi/crates/example");
    std::fs::create_dir_all(&installed_dir).expect("install dir");
    rig.write(
        ".pi/crates/example/pi-package-install.json",
        r#"{"kind":"pi-package-install","schemaVersion":1,"channel":"crate","source":"crate:example","resolvedVersion":"1.0.0","files":{}}"#,
    );
    rig.set_project_packages(&json!(["crate:example@2.0.0"]));

    rig.with_runner(FakeRunner::new(fake_cargo_install()));
    rig.package_manager.resolve(None).await.expect("resolve");
    let recorded = rig.package_manager_list_commands();
    assert_eq!(
        recorded.len(),
        1,
        "the resolve reinstalls the mismatched pin"
    );
    let (command, args, _) = &recorded[0];
    assert_eq!(command, "cargo");
    assert!(args.contains(&"--version".to_string()));
    assert!(args.contains(&"2.0.0".to_string()));
    let _ = installed_dir;
}

#[tokio::test]
async fn reports_updates_for_installed_unpinned_crates() {
    let mut rig = Rig::new();
    rig.write(
        ".pi/crates/example/pi-package-install.json",
        r#"{"kind":"pi-package-install","schemaVersion":1,"channel":"crate","source":"crate:example","resolvedVersion":"1.0.0","files":{}}"#,
    );
    std::fs::create_dir_all(rig.temp_dir.join(".pi/crates/example")).expect("install dir");
    rig.set_project_packages(&json!(["crate:example"]));

    let mock = pi_ai::http::MockHttpClient::new();
    mock.on(|request| request.url.contains("/api/v1/crates/example"))
        .respond(pi_ai::http::json_response(
            200,
            &json!({ "crate": { "max_version": "1.2.3" } }),
        ));
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(mock.clone());
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: None,
        env: Some(common::env_with(&[])),
        http_client: Some(client),
    });

    let updates = rig
        .package_manager
        .check_for_available_updates()
        .await
        .expect("check");
    assert_eq!(updates.len(), 1);
    assert_eq!(updates[0].source, "crate:example");
    assert_eq!(updates[0].display_name, "example");
    assert_eq!(updates[0].update_type.as_str(), "crate");
    assert_eq!(updates[0].scope, SourceScope::Project);
}

#[tokio::test]
async fn does_not_report_updates_when_the_installed_version_is_newer_than_the_registry() {
    let mut rig = Rig::new();
    rig.write(
        ".pi/crates/example/pi-package-install.json",
        r#"{"kind":"pi-package-install","schemaVersion":1,"channel":"crate","source":"crate:example","resolvedVersion":"2.0.0","files":{}}"#,
    );
    std::fs::create_dir_all(rig.temp_dir.join(".pi/crates/example")).expect("install dir");
    rig.set_project_packages(&json!(["crate:example"]));

    let mock = pi_ai::http::MockHttpClient::new();
    mock.on(|request| request.url.contains("/api/v1/crates/example"))
        .respond(pi_ai::http::json_response(
            200,
            &json!({ "crate": { "max_version": "1.9.0" } }),
        ));
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(mock.clone());
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: None,
        env: Some(common::env_with(&[])),
        http_client: Some(client),
    });

    let updates = rig
        .package_manager
        .check_for_available_updates()
        .await
        .expect("check");
    assert!(updates.is_empty());
}

#[tokio::test]
async fn skips_pinned_sources_when_checking_for_updates() {
    let mut rig = Rig::new();
    rig.write(
        ".pi/crates/example/pi-package-install.json",
        r#"{"kind":"pi-package-install","schemaVersion":1,"channel":"crate","source":"crate:example","resolvedVersion":"1.0.0","files":{}}"#,
    );
    std::fs::create_dir_all(rig.temp_dir.join(".pi/crates/example")).expect("install dir");
    let parsed = rig
        .package_manager
        .parse_source("git:github.com/example/repo@v1")
        .expect("parses");
    let ParsedSource::Git(git_source) = parsed else {
        panic!("expected git");
    };
    let git_path = rig
        .temp_dir
        .join(".pi/git")
        .join("github.com")
        .join("example")
        .join("repo");
    std::fs::create_dir_all(&git_path).expect("git checkout dir");
    rig.set_project_packages(&json!([
        "crate:example@1.0.0",
        "git:github.com/example/repo@v1"
    ]));

    let mock = pi_ai::http::MockHttpClient::new();
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(mock.clone());
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: Some(Arc::new(FakeRunner::rejecting("no command may run"))),
        env: Some(common::env_with(&[])),
        http_client: Some(client),
    });

    let updates = rig
        .package_manager
        .check_for_available_updates()
        .await
        .expect("check");
    assert!(updates.is_empty());
    assert_eq!(
        mock.recorded().len(),
        0,
        "no registry query for pinned sources"
    );
    let _ = git_source;
}

#[tokio::test]
async fn check_for_available_updates_answers_nothing_when_offline() {
    let rig = Rig::with_env(&[("PI_OFFLINE", "1")]);
    let updates = rig
        .package_manager
        .check_for_available_updates()
        .await
        .expect("check");
    assert!(updates.is_empty());
}

// =============================================================================
// The crate: install writes a receipt; the git install reconciles
// =============================================================================

#[tokio::test]
async fn installs_a_crate_source_with_a_locked_build_and_a_receipt() {
    let mut rig = Rig::new();
    let stage_recorded: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&stage_recorded);
    let installed_dir = rig.agent_dir.join("crates/example");
    rig.with_runner(FakeRunner::new(Box::new(move |command, args, _options| {
        if command == "cargo" && args.first().map(String::as_str) == Some("install") {
            let root_index = args
                .iter()
                .position(|arg| arg == "--root")
                .expect("--root present");
            let stage_root = args.get(root_index + 1).cloned().unwrap_or_default();
            std::fs::create_dir_all(Path::new(&stage_root).join("bin")).expect("stage bin");
            std::fs::write(Path::new(&stage_root).join("bin/example"), "#!/bin/sh\n").expect("bin");
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(
                    Path::new(&stage_root).join("bin/example"),
                    std::fs::Permissions::from_mode(0o755),
                )
                .expect("chmod");
            }
            *sink
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(stage_root);
        }
        Ok(FakeOutcome::ok())
    })));

    rig.package_manager
        .install("crate:example@1.2.3", false)
        .await
        .expect("install");

    let receipt_path = installed_dir.join("pi-package-install.json");
    let receipt: Value =
        serde_json::from_str(&std::fs::read_to_string(&receipt_path).expect("receipt written"))
            .expect("receipt parses");
    assert_eq!(
        receipt.get("kind").and_then(Value::as_str),
        Some("pi-package-install")
    );
    assert_eq!(
        receipt.get("channel").and_then(Value::as_str),
        Some("crate")
    );
    assert_eq!(
        receipt.get("resolvedVersion").and_then(Value::as_str),
        Some("1.2.3")
    );
    assert!(
        receipt
            .get("files")
            .and_then(Value::as_object)
            .is_some_and(|files| files.contains_key("bin/example"))
    );
    let cargo_calls = rig.package_manager_list_commands();
    assert_eq!(cargo_calls.len(), 1, "one cargo invocation");
    let (command, args, _) = &cargo_calls[0];
    assert_eq!(command, "cargo");
    assert!(args.contains(&"--locked".to_string()));
    let stage = stage_recorded
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert!(stage.is_some(), "the install staged");
}

#[tokio::test]
async fn removes_a_newly_created_checkout_when_the_git_clone_fails() {
    let mut rig = Rig::new();
    let target_dir = rig.agent_dir.join("git/github.com/user/repo");
    rig.with_runner(FakeRunner::new(Box::new(move |command, args, _options| {
        if command == "git" && args.first().map(String::as_str) == Some("clone") {
            let target = args.get(2).cloned().unwrap_or_default();
            std::fs::create_dir_all(&target).expect("target dir");
            return Err(PackageManagerError(
                "simulated git clone failure".to_string(),
            ));
        }
        Ok(FakeOutcome::ok())
    })));

    let error = rig
        .package_manager
        .install("git:github.com/user/repo", false)
        .await
        .expect_err("clone fails");
    assert_eq!(error.0, "simulated git clone failure");
    assert!(!target_dir.exists(), "the failed clone cleans up");
}

#[tokio::test]
async fn reconciles_an_existing_checkout_to_a_pinned_ref_during_install() {
    let mut rig = Rig::new();
    let target_dir = rig.agent_dir.join("git/github.com/user/repo");
    std::fs::create_dir_all(&target_dir).expect("checkout dir");
    rig.write(
        "agent/git/github.com/user/repo/package.json",
        r#"{"name":"repo","version":"1.0.0"}"#,
    );

    rig.with_runner(FakeRunner::new(Box::new(|command, args, _options| {
        if command == "git" && args.first().map(String::as_str) == Some("rev-parse") {
            if args.get(1).map(String::as_str) == Some("HEAD") {
                return Ok(FakeOutcome::stdout("old-head"));
            }
            if args.get(1).map(String::as_str) == Some("FETCH_HEAD^{commit}") {
                return Ok(FakeOutcome::stdout("new-head"));
            }
        }
        Ok(FakeOutcome::ok())
    })));

    rig.package_manager
        .install("git:github.com/user/repo@v2", false)
        .await
        .expect("install");

    let recorded = rig.package_manager_list_commands();
    let commands: Vec<RecordedCall> = recorded
        .iter()
        .filter(|(command, args, _)| {
            command == "git" && args.first().map(String::as_str) == Some("fetch")
        })
        .cloned()
        .collect();
    assert_eq!(commands.len(), 1, "one fetch");
    assert!(commands[0].1.contains(&"v2".to_string()));
    let resets: Vec<_> = recorded
        .iter()
        .filter(|(command, args, _)| {
            command == "git" && args.first().map(String::as_str) == Some("reset")
        })
        .collect();
    assert_eq!(resets.len(), 1, "one hard reset");
    assert!(
        resets[0].1.contains(&"--hard".to_string())
            && resets[0].1.contains(&"FETCH_HEAD^{commit}".to_string())
    );
    // The clean-and-install tail removes the marker once the checkout is
    // pristine, upstream's rmSync at the end of cleanAndInstallGitDependencies.
    let marker = target_dir
        .parent()
        .expect("parent")
        .join(".repo.pi-update-incomplete");
    assert!(!marker.exists(), "the completed update clears the marker");
}

#[tokio::test]
async fn preserves_argv_entries_containing_spaces_through_the_real_runner() {
    let _rig = Rig::new();
    let value_with_space = "/tmp/A B/.pi/npm";
    let output = {
        let runner = pi_coding_agent::package_manager::ProcessCommandRunner;
        let args: Vec<String> = vec![
            "-c".to_string(),
            "printf %s \"$1\"".to_string(),
            "sh".to_string(),
            value_with_space.to_string(),
        ];
        runner.run_sync("sh", &args).expect("spawn")
    };
    assert_eq!(output, value_with_space);
}

#[tokio::test]
async fn the_real_runner_resolves_capture_only_after_the_child_settles() {
    let runner = pi_coding_agent::package_manager::ProcessCommandRunner;
    let options = CommandRunOptions::default();
    let args: Vec<String> = vec!["-c".to_string(), "sleep 0.1; printf abc123".to_string()];
    let output = runner
        .run_capture("sh", &args, &options, Some(5_000))
        .await
        .expect("capture");
    assert_eq!(output, "abc123");
}

// =============================================================================
// The ssh suite, upstream's test/package-manager-ssh.test.ts
// =============================================================================

#[tokio::test]
async fn ssh_parses_protocol_urls_without_the_git_prefix() {
    let rig = Rig::new();
    let https = rig
        .package_manager
        .parse_source("https://github.com/user/repo")
        .expect("parses");
    match https {
        ParsedSource::Git(parsed) => {
            assert_eq!(parsed.host, "github.com");
            assert_eq!(parsed.path, "user/repo");
        }
        _ => panic!("expected git"),
    }
    let ssh = rig
        .package_manager
        .parse_source("ssh://git@github.com/user/repo")
        .expect("parses");
    match ssh {
        ParsedSource::Git(parsed) => {
            assert_eq!(parsed.host, "github.com");
            assert_eq!(parsed.path, "user/repo");
            assert_eq!(parsed.repo, "ssh://git@github.com/user/repo");
        }
        _ => panic!("expected git"),
    }
}

#[tokio::test]
async fn ssh_parses_shorthand_urls_with_the_git_prefix() {
    let rig = Rig::new();
    let git_at = rig
        .package_manager
        .parse_source("git:git@github.com:user/repo")
        .expect("parses");
    match git_at {
        ParsedSource::Git(parsed) => {
            assert_eq!(parsed.host, "github.com");
            assert_eq!(parsed.path, "user/repo");
            assert_eq!(parsed.repo, "git@github.com:user/repo");
            assert!(!parsed.pinned);
        }
        _ => panic!("expected git"),
    }
    let shorthand = rig
        .package_manager
        .parse_source("git:github.com/user/repo")
        .expect("parses");
    match shorthand {
        ParsedSource::Git(parsed) => {
            assert_eq!(parsed.host, "github.com");
            assert_eq!(parsed.path, "user/repo");
        }
        _ => panic!("expected git"),
    }
    let with_ref = rig
        .package_manager
        .parse_source("git:git@github.com:user/repo@v1.0.0")
        .expect("parses");
    match with_ref {
        ParsedSource::Git(parsed) => {
            assert_eq!(parsed.r#ref.as_deref(), Some("v1.0.0"));
            assert!(parsed.pinned);
        }
        _ => panic!("expected git"),
    }
}

#[tokio::test]
async fn ssh_treats_shorthands_as_local_without_the_git_prefix() {
    let rig = Rig::new();
    assert!(matches!(
        rig.package_manager
            .parse_source("git@github.com:user/repo")
            .expect("parses"),
        ParsedSource::Local(_)
    ));
    assert!(matches!(
        rig.package_manager
            .parse_source("github.com/user/repo")
            .expect("parses"),
        ParsedSource::Local(_)
    ));
}

#[tokio::test]
async fn ssh_normalizes_protocol_and_shorthand_urls_to_one_identity() {
    let rig = Rig::new();
    let prefixed = identity_of(&rig, "git:git@github.com:user/repo");
    let https = identity_of(&rig, "https://github.com/user/repo");
    let ssh = identity_of(&rig, "ssh://git@github.com/user/repo");
    assert_eq!(prefixed, "git:github.com/user/repo");
    assert_eq!(prefixed, https);
    assert_eq!(prefixed, ssh);
}

// =============================================================================
// The real runner's failure, signal, and timeout paths
// =============================================================================

/// Whether a recorded runner invocation is `git <key>`, the joined-args probe
/// the git fake's table rides.
fn ran_git(recorded: &[RecordedCall], key: &str) -> bool {
    recorded
        .iter()
        .any(|(command, args, _)| command == "git" && args.join(" ") == key)
}

#[tokio::test]
async fn the_real_runner_applies_env_and_reports_a_clean_exit_from_run() {
    let rig = Rig::new();
    let runner = pi_coding_agent::package_manager::ProcessCommandRunner;
    let options = CommandRunOptions {
        cwd: Some(rig.temp_dir.to_string_lossy().into_owned()),
        env: vec![("PM_RUNNER_PROBE".to_string(), "runner-ok".to_string())],
    };
    // The command succeeds only when the injected env reached the child.
    let args: Vec<String> = vec![
        "-c".to_string(),
        "[ \"$PM_RUNNER_PROBE\" = runner-ok ]".to_string(),
    ];
    runner
        .run("sh", &args, &options)
        .await
        .expect("env reaches the child");
}

#[tokio::test]
async fn the_real_runner_reports_a_nonzero_exit_from_run() {
    let runner = pi_coding_agent::package_manager::ProcessCommandRunner;
    let args: Vec<String> = vec!["-c".to_string(), "exit 3".to_string()];
    let error = runner
        .run("sh", &args, &CommandRunOptions::default())
        .await
        .expect_err("nonzero exit fails");
    assert!(error.0.contains("failed with code 3"), "{error}");
}

#[tokio::test]
async fn the_real_runner_reports_a_spawn_failure_from_run() {
    let runner = pi_coding_agent::package_manager::ProcessCommandRunner;
    let error = runner
        .run(
            "pi-pm-missing-command-xyz",
            &[],
            &CommandRunOptions::default(),
        )
        .await
        .expect_err("missing command fails");
    assert!(!error.0.is_empty(), "{error}");
}

#[tokio::test]
async fn the_real_runner_reports_a_signal_exit_from_run() {
    let runner = pi_coding_agent::package_manager::ProcessCommandRunner;
    let args: Vec<String> = vec!["-c".to_string(), "kill -TERM $$".to_string()];
    let error = runner
        .run("sh", &args, &CommandRunOptions::default())
        .await
        .expect_err("signal exit fails");
    assert!(
        error.0.contains("unknown exit status"),
        "a terminated child carries no code: {error}"
    );
}

#[tokio::test]
async fn the_real_runner_captures_stderr_detail_from_a_failing_command() {
    let runner = pi_coding_agent::package_manager::ProcessCommandRunner;
    let options = CommandRunOptions::default();
    let args: Vec<String> = vec![
        "-c".to_string(),
        "echo to-stdout; echo to-stderr 1>&2; exit 7".to_string(),
    ];
    let error = runner
        .run_capture("sh", &args, &options, Some(5_000))
        .await
        .expect_err("nonzero exit fails");
    assert!(error.0.contains("code 7"), "{error}");
    // The stderr pipe is non-empty, so it is the reported detail.
    assert!(error.0.contains("to-stderr"), "{error}");
}

#[tokio::test]
async fn the_real_runner_captures_stdout_detail_when_stderr_is_empty() {
    let runner = pi_coding_agent::package_manager::ProcessCommandRunner;
    let options = CommandRunOptions::default();
    let args: Vec<String> = vec!["-c".to_string(), "echo only-stdout; exit 7".to_string()];
    let error = runner
        .run_capture("sh", &args, &options, Some(5_000))
        .await
        .expect_err("nonzero exit fails");
    assert!(error.0.contains("code 7"), "{error}");
    assert!(error.0.contains("only-stdout"), "{error}");
}

#[tokio::test]
async fn the_real_runner_reports_a_signal_exit_from_run_capture() {
    let runner = pi_coding_agent::package_manager::ProcessCommandRunner;
    let options = CommandRunOptions::default();
    let args: Vec<String> = vec!["-c".to_string(), "kill -TERM $$".to_string()];
    let error = runner
        .run_capture("sh", &args, &options, Some(5_000))
        .await
        .expect_err("signal exit fails");
    assert!(error.0.contains("signal"), "{error}");
}

#[tokio::test]
async fn the_real_runner_times_out_a_capture_after_the_deadline() {
    let runner = pi_coding_agent::package_manager::ProcessCommandRunner;
    let options = CommandRunOptions::default();
    let args: Vec<String> = vec!["-c".to_string(), "sleep 2".to_string()];
    let error = runner
        .run_capture("sh", &args, &options, Some(100))
        .await
        .expect_err("timeout kills");
    assert!(error.0.contains("timed out after 100ms"), "{error}");
}

#[tokio::test]
async fn the_real_runner_reports_a_spawn_failure_from_run_capture() {
    let runner = pi_coding_agent::package_manager::ProcessCommandRunner;
    let error = runner
        .run_capture(
            "pi-pm-missing-command-xyz",
            &[],
            &CommandRunOptions::default(),
            Some(5_000),
        )
        .await
        .expect_err("missing command fails");
    assert!(!error.0.is_empty(), "{error}");
}

// =============================================================================
// The git fake: a table-driven rev-parse / ls-remote double
// =============================================================================

/// The 40-hex heads the ls-remote fakes answer with, the shape
/// `first_hex40_line` matches.
const FAKE_HEAD_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const FAKE_HEAD_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

/// A git double: exact joined-argv keys answer captured stdout, `failures`
/// keys fail, every other command succeeds. The keys are the command names
/// the reconcile and update-check ladders spawn.
fn git_fake(outputs: &[(&str, &str)], failures: &[&str]) -> FakeRunner {
    let outputs: Vec<(String, String)> = outputs
        .iter()
        .map(|(key, output)| ((*key).to_string(), (*output).to_string()))
        .collect();
    let failures: Vec<String> = failures.iter().map(|key| (*key).to_string()).collect();
    FakeRunner::new(Box::new(move |command, args, _options| {
        if command == "git" {
            let key = args.join(" ");
            if failures.contains(&key) {
                return Err(PackageManagerError(format!("{key} failed")));
            }
            for (candidate, output) in &outputs {
                if *candidate == key {
                    return Ok(FakeOutcome::stdout(output.clone()));
                }
            }
        }
        Ok(FakeOutcome::ok())
    }))
}

const UPSTREAM_ABBREV_KEY: &str = "rev-parse --abbrev-ref @{upstream}";
const UPSTREAM_REV_KEY: &str = "rev-parse @{upstream}";
const UPSTREAM_COMMIT_KEY: &str = "rev-parse @{upstream}^{commit}";
const HEAD_REV_KEY: &str = "rev-parse HEAD";
const ORIGIN_HEAD_REV_KEY: &str = "rev-parse origin/HEAD";
const SYMREF_KEY: &str = "symbolic-ref refs/remotes/origin/HEAD";
const LS_REMOTE_UPSTREAM_KEY: &str = "ls-remote origin refs/heads/main";
const LS_REMOTE_HEAD_KEY: &str = "ls-remote origin HEAD";
const UPSTREAM_FETCH_KEY: &str =
    "fetch --prune --no-tags origin +refs/heads/main:refs/remotes/origin/main";
const UPSTREAM_RESET_KEY: &str = "reset --hard @{upstream}^{commit}";
const ORIGIN_HEAD_RESET_KEY: &str = "reset --hard origin/HEAD^{commit}";
const HARD_HEAD_FETCH_KEY: &str = "fetch --prune --no-tags origin +HEAD:refs/remotes/origin/HEAD";

/// The pre-existing checkout the reconcile ladder drives, plus its marker
/// path.
fn existing_git_checkout(rig: &Rig, slug: &str) -> PathBuf {
    let target = rig.agent_dir.join("git/github.com/user").join(slug);
    std::fs::create_dir_all(&target).expect("checkout dir");
    target
}

// =============================================================================
// The tarball channel
// =============================================================================

/// A gzipped tar the unpack drives, the `tar::Builder`-built archive the
/// boundary suite's manual ustar headers restate.
fn gz_tarball(entries: Vec<(&str, &[u8], u32)>) -> Vec<u8> {
    let encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    let mut builder = tar::Builder::new(encoder);
    for (name, contents, mode) in entries {
        let mut header = tar::Header::new_gnu();
        header.set_size(contents.len() as u64);
        header.set_mode(mode);
        header.set_entry_type(tar::EntryType::Regular);
        header.set_cksum();
        builder
            .append_data(&mut header, name, contents)
            .expect("tar entry");
    }
    builder
        .into_inner()
        .expect("tar finish")
        .finish()
        .expect("gz finish")
}

/// The tarball install dir name, the sha256-8 the URL hashes to — the same
/// digest `tarball_dir_name` derives.
fn tarball_hash(url: &str) -> String {
    use sha2::Digest as _;
    let mut hasher = sha2::Sha256::new();
    hasher.update(format!("tarball-{url}"));
    format!("{:x}", hasher.finalize())[..8].to_string()
}

/// The manager over the rig's dirs with an injected HTTP client.
fn rig_with_client(
    rig: &mut Rig,
    client: Arc<dyn pi_ai::http::HttpClient>,
) -> &DefaultPackageManager<InMemorySettingsStorage> {
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: None,
        env: Some(common::env_with(&[])),
        http_client: Some(client),
    });
    &rig.package_manager
}

#[tokio::test]
async fn installs_a_tarball_source_with_a_receipt() {
    let mut rig = Rig::new();
    let mock = pi_ai::http::MockHttpClient::new();
    let url = "https://example.com/tool-1.0.0.tgz";
    mock.on(move |request| request.url == url).respond(
        pi_ai::http::MockResponse::status(200).with_body(gz_tarball(vec![
            ("bin/tool", b"#!/bin/sh\n".as_slice(), 0o755),
            ("README.md", b"readme".as_slice(), 0o644),
        ])),
    );
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(mock);
    let manager = rig_with_client(&mut rig, Arc::clone(&client));

    manager.install(url, false).await.expect("install");

    let target = rig.agent_dir.join("tarballs").join(tarball_hash(url));
    let receipt: Value = serde_json::from_str(
        &std::fs::read_to_string(target.join("pi-package-install.json")).expect("receipt written"),
    )
    .expect("receipt parses");
    assert_eq!(
        receipt.get("channel").and_then(Value::as_str),
        Some("tarball")
    );
    assert_eq!(receipt.get("source").and_then(Value::as_str), Some(url));
    assert!(
        receipt.get("resolvedVersion").is_none(),
        "a tarball carries no resolved version"
    );
    assert!(
        receipt
            .get("files")
            .and_then(Value::as_object)
            .is_some_and(|files| files.contains_key("bin/tool")),
        "the receipt hashes every file: {receipt}"
    );
    // The unpacked executable keeps its mode bit.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(target.join("bin/tool"))
            .expect("unpacked binary")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o755, "the executable bit rides the unpack");
    }
}

#[tokio::test]
async fn installs_a_tarball_source_reports_a_transport_failure_and_cleans_the_stage() {
    let mut rig = Rig::new();
    // The empty mock answers nothing: every attempt fails at the transport.
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(pi_ai::http::MockHttpClient::new());
    let manager = rig_with_client(&mut rig, client);

    let error = manager
        .install("https://example.com/tool-1.0.0.tgz", false)
        .await
        .expect_err("transport failure");
    assert!(error.0.contains("example.com"), "{error}");

    let install_root = rig.agent_dir.join("tarballs");
    let leftovers: Vec<String> = std::fs::read_dir(&install_root)
        .expect("install root exists")
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        leftovers.iter().all(|name| !name.contains(".staging-")),
        "the failed install sweeps its stage: {leftovers:?}"
    );
    assert!(
        !install_root
            .join(tarball_hash("https://example.com/tool-1.0.0.tgz"))
            .exists(),
        "no install lands on a failed download"
    );
}

#[tokio::test]
async fn installs_a_tarball_source_reports_an_http_error_status() {
    let mut rig = Rig::new();
    let mock = pi_ai::http::MockHttpClient::new();
    mock.on(|request| request.url.contains("example.com"))
        .respond(pi_ai::http::MockResponse::status(500));
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(mock);
    let manager = rig_with_client(&mut rig, client);

    let error = manager
        .install("https://example.com/tool-1.0.0.tgz", false)
        .await
        .expect_err("500 fails");
    assert!(
        error.0.contains("HTTP 500"),
        "the status rides the message: {error}"
    );
}

#[tokio::test]
async fn installs_a_tarball_source_rejects_a_corrupt_archive_and_cleans_the_stage() {
    let mut rig = Rig::new();
    let mock = pi_ai::http::MockHttpClient::new();
    mock.on(|request| request.url.contains("example.com"))
        .respond(pi_ai::http::MockResponse::status(200).with_body(b"not a gzip stream".as_slice()));
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(mock);
    let manager = rig_with_client(&mut rig, client);

    let error = manager
        .install("https://example.com/tool-1.0.0.tgz", false)
        .await
        .expect_err("corrupt archive fails");
    assert!(!error.0.is_empty(), "{error}");
    let install_root = rig.agent_dir.join("tarballs");
    let leftovers: Vec<String> = std::fs::read_dir(&install_root)
        .expect("install root exists")
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        leftovers.iter().all(|name| !name.contains(".staging-")),
        "the failed unpack sweeps its stage: {leftovers:?}"
    );
}

/// A client whose body stream fails mid-read: the seam the mock's canned
/// bodies cannot express (their chunks are always `Ok`).
#[derive(Debug)]
struct FailingBodyClient;

impl pi_ai::http::HttpClient for FailingBodyClient {
    fn execute(
        &self,
        _request: pi_ai::http::HttpRequest,
    ) -> pi_ai::http::BoxHttpFuture<Result<pi_ai::http::HttpResponse, pi_ai::http::HttpError>> {
        Box::pin(async move {
            Ok(pi_ai::http::HttpResponse {
                status: 200,
                headers: Vec::new(),
                body: pi_ai::http::HttpByteStream::from_chunks(vec![Err(
                    pi_ai::http::HttpError::Transport("simulated mid-stream failure".to_string()),
                )]),
            })
        })
    }
}

#[tokio::test]
async fn installs_a_tarball_source_surfaces_a_midstream_body_failure() {
    let mut rig = Rig::new();
    let manager = rig_with_client(&mut rig, Arc::new(FailingBodyClient));

    let error = manager
        .install("https://example.com/tool-1.0.0.tgz", false)
        .await
        .expect_err("body failure");
    assert!(error.0.contains("simulated mid-stream failure"), "{error}");
}

#[tokio::test]
async fn the_tarball_install_surfaces_a_rename_failure_when_the_target_cannot_yield() {
    let mut rig = Rig::new();
    let url = "https://example.com/tool-1.0.0.tgz";
    let mock = pi_ai::http::MockHttpClient::new();
    mock.on(move |request| request.url == url).respond(
        pi_ai::http::MockResponse::status(200).with_body(gz_tarball(vec![(
            "bin/tool",
            b"#!/bin/sh\n".as_slice(),
            0o755,
        )])),
    );
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(mock);

    // A stuck child directory makes the pre-rename sweep fail silently, so
    // the target is still occupied when the rename fires.
    let target = rig.agent_dir.join("tarballs").join(tarball_hash(url));
    std::fs::create_dir_all(target.join("locked")).expect("stuck dir");
    std::fs::write(target.join("locked").join("payload"), b"x").expect("payload");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            target.join("locked"),
            std::fs::Permissions::from_mode(0o500),
        )
        .expect("lock the child dir");
    }

    let manager = rig_with_client(&mut rig, client);
    let error = manager.install(url, false).await.expect_err("rename fails");
    assert!(!error.0.is_empty(), "{error}");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            target.join("locked"),
            std::fs::Permissions::from_mode(0o755),
        )
        .expect("unlock for cleanup");
    }
}

#[tokio::test]
async fn tarball_and_git_install_paths_follow_their_scope_roots() {
    let rig = Rig::new();
    let tarball = TarballSource {
        url: "https://example.com/tool-1.0.0.tgz".to_string(),
    };
    let user_path = rig
        .package_manager
        .get_tarball_install_path(&tarball, SourceScope::User)
        .expect("user path");
    assert_eq!(
        Path::new(&user_path),
        rig.agent_dir
            .join("tarballs")
            .join(tarball_hash(&tarball.url))
            .as_path(),
        "user tarballs stage under the agent dir"
    );

    // The trusted project root rides the cwd's .pi tree.
    let project_path = rig
        .package_manager
        .get_tarball_install_path(&tarball, SourceScope::Project)
        .expect("project path");
    assert!(
        project_path.replace('\\', "/").contains(".pi/tarballs/"),
        "{project_path}"
    );

    let temporary_path = rig
        .package_manager
        .get_tarball_install_path(&tarball, SourceScope::Temporary)
        .expect("temporary path");
    assert!(
        temporary_path
            .replace('\\', "/")
            .contains("tmp/extensions/tarball/"),
        "temporary tarballs stage under the managed temp folder: {temporary_path}"
    );

    let git_source = pi_coding_agent::utils::git::GitSource {
        repo: "https://github.com/user/repo".to_string(),
        host: "github.com".to_string(),
        path: "user/repo".to_string(),
        r#ref: None,
        pinned: false,
    };
    let git_temporary = rig
        .package_manager
        .get_git_install_path(&git_source, SourceScope::Temporary)
        .expect("temporary git path");
    // The suffix join: <hash>/<path>, the shape get_temporary_dir's
    // non-empty-suffix arm builds.
    assert!(
        git_temporary
            .replace('\\', "/")
            .contains("tmp/extensions/git-github.com/"),
        "{git_temporary}"
    );
    assert!(
        git_temporary.replace('\\', "/").ends_with("/user/repo"),
        "the path suffix rides after the hash: {git_temporary}"
    );

    // The project root is trust-gated.
    rig.settings
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .set_project_trusted(false);
    let error = rig
        .package_manager
        .get_tarball_install_path(&tarball, SourceScope::Project)
        .expect_err("untrusted project refuses");
    assert!(error.0.contains("Project is not trusted"), "{error}");
}

#[tokio::test]
async fn unpack_tarball_accepts_an_inside_staged_symlink_and_rejects_a_hardlink() {
    let stage = tempfile::tempdir().expect("stage");
    // One archive: a regular file first, then a zero-size symlink entry
    // naming it — the link resolves inside the staging root, so it unpacks.
    let encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    let mut builder = tar::Builder::new(encoder);
    let mut header = tar::Header::new_gnu();
    header.set_size(7);
    header.set_mode(0o644);
    header.set_entry_type(tar::EntryType::Regular);
    header.set_cksum();
    builder
        .append_data(&mut header, "target-file", b"payload".as_slice())
        .expect("regular entry");
    let mut header = tar::Header::new_gnu();
    header.set_size(0);
    header.set_entry_type(tar::EntryType::Symlink);
    header.set_link_name("target-file").expect("link name");
    header.set_cksum();
    builder
        .append_data(&mut header, "inside-link", std::io::empty())
        .expect("symlink entry");
    let with_symlink = builder.into_inner().expect("tar").finish().expect("gz");
    pi_coding_agent::package_manager::unpack_tarball(&with_symlink, stage.path())
        .expect("inside symlink unpacks");
    assert!(stage.path().join("target-file").exists());
    #[cfg(unix)]
    {
        let link = std::fs::symlink_metadata(stage.path().join("inside-link"))
            .expect("symlink created")
            .file_type();
        assert!(link.is_symlink(), "the inside link lands as a symlink");
    }

    let stage2 = tempfile::tempdir().expect("stage");
    let encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    let mut builder = tar::Builder::new(encoder);
    let mut header = tar::Header::new_gnu();
    header.set_size(0);
    header.set_entry_type(tar::EntryType::Link);
    header.set_link_name("target-file").expect("link name");
    header.set_cksum();
    builder
        .append_data(&mut header, "hard-link", std::io::empty())
        .expect("hardlink entry");
    let with_hardlink = builder.into_inner().expect("tar").finish().expect("gz");
    let error = pi_coding_agent::package_manager::unpack_tarball(&with_hardlink, stage2.path())
        .expect_err("hardlink rejects");
    assert!(error.0.contains("unsupported entry type"), "{error}");
}

// =============================================================================
// The git reconcile ladder
// =============================================================================

/// The checkout exists, so `install` without a ref reconciles it to the
/// upstream branch's target: fetch the branch, reset when HEAD moved, clean.
#[tokio::test]
async fn installs_an_existing_unpinned_checkout_to_its_upstream_branch_target() {
    let mut rig = Rig::new();
    let target = existing_git_checkout(&rig, "repo");
    rig.with_runner(git_fake(
        &[
            (UPSTREAM_ABBREV_KEY, "origin/main"),
            (UPSTREAM_REV_KEY, "remote-head"),
            (HEAD_REV_KEY, "local-head"),
            (UPSTREAM_COMMIT_KEY, "remote-head"),
        ],
        &[],
    ));

    rig.package_manager
        .install("git:github.com/user/repo", false)
        .await
        .expect("reconcile");

    let recorded = rig.package_manager_list_commands();
    assert!(ran_git(&recorded, UPSTREAM_FETCH_KEY), "{recorded:?}");
    assert!(ran_git(&recorded, UPSTREAM_RESET_KEY), "{recorded:?}");
    assert!(ran_git(&recorded, "clean -fdx"), "{recorded:?}");
    // The completed reconcile removes the in-flight marker.
    let marker = target
        .parent()
        .expect("parent")
        .join(".repo.pi-update-incomplete");
    assert!(!marker.exists());
    // The fetch and the reset run inside the checkout (the recorded cwd is
    // the canonicalized root, so compare on the tail).
    let fetch = recorded
        .iter()
        .find(|(_, args, _)| args.first().map(String::as_str) == Some("fetch"))
        .expect("fetch recorded");
    assert!(
        fetch
            .2
            .as_deref()
            .is_some_and(|cwd| cwd.replace('\\', "/").ends_with("git/github.com/user/repo")),
        "{fetch:?}"
    );
}

#[tokio::test]
async fn rejects_an_unsupported_upstream_remote_when_reconciling() {
    let mut rig = Rig::new();
    let _target = existing_git_checkout(&rig, "repo");
    rig.with_runner(git_fake(&[(UPSTREAM_ABBREV_KEY, "upstream/main")], &[]));

    let error = rig
        .package_manager
        .install("git:github.com/user/repo", false)
        .await
        .expect_err("unsupported upstream");
    assert!(
        error
            .0
            .contains("Unsupported upstream remote: upstream/main"),
        "{error}"
    );
}

#[tokio::test]
async fn rejects_a_missing_upstream_branch_name_when_reconciling() {
    let mut rig = Rig::new();
    let _target = existing_git_checkout(&rig, "repo");
    rig.with_runner(git_fake(&[(UPSTREAM_ABBREV_KEY, "origin/")], &[]));

    let error = rig
        .package_manager
        .install("git:github.com/user/repo", false)
        .await
        .expect_err("empty branch name");
    assert!(error.0.contains("Missing upstream branch name"), "{error}");
}

/// No upstream branch configured: the reconciler falls back to the remote's
/// HEAD symref, reading the branch name from `symbolic-ref`.
#[tokio::test]
async fn falls_back_to_the_remote_head_symref_when_no_upstream_is_configured() {
    let mut rig = Rig::new();
    let target = existing_git_checkout(&rig, "repo");
    // The upstream probe itself fails, which is what sends the ladder to the
    // symref fallback.
    rig.with_runner(git_fake(
        &[
            (ORIGIN_HEAD_REV_KEY, "sym-head"),
            (SYMREF_KEY, "refs/remotes/origin/main"),
            (HEAD_REV_KEY, "local-head"),
            ("rev-parse origin/HEAD^{commit}", "sym-head"),
        ],
        &[UPSTREAM_ABBREV_KEY],
    ));

    rig.package_manager
        .install("git:github.com/user/repo", false)
        .await
        .expect("reconcile");

    let recorded = rig.package_manager_list_commands();
    assert!(ran_git(&recorded, UPSTREAM_FETCH_KEY), "{recorded:?}");
    assert!(ran_git(&recorded, ORIGIN_HEAD_RESET_KEY), "{recorded:?}");
    assert!(ran_git(&recorded, "clean -fdx"), "{recorded:?}");
    // The set-head probe rides the fallback path.
    assert!(
        ran_git(&recorded, "remote set-head origin -a"),
        "{recorded:?}"
    );
    let marker = target
        .parent()
        .expect("parent")
        .join(".repo.pi-update-incomplete");
    assert!(!marker.exists());
}

/// The symref is unreadable, so the reconciler fetches a hard HEAD mapping.
#[tokio::test]
async fn falls_back_to_a_hard_head_fetch_when_the_symref_is_unset() {
    let mut rig = Rig::new();
    let _target = existing_git_checkout(&rig, "repo");
    rig.with_runner(git_fake(
        &[
            (ORIGIN_HEAD_REV_KEY, "sym-head"),
            (HEAD_REV_KEY, "local-head"),
            ("rev-parse origin/HEAD^{commit}", "sym-head"),
        ],
        &[UPSTREAM_ABBREV_KEY, SYMREF_KEY],
    ));

    rig.package_manager
        .install("git:github.com/user/repo", false)
        .await
        .expect("reconcile");

    let recorded = rig.package_manager_list_commands();
    assert!(ran_git(&recorded, HARD_HEAD_FETCH_KEY), "{recorded:?}");
    assert!(ran_git(&recorded, ORIGIN_HEAD_RESET_KEY), "{recorded:?}");
    assert!(ran_git(&recorded, "clean -fdx"), "{recorded:?}");
}

#[tokio::test]
async fn update_reconciles_an_unpinned_git_checkout_through_the_pool() {
    let mut rig = Rig::new();
    existing_git_checkout(&rig, "repo");
    rig.set_packages(&json!(["git:github.com/user/repo"]));
    rig.with_runner(git_fake(
        &[
            (UPSTREAM_ABBREV_KEY, "origin/main"),
            (UPSTREAM_REV_KEY, "remote-head"),
            (HEAD_REV_KEY, "local-head"),
            (UPSTREAM_COMMIT_KEY, "remote-head"),
        ],
        &[],
    ));

    rig.package_manager
        .update(Some("git:github.com/user/repo"))
        .await
        .expect("update");

    let recorded = rig.package_manager_list_commands();
    assert!(ran_git(&recorded, UPSTREAM_FETCH_KEY), "{recorded:?}");
    assert!(ran_git(&recorded, UPSTREAM_RESET_KEY), "{recorded:?}");
    assert!(ran_git(&recorded, "clean -fdx"), "{recorded:?}");
}

#[tokio::test]
async fn update_installs_a_missing_git_checkout() {
    let mut rig = Rig::new();
    rig.set_packages(&json!(["git:github.com/user/repo"]));
    rig.with_runner(FakeRunner::new(Box::new(|command, args, _options| {
        if command == "git" && args.first().map(String::as_str) == Some("clone") {
            let target = args.get(2).cloned().unwrap_or_default();
            std::fs::create_dir_all(&target).expect("target dir");
        }
        Ok(FakeOutcome::ok())
    })));

    rig.package_manager
        .update(Some("git:github.com/user/repo"))
        .await
        .expect("update");

    let recorded = rig.package_manager_list_commands();
    let clone = recorded
        .iter()
        .find(|(command, args, _)| {
            command == "git" && args.first().map(String::as_str) == Some("clone")
        })
        .expect("clone recorded");
    assert_eq!(clone.1[1], "https://github.com/user/repo");
    assert!(
        clone
            .1
            .get(2)
            .is_some_and(|target| target.contains("git/github.com/user/repo")),
        "{clone:?}"
    );
}

/// A current checkout without a marker stays untouched: no reset, no clean,
/// the dependency-repair no-op.
#[tokio::test]
async fn update_keeps_a_current_git_checkout_untouched() {
    let mut rig = Rig::new();
    existing_git_checkout(&rig, "repo");
    rig.set_packages(&json!(["git:github.com/user/repo"]));
    rig.with_runner(git_fake(
        &[
            (UPSTREAM_ABBREV_KEY, "origin/main"),
            (UPSTREAM_REV_KEY, "same-head"),
            (HEAD_REV_KEY, "same-head"),
            (UPSTREAM_COMMIT_KEY, "same-head"),
        ],
        &[],
    ));

    rig.package_manager
        .update(Some("git:github.com/user/repo"))
        .await
        .expect("update");

    let recorded = rig.package_manager_list_commands();
    assert!(ran_git(&recorded, UPSTREAM_FETCH_KEY), "{recorded:?}");
    assert!(
        !ran_git(&recorded, "clean -fdx"),
        "a current checkout needs no clean: {recorded:?}"
    );
    assert!(
        !recorded
            .iter()
            .any(|(_, args, _)| args.first().map(String::as_str) == Some("reset")),
        "{recorded:?}"
    );
}

/// A leftover in-flight marker on a current checkout recovers through the
/// clean step, which removes the marker.
#[tokio::test]
async fn update_recovers_a_clean_state_when_the_incomplete_marker_exists() {
    let mut rig = Rig::new();
    let target = existing_git_checkout(&rig, "repo");
    let marker = target
        .parent()
        .expect("parent")
        .join(".repo.pi-update-incomplete");
    std::fs::write(&marker, "").expect("marker");
    rig.set_packages(&json!(["git:github.com/user/repo"]));
    rig.with_runner(git_fake(
        &[
            (UPSTREAM_ABBREV_KEY, "origin/main"),
            (UPSTREAM_REV_KEY, "same-head"),
            (HEAD_REV_KEY, "same-head"),
            (UPSTREAM_COMMIT_KEY, "same-head"),
        ],
        &[],
    ));

    rig.package_manager
        .update(Some("git:github.com/user/repo"))
        .await
        .expect("update");

    let recorded = rig.package_manager_list_commands();
    assert!(ran_git(&recorded, "clean -fdx"), "{recorded:?}");
    assert!(!marker.exists(), "the clean step removes the marker");
}

/// The pool swallows a failing git task — upstream's `.then(() => {})` — so
/// the update pass itself succeeds even when a clean fails.
#[tokio::test]
async fn update_swallows_a_failing_git_clean() {
    let mut rig = Rig::new();
    existing_git_checkout(&rig, "repo");
    rig.set_packages(&json!(["git:github.com/user/repo"]));
    rig.with_runner(git_fake(
        &[
            (UPSTREAM_ABBREV_KEY, "origin/main"),
            (UPSTREAM_REV_KEY, "remote-head"),
            (HEAD_REV_KEY, "local-head"),
            (UPSTREAM_COMMIT_KEY, "remote-head"),
        ],
        &["clean -fdx"],
    ));

    rig.package_manager
        .update(Some("git:github.com/user/repo"))
        .await
        .expect("the update pass reports success");
    let recorded = rig.package_manager_list_commands();
    assert!(ran_git(&recorded, "clean -fdx"), "{recorded:?}");
}

/// A temporary checkout refreshes through the Pull span when online; a
/// failed refresh would keep the cached checkout.
#[tokio::test]
async fn refreshes_a_temporary_git_checkout_when_online() {
    let mut rig = Rig::new();
    let git_source = pi_coding_agent::utils::git::GitSource {
        repo: "https://github.com/user/repo".to_string(),
        host: "github.com".to_string(),
        path: "user/repo".to_string(),
        r#ref: None,
        pinned: false,
    };
    let temp_root =
        pi_coding_agent::package_manager::get_extension_temp_folder(&rig.agent_dir).expect("temp");
    let root = temp_root.join("git-github.com");
    let hash = {
        use sha2::Digest as _;
        let mut hasher = sha2::Sha256::new();
        hasher.update(format!("git-github.com-{}", git_source.path));
        hasher.finalize()
    };
    let digest = format!("{hash:x}");
    let installed_path = root.join(&digest[..8]).join(&git_source.path);
    std::fs::create_dir_all(installed_path.join("extensions")).expect("temp checkout");
    let index = installed_path.join("extensions/index");
    std::fs::write(&index, "#!/bin/sh\n").expect("write index");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&index, std::fs::Permissions::from_mode(0o755))
            .expect("chmod index");
    }

    let events: Arc<Mutex<Vec<ProgressEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    rig.with_runner(git_fake(
        &[
            (UPSTREAM_ABBREV_KEY, "origin/main"),
            (UPSTREAM_REV_KEY, "remote-head"),
            (HEAD_REV_KEY, "local-head"),
            (UPSTREAM_COMMIT_KEY, "remote-head"),
        ],
        &[],
    ));
    rig.package_manager
        .set_progress_callback(Some(Box::new(move |event| {
            sink.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(event.clone());
        })));

    let result = rig
        .package_manager
        .resolve_extension_sources(&["git:github.com/user/repo".to_string()], false, true)
        .await
        .expect("resolve");
    assert!(find_with(&result.extensions, "extensions/index"));

    let recorded = rig.package_manager_list_commands();
    assert!(ran_git(&recorded, UPSTREAM_FETCH_KEY), "{recorded:?}");
    assert!(ran_git(&recorded, UPSTREAM_RESET_KEY), "{recorded:?}");
    let events = events
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert!(
        events
            .iter()
            .any(|event| event.action == ProgressAction::Pull
                && event.event_type == ProgressEventType::Start),
        "{events:?}"
    );
}

#[tokio::test]
async fn removes_a_git_checkout_its_marker_and_empty_parents() {
    let mut rig = Rig::new();
    rig.with_runner(FakeRunner::new(Box::new(|command, args, _options| {
        if command == "git" && args.first().map(String::as_str) == Some("clone") {
            let target = args.get(2).cloned().unwrap_or_default();
            std::fs::create_dir_all(&target).expect("target dir");
        }
        Ok(FakeOutcome::ok())
    })));
    rig.package_manager
        .install("git:github.com/user/repo", false)
        .await
        .expect("install");

    let git_root = rig.agent_dir.join("git");
    let host_dir = git_root.join("github.com");
    let marker = host_dir.join("user/.repo.pi-update-incomplete");
    std::fs::write(&marker, "").expect("marker");

    rig.package_manager
        .remove("git:github.com/user/repo", false)
        .await
        .expect("remove");

    assert!(!git_root.join("github.com/user/repo").exists());
    assert!(!marker.exists(), "the removal clears the marker");
    // The emptied parents prune up to the install root.
    assert!(!host_dir.join("user").exists(), "the emptied leaf prunes");
    assert!(!host_dir.exists(), "the emptied host dir prunes");
    assert!(git_root.exists(), "the install root itself stays");
}

#[tokio::test]
async fn prunes_against_a_deleted_install_root() {
    let mut rig = Rig::new();
    rig.with_runner(FakeRunner::new(Box::new(|command, args, _options| {
        if command == "git" && args.first().map(String::as_str) == Some("clone") {
            let target = args.get(2).cloned().unwrap_or_default();
            std::fs::create_dir_all(&target).expect("target dir");
        }
        Ok(FakeOutcome::ok())
    })));
    rig.package_manager
        .install("git:github.com/user/repo", false)
        .await
        .expect("install");

    // The install root vanished before the removal: the prune's canonicalize
    // falls back to the raw root and the walk must not panic.
    std::fs::remove_dir_all(rig.agent_dir.join("git")).expect("root removed");
    rig.package_manager
        .remove("git:github.com/user/repo", false)
        .await
        .expect("remove");
}

#[tokio::test]
async fn remove_clears_the_crate_tarball_and_local_channels() {
    let rig = Rig::new();
    std::fs::create_dir_all(rig.agent_dir.join("crates/example")).expect("crate install");
    std::fs::create_dir_all(
        rig.agent_dir
            .join("tarballs")
            .join(tarball_hash("https://example.com/tool-1.0.0.tgz")),
    )
    .expect("tarball install");

    rig.package_manager
        .remove("crate:example@1.0.0", false)
        .await
        .expect("crate removal");
    assert!(!rig.agent_dir.join("crates/example").exists());

    rig.package_manager
        .remove("https://example.com/tool-1.0.0.tgz", false)
        .await
        .expect("tarball removal");
    assert!(
        !rig.agent_dir
            .join("tarballs")
            .join(tarball_hash("https://example.com/tool-1.0.0.tgz"))
            .exists()
    );

    // A local source removes nothing: it resolves in place.
    rig.package_manager
        .remove("./whatever", false)
        .await
        .expect("local removal is a no-op");
    assert!(rig.global_packages().is_empty(), "settings untouched");
}

#[tokio::test]
async fn install_fails_when_creating_the_git_parent_fails() {
    let mut rig = Rig::new();
    // The git root pre-exists with its .gitignore, but the host segment is a
    // regular file — the parent creation for the checkout fails.
    std::fs::create_dir_all(rig.agent_dir.join("git")).expect("git root");
    rig.write("agent/git/.gitignore", "*\n!.gitignore\n");
    rig.write("agent/git/github.com", "a file, not a directory");
    rig.with_runner(FakeRunner::rejecting("no command may run"));

    let error = rig
        .package_manager
        .install("git:github.com/user/repo", false)
        .await
        .expect_err("parent creation fails");
    assert!(!error.0.is_empty(), "{error}");
    assert!(
        !rig.agent_dir.join("git/github.com/user/repo").exists(),
        "no checkout lands"
    );
}

#[tokio::test]
async fn install_git_surfaces_a_gitignore_creation_failure() {
    let mut rig = Rig::new();
    // A read-only agent dir cannot host the git install root.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&rig.agent_dir, std::fs::Permissions::from_mode(0o555))
            .expect("lock the agent dir");
    }
    rig.with_runner(FakeRunner::rejecting("no command may run"));

    let result = rig
        .package_manager
        .install("git:github.com/user/repo", false)
        .await;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&rig.agent_dir, std::fs::Permissions::from_mode(0o755))
            .expect("unlock the agent dir");
    }
    let error = result.expect_err("read-only agent dir fails");
    assert!(!error.0.is_empty(), "{error}");
}

#[tokio::test]
async fn install_git_surfaces_a_gitignore_write_failure() {
    let mut rig = Rig::new();
    // The git root exists but is read-only: the .gitignore write fails.
    std::fs::create_dir_all(rig.agent_dir.join("git")).expect("git root");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            rig.agent_dir.join("git"),
            std::fs::Permissions::from_mode(0o555),
        )
        .expect("lock the git root");
    }
    rig.with_runner(FakeRunner::rejecting("no command may run"));

    let result = rig
        .package_manager
        .install("git:github.com/user/repo", false)
        .await;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            rig.agent_dir.join("git"),
            std::fs::Permissions::from_mode(0o755),
        )
        .expect("unlock the git root");
    }
    let error = result.expect_err("read-only git root fails");
    assert!(!error.0.is_empty(), "{error}");
}

// =============================================================================
// The update pass's messaging, matching, and registry arms
// =============================================================================

#[tokio::test]
async fn reports_a_plain_message_when_no_configured_source_resembles_the_input() {
    let rig = Rig::new();
    // The configured set covers every suggestion-scan shape: an unparseable
    // entry, a tarball, a local path — none suggests for "crate:absent".
    rig.set_project_packages(&json!([
        "npm:broken-entry",
        "https://example.com/pkg.tgz",
        "/abs/local-pkg"
    ]));

    let error = rig
        .package_manager
        .update(Some("crate:absent"))
        .await
        .expect_err("no match");
    assert_eq!(
        error.0, "No matching package found for crate:absent",
        "no suggestion when nothing resembles the input"
    );
}

#[tokio::test]
async fn suggests_a_git_shorthand_with_ref() {
    let rig = Rig::new();
    rig.set_project_packages(&json!(["git:github.com/user/repo@v2"]));

    let error = rig
        .package_manager
        .update(Some("github.com/user/repo@v2"))
        .await
        .expect_err("no match");
    assert_eq!(
        error.0,
        "No matching package found for github.com/user/repo@v2. Did you mean git:github.com/user/repo@v2?"
    );
}

#[tokio::test]
async fn update_skips_pinned_crate_sources() {
    let mut rig = Rig::new();
    std::fs::create_dir_all(rig.temp_dir.join(".pi/crates/example")).expect("install dir");
    rig.write(
        ".pi/crates/example/pi-package-install.json",
        r#"{"kind":"pi-package-install","schemaVersion":1,"channel":"crate","source":"crate:example","resolvedVersion":"1.0.0","files":{}}"#,
    );
    rig.set_project_packages(&json!(["crate:example@1.0.0"]));
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(pi_ai::http::MockHttpClient::new());
    let runner = FakeRunner::rejecting("no command may run");
    rig.runner = Some(runner.clone());
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: Some(Arc::new(runner)),
        env: Some(common::env_with(&[])),
        http_client: Some(client),
    });

    rig.package_manager
        .update(Some("crate:example@1.0.0"))
        .await
        .expect("a pinned source needs no update");
    assert_eq!(
        rig.package_manager_list_commands().len(),
        0,
        "a pinned crate never reinstalls through update"
    );
}

/// A registry lookup failure preserves the update behavior — upstream's
/// catch — so the unpinned source reinstalls.
#[tokio::test]
async fn a_registry_lookup_failure_preserves_the_update() {
    let mut rig = Rig::new();
    std::fs::create_dir_all(rig.temp_dir.join(".pi/crates/example")).expect("install dir");
    rig.write(
        ".pi/crates/example/pi-package-install.json",
        r#"{"kind":"pi-package-install","schemaVersion":1,"channel":"crate","source":"crate:example","resolvedVersion":"1.0.0","files":{}}"#,
    );
    rig.set_project_packages(&json!(["crate:example"]));
    // The empty mock answers nothing: the registry lookup fails.
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(pi_ai::http::MockHttpClient::new());
    let runner = FakeRunner::new(fake_cargo_install());
    rig.runner = Some(runner.clone());
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: Some(Arc::new(runner)),
        env: Some(common::env_with(&[])),
        http_client: Some(client),
    });

    rig.package_manager
        .update(Some("crate:example"))
        .await
        .expect("update");
    let recorded = rig.package_manager_list_commands();
    assert_eq!(recorded.len(), 1, "the lookup failure still updates");
    assert_eq!(recorded[0].0, "cargo");
}

#[tokio::test]
async fn update_installs_a_missing_crate() {
    let mut rig = Rig::new();
    rig.set_project_packages(&json!(["crate:fresh"]));
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(pi_ai::http::MockHttpClient::new());
    let runner = FakeRunner::new(fake_cargo_install());
    rig.runner = Some(runner.clone());
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: Some(Arc::new(runner)),
        env: Some(common::env_with(&[])),
        http_client: Some(client),
    });

    rig.package_manager
        .update(Some("crate:fresh"))
        .await
        .expect("update");
    let recorded = rig.package_manager_list_commands();
    assert_eq!(recorded.len(), 1, "a missing install updates");
    assert_eq!(recorded[0].0, "cargo");
}

/// The registry check's error arms degrade to "no update reported" — each
/// failure shape rides through `check_for_available_updates`.
#[tokio::test]
async fn check_degrades_to_no_update_when_the_registry_is_unreachable() {
    let mut rig = Rig::new();
    std::fs::create_dir_all(rig.temp_dir.join(".pi/crates/example")).expect("install dir");
    rig.write(
        ".pi/crates/example/pi-package-install.json",
        r#"{"kind":"pi-package-install","schemaVersion":1,"channel":"crate","source":"crate:example","resolvedVersion":"1.0.0","files":{}}"#,
    );
    rig.set_project_packages(&json!(["crate:example"]));
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(pi_ai::http::MockHttpClient::new());
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: None,
        env: Some(common::env_with(&[])),
        http_client: Some(client),
    });

    let updates = rig
        .package_manager
        .check_for_available_updates()
        .await
        .expect("check");
    assert!(updates.is_empty(), "a transport failure reports nothing");
}

#[tokio::test]
async fn check_degrades_to_no_update_when_the_registry_body_fails_midstream() {
    let mut rig = Rig::new();
    std::fs::create_dir_all(rig.temp_dir.join(".pi/crates/example")).expect("install dir");
    rig.write(
        ".pi/crates/example/pi-package-install.json",
        r#"{"kind":"pi-package-install","schemaVersion":1,"channel":"crate","source":"crate:example","resolvedVersion":"1.0.0","files":{}}"#,
    );
    rig.set_project_packages(&json!(["crate:example"]));
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: None,
        env: Some(common::env_with(&[])),
        http_client: Some(Arc::new(FailingBodyClient)),
    });

    let updates = rig
        .package_manager
        .check_for_available_updates()
        .await
        .expect("check");
    assert!(updates.is_empty(), "a body failure reports nothing");
}

#[tokio::test]
async fn check_degrades_to_no_update_when_the_registry_answers_an_empty_body() {
    let mut rig = Rig::new();
    std::fs::create_dir_all(rig.temp_dir.join(".pi/crates/example")).expect("install dir");
    rig.write(
        ".pi/crates/example/pi-package-install.json",
        r#"{"kind":"pi-package-install","schemaVersion":1,"channel":"crate","source":"crate:example","resolvedVersion":"1.0.0","files":{}}"#,
    );
    rig.set_project_packages(&json!(["crate:example"]));
    let mock = pi_ai::http::MockHttpClient::new();
    mock.on(|request| request.url.contains("/api/v1/crates/example"))
        .respond(pi_ai::http::MockResponse::status(200));
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(mock);
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: None,
        env: Some(common::env_with(&[])),
        http_client: Some(client),
    });

    let updates = rig
        .package_manager
        .check_for_available_updates()
        .await
        .expect("check");
    assert!(updates.is_empty(), "an empty body reports nothing");
}

#[tokio::test]
async fn check_degrades_to_no_update_when_the_registry_answers_garbage() {
    let mut rig = Rig::new();
    std::fs::create_dir_all(rig.temp_dir.join(".pi/crates/example")).expect("install dir");
    rig.write(
        ".pi/crates/example/pi-package-install.json",
        r#"{"kind":"pi-package-install","schemaVersion":1,"channel":"crate","source":"crate:example","resolvedVersion":"1.0.0","files":{}}"#,
    );
    rig.set_project_packages(&json!(["crate:example"]));
    let mock = pi_ai::http::MockHttpClient::new();
    mock.on(|request| request.url.contains("/api/v1/crates/example"))
        .respond(pi_ai::http::MockResponse::status(200).with_body("not-json"));
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(mock);
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: None,
        env: Some(common::env_with(&[])),
        http_client: Some(client),
    });

    let updates = rig
        .package_manager
        .check_for_available_updates()
        .await
        .expect("check");
    assert!(updates.is_empty(), "garbage reports nothing");
}

#[tokio::test]
async fn check_degrades_to_no_update_when_the_registry_omits_max_version() {
    let mut rig = Rig::new();
    std::fs::create_dir_all(rig.temp_dir.join(".pi/crates/example")).expect("install dir");
    rig.write(
        ".pi/crates/example/pi-package-install.json",
        r#"{"kind":"pi-package-install","schemaVersion":1,"channel":"crate","source":"crate:example","resolvedVersion":"1.0.0","files":{}}"#,
    );
    rig.set_project_packages(&json!(["crate:example"]));
    let mock = pi_ai::http::MockHttpClient::new();
    mock.on(|request| request.url.contains("/api/v1/crates/example"))
        .respond(pi_ai::http::json_response(200, &json!({ "crate": {} })));
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(mock);
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: None,
        env: Some(common::env_with(&[])),
        http_client: Some(client),
    });

    let updates = rig
        .package_manager
        .check_for_available_updates()
        .await
        .expect("check");
    assert!(updates.is_empty(), "a missing max_version reports nothing");
}

// =============================================================================
// The update check's git arms
// =============================================================================

fn rig_with_git_check(rig: &mut Rig, runner: FakeRunner) {
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(pi_ai::http::MockHttpClient::new());
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: Some(Arc::new(runner)),
        env: Some(common::env_with(&[])),
        http_client: Some(client),
    });
}

#[tokio::test]
async fn check_reports_a_git_update_when_the_remote_moved() {
    let mut rig = Rig::new();
    existing_git_checkout(&rig, "repo");
    rig.set_packages(&json!(["git:github.com/user/repo"]));
    rig_with_git_check(
        &mut rig,
        git_fake(
            &[
                (HEAD_REV_KEY, "local-head"),
                (UPSTREAM_ABBREV_KEY, "origin/main"),
                (
                    LS_REMOTE_UPSTREAM_KEY,
                    &format!("{FAKE_HEAD_A}\trefs/heads/main"),
                ),
            ],
            &[],
        ),
    );

    let updates = rig
        .package_manager
        .check_for_available_updates()
        .await
        .expect("check");
    assert_eq!(updates.len(), 1, "{updates:?}");
    assert_eq!(updates[0].source, "git:github.com/user/repo");
    assert_eq!(updates[0].display_name, "github.com/user/repo");
    assert_eq!(updates[0].update_type.as_str(), "git");
    assert_eq!(updates[0].scope, SourceScope::User);
}

#[tokio::test]
async fn check_falls_back_to_the_remote_head_symref() {
    let mut rig = Rig::new();
    existing_git_checkout(&rig, "repo");
    rig.set_packages(&json!(["git:github.com/user/repo"]));
    // The upstream branch's ls-remote answers without a hash line, so the
    // check falls back to the HEAD symref query.
    rig_with_git_check(
        &mut rig,
        git_fake(
            &[
                (HEAD_REV_KEY, "local-head"),
                (UPSTREAM_ABBREV_KEY, "origin/main"),
                (
                    LS_REMOTE_HEAD_KEY,
                    &format!("{FAKE_HEAD_B}\trefs/heads/other\n{FAKE_HEAD_A}\tHEAD"),
                ),
            ],
            &[],
        ),
    );

    let updates = rig
        .package_manager
        .check_for_available_updates()
        .await
        .expect("check");
    assert_eq!(updates.len(), 1, "{updates:?}");
    assert_eq!(updates[0].update_type.as_str(), "git");
}

#[tokio::test]
async fn check_reports_no_update_when_the_remote_head_matches() {
    let mut rig = Rig::new();
    existing_git_checkout(&rig, "repo");
    rig.set_packages(&json!(["git:github.com/user/repo"]));
    rig_with_git_check(
        &mut rig,
        git_fake(
            &[
                (HEAD_REV_KEY, FAKE_HEAD_A),
                (UPSTREAM_ABBREV_KEY, "origin/main"),
                (
                    LS_REMOTE_UPSTREAM_KEY,
                    &format!("{FAKE_HEAD_A}\trefs/heads/main"),
                ),
            ],
            &[],
        ),
    );

    let updates = rig
        .package_manager
        .check_for_available_updates()
        .await
        .expect("check");
    assert!(updates.is_empty(), "matching heads report nothing");
}

#[tokio::test]
async fn check_tolerates_a_failing_git_probe() {
    let mut rig = Rig::new();
    existing_git_checkout(&rig, "repo");
    rig.set_packages(&json!(["git:github.com/user/repo"]));
    // The local rev-parse fails: the check degrades to no update.
    rig_with_git_check(
        &mut rig,
        git_fake(&[(UPSTREAM_ABBREV_KEY, "origin/main")], &[HEAD_REV_KEY]),
    );

    let updates = rig
        .package_manager
        .check_for_available_updates()
        .await
        .expect("check");
    assert!(updates.is_empty(), "a failing probe reports nothing");
}

#[tokio::test]
async fn check_determines_no_remote_head_when_ls_remote_answers_nothing() {
    let mut rig = Rig::new();
    existing_git_checkout(&rig, "repo");
    rig.set_packages(&json!(["git:github.com/user/repo"]));
    // Both ls-remote queries answer empty: no remote HEAD exists, so the
    // head lookup fails and the check degrades to no update.
    rig_with_git_check(
        &mut rig,
        git_fake(
            &[
                (HEAD_REV_KEY, "local-head"),
                (UPSTREAM_ABBREV_KEY, "origin/main"),
            ],
            &[],
        ),
    );

    let updates = rig
        .package_manager
        .check_for_available_updates()
        .await
        .expect("check");
    assert!(updates.is_empty(), "a missing remote HEAD reports nothing");
}

// =============================================================================
// Receipts, install error surfaces, and the small seams
// =============================================================================

#[tokio::test]
async fn from_value_rejects_a_receipt_without_a_channel() {
    let receipt = json!({
        "kind": "pi-package-install",
        "schemaVersion": 1,
        "files": {}
    });
    let error = InstallReceipt::from_value(&receipt).expect_err("no channel");
    assert_eq!(error.0, "Install receipt carries no channel");
}

#[tokio::test]
async fn the_crate_install_surfaces_a_rename_failure_when_the_stage_never_lands() {
    let mut rig = Rig::new();
    // The fake cargo "succeeds" without staging anything: the atomic rename
    // then fails on the missing stage and the error surfaces.
    rig.with_runner(FakeRunner::new(Box::new(|_command, _args, _options| {
        Ok(FakeOutcome::ok())
    })));

    let error = rig
        .package_manager
        .install("crate:example@1.0.0", false)
        .await
        .expect_err("rename fails");
    assert!(!error.0.is_empty(), "{error}");
    assert!(
        !rig.agent_dir.join("crates/example").exists(),
        "no install lands from a missing stage"
    );
}

#[tokio::test]
async fn the_receipt_write_failure_surfaces_when_the_package_plants_a_receipt_directory() {
    let mut rig = Rig::new();
    // The staged package carries a DIRECTORY where the receipt file must
    // land: the post-rename receipt write fails closed.
    rig.with_runner(FakeRunner::new(Box::new(|command, args, _options| {
        if command == "cargo" && args.first().map(String::as_str) == Some("install") {
            let root_index = args
                .iter()
                .position(|arg| arg == "--root")
                .expect("--root present");
            let stage_root = args.get(root_index + 1).cloned().unwrap_or_default();
            std::fs::create_dir_all(Path::new(&stage_root).join("pi-package-install.json"))
                .expect("receipt directory");
        }
        Ok(FakeOutcome::ok())
    })));

    let error = rig
        .package_manager
        .install("crate:example@1.0.0", false)
        .await
        .expect_err("receipt write fails");
    assert!(!error.0.is_empty(), "{error}");
}

#[tokio::test]
async fn install_fails_when_the_agent_dir_cannot_host_the_roots() {
    let rig = Rig::new();
    // The agent dir path names a regular file: no install root under it can
    // exist.
    let agent_file = rig.write("agent-file", "not a directory");
    let package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: agent_file.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: None,
        env: Some(common::env_with(&[])),
        http_client: None,
    });

    let error = package_manager
        .install("crate:blocker", false)
        .await
        .expect_err("crate root cannot exist");
    assert!(!error.0.is_empty(), "{error}");

    let error = package_manager
        .install("https://example.com/tool-1.0.0.tgz", false)
        .await
        .expect_err("tarball root cannot exist");
    assert!(!error.0.is_empty(), "{error}");

    // The temporary dir tree rides the same agent dir: the temp folder
    // creation fails the same way.
    let git_source = pi_coding_agent::utils::git::GitSource {
        repo: "https://github.com/user/repo".to_string(),
        host: "github.com".to_string(),
        path: "user/repo".to_string(),
        r#ref: None,
        pinned: false,
    };
    let error = package_manager
        .get_git_install_path(&git_source, SourceScope::Temporary)
        .expect_err("temp folder cannot exist");
    assert!(!error.0.is_empty(), "{error}");
}

#[tokio::test]
async fn install_and_persist_then_remove_and_persist_round_trip() {
    let mut rig = Rig::new();
    rig.with_runner(FakeRunner::new(fake_cargo_install()));

    rig.package_manager
        .install_and_persist("crate:example@1.0.0", false)
        .await
        .expect("install and persist");
    assert_eq!(
        rig.global_packages(),
        vec![json!("crate:example@1.0.0")],
        "the persisted entry lands in settings"
    );
    assert!(
        rig.agent_dir
            .join("crates/example/pi-package-install.json")
            .exists(),
        "the install carries its receipt"
    );

    let removed = rig
        .package_manager
        .remove_and_persist("crate:example@1.0.0", false)
        .await
        .expect("remove and persist");
    assert!(removed);
    assert!(rig.global_packages().is_empty());
    assert!(
        !rig.agent_dir.join("crates/example").exists(),
        "the removal clears the install"
    );
}

#[tokio::test]
async fn install_reports_missing_and_existing_local_paths() {
    let rig = Rig::new();
    let error = rig
        .package_manager
        .install("./definitely-missing-dir", false)
        .await
        .expect_err("missing path");
    assert!(error.0.contains("Path does not exist"), "{error}");

    let _dir = rig.mkdir("local-ok");
    rig.package_manager
        .install("./local-ok", false)
        .await
        .expect("existing path installs");
}

#[tokio::test]
async fn a_prefixless_delta_pattern_enables_by_glob() {
    let rig = Rig::new();
    rig.write_executable("delta-pkg/extensions/foo");
    rig.write_executable("delta-pkg/extensions/bar");
    rig.set_project_packages(&json!([{
        "source": "../delta-pkg",
        "autoload": false,
        "extensions": ["extensions/foo"],
    }]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    let foo = result
        .extensions
        .iter()
        .find(|resource| {
            same_path(
                &resource.path,
                &rig.temp_dir.join("delta-pkg/extensions/foo"),
            )
        })
        .expect("the pattern's match registers");
    assert!(foo.enabled);
    assert_eq!(foo.metadata.scope, SourceScope::Project);
    // The prefixless pattern registers only its matches; the sibling stays
    // out of the delta entirely.
    assert!(
        !result
            .extensions
            .iter()
            .any(|resource| resource.path.contains("bar")),
        "{:?}",
        result
            .extensions
            .iter()
            .map(|resource| resource.path.clone())
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn manifest_glob_entries_expand_for_plain_slash_and_absolute_spellings() {
    let rig = Rig::new();
    let pkg_dir = rig.temp_dir.join("glob-spellings-pkg");
    rig.write_executable("glob-spellings-pkg/top-a");
    rig.write_executable("glob-spellings-pkg/top-b");
    rig.write_executable("glob-spellings-pkg/sub/x");
    rig.write_executable("glob-spellings-pkg/inside/y");
    rig.write_executable("glob-spellings-pkg/qxmark");
    rig.write(
        "glob-spellings-pkg/package.json",
        r#"{"name":"glob-spellings-pkg","pi":{"extensions":["top*","./sub/*","/inside/*","q?mark"]}}"#,
    );

    let result = rig
        .package_manager
        .resolve_extension_sources(&[pkg_dir.to_string_lossy().into_owned()], false, false)
        .await
        .expect("resolve");
    let mut names: Vec<String> = result
        .extensions
        .iter()
        .map(|resource| {
            Path::new(&resource.path)
                .strip_prefix(&pkg_dir)
                .expect("under pkg")
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec!["inside/y", "qxmark", "sub/x", "top-a", "top-b"],
        "every glob spelling expands under the package root: {names:?}"
    );
}

#[tokio::test]
async fn a_subdirectory_bin_convention_surfaces_through_auto_discovery() {
    let rig = Rig::new();
    // No package.json and no index in the subdirectory: the bin convention
    // is its entry declaration.
    rig.write_executable("agent/extensions/mytool/bin/tool");

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_with(&result.extensions, "mytool/bin/tool"));
}

#[tokio::test]
async fn an_unfiltered_type_collects_through_the_manifest_or_the_convention_dir() {
    let rig = Rig::new();
    // Package A filters only extensions; its manifest declares skills, so
    // skills collect through the manifest arm.
    rig.write_executable("manifest-arm-pkg/extensions/only");
    rig.write(
        "manifest-arm-pkg/skills/manifest-skill/SKILL.md",
        "---\nname: manifest-skill\ndescription: Manifest\n---\n",
    );
    rig.write(
        "manifest-arm-pkg/package.json",
        r#"{"name":"manifest-arm-pkg","pi":{"skills":["./skills"]}}"#,
    );
    // Package B filters only prompts; with no manifest, its themes collect
    // through the convention directory.
    rig.write("convention-arm-pkg/prompts/review.md", "Review");
    rig.write("convention-arm-pkg/themes/dark.json", "{}");

    let manifest_arm = rig.temp_dir.join("manifest-arm-pkg");
    let convention_arm = rig.temp_dir.join("convention-arm-pkg");
    rig.set_packages(&json!([
        {
            "source": manifest_arm.to_string_lossy().into_owned(),
            "extensions": ["extensions/only"],
        },
        {
            "source": convention_arm.to_string_lossy().into_owned(),
            "prompts": [],
        },
    ]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(find_with(&result.skills, "manifest-skill/SKILL.md"));
    assert!(find_with(&result.themes, "dark.json"));
}

#[tokio::test]
async fn a_bad_file_url_degrades_to_the_raw_input() {
    let rig = Rig::new();
    // `file://host/...` cannot convert to a local path; the resolver's
    // failure node degrades to the raw input, which resolves to nothing.
    let result = rig
        .package_manager
        .resolve_extension_sources(&["file://host/pkg".to_string()], false, false)
        .await
        .expect("the failure node degrades");
    assert!(result.extensions.is_empty());
}

#[tokio::test]
async fn the_error_and_origin_helpers_render_their_wire_forms() {
    use pi_coding_agent::package_manager::ResourceOrigin;

    let error = PackageManagerError::from("boom".to_string());
    assert_eq!(error.0, "boom");
    assert_eq!(format!("{error}"), "boom");

    assert_eq!(ResourceOrigin::Package.as_str(), "package");
    assert_eq!(ResourceOrigin::TopLevel.as_str(), "top-level");
}

#[tokio::test]
async fn the_debug_impls_render_their_struct_names() {
    let rig = Rig::new();
    let options = PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: None,
        env: None,
        http_client: None,
    };
    assert!(format!("{options:?}").contains("PackageManagerOptions"));
    assert!(format!("{:?}", rig.package_manager).contains("DefaultPackageManager"));
}

#[tokio::test]
async fn an_object_source_entry_round_trips_through_to_value() {
    let entry = json!({
        "source": "git:github.com/user/repo",
        "autoload": false,
        "extensions": ["extensions/main"],
        "skills": [],
    });
    let view = PackageSourceView::parse(&entry);
    assert_eq!(view.source(), "git:github.com/user/repo");
    let round_tripped = PackageSourceView::parse(&view.to_value());
    assert_eq!(view, round_tripped);
    assert!(
        view.filter()
            .is_some_and(|filter| filter.autoload == Some(false))
    );

    let plain = PackageSourceView::parse(&json!("crate:example"));
    assert_eq!(
        plain.to_value(),
        json!("crate:example"),
        "a plain entry round-trips as its string"
    );
}

// =============================================================================
// The tarball resolve flow and the installed-path surfaces
// =============================================================================

#[tokio::test]
async fn resolve_installs_a_missing_tarball_source() {
    let mut rig = Rig::new();
    let url = "https://example.com/tool-1.0.0.tgz";
    let mock = pi_ai::http::MockHttpClient::new();
    mock.on(move |request| request.url == url).respond(
        pi_ai::http::MockResponse::status(200).with_body(gz_tarball(vec![(
            "bin/tool",
            b"#!/bin/sh\n".as_slice(),
            0o755,
        )])),
    );
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(mock);
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: None,
        env: Some(common::env_with(&[])),
        http_client: Some(client),
    });
    rig.set_packages(&json!([url]));

    let result = rig.package_manager.resolve(None).await.expect("resolve");
    assert!(
        find_with(&result.extensions, "bin/tool"),
        "the installed tarball's binaries surface: {:?}",
        result
            .extensions
            .iter()
            .map(|resource| resource.path.clone())
            .collect::<Vec<_>>()
    );
    assert!(
        result.extensions[0]
            .metadata
            .base_dir
            .as_deref()
            .is_some_and(|base| base.contains("tarballs")),
        "the package root rides as the base dir"
    );
    assert!(
        rig.agent_dir
            .join("tarballs")
            .join(tarball_hash(url))
            .join("pi-package-install.json")
            .exists(),
        "the resolve-driven install carries its receipt"
    );
}

#[tokio::test]
async fn list_configured_packages_reports_user_scoped_install_paths() {
    let mut rig = Rig::new();
    let runner = FakeRunner::new(fake_cargo_install());
    rig.runner = Some(runner.clone());
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: Some(Arc::new(runner)),
        env: Some(common::env_with(&[])),
        http_client: Some(Arc::new(pi_ai::http::MockHttpClient::new())),
    });
    rig.set_packages(&json!(["crate:installed"]));
    rig.set_project_packages(&json!(["crate:installed"]));

    rig.package_manager
        .install("crate:installed", false)
        .await
        .expect("user install");

    let configured = rig.package_manager.list_configured_packages();
    assert_eq!(configured.len(), 2, "{configured:?}");
    // The user entry is installed; the project entry with the same source is
    // not (it stages under .pi/crates).
    let user_entry = configured
        .iter()
        .find(|entry| entry.scope == SourceScope::User)
        .expect("user entry");
    assert_eq!(user_entry.source, "crate:installed");
    assert!(
        user_entry
            .installed_path
            .as_deref()
            .is_some_and(|path| path.contains("crates/installed")),
        "{user_entry:?}"
    );
    let project_entry = configured
        .iter()
        .find(|entry| entry.scope == SourceScope::Project)
        .expect("project entry");
    assert_eq!(project_entry.installed_path, None);
}

#[tokio::test]
async fn get_installed_path_reports_each_channel() {
    let rig = Rig::new();
    // Git: the checkout exists, so the path reports.
    let checkout = rig.agent_dir.join("git/github.com/user/repo");
    std::fs::create_dir_all(&checkout).expect("checkout dir");
    let git_source = pi_coding_agent::utils::git::GitSource {
        repo: "https://github.com/user/repo".to_string(),
        host: "github.com".to_string(),
        path: "user/repo".to_string(),
        r#ref: None,
        pinned: false,
    };
    let git_path = rig
        .package_manager
        .get_installed_path("git:github.com/user/repo", SourceScope::User)
        .expect("parse");
    assert!(
        git_path
            .as_deref()
            .is_some_and(|path| path.contains("git/github.com/user/repo")),
        "{git_path:?}"
    );
    let _ = git_source;

    // Tarball and local: only existing paths report.
    let tarball_source = TarballSource {
        url: "https://example.com/tool-1.0.0.tgz".to_string(),
    };
    std::fs::create_dir_all(
        rig.agent_dir
            .join("tarballs")
            .join(tarball_hash(&tarball_source.url)),
    )
    .expect("tarball dir");
    let tarball_path = rig
        .package_manager
        .get_installed_path(&tarball_source.url, SourceScope::User)
        .expect("parse");
    assert!(
        tarball_path
            .as_deref()
            .is_some_and(|path| path.contains("tarballs/")),
        "{tarball_path:?}"
    );

    // The user scope's local base is the agent dir.
    let _dir = rig.mkdir("agent/local-pkg");
    let local_path = rig
        .package_manager
        .get_installed_path("./local-pkg", SourceScope::User)
        .expect("parse");
    assert!(local_path.is_some(), "{local_path:?}");

    let missing = rig
        .package_manager
        .get_installed_path("./definitely-missing", SourceScope::User)
        .expect("parse");
    assert_eq!(missing, None);
}

// =============================================================================
// The pinned git clone and update arms
// =============================================================================

#[tokio::test]
async fn installs_a_pinned_git_checkout_by_cloning_and_checking_out() {
    let mut rig = Rig::new();
    rig.with_runner(FakeRunner::new(Box::new(|command, args, _options| {
        if command == "git" && args.first().map(String::as_str) == Some("clone") {
            let target = args.get(2).cloned().unwrap_or_default();
            std::fs::create_dir_all(&target).expect("target dir");
        }
        Ok(FakeOutcome::ok())
    })));

    rig.package_manager
        .install("git:github.com/user/repo@v2", false)
        .await
        .expect("install");

    let recorded = rig.package_manager_list_commands();
    let checkout = recorded
        .iter()
        .find(|(command, args, _)| {
            command == "git" && args.first().map(String::as_str) == Some("checkout")
        })
        .expect("checkout recorded");
    assert_eq!(checkout.1[1], "v2");
    assert!(
        checkout
            .2
            .as_deref()
            .is_some_and(|cwd| cwd.replace('\\', "/").ends_with("git/github.com/user/repo")),
        "the checkout runs inside the fresh clone: {checkout:?}"
    );
}

#[tokio::test]
async fn update_reconciles_a_pinned_git_checkout() {
    let mut rig = Rig::new();
    existing_git_checkout(&rig, "repo");
    rig.set_packages(&json!(["git:github.com/user/repo@v2"]));
    rig.with_runner(git_fake(
        &[
            (HEAD_REV_KEY, "old-head"),
            ("rev-parse FETCH_HEAD^{commit}", "new-head"),
        ],
        &[],
    ));

    rig.package_manager
        .update(Some("git:github.com/user/repo@v2"))
        .await
        .expect("update");

    let recorded = rig.package_manager_list_commands();
    assert!(ran_git(&recorded, "fetch origin v2"), "{recorded:?}");
    assert!(
        ran_git(&recorded, "reset --hard FETCH_HEAD^{commit}"),
        "{recorded:?}"
    );
    assert!(ran_git(&recorded, "clean -fdx"), "{recorded:?}");
}

// =============================================================================
// The unpack's adversarial entry orders and the staging failure arms
// =============================================================================

/// One archive with a single entry, built through `tar::Builder` — the
/// adversarial orders the unpack's per-entry error arms answer.
fn gz_tarball_with(
    build: impl FnOnce(&mut tar::Builder<flate2::write::GzEncoder<Vec<u8>>>),
) -> Vec<u8> {
    let encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    let mut builder = tar::Builder::new(encoder);
    build(&mut builder);
    builder.into_inner().expect("tar").finish().expect("gz")
}

fn file_entry(builder: &mut tar::Builder<flate2::write::GzEncoder<Vec<u8>>>, name: &str) {
    let mut header = tar::Header::new_gnu();
    header.set_size(1);
    header.set_mode(0o644);
    header.set_entry_type(tar::EntryType::Regular);
    header.set_cksum();
    builder
        .append_data(&mut header, name, b"x".as_slice())
        .expect("file entry");
}

fn dir_entry(builder: &mut tar::Builder<flate2::write::GzEncoder<Vec<u8>>>, name: &str) {
    let mut header = tar::Header::new_gnu();
    header.set_size(0);
    header.set_mode(0o755);
    header.set_entry_type(tar::EntryType::Directory);
    header.set_cksum();
    builder
        .append_data(&mut header, name, std::io::empty())
        .expect("dir entry");
}

fn symlink_entry(
    builder: &mut tar::Builder<flate2::write::GzEncoder<Vec<u8>>>,
    name: &str,
    target: &str,
) {
    let mut header = tar::Header::new_gnu();
    header.set_size(0);
    header.set_entry_type(tar::EntryType::Symlink);
    header.set_link_name(target).expect("link name");
    header.set_cksum();
    builder
        .append_data(&mut header, name, std::io::empty())
        .expect("symlink entry");
}

#[tokio::test]
async fn unpack_tarball_rejects_adversarial_entry_orders() {
    let stage = tempfile::tempdir().expect("stage");

    // A file lands where a directory already sits: the write fails.
    let archive = gz_tarball_with(|builder| {
        dir_entry(builder, "x");
        file_entry(builder, "x");
    });
    let error = pi_coding_agent::package_manager::unpack_tarball(&archive, stage.path())
        .expect_err("file-over-dir rejects");
    assert!(!error.0.is_empty(), "{error}");

    // A directory lands where a file already sits: the create fails.
    let archive = gz_tarball_with(|builder| {
        file_entry(builder, "y");
        dir_entry(builder, "y");
    });
    let error = pi_coding_agent::package_manager::unpack_tarball(&archive, stage.path())
        .expect_err("dir-over-file rejects");
    assert!(!error.0.is_empty(), "{error}");

    // A nested file's parent is a file: the parent creation fails.
    let archive = gz_tarball_with(|builder| {
        file_entry(builder, "z");
        file_entry(builder, "z/child");
    });
    let error = pi_coding_agent::package_manager::unpack_tarball(&archive, stage.path())
        .expect_err("file-parent rejects");
    assert!(!error.0.is_empty(), "{error}");

    // Two symlinks share a name: the second creation fails.
    let archive = gz_tarball_with(|builder| {
        symlink_entry(builder, "link", "target");
        symlink_entry(builder, "link", "target");
    });
    let error = pi_coding_agent::package_manager::unpack_tarball(&archive, stage.path())
        .expect_err("duplicate symlink rejects");
    assert!(!error.0.is_empty(), "{error}");
}

#[tokio::test]
async fn the_tarball_stage_surfaces_a_creation_failure() {
    let mut rig = Rig::new();
    // The tarballs root pre-exists read-only with its .gitignore, so the
    // ensure-git-ignore pass passes and the stage creation fails.
    std::fs::create_dir_all(rig.agent_dir.join("tarballs")).expect("tarballs root");
    rig.write("agent/tarballs/.gitignore", "*\n!.gitignore\n");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            rig.agent_dir.join("tarballs"),
            std::fs::Permissions::from_mode(0o555),
        )
        .expect("lock the tarballs root");
    }
    let client: Arc<dyn pi_ai::http::HttpClient> = Arc::new(pi_ai::http::MockHttpClient::new());
    rig.package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: rig.temp_dir.to_string_lossy().into_owned(),
        agent_dir: rig.agent_dir.to_string_lossy().into_owned(),
        settings: Arc::clone(&rig.settings),
        command_runner: None,
        env: Some(common::env_with(&[])),
        http_client: Some(client),
    });

    let result = rig
        .package_manager
        .install("https://example.com/tool-1.0.0.tgz", false)
        .await;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            rig.agent_dir.join("tarballs"),
            std::fs::Permissions::from_mode(0o755),
        )
        .expect("unlock the tarballs root");
    }
    let error = result.expect_err("read-only tarballs root fails");
    assert!(!error.0.is_empty(), "{error}");
}

#[tokio::test]
async fn the_git_marker_write_failure_surfaces() {
    let mut rig = Rig::new();
    let target = existing_git_checkout(&rig, "repo");
    // The checkout's parent is read-only: the in-flight marker write fails.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            target.parent().expect("parent"),
            std::fs::Permissions::from_mode(0o555),
        )
        .expect("lock the parent");
    }
    rig.with_runner(git_fake(
        &[
            (UPSTREAM_ABBREV_KEY, "origin/main"),
            (UPSTREAM_REV_KEY, "remote-head"),
            (HEAD_REV_KEY, "local-head"),
            (UPSTREAM_COMMIT_KEY, "remote-head"),
        ],
        &[],
    ));

    let result = rig
        .package_manager
        .install("git:github.com/user/repo", false)
        .await;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            target.parent().expect("parent"),
            std::fs::Permissions::from_mode(0o755),
        )
        .expect("unlock the parent");
    }
    let error = result.expect_err("marker write fails");
    assert!(!error.0.is_empty(), "{error}");
}
