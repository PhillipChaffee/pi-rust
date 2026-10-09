//! The config foundation, upstream's `packages/coding-agent/src/config.ts`.
//!
//! The foundation carries the install-detection probes restated as
//! build-time constants, the package.json-derived app constants, the
//! agent-dir derivation and its on-disk layout, and the session-directory
//! cwd encoding from upstream's `src/core/session-manager.ts`.
//!
//! Upstream reads its app constants out of `package.json` (`piConfig`) so a
//! fork can rename the app and the config directory at runtime; the Rust
//! crate has no manifest to read there, so the values are build-time
//! constants — the fork-rename mechanism has no runtime counterpart. The
//! self-update surface redesigns upstream's npm/pnpm/yarn/bun command
//! builders for the single native binary: cargo's install is the one
//! managed method, the installer-managed layout rides the package-manager
//! CLI, and the detection probes take the executable path and environment
//! through `_with` seams.
//!
//! The environment seam mirrors the tui crate's: Rust cannot mutate the
//! process environment without the `unsafe` this workspace forbids, so the
//! `_with` variants inject an [`EnvLookup`] and the plain getters read the
//! real environment.

use std::path::{Path, PathBuf};

use crate::utils::paths::{expand_tilde, resolve_path};

// =============================================================================
// Install detection, restated as build-time constants
// =============================================================================

/// Whether the running build is a Bun compiled binary — upstream's
/// `isBunBinary` probe over `import.meta.url`'s `$bunfs`/`~BUN`/`%7EBUN`
/// virtual-filesystem markers. A Rust build is never one.
pub const IS_BUN_BINARY: bool = false;

/// Whether Bun is the runtime — upstream's `isBunRuntime` probe over
/// `process.versions.bun`. A Rust build is never one.
pub const IS_BUN_RUNTIME: bool = false;

/// Whether the build is the esbuild-bundled Node distribution — upstream's
/// `isBundledNode` probe over the `PI_BUNDLED_NODE` build define. A Rust
/// build is never one.
pub const IS_BUNDLED_NODE: bool = false;

// =============================================================================
// App constants, upstream's package.json reads
// =============================================================================

/// The package name, upstream's `PACKAGE_NAME` (`package.json` `name`,
/// falling back to `@earendil-works/pi-coding-agent`), restated as the crate
/// manifest's name.
pub const PACKAGE_NAME: &str = env!("CARGO_PKG_NAME");

/// The user-facing app name, upstream's `APP_NAME` (`package.json`
/// `piConfig.name`, falling back to `"pi"`).
pub const APP_NAME: &str = "pi";

/// The display title, upstream's `APP_TITLE`: the app name when a
/// `piConfig.name` override is present, `"π"` otherwise.
pub const APP_TITLE: &str = "π";

/// The config directory under the home directory, upstream's
/// `CONFIG_DIR_NAME` (`package.json` `piConfig.configDir`, falling back to
/// `".pi"`).
pub const CONFIG_DIR_NAME: &str = ".pi";

/// The release version, upstream's `VERSION` (`package.json` `version`),
/// restated as the crate manifest's version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The agent-dir override variable, upstream's `ENV_AGENT_DIR`.
///
/// Upstream derives the name as `` `${APP_NAME.toUpperCase()}_CODING_AGENT_DIR` ``
/// — the reason a fork renaming the app also renames the variable, which the
/// constants above cannot do at runtime.
pub const ENV_AGENT_DIR: &str = "PI_CODING_AGENT_DIR";

/// The session-dir override variable, upstream's `ENV_SESSION_DIR`, derived
/// the same way upstream derives [`ENV_AGENT_DIR`].
pub const ENV_SESSION_DIR: &str = "PI_CODING_AGENT_SESSION_DIR";

// =============================================================================
// Environment seam
// =============================================================================

/// The environment lookup upstream resolved through `process.env`.
///
/// Rust cannot mutate the process environment without the `unsafe` this
/// workspace forbids, so tests inject a map-backed lookup and the process
/// default reads the real environment. `Send + Sync` because consumers ride
/// futures the stores share across waiters.
pub type EnvLookup = Box<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// The process environment, upstream's `process.env` default.
#[must_use]
pub fn default_env_lookup() -> EnvLookup {
    Box::new(|key| std::env::var(key).ok())
}

/// The home directory, upstream's `os.homedir()`. Degrades to the empty
/// string when the platform cannot locate one, the same degenerate input
/// Node's `homedir()` hands its joiners.
#[must_use]
pub fn home_dir() -> String {
    std::env::home_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

/// The process working directory, upstream's `process.cwd()`. Degrades to
/// the empty string when the platform cannot report one (Node throws); the
/// only consumer is the resolver base below.
pub(crate) fn process_cwd() -> String {
    std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

// =============================================================================
// Agent config paths (~/.pi/agent/*)
// =============================================================================

/// The agent config directory, upstream's `getAgentDir()`.
///
/// [`ENV_AGENT_DIR`] wins when set (tilde-expanded, upstream's
/// `expandTildePath`; a relative value stays relative), else
/// `<home>/<CONFIG_DIR_NAME>/agent` — `~/.pi/agent`.
#[must_use]
pub fn get_agent_dir() -> PathBuf {
    get_agent_dir_with(&default_env_lookup())
}

/// [`get_agent_dir`] over an injected environment lookup, the test seam for
/// the override behavior.
#[must_use]
pub fn get_agent_dir_with(env: &EnvLookup) -> PathBuf {
    if let Some(env_dir) = env(ENV_AGENT_DIR) {
        return PathBuf::from(expand_tilde(&env_dir, &home_dir()));
    }
    Path::new(&home_dir()).join(CONFIG_DIR_NAME).join("agent")
}

/// The user's custom themes directory, upstream's `getCustomThemesDir()`.
#[must_use]
pub fn get_custom_themes_dir() -> PathBuf {
    get_agent_dir().join("themes")
}

/// The custom-provider catalog path, upstream's `getModelsPath()`.
#[must_use]
pub fn get_models_path() -> PathBuf {
    get_agent_dir().join("models.json")
}

/// The provider credentials path, upstream's `getAuthPath()`.
#[must_use]
pub fn get_auth_path() -> PathBuf {
    get_agent_dir().join("auth.json")
}

/// The settings path, upstream's `getSettingsPath()`.
#[must_use]
pub fn get_settings_path() -> PathBuf {
    get_agent_dir().join("settings.json")
}

/// The skills/tools directory, upstream's `getToolsDir()`.
#[must_use]
pub fn get_tools_dir() -> PathBuf {
    get_agent_dir().join("tools")
}

/// The managed-binaries directory (`fd`, `rg`), upstream's `getBinDir()`.
#[must_use]
pub fn get_bin_dir() -> PathBuf {
    get_agent_dir().join("bin")
}

/// The prompt-templates directory, upstream's `getPromptsDir()`.
#[must_use]
pub fn get_prompts_dir() -> PathBuf {
    get_agent_dir().join("prompts")
}

/// The sessions root, upstream's `getSessionsDir()`; per-cwd session
/// directories encode below it via [`default_session_dir_path`].
#[must_use]
pub fn get_sessions_dir() -> PathBuf {
    get_agent_dir().join("sessions")
}

/// The debug log file, upstream's `getDebugLogPath()`:
/// `<APP_NAME>-debug.log` under the agent dir — `pi-debug.log`.
#[must_use]
pub fn get_debug_log_path() -> PathBuf {
    get_agent_dir().join(format!("{APP_NAME}-debug.log"))
}

/// The README path shipped next to the binary, upstream's
/// `getReadmePath()`.
#[must_use]
pub fn get_readme_path() -> String {
    get_package_dir()
        .join("README.md")
        .to_string_lossy()
        .into_owned()
}

/// The docs directory shipped next to the binary, upstream's
/// `getDocsPath()`.
#[must_use]
pub fn get_docs_path() -> String {
    get_package_dir()
        .join("docs")
        .to_string_lossy()
        .into_owned()
}

/// The examples directory shipped next to the binary, upstream's
/// `getExamplesPath()`.
#[must_use]
pub fn get_examples_path() -> String {
    get_package_dir()
        .join("examples")
        .to_string_lossy()
        .into_owned()
}

/// The share viewer base URL override, upstream's `PI_SHARE_VIEWER_URL`.
pub const ENV_SHARE_VIEWER_URL: &str = "PI_SHARE_VIEWER_URL";

/// The share viewer URL for a gist id, upstream's `getShareViewerUrl`:
/// the override base or the default, suffixed with `#<gist id>`.
#[must_use]
pub fn get_share_viewer_url(gist_id: &str) -> String {
    static DEFAULT_SHARE_VIEWER_URL: &str = "https://pi.dev/session/";
    let base = std::env::var(ENV_SHARE_VIEWER_URL)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_SHARE_VIEWER_URL.to_owned());
    format!("{base}#{gist_id}")
}

// =============================================================================
// Package dir
// =============================================================================

/// The package-dir override variable, upstream's `PI_PACKAGE_DIR` literal
/// (upstream does not derive this one from the app name).
const PACKAGE_DIR_ENV: &str = "PI_PACKAGE_DIR";

/// The package directory, upstream's `getPackageDir()`:
/// `PI_PACKAGE_DIR` when set (tilde-expanded), else the directory holding
/// the executable.
///
/// Upstream's two fallback branches — the Node walk-up (`findNodePackageDir`
/// with its bun/dist metadata skip) and the Bun binary's `dirname(execPath)`
/// — have no counterpart: a Rust binary ships its assets next to the
/// executable, which is the layout the Bun branch produced. Degrades to the
/// empty path when the platform cannot report the executable's directory.
#[must_use]
pub fn get_package_dir() -> PathBuf {
    get_package_dir_with(&default_env_lookup())
}

/// [`get_package_dir`] over an injected environment lookup, the test seam
/// for the override behavior.
#[must_use]
pub fn get_package_dir_with(env: &EnvLookup) -> PathBuf {
    if let Some(env_dir) = env(PACKAGE_DIR_ENV) {
        return PathBuf::from(expand_tilde(&env_dir, &home_dir()));
    }
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
        .unwrap_or_default()
}

/// The changelog file, upstream's `getChangelogPath()`.
#[must_use]
pub fn get_changelog_path() -> PathBuf {
    get_package_dir().join("CHANGELOG.md")
}

// =============================================================================
// Self-update surface, redesigned for the single native binary
// =============================================================================

/// The install method, upstream's `InstallMethod`.
///
/// Upstream detects npm/pnpm/yarn/bun installs from path shapes and builds
/// their package-manager commands; the Rust binary is a single native
/// executable whose only managed install method is a cargo install, so the
/// method set collapses to [`InstallMethod::Cargo`] and
/// [`InstallMethod::Unknown`]. The installer-managed layout is not a method
/// here — the package-manager CLI detects it separately through
/// `PI_MANAGED_INSTALL_ROOT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallMethod {
    /// Installed by `cargo install` into cargo's bin directory.
    Cargo,
    /// A wrapper, source checkout, or unrecognized layout: no
    /// self-update command exists.
    Unknown,
}

impl InstallMethod {
    /// The lowercase name upstream's instruction strings interpolate, the
    /// `method` branch of `getSelfUpdateUnavailableInstruction`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cargo => "cargo",
            Self::Unknown => "unknown",
        }
    }
}

/// One self-update command step, upstream's `SelfUpdateCommandStep`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfUpdateCommandStep {
    /// The executable, upstream's `command`.
    pub command: String,
    /// The arguments, upstream's `args`.
    pub args: Vec<String>,
    /// The printable rendering, upstream's `display`: whitespace-bearing
    /// arguments double-quoted, the rest joined with single spaces.
    pub display: String,
}

/// The self-update command, upstream's `SelfUpdateCommand`.
///
/// A renamed package updates in two steps (uninstall the old name, install
/// the new one); the top-level fields carry the install step so a caller
/// that only renders `display` sees the composed command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfUpdateCommand {
    /// The primary (install) step's executable, upstream's `command`.
    pub command: String,
    /// The primary (install) step's arguments, upstream's `args`.
    pub args: Vec<String>,
    /// The composed display, upstream's `display` — the uninstall step's
    /// display, ` && `, then the install step's, when a rename is involved.
    pub display: String,
    /// The two steps in order, upstream's `steps?`; `None` when no rename
    /// is involved.
    pub steps: Option<Vec<SelfUpdateCommandStep>>,
}

/// The update target, upstream's `SelfUpdatePackageTarget`.
///
/// Upstream's string-or-object union restates as the struct both forms
/// normalize to: a bare package name is its own install spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfUpdatePackageTarget {
    /// The package being installed, upstream's `packageName`.
    pub package_name: String,
    /// The spec to install, upstream's `installSpec` — a bare name when
    /// absent, else `name@version`.
    pub install_spec: String,
}

impl SelfUpdatePackageTarget {
    /// The bare-name form, upstream's string target.
    #[must_use]
    pub fn from_package_name(package_name: &str) -> Self {
        Self {
            package_name: package_name.to_string(),
            install_spec: package_name.to_string(),
        }
    }

    /// The named-fields form, upstream's object target with its
    /// `installSpec ?? packageName` fallback.
    #[must_use]
    pub fn new(package_name: &str, install_spec: Option<&str>) -> Self {
        Self {
            package_name: package_name.to_string(),
            install_spec: install_spec.unwrap_or(package_name).to_string(),
        }
    }
}

fn make_self_update_command_step(command: &str, args: &[String]) -> SelfUpdateCommandStep {
    SelfUpdateCommandStep {
        command: command.to_string(),
        args: args.to_vec(),
        display: args
            .iter()
            .map(|arg| {
                if arg.chars().any(char::is_whitespace) {
                    format!("\"{arg}\"")
                } else {
                    arg.clone()
                }
            })
            .fold(command.to_string(), |joined, arg| format!("{joined} {arg}")),
    }
}

fn make_self_update_command(
    install_step: SelfUpdateCommandStep,
    uninstall_step: Option<SelfUpdateCommandStep>,
) -> SelfUpdateCommand {
    match uninstall_step {
        None => SelfUpdateCommand {
            command: install_step.command,
            args: install_step.args,
            display: install_step.display,
            steps: None,
        },
        Some(uninstall_step) => {
            let steps = vec![uninstall_step, install_step];
            let install_step = &steps[1];
            SelfUpdateCommand {
                command: install_step.command.clone(),
                args: install_step.args.clone(),
                display: format!("{} && {}", steps[0].display, install_step.display),
                steps: Some(steps),
            }
        }
    }
}

/// The installed cargo crate name, the `installedPackageName` input.
///
/// A cargo-installed binary carries no crate metadata, so the package name
/// rides the caller — [`PACKAGE_NAME`] for the running build.
fn self_update_command_for_method(
    method: InstallMethod,
    installed_package_name: &str,
    update_package_target: &SelfUpdatePackageTarget,
) -> Option<SelfUpdateCommand> {
    match method {
        InstallMethod::Unknown => None,
        InstallMethod::Cargo => {
            // The `crate:` channel's compile flag; the spec's `@version`
            // suffix pins the build when the target carries one — crate
            // names cannot contain `@`, so the suffix is always a version.
            let mut args = vec!["install".to_string(), "--locked".to_string()];
            match update_package_target
                .install_spec
                .rsplit_once('@')
                .filter(|(_, version)| semver::Version::parse(version).is_ok())
            {
                Some((name, version)) => {
                    args.push(name.to_string());
                    args.push("--version".to_string());
                    args.push(version.to_string());
                }
                None => args.push(update_package_target.install_spec.clone()),
            }
            let install_step = make_self_update_command_step("cargo", &args);
            let uninstall_step = (update_package_target.package_name != installed_package_name)
                .then(|| {
                    make_self_update_command_step(
                        "cargo",
                        &["uninstall".to_string(), installed_package_name.to_string()],
                    )
                });
            Some(make_self_update_command(install_step, uninstall_step))
        }
    }
}

/// The executable path, upstream's `process.execPath`/`process.argv[1]`
/// pair collapsed to the one input a native binary has.
pub(crate) fn current_exe_path() -> PathBuf {
    std::env::current_exe().unwrap_or_default()
}

/// The cargo bin directory, the managed install method's probe target:
/// `$CARGO_HOME/bin`, else `~/.cargo/bin`.
fn cargo_bin_dir_with(env: &EnvLookup) -> PathBuf {
    match env("CARGO_HOME") {
        Some(cargo_home) if !cargo_home.is_empty() => Path::new(&cargo_home).join("bin"),
        // Cargo itself resolves its home from $HOME when CARGO_HOME is
        // unset, so the probe follows the same ladder.
        _ => {
            let home = env("HOME")
                .filter(|home| !home.is_empty())
                .unwrap_or_else(home_dir);
            Path::new(&home).join(".cargo").join("bin")
        }
    }
}

/// Detect how the running binary was installed, upstream's
/// `detectInstallMethod`.
///
/// The single native binary recognizes one managed layout: the executable
/// living under cargo's bin directory. Everything else is
/// [`InstallMethod::Unknown`] — wrappers and source checkouts update
/// through whatever provides them.
#[must_use]
pub fn detect_install_method() -> InstallMethod {
    detect_install_method_with(&current_exe_path(), &default_env_lookup())
}

/// [`detect_install_method`] over an injected executable path and
/// environment, the test seam for the layout probes.
#[must_use]
pub fn detect_install_method_with(exe_path: &Path, env: &EnvLookup) -> InstallMethod {
    let bin_dir = cargo_bin_dir_with(env);
    if exe_path.starts_with(&bin_dir) {
        return InstallMethod::Cargo;
    }
    InstallMethod::Unknown
}

/// Whether the install path accepts writes, upstream's
/// `isSelfUpdatePathWritable`: `W_OK` on the package directory and its
/// parent.
fn is_self_update_path_writable(package_dir: &Path) -> bool {
    #[cfg(unix)]
    {
        use nix::unistd::AccessFlags;
        let parent = package_dir.parent().unwrap_or(package_dir);
        nix::unistd::access(package_dir, AccessFlags::W_OK).is_ok()
            && nix::unistd::access(parent, AccessFlags::W_OK).is_ok()
    }
    #[cfg(not(unix))]
    {
        !package_dir
            .metadata()
            .map(|meta| meta.permissions().readonly())
            .unwrap_or(true)
    }
}

/// Whether the executable sits inside the install method's managed root,
/// upstream's `isManagedByGlobalPackageManager` reduced to the cargo
/// branch: the exe directory must be cargo's bin directory itself.
fn is_managed_by_global_package_manager_with(exe_path: &Path, env: &EnvLookup) -> bool {
    matches!(
        detect_install_method_with(exe_path, env),
        InstallMethod::Cargo
    )
}

/// The self-update command for this installation, upstream's
/// `getSelfUpdateCommand`.
///
/// The method's command when the installation is managed by its global
/// package manager and the install path is writable, else `None`.
#[must_use]
pub fn get_self_update_command(
    installed_package_name: &str,
    update_package_target: &SelfUpdatePackageTarget,
) -> Option<SelfUpdateCommand> {
    get_self_update_command_with(
        installed_package_name,
        update_package_target,
        &current_exe_path(),
        &default_env_lookup(),
    )
}

/// [`get_self_update_command`] over an injected executable path and
/// environment, the test seam.
#[must_use]
pub fn get_self_update_command_with(
    installed_package_name: &str,
    update_package_target: &SelfUpdatePackageTarget,
    exe_path: &Path,
    env: &EnvLookup,
) -> Option<SelfUpdateCommand> {
    let method = detect_install_method_with(exe_path, env);
    let command =
        self_update_command_for_method(method, installed_package_name, update_package_target)?;
    if !is_managed_by_global_package_manager_with(exe_path, env) {
        return None;
    }
    let package_dir = get_package_dir_with(env);
    if !is_self_update_path_writable(&package_dir) {
        return None;
    }
    Some(command)
}

/// The instruction shown when self-update cannot run, upstream's
/// `getSelfUpdateUnavailableInstruction`.
#[must_use]
pub fn get_self_update_unavailable_instruction(
    installed_package_name: &str,
    update_package_target: &SelfUpdatePackageTarget,
) -> String {
    get_self_update_unavailable_instruction_with(
        installed_package_name,
        update_package_target,
        &current_exe_path(),
        &default_env_lookup(),
    )
}

/// [`get_self_update_unavailable_instruction`] over an injected executable
/// path and environment, the test seam.
#[must_use]
pub fn get_self_update_unavailable_instruction_with(
    installed_package_name: &str,
    update_package_target: &SelfUpdatePackageTarget,
    exe_path: &Path,
    env: &EnvLookup,
) -> String {
    let method = detect_install_method_with(exe_path, env);
    let command =
        self_update_command_for_method(method, installed_package_name, update_package_target);
    match command {
        Some(command) => {
            if is_managed_by_global_package_manager_with(exe_path, env)
                && !is_self_update_path_writable(&get_package_dir_with(env))
            {
                format!(
                    "This installation is managed by a global {} install, but the install path is not writable. Update it yourself with: {}",
                    method.name(),
                    command.display
                )
            } else {
                format!(
                    "This installation is not managed by a global {} install. Update it with the package manager, wrapper, or source checkout that provides it.",
                    method.name()
                )
            }
        }
        None => format!(
            "Update {} using the package manager, wrapper, or source checkout that provides this installation.",
            update_package_target.install_spec
        ),
    }
}

/// The instruction for updating the given package, upstream's
/// `getUpdateInstruction`.
#[must_use]
pub fn get_update_instruction(package_name: &str) -> String {
    get_update_instruction_with(package_name, &current_exe_path(), &default_env_lookup())
}

/// [`get_update_instruction`] over an injected executable path and
/// environment, the test seam.
#[must_use]
pub fn get_update_instruction_with(package_name: &str, exe_path: &Path, env: &EnvLookup) -> String {
    let method = detect_install_method_with(exe_path, env);
    match self_update_command_for_method(
        method,
        package_name,
        &SelfUpdatePackageTarget::from_package_name(package_name),
    ) {
        Some(command) => format!("Run: {}", command.display),
        None => get_self_update_unavailable_instruction_with(
            package_name,
            &SelfUpdatePackageTarget::from_package_name(package_name),
            exe_path,
            env,
        ),
    }
}

// =============================================================================
// Session directory encoding, upstream's src/core/session-manager.ts
// =============================================================================

/// Encode a resolved absolute cwd into the session-directory name.
///
/// This is the body of upstream's `getDefaultSessionDirPath`: one leading
/// `/` or `\` is stripped first (the strip-then-replace order matters — a
/// leading slash disappears instead of becoming a dash), then every
/// remaining `/`, `\`, and `:` becomes `-`, and the whole thing wraps in
/// `--…--`.
#[must_use]
pub fn encode_session_cwd(resolved_cwd: &str) -> String {
    let stripped = resolved_cwd
        .strip_prefix(['/', '\\'])
        .unwrap_or(resolved_cwd);
    let mut encoded = String::with_capacity(stripped.len() + 4);
    encoded.push_str("--");
    for ch in stripped.chars() {
        match ch {
            '/' | '\\' | ':' => encoded.push('-'),
            other => encoded.push(other),
        }
    }
    encoded.push_str("--");
    encoded
}

/// The default session directory for a cwd, upstream's
/// `getDefaultSessionDirPath`.
///
/// Both inputs run through the resolver (tilde expansion, then the process
/// cwd as the relative base — see [`crate::utils::paths::resolve_path`]),
/// the cwd encodes via [`encode_session_cwd`], and the result joins
/// `<resolved agent dir>/sessions/<encoded>`. The mkdir-ing wrapper upstream
/// also exports (`getDefaultSessionDir`) rides the session-manager port.
#[must_use]
pub fn default_session_dir_path(cwd: &str, agent_dir: &str) -> PathBuf {
    let base = process_cwd();
    let home = home_dir();
    let resolved_cwd = resolve_path(cwd, &base, &home);
    let resolved_agent_dir = resolve_path(agent_dir, &base, &home);
    Path::new(&resolved_agent_dir)
        .join("sessions")
        .join(encode_session_cwd(&resolved_cwd))
}
