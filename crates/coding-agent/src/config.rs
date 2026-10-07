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
//! self-update machinery this module also carries upstream (the
//! npm/pnpm/yarn/bun command builders and their install-method probes)
//! rides its own ticket.
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
