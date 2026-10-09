//! Shell resolution and binary-output sanitation, upstream's
//! `src/utils/shell.ts`.
//!
//! The win32 branches — the Git-for-Windows discovery, the `where` probe,
//! and the `taskkill` kill — ride the map's Windows exclusion;
//! `normalize_windows_shell_path`'s pure string work lives in
//! [`super::paths`]. The PATH-entry lookup upstream runs against
//! `process.env` takes the crate's `crate::config::EnvLookup`
//! seam, and the environment overlay [`get_shell_env_path_update`] returns
//! is the `{ [pathKey]: updatedPath }` slice of upstream's full
//! `{ ...process.env }` object, which Rust callers apply over their own
//! environment base.

use std::path::Path;
use std::sync::Mutex;

use crate::config::{EnvLookup, get_bin_dir};

/// The shell a command runs through, upstream's `ShellConfig` with its
/// transport mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellConfig {
    /// The shell executable.
    pub shell: String,
    /// The arguments the command rides behind, `["-c"]` or the stdin form.
    pub args: Vec<String>,
    /// How the command reaches the shell: argv (upstream's `"argv"`) or
    /// stdin (the legacy WSL bash), absent when upstream's field is absent.
    pub command_transport: Option<CommandTransport>,
}

/// How the command reaches the shell, upstream's `"argv"`/`"stdin"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandTransport {
    /// The command rides as the final argv entry.
    Argv,
    /// The command rides on stdin, the legacy WSL bash shape.
    Stdin,
}

/// The shell-resolution failure, upstream's thrown `Error` messages
/// verbatim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShellError(pub String);

impl std::fmt::Display for ShellError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ShellError {}

/// Whether a shell path names the legacy WSL bash, whose argv mode strips
/// commands — it runs commands from stdin instead, upstream's
/// `isLegacyWslBashPath`.
#[must_use]
pub fn is_legacy_wsl_bash_path(path: &str) -> bool {
    let normalized = path.replace('/', "\\").to_lowercase();
    let Some((head, tail)) = normalized.split_once(":\\") else {
        return false;
    };
    if head.len() != 1 || !head.as_bytes()[0].is_ascii_lowercase() {
        return false;
    }
    tail == "windows\\system32\\bash.exe" || tail == "windows\\sysnative\\bash.exe"
}

fn get_bash_shell_config(shell: &str) -> ShellConfig {
    if is_legacy_wsl_bash_path(shell) {
        ShellConfig {
            shell: shell.to_string(),
            args: vec!["-s".to_string()],
            command_transport: Some(CommandTransport::Stdin),
        }
    } else {
        ShellConfig {
            shell: shell.to_string(),
            args: vec!["-c".to_string()],
            command_transport: None,
        }
    }
}

/// Find an executable on PATH through the `which` probe, upstream's
/// `findExecutableOnPath` unix branch — the probe's own word is trusted
/// because it handles Termux and special filesystems the PATH walk misses.
/// The win32 `where` branch rides the map's Windows exclusion.
fn find_executable_on_path(executable: &str) -> Option<String> {
    let result = super::child_process::spawn_process_sync(
        "which",
        &[executable],
        &super::child_process::SpawnSyncOptions {
            capture_output: true,
            timeout_ms: Some(5000),
        },
    );
    if result.status == Some(0) {
        let first_match = result
            .stdout
            .trim()
            .split(['\r', '\n'])
            .next()
            .unwrap_or_default()
            .to_string();
        if !first_match.is_empty() {
            return Some(first_match);
        }
    }
    None
}

/// Resolve shell configuration based on platform and an optional explicit
/// shell path, upstream's `getShellConfig`.
///
/// Resolution order: a user-specified path (an error when it does not
/// exist), then `/bin/bash`, then bash on PATH, then the `sh` fallback.
///
/// # Errors
/// [`ShellError`] when a custom path does not exist, upstream's throw.
pub fn get_shell_config(custom_shell_path: Option<&str>) -> Result<ShellConfig, ShellError> {
    if let Some(custom_shell_path) = custom_shell_path {
        if Path::new(custom_shell_path).exists() {
            return Ok(get_bash_shell_config(custom_shell_path));
        }
        return Err(ShellError(format!(
            "Custom shell path not found: {custom_shell_path}"
        )));
    }

    // Unix: try /bin/bash, then bash on PATH, then fallback to sh
    if Path::new("/bin/bash").exists() {
        return Ok(get_bash_shell_config("/bin/bash"));
    }
    if let Some(bash_on_path) = find_executable_on_path("bash") {
        return Ok(get_bash_shell_config(&bash_on_path));
    }
    Ok(ShellConfig {
        shell: "sh".to_string(),
        args: vec!["-c".to_string()],
        command_transport: None,
    })
}

/// The PowerShell argv preamble, upstream's `POWERSHELL_ARGS`.
pub const POWERSHELL_ARGS: [&str; 5] = [
    "-NoProfile",
    "-NonInteractive",
    "-ExecutionPolicy",
    "Bypass",
    "-Command",
];

/// Resolve PowerShell on Windows, preferring PowerShell 7 when available.
///
/// # Errors
/// [`ShellError`] with the guard message everywhere this effort targets —
/// the pwsh discovery is a win32 branch and Windows is out of scope, so the
/// only reachable contract is upstream's non-Windows throw.
pub fn get_power_shell_config() -> Result<ShellConfig, ShellError> {
    Err(ShellError(
        "The powershell tool is only available on Windows.".to_string(),
    ))
}

/// The PATH overlay the shell commands apply, upstream's `getShellEnv`
/// reduced to the one entry it changes: the case-insensitively located PATH
/// key, with the managed-bin dir prepended when absent.
#[must_use]
pub fn get_shell_env_path_update(env: &EnvLookup) -> (String, String) {
    let bin_dir = get_bin_dir().to_string_lossy().into_owned();
    let path_key = find_path_key(env).unwrap_or_else(|| "PATH".to_string());
    let current_path = env(&path_key).unwrap_or_default();
    let path_entries: Vec<&str> = current_path
        .split(DELIMITER)
        .filter(|entry| !entry.is_empty())
        .collect();
    let has_bin_dir = path_entries.iter().any(|entry| *entry == bin_dir);
    let updated_path = if has_bin_dir {
        current_path
    } else {
        [vec![bin_dir.as_str()], path_entries]
            .concat()
            .join(DELIMITER)
    };
    (path_key, updated_path)
}

/// The full spawn environment the shell tools run, upstream's
/// `getShellEnv`.
///
/// The process environment with the managed-bin dir prepended to the PATH
/// entry. Non-UTF-8 entries drop, node's string map has no counterpart for
/// them.
#[must_use]
pub fn get_shell_env() -> std::collections::BTreeMap<String, String> {
    let mut map: std::collections::BTreeMap<String, String> = std::env::vars_os()
        .map(|(key, value)| {
            (
                key.to_string_lossy().into_owned(),
                value.to_string_lossy().into_owned(),
            )
        })
        .collect();
    let lookup: EnvLookup = {
        let snapshot = map.clone();
        Box::new(move |key| snapshot.get(key).cloned())
    };
    let (path_key, updated_path) = get_shell_env_path_update(&lookup);
    map.insert(path_key, updated_path);
    map
}

fn find_path_key(env: &EnvLookup) -> Option<String> {
    // The known spellings of the PATH variable, the case-insensitive scan
    // upstream runs over `Object.keys(process.env)`; the injected seam is
    // keyed by exact name, so the scan tries the spellings.
    ["PATH", "Path", "path"]
        .into_iter()
        .find(|key| env(key).is_some())
        .map(str::to_string)
}

/// The PATH separator, node's `path.delimiter` on the unix targets.
const DELIMITER: &str = ":";

/// Sanitize binary output for display/storage, upstream's
/// `sanitizeBinaryOutput`.
///
/// Removes the characters that crash string-width or garble display:
/// control characters except tab, newline, and carriage return, and the
/// Unicode Format run that trips a string-width bug. Rust's `char` cannot
/// hold the lone surrogates upstream's `Array.from` filter also drops.
#[must_use]
pub fn sanitize_binary_output(value: &str) -> String {
    value
        .chars()
        .filter(|ch| {
            let code = u32::from(*ch);
            // Allow tab, newline, carriage return
            if code == 0x09 || code == 0x0a || code == 0x0d {
                return true;
            }
            // Filter out control characters (0x00-0x1F, except the three above)
            if code <= 0x1f {
                return false;
            }
            // Filter out the Unicode Format run that crashes string-width
            if (0xfff9..=0xfffb).contains(&code) {
                return false;
            }
            true
        })
        .collect()
}

/// The detached child processes the shutdown kill sweeps, upstream's
/// `trackedDetachedChildPids`.
static TRACKED_DETACHED_CHILD_PIDS: Mutex<Vec<u32>> = Mutex::new(Vec::new());

fn tracked_pids() -> std::sync::MutexGuard<'static, Vec<u32>> {
    // A poisoned lock means a previous access panicked mid-mutation; the
    // belt reads whatever list survived rather than crash the shutdown
    // sweep.
    TRACKED_DETACHED_CHILD_PIDS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Track a detached child so shutdown kills it, upstream's
/// `trackDetachedChildPid`.
pub fn track_detached_child_pid(pid: u32) {
    tracked_pids().push(pid);
}

/// Stop tracking a detached child, upstream's `untrackDetachedChildPid`.
pub fn untrack_detached_child_pid(pid: u32) {
    tracked_pids().retain(|tracked| *tracked != pid);
}

/// Kill every tracked detached child and clear the set, upstream's
/// `killTrackedDetachedChildren`.
pub fn kill_tracked_detached_children() {
    let pids = tracked_pids().drain(..).collect::<Vec<u32>>();
    for pid in pids {
        kill_process_tree(pid);
    }
}

/// Kill a process and all its children, upstream's `killProcessTree`.
///
/// The children spawn detached in their own process group, so a
/// negative-pid `SIGKILL` reaches the whole tree; a failed group kill falls
/// back to the child alone. The win32 `taskkill` branch rides the map's
/// Windows exclusion.
#[expect(
    clippy::cast_possible_wrap,
    reason = "the kernel's pid_t is i32; a process id beyond it cannot exist on a supported platform"
)]
pub fn kill_process_tree(pid: u32) {
    #[cfg(unix)]
    {
        use nix::sys::signal::{Signal, kill};
        use nix::unistd::Pid;
        let group = Pid::from_raw(-(pid as i32));
        if kill(group, Signal::SIGKILL).is_err() {
            let _ = kill(Pid::from_raw(pid as i32), Signal::SIGKILL);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
    }
}
