//! The path-normalization belt, upstream's `src/utils/paths.ts`.
//!
//! Two restatements pin the byte-for-byte contract. The default-options
//! resolver the config foundation rides ([`resolve_path`], [`expand_tilde`])
//! carries the #118 tilde slice: Node's POSIX `path.resolve`, which differs
//! from Rust's `std::path::absolute` in three observable ways the vectors
//! pin — `..` segments collapse lexically instead of surviving, trailing
//! slashes drop, and a leading `//` collapses to `/` (upstream runs
//! `path.resolve`, whose output feeds the session-dir encoding).
//! [`normalize_path`] and [`resolve_path_with`] port the full upstream
//! surface, whose `file://` branch converts through node's `fileURLToPath`
//! and fails the conversion for an invalid URL the way node throws.
//!
//! The win32 branch of `normalizePath` (the shell-path rewrite trigger) and
//! the win32 `~\` tilde form ride the map's Windows exclusion —
//! [`normalize_windows_shell_path`] itself ports as the pure function the
//! upstream suite exercises on every platform.

use regex::Regex;

use crate::config::{home_dir, process_cwd};

use super::child_process::spawn_process_sync;

/// The optional path-input normalizations upstream's `PathInputOptions`
/// carries. Defaults mirror upstream's: only tilde expansion is on.
///
/// The four independent toggles are upstream's own option shape; the
/// state-machine collapse the lint suggests would not mirror it.
#[expect(
    clippy::struct_excessive_bools,
    reason = "the option shape mirrors upstream's PathInputOptions interface"
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathInputOptions {
    /// Trim leading/trailing whitespace before normalization.
    pub trim: bool,
    /// Expand a leading `~` to a home directory. Defaults to true.
    pub expand_tilde: bool,
    /// Home directory used for `~` expansion; the process home when absent.
    pub home_dir: Option<String>,
    /// Strip a leading `@`, used for CLI @file paths.
    pub strip_at_prefix: bool,
    /// Normalize unicode space variants to regular spaces.
    pub normalize_unicode_spaces: bool,
}

impl Default for PathInputOptions {
    fn default() -> Self {
        Self {
            trim: false,
            expand_tilde: true,
            home_dir: None,
            strip_at_prefix: false,
            normalize_unicode_spaces: false,
        }
    }
}

impl PathInputOptions {
    /// The default options with an explicit home, the form the config
    /// foundation's callers use.
    #[must_use]
    pub fn with_home(home: &str) -> Self {
        Self {
            home_dir: Some(home.to_string()),
            ..Self::default()
        }
    }

    fn options_home(&self) -> String {
        self.home_dir.clone().unwrap_or_else(home_dir)
    }
}

/// The failure the `file://` branch of [`normalize_path`] reports, the
/// port of node's `URIError` from `fileURLToPath`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathNormalizeError;

impl std::fmt::Display for PathNormalizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Invalid URL")
    }
}

impl std::error::Error for PathNormalizeError {}

/// Expand a leading tilde to `home`, the slice of upstream's `normalizePath`
/// its default options run.
///
/// Exactly `"~"` becomes `home`, a `"~/"` prefix joins the remainder onto
/// `home` (Node's `path.join`, which lexically normalizes — `"~/.."` lands
/// on `home`), and anything else passes through. `home` is the
/// `os.homedir()` replacement; the config callers pass the process home. The
/// `~\` Windows form of upstream's tilde check is not reproduced — Windows is
/// out of scope for this effort.
#[must_use]
pub fn expand_tilde(input: &str, home: &str) -> String {
    if input == "~" {
        return home.to_string();
    }
    if let Some(rest) = input.strip_prefix("~/") {
        return normalize_posix_joined(&format!("{home}/{rest}"));
    }
    input.to_string()
}

/// Resolve `input` against `base_dir`, the default-options resolver slice
/// the config foundation rides: the tilde slice of [`normalize_path`]
/// first, then Node's POSIX `path.resolve`.
///
/// The base is always absolute in this crate's calls (the process cwd),
/// which is what makes the lexical collapse of `..` segments exact: a `..`
/// climbing past the root drops, matching upstream's `normalizeString`
/// with `allowAboveRoot = false`.
#[must_use]
pub fn resolve_path(input: &str, base_dir: &str, home: &str) -> String {
    let normalized = expand_tilde(input, home);
    if normalized.is_empty() {
        return normalize_posix_joined(base_dir);
    }
    if normalized.starts_with('/') {
        return normalize_posix_joined(&normalized);
    }
    normalize_posix_joined(&format!("{base_dir}/{normalized}"))
}

/// Resolve `input` against `base_dir` over explicit options, upstream's
/// `resolvePath` with its `PathInputOptions` parameter set.
///
/// The base normalizes with the default options, upstream's
/// `normalizePath(baseDir)`.
///
/// # Errors
/// [`PathNormalizeError`] when `input` (or a tilde-expanded `base_dir`) is a
/// `file://` URL that does not convert to a local path — node's
/// `fileURLToPath` throw.
pub fn resolve_path_with(
    input: &str,
    base_dir: &str,
    options: &PathInputOptions,
) -> Result<String, PathNormalizeError> {
    let normalized = normalize_path(input, options)?;
    let normalized_base_dir = normalize_path(base_dir, &PathInputOptions::default())?;
    Ok(if normalized.starts_with('/') {
        normalize_posix_joined(&normalized)
    } else {
        normalize_posix_joined(&format!("{normalized_base_dir}/{normalized}"))
    })
}

/// Normalize a path input, upstream's `normalizePath`.
///
/// Whitespace trims first when asked, unicode space variants collapse, a
/// leading `@` strips, the win32 shell-path rewrite rides the map's
/// Windows exclusion, tilde expansion defaults on, and a `file://` URL
/// converts through node's `fileURLToPath`.
///
/// # Errors
/// [`PathNormalizeError`] when a `file://` input does not convert to a
/// local path — a non-empty host, a malformed percent sequence, or a path
/// that is not valid UTF-8.
pub fn normalize_path(
    input: &str,
    options: &PathInputOptions,
) -> Result<String, PathNormalizeError> {
    let mut normalized = if options.trim {
        input.trim().to_string()
    } else {
        input.to_string()
    };
    if options.normalize_unicode_spaces {
        normalized = normalize_unicode_spaces(&normalized);
    }
    if options.strip_at_prefix && normalized.starts_with('@') {
        normalized.remove(0);
    }

    if options.expand_tilde {
        let home = options.options_home();
        if normalized == "~" {
            return Ok(home);
        }
        if let Some(rest) = normalized.strip_prefix("~/") {
            return Ok(normalize_posix_joined(&format!("{home}/{rest}")));
        }
    }

    if normalized.starts_with("file://") {
        return file_url_to_path(&normalized).ok_or(PathNormalizeError);
    }

    Ok(normalized)
}

/// node's `fileURLToPath` for the POSIX shapes the belt meets: the host
/// must be empty (`file://`/`file:///`), the pathname must start with a
/// slash, and it must percent-decode to valid UTF-8 — every failure is the
/// `URIError` node throws. The strict decode rides the hosted port's
/// `decodeURIComponent` re-expression.
fn file_url_to_path(url: &str) -> Option<String> {
    let parsed = url::Url::parse(url).ok()?;
    if parsed.scheme() != "file" {
        return None;
    }
    if parsed.host_str().is_some_and(|host| !host.is_empty()) {
        return None;
    }
    let path = parsed.path();
    if !path.starts_with('/') {
        return None;
    }
    super::git::hosted::decode_uri_component(path)
}

/// The unicode space variants upstream's `UNICODE_SPACES` regex rewrites:
/// NBSP, the U+2000 block, NNBSP, MMLSP, and ideographic space.
fn normalize_unicode_spaces(value: &str) -> String {
    #[expect(
        clippy::expect_used,
        reason = "the grammar is a compile-time constant; a failure is a programming error, not a runtime condition"
    )]
    let pattern =
        Regex::new(r"[\u{00A0}\u{2000}-\u{200A}\u{202F}\u{205F}\u{3000}]").expect("static pattern");
    pattern.replace_all(value, " ").into_owned()
}

/// Resolve a path to its canonical (real) form, following symlinks.
///
/// Falls back to the raw path if resolution fails (e.g. the target does
/// not exist yet), so that callers never crash on missing filesystem
/// entries.
#[must_use]
pub fn canonicalize_path(path: &str) -> String {
    std::fs::canonicalize(path).map_or_else(
        |_| path.to_string(),
        |canonical| canonical.to_string_lossy().into_owned(),
    )
}

/// The filesystem revision stamp upstream's `getFileRevision` builds:
/// `dev:ino:size:mtimeNs:ctimeNs`, absent when the file cannot be statted.
#[must_use]
#[expect(
    clippy::cast_sign_loss,
    reason = "node's bigint stats carry unsigned ns fields; the kernel's i32 nsec fields are non-negative for real timestamps"
)]
pub fn get_file_revision(path: &str) -> Option<String> {
    use std::os::unix::fs::MetadataExt;

    let metadata = std::fs::metadata(path).ok()?;
    Some(format!(
        "{}:{}:{}:{}:{}",
        metadata.dev(),
        metadata.ino(),
        metadata.size(),
        metadata.mtime_nsec() as u64,
        metadata.ctime_nsec() as u64
    ))
}

/// Returns true if the value is NOT a package source (npm:, git:, etc.)
/// or a remote URL protocol. Bare names, relative paths, and file: URLs
/// are considered local.
#[must_use]
pub fn is_local_path(value: &str) -> bool {
    let trimmed = value.trim();
    // Known non-local prefixes. file: URLs are local paths and are intentionally resolved by resolve_path().
    !trimmed.starts_with("npm:")
        && !trimmed.starts_with("git:")
        && !trimmed.starts_with("github:")
        && !trimmed.starts_with("http:")
        && !trimmed.starts_with("https:")
        && !trimmed.starts_with("ssh:")
}

/// Convert Git Bash, MSYS, Cygwin, and WSL drive paths to a form native
/// Windows APIs accept, upstream's `normalizeWindowsShellPath` — the pure
/// string rewrite its suite exercises on every platform.
///
/// # Panics
/// Never: the grammar is a compile-time constant, and its capture groups
/// index what the same constant defines.
#[must_use]
pub fn normalize_windows_shell_path(file_path: &str) -> String {
    if !file_path.starts_with('/') || file_path.starts_with("//") || file_path.contains('\\') {
        return file_path.to_string();
    }
    #[expect(
        clippy::expect_used,
        reason = "the grammar is a compile-time constant; a failure is a programming error, not a runtime condition"
    )]
    let drive_shell_path =
        Regex::new(r"(?i)^/(?:mnt/|cygdrive/)?([a-z])(?:/(.*))?$").expect("static pattern");
    let Some(captures) = drive_shell_path.captures(file_path) else {
        return file_path.to_string();
    };
    let drive = captures
        .get(1)
        .map_or(String::new(), |drive| drive.as_str().to_ascii_uppercase());
    let suffix = captures
        .get(2)
        .map(|suffix| suffix.as_str().replace('/', "\\"))
        .unwrap_or_default();
    format!("{drive}:\\{suffix}")
}

/// The path relative to `cwd` when it lives inside it, upstream's
/// `getCwdRelativePath` — `None` marks a parent-directory traversal (and,
/// unlike upstream's throw, a `file://` normalization failure).
///
/// Both inputs resolve through the default-options resolver with the
/// process cwd as the relative base, upstream's `resolvePath(cwd)` /
/// `resolvePath(filePath, resolvedCwd)` pair.
#[must_use]
pub fn get_cwd_relative_path(file_path: &str, cwd: &str) -> Option<String> {
    let base = process_cwd();
    let home = home_dir();
    let resolved_cwd = resolve_path_with(cwd, &base, &PathInputOptions::with_home(&home)).ok()?;
    let resolved_path = resolve_path_with(
        file_path,
        &resolved_cwd,
        &PathInputOptions::with_home(&home),
    )
    .ok()?;
    let relative_path = relative_posix(&resolved_cwd, &resolved_path);
    let is_inside_cwd = relative_path.is_empty()
        || (relative_path != ".."
            && !relative_path.starts_with("../")
            && !relative_path.starts_with('/'));

    if is_inside_cwd {
        Some(if relative_path.is_empty() {
            ".".to_string()
        } else {
            relative_path
        })
    } else {
        None
    }
}

/// The path rendered relative to `cwd` when it lives inside it, absolute
/// with `/` separators otherwise, upstream's
/// `formatPathRelativeToCwdOrAbsolute`.
///
/// The separator round-trip upstream runs (`split(sep).join("/")`) is the
/// identity on POSIX.
#[must_use]
pub fn format_path_relative_to_cwd_or_absolute(file_path: &str, cwd: &str) -> String {
    let home = home_dir();
    let Ok(absolute_path) = resolve_path_with(file_path, cwd, &PathInputOptions::with_home(&home))
    else {
        return file_path.to_string();
    };
    get_cwd_relative_path(&absolute_path, cwd).unwrap_or(absolute_path)
}

/// Tag a path so cloud-sync providers ignore it, upstream's
/// `markPathIgnoredByCloudSync`: xattr on darwin, setfattr on linux, and
/// nothing elsewhere.
///
/// The spawned helpers' output is ignored; failures stay silent, upstream's
/// `stdio: "ignore"` spawn.
pub fn mark_path_ignored_by_cloud_sync(path: &str) {
    use super::child_process::SpawnSyncOptions;

    #[cfg(target_os = "macos")]
    for attr in ["com.dropbox.ignored", "com.apple.fileprovider.ignore#P"] {
        let _outcome = spawn_process_sync(
            "xattr",
            &["-w", attr, "1", path],
            &SpawnSyncOptions::IGNORE_OUTPUT,
        );
    }
    #[cfg(target_os = "linux")]
    let _outcome = spawn_process_sync(
        "setfattr",
        &["-n", "user.com.dropbox.ignored", "-v", "1", path],
        &SpawnSyncOptions::IGNORE_OUTPUT,
    );
}

/// Node's POSIX `path.relative` over two resolved absolute paths: the
/// common segment prefix drops, the rest of `from` climbs as `..` chains,
/// and identical paths yield an empty string.
fn relative_posix(from: &str, to: &str) -> String {
    let from_segments: Vec<&str> = from.split('/').collect();
    let to_segments: Vec<&str> = to.split('/').collect();
    let mut shared = 0usize;
    while shared < from_segments.len()
        && shared < to_segments.len()
        && from_segments[shared] == to_segments[shared]
    {
        shared += 1;
    }
    let mut parts: Vec<&str> = vec![".."; from_segments.len() - shared];
    parts.extend(&to_segments[shared..]);
    parts.join("/")
}

/// Node's POSIX `normalizeString` over a joined absolute path: empty and
/// `"."` segments drop, `".."` pops one segment and drops entirely at the
/// root, duplicate and trailing separators collapse, and every other segment
/// — including `"..."` and `"..b"` — passes through.
fn normalize_posix_joined(joined: &str) -> String {
    let mut segments: Vec<&str> = Vec::new();
    for segment in joined.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            other => segments.push(other),
        }
    }
    if segments.is_empty() {
        "/".to_string()
    } else {
        format!("/{}", segments.join("/"))
    }
}
