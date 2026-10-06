//! The path-normalization slice of upstream's `src/utils/paths.ts` that the
//! config foundation consumes.
//!
//! The belt's remaining exports (`canonicalize`, the cwd-relative
//! formatters, the Windows shell-path rewrites) land with the utils ticket
//! and their consumers.
//!
//! Two restatements pin the byte-for-byte contract. `normalizePath` carries
//! more options than the config foundation reads — the tilde slice is the
//! only behavior this module reproduces, because upstream's
//! `expandTildePath`/`resolvePath` calls all run with default options, whose
//! tilde handling is exactly this. `resolvePath` resolves with Node's POSIX
//! `path.resolve`, which differs from Rust's `std::path::absolute` in three
//! observable ways the vectors pin: `..` segments collapse lexically instead
//! of surviving, trailing slashes drop, and a leading `//` collapses to `/`
//! (upstream runs `path.resolve`, whose output feeds the session-dir
//! encoding).

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

/// Resolve `input` against `base_dir`, upstream's `resolvePath` at default
/// options.
///
/// The tilde slice of `normalizePath` runs first, an empty input collapses
/// to the normalized base (Node's `path.resolve` skips empty arguments), an
/// absolute input normalizes in place, and a relative input joins onto
/// `base_dir` first. The result is Node's POSIX `path.resolve` byte for
/// byte; see the module docs for the three places that differs from
/// `std::path::absolute`. The base is always absolute in this crate's calls
/// (the process cwd), which is what makes the lexical collapse of `..`
/// segments exact: a `..` climbing past the root drops, matching upstream's
/// `normalizeString` with `allowAboveRoot = false`.
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
