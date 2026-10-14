//! Changelog parsing and release-note link normalization, upstream's
//! `src/utils/changelog.ts` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! No upstream suite covers this belt; the boundary suite binds the
//! regex-driven arms (legacy-repo rewrite, floating-ref pinning, local-path
//! resolution, directory-vs-blob routing, and the `encodeURI` keep-set).

use std::fmt::Write as _;
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;

/// One `## [x.y.z]` entry, upstream's `ChangelogEntry`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangelogEntry {
    /// The major component, upstream's `major`.
    pub major: i64,
    /// The minor component, upstream's `minor`.
    pub minor: i64,
    /// The patch component, upstream's `patch`.
    pub patch: i64,
    /// The body lines under the header, trimmed, upstream's `content`.
    pub content: String,
}

const GITHUB_REPO: &str = "earendil-works/pi";
const CHANGELOG_LINK_BASE_PATH: &str = "packages/coding-agent";

/// The inline Markdown link shape, upstream's `LINK_RE`.
#[expect(
    clippy::expect_used,
    reason = "the pattern is a compile-time constant; a failure is a programming error, not a runtime condition"
)]
static LINK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(!?\[[^\]\n]+\]\()([^\s)]+)((?:\s+[^)]*)?\))")
        .expect("the link pattern is a valid regex")
});

/// The `## [x.y.z]` release header, upstream's `RELEASE_HEADER_RE`.
#[expect(
    clippy::expect_used,
    reason = "the pattern is a compile-time constant; a failure is a programming error, not a runtime condition"
)]
static HEADER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"##\s+\[?(\d+)\.(\d+)\.(\d+)\]?").expect("the header pattern is a valid regex")
});

/// The tag a version renders to, upstream's `normalizeTag`: a `v` prefix
/// added when absent.
#[must_use]
pub fn normalize_tag(version: &str) -> String {
    if version.starts_with('v') {
        version.to_string()
    } else {
        format!("v{version}")
    }
}

/// Split a link target into its path, query, and fragment parts, upstream's
/// `splitLocalTarget`: the fragment is everything from the first `#`, the
/// query everything from the first `?` before it.
fn split_local_target(target: &str) -> (String, String, String) {
    let (before_hash, fragment) = target.find('#').map_or((target, ""), |hash_index| {
        (&target[..hash_index], &target[hash_index..])
    });
    before_hash.find('?').map_or_else(
        || (before_hash.to_string(), String::new(), fragment.to_string()),
        |query_index| {
            (
                before_hash[..query_index].to_string(),
                before_hash[query_index..].to_string(),
                fragment.to_string(),
            )
        },
    )
}

/// Node's `path.posix.normalize` over a lexical path: collapse duplicate
/// separators, resolve `.` segments, and consume `..` against the segments
/// already present (a leading `..` survives).
fn posix_normalize(value: &str) -> String {
    // Node's `path.posix.normalize` contract: dot segments fold, empty
    // segments collapse, a trailing separator survives exactly once, and the
    // fully-collapsed shapes come back as `/`, `./`, or `.`.
    if value.is_empty() {
        return ".".to_string();
    }
    let is_absolute = value.starts_with('/');
    let trailing_slash = value.ends_with('/');
    let mut segments: Vec<&str> = Vec::new();
    for segment in value.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                if let Some(last) = segments.last()
                    && *last != ".."
                {
                    segments.pop();
                    continue;
                }
                segments.push("..");
            }
            other => segments.push(other),
        }
    }
    let joined = segments.join("/");
    if joined.is_empty() {
        if is_absolute {
            return "/".to_string();
        }
        return if trailing_slash {
            "./".to_string()
        } else {
            ".".to_string()
        };
    }
    let mut normalized = joined;
    if trailing_slash {
        normalized.push('/');
    }
    if is_absolute {
        normalized.insert(0, '/');
    }
    normalized
}

/// Resolve a link's path part to its repository-relative path, upstream's
/// `resolveRepositoryPath`: package-rooted relative paths join under the
/// coding-agent package; absolute paths normalize in place; paths escaping
/// the repository are dropped.
fn resolve_repository_path(target_path: &str) -> Option<String> {
    let normalized_target = target_path.replace('\\', "/");
    let joined = if normalized_target.starts_with('/') {
        posix_normalize(normalized_target.trim_start_matches('/'))
    } else {
        posix_normalize(&format!("{CHANGELOG_LINK_BASE_PATH}/{normalized_target}"))
    };

    if joined == "." || joined == ".." || joined.starts_with("../") {
        return None;
    }
    Some(joined)
}

/// Whether the link target names a directory, upstream's
/// `isDirectoryTarget`: a trailing slash, or a basename without a dot.
fn is_directory_target(original_path: &str, repository_path: &str) -> bool {
    if original_path.ends_with('/') {
        return true;
    }
    let basename = repository_path
        .rsplit('/')
        .next()
        .unwrap_or(repository_path);
    !basename.contains('.')
}

/// ECMA-262 `encodeURI`: percent-encode every byte outside the unreserved
/// set plus the URI reserved characters it keeps literally.
fn encode_uri(value: &str) -> String {
    let kept = |byte: u8| {
        byte.is_ascii_alphanumeric()
            || matches!(
                byte,
                b';' | b','
                    | b'/'
                    | b'?'
                    | b':'
                    | b'@'
                    | b'&'
                    | b'='
                    | b'+'
                    | b'$'
                    | b'-'
                    | b'_'
                    | b'.'
                    | b'!'
                    | b'~'
                    | b'*'
                    | b'\''
                    | b'('
                    | b')'
                    | b'#'
            )
    };
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if kept(byte) {
            encoded.push(byte as char);
        } else {
            // The fmt error cannot fire: appending to a String never fails.
            let _ = write!(encoded, "%{byte:02X}");
        }
    }
    encoded
}

/// Rewrite one link target to its tagged repository URL, upstream's
/// `normalizeChangelogLinkTarget`.
fn normalize_changelog_link_target(target: &str, tag: &str) -> String {
    let repo_url = format!("https://github.com/{GITHUB_REPO}");
    let legacy_repo_prefixes = [
        "https://github.com/badlogic/pi-mono".to_string(),
        "https://github.com/earendil-works/pi-mono".to_string(),
    ];
    let mut canonical_target = target.to_string();
    for prefix in legacy_repo_prefixes {
        // The upstream regex anchors at the start and requires a following
        // slash or end; the rewrite replaces exactly that prefix.
        if canonical_target.starts_with(&format!("{prefix}/")) || canonical_target == prefix {
            let suffix = canonical_target.strip_prefix(&prefix).unwrap_or("");
            canonical_target = format!("{repo_url}{suffix}");
            break;
        }
    }

    for route in ["blob", "tree"] {
        for branch in ["main", "master"] {
            let floating_ref_prefix = format!("{repo_url}/{route}/{branch}/");
            if let Some(rest) = canonical_target.strip_prefix(&floating_ref_prefix) {
                canonical_target = format!("{repo_url}/{route}/{tag}/{rest}");
            }
        }
    }

    if canonical_target.starts_with('#')
        || canonical_target.starts_with("//")
        || url_scheme_prefix(canonical_target.as_bytes())
    {
        return canonical_target;
    }

    let (path_part, query, fragment) = split_local_target(&canonical_target);
    if path_part.is_empty() {
        return canonical_target;
    }

    let Some(repository_path) = resolve_repository_path(&path_part) else {
        return canonical_target;
    };

    let route = if is_directory_target(&path_part, &repository_path) {
        "tree"
    } else {
        "blob"
    };
    format!(
        "{repo_url}/{route}/{tag}/{}{query}{fragment}",
        encode_uri(&repository_path)
    )
}

/// Whether the bytes open with a URL scheme, upstream's `URL_SCHEME_RE`:
/// an ASCII letter followed by letters, digits, `+`, `.`, or `-`, then `:`.
const fn url_scheme_prefix(bytes: &[u8]) -> bool {
    let Some(first) = bytes.first() else {
        return false;
    };
    if !first.is_ascii_alphabetic() {
        return false;
    }
    let mut index = 1;
    while index < bytes.len()
        && (bytes[index].is_ascii_alphanumeric() || matches!(bytes[index], b'+' | b'.' | b'-'))
    {
        index += 1;
    }
    index < bytes.len() && bytes[index] == b':'
}

/// Rewrite release-note links to their tagged repository URLs, upstream's
/// `normalizeChangelogLinks`.
///
/// Inline Markdown links (`[text](target)` and image forms) get their
/// targets normalized; everything else passes through.
#[must_use]
pub fn normalize_changelog_links(markdown: &str, version: &str) -> String {
    let tag = normalize_tag(version);
    LINK_RE
        .replace_all(markdown, |captures: &regex::Captures| {
            let prefix = captures.get(1).map(|m| m.as_str()).unwrap_or_default();
            let target = captures.get(2).map(|m| m.as_str()).unwrap_or_default();
            let suffix = captures.get(3).map(|m| m.as_str()).unwrap_or_default();
            format!(
                "{prefix}{}{suffix}",
                normalize_changelog_link_target(target, &tag)
            )
        })
        .into_owned()
}

/// Parse `## [x.y.z]` entries from a changelog file, upstream's
/// `parseChangelog`.
///
/// A missing file is no entries; a version header switches the collector
/// (an unparseable header resets it); content runs to the next header or
/// EOF, trimmed. Read failures print the upstream warning to stderr and
/// answer no entries, upstream's `console.error` catch.
#[must_use]
pub fn parse_changelog(changelog_path: &Path) -> Vec<ChangelogEntry> {
    if !changelog_path.exists() {
        return Vec::new();
    }
    let content = match std::fs::read_to_string(changelog_path) {
        Ok(content) => content,
        Err(error) => {
            #[expect(
                clippy::print_stderr,
                reason = "console.error is the surface upstream carries; the warning must reach the operator on stderr"
            )]
            fn warn_parse_failed(error: &std::io::Error) {
                eprintln!("Warning: Could not parse changelog: {error}");
            }
            warn_parse_failed(&error);
            return Vec::new();
        }
    };

    let mut entries: Vec<ChangelogEntry> = Vec::new();
    let mut current_lines: Vec<String> = Vec::new();
    let mut current_version: Option<(i64, i64, i64)> = None;

    for line in content.split('\n') {
        if line.starts_with("## ") {
            if let Some((major, minor, patch)) = current_version
                && !current_lines.is_empty()
            {
                entries.push(ChangelogEntry {
                    major,
                    minor,
                    patch,
                    content: current_lines.join("\n").trim().to_string(),
                });
            }

            if let Some(captures) = HEADER_RE.captures(line) {
                let parse = |index: usize| {
                    captures
                        .get(index)
                        .and_then(|m| m.as_str().parse::<i64>().ok())
                        .unwrap_or_default()
                };
                current_version = Some((parse(1), parse(2), parse(3)));
                current_lines = vec![line.to_string()];
            } else {
                current_version = None;
                current_lines.clear();
            }
        } else if current_version.is_some() {
            current_lines.push(line.to_string());
        }
    }

    if let Some((major, minor, patch)) = current_version
        && !current_lines.is_empty()
    {
        entries.push(ChangelogEntry {
            major,
            minor,
            patch,
            content: current_lines.join("\n").trim().to_string(),
        });
    }
    entries
}

/// Order two entries, upstream's `compareVersions`: major, then minor,
/// then patch.
#[must_use]
pub fn compare_versions(left: &ChangelogEntry, right: &ChangelogEntry) -> std::cmp::Ordering {
    (left.major, left.minor, left.patch).cmp(&(right.major, right.minor, right.patch))
}

/// The entries newer than the version string, upstream's `getNewEntries`.
///
/// The version splits on `.`; each component parses to a number or falls
/// back to zero, upstream's `Number(...) || 0` (missing and non-numeric
/// parts both land at zero).
#[must_use]
pub fn get_new_entries<'a>(
    entries: &'a [ChangelogEntry],
    last_version: &str,
) -> Vec<&'a ChangelogEntry> {
    let parse_component = |part: Option<&str>| -> i64 {
        part.and_then(|part| part.trim().parse::<i64>().ok())
            .unwrap_or(0)
    };
    let mut parts = last_version.split('.');
    let last = ChangelogEntry {
        major: parse_component(parts.next()),
        minor: parse_component(parts.next()),
        patch: parse_component(parts.next()),
        content: String::new(),
    };

    entries
        .iter()
        .filter(|entry| compare_versions(entry, &last) == std::cmp::Ordering::Greater)
        .collect()
}

#[cfg(test)]
mod tests;
