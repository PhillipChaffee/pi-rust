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
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "the unit tests pin changelog parsing; an unexpected result panics the test by design"
    )]
    use super::*;

    #[test]
    fn parses_entries_between_headers() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("CHANGELOG.md");
        std::fs::write(
            &path,
            "# Title\n\n## [1.2.3] - date\n\nFirst body.\n\n## [0.9.0]\n\nOlder.\ntrailing line\n",
        )
        .expect("write");
        let entries = parse_changelog(&path);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].major, 1);
        assert_eq!(entries[0].content, "## [1.2.3] - date\n\nFirst body.");
        assert_eq!(entries[1].content, "## [0.9.0]\n\nOlder.\ntrailing line");
    }

    #[test]
    fn resets_on_unparseable_headers_and_missing_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("CHANGELOG.md");
        std::fs::write(&path, "## [1.0.0]\n\nA\n\n## Nope\n\nB\n").expect("write");
        let entries = parse_changelog(&path);
        assert_eq!(entries.len(), 1);
        assert!(parse_changelog(&dir.path().join("absent.md")).is_empty());
    }

    #[test]
    fn normalizes_legacy_repo_and_floating_refs() {
        assert_eq!(
            normalize_changelog_links(
                "[x](https://github.com/badlogic/pi-mono/blob/main/docs/a.md)",
                "1.2.3"
            ),
            "[x](https://github.com/earendil-works/pi/blob/v1.2.3/docs/a.md)"
        );
        assert_eq!(
            normalize_changelog_links(
                "[x](https://github.com/earendil-works/pi/tree/master/packages/coding-agent)",
                "1.2.3"
            ),
            "[x](https://github.com/earendil-works/pi/tree/v1.2.3/packages/coding-agent)"
        );
    }

    #[test]
    fn routes_local_paths_to_blob_or_tree() {
        let out = normalize_changelog_links("[x](docs/a.md)", "1.2.3");
        assert_eq!(
            out,
            "[x](https://github.com/earendil-works/pi/blob/v1.2.3/packages/coding-agent/docs/a.md)"
        );
        let dir_out = normalize_changelog_links("[x](docs/)", "1.2.3");
        assert_eq!(
            dir_out,
            "[x](https://github.com/earendil-works/pi/tree/v1.2.3/packages/coding-agent/docs/)"
        );
        // A basename without a dot routes as a tree even without a slash.
        let pkg_out = normalize_changelog_links("[x](skills)", "1.2.3");
        assert_eq!(
            pkg_out,
            "[x](https://github.com/earendil-works/pi/tree/v1.2.3/packages/coding-agent/skills)"
        );
    }

    #[test]
    fn leaves_anchors_urls_and_escaping_paths_untouched() {
        assert_eq!(
            normalize_changelog_links("[x](#section)", "1.2.3"),
            "[x](#section)"
        );
        assert_eq!(
            normalize_changelog_links("[x](https://example.com/a.md)", "1.2.3"),
            "[x](https://example.com/a.md)"
        );
        // A climb that folds back inside the repository rewrites; the fold
        // above the repository root is what stays untouched (upstream's
        // `resolveRepositoryPath` rejects only `joined.startsWith("../")`).
        assert_eq!(
            normalize_changelog_links("[x](../../outside.md)", "1.2.3"),
            "[x](https://github.com/earendil-works/pi/blob/v1.2.3/outside.md)"
        );
        assert_eq!(
            normalize_changelog_links("[x](../../../outside.md)", "1.2.3"),
            "[x](../../../outside.md)"
        );
        // The fragment and query ride the rewritten URL verbatim.
        assert_eq!(
            normalize_changelog_links("[x](docs/a.md?x=1#frag)", "1.2.3"),
            "[x](https://github.com/earendil-works/pi/blob/v1.2.3/packages/coding-agent/docs/a.md?x=1#frag)"
        );
    }

    #[test]
    fn encodes_spaces_and_keeps_reserved_bytes() {
        // The encoding contract, ECMA-262 `encodeURI`: the space escapes,
        // the reserved bytes ride through.
        assert_eq!(encode_uri("docs/my file.md"), "docs/my%20file.md");
        // At the markdown level the inline-link regex's target group
        // excludes whitespace (upstream's `[^\s)]+`), so a space-bearing
        // link splits: the "docs/my" prefix rewrites (dotless basename →
        // tree route) and the " file.md)" suffix rides verbatim.
        assert_eq!(
            normalize_changelog_links("[x](docs/my file.md)", "1.2.3"),
            "[x](https://github.com/earendil-works/pi/tree/v1.2.3/packages/coding-agent/docs/my file.md)"
        );
    }

    #[test]
    fn compares_and_filters_entries() {
        let one = ChangelogEntry {
            major: 1,
            minor: 2,
            patch: 3,
            content: String::new(),
        };
        let two = ChangelogEntry {
            major: 1,
            minor: 2,
            patch: 4,
            content: String::new(),
        };
        assert_eq!(compare_versions(&two, &one), std::cmp::Ordering::Greater);
        let entries = vec![one.clone(), two.clone()];
        assert_eq!(get_new_entries(&entries, "1.2.3"), vec![&two]);
        // `Number("v1")` and the missing components both land at zero, so
        // the fallback baseline (0,2,2) and (0,0,0) leave both entries newer.
        assert_eq!(get_new_entries(&entries, "v1.2.2"), vec![&one, &two]);
        assert_eq!(get_new_entries(&entries, ""), vec![&one, &two]);
        assert_eq!(normalize_tag("1.2.3"), "v1.2.3");
        assert_eq!(normalize_tag("v1.2.3"), "v1.2.3");
    }
}
