//! Version checks against the pi.dev release API, upstream's
//! `src/utils/version-check.ts` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream reads the ambient `globalThis.fetch` and `process.env`; the
//! Rust seam threads the `HttpClient` explicitly (the crate's
//! `fetchWithRetry` convention) and takes an [`EnvLookup`] — the plain
//! functions read the real environment, the `_with` forms inject one.

use std::cmp::Ordering;
use std::error::Error as StdError;
use std::sync::Arc;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use pi_ai::http::HttpClient;

use crate::config::EnvLookup;
use crate::utils::management_http::{FetchRetryOptions, fetch_with_retry};
use crate::utils::pi_user_agent::get_pi_user_agent;

/// The release-check endpoint, upstream's `LATEST_VERSION_URL`.
pub const LATEST_VERSION_URL: &str = "https://pi.dev/api/latest-version";
/// The default request budget, upstream's `DEFAULT_VERSION_CHECK_TIMEOUT_MS`.
pub const DEFAULT_VERSION_CHECK_TIMEOUT_MS: u64 = 10_000;

/// The release record the version API answers with, upstream's
/// `LatestPiRelease`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatestPiRelease {
    /// The latest version string, upstream's `version`.
    pub version: String,
    /// The package the release publishes under, when the check relocates
    /// pi, upstream's `packageName?`.
    pub package_name: Option<String>,
    /// A rendered release note, upstream's `note?`.
    pub note: Option<String>,
}

/// Include the error-chain details node's generic "fetch failed" hides,
/// upstream's `formatVersionCheckError`.
///
/// Upstream walks `error.cause` (and `AggregateError.errors`) collecting
/// errno `code` strings, deduplicating them into `root (A, B)`; when no
/// codes surface it reports the first cause message as `root (cause: m)`.
/// Rust errors carry no errno fields, so the code collection reads
/// `std::io::ErrorKind` names off the source chain — the closest portable
/// detail — and the cause branch reads the chain's first non-empty message.
#[must_use]
pub fn format_version_check_error(error: &(dyn StdError + 'static)) -> String {
    let root_message = error.to_string();
    let mut causes: Vec<String> = Vec::new();
    let mut codes: Vec<String> = Vec::new();
    let mut source = error.source();
    while let Some(cause) = source {
        if let Some(io_error) = cause.downcast_ref::<std::io::Error>() {
            let kind = format!("{:?}", io_error.kind());
            if !codes.contains(&kind) {
                codes.push(kind);
            }
        } else {
            let message = cause.to_string();
            if !message.is_empty() {
                causes.push(message);
            }
        }
        source = cause.source();
    }

    if !codes.is_empty() {
        return format!("{root_message} ({})", codes.join(", "));
    }
    if let Some(cause_message) = causes.first() {
        return format!("{root_message} (cause: {cause_message})");
    }
    root_message
}

/// Compare two version strings, upstream's `comparePackageVersions`:
/// semver-valid versions order, anything else is no comparison
/// (upstream's `undefined`).
#[must_use]
pub fn compare_package_versions(left_version: &str, right_version: &str) -> Option<Ordering> {
    let left = semver::Version::parse(left_version.trim()).ok()?;
    let right = semver::Version::parse(right_version.trim()).ok()?;
    Some(left.cmp(&right))
}

/// Whether the candidate is a newer release than the running version,
/// upstream's `isNewerPackageVersion`.
///
/// Valid semver orders; when either side does not parse, any difference in
/// the trimmed strings counts as newer, upstream's fallback.
#[must_use]
pub fn is_newer_package_version(candidate_version: &str, current_version: &str) -> bool {
    compare_package_versions(candidate_version, current_version).map_or_else(
        || candidate_version.trim() != current_version.trim(),
        |ordering| ordering == Ordering::Greater,
    )
}

/// Fetch the latest release record, upstream's `getLatestPiRelease`.
///
/// Offline mode answers `Ok(None)` without a request (upstream's truthy
/// `PI_OFFLINE` check — any non-empty value counts, unlike the package
/// manager's `1`/`true`/`yes` vocabulary). A failed request is
/// [`Err`](pi_ai::http::HttpError); a non-ok response or a body without a
/// usable version is `Ok(None)`.
///
/// # Errors
/// The last attempt's transport failure from [`fetch_with_retry`], the
/// caller's cancellation, or the total budget's expiry.
pub async fn get_latest_pi_release_with(
    client: &Arc<dyn HttpClient>,
    signal: CancellationToken,
    current_version: &str,
    options: VersionCheckOptions,
    env: &EnvLookup,
) -> Result<Option<LatestPiRelease>, pi_ai::http::HttpError> {
    if env("PI_OFFLINE").is_some_and(|value| !value.is_empty()) {
        return Ok(None);
    }

    let response = fetch_with_retry(
        client,
        LATEST_VERSION_URL,
        vec![
            ("User-Agent".to_string(), get_pi_user_agent(current_version)),
            ("accept".to_string(), "application/json".to_string()),
        ],
        signal,
        FetchRetryOptions {
            max_retries: Some(if options.retry { 2 } else { 0 }),
            timeout_ms: Some(
                options
                    .timeout_ms
                    .unwrap_or(DEFAULT_VERSION_CHECK_TIMEOUT_MS),
            ),
            ..FetchRetryOptions::default()
        },
    )
    .await?;
    if response.status < 200 || response.status >= 300 {
        return Ok(None);
    }

    let text = pi_ai::http::client::read_body_text(response.body).await?;
    let Ok(data) = serde_json::from_str::<Value>(&text) else {
        return Ok(None);
    };
    let Some(version) = data.get("version").and_then(Value::as_str) else {
        return Ok(None);
    };
    if version.trim().is_empty() {
        return Ok(None);
    }
    let package_name = data
        .get("packageName")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string);
    let note = data
        .get("note")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|note| !note.is_empty())
        .map(str::to_string);
    Ok(Some(LatestPiRelease {
        version: version.trim().to_string(),
        package_name,
        note,
    }))
}

/// The per-call knobs, upstream's `{ timeoutMs?: number; retry?: boolean }`.
#[derive(Debug, Clone, Copy, Default)]
pub struct VersionCheckOptions {
    /// The total request budget in milliseconds; the 10 s default rides
    /// [`DEFAULT_VERSION_CHECK_TIMEOUT_MS`].
    pub timeout_ms: Option<u64>,
    /// Retry transient failures twice, upstream's `retry?: boolean`.
    pub retry: bool,
}

/// [`get_latest_pi_release_with`] over the real environment.
///
/// # Errors
/// As [`get_latest_pi_release_with`].
pub async fn get_latest_pi_release(
    client: &Arc<dyn HttpClient>,
    signal: CancellationToken,
    current_version: &str,
    options: VersionCheckOptions,
) -> Result<Option<LatestPiRelease>, pi_ai::http::HttpError> {
    get_latest_pi_release_with(
        client,
        signal,
        current_version,
        options,
        &crate::config::default_env_lookup(),
    )
    .await
}

/// The latest version string only, upstream's `getLatestPiVersion`.
///
/// # Errors
/// As [`get_latest_pi_release_with`].
pub async fn get_latest_pi_version_with(
    client: &Arc<dyn HttpClient>,
    signal: CancellationToken,
    current_version: &str,
    options: VersionCheckOptions,
    env: &EnvLookup,
) -> Result<Option<String>, pi_ai::http::HttpError> {
    Ok(
        get_latest_pi_release_with(client, signal, current_version, options, env)
            .await?
            .map(|release| release.version),
    )
}

/// The automatic startup check, upstream's `checkForNewPiVersion`:
/// `PI_SKIP_VERSION_CHECK` suppresses the call entirely, a failed or
/// stale check answers `None` rather than an error.
pub async fn check_for_new_pi_version_with(
    client: &Arc<dyn HttpClient>,
    signal: CancellationToken,
    current_version: &str,
    env: &EnvLookup,
) -> Option<LatestPiRelease> {
    if env("PI_SKIP_VERSION_CHECK").is_some_and(|value| !value.is_empty()) {
        return None;
    }
    let latest_release = get_latest_pi_release_with(
        client,
        signal,
        current_version,
        VersionCheckOptions::default(),
        env,
    )
    .await
    .ok()
    .flatten();
    match latest_release {
        Some(release) if is_newer_package_version(&release.version, current_version) => {
            Some(release)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "the unit tests pin release checks; an unexpected result panics the test by design"
    )]
    use super::*;
    use std::io::ErrorKind;

    use pi_ai::http::{MockHttpClient, json_response};

    use crate::utils::test_env::lookup_with;

    fn offline_lookup(entries: &[(&str, &str)]) -> EnvLookup {
        lookup_with(entries)
    }

    #[tokio::test]
    async fn fetches_the_release_record_with_the_pi_user_agent() {
        let client = MockHttpClient::new();
        client
            .on(|request| request.url == LATEST_VERSION_URL)
            .respond(json_response(
                200,
                &serde_json::json!({ "version": "1.2.4" }),
            ));
        let client_arc: Arc<dyn HttpClient> = Arc::new(client.clone());
        let release = get_latest_pi_release_with(
            &client_arc,
            CancellationToken::new(),
            "1.2.3",
            VersionCheckOptions::default(),
            &offline_lookup(&[]),
        )
        .await
        .expect("release")
        .expect("version present");
        assert_eq!(release.version, "1.2.4");

        let recorded = client.recorded();
        assert_eq!(recorded.len(), 1);
        let user_agent = recorded[0]
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("user-agent"))
            .map(|(_, value)| value.clone())
            .unwrap_or_default();
        assert!(user_agent.starts_with("pi/1.2.3 "));
        let accept = recorded[0]
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("accept"))
            .map(|(_, value)| value.clone())
            .unwrap_or_default();
        assert_eq!(accept, "application/json");
    }

    #[tokio::test]
    async fn offline_mode_skips_the_request() {
        let client = MockHttpClient::new();
        // No route: an unmatched request would fail loudly; offline mode
        // must not reach it.
        let client_arc: Arc<dyn HttpClient> = Arc::new(client.clone());
        let release = get_latest_pi_release_with(
            &client_arc,
            CancellationToken::new(),
            "1.2.3",
            VersionCheckOptions::default(),
            &offline_lookup(&[("PI_OFFLINE", "1")]),
        )
        .await
        .expect("no error");
        assert!(release.is_none());
        assert!(client.recorded().is_empty());
    }

    #[test]
    fn compares_semver_and_falls_back_to_trimmed_difference() {
        assert_eq!(
            compare_package_versions("0.70.6", "0.70.5"),
            Some(Ordering::Greater)
        );
        assert_eq!(
            compare_package_versions("0.70.5", "0.70.5"),
            Some(Ordering::Equal)
        );
        assert_eq!(
            compare_package_versions("0.70.4", "0.70.5"),
            Some(Ordering::Less)
        );
        assert_eq!(
            compare_package_versions("5.0.0-beta.20", "5.0.0-beta.9"),
            Some(Ordering::Greater)
        );
        assert_eq!(compare_package_versions("abc", "0.70.5"), None);
        assert!(is_newer_package_version("0.70.6", "0.70.5"));
        assert!(!is_newer_package_version("0.70.5", "0.70.5"));
        assert!(!is_newer_package_version("next", "next "));
        assert!(is_newer_package_version("next", "other"));
    }

    #[test]
    fn formats_error_chain_details() {
        // The io-kind collection rides the source chain.
        struct Chain(std::io::Error);
        impl std::fmt::Debug for Chain {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("Chain")
            }
        }
        impl std::fmt::Display for Chain {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("fetch failed")
            }
        }
        impl StdError for Chain {
            fn source(&self) -> Option<&(dyn StdError + 'static)> {
                Some(&self.0)
            }
        }
        // No source chain: the root message alone.
        let error = pi_ai::http::HttpError::Transport("fetch failed".to_string());
        assert_eq!(format_version_check_error(&error), "fetch failed");
        let chained = Chain(std::io::Error::new(ErrorKind::TimedOut, "connect timeout"));
        assert_eq!(
            format_version_check_error(&chained),
            "fetch failed (TimedOut)"
        );
    }
}
