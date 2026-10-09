//! The version-check suite, upstream's `test/version-check.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The suite also carries the changelog link-normalization edge arms
//! (upstream's `src/utils/changelog.ts`): the changelog feeds the
//! release-note announcements this belt's version check drives.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;
use tokio_util::sync::CancellationToken;

use pi_ai::http::{HttpClient, MockHttpClient, MockResponse, json_response};
use pi_coding_agent::config::EnvLookup;
use pi_coding_agent::utils::changelog::{normalize_changelog_links, parse_changelog};
use pi_coding_agent::utils::version_check::{
    LATEST_VERSION_URL, LatestPiRelease, VersionCheckOptions, check_for_new_pi_version_with,
    compare_package_versions, format_version_check_error, get_latest_pi_release,
    get_latest_pi_release_with, get_latest_pi_version_with, is_newer_package_version,
};

fn env_with(entries: &[(&str, &str)]) -> EnvLookup {
    let owned: Vec<(String, String)> = entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect();
    Box::new(move |key| {
        owned
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.clone())
    })
}

fn client_for(body: &serde_json::Value) -> (MockHttpClient, Arc<dyn HttpClient>) {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == LATEST_VERSION_URL)
        .respond(json_response(200, body));
    let seam: Arc<dyn HttpClient> = Arc::new(mock.clone());
    (mock, seam)
}

#[test]
fn compares_package_versions() {
    assert!(
        compare_package_versions("0.70.6", "0.70.5")
            .is_some_and(|ordering| ordering == std::cmp::Ordering::Greater)
    );
    assert!(
        compare_package_versions("0.70.5", "0.70.5")
            .is_some_and(|ordering| ordering == std::cmp::Ordering::Equal)
    );
    assert!(
        compare_package_versions("0.70.4", "0.70.5")
            .is_some_and(|ordering| ordering == std::cmp::Ordering::Less)
    );
    assert!(
        compare_package_versions("5.0.0-beta.20", "5.0.0-beta.9")
            .is_some_and(|ordering| ordering == std::cmp::Ordering::Greater)
    );
    assert!(!is_newer_package_version("0.70.5", "0.70.5"));
    assert!(is_newer_package_version("0.70.6", "0.70.5"));
}

#[tokio::test]
async fn returns_only_newer_versions() {
    let (_mock, seam) = client_for(&json!({ "version": "1.2.3" }));
    let env = env_with(&[]);

    assert!(
        check_for_new_pi_version_with(&seam, CancellationToken::new(), "1.2.3", &env)
            .await
            .is_none()
    );
    assert_eq!(
        check_for_new_pi_version_with(&seam, CancellationToken::new(), "1.2.2", &env).await,
        Some(LatestPiRelease {
            version: "1.2.3".to_string(),
            package_name: None,
            note: None,
        })
    );
}

#[tokio::test]
async fn uses_the_pi_dev_version_check_api_with_a_pi_user_agent() {
    let (mock, seam) = client_for(&json!({ "version": "1.2.4" }));
    let version = get_latest_pi_version_with(
        &seam,
        CancellationToken::new(),
        "1.2.3",
        VersionCheckOptions::default(),
        &env_with(&[]),
    )
    .await
    .expect("check")
    .expect("version");
    assert_eq!(version, "1.2.4");

    let recorded = mock.recorded();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].url, LATEST_VERSION_URL);
    let user_agent = recorded[0]
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("user-agent"))
        .map(|(_, value)| value.clone())
        .unwrap_or_default();
    assert!(
        user_agent.starts_with("pi/1.2.3 "),
        "the pi user agent prefixes the version: {user_agent}"
    );
    let accept = recorded[0]
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("accept"))
        .map(|(_, value)| value.clone())
        .unwrap_or_default();
    assert_eq!(accept, "application/json");
}

#[tokio::test]
async fn retries_a_transient_version_request_when_explicitly_requested() {
    let mock = MockHttpClient::new();
    let attempts = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&attempts);
    mock.on(|request| request.url == LATEST_VERSION_URL)
        .respond_fn(move |_request| {
            let nth = counter.fetch_add(1, Ordering::SeqCst);
            async move {
                if nth < 2 {
                    Err(pi_ai::http::HttpError::Transport(
                        "fetch failed".to_string(),
                    ))
                } else {
                    Ok(json_response(200, &json!({ "version": "1.2.4" })))
                }
            }
        });
    let seam: Arc<dyn HttpClient> = Arc::new(mock.clone());

    let release = get_latest_pi_release_with(
        &seam,
        CancellationToken::new(),
        "1.2.3",
        VersionCheckOptions {
            retry: true,
            ..VersionCheckOptions::default()
        },
        &env_with(&[]),
    )
    .await
    .expect("check")
    .expect("version");
    assert_eq!(release.version, "1.2.4");
    assert_eq!(
        attempts.load(Ordering::SeqCst),
        3,
        "two failures then the answer"
    );
}

#[tokio::test]
async fn keeps_automatic_version_checks_to_one_request() {
    let mock = MockHttpClient::new();
    let attempts = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&attempts);
    mock.on(|request| request.url == LATEST_VERSION_URL)
        .respond_fn(move |_request| {
            let _ = counter.fetch_add(1, Ordering::SeqCst);
            async move {
                Err(pi_ai::http::HttpError::Transport(
                    "fetch failed".to_string(),
                ))
            }
        });
    let seam: Arc<dyn HttpClient> = Arc::new(mock.clone());

    assert!(
        check_for_new_pi_version_with(&seam, CancellationToken::new(), "1.2.3", &env_with(&[]))
            .await
            .is_none()
    );
    assert_eq!(
        attempts.load(Ordering::SeqCst),
        1,
        "the automatic check never retries"
    );
}

#[test]
fn formats_nested_network_error_details() {
    // The Rust-side restatement: the io-kind names ride the source chain
    // where upstream surfaced errno codes from the AggregateError.
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
    impl std::error::Error for Chain {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(&self.0)
        }
    }
    let chained = Chain(std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        "connect timeout",
    ));
    assert_eq!(
        format_version_check_error(&chained),
        "fetch failed (TimedOut)"
    );
}

#[tokio::test]
async fn returns_the_active_package_metadata_from_the_version_check_api() {
    let (_mock, seam) = client_for(&json!({ "packageName": "@new-scope/pi", "version": "1.2.4" }));
    let release = get_latest_pi_release_with(
        &seam,
        CancellationToken::new(),
        "1.2.3",
        VersionCheckOptions::default(),
        &env_with(&[]),
    )
    .await
    .expect("check")
    .expect("release");
    assert_eq!(
        release,
        LatestPiRelease {
            version: "1.2.4".to_string(),
            package_name: Some("@new-scope/pi".to_string()),
            note: None,
        }
    );
}

#[tokio::test]
async fn returns_update_notes_from_the_version_check_api() {
    let (_mock, seam) = client_for(&json!({ "note": " **Read this** ", "version": "1.2.4" }));
    let release = get_latest_pi_release_with(
        &seam,
        CancellationToken::new(),
        "1.2.3",
        VersionCheckOptions::default(),
        &env_with(&[]),
    )
    .await
    .expect("check")
    .expect("release");
    assert_eq!(release.note.as_deref(), Some("**Read this**"));
    assert_eq!(release.version, "1.2.4");
}

#[tokio::test]
async fn skips_automatic_api_calls_when_version_checks_are_disabled() {
    let mock = MockHttpClient::new();
    let seam: Arc<dyn HttpClient> = Arc::new(mock.clone());
    assert!(
        check_for_new_pi_version_with(
            &seam,
            CancellationToken::new(),
            "1.2.3",
            &env_with(&[("PI_SKIP_VERSION_CHECK", "1")])
        )
        .await
        .is_none()
    );
    assert_eq!(mock.recorded().len(), 0, "no request leaves the process");
}

#[tokio::test]
async fn allows_direct_api_calls_when_automatic_version_checks_are_disabled() {
    let (mock, seam) = client_for(&json!({ "version": "1.2.4" }));
    let version = get_latest_pi_version_with(
        &seam,
        CancellationToken::new(),
        "1.2.3",
        VersionCheckOptions::default(),
        &env_with(&[("PI_SKIP_VERSION_CHECK", "1")]),
    )
    .await
    .expect("check")
    .expect("version");
    assert_eq!(version, "1.2.4");
    assert_eq!(mock.recorded().len(), 1, "the direct call runs");
}

#[tokio::test]
async fn returns_none_on_error_statuses() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == LATEST_VERSION_URL)
        .respond(json_response(500, &json!({ "version": "1.2.4" })));
    let seam: Arc<dyn HttpClient> = Arc::new(mock.clone());

    let release = get_latest_pi_release_with(
        &seam,
        CancellationToken::new(),
        "1.2.3",
        VersionCheckOptions::default(),
        &env_with(&[]),
    )
    .await
    .expect("no error");
    assert!(release.is_none(), "an error status is no release");
    assert_eq!(mock.recorded().len(), 1);
}

#[tokio::test]
async fn returns_none_when_the_body_is_not_json() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == LATEST_VERSION_URL)
        .respond(MockResponse::status(200).with_body("not json"));
    let seam: Arc<dyn HttpClient> = Arc::new(mock.clone());

    let release = get_latest_pi_release_with(
        &seam,
        CancellationToken::new(),
        "1.2.3",
        VersionCheckOptions::default(),
        &env_with(&[]),
    )
    .await
    .expect("no error");
    assert!(release.is_none(), "a non-JSON body is no release");
}

#[tokio::test]
async fn returns_none_when_the_body_has_no_version() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == LATEST_VERSION_URL)
        .respond(json_response(200, &json!({ "packageName": "pi" })));
    let seam: Arc<dyn HttpClient> = Arc::new(mock.clone());

    let release = get_latest_pi_release_with(
        &seam,
        CancellationToken::new(),
        "1.2.3",
        VersionCheckOptions::default(),
        &env_with(&[]),
    )
    .await
    .expect("no error");
    assert!(release.is_none(), "a body without a version is no release");
}

#[tokio::test]
async fn returns_none_when_the_version_is_blank() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == LATEST_VERSION_URL)
        .respond(json_response(200, &json!({ "version": "   " })));
    let seam: Arc<dyn HttpClient> = Arc::new(mock.clone());

    let release = get_latest_pi_release_with(
        &seam,
        CancellationToken::new(),
        "1.2.3",
        VersionCheckOptions::default(),
        &env_with(&[]),
    )
    .await
    .expect("no error");
    assert!(release.is_none(), "a blank version is no release");
}

#[tokio::test]
async fn drops_blank_package_names_and_notes() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == LATEST_VERSION_URL)
        .respond(json_response(
            200,
            &json!({
                "version": " 1.2.4 ",
                "packageName": "   ",
                "note": "  ",
            }),
        ));
    let seam: Arc<dyn HttpClient> = Arc::new(mock.clone());

    let release = get_latest_pi_release_with(
        &seam,
        CancellationToken::new(),
        "1.2.3",
        VersionCheckOptions::default(),
        &env_with(&[]),
    )
    .await
    .expect("check")
    .expect("the version carries the release");
    assert_eq!(
        release,
        LatestPiRelease {
            version: "1.2.4".to_string(),
            package_name: None,
            note: None,
        },
        "blank metadata fields drop, the version trims"
    );
}

#[tokio::test]
async fn honors_the_caller_timeout_budget() {
    let (mock, seam) = client_for(&json!({ "version": "1.2.4" }));
    let version = get_latest_pi_version_with(
        &seam,
        CancellationToken::new(),
        "1.2.3",
        VersionCheckOptions {
            timeout_ms: Some(1_000),
            retry: false,
        },
        &env_with(&[]),
    )
    .await
    .expect("check")
    .expect("version");
    assert_eq!(version, "1.2.4");
    assert_eq!(mock.recorded().len(), 1);
}

#[tokio::test]
async fn surfaces_exhausted_retries_as_the_last_transport_error() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == LATEST_VERSION_URL)
        .respond_fn(|_request| async move {
            Err(pi_ai::http::HttpError::Transport(
                "fetch failed".to_string(),
            ))
        });
    let seam: Arc<dyn HttpClient> = Arc::new(mock.clone());

    let error = get_latest_pi_release_with(
        &seam,
        CancellationToken::new(),
        "1.2.3",
        VersionCheckOptions {
            retry: true,
            ..VersionCheckOptions::default()
        },
        &env_with(&[]),
    )
    .await
    .expect_err("the exhausted retries surface");
    assert_eq!(
        error,
        pi_ai::http::HttpError::Transport("fetch failed".to_string()),
        "the last attempt's failure is the answer"
    );
}

#[tokio::test]
async fn the_release_wrapper_reads_the_process_environment() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == LATEST_VERSION_URL)
        .respond(json_response(200, &json!({ "version": "1.2.4" })));
    let seam: Arc<dyn HttpClient> = Arc::new(mock.clone());

    // The process environment decides offline mode; the wrapper resolves
    // either way and the mock answers whenever a request leaves.
    let release = get_latest_pi_release(
        &seam,
        CancellationToken::new(),
        "1.2.3",
        VersionCheckOptions::default(),
    )
    .await
    .expect("the process-env wrapper resolves");
    if let Some(release) = release {
        assert_eq!(release.version, "1.2.4");
    }
}

#[test]
fn formats_cause_details_and_deduplicates_io_kinds() {
    struct TestError {
        message: &'static str,
        source: Option<Box<dyn std::error::Error + 'static>>,
    }
    impl std::fmt::Debug for TestError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.message)
        }
    }
    impl std::fmt::Display for TestError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.message)
        }
    }
    impl std::error::Error for TestError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            self.source.as_deref()
        }
    }
    let error = |message: &'static str, source: Option<Box<dyn std::error::Error + 'static>>| {
        TestError { message, source }
    };

    // A non-io source chain reports the first cause message, upstream's
    // `root (cause: m)` shape.
    let caused = error(
        "fetch failed",
        Some(Box::new(error("connect timeout", None))),
    );
    assert_eq!(
        format_version_check_error(&caused),
        "fetch failed (cause: connect timeout)"
    );

    // An empty cause message is skipped; the next one carries the detail.
    let skipped = error(
        "fetch failed",
        Some(Box::new(error(
            "",
            Some(Box::new(error("connect timeout", None))),
        ))),
    );
    assert_eq!(
        format_version_check_error(&skipped),
        "fetch failed (cause: connect timeout)"
    );

    // Repeated io kinds deduplicate into one code.
    let inner = std::io::Error::new(std::io::ErrorKind::ConnectionAborted, "leaf");
    let outer = std::io::Error::new(std::io::ErrorKind::ConnectionAborted, inner);
    let duplicated = error("fetch failed", Some(Box::new(outer)));
    assert_eq!(
        format_version_check_error(&duplicated),
        "fetch failed (ConnectionAborted)"
    );

    // The Debug impl rides the same message the Display impl reports.
    assert_eq!(format!("{caused:?}"), "fetch failed");
}

#[test]
fn normalizes_release_note_link_edge_targets() {
    // The repository-root target folds to "." and is rejected: the link
    // stays verbatim.
    assert_eq!(normalize_changelog_links("[x](/)", "1.2.3"), "[x](/)");
    // A repository-rooted path rewrites from the repo root, no package
    // prefix joins.
    assert_eq!(
        normalize_changelog_links("[x](/docs/a.md)", "1.2.3"),
        "[x](https://github.com/earendil-works/pi/blob/v1.2.3/docs/a.md)"
    );
    // A climb that folds the whole package path away rejects the target.
    assert_eq!(
        normalize_changelog_links("[x](../..)", "1.2.3"),
        "[x](../..)"
    );
    // A fully-folded trailing climb routes as the folded directory.
    assert_eq!(
        normalize_changelog_links("[x](../../)", "1.2.3"),
        "[x](https://github.com/earendil-works/pi/tree/v1.2.3/./)"
    );
    // A climb folding into the package root routes as the package tree.
    assert_eq!(
        normalize_changelog_links("[x](../)", "1.2.3"),
        "[x](https://github.com/earendil-works/pi/tree/v1.2.3/packages/)"
    );
    // A query-only target carries no path to resolve and rides verbatim.
    assert_eq!(normalize_changelog_links("[x](?)", "1.2.3"), "[x](?)");
}

#[test]
fn warns_and_answers_no_entries_when_the_changelog_is_unreadable() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("CHANGELOG.md");
        std::fs::write(&path, "## [1.2.3]\n\nBody.\n").expect("write");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000))
            .expect("chmod unreadable");
        assert!(
            parse_changelog(&path).is_empty(),
            "an unreadable changelog is no entries"
        );
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))
            .expect("restore permissions");
    }
}
