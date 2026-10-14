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
