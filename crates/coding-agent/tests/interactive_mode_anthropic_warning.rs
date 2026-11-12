//! `test/interactive-mode-anthropic-warning.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! Restatements:
//! - The mock `modelRuntime.checkAuth`/`getAuth` become the real in-memory
//!   model runtime over a seeded auth store: an api-key credential whose key
//!   starts `sk-ant-oat`, and an OAuth credential.
//! - The `showWarning` spy becomes the transcript's warning rows; the counts
//!   assert the same once-per-process behavior.

#![expect(
    clippy::expect_used,
    reason = "the fixture auth map and model are pinned; a malformed fixture is a test bug"
)]

mod common;

use std::collections::BTreeMap;

use common::interactive::{Harness, HarnessOptions};
use pi_ai::auth::types::{Credential, OAuthCredentials};
use pi_coding_agent::auth_storage::AuthStorageData;
use pi_coding_agent::settings_manager::Settings;

/// The warning text prefix, upstream's `ANTHROPIC_SUBSCRIPTION_AUTH_WARNING`.
const WARNING: &str = "Anthropic subscription auth is active.";

fn api_key_auth(key: &str) -> AuthStorageData {
    let mut data = AuthStorageData::new();
    data.insert(
        "anthropic".to_owned(),
        serde_json::json!({ "type": "api_key", "key": key }),
    );
    data
}

fn oauth_auth() -> AuthStorageData {
    let credential = Credential::OAuth(OAuthCredentials {
        refresh: "refresh-token".to_owned(),
        access: "access-token".to_owned(),
        expires: 4_102_444_800_000,
        extra: BTreeMap::new(),
    });
    let mut data = AuthStorageData::new();
    data.insert(
        "anthropic".to_owned(),
        serde_json::to_value(&credential).expect("credential serializes"),
    );
    data
}

fn warning_count(harness: &Harness) -> usize {
    harness.chat_text().matches(WARNING).count()
}

#[tokio::test]
async fn warns_once_when_anthropic_subscription_auth_is_detected() {
    let harness = Harness::new(HarnessOptions {
        model: common::agent_session::anthropic_model(),
        auth: api_key_auth("sk-ant-oat01-test"),
        ..HarnessOptions::default()
    })
    .await;

    harness
        .frame
        .maybe_warn_about_anthropic_subscription_auth(None)
        .await;
    harness
        .frame
        .maybe_warn_about_anthropic_subscription_auth(None)
        .await;

    assert_eq!(warning_count(&harness), 1);
}

#[tokio::test]
async fn warns_when_anthropic_oauth_is_stored_even_if_token_refresh_lookup_would_fail() {
    let harness = Harness::new(HarnessOptions {
        model: common::agent_session::anthropic_model(),
        auth: oauth_auth(),
        ..HarnessOptions::default()
    })
    .await;

    harness
        .frame
        .maybe_warn_about_anthropic_subscription_auth(None)
        .await;

    assert_eq!(warning_count(&harness), 1);
}

#[tokio::test]
async fn does_not_warn_for_non_anthropic_models() {
    let harness = Harness::new(HarnessOptions {
        model: common::model_layer::model("openai", "gpt-test"),
        auth: AuthStorageData::new(),
        ..HarnessOptions::default()
    })
    .await;

    harness
        .frame
        .maybe_warn_about_anthropic_subscription_auth(None)
        .await;

    assert_eq!(warning_count(&harness), 0);
}

#[tokio::test]
async fn does_not_warn_when_anthropic_extra_usage_warning_is_disabled() {
    let mut settings = Settings::new();
    settings.insert(
        "warnings".to_owned(),
        serde_json::json!({ "anthropicExtraUsage": false }),
    );
    let harness = Harness::new(HarnessOptions {
        model: common::agent_session::anthropic_model(),
        settings,
        auth: AuthStorageData::new(),
        ..HarnessOptions::default()
    })
    .await;

    harness
        .frame
        .maybe_warn_about_anthropic_subscription_auth(None)
        .await;

    assert_eq!(warning_count(&harness), 0);
}
