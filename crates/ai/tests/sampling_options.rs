//! The sampling-parameter suite, ported from
//! `packages/ai/test/sampling-options.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`. Every case drives
//! `streamSimple` from the compat layer and captures the outgoing payload
//! through the request hook; the pinned refused endpoint keeps the real send
//! a local failure after the capture.

use std::collections::BTreeMap;

use pi_ai::compat::stream_simple;
use pi_ai::types::{
    Api, Context, Modality, Model, ProviderId, SimpleStreamOptions, TransportOptions,
};

mod common;

/// The single user-message context, upstream's `makeContext`.
fn make_context() -> Context {
    Context {
        messages: vec![common::user_message_now("Hello")],
        ..Context::default()
    }
}

/// The completions model on the refused loopback endpoint, upstream's
/// `makeCompletionsModel` with the sampling-params override.
fn completions_model(sampling_params: Option<BTreeMap<String, serde_json::Value>>) -> Model {
    Model {
        id: "custom-model".to_owned(),
        name: "Custom Model".to_owned(),
        api: Api::from("openai-completions"),
        provider: ProviderId::from("custom-provider"),
        base_url: "http://127.0.0.1:9/v1".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 128_000,
        max_tokens: 16_384,
        sampling_params,
        headers: None,
        compat: None,
    }
}

/// The anthropic model on the refused loopback endpoint, upstream's
/// `makeAnthropicModel`.
fn anthropic_model() -> Model {
    Model {
        id: "vendor--claude".to_owned(),
        name: "Vendor Proxy Claude".to_owned(),
        api: Api::from("anthropic-messages"),
        provider: ProviderId::from("vendor-proxy"),
        base_url: "http://127.0.0.1:9".to_owned(),
        reasoning: true,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 200_000,
        max_tokens: 32_000,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// Drive `streamSimple` from the compat layer and return the outgoing
/// payload, upstream's `capturePayload`: the hook captures the request body
/// before the send fails against the refused endpoint.
async fn capture_payload(
    model: &Model,
    sampling_params: Option<BTreeMap<String, serde_json::Value>>,
    temperature: Option<f64>,
) -> serde_json::Value {
    let (on_payload, captured) = common::payload_capture();
    let options = SimpleStreamOptions {
        api_key: Some("fake-key".to_owned()),
        sampling_params,
        temperature,
        transport_options: TransportOptions {
            on_payload: Some(on_payload),
            ..TransportOptions::default()
        },
        ..SimpleStreamOptions::default()
    };

    stream_simple(model, &make_context(), Some(&options))
        .result()
        .await;

    common::captured_payload(&captured)
}

/// Stream-option sampling params merge into the request body, upstream's
/// `merges stream-option sampling params into the request body`.
#[tokio::test]
async fn merges_stream_option_sampling_params_into_the_request_body() {
    let _guard = common::registry_guard().await;
    let payload = capture_payload(
        &completions_model(None),
        Some(BTreeMap::from([
            ("top_p".to_owned(), serde_json::json!(0.95)),
            ("top_k".to_owned(), serde_json::json!(0)),
            ("min_p".to_owned(), serde_json::json!(0)),
        ])),
        None,
    )
    .await;
    assert_eq!(payload.get("top_p"), Some(&serde_json::json!(0.95)));
    assert_eq!(payload.get("top_k"), Some(&serde_json::json!(0)));
    assert_eq!(payload.get("min_p"), Some(&serde_json::json!(0)));
}

/// No sampling params ride when neither the options nor the model set them,
/// upstream's `omits sampling params when neither options nor model set
/// them`.
#[tokio::test]
async fn omits_sampling_params_when_neither_options_nor_model_set_them() {
    let _guard = common::registry_guard().await;
    let payload = capture_payload(&completions_model(None), None, None).await;
    assert_eq!(payload.get("temperature"), None);
    assert_eq!(payload.get("top_p"), None);
}

/// Model-level sampling params apply, upstream's `applies model-level
/// sampling params`.
#[tokio::test]
async fn applies_model_level_sampling_params() {
    let _guard = common::registry_guard().await;
    let payload = capture_payload(
        &completions_model(Some(BTreeMap::from([
            ("temperature".to_owned(), serde_json::json!(1)),
            ("top_p".to_owned(), serde_json::json!(0.95)),
        ]))),
        None,
        None,
    )
    .await;
    assert_eq!(payload.get("temperature"), Some(&serde_json::json!(1)));
    assert_eq!(payload.get("top_p"), Some(&serde_json::json!(0.95)));
}

/// Stream-option keys override model-level keys per key, upstream's
/// `merges stream-option keys over model-level keys`.
#[tokio::test]
async fn merges_stream_option_keys_over_model_level_keys() {
    let _guard = common::registry_guard().await;
    let payload = capture_payload(
        &completions_model(Some(BTreeMap::from([
            ("top_p".to_owned(), serde_json::json!(0.95)),
            ("min_p".to_owned(), serde_json::json!(0.05)),
        ]))),
        Some(BTreeMap::from([(
            "top_p".to_owned(),
            serde_json::json!(0.5),
        )])),
        None,
    )
    .await;
    assert_eq!(payload.get("top_p"), Some(&serde_json::json!(0.5)));
    assert_eq!(payload.get("min_p"), Some(&serde_json::json!(0.05)));
}

/// Sampling-param keys override the named request fields, upstream's
/// `overrides named request fields`.
#[tokio::test]
async fn overrides_named_request_fields() {
    let _guard = common::registry_guard().await;
    let payload = capture_payload(
        &completions_model(None),
        Some(BTreeMap::from([(
            "temperature".to_owned(),
            serde_json::json!(1),
        )])),
        Some(0.0),
    )
    .await;
    assert_eq!(payload.get("temperature"), Some(&serde_json::json!(1)));
}

/// Sampling params are ignored by non-OpenAI-compatible APIs, upstream's
/// `is ignored by non-OpenAI-compatible APIs`.
#[tokio::test]
async fn sampling_params_are_ignored_by_non_openai_compatible_apis() {
    let _guard = common::registry_guard().await;
    let payload = capture_payload(
        &anthropic_model(),
        Some(BTreeMap::from([
            ("top_p".to_owned(), serde_json::json!(0.9)),
            ("top_k".to_owned(), serde_json::json!(40)),
        ])),
        None,
    )
    .await;
    assert_eq!(payload.get("top_p"), None);
    assert_eq!(payload.get("top_k"), None);
}
