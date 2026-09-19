//! The [`ProviderStreams`] default deferred-response methods, from
//! `packages/ai/src/types.ts`: adapters that do not support deferred
//! responses return `None` and the abort-shaped failure without
//! implementing the methods, upstream's optional-method contract.

use pi_ai::types::{
    Api, Context, DeferredFetchOptions, DeferredHandle, Modality, Model, ModelCost, ProviderId,
    ProviderStreams, SimpleStreamOptions, StreamOptions,
};
use pi_ai::utils::event_stream::{
    AssistantMessageEventStream, create_assistant_message_event_stream,
};

/// A minimal stream adapter that implements only the two required methods,
/// standing in for an API module without deferred responses.
struct PlainStreams;

impl ProviderStreams for PlainStreams {
    fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        create_assistant_message_event_stream()
    }

    fn stream_simple(
        &self,
        _model: &Model,
        _context: &Context,
        _options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        create_assistant_message_event_stream()
    }
}

fn model() -> Model {
    Model {
        id: String::from("test-model"),
        name: String::from("Test Model"),
        api: Api::from("test-api"),
        provider: ProviderId::from("test-provider"),
        base_url: String::from("https://api.example.com"),
        reasoning: false,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: ModelCost {
            rates: pi_ai::types::ModelCostRates {
                input: 1.0,
                output: 2.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
            tiers: None,
        },
        context_window: 128_000,
        max_tokens: 8_192,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn deferred_handle() -> DeferredHandle {
    DeferredHandle {
        provider: String::from("test-provider"),
        model_id: String::from("test-model"),
        api: String::from("test-api"),
        id: String::from("resp_1"),
        expires_at: None,
        poll_after_ms: None,
        data: None,
    }
}

#[tokio::test]
async fn the_default_fetch_and_cancel_methods_decline_deferred_responses() {
    let streams = PlainStreams;
    let model = model();

    assert!(
        streams
            .fetch_deferred(
                &model,
                &deferred_handle(),
                Some(&DeferredFetchOptions::default())
            )
            .is_none(),
        "an adapter without deferred support fetches nothing"
    );
    let cancelled = streams
        .cancel_deferred(&model, &deferred_handle(), None)
        .await;
    assert_eq![
        cancelled,
        Err(pi_ai::utils::provider_retry::ProviderRequestError::aborted())
    ];
}
