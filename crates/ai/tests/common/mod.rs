//! Shared test fixtures for the utils belt suites, mirroring the shapes
//! upstream's `fauxAssistantMessage` and message helpers feed the same
//! code paths.

#![expect(
    dead_code,
    reason = "shared fixtures; each test binary uses the subset it needs"
)]
#![expect(
    unreachable_pub,
    reason = "the fixture module is compiled into every integration test binary as a private module"
)]
#![expect(
    clippy::expect_used,
    reason = "the block helper panics on a runtime build failure by design; tests pin outcomes"
)]

pub mod auth_fixtures;
pub mod auth_guards;
pub mod auth_interaction;
pub mod auth_json;
pub mod oauth_fixtures;
pub mod paused_clock;
pub mod radius_fixtures;
pub mod seam_forms;

use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, Message, ProviderId, StopReason, TextContent, Usage,
    UsageCost, UserContent, UserMessage,
};

/// A provider-shaped usage block with the given total.
#[must_use]
pub fn usage(total_tokens: u64) -> Usage {
    Usage {
        input: total_tokens,
        output: 0,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens,
        cost: UsageCost::default(),
    }
}

/// An assistant message carrying one text block, the faux provider's
/// success shape.
#[must_use]
pub fn assistant_message(text: &str) -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        api: Api::from("test-api"),
        provider: ProviderId::from("test-provider"),
        model: String::from("test-model"),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    }
}

/// An assistant message with overridden stop reason and error text.
#[must_use]
pub fn assistant_message_with(
    text: &str,
    stop_reason: StopReason,
    error_message: Option<String>,
) -> AssistantMessage {
    let mut message = assistant_message(text);
    message.stop_reason = stop_reason;
    message.error_message = error_message;
    message
}

/// A minimal assistant message with no content, for error/usage shapes.
#[must_use]
pub fn bare_assistant_message() -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: Api::from("test-api"),
        provider: ProviderId::from("test-provider"),
        model: String::from("test-model"),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Pending,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    }
}

/// A user message with a plain-string content.
#[must_use]
pub fn user_message(text: &str, timestamp: i64) -> Message {
    Message::User(UserMessage {
        content: UserContent::Text(text.to_owned()),
        timestamp,
    })
}

/// A text block for content comparisons.
#[must_use]
pub fn text_block(text: &str) -> TextContent {
    TextContent {
        text: text.to_owned(),
        text_signature: None,
    }
}

/// An assistant message with only stop reason set, for retry shapes that
/// check the reason alone.
#[must_use]
pub fn aborted_message() -> AssistantMessage {
    assistant_message_with("", StopReason::Aborted, None)
}

// --- The Models-runtime fixtures the #28 runtime suites share ---

use std::future::Future;
use std::sync::{Arc, Mutex};

use pi_ai::auth::types::{ApiKeyAuth, ApiKeyAuthInput, ProviderAuth};
use pi_ai::types::{
    BoxedFuture, Context, DeferredCancelOptions, DeferredFetchOptions, DeferredHandle, Model,
    ProviderStreams, SimpleStreamOptions, StreamOptions,
};

/// Run a future to completion on a fresh current-thread runtime, the seam
/// sync tests use to drive the async surface.
pub fn block<F: Future>(future: F) -> F::Output {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    runtime.block_on(future)
}

/// The minimal chat model every runtime suite streams against.
#[must_use]
pub fn fixture_model() -> Model {
    Model {
        id: "m".to_owned(),
        name: "m".to_owned(),
        api: Api::from("test-api"),
        provider: ProviderId::from("p"),
        base_url: "https://example.test/v1".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![pi_ai::types::Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 1000,
        max_tokens: 100,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// The settled assistant message a fixture stream ends with.
#[must_use]
pub fn message_fixture(model: &Model, stop_reason: StopReason) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }
}

/// Ambient api-key auth whose resolve always succeeds, the configured stub.
#[must_use]
pub fn ambient_auth() -> ProviderAuth {
    ProviderAuth {
        api_key: Some(ApiKeyAuth {
            name: "Ambient".to_owned(),
            login: None,
            check: None,
            resolve: Arc::new(|_input: ApiKeyAuthInput| {
                Box::pin(async move { Ok(Some(pi_ai::auth::types::AuthResult::default())) })
            }),
        }),
        oauth: None,
    }
}

/// The empty context every stream request carries.
#[must_use]
pub fn context() -> Context {
    Context::default()
}

/// The deferred handle the deferred dispatch tests route through.
#[must_use]
pub fn deferred_handle() -> DeferredHandle {
    DeferredHandle {
        provider: "p".to_owned(),
        model_id: "m".to_owned(),
        api: "test-api".to_owned(),
        id: "id".to_owned(),
        expires_at: None,
        poll_after_ms: None,
        data: None,
    }
}

/// A stream that settles immediately with the fixture message.
#[must_use]
pub fn end_with(model: &Model) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
    let stream = pi_ai::utils::event_stream::assistant_message_event_stream();
    stream.end(Some(&message_fixture(model, StopReason::Stop)));
    stream
}

/// The deferred-capable streams fixture: streams end with the fixture
/// message, fetches carry the handle through, and both fetch and cancel are
/// counted so suites can assert the dispatch reached the provider.
pub struct DeferredStreams {
    /// How many fetches reached the fixture.
    pub fetches: Arc<Mutex<u64>>,
    /// How many cancels reached the fixture.
    pub cancels: Arc<Mutex<u64>>,
}

impl DeferredStreams {
    /// The fixture with its two counters, the shape suites assert on.
    #[must_use]
    pub fn new() -> (Self, Arc<Mutex<u64>>, Arc<Mutex<u64>>) {
        let fetches = Arc::new(Mutex::new(0));
        let cancels = Arc::new(Mutex::new(0));
        (
            Self {
                fetches: Arc::clone(&fetches),
                cancels: Arc::clone(&cancels),
            },
            fetches,
            cancels,
        )
    }

    /// The fixture when only the behavior matters, not the counters.
    #[must_use]
    pub fn uncounted() -> Self {
        Self {
            fetches: Arc::new(Mutex::new(0)),
            cancels: Arc::new(Mutex::new(0)),
        }
    }
}

impl ProviderStreams for DeferredStreams {
    fn stream(
        &self,
        model: &Model,
        _context: &Context,
        _options: Option<&StreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        end_with(model)
    }

    fn stream_simple(
        &self,
        model: &Model,
        _context: &Context,
        _options: Option<&SimpleStreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        end_with(model)
    }

    fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        _options: Option<&DeferredFetchOptions>,
    ) -> Option<pi_ai::utils::event_stream::AssistantMessageEventStream> {
        *self
            .fetches
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
        let stream = pi_ai::utils::event_stream::assistant_message_event_stream();
        let mut message = message_fixture(model, StopReason::Stop);
        message.deferred = Some(handle.clone());
        stream.end(Some(&message));
        Some(stream)
    }

    fn cancel_deferred<'a>(
        &'a self,
        _model: &'a Model,
        _handle: &'a DeferredHandle,
        _options: Option<&'a DeferredCancelOptions>,
    ) -> BoxedFuture<'a, Result<(), pi_ai::utils::provider_retry::ProviderRequestError>> {
        *self
            .cancels
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
        Box::pin(async { Ok(()) })
    }

    fn supports_fetch_deferred(&self) -> bool {
        true
    }

    fn supports_cancel_deferred(&self) -> bool {
        true
    }
}

/// The minimal image model every image suite streams against.
#[must_use]
pub fn image_model(provider: &str, id: &str) -> pi_ai::types::ImagesModel {
    pi_ai::types::ImagesModel {
        id: id.to_owned(),
        name: id.to_owned(),
        api: pi_ai::types::ImagesApi::from("test-images"),
        provider: pi_ai::types::ImagesProviderId::from(provider),
        base_url: "https://example.test/v1".to_owned(),
        thinking_level_map: None,
        input: vec![pi_ai::types::Modality::Text],
        output: vec![pi_ai::types::Modality::Image],
        cost: pi_ai::types::ModelCost {
            rates: pi_ai::types::ModelCostRates::default(),
            tiers: None,
        },
        sampling_params: None,
        headers: None,
    }
}

/// The settled image result a fixture image api returns.
#[must_use]
pub fn ok_images_result(model: &pi_ai::types::ImagesModel) -> pi_ai::types::AssistantImages {
    pi_ai::types::AssistantImages {
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        output: vec![pi_ai::types::ImagesBlock::Image(
            pi_ai::types::ImageContent {
                data: "aGk=".to_owned(),
                mime_type: "image/png".to_owned(),
            },
        )],
        response_id: None,
        usage: None,
        stop_reason: pi_ai::types::ImagesStopReason::Stop,
        error_message: None,
        timestamp: pi_ai::auth::resolve::now_ms(),
    }
}

/// The image-generation context every image suite sends.
#[must_use]
pub fn images_context() -> pi_ai::types::ImagesContext {
    pi_ai::types::ImagesContext {
        input: vec![pi_ai::types::ImagesBlock::Text(TextContent {
            text: "a red circle".to_owned(),
            text_signature: None,
        })],
    }
}
