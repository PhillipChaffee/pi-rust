//! The wire-API module seam, ported from `packages/ai/src/api/` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatement: upstream's per-API modules under `src/api/` export
//! `ProviderStreams` implementations and are loaded lazily by the provider
//! factories. The wire-API implementations land with their own tickets;
//! until then each constructor returns the [`not_ported_streams`] stub whose
//! streams fail on dispatch with the same shape upstream's missing-API
//! dispatch produces.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::types::ProviderStreams;

pub mod anthropic_messages;
pub mod constrained_sampling;
pub mod github_copilot_headers;
pub mod google_generative_ai;
pub mod google_shared;
pub mod google_vertex;
pub mod lazy;
pub mod simple_options;
pub mod transform_messages;

/// The stub failure an unported wire-API stream reports.
#[derive(Debug)]
struct NotPortedError(String);

impl std::fmt::Display for NotPortedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NotPortedError {}

/// The stub streams of an unported wire API: every stream terminates with an
/// error event carrying the not-yet-ported notice.
struct NotPortedStreams {
    api: String,
}

impl ProviderStreams for NotPortedStreams {
    fn stream(
        &self,
        model: &crate::types::Model,
        _context: &crate::types::Context,
        _options: Option<&crate::types::StreamOptions>,
    ) -> crate::utils::event_stream::AssistantMessageEventStream {
        not_ported_stream(model, &self.api)
    }

    fn stream_simple(
        &self,
        model: &crate::types::Model,
        _context: &crate::types::Context,
        _options: Option<&crate::types::SimpleStreamOptions>,
    ) -> crate::utils::event_stream::AssistantMessageEventStream {
        not_ported_stream(model, &self.api)
    }
}

fn not_ported_stream(
    model: &crate::types::Model,
    api: &str,
) -> crate::utils::event_stream::AssistantMessageEventStream {
    let stream = crate::utils::event_stream::assistant_message_event_stream();
    let message = format!(
        "The {api} wire API has not been ported yet; the implementation lands with its ticket"
    );
    let error: lazy::LazyStreamError = Box::new(NotPortedError(message));
    let static_error: &(dyn std::error::Error + 'static) = error.as_ref();
    let failing = lazy::setup_error_message(model, static_error);
    stream.push(crate::types::AssistantMessageEvent::Error {
        reason: crate::types::StopReason::Error,
        error: failing.clone(),
    });
    stream.end(Some(&failing));
    stream
}

/// The not-yet-ported wire-API seam: a [`ProviderStreams`] whose streams
/// report the missing implementation. Replaced module by module as the
/// wire-API tickets land.
#[must_use]
pub fn not_ported_streams(api: &str) -> Arc<dyn ProviderStreams> {
    Arc::new(NotPortedStreams {
        api: api.to_owned(),
    })
}

/// The Anthropic Messages wire API, upstream's `anthropicMessagesApi()`.
#[must_use]
pub fn anthropic_messages() -> Arc<dyn ProviderStreams> {
    Arc::new(anthropic_messages::AnthropicStreams)
}

/// The OpenAI Responses wire API, upstream's `openAIResponsesApi()`.
#[must_use]
pub fn openai_responses() -> Arc<dyn ProviderStreams> {
    not_ported_streams("openai-responses")
}

/// The Azure OpenAI Responses wire API, upstream's
/// `azureOpenAIResponsesApi()`.
#[must_use]
pub fn azure_openai_responses() -> Arc<dyn ProviderStreams> {
    not_ported_streams("azure-openai-responses")
}

/// The OpenAI Completions wire API, upstream's `openAICompletionsApi()`.
#[must_use]
pub fn openai_completions() -> Arc<dyn ProviderStreams> {
    not_ported_streams("openai-completions")
}

/// The Google Generative AI wire API, upstream's `googleGenerativeAIApi()`.
#[must_use]
pub fn google_generative_ai() -> Arc<dyn ProviderStreams> {
    Arc::new(google_generative_ai::GoogleStreams)
}

/// The Google Vertex AI wire API, upstream's `googleVertexApi()`.
#[must_use]
pub fn google_vertex() -> Arc<dyn ProviderStreams> {
    Arc::new(google_vertex::GoogleVertexStreams)
}

/// The Bedrock Converse Stream wire API, upstream's
/// `bedrockConverseStreamApi()`.
#[must_use]
pub fn bedrock_converse_stream() -> Arc<dyn ProviderStreams> {
    not_ported_streams("bedrock-converse-stream")
}

/// The Mistral Conversations wire API, upstream's `mistralConversationsApi()`.
#[must_use]
pub fn mistral_conversations() -> Arc<dyn ProviderStreams> {
    not_ported_streams("mistral-conversations")
}

/// The OpenAI Codex Responses wire API, upstream's
/// `openAICodexResponsesApi()`.
#[must_use]
pub fn openai_codex_responses() -> Arc<dyn ProviderStreams> {
    not_ported_streams("openai-codex-responses")
}

/// The pi-messages wire API, upstream's `piMessagesApi()`.
#[must_use]
pub fn pi_messages() -> Arc<dyn ProviderStreams> {
    not_ported_streams("pi-messages")
}

/// An API implementation map keyed by wire-API id, upstream's
/// `Partial<Record<TApi, ProviderStreams>>` argument to `createProvider`.
pub type ApiMap = BTreeMap<String, Arc<dyn ProviderStreams>>;

/// The not-yet-ported image-API seam: a [`crate::types::ProviderImages`]
/// whose generation reports the missing implementation. Replaced as the
/// images ticket lands.
#[must_use]
pub fn not_ported_images(api: &str) -> Arc<dyn crate::types::ProviderImages> {
    Arc::new(NotPortedImages {
        api: api.to_owned(),
    })
}

/// The stub generation of an unported image API.
struct NotPortedImages {
    api: String,
}

impl crate::types::ProviderImages for NotPortedImages {
    fn generate_images<'a>(
        &'a self,
        _model: &'a crate::types::ImagesModel,
        _context: &'a crate::types::ImagesContext,
        _options: Option<&'a crate::types::ImagesOptions>,
    ) -> crate::types::BoxedFuture<
        'a,
        Result<crate::types::AssistantImages, crate::utils::provider_retry::ProviderRequestError>,
    > {
        let message = format!(
            "The {} image API has not been ported yet; the implementation lands with its ticket",
            self.api
        );
        Box::pin(async move {
            Err(crate::utils::provider_retry::ProviderRequestError::new(
                None, None, message,
            ))
        })
    }
}
