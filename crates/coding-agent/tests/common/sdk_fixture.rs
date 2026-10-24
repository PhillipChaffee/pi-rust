//! The SDK factory/runtime fixtures the #150 suites share: the faux provider,
//! its model, and the temp cwd/agent-dir pair.

use std::sync::Arc;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::Api;
use pi_ai::types::AssistantBlock;
use pi_ai::types::AssistantMessage;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::Modality;
use pi_ai::types::Model;
use pi_ai::types::ProviderId;
use pi_ai::types::StopReason;
use pi_ai::types::TextContent;
use pi_coding_agent::model_registry::ModelRegistry;
use pi_coding_agent::provider_composer::ProviderConfigInput;
use pi_coding_agent::provider_composer::ProviderModelInput;
use tempfile::TempDir;

/// The fixture's model id.
pub const FAUX_MODEL_ID: &str = "faux-1";

/// The fixture's second model id.
pub const FAUX_SECOND_MODEL_ID: &str = "faux-2";

/// One provider model input with the fixture's defaults.
#[must_use]
pub fn provider_model(id: &str, reasoning: bool, modalities: Vec<Modality>) -> ProviderModelInput {
    ProviderModelInput {
        id: id.to_owned(),
        name: id.to_owned(),
        api: Some(Api::from("openai-completions")),
        base_url: None,
        reasoning,
        thinking_level_map: None,
        input: modalities,
        cost: super::model_layer::zero_cost(),
        context_window: 128_000,
        max_tokens: 4096,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// The fixture model, an `openai-completions` model over the faux provider.
#[must_use]
pub fn faux_model() -> Model {
    Model {
        id: FAUX_MODEL_ID.to_owned(),
        name: "Faux 1".to_owned(),
        api: Api::from("openai-completions"),
        provider: ProviderId("faux".to_owned()),
        base_url: "https://faux.invalid/v1".to_owned(),
        reasoning: true,
        thinking_level_map: None,
        input: vec![Modality::Text, Modality::Image],
        cost: super::model_layer::zero_cost(),
        context_window: 128_000,
        max_tokens: 4096,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// The done assistant message the faux stream settles with.
#[must_use]
pub fn done_message() -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantBlock::Text(TextContent {
            text: "ok".to_owned(),
            text_signature: None,
        })],
        api: Api::from("openai-completions"),
        provider: ProviderId("faux".to_owned()),
        model: FAUX_MODEL_ID.to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: pi_ai::types::Usage::default(),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    }
}

/// A registry with the faux provider registered, its key configured, and a
/// stream that settles with [`done_message`].
///
/// # Panics
/// The provider registration failure, which the fixture cannot satisfy.
pub async fn registry_with_faux() -> ModelRegistry {
    let registry = super::model_layer::create_in_memory_model_registry(
        super::model_layer::empty_auth_storage(),
    )
    .await;
    registry
        .runtime()
        .register_provider(
            "faux",
            ProviderConfigInput {
                base_url: Some("https://faux.invalid/v1".to_owned()),
                api: Some(Api::from("openai-completions")),
                api_key: Some("faux-key".to_owned()),
                stream_simple: Some(Arc::new(|_model, _context, _options| {
                    let stream = pi_ai::utils::event_stream::assistant_message_event_stream();
                    stream.push(AssistantMessageEvent::Done {
                        reason: StopReason::Stop,
                        message: done_message(),
                    });
                    stream
                })),
                models: Some(vec![
                    provider_model(FAUX_MODEL_ID, true, vec![Modality::Text, Modality::Image]),
                    provider_model(FAUX_SECOND_MODEL_ID, true, vec![Modality::Text]),
                ]),
                ..ProviderConfigInput::default()
            },
        )
        .expect("the provider registers");
    registry
}

/// A user message carrying `text`.
#[must_use]
pub fn user_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(pi_ai::types::Message::User(pi_ai::types::UserMessage {
        content: pi_ai::types::UserContent::Text(text.to_owned()),
        timestamp: 1,
    }))
}

/// An assistant message carrying `text`.
#[must_use]
pub fn assistant_message(text: &str) -> AgentMessage {
    let mut message = done_message();
    message.content = vec![AssistantBlock::Text(TextContent {
        text: text.to_owned(),
        text_signature: None,
    })];
    AgentMessage::Standard(pi_ai::types::Message::Assistant(message))
}

/// A fresh temp cwd/agent-dir pair.
#[derive(Debug)]
pub struct Fixture {
    pub temp: TempDir,
    pub cwd: String,
    pub agent_dir: String,
}

impl Fixture {
    /// Create the fixture's directories.
    #[must_use]
    pub fn new() -> Self {
        let temp = TempDir::new().expect("tempdir");
        let cwd = temp.path().join("project");
        let agent_dir = temp.path().join("agent");
        std::fs::create_dir_all(&cwd).expect("cwd");
        std::fs::create_dir_all(&agent_dir).expect("agent dir");
        Self {
            cwd: cwd.display().to_string(),
            agent_dir: agent_dir.display().to_string(),
            temp,
        }
    }
}

impl Default for Fixture {
    fn default() -> Self {
        Self::new()
    }
}
