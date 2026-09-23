//! The live-provider harness the credential-gated suites share, ported from
//! the helpers of `packages/ai/test/stream.test.ts`, `packages/ai/test/
//! empty.test.ts`, `packages/ai/test/abort.test.ts`, `packages/ai/test/
//! tokens.test.ts`, `packages/ai/test/total-tokens.test.ts`, and
//! `packages/ai/test/context-overflow.test.ts`, plus the `azure-utils.ts`,
//! `bedrock-utils.ts`, `cloudflare-utils.ts`, and `oauth.ts` test utilities,
//! all at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream rides `ProviderStreamOptions = StreamOptions & Record<string,
//! unknown>`: the option extras duck-type through compat `stream`/`complete`
//! into whichever adapter owns them. Rust has no duck typing, so
//! [`LiveOptions`] carries the extras explicitly and [`stream`] dispatches
//! each extra through the surface that owns it — the base fields through
//! compat `stream`, the reasoning level and thinking budgets through the
//! simple surface (`compat::stream_simple`), and the adapter-owned extras
//! (Anthropic thinking knobs, Azure deployment names, the Google thinking
//! control, Bedrock interleaving and request metadata) through the owning
//! adapter directly. Upstream resolves OAuth tokens once at module load; the
//! port resolves per test, the skipIf semantics preserved.

#![expect(
    clippy::expect_used,
    reason = "the live helpers pin upstream expectations; an unexpected shape panics the test by design"
)]
#![expect(
    clippy::print_stdout,
    reason = "an OAuth refresh failure surfaces through the log, upstream's console.log(JSON.stringify(error))"
)]
#![expect(
    deprecated,
    reason = "the harness drives the deprecated compat surface, upstream's getModel/getModels"
)]
#![expect(
    clippy::panic,
    reason = "the helpers pin live outcomes; an unexpected shape panics the test by design"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "the block-index reads, the overflow estimator, and the tool-result formatter mirror upstream's arithmetic; the lossy paths are unreachable at these magnitudes"
)]
#![expect(
    clippy::cast_sign_loss,
    reason = "the overflow estimator and tool-result formatter operate on non-negative upstream values"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "the overflow estimator mirrors upstream's chars/4 float math exactly"
)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use pi_ai::api::anthropic_messages::{AnthropicEffort, AnthropicStreamOptions};
use pi_ai::api::azure_openai_responses::AzureOpenAiResponsesOptions;
use pi_ai::api::bedrock_options::BedrockStreamOptions;
use pi_ai::api::google_generative_ai;
use pi_ai::api::google_shared::{GoogleOptions, GoogleThinkingControl};
use pi_ai::api::google_vertex;
use pi_ai::auth::resolve::now_ms;
use pi_ai::auth::types::{Credential, OAuthCredentials};
use pi_ai::cli::{load_credentials, save_credentials};
use pi_ai::compat::{
    get_env_api_key, get_model, get_models, stream as compat_stream,
    stream_simple as compat_stream_simple,
};
use pi_ai::providers::all::builtin_providers;

use pi_ai::types::{
    AssistantBlock, AssistantMessage, AssistantMessageEvent, Context, ImageContent, KnownApi,
    Message, Modality, Model, OnPayload, ProviderHeaders, SimpleStreamOptions, StopReason,
    StreamOptions, TextContent, ThinkingBudgets, ThinkingLevel, Tool, ToolResultBlock,
    ToolResultMessage, Transport, TransportOptions, Usage, UserBlock, UserContent, UserMessage,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use super::auth_guards;

/// The process environment as the guard signature reads it.
fn env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// Whether the Azure OpenAI live credentials are configured, upstream's
/// `hasAzureOpenAICredentials`: the api key plus a base URL or a resource
/// name.
#[must_use]
pub fn has_azure_openai_credentials() -> bool {
    auth_guards::has_azure_openai_credentials(&env)
}

/// The Azure deployment name mapped for `model_id`, upstream's
/// `resolveAzureDeploymentName`.
#[must_use]
pub fn azure_deployment_name(model_id: &str) -> Option<String> {
    auth_guards::resolve_azure_deployment_name(model_id, &env)
}

/// Whether any valid AWS credentials are configured for Bedrock, upstream's
/// `hasBedrockCredentials`.
#[must_use]
pub fn has_bedrock_credentials() -> bool {
    auth_guards::has_bedrock_credentials(&env)
}

/// Whether the Cloudflare Workers AI credentials are configured: the api key
/// plus the account id.
#[must_use]
pub fn has_cloudflare_workers_ai_credentials() -> bool {
    auth_guards::has_cloudflare_workers_ai_credentials(&env)
}

/// Whether the Cloudflare AI Gateway credentials are configured: the api key,
/// the account id, and the gateway id.
#[must_use]
pub fn has_cloudflare_ai_gateway_credentials() -> bool {
    auth_guards::has_cloudflare_ai_gateway_credentials(&env)
}

/// The credential store the OAuth-backed probes resolve through, upstream's
/// `AUTH_PATH` under the user's home.
fn auth_store() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .expect("the test environment carries HOME")
        .join(".pi")
        .join("agent")
        .join("auth.json")
}

/// The api key a probe uses, upstream's `test/oauth.ts` `resolveApiKey`: the
/// pi credential store first — refreshing an expired OAuth token and saving
/// it back — then the provider's env vars.
pub async fn resolve_api_key(provider: &str) -> Option<String> {
    let path = auth_store();
    let storage = load_credentials(&path.to_string_lossy());
    let stored = match storage.get(provider) {
        Some(credential) => match credential.api_key() {
            Some(key) => Some(key.to_owned()),
            None => match credential.as_oauth() {
                Some(oauth) => resolve_oauth_key(provider, &path, oauth).await,
                None => None,
            },
        },
        None => None,
    };
    stored.or_else(|| get_env_api_key(provider, None))
}

/// The credential store's OAuth arm: refresh an expired token, save it back,
/// and derive the request key.
async fn resolve_oauth_key(
    provider: &str,
    path: &Path,
    stored: &OAuthCredentials,
) -> Option<String> {
    let flow = builtin_providers()
        .into_iter()
        .find(|candidate| candidate.id() == provider)?
        .auth()
        .oauth
        .clone()?;
    let mut current = stored.clone();
    if now_ms() >= current.expires {
        let refreshed = (flow.refresh)(current, CancellationToken::new())
            .await
            .inspect_err(|error| println!("{error}"))
            .ok()?;
        save_credentials(path, provider, &Credential::OAuth(refreshed.clone())).ok()?;
        current = refreshed;
    }
    (flow.to_auth)(current).await.ok()?.api_key
}

/// The catalog model a block probes, upstream's `getModel`: absent from the
/// catalog is a test bug, so it panics.
#[must_use]
pub fn model(provider: &str, id: &str) -> Model {
    get_model(provider, id).unwrap_or_else(|| panic!("no catalog model {provider}/{id}"))
}

/// The catalog models a provider carries, upstream's `getModels`.
#[must_use]
pub fn models(provider: &str) -> Vec<Model> {
    get_models(provider)
}

/// The environment variable value, present and non-empty, upstream's truthy
/// `process.env.X` check.
#[must_use]
pub fn env_key(name: &str) -> Option<String> {
    env(name).filter(|value| !value.is_empty())
}

/// The calculator tool the tool-call and multi-turn helpers drive, upstream's
/// `calculatorTool` over the `StringEnum` schema Google accepts.
#[must_use]
pub fn calculator_tool() -> Tool {
    Tool {
        name: "math_operation".to_owned(),
        description: "Perform basic arithmetic operations".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {
                "a": { "type": "number", "description": "First number" },
                "b": { "type": "number", "description": "Second number" },
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The operation to perform. One of 'add', 'subtract', 'multiply', 'divide'."
                }
            },
            "required": ["a", "b", "operation"]
        }),
        constrained_sampling: None,
    }
}

/// The prompt-varying counter standing in for upstream's `Math.random()`
/// arithmetic probe: the value only keeps repeat calls from landing on a
/// warmed cache, so a deterministic sequence preserves that purpose.
fn next_arithmetic_operand() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed) % 255
}

/// The option extras a live block carries, upstream's
/// `StreamOptionsWithExtras` keys restated as fields. [`Self::reasoning`] is
/// upstream's `reasoningEffort` (and Bedrock's `reasoning`); the
/// adapter-owned extras dispatch to their owning adapter through [`stream`].
#[derive(Clone, Debug, Default)]
pub struct LiveOptions {
    /// The api key, overriding credential resolution.
    pub api_key: Option<String>,
    /// Cancellation for the request and its streamed body, upstream's
    /// `signal: AbortSignal`.
    pub signal: Option<CancellationToken>,
    /// Custom headers merged with the provider defaults.
    pub headers: Option<ProviderHeaders>,
    /// Preferred transport for multi-transport providers.
    pub transport: Option<Transport>,
    /// The request-payload hook, upstream's `onPayload`.
    pub on_payload: Option<OnPayload>,
    /// The pi reasoning level, upstream's `reasoningEffort`.
    pub reasoning: Option<ThinkingLevel>,
    /// Custom per-level thinking budgets.
    pub thinking_budgets: Option<ThinkingBudgets>,
    /// Anthropic-family extended thinking, upstream's `thinkingEnabled`.
    pub thinking_enabled: Option<bool>,
    /// Anthropic-family thinking budget, upstream's `thinkingBudgetTokens`.
    pub thinking_budget_tokens: Option<u64>,
    /// Anthropic adaptive-thinking effort, upstream's `effort`.
    pub effort: Option<AnthropicEffort>,
    /// The interleaved-thinking beta, upstream's `interleavedThinking`
    /// (Anthropic and Bedrock).
    pub interleaved_thinking: Option<bool>,
    /// The Azure deployment override, upstream's `azureDeploymentName`.
    pub azure_deployment_name: Option<String>,
    /// The GCP project, upstream's Vertex `project`.
    pub vertex_project: Option<String>,
    /// The GCP region, upstream's Vertex `location`.
    pub vertex_location: Option<String>,
    /// The Google thinking control, upstream's
    /// `thinking: { enabled, budgetTokens?, level? }`.
    pub google_thinking: Option<GoogleThinkingControl>,
    /// Request metadata, upstream's Bedrock `requestMetadata`.
    pub request_metadata: Option<BTreeMap<String, Value>>,
}

impl LiveOptions {
    /// The options a block passes when it only names a credential, upstream's
    /// `{ apiKey }`.
    #[must_use]
    pub fn key(api_key: impl Into<String>) -> Self {
        Self {
            api_key: Some(api_key.into()),
            ..Self::default()
        }
    }

    /// Whether any Anthropic-family thinking knob is set.
    const fn carries_anthropic_extras(&self) -> bool {
        self.thinking_enabled.is_some()
            || self.thinking_budget_tokens.is_some()
            || self.effort.is_some()
            || self.interleaved_thinking.is_some()
    }
}

/// The base options an extras object restates to, upstream's base keys.
fn base_stream_options(options: &LiveOptions) -> StreamOptions {
    StreamOptions {
        transport_options: TransportOptions {
            signal: options.signal.clone(),
            on_payload: options.on_payload.clone(),
            ..TransportOptions::default()
        },
        api_key: options.api_key.clone(),
        headers: options.headers.clone(),
        transport: options.transport,
        ..StreamOptions::default()
    }
}

/// The simple options the reasoning level and thinking budgets ride.
fn simple_stream_options(options: &LiveOptions) -> SimpleStreamOptions {
    SimpleStreamOptions {
        transport_options: TransportOptions {
            signal: options.signal.clone(),
            on_payload: options.on_payload.clone(),
            ..TransportOptions::default()
        },
        api_key: options.api_key.clone(),
        headers: options.headers.clone(),
        transport: options.transport,
        reasoning: options.reasoning,
        thinking_budgets: options.thinking_budgets,
        ..SimpleStreamOptions::default()
    }
}

/// The Anthropic options the extras object restates to.
fn anthropic_options(options: &LiveOptions) -> AnthropicStreamOptions {
    let mut restated = AnthropicStreamOptions::from(base_stream_options(options));
    restated.thinking_enabled = options.thinking_enabled;
    restated.thinking_budget_tokens = options.thinking_budget_tokens;
    restated.effort = options.effort;
    restated.interleaved_thinking = options.interleaved_thinking;
    restated
}

/// The Azure options the extras object restates to, the deployment override
/// riding the adapter field the environment map would otherwise fill.
fn azure_options(options: &LiveOptions) -> AzureOpenAiResponsesOptions {
    let mut restated = AzureOpenAiResponsesOptions::from(base_stream_options(options));
    restated
        .azure_deployment_name
        .clone_from(&options.azure_deployment_name);
    restated
}

/// The Google options the extras object restates to: project and location for
/// Vertex, the thinking control for both backends.
fn google_options(options: &LiveOptions) -> GoogleOptions {
    let mut restated = GoogleOptions::from(base_stream_options(options));
    restated.project.clone_from(&options.vertex_project);
    restated.location.clone_from(&options.vertex_location);
    restated.thinking.clone_from(&options.google_thinking);
    restated
}

/// The Bedrock options the extras object restates to, upstream's
/// `{ reasoning, interleavedThinking, requestMetadata, onPayload }`.
fn bedrock_options(options: &LiveOptions) -> BedrockStreamOptions {
    let mut restated = BedrockStreamOptions::from(base_stream_options(options));
    restated.reasoning = options.reasoning;
    restated.interleaved_thinking = options.interleaved_thinking;
    restated
        .request_metadata
        .clone_from(&options.request_metadata);
    restated
}

/// The streaming dispatch upstream's compat `stream(model, context, options)`
/// duck-types into: adapter-owned extras call the owning adapter, the
/// reasoning level and thinking budgets ride the simple surface, and plain
/// base fields ride compat.
#[must_use]
pub fn stream(
    model: &Model,
    context: &Context,
    options: &LiveOptions,
) -> AssistantMessageEventStream {
    match model.api.as_known() {
        Some(KnownApi::AnthropicMessages) if options.carries_anthropic_extras() => {
            pi_ai::api::anthropic_messages::stream(
                model,
                context,
                Some(&anthropic_options(options)),
            )
        }
        Some(KnownApi::AzureOpenaiResponses) if options.azure_deployment_name.is_some() => {
            pi_ai::api::azure_openai_responses::stream(
                model,
                context,
                Some(&azure_options(options)),
            )
        }
        Some(KnownApi::GoogleGenerativeAi) if options.google_thinking.is_some() => {
            google_generative_ai::stream(model, context, Some(&google_options(options)))
        }
        Some(KnownApi::GoogleVertex)
            if options.google_thinking.is_some()
                || options.vertex_project.is_some()
                || options.vertex_location.is_some() =>
        {
            google_vertex::stream(model, context, Some(&google_options(options)))
        }
        Some(KnownApi::BedrockConverseStream)
            if options.interleaved_thinking.is_some()
                || options.request_metadata.is_some()
                || options.on_payload.is_some() =>
        {
            pi_ai::api::bedrock_converse_stream::stream(
                model,
                context,
                Some(&bedrock_options(options)),
            )
        }
        _ if options.reasoning.is_some() || options.thinking_budgets.is_some() => {
            compat_stream_simple(model, context, Some(&simple_stream_options(options)))
        }
        _ => compat_stream(model, context, Some(&base_stream_options(options))),
    }
}

/// The completion dispatch upstream's compat `complete` restates to: the
/// stream's final assistant message, including error-terminal messages.
pub async fn complete(model: &Model, context: &Context, options: &LiveOptions) -> AssistantMessage {
    stream(model, context, options).result().await
}

/// The user message a probe sends.
#[must_use]
pub fn user_message(text: &str) -> Message {
    Message::User(UserMessage {
        content: UserContent::Text(text.to_owned()),
        timestamp: now_ms(),
    })
}

/// The joined text of an assistant response, upstream's
/// `content.map(b => b.type === "text" ? b.text : "").join("")`.
#[must_use]
pub fn response_text(response: &AssistantMessage) -> String {
    response
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantBlock::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect()
}

/// The first upstream helper, upstream's `basicTextGeneration`: two chained
/// completions, each asserting its usage accounting and the requested phrase.
pub async fn basic_text_generation(model: &Model, options: &LiveOptions) {
    let mut context = Context {
        system_prompt: Some("You are a helpful assistant. Be concise.".to_owned()),
        messages: vec![user_message("Reply with exactly: 'Hello test successful'")],
        tools: None,
    };
    let response = complete(model, &context, options).await;

    assert!(response.usage.input + response.usage.cache_read > 0);
    assert!(response.usage.output > 0);
    assert!(
        response.error_message.is_none(),
        "{}/{}: {:?}",
        model.provider.0,
        model.id,
        response.error_message
    );
    assert!(
        response_text(&response).contains("Hello test successful"),
        "{}/{}: {:?}",
        model.provider.0,
        model.id,
        response_text(&response)
    );

    context.messages.push(Message::Assistant(response));
    context
        .messages
        .push(user_message("Now say 'Goodbye test successful'"));

    let second = complete(model, &context, options).await;

    assert!(second.usage.input + second.usage.cache_read > 0);
    assert!(second.usage.output > 0);
    assert!(
        second.error_message.is_none(),
        "{}/{}: {:?}",
        model.provider.0,
        model.id,
        second.error_message
    );
    assert!(
        response_text(&second).contains("Goodbye test successful"),
        "{}/{}: {:?}",
        model.provider.0,
        model.id,
        response_text(&second)
    );
}

/// The streamed tool-call helper, upstream's `handleToolCall`: the calculator
/// tool must start, stream parsed deltas, and end with the computed sum.
pub async fn handle_tool_call(model: &Model, options: &LiveOptions) {
    let context = Context {
        system_prompt: Some("You are a helpful assistant that uses tools when asked.".to_owned()),
        messages: vec![user_message(
            "Calculate 15 + 27 using the math_operation tool.",
        )],
        tools: Some(vec![calculator_tool()]),
    };

    let mut has_tool_start = false;
    let mut has_tool_delta = false;
    let mut has_tool_end = false;
    let mut accumulated_args = String::new();
    let mut index = 0;
    let stream = stream(model, &context, options);
    while let Some(event) = stream.next().await {
        match event {
            AssistantMessageEvent::ToolcallStart {
                content_index,
                partial,
                ..
            } => {
                has_tool_start = true;
                index = content_index;
                let tool_call = partial
                    .content
                    .get(content_index as usize)
                    .and_then(|block| match block {
                        AssistantBlock::ToolCall(tool_call) => Some(tool_call),
                        _ => None,
                    })
                    .unwrap_or_else(|| panic!("toolcall_start carries the tool call block"));
                assert_eq!(tool_call.name, "math_operation");
                assert!(!tool_call.id.is_empty());
            }
            AssistantMessageEvent::ToolcallDelta {
                content_index,
                delta,
                partial,
                ..
            } => {
                has_tool_delta = true;
                assert_eq!(content_index, index);
                let tool_call = partial
                    .content
                    .get(content_index as usize)
                    .and_then(|block| match block {
                        AssistantBlock::ToolCall(tool_call) => Some(tool_call),
                        _ => None,
                    })
                    .unwrap_or_else(|| panic!("toolcall_delta carries the tool call block"));
                accumulated_args.push_str(&delta);
                assert!(
                    tool_call.arguments.contains_key("operation") || tool_call.arguments.is_empty()
                );
            }
            AssistantMessageEvent::ToolcallEnd {
                content_index,
                tool_call,
                ..
            } => {
                has_tool_end = true;
                assert_eq!(content_index, index);
                assert_eq!(tool_call.name, "math_operation");
                let parsed: Value =
                    serde_json::from_str(&accumulated_args).expect("the streamed deltas parse");
                assert_eq!(parsed["a"], json!(15));
                assert_eq!(parsed["b"], json!(27));
                assert!(
                    ["add", "subtract", "multiply", "divide"]
                        .contains(&parsed["operation"].as_str().unwrap_or_default()),
                    "the operation is one of the four"
                );
            }
            _ => {}
        }
    }

    assert!(has_tool_start, "toolcall_start fired");
    assert!(has_tool_delta, "toolcall_delta fired");
    assert!(has_tool_end, "toolcall_end fired");

    let response = stream.result().await;
    assert_eq!(response.stop_reason, StopReason::ToolUse);
    let tool_call = response
        .content
        .iter()
        .find_map(|block| match block {
            AssistantBlock::ToolCall(tool_call) => Some(tool_call),
            _ => None,
        })
        .expect("a tool call block in the final content");
    assert_eq!(tool_call.name, "math_operation");
    assert!(!tool_call.id.is_empty());
}

/// The plain streaming helper, upstream's `handleStreaming`: text starts,
/// accumulates, and closes.
pub async fn handle_streaming(model: &Model, options: &LiveOptions) {
    let mut text_started = false;
    let mut text_chunks = String::new();
    let mut text_completed = false;

    let context = Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![user_message("Count from 1 to 3")],
        tools: None,
    };

    let stream = stream(model, &context, options);
    while let Some(event) = stream.next().await {
        match event {
            AssistantMessageEvent::TextStart { .. } => text_started = true,
            AssistantMessageEvent::TextDelta { delta, .. } => {
                text_chunks.push_str(&delta);
            }
            AssistantMessageEvent::TextEnd { .. } => text_completed = true,
            _ => {}
        }
    }

    let response = stream.result().await;

    assert!(text_started, "text_start fired");
    assert!(!text_chunks.is_empty(), "text deltas accumulated");
    assert!(text_completed, "text_end fired");
    assert!(
        response
            .content
            .iter()
            .any(|block| matches!(block, AssistantBlock::Text(_))),
        "the final content carries text"
    );
}

/// The thinking helper, upstream's `handleThinking`: thinking starts,
/// accumulates, and closes, ending in a clean stop.
pub async fn handle_thinking(model: &Model, options: &LiveOptions) {
    let mut thinking_started = false;
    let mut thinking_chunks = String::new();
    let mut thinking_completed = false;

    let operand = next_arithmetic_operand();
    let context = Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![user_message(&format!(
            "Think long and hard about {operand} + 27. Think step by step. Then output the result."
        ))],
        tools: None,
    };

    let stream = stream(model, &context, options);
    while let Some(event) = stream.next().await {
        match event {
            AssistantMessageEvent::ThinkingStart { .. } => thinking_started = true,
            AssistantMessageEvent::ThinkingDelta { delta, .. } => {
                thinking_chunks.push_str(&delta);
            }
            AssistantMessageEvent::ThinkingEnd { .. } => thinking_completed = true,
            _ => {}
        }
    }

    let response = stream.result().await;

    assert_eq!(
        response.stop_reason,
        StopReason::Stop,
        "error: {:?}",
        response.error_message
    );
    assert!(thinking_started, "thinking_start fired");
    assert!(!thinking_chunks.is_empty(), "thinking deltas accumulated");
    assert!(thinking_completed, "thinking_end fired");
    assert!(
        response
            .content
            .iter()
            .any(|block| matches!(block, AssistantBlock::Thinking(_))),
        "the final content carries thinking"
    );
}

/// The image helper, upstream's `handleImage`: models without image input
/// skip; models with it describe the red circle.
pub async fn handle_image(model: &Model, options: &LiveOptions) {
    if !model.input.contains(&Modality::Image) {
        return;
    }

    let image_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("red-circle.png");
    let image_bytes = std::fs::read(&image_path).expect("the red-circle fixture reads");
    let image = ImageContent {
        data: base64_encode(&image_bytes),
        mime_type: "image/png".to_owned(),
    };

    let context = Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![Message::User(UserMessage {
            content: UserContent::Blocks(vec![
                UserBlock::Text(TextContent {
                    text: "What do you see in this image? Please describe the shape (circle, \
                           rectangle, square, triangle, ...) and color (red, blue, green, ...). \
                           You MUST reply in English."
                        .to_owned(),
                    text_signature: None,
                }),
                UserBlock::Image(image),
            ]),
            timestamp: now_ms(),
        })],
        tools: None,
    };

    let response = complete(model, &context, options).await;

    let text = response_text(&response);
    assert!(!text.is_empty(), "the response describes the image");
    let lowered = text.to_lowercase();
    assert!(
        lowered.contains("red"),
        "the description names the color: {lowered}"
    );
    assert!(
        lowered.contains("circle"),
        "the description names the shape: {lowered}"
    );
}

/// The multi-turn helper, upstream's `multiTurn`: tool calls are answered
/// until a clean stop, and both computed values surface in the text.
pub async fn multi_turn(model: &Model, options: &LiveOptions) {
    let mut context = Context {
        system_prompt: Some(
            "You are a helpful assistant that can use tools to answer questions.".to_owned(),
        ),
        messages: vec![user_message(
            "Think about this briefly, then calculate 42 * 17 and 453 + 434 using the \
             math_operation tool.",
        )],
        tools: Some(vec![calculator_tool()]),
    };

    let mut all_text = String::new();
    let mut has_seen_thinking = false;
    let mut has_seen_tool_calls = false;
    let max_turns = 5;

    for _ in 0..max_turns {
        let response = complete(model, &context, options).await;
        context.messages.push(Message::Assistant(response.clone()));

        let mut results: Vec<Message> = Vec::new();
        for block in &response.content {
            match block {
                AssistantBlock::Text(text) => all_text.push_str(&text.text),
                AssistantBlock::Thinking(_) => has_seen_thinking = true,
                AssistantBlock::ToolCall(tool_call) => {
                    has_seen_tool_calls = true;
                    assert_eq!(tool_call.name, "math_operation");
                    assert!(!tool_call.id.is_empty());

                    let a = tool_call.arguments["a"].as_f64().unwrap_or_default();
                    let b = tool_call.arguments["b"].as_f64().unwrap_or_default();
                    let result = match tool_call.arguments["operation"].as_str() {
                        Some("add") => a + b,
                        Some("multiply") => a * b,
                        _ => 0.0,
                    };
                    let text = if result.fract() == 0.0 {
                        format!("{}", result as u64)
                    } else {
                        format!("{result}")
                    };
                    results.push(Message::ToolResult(ToolResultMessage {
                        tool_call_id: tool_call.id.clone(),
                        tool_name: tool_call.name.clone(),
                        content: vec![ToolResultBlock::Text(TextContent {
                            text,
                            text_signature: None,
                        })],
                        added_tool_names: None,
                        details: None,
                        is_error: false,
                        usage: None,
                        timestamp: now_ms(),
                    }));
                }
            }
        }
        context.messages.extend(results);

        assert_ne!(
            response.stop_reason,
            StopReason::Error,
            "error: {:?}",
            response.error_message
        );
        if response.stop_reason == StopReason::Stop {
            break;
        }
    }

    assert!(
        has_seen_thinking || has_seen_tool_calls,
        "thinking content or tool calls appeared"
    );
    assert!(all_text.contains("714"), "42 * 17 computed: {all_text}");
    assert!(all_text.contains("887"), "453 + 434 computed: {all_text}");
}

/// The empty-content-array helper, upstream's `testEmptyMessage`: the
/// provider handles the empty array or reports a defined error.
pub async fn test_empty_message(model: &Model, options: &LiveOptions) {
    let context = Context {
        system_prompt: None,
        messages: vec![Message::User(UserMessage {
            content: UserContent::Blocks(vec![]),
            timestamp: now_ms(),
        })],
        tools: None,
    };
    let response = complete(model, &context, options).await;
    if response.stop_reason == StopReason::Error {
        assert!(response.error_message.is_some());
    }
}

/// The empty-string helper, upstream's `testEmptyStringMessage`.
pub async fn test_empty_string_message(model: &Model, options: &LiveOptions) {
    let context = Context {
        system_prompt: None,
        messages: vec![user_message("")],
        tools: None,
    };
    let response = complete(model, &context, options).await;
    if response.stop_reason == StopReason::Error {
        assert!(response.error_message.is_some());
    }
}

/// The whitespace-only helper, upstream's `testWhitespaceOnlyMessage`.
pub async fn test_whitespace_only_message(model: &Model, options: &LiveOptions) {
    let context = Context {
        system_prompt: None,
        messages: vec![user_message("   \n\t  ")],
        tools: None,
    };
    let response = complete(model, &context, options).await;
    if response.stop_reason == StopReason::Error {
        assert!(response.error_message.is_some());
    }
}

/// The empty-assistant helper, upstream's `testEmptyAssistantMessage`: user,
/// empty assistant, user rides the round trip.
pub async fn test_empty_assistant_message(model: &Model, options: &LiveOptions) {
    let empty_assistant = AssistantMessage {
        content: vec![],
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        usage: Usage {
            input: 10,
            total_tokens: 10,
            ..Usage::default()
        },
        stop_reason: StopReason::Stop,
        timestamp: now_ms(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
    };

    let context = Context {
        system_prompt: None,
        messages: vec![
            user_message("Hello, how are you?"),
            Message::Assistant(empty_assistant),
            user_message("Please respond this time."),
        ],
        tools: None,
    };
    let response = complete(model, &context, options).await;
    if response.stop_reason == StopReason::Error {
        assert!(response.error_message.is_some());
    } else {
        assert!(!response.content.is_empty());
    }
}

/// The mid-stream abort helper, upstream's `testAbortSignal`: abort once 50
/// characters have streamed, then complete a follow-up over the aborted
/// message.
pub async fn test_abort_signal(model: &Model, options: &LiveOptions) {
    let mut context = Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![user_message(
            "What is 15 + 27? Think step by step. Then list 50 first names.",
        )],
        tools: None,
    };

    let mut abort_fired = false;
    let mut text = String::new();
    let cancellation = CancellationToken::new();
    let mut stream_options = options.clone();
    stream_options.signal = Some(cancellation.clone());
    let stream = stream(model, &context, &stream_options);
    // The result waiter registers before the drain: the stream's terminal
    // push settles the message and `end(None)` clears the stored copy, so a
    // drain-then-result sequence would wait on a result nothing will send.
    let ((), aborted) = tokio::join!(
        async {
            while let Some(event) = stream.next().await {
                if abort_fired {
                    break;
                }
                if let AssistantMessageEvent::TextDelta { delta, .. }
                | AssistantMessageEvent::ThinkingDelta { delta, .. } = &event
                {
                    text.push_str(delta);
                }
                if text.chars().count() >= 50 {
                    cancellation.cancel();
                    abort_fired = true;
                }
            }
        },
        stream.result()
    );

    assert_eq!(aborted.stop_reason, StopReason::Aborted);
    assert!(!aborted.content.is_empty());

    context.messages.push(Message::Assistant(aborted));
    context
        .messages
        .push(user_message("Please continue, but only generate 5 names."));

    let follow_up = complete(model, &context, options).await;
    assert_eq!(follow_up.stop_reason, StopReason::Stop);
    assert!(!follow_up.content.is_empty());
}

/// The pre-aborted helper, upstream's `testImmediateAbort`.
pub async fn test_immediate_abort(model: &Model, options: &LiveOptions) {
    let cancellation = CancellationToken::new();
    cancellation.cancel();

    let context = Context {
        system_prompt: None,
        messages: vec![user_message("Hello")],
        tools: None,
    };
    let mut stream_options = options.clone();
    stream_options.signal = Some(cancellation);
    let response = complete(model, &context, &stream_options).await;
    assert_eq!(response.stop_reason, StopReason::Aborted);
}

/// The abort-then-continue helper, upstream's `testAbortThenNewMessage`: the
/// aborted empty assistant rides in context and the follow-up still answers.
pub async fn test_abort_then_new_message(model: &Model, options: &LiveOptions) {
    let cancellation = CancellationToken::new();
    cancellation.cancel();

    let mut context = Context {
        system_prompt: None,
        messages: vec![user_message("Hello, how are you?")],
        tools: None,
    };

    let mut stream_options = options.clone();
    stream_options.signal = Some(cancellation);
    let aborted = complete(model, &context, &stream_options).await;
    assert_eq!(aborted.stop_reason, StopReason::Aborted);
    assert!(aborted.content.is_empty());

    context.messages.push(Message::Assistant(aborted));
    context.messages.push(user_message("What is 2 + 2?"));

    let follow_up = complete(model, &context, options).await;
    assert_eq!(follow_up.stop_reason, StopReason::Stop);
    assert!(!follow_up.content.is_empty());
}

/// The token-accounting-on-abort helper, upstream's `testTokensOnAbort`:
/// abort at 1000 characters, then the usage split per API class.
pub async fn test_tokens_on_abort(model: &Model, options: &LiveOptions) {
    let context = Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![user_message(
            "Write a long poem with 20 stanzas about the beauty of nature.",
        )],
        tools: None,
    };

    let cancellation = CancellationToken::new();
    let mut stream_options = options.clone();
    stream_options.signal = Some(cancellation.clone());

    let mut abort_fired = false;
    let mut text = String::new();
    let stream = stream(model, &context, &stream_options);
    // The result waiter registers before the drain: the stream's terminal
    // push settles the message and `end(None)` clears the stored copy, so a
    // drain-then-result sequence would wait on a result nothing will send.
    let ((), aborted) = tokio::join!(
        async {
            while let Some(event) = stream.next().await {
                if !abort_fired
                    && let AssistantMessageEvent::TextDelta { delta, .. }
                    | AssistantMessageEvent::ThinkingDelta { delta, .. } = &event
                {
                    text.push_str(delta);
                    if text.chars().count() >= 1000 {
                        abort_fired = true;
                        cancellation.cancel();
                    }
                }
            }
        },
        stream.result()
    );

    assert_eq!(aborted.stop_reason, StopReason::Aborted);

    // OpenAI-family APIs, z.ai, Bedrock, and the gateway only send usage in
    // the final chunk, so aborted requests carry none. MiniMax does not report
    // usage for aborted requests; Kimi reports input early but output only in
    // the final chunk.
    let zero_usage = matches!(
        model.api.as_known(),
        Some(
            KnownApi::OpenaiCompletions
                | KnownApi::MistralConversations
                | KnownApi::OpenaiResponses
                | KnownApi::AzureOpenaiResponses
                | KnownApi::OpenaiCodexResponses
        )
    ) || matches!(
        model.provider.0.as_str(),
        "zai" | "amazon-bedrock" | "vercel-ai-gateway"
    );
    if zero_usage || model.provider.0 == "minimax" {
        assert_eq!(aborted.usage.input, 0);
        assert_eq!(aborted.usage.output, 0);
    } else if model.provider.0 == "kimi-coding" {
        assert!(aborted.usage.input > 0);
        assert_eq!(aborted.usage.output, 0);
    } else {
        assert!(aborted.usage.input > 0);
        assert!(aborted.usage.output > 0);
        if model.cost.rates.input > 0.0 {
            assert!(aborted.usage.cost.input > 0.0);
            assert!(aborted.usage.cost.total > 0.0);
        }
    }
}

/// The long system prompt that triggers caching, upstream's
/// `LONG_SYSTEM_PROMPT`.
#[must_use]
pub fn long_system_prompt() -> String {
    let filler = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor \
         incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud \
         exercitation ullamco laboris.";
    let mut prompt = String::from(
        "You are a helpful assistant. Be concise in your responses.\n\nHere is some additional \
         context that makes this system prompt long enough to trigger caching:\n\n",
    );
    for _ in 0..50 {
        prompt.push_str(filler);
        prompt.push_str("\n\n");
    }
    prompt.push_str("Remember: Always be helpful and concise.");
    prompt
}

/// The two cache-primed completions, upstream's `testTotalTokensWithCache`:
/// the first request primes, the second rides the same system prompt, both
/// must stop.
pub async fn test_total_tokens_with_cache(model: &Model, options: &LiveOptions) -> (Usage, Usage) {
    let system_prompt = long_system_prompt();
    let first_context = Context {
        system_prompt: Some(system_prompt.clone()),
        messages: vec![user_message("What is 2 + 2? Reply with just the number.")],
        tools: None,
    };
    let first = complete(model, &first_context, options).await;
    assert_eq!(first.stop_reason, StopReason::Stop);

    let mut messages = first_context.messages.clone();
    messages.push(Message::Assistant(first.clone()));
    messages.push(user_message("What is 3 + 3? Reply with just the number."));
    let second_context = Context {
        system_prompt: Some(system_prompt),
        messages,
        tools: None,
    };
    let second = complete(model, &second_context, options).await;
    assert_eq!(second.stop_reason, StopReason::Stop);

    (first.usage, second.usage)
}

/// `totalTokens` must equal the sum of its components, upstream's
/// `assertTotalTokensEqualsComponents`.
pub fn assert_total_tokens_equals_components(usage: &Usage) {
    assert_eq!(
        usage.total_tokens,
        usage.input + usage.output + usage.cache_read + usage.cache_write
    );
}

/// The lorem paragraph the overflow probes repeat, upstream's `LOREM_IPSUM`.
const LOREM_IPSUM: &str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do \
     eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis \
     nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute \
     irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla \
     pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia \
     deserunt mollit anim id est laborum. ";

/// A string exceeding the context window by 10k estimated tokens, upstream's
/// `generateOverflowContent` over the chars/4 estimate of varied text.
#[must_use]
pub fn generate_overflow_content(context_window: u64) -> String {
    let target_tokens = context_window + 10_000;
    let target_chars = f64::from(u32::try_from(target_tokens).unwrap_or(u32::MAX)) * 4.0 * 1.5;
    let repetitions = (target_chars / LOREM_IPSUM.len() as f64).ceil() as usize;
    LOREM_IPSUM.repeat(repetitions.max(1))
}

/// The oversized completion, upstream's `testContextOverflow`: whether usage
/// data arrived rides with the message.
pub async fn complete_overflow(model: &Model, api_key: &str) -> (AssistantMessage, bool) {
    let context = Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![user_message(&generate_overflow_content(
            model.context_window,
        ))],
        tools: None,
    };
    let options = LiveOptions::key(api_key);
    let response = complete(model, &context, &options).await;
    let has_usage_data = response.usage.input > 0 || response.usage.cache_read > 0;
    (response, has_usage_data)
}

/// The standard base64 alphabet the image fixture rides in, upstream's
/// `imageBuffer.toString("base64")`.
#[must_use]
pub fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        encoded.push(ALPHABET[(n >> 18) as usize & 63] as char);
        encoded.push(ALPHABET[(n >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            encoded.push(ALPHABET[(n >> 6) as usize & 63] as char);
        } else {
            encoded.push('=');
        }
        if chunk.len() > 2 {
            encoded.push(ALPHABET[n as usize & 63] as char);
        } else {
            encoded.push('=');
        }
    }
    encoded
}
