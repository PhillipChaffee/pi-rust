//! The unified LLM data model, ported from `packages/ai/src/types.ts` at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Every struct here mirrors an upstream interface one to one; doc comments
//! carry the wire's field names (`camelCase`) and the defaults the other side
//! assumes. The porting restatements are recorded at the crate root.
//!
//! [`AssistantMessageDiagnostic`] lives here (it is an [`AssistantMessage`]
//! field type); the diagnostics helpers port with the utils belt.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use pi_telemetry::TelemetryHandle;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::http::client::HttpClient;

/// A JSON value; upstream's recursive [`JsonValue`] type alias.
pub type JsonValue = serde_json::Value;

/// The ten wire-protocol implementations upstream ships, spelled as the wire
/// carries them. Custom APIs are arbitrary strings — see [`Api`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KnownApi {
    /// The OpenAI Chat Completions protocol.
    OpenaiCompletions,
    /// The Mistral Conversations protocol.
    MistralConversations,
    /// The OpenAI Responses protocol.
    OpenaiResponses,
    /// Azure's OpenAI Responses deployment.
    AzureOpenaiResponses,
    /// The OpenAI Codex Responses protocol (SSE and WebSocket transports).
    OpenaiCodexResponses,
    /// The Anthropic Messages protocol.
    AnthropicMessages,
    /// The Amazon Bedrock Converse Stream protocol.
    BedrockConverseStream,
    /// The Google Generative AI (Gemini) protocol.
    GoogleGenerativeAi,
    /// The Google Vertex AI protocol.
    GoogleVertex,
    /// pi's own messages protocol.
    PiMessages,
}

impl TryFrom<&str> for KnownApi {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "openai-completions" => Ok(Self::OpenaiCompletions),
            "mistral-conversations" => Ok(Self::MistralConversations),
            "openai-responses" => Ok(Self::OpenaiResponses),
            "azure-openai-responses" => Ok(Self::AzureOpenaiResponses),
            "openai-codex-responses" => Ok(Self::OpenaiCodexResponses),
            "anthropic-messages" => Ok(Self::AnthropicMessages),
            "bedrock-converse-stream" => Ok(Self::BedrockConverseStream),
            "google-generative-ai" => Ok(Self::GoogleGenerativeAi),
            "google-vertex" => Ok(Self::GoogleVertex),
            "pi-messages" => Ok(Self::PiMessages),
            other => Err(other.to_string()),
        }
    }
}

impl std::fmt::Display for KnownApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let wire = match self {
            Self::OpenaiCompletions => "openai-completions",
            Self::MistralConversations => "mistral-conversations",
            Self::OpenaiResponses => "openai-responses",
            Self::AzureOpenaiResponses => "azure-openai-responses",
            Self::OpenaiCodexResponses => "openai-codex-responses",
            Self::AnthropicMessages => "anthropic-messages",
            Self::BedrockConverseStream => "bedrock-converse-stream",
            Self::GoogleGenerativeAi => "google-generative-ai",
            Self::GoogleVertex => "google-vertex",
            Self::PiMessages => "pi-messages",
        };
        f.write_str(wire)
    }
}

/// An API identifier: a [`KnownApi`] or any custom string, upstream's
/// `KnownApi | (string & {})`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Api(pub String);

impl Api {
    /// The known protocol this id names, when it is one.
    #[must_use]
    pub fn as_known(&self) -> Option<KnownApi> {
        KnownApi::try_from(self.0.as_str()).ok()
    }

    /// Whether this id names one of the ten known protocols.
    #[must_use]
    pub fn is_known(&self) -> bool {
        self.as_known().is_some()
    }
}

impl From<KnownApi> for Api {
    fn from(value: KnownApi) -> Self {
        Self(value.to_string())
    }
}

impl From<&str> for Api {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for Api {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl std::fmt::Display for Api {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::ops::Deref for Api {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// The one image-generation protocol upstream ships at the pin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KnownImagesApi {
    /// The OpenRouter image-generation protocol.
    OpenrouterImages,
}

impl std::fmt::Display for KnownImagesApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let wire = match self {
            Self::OpenrouterImages => "openrouter-images",
        };
        f.write_str(wire)
    }
}

/// An image API identifier: [`KnownImagesApi`] or any custom string.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ImagesApi(pub String);

impl From<KnownImagesApi> for ImagesApi {
    fn from(value: KnownImagesApi) -> Self {
        Self(value.to_string())
    }
}

impl From<&str> for ImagesApi {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for ImagesApi {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl std::ops::Deref for ImagesApi {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// The 39 providers upstream knows by id at the pin. Custom providers are
/// arbitrary strings — see [`ProviderId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KnownProvider {
    /// Amazon Bedrock.
    AmazonBedrock,
    /// Ant Ling.
    AntLing,
    /// Anthropic.
    Anthropic,
    /// Google AI Studio.
    Google,
    /// Google Vertex AI.
    GoogleVertex,
    /// OpenAI.
    Openai,
    /// Azure OpenAI Responses.
    AzureOpenaiResponses,
    /// OpenAI Codex.
    OpenaiCodex,
    /// Radius.
    Radius,
    /// NVIDIA.
    Nvidia,
    /// DeepSeek.
    Deepseek,
    /// GitHub Copilot.
    GithubCopilot,
    /// xAI.
    Xai,
    /// Groq.
    Groq,
    /// Cerebras.
    Cerebras,
    /// OpenRouter.
    Openrouter,
    /// Vercel AI Gateway.
    VercelAiGateway,
    /// Z.ai.
    Zai,
    /// Z.ai coding plan (China).
    ZaiCodingCn,
    /// Mistral.
    Mistral,
    /// MiniMax.
    Minimax,
    /// MiniMax (China).
    MinimaxCn,
    /// Moonshot AI.
    Moonshotai,
    /// Moonshot AI (China).
    MoonshotaiCn,
    /// Hugging Face.
    Huggingface,
    /// Fireworks.
    Fireworks,
    /// Together.
    Together,
    /// Baseten.
    Baseten,
    /// OpenCode.
    Opencode,
    /// OpenCode Go.
    OpencodeGo,
    /// Kimi Coding.
    KimiCoding,
    /// Cloudflare Workers AI.
    CloudflareWorkersAi,
    /// Cloudflare AI Gateway.
    CloudflareAiGateway,
    /// Qwen token plan.
    QwenTokenPlan,
    /// Qwen token plan (China).
    QwenTokenPlanCn,
    /// Qwen token plan (individual).
    QwenTokenPlanIndividual,
    /// Xiaomi.
    Xiaomi,
    /// Xiaomi token plan (China).
    XiaomiTokenPlanCn,
    /// Xiaomi token plan (Amsterdam).
    XiaomiTokenPlanAms,
    /// Xiaomi token plan (Singapore).
    XiaomiTokenPlanSgp,
}

impl TryFrom<&str> for KnownProvider {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "amazon-bedrock" => Ok(Self::AmazonBedrock),
            "ant-ling" => Ok(Self::AntLing),
            "anthropic" => Ok(Self::Anthropic),
            "google" => Ok(Self::Google),
            "google-vertex" => Ok(Self::GoogleVertex),
            "openai" => Ok(Self::Openai),
            "azure-openai-responses" => Ok(Self::AzureOpenaiResponses),
            "openai-codex" => Ok(Self::OpenaiCodex),
            "radius" => Ok(Self::Radius),
            "nvidia" => Ok(Self::Nvidia),
            "deepseek" => Ok(Self::Deepseek),
            "github-copilot" => Ok(Self::GithubCopilot),
            "xai" => Ok(Self::Xai),
            "groq" => Ok(Self::Groq),
            "cerebras" => Ok(Self::Cerebras),
            "openrouter" => Ok(Self::Openrouter),
            "vercel-ai-gateway" => Ok(Self::VercelAiGateway),
            "zai" => Ok(Self::Zai),
            "zai-coding-cn" => Ok(Self::ZaiCodingCn),
            "mistral" => Ok(Self::Mistral),
            "minimax" => Ok(Self::Minimax),
            "minimax-cn" => Ok(Self::MinimaxCn),
            "moonshotai" => Ok(Self::Moonshotai),
            "moonshotai-cn" => Ok(Self::MoonshotaiCn),
            "huggingface" => Ok(Self::Huggingface),
            "fireworks" => Ok(Self::Fireworks),
            "together" => Ok(Self::Together),
            "baseten" => Ok(Self::Baseten),
            "opencode" => Ok(Self::Opencode),
            "opencode-go" => Ok(Self::OpencodeGo),
            "kimi-coding" => Ok(Self::KimiCoding),
            "cloudflare-workers-ai" => Ok(Self::CloudflareWorkersAi),
            "cloudflare-ai-gateway" => Ok(Self::CloudflareAiGateway),
            "qwen-token-plan" => Ok(Self::QwenTokenPlan),
            "qwen-token-plan-cn" => Ok(Self::QwenTokenPlanCn),
            "qwen-token-plan-individual" => Ok(Self::QwenTokenPlanIndividual),
            "xiaomi" => Ok(Self::Xiaomi),
            "xiaomi-token-plan-cn" => Ok(Self::XiaomiTokenPlanCn),
            "xiaomi-token-plan-ams" => Ok(Self::XiaomiTokenPlanAms),
            "xiaomi-token-plan-sgp" => Ok(Self::XiaomiTokenPlanSgp),
            other => Err(other.to_string()),
        }
    }
}

impl std::fmt::Display for KnownProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let wire = match self {
            Self::AmazonBedrock => "amazon-bedrock",
            Self::AntLing => "ant-ling",
            Self::Anthropic => "anthropic",
            Self::Google => "google",
            Self::GoogleVertex => "google-vertex",
            Self::Openai => "openai",
            Self::AzureOpenaiResponses => "azure-openai-responses",
            Self::OpenaiCodex => "openai-codex",
            Self::Radius => "radius",
            Self::Nvidia => "nvidia",
            Self::Deepseek => "deepseek",
            Self::GithubCopilot => "github-copilot",
            Self::Xai => "xai",
            Self::Groq => "groq",
            Self::Cerebras => "cerebras",
            Self::Openrouter => "openrouter",
            Self::VercelAiGateway => "vercel-ai-gateway",
            Self::Zai => "zai",
            Self::ZaiCodingCn => "zai-coding-cn",
            Self::Mistral => "mistral",
            Self::Minimax => "minimax",
            Self::MinimaxCn => "minimax-cn",
            Self::Moonshotai => "moonshotai",
            Self::MoonshotaiCn => "moonshotai-cn",
            Self::Huggingface => "huggingface",
            Self::Fireworks => "fireworks",
            Self::Together => "together",
            Self::Baseten => "baseten",
            Self::Opencode => "opencode",
            Self::OpencodeGo => "opencode-go",
            Self::KimiCoding => "kimi-coding",
            Self::CloudflareWorkersAi => "cloudflare-workers-ai",
            Self::CloudflareAiGateway => "cloudflare-ai-gateway",
            Self::QwenTokenPlan => "qwen-token-plan",
            Self::QwenTokenPlanCn => "qwen-token-plan-cn",
            Self::QwenTokenPlanIndividual => "qwen-token-plan-individual",
            Self::Xiaomi => "xiaomi",
            Self::XiaomiTokenPlanCn => "xiaomi-token-plan-cn",
            Self::XiaomiTokenPlanAms => "xiaomi-token-plan-ams",
            Self::XiaomiTokenPlanSgp => "xiaomi-token-plan-sgp",
        };
        f.write_str(wire)
    }
}

/// A provider identifier: a [`KnownProvider`] or any custom string, upstream's
/// `ProviderId = KnownProvider | string`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProviderId(pub String);

impl ProviderId {
    /// The known provider this id names, when it is one.
    #[must_use]
    pub fn as_known(&self) -> Option<KnownProvider> {
        KnownProvider::try_from(self.0.as_str()).ok()
    }

    /// Whether this id names one of the known providers.
    #[must_use]
    pub fn is_known(&self) -> bool {
        self.as_known().is_some()
    }
}

impl From<KnownProvider> for ProviderId {
    fn from(value: KnownProvider) -> Self {
        Self(value.to_string())
    }
}

impl From<&str> for ProviderId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for ProviderId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl std::ops::Deref for ProviderId {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// The one image-generation provider upstream ships at the pin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KnownImagesProvider {
    /// OpenRouter.
    Openrouter,
}

impl std::fmt::Display for KnownImagesProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let wire = match self {
            Self::Openrouter => "openrouter",
        };
        f.write_str(wire)
    }
}

/// An image provider identifier: [`KnownImagesProvider`] or any custom string.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ImagesProviderId(pub String);

impl From<KnownImagesProvider> for ImagesProviderId {
    fn from(value: KnownImagesProvider) -> Self {
        Self(value.to_string())
    }
}

impl From<&str> for ImagesProviderId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for ImagesProviderId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl std::ops::Deref for ImagesProviderId {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Provider-neutral tool selection for simple requests. When omitted, adapters
/// use provider-specific behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolChoice {
    /// Let the provider decide.
    #[serde(rename = "auto")]
    Auto,
    /// Never call tools.
    #[serde(rename = "none")]
    None,
}

/// The pi thinking levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    /// Minimal thinking.
    Minimal,
    /// Low thinking.
    Low,
    /// Medium thinking.
    Medium,
    /// High thinking.
    High,
    /// Extra-high thinking.
    #[serde(rename = "xhigh")]
    Xhigh,
    /// Maximum thinking.
    Max,
}

/// The model thinking levels: [`ThinkingLevel`] plus the explicit off state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ModelThinkingLevel {
    /// Thinking off.
    #[serde(rename = "off")]
    Off,
    /// Minimal thinking.
    #[serde(rename = "minimal")]
    Minimal,
    /// Low thinking.
    #[serde(rename = "low")]
    Low,
    /// Medium thinking.
    #[serde(rename = "medium")]
    Medium,
    /// High thinking.
    #[serde(rename = "high")]
    High,
    /// Extra-high thinking.
    #[serde(rename = "xhigh")]
    Xhigh,
    /// Maximum thinking.
    #[serde(rename = "max")]
    Max,
}

/// Maps pi thinking levels to provider/model-specific values, upstream's
/// `ThinkingLevelMap = Partial<Record<ModelThinkingLevel, string | null>>`.
///
/// A missing key uses provider defaults; a `None` value (the wire's `null`)
/// marks a level as unsupported.
pub type ThinkingLevelMap = BTreeMap<ModelThinkingLevel, Option<String>>;

/// The thinking-control variable a chat template kwarg substitutes, upstream's
/// `$var` discriminator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinkingTemplateVar {
    /// `thinking.enabled`.
    #[serde(rename = "thinking.enabled")]
    ThinkingEnabled,
    /// `thinking.effort`.
    #[serde(rename = "thinking.effort")]
    ThinkingEffort,
    /// `thinking.budget`.
    #[serde(rename = "thinking.budget")]
    ThinkingBudget,
}

/// The `$var` object form of a chat template kwarg value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatTemplateVar {
    /// Which pi-controlled thinking value this kwarg substitutes.
    #[serde(rename = "$var")]
    pub var: ThinkingTemplateVar,
    /// Whether to omit the kwarg entirely when thinking is off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub omit_when_off: Option<bool>,
}

/// One chat-template kwarg value: a scalar or a `$var` object, upstream's
/// `ChatTemplateKwargValue`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatTemplateKwargValue {
    /// A `$var` substitution object.
    Template(ChatTemplateVar),
    /// A plain string value.
    Str(String),
    /// A plain number value.
    Number(serde_json::Number),
    /// A plain boolean value.
    Bool(bool),
    /// The wire's `null`.
    Null,
}

/// Top-level request field used to cap reasoning tokens on OpenAI-compatible
/// servers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinkingTokenBudgetField {
    /// vLLM's field.
    #[serde(rename = "thinking_token_budget")]
    ThinkingTokenBudget,
    /// Qwen/DashScope/SGLang's field.
    #[serde(rename = "thinking_budget")]
    ThinkingBudget,
    /// llama.cpp's field.
    #[serde(rename = "thinking_budget_tokens")]
    ThinkingBudgetTokens,
}

/// Token budgets for each thinking level, token-based providers only. Levels
/// left unset use the provider default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingBudgets {
    /// Budget for the minimal level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimal: Option<u64>,
    /// Budget for the low level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub low: Option<u64>,
    /// Budget for the medium level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub medium: Option<u64>,
    /// Budget for the high level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub high: Option<u64>,
}

/// Prompt-cache retention preference; providers map this to their supported
/// values. Default on requests: `"short"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheRetention {
    /// No caching.
    #[serde(rename = "none")]
    None,
    /// Short retention.
    #[serde(rename = "short")]
    Short,
    /// Long retention.
    #[serde(rename = "long")]
    Long,
}

/// Preferred transport for providers that support multiple transports;
/// providers that do not support the option ignore it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Transport {
    /// Server-sent events over HTTP.
    #[serde(rename = "sse")]
    Sse,
    /// A WebSocket connection.
    #[serde(rename = "websocket")]
    Websocket,
    /// A cached WebSocket probe before connecting.
    #[serde(rename = "websocket-cached")]
    WebsocketCached,
    /// Let the adapter choose.
    #[serde(rename = "auto")]
    Auto,
}

/// Provider-scoped environment overrides. Values take precedence over the
/// process environment.
pub type ProviderEnv = BTreeMap<String, String>;

/// Custom HTTP headers merged over provider defaults; caller values override
/// default headers. A `None` value (the wire's `null`) suppresses a
/// provider/API default header with the same name.
pub type ProviderHeaders = BTreeMap<String, Option<String>>;

/// Session-affinity header format, upstream's `SessionAffinityFormat`.
///
/// `"openai"` sends `session_id`, `x-client-request-id`, and
/// `x-session-affinity`; `"openai-nosession"` sends `x-client-request-id` and
/// `x-session-affinity`; `"openrouter"` sends `x-session-id`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionAffinityFormat {
    /// OpenAI's three-header format.
    #[serde(rename = "openai")]
    Openai,
    /// OpenAI without the `session_id` header.
    #[serde(rename = "openai-nosession")]
    OpenaiNosession,
    /// OpenRouter's `x-session-id` format.
    #[serde(rename = "openrouter")]
    Openrouter,
}

/// What the transport layer observed for one provider response, handed to
/// response callbacks.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderResponse {
    /// The HTTP status code.
    pub status: u16,
    /// The response headers.
    pub headers: BTreeMap<String, String>,
}

/// The request-payload hook, upstream's `onPayload`
/// (`packages/ai/src/types.ts:145`).
///
/// Receives the payload an adapter is about to send and the model it is for;
/// `Some` replaces the payload, `None` keeps it unchanged. Adapters that
/// cannot re-bind a payload reject it instead of silently bypassing it.
#[derive(Clone)]
pub struct OnPayload(
    Arc<dyn Fn(JsonValue, Model) -> BoxedFuture<'static, Option<JsonValue>> + Send + Sync>,
);

impl OnPayload {
    /// Wrap a hook.
    pub fn new(
        hook: impl Fn(JsonValue, Model) -> BoxedFuture<'static, Option<JsonValue>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self(Arc::new(hook))
    }

    /// Invoke the hook.
    #[must_use]
    pub fn call(
        &self,
        payload: JsonValue,
        model: Model,
    ) -> BoxedFuture<'static, Option<JsonValue>> {
        self.0(payload, model)
    }
}

impl std::fmt::Debug for OnPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OnPayload(..)")
    }
}

/// The response hook, upstream's `onResponse`
/// (`packages/ai/src/types.ts:149`): invoked after an HTTP response is
/// received, before its body is consumed.
#[derive(Clone)]
pub struct OnResponse(
    Arc<dyn Fn(ProviderResponse, Model) -> BoxedFuture<'static, ()> + Send + Sync>,
);

impl OnResponse {
    /// Wrap a hook.
    pub fn new(
        hook: impl Fn(ProviderResponse, Model) -> BoxedFuture<'static, ()> + Send + Sync + 'static,
    ) -> Self {
        Self(Arc::new(hook))
    }

    /// Invoke the hook.
    #[must_use]
    pub fn call(&self, response: ProviderResponse, model: Model) -> BoxedFuture<'static, ()> {
        self.0(response, model)
    }
}

impl std::fmt::Debug for OnResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OnResponse(..)")
    }
}

/// The transport fields of the request options, grouped because Rust has no
/// interface extension and the option structs spell their base's fields out.
///
/// Upstream's `fetch`, `signal`, `onPayload`, and `onResponse` of
/// `ProviderRequestOptions` (`packages/ai/src/types.ts:124`). Never
/// serialized: request plumbing.
#[derive(Clone, Debug, Default)]
pub struct TransportOptions {
    /// The HTTP client for provider requests, upstream's `fetch`
    /// (`packages/ai/src/types.ts:115`). `None` uses the process default,
    /// the reqwest 0.12 + rustls client the stack decision pins behind the
    /// `HttpClient` seam; adapters that cannot take a custom client reject
    /// it instead of bypassing it. This does not affect WebSocket transports.
    pub http_client: Option<Arc<dyn HttpClient>>,
    /// Cancellation for the request and its streamed body, upstream's
    /// `signal: AbortSignal`.
    pub signal: Option<CancellationToken>,
    /// The request-payload hook, upstream's `onPayload`.
    pub on_payload: Option<OnPayload>,
    /// The response hook, upstream's `onResponse`.
    pub on_response: Option<OnResponse>,
}

impl TransportOptions {
    /// The client to send on: the injected one, else the process default.
    #[must_use]
    pub fn client(&self) -> Arc<dyn HttpClient> {
        self.http_client
            .clone()
            .unwrap_or_else(crate::http::default_http_client)
    }

    /// The operation-local cancellation token: the caller's token when
    /// supplied, a fresh uncancelled one otherwise.
    #[must_use]
    pub fn signal(&self) -> CancellationToken {
        crate::utils::abort::operation_signal(self.signal.as_ref())
    }
}

/// Authentication, environment, and lifecycle options shared by provider
/// requests, upstream's `ProviderRequestOptions<TModel>`.
///
/// The generic parameter's only use upstream is typing the `onPayload` and
/// `onResponse` callbacks; Rust passes the [`Model`] by value into the hooks.
/// The transport fields — `fetch`, `signal`, and the two callbacks — ride the
/// [`TransportOptions`] bundle. Options are request plumbing, never
/// serialized.
#[derive(Clone, Debug, Default)]
pub struct ProviderRequestOptions {
    /// The transport seam: the request's HTTP client, cancellation token,
    /// and lifecycle callbacks, upstream's `fetch`, `signal`, `onPayload`,
    /// and `onResponse`.
    pub transport_options: TransportOptions,
    /// The API key, overriding credential resolution.
    pub api_key: Option<String>,
    /// Explicit parent context for telemetry produced by this logical request.
    pub telemetry_context: Option<TelemetryHandle>,
    /// Provider-scoped environment values; these take precedence over the
    /// process environment for provider configuration such as regional
    /// settings, endpoint placeholders, and proxy variables.
    pub env: Option<ProviderEnv>,
    /// Custom HTTP headers merged with provider defaults; caller values
    /// override default headers. On AWS Bedrock these are injected via a
    /// Smithy `build`-step middleware so they are covered by SigV4 signing;
    /// reserved headers (`x-amz-*`, `authorization`, `host`) are silently
    /// ignored to preserve SigV4 / bearer auth. A `None` value suppresses a
    /// provider/API default header with the same name.
    pub headers: Option<ProviderHeaders>,
    /// HTTP request timeout in milliseconds for providers/SDKs that support
    /// it. OpenAI and Anthropic SDK clients default to 10 minutes.
    pub timeout_ms: Option<u64>,
    /// Maximum retry attempts for providers/SDKs that support client-side
    /// retries. OpenAI and Anthropic SDK clients default to 2.
    pub max_retries: Option<u32>,
    /// Maximum delay in milliseconds to wait for a retry when the server
    /// requests a long wait. If the server's requested delay exceeds this
    /// value, the request fails immediately with an error containing the
    /// requested delay, letting higher-level retry logic handle it with user
    /// visibility. Default: 60000 (60 seconds). `0` disables the cap.
    pub max_retry_delay_ms: Option<u64>,
}

/// Streaming request options, upstream's `StreamOptions` — the
/// [`ProviderRequestOptions`] fields plus the sampling and caching extras,
/// spelled out because Rust has no interface extension.
#[derive(Clone, Debug, Default)]
pub struct StreamOptions {
    /// The transport seam: the request's HTTP client, cancellation token,
    /// and lifecycle callbacks.
    pub transport_options: TransportOptions,
    /// The API key, overriding credential resolution.
    pub api_key: Option<String>,
    /// Explicit parent context for telemetry produced by this logical request.
    pub telemetry_context: Option<TelemetryHandle>,
    /// Provider-scoped environment values; these take precedence over the
    /// process environment for provider configuration such as regional
    /// settings, endpoint placeholders, and proxy variables.
    pub env: Option<ProviderEnv>,
    /// Custom HTTP headers merged with provider defaults; caller values
    /// override default headers. A `None` value suppresses a provider/API
    /// default header with the same name.
    pub headers: Option<ProviderHeaders>,
    /// HTTP request timeout in milliseconds for providers/SDKs that support
    /// it. OpenAI and Anthropic SDK clients default to 10 minutes.
    pub timeout_ms: Option<u64>,
    /// Maximum retry attempts for providers/SDKs that support client-side
    /// retries. OpenAI and Anthropic SDK clients default to 2.
    pub max_retries: Option<u32>,
    /// Maximum delay in milliseconds to wait for a retry when the server
    /// requests a long wait. Default: 60000 (60 seconds). `0` disables the cap.
    pub max_retry_delay_ms: Option<u64>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Arbitrary sampling parameters merged into the request body as-is, after
    /// the named request fields, so keys here override them. Lets custom
    /// OpenAI-compatible servers (llama.cpp, vLLM, SGLang, ...) receive
    /// parameters pi does not model, e.g. `top_p`, `top_k`, `min_p`,
    /// `repetition_penalty`. Merged over `Model.samplingParams` per key. Only
    /// applied by OpenAI-compatible adapters (completions, responses, Azure
    /// responses); other APIs ignore it.
    pub sampling_params: Option<BTreeMap<String, JsonValue>>,
    /// Maximum output tokens.
    pub max_tokens: Option<u64>,
    /// Preferred transport for providers that support multiple transports;
    /// providers that do not support this option ignore it.
    pub transport: Option<Transport>,
    /// Prompt cache retention preference. Providers map this to their
    /// supported values. Default: `"short"`.
    pub cache_retention: Option<CacheRetention>,
    /// Optional session identifier for providers that support session-based
    /// caching. Providers can use this to enable prompt caching, request
    /// routing, or other session-aware features. Ignored by providers that
    /// don't support it.
    pub session_id: Option<String>,
    /// WebSocket connect timeout in milliseconds for providers that support
    /// WebSocket transports. This covers the connection/open handshake only;
    /// stream idleness after connection uses `timeout_ms`.
    pub websocket_connect_timeout_ms: Option<u64>,
    /// Optional metadata to include in API requests. Providers extract the
    /// fields they understand and ignore the rest. For example, Anthropic uses
    /// `user_id` for abuse tracking and rate limiting.
    pub metadata: Option<BTreeMap<String, JsonValue>>,
}

/// The adapter-facing stream options, upstream's
/// `ProviderStreamOptions = StreamOptions & Record<string, unknown>`.
///
/// The open `Record` intersection ports to this alias; an adapter that adds
/// keys extends [`StreamOptions`] with its own fields.
pub type ProviderStreamOptions = StreamOptions;

/// Options for best-effort deferred-response fetches, upstream's
/// `DeferredFetchOptions extends ProviderRequestOptions`.
#[derive(Clone, Debug, Default)]
pub struct DeferredFetchOptions {
    /// The transport seam: the request's HTTP client, cancellation token,
    /// and lifecycle callbacks.
    pub transport_options: TransportOptions,
    /// The API key, overriding credential resolution.
    pub api_key: Option<String>,
    /// Explicit parent context for telemetry produced by this logical request.
    pub telemetry_context: Option<TelemetryHandle>,
    /// Provider-scoped environment values; these take precedence over the
    /// process environment.
    pub env: Option<ProviderEnv>,
    /// Custom HTTP headers merged with provider defaults.
    pub headers: Option<ProviderHeaders>,
    /// HTTP request timeout in milliseconds for providers/SDKs that support it.
    pub timeout_ms: Option<u64>,
    /// Maximum retry attempts for providers/SDKs that support client-side
    /// retries.
    pub max_retries: Option<u32>,
    /// Maximum delay in milliseconds to wait for a retry when the server
    /// requests a long wait.
    pub max_retry_delay_ms: Option<u64>,
    /// Maximum provider long-poll duration in milliseconds. Defaults to `0`,
    /// which performs one status check.
    pub wait: Option<u64>,
}

/// Request options for best-effort deferred-response cancellation, upstream's
/// `DeferredCancelOptions` alias.
pub type DeferredCancelOptions = ProviderRequestOptions;

/// The long-poll window a deferred request asks the provider to hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeferredWindow {
    /// Fifteen minutes.
    #[serde(rename = "15m")]
    M15,
    /// One hour.
    #[serde(rename = "1h")]
    H1,
    /// Twenty-four hours.
    #[serde(rename = "24h")]
    H24,
}

/// Ask a capable provider to return a durable handle and continue the request
/// asynchronously, upstream's `deferred?: boolean | { window? }`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DeferredRequest {
    /// A deferred request with an optional window.
    Windowed {
        /// The long-poll window the provider holds the request for.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        window: Option<DeferredWindow>,
    },
    /// A deferred request with the provider's default window.
    Enabled(bool),
}

/// Provider-neutral simple-request options, upstream's
/// `SimpleStreamOptions extends StreamOptions`.
#[derive(Clone, Debug, Default)]
pub struct SimpleStreamOptions {
    /// The transport seam: the request's HTTP client, cancellation token,
    /// and lifecycle callbacks.
    pub transport_options: TransportOptions,
    /// The API key, overriding credential resolution.
    pub api_key: Option<String>,
    /// Explicit parent context for telemetry produced by this logical request.
    pub telemetry_context: Option<TelemetryHandle>,
    /// Provider-scoped environment values; these take precedence over the
    /// process environment.
    pub env: Option<ProviderEnv>,
    /// Custom HTTP headers merged with provider defaults.
    pub headers: Option<ProviderHeaders>,
    /// HTTP request timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Maximum retry attempts for client-side retries.
    pub max_retries: Option<u32>,
    /// Maximum delay in milliseconds to wait for a retry when the server
    /// requests a long wait.
    pub max_retry_delay_ms: Option<u64>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Arbitrary sampling parameters merged into the request body as-is.
    pub sampling_params: Option<BTreeMap<String, JsonValue>>,
    /// Maximum output tokens.
    pub max_tokens: Option<u64>,
    /// Preferred transport for providers that support multiple transports.
    pub transport: Option<Transport>,
    /// Prompt cache retention preference. Default: `"short"`.
    pub cache_retention: Option<CacheRetention>,
    /// Optional session identifier for providers that support session-based
    /// caching.
    pub session_id: Option<String>,
    /// WebSocket connect timeout in milliseconds.
    pub websocket_connect_timeout_ms: Option<u64>,
    /// Optional metadata to include in API requests.
    pub metadata: Option<BTreeMap<String, JsonValue>>,
    /// Provider-neutral tool selection for simple requests. When omitted,
    /// adapters use provider-specific behavior.
    pub tool_choice: Option<ToolChoice>,
    /// The pi thinking level for this request.
    pub reasoning: Option<ThinkingLevel>,
    /// Ask a capable provider to return a durable handle and continue the
    /// request asynchronously.
    pub deferred: Option<DeferredRequest>,
    /// Custom token budgets for thinking levels, token-based providers only.
    pub thinking_budgets: Option<ThinkingBudgets>,
}

/// Options for image-generation requests, upstream's `ImagesOptions extends
/// ProviderRequestOptions<ImagesModel<ImagesApi>>`.
#[derive(Clone, Debug, Default)]
pub struct ImagesOptions {
    /// The API key, overriding credential resolution.
    pub api_key: Option<String>,
    /// Explicit parent context for telemetry produced by this logical request.
    pub telemetry_context: Option<TelemetryHandle>,
    /// Provider-scoped environment values; these take precedence over the
    /// process environment.
    pub env: Option<ProviderEnv>,
    /// Custom HTTP headers merged with provider defaults.
    pub headers: Option<ProviderHeaders>,
    /// HTTP request timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Maximum retry attempts for client-side retries.
    pub max_retries: Option<u32>,
    /// Maximum delay in milliseconds to wait for a retry when the server
    /// requests a long wait.
    pub max_retry_delay_ms: Option<u64>,
    /// Optional metadata to include in API requests. Providers extract the
    /// fields they understand and ignore the rest.
    pub metadata: Option<BTreeMap<String, JsonValue>>,
}

/// The adapter-facing image options, upstream's
/// `ProviderImagesOptions = ImagesOptions & Record<string, unknown>`.
pub type ProviderImagesOptions = ImagesOptions;

/// A model Anthropic accepts in `fallbacks` for server-side refusal fallback,
/// with local pricing metadata for returned fallback responses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicAllowedFallbackModel {
    /// The fallback provider id.
    pub provider: ProviderId,
    /// The fallback model id.
    pub model: String,
    /// Local pricing metadata for returned fallback responses.
    pub cost: ModelCost,
}

/// A durable handle a provider returned for a deferred response: the request
/// continues asynchronously and later turns into an assistant message.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeferredHandle {
    /// The provider that issued the handle.
    pub provider: String,
    /// The model id the deferred request ran against.
    pub model_id: String,
    /// The wire API that issued the handle.
    pub api: String,
    /// Provider token, such as a response id or batch id plus row id.
    pub id: String,
    /// Unix epoch milliseconds when the handle expires.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
    /// Milliseconds the provider asks callers to wait before the first poll.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub poll_after_ms: Option<u64>,
    /// Provider conversion data required to reconstruct the final assistant
    /// message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<JsonValue>,
}

/// A text block. The wire discriminates it with `"type": "text"`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextContent {
    /// The text content.
    pub text: String,
    /// Message metadata for OpenAI Responses (legacy id string or a
    /// `TextSignatureV1` JSON), serialized as `textSignature`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_signature: Option<String>,
}

/// A thinking block. The wire discriminates it with `"type": "thinking"`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingContent {
    /// The thinking text.
    pub thinking: String,
    /// Provider-specific opaque or serialized reasoning replay data, serialized
    /// as `thinkingSignature`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
    /// When true, the thinking content was redacted by safety filters. The
    /// opaque encrypted payload is stored in `thinkingSignature` so it can be
    /// passed back to the API for multi-turn continuity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redacted: Option<bool>,
}

/// An image block: base64-encoded data plus its MIME type. The wire
/// discriminates it with `"type": "image"`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    /// Base64-encoded image data, serialized as `data`.
    pub data: String,
    /// The MIME type, e.g. `image/jpeg` or `image/png`, serialized as
    /// `mimeType`.
    pub mime_type: String,
}

/// A tool call block. The wire discriminates it with `"type": "toolCall"`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    /// The tool call id.
    pub id: String,
    /// The tool name.
    pub name: String,
    /// The parsed arguments object.
    pub arguments: serde_json::Map<String, JsonValue>,
    /// Google-specific opaque signature for reusing thought context, serialized
    /// as `thoughtSignature`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    /// OpenAI Responses namespace for calls to dynamically loaded or
    /// namespaced tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

/// Token and cost accounting for one assistant response, upstream's `Usage`.
/// Every number here is provider-reported.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    /// Prompt tokens, including cache reads and writes.
    pub input: u64,
    /// Completion tokens, including reasoning tokens when reported.
    pub output: u64,
    /// Prompt tokens served from cache.
    pub cache_read: u64,
    /// Prompt tokens written to cache.
    pub cache_write: u64,
    /// Subset of `cacheWrite` written with 1h retention. Only Anthropic
    /// reports this split, serialized as `cacheWrite1h`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_1h: Option<u64>,
    /// Reasoning/thinking tokens, when the provider reports them. This is a
    /// subset of `output`: `output` already includes these tokens. Set to a
    /// number (possibly 0) by providers that expose a reasoning breakdown;
    /// left absent by providers that don't.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
    /// The provider-reported total token count, serialized as `totalTokens`.
    pub total_tokens: u64,
    /// The provider-priced cost breakdown.
    pub cost: UsageCost,
}

/// The cost fields of a [`Usage`], upstream's inline `cost` object. All
/// values are provider-priced dollars.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageCost {
    /// Cost attributed to input tokens.
    pub input: f64,
    /// Cost attributed to output tokens.
    pub output: f64,
    /// Cost attributed to cache reads.
    pub cache_read: f64,
    /// Cost attributed to cache writes.
    pub cache_write: f64,
    /// Total request cost.
    pub total: f64,
}

/// Why an assistant message stopped, upstream's `StopReason`. The `"pending"`
/// state is the stream's pre-settlement value; terminal events carry one of
/// the other five.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// The stream is still running.
    #[serde(rename = "pending")]
    Pending,
    /// The model finished its turn.
    #[serde(rename = "stop")]
    Stop,
    /// The model hit a token limit.
    #[serde(rename = "length")]
    Length,
    /// The model called a tool.
    #[serde(rename = "toolUse")]
    ToolUse,
    /// The request failed.
    #[serde(rename = "error")]
    Error,
    /// The request was aborted.
    #[serde(rename = "aborted")]
    Aborted,
    /// The provider deferred the request and returned a durable handle.
    #[serde(rename = "deferred")]
    Deferred,
}

/// Why an image-generation result stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImagesStopReason {
    /// Generation finished.
    #[serde(rename = "stop")]
    Stop,
    /// Generation failed.
    #[serde(rename = "error")]
    Error,
    /// Generation was aborted.
    #[serde(rename = "aborted")]
    Aborted,
}

/// A user message, upstream's `UserMessage`. The wire discriminates it with
/// `"role": "user"`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserMessage {
    /// The message content: a plain string or content blocks.
    pub content: UserContent,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

/// User message content: a plain string or an array of text/image blocks,
/// upstream's `string | (TextContent | ImageContent)[]`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    /// The plain-string form.
    Text(String),
    /// The block-array form.
    Blocks(Vec<UserBlock>),
}

/// One block of [`UserContent::Blocks`]: text or an image.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum UserBlock {
    /// A text block, wire `"type": "text"`.
    #[serde(rename = "text")]
    Text(TextContent),
    /// An image block, wire `"type": "image"`.
    Image(ImageContent),
}

/// An assistant message, upstream's `AssistantMessage`. The wire discriminates
/// it with `"role": "assistant"`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    /// The message content: text, thinking, and tool-call blocks.
    pub content: Vec<AssistantBlock>,
    /// The wire API that produced this message.
    pub api: Api,
    /// The provider that produced this message.
    pub provider: ProviderId,
    /// The requested model id.
    pub model: String,
    /// Concrete `chunk.model` when different from the requested `model`
    /// (e.g. OpenRouter `auto` -> `anthropic/...`), serialized as
    /// `responseModel`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_model: Option<String>,
    /// Provider-specific response/message identifier when the upstream API
    /// exposes one, serialized as `responseId`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Exact provider-native effort level used for this response, serialized
    /// as `providerThinkingLevel`. Absent for legacy or unmanaged responses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_thinking_level: Option<String>,
    /// Redacted provider/runtime diagnostics for failures and recoveries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Vec<AssistantMessageDiagnostic>>,
    /// Provider-reported usage.
    pub usage: Usage,
    /// Why the message stopped.
    pub stop_reason: StopReason,
    /// The provider's deferred handle when the request continues asynchronously.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deferred: Option<DeferredHandle>,
    /// The failure description when `stopReason` is `"error"`, serialized as
    /// `errorMessage`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// The provider's own stop-reason string, serialized as `rawStopReason`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_stop_reason: Option<String>,
    /// Provider indication of whether the model explicitly ended its turn,
    /// serialized as `endTurn`. Preserved for debugging and does not currently
    /// affect agent control flow.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_turn: Option<bool>,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

/// One block of [`AssistantMessage::content`]: text, thinking, or a tool call.
/// The wire discriminates with `"type"`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum AssistantBlock {
    /// A text block, wire `"type": "text"`.
    Text(TextContent),
    /// A thinking block, wire `"type": "thinking"`.
    Thinking(ThinkingContent),
    /// A tool call, wire `"type": "toolCall"`.
    ToolCall(ToolCall),
}

/// One block of a tool result's content: text or an image.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ToolResultBlock {
    /// A text block, wire `"type": "text"`.
    Text(TextContent),
    /// An image block, wire `"type": "image"`.
    Image(ImageContent),
}

/// A tool-result message, upstream's `ToolResultMessage`. The wire
/// discriminates it with `"role": "toolResult"`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultMessage {
    /// The tool call this result answers, serialized as `toolCallId`.
    pub tool_call_id: String,
    /// The tool name, serialized as `toolName`.
    pub tool_name: String,
    /// The result content: text and images.
    pub content: Vec<ToolResultBlock>,
    /// Structured details from the tool execution. Upstream types this
    /// generically (`details?: TDetails`); on the wire it is JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<JsonValue>,
    /// Usage from the tool execution itself, if available. Not part of main
    /// LLM context accounting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Names from `Context.tools` that became available after this result,
    /// serialized as `addedToolNames`. Providers with native deferred tool
    /// loading use this as the load point; other providers ignore it and use
    /// `Context.tools` normally.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub added_tool_names: Option<Vec<String>>,
    /// Whether the tool execution failed.
    #[serde(rename = "isError")]
    pub is_error: bool,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

/// One message of a [`Context`]: user, assistant, or tool result. The wire
/// discriminates by the `"role"` field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role")]
#[allow(
    clippy::large_enum_variant,
    reason = "the message model mirrors upstream: an assistant message carries the full usage and accounting while a user message carries text"
)]
pub enum Message {
    /// A user message, wire `"role": "user"`.
    #[serde(rename = "user")]
    User(UserMessage),
    /// An assistant message, wire `"role": "assistant"`.
    #[serde(rename = "assistant")]
    Assistant(AssistantMessage),
    /// A tool-result message, wire `"role": "toolResult"`.
    #[serde(rename = "toolResult")]
    ToolResult(ToolResultMessage),
}

/// Redacted diagnostic error details, upstream's `DiagnosticErrorInfo`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiagnosticErrorInfo {
    /// The error classification, when the provider/runtime supplied one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The human-readable failure description.
    pub message: String,
    /// The stack trace, when captured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
    /// The provider's error code, string or number.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<serde_json::Number>,
}

/// Redacted provider/runtime diagnostics attached to assistant messages for
/// failures and recoveries, upstream's `AssistantMessageDiagnostic`. The
/// helpers around it port with the utils child.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessageDiagnostic {
    /// The diagnostic classification, serialized as `type`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
    /// The underlying error details, when there were any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<DiagnosticErrorInfo>,
    /// Additional redacted details.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<BTreeMap<String, JsonValue>>,
}

/// An image-generation conversation: input blocks only, upstream's
/// `ImagesContext`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImagesContext {
    /// The image-request input blocks.
    pub input: Vec<ImagesInputContent>,
}

/// Input block of an image-generation context.
pub type ImagesInputContent = ImagesBlock;

/// Output block of an image-generation result.
pub type ImagesOutputContent = ImagesBlock;

/// One block of an image-generation input or output: text or an image.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ImagesBlock {
    /// A text block, wire `"type": "text"`.
    Text(TextContent),
    /// An image block, wire `"type": "image"`.
    Image(ImageContent),
}

/// The result of one image-generation run, upstream's `AssistantImages`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantImages {
    /// The image API that produced the output.
    pub api: ImagesApi,
    /// The image provider that produced the output.
    pub provider: ImagesProviderId,
    /// The requested image model id.
    pub model: String,
    /// The generated blocks.
    pub output: Vec<ImagesOutputContent>,
    /// Provider-specific response identifier when the upstream API exposes
    /// one, serialized as `responseId`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Provider-reported usage when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Why generation stopped.
    pub stop_reason: ImagesStopReason,
    /// The failure description when `stopReason` is `"error"`, serialized as
    /// `errorMessage`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

/// OpenAI grammar variants for constrained sampling, upstream's
/// `GrammarFormat`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GrammarFormat {
    /// OpenAI's Lark grammar format.
    #[serde(rename = "openai_lark")]
    OpenaiLark,
    /// OpenAI's regex grammar format.
    #[serde(rename = "openai_regex")]
    OpenaiRegex,
}

/// Provider-specific encodings of a tool's intended language, upstream's
/// `GrammarVariants = Partial<Record<GrammarFormat, string>>`.
pub type GrammarVariants = BTreeMap<GrammarFormat, String>;

/// How strictly the provider must enforce a tool's schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Strictness {
    /// Prefer strict JSON-schema constrained sampling.
    #[serde(rename = "prefer")]
    Prefer,
    /// Require strict JSON-schema constrained sampling.
    #[serde(rename = "require")]
    Require,
}

/// Optional provider-side constrained sampling configs for a tool, upstream's
/// `ConstrainedSamplingConfig`.
///
/// The `json_schema` value roughly maps to the concept of `strict` in APIs
/// which is implemented as json-schema constrained sampling; grammar variants
/// let callers provide provider-specific encodings of the same intended
/// language.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConstrainedSamplingConfig {
    /// JSON-schema constrained sampling with a strictness preference.
    #[serde(rename = "json_schema")]
    JsonSchema {
        /// How strictly the schema binds.
        strict: Strictness,
    },
    /// Grammar variants for provider-specific encodings.
    #[serde(rename = "grammar")]
    Grammar {
        /// The provider-specific encodings of the same intended language.
        variants: GrammarVariants,
    },
}

/// A tool's constrained-sampling setting, upstream's
/// `constrainedSampling?: false | ConstrainedSamplingConfig`. The wire value
/// is either the config object or `false`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConstrainedSamplingSetting {
    /// A constrained sampling configuration.
    Config(ConstrainedSamplingConfig),
    /// The wire's `false`: no constrained sampling.
    Disabled(bool),
}

/// One callable tool, upstream's `Tool<TParameters extends TSchema>`. The
/// typebox schema becomes its JSON-Schema document; the schema representation
/// decision is recorded on the child ticket.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    /// The tool name.
    pub name: String,
    /// The tool description.
    pub description: String,
    /// The tool-parameter JSON Schema document (typebox's output on the wire).
    pub parameters: JsonValue,
    /// Optional provider-side constrained sampling configuration; the wire's
    /// `false` disables it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constrained_sampling: Option<ConstrainedSamplingSetting>,
}

/// The conversation a request runs against, upstream's `Context`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Context {
    /// The system prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// The conversation so far.
    pub messages: Vec<Message>,
    /// The tools available to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

/// One event of the assistant-message stream protocol, upstream's
/// `AssistantMessageEvent`.
///
/// Successful streams emit `start` before partial updates and terminate with
/// `done`. A stream may terminate directly with `error` when request setup
/// fails before generation starts; after `start`, failures also terminate with
/// `error`. Updates and `done` must never appear before `start`.
///
/// `partial` is the shared live response-so-far helper, not an event-time
/// snapshot. Text and thinking blocks are empty when their `*_start` event is
/// emitted and grow only through their corresponding `*_delta` events until
/// the authoritative `*_end`. Redacted thinking may be complete at start and
/// emit no deltas. Tool-call arguments at `toolcall_start` are
/// provider-specific; `toolcall_delta` carries subsequent JSON updates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AssistantMessageEvent {
    /// The stream opened; carries the live accumulator.
    #[serde(rename = "start")]
    Start {
        /// The shared live response-so-far accumulator.
        partial: AssistantMessage,
    },
    /// A text block opened, wire `"text_start"`.
    #[serde(rename = "text_start", rename_all = "camelCase")]
    TextStart {
        /// The block's index in `content`.
        content_index: u64,
        /// The shared live accumulator.
        partial: AssistantMessage,
    },
    /// Text grew, wire `"text_delta"`.
    #[serde(rename = "text_delta", rename_all = "camelCase")]
    TextDelta {
        /// The block's index in `content`.
        content_index: u64,
        /// The appended text.
        delta: String,
        /// The shared live accumulator.
        partial: AssistantMessage,
    },
    /// A text block closed authoritatively, wire `"text_end"`.
    #[serde(rename = "text_end", rename_all = "camelCase")]
    TextEnd {
        /// The block's index in `content`.
        content_index: u64,
        /// The authoritative full text.
        content: String,
        /// The shared live accumulator.
        partial: AssistantMessage,
    },
    /// A thinking block opened, wire `"thinking_start"`.
    #[serde(rename = "thinking_start", rename_all = "camelCase")]
    ThinkingStart {
        /// The block's index in `content`.
        content_index: u64,
        /// The shared live accumulator.
        partial: AssistantMessage,
    },
    /// Thinking grew, wire `"thinking_delta"`.
    #[serde(rename = "thinking_delta", rename_all = "camelCase")]
    ThinkingDelta {
        /// The block's index in `content`.
        content_index: u64,
        /// The appended thinking text.
        delta: String,
        /// The shared live accumulator.
        partial: AssistantMessage,
    },
    /// A thinking block closed authoritatively, wire `"thinking_end"`.
    #[serde(rename = "thinking_end", rename_all = "camelCase")]
    ThinkingEnd {
        /// The block's index in `content`.
        content_index: u64,
        /// The authoritative full thinking text.
        content: String,
        /// The shared live accumulator.
        partial: AssistantMessage,
    },
    /// A tool call opened, wire `"toolcall_start"`.
    #[serde(rename = "toolcall_start", rename_all = "camelCase")]
    ToolcallStart {
        /// The block's index in `content`.
        content_index: u64,
        /// The shared live accumulator.
        partial: AssistantMessage,
    },
    /// Tool-call arguments grew, wire `"toolcall_delta"`.
    #[serde(rename = "toolcall_delta", rename_all = "camelCase")]
    ToolcallDelta {
        /// The block's index in `content`.
        content_index: u64,
        /// The appended JSON update.
        delta: String,
        /// The shared live accumulator.
        partial: AssistantMessage,
    },
    /// A tool call closed authoritatively, wire `"toolcall_end"`.
    #[serde(rename = "toolcall_end", rename_all = "camelCase")]
    ToolcallEnd {
        /// The block's index in `content`.
        content_index: u64,
        /// The authoritative tool call.
        tool_call: ToolCall,
        /// The shared live accumulator.
        partial: AssistantMessage,
    },
    /// The stream finished successfully. `reason` is always one of
    /// `stop`, `length`, `toolUse`, or `deferred` — upstream's
    /// `Extract<StopReason, ...>` subset.
    #[serde(rename = "done")]
    Done {
        /// Why the stream finished.
        reason: StopReason,
        /// The final assistant message.
        message: AssistantMessage,
    },
    /// The stream failed or was aborted; `reason` is always `aborted` or
    /// `error` — the other `Extract<StopReason, ...>` subset — and carries
    /// the failing assistant message.
    #[serde(rename = "error")]
    Error {
        /// The failure kind, always `aborted` or `error`.
        reason: StopReason,
        /// The failing assistant message, with `errorMessage` set.
        error: AssistantMessage,
    },
}

/// Which field carries the max-token cap on OpenAI-compatible endpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaxTokensField {
    /// The newer `max_completion_tokens` field.
    #[serde(rename = "max_completion_tokens")]
    MaxCompletionTokens,
    /// The legacy `max_tokens` field.
    #[serde(rename = "max_tokens")]
    MaxTokens,
}

/// The thinking-parameter convention an OpenAI-compatible endpoint expects,
/// upstream's `thinkingFormat` union.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinkingFormat {
    /// `reasoning_effort`.
    #[serde(rename = "openai")]
    Openai,
    /// `reasoning: { effort }`.
    #[serde(rename = "openrouter")]
    Openrouter,
    /// `thinking: { type }` plus `reasoning_effort` when supported.
    #[serde(rename = "deepseek")]
    Deepseek,
    /// `reasoning: { enabled }` plus `reasoning_effort` when supported.
    #[serde(rename = "together")]
    Together,
    /// Configurable `chat_template_args` plus `reasoning_effort` when
    /// supported.
    #[serde(rename = "baseten")]
    Baseten,
    /// `thinking: { type }`.
    #[serde(rename = "zai")]
    Zai,
    /// Top-level `enable_thinking: boolean`.
    #[serde(rename = "qwen")]
    Qwen,
    /// `chat_template_kwargs.enable_thinking` and `preserve_thinking`.
    #[serde(rename = "chat-template")]
    ChatTemplate,
    /// The qwen chat-template shape with preserve semantics.
    #[serde(rename = "qwen-chat-template")]
    QwenChatTemplate,
    /// Top-level `thinking: string`.
    #[serde(rename = "string-thinking")]
    StringThinking,
    /// `reasoning: { effort }` only when the mapped effort is non-null.
    #[serde(rename = "ant-ling")]
    AntLing,
}

/// Cache-control convention for prompt caching, upstream's
/// `cacheControlFormat`.
///
/// `"anthropic"` applies Anthropic-style `cache_control` markers to the
/// system prompt, last tool definition, and last user, assistant, or
/// tool-result text content.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheControlFormat {
    /// Anthropic-style `cache_control` markers.
    #[serde(rename = "anthropic")]
    Anthropic,
}

/// Provider-specific deferred tool serialization mode, upstream's
/// `deferredToolsMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeferredToolsMode {
    /// Kimi's deferred tool serialization.
    #[serde(rename = "kimi")]
    Kimi,
}

/// Compatibility settings for OpenAI-compatible completions APIs, the union
/// of upstream's `OpenAICompletionsCompat` and the other per-API compat
/// interfaces.
///
/// One struct: the wire object carries no discriminator, and the field sets
/// never conflict. Each field's doc names the APIs that read it. Use this to
/// override URL-based auto-detection for custom providers.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCompat {
    /// Whether the provider supports the `store` field (openai-completions).
    /// Default: auto-detected from URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_store: Option<bool>,
    /// Whether the provider supports the `developer` role (vs `system`)
    /// (openai-completions, openai-responses family). Default:
    /// auto-detected from URL for completions, `true` for responses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_developer_role: Option<bool>,
    /// Whether the provider supports `reasoning_effort` (openai-completions).
    /// Default: auto-detected from URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_reasoning_effort: Option<bool>,
    /// Whether the provider supports `stream_options: { include_usage: true }`
    /// for token usage in streaming responses (openai-completions).
    /// Default: true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_usage_in_streaming: Option<bool>,
    /// Whether streamed responses include `finish_reason`. When false, pi
    /// infers `stop` or `toolUse` when the stream ends (openai-completions).
    /// Default: true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_finish_reason: Option<bool>,
    /// Which field to use for max tokens (openai-completions). Default:
    /// auto-detected from URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens_field: Option<MaxTokensField>,
    /// Whether tool results require the `name` field (openai-completions).
    /// Default: auto-detected from URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_tool_result_name: Option<bool>,
    /// Whether a user message after tool results requires an assistant message
    /// in between (openai-completions). Default: auto-detected from URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_assistant_after_tool_result: Option<bool>,
    /// Whether thinking blocks must be converted to text blocks with
    /// `<thinking>` delimiters (openai-completions). Default: auto-detected
    /// from URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_thinking_as_text: Option<bool>,
    /// Whether all replayed assistant messages must include an empty
    /// `reasoning_content` field when reasoning is enabled
    /// (openai-completions). Default: auto-detected from URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_reasoning_content_on_assistant_messages: Option<bool>,
    /// Format for reasoning/thinking parameter (openai-completions), serialized as
    /// `thinkingFormat`:
    /// `"openai"` uses `reasoning_effort`, `"openrouter"` uses `reasoning: { effort }`,
    /// `"deepseek"` uses `thinking: { type }` plus `reasoning_effort` when supported,
    /// `"together"` uses `reasoning: { enabled }` plus `reasoning_effort` when
    /// supported, `"baseten"` uses configurable `chat_template_args` plus
    /// `reasoning_effort` when supported, `"zai"` uses `thinking: { type }`,
    /// `"qwen"` uses top-level `enable_thinking: boolean`,
    /// `"qwen-chat-template"` uses `chat_template_kwargs.enable_thinking` and
    /// `preserve_thinking`, `"chat-template"` uses configurable
    /// `chat_template_kwargs`, `"string-thinking"` uses top-level
    /// `thinking: string`, and `"ant-ling"` uses `reasoning: { effort }` only
    /// when the mapped effort is non-null. Default: `"openai"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_format: Option<ThinkingFormat>,
    /// Kwargs to send as `chat_template_kwargs` when `thinkingFormat` is
    /// `"chat-template"` (openai-completions). Use `{ "$var":
    /// "thinking.enabled" }`, `{ "$var": "thinking.effort" }`, or
    /// `{ "$var": "thinking.budget" }` for pi-controlled thinking values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<BTreeMap<String, ChatTemplateKwargValue>>,
    /// Arguments to send as `chat_template_args` when `thinkingFormat` is
    /// `"baseten"` (openai-completions). Use `{ "$var": "thinking.enabled" }`,
    /// `{ "$var": "thinking.effort" }`, or `{ "$var": "thinking.budget" }` for
    /// pi-controlled thinking values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_args: Option<BTreeMap<String, ChatTemplateKwargValue>>,
    /// OpenRouter-compatible routing preferences sent as the `provider`
    /// request field (openai-completions), serialized as `openRouterRouting`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_router_routing: Option<OpenRouterRouting>,
    /// Vercel AI Gateway routing preferences (openai-completions). Only used
    /// when baseUrl points to Vercel AI Gateway, serialized as
    /// `vercelGatewayRouting`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vercel_gateway_routing: Option<VercelGatewayRouting>,
    /// Whether z.ai supports top-level `tool_stream: true` for streaming tool
    /// call deltas (openai-completions). Default: false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zai_tool_stream: Option<bool>,
    /// Top-level request field used to cap reasoning tokens from
    /// `thinkingBudgets` (openai-completions). Reasoning and the answer share
    /// `max_tokens` on these endpoints, so without a budget a reasoning-heavy
    /// turn can consume the whole response and emit no answer.
    /// `"thinking_token_budget"` is vLLM, `"thinking_budget"` is
    /// Qwen/DashScope/SGLang, `"thinking_budget_tokens"` is llama.cpp. Off by
    /// default; not set on the generated catalog.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_token_budget_field: Option<ThinkingTokenBudgetField>,
    /// Alias for `thinkingTokenBudgetField: "thinking_token_budget"` (vLLM).
    /// Prefer `thinkingTokenBudgetField`. Default: false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_thinking_token_budget: Option<bool>,
    /// Whether the provider supports OpenAI custom tools with Lark/regex
    /// grammar formats (openai-completions, OpenAI responses). When false,
    /// grammar-constrained tools fall back to normal function tools.
    /// Default: false; the generated model catalog enables it for capable
    /// models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_openai_grammar_tools: Option<bool>,
    /// Whether the provider supports the `strict` field in tool definitions
    /// (openai-completions, OpenAI Responses, Bedrock). Default: true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_strict_mode: Option<bool>,
    /// Cache control convention for prompt caching (openai-completions),
    /// serialized as `cacheControlFormat`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control_format: Option<CacheControlFormat>,
    /// Whether to send session-affinity data from `options.sessionId`
    /// (openai-completions, Anthropic Messages). Default: true for OpenRouter
    /// endpoints, false otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub send_session_affinity_headers: Option<bool>,
    /// Provider-specific deferred tool serialization mode
    /// (openai-completions), serialized as `deferredToolsMode`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deferred_tools_mode: Option<DeferredToolsMode>,
    /// Session-affinity header format (openai-completions, OpenAI Responses,
    /// and the Anthropic Messages `"openrouter"`-only form).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_affinity_format: Option<SessionAffinityFormat>,
    /// Whether the provider supports long prompt cache retention
    /// (openai-completions: `prompt_cache_retention: "24h"` or Anthropic-style
    /// `cache_control.ttl: "1h"` depending on format; OpenAI Responses:
    /// `prompt_cache_options.ttl: "30m"` on GPT-5.6+ and
    /// `prompt_cache_retention: "24h"` on earlier models; Anthropic Messages:
    /// `cache_control.ttl: "1h"`). Default: true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
    /// Whether the model supports message-anchored `additional_tools` input
    /// items (OpenAI Responses). Default: false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_additional_tools: Option<bool>,
    /// Whether the model supports client-executed tool search for deferred
    /// tools (OpenAI Responses). Default: false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_tool_search: Option<bool>,
    /// Whether the model accepts `prompt_cache_options` (OpenAI GPT-5.6+
    /// prompt caching). Older OpenAI models reject the parameter.
    /// Default: false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_explicit_prompt_cache_mode: Option<bool>,
    /// Whether the provider accepts the `max_output_tokens` parameter (OpenAI
    /// Responses). Some Codex-protocol gateways reject it. Default: true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_max_output_tokens: Option<bool>,
    /// Whether the provider accepts per-tool `eager_input_streaming`
    /// (Anthropic Messages), serialized as `supportsEagerToolInputStreaming`.
    /// When false, the Anthropic provider omits
    /// `tools[].eager_input_streaming` and sends the legacy
    /// `fine-grained-tool-streaming-2025-05-14` beta header for tool-enabled
    /// requests. Default: true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_eager_tool_input_streaming: Option<bool>,
    /// Whether the provider supports Anthropic-style `cache_control` markers
    /// on tool definitions (Anthropic Messages). When false, `cache_control`
    /// is omitted from tool params. Some Anthropic-compatible providers (e.g.,
    /// Fireworks) do not support this field on tools and may reject or ignore
    /// it. Default: true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_cache_control_on_tools: Option<bool>,
    /// Whether the model accepts the Anthropic `temperature` request field
    /// (Anthropic Messages). Claude Opus 4.7+ rejects non-default temperature
    /// values. Default: true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_temperature: Option<bool>,
    /// Whether to force adaptive thinking (`thinking.type: "adaptive"` plus
    /// `output_config.effort`) regardless of the model id (Anthropic
    /// Messages), serialized as `forceAdaptiveThinking`. Built-in models that
    /// require adaptive thinking set this in generated metadata. Custom
    /// Anthropic-compatible providers can set this to `true` for any model
    /// whose upstream requires the adaptive format. Set to `false` to opt out
    /// on overridden built-in models. Default: false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force_adaptive_thinking: Option<bool>,
    /// Whether to replay empty thinking signatures as `signature: ""` instead
    /// of converting thinking to text (Anthropic Messages), serialized as
    /// `allowEmptySignature`. Default: false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_empty_signature: Option<bool>,
    /// Whether the provider supports Anthropic strict tool schemas (Anthropic
    /// Messages), serialized as `supportsStrictTools`. Default: false;
    /// generated Anthropic models enable it explicitly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_strict_tools: Option<bool>,
    /// Whether the exact model transport supports effort-only system messages
    /// and thinking binding controls (Anthropic Messages), serialized as
    /// `supportsMidConvoEffort`. Default: false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_mid_convo_effort: Option<bool>,
    /// Models Anthropic accepts in `fallbacks` for server-side refusal
    /// fallback, with local pricing metadata for returned fallback responses,
    /// serialized as `allowedFallbackModels`. When absent or empty, callers
    /// must omit `fallbacks`; Anthropic rejects the field for models with no
    /// permitted fallback targets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_fallback_models: Option<Vec<AnthropicAllowedFallbackModel>>,
    /// Whether the provider supports deferred tools loaded by
    /// `tool_reference` blocks in tool results (Anthropic Messages),
    /// serialized as `supportsToolReferences`. Default: true for first-party
    /// Anthropic models except Haiku and models older than Claude 4.5; false
    /// for other providers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_tool_references: Option<bool>,
    /// Whether the model supports Bedrock strict tool schemas
    /// (bedrock-converse-stream). Default: false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_bedrock_strict_mode: Option<bool>,
}

/// OpenRouter provider routing preferences, sent as the `provider` field in
/// the OpenRouter API request body. Field names are the wire's `snake_case`
/// spellings.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterRouting {
    /// Whether to allow backup providers to serve requests (`allow_fallbacks`).
    /// Default: true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    /// Whether to filter providers to only those that support all parameters
    /// in the request (`require_parameters`). Default: false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,
    /// Data collection setting (`data_collection`). `"allow"` (default): allow
    /// providers that may store/train on data. `"deny"`: only use providers
    /// that don't collect user data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<DataCollection>,
    /// Whether to restrict routing to only ZDR (Zero Data Retention)
    /// endpoints (`zdr`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zdr: Option<bool>,
    /// Whether to restrict routing to only models that allow text
    /// distillation (`enforce_distillable_text`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enforce_distillable_text: Option<bool>,
    /// An ordered list of provider names/slugs to try in sequence, falling
    /// back to the next if unavailable (`order`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
    /// List of provider names/slugs to exclusively allow for this request
    /// (`only`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    /// List of provider names/slugs to skip for this request (`ignore`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,
    /// A list of quantization levels to filter providers by, e.g.
    /// `["fp16", "bf16", "fp8", "fp6", "int8", "int4", "fp4", "fp32"]`
    /// (`quantizations`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantizations: Option<Vec<String>>,
    /// Sorting strategy (`sort`): a string or an object with `by` and
    /// `partition`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort: Option<SortPreference>,
    /// Maximum price per million tokens, USD (`max_price`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_price: Option<MaxPrice>,
    /// Preferred minimum throughput in tokens/second
    /// (`preferred_min_throughput`): a number (applies to p50) or an object
    /// with percentile-specific cutoffs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_min_throughput: Option<PercentilePreference>,
    /// Preferred maximum latency in seconds (`preferred_max_latency`): a
    /// number (applies to p50) or an object with percentile-specific cutoffs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_max_latency: Option<PercentilePreference>,
}

/// Data-collection routing setting, the wire's `data_collection` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataCollection {
    /// Only use providers that don't collect user data.
    #[serde(rename = "deny")]
    Deny,
    /// Allow providers that may store/train on data (default).
    #[serde(rename = "allow")]
    Allow,
}

/// The sorting metric of an OpenRouter route sort.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SortMetric {
    /// Sort by price.
    #[serde(rename = "price")]
    Price,
    /// Sort by throughput.
    #[serde(rename = "throughput")]
    Throughput,
    /// Sort by latency.
    #[serde(rename = "latency")]
    Latency,
}

/// OpenRouter's `sort` preference: a metric name or an object with `by` and
/// `partition`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SortPreference {
    /// The sorting strategy as a string (e.g. `"price"`, `"throughput"`,
    /// `"latency"`).
    Name(String),
    /// The sorting strategy as an object.
    Detailed {
        /// The sorting metric.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        by: Option<SortMetric>,
        /// The partitioning strategy: `"model"` (default) or `"none"`
        /// (the wire's `null`); absent means the default.
        /// The double option is the semantics: the outer `None` is an absent
        /// field, `Some(None)` is the wire's `null`.
        #[allow(
            clippy::type_complexity,
            clippy::option_option,
            reason = "the double option distinguishes the wire's absent field from its explicit null; no named type reads clearer"
        )]
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_partition"
        )]
        partition: Option<Option<String>>,
    },
}

/// Reads the wire's `partition` field: absent means the default (`None` of the
/// outer option), `null` means `"none"` (`Some(None)`), a string is a
/// partitioning strategy.
#[allow(
    clippy::type_complexity,
    clippy::option_option,
    reason = "the double option distinguishes the wire's absent field from its explicit null; no named type reads clearer"
)]
fn deserialize_partition<'de, D>(deserializer: D) -> Result<Option<Option<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct PartitionVisitor;

    impl<'de> serde::de::Visitor<'de> for PartitionVisitor {
        type Value = Option<Option<String>>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a partition strategy string, null, or nothing")
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(Some(None))
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(Some(None))
        }

        fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            Option::<String>::deserialize(deserializer).map(Some)
        }
    }

    deserializer.deserialize_option(PartitionVisitor)
}

/// A price cap per million tokens (USD) or per image/audio/request: the wire
/// carries a number or a string.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PriceValue {
    /// The numeric form.
    Num(f64),
    /// The string form.
    Str(String),
}

/// OpenRouter's `max_price` object: price caps per million tokens, per image,
/// per audio unit, and per request (USD).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MaxPrice {
    /// Price per million prompt tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<PriceValue>,
    /// Price per million completion tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion: Option<PriceValue>,
    /// Price per image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<PriceValue>,
    /// Price per audio unit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<PriceValue>,
    /// Price per request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<PriceValue>,
}

/// A percentile-specific throughput or latency cutoff set: p50, p75, p90, p99.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PercentileCutoffs {
    /// Minimum/maximum at the 50th percentile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p50: Option<f64>,
    /// Minimum/maximum at the 75th percentile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p75: Option<f64>,
    /// Minimum/maximum at the 90th percentile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p90: Option<f64>,
    /// Minimum/maximum at the 99th percentile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p99: Option<f64>,
}

/// A throughput or latency preference: the plain number form (applies to p50)
/// or the percentile-object form.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PercentilePreference {
    /// The plain number form, applying to the p50 cutoff.
    Value(f64),
    /// The percentile-specific cutoffs.
    Percentiles(PercentileCutoffs),
}

/// Vercel AI Gateway routing preferences, controlling which upstream providers
/// the gateway routes requests to.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VercelGatewayRouting {
    /// List of provider slugs to exclusively use for this request, e.g.
    /// `["bedrock", "anthropic"]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    /// List of provider slugs to try in order, e.g.
    /// `["anthropic", "openai"]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
}

/// Per-million-token pricing rates, upstream's `ModelCostRates`. All values
/// are dollars per million tokens from the generated catalog.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCostRates {
    /// Input token rate.
    pub input: f64,
    /// Output token rate.
    pub output: f64,
    /// Cache-read token rate, serialized as `cacheRead`.
    pub cache_read: f64,
    /// Cache-write token rate, serialized as `cacheWrite`.
    pub cache_write: f64,
}

/// Request-wide pricing tier, upstream's `ModelCostTier`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCostTier {
    /// The tier's rates.
    #[serde(flatten)]
    pub rates: ModelCostRates,
    /// Use this tier for requests whose total input usage exceeds this token
    /// count, serialized as `inputTokensAbove`.
    pub input_tokens_above: u64,
}

/// A model's pricing, upstream's `ModelCost`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    /// The base rates.
    #[serde(flatten)]
    pub rates: ModelCostRates,
    /// Request-wide pricing tiers. The highest matching input threshold
    /// applies to the full request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiers: Option<Vec<ModelCostTier>>,
}

/// The input modality a model accepts (and an image model outputs), upstream's
/// `("text" | "image")[]` element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Modality {
    /// Text content.
    #[serde(rename = "text")]
    Text,
    /// Image content.
    #[serde(rename = "image")]
    Image,
}

/// A model in the unified model system, upstream's `Model<TApi>`.
///
/// Upstream's `TApi` type parameter selects the compat field's shape at the
/// type level; Rust holds one [`ModelCompat`] struct (the wire object carries
/// no discriminator and the field sets never conflict).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    /// The model id.
    pub id: String,
    /// The display name.
    pub name: String,
    /// The wire API this model speaks.
    pub api: Api,
    /// The provider id.
    pub provider: ProviderId,
    /// The API base URL, serialized as `baseUrl`.
    pub base_url: String,
    /// Whether the model supports reasoning.
    pub reasoning: bool,
    /// Maps pi thinking levels to provider/model-specific values, serialized
    /// as `thinkingLevelMap`. Missing keys use provider defaults; `null` marks
    /// a level as unsupported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    /// The input modalities.
    pub input: Vec<Modality>,
    /// The pricing.
    pub cost: ModelCost,
    /// The model's context window in tokens, serialized as `contextWindow`.
    pub context_window: u64,
    /// The maximum output tokens, serialized as `maxTokens`.
    pub max_tokens: u64,
    /// Default sampling parameters for this model. Per-request keys in
    /// `StreamOptions.samplingParams` override these.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<BTreeMap<String, JsonValue>>,
    /// Custom HTTP headers merged into API requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<BTreeMap<String, String>>,
    /// Compatibility overrides for OpenAI-compatible APIs. If not set,
    /// auto-detected from `baseUrl`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<ModelCompat>,
}

/// An image-generation model, upstream's `ImagesModel<TApi>`: a [`Model`]
/// without `reasoning`, `contextWindow`, `maxTokens`, or `compat`, plus the
/// output modalities.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImagesModel {
    /// The model id.
    pub id: String,
    /// The display name.
    pub name: String,
    /// The image API this model serves.
    pub api: ImagesApi,
    /// The image provider id.
    pub provider: ImagesProviderId,
    /// The API base URL, serialized as `baseUrl`.
    pub base_url: String,
    /// Maps pi thinking levels to provider/model-specific values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    /// The input modalities.
    pub input: Vec<Modality>,
    /// The pricing.
    pub cost: ModelCost,
    /// Default sampling parameters for this model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<BTreeMap<String, JsonValue>>,
    /// Custom HTTP headers merged into API requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<BTreeMap<String, String>>,
    /// The output modalities.
    pub output: Vec<Modality>,
}

/// A boxed future, the port of the crate's promise-returning contracts.
pub type BoxedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// The uniform stream contract of an API implementation module.
///
/// Every module under upstream's `src/api/` exports `stream` and
/// `streamSimple`; capable modules may also export deferred-response methods.
/// Lazy wrappers and provider factories pass these around as values. This is
/// the untyped dispatch shape; per-API option typing lives on the
/// implementation modules themselves and on `Provider.stream()` via
/// `ApiStreamOptions`.
///
/// Contract: `stream` and `streamSimple` return the
/// [`AssistantMessageEventStream`](crate::utils::event_stream::AssistantMessageEventStream) synchronously, matching upstream's
/// lazy-stream contract — a direct call may fail synchronously when request
/// auth is missing; once a stream is returned, request, model, and runtime
/// failures are encoded in that stream. Error termination must produce an
/// assistant message with stop reason `"error"` or `"aborted"` and
/// `errorMessage`, emitted via the stream protocol.
pub trait ProviderStreams: Send + Sync {
    /// Stream an assistant response for the model and context.
    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> crate::utils::event_stream::AssistantMessageEventStream;

    /// Stream a simple assistant response for the model and context.
    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> crate::utils::event_stream::AssistantMessageEventStream;

    /// Fetch a deferred response by its durable handle; only adapters that
    /// support deferred responses implement it, upstream's optional method.
    fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: Option<&DeferredFetchOptions>,
    ) -> Option<crate::utils::event_stream::AssistantMessageEventStream> {
        let _ = (model, handle, options);
        None
    }

    /// Cancel a deferred response; only adapters that support deferred
    /// responses implement it, upstream's optional method.
    ///
    /// # Errors
    /// The returned future resolves to the provider's failure.
    fn cancel_deferred<'a>(
        &'a self,
        _model: &'a Model,
        _handle: &'a DeferredHandle,
        _options: Option<&'a DeferredCancelOptions>,
    ) -> BoxedFuture<'a, Result<(), crate::utils::provider_retry::ProviderRequestError>> {
        Box::pin(async { Err(crate::utils::provider_retry::ProviderRequestError::aborted()) })
    }

    /// Whether this implementation provides `fetch_deferred`, the port of
    /// upstream's `entry.fetchDeferred !== undefined` presence check.
    #[must_use]
    fn supports_fetch_deferred(&self) -> bool {
        false
    }

    /// Whether this implementation provides `cancel_deferred`, the port of
    /// upstream's `entry.cancelDeferred !== undefined` presence check.
    #[must_use]
    fn supports_cancel_deferred(&self) -> bool {
        false
    }
}

/// The uniform contract of an image-generation API implementation module.
///
/// Every image API module under upstream's `src/api/` exports exactly
/// `generateImages`, so the module itself satisfies this interface.
pub trait ProviderImages: Send + Sync {
    /// Generate images for the input context.
    ///
    /// # Errors
    /// The boxed future resolves to the provider's failure; per-image
    /// failures inside a completed run surface through
    /// [`AssistantImages::stop_reason`] instead.
    fn generate_images<'a>(
        &'a self,
        model: &'a ImagesModel,
        context: &'a ImagesContext,
        options: Option<&'a ImagesOptions>,
    ) -> BoxedFuture<'a, Result<AssistantImages, crate::utils::provider_retry::ProviderRequestError>>;
}

/// A stream function, upstream's `StreamFunction`: the typed function shape
/// lazy wrappers and provider factories pass around as values. See
/// [`ProviderStreams`] for the stream contract.
pub type StreamFunction = Box<
    dyn Fn(
            &Model,
            &Context,
            Option<&StreamOptions>,
        ) -> crate::utils::event_stream::AssistantMessageEventStream
        + Send
        + Sync,
>;

/// An image-generation function, upstream's `ImagesFunction`. See
/// [`ProviderImages`] for the contract.
pub type ImagesFunction = Box<
    dyn for<'a> Fn(
            &'a ImagesModel,
            &'a ImagesContext,
            Option<&'a ImagesOptions>,
        ) -> BoxedFuture<
            'a,
            Result<AssistantImages, crate::utils::provider_retry::ProviderRequestError>,
        > + Send
        + Sync,
>;

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "the tests pin parse outcomes; an unexpected result panics the test by design"
)]
mod partition_tests {
    use super::deserialize_partition;

    #[test]
    fn partition_reader_distinguishes_null_from_a_value() {
        let mut null_path = serde_json::Deserializer::from_str("null");
        let none = deserialize_partition(&mut null_path).expect("null parses");
        assert_eq!(none, Some(None), "the wire's null is the none strategy");

        let mut model_path = serde_json::Deserializer::from_str("\"model\"");
        let model = deserialize_partition(&mut model_path).expect("string parses");
        assert_eq!(model, Some(Some("model".to_string())));

        let mut number_path = serde_json::Deserializer::from_str("7");
        let failed = deserialize_partition(&mut number_path)
            .err()
            .map(|error| error.to_string());
        let _ = failed.expect("a number is not a partition strategy");
    }
}
